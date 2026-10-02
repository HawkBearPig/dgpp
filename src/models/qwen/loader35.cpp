// Qwen3.5-27B loader family: FP8-native weights (the checkpoint already holds
// e4m3 payload + BF16 scales — memcpy + BF16→F32 widen, never a BF16→FP8
// re-encode), text-only, dense SwiGLU MLP, standard pre-norm residual.
#include "models/qwen/loader35.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "common/dtypes.hpp"

namespace dgpp {
namespace {

bool ends_with(const std::string& s, const std::string& suffix) {
  const size_t n = suffix.size();
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// Per-layer build progress logs, off by default (the sibling qwen loader
// prints nothing). Set DGPP_QWEN35_LOADER_VERBOSE to a non-'0' value to
// trace layer construction at boot.
bool loader_verbose() {
  static const bool v = [] {
    const char* e = std::getenv("DGPP_QWEN35_LOADER_VERBOSE");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return v;
}

// Rank-invariant reads at world > 1 (mirrors the qwen loader's rule):
// norms replicate, projections slice. The MTP draft head is BF16 and
// replicated whole (like the norms).
bool is_replicated_35(const QwenExpectedTensor& e) {
  switch (e.cls) {
    case QwenWeightClass::Norm:
      return true;
    case QwenWeightClass::Mtp:
      return true;
    case QwenWeightClass::FullAttn:
      return ends_with(e.name, "q_norm.weight") || ends_with(e.name, "k_norm.weight");
    case QwenWeightClass::Gdn:
      return ends_with(e.name, "norm.weight");
    default:
      return false;
  }
}

void widen_bf16_to_f32(const uint16_t* src, float* dst, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] = bf16_bits_to_float(src[i]);
}

}  // namespace

Qwen35LocalGeometry Qwen35LocalGeometry::from_config(const Qwen35TextConfig& cfg, int rank,
                                                     int world, LoaderHeadSharding) {
  qwen35_tp_validate_geometry(cfg, rank, world);
  if (world < 1 || rank < 0 || rank >= world)
    throw std::invalid_argument("qwen35 loader: rank/world out of range");
  Qwen35LocalGeometry g;
  g.world = world;
  g.rank = rank;
  g.local_key_heads = cfg.gdn_key_heads / world;
  g.local_value_heads = cfg.gdn_value_heads / world;
  g.local_heads = cfg.num_attention_heads / world;
  g.head_begin = g.local_heads * rank;
  if (cfg.num_key_value_heads >= world) {
    g.local_kv_heads = cfg.num_key_value_heads / world;
    g.kv_head_begin = g.local_kv_heads * rank;
  } else {
    g.local_kv_heads = 1;
    g.kv_head_begin = rank / (world / cfg.num_key_value_heads);
  }
  // The rank's query heads must belong to its kv head(s).
  const int64_t lh_total = cfg.num_attention_heads, kv_total = cfg.num_key_value_heads;
  for (int h = g.head_begin; h < g.head_begin + g.local_heads; ++h) {
    const int64_t owner = static_cast<int64_t>(h) * kv_total / lh_total;
    if (owner < g.kv_head_begin || owner >= g.kv_head_begin + g.local_kv_heads)
      throw std::invalid_argument("qwen35 loader: query/kv head sharding mismatch");
  }
  g.local_inter = cfg.intermediate_size / world;
  g.lm_vocab_begin =
      static_cast<int>(static_cast<int64_t>(cfg.vocab_size) * rank / world);
  g.lm_vocab_count =
      static_cast<int>(static_cast<int64_t>(cfg.vocab_size) * (rank + 1) / world) - g.lm_vocab_begin;
  return g;
}

struct Qwen35LoaderFamily::Builder : WeightBuilder<QwenExpectedTensor> {
  const Qwen35TextConfig& cfg;
  const Qwen35LocalGeometry& geo;
  Qwen35LayerResident& out;

  Builder(const Qwen35TextConfig& cfg_, const Qwen35LocalGeometry& geo_,
          const std::vector<QwenExpectedTensor>& table_,
          const std::unordered_map<std::string, const QwenExpectedTensor*>& by_name_,
          LayerBump& bump_, Qwen35LayerResident& out_,
          const std::unordered_map<std::string, const TensorInfo*>& tensors_,
          std::vector<DequantJob>& jobs_, std::vector<PackJob>& packs_, bool copy_)
      : WeightBuilder<QwenExpectedTensor>(table_, by_name_, bump_, tensors_, jobs_, packs_, copy_,
                                          geo_.rank, geo_.world, "qwen35 loader"),
        cfg(cfg_),
        geo(geo_),
        out(out_) {}

  bool replicated(const QwenExpectedTensor& e) const override { return is_replicated_35(e); }

  // An FP8 matrix's two checkpoint tensors: e4m3 [N, K] payload + BF16
  // [N/128, K/128] scales (the binding table registers both; the suffix is
  // part of the contract).
  struct Fp8Native {
    const TensorInfo* payload = nullptr;
    const TensorInfo* scales = nullptr;
    const QwenExpectedTensor* payload_entry = nullptr;
    const QwenExpectedTensor* scales_entry = nullptr;
    int64_t N = 0, K = 0, SN = 0, SK = 0;
  };

  Fp8Native fp8_source(const std::string& name) {
    const QwenExpectedTensor& e = expected(name);
    if (e.shape.size() != 2) fail(name + ": fp8 matrix needs 2 dims");
    const int64_t N = e.shape[0], K = e.shape[1];
    const std::string sname = name + "_scale_inv";
    const QwenExpectedTensor& se = expected(sname);
    const int64_t SN = (N + 127) / 128, SK = (K + 127) / 128;
    if (se.shape.size() != 2 || se.shape[0] != SN || se.shape[1] != SK)
      fail(sname + ": BF16 scale grid must be [N/128, K/128]");
    Fp8Native s;
    // source() needs the checkpoint map: only the copy build has it. The
    // counting build (copy=false) runs on an empty map, so it must never
    // touch sources — every dereference below is already inside if (copy).
    if (copy) {
      s.payload = &source(name);
      s.scales = &source(sname);
    } else {
      s.payload = nullptr;
      s.scales = nullptr;
    }
    s.payload_entry = &e;
    s.scales_entry = &se;
    s.N = N;
    s.K = K;
    s.SN = SN;
    s.SK = SK;
    return s;
  }

  // FP8 row range into the bump (128-aligned bounds: whole scale rows, so no
  // re-blocking — the scales concatenate exactly like the payload).
  GlmQuantMatrix load_fp8_native_rows(const std::string& name, int64_t r0, int64_t rn) {
    Fp8Native s = fp8_source(name);
    check_range(name, r0, rn, s.N);
    if (r0 % 128 != 0 || rn % 128 != 0)
      fail(name + ": fp8 row slice needs 128-aligned bounds");
    GlmQuantMatrix q;
    q.rows = rn;
    q.cols = s.K;
    q.scale_block_rows = 128;
    q.scale_block_cols = 128;
    const int64_t sn = rn / 128;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rn) * s.K));
    q.scales = static_cast<const float*>(bump.alloc(static_cast<size_t>(sn) * s.SK * 4));
    if (copy) {
      std::memcpy(bump.host(const_cast<uint8_t*>(q.payload)),
                  static_cast<const uint8_t*>(s.payload->data) + static_cast<size_t>(r0) * s.K,
                  static_cast<size_t>(rn) * s.K);
      widen_bf16_to_f32(static_cast<const uint16_t*>(s.scales->data) +
                            static_cast<size_t>(r0 / 128) * s.SK,
                        static_cast<float*>(bump.host(const_cast<float*>(q.scales))),
                        static_cast<size_t>(sn) * s.SK);
    }
    note_read(*s.payload_entry, static_cast<size_t>(rn) * s.K);
    note_read(*s.scales_entry, static_cast<size_t>(sn) * s.SK * 2);
    if (copy) {
      consumed(*s.payload);
      consumed(*s.scales);
    }
    return q;
  }

  // FP8 column range into the bump (128-aligned; payload rows packed, scale
  // rows packed + widened).
  GlmQuantMatrix load_fp8_native_cols(const std::string& name, int64_t c0, int64_t cn) {
    Fp8Native s = fp8_source(name);
    check_range(name + " cols", c0, cn, s.K);
    if (c0 % 128 != 0 || cn % 128 != 0)
      fail(name + ": fp8 column slice needs 128-aligned bounds");
    GlmQuantMatrix q;
    q.rows = s.N;
    q.cols = cn;
    q.scale_block_rows = 128;
    q.scale_block_cols = 128;
    const int64_t sk = cn / 128;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(s.N) * cn));
    q.scales = static_cast<const float*>(bump.alloc(static_cast<size_t>(s.SN) * sk * 4));
    if (copy) {
      const uint8_t* sp = static_cast<const uint8_t*>(s.payload->data);
      uint8_t* dp = static_cast<uint8_t*>(bump.host(const_cast<uint8_t*>(q.payload)));
      for (int64_t r = 0; r < s.N; ++r)
        std::memcpy(dp + static_cast<size_t>(r) * cn, sp + static_cast<size_t>(r) * s.K + c0,
                    static_cast<size_t>(cn));
      const uint16_t* ss = static_cast<const uint16_t*>(s.scales->data);
      float* ds = const_cast<float*>(q.scales);
      float* dh = static_cast<float*>(bump.host(ds));
      for (int64_t sr = 0; sr < s.SN; ++sr)
        widen_bf16_to_f32(ss + static_cast<size_t>(sr) * s.SK + c0 / 128,
                          dh + static_cast<size_t>(sr) * sk, static_cast<size_t>(sk));
    }
    note_read(*s.payload_entry, static_cast<size_t>(s.N) * cn);
    note_read(*s.scales_entry, static_cast<size_t>(s.SN) * sk * 2);
    if (copy) {
      consumed(*s.payload);
      consumed(*s.scales);
    }
    return q;
  }

  // Contiguous BF16 row ranges of several source row spans concatenated into
  // one bump buffer (the GDN's segmented qkv/conv assembly + lm head shard).
  void copy_rows_into(const std::string& name, int64_t src_row, int64_t rows, uint16_t* dst,
                      int64_t dst_row, int64_t width) {
    const QwenExpectedTensor& e = expected(name);
    check_range(name, src_row, rows, e.shape[0]);
    if (copy) {
      const TensorInfo& t = source(name);
      std::memcpy(bump.host(dst) + static_cast<size_t>(dst_row) * width,
                  static_cast<const uint8_t*>(t.data) + static_cast<size_t>(src_row) * width * 2,
                  static_cast<size_t>(rows) * width * 2);
    }
    note_read(e, static_cast<size_t>(rows) * width * 2);
  }

  // FP8 row segments (each 128-aligned) concatenated into one bump matrix
  // (the GDN's merged in_proj_qkv): payload and scale rows both concatenate.
  GlmQuantMatrix merge_fp8_segments(const std::string& name, int64_t total_rows, int64_t K,
                                    const std::vector<std::array<int64_t, 3>>& segs) {
    Fp8Native s = fp8_source(name);
    if (K != s.K) fail(name + ": merged width disagrees with the checkpoint");
    GlmQuantMatrix q;
    q.rows = total_rows;
    q.cols = K;
    q.scale_block_rows = 128;
    q.scale_block_cols = 128;
    const int64_t SK = s.SK;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(total_rows) * K));
    q.scales = static_cast<const float*>(
        bump.alloc(static_cast<size_t>(total_rows / 128) * SK * 4));
    if (copy) {
      uint8_t* dp = reinterpret_cast<uint8_t*>(bump.host(const_cast<uint8_t*>(q.payload)));
      float* ds = static_cast<float*>(bump.host(const_cast<float*>(q.scales)));
      const uint8_t* sp = static_cast<const uint8_t*>(s.payload->data);
      const uint16_t* ss = static_cast<const uint16_t*>(s.scales->data);
      for (const auto& sg : segs) {
        const int64_t src_row = sg[0], rows = sg[1], dst_row = sg[2];
        check_range(name, src_row, rows, s.N);
        if (src_row % 128 != 0 || rows % 128 != 0 || dst_row % 128 != 0)
          fail(name + ": merged fp8 segments need 128-aligned bounds");
        std::memcpy(dp + static_cast<size_t>(dst_row) * K,
                    sp + static_cast<size_t>(src_row) * K, static_cast<size_t>(rows) * K);
        widen_bf16_to_f32(ss + static_cast<size_t>(src_row / 128) * SK,
                          ds + static_cast<size_t>(dst_row / 128) * SK,
                          static_cast<size_t>(rows / 128) * SK);
      }
      consumed(*s.payload);
      consumed(*s.scales);
    }
    // Accounting runs in both modes (the counting pass plans source bytes).
    for (const auto& sg : segs) {
      note_read(*s.payload_entry, static_cast<size_t>(sg[1]) * K);
      note_read(*s.scales_entry, static_cast<size_t>(sg[1] / 128) * SK * 2);
    }
    return q;
  }

  void build_full(const std::string& p) {
    const int64_t d = cfg.head_dim;
    QwenFullAttnResident& a = out.full;
    a.local_heads = geo.local_heads;
    a.head_begin = geo.head_begin;
    a.local_kv_heads = geo.local_kv_heads;
    a.kv_head_begin = geo.kv_head_begin;
    const int64_t q0 = static_cast<int64_t>(geo.head_begin) * 2 * d;
    const int64_t qn = static_cast<int64_t>(geo.local_heads) * 2 * d;
    const int64_t kv0 = static_cast<int64_t>(geo.kv_head_begin) * d;
    const int64_t kvn = static_cast<int64_t>(geo.local_kv_heads) * d;
    const int64_t o0 = static_cast<int64_t>(geo.head_begin) * d;
    const int64_t on = static_cast<int64_t>(geo.local_heads) * d;
    a.q_proj_fp8 = load_fp8_native_rows(p + "self_attn.q_proj.weight", q0, qn);
    a.k_proj_fp8 = load_fp8_native_rows(p + "self_attn.k_proj.weight", kv0, kvn);
    a.v_proj_fp8 = load_fp8_native_rows(p + "self_attn.v_proj.weight", kv0, kvn);
    a.o_proj_fp8 = load_fp8_native_cols(p + "self_attn.o_proj.weight", o0, on);
    a.q_norm = load_bf16(p + "self_attn.q_norm.weight");
    a.k_norm = load_bf16(p + "self_attn.k_norm.weight");
  }

  void build_gdn(const std::string& p) {
    const int64_t H = cfg.hidden_size;
    const int64_t dk = cfg.gdn_key_head_dim, dv = cfg.gdn_value_head_dim;
    const int64_t K = static_cast<int64_t>(cfg.gdn_key_heads) * dk;
    const int64_t lk = geo.local_key_heads, lv = geo.local_value_heads;
    const int64_t r = geo.rank;
    QwenGdnResident& g = out.gdn;
    g.local_key_heads = static_cast<int>(lk);
    g.local_value_heads = static_cast<int>(lv);
    // in_proj_qkv rows [q | k | v] at the local geometry (world=1: whole).
    const int64_t local_rows = 2 * lk * dk + lv * dv;
    const std::string qkv_name = p + "linear_attn.in_proj_qkv.weight";
    g.in_proj_qkv_fp8 = merge_fp8_segments(
        qkv_name, local_rows, H,
        {{r * lk * dk, lk * dk, 0},
         {K + r * lk * dk, lk * dk, lk * dk},
         {2 * K + r * lv * dv, lv * dv, 2 * lk * dk}});
    if (copy) consumed(source(qkv_name));
    // The conv channels follow the same three segments ([C, 1, w] rows).
    const int64_t w = cfg.gdn_conv_width;
    uint16_t* conv = static_cast<uint16_t*>(bump.alloc(static_cast<size_t>(local_rows) *
                                                       static_cast<size_t>(w) * 2));
    const std::string conv_name = p + "linear_attn.conv1d.weight";
    copy_rows_into(conv_name, r * lk * dk, lk * dk, conv, 0, w);
    copy_rows_into(conv_name, K + r * lk * dk, lk * dk, conv, lk * dk, w);
    copy_rows_into(conv_name, 2 * K + r * lv * dv, lv * dv, conv, 2 * lk * dk, w);
    if (copy) consumed(source(conv_name));
    g.conv = conv;
    g.in_proj_z_fp8 =
        load_fp8_native_rows(p + "linear_attn.in_proj_z.weight", r * lv * dv, lv * dv);
    g.in_proj_a = load_bf16_rows(p + "linear_attn.in_proj_a.weight", r * lv, lv);
    g.in_proj_b = load_bf16_rows(p + "linear_attn.in_proj_b.weight", r * lv, lv);
    g.a_log = load_bf16_as_f32(p + "linear_attn.A_log", r * lv, lv);
    g.dt_bias = load_bf16_as_f32(p + "linear_attn.dt_bias", r * lv, lv);
    g.norm = load_bf16(p + "linear_attn.norm.weight");
    g.out_proj_fp8 =
        load_fp8_native_cols(p + "linear_attn.out_proj.weight", r * lv * dv, lv * dv);
  }

  void build_mlp(const std::string& p) {
    const int64_t li = geo.local_inter;
    const int64_t r0 = static_cast<int64_t>(geo.rank) * li;
    Qwen35DenseMlpResident& m = out.mlp;
    m.gate_fp8 = load_fp8_native_rows(p + "mlp.gate_proj.weight", r0, li);
    m.up_fp8 = load_fp8_native_rows(p + "mlp.up_proj.weight", r0, li);
    m.down_fp8 = load_fp8_native_cols(p + "mlp.down_proj.weight", r0, li);
  }

  void build_layer(int layer) {
    const int mtp_layer = cfg.mtp_layer();
    if (layer < 0 || layer >= cfg.num_hidden_layers + (mtp_layer >= 0 ? 1 : 0))
      fail("build_layer: layer out of range");
    const bool is_mtp = layer == mtp_layer;
    const Qwen35LayerKind kind = is_mtp ? Qwen35LayerKind::Full : cfg.layers[layer];
    const std::string p = qwen35_layer_prefix(cfg, layer);
    if (loader_verbose())
      std::fprintf(stderr, "[qwen35] build_layer %d kind=%d prefix=%s\n", layer, (int)kind, p.c_str());
    Qwen35LayerResident& o = out;
    o.kind = kind;
    o.layer = layer;
    o.input_norm = load_bf16(p + "input_layernorm.weight");
    o.post_norm = load_bf16(p + "post_attention_layernorm.weight");
    if (kind == Qwen35LayerKind::Gdn)
      build_gdn(p);
    else
      build_full(p);
    build_mlp(p);
    if (loader_verbose())
      std::fprintf(stderr, "[qwen35] layer %d done (kind=%d)\n", layer, (int)kind);
  }
};

const char* Qwen35LoaderFamily::who() { return "qwen35 loader"; }

uint64_t Qwen35LoaderFamily::loader_format() { return 2; }

int Qwen35LoaderFamily::max_layer(const Config& c) {
  return c.num_hidden_layers + (c.mtp_layer() >= 0 ? 1 : 0);
}

int Qwen35LoaderFamily::main_layers(const Config& c) { return c.num_hidden_layers; }

std::vector<Qwen35LoaderFamily::Expected> Qwen35LoaderFamily::layer_table(const Config& c,
                                                                          int layer) {
  return qwen35_expected_layer_tensors(c, layer);
}

std::vector<Qwen35LoaderFamily::Expected> Qwen35LoaderFamily::global_table(const Config& c) {
  return qwen35_expected_global_tensors(c);
}

void Qwen35LoaderFamily::validate_binding(const Config& c, const PresentMap& present) {
  const QwenBindReport rep = qwen35_validate_text_binding(c, present);
  if (rep.ok()) return;
  std::string msg = "qwen35 loader: checkpoint binding failed: ";
  for (size_t i = 0; i < rep.errors.size() && i < 8; ++i) {
    if (i) msg += "; ";
    msg += rep.errors[i];
  }
  throw std::runtime_error(msg);
}

void Qwen35LoaderFamily::check_sources(const Config&, const LoaderTensorMap&) {
  // No PLE n-gram hash buffers on this family: nothing to check beyond the
  // binding table (names, dtypes, shapes), already gated above.
}

bool Qwen35LoaderFamily::digest_included(const Expected& e) { return is_replicated_35(e); }

bool Qwen35LoaderFamily::discard_after_pack(const Expected& e) {
  return e.cls == QwenWeightClass::Gdn || e.cls == QwenWeightClass::FullAttn ||
         e.cls == QwenWeightClass::DenseMlp;
}

size_t Qwen35LoaderFamily::globals_bytes(const Config& c, int rank, int world,
                                         LoaderHeadSharding) {
  const size_t H = static_cast<size_t>(c.hidden_size);
  const size_t V = static_cast<size_t>(c.vocab_size);
  const size_t Vn = static_cast<size_t>(V * (rank + 1) / world) - static_cast<size_t>(V * rank / world);
  size_t b = 0;
  b += align_up_256(V * H * 2);   // embed
  b += align_up_256(Vn * H * 2);  // lm head shard
  b += align_up_256(H * 2);       // final norm
  if (c.mtp_layer() >= 0) {
    b += align_up_256(H * 2 * H * 2);  // mtp.fc [H, 2H] BF16
    b += align_up_256(H * 2);          // mtp.norm
    b += align_up_256(H * 2);          // mtp.pre_fc_norm_embedding
    b += align_up_256(H * 2);          // mtp.pre_fc_norm_hidden
  }
  return b;
}

size_t Qwen35LoaderFamily::extra_resident_bytes(const Config&, int, int) { return 0; }

void Qwen35LoaderFamily::after_restore(const Config&, int, const LoaderTensorMap&,
                                       LayerResident&) {
  // No PLE table scale to re-seed: nothing to do.
}

void Qwen35LoaderFamily::build_globals(const Config& c, const Geometry& geo,
                                       const LoaderTensorMap& tensors, LayerBump& bump,
                                       GlobalsResident& out, uint64_t& source_bytes,
                                       uint64_t& verbatim_bytes, LoaderHeadSharding) {
  auto lookup = [&](const std::string& name) -> const TensorInfo& {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second)
      throw std::runtime_error("qwen35 loader: global tensor missing: " + name);
    return *it->second;
  };
  auto copy_global = [&](const std::string& name) -> uint16_t* {
    const TensorInfo& t = lookup(name);
    uint16_t* dst = static_cast<uint16_t*>(bump.alloc(t.nbytes()));
    std::memcpy(bump.host(dst), t.data, t.nbytes());
    source_bytes += t.nbytes();
    verbatim_bytes += t.nbytes();
    return dst;
  };
  out.embed = copy_global("model.language_model.embed_tokens.weight");
  const int64_t V = c.vocab_size;
  const int64_t V0 = V * geo.rank / geo.world;
  const int64_t Vn = V * (geo.rank + 1) / geo.world - V0;
  const size_t H = static_cast<size_t>(c.hidden_size);
  const TensorInfo& lm = lookup("lm_head.weight");
  uint16_t* head = static_cast<uint16_t*>(bump.alloc(static_cast<size_t>(Vn) * H * 2));
  std::memcpy(reinterpret_cast<uint8_t*>(bump.host(head)),
              static_cast<const uint8_t*>(lm.data) + static_cast<size_t>(V0) * H * 2,
              static_cast<size_t>(Vn) * H * 2);
  source_bytes += static_cast<uint64_t>(Vn) * H * 2;
  verbatim_bytes += static_cast<uint64_t>(Vn) * H * 2;
  out.lm_head = head;
  out.lm_vocab_begin = geo.lm_vocab_begin;
  out.lm_vocab_count = geo.lm_vocab_count;
  out.final_norm = copy_global("model.language_model.norm.weight");
  if (c.mtp_layer() >= 0) {
    out.mtp_fc = copy_global("mtp.fc.weight");
    out.mtp_norm = copy_global("mtp.norm.weight");
    out.mtp_pre_fc_norm_embedding = copy_global("mtp.pre_fc_norm_embedding.weight");
    out.mtp_pre_fc_norm_hidden = copy_global("mtp.pre_fc_norm_hidden.weight");
  } else {
    out.mtp_fc = nullptr;
    out.mtp_norm = nullptr;
    out.mtp_pre_fc_norm_embedding = nullptr;
    out.mtp_pre_fc_norm_hidden = nullptr;
  }
}

std::string& resident_image_dir_storage_35() {
  static std::string dir;
  return dir;
}

Qwen35LayerStream::Qwen35LayerStream(const Qwen35TextConfig& cfg, const std::string& checkpoint_dir,
                                     int rank, int world, LoaderResidency residency,
                                     LoaderHeadSharding head, bool resident_mtp)
    : ResidentLayerStream<Qwen35LoaderFamily>(cfg, checkpoint_dir, rank, world, residency, head,
                                              resident_mtp) {
  if (resident_image_dir().empty()) resident_image_dir_storage_35() = checkpoint_dir + "/resident_qwen35";
  open_resident_image();
}

void Qwen35LayerStream::set_resident_image_dir(const std::string& dir) {
  resident_image_dir_storage_35() = dir;
}

const std::string& Qwen35LayerStream::resident_image_dir() { return resident_image_dir_storage_35(); }

const std::string& Qwen35LayerStream::image_dir() const { return resident_image_dir(); }

template class ResidentLayerStream<Qwen35LoaderFamily>;

}  // namespace dgpp
