#include "models/inkling2/loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/log.hpp"

namespace dgpp {
namespace {

bool ends_with(const std::string& s, const char* suffix) {
  const size_t n = std::strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// The replicated set (rank-invariant reads at world > 1): norms, mix
// queries, the hidden-stream convolution taps, the router, the controllers,
// the relative table, the embedding and the NVFP4 experts' per-expert
// factors. Everything else is a slice: the projections' head rows and
// columns, the per-head scales and K/V taps, the experts' payload and block
// scales, the unembedding under VocabSharded. The vision tower and the draft
// model are not this stream's.
bool is_replicated(const Inkling2ExpectedTensor& e) {
  switch (e.cls) {
    case Inkling2WeightClass::Embed:
    case Inkling2WeightClass::GlobalNorm:
    case Inkling2WeightClass::MixQuery:
    case Inkling2WeightClass::LayerNorm:
    case Inkling2WeightClass::HiddenConv:
    case Inkling2WeightClass::Router:
    case Inkling2WeightClass::Controller:
      return true;
    case Inkling2WeightClass::AttentionAux:
      return ends_with(e.name, "rel_logits_proj.proj");
    case Inkling2WeightClass::RoutedExpert:
      return e.role == Inkling2TensorRole::Fp4Amax || e.role == Inkling2TensorRole::Fp4Denominator;
    case Inkling2WeightClass::Attention:
    case Inkling2WeightClass::Unembed:
    case Inkling2WeightClass::Vision:
    case Inkling2WeightClass::Draft:
      return false;
  }
  return false;
}

float bf16_at(const uint8_t* src, size_t index) {
  uint16_t bits;
  std::memcpy(&bits, src + index * 2, 2);
  return bf16_bits_to_float(bits);
}

float f32_at(const uint8_t* src, size_t index) {
  float v;
  std::memcpy(&v, src + index * 4, 4);
  return v;
}

}  // namespace

struct Inkling2LoaderFamily::Builder : WeightBuilder<Inkling2ExpectedTensor> {
  const Inkling2Config& cfg;
  const Inkling2LocalGeometry& geo;
  Inkling2LayerResident& out;

  Builder(const Inkling2Config& cfg_, const Inkling2LocalGeometry& geo_,
          const std::vector<Inkling2ExpectedTensor>& table_,
          const std::unordered_map<std::string, const Inkling2ExpectedTensor*>& by_name_, LayerBump& bump_,
          Inkling2LayerResident& out_, const std::unordered_map<std::string, const TensorInfo*>& tensors_,
          std::vector<DequantJob>& jobs_, std::vector<PackJob>& packs_, bool copy_)
      : WeightBuilder<Inkling2ExpectedTensor>(table_, by_name_, bump_, tensors_, jobs_, packs_, copy_, geo_.rank,
                                              geo_.world, "inkling2 loader"),
        cfg(cfg_), geo(geo_), out(out_) {}

  bool replicated(const Inkling2ExpectedTensor& e) const override { return is_replicated(e); }

  // ---- stacked BF16 row ranges ----------------------------------------------
  // One packable grant holding contiguous row ranges of several [rows,
  // hidden] sources back to back.
  struct RowPart {
    std::string name;
    int64_t row0, rows;
  };
  const uint16_t* load_row_stack(const std::vector<RowPart>& parts, int64_t* total_rows) {
    const int64_t H = cfg.hidden_size;
    int64_t rows = 0;
    for (const RowPart& part : parts) rows += part.rows;
    uint16_t* dst = static_cast<uint16_t*>(grant_bf16(rows, H, /*packable=*/true));
    int64_t at = 0;
    for (const RowPart& part : parts) {
      const Inkling2ExpectedTensor& e = expected(part.name);
      check_range(e.name, part.row0, part.rows, e.shape[0]);
      if (e.shape.size() != 2 || e.shape[1] != H) fail("'" + e.name + "' is not [rows, hidden]");
      if (copy) {
        const TensorInfo& t = source(e.name);
        std::memcpy(reinterpret_cast<uint8_t*>(bump.host(dst)) + static_cast<size_t>(at) * H * 2,
                    static_cast<const uint8_t*>(t.data) + static_cast<size_t>(part.row0) * H * 2,
                    static_cast<size_t>(part.rows) * H * 2);
        consumed(t);
      }
      note_read(e, static_cast<size_t>(part.rows) * H * 2);
      at += part.rows;
    }
    *total_rows = rows;
    return dst;
  }

  // ---- convolution taps by lag ------------------------------------------------
  // F32 [channels, W], taps[c][lag] with lag 0 the current row. `static_form`
  // reads a [C, 1, W] tensor whose last tap is the current row, otherwise a
  // [W, C] tensor whose first row is the current row.
  const float* load_taps(const std::string& name, bool static_form, int64_t c0, int64_t channels) {
    const Inkling2ExpectedTensor& e = expected(name);
    const int64_t W = cfg.sconv_kernel_size;
    const int64_t C = static_form ? e.shape[0] : e.shape[1];
    check_range(name, c0, channels, C);
    float* dst = static_cast<float*>(bump.alloc(static_cast<size_t>(channels) * W * 4));
    if (copy) {
      const TensorInfo& t = source(name);
      const uint8_t* src = static_cast<const uint8_t*>(t.data);
      float* h = bump.host(dst);
      for (int64_t c = 0; c < channels; ++c)
        for (int64_t lag = 0; lag < W; ++lag)
          h[c * W + lag] = static_form ? bf16_at(src, static_cast<size_t>((c0 + c) * W + (W - 1 - lag)))
                                       : bf16_at(src, static_cast<size_t>(lag * C + c0 + c));
      consumed(t);
    }
    note_read(e, static_cast<size_t>(channels) * W * 2);
    return dst;
  }

  // ---- FP32 queries scaled by hidden^-0.5 ----------------------------------------
  const float* load_query(const std::string& name) {
    const Inkling2ExpectedTensor& e = expected(name);
    const size_t n = e.numel();
    float* dst = static_cast<float*>(bump.alloc(n * 4));
    if (copy) {
      const TensorInfo& t = source(name);
      const uint8_t* src = static_cast<const uint8_t*>(t.data);
      const float scale = static_cast<float>(std::pow(static_cast<double>(cfg.hidden_size), -0.5));
      float* h = bump.host(dst);
      for (size_t i = 0; i < n; ++i) h[i] = f32_at(src, i) * scale;
      consumed(t);
    }
    note_read(e, n * 4);
    return dst;
  }

  // ---- the experts ---------------------------------------------------------------
  // One projection of every expert from its stacked tensors, read once. A
  // row slice (w13) is a contiguous range per expert. A column slice (w2)
  // is packed row by row. `slot` is the view-table position (0 or 2).
  void load_fp4_stack(const std::string& base, bool row_slice, int64_t start, int64_t count, int slot) {
    const Inkling2ExpectedTensor& ep = expected(base + "_weight");
    const Inkling2ExpectedTensor& es = expected(base + "_weight_scale");
    const Inkling2ExpectedTensor& ea = expected(base + "_weight_amax");
    const Inkling2ExpectedTensor& ed = expected(base + "_weight_outer_scale_denominator");
    const int64_t E = ep.shape[0], N = ep.shape[1], K = ep.shape[2] * 2;
    if (es.shape[0] != E || es.shape[1] != N || es.shape[2] * 16 != K) fail("'" + es.name + "' is not [E, N, K/16]");
    if (row_slice) {
      check_range(ep.name, start, count, N);
    } else {
      check_range(ep.name, start, count, K);
      if (start % 16 != 0 || count % 16 != 0) fail("a column slice of '" + ep.name + "' must cover whole 16-value blocks");
    }
    const int64_t n = row_slice ? count : N, k = row_slice ? K : count;
    const size_t pay_row = static_cast<size_t>(k / 2), sc_row = static_cast<size_t>(k / 16);
    const size_t pay_bytes = static_cast<size_t>(n) * pay_row, sc_bytes = static_cast<size_t>(n) * sc_row;
    uint8_t* payload = static_cast<uint8_t*>(bump.alloc(static_cast<size_t>(E) * pay_bytes));
    uint8_t* scales = static_cast<uint8_t*>(bump.alloc(static_cast<size_t>(E) * sc_bytes));
    float* globals = static_cast<float*>(bump.alloc(static_cast<size_t>(E) * 4));
    if (copy) {
      const TensorInfo& tp = source(ep.name);
      const TensorInfo& ts = source(es.name);
      const uint8_t* sp = static_cast<const uint8_t*>(tp.data);
      const uint8_t* ss = static_cast<const uint8_t*>(ts.data);
      uint8_t* hp = bump.host(payload);
      uint8_t* hs = bump.host(scales);
      const size_t full_pay = static_cast<size_t>(K / 2), full_sc = static_cast<size_t>(K / 16);
      for (int64_t e = 0; e < E; ++e) {
        if (row_slice) {
          std::memcpy(hp + static_cast<size_t>(e) * pay_bytes,
                      sp + (static_cast<size_t>(e) * N + static_cast<size_t>(start)) * full_pay, pay_bytes);
          std::memcpy(hs + static_cast<size_t>(e) * sc_bytes,
                      ss + (static_cast<size_t>(e) * N + static_cast<size_t>(start)) * full_sc, sc_bytes);
        } else {
          for (int64_t r = 0; r < N; ++r) {
            const size_t row = static_cast<size_t>(e) * N + static_cast<size_t>(r);
            std::memcpy(hp + static_cast<size_t>(e) * pay_bytes + static_cast<size_t>(r) * pay_row,
                        sp + row * full_pay + static_cast<size_t>(start / 2), pay_row);
            std::memcpy(hs + static_cast<size_t>(e) * sc_bytes + static_cast<size_t>(r) * sc_row,
                        ss + row * full_sc + static_cast<size_t>(start / 16), sc_row);
          }
        }
      }
      consumed(tp);
      consumed(ts);
      // The divisor of a finished dot: denominator / amax, the reciprocal of
      // the checkpoint's per-expert factor.
      const TensorInfo& ta = source(ea.name);
      const TensorInfo& td = source(ed.name);
      const float den = f32_at(static_cast<const uint8_t*>(td.data), 0);
      float* hg = bump.host(globals);
      for (int64_t e = 0; e < E; ++e) hg[e] = den / f32_at(static_cast<const uint8_t*>(ta.data), static_cast<size_t>(e));
      consumed(ta);
      consumed(td);
    }
    note_read(ep, static_cast<size_t>(E) * pay_bytes);
    note_read(es, static_cast<size_t>(E) * sc_bytes);
    note_read(ea, static_cast<size_t>(E) * 4);
    note_read(ed, 4);
    for (int64_t e = 0; e < E; ++e) {
      GlmFp4Matrix& m = out.moe.fp4[static_cast<size_t>(e) * 3 + static_cast<size_t>(slot)];
      m.payload = payload + static_cast<size_t>(e) * pay_bytes;
      m.scales = scales + static_cast<size_t>(e) * sc_bytes;
      m.global_scale = globals + e;
      m.rows = n;
      m.cols = k;
      m.scale_group = kFp4Group;
    }
  }

  void load_mx8_stack(const std::string& base, bool row_slice, int64_t start, int64_t count, int slot) {
    const Inkling2ExpectedTensor& ep = expected(base + "_weight");
    const Inkling2ExpectedTensor& es = expected(base + "_weight_scale");
    const int64_t E = ep.shape[0], N = ep.shape[1], K = ep.shape[2];
    if (es.shape[0] != E || es.shape[1] != N || es.shape[2] * 32 != K) fail("'" + es.name + "' is not [E, N, K/32]");
    if (row_slice) {
      check_range(ep.name, start, count, N);
    } else {
      check_range(ep.name, start, count, K);
      if (start % 32 != 0 || count % 32 != 0) fail("a column slice of '" + ep.name + "' must cover whole 32-value blocks");
    }
    const int64_t n = row_slice ? count : N, k = row_slice ? K : count;
    const size_t pay_row = static_cast<size_t>(k), sc_row = static_cast<size_t>(k / 32);
    const size_t pay_bytes = static_cast<size_t>(n) * pay_row, sc_count = static_cast<size_t>(n) * sc_row;
    uint8_t* payload = static_cast<uint8_t*>(bump.alloc(static_cast<size_t>(E) * pay_bytes));
    float* scales = static_cast<float*>(bump.alloc(static_cast<size_t>(E) * sc_count * 4));
    if (copy) {
      const TensorInfo& tp = source(ep.name);
      const TensorInfo& ts = source(es.name);
      const uint8_t* sp = static_cast<const uint8_t*>(tp.data);
      const uint8_t* ss = static_cast<const uint8_t*>(ts.data);
      uint8_t* hp = bump.host(payload);
      float* hs = bump.host(scales);
      const size_t full_sc = static_cast<size_t>(K / 32);
      for (int64_t e = 0; e < E; ++e) {
        for (int64_t r = 0; r < n; ++r) {
          const size_t row = static_cast<size_t>(e) * N + static_cast<size_t>(row_slice ? start + r : r);
          const size_t col = row_slice ? 0 : static_cast<size_t>(start);
          std::memcpy(hp + static_cast<size_t>(e) * pay_bytes + static_cast<size_t>(r) * pay_row,
                      sp + row * static_cast<size_t>(K) + col, pay_row);
          const uint8_t* srow = ss + row * full_sc + col / 32;
          float* drow = hs + static_cast<size_t>(e) * sc_count + static_cast<size_t>(r) * sc_row;
          for (size_t b = 0; b < sc_row; ++b) {
            if (srow[b] == 255) fail("'" + es.name + "' holds a NaN scale exponent");
            drow[b] = std::ldexp(1.0f, static_cast<int>(srow[b]) - 127);  // exact
          }
        }
      }
      consumed(tp);
      consumed(ts);
    }
    note_read(ep, static_cast<size_t>(E) * pay_bytes);
    note_read(es, static_cast<size_t>(E) * sc_count);
    for (int64_t e = 0; e < E; ++e) {
      GlmQuantMatrix& m = out.moe.fp8[static_cast<size_t>(e) * 3 + static_cast<size_t>(slot)];
      m.payload = payload + static_cast<size_t>(e) * pay_bytes;
      m.scales = scales + static_cast<size_t>(e) * sc_count;
      m.rows = n;
      m.cols = k;
      m.scale_block_rows = 1;
      m.scale_block_cols = 32;
    }
  }

  // ---- the classes ---------------------------------------------------------
  void build_attention(const std::string& a, int layer) {
    Inkling2AttnResident& r = out.attn;
    const int64_t D = cfg.head_dim, R = cfg.d_rel, W = cfg.sconv_kernel_size;
    const int64_t lh = geo.local_heads, hb = geo.head_begin, lkv = geo.local_kv_heads, kvb = geo.kv_head_begin;
    r.local = cfg.is_local_layer(layer);
    r.extent = cfg.rel_extent_of(layer);
    r.local_heads = geo.local_heads;
    r.head_begin = geo.head_begin;
    r.local_kv_heads = geo.local_kv_heads;
    r.kv_head_begin = geo.kv_head_begin;
    r.q_side = load_row_stack({{a + "wq_du.weight", hb * D, lh * D},
                               {a + "wr_du.weight", hb * R, lh * R},
                               {a + "gate_attn.weight", hb * D, lh * D},
                               {a + "attn_sink_proj.weight", hb, lh}},
                              &r.q_side_rows);
    for (int which = 0; which < 2; ++which) {
      const std::string proj = a + (which == 0 ? "wk_dv.weight" : "wv_dv.weight");
      const std::string conv = a + (which == 0 ? "k_sconv." : "v_sconv.");
      std::vector<RowPart> parts = {{proj, kvb * D, lkv * D}};
      if (!r.local) parts.push_back({conv + "weight_proj", kvb * W, lkv * W});
      const uint16_t* m = load_row_stack(parts, &r.kv_rows);
      const float* taps = r.local ? load_taps(conv + "weight", /*static_form=*/true, kvb * D, lkv * D)
                                  : load_taps(conv + "static_w", /*static_form=*/false, kvb * D, lkv * D);
      (which == 0 ? r.k : r.v) = m;
      (which == 0 ? r.k_taps : r.v_taps) = taps;
    }
    r.o_proj = load_bf16_cols(a + "wo_ud.weight", hb * D, lh * D, /*packable=*/true);
    r.q_scale = load_bf16_as_f32(a + "q_l2_clip.scale", hb * D, lh * D);
    r.r_scale = load_bf16_as_f32(a + "rel_l2_clip.scale", hb * R, lh * R);
    r.rel_proj = load_bf16_as_f32(a + "rel_logits_proj.proj");
  }

  void build_moe(const std::string& p, int layer) {
    Inkling2MoeResident& m = out.moe;
    m.router = load_bf16(p + "gate.weight");
    m.router_bias = load_f32(p + "gate.bias");
    m.mxfp8 = cfg.layer_mxfp8(layer);
    m.local_inter = geo.local_inter;
    const size_t entries = static_cast<size_t>(cfg.n_routed_experts) * 3;
    m.fp4.clear();
    m.fp8.clear();
    const int64_t I = geo.local_inter, lo = geo.inter_begin;
    if (m.mxfp8) {
      m.fp8.assign(entries, GlmQuantMatrix{});
      load_mx8_stack(p + "experts.w13", /*row_slice=*/true, 2 * lo, 2 * I, 0);
      load_mx8_stack(p + "experts.w2", /*row_slice=*/false, lo, I, 2);
    } else {
      m.fp4.assign(entries, GlmFp4Matrix{});
      load_fp4_stack(p + "experts.w13", /*row_slice=*/true, 2 * lo, 2 * I, 0);
      load_fp4_stack(p + "experts.w2", /*row_slice=*/false, lo, I, 2);
    }
  }

  void build_controllers(const std::string& p) {
    static const char* names[4] = {"attn", "k", "v", "mlp"};
    const int64_t H = cfg.hidden_size, r = cfg.gate_rank;
    for (int s = 0; s < 4; ++s) {
      const std::string c = p + "controllers." + names[s] + ".";
      out.ctrl[s].down = load_bf16(c + "down_weight", /*packable=*/true);
      uint16_t* dst = static_cast<uint16_t*>(bump.alloc(static_cast<size_t>(2 * H * r) * 2));
      int64_t at = 0;
      for (const char* part : {"gain_weight", "forget_weight"}) {
        const Inkling2ExpectedTensor& e = expected(c + part);
        if (copy) {
          const TensorInfo& t = source(e.name);
          std::memcpy(reinterpret_cast<uint8_t*>(bump.host(dst)) + static_cast<size_t>(at) * 2, t.data, e.nbytes());
          consumed(t);
        }
        note_read(e, e.nbytes());
        at += H * r;
      }
      out.ctrl[s].gain_forget = dst;
    }
  }

  void build_layer(int layer) {
    if (layer < 0 || layer >= cfg.num_hidden_layers) fail("layer index out of range: " + std::to_string(layer));
    const std::string p = inkling2_layer_prefix(layer);
    out.layer = layer;
    out.local = cfg.is_local_layer(layer);
    out.core = cfg.is_core_layer(layer);
    if (out.local) {
      const uint16_t* norm = load_bf16(p + "attn_norm.weight");
      out.attn_norm[0] = out.attn_norm[1] = out.attn_norm[2] = norm;
    } else {
      out.attn_norm[0] = load_bf16(p + "q_input_norm.weight");
      out.attn_norm[1] = load_bf16(p + "k_input_norm.weight");
      out.attn_norm[2] = load_bf16(p + "v_input_norm.weight");
    }
    out.mlp_norm = load_bf16(p + "mlp_norm.weight");
    out.attn_queries = nullptr;
    out.attn_query_count = 0;
    if (layer > 0) {
      out.attn_queries = load_query("model.llm.attn_res.attn_queries." + std::to_string(layer - 1));
      out.attn_query_count = out.local ? 1 : 3;
    }
    out.mlp_query = load_query("model.llm.attn_res.mlp_queries." + std::to_string(layer));
    build_attention(p + "attn.", layer);
    out.attn_conv = load_taps(p + "attn_sconv.weight", /*static_form=*/true, 0, cfg.hidden_size);
    out.mlp_conv = load_taps(p + "mlp_sconv.weight", /*static_form=*/true, 0, cfg.hidden_size);
    build_moe(p + "mlp.", layer);
    for (auto& c : out.ctrl) c = Inkling2ControllerResident{};
    if (out.core) build_controllers(p);
  }
};

namespace {

std::pair<int, int> lm_head_slice(const Inkling2Config& cfg, int rank, int world) {
  const int64_t V = cfg.vocab_size;
  const int64_t begin = V * rank / world;
  const int64_t end = V * (rank + 1) / world;
  return {static_cast<int>(begin), static_cast<int>(end - begin)};
}

std::string& resident_image_dir_storage() {
  static std::string dir;
  return dir;
}

}  // namespace

Inkling2LocalGeometry Inkling2LocalGeometry::from_config(const Inkling2Config& cfg, int rank, int world,
                                                         Inkling2HeadSharding head) {
  if (world < 1 || rank < 0 || rank >= world) throw std::invalid_argument("inkling2 loader: rank/world out of range");
  if (world > 1) inkling2_tp_validate_geometry(cfg, rank, world);
  Inkling2LocalGeometry g;
  g.world = world;
  g.rank = rank;
  g.local_heads = cfg.num_attention_heads / world;
  g.head_begin = g.local_heads * rank;
  g.local_kv_heads = cfg.num_key_value_heads / world;
  g.kv_head_begin = g.local_kv_heads * rank;
  g.local_inter = cfg.intermediate_size / world;
  g.inter_begin = g.local_inter * rank;
  if (head == Inkling2HeadSharding::VocabSharded) {
    const auto [b, n] = lm_head_slice(cfg, rank, world);
    g.lm_vocab_begin = b;
    g.lm_vocab_count = n;
  } else {
    g.lm_vocab_begin = 0;
    g.lm_vocab_count = cfg.vocab_size;
  }
  return g;
}

// ---- the family hooks (loaders/resident_stream.hpp) -------------------------

void Inkling2LoaderFamily::validate_binding(const Inkling2Config& cfg, const PresentMap& present) {
  // The checkpoint file holds the text model and the vision tower.
  std::vector<Inkling2ExpectedTensor> expected = inkling2_expected_text_tensors(cfg);
  std::vector<Inkling2ExpectedTensor> vision = inkling2_expected_vision_tensors(cfg);
  expected.insert(expected.end(), std::make_move_iterator(vision.begin()), std::make_move_iterator(vision.end()));
  const Inkling2BindReport rep = inkling2_validate_binding(expected, present);
  if (rep.ok()) return;
  std::string msg = "inkling2 loader: checkpoint binding failed: ";
  for (size_t i = 0; i < rep.errors.size() && i < 8; ++i) {
    if (i) msg += "; ";
    msg += rep.errors[i];
  }
  throw std::runtime_error(msg);
}

bool Inkling2LoaderFamily::digest_included(const Inkling2ExpectedTensor& e) { return is_replicated(e); }

// o_proj's packed column slice reads its source after the builder returns.
bool Inkling2LoaderFamily::discard_after_pack(const Inkling2ExpectedTensor& e) {
  return e.cls == Inkling2WeightClass::Attention;
}

size_t Inkling2LoaderFamily::globals_bytes(const Inkling2Config& cfg, int rank, int world, LoaderHeadSharding head) {
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t count = static_cast<size_t>(Inkling2LocalGeometry::from_config(cfg, rank, world, head).lm_vocab_count);
  size_t b = 0;
  b += align_up_256(static_cast<size_t>(cfg.vocab_size) * H * 2);          // embed, the real rows
  b += align_up_256(H * 2);                                                // embed_norm
  b += align_up_256(H * 2);                                                // norm
  b += align_up_256(count * H * 2);                                        // unembed
  b += align_up_256(H * 4);                                                // output_query
  b += align_up_256(static_cast<size_t>(cfg.core_layers()) * 4 * H * 4);   // message queries
  return b;
}

size_t Inkling2LoaderFamily::globals_side_bytes(const Inkling2Config& cfg, int rank, int world, LoaderHeadSharding head) {
  const int count = Inkling2LocalGeometry::from_config(cfg, rank, world, head).lm_vocab_count;
  if (!bf12_shape_ok(count, cfg.hidden_size)) return 0;
  return align_up_256(static_cast<size_t>(count) * static_cast<size_t>(cfg.hidden_size) * 2);
}

void Inkling2LoaderFamily::build_globals(const Inkling2Config& cfg, const Inkling2LocalGeometry& geo,
                                         const LoaderTensorMap& tensors, LayerBump& bump, Inkling2GlobalsResident& out,
                                         uint64_t& source_bytes, uint64_t& verbatim_bytes, LoaderHeadSharding head) {
  auto lookup = [&](const std::string& name) -> const TensorInfo& {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second) throw std::runtime_error("inkling2 loader: global tensor missing: " + name);
    return *it->second;
  };
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t row_bytes = H * 2;
  auto copy_rows = [&](const std::string& name, size_t row0, size_t rows, bool side, bool verbatim) -> uint16_t* {
    const TensorInfo& t = lookup(name);
    const size_t bytes = rows * row_bytes;
    uint16_t* dst = static_cast<uint16_t*>(side ? bump.alloc_side(bytes) : bump.alloc(bytes));
    std::memcpy(bump.host(dst), static_cast<const uint8_t*>(t.data) + row0 * row_bytes, bytes);
    source_bytes += bytes;
    if (verbatim) verbatim_bytes += bytes;
    return dst;
  };
  const float scale = static_cast<float>(std::pow(static_cast<double>(cfg.hidden_size), -0.5));
  // The embedding and unembedding tensors are padded to padded_vocab_size
  // rows. The engine serves the real rows.
  out.embed = copy_rows("model.llm.embed.weight", 0, static_cast<size_t>(cfg.vocab_size), false, true);
  out.embed_norm = copy_rows("model.llm.embed_norm.weight", 0, 1, false, true);
  out.final_norm = copy_rows("model.llm.norm.weight", 0, 1, false, true);
  const int begin = geo.lm_vocab_begin, count = geo.lm_vocab_count;
  out.unembed = copy_rows("model.llm.unembed.weight", static_cast<size_t>(begin), static_cast<size_t>(count),
                          bf12_shape_ok(count, cfg.hidden_size), head == LoaderHeadSharding::Full);
  out.lm_vocab_begin = begin;
  out.lm_vocab_count = count;
  {
    const TensorInfo& t = lookup("model.llm.attn_res.output_query");
    float* dst = static_cast<float*>(bump.alloc(H * 4));
    float* h = bump.host(dst);
    for (size_t i = 0; i < H; ++i) h[i] = f32_at(static_cast<const uint8_t*>(t.data), i) * scale;
    source_bytes += H * 4;
    verbatim_bytes += H * 4;
    out.output_query = dst;
  }
  {
    // Two rows per core layer in each tensor: (attention, MLP) and (key,
    // value). The resident order is attention, key, value, MLP.
    const TensorInfo& tq = lookup("model.llm.middle_cycle_cross_pass_queries");
    const TensorInfo& tkv = lookup("model.llm.middle_cycle_cross_pass_kv_queries");
    const size_t L = static_cast<size_t>(cfg.core_layers());
    float* dst = static_cast<float*>(bump.alloc(L * 4 * H * 4));
    float* h = bump.host(dst);
    const uint8_t* q = static_cast<const uint8_t*>(tq.data);
    const uint8_t* kv = static_cast<const uint8_t*>(tkv.data);
    for (size_t l = 0; l < L; ++l) {
      const uint8_t* src[4] = {q, kv, kv, q};
      const size_t row[4] = {2 * l, 2 * l, 2 * l + 1, 2 * l + 1};
      for (size_t s = 0; s < 4; ++s)
        for (size_t i = 0; i < H; ++i) h[(l * 4 + s) * H + i] = f32_at(src[s], row[s] * H + i) * scale;
    }
    source_bytes += 2 * (2 * L * H * 4);
    verbatim_bytes += 2 * (2 * L * H * 4);
    out.message_queries = dst;
  }
}

template class ResidentLayerStream<Inkling2LoaderFamily>;

// ---------------------------------------------------------------------------

Inkling2LayerStream::Inkling2LayerStream(const Inkling2Config& cfg, const std::string& checkpoint_dir, int rank,
                                         int world, Inkling2Residency residency, Inkling2HeadSharding head)
    : ResidentLayerStream<Inkling2LoaderFamily>(cfg, checkpoint_dir, rank, world, residency, head,
                                                /*resident_mtp=*/false) {
  open_resident_image();
}

void Inkling2LayerStream::set_resident_image_dir(const std::string& dir) { resident_image_dir_storage() = dir; }
const std::string& Inkling2LayerStream::resident_image_dir() { return resident_image_dir_storage(); }

}  // namespace dgpp
