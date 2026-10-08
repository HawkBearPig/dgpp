// Qwen3.5-27B loader family: native-weight builders for both releases —
// the FP8 release's block FP8 (e4m3 payload + BF16 scales — memcpy +
// BF16→F32 widen, never a BF16→FP8 re-encode), and the mixed NVFP4
// release's channel FP8 (attention, late MLP layers, lm_head) plus NVFP4
// MLP (e2m1 pairs, e4m3 scales per 16, F32 global served as its
// reciprocal). Text-only, dense SwiGLU MLP, standard pre-norm residual.
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

// The channel-FP8 resident grid: one F32 scale per row, so the kernel's
// column shift must cover the whole K (the smallest power of two >= K);
// see quant_scale_shifts.
constexpr int64_t pow2_ceil(int64_t v) {
  int64_t p = 1;
  while (p < v) p <<= 1;
  return p;
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
  if (e.role == QwenTensorRole::Fp4Global || e.role == QwenTensorRole::InputScale)
    return true;  // the mixed release's F32 scalars: every rank reads the whole
  switch (e.cls) {
    case QwenWeightClass::Norm:
      return true;
    case QwenWeightClass::Mtp:
      return true;
    case QwenWeightClass::Router:
      return true;  // mlp.gate + shared_expert_gate (loader.cpp precedent)
    case QwenWeightClass::SharedExpert:
    case QwenWeightClass::RoutedExpert:
      return false;  // I/W + S/W slices
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
  g.local_inter = cfg.is_moe ? 0 : cfg.intermediate_size / world;
  g.local_moe_inter = cfg.is_moe ? cfg.moe_intermediate_size / world : 0;
  g.local_shared_inter = cfg.is_moe ? cfg.shared_expert_intermediate_size / world : 0;
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
  // BF16 segment merge (the MoE release's GDN in_proj_qkv): payload rows
  // concatenate; no scales.
  const uint16_t* merge_bf16_segments(const std::string& name, int64_t total_rows, int64_t K,
                                      const std::vector<std::array<int64_t, 3>>& segs) {
    const QwenExpectedTensor& e = expected(name);
    if (e.shape.size() != 2 || e.shape[1] != K)
      fail(name + ": merged width disagrees with the checkpoint");
    uint16_t* dst =
        static_cast<uint16_t*>(bump.alloc(static_cast<size_t>(total_rows) * K * 2));
    for (const auto& sg : segs) copy_rows_into(name, sg[0], sg[1], dst, sg[2], K);
    if (copy) consumed(source(name));
    return dst;
  }
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

  // ---- mixed-release builders ---------------------------------------------
  // Every per-matrix loader takes the checkpoint MODULE name (no .weight
  // suffix); the format is resolved from the parsed config_groups.

  enum class Form { BlockFp8, ChannelFp8, Nvfp4, Bf16 };

  Form form_of(const std::string& module) {
    if (cfg.quant_kind == Qwen35QuantKind::Fp8Block) return Form::BlockFp8;
    switch (cfg.tensor_quant(module)) {
      case Qwen35TensorQuant::Bf16: return Form::Bf16;
      case Qwen35TensorQuant::Fp8Channel: return Form::ChannelFp8;
      case Qwen35TensorQuant::Nvfp4:
        fail(module + ": NVFP4 is implemented for the dense MLP only "
                       "(the release's targets keep it there)");
    }
    fail(module + ": unreachable");
  }

  // A channel-FP8 matrix: e4m3 [N, K] payload + BF16 [N, 1] scales.
  struct ChannelSource {
    const QwenExpectedTensor* payload_entry = nullptr;
    const QwenExpectedTensor* scale_entry = nullptr;
    const TensorInfo* payload = nullptr;
    const TensorInfo* scales = nullptr;
    int64_t N = 0, K = 0, SB = 0;  // SB: the resident grid's column block (pow2 >= K)
  };

  ChannelSource channel_source(const std::string& module) {
    const std::string pname = module + ".weight";
    const QwenExpectedTensor& e = expected(pname);
    if (e.shape.size() != 2) fail(pname + ": fp8 matrix needs 2 dims");
    const std::string sname = module + ".weight_scale";
    const QwenExpectedTensor& se = expected(sname);
    if (se.shape.size() != 2 || se.shape[0] != e.shape[0] || se.shape[1] != 1)
      fail(sname + ": the channel scale grid must be [N, 1]");
    ChannelSource s;
    // As in fp8_source: the counting pass never dereferences a source.
    if (copy) {
      s.payload = &source(pname);
      s.scales = &source(sname);
    } else {
      s.payload = nullptr;
      s.scales = nullptr;
    }
    s.payload_entry = &e;
    s.scale_entry = &se;
    s.N = e.shape[0];
    s.K = e.shape[1];
    s.SB = pow2_ceil(s.K);
    return s;
  }

  GlmQuantMatrix channel_matrix(const ChannelSource& s, int64_t rows, int64_t cols) {
    GlmQuantMatrix q;
    q.rows = rows;
    q.cols = cols;
    q.scale_block_rows = 1;
    q.scale_block_cols = s.SB;
    return q;
  }

  // Rows [r0, +rn): payload rows contiguous, one widened F32 scale per row.
  // No 128-alignment contract: a channel matrix slices anywhere.
  GlmQuantMatrix load_channel_rows(const std::string& module, int64_t r0, int64_t rn) {
    ChannelSource s = channel_source(module);
    check_range(module, r0, rn, s.N);
    GlmQuantMatrix q = channel_matrix(s, rn, s.K);
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rn) * s.K));
    q.scales = static_cast<const float*>(bump.alloc(static_cast<size_t>(rn) * 4));
    if (copy) {
      std::memcpy(bump.host(const_cast<uint8_t*>(q.payload)),
                  static_cast<const uint8_t*>(s.payload->data) + static_cast<size_t>(r0) * s.K,
                  static_cast<size_t>(rn) * s.K);
      widen_bf16_to_f32(static_cast<const uint16_t*>(s.scales->data) + static_cast<size_t>(r0),
                        static_cast<float*>(bump.host(const_cast<float*>(q.scales))),
                        static_cast<size_t>(rn));
    }
    note_read(*s.payload_entry, static_cast<size_t>(rn) * s.K);
    note_read(*s.scale_entry, static_cast<size_t>(rn) * 2);
    if (copy) {
      consumed(*s.payload);
      consumed(*s.scales);
    }
    return q;
  }

  // Column range: strided payload rows; the per-row scales are not a sliced
  // axis, so they copy whole.
  GlmQuantMatrix load_channel_cols(const std::string& module, int64_t c0, int64_t cn) {
    ChannelSource s = channel_source(module);
    check_range(module + " cols", c0, cn, s.K);
    GlmQuantMatrix q = channel_matrix(s, s.N, cn);
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(s.N) * cn));
    q.scales = static_cast<const float*>(bump.alloc(static_cast<size_t>(s.N) * 4));
    if (copy) {
      const uint8_t* sp = static_cast<const uint8_t*>(s.payload->data);
      uint8_t* dp = static_cast<uint8_t*>(bump.host(const_cast<uint8_t*>(q.payload)));
      for (int64_t r = 0; r < s.N; ++r)
        std::memcpy(dp + static_cast<size_t>(r) * cn, sp + static_cast<size_t>(r) * s.K + c0,
                    static_cast<size_t>(cn));
      widen_bf16_to_f32(static_cast<const uint16_t*>(s.scales->data),
                        static_cast<float*>(bump.host(const_cast<float*>(q.scales))), s.N);
    }
    note_read(*s.payload_entry, static_cast<size_t>(s.N) * cn);
    note_read(*s.scale_entry, static_cast<size_t>(s.N) * 2);
    if (copy) {
      consumed(*s.payload);
      consumed(*s.scales);
    }
    return q;
  }

  // Channel-FP8 row segments concatenated (the GDN's merged in_proj_qkv):
  // payload and scale rows both concatenate row-for-row.
  GlmQuantMatrix merge_channel_segments(const std::string& module, int64_t total_rows, int64_t K,
                                        const std::vector<std::array<int64_t, 3>>& segs) {
    ChannelSource s = channel_source(module);
    if (K != s.K) fail(module + ": merged width disagrees with the checkpoint");
    GlmQuantMatrix q = channel_matrix(s, total_rows, K);
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(total_rows) * K));
    q.scales = static_cast<const float*>(bump.alloc(static_cast<size_t>(total_rows) * 4));
    if (copy) {
      uint8_t* dp = reinterpret_cast<uint8_t*>(bump.host(const_cast<uint8_t*>(q.payload)));
      float* ds = static_cast<float*>(bump.host(const_cast<float*>(q.scales)));
      const uint8_t* sp = static_cast<const uint8_t*>(s.payload->data);
      const uint16_t* ss = static_cast<const uint16_t*>(s.scales->data);
      for (const auto& sg : segs) {
        const int64_t src_row = sg[0], rows = sg[1], dst_row = sg[2];
        check_range(module, src_row, rows, s.N);
        std::memcpy(dp + static_cast<size_t>(dst_row) * K,
                    sp + static_cast<size_t>(src_row) * K, static_cast<size_t>(rows) * K);
        widen_bf16_to_f32(ss + static_cast<size_t>(src_row), ds + static_cast<size_t>(dst_row),
                          static_cast<size_t>(rows));
      }
      consumed(*s.payload);
      consumed(*s.scales);
    }
    for (const auto& sg : segs) {
      note_read(*s.payload_entry, static_cast<size_t>(sg[1]) * K);
      note_read(*s.scale_entry, static_cast<size_t>(sg[1]) * 2);
    }
    return q;
  }

  // An NVFP4 matrix (compressed-tensors naming): weight_packed U8 [N, K/2]
  // + weight_scale e4m3 [N, K/16]; the resident global slot holds
  // weight_global_scale itself — this release's global is already the
  // divisor the kernels apply to the finished dot once (models/
  // quant_matrix.hpp), so load_fp4_global stages it as is.
  struct Fp4Source {
    const QwenExpectedTensor* payload_entry = nullptr;
    const QwenExpectedTensor* scale_entry = nullptr;
    const TensorInfo* payload = nullptr;
    const TensorInfo* scales = nullptr;
    int64_t N = 0, K = 0;
  };

  Fp4Source fp4_source(const std::string& module) {
    const std::string pname = module + ".weight_packed";
    const QwenExpectedTensor& e = expected(pname);
    if (e.shape.size() != 2 || e.shape[1] <= 0) fail(pname + ": packed matrix needs 2 dims");
    const int64_t N = e.shape[0], K = e.shape[1] * 2;
    if (K % kFp4Group != 0) fail(pname + ": fp4 K must be a multiple of 16");
    const std::string sname = module + ".weight_scale";
    const QwenExpectedTensor& se = expected(sname);
    if (se.shape.size() != 2 || se.shape[0] != N || se.shape[1] != K / kFp4Group)
      fail(sname + ": the NVFP4 scale grid must be [N, K/16]");
    Fp4Source s;
    if (copy) {
      s.payload = &source(pname);
      s.scales = &source(sname);
    } else {
      s.payload = nullptr;
      s.scales = nullptr;
    }
    s.payload_entry = &e;
    s.scale_entry = &se;
    s.N = N;
    s.K = K;
    return s;
  }

  // weight_global_scale into a bump slot AS IS: the kernels divide the
  // finished dot by the slot once, and this release's global is a
  // weight-side divisor (per-layer RMS against the FP8 sibling release
  // matches at q*s_block/ws, within 6% on every layer; the reciprocal or
  // the product with the input global are orders of magnitude off). The
  // input_global_scale is the activation-side factor of the W4A4 recipe;
  // under W4A16 serving (exact BF16 activations) it has no subject, so it
  // is read for the byte reconciliation and never resident.
  void load_fp4_global(const std::string& module, float* slot) {
    const QwenExpectedTensor& eg = expected(module + ".weight_global_scale");
    const QwenExpectedTensor& ei = expected(module + ".input_global_scale");
    if (copy) {
      const TensorInfo& t = source(module + ".weight_global_scale");
      float ws2 = 0.f;
      std::memcpy(&ws2, t.data, 4);
      if (!(ws2 > 0.0f) || !std::isfinite(ws2))
        fail("'" + eg.name + "' is not a positive finite scale");
      *bump.host(slot) = ws2;
      consumed(t);
      consumed(source(module + ".input_global_scale"));
    }
    note_read(eg, 4);
    note_read(ei, 4);
  }

  GlmFp4Matrix load_fp4_rows(const std::string& module, int64_t r0, int64_t rn, float* global) {
    Fp4Source s = fp4_source(module);
    check_range(module, r0, rn, s.N);
    const int64_t pc = s.K / 2, sc = s.K / kFp4Group;
    GlmFp4Matrix q;
    q.rows = rn;
    q.cols = s.K;
    q.scale_group = kFp4Group;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rn) * pc));
    q.scales = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rn) * sc));
    if (copy) {
      std::memcpy(bump.host(const_cast<uint8_t*>(q.payload)),
                  static_cast<const uint8_t*>(s.payload->data) + static_cast<size_t>(r0) * pc,
                  static_cast<size_t>(rn) * pc);
      std::memcpy(bump.host(const_cast<uint8_t*>(q.scales)),
                  static_cast<const uint8_t*>(s.scales->data) + static_cast<size_t>(r0) * sc,
                  static_cast<size_t>(rn) * sc);
      consumed(*s.payload);
      consumed(*s.scales);
    }
    note_read(*s.payload_entry, static_cast<size_t>(rn) * pc);
    note_read(*s.scale_entry, static_cast<size_t>(rn) * sc);
    load_fp4_global(module, global);
    q.global_scale = global;
    return q;
  }

  // Whole scale blocks on the sliced axis (16-aligned bounds), every row's
  // payload and scale rows packed.
  GlmFp4Matrix load_fp4_cols(const std::string& module, int64_t c0, int64_t cn, float* global) {
    Fp4Source s = fp4_source(module);
    check_range(module + " cols", c0, cn, s.K);
    if (c0 % kFp4Group != 0 || cn % kFp4Group != 0)
      fail(module + ": fp4 column slice needs 16-aligned bounds");
    const int64_t pc = s.K / 2, lc = cn / 2, sc = s.K / kFp4Group, lsc = cn / kFp4Group;
    GlmFp4Matrix q;
    q.rows = s.N;
    q.cols = cn;
    q.scale_group = kFp4Group;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(s.N) * lc));
    q.scales = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(s.N) * lsc));
    if (copy) {
      const uint8_t* sp = static_cast<const uint8_t*>(s.payload->data);
      uint8_t* dp = static_cast<uint8_t*>(bump.host(const_cast<uint8_t*>(q.payload)));
      const uint8_t* ss = static_cast<const uint8_t*>(s.scales->data);
      uint8_t* ds = static_cast<uint8_t*>(bump.host(const_cast<uint8_t*>(q.scales)));
      for (int64_t r = 0; r < s.N; ++r) {
        std::memcpy(dp + static_cast<size_t>(r) * lc, sp + static_cast<size_t>(r) * pc + c0 / 2,
                    static_cast<size_t>(lc));
        std::memcpy(ds + static_cast<size_t>(r) * lsc, ss + static_cast<size_t>(r) * sc + c0 / 16,
                    static_cast<size_t>(lsc));
      }
      consumed(*s.payload);
      consumed(*s.scales);
    }
    note_read(*s.payload_entry, static_cast<size_t>(s.N) * lc);
    note_read(*s.scale_entry, static_cast<size_t>(s.N) * lsc);
    load_fp4_global(module, global);
    q.global_scale = global;
    return q;
  }

  // ---- ModelOpt NVFP4 (nvidia 122B MoE) --------------------------------------
  // base.weight U8 [N, K/2], base.weight_scale F8 [N, K/16],
  // base.weight_scale_2 F32 [] multiplier stored as its reciprocal (the
  // kernels divide once — loader.cpp load_global_reciprocal_into
  // precedent; compressed-tensors globals are divisors already).
  // base.input_scale consumed for reconciliation, never resident (W4A16).
  void load_global_mo_into(const std::string& base, float* slot) {
    const QwenExpectedTensor& eg = expected(base + ".weight_scale_2");
    if (copy) {
      const TensorInfo& t = source(eg.name);
      float ws2 = 0.f;
      std::memcpy(&ws2, t.data, 4);
      if (!(ws2 > 0.0f) || !std::isfinite(ws2)) fail("'" + eg.name + "' is not positive finite");
      const float inv = 1.0f / ws2;
      std::memcpy(bump.host(slot), &inv, 4);
      consumed(t);
      consumed(source(base + ".input_scale"));
    }
    note_read(eg, 4);
    const QwenExpectedTensor& ei = expected(base + ".input_scale");
    note_read(ei, 4);
  }

  GlmFp4Matrix load_fp4_mo_rows(const std::string& base, int64_t r0, int64_t rn,
                                float* global) {
    const QwenExpectedTensor& ep = expected(base + ".weight");
    const QwenExpectedTensor& es = expected(base + ".weight_scale");
    const int64_t N = ep.shape[0], K = ep.shape[1] * 2;
    if (K % kFp4Group != 0) fail(base + ": fp4 K must be a multiple of 16");
    if (es.shape.size() != 2 || es.shape[0] != N || es.shape[1] != K / kFp4Group)
      fail(base + ".weight_scale: NVFP4 scale grid must be [N, K/16]");
    check_range(base, r0, rn, N);
    const int64_t pc = K / 2, sc = K / kFp4Group;
    GlmFp4Matrix q;
    q.rows = rn;
    q.cols = K;
    q.scale_group = kFp4Group;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rn) * pc));
    q.scales = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rn) * sc));
    if (copy) {
      std::memcpy(bump.host(const_cast<uint8_t*>(q.payload)),
                  static_cast<const uint8_t*>(source(ep.name).data) + static_cast<size_t>(r0) * pc,
                  static_cast<size_t>(rn) * pc);
      std::memcpy(bump.host(const_cast<uint8_t*>(q.scales)),
                  static_cast<const uint8_t*>(source(es.name).data) + static_cast<size_t>(r0) * sc,
                  static_cast<size_t>(rn) * sc);
      consumed(source(ep.name));
      consumed(source(es.name));
    }
    note_read(ep, static_cast<size_t>(rn) * pc);
    note_read(es, static_cast<size_t>(rn) * sc);
    load_global_mo_into(base, global);
    q.global_scale = global;
    return q;
  }

  GlmFp4Matrix load_fp4_mo_cols(const std::string& base, int64_t c0, int64_t cn,
                                float* global) {
    const QwenExpectedTensor& ep = expected(base + ".weight");
    const QwenExpectedTensor& es = expected(base + ".weight_scale");
    const int64_t N = ep.shape[0], K = ep.shape[1] * 2;
    if (K % kFp4Group != 0) fail(base + ": fp4 K must be a multiple of 16");
    check_range(base + " cols", c0, cn, K);
    if (c0 % kFp4Group != 0 || cn % kFp4Group != 0)
      fail(base + ": fp4 column slice needs 16-aligned bounds");
    if (es.shape.size() != 2 || es.shape[0] != N || es.shape[1] != K / kFp4Group)
      fail(base + ".weight_scale: NVFP4 scale grid must be [N, K/16]");
    const int64_t pc = K / 2, lc = cn / 2, sc = K / kFp4Group, lsc = cn / kFp4Group;
    GlmFp4Matrix q;
    q.rows = N;
    q.cols = cn;
    q.scale_group = kFp4Group;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(N) * lc));
    q.scales = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(N) * lsc));
    if (copy) {
      const uint8_t* sp = static_cast<const uint8_t*>(source(ep.name).data);
      uint8_t* dp = static_cast<uint8_t*>(bump.host(const_cast<uint8_t*>(q.payload)));
      const uint8_t* ss = static_cast<const uint8_t*>(source(es.name).data);
      uint8_t* ds = static_cast<uint8_t*>(bump.host(const_cast<uint8_t*>(q.scales)));
      for (int64_t r = 0; r < N; ++r) {
        std::memcpy(dp + static_cast<size_t>(r) * lc, sp + static_cast<size_t>(r) * pc + c0 / 2,
                    static_cast<size_t>(lc));
        std::memcpy(ds + static_cast<size_t>(r) * lsc, ss + static_cast<size_t>(r) * sc + c0 / 16,
                    static_cast<size_t>(lsc));
      }
      consumed(source(ep.name));
      consumed(source(es.name));
    }
    note_read(ep, static_cast<size_t>(N) * lc);
    note_read(es, static_cast<size_t>(N) * lsc);
    load_global_mo_into(base, global);
    q.global_scale = global;
    return q;
  }

  // A full/GDN projection's row slice in its release's format (the MLP's
  // NVFP4/BF16 forms do not reach here: form_of refuses NVFP4 off the MLP,
  // and the draft layer's BF16 matrices do).
  GlmQuantMatrix load_attn_rows(const std::string& module, int64_t r0, int64_t rn,
                                const uint16_t** bf16_out) {
    switch (form_of(module)) {
      case Form::BlockFp8:
        return load_fp8_native_rows(module + ".weight", r0, rn);
      case Form::ChannelFp8:
        return load_channel_rows(module, r0, rn);
      case Form::Bf16:
        *bf16_out = load_bf16_rows(module + ".weight", r0, rn);
        return {};
      case Form::Nvfp4: break;  // form_of refused
    }
    return {};
  }

  GlmQuantMatrix load_attn_cols(const std::string& module, int64_t c0, int64_t cn,
                                const uint16_t** bf16_out) {
    switch (form_of(module)) {
      case Form::BlockFp8:
        return load_fp8_native_cols(module + ".weight", c0, cn);
      case Form::ChannelFp8:
        return load_channel_cols(module, c0, cn);
      case Form::Bf16:
        *bf16_out = load_bf16_cols(module + ".weight", c0, cn);
        return {};
      case Form::Nvfp4: break;  // form_of refused
    }
    return {};
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
    a.q_proj_fp8 = load_attn_rows(p + "self_attn.q_proj", q0, qn, &a.q_proj);
    a.k_proj_fp8 = load_attn_rows(p + "self_attn.k_proj", kv0, kvn, &a.k_proj);
    a.v_proj_fp8 = load_attn_rows(p + "self_attn.v_proj", kv0, kvn, &a.v_proj);
    a.o_proj_fp8 = load_attn_cols(p + "self_attn.o_proj", o0, on, &a.o_proj);
    a.q_norm = load_bf16(p + "self_attn.q_norm.weight");
    a.k_norm = load_bf16(p + "self_attn.k_norm.weight");
    // The mixed release's FP8 kv_cache_scheme scalars: bound for the
    // binding check, read for the byte reconciliation, never resident.
    if (cfg.quant_kind == Qwen35QuantKind::Nvfp4Mixed && out.layer != cfg.mtp_layer()) {
      load_raw(p + "self_attn.k_scale");
      load_raw(p + "self_attn.v_scale");
    }
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
    const std::string qkv_mod = p + "linear_attn.in_proj_qkv";
    const std::vector<std::array<int64_t, 3>> segs = {
        {r * lk * dk, lk * dk, 0},
        {K + r * lk * dk, lk * dk, lk * dk},
        {2 * K + r * lv * dv, lv * dv, 2 * lk * dk}};
    switch (form_of(qkv_mod)) {
      case Form::BlockFp8:
        g.in_proj_qkv_fp8 = merge_fp8_segments(qkv_mod + ".weight", local_rows, H, segs);
        break;
      case Form::ChannelFp8:
        g.in_proj_qkv_fp8 = merge_channel_segments(qkv_mod, local_rows, H, segs);
        break;
      case Form::Bf16:
        // BF16 GDN (the MoE release: every non-expert matrix is BF16).
        // Concatenate the three row segments; the resident is plain BF16.
        g.in_proj_qkv = merge_bf16_segments(qkv_mod + ".weight", local_rows, H, segs);
        break;
      case Form::Nvfp4: break;  // form_of refused
    }
    if (copy) consumed(source(qkv_mod + ".weight"));
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
    g.in_proj_z_fp8 = load_attn_rows(p + "linear_attn.in_proj_z", r * lv * dv, lv * dv,
                                     &g.in_proj_z);
    g.in_proj_a = load_bf16_rows(p + "linear_attn.in_proj_a.weight", r * lv, lv);
    g.in_proj_b = load_bf16_rows(p + "linear_attn.in_proj_b.weight", r * lv, lv);
    g.a_log = load_bf16_as_f32(p + "linear_attn.A_log", r * lv, lv);
    g.dt_bias = load_bf16_as_f32(p + "linear_attn.dt_bias", r * lv, lv);
    g.norm = load_bf16(p + "linear_attn.norm.weight");
    g.out_proj_fp8 =
        load_attn_cols(p + "linear_attn.out_proj", r * lv * dv, lv * dv, &g.out_proj);
  }

  void build_moe(const std::string& p) {
    // ModelOpt NVFP4 MoE (nvidia 122B backbone): router + shared BF16,
    // routed gate/up rows [r*I, I), down cols [r*I, I), globals reciprocal.
    // MTP draft (BF16 experts): engine.mtp_expert_format=bf16 below; the
    // dense variant has no MoE draft at all.
    if (!cfg.is_moe) fail("build_moe: dense variant carries no MoE");
    const bool is_mtp = out.layer == cfg.mtp_layer();
    QwenMoeResident& m = out.moe;
    const int64_t I = geo.local_moe_inter, S = geo.local_shared_inter;
    const int64_t r = geo.rank;
    const int64_t E = cfg.num_experts;
    m.local_inter = I;
    m.local_shared_inter = S;
    m.router = load_bf16(p + "mlp.gate.weight");
    m.shared_gate = load_bf16(p + "mlp.shared_expert_gate.weight");
    m.shared[0] = load_bf16_rows(p + "mlp.shared_expert.gate_proj.weight", r * S, S);
    m.shared[1] = load_bf16_rows(p + "mlp.shared_expert.up_proj.weight", r * S, S);
    m.shared[2] = load_bf16_cols(p + "mlp.shared_expert.down_proj.weight", r * S, S);
    if (is_mtp) {
      // The draft's BF16 experts: engine.mtp_expert_format=bf16 encodes
      // each expert's slice to block FP8 at load (the fused-release
      // precedent); the proposals are lossy, the verify exact. Any other
      // value cannot serve the BF16 draft — refused by key name on a real
      // load (sizing walks every layer, served or not, so it measures).
      if (Qwen35LayerStream::mtp_expert_format() != "bf16" && copy)
        fail("mtp draft MoE needs engine.mtp_expert_format=bf16 (the checkpoint holds "
             "per-expert BF16, encoded to block FP8 at load for the proposals only)");
      m.experts.resize(static_cast<size_t>(E) * 3);
      m.scale_block = 128;
      for (int64_t e = 0; e < E; ++e) {
        const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
        m.experts[static_cast<size_t>(e) * 3 + 0] =
            load_bf16_rows_fp8(ep + "gate_proj.weight", r * I, I);
        m.experts[static_cast<size_t>(e) * 3 + 1] =
            load_bf16_rows_fp8(ep + "up_proj.weight", r * I, I);
        m.experts[static_cast<size_t>(e) * 3 + 2] =
            load_bf16_cols_fp8(ep + "down_proj.weight", r * I, I);
      }
      return;
    }
    m.experts_fp4.resize(static_cast<size_t>(E) * 3);
    m.expert_globals = static_cast<float*>(bump.alloc(static_cast<size_t>(E) * 3 * sizeof(float)));
    for (int64_t e = 0; e < E; ++e) {
      const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
      float* gb = m.expert_globals + e * 3;
      m.experts_fp4[static_cast<size_t>(e) * 3 + 0] =
          load_fp4_mo_rows(ep + "gate_proj", r * I, I, gb + 0);
      m.experts_fp4[static_cast<size_t>(e) * 3 + 1] =
          load_fp4_mo_rows(ep + "up_proj", r * I, I, gb + 1);
      m.experts_fp4[static_cast<size_t>(e) * 3 + 2] =
          load_fp4_mo_cols(ep + "down_proj", r * I, I, gb + 2);
    }
    // W4A4 input-scale maxima (loader.cpp precedent; unused under W4A16,
    // kept for the byte reconciliation via load_global_mo_into).
    m.act_scale_w13 = 0.0f;
    m.act_scale_w2 = 0.0f;
  }

  void build_mlp(const std::string& p) {
    const int64_t li = geo.local_inter;
    const int64_t r0 = static_cast<int64_t>(geo.rank) * li;
    Qwen35DenseMlpResident& m = out.mlp;
    m.inter = li;
    const std::string gate = p + "mlp.gate_proj", up = p + "mlp.up_proj",
                      down = p + "mlp.down_proj";
    if (cfg.quant_kind == Qwen35QuantKind::Fp8Block) {
      m.gate_fp8 = load_fp8_native_rows(gate + ".weight", r0, li);
      m.up_fp8 = load_fp8_native_rows(up + ".weight", r0, li);
      m.down_fp8 = load_fp8_native_cols(down + ".weight", r0, li);
      return;
    }
    // The mixed release: the fp4 global slots only exist for this release
    // (the FP8 release's resident image layout must not shift).
    float* globals = static_cast<float*>(bump.alloc(3 * sizeof(float)));
    if (copy) {
      float* g = bump.host(globals);
      g[0] = g[1] = g[2] = 1.0f;  // only the fp4 matrices overwrite these
    }
    const std::string rows_mods[2] = {gate, up};
    for (int i = 0; i < 2; ++i) {
      switch (cfg.tensor_quant(rows_mods[i])) {
        case Qwen35TensorQuant::Bf16:
          *(i == 0 ? &m.gate : &m.up) = load_bf16_rows(rows_mods[i] + ".weight", r0, li);
          break;
        case Qwen35TensorQuant::Fp8Channel:
          *(i == 0 ? &m.gate_fp8 : &m.up_fp8) = load_channel_rows(rows_mods[i], r0, li);
          break;
        case Qwen35TensorQuant::Nvfp4:
          *(i == 0 ? &m.gate_fp4 : &m.up_fp4) = load_fp4_rows(rows_mods[i], r0, li, globals + i);
          break;
      }
    }
    // down_proj slices its K (the SwiGLU's intermediate axis) as columns.
    switch (cfg.tensor_quant(down)) {
      case Qwen35TensorQuant::Bf16:
        m.down = load_bf16_cols(down + ".weight", r0, li);
        break;
      case Qwen35TensorQuant::Fp8Channel:
        m.down_fp8 = load_channel_cols(down, r0, li);
        break;
      case Qwen35TensorQuant::Nvfp4:
        m.down_fp4 = load_fp4_cols(down, r0, li, globals + 2);
        break;
    }
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
    if (cfg.is_moe)
      build_moe(p);
    else
      build_mlp(p);
    if (loader_verbose())
      std::fprintf(stderr, "[qwen35] layer %d done (kind=%d)\n", layer, (int)kind);
  }
};

const char* Qwen35LoaderFamily::who() { return "qwen35 loader"; }

// 5: MoE MTP draft encoded (engine.mtp_expert_format=bf16 carries bit 8).
// 4: MoE variant adds routed/shared slices (Nvfp4Moe, ModelOpt naming).
// 3: the NVFP4 global slot holds the weight-side divisor itself (2026-10-06;
// the 2-era images hold its reciprocal, and the kernels divide by the slot).
uint64_t Qwen35LoaderFamily::loader_format() {
  return 4 | (Qwen35LayerStream::mtp_expert_format() == "bf16" ? 8 : 0);
}

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
         e.cls == QwenWeightClass::DenseMlp || e.cls == QwenWeightClass::Router ||
         e.cls == QwenWeightClass::SharedExpert || e.cls == QwenWeightClass::RoutedExpert;
}

size_t Qwen35LoaderFamily::globals_bytes(const Config& c, int rank, int world,
                                         LoaderHeadSharding) {
  const size_t H = static_cast<size_t>(c.hidden_size);
  const size_t V = static_cast<size_t>(c.vocab_size);
  const size_t Vn = static_cast<size_t>(V * (rank + 1) / world) - static_cast<size_t>(V * rank / world);
  size_t b = 0;
  b += align_up_256(V * H * 2);   // embed
  if (c.quant_kind == Qwen35QuantKind::Nvfp4Mixed &&
      c.tensor_quant("lm_head") == Qwen35TensorQuant::Fp8Channel)
    b += align_up_256(Vn * H) + align_up_256(Vn * 4);  // head: e4m3 + F32 rows
  else
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
  out.lm_vocab_begin = geo.lm_vocab_begin;
  out.lm_vocab_count = geo.lm_vocab_count;
  if (c.quant_kind == Qwen35QuantKind::Nvfp4Mixed &&
      c.tensor_quant("lm_head") == Qwen35TensorQuant::Fp8Channel) {
    // The mixed release's native head: e4m3 rows + BF16 [N, 1] scales
    // widened to F32 (channel grid, scale_block_rows = 1).
    const TensorInfo& lm = lookup("lm_head.weight");
    const TensorInfo& ls = lookup("lm_head.weight_scale");
    uint8_t* payload = static_cast<uint8_t*>(bump.alloc(static_cast<size_t>(Vn) * H));
    std::memcpy(bump.host(payload), static_cast<const uint8_t*>(lm.data) + V0 * H,
                static_cast<size_t>(Vn) * H);
    float* scales = static_cast<float*>(bump.alloc(static_cast<size_t>(Vn) * 4));
    const uint16_t* ss = static_cast<const uint16_t*>(ls.data) + V0;
    float* hs = bump.host(scales);
    for (int64_t i = 0; i < Vn; ++i) hs[i] = bf16_bits_to_float(ss[i]);
    source_bytes += static_cast<uint64_t>(Vn) * H + static_cast<uint64_t>(Vn) * 2;
    out.lm_head_fp8.payload = payload;
    out.lm_head_fp8.scales = scales;
    out.lm_head_fp8.rows = Vn;
    out.lm_head_fp8.cols = static_cast<int>(H);
    out.lm_head_fp8.scale_block_rows = 1;
    out.lm_head_fp8.scale_block_cols = static_cast<int>(pow2_ceil(static_cast<int64_t>(H)));
    out.lm_head = nullptr;
  } else {
    const TensorInfo& lm = lookup("lm_head.weight");
    uint16_t* head = static_cast<uint16_t*>(bump.alloc(static_cast<size_t>(Vn) * H * 2));
    std::memcpy(reinterpret_cast<uint8_t*>(bump.host(head)),
                static_cast<const uint8_t*>(lm.data) + static_cast<size_t>(V0) * H * 2,
                static_cast<size_t>(Vn) * H * 2);
    source_bytes += static_cast<uint64_t>(Vn) * H * 2;
    verbatim_bytes += static_cast<uint64_t>(Vn) * H * 2;
    out.lm_head = head;
  }
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

std::string& mtp_expert_format_storage_35() {
  static std::string fmt = "fp8";
  return fmt;
}

void Qwen35LayerStream::set_mtp_expert_format(const std::string& fmt) {
  if (fmt != "fp8" && fmt != "bf16_fused" && fmt != "bf16")
    throw std::invalid_argument("qwen35 mtp_expert_format: fp8 | bf16_fused | bf16");
  mtp_expert_format_storage_35() = fmt;
}

const std::string& Qwen35LayerStream::mtp_expert_format() {
  return mtp_expert_format_storage_35();
}

const std::string& Qwen35LayerStream::resident_image_dir() { return resident_image_dir_storage_35(); }

const std::string& Qwen35LayerStream::image_dir() const { return resident_image_dir(); }

template class ResidentLayerStream<Qwen35LoaderFamily>;

}  // namespace dgpp
