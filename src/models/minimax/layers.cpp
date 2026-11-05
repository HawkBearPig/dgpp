#include "models/minimax/layers.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/minimax.hpp"
#include "kernels/minimax_native_attention.hpp"
#include "kernels/minimax_native_moe.hpp"
#include "kernels/moe_w4a4.hpp"
#include "kernels/mxfp8_gemm.hpp"
#include "kernels/qsa.hpp"
#include "kernels/scale_gemm.hpp"
#include "kernels/sink_attn.hpp"

namespace dgpp {
namespace {
struct CudaDelete {
  void operator()(void* p) const { cudaFree(p); }
};
struct Scratch {
  std::vector<std::unique_ptr<void, CudaDelete>> buffers;
  template <class T>
  T* alloc(size_t count) {
    void* p = nullptr;
    DGPP_CUDA_OK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T)));
    buffers.emplace_back(p);
    return static_cast<T*>(p);
  }
};
void linear(const uint16_t* x, const GlmQuantMatrix& w, uint16_t* y, int rows, cudaStream_t stream,
            const uint8_t* codes = nullptr, const uint8_t* scales = nullptr) {
  if (codes) {
    launch_mxfp8_gemm_bf16(codes, scales, w.payload, w.scales_e8m0, y, rows,
                           static_cast<int>(w.rows), static_cast<int>(w.cols), stream);
    return;
  }
  launch_scale_gemm_grid_e8m0_bf16(x, w.cols, w.payload, w.scales_e8m0, y, rows,
                                   static_cast<int>(w.rows), static_cast<int>(w.cols), stream,
                                   w.rows, 0, 5, nullptr, 0, false);
}
void row_count(int n, int limit) {
  if (n <= 0 || n > limit)
    throw std::invalid_argument("MiniMax layer: row count outside scratch capacity");
}
}  // namespace

struct MinimaxAttentionLayer::Impl : Scratch {
  MinimaxConfig cfg;
  MinimaxLocalGeometry geo;
  int max_tokens, max_blocks;
  bool native;
  LatentFormat format;
  uint8_t *codes = nullptr, *scales = nullptr, *q8 = nullptr, *iq8 = nullptr;
  float* inv;
  uint16_t *q, *k, *v, *iq, *ik, *attn;
  float *scores, *m, *l, *c;
  int32_t* selected;
  Impl(const MinimaxConfig& cfg_, const MinimaxLocalGeometry& geo_, int tokens, int64_t cache,
       bool native_, LatentFormat format_)
      : cfg(cfg_),
        geo(geo_),
        max_tokens(tokens),
        max_blocks(static_cast<int>(cache / 128)),
        native(native_),
        format(format_) {
    if (tokens <= 0 || cache <= 0 || cache % 128)
      throw std::invalid_argument("MiniMax attention capacity");
    const size_t t = static_cast<size_t>(tokens);
    if (native) {
      const size_t width = std::max<int64_t>(cfg.hidden_size, geo.local_heads * 128);
      codes = alloc<uint8_t>(t * width);
      scales = alloc<uint8_t>(t * width / 32);
    }
    if (format == LatentFormat::kFp4) {
      q8 = alloc<uint8_t>(t * geo.local_heads * 128);
      iq8 = alloc<uint8_t>(t * geo.local_index_heads * 128);
    }
    q = alloc<uint16_t>(t * geo.local_heads * 128);
    k = alloc<uint16_t>(t * geo.local_kv_heads * 128);
    v = alloc<uint16_t>(t * geo.local_kv_heads * 128);
    iq = alloc<uint16_t>(t * geo.local_index_heads * 128);
    ik = alloc<uint16_t>(t * 128);
    attn = alloc<uint16_t>(t * geo.local_heads * 128);
    scores = alloc<float>(t * geo.local_index_heads * max_blocks);
    selected = alloc<int32_t>(t * geo.local_index_heads * cfg.sparse_topk_blocks);
    const size_t partial = t * cfg.sparse_topk_blocks * geo.local_heads;
    m = alloc<float>(partial);
    l = alloc<float>(partial);
    c = alloc<float>(partial * 128);
    inv = alloc<float>(cfg.rotary_dim / 2);
    minimax_rope_inv_freq(inv, cfg.rotary_dim / 2, cfg.rotary_dim, cfg.rope_theta, nullptr);
    DGPP_CUDA_OK(cudaStreamSynchronize(nullptr));
  }
};
MinimaxAttentionLayer::MinimaxAttentionLayer(const MinimaxConfig& c, const MinimaxLocalGeometry& g,
                                             int t, int64_t cache, bool native, LatentFormat format)
    : impl_(std::make_unique<Impl>(c, g, t, cache, native, format)) {}
MinimaxAttentionLayer::~MinimaxAttentionLayer() = default;
size_t MinimaxAttentionLayer::scratch_bytes(const MinimaxConfig& c, const MinimaxLocalGeometry& g,
                                            int t, int64_t cache, bool native,
                                            LatentFormat format) {
  const size_t T = static_cast<size_t>(t), P = T * c.sparse_topk_blocks * g.local_heads;
  return T * (2 * g.local_heads + 2 * g.local_kv_heads + g.local_index_heads + 1) * 128 * 2 +
         T * g.local_index_heads * (cache / 128) * 4 +
         T * g.local_index_heads * c.sparse_topk_blocks * 4 + P * (128 + 2) * 4 +
         c.rotary_dim / 2 * 4 +
         (native ? T * std::max<int64_t>(c.hidden_size, g.local_heads * 128) * 33 / 32 : 0) +
         (format == LatentFormat::kFp4 ? T * (g.local_heads + g.local_index_heads) * 128 : 0);
}
void MinimaxAttentionLayer::enqueue(const MinimaxAttnResident& w, const uint16_t* x, int tokens,
                                    const PagedAttentionRows& rows, const MinimaxKvCache& cache,
                                    uint16_t* out, cudaStream_t stream) {
  auto& a = *impl_;
  row_count(tokens, a.max_tokens);
  if (rows.split(x, tokens, a.cfg.hidden_size, out,
                 [&](const uint16_t* in, int n, const auto& map, uint16_t* dst) {
                   enqueue(w, in, n, map, cache, dst, stream);
                 }))
    return;
  if (cache.block_tokens != 128 || w.local_index_heads != w.local_kv_heads ||
      cache.format != a.format)
    throw std::invalid_argument("MiniMax attention: incompatible cache/index geometry");
  if (a.native)
    launch_mxfp8_quantize_rows(x, a.cfg.hidden_size, tokens, a.cfg.hidden_size, a.codes, a.scales,
                               stream);
  linear(x, w.q, a.q, tokens, stream, a.codes, a.scales);
  linear(x, w.k, a.k, tokens, stream, a.codes, a.scales);
  linear(x, w.v, a.v, tokens, stream, a.codes, a.scales);
  linear(x, w.index_q, a.iq, tokens, stream, a.codes, a.scales);
  linear(x, w.index_k, a.ik, tokens, stream, a.codes, a.scales);
  auto finish = [&](uint16_t* p, const uint16_t* norm, int heads) {
    minimax_norm_rope(p, norm, p, rows.pos, a.inv, tokens, heads, a.cfg.rotary_dim,
                      a.cfg.rms_norm_eps, stream);
  };
  finish(a.q, w.q_norm, w.local_heads);
  finish(a.k, w.k_norm, w.local_kv_heads);
  finish(a.iq, w.index_q_norm, w.local_index_heads);
  finish(a.ik, w.index_k_norm, 1);
  if (a.format == LatentFormat::kFp4) {
    minimax_cache_nvfp4_append(a.k, cache.k_nvfp4, rows.req_ids, rows.pos, cache.block_tables,
                               cache.blocks_per_request, tokens, w.local_kv_heads, stream);
    minimax_cache_nvfp4_append(a.v, cache.v_nvfp4, rows.req_ids, rows.pos, cache.block_tables,
                               cache.blocks_per_request, tokens, w.local_kv_heads, stream);
    minimax_cache_nvfp4_append(a.ik, cache.index_nvfp4, rows.req_ids, rows.pos, cache.block_tables,
                               cache.blocks_per_request, tokens, 1, stream);
  } else {
    qsa_kv_append(a.k, w.local_kv_heads * 128, a.v, w.local_kv_heads * 128, rows.req_ids, rows.pos,
                  tokens, cache.block_tables, cache.blocks_per_request, 128, w.local_kv_heads, 128,
                  cache.k_cache, cache.v_cache, nullptr, nullptr, stream);
    minimax_index_append(a.ik, cache.index_cache, rows.req_ids, rows.pos, cache.block_tables,
                         cache.blocks_per_request, tokens, stream);
  }
  if (rows.cache_only) return;
  if (a.format == LatentFormat::kFp4) {
    minimax_query_fp8(a.q, a.q8, static_cast<int64_t>(tokens) * w.local_heads * 128, stream);
    minimax_query_fp8(a.iq, a.iq8, static_cast<int64_t>(tokens) * w.local_index_heads * 128,
                      stream);
    minimax_index_nvfp4_score(a.iq8, cache.index_nvfp4, rows.req_ids, rows.pos, cache.block_tables,
                              cache.blocks_per_request, tokens, w.local_index_heads, a.max_blocks,
                              a.scores, stream);
  } else {
    minimax_index_score(a.iq, cache.index_cache, rows.req_ids, rows.pos, cache.block_tables,
                        cache.blocks_per_request, tokens, w.local_index_heads, a.max_blocks,
                        a.scores, stream);
  }
  minimax_index_select(a.scores, rows.pos, tokens, w.local_index_heads, a.max_blocks,
                       a.cfg.sparse_topk_blocks, a.selected, stream);
  if (a.format == LatentFormat::kFp4 && rows.decode) {
    minimax_attn_nvfp4_decode(a.q8, cache.k_nvfp4, cache.v_nvfp4, rows.req_ids, rows.pos,
                              cache.block_tables, cache.blocks_per_request, a.selected, tokens,
                              w.local_heads, w.local_kv_heads, a.cfg.sparse_topk_blocks, a.attn,
                              stream);
  } else if (a.format == LatentFormat::kFp4) {
    minimax_attn_nvfp4_partial(a.q8, cache.k_nvfp4, cache.v_nvfp4, rows.req_ids, rows.pos,
                               cache.block_tables, cache.blocks_per_request, a.selected, tokens,
                               w.local_heads, w.local_kv_heads, a.cfg.sparse_topk_blocks, a.m, a.l,
                               a.c, stream);
  } else {
    minimax_attn_partial(a.q, cache.k_cache, cache.v_cache, rows.req_ids, rows.pos,
                         cache.block_tables, cache.blocks_per_request, a.selected, tokens,
                         w.local_heads, w.local_kv_heads, a.cfg.sparse_topk_blocks, a.m, a.l, a.c,
                         stream);
  }
  if (a.format != LatentFormat::kFp4 || !rows.decode)
    sink_attn_combine(a.m, a.l, a.c, tokens, a.cfg.sparse_topk_blocks, w.local_heads, a.attn, stream);
  if (a.native)
    launch_mxfp8_quantize_rows(a.attn, w.local_heads * 128, tokens, w.local_heads * 128, a.codes,
                               a.scales, stream);
  linear(a.attn, w.o, out, tokens, stream, a.codes, a.scales);
}

struct MinimaxMlp::Impl : Scratch {
  int inter, max_tokens;
  float alpha, limit;
  uint16_t *gate, *up, *act;
  uint8_t *codes = nullptr, *scales = nullptr;
  Impl(const MinimaxConfig& c, int i, int t, bool native)
      : inter(i), max_tokens(t), alpha(c.swiglu_alpha), limit(c.swiglu_limit) {
    if (i <= 0 || i % 64 || t <= 0)
      throw std::invalid_argument("MiniMax MLP: requires positive 64-aligned intermediate slice");
    gate = alloc<uint16_t>(static_cast<size_t>(t) * i);
    up = alloc<uint16_t>(static_cast<size_t>(t) * i);
    act = alloc<uint16_t>(static_cast<size_t>(t) * i);
    if (native) {
      const size_t width = std::max<int64_t>(c.hidden_size, i);
      codes = alloc<uint8_t>(static_cast<size_t>(t) * width);
      scales = alloc<uint8_t>(static_cast<size_t>(t) * width / 32);
    }
  }
};
MinimaxMlp::MinimaxMlp(const MinimaxConfig& c, int i, int t, bool native)
    : impl_(std::make_unique<Impl>(c, i, t, native)) {}
MinimaxMlp::~MinimaxMlp() = default;
size_t MinimaxMlp::scratch_bytes(const MinimaxConfig& c, int i, int t, bool native) {
  return static_cast<size_t>(i) * t * 6 +
         (native ? static_cast<size_t>(t) * std::max<int64_t>(c.hidden_size, i) * 33 / 32 : 0);
}
void MinimaxMlp::enqueue(const MinimaxMlpResident& w, const uint16_t* x, int tokens, uint16_t* out,
                         cudaStream_t stream) {
  auto& m = *impl_;
  row_count(tokens, m.max_tokens);
  if (w.local_inter != m.inter) throw std::invalid_argument("MiniMax MLP changed geometry");
  if (m.codes)
    launch_mxfp8_quantize_rows(x, w.gate.cols, tokens, w.gate.cols, m.codes, m.scales, stream);
  linear(x, w.gate, m.gate, tokens, stream, m.codes, m.scales);
  linear(x, w.up, m.up, tokens, stream, m.codes, m.scales);
  if (m.codes)
    launch_minimax_swiglu_mxfp8_rows(m.gate, m.up, m.inter, tokens, m.inter, m.alpha, m.limit,
                                     m.codes, m.scales, stream);
  else
    minimax_swiglu(m.gate, m.up, m.act, static_cast<int64_t>(tokens) * m.inter, m.alpha, m.limit,
                   stream);
  linear(m.act, w.down, out, tokens, stream, m.codes, m.scales);
}

struct MinimaxMoeLayer::Impl : Scratch {
  MinimaxConfig cfg;
  int inter, max_tokens;
  bool native;
  MinimaxMlp shared;
  float *scores, *weights, *down = nullptr;
  float *gate_f32 = nullptr, *up_f32 = nullptr, *input_globals = nullptr, *act_globals = nullptr;
  uint8_t *input_codes = nullptr, *input_scales = nullptr, *act_codes = nullptr,
          *act_scales = nullptr;
  int32_t *ids, *rows, *slot_row;
  uint16_t *gate = nullptr, *up = nullptr, *act, *shared_out, *down_bf16 = nullptr;
  MoeSegment *segs, *compact_segs;
  MoeExpertView* views;
  std::vector<bool> prepared;
  Impl(const MinimaxConfig& c, const MinimaxLocalGeometry& g, int t, bool native_)
      : cfg(c),
        inter(static_cast<int>(g.local_inter)),
        max_tokens(t),
        native(native_),
        shared(c, static_cast<int>(g.local_shared_inter), t, native),
        prepared(static_cast<size_t>(c.num_hidden_layers), false) {
    const size_t T = static_cast<size_t>(t), K = c.num_experts_per_tok, E = c.num_local_experts,
                 H = c.hidden_size;
    scores = alloc<float>(T * E);
    weights = alloc<float>(T * K);
    ids = alloc<int32_t>(T * K);
    rows = alloc<int32_t>(T * (K + 1));
    slot_row = alloc<int32_t>(T * K);
    segs = alloc<MoeSegment>(E + 1);
    compact_segs = alloc<MoeSegment>(E);
    act = alloc<uint16_t>(T * K * inter);
    if (native) {
      if (H % 128 || inter % 128)
        throw std::invalid_argument("MiniMax native experts require 128-aligned TP slices");
      gate_f32 = alloc<float>(T * K * inter);
      up_f32 = alloc<float>(T * K * inter);
      down_bf16 = alloc<uint16_t>(T * K * H);
      input_codes = alloc<uint8_t>(T * H / 2);
      input_scales = alloc<uint8_t>(T * H / 16);
      input_globals = alloc<float>(T);
      act_codes = alloc<uint8_t>(T * K * inter / 2);
      act_scales = alloc<uint8_t>(T * K * inter / 16);
      act_globals = alloc<float>(T * K);
    } else {
      gate = alloc<uint16_t>(T * K * inter);
      up = alloc<uint16_t>(T * K * inter);
      down = alloc<float>(T * K * H);
    }
    shared_out = alloc<uint16_t>(T * H);
    views = alloc<MoeExpertView>(static_cast<size_t>(c.num_hidden_layers) * E * 3);
  }
};
MinimaxMoeLayer::MinimaxMoeLayer(const MinimaxConfig& c, const MinimaxLocalGeometry& g, int t,
                                 bool native)
    : impl_(std::make_unique<Impl>(c, g, t, native)) {}
MinimaxMoeLayer::~MinimaxMoeLayer() = default;
size_t MinimaxMoeLayer::scratch_bytes(const MinimaxConfig& c, const MinimaxLocalGeometry& g, int t,
                                      bool native) {
  const size_t T = static_cast<size_t>(t), K = c.num_experts_per_tok, E = c.num_local_experts,
               H = c.hidden_size;
  return T * E * 4 + T * K * 4 * 3 + T * (K + 1) * 4 + (2 * E + 1) * sizeof(MoeSegment) +
         T * K * g.local_inter * (native ? 10 : 6) + T * K * H * (native ? 2 : 4) + T * H * 2 +
         (native ? (T * H + T * K * g.local_inter) * 9 / 16 + T * (K + 1) * 4 : 0) +
         static_cast<size_t>(c.num_hidden_layers) * E * 3 * sizeof(MoeExpertView) +
         MinimaxMlp::scratch_bytes(c, static_cast<int>(g.local_shared_inter), t, native);
}
void MinimaxMoeLayer::prepare_table(int layer, const MinimaxMoeResident& w, cudaStream_t stream) {
  auto& m = *impl_;
  if (layer < 0 || layer >= m.cfg.num_hidden_layers ||
      w.experts.size() != static_cast<size_t>(m.cfg.num_local_experts) * 3)
    throw std::invalid_argument("MiniMax expert table geometry");
  if (m.prepared[layer]) return;
  std::vector<MoeExpertView> table;
  table.reserve(w.experts.size());
  for (const auto& expert : w.experts) table.push_back(MoeExpertView::of(expert));
  DGPP_CUDA_OK(cudaMemcpyAsync(m.views + static_cast<size_t>(layer) * table.size(), table.data(),
                               table.size() * sizeof(MoeExpertView), cudaMemcpyHostToDevice,
                               stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  m.prepared[layer] = true;
}
void MinimaxMoeLayer::enqueue(int layer, const MinimaxMoeResident& w, const uint16_t* x, int tokens,
                              uint16_t* out, cudaStream_t stream, bool compact_segments,
                              bool narrow_tiles) {
  auto& m = *impl_;
  row_count(tokens, m.max_tokens);
  if (!m.prepared.at(static_cast<size_t>(layer)))
    throw std::logic_error("MiniMax expert table was not prepared");
  const int E = m.cfg.num_local_experts, K = m.cfg.num_experts_per_tok, H = m.cfg.hidden_size,
            I = m.inter;
  if (w.local_inter != I) throw std::invalid_argument("MiniMax expert geometry changed");
  const auto* views = m.views + static_cast<size_t>(layer) * E * 3;
  minimax_router(x, w.router, w.router_bias, tokens, H, E, K, m.cfg.routed_scaling_factor, m.scores,
                 m.ids, m.weights, stream);
  launch_moe_segment(m.ids, tokens, K, E, m.rows, m.slot_row, m.segs, stream);
  const int segments = compact_segments ? std::min(E, tokens * K) : E;
  const MoeSegment* segs = m.segs;
  if (segments < E) {
    // A short verification can touch only tokens * top_k experts. Avoid
    // scheduling empty tiles for the rest, retaining every dot's arithmetic.
    minimax_compact_segments(m.segs, m.compact_segs, E, segments, stream);
    segs = m.compact_segs;
  }
  // The smaller ring helps one to four rows. Wider verification batches
  // keep the measured 64-row tile, including its lower memory traffic.
  const int tile_rows = narrow_tiles && tokens <= 4 ? 16 : 64;
  if (m.native) {
    minimax_quantize_rows_nvfp4(x, H, tokens, H, m.input_codes, m.input_scales, m.input_globals,
                                stream);
    launch_moe_grouped_w4a4_f32(m.input_codes, m.input_scales, m.input_globals, m.rows, segs,
                                segments, tokens, views, 0, m.gate_f32, I, I, H, stream);
    launch_moe_grouped_w4a4_f32(m.input_codes, m.input_scales, m.input_globals, m.rows, segs,
                                segments, tokens, views, 1, m.up_f32, I, I, H, stream);
    minimax_swiglu_f32(m.gate_f32, m.up_f32, m.act, static_cast<int64_t>(tokens) * K * I,
                       m.cfg.swiglu_alpha, m.cfg.swiglu_limit, stream);
    minimax_quantize_rows_nvfp4(m.act, I, tokens * K, I, m.act_codes, m.act_scales, m.act_globals,
                                stream);
    launch_moe_grouped_w4a4_bf16(m.act_codes, m.act_scales, m.act_globals, nullptr, segs, segments,
                                 tokens, views, 2, m.down_bf16, H, H, I, stream);
    launch_moe_accum_ordered_bf16down(out, m.down_bf16, H, m.slot_row, m.ids, m.weights, tokens, K,
                                      H, stream);
  } else {
    launch_moe_grouped_mma_fp4_bf16(x, H, segs, segments, tokens, 0, views, 0, m.gate, I, I, H,
                                    stream, m.rows, kFp4Group, tile_rows);
    launch_moe_grouped_mma_fp4_bf16(x, H, segs, segments, tokens, 0, views, 1, m.up, I, I, H,
                                    stream, m.rows, kFp4Group, tile_rows);
    minimax_swiglu(m.gate, m.up, m.act, static_cast<int64_t>(tokens) * K * I, m.cfg.swiglu_alpha,
                   m.cfg.swiglu_limit, stream);
    launch_moe_grouped_mma_fp4_f32(m.act, I, segs, segments, tokens, 0, views, 2, m.down, H, H, I,
                                   stream, nullptr, kFp4Group, tile_rows);
    launch_moe_accum_ordered(out, m.down, H, m.slot_row, m.ids, m.weights, -1, tokens, K, H,
                             stream);
  }
  m.shared.enqueue(w.shared, x, tokens, m.shared_out, stream);
  glm_residual_add_bf16(out, m.shared_out, static_cast<int64_t>(tokens) * H, stream);
}
const int32_t* MinimaxMoeLayer::route_ids() const {
  return impl_->ids;
}
const float* MinimaxMoeLayer::route_weights() const {
  return impl_->weights;
}
}  // namespace dgpp
