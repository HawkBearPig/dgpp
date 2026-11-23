#include "models/inkling2/layers.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"
#include "kernels/inkling2.hpp"
#include "kernels/inkling2_attn.hpp"
#include "kernels/moe_w4a4.hpp"
#include "kernels/minimax.hpp"

namespace dgpp {
namespace {

template <class T>
T* dev_alloc(size_t n) {
  T* p = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  return p;
}

int max_extent_of(const Inkling2Config& cfg) { return std::max(cfg.rel_extent, cfg.sliding_window_size); }

}  // namespace

// ---- Inkling2AttentionLayer ------------------------------------------------------

int Inkling2AttentionLayer::decode_splits_of(bool local) const {
  if (!local) return n_split_;
  // A local row sees at most `window` tokens: ceil(window / tile) tiles.
  return std::max(1, std::min(n_split_, (cfg_.sliding_window_size + kInk2AttnTile - 1) / kInk2AttnTile));
}

Inkling2AttentionLayer::Inkling2AttentionLayer(const Inkling2Config& cfg, const Inkling2LocalGeometry& geo,
                                               const Inkling2GemmWorkspace& gemm, int max_tokens, int decode_rows,
                                               int n_split_decode)
    : cfg_(cfg), g_(gemm), hidden_(cfg.hidden_size), lh_(geo.local_heads), lkv_(geo.local_kv_heads),
      max_tokens_(max_tokens), decode_rows_(decode_rows), n_split_(n_split_decode) {
  if (!g_.gemm || !g_.ws) throw std::invalid_argument("Inkling2AttentionLayer: GEMM workspace required");
  if (max_tokens_ <= 0) throw std::invalid_argument("Inkling2AttentionLayer: max_tokens must be positive");
  if (decode_rows_ < 0) throw std::invalid_argument("Inkling2AttentionLayer: decode_rows");
  if (n_split_ <= 0) throw std::invalid_argument("Inkling2AttentionLayer: n_split_decode must be positive");
  if (lh_ <= 0 || lkv_ <= 0 || lh_ % lkv_ != 0)
    throw std::invalid_argument("Inkling2AttentionLayer: query heads must be a multiple of kv heads");
  if (cfg.head_dim != kInk2HeadDim) throw std::invalid_argument("Inkling2AttentionLayer: 128-wide heads");
  if (cfg.sconv_kernel_size != kInk2ConvTaps) throw std::invalid_argument("Inkling2AttentionLayer: four taps");
  scale_ = static_cast<float>(std::pow(static_cast<double>(kInk2HeadDim), -0.5));
  q_side_rows_ = static_cast<int64_t>(lh_) * (2 * kInk2HeadDim + cfg.d_rel + 1);
  kv_rows_ = static_cast<int64_t>(lkv_) * (kInk2HeadDim + kInk2ConvTaps);
  max_extent_ = max_extent_of(cfg);
  const size_t M = static_cast<size_t>(max_tokens_);
  const size_t Q = static_cast<size_t>(lh_) * kInk2HeadDim;
  q_side_ = dev_alloc<uint16_t>(M * static_cast<size_t>(q_side_rows_));
  k_ = dev_alloc<uint16_t>(M * static_cast<size_t>(kv_rows_));
  v_ = dev_alloc<uint16_t>(M * static_cast<size_t>(kv_rows_));
  q_ = dev_alloc<uint16_t>(M * Q);
  bias_ = dev_alloc<uint16_t>(M * static_cast<size_t>(lh_) * static_cast<size_t>(max_extent_));
  part_rows_ = std::max(M, static_cast<size_t>(decode_rows_) * static_cast<size_t>(n_split_));
  const size_t part = part_rows_ * static_cast<size_t>(lh_);
  m_ws_ = dev_alloc<float>(part);
  l_ws_ = dev_alloc<float>(part);
  c_ws_ = dev_alloc<float>(part * kInk2HeadDim);
  o_ = dev_alloc<uint16_t>(M * Q);
}

Inkling2AttentionLayer::~Inkling2AttentionLayer() {
  cudaFree(q_side_);
  cudaFree(k_);
  cudaFree(v_);
  cudaFree(q_);
  cudaFree(bias_);
  cudaFree(m_ws_);
  cudaFree(l_ws_);
  cudaFree(c_ws_);
  cudaFree(o_);
}

size_t Inkling2AttentionLayer::scratch_bytes(const Inkling2Config& cfg, const Inkling2LocalGeometry& geo,
                                             int max_tokens, int decode_rows, int n_split_decode) {
  const size_t M = static_cast<size_t>(std::max(max_tokens, 0));
  const size_t lh = static_cast<size_t>(geo.local_heads), lkv = static_cast<size_t>(geo.local_kv_heads);
  const size_t Q = lh * kInk2HeadDim;
  const size_t q_side = lh * static_cast<size_t>(2 * kInk2HeadDim + cfg.d_rel + 1);
  const size_t kv = lkv * (kInk2HeadDim + kInk2ConvTaps);
  const size_t part_rows =
      std::max(M, static_cast<size_t>(std::max(decode_rows, 0)) * static_cast<size_t>(n_split_decode));
  const size_t part = part_rows * lh;
  return M * q_side * 2 + 2 * M * kv * 2 + M * Q * 2 + M * lh * static_cast<size_t>(max_extent_of(cfg)) * 2 +
         part * 4 * 2 + part * kInk2HeadDim * 4 + M * Q * 2;
}

void Inkling2AttentionLayer::enqueue(const Inkling2AttnResident& w, const uint16_t* xq, const uint16_t* xk,
                                     const uint16_t* xv, int tokens, const Inkling2AttnRows& rows,
                                     const Inkling2KvCache& cache, const Inkling2Ring& ring, uint16_t* out,
                                     cudaStream_t stream) {
  if (tokens <= 0) return;
  if (tokens > max_tokens_) throw std::invalid_argument("Inkling2AttentionLayer: tokens exceed max_tokens");
  if (!xq || !xk || !xv || !out || !rows.req_ids || !rows.pos || !cache.k_cache || !cache.v_cache ||
      !cache.block_tables || !ring.base)
    throw std::invalid_argument("Inkling2AttentionLayer: null pointer");
  if (!w.q_side || !w.k || !w.v || !w.o_proj || !w.q_scale || !w.r_scale || !w.rel_proj || !w.k_taps || !w.v_taps)
    throw std::invalid_argument("Inkling2AttentionLayer: unbound weights");
  if (w.local_heads != lh_ || w.local_kv_heads != lkv_ || w.q_side_rows != q_side_rows_ || w.kv_rows > kv_rows_)
    throw std::invalid_argument("Inkling2AttentionLayer: the weights' head geometry differs from the layer's");
  if (cache.kv_heads != lkv_)
    throw std::invalid_argument("Inkling2AttentionLayer: the cache plane's kv heads differ from the weights'");
  if (cache.fp8()) throw std::invalid_argument("Inkling2AttentionLayer: the K/V cache must be bf16");
  if (w.extent <= 0 || w.extent > max_extent_) throw std::invalid_argument("Inkling2AttentionLayer: bias extent");
  if (rows.decode_prefix != 0)
    throw std::invalid_argument("Inkling2AttentionLayer: the walk hands decode and prefill rows over separately");
  const int H = hidden_;
  const int Q = lh_ * kInk2HeadDim;
  const int kv_cols = static_cast<int>(w.kv_rows);
  const auto project = [&](const uint16_t* x, const uint16_t* weight, uint16_t* dst, int n) {
    g_.gemm->matmul(x, weight, dst, tokens, n, H, DType::BF16, GemmOut::BF16, static_cast<size_t>(H), g_.ws,
                    g_.ws_bytes, stream);
  };
  project(xq, w.q_side, q_side_, static_cast<int>(q_side_rows_));
  project(xk, w.k, k_, kv_cols);
  project(xv, w.v, v_, kv_cols);

  Ink2QueryArgs qa;
  qa.q_side = q_side_;
  qa.q_side_stride = q_side_rows_;
  qa.q_scale = w.q_scale;
  qa.r_scale = w.r_scale;
  qa.rel_proj = w.rel_proj;
  qa.pos = rows.pos;
  qa.rows = tokens;
  qa.local_heads = lh_;
  qa.d_rel = cfg_.d_rel;
  qa.extent = w.extent;
  qa.qk_clip = cfg_.qk_l2_clip_threshold;
  qa.rel_clip = cfg_.rel_l2_clip_threshold;
  qa.log_scaling = !w.local && cfg_.log_scaling;
  qa.log_alpha = cfg_.log_scaling_alpha;
  qa.log_n_floor = static_cast<float>(cfg_.log_scaling_n_floor);
  qa.q_out = q_;
  qa.bias = bias_;
  ink2_query_finish(qa, stream);

  const int W = cfg_.sconv_kernel_size;
  Ink2KvArgs ka;
  ka.k_in = k_;
  ka.v_in = v_;
  ka.kv_stride = kv_cols;
  ka.k_taps = w.k_taps;
  ka.v_taps = w.v_taps;
  ka.dynamic = !w.local;
  ka.conv_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(w.local ? W + 1 : 2 * W + 1)));
  ka.ring = ring.base;
  ka.ring_request_stride = ring.request_stride;
  ka.ring_slot_stride = ring.slot_stride;
  ka.ring_offset_k = ring.k_offset();
  ka.ring_offset_v = ring.v_offset();
  ka.store_inline = rows.decode;
  ka.req_ids = rows.req_ids;
  ka.pos = rows.pos;
  ka.rows = tokens;
  ka.kv_heads = lkv_;
  ka.qk_clip = cfg_.qk_l2_clip_threshold;
  ka.eps = cfg_.rms_norm_eps;
  ka.block_tables = cache.block_tables;
  ka.blocks_per_request = cache.blocks_per_request;
  ka.block_tokens = cache.block_tokens;
  ka.k_cache = cache.k_cache;
  ka.v_cache = cache.v_cache;
  ink2_kv_finish(ka, stream);
  if (!rows.decode) {
    // A prefill run's inputs reach the ring after the launch that read it.
    const int width = lkv_ * kInk2HeadDim;
    ink2_ring_store(k_, kv_cols, 0, ring.base, ring.request_stride, ring.slot_stride, ring.k_offset(), rows.req_ids,
                    rows.pos, tokens, width, stream);
    ink2_ring_store(v_, kv_cols, 0, ring.base, ring.request_stride, ring.slot_stride, ring.v_offset(), rows.req_ids,
                    rows.pos, tokens, width, stream);
  }
  if (rows.cache_only) return;

  const uint16_t* gate = q_side_ + static_cast<int64_t>(lh_) * (kInk2HeadDim + cfg_.d_rel);
  const uint16_t* sink = q_side_ + static_cast<int64_t>(lh_) * (2 * kInk2HeadDim + cfg_.d_rel);
  if (!rows.decode) {
    // Reader rows: the query-tiled tensor-core form, the gate in its epilogue.
    Ink2AttnPrefillArgs pa;
    pa.q = q_;
    pa.q_stride = Q;
    pa.bias = bias_;
    pa.extent = w.extent;
    pa.sink = sink;
    pa.sink_stride = q_side_rows_;
    pa.gate = gate;
    pa.gate_stride = q_side_rows_;
    pa.k_cache = cache.k_cache;
    pa.v_cache = cache.v_cache;
    pa.req_ids = rows.req_ids;
    pa.pos = rows.pos;
    pa.rows = tokens;
    pa.local_heads = lh_;
    pa.kv_heads = lkv_;
    pa.block_tokens = cache.block_tokens;
    pa.block_tables = cache.block_tables;
    pa.blocks_per_request = cache.blocks_per_request;
    pa.window = w.local ? cfg_.sliding_window_size : 0;
    pa.scale = scale_;
    pa.out = o_;
    ink2_attn_prefill(pa, stream);
    g_.gemm->matmul(o_, w.o_proj, out, tokens, H, Q, DType::BF16, GemmOut::BF16, static_cast<size_t>(Q), g_.ws,
                    g_.ws_bytes, stream);
    return;
  }
  const int n_split = decode_splits_of(w.local);
  if (static_cast<size_t>(tokens) * static_cast<size_t>(n_split) > part_rows_)
    throw std::invalid_argument("Inkling2AttentionLayer: decode rows exceed the layer's decode_rows");
  Ink2AttnArgs aa;
  aa.q = q_;
  aa.q_stride = Q;
  aa.bias = bias_;
  aa.extent = w.extent;
  aa.sink = sink;
  aa.sink_stride = q_side_rows_;
  aa.k_cache = cache.k_cache;
  aa.v_cache = cache.v_cache;
  aa.req_ids = rows.req_ids;
  aa.pos = rows.pos;
  aa.rows = tokens;
  aa.n_split = n_split;
  aa.local_heads = lh_;
  aa.kv_heads = lkv_;
  aa.block_tokens = cache.block_tokens;
  aa.block_tables = cache.block_tables;
  aa.blocks_per_request = cache.blocks_per_request;
  aa.window = w.local ? cfg_.sliding_window_size : 0;
  aa.scale = scale_;
  aa.m_ws = m_ws_;
  aa.l_ws = l_ws_;
  aa.c_ws = c_ws_;
  ink2_attn_partial(aa, stream);
  ink2_attn_combine(m_ws_, l_ws_, c_ws_, gate, q_side_rows_, tokens, n_split, lh_, o_, stream);
  g_.gemm->matmul(o_, w.o_proj, out, tokens, H, Q, DType::BF16, GemmOut::BF16, static_cast<size_t>(Q), g_.ws,
                  g_.ws_bytes, stream);
}

// ---- Inkling2MoeLayer ------------------------------------------------------------------

GlmMoeConfig Inkling2MoeLayer::router_config(const Inkling2Config& cfg, int local_inter) {
  GlmMoeConfig m;
  m.hidden = cfg.hidden_size;
  m.inter = local_inter;
  m.n_experts = cfg.n_routed_experts;
  m.top_k = cfg.num_experts_per_tok;
  m.n_shared_experts = 0;
  m.routed_scaling_factor = cfg.route_scale;
  m.norm_topk_prob = true;
  m.router_mode = MoeRouterMode::SigmoidBias;
  GlmMoeConfig::validate_config(m);
  return m;
}

Inkling2MoeLayer::Inkling2MoeLayer(const Inkling2Config& cfg, const Inkling2LocalGeometry& geo, int max_tokens,
                                   bool native, bool resident)
    : cfg_(cfg), router_cfg_(router_config(cfg, static_cast<int>(geo.local_inter))), hidden_(cfg.hidden_size),
      inter_(static_cast<int>(geo.local_inter)), experts_(cfg.n_routed_experts), top_k_(cfg.num_experts_per_tok),
      max_tokens_(max_tokens), native_(native), resident_(resident),
      bound_(static_cast<size_t>(cfg.num_hidden_layers), 0) {
  if (max_tokens_ <= 0) throw std::invalid_argument("Inkling2MoeLayer: max_tokens must be positive");
  if (inter_ <= 0 || inter_ % 32 != 0)
    throw std::invalid_argument("Inkling2MoeLayer: the expert slice must be a multiple of 32");
  if (experts_ > 1024) throw std::invalid_argument("Inkling2MoeLayer: at most 1024 experts (the segmentation kernel)");
  const size_t M = static_cast<size_t>(max_tokens_), K = static_cast<size_t>(top_k_), E = static_cast<size_t>(experts_);
  const size_t H = static_cast<size_t>(hidden_), I = static_cast<size_t>(inter_);
  scores_ = dev_alloc<float>(M * E);
  biased_ = dev_alloc<float>(M * E);
  weights_ = dev_alloc<float>(M * K);
  ids_ = dev_alloc<int32_t>(M * K);
  rows_ = dev_alloc<int32_t>(M * (K + 1));
  slot_row_ = dev_alloc<int32_t>(M * K);
  row_slot_ = dev_alloc<int32_t>(M * K);
  counters_ = dev_alloc<int>(M);
  DGPP_CUDA_OK(cudaMemset(counters_, 0, M * sizeof(int)));
  segs_ = dev_alloc<MoeSegment>(E + 1);
  compact_ = dev_alloc<MoeSegment>(E);
  views_ = dev_alloc<MoeExpertView>(static_cast<size_t>(cfg.num_hidden_layers) * E * 3);
  gate_up_ = dev_alloc<float>(M * K * 2 * I);
  act_ = dev_alloc<uint16_t>(M * K * I);
  down_ = dev_alloc<float>(M * K * H);
  ones_ = dev_alloc<float>(M * K);
  {
    std::vector<float> one(M * K, 1.0f);
    DGPP_CUDA_OK(cudaMemcpy(ones_, one.data(), one.size() * 4, cudaMemcpyHostToDevice));
  }
  if (native_) {
    x_inner_ = dev_alloc<uint16_t>(M * H);
    x_outer_ = dev_alloc<float>(M);
    act_inner_ = dev_alloc<uint16_t>(M * K * I);
    act_outer_ = dev_alloc<float>(M * K);
    x_scale_stride_ = static_cast<int64_t>(nvfp4_act_scale_stride(hidden_));
    act_scale_stride_ = static_cast<int64_t>(nvfp4_act_scale_stride(inter_));
    x_codes_ = dev_alloc<uint8_t>(M * H / 2);
    x_scales_ = dev_alloc<uint8_t>(M * static_cast<size_t>(x_scale_stride_));
    act_codes_ = dev_alloc<uint8_t>(M * K * I / 2);
    act_scales_ = dev_alloc<uint8_t>(M * K * static_cast<size_t>(act_scale_stride_));
    act_gs_ = dev_alloc<float>(M * K);
  }
}

Inkling2MoeLayer::~Inkling2MoeLayer() {
  cudaFree(scores_);
  cudaFree(biased_);
  cudaFree(weights_);
  cudaFree(ids_);
  cudaFree(rows_);
  cudaFree(slot_row_);
  cudaFree(row_slot_);
  cudaFree(counters_);
  cudaFree(segs_);
  cudaFree(compact_);
  cudaFree(views_);
  cudaFree(gate_up_);
  cudaFree(act_);
  cudaFree(down_);
  cudaFree(ones_);
  cudaFree(x_inner_);
  cudaFree(x_outer_);
  cudaFree(act_inner_);
  cudaFree(act_outer_);
  cudaFree(x_codes_);
  cudaFree(x_scales_);
  cudaFree(act_codes_);
  cudaFree(act_scales_);
  cudaFree(act_gs_);
}

size_t Inkling2MoeLayer::scratch_bytes(const Inkling2Config& cfg, const Inkling2LocalGeometry& geo, int max_tokens,
                                       bool native) {
  const size_t M = static_cast<size_t>(std::max(max_tokens, 0)), K = static_cast<size_t>(cfg.num_experts_per_tok);
  const size_t E = static_cast<size_t>(cfg.n_routed_experts), H = static_cast<size_t>(cfg.hidden_size);
  const size_t I = static_cast<size_t>(geo.local_inter);
  size_t b = 2 * M * E * 4 + M * K * 4 * 2 + M * (K + 1) * 4 + 2 * M * K * 4 + M * 4 +
             (2 * E + 1) * sizeof(MoeSegment) +
             static_cast<size_t>(cfg.num_hidden_layers) * E * 3 * sizeof(MoeExpertView) + M * K * 2 * I * 4 +
             M * K * I * 2 + M * K * H * 4 + M * K * 4;
  if (native)
    b += M * H * 2 + M * 4 + M * K * I * 2 + M * K * 4 + M * H / 2 + M * nvfp4_act_scale_stride(cfg.hidden_size) +
         M * K * I / 2 + M * K * nvfp4_act_scale_stride(static_cast<int>(geo.local_inter)) + M * K * 4;
  return b;
}

void Inkling2MoeLayer::bind(int layer, const Inkling2MoeResident& w, cudaStream_t stream) {
  if (layer < 0 || layer >= cfg_.num_hidden_layers) throw std::invalid_argument("Inkling2MoeLayer: layer");
  if (resident_ && bound_[static_cast<size_t>(layer)]) return;
  const size_t entries = static_cast<size_t>(experts_) * 3;
  if ((w.mxfp8 ? w.fp8.size() : w.fp4.size()) != entries)
    throw std::invalid_argument("Inkling2MoeLayer: expert table geometry");
  if (w.local_inter != inter_) throw std::invalid_argument("Inkling2MoeLayer: the expert slice changed");
  std::vector<MoeExpertView> table(entries);
  for (size_t i = 0; i < entries; ++i) {
    if (i % 3 == 1) continue;  // gate and up are one interleaved matrix at entry 0
    table[i] = w.mxfp8 ? MoeExpertView::of(w.fp8[i]) : MoeExpertView::of(w.fp4[i]);
  }
  DGPP_CUDA_OK(cudaMemcpyAsync(views_ + static_cast<size_t>(layer) * entries, table.data(),
                               entries * sizeof(MoeExpertView), cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  bound_[static_cast<size_t>(layer)] = 1;
}

void Inkling2MoeLayer::enqueue(int layer, const Inkling2MoeResident& w, const uint16_t* x, int tokens, bool decode,
                               uint16_t* out, cudaStream_t stream, const int64_t* positions) {
  if (tokens <= 0) return;
  if (tokens > max_tokens_) throw std::invalid_argument("Inkling2MoeLayer: tokens exceed max_tokens");
  if (!x || !out || !w.router || !w.router_bias) throw std::invalid_argument("Inkling2MoeLayer: null pointer");
  if (layer < 0 || layer >= cfg_.num_hidden_layers || !bound_[static_cast<size_t>(layer)])
    throw std::logic_error("Inkling2MoeLayer: the layer's expert table is not bound");
  const int E = experts_, K = top_k_, H = hidden_, I = inter_;
  const int routed = tokens * K;
  const MoeExpertView* views = views_ + static_cast<size_t>(layer) * E * 3;
  launch_moe_router(x, w.router, w.router_bias, ids_, weights_, scores_, biased_, router_cfg_, tokens, stream,
                    counters_);
  launch_moe_segment(ids_, tokens, K, E, rows_, slot_row_, segs_, stream, positions);
  ink2_slot_rows(slot_row_, routed, row_slot_, stream);
  const int segments = std::min(E, routed);
  const MoeSegment* segs = segs_;
  if (segments < E) {
    // A short batch touches at most tokens x top_k experts: schedule those.
    minimax_compact_segments(segs_, compact_, E, segments, stream);
    segs = compact_;
  }
  const int tile_rows = tokens <= 4 ? 16 : 64;
  // Every form below is one recipe. An expert's gate / up and down outputs
  // are rounded to BF16 at their row scale, the routing weight rides the
  // activation, and a token's rounded expert outputs are summed in
  // ascending expert id and rounded once.
  if (decode) {
    // Decode rows: a few rows against whole expert matrices, read once each.
    // The FP32 forms carry the dot unrounded and the row scale meets it
    // where the value is rounded.
    const uint16_t* x_in = x;
    const float* x_scale = nullptr;
    if (native_) {
      if (w.mxfp8) {
        ink2_requantize_mxfp8(x, H, tokens, H, x_inner_, H, stream);
      } else {
        ink2_quantize_nvfp4(x, H, tokens, H, x_inner_, H, x_outer_, nullptr, stream);
        x_scale = x_outer_;
      }
      x_in = x_inner_;
    }
    if (w.mxfp8)
      launch_moe_grouped_mma_f32(x_in, H, segs, segments, tokens, 0, views, 0, gate_up_, 2 * I, 2 * I, H, stream,
                                 rows_);
    else
      launch_moe_grouped_mma_fp4_f32(x_in, H, segs, segments, tokens, 0, views, 0, gate_up_, 2 * I, 2 * I, H, stream,
                                     rows_, kFp4Group, tile_rows);
    ink2_swiglu(gate_up_, 2 * I, routed, I, x_scale, rows_, weights_, row_slot_, cfg_.mlp_gate_softcap,
                cfg_.mlp_up_softcap, act_, I, stream);
    const uint16_t* act_in = act_;
    const float* slot_scale = ones_;
    if (native_) {
      if (w.mxfp8) {
        ink2_requantize_mxfp8(act_, I, routed, I, act_inner_, I, stream);
      } else {
        ink2_quantize_nvfp4(act_, I, routed, I, act_inner_, I, act_outer_, row_slot_, stream);
        slot_scale = act_outer_;
      }
      act_in = act_inner_;
    }
    if (w.mxfp8)
      launch_moe_grouped_mma_f32(act_in, I, segs, segments, tokens, 0, views, 2, down_, H, H, I, stream, nullptr);
    else
      launch_moe_grouped_mma_fp4_f32(act_in, I, segs, segments, tokens, 0, views, 2, down_, H, H, I, stream, nullptr,
                                     kFp4Group, tile_rows);
    ink2_moe_accum_rounded(out, down_, H, slot_row_, ids_, slot_scale, tokens, K, H, stream, positions);
    return;
  }
  // Reader rows: BF16 expert outputs, so the intermediates cross memory at
  // half the width, and under the native recipe the NVFP4 experts run on
  // the tensor cores' block-scaled form with both operands in four bits.
  uint16_t* gate_up16 = reinterpret_cast<uint16_t*>(gate_up_);
  uint16_t* down16 = reinterpret_cast<uint16_t*>(down_);
  if (w.mxfp8) {
    const uint16_t* x_in = x;
    if (native_) {
      ink2_requantize_mxfp8(x, H, tokens, H, x_inner_, H, stream);
      x_in = x_inner_;
    }
    launch_moe_grouped_mma_bf16(x_in, H, segs, segments, tokens, 0, views, 0, gate_up16, 2 * I, 2 * I, H, stream, rows_);
    ink2_swiglu_bf16(gate_up16, 2 * I, routed, I, weights_, row_slot_, cfg_.mlp_gate_softcap, cfg_.mlp_up_softcap, act_,
                     I, stream);
    const uint16_t* act_in = act_;
    if (native_) {
      ink2_requantize_mxfp8(act_, I, routed, I, act_inner_, I, stream);
      act_in = act_inner_;
    }
    launch_moe_grouped_mma_bf16(act_in, I, segs, segments, tokens, 0, views, 2, down16, H, H, I, stream, nullptr);
    launch_moe_accum_ordered_bf16down(out, down16, H, slot_row_, ids_, ones_, tokens, K, H, stream);
    return;
  }
  if (!native_) {
    launch_moe_grouped_mma_fp4_bf16(x, H, segs, segments, tokens, 0, views, 0, gate_up16, 2 * I, 2 * I, H, stream,
                                    rows_, kFp4Group, tile_rows);
    ink2_swiglu_bf16(gate_up16, 2 * I, routed, I, weights_, row_slot_, cfg_.mlp_gate_softcap, cfg_.mlp_up_softcap, act_,
                     I, stream);
    launch_moe_grouped_mma_fp4_bf16(act_, I, segs, segments, tokens, 0, views, 2, down16, H, H, I, stream, nullptr,
                                    kFp4Group, tile_rows);
    launch_moe_accum_ordered_bf16down(out, down16, H, slot_row_, ids_, ones_, tokens, K, H, stream);
    return;
  }
  // Native NVFP4. The block-scaled instruction steps 64 values of k at a
  // time: the gate / up projection reads the hidden width, a multiple of
  // 64 at every world, and the down projection reads the rank's slice of
  // the expert width, which is not one at world 4. That slice takes the
  // BF16 tensor-core form on the exact dequantized rows.
  if (H % 64 == 0) {
    ink2_quantize_nvfp4_codes(x, H, tokens, H, x_codes_, x_scales_, x_scale_stride_, x_outer_, stream);
    launch_moe_grouped_w4a4_bf16(x_codes_, x_scales_, x_outer_, rows_, segs, segments, tokens, views, 0, gate_up16,
                                 2 * I, 2 * I, H, stream);
  } else {
    ink2_quantize_nvfp4(x, H, tokens, H, x_inner_, H, x_outer_, nullptr, stream);
    launch_moe_grouped_mma_fp4_f32(x_inner_, H, segs, segments, tokens, 0, views, 0, gate_up_, 2 * I, 2 * I, H, stream,
                                   rows_, kFp4Group, tile_rows);
    ink2_swiglu(gate_up_, 2 * I, routed, I, x_outer_, rows_, weights_, row_slot_, cfg_.mlp_gate_softcap,
                cfg_.mlp_up_softcap, act_, I, stream);
  }
  if (H % 64 == 0 && I % 64 == 0) {
    ink2_swiglu_quantize_nvfp4(gate_up16, 2 * I, routed, I, weights_, row_slot_, cfg_.mlp_gate_softcap,
                               cfg_.mlp_up_softcap, act_codes_, act_scales_, act_scale_stride_, act_gs_, stream);
    launch_moe_grouped_w4a4_bf16(act_codes_, act_scales_, act_gs_, nullptr, segs, segments, tokens, views, 2, down16, H,
                                 H, I, stream);
    launch_moe_accum_ordered_bf16down(out, down16, H, slot_row_, ids_, ones_, tokens, K, H, stream);
    return;
  }
  if (H % 64 == 0)
    ink2_swiglu_bf16(gate_up16, 2 * I, routed, I, weights_, row_slot_, cfg_.mlp_gate_softcap, cfg_.mlp_up_softcap, act_,
                     I, stream);
  ink2_quantize_nvfp4(act_, I, routed, I, act_inner_, I, act_outer_, row_slot_, stream);
  launch_moe_grouped_mma_fp4_f32(act_inner_, I, segs, segments, tokens, 0, views, 2, down_, H, H, I, stream, nullptr,
                                 kFp4Group, tile_rows);
  ink2_moe_accum_rounded(out, down_, H, slot_row_, ids_, act_outer_, tokens, K, H, stream);
}

}  // namespace dgpp
