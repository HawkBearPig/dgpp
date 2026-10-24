#include "models/mimo/layers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/mimo_attn.hpp"
#include "kernels/qsa.hpp"
#include "kernels/scale_gemm.hpp"

namespace dgpp {
namespace {

template <class T>
T* dev_alloc(size_t n) {
  T* p = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  return p;
}

// The fp32 projection buffer's width for a layer kind: the rank's chunks
// at the padded stride (the last chunk's padding never written).
int64_t qkv_cols_of(const MimoTextConfig& cfg, int chunks, bool swa) {
  const int64_t chunk_rows = cfg.qkv_chunk_rows_of_kind(swa);
  const int64_t stride = ((chunk_rows + 127) / 128) * 128;
  return static_cast<int64_t>(chunks) * stride;
}

}  // namespace

// ---- MimoAttentionLayer ----------------------------------------------------------

int MimoAttentionLayer::default_decode_splits() {
  // 32 splits for the global layers (the GLM-4.7 default: at TP=4, 16
  // query heads over one kv head per rank, a T=1 row is 32 blocks of 16
  // warps; a short context leaves splits empty, a 32K context gives each
  // split 32 tiles). A sliding-window layer splits per tile of its window.
  if (const char* v = std::getenv("DGPP_MIMO_ATTN_SPLITS"); v && *v) {
    const int n = std::atoi(v);
    if (n > 0) return n;
  }
  return 32;
}

int MimoAttentionLayer::decode_splits_of(const MimoAttnResident& w) const {
  if (!w.swa) return n_split_;
  return std::max(1, std::min(n_split_, window_ / kMimoAttnTile));
}

MimoAttentionLayer::MimoAttentionLayer(const MimoAttnResident& w, const MimoGemmWorkspace& gemm,
                                       const MimoTextConfig& cfg, int max_tokens, int decode_rows,
                                       int n_split_decode, bool decode_mma)
    : w_(w), g_(gemm), hidden_(cfg.hidden_size), lh_(w.local_heads), max_tokens_(max_tokens), decode_rows_(decode_rows), n_split_(n_split_decode), window_(cfg.sliding_window),
      decode_mma_(decode_mma), value_scale_(cfg.attention_value_scale) {
  if (!g_.gemm || !g_.ws) throw std::invalid_argument("MimoAttentionLayer: GEMM workspace required");
  if (max_tokens_ <= 0) throw std::invalid_argument("MimoAttentionLayer: max_tokens must be positive");
  if (decode_rows_ < 0) throw std::invalid_argument("MimoAttentionLayer: decode_rows");
  if (n_split_ <= 0) throw std::invalid_argument("MimoAttentionLayer: n_split_decode must be positive");
  if (lh_ <= 0 || w_.local_kv_heads <= 0 || lh_ % w_.local_kv_heads != 0)
    throw std::invalid_argument("MimoAttentionLayer: query heads must be a multiple of kv heads");
  if (cfg.head_dim != kMimoQkDim || cfg.v_head_dim != kMimoVDim || cfg.rotary_dim != kMimoRotaryDim)
    throw std::invalid_argument("MimoAttentionLayer: qk 192 / v 128 / rotary 64 (the kernels' shape)");
  if (window_ <= 0 || window_ % kMimoAttnTile != 0)
    throw std::invalid_argument("MimoAttentionLayer: the window must be a positive multiple of the 32-token tile");
  // The reference's scale: head_dim ** -0.5 in double, narrowed to fp32.
  scale_ = static_cast<float>(std::pow(static_cast<double>(cfg.head_dim), -0.5));
  const size_t M = static_cast<size_t>(max_tokens_);
  const size_t Q = static_cast<size_t>(lh_) * kMimoQkDim, O = static_cast<size_t>(lh_) * kMimoVDim;
  {
    std::vector<float> inv(static_cast<size_t>(kMimoRotaryDim / 2));
    qsa_rope_inv_freq(cfg.rope_theta, kMimoRotaryDim, inv.data());
    d_inv_freq_ga_ = dev_alloc<float>(inv.size());
    DGPP_CUDA_OK(cudaMemcpy(d_inv_freq_ga_, inv.data(), inv.size() * 4, cudaMemcpyHostToDevice));
    qsa_rope_inv_freq(cfg.swa_rope_theta, kMimoRotaryDim, inv.data());
    d_inv_freq_swa_ = dev_alloc<float>(inv.size());
    DGPP_CUDA_OK(cudaMemcpy(d_inv_freq_swa_, inv.data(), inv.size() * 4, cudaMemcpyHostToDevice));
  }
  // The projection buffer covers the wider kind (the sliding-window
  // layers' chunks carry more kv heads).
  const int chunks = w_.chunks;
  qkv_cols_ = std::max(qkv_cols_of(cfg, chunks, false), qkv_cols_of(cfg, chunks, true));
  qkv_ = dev_alloc<float>(M * static_cast<size_t>(qkv_cols_));
  q_ = dev_alloc<uint16_t>(M * Q);
  part_rows_ = std::max(M, static_cast<size_t>(decode_rows_) * static_cast<size_t>(n_split_));
  const size_t part = part_rows_ * lh_;
  m_ws_ = dev_alloc<float>(part);
  l_ws_ = dev_alloc<float>(part);
  c_ws_ = dev_alloc<float>(part * kMimoVDim);
  o_ = dev_alloc<uint16_t>(M * O);
  // The fused decode kernel's arrival counters: one per (decode row, kv
  // head), zero between launches (the kernel resets its own).
  const size_t counters = static_cast<size_t>(std::max(decode_rows_, 1)) * static_cast<size_t>(w_.local_kv_heads);
  counters_ = dev_alloc<int>(counters);
  DGPP_CUDA_OK(cudaMemset(counters_, 0, counters * sizeof(int)));
  if (const char* v = std::getenv("DGPP_MIMO_ATTN_FUSED"); v && *v) fused_decode_ = std::atoi(v) != 0;
  if (const char* v = std::getenv("DGPP_MIMO_PREFILL_ATTN"); v && *v) tiled_prefill_ = std::string(v) != "chain";
  if (kSinkAttnPrefillQueryVecs % (lh_ / w_.local_kv_heads) != 0) tiled_prefill_ = false;  // the tiled kernel's query vectors
}

MimoAttentionLayer::~MimoAttentionLayer() {
  cudaFree(d_inv_freq_ga_);
  cudaFree(d_inv_freq_swa_);
  cudaFree(qkv_);
  cudaFree(q_);
  cudaFree(m_ws_);
  cudaFree(l_ws_);
  cudaFree(c_ws_);
  cudaFree(counters_);
  cudaFree(o_);
}

void MimoAttentionLayer::rebind(const MimoAttnResident& w) {
  if (w.local_heads != lh_ || w.chunks != w_.chunks)
    throw std::invalid_argument("MimoAttentionLayer: rebind changes the head geometry");
  if (w.local_kv_heads <= 0 || lh_ % w.local_kv_heads != 0)
    throw std::invalid_argument("MimoAttentionLayer: rebind's kv heads do not divide the query heads");
  if (static_cast<int64_t>(w.chunks - 1) * w.chunk_stride + w.chunk_rows > qkv_cols_)
    throw std::invalid_argument("MimoAttentionLayer: rebind's projection exceeds the buffer");
  w_ = w;
}

size_t MimoAttentionLayer::scratch_bytes(const MimoTextConfig& cfg, int local_heads, int max_tokens, int decode_rows,
                                         int n_split_decode) {
  const size_t M = static_cast<size_t>(std::max(max_tokens, 0));
  const size_t Q = static_cast<size_t>(local_heads) * kMimoQkDim, O = static_cast<size_t>(local_heads) * kMimoVDim;
  const int chunks = cfg.qkv_chunks() * local_heads / std::max(1, cfg.num_attention_heads);
  const size_t cols = static_cast<size_t>(std::max(qkv_cols_of(cfg, chunks, false), qkv_cols_of(cfg, chunks, true)));
  const size_t part_rows = std::max(M, static_cast<size_t>(std::max(decode_rows, 0)) * static_cast<size_t>(n_split_decode));
  const size_t part = part_rows * local_heads;
  return 2 * static_cast<size_t>(kMimoRotaryDim / 2) * 4 + M * cols * 4 + M * Q * 2 + part * 4 * 2 +
         part * kMimoVDim * 4 + M * O * 2;
}

void MimoAttentionLayer::enqueue(const uint16_t* x, int tokens, const MimoAttnRows& rows, const MimoKvCache& cache,
                                 uint16_t* out, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (tokens > max_tokens_) throw std::invalid_argument("MimoAttentionLayer: tokens exceed max_tokens");
  if (!x || !out || !rows.req_ids || !rows.pos || !cache.k_cache || !cache.v_cache || !cache.block_tables)
    throw std::invalid_argument("MimoAttentionLayer: null pointer");
  if (cache.kv_heads != w_.local_kv_heads)
    throw std::invalid_argument("MimoAttentionLayer: the cache layer's kv heads differ from the weights'");
  if (cache.fp8() != (cache.k_scale != nullptr && cache.v_scale != nullptr))
    throw std::invalid_argument("MimoAttentionLayer: an fp8 cache carries its scale planes");
  if (rows.split(x, tokens, hidden_, out, [&](const uint16_t* input, int n,
      const auto& map, uint16_t* output) { enqueue(input, n, map, cache, output, stream); })) return;
  const int H = hidden_;
  const int lkv = w_.local_kv_heads;
  const int Q = lh_ * kMimoQkDim, O = lh_ * kMimoVDim;
  // The fused projection, fp32 out (the finish rounds once): decode rows
  // on the streaming tensor-core GEMM (a row's chain the same whatever
  // rows share the launch), large prefill chunks on the exact pipelined
  // fp8-weight GEMM; small tails retain the GEMV chunks.
  const int64_t n = static_cast<int64_t>(w_.chunks - 1) * w_.chunk_stride + w_.chunk_rows;
  launch_scale_gemm_grid_f32(x, static_cast<size_t>(H), w_.qkv.payload, w_.qkv.scales, qkv_, tokens,
                             static_cast<int>(n), H, stream, static_cast<size_t>(qkv_cols_), 7, 7,
                             decode_mma_ && rows.decode);
  MimoQkvLayout layout;
  layout.chunks = w_.chunks;
  layout.chunk_stride = w_.chunk_stride;
  layout.q_per_chunk = w_.q_per_chunk;
  layout.kv_per_chunk = w_.kv_per_chunk;
  // The attention: decode rows split the visible range (the capture's
  // geometry), prefill rows run one split each (T x kv-head blocks).
  const int n_split = rows.decode ? decode_splits_of(w_) : 1;
  if (static_cast<size_t>(tokens) * static_cast<size_t>(n_split) > part_rows_)
    throw std::invalid_argument("MimoAttentionLayer: decode rows exceed the layer's decode_rows");
  const float* inv_freq = w_.swa ? d_inv_freq_swa_ : d_inv_freq_ga_;
  if (rows.cache_only) {
    mimo_qkv_finish(qkv_, qkv_cols_, layout, inv_freq, value_scale_, rows.req_ids, rows.pos, tokens,
                    lh_, lkv, cache.block_tables, cache.blocks_per_request, cache.block_tokens, q_,
                    Q, cache.k_cache, cache.v_cache, stream, cache.k_scale, cache.v_scale);
    return;
  }
  if (rows.decode && fused_decode_ && tokens <= kMimoAttnFusedMaxRows && tokens <= decode_rows_ &&
      lh_ / lkv >= 4 && lh_ / lkv <= 16) {
    // A decode batch: finish + partials + combine in one launch (plan §7.1).
    MimoAttnFusedArgs a;
    a.qkv = qkv_;
    a.qkv_stride = qkv_cols_;
    a.layout = layout;
    a.inv_freq = inv_freq;
    a.value_scale = value_scale_;
    a.req_ids = rows.req_ids;
    a.pos = rows.pos;
    a.rows = tokens;
    a.n_split = n_split;
    a.local_heads = lh_;
    a.kv_heads = lkv;
    a.block_tables = cache.block_tables;
    a.blocks_per_request = cache.blocks_per_request;
    a.block_tokens = cache.block_tokens;
    a.window = w_.swa ? window_ : 0;
    a.scale = scale_;
    a.sink = w_.sink;
    a.k_cache = cache.k_cache;
    a.v_cache = cache.v_cache;
    a.k_scale = cache.k_scale;
    a.v_scale = cache.v_scale;
    a.m_ws = m_ws_;
    a.l_ws = l_ws_;
    a.c_ws = c_ws_;
    a.counters = counters_;
    a.out = o_;
    mimo_attn_fused(a, stream);
  } else if (!rows.decode && tiled_prefill_) {
    // Prefill rows: every row's K/V appended first, then the query-tiled
    // tensor-core attention (plan §7.2).
    mimo_qkv_finish(qkv_, qkv_cols_, layout, inv_freq, value_scale_, rows.req_ids, rows.pos, tokens, lh_, lkv,
                    cache.block_tables, cache.blocks_per_request, cache.block_tokens, q_, Q, cache.k_cache,
                    cache.v_cache, stream, cache.k_scale, cache.v_scale);
    MimoAttnPrefillArgs a;
    a.q = q_;
    a.q_stride = Q;
    a.k_cache = cache.k_cache;
    a.v_cache = cache.v_cache;
    a.k_scale = cache.k_scale;
    a.v_scale = cache.v_scale;
    a.req_ids = rows.req_ids;
    a.pos = rows.pos;
    a.rows = tokens;
    a.local_heads = lh_;
    a.kv_heads = lkv;
    a.block_tokens = cache.block_tokens;
    a.block_tables = cache.block_tables;
    a.blocks_per_request = cache.blocks_per_request;
    a.window = w_.swa ? window_ : 0;
    a.scale = scale_;
    a.sink = w_.sink;
    a.out = o_;
    mimo_attn_prefill(a, stream);
  } else {
    mimo_qkv_finish(qkv_, qkv_cols_, layout, inv_freq, value_scale_, rows.req_ids, rows.pos, tokens, lh_, lkv,
                    cache.block_tables, cache.blocks_per_request, cache.block_tokens, q_, Q, cache.k_cache,
                    cache.v_cache, stream, cache.k_scale, cache.v_scale);
    mimo_attn_partial(q_, Q, cache.k_cache, cache.v_cache, rows.req_ids, rows.pos, tokens, n_split, lh_, lkv,
                      cache.block_tokens, cache.block_tables, cache.blocks_per_request, w_.swa ? window_ : 0, scale_,
                      w_.sink, m_ws_, l_ws_, c_ws_, stream, cache.k_scale, cache.v_scale);
    mimo_attn_combine(m_ws_, l_ws_, c_ws_, tokens, n_split, lh_, o_, stream);
  }
  g_.gemm->matmul(o_, w_.o_proj, out, tokens, H, O, DType::BF16, GemmOut::BF16, static_cast<size_t>(O), g_.ws,
                  g_.ws_bytes, stream);
}

// ---- MimoDenseMlp ------------------------------------------------------------------

MimoDenseMlp::MimoDenseMlp(const MimoDenseMlpResident& w, const MimoTextConfig& cfg, int max_tokens, int decode_rows,
                           bool decode_mma)
    : w_(w), hidden_(cfg.hidden_size), max_tokens_(max_tokens), decode_rows_(decode_rows), inter_(w.local_inter),
      decode_mma_(decode_mma) {
  if (max_tokens_ <= 0) throw std::invalid_argument("MimoDenseMlp: max_tokens must be positive");
  if (inter_ <= 0 || inter_ % 128 != 0) throw std::invalid_argument("MimoDenseMlp: the inter slice must be a multiple of 128");
  if (!w_.gate.payload || !w_.up.payload || !w_.down.payload) throw std::invalid_argument("MimoDenseMlp: unbound weights");
  if (w_.gate.rows != inter_ || w_.up.rows != inter_ || w_.gate.cols != hidden_ || w_.up.cols != hidden_ ||
      w_.down.rows != hidden_ || w_.down.cols != inter_)
    throw std::invalid_argument("MimoDenseMlp: inconsistent matrices");
  if (w_.gate.scale_block_rows != 128 || w_.gate.scale_block_cols != 128 || w_.up.scale_block_rows != 128 ||
      w_.up.scale_block_cols != 128 || w_.down.scale_block_rows != 128 || w_.down.scale_block_cols != 128)
    throw std::invalid_argument("MimoDenseMlp: the scale grids must be 128 x 128 (the slices are 128-aligned)");
  const size_t M = static_cast<size_t>(max_tokens_), I = static_cast<size_t>(inter_);
  gate_ = dev_alloc<uint16_t>(M * I);
  up_ = dev_alloc<uint16_t>(M * I);
  act_ = dev_alloc<uint16_t>(M * I);
}

MimoDenseMlp::~MimoDenseMlp() {
  cudaFree(gate_);
  cudaFree(up_);
  cudaFree(act_);
}

void MimoDenseMlp::rebind(const MimoDenseMlpResident& w) {
  if (w.local_inter != inter_) throw std::invalid_argument("MimoDenseMlp: rebind changes the inter slice");
  w_ = w;
}

size_t MimoDenseMlp::scratch_bytes(const MimoTextConfig&, int64_t local_inter, int max_tokens) {
  return 3 * static_cast<size_t>(std::max(max_tokens, 0)) * static_cast<size_t>(local_inter) * 2;
}

void MimoDenseMlp::enqueue(const uint16_t* x, int tokens, uint16_t* out, cudaStream_t stream,
                           int decode_prefix) {
  if (decode_prefix < 0 || decode_prefix > std::max(0, tokens) || decode_prefix > decode_rows_)
    throw std::invalid_argument("MimoDenseMlp: mixed decoder prefix exceeds row capacity");
  if (decode_prefix && decode_prefix < tokens) {
    if (tokens > max_tokens_ || !x || !out)
      throw std::invalid_argument("MimoDenseMlp: invalid mixed rows");
    enqueue(x, decode_prefix, out, stream);
    const size_t offset = static_cast<size_t>(decode_prefix) * hidden_;
    enqueue(x + offset, tokens - decode_prefix, out + offset, stream);
    return;
  }
  if (tokens <= 0) return;
  if (tokens > max_tokens_) throw std::invalid_argument("MimoDenseMlp: tokens exceed max_tokens");
  if (!x || !out) throw std::invalid_argument("MimoDenseMlp: null pointer");
  const int H = hidden_, I = static_cast<int>(inter_);
  const float no_limit = std::numeric_limits<float>::infinity();
  const bool mma = decode_mma_ && tokens <= decode_rows_;
  launch_scale_gemm_grid_bf16(x, static_cast<size_t>(H), w_.gate.payload, w_.gate.scales, gate_, tokens, I, H, stream, 0,
                              7, 7, mma);
  launch_scale_gemm_grid_bf16(x, static_cast<size_t>(H), w_.up.payload, w_.up.scales, up_, tokens, I, H, stream, 0, 7, 7,
                              mma);
  launch_moe_swiglu_clamp(gate_, up_, act_, static_cast<int64_t>(tokens) * I, no_limit, stream);
  launch_scale_gemm_grid_bf16(act_, static_cast<size_t>(I), w_.down.payload, w_.down.scales, out, tokens, H, I, stream, 0,
                              7, 7, mma);
}

}  // namespace dgpp
