#include "models/flint/layers.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/qsa.hpp"
#include "kernels/sink_attn.hpp"

namespace dgpp {
namespace {

constexpr int kQk = 128;  // q / k / v head width (the kernels' qk-128 instantiation)

template <class T>
T* dev_alloc(size_t n) {
  T* p = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  return p;
}

}  // namespace

// ---- FlintAttentionLayer ------------------------------------------------------

int FlintAttentionLayer::decode_splits_of(const FlintAttnResident& w) const {
  if (!w.swa) return n_split_;
  // A sliding row sees at most `window` tokens: ceil(window / tile) tiles.
  return std::max(1, std::min(n_split_, (window_ + kSinkAttnTile - 1) / kSinkAttnTile));
}

FlintAttentionLayer::FlintAttentionLayer(const FlintAttnResident& w, const FlintGemmWorkspace& gemm,
                                         const FlintTextConfig& cfg, int max_tokens, int decode_rows,
                                         int n_split_decode)
    : w_(w), g_(gemm), hidden_(cfg.hidden_size), lh_(w.local_heads), lkv_(w.local_kv_heads),
      max_tokens_(max_tokens), decode_rows_(decode_rows), n_split_(n_split_decode), window_(cfg.sliding_window),
      eps_(cfg.rms_norm_eps) {
  if (!g_.gemm || !g_.ws) throw std::invalid_argument("FlintAttentionLayer: GEMM workspace required");
  if (max_tokens_ <= 0) throw std::invalid_argument("FlintAttentionLayer: max_tokens must be positive");
  if (decode_rows_ < 0) throw std::invalid_argument("FlintAttentionLayer: decode_rows");
  if (n_split_ <= 0) throw std::invalid_argument("FlintAttentionLayer: n_split_decode must be positive");
  if (lh_ <= 0 || lkv_ <= 0 || lh_ % lkv_ != 0)
    throw std::invalid_argument("FlintAttentionLayer: query heads must be a multiple of kv heads");
  if (cfg.head_dim != kQk || !sink_attn_supported(kQk))
    throw std::invalid_argument("FlintAttentionLayer: 128-wide heads (the kernels' shape)");
  if (!w_.qkv || !w_.o_proj || !w_.q_norm || !w_.k_norm)
    throw std::invalid_argument("FlintAttentionLayer: unbound weights");
  if (cfg.num_swa_layers() > 0 && window_ <= 0)
    throw std::invalid_argument("FlintAttentionLayer: the sliding window must be positive");
  // The reference's scale: head_dim ** -0.5 in double, narrowed to fp32.
  scale_ = static_cast<float>(std::pow(static_cast<double>(kQk), -0.5));
  const size_t M = static_cast<size_t>(max_tokens_);
  const size_t Q = static_cast<size_t>(lh_) * kQk;
  {
    // RoPE over the whole head: inv_freq[i] = theta ^ (-2i / 128), i < 64.
    std::vector<float> inv(static_cast<size_t>(kQk / 2));
    qsa_rope_inv_freq(cfg.rope_theta, kQk, inv.data());
    d_inv_freq_ = dev_alloc<float>(inv.size());
    DGPP_CUDA_OK(cudaMemcpy(d_inv_freq_, inv.data(), inv.size() * 4, cudaMemcpyHostToDevice));
  }
  qkv_cols_ = w_.qkv_rows();
  qkv_ = dev_alloc<float>(M * static_cast<size_t>(qkv_cols_));
  q_ = dev_alloc<uint16_t>(M * Q);
  part_rows_ = std::max(M, static_cast<size_t>(decode_rows_) * static_cast<size_t>(n_split_));
  const size_t part = part_rows_ * static_cast<size_t>(lh_);
  m_ws_ = dev_alloc<float>(part);
  l_ws_ = dev_alloc<float>(part);
  c_ws_ = dev_alloc<float>(part * kSinkAttnVDim);
  o_ = dev_alloc<uint16_t>(M * Q);
  // The fused decode kernel's arrival counters: one per (decode row, kv
  // head), zero between launches (the kernel resets its own).
  const size_t counters = static_cast<size_t>(std::max(decode_rows_, 1)) * static_cast<size_t>(lkv_);
  counters_ = dev_alloc<int>(counters);
  DGPP_CUDA_OK(cudaMemset(counters_, 0, counters * sizeof(int)));
}

FlintAttentionLayer::~FlintAttentionLayer() {
  cudaFree(d_inv_freq_);
  cudaFree(qkv_);
  cudaFree(q_);
  cudaFree(m_ws_);
  cudaFree(l_ws_);
  cudaFree(c_ws_);
  cudaFree(counters_);
  cudaFree(o_);
}

void FlintAttentionLayer::rebind(const FlintAttnResident& w) {
  if (w.local_heads != lh_ || w.local_kv_heads != lkv_)
    throw std::invalid_argument("FlintAttentionLayer: rebind changes the head geometry");
  if (!w.qkv || !w.o_proj || !w.q_norm || !w.k_norm) throw std::invalid_argument("FlintAttentionLayer: unbound weights");
  w_ = w;
}

size_t FlintAttentionLayer::scratch_bytes(const FlintTextConfig& cfg, int local_heads, int local_kv_heads,
                                          int max_tokens, int decode_rows, int n_split_decode) {
  const size_t M = static_cast<size_t>(std::max(max_tokens, 0));
  const size_t Q = static_cast<size_t>(local_heads) * kQk;
  const size_t cols = static_cast<size_t>(local_heads + 2 * local_kv_heads) * cfg.head_dim;
  const size_t part_rows =
      std::max(M, static_cast<size_t>(std::max(decode_rows, 0)) * static_cast<size_t>(n_split_decode));
  const size_t part = part_rows * static_cast<size_t>(local_heads);
  return static_cast<size_t>(kQk / 2) * 4 + M * cols * 4 + M * Q * 2 + part * 4 * 2 + part * kSinkAttnVDim * 4 +
         M * Q * 2 + static_cast<size_t>(std::max(decode_rows, 1)) * static_cast<size_t>(local_kv_heads) * 4;
}

void FlintAttentionLayer::enqueue(const uint16_t* x, int tokens, const FlintAttnRows& rows,
                                  const FlintKvCache& cache, uint16_t* out, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (tokens > max_tokens_) throw std::invalid_argument("FlintAttentionLayer: tokens exceed max_tokens");
  if (!x || !out || !rows.req_ids || !rows.pos || !cache.k_cache || !cache.v_cache || !cache.block_tables)
    throw std::invalid_argument("FlintAttentionLayer: null pointer");
  if (cache.kv_heads != lkv_)
    throw std::invalid_argument("FlintAttentionLayer: the cache layer's kv heads differ from the weights'");
  if (cache.fp8() != (cache.k_scale != nullptr && cache.v_scale != nullptr))
    throw std::invalid_argument("FlintAttentionLayer: an fp8 cache carries its scale planes");
  if (rows.split(x, tokens, hidden_, out, [&](const uint16_t* input, int n, const auto& map, uint16_t* output) {
        enqueue(input, n, map, cache, output, stream);
      }))
    return;
  const int H = hidden_;
  const int Q = lh_ * kQk;
  // The fused projection, fp32 out (the finish rounds once): the GEMM
  // interface's lowering — the 12-bit companion's GEMV at decode rows,
  // cuBLASLt above them.
  g_.gemm->matmul(x, w_.qkv, qkv_, tokens, static_cast<int>(qkv_cols_), H, DType::BF16, GemmOut::F32,
                  static_cast<size_t>(H), g_.ws, g_.ws_bytes, stream);
  SinkQkvLayout layout;
  layout.chunks = 1;
  layout.chunk_stride = qkv_cols_;
  layout.q_per_chunk = lh_;
  layout.kv_per_chunk = lkv_;
  SinkFinishParams fin;
  fin.inv_freq = w_.rope ? d_inv_freq_ : nullptr;
  fin.q_norm = w_.q_norm;
  fin.k_norm = w_.k_norm;
  fin.eps = eps_;
  fin.value_scale = 1.0f;
  const int window = w_.swa ? window_ : 0;
  // The attention: decode rows split the visible range (the capture's
  // geometry), prefill rows run one split each (T x kv-head blocks).
  const int n_split = rows.decode ? decode_splits_of(w_) : 1;
  if (static_cast<size_t>(tokens) * static_cast<size_t>(n_split) > part_rows_)
    throw std::invalid_argument("FlintAttentionLayer: decode rows exceed the layer's decode_rows");
  const auto finish = [&] {
    sink_qkv_finish(kQk, qkv_, qkv_cols_, layout, fin, rows.req_ids, rows.pos, tokens, lh_, lkv_, cache.block_tables,
                    cache.blocks_per_request, cache.block_tokens, q_, Q, cache.k_cache, cache.v_cache, stream,
                    cache.k_scale, cache.v_scale);
  };
  if (rows.cache_only) {
    finish();
    return;
  }
  const int hpk = lh_ / lkv_;
  if (rows.decode && tokens <= kSinkAttnFusedMaxRows && tokens <= decode_rows_ && hpk >= 4 && hpk <= 16) {
    // A decode batch: finish + partials + combine in one launch.
    SinkAttnFusedArgs a;
    a.qkv = qkv_;
    a.qkv_stride = qkv_cols_;
    a.layout = layout;
    a.inv_freq = fin.inv_freq;
    a.q_norm = fin.q_norm;
    a.k_norm = fin.k_norm;
    a.eps = fin.eps;
    a.value_scale = 1.0f;
    a.req_ids = rows.req_ids;
    a.pos = rows.pos;
    a.rows = tokens;
    a.n_split = n_split;
    a.local_heads = lh_;
    a.kv_heads = lkv_;
    a.block_tables = cache.block_tables;
    a.blocks_per_request = cache.blocks_per_request;
    a.block_tokens = cache.block_tokens;
    a.window = window;
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
    sink_attn_fused(kQk, a, stream);
  } else if (!rows.decode && kSinkAttnPrefillQueryVecs % hpk == 0) {
    // Prefill rows: every row's K/V appended first, then the query-tiled
    // tensor-core attention.
    finish();
    SinkAttnPrefillArgs a;
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
    a.kv_heads = lkv_;
    a.block_tokens = cache.block_tokens;
    a.block_tables = cache.block_tables;
    a.blocks_per_request = cache.blocks_per_request;
    a.window = window;
    a.scale = scale_;
    a.sink = w_.sink;
    a.out = o_;
    sink_attn_prefill(kQk, a, stream);
  } else {
    finish();
    sink_attn_partial(kQk, q_, Q, cache.k_cache, cache.v_cache, rows.req_ids, rows.pos, tokens, n_split, lh_, lkv_,
                      cache.block_tokens, cache.block_tables, cache.blocks_per_request, window, scale_, w_.sink, m_ws_,
                      l_ws_, c_ws_, stream, cache.k_scale, cache.v_scale);
    sink_attn_combine(m_ws_, l_ws_, c_ws_, tokens, n_split, lh_, o_, stream);
  }
  g_.gemm->matmul(o_, w_.o_proj, out, tokens, H, Q, DType::BF16, GemmOut::BF16, static_cast<size_t>(Q), g_.ws,
                  g_.ws_bytes, stream);
}

// ---- FlintBf16Mlp ---------------------------------------------------------------

FlintBf16Mlp::FlintBf16Mlp(const FlintBf16MlpResident& w, const FlintGemmWorkspace& gemm, const FlintTextConfig& cfg,
                           int max_tokens)
    : w_(w), g_(gemm), hidden_(cfg.hidden_size), max_tokens_(max_tokens), inter_(w.local_inter) {
  if (!g_.gemm || !g_.ws) throw std::invalid_argument("FlintBf16Mlp: GEMM workspace required");
  if (max_tokens_ <= 0) throw std::invalid_argument("FlintBf16Mlp: max_tokens must be positive");
  if (inter_ <= 0 || inter_ % 8 != 0) throw std::invalid_argument("FlintBf16Mlp: the inter slice must be a multiple of 8");
  if (!w_.gate || !w_.up || !w_.down) throw std::invalid_argument("FlintBf16Mlp: unbound weights");
  const size_t M = static_cast<size_t>(max_tokens_), I = static_cast<size_t>(inter_);
  gate_ = dev_alloc<uint16_t>(M * I);
  up_ = dev_alloc<uint16_t>(M * I);
  act_ = dev_alloc<uint16_t>(M * I);
}

FlintBf16Mlp::~FlintBf16Mlp() {
  cudaFree(gate_);
  cudaFree(up_);
  cudaFree(act_);
}

void FlintBf16Mlp::rebind(const FlintBf16MlpResident& w) {
  if (w.local_inter != inter_) throw std::invalid_argument("FlintBf16Mlp: rebind changes the inter slice");
  if (!w.gate || !w.up || !w.down) throw std::invalid_argument("FlintBf16Mlp: unbound weights");
  w_ = w;
}

size_t FlintBf16Mlp::scratch_bytes(int64_t local_inter, int max_tokens) {
  return 3 * static_cast<size_t>(std::max(max_tokens, 0)) * static_cast<size_t>(std::max<int64_t>(local_inter, 0)) * 2;
}

void FlintBf16Mlp::enqueue(const uint16_t* x, int tokens, uint16_t* out, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (tokens > max_tokens_) throw std::invalid_argument("FlintBf16Mlp: tokens exceed max_tokens");
  if (!x || !out) throw std::invalid_argument("FlintBf16Mlp: null pointer");
  const int H = hidden_, I = static_cast<int>(inter_);
  g_.gemm->matmul(x, w_.gate, gate_, tokens, I, H, DType::BF16, GemmOut::BF16, static_cast<size_t>(H), g_.ws,
                  g_.ws_bytes, stream);
  g_.gemm->matmul(x, w_.up, up_, tokens, I, H, DType::BF16, GemmOut::BF16, static_cast<size_t>(H), g_.ws,
                  g_.ws_bytes, stream);
  launch_moe_swiglu_clamp(gate_, up_, act_, static_cast<int64_t>(tokens) * I,
                          std::numeric_limits<float>::infinity(), stream);
  g_.gemm->matmul(act_, w_.down, out, tokens, H, I, DType::BF16, GemmOut::BF16, static_cast<size_t>(I), g_.ws,
                  g_.ws_bytes, stream);
}

}  // namespace dgpp
