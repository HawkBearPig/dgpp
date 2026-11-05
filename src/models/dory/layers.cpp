#include "models/dory/layers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "kernels/dory_attn.hpp"
#include "kernels/dory_norm.hpp"
#include "kernels/fp4_gemv.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "models/dory/reference.hpp"

namespace dgpp {
namespace {

template <class T>
T* dev_alloc(size_t n) {
  T* p = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  return p;
}

// One packed NVFP4 projection: the GEMV core on decode rows (chunked rows
// through the same kernel), the dense MMA on prefill rows.
void project(const uint16_t* act, int k, const GlmFp4Matrix& w, uint16_t* out, int m, int n,
             int decode_rows, cudaStream_t stream) {
  if (m <= decode_rows)
    launch_fp4_gemv_bf16(act, static_cast<size_t>(k), w, out, m, n, k, stream);
  else
    launch_dense_mma_fp4_bf16(act, static_cast<size_t>(k), w, out, m, n, k, stream);
}

}  // namespace

void dory_rope_inv_freq(double theta, int dim, float* out) {
  for (int i = 0; i < dim / 2; ++i)
    out[i] = static_cast<float>(std::pow(theta, -2.0 * static_cast<double>(i) / dim));
}

// ---- DoryAttentionLayer ----------------------------------------------------------

int DoryAttentionLayer::default_decode_splits() {
  // The partial's grid is rows x n_split x kv_heads; a single decode row is
  // only 4 kv-head blocks, so the split count is what fills the device (a
  // 2K-token history is 64 tiles and saturates around 16).
  if (const char* v = std::getenv("DGPP_DORY_ATTN_SPLITS"); v && *v) {
    const int n = std::atoi(v);
    if (n > 0) return n;
  }
  return 16;
}

DoryAttentionLayer::DoryAttentionLayer(const DoryAttentionResident& w, const DoryConfig& cfg, int layer,
                                       int max_tokens, int decode_rows, int n_split_decode,
                                       cudaStream_t stream)
    : w_(&w), cfg_(cfg), hidden_(cfg.hidden_size), lh_(cfg.num_attention_heads),
      lkv_(cfg.num_key_value_heads), dim_(cfg.head_dim), max_tokens_(max_tokens),
      decode_rows_(decode_rows), n_split_(n_split_decode), layer_(layer),
      scale_(static_cast<float>(std::pow(static_cast<double>(cfg.head_dim), -0.5))),
      sqk_factor_(cfg.sqk_factor()), s_gate_factor_(cfg.s_gate_factor()),
      alpha_scale_(cfg.alpha_scale()), eps_(kDoryJustnormEps) {
  if (max_tokens_ <= 0) throw std::invalid_argument("DoryAttentionLayer: max_tokens must be positive");
  if (decode_rows_ < 0) throw std::invalid_argument("DoryAttentionLayer: decode_rows");
  if (n_split_ <= 0) throw std::invalid_argument("DoryAttentionLayer: n_split_decode must be positive");
  if (lh_ <= 0 || lkv_ <= 0 || lh_ % lkv_ != 0)
    throw std::invalid_argument("DoryAttentionLayer: query heads must be a multiple of kv heads");
  if (dim_ != kDoryHeadDim)
    throw std::invalid_argument("DoryAttentionLayer: 256-wide heads (the kernels' shape)");
  if (lh_ / lkv_ > 32) throw std::invalid_argument("DoryAttentionLayer: at most 32 query heads per kv head");
  if (layer < 0 || layer >= cfg.num_hidden_layers || !cfg.is_attention(layer))
    throw std::invalid_argument("DoryAttentionLayer: layer " + std::to_string(layer) + " is not an attention layer");
  if (!fp4_gemv_accepts(w.q_proj) || !fp4_gemv_accepts(w.o_proj))
    throw std::invalid_argument("DoryAttentionLayer: the fp4 GEMV core does not accept these widths");
  rebind(w, layer, stream);
  const size_t M = static_cast<size_t>(max_tokens_);
  const size_t Q = static_cast<size_t>(lh_) * dim_, KV = static_cast<size_t>(lkv_) * dim_;
  qd_ = dev_alloc<uint16_t>(M * Q);
  kd_ = dev_alloc<uint16_t>(M * KV);
  vd_ = dev_alloc<uint16_t>(M * KV);
  q_ = dev_alloc<uint16_t>(M * Q);
  gate_ = dev_alloc<uint16_t>(M * Q);
  attn_ = dev_alloc<uint16_t>(M * Q);
  normed_ = dev_alloc<uint16_t>(M * Q);
  o_ = dev_alloc<uint16_t>(M * hidden_);
  part_rows_ = std::max(M, static_cast<size_t>(decode_rows_) * static_cast<size_t>(n_split_));
  const size_t part = part_rows_ * static_cast<size_t>(lh_);
  m_ws_ = dev_alloc<float>(part);
  l_ws_ = dev_alloc<float>(part);
  c_ws_ = dev_alloc<float>(part * static_cast<size_t>(dim_));
  // The two rope tables, built once (the decode walk must copy nothing; see
  // the members' comment). The blocking uploads run at build time, outside
  // any capture, on the default stream's order.
  std::vector<float> inv(static_cast<size_t>(dim_) / 2);
  dory_rope_inv_freq(cfg_.rope_theta_full, dim_, inv.data());
  d_inv_freq_full_ = dev_alloc<float>(inv.size());
  DGPP_CUDA_OK(cudaMemcpy(d_inv_freq_full_, inv.data(), inv.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
  if (cfg_.rope_theta_sliding == cfg_.rope_theta_full) {
    d_inv_freq_sliding_ = d_inv_freq_full_;
  } else {
    dory_rope_inv_freq(cfg_.rope_theta_sliding, dim_, inv.data());
    d_inv_freq_sliding_ = dev_alloc<float>(inv.size());
    DGPP_CUDA_OK(cudaMemcpy(d_inv_freq_sliding_, inv.data(), inv.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
  }
  d_inv_freq_ = d_inv_freq_sliding_;
}

DoryAttentionLayer::~DoryAttentionLayer() {
  if (d_inv_freq_sliding_ && d_inv_freq_sliding_ != d_inv_freq_full_) cudaFree(d_inv_freq_sliding_);
  cudaFree(d_inv_freq_full_);
  cudaFree(qd_);
  cudaFree(kd_);
  cudaFree(vd_);
  cudaFree(q_);
  cudaFree(gate_);
  cudaFree(attn_);
  cudaFree(normed_);
  cudaFree(o_);
  cudaFree(m_ws_);
  cudaFree(l_ws_);
  cudaFree(c_ws_);
}

void DoryAttentionLayer::rebind(const DoryAttentionResident& w, int layer, cudaStream_t stream) {
  (void)stream;  // nothing uploads any more: the walk is capture-clean
  w_ = &w;
  layer_ = layer;
  window_ = cfg_.is_sliding(layer) ? cfg_.sliding_window : 0;
  theta_ = cfg_.rope_theta_of(layer);
  // The theta table selection: the two are staged at construction, so a
  // rebind inside a decode-graph capture records no memcpy node.
  d_inv_freq_ = cfg_.is_full_attention(layer) ? d_inv_freq_full_ : d_inv_freq_sliding_;
}

size_t DoryAttentionLayer::scratch_bytes(const DoryConfig& cfg, int max_tokens, int decode_rows,
                                         int n_split_decode) {
  const size_t M = static_cast<size_t>(std::max(max_tokens, 0));
  const size_t Q = static_cast<size_t>(cfg.num_attention_heads) * cfg.head_dim;
  const size_t KV = static_cast<size_t>(cfg.num_key_value_heads) * cfg.head_dim;
  const size_t part_rows =
      std::max(M, static_cast<size_t>(std::max(decode_rows, 0)) * static_cast<size_t>(n_split_decode));
  const size_t part = part_rows * static_cast<size_t>(cfg.num_attention_heads);
  // The first term: the rope tables staged at construction (one when the
  // two thetas agree, two when they do not).
  const size_t theta_tables = cfg.rope_theta_sliding == cfg.rope_theta_full ? 1 : 2;
  return theta_tables * static_cast<size_t>(cfg.head_dim / 2) * 4 + M * Q * 2 + 2 * M * KV * 2 + 4 * M * Q * 2 +
         M * cfg.hidden_size * 2 + part * 4 * 2 + part * cfg.head_dim * 4;
}

void DoryAttentionLayer::enqueue(const uint16_t* x, int tokens, const DoryAttnRows& rows,
                                 const DoryKvCache& cache, uint16_t* out, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (tokens > max_tokens_) throw std::invalid_argument("DoryAttentionLayer: tokens exceed max_tokens");
  if (!x || !out || !rows.req_ids || !rows.pos || !cache.k_cache || !cache.v_cache || !cache.block_tables)
    throw std::invalid_argument("DoryAttentionLayer: null pointer");
  if (rows.split(x, tokens, hidden_, out, [&](const uint16_t* input, int n, const auto& map,
                                              uint16_t* output) {
        enqueue(input, n, map, cache, output, stream);
      }))
    return;
  const DoryAttentionResident& w = *w_;
  const int H = hidden_, Q = lh_ * dim_, KV = lkv_ * dim_;
  project(x, H, w.q_proj, qd_, tokens, Q, decode_rows_, stream);
  project(x, H, w.k_proj, kd_, tokens, KV, decode_rows_, stream);
  project(x, H, w.v_proj, vd_, tokens, KV, decode_rows_, stream);
  project(x, H, w.g_proj, gate_, tokens, Q, decode_rows_, stream);
  // RoPE, the per-head justnorm, the sqk square, the K/V append.
  dory_qkv_finish(qd_, Q, kd_, KV, vd_, KV, w.sqk, sqk_factor_, d_inv_freq_, eps_, rows.req_ids,
                  rows.pos, tokens, lh_, lkv_, cache.block_tables, cache.blocks_per_request,
                  cache.block_tokens, q_, Q, cache.k_cache, cache.v_cache, stream);
  // The attention: decode rows split the visible range, prefill rows one
  // split each (T x kv-head blocks).
  const int n_split = rows.decode ? n_split_ : 1;
  if (static_cast<size_t>(tokens) * static_cast<size_t>(n_split) > part_rows_)
    throw std::invalid_argument("DoryAttentionLayer: rows exceed the layer's part_rows");
  dory_attn_partial(q_, Q, cache.k_cache, cache.v_cache, rows.req_ids, rows.pos, tokens, n_split,
                    lh_, lkv_, cache.block_tokens, cache.block_tables, cache.blocks_per_request,
                    window_, scale_, m_ws_, l_ws_, c_ws_, stream);
  dory_attn_combine(m_ws_, l_ws_, c_ws_, tokens, n_split, lh_, attn_, stream);
  // The output gate, the 4096-wide justnorm, o_proj, the residual.
  dory_gate_justnorm_bf16(attn_, gate_, w.s_gate, s_gate_factor_, normed_, tokens, Q, eps_, stream);
  project(normed_, Q, w.o_proj, o_, tokens, H, decode_rows_, stream);
  dory_slerp_alpha_bf16(x, o_, w.attn_alpha, alpha_scale_, out, tokens, H, eps_, stream);
}

// ---- DoryMlpLayer ----------------------------------------------------------------

DoryMlpLayer::DoryMlpLayer(const DoryMlpResident& w, const DoryConfig& cfg, int max_tokens,
                           int decode_rows)
    : w_(&w), cfg_(cfg), hidden_(cfg.hidden_size), inter_(cfg.intermediate_size),
      max_tokens_(max_tokens), decode_rows_(decode_rows), suv_factor_(cfg.suv_factor()),
      alpha_scale_(cfg.alpha_scale()), eps_(kDoryJustnormEps) {
  if (max_tokens_ <= 0) throw std::invalid_argument("DoryMlpLayer: max_tokens must be positive");
  if (decode_rows_ < 0) throw std::invalid_argument("DoryMlpLayer: decode_rows");
  if (!fp4_gemv_accepts(w.gate) || !fp4_gemv_accepts(w.up) || !fp4_gemv_accepts(w.down))
    throw std::invalid_argument("DoryMlpLayer: the fp4 GEMV core does not accept these widths");
  const size_t M = static_cast<size_t>(max_tokens_), I = static_cast<size_t>(inter_);
  gate_ = dev_alloc<uint16_t>(M * I);
  up_ = dev_alloc<uint16_t>(M * I);
  m_ = dev_alloc<uint16_t>(M * I);
  o_ = dev_alloc<uint16_t>(M * hidden_);
}

DoryMlpLayer::~DoryMlpLayer() {
  cudaFree(gate_);
  cudaFree(up_);
  cudaFree(m_);
  cudaFree(o_);
}

void DoryMlpLayer::rebind(const DoryMlpResident& w) { w_ = &w; }

size_t DoryMlpLayer::scratch_bytes(const DoryConfig& cfg, int max_tokens) {
  const size_t M = static_cast<size_t>(std::max(max_tokens, 0));
  return 4 * M * static_cast<size_t>(cfg.intermediate_size) * 2 + M * cfg.hidden_size * 2;
}

void DoryMlpLayer::enqueue(const uint16_t* x, int tokens, uint16_t* out, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (tokens > max_tokens_) throw std::invalid_argument("DoryMlpLayer: tokens exceed max_tokens");
  if (!x || !out) throw std::invalid_argument("DoryMlpLayer: null pointer");
  const DoryMlpResident& w = *w_;
  const int H = hidden_, I = inter_;
  project(x, H, w.gate, gate_, tokens, I, decode_rows_, stream);
  project(x, H, w.up, up_, tokens, I, decode_rows_, stream);
  dory_silu_justnorm_bf16(gate_, up_, w.suv_gate, w.suv_up, suv_factor_, m_, tokens, I, eps_, stream);
  project(m_, I, w.down, o_, tokens, H, decode_rows_, stream);
  dory_slerp_alpha_bf16(x, o_, w.mlp_alpha, alpha_scale_, out, tokens, H, eps_, stream);
}

}  // namespace dgpp
