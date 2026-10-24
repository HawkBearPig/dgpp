#include "kernels/sink_attn.hpp"

#include <stdexcept>

#include "kernels/sink_attn_impl.cuh"

// The sink-attention family's entry points: the qk-192 geometry (MiMo:
// rotary 64) and the qk-128 one (Flint: rotary 128, the head norm, the
// no-RoPE form), dispatched on qk_dim. The kernels are
// kernels/sink_attn_impl.cuh's templates.
namespace dgpp {

namespace {

[[noreturn]] void refuse_dim(int qk_dim) {
  throw std::invalid_argument("sink attention: qk width " + std::to_string(qk_dim) +
                              " is not compiled (192 or 128)");
}

}  // namespace

bool sink_attn_supported(int qk_dim) { return qk_dim == 192 || qk_dim == 128; }

void sink_qkv_finish(int qk_dim, const float* qkv, int64_t qkv_stride, const SinkQkvLayout& layout,
                     const SinkFinishParams& fin, const int32_t* req_ids, const int64_t* pos, int rows,
                     int local_heads, int kv_heads, const int32_t* block_tables, int blocks_per_request,
                     int block_tokens, uint16_t* q_out, int64_t q_out_stride, uint16_t* k_cache,
                     uint16_t* v_cache, cudaStream_t stream, float* k_scale, float* v_scale) {
  if (qk_dim == 192)
    sink_attn::qkv_finish<192, 64>(qkv, qkv_stride, layout, fin, req_ids, pos, rows, local_heads, kv_heads, block_tables,
                                   blocks_per_request, block_tokens, q_out, q_out_stride, k_cache, v_cache, stream,
                                   k_scale, v_scale);
  else if (qk_dim == 128)
    sink_attn::qkv_finish<128, 128>(qkv, qkv_stride, layout, fin, req_ids, pos, rows, local_heads, kv_heads, block_tables,
                                    blocks_per_request, block_tokens, q_out, q_out_stride, k_cache, v_cache, stream,
                                    k_scale, v_scale);
  else
    refuse_dim(qk_dim);
}

void sink_attn_partial(int qk_dim, const uint16_t* q, int64_t q_stride, const uint16_t* k_cache,
                       const uint16_t* v_cache, const int32_t* req_ids, const int64_t* pos, int rows,
                       int n_split, int local_heads, int kv_heads, int block_tokens, const int32_t* block_tables,
                       int blocks_per_request, int window, float scale, const float* sink, float* m_ws,
                       float* l_ws, float* c_ws, cudaStream_t stream, const float* k_scale, const float* v_scale) {
  if (qk_dim == 192)
    sink_attn::attn_partial<192>(q, q_stride, k_cache, v_cache, req_ids, pos, rows, n_split, local_heads, kv_heads,
                                 block_tokens, block_tables, blocks_per_request, window, scale, sink, m_ws, l_ws, c_ws,
                                 stream, k_scale, v_scale);
  else if (qk_dim == 128)
    sink_attn::attn_partial<128>(q, q_stride, k_cache, v_cache, req_ids, pos, rows, n_split, local_heads, kv_heads,
                                 block_tokens, block_tables, blocks_per_request, window, scale, sink, m_ws, l_ws, c_ws,
                                 stream, k_scale, v_scale);
  else
    refuse_dim(qk_dim);
}

void sink_attn_combine(const float* m_ws, const float* l_ws, const float* c_ws, int rows, int n_split,
                       int local_heads, uint16_t* out, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!m_ws || !l_ws || !c_ws || !out) throw std::invalid_argument("sink_attn_combine: null pointer");
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(local_heads));
  sink_attn::attn_combine_kernel<<<grid, sink_attn::DV, 0, stream>>>(m_ws, l_ws, c_ws, n_split, local_heads, out);
  DGPP_CUDA_OK(cudaGetLastError());
}

size_t sink_attn_fused_smem_bytes(int qk_dim, int local_heads, int kv_heads) {
  if (qk_dim == 192) return sink_attn::fused_smem_bytes<192>(local_heads, kv_heads);
  if (qk_dim == 128) return sink_attn::fused_smem_bytes<128>(local_heads, kv_heads);
  refuse_dim(qk_dim);
}

void sink_attn_fused(int qk_dim, const SinkAttnFusedArgs& a, cudaStream_t stream) {
  if (qk_dim == 192)
    sink_attn::attn_fused<192, 64>(a, stream);
  else if (qk_dim == 128)
    sink_attn::attn_fused<128, 128>(a, stream);
  else
    refuse_dim(qk_dim);
}

}  // namespace dgpp
