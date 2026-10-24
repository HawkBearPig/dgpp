#pragma once
// MiMo-V2.6-Flash attention kernels (2026-09-22, docs/mimo_v26_flash_plan.md
// §1.3, D3; the reference is the release's MiMoV2Attention with the
// engine's flash-style attention numerics — the GLM-4.7 kernels' pin,
// which these fork for the 192-wide qk / 128-wide v heads, the fused
// pre-sharded projection, the sliding window and the attention sink). The
// kernels are the sink-attention family's qk-192 instantiation
// (kernels/sink_attn.hpp, 2026-10-22): the fused qkv projection runs
// through the fp8 GEMV / tile cores with fp32 outputs; these entries are
// the rest —
//
//   qkv finish   per (row, head): bf16(dot) — the Linear's one rounding —
//                then, for the q and k heads, the partial RoPE on the first
//                rotary_dim = 64 dims with the reference's bf16 ops (cos/sin
//                bf16 from fp32 pos x inv_freq; pair (i, i + 32) —
//                transformers' rotate_half over the rotary slice: x*cos and
//                rotate(x)*sin rounded, the sum rounded), the q heads written
//                to a bf16 buffer and the k heads appended to the paged K
//                cache; the v heads scaled — bf16(bf16(dot) x value_scale),
//                the reference's `value_states * v_scale` before the cache —
//                and appended to the V cache at each row's slot;
//   attention    the split-KV partials, the combine, the fused decode
//                launch and the query-tiled prefill (sink_attn.hpp).
#include <cstdint>

#include <cuda_runtime.h>

#include "kernels/sink_attn.hpp"

namespace dgpp {

constexpr int kMimoQkDim = 192;
constexpr int kMimoVDim = kSinkAttnVDim;
constexpr int kMimoRotaryDim = 64;
constexpr int kMimoAttnTile = kSinkAttnTile;
constexpr int kMimoAttnFusedMaxRows = kSinkAttnFusedMaxRows;

using MimoQkvLayout = SinkQkvLayout;
using MimoAttnFusedArgs = SinkAttnFusedArgs;
using MimoAttnPrefillArgs = SinkAttnPrefillArgs;

inline void mimo_qkv_finish(const float* qkv, int64_t qkv_stride, const MimoQkvLayout& layout,
                            const float* inv_freq, float value_scale, const int32_t* req_ids,
                            const int64_t* pos, int rows, int local_heads, int kv_heads,
                            const int32_t* block_tables, int blocks_per_request, int block_tokens,
                            uint16_t* q_out, int64_t q_out_stride, uint16_t* k_cache, uint16_t* v_cache,
                            cudaStream_t stream, float* k_scale = nullptr, float* v_scale = nullptr) {
  SinkFinishParams fin;
  fin.inv_freq = inv_freq;
  fin.value_scale = value_scale;
  sink_qkv_finish(kMimoQkDim, qkv, qkv_stride, layout, fin, req_ids, pos, rows, local_heads, kv_heads, block_tables,
                  blocks_per_request, block_tokens, q_out, q_out_stride, k_cache, v_cache, stream, k_scale, v_scale);
}

inline void mimo_attn_partial(const uint16_t* q, int64_t q_stride, const uint16_t* k_cache,
                              const uint16_t* v_cache, const int32_t* req_ids, const int64_t* pos,
                              int rows, int n_split, int local_heads, int kv_heads, int block_tokens,
                              const int32_t* block_tables, int blocks_per_request, int window,
                              float scale, const float* sink, float* m_ws, float* l_ws, float* c_ws,
                              cudaStream_t stream, const float* k_scale = nullptr,
                              const float* v_scale = nullptr) {
  sink_attn_partial(kMimoQkDim, q, q_stride, k_cache, v_cache, req_ids, pos, rows, n_split, local_heads, kv_heads,
                    block_tokens, block_tables, blocks_per_request, window, scale, sink, m_ws, l_ws, c_ws, stream,
                    k_scale, v_scale);
}

inline void mimo_attn_combine(const float* m_ws, const float* l_ws, const float* c_ws, int rows,
                              int n_split, int local_heads, uint16_t* out, cudaStream_t stream) {
  sink_attn_combine(m_ws, l_ws, c_ws, rows, n_split, local_heads, out, stream);
}

inline size_t mimo_attn_fused_smem_bytes(int local_heads, int kv_heads) {
  return sink_attn_fused_smem_bytes(kMimoQkDim, local_heads, kv_heads);
}
inline void mimo_attn_fused(const MimoAttnFusedArgs& a, cudaStream_t stream) { sink_attn_fused(kMimoQkDim, a, stream); }

inline size_t mimo_attn_prefill_smem_bytes() { return sink_attn_prefill_smem_bytes(kMimoQkDim); }
inline int mimo_attn_prefill_rows_per_block(int local_heads, int kv_heads) {
  return sink_attn_prefill_rows_per_block(local_heads, kv_heads);
}
inline void mimo_attn_prefill(const MimoAttnPrefillArgs& a, cudaStream_t stream) {
  sink_attn_prefill(kMimoQkDim, a, stream);
}

}  // namespace dgpp
