#pragma once
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {
void minimax_swiglu(const uint16_t* gate, const uint16_t* up, uint16_t* out, int64_t count,
                    float alpha, float limit, cudaStream_t);
// x and y are [rows,heads,128], in-place allowed. Gemma norm rounds before
// the partial rotation; cos/sin and the rotation arithmetic stay FP32.
void minimax_norm_rope(const uint16_t* x, const uint16_t* weight, uint16_t* y, const int64_t* pos,
                       const float* inv_freq, int rows, int heads, int rotary_dim, float eps,
                       cudaStream_t);
void minimax_index_append(const uint16_t* index_k, uint16_t* cache, const int32_t* req_ids,
                          const int64_t* pos, const int32_t* tables, int table_stride, int rows,
                          cudaStream_t);
// scores [rows,index_heads,max_blocks], selected [rows,index_heads,topk].
// Every padding row produces -1 selections; no host reads or dynamic grids.
void minimax_index_score(const uint16_t* q, const uint16_t* index_cache, const int32_t* req_ids,
                         const int64_t* pos, const int32_t* tables, int table_stride, int rows,
                         int index_heads, int max_blocks, float* scores, cudaStream_t);
void minimax_index_select(const float* scores, const int64_t* pos, int rows, int index_heads,
                          int max_blocks, int topk, int32_t* selected, cudaStream_t);
// One partial per selected logical block. Workspace [rows,topk,heads],
// with an extra 128-wide axis on c. Merge with sink_attn_combine.
void minimax_attn_partial(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                          const int32_t* req_ids, const int64_t* pos, const int32_t* tables,
                          int table_stride, const int32_t* selected, int rows, int heads,
                          int kv_heads, int topk, float* m, float* l, float* c, cudaStream_t);
// FP32 router weights are never narrowed. Biased scores affect selection
// only; normalized unbiassed sigmoid scores carry routed_scaling_factor.
void minimax_router(const uint16_t* x, const float* gate, const float* bias, int rows, int hidden,
                    int experts, int topk, float factor, float* scores, int32_t* ids,
                    float* weights, cudaStream_t);
}  // namespace dgpp
