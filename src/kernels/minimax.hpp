#pragma once
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {
struct MoeSegment;
// Stable compaction of nonempty routed segments. Capacity bounds the number
// of nonempty experts (min(experts, tokens * top_k)); unused entries are zero.
// Row offsets and expert ids are preserved, so the grouped MMA chain is unchanged.
void minimax_compact_segments(const MoeSegment* input, MoeSegment* output, int experts,
                              int capacity, cudaStream_t);
// Save the BF16 residual, but normalize its FP32 pre-rounding sum, matching
// MiniMax's fused Gemma normalization. Both outputs have one BF16 rounding.
void minimax_add_norm(uint16_t* residual, const uint16_t* add, const uint16_t* weight,
                      uint16_t* out, int rows, int hidden, float eps, cudaStream_t);
void minimax_swiglu(const uint16_t* gate, const uint16_t* up, uint16_t* out, int64_t count,
                    float alpha, float limit, cudaStream_t);
// Upstream CUDA's FP32 power and reciprocal, without a double intermediate.
// count may include unused tail frequencies in a partial-rotary draft table.
void minimax_rope_inv_freq(float* out, int count, int rotary_dim, float theta, cudaStream_t);
// x and y are [rows,heads,128], in-place allowed. Gemma norm rounds before
// the partial rotation; cos/sin and the rotation arithmetic stay FP32.
void minimax_norm_rope(const uint16_t* x, const uint16_t* weight, uint16_t* y, const int64_t* pos,
                       const float* inv_freq, int rows, int heads, int rotary_dim, float eps,
                       cudaStream_t, int64_t x_stride = 0, int64_t y_stride = 0);
// The context FC is rounded to BF16 before Gemma normalization.
void minimax_context_norm(const float*, const uint16_t* weight, uint16_t* out, int rows, int hidden,
                          float eps, cudaStream_t);
// Learned confidence for each proposed token, conditioned on its predecessor.
void minimax_dspark_confidence(const uint16_t* hidden, const uint16_t* predecessor,
                               const int64_t* anchor, const int32_t* proposals, const float* weight,
                               const float* bias, float* out, int drafts, int width, int rank,
                               int vocab, cudaStream_t);
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
