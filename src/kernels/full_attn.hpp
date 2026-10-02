#pragma once
// Dense causal paged attention for the Qwen3.5 full-attention layers: the
// shape of qsa_warp.cu (one warp per (query, KV group) on the tensor cores,
// FA2 online softmax, fp32 normalized output) with the listed top-k gather
// replaced by a dense causal range. Each row r attends positions [0, pos[r]]
// over the paged K/V cache — no scoring, no selection, no combine pass.
//
// Numerics follow the QSA rule (probabilities rounded to bf16 for P V, the
// denominator summing the unrounded values); tolerance-equal to a dense
// reference, not bitwise.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// dim 256 and at most 16 query heads per KV head.
bool full_attn_supported(int dim, int local_heads, int kv_heads);
// out fp32 [rows, local_heads, 256]. q bf16 rows of local_heads x dim pairs
// ([q|gate] interleave, row stride q_row_stride, heads contiguous); caches as
// qsa_kv_append writes them (kv_heads local); every query head h reads kv
// head h / (local_heads / kv_heads). pos int64 [rows]: each row's position;
// the visible set is [0, pos[r]].
void full_attn_prefill_warp(const uint16_t* q, int64_t q_row_stride, const uint16_t* k_cache,
                            const uint16_t* v_cache, const int32_t* req_ids, const int64_t* pos,
                            int rows, int local_heads, int kv_heads, int block_tokens,
                            const int32_t* block_tables, int blocks_per_request, float scale,
                            float* out, cudaStream_t stream);

}  // namespace dgpp
