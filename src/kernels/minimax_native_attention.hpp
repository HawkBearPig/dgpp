#pragma once
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {
// Q8KV4 cache layout: [physical_token, kv_head, 72 bytes], with 64 packed
// E2M1 bytes followed by eight E4M3 scales. The outer scale is exactly one.
constexpr int kMinimaxNvfp4HeadBytes = 72;
void minimax_query_fp8(const uint16_t* input, uint8_t* output, int64_t count, cudaStream_t);
void minimax_cache_nvfp4_append(const uint16_t* input, uint8_t* cache, const int32_t* req_ids,
                                const int64_t* positions, const int32_t* block_tables,
                                int table_stride, int rows, int heads, cudaStream_t);

// Q8 index queries against the packed single-head index cache. Writes the
// causally visible page maxima for minimax_index_select. The fixed k32 dot
// chain is shared by single-row, contiguous-prefill and mixed-request tiles.
void minimax_index_nvfp4_score(const uint8_t* q, const uint8_t* index_cache, const int32_t* req_ids,
                               const int64_t* positions, const int32_t* block_tables,
                               int table_stride, int rows, int heads, int max_blocks, float* scores,
                               cudaStream_t);

// MSA's per-page prefill precision formula, applied independently of row
// geometry: Q/K/V E4M3, unnormalized probabilities E4M3 with scale 448,
// normalized page output BF16. For sink_attn_combine, m is page LSE, l is
// one and c is the BF16-rounded page output promoted exactly to FP32.
// q [rows,heads,128]; selected [rows,kv_heads,topk]. One to sixteen query
// heads share each KV head. Invalid/padded rows produce empty partials.
void minimax_attn_nvfp4_partial(const uint8_t* q, const uint8_t* k, const uint8_t* v,
                                const int32_t* req_ids, const int64_t* positions,
                                const int32_t* block_tables, int table_stride,
                                const int32_t* selected, int rows, int heads, int kv_heads,
                                int topk, float* m, float* l, float* c, cudaStream_t);
}  // namespace dgpp
