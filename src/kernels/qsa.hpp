#pragma once
// Qwen Sparse Attention (QSA) kernels (Q3, 2026-09-09; docs/qwen38_flash_next_plan.md
// §1.4, D5; the reference is transformers Qwen4ExpTextAttention +
// Qwen4ExpTextQSAIndexer over Qwen3_5Attention). The projections run
// through the GEMM interface; these kernels are the rest:
//
//   norm+RoPE   per-head (1+w) RMSNorm, then RoPE on the first rotary_dim
//               dims with the reference's bf16 ops (cos/sin bf16 tables
//               from fp32 pos x inv_freq; x*cos, rotate_half(x)*sin, the
//               sum — three roundings);
//   kv append   the rank's kv heads' K (normed+roped) and V rows into the
//               paged bf16 caches;
//   indexer     raw keys into a per-request ring (the last kpool), every
//               completed block of kpool tokens compressed ONCE —
//               bf16(mean_fp32) -> (1+w) RMSNorm -> RoPE at the block's
//               first position — into the paged compressed-key cache
//               (co-located with the token blocks: pools_per_block =
//               block_tokens / kpool);
//   score       composite keys (highest score first, ties to the lower
//               block) over every visible block of every row, the score
//               sum_h relu(<q_h, c_b>) / sqrt(dim) in a FIXED fp32 order
//               the host oracle reproduces bit for bit;
//   select      the select_k smallest keys per row, expanded to token
//               positions ascending, the incomplete tail appended;
//   attention   split-KV listed GQA attention over the paged K/V caches
//               (the DSA partial layout: merge with dsa_attn_combine);
//   gate        out = bf16(bf16(c) x bf16(sigmoid(gate))).
//
// Every kernel is deterministic and capturable (fixed grids, no host reads).
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// inv_freq[i] = 1 / theta^(2i / rotary_dim), i < rotary_dim / 2, in fp32 as
// the reference computes it (host helper; upload the result). The YaRN
// variant of the table is yarn_rope_inv_freq_host (kernels/rope_scaling.hpp),
// which the layer builds instead when engine.rope_scaling is set.
void qsa_rope_inv_freq(double theta, int rotary_dim, float* inv_freq);

// Every rotation below takes `mscale`: the YaRN attention factor the cos
// and sin are built with (vLLM bakes it into its cos/sin cache — its
// attention scale stays head_dim^-0.5). 1.0f is the plain rope, bit for
// bit: the kernel multiplies the fp32 cosf/sinf by it before the bf16
// rounding, and x * 1.0f is exact.
//
// out[r, h, :] = RoPE(RMSNorm_dim(x[r, h, :]) x (1 + w), pos[r]) for heads
// [0, heads): x head h of row r at x + r * x_row_stride + h * x_head_stride
// (the attention's [q | gate] interleave: head stride 2 * dim); out rows
// out_row_stride apart, heads contiguous. Rows with pos < 0 are skipped.
void qsa_norm_rope_bf16(const uint16_t* x, int64_t x_row_stride, int64_t x_head_stride,
                        const uint16_t* w, const int64_t* pos, const float* inv_freq,
                        uint16_t* out, int64_t out_row_stride, int rows, int heads,
                        int dim, int rotary_dim, float eps, float mscale, cudaStream_t stream);

// k / v: bf16 [rows, kv_heads * dim] (row strides in elements) into the
// caches at each row's physical slot (block_tables[req][pos / block_tokens]
// * block_tokens + pos % block_tokens); caches bf16 [slots, kv_heads * dim].
void qsa_kv_append(const uint16_t* k, int64_t k_row_stride, const uint16_t* v,
                   int64_t v_row_stride, const int32_t* req_ids, const int64_t* pos,
                   int rows, const int32_t* block_tables, int blocks_per_request,
                   int block_tokens, int kv_heads, int dim, uint16_t* k_cache,
                   uint16_t* v_cache, cudaStream_t stream);

// Prefill compression: pools [first_pool, first_pool + n_pools) of one
// request, pool i's kpool raw keys at chunk rows [i * kpool, +kpool) of
// raw_k (row stride k_stride). The compressed key lands at the pool's slot
// through the block table; its RoPE position is pool * kpool.
void qsa_index_compress_write(const uint16_t* raw_k, int64_t k_stride, const uint16_t* w_k,
                              const float* inv_freq, const int32_t* block_table,
                              int pools_per_block, int64_t first_pool, int n_pools,
                              uint16_t* index_cache, int kpool, int dim, int rotary_dim,
                              float eps, float mscale, cudaStream_t stream);

// Seed each request's ring (bf16 [max_requests, kpool, dim]) with its last
// kpool raw keys of the batch (the ahead-check rule of the DSA seed).
void qsa_index_tail_seed(const uint16_t* raw_k, int64_t k_stride, const int32_t* req_ids,
                         const int64_t* pos, int64_t tokens, uint16_t* ring, int kpool,
                         int dim, cudaStream_t stream);

// Decode update: one block per request span (req_spans int32 [num, 2]:
// start, len into the batch; tokens of a request contiguous, in position
// order; pos < 0 rows skipped). Each token stashes its raw key in the
// ring at pos % kpool; a token completing a pool (pos % kpool == kpool - 1)
// first compresses the pool from the ring with itself in its own slot and
// writes the block-mapped slot. ring_snapshots (optional, bf16 [tokens,
// kpool, dim]): after every row that is not its span's last, the ring as
// it stands — the state to restore when later rows are rejected.
void qsa_index_decode_update(const uint16_t* raw_k, int64_t k_stride, const uint16_t* w_k,
                             const float* inv_freq, const int32_t* req_ids,
                             const int64_t* pos, const int32_t* req_spans, int num_requests,
                             const int32_t* block_tables, int blocks_per_request,
                             uint16_t* ring, uint16_t* index_cache, int pools_per_block,
                             int kpool, int dim, int rotary_dim, float eps, float mscale,
                             cudaStream_t stream, uint16_t* ring_snapshots = nullptr);

// keys_ws[r * ws_stride + b] = (~sortable(score) << kIdxBits) | b for every
// visible pool b < (pos[r] + 1) / kpool of every row (pos < 0: none). q:
// bf16 [rows, heads * dim] with heads <= 4 and dim == 128 (the lane
// layout; absent heads contribute nothing). The score's fp32 order: lane (h, c) sums dims [16c, 16c + 16)
// with fma, a 3-level xor tree over the head's 8 lanes, relu, a 2-level
// tree over the heads, one division by sqrt(dim). Grid: pool stripes x
// rows. Every row's visible count must fit ws_stride.
void qsa_index_score(const uint16_t* q, int64_t q_row_stride, const int32_t* req_ids,
                     const int64_t* pos, int rows, const int32_t* block_tables,
                     int blocks_per_request, const uint16_t* index_cache,
                     int pools_per_block, int heads, int dim, int kpool, uint64_t* keys_ws,
                     int64_t ws_stride, cudaStream_t stream);

// One block per row: the select_k smallest keys of keys_ws[r, 0..visible)
// (the streaming composite-key top-k), their pools expanded to tokens in
// ascending order, the row's incomplete tail appended; topk_out int32
// [rows, max_selected] (-1 padded), out_counts [rows]. select_k <= 1024.
void qsa_select_from_keys(const uint64_t* keys_ws, int64_t ws_stride, const int64_t* pos,
                          int rows, int select_k, int kpool, int max_selected,
                          int32_t* topk_out, int32_t* out_counts, cudaStream_t stream);

// Listed GQA attention partials: one block per (row, split, head group of
// hpb heads sharing a kv head); q bf16 rows of local_heads x dim (row
// stride q_row_stride, heads contiguous); caches as qsa_kv_append writes
// them (kv_heads local); every query head h reads kv head h / (local_heads
// / kv_heads). Probabilities round to bf16 for the V accumulation, l stays
// unrounded (the DSA pin). m_ws / l_ws fp32 [rows, n_split, local_heads],
// c_ws fp32 [rows, n_split, local_heads, dim]; merge with dsa_attn_combine
// (kv_lora = dim). dim in {256, 512}.
void qsa_attn_partial(const uint16_t* q, int64_t q_row_stride, const uint16_t* k_cache,
                      const uint16_t* v_cache, const int32_t* req_ids, const int32_t* topk,
                      int topk_stride, const int32_t* counts, int rows, int n_split,
                      int local_heads, int kv_heads, int dim, int block_tokens,
                      const int32_t* block_tables, int blocks_per_request, float scale,
                      float* m_ws, float* l_ws, float* c_ws, cudaStream_t stream);

// Prefill variant with wider KV sharing and cooperative warp softmax at dim=256;
// other dimensions use qsa_attn_partial. Identical split/tile arithmetic and
// workspace layout. The caller retains the decode kernel for small prefill grids.
void qsa_attn_prefill_partial(const uint16_t* q, int64_t q_row_stride, const uint16_t* k_cache,
                      const uint16_t* v_cache, const int32_t* req_ids, const int32_t* topk,
                      int topk_stride, const int32_t* counts, int rows, int n_split,
                      int local_heads, int kv_heads, int dim, int block_tokens,
                      const int32_t* block_tables, int blocks_per_request, float scale,
                      float* m_ws, float* l_ws, float* c_ws, cudaStream_t stream);

// out[r, h * dim + d] = bf16(bf16(c[r, h, d]) x bf16(sigmoid(gate))) with
// the gate of head h at gate + r * gate_row_stride + h * gate_head_stride.
void qsa_gate_out(const float* c, const uint16_t* gate, int64_t gate_row_stride,
                  int64_t gate_head_stride, uint16_t* out, int rows, int heads, int dim,
                  cudaStream_t stream);

}  // namespace dgpp
