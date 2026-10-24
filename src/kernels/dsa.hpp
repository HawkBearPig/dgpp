#pragma once
// DSA/MLA sparse-attention forward kernels, DESIGN §7.2.
//
// Semantics are pinned to the vLLM GLM-5 reference (glm-release fork,
// kpool_compress.py / sparse_attn_indexer_kpool.py); see dsa_reference.hpp
// for the numeric contract this mirrors and the documented power-of-two
// scale divergence.
//
// Cache layout (DGPP's own, not the reference's packed 132-byte pages):
//   * index K cache: planar. `index_k` is FP8-E4M3 [total_slots, 128] and
//     `index_scale` is FP32 [total_slots], one slot per pool. Rows are
//     128-byte aligned, so the decode streaming path reads perfectly aligned
//     lines.
//   * latent cache: [total_token_slots] rows of kv_lora_rank elements in the
//     cache's format (kernels/latent_format.hpp): BF16, or fp8/fp4 codes
//     beside a FP32 [total_token_slots] row-scale array. The attention
//     kernels dequantize a tile into bf16 shared memory as they gather it,
//     so the math past the load is the bf16 kernel's.
//   * tail cache: BF16 [max_requests, 2, kpool, 128] — half 0 is raw K,
//     half 1 the gate score; a per-request ring indexed by pos % kpool.
//   * one shared block table per request maps logical to physical slots:
//     block_tables[req][token_block] for the latent cache, and the same
//     entry's pool range [token_block*kpool .. +kpool) for the index cache
//     (co-located blocks, DESIGN §8).
//
// Decode is the hot path (single-stream goal): the fused select kernel
// streams the index cache once, keeps a running top-select_k composite-key
// selection in shared memory ((sortable_fp32 << 21) | pool_idx — a total
// order, so exact ties resolve to the lower pool index by construction),
// and merges block partials with a last-block reduction. No logits are
// materialized and no per-step allocation happens; grid sizes are fixed
// with grid-stride loops over device-visible counts, so the whole decode
// path is CUDA-graph capturable.
//
// Full GLM long prefill fuses FP8 dots with the head reduction, writing
// compact score keys into the bounded workspace. Flash and short contexts
// keep the IGemm FP8 -> F32 dot path. Both use
// the same exact key selection, merge and token expansion.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "kernels/latent_format.hpp"

namespace dgpp {

// ---- indexer elementwise paths --------------------------------------

// Hadamard-128 + per-row absmax FP8 quant (the indexer q path).
//   q_bf16: [rows, 128] bf16 bits; q_fp8: [rows, 128] fp8 bits;
//   q_scale: [rows] fp32 (power-of-two scales).
void dsa_fwht_quant_rows(const void* q_bf16, int64_t rows, void* q_fp8,
                         float* q_scale, cudaStream_t stream);

// w_folded[i] = (weights[i] * q_scale[i]) * scale, elementwise fp32.
void dsa_fold_weights(const float* weights, const float* q_scale, float* out,
                      int64_t n, float scale, cudaStream_t stream);

// Full LayerNorm (fp32 compute, weight+bias upcast) on a strided input —
// the indexer's k_norm (eps 1e-6).
//   k_raw: [rows, dim] with row stride k_stride elements; k_out: [rows, dim]
//   contiguous bf16.
void dsa_k_layernorm(const void* k_raw, int64_t k_stride, const void* w,
                     const void* b, void* k_out, int64_t rows, int dim,
                     float eps, cudaStream_t stream);

// Fused q_a/kv_a RMSNorm over the split halves of the fused [q_a|kv_a]
// projection (eps 1e-5).
//   qkv: [rows, row_stride] bf16 (row_stride 0 = q_dim + kv_dim; the full
//   model's rows carry the 64-wide rope key after the kv half);
//   q_c/kv_c: [rows, dim] contiguous.
void dsa_fused_qkv_rmsnorm(const void* qkv, void* q_c, void* kv_c, int q_dim,
                           int kv_dim, int64_t rows, const void* q_w,
                           const void* kv_w, float eps, cudaStream_t stream,
                           int64_t row_stride = 0);

// ---- interleaved RoPE (the full GLM-5.3, plan D3) ---------------------
//
// The rotary table: bf16 [positions][2][rope_dim / 2] — for position p,
// row 0 the cosines and row 1 the sines of angle_i = fp32(p) * inv_freq[i],
// inv_freq[i] = fp32(1 / theta^(2i / rope_dim)) (the power in double,
// rounded to fp32 — within an fp32 ulp of transformers' powf), angle_i the
// fp32 product (the reference's), the trig in double rounded to fp32 then
// bf16 — the host, the device and the python reference read one table,
// so the rotation is bitwise across them. Built on the host, uploaded
// once per model; `positions` bounds the cache (DsaLayer's
// max_cache_tokens).
void dsa_rope_table_host(double theta, int rope_dim, int64_t positions,
                         uint16_t* out);

// Rotates the first rope_dim elements of every (row, head) in place or
// into `out`: pair i = (x[2i], x[2i+1]) becomes (bf16(bf16(x0 c) -
// bf16(x1 s)), bf16(bf16(x1 c) + bf16(x0 s))) with (c, s) the table's
// entries for pos[row] — transformers' apply_rotary_pos_emb_interleave in
// bf16 (three roundings), the pair kept in place. Rows with pos < 0 are
// copied unrotated (a fixed-shape batch's padding rows); a position past
// the table reads its last entry (the layer bounds positions at
// admission; this keeps a stale row from faulting).
//   x: bf16 rows at x + r * x_row_stride + h * x_head_stride;
//   out: bf16 at out + r * out_row_stride + h * out_head_stride (out may
//   alias x with equal strides).
void dsa_rope_interleave(const void* x, int64_t x_row_stride,
                         int64_t x_head_stride, int heads, int rope_dim,
                         const int64_t* pos, const void* table,
                         int64_t table_positions, void* out,
                         int64_t out_row_stride, int64_t out_head_stride,
                         int64_t rows, cudaStream_t stream);
// Up to kRopeMaxSegments of the rotation above in one launch (the same
// rows, positions and table; each segment its own x / out / strides /
// heads) — bitwise the separate launches (2026-10-13).
constexpr int kRopeMaxSegments = 4;
struct DsaRopeSegment {
  const void* x = nullptr;
  void* out = nullptr;
  int64_t x_row_stride = 0, x_head_stride = 0, out_row_stride = 0, out_head_stride = 0;
  int heads = 0;
};
void dsa_rope_interleave_multi(const DsaRopeSegment* segs, int count, int rope_dim, const int64_t* pos,
                               const void* table, int64_t table_positions, int64_t rows, cudaStream_t stream);

// ---- index cache + tail state machinery ------------------------------

// Prefill pool compression: one block per pool. Pools [first_pool,
// first_pool + n_pools) are complete pools whose kpool tokens start at
// chunk row (i * kpool) for i in [0, n_pools). Physical slot via the
// request's block table.
//   k/gate: bf16 [tokens, dim], row strides in elements; gate and ape may
//   be null at kpool 1 (a pool is its token: the softmax over one slot is
//   exactly 1, the entry is Hadamard(k) quantized);
//   block_table: int32 [blocks_per_request] (single request);
//   index_k/index_scale: planar cache, physical slot indexing.
void dsa_kpool_compress_write(const void* k, int64_t k_stride,
                              const void* gate, int64_t gate_stride,
                              const float* ape, const int32_t* block_table,
                              int pools_per_block, int64_t first_pool,
                              int n_pools, void* index_k, float* index_scale,
                              int kpool, int dim, cudaStream_t stream);

// Seed the per-request tail ring with each request's last kpool tokens of
// the batch (the reference's ahead-check rule). One block per token.
//   tail: bf16 [max_requests, 2, kpool, dim]; gate may be null (kpool 1:
//   the gate half is zeroed).
void dsa_kpool_tail_seed(const void* k, int64_t k_stride, const void* gate,
                         int64_t gate_stride, const int32_t* req_ids,
                         const int64_t* pos, int64_t tokens, void* tail,
                         int kpool, int dim, cudaStream_t stream);

// Decode update: one block per request, tokens processed in position order
// (read-after-stash within the call). Each token stashes raw K + gate into
// the ring; a token completing its pool (pos % kpool == kpool-1) compresses
// the ring (with the current token overriding its ring slot, exactly the
// reference's is_current rule) and writes the pool to the block-mapped slot.
//   req_spans: int32 [num_requests, 2] (start, len) into the token batch;
//   eager callers may pass only active spans, while a fixed-shape graph may
//   include an all-padding span for every unoccupied configured slot. req_ids
//   selects the actual state slot (active slots need not be
//   0..num_requests-1). Tokens of a request must be batch-contiguous; pos:
//   [tokens] (may be -1 for padding rows — skipped). An all-padding span
//   touches nothing; its req_ids may be a sentinel.
//   tail_snapshots (optional, speculative decode): bf16 [tokens, 2, kpool,
//   dim]; after every batch row t that is not its request's last, the
//   request's ring as it stands is copied to row t. Rolling a request back
//   to `a` accepted rows = copying row (start + a - 1) over its ring.
//   gate and ape may be null at kpool 1 (every token completes its pool).
void dsa_kpool_decode_update(const void* k, int64_t k_stride,
                             const void* gate, int64_t gate_stride,
                             const float* ape, const int32_t* req_ids,
                             const int64_t* pos, const int32_t* req_spans,
                             int num_requests,
                             const int32_t* block_tables,
                             int blocks_per_request, void* tail,
                             void* index_k, float* index_scale,
                             int pools_per_block, int kpool, int dim,
                             cudaStream_t stream,
                             void* tail_snapshots = nullptr);

// Writes zeros to every output row whose position is negative (a fixed-shape
// padding row). The decode kernels skip such rows' state writes and leave
// their attention output unwritten, so without this the row's block output
// is whatever the scratch held; zeroing makes a padding row inert by
// construction (deterministic, finite, rank-identical) rather than merely
// unobserved downstream.
void dsa_zero_padding_rows(void* out, const int64_t* pos, int tokens,
                           int hidden, cudaStream_t stream);

// Append normed latent rows to the blocked latent cache. One block per row.
//   latent_rows: bf16 [tokens, kv_lora]; block_tables: int32
//   [max_requests, blocks_per_request].
// A quantized cache (fp8/fp4) quantizes each row on the way in — the
// codes and the row scale (`latent_scale`, FP32 per physical slot) are
// bitwise the host reference's latent_quantize_row_host.
// A rope tail (rope_rows: bf16 [tokens, rope_dim], the roped k_rot) is
// stored bf16 after the row's payload in every format — the cache row is
// latent_row_bytes(format, kv_lora) + rope_dim * 2 bytes (DsaGeometry's
// latent_bytes_per_token).
void dsa_latent_append(const void* latent_rows, const int32_t* req_ids,
                       const int64_t* pos, int64_t tokens,
                       const int32_t* block_tables, int blocks_per_request,
                       int block_tokens, void* latent_cache, int kv_lora,
                       cudaStream_t stream,
                       LatentFormat format = LatentFormat::kBf16,
                       float* latent_scale = nullptr,
                       const void* rope_rows = nullptr, int rope_dim = 0);

// The decode walk's tail behind the q projection in one launch per row
// (2026-10-20, kernels/dsa.cu decode_tail_fp8block_kernel): the q rope over
// q_rope_dim elements (q_rope_dim / 2 pairs) of each of q_heads heads at
// q + head * q_head_stride, the row's ring slot
// into `slots` (pos % ring_slots, -1 for a negative position), the window
// slot list and counts, the entry position into `entry_out` when it is
// given (ratio > 0), and the latent row's fp8-block append into the ring at
// that slot (ring_table: the ring's block table, one block of ring_slots
// tokens per request). Bitwise the five kernels it replaces.
void dsa_decode_tail_fp8block(void* q, int64_t q_row_stride, int64_t q_head_stride, int q_heads, int q_rope_dim,
                              const int64_t* pos, const float* inv_freq, int64_t* slots, int ring_slots, int window,
                              int32_t* wlist, int32_t* wcounts, int64_t* entry_out, int ratio, const void* kv_rows,
                              const int32_t* req_ids, const int32_t* ring_table, void* ring, int kv_lora, int rows,
                              cudaStream_t stream);

// Gather a request's pools [0, n_pools) into a contiguous buffer for the
// prefill logits GEMM.
void dsa_gather_index_pools(const int32_t* block_table, int pools_per_block,
                            const void* index_k, const float* index_scale,
                            int64_t n_pools, void* out_k, float* out_scale,
                            int dim, cudaStream_t stream);

// ---- top-k selection --------------------------------------------------
//
// The widest selection the kernels serve (select_k = index_topk / kpool):
// the expansion's eight rounds of a 256-thread block.
constexpr int kDsaSelectMaxK = 2048;
//
// Pinned spec: the select_k pools with the highest fp32 logits, exact ties
// broken to the lower pool index, output ascending in pool index. The
// composite sort key (~sortable_fp32 << 21) | pool_idx is a total order, so
// the selection is deterministic on any correct implementation and identical
// across the decode, prefill, and merge paths. Finite logits are assumed:
// real quantized cache rows are always finite (the saturating fp8 encoder
// never mints NaN), and NaN logits would sort as if near +infinity.

// Opts the select-decode and attention kernels into large dynamic shared
// memory (idempotent, cheap after the first call). Call it once per process
// before CUDA graph capture of the decode path — cudaFuncSetAttribute is a
// context mutation, not a stream operation, and must not happen inside
// capture. DsaLayer::prepare()/enqueue*() call this automatically; a bare
// kernel consumer that captures graphs should call it explicitly.
void dsa_prepare_kernel_smem();


// Fused decode select: streams pools [0, visible(pos[r])) per row straight
// from the blocked index cache, computes pool logits inline (warp per pool,
// lane per head), and keeps a running top-select_k composite-key selection.
// Fixed grid, grid-stride over pools; the last block finishes the
// selection, extracts ids ascending, expands pools to tokens, and appends
// each row's incomplete tail. The workspace (dsa_select_workspace_bytes,
// sized for the caller's largest row count and pool count) and counter_ws
// must be zeroed once at allocation; both self-reset after each call.
//   q_fp8: [rows, heads, dim] fp8; w_folded: [rows, heads] fp32;
//   block_tables: int32 [max_requests, blocks_per_request]; req_ids: [rows];
//   pos: [rows]; topk_out: int32 [rows, max_selected] (-1 padded);
//   out_counts: int32 [rows];
//   select_ws: dsa_select_workspace_bytes(max_rows, ws_max_pools) bytes,
//     256-byte aligned; ws_max_pools is the pool capacity it was sized for
//     (every row's visible count must fit);
//   counter_ws: int32 [2] (the scoring ticket and the rows-done count).
// rows <= 16 (the decode batch's cap), launched in groups of at most eight
// (the fused kernel's shared-memory bound; a group is the same work at any
// grouping); grid_blocks <= 0 picks the default (at least a group's rows
// either way: the last `rows` to finish scoring each select one row). `relu` clamps each head's fp8 dot at zero before
// its weight (the full GLM-5.3's indexer; GLM-5.3-Flash's sums the raw
// dots) — the same clamp in the prefill select and the host oracle.
size_t dsa_select_workspace_bytes(int max_rows, int64_t max_pools);
// The last call's phase stamps (globaltimer ns, the last block's view):
// [0] entry, [1] scoring done, [2] selection start, [3] selection total
// over rows, [4] expansion total over rows, [5] exit. A diagnostic for
// dsa_select_bench; synchronizes the device.
void dsa_select_debug_phases(uint64_t out[8]);
// The listed attention gather's anomaly record (2026-09-06): the number of
// gathers that found a token outside the request's table row or a block
// outside the pool (zero-filled instead of read), and the first one's
// (token, block, query row, split, list index, live count). `clear` resets
// the count. Stream-ordered on `stream` (synchronizes it).
unsigned long long dsa_attn_anomalies(long long out[6], bool clear,
                                      cudaStream_t stream);
// The decode select's anomaly record (2026-09-06): phase-2 fills whose scan
// disagreed with the histogram, the first one's (visible pools, lower,
// defs found, remaining, bin count, candidates found). `clear` resets.
unsigned long long dsa_select_anomalies(long long out[6], bool clear,
                                        cudaStream_t stream);
void dsa_select_decode(const void* q_fp8, const float* w_folded,
                       const int32_t* req_ids, const int64_t* pos, int rows,
                       const int32_t* block_tables, int blocks_per_request,
                       const void* index_k, const float* index_scale,
                       int pools_per_block, int heads, int dim, int select_k,
                       int kpool, int max_selected, int32_t* topk_out,
                       int32_t* out_counts, void* select_ws,
                       int64_t ws_max_pools, int32_t* counter_ws,
                       int grid_blocks, cudaStream_t stream,
                       bool relu = false);

// Prefill select over the materialized dot buffer. Optional workspace:
// dsa_select_prefill_workspace_bytes(rows, dot_stride, select_k) bytes,
// aligned for uint64_t, permits
// parallel coalesced scoring followed by exact radix selection at long
// contexts. Without workspace, use the streaming selection oracle.
//   dot: fp32 [rows * heads, dot_stride] (row r head h at
//   dot[r*heads + h]); k_scale: fp32 [n_pools] contiguous (gathered);
//   pos: [rows]; visible per row is derived on device.
size_t dsa_select_prefill_workspace_bytes(int rows, int64_t dot_stride, int select_k);
// Fused FP8 dot/head reduction. Q is [rows,32,128], K is [pools,128];
// keys is [rows,stride] with stride >= pools. The tensor-core reduction
// may differ by FP32 rounding from a small single-query cuBLAS plan;
// accuracy is checked against FP64, including selection boundaries.
void dsa_prefill_score_keys(const uint8_t* q, const uint8_t* k,
                            const float* w_folded, const float* k_scale,
                            const int64_t* pos, int rows, int64_t pools,
                            int64_t stride, int select_k, int kpool,
                            uint64_t* keys, cudaStream_t stream, bool relu);
// Select from reduced keys. partials holds 192 * select_k uint64_t values;
// it must not overlap keys, which has rows * stride values.
void dsa_select_prefill_keys(const uint64_t* keys, int64_t stride,
                             const int64_t* pos, int rows, int64_t pools,
                             int select_k, int kpool, int max_selected,
                             int32_t* topk_out, int32_t* out_counts,
                             uint64_t* partials, cudaStream_t stream);
void dsa_select_prefill(const float* dot, int64_t dot_stride,
                        const float* w_folded, const float* k_scale,
                        const int64_t* pos, int rows, int64_t n_pools,
                        int heads, int select_k, int kpool, int max_selected,
                        int32_t* topk_out, int32_t* out_counts,
                        cudaStream_t stream, bool relu = false,
                        void* workspace = nullptr);

// ---- MLA absorbed attention -------------------------------------------

// Absorbed query: q_tilde[r,h,c] = sum_d q[r,h,d] * W_uk[h,d][c], bf16
// output rounding (the production absorbed-MLA prep). kv_b is the
// checkpoint's interleaved layout [local_heads*(nope+v), kv_lora]: head h
// owns rows [h*(nope+v), h*(nope+v)+nope) for W_uk. With a rope tail the
// head's q row is [nope | rope] (the rope part already rotated) and the
// absorbed row is [W_uk^T q_nope | q_rot], kv_lora + rope wide.
//   q: bf16 [rows, local_heads * (nope + rope)];
//   q_tilde: bf16 [rows, local_heads * (kv_lora + rope)].
//   tensor_cores: false keeps the warp kernel whatever the row count — the
//   decode path's contract (a batched row bitwise the row alone; the
//   tensor-core form is tolerance-equal, not bitwise, to the warp chain).
void dsa_absorb_q(const void* q, const void* kv_b, void* q_tilde,
                  int64_t rows, int local_heads, int nope, int v, int kv_lora,
                  cudaStream_t stream, int rope = 0, bool tensor_cores = true);

// Split-KV sparse attention over gathered latent rows (flash-decoding
// shape: the decode hot path). One block per (row, split, head-group).
//   q_tilde: bf16 [rows, local_heads * (kv_lora + rope)];
//   topk: int32 [rows, topk_stride]; counts: int32 [rows];
//   m_ws/l_ws: fp32 [rows, n_split, local_heads];
//   c_ws: fp32 [rows, n_split, local_heads, kv_lora].
//   format/latent_scale: the cache's format and its row scales (fp8/fp4);
//   rope: the row's bf16 tail width (the score runs over kv_lora + rope,
//   the value accumulation over kv_lora).
void dsa_attn_partial(const void* q_tilde, const void* latent_cache,
                      const int32_t* req_ids, const int32_t* topk,
                      int topk_stride, const int32_t* counts, int rows,
                      int n_split, int local_heads, int kv_lora,
                      int block_tokens, const int32_t* block_tables,
                      int blocks_per_request, float scale,
                      float* m_ws, float* l_ws, float* c_ws,
                      cudaStream_t stream,
                      LatentFormat format = LatentFormat::kBf16,
                      const float* latent_scale = nullptr, int rope = 0);

// Dense causal attention on tensor cores: the prefill path
// below index_topk tokens of context, where the selection is provably
// dense (every visible pool selected, the tail appended) and a query at
// position p attends to tokens [0, p]. Same partial layout and split
// semantics as dsa_attn_partial — combine with dsa_attn_combine. The M
// dimension is (row, head) pairs; 32 per block. Returns false without
// launching when the geometry is outside the compiled set — kv_lora 512
// or 256 with no rope tail, kv_lora 512 with a 64-wide tail (the caller
// keeps the split kernel). Tolerance-equal to the split kernel (a
// different summation order), deterministic.
//   pos: int64 [rows] — the rows' token positions (causal bound per row).
bool dsa_attn_dense(const void* q_tilde, const void* latent_cache,
                    const int32_t* req_ids, const int64_t* pos, int rows,
                    int n_split, int local_heads, int kv_lora, int block_tokens,
                    const int32_t* block_tables, int blocks_per_request,
                    float scale, float* m_ws, float* l_ws, float* c_ws,
                    cudaStream_t stream,
                    LatentFormat format = LatentFormat::kBf16,
                    const float* latent_scale = nullptr, int rope = 0);

// The same kernel over each row's SELECTED tokens (2026-09-05, the sparse
// regime past index_topk tokens of context): topk/counts as
// dsa_attn_partial takes them, each 16-row slab (one query row) walking its
// own list. Requires local_heads a multiple of 16 (returns false otherwise,
// as for kv_lora outside 512/256); combine with dsa_attn_combine.
// `group` (2026-10-23): kAttnSplitGroup merges each kAttnSplitGroup
// consecutive splits of a row — the last of the group's blocks to arrive
// folds the per-split partials from L2 with the finish's formula into the
// group's partial (mg_ws / lg_ws / cg_ws, n_split / kAttnSplitGroup per
// row) and discards the dead per-split rows from L2 — so the finish reads
// eight times fewer partials and the per-split ones never write back. The
// grouping is fixed per request, so a row's output is the same whatever
// rows share the launch. n_split must be a multiple of the group;
// `tickets` holds ceil(rows * local_heads / 32) * n_split / group ints,
// zero at rest (each group's last block resets its own).
constexpr int kAttnSplitGroup = 8;
bool dsa_attn_listed(const void* q_tilde, const void* latent_cache,
                     const int32_t* req_ids, const int32_t* topk, int topk_stride,
                     const int32_t* counts, int rows, int n_split, int local_heads,
                     int kv_lora, int block_tokens, const int32_t* block_tables,
                     int blocks_per_request, float scale, float* m_ws, float* l_ws,
                     float* c_ws, cudaStream_t stream,
                     LatentFormat format = LatentFormat::kBf16,
                     const float* latent_scale = nullptr, int rope = 0,
                     int blocks_total = -1, int group = 1, int32_t* tickets = nullptr,
                     float* mg_ws = nullptr, float* lg_ws = nullptr, float* cg_ws = nullptr);
// blocks_total (2026-10-19): the pool's physical block count, the bound a
// table entry must lie under; -1 = blocks_per_request (a table whose row
// width is the pool: the paged main caches). The decode window's ring is
// a one-entry table per request over max_requests regions: pass that.

// Merge the split partials into normalized c rows: c_out[r,h,:] =
// (sum_s p_s * c_s) / (sum_s p_s * l_s), p_s = exp(m_s - max m).
void dsa_attn_combine(const float* m_ws, const float* l_ws, const float* c_ws,
                      int rows, int n_split, int local_heads, int kv_lora,
                      float* c_out, cudaStream_t stream);

// v-absorbed output: out[r, h*v+d] = sum_c W_uv[h,d][c] * c[r,h,c]. bf16
// out; c is fp32. kv_b interleaved: head h's W_uv rows are
// [h*(nope+v)+nope, (h+1)*(nope+v)).
//   c: fp32 [rows, local_heads * kv_lora];
//   out: bf16 [rows, local_heads * v].
void dsa_vout_gemm(const void* c, const void* kv_b, void* out,
                   int64_t rows, int local_heads, int nope, int v,
                   int kv_lora, cudaStream_t stream, bool tensor_cores = true);

}  // namespace dgpp
