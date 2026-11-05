#pragma once
// Dory1.0-4B attention kernels (docs/dory_plan.md §1.3, D6, D7): the finish
// (RoPE, the per-head `justnorm`, the `sqk` square) and the paged GQA
// attention of the 36 attention blocks — 16 query heads over 4 kv heads,
// head_dim 256, causal, the sliding layers' window applied at read time.
// The q/k/v/g projections run through the fp4 GEMV/GEMM launchers (bf16
// outputs); these kernels are the rest. The shape is the glm4/MiMo split-KV one
// (kernels/glm4_attn.cu), forked for the 256-wide heads and the justnorm
// finish.
//
//   finish   the q/k/v bf16 projection rows — a Linear's one rounding —
//            then the full
//            width neoX RoPE (pairs (i, i + 128), θ per layer type: 1e4 on
//            the sliding layers, 1e7 on the full ones) with the engine's
//            bf16 ops (cos/sin bf16 from fp32 pos × inv_freq; x·cos and
//            rotate(x)·sin rounded, the sum rounded), then, per head,
//                k = bf16(justnorm(k)),  q = bf16( fp32(justnorm(q)) · m )
//            with m = (fp32(sqk[i]) · sqk_factor)² the fp32 square the
//            reference applies AFTER the RoPE and before the attention
//            (§1.3's order — a reordering is a silent accuracy change).
//            `q`'s two roundings are the primitives': `justnorm`'s own and
//            the bf16 the attention takes. The k and v heads are appended
//            to the paged caches at each row's slot;
//   attention split-KV paged GQA: one block per (row, split, kv head), the
//            kv head's query heads sharing every K/V tile, fp32 scores and
//            online softmax, probabilities rounded to bf16 for the V
//            accumulation, the denominator unrounded (the engine's QSA
//            rule, the QSA / DSA / GLM / MiMo pin — tolerance-equal to a
//            dense fp32 softmax, not bitwise). A row attends
//            [max(0, pos - window + 1), pos] of its request (window <= 0:
//            [0, pos]); its own row is already appended. Rows with pos < 0
//            are padding;
//   combine  the splits merged, out = bf16(C / L).
//
// Every kernel is deterministic and capturable (fixed grids, no host
// reads). Cache layout: bf16 [slots, kv_heads * 256] per layer, slot =
// block_tables[req][pos / block_tokens] * block_tokens + pos % block_tokens.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

constexpr int kDoryHeadDim = 256;
// Tiles of the attention kernel (the split geometry the host derives).
constexpr int kDoryAttnTile = 32;

// Rows [0, rows) of the projections' bf16 outputs (a Linear's output is
// bf16 — the reference's own dtype, and the fp4 paths' only prefill form):
// q [rows, local_heads * 256] (row stride q_stride elements), k / v [rows,
// kv_heads * 256] (row strides k_stride / v_stride). sqk bf16
// [local_heads * 256] — the release's per-query-element `sqk`, scaled by
// `sqk_factor` in fp32. inv_freq fp32 [128]: this layer's θ, the sliding and
// full layers differ. pos / req_ids: device [rows]; a row with pos < 0
// writes nothing.
void dory_qkv_finish(const uint16_t* q_dot, int64_t q_stride, const uint16_t* k_dot,
                     int64_t k_stride, const uint16_t* v_dot, int64_t v_stride,
                     const uint16_t* sqk, float sqk_factor,
                     const float* inv_freq, float eps, const int32_t* req_ids,
                     const int64_t* pos, int rows, int local_heads, int kv_heads,
                     const int32_t* block_tables, int blocks_per_request, int block_tokens,
                     uint16_t* q_out, int64_t q_out_stride, uint16_t* k_cache,
                     uint16_t* v_cache, cudaStream_t stream);

// The partials of row r, split s, head h: m_ws / l_ws fp32 [rows, n_split,
// local_heads], c_ws fp32 [rows, n_split, local_heads, 256]. q: bf16 rows
// of local_heads x 256 (row stride q_stride, heads contiguous). Every query
// head h reads kv head h / (local_heads / kv_heads). A row's visible tokens
// are [max(0, pos - window + 1), pos] (window <= 0: from 0); split s of a
// row covers tiles [s * chunk, (s + 1) * chunk) of its 32-token tiles
// counted from the first visible token (chunk = ceil(tiles / n_split)); an
// empty split writes m = -inf, l = 0.
void dory_attn_partial(const uint16_t* q, int64_t q_stride, const uint16_t* k_cache,
                       const uint16_t* v_cache, const int32_t* req_ids, const int64_t* pos,
                       int rows, int n_split, int local_heads, int kv_heads, int block_tokens,
                       const int32_t* block_tables, int blocks_per_request, int window,
                       float scale, float* m_ws, float* l_ws, float* c_ws, cudaStream_t stream);

// out[r, h * 256 + d] = bf16(sum_s e^(m_s - M) c_s / sum_s e^(m_s - M) l_s)
// (0 when every split is empty); out rows local_heads * 256 wide.
void dory_attn_combine(const float* m_ws, const float* l_ws, const float* c_ws, int rows,
                       int n_split, int local_heads, uint16_t* out, cudaStream_t stream);

}  // namespace dgpp
