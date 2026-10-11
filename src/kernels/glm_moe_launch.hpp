#pragma once
// Launchers for the MoE kernels (DESIGN §7.4; semantics in
// models/glm/moe.hpp). All deterministic (fixed reduction/selection order)
// and CUDA-graph capturable: no scratch, no host reads.
#include <cuda_runtime.h>

#include "models/glm/moe.hpp"

namespace dgpp {

// Router: hidden bf16 [tokens, hidden] -> ids int32 [tokens, top_k] in
// ASCENDING expert order (the accumulation order the reference's index_add
// produces), weights f32 [tokens, top_k] normalized and scaled. Per-(expert,
// token) dots, then per-token selection — two kernels, or one when
// `counters` (an int per token, zeroed once; the kernel leaves them zero)
// is given: the last dots block of a token runs the selection, bitwise the
// two-kernel form. `scores` and `biased` are f32 [tokens, n_experts] device
// buffers the caller owns: scores is kernel-internal scratch (the sigmoid
// scores the selection reads back), biased receives the full biased score
// row — the hand-off between the two phases AND the exported selection
// inputs (near-tie certification needs every expert's true biased score).
void launch_moe_router(const uint16_t* hidden, const uint16_t* gate,
                       const float* bias, int32_t* ids, float* weights,
                       float* scores, float* biased, const GlmMoeConfig& cfg,
                       int tokens, cudaStream_t stream,
                       int* counters = nullptr,
                       bool allow_tiled = true);

// Hash routing (DeepSeek-V4-Flash's first layers, 2026-10-01): after
// launch_moe_router has left every expert's score in `scores` (run it with
// a zero bias), the routes are REPLACED — ids [tokens, top_k] =
// tid2eid[tokens[t]] (int32 [vocab, top_k], rows ascending), weights the
// scores at those ids normalized and scaled as the select does. tokens:
// device int64 [n_tokens] in the hidden rows' order.
void launch_moe_hash_routes(const int64_t* tokens, const int32_t* tid2eid,
                            const float* scores, int32_t* ids, float* weights,
                            const GlmMoeConfig& cfg, int vocab, int n_tokens,
                            cudaStream_t stream);

// swiglu with asymmetric clamps: gate clamp_max only, up clamp both; two
// bf16 rounding points (silu result, then the product). n = rows*inter.
void launch_moe_swiglu_clamp(const uint16_t* gate, const uint16_t* up,
                             uint16_t* out, int64_t n, float limit,
                             cudaStream_t stream);

// dst[r, :] = src[rows[r], :] for n rows of width hidden.
void launch_moe_gather_rows(const uint16_t* src, const int32_t* rows,
                            uint16_t* dst, int n_rows, int hidden,
                            cudaStream_t stream);

// acc[rows[r], c] = fma(row_w[r], y[r, c], acc[rows[r], c]) — the fp32
// accumulation chain, one expert's segment at a time (the host loops
// experts in ascending order, the shared expert last with weight 1). y is
// the down projection's UNROUNDED fp32 output (launch_scale_gemm_f32).
void launch_moe_accum(float* acc, const float* y, const int32_t* rows,
                      const float* row_weights, int n_rows, int hidden,
                      cudaStream_t stream);

// ---- the prefill's grouped expert path ----------------------
// One launch per matrix per layer over EVERY non-empty expert segment: block
// (x, y) is the y-th segment against weight rows [x*kWarps, +kWarps) of its
// expert's [n, k] matrix (views[segment.expert * 3 + which]), its rows
// staged four at a time through the same fp8_gemv core the per-segment
// scale GEMM used — every output row bitwise that path's. `act` rows are
// bf16 with stride `act_stride`; `out` rows have stride `out_stride`.
struct MoeSegment {
  int32_t row0 = 0;    // first row of the segment in `act` / `out`
  int32_t rows = 0;    // rows in the segment
  int32_t expert = 0;  // view-table expert index (the shared expert last)
};
// `max_rows` is the longest segment's row count; `rows_per_block` (0 = no
// split) splits every segment across blocks along z in pieces of that many
// rows — for a launch whose segments are long and even (the shared
// expert's), never for the routed segments (short, uneven: the extra
// blocks only exit, and their launch cost showed).
void launch_moe_grouped_gemv_bf16(const uint16_t* act, size_t act_stride,
                                  const MoeSegment* segs, int n_segs,
                                  int max_rows, int rows_per_block,
                                  const MoeExpertView* views, int which,
                                  uint16_t* out, size_t out_stride, int n, int k,
                                  cudaStream_t stream);
void launch_moe_grouped_gemv_f32(const uint16_t* act, size_t act_stride,
                                 const MoeSegment* segs, int n_segs, int max_rows,
                                 int rows_per_block, const MoeExpertView* views,
                                 int which, float* out, size_t out_stride, int n,
                                 int k, cudaStream_t stream);
// Bench A/B: the fp8 grouped GEMM's ldmatrix kernel (default) or the
// reference tile kernel for every width.
void moe_set_fp8_ldm(bool on);

// The grouped tensor-core GEMM: the same contract and
// arguments, computed by bf16 mma.sync in 128-row m-tiles per block — every
// output element bitwise the tile kernel's (scale_gemm_kernel: the same
// dequantized weights and the same ascending-k16 accumulation), not the
// GEMV core's. The prefill path uses it (a segment up to 128 rows reads
// its expert's weights once instead of once per four rows); the decode
// path keeps the GEMV core. rows_per_block, when set, must be a multiple
// of 128 (the shared expert's z split); k a multiple of 16.
// `act_rows` (nullable, 2026-09-05): the activation row for segment row i is
// act_rows[i] — the gather folded into the tile load; the values are the
// gathered buffer's, so the outputs are unchanged.
void launch_moe_grouped_mma_bf16(const uint16_t* act, size_t act_stride,
                                 const MoeSegment* segs, int n_segs, int max_rows,
                                 int rows_per_block, const MoeExpertView* views,
                                 int which, uint16_t* out, size_t out_stride, int n,
                                 int k, cudaStream_t stream,
                                 const int32_t* act_rows = nullptr);
void launch_moe_grouped_mma_f32(const uint16_t* act, size_t act_stride,
                                const MoeSegment* segs, int n_segs, int max_rows,
                                int rows_per_block, const MoeExpertView* views,
                                int which, float* out, size_t out_stride, int n,
                                int k, cudaStream_t stream,
                                const int32_t* act_rows = nullptr);
// The dense form of the same kernel: out[m, n] = act[m, k] x
// W[n, k]^T (fp8 payload + block scales, out row stride n) — bitwise the
// scale GEMM's tile kernel; the scale GEMM routes m > 128 here.
void launch_dense_mma_bf16(const uint16_t* act, size_t act_stride,
                           const uint8_t* payload, const float* scales,
                           uint16_t* out, int m, int n, int k, cudaStream_t stream,
                           size_t out_stride = 0);
void launch_dense_mma_f32(const uint16_t* act, size_t act_stride,
                          const uint8_t* payload, const float* scales, float* out,
                          int m, int n, int k, cudaStream_t stream,
                          size_t out_stride = 0);
// Device-side segmentation (2026-09-04, the prefill's last host sync): from
// the router's ids [tokens * top_k] — the same segmentation the host path
// computes, on the device: rows[] = every routed (token, slot) in
// ascending-expert segment order (stable in (token, slot) within an
// expert), then the tokens once more for the shared expert; slot_row[t*K
// + j] = the gathered row of token t's j-th slot; segs[e] = {row0, rows
// (possibly 0), e} for every expert and segs[E] = the shared segment. One
// block; top_k <= 16; n_experts <= 1024.
void launch_moe_segment(const int32_t* ids, int tokens, int top_k,
                        int n_experts, int32_t* rows, int32_t* slot_row,
                        MoeSegment* segs, cudaStream_t stream);

// The compact tile list (2026-09-29, the hybrid's 8K prefill profile): a
// grouped launch whose grid is n_segs x ceil(max_rows / tile_rows) sizes
// every segment by the longest one, and one hot expert (a routed expert in
// nearly every token's top-k: 3,000-4,096 rows of a 4,096-token chunk in
// most layers) makes that 62 m-tiles x 512 experts x n-tiles of blocks
// of which ~97 % read their segment and exit — gate/up 4.0-4.4 ms and down
// 5.4-7.0 ms a launch against 2.0-2.5 ms for the same rows spread evenly.
// This list is the real tiles only: entry i = (segment, first row within
// it) in segment order, tile j of segment e at row j * tile_rows, and
// *count = the number of entries. A kernel then runs a 1-D grid of
// n_tiles x capacity blocks and block b takes tile b / n_tiles (exiting
// past *count). moe_tile_list_capacity bounds the count from the host
// without a sync: every non-empty segment adds at most one partial tile.
struct MoeTile {
  int32_t seg = 0;  // index into the segment table
  int32_t m0 = 0;   // first row of the tile within the segment
};
inline int moe_tile_list_capacity(int n_segs, int total_rows, int tile_rows) {
  return n_segs + (total_rows + tile_rows - 1) / tile_rows;
}
// One block; n_segs <= 65535; tiles has moe_tile_list_capacity entries.
void launch_moe_tile_list(const MoeSegment* segs, int n_segs, int tile_rows,
                          MoeTile* tiles, int32_t* count, cudaStream_t stream);

// The ordered accumulation in one pass: for every token, its top_k routed
// slots in ASCENDING expert id (sorted here, whatever order the router left)
// then the shared expert's row — moe_accum_kernel's __fmaf_rn chain from
// zero, op for op (the shared row's weight is 1) — rounded once to bf16.
// slot_row[t*K + j] is the gathered row of token t's j-th slot; the shared
// rows sit at shared_row0 + t. top_k <= 16.
void launch_moe_accum_ordered(uint16_t* out, const float* down,
                              size_t down_stride, const int32_t* slot_row,
                              const int32_t* slot_ids, const float* slot_w,
                              int shared_row0, int tokens, int top_k,
                              int hidden, cudaStream_t stream);
// The same chain left UNROUNDED in fp32 (out [tokens, hidden] f32) for a
// caller that continues it; shared_row0 < 0 ends the chain after the
// routed slots (no shared row) on both launchers.
void launch_moe_accum_ordered_f32(float* out, const float* down,
                                  size_t down_stride, const int32_t* slot_row,
                                  const int32_t* slot_ids, const float* slot_w,
                                  int shared_row0, int tokens, int top_k,
                                  int hidden, cudaStream_t stream);

// The same chain over bf16 down rows (the W4A4 grouped chain; no shared row),
// rounded to bf16 or left in fp32 for a caller that continues the sum.
void launch_moe_accum_ordered_bf16down(uint16_t* out, const uint16_t* down, size_t down_stride,
                                       const int32_t* slot_row, const int32_t* slot_ids,
                                       const float* slot_w, int tokens, int top_k, int hidden,
                                       cudaStream_t stream);
void launch_moe_accum_ordered_f32_bf16down(float* out, const uint16_t* down, size_t down_stride,
                                           const int32_t* slot_row, const int32_t* slot_ids,
                                           const float* slot_w, int tokens, int top_k, int hidden,
                                           cudaStream_t stream);

// out[i] = bf16(acc[i]) — the chain's single rounding, as the sum leaves for
// the FFN all-reduce (bf16 on the wire).
void launch_moe_round_bf16(uint16_t* out, const float* acc, int64_t n,
                           cudaStream_t stream);

// ---- decode-slot path (the sync-free MoE, 2026-09-01) --------------------
//
// The decode step's MoE without host round-trips: the router leaves
// ids/weights on the device (ids ASCENDING per row — the router kernel's
// contract), the slot kernels read the route from device memory, and the
// accumulation reproduces the host path's exact op order. Slot layout:
// tokens*(top_k+1) slots, slot s = t*(K+1)+j; j<K is row t's routed expert
// j (ascending expert id), j==K is the shared expert (all rows, weight 1,
// accumulated last). Every expert is local: each rank holds a slice of all
// of them (gate/up rows, down columns of the intermediate dim), so `views`
// is the device expert table [n_experts, 3] (gate,up,down). Routed dims
// (n_routed, k_routed) vs the shared expert's (n_shared, k_shared); shared
// matrices arrive as args (host-known constants, not table entries).
// Contract: both k a multiple of 16, payloads 16B-aligned (the loader's).
// The arithmetic is exactly launch_scale_gemm_*'s at m<=4 (same core) —
// glm_moe_test's bitwise gate pins the equivalence.
//
// slot_gate_up_swiglu: the gate and up GEMVs and the swiglu in one launch —
// act[slot, row] = swiglu(gate_dot, up_dot) with the bf16 rounding points of
// the three-launch chain (the dots round to bf16 exactly where the
// intermediate buffers rounded them), so the result is bit-identical; the
// intermediates never touch memory. Gate and up share n and k.
//
// `order` (nullable): the slots' EXECUTION order, blockIdx.y -> logical
// slot, from launch_moe_slot_order — a multi-token batch sorted by expert
// so a shared expert's second read is an L2 hit. Null = identity. Results
// are indexed by logical slot either way (bitwise identical).
// `shared_early`: the shared expert's slots interleaved with the first
// routed ones instead of last (the kernel's comment: compute over
// L2-resident weights beside the routed DRAM stream).
void launch_moe_slot_order(const int32_t* ids, int32_t* order, int slots,
                           int top_k, int n_experts, cudaStream_t stream,
                           bool shared_early = false);
void launch_moe_slot_gate_up_swiglu(
    const uint16_t* x, size_t x_stride, const int32_t* ids,
    const int32_t* order, const MoeExpertView* views, int n_routed,
    int k_routed, int n_shared, int k_shared, const uint8_t* sh_gate_payload,
    const float* sh_gate_scales, const uint8_t* sh_up_payload,
    const float* sh_up_scales, uint16_t* act, int act_stride, int slots,
    int top_k, float limit, cudaStream_t stream, int sh_rs = 7, int sh_cs = 7);

// slot_down: out[slot, :] = fp32 dot(down rows, act[slot]) per slot,
// UNROUNDED (the accumulation owns the single rounding). Routed slots read
// act row 0..k_routed, the shared slot 0..k_shared.
void launch_moe_slot_down(const uint16_t* act, size_t act_stride,
                          const int32_t* ids, const int32_t* order,
                          const MoeExpertView* views,
                          int n_routed, int k_routed, int n_shared,
                          int k_shared, const uint8_t* sh_payload,
                          const float* sh_scales, float* out, int out_stride,
                          int slots, int top_k, cudaStream_t stream, int sh_rs = 7,
                          int sh_cs = 7);
// sh_rs / sh_cs: the fp8 shared expert's scale grid as log2 block sizes
// (7 = 128 x 128; 5 = the DeepSeek-V4.1 release's 32 x 32, 2026-09-13).

// slot_accum: per (token, element), the ordered fp32 chain
//   out = bf16( fma(1, y_shared, fma(w_{K-1}, y_{K-1}, ... fma(w_0, y_0, 0))) )
// — exactly the host path's ascending-expert accumulation (launch_moe_accum
// per segment, then launch_moe_round_bf16).
void launch_moe_slot_accum(uint16_t* out, const float* contrib,
                           const float* weights, int tokens, int hidden,
                           int top_k, cudaStream_t stream);
// The routed slots' chain alone, UNROUNDED fp32 (out [tokens, hidden]) —
// the slot layout keeps its (unused) shared slot; the Qwen decode
// continues the chain with its BF16 shared expert. The slot kernels above
// take n_shared == 0 for that layout: their shared-slot blocks return.
void launch_moe_slot_accum_routed_f32(float* out, const float* contrib,
                                      const float* weights, int tokens,
                                      int hidden, int top_k,
                                      cudaStream_t stream);

// ---- the NVFP4 routed experts (2026-09-08, docs/nvfp4_plan.md §3) ---------
// The same contracts as the FP8 launchers above, over expert-view tables
// whose routed entries are NVFP4 (payload = e2m1 pairs [n, k/2], fp4_scales
// = e4m3 [n, k/16], fp4_global = the matrix's F32 global scale). The shared
// expert stays FP8 and, in the slot kernels, runs the fp8 core inside the
// same launch (a block's slot decides, uniformly). Routed k must satisfy
// fp4_gemv::shape_ok (a multiple of 32, k | 1024 or 1024 | k). Every
// routed row's arithmetic is the fp4 core's, bitwise across the grouped
// (host) and slot (decode) launchers.
// The fp4 twin of the grouped tensor-core kernel (docs/nvfp4_plan.md §3.4):
// the same tiles and mma.sync chain, the weight tile decoded from the NVFP4
// triple exactly, the global scale dividing the finished dot in the
// epilogue. k must be a multiple of 16 (any width — the fp4 GEMV core's
// power-of-two set does not apply). The grouped launcher runs the ldmatrix
// kernel (2026-09-08: 64 x 128 x 64 tiles, a three-slot cp.async ring of the
// activation tile and the RAW fp4 codes, B fragments decoded at fragment
// time, one 64-row m-tile per block with the m-tile the fastest grid index,
// the weight rows' next L2 line prefetched ahead); the dense launcher runs
// the synchronous 128 x 64 x 64 tile reference; the two are bitwise
// (glm_moe_test pins it per segment) — neither is bitwise the fp4 GEMV core
// (summation order). rows_per_block is accepted for the family's signature
// and ignored by the grouped launcher (one m-tile per block always).
// fp4_group 32 (2026-09-14): the MXFP4 table (e8m0 scales per 32 codes, no
// global) through the same kernel, the pair decoded in fp32 (exact); k a
// multiple of 32 then.
void launch_moe_grouped_mma_fp4_bf16(const uint16_t* act, size_t act_stride,
                                     const MoeSegment* segs, int n_segs,
                                     int max_rows, int rows_per_block,
                                     const MoeExpertView* views, int which,
                                     uint16_t* out, size_t out_stride, int n,
                                     int k, cudaStream_t stream,
                                     const int32_t* act_rows = nullptr,
                                     int fp4_group = kFp4Group);
void launch_moe_grouped_mma_fp4_f32(const uint16_t* act, size_t act_stride,
                                    const MoeSegment* segs, int n_segs,
                                    int max_rows, int rows_per_block,
                                    const MoeExpertView* views, int which,
                                    float* out, size_t out_stride, int n, int k,
                                    cudaStream_t stream,
                                    const int32_t* act_rows = nullptr,
                                    int fp4_group = kFp4Group);
// The reference kernel's grouped form (the microbench's A/B and the gate).
void launch_moe_grouped_mma_fp4_ref_bf16(const uint16_t* act, size_t act_stride,
                                         const MoeSegment* segs, int n_segs,
                                         int max_rows, int rows_per_block,
                                         const MoeExpertView* views, int which,
                                         uint16_t* out, size_t out_stride, int n,
                                         int k, cudaStream_t stream,
                                         const int32_t* act_rows = nullptr,
                                         int variant = 0);
void launch_moe_grouped_mma_fp4_ref_f32(const uint16_t* act, size_t act_stride,
                                        const MoeSegment* segs, int n_segs,
                                        int max_rows, int rows_per_block,
                                        const MoeExpertView* views, int which,
                                        float* out, size_t out_stride, int n, int k,
                                        cudaStream_t stream,
                                        const int32_t* act_rows = nullptr,
                                        int variant = 0);
void launch_dense_mma_fp4_bf16(const uint16_t* act, size_t act_stride,
                               const GlmFp4Matrix& w, uint16_t* out, int m, int n,
                               int k, cudaStream_t stream);
void launch_dense_mma_fp4_f32(const uint16_t* act, size_t act_stride,
                              const GlmFp4Matrix& w, float* out, int m, int n,
                              int k, cudaStream_t stream);
// The dense matrix through the PRODUCTION ldmatrix kernel (the grouped
// launcher with one segment): bitwise the dense reference above
// (glm_moe_test pins the pair per segment) at the production kernel's
// rate (moe_tile_bench: ~1.5-2x on prefill shapes). `seg` is device
// memory holding {0, m, 0}; `views` device memory holding at least
// [which] as MoeExpertView::of(w) (a 3-entry [gate,up,down] table with
// which = 0/1/2 mirrors the MoE layout). Both stay caller-owned: the
// kernel only reads them.
void launch_dense_mma_fp4_prod_bf16(const uint16_t* act, size_t act_stride,
                                    const MoeSegment* seg, const MoeExpertView* views,
                                    int which, uint16_t* out, int m, int n, int k,
                                    cudaStream_t stream, int fp4_group = kFp4Group);

// fp4_group (2026-09-13): 16 = NVFP4 view tables (e4m3 scales + globals),
// 32 = MXFP4 tables (e8m0 scales, no globals — DeepSeek-V4.1-Flash,
// docs/deepseek_v41_flash_plan.md D2); the routed k must be in the group's
// compiled set (fp4_gemv::k_compiled_for).
void launch_moe_grouped_gemv_fp4_bf16(const uint16_t* act, size_t act_stride,
                                      const MoeSegment* segs, int n_segs,
                                      int max_rows, int rows_per_block,
                                      const MoeExpertView* views, int which,
                                      uint16_t* out, size_t out_stride, int n,
                                      int k, cudaStream_t stream, int fp4_group = 16);
void launch_moe_grouped_gemv_fp4_f32(const uint16_t* act, size_t act_stride,
                                     const MoeSegment* segs, int n_segs,
                                     int max_rows, int rows_per_block,
                                     const MoeExpertView* views, int which,
                                     float* out, size_t out_stride, int n,
                                     int k, cudaStream_t stream, int fp4_group = 16);
// `shared_view_base` (2026-09-09, GLM-4.7): -1 = the FP8 shared expert
// from the sh_* arguments; >= 0 = the NVFP4 shared expert at view-table
// entries [shared_view_base, +3) (n_experts * 3: the loader's (E+1)-entry
// table), read through the fp4 core at the routed k (k_shared must equal
// k_routed); the sh_* arguments are then ignored.
void launch_moe_slot_gate_up_swiglu_fp4(
    const uint16_t* x, size_t x_stride, const int32_t* ids,
    const int32_t* order, const MoeExpertView* views, int n_routed,
    int k_routed, int n_shared, int k_shared, const uint8_t* sh_gate_payload,
    const float* sh_gate_scales, const uint8_t* sh_up_payload,
    const float* sh_up_scales, uint16_t* act, int act_stride, int slots,
    int top_k, float limit, cudaStream_t stream, int shared_view_base = -1,
    int fp4_group = 16, int sh_rs = 7, int sh_cs = 7);
void launch_moe_slot_down_fp4(const uint16_t* act, size_t act_stride,
                              const int32_t* ids, const int32_t* order,
                              const MoeExpertView* views, int n_routed,
                              int k_routed, int n_shared, int k_shared,
                              const uint8_t* sh_payload, const float* sh_scales,
                              float* out, int out_stride, int slots, int top_k,
                              cudaStream_t stream, int shared_view_base = -1,
                              int fp4_group = 16, int sh_rs = 7, int sh_cs = 7);

// ---- the packed-int routed experts (2026-09-12, docs/glm53_plan.md D2) ----
// The same contracts as the FP8 / NVFP4 launchers above over expert-view
// tables whose entries are packed-int (payload = the I32 words,
// packed_scales = bf16 per 64, bits = 4 or 8): the grouped kernel takes
// the launch's width (the routed segments' or the shared segment's); the
// slot kernels read the routed entries at `routed_bits` and the shared
// expert (view-table entries [shared_view_base, +3), n_experts * 3) at
// `shared_bits`, which must be 8, at the routed k. n_shared == 0: no shared
// slot. k must satisfy packq_gemv's contract (a multiple of 64 in the
// compiled set). Every row's arithmetic is the packed core's, bitwise
// across the grouped and slot launchers and the single-matrix launcher.
// scale_fmt / routed_scale_fmt: the routed entries' packed scale format
// (kPackedScale*, quant_matrix.hpp; 0 = bf16 per 64, 1 = f16 per 128, 2 =
// the NF4I8 codebook, bf16 per 128); the shared expert's entries are
// always format 0.
// The NF4I8 checkpoint's H32 input rotation (kMoeInputHadamard32,
// models/glm/moe.hpp) is the layer's: the grouped launchers read rows the
// layer rotated (kernels/hadamard32.cu); the slot gate/up launcher takes
// the rotated token rows as `x_routed` (nullptr: the rows of `x`), which
// the routed slots stage while the shared slot stages `x`; the slot down
// launcher reads activation rows the layer rotated in place (the shared
// slots' rows skipped).
void launch_moe_grouped_gemv_packq_bf16(const uint16_t* act, size_t act_stride,
                                        const MoeSegment* segs, int n_segs,
                                        int max_rows, int rows_per_block,
                                        const MoeExpertView* views, int which,
                                        uint16_t* out, size_t out_stride, int n,
                                        int k, int bits, cudaStream_t stream,
                                        int scale_fmt = 0);
void launch_moe_grouped_gemv_packq_f32(const uint16_t* act, size_t act_stride,
                                       const MoeSegment* segs, int n_segs,
                                       int max_rows, int rows_per_block,
                                       const MoeExpertView* views, int which,
                                       float* out, size_t out_stride, int n,
                                       int k, int bits, cudaStream_t stream,
                                       int scale_fmt = 0);
void launch_moe_slot_gate_up_swiglu_packq(
    const uint16_t* x, size_t x_stride, const int32_t* ids, const int32_t* order,
    const MoeExpertView* views, int n_routed, int k_routed, int routed_bits,
    int n_shared, int shared_bits, uint16_t* act, int act_stride, int slots,
    int top_k, float limit, cudaStream_t stream, int shared_view_base,
    int routed_scale_fmt = 0, const uint16_t* x_routed = nullptr);
void launch_moe_slot_down_packq(const uint16_t* act, size_t act_stride,
                                const int32_t* ids, const int32_t* order,
                                const MoeExpertView* views, int n_routed, int k_routed,
                                int routed_bits, int n_shared, int shared_bits,
                                float* out, int out_stride, int slots, int top_k,
                                cudaStream_t stream, int shared_view_base,
                                int routed_scale_fmt = 0);

// ---- the Mixed346 routed experts (2026-10-09, kMoeInputHadamard32Int8) ----
// The same contracts over expert-view tables that mix forms expert by
// expert (HawkBearPig/GLM-5.3-Mixed346-GPTQ-H32-A8-g128): a converted
// triple (packed_scale_fmt kPackedScaleBf16G128Mixed346, bits 3 / 4 / 6 —
// gate and up at one width) multiplies the rows' INT8 codes (`x_codes` /
// `act_codes`, 16-byte aligned rows of k bytes) and fp32 scales
// (`x_scales` / `act_scales`, k / 128 a row) — the layer's quantized
// copies (kernels/hadamard32.hpp) of the rows a baseline (format 0, int4
// g64) expert and the shared expert (int8 g64, view-table entries
// [shared_view_base, +3)) read plain from `x` / `act`. k must be a
// multiple of 128 in both cores' compiled sets (128, 256, 512, 1024,
// 2048, 6144). The grouped launchers take the ROUTED segments only; the
// shared segment keeps the packed int8 grouped launcher.
void launch_moe_slot_gate_up_swiglu_m346(const uint16_t* x, size_t x_stride, const int8_t* x_codes,
                                         size_t x_code_stride, const float* x_scales, size_t x_scale_stride,
                                         const int32_t* ids, const int32_t* order, const MoeExpertView* views,
                                         int n_routed, int k_routed, int n_shared, uint16_t* act, int act_stride,
                                         int slots, int top_k, float limit, cudaStream_t stream,
                                         int shared_view_base);
void launch_moe_slot_down_m346(const uint16_t* act, size_t act_stride, const int8_t* act_codes,
                               size_t act_code_stride, const float* act_scales, size_t act_scale_stride,
                               const int32_t* ids, const int32_t* order, const MoeExpertView* views, int n_routed,
                               int k_routed, int n_shared, float* out, int out_stride, int slots, int top_k,
                               cudaStream_t stream, int shared_view_base);
void launch_moe_grouped_gemv_m346_bf16(const uint16_t* act, size_t act_stride, const int8_t* act_codes,
                                       size_t act_code_stride, const float* act_scales, size_t act_scale_stride,
                                       const MoeSegment* segs, int n_segs, int max_rows, int rows_per_block,
                                       const MoeExpertView* views, int which, uint16_t* out, size_t out_stride,
                                       int n, int k, cudaStream_t stream);
void launch_moe_grouped_gemv_m346_f32(const uint16_t* act, size_t act_stride, const int8_t* act_codes,
                                      size_t act_code_stride, const float* act_scales, size_t act_scale_stride,
                                      const MoeSegment* segs, int n_segs, int max_rows, int rows_per_block,
                                      const MoeExpertView* views, int which, float* out, size_t out_stride, int n,
                                      int k, cudaStream_t stream);

}  // namespace dgpp
