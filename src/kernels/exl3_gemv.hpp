#pragma once
// The EXL3 streaming decode GEMM (2026-10-22, docs/deepseek_v41_flash_exl3_plan.md
// §3.2): out[m, n] = rot_out(rot_in(act)[m, k] x w_inner[k, n]) for an EXL3
// matrix (models/exl3_matrix.hpp) and bf16 activation rows, the plan's
// §2.2 chain:
//   xh = fp16( (x o suh) H_128 / sqrt(128) )       per 128-k block, in fp32
//   y_inner = xh . w_inner                           mma.sync fp16 x fp16, fp32 accumulate, ascending k
//   out = ( y_inner H_128 / sqrt(128) ) o svh        per 128-n block, in fp32
// A block of eight warps owns one 128-output block (eight 16-wide n tiles,
// one per warp) and walks the k range in 128-k windows: the window's
// activation rows are rotated once into shared memory (fp16) and read by
// every warp; each warp streams its n tile's eight k tiles of the window
// through a cp.async ring, decodes its lane's eight trellis states per tile
// into the two B fragments and runs the mma per 16-row A tile, so m = 64
// costs the same weight stream as m = 1. The epilogue rotates the block's
// 128 outputs per row and scales them by svh. Narrow matrices split the k
// range across grid.y (fp32 pre-rotation partials in the caller's
// workspace, one reduce launch that sums in split order and rotates); the
// split count is a function of (n, k) alone, so a row's chain — its mma
// accumulation over k in window order, then the split sum — is the same
// whatever rows ride in the launch, 1..128 (the row-count invariance the
// batch family relies on). Above 16 rows the A tile grows to 32, 64 and
// 128 rows (kMT 2, 4, 8 — the 128-row tile since 2026-10-23: a dense
// prefill site read its weights once per 64 rows, 0.74 s of an 8K
// prefill), and above 128 rows the launch runs 128-row groups, each with
// that same chain.
//
// The rotated rows are produced once by a small kernel ahead of the GEMV
// (xh [m, k] fp16 in the workspace: the GEMV's output blocks then stage A
// by cp.async like B, with no compute on the staging path), and in the
// split form the last split block of an output block to arrive folds the
// splits in split order and finishes the block (arrival counters at the
// workspace's head, zero between launches) — a launch is two kernels
// (2026-10-23: the separate reduce launch cost 0.7 ms a decode pass on the
// 1017 pack; rotating the rows inside the GEMV's stage instead of ahead
// was measured too and cost 30-60 % of every kernel's time, the rotating
// warp also being a computing one — the plan's §7).
//
// CONTRACT: n == w.rows, k == w.cols (the whole view), both multiples of
// 128; act rows 16-byte aligned with an even-16-byte stride; the tiles
// 16-byte aligned (cp.async); out_stride >= n (0: n); a 16-byte aligned
// workspace of at least exl3_gemm_ws_bytes(n, k, 1) — with room for the
// splits' partials the narrow sites split (exl3_gemm_splits) — that the
// owner ZEROES ONCE at allocation and shares with no other writer (its
// tail holds the split form's arrival counters, which must start at zero;
// every launch leaves them zero; cuBLASLt's workspace is not a home for
// it) and uses from one stream at a time. Bit widths
// 3, 4, 6 and 8 are compiled (the pack's). bf16 out rounds the fp32 result
// once; f32 out stores it unrounded (bf16(out_f32) == out_bf16 bit for bit).
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "kernels/glm_moe_launch.hpp"
#include "models/exl3_matrix.hpp"

namespace dgpp {

// The decode batch bound the split decision is sized for (the workspace
// check uses it, so the split never depends on the rows in the launch).
constexpr int kExl3GemvSplitRows = 128;

void launch_exl3_gemm_bf16(const uint16_t* act, size_t act_stride, const Exl3Matrix& w, uint16_t* out, int m, int n,
                           int k, size_t out_stride, cudaStream_t stream, void* ws = nullptr, size_t ws_bytes = 0);
void launch_exl3_gemm_f32(const uint16_t* act, size_t act_stride, const Exl3Matrix& w, float* out, int m, int n, int k,
                          size_t out_stride, cudaStream_t stream, void* ws = nullptr, size_t ws_bytes = 0);

// The shape the kernels take (the contract above); false names nothing.
bool exl3_gemm_shape_ok(const Exl3Matrix& w, const void* act, size_t act_stride, int m, int n, int k);
// The split count a launch at (n, k) takes with this workspace (1: unsplit).
int exl3_gemm_splits(int n, int k, const void* ws, size_t ws_bytes);
// The workspace a launch needs: the rotated rows (kExl3GemvSplitRows x k
// fp16) and, when split, the partials (splits x kExl3GemvSplitRows x n
// fp32). A site sizes its workspace at exl3_gemm_ws_bytes(n, k, 16) to
// let the split rule choose freely.
size_t exl3_gemm_ws_bytes(int n, int k, int splits);
// The workspace the split rule would take at (n, k) with room to choose
// freely (the splits the shape wants, not more): what a site allocates.
size_t exl3_gemm_ws_bytes_for(int n, int k);

// ---- the MoE forms (docs/deepseek_v41_flash_exl3_plan.md §3.3-3.4) --------
// Expert-view tables whose entries are EXL3 (MoeExpertView::of(const
// Exl3Matrix&): payload = the n-tile-major tiles, exl3_suh / exl3_svh,
// exl3_bits, exl3_rows / exl3_cols); the shared expert is view-table
// entries [shared_view_base, +3) (n_experts * 3). The slot layout is the
// decode chain's: slot t * (top_k + 1) + j, j == top_k the shared slot;
// `order` (nullable) the expert-sorted block order.
//
// The rotated inputs are produced once per slot (per segment row) ahead of
// the kernels, one row per matrix: which == 0 rotates the token rows by
// the slot's gate suh into xh0 and up suh into xh1 (k_max = hidden); which
// == 2 rotates the slot's own SwiGLU row by its down suh into xh0 (k_max =
// the widest down K; a narrower matrix's slot stops at its own width). The
// rows are laid out by POSITION of `order` (null: the slot index) — the
// act rows read for which == 2 too — so an equal-expert run's rows are
// contiguous. xh rows at xh_stride (16-byte multiple). n_shared: the
// shared expert's width in this launch (0: the shared slots hold nothing —
// their blocks return before reading a view).
void launch_exl3_moe_rotate_slots(const uint16_t* x, size_t x_stride, const int32_t* ids, const int32_t* order,
                                  const MoeExpertView* views, int slots, int top_k, int k_max, int shared_view_base,
                                  int n_shared, int which, uint16_t* xh0, uint16_t* xh1, size_t xh_stride,
                                  cudaStream_t stream);
// Gate + up + SwiGLU into act [slots, act_stride] bf16, rows by position:
// the gate and up dots through the core at the rotated rows (k = hidden),
// rotated and scaled, rounded to bf16, then the SwiGLU as the fp8 / fp4
// slot kernels round it. THE EXPERT-RUN FORM (2026-10-23): `runs`
// (nullable; from launch_moe_slot_order with `order`, required beside it)
// gives at each position the length of the equal-expert run starting
// there (0 inside a run), and the run's leader computes all its rows as
// one A tile — up to 16 rows on the deep-ring tile, up to 64 (the decode
// batch's bound: at most 64 token rows) on the 64-row tile, a second
// launch that exists only above 16 token rows — so an expert several rows
// share is streamed and trellis-decoded once. Per row the chain is the
// one-row form's: bitwise the per-slot form (null runs) at any
// duplication. n_max: the widest gate width among the views (routed or
// shared); a run's blocks past its own width return. routed_bits /
// shared_bits: the compiled (routed, shared) pairs 3/6, 4/6, 6/6, 3/3,
// 4/4, 8/8, 4/8.
void launch_exl3_moe_slot_gate_up_swiglu(const uint16_t* xh_g, const uint16_t* xh_u, size_t xh_stride,
                                         const int32_t* ids, const int32_t* order, const int32_t* runs,
                                         const MoeExpertView* views, int n_max, int k, int routed_bits,
                                         int shared_bits, uint16_t* act, int act_stride, int slots, int top_k,
                                         float limit, int shared_view_base, int n_shared, cudaStream_t stream);
// The down projection into out [slots, out_stride] fp32 (unrounded), the
// rows at their LOGICAL slots (the accumulation's layout): the rotated
// SwiGLU rows (by position) against the run's expert's w2 over that
// matrix's own K; n = hidden. The same run form and `runs` contract.
void launch_exl3_moe_slot_down(const uint16_t* xh_d, size_t xh_stride, const int32_t* ids, const int32_t* order,
                               const int32_t* runs, const MoeExpertView* views, int n, int routed_bits,
                               int shared_bits, float* out, int out_stride, int slots, int top_k,
                               int shared_view_base, int n_shared, cudaStream_t stream);
// The grouped (prefill) forms over the device segment table: the rotated
// rows of every segment (segment y's rows [row0, row0 + rows) of `act`
// against its expert's matrix `which` — which 0 writes the gate rotation
// into xh0 and the up rotation into xh1; which 2 the down rotation into
// xh0), then the GEMM per segment at one width per launch into out rows
// [row0, ...) (bf16 for gate / up, f32 for down). n, k: the launch's
// matrices' N and K.
void launch_exl3_moe_rotate_segments(const uint16_t* act, size_t act_stride, const MoeSegment* segs, int n_segs,
                                     int max_rows, const MoeExpertView* views, int which, int k, uint16_t* xh0,
                                     uint16_t* xh1, size_t xh_stride, cudaStream_t stream);
void launch_exl3_moe_grouped_bf16(const uint16_t* xh, size_t xh_stride, const MoeSegment* segs, int n_segs,
                                  int max_rows, const MoeExpertView* views, int which, int bits, uint16_t* out,
                                  size_t out_stride, int n, int k, cudaStream_t stream);
void launch_exl3_moe_grouped_f32(const uint16_t* xh, size_t xh_stride, const MoeSegment* segs, int n_segs,
                                 int max_rows, const MoeExpertView* views, int which, int bits, float* out,
                                 size_t out_stride, int n, int k, cudaStream_t stream);

}  // namespace dgpp
