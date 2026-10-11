#pragma once
// Scale-aware GEMM (DESIGN §4): consumes the resident compressed form of a
// quantized weight — E4M3 payload [N, K] plus F32 128x128 block scales —
// natively, with no BF16 materialization of the weight. The block scale is
// applied inside the weight-tile load path (decode x scale, one BF16
// round), producing weight tiles bit-identical to the transient dequant
// bridge; the MMA runs bf16 x bf16 with fp32 accumulation — the pinned
// reference numerics (dequantized weights, bf16 math), not DeepSeek-style
// dynamic activation quantization, which would change the semantics the
// M2/M3 references pinned.
//
//   D[M, N] = Act[M, K] x W[N, K]^T
//
// Tile geometry (BM=16, BN=64, BK=32) divides the 128-wide scale block
// exactly, so every stage applies one scalar scale (no per-element scale
// gather) and ragged N/K tails — anything not a multiple of 128 — read the
// true last block row/col with masked tile loads. Decode shapes up to eight
// rows use row-independent GEMV chunks, keeping each row bitwise invariant
// to serving occupancy. Both paths are deterministic and graph-capturable.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "models/quant_matrix.hpp"

namespace dgpp {

// act: bf16 row-major [M, act_row_stride_elems] (K-column slices allowed,
// matching the IGemm interface's fused-buffer views); w_payload: E4M3 row-major
// [N, K] contiguous; w_scales: F32 [ceil(N/128), ceil(K/128)] row-major;
// out: bf16 row-major [M, N], or [M, out_row_stride_elems] with the product
// in its first N columns when the stride is given (0: N) — a projection
// written into a column range of a wider buffer (the DSA layer's fused
// [q_a | kv_a] output from two fp8 pairs, 2026-09-08).
// mma_from_rows: rows from this count up to kScaleGemmMmaMaxRows take the
// streaming tensor-core GEMM (mma_gemv.hpp) instead of the chunked GEMV (0:
// never — the chunks to 128 rows, the 128-row dense kernel above, as
// before 2026-09-14). The chunks read the weights once per four rows; the
// streaming form once for every row count (m = 16 on the full GLM-5.3 DSA
// shapes: o_proj [6144 x 4096] 320 to 109 us, q_a [2048 x 6144] 155 to 59,
// q_b [4096 x 2048] 104 to 38; the 128-row dense kernel at m = 30, 726 to
// 147 on [5120 x 5120]). Above the bound the dense kernel keeps the prefill
// chunks (ahead of the streaming form's 128-row groups from 512 rows: 1060
// against 1099 us, 3479 against 4355 at 2048). The families pass
// dense_gemv_rows() + 1 (kernels/gemm.hpp): the GEMV chunks to four rows —
// the one-row floor and the fused multi-problem launches — the streaming
// form above. A row's chain is the chunk core's under the bound, the
// streaming form's from it, the dense kernel's past 256: not bitwise each
// other, so a site's rows reorder across the bounds (the engine gates'
// near-tie rule).
inline constexpr int kScaleGemmMmaMaxRows = 256;
// ws / ws_bytes: the caller's GEMM workspace, handed to the streaming
// tensor-core form for its split-K partials at a small n (mma_gemv.hpp);
// nullptr keeps that form unsplit. Within each 128-row launch group, only
// groups of at most 32 rows split, introducing another numerical boundary.
void launch_scale_gemm_bf16(const uint16_t* act, size_t act_row_stride_elems,
                            const uint8_t* w_payload, const float* w_scales,
                            uint16_t* out, int m, int n, int k,
                            cudaStream_t stream, size_t out_row_stride_elems = 0,
                            int mma_from_rows = 0, void* ws = nullptr, size_t ws_bytes = 0);

// The same product with the fp32 accumulators stored UNROUNDED: out is f32
// row-major [M, N]. bf16(out_f32[i]) == out_bf16[i] bit for bit — the two
// launchers differ only in the epilogue store. This is the MoE down
// projection's output (its partials feed an fp32 accumulation chain that
// rounds to bf16 once, at the end — see models/glm/moe_layer.hpp).
// last_row_only: write only row m-1 at its original output offset, retaining
// the dispatch and accumulation order of the full m-row product. Earlier
// output rows are untouched. Selecting an m=1 GEMV instead is not bitwise
// equivalent when the full product uses a tensor-core kernel.
// compact_row >= 0 selects that input row and writes it at output row zero,
// preserving the full m-row dispatch. Mutually exclusive with last_row_only.
void launch_scale_gemm_f32(const uint16_t* act, size_t act_row_stride_elems,
                           const uint8_t* w_payload, const float* w_scales,
                           float* out, int m, int n, int k,
                           cudaStream_t stream, size_t out_row_stride_elems = 0,
                           int mma_from_rows = 0, bool last_row_only = false, void* ws = nullptr,
                           size_t ws_bytes = 0, int compact_row = -1);

// The tile kernel regardless of m (the bf16 mma.sync m16n8k16 path the
// large-m route takes): the reference the grouped tensor-core MoE kernel is
// pinned bitwise against (glm_moe_test) — the routed launcher above would
// send small m to the GEMV core instead.
// rs / cs: the scale grid as log2 block sizes — 7 the
// checkpoint's 128 x 128, a TP slice's re-blocked axis 6 or 5 (plan D2);
// every row reads its own scale row, a 32-deep stage its one column.
// Several [n_i, k] fp8 matrices against the same activation rows in one
// launch (2026-09-10, the Qwen dense stack in FP8: a GDN layer's qkv + z,
// a QSA layer's q / k / v / indexer projections — one graph node in place
// of two or four). rows <= 8 (chunks of four); every out_i is bitwise
// launch_scale_gemm_bf16's for its problem (the same staged rows, the
// same row chain). out_stride 0: n.
struct Fp8GemvProblem {
  const uint8_t* payload = nullptr;
  const float* scales = nullptr;
  uint16_t* out = nullptr;
  int n = 0;
  size_t out_stride = 0;
  int rs = 7;  // the problem's scale grid, log2 (2026-09-13: 5 for a 32 x 32 grid)
  int cs = 7;
  // A bf16 [n, k] matrix in place of the fp8 payload (2026-09-29): its rows
  // run bf16_gemv::row_dots over the same staged activations, so the output
  // is bitwise launch_bf16_gemv_multi's for that problem — a GDN layer's
  // a / b projections ride the qkv + z launch (one launch and gap fewer).
  const uint16_t* bf16_weight = nullptr;
};
constexpr int kFp8GemvMaxProblems = 4;
void launch_scale_gemv_multi_bf16(const Fp8GemvProblem* problems, int n_problems,
                                  const uint16_t* act, size_t act_row_stride_elems, int rows,
                                  int k, cudaStream_t stream);

// The routed launcher on a stated scale grid (2026-09-13, the DeepSeek-V4.1
// release's fp8 matrices on 32 x 32 blocks): rs / cs as the tile launchers
// take them; small m runs the GEMV rows (the core reads any grid), larger
// m the tile kernel, or the exact pipelined fp8-weight GEMM for a 128x128
// scale grid and supported shape. Each output row of the GEMV path is bitwise the
// single-row launch; the two paths are tolerance-equal.
// decode_mma: every row count takes the streaming tensor-core GEMM
// (mma_gemv.hpp: the weights read once per 128 rows, each row's chain the
// same whatever m) instead of the 4-row GEMV chunks (m <= 128) or the
// large-row kernels above. A per-call opt-in — the forms are
// tolerance-equal, not bitwise — so a family switches every site or none
// (its batched decode rows must stay bitwise its rows alone). Shapes the
// mma form cannot take (k % 64, alignment) keep the older forms.
// ws / ws_bytes: the tensor-core form's split-K workspace (mma_gemv.hpp: a
// small-n site's k range across blocks at m <= 32, a chain that is the
// same whatever m but not the unsplit one) — a caller passes it for its
// decode walks only when its prefill chunks must stay bitwise each other.
void launch_scale_gemm_grid_bf16(const uint16_t* act, size_t act_row_stride_elems,
                                 const uint8_t* w_payload, const float* w_scales,
                                 uint16_t* out, int m, int n, int k, cudaStream_t stream,
                                 size_t out_row_stride_elems, int rs, int cs,
                                 bool decode_mma = false, void* ws = nullptr, size_t ws_bytes = 0);
void launch_scale_gemm_grid_f32(const uint16_t* act, size_t act_row_stride_elems,
                                const uint8_t* w_payload, const float* w_scales, float* out,
                                int m, int n, int k, cudaStream_t stream,
                                size_t out_row_stride_elems, int rs, int cs,
                                bool decode_mma = false, void* ws = nullptr, size_t ws_bytes = 0);

// Whether the fp8 GEMV (the scale GEMM's small-m lowering) can stage a payload
// at all: the grid launcher's shape check at one activation row (it chunks
// rows down to fit its smem, so one row fitting is the whole question).
// Callers deciding between the GEMV and the dequant bridge use this rather
// than re-deriving fp8_gemv's shape rules.
bool fp8_gemv_can_stage(const void* w_payload, int k);

void launch_scale_gemm_tile_bf16(const uint16_t* act, size_t act_row_stride_elems,
                                 const uint8_t* w_payload, const float* w_scales,
                                 uint16_t* out, int m, int n, int k,
                                 cudaStream_t stream, int rs = 7, int cs = 7);
void launch_scale_gemm_tile_f32(const uint16_t* act, size_t act_row_stride_elems,
                                const uint8_t* w_payload, const float* w_scales,
                                float* out, int m, int n, int k,
                                cudaStream_t stream, int rs = 7, int cs = 7);

}  // namespace dgpp
