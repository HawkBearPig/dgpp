// The NVFP4-weight GEMM (fp4w_gemm.hpp): fp8w_gemm's structure (128 x 128
// x 64 tiles, sixteen warps, the three-stage cp.async activation pipeline,
// weights streamed once per 8-m-tile group) with an e2m1 W-side. The
// payload is k/2 bytes per row plus e4m3 scales per 16 codes; each
// thread places 16 codes a step (one 8-byte load plus the step's four
// scale bytes as one word), decoded pair by pair through the exact
// decode (cvt + hmul2, <= 5 significant bits — glm_moe's idiom,
// in-register here) with the row-step's e4m3 scale converted once and
// reused across the eight pairs. NVFP4 scales are per-16, finer than the
// fp8 form's per-128 groups, so there is no group promotion: the placed
// bf16 words are already scaled, one accumulator, and the F32 global
// divides the finished dot once in the epilogue. The dense NVFP4 path's
// prefill form (tokens > 128), the production ldmatrix kernel at and
// below (one chain everywhere at <= 128 rows; tolerance-equal, never
// bitwise, across the boundary — a different fp32 summation order).
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "models/quant_matrix.hpp"

namespace dgpp {

// out [m, n] at out_stride n (0 defaults to n). Requires k % 64 == 0,
// 16-byte-aligned act, act_stride a multiple of 8 with act_stride >= k,
// and the non-null NVFP4 triple (e2m1 payload [n, k/2], e4m3 scales
// [n, k/16], F32 global).
void launch_fp4w_gemm_bf16(const uint16_t* act, size_t act_stride, const GlmFp4Matrix& w,
                           uint16_t* out, int m, int n, int k, cudaStream_t stream,
                           size_t out_stride = 0);

bool fp4w_gemm_shape_ok(const void* act, size_t act_stride, int k);

}  // namespace dgpp
