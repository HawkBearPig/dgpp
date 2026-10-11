// The dense NVFP4 prefill GEMM: 256 x 128 x 64 tiles, eight warps,
// two cp.async activation stages and two XOR-swizzled weight tiles in
// 96 KiB of shared memory. Each e2m1/e4m3 weight product is exactly
// representable in BF16. The dot accumulates in FP32 in ascending k order,
// divides once by the FP32 global scale, then rounds once to BF16.
// Qwen35 uses this from 1024 rows; its smaller calls retain the production
// ldmatrix kernel and the established decode chain.
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
