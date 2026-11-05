#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {
// Dynamic MXFP8: BF16 [rows,k] -> E4M3 [rows,k], E8M0 [rows,k/32].
// Each 32-value group's scale is ceil-rounded to a power of two, matching
// the pinned SGLang MiniMax producer. x_stride is in BF16 elements.
void launch_mxfp8_quantize_rows(const uint16_t* x, size_t x_stride, int rows, int k, uint8_t* codes,
                                uint8_t* scales, cudaStream_t);

// Pinned SGLang dense/shared SwiGLU-OAI producer, including its sigmoid
// evaluation and BF16 boundary. Gate and up have the same BF16 row stride.
void launch_minimax_swiglu_mxfp8_rows(const uint16_t* gate, const uint16_t* up, size_t stride,
                                      int rows, int k, float alpha, float limit, uint8_t* codes,
                                      uint8_t* scales, cudaStream_t);

// Native SM121 block-scaled E4M3 x E4M3, FP32 accumulation. Both operands
// are row-major with k-contiguous payloads and k/32-contiguous E8M0 scales.
// out[m,n] = A[m,k] * W[n,k]^T; out_stride is in output elements (0 -> n).
// All supported row geometries retain the same ordered k32 MMA chain.
// output_tile overrides the column tile for kernel diagnostics (8/16/32/64);
// zero selects the production policy. Overrides use sixteen-row tiles.
void launch_mxfp8_gemm_bf16(const uint8_t* a, const uint8_t* a_scales, const uint8_t* w,
                            const uint8_t* w_scales, uint16_t* out, int m, int n, int k,
                            cudaStream_t, size_t out_stride = 0, int output_tile = 0);
void launch_mxfp8_gemm_f32(const uint8_t* a, const uint8_t* a_scales, const uint8_t* w,
                           const uint8_t* w_scales, float* out, int m, int n, int k, cudaStream_t,
                           size_t out_stride = 0, int output_tile = 0);
}  // namespace dgpp
