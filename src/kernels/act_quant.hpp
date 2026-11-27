#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {
// The DeepSeek release's activation quantization (inference/kernel.py
// act_quant with a power-of-two scale: FP8 e4m3, one e8m0 scale per 32
// values along the row) as quantize-then-dequantize. Per row and per group
// of 32: absmax floored at 1e-4, s = 2^ceil(log2(absmax / 448)) from the
// fp32 quotient's bit fields, y = bf16(e4m3(x / s) * s). Every y is exactly
// representable in BF16, and the release's fp8_gemm / fp4_gemm accumulate
// the scaled products in fp32, so a BF16-activation multiply of y against
// the exactly decoded weights is the release's multiply up to the order of
// the sum. Uses latent_format.hpp's helpers: bitwise the window cache rows'
// quantizer.
//
// x and y are BF16 [rows, k] with strides in elements (multiples of 4, at
// least k); k is a multiple of 32. y may be x (in place).
void launch_act_qdq_fp8_block(const uint16_t* x, size_t x_stride, uint16_t* y, size_t y_stride, int rows, int k,
                              cudaStream_t stream);

// The release's Expert.forward between its projections, then the same
// quantize-dequantize: act = qdq(bf16(silu(min(gate, limit)) * clamp(up,
// -limit, limit) * w)), the product in fp32 with one BF16 rounding (gate
// and up are the BF16 rows of the two projections). row_weight[row] is the
// route's weight (the release multiplies it in before the down
// projection); nullptr is 1 for every row. gate, up and act share one
// stride; act may be gate or up.
void launch_swiglu_weight_act_qdq_fp8_block(const uint16_t* gate, const uint16_t* up, size_t stride, uint16_t* act,
                                            int rows, int k, float limit, const float* row_weight,
                                            cudaStream_t stream);

// The same two quantizations as the operands of the block-scaled tensor-core
// multiplies (kernels/moe_a8w4.hpp): codes are the E4M3 bytes [rows,
// code_stride] and scales one E8M0 byte per group of 32 [rows, scale_stride]
// (value = code * 2^(scale - 127)). Code rows are aligned to four bytes;
// act_fp8_block_scale_stride(k) is k / 32 rounded up to four. y, when given,
// also receives the dequantized BF16 rows: bitwise launch_act_qdq_fp8_block.
size_t act_fp8_block_scale_stride(int k);
void launch_act_quant_fp8_block(const uint16_t* x, size_t x_stride, uint8_t* codes, size_t code_stride,
                                uint8_t* scales, size_t scale_stride, int rows, int k, cudaStream_t stream,
                                uint16_t* y = nullptr, size_t y_stride = 0);
void launch_swiglu_weight_act_quant_fp8_block(const uint16_t* gate, const uint16_t* up, size_t stride, uint8_t* codes,
                                              size_t code_stride, uint8_t* scales, size_t scale_stride, int rows,
                                              int k, float limit, const float* row_weight, cudaStream_t stream);

// row_weight[slot_row[i]] = weights[i] for i in [0, n): the routes' weights
// by grouped row (slot_row is the segmentation's (token, slot) -> row map).
void launch_scatter_row_weights(const int32_t* slot_row, const float* weights, int n, float* row_weight,
                                cudaStream_t stream);
// dst[i] = value for i in [0, n).
void launch_fill_f32(float* dst, int n, float value, cudaStream_t stream);
}  // namespace dgpp
