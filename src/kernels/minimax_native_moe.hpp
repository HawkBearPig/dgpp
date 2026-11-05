#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {
// FlashInfer 0.7.2's default dynamic per-token NVFP4 producer. Payloads are
// [rows,k/2], E4M3 scales [rows,k/16], and FP32 decode globals [rows]. The
// reciprocal and multiplication boundaries match its CuTe/CUDA producer.
// Both inputs to the native routed-expert GEMMs use this producer.
void minimax_quantize_rows_nvfp4(const uint16_t* input, size_t stride, int rows, int k,
                                 uint8_t* codes, uint8_t* scales, float* globals, cudaStream_t);

// The native expert FC1 epilogue: FP32 gate/up accumulations, SwiGLU-OAI,
// then one BF16 rounding before the second GEMM's dynamic quantization.
void minimax_swiglu_f32(const float* gate, const float* up, uint16_t* output, int64_t count,
                        float alpha, float limit, cudaStream_t);
}  // namespace dgpp
