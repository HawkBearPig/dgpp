#pragma once
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {
// FP32 factors, BF16 inputs, FP32 rank-space products. No intermediate BF16
// rounding. Input rows may be strided shared-expert slots.
void low_rank_project(const uint16_t* x, int x_stride, const float* a, float* z, int rows,
                      int inputs, int rank, cudaStream_t stream);
// Add the global FP32 Bz once to an already folded BF16 branch output.
void low_rank_add(const float* z, const float* b, uint16_t* output, int rows, int outputs, int rank,
                  cudaStream_t stream);
void low_rank_delta_last(const float* z, const float* b, float* delta, int rows, int outputs,
                         int rank, cudaStream_t stream);
// Lossless FP32 allgather over the BF16 collective: each rank writes four
// numeric byte digits into its own slice and zero into every other slice.
void low_rank_pack_digits(const float* values, uint16_t* digits, size_t count, int rank, int world,
                          cudaStream_t stream);
void low_rank_sum_digits(const uint16_t* digits, float* values, size_t count, int world,
                         cudaStream_t stream);
}  // namespace dgpp
