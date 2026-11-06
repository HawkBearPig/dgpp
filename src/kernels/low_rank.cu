#include <cuda_bf16.h>

#include "common/cuda_check.hpp"
#include "kernels/low_rank.hpp"
#include "kernels/pdl.cuh"

namespace dgpp {
namespace {
__global__ void project(const uint16_t* x, int stride, const float* a, float* z, int inputs,
                        int rank) {
  dgpp_pdl_wait();
  const int factor = blockIdx.x, row = blockIdx.y;
  float sum = 0;
  for (int c = threadIdx.x; c < inputs; c += blockDim.x) {
    const float input = __bfloat162float(
        reinterpret_cast<const __nv_bfloat16*>(x)[static_cast<size_t>(row) * stride + c]);
    sum = __fadd_rn(sum, __fmul_rn(input, a[static_cast<size_t>(factor) * inputs + c]));
  }
  __shared__ float partial[128];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = 64; offset; offset /= 2) {
    if (threadIdx.x < offset)
      partial[threadIdx.x] = __fadd_rn(partial[threadIdx.x], partial[threadIdx.x + offset]);
    __syncthreads();
  }
  if (threadIdx.x == 0) z[static_cast<size_t>(row) * rank + factor] = partial[0];
}
template <bool Add>
__global__ void finish(const float* z, const float* b, uint16_t* output, float* delta, int rows,
                       int outputs, int rank) {
  dgpp_pdl_wait();
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = Add ? blockIdx.y : rows - 1;
  if (column >= outputs) return;
  float sum = 0;
  for (int r = 0; r < rank; ++r)
    sum = __fadd_rn(sum, __fmul_rn(b[static_cast<size_t>(column) * rank + r],
                                   z[static_cast<size_t>(row) * rank + r]));
  if constexpr (Add) {
    auto* out =
        reinterpret_cast<__nv_bfloat16*>(output) + static_cast<size_t>(row) * outputs + column;
    *out = __float2bfloat16_rn(__fadd_rn(__bfloat162float(*out), sum));
  } else
    delta[column] = sum;
}
__global__ void pack_digits(const float* values, __nv_bfloat16* digits, size_t count, int rank,
                            int world) {
  dgpp_pdl_wait();
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count * world * 4) return;
  const size_t value = i / 4;
  const unsigned byte = value / count == static_cast<size_t>(rank)
                            ? (__float_as_uint(values[value % count]) >> (8 * (i % 4))) & 255u
                            : 0;
  digits[i] = __float2bfloat16_rn(static_cast<float>(byte));
}
__global__ void sum_digits(const __nv_bfloat16* digits, float* values, size_t count, int world) {
  dgpp_pdl_wait();
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  float sum = 0;
  for (int rank = 0; rank < world; ++rank) {
    uint32_t bits = 0;
    for (int byte = 0; byte < 4; ++byte)
      bits |= static_cast<uint32_t>(__bfloat162float(digits[(rank * count + i) * 4 + byte]))
              << (8 * byte);
    const float value = __uint_as_float(bits);
    sum = rank == 0 ? value : __fadd_rn(sum, value);
  }
  values[i] = sum;
}
}  // namespace
void low_rank_project(const uint16_t* x, int stride, const float* a, float* z, int rows, int inputs,
                      int rank, cudaStream_t stream) {
  project<<<dim3(rank, rows), 128, 0, stream>>>(x, stride, a, z, inputs, rank);
  DGPP_CUDA_OK(cudaGetLastError());
}
void low_rank_add(const float* z, const float* b, uint16_t* output, int rows, int outputs, int rank,
                  cudaStream_t stream) {
  finish<true><<<dim3((outputs + 127) / 128, rows), 128, 0, stream>>>(z, b, output, nullptr, rows,
                                                                      outputs, rank);
  DGPP_CUDA_OK(cudaGetLastError());
}
void low_rank_delta_last(const float* z, const float* b, float* delta, int rows, int outputs,
                         int rank, cudaStream_t stream) {
  finish<false>
      <<<(outputs + 127) / 128, 128, 0, stream>>>(z, b, nullptr, delta, rows, outputs, rank);
  DGPP_CUDA_OK(cudaGetLastError());
}
void low_rank_pack_digits(const float* values, uint16_t* digits, size_t count, int rank, int world,
                          cudaStream_t stream) {
  pack_digits<<<(count * world * 4 + 255) / 256, 256, 0, stream>>>(
      values, reinterpret_cast<__nv_bfloat16*>(digits), count, rank, world);
  DGPP_CUDA_OK(cudaGetLastError());
}
void low_rank_sum_digits(const uint16_t* digits, float* values, size_t count, int world,
                         cudaStream_t stream) {
  sum_digits<<<(count + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(digits), values, count, world);
  DGPP_CUDA_OK(cudaGetLastError());
}
}  // namespace dgpp
