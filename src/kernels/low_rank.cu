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
}  // namespace dgpp
