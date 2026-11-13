#include <cuda_bf16.h>

#include "common/cuda_check.hpp"
#include "kernels/low_rank.hpp"
#include "kernels/pdl.cuh"

namespace dgpp {
namespace {
// The two dot products below issue their loads kLoadBatch terms at a time
// and consume them in order. Written a load a term, each term waited a
// memory round trip before the next was asked for: 50 terms a thread took
// 20 us in finish on the GB10. The sum is the same sequence of separate
// multiplies and adds either way, so the values are bit for bit the same.
constexpr int kLoadBatch = 16;

// digits (may be null): the rider's digit rows of a boundary whose
// rank-space rows number `count` in all, this launch's first at value0.
// The block writes its value's digits itself, pack_digits' bytes for this
// rank and its zeros for every other, so the collective launches behind
// this kernel with no pack kernel between.
__global__ void project(const uint16_t* x, int stride, const float* a, float* z, int inputs,
                        int rank, __nv_bfloat16* digits, size_t value0, size_t count, int my_rank,
                        int world) {
  dgpp_pdl_wait();
  const int factor = blockIdx.x, row = blockIdx.y;
  float sum = 0;
  const __nv_bfloat16* xr = reinterpret_cast<const __nv_bfloat16*>(x) + static_cast<size_t>(row) * stride;
  const float* ar = a + static_cast<size_t>(factor) * inputs;
  const int step = static_cast<int>(blockDim.x);
  for (int c = threadIdx.x; c < inputs; c += step * kLoadBatch) {
    float xv[kLoadBatch], av[kLoadBatch];
#pragma unroll
    for (int i = 0; i < kLoadBatch; ++i) {
      const int ci = c + i * step;
      const bool ok = ci < inputs;
      xv[i] = ok ? __bfloat162float(xr[ci]) : 0.f;
      av[i] = ok ? ar[ci] : 0.f;
    }
#pragma unroll
    for (int i = 0; i < kLoadBatch; ++i)
      if (c + i * step < inputs) sum = __fadd_rn(sum, __fmul_rn(xv[i], av[i]));
  }
  __shared__ float partial[128];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = 64; offset; offset /= 2) {
    if (threadIdx.x < offset)
      partial[threadIdx.x] = __fadd_rn(partial[threadIdx.x], partial[threadIdx.x + offset]);
    __syncthreads();
  }
  const float value = partial[0];
  if (threadIdx.x == 0) z[static_cast<size_t>(row) * rank + factor] = value;
  if (digits == nullptr) return;
  const size_t v = value0 + static_cast<size_t>(row) * rank + factor;
  for (int i = threadIdx.x; i < world * 4; i += blockDim.x) {
    const unsigned byte = i / 4 == my_rank ? (__float_as_uint(value) >> (8 * (i % 4))) & 255u : 0;
    digits[(static_cast<size_t>(i / 4) * count + v) * 4 + (i % 4)] =
        __float2bfloat16_rn(static_cast<float>(byte));
  }
}
template <bool Add>
__global__ void finish(const float* z, const float* b, uint16_t* output, float* delta, int rows,
                       int outputs, int rank) {
  dgpp_pdl_wait();
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = Add ? blockIdx.y : rows - 1;
  if (column >= outputs) return;
  float sum = 0;
  const float* bc = b + static_cast<size_t>(column) * rank;
  const float* zr = z + static_cast<size_t>(row) * rank;
  for (int r0 = 0; r0 < rank; r0 += kLoadBatch) {
    float bv[kLoadBatch], zv[kLoadBatch];
#pragma unroll
    for (int i = 0; i < kLoadBatch; ++i) {
      const bool ok = r0 + i < rank;
      bv[i] = ok ? bc[r0 + i] : 0.f;
      zv[i] = ok ? zr[r0 + i] : 0.f;
    }
#pragma unroll
    for (int i = 0; i < kLoadBatch; ++i)
      if (r0 + i < rank) sum = __fadd_rn(sum, __fmul_rn(bv[i], zv[i]));
  }
  if constexpr (Add) {
    auto* out =
        reinterpret_cast<__nv_bfloat16*>(output) + static_cast<size_t>(row) * outputs + column;
    *out = __float2bfloat16_rn(__fadd_rn(__bfloat162float(*out), sum));
  } else
    delta[column] = sum;
}
// The rider's finish: sum_digits and finish<true> in one kernel behind the
// collective that carries the digits. A block is C columns by R rows of
// threads (C = blockDim / R). Its slice of B goes to shared memory in the
// preamble, before the wait: behind a collective that releases its
// dependents that read runs beside the network round trip, and it is one
// read of B whatever the row count (finish reads B once a row). Past the
// wait the block sums the digits into shared memory with sum_digits'
// arithmetic, and thread (c, r) runs the chain of rows r, r + R, ... in
// finish's order. Block 0 also stores the summed rows to z, where
// sum_digits leaves them.
constexpr int kRiderThreads = 256;
constexpr int kDigitRanks = 4;  // ranks' digits issued together
template <int R>
__global__ void finish_rider(const __nv_bfloat16* digits, int world, float* z, const float* b,
                             uint16_t* output, int rows, int outputs, int rank) {
  dgpp_pdl_preamble();
  extern __shared__ float smem[];
  constexpr int C = kRiderThreads / R;
  float* bs = smem;             // [C][rank]: the block's columns of B
  float* zs = smem + C * rank;  // [rows][rank]: the summed rank-space rows
  const int col0 = blockIdx.x * C;
  const int nb = min(C, outputs - col0) * rank;
  const float* bsrc = b + static_cast<size_t>(col0) * rank;
  for (int i0 = threadIdx.x; i0 < nb; i0 += kRiderThreads * kLoadBatch) {
    float v[kLoadBatch];
#pragma unroll
    for (int k = 0; k < kLoadBatch; ++k) {
      const int i = i0 + k * kRiderThreads;
      v[k] = i < nb ? bsrc[i] : 0.f;
    }
#pragma unroll
    for (int k = 0; k < kLoadBatch; ++k) {
      const int i = i0 + k * kRiderThreads;
      if (i < nb) bs[i] = v[k];
    }
  }
  dgpp_pdl_wait();
  const int count = rows * rank;
  for (int v = threadIdx.x; v < count; v += kRiderThreads) {
    float sum = 0;
    for (int r0 = 0; r0 < world; r0 += kDigitRanks) {
      float d[kDigitRanks * 4];
#pragma unroll
      for (int k = 0; k < kDigitRanks * 4; ++k) {
        const int r = r0 + k / 4;
        d[k] = r < world
                   ? __bfloat162float(digits[(static_cast<size_t>(r) * count + v) * 4 + (k % 4)])
                   : 0.f;
      }
#pragma unroll
      for (int q = 0; q < kDigitRanks; ++q) {
        if (r0 + q < world) {
          uint32_t bits = 0;
#pragma unroll
          for (int byte = 0; byte < 4; ++byte)
            bits |= static_cast<uint32_t>(d[q * 4 + byte]) << (8 * byte);
          const float value = __uint_as_float(bits);
          sum = r0 + q == 0 ? value : __fadd_rn(sum, value);
        }
      }
    }
    zs[v] = sum;
    if (blockIdx.x == 0) z[v] = sum;
  }
  __syncthreads();
  const int c = threadIdx.x % C, column = col0 + c;
  if (column >= outputs) return;
  const float* bc = bs + c * rank;
  for (int row = threadIdx.x / C; row < rows; row += R) {
    const float* zr = zs + row * rank;
    float sum = 0;
#pragma unroll 8
    for (int i = 0; i < rank; ++i) sum = __fadd_rn(sum, __fmul_rn(bc[i], zr[i]));
    auto* out =
        reinterpret_cast<__nv_bfloat16*>(output) + static_cast<size_t>(row) * outputs + column;
    *out = __float2bfloat16_rn(__fadd_rn(__bfloat162float(*out), sum));
  }
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
  project<<<dim3(rank, rows), 128, 0, stream>>>(x, stride, a, z, inputs, rank, nullptr, 0, 0, 0, 0);
  DGPP_CUDA_OK(cudaGetLastError());
}
void low_rank_project_digits(const uint16_t* x, int stride, const float* a, float* z, int rows,
                             int inputs, int rank, uint16_t* digits, size_t value0, size_t count,
                             int my_rank, int world, cudaStream_t stream) {
  project<<<dim3(rank, rows), 128, 0, stream>>>(x, stride, a, z, inputs, rank,
                                                reinterpret_cast<__nv_bfloat16*>(digits), value0,
                                                count, my_rank, world);
  DGPP_CUDA_OK(cudaGetLastError());
}
namespace {
// Threads a block gives a column: the fewest of 2, 4, 8 that cover the rows
// (rows beyond eight loop) and keep the block's slice of B and the summed
// rows inside the default shared memory. 0: not even eight fit.
constexpr size_t kRiderSmemBytes = 48 * 1024;
size_t rider_smem(int rows_a_block, int rows, int rank) {
  return (static_cast<size_t>(kRiderThreads / rows_a_block) + rows) * rank * sizeof(float);
}
int rider_rows_a_block(int rows, int rank) {
  for (int r : {2, 4, 8})
    if ((rows <= r || r == 8) && rider_smem(r, rows, rank) <= kRiderSmemBytes) return r;
  return 0;
}
}  // namespace
void low_rank_add_digits(const uint16_t* digits, int world, float* z, const float* b,
                         uint16_t* output, int rows, int outputs, int rank, cudaStream_t stream) {
  const int r = rider_rows_a_block(rows, rank);
  if (r == 0) {  // a rank too wide for the block's shared memory: the two kernels
    low_rank_sum_digits(digits, z, static_cast<size_t>(rows) * rank, world, stream);
    low_rank_add(z, b, output, rows, outputs, rank, stream);
    return;
  }
  const int columns = kRiderThreads / r, blocks = (outputs + columns - 1) / columns;
  const size_t smem = rider_smem(r, rows, rank);
  const auto* d = reinterpret_cast<const __nv_bfloat16*>(digits);
  if (r == 2)
    finish_rider<2><<<blocks, kRiderThreads, smem, stream>>>(d, world, z, b, output, rows, outputs, rank);
  else if (r == 4)
    finish_rider<4><<<blocks, kRiderThreads, smem, stream>>>(d, world, z, b, output, rows, outputs, rank);
  else
    finish_rider<8><<<blocks, kRiderThreads, smem, stream>>>(d, world, z, b, output, rows, outputs, rank);
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
