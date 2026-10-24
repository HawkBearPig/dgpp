#include "kernels/hadamard32.hpp"
#include "kernels/pdl.cuh"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "kernels/hadamard32.cuh"

namespace dgpp {
namespace {

// One warp per 32-element block; eight warps a CTA.
__global__ void hadamard32_rows_kernel(const uint16_t* __restrict__ in, size_t in_stride,
                                       uint16_t* __restrict__ out, size_t out_stride, int rows,
                                       int k, int period) {
  dgpp_pdl_wait();
  const int blocks_per_row = k / kHadamard32Block;
  const long long b = static_cast<long long>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const long long total = static_cast<long long>(rows) * blocks_per_row;
  if (b >= total) return;
  const int r = static_cast<int>(b / blocks_per_row), c = static_cast<int>(b % blocks_per_row) * kHadamard32Block;
  if (period > 0 && r % period == period - 1) return;
  const int lane = threadIdx.x & 31;
  const uint16_t x = in[static_cast<size_t>(r) * in_stride + c + lane];
  out[static_cast<size_t>(r) * out_stride + c + lane] = hadamard32_warp_bf16(x);
}

// The Mixed346 activation quantizer: one warp per 128-value group of a
// row — four H32 blocks (lane j holds element j of each), the rotated
// values rounded to bf16, then the group's fp32 scale s = max(amax / 127,
// 1e-30) and the codes clamp(rint(x / s), -128, 127) (ties to even; the
// division rounded to nearest). Rows `r % period == period - 1` are left
// unwritten when period > 0.
__global__ void hadamard32_quant_int8_kernel(const uint16_t* __restrict__ in, size_t in_stride,
                                             int8_t* __restrict__ codes, size_t code_stride,
                                             float* __restrict__ scales, size_t scale_stride, int rows,
                                             int k, int period) {
  dgpp_pdl_wait();
  const int groups_per_row = k / 128;
  const long long g = static_cast<long long>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const long long total = static_cast<long long>(rows) * groups_per_row;
  if (g >= total) return;
  const int r = static_cast<int>(g / groups_per_row), c = static_cast<int>(g % groups_per_row) * 128;
  if (period > 0 && r % period == period - 1) return;
  const int lane = threadIdx.x & 31;
  const uint16_t* src = in + static_cast<size_t>(r) * in_stride + c;
  float v[4];
  float amax = 0.f;
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    v[b] = bf16_bits_to_float(hadamard32_warp_bf16(src[b * kHadamard32Block + lane]));
    amax = fmaxf(amax, fabsf(v[b]));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
  const float s = fmaxf(__fdiv_rn(amax, 127.f), 1e-30f);
  int8_t* dst = codes + static_cast<size_t>(r) * code_stride + c;
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    float q = rintf(__fdiv_rn(v[b], s));
    q = fminf(fmaxf(q, -128.f), 127.f);
    dst[b * kHadamard32Block + lane] = static_cast<int8_t>(static_cast<int>(q));
  }
  if (lane == 0) scales[static_cast<size_t>(r) * scale_stride + c / 128] = s;
}

}  // namespace

void launch_hadamard32_quant_int8_rows(const uint16_t* in, size_t in_stride, int8_t* codes, size_t code_stride,
                                       float* scales, size_t scale_stride, int rows, int k, int period,
                                       cudaStream_t stream) {
  if (rows <= 0) return;
  if (period < 0) throw std::invalid_argument("hadamard32 quant: period must be non-negative");
  if (!in || !codes || !scales) throw std::invalid_argument("hadamard32 quant: null pointer");
  if (k <= 0 || k % 128 != 0) throw std::invalid_argument("hadamard32 quant: k must be a positive multiple of 128");
  if (in_stride < static_cast<size_t>(k) || code_stride < static_cast<size_t>(k) ||
      scale_stride < static_cast<size_t>(k / 128))
    throw std::invalid_argument("hadamard32 quant: row stride below k");
  constexpr int kThreads = 256;
  const long long groups = static_cast<long long>(rows) * (k / 128);
  const long long grid = (groups + (kThreads / 32) - 1) / (kThreads / 32);
  hadamard32_quant_int8_kernel<<<static_cast<unsigned>(grid), kThreads, 0, stream>>>(
      in, in_stride, codes, code_stride, scales, scale_stride, rows, k, period);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_hadamard32_rows_skip(const uint16_t* in, size_t in_stride, uint16_t* out,
                                 size_t out_stride, int rows, int k, int period,
                                 cudaStream_t stream) {
  if (rows <= 0) return;
  if (period < 0) throw std::invalid_argument("hadamard32: period must be non-negative");
  if (!in || !out) throw std::invalid_argument("hadamard32: null pointer");
  if (k <= 0 || k % kHadamard32Block != 0)
    throw std::invalid_argument("hadamard32: k must be a positive multiple of 32");
  if (in_stride < static_cast<size_t>(k) || out_stride < static_cast<size_t>(k))
    throw std::invalid_argument("hadamard32: row stride below k");
  constexpr int kThreads = 256;
  const long long blocks = static_cast<long long>(rows) * (k / kHadamard32Block);
  const long long grid = (blocks + (kThreads / 32) - 1) / (kThreads / 32);
  hadamard32_rows_kernel<<<static_cast<unsigned>(grid), kThreads, 0, stream>>>(in, in_stride, out,
                                                                               out_stride, rows, k,
                                                                               period);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_hadamard32_rows(const uint16_t* in, size_t in_stride, uint16_t* out, size_t out_stride,
                            int rows, int k, cudaStream_t stream) {
  launch_hadamard32_rows_skip(in, in_stride, out, out_stride, rows, k, 0, stream);
}

}  // namespace dgpp
