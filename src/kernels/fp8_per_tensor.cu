// Per-tensor FP8 helpers for the Qwen3.8-27B prefill recipe
// (engine.prefill_fp8_per_tensor): the multi-CTA absmax over BF16 rows and
// the x/max quantize into E4M3 codes. Both are deterministic (max is
// order-independent; the quantize is elementwise).
#include "kernels/fp8_per_tensor.hpp"
#include "kernels/pdl.cuh"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {
constexpr int kThreads = 256;
}  // namespace

// Multi-CTA absmax: each CTA grid-strides over float4 groups (8 BF16 per
// coalesced 16B load; K is a multiple of 8 so groups tile the count, with a
// scalar tail for safety), tree-reduces in smem, then one atomicMax per CTA
// to *out_max. Values are finite non-negative by construction (inf folds to
// 448, NaN compares false and is ignored — same policy as the old single-CTA
// kernel), so integer atomicMax preserves float order. The caller zeroes
// *out_max on-stream first; max is order-independent, hence deterministic.
__global__ void fp8_maxabs_kernel(const uint16_t* __restrict__ data, size_t count,
                                  float* __restrict__ out_max) {
  dgpp_pdl_wait();
  __shared__ float sRed[kThreads];
  const int tid = threadIdx.x;
  const size_t n8 = count / 8;
  const size_t span = (size_t)gridDim.x * kThreads;
  const size_t base = (size_t)blockIdx.x * kThreads + tid;
  float mx = 0.f;
  const float4* vec = reinterpret_cast<const float4*>(data);
  for (size_t g = base; g < n8; g += span) {
    const float4 v = vec[g];
    const uint32_t* w = reinterpret_cast<const uint32_t*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t pair = w[j];
      float ax0 = fabsf(bf16_bits_to_float((uint16_t)(pair & 0xFFFFu)));
      float ax1 = fabsf(bf16_bits_to_float((uint16_t)(pair >> 16)));
      if (ax0 > 448.f) ax0 = 448.f;  // inf folds; NaN fails compare, ignored
      if (ax1 > 448.f) ax1 = 448.f;
      if (ax0 > mx) mx = ax0;
      if (ax1 > mx) mx = ax1;
    }
  }
  for (size_t j = n8 * 8 + base; j < count; j += span) {
    float ax = fabsf(bf16_bits_to_float(data[j]));
    if (ax > 448.f) ax = 448.f;
    if (ax > mx) mx = ax;
  }
  sRed[tid] = mx;
  __syncthreads();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (tid < s && sRed[tid + s] > sRed[tid]) sRed[tid] = sRed[tid + s];
    __syncthreads();
  }
  if (tid == 0 && sRed[0] > 0.f)
    atomicMax(reinterpret_cast<int*>(out_max), static_cast<int>(__float_as_int(sRed[0])));
}

// Elementwise x/mx quantize (`dev_max` the same-stream maxabs output, read
// once per thread and broadcast-cached): the codes span [-1, 1], so the Lt
// scalar scales are the raw maxabs values. (An x/448 map would need
// mx/448 scales; the relative quant error is identical either way — E4M3's
// precision is relative — but raw-mx scales keep the contract obvious.)
__global__ void fp8_quant_kernel(const uint16_t* __restrict__ in, uint8_t* __restrict__ out,
                                  size_t count, const float* __restrict__ dev_max) {
  dgpp_pdl_wait();
  float mx = *dev_max;
  if (mx == 0.f) mx = 1.f;  // all-zero guard (matches the old maxabs policy)
  const float inv = 1.f / mx;
  const size_t base = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = base; i < count; i += stride)
    out[i] = float_to_fp8_e4m3_bits(bf16_bits_to_float(in[i]) * inv);
}

void launch_fp8_row_maxabs(const uint16_t* data, size_t count, float* out_max,
                           cudaStream_t stream) {
  if (!data || !out_max || count == 0) throw std::invalid_argument("fp8 maxabs: bad args");
  if ((reinterpret_cast<uintptr_t>(data) % 16) != 0)
    throw std::invalid_argument("fp8 maxabs: data not 16B aligned for float4 loads");
  DGPP_CUDA_OK(cudaMemsetAsync(out_max, 0, sizeof(float), stream));  // atomicMax baseline
  const size_t n8 = count / 8;
  size_t blocks = (n8 + kThreads - 1) / kThreads;
  if (blocks > 1024) blocks = 1024;
  if (blocks == 0) blocks = 1;
  fp8_maxabs_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(data, count,
                                                                              out_max);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_fp8_quant_bf16(const uint16_t* in, uint8_t* out, size_t count,
                           const float* dev_max, cudaStream_t stream) {
  if (!in || !out || !dev_max || count == 0)
    throw std::invalid_argument("fp8 quant: bad args");
  const size_t blocks = (count + kThreads - 1) / kThreads;
  fp8_quant_kernel<<<(unsigned)(blocks > 1024 ? 1024 : blocks), kThreads, 0, stream>>>(
      in, out, count, dev_max);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
