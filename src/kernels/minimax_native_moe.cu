#include <algorithm>
#include <cfloat>
#include <stdexcept>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "common/cuda_check.hpp"
#include "kernels/minimax_native_moe.hpp"
#include "kernels/pdl.cuh"

namespace dgpp {
namespace {
__device__ __forceinline__ float reciprocal(float value) {
  float result;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}
__device__ __forceinline__ uint32_t pack8(const float* x) {
  uint32_t packed;
  asm volatile(
      "{ .reg .b8 a,b,c,d;"
      "cvt.rn.satfinite.e2m1x2.f32 a, %2, %1;"
      "cvt.rn.satfinite.e2m1x2.f32 b, %4, %3;"
      "cvt.rn.satfinite.e2m1x2.f32 c, %6, %5;"
      "cvt.rn.satfinite.e2m1x2.f32 d, %8, %7;"
      "mov.b32 %0, {a,b,c,d}; }"
      : "=r"(packed)
      : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
  return packed;
}

// One thread owns a group of sixteen, up to two groups per thread. This
// retains group values through the row-maximum reduction without a second
// read. All row geometries use the identical scale and encoding operations.
__global__ void minimax_quantize_nvfp4_kernel(const uint16_t* input, size_t stride, int k,
                                              uint8_t* codes, uint8_t* scales, float* globals) {
  dgpp_pdl_wait();
  const int row = blockIdx.x, tid = threadIdx.x, groups = k / 16;
  float values[2][16], maxima[2] = {0.f, 0.f};
  float maximum = 0.f;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int group = tid + j * blockDim.x;
    if (group >= groups) continue;
    const uint16_t* source = input + static_cast<size_t>(row) * stride + group * 16;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      values[j][i] = __uint_as_float(static_cast<uint32_t>(source[i]) << 16);
      maxima[j] = fmaxf(maxima[j], fabsf(values[j][i]));
    }
    maximum = fmaxf(maximum, maxima[j]);
  }
  for (int offset = 16; offset; offset >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(~0u, maximum, offset));
  __shared__ float warp_max[32];
  if (tid % 32 == 0) warp_max[tid / 32] = maximum;
  __syncthreads();
  if (tid < 32) {
    float value = tid < blockDim.x / 32 ? warp_max[tid] : 0.f;
    for (int offset = 16; offset; offset >>= 1)
      value = fmaxf(value, __shfl_xor_sync(~0u, value, offset));
    if (tid == 0) warp_max[0] = value;
  }
  __syncthreads();
  const float row_max = warp_max[0];
  const float decode = __fmul_rn(row_max, 1.f / (6.f * 448.f));
  const float encode = row_max == 0.f ? FLT_MAX : reciprocal(decode);
  if (tid == 0) globals[row] = decode;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int group = tid + j * blockDim.x;
    if (group >= groups) continue;
    const float scale_float = __fmul_rn(encode, __fmul_rn(maxima[j], reciprocal(6.f)));
    const __nv_fp8_e4m3 scale(scale_float);
    const float scale_value = static_cast<float>(scale);
    const float inverse =
        scale_value == 0.f ? 0.f : reciprocal(__fmul_rn(scale_value, reciprocal(encode)));
    scales[static_cast<size_t>(row) * groups + group] = scale.__x;
#pragma unroll
    for (int i = 0; i < 16; ++i) values[j][i] = __fmul_rn(values[j][i], inverse);
    uint8_t* dest = codes + static_cast<size_t>(row) * (k / 2) + group * 8;
    *reinterpret_cast<uint2*>(dest) = make_uint2(pack8(values[j]), pack8(values[j] + 8));
  }
}

__global__ void minimax_swiglu_f32_kernel(const float* gate, const float* up, uint16_t* out,
                                          int64_t count, float alpha, float limit) {
  dgpp_pdl_wait();
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float g = fminf(gate[i], limit), u = fminf(fmaxf(up[i], -limit), limit);
  const float activated = (g / (1.f + expf(-alpha * g))) * (u + 1.f);
  out[i] = __bfloat16_as_ushort(__float2bfloat16_rn(activated));
}
}  // namespace

void minimax_quantize_rows_nvfp4(const uint16_t* input, size_t stride, int rows, int k,
                                 uint8_t* codes, uint8_t* scales, float* globals,
                                 cudaStream_t stream) {
  if (rows <= 0) return;
  if (k <= 0 || k > 32768 || k % 128 || stride < static_cast<size_t>(k))
    throw std::invalid_argument("MiniMax native NVFP4: k must be 128-aligned and <= 32768");
  const int threads = std::min(1024, std::max(32, ((k / 16 + 31) / 32) * 32));
  minimax_quantize_nvfp4_kernel<<<rows, threads, 0, stream>>>(input, stride, k, codes, scales,
                                                              globals);
  DGPP_CUDA_OK(cudaGetLastError());
}
void minimax_swiglu_f32(const float* gate, const float* up, uint16_t* output, int64_t count,
                        float alpha, float limit, cudaStream_t stream) {
  if (count <= 0) return;
  minimax_swiglu_f32_kernel<<<(count + 255) / 256, 256, 0, stream>>>(gate, up, output, count, alpha,
                                                                     limit);
  DGPP_CUDA_OK(cudaGetLastError());
}
}  // namespace dgpp
