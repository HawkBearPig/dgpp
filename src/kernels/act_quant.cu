#include <algorithm>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/act_quant.hpp"
#include "kernels/latent_format.hpp"
#include "kernels/pdl.cuh"

namespace dgpp {
namespace {
constexpr int kThreads = 128;
// blockIdx.y carries the row: a launch takes at most this many.
constexpr int kRowsPerLaunch = 65535;

__device__ inline float sigmoidf_acc(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// Eight lanes cooperate on each group of 32; each lane holds four adjacent
// values. kSwiglu: the values are the release's SwiGLU product of the gate
// and up rows, weighted, rounded once to BF16.
// y (BF16 rows of the dequantized values) and codes with scales (the E4M3
// bytes and one E8M0 byte per group) are each written when given: the two
// forms of one quantization.
template <bool kSwiglu>
__global__ void act_qdq_fp8_block_kernel(const uint16_t* __restrict__ x, size_t x_stride, uint16_t* __restrict__ y,
                                         size_t y_stride, int k, int row0, const uint16_t* __restrict__ up,
                                         const float* __restrict__ row_weight, float limit,
                                         uint8_t* __restrict__ codes, size_t code_stride,
                                         uint8_t* __restrict__ scales, size_t scale_stride) {
  dgpp_pdl_wait();
  const int row = row0 + static_cast<int>(blockIdx.y);
  const int column = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (column >= k) return;  // k % 32 == 0: a group's eight lanes leave together.
  const uint2 bits = *reinterpret_cast<const uint2*>(x + static_cast<size_t>(row) * x_stride + column);
  float v[4] = {__uint_as_float(bits.x << 16), __uint_as_float(bits.x & 0xffff0000u), __uint_as_float(bits.y << 16),
                __uint_as_float(bits.y & 0xffff0000u)};
  if constexpr (kSwiglu) {
    const uint2 ub = *reinterpret_cast<const uint2*>(up + static_cast<size_t>(row) * x_stride + column);
    const float u[4] = {__uint_as_float(ub.x << 16), __uint_as_float(ub.x & 0xffff0000u), __uint_as_float(ub.y << 16),
                        __uint_as_float(ub.y & 0xffff0000u)};
    const float w = row_weight != nullptr ? row_weight[row] : 1.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float g = v[i];
      if (g > limit) g = limit;  // the gate: no lower clamp
      const float uu = fminf(fmaxf(u[i], -limit), limit);
      // x = silu(gate) * up, then weights * x, all in fp32; one rounding.
      const float silu = __fmul_rn(g, sigmoidf_acc(g));
      v[i] = bf16_bits_to_float(float_to_bf16_bits(__fmul_rn(w, __fmul_rn(silu, uu))));
    }
  }
  float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  const unsigned active = __activemask();
#pragma unroll
  for (int offset = 4; offset; offset >>= 1) amax = fmaxf(amax, __shfl_xor_sync(active, amax, offset, 8));
  const uint8_t scale_byte = latent_fp8_block_scale_byte(amax);
  const float s = e8m0_byte_to_float(scale_byte);
  uint8_t c[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) c[i] = latent_fp8_block_encode(v[i], s);
  if (codes != nullptr) {
    *reinterpret_cast<uint32_t*>(codes + static_cast<size_t>(row) * code_stride + column) =
        static_cast<uint32_t>(c[0]) | (static_cast<uint32_t>(c[1]) << 8) | (static_cast<uint32_t>(c[2]) << 16) |
        (static_cast<uint32_t>(c[3]) << 24);
    if ((threadIdx.x & 7) == 0) scales[static_cast<size_t>(row) * scale_stride + column / 32] = scale_byte;
  }
  if (y != nullptr) {
    uint16_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = latent_fp8_block_decode_bf16(c[i], s);
    uint2 out;
    out.x = static_cast<uint32_t>(o[0]) | (static_cast<uint32_t>(o[1]) << 16);
    out.y = static_cast<uint32_t>(o[2]) | (static_cast<uint32_t>(o[3]) << 16);
    *reinterpret_cast<uint2*>(y + static_cast<size_t>(row) * y_stride + column) = out;
  }
}

__global__ void scatter_row_weights_kernel(const int32_t* __restrict__ slot_row, const float* __restrict__ weights, int n,
                                           float* __restrict__ row_weight) {
  dgpp_pdl_wait();
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  const int32_t r = slot_row[i];
  if (r >= 0) row_weight[r] = weights[i];
}

__global__ void fill_f32_kernel(float* __restrict__ dst, int n, float value) {
  dgpp_pdl_wait();
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) dst[i] = value;
}

void check_rows(const char* what, size_t stride, int k) {
  if (k <= 0 || k % 32 || stride < static_cast<size_t>(k) || stride % 4)
    throw std::invalid_argument(std::string(what) + ": requires 32-wide groups and rows aligned to four values");
}
void check_codes(const char* what, const uint8_t* codes, size_t code_stride, const uint8_t* scales,
                 size_t scale_stride, int k) {
  if (!codes || !scales) throw std::invalid_argument(std::string(what) + ": null codes");
  if (reinterpret_cast<uintptr_t>(codes) % 4 || code_stride % 4 || code_stride < static_cast<size_t>(k) ||
      scale_stride < static_cast<size_t>(k / 32))
    throw std::invalid_argument(std::string(what) + ": code rows must be aligned to four bytes and hold the row");
}
}  // namespace

size_t act_fp8_block_scale_stride(int k) { return (static_cast<size_t>(k) / 32 + 3) / 4 * 4; }

void launch_act_qdq_fp8_block(const uint16_t* x, size_t x_stride, uint16_t* y, size_t y_stride, int rows, int k,
                              cudaStream_t stream) {
  if (rows <= 0) return;
  if (!x || !y) throw std::invalid_argument("act_qdq_fp8_block: null rows");
  check_rows("act_qdq_fp8_block", x_stride, k);
  check_rows("act_qdq_fp8_block", y_stride, k);
  const unsigned gx = static_cast<unsigned>((k + 4 * kThreads - 1) / (4 * kThreads));
  for (int r0 = 0; r0 < rows; r0 += kRowsPerLaunch) {
    const dim3 grid(gx, static_cast<unsigned>(std::min(kRowsPerLaunch, rows - r0)));
    act_qdq_fp8_block_kernel<false><<<grid, kThreads, 0, stream>>>(x, x_stride, y, y_stride, k, r0, nullptr, nullptr,
                                                                 0.0f, nullptr, 0, nullptr, 0);
    DGPP_CUDA_OK(cudaGetLastError());
  }
}

void launch_act_quant_fp8_block(const uint16_t* x, size_t x_stride, uint8_t* codes, size_t code_stride,
                                uint8_t* scales, size_t scale_stride, int rows, int k, cudaStream_t stream,
                                uint16_t* y, size_t y_stride) {
  if (rows <= 0) return;
  if (!x) throw std::invalid_argument("act_quant_fp8_block: null rows");
  check_rows("act_quant_fp8_block", x_stride, k);
  if (y != nullptr) check_rows("act_quant_fp8_block", y_stride, k);
  check_codes("act_quant_fp8_block", codes, code_stride, scales, scale_stride, k);
  const unsigned gx = static_cast<unsigned>((k + 4 * kThreads - 1) / (4 * kThreads));
  for (int r0 = 0; r0 < rows; r0 += kRowsPerLaunch) {
    const dim3 grid(gx, static_cast<unsigned>(std::min(kRowsPerLaunch, rows - r0)));
    act_qdq_fp8_block_kernel<false><<<grid, kThreads, 0, stream>>>(x, x_stride, y, y_stride, k, r0, nullptr, nullptr,
                                                                 0.0f, codes, code_stride, scales, scale_stride);
    DGPP_CUDA_OK(cudaGetLastError());
  }
}

void launch_swiglu_weight_act_qdq_fp8_block(const uint16_t* gate, const uint16_t* up, size_t stride, uint16_t* act,
                                            int rows, int k, float limit, const float* row_weight,
                                            cudaStream_t stream) {
  if (rows <= 0) return;
  if (!gate || !up || !act) throw std::invalid_argument("swiglu_weight_act_qdq_fp8_block: null rows");
  if (!(limit > 0.0f)) throw std::invalid_argument("swiglu_weight_act_qdq_fp8_block: the clamp limit must be positive");
  check_rows("swiglu_weight_act_qdq_fp8_block", stride, k);
  const unsigned gx = static_cast<unsigned>((k + 4 * kThreads - 1) / (4 * kThreads));
  for (int r0 = 0; r0 < rows; r0 += kRowsPerLaunch) {
    const dim3 grid(gx, static_cast<unsigned>(std::min(kRowsPerLaunch, rows - r0)));
    act_qdq_fp8_block_kernel<true><<<grid, kThreads, 0, stream>>>(gate, stride, act, stride, k, r0, up, row_weight,
                                                                limit, nullptr, 0, nullptr, 0);
    DGPP_CUDA_OK(cudaGetLastError());
  }
}

void launch_swiglu_weight_act_quant_fp8_block(const uint16_t* gate, const uint16_t* up, size_t stride, uint8_t* codes,
                                              size_t code_stride, uint8_t* scales, size_t scale_stride, int rows,
                                              int k, float limit, const float* row_weight, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!gate || !up) throw std::invalid_argument("swiglu_weight_act_quant_fp8_block: null rows");
  if (!(limit > 0.0f)) throw std::invalid_argument("swiglu_weight_act_quant_fp8_block: the clamp limit must be positive");
  check_rows("swiglu_weight_act_quant_fp8_block", stride, k);
  check_codes("swiglu_weight_act_quant_fp8_block", codes, code_stride, scales, scale_stride, k);
  const unsigned gx = static_cast<unsigned>((k + 4 * kThreads - 1) / (4 * kThreads));
  for (int r0 = 0; r0 < rows; r0 += kRowsPerLaunch) {
    const dim3 grid(gx, static_cast<unsigned>(std::min(kRowsPerLaunch, rows - r0)));
    act_qdq_fp8_block_kernel<true><<<grid, kThreads, 0, stream>>>(gate, stride, nullptr, 0, k, r0, up, row_weight,
                                                                limit, codes, code_stride, scales, scale_stride);
    DGPP_CUDA_OK(cudaGetLastError());
  }
}

void launch_scatter_row_weights(const int32_t* slot_row, const float* weights, int n, float* row_weight,
                                cudaStream_t stream) {
  if (n <= 0) return;
  if (!slot_row || !weights || !row_weight) throw std::invalid_argument("scatter_row_weights: null buffer");
  scatter_row_weights_kernel<<<(n + 255) / 256, 256, 0, stream>>>(slot_row, weights, n, row_weight);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_fill_f32(float* dst, int n, float value, cudaStream_t stream) {
  if (n <= 0) return;
  if (!dst) throw std::invalid_argument("fill_f32: null buffer");
  fill_f32_kernel<<<(n + 255) / 256, 256, 0, stream>>>(dst, n, value);
  DGPP_CUDA_OK(cudaGetLastError());
}
}  // namespace dgpp
