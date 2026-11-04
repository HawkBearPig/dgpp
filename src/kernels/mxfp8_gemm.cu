#include <stdexcept>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "common/cuda_check.hpp"
#include "kernels/mxfp8_gemm.hpp"
#include "kernels/pdl.cuh"

namespace dgpp {
namespace {
constexpr int kThreads = 128, kBK = 64, kBN = 64, kStride = 80, kStages = 3;

__global__ void mxfp8_quantize_kernel(const uint16_t* x, size_t stride, int k, uint8_t* codes,
                                      uint8_t* scales) {
  dgpp_pdl_wait();
  // Eight lanes cooperate on each group; each lane loads four adjacent BF16s.
  const int row = blockIdx.y;
  const int column = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (column >= k) return;  // k % 32 guarantees complete participating groups.
  const uint2 bits =
      *reinterpret_cast<const uint2*>(x + static_cast<size_t>(row) * stride + column);
  float values[4] = {__uint_as_float(bits.x << 16), __uint_as_float(bits.x & 0xffff0000u),
                     __uint_as_float(bits.y << 16), __uint_as_float(bits.y & 0xffff0000u)};
  float maximum =
      fmaxf(fmaxf(fabsf(values[0]), fabsf(values[1])), fmaxf(fabsf(values[2]), fabsf(values[3])));
  const unsigned active = __activemask();
#pragma unroll
  for (int offset = 4; offset; offset >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(active, maximum, offset, 8));
  // Exactly ceil(log2(amax / 448)) from IEEE bits, without log2 rounding
  // at the exponent boundary. Zero groups use scale byte zero and zero codes.
  const int biased = static_cast<int>((__float_as_uint(maximum) >> 23) & 255);
  const float threshold = __uint_as_float((static_cast<unsigned>(biased) << 23) | 0x600000u);
  const int exponent = max(-127, biased - 135 + (maximum > threshold));
  const float inverse = __uint_as_float(static_cast<unsigned>(127 - exponent) << 23);
  const uint16_t lo = __nv_cvt_float2_to_fp8x2(
      make_float2(values[0] * inverse, values[1] * inverse), __NV_SATFINITE, __NV_E4M3);
  const uint16_t hi = __nv_cvt_float2_to_fp8x2(
      make_float2(values[2] * inverse, values[3] * inverse), __NV_SATFINITE, __NV_E4M3);
  *reinterpret_cast<uint32_t*>(codes + static_cast<size_t>(row) * k + column) =
      static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
  if ((threadIdx.x & 7) == 0)
    scales[static_cast<size_t>(row) * (k / 32) + column / 32] =
        static_cast<uint8_t>(exponent + 127);
}

__device__ __forceinline__ void copy16(void* dst, const void* src, bool valid) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address), "l"(src),
               "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void commit() {
  asm volatile("cp.async.commit_group;");
}
template <int Count>
__device__ __forceinline__ void wait_pending() {
  asm volatile("cp.async.wait_group %0;" ::"n"(Count));
}

__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                    uint32_t as, uint32_t bs) {
  // PTX/CUTLASS SM120 scale layout: A scale row = lane/4 + 8*(lane%2),
  // B scale row = lane/4; the byte/thread selectors are both zero.
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
      "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
      "{%10}, {%11,%12}, {%13}, {%14,%15};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(as),
        "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)), "r"(bs),
        "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)));
}

template <typename Out, int BM>
__global__ __launch_bounds__(kThreads) void mxfp8_gemm_kernel(
    const uint8_t* a, const uint8_t* a_scales, const uint8_t* w, const uint8_t* w_scales, Out* out,
    int m, int n, int k, size_t out_stride) {
  dgpp_pdl_wait();
  constexpr int warp_m = BM == 16 ? 1 : 2, warp_n = 4 / warp_m;
  constexpr int MT = BM / warp_m / 16, NT = kBN / warp_n / 8;
  constexpr int slot_bytes = (BM + kBN) * kStride;
  extern __shared__ __align__(16) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int m_tiles = (m + BM - 1) / BM;
  const int m0 = (blockIdx.x % m_tiles) * BM, n0 = (blockIdx.x / m_tiles) * kBN;
  const int row_base = (warp / warp_n) * (BM / warp_m);
  const int col_base = (warp % warp_n) * (kBN / warp_n);
  const int group = lane / 4, part = lane % 4;
  const int steps = k / kBK, scale_stride = k / 32;
  auto issue = [&](int step, int slot) {
#pragma unroll
    for (int chunk = tid; chunk < (BM + kBN) * 4; chunk += kThreads) {
      const int row = chunk / 4, column = (chunk % 4) * 16;
      const bool is_a = row < BM;
      const int global_row = is_a ? m0 + row : n0 + row - BM;
      const bool valid = global_row < (is_a ? m : n);
      const uint8_t* src = (is_a ? a : w) + static_cast<size_t>(valid ? global_row : 0) * k;
      copy16(smem + slot * slot_bytes + row * kStride + column, src + step * kBK + column, valid);
    }
  };
  float accum[MT][NT][4] = {};
#pragma unroll
  for (int stage = 0; stage < kStages - 1; ++stage) {
    if (stage < steps) issue(stage, stage);
    commit();
  }
  for (int step = 0; step < steps; ++step) {
    wait_pending<kStages - 2>();
    __syncthreads();
    if (step + kStages - 1 < steps) issue(step + kStages - 1, (step + kStages - 1) % kStages);
    commit();
    const uint8_t* sa = smem + (step % kStages) * slot_bytes;
    const uint8_t* sw = sa + BM * kStride;
#pragma unroll
    for (int kk = 0; kk < kBK; kk += 32) {
      uint32_t af[MT][4], bf[NT][2], as[MT], bs[NT];
      const int scale_column = (step * kBK + kk) / 32;
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        const int row = row_base + i * 16 + group;
        const uint8_t* p = sa + row * kStride + kk + part * 4;
        af[i][0] = *reinterpret_cast<const uint32_t*>(p);
        af[i][1] = *reinterpret_cast<const uint32_t*>(p + 8 * kStride);
        af[i][2] = *reinterpret_cast<const uint32_t*>(p + 16);
        af[i][3] = *reinterpret_cast<const uint32_t*>(p + 8 * kStride + 16);
        const int scale_row = m0 + row + (lane & 1) * 8;
        as[i] = scale_row < m
                    ? a_scales[static_cast<size_t>(scale_row) * scale_stride + scale_column]
                    : 127;
      }
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const int row = col_base + j * 8 + group;
        const uint8_t* p = sw + row * kStride + kk + part * 4;
        bf[j][0] = *reinterpret_cast<const uint32_t*>(p);
        bf[j][1] = *reinterpret_cast<const uint32_t*>(p + 16);
        const int scale_row = n0 + row;
        bs[j] = scale_row < n
                    ? w_scales[static_cast<size_t>(scale_row) * scale_stride + scale_column]
                    : 127;
      }
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) mma(accum[i][j], af[i], bf[j][0], bf[j][1], as[i], bs[j]);
    }
    // No warp may reuse a ring slot until every warp has consumed it.
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int value = 0; value < 4; ++value) {
        const int row = m0 + row_base + i * 16 + group + (value / 2) * 8;
        const int column = n0 + col_base + j * 8 + part * 2 + value % 2;
        if (row < m && column < n) {
          if constexpr (std::is_same_v<Out, float>)
            out[static_cast<size_t>(row) * out_stride + column] = accum[i][j][value];
          else
            out[static_cast<size_t>(row) * out_stride + column] =
                __bfloat16_as_ushort(__float2bfloat16_rn(accum[i][j][value]));
        }
      }
}

template <typename Out, int BM>
void launch(const uint8_t* a, const uint8_t* as, const uint8_t* w, const uint8_t* ws, Out* out,
            int m, int n, int k, cudaStream_t stream, size_t stride) {
  constexpr int smem = kStages * (BM + kBN) * kStride;
  const unsigned blocks = static_cast<unsigned>(((m + BM - 1) / BM) * ((n + kBN - 1) / kBN));
  mxfp8_gemm_kernel<Out, BM>
      <<<blocks, kThreads, smem, stream>>>(a, as, w, ws, out, m, n, k, stride);
  DGPP_CUDA_OK(cudaGetLastError());
}

template <typename Out>
void dispatch(const uint8_t* a, const uint8_t* as, const uint8_t* w, const uint8_t* ws, Out* out,
              int m, int n, int k, cudaStream_t stream, size_t stride) {
  if (m <= 0 || n <= 0) return;
  if (k <= 0 || k % 64)
    throw std::invalid_argument("MXFP8 GEMM: k must be a positive multiple of 64");
  if (stride == 0) stride = static_cast<size_t>(n);
  if (stride < static_cast<size_t>(n))
    throw std::invalid_argument("MXFP8 GEMM: output stride is too small");
  if (m <= 16)
    launch<Out, 16>(a, as, w, ws, out, m, n, k, stream, stride);
  else
    launch<Out, 64>(a, as, w, ws, out, m, n, k, stream, stride);
}
}  // namespace

void launch_mxfp8_quantize_rows(const uint16_t* x, size_t stride, int rows, int k, uint8_t* codes,
                                uint8_t* scales, cudaStream_t stream) {
  if (rows <= 0) return;
  if (k <= 0 || k % 32 || stride < static_cast<size_t>(k) || stride % 4)
    throw std::invalid_argument("MXFP8 quantize: requires 32-wide groups and aligned input rows");
  const dim3 grid(static_cast<unsigned>((k + 4 * kThreads - 1) / (4 * kThreads)),
                  static_cast<unsigned>(rows));
  mxfp8_quantize_kernel<<<grid, kThreads, 0, stream>>>(x, stride, k, codes, scales);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_mxfp8_gemm_bf16(const uint8_t* a, const uint8_t* as, const uint8_t* w,
                            const uint8_t* ws, uint16_t* out, int m, int n, int k,
                            cudaStream_t stream, size_t stride) {
  dispatch(a, as, w, ws, out, m, n, k, stream, stride);
}
void launch_mxfp8_gemm_f32(const uint8_t* a, const uint8_t* as, const uint8_t* w, const uint8_t* ws,
                           float* out, int m, int n, int k, cudaStream_t stream, size_t stride) {
  dispatch(a, as, w, ws, out, m, n, k, stream, stride);
}
}  // namespace dgpp
