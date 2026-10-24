#include "kernels/fp8_gemm.hpp"
#include "kernels/pdl.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <stdexcept>
#include <string>
#include <type_traits>

#include "common/cuda_check.hpp"

namespace dgpp {
namespace {

// ---- the quantizer ---------------------------------------------------------
constexpr int kQThreads = 256;  // eight warps: eight (row, group) pairs a block

__device__ __forceinline__ float bf16_bits_to_float_d(uint32_t bits16) {
  return __uint_as_float(bits16 << 16);
}

__global__ __launch_bounds__(kQThreads) void fp8_quantize_rows_kernel(const uint16_t* __restrict__ act,
                                                                     size_t act_stride, int m, int k,
                                                                     uint8_t* __restrict__ q,
                                                                     float* __restrict__ scales) {
  dgpp_pdl_wait();
  const int groups = k / 128;
  const int gid = blockIdx.x * (kQThreads / 32) + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (gid >= m * groups) return;
  const int row = gid / groups, g = gid - row * groups;
  const uint2 raw = *reinterpret_cast<const uint2*>(act + static_cast<size_t>(row) * act_stride + g * 128 + lane * 4);
  float x[4] = {bf16_bits_to_float_d(raw.x & 0xFFFFu), bf16_bits_to_float_d(raw.x >> 16),
                bf16_bits_to_float_d(raw.y & 0xFFFFu), bf16_bits_to_float_d(raw.y >> 16)};
  float amax = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
  const float scale = amax > 0.f ? amax / 448.f : 1.f;
  const __nv_fp8x2_storage_t lo =
      __nv_cvt_float2_to_fp8x2(make_float2(x[0] / scale, x[1] / scale), __NV_SATFINITE, __NV_E4M3);
  const __nv_fp8x2_storage_t hi =
      __nv_cvt_float2_to_fp8x2(make_float2(x[2] / scale, x[3] / scale), __NV_SATFINITE, __NV_E4M3);
  *reinterpret_cast<uint32_t*>(q + static_cast<size_t>(row) * k + g * 128 + lane * 4) =
      static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
  if (lane == 0) scales[static_cast<size_t>(row) * groups + g] = scale;
}

// The per-token quantizer: one block per row, each thread striding over
// the row in 8-element chunks (amax, a block reduction, then the codes).
constexpr int kTThreads = 256;
__global__ __launch_bounds__(kTThreads) void fp8_quantize_tokens_kernel(const uint16_t* __restrict__ act,
                                                                       size_t act_stride, int m, int k,
                                                                       uint8_t* __restrict__ q,
                                                                       float* __restrict__ scales) {
  dgpp_pdl_wait();
  const int row = blockIdx.x;
  if (row >= m) return;
  const uint16_t* src = act + static_cast<size_t>(row) * act_stride;
  float amax = 0.f;
  for (int c = threadIdx.x * 8; c < k; c += kTThreads * 8) {
    const uint4 raw = *reinterpret_cast<const uint4*>(src + c);
    const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for (int i = 0; i < 4; ++i)
      amax = fmaxf(amax, fmaxf(fabsf(bf16_bits_to_float_d(w[i] & 0xFFFFu)), fabsf(bf16_bits_to_float_d(w[i] >> 16))));
  }
  __shared__ float warp_max[kTThreads / 32];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
  if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x >> 5] = amax;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < kTThreads / 32 ? warp_max[threadIdx.x] : 0.f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xFFFFFFFFu, t, o));
    if (threadIdx.x == 0) warp_max[0] = t;
  }
  __syncthreads();
  const float row_amax = warp_max[0];
  const float scale = row_amax > 0.f ? row_amax / 448.f : 1.f;
  if (threadIdx.x == 0) scales[row] = scale;
  uint8_t* dst = q + static_cast<size_t>(row) * k;
  for (int c = threadIdx.x * 8; c < k; c += kTThreads * 8) {
    const uint4 raw = *reinterpret_cast<const uint4*>(src + c);
    const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
    uint32_t packed[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(
          make_float2(bf16_bits_to_float_d(w[2 * i] & 0xFFFFu) / scale, bf16_bits_to_float_d(w[2 * i] >> 16) / scale),
          __NV_SATFINITE, __NV_E4M3);
      const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(
          make_float2(bf16_bits_to_float_d(w[2 * i + 1] & 0xFFFFu) / scale, bf16_bits_to_float_d(w[2 * i + 1] >> 16) / scale),
          __NV_SATFINITE, __NV_E4M3);
      packed[i] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
    }
    *reinterpret_cast<uint2*>(dst + c) = make_uint2(packed[0], packed[1]);
  }
}

// The per-token quantizer fused behind the Qwen3.5 SwiGLU (qwen35_swiglu_kernel's
// ops: silu as g / (1 + exp(-g)), one bf16 rounding of the product): the row's
// amax in a first pass, the codes in a second, the product recomputed rather
// than written and re-read as bf16 (the down projection's 35 KB rows).
__device__ __forceinline__ float swiglu_qwen_d(uint32_t gb, uint32_t ub) {
  const float g = bf16_bits_to_float_d(gb), u = bf16_bits_to_float_d(ub);
  const float s = g / (1.0f + expf(-g));
  return __bfloat162float(__float2bfloat16_rn(s * u));
}
__global__ __launch_bounds__(kTThreads) void fp8_swiglu_quantize_tokens_kernel(const uint16_t* __restrict__ gate,
                                                                              const uint16_t* __restrict__ up,
                                                                              size_t stride, int m, int k,
                                                                              uint8_t* __restrict__ q,
                                                                              float* __restrict__ scales) {
  dgpp_pdl_wait();
  const int row = blockIdx.x;
  if (row >= m) return;
  const uint16_t* gr = gate + static_cast<size_t>(row) * stride;
  const uint16_t* ur = up + static_cast<size_t>(row) * stride;
  float amax = 0.f;
  for (int c = threadIdx.x * 8; c < k; c += kTThreads * 8) {
    const uint4 g4 = *reinterpret_cast<const uint4*>(gr + c), u4 = *reinterpret_cast<const uint4*>(ur + c);
    const uint32_t g[4] = {g4.x, g4.y, g4.z, g4.w}, u[4] = {u4.x, u4.y, u4.z, u4.w};
#pragma unroll
    for (int i = 0; i < 4; ++i)
      amax = fmaxf(amax, fmaxf(fabsf(swiglu_qwen_d(g[i] & 0xFFFFu, u[i] & 0xFFFFu)), fabsf(swiglu_qwen_d(g[i] >> 16, u[i] >> 16))));
  }
  __shared__ float warp_max[kTThreads / 32];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
  if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x >> 5] = amax;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < kTThreads / 32 ? warp_max[threadIdx.x] : 0.f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xFFFFFFFFu, t, o));
    if (threadIdx.x == 0) warp_max[0] = t;
  }
  __syncthreads();
  const float row_amax = warp_max[0];
  const float scale = row_amax > 0.f ? row_amax / 448.f : 1.f;
  if (threadIdx.x == 0) scales[row] = scale;
  uint8_t* dst = q + static_cast<size_t>(row) * k;
  for (int c = threadIdx.x * 8; c < k; c += kTThreads * 8) {
    const uint4 g4 = *reinterpret_cast<const uint4*>(gr + c), u4 = *reinterpret_cast<const uint4*>(ur + c);
    const uint32_t g[4] = {g4.x, g4.y, g4.z, g4.w}, u[4] = {u4.x, u4.y, u4.z, u4.w};
    uint32_t packed[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(
          make_float2(swiglu_qwen_d(g[2 * i] & 0xFFFFu, u[2 * i] & 0xFFFFu) / scale, swiglu_qwen_d(g[2 * i] >> 16, u[2 * i] >> 16) / scale),
          __NV_SATFINITE, __NV_E4M3);
      const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(
          make_float2(swiglu_qwen_d(g[2 * i + 1] & 0xFFFFu, u[2 * i + 1] & 0xFFFFu) / scale, swiglu_qwen_d(g[2 * i + 1] >> 16, u[2 * i + 1] >> 16) / scale),
          __NV_SATFINITE, __NV_E4M3);
      packed[i] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
    }
    *reinterpret_cast<uint2*>(dst + c) = make_uint2(packed[0], packed[1]);
  }
}

// ---- the GEMM --------------------------------------------------------------
// 128 x 128 x 64 block tiles on eight warps (2 x 4: a 64 x 32 warp tile of
// four m16 by four n8 mma tiles), a four-stage cp.async pipeline of the
// e4m3 rows (80 KB: one block per SM), both operands k-contiguous in smem
// at an 80-byte row stride (ldmatrix conflict-free), mma m16n8k32 e4m3 into
// a per-group fp32 partial promoted every two k-steps by the row's and the
// column's scale.
constexpr int kBM = 128, kBN = 128, kBK = 64, kGroup = 128;
constexpr int kThreads = 256;
constexpr int kStages = 4;
constexpr int kStride = kBK + 16;                   // smem row bytes
constexpr int kTileBytes = kBM * kStride;           // one operand, one stage
constexpr int kSmem = kStages * 2 * kTileBytes;     // 81,920
static_assert(kBN == kBM, "the copy map covers both operands the same way");
static_assert(kGroup == 2 * kBK, "a scale group is two k-steps");

__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::); }
template <int N>
__device__ __forceinline__ void wait_pending() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(address));
}

// kChannel (2026-10-15): one scale per activation row and per weight row,
// so the e4m3 products accumulate in fp32 over the whole k and the two
// scales apply once in the epilogue (no per-group promotion; k % 64).
template <typename OutT, bool kChannel>
__global__ __launch_bounds__(kThreads, 1) void fp8_gemm_kernel(const uint8_t* __restrict__ a,
                                                              const float* __restrict__ a_scales,
                                                              const uint8_t* __restrict__ w,
                                                              const float* __restrict__ w_scales, int sbr,
                                                              OutT* __restrict__ out, int m, int n, int k,
                                                              size_t out_stride) {
  dgpp_pdl_wait();
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sa = smem;
  uint8_t* sb = smem + kStages * kTileBytes;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  // The tile order: groups of kGroupM m-tiles walked n-tile by n-tile (a
  // wave's blocks share kGroupM activation tiles from L2 while the weight
  // tiles stream once per group; the plain n-fastest raster streamed a
  // [10240 x 2560] weight once per m-tile — 4.3 ms against 2.2 for cuBLAS).
  constexpr int kGroupM = 8;
  const int m_tiles = (m + kBM - 1) / kBM, n_tiles = (n + kBN - 1) / kBN;
  const int bid = static_cast<int>(blockIdx.x);
  const int group = bid / (kGroupM * n_tiles);
  const int first_m = group * kGroupM;
  const int group_rows = min(kGroupM, m_tiles - first_m);
  const int in_group = bid - group * kGroupM * n_tiles;
  const int m0 = (first_m + in_group % group_rows) * kBM;
  const int n0 = (in_group / group_rows) * kBN;
  const int row_base = (warp / 4) * 64, col_base = (warp % 4) * 32;
  const int steps = k / kBK;
  const int scale_cols = k / kGroup;
  // The copy map: thread -> rows tid / 4 and tid / 4 + 64, 16-byte chunk tid % 4.
  const int crow = tid / 4, cchunk = (tid % 4) * 16;
  const bool av0 = m0 + crow < m, av1 = m0 + crow + 64 < m;
  const bool bv0 = n0 + crow < n, bv1 = n0 + crow + 64 < n;
  const uint8_t* a0 = a + static_cast<size_t>(av0 ? m0 + crow : 0) * k + cchunk;
  const uint8_t* a1 = a + static_cast<size_t>(av1 ? m0 + crow + 64 : 0) * k + cchunk;
  const uint8_t* b0 = w + static_cast<size_t>(bv0 ? n0 + crow : 0) * k + cchunk;
  const uint8_t* b1 = w + static_cast<size_t>(bv1 ? n0 + crow + 64 : 0) * k + cchunk;
  auto issue = [&](int step, int slot) {
    uint8_t* as = sa + slot * kTileBytes;
    uint8_t* bs = sb + slot * kTileBytes;
    const int koff = step * kBK;
    cp16(as + crow * kStride + cchunk, a0 + koff, av0);
    cp16(as + (crow + 64) * kStride + cchunk, a1 + koff, av1);
    cp16(bs + crow * kStride + cchunk, b0 + koff, bv0);
    cp16(bs + (crow + 64) * kStride + cchunk, b1 + koff, bv1);
  };
  const int r = lane / 4, cc = (lane % 4) * 2;
  // The scale rows / columns this thread's accumulators own.
  int arow[4][2];
  bool arow_ok[4][2];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = m0 + row_base + i * 16 + r + h * 8;
      arow_ok[i][h] = row < m;
      arow[i][h] = arow_ok[i][h] ? row : 0;
    }
  int wcol[4][2];
  bool wcol_ok[4][2];
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int v = 0; v < 2; ++v) {
      const int col = n0 + col_base + j * 8 + cc + v;
      wcol_ok[j][v] = col < n;
      wcol[j][v] = wcol_ok[j][v] ? col / sbr : 0;
    }

  float acc[4][4][4] = {};
  float partial[4][4][4] = {};
#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < steps) issue(s, s);
    commit();
  }
  for (int step = 0; step < steps; ++step) {
    wait_pending<kStages - 2>();
    __syncthreads();
    const int nxt = step + kStages - 1;
    if (nxt < steps) issue(nxt, nxt % kStages);
    commit();
    const uint8_t* as = sa + (step % kStages) * kTileBytes;
    const uint8_t* bs = sb + (step % kStages) * kTileBytes;
#pragma unroll
    for (int kk = 0; kk < kBK; kk += 32) {
      uint32_t af[4][4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        ldsm_x4(af[i], as + (row_base + i * 16 + lane % 16) * kStride + kk + (lane / 16) * 16);
      uint32_t bf[4][2];
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        // Matrices 0/1 = n8 tile 2jj at k kk..kk+15 / kk+16..kk+31, 2/3 = tile 2jj+1.
        const int tile = col_base + jj * 16 + (lane / 16) * 8 + lane % 8;
        const int kb = kk + ((lane / 8) % 2) * 16;
        uint32_t q4[4];
        ldsm_x4(q4, bs + tile * kStride + kb);
        bf[jj * 2][0] = q4[0];
        bf[jj * 2][1] = q4[1];
        bf[jj * 2 + 1][0] = q4[2];
        bf[jj * 2 + 1][1] = q4[3];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          asm volatile(
              "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(partial[i][j][0]), "+f"(partial[i][j][1]), "+f"(partial[i][j][2]), "+f"(partial[i][j][3])
              : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(bf[j][0]), "r"(bf[j][1]));
    }
    if constexpr (!kChannel) if (step % 2 == 1) {
      // A 128-wide group is complete: promote by the row and column scales.
      const int g = step / 2;
      float as_[4][2], ws_[4][2];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
          as_[i][h] = arow_ok[i][h] ? a_scales[static_cast<size_t>(arow[i][h]) * scale_cols + g] : 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int v = 0; v < 2; ++v)
          ws_[j][v] = wcol_ok[j][v] ? w_scales[static_cast<size_t>(wcol[j][v]) * scale_cols + g] : 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int v = 0; v < 4; ++v) {
            acc[i][j][v] = __fmaf_rn(as_[i][v / 2] * ws_[j][v % 2], partial[i][j][v], acc[i][j][v]);
            partial[i][j][v] = 0.f;
          }
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const int row = m0 + row_base + i * 16 + r + (v / 2) * 8;
        const int col = n0 + col_base + j * 8 + cc + (v % 2);
        if (row < m && col < n) {
          const size_t index = static_cast<size_t>(row) * out_stride + col;
          float val;
          if constexpr (kChannel)
            val = (partial[i][j][v] * a_scales[row]) * w_scales[col];
          else
            val = acc[i][j][v];
          if constexpr (std::is_same_v<OutT, float>)
            out[index] = val;
          else
            out[index] = __bfloat16_as_ushort(__float2bfloat16_rn(val));
        }
      }
}

template <typename OutT, bool kChannel>
void launch(const uint8_t* a, const float* a_scales, const uint8_t* w, const float* w_scales, int sbr, OutT* out,
            int m, int n, int k, cudaStream_t stream, size_t out_stride) {
  if (m <= 0 || n <= 0) return;
  if (k <= 0 || k % (kChannel ? kBK : kGroup) != 0)
    throw std::invalid_argument(kChannel ? "fp8 gemm (channel): k must be a positive multiple of 64"
                                         : "fp8 gemm: k must be a positive multiple of 128");
  if (sbr <= 0) throw std::invalid_argument("fp8 gemm: the weight scale block rows must be positive");
  if (out_stride == 0) out_stride = static_cast<size_t>(n);
  static bool attr_set = false;  // once per instantiation
  if (!attr_set) {
    DGPP_CUDA_OK(cudaFuncSetAttribute(fp8_gemm_kernel<OutT, kChannel>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    attr_set = true;
  }
  const dim3 grid(static_cast<unsigned>(((n + kBN - 1) / kBN) * ((m + kBM - 1) / kBM)));
  fp8_gemm_kernel<OutT, kChannel><<<grid, kThreads, kSmem, stream>>>(a, a_scales, w, w_scales, sbr, out, m, n, k, out_stride);
  DGPP_CUDA_OK(cudaGetLastError());
}


// ---- the channel form on TMA boxes ---------------------------------------------
// 128 x 256 x 128 a stage (A 16 KB + W 32 KB, SWIZZLE_128B: 16-byte chunk c of
// row r at chunk c ^ (r & 7)), two stages, one mbarrier a stage, eight
// 64 x 64 warp tiles (2 x 4), the swizzle folded into per-lane ldmatrix bases.
constexpr int kTBN = 256, kTBK = 128, kTStages = 2;
constexpr int kTABytes = kBM * kTBK, kTBBytes = kTBN * kTBK, kTStageBytes = kTABytes + kTBBytes;
constexpr size_t kTSmem = size_t(kTStages) * kTStageBytes + 1024;  // + the 1 KB alignment slack and barriers

__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(static_cast<unsigned>(__cvta_generic_to_shared(bar))), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(static_cast<unsigned>(__cvta_generic_to_shared(bar))), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned parity) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(bar));
  unsigned done = 0;
  do {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                 : "=r"(done) : "r"(a), "r"(parity) : "memory");
  } while (!done);
}
__device__ __forceinline__ void tma_box_2d(unsigned dst, const CUtensorMap* map, int x, int y, uint64_t* bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               ::"r"(dst), "l"(map), "r"(x), "r"(y), "r"(static_cast<unsigned>(__cvta_generic_to_shared(bar))) : "memory");
}
__device__ __forceinline__ void ldsm_x4_a(uint32_t (&r)[4], unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

template <typename OutT>
__global__ __launch_bounds__(kThreads, 1) void fp8_gemm_channel_tma_kernel(const __grid_constant__ CUtensorMap amap,
                                                                          const __grid_constant__ CUtensorMap wmap,
                                                                          const float* __restrict__ a_scales,
                                                                          const float* __restrict__ w_scales,
                                                                          OutT* __restrict__ out, int m, int n, int k,
                                                                          size_t out_stride) {
  dgpp_pdl_wait();
  extern __shared__ __align__(1024) uint8_t smraw[];
  const unsigned sbase = static_cast<unsigned>(__cvta_generic_to_shared(smraw));
  uint8_t* sm = smraw + ((1024 - (sbase & 1023)) & 1023);
  const unsigned s0 = static_cast<unsigned>(__cvta_generic_to_shared(sm));
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + kTStages * kTStageBytes);
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  constexpr int kGroupM = 8;
  const int m_tiles = (m + kBM - 1) / kBM, n_tiles = (n + kTBN - 1) / kTBN;
  const int bid = static_cast<int>(blockIdx.x);
  const int group = bid / (kGroupM * n_tiles);
  const int first_m = group * kGroupM;
  const int group_rows = min(kGroupM, m_tiles - first_m);
  const int in_group = bid - group * kGroupM * n_tiles;
  const int m0 = (first_m + in_group % group_rows) * kBM;
  const int n0 = (in_group / group_rows) * kTBN;
  const int row_base = (warp / 4) * 64, col_base = (warp % 4) * 64;
  const int steps = k / kTBK;
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < kTStages; ++s) mbar_init(&full[s], 1);
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  auto stage = [&](int step, int slot) {
    if (tid != 0) return;
    const unsigned dst = s0 + slot * kTStageBytes;
    mbar_expect_tx(&full[slot], static_cast<unsigned>(kTStageBytes));
    tma_box_2d(dst, &amap, step * kTBK, m0, &full[slot]);
    tma_box_2d(dst + kTABytes, &wmap, step * kTBK, n0, &full[slot]);
  };
  // Per-lane ldmatrix bases for slot 0: A rows row_base + 16 i + lane % 16 at
  // chunk kk/16 + lane/16; W rows col_base + 16 jj + 8 (lane/16) + lane % 8 at
  // chunk kk/16 + (lane/8) % 2. The XOR operand (row & 7) is lane & 7 for all.
  unsigned lba[4], lbb[4];
  {
    const int r7 = lane & 7;
#pragma unroll
    for (int i = 0; i < 4; ++i) lba[i] = s0 + (row_base + i * 16 + (lane & 15)) * kTBK + (((lane >> 4) ^ r7) << 4);
#pragma unroll
    for (int jj = 0; jj < 4; ++jj)
      lbb[jj] = s0 + kTABytes + (col_base + jj * 16 + (lane >> 4) * 8 + r7) * kTBK + ((((lane >> 3) & 1) ^ r7) << 4);
  }
  float acc[4][8][4] = {};
  if (steps > 0) stage(0, 0);
  int slot = 0;
  unsigned parity = 0;
  for (int step = 0; step < steps; ++step) {
    mbar_wait(&full[slot], parity);
    __syncthreads();  // every warp is done with step - 1: its slot is free
    if (step + 1 < steps) stage(step + 1, slot ^ 1);
    const unsigned so = static_cast<unsigned>(slot) * kTStageBytes;
#pragma unroll
    for (int kk = 0; kk < kTBK; kk += 32) {
      const unsigned x = static_cast<unsigned>(kk / 16) << 4;
      uint32_t af[4][4];
#pragma unroll
      for (int i = 0; i < 4; ++i) ldsm_x4_a(af[i], (lba[i] ^ x) + so);
      uint32_t bf[8][2];
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        uint32_t q4[4];
        ldsm_x4_a(q4, (lbb[jj] ^ x) + so);
        bf[jj * 2][0] = q4[0];
        bf[jj * 2][1] = q4[1];
        bf[jj * 2 + 1][0] = q4[2];
        bf[jj * 2 + 1][1] = q4[3];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
          asm volatile(
              "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
              : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(bf[j][0]), "r"(bf[j][1]));
    }
    if (++slot == kTStages) {
      slot = 0;
      parity ^= 1u;
    }
  }
  const int r = lane / 4, cc = (lane % 4) * 2;
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = m0 + row_base + i * 16 + r + h * 8;
        const int col = n0 + col_base + j * 8 + cc;
        if (row >= m) continue;
        const size_t index = static_cast<size_t>(row) * out_stride + col;
        const float sa = a_scales[row];
        if constexpr (!std::is_same_v<OutT, float>) {
          if (col + 1 < n && (out_stride % 2) == 0) {
            const float v0 = (acc[i][j][2 * h] * sa) * w_scales[col], v1 = (acc[i][j][2 * h + 1] * sa) * w_scales[col + 1];
            *reinterpret_cast<uint32_t*>(out + index) =
                static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(v0))) |
                (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(v1))) << 16);
            continue;
          }
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          if (col + e >= n) continue;
          const float val = (acc[i][j][2 * h + e] * sa) * w_scales[col + e];
          if constexpr (std::is_same_v<OutT, float>)
            out[index + e] = val;
          else
            out[index + e] = __bfloat16_as_ushort(__float2bfloat16_rn(val));
        }
      }
}

template <typename OutT>
void launch_tma(const CUtensorMap& amap, const CUtensorMap& wmap, const float* a_scales, const float* w_scales, OutT* out,
                int m, int n, int k, cudaStream_t stream, size_t out_stride) {
  if (m <= 0 || n <= 0) return;
  if (k <= 0 || k % kTBK != 0) throw std::invalid_argument("fp8 gemm (channel tma): k must be a positive multiple of 128");
  if (out_stride == 0) out_stride = static_cast<size_t>(n);
  static bool attr_set = false;  // once per OutT
  if (!attr_set) {
    DGPP_CUDA_OK(cudaFuncSetAttribute(fp8_gemm_channel_tma_kernel<OutT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(kTSmem)));
    attr_set = true;
  }
  const dim3 grid(static_cast<unsigned>(((n + kTBN - 1) / kTBN) * ((m + kBM - 1) / kBM)));
  fp8_gemm_channel_tma_kernel<OutT><<<grid, kThreads, kTSmem, stream>>>(amap, wmap, a_scales, w_scales, out, m, n, k, out_stride);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace

bool fp8_channel_tma_supported(int k) { return k > 0 && k % kTBK == 0; }

CUtensorMap fp8_channel_tma_map(const uint8_t* base, int64_t rows, int64_t k, int box_rows) {
  if (!fp8_channel_tma_supported(static_cast<int>(k)) || (reinterpret_cast<uintptr_t>(base) % 16) != 0)
    throw std::invalid_argument("fp8_channel_tma_map: k must be a multiple of 128 and the matrix 16-byte aligned");
  if (box_rows != 128 && box_rows != 256) throw std::invalid_argument("fp8_channel_tma_map: box rows must be 128 or 256");
  CUtensorMap map{};
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(k), static_cast<cuuint64_t>(rows)};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(k)};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(kTBK), static_cast<cuuint32_t>(box_rows)};
  const cuuint32_t estr[2] = {1, 1};
  const CUresult r = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<uint8_t*>(base), dims,
                                            strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                            CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS)
    throw std::runtime_error("fp8_channel_tma_map: cuTensorMapEncodeTiled failed (" + std::to_string(static_cast<int>(r)) + ")");
  return map;
}
void launch_fp8_gemm_channel_tma_bf16(const CUtensorMap& amap, const CUtensorMap& wmap, const float* a_scales,
                                      const float* w_scales, uint16_t* out, int m, int n, int k, cudaStream_t stream,
                                      size_t out_stride) {
  launch_tma<uint16_t>(amap, wmap, a_scales, w_scales, out, m, n, k, stream, out_stride);
}
void launch_fp8_gemm_channel_tma_f32(const CUtensorMap& amap, const CUtensorMap& wmap, const float* a_scales,
                                     const float* w_scales, float* out, int m, int n, int k, cudaStream_t stream,
                                     size_t out_stride) {
  launch_tma<float>(amap, wmap, a_scales, w_scales, out, m, n, k, stream, out_stride);
}

void launch_fp8_quantize_rows(const uint16_t* act, size_t act_stride, int m, int k, uint8_t* q, float* scales,
                              cudaStream_t stream) {
  if (m <= 0) return;
  if (k <= 0 || k % 128 != 0) throw std::invalid_argument("fp8 quantize: k must be a positive multiple of 128");
  if (act_stride % 4 != 0 || (reinterpret_cast<uintptr_t>(act) % 8) != 0)
    throw std::invalid_argument("fp8 quantize: the activation rows must be 8-byte aligned");
  const int pairs = m * (k / 128);
  const int blocks = (pairs + kQThreads / 32 - 1) / (kQThreads / 32);
  fp8_quantize_rows_kernel<<<blocks, kQThreads, 0, stream>>>(act, act_stride, m, k, q, scales);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_fp8_gemm_bf16(const uint8_t* a, const float* a_scales, const uint8_t* w, const float* w_scales,
                          int w_scale_block_rows, uint16_t* out, int m, int n, int k, cudaStream_t stream,
                          size_t out_stride) {
  launch<uint16_t, false>(a, a_scales, w, w_scales, w_scale_block_rows, out, m, n, k, stream, out_stride);
}
void launch_fp8_gemm_f32(const uint8_t* a, const float* a_scales, const uint8_t* w, const float* w_scales,
                         int w_scale_block_rows, float* out, int m, int n, int k, cudaStream_t stream,
                         size_t out_stride) {
  launch<float, false>(a, a_scales, w, w_scales, w_scale_block_rows, out, m, n, k, stream, out_stride);
}

void launch_fp8_quantize_tokens(const uint16_t* act, size_t act_stride, int m, int k, uint8_t* q, float* scales,
                                cudaStream_t stream) {
  if (m <= 0) return;
  if (k <= 0 || k % 8 != 0) throw std::invalid_argument("fp8 quantize tokens: k must be a positive multiple of 8");
  if (act_stride % 8 != 0 || (reinterpret_cast<uintptr_t>(act) % 16) != 0)
    throw std::invalid_argument("fp8 quantize tokens: rows must be 16-byte aligned (act_stride % 8 == 0)");
  fp8_quantize_tokens_kernel<<<m, kTThreads, 0, stream>>>(act, act_stride, m, k, q, scales);
  DGPP_CUDA_OK(cudaGetLastError());
}
void launch_fp8_swiglu_quantize_tokens(const uint16_t* gate, const uint16_t* up, size_t stride, int m, int k, uint8_t* q,
                                       float* scales, cudaStream_t stream) {
  if (m <= 0) return;
  if (k <= 0 || k % 8 != 0) throw std::invalid_argument("fp8 swiglu quantize: k must be a positive multiple of 8");
  if (stride % 8 != 0 || (reinterpret_cast<uintptr_t>(gate) % 16) != 0 || (reinterpret_cast<uintptr_t>(up) % 16) != 0)
    throw std::invalid_argument("fp8 swiglu quantize: rows must be 16-byte aligned (stride % 8 == 0)");
  fp8_swiglu_quantize_tokens_kernel<<<m, kTThreads, 0, stream>>>(gate, up, stride, m, k, q, scales);
  DGPP_CUDA_OK(cudaGetLastError());
}
void launch_fp8_gemm_channel_bf16(const uint8_t* a, const float* a_scales, const uint8_t* w, const float* w_scales,
                                  uint16_t* out, int m, int n, int k, cudaStream_t stream, size_t out_stride) {
  launch<uint16_t, true>(a, a_scales, w, w_scales, 1, out, m, n, k, stream, out_stride);
}
void launch_fp8_gemm_channel_f32(const uint8_t* a, const float* a_scales, const uint8_t* w, const float* w_scales,
                                 float* out, int m, int n, int k, cudaStream_t stream, size_t out_stride) {
  launch<float, true>(a, a_scales, w, w_scales, 1, out, m, n, k, stream, out_stride);
}

}  // namespace dgpp
