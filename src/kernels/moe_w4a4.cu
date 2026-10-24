// W4A4 NVFP4 grouped MoE GEMM on GB10's native block-scaled tensor cores.
//
// Why (2026-09-23): DGPP's moe_grouped_mma_fp4_ldm_kernel runs the NVFP4
// experts W4A16 (dequantize the e2m1 codes to bf16, HMMA m16n8k16), while
// prod SGLang runs them W4A4 through flashinfer_cutlass. sm_121a has
// mma.sync kind::mxf4nvf4.block_scale (OMMA.SF.16864): e2m1 x e2m1 with one
// ue4m3 scale per 16 elements, i.e. NVFP4 on both operands, at 4x the bf16
// instruction rate (bench/micro/mma_peak.cu: 194.6 vs 48.8 TFLOPS). The
// checkpoint is authored for W4A4 (73,728 expert input_scale tensors).
//
// Activations are quantized per row with a global gs -- the layer's static
// checkpoint input_scale (max over its experts, as SGLang's flashinfer_cutlass
// MoE does; values past 6 * 448 * gs clip, which the NVFP4 experts were
// calibrated with) or, when none is given, a dynamic amax / (6 * 448) -- and
// one e4m3 scale per 16 elements, bs = e4m3(amax16 / (6 * gs)); the codes are
// x / (bs * gs) rounded to e2m1 (cvt.rn.satfinite). x ~= e2m1 * bs * gs. The weights are DGPP's NVFP4 views unchanged: e2m1
// codes [n, k/2], e4m3 scales [n, k/16], value = e2m1 * s / g. So
//   out[row, col] = acc(row, col) * gs[row] / g.
//
// Fragment layouts (m16n8k64, PTX ISA 9.7.14.6; the same as upstream SGLang's
// validated sm120 kernel, jit/csrc/moe/nvfp4_moe_sm120.cuh): lane = 4q + t.
//   A (row-major 16 x 64 e2m1): a0 = row q bytes [4t, 4t+4), a1 = row q+8,
//      a2 = row q bytes [16 + 4t, ...), a3 = row q+8 likewise.
//   B (col-major 64 x 8): b0 = column q bytes [4t, 4t+4), b1 = +16 bytes.
//   sfa: the four ue4m3 scales (k blocks 0..3) of row q + 8 * (lane & 1);
//   sfb: the four scales of column q. Selectors zero (scale_vec::4X).
//   C: c0, c1 = row q cols 2t, 2t+1; c2, c3 = row q+8.
#include "kernels/moe_w4a4.hpp"
#include "kernels/pdl.cuh"

#include <cstdlib>
#include <stdexcept>
#include <string>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "common/cuda_check.hpp"

namespace dgpp {
namespace {

namespace w4a4 {
constexpr int BN = 128, BK = 128;  // BK in elements: 64 code bytes per row
constexpr int kThreads = 256;
constexpr int kRowBytes = 80;               // 64 code bytes + 16 pad: conflict-free fragment reads
constexpr int kAScaleBytes = 8;             // 8 ue4m3 scales per row per stage
constexpr size_t kBCodes = size_t(BN) * kRowBytes;       // 10,240
constexpr size_t kBScales = size_t(BN) * kAScaleBytes;   // 1,024
// The row tile (2026-09-23): 64 or 128 rows. Stages: 3 at BM 64 (50,688 B)
// fit ONE block per GB10 SM; 2 fit two (DGPP_W4A4_STAGES, default 2).
template <int kBM>
struct Tile {
  static constexpr size_t a_codes = size_t(kBM) * kRowBytes;
  static constexpr size_t a_scales = size_t(kBM) * kAScaleBytes;
  static constexpr size_t slot = a_codes + kBCodes + a_scales + kBScales;
  static constexpr int wm_warps = kBM / 32, wn_warps = 8 / wm_warps;
  static constexpr int ncols = BN / wn_warps, nt = ncols / 8;
};
template <int S, int kBM>
constexpr size_t smem_bytes_bm() { return S * Tile<kBM>::slot; }
}  // namespace w4a4

__device__ __forceinline__ void cp_async(void* dst, const void* src, int bytes, int src_bytes) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  if (bytes == 16)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(src_bytes));
  else if (bytes == 8)
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(d), "l"(src), "r"(src_bytes));
  else
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(d), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem_ptr) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

__device__ __forceinline__ void mma_nvfp4(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                          uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
      "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
      "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa),
        "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)), "r"(sfb),
        "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)));
}

__device__ __forceinline__ uint32_t f32x8_to_e2m1(const float* v) {
  uint32_t packed;
  asm volatile(
      "{\n.reg .b8 b0, b1, b2, b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n}"
      : "=r"(packed)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  return packed;
}

// kSwiglu (2026-09-23): the row is the MoE's gate/up pair, activated here --
// moe_swiglu_clamp_kernel's math and its two bf16 roundings, op for op -- so
// the down projection's NVFP4 activations come straight from gate/up without
// the bf16 act buffer's write and re-read (bitwise the two-kernel chain).
// The Qwen3.5 dense MLP's form: silu as g / (1 + exp(-g)) and one bf16 rounding
// of the product (qwen35_swiglu_kernel, models/qwen/model35_kernels.cu), op for op.
__device__ __forceinline__ float swiglu_qwen_bf16(uint16_t gb, uint16_t ub) {
  const float g = __uint_as_float(static_cast<uint32_t>(gb) << 16);
  const float u = __uint_as_float(static_cast<uint32_t>(ub) << 16);
  const float s = g / (1.0f + expf(-g));
  return __bfloat162float(__float2bfloat16_rn(s * u));
}
__device__ __forceinline__ float swiglu_bf16(uint16_t gb, uint16_t ub, float limit) {
  float gf = __uint_as_float(static_cast<uint32_t>(gb) << 16);
  float uf = __uint_as_float(static_cast<uint32_t>(ub) << 16);
  if (gf > limit) gf = limit;  // gate: no lower clamp (the reference's asymmetry)
  uf = fminf(fmaxf(uf, -limit), limit);
  const float silu = gf * (1.0f / (1.0f + expf(-gf)));
  const float t = __bfloat162float(__float2bfloat16_rn(silu));  // rounding 1 (silu)
  return __bfloat162float(__float2bfloat16_rn(t * uf));          // rounding 2 (product)
}

// One block per row; one thread per 16-element group, up to kGroupsPerThread
// of them strided by the block (k <= 16 * 1024 * kGroupsPerThread: the
// Qwen3.8-27B down projection's 17408, 2026-10-15); each group's math is
// the single-group kernel's, so a row's codes do not depend on the mapping.
constexpr int kGroupsPerThread = 2;
template <bool kSwiglu, bool kQwen = false>
__global__ void quantize_rows_nvfp4_kernel(const uint16_t* __restrict__ x, size_t x_stride, int rows, int k,
                                           uint8_t* __restrict__ codes, size_t code_stride,
                                           uint8_t* __restrict__ scales, size_t scale_stride,
                                           float* __restrict__ gs, float static_gs,
                                           const float* __restrict__ static_gs_dev,
                                           const uint16_t* __restrict__ up = nullptr, float limit = 0.f) {
  dgpp_pdl_wait();
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int groups = k / 16;
  float v[kGroupsPerThread][16];
  float gmax[kGroupsPerThread];
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < kGroupsPerThread; ++j) {
    const int g = threadIdx.x + j * blockDim.x;
    gmax[j] = 0.f;
    if (g >= groups) continue;
    const uint4* src = reinterpret_cast<const uint4*>(x + static_cast<size_t>(row) * x_stride + g * 16);
    const uint4 w0 = src[0], w1 = src[1];
    const uint32_t w[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
    if constexpr (kSwiglu) {
      const uint4* us = reinterpret_cast<const uint4*>(up + static_cast<size_t>(row) * x_stride + g * 16);
      const uint4 u0 = us[0], u1 = us[1];
      const uint32_t u[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if constexpr (kQwen) {
          v[j][2 * i] = swiglu_qwen_bf16(static_cast<uint16_t>(w[i] & 0xFFFFu), static_cast<uint16_t>(u[i] & 0xFFFFu));
          v[j][2 * i + 1] = swiglu_qwen_bf16(static_cast<uint16_t>(w[i] >> 16), static_cast<uint16_t>(u[i] >> 16));
        } else {
          v[j][2 * i] = swiglu_bf16(static_cast<uint16_t>(w[i] & 0xFFFFu), static_cast<uint16_t>(u[i] & 0xFFFFu), limit);
          v[j][2 * i + 1] = swiglu_bf16(static_cast<uint16_t>(w[i] >> 16), static_cast<uint16_t>(u[i] >> 16), limit);
        }
        gmax[j] = fmaxf(gmax[j], fmaxf(fabsf(v[j][2 * i]), fabsf(v[j][2 * i + 1])));
      }
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        v[j][2 * i] = __uint_as_float(w[i] << 16);
        v[j][2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
        gmax[j] = fmaxf(gmax[j], fmaxf(fabsf(v[j][2 * i]), fabsf(v[j][2 * i + 1])));
      }
    }
    amax = fmaxf(amax, gmax[j]);
  }
  // The row's amax: warp shuffles, then one word per warp.
  __shared__ float warp_max[32];
  float m = amax;
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x < 32) {
    const int nw = (blockDim.x + 31) >> 5;
    float t = threadIdx.x < nw ? warp_max[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, o));
    if (threadIdx.x == 0) warp_max[0] = t;
  }
  __syncthreads();
  const float row_amax = warp_max[0];
  // A static global (the checkpoint's input_scale) clips what exceeds
  // 6 * 448 * gs: the e4m3 block scale and the e2m1 codes both saturate.
  // The static global comes from device memory when given (the layer image's
  // copy of the checkpoint's input_scale), else from the argument.
  const float sg = static_gs_dev != nullptr ? *static_gs_dev : static_gs;
  const float global = sg > 0.f ? sg : row_amax > 0.f ? row_amax / (6.f * 448.f) : 1.f;
  if (threadIdx.x == 0) gs[row] = global;
  uint8_t* scale_row = scales + static_cast<size_t>(row) * scale_stride;
#pragma unroll
  for (int j = 0; j < kGroupsPerThread; ++j) {
    const int g = threadIdx.x + j * blockDim.x;
    if (g >= groups) break;
    uint8_t* code_row = codes + static_cast<size_t>(row) * code_stride + g * 8;
    const __nv_fp8_e4m3 s(gmax[j] / (6.f * global));
    const float bs = static_cast<float>(s);
    if (gmax[j] == 0.f || bs == 0.f) {
      scale_row[g] = 0;
      *reinterpret_cast<uint2*>(code_row) = make_uint2(0u, 0u);
      continue;
    }
    scale_row[g] = s.__x;
    const float q = 1.f / (bs * global);
#pragma unroll
    for (int i = 0; i < 16; ++i) v[j][i] *= q;
    *reinterpret_cast<uint2*>(code_row) = make_uint2(f32x8_to_e2m1(v[j]), f32x8_to_e2m1(v[j] + 8));
  }
}

__device__ __forceinline__ void store_out(uint16_t* p, float v) {
  const __nv_bfloat16 b = __float2bfloat16_rn(v);
  *p = *reinterpret_cast<const uint16_t*>(&b);
}
__device__ __forceinline__ void store_out(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_pair(uint16_t* p, float a, float b) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  *reinterpret_cast<__nv_bfloat162*>(p) = v;
}
__device__ __forceinline__ void store_pair(float* p, float a, float b) { *reinterpret_cast<float2*>(p) = make_float2(a, b); }

template <typename OutT, int kStages, int kBM, int kNT = 1>
__global__ __launch_bounds__(w4a4::kThreads, 2) void moe_grouped_w4a4_kernel(
    const uint8_t* __restrict__ a_codes, size_t a_code_stride, const uint8_t* __restrict__ a_scales,
    size_t a_scale_stride, const float* __restrict__ a_gs, const int32_t* __restrict__ act_rows,
    const MoeSegment* __restrict__ segs, const MoeExpertView* __restrict__ views, int which,
    OutT* __restrict__ out, size_t out_stride, int n, int k, int m_tiles) {
  dgpp_pdl_wait();
  using namespace w4a4;
  using TL = Tile<kBM>;
  constexpr size_t kACodesT = TL::a_codes, kAScalesT = TL::a_scales, kSlotT = TL::slot;
  extern __shared__ __align__(16) uint8_t smem[];
  const MoeSegment seg = segs[blockIdx.y];
  const int m_tile = static_cast<int>(blockIdx.x) % m_tiles;
  // kNT consecutive n-tiles per CTA: one cp.async ring runs across
  // the tiles, so a short-k GEMM (the down projection, k = 320: 2.5 stages a
  // tile) pays its pipeline prologue once per CTA, and each tile's epilogue
  // stores overlap the next tile's loads. kNT = 1 is the original kernel.
  const int n_group = static_cast<int>(blockIdx.x) / m_tiles;
  const int n_tiles_total = (n + BN - 1) / BN;
  const int t0 = n_group * kNT;
  const int tiles_here = min(kNT, n_tiles_total - t0);
  if (tiles_here <= 0) return;
  const int m0 = m_tile * kBM;
  if (m0 >= seg.rows) return;
  const int m_rows = min(kBM, seg.rows - m0);
  const MoeExpertView v = views[seg.expert * 3 + which];
  const float inv_g = 1.f / *v.fp4_global;
  const size_t w_code_stride = static_cast<size_t>(k) / 2;
  const size_t w_scale_stride = static_cast<size_t>(k) / 16;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5, lane = tid & 31;
  const int q = lane >> 2, t = lane & 3;
  const int wm = warp / TL::wn_warps, wn = warp % TL::wn_warps;  // (kBM/32) x (8 / that) warps
  const int stages = (k + BK - 1) / BK;

  // Per-thread copy assignments (fixed across stages).
  // A codes: kBM / 64 passes of (row tid/4 + 64 p, 16-byte chunk tid%4).
  constexpr int kAPasses = kBM / 64;
  const int a_chunk = tid & 3;
  int a_row[kAPasses];
  bool a_ok[kAPasses];
  const uint8_t* a_code_src[kAPasses];
#pragma unroll
  for (int p = 0; p < kAPasses; ++p) {
    a_row[p] = (tid >> 2) + 64 * p;
    a_ok[p] = a_row[p] < m_rows;
    int src_row = 0;
    if (a_ok[p]) {
      const int srow = seg.row0 + m0 + a_row[p];
      src_row = act_rows != nullptr ? act_rows[srow] : srow;
    }
    a_code_src[p] = a_codes + static_cast<size_t>(src_row) * a_code_stride;
  }
  // A scales: threads 0..63, one 8-byte copy each (row tid).
  int as_src_row = 0;
  const bool as_ok = tid < kBM && tid < m_rows;
  if (as_ok) {
    const int srow = seg.row0 + m0 + tid;
    as_src_row = act_rows != nullptr ? act_rows[srow] : srow;
  }
  const uint8_t* a_scale_src = a_scales + static_cast<size_t>(as_src_row) * a_scale_stride;
  // B codes: two 16-byte chunks per thread: rows (tid + 256 i) / 4.
  // B scales: row tid/2, 4-byte half tid%2.
  const int bs_row = tid >> 1, bs_half = tid & 1;

  auto slot_ptr = [&](int slot) { return smem + static_cast<size_t>(slot) * kSlotT; };

  // f: the flattened (tile, stage) index across this CTA's tiles.
  auto issue = [&](int f, int slot) {
    const int s = f % stages;
    const int n0 = (t0 + f / stages) * BN;
    const bool bs_ok = n0 + bs_row < n;
    const uint8_t* b_scale_src = v.fp4_scales + static_cast<size_t>(bs_ok ? n0 + bs_row : 0) * w_scale_stride;
    uint8_t* base = slot_ptr(slot);
    const int kb0 = s * (BK / 2);  // code byte offset of this stage
    const int kg0 = s * (BK / 16); // scale group offset
#pragma unroll
    for (int p = 0; p < kAPasses; ++p) {
      const int gb = kb0 + a_chunk * 16;
      const int in = a_ok[p] ? max(0, min(16, k / 2 - gb)) : 0;
      cp_async(base + a_row[p] * kRowBytes + a_chunk * 16, in > 0 ? a_code_src[p] + gb : a_codes, 16, in);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int idx = tid + kThreads * i;
      const int row = idx >> 2, chunk = idx & 3;
      const int gn = n0 + row;
      const int gb = kb0 + chunk * 16;
      const int in = gn < n ? max(0, min(16, k / 2 - gb)) : 0;
      const uint8_t* src = v.payload + static_cast<size_t>(gn < n ? gn : 0) * w_code_stride + gb;
      cp_async(base + kACodesT + row * kRowBytes + chunk * 16, in > 0 ? src : v.payload, 16, in);
    }
    if (tid < kBM) {
      const int in = as_ok ? max(0, min(8, k / 16 - kg0)) : 0;
      cp_async(base + kACodesT + kBCodes + tid * kAScaleBytes, in > 0 ? a_scale_src + kg0 : a_scales, 8, in);
    }
    {
      const int gg = kg0 + bs_half * 4;
      const int in = bs_ok ? max(0, min(4, k / 16 - gg)) : 0;
      cp_async(base + kACodesT + kBCodes + kAScalesT + bs_row * kAScaleBytes + bs_half * 4,
               in > 0 ? b_scale_src + gg : v.fp4_scales, 4, in);
    }
  };

  constexpr int NT = TL::nt, NC = TL::ncols;
  float acc[2][NT][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  const int total = tiles_here * stages;
#pragma unroll
  for (int f = 0; f < kStages - 1; ++f) {
    if (f < total) issue(f, f);
    cp_commit();
  }
  for (int f = 0; f < total; ++f) {
    const int s = f % stages;
    cp_wait<kStages - 2>();
    __syncthreads();
    {
      const int nx = f + kStages - 1;
      if (nx < total) issue(nx, nx % kStages);
      cp_commit();
    }
    const uint8_t* base = slot_ptr(f % kStages);
    const uint8_t* ab = base;
    const uint8_t* bb = base + kACodesT;
    const uint32_t* asw = reinterpret_cast<const uint32_t*>(base + kACodesT + kBCodes);
    const uint32_t* bsw = reinterpret_cast<const uint32_t*>(base + kACodesT + kBCodes + kAScalesT);
    // ldmatrix.x4 lane addressing: lane l feeds row (l & 7) of matrix l >> 3.
    const int lm = lane >> 3, lr = lane & 7;
    // The stage's block-scale words for both k64 halves in one 8-byte load per
    // row (2026-09-23: ncu put the kernel MIO-throttled -- 20 scale LDS per
    // stage per warp beside 12 ldmatrix for 32 MMAs; this halves the scale LDS).
    uint2 sfa2[2], sfb2[NT];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int sr = wm * 32 + i * 16 + q + ((lane & 1) << 3);
      sfa2[i] = *reinterpret_cast<const uint2*>(asw + sr * 2);
    }
#pragma unroll
    for (int j = 0; j < NT; ++j) sfb2[j] = *reinterpret_cast<const uint2*>(bsw + (wn * NC + j * 8 + q) * 2);
#pragma unroll
    for (int kh = 0; kh < 2; ++kh) {  // two k64 steps per stage
      uint32_t af[2][4], sfa[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        // matrices: rows +0/+8 (lm & 1) x bytes +0/+16 (lm >> 1) -> a0, a1, a2, a3.
        const int row = wm * 32 + i * 16 + (lm & 1) * 8 + lr;
        ldmatrix_x4(af[i], ab + row * kRowBytes + kh * 32 + (lm >> 1) * 16);
        sfa[i] = kh == 0 ? sfa2[i].x : sfa2[i].y;
      }
#pragma unroll
      for (int jp = 0; jp < NT / 2; ++jp) {
        // matrices: bytes +0/+16 (lm & 1) x n-tiles j/j+1 (lm >> 1) -> b0, b1 of each.
        uint32_t bf[4];
        const int col = wn * NC + jp * 16 + (lm >> 1) * 8 + lr;
        ldmatrix_x4(bf, bb + col * kRowBytes + kh * 32 + (lm & 1) * 16);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int j = jp * 2 + h;
          const uint32_t sfb = kh == 0 ? sfb2[j].x : sfb2[j].y;
#pragma unroll
          for (int i = 0; i < 2; ++i) mma_nvfp4(acc[i][j], af[i], bf[2 * h], bf[2 * h + 1], sfa[i], sfb);
        }
      }
    }
    if (s == stages - 1) {  // this tile's last stage: store it, clear for the next
      const int n0 = (t0 + f / stages) * BN;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int mm = wm * 32 + i * 16 + q + h * 8;
          if (mm >= m_rows) continue;
          const int srow = seg.row0 + m0 + mm;
          const float scale = a_gs[act_rows != nullptr ? act_rows[srow] : srow] * inv_g;
          OutT* orow = out + static_cast<size_t>(srow) * out_stride;
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            const int col = n0 + wn * NC + j * 8 + 2 * t;
            // Pairs as one 4-byte (bf16x2) / 8-byte (float2) store: col is even
            // and rows are even-strided (the element-at-a-time stores made the
            // output-heavy down projection store-instruction bound).
            if (col + 1 < n) {
              store_pair(orow + col, acc[i][j][2 * h] * scale, acc[i][j][2 * h + 1] * scale);
            } else if (col < n) {
              store_out(orow + col, acc[i][j][2 * h] * scale);
            }
          }
        }
      }
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
    }
  }
  cp_wait<0>();
}


// ---- the dense form on TMA boxes -------------------------------------------------
namespace dense_tma {
constexpr int kBM = 128, kBN = 256, kBK = 128, kStages = 3, kThreads = 256;
constexpr int kRow = kBK / 2;  // 64 code bytes a row a stage
constexpr int kACodes = kBM * kRow, kBCodes = kBN * kRow;  // 8 KB + 16 KB
constexpr int kAScales = kBM * 8, kBScales = kBN * 8;      // 1 KB + 2 KB (8 e4m3 a row a stage)
constexpr int kStageBytes = kACodes + kBCodes + kAScales + kBScales;  // 27 KB
constexpr size_t kSmem = size_t(kStages) * kStageBytes + 1024 + 64;

__device__ __forceinline__ unsigned smem_u32(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned parity) {
  unsigned done = 0;
  do {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                 : "=r"(done) : "r"(smem_u32(bar)), "r"(parity) : "memory");
  } while (!done);
}
__device__ __forceinline__ void tma_box_2d(unsigned dst, const CUtensorMap* map, int x, int y, uint64_t* bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               ::"r"(dst), "l"(map), "r"(x), "r"(y), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void cp_async8(unsigned dst, const void* src, int src_bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__global__ __launch_bounds__(kThreads, 1) void dense_w4a4_tma_kernel(
    const __grid_constant__ CUtensorMap amap, const __grid_constant__ CUtensorMap wmap,
    const uint8_t* __restrict__ a_scales, size_t a_scale_stride, const float* __restrict__ a_gs,
    const uint8_t* __restrict__ w_scales, const float* __restrict__ w_global, uint16_t* __restrict__ out,
    size_t out_stride, int m, int n, int k) {
  dgpp_pdl_wait();
  extern __shared__ __align__(1024) uint8_t smraw[];
  const unsigned sbase = smem_u32(smraw);
  uint8_t* sm = smraw + ((1024 - (sbase & 1023)) & 1023);
  const unsigned s0 = smem_u32(sm);
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + kStages * kStageBytes);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int q = lane >> 2, t = lane & 3, lm = lane >> 3, lr = lane & 7;
  const int wm = warp >> 2, wn = warp & 3;  // 2 x 4 warps of 64 x 64
  const int m_tiles = (m + kBM - 1) / kBM;
  const int mt = static_cast<int>(blockIdx.x) % m_tiles, nt = static_cast<int>(blockIdx.x) / m_tiles;
  const int m0 = mt * kBM, n0 = nt * kBN;
  const int steps = k / kBK;
  const size_t w_scale_stride = static_cast<size_t>(k) / 16;
  const float inv_g = 1.f / *w_global;
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) mbar_init(&full[s], 1);
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  const bool a_ok = tid < kBM && m0 + tid < m, w_ok = n0 + tid < n;
  const uint8_t* a_sc = a_scales + static_cast<size_t>(a_ok ? m0 + tid : 0) * a_scale_stride;
  const uint8_t* w_sc = w_scales + static_cast<size_t>(w_ok ? n0 + tid : 0) * w_scale_stride;
  auto stage = [&](int step, int slot) {
    const unsigned base = s0 + slot * kStageBytes;
    if (tid == 0) {
      mbar_expect_tx(&full[slot], static_cast<unsigned>(kACodes + kBCodes));
      tma_box_2d(base, &amap, step * kRow, m0, &full[slot]);
      tma_box_2d(base + kACodes, &wmap, step * kRow, n0, &full[slot]);
    }
    if (tid < kBM) cp_async8(base + kACodes + kBCodes + tid * 8, a_sc + step * 8, a_ok ? 8 : 0);
    cp_async8(base + kACodes + kBCodes + kAScales + tid * 8, w_sc + step * 8, w_ok ? 8 : 0);
    cp_commit();
  };
  // Per-lane ldmatrix bases for slot 0 with the 64B swizzle folded in (the XOR
  // operand (row >> 1) & 3 is (lr >> 1) & 3 for every row a lane touches).
  unsigned lba[4], lbb[4];
  {
    const int sw = (lr >> 1) & 3;
#pragma unroll
    for (int i = 0; i < 4; ++i) lba[i] = s0 + (wm * 64 + i * 16 + (lm & 1) * 8 + lr) * kRow + (((lm >> 1) ^ sw) << 4);
#pragma unroll
    for (int jp = 0; jp < 4; ++jp)
      lbb[jp] = s0 + kACodes + (wn * 64 + jp * 16 + (lm >> 1) * 8 + lr) * kRow + (((lm & 1) ^ sw) << 4);
  }
  float acc[4][8][4] = {};
#pragma unroll
  for (int s = 0; s < kStages - 1; ++s)
    if (s < steps) stage(s, s);
  int slot = 0;
  unsigned parity = 0;
  for (int step = 0; step < steps; ++step) {
    mbar_wait(&full[slot], parity);
    // This step's scale words: wait_group N leaves the N most recent groups in
    // flight, so N is the number of groups committed after this step's — one
    // per stage still ahead, none at the drain (the last step's own group would
    // otherwise be the most recent and could be read before it landed:
    // intermittent, 4 of 49 launches at the 27B MLP's gate shape, 2026-10-17).
    if (step + kStages - 2 < steps) cp_wait<kStages - 2>(); else cp_wait<0>();
    __syncthreads();  // this step's scales landed for every thread; every warp is done with the slot being refilled
    if (step + kStages - 1 < steps) stage(step + kStages - 1, (slot + kStages - 1) % kStages);
    const unsigned so = static_cast<unsigned>(slot) * kStageBytes;
    const uint32_t* asw = reinterpret_cast<const uint32_t*>(sm + so + kACodes + kBCodes);
    const uint32_t* bsw = reinterpret_cast<const uint32_t*>(sm + so + kACodes + kBCodes + kAScales);
    uint2 sfa2[4], sfb2[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) sfa2[i] = *reinterpret_cast<const uint2*>(asw + (wm * 64 + i * 16 + q + ((lane & 1) << 3)) * 2);
#pragma unroll
    for (int j = 0; j < 8; ++j) sfb2[j] = *reinterpret_cast<const uint2*>(bsw + (wn * 64 + j * 8 + q) * 2);
#pragma unroll
    for (int kh = 0; kh < 2; ++kh) {
      const unsigned x = static_cast<unsigned>(kh) << 5;
      uint32_t af[4][4], sfa[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        ldsm_x4(af[i], (lba[i] ^ x) + so);
        sfa[i] = kh == 0 ? sfa2[i].x : sfa2[i].y;
      }
#pragma unroll
      for (int jp = 0; jp < 4; ++jp) {
        uint32_t bf[4];
        ldsm_x4(bf, (lbb[jp] ^ x) + so);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int j = jp * 2 + h;
          const uint32_t sfb = kh == 0 ? sfb2[j].x : sfb2[j].y;
#pragma unroll
          for (int i = 0; i < 4; ++i) mma_nvfp4(acc[i][j], af[i], bf[2 * h], bf[2 * h + 1], sfa[i], sfb);
        }
      }
    }
    if (++slot == kStages) {
      slot = 0;
      parity ^= 1u;
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = m0 + wm * 64 + i * 16 + q + h * 8;
      if (row >= m) continue;
      const float scale = a_gs[row] * inv_g;
      uint16_t* orow = out + static_cast<size_t>(row) * out_stride;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int col = n0 + wn * 64 + j * 8 + 2 * t;
        if (col + 1 < n) store_pair(orow + col, acc[i][j][2 * h] * scale, acc[i][j][2 * h + 1] * scale);
        else if (col < n) store_out(orow + col, acc[i][j][2 * h] * scale);
      }
    }
}
}  // namespace dense_tma

}  // namespace

size_t nvfp4_act_scale_stride(int k) { return static_cast<size_t>(((k / 16) + 7) / 8 * 8); }

void launch_quantize_rows_nvfp4(const uint16_t* x, size_t x_stride, int rows, int k, uint8_t* codes,
                                uint8_t* scales, float* gs, cudaStream_t stream, float static_gs,
                                const float* static_gs_dev) {
  if (rows <= 0) return;
  if (k % 64 != 0 || k / 16 > 1024 * kGroupsPerThread)
    throw std::invalid_argument("quantize nvfp4: k must be a multiple of 64, <= 32768");
  if (x_stride % 8 != 0 || reinterpret_cast<uintptr_t>(x) % 16 != 0)
    throw std::invalid_argument("quantize nvfp4: rows must be 16-byte aligned");
  const int threads = std::min(1024, ((k / 16) + 31) / 32 * 32);
  quantize_rows_nvfp4_kernel<false><<<rows, threads, 0, stream>>>(x, x_stride, rows, k, codes,
                                                                   static_cast<size_t>(k) / 2, scales,
                                                                   nvfp4_act_scale_stride(k), gs, static_gs, static_gs_dev);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_swiglu_quantize_rows_nvfp4(const uint16_t* gate, const uint16_t* up, size_t stride, int rows, int k,
                                       float limit, uint8_t* codes, uint8_t* scales, float* gs,
                                       cudaStream_t stream, float static_gs, const float* static_gs_dev,
                                       bool qwen_rounding) {
  if (rows <= 0) return;
  if (k % 64 != 0 || k / 16 > 1024 * kGroupsPerThread)
    throw std::invalid_argument("swiglu quantize nvfp4: k % 64, <= 32768");
  if (stride % 8 != 0 || reinterpret_cast<uintptr_t>(gate) % 16 != 0 || reinterpret_cast<uintptr_t>(up) % 16 != 0)
    throw std::invalid_argument("swiglu quantize nvfp4: rows must be 16-byte aligned");
  const int threads = std::min(1024, ((k / 16) + 31) / 32 * 32);
  if (qwen_rounding)
    quantize_rows_nvfp4_kernel<true, true><<<rows, threads, 0, stream>>>(gate, stride, rows, k, codes,
                                                                          static_cast<size_t>(k) / 2, scales,
                                                                          nvfp4_act_scale_stride(k), gs, static_gs, static_gs_dev, up, limit);
  else
    quantize_rows_nvfp4_kernel<true><<<rows, threads, 0, stream>>>(gate, stride, rows, k, codes,
                                                                    static_cast<size_t>(k) / 2, scales,
                                                                    nvfp4_act_scale_stride(k), gs, static_gs, static_gs_dev, up, limit);
  DGPP_CUDA_OK(cudaGetLastError());
}

template <typename OutT>
static void launch_w4a4(const uint8_t* codes, const uint8_t* scales, const float* gs, const int32_t* act_rows,
                        const MoeSegment* segs, int n_segs, int max_rows, const MoeExpertView* views, int which,
                        OutT* out, size_t out_stride, int n, int k, cudaStream_t stream) {
  using namespace w4a4;
  if (n_segs <= 0 || n <= 0) return;
  if (k % 64 != 0) throw std::invalid_argument("moe w4a4: k must be a multiple of 64");
  if (max_rows <= 0) throw std::invalid_argument("moe w4a4: max_rows");
  // The epilogue stores column pairs (bf16x2 / float2): rows must be even-strided and aligned.
  if (out_stride % 2 != 0 || (reinterpret_cast<uintptr_t>(out) % (2 * sizeof(OutT))) != 0)
    throw std::invalid_argument("moe w4a4: out rows must be even-strided and pair-aligned");
  static const int stages = [] {
    const char* e = std::getenv("DGPP_W4A4_STAGES");
    return e != nullptr && std::atoi(e) == 3 ? 3 : 2;
  }();
  // DGPP_W4A4_BM=64|128 (default 128; idle GPU, moe_w4a4_test at 8,192
  // tokens x top-10: gate/up 3.40 -> 3.27 ms, down 5.27 -> 4.89 ms).
  static const int bm = [] {
    const char* e = std::getenv("DGPP_W4A4_BM");
    return e != nullptr && std::atoi(e) == 64 ? 64 : 128;
  }();
  static const bool opted = [] {  // once per process and OutT, thread-safe
    DGPP_CUDA_OK(cudaFuncSetAttribute(moe_grouped_w4a4_kernel<OutT, 2, 64>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes_bm<2, 64>())));
    DGPP_CUDA_OK(cudaFuncSetAttribute(moe_grouped_w4a4_kernel<OutT, 3, 64>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes_bm<3, 64>())));
    DGPP_CUDA_OK(cudaFuncSetAttribute(moe_grouped_w4a4_kernel<OutT, 2, 128>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes_bm<2, 128>())));
    return true;
  }();
  (void)opted;
  const int m_tiles = (max_rows + bm - 1) / bm;
  const unsigned n_tiles = static_cast<unsigned>((n + BN - 1) / BN);
  const size_t cs = static_cast<size_t>(k) / 2, ss = nvfp4_act_scale_stride(k);
  // n-tiles per CTA (DGPP_W4A4_NT=1|2|4; measured neutral, default 1).
  static const int nt_env = [] {
    const char* e = std::getenv("DGPP_W4A4_NT");
    const int v = e != nullptr ? std::atoi(e) : 1;
    return v == 2 || v == 4 ? v : 1;
  }();
  if (bm == 128 && nt_env > 1) {
    static const bool opted_nt = [] {
      DGPP_CUDA_OK(cudaFuncSetAttribute(moe_grouped_w4a4_kernel<OutT, 2, 128, 2>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smem_bytes_bm<2, 128>())));
      DGPP_CUDA_OK(cudaFuncSetAttribute(moe_grouped_w4a4_kernel<OutT, 2, 128, 4>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smem_bytes_bm<2, 128>())));
      return true;
    }();
    (void)opted_nt;
    const unsigned groups = (n_tiles + nt_env - 1) / nt_env;
    const dim3 g2(groups * static_cast<unsigned>(m_tiles), static_cast<unsigned>(n_segs), 1u);
    if (nt_env == 2)
      moe_grouped_w4a4_kernel<OutT, 2, 128, 2><<<g2, kThreads, smem_bytes_bm<2, 128>(), stream>>>(
          codes, cs, scales, ss, gs, act_rows, segs, views, which, out, out_stride, n, k, m_tiles);
    else
      moe_grouped_w4a4_kernel<OutT, 2, 128, 4><<<g2, kThreads, smem_bytes_bm<2, 128>(), stream>>>(
          codes, cs, scales, ss, gs, act_rows, segs, views, which, out, out_stride, n, k, m_tiles);
    DGPP_CUDA_OK(cudaGetLastError());
    return;
  }
  const dim3 grid(n_tiles * static_cast<unsigned>(m_tiles), static_cast<unsigned>(n_segs), 1u);
  if (bm == 128)
    moe_grouped_w4a4_kernel<OutT, 2, 128><<<grid, kThreads, smem_bytes_bm<2, 128>(), stream>>>(
        codes, cs, scales, ss, gs, act_rows, segs, views, which, out, out_stride, n, k, m_tiles);
  else if (stages == 3)
    moe_grouped_w4a4_kernel<OutT, 3, 64><<<grid, kThreads, smem_bytes_bm<3, 64>(), stream>>>(
        codes, cs, scales, ss, gs, act_rows, segs, views, which, out, out_stride, n, k, m_tiles);
  else
    moe_grouped_w4a4_kernel<OutT, 2, 64><<<grid, kThreads, smem_bytes_bm<2, 64>(), stream>>>(
        codes, cs, scales, ss, gs, act_rows, segs, views, which, out, out_stride, n, k, m_tiles);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_grouped_w4a4_bf16(const uint8_t* codes, const uint8_t* scales, const float* gs,
                                  const int32_t* act_rows, const MoeSegment* segs, int n_segs, int max_rows,
                                  const MoeExpertView* views, int which, uint16_t* out, size_t out_stride, int n,
                                  int k, cudaStream_t stream) {
  launch_w4a4<uint16_t>(codes, scales, gs, act_rows, segs, n_segs, max_rows, views, which, out, out_stride, n, k,
                        stream);
}
void launch_moe_grouped_w4a4_f32(const uint8_t* codes, const uint8_t* scales, const float* gs,
                                 const int32_t* act_rows, const MoeSegment* segs, int n_segs, int max_rows,
                                 const MoeExpertView* views, int which, float* out, size_t out_stride, int n, int k,
                                 cudaStream_t stream) {
  launch_w4a4<float>(codes, scales, gs, act_rows, segs, n_segs, max_rows, views, which, out, out_stride, n, k,
                     stream);
}

bool dense_w4a4_tma_supported(int k) { return k > 0 && k % dense_tma::kBK == 0; }

CUtensorMap nvfp4_code_tma_map(const uint8_t* codes, int64_t rows, int64_t k, int box_rows) {
  if (!dense_w4a4_tma_supported(static_cast<int>(k)) || (reinterpret_cast<uintptr_t>(codes) % 16) != 0)
    throw std::invalid_argument("nvfp4_code_tma_map: k must be a multiple of 128 and the codes 16-byte aligned");
  if (box_rows != 128 && box_rows != 256) throw std::invalid_argument("nvfp4_code_tma_map: box rows must be 128 or 256");
  CUtensorMap map{};
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(k / 2), static_cast<cuuint64_t>(rows)};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(k / 2)};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(dense_tma::kRow), static_cast<cuuint32_t>(box_rows)};
  const cuuint32_t estr[2] = {1, 1};
  const CUresult r = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<uint8_t*>(codes), dims,
                                            strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
                                            CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS)
    throw std::runtime_error("nvfp4_code_tma_map: cuTensorMapEncodeTiled failed (" + std::to_string(static_cast<int>(r)) + ")");
  return map;
}

void launch_dense_w4a4_tma_bf16(const CUtensorMap& amap, const CUtensorMap& wmap, const uint8_t* a_scales,
                                size_t a_scale_stride, const float* a_gs, const uint8_t* w_scales, const float* w_global,
                                uint16_t* out, size_t out_stride, int m, int n, int k, cudaStream_t stream) {
  using namespace dense_tma;
  if (m <= 0 || n <= 0) return;
  if (!dense_w4a4_tma_supported(k)) throw std::invalid_argument("dense w4a4 tma: k must be a multiple of 128");
  if (out_stride % 2 != 0 || (reinterpret_cast<uintptr_t>(out) % 4) != 0)
    throw std::invalid_argument("dense w4a4 tma: out rows must be even-strided and pair-aligned");
  static const bool opted = [] {
    DGPP_CUDA_OK(cudaFuncSetAttribute(dense_w4a4_tma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmem)));
    return true;
  }();
  (void)opted;
  const dim3 grid(static_cast<unsigned>(((m + kBM - 1) / kBM) * ((n + kBN - 1) / kBN)));
  dense_w4a4_tma_kernel<<<grid, kThreads, kSmem, stream>>>(amap, wmap, a_scales, a_scale_stride, a_gs, w_scales, w_global, out,
                                                           out_stride, m, n, k);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
