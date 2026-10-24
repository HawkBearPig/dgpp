// Dense causal paged attention for the Qwen3.8-27B full-attention layers, in
// two forms over the same paged K/V cache (bf16 [slots, kv_heads * 256]):
//
//  * The row form (decode): one warp per (query row, KV group) — the group's
//    <= 16 query heads are the M rows of m16n8k16 A-fragments, K/V tiles of
//    16 tokens gathered into shared memory with cp.async, S = Q K^T and
//    O += P V on the tensor cores, P reused from S's accumulator layout
//    (FlashAttention-2). A row's keys are split over blockIdx.z ranges so a
//    long context walks in parallel (kMaxSplits warps per (row, group)); each
//    split leaves an unnormalized partial (m, l, acc) and a combine pass
//    merges them. splits = 1 writes the normalized fp32 output directly.
//
//  * The tile form (prefill): one CTA per (16 consecutive query rows of one
//    request, KV group), one warp per query head of the group — 16 queries
//    are the M rows, so every mma row carries a query — over 32-token K/V
//    tiles staged once per CTA and read by all of the group's warps. Rows
//    attend [0, pos[r]] through the per-row mask; the CTA walks to the
//    tile's furthest position.
//    2026-10-15 (measured on the GB10 at 262K context, full waves, bitwise
//    the previous loop): the PV pass runs its two key halves as two passes,
//    the rescale is skipped when no row's maximum moved, a tile's K/V rows
//    are one contiguous strip (32 divides the page) resolved once per tile
//    with shifts instead of a per-thread table and 64-bit divides, one
//    barrier per tile instead of three, and the staging copies bypass L1:
//    45.9 -> 56.5 TFLOP/s. A bisection of what is left (the inner loops
//    alone reach 86 at this occupancy) puts the staging path at 23 % of
//    the kernel, the softmax chain at 15 % and the lockstep barrier at
//    12 %; bulk (TMA 1-D) row copies were measured slower (64 half-KB
//    descriptors a tile), the 2-D tensor form and a ping-pong warp
//    schedule are the open items.
//
// Numerics (both forms): probabilities rounded to bf16 for P V, the
// denominator summing the unrounded values (the QSA rule); the dots'
// summation order is the tensor cores'. Tolerance-equal to a dense
// reference, not bitwise, and the two forms differ from each other in their
// rescaling points (16- vs 32-token tiles, the split boundaries).
//
// The fp8 pool (engine.kv_dtype = fp8, 2026-10-16) adds the storage form
// (the same math on codes widened in shared memory) and, for prefill under
// engine.prefill_attention = fp8, the compute form (Q and P in e4m3 on the
// fp8 tensor cores) — their sections below.
#include "kernels/full_attn.hpp"
#include "kernels/pdl.cuh"

#include <cmath>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {

constexpr int kD = 256;
constexpr int kRow = kD + 8;  // padded bf16 row: conflict-free ldmatrix

// The row form.
constexpr int kTileTok = 16;
constexpr int kRowStageElems = 2 * kTileTok * kRow;  // K then V
constexpr size_t kRowSmem = size_t(kRowStageElems) * 2;  // bytes, one stage
constexpr int kMaxSplits = 32;
constexpr int kPartialFloats = 16 * kD + 32;  // acc[16][256], m[16], l[16]

// The tile form.
constexpr int kQTile = 16;
constexpr int kKTile = 32;
constexpr int kMaxTileGroup = 8;  // warps per CTA (255 registers per thread)
constexpr int kTileStageElems = 2 * kKTile * kRow;
constexpr size_t kTileSmem = size_t(2) * kTileStageElems * 2;  // two stages, bytes

__device__ __forceinline__ void fldsm_x4(uint32_t (&r)[4], const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
// The same loads from a 32-bit shared address a lane keeps in a register.
__device__ __forceinline__ void fldsm_x4_a(uint32_t (&r)[4], unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void fldsm_x4_ta(uint32_t (&r)[4], unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void fldsm_x4_t(uint32_t (&r)[4], const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void fmma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void fcp_async16(void* dst, const void* src, bool valid) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  const int bytes = valid ? 16 : 0;  // 0: zero-fill
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(bytes));
}
// The L1-bypassing form for the tile walk's staging (2026-10-15: +7 % at
// full waves over .ca on the GB10; the row form is unmeasured and keeps .ca).
__device__ __forceinline__ void fcp_async16_cg(void* dst, const void* src, bool valid) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  const int bytes = valid ? 16 : 0;  // 0: zero-fill
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(bytes));
}
// The hardware pack: cvt.rn.bf16x2.f32 agrees with float_to_bf16_bits on
// every one of the 2^32 float patterns (checked exhaustively 2026-10-15, NaN
// inputs included); the bit-trick form cost ~9 instructions per value.
__device__ __forceinline__ uint32_t fpack_bf16(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
// exp2 of a masked score against a running max: a fully masked row keeps
// its running max at -inf, and contributes nothing.
__device__ __forceinline__ float fpexp(float s, float n) { return n == -INFINITY ? 0.f : exp2f(s - n); }

// ---- the fp8 storage form (engine.kv_dtype = fp8, 2026-10-16) ------------------
// The K/V planes hold e4m3 codes of K / k_scale and V / v_scale (the mixed
// release's per-layer kv_cache_scheme scalars); the attention math stays
// the bf16 chain's: a tile's codes are widened to bf16 in shared memory
// (exact — every e4m3 value has <= 4 significant bits and sits inside the
// bf16 range), k_scale rides the softmax scale and v_scale the output.
// Two codes of a word (bytes 0,1 or 2,3) to one bf16x2: the hardware
// e4m3x2 -> f16x2 unpack, two f16 -> f32 widenings and the f32x2 -> bf16x2
// pack — four instructions a pair, exact (no direct e4m3 -> bf16 cvt
// exists). An integer form (prmt, shift, two lop3 and a bf16x2 multiply by
// 2^120 — exact on every code too) is five instructions a pair and read
// 3 % slower on a 262K prefill in situ (2026-10-16: 362 vs 352 s); the
// widening is issue-bound at the kernel's occupancy, so the shorter chain
// wins whatever the conversion units' rate.
__device__ __forceinline__ uint32_t fe4m3x2_to_bf16x2(uint32_t word, bool high) {
  uint32_t h;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(static_cast<unsigned short>(high ? word >> 16 : word)));
  float lo, hi;
  asm("{ .reg .f16 a, b; mov.b32 {a, b}, %2; cvt.f32.f16 %0, a; cvt.f32.f16 %1, b; }" : "=f"(lo), "=f"(hi) : "r"(h));
  return fpack_bf16(lo, hi);
}
// One 16-byte chunk of codes to its two 16-byte chunks of bf16.
__device__ __forceinline__ void fconvert_chunk16(const uint4 c, uint4& lo, uint4& hi) {
  lo.x = fe4m3x2_to_bf16x2(c.x, false);
  lo.y = fe4m3x2_to_bf16x2(c.x, true);
  lo.z = fe4m3x2_to_bf16x2(c.y, false);
  lo.w = fe4m3x2_to_bf16x2(c.y, true);
  hi.x = fe4m3x2_to_bf16x2(c.z, false);
  hi.y = fe4m3x2_to_bf16x2(c.z, true);
  hi.z = fe4m3x2_to_bf16x2(c.w, false);
  hi.w = fe4m3x2_to_bf16x2(c.w, true);
}
// The row form's staging of one fp8 tile: K then V, 16 tokens x 256 codes,
// chunk c of token i at chunk c ^ (i & 7) of its 256-byte row — the widening
// pass reads a lane's token down its chunks with eight lanes on eight rows,
// conflict-free either way.
constexpr int kRow8PlaneBytes = kTileTok * kD;  // 4 KB
constexpr int kRow8StageBytes = 2 * kRow8PlaneBytes;

// ---- the row form ---------------------------------------------------------------

template <bool kFp8>
__global__ __launch_bounds__(32) void full_attn_row_kernel(
    const uint16_t* __restrict__ q, int64_t q_row_stride, const void* __restrict__ k_cache,
    const void* __restrict__ v_cache, const int32_t* __restrict__ req_ids, const int64_t* __restrict__ pos,
    int local_heads, int kv_heads, int block_tokens, const int32_t* __restrict__ block_tables,
    int blocks_per_request, float scale, const float* __restrict__ kv_scales, float* __restrict__ out,
    float* __restrict__ partials, int splits) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint16_t sm[];
  // fp8: the staged codes sit past the bf16 tile; the scales fold into the
  // softmax scale (k) and the output (v).
  uint8_t* s8 = reinterpret_cast<uint8_t*>(sm + kRowStageElems);
  const float kscale = kFp8 ? scale * kv_scales[0] : scale;
  const float vs = kFp8 ? kv_scales[1] : 1.f;
  const int64_t r = blockIdx.x;
  const int kvh = blockIdx.y;
  const int split = blockIdx.z;
  const int group = local_heads / kv_heads;
  const int h0 = kvh * group;
  const int lane = threadIdx.x;
  const int g = lane >> 2, t = lane & 3;
  const int width = kv_heads * kD;
  const int64_t cnt = pos[r] + 1;  // dense causal visible set [0, pos[r]]
  const int32_t* bt = block_tables + static_cast<int64_t>(req_ids[r]) * blocks_per_request;
  const float sl2 = kscale * 1.4426950408889634f;
  // This split's key range: tile-aligned chunks of ceil(cnt / splits).
  const int64_t chunk = ((cnt + splits - 1) / splits + kTileTok - 1) / kTileTok * kTileTok;
  const int64_t begin = static_cast<int64_t>(split) * chunk;
  const int64_t end = min(cnt, begin + chunk);

  // Q A-fragments, 16 k-steps x 4 registers; rows past the group are zero.
  uint32_t qa[16][4];
  {
    const uint16_t* q0 = q + r * q_row_stride + static_cast<int64_t>(h0) * kD;
    const bool v0 = g < group, v1 = g + 8 < group;
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      const int c = ks * 16 + 2 * t;
      qa[ks][0] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + g * kD + c) : 0u;
      qa[ks][1] = v1 ? *reinterpret_cast<const uint32_t*>(q0 + (g + 8) * kD + c) : 0u;
      qa[ks][2] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + g * kD + c + 8) : 0u;
      qa[ks][3] = v1 ? *reinterpret_cast<const uint32_t*>(q0 + (g + 8) * kD + c + 8) : 0u;
    }
  }

  // Stage a tile: 16 tokens x (K, V) x 32 16-byte chunks = 1024 chunks, 32 per lane.
  // Lanes 0..15 resolve the tile's 16 physical rows in one round of loads.
  auto stage = [&](int64_t base) {
    uint16_t* kt = sm;
    uint16_t* vt = kt + kTileTok * kRow;
    int64_t my_off = 0;
    if (lane < kTileTok && base + lane < end) {
      const int64_t tok = base + lane;
      const int64_t phys = static_cast<int64_t>(bt[tok / block_tokens]) * block_tokens + tok % block_tokens;
      my_off = phys * width + static_cast<int64_t>(kvh) * kD;
    }
    if constexpr (kFp8) {
      // Sixteen 16-byte chunks a lane: lanes 0..15 the K plane, 16..31 V.
      const uint8_t* src = static_cast<const uint8_t*>(lane < 16 ? k_cache : v_cache) + (lane & 15) * 16;
      uint8_t* dst = s8 + (lane >> 4) * kRow8PlaneBytes;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int64_t off = __shfl_sync(0xffffffffu, my_off, i);
        fcp_async16(dst + i * kD + (((lane & 15) ^ (i & 7)) << 4), src + off, base + i < end);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int64_t off = __shfl_sync(0xffffffffu, my_off, i) + lane * 8;
        const bool valid = base + i < end;
        fcp_async16(kt + i * kRow + lane * 8, static_cast<const uint16_t*>(k_cache) + off, valid);
        fcp_async16(vt + i * kRow + lane * 8, static_cast<const uint16_t*>(v_cache) + off, valid);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };

  float acc[32][4];
#pragma unroll
  for (int d = 0; d < 32; ++d) acc[d][0] = acc[d][1] = acc[d][2] = acc[d][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;  // rows g, g + 8 (lane-partial l)

  for (int64_t base = begin; base < end; base += kTileTok) {
    stage(base);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncwarp();
    const uint16_t* kt = sm;
    const uint16_t* vt = kt + kTileTok * kRow;
    if constexpr (kFp8) {
      // Widen the tile: a lane walks its token's sixteen chunks (lanes
      // 0..15 K, 16..31 V) into the padded bf16 rows.
      const int tok = lane & 15;
      const uint8_t* src = s8 + (lane >> 4) * kRow8PlaneBytes + tok * kD;
      uint16_t* dst = sm + (lane >> 4) * (kTileTok * kRow) + tok * kRow;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const uint4 c = *reinterpret_cast<const uint4*>(src + ((j ^ (tok & 7)) << 4));
        uint4 lo, hi;
        fconvert_chunk16(c, lo, hi);
        *reinterpret_cast<uint4*>(dst + j * 16) = lo;
        *reinterpret_cast<uint4*>(dst + j * 16 + 8) = hi;
      }
      __syncwarp();
    }
    float sc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      uint32_t b[4];
      fldsm_x4(b, kt + ((lane & 7) + ((lane >> 4) << 3)) * kRow + ks * 16 + ((lane >> 3) & 1) * 8);
      fmma_bf16(sc[0], qa[ks], b[0], b[1]);
      fmma_bf16(sc[1], qa[ks], b[2], b[3]);
    }
    const int n = min(kTileTok, static_cast<int>(end - base));
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int col = j * 8 + 2 * t + e;
        sc[j][e] = col < n ? sc[j][e] * sl2 : -INFINITY;
        sc[j][2 + e] = col < n ? sc[j][2 + e] * sl2 : -INFINITY;
        mx0 = fmaxf(mx0, sc[j][e]);
        mx1 = fmaxf(mx1, sc[j][2 + e]);
      }
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
    }
    const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
    const float a0 = fpexp(m0, n0), a1 = fpexp(m1, n1);  // m = -inf on the first tile: 0
    m0 = n0;
    m1 = n1;
    float p[2][4];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      p[j][0] = fpexp(sc[j][0], n0);
      p[j][1] = fpexp(sc[j][1], n0);
      p[j][2] = fpexp(sc[j][2], n1);
      p[j][3] = fpexp(sc[j][3], n1);
    }
    l0 = l0 * a0 + p[0][0] + p[0][1] + p[1][0] + p[1][1];
    l1 = l1 * a1 + p[0][2] + p[0][3] + p[1][2] + p[1][3];
    const uint32_t pa[4] = {fpack_bf16(p[0][0], p[0][1]), fpack_bf16(p[0][2], p[0][3]),
                            fpack_bf16(p[1][0], p[1][1]), fpack_bf16(p[1][2], p[1][3])};
#pragma unroll
    for (int d = 0; d < 32; ++d) {
      acc[d][0] *= a0;
      acc[d][1] *= a0;
      acc[d][2] *= a1;
      acc[d][3] *= a1;
    }
#pragma unroll
    for (int dp = 0; dp < 16; ++dp) {
      uint32_t b[4];
      fldsm_x4_t(b, vt + ((lane & 7) + ((lane >> 3) & 1) * 8) * kRow + dp * 16 + (lane >> 4) * 8);
      fmma_bf16(acc[dp * 2], pa, b[0], b[1]);
      fmma_bf16(acc[dp * 2 + 1], pa, b[2], b[3]);
    }
    __syncwarp();  // this buffer's reads done before the next stage() overwrites it
  }
#pragma unroll
  for (int o = 1; o <= 2; o <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, o);
    l1 += __shfl_xor_sync(0xffffffffu, l1, o);
  }
  if (partials == nullptr) {
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
    float* o0 = out + (r * local_heads + h0 + g) * kD;
    float* o1 = out + (r * local_heads + h0 + g + 8) * kD;
#pragma unroll
    for (int d = 0; d < 32; ++d) {
      const int col = d * 8 + 2 * t;
      if (g < group) *reinterpret_cast<float2*>(o0 + col) = make_float2((acc[d][0] * i0) * vs, (acc[d][1] * i0) * vs);
      if (g + 8 < group)
        *reinterpret_cast<float2*>(o1 + col) = make_float2((acc[d][2] * i1) * vs, (acc[d][3] * i1) * vs);
    }
    return;
  }
  // The split's unnormalized partial: acc rows g / g + 8 at their running maxima.
  float* part = partials + ((r * kv_heads + kvh) * static_cast<int64_t>(splits) + split) * kPartialFloats;
  float* pa0 = part + g * kD;
  float* pa1 = part + (g + 8) * kD;
#pragma unroll
  for (int d = 0; d < 32; ++d) {
    const int col = d * 8 + 2 * t;
    *reinterpret_cast<float2*>(pa0 + col) = make_float2(acc[d][0], acc[d][1]);
    *reinterpret_cast<float2*>(pa1 + col) = make_float2(acc[d][2], acc[d][3]);
  }
  if (t == 0) {
    part[16 * kD + g] = m0;
    part[16 * kD + g + 8] = m1;
    part[16 * kD + 16 + g] = l0;
    part[16 * kD + 16 + g + 8] = l1;
  }
}

// One block per (row, KV group): every head's 256 outputs from the splits'
// (m, l, acc) — the largest maximum, the rescaled denominators, the sum.
// The group's (head, split) m and l are staged once per block and one
// thread per head forms the maximum, the splits' weights and the
// denominator from shared memory (2026-10-20: each of the 256 threads
// re-read every split's m for the maximum and its m and l again in the
// merge, and computed the weight itself — the same arithmetic in the same
// order, so the outputs are bitwise); the element loop reads its partials
// and the staged weights.
constexpr int kCombineMaxGroup = 16;  // kPartialFloats carries 16 heads' m and l
__global__ __launch_bounds__(256) void full_attn_combine_kernel(const float* __restrict__ partials,
                                                                 int splits, int local_heads, int kv_heads,
                                                                 const float* __restrict__ v_scale,
                                                                 float* __restrict__ out) {
  dgpp_pdl_wait();
  __shared__ float s_m[kCombineMaxGroup * kMaxSplits];
  __shared__ float s_l[kCombineMaxGroup * kMaxSplits];
  __shared__ float s_w[kCombineMaxGroup * kMaxSplits];
  __shared__ float s_lt[kCombineMaxGroup];
  const float vs = v_scale != nullptr ? *v_scale : 1.f;
  const int64_t r = blockIdx.x;
  const int kvh = blockIdx.y;
  const int group = local_heads / kv_heads;
  const int d = threadIdx.x;
  const float* base = partials + (r * kv_heads + kvh) * static_cast<int64_t>(splits) * kPartialFloats;
  if (group <= kCombineMaxGroup && splits <= kMaxSplits) {
    for (int i = threadIdx.x; i < group * splits; i += blockDim.x) {
      const int h = i / splits, s = i - h * splits;
      s_m[i] = base[s * kPartialFloats + 16 * kD + h];
      s_l[i] = base[s * kPartialFloats + 16 * kD + 16 + h];
    }
    __syncthreads();
    if (threadIdx.x < group) {
      const int h = threadIdx.x;
      float mt = -INFINITY;
      for (int s = 0; s < splits; ++s) mt = fmaxf(mt, s_m[h * splits + s]);
      float lt = 0.f;
      if (mt != -INFINITY) {
        for (int s = 0; s < splits; ++s) {
          const float ms = s_m[h * splits + s];
          if (ms == -INFINITY) continue;
          const float w = exp2f(ms - mt);
          lt += s_l[h * splits + s] * w;
          s_w[h * splits + s] = w;
        }
      }
      s_lt[h] = lt;
    }
    __syncthreads();
    for (int h = 0; h < group; ++h) {
      const float lt = s_lt[h];
      float o = 0.f;
      if (lt > 0.f) {
        for (int s = 0; s < splits; ++s) {
          if (s_m[h * splits + s] == -INFINITY) continue;
          o += base[s * kPartialFloats + h * kD + d] * s_w[h * splits + s];
        }
      }
      out[(r * local_heads + kvh * group + h) * kD + d] = (lt > 0.f ? o / lt : 0.f) * vs;
    }
    return;
  }
  for (int h = 0; h < group; ++h) {
    float mt = -INFINITY;
    for (int s = 0; s < splits; ++s) mt = fmaxf(mt, base[s * kPartialFloats + 16 * kD + h]);
    float lt = 0.f, o = 0.f;
    if (mt != -INFINITY) {
      for (int s = 0; s < splits; ++s) {
        const float* p = base + s * kPartialFloats;
        const float ms = p[16 * kD + h];
        if (ms == -INFINITY) continue;
        const float w = exp2f(ms - mt);
        lt += p[16 * kD + 16 + h] * w;
        o += p[h * kD + d] * w;
      }
    }
    out[(r * local_heads + kvh * group + h) * kD + d] = (lt > 0.f ? o / lt : 0.f) * vs;
  }
}

// ---- the tile form -------------------------------------------------------------

__global__ __launch_bounds__(kMaxTileGroup * 32) void full_attn_tile_kernel(
    const uint16_t* __restrict__ q, int64_t q_row_stride, const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache, const int32_t* __restrict__ bt, const int64_t* __restrict__ pos,
    int rows, int local_heads, int kv_heads, int block_shift, int width_shift, float scale,
    float* __restrict__ out) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint16_t sm[];
  const int r0 = blockIdx.x * kQTile;
  const int kvh = blockIdx.y;
  const int group = local_heads / kv_heads;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int head = kvh * group + warp;
  const int nthreads = blockDim.x;
  const float sl2 = scale * 1.4426950408889634f;
  // The tile's rows and their positions; the walk covers the furthest one.
  const int rg = r0 + g, rg8 = r0 + g + 8;
  const int64_t pg = rg < rows ? pos[rg] : -1, pg8 = rg8 < rows ? pos[rg8] : -1;
  int64_t cnt = 0;
  {
    int64_t p = lane < kQTile && r0 + lane < rows ? pos[r0 + lane] + 1 : 0;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) p = max(p, __shfl_xor_sync(0xffffffffu, p, o));
    cnt = p;
  }
  // Q A-fragments: this warp's head, the tile's 16 query rows.
  uint32_t qa[16][4];
  {
    const uint16_t* q0 = q + static_cast<int64_t>(r0) * q_row_stride + static_cast<int64_t>(head) * kD;
    const bool v0 = rg < rows, v1 = rg8 < rows;
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      const int c = ks * 16 + 2 * t;
      qa[ks][0] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + g * q_row_stride + c) : 0u;
      qa[ks][1] = v1 ? *reinterpret_cast<const uint32_t*>(q0 + (g + 8) * q_row_stride + c) : 0u;
      qa[ks][2] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + g * q_row_stride + c + 8) : 0u;
      qa[ks][3] = v1 ? *reinterpret_cast<const uint32_t*>(q0 + (g + 8) * q_row_stride + c + 8) : 0u;
    }
  }
  // Stage a 32-token tile into buffer `buf`. A tile never straddles a page
  // (32 divides the page), so its K and V rows are one contiguous strip:
  // one lane resolves the page once per tile and the warp shuffles the base
  // around, and every 16-byte chunk's address is base + row * width + chunk
  // (v6, 2026-10-15: the per-chunk page lookups and 64-bit address chains
  // of the per-thread form carried 26 % of the stall samples).
  const int64_t head_off = static_cast<int64_t>(kvh) * kD;
  const int tile_shift = block_shift - 5;  // pages per tile: log2(block_tokens / 32)
  const int64_t width_elems = int64_t{1} << width_shift;
  auto stage = [&](int64_t base, int buf) {
    uint16_t* kt = sm + buf * kTileStageElems;
    uint16_t* vt = kt + kKTile * kRow;
    const int64_t tile = base >> 5;
    int64_t strip = 0;
    if (lane == 0) {
      const int64_t page = static_cast<int64_t>(bt[tile >> tile_shift]);
      const int64_t in_page = (tile & ((int64_t{1} << tile_shift) - 1)) << 5;
      strip = (((page << block_shift) + in_page) << width_shift) + head_off;
    }
    strip = __shfl_sync(0xffffffffu, strip, 0);
    const int64_t valid_rows = cnt - base < kKTile ? cnt - base : kKTile;
    for (int c = threadIdx.x; c < 2 * kKTile * 32; c += nthreads) {
      const int plane = c >> 10;        // 0: K, 1: V
      const int i = (c >> 5) & 31;      // token
      const int ch = c & 31;            // 16-byte chunk of the 256-dim row
      const bool valid = i < valid_rows;
      const int64_t off = strip + i * width_elems + ch * 8;
      uint16_t* dst = (plane ? vt : kt) + i * kRow + ch * 8;
      fcp_async16_cg(dst, (plane ? v_cache : k_cache) + (valid ? off : 0), valid);
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };

  float acc[32][4];
#pragma unroll
  for (int d = 0; d < 32; ++d) acc[d][0] = acc[d][1] = acc[d][2] = acc[d][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

  const int64_t tiles = (cnt + kKTile - 1) / kKTile;
  if (tiles > 0) stage(0, 0);
  for (int64_t it = 0; it < tiles; ++it) {
    const int buf = static_cast<int>(it & 1);
    // Tile `it` has landed, and every warp is done with the other buffer
    // (tile it-1): one barrier per tile, then the next tile's loads go out
    // into that buffer and fly behind this tile's compute.
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    if (it + 1 < tiles) stage((it + 1) * kKTile, buf ^ 1);
    const uint16_t* kt = sm + buf * kTileStageElems;
    const uint16_t* vt = kt + kKTile * kRow;
    const int64_t base = it * kKTile;
    // S = Q K^T over the 32 tokens: four n8 tiles.
    float sc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      uint32_t b[4];
      const uint16_t* kp = kt + ((lane & 7) + ((lane >> 4) << 3)) * kRow + ks * 16 + ((lane >> 3) & 1) * 8;
      fldsm_x4(b, kp);
      fmma_bf16(sc[0], qa[ks], b[0], b[1]);
      fmma_bf16(sc[1], qa[ks], b[2], b[3]);
      fldsm_x4(b, kp + 16 * kRow);
      fmma_bf16(sc[2], qa[ks], b[0], b[1]);
      fmma_bf16(sc[3], qa[ks], b[2], b[3]);
    }
    // The causal mask per row (key <= pos[r]) and the tile bound.
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int64_t key = base + j * 8 + 2 * t + e;
        sc[j][e] = key <= pg ? sc[j][e] * sl2 : -INFINITY;
        sc[j][2 + e] = key <= pg8 ? sc[j][2 + e] * sl2 : -INFINITY;
        mx0 = fmaxf(mx0, sc[j][e]);
        mx1 = fmaxf(mx1, sc[j][2 + e]);
      }
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
    }
    const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
    const float a0 = fpexp(m0, n0), a1 = fpexp(m1, n1);
    m0 = n0;
    m1 = n1;
    float p[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      p[j][0] = fpexp(sc[j][0], n0);
      p[j][1] = fpexp(sc[j][1], n0);
      p[j][2] = fpexp(sc[j][2], n1);
      p[j][3] = fpexp(sc[j][3], n1);
      l0 = (j == 0 ? l0 * a0 : l0) + p[j][0] + p[j][1];
      l1 = (j == 0 ? l1 * a1 : l1) + p[j][2] + p[j][3];
    }
    const uint32_t pa[4] = {fpack_bf16(p[0][0], p[0][1]), fpack_bf16(p[0][2], p[0][3]),
                            fpack_bf16(p[1][0], p[1][1]), fpack_bf16(p[1][2], p[1][3])};
    const uint32_t pb[4] = {fpack_bf16(p[2][0], p[2][1]), fpack_bf16(p[2][2], p[2][3]),
                            fpack_bf16(p[3][0], p[3][1]), fpack_bf16(p[3][2], p[3][3])};
    if (__any_sync(0xffffffffu, a0 != 1.f || a1 != 1.f)) {
#pragma unroll
      for (int d = 0; d < 32; ++d) {
        acc[d][0] *= a0;
        acc[d][1] *= a0;
        acc[d][2] *= a1;
        acc[d][3] *= a1;
      }
    }
    // O += P V: every column pair with tokens 0-15 first, then with 16-31,
    // so an accumulator's two dependent mmas sit 32 instructions apart (the
    // interleaved order put them two apart and stalled the warp on the mma
    // latency every pair: ncu 2026-10-15). Per accumulator the order is the
    // same, tokens 0-15 then 16-31: bitwise the interleaved loop.
#pragma unroll
    for (int dp = 0; dp < 16; ++dp) {
      uint32_t b[4];
      const uint16_t* vp = vt + ((lane & 7) + ((lane >> 3) & 1) * 8) * kRow + dp * 16 + (lane >> 4) * 8;
      fldsm_x4_t(b, vp);
      fmma_bf16(acc[dp * 2], pa, b[0], b[1]);
      fmma_bf16(acc[dp * 2 + 1], pa, b[2], b[3]);
    }
#pragma unroll
    for (int dp = 0; dp < 16; ++dp) {
      uint32_t b[4];
      const uint16_t* vp = vt + ((lane & 7) + ((lane >> 3) & 1) * 8 + 16) * kRow + dp * 16 + (lane >> 4) * 8;
      fldsm_x4_t(b, vp);
      fmma_bf16(acc[dp * 2], pb, b[0], b[1]);
      fmma_bf16(acc[dp * 2 + 1], pb, b[2], b[3]);
    }
  }
#pragma unroll
  for (int o = 1; o <= 2; o <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, o);
    l1 += __shfl_xor_sync(0xffffffffu, l1, o);
  }
  const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
  float* o0 = out + (static_cast<int64_t>(rg) * local_heads + head) * kD;
  float* o1 = out + (static_cast<int64_t>(rg8) * local_heads + head) * kD;
#pragma unroll
  for (int d = 0; d < 32; ++d) {
    const int col = d * 8 + 2 * t;
    if (rg < rows) *reinterpret_cast<float2*>(o0 + col) = make_float2(acc[d][0] * i0, acc[d][1] * i0);
    if (rg8 < rows) *reinterpret_cast<float2*>(o1 + col) = make_float2(acc[d][2] * i1, acc[d][3] * i1);
  }
}

// ---- the staged tile form (TMA boxes) --------------------------------------------

constexpr int kBoxCols = 64;                               // 128 bytes: the swizzle span
constexpr int kBoxBytes = kKTile * kBoxCols * 2;           // 4 KB
constexpr int kPlaneBytes = (kD / kBoxCols) * kBoxBytes;   // 16 KB: four boxes
constexpr int kTmaStageBytes = 2 * kPlaneBytes;            // K then V
constexpr int kTmaStages = 3;                              // the ring: tile it+2 lands while `it` computes
constexpr size_t kTmaSmem = size_t(kTmaStages) * kTmaStageBytes + 1024;  // the ring + the barriers, 1 KB aligned
// The fp8 storage form's ring: e4m3 boxes of 32 x 128 codes (two a plane,
// the same 4 KB and 128-byte swizzle), widened tile by tile into one bf16
// buffer in the box layout above; the ring's slot refills once its tile is
// widened (prefetch depth three).
constexpr int kBox8Cols = 128;
constexpr int kPlane8Bytes = (kD / kBox8Cols) * kBoxBytes;  // 8 KB: two boxes
constexpr int kTma8StageBytes = 2 * kPlane8Bytes;           // K then V
constexpr size_t kTma8Smem = size_t(kTmaStages) * kTma8StageBytes + kTmaStageBytes + 1024;

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
// The CTA-scoped destination form: on sm_121a the `.shared::cluster` form
// lowers every copy to a cluster-window check, a guarded call to an address
// translation helper and ~40 instructions of register shuffling (2026-10-15
// SASS histogram: 26 CALL, 306 MOV, 114 R2UR per kernel), all on the one
// issuing thread while the other warps wait at the tile barrier; the CTA form
// is an elect loop and one UTMALDG per box (2 CALL, 85 MOV, 27 R2UR).
__device__ __forceinline__ void tma_box_2d(void* dst, const CUtensorMap* map, int col, int64_t row, uint64_t* bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               ::"r"(static_cast<unsigned>(__cvta_generic_to_shared(dst))), "l"(map), "r"(col), "r"(static_cast<int>(row)),
                 "r"(static_cast<unsigned>(__cvta_generic_to_shared(bar))) : "memory");
}
// Element (row, col) of a plane in the swizzled box layout: 16-byte chunk c
// of row r sits at chunk c ^ (r & 7) of its 128-byte line (box * kBoxBytes +
// row * 128 + ((c ^ (r & 7)) << 4) bytes); the kernel folds the XOR into
// per-lane base addresses below.

template <bool kFp8>
__global__ __launch_bounds__(kMaxTileGroup * 32) void full_attn_tile_tma_kernel(
    const __grid_constant__ CUtensorMap kmap, const __grid_constant__ CUtensorMap vmap, int64_t row_base,
    const uint16_t* __restrict__ q, int64_t q_row_stride, const int32_t* __restrict__ bt,
    const int64_t* __restrict__ pos, int rows, int local_heads, int kv_heads, int block_shift, float scale,
    const float* __restrict__ kv_scales, float* __restrict__ out) {
  dgpp_pdl_wait();
  extern __shared__ __align__(1024) uint8_t smraw[];
  // The stages need 1 KB alignment for the swizzle: carve them past the
  // first aligned boundary and put the barriers in the slack.
  const unsigned sbase = static_cast<unsigned>(__cvta_generic_to_shared(smraw));
  uint8_t* sm = smraw + ((1024 - (sbase & 1023)) & 1023);
  // fp8: the e4m3 ring, then the one bf16 tile buffer the compute reads.
  constexpr int kStageBytes = kFp8 ? kTma8StageBytes : kTmaStageBytes;
  constexpr int kBf16Off = kFp8 ? kTmaStages * kTma8StageBytes : 0;
  uint8_t* sm8 = sm;
  uint64_t* full_bar = reinterpret_cast<uint64_t*>(sm + kTmaStages * kStageBytes + (kFp8 ? kTmaStageBytes : 0));
  const float kscale = kFp8 ? scale * kv_scales[0] : scale;
  const float vs = kFp8 ? kv_scales[1] : 1.f;
  const int r0 = blockIdx.x * kQTile;
  const int kvh = blockIdx.y;
  const int group = local_heads / kv_heads;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int head = kvh * group + warp;
  const float sl2 = kscale * 1.4426950408889634f;
  const int rg = r0 + g, rg8 = r0 + g + 8;
  const int64_t pg = rg < rows ? pos[rg] : -1, pg8 = rg8 < rows ? pos[rg8] : -1;
  int64_t cnt = 0;
  {
    int64_t p = lane < kQTile && r0 + lane < rows ? pos[r0 + lane] + 1 : 0;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) p = max(p, __shfl_xor_sync(0xffffffffu, p, o));
    cnt = p;
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kTmaStages; ++s) mbar_init(&full_bar[s], 1);
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  uint32_t qa[16][4];
  {
    const uint16_t* q0 = q + static_cast<int64_t>(r0) * q_row_stride + static_cast<int64_t>(head) * kD;
    const bool v0 = rg < rows, v1 = rg8 < rows;
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      const int c = ks * 16 + 2 * t;
      qa[ks][0] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + g * q_row_stride + c) : 0u;
      qa[ks][1] = v1 ? *reinterpret_cast<const uint32_t*>(q0 + (g + 8) * q_row_stride + c) : 0u;
      qa[ks][2] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + g * q_row_stride + c + 8) : 0u;
      qa[ks][3] = v1 ? *reinterpret_cast<const uint32_t*>(q0 + (g + 8) * q_row_stride + c + 8) : 0u;
    }
  }
  // Stage tile `base`: its 32 slots are one contiguous strip of the pool, so
  // the box row is the strip's first slot (plus the layer's row base); four
  // boxes per plane, eight copies from one thread, one transaction count.
  const int tile_shift = block_shift - 5;
  const int col0 = kvh * kD;
  auto stage = [&](int64_t base, int buf) {
    if (threadIdx.x != 0) return;
    const int64_t tile = base >> 5;
    const int64_t page = static_cast<int64_t>(bt[tile >> tile_shift]);
    const int64_t slot0 = row_base + (page << block_shift) + ((tile & ((int64_t{1} << tile_shift) - 1)) << 5);
    mbar_expect_tx(&full_bar[buf], static_cast<unsigned>(kStageBytes));
    if constexpr (kFp8) {
      uint8_t* kt = sm8 + buf * kTma8StageBytes;
      uint8_t* vt = kt + kPlane8Bytes;
#pragma unroll
      for (int b = 0; b < kD / kBox8Cols; ++b) {
        tma_box_2d(kt + b * kBoxBytes, &kmap, col0 + b * kBox8Cols, slot0, &full_bar[buf]);
        tma_box_2d(vt + b * kBoxBytes, &vmap, col0 + b * kBox8Cols, slot0, &full_bar[buf]);
      }
    } else {
      uint8_t* kt = sm + buf * kTmaStageBytes;
      uint8_t* vt = kt + kPlaneBytes;
#pragma unroll
      for (int b = 0; b < kD / kBoxCols; ++b) {
        tma_box_2d(kt + b * kBoxBytes, &kmap, col0 + b * kBoxCols, slot0, &full_bar[buf]);
        tma_box_2d(vt + b * kBoxBytes, &vmap, col0 + b * kBoxCols, slot0, &full_bar[buf]);
      }
    }
  };
  // fp8: widen ring slot `buf` into the bf16 buffer. 1024 chunks of 16
  // codes a tile; a chunk's 16 bf16 fill two chunks of the bf16 box
  // (c -> box c / 4, chunks 2 (c & 3) and +1), both planes with the
  // 128-byte swizzle. Consecutive threads take consecutive rows of one
  // chunk column: eight rows see eight swizzled chunk slots, so the loads
  // and both stores are conflict-free.
  auto widen = [&](int buf) {
    const uint8_t* src8 = sm8 + buf * kTma8StageBytes;
    uint8_t* dst = sm + kBf16Off;
    for (int id = threadIdx.x; id < 2 * kKTile * (kD / 16); id += blockDim.x) {
      const int plane = id / (kKTile * (kD / 16));
      const int c = (id / kKTile) & 15;
      const int row = id & (kKTile - 1);
      const int sw = row & 7;
      const uint4 codes = *reinterpret_cast<const uint4*>(src8 + plane * kPlane8Bytes + (c >> 3) * kBoxBytes +
                                                          row * kBox8Cols + (((c & 7) ^ sw) << 4));
      uint4 lo, hi;
      fconvert_chunk16(codes, lo, hi);
      uint8_t* d = dst + plane * kPlaneBytes + (c >> 2) * kBoxBytes + row * (kBoxCols * 2);
      const int bc = 2 * (c & 3);
      *reinterpret_cast<uint4*>(d + ((bc ^ sw) << 4)) = lo;
      *reinterpret_cast<uint4*>(d + (((bc + 1) ^ sw) << 4)) = hi;
    }
  };
  // Per-lane ldmatrix bases for slot 0 with the swizzle folded in once: a
  // lane's rows share (row & 7), so XORing bits 4-6 with the chunk index
  // makes every ldmatrix address a base register plus an immediate (box,
  // row half, slot). Recomputing the swizzle per load cost two LOP3 and an
  // IMAD for each of the 64 loads a tile (SASS histogram, 2026-10-15).
  unsigned kb0[4], vb0[4];
  {
    const unsigned s0 = static_cast<unsigned>(__cvta_generic_to_shared(sm + kBf16Off));
    const int r7 = lane & 7;
    const int krow = (lane & 7) + ((lane >> 4) << 3), kc = (lane >> 3) & 1;
    const int vrow = (lane & 7) + ((lane >> 3) & 1) * 8, vc = lane >> 4;
    const unsigned kbase = (s0 + krow * (kBoxCols * 2) + (r7 << 4)) ^ (kc << 4);
    const unsigned vbase = (s0 + kPlaneBytes + vrow * (kBoxCols * 2) + (r7 << 4)) ^ (vc << 4);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      kb0[j] = kbase ^ (j << 5);
      vb0[j] = vbase ^ (j << 5);
    }
  }

  float acc[32][4];
#pragma unroll
  for (int d = 0; d < 32; ++d) acc[d][0] = acc[d][1] = acc[d][2] = acc[d][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

  const int64_t tiles = (cnt + kKTile - 1) / kKTile;
  if (tiles > 0) stage(0, 0);
  if (tiles > 1) stage(kKTile, 1);
  // The slot of tile `it` and the use parity of that slot, kept incrementally:
  // a modulo per tile lowered to a magic-constant multiply chain on the
  // issuing thread's critical path.
  int buf = 0, nbuf = 2;  // the slot of tile `it`; the slot tile it+2 goes into
  unsigned parity = 0;
  if constexpr (kFp8) {
    if (tiles > 2) stage(2 * kKTile, 2);
  }
  for (int64_t it = 0; it < tiles; ++it) {
    mbar_wait(&full_bar[buf], parity);  // tile `it` has landed
    __syncthreads();  // every warp is done with tile it-1, whose slot tile it+2 refills
    const int cur = buf;
    if constexpr (kFp8) {
      // Widen tile `it` (every warp is past tile it-1's reads of the bf16
      // buffer), then its ring slot takes tile it+3.
      widen(cur);
      __syncthreads();
      if (it + 3 < tiles) stage((it + 3) * kKTile, cur);
    } else {
      if (it + 2 < tiles) stage((it + 2) * kKTile, nbuf);
    }
    nbuf = buf;  // tile it+3 refills the slot tile `it` used
    if (++buf == kTmaStages) { buf = 0; parity ^= 1u; }
    unsigned kb[4], vb[4];
    {
      const unsigned so = kFp8 ? 0u : static_cast<unsigned>(cur) * kTmaStageBytes;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        kb[j] = kb0[j] + so;
        vb[j] = vb0[j] + so;
      }
    }
    const int64_t base = it * kKTile;
    float sc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      uint32_t b[4];
      fldsm_x4_a(b, kb[ks & 3] + (ks >> 2) * kBoxBytes);
      fmma_bf16(sc[0], qa[ks], b[0], b[1]);
      fmma_bf16(sc[1], qa[ks], b[2], b[3]);
      fldsm_x4_a(b, kb[ks & 3] + (ks >> 2) * kBoxBytes + 16 * kBoxCols * 2);
      fmma_bf16(sc[2], qa[ks], b[0], b[1]);
      fmma_bf16(sc[3], qa[ks], b[2], b[3]);
    }
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int64_t key = base + j * 8 + 2 * t + e;
        sc[j][e] = key <= pg ? sc[j][e] * sl2 : -INFINITY;
        sc[j][2 + e] = key <= pg8 ? sc[j][2 + e] * sl2 : -INFINITY;
        mx0 = fmaxf(mx0, sc[j][e]);
        mx1 = fmaxf(mx1, sc[j][2 + e]);
      }
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
    }
    const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
    const float a0 = fpexp(m0, n0), a1 = fpexp(m1, n1);
    m0 = n0;
    m1 = n1;
    float p[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      p[j][0] = fpexp(sc[j][0], n0);
      p[j][1] = fpexp(sc[j][1], n0);
      p[j][2] = fpexp(sc[j][2], n1);
      p[j][3] = fpexp(sc[j][3], n1);
      l0 = (j == 0 ? l0 * a0 : l0) + p[j][0] + p[j][1];
      l1 = (j == 0 ? l1 * a1 : l1) + p[j][2] + p[j][3];
    }
    const uint32_t pa[4] = {fpack_bf16(p[0][0], p[0][1]), fpack_bf16(p[0][2], p[0][3]),
                            fpack_bf16(p[1][0], p[1][1]), fpack_bf16(p[1][2], p[1][3])};
    const uint32_t pb[4] = {fpack_bf16(p[2][0], p[2][1]), fpack_bf16(p[2][2], p[2][3]),
                            fpack_bf16(p[3][0], p[3][1]), fpack_bf16(p[3][2], p[3][3])};
    if (__any_sync(0xffffffffu, a0 != 1.f || a1 != 1.f)) {
#pragma unroll
      for (int d = 0; d < 32; ++d) {
        acc[d][0] *= a0;
        acc[d][1] *= a0;
        acc[d][2] *= a1;
        acc[d][3] *= a1;
      }
    }
#pragma unroll
    for (int dp = 0; dp < 16; ++dp) {
      uint32_t b[4];
      fldsm_x4_ta(b, vb[dp & 3] + (dp >> 2) * kBoxBytes);
      fmma_bf16(acc[dp * 2], pa, b[0], b[1]);
      fmma_bf16(acc[dp * 2 + 1], pa, b[2], b[3]);
    }
#pragma unroll
    for (int dp = 0; dp < 16; ++dp) {
      uint32_t b[4];
      fldsm_x4_ta(b, vb[dp & 3] + (dp >> 2) * kBoxBytes + 16 * kBoxCols * 2);
      fmma_bf16(acc[dp * 2], pb, b[0], b[1]);
      fmma_bf16(acc[dp * 2 + 1], pb, b[2], b[3]);
    }
  }
#pragma unroll
  for (int o = 1; o <= 2; o <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, o);
    l1 += __shfl_xor_sync(0xffffffffu, l1, o);
  }
  const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
  float* o0 = out + (static_cast<int64_t>(rg) * local_heads + head) * kD;
  float* o1 = out + (static_cast<int64_t>(rg8) * local_heads + head) * kD;
#pragma unroll
  for (int d = 0; d < 32; ++d) {
    const int col = d * 8 + 2 * t;
    if (rg < rows) *reinterpret_cast<float2*>(o0 + col) = make_float2((acc[d][0] * i0) * vs, (acc[d][1] * i0) * vs);
    if (rg8 < rows) *reinterpret_cast<float2*>(o1 + col) = make_float2((acc[d][2] * i1) * vs, (acc[d][3] * i1) * vs);
  }
}

// ---- the fp8 compute form (engine.prefill_attention = fp8, 2026-10-16) -----------
// The same e4m3 pool, the prefill attention on the fp8 tensor cores
// (mma.m16n8k32.e4m3 issues at twice the bf16 rate on the GB10: 188 vs 94
// TFLOP/s at this occupancy). Q is quantized per row to e4m3 (amax / 448,
// the row's scale folded with scale * k_scale into the softmax multiplier),
// S = Q8 K8^T, P = exp2(s - m + 8) rounded to e4m3 (the +8 spends the code
// range: the row's maximum is the code 256 exactly and the flush floor is
// 2^-17 of it instead of 2^-9), O += P8 V8, the denominator the unrounded
// P sum (the 2^8 cancels in the ratio), the output times v_scale.
//
// Layouts, with nothing transposed in memory: a non-transposed ldmatrix of
// a K row gives the B fragment's dims 4t..4t+3 directly, so the Q A
// fragments are loaded in that order. The S accumulators sit in their
// natural layout (n-tile j: keys 8j+2t, 8j+2t+1), so P's A fragment has the
// keys {2t, 2t+1, 8+2t, 9+2t} at its k positions 4t..4t+3; a transposed
// ldmatrix of V viewed as 16-bit dim pairs gives each thread (keys 2t, 2t+1)
// x (dims 2g, 2g+1) per key octet, and one prmt per fragment splits the
// even and odd dims into two n-tiles whose k order is exactly that key set.
// Measured (scratch bench, 262K context, 4032 rows): 112 TFLOP/s against
// the bf16 tile form's 73 and the storage form's 60; the flat-attention
// check (every P the code 256) matches the bf16 chain to 5e-8 and the
// one-hot check pins the Q/K layout. Not bitwise anything: its gate is the
// recipe itself (full_attn_test's host reference) and the serving suite.
constexpr int kTmaCStages = 4;
constexpr size_t kTmaCSmem = size_t(kTmaCStages) * kTma8StageBytes + 2048;  // the ring + barriers, 1 KB aligned

__device__ __forceinline__ void fmma_e4m3(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// Four floats to four e4m3 codes, v0 in the low byte (RNE, saturating).
__device__ __forceinline__ uint32_t fpack_e4m3x4(float v0, float v1, float v2, float v3) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(v1), "f"(v0));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(v3), "f"(v2));
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}
__device__ __forceinline__ uint32_t fprmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}
__device__ __forceinline__ float fbf16lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float fbf16hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }
__device__ __forceinline__ float fpexp8(float s, float n) { return n == -INFINITY ? 0.f : exp2f(s - n + 8.f); }

__global__ __launch_bounds__(kMaxTileGroup * 32) void full_attn_tile_fp8c_kernel(
    const __grid_constant__ CUtensorMap kmap, const __grid_constant__ CUtensorMap vmap, int64_t row_base,
    const uint16_t* __restrict__ q, int64_t q_row_stride, const int32_t* __restrict__ bt,
    const int64_t* __restrict__ pos, int rows, int local_heads, int kv_heads, int block_shift, float scale,
    const float* __restrict__ kv_scales, float* __restrict__ out) {
  dgpp_pdl_wait();
  extern __shared__ __align__(1024) uint8_t smraw[];
  const unsigned sbase = static_cast<unsigned>(__cvta_generic_to_shared(smraw));
  uint8_t* sm = smraw + ((1024 - (sbase & 1023)) & 1023);
  uint64_t* full_bar = reinterpret_cast<uint64_t*>(sm + kTmaCStages * kTma8StageBytes);
  const float ks = kv_scales[0], vs = kv_scales[1];
  const int r0 = blockIdx.x * kQTile;
  const int kvh = blockIdx.y;
  const int group = local_heads / kv_heads;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int head = kvh * group + warp;
  const int rg = r0 + g, rg8 = r0 + g + 8;
  const int64_t pg = rg < rows ? pos[rg] : -1, pg8 = rg8 < rows ? pos[rg8] : -1;
  int64_t cnt = 0;
  {
    int64_t p = lane < kQTile && r0 + lane < rows ? pos[r0 + lane] + 1 : 0;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) p = max(p, __shfl_xor_sync(0xffffffffu, p, o));
    cnt = p;
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kTmaCStages; ++s) mbar_init(&full_bar[s], 1);
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();

  // Q: the row's amax over its 256 dims, then its e4m3 codes in the
  // m16n8k32 A layout (a0 / a2: row g, dims 32c + {0, 16} + 4t..; a1 / a3:
  // row g+8). Two passes over the row (the second re-reads L1-resident
  // bytes) instead of 128 floats held across the reduction.
  uint32_t qa[8][4];
  float sl2_0, sl2_1;
  {
    const uint16_t* q0 = q + static_cast<int64_t>(r0) * q_row_stride + static_cast<int64_t>(head) * kD;
    const bool v0 = rg < rows, v1 = rg8 < rows;
    const uint16_t* qr0 = q0 + g * q_row_stride;
    const uint16_t* qr1 = q0 + (g + 8) * q_row_stride;
    float am0 = 0.f, am1 = 0.f;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int d0 = c * 32 + h * 16 + 4 * t;
        if (v0) {
          const uint2 w = *reinterpret_cast<const uint2*>(qr0 + d0);
          am0 = fmaxf(am0, fmaxf(fmaxf(fabsf(fbf16lo(w.x)), fabsf(fbf16hi(w.x))),
                                 fmaxf(fabsf(fbf16lo(w.y)), fabsf(fbf16hi(w.y)))));
        }
        if (v1) {
          const uint2 w = *reinterpret_cast<const uint2*>(qr1 + d0);
          am1 = fmaxf(am1, fmaxf(fmaxf(fabsf(fbf16lo(w.x)), fabsf(fbf16hi(w.x))),
                                 fmaxf(fabsf(fbf16lo(w.y)), fabsf(fbf16hi(w.y)))));
        }
      }
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      am0 = fmaxf(am0, __shfl_xor_sync(0xffffffffu, am0, o));
      am1 = fmaxf(am1, __shfl_xor_sync(0xffffffffu, am1, o));
    }
    const float iq0 = am0 > 0.f ? 448.f / am0 : 0.f, iq1 = am1 > 0.f ? 448.f / am1 : 0.f;
    const float qs0 = am0 > 0.f ? am0 * (1.f / 448.f) : 0.f, qs1 = am1 > 0.f ? am1 * (1.f / 448.f) : 0.f;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int d0 = c * 32 + h * 16 + 4 * t;
        uint32_t a0 = 0u, a1 = 0u;
        if (v0) {
          const uint2 w = *reinterpret_cast<const uint2*>(qr0 + d0);
          a0 = fpack_e4m3x4(fbf16lo(w.x) * iq0, fbf16hi(w.x) * iq0, fbf16lo(w.y) * iq0, fbf16hi(w.y) * iq0);
        }
        if (v1) {
          const uint2 w = *reinterpret_cast<const uint2*>(qr1 + d0);
          a1 = fpack_e4m3x4(fbf16lo(w.x) * iq1, fbf16hi(w.x) * iq1, fbf16lo(w.y) * iq1, fbf16hi(w.y) * iq1);
        }
        qa[c][h * 2] = a0;
        qa[c][h * 2 + 1] = a1;
      }
    }
    const float l2e = 1.4426950408889634f;
    sl2_0 = scale * ks * qs0 * l2e;
    sl2_1 = scale * ks * qs1 * l2e;
  }

  // Stage tile `base` (the storage form's boxes: two 32 x 128 UINT8 boxes
  // a plane) into ring slot `buf`.
  const int tile_shift = block_shift - 5;
  const int col0 = kvh * kD;
  auto stage = [&](int64_t base, int buf) {
    if (threadIdx.x != 0) return;
    const int64_t tile = base >> 5;
    const int64_t page = static_cast<int64_t>(bt[tile >> tile_shift]);
    const int64_t slot0 = row_base + (page << block_shift) + ((tile & ((int64_t{1} << tile_shift) - 1)) << 5);
    mbar_expect_tx(&full_bar[buf], static_cast<unsigned>(kTma8StageBytes));
    uint8_t* kt = sm + buf * kTma8StageBytes;
    uint8_t* vt = kt + kPlane8Bytes;
#pragma unroll
    for (int b = 0; b < kD / kBox8Cols; ++b) {
      tma_box_2d(kt + b * kBoxBytes, &kmap, col0 + b * kBox8Cols, slot0, &full_bar[buf]);
      tma_box_2d(vt + b * kBoxBytes, &vmap, col0 + b * kBox8Cols, slot0, &full_bar[buf]);
    }
  };

  // Per-lane ldmatrix bases for slot 0 with the swizzle folded in (chunk c
  // of key row r sits at chunk c ^ (r & 7) of its 128-byte line). K: matrix
  // m of an x4 = key octet (m >> 1) of a 16-key half, dim half (m & 1) of a
  // 32-dim chunk c; kb[c & 3] carries the chunk bits. V (transposed): matrix
  // m = key octet m of the tile, the 16 dims of group dp at chunk dp & 7 of
  // box dp >> 3; vb[dp & 7].
  unsigned kb[4], vb[8];
  {
    const unsigned s0 = static_cast<unsigned>(__cvta_generic_to_shared(sm));
    const int r7 = lane & 7, m = lane >> 3;
    const unsigned kbase = (s0 + (8 * (m >> 1) + r7) * kBox8Cols + ((m & 1) << 4)) ^ (r7 << 4);
    const unsigned vbase = (s0 + kPlane8Bytes + (8 * m + r7) * kBox8Cols) ^ (r7 << 4);
#pragma unroll
    for (int j = 0; j < 4; ++j) kb[j] = kbase ^ (j << 5);
#pragma unroll
    for (int j = 0; j < 8; ++j) vb[j] = vbase ^ (j << 4);
  }

  float acc[32][4];
#pragma unroll
  for (int d = 0; d < 32; ++d) acc[d][0] = acc[d][1] = acc[d][2] = acc[d][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

  const int64_t tiles = (cnt + kKTile - 1) / kKTile;
#pragma unroll
  for (int s = 0; s < kTmaCStages - 1; ++s)
    if (s < tiles) stage(static_cast<int64_t>(s) * kKTile, s);
  int buf = 0;
  unsigned parity = 0;
  for (int64_t it = 0; it < tiles; ++it) {
    mbar_wait(&full_bar[buf], parity);  // tile `it` has landed
    __syncthreads();  // every warp is done with tile it-1: its slot takes tile it+kTmaCStages-1
    {
      const int prev = buf == 0 ? kTmaCStages - 1 : buf - 1;
      if (it + kTmaCStages - 1 < tiles) stage((it + kTmaCStages - 1) * kKTile, prev);
    }
    const unsigned so = static_cast<unsigned>(buf) * kTma8StageBytes;
    if (++buf == kTmaCStages) { buf = 0; parity ^= 1u; }
    const int64_t base = it * kKTile;
    float sc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      uint32_t b[4];
      fldsm_x4_a(b, kb[c & 3] + (c >> 2) * kBoxBytes + so);
      fmma_e4m3(sc[0], qa[c], b[0], b[1]);
      fmma_e4m3(sc[1], qa[c], b[2], b[3]);
      fldsm_x4_a(b, kb[c & 3] + (c >> 2) * kBoxBytes + 16 * kBox8Cols + so);
      fmma_e4m3(sc[2], qa[c], b[0], b[1]);
      fmma_e4m3(sc[3], qa[c], b[2], b[3]);
    }
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int64_t key = base + j * 8 + 2 * t + e;
        sc[j][e] = key <= pg ? sc[j][e] * sl2_0 : -INFINITY;
        sc[j][2 + e] = key <= pg8 ? sc[j][2 + e] * sl2_1 : -INFINITY;
        mx0 = fmaxf(mx0, sc[j][e]);
        mx1 = fmaxf(mx1, sc[j][2 + e]);
      }
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
    }
    const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
    const float a0 = fpexp(m0, n0), a1 = fpexp(m1, n1);
    m0 = n0;
    m1 = n1;
    float p[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      p[j][0] = fpexp8(sc[j][0], n0);
      p[j][1] = fpexp8(sc[j][1], n0);
      p[j][2] = fpexp8(sc[j][2], n1);
      p[j][3] = fpexp8(sc[j][3], n1);
      l0 = (j == 0 ? l0 * a0 : l0) + p[j][0] + p[j][1];
      l1 = (j == 0 ? l1 * a1 : l1) + p[j][2] + p[j][3];
    }
    const uint32_t pa[4] = {fpack_e4m3x4(p[0][0], p[0][1], p[1][0], p[1][1]),
                            fpack_e4m3x4(p[0][2], p[0][3], p[1][2], p[1][3]),
                            fpack_e4m3x4(p[2][0], p[2][1], p[3][0], p[3][1]),
                            fpack_e4m3x4(p[2][2], p[2][3], p[3][2], p[3][3])};
    if (__any_sync(0xffffffffu, a0 != 1.f || a1 != 1.f)) {
#pragma unroll
      for (int d = 0; d < 32; ++d) {
        acc[d][0] *= a0;
        acc[d][1] *= a0;
        acc[d][2] *= a1;
        acc[d][3] *= a1;
      }
    }
#pragma unroll
    for (int dp = 0; dp < 16; ++dp) {
      uint32_t r[4];
      fldsm_x4_ta(r, vb[dp & 7] + (dp >> 3) * kBoxBytes + so);
      fmma_e4m3(acc[2 * dp], pa, fprmt(r[0], r[1], 0x6420u), fprmt(r[2], r[3], 0x6420u));
      fmma_e4m3(acc[2 * dp + 1], pa, fprmt(r[0], r[1], 0x7531u), fprmt(r[2], r[3], 0x7531u));
    }
  }
#pragma unroll
  for (int o = 1; o <= 2; o <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, o);
    l1 += __shfl_xor_sync(0xffffffffu, l1, o);
  }
  const float i0 = (l0 > 0.f ? 1.f / l0 : 0.f) * vs, i1 = (l1 > 0.f ? 1.f / l1 : 0.f) * vs;
  float* o0 = out + (static_cast<int64_t>(rg) * local_heads + head) * kD;
  float* o1 = out + (static_cast<int64_t>(rg8) * local_heads + head) * kD;
  // The even n-tile holds dims 16dp + 2n, the odd one 16dp + 2n + 1: a
  // thread's c0 / c1 of both are dims 16dp + 4t .. 16dp + 4t + 3.
#pragma unroll
  for (int dp = 0; dp < 16; ++dp) {
    const int d = dp * 16 + 4 * t;
    if (rg < rows)
      *reinterpret_cast<float4*>(o0 + d) = make_float4(acc[2 * dp][0] * i0, acc[2 * dp + 1][0] * i0,
                                                        acc[2 * dp][1] * i0, acc[2 * dp + 1][1] * i0);
    if (rg8 < rows)
      *reinterpret_cast<float4*>(o1 + d) = make_float4(acc[2 * dp][2] * i1, acc[2 * dp + 1][2] * i1,
                                                        acc[2 * dp][3] * i1, acc[2 * dp + 1][3] * i1);
  }
}

// ---- the block decode form (2026-10-16) -----------------------------------------
// Decode rows grouped by request (the padded verify's 8-row blocks, a
// batch's spans, a lone row): one CTA per (group, KV head, key split)
// stages each 32-key K/V tile ONCE through the TMA ring for every row and
// head of the group, where the row form staged it once per (row, head
// group) warp — eight times over for a verify block, and widened the fp8
// codes as many times. Per row the arithmetic is the row form's exactly:
// 16-key steps over the row's own split range [begin, end) at 16-key
// granularity (keys outside it masked to -inf), the same max / rescale /
// bf16 P / k16 PV chain and the same partials, so a row's bits match the
// row form whatever its group — the batch-composition invariance holds and
// the two forms may mix. Groups hold at most 16 rows (the Q tile).
template <bool kFp8>
__global__ __launch_bounds__(kMaxTileGroup * 32) void full_attn_block_decode_kernel(
    const __grid_constant__ CUtensorMap kmap, const __grid_constant__ CUtensorMap vmap, int64_t row_base,
    const uint16_t* __restrict__ q, int64_t q_row_stride, const int32_t* __restrict__ req_ids,
    const int64_t* __restrict__ pos, const int32_t* __restrict__ spans, int rows,
    const int32_t* __restrict__ block_tables, int blocks_per_request, int local_heads, int kv_heads,
    int block_shift, float scale, const float* __restrict__ kv_scales, float* __restrict__ out,
    float* __restrict__ partials, int splits) {
  dgpp_pdl_wait();
  extern __shared__ __align__(1024) uint8_t smraw[];
  const unsigned sbase = static_cast<unsigned>(__cvta_generic_to_shared(smraw));
  uint8_t* sm = smraw + ((1024 - (sbase & 1023)) & 1023);
  constexpr int kStageBytes = kFp8 ? kTma8StageBytes : kTmaStageBytes;
  constexpr int kBf16Off = kFp8 ? kTmaStages * kTma8StageBytes : 0;
  uint8_t* sm8 = sm;
  uint64_t* full_bar = reinterpret_cast<uint64_t*>(sm + kTmaStages * kStageBytes + (kFp8 ? kTmaStageBytes : 0));
  const float kscale = kFp8 ? scale * kv_scales[0] : scale;
  const float vs = kFp8 ? kv_scales[1] : 1.f;
  // The group's rows [r0, rend) and request.
  int r0 = 0, nrows = rows;
  if (spans != nullptr) {
    r0 = spans[2 * blockIdx.x];
    nrows = spans[2 * blockIdx.x + 1];
  }
  const int rend = min(rows, r0 + min(nrows, kQTile));
  const int kvh = blockIdx.y;
  const int split = blockIdx.z;
  const int group = local_heads / kv_heads;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int head = kvh * group + warp;
  const float sl2 = kscale * 1.4426950408889634f;
  const int rg = r0 + g, rg8 = r0 + g + 8;
  const int64_t pg = rg < rend ? pos[rg] : -1, pg8 = rg8 < rend ? pos[rg8] : -1;
  const int32_t* bt = block_tables + static_cast<int64_t>(req_ids[r0]) * blocks_per_request;
  // The row form's split range of a row: tile-aligned chunks of ceil(cnt / splits).
  auto range = [&](int64_t p, int64_t& begin, int64_t& end) {
    const int64_t cnt = p + 1;
    const int64_t chunk = ((cnt + splits - 1) / splits + kTileTok - 1) / kTileTok * kTileTok;
    begin = static_cast<int64_t>(split) * chunk;
    end = min(cnt, begin + chunk);
  };
  int64_t b0, e0, b1, e1;
  range(pg, b0, e0);
  range(pg8, b1, e1);
  // The CTA's walk: the union of its rows' ranges, in 32-key tiles.
  int64_t lo, hi;
  {
    int64_t rb = INT64_MAX, re = 0;
    if (lane < kQTile && r0 + lane < rend) {
      int64_t b, e;
      range(pos[r0 + lane], b, e);
      if (b < e) { rb = b; re = e; }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      rb = min(rb, __shfl_xor_sync(0xffffffffu, rb, o));
      re = max(re, __shfl_xor_sync(0xffffffffu, re, o));
    }
    lo = rb;
    hi = re;
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kTmaStages; ++s) mbar_init(&full_bar[s], 1);
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  uint32_t qa[16][4];
  {
    const uint16_t* q0 = q + static_cast<int64_t>(r0) * q_row_stride + static_cast<int64_t>(head) * kD;
    const bool v0 = rg < rend, v1 = rg8 < rend;
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      const int c = ks * 16 + 2 * t;
      qa[ks][0] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + g * q_row_stride + c) : 0u;
      qa[ks][1] = v1 ? *reinterpret_cast<const uint32_t*>(q0 + (g + 8) * q_row_stride + c) : 0u;
      qa[ks][2] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + g * q_row_stride + c + 8) : 0u;
      qa[ks][3] = v1 ? *reinterpret_cast<const uint32_t*>(q0 + (g + 8) * q_row_stride + c + 8) : 0u;
    }
  }
  const int tile_shift = block_shift - 5;
  const int col0 = kvh * kD;
  auto stage = [&](int64_t base, int buf) {
    if (threadIdx.x != 0) return;
    const int64_t tile = base >> 5;
    const int64_t page = static_cast<int64_t>(bt[tile >> tile_shift]);
    const int64_t slot0 = row_base + (page << block_shift) + ((tile & ((int64_t{1} << tile_shift) - 1)) << 5);
    mbar_expect_tx(&full_bar[buf], static_cast<unsigned>(kStageBytes));
    if constexpr (kFp8) {
      uint8_t* kt = sm8 + buf * kTma8StageBytes;
      uint8_t* vt = kt + kPlane8Bytes;
#pragma unroll
      for (int b = 0; b < kD / kBox8Cols; ++b) {
        tma_box_2d(kt + b * kBoxBytes, &kmap, col0 + b * kBox8Cols, slot0, &full_bar[buf]);
        tma_box_2d(vt + b * kBoxBytes, &vmap, col0 + b * kBox8Cols, slot0, &full_bar[buf]);
      }
    } else {
      uint8_t* kt = sm + buf * kTmaStageBytes;
      uint8_t* vt = kt + kPlaneBytes;
#pragma unroll
      for (int b = 0; b < kD / kBoxCols; ++b) {
        tma_box_2d(kt + b * kBoxBytes, &kmap, col0 + b * kBoxCols, slot0, &full_bar[buf]);
        tma_box_2d(vt + b * kBoxBytes, &vmap, col0 + b * kBoxCols, slot0, &full_bar[buf]);
      }
    }
  };
  auto widen = [&](int buf) {
    const uint8_t* src8 = sm8 + buf * kTma8StageBytes;
    uint8_t* dst = sm + kBf16Off;
    for (int id = threadIdx.x; id < 2 * kKTile * (kD / 16); id += blockDim.x) {
      const int plane = id / (kKTile * (kD / 16));
      const int c = (id / kKTile) & 15;
      const int row = id & (kKTile - 1);
      const int sw = row & 7;
      const uint4 codes = *reinterpret_cast<const uint4*>(src8 + plane * kPlane8Bytes + (c >> 3) * kBoxBytes +
                                                          row * kBox8Cols + (((c & 7) ^ sw) << 4));
      uint4 lo8, hi8;
      fconvert_chunk16(codes, lo8, hi8);
      uint8_t* d = dst + plane * kPlaneBytes + (c >> 2) * kBoxBytes + row * (kBoxCols * 2);
      const int bc = 2 * (c & 3);
      *reinterpret_cast<uint4*>(d + ((bc ^ sw) << 4)) = lo8;
      *reinterpret_cast<uint4*>(d + (((bc + 1) ^ sw) << 4)) = hi8;
    }
  };
  unsigned kb0[4], vb0[4];
  {
    const unsigned s0 = static_cast<unsigned>(__cvta_generic_to_shared(sm + kBf16Off));
    const int r7 = lane & 7;
    const int krow = (lane & 7) + ((lane >> 4) << 3), kc = (lane >> 3) & 1;
    const int vrow = (lane & 7) + ((lane >> 3) & 1) * 8, vc = lane >> 4;
    const unsigned kbase = (s0 + krow * (kBoxCols * 2) + (r7 << 4)) ^ (kc << 4);
    const unsigned vbase = (s0 + kPlaneBytes + vrow * (kBoxCols * 2) + (r7 << 4)) ^ (vc << 4);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      kb0[j] = kbase ^ (j << 5);
      vb0[j] = vbase ^ (j << 5);
    }
  }
  float acc[32][4];
#pragma unroll
  for (int d = 0; d < 32; ++d) acc[d][0] = acc[d][1] = acc[d][2] = acc[d][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

  const int64_t t0 = lo < hi ? lo / kKTile : 0;
  const int64_t tiles = lo < hi ? (hi + kKTile - 1) / kKTile - t0 : 0;
  if (tiles > 0) stage(t0 * kKTile, 0);
  if (tiles > 1) stage((t0 + 1) * kKTile, 1);
  if constexpr (kFp8) {
    if (tiles > 2) stage((t0 + 2) * kKTile, 2);
  }
  int buf = 0, nbuf = 2;
  unsigned parity = 0;
  for (int64_t it = 0; it < tiles; ++it) {
    mbar_wait(&full_bar[buf], parity);
    __syncthreads();
    const int cur = buf;
    if constexpr (kFp8) {
      widen(cur);
      __syncthreads();
      if (it + 3 < tiles) stage((t0 + it + 3) * kKTile, cur);
    } else {
      if (it + 2 < tiles) stage((t0 + it + 2) * kKTile, nbuf);
    }
    nbuf = buf;
    if (++buf == kTmaStages) { buf = 0; parity ^= 1u; }
    unsigned kb[4], vb[4];
    {
      const unsigned so = kFp8 ? 0u : static_cast<unsigned>(cur) * kTmaStageBytes;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        kb[j] = kb0[j] + so;
        vb[j] = vb0[j] + so;
      }
    }
    // Two 16-key steps a tile: the row form's chain, step by step.
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int64_t base = (t0 + it) * kKTile + h * kTileTok;
      const unsigned ho = static_cast<unsigned>(h) * (kTileTok * kBoxCols * 2);
      float sc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
      for (int ks = 0; ks < 16; ++ks) {
        uint32_t b[4];
        fldsm_x4_a(b, kb[ks & 3] + (ks >> 2) * kBoxBytes + ho);
        fmma_bf16(sc[0], qa[ks], b[0], b[1]);
        fmma_bf16(sc[1], qa[ks], b[2], b[3]);
      }
      float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int64_t key = base + j * 8 + 2 * t + e;
          sc[j][e] = (key >= b0 && key < e0) ? sc[j][e] * sl2 : -INFINITY;
          sc[j][2 + e] = (key >= b1 && key < e1) ? sc[j][2 + e] * sl2 : -INFINITY;
          mx0 = fmaxf(mx0, sc[j][e]);
          mx1 = fmaxf(mx1, sc[j][2 + e]);
        }
      }
#pragma unroll
      for (int o = 1; o <= 2; o <<= 1) {
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
      }
      const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
      const float a0 = fpexp(m0, n0), a1 = fpexp(m1, n1);
      m0 = n0;
      m1 = n1;
      float p[2][4];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        p[j][0] = fpexp(sc[j][0], n0);
        p[j][1] = fpexp(sc[j][1], n0);
        p[j][2] = fpexp(sc[j][2], n1);
        p[j][3] = fpexp(sc[j][3], n1);
      }
      l0 = l0 * a0 + p[0][0] + p[0][1] + p[1][0] + p[1][1];
      l1 = l1 * a1 + p[0][2] + p[0][3] + p[1][2] + p[1][3];
      const uint32_t pa[4] = {fpack_bf16(p[0][0], p[0][1]), fpack_bf16(p[0][2], p[0][3]),
                              fpack_bf16(p[1][0], p[1][1]), fpack_bf16(p[1][2], p[1][3])};
      if (__any_sync(0xffffffffu, a0 != 1.f || a1 != 1.f)) {  // x 1 is the identity: skipping it keeps the bits
#pragma unroll
        for (int d = 0; d < 32; ++d) {
          acc[d][0] *= a0;
          acc[d][1] *= a0;
          acc[d][2] *= a1;
          acc[d][3] *= a1;
        }
      }
#pragma unroll
      for (int dp = 0; dp < 16; ++dp) {
        uint32_t b[4];
        fldsm_x4_ta(b, vb[dp & 3] + (dp >> 2) * kBoxBytes + ho);
        fmma_bf16(acc[dp * 2], pa, b[0], b[1]);
        fmma_bf16(acc[dp * 2 + 1], pa, b[2], b[3]);
      }
    }
  }
#pragma unroll
  for (int o = 1; o <= 2; o <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, o);
    l1 += __shfl_xor_sync(0xffffffffu, l1, o);
  }
  if (partials == nullptr) {
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
    float* o0 = out + (static_cast<int64_t>(rg) * local_heads + head) * kD;
    float* o1 = out + (static_cast<int64_t>(rg8) * local_heads + head) * kD;
#pragma unroll
    for (int d = 0; d < 32; ++d) {
      const int col = d * 8 + 2 * t;
      if (rg < rend) *reinterpret_cast<float2*>(o0 + col) = make_float2((acc[d][0] * i0) * vs, (acc[d][1] * i0) * vs);
      if (rg8 < rend)
        *reinterpret_cast<float2*>(o1 + col) = make_float2((acc[d][2] * i1) * vs, (acc[d][3] * i1) * vs);
    }
    return;
  }
  // The partials of (row, KV head, split) hold the group's heads by their
  // index within the group (`warp`), as the row form writes and the
  // combine reads them — not the global head.
  float* part0 = partials + ((static_cast<int64_t>(rg) * kv_heads + kvh) * splits + split) * kPartialFloats;
  float* part1 = partials + ((static_cast<int64_t>(rg8) * kv_heads + kvh) * splits + split) * kPartialFloats;
  float* pa0 = part0 + warp * kD;
  float* pa1 = part1 + warp * kD;
#pragma unroll
  for (int d = 0; d < 32; ++d) {
    const int col = d * 8 + 2 * t;
    if (rg < rend) *reinterpret_cast<float2*>(pa0 + col) = make_float2(acc[d][0], acc[d][1]);
    if (rg8 < rend) *reinterpret_cast<float2*>(pa1 + col) = make_float2(acc[d][2], acc[d][3]);
  }
  if (t == 0) {
    if (rg < rend) {
      part0[16 * kD + warp] = m0;
      part0[16 * kD + 16 + warp] = l0;
    }
    if (rg8 < rend) {
      part1[16 * kD + warp] = m1;
      part1[16 * kD + 16 + warp] = l1;
    }
  }
}

// Each kernel opts into its shared memory at its own first launch (a
// thread-safe static per instantiation). Registering every instantiation
// at once — the unlaunched fp8 and block forms included — left a later
// DeepSeek-V4 test in the same process with a sticky illegal address
// (2026-10-16, shared_prefill_test: 20/20 without those registrations,
// 7 failures with them); a kernel's attributes are set only where it runs.
template <typename K>
void opt_in_smem(K kernel, size_t bytes) {
  DGPP_CUDA_OK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
  DGPP_CUDA_OK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
}
template <bool kFp8>
void row_attrs_once() {
  static const bool done = [] {
    opt_in_smem(full_attn_row_kernel<kFp8>, kRowSmem + (kFp8 ? kRow8StageBytes : 0));
    return true;
  }();
  (void)done;
}
void tile_attrs_once() {
  static const bool done = [] {
    opt_in_smem(full_attn_tile_kernel, kTileSmem);
    return true;
  }();
  (void)done;
}
template <bool kFp8>
void tile_tma_attrs_once() {
  static const bool done = [] {
    opt_in_smem(full_attn_tile_tma_kernel<kFp8>, kFp8 ? kTma8Smem : kTmaSmem);
    return true;
  }();
  (void)done;
}
void tile_fp8c_attrs_once() {
  static const bool done = [] {
    opt_in_smem(full_attn_tile_fp8c_kernel, kTmaCSmem);
    return true;
  }();
  (void)done;
}
template <bool kFp8>
void block_attrs_once() {
  static const bool done = [] {
    opt_in_smem(full_attn_block_decode_kernel<kFp8>, kFp8 ? kTma8Smem : kTmaSmem);
    return true;
  }();
  (void)done;
}

}  // namespace

bool full_attn_supported(int dim, int local_heads, int kv_heads) {
  return dim == kD && kv_heads > 0 && local_heads % kv_heads == 0 && local_heads / kv_heads <= 16;
}

int full_attn_max_splits() { return kMaxSplits; }

size_t full_attn_partials_bytes(int rows, int kv_heads) {
  return static_cast<size_t>(rows) * kv_heads * kMaxSplits * kPartialFloats * sizeof(float);
}

namespace {
template <bool kFp8>
void decode_rows(const uint16_t* q, int64_t q_row_stride, const void* k_cache, const void* v_cache,
                 const int32_t* req_ids, const int64_t* pos, int rows, int local_heads, int kv_heads,
                 int block_tokens, const int32_t* block_tables, int blocks_per_request, float scale,
                 const float* kv_scales, float* out, float* partials, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!full_attn_supported(kD, local_heads, kv_heads))
    throw std::invalid_argument("full_attn_decode: dim 256, <= 16 query heads per KV head");
  if (kFp8 && kv_scales == nullptr) throw std::invalid_argument("full_attn_decode_fp8: null kv scales");
  row_attrs_once<kFp8>();
  const int splits = partials != nullptr ? kMaxSplits : 1;
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(kv_heads), static_cast<unsigned>(splits));
  full_attn_row_kernel<kFp8><<<grid, 32, kRowSmem + (kFp8 ? kRow8StageBytes : 0), stream>>>(
      q, q_row_stride, k_cache, v_cache, req_ids, pos, local_heads, kv_heads, block_tokens, block_tables,
      blocks_per_request, scale, kv_scales, out, partials, splits);
  DGPP_CUDA_OK(cudaGetLastError());
  if (splits > 1) {
    // The v scale lands once, on the combined output (the partials stay
    // unscaled); the bf16 form's scale of 1 leaves its bits as they were.
    const dim3 cgrid(static_cast<unsigned>(rows), static_cast<unsigned>(kv_heads));
    full_attn_combine_kernel<<<cgrid, 256, 0, stream>>>(partials, splits, local_heads, kv_heads,
                                                        kFp8 ? kv_scales + 1 : nullptr, out);
    DGPP_CUDA_OK(cudaGetLastError());
  }
}
}  // namespace

void full_attn_decode(const uint16_t* q, int64_t q_row_stride, const uint16_t* k_cache,
                      const uint16_t* v_cache, const int32_t* req_ids, const int64_t* pos, int rows,
                      int local_heads, int kv_heads, int block_tokens, const int32_t* block_tables,
                      int blocks_per_request, float scale, float* out, float* partials,
                      cudaStream_t stream) {
  decode_rows<false>(q, q_row_stride, k_cache, v_cache, req_ids, pos, rows, local_heads, kv_heads, block_tokens,
                     block_tables, blocks_per_request, scale, nullptr, out, partials, stream);
}

void full_attn_decode_fp8(const uint16_t* q, int64_t q_row_stride, const uint8_t* k_cache,
                          const uint8_t* v_cache, const int32_t* req_ids, const int64_t* pos, int rows,
                          int local_heads, int kv_heads, int block_tokens, const int32_t* block_tables,
                          int blocks_per_request, float scale, const float* kv_scales, float* out,
                          float* partials, cudaStream_t stream) {
  decode_rows<true>(q, q_row_stride, k_cache, v_cache, req_ids, pos, rows, local_heads, kv_heads, block_tokens,
                    block_tables, blocks_per_request, scale, kv_scales, out, partials, stream);
}

bool full_attn_tma_supported(int local_heads, int kv_heads, int block_tokens) {
  return full_attn_supported(kD, local_heads, kv_heads) && local_heads / kv_heads <= kMaxTileGroup &&
         block_tokens >= kKTile && (block_tokens & (block_tokens - 1)) == 0;
}

FullAttnStageMaps full_attn_stage_maps(const uint16_t* k_cache, const uint16_t* v_cache, int64_t rows,
                                       int kv_heads) {
  if (rows <= 0 || kv_heads <= 0) throw std::invalid_argument("full_attn_stage_maps: rows and kv_heads must be positive");
  const int width = kv_heads * kD;
  FullAttnStageMaps maps;
  for (int plane = 0; plane < 2; ++plane) {
    const cuuint64_t dims[2] = {static_cast<cuuint64_t>(width), static_cast<cuuint64_t>(rows)};
    const cuuint64_t strides[1] = {static_cast<cuuint64_t>(width) * 2};
    const cuuint32_t box[2] = {static_cast<cuuint32_t>(kBoxCols), static_cast<cuuint32_t>(kKTile)};
    const cuuint32_t estr[2] = {1, 1};
    const CUresult r = cuTensorMapEncodeTiled(
        plane ? &maps.v : &maps.k, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        const_cast<uint16_t*>(plane ? v_cache : k_cache), dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS)
      throw std::runtime_error("full_attn_stage_maps: cuTensorMapEncodeTiled failed (" + std::to_string(static_cast<int>(r)) + ")");
  }
  return maps;
}

FullAttnStageMaps full_attn_stage_maps_fp8(const uint8_t* k_cache, const uint8_t* v_cache, int64_t rows,
                                           int kv_heads) {
  if (rows <= 0 || kv_heads <= 0)
    throw std::invalid_argument("full_attn_stage_maps_fp8: rows and kv_heads must be positive");
  const int width = kv_heads * kD;
  FullAttnStageMaps maps;
  for (int plane = 0; plane < 2; ++plane) {
    const cuuint64_t dims[2] = {static_cast<cuuint64_t>(width), static_cast<cuuint64_t>(rows)};
    const cuuint64_t strides[1] = {static_cast<cuuint64_t>(width)};
    const cuuint32_t box[2] = {static_cast<cuuint32_t>(kBox8Cols), static_cast<cuuint32_t>(kKTile)};
    const cuuint32_t estr[2] = {1, 1};
    const CUresult r = cuTensorMapEncodeTiled(
        plane ? &maps.v : &maps.k, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<uint8_t*>(plane ? v_cache : k_cache),
        dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS)
      throw std::runtime_error("full_attn_stage_maps_fp8: cuTensorMapEncodeTiled failed (" +
                               std::to_string(static_cast<int>(r)) + ")");
  }
  return maps;
}

namespace {
template <bool kFp8>
void prefill_tma(const FullAttnStageMaps& maps, int64_t row_base, const uint16_t* q, int64_t q_row_stride,
                 const int32_t* block_table, const int64_t* pos, int rows, int local_heads, int kv_heads,
                 int block_tokens, float scale, const float* kv_scales, float* out, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!full_attn_tma_supported(local_heads, kv_heads, block_tokens))
    throw std::invalid_argument("full_attn_prefill_tma: geometry outside the staged form (full_attn_tma_supported)");
  if (kFp8 && kv_scales == nullptr) throw std::invalid_argument("full_attn_prefill_tma_fp8: null kv scales");
  tile_tma_attrs_once<kFp8>();
  const int group = local_heads / kv_heads;
  const int block_shift = __builtin_ctz(static_cast<unsigned>(block_tokens));
  const dim3 grid(static_cast<unsigned>((rows + kQTile - 1) / kQTile), static_cast<unsigned>(kv_heads));
  full_attn_tile_tma_kernel<kFp8><<<grid, group * 32, kFp8 ? kTma8Smem : kTmaSmem, stream>>>(
      maps.k, maps.v, row_base, q, q_row_stride, block_table, pos, rows, local_heads, kv_heads, block_shift, scale,
      kv_scales, out);
  DGPP_CUDA_OK(cudaGetLastError());
}
}  // namespace

namespace {
template <bool kFp8>
void decode_blocks(const FullAttnStageMaps& maps, int64_t row_base, const uint16_t* q, int64_t q_row_stride,
                   const int32_t* req_ids, const int64_t* pos, const int32_t* spans, int groups, int rows,
                   const int32_t* block_tables, int blocks_per_request, int local_heads, int kv_heads,
                   int block_tokens, float scale, const float* kv_scales, float* out, float* partials,
                   cudaStream_t stream) {
  if (rows <= 0) return;
  if (!full_attn_tma_supported(local_heads, kv_heads, block_tokens))
    throw std::invalid_argument("full_attn_decode_blocks: geometry outside the staged form (full_attn_tma_supported)");
  if (spans == nullptr && rows > kQTile)
    throw std::invalid_argument("full_attn_decode_blocks: a lone group holds at most 16 rows");
  if (spans != nullptr && groups <= 0) throw std::invalid_argument("full_attn_decode_blocks: no groups");
  if (kFp8 && kv_scales == nullptr) throw std::invalid_argument("full_attn_decode_blocks_fp8: null kv scales");
  block_attrs_once<kFp8>();
  const int group = local_heads / kv_heads;
  const int block_shift = __builtin_ctz(static_cast<unsigned>(block_tokens));
  const int splits = partials != nullptr ? kMaxSplits : 1;
  const dim3 grid(static_cast<unsigned>(spans != nullptr ? groups : 1), static_cast<unsigned>(kv_heads),
                  static_cast<unsigned>(splits));
  full_attn_block_decode_kernel<kFp8><<<grid, group * 32, kFp8 ? kTma8Smem : kTmaSmem, stream>>>(
      maps.k, maps.v, row_base, q, q_row_stride, req_ids, pos, spans, rows, block_tables, blocks_per_request,
      local_heads, kv_heads, block_shift, scale, kv_scales, out, partials, splits);
  DGPP_CUDA_OK(cudaGetLastError());
  if (splits > 1) {
    const dim3 cgrid(static_cast<unsigned>(rows), static_cast<unsigned>(kv_heads));
    full_attn_combine_kernel<<<cgrid, 256, 0, stream>>>(partials, splits, local_heads, kv_heads,
                                                        kFp8 ? kv_scales + 1 : nullptr, out);
    DGPP_CUDA_OK(cudaGetLastError());
  }
}
}  // namespace

void full_attn_decode_blocks(const FullAttnStageMaps& maps, int64_t row_base, const uint16_t* q,
                             int64_t q_row_stride, const int32_t* req_ids, const int64_t* pos,
                             const int32_t* spans, int groups, int rows, const int32_t* block_tables,
                             int blocks_per_request, int local_heads, int kv_heads, int block_tokens, float scale,
                             float* out, float* partials, cudaStream_t stream) {
  decode_blocks<false>(maps, row_base, q, q_row_stride, req_ids, pos, spans, groups, rows, block_tables,
                       blocks_per_request, local_heads, kv_heads, block_tokens, scale, nullptr, out, partials, stream);
}

void full_attn_decode_blocks_fp8(const FullAttnStageMaps& maps, int64_t row_base, const uint16_t* q,
                                 int64_t q_row_stride, const int32_t* req_ids, const int64_t* pos,
                                 const int32_t* spans, int groups, int rows, const int32_t* block_tables,
                                 int blocks_per_request, int local_heads, int kv_heads, int block_tokens,
                                 float scale, const float* kv_scales, float* out, float* partials,
                                 cudaStream_t stream) {
  decode_blocks<true>(maps, row_base, q, q_row_stride, req_ids, pos, spans, groups, rows, block_tables,
                      blocks_per_request, local_heads, kv_heads, block_tokens, scale, kv_scales, out, partials, stream);
}

void full_attn_prefill_tma(const FullAttnStageMaps& maps, int64_t row_base, const uint16_t* q,
                           int64_t q_row_stride, const int32_t* block_table, const int64_t* pos, int rows,
                           int local_heads, int kv_heads, int block_tokens, float scale, float* out,
                           cudaStream_t stream) {
  prefill_tma<false>(maps, row_base, q, q_row_stride, block_table, pos, rows, local_heads, kv_heads, block_tokens,
                     scale, nullptr, out, stream);
}

void full_attn_prefill_tma_fp8(const FullAttnStageMaps& maps, int64_t row_base, const uint16_t* q,
                               int64_t q_row_stride, const int32_t* block_table, const int64_t* pos, int rows,
                               int local_heads, int kv_heads, int block_tokens, float scale,
                               const float* kv_scales, float* out, cudaStream_t stream) {
  prefill_tma<true>(maps, row_base, q, q_row_stride, block_table, pos, rows, local_heads, kv_heads, block_tokens,
                    scale, kv_scales, out, stream);
}

void full_attn_prefill_tma_fp8_compute(const FullAttnStageMaps& maps, int64_t row_base, const uint16_t* q,
                                       int64_t q_row_stride, const int32_t* block_table, const int64_t* pos,
                                       int rows, int local_heads, int kv_heads, int block_tokens, float scale,
                                       const float* kv_scales, float* out, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!full_attn_tma_supported(local_heads, kv_heads, block_tokens))
    throw std::invalid_argument(
        "full_attn_prefill_tma_fp8_compute: geometry outside the staged form (full_attn_tma_supported)");
  if (kv_scales == nullptr) throw std::invalid_argument("full_attn_prefill_tma_fp8_compute: null kv scales");
  tile_fp8c_attrs_once();
  const int group = local_heads / kv_heads;
  const int block_shift = __builtin_ctz(static_cast<unsigned>(block_tokens));
  const dim3 grid(static_cast<unsigned>((rows + kQTile - 1) / kQTile), static_cast<unsigned>(kv_heads));
  full_attn_tile_fp8c_kernel<<<grid, group * 32, kTmaCSmem, stream>>>(maps.k, maps.v, row_base, q, q_row_stride,
                                                                       block_table, pos, rows, local_heads, kv_heads,
                                                                       block_shift, scale, kv_scales, out);
  DGPP_CUDA_OK(cudaGetLastError());
}

void full_attn_prefill_warp(const uint16_t* q, int64_t q_row_stride, const uint16_t* k_cache,
                            const uint16_t* v_cache, const int32_t* req_ids, const int64_t* pos,
                            int rows, int local_heads, int kv_heads, int block_tokens,
                            const int32_t* block_tables, int blocks_per_request, float scale,
                            float* out, cudaStream_t stream) {
  full_attn_decode(q, q_row_stride, k_cache, v_cache, req_ids, pos, rows, local_heads, kv_heads, block_tokens,
                   block_tables, blocks_per_request, scale, out, nullptr, stream);
}

void full_attn_prefill(const uint16_t* q, int64_t q_row_stride, const uint16_t* k_cache,
                       const uint16_t* v_cache, const int32_t* block_table, const int32_t* req_ids,
                       const int64_t* pos, int rows, int local_heads, int kv_heads, int block_tokens,
                       const int32_t* block_tables, int blocks_per_request, float scale, float* out,
                       cudaStream_t stream) {
  if (rows <= 0) return;
  if (!full_attn_supported(kD, local_heads, kv_heads))
    throw std::invalid_argument("full_attn_prefill: dim 256, <= 16 query heads per KV head");
  const int group = local_heads / kv_heads;
  if (group > kMaxTileGroup) {  // wider groups keep the row form
    full_attn_decode(q, q_row_stride, k_cache, v_cache, req_ids, pos, rows, local_heads, kv_heads, block_tokens,
                     block_tables, blocks_per_request, scale, out, nullptr, stream);
    return;
  }
  tile_attrs_once();
  // The strip staging wants a page of >= 32 tokens and a K/V row width that are
  // powers of two (the serving pool: 64-token pages, 1/2/4 KV heads x 256);
  // any other geometry walks the row form like the wider groups do.
  const int width = kv_heads * kD;
  if (block_tokens < kKTile || (block_tokens & (block_tokens - 1)) != 0 || (width & (width - 1)) != 0) {
    full_attn_decode(q, q_row_stride, k_cache, v_cache, req_ids, pos, rows, local_heads, kv_heads, block_tokens,
                     block_tables, blocks_per_request, scale, out, nullptr, stream);
    return;
  }
  const int block_shift = __builtin_ctz(static_cast<unsigned>(block_tokens));
  const int width_shift = __builtin_ctz(static_cast<unsigned>(width));
  const dim3 grid(static_cast<unsigned>((rows + kQTile - 1) / kQTile), static_cast<unsigned>(kv_heads));
  full_attn_tile_kernel<<<grid, group * 32, kTileSmem, stream>>>(q, q_row_stride, k_cache, v_cache, block_table,
                                                                  pos, rows, local_heads, kv_heads, block_shift,
                                                                  width_shift, scale, out);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
