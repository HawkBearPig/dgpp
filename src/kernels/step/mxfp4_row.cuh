#pragma once
// The MXFP4 expert consumer over the checkpoint's row-major layout
// (MOONSHOT-1.md section 2.4's layout rule applied to the routed experts
// too): the same unit, group and tensor-core form as mxfp4_expert.cuh, with
// a lane's sixteen bytes being one 32-code chunk of one row rather than two
// chunks of a tile-major row, and the L2 bridge of fp8_dense.cuh ahead of
// the fragment loads: a bulk prefetch of each row's next 512-byte slab, so
// the 32-byte runs the warp's load would show the memory are L2 hits. If
// this form holds the ceiling the arena needs no second layout.
#include <cstdint>

#include "kernels/step/fp8_dense.cuh"
#include "kernels/step/mxfp4_expert.cuh"
#include "kernels/step/primitives.cuh"

namespace dgpp::step {

#if defined(__CUDACC__)

// Lane l holds chunk (2 c + (l & 1)) of row l / 2 as four words of eight
// codes; the fragment owner (gid, tid) of chunk pair c takes word tid of
// rows gid and gid + 8 from lanes 2 gid + h and 2 gid + 16 + h for half h.
__device__ __forceinline__ void mxfp4_row_fragments(const uint4& v, int gid, int tid, int h, uint32_t& a, uint32_t& b) {
  const unsigned full = 0xffffffffu;
  const int src0 = 2 * gid + h, src1 = src0 + 16;
  a = 0;
  b = 0;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const uint32_t vw = w == 0 ? v.x : w == 1 ? v.y : w == 2 ? v.z : v.w;
    const uint32_t aa = __shfl_sync(full, vw, src0), bb = __shfl_sync(full, vw, src1);
    if (w == tid) { a = aa; b = bb; }
  }
}

template <int kH, int kI, int kTG>
__device__ void mxfp4_row_gate_up_unit(const Mxfp4Expert& e, const Mxfp4Group& g, int rb, const uint16_t* __restrict__ x,
                                       uint16_t* __restrict__ act, uint8_t* smem) {
  static_assert(kH % kMxfp4WindowCodes == 0 && kI % kTileRows == 0, "the pair's shape");
  constexpr int kD = 4;                        // chunk pairs in flight a matrix
  constexpr int kPairs = kH / 64;              // chunk pairs a row
  constexpr int kWindows = kH / kMxfp4WindowCodes;
  constexpr int kRowBytes = kH / 2;
  constexpr int kScaleBytes = kH / kChunkCodes;
  constexpr int kPairsPerSlab = kFp8SlabBytes / 32;  // a slab holds 16 pairs of a row
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tid = lane & 3;
  const int n0 = rb * kMxfp4UnitRows + warp * kTileRows, r0 = n0 + gid, r1 = r0 + 8;
  const bool live = n0 < kI;
  const int row_l = (live ? n0 : 0) + (lane >> 1), half = lane & 1;
  const uint8_t* gp = e.gate + static_cast<size_t>(row_l) * kRowBytes + half * 16;
  const uint8_t* up = e.up + static_cast<size_t>(row_l) * kRowBytes + half * 16;
  const int rr0 = live ? r0 : 0, rr1 = live ? r1 : 0;
  const unsigned long long* sg0 = reinterpret_cast<const unsigned long long*>(e.gate_scales + static_cast<size_t>(rr0) * kScaleBytes);
  const unsigned long long* sg1 = reinterpret_cast<const unsigned long long*>(e.gate_scales + static_cast<size_t>(rr1) * kScaleBytes);
  const unsigned long long* su0 = reinterpret_cast<const unsigned long long*>(e.up_scales + static_cast<size_t>(rr0) * kScaleBytes);
  const unsigned long long* su1 = reinterpret_cast<const unsigned long long*>(e.up_scales + static_cast<size_t>(rr1) * kScaleBytes);
  auto bridge = [&](int slab) {
    if (slab >= kRowBytes / kFp8SlabBytes) return;
    const int r = (live ? n0 : 0) + (lane & 15);
    const uint8_t* m = lane < 16 ? e.gate : e.up;
    prefetch_l2_bulk(m + static_cast<size_t>(r) * kRowBytes + static_cast<size_t>(slab) * kFp8SlabBytes, kFp8SlabBytes);
  };
  bridge(0);
  bridge(1);
  uint4 ring[kD][2];
#pragma unroll
  for (int i = 0; i < kD; ++i) {
    ring[i][0] = __ldcs(reinterpret_cast<const uint4*>(gp + i * 32));
    ring[i][1] = __ldcs(reinterpret_cast<const uint4*>(up + i * 32));
  }
  unsigned long long g0 = __ldg(sg0), g1 = __ldg(sg1), u0 = __ldg(su0), u1 = __ldg(su1);
  unsigned long long ng0 = __ldg(sg0 + 1), ng1 = __ldg(sg1 + 1), nu0 = __ldg(su0 + 1), nu1 = __ldg(su1 + 1);
  mxfp4_fill_window<kH>(x, g, 0, 0, smem);
  mxfp4_fill_window<kH>(x, g, 1, 1, smem);
  float acc_g[kTG][4], acc_u[kTG][4];
#pragma unroll
  for (int t = 0; t < kTG; ++t)
    for (int i = 0; i < 4; ++i) { acc_g[t][i] = 0.f; acc_u[t][i] = 0.f; }
#pragma unroll 1
  for (int w = 0; w < kWindows; ++w) {
    cp_async_wait<1>();
    __syncthreads();
    if (w + 2 < kWindows) mxfp4_fill_window<kH>(x, g, w + 2, (w + 2) % kMxfp4Windows, smem); else cp_async_commit();
    const uint8_t* xb = smem + (w % kMxfp4Windows) * kMxfp4WindowBytes;
#pragma unroll
    for (int pq = 0; pq < kMxfp4WindowCodes / 64; ++pq) {
      const int pair = w * (kMxfp4WindowCodes / 64) + pq;
      if (pair % kPairsPerSlab == 0 && pair > 0) bridge(pair / kPairsPerSlab + 1);
      const uint4 vg = ring[pair % kD][0], vu = ring[pair % kD][1];
      if (pair + kD < kPairs) {
        ring[pair % kD][0] = __ldcs(reinterpret_cast<const uint4*>(gp + (pair + kD) * 32));
        ring[pair % kD][1] = __ldcs(reinterpret_cast<const uint4*>(up + (pair + kD) * 32));
      }
      if (pair % 4 == 0 && pair > 0) {
        g0 = ng0; g1 = ng1; u0 = nu0; u1 = nu1;
        const int next = pair / 4 + 1;
        if (next < kScaleBytes / 8) { ng0 = __ldg(sg0 + next); ng1 = __ldg(sg1 + next); nu0 = __ldg(su0 + next); nu1 = __ldg(su1 + next); }
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int c = pair * 2 + h, cb = c % 8, cr = c % (kMxfp4WindowCodes / 32);
        uint32_t ga, gb, ua, ub;
        mxfp4_row_fragments(vg, gid, tid, h, ga, gb);
        mxfp4_row_fragments(vu, gid, tid, h, ua, ub);
        const uint32_t eg0 = static_cast<uint32_t>(g0 >> (8 * cb)) & 0xFFu, eg1 = static_cast<uint32_t>(g1 >> (8 * cb)) & 0xFFu;
        const uint32_t eu0 = static_cast<uint32_t>(u0 >> (8 * cb)) & 0xFFu, eu1 = static_cast<uint32_t>(u1 >> (8 * cb)) & 0xFFu;
        uint32_t wg0[4], wg1[4], wu0[4], wu1[4];
        decode8_e2m1(ga, e8m0_bf16x2(eg0), wg0);
        decode8_e2m1(gb, e8m0_bf16x2(eg1), wg1);
        decode8_e2m1(ua, e8m0_bf16x2(eu0), wu0);
        decode8_e2m1(ub, e8m0_bf16x2(eu1), wu1);
        const uint32_t ag0[4] = {wg0[0], wg1[0], wg0[1], wg1[1]}, ag1[4] = {wg0[2], wg1[2], wg0[3], wg1[3]};
        const uint32_t au0[4] = {wu0[0], wu1[0], wu0[1], wu1[1]}, au1[4] = {wu0[2], wu1[2], wu0[3], wu1[3]};
#pragma unroll
        for (int t = 0; t < kTG; ++t) {
          const uint4 v = *reinterpret_cast<const uint4*>(xb + (t * 8 + gid) * kMxfp4WindowStride * 2 + (cr * 32 + 8 * tid) * 2);
          mma_bf16_16x8x16(acc_g[t], ag0, v.x, v.y);
          mma_bf16_16x8x16(acc_u[t], au0, v.x, v.y);
          mma_bf16_16x8x16(acc_g[t], ag1, v.z, v.w);
          mma_bf16_16x8x16(acc_u[t], au1, v.z, v.w);
        }
      }
    }
  }
  cp_async_wait<0>();
  if (!live) return;
#pragma unroll
  for (int t = 0; t < kTG; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = t * 8 + 2 * tid + (i & 1), row = (i < 2) ? r0 : r1;
      if (r >= g.count) continue;
      act[static_cast<size_t>(g.slots[r]) * kI + row] = swiglu_bf16(acc_g[t][i], acc_u[t][i]);
    }
}

template <int kH, int kI, int kTG, class OutIndex>
__device__ void mxfp4_row_down_unit(const Mxfp4Expert& e, const Mxfp4Group& g, int rb, const uint8_t* smem, float* __restrict__ out,
                                    OutIndex out_index) {
  static_assert(kI % 64 == 0 && kH % kMxfp4UnitRows == 0, "the down shape");
  constexpr int kPairs = kI / 64;
  constexpr int kD = kPairs < 4 ? kPairs : 4;
  constexpr int kRowBytes = kI / 2;
  constexpr int kScaleBytes = kI / kChunkCodes;
  constexpr int stride = kI + 8;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tid = lane & 3;
  const int n0 = rb * kMxfp4UnitRows + warp * kTileRows, r0 = n0 + gid, r1 = r0 + 8;
  const int row_l = n0 + (lane >> 1), half = lane & 1;
  const uint8_t* dp = e.down + static_cast<size_t>(row_l) * kRowBytes + half * 16;
  const uint16_t* s0 = reinterpret_cast<const uint16_t*>(e.down_scales + static_cast<size_t>(r0) * kScaleBytes);
  const uint16_t* s1 = reinterpret_cast<const uint16_t*>(e.down_scales + static_cast<size_t>(r1) * kScaleBytes);
  if (lane < 16) prefetch_l2_bulk(e.down + static_cast<size_t>(n0 + lane) * kRowBytes, kRowBytes);
  uint4 ring[kD];
#pragma unroll
  for (int i = 0; i < kD; ++i) ring[i] = __ldcs(reinterpret_cast<const uint4*>(dp + i * 32));
  const uint16_t* sx = reinterpret_cast<const uint16_t*>(smem);
  float acc[kTG][4];
#pragma unroll
  for (int t = 0; t < kTG; ++t)
    for (int i = 0; i < 4; ++i) acc[t][i] = 0.f;
#pragma unroll
  for (int pair = 0; pair < kPairs; ++pair) {
    const uint4 v = ring[pair % kD];
    if (pair + kD < kPairs) ring[pair % kD] = __ldcs(reinterpret_cast<const uint4*>(dp + (pair + kD) * 32));
    const uint32_t sc0 = __ldg(s0 + pair), sc1 = __ldg(s1 + pair);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int c = 2 * pair + h;
      uint32_t a, b;
      mxfp4_row_fragments(v, gid, tid, h, a, b);
      const uint32_t e0 = (sc0 >> (8 * h)) & 0xFFu, e1 = (sc1 >> (8 * h)) & 0xFFu;
      uint32_t w0[4], w1[4];
      decode8_e2m1(a, e8m0_bf16x2(e0), w0);
      decode8_e2m1(b, e8m0_bf16x2(e1), w1);
      const uint32_t a0[4] = {w0[0], w1[0], w0[1], w1[1]}, a1[4] = {w0[2], w1[2], w0[3], w1[3]};
#pragma unroll
      for (int t = 0; t < kTG; ++t) {
        const uint4 bv = *reinterpret_cast<const uint4*>(sx + (t * 8 + gid) * stride + c * 32 + 8 * tid);
        mma_bf16_16x8x16(acc[t], a0, bv.x, bv.y);
        mma_bf16_16x8x16(acc[t], a1, bv.z, bv.w);
      }
    }
  }
#pragma unroll
  for (int t = 0; t < kTG; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = t * 8 + 2 * tid + (i & 1);
      if (r >= g.count) continue;
      out[out_index(g.slots[r], (i < 2) ? r0 : r1)] = acc[t][i];
    }
}

template <int kH, int kI>
__device__ __forceinline__ void mxfp4_row_gate_up(const Mxfp4Expert& e, const Mxfp4Group& g, int rb, const uint16_t* x, uint16_t* act,
                                                  uint8_t* smem) {
  switch ((g.count + 7) / 8) {
    case 1: mxfp4_row_gate_up_unit<kH, kI, 1>(e, g, rb, x, act, smem); break;
    case 2: mxfp4_row_gate_up_unit<kH, kI, 2>(e, g, rb, x, act, smem); break;
    case 3: mxfp4_row_gate_up_unit<kH, kI, 3>(e, g, rb, x, act, smem); break;
    default: mxfp4_row_gate_up_unit<kH, kI, 4>(e, g, rb, x, act, smem); break;
  }
}
template <int kH, int kI, class OutIndex>
__device__ __forceinline__ void mxfp4_row_down(const Mxfp4Expert& e, const Mxfp4Group& g, int rb, const uint8_t* smem, float* out,
                                               OutIndex out_index) {
  switch ((g.count + 7) / 8) {
    case 1: mxfp4_row_down_unit<kH, kI, 1>(e, g, rb, smem, out, out_index); break;
    case 2: mxfp4_row_down_unit<kH, kI, 2>(e, g, rb, smem, out, out_index); break;
    case 3: mxfp4_row_down_unit<kH, kI, 3>(e, g, rb, smem, out, out_index); break;
    default: mxfp4_row_down_unit<kH, kI, 4>(e, g, rb, smem, out, out_index); break;
  }
}

#endif  // __CUDACC__

}  // namespace dgpp::step
