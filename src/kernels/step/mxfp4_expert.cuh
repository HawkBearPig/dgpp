#pragma once
// The MXFP4 expert consumer (MOONSHOT-1.md sections 1.3 and 2.7, the form
// measured 2026-10-30 at 94 to 96 percent of the ceiling): a block of eight
// warps takes one unit, a group of up to 32 token slots of one expert and a
// 128-row block of the expert's gate and up pair, or of its down matrix.
// Each warp owns a 16-row tile of the tile-major layout and keeps four
// lane-contiguous 512-byte loads a matrix in flight in registers; a lane's
// 16 bytes are two chunks' codes of one row, decoded once by byte permute
// into bf16 A fragments and multiplied against the group's activations, the
// B operand, which the block stages in shared memory with cp.async windows.
// The gate and up pair shares one B fetch. Accumulation is fp32 on the
// tensor core per 16-row tile, the epilogue is the slot kernels' SwiGLU
// op for op, so the result is within two bf16 ulps of the slot form and
// bitwise its own across launches and row counts.
#include <cstdint>

#include "kernels/step/primitives.cuh"

namespace dgpp::step {

// A routed expert's six ranges in the arena, as the plan places them.
struct Mxfp4Expert {
  const uint8_t* gate = nullptr;      // tile-major [I, H]
  const uint8_t* gate_scales = nullptr;  // e8m0 [I, H / 32]
  const uint8_t* up = nullptr;
  const uint8_t* up_scales = nullptr;
  const uint8_t* down = nullptr;      // tile-major [H, I]
  const uint8_t* down_scales = nullptr;  // e8m0 [H, I / 32]
};

// A group: the slots of one expert in the step, `count` of them (at most
// kMxfp4GroupRows), slot i's token row in `token`.
struct Mxfp4Group {
  const int32_t* slots = nullptr;  // the slot ids, for the act and out rows
  const int32_t* tokens = nullptr; // slot i's token row of x
  int count = 0;
};

constexpr int kMxfp4GroupRows = 32;   // slots a group computes at once (four m16n8 tiles of eight)
constexpr int kMxfp4UnitRows = 128;   // output rows a unit covers: eight warps, one 16-row tile each
constexpr int kMxfp4Threads = 256;
constexpr int kMxfp4WindowCodes = 256;  // the B window: this many codes of every group row
constexpr int kMxfp4WindowStride = kMxfp4WindowCodes + 8;  // bf16 elements a row, padded past the bank period
constexpr int kMxfp4WindowBytes = kMxfp4GroupRows * kMxfp4WindowStride * 2;
constexpr int kMxfp4Windows = 3;
constexpr float kMxfp4SwigluLimit = 10.0f;  // GlmMoeConfig::swiglu_limit of the DeepSeek families

// Shared memory a unit needs: the three B windows of the pair, or the group's
// staged down activations [32][I + 8].
template <int kI>
__host__ __device__ constexpr int mxfp4_unit_smem_bytes() {
  constexpr int windows = kMxfp4Windows * kMxfp4WindowBytes;
  constexpr int down = kMxfp4GroupRows * (kI + 8) * 2;
  return windows > down ? windows : down;
}

#if defined(__CUDACC__)

__device__ __forceinline__ float sigmoid_acc(float x) { return 1.0f / (1.0f + expf(-x)); }

// The slot kernels' SwiGLU epilogue: bf16 roundings in their order.
__device__ __forceinline__ uint16_t swiglu_bf16(float gate, float up) {
  float g = bf16_to_f32(f32_to_bf16(gate)), u = bf16_to_f32(f32_to_bf16(up));
  if (g > kMxfp4SwigluLimit) g = kMxfp4SwigluLimit;
  u = fminf(fmaxf(u, -kMxfp4SwigluLimit), kMxfp4SwigluLimit);
  const uint16_t t = f32_to_bf16(g * sigmoid_acc(g));
  return f32_to_bf16(bf16_to_f32(t) * u);
}

// Stages window `w` of the group's token rows: codes [w * 256, +256) of x
// (bf16 [tokens, kH]) into buffer `buf`, one cp.async group.
template <int kH>
__device__ __forceinline__ void mxfp4_fill_window(const uint16_t* __restrict__ x, const Mxfp4Group& g, int w, int buf, uint8_t* smem) {
  uint8_t* dst = smem + buf * kMxfp4WindowBytes;
  constexpr int chunks16 = kMxfp4GroupRows * (kMxfp4WindowCodes / 8);
  for (int q = threadIdx.x; q < chunks16; q += kMxfp4Threads) {
    const int t = q / (kMxfp4WindowCodes / 8), c16 = q % (kMxfp4WindowCodes / 8);
    const int tok = t < g.count ? g.tokens[t] : g.tokens[0];
    cp_async_16(dst + t * kMxfp4WindowStride * 2 + c16 * 16, x + static_cast<size_t>(tok) * kH + static_cast<size_t>(w) * kMxfp4WindowCodes + c16 * 8);
  }
  cp_async_commit();
}

// The gate and up pair over one 128-row block `rb` of the expert for group
// `g`: writes act[slot][row] (bf16 [slots, kI]) for the block's rows.
// kTG: the group's m16n8 tiles of eight slots (1 to 4).
template <int kH, int kI, int kTG>
__device__ void mxfp4_gate_up_unit(const Mxfp4Expert& e, const Mxfp4Group& g, int rb, const uint16_t* __restrict__ x,
                                   uint16_t* __restrict__ act, uint8_t* smem) {
  static_assert(kH % kMxfp4WindowCodes == 0 && kI % kTileRows == 0, "the pair's shape");
  constexpr int kD = 4;                        // 512-byte loads in flight a matrix
  constexpr int kRuns = kH / 64;               // 512-byte runs a tile: two chunks each
  constexpr int kWindows = kH / kMxfp4WindowCodes;
  constexpr int kScaleBytes = kH / kChunkCodes;  // a row's e8m0 scales
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tid = lane & 3;
  const int n0 = rb * kMxfp4UnitRows + warp * kTileRows, r0 = n0 + gid, r1 = r0 + 8;
  const bool live = n0 < kI;
  const int tile = live ? n0 / kTileRows : 0;
  const uint4* tg = reinterpret_cast<const uint4*>(e.gate + static_cast<size_t>(tile) * tile_major_tile_bytes(kH)) + lane;
  const uint4* tu = reinterpret_cast<const uint4*>(e.up + static_cast<size_t>(tile) * tile_major_tile_bytes(kH)) + lane;
  const int rr0 = live ? r0 : 0, rr1 = live ? r1 : 0;
  const unsigned long long* sg0 = reinterpret_cast<const unsigned long long*>(e.gate_scales + static_cast<size_t>(rr0) * kScaleBytes);
  const unsigned long long* sg1 = reinterpret_cast<const unsigned long long*>(e.gate_scales + static_cast<size_t>(rr1) * kScaleBytes);
  const unsigned long long* su0 = reinterpret_cast<const unsigned long long*>(e.up_scales + static_cast<size_t>(rr0) * kScaleBytes);
  const unsigned long long* su1 = reinterpret_cast<const unsigned long long*>(e.up_scales + static_cast<size_t>(rr1) * kScaleBytes);
  const unsigned full = 0xffffffffu;
  uint4 ring[kD][2];
#pragma unroll
  for (int i = 0; i < kD; ++i) { ring[i][0] = __ldcs(tg + i * 32); ring[i][1] = __ldcs(tu + i * 32); }
  // Eight scale bytes a load: the scales of eight chunks, four runs.
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
      const int run = w * (kMxfp4WindowCodes / 64) + pq;
      const uint4 vg = ring[run % kD][0], vu = ring[run % kD][1];
      if (run + kD < kRuns) { ring[run % kD][0] = __ldcs(tg + (run + kD) * 32); ring[run % kD][1] = __ldcs(tu + (run + kD) * 32); }
      if (run % 4 == 0 && run > 0) {
        g0 = ng0; g1 = ng1; u0 = nu0; u1 = nu1;
        const int next = run / 4 + 1;
        if (next < kScaleBytes / 8) { ng0 = __ldg(sg0 + next); ng1 = __ldg(sg1 + next); nu0 = __ldg(su0 + next); nu1 = __ldg(su1 + next); }
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int c = run * 2 + h, cb = c % 8, cr = c % (kMxfp4WindowCodes / 32);
        uint32_t ga = 0, gb = 0, ua = 0, ub = 0;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint32_t vgq = q == 0 ? vg.x : q == 1 ? vg.y : q == 2 ? vg.z : vg.w;
          const uint32_t vuq = q == 0 ? vu.x : q == 1 ? vu.y : q == 2 ? vu.z : vu.w;
          const uint32_t a = __shfl_sync(full, vgq, gid + 16 * h), b = __shfl_sync(full, vgq, gid + 8 + 16 * h);
          const uint32_t c2 = __shfl_sync(full, vuq, gid + 16 * h), d = __shfl_sync(full, vuq, gid + 8 + 16 * h);
          if (q == tid) { ga = a; gb = b; ua = c2; ub = d; }
        }
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

// Stages the group's act rows [count][kI] into shared memory at a padded stride.
template <int kI>
__device__ __forceinline__ void mxfp4_stage_down_act(const uint16_t* __restrict__ act, const Mxfp4Group& g, uint8_t* smem) {
  uint16_t* sx = reinterpret_cast<uint16_t*>(smem);
  constexpr int stride = kI + 8;
  for (int q = threadIdx.x; q < kMxfp4GroupRows * (kI / 8); q += kMxfp4Threads) {
    const int t = q / (kI / 8), c = q % (kI / 8);
    const int slot = t < g.count ? g.slots[t] : g.slots[0];
    *reinterpret_cast<uint4*>(sx + t * stride + c * 8) = *reinterpret_cast<const uint4*>(act + static_cast<size_t>(slot) * kI + c * 8);
  }
}

// The down matrix over one 128-row block `rb` of the expert for group `g`,
// the group's act rows staged in `smem` (mxfp4_stage_down_act): writes the
// fp32 partial out[out_index(slot, row)] for the block's rows.
template <int kH, int kI, int kTG, class OutIndex>
__device__ void mxfp4_down_unit(const Mxfp4Expert& e, const Mxfp4Group& g, int rb, const uint8_t* smem, float* __restrict__ out,
                                OutIndex out_index) {
  static_assert(kI % 64 == 0 && kH % kMxfp4UnitRows == 0, "the down shape");
  constexpr int kRuns = kI / 64;
  constexpr int kD = kRuns < 4 ? kRuns : 4;
  constexpr int kScaleBytes = kI / kChunkCodes;
  constexpr int stride = kI + 8;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tid = lane & 3;
  const int n0 = rb * kMxfp4UnitRows + warp * kTileRows, r0 = n0 + gid, r1 = r0 + 8;
  const uint4* td = reinterpret_cast<const uint4*>(e.down + static_cast<size_t>(n0 / kTileRows) * tile_major_tile_bytes(kI)) + lane;
  const uint16_t* s0 = reinterpret_cast<const uint16_t*>(e.down_scales + static_cast<size_t>(r0) * kScaleBytes);
  const uint16_t* s1 = reinterpret_cast<const uint16_t*>(e.down_scales + static_cast<size_t>(r1) * kScaleBytes);
  const unsigned full = 0xffffffffu;
  uint4 ring[kD];
#pragma unroll
  for (int i = 0; i < kD; ++i) ring[i] = __ldcs(td + i * 32);
  const uint16_t* sx = reinterpret_cast<const uint16_t*>(smem);
  float acc[kTG][4];
#pragma unroll
  for (int t = 0; t < kTG; ++t)
    for (int i = 0; i < 4; ++i) acc[t][i] = 0.f;
#pragma unroll
  for (int run = 0; run < kRuns; ++run) {
    const uint4 v = ring[run % kD];
    if (run + kD < kRuns) ring[run % kD] = __ldcs(td + (run + kD) * 32);
    const uint32_t sc0 = __ldg(s0 + run), sc1 = __ldg(s1 + run);  // two chunks' scales a run
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int c = 2 * run + h;
      uint32_t a = 0, b = 0;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint32_t vq = q == 0 ? v.x : q == 1 ? v.y : q == 2 ? v.z : v.w;
        const uint32_t aa = __shfl_sync(full, vq, gid + 16 * h), bb = __shfl_sync(full, vq, gid + 8 + 16 * h);
        if (q == tid) { a = aa; b = bb; }
      }
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

// The dispatch on the group's size: tiles of eight slots.
template <int kH, int kI>
__device__ __forceinline__ void mxfp4_gate_up(const Mxfp4Expert& e, const Mxfp4Group& g, int rb, const uint16_t* x, uint16_t* act,
                                              uint8_t* smem) {
  switch ((g.count + 7) / 8) {
    case 1: mxfp4_gate_up_unit<kH, kI, 1>(e, g, rb, x, act, smem); break;
    case 2: mxfp4_gate_up_unit<kH, kI, 2>(e, g, rb, x, act, smem); break;
    case 3: mxfp4_gate_up_unit<kH, kI, 3>(e, g, rb, x, act, smem); break;
    default: mxfp4_gate_up_unit<kH, kI, 4>(e, g, rb, x, act, smem); break;
  }
}
template <int kH, int kI, class OutIndex>
__device__ __forceinline__ void mxfp4_down(const Mxfp4Expert& e, const Mxfp4Group& g, int rb, const uint8_t* smem, float* out,
                                           OutIndex out_index) {
  switch ((g.count + 7) / 8) {
    case 1: mxfp4_down_unit<kH, kI, 1>(e, g, rb, smem, out, out_index); break;
    case 2: mxfp4_down_unit<kH, kI, 2>(e, g, rb, smem, out, out_index); break;
    case 3: mxfp4_down_unit<kH, kI, 3>(e, g, rb, smem, out, out_index); break;
    default: mxfp4_down_unit<kH, kI, 4>(e, g, rb, smem, out, out_index); break;
  }
}

#endif  // __CUDACC__

}  // namespace dgpp::step
