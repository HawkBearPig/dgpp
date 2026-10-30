#pragma once
// The fp8 block-scaled consumer over the checkpoint's row-major layout
// (MOONSHOT-1.md sections 1.3 and 2.7): the shared expert's form, and the
// candidate for the fp8 projections. Each matrix is e4m3 [rows, K] with an
// F32 scale grid on kBlock x kBlock blocks (32 on DeepSeek-V4.1, 128 on
// GLM), kept in the layout the loader produces, so the prefill path reads
// the same bytes untouched. A unit is a group of up to 32 token slots and a
// 128-row block, eight warps on 16-row tiles. The rows stay row-major, so
// a lane's 16 bytes are half a chunk of one row and the DRAM would see
// 32-byte runs; the bridge the probes measured fixes that: before a warp
// starts a 512-byte slab of its 16 rows it prefetches the next slab of
// every row into L2 with one bulk prefetch a row, and the fragment loads
// then hit L2. Per chunk the e4m3 codes become bf16 exactly, the chunk's
// tensor-core partial is scaled by its F32 block scale and accumulated, the
// slot kernels' structure; the epilogue is their SwiGLU op for op.
#include <cstdint>

#include "kernels/step/mxfp4_expert.cuh"
#include "kernels/step/primitives.cuh"

namespace dgpp::step {

struct Fp8Expert {
  const uint8_t* gate = nullptr;    // e4m3 [I, H] row-major
  const float* gate_scales = nullptr;  // F32 [I / kBlock, H / kBlock]
  const uint8_t* up = nullptr;
  const float* up_scales = nullptr;
  const uint8_t* down = nullptr;    // e4m3 [H, I] row-major
  const float* down_scales = nullptr;  // F32 [H / kBlock, I / kBlock]
};

constexpr int kFp8SlabBytes = 512;  // the bridge's prefetch granule a row
// The slab a row of K bytes takes: 512 where K divides into it, else 256.
__host__ __device__ constexpr int fp8_slab_bytes(int k_bytes) { return k_bytes % kFp8SlabBytes == 0 ? kFp8SlabBytes : kFp8SlabBytes / 2; }

#if defined(__CUDACC__)

__device__ __forceinline__ void prefetch_l2_bulk(const void* gmem, unsigned bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(gmem), "r"(bytes) : "memory");
}

// Four e4m3 bytes (codes k..k+3 of one row) as two bf16x2 words {(k, k+1),
// (k+2, k+3)}: exponent rebias 7 to 127, the mantissa into bf16's top three
// bits, zero kept zero; e4m3 weights have no NaN and the scale grid holds
// their subnormals out.
__device__ __forceinline__ void e4m3x4_bf16x2(uint32_t w, uint32_t& lo, uint32_t& hi) {
  const uint32_t b0 = prmt(w, 0u, 0x4140u), b1 = prmt(w, 0u, 0x4342u);
  const uint32_t m0 = (b0 & 0x007F007Fu) << 4, m1 = (b1 & 0x007F007Fu) << 4;
  const uint32_t s0 = (b0 & 0x00800080u) << 8, s1 = (b1 & 0x00800080u) << 8;
  const uint32_t nz0 = ((b0 & 0x007F007Fu) + 0x007F007Fu) & 0x00800080u, nz1 = ((b1 & 0x007F007Fu) + 0x007F007Fu) & 0x00800080u;
  lo = (m0 + ((nz0 >> 7) * 0x3C00u)) | s0;
  hi = (m1 + ((nz1 >> 7) * 0x3C00u)) | s1;
}

// Lane l of a warp holds 16 bytes of row l / 2, half l & 1 of a 32-code
// chunk; the fragment owner (gid, tid) takes its two words from lanes
// 2 gid + (tid >> 1) and that plus 16, word (tid & 1) * 2 and the next.
__device__ __forceinline__ void fp8_fragments(const uint4& v, int gid, int tid, uint32_t (&t0)[4], uint32_t (&t1)[4]) {
  const unsigned full = 0xffffffffu;
  const int src0 = 2 * gid + (tid >> 1), src1 = src0 + 16, wsel = (tid & 1) * 2;
  uint32_t a0 = 0, a1 = 0, b0 = 0, b1 = 0;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const uint32_t vw = w == 0 ? v.x : w == 1 ? v.y : w == 2 ? v.z : v.w;
    const uint32_t a = __shfl_sync(full, vw, src0), b = __shfl_sync(full, vw, src1);
    if (w == wsel) { a0 = a; b0 = b; }
    if (w == wsel + 1) { a1 = a; b1 = b; }
  }
  e4m3x4_bf16x2(a0, t0[0], t0[2]);
  e4m3x4_bf16x2(b0, t0[1], t0[3]);
  e4m3x4_bf16x2(a1, t1[0], t1[2]);
  e4m3x4_bf16x2(b1, t1[1], t1[3]);
}

// The gate and up pair over 128-row block `rb` of a row-major fp8 expert for
// group `g` (the activations x, bf16 [tokens, kH]), into act[slot][row].
template <int kH, int kI, int kBlock, int kTG>
__device__ void fp8_gate_up_unit(const Fp8Expert& e, const Mxfp4Group& g, int rb, const uint16_t* __restrict__ x, uint16_t* __restrict__ act,
                                 uint8_t* smem) {
  static_assert(kH % kFp8SlabBytes == 0 && kBlock % 32 == 0 && kI % kTileRows == 0, "the pair's shape");
  constexpr int kD = 4;                     // chunks in flight a matrix
  constexpr int kChunks = kH / 32;
  constexpr int kChunksPerBlock = kBlock / 32;
  constexpr int kSC = kH / kBlock;          // scale columns
  constexpr int kWindows = kH / kMxfp4WindowCodes;
  constexpr int kChunksPerSlab = kFp8SlabBytes / 32;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tid = lane & 3;
  const int n0 = rb * kMxfp4UnitRows + warp * kTileRows, r0 = n0 + gid, r1 = r0 + 8;
  const bool live = n0 < kI;
  const int row_l = (live ? n0 : 0) + (lane >> 1), half = lane & 1;
  const uint8_t* gp = e.gate + static_cast<size_t>(row_l) * kH + half * 16;
  const uint8_t* up = e.up + static_cast<size_t>(row_l) * kH + half * 16;
  const float* sg = e.gate_scales + static_cast<size_t>((live ? n0 : 0) / kBlock) * kSC;
  const float* su = e.up_scales + static_cast<size_t>((live ? n0 : 0) / kBlock) * kSC;
  // The bridge: slab s + 1 of every row of the tile into L2, one row a lane, both matrices.
  auto bridge = [&](int slab) {
    if (slab >= kH / kFp8SlabBytes) return;
    const int r = (live ? n0 : 0) + (lane & 15);
    const uint8_t* m = lane < 16 ? e.gate : e.up;
    prefetch_l2_bulk(m + static_cast<size_t>(r) * kH + static_cast<size_t>(slab) * kFp8SlabBytes, kFp8SlabBytes);
  };
  bridge(0);
  bridge(1);
  uint4 ring[kD][2];
#pragma unroll
  for (int i = 0; i < kD; ++i) {
    ring[i][0] = __ldcs(reinterpret_cast<const uint4*>(gp + i * 32));
    ring[i][1] = __ldcs(reinterpret_cast<const uint4*>(up + i * 32));
  }
  mxfp4_fill_window<kH>(x, g, 0, 0, smem);
  mxfp4_fill_window<kH>(x, g, 1, 1, smem);
  float acc_g[kTG][4], acc_u[kTG][4], blk_g[kTG][4], blk_u[kTG][4];
#pragma unroll
  for (int t = 0; t < kTG; ++t)
    for (int i = 0; i < 4; ++i) { acc_g[t][i] = acc_u[t][i] = blk_g[t][i] = blk_u[t][i] = 0.f; }
  float scale_g = __ldg(sg), scale_u = __ldg(su);
#pragma unroll 1
  for (int w = 0; w < kWindows; ++w) {
    cp_async_wait<1>();
    __syncthreads();
    if (w + 2 < kWindows) mxfp4_fill_window<kH>(x, g, w + 2, (w + 2) % kMxfp4Windows, smem); else cp_async_commit();
    const uint8_t* xb = smem + (w % kMxfp4Windows) * kMxfp4WindowBytes;
#pragma unroll
    for (int cc = 0; cc < kMxfp4WindowCodes / 32; ++cc) {
      const int c = w * (kMxfp4WindowCodes / 32) + cc;
      if (c % kChunksPerSlab == 0 && c > 0) bridge(c / kChunksPerSlab + 1);
      const uint4 vg = ring[c % kD][0], vu = ring[c % kD][1];
      if (c + kD < kChunks) {
        ring[c % kD][0] = __ldcs(reinterpret_cast<const uint4*>(gp + (c + kD) * 32));
        ring[c % kD][1] = __ldcs(reinterpret_cast<const uint4*>(up + (c + kD) * 32));
      }
      uint32_t ag0[4], ag1[4], au0[4], au1[4];
      fp8_fragments(vg, gid, tid, ag0, ag1);
      fp8_fragments(vu, gid, tid, au0, au1);
#pragma unroll
      for (int t = 0; t < kTG; ++t) {
        const uint4 v = *reinterpret_cast<const uint4*>(xb + (t * 8 + gid) * kMxfp4WindowStride * 2 + (cc * 32 + 8 * tid) * 2);
        mma_bf16_16x8x16(blk_g[t], ag0, v.x, v.y);
        mma_bf16_16x8x16(blk_u[t], au0, v.x, v.y);
        mma_bf16_16x8x16(blk_g[t], ag1, v.z, v.w);
        mma_bf16_16x8x16(blk_u[t], au1, v.z, v.w);
      }
      if ((c + 1) % kChunksPerBlock == 0) {
#pragma unroll
        for (int t = 0; t < kTG; ++t)
          for (int i = 0; i < 4; ++i) {
            acc_g[t][i] = fmaf(scale_g, blk_g[t][i], acc_g[t][i]);
            acc_u[t][i] = fmaf(scale_u, blk_u[t][i], acc_u[t][i]);
            blk_g[t][i] = 0.f;
            blk_u[t][i] = 0.f;
          }
        const int nb = (c + 1) / kChunksPerBlock;
        if (nb < kSC) { scale_g = __ldg(sg + nb); scale_u = __ldg(su + nb); }
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

// The down matrix over 128-row block `rb` for group `g`, the group's act rows
// staged in smem (mxfp4_stage_down_act<kI>): fp32 partials out[out_index(slot, row)].
template <int kH, int kI, int kBlock, int kTG, class OutIndex>
__device__ void fp8_down_unit(const Fp8Expert& e, const Mxfp4Group& g, int rb, const uint8_t* smem, float* __restrict__ out,
                              OutIndex out_index) {
  static_assert(kI % 32 == 0 && kH % kMxfp4UnitRows == 0 && kBlock % 32 == 0, "the down shape");
  constexpr int kChunks = kI / 32;
  constexpr int kD = kChunks < 4 ? kChunks : 4;
  constexpr int kChunksPerBlock = kBlock / 32;
  constexpr int kSC = (kI + kBlock - 1) / kBlock;
  constexpr int stride = kI + 8;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tid = lane & 3;
  const int n0 = rb * kMxfp4UnitRows + warp * kTileRows, r0 = n0 + gid, r1 = r0 + 8;
  const int row_l = n0 + (lane >> 1), half = lane & 1;
  const uint8_t* dp = e.down + static_cast<size_t>(row_l) * kI + half * 16;
  const float* sd = e.down_scales + static_cast<size_t>(n0 / kBlock) * kSC;
  // The bridge: the tile's whole rows into L2, one a lane (K is a slab or so).
  if (lane < 16) prefetch_l2_bulk(e.down + static_cast<size_t>(n0 + lane) * kI, kI);
  uint4 ring[kD];
#pragma unroll
  for (int i = 0; i < kD; ++i) ring[i] = __ldcs(reinterpret_cast<const uint4*>(dp + i * 32));
  const uint16_t* sx = reinterpret_cast<const uint16_t*>(smem);
  float acc[kTG][4], blk[kTG][4];
#pragma unroll
  for (int t = 0; t < kTG; ++t)
    for (int i = 0; i < 4; ++i) { acc[t][i] = 0.f; blk[t][i] = 0.f; }
  float scale = __ldg(sd);
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    const uint4 v = ring[c % kD];
    if (c + kD < kChunks) ring[c % kD] = __ldcs(reinterpret_cast<const uint4*>(dp + (c + kD) * 32));
    uint32_t a0[4], a1[4];
    fp8_fragments(v, gid, tid, a0, a1);
#pragma unroll
    for (int t = 0; t < kTG; ++t) {
      const uint4 bv = *reinterpret_cast<const uint4*>(sx + (t * 8 + gid) * stride + c * 32 + 8 * tid);
      mma_bf16_16x8x16(blk[t], a0, bv.x, bv.y);
      mma_bf16_16x8x16(blk[t], a1, bv.z, bv.w);
    }
    if ((c + 1) % kChunksPerBlock == 0 || c == kChunks - 1) {
#pragma unroll
      for (int t = 0; t < kTG; ++t)
        for (int i = 0; i < 4; ++i) { acc[t][i] = fmaf(scale, blk[t][i], acc[t][i]); blk[t][i] = 0.f; }
      const int nb = (c + 1) / kChunksPerBlock;
      if (nb < kSC) scale = __ldg(sd + nb);
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

// One row-major fp8 matrix W [kN, kK] with F32 kBlock scales over 128-row
// block `rb` for group `g` (its activations x, bf16 [tokens, kK]), over the
// windows [w0, w1) of 256 codes: writes y[slot * kN + row] as bf16 when
// `partial` is null, else the fp32 slab partial partial[slot * kN + row]. A
// projection's form: the pair's with one matrix.
template <int kK, int kN, int kBlock, int kTG>
__device__ void fp8_matrix_unit(const uint8_t* __restrict__ w, const float* __restrict__ scales, const Mxfp4Group& g, int rb,
                                const uint16_t* __restrict__ x, uint16_t* __restrict__ y, float* __restrict__ partial, uint8_t* smem,
                                int w0, int w1, int unit_rows) {
  static_assert(kK % kMxfp4WindowCodes == 0 && kBlock % 32 == 0 && kN % kTileRows == 0, "the matrix's shape");
  constexpr int kSlab = fp8_slab_bytes(kK);
  constexpr int kD = kMxfp4WindowCodes / 32;  // chunks in flight a lane: a window's worth, so the ring index is the unrolled chunk index
  constexpr int kAhead = 4;    // slabs the bridge runs ahead of the loads: 32 KB a warp in L2 prefetch against the loaded DRAM's latency
  constexpr int kChunksPerBlock = kBlock / 32;
  constexpr int kSC = kK / kBlock;
  constexpr int kChunksPerSlab = kSlab / 32;
  const int c_begin = w0 * (kMxfp4WindowCodes / 32), c_end = w1 * (kMxfp4WindowCodes / 32);
  const int slab_end = (c_end * 32 + kSlab - 1) / kSlab;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tid = lane & 3;
  const int n0 = rb * unit_rows + warp * kTileRows, r0 = n0 + gid, r1 = r0 + 8;
  const bool live = n0 < kN && warp * kTileRows < unit_rows;  // warps past the unit's rows only help stage the windows
  const int row_l = (live ? n0 : 0) + (lane >> 1), half = lane & 1;
  const uint8_t* wp = w + static_cast<size_t>(row_l) * kK + half * 16;
  const float* sw = scales + static_cast<size_t>((live ? n0 : 0) / kBlock) * kSC;
  auto bridge = [&](int slab) {
    if (!live || slab >= slab_end || lane >= 16) return;
    const int r = (live ? n0 : 0) + lane;
    prefetch_l2_bulk(w + static_cast<size_t>(r) * kK + static_cast<size_t>(slab) * kSlab, kSlab);
  };
  const int slab0 = c_begin * 32 / kSlab;
#pragma unroll
  for (int a = 0; a <= kAhead; ++a) bridge(slab0 + a);
  uint4 ring[kD];
#pragma unroll
  for (int i = 0; i < kD; ++i) ring[i] = __ldcs(reinterpret_cast<const uint4*>(wp + (c_begin + i) * 32));
  mxfp4_fill_window<kK>(x, g, w0, 0, smem);
  mxfp4_fill_window<kK>(x, g, w0 + 1, 1, smem);
  float acc[kTG][4], blk[kTG][4];
#pragma unroll
  for (int t = 0; t < kTG; ++t)
    for (int i = 0; i < 4; ++i) { acc[t][i] = 0.f; blk[t][i] = 0.f; }
  float scale = __ldg(sw + c_begin / kChunksPerBlock);
#pragma unroll 1
  for (int wdw = w0; wdw < w1; ++wdw) {
    cp_async_wait<1>();
    __syncthreads();
    if (wdw + 2 < w1) mxfp4_fill_window<kK>(x, g, wdw + 2, (wdw - w0 + 2) % kMxfp4Windows, smem); else cp_async_commit();
    const uint8_t* xb = smem + ((wdw - w0) % kMxfp4Windows) * kMxfp4WindowBytes;
#pragma unroll
    for (int cc = 0; cc < kMxfp4WindowCodes / 32; ++cc) {
      const int c = wdw * (kMxfp4WindowCodes / 32) + cc;
      if (c % kChunksPerSlab == 0 && c > c_begin) bridge(c / kChunksPerSlab + kAhead);
      const uint4 v = ring[cc];  // kD chunks a window: slot cc holds chunk c
      if (c + kD < c_end) ring[cc] = __ldcs(reinterpret_cast<const uint4*>(wp + (c + kD) * 32));
      uint32_t a0[4], a1[4];
      fp8_fragments(v, gid, tid, a0, a1);
#pragma unroll
      for (int t = 0; t < kTG; ++t) {
        const uint4 bv = *reinterpret_cast<const uint4*>(xb + (t * 8 + gid) * kMxfp4WindowStride * 2 + (cc * 32 + 8 * tid) * 2);
        mma_bf16_16x8x16(blk[t], a0, bv.x, bv.y);
        mma_bf16_16x8x16(blk[t], a1, bv.z, bv.w);
      }
      if ((c + 1) % kChunksPerBlock == 0) {
#pragma unroll
        for (int t = 0; t < kTG; ++t)
          for (int i = 0; i < 4; ++i) { acc[t][i] = fmaf(scale, blk[t][i], acc[t][i]); blk[t][i] = 0.f; }
        const int nb = (c + 1) / kChunksPerBlock;
        if (nb < kSC) scale = __ldg(sw + nb);
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
      const size_t at = static_cast<size_t>(g.slots[r]) * kN + row;
      if (partial) partial[at] = acc[t][i]; else y[at] = f32_to_bf16(acc[t][i]);
    }
}
// The small-group form of one fp8 matrix (groups of at most eight slots):
// a unit is 16 kRowTiles rows, and the eight warps split the inner width
// kSplit ways, warp w on row tile w % kRowTiles and part w / kRowTiles.
// The group's activations are staged once for the unit, every part's slice
// in shared memory, and each warp's fp32 partial is summed across the parts
// in part order, so the unit is a quarter the length of the 128-row form
// at the same loads in flight and a phase of a few matrices has units
// enough for every worker. Shared memory: kSplit x 8 x (kK / kSplit + 8)
// bf16 plus the reduce's 8 x 16 x 8 fp32.
constexpr int kFp8SplitRows = 8;  // slots the form takes
template <int kK, int kRowTiles>
__host__ __device__ constexpr int fp8_split_smem_bytes() {
  constexpr int split = 8 / kRowTiles;
  return split * kFp8SplitRows * (kK / split + 8) * 2 + 8 * 16 * 8 * 4;
}
template <int kK, int kN, int kBlock, int kRowTiles>
__device__ void fp8_matrix_split_unit(const uint8_t* __restrict__ w, const float* __restrict__ scales, const Mxfp4Group& g, int rb,
                                      const uint16_t* __restrict__ x, uint16_t* __restrict__ y, uint8_t* smem, bool stage = true) {
  constexpr int kSplit = 8 / kRowTiles;
  constexpr int kPartK = kK / kSplit;  // codes a part
  static_assert(kK % (kSplit * 32) == 0 && kN % kTileRows == 0 && kBlock % 32 == 0 && (kPartK % kBlock) == 0, "the split shape");
  constexpr int kSlab = fp8_slab_bytes(kPartK);
  constexpr int kD = 8;
  constexpr int kChunks = kPartK / 32;
  constexpr int kChunksPerBlock = kBlock / 32;
  constexpr int kSC = kK / kBlock;
  constexpr int kChunksPerSlab = kSlab / 32;
  constexpr int kStride = kPartK + 8;  // bf16 elements a staged row
  constexpr int kUnitRows = kTileRows * kRowTiles;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tid = lane & 3;
  const int tile = warp % kRowTiles, part = warp / kRowTiles;
  const int n0 = rb * kUnitRows + tile * kTileRows, r0 = n0 + gid, r1 = r0 + 8;
  const bool live = n0 < kN;
  const int k0 = part * kPartK;  // this warp's codes
  // Stage the group's rows of every part: part p's slice at smem + p * 8 * kStride * 2.
  uint16_t* act = reinterpret_cast<uint16_t*>(smem);
  float* red = reinterpret_cast<float*>(smem + kSplit * kFp8SplitRows * kStride * 2);
  if (stage) {  // consecutive units of one matrix reuse the group's staged rows
    const size_t xs = g.x_stride ? static_cast<size_t>(g.x_stride) : static_cast<size_t>(kK);
    constexpr int per_row16 = kK / 8;  // 16-byte pieces a row over the whole width
    for (int q = threadIdx.x; q < kFp8SplitRows * per_row16; q += kMxfp4Threads) {
      const int t = q / per_row16, c16 = q % per_row16;
      const int tok = t < g.count ? g.tokens[t] : g.tokens[0];
      const int code = c16 * 8, pp = code / kPartK, kk = code - pp * kPartK;
      cp_async_16(act + (static_cast<size_t>(pp) * kFp8SplitRows + t) * kStride + kk, x + static_cast<size_t>(tok) * xs + code);
    }
    cp_async_commit();
  }
  const int row_l = (live ? n0 : 0) + (lane >> 1), half = lane & 1;
  const uint8_t* wp = w + static_cast<size_t>(row_l) * kK + k0 + half * 16;
  const float* sw = scales + static_cast<size_t>((live ? n0 : 0) / kBlock) * kSC + k0 / kBlock;
  auto bridge = [&](int slab) {
    if (!live || slab >= kPartK / kSlab || lane >= 16) return;
    prefetch_l2_bulk(w + static_cast<size_t>(n0 + lane) * kK + k0 + static_cast<size_t>(slab) * kSlab, kSlab);
  };
  bridge(0);
  bridge(1);
  bridge(2);
  uint4 ring[kD];
#pragma unroll
  for (int i = 0; i < kD; ++i) ring[i] = __ldcs(reinterpret_cast<const uint4*>(wp + i * 32));
  const uint16_t* xb = act + static_cast<size_t>(part) * kFp8SplitRows * kStride;
  float acc[4] = {0.f, 0.f, 0.f, 0.f}, blk[4] = {0.f, 0.f, 0.f, 0.f};
  float scale = __ldg(sw);
  cp_async_wait<0>();
  __syncthreads();
#pragma unroll 1
  for (int c8 = 0; c8 < kChunks; c8 += kD) {
#pragma unroll
    for (int cc = 0; cc < kD; ++cc) {
      const int c = c8 + cc;
      if (c >= kChunks) break;  // a part whose chunks are not a multiple of the ring
      if (c % kChunksPerSlab == 0 && c > 0) bridge(c / kChunksPerSlab + 2);
      const uint4 v = ring[cc];
      if (c + kD < kChunks) ring[cc] = __ldcs(reinterpret_cast<const uint4*>(wp + (c + kD) * 32));
      uint32_t a0[4], a1[4];
      fp8_fragments(v, gid, tid, a0, a1);
      const uint4 bv = *reinterpret_cast<const uint4*>(xb + gid * kStride + c * 32 + 8 * tid);
      mma_bf16_16x8x16(blk, a0, bv.x, bv.y);
      mma_bf16_16x8x16(blk, a1, bv.z, bv.w);
      if ((c + 1) % kChunksPerBlock == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[i] = fmaf(scale, blk[i], acc[i]); blk[i] = 0.f; }
        const int nb = (c + 1) / kChunksPerBlock;
        if (nb < kPartK / kBlock) scale = __ldg(sw + nb);
      }
    }
  }
  // The parts' partials in part order: warp (tile, part) -> red[part][tile][lane][4].
  float* mine = red + ((static_cast<size_t>(part) * kRowTiles + tile) * 32 + lane) * 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) mine[i] = acc[i];
  __syncthreads();
  if (part != 0 || !live) return;
  float sum[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int pp = 0; pp < kSplit; ++pp) {
    const float* other = red + ((static_cast<size_t>(pp) * kRowTiles + tile) * 32 + lane) * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) sum[i] += other[i];
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int r = 2 * tid + (i & 1), row = (i < 2) ? r0 : r1;
    if (r >= g.count) continue;
    y[static_cast<size_t>(g.slots[r]) * kN + row] = f32_to_bf16(sum[i]);
  }
}

template <int kK, int kN, int kBlock>
__device__ __forceinline__ void fp8_matrix(const uint8_t* w, const float* scales, const Mxfp4Group& g, int rb, const uint16_t* x, uint16_t* y,
                                           float* partial, uint8_t* smem, int w0, int w1, int unit_rows = kMxfp4UnitRows) {
  switch ((g.count + 7) / 8) {
    case 1: fp8_matrix_unit<kK, kN, kBlock, 1>(w, scales, g, rb, x, y, partial, smem, w0, w1, unit_rows); break;
    case 2: fp8_matrix_unit<kK, kN, kBlock, 2>(w, scales, g, rb, x, y, partial, smem, w0, w1, unit_rows); break;
    case 3: fp8_matrix_unit<kK, kN, kBlock, 3>(w, scales, g, rb, x, y, partial, smem, w0, w1, unit_rows); break;
    default: fp8_matrix_unit<kK, kN, kBlock, 4>(w, scales, g, rb, x, y, partial, smem, w0, w1, unit_rows); break;
  }
}

template <int kH, int kI, int kBlock>
__device__ __forceinline__ void fp8_gate_up(const Fp8Expert& e, const Mxfp4Group& g, int rb, const uint16_t* x, uint16_t* act, uint8_t* smem) {
  switch ((g.count + 7) / 8) {
    case 1: fp8_gate_up_unit<kH, kI, kBlock, 1>(e, g, rb, x, act, smem); break;
    case 2: fp8_gate_up_unit<kH, kI, kBlock, 2>(e, g, rb, x, act, smem); break;
    case 3: fp8_gate_up_unit<kH, kI, kBlock, 3>(e, g, rb, x, act, smem); break;
    default: fp8_gate_up_unit<kH, kI, kBlock, 4>(e, g, rb, x, act, smem); break;
  }
}
template <int kH, int kI, int kBlock, class OutIndex>
__device__ __forceinline__ void fp8_down(const Fp8Expert& e, const Mxfp4Group& g, int rb, const uint8_t* smem, float* out, OutIndex out_index) {
  switch ((g.count + 7) / 8) {
    case 1: fp8_down_unit<kH, kI, kBlock, 1>(e, g, rb, smem, out, out_index); break;
    case 2: fp8_down_unit<kH, kI, kBlock, 2>(e, g, rb, smem, out, out_index); break;
    case 3: fp8_down_unit<kH, kI, kBlock, 3>(e, g, rb, smem, out, out_index); break;
    default: fp8_down_unit<kH, kI, kBlock, 4>(e, g, rb, smem, out, out_index); break;
  }
}

#endif  // __CUDACC__

}  // namespace dgpp::step
