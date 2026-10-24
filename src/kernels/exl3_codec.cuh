#pragma once
// Device side of the EXL3 trellis format (models/exl3_matrix.hpp holds the
// contract and the host codec): the mul1 codebook decode, the per-lane
// state windows that turn one 16x16 tile into the two mma B fragments of
// its lane, the fp32 128-point rotation a warp applies to one row block,
// and the fp16 mma. The window arithmetic and the codebook instruction
// sequence are exllamav3's (exl3_dq.cuh, codebook.cuh, hadamard_inner.cuh;
// MIT, docs/licenses/exllamav3.md), kept op for op so the values and the
// rotation's fp32 sums are the reference's.
#include <cuda_fp16.h>

#include <cstdint>

#include "common/dtypes.hpp"
#include "models/exl3_matrix.hpp"

namespace dgpp {
namespace exl3 {

// Two states -> the half2 of their codebook values (low half = s0's).
__device__ __forceinline__ uint32_t mul1_pair(uint32_t s0, uint32_t s1) {
  const uint32_t x0 = s0 * kExl3Mul1Multiplier, x1 = s1 * kExl3Mul1Multiplier;
  const uint32_t h0 = __dp4a(x0, 0x01010101u, 0x6400u), h1 = __dp4a(x1, 0x01010101u, 0x6400u);
  const half2 h = __halves2half2(__ushort_as_half(static_cast<uint16_t>(h0)), __ushort_as_half(static_cast<uint16_t>(h1)));
  const half2 r = __hfma2(h, __half2half2(__ushort_as_half(kExl3Mul1KInv)), __half2half2(__ushort_as_half(kExl3Mul1KBias)));
  return *reinterpret_cast<const uint32_t*>(&r);
}

// A word index into the tile's ring: the windows start inside the tile
// (b0 < 2 x the stream length), so at most one wrap.
template <int kWords>
__device__ __forceinline__ int wrap(int i) {
  return i >= kWords ? i - kWords : i;
}

// The low 32 bits of (a:b) >> s for 0 <= s < 64 (a the earlier stream word).
__device__ __forceinline__ uint32_t fshift(uint32_t b, uint32_t a, int s) {
  const uint64_t merged = (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
  return static_cast<uint32_t>(merged >> s);
}

// Position p's state out of a tile's words (any width): the window's two
// words and the funnel shift that puts its last bit at bit 0.
template <int kBits>
__device__ __forceinline__ uint32_t state_at(const uint32_t* words, int p) {
  constexpr int kWords = 8 * kBits;
  const int b0 = p * kBits + kBits - 16 + 256 * kBits;  // the window's first bit (+ one ring: never negative)
  const int b1 = b0 + 16;                              // one past its last bit
  const int i0 = b0 / 32, i1 = (b1 - 1) / 32;
  const int s0 = (i1 + 1) * 32 - b1;
  return __funnelshift_r(words[wrap<kWords>(i1)], words[wrap<kWords>(i0)], s0) & 0xFFFFu;
}

// Lane t's eight states (positions 8 t .. 8 t + 7) as the four B words of
// its two n8 fragments: b[0] = positions 0, 1 and b[1] = 2, 3 (column
// t / 4), b[2] = 4, 5 and b[3] = 6, 7 (column t / 4 + 8). The fixed-width
// forms read two words per group of states: a group's window — 16 + 3 or
// 7 times the width bits — never spans three words at these widths (the
// window starts 8 t bits + width - 16 into the stream; exllamav3's dq8 /
// dq4 arithmetic), and the states inside the group are shifts of one
// funnel result.
template <int kBits>
__device__ __forceinline__ void lane_fragments(const uint32_t* words, int lane, uint32_t (&b)[4]) {
  constexpr int kWords = 8 * kBits;
  uint32_t w[8];
  if constexpr (kBits == 3) {
    const int b1 = (lane * 8 + 257) * kBits;  // one past position 0's last bit
    const int b0 = b1 - 16;
    const int b2 = b1 + kBits * 7;
    const int i0 = b0 / 32, i2 = (b2 - 1) / 32;
    const int s2 = (i2 + 1) * 32 - b2;
    const uint32_t lo = words[wrap<kWords>(i0)], hi = words[wrap<kWords>(i2)];
    w[7] = fshift(hi, lo, s2);
    w[6] = w[7] >> kBits;
    w[5] = w[6] >> kBits;
    w[4] = w[5] >> kBits;
    w[3] = fshift(hi, lo, s2 + 4 * kBits);
    w[2] = w[3] >> kBits;
    w[1] = w[2] >> kBits;
    w[0] = w[1] >> kBits;
  } else if constexpr (kBits == 4) {
    const int i1 = lane;                 // position 8 t + 7's word
    const int i0 = (i1 + 31) & 31;       // the word before (the ring)
    const uint32_t lo = words[i0], hi = words[i1];
    const uint32_t s = __funnelshift_r(hi, lo, 20);
    w[7] = hi & 0xFFFFu;
    w[6] = (hi >> 4) & 0xFFFFu;
    w[5] = (hi >> 8) & 0xFFFFu;
    w[4] = (hi >> 12) & 0xFFFFu;
    w[3] = (hi >> 16) & 0xFFFFu;
    w[2] = s & 0xFFFFu;
    w[1] = (s >> 4) & 0xFFFFu;
    w[0] = (s >> 8) & 0xFFFFu;
  } else if constexpr (kBits == 5 || kBits == 6 || kBits == 8) {
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      const int p = lane * 8 + 4 * g;
      const int b0 = (p + 257) * kBits - 16;
      const int b1 = b0 + 3 * kBits;
      const int b2 = b1 + 16;
      const int i0 = b0 / 32, i2 = (b2 - 1) / 32;
      const int s2 = (i2 + 1) * 32 - b2;
      const uint32_t lo = words[wrap<kWords>(i0)], hi = words[wrap<kWords>(i2)];
      w[4 * g + 3] = fshift(hi, lo, s2);
      w[4 * g + 2] = fshift(hi, lo, s2 + kBits);
      w[4 * g + 1] = fshift(hi, lo, s2 + 2 * kBits);
      w[4 * g + 0] = fshift(hi, lo, s2 + 3 * kBits);
    }
  } else if constexpr (kBits == 7) {
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      const int p = lane * 8 + 2 * g;
      const int b0 = (p + 257) * kBits - 16;
      const int b1 = b0 + kBits;
      const int b2 = b1 + 16;
      const int i0 = b0 / 32, i2 = (b2 - 1) / 32;
      const int s2 = (i2 + 1) * 32 - b2;
      const uint32_t lo = words[wrap<kWords>(i0)], hi = words[wrap<kWords>(i2)];
      w[2 * g + 1] = fshift(hi, lo, s2);
      w[2 * g + 0] = fshift(hi, lo, s2 + kBits);
    }
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j) w[j] = state_at<kBits>(words, lane * 8 + j);
  }
  b[0] = mul1_pair(w[0] & 0xFFFFu, w[1] & 0xFFFFu);
  b[1] = mul1_pair(w[2] & 0xFFFFu, w[3] & 0xFFFFu);
  b[2] = mul1_pair(w[4] & 0xFFFFu, w[5] & 0xFFFFu);
  b[3] = mul1_pair(w[6] & 0xFFFFu, w[7] & 0xFFFFu);
}

// The 128-point rotation of one row block across a warp: lane t holds
// elements 4 t .. 4 t + 3. The in-lane stages (strides 1 and 2), then the
// shuffle stages (strides 4 .. 64: lane bit i, the lane with the bit set
// negates its own value before adding its partner's), then one multiply
// by `scale` — the reference's operations in its order (hadamard_inner.cuh
// shuffle_had_f4x32), and models/exl3_matrix.hpp's exl3_hadamard128_f32.
__device__ __forceinline__ void had128_warp(float& v0, float& v1, float& v2, float& v3, int lane, float scale) {
  const float s0 = v0 + v1, d0 = v0 - v1, s1 = v2 + v3, d1 = v2 - v3;
  v0 = s0 + s1;
  v1 = d0 + d1;
  v2 = s0 - s1;
  v3 = d0 - d1;
#pragma unroll
  for (int i = 1; i < 32; i <<= 1) {
    const float p0 = __shfl_xor_sync(0xFFFFFFFFu, v0, i);
    const float p1 = __shfl_xor_sync(0xFFFFFFFFu, v1, i);
    const float p2 = __shfl_xor_sync(0xFFFFFFFFu, v2, i);
    const float p3 = __shfl_xor_sync(0xFFFFFFFFu, v3, i);
    const bool flip = (lane & i) != 0;
    v0 = (flip ? -v0 : v0) + p0;
    v1 = (flip ? -v1 : v1) + p1;
    v2 = (flip ? -v2 : v2) + p2;
    v3 = (flip ? -v3 : v3) + p3;
  }
  v0 *= scale;
  v1 *= scale;
  v2 *= scale;
  v3 *= scale;
}

// The A operand of one row block from the lane's four x (bf16, xr) and
// four suh (f16, sr): x times suh in fp32, the rotation, the scale, rounded
// to fp16 — stored as four f16 at xh + 4 lane (8-byte aligned).
__device__ __forceinline__ void rotate_in_regs(uint2 xr, uint2 sr, uint16_t* __restrict__ xh, int lane) {
  float v0 = bf16_bits_to_float(static_cast<uint16_t>(xr.x)) * __half2float(__ushort_as_half(static_cast<uint16_t>(sr.x)));
  float v1 = bf16_bits_to_float(static_cast<uint16_t>(xr.x >> 16)) * __half2float(__ushort_as_half(static_cast<uint16_t>(sr.x >> 16)));
  float v2 = bf16_bits_to_float(static_cast<uint16_t>(xr.y)) * __half2float(__ushort_as_half(static_cast<uint16_t>(sr.y)));
  float v3 = bf16_bits_to_float(static_cast<uint16_t>(xr.y >> 16)) * __half2float(__ushort_as_half(static_cast<uint16_t>(sr.y >> 16)));
  had128_warp(v0, v1, v2, v3, lane, kExl3HadamardScale);
  const half2 h01 = __floats2half2_rn(v0, v1), h23 = __floats2half2_rn(v2, v3);
  uint2 o;
  o.x = *reinterpret_cast<const uint32_t*>(&h01);
  o.y = *reinterpret_cast<const uint32_t*>(&h23);
  *reinterpret_cast<uint2*>(xh + 4 * lane) = o;
}
// The same from memory (x and suh 8-byte aligned).
__device__ __forceinline__ void rotate_in_warp(const uint16_t* __restrict__ x, const uint16_t* __restrict__ suh,
                                               uint16_t* __restrict__ xh, int lane) {
  rotate_in_regs(*reinterpret_cast<const uint2*>(x + 4 * lane), *reinterpret_cast<const uint2*>(suh + 4 * lane), xh,
                 lane);
}

// The output of one row block: the fp32 pre-rotation values of elements
// 4 lane .. 4 lane + 3 rotated, scaled, times svh (f16 [128]) in fp32.
__device__ __forceinline__ void rotate_out_warp(float& v0, float& v1, float& v2, float& v3,
                                                const uint16_t* __restrict__ svh, int lane) {
  had128_warp(v0, v1, v2, v3, lane, kExl3HadamardScale);
  const uint2 sr = *reinterpret_cast<const uint2*>(svh + 4 * lane);
  v0 *= __half2float(__ushort_as_half(static_cast<uint16_t>(sr.x)));
  v1 *= __half2float(__ushort_as_half(static_cast<uint16_t>(sr.x >> 16)));
  v2 *= __half2float(__ushort_as_half(static_cast<uint16_t>(sr.y)));
  v3 *= __half2float(__ushort_as_half(static_cast<uint16_t>(sr.y >> 16)));
}

// mma.m16n8k16 on fp16 A and B fragments, fp32 accumulators.
__device__ __forceinline__ void mma_f16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// The A fragments of one 16 x 16 fp16 tile in row-major shared memory
// (row stride in bytes `row_bytes`, 16-byte aligned rows): lane l
// addresses row l % 16 at k offset 8 (l / 16); r[0..3] come out as the mma
// wants them (rows 0-7 / k 0-7, rows 8-15 / k 0-7, rows 0-7 / k 8-15, rows
// 8-15 / k 8-15).
__device__ __forceinline__ void ldmatrix_a(uint32_t (&r)[4], const uint16_t* tile, int row_bytes, int lane) {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(tile) + static_cast<size_t>(lane % 16) * row_bytes +
                     static_cast<size_t>(lane / 16) * 16;
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

}  // namespace exl3
}  // namespace dgpp
