#pragma once
// The bf12 GEMV's row chain (2026-09-19, factored out of bf12_gemv.cu as
// bf16_gemv.cuh was out of bf16_gemv.cu, and for the same reason): the fused
// decode kernels — the Qwen GR site's norm-staged GEMVs, the multi-problem
// launch of the GDN's input projections — run this code on a packed row, and
// a row's chain is the same sequence of FMAs on the same values whichever
// launcher issued its loads. That chain is bf16_gemv::row_dots' (the lane /
// step ownership and the element order), so every fusion is bitwise the
// bf16 launch it replaces.
//
// THE TAIL (format v2). A row is k / 1024 super-blocks (bf12_gemv.hpp) and
// then the k % 1024 columns that do not fill one — Qwen's hidden 2560 and
// its 1536-wide slices leave 512, the GR up projection's k = 320 is all
// tail. No padding (a padded column is bytes streamed for nothing: 2560
// padded to 3072 gives most of the format's gain back), so the tail is cut
// into units by the steps it holds — a step is 256 columns, a lane's share
// of it eight:
//   P  two FULL steps: [sm 512 B | exp 256 B]. Lane l owns sm bytes
//      [16 l, +16) — its steps' bytes as the super-block's A segment — and
//      exp bytes [8 l, +8): a 16-byte exp vector is shared by a lane PAIR.
//   F  one step of L lanes (32: full; fewer: the partial last step):
//      [sm 8 L B | exp 4 L B], each rounded up to 16. Lane l (< L) owns sm
//      bytes [8 l, +8) — a vector per lane pair — and exp bytes [4 l, +4) —
//      a vector per four lanes.
// A tail is [P]? [F32]? [F L]? in that order: 512 -> P (768 B), 320 -> F32
// F8 (384 + 96 B), 768 -> P F32. Twelve bits a weight wherever L is a
// multiple of four. Every load is a 16-byte-aligned vector inside the row.
#include <bit>
#include <cstddef>
#include <cstdint>

#include "common/dtypes.hpp"
#include "kernels/bf12_gemv.hpp"
#include "kernels/bf16_gemv.cuh"
#include "kernels/gemv_common.cuh"

namespace dgpp {
namespace bf12_gemv {

constexpr int kTailPairBytes = 768;  // a P unit
constexpr int kTailFullBytes = 384;  // an F unit of 32 lanes

// The tail's units for a row of k columns.
struct Tail {
  int nsb = 0;    // whole super-blocks before it
  int cols = 0;   // its columns
  int pair = 0;   // 1: a P unit
  int full = 0;   // 1: an F unit of 32 lanes
  int lanes = 0;  // the partial F unit's lanes (0: none)
};
__host__ __device__ inline Tail tail_of(int k) {
  Tail t;
  t.nsb = k / kBf12Super;
  t.cols = k - t.nsb * kBf12Super;
  const int steps = t.cols / 256;
  t.pair = steps >> 1;
  t.full = steps & 1;
  t.lanes = (t.cols - steps * 256) / 8;
  return t;
}
__host__ __device__ inline int round16(int bytes) { return (bytes + 15) & ~15; }
// The partial unit's sm segment (its exp segment follows).
__host__ __device__ inline int tail_part_sm_bytes(int lanes) { return round16(8 * lanes); }
__host__ __device__ inline int tail_bytes(const Tail& t) {
  return t.pair * kTailPairBytes + t.full * kTailFullBytes +
         (t.lanes > 0 ? tail_part_sm_bytes(t.lanes) + round16(4 * t.lanes) : 0);
}

// One step of one lane: eight elements at columns [c0, c0 + 8). lo/hi carry
// their sign+mantissa bytes, ew their exponent nibbles. The weight's f32
// bits are its bf16 bits << 16 (bf16_bits_to_float), rebuilt from the byte
// and base + code; an escaped element takes its bits from the row's table.
// The FMA order is bf16_gemv::row_dots' (elements ascending per row).
// sx holds the staged activations at row stride sk, the step's elements at
// staged column cs (whole staged rows: sk = k, cs = c0; a window of them:
// the window's).
template <int kRows>
__device__ __forceinline__ void consume_step(uint32_t lo, uint32_t hi, uint32_t ew,
                                             uint32_t base23,
                                             const uint16_t* __restrict__ sx, int sk, int cs,
                                             int c0, const uint32_t* __restrict__ esc,
                                             uint32_t esc_b, uint32_t esc_e,
                                             float (&acc)[kRows]) {
  uint32_t wbits[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint32_t b = ((j < 4 ? lo : hi) >> (8 * (j & 3))) & 0xFFu;
    const uint32_t code = (ew >> (4 * j)) & 0xFu;
    wbits[j] = ((b & 0x80u) << 24) | ((b & 0x7Fu) << 16) | (base23 + (code << 23));
  }
  // Any nibble == 15 (about one step in 25 on a trained matrix).
  const uint32_t any15 = ew & (ew >> 1) & (ew >> 2) & (ew >> 3) & 0x11111111u;
  if (any15 != 0u) {
    for (int j = 0; j < 8; ++j) {
      if (((ew >> (4 * j)) & 0xFu) != 0xFu) continue;
      const uint32_t col = static_cast<uint32_t>(c0 + j);
      for (uint32_t e = esc_b; e < esc_e; ++e) {
        const uint32_t ent = esc[e];
        if ((ent >> 16) == col) {
          wbits[j] = (ent & 0xFFFFu) << 16;
          break;
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const uint4 xv = *reinterpret_cast<const uint4*>(sx + static_cast<size_t>(r) * sk + cs);
    const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float w = std::bit_cast<float>(wbits[j]);
      const float x =
          bf16_bits_to_float(static_cast<uint16_t>((xw[j >> 1] >> (16 * (j & 1))) & 0xFFFFu));
      acc[r] = __fmaf_rn(w, x, acc[r]);
    }
  }
}

// One step of one lane: eight elements' bf16 bits as one 16-byte vector
// (the expansion kernel and the tensor-core decode form's fragments).
// base7: the row's window base at the bf16 exponent field ((base23 >> 23) << 7).
__device__ __forceinline__ uint4 expand_step(uint32_t lo, uint32_t hi, uint32_t ew, uint32_t base7,
                                             int c0, const uint32_t* __restrict__ esc,
                                             uint32_t esc_b, uint32_t esc_e) {
  uint32_t v[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint32_t b = ((j < 4 ? lo : hi) >> (8 * (j & 3))) & 0xFFu;
    const uint32_t code = (ew >> (4 * j)) & 0xFu;
    v[j] = ((b & 0x80u) << 8) | (b & 0x7Fu) | (base7 + (code << 7));
  }
  const uint32_t any15 = ew & (ew >> 1) & (ew >> 2) & (ew >> 3) & 0x11111111u;
  if (any15 != 0u) {
    for (int j = 0; j < 8; ++j) {
      if (((ew >> (4 * j)) & 0xFu) != 0xFu) continue;
      const uint32_t col = static_cast<uint32_t>(c0 + j);
      for (uint32_t e = esc_b; e < esc_e; ++e) {
        const uint32_t ent = esc[e];
        if ((ent >> 16) == col) {
          v[j] = ent & 0xFFFFu;
          break;
        }
      }
    }
  }
  uint4 o;
  o.x = v[0] | (v[1] << 16);
  o.y = v[2] | (v[3] << 16);
  o.z = v[4] | (v[5] << 16);
  o.w = v[6] | (v[7] << 16);
  return o;
}

// A lane's tail loads, issued together (at most six vectors).
struct TailLoads {
  uint4 pair_sm, pair_exp, full_sm, full_exp, part_sm, part_exp;
};
__device__ __forceinline__ void load_tail(const uint8_t* __restrict__ tp, const Tail& t, int lane,
                                          TailLoads& v) {
  const uint8_t* p = tp;
  if (t.pair != 0) {
    v.pair_sm = *reinterpret_cast<const uint4*>(p + lane * 16);
    v.pair_exp = *reinterpret_cast<const uint4*>(p + 512 + (lane >> 1) * 16);
    p += kTailPairBytes;
  }
  if (t.full != 0) {
    v.full_sm = *reinterpret_cast<const uint4*>(p + (lane >> 1) * 16);
    v.full_exp = *reinterpret_cast<const uint4*>(p + 256 + (lane >> 2) * 16);
    p += kTailFullBytes;
  }
  if (lane < t.lanes) {
    v.part_sm = *reinterpret_cast<const uint4*>(p + (lane >> 1) * 16);
    v.part_exp = *reinterpret_cast<const uint4*>(p + tail_part_sm_bytes(t.lanes) + (lane >> 2) * 16);
  }
}
__device__ __forceinline__ uint32_t word_of(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}
// The tail's steps in column order. c0: the row's column of the tail's
// first step at this lane (nsb * 1024 + lane * 8); cs: its staged column.
template <int kRows>
__device__ __forceinline__ void consume_tail(const TailLoads& v, const Tail& t, int lane,
                                             uint32_t base23, const uint16_t* __restrict__ sx,
                                             int sk, int cs, int c0,
                                             const uint32_t* __restrict__ esc, uint32_t eb,
                                             uint32_t ee, float (&acc)[kRows]) {
  const int odd = lane & 1;
  if (t.pair != 0) {
    consume_step<kRows>(v.pair_sm.x, v.pair_sm.y, odd ? v.pair_exp.z : v.pair_exp.x, base23, sx, sk,
                        cs, c0, esc, eb, ee, acc);
    consume_step<kRows>(v.pair_sm.z, v.pair_sm.w, odd ? v.pair_exp.w : v.pair_exp.y, base23, sx, sk,
                        cs + 256, c0 + 256, esc, eb, ee, acc);
    cs += 512;
    c0 += 512;
  }
  if (t.full != 0) {
    consume_step<kRows>(odd ? v.full_sm.z : v.full_sm.x, odd ? v.full_sm.w : v.full_sm.y,
                        word_of(v.full_exp, lane & 3), base23, sx, sk, cs, c0, esc, eb, ee, acc);
    cs += 256;
    c0 += 256;
  }
  if (lane < t.lanes)
    consume_step<kRows>(odd ? v.part_sm.z : v.part_sm.x, odd ? v.part_sm.w : v.part_sm.y,
                        word_of(v.part_exp, lane & 3), base23, sx, sk, cs, c0, esc, eb, ee, acc);
}

// One warp, one PACKED row (wr: its bytes, base23 / [eb, ee): its window
// base and escapes), kRows staged activation rows of stride k: the lane's
// chain into acc (zeroed here; no reduce). kSB super-blocks (3 * kSB warp
// loads) are in flight per lane before any is consumed — the bf16 core's
// reason (bytes in flight), the same shape; the tail's loads follow the
// last batch.
template <int kRows, int kSB>
__device__ __forceinline__ void packed_row_chain(const uint8_t* __restrict__ wr, uint32_t base23,
                                                 const uint32_t* __restrict__ esc, uint32_t eb,
                                                 uint32_t ee, const uint16_t* __restrict__ sx,
                                                 int k, int lane, float (&acc)[kRows]) {
#pragma unroll
  for (int r = 0; r < kRows; ++r) acc[r] = 0.f;
  const Tail t = tail_of(k);
  const int nsb = t.nsb;
  for (int s0 = 0; s0 < nsb; s0 += kSB) {
    uint4 a[kSB], b[kSB], e[kSB];
#pragma unroll
    for (int s = 0; s < kSB; ++s) {
      // Past the row's last super-block a lane re-reads a block it holds
      // (uniform across the warp; never consumed).
      const int sb = s0 + s < nsb ? s0 + s : s0;
      const uint8_t* p = wr + static_cast<size_t>(sb) * kBf12SuperBytes + lane * 16;
      a[s] = *reinterpret_cast<const uint4*>(p);
      b[s] = *reinterpret_cast<const uint4*>(p + 512);
      e[s] = *reinterpret_cast<const uint4*>(p + 1024);
    }
#pragma unroll
    for (int s = 0; s < kSB; ++s) {
      if (s0 + s >= nsb) break;
      const int cb = (s0 + s) * kBf12Super + lane * 8;
      consume_step<kRows>(a[s].x, a[s].y, e[s].x, base23, sx, k, cb, cb, esc, eb, ee, acc);
      consume_step<kRows>(a[s].z, a[s].w, e[s].y, base23, sx, k, cb + 256, cb + 256, esc, eb, ee, acc);
      consume_step<kRows>(b[s].x, b[s].y, e[s].z, base23, sx, k, cb + 512, cb + 512, esc, eb, ee, acc);
      consume_step<kRows>(b[s].z, b[s].w, e[s].w, base23, sx, k, cb + 768, cb + 768, esc, eb, ee, acc);
    }
  }
  if (t.cols != 0) {
    TailLoads v;
    load_tail(wr + static_cast<size_t>(nsb) * kBf12SuperBytes, t, lane, v);
    const int c0 = nsb * kBf12Super + lane * 8;
    consume_tail<kRows>(v, t, lane, base23, sx, k, c0, c0, esc, eb, ee, acc);
  }
}

// bf16_gemv::row_dots on row `row` of a packed matrix: acc[r] =
// dot(w_row, sx[r]), reduced over the warp — bitwise the bf16 chain on the
// unpacked row. A row kept bf16 (uniform across its warp) runs that chain
// itself. row_bytes: bf12_row_bytes(k).
template <int kRows, int kSB>
__device__ __forceinline__ void row_dots(const uint8_t* __restrict__ packed,
                                         const uint32_t* __restrict__ rows,
                                         const uint32_t* __restrict__ esc,
                                         const uint16_t* __restrict__ raw, int row, int k,
                                         size_t row_bytes, const uint16_t* __restrict__ sx,
                                         int lane, float (&acc)[kRows]) {
  const uint32_t eb = rows[2 * row];
  const uint32_t base23 = rows[2 * row + 1];
  const uint32_t ee = rows[2 * row + 2];
  if ((base23 & kBf12RawRow) != 0u) {
    bf16_gemv::row_dots<kRows>(raw + static_cast<size_t>(base23 & ~kBf12RawRow) * k, sx, k, lane,
                               acc);
    return;
  }
  packed_row_chain<kRows, kSB>(packed + static_cast<size_t>(row) * row_bytes, base23, esc, eb, ee,
                               sx, k, lane, acc);
  gemv::warp_reduce<kRows>(acc);
}

}  // namespace bf12_gemv
}  // namespace dgpp
