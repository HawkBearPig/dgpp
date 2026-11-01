#pragma once
// Device primitives the step engine's consumers share (MOONSHOT-1.md
// section 2.7): the tensor-core instruction, the byte-permute decode of
// MXFP4 codes, the exact power-of-two bf16x2 scale, the asynchronous
// shared-memory copies and the tile-major layout's address.
#include <cstddef>
#include <cstdint>

#include "common/dtypes.hpp"

namespace dgpp::step {

// The tile-major layout of a 4-bit matrix (section 2.4): 16-row tiles, a
// tile's 32-code chunks contiguous as [chunk][16 rows][16 bytes], so one
// lane-contiguous 512-byte load is two chunks of sixteen rows. Rows are a
// multiple of 16 and K of 32. The scales keep their row-major grid.
constexpr int kTileRows = 16;
constexpr int kChunkCodes = 32;
constexpr int kChunkBytes = 16;
__host__ __device__ constexpr size_t tile_major_bytes(int rows, int k) {
  return static_cast<size_t>(rows) * static_cast<size_t>(k) / 2;
}
__host__ __device__ constexpr size_t tile_major_tile_bytes(int k) {
  return static_cast<size_t>(k / kChunkCodes) * kTileRows * kChunkBytes;
}
// The byte holding codes (k, k + 1) of `row` in a [rows, k] tile-major matrix.
__host__ __device__ constexpr size_t tile_major_offset(int row, int k, int K) {
  return static_cast<size_t>(row / kTileRows) * tile_major_tile_bytes(K) +
         static_cast<size_t>(k / kChunkCodes) * (kTileRows * kChunkBytes) + static_cast<size_t>(row % kTileRows) * kChunkBytes +
         static_cast<size_t>((k % kChunkCodes) / 2);
}

#if defined(__CUDACC__)

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}
// bf16x2 multiply by a power of two in both halves: exact.
__device__ __forceinline__ uint32_t bf16x2_mul_pow2(uint32_t a, uint32_t s) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(s));
  return d;
}
// Eight consecutive e2m1 codes (nibble i of g is code i in k order) as four
// bf16x2 words {(c0,c1), (c2,c3), (c4,c5), (c6,c7)} times s2, a bf16x2
// {2^(e-127), 2^(e-127)}: byte-permute tables for the magnitudes, the sign
// replicated into bit 7 of each byte.
__device__ __forceinline__ void decode8_e2m1(uint32_t g, uint32_t s2, uint32_t (&w)[4]) {
  const uint32_t m = g & 0x77777777u, t = g & 0x88888888u, t2 = t << 4;
  constexpr uint32_t kHi0 = 0x3F3F3F00u, kHi1 = 0x40404040u, kLo0 = 0xC0800000u, kLo1 = 0xC0804000u;
#pragma unroll
  for (int tile = 0; tile < 2; ++tile) {
    const uint32_t mm = tile ? (m >> 16) : m, tt = tile ? (t >> 16) : t, tt2 = tile ? (t2 >> 16) : t2;
    const uint32_t lo = prmt(kLo0, kLo1, mm);
    const uint32_t hi = prmt(kHi0, kHi1, mm) | (prmt(tt2, tt, 0xD9C8u) & 0x80808080u);
    w[2 * tile] = bf16x2_mul_pow2(prmt(lo, hi, 0x5140u), s2);
    w[2 * tile + 1] = bf16x2_mul_pow2(prmt(lo, hi, 0x7362u), s2);
  }
}
// The e8m0 scale byte as the bf16x2 {2^(e-127), 2^(e-127)} decode8_e2m1 takes.
__device__ __forceinline__ uint32_t e8m0_bf16x2(uint32_t e) { return (e << 7) * 0x00010001u; }

__device__ __forceinline__ void mma_bf16_16x8x16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// The same with a zero C operand: the first mma of an accumulation block,
// so the block's partial needs no reset. Bitwise the mma over a zeroed c.
__device__ __forceinline__ void mma_bf16_16x8x16_zero(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.f));
}

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// The graph engine's bf16 helpers (common/dtypes.hpp), so the consumers'
// roundings are the slot kernels' op for op.
__device__ __forceinline__ float bf16_to_f32(uint16_t b) { return bf16_bits_to_float(b); }
__device__ __forceinline__ uint16_t f32_to_bf16(float f) { return float_to_bf16_bits(f); }

#endif  // __CUDACC__

}  // namespace dgpp::step
