#pragma once
// The NVFP4 GEMV core (2026-09-08, docs/nvfp4_plan.md §3): the decode
// step's routed experts at m = 1..4 activation rows.
//
// SHAPE: the fp8 core's (fp8_gemv.cuh) — a bandwidth kernel shaped for
// bytes in flight: 16-byte weight chunks per lane, a whole batch issued
// before any is consumed, no barrier in the k loop, a fixed xor-shuffle
// tree. A chunk holds 32 e2m1 codes (two per byte, low nibble = even
// element) and spans two 16-wide scale blocks. The row geometry is a
// COMPILE-TIME function of K (one of 32, 64, ..., 4096): a lane owns up to
// four chunks of a row, a row is covered by K/128 lanes (32 at K = 4096,
// four at the world-4 down projection's K = 512), and a warp advances
// rows_per_step = 32 / lanes_per_row rows per step, two steps per warp —
// so every lane keeps eight 16-byte loads in flight whatever the row
// length, and every loop below unrolls with constant indices (the first
// draft indexed its accumulators by a lane-dependent runtime value, which
// put them in local memory: 14 GB/s). Which lane owns which chunk is a
// function of K alone, and every row's chain is the same sequence of FMAs
// whatever kRows or which launcher issued it — the property the slot path
// (kRows = 1) and the host grouped path (kRows <= 4) are pinned on, bitwise.
//
// NUMERICS (the plan's §3.1): an e2m1 code has <= 2 significant bits and an
// e4m3 scale <= 4, so the scaled weight fp4 * s is EXACT in fp32 (and in
// bf16); its product with a bf16 activation is exact in fp32; the
// accumulation is fp32 FMA in a fixed order; the one inexact operation of
// the format — the division by the tensor's global scale — is applied once
// to the finished dot. Nothing is rounded to bf16 before the epilogue.
//
// DECODE: e2m1 -> fp16 by bit placement, no table and no arch-specific
// instruction: the code's three magnitude bits (e1 e0 m) land in fp16 bits
// 11..9 (the low two exponent bits and the top mantissa bit), the sign in
// bit 15, and the value read back is the e2m1 value times 2^-14 for every
// code, subnormals included (e = 0: fp16 subnormal m * 2^-15 = m/2 *
// 2^-14). The 2^14 folds into the block scale (s * 16384, exact), so the
// per-code cost is the bit placement, one f16 -> f32 conversion and the
// multiply — no fp16 arithmetic anywhere.
//
// CONTRACT: K a multiple of 32 with at most 16 chunks per lane (K <=
// 16384; the compiled set is dispatch_k's); payload rows 16-byte aligned.
// Scales are e4m3 bytes [n, K/16] row-major; NaN scale codes (0x7F/0xFF)
// propagate as NaN.
//
// MXFP4 (2026-09-13, DeepSeek-V4.1-Flash, docs/deepseek_v41_flash_plan.md
// D2; kGroup = 32): one e8m0 scale byte per chunk (scales [n, K/32]), the
// value 2^(byte - 127), NO global scale. The chunk's 32 codes are decoded
// through the same bit placement (code x 2^-14 as f16, read as f32) and
// multiplied by the scale pre-shifted by 2^14 (2^(byte - 113), an exact
// fp32 power of two down to the denormals) — one fp32 multiply per code,
// exact for EVERY e8m0 exponent (the f16 product of the NVFP4 path is
// exact only inside f16's range, which e4m3 guarantees and e8m0 does
// not). Byte 255 is NaN and propagates; 241..254 saturate to +inf (a
// scale above 2^113 is outside any weight's range). The FMA chain, the
// lane geometry and the xor tree are the NVFP4 path's, so the slot, the
// grouped and the single-matrix launches stay bitwise twins per row, and
// the epilogue stores the accumulator undivided.
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/dtypes.hpp"
#include "kernels/gemv_common.cuh"

namespace dgpp {
namespace fp4_gemv {

using gemv::kChunkBytes;
using gemv::kMaxRows;
using gemv::kThreads;
using gemv::kWarps;
using gemv::smem_bytes;
using gemv::stage_activations;

constexpr int kCodesPerChunk = 2 * kChunkBytes;   // 32
constexpr int kGroup = 16;                        // codes per e4m3 scale (NVFP4)
constexpr int kMxGroup = 32;                      // codes per e8m0 scale (MXFP4)
constexpr int kMaxChunksPerLane = 4;
// The scale bytes per row for a K-wide row under either group.
__host__ __device__ constexpr int scale_cols_of(int k, int group) { return k / group; }
// One row step per warp (2026-09-08, moe_slot_bench, twice: with the
// activation window re-read per step and with it loaded once per chunk
// column): two steps put eight loads in flight per lane but halved the
// block count, and the launch ran 4-10 % slower at one row; four steps
// 8 %. The way to more loads in flight without fewer blocks is the pair
// (warp_row_dots_pair: gate and up issued together). A row's chain is the
// same whatever the step count, so the outputs are bitwise across it.
constexpr int kSteps = 1;                         // row steps per warp
constexpr int kMaxPasses = 4;                     // chunk passes per lane
constexpr int kMaxK = kCodesPerChunk * 32 * kMaxChunksPerLane * kMaxPasses;  // 16384

// The row geometry as a function of K = 32 x row_chunks (2026-09-09,
// docs/glm47_plan.md D2 — GLM-4.7's 5120 / 384 / 3072 widths): the
// power-of-two widths keep their 2026-09-08 geometry exactly (up to four
// chunks per lane, 1..32 lanes per row); any other multiple of 32 puts the
// largest power of two <= 32 that divides row_chunks on a row, each lane
// owning row_chunks / lanes chunks consumed in PASSES of at most four (the
// loads in flight per lane). A row's FMA chain — each lane's chunks in k
// order, the fixed xor tree across the lanes — is the same whatever the
// pass split, so every bitwise pin (slot == grouped == launcher) holds.
__host__ __device__ constexpr int pow2_divisor_le32(int r) {
  int l = 1;
  while (l < 32 && r % (l * 2) == 0) l *= 2;
  return l;
}
__host__ __device__ constexpr int lanes_per_row_of(int k) {
  const int r = k / kCodesPerChunk;
  const int c0 = r < kMaxChunksPerLane ? r : kMaxChunksPerLane;
  if (c0 > 0 && r % c0 == 0) {
    const int l = r / c0;
    if (l >= 1 && l <= 32 && (l & (l - 1)) == 0) return l;
  }
  return pow2_divisor_le32(r);
}
__host__ __device__ constexpr int chunks_of(int k) {
  return (k / kCodesPerChunk) / lanes_per_row_of(k);
}
__host__ __device__ constexpr bool k_supported(int k) {
  return k >= kCodesPerChunk && k % kCodesPerChunk == 0 &&
         chunks_of(k) <= kMaxChunksPerLane * kMaxPasses;
}

__host__ inline bool shape_ok(const void* payload, int k) {
  return k_supported(k) && gemv::aligned16(payload);
}

// The compile-time row geometry.
template <int K>
struct Geom {
  static_assert(k_supported(K), "fp4_gemv: K must be a multiple of 32 with at most 16 chunks per lane");
  static constexpr int row_chunks = K / kCodesPerChunk;
  static constexpr int lanes_per_row = lanes_per_row_of(K);   // 1 .. 32
  static constexpr int chunks = row_chunks / lanes_per_row;   // per lane, over every pass
  static constexpr int passes = (chunks + kMaxChunksPerLane - 1) / kMaxChunksPerLane;
  static constexpr int rows_per_step = 32 / lanes_per_row;
  static constexpr int rows_per_warp = kSteps * rows_per_step;
  static constexpr int rows_per_block = kWarps * rows_per_warp;
  static constexpr int row_bytes = K / 2;
  static constexpr int scale_cols = K / kGroup;
  // Chunks in pass p (at least 1 so a discarded instantiation stays legal).
  static constexpr int pass_chunks(int p) {
    const int left = chunks - p * kMaxChunksPerLane;
    return left < 1 ? 1 : (left < kMaxChunksPerLane ? left : kMaxChunksPerLane);
  }
};

// The runtime twins for launch arithmetic (the host dispatches on k).
__host__ __device__ constexpr int rows_per_step_of(int k) {
  return 32 / lanes_per_row_of(k);
}
__host__ __device__ constexpr int rows_per_warp_of(int k) {
  return kSteps * rows_per_step_of(k);
}
__host__ __device__ constexpr int rows_per_block_of(int k) {
  return kWarps * rows_per_warp_of(k);
}

// The two e4m3 scales a chunk spans, {s(e0), s(e0+16)}, as f16 — exact
// (e4m3 is a subset of f16: exponents 2^-9 .. 2^8, three mantissa bits).
__device__ __forceinline__ __half2 scales_f16(uint16_t packed) {
  return __half2(__nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(packed), __NV_E4M3));
}

// e4m3 -> f32, exact (the fp8 core's conversion), for the two scales a
// chunk spans, pre-multiplied by 2^14 (exact) — the correction of the
// register decode below, which the tile kernels' decode still uses.
__device__ __forceinline__ float2 scales_x16384(uint16_t packed) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(packed), __NV_E4M3);
  float2 s = __half22float2(__half2(h));
  s.x *= 16384.f;
  s.y *= 16384.f;
  return s;
}

// One e2m1 code -> its value times 2^-14, as fp32, exactly.
__device__ __forceinline__ float e2m1_scaled(uint32_t code) {
  const uint32_t bits = ((code & 7u) << 9) | ((code & 8u) << 12);
  return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
}

// One e8m0 scale byte -> 2^(byte - 127) times 2^14 as fp32, exactly (the
// register decode's correction folded in): 2^(byte - 113) for bytes 0..240
// (byte 0 = 2^-113, a normal fp32), +inf for 241..254, NaN for 255.
__device__ __forceinline__ float e8m0_scale_x16384(uint32_t byte) {
  if (byte == 255u) return __int_as_float(0x7FC00000);
  const uint32_t e = byte + 14u;
  return e > 254u ? __int_as_float(0x7F800000) : __uint_as_float(e << 23);
}

// The MXFP4 chunk: 32 codes under ONE scale (s14 = the chunk's e8m0 scale
// times 2^14), each code decoded to fp32 by bit placement and multiplied
// once — exact — then the same FMA chain as consume_chunk's.
template <int kRows>
__device__ __forceinline__ void consume_chunk_mx_exact(const uint4& wchunk, float s14,
                                                       const uint4 (&xv)[kRows][4],
                                                       float (&acc)[kRows]) {
  const uint32_t words[4] = {wchunk.x, wchunk.y, wchunk.z, wchunk.w};
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint32_t word = words[q];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t byte = (word >> (8 * j)) & 0xFFu;
      const float w0 = e2m1_scaled(byte & 0xFu) * s14;
      const float w1 = e2m1_scaled(byte >> 4) * s14;
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        const uint32_t xw = j == 0 ? xv[r][q].x : j == 1 ? xv[r][q].y
                          : j == 2 ? xv[r][q].z : xv[r][q].w;
        const float x0 = bf16_bits_to_float(static_cast<uint16_t>(xw & 0xFFFFu));
        const float x1 = bf16_bits_to_float(static_cast<uint16_t>(xw >> 16));
        acc[r] = __fmaf_rn(w0, x0, acc[r]);
        acc[r] = __fmaf_rn(w1, x1, acc[r]);
      }
    }
  }
}

// The MXFP4 chunk through the f16 path when its scale allows (2026-09-22):
// the hardware e2m1x2 -> f16x2 conversion and ONE f16x2 multiply by the
// scale, as consume_chunk does for NVFP4. An e2m1 value c (0, ±0.5 .. ±6)
// times 2^(e - 127) is exact in f16 for 104 <= e <= 140: within that
// range the product is either an f16 normal (c has three significant
// bits) or a subnormal multiple of 2^-24 (2c is an integer, e - 128 >=
// -24), and the scale itself is an f16 power of two; so the fp32 values
// the FMA chain consumes are the exact-path's, bitwise. Every scale a
// weight tensor carries sits in that range (|w| <= 6 x 2^(e - 127) puts
// e near 117..124); the exact path takes any other (the codec's NaN and
// saturation semantics included). The per-byte cost drops from eight
// instructions to four.
template <int kRows>
__device__ __forceinline__ void consume_chunk_mx(const uint4& wchunk, uint32_t scale_byte,
                                                 const uint4 (&xv)[kRows][4],
                                                 float (&acc)[kRows]) {
  if (scale_byte < 104u || scale_byte > 140u) {
    consume_chunk_mx_exact<kRows>(wchunk, e8m0_scale_x16384(scale_byte), xv, acc);
    return;
  }
  // 2^(e - 127) as f16 bits: exponent field e - 127 + 15 (positive for
  // e >= 113); below that a subnormal 2^(e - 127) = 2^(e - 103) x 2^-24
  // (mantissa bit e - 103: e = 104 is 2 x 2^-24, e = 112 is 2^9 x 2^-24).
  const uint16_t sbits = scale_byte >= 113u ? static_cast<uint16_t>((scale_byte - 112u) << 10)
                                            : static_cast<uint16_t>(1u << (scale_byte - 103u));
  const __half2 s2 = __half2half2(__ushort_as_half(sbits));
  const uint32_t words[4] = {wchunk.x, wchunk.y, wchunk.z, wchunk.w};
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint32_t word = words[q];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t byte = (word >> (8 * j)) & 0xFFu;
      const __half2 pair(__nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(byte), __NV_E2M1));
      const __half2 p = __hmul2(pair, s2);  // {code lo, code hi} x s, exact
      const float w0 = __low2float(p);
      const float w1 = __high2float(p);
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        const uint32_t xw = j == 0 ? xv[r][q].x : j == 1 ? xv[r][q].y
                          : j == 2 ? xv[r][q].z : xv[r][q].w;
        const float x0 = bf16_bits_to_float(static_cast<uint16_t>(xw & 0xFFFFu));
        const float x1 = bf16_bits_to_float(static_cast<uint16_t>(xw >> 16));
        acc[r] = __fmaf_rn(w0, x0, acc[r]);
        acc[r] = __fmaf_rn(w1, x1, acc[r]);
      }
    }
  }
}

// One 16-byte chunk (codes [e0, e0+32) of a row, scales s01 = {s(e0), s(e0+16)}
// times 2^14) into kRows accumulators. Element order within the chunk is
// the storage order: byte j of word q holds elements e0+8q+2j (low nibble)
// and e0+8q+2j+1.
// The 32 activation elements a chunk multiplies, per staged row: four
// uint4 out of shared memory — loaded once per chunk column and reused
// across the warp's row steps (2026-09-08: hardware counters put the
// fp4 core's stalls on the shared-memory pipe — two codes per weight byte
// is twice the activation traffic per byte of the fp8 core's, and the
// per-step re-read doubled it again).
template <int kRows>
__device__ __forceinline__ void load_window(const uint16_t* __restrict__ sx,
                                            int k, int e0,
                                            uint4 (&xv)[kRows][4]) {
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const uint4* xp = reinterpret_cast<const uint4*>(
        sx + static_cast<size_t>(r) * k + e0);
#pragma unroll
    for (int q = 0; q < 4; ++q) xv[r][q] = xp[q];
  }
}

// The decode of a byte (two codes): the hardware e2m1x2 -> f16x2
// conversion (one F2FP on sm_121a; cuda_fp4.h's software form elsewhere),
// then one f16x2 multiply by the group's scale. The product is exact in
// f16 — e2m1 x e4m3 carries at most six significant bits, and every
// nonzero product lies in [2^-10, 2688], all f16 normals — so its f32 is
// the same value the f32 decode produced: the FMA chain is bitwise the
// one before it (2026-09-08; counters had the fp4 slot kernels' compute
// phase, ~8 instructions a code, un-overlapped with their loads).
template <int kRows>
__device__ __forceinline__ void consume_chunk(const uint4& wchunk, __half2 s01,
                                              const uint4 (&xv)[kRows][4],
                                              float (&acc)[kRows]) {
  const uint32_t words[4] = {wchunk.x, wchunk.y, wchunk.z, wchunk.w};
  const __half2 s00 = __half2half2(__low2half(s01));   // codes 0-15
  const __half2 s11 = __half2half2(__high2half(s01));  // codes 16-31
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint32_t word = words[q];
    const __half2 sq = q < 2 ? s00 : s11;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t byte = (word >> (8 * j)) & 0xFFu;
      const __half2 pair(__nv_cvt_fp4x2_to_halfraw2(
          static_cast<__nv_fp4x2_storage_t>(byte), __NV_E2M1));
      const __half2 p = __hmul2(pair, sq);  // {code lo, code hi} x s, exact
      const float w0 = __low2float(p);
      const float w1 = __high2float(p);
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        const uint32_t xw = j == 0 ? xv[r][q].x : j == 1 ? xv[r][q].y
                          : j == 2 ? xv[r][q].z : xv[r][q].w;
        const float x0 = bf16_bits_to_float(static_cast<uint16_t>(xw & 0xFFFFu));
        const float x1 = bf16_bits_to_float(static_cast<uint16_t>(xw >> 16));
        acc[r] = __fmaf_rn(w0, x0, acc[r]);
        acc[r] = __fmaf_rn(w1, x1, acc[r]);
      }
    }
  }
}

// Reduce kRows accumulators across a lane group of Lanes lanes (a power of
// two, aligned): the fixed xor tree, launch-shape independent.
template <int kRows, int Lanes>
__device__ __forceinline__ void group_reduce(float (&acc)[kRows]) {
#pragma unroll
  for (int r = 0; r < kRows; ++r)
#pragma unroll
    for (int off = Lanes >> 1; off > 0; off >>= 1)
      acc[r] += __shfl_xor_sync(0xFFFFFFFFu, acc[r], off);
}

// The output policy of one dot (the fp8 core's): bf16 or raw fp32.
__device__ __forceinline__ void store_dot(uint16_t* out, float v) {
  *out = float_to_bf16_bits(v);
}
__device__ __forceinline__ void store_dot(float* out, float v) { *out = v; }

// One pass of a warp's row dots: chunks [C0, C0 + NC) of each lane's row
// (rows [n0 + warp*rows_per_warp, +rows_per_warp) of an [n, K] matrix)
// against kRows staged activation rows, accumulated into acc. Every load
// of the pass is issued before any is consumed; the chunks are then
// consumed column by column (the activation window once, every row step
// against it), each row's chunks in k order into its own accumulator.
// No warp-uniform early return: callers may barrier after this.
// The chunk's scale word: NVFP4 the two e4m3 bytes at scale_cols = K/16
// (offset boff / 8), MXFP4 the one e8m0 byte at K/32 (boff / 16).
template <int K, int kGroupT>
__device__ __forceinline__ uint16_t load_chunk_scale(const uint8_t* __restrict__ scales,
                                                     int row, int boff) {
  if constexpr (kGroupT == kGroup) {
    return *reinterpret_cast<const uint16_t*>(
        scales + static_cast<size_t>(row) * scale_cols_of(K, kGroup) + boff / 8);
  } else {
    return scales[static_cast<size_t>(row) * scale_cols_of(K, kMxGroup) + boff / 16];
  }
}
template <int kRows, int kGroupT>
__device__ __forceinline__ void consume_by_group(const uint4& wchunk, uint16_t sv,
                                                 const uint4 (&xv)[kRows][4],
                                                 float (&acc)[kRows]) {
  if constexpr (kGroupT == kGroup)
    consume_chunk<kRows>(wchunk, scales_f16(sv), xv, acc);
  else
    consume_chunk_mx<kRows>(wchunk, static_cast<uint32_t>(sv), xv, acc);
}

// One pass's loads, held in registers between their issue and their
// consumption (2026-09-22: the slot kernels issue the first pass BEFORE
// staging the activations, so the DRAM latency overlaps the staging and
// its barrier — a block's life is one pass at K <= 4096, and its prologue
// was a third of it with nothing in flight; the FMA chain is untouched).
template <int NC>
struct PassLoads {
  uint4 wv[kSteps * NC];
  uint16_t sv[kSteps * NC];
};

// The chunk load: `kStream` reads it as a streaming (evict-first) line —
// the decode slot kernels' weights are read once per step, and marking
// them so keeps the L2 for what the prefetcher put there (2026-09-22
// experiment; the grouped prefill kernels re-read rows across their row
// groups and keep the normal loads).
template <bool kStream>
__device__ __forceinline__ uint4 load_chunk(const uint4* p) {
  if constexpr (kStream) return __ldcs(p);
  else return *p;
}

template <int K, int C0, int NC, int kGroupT = kGroup, bool kStream = false>
__device__ __forceinline__ void issue_pass(const uint8_t* __restrict__ w,
                                           const uint8_t* __restrict__ scales, int n0, int n,
                                           PassLoads<NC>& L) {
  using G = Geom<K>;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int group = lane / G::lanes_per_row;
  const int lig = lane % G::lanes_per_row;
  const int row_base = n0 + warp * G::rows_per_warp;
#pragma unroll
  for (int st = 0; st < kSteps; ++st) {
    const int row = row_base + st * G::rows_per_step + group;
    const bool ok = row < n;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      const int boff = ((C0 + c) * G::lanes_per_row + lig) * kChunkBytes;
      const int i = st * NC + c;
      L.wv[i] = ok ? load_chunk<kStream>(reinterpret_cast<const uint4*>(
                         w + static_cast<size_t>(row) * G::row_bytes + boff))
                   : make_uint4(0u, 0u, 0u, 0u);
      L.sv[i] = ok ? load_chunk_scale<K, kGroupT>(scales, row, boff) : static_cast<uint16_t>(0);
    }
  }
}

template <int K, int kRows, int C0, int NC, int kGroupT = kGroup>
__device__ __forceinline__ void consume_pass(const PassLoads<NC>& L,
                                             const uint16_t* __restrict__ sx, int n0, int n,
                                             float (&acc)[kSteps][kRows]) {
  using G = Geom<K>;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int group = lane / G::lanes_per_row;
  const int lig = lane % G::lanes_per_row;
  const int row_base = n0 + warp * G::rows_per_warp;
#pragma unroll
  for (int c = 0; c < NC; ++c) {
    const int e0 = ((C0 + c) * G::lanes_per_row + lig) * kCodesPerChunk;
    uint4 xv[kRows][4];
    load_window<kRows>(sx, K, e0, xv);
#pragma unroll
    for (int st = 0; st < kSteps; ++st) {
      const int row = row_base + st * G::rows_per_step + group;
      if (row >= n) continue;  // per lane; no barrier inside
      const int i = st * NC + c;
      consume_by_group<kRows, kGroupT>(L.wv[i], L.sv[i], xv, acc[st]);
    }
  }
}

template <int K, int kRows, int C0, int NC, int kGroupT = kGroup>
__device__ __forceinline__ void pass_row_dots(const uint8_t* __restrict__ w,
                                              const uint8_t* __restrict__ scales,
                                              const uint16_t* __restrict__ sx,
                                              int n0, int n,
                                              float (&acc)[kSteps][kRows]) {
  PassLoads<NC> L;
  issue_pass<K, C0, NC, kGroupT>(w, scales, n0, n, L);
  consume_pass<K, kRows, C0, NC, kGroupT>(L, sx, n0, n, acc);
}

// The dots of a warp's rows_per_warp rows against kRows staged activation
// rows, UNDIVIDED by the global scale and reduced across each row's lane
// group. acc[st][a] is this lane's row of step st (row_base + st *
// rows_per_step + group); lane `lig == 0` of the group holds the reduced
// value. Each row's chunks are consumed in k order into its own
// accumulator across the passes, so a row's result is the same whatever
// kRows, the pass split or the launcher.
template <int K, int kRows, int kGroupT = kGroup>
__device__ __forceinline__ void warp_row_dots(const uint8_t* __restrict__ w,
                                              const uint8_t* __restrict__ scales,
                                              const uint16_t* __restrict__ sx,
                                              int n0, int n,
                                              float (&acc)[kSteps][kRows]) {
  using G = Geom<K>;
#pragma unroll
  for (int st = 0; st < kSteps; ++st)
#pragma unroll
    for (int a = 0; a < kRows; ++a) acc[st][a] = 0.f;
  pass_row_dots<K, kRows, 0, G::pass_chunks(0), kGroupT>(w, scales, sx, n0, n, acc);
  if constexpr (G::passes > 1)
    pass_row_dots<K, kRows, kMaxChunksPerLane, G::pass_chunks(1), kGroupT>(w, scales, sx, n0, n, acc);
  if constexpr (G::passes > 2)
    pass_row_dots<K, kRows, 2 * kMaxChunksPerLane, G::pass_chunks(2), kGroupT>(w, scales, sx, n0, n, acc);
  if constexpr (G::passes > 3)
    pass_row_dots<K, kRows, 3 * kMaxChunksPerLane, G::pass_chunks(3), kGroupT>(w, scales, sx, n0, n, acc);
#pragma unroll
  for (int st = 0; st < kSteps; ++st) group_reduce<kRows, G::lanes_per_row>(acc[st]);
}

// One pass of two matrices over the same rows and activations (gate and
// up): every load of both issued before either is consumed — twice the
// bytes in flight per lane at the same block count. Each matrix's row
// chain is pass_row_dots's exactly (bitwise).
template <int K, int C0, int NC, int kGroupT = kGroup, bool kStream = false>
__device__ __forceinline__ void issue_pass_pair(
    const uint8_t* __restrict__ w0, const uint8_t* __restrict__ scales0,
    const uint8_t* __restrict__ w1, const uint8_t* __restrict__ scales1, int n0, int n,
    PassLoads<NC>& L0, PassLoads<NC>& L1) {
  using G = Geom<K>;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int group = lane / G::lanes_per_row;
  const int lig = lane % G::lanes_per_row;
  const int row_base = n0 + warp * G::rows_per_warp;
#pragma unroll
  for (int st = 0; st < kSteps; ++st) {
    const int row = row_base + st * G::rows_per_step + group;
    const bool ok = row < n;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      const int boff = ((C0 + c) * G::lanes_per_row + lig) * kChunkBytes;
      const int i = st * NC + c;
      const size_t wo = static_cast<size_t>(row) * G::row_bytes + boff;
      L0.wv[i] = ok ? load_chunk<kStream>(reinterpret_cast<const uint4*>(w0 + wo)) : make_uint4(0u, 0u, 0u, 0u);
      L1.wv[i] = ok ? load_chunk<kStream>(reinterpret_cast<const uint4*>(w1 + wo)) : make_uint4(0u, 0u, 0u, 0u);
      L0.sv[i] = ok ? load_chunk_scale<K, kGroupT>(scales0, row, boff) : static_cast<uint16_t>(0);
      L1.sv[i] = ok ? load_chunk_scale<K, kGroupT>(scales1, row, boff) : static_cast<uint16_t>(0);
    }
  }
}

template <int K, int kRows, int C0, int NC, int kGroupT = kGroup>
__device__ __forceinline__ void consume_pass_pair(const PassLoads<NC>& L0, const PassLoads<NC>& L1,
                                                  const uint16_t* __restrict__ sx, int n0, int n,
                                                  float (&acc0)[kSteps][kRows],
                                                  float (&acc1)[kSteps][kRows]) {
  using G = Geom<K>;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int group = lane / G::lanes_per_row;
  const int lig = lane % G::lanes_per_row;
  const int row_base = n0 + warp * G::rows_per_warp;
#pragma unroll
  for (int c = 0; c < NC; ++c) {
    const int e0 = ((C0 + c) * G::lanes_per_row + lig) * kCodesPerChunk;
    uint4 xv[kRows][4];
    load_window<kRows>(sx, K, e0, xv);
#pragma unroll
    for (int st = 0; st < kSteps; ++st) {
      const int row = row_base + st * G::rows_per_step + group;
      if (row >= n) continue;
      const int i = st * NC + c;
      consume_by_group<kRows, kGroupT>(L0.wv[i], L0.sv[i], xv, acc0[st]);
      consume_by_group<kRows, kGroupT>(L1.wv[i], L1.sv[i], xv, acc1[st]);
    }
  }
}

template <int K, int kRows, int C0, int NC, int kGroupT = kGroup>
__device__ __forceinline__ void pass_row_dots_pair(
    const uint8_t* __restrict__ w0, const uint8_t* __restrict__ scales0,
    const uint8_t* __restrict__ w1, const uint8_t* __restrict__ scales1,
    const uint16_t* __restrict__ sx, int n0, int n,
    float (&acc0)[kSteps][kRows], float (&acc1)[kSteps][kRows]) {
  PassLoads<NC> L0, L1;
  issue_pass_pair<K, C0, NC, kGroupT>(w0, scales0, w1, scales1, n0, n, L0, L1);
  consume_pass_pair<K, kRows, C0, NC, kGroupT>(L0, L1, sx, n0, n, acc0, acc1);
}

template <int K, int kRows, int kGroupT = kGroup>
__device__ __forceinline__ void warp_row_dots_pair(
    const uint8_t* __restrict__ w0, const uint8_t* __restrict__ scales0,
    const uint8_t* __restrict__ w1, const uint8_t* __restrict__ scales1,
    const uint16_t* __restrict__ sx, int n0, int n,
    float (&acc0)[kSteps][kRows], float (&acc1)[kSteps][kRows]) {
  using G = Geom<K>;
#pragma unroll
  for (int st = 0; st < kSteps; ++st)
#pragma unroll
    for (int a = 0; a < kRows; ++a) acc0[st][a] = acc1[st][a] = 0.f;
  pass_row_dots_pair<K, kRows, 0, G::pass_chunks(0), kGroupT>(w0, scales0, w1, scales1, sx, n0, n, acc0, acc1);
  if constexpr (G::passes > 1)
    pass_row_dots_pair<K, kRows, kMaxChunksPerLane, G::pass_chunks(1), kGroupT>(w0, scales0, w1, scales1, sx, n0, n, acc0, acc1);
  if constexpr (G::passes > 2)
    pass_row_dots_pair<K, kRows, 2 * kMaxChunksPerLane, G::pass_chunks(2), kGroupT>(w0, scales0, w1, scales1, sx, n0, n, acc0, acc1);
  if constexpr (G::passes > 3)
    pass_row_dots_pair<K, kRows, 3 * kMaxChunksPerLane, G::pass_chunks(3), kGroupT>(w0, scales0, w1, scales1, sx, n0, n, acc0, acc1);
#pragma unroll
  for (int st = 0; st < kSteps; ++st) {
    group_reduce<kRows, G::lanes_per_row>(acc0[st]);
    group_reduce<kRows, G::lanes_per_row>(acc1[st]);
  }
}

// The early-issue forms (the slot kernels): the caller issued pass 0's
// loads (issue_pass / issue_pass_pair with C0 = 0, NC = pass_chunks(0))
// before staging the activations; these consume them and run the
// remaining passes as warp_row_dots / warp_row_dots_pair do. Bitwise the
// same chains.
template <int K, int kRows, int C0, int NC, int kGroupT = kGroup, bool kStream = false>
__device__ __forceinline__ void pass_row_dots_s(const uint8_t* __restrict__ w,
                                                const uint8_t* __restrict__ scales,
                                                const uint16_t* __restrict__ sx, int n0, int n,
                                                float (&acc)[kSteps][kRows]) {
  PassLoads<NC> L;
  issue_pass<K, C0, NC, kGroupT, kStream>(w, scales, n0, n, L);
  consume_pass<K, kRows, C0, NC, kGroupT>(L, sx, n0, n, acc);
}
template <int K, int kRows, int C0, int NC, int kGroupT = kGroup, bool kStream = false>
__device__ __forceinline__ void pass_row_dots_pair_s(
    const uint8_t* __restrict__ w0, const uint8_t* __restrict__ scales0,
    const uint8_t* __restrict__ w1, const uint8_t* __restrict__ scales1,
    const uint16_t* __restrict__ sx, int n0, int n,
    float (&acc0)[kSteps][kRows], float (&acc1)[kSteps][kRows]) {
  PassLoads<NC> L0, L1;
  issue_pass_pair<K, C0, NC, kGroupT, kStream>(w0, scales0, w1, scales1, n0, n, L0, L1);
  consume_pass_pair<K, kRows, C0, NC, kGroupT>(L0, L1, sx, n0, n, acc0, acc1);
}

template <int K, int kRows, int kGroupT = kGroup, bool kStream = false>
__device__ __forceinline__ void warp_row_dots_issued(
    const PassLoads<Geom<K>::pass_chunks(0)>& L0, const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ scales, const uint16_t* __restrict__ sx, int n0, int n,
    float (&acc)[kSteps][kRows]) {
  using G = Geom<K>;
#pragma unroll
  for (int st = 0; st < kSteps; ++st)
#pragma unroll
    for (int a = 0; a < kRows; ++a) acc[st][a] = 0.f;
  consume_pass<K, kRows, 0, G::pass_chunks(0), kGroupT>(L0, sx, n0, n, acc);
  if constexpr (G::passes > 1)
    pass_row_dots_s<K, kRows, kMaxChunksPerLane, G::pass_chunks(1), kGroupT, kStream>(w, scales, sx, n0, n, acc);
  if constexpr (G::passes > 2)
    pass_row_dots_s<K, kRows, 2 * kMaxChunksPerLane, G::pass_chunks(2), kGroupT, kStream>(w, scales, sx, n0, n, acc);
  if constexpr (G::passes > 3)
    pass_row_dots_s<K, kRows, 3 * kMaxChunksPerLane, G::pass_chunks(3), kGroupT, kStream>(w, scales, sx, n0, n, acc);
#pragma unroll
  for (int st = 0; st < kSteps; ++st) group_reduce<kRows, G::lanes_per_row>(acc[st]);
}

template <int K, int kRows, int kGroupT = kGroup, bool kStream = false>
__device__ __forceinline__ void warp_row_dots_pair_issued(
    const PassLoads<Geom<K>::pass_chunks(0)>& L0, const PassLoads<Geom<K>::pass_chunks(0)>& L1,
    const uint8_t* __restrict__ w0, const uint8_t* __restrict__ scales0,
    const uint8_t* __restrict__ w1, const uint8_t* __restrict__ scales1,
    const uint16_t* __restrict__ sx, int n0, int n,
    float (&acc0)[kSteps][kRows], float (&acc1)[kSteps][kRows]) {
  using G = Geom<K>;
#pragma unroll
  for (int st = 0; st < kSteps; ++st)
#pragma unroll
    for (int a = 0; a < kRows; ++a) acc0[st][a] = acc1[st][a] = 0.f;
  consume_pass_pair<K, kRows, 0, G::pass_chunks(0), kGroupT>(L0, L1, sx, n0, n, acc0, acc1);
  if constexpr (G::passes > 1)
    pass_row_dots_pair_s<K, kRows, kMaxChunksPerLane, G::pass_chunks(1), kGroupT, kStream>(w0, scales0, w1, scales1, sx, n0, n, acc0, acc1);
  if constexpr (G::passes > 2)
    pass_row_dots_pair_s<K, kRows, 2 * kMaxChunksPerLane, G::pass_chunks(2), kGroupT, kStream>(w0, scales0, w1, scales1, sx, n0, n, acc0, acc1);
  if constexpr (G::passes > 3)
    pass_row_dots_pair_s<K, kRows, 3 * kMaxChunksPerLane, G::pass_chunks(3), kGroupT, kStream>(w0, scales0, w1, scales1, sx, n0, n, acc0, acc1);
#pragma unroll
  for (int st = 0; st < kSteps; ++st) {
    group_reduce<kRows, G::lanes_per_row>(acc0[st]);
    group_reduce<kRows, G::lanes_per_row>(acc1[st]);
  }
}

// Which lane owns the reduced dot of the warp's step-st row, and that
// row's index: lane `lig == 0` of its group.
template <int K>
__device__ __forceinline__ int owned_row(int n0, int st, bool& mine) {
  using G = Geom<K>;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  mine = (lane % G::lanes_per_row) == 0;
  return n0 + warp * G::rows_per_warp + st * G::rows_per_step + lane / G::lanes_per_row;
}

// The block body: kWarps warps x rows_per_warp rows each of an [n, K]
// NVFP4 matrix (payload [n, K/2], scales [n, K/16], global scale g):
//   out[a * out_stride + row] = store_dot(dot(row, sx[a]) / g)
// or, at kGroupT = kMxGroup, of an MXFP4 matrix (scales [n, K/32], g
// unused): store_dot(dot(row, sx[a])).
template <int K, int kRows, typename OutT, int kGroupT = kGroup>
__device__ __forceinline__ void block_rows(const uint8_t* __restrict__ w,
                                           const uint8_t* __restrict__ scales,
                                           float g,
                                           const uint16_t* __restrict__ sx,
                                           int n0, int n,
                                           OutT* __restrict__ out,
                                           size_t out_stride) {
  float acc[kSteps][kRows];
  warp_row_dots<K, kRows, kGroupT>(w, scales, sx, n0, n, acc);
#pragma unroll
  for (int st = 0; st < kSteps; ++st) {
    bool mine = false;
    const int row = owned_row<K>(n0, st, mine);
    if (mine && row < n) {
#pragma unroll
      for (int a = 0; a < kRows; ++a) {
        if constexpr (kGroupT == kGroup)
          store_dot(out + static_cast<size_t>(a) * out_stride + row, __fdiv_rn(acc[st][a], g));
        else
          store_dot(out + static_cast<size_t>(a) * out_stride + row, acc[st][a]);
      }
    }
  }
}

// Dispatch over the compiled K set: f(std::integral_constant<int, K>{}).
// The power-of-two set (GLM-5.3-Flash's 4096 / 512 and the fixtures) and
// GLM-4.7's widths: 5120 (hidden: gate/up, dense gate/up), 384 / 768 /
// 1536 (the expert down at worlds 4 / 2 / 1), 3072 / 6144 / 12288 (the
// dense down at worlds 4 / 2 / 1). Qwen3.8-Flash-Next's NVFP4 experts
//: 2560 (hidden: gate/up), 640 / 320 / 160 (the expert down
// at worlds 1 / 2 / 4).
template <typename F>
__host__ inline void dispatch_k(int k, F&& f) {
  switch (k) {
    case 32: f(std::integral_constant<int, 32>{}); return;
    case 64: f(std::integral_constant<int, 64>{}); return;
    case 128: f(std::integral_constant<int, 128>{}); return;
    case 160: f(std::integral_constant<int, 160>{}); return;
    case 256: f(std::integral_constant<int, 256>{}); return;
    case 320: f(std::integral_constant<int, 320>{}); return;
    case 384: f(std::integral_constant<int, 384>{}); return;
    case 512: f(std::integral_constant<int, 512>{}); return;
    case 640: f(std::integral_constant<int, 640>{}); return;
    case 768: f(std::integral_constant<int, 768>{}); return;
    case 1024: f(std::integral_constant<int, 1024>{}); return;
    case 1536: f(std::integral_constant<int, 1536>{}); return;
    case 2048: f(std::integral_constant<int, 2048>{}); return;
    case 2560: f(std::integral_constant<int, 2560>{}); return;
    case 3072: f(std::integral_constant<int, 3072>{}); return;
    case 4096: f(std::integral_constant<int, 4096>{}); return;
    case 5120: f(std::integral_constant<int, 5120>{}); return;
    case 6144: f(std::integral_constant<int, 6144>{}); return;
    case 12288: f(std::integral_constant<int, 12288>{}); return;
    default:
      throw std::invalid_argument(
          "fp4_gemv: K is not in the compiled set (32..4096 powers of two, "
          "160, 320, 384, 640, 768, 1536, 2560, 3072, 5120, 6144, 12288)");
  }
}
// The MXFP4 compiled set (2026-09-13): DeepSeek-V4.1-Flash's widths —
// 5120 (hidden: w1/w3) and the expert down at worlds 4 / 2 / 1 (576 /
// 1152 / 2304: 18 / 36 / 72 chunks — 2 x 9, 4 x 9 and 8 x 9 in passes of
// four), plus the power-of-two test geometries; MiMo-V2.6-Flash's
// (2026-09-22) — 4096 (hidden: gate/up) and the expert down at worlds 4 /
// 2 / 1 (512 / 1024 / 2048). Its own switch keeps the NVFP4 instantiation
// set as it is.
template <typename F>
__host__ inline void dispatch_k_mx(int k, F&& f) {
  switch (k) {
    case 32: f(std::integral_constant<int, 32>{}); return;
    case 64: f(std::integral_constant<int, 64>{}); return;
    case 128: f(std::integral_constant<int, 128>{}); return;
    case 256: f(std::integral_constant<int, 256>{}); return;
    case 512: f(std::integral_constant<int, 512>{}); return;
    case 576: f(std::integral_constant<int, 576>{}); return;
    case 1024: f(std::integral_constant<int, 1024>{}); return;
    case 1152: f(std::integral_constant<int, 1152>{}); return;
    case 2048: f(std::integral_constant<int, 2048>{}); return;
    case 2304: f(std::integral_constant<int, 2304>{}); return;
    case 4096: f(std::integral_constant<int, 4096>{}); return;
    case 5120: f(std::integral_constant<int, 5120>{}); return;
    default:
      throw std::invalid_argument(
          "fp4_gemv: K is not in the MXFP4 compiled set (32, 64, 128, 256, 512, "
          "576, 1024, 1152, 2048, 2304, 4096, 5120)");
  }
}
__host__ __device__ constexpr bool k_compiled_mx(int k) {
  return k == 32 || k == 64 || k == 128 || k == 256 || k == 512 || k == 576 || k == 1024 ||
         k == 1152 || k == 2048 || k == 2304 || k == 4096 || k == 5120;
}
// True when dispatch_k compiles a kernel for k.
__host__ __device__ constexpr bool k_compiled(int k) {
  return k == 32 || k == 64 || k == 128 || k == 160 || k == 256 || k == 320 || k == 384 ||
         k == 512 || k == 640 || k == 768 || k == 1024 || k == 1536 || k == 2048 || k == 2560 ||
         k == 3072 || k == 4096 || k == 5120 || k == 6144 || k == 12288;
}
// The compiled check for either group.
__host__ __device__ constexpr bool k_compiled_for(int k, int group) {
  return group == kMxGroup ? k_compiled_mx(k) : k_compiled(k);
}

}  // namespace fp4_gemv
}  // namespace dgpp
