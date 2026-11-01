#pragma once
// The latent cache's tile loaders and the flash attention block body, shared
// by the production kernels (kernels/dsa.cu) and the step engine's attention
// unit (kernels/step/stream.cu): one body, called with a block's coordinates,
// so the unit's partials are the kernel's bitwise. Everything here was
// kernels/dsa.cu's until 2026-11-01.
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "kernels/latent_format.hpp"

namespace dgpp {
namespace dsa_flash {

using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;
using dgpp::float_to_fp8_e4m3_bits;
using dgpp::fp8_e4m3_bits_to_float;

// ---- native conversion helpers ----------------------------------------
// The GB10 has hardware e4m3->f16 and bf16->f32 conversions; the software
// decoders in dtypes.hpp are branchy (NaN/denormal special cases) and were
// ~90% of the select kernel's cycles. Both paths are EXACT — e4m3 fits
// entirely inside f16, bf16 inside f32 — so these are bit-identical
// replacements, verified by the bitwise select fuzz.

__device__ inline float2 fp8x2_to_float2(uint16_t v) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(v, __NV_E4M3);
  return __half22float2(__half2(h));
}

__device__ inline float2 bf16x2_to_float2(uint32_t v) {
  __nv_bfloat16_raw lo, hi;
  lo.x = static_cast<unsigned short>(v & 0xFFFFu);
  hi.x = static_cast<unsigned short>(v >> 16);
  return __bfloat1622float2(
      __nv_bfloat162(__nv_bfloat16(lo), __nv_bfloat16(hi)));
}

// acc += sum(element-wise products of 8 bf16 pairs), accumulated in element
// order (same sequence as the scalar loop it replaces — FFMA-friendly; the
// attention path is tolerance-pinned, not bitwise).
__device__ inline float dot8_bf16(uint4 a, uint4 b, float acc) {
  const uint32_t* a32 = reinterpret_cast<const uint32_t*>(&a);
  const uint32_t* b32 = reinterpret_cast<const uint32_t*>(&b);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 x = bf16x2_to_float2(a32[j]);
    const float2 y = bf16x2_to_float2(b32[j]);
    acc += x.x * y.x;
    acc += x.y * y.y;
  }
  return acc;
}

// ---- the latent cache's formats (kernels/latent_format.hpp) ------------
// The attention kernels gather a latent tile into bf16 shared memory; a
// quantized cache dequantizes on the way in (the codes and the row scale
// -> the bf16 the host oracle sees, bitwise) and the math past the load
// is the bf16 kernel's. load8 fills 8 consecutive elements at element
// offset `e` (a multiple of 8) of physical row `phys`; load1 one element.
// A row is `row_bytes` long: the format's payload of kv_lora elements
// (latent_row_bytes) then the bf16 rope tail (plan D3) — an element
// offset at or past kv_lora reads the tail, in every format.
__device__ __forceinline__ uint32_t pack_bf16x2(uint16_t lo, uint16_t hi) {
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

template <LatentFormat F>
struct LatentTile;

template <>
struct LatentTile<LatentFormat::kBf16> {
  // The payload and the tail are one bf16 row: no branch.
  static __device__ __forceinline__ uint4 load8(const uint8_t* cache,
                                                const float*, int64_t phys,
                                                size_t row_bytes, int, int e) {
    return *reinterpret_cast<const uint4*>(cache + phys * int64_t(row_bytes) + e * 2);
  }
  static __device__ __forceinline__ uint16_t load1(const uint8_t* cache,
                                                   const float*, int64_t phys,
                                                   size_t row_bytes, int, int e) {
    return *reinterpret_cast<const uint16_t*>(cache + phys * int64_t(row_bytes) + e * 2);
  }
};

template <>
struct LatentTile<LatentFormat::kFp8> {
  // The hardware e4m3 -> f16 conversion is exact (the select kernel's pin);
  // one fp32 multiply by the row scale and a bf16 rounding follow, exactly
  // latent_fp8_decode_bf16.
  static __device__ __forceinline__ uint32_t decode2(uint16_t codes, float s) {
    const float2 v = fp8x2_to_float2(codes);
    return pack_bf16x2(float_to_bf16_bits(v.x * s), float_to_bf16_bits(v.y * s));
  }
  static __device__ __forceinline__ uint4 load8(const uint8_t* cache,
                                                const float* scales,
                                                int64_t phys, size_t row_bytes,
                                                int kv_lora, int e) {
    const uint8_t* row = cache + phys * int64_t(row_bytes);
    if (e >= kv_lora)
      return *reinterpret_cast<const uint4*>(row + kv_lora + (e - kv_lora) * 2);
    const uint2 raw = *reinterpret_cast<const uint2*>(row + e);
    const float s = scales[phys];
    uint4 out;
    out.x = decode2(static_cast<uint16_t>(raw.x & 0xFFFFu), s);
    out.y = decode2(static_cast<uint16_t>(raw.x >> 16), s);
    out.z = decode2(static_cast<uint16_t>(raw.y & 0xFFFFu), s);
    out.w = decode2(static_cast<uint16_t>(raw.y >> 16), s);
    return out;
  }
  static __device__ __forceinline__ uint16_t load1(const uint8_t* cache,
                                                   const float* scales,
                                                   int64_t phys, size_t row_bytes,
                                                   int kv_lora, int e) {
    const uint8_t* row = cache + phys * int64_t(row_bytes);
    if (e >= kv_lora)
      return *reinterpret_cast<const uint16_t*>(row + kv_lora + (e - kv_lora) * 2);
    return latent_fp8_decode_bf16(row[e], scales[phys]);
  }
};

template <>
struct LatentTile<LatentFormat::kFp4> {
  // Both e2m1 values and finite e4m3 block scales fit exactly in fp16.
  // Decode pairs in hardware, preserving the codec's two fp32 multiplies
  // and final bf16 rounding. Scalar software conversion inside every
  // gathered tile was a substantial full-GLM prefill cost.
  static __device__ __forceinline__ uint32_t decode2(uint8_t codes, float s) {
    const __half2_raw h = __nv_cvt_fp4x2_to_halfraw2(codes, __NV_E2M1);
    const float2 v = __half22float2(__half2(h));
    return pack_bf16x2(float_to_bf16_bits(v.x * s), float_to_bf16_bits(v.y * s));
  }
  static __device__ __forceinline__ float block_scale(uint8_t code, float s) {
    return fp8x2_to_float2(code).x * s;
  }
  static __device__ __forceinline__ uint4 load8(const uint8_t* cache,
                                                const float* scales,
                                                int64_t phys, size_t row_bytes,
                                                int kv_lora, int e) {
    const uint8_t* row = cache + phys * int64_t(row_bytes);
    if (e >= kv_lora)
      return *reinterpret_cast<const uint4*>(
          row + latent_row_bytes(LatentFormat::kFp4, kv_lora) + (e - kv_lora) * 2);
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(row + (e >> 1));
    const float S = block_scale(
        row[latent_fp4_scale_offset(kv_lora) + (e / kLatentFp4Block)], scales[phys]);
    uint4 out;
    out.x = decode2(static_cast<uint8_t>(raw), S);
    out.y = decode2(static_cast<uint8_t>(raw >> 8), S);
    out.z = decode2(static_cast<uint8_t>(raw >> 16), S);
    out.w = decode2(static_cast<uint8_t>(raw >> 24), S);
    return out;
  }
  static __device__ __forceinline__ uint16_t load1(const uint8_t* cache,
                                                   const float* scales,
                                                   int64_t phys, size_t row_bytes,
                                                   int kv_lora, int e) {
    const uint8_t* row = cache + phys * int64_t(row_bytes);
    if (e >= kv_lora)
      return *reinterpret_cast<const uint16_t*>(
          row + latent_row_bytes(LatentFormat::kFp4, kv_lora) + (e - kv_lora) * 2);
    const float S = block_scale(
        row[latent_fp4_scale_offset(kv_lora) + (e / kLatentFp4Block)], scales[phys]);
    const uint8_t byte = row[e >> 1];
    const uint32_t pair = decode2(byte, S);
    return static_cast<uint16_t>((e & 1) ? (pair >> 16) : pair);
  }
};

// fp8_block (2026-09-13, DeepSeek-V4.1's window rows): e4m3 codes with an
// e8m0 scale per 32 inside the row, no row scale — the decode is one
// exact fp32 multiply by a power of two and a bf16 rounding that changes
// nothing (latent_fp8_block_decode_bf16).
template <>
struct LatentTile<LatentFormat::kFp8Block> {
  static __device__ __forceinline__ uint32_t decode2(uint16_t codes, float s) {
    const float2 v = fp8x2_to_float2(codes);
    return pack_bf16x2(float_to_bf16_bits(v.x * s), float_to_bf16_bits(v.y * s));
  }
  static __device__ __forceinline__ uint4 load8(const uint8_t* cache,
                                                const float*, int64_t phys,
                                                size_t row_bytes, int kv_lora, int e) {
    const uint8_t* row = cache + phys * int64_t(row_bytes);
    if (e >= kv_lora)
      return *reinterpret_cast<const uint4*>(
          row + latent_row_bytes(LatentFormat::kFp8Block, kv_lora) + (e - kv_lora) * 2);
    const uint2 raw = *reinterpret_cast<const uint2*>(row + e);
    const float s = e8m0_byte_to_float(
        row[latent_fp8_block_scale_offset(kv_lora) + (e / kLatentFp8BlockGroup)]);
    uint4 out;
    out.x = decode2(static_cast<uint16_t>(raw.x & 0xFFFFu), s);
    out.y = decode2(static_cast<uint16_t>(raw.x >> 16), s);
    out.z = decode2(static_cast<uint16_t>(raw.y & 0xFFFFu), s);
    out.w = decode2(static_cast<uint16_t>(raw.y >> 16), s);
    return out;
  }
  static __device__ __forceinline__ uint16_t load1(const uint8_t* cache,
                                                   const float*, int64_t phys,
                                                   size_t row_bytes, int kv_lora, int e) {
    const uint8_t* row = cache + phys * int64_t(row_bytes);
    if (e >= kv_lora)
      return *reinterpret_cast<const uint16_t*>(
          row + latent_row_bytes(LatentFormat::kFp8Block, kv_lora) + (e - kv_lora) * 2);
    const float s = e8m0_byte_to_float(
        row[latent_fp8_block_scale_offset(kv_lora) + (e / kLatentFp8BlockGroup)]);
    return latent_fp8_block_decode_bf16(row[e], s);
  }
};

// fp4_block (DeepSeek-V4.1's compressed main KV): e2m1 codes with an
// ABSOLUTE e4m3 scale per 16 (no row scale); the product of two short
// mantissas is exact in fp32 and in bf16.
template <>
struct LatentTile<LatentFormat::kFp4Block> {
  static __device__ __forceinline__ uint4 load8(const uint8_t* cache,
                                                const float*, int64_t phys,
                                                size_t row_bytes, int kv_lora, int e) {
    const uint8_t* row = cache + phys * int64_t(row_bytes);
    if (e >= kv_lora)
      return *reinterpret_cast<const uint4*>(
          row + latent_row_bytes(LatentFormat::kFp4Block, kv_lora) + (e - kv_lora) * 2);
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(row + (e >> 1));
    const float S = latent_fp4_block_scale_abs(row[latent_fp4_scale_offset(kv_lora) + (e / kLatentFp4Block)]);
    uint16_t v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
      v[j] = latent_fp4_decode_bf16(static_cast<uint8_t>((raw >> (4 * j)) & 0xFu), S);
    uint4 out;
    out.x = pack_bf16x2(v[0], v[1]);
    out.y = pack_bf16x2(v[2], v[3]);
    out.z = pack_bf16x2(v[4], v[5]);
    out.w = pack_bf16x2(v[6], v[7]);
    return out;
  }
  static __device__ __forceinline__ uint16_t load1(const uint8_t* cache,
                                                   const float*, int64_t phys,
                                                   size_t row_bytes, int kv_lora, int e) {
    const uint8_t* row = cache + phys * int64_t(row_bytes);
    if (e >= kv_lora)
      return *reinterpret_cast<const uint16_t*>(
          row + latent_row_bytes(LatentFormat::kFp4Block, kv_lora) + (e - kv_lora) * 2);
    const float S = latent_fp4_block_scale_abs(row[latent_fp4_scale_offset(kv_lora) + (e / kLatentFp4Block)]);
    const uint8_t byte = row[e >> 1];
    return latent_fp4_decode_bf16(
        static_cast<uint8_t>((e & 1) ? (byte >> 4) : (byte & 0xFu)), S);
  }
};

// The cache row's stride: the format's payload plus the bf16 rope tail.
__host__ __device__ constexpr size_t latent_cache_row_bytes(LatentFormat f, int kv_lora,
                                                            int rope) {
  return latent_row_bytes(f, kv_lora) + static_cast<size_t>(rope) * 2;
}


// Dense causal attention on tensor cores (the prefill path, 2026-09-05)
// ---------------------------------------------------------------------
// Below index_topk tokens of context the selection is provably dense —
// every visible pool is selected and the tail appended, so a query at
// position p attends to tokens [0, p] — and the per-row gather kernel
// above spends ~130 scalar FMAs per token per thread re-reading the
// prefix once per row (a 2048-token prefill: 1,903 eight-row launches of
// 318 us per layer). This kernel is the dense case on mma.sync:
//   * the M dimension is (row, head) pairs — q_tilde is already the
//     [rows * local_heads, kv_lora] matrix — 32 M-rows per block as two
//     slabs of 16; the 8 warps are (slab, column quarter);
//   * per 32-token latent tile (gathered through the block table into
//     shared memory, shared by all 32 M-rows): S = Q~ . L^T over the
//     warp's k-quarter (16 x 32 per warp), the four quarters' partials
//     summed through shared memory in a FIXED order so every warp of a
//     slab holds identical S; then the FA2 online softmax on the C
//     fragments (row max by quad shuffles, l unrounded, probabilities
//     rounded to bf16 as the split kernel pins) and O += P . L over the
//     warp's 128 output columns, P re-used straight from the C fragments
//     as A fragments;
//   * causal masking per M-row against pos[row]; splits over the token
//     range exactly as the split kernel, publishing the same (m, l, c)
//     partials so dsa_attn_combine merges them unchanged.
// Numerics: bf16 inputs, fp32 accumulation, the same scale/exp/bf16-prob
// chain as the split kernel in a different summation order (the tensor
// core's) — tolerance-equal, not bitwise; deterministic (bitwise repeat).
namespace dense {
constexpr int kM = 32;             // (row, head) M-rows per block
constexpr int kTile = 32;          // latent tokens per tile
constexpr int kThreads = 256;      // 8 warps: 2 slabs x 4 column quarters
// KV: the value width (kv_lora); SD: the score width (kv_lora + rope, the
// cache row and the absorbed query — plan D3). Flash: SD == KV. NTOK: the
// tokens a slab's S tile spans (16 listed, kTile dense) — the exchange
// holds one warp's S partial in the C layout, 4 floats per n8 tile per
// lane. Shared memory holds the latent tile and the exchange; the Q~
// fragments live in registers (2026-10-18: 83 KB per block held one
// block per SM at the 512 widths, and the kernel was issue-latency-bound
// at two warps per scheduler; two blocks fit now).
template <int KV, int SD, int NTOK = kTile>
struct Geo {
  static_assert(SD >= KV && SD % 64 == 0 && KV % 32 == 0, "dsa flash geometry");
  static constexpr int SQ = SD + 8;        // padded smem row (u16 elements)
  static constexpr int CW = KV / 4;        // output columns per warp
  static constexpr int NT = CW / 8;        // n8 tiles per warp (PV)
  static constexpr int KW = SD / 4;        // k dims per warp (S)
  static constexpr int KS = KW / 16;       // k16 steps per warp (S)
  static constexpr int XF = NTOK * 16;     // one warp's S partial: NTOK/8 n8 tiles x 4 floats x 32 lanes
  static constexpr size_t smem_bytes = size_t(kTile) * SQ * 2 + size_t(8) * XF * 4;
};
}  // namespace dense

__device__ __forceinline__ void mma_bf16_16816(float (&c)[4], uint32_t a0,
                                               uint32_t a1, uint32_t a2,
                                               uint32_t a3, uint32_t b0,
                                               uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// kListed (2026-09-05, the sparse regime): instead of the causal range,
// each 16-row SLAB — one query row at local_heads >= 16 — walks its own
// selected-token list (topk[row], counts[row], the split kernel's inputs)
// into its own 16-token latent tile; the two slabs' loops run to the longer
// one's tile count. No union, no membership masks: the slab-is-a-row
// property makes the per-row selection the block's natural structure.
// The listed gather's anomaly record: the first out-of-range token or block
// (first writer wins) and a count; dsa_attn_anomalies() reads and clears it.
namespace {
__device__ unsigned long long g_attn_anomaly_count = 0;  // this translation unit's record (dsa_attn_anomalies reads dsa.cu's)
__device__ long long g_attn_anomaly[6] = {0, 0, 0, 0, 0, 0};
}  // namespace
__device__ __forceinline__ void attn_anomaly_record(int64_t tok, int32_t blk,
                                                    int qrow, int split,
                                                    int index, int live) {
  if (atomicAdd(&g_attn_anomaly_count, 1ull) == 0ull) {
    g_attn_anomaly[0] = tok;
    g_attn_anomaly[1] = blk;
    g_attn_anomaly[2] = qrow;
    g_attn_anomaly[3] = split;
    g_attn_anomaly[4] = index;
    g_attn_anomaly[5] = live;
    __threadfence();
  }
}

// Two blocks per SM where two of the geometry's shared tiles fit (the 256
// widths); one at 512, where the 83 KB tile already allows one — the
// register budget then holds the prefetched tile beside the accumulators.
template <int KV, int SD, bool kListed>
constexpr int attn_flash_min_blocks() {
  // The dynamic tile, the listed walk's static address round (below) and
  // the driver's 1 KB per block, twice, within the SM's 100 KB — for the
  // rope-less widths only. The 512 + 64 tail's kernel holds nine k16 steps
  // of Q~ fragments and nine fetches per tile and does not fit the
  // two-block register budget without spilling; compiled that way it
  // faulted DeepSeek-V4's mixed walk (shared_prefill_test, 2026-10-18)
  // while the same source at one block, or the two-block build of the
  // 512-wide kernels, ran clean — so that geometry keeps one block.
  const size_t block_smem = dense::Geo<KV, SD, kListed ? 16 : dense::kTile>::smem_bytes +
                            (kListed ? size_t(2) * 512 * 4 : size_t(8)) + 1024;
  return (kListed && SD == KV && 2 * block_smem <= size_t(100) * 1024) ? 2 : 1;  // the dense tile spills at 128 too
}

// kGroup (2026-10-23, the decode batch past a few rows): the kGroup
// consecutive splits of a row pair also publish ONE merged partial per
// group (the epilogue below: the last of the group's blocks to arrive
// merges the eight per-split partials from L2 with the finish's formula,
// writes the group's partial and discards the eight dead rows from L2 so
// they never write back) — the per-split partials of forty rows at
// thirty-two main and eight window splits were fifty megabytes a layer,
// written by this kernel and read back by the finish, 160 us a layer each
// way at the DRAM rate. A cluster form (distributed shared memory) was
// measured first: it wins only past forty rows, since clusters beyond the
// twelve co-resident ones schedule in coarse waves (+35 us a layer at ten
// to twenty rows). 1 = a partial per split (the prefill tile and the
// scalar form's layout).
// The block body of the flash kernel, callable with the block's coordinates
// (bx: the (row, head) block of 32 M-rows; s: the split) and its dynamic shared
// memory (attn_flash_smem_bytes): the production kernel (kernels/dsa.cu) and
// the step engine's attention unit (kernels/step/stream.cu) run the same body,
// so a unit's partials are the kernel's bitwise. The former static shared
// arrays (the slabs' tile counts, the listed walk's address round, the group
// merge's ticket flag and weights) live behind the exchange in that memory.
template <int KV, int SD, bool kListed, int kGroup>
constexpr size_t attn_flash_smem_bytes() {
  return dense::Geo<KV, SD, kListed ? 16 : dense::kTile>::smem_bytes + 16 + (kListed ? size_t(2) * 512 * 4 : 0) + 16 +
         size_t(kGroup) * dense::kM * 4;
}
// The body's load batching, the same arithmetic in every form: kFetchPasses
// passes fetch a tile's rows (the production's four hold the registers of two
// blocks an SM; one pass issues every row's loads together, the step engine's
// form at one block a worker), kFoldPer outputs a thread per round of the
// group merge (two in production, four in the step: each round is a round
// trip to L2 under a saturated memory), and kIdentityTable resolves a listed
// token's row without the table load when the table is the identity over the
// requests at one block each (the decode window's ring: request r's region r).
template <int KV, int SD, bool kListed, LatentFormat F, int kGroup = 1, int kFetchPasses = 4, int kFoldPer = 2, bool kIdentityTable = false>
__device__ __forceinline__ void attn_flash_body(
    int bx, int s, uint8_t* dsmem, const uint16_t* __restrict__ q_tilde, const uint8_t* __restrict__ latent,
    const float* __restrict__ latent_scale,
    const int32_t* __restrict__ req_ids, const int64_t* __restrict__ pos,
    const int32_t* __restrict__ topk, int topk_stride,
    const int32_t* __restrict__ counts, int rows, int n_split, int local_heads,
    int block_tokens, const int32_t* __restrict__ block_tables,
    int blocks_per_request, int blocks_total, float scale, float* __restrict__ m_ws,
    float* __restrict__ l_ws, float* __restrict__ c_ws, int32_t* __restrict__ tickets,
    float* __restrict__ mg_ws, float* __restrict__ lg_ws, float* __restrict__ cg_ws) {
  constexpr int NTOK = kListed ? 16 : dense::kTile;  // tokens per tile (per slab when listed)
  using G = dense::Geo<KV, SD, NTOK>;
  constexpr int SQ = G::SQ, CW = G::CW, NT = G::NT, KW = G::KW, KS = G::KS, XF = G::XF;
  constexpr int M = dense::kM;
  constexpr size_t kRowBytes = latent_cache_row_bytes(F, KV, SD - KV);
  constexpr int JT = NTOK / 8;    // n8 tiles of tokens in S
  constexpr int KK = NTOK / 16;   // k16 steps of tokens in P.L
  uint16_t* sLall = reinterpret_cast<uint16_t*>(dsmem);        // dense: [32][SQ]; listed: [2][16][SQ]
  float* sX = reinterpret_cast<float*>(sLall + dense::kTile * SQ);  // [8][XF]
  uint8_t* scratch = dsmem + G::smem_bytes;
  int* s_tiles = reinterpret_cast<int*>(scratch);                                      // [2]
  int32_t* s_phys = reinterpret_cast<int32_t*>(scratch + 16);                          // [2][kAddrRound] (listed)
  int* s_last = reinterpret_cast<int*>(scratch + 16 + (kListed ? size_t(2) * 512 * 4 : 0));  // [1]
  float* s_w = reinterpret_cast<float*>(s_last + 4);                                   // [kGroup][M]

  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int slab = warp / 4, quarter = warp % 4;
  const int r = lane / 4, cc = (lane % 4) * 2;
  const int total_m = rows * local_heads;
  const int m0 = bx * M;
  uint16_t* sL = kListed ? sLall + slab * NTOK * SQ : sLall;

  // The lane's two M-rows (slab rows r and r+8): validity and positions.
  const int m_lo = m0 + slab * 16 + r;
  const int m_hi = m_lo + 8;
  const bool v_lo = m_lo < total_m, v_hi = m_hi < total_m;
  const int64_t p_lo = (!kListed && v_lo) ? pos[m_lo / local_heads] : -1;
  const int64_t p_hi = (!kListed && v_hi) ? pos[m_hi / local_heads] : -1;
  const int qrow_first = m0 / local_heads;
  const int32_t req = req_ids[qrow_first];
  const int32_t* bt = block_tables + int64_t(req) * blocks_per_request;
  int32_t req_slab = req;  // the slab's own request (listed): the identity table's block

  // The split's token range: dense — the block's longest causal prefix,
  // split n ways, masked per M-row; listed — the slab's row's selection,
  // split n ways (the split kernel's formula), no mask beyond the count.
  int t_begin = 0, t_end = 0;
  const int32_t* list = nullptr;
  if constexpr (kListed) {
    const int slab_m = m0 + slab * 16;
    const int qrow = min(slab_m, total_m - 1) / local_heads;
    const int cnt = slab_m < total_m ? counts[qrow] : 0;
    const int chunk = (cnt + n_split - 1) / n_split;
    t_begin = s * chunk;
    t_end = min(cnt, t_begin + chunk);
    list = topk + int64_t(qrow) * topk_stride;
    // The slab's own request's block table (2026-09-06, the decode use:
    // a block's two slabs may be rows of different requests).
    bt = block_tables + int64_t(req_ids[qrow]) * blocks_per_request;
    req_slab = req_ids[qrow];
  } else {
    const int qrow_last = min(total_m, m0 + M) > m0
                              ? (min(total_m, m0 + M) - 1) / local_heads
                              : qrow_first;
    int64_t p_max = -1;
    for (int q = qrow_first; q <= qrow_last; ++q) p_max = max(p_max, pos[q]);
    const int count = static_cast<int>(p_max + 1);
    const int chunk = (count + n_split - 1) / n_split;
    t_begin = s * chunk;
    t_end = min(count, t_begin + chunk);
  }
  // The block's tile count: the longer slab's (listed) or the shared range's.
  int n_tiles = (t_end - t_begin + NTOK - 1) / NTOK;
  if (n_tiles < 0) n_tiles = 0;
  if constexpr (kListed) {
    // Both slabs must run the same number of barrier rounds.
    if (threadIdx.x % 128 == 0) s_tiles[slab] = n_tiles;
    __syncthreads();
    n_tiles = max(s_tiles[0], s_tiles[1]);
  }

  // Q~ fragments of this warp's two rows over its k-quarter, in the mma's
  // A layout, from global once per block (zero past the last M-row).
  uint32_t qa[KS][4];
  {
    const uint16_t* q_lo = q_tilde + int64_t(m_lo) * SD;
    const uint16_t* q_hi = q_tilde + int64_t(m_hi) * SD;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      const int k0 = quarter * KW + ks * 16 + cc;
      qa[ks][0] = v_lo ? *reinterpret_cast<const uint32_t*>(q_lo + k0) : 0u;
      qa[ks][1] = v_hi ? *reinterpret_cast<const uint32_t*>(q_hi + k0) : 0u;
      qa[ks][2] = v_lo ? *reinterpret_cast<const uint32_t*>(q_lo + k0 + 8) : 0u;
      qa[ks][3] = v_hi ? *reinterpret_cast<const uint32_t*>(q_hi + k0 + 8) : 0u;
    }
  }

  float acc[NT][4];
#pragma unroll
  for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  float m_lo_run = -INFINITY, m_hi_run = -INFINITY;
  float l_lo = 0.f, l_hi = 0.f;

  // ---- the gather (2026-10-18) ------------------------------------------------
  // A listed row cost three dependent global loads (the list, the block
  // table, the cache row), eight rows per thread in turn, with one block
  // resident per SM: the DeepSeek-V4.1 32K prefill's largest kernel ran at
  // a few percent of every limit. The listed addresses are now resolved
  // per round of kAddrRound tokens into shared memory — once per token,
  // not once per thread per token — and a tile's rows are fetched in two
  // halves whose loads issue together; the second block on the SM covers
  // the latency. The rows decoded and their placement are unchanged:
  // bitwise the previous walk.
  constexpr int kAddrRound = 512;
  static_assert(!kListed || kAddrRound == 512, "the address round's shared memory is carved for 512");
  constexpr int kTilesPerRound = kAddrRound / NTOK;
  constexpr int GTHREADS = kListed ? 128 : dense::kThreads;  // the gather's threads (per slab when listed)
  constexpr int TPT = NTOK * (SD / 8) / GTHREADS;              // uint4 per thread per tile
  constexpr int HALF = (TPT + kFetchPasses - 1) / kFetchPasses;  // uint4 per pass (four passes: the registers of two blocks per SM)
  static_assert(NTOK * (SD / 8) % GTHREADS == 0, "the gather's uint4s divide its threads");
  const int gtid = threadIdx.x % GTHREADS;
  const auto resolve_round = [&](int t_round0) {
    if constexpr (kListed) {
      for (int i = gtid; i < kAddrRound; i += GTHREADS) {
        const int t = t_round0 + i;
        int32_t phys = -1;
        if (t < t_end) {
          // The gather is guarded: a token outside the table row or a
          // block outside the pool zero-fills the row and records the first
          // anomaly instead of faulting the context — the listed kernel
          // read an unmapped page on the fabric once in ~10 eager rows of
          // the fallback path; the record names the values.
          const int64_t tok = list[t];
          const int64_t bidx = tok >= 0 ? tok / block_tokens : -1;
          // The block must lie in the pool: blocks_total physical blocks
          // (the table's row width bounds the index into the row; a
          // one-entry table over many physical blocks — the decode
          // window's ring, request r's region r — is bounded by the
          // pool's count, 2026-10-19).
          const int32_t blk = kIdentityTable ? (bidx == 0 && blocks_per_request == 1 ? req_slab : -1)
                                             : ((bidx >= 0 && bidx < blocks_per_request) ? bt[bidx] : -1);
          if (blk >= 0 && blk < blocks_total)
            phys = static_cast<int32_t>(int64_t(blk) * block_tokens + (tok % block_tokens));
          else
            attn_anomaly_record(tok, blk, qrow_first, s, t, t_end - t_begin);
        }
        s_phys[slab * kAddrRound + i] = phys;
      }
    }
  };
  // The tile's rows into the shared tile (zero-filled past the slab's live
  // tokens), HALF uint4 per thread per pass.
  const auto fetch_tile = [&](int tile) {
    const int t0 = t_begin + tile * NTOK;
    const int n = max(0, min(NTOK, t_end - t0));
#pragma unroll
    for (int h = 0; h < TPT; h += HALF) {
      uint4 val[HALF];
#pragma unroll
      for (int k = 0; k < HALF; ++k) {
        const int idx = gtid + (h + k) * GTHREADS;
        const int tt = idx / (SD / 8), c8 = idx % (SD / 8);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (h + k < TPT && tt < n) {
          int64_t phys;
          if constexpr (kListed) {
            phys = s_phys[slab * kAddrRound + (tile % kTilesPerRound) * NTOK + tt];
          } else {
            const int64_t tok = t0 + tt;
            const int32_t blk = bt[tok / block_tokens];
            phys = int64_t(blk) * block_tokens + (tok % block_tokens);
          }
          if (phys >= 0) v = LatentTile<F>::load8(latent, latent_scale, phys, kRowBytes, KV, c8 * 8);
        }
        val[k] = v;
      }
#pragma unroll
      for (int k = 0; k < HALF; ++k) {
        if (h + k >= TPT) break;
        const int idx = gtid + (h + k) * GTHREADS;
        const int tt = idx / (SD / 8), c8 = idx % (SD / 8);
        *reinterpret_cast<uint4*>(sL + tt * SQ + c8 * 8) = val[k];
      }
    }
  };
  resolve_round(t_begin);

  for (int tile = 0; tile < n_tiles; ++tile) {
    const int t0 = t_begin + tile * NTOK;
    const int n = max(0, min(NTOK, t_end - t0));  // this slab's live tokens
    (void)t0;  // the dense mask reads it
    if (tile > 0 && tile % kTilesPerRound == 0) resolve_round(t0);  // the round's addresses, ahead of its first fetch
    __syncthreads();  // the previous tile's readers are done; the round's addresses landed
    fetch_tile(tile);
    __syncthreads();

    // S partial over this warp's k-quarter: 16 rows x NTOK tokens.
    float sp[JT][4];
#pragma unroll
    for (int j = 0; j < JT; ++j) sp[j][0] = sp[j][1] = sp[j][2] = sp[j][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      const int k0 = quarter * KW + ks * 16;
      const uint32_t a0 = qa[ks][0], a1 = qa[ks][1], a2 = qa[ks][2], a3 = qa[ks][3];
#pragma unroll
      for (int j = 0; j < JT; ++j) {
        const int tn = j * 8 + r;  // token within the tile (the n index)
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sL[tn * SQ + k0 + cc]);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sL[tn * SQ + k0 + cc + 8]);
        mma_bf16_16816(sp[j], a0, a1, a2, a3, b0, b1);
      }
    }
    // Exchange: every warp of the slab sums the four quarters in the same
    // order, so the slab's warps hold bitwise-identical S.
    float* mine = sX + warp * XF + lane * (JT * 4);
#pragma unroll
    for (int j = 0; j < JT; ++j) {
      mine[j * 4 + 0] = sp[j][0];
      mine[j * 4 + 1] = sp[j][1];
      mine[j * 4 + 2] = sp[j][2];
      mine[j * 4 + 3] = sp[j][3];
    }
    __syncthreads();
    float sc[JT][4];
#pragma unroll
    for (int j = 0; j < JT; ++j) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float* q0 = sX + (slab * 4 + 0) * XF + lane * (JT * 4) + j * 4 + i;
        float v = q0[0];
        v += q0[1 * XF];
        v += q0[2 * XF];
        v += q0[3 * XF];
        sc[j][i] = v;
      }
    }
    // Scale and mask (causal per M-row when dense; the count when listed);
    // the row maxima of this tile.
    float mx_lo = -INFINITY, mx_hi = -INFINITY;
#pragma unroll
    for (int j = 0; j < JT; ++j) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int tt = j * 8 + cc + (i & 1);
        const bool hi = i >= 2;
        bool ok = tt < n && (hi ? v_hi : v_lo);
        if constexpr (!kListed) {
          const int64_t tok = t0 + tt;
          ok = ok && (hi ? tok <= p_hi : tok <= p_lo);
        }
        const float val = ok ? sc[j][i] * scale : -INFINITY;
        sc[j][i] = val;
        if (hi) mx_hi = fmaxf(mx_hi, val); else mx_lo = fmaxf(mx_lo, val);
      }
    }
    mx_lo = fmaxf(mx_lo, __shfl_xor_sync(~0u, mx_lo, 1));
    mx_lo = fmaxf(mx_lo, __shfl_xor_sync(~0u, mx_lo, 2));
    mx_hi = fmaxf(mx_hi, __shfl_xor_sync(~0u, mx_hi, 1));
    mx_hi = fmaxf(mx_hi, __shfl_xor_sync(~0u, mx_hi, 2));
    const float mn_lo = fmaxf(m_lo_run, mx_lo);
    const float mn_hi = fmaxf(m_hi_run, mx_hi);
    const float rs_lo = mn_lo == -INFINITY ? 1.f : expf(m_lo_run - mn_lo);
    const float rs_hi = mn_hi == -INFINITY ? 1.f : expf(m_hi_run - mn_hi);
    // Probabilities: fp32 for l, bf16 for the c accumulation (the pin).
    uint32_t pa[KK][4];
    float la_lo = 0.f, la_hi = 0.f;
#pragma unroll
    for (int j = 0; j < JT; ++j) {
      float pv[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const bool hi = i >= 2;
        const float mn = hi ? mn_hi : mn_lo;
        pv[i] = (sc[j][i] == -INFINITY) ? 0.f : expf(sc[j][i] - mn);
        if (hi) la_hi += pv[i]; else la_lo += pv[i];
      }
      const uint32_t lo01 = uint32_t(float_to_bf16_bits(pv[0])) |
                            (uint32_t(float_to_bf16_bits(pv[1])) << 16);
      const uint32_t hi01 = uint32_t(float_to_bf16_bits(pv[2])) |
                            (uint32_t(float_to_bf16_bits(pv[3])) << 16);
      // k16 step kk = j / 2 holds n8 tiles 2kk (regs 0,1) and 2kk+1 (2,3).
      pa[j / 2][(j % 2) * 2 + 0] = lo01;
      pa[j / 2][(j % 2) * 2 + 1] = hi01;
    }
    la_lo += __shfl_xor_sync(~0u, la_lo, 1);
    la_lo += __shfl_xor_sync(~0u, la_lo, 2);
    la_hi += __shfl_xor_sync(~0u, la_hi, 1);
    la_hi += __shfl_xor_sync(~0u, la_hi, 2);
    l_lo = l_lo * rs_lo + la_lo;
    l_hi = l_hi * rs_hi + la_hi;
    m_lo_run = mn_lo;
    m_hi_run = mn_hi;
    // O rescale, then O += P . L over the warp's columns.
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      acc[j][0] *= rs_lo;
      acc[j][1] *= rs_lo;
      acc[j][2] *= rs_hi;
      acc[j][3] *= rs_hi;
    }
#pragma unroll
    for (int kk = 0; kk < KK; ++kk) {
      const int tb = kk * 16 + cc;  // the fragment's first token (k index)
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const int d = quarter * CW + j * 8 + r;  // output column (n index)
        const uint32_t b0 = uint32_t(sL[tb * SQ + d]) | (uint32_t(sL[(tb + 1) * SQ + d]) << 16);
        const uint32_t b1 = uint32_t(sL[(tb + 8) * SQ + d]) | (uint32_t(sL[(tb + 9) * SQ + d]) << 16);
        mma_bf16_16816(acc[j], pa[kk][0], pa[kk][1], pa[kk][2], pa[kk][3], b0, b1);
      }
    }
  }

  // Publish the split partials in the combine kernel's layout.
  auto publish = [&](int mrow, float mrun, float lrun, int which) {
    if (mrow >= total_m) return;
    const int qr = mrow / local_heads, h = mrow % local_heads;
    const int64_t idx = (int64_t(qr) * n_split + s) * local_heads + h;
    if (cc == 0) {
      m_ws[idx] = mrun;
      l_ws[idx] = lrun;
    }
    float* crow = c_ws + idx * KV;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int d = quarter * CW + j * 8 + cc;
      *reinterpret_cast<float2*>(crow + d) =
          which == 0 ? make_float2(acc[j][0], acc[j][1])
                     : make_float2(acc[j][2], acc[j][3]);
    }
  };
  publish(m_lo, m_lo_run, l_lo, 0);
  publish(m_hi, m_hi_run, l_hi, 1);
  if constexpr (kGroup > 1) {
    // The group merge: splits s0 .. s0 + kGroup - 1 of this row pair take a
    // ticket (tickets[bx * n_groups + g], zero at rest, reset by
    // the last — replay-safe), and the LAST block to arrive merges the
    // group's partials from L2 with the finish kernel's formula — mhat =
    // max m_s, w_s = exp(m_s - mhat), l = sum w_s l_s, c = sum w_s c_s in
    // split order — into the group's partial at split index g of
    // n_split / kGroup (mg_ws / lg_ws / cg_ws), then discards the eight
    // dead c rows from L2. The grouping is fixed per request (kGroup
    // splits a group whatever the batch), so a row's output does not
    // depend on the rows beside it; an empty split (m = -inf, l = 0,
    // c = 0) weighs nothing and an empty group publishes m = -inf, which
    // the finish skips. The fence / ticket order is the router's: a block
    // fences its partial before its ticket; the last fences again before
    // reading them (L2 loads: the rows were written by other SMs).
    const int g = s / kGroup;
    const int n_groups = n_split / kGroup;
    const int s0 = g * kGroup;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      int32_t* t = tickets + static_cast<int64_t>(bx) * n_groups + g;
      const int ticket = atomicAdd(t, 1);
      *s_last = ticket == kGroup - 1 ? 1 : 0;
      if (*s_last) *t = 0;
    }
    __syncthreads();
    if (!*s_last) return;
    __threadfence();
    for (int mrow = static_cast<int>(threadIdx.x); mrow < M; mrow += dense::kThreads) {
      const int m = m0 + mrow;
      if (m >= total_m) continue;
      const int qr = m / local_heads, h = m % local_heads;
      float mb[kGroup], lb[kGroup];
      float mhat = -INFINITY;
#pragma unroll
      for (int b = 0; b < kGroup; ++b) {
        const int64_t idx = (int64_t(qr) * n_split + s0 + b) * local_heads + h;
        mb[b] = __ldcg(m_ws + idx);
        lb[b] = __ldcg(l_ws + idx);
        mhat = fmaxf(mhat, mb[b]);
      }
      float lsum = 0.f;
#pragma unroll
      for (int b = 0; b < kGroup; ++b) {
        const float w = (mhat == -INFINITY || mb[b] == -INFINITY) ? 0.f : expf(mb[b] - mhat);
        lsum += w * lb[b];
        s_w[b * M + mrow] = w;
      }
      const int64_t gidx = (int64_t(qr) * n_groups + g) * local_heads + h;
      mg_ws[gidx] = mhat;
      lg_ws[gidx] = lsum;
    }
    __syncthreads();
    // The c rows: M rows x KV columns as float4s over the block, kPer
    // outputs a thread a round with every group vector of the round
    // loaded before the first add — 16 L2 loads in flight a thread (one
    // output a round left one block reading 512 KB at ~20 GB/s: a 25 us
    // tail on every launch, 2026-10-23). A dead output (an M-row past the
    // batch) loads the block's first row and stores nothing.
    constexpr int V4 = KV / 4;
    constexpr int kPer = kFoldPer;
    const int qr0 = m0 / local_heads, h0 = m0 % local_heads;
    for (int i0 = static_cast<int>(threadIdx.x); i0 < M * V4; i0 += kPer * dense::kThreads) {
      float4 v[kPer][kGroup];
      int mrow[kPer], col[kPer], qr[kPer], h[kPer];
      bool live[kPer];
#pragma unroll
      for (int u = 0; u < kPer; ++u) {
        const int i = i0 + u * dense::kThreads;
        live[u] = i < M * V4 && m0 + i / V4 < total_m;
        mrow[u] = live[u] ? i / V4 : 0;
        col[u] = live[u] ? (i % V4) * 4 : 0;
        const int m = m0 + mrow[u];
        qr[u] = live[u] ? m / local_heads : qr0;
        h[u] = live[u] ? m % local_heads : h0;
#pragma unroll
        for (int b = 0; b < kGroup; ++b)
          v[u][b] = __ldcg(reinterpret_cast<const float4*>(
              c_ws + ((int64_t(qr[u]) * n_split + s0 + b) * local_heads + h[u]) * KV + col[u]));
      }
#pragma unroll
      for (int u = 0; u < kPer; ++u) {
        if (!live[u]) continue;
        float4 csum = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int b = 0; b < kGroup; ++b) {
          const float w = s_w[b * M + mrow[u]];
          csum.x += w * v[u][b].x;
          csum.y += w * v[u][b].y;
          csum.z += w * v[u][b].z;
          csum.w += w * v[u][b].w;
        }
        const int64_t gidx = (int64_t(qr[u]) * n_groups + g) * local_heads + h[u];
        *reinterpret_cast<float4*>(cg_ws + gidx * KV + col[u]) = csum;
      }
    }
    // The group's per-split c rows are dead: discard their 128-byte lines
    // from L2 (the rows are 2 KB on a 256-byte-aligned buffer) so the
    // fifty megabytes a layer never write back to DRAM — after every
    // thread's merge reads of them (a thread ahead discarded lines a
    // thread behind was still loading: the group-prefill invariance gate
    // and the 40-step decode budget caught it, 2026-10-23).
    __syncthreads();
    constexpr int kLines = KV * 4 / 128;
    for (int i = static_cast<int>(threadIdx.x); i < kGroup * M * kLines; i += dense::kThreads) {
      const int b = i / (M * kLines), rem = i % (M * kLines), mrow = rem / kLines, line = rem % kLines;
      const int m = m0 + mrow;
      if (m >= total_m) continue;
      const int qr = m / local_heads, h = m % local_heads;
      const float* row = c_ws + ((int64_t(qr) * n_split + s0 + b) * local_heads + h) * KV;
      asm volatile("discard.global.L2 [%0], 128;" ::"l"(row + line * 32) : "memory");
    }
  }
}

}  // namespace dsa_flash
}  // namespace dgpp
