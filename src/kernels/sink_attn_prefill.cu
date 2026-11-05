#include <cmath>
#include "kernels/pdl.cuh"
#include <stdexcept>
#include <string>

#include <cuda_bf16.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/latent_format.hpp"
#include "kernels/sink_attn.hpp"

// The query-tiled prefill attention (2026-09-22, docs/mimo_v26_flash_plan.md
// §7.2; a template over the qk width since 2026-10-22, docs/flint_plan.md
// D3). The split-KV kernel runs one block per (query row, kv head) over
// the row's whole visible range: every K/V tile is staged — and under the
// fp8 cache dequantized — once per QUERY ROW, and the scores and the PV
// products run on scalar fp32 FMAs: an O(T²) re-read plus O(T²) scalar
// work that the 32K prefill shows. Here a block owns a TILE of kQueryRows
// consecutive rows x the kv head's hpk query heads (128 query vectors: 8
// rows x 16 heads on a MiMo global layer, 16 x 8 on its sliding-window
// layers and on Flint's), stages each 32-token K/V tile once for all of
// them, and runs the arithmetic on the tensor cores. The query vectors
// per block set the kernel's arithmetic intensity on the K/V bytes
// (every vector does 512 FLOP per 512-byte key/value pair: the intensity
// in FLOP per byte IS the vector count): at 64 the tiles streamed from
// DRAM capped Flint's 32K prefill near 15 TF (2026-10-22, the campaign's
// first leg: 0.43 ms per token at 32K, the 14 full-attention layers'
// tiles past the L2); 128 doubles the bound at the same per-warp code
// (eight warps, each still its own 16-vector m-tile).
//
//
//   S = Q K^T   mma.m16n8k16 bf16 x bf16 -> fp32 (the bf16 queries and
//               cached keys, exact products, fp32 accumulation), x scale
//   softmax     online per query row over the tiles: fp32 max, expf, the
//               probabilities rounded to bf16 for the PV product (the
//               decode kernel's rounding), the denominator unrounded
//   O += P V    mma bf16 x bf16 -> fp32
//   out = bf16(O / L)
//
// The summation ORDER differs from the split-KV kernel's fixed fp32 FMA
// chain (the tensor core's), so a prefill row is no longer bitwise the
// decode kernel's row; the terms are the same exact products, the
// tolerance is the fp32 accumulation's, and the kernel test pins it at a
// bf16 ulp. The per-row masks carry the causal bound and the window; the
// sink enters each head's initial (m, l) as in the decode kernel; a
// block whose rows span several requests (the group prefill's span
// boundaries) loops over the runs of one request, the other rows inert
// (every key masked: their statistics and accumulators do not move).
//
// Warp w of the eight owns query vectors [16 w, 16 w + 16) (mma rows): on
// a MiMo global layer one row's 16 heads, on an 8-heads-per-kv layer two
// rows' eight heads each. Fragments: the C layout puts rows g = lane / 4 and
// g + 8 in each lane, columns 2 (lane % 4) + {0, 1} of each n-tile — so
// the row statistics reduce over the four lanes of a row, and the S
// fragments re-pack directly as the A operand of PV (the FA2 register
// trick). K rows (row-major, stride kKStride) serve the B fragments of
// S as 32-bit loads; V is staged TRANSPOSED (Vt[dim][key], stride kVtStride)
// so the B fragments of PV are 32-bit loads too. Both strides are
// bank-conflict-free for the fragment access pattern (see the constants).
namespace dgpp {
namespace sink_prefill {

constexpr int DV = kSinkAttnVDim;
constexpr int kTile = kSinkAttnTile;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kQueryVecs = kWarps * 16;  // 128 query vectors per block
static_assert(kQueryVecs == kSinkAttnPrefillQueryVecs, "the header's constant is this kernel's");
constexpr int kVtStride = kTile + 8;     // 40 bf16 = 20 words: (20 g + t) mod 32 distinct for g < 8, t < 4
constexpr int kNTilesS = kTile / 8;      // 4 n-tiles of 8 keys
constexpr int kNTilesO = DV / 8;         // 16 n-tiles of 8 v dims
constexpr int kKStepsO = kTile / 16;     // 2 k-steps over the tile's keys

// The qk-width-dependent strides: DK + 8 bf16 per staged Q / K row — at
// 192 200 bf16 = 100 words, at 128 136 = 68 words: (words g + t) mod 32
// distinct for g < 8, t < 4 either way — and DK / 16 k-steps of S.
template <int DK>
struct PGeo {
  static constexpr int kQStride = DK + 8;
  static constexpr int kKStride = DK + 8;
  static constexpr int kKStepsS = DK / 16;
  static_assert(DK % 16 == 0 && DV % 8 == 0 && kTile % 16 == 0, "the mma tiling");
  static_assert((kQStride * 2) % 16 == 0 && (kKStride * 2) % 16 == 0, "16-byte tile rows");
};

__device__ __forceinline__ void mma_bf16_16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  return static_cast<uint32_t>(float_to_bf16_bits(lo)) | (static_cast<uint32_t>(float_to_bf16_bits(hi)) << 16);
}

__device__ __forceinline__ float round_bf16(float v) { return bf16_bits_to_float(float_to_bf16_bits(v)); }

// Sixteen e4m3 codes decoded with their row's scale into eight bf16 words.
__device__ __forceinline__ void decode_fp8_piece(const uint4 codes, float scale, uint32_t* words) {
  const uint32_t c[4] = {codes.x, codes.y, codes.z, codes.w};
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const uint16_t e0 = latent_fp8_decode_bf16(static_cast<uint8_t>(c[w] & 0xFFu), scale);
    const uint16_t e1 = latent_fp8_decode_bf16(static_cast<uint8_t>((c[w] >> 8) & 0xFFu), scale);
    const uint16_t e2 = latent_fp8_decode_bf16(static_cast<uint8_t>((c[w] >> 16) & 0xFFu), scale);
    const uint16_t e3 = latent_fp8_decode_bf16(static_cast<uint8_t>(c[w] >> 24), scale);
    words[2 * w] = static_cast<uint32_t>(e0) | (static_cast<uint32_t>(e1) << 16);
    words[2 * w + 1] = static_cast<uint32_t>(e2) | (static_cast<uint32_t>(e3) << 16);
  }
}

template <int DK, bool kFp8>
__global__ void __launch_bounds__(kThreads) attn_prefill_kernel(SinkAttnPrefillArgs a) {
  dgpp_pdl_wait();
  constexpr int kQStride = PGeo<DK>::kQStride;
  constexpr int kKStride = PGeo<DK>::kKStride;
  constexpr int kKStepsS = PGeo<DK>::kKStepsS;
  extern __shared__ __align__(16) uint16_t smem[];
  uint16_t* qt = smem;                        // [kQueryVecs][kQStride]
  uint16_t* kt = qt + kQueryVecs * kQStride;  // [kTile][kKStride]
  uint16_t* vt = kt + kTile * kKStride;       // [DV][kVtStride] transposed
  const int hpk = a.local_heads / a.kv_heads;
  const int rows_per_block = kQueryVecs / hpk;
  const int r0 = static_cast<int>(blockIdx.x) * rows_per_block;
  const int kvh = static_cast<int>(blockIdx.y);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int g = lane / 4, t = lane % 4;
  const int kwidth = a.kv_heads * DK, vwidth = a.kv_heads * DV;

  // The block's rows: (request, position) — rows past the batch are padding.
  __shared__ int32_t s_req[kQueryVecs];
  __shared__ int64_t s_pos[kQueryVecs];
  for (int i = threadIdx.x; i < rows_per_block; i += kThreads) {
    const int r = r0 + i;
    s_req[i] = r < a.rows ? a.req_ids[r] : -1;
    s_pos[i] = r < a.rows ? a.pos[r] : -1;
  }
  // The query vectors: qv = row_in_tile * hpk + head, the finished bf16
  // rows of `q` (stride q_stride, heads contiguous).
  for (int i = threadIdx.x; i < kQueryVecs * (DK / 8); i += kThreads) {
    const int qv = i / (DK / 8), c8 = i - qv * (DK / 8);
    const int ri = qv / hpk, h = kvh * hpk + qv - ri * hpk;
    const int r = r0 + ri;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (r < a.rows && a.pos[r] >= 0)
      v = *reinterpret_cast<const uint4*>(a.q + static_cast<int64_t>(r) * a.q_stride + static_cast<int64_t>(h) * DK + c8 * 8);
    *reinterpret_cast<uint4*>(qt + qv * kQStride + c8 * 8) = v;
  }
  __syncthreads();

  // This lane's two query vectors (mma rows g and g + 8 of the warp's m-tile).
  const int qv0 = warp * 16 + g, qv1 = qv0 + 8;
  const int ri0 = qv0 / hpk, ri1 = qv1 / hpk;
  const int h0 = kvh * hpk + qv0 - ri0 * hpk, h1 = kvh * hpk + qv1 - ri1 * hpk;
  const int64_t p0 = s_pos[ri0], p1 = s_pos[ri1];
  const int64_t lo0 = a.window > 0 ? (p0 - a.window + 1 > 0 ? p0 - a.window + 1 : 0) : 0;
  const int64_t lo1 = a.window > 0 ? (p1 - a.window + 1 > 0 ? p1 - a.window + 1 : 0) : 0;
  float m0 = -INFINITY, l0 = 0.f, m1 = -INFINITY, l1 = 0.f;
  if (a.sink != nullptr) {
    if (p0 >= 0) { m0 = a.sink[h0]; l0 = 1.f; }
    if (p1 >= 0) { m1 = a.sink[h1]; l1 = 1.f; }
  }
  float o[kNTilesO][4];
#pragma unroll
  for (int j = 0; j < kNTilesO; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;

  // The runs of one request among the block's rows (a run is consecutive
  // rows, consecutive positions): each run's union range, tile by tile.
  int run0 = 0;
  while (run0 < rows_per_block) {
    const int32_t req = s_req[run0];
    if (req < 0 || s_pos[run0] < 0) { ++run0; continue; }
    int run1 = run0 + 1;
    while (run1 < rows_per_block && s_req[run1] == req && s_pos[run1] == s_pos[run1 - 1] + 1) ++run1;
    const int64_t first = s_pos[run0], last = s_pos[run1 - 1];
    const int64_t t_lo = a.window > 0 ? (first - a.window + 1 > 0 ? first - a.window + 1 : 0) : 0;
    const int32_t* bt = a.block_tables + static_cast<int64_t>(req) * a.blocks_per_request;
    const bool in0 = ri0 >= run0 && ri0 < run1, in1 = ri1 >= run0 && ri1 < run1;
    for (int64_t t0 = t_lo; t0 <= last; t0 += kTile) {
      const int n = last + 1 - t0 < kTile ? static_cast<int>(last + 1 - t0) : kTile;
      __syncthreads();  // the previous tile's readers are done
      // K rows: pieces of 16 bytes.
      if constexpr (kFp8) {
        const uint8_t* kc = reinterpret_cast<const uint8_t*>(a.k_cache);
        for (int idx = threadIdx.x; idx < n * (DK / 16); idx += kThreads) {
          const int tt = idx / (DK / 16), c16 = idx - tt * (DK / 16);
          const int64_t tok = t0 + tt;
          const int64_t phys = static_cast<int64_t>(bt[tok / a.block_tokens]) * a.block_tokens + tok % a.block_tokens;
          const uint4 codes = *reinterpret_cast<const uint4*>(kc + phys * kwidth + kvh * DK + c16 * 16);
          uint32_t words[8];
          decode_fp8_piece(codes, a.k_scale[phys * a.kv_heads + kvh], words);
          uint32_t* dst = reinterpret_cast<uint32_t*>(kt + tt * kKStride) + c16 * 8;
#pragma unroll
          for (int w = 0; w < 8; ++w) dst[w] = words[w];
        }
        const uint8_t* vc = reinterpret_cast<const uint8_t*>(a.v_cache);
        for (int idx = threadIdx.x; idx < n * (DV / 16); idx += kThreads) {
          const int tt = idx / (DV / 16), c16 = idx - tt * (DV / 16);
          const int64_t tok = t0 + tt;
          const int64_t phys = static_cast<int64_t>(bt[tok / a.block_tokens]) * a.block_tokens + tok % a.block_tokens;
          const uint4 codes = *reinterpret_cast<const uint4*>(vc + phys * vwidth + kvh * DV + c16 * 16);
          uint32_t words[8];
          decode_fp8_piece(codes, a.v_scale[phys * a.kv_heads + kvh], words);
#pragma unroll
          for (int w = 0; w < 8; ++w) {
            vt[(c16 * 16 + 2 * w) * kVtStride + tt] = static_cast<uint16_t>(words[w] & 0xFFFFu);
            vt[(c16 * 16 + 2 * w + 1) * kVtStride + tt] = static_cast<uint16_t>(words[w] >> 16);
          }
        }
      } else {
        for (int idx = threadIdx.x; idx < n * (DK / 8); idx += kThreads) {
          const int tt = idx / (DK / 8), c8 = idx - tt * (DK / 8);
          const int64_t tok = t0 + tt;
          const int64_t phys = static_cast<int64_t>(bt[tok / a.block_tokens]) * a.block_tokens + tok % a.block_tokens;
          const uint4 val = *reinterpret_cast<const uint4*>(a.k_cache + phys * kwidth + kvh * DK + c8 * 8);
          uint32_t* dst = reinterpret_cast<uint32_t*>(kt + tt * kKStride) + c8 * 4;
          dst[0] = val.x;
          dst[1] = val.y;
          dst[2] = val.z;
          dst[3] = val.w;
        }
        for (int idx = threadIdx.x; idx < n * (DV / 8); idx += kThreads) {
          const int tt = idx / (DV / 8), c8 = idx - tt * (DV / 8);
          const int64_t tok = t0 + tt;
          const int64_t phys = static_cast<int64_t>(bt[tok / a.block_tokens]) * a.block_tokens + tok % a.block_tokens;
          const uint4 val = *reinterpret_cast<const uint4*>(a.v_cache + phys * vwidth + kvh * DV + c8 * 8);
          const uint32_t words[4] = {val.x, val.y, val.z, val.w};
#pragma unroll
          for (int w = 0; w < 4; ++w) {
            vt[(c8 * 8 + 2 * w) * kVtStride + tt] = static_cast<uint16_t>(words[w] & 0xFFFFu);
            vt[(c8 * 8 + 2 * w + 1) * kVtStride + tt] = static_cast<uint16_t>(words[w] >> 16);
          }
        }
      }
      // Rows past n: zero keys (their scores are masked below anyway, but
      // the operands must be finite).
      for (int idx = threadIdx.x; idx < (kTile - n) * (DK / 8); idx += kThreads) {
        const int tt = n + idx / (DK / 8), c8 = idx - (tt - n) * (DK / 8);
        *reinterpret_cast<uint4*>(kt + tt * kKStride + c8 * 8) = make_uint4(0u, 0u, 0u, 0u);
      }
      for (int idx = threadIdx.x; idx < (kTile - n) * DV; idx += kThreads) {
        const int tt = n + idx / DV, d = idx - (tt - n) * DV;
        vt[d * kVtStride + tt] = 0;
      }
      __syncthreads();

      // S = Q K^T for the warp's 16 query vectors x the tile's 32 keys.
      float s[kNTilesS][4];
#pragma unroll
      for (int j = 0; j < kNTilesS; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
      const uint32_t* q0w = reinterpret_cast<const uint32_t*>(qt + qv0 * kQStride);
      const uint32_t* q1w = reinterpret_cast<const uint32_t*>(qt + qv1 * kQStride);
#pragma unroll
      for (int ks = 0; ks < kKStepsS; ++ks) {
        uint32_t af[4];
        af[0] = q0w[ks * 8 + t];
        af[1] = q1w[ks * 8 + t];
        af[2] = q0w[ks * 8 + t + 4];
        af[3] = q1w[ks * 8 + t + 4];
#pragma unroll
        for (int j = 0; j < kNTilesS; ++j) {
          const uint32_t* kw = reinterpret_cast<const uint32_t*>(kt + (j * 8 + g) * kKStride);
          uint32_t bf[2];
          bf[0] = kw[ks * 8 + t];
          bf[1] = kw[ks * 8 + t + 4];
          mma_bf16_16816(s[j], af, bf);
        }
      }
      // Scale, mask (causal, window, the run, padding), the row statistics.
      float tmax0 = -INFINITY, tmax1 = -INFINITY;
#pragma unroll
      for (int j = 0; j < kNTilesS; ++j) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const int64_t key = t0 + j * 8 + 2 * t + c;
          const bool ok0 = in0 && key < n + t0 && key >= lo0 && key <= p0;
          const bool ok1 = in1 && key < n + t0 && key >= lo1 && key <= p1;
          s[j][c] = ok0 ? s[j][c] * a.scale : -INFINITY;
          s[j][2 + c] = ok1 ? s[j][2 + c] * a.scale : -INFINITY;
          tmax0 = fmaxf(tmax0, s[j][c]);
          tmax1 = fmaxf(tmax1, s[j][2 + c]);
        }
      }
#pragma unroll
      for (int off = 1; off <= 2; off <<= 1) {
        tmax0 = fmaxf(tmax0, __shfl_xor_sync(~0u, tmax0, off));
        tmax1 = fmaxf(tmax1, __shfl_xor_sync(~0u, tmax1, off));
      }
      const float mn0 = fmaxf(m0, tmax0), mn1 = fmaxf(m1, tmax1);
      // The rescale: 0 when m was -inf and the tile brought a key, 1 when
      // nothing has arrived yet (an inert row; expf(-inf - -inf) is NaN).
      const float rs0 = mn0 > -INFINITY ? expf(m0 - mn0) : 1.f;
      const float rs1 = mn1 > -INFINITY ? expf(m1 - mn1) : 1.f;
      float e0 = 0.f, e1 = 0.f;
      uint32_t pf[kKStepsO][4];
#pragma unroll
      for (int j = 0; j < kNTilesS; ++j) {
        float p[4];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          p[c] = s[j][c] > -INFINITY ? expf(s[j][c] - mn0) : 0.f;
          p[2 + c] = s[j][2 + c] > -INFINITY ? expf(s[j][2 + c] - mn1) : 0.f;
          e0 += p[c];
          e1 += p[2 + c];
        }
        // The A fragment of PV from the C fragment of S: n-tile j is
        // k-step j / 2, half j % 2 (rows g: a0 / a2; rows g + 8: a1 / a3).
        pf[j / 2][(j % 2) * 2 + 0] = pack_bf16x2(round_bf16(p[0]), round_bf16(p[1]));
        pf[j / 2][(j % 2) * 2 + 1] = pack_bf16x2(round_bf16(p[2]), round_bf16(p[3]));
      }
#pragma unroll
      for (int off = 1; off <= 2; off <<= 1) {
        e0 += __shfl_xor_sync(~0u, e0, off);
        e1 += __shfl_xor_sync(~0u, e1, off);
      }
      l0 = l0 * rs0 + e0;
      l1 = l1 * rs1 + e1;
      m0 = mn0;
      m1 = mn1;
#pragma unroll
      for (int j = 0; j < kNTilesO; ++j) {
        o[j][0] *= rs0;
        o[j][1] *= rs0;
        o[j][2] *= rs1;
        o[j][3] *= rs1;
      }
      // O += P V over the tile's 32 keys (two k-steps), 16 n-tiles of dims.
#pragma unroll
      for (int ks = 0; ks < kKStepsO; ++ks) {
        // pf[ks]: {a0 = P[g][16ks + 2t..], a1 = P[g+8][16ks + 2t..], a2 = P[g][16ks + 8 + 2t..], a3 = P[g+8][...]}
        const uint32_t af[4] = {pf[ks][0], pf[ks][1], pf[ks][2], pf[ks][3]};
#pragma unroll
        for (int j = 0; j < kNTilesO; ++j) {
          const uint32_t* vw = reinterpret_cast<const uint32_t*>(vt + (j * 8 + g) * kVtStride);
          uint32_t bf[2];
          bf[0] = vw[ks * 8 + t];
          bf[1] = vw[ks * 8 + t + 4];
          mma_bf16_16816(o[j], af, bf);
        }
      }
    }
    run0 = run1;
  }
  // out[(row, head)][dims]: rows g and g + 8 of the m-tile; dims 8 j + 2 t, + 1.
  const int r0g = r0 + ri0, r1g = r0 + ri1;
  if (r0g < a.rows) {
    uint16_t* dst = a.out + (static_cast<int64_t>(r0g) * a.local_heads + h0) * DV;
#pragma unroll
    for (int j = 0; j < kNTilesO; ++j)
      *reinterpret_cast<uint32_t*>(dst + j * 8 + 2 * t) =
          pack_bf16x2(l0 > 0.f ? o[j][0] / l0 : 0.f, l0 > 0.f ? o[j][1] / l0 : 0.f);
  }
  if (r1g < a.rows) {
    uint16_t* dst = a.out + (static_cast<int64_t>(r1g) * a.local_heads + h1) * DV;
#pragma unroll
    for (int j = 0; j < kNTilesO; ++j)
      *reinterpret_cast<uint32_t*>(dst + j * 8 + 2 * t) =
          pack_bf16x2(l1 > 0.f ? o[j][2] / l1 : 0.f, l1 > 0.f ? o[j][3] / l1 : 0.f);
  }
}

template <int DK>
size_t smem_bytes() {
  return static_cast<size_t>(kQueryVecs) * PGeo<DK>::kQStride * 2 + static_cast<size_t>(kTile) * PGeo<DK>::kKStride * 2 +
         static_cast<size_t>(DV) * kVtStride * 2;
}

template <int DK>
void launch(const SinkAttnPrefillArgs& a, cudaStream_t stream) {
  if (a.rows <= 0) return;
  if (!a.q || !a.k_cache || !a.v_cache || !a.req_ids || !a.pos || !a.block_tables || !a.out)
    throw std::invalid_argument("sink_attn_prefill: null pointer");
  if ((a.k_scale == nullptr) != (a.v_scale == nullptr))
    throw std::invalid_argument("sink_attn_prefill: the fp8 cache's K and V scale planes come together");
  if (a.local_heads <= 0 || a.kv_heads <= 0 || a.local_heads % a.kv_heads != 0)
    throw std::invalid_argument("sink_attn_prefill: heads");
  const int hpk = a.local_heads / a.kv_heads;
  if (kQueryVecs % hpk != 0 || hpk > kQueryVecs)
    throw std::invalid_argument("sink_attn_prefill: the query heads per kv head must divide 128");
  if (a.block_tokens <= 0 || a.block_tokens % kTile != 0)
    throw std::invalid_argument("sink_attn_prefill: block_tokens must be a multiple of the 32-token tile");
  if (a.window < 0) throw std::invalid_argument("sink_attn_prefill: window");
  if ((a.q_stride * 2) % 16 != 0) throw std::invalid_argument("sink_attn_prefill: q rows must be 16-byte aligned");
  const size_t smem = smem_bytes<DK>();
  static const bool attr_ok = [smem] {
    return cudaFuncSetAttribute(attn_prefill_kernel<DK, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(smem)) == cudaSuccess &&
           cudaFuncSetAttribute(attn_prefill_kernel<DK, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(smem)) == cudaSuccess;
  }();
  if (!attr_ok) throw std::runtime_error("sink_attn_prefill: the shared-memory opt-in was refused");
  const int rpb = kQueryVecs / hpk;
  const dim3 grid(static_cast<unsigned>((a.rows + rpb - 1) / rpb), static_cast<unsigned>(a.kv_heads));
  if (a.k_scale != nullptr)
    attn_prefill_kernel<DK, true><<<grid, kThreads, smem, stream>>>(a);
  else
    attn_prefill_kernel<DK, false><<<grid, kThreads, smem, stream>>>(a);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace sink_prefill

size_t sink_attn_prefill_smem_bytes(int qk_dim) {
  if (qk_dim == 192) return sink_prefill::smem_bytes<192>();
  if (qk_dim == 128) return sink_prefill::smem_bytes<128>();
  throw std::invalid_argument("sink_attn_prefill: qk width " + std::to_string(qk_dim) + " is not compiled (192 or 128)");
}

int sink_attn_prefill_rows_per_block(int local_heads, int kv_heads) {
  const int hpk = local_heads / kv_heads;
  return sink_prefill::kQueryVecs / hpk;
}

void sink_attn_prefill(int qk_dim, const SinkAttnPrefillArgs& a, cudaStream_t stream) {
  if (qk_dim == 192)
    sink_prefill::launch<192>(a, stream);
  else if (qk_dim == 128)
    sink_prefill::launch<128>(a, stream);
  else
    throw std::invalid_argument("sink_attn_prefill: qk width " + std::to_string(qk_dim) + " is not compiled (192 or 128)");
}

}  // namespace dgpp
