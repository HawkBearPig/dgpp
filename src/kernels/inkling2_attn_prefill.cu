#include <cmath>
#include <stdexcept>
#include <string>

#include <cuda_bf16.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/inkling2_attn.hpp"
#include "kernels/pdl.cuh"

// The query-tiled prefill attention of the Inkling 2 family: the sink
// family's tensor-core form (kernels/sink_attn_prefill.cu) with this
// family's three additions.
//
// A block owns a tile of consecutive query rows x one key/value head's query
// heads, stages each 32-token K/V tile once for all of them, and runs the
// scores and the value products on the tensor cores:
//
//   S = Q K^T   mma.m16n8k16 bf16 x bf16 -> fp32, x scale, + the row's
//               relative bias at the key's distance
//   softmax     online per query vector over the tiles, the row's sink as an
//               extra column that carries no value, the probabilities
//               rounded to bf16 for the value product
//   O += P V    mma bf16 x bf16 -> fp32
//   out = bf16(O / L * sigmoid(gate))
//
// Three query heads share a key/value head here, so a warp's 16-row mma tile
// holds five rows' three heads and leaves its last row idle: a block is 40
// query rows, 120 live vectors. The relative bias is a table per (row, head)
// over the distances [0, extent): a key inside the extent adds its entry
// after the scale. The sink is the row's own value per head. The output
// gate rides the epilogue, so a prefill row needs no combine pass.
//
// The summation order is the tensor core's, not the split-KV decode
// kernel's fixed fp32 chain: the terms are the same exact products and a
// prefill row is the decode row at accumulation tolerance, not bit for bit.
// A block whose rows span several requests loops over the runs of one
// request with the other rows inert.
namespace dgpp {
namespace ink2_prefill {

constexpr int D = kInk2HeadDim;          // 128: the query/key and value width
constexpr int kTile = kInk2AttnTile;     // 32 keys per staged tile
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kQueryVecs = kWarps * 16;  // 128 mma rows per block
constexpr int kQStride = D + 8;          // 136 bf16 = 68 words: (68 g + t) mod 32 distinct for g < 8, t < 4
constexpr int kKStride = D + 8;
constexpr int kVtStride = kTile + 8;     // 40 bf16 = 20 words
constexpr int kKStepsS = D / 16;         // 8 k-steps over the key width
constexpr int kNTilesS = kTile / 8;      // 4 n-tiles of 8 keys
constexpr int kNTilesO = D / 8;          // 16 n-tiles of 8 value dims
constexpr int kKStepsO = kTile / 16;     // 2 k-steps over the tile's keys

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

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void __launch_bounds__(kThreads) attn_prefill_kernel(Ink2AttnPrefillArgs a) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint16_t smem[];
  uint16_t* qt = smem;                        // [kQueryVecs][kQStride]
  uint16_t* kt = qt + kQueryVecs * kQStride;  // [kTile][kKStride]
  uint16_t* vt = kt + kTile * kKStride;       // [D][kVtStride] transposed
  const int hpk = a.local_heads / a.kv_heads;
  const int rows_per_warp = 16 / hpk;        // whole rows in a warp's m-tile
  const int live = rows_per_warp * hpk;      // the m-tile's live rows
  const int rows_per_block = kWarps * rows_per_warp;
  const int r0 = static_cast<int>(blockIdx.x) * rows_per_block;
  const int kvh = static_cast<int>(blockIdx.y);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int g = lane / 4, t = lane % 4;
  const int width = a.kv_heads * D;

  // The block's rows: (request, position). Rows past the batch are padding.
  __shared__ int32_t s_req[kQueryVecs];
  __shared__ int64_t s_pos[kQueryVecs];
  for (int i = threadIdx.x; i < rows_per_block; i += kThreads) {
    const int r = r0 + i;
    s_req[i] = r < a.rows ? a.req_ids[r] : -1;
    s_pos[i] = r < a.rows ? a.pos[r] : -1;
  }
  // The query vectors: mma row lv of warp w is row w * rows_per_warp +
  // lv / hpk, head lv % hpk of this key/value head. The idle rows are zero.
  for (int i = threadIdx.x; i < kQueryVecs * (D / 8); i += kThreads) {
    const int qv = i / (D / 8), c8 = i - qv * (D / 8);
    const int w = qv / 16, lv = qv - w * 16;
    const int r = r0 + w * rows_per_warp + lv / hpk;
    const int h = kvh * hpk + lv % hpk;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (lv < live && r < a.rows && a.pos[r] >= 0)
      v = *reinterpret_cast<const uint4*>(a.q + static_cast<int64_t>(r) * a.q_stride + static_cast<int64_t>(h) * D + c8 * 8);
    *reinterpret_cast<uint4*>(qt + qv * kQStride + c8 * 8) = v;
  }
  __syncthreads();

  // This lane's two query vectors (mma rows g and g + 8 of the warp's m-tile).
  const int lv0 = g, lv1 = g + 8;
  const bool live0 = lv0 < live, live1 = lv1 < live;
  const int ri0 = warp * rows_per_warp + (live0 ? lv0 / hpk : 0);
  const int ri1 = warp * rows_per_warp + (live1 ? lv1 / hpk : 0);
  const int h0 = kvh * hpk + lv0 % hpk, h1 = kvh * hpk + lv1 % hpk;
  const int qv0 = warp * 16 + lv0, qv1 = warp * 16 + lv1;
  const int64_t p0 = live0 ? s_pos[ri0] : -1, p1 = live1 ? s_pos[ri1] : -1;
  const int64_t lo0 = a.window > 0 ? (p0 - a.window + 1 > 0 ? p0 - a.window + 1 : 0) : 0;
  const int64_t lo1 = a.window > 0 ? (p1 - a.window + 1 > 0 ? p1 - a.window + 1 : 0) : 0;
  const int64_t gr0 = r0 + ri0, gr1 = r0 + ri1;
  const uint16_t* bias0 = a.bias + (gr0 * a.local_heads + h0) * a.extent;
  const uint16_t* bias1 = a.bias + (gr1 * a.local_heads + h1) * a.extent;
  // The sink: a column of logit sink that carries no value.
  float m0 = -INFINITY, l0 = 0.f, m1 = -INFINITY, l1 = 0.f;
  if (p0 >= 0) {
    m0 = bf16_bits_to_float(a.sink[gr0 * a.sink_stride + h0]);
    l0 = 1.f;
  }
  if (p1 >= 0) {
    m1 = bf16_bits_to_float(a.sink[gr1 * a.sink_stride + h1]);
    l1 = 1.f;
  }
  float o[kNTilesO][4];
#pragma unroll
  for (int j = 0; j < kNTilesO; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;

  // The runs of one request among the block's rows (consecutive rows at
  // consecutive positions): each run's union range, tile by tile.
  int run0 = 0;
  while (run0 < rows_per_block) {
    const int32_t req = s_req[run0];
    if (req < 0 || s_pos[run0] < 0) {
      ++run0;
      continue;
    }
    int run1 = run0 + 1;
    while (run1 < rows_per_block && s_req[run1] == req && s_pos[run1] == s_pos[run1 - 1] + 1) ++run1;
    const int64_t first = s_pos[run0], last = s_pos[run1 - 1];
    const int64_t t_lo = a.window > 0 ? (first - a.window + 1 > 0 ? first - a.window + 1 : 0) : 0;
    const int32_t* bt = a.block_tables + static_cast<int64_t>(req) * a.blocks_per_request;
    const bool in0 = live0 && ri0 >= run0 && ri0 < run1, in1 = live1 && ri1 >= run0 && ri1 < run1;
    for (int64_t t0 = t_lo; t0 <= last; t0 += kTile) {
      const int n = last + 1 - t0 < kTile ? static_cast<int>(last + 1 - t0) : kTile;
      __syncthreads();  // the previous tile's readers are done
      for (int idx = threadIdx.x; idx < n * (D / 8); idx += kThreads) {
        const int tt = idx / (D / 8), c8 = idx - tt * (D / 8);
        const int64_t tok = t0 + tt;
        const int64_t phys = static_cast<int64_t>(bt[tok / a.block_tokens]) * a.block_tokens + tok % a.block_tokens;
        const uint4 kv = *reinterpret_cast<const uint4*>(a.k_cache + phys * width + kvh * D + c8 * 8);
        *reinterpret_cast<uint4*>(kt + tt * kKStride + c8 * 8) = kv;
        const uint4 vv = *reinterpret_cast<const uint4*>(a.v_cache + phys * width + kvh * D + c8 * 8);
        const uint32_t words[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          vt[(c8 * 8 + 2 * w) * kVtStride + tt] = static_cast<uint16_t>(words[w] & 0xFFFFu);
          vt[(c8 * 8 + 2 * w + 1) * kVtStride + tt] = static_cast<uint16_t>(words[w] >> 16);
        }
      }
      // Keys past n: zero operands (their scores are masked below, but the
      // products must be finite).
      for (int idx = threadIdx.x; idx < (kTile - n) * (D / 8); idx += kThreads) {
        const int tt = n + idx / (D / 8), c8 = idx - (tt - n) * (D / 8);
        *reinterpret_cast<uint4*>(kt + tt * kKStride + c8 * 8) = make_uint4(0u, 0u, 0u, 0u);
      }
      for (int idx = threadIdx.x; idx < (kTile - n) * D; idx += kThreads) {
        const int tt = n + idx / D, d = idx - (tt - n) * D;
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
      // Scale, the relative bias, the mask (causal, window, the run,
      // padding), the row statistics.
      float tmax0 = -INFINITY, tmax1 = -INFINITY;
#pragma unroll
      for (int j = 0; j < kNTilesS; ++j) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const int64_t key = t0 + j * 8 + 2 * t + c;
          const bool ok0 = in0 && key < n + t0 && key >= lo0 && key <= p0;
          const bool ok1 = in1 && key < n + t0 && key >= lo1 && key <= p1;
          float v0 = -INFINITY, v1 = -INFINITY;
          if (ok0) {
            const int64_t dist = p0 - key;
            v0 = s[j][c] * a.scale;
            if (dist < a.extent) v0 += bf16_bits_to_float(bias0[dist]);
          }
          if (ok1) {
            const int64_t dist = p1 - key;
            v1 = s[j][2 + c] * a.scale;
            if (dist < a.extent) v1 += bf16_bits_to_float(bias1[dist]);
          }
          s[j][c] = v0;
          s[j][2 + c] = v1;
          tmax0 = fmaxf(tmax0, v0);
          tmax1 = fmaxf(tmax1, v1);
        }
      }
#pragma unroll
      for (int off = 1; off <= 2; off <<= 1) {
        tmax0 = fmaxf(tmax0, __shfl_xor_sync(~0u, tmax0, off));
        tmax1 = fmaxf(tmax1, __shfl_xor_sync(~0u, tmax1, off));
      }
      const float mn0 = fmaxf(m0, tmax0), mn1 = fmaxf(m1, tmax1);
      // The rescale: 1 when nothing has arrived yet (an inert row).
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
        // The A fragment of PV from the C fragment of S: n-tile j is k-step
        // j / 2, half j % 2 (rows g: a0 / a2, rows g + 8: a1 / a3).
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
  // out[(row, head)][dims] through the head's gate: rows g and g + 8 of the
  // m-tile, dims 8 j + 2 t and + 1.
  if (live0 && gr0 < a.rows) {
    uint16_t* dst = a.out + (gr0 * a.local_heads + h0) * D;
    const uint16_t* gate = a.gate + gr0 * a.gate_stride + static_cast<int64_t>(h0) * D;
#pragma unroll
    for (int j = 0; j < kNTilesO; ++j) {
      const float ga = sigmoid(bf16_bits_to_float(gate[j * 8 + 2 * t]));
      const float gb = sigmoid(bf16_bits_to_float(gate[j * 8 + 2 * t + 1]));
      *reinterpret_cast<uint32_t*>(dst + j * 8 + 2 * t) =
          pack_bf16x2(l0 > 0.f ? (o[j][0] / l0) * ga : 0.f, l0 > 0.f ? (o[j][1] / l0) * gb : 0.f);
    }
  }
  if (live1 && gr1 < a.rows) {
    uint16_t* dst = a.out + (gr1 * a.local_heads + h1) * D;
    const uint16_t* gate = a.gate + gr1 * a.gate_stride + static_cast<int64_t>(h1) * D;
#pragma unroll
    for (int j = 0; j < kNTilesO; ++j) {
      const float ga = sigmoid(bf16_bits_to_float(gate[j * 8 + 2 * t]));
      const float gb = sigmoid(bf16_bits_to_float(gate[j * 8 + 2 * t + 1]));
      *reinterpret_cast<uint32_t*>(dst + j * 8 + 2 * t) =
          pack_bf16x2(l1 > 0.f ? (o[j][2] / l1) * ga : 0.f, l1 > 0.f ? (o[j][3] / l1) * gb : 0.f);
    }
  }
}

constexpr size_t kSmemBytes = static_cast<size_t>(kQueryVecs) * kQStride * 2 + static_cast<size_t>(kTile) * kKStride * 2 +
                              static_cast<size_t>(D) * kVtStride * 2;

}  // namespace ink2_prefill

int ink2_attn_prefill_rows_per_block(int local_heads, int kv_heads) {
  const int hpk = local_heads / kv_heads;
  return ink2_prefill::kWarps * (16 / hpk);
}

void ink2_attn_prefill(const Ink2AttnPrefillArgs& a, cudaStream_t stream) {
  using namespace ink2_prefill;
  if (a.rows <= 0) return;
  if (!a.q || !a.bias || !a.sink || !a.gate || !a.k_cache || !a.v_cache || !a.req_ids || !a.pos || !a.block_tables || !a.out)
    throw std::invalid_argument("ink2_attn_prefill: null pointer");
  if (a.local_heads <= 0 || a.kv_heads <= 0 || a.local_heads % a.kv_heads != 0)
    throw std::invalid_argument("ink2_attn_prefill: heads");
  const int hpk = a.local_heads / a.kv_heads;
  if (hpk > 16) throw std::invalid_argument("ink2_attn_prefill: at most 16 query heads per key/value head");
  if (a.block_tokens <= 0 || a.block_tokens % kTile != 0)
    throw std::invalid_argument("ink2_attn_prefill: block_tokens must be a multiple of the 32-token tile");
  if (a.window < 0 || a.extent <= 0) throw std::invalid_argument("ink2_attn_prefill: window / extent");
  if ((a.q_stride * 2) % 16 != 0) throw std::invalid_argument("ink2_attn_prefill: q rows must be 16-byte aligned");
  // The kernel's own attribute, set at its first launch.
  static const bool attr_ok = [] {
    return cudaFuncSetAttribute(attn_prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(kSmemBytes)) == cudaSuccess;
  }();
  if (!attr_ok) throw std::runtime_error("ink2_attn_prefill: the shared-memory opt-in was refused");
  const int rpb = kWarps * (16 / hpk);
  const dim3 grid(static_cast<unsigned>((a.rows + rpb - 1) / rpb), static_cast<unsigned>(a.kv_heads));
  attn_prefill_kernel<<<grid, kThreads, kSmemBytes, stream>>>(a);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
