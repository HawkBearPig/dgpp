#include "kernels/packq_head.hpp"
#include "kernels/pdl.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#include "common/cuda_check.hpp"
#include "kernels/packq_gemv.cuh"
#include "kernels/packq_planes.cuh"

namespace dgpp {

// ---------------------------------------------------------------------------
// The plane layout of one 128-code group (packq_head.hpp).
// ---------------------------------------------------------------------------
void packq_planes_permute_int8(uint8_t* payload, int64_t rows, int64_t cols) {
  if (payload == nullptr || rows <= 0 || cols <= 0 || cols % 128 != 0)
    throw std::invalid_argument("packq_planes_permute_int8: cols must be a positive multiple of 128");
  // Per ROW: [cols/2 B of high nibbles][cols/4 B of bits 3..2][cols/4 B of
  // bits 1..0] (2026-09-29, second layout: a per-128-code-group interleave
  // left the skipped plane inside the same DRAM bursts as the read ones
  // and saved no bytes at the memory; each plane is now a contiguous run).
  const size_t K = static_cast<size_t>(cols);
  auto permute_rows = [&](int64_t r0, int64_t r1) {
    std::vector<uint8_t> tmp(K);
    for (int64_t r = r0; r < r1; ++r) {
      uint8_t* row = payload + static_cast<size_t>(r) * K;
      std::memcpy(tmp.data(), row, K);
      std::memset(row, 0, K);
      uint8_t* nib = row;
      uint8_t* mid = row + K / 2;
      uint8_t* lo = row + K / 2 + K / 4;
      for (size_t j = 0; j < K; ++j) {
        const uint8_t c = tmp[j];
        nib[j / 2] |= static_cast<uint8_t>((c >> 4) << (4 * (j & 1)));
        const size_t chunk = j / 16;
        const int pos = packq_planes_pos(static_cast<int>(j % 16));
        mid[chunk * 4 + pos / 8] |= static_cast<uint8_t>(((c >> 2) & 3u) << (pos % 8));
        lo[chunk * 4 + pos / 8] |= static_cast<uint8_t>((c & 3u) << (pos % 8));
      }
    }
  };
  const int threads = static_cast<int>(std::min<int64_t>(16, rows));
  std::vector<std::thread> pool;
  const int64_t per = (rows + threads - 1) / threads;
  for (int t = 0; t < threads; ++t) {
    const int64_t r0 = t * per, r1 = std::min(rows, r0 + per);
    if (r0 < r1) pool.emplace_back(permute_rows, r0, r1);
  }
  for (auto& th : pool) th.join();
}

namespace {

using packq_gemv::Fmt;
using packq_gemv::Geom;
using packq_gemv::kChunkBytes;
using packq_gemv::kMaxChunksPerLane;
using packq_gemv::kSteps;
constexpr int kBits = 8;
constexpr int SF = packq_gemv::kScaleF16G128;
constexpr int kGroupCodes = 128;
using packq_planes::assemble;
using packq_planes::load_hi6;
using packq_planes::load_lo2;

// The six-bit pass's chunk: part[a] = sum_j (16 nb_j + 4 md_j - 126.5) x_j
// (the code with its low two bits at their centre 1.5, minus the zero
// point 128), acc[a] += s * part[a]; babs[a] += |s| * |x|_chunk[a]. The
// codes come out in pairs through the int4 core's fp16 trick: a nibble
// pair and its two-bit pair are each one shift-and-mask into a half2 with
// the 1024 bias, and v = 16 (1024 + nb) + 4 (1024 + md) - 20606.5 is
// exact in fp32.
template <int kRows>
__device__ __forceinline__ void consume_hi6(const uint2& nib, uint32_t mid, float s, const uint4 (&xv)[kRows][2],
                                            const float* __restrict__ sabs_chunk, float (&acc)[kRows],
                                            float (&babs)[kRows]) {
  float part[kRows];
#pragma unroll
  for (int a = 0; a < kRows; ++a) part[a] = 0.f;
#pragma unroll
  for (int w = 0; w < 2; ++w) {
    const uint32_t nw = w == 0 ? nib.x : nib.y;  // codes 8w .. 8w+7, nibble j at bits 4j
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // Codes 8w + j and 8w + j + 4.
      const uint32_t tn = ((nw >> (4 * j)) & 0x000F000Fu) | 0x64006400u;
      const uint32_t tm = ((mid >> (8 * w + 2 * j)) & 0x00030003u) | 0x64006400u;
      const float2 nf = __half22float2(packq_gemv::as_half2(tn));
      const float2 mf = __half22float2(packq_gemv::as_half2(tm));
      const float v0 = __fmaf_rn(16.f, nf.x, __fmaf_rn(4.f, mf.x, -20606.5f));
      const float v1 = __fmaf_rn(16.f, nf.y, __fmaf_rn(4.f, mf.y, -20606.5f));
#pragma unroll
      for (int a = 0; a < kRows; ++a) {
        const uint4& v = xv[a][w];
        const uint32_t xa = (j < 2) ? v.x : v.y;  // elements j / 2 pair
        const uint32_t xb = (j < 2) ? v.z : v.w;  // elements j / 2 + 2 pair
        const float x0 = (j & 1) ? packq_gemv::bf16_hi(xa) : packq_gemv::bf16_lo(xa);
        const float x1 = (j & 1) ? packq_gemv::bf16_hi(xb) : packq_gemv::bf16_lo(xb);
        part[a] = __fmaf_rn(v0, x0, part[a]);
        part[a] = __fmaf_rn(v1, x1, part[a]);
      }
    }
  }
  const float sa = fabsf(s);
#pragma unroll
  for (int a = 0; a < kRows; ++a) {
    acc[a] = __fmaf_rn(s, part[a], acc[a]);
    babs[a] = __fmaf_rn(sa, sabs_chunk[a], babs[a]);
  }
}

// One pass of a warp's row dots from the planes (packq_gemv::pass_row_dots
// with the chunk loads replaced): Full reads all three planes and runs
// consume_chunk<8> on the assembled words — the production chain, bitwise;
// Hi6 reads two planes into consume_hi6.
template <int K, int kRows, int C0, int NC, bool Full>
__device__ __forceinline__ void pass_planes(const uint8_t* __restrict__ w, const uint16_t* __restrict__ scales,
                                            const uint16_t* __restrict__ sx, const float* __restrict__ sabs,
                                            int row_chunks, int n0, int n, float (&acc)[kSteps][kRows],
                                            float (&babs)[kSteps][kRows]) {
  using G = Geom<kBits, K, SF>;
  using F = Fmt<kBits, SF>;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int group = lane / G::lanes_per_row;
  const int lig = lane % G::lanes_per_row;
  const int row_base = n0 + warp * G::rows_per_warp;
  uint2 nb[kSteps * NC];
  uint32_t md[kSteps * NC], lw[kSteps * NC];
  uint16_t sv[kSteps * NC];
#pragma unroll
  for (int st = 0; st < kSteps; ++st) {
    const int row = row_base + st * G::rows_per_step + group;
    const bool ok = row < n;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      const int ci = (C0 + c) * G::lanes_per_row + lig;
      const int i = st * NC + c;
      lw[i] = 0u;
      if (ok) {
        load_hi6(w, row, K, ci, nb[i], md[i]);
        if constexpr (Full) lw[i] = load_lo2(w, row, K, ci);
        sv[i] = scales[static_cast<size_t>(row) * G::scale_cols + ci / F::chunks_per_group];
      } else {
        nb[i] = make_uint2(0u, 0u);
        md[i] = 0u;
        lw[i] = 0u;
        sv[i] = 0;
      }
    }
  }
#pragma unroll
  for (int c = 0; c < NC; ++c) {
    const int ci = (C0 + c) * G::lanes_per_row + lig;
    const int e0 = ci * F::codes_per_chunk;
    uint4 xv[kRows][F::window_vecs];
    packq_gemv::load_window<kBits, kRows>(sx, K, e0, xv);
#pragma unroll
    for (int st = 0; st < kSteps; ++st) {
      const int row = row_base + st * G::rows_per_step + group;
      if (row >= n) continue;
      const int i = st * NC + c;
      const float s = packq_gemv::ScaleFmt<SF>::to_float(sv[i]);
      if constexpr (Full) {
        packq_gemv::consume_chunk<kBits, kRows>(assemble(nb[i], md[i], lw[i]), s, xv, acc[st]);
      } else {
        consume_hi6<kRows>(nb[i], md[i], s, xv, sabs + static_cast<size_t>(ci) * kRows, acc[st], babs[st]);
      }
    }
  }
  (void)row_chunks;
}

template <int K, int kRows, int P, bool Full>
__device__ __forceinline__ void chain_planes(const uint8_t* __restrict__ w, const uint16_t* __restrict__ scales,
                                             const uint16_t* __restrict__ sx, const float* __restrict__ sabs,
                                             int n0, int n, float (&acc)[kSteps][kRows],
                                             float (&babs)[kSteps][kRows]) {
  using G = Geom<kBits, K, SF>;
  if constexpr (P < G::passes) {
    pass_planes<K, kRows, P * kMaxChunksPerLane, G::pass_chunks(P), Full>(w, scales, sx, sabs, G::row_chunks, n0, n,
                                                                          acc, babs);
    chain_planes<K, kRows, P + 1, Full>(w, scales, sx, sabs, n0, n, acc, babs);
  }
}

template <int K, int kRows, bool Full>
__device__ __forceinline__ void warp_dots_planes(const uint8_t* __restrict__ w, const uint16_t* __restrict__ scales,
                                                 const uint16_t* __restrict__ sx, const float* __restrict__ sabs,
                                                 int n0, int n, float (&acc)[kSteps][kRows],
                                                 float (&babs)[kSteps][kRows]) {
  using G = Geom<kBits, K, SF>;
#pragma unroll
  for (int st = 0; st < kSteps; ++st)
#pragma unroll
    for (int a = 0; a < kRows; ++a) acc[st][a] = babs[st][a] = 0.f;
  chain_planes<K, kRows, 0, Full>(w, scales, sx, sabs, n0, n, acc, babs);
#pragma unroll
  for (int st = 0; st < kSteps; ++st) {
    packq_gemv::group_reduce<kRows, G::lanes_per_row>(acc[st]);
    if constexpr (!Full) packq_gemv::group_reduce<kRows, G::lanes_per_row>(babs[st]);
  }
}

// ---------------------------------------------------------------------------
// The mode, the ordered-int max, the scratch layout.
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool rows_all_greedy(const PackqHeadMode& mode, int row0, int rows) {
  if (mode.greedy == nullptr) return false;
  for (int r = 0; r < rows; ++r) {
    const int row = row0 + r;
    const int slot = mode.request_map != nullptr ? mode.request_map[row / mode.rows_per_request] : mode.req0;
    if (slot < 0 || mode.greedy[slot] == 0) return false;
  }
  return true;
}
__device__ __forceinline__ int32_t ordered_of(float f) {
  const int32_t i = __float_as_int(f);
  return i >= 0 ? i : static_cast<int32_t>(static_cast<uint32_t>(i) ^ 0x7FFFFFFFu);
}
__device__ __forceinline__ float float_of_ordered(int32_t i) {
  return __int_as_float(i >= 0 ? i : static_cast<int32_t>(static_cast<uint32_t>(i) ^ 0x7FFFFFFFu));
}
constexpr float kBoundSlack = 1.0e-3f;   // fp32 rounding of A and of the exact chain, x10 margin
constexpr float kBoundEps = 1.0e-6f;

__global__ void head_prepare_kernel(int32_t* __restrict__ lo, int m, int32_t* __restrict__ cand_count) {
  dgpp_pdl_wait();
  if (threadIdx.x < static_cast<unsigned>(m)) lo[threadIdx.x] = ordered_of(-INFINITY);
  if (threadIdx.x == 0) *cand_count = 0;
}
// The output column of head row `row`: itself, or its id in the wider row.
struct OutMap {
  const int32_t* ids;
  int out_n;
  __device__ __forceinline__ size_t at(int a, int row) const {
    return static_cast<size_t>(a) * out_n + (ids != nullptr ? ids[row] : row);
  }
};
__global__ void head_fill_kernel(float* __restrict__ out, size_t elems) {
  dgpp_pdl_wait();
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < elems; i += stride)
    out[i] = -INFINITY;
}

// The staged |x| per chunk per activation row: sabs[ci * kRows + a].
template <int K, int kRows>
__device__ __forceinline__ void stage_abs(const uint16_t* __restrict__ sx, float* __restrict__ sabs) {
  constexpr int kRowChunks = Geom<kBits, K, SF>::row_chunks;
  for (int i = threadIdx.x; i < kRowChunks * kRows; i += blockDim.x) {
    const int ci = i / kRows, a = i - ci * kRows;
    const uint16_t* x = sx + static_cast<size_t>(a) * K + ci * 16;
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) s += fabsf(bf16_bits_to_float(x[j]));
    sabs[i] = s;
  }
}

// Pass 1: the greedy chunk's bounds (ab = A + B, lo = max(A - B) per row),
// or the full exact product when any row is sampled.
template <int K, int kRows>
__global__ void __launch_bounds__(packq_gemv::kThreads)
    head_pass1_kernel(const uint16_t* __restrict__ act, size_t act_stride, const uint8_t* __restrict__ w,
                      const uint16_t* __restrict__ scales, int n, PackqHeadMode mode, int row0,
                      float* __restrict__ ab, int32_t* __restrict__ lo, float* __restrict__ out, OutMap om) {
  dgpp_pdl_wait();
  using G = Geom<kBits, K, SF>;
  extern __shared__ __align__(16) uint16_t sx[];
  float* sabs = reinterpret_cast<float*>(sx + static_cast<size_t>(kRows) * K);
  __shared__ float wmax[packq_gemv::kWarps][kRows];
  const bool greedy = rows_all_greedy(mode, row0, kRows);
  packq_gemv::stage_activations<kRows>(act, act_stride, K, sx);
  __syncthreads();
  if (greedy) {
    stage_abs<K, kRows>(sx, sabs);
    __syncthreads();
  }
  const int n0 = blockIdx.x * G::rows_per_block;
  float acc[kSteps][kRows], babs[kSteps][kRows];
  if (!greedy) {
    warp_dots_planes<K, kRows, true>(w, scales, sx, sabs, n0, n, acc, babs);
#pragma unroll
    for (int st = 0; st < kSteps; ++st) {
      bool mine = false;
      const int row = packq_gemv::owned_row<kBits, K, SF>(n0, st, mine);
      if (mine && row < n)
#pragma unroll
        for (int a = 0; a < kRows; ++a) out[om.at(a, row)] = acc[st][a];
    }
    return;
  }
  warp_dots_planes<K, kRows, false>(w, scales, sx, sabs, n0, n, acc, babs);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  float lower[kRows];
#pragma unroll
  for (int a = 0; a < kRows; ++a) lower[a] = -INFINITY;
#pragma unroll
  for (int st = 0; st < kSteps; ++st) {
    bool mine = false;
    const int row = packq_gemv::owned_row<kBits, K, SF>(n0, st, mine);
    if (mine && row < n) {
#pragma unroll
      for (int a = 0; a < kRows; ++a) {
        const float A = acc[st][a];
        float B = 1.5f * babs[st][a];
        B = __fmaf_rn(kBoundSlack, B + fabsf(A), B) + kBoundEps;
        ab[static_cast<size_t>(a) * n + row] = A + B;
        lower[a] = fmaxf(lower[a], A - B);
      }
    }
  }
#pragma unroll
  for (int a = 0; a < kRows; ++a) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) lower[a] = fmaxf(lower[a], __shfl_xor_sync(0xFFFFFFFFu, lower[a], off));
    if (lane == 0) wmax[warp][a] = lower[a];
  }
  __syncthreads();
  if (threadIdx.x < static_cast<unsigned>(kRows)) {
    float v = -INFINITY;
    for (int wi = 0; wi < packq_gemv::kWarps; ++wi) v = fmaxf(v, wmax[wi][threadIdx.x]);
    if (v != -INFINITY) atomicMax(lo + threadIdx.x, ordered_of(v));
  }
}

// Pass 2 (greedy chunks only): every row's logits are -inf unless its
// upper bound reaches the best lower bound of some activation row, in
// which case the row runs the exact chain from all three planes.
template <int K, int kRows>
__global__ void __launch_bounds__(packq_gemv::kThreads)
    head_pass2_kernel(const uint16_t* __restrict__ act, size_t act_stride, const uint8_t* __restrict__ w,
                      const uint16_t* __restrict__ scales, int n, PackqHeadMode mode, int row0,
                      const float* __restrict__ ab, const int32_t* __restrict__ lo, float* __restrict__ out,
                      OutMap om) {
  dgpp_pdl_wait();
  using G = Geom<kBits, K, SF>;
  extern __shared__ __align__(16) uint16_t sx[];
  __shared__ float slo[kRows];
  if (!rows_all_greedy(mode, row0, kRows)) return;
  if (threadIdx.x < static_cast<unsigned>(kRows)) slo[threadIdx.x] = float_of_ordered(lo[threadIdx.x]);
  __syncthreads();
  const int n0 = blockIdx.x * G::rows_per_block;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int group = lane / G::lanes_per_row;
  // Each lane group's row: a candidate when any activation row's bound
  // admits it. Candidacy is decided BEFORE the activations are staged: a
  // block without a candidate row writes -inf and leaves without the
  // kRows x K staging read (31k blocks x 20 KB would be the head's own
  // bytes again, out of L2).
  bool cand = false;
#pragma unroll
  for (int st = 0; st < kSteps; ++st) {
    const int row = n0 + warp * G::rows_per_warp + st * G::rows_per_step + group;
    if (row < n)
#pragma unroll
      for (int a = 0; a < kRows; ++a) cand = cand || ab[static_cast<size_t>(a) * n + row] >= slo[a];
  }
  const bool block_any = __syncthreads_or(cand ? 1 : 0) != 0;
  const bool any = __any_sync(0xFFFFFFFFu, cand);
  if (!any) {
#pragma unroll
    for (int st = 0; st < kSteps; ++st) {
      bool mine = false;
      const int row = packq_gemv::owned_row<kBits, K, SF>(n0, st, mine);
      if (mine && row < n)
#pragma unroll
        for (int a = 0; a < kRows; ++a) out[om.at(a, row)] = -INFINITY;
    }
    if (!block_any) return;
  }
  packq_gemv::stage_activations<kRows>(act, act_stride, K, sx);
  __syncthreads();
  if (!any) return;
  float acc[kSteps][kRows], babs[kSteps][kRows];
  warp_dots_planes<K, kRows, true>(w, scales, sx, nullptr, n0, n, acc, babs);
#pragma unroll
  for (int st = 0; st < kSteps; ++st) {
    bool mine = false;
    const int row = packq_gemv::owned_row<kBits, K, SF>(n0, st, mine);
    if (mine && row < n) {
      // The lane group's own candidacy decides this row (a warp may hold
      // several rows when lanes_per_row < 32).
      bool c = false;
#pragma unroll
      for (int a = 0; a < kRows; ++a) c = c || ab[static_cast<size_t>(a) * n + row] >= slo[a];
#pragma unroll
      for (int a = 0; a < kRows; ++a) out[om.at(a, row)] = c ? acc[st][a] : -INFINITY;
    }
  }
}

// Pass 2a (greedy chunks): one thread per head row — the coalesced
// candidacy test against every activation row's best lower bound, -inf
// into the non-candidates' output columns, the candidates compacted into
// a list (one atomic per warp).
template <int kRows>
__global__ void __launch_bounds__(256)
    head_candidates_kernel(int n, PackqHeadMode mode, int row0, const float* __restrict__ ab,
                           const int32_t* __restrict__ lo, float* __restrict__ out, OutMap om,
                           int32_t* __restrict__ cand, int32_t* __restrict__ cand_count) {
  dgpp_pdl_wait();
  if (!rows_all_greedy(mode, row0, kRows)) return;
  const int row = static_cast<int>(blockIdx.x) * 256 + threadIdx.x;
  bool c = false;
  if (row < n) {
#pragma unroll
    for (int a = 0; a < kRows; ++a) c = c || ab[static_cast<size_t>(a) * n + row] >= float_of_ordered(lo[a]);
    if (!c)
#pragma unroll
      for (int a = 0; a < kRows; ++a) out[om.at(a, row)] = -INFINITY;
  }
  const unsigned mask = __ballot_sync(0xFFFFFFFFu, c);
  const int lane = threadIdx.x % 32;
  int base = 0;
  if (lane == 0 && mask != 0u) base = atomicAdd(cand_count, __popc(mask));
  base = __shfl_sync(0xFFFFFFFFu, base, 0);
  if (c) cand[base + __popc(mask & ((1u << lane) - 1u))] = row;
}

// Pass 2b (greedy chunks): the exact chain over the candidate list, a
// warp per row (the head's K keeps a row in one warp), a fixed grid
// striding over the list.
template <int K, int kRows>
__global__ void __launch_bounds__(packq_gemv::kThreads)
    head_exact_list_kernel(const uint16_t* __restrict__ act, size_t act_stride, const uint8_t* __restrict__ w,
                           const uint16_t* __restrict__ scales, int n, PackqHeadMode mode, int row0,
                           const int32_t* __restrict__ cand, const int32_t* __restrict__ cand_count,
                           float* __restrict__ out, OutMap om) {
  dgpp_pdl_wait();
  using G = Geom<kBits, K, SF>;
  static_assert(G::rows_per_warp == 1, "the list form needs a row per warp");
  extern __shared__ __align__(16) uint16_t sx[];
  if (!rows_all_greedy(mode, row0, kRows)) return;
  const int total = *cand_count;
  const int warp = threadIdx.x / 32;
  const int gw = static_cast<int>(blockIdx.x) * packq_gemv::kWarps + warp;
  const int stride = static_cast<int>(gridDim.x) * packq_gemv::kWarps;
  if (static_cast<int>(blockIdx.x) * packq_gemv::kWarps >= total) return;  // the whole block idle
  packq_gemv::stage_activations<kRows>(act, act_stride, K, sx);
  __syncthreads();
  for (int i = gw; i < total; i += stride) {
    const int row = cand[i];
    float acc[kSteps][kRows], babs[kSteps][kRows];
    // n0 such that this warp's row_base is `row`: the chain then runs on it alone.
    warp_dots_planes<K, kRows, true>(w, scales, sx, nullptr, row - warp * G::rows_per_warp, row + 1, acc, babs);
    if (threadIdx.x % 32 == 0)
#pragma unroll
      for (int a = 0; a < kRows; ++a) out[om.at(a, row)] = acc[0][a];
  }
}

template <int K, int kRows>
void launch_chunk(const uint16_t* act, size_t act_stride, const uint8_t* w, const uint16_t* scales, float* out,
                  int n, const PackqHeadMode& mode, int row0, const PackqHeadScratch& scratch,
                  cudaStream_t stream, OutMap om) {
  using G = Geom<kBits, K, SF>;
  const dim3 grid((n + G::rows_per_block - 1) / G::rows_per_block);
  const size_t smem = packq_gemv::smem_bytes(kRows, K) + sizeof(float) * static_cast<size_t>(G::row_chunks) * kRows;
  static bool opted1 = false, opted2 = false;
  if (smem > gemv::kMaxSmemBytes && !opted1) {
    DGPP_CUDA_OK(cudaFuncSetAttribute(head_pass1_kernel<K, kRows>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem)));
    opted1 = true;
  }
  if (smem > gemv::kMaxSmemBytes && !opted2) {
    DGPP_CUDA_OK(cudaFuncSetAttribute(head_pass2_kernel<K, kRows>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem)));
    opted2 = true;
  }
  if (om.ids != nullptr) {
    // The scattered form: every column of the chunk's output rows -inf
    // first, the slice's logits over them.
    const size_t elems = static_cast<size_t>(kRows) * om.out_n;
    head_fill_kernel<<<static_cast<unsigned>(std::min<size_t>(1024, (elems + 255) / 256)), 256, 0, stream>>>(out, elems);
    DGPP_CUDA_OK(cudaGetLastError());
  }
  head_prepare_kernel<<<1, 32, 0, stream>>>(scratch.lo, kRows, scratch.cand_count);
  DGPP_CUDA_OK(cudaGetLastError());
  head_pass1_kernel<K, kRows><<<grid, packq_gemv::kThreads, smem, stream>>>(act, act_stride, w, scales, n, mode, row0,
                                                                           scratch.ab, scratch.lo, out, om);
  DGPP_CUDA_OK(cudaGetLastError());
  if constexpr (G::rows_per_warp == 1) {
    head_candidates_kernel<kRows><<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
        n, mode, row0, scratch.ab, scratch.lo, out, om, scratch.cand, scratch.cand_count);
    DGPP_CUDA_OK(cudaGetLastError());
    static bool opted3 = false;
    if (smem > gemv::kMaxSmemBytes && !opted3) {
      DGPP_CUDA_OK(cudaFuncSetAttribute(head_exact_list_kernel<K, kRows>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smem)));
      opted3 = true;
    }
    head_exact_list_kernel<K, kRows><<<512, packq_gemv::kThreads, smem, stream>>>(
        act, act_stride, w, scales, n, mode, row0, scratch.cand, scratch.cand_count, out, om);
    DGPP_CUDA_OK(cudaGetLastError());
  } else {
    head_pass2_kernel<K, kRows><<<grid, packq_gemv::kThreads, smem, stream>>>(act, act_stride, w, scales, n, mode, row0,
                                                                             scratch.ab, scratch.lo, out, om);
    DGPP_CUDA_OK(cudaGetLastError());
  }
}

template <int kRows>
void launch_rows(const uint16_t* act, size_t act_stride, const GlmPackedMatrix& w, float* out, int n, int k,
                 const PackqHeadMode& mode, int row0, const PackqHeadScratch& scratch, cudaStream_t stream,
                 OutMap om) {
  packq_gemv::dispatch_k(k, [&](auto kc) {
    constexpr int K = decltype(kc)::value;
    if constexpr (packq_gemv::k_supported<kBits, SF>(K) && packq_gemv::sf_compiled(SF, K) && K % kGroupCodes == 0) {
      launch_chunk<K, kRows>(act, act_stride, reinterpret_cast<const uint8_t*>(w.packed), w.scales, out, n, mode,
                             row0, scratch, stream, om);
    } else {
      throw std::invalid_argument("packq_head: K is outside the plane layout's compiled set");
    }
  });
}

}  // namespace

void launch_packq_head_f32(const uint16_t* act, size_t act_stride, const GlmPackedMatrix& w, float* out, int m,
                           int n, int k, const PackqHeadMode& mode, const PackqHeadScratch& scratch,
                           cudaStream_t stream, const int32_t* out_ids, int out_n) {
  if (m <= 0 || n <= 0) return;
  if (out_ids != nullptr && out_n < n) throw std::invalid_argument("packq_head: out_n below the slice's rows");
  const OutMap om{out_ids, out_ids != nullptr ? out_n : n};
  if (!act || !w.packed || !w.scales || !out || !scratch.ab || !scratch.lo || !scratch.cand || !scratch.cand_count)
    throw std::invalid_argument("packq_head: null pointer");
  if (w.bits != 8 || w.scale_fmt != kPackedScaleF16G128 || w.layout != kPackedLayoutPlanes8)
    throw std::invalid_argument("packq_head: the head must be int8 g128 in the plane layout");
  if (w.rows < n || w.cols != k || k % kGroupCodes != 0)
    throw std::invalid_argument("packq_head: matrix geometry does not match n, k");
  if (!packq_gemv::shape_ok(w.packed, w.bits, k, w.scale_fmt))
    throw std::invalid_argument("packq_head: K outside the packed core's contract or a misaligned payload");
  if (mode.greedy != nullptr && mode.rows_per_request < 1)
    throw std::invalid_argument("packq_head: rows per request");
  for (int row0 = 0; row0 < m;) {
    int rows = std::min(packq_gemv::kMaxRows, m - row0);
    while (!gemv::smem_fits(rows, k)) --rows;
    const uint16_t* a = act + static_cast<size_t>(row0) * act_stride;
    float* o = out + static_cast<size_t>(row0) * om.out_n;
    switch (rows) {
      case 4: launch_rows<4>(a, act_stride, w, o, n, k, mode, row0, scratch, stream, om); break;
      case 3: launch_rows<3>(a, act_stride, w, o, n, k, mode, row0, scratch, stream, om); break;
      case 2: launch_rows<2>(a, act_stride, w, o, n, k, mode, row0, scratch, stream, om); break;
      default: launch_rows<1>(a, act_stride, w, o, n, k, mode, row0, scratch, stream, om); break;
    }
    row0 += rows;
  }
}

}  // namespace dgpp
