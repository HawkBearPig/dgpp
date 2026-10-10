#include "kernels/qsa.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

#include <cuda_bf16.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/latent_format.hpp"
#include "kernels/topk_select.cuh"

namespace dgpp {
namespace {

__device__ __forceinline__ float round_bf16(float v) {
  return bf16_bits_to_float(float_to_bf16_bits(v));
}
__device__ __forceinline__ float sigmoid_f(float v) { return 1.0f / (1.0f + expf(-v)); }

__device__ inline float2 bf16x2_to_float2_q(uint32_t v) {
  __nv_bfloat16_raw lo, hi;
  lo.x = static_cast<unsigned short>(v & 0xFFFFu);
  hi.x = static_cast<unsigned short>(v >> 16);
  return __bfloat1622float2(__nv_bfloat162(__nv_bfloat16(lo), __nv_bfloat16(hi)));
}

// Block sum over one value per thread (blockDim <= 1024): warp trees, then
// warp 0 sums the warps in order.
__device__ inline float block_sum(float v, float* scratch) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) / 32;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(~0u, v, off);
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  float total = 0.f;
  if (threadIdx.x == 0) {
    for (int w = 0; w < nwarps; ++w) total += scratch[w];
    scratch[0] = total;
  }
  __syncthreads();
  total = scratch[0];
  __syncthreads();
  return total;
}

// Block-wide max (mirrors block_sum's shape; scratch >= 33 floats).
__device__ inline float block_max(float v, float* scratch) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) / 32;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, off));
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  float total = 0.f;
  if (threadIdx.x == 0) {
    for (int w = 0; w < nwarps; ++w) total = fmaxf(total, scratch[w]);
    scratch[0] = total;
  }
  __syncthreads();
  total = scratch[0];
  __syncthreads();
  return total;
}

// The (1+w) RMSNorm of the bf16 values in xs[0, dim) (one rounding) then
// RoPE on [0, rotary_dim) at `pos` with the reference's bf16 ops; every
// thread owns dim d = threadIdx.x. scratch: smem [33] floats.
// `mscale` is the YaRN attention factor the cos/sin tables are built with
// (vLLM bakes it into its bf16 cos/sin cache): exactly 1.0f off the knob,
// and x * 1.0f is bit-exact, so the plain path is the one it always was.
__device__ inline uint16_t norm_rope_thread(float* xs, const uint16_t* __restrict__ w,
                                            int64_t pos, const float* __restrict__ inv_freq,
                                            int dim, int rotary_dim, float eps,
                                            float* scratch, float mscale) {
  const int d = threadIdx.x;
  const float x = d < dim ? xs[d] : 0.f;
  const float ss = block_sum(x * x, scratch);
  const float rstd = rsqrtf(ss / static_cast<float>(dim) + eps);
  float xn = 0.f;
  if (d < dim) xn = round_bf16(x * rstd * (1.0f + bf16_bits_to_float(w[d])));
  __syncthreads();
  if (d < dim) xs[d] = xn;
  __syncthreads();
  if (d >= dim) return 0;
  if (d >= rotary_dim) return float_to_bf16_bits(xn);
  const int half = rotary_dim / 2;
  const int i = d < half ? d : d - half;
  const float ang = __fmul_rn(static_cast<float>(pos), inv_freq[i]);
  const float c = round_bf16(cosf(ang) * mscale);
  const float s = round_bf16(sinf(ang) * mscale);
  const float rot = d < half ? -xs[d + half] : xs[d - half];
  const float t1 = round_bf16(xn * c);
  const float t2 = round_bf16(rot * s);
  return float_to_bf16_bits(t1 + t2);
}

__global__ void norm_rope_kernel(const uint16_t* __restrict__ x, int64_t x_row_stride,
                                 int64_t x_head_stride, const uint16_t* __restrict__ w,
                                 const int64_t* __restrict__ pos,
                                 const float* __restrict__ inv_freq, uint16_t* __restrict__ out,
                                 int64_t out_row_stride, int heads, int dim, int rotary_dim,
                                 float eps, float mscale) {
  extern __shared__ float xs[];
  __shared__ float scratch[33];
  const int64_t r = blockIdx.x / heads;
  const int h = static_cast<int>(blockIdx.x % heads);
  const int64_t p = pos[r];
  if (p < 0) return;
  const int d = threadIdx.x;
  if (d < dim) xs[d] = bf16_bits_to_float(x[r * x_row_stride + h * x_head_stride + d]);
  __syncthreads();
  const uint16_t o = norm_rope_thread(xs, w, p, inv_freq, dim, rotary_dim, eps, scratch, mscale);
  if (d < dim) out[r * out_row_stride + static_cast<int64_t>(h) * dim + d] = o;
}

// Fused (1+w) RMSNorm + RoPE + fp8 quantize + paged scatter for the QSA K
// cache (the zero-copy K write, 2026-10-10): it replaces
// `qsa_norm_rope_bf16(k_ -> kn_)` followed by the K half of `qsa_kv_append`,
// so the kn_ staging buffer (a full bf16 K round trip per token per layer) is
// gone. The norm + RoPE math is norm_rope_thread's, bit for bit; the
// absmax / scale / encode are kv_append_kernel's — so the cache bytes are
// exactly those of the two-kernel chain.
__global__ void norm_rope_append_fp8_kernel(const uint16_t* __restrict__ x,
                                            int64_t x_row_stride, int64_t x_head_stride,
                                            const uint16_t* __restrict__ w,
                                            const int64_t* __restrict__ pos,
                                            const float* __restrict__ inv_freq,
                                            const int32_t* __restrict__ req_ids,
                                            const int32_t* __restrict__ block_tables,
                                            int blocks_per_request, int block_tokens, int kv_heads,
                                            int dim, int rotary_dim, float eps, float mscale,
                                            uint8_t* __restrict__ k_cache,
                                            float* __restrict__ k_scale) {
  extern __shared__ float xs[];
  __shared__ float scratch[33];
  const int64_t r = blockIdx.x / kv_heads;
  const int h = static_cast<int>(blockIdx.x % kv_heads);
  const int64_t p = pos[r];
  if (p < 0) return;
  const int d = threadIdx.x;
  if (d < dim) xs[d] = bf16_bits_to_float(x[r * x_row_stride + h * x_head_stride + d]);
  __syncthreads();
  const uint16_t o = norm_rope_thread(xs, w, p, inv_freq, dim, rotary_dim, eps, scratch, mscale);
  __syncthreads();  // every thread is done reading xs before it is overwritten
  if (d < dim) xs[d] = bf16_bits_to_float(o);
  __syncthreads();
  float amax = 0.f;
  if (d < dim) amax = fabsf(xs[d]);
  amax = block_max(amax, scratch);
  if (d < dim) {
    const auto s = latent_fp8_row_scale(amax);
    const int32_t blk =
        block_tables[static_cast<int64_t>(req_ids[r]) * blocks_per_request + p / block_tokens];
    const int64_t phys = static_cast<int64_t>(blk) * block_tokens + p % block_tokens;
    const int width = kv_heads * dim;
    k_cache[phys * width + h * dim + d] = latent_fp8_encode(xs[d], s.inv);
    if (d == 0) k_scale[phys * kv_heads + h] = s.scale;
  }
}

__global__ void kv_append_kernel(const uint16_t* __restrict__ k, int64_t k_row_stride,
                                 const uint16_t* __restrict__ v, int64_t v_row_stride,
                                 const int32_t* __restrict__ req_ids,
                                 const int64_t* __restrict__ pos,
                                 const int32_t* __restrict__ block_tables,
                                 int blocks_per_request, int block_tokens, int kv_heads, int dim,
                                 uint16_t* __restrict__ k_cache, uint16_t* __restrict__ v_cache,
                                 float* __restrict__ k_scale, float* __restrict__ v_scale) {
  const int64_t r = blockIdx.x;
  const int64_t p = pos[r];
  if (p < 0) return;
  const int32_t blk = block_tables[static_cast<int64_t>(req_ids[r]) * blocks_per_request +
                                   p / block_tokens];
  const int64_t phys = static_cast<int64_t>(blk) * block_tokens + p % block_tokens;
  if (k_scale == nullptr) {
    // bf16: the historical copy, element-strided over the row (bitwise unchanged).
    const int width = kv_heads * dim;
    for (int e = threadIdx.x; e < width; e += blockDim.x) {
      k_cache[phys * width + e] = k[r * k_row_stride + e];
      v_cache[phys * width + e] = v[r * v_row_stride + e];
    }
    return;
  }
  // fp8: cooperative over the block. Stage this token's K/V rows in smem ONCE
  // (a single global read — the previous per-head form read every row twice:
  // an absmax pass then an encode pass), reduce the per-head absmax with one
  // warp per head, then encode every element. k == nullptr: the K half was
  // already written by the fused norm+RoPE+quantize kernel
  // (norm_rope_append_fp8); only V is written here.
  const int width = kv_heads * dim;
  extern __shared__ uint16_t kvstage[];  // [width] k bf16, then [width] v bf16
  uint16_t* kb = kvstage;
  uint16_t* vb = kvstage + width;
  for (int e = threadIdx.x; e < width; e += blockDim.x) {
    kb[e] = k ? k[r * k_row_stride + e] : uint16_t(0);
    vb[e] = v[r * v_row_stride + e];
  }
  __syncthreads();
  __shared__ float kinv[32], vinv[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (warp < kv_heads) {
    float ka = 0.f, va = 0.f;
    for (int e = lane; e < dim; e += 32) {
      if (k) ka = fmaxf(ka, fabsf(bf16_bits_to_float(kb[warp * dim + e])));
      va = fmaxf(va, fabsf(bf16_bits_to_float(vb[warp * dim + e])));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      ka = fmaxf(ka, __shfl_xor_sync(~0u, ka, off));
      va = fmaxf(va, __shfl_xor_sync(~0u, va, off));
    }
    if (lane == 0) {
      const auto sk = latent_fp8_row_scale(ka);
      const auto sv = latent_fp8_row_scale(va);
      kinv[warp] = sk.inv;
      vinv[warp] = sv.inv;
      if (k) k_scale[phys * kv_heads + warp] = sk.scale;
      v_scale[phys * kv_heads + warp] = sv.scale;
    }
  }
  __syncthreads();
  uint8_t* kdst = reinterpret_cast<uint8_t*>(k_cache);
  uint8_t* vdst = reinterpret_cast<uint8_t*>(v_cache);
  for (int e = threadIdx.x; e < width; e += blockDim.x) {
    const int h = e / dim;
    if (k) kdst[phys * width + e] = latent_fp8_encode(bf16_bits_to_float(kb[e]), kinv[h]);
    vdst[phys * width + e] = latent_fp8_encode(bf16_bits_to_float(vb[e]), vinv[h]);
  }
}

// xs holds the bf16 mean of a pool's raw keys; the compressed key (norm,
// RoPE at `pos`) goes to out.
__device__ inline void compress_finish(float* xs, const uint16_t* __restrict__ w,
                                       const float* __restrict__ inv_freq, int64_t pos, int dim,
                                       int rotary_dim, float eps, float mscale, float* scratch,
                                       uint16_t* __restrict__ out) {
  const uint16_t o = norm_rope_thread(xs, w, pos, inv_freq, dim, rotary_dim, eps, scratch, mscale);
  if (threadIdx.x < dim) out[threadIdx.x] = o;
}

__global__ void index_compress_write_kernel(const uint16_t* __restrict__ raw_k, int64_t k_stride,
                                            const uint16_t* __restrict__ w_k,
                                            const float* __restrict__ inv_freq,
                                            const int32_t* __restrict__ block_table,
                                            int pools_per_block, int64_t first_pool, int n_pools,
                                            uint16_t* __restrict__ index_cache, int kpool, int dim,
                                            int rotary_dim, float eps, float mscale) {
  extern __shared__ float xs[];
  __shared__ float scratch[33];
  const int i = blockIdx.x;
  if (i >= n_pools) return;
  const int64_t pool = first_pool + i;
  const int32_t blk = block_table[pool / pools_per_block];
  const int64_t slot = static_cast<int64_t>(blk) * pools_per_block + pool % pools_per_block;
  const int d = threadIdx.x;
  if (d < dim) {
    float acc = 0.f;
    for (int s = 0; s < kpool; ++s)
      acc += bf16_bits_to_float(raw_k[(static_cast<int64_t>(i) * kpool + s) * k_stride + d]);
    xs[d] = round_bf16(acc / static_cast<float>(kpool));
  }
  __syncthreads();
  compress_finish(xs, w_k, inv_freq, pool * kpool, dim, rotary_dim, eps, mscale, scratch,
                  index_cache + slot * dim);
}

__global__ void index_tail_seed_kernel(const uint16_t* __restrict__ raw_k, int64_t k_stride,
                                       const int32_t* __restrict__ req_ids,
                                       const int64_t* __restrict__ pos, int64_t tokens,
                                       uint16_t* __restrict__ ring, int kpool, int dim) {
  const int64_t i = blockIdx.x;
  if (i >= tokens) return;
  const int32_t req = req_ids[i];
  const int64_t p = pos[i];
  if (p < 0) return;
  const int64_t ahead_idx = min(i + kpool, tokens - 1);
  const bool ahead_same = (i + kpool < tokens) && req_ids[ahead_idx] == req && pos[ahead_idx] >= 0;
  if (ahead_same) return;
  const int slot = static_cast<int>(p % kpool);
  const int d = threadIdx.x;
  if (d < dim) ring[(static_cast<int64_t>(req) * kpool + slot) * dim + d] = raw_k[i * k_stride + d];
}

__global__ void index_decode_update_kernel(
    const uint16_t* __restrict__ raw_k, int64_t k_stride, const uint16_t* __restrict__ w_k,
    const float* __restrict__ inv_freq, const int32_t* __restrict__ req_ids,
    const int64_t* __restrict__ pos, const int32_t* __restrict__ req_spans,
    const int32_t* __restrict__ block_tables, int blocks_per_request, uint16_t* __restrict__ ring,
    uint16_t* __restrict__ index_cache, int pools_per_block, int kpool, int dim, int rotary_dim,
    float eps, float mscale, uint16_t* __restrict__ ring_snapshots) {
  extern __shared__ float xs[];
  __shared__ float scratch[33];
  const int span = blockIdx.x;
  const int t0 = req_spans[span * 2];
  const int t1 = t0 + req_spans[span * 2 + 1];
  int first_real = t0;
  while (first_real < t1 && pos[first_real] < 0) ++first_real;
  if (first_real == t1) return;
  const int req = req_ids[first_real];
  const int d = threadIdx.x;
  const int64_t ring_elems = static_cast<int64_t>(kpool) * dim;
  uint16_t* my_ring = ring + static_cast<int64_t>(req) * ring_elems;
  for (int t = t0; t < t1; ++t) {
    const int64_t p = pos[t];
    if (p < 0) continue;
    const int slot = static_cast<int>(p % kpool);
    if (slot == kpool - 1) {
      const int64_t pool_start = p - (kpool - 1);
      if (d < dim) {
        float acc = 0.f;
        for (int s = 0; s < kpool; ++s) {
          const int rs = static_cast<int>((pool_start + s) % kpool);
          const float kk = (s == kpool - 1)
                               ? bf16_bits_to_float(raw_k[static_cast<int64_t>(t) * k_stride + d])
                               : bf16_bits_to_float(my_ring[static_cast<int64_t>(rs) * dim + d]);
          acc += kk;
        }
        xs[d] = round_bf16(acc / static_cast<float>(kpool));
      }
      __syncthreads();
      const int64_t pool = p / kpool;
      const int32_t blk = block_tables[static_cast<int64_t>(req) * blocks_per_request +
                                       pool / pools_per_block];
      const int64_t phys = static_cast<int64_t>(blk) * pools_per_block + pool % pools_per_block;
      compress_finish(xs, w_k, inv_freq, pool_start, dim, rotary_dim, eps, mscale, scratch,
                      index_cache + phys * dim);
      __syncthreads();
    }
    if (d < dim) my_ring[static_cast<int64_t>(slot) * dim + d] = raw_k[static_cast<int64_t>(t) * k_stride + d];
    if (ring_snapshots && t + 1 < t1) {
      __syncthreads();
      uint16_t* snap = ring_snapshots + static_cast<int64_t>(t) * ring_elems;
      for (int64_t e = threadIdx.x; e < ring_elems; e += blockDim.x) snap[e] = my_ring[e];
    }
  }
}

// ---- scoring ----------------------------------------------------------------
// Block (stripe, row): the row's q (4 heads x 128) in smem as floats; each
// warp scores one pool per iteration: lane (h = lane / 8, c = lane % 8)
// holds dims [16c, 16c + 16) of head h and of the key row.
constexpr int kScoreWarps = 8;
constexpr int kScoreThreads = kScoreWarps * 32;
constexpr int kScorePoolsPerBlock = 256;

__global__ __launch_bounds__(kScoreThreads) void index_score_kernel(
    const uint16_t* __restrict__ q, int64_t q_row_stride, const int32_t* __restrict__ req_ids,
    const int64_t* __restrict__ pos, const int32_t* __restrict__ block_tables,
    int blocks_per_request, const uint16_t* __restrict__ index_cache, int pools_per_block,
    int heads, int kpool, uint64_t* __restrict__ keys_ws, int64_t ws_stride, float sqrt_dim,
    int select_k) {
  __shared__ float qs[4 * 128];
  const int64_t r = blockIdx.y;
  const int64_t p = pos[r];
  if (p < 0) return;
  const int64_t visible = (p + 1) / kpool;
  if (visible <= select_k) return;
  const int64_t p0 = static_cast<int64_t>(blockIdx.x) * kScorePoolsPerBlock;
  if (p0 >= visible) return;
  // Heads beyond `heads` (<= 4) hold zeros: their lanes add nothing.
  for (int i = threadIdx.x; i < 4 * 128; i += kScoreThreads)
    qs[i] = i < heads * 128 ? bf16_bits_to_float(q[r * q_row_stride + i]) : 0.f;
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int h = lane >> 3, c = lane & 7;
  const int32_t* bt = block_tables + static_cast<int64_t>(req_ids[r]) * blocks_per_request;
  const int64_t p1 = min(visible, p0 + kScorePoolsPerBlock);
  for (int64_t pool = p0 + warp; pool < p1; pool += kScoreWarps) {
    const int32_t blk = bt[pool / pools_per_block];
    const int64_t slot = static_cast<int64_t>(blk) * pools_per_block + pool % pools_per_block;
    const uint16_t* krow = index_cache + slot * 128 + c * 16;
    const float* qrow = qs + h * 128 + c * 16;
    // 16 dims as two uint4 loads (8 bf16 each), the fma chain in dim order.
    float partial = 0.f;
    const uint4 k0 = reinterpret_cast<const uint4*>(krow)[0];
    const uint4 k1 = reinterpret_cast<const uint4*>(krow)[1];
    const uint32_t kw[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float2 kf = bf16x2_to_float2_q(kw[j]);
      partial = __fmaf_rn(qrow[2 * j], kf.x, partial);
      partial = __fmaf_rn(qrow[2 * j + 1], kf.y, partial);
    }
    // The head's 8 lanes: xor 1, 2, 4.
    partial = __fadd_rn(partial, __shfl_xor_sync(~0u, partial, 1));
    partial = __fadd_rn(partial, __shfl_xor_sync(~0u, partial, 2));
    partial = __fadd_rn(partial, __shfl_xor_sync(~0u, partial, 4));
    float total = fmaxf(partial, 0.f);
    total = __fadd_rn(total, __shfl_xor_sync(~0u, total, 8));
    total = __fadd_rn(total, __shfl_xor_sync(~0u, total, 16));
    const float score = __fdiv_rn(total, sqrt_dim);
    if (lane == 0)
      keys_ws[r * ws_stride + pool] =
          (static_cast<uint64_t>(~sortable_f32_dev(score)) << kIdxBits) | static_cast<uint64_t>(pool);
  }
}

struct KeysRowFn {
  static constexpr bool kWarpCooperative = false;
  const uint64_t* keys;
  __device__ uint64_t operator()(int64_t pool) const { return keys[pool]; }
};

constexpr int kSelectThreads = 256;

// Find the exact top-k boundary by radix digits. Sorting every 2048-key
// tile leaves almost all SMs idle during long-context decode; only the
// boundary bin needs ordering. Composite keys make ties unambiguous, and
// expand_from_best sorts the final pool ids independently of gather order.
constexpr int kQsaRadixBits = 10;
constexpr int kQsaHistBins = 1 << kQsaRadixBits;
constexpr int kQsaBoundaryKeys = 256;

__device__ inline void qsa_hist_add(int32_t* hist, int bin) {
  const unsigned peers = __match_any_sync(__activemask(), bin);
  if ((threadIdx.x & 31) == __ffs(peers) - 1) atomicAdd(hist + bin, __popc(peers));
}

__device__ inline void qsa_boundary_bin(const int32_t* hist, int remaining, int* boundary,
                                        int* below) {
  __shared__ int warp_totals[kSelectThreads / 32];
  constexpr int per = kQsaHistBins / kSelectThreads;
  const int first = threadIdx.x * per;
  int sum = 0;
#pragma unroll
  for (int i = 0; i < per; ++i) sum += hist[first + i];
  int inclusive = sum;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int value = __shfl_up_sync(0xffffffffu, inclusive, offset);
    if (lane >= offset) inclusive += value;
  }
  if (lane == 31) warp_totals[warp] = inclusive;
  __syncthreads();
  int exclusive = inclusive - sum;
  for (int w = 0; w < warp; ++w) exclusive += warp_totals[w];
  if (exclusive < remaining && remaining <= exclusive + sum) {
    for (int i = 0; i < per; ++i) {
      if (exclusive + hist[first + i] >= remaining) {
        *boundary = first + i;
        *below = exclusive;
        break;
      }
      exclusive += hist[first + i];
    }
  }
  __syncthreads();
}

__device__ inline void qsa_select_radix(const uint64_t* keys, int64_t visible, int select_k,
                                        uint32_t* best_hi, uint32_t* best_lo, uint32_t* tile_hi,
                                        uint32_t* tile_lo) {
  __shared__ int boundary, below, definite_count, candidate_count;
  int32_t* hist = reinterpret_cast<int32_t*>(tile_hi);
  uint64_t prefix = 0;
  int prefix_shift = 32 + kIdxBits;
  int remaining = select_k, lower = 0, count = 0, shift = 0;
  for (;;) {
    shift = max(0, prefix_shift - kQsaRadixBits);
    const int bits = prefix_shift - shift;
    const uint64_t mask = (1ull << bits) - 1;
    for (int i = threadIdx.x; i < kQsaHistBins; i += blockDim.x) hist[i] = 0;
    __syncthreads();
    for (int64_t p = threadIdx.x; p < visible; p += blockDim.x) {
      const uint64_t key = keys[p];
      if ((key >> prefix_shift) == prefix) qsa_hist_add(hist, int((key >> shift) & mask));
    }
    __syncthreads();
    qsa_boundary_bin(hist, remaining, &boundary, &below);
    count = hist[boundary];
    lower += below;
    remaining -= below;
    prefix = (prefix << bits) | uint64_t(boundary);
    // The pool-id suffix makes keys unique, so the final digit always
    // leaves at most one key. Equal scores need no approximation/fallback.
    __syncthreads();
    if (count <= kQsaBoundaryKeys || shift == 0) break;
    prefix_shift = shift;
  }

  // The histogram storage becomes the boundary-key array. Values below
  // the boundary go straight into best; only its <=256 candidates are ranked.
  if (threadIdx.x == 0) definite_count = candidate_count = 0;
  __syncthreads();
  for (int64_t p = threadIdx.x; p < visible; p += blockDim.x) {
    const uint64_t key = keys[p];
    const uint64_t top = key >> shift;
    if (top < prefix) {
      const int i = atomicAdd(&definite_count, 1);
      if (i < select_k) {
        best_hi[i] = uint32_t(key >> 32);
        best_lo[i] = uint32_t(key);
      }
    } else if (top == prefix) {
      const int i = atomicAdd(&candidate_count, 1);
      if (i < kQsaBoundaryKeys) {
        tile_hi[i] = uint32_t(key >> 32);
        tile_lo[i] = uint32_t(key);
      }
    }
  }
  __syncthreads();
  const int i = threadIdx.x;
  const uint32_t hi = i < count ? tile_hi[i] : 0xffffffffu;
  const uint32_t lo = i < count ? tile_lo[i] : 0xffffffffu;
  int rank = 0;
#pragma unroll 8
  for (int j = 0; j < count; ++j) rank += key_less(tile_hi[j], tile_lo[j], hi, lo);
  if (i < count && rank < remaining) {
    best_hi[lower + rank] = hi;
    best_lo[lower + rank] = lo;
  }
  __syncthreads();
}

__global__ __launch_bounds__(kSelectThreads) void select_from_keys_kernel(
    const uint64_t* __restrict__ keys_ws, int64_t ws_stride, const int64_t* __restrict__ pos,
    int select_k, int kpool, int max_selected, int32_t* __restrict__ topk_out,
    int32_t* __restrict__ out_counts) {
  extern __shared__ uint64_t smem_u64[];
  uint32_t* best_hi = reinterpret_cast<uint32_t*>(smem_u64);
  uint32_t* best_lo = best_hi + select_k;
  uint32_t* tile_hi = best_lo + select_k;
  uint32_t* tile_lo = tile_hi + kSelectTile;
  int32_t* scratch = reinterpret_cast<int32_t*>(tile_lo + kSelectTile);
  int* smem_count = reinterpret_cast<int*>(scratch + select_k);
  const int64_t r = blockIdx.x;
  const int64_t p = pos[r];
  if (p < 0) {
    for (int col = threadIdx.x; col < max_selected; col += blockDim.x)
      topk_out[r * max_selected + col] = -1;
    if (threadIdx.x == 0) out_counts[r] = 0;
    return;
  }
  const int64_t visible = (p + 1) / kpool;
  // Every complete pool and the incomplete tail survive. Their final token
  // order is just [0, p], independent of scores (which need not be written).
  if (visible <= select_k) {
    for (int col = threadIdx.x; col < max_selected; col += blockDim.x)
      topk_out[r * max_selected + col] = col <= p ? col : -1;
    if (threadIdx.x == 0) out_counts[r] = static_cast<int32_t>(p + 1);
    return;
  }
  for (int i = threadIdx.x; i < select_k; i += blockDim.x) {
    best_hi[i] = 0xFFFFFFFFu;
    best_lo[i] = 0xFFFFFFFFu;
  }
  __syncthreads();
  if (visible > kSelectTile)
    qsa_select_radix(keys_ws + r * ws_stride, visible, select_k, best_hi, best_lo, tile_hi,
                     tile_lo);
  else {
    KeysRowFn fn{keys_ws + r * ws_stride};
    select_topk_stream(fn, 0, visible, best_hi, best_lo, tile_hi, tile_lo, select_k);
  }
  __syncthreads();
  const int cnt = expand_from_best(best_hi, best_lo, select_k, p, kpool, max_selected,
                                   topk_out + r * max_selected, scratch, smem_count);
  if (threadIdx.x == 0) out_counts[r] = cnt;
}

// ---- listed GQA attention -----------------------------------------------------
constexpr int kQsaTile = 32;

// kHpw heads per warp (2026-09-29): a block covers hpb = warps x kHpw heads
// of one kv head. Each head's chain — its scores tree per token, its running
// max and rescale, its PV order — is the one-head-per-warp kernel's exactly,
// so every geometry is bitwise every other (the test pins them against each
// other). The default stays one head per warp, up to 8 per block: gathering
// a kv head's rows once per block (12 heads per block, one or two per warp)
// measured +0.1 to +0.3 ms a step at 12K and 47K context on the fabric.
template <int D, int kHpw>
__global__ void attn_partial_kernel(const uint16_t* __restrict__ q, int64_t q_row_stride,
                                    const uint16_t* __restrict__ k_cache,
                                    const uint16_t* __restrict__ v_cache,
                                    const float* __restrict__ k_scale,
                                    const float* __restrict__ v_scale,
                                    const int32_t* __restrict__ req_ids,
                                    const int32_t* __restrict__ topk, int topk_stride,
                                    const int32_t* __restrict__ counts, int n_split,
                                    int local_heads, int kv_heads, int block_tokens,
                                    const int32_t* __restrict__ block_tables,
                                    int blocks_per_request, float scale, float* __restrict__ m_ws,
                                    float* __restrict__ l_ws, float* __restrict__ c_ws,
                                    int async_gather) {
  constexpr int kGroups = 32;
  constexpr int kDslice = D / kGroups;  // 8 at D=256
  constexpr int kRowStride = D + 8;     // padded smem row (u16)
  const int64_t r = blockIdx.x;
  const int s = blockIdx.y;
  const int warps = blockDim.x / 32;
  const int hpb = warps * kHpw;  // heads per block
  const int h0 = blockIdx.z * hpb;
  const int hl = threadIdx.x / 32;
  const int g = threadIdx.x % 32;
  const int hw0 = h0 + hl * kHpw;  // this warp's first head
  const int heads_per_kv = local_heads / kv_heads;
  const int kvh = h0 / heads_per_kv;
  const int width = kv_heads * D;

  const int cnt = counts[r];
  const int chunk = (cnt + n_split - 1) / n_split;
  const int t_begin = s * chunk;
  const int t_end = min(cnt, t_begin + chunk);
  const int64_t base_m = (r * n_split + s) * local_heads;
  if (t_begin >= t_end) {
    for (int dd = threadIdx.x; dd < hpb * D; dd += blockDim.x)
      c_ws[(base_m + h0 + dd / D) * D + dd % D] = 0.0f;
    if (threadIdx.x < hpb) {
      m_ws[base_m + h0 + threadIdx.x] = -INFINITY;
      l_ws[base_m + h0 + threadIdx.x] = 0.0f;
    }
    return;
  }

  extern __shared__ uint16_t sm16[];
  uint16_t* kt = sm16;                            // [kQsaTile][kRowStride]
  uint16_t* vt = kt + kQsaTile * kRowStride;      // [kQsaTile][kRowStride]
  float* scores = reinterpret_cast<float*>(vt + kQsaTile * kRowStride);  // [hpb][kQsaTile + 1]
  float* l = scores + hpb * (kQsaTile + 1);       // [hpb]
  const int scores_stride = kQsaTile + 1;
  // The tile's physical K/V rows, resolved once per tile by kQsaTile threads
  // (2026-09-29): the gather loop below used to chase topk -> block table ->
  // row inside every iteration — three dependent global loads per uint4,
  // eleven iterations per thread in series, ~25 us a tile against the 2 us
  // one round trip costs; the kernel ran 180 us per layer at 2K context for
  // ~16 MB of K/V. With the offsets in shared memory every row load of the
  // tile issues back to back. The arithmetic is untouched: bitwise.
  __shared__ int64_t phys_rows[kQsaTile];
  __shared__ int64_t phys_next[kQsaTile];

  float m_reg[kHpw];
  float creg[kHpw][kDslice];
  float qreg[kHpw][kDslice];
#pragma unroll
  for (int j = 0; j < kHpw; ++j) {
    m_reg[j] = -INFINITY;
    const uint16_t* qrow = q + r * q_row_stride + static_cast<int64_t>(hw0 + j) * D + g * kDslice;
#pragma unroll
    for (int i = 0; i < kDslice; ++i) {
      creg[j][i] = 0.0f;
      qreg[j][i] = bf16_bits_to_float(qrow[i]);
    }
  }
  if (threadIdx.x < hpb) l[threadIdx.x] = 0.0f;
  __syncthreads();

  const int32_t* bt = block_tables + static_cast<int64_t>(req_ids[r]) * blocks_per_request;
  const int32_t* toks = topk + r * topk_stride;
  constexpr int kVecPerRow = D / 8;

  // The tile loop with the gather as cp.async copies (2026-09-29, the
  // register-lean revision): a tile's K rows and V rows are two async
  // groups into the same kt / vt buffers; the next tile's K copy is issued
  // as soon as this tile's scores have read kt (it runs under this tile's
  // PV phase), the next V copy as soon as PV has read vt (it runs under the
  // next tile's resolve and scores). No registers hold the tile (the first
  // pipelined form held twelve uint4 across the compute: 125 registers, one
  // block per SM, 170 us against the serial 137 at 2K context), the shared
  // footprint is the serial form's, and every arithmetic step is the serial
  // loop's in the same order: bitwise. DGPP_QSA_ASYNC=0 keeps the serial
  // form (the A/B's other leg).
  constexpr int kGatherBatch = 12;
  const auto resolve = [&](int t0, int n, int64_t* phys) {
    if (static_cast<int>(threadIdx.x) < n) {
      const int64_t tok = toks[t0 + threadIdx.x];
      const int32_t blk = bt[tok / block_tokens];
      phys[threadIdx.x] = static_cast<int64_t>(blk) * block_tokens + tok % block_tokens;
    }
  };
  const auto load_tile = [&](int n, const int64_t* phys, uint4 (&buf)[kGatherBatch], int base) {
    const int total = n * kVecPerRow * 2;
#pragma unroll
    for (int j = 0; j < kGatherBatch; ++j) {
      const int idx = base + j * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
      if (idx < total) {
        const int which = idx / (n * kVecPerRow);
        const int rem = idx - which * (n * kVecPerRow);
        const int tt = rem / kVecPerRow;
        const int c8 = rem - tt * kVecPerRow;
        const uint16_t* src = (which == 0 ? k_cache : v_cache) + phys[tt] * width + kvh * D + c8 * 8;
        buf[j] = *reinterpret_cast<const uint4*>(src);
      }
    }
  };
  const auto store_tile = [&](int n, const uint4 (&buf)[kGatherBatch], int base) {
    const int total = n * kVecPerRow * 2;
#pragma unroll
    for (int j = 0; j < kGatherBatch; ++j) {
      const int idx = base + j * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
      if (idx < total) {
        const int which = idx / (n * kVecPerRow);
        const int rem = idx - which * (n * kVecPerRow);
        const int tt = rem / kVecPerRow;
        const int c8 = rem - tt * kVecPerRow;
        uint16_t* dst = (which == 0 ? kt : vt) + tt * kRowStride + c8 * 8;
        *reinterpret_cast<uint4*>(dst) = buf[j];
      }
    }
  };
  // One matrix's rows of a tile as a cp.async group (16 bytes per copy;
  // the rows and the vectors are 16-byte aligned as the uint4 loads were).
  const auto issue_rows = [&](int n, const int64_t* phys, const uint16_t* cache, uint16_t* dst_tile) {
    const int total = n * kVecPerRow;
    for (int idx = static_cast<int>(threadIdx.x); idx < total; idx += static_cast<int>(blockDim.x)) {
      const int tt = idx / kVecPerRow;
      const int c8 = idx - tt * kVecPerRow;
      const uint16_t* src = cache + phys[tt] * width + kvh * D + c8 * 8;
      const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst_tile + tt * kRowStride + c8 * 8));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };
  // The tile's scores out of kt (part 1) and, after the barrier the serial
  // form had between them, the softmax bookkeeping and PV out of vt (part 2),
  // head by head for the warp's kHpw heads.
  float tile_max[kHpw];
  const auto scores_phase = [&](int n) {
#pragma unroll
    for (int j = 0; j < kHpw; ++j) {
      // The tile's scores: every lane's 8-dim partial for all 32 tokens
      // first, then the xor tree over all 32 at once (the per-token tree was
      // five dependent shuffles per token, 32 tokens in series). The same
      // tree per token, so the scores are bitwise.
      float partial[kQsaTile];
#pragma unroll
      for (int tt = 0; tt < kQsaTile; ++tt) {
        float acc = 0.f;
        if (tt < n) {
          const uint16_t* krow = kt + tt * kRowStride + g * kDslice;
#pragma unroll
          for (int i = 0; i < kDslice; ++i) acc += qreg[j][i] * bf16_bits_to_float(krow[i]);
        }
        partial[tt] = acc;
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int tt = 0; tt < kQsaTile; ++tt) partial[tt] += __shfl_xor_sync(~0u, partial[tt], off);
      }
      float tm = -INFINITY;
      float* srow = scores + (hl * kHpw + j) * scores_stride;
#pragma unroll
      for (int tt = 0; tt < kQsaTile; ++tt) {
        if (tt < n) {
          const float score = partial[tt] * scale;
          if (g == 0) srow[tt] = score;
          tm = fmaxf(tm, score);
        }
      }
      tile_max[j] = tm;
    }
  };
  const auto pv_phase = [&](int n) {
#pragma unroll
    for (int j = 0; j < kHpw; ++j) {
      const int hrow = hl * kHpw + j;
      const float* srow = scores + hrow * scores_stride;
      const float m_new = fmaxf(m_reg[j], tile_max[j]);
      const float rescale = expf(m_reg[j] - m_new);
#pragma unroll
      for (int i = 0; i < kDslice; ++i) creg[j][i] *= rescale;
      if (g == 0) {
        float ladd = 0.0f;
        for (int tt = 0; tt < n; ++tt) ladd += expf(srow[tt] - m_new);
        l[hrow] = l[hrow] * rescale + ladd;
      }
      for (int tt = 0; tt < n; ++tt) {
        const float p = round_bf16(expf(srow[tt] - m_new));
        const uint16_t* vrow = vt + tt * kRowStride + g * kDslice;
#pragma unroll
        for (int i = 0; i < kDslice; ++i) creg[j][i] += p * bf16_bits_to_float(vrow[i]);
      }
      m_reg[j] = m_new;
    }
  };

  if (k_scale != nullptr) {
    // fp8 cache: vectorized gather (16 codes per 16-byte load) of the e4m3
    // codes + per-(row, kv-head) scale, dequantized to bf16 in the smem
    // tiles; the scores / PV phases below are unchanged (they read bf16 from
    // kt / vt). The bf16 path is untouched (k_scale null). The 16-byte loads
    // are as the uint4 bf16 loads were: the rows and the chunks are 16-byte
    // aligned (width and D are multiples of 16, c is a multiple of 16).
    const uint8_t* k8 = reinterpret_cast<const uint8_t*>(k_cache);
    const uint8_t* v8 = reinterpret_cast<const uint8_t*>(v_cache);
    constexpr int kFp8ChunksPerRow = D / 16;
    for (int t0 = t_begin; t0 < t_end; t0 += kQsaTile) {
      const int n = min(kQsaTile, t_end - t0);
      resolve(t0, n, phys_rows);
      __syncthreads();
      for (int idx = static_cast<int>(threadIdx.x); idx < n * kFp8ChunksPerRow * 2;
           idx += static_cast<int>(blockDim.x)) {
        const int which = idx / (n * kFp8ChunksPerRow);
        const int rem = idx - which * (n * kFp8ChunksPerRow);
        const int tt = rem / kFp8ChunksPerRow;
        const int c = (rem - tt * kFp8ChunksPerRow) * 16;
        const int64_t phys = phys_rows[tt];
        const uint8_t* src = (which == 0 ? k8 : v8) + phys * width + kvh * D + c;
        const float sc = (which == 0 ? k_scale : v_scale)[phys * kv_heads + kvh];
        uint16_t* dst = (which == 0 ? kt : vt) + tt * kRowStride + c;
        const uint4 v4 = *reinterpret_cast<const uint4*>(src);
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&v4);
#pragma unroll
        for (int j = 0; j < 16; ++j) dst[j] = latent_fp8_decode_bf16(b[j], sc);
      }
      __syncthreads();
      scores_phase(n);
      __syncthreads();
      pv_phase(n);
      __syncthreads();
    }
  } else if (async_gather) {
    int n = min(kQsaTile, t_end - t_begin);
    int64_t* phys_cur = phys_rows;
    int64_t* phys_nxt = phys_next;
    resolve(t_begin, n, phys_cur);
    __syncthreads();
    issue_rows(n, phys_cur, k_cache, kt);
    issue_rows(n, phys_cur, v_cache, vt);
    for (int t0 = t_begin; t0 < t_end; t0 += kQsaTile) {
      const int t1 = t0 + kQsaTile;
      const bool has_next = t1 < t_end;
      const int n1 = has_next ? min(kQsaTile, t_end - t1) : 0;
      if (has_next) resolve(t1, n1, phys_nxt);
      asm volatile("cp.async.wait_group 1;\n" ::);  // this tile's K rows landed (its V may still be in flight)
      __syncthreads();
      scores_phase(n);
      __syncthreads();  // every warp is done with kt
      if (has_next) issue_rows(n1, phys_nxt, k_cache, kt);  // under this tile's PV
      if (has_next)
        asm volatile("cp.async.wait_group 1;\n" ::);  // this tile's V rows landed
      else
        asm volatile("cp.async.wait_group 0;\n" ::);
      __syncthreads();
      pv_phase(n);
      __syncthreads();  // every warp is done with vt and scores
      if (has_next) issue_rows(n1, phys_nxt, v_cache, vt);  // under the next tile's resolve and scores
      n = n1;
      int64_t* tmp = phys_cur; phys_cur = phys_nxt; phys_nxt = tmp;
    }
  } else {
    for (int t0 = t_begin; t0 < t_end; t0 += kQsaTile) {
      const int n = min(kQsaTile, t_end - t0);
      resolve(t0, n, phys_rows);
      __syncthreads();
      const int total = n * kVecPerRow * 2;
      for (int base = 0; base < total; base += kGatherBatch * static_cast<int>(blockDim.x)) {
        uint4 buf[kGatherBatch];
        load_tile(n, phys_rows, buf, base);
        store_tile(n, buf, base);
      }
      __syncthreads();
      scores_phase(n);
      __syncthreads();
      pv_phase(n);
      __syncthreads();
    }
  }
#pragma unroll
  for (int j = 0; j < kHpw; ++j) {
    const int h = hw0 + j;
    if (g == 0) {
      m_ws[base_m + h] = m_reg[j];
      l_ws[base_m + h] = l[hl * kHpw + j];
    }
    float* crow = c_ws + (base_m + h) * D + g * kDslice;
#pragma unroll
    for (int i = 0; i < kDslice; ++i) crow[i] = creg[j][i];
  }
}

__global__ void gate_out_kernel(const float* __restrict__ c, const uint16_t* __restrict__ gate,
                                int64_t gate_row_stride, int64_t gate_head_stride,
                                uint16_t* __restrict__ out, int heads, int dim) {
  const int64_t r = blockIdx.y;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<int64_t>(heads) * dim) return;
  const int h = static_cast<int>(i / dim), d = static_cast<int>(i % dim);
  const float o = round_bf16(c[r * heads * dim + i]);
  const float gs = round_bf16(sigmoid_f(bf16_bits_to_float(gate[r * gate_row_stride + h * gate_head_stride + d])));
  out[r * heads * dim + i] = float_to_bf16_bits(o * gs);
}

int block_threads_for(int dim) {
  int t = 32;
  while (t < dim) t *= 2;
  return t;
}

}  // namespace

void qsa_rope_inv_freq(double theta, int rotary_dim, float* inv_freq) {
  const float base = static_cast<float>(theta);
  for (int i = 0; i < rotary_dim / 2; ++i) {
    const float e = static_cast<float>(2 * i) / static_cast<float>(rotary_dim);
    inv_freq[i] = 1.0f / std::pow(base, e);
  }
}

void qsa_norm_rope_bf16(const uint16_t* x, int64_t x_row_stride, int64_t x_head_stride,
                        const uint16_t* w, const int64_t* pos, const float* inv_freq,
                        uint16_t* out, int64_t out_row_stride, int rows, int heads, int dim,
                        int rotary_dim, float eps, float mscale, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!x || !w || !pos || !inv_freq || !out) throw std::invalid_argument("qsa_norm_rope: null pointer");
  if (dim <= 0 || dim > 1024 || rotary_dim < 0 || rotary_dim > dim || rotary_dim % 2 != 0)
    throw std::invalid_argument("qsa_norm_rope: bad dims");
  const int64_t blocks = static_cast<int64_t>(rows) * heads;
  const int threads = block_threads_for(dim);
  norm_rope_kernel<<<static_cast<unsigned>(blocks), threads, dim * sizeof(float), stream>>>(
      x, x_row_stride, x_head_stride, w, pos, inv_freq, out, out_row_stride, heads, dim,
      rotary_dim, eps, mscale);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qsa_kv_append(const uint16_t* k, int64_t k_row_stride, const uint16_t* v, int64_t v_row_stride,
                   const int32_t* req_ids, const int64_t* pos, int rows,
                   const int32_t* block_tables, int blocks_per_request, int block_tokens,
                   int kv_heads, int dim, uint16_t* k_cache, uint16_t* v_cache,
                   float* k_scale, float* v_scale, cudaStream_t stream) {
  if (rows <= 0) return;
  // k may be null when the fused norm+RoPE+quantize kernel already wrote K.
  if (!v || !req_ids || !pos || !block_tables || !v_cache)
    throw std::invalid_argument("qsa_kv_append: null pointer");
  if (k == nullptr && k_scale == nullptr)
    throw std::invalid_argument("qsa_kv_append: null k with a bf16 cache");
  const size_t smem =
      k_scale != nullptr ? 2 * static_cast<size_t>(kv_heads) * dim * sizeof(uint16_t) : 0;
  kv_append_kernel<<<static_cast<unsigned>(rows), 256, smem, stream>>>(
      k, k_row_stride, v, v_row_stride, req_ids, pos, block_tables, blocks_per_request,
      block_tokens, kv_heads, dim, k_cache, v_cache, k_scale, v_scale);
  DGPP_CUDA_OK(cudaGetLastError());
}

// Fused (1+w) RMSNorm + RoPE + fp8 quantize + paged scatter for the QSA K
// cache (the zero-copy K write): one launch replaces qsa_norm_rope_bf16 into
// the kn_ staging buffer plus the K half of qsa_kv_append. The cache bytes are
// bitwise the two-kernel chain's. x is the raw k_proj output [rows, lkv * D].
void qsa_norm_rope_append_fp8(const uint16_t* x, int64_t x_row_stride, int64_t x_head_stride,
                              const uint16_t* w, const int64_t* pos, const float* inv_freq,
                              const int32_t* req_ids, const int32_t* block_tables,
                              int blocks_per_request, int block_tokens, int rows, int kv_heads,
                              int dim, int rotary_dim, float eps, float mscale, uint8_t* k_cache,
                              float* k_scale, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!x || !w || !pos || !inv_freq || !req_ids || !block_tables || !k_cache || !k_scale)
    throw std::invalid_argument("qsa_norm_rope_append_fp8: null pointer");
  const int threads = block_threads_for(dim);
  norm_rope_append_fp8_kernel<<<static_cast<unsigned>(rows * kv_heads), threads,
                                dim * sizeof(float), stream>>>(
      x, x_row_stride, x_head_stride, w, pos, inv_freq, req_ids, block_tables, blocks_per_request,
      block_tokens, kv_heads, dim, rotary_dim, eps, mscale, k_cache, k_scale);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qsa_index_compress_write(const uint16_t* raw_k, int64_t k_stride, const uint16_t* w_k,
                              const float* inv_freq, const int32_t* block_table,
                              int pools_per_block, int64_t first_pool, int n_pools,
                              uint16_t* index_cache, int kpool, int dim, int rotary_dim, float eps,
                              float mscale, cudaStream_t stream) {
  if (n_pools <= 0) return;
  if (!raw_k || !w_k || !inv_freq || !block_table || !index_cache)
    throw std::invalid_argument("qsa_index_compress: null pointer");
  if (dim <= 0 || dim > 1024) throw std::invalid_argument("qsa_index_compress: bad dim");
  const int threads = block_threads_for(dim);
  index_compress_write_kernel<<<static_cast<unsigned>(n_pools), threads, dim * sizeof(float), stream>>>(
      raw_k, k_stride, w_k, inv_freq, block_table, pools_per_block, first_pool, n_pools,
      index_cache, kpool, dim, rotary_dim, eps, mscale);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qsa_index_tail_seed(const uint16_t* raw_k, int64_t k_stride, const int32_t* req_ids,
                         const int64_t* pos, int64_t tokens, uint16_t* ring, int kpool, int dim,
                         cudaStream_t stream) {
  if (tokens <= 0) return;
  if (!raw_k || !req_ids || !pos || !ring) throw std::invalid_argument("qsa_index_tail_seed: null pointer");
  index_tail_seed_kernel<<<static_cast<unsigned>(tokens), block_threads_for(dim), 0, stream>>>(
      raw_k, k_stride, req_ids, pos, tokens, ring, kpool, dim);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qsa_index_decode_update(const uint16_t* raw_k, int64_t k_stride, const uint16_t* w_k,
                             const float* inv_freq, const int32_t* req_ids, const int64_t* pos,
                             const int32_t* req_spans, int num_requests,
                             const int32_t* block_tables, int blocks_per_request, uint16_t* ring,
                             uint16_t* index_cache, int pools_per_block, int kpool, int dim,
                             int rotary_dim, float eps, float mscale, cudaStream_t stream,
                             uint16_t* ring_snapshots) {
  if (num_requests <= 0) return;
  if (!raw_k || !w_k || !inv_freq || !req_ids || !pos || !req_spans || !block_tables || !ring ||
      !index_cache)
    throw std::invalid_argument("qsa_index_decode_update: null pointer");
  if (dim <= 0 || dim > 1024) throw std::invalid_argument("qsa_index_decode_update: bad dim");
  const int threads = block_threads_for(dim);
  index_decode_update_kernel<<<static_cast<unsigned>(num_requests), threads, dim * sizeof(float), stream>>>(
      raw_k, k_stride, w_k, inv_freq, req_ids, pos, req_spans, block_tables, blocks_per_request,
      ring, index_cache, pools_per_block, kpool, dim, rotary_dim, eps, mscale, ring_snapshots);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qsa_index_score(const uint16_t* q, int64_t q_row_stride, const int32_t* req_ids,
                     const int64_t* pos, int rows, const int32_t* block_tables,
                     int blocks_per_request, const uint16_t* index_cache, int pools_per_block,
                     int heads, int dim, int kpool, uint64_t* keys_ws, int64_t ws_stride,
                     cudaStream_t stream, int64_t visible_pool_bound, int select_k) {
  if (rows <= 0) return;
  if (!q || !req_ids || !pos || !block_tables || !index_cache || !keys_ws)
    throw std::invalid_argument("qsa_index_score: null pointer");
  if (heads < 1 || heads > 4 || dim != 128)
    throw std::invalid_argument("qsa_index_score: the lane layout is <= 4 heads x 128 dims");
  if (ws_stride > (int64_t(1) << kIdxBits))
    throw std::invalid_argument("qsa_index_score: pool ids must fit kIdxBits");
  if (visible_pool_bound < -1 || visible_pool_bound > ws_stride)
    throw std::invalid_argument("qsa_index_score: visible pool bound outside workspace");
  if (select_k < 0 || select_k > 1024)
    throw std::invalid_argument("qsa_index_score: select_k outside [0, 1024]");
  if (rows > 65535) throw std::invalid_argument("qsa_index_score: too many rows per launch");
  const int64_t pools = visible_pool_bound < 0 ? ws_stride : visible_pool_bound;
  if (pools <= select_k) return;
  const int64_t stripes = (pools + kScorePoolsPerBlock - 1) / kScorePoolsPerBlock;
  if (stripes > 0x7fffffff) throw std::invalid_argument("qsa_index_score: too many pools");
  const dim3 grid(static_cast<unsigned>(stripes), static_cast<unsigned>(rows));
  index_score_kernel<<<grid, kScoreThreads, 0, stream>>>(
      q, q_row_stride, req_ids, pos, block_tables, blocks_per_request, index_cache, pools_per_block,
      heads, kpool, keys_ws, ws_stride, std::sqrt(static_cast<float>(dim)), select_k);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qsa_select_from_keys(const uint64_t* keys_ws, int64_t ws_stride, const int64_t* pos, int rows,
                          int select_k, int kpool, int max_selected, int32_t* topk_out,
                          int32_t* out_counts, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!keys_ws || !pos || !topk_out || !out_counts)
    throw std::invalid_argument("qsa_select_from_keys: null pointer");
  if (select_k <= 0 || select_k > kSelectTile / 2 || select_k > 4 * kSelectThreads)
    throw std::invalid_argument("qsa_select_from_keys: select_k outside (0, 1024]");
  if (max_selected < select_k * kpool + kpool - 1)
    throw std::invalid_argument("qsa_select_from_keys: max_selected too small for the budget + tail");
  const size_t smem = static_cast<size_t>(select_k) * 8 + static_cast<size_t>(kSelectTile) * 8 +
                      static_cast<size_t>(select_k + 1) * 4 + 8;
  select_from_keys_kernel<<<static_cast<unsigned>(rows), kSelectThreads, smem, stream>>>(
      keys_ws, ws_stride, pos, select_k, kpool, max_selected, topk_out, out_counts);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qsa_attn_partial(const uint16_t* q, int64_t q_row_stride, const uint16_t* k_cache,
                      const uint16_t* v_cache, const int32_t* req_ids, const int32_t* topk,
                      int topk_stride, const int32_t* counts, int rows, int n_split,
                      int local_heads, int kv_heads, int dim, int block_tokens,
                      const int32_t* block_tables, int blocks_per_request, float scale,
                      float* m_ws, float* l_ws, float* c_ws, cudaStream_t stream,
                      const float* k_scale, const float* v_scale) {
  qsa_attn_partial_gather(q, q_row_stride, k_cache, v_cache, req_ids, topk, topk_stride, counts, rows,
                          n_split, local_heads, kv_heads, dim, block_tokens, block_tables,
                          blocks_per_request, scale, m_ws, l_ws, c_ws, stream, -1, 0, 0, k_scale,
                          v_scale);
}

void qsa_attn_partial_gather(const uint16_t* q, int64_t q_row_stride, const uint16_t* k_cache,
                             const uint16_t* v_cache, const int32_t* req_ids, const int32_t* topk,
                             int topk_stride, const int32_t* counts, int rows, int n_split,
                             int local_heads, int kv_heads, int dim, int block_tokens,
                             const int32_t* block_tables, int blocks_per_request, float scale,
                             float* m_ws, float* l_ws, float* c_ws, cudaStream_t stream,
                             int async_gather, int heads_per_block, int heads_per_warp,
                             const float* k_scale, const float* v_scale) {
  if (rows <= 0) return;
  if (!q || !k_cache || !v_cache || !req_ids || !topk || !counts || !block_tables || !m_ws ||
      !l_ws || !c_ws)
    throw std::invalid_argument("qsa_attn_partial: null pointer");
  if (kv_heads <= 0 || local_heads % kv_heads != 0)
    throw std::invalid_argument("qsa_attn_partial: local_heads must be a multiple of kv_heads");
  const int heads_per_kv = local_heads / kv_heads;
  if (n_split <= 0) throw std::invalid_argument("qsa_attn_partial: n_split must be positive");
  // The block's heads (a divisor of the heads per kv head) and the heads per
  // warp: a caller's pin, or the default — up to 8 heads, one per warp. The
  // other geometries (every head of a kv head in one block, so its rows are
  // gathered once; two heads per warp) are bitwise this one and were
  // measured on the fabric 2026-09-29 at 12K and 47K context: +0.1 to +0.3
  // ms a step against the default, so the default stays; the test keeps
  // them bitwise. DGPP_QSA_ASYNC=0: the serial gather.
  static const int env_async = [] {
    const char* v = std::getenv("DGPP_QSA_ASYNC");
    return (v && v[0] == '0') ? 0 : 1;
  }();
  if (async_gather < 0) async_gather = env_async;
  int hpb = heads_per_block;
  if (hpb <= 0) {
    hpb = 8;
    while (hpb > 1 && heads_per_kv % hpb != 0) --hpb;
  }
  if (hpb > heads_per_kv || heads_per_kv % hpb != 0)
    throw std::invalid_argument("qsa_attn_partial: heads per block must divide the heads per kv head");
  const int hpw = heads_per_warp > 0 ? heads_per_warp : 1;
  if (hpw > 3 || hpb % hpw != 0 || (hpb / hpw) * 32 > 1024)
    throw std::invalid_argument("qsa_attn_partial: heads per warp must be 1..3 and divide the block's heads");
  const int row_stride = dim + 8;
  const size_t smem = static_cast<size_t>(2 * kQsaTile) * row_stride * 2 +
                      static_cast<size_t>(hpb) * (kQsaTile + 1) * 4 + static_cast<size_t>(hpb) * 4;
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(n_split),
                  static_cast<unsigned>(local_heads / hpb));
  const int threads = (hpb / hpw) * 32;
  const auto launch = [&](auto dim_tag, auto hpw_tag) {
    constexpr int D = decltype(dim_tag)::value;
    constexpr int H = decltype(hpw_tag)::value;
    attn_partial_kernel<D, H><<<grid, threads, smem, stream>>>(
        q, q_row_stride, k_cache, v_cache, k_scale, v_scale, req_ids, topk, topk_stride, counts,
        n_split, local_heads, kv_heads, block_tokens, block_tables, blocks_per_request, scale, m_ws,
        l_ws, c_ws, async_gather);
  };
  const auto by_hpw = [&](auto dim_tag) {
    switch (hpw) {
      case 1: launch(dim_tag, std::integral_constant<int, 1>{}); break;
      case 2: launch(dim_tag, std::integral_constant<int, 2>{}); break;
      default: launch(dim_tag, std::integral_constant<int, 3>{}); break;
    }
  };
  switch (dim) {
    case 256: by_hpw(std::integral_constant<int, 256>{}); break;
    case 512: by_hpw(std::integral_constant<int, 512>{}); break;
    default:
      throw std::invalid_argument("qsa_attn_partial: dim must be 256 or 512");
  }
  DGPP_CUDA_OK(cudaGetLastError());
}

void qsa_gate_out(const float* c, const uint16_t* gate, int64_t gate_row_stride,
                  int64_t gate_head_stride, uint16_t* out, int rows, int heads, int dim,
                  cudaStream_t stream) {
  if (rows <= 0) return;
  if (!c || !gate || !out) throw std::invalid_argument("qsa_gate_out: null pointer");
  const int64_t n = static_cast<int64_t>(heads) * dim;
  const dim3 grid(static_cast<unsigned>((n + 255) / 256), static_cast<unsigned>(rows));
  gate_out_kernel<<<grid, 256, 0, stream>>>(c, gate, gate_row_stride, gate_head_stride, out, heads, dim);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
