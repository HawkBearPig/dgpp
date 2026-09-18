#include "kernels/qsa.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <cuda_bf16.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
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

__global__ void kv_append_kernel(const uint16_t* __restrict__ k, int64_t k_row_stride,
                                 const uint16_t* __restrict__ v, int64_t v_row_stride,
                                 const int32_t* __restrict__ req_ids,
                                 const int64_t* __restrict__ pos,
                                 const int32_t* __restrict__ block_tables,
                                 int blocks_per_request, int block_tokens, int width,
                                 uint16_t* __restrict__ k_cache, uint16_t* __restrict__ v_cache) {
  const int64_t r = blockIdx.x;
  const int64_t p = pos[r];
  if (p < 0) return;
  const int32_t blk = block_tables[static_cast<int64_t>(req_ids[r]) * blocks_per_request +
                                   p / block_tokens];
  const int64_t phys = static_cast<int64_t>(blk) * block_tokens + p % block_tokens;
  for (int e = threadIdx.x; e < width; e += blockDim.x) {
    k_cache[phys * width + e] = k[r * k_row_stride + e];
    v_cache[phys * width + e] = v[r * v_row_stride + e];
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
    int heads, int kpool, uint64_t* __restrict__ keys_ws, int64_t ws_stride, float sqrt_dim) {
  __shared__ float qs[4 * 128];
  const int64_t r = blockIdx.y;
  const int64_t p = pos[r];
  if (p < 0) return;
  const int64_t visible = (p + 1) / kpool;
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
  for (int i = threadIdx.x; i < select_k; i += blockDim.x) {
    best_hi[i] = 0xFFFFFFFFu;
    best_lo[i] = 0xFFFFFFFFu;
  }
  __syncthreads();
  KeysRowFn fn{keys_ws + r * ws_stride};
  select_topk_stream(fn, 0, visible, best_hi, best_lo, tile_hi, tile_lo, select_k);
  __syncthreads();
  const int cnt = expand_from_best(best_hi, best_lo, select_k, p, kpool, max_selected,
                                   topk_out + r * max_selected, scratch, smem_count);
  if (threadIdx.x == 0) out_counts[r] = cnt;
}

// ---- listed GQA attention -----------------------------------------------------
constexpr int kQsaTile = 32;

template <int D>
__global__ void attn_partial_kernel(const uint16_t* __restrict__ q, int64_t q_row_stride,
                                    const uint16_t* __restrict__ k_cache,
                                    const uint16_t* __restrict__ v_cache,
                                    const int32_t* __restrict__ req_ids,
                                    const int32_t* __restrict__ topk, int topk_stride,
                                    const int32_t* __restrict__ counts, int n_split,
                                    int local_heads, int kv_heads, int block_tokens,
                                    const int32_t* __restrict__ block_tables,
                                    int blocks_per_request, float scale, float* __restrict__ m_ws,
                                    float* __restrict__ l_ws, float* __restrict__ c_ws) {
  constexpr int kGroups = 32;
  constexpr int kDslice = D / kGroups;  // 8 at D=256
  constexpr int kRowStride = D + 8;     // padded smem row (u16)
  const int64_t r = blockIdx.x;
  const int s = blockIdx.y;
  const int hpb = blockDim.x / 32;
  const int h0 = blockIdx.z * hpb;
  const int hl = threadIdx.x / 32;
  const int g = threadIdx.x % 32;
  const int h = h0 + hl;
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

  float m_reg = -INFINITY;
  float creg[kDslice];
#pragma unroll
  for (int i = 0; i < kDslice; ++i) creg[i] = 0.0f;
  float qreg[kDslice];
  {
    const uint16_t* qrow = q + r * q_row_stride + static_cast<int64_t>(h) * D + g * kDslice;
#pragma unroll
    for (int i = 0; i < kDslice; ++i) qreg[i] = bf16_bits_to_float(qrow[i]);
  }
  if (threadIdx.x < hpb) l[threadIdx.x] = 0.0f;
  __syncthreads();

  const int32_t* bt = block_tables + static_cast<int64_t>(req_ids[r]) * blocks_per_request;
  const int32_t* toks = topk + r * topk_stride;
  constexpr int kVecPerRow = D / 8;

  for (int t0 = t_begin; t0 < t_end; t0 += kQsaTile) {
    const int n = min(kQsaTile, t_end - t0);
    // Gather the tile's K and V rows (kv head kvh) as uint4s.
    for (int idx = threadIdx.x; idx < n * kVecPerRow * 2; idx += blockDim.x) {
      const int which = idx / (n * kVecPerRow);
      const int rem = idx - which * (n * kVecPerRow);
      const int tt = rem / kVecPerRow;
      const int c8 = rem - tt * kVecPerRow;
      const int64_t tok = toks[t0 + tt];
      const int32_t blk = bt[tok / block_tokens];
      const int64_t phys = static_cast<int64_t>(blk) * block_tokens + tok % block_tokens;
      const uint16_t* src = (which == 0 ? k_cache : v_cache) + phys * width + kvh * D + c8 * 8;
      uint16_t* dst = (which == 0 ? kt : vt) + tt * kRowStride + c8 * 8;
      *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
    }
    __syncthreads();
    float tile_max = -INFINITY;
    for (int tt = 0; tt < n; ++tt) {
      const uint16_t* krow = kt + tt * kRowStride + g * kDslice;
      float partial = 0.f;
#pragma unroll
      for (int i = 0; i < kDslice; ++i) partial += qreg[i] * bf16_bits_to_float(krow[i]);
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(~0u, partial, off);
      const float score = partial * scale;
      if (g == 0) scores[hl * scores_stride + tt] = score;
      tile_max = fmaxf(tile_max, score);
    }
    __syncthreads();
    const float m_new = fmaxf(m_reg, tile_max);
    const float rescale = expf(m_reg - m_new);
#pragma unroll
    for (int i = 0; i < kDslice; ++i) creg[i] *= rescale;
    if (g == 0) {
      float ladd = 0.0f;
      for (int tt = 0; tt < n; ++tt) ladd += expf(scores[hl * scores_stride + tt] - m_new);
      l[hl] = l[hl] * rescale + ladd;
    }
    for (int tt = 0; tt < n; ++tt) {
      const float p = round_bf16(expf(scores[hl * scores_stride + tt] - m_new));
      const uint16_t* vrow = vt + tt * kRowStride + g * kDslice;
#pragma unroll
      for (int i = 0; i < kDslice; ++i) creg[i] += p * bf16_bits_to_float(vrow[i]);
    }
    m_reg = m_new;
    __syncthreads();
  }
  if (g == 0) {
    m_ws[base_m + h] = m_reg;
    l_ws[base_m + h] = l[hl];
  }
  float* crow = c_ws + (base_m + h) * D + g * kDslice;
#pragma unroll
  for (int i = 0; i < kDslice; ++i) crow[i] = creg[i];
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
                   cudaStream_t stream) {
  if (rows <= 0) return;
  if (!k || !v || !req_ids || !pos || !block_tables || !k_cache || !v_cache)
    throw std::invalid_argument("qsa_kv_append: null pointer");
  kv_append_kernel<<<static_cast<unsigned>(rows), 256, 0, stream>>>(
      k, k_row_stride, v, v_row_stride, req_ids, pos, block_tables, blocks_per_request,
      block_tokens, kv_heads * dim, k_cache, v_cache);
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
                     cudaStream_t stream) {
  if (rows <= 0) return;
  if (!q || !req_ids || !pos || !block_tables || !index_cache || !keys_ws)
    throw std::invalid_argument("qsa_index_score: null pointer");
  if (heads < 1 || heads > 4 || dim != 128)
    throw std::invalid_argument("qsa_index_score: the lane layout is <= 4 heads x 128 dims");
  if (ws_stride > (int64_t(1) << kIdxBits))
    throw std::invalid_argument("qsa_index_score: pool ids must fit kIdxBits");
  if (rows > 65535) throw std::invalid_argument("qsa_index_score: too many rows per launch");
  const int64_t stripes = (ws_stride + kScorePoolsPerBlock - 1) / kScorePoolsPerBlock;
  if (stripes > 0x7fffffff) throw std::invalid_argument("qsa_index_score: too many pools");
  const dim3 grid(static_cast<unsigned>(stripes), static_cast<unsigned>(rows));
  index_score_kernel<<<grid, kScoreThreads, 0, stream>>>(
      q, q_row_stride, req_ids, pos, block_tables, blocks_per_request, index_cache,
      pools_per_block, heads, kpool, keys_ws, ws_stride, std::sqrt(static_cast<float>(dim)));
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
                      float* m_ws, float* l_ws, float* c_ws, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!q || !k_cache || !v_cache || !req_ids || !topk || !counts || !block_tables || !m_ws ||
      !l_ws || !c_ws)
    throw std::invalid_argument("qsa_attn_partial: null pointer");
  if (kv_heads <= 0 || local_heads % kv_heads != 0)
    throw std::invalid_argument("qsa_attn_partial: local_heads must be a multiple of kv_heads");
  const int heads_per_kv = local_heads / kv_heads;
  int hpb = 8;
  while (hpb > 1 && heads_per_kv % hpb != 0) --hpb;
  if (n_split <= 0) throw std::invalid_argument("qsa_attn_partial: n_split must be positive");
  const int row_stride = dim + 8;
  const size_t smem = static_cast<size_t>(2 * kQsaTile) * row_stride * 2 +
                      static_cast<size_t>(hpb) * (kQsaTile + 1) * 4 + static_cast<size_t>(hpb) * 4;
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(n_split),
                  static_cast<unsigned>(local_heads / hpb));
  const int threads = hpb * 32;
  switch (dim) {
    case 256:
      attn_partial_kernel<256><<<grid, threads, smem, stream>>>(
          q, q_row_stride, k_cache, v_cache, req_ids, topk, topk_stride, counts, n_split,
          local_heads, kv_heads, block_tokens, block_tables, blocks_per_request, scale, m_ws,
          l_ws, c_ws);
      break;
    case 512:
      attn_partial_kernel<512><<<grid, threads, smem, stream>>>(
          q, q_row_stride, k_cache, v_cache, req_ids, topk, topk_stride, counts, n_split,
          local_heads, kv_heads, block_tokens, block_tables, blocks_per_request, scale, m_ws,
          l_ws, c_ws);
      break;
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
