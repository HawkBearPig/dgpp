#include "kernels/dory_attn.hpp"
#include "kernels/pdl.cuh"

#include <cmath>
#include <stdexcept>
#include <string>

#include <cuda_bf16.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {

constexpr int D = kDoryHeadDim;
constexpr int kTile = kDoryAttnTile;
// K tile rows padded to an odd word count: a lane's word read of its own row
// lands in its own bank (the glm4 kernel's reason).
constexpr int kRowStride = D + 2;  // 258 bf16 = 129 words
constexpr int kPerLane = D / 32;   // 8 bf16 per lane: one 16-byte vector

__device__ __forceinline__ float round_bf16(float v) {
  return bf16_bits_to_float(float_to_bf16_bits(v));
}

__device__ __forceinline__ double warp_sum(double v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ uint32_t pack_bf16(uint16_t lo, uint16_t hi) {
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

// Eight fp32 values (already the model's bf16 values) as one 16-byte vector.
__device__ __forceinline__ uint4 pack8(const float* x) {
  uint32_t w[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    w[j] = pack_bf16(float_to_bf16_bits(x[2 * j]), float_to_bf16_bits(x[2 * j + 1]));
  return make_uint4(w[0], w[1], w[2], w[3]);
}

// ---- qkv finish ---------------------------------------------------------------
// One warp per (row, head); lane l owns dims [8l, 8l + 8). The full-width
// neoX RoPE pairs dim i with dim i + 128, so a pair spans two lanes 16 apart
// and the partner's values come through a shuffle. blockDim 256 = 8
// (row, head) items.
__global__ void __launch_bounds__(256) qkv_finish_kernel(
    const uint16_t* __restrict__ q_dot, int64_t q_stride, const uint16_t* __restrict__ k_dot,
    int64_t k_stride, const uint16_t* __restrict__ v_dot, int64_t v_stride,
    const uint16_t* __restrict__ sqk, float sqk_factor, const float* __restrict__ inv_freq,
    float eps, const int32_t* __restrict__ req_ids, const int64_t* __restrict__ pos, int rows,
    int local_heads, int kv_heads, const int32_t* __restrict__ block_tables,
    int blocks_per_request, int block_tokens, uint16_t* __restrict__ q_out,
    int64_t q_out_stride, uint16_t* __restrict__ k_cache, uint16_t* __restrict__ v_cache) {
  dgpp_pdl_wait();
  const int heads_total = local_heads + 2 * kv_heads;
  const int item = static_cast<int>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  if (item >= rows * heads_total) return;
  const int r = item / heads_total;
  const int hh = item - r * heads_total;
  const int64_t p = pos[r];
  if (p < 0) return;  // a padding row writes nothing
  const int lane = threadIdx.x % 32;
  const int d0 = lane * kPerLane;
  int kind, h;  // 0 = q, 1 = k, 2 = v
  if (hh < local_heads) {
    kind = 0;
    h = hh;
  } else if (hh < local_heads + kv_heads) {
    kind = 1;
    h = hh - local_heads;
  } else {
    kind = 2;
    h = hh - local_heads - kv_heads;
  }
  const uint16_t* src = kind == 0 ? q_dot + static_cast<int64_t>(r) * q_stride + h * D
                                  : kind == 1 ? k_dot + static_cast<int64_t>(r) * k_stride + h * D
                                              : v_dot + static_cast<int64_t>(r) * v_stride + h * D;
  // The Linear's output is bf16 (the reference's dtype): widen it exactly.
  float x[kPerLane];
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) x[i] = bf16_bits_to_float(src[d0 + i]);

  if (kind != 2) {
    // The full-width neoX RoPE: every dim is rotary (partial_rotary_factor
    // 1.0), pair i = (i, i + D/2): x1' = x1 c - x2 s, x2' = x2 c + x1 s with
    // the reference's bf16 ops.
    constexpr int kHalf = D / 2;
    constexpr int kLaneOff = kHalf / kPerLane;  // 16
    float mate[kPerLane];
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) mate[j] = __shfl_xor_sync(0xffffffffu, x[j], kLaneOff);
    const bool first = d0 < kHalf;
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
      const int i = (first ? d0 : d0 - kHalf) + j;  // the pair index
      const float ang = __fmul_rn(static_cast<float>(p), inv_freq[i]);
      const float c = round_bf16(cosf(ang));
      const float s = round_bf16(sinf(ang));
      x[j] = first ? round_bf16(round_bf16(x[j] * c) + round_bf16(-mate[j] * s))
                   : round_bf16(round_bf16(x[j] * c) + round_bf16(mate[j] * s));
    }
    // The head's justnorm over the 256 dims, in the kernel family's form: an
    // exact fp64 sum (every bf16 square is exact there), so `float(sum) +
    // eps` rounds once.
    double ss0 = 0.0, ss1 = 0.0;
#pragma unroll
    for (int i = 0; i < kPerLane; i += 2) {
      ss0 = fma(static_cast<double>(x[i]), static_cast<double>(x[i]), ss0);
      ss1 = fma(static_cast<double>(x[i + 1]), static_cast<double>(x[i + 1]), ss1);
    }
    const float rstd = 1.0f / sqrtf(static_cast<float>(warp_sum(ss0 + ss1)) + eps);
    if (kind == 0) {
      // q = bf16( fp32(justnorm(q)) · m ), m = (fp32(sqk[i]) · sqk_factor)²
      // — the square after the RoPE, in fp32 (plan §1.3).
#pragma unroll
      for (int j = 0; j < kPerLane; ++j) {
        const float m = bf16_bits_to_float(sqk[h * D + d0 + j]) * sqk_factor;
        const float u = round_bf16(x[j] * rstd);
        x[j] = round_bf16(u * (m * m));
      }
      *reinterpret_cast<uint4*>(q_out + static_cast<int64_t>(r) * q_out_stride + h * D + d0) =
          pack8(x);
      return;
    }
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) x[j] = round_bf16(x[j] * rstd);
  }
  const int32_t blk = block_tables[static_cast<int64_t>(req_ids[r]) * blocks_per_request +
                                   p / block_tokens];
  const int64_t phys = static_cast<int64_t>(blk) * block_tokens + p % block_tokens;
  uint16_t* cache = kind == 1 ? k_cache : v_cache;
  *reinterpret_cast<uint4*>(cache + phys * (static_cast<int64_t>(kv_heads) * D) + h * D + d0) = pack8(x);
}

// ---- the split-KV paged GQA attention ------------------------------------------
// Block: one warp per query head of the kv head (hpk warps), lane t owns
// tile token t for the scores and dims [8l, 8l + 8) for the V accumulation.
__global__ void attn_partial_kernel(const uint16_t* __restrict__ q, int64_t q_stride,
                                    const uint16_t* __restrict__ k_cache,
                                    const uint16_t* __restrict__ v_cache,
                                    const int32_t* __restrict__ req_ids,
                                    const int64_t* __restrict__ pos, int n_split,
                                    int local_heads, int kv_heads, int block_tokens,
                                    const int32_t* __restrict__ block_tables,
                                    int blocks_per_request, int window, float scale,
                                    float* __restrict__ m_ws, float* __restrict__ l_ws,
                                    float* __restrict__ c_ws) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint16_t smem[];
  const int hpk = local_heads / kv_heads;
  const int r = static_cast<int>(blockIdx.x);
  const int s = static_cast<int>(blockIdx.y);
  const int kvh = static_cast<int>(blockIdx.z);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int h = kvh * hpk + warp;  // this warp's query head
  const int64_t base_m = (static_cast<int64_t>(r) * n_split + s) * local_heads;
  uint16_t* kt = smem;                                     // [kTile][kRowStride]
  uint16_t* vt = kt + kTile * kRowStride;                  // [kTile][D]
  float* qs = reinterpret_cast<float*>(vt + kTile * D);    // [hpk][D] the heads' queries, fp32
  float* pbuf = qs + hpk * D;                              // [hpk][kTile] the rounded probabilities

  // The row's visible range: the window's last tokens, or everything.
  const int64_t p = pos[r];
  int t_begin = 0, t_end = 0;
  if (p >= 0) {
    const int64_t visible_end = p + 1;
    const int64_t first = (window > 0 && visible_end > window) ? visible_end - window : 0;
    const int64_t visible = visible_end - first;
    const int64_t tiles = (visible + kTile - 1) / kTile;
    const int64_t chunk = (tiles + n_split - 1) / n_split;
    const int64_t tile0 = static_cast<int64_t>(s) * chunk;
    const int64_t tile1 = tile0 + chunk < tiles ? tile0 + chunk : tiles;
    if (tile0 < tile1) {
      t_begin = static_cast<int>(first + tile0 * kTile);
      const int64_t end = first + tile1 * kTile < visible_end ? first + tile1 * kTile : visible_end;
      t_end = static_cast<int>(end);
    }
  }
  if (t_begin >= t_end) {
    // An empty split (or a padding row): the neutral partial.
    for (int i = threadIdx.x; i < hpk * D; i += blockDim.x)
      c_ws[(base_m + kvh * hpk + i / D) * D + i % D] = 0.0f;
    if (threadIdx.x < hpk) {
      m_ws[base_m + kvh * hpk + threadIdx.x] = -INFINITY;
      l_ws[base_m + kvh * hpk + threadIdx.x] = 0.0f;
    }
    return;
  }
  // The head's query into fp32 smem (every lane reads every dim later).
  {
    const uint16_t* qrow = q + static_cast<int64_t>(r) * q_stride + static_cast<int64_t>(h) * D;
    for (int d = lane; d < D; d += 32) qs[warp * D + d] = bf16_bits_to_float(qrow[d]);
  }
  __syncthreads();
  const int32_t* bt = block_tables + static_cast<int64_t>(req_ids[r]) * blocks_per_request;
  const int width = kv_heads * D;
  float m_run = -INFINITY, l_run = 0.0f;
  float creg[kPerLane];
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) creg[i] = 0.0f;
  const float* myq = qs + warp * D;
  for (int t0 = t_begin; t0 < t_end; t0 += kTile) {
    const int n = min(kTile, t_end - t0);
    // Stage the tile's K and V rows of this kv head: 16-byte pieces.
    __syncthreads();  // the previous tile's readers are done
    for (int idx = threadIdx.x; idx < n * (D / 8) * 2; idx += blockDim.x) {
      const int which = idx / (n * (D / 8));
      const int rem = idx - which * (n * (D / 8));
      const int tt = rem / (D / 8);
      const int c8 = rem - tt * (D / 8);
      const int64_t tok = t0 + tt;
      const int32_t blk = bt[tok / block_tokens];
      const int64_t phys = static_cast<int64_t>(blk) * block_tokens + tok % block_tokens;
      const uint16_t* src = (which == 0 ? k_cache : v_cache) + phys * width + kvh * D + c8 * 8;
      const uint4 val = *reinterpret_cast<const uint4*>(src);
      if (which == 0) {
        // The padded K row: 8 elements as four 32-bit words.
        uint32_t* dst = reinterpret_cast<uint32_t*>(kt + tt * kRowStride) + c8 * 4;
        dst[0] = val.x;
        dst[1] = val.y;
        dst[2] = val.z;
        dst[3] = val.w;
      } else {
        *reinterpret_cast<uint4*>(vt + tt * D + c8 * 8) = val;
      }
    }
    __syncthreads();
    // Scores: lane t against tile token t (a fixed fp32 order over d).
    float score = -INFINITY;
    if (lane < n) {
      const uint32_t* krow = reinterpret_cast<const uint32_t*>(kt + lane * kRowStride);
      float acc = 0.f;
#pragma unroll 8
      for (int w = 0; w < D / 2; ++w) {
        const uint32_t kk = krow[w];
        acc = fmaf(myq[2 * w], bf16_bits_to_float(static_cast<uint16_t>(kk & 0xFFFFu)), acc);
        acc = fmaf(myq[2 * w + 1], bf16_bits_to_float(static_cast<uint16_t>(kk >> 16)), acc);
      }
      score = acc * scale;
    }
    const float tile_max = warp_max(score);
    const float m_new = fmaxf(m_run, tile_max);
    const float rescale = expf(m_run - m_new);  // 0 when m_run is -inf
    const float e = lane < n ? expf(score - m_new) : 0.f;
    l_run = l_run * rescale + warp_sum(e);
    pbuf[warp * kTile + lane] = round_bf16(e);
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) creg[i] *= rescale;
    __syncwarp();
    // The V accumulation: lane owns dims [8l, 8l + 8).
    for (int tt = 0; tt < n; ++tt) {
      const float pt = pbuf[warp * kTile + tt];
      const uint4 vv = *reinterpret_cast<const uint4*>(vt + tt * D + lane * kPerLane);
      creg[0] = fmaf(pt, bf16_bits_to_float(static_cast<uint16_t>(vv.x & 0xFFFFu)), creg[0]);
      creg[1] = fmaf(pt, bf16_bits_to_float(static_cast<uint16_t>(vv.x >> 16)), creg[1]);
      creg[2] = fmaf(pt, bf16_bits_to_float(static_cast<uint16_t>(vv.y & 0xFFFFu)), creg[2]);
      creg[3] = fmaf(pt, bf16_bits_to_float(static_cast<uint16_t>(vv.y >> 16)), creg[3]);
      creg[4] = fmaf(pt, bf16_bits_to_float(static_cast<uint16_t>(vv.z & 0xFFFFu)), creg[4]);
      creg[5] = fmaf(pt, bf16_bits_to_float(static_cast<uint16_t>(vv.z >> 16)), creg[5]);
      creg[6] = fmaf(pt, bf16_bits_to_float(static_cast<uint16_t>(vv.w & 0xFFFFu)), creg[6]);
      creg[7] = fmaf(pt, bf16_bits_to_float(static_cast<uint16_t>(vv.w >> 16)), creg[7]);
    }
    m_run = m_new;
    __syncwarp();
  }
  if (lane == 0) {
    m_ws[base_m + h] = m_run;
    l_ws[base_m + h] = l_run;
  }
  float* crow = c_ws + (base_m + h) * D + lane * kPerLane;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) crow[i] = creg[i];
}

__global__ void attn_combine_kernel(const float* __restrict__ m_ws, const float* __restrict__ l_ws,
                                    const float* __restrict__ c_ws, int n_split, int local_heads,
                                    uint16_t* __restrict__ out) {
  dgpp_pdl_wait();
  const int r = static_cast<int>(blockIdx.x);
  const int h = static_cast<int>(blockIdx.y);
  const int d = threadIdx.x;  // D threads
  const int64_t base = (static_cast<int64_t>(r) * n_split) * local_heads + h;
  float M = -INFINITY;
  for (int s = 0; s < n_split; ++s) M = fmaxf(M, m_ws[base + static_cast<int64_t>(s) * local_heads]);
  float L = 0.f, C = 0.f;
  if (M > -INFINITY) {
    for (int s = 0; s < n_split; ++s) {
      const int64_t i = base + static_cast<int64_t>(s) * local_heads;
      const float w = expf(m_ws[i] - M);
      L = fmaf(l_ws[i], w, L);
      C = fmaf(c_ws[i * D + d], w, C);
    }
  }
  out[(static_cast<int64_t>(r) * local_heads + h) * D + d] =
      float_to_bf16_bits(L > 0.f ? C / L : 0.f);
}

}  // namespace

void dory_qkv_finish(const uint16_t* q_dot, int64_t q_stride, const uint16_t* k_dot,
                     int64_t k_stride, const uint16_t* v_dot, int64_t v_stride,
                     const uint16_t* sqk, float sqk_factor,
                     const float* inv_freq, float eps, const int32_t* req_ids,
                     const int64_t* pos, int rows, int local_heads, int kv_heads,
                     const int32_t* block_tables, int blocks_per_request, int block_tokens,
                     uint16_t* q_out, int64_t q_out_stride, uint16_t* k_cache,
                     uint16_t* v_cache, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!q_dot || !k_dot || !v_dot || !sqk || !inv_freq || !req_ids || !pos || !block_tables ||
      !q_out || !k_cache || !v_cache)
    throw std::invalid_argument("dory_qkv_finish: null pointer");
  if (local_heads <= 0 || kv_heads <= 0 || local_heads % kv_heads != 0)
    throw std::invalid_argument("dory_qkv_finish: heads");
  if (!(q_out_stride >= static_cast<int64_t>(local_heads) * D))
    throw std::invalid_argument("dory_qkv_finish: q_out_stride too small");
  if (block_tokens <= 0 || block_tokens % kTile != 0)
    throw std::invalid_argument("dory_qkv_finish: block_tokens must be a multiple of the 32-token tile");
  if (!(eps > 0.0f)) throw std::invalid_argument("dory_qkv_finish: eps");
  const int items = rows * (local_heads + 2 * kv_heads);
  const int blocks = (items + 7) / 8;
  qkv_finish_kernel<<<blocks, 256, 0, stream>>>(
      q_dot, q_stride, k_dot, k_stride, v_dot, v_stride, sqk, sqk_factor, inv_freq, eps, req_ids,
      pos, rows, local_heads, kv_heads, block_tables, blocks_per_request, block_tokens, q_out,
      q_out_stride, k_cache, v_cache);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dory_attn_partial(const uint16_t* q, int64_t q_stride, const uint16_t* k_cache,
                       const uint16_t* v_cache, const int32_t* req_ids, const int64_t* pos,
                       int rows, int n_split, int local_heads, int kv_heads, int block_tokens,
                       const int32_t* block_tables, int blocks_per_request, int window,
                       float scale, float* m_ws, float* l_ws, float* c_ws, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!q || !k_cache || !v_cache || !req_ids || !pos || !block_tables || !m_ws || !l_ws || !c_ws)
    throw std::invalid_argument("dory_attn_partial: null pointer");
  if (local_heads <= 0 || kv_heads <= 0 || local_heads % kv_heads != 0)
    throw std::invalid_argument("dory_attn_partial: heads");
  if (n_split <= 0) throw std::invalid_argument("dory_attn_partial: n_split");
  if (block_tokens <= 0 || block_tokens % kTile != 0)
    throw std::invalid_argument("dory_attn_partial: block_tokens must be a multiple of the 32-token tile");
  const int hpk = local_heads / kv_heads;
  if (hpk > 32) throw std::invalid_argument("dory_attn_partial: at most 32 query heads per kv head");
  const size_t smem = static_cast<size_t>(kTile) * kRowStride * 2 + static_cast<size_t>(kTile) * D * 2 +
                      static_cast<size_t>(hpk) * D * 4 + static_cast<size_t>(hpk) * kTile * 4;
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(n_split),
                  static_cast<unsigned>(kv_heads));
  attn_partial_kernel<<<grid, hpk * 32, smem, stream>>>(
      q, q_stride, k_cache, v_cache, req_ids, pos, n_split, local_heads, kv_heads, block_tokens,
      block_tables, blocks_per_request, window, scale, m_ws, l_ws, c_ws);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dory_attn_combine(const float* m_ws, const float* l_ws, const float* c_ws, int rows,
                       int n_split, int local_heads, uint16_t* out, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!m_ws || !l_ws || !c_ws || !out) throw std::invalid_argument("dory_attn_combine: null pointer");
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(local_heads));
  attn_combine_kernel<<<grid, D, 0, stream>>>(m_ws, l_ws, c_ws, n_split, local_heads, out);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
