#include "kernels/glm4_attn.hpp"
#include "kernels/pdl.cuh"

#include <cmath>
#include <stdexcept>

#include <cuda_bf16.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {

constexpr int D = kGlm4HeadDim;
constexpr int kTile = kGlm4AttnTile;

__device__ __forceinline__ float round_bf16(float v) {
  return bf16_bits_to_float(float_to_bf16_bits(v));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(~0u, v, off);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, off));
  return v;
}

// ---- qkv finish ---------------------------------------------------------------
// One warp per (row, head); lane l owns dims [4l, 4l + 4). The partial
// RoPE pairs dim i with dim i + rotary_dim/2 (transformers' rotate_half:
// x1 = the first half, x2 = the second — not the interleaved form of the
// older glm / glm4 architectures, 2026-09-10), so a pair spans two lanes
// rotary_dim/8 apart and the partner's values come through a shuffle.
// blockDim 256 = 8 (row, head) items.
__global__ void qkv_finish_kernel(const float* __restrict__ q_dot, int64_t q_stride,
                                  const float* __restrict__ k_dot, int64_t k_stride,
                                  const float* __restrict__ v_dot, int64_t v_stride,
                                  const uint16_t* __restrict__ q_bias,
                                  const uint16_t* __restrict__ k_bias,
                                  const uint16_t* __restrict__ v_bias,
                                  const uint16_t* __restrict__ q_norm,
                                  const uint16_t* __restrict__ k_norm, float eps,
                                  const float* __restrict__ inv_freq, int rotary_dim,
                                  const int32_t* __restrict__ req_ids,
                                  const int64_t* __restrict__ pos, int rows, int local_heads,
                                  int kv_heads, const int32_t* __restrict__ block_tables,
                                  int blocks_per_request, int block_tokens,
                                  uint16_t* __restrict__ q_out, int64_t q_out_stride,
                                  uint16_t* __restrict__ k_cache, uint16_t* __restrict__ v_cache) {
  dgpp_pdl_wait();
  const int heads_total = local_heads + 2 * kv_heads;
  const int item = static_cast<int>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  if (item >= rows * heads_total) return;
  const int r = item / heads_total;
  const int hh = item - r * heads_total;
  const int64_t p = pos[r];
  if (p < 0) return;
  const int lane = threadIdx.x % 32;
  const int d0 = lane * 4;
  // Which projection, which head, the source, the bias, the norm, the sink.
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
  const float* src = kind == 0 ? q_dot + static_cast<int64_t>(r) * q_stride + h * D
                     : kind == 1 ? k_dot + static_cast<int64_t>(r) * k_stride + h * D
                                 : v_dot + static_cast<int64_t>(r) * v_stride + h * D;
  const uint16_t* bias = kind == 0 ? q_bias : kind == 1 ? k_bias : v_bias;
  const uint16_t* norm = kind == 0 ? q_norm : kind == 1 ? k_norm : nullptr;
  // The Linear's rounding: bf16(dot + bias), the bias added in fp32.
  float x[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float v = src[d0 + i];
    if (bias) v += bf16_bits_to_float(bias[h * D + d0 + i]);
    x[i] = round_bf16(v);
  }
  if (norm != nullptr) {
    // The two-rounding GLM norm over the head: fp32 sum of squares (a
    // fixed warp tree), u = bf16(x * rstd), y = bf16(w * u).
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) ss += x[i] * x[i];
    ss = warp_sum(ss);
    const float rstd = rsqrtf(ss / static_cast<float>(D) + eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float u = round_bf16(x[i] * rstd);
      x[i] = round_bf16(bf16_bits_to_float(norm[d0 + i]) * u);
    }
    // RoPE on dims [0, rotary_dim): pair i = (i, i + half) rotated by the
    // angle pos x inv_freq[i] — x1' = x1 c - x2 s, x2' = x2 c + x1 s with
    // the reference's bf16 ops. The partner lane sits half/4 lanes away;
    // every lane shuffles (uniform), the rotary lanes use the values.
    if (rotary_dim > 0) {
      const int half = rotary_dim / 2;
      const int lane_off = half / 4;
      float mate[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) mate[j] = __shfl_xor_sync(~0u, x[j], lane_off);
      if (d0 < rotary_dim) {
        const bool first = d0 < half;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int i = (first ? d0 : d0 - half) + j;  // the pair index
          const float ang = __fmul_rn(static_cast<float>(p), inv_freq[i]);
          const float c = round_bf16(cosf(ang));
          const float s = round_bf16(sinf(ang));
          x[j] = first ? round_bf16(round_bf16(x[j] * c) + round_bf16(-mate[j] * s))
                       : round_bf16(round_bf16(x[j] * c) + round_bf16(mate[j] * s));
        }
      }
    }
  }
  uint16_t packed[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) packed[i] = float_to_bf16_bits(x[i]);
  uint2 word;
  word.x = static_cast<uint32_t>(packed[0]) | (static_cast<uint32_t>(packed[1]) << 16);
  word.y = static_cast<uint32_t>(packed[2]) | (static_cast<uint32_t>(packed[3]) << 16);
  if (kind == 0) {
    *reinterpret_cast<uint2*>(q_out + static_cast<int64_t>(r) * q_out_stride + h * D + d0) = word;
    return;
  }
  const int32_t blk = block_tables[static_cast<int64_t>(req_ids[r]) * blocks_per_request + p / block_tokens];
  const int64_t phys = static_cast<int64_t>(blk) * block_tokens + p % block_tokens;
  uint16_t* cache = kind == 1 ? k_cache : v_cache;
  *reinterpret_cast<uint2*>(cache + phys * (static_cast<int64_t>(kv_heads) * D) + h * D + d0) = word;
}

// ---- the split-KV paged GQA attention ------------------------------------------
// Block: one warp per query head of the kv head (hpk warps), lane t owns
// tile token t for the scores and dims [4l, 4l + 4) for the V accumulation.
// K tile rows are padded to kRowStride elements (65 words: a lane's word
// read of its own row lands in its own bank).
constexpr int kRowStride = D + 2;  // 130 bf16 = 65 words

__global__ void attn_partial_kernel(const uint16_t* __restrict__ q, int64_t q_stride,
                                    const uint16_t* __restrict__ k_cache,
                                    const uint16_t* __restrict__ v_cache,
                                    const int32_t* __restrict__ req_ids,
                                    const int64_t* __restrict__ pos, int n_split,
                                    int local_heads, int kv_heads, int block_tokens,
                                    const int32_t* __restrict__ block_tables,
                                    int blocks_per_request, float scale,
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
  uint16_t* kt = smem;                       // [kTile][kRowStride]
  uint16_t* vt = kt + kTile * kRowStride;    // [kTile][D]
  float* qs = reinterpret_cast<float*>(vt + kTile * D);  // [hpk][D] the heads' queries, fp32
  float* pbuf = qs + hpk * D;                // [hpk][kTile] the tile's bf16-rounded probabilities

  const int64_t p = pos[r];
  int t_begin = 0, t_end = 0;
  if (p >= 0) {
    const int64_t visible = p + 1;
    const int64_t tiles = (visible + kTile - 1) / kTile;
    const int64_t chunk = (tiles + n_split - 1) / n_split;
    const int64_t tile0 = static_cast<int64_t>(s) * chunk;
    const int64_t tile1 = tile0 + chunk < tiles ? tile0 + chunk : tiles;
    if (tile0 < tile1) {
      t_begin = static_cast<int>(tile0 * kTile);
      t_end = static_cast<int>(tile1 * kTile < visible ? tile1 * kTile : visible);
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
  float creg[4] = {0.f, 0.f, 0.f, 0.f};
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
    for (int i = 0; i < 4; ++i) creg[i] *= rescale;
    __syncwarp();
    // The V accumulation: lane owns dims [4l, 4l + 4).
    for (int tt = 0; tt < n; ++tt) {
  dgpp_pdl_wait();
      const float pt = pbuf[warp * kTile + tt];
      const uint2 vv = *reinterpret_cast<const uint2*>(vt + tt * D + lane * 4);
      creg[0] = fmaf(pt, bf16_bits_to_float(static_cast<uint16_t>(vv.x & 0xFFFFu)), creg[0]);
      creg[1] = fmaf(pt, bf16_bits_to_float(static_cast<uint16_t>(vv.x >> 16)), creg[1]);
      creg[2] = fmaf(pt, bf16_bits_to_float(static_cast<uint16_t>(vv.y & 0xFFFFu)), creg[2]);
      creg[3] = fmaf(pt, bf16_bits_to_float(static_cast<uint16_t>(vv.y >> 16)), creg[3]);
    }
    m_run = m_new;
    __syncwarp();
  }
  if (lane == 0) {
    m_ws[base_m + h] = m_run;
    l_ws[base_m + h] = l_run;
  }
  float* crow = c_ws + (base_m + h) * D + lane * 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) crow[i] = creg[i];
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

void glm4_qkv_finish(const float* q_dot, int64_t q_stride, const float* k_dot, int64_t k_stride,
                     const float* v_dot, int64_t v_stride, const uint16_t* q_bias,
                     const uint16_t* k_bias, const uint16_t* v_bias, const uint16_t* q_norm,
                     const uint16_t* k_norm, float eps, const float* inv_freq, int rotary_dim,
                     const int32_t* req_ids, const int64_t* pos, int rows, int local_heads,
                     int kv_heads, const int32_t* block_tables, int blocks_per_request,
                     int block_tokens, uint16_t* q_out, int64_t q_out_stride, uint16_t* k_cache,
                     uint16_t* v_cache, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!q_dot || !k_dot || !v_dot || !req_ids || !pos || !block_tables || !q_out || !k_cache || !v_cache)
    throw std::invalid_argument("glm4_qkv_finish: null pointer");
  if (local_heads <= 0 || kv_heads <= 0 || local_heads % kv_heads != 0)
    throw std::invalid_argument("glm4_qkv_finish: heads");
  if (rotary_dim < 0 || rotary_dim > D || rotary_dim % 8 != 0 || (rotary_dim > 0 && !inv_freq))
    throw std::invalid_argument("glm4_qkv_finish: rotary_dim (a multiple of 8, at most head_dim)");
  if ((q_norm == nullptr) != (k_norm == nullptr))
    throw std::invalid_argument("glm4_qkv_finish: q and k norms come together");
  if (q_norm == nullptr && rotary_dim > 0)
    throw std::invalid_argument("glm4_qkv_finish: RoPE is applied after the head norms (both or neither)");
  const int items = rows * (local_heads + 2 * kv_heads);
  const int blocks = (items + 7) / 8;
  qkv_finish_kernel<<<blocks, 256, 0, stream>>>(
      q_dot, q_stride, k_dot, k_stride, v_dot, v_stride, q_bias, k_bias, v_bias, q_norm, k_norm, eps,
      inv_freq, rotary_dim, req_ids, pos, rows, local_heads, kv_heads, block_tables,
      blocks_per_request, block_tokens, q_out, q_out_stride, k_cache, v_cache);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm4_attn_partial(const uint16_t* q, int64_t q_stride, const uint16_t* k_cache,
                       const uint16_t* v_cache, const int32_t* req_ids, const int64_t* pos,
                       int rows, int n_split, int local_heads, int kv_heads, int block_tokens,
                       const int32_t* block_tables, int blocks_per_request, float scale,
                       float* m_ws, float* l_ws, float* c_ws, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!q || !k_cache || !v_cache || !req_ids || !pos || !block_tables || !m_ws || !l_ws || !c_ws)
    throw std::invalid_argument("glm4_attn_partial: null pointer");
  if (local_heads <= 0 || kv_heads <= 0 || local_heads % kv_heads != 0)
    throw std::invalid_argument("glm4_attn_partial: heads");
  if (n_split <= 0) throw std::invalid_argument("glm4_attn_partial: n_split");
  if (block_tokens <= 0 || block_tokens % kTile != 0)
    throw std::invalid_argument("glm4_attn_partial: block_tokens must be a multiple of the 32-token tile");
  const int hpk = local_heads / kv_heads;
  if (hpk > 32) throw std::invalid_argument("glm4_attn_partial: at most 32 query heads per kv head");
  const size_t smem = static_cast<size_t>(kTile) * kRowStride * 2 + static_cast<size_t>(kTile) * D * 2 +
                      static_cast<size_t>(hpk) * D * 4 + static_cast<size_t>(hpk) * kTile * 4;
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(n_split),
                  static_cast<unsigned>(kv_heads));
  attn_partial_kernel<<<grid, hpk * 32, smem, stream>>>(
      q, q_stride, k_cache, v_cache, req_ids, pos, n_split, local_heads, kv_heads, block_tokens,
      block_tables, blocks_per_request, scale, m_ws, l_ws, c_ws);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm4_attn_combine(const float* m_ws, const float* l_ws, const float* c_ws, int rows,
                       int n_split, int local_heads, uint16_t* out, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!m_ws || !l_ws || !c_ws || !out) throw std::invalid_argument("glm4_attn_combine: null pointer");
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(local_heads));
  attn_combine_kernel<<<grid, D, 0, stream>>>(m_ws, l_ws, c_ws, n_split, local_heads, out);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
