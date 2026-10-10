// Prefill listed attention: reuse a KV tile across up to twelve query heads
// and evaluate its probabilities cooperatively across each warp. The split,
// tile, dot-product, sum and BF16 probability rounding orders match decode.
// A separate translation unit leaves the existing decode kernel bodies intact.
#include "kernels/qsa.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/latent_format.hpp"

namespace dgpp {
namespace {
__device__ __forceinline__ float round_bf16(float v) {
  return bf16_bits_to_float(float_to_bf16_bits(v));
}
// ---- listed GQA attention -----------------------------------------------------
constexpr int kQsaTile = 32;

template <int D>
__global__ void attn_prefill_partial_kernel(const uint16_t* __restrict__ q, int64_t q_row_stride,
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
    if (k_scale == nullptr) {
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
    } else {
      // fp8 cache: vectorized gather (16 codes per 16-byte load) of the e4m3
      // codes + per-(row, kv-head) scale, dequantized to bf16 in the smem
      // tiles; the math below is unchanged (reads bf16).
      const uint8_t* k8 = reinterpret_cast<const uint8_t*>(k_cache);
      const uint8_t* v8 = reinterpret_cast<const uint8_t*>(v_cache);
      constexpr int kFp8ChunksPerRow = D / 16;
      for (int idx = threadIdx.x; idx < n * kFp8ChunksPerRow * 2; idx += blockDim.x) {
        const int which = idx / (n * kFp8ChunksPerRow);
        const int rem = idx - which * (n * kFp8ChunksPerRow);
        const int tt = rem / kFp8ChunksPerRow;
        const int c = (rem - tt * kFp8ChunksPerRow) * 16;
        const int64_t tok = toks[t0 + tt];
        const int32_t blk = bt[tok / block_tokens];
        const int64_t phys = static_cast<int64_t>(blk) * block_tokens + tok % block_tokens;
        const uint8_t* src = (which == 0 ? k8 : v8) + phys * width + kvh * D + c;
        const float sc = (which == 0 ? k_scale : v_scale)[phys * kv_heads + kvh];
        uint16_t* dst = (which == 0 ? kt : vt) + tt * kRowStride + c;
        const uint4 v4 = *reinterpret_cast<const uint4*>(src);
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&v4);
#pragma unroll
        for (int j = 0; j < 16; ++j) dst[j] = latent_fp8_decode_bf16(b[j], sc);
      }
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
    // Each lane evaluates one probability. Broadcast in the original token
    // order so both the denominator and P*V keep the existing FP32 chain.
    const float e = g < n ? expf(scores[hl * scores_stride + g] - m_new) : 0.0f;
    const float p_reg = round_bf16(e);
    float ladd = 0.0f;
    for (int tt = 0; tt < n; ++tt) {
      ladd += __shfl_sync(~0u, e, tt);
      const float p = __shfl_sync(~0u, p_reg, tt);
      const uint16_t* vrow = vt + tt * kRowStride + g * kDslice;
#pragma unroll
      for (int i = 0; i < kDslice; ++i) creg[i] += p * bf16_bits_to_float(vrow[i]);
    }
    if (g == 0) l[hl] = l[hl] * rescale + ladd;
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

}  // namespace

void qsa_attn_prefill_partial(const uint16_t* q, int64_t q_row_stride, const uint16_t* k_cache,
                      const uint16_t* v_cache, const int32_t* req_ids, const int32_t* topk,
                      int topk_stride, const int32_t* counts, int rows, int n_split,
                      int local_heads, int kv_heads, int dim, int block_tokens,
                      const int32_t* block_tables, int blocks_per_request, float scale,
                      float* m_ws, float* l_ws, float* c_ws, cudaStream_t stream,
                      const float* k_scale, const float* v_scale) {
  if (rows <= 0) return;
  if (dim != 256) {
    qsa_attn_partial(q, q_row_stride, k_cache, v_cache, req_ids, topk, topk_stride, counts,
                     rows, n_split, local_heads, kv_heads, dim, block_tokens, block_tables,
                     blocks_per_request, scale, m_ws, l_ws, c_ws, stream, k_scale, v_scale);
    return;
  }
  if (!q || !k_cache || !v_cache || !req_ids || !topk || !counts || !block_tables || !m_ws ||
      !l_ws || !c_ws)
    throw std::invalid_argument("qsa_attn_prefill_partial: null pointer");
  if (local_heads <= 0 || kv_heads <= 0 || local_heads % kv_heads != 0)
    throw std::invalid_argument("qsa_attn_prefill_partial: local_heads must be a multiple of kv_heads");
  const int heads_per_kv = local_heads / kv_heads;
  // TP1/TP2 have twelve query heads per KV head. Six warps per block
  // occupy the same ~34 KiB tile but allow only 25% occupancy on GB10;
  // twelve warps double occupancy and avoid loading that tile twice.
  int hpb = std::min(12, heads_per_kv);
  while (hpb > 1 && heads_per_kv % hpb != 0) --hpb;
  if (n_split <= 0) throw std::invalid_argument("qsa_attn_prefill_partial: n_split must be positive");
  const int row_stride = dim + 8;
  const size_t smem = static_cast<size_t>(2 * kQsaTile) * row_stride * 2 +
                      static_cast<size_t>(hpb) * (kQsaTile + 1) * 4 + static_cast<size_t>(hpb) * 4;
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(n_split),
                  static_cast<unsigned>(local_heads / hpb));
  const int threads = hpb * 32;
  attn_prefill_partial_kernel<256><<<grid, threads, smem, stream>>>(
      q, q_row_stride, k_cache, v_cache, k_scale, v_scale, req_ids, topk, topk_stride, counts, n_split,
      local_heads, kv_heads, block_tokens, block_tables, blocks_per_request, scale, m_ws,
      l_ws, c_ws);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
