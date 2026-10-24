#include "kernels/csa2.hpp"
#include "kernels/pdl.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/rope_scaling.hpp"
#include "kernels/topk_select.cuh"

namespace dgpp {
namespace {

constexpr int kCsa2Threads = 256;
constexpr uint64_t kCsa2KeyMax = ~0ull;

// Exact hardware conversions (dsa.cu's pin: e4m3 fits f16, bf16 fits f32).
__device__ __forceinline__ float2 e4m3x2_to_float2(uint16_t v) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(v, __NV_E4M3);
  return __half22float2(__half2(h));
}

// Block-wide sum in a fixed order (warp shuffles then one warp over the
// warp sums): deterministic, launch-shape independent for a given
// blockDim.
__device__ __forceinline__ float block_sum_256(float v, float* red /*[8]*/) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __syncthreads();
  if (lane == 0) red[warp] = v;
  __syncthreads();
  float t = 0.0f;
  if (warp == 0) {
    t = lane < (int(blockDim.x) >> 5) ? red[lane] : 0.0f;
    for (int off = 4; off > 0; off >>= 1) t += __shfl_xor_sync(0xffffffffu, t, off);
    if (lane == 0) red[0] = t;
  }
  __syncthreads();
  t = red[0];
  __syncthreads();
  return t;
}
// ---- RMSNorm ----------------------------------------------------------------------
// One block per row; each thread owns the elements t, t + 256, ...; the
// weight is applied in fp32 after the fp32 normalization (two fp32
// products, the reference's `x * rsqrt(var + eps)` then `weight * x`),
// one bf16 rounding at the end.
__global__ void rmsnorm_kernel(const uint16_t* x, int64_t x_stride, const uint16_t* w,
                               uint16_t* y, int64_t y_stride, int dim, float eps) {
  dgpp_pdl_wait();
  __shared__ float red[8];
  const int64_t r = blockIdx.x;
  const uint16_t* xr = x + r * x_stride;
  uint16_t* yr = y + r * y_stride;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = bf16_bits_to_float(xr[i]);
    ss = __fmaf_rn(v, v, ss);
  }
  const float total = block_sum_256(ss, red);
  const float rs = rsqrtf(__fadd_rn(__fdiv_rn(total, float(dim)), eps));
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = __fmul_rn(bf16_bits_to_float(xr[i]), rs);
    yr[i] = float_to_bf16_bits(__fmul_rn(bf16_bits_to_float(w[i]), v));
  }
}

// The two norms behind the paired q/kv projection and the kv rope in one
// launch per row (2026-10-20): rmsnorm_kernel's chain twice (block_sum_256
// over the same 256 threads, the same ops) then rope_apply_kernel's per-pair
// math over the kv row's rope pairs — bitwise the three kernels.
__global__ void __launch_bounds__(kCsa2Threads) norms_rope_kernel(
    uint16_t* q, int64_t q_stride, const uint16_t* q_w, int q_dim, uint16_t* kv, int64_t kv_stride,
    const uint16_t* kv_w, int kv_dim, float eps, int64_t rope_off, int half, const int64_t* pos,
    const float* inv_freq) {
  dgpp_pdl_wait();
  __shared__ float red[8];
  const int64_t r = blockIdx.x;
  {
    uint16_t* xr = q + r * q_stride;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < q_dim; i += blockDim.x) {
      const float v = bf16_bits_to_float(xr[i]);
      ss = __fmaf_rn(v, v, ss);
    }
    const float total = block_sum_256(ss, red);
    const float rs = rsqrtf(__fadd_rn(__fdiv_rn(total, float(q_dim)), eps));
    for (int i = threadIdx.x; i < q_dim; i += blockDim.x) {
      const float v = __fmul_rn(bf16_bits_to_float(xr[i]), rs);
      xr[i] = float_to_bf16_bits(__fmul_rn(bf16_bits_to_float(q_w[i]), v));
    }
  }
  {
    uint16_t* xr = kv + r * kv_stride;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
      const float v = bf16_bits_to_float(xr[i]);
      ss = __fmaf_rn(v, v, ss);
    }
    const float total = block_sum_256(ss, red);
    const float rs = rsqrtf(__fadd_rn(__fdiv_rn(total, float(kv_dim)), eps));
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
      const float v = __fmul_rn(bf16_bits_to_float(xr[i]), rs);
      xr[i] = float_to_bf16_bits(__fmul_rn(bf16_bits_to_float(kv_w[i]), v));
    }
  }
  __syncthreads();  // the normed kv row is written before its rope
  const int64_t p = pos[r];
  if (p < 0) return;
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    uint16_t* v = kv + r * kv_stride + rope_off + 2 * i;
    const float ang = __fmul_rn(float(p), inv_freq[i]);
    const float c = cosf(ang);
    const float sn = sinf(ang);
    const float x0 = bf16_bits_to_float(v[0]), x1 = bf16_bits_to_float(v[1]);
    v[0] = float_to_bf16_bits(x0 * c - x1 * sn);
    v[1] = float_to_bf16_bits(x1 * c + x0 * sn);
  }
}

// ---- rotation ------------------------------------------------------------------------
__global__ void rope_apply_kernel(uint16_t* x, int64_t row_stride, int64_t head_stride,
                                  int heads, int half, const int64_t* pos,
                                  const float* inv_freq, bool inverse, int64_t total) {
  dgpp_pdl_wait();
  const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int i = int(idx % half);
  const int h = int((idx / half) % heads);
  const int64_t r = idx / half / heads;
  const int64_t p = pos[r];
  if (p < 0) return;
  uint16_t* v = x + r * row_stride + int64_t(h) * head_stride + 2 * i;
  const float ang = __fmul_rn(float(p), inv_freq[i]);
  const float c = cosf(ang);
  float s = sinf(ang);
  if (inverse) s = -s;
  const float x0 = bf16_bits_to_float(v[0]), x1 = bf16_bits_to_float(v[1]);
  v[0] = float_to_bf16_bits(x0 * c - x1 * s);
  v[1] = float_to_bf16_bits(x1 * c + x0 * s);
}

// ---- positions ------------------------------------------------------------------------
__global__ void entry_positions_kernel(const int64_t* pos, int64_t* out, int rows, int ratio) {
  dgpp_pdl_wait();
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const int64_t p = pos[r];
  out[r] = p < 0 ? -1 : (p + 1) / ratio - 1;
}
__global__ void scaled_positions_kernel(const int64_t* src, int64_t* out, int rows, int64_t mul,
                                        int64_t add) {
  dgpp_pdl_wait();
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const int64_t p = src[r];
  out[r] = p < 0 ? -1 : p * mul + add;
}
__global__ void ring_slot_positions_kernel(const int64_t* pos, int64_t* out, int rows, int ring_slots) {
  dgpp_pdl_wait();
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const int64_t p = pos[r];
  out[r] = p < 0 ? -1 : p % ring_slots;
}

// ---- the window ring ----------------------------------------------------------------------
__global__ void ring_table_kernel(int32_t* table, int n) {
  dgpp_pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) table[i] = i;
}
__global__ void window_slots_decode_kernel(const int64_t* pos, int rows, int window,
                                           int ring_slots, int32_t* list, int32_t* counts) {
  dgpp_pdl_wait();
  const int r = blockIdx.x;
  const int64_t p = pos[r];
  const int n = p < 0 ? 0 : int(min(int64_t(window), p + 1));
  for (int k = threadIdx.x; k < window; k += blockDim.x) {
    const int64_t q = p - (n - 1) + k;  // ascending positions
    list[int64_t(r) * window + k] = k < n ? int32_t(q % ring_slots) : -1;
  }
  if (threadIdx.x == 0) counts[r] = n;
}
__global__ void dspark_window_slots_kernel(const int64_t* pos, int rows, int block, int window, int ring_slots,
                                           int32_t* list, int32_t* counts) {
  dgpp_pdl_wait();
  const int r = blockIdx.x;
  const int stride = window + block;
  const int64_t p = pos[r];
  const int64_t P = p < 0 ? -1 : p - (r % block);  // the block's first position
  const int n_ring = P < 0 ? 0 : int(min(int64_t(window), P));
  const int n = P < 0 ? 0 : n_ring + block;
  for (int k = threadIdx.x; k < stride; k += blockDim.x) {
    int32_t slot = -1;
    if (k < n_ring) slot = int32_t((P - n_ring + k) % ring_slots);          // ascending real positions
    else if (k < n) slot = int32_t((P + (k - n_ring)) % ring_slots);        // the block's rows
    list[int64_t(r) * stride + k] = slot;
  }
  if (threadIdx.x == 0) counts[r] = n;
}
__global__ void window_slots_prefill_kernel(int64_t pos0, int T, int window, int64_t floor, int32_t* list,
                                            int32_t* counts) {
  dgpp_pdl_wait();
  const int i = blockIdx.x;
  const int64_t p = pos0 + i;
  const int64_t first = p - window + 1 < floor ? floor : p - window + 1;
  const int n = int(p - first + 1);
  for (int k = threadIdx.x; k < window; k += blockDim.x) {
    const int64_t q = first + k;
    list[int64_t(i) * window + k] = k < n ? int32_t(q - pos0 + window - 1) : -1;
  }
  if (threadIdx.x == 0) counts[i] = n;
}
__global__ void window_scratch_prologue_kernel(const uint8_t* ring, int ring_slots, int64_t pos0,
                                               int window, int64_t floor, size_t row_bytes, uint8_t* scratch) {
  dgpp_pdl_wait();
  const int j = blockIdx.x;  // scratch row j: position pos0 - (window - 1) + j
  const int64_t q = pos0 - (window - 1) + j;
  uint4* dst = reinterpret_cast<uint4*>(scratch + size_t(j) * row_bytes);
  const int n16 = int(row_bytes / 16);
  if (q < floor) {  // before the context, or before the replay floor: never listed
    for (int k = threadIdx.x; k < n16; k += blockDim.x) dst[k] = make_uint4(0, 0, 0, 0);
    return;
  }
  const uint4* src = reinterpret_cast<const uint4*>(ring + size_t(q % ring_slots) * row_bytes);
  for (int k = threadIdx.x; k < n16; k += blockDim.x) dst[k] = src[k];
}
__global__ void window_ring_writeback_kernel(const uint8_t* scratch, int window, int64_t pos0,
                                             int i0, int T, int ring_slots, size_t row_bytes,
                                             uint8_t* ring) {
  dgpp_pdl_wait();
  const int i = i0 + blockIdx.x;
  if (i >= T) return;
  const uint4* src = reinterpret_cast<const uint4*>(scratch + size_t(window - 1 + i) * row_bytes);
  uint4* dst = reinterpret_cast<uint4*>(ring + size_t((pos0 + i) % ring_slots) * row_bytes);
  const int n16 = int(row_bytes / 16);
  for (int k = threadIdx.x; k < n16; k += blockDim.x) dst[k] = src[k];
}

// ---- the fp4 e8m0/32 index forms ------------------------------------------------------
// One block of 128 threads per 128-wide row: warp b is the 32-element
// block b. The reference's fp4_act_quant(block 32, e8m0): amax floored at
// 6 * 2^-126, s = 2^ceil(log2(amax * (1/6))), v = e2m1(x / s) * s. Stored
// as e4m3 codes on the row scale S = 2^(k_max - 6); the code's decode
// times S equals v for every block within 14 binades of the largest.
__device__ __forceinline__ void index_row_quant(const uint16_t* x, uint8_t* out_codes,
                                                float* out_scale, unsigned* violations,
                                                int* s_kmax /*smem [4]*/) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const float v = bf16_bits_to_float(x[threadIdx.x]);
  float amax = fabsf(v);
  for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  const float floor_v = kLatentFp4Max * 1.1754943508222875e-38f;  // 6 * 2^-126
  amax = amax > floor_v ? amax : floor_v;
  const uint8_t kb = e8m0_ceil_log2_byte(amax * (1.0f / kLatentFp4Max));
  const float s = e8m0_byte_to_float(kb);
  const float q = fp4_e2m1_bits_to_float(float_to_fp4_e2m1_bits(v / s)) * s;  // the dequantized value
  if (lane == 0) s_kmax[warp] = int(kb);
  __syncthreads();
  const int kmax = max(max(s_kmax[0], s_kmax[1]), max(s_kmax[2], s_kmax[3]));
  const float S = ldexpf(1.0f, kmax - 127 - 6);
  const uint8_t code = float_to_fp8_e4m3_bits(q / S);
  const bool bad = fp8_e4m3_bits_to_float(code) * S != q;
  out_codes[threadIdx.x] = code;
  if (threadIdx.x == 0) *out_scale = S;
  const unsigned any = __ballot_sync(0xffffffffu, bad);
  if (violations != nullptr && lane == 0 && any != 0u) atomicAdd(violations, 1u);
}
__global__ void index_q_quant_kernel(const uint16_t* q, uint8_t* q_fp8, float* q_scale,
                                     unsigned* violations) {
  dgpp_pdl_wait();
  __shared__ int s_kmax[4];
  const int64_t row = blockIdx.x;  // (token, head)
  index_row_quant(q + row * kCsa2IndexDim, q_fp8 + row * kCsa2IndexDim, q_scale + row, violations,
                  s_kmax);
}
__global__ void index_k_append_kernel(const uint16_t* k, const int32_t* req_ids,
                                      const int64_t* entries, const int32_t* block_tables,
                                      int blocks_per_request, int entries_per_block,
                                      uint8_t* index_k, float* index_scale, unsigned* violations) {
  dgpp_pdl_wait();
  __shared__ int s_kmax[4];
  const int64_t i = blockIdx.x;
  const int64_t e = entries[i];
  if (e < 0) return;
  const int32_t blk = block_tables[int64_t(req_ids[i]) * blocks_per_request + e / entries_per_block];
  const int64_t slot = int64_t(blk) * entries_per_block + (e % entries_per_block);
  index_row_quant(k + i * kCsa2IndexDim, index_k + slot * kCsa2IndexDim, index_scale + slot,
                  violations, s_kmax);
}
__global__ void fold_weights_kernel(const uint16_t* w, const float* q_scale, float* out, int64_t n) {
  dgpp_pdl_wait();
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = __fmul_rn(__fmul_rn(bf16_bits_to_float(w[i]), 0.015625f), q_scale[i]);
}

// ---- the compressor -------------------------------------------------------------------------
// The pair pooling of one entry: per channel c (thread t owns c = t, t +
// 256), softmax over (s0, s1), the fp32 weighted kv sum, a bf16 rounding,
// then the RMSNorm over the 512 (norm_w in fp32, one rounding).
__device__ __forceinline__ void pool_pair_and_norm(const float* kv0, const float* s0,
                                                   const float* kv1, const float* s1,
                                                   const uint16_t* norm_w, float eps,
                                                   uint16_t* out, float* red) {
  constexpr int kPer = kCsa2Latent / kCsa2Threads;  // 2
  float pooled[kPer];
  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int c = threadIdx.x + j * kCsa2Threads;
    const float a = s0[c], b = s1[c];
    const float m = fmaxf(a, b);
    const float e0 = expf(a - m), e1 = expf(b - m);
    const float den = e0 + e1;
    const float w0 = e0 / den, w1 = e1 / den;
    const float sum = __fadd_rn(__fmul_rn(kv0[c], w0), __fmul_rn(kv1[c], w1));
    const float v = bf16_bits_to_float(float_to_bf16_bits(sum));  // .to(bf16) before the norm
    pooled[j] = v;
    ss = __fmaf_rn(v, v, ss);
  }
  const float total = block_sum_256(ss, red);
  const float rs = rsqrtf(__fadd_rn(__fdiv_rn(total, float(kCsa2Latent)), eps));
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int c = threadIdx.x + j * kCsa2Threads;
    const float v = __fmul_rn(pooled[j], rs);
    out[c] = float_to_bf16_bits(__fmul_rn(bf16_bits_to_float(norm_w[c]), v));
  }
}
__global__ void compress_pairs_prefill_kernel(const float* kv, const float* score, int T,
                                              const uint16_t* norm_w, float eps,
                                              uint16_t* latent_out, float* tail) {
  dgpp_pdl_wait();
  __shared__ float red[8];
  const int j = blockIdx.x;
  if (2 * j + 1 < T) {
    pool_pair_and_norm(kv + int64_t(2 * j) * kCsa2Latent, score + int64_t(2 * j) * kCsa2Latent,
                       kv + int64_t(2 * j + 1) * kCsa2Latent, score + int64_t(2 * j + 1) * kCsa2Latent,
                       norm_w, eps, latent_out + int64_t(j) * kCsa2Latent, red);
    return;
  }
  // The odd trailing row (block T / 2 when T is odd): the request's tail.
  if (tail == nullptr) return;
  const int t = T - 1;
  for (int c = threadIdx.x; c < kCsa2Latent; c += blockDim.x) {
    tail[c] = kv[int64_t(t) * kCsa2Latent + c];
    tail[kCsa2Latent + c] = score[int64_t(t) * kCsa2Latent + c];
  }
}
__global__ void compress_decode_update_kernel(const float* kv, const float* score,
                                              const int32_t* req_ids, const int64_t* pos,
                                              const int32_t* req_spans, const uint16_t* norm_w,
                                              float eps, float* tails, uint16_t* latent_out,
                                              int64_t* entries_out, float* tail_snapshots) {
  dgpp_pdl_wait();
  __shared__ float red[8];
  const int q = blockIdx.x;
  const int start = req_spans[2 * q], len = req_spans[2 * q + 1];
  if (len <= 0) return;
  float* tail = tails + int64_t(req_ids[start]) * 2 * kCsa2Latent;
  for (int t = start; t < start + len; ++t) {
    const int64_t p = pos[t];
    const float* kvt = kv + int64_t(t) * kCsa2Latent;
    const float* st = score + int64_t(t) * kCsa2Latent;
    uint16_t* lat = latent_out + int64_t(t) * kCsa2Latent;
    if (p < 0) {
      for (int c = threadIdx.x; c < kCsa2Latent; c += blockDim.x) lat[c] = 0;
      if (threadIdx.x == 0) entries_out[t] = -1;
      continue;
    }
    if ((p & 1) == 0) {
      for (int c = threadIdx.x; c < kCsa2Latent; c += blockDim.x) {
        tail[c] = kvt[c];
        tail[kCsa2Latent + c] = st[c];
        lat[c] = 0;
      }
      if (threadIdx.x == 0) entries_out[t] = -1;
    } else {
      __syncthreads();  // the tail's stash (an earlier row of this block) is complete
      pool_pair_and_norm(tail, tail + kCsa2Latent, kvt, st, norm_w, eps, lat, red);
      if (threadIdx.x == 0) entries_out[t] = p / 2;
    }
    if (tail_snapshots != nullptr && t != start + len - 1) {
      __syncthreads();
      float* snap = tail_snapshots + int64_t(t) * 2 * kCsa2Latent;
      for (int c = threadIdx.x; c < 2 * kCsa2Latent; c += blockDim.x) snap[c] = tail[c];
    }
    __syncthreads();
  }
}

// ---- the selections -----------------------------------------------------------------------
// One entry's logit, warp-cooperative (lane = head): the fp8 dot in fp32
// with contraction-proof arithmetic, relu, the folded weight, the entry's
// scale, the warp sum — dsa_select_decode's DecodeKeyFn arithmetic. q8:
// smem [16 chunks][32 heads] uint2 (8 e4m3 each); krow: smem [32] u32
// (the entry's 128 codes, lane l loads word l).
__device__ __forceinline__ float entry_logit(const uint2* q8, const uint32_t* krow, const float* w,
                                             float ks) {
  const int lane = threadIdx.x & 31;
  const uint2* k2 = reinterpret_cast<const uint2*>(krow);
  float partial = 0.0f;
#pragma unroll 8
  for (int p = 0; p < 16; ++p) {
    const uint2 qv = q8[p * 32 + lane];
    const uint2 kv = k2[p];
    const uint16_t* qq = reinterpret_cast<const uint16_t*>(&qv);
    const uint16_t* kk = reinterpret_cast<const uint16_t*>(&kv);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 a = e4m3x2_to_float2(qq[j]);
      const float2 b = e4m3x2_to_float2(kk[j]);
      partial = __fadd_rn(partial, __fmul_rn(a.x, b.x));
      partial = __fadd_rn(partial, __fmul_rn(a.y, b.y));
    }
  }
  partial = fmaxf(partial, 0.0f);
  const float contrib = __fmul_rn(__fmul_rn(w[lane], ks), partial);
  return warp_sum(contrib);
}
// Stages a row's q (32 heads x 128 e4m3) chunk-major into smem and its
// folded weights.
__device__ __forceinline__ void stage_query(const uint8_t* q_fp8_row, const float* w_row,
                                            uint2* q8, float* w) {
  // q_fp8_row: [32 heads][128] bytes = [32][16 chunks] uint2.
  const uint2* src = reinterpret_cast<const uint2*>(q_fp8_row);
  for (int i = threadIdx.x; i < 32 * 16; i += blockDim.x) {
    const int h = i / 16, c = i % 16;
    q8[c * 32 + h] = src[h * 16 + c];
  }
  if (threadIdx.x < 32) w[threadIdx.x] = w_row[threadIdx.x];
}
__device__ __forceinline__ int64_t entry_slot(const int32_t* table, int entries_per_block, int64_t e) {
  const int32_t blk = table[e / entries_per_block];
  return int64_t(blk) * entries_per_block + (e % entries_per_block);
}
__device__ __forceinline__ uint64_t make_key(float logit, int64_t idx) {
  return (uint64_t(~sortable_f32_dev(logit)) << kIdxBits) | uint64_t(idx);
}

constexpr int kCandTile = 2 * kSelectTile;  // prefill top-2048 block selection

// The candidate pool's i-th entry: the listed complete blocks' entries in
// list order, then the newest partial block's [nb * bs, visible).
__device__ __forceinline__ int64_t pool_entry(const int32_t* cand, int n_blocks, int block_size, int64_t visible,
                                              int64_t i) {
  dgpp_pdl_wait();
  const int64_t listed = int64_t(n_blocks) * block_size;
  if (i < listed) {
    const int32_t b = cand[i / block_size];
    return b < 0 ? -1 : int64_t(b) * block_size + (i % block_size);
  }
  return (visible / block_size) * block_size + (i - listed);
}
__device__ __forceinline__ int64_t pool_size(int n_blocks, int block_size, int64_t visible) {
  return int64_t(n_blocks) * block_size + (visible - (visible / block_size) * block_size);
}
constexpr int kCsaRadixBits = 10;
constexpr int kCsaHistBins = 1 << kCsaRadixBits;
constexpr int kCsaBoundaryKeys = 256;

__device__ inline void csa_hist_add(int32_t* hist, int bin) {
  const unsigned peers = __match_any_sync(__activemask(), bin);
  if ((threadIdx.x & 31) == __ffs(peers) - 1) atomicAdd(hist + bin, __popc(peers));
}

__device__ inline void csa_boundary_bin(const int32_t* hist, int remaining, int* boundary,
                                        int* below) {
  __shared__ int warp_totals[256 / 32];
  constexpr int per = kCsaHistBins / 256;
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

__device__ inline void csa_select_radix(const uint64_t* keys, int64_t visible, int select_k,
                                        uint32_t* best_hi, uint32_t* best_lo, uint32_t* tile_hi,
                                        uint32_t* tile_lo) {
  __shared__ int boundary, below, definite_count, candidate_count;
  int32_t* hist = reinterpret_cast<int32_t*>(tile_hi);
  uint64_t prefix = 0;
  int prefix_shift = 32 + kIdxBits;
  int remaining = select_k, lower = 0, count = 0, shift = 0;
  for (;;) {
    shift = max(0, prefix_shift - kCsaRadixBits);
    const int bits = prefix_shift - shift;
    const uint64_t mask = (1ull << bits) - 1;
    for (int i = threadIdx.x; i < kCsaHistBins; i += blockDim.x) hist[i] = 0;
    __syncthreads();
    for (int64_t p = threadIdx.x; p < visible; p += blockDim.x) {
      const uint64_t key = keys[p];
      if ((key >> prefix_shift) == prefix) csa_hist_add(hist, int((key >> shift) & mask));
    }
    __syncthreads();
    csa_boundary_bin(hist, remaining, &boundary, &below);
    count = hist[boundary];
    lower += below;
    remaining -= below;
    prefix = (prefix << bits) | uint64_t(boundary);
    // The pool-id suffix makes keys unique, so the final digit always
    // leaves at most one key. Equal scores need no approximation/fallback.
    __syncthreads();
    if (count <= kCsaBoundaryKeys || shift == 0) break;
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
      if (i < kCsaBoundaryKeys) {
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

// Independent scoring stripes fill the existing decode key workspace.
// Four index rows are fetched before their dots, preserving entry_logit's
// per-head arithmetic while hiding the dependent table/key/scale loads.
template <bool Candidates>
__global__ void csa_decode_scores_kernel(
    const uint8_t* q_fp8, const float* w_folded, const int32_t* req_ids, const int64_t* pos_sel,
    const int32_t* block_tables, int blocks_per_request, const uint8_t* index_k,
    const float* index_scale, int entries_per_block, const int32_t* cand, int cand_stride,
    const int32_t* cand_counts, int block_size, int select_k, uint64_t* keys, int64_t keys_stride) {
  dgpp_pdl_wait();
  __shared__ uint2 q8[16 * 32];
  __shared__ float w[32];
  __shared__ uint32_t krows[8 * 4 * 32];
  const int r = blockIdx.y;
  const int64_t p = pos_sel[r];
  if (p < 0) return;
  const int64_t visible = p + 1;
  const int n_blocks = !Candidates && cand ? cand_counts[r] : 0;
  const int64_t n = Candidates ? visible / block_size
                               : (cand ? pool_size(n_blocks, block_size, visible) : visible);
  // All entries will be kept. The selector writes their IDs directly.
  if (n <= select_k) return;
  const int32_t* table = block_tables + int64_t(req_ids[r]) * blocks_per_request;
  const int32_t* list = cand ? cand + int64_t(r) * cand_stride : nullptr;
  stage_query(q_fp8 + int64_t(r) * 32 * kCsa2IndexDim, w_folded + int64_t(r) * 32, q8, w);
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint32_t* staged = krows + warp * 4 * 32;
  const int64_t stripe = (n + gridDim.x - 1) / gridDim.x;
  const int64_t lo = min(n, int64_t(blockIdx.x) * stripe), hi = min(n, lo + stripe);
  const int64_t sub = (hi - lo + 7) / 8;
  const int64_t wlo = min(hi, lo + warp * sub), whi = min(hi, wlo + sub);
  if constexpr (Candidates) {
    const int64_t pinned = visible % block_size == 0 ? n - 1 : -1;
    for (int64_t b = wlo; b < whi; ++b) {
      float best = -INFINITY;
      if (b == pinned) {
        best = INFINITY;
      } else {
        for (int j = 0; j < block_size; j += 4) {
          int64_t slots[4];
          uint32_t words[4];
          float scales[4];
#pragma unroll
          for (int u = 0; u < 4; ++u)
            slots[u] = j + u < block_size
                           ? entry_slot(table, entries_per_block, b * block_size + j + u)
                           : -1;
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            words[u] =
                slots[u] >= 0
                    ? reinterpret_cast<const uint32_t*>(index_k + slots[u] * kCsa2IndexDim)[lane]
                    : 0;
            scales[u] = slots[u] >= 0 ? index_scale[slots[u]] : 0.f;
          }
#pragma unroll
          for (int u = 0; u < 4; ++u) staged[u * 32 + lane] = words[u];
          __syncwarp();
#pragma unroll
          for (int u = 0; u < 4; ++u)
            if (slots[u] >= 0) best = fmaxf(best, entry_logit(q8, staged + u * 32, w, scales[u]));
          __syncwarp();
        }
      }
      if (lane == 0) keys[int64_t(r) * keys_stride + b] = make_key(best, b);
    }
  } else {
    for (int64_t i = wlo; i < whi; i += 4) {
      int64_t ids[4], slots[4];
      uint32_t words[4];
      float scales[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        ids[u] = i + u < whi
                     ? (list ? pool_entry(list, n_blocks, block_size, visible, i + u) : i + u)
                     : -1;
        slots[u] = ids[u] >= 0 && ids[u] <= p ? entry_slot(table, entries_per_block, ids[u]) : -1;
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        words[u] = slots[u] >= 0
                       ? reinterpret_cast<const uint32_t*>(index_k + slots[u] * kCsa2IndexDim)[lane]
                       : 0;
        scales[u] = slots[u] >= 0 ? index_scale[slots[u]] : 0.f;
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) staged[u * 32 + lane] = words[u];
      __syncwarp();
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        if (i + u >= whi) break;
        const uint64_t key = slots[u] >= 0
                                 ? make_key(entry_logit(q8, staged + u * 32, w, scales[u]), ids[u])
                                 : kCsa2KeyMax;
        if (lane == 0) keys[int64_t(r) * keys_stride + i + u] = key;
      }
      __syncwarp();
    }
  }
}

template <bool Candidates>
__global__ void csa_decode_select_kernel(const uint64_t* keys, int64_t keys_stride,
                                         const int64_t* pos_sel, const int32_t* cand,
                                         int cand_stride, const int32_t* cand_counts,
                                         int block_size, int select_k, int32_t* out,
                                         int32_t* counts) {
  dgpp_pdl_wait();
  const int r = blockIdx.x;
  const int64_t p = pos_sel[r], visible = p + 1;
  const int n_blocks = !Candidates && cand && p >= 0 ? cand_counts[r] : 0;
  const int64_t n =
      p < 0 ? 0
            : (Candidates ? visible / block_size
                          : (cand ? pool_size(n_blocks, block_size, visible) : visible));
  int32_t* dst = out + int64_t(r) * select_k;
  if (n <= select_k) {
    const int32_t* list = cand ? cand + int64_t(r) * cand_stride : nullptr;
    for (int i = threadIdx.x; i < select_k; i += blockDim.x) {
      int32_t id = -1;
      if (i < n)
        id = !Candidates && list ? int32_t(pool_entry(list, n_blocks, block_size, visible, i)) : i;
      dst[i] = id;
    }
    if (threadIdx.x == 0) counts[r] = int(n);
    return;
  }
  extern __shared__ uint32_t smem[];
  uint32_t* best_hi = smem;
  uint32_t* best_lo = best_hi + select_k;
  uint32_t* boundary_hi = best_lo + select_k;  // also the 1024-bin histogram
  uint32_t* boundary_lo = boundary_hi + kCsaHistBins;
  int32_t* scratch = reinterpret_cast<int32_t*>(boundary_lo + kCsaBoundaryKeys);
  __shared__ int count;
  for (int i = threadIdx.x; i < select_k; i += blockDim.x) best_hi[i] = best_lo[i] = 0xffffffffu;
  __syncthreads();
  csa_select_radix(keys + int64_t(r) * keys_stride, n, select_k, best_hi, best_lo, boundary_hi,
                   boundary_lo);
  const int64_t last = Candidates ? n - 1 : p;
  const int kept =
      expand_from_best(best_hi, best_lo, select_k, last, 1, select_k, dst, scratch, &count);
  if (threadIdx.x == 0) counts[r] = kept;
}

size_t decode_select_smem(int select_k) {
  return size_t(select_k) * 12 + size_t(kCsaHistBins + kCsaBoundaryKeys) * 4;
}

// Prefill: the logits rows.
__global__ void logits_prefill_kernel(const float* dot, int64_t dot_stride, const float* w_folded,
                                      const float* k_scale, const int64_t* pos_sel, int64_t n, int heads,
                                      float* logits, int64_t logits_stride) {
  dgpp_pdl_wait();
  const int r = blockIdx.y;
  const int64_t p = pos_sel[r];
  const int64_t visible = p < 0 ? 0 : p + 1;
  const int64_t j = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (j >= n) return;
  float total = -INFINITY;
  if (j < visible) {
    total = 0.0f;
    const float ks = k_scale[j];
    for (int h = 0; h < heads; ++h) {
      const float dv = fmaxf(dot[(int64_t(r) * heads + h) * dot_stride + j], 0.0f);
      total = __fadd_rn(total, __fmul_rn(__fmul_rn(w_folded[int64_t(r) * heads + h], ks), dv));
    }
  }
  logits[int64_t(r) * logits_stride + j] = total;
}
// The fused prefill scorer (csa2_score_prefill): one query row's 32 heads
// against a tile of 128 entries per CTA. The e4m3 query heads and keys are
// staged in shared memory (a 144-byte row stride keeps ldmatrix off the
// bank conflicts), four k-steps of mma.m16n8k32 e4m3 -> fp32 give every
// head's dot with every entry of the tile, and the heads fold in the
// fragments: each lane holds four heads of two entries, the four products
// (w_folded * k_scale) * relu(dot) sum pairwise, then the eight lane
// groups sum by xor shuffles — the decode scorer's tree, not the retired
// GEMM chain's sequential sum (the products are the same terms; the order
// moves the sum by fp32 rounding, which the selection's near-tie
// certification covers as it does between decode and prefill). No dot
// tile leaves the registers: 23 KB of shared memory, four CTAs per SM.
// Entries at or past the row's visible count are -inf; keys past it are
// not read.
constexpr int kScoreTile = 128;
__global__ void __launch_bounds__(128, 4)
score_prefill_kernel(const uint8_t* q_fp8, const uint8_t* k, const float* w_folded, const float* k_scale,
                     const int64_t* pos_sel, int rows, int64_t n, float* logits, int64_t logits_stride) {
  dgpp_pdl_wait();
  constexpr int S = 144, N = kScoreTile;
  __shared__ __align__(16) uint8_t a[32 * S];
  __shared__ __align__(16) uint8_t b[N * S];
  const int r = blockIdx.x % rows;
  const int64_t p0 = int64_t(blockIdx.x / rows) * N;
  if (p0 >= n) return;
  const int64_t p = pos_sel[r];
  const int64_t visible = p < 0 ? 0 : min(p + 1, n);
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  if (p0 >= visible) {  // the tile holds no visible entry: -inf, no key reads
    const int64_t e = p0 + tid;
    if (e < n) logits[int64_t(r) * logits_stride + e] = -INFINITY;
    return;
  }
  for (int i = tid; i < 32 * 8; i += 128) {
    const int row = i / 8, col = (i % 8) * 16;
    *reinterpret_cast<uint4*>(a + row * S + col) =
        *reinterpret_cast<const uint4*>(q_fp8 + (int64_t(r) * 32 + row) * kCsa2IndexDim + col);
  }
  for (int i = tid; i < N * 8; i += 128) {
    const int row = i / 8, col = (i % 8) * 16;
    const int64_t e = p0 + row;
    *reinterpret_cast<uint4*>(b + row * S + col) =
        e < visible ? *reinterpret_cast<const uint4*>(k + e * kCsa2IndexDim + col) : make_uint4(0, 0, 0, 0);
  }
  __syncthreads();
  float acc[2][4][4] = {};
#pragma unroll
  for (int kk = 0; kk < kCsa2IndexDim; kk += 32) {
    uint32_t af[2][4], bf[4][2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(
          a + (i * 16 + lane % 16) * S + kk + (lane / 16) * 16));
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(af[i][0]), "=r"(af[i][1]), "=r"(af[i][2]), "=r"(af[i][3])
                   : "r"(addr));
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int row = warp * 32 + j * 16 + (lane / 16) * 8 + lane % 8;
      const int col = kk + ((lane / 8) % 2) * 16;
      const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(b + row * S + col));
      uint32_t v[4];
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
                   : "r"(addr));
      bf[j * 2][0] = v[0];
      bf[j * 2][1] = v[1];
      bf[j * 2 + 1][0] = v[2];
      bf[j * 2 + 1][1] = v[3];
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
            : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(bf[j][0]), "r"(bf[j][1]));
  }
  // The fragments: head i * 16 + h * 8 + lane / 4 at entry warp * 32 + j * 8
  // + (lane % 4) * 2 + v holds acc[i][j][h * 2 + v]. Fold the lane's four
  // heads, then the eight lane groups (xor 16, 8, 4); lanes 0..3 hold the
  // row's logit for their two entries.
  float weight[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int h = 0; h < 2; ++h) weight[i][h] = w_folded[int64_t(r) * 32 + i * 16 + h * 8 + lane / 4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int v = 0; v < 2; ++v) {
      const int64_t e = p0 + warp * 32 + j * 8 + (lane % 4) * 2 + v;
      const float ks = e < visible ? k_scale[e] : 0.0f;
      float c[2][2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
          c[i][h] = __fmul_rn(__fmul_rn(weight[i][h], ks), fmaxf(acc[i][j][h * 2 + v], 0.0f));
      float sum = __fadd_rn(__fadd_rn(c[0][0], c[1][0]), __fadd_rn(c[0][1], c[1][1]));
#pragma unroll
      for (int off = 16; off >= 4; off >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, off));
      if (lane < 4 && e < n) logits[int64_t(r) * logits_stride + e] = e < visible ? sum : -INFINITY;
    }
}
struct RowKeyFn {
  static constexpr bool kWarpCooperative = false;
  const float* logits;
  const int32_t* cand;  // null: the entry is its index
  int n_blocks;
  int block_size;
  int64_t pos_sel;
  __device__ uint64_t operator()(int64_t i) const {
    const int64_t e = cand ? pool_entry(cand, n_blocks, block_size, pos_sel + 1, i) : i;
    if (e < 0 || e > pos_sel) return kCsa2KeyMax;
    return make_key(logits[e], e);
  }
};
// TILE: the streaming tile (kSelectTile for select_k <= 1024, kCandTile for
// the 2048 of a widened index budget — the same network, twice the smem).
template <int TILE>
__global__ void select_rows_prefill_kernel(const float* logits, int64_t logits_stride,
                                           const int64_t* pos_sel, int select_k, const int32_t* cand,
                                           int cand_stride, const int32_t* cand_counts, int block_size,
                                           int32_t* topk_out, int32_t* counts) {
  dgpp_pdl_wait();
  extern __shared__ uint8_t smem_raw[];
  uint32_t* best_hi = reinterpret_cast<uint32_t*>(smem_raw);
  uint32_t* best_lo = best_hi + select_k;
  uint32_t* tile_hi = best_lo + select_k;
  uint32_t* tile_lo = tile_hi + TILE;
  int32_t* scratch = reinterpret_cast<int32_t*>(tile_lo + TILE);
  __shared__ int smem_count;
  const int r = blockIdx.x;
  const int64_t p = pos_sel[r];
  int32_t* out = topk_out + int64_t(r) * select_k;
  if (p < 0) {
    for (int i = threadIdx.x; i < select_k; i += blockDim.x) out[i] = -1;
    if (threadIdx.x == 0) counts[r] = 0;
    return;
  }
  for (int i = threadIdx.x; i < select_k; i += blockDim.x) {
    best_hi[i] = 0xFFFFFFFFu;
    best_lo[i] = 0xFFFFFFFFu;
  }
  __syncthreads();
  const int n_blocks = cand ? cand_counts[r] : 0;
  const int64_t n = cand ? pool_size(n_blocks, block_size, p + 1) : p + 1;
  RowKeyFn fn{logits + int64_t(r) * logits_stride, cand ? cand + int64_t(r) * cand_stride : nullptr, n_blocks,
              block_size, p};
  if (n > 0) select_topk_stream<TILE>(fn, 0, n, best_hi, best_lo, tile_hi, tile_lo, select_k);
  const int cnt = expand_from_best(best_hi, best_lo, select_k, p, 1, select_k, out, scratch, &smem_count);
  if (threadIdx.x == 0) counts[r] = cnt;
}
struct RowBlockKeyFn {
  static constexpr bool kWarpCooperative = false;
  const float* logits;
  int block_size;
  int64_t pinned;
  __device__ uint64_t operator()(int64_t b) const {
    if (b == pinned) return make_key(INFINITY, b);
    float best = -INFINITY;
    for (int k = 0; k < block_size; ++k) best = fmaxf(best, logits[b * block_size + k]);
    return make_key(best, b);
  }
};
__global__ void select_candidates_prefill_kernel(const float* logits, int64_t logits_stride,
                                                 const int64_t* pos_sel, int block_size, int topk_blocks,
                                                 int cand_stride, int32_t* cand_out, int32_t* cand_counts) {
  dgpp_pdl_wait();
  extern __shared__ uint8_t smem_raw[];
  uint32_t* best_hi = reinterpret_cast<uint32_t*>(smem_raw);
  uint32_t* best_lo = best_hi + topk_blocks;
  uint32_t* tile_hi = best_lo + topk_blocks;
  uint32_t* tile_lo = tile_hi + kCandTile;
  int32_t* scratch = reinterpret_cast<int32_t*>(tile_lo + kCandTile);
  __shared__ int smem_count;
  const int r = blockIdx.x;
  const int64_t p = pos_sel[r];
  int32_t* out = cand_out + int64_t(r) * cand_stride;
  if (p < 0) {
    for (int i = threadIdx.x; i < cand_stride; i += blockDim.x) out[i] = -1;
    if (threadIdx.x == 0) cand_counts[r] = 0;
    return;
  }
  for (int i = threadIdx.x; i < topk_blocks; i += blockDim.x) {
    best_hi[i] = 0xFFFFFFFFu;
    best_lo[i] = 0xFFFFFFFFu;
  }
  __syncthreads();
  const int64_t visible = p + 1;
  const int64_t nb = visible / block_size;
  RowBlockKeyFn fn{logits + int64_t(r) * logits_stride, block_size,
                   (visible % block_size == 0) ? nb - 1 : -1};
  if (nb > 0) select_topk_stream<kCandTile>(fn, 0, nb, best_hi, best_lo, tile_hi, tile_lo, topk_blocks);
  const int n = expand_from_best(best_hi, best_lo, topk_blocks, nb - 1, 1, cand_stride, out, scratch,
                                 &smem_count);
  if (threadIdx.x == 0) cand_counts[r] = n;
}

// ---- the attention finish ---------------------------------------------------------------------
// One block per (row, head): the splits' weights exp(m - mhat) and the
// denominator once per block (thread 0, the splits in order: main, then
// window, then the sink), then four output dims per thread as one 16-byte
// load per split, eight splits in flight. (Until 2026-10-01 every thread
// recomputed mhat, the denominator and every split's exp per dim behind
// one load at a time: 17-41 us a launch for 5 MB of L2-resident partials.)
// A split with m = -inf (no key in it) weighs exactly zero and is skipped:
// the sums are the full loops' bit for bit.
constexpr int kFinishMaxSplits = 128;
constexpr int kFinishBatch = 8;
__global__ void attn_finish_kernel(const float* m_main, const float* l_main, const float* c_main,
                                   int n_main, const float* m_win, const float* l_win, const float* c_win,
                                   int n_win, const float* sink, int local_heads, const int64_t* pos,
                                   const float* inv_freq, uint16_t* out) {
  dgpp_pdl_wait();
  __shared__ float o[kCsa2Latent];
  __shared__ float s_w[kFinishMaxSplits];          // exp(m - mhat) of the live splits, in order
  __shared__ const float* s_c[kFinishMaxSplits];   // their c rows [kCsa2Latent]
  __shared__ float s_m[kFinishMaxSplits];
  __shared__ float s_l[kFinishMaxSplits];          // every split's l, staged beside its m (2026-10-20)
  __shared__ float s_den;
  __shared__ int s_live;
  const int r = blockIdx.x, h = blockIdx.y;
  uint16_t* orow = out + (int64_t(r) * local_heads + h) * kCsa2Latent;
  const int64_t p = pos[r];
  if (p < 0) {
    for (int d = threadIdx.x; d < kCsa2Latent; d += blockDim.x) orow[d] = 0;
    return;
  }
  const int64_t total = int64_t(local_heads) * kCsa2Latent;
  const int n_all = n_main + n_win;
  // Every split's m and l, one thread per split — the l beside the m
  // (2026-10-20: thread 0's loop below loaded each live split's l as it
  // went, a dependent L2 round trip per split on the block's critical
  // path; 13 us a launch at eight window splits, the same arithmetic in
  // the same order from shared memory).
  for (int s = threadIdx.x; s < n_all; s += blockDim.x) {
    const bool main = s < n_main;
    const int64_t i = main ? (int64_t(r) * n_main + s) * local_heads + h
                           : (int64_t(r) * n_win + (s - n_main)) * local_heads + h;
    s_m[s] = main ? m_main[i] : m_win[i];
    s_l[s] = main ? l_main[i] : l_win[i];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float mhat = -INFINITY;
    for (int s = 0; s < n_all; ++s) mhat = fmaxf(mhat, s_m[s]);
    int live = 0;
    float den = 0.0f;
    if (mhat != -INFINITY) {
      for (int s = 0; s < n_all; ++s) {
        if (s_m[s] == -INFINITY) continue;
        const bool main = s < n_main;
        const float w = expf(s_m[s] - mhat);
        den += w * s_l[s];
        s_w[live] = w;
        s_c[live] = main ? c_main + (int64_t(r) * n_main + s) * total + int64_t(h) * kCsa2Latent
                         : c_win + (int64_t(r) * n_win + (s - n_main)) * total + int64_t(h) * kCsa2Latent;
        ++live;
      }
      den += expf(sink[h] - mhat);
    }
    s_den = den;
    s_live = mhat == -INFINITY ? -1 : live;
  }
  __syncthreads();
  const int live = s_live;
  const bool empty = live < 0;
  const float den = s_den;
  for (int d0 = threadIdx.x * 4; d0 < kCsa2Latent; d0 += blockDim.x * 4) {
    float num[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int s0 = 0; s0 < live; s0 += kFinishBatch) {
      float4 v[kFinishBatch];
#pragma unroll
      for (int b = 0; b < kFinishBatch; ++b)
        v[b] = s0 + b < live ? *reinterpret_cast<const float4*>(s_c[s0 + b] + d0) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int b = 0; b < kFinishBatch; ++b) {
        if (s0 + b >= live) break;
        const float w = s_w[s0 + b];
        num[0] += w * v[b].x;
        num[1] += w * v[b].y;
        num[2] += w * v[b].z;
        num[3] += w * v[b].w;
      }
    }
    // The attention output rounds to bf16; the inverse rotation then reads
    // that bf16 (the reference's two steps).
#pragma unroll
    for (int e = 0; e < 4; ++e) o[d0 + e] = bf16_bits_to_float(float_to_bf16_bits(empty ? 0.0f : num[e] / den));
  }
  __syncthreads();
  for (int d = threadIdx.x; d < kCsa2Latent - kCsa2Rope; d += blockDim.x) orow[d] = float_to_bf16_bits(o[d]);
  if (threadIdx.x < kCsa2Rope / 2) {
    const int i = threadIdx.x;
    const int d0 = kCsa2Latent - kCsa2Rope + 2 * i;
    const float ang = __fmul_rn(float(p), inv_freq[i]);
    const float c = cosf(ang), s = -sinf(ang);  // the conjugate
    const float x0 = o[d0], x1 = o[d0 + 1];
    orow[d0] = float_to_bf16_bits(x0 * c - x1 * s);
    orow[d0 + 1] = float_to_bf16_bits(x1 * c + x0 * s);
  }
}

int g_csa2_smem_cap = -1;
// The rows select's tile: kSelectTile serves select_k <= 1024 (the release's
// 512), kCandTile the 2048 a deployment may widen the index budget to.
int rows_tile(int select_k) { return select_k > kSelectTile / 2 ? kCandTile : kSelectTile; }
size_t rows_smem(int select_k) {
  return size_t(select_k) * 8 + size_t(rows_tile(select_k)) * 8 + size_t(select_k) * 4;
}
size_t cand_prefill_smem(int topk_blocks) {
  return size_t(topk_blocks) * 8 + size_t(kCandTile) * 8 + size_t(topk_blocks) * 4;
}

}  // namespace

void csa2_prepare_kernel_smem() {
  if (g_csa2_smem_cap >= 0) return;
  int cap = 0;
  DGPP_CUDA_OK(cudaDeviceGetAttribute(&cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  g_csa2_smem_cap = cap - 1024;
  DGPP_CUDA_OK(cudaFuncSetAttribute(select_rows_prefill_kernel<kSelectTile>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, g_csa2_smem_cap));
  DGPP_CUDA_OK(cudaFuncSetAttribute(select_rows_prefill_kernel<kCandTile>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, g_csa2_smem_cap));
  DGPP_CUDA_OK(cudaFuncSetAttribute(select_candidates_prefill_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, g_csa2_smem_cap));
}

void csa2_rmsnorm_bf16(const void* x, int64_t x_stride, const void* w, void* y, int64_t y_stride,
                       int rows, int dim, float eps, cudaStream_t stream) {
  if (rows <= 0) return;
  if (dim <= 0 || dim > 8192) throw std::invalid_argument("csa2_rmsnorm_bf16: dim must be in [1, 8192]");
  rmsnorm_kernel<<<unsigned(rows), kCsa2Threads, 0, stream>>>(
      static_cast<const uint16_t*>(x), x_stride, static_cast<const uint16_t*>(w),
      static_cast<uint16_t*>(y), y_stride, dim, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void csa2_norms_rope(void* q, int64_t q_stride, const void* q_w, int q_dim, void* kv, int64_t kv_stride,
                     const void* kv_w, int kv_dim, float eps, int64_t rope_off, int rope_dim, const int64_t* pos,
                     const float* inv_freq, int rows, cudaStream_t stream) {
  if (rows <= 0) return;
  if (q_dim <= 0 || q_dim > 8192 || kv_dim <= 0 || kv_dim > 8192)
    throw std::invalid_argument("csa2_norms_rope: dims must be in [1, 8192]");
  if (rope_dim <= 0 || rope_dim % 2 != 0) throw std::invalid_argument("csa2_norms_rope: rope_dim must be even");
  const int half = rope_dim / 2;
  if (rope_off < 0 || rope_off + int64_t(rope_dim) > kv_dim) throw std::invalid_argument("csa2_norms_rope: rope past the kv row");
  norms_rope_kernel<<<unsigned(rows), kCsa2Threads, 0, stream>>>(
      static_cast<uint16_t*>(q), q_stride, static_cast<const uint16_t*>(q_w), q_dim, static_cast<uint16_t*>(kv),
      kv_stride, static_cast<const uint16_t*>(kv_w), kv_dim, eps, rope_off, half, pos, inv_freq);
  DGPP_CUDA_OK(cudaGetLastError());
}

void csa2_rope_inv_freq_host(int rope_dim, double theta, int64_t original_seq_len, double factor,
                             double beta_fast, double beta_slow, float* out) {
  // The one YaRN builder (kernels/rope_scaling.cpp, moved verbatim): this
  // name stays for the DeepSeek-V4.1 callers and their tests, `original_seq_len`
  // being the builder's correction_max_position.
  yarn_rope_inv_freq_host(rope_dim, theta, original_seq_len, factor, beta_fast, beta_slow, out);
}

void csa2_rope_apply(void* x, int64_t row_stride, int64_t head_stride, int heads, int rope_dim,
                     const int64_t* pos, const float* inv_freq, bool inverse, int64_t rows,
                     cudaStream_t stream) {
  if (rows <= 0 || heads <= 0) return;
  if (rope_dim <= 0 || rope_dim % 2 != 0) throw std::invalid_argument("csa2_rope_apply: rope_dim must be even");
  const int half = rope_dim / 2;
  const int64_t total = rows * heads * half;
  rope_apply_kernel<<<unsigned((total + 255) / 256), 256, 0, stream>>>(
      static_cast<uint16_t*>(x), row_stride, head_stride, heads, half, pos, inv_freq, inverse, total);
  DGPP_CUDA_OK(cudaGetLastError());
}

void csa2_entry_positions(const int64_t* pos, int64_t* out, int rows, int ratio, cudaStream_t stream) {
  if (rows <= 0) return;
  if (ratio <= 0) throw std::invalid_argument("csa2_entry_positions: ratio must be positive");
  entry_positions_kernel<<<unsigned((rows + 255) / 256), 256, 0, stream>>>(pos, out, rows, ratio);
  DGPP_CUDA_OK(cudaGetLastError());
}
void csa2_scaled_positions(const int64_t* src, int64_t* out, int rows, int64_t mul, int64_t add,
                           cudaStream_t stream) {
  if (rows <= 0) return;
  scaled_positions_kernel<<<unsigned((rows + 255) / 256), 256, 0, stream>>>(src, out, rows, mul, add);
  DGPP_CUDA_OK(cudaGetLastError());
}
void csa2_ring_slot_positions(const int64_t* pos, int64_t* out, int rows, int ring_slots, cudaStream_t stream) {
  if (rows <= 0) return;
  if (ring_slots <= 0) throw std::invalid_argument("csa2_ring_slot_positions: ring_slots");
  ring_slot_positions_kernel<<<unsigned((rows + 255) / 256), 256, 0, stream>>>(pos, out, rows, ring_slots);
  DGPP_CUDA_OK(cudaGetLastError());
}

void csa2_ring_table(int32_t* table, int max_requests, cudaStream_t stream) {
  if (max_requests <= 0) return;
  ring_table_kernel<<<unsigned((max_requests + 255) / 256), 256, 0, stream>>>(table, max_requests);
  DGPP_CUDA_OK(cudaGetLastError());
}
void csa2_window_slots_decode(const int64_t* pos, int rows, int window, int ring_slots, int32_t* list,
                              int32_t* counts, cudaStream_t stream) {
  if (rows <= 0) return;
  if (window <= 0 || ring_slots < window) throw std::invalid_argument("csa2_window_slots_decode: ring < window");
  window_slots_decode_kernel<<<unsigned(rows), 128, 0, stream>>>(pos, rows, window, ring_slots, list, counts);
  DGPP_CUDA_OK(cudaGetLastError());
}
void csa2_dspark_window_slots(const int64_t* pos, int rows, int block, int window, int ring_slots, int32_t* list,
                              int32_t* counts, cudaStream_t stream) {
  if (rows <= 0) return;
  if (block <= 0 || rows % block != 0 || window <= 0 || ring_slots < window + block)
    throw std::invalid_argument("csa2_dspark_window_slots: rows in blocks, ring >= window + block");
  dspark_window_slots_kernel<<<unsigned(rows), 128, 0, stream>>>(pos, rows, block, window, ring_slots, list, counts);
  DGPP_CUDA_OK(cudaGetLastError());
}
void csa2_window_slots_prefill(int64_t pos0, int T, int window, int32_t* list, int32_t* counts,
                               cudaStream_t stream, int64_t floor) {
  if (T <= 0) return;
  if (pos0 < 0 || window <= 0 || floor < 0 || floor > pos0) throw std::invalid_argument("csa2_window_slots_prefill: shape");
  window_slots_prefill_kernel<<<unsigned(T), 128, 0, stream>>>(pos0, T, window, floor, list, counts);
  DGPP_CUDA_OK(cudaGetLastError());
}
void csa2_window_scratch_prologue(const void* ring, int ring_slots, int64_t pos0, int window,
                                  size_t row_bytes, void* scratch, cudaStream_t stream, int64_t floor) {
  if (window <= 1) return;
  if (row_bytes % 16 != 0 || ring_slots < window)
    throw std::invalid_argument("csa2_window_scratch_prologue: row bytes in 16s, ring >= window");
  window_scratch_prologue_kernel<<<unsigned(window - 1), 64, 0, stream>>>(
      static_cast<const uint8_t*>(ring), ring_slots, pos0, window, floor < 0 ? 0 : floor, row_bytes,
      static_cast<uint8_t*>(scratch));
  DGPP_CUDA_OK(cudaGetLastError());
}
void csa2_window_ring_writeback(const void* scratch, int window, int64_t pos0, int T, int ring_slots,
                                size_t row_bytes, void* ring, cudaStream_t stream) {
  if (T <= 0) return;
  if (row_bytes % 16 != 0) throw std::invalid_argument("csa2_window_ring_writeback: row bytes in 16s");
  const int i0 = std::max(0, T - ring_slots);
  window_ring_writeback_kernel<<<unsigned(T - i0), 64, 0, stream>>>(
      static_cast<const uint8_t*>(scratch), window, pos0, i0, T, ring_slots, row_bytes,
      static_cast<uint8_t*>(ring));
  DGPP_CUDA_OK(cudaGetLastError());
}

void csa2_index_q_quant(const void* q, int rows, int heads, void* q_fp8, float* q_scale,
                        unsigned* violations, cudaStream_t stream) {
  if (rows <= 0 || heads <= 0) return;
  index_q_quant_kernel<<<unsigned(int64_t(rows) * heads), kCsa2IndexDim, 0, stream>>>(
      static_cast<const uint16_t*>(q), static_cast<uint8_t*>(q_fp8), q_scale, violations);
  DGPP_CUDA_OK(cudaGetLastError());
}
void csa2_index_k_append(const void* k, const int32_t* req_ids, const int64_t* entries, int n,
                         const int32_t* block_tables, int blocks_per_request, int entries_per_block,
                         void* index_k, float* index_scale, unsigned* violations, cudaStream_t stream) {
  if (n <= 0) return;
  if (entries_per_block <= 0) throw std::invalid_argument("csa2_index_k_append: entries_per_block");
  index_k_append_kernel<<<unsigned(n), kCsa2IndexDim, 0, stream>>>(
      static_cast<const uint16_t*>(k), req_ids, entries, block_tables, blocks_per_request, entries_per_block,
      static_cast<uint8_t*>(index_k), index_scale, violations);
  DGPP_CUDA_OK(cudaGetLastError());
}
void csa2_fold_weights(const void* w_bf16, const float* q_scale, float* out, int64_t n, cudaStream_t stream) {
  if (n <= 0) return;
  fold_weights_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(static_cast<const uint16_t*>(w_bf16),
                                                                     q_scale, out, n);
  DGPP_CUDA_OK(cudaGetLastError());
}

void csa2_compress_pairs_prefill(const float* kv, const float* score, int T, const void* norm_w, float eps,
                                 void* latent_out, float* tail, cudaStream_t stream) {
  if (T <= 0) return;
  const int blocks = (T + 1) / 2;  // pairs, plus the odd tail's block
  compress_pairs_prefill_kernel<<<unsigned(blocks), kCsa2Threads, 0, stream>>>(
      kv, score, T, static_cast<const uint16_t*>(norm_w), eps, static_cast<uint16_t*>(latent_out), tail);
  DGPP_CUDA_OK(cudaGetLastError());
}
void csa2_compress_decode_update(const float* kv, const float* score, const int32_t* req_ids,
                                 const int64_t* pos, const int32_t* req_spans, int num_requests,
                                 const void* norm_w, float eps, float* tails, void* latent_out,
                                 int64_t* entries_out, int tokens, float* tail_snapshots,
                                 cudaStream_t stream) {
  if (num_requests <= 0 || tokens <= 0) return;
  compress_decode_update_kernel<<<unsigned(num_requests), kCsa2Threads, 0, stream>>>(
      kv, score, req_ids, pos, req_spans, static_cast<const uint16_t*>(norm_w), eps, tails,
      static_cast<uint16_t*>(latent_out), entries_out, tail_snapshots);
  DGPP_CUDA_OK(cudaGetLastError());
}

namespace {
void check_select_common(int heads, int select_k, const char* who) {
  if (heads != 32) throw std::invalid_argument(std::string(who) + ": the selection kernels pin 32 index heads");
  if (select_k <= 0 || (select_k & (select_k - 1)) != 0 || select_k > kSelectMaxK)
    throw std::invalid_argument(std::string(who) + ": select_k must be a power of two <= 2048");
}
}  // namespace

size_t csa2_select_workspace_bytes(int max_rows, int64_t max_entries) {
  if (max_rows <= 0 || max_rows > 96 || max_entries <= 0 || max_entries > (int64_t(1) << kIdxBits))
    throw std::invalid_argument(
        "csa2_select_workspace_bytes: rows in [1, 96], entries in [1, 2^21]");
  return size_t(max_rows) * size_t(max_entries) * sizeof(uint64_t);
}

void csa2_select_candidates_decode(const void* q_fp8, const float* w_folded, const int32_t* req_ids,
                                   const int64_t* pos_sel, int rows, const int32_t* block_tables,
                                   int blocks_per_request, const void* index_k,
                                   const float* index_scale, int entries_per_block, int heads,
                                   int block_size, int topk_blocks, uint64_t* keys_ws,
                                   int64_t max_entries, int32_t* cand_out, int32_t* cand_counts,
                                   cudaStream_t stream) {
  if (rows <= 0) return;
  check_select_common(heads, topk_blocks, "csa2_select_candidates_decode");
  if (block_size <= 0 || entries_per_block <= 0 || blocks_per_request <= 0 || keys_ws == nullptr)
    throw std::invalid_argument("csa2_select_candidates_decode: shape");
  if (topk_blocks > kCsa2CandidateMaxBlocks)
    throw std::invalid_argument("csa2_select_candidates_decode: topk_blocks exceeds the select bound");
  (void)csa2_select_workspace_bytes(rows, max_entries);
  const int stripes = std::max(1, 96 / rows);
  csa_decode_scores_kernel<true>
      <<<dim3(unsigned(stripes), unsigned(rows)), kCsa2Threads, 0, stream>>>(
          static_cast<const uint8_t*>(q_fp8), w_folded, req_ids, pos_sel, block_tables,
          blocks_per_request, static_cast<const uint8_t*>(index_k), index_scale, entries_per_block,
          nullptr, 0, nullptr, block_size, topk_blocks, keys_ws, max_entries);
  DGPP_CUDA_OK(cudaGetLastError());
  csa_decode_select_kernel<true>
      <<<unsigned(rows), kCsa2Threads, decode_select_smem(topk_blocks), stream>>>(
          keys_ws, max_entries, pos_sel, nullptr, 0, nullptr, block_size, topk_blocks, cand_out,
          cand_counts);
  DGPP_CUDA_OK(cudaGetLastError());
}

void csa2_select_listed_decode(const void* q_fp8, const float* w_folded, const int32_t* req_ids,
                               const int64_t* pos_sel, int rows, const int32_t* block_tables,
                               int blocks_per_request, const void* index_k,
                               const float* index_scale, int entries_per_block, int heads,
                               const int32_t* cand, int cand_stride, const int32_t* cand_counts,
                               int block_size, int select_k, uint64_t* keys_ws, int64_t max_entries,
                               int32_t* topk_out, int32_t* counts, cudaStream_t stream) {
  if (rows <= 0) return;
  // The radix path has no streaming tile: the expansion (eight rounds of
  // the 256-thread block) bounds select_k at kSelectMaxK, checked above.
  check_select_common(heads, select_k, "csa2_select_listed_decode");
  if (cand != nullptr && (cand_counts == nullptr || cand_stride <= 0 || block_size <= 0))
    throw std::invalid_argument("csa2_select_listed_decode: a candidate pool needs its counts, stride and block size");
  if (entries_per_block <= 0 || blocks_per_request <= 0 || keys_ws == nullptr)
    throw std::invalid_argument("csa2_select_listed_decode: shape or workspace");
  (void)csa2_select_workspace_bytes(rows, max_entries);
  const int stripes = std::max(1, 96 / rows);
  csa_decode_scores_kernel<false>
      <<<dim3(unsigned(stripes), unsigned(rows)), kCsa2Threads, 0, stream>>>(
          static_cast<const uint8_t*>(q_fp8), w_folded, req_ids, pos_sel, block_tables,
          blocks_per_request, static_cast<const uint8_t*>(index_k), index_scale, entries_per_block,
          cand, cand_stride, cand_counts, block_size, select_k, keys_ws, max_entries);
  DGPP_CUDA_OK(cudaGetLastError());
  csa_decode_select_kernel<false>
      <<<unsigned(rows), kCsa2Threads, decode_select_smem(select_k), stream>>>(
          keys_ws, max_entries, pos_sel, cand, cand_stride, cand_counts, block_size, select_k,
          topk_out, counts);
  DGPP_CUDA_OK(cudaGetLastError());
}

void csa2_logits_prefill(const float* dot, int64_t dot_stride, const float* w_folded, const float* k_scale,
                         const int64_t* pos_sel, int rows, int64_t n, int heads, float* logits,
                         int64_t logits_stride, cudaStream_t stream) {
  if (rows <= 0 || n <= 0) return;
  if (heads <= 0 || logits_stride < n) throw std::invalid_argument("csa2_logits_prefill: shape");
  logits_prefill_kernel<<<dim3(unsigned((n + 255) / 256), unsigned(rows)), 256, 0, stream>>>(
      dot, dot_stride, w_folded, k_scale, pos_sel, n, heads, logits, logits_stride);
  DGPP_CUDA_OK(cudaGetLastError());
}
void csa2_score_prefill(const void* q_fp8, const void* k, const float* w_folded, const float* k_scale,
                        const int64_t* pos_sel, int rows, int64_t n, int heads, float* logits,
                        int64_t logits_stride, cudaStream_t stream) {
  if (heads != 32) throw std::invalid_argument("csa2_score_prefill: 32 index heads");
  if (rows <= 0 || n < 0 || logits_stride < n) throw std::invalid_argument("csa2_score_prefill: shape");
  if (n == 0) return;
  const int64_t tiles = (n + kScoreTile - 1) / kScoreTile;
  const int64_t blocks = tiles * rows;
  if (blocks > 0x7fffffffLL) throw std::invalid_argument("csa2_score_prefill: grid");
  score_prefill_kernel<<<static_cast<unsigned>(blocks), 128, 0, stream>>>(
      static_cast<const uint8_t*>(q_fp8), static_cast<const uint8_t*>(k), w_folded, k_scale, pos_sel, rows, n, logits,
      logits_stride);
  DGPP_CUDA_OK(cudaGetLastError());
}

void csa2_select_rows_prefill(const float* logits, int64_t logits_stride, const int64_t* pos_sel, int rows,
                              int select_k, const int32_t* cand, int cand_stride, const int32_t* cand_counts,
                              int block_size, int32_t* topk_out, int32_t* counts, cudaStream_t stream) {
  if (rows <= 0) return;
  check_select_common(32, select_k, "csa2_select_rows_prefill");
  if (cand != nullptr && (cand_counts == nullptr || cand_stride <= 0 || block_size <= 0))
    throw std::invalid_argument("csa2_select_rows_prefill: a candidate pool needs its counts, stride and block size");
  csa2_prepare_kernel_smem();
  if (rows_tile(select_k) == kCandTile)
    select_rows_prefill_kernel<kCandTile><<<unsigned(rows), kCsa2Threads, rows_smem(select_k), stream>>>(
        logits, logits_stride, pos_sel, select_k, cand, cand_stride, cand_counts, block_size, topk_out, counts);
  else
    select_rows_prefill_kernel<kSelectTile><<<unsigned(rows), kCsa2Threads, rows_smem(select_k), stream>>>(
        logits, logits_stride, pos_sel, select_k, cand, cand_stride, cand_counts, block_size, topk_out, counts);
  DGPP_CUDA_OK(cudaGetLastError());
}
void csa2_select_candidates_prefill(const float* logits, int64_t logits_stride, const int64_t* pos_sel,
                                    int rows, int block_size, int topk_blocks, int32_t* cand_out,
                                    int32_t* cand_counts, cudaStream_t stream) {
  if (rows <= 0) return;
  check_select_common(32, topk_blocks, "csa2_select_candidates_prefill");
  if (block_size <= 0 || topk_blocks > kCsa2CandidateMaxBlocks)
    throw std::invalid_argument("csa2_select_candidates_prefill: shape");
  csa2_prepare_kernel_smem();
  const int cand_stride = topk_blocks;
  select_candidates_prefill_kernel<<<unsigned(rows), kCsa2Threads, cand_prefill_smem(topk_blocks), stream>>>(
      logits, logits_stride, pos_sel, block_size, topk_blocks, cand_stride, cand_out, cand_counts);
  DGPP_CUDA_OK(cudaGetLastError());
}

void csa2_attn_finish(const float* m_main, const float* l_main, const float* c_main, int n_main,
                      const float* m_win, const float* l_win, const float* c_win, int n_win,
                      const float* sink, int rows, int local_heads, const int64_t* pos,
                      const float* inv_freq, void* out, cudaStream_t stream) {
  if (rows <= 0) return;
  if (local_heads <= 0 || n_main < 0 || n_win < 0 || (n_main > 0 && (m_main == nullptr || c_main == nullptr)) ||
      (n_win > 0 && (m_win == nullptr || c_win == nullptr)))
    throw std::invalid_argument("csa2_attn_finish: shape");
  if (n_main + n_win > kFinishMaxSplits) throw std::invalid_argument("csa2_attn_finish: more splits than the kernel stages");
  if ((n_main > 0 && reinterpret_cast<uintptr_t>(c_main) % 16 != 0) || (n_win > 0 && reinterpret_cast<uintptr_t>(c_win) % 16 != 0))
    throw std::invalid_argument("csa2_attn_finish: the partial rows are read as 16-byte vectors");
  attn_finish_kernel<<<dim3(unsigned(rows), unsigned(local_heads)), 128, 0, stream>>>(
      m_main, l_main, c_main, n_main, m_win, l_win, c_win, n_win, sink, local_heads, pos, inv_freq,
      static_cast<uint16_t*>(out));
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
