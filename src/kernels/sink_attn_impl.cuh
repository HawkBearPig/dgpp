#pragma once
// The sink-attention kernels (kernels/sink_attn.hpp) as templates over the
// head geometry: DK the qk width, R the rotary width — <192, 64> MiMo's
// (a 64-wide rotary slice, bf16 rotary ops, nope dims after), <128, 128>
// Flint's (the whole head rotated with fp32 cos / sin and one rounding, or
// not at all, after an optional per-head one-rounding RMSNorm). The v
// width is 128 for both. The MiMo instantiation is the 2026-09-22 kernel,
// op for op (its kernel tests pin it); the Flint one changes the finish's
// lane layout and arithmetic alone (HeadLane below).
#include <cmath>
#include <stdexcept>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/latent_format.hpp"
#include "kernels/pdl.cuh"
#include "kernels/sink_attn.hpp"

namespace dgpp {
namespace sink_attn {

constexpr int DV = kSinkAttnVDim;
constexpr int kTile = kSinkAttnTile;
constexpr int kFusedMaxRows = kSinkAttnFusedMaxRows;
static_assert(DV == 4 * 32, "the kernels' lane mapping: 4 v dims per lane");

__device__ __forceinline__ float round_bf16(float v) {
  return bf16_bits_to_float(float_to_bf16_bits(v));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(~0u, v, off);
  return v;
}
__device__ __forceinline__ double warp_sum(double v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(~0u, v, off);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, off));
  return v;
}

__device__ __forceinline__ uint2 pack4(const float* x) {
  uint2 word;
  word.x = static_cast<uint32_t>(float_to_bf16_bits(x[0])) | (static_cast<uint32_t>(float_to_bf16_bits(x[1])) << 16);
  word.y = static_cast<uint32_t>(float_to_bf16_bits(x[2])) | (static_cast<uint32_t>(float_to_bf16_bits(x[3])) << 16);
  return word;
}

// ---- the head lane layouts ------------------------------------------------------
// A q or k head's values are split over the warp's lanes, kN per lane;
// `load` reads a lane's raw fp32 dots, `finish` turns them into the
// finished bf16-valued fp32 values (the Linear's rounding, the norm, the
// rotation), `offsets` names the head columns the lane's values occupy.
template <int DK, int R>
struct HeadLane;

// MiMo (qk 192, rotary 64): lane l owns the rotary pair (l, l + 32) — the
// pair of transformers' rotate_half over the 64-wide slice — and the nope
// dims [64 + 4l, 64 + 4l + 4). The rotation with the reference's bf16 ops:
// cos / sin bf16 from fp32 pos x inv_freq, x*cos and rotate(x)*sin
// rounded, the sum rounded.
template <>
struct HeadLane<192, 64> {
  static constexpr int kN = 6;
  static constexpr int DK = 192, R = 64;
  __device__ static __forceinline__ void offsets(int lane, int (&off)[kN]) {
    off[0] = lane;
    off[1] = lane + R / 2;
#pragma unroll
    for (int i = 0; i < 4; ++i) off[2 + i] = R + lane * 4 + i;
  }
  __device__ static __forceinline__ void load(const float* __restrict__ src, int lane, float (&raw)[kN]) {
    raw[0] = src[lane];
    raw[1] = src[lane + R / 2];
#pragma unroll
    for (int i = 0; i < 4; ++i) raw[2 + i] = src[R + lane * 4 + i];
  }
  // inv_freq fp32 [32]; no norm on this geometry (norm_w ignored).
  __device__ static __forceinline__ void finish(const float* raw, int64_t p, const float* __restrict__ inv_freq,
                                                const uint16_t* __restrict__, float, int lane, float (&x)[kN]) {
    const float a = round_bf16(raw[0]);
    const float b = round_bf16(raw[1]);
    const float ang = __fmul_rn(static_cast<float>(p), inv_freq[lane]);
    const float c = round_bf16(cosf(ang));
    const float s = round_bf16(sinf(ang));
    x[0] = round_bf16(round_bf16(a * c) + round_bf16(-b * s));
    x[1] = round_bf16(round_bf16(b * c) + round_bf16(a * s));
#pragma unroll
    for (int i = 0; i < 4; ++i) x[2 + i] = round_bf16(raw[2 + i]);
  }
};

// Flint (qk 128, rotary 128): lane l owns dims {l, l + 32, l + 64, l + 96};
// the rotate_half pairs over the whole head are (l, l + 64) and (l + 32,
// l + 96) — both on the lane. The finish: bf16(dot), then the per-head
// one-rounding RMSNorm (fp32 x * rstd * w, the sum of squares in fp64 —
// exact for a head of bf16 squares — narrowed once), then the rotation
// with fp32 cos / sin of fp32 pos x inv_freq and one rounding of each
// rotated element (the reference's fp32 rope); inv_freq null: no rotation.
template <>
struct HeadLane<128, 128> {
  static constexpr int kN = 4;
  static constexpr int DK = 128, R = 128;
  __device__ static __forceinline__ void offsets(int lane, int (&off)[kN]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) off[i] = lane + 32 * i;
  }
  __device__ static __forceinline__ void load(const float* __restrict__ src, int lane, float (&raw)[kN]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) raw[i] = src[lane + 32 * i];
  }
  __device__ static __forceinline__ void finish(const float* raw, int64_t p, const float* __restrict__ inv_freq,
                                                const uint16_t* __restrict__ norm_w, float eps, int lane,
                                                float (&x)[kN]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = round_bf16(raw[i]);
    if (norm_w != nullptr) {
      double ss = 0.0;
#pragma unroll
      for (int i = 0; i < 4; ++i) ss = fma(static_cast<double>(x[i]), static_cast<double>(x[i]), ss);
      ss = warp_sum(ss);
      const float rstd = rsqrtf(static_cast<float>(ss / DK) + eps);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        x[i] = round_bf16(__fmul_rn(__fmul_rn(x[i], rstd), bf16_bits_to_float(norm_w[lane + 32 * i])));
    }
    if (inv_freq != nullptr) {
      const float pf = static_cast<float>(p);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        // Pair j: dims (lane + 32 j, lane + 32 j + 64) — rotary index lane + 32 j.
        const float ang = __fmul_rn(pf, inv_freq[lane + 32 * j]);
        const float c = cosf(ang), s = sinf(ang);
        const float x1 = x[j], x2 = x[j + 2];
        x[j] = round_bf16(__fsub_rn(__fmul_rn(x1, c), __fmul_rn(x2, s)));
        x[j + 2] = round_bf16(__fadd_rn(__fmul_rn(x2, c), __fmul_rn(x1, s)));
      }
    }
  }
};

// The fp32 column of head h in the fused projection's row: its chunk, then
// its offset inside the chunk's [Q | K | V] rows.
template <int DK>
__device__ __forceinline__ int64_t q_col0(const SinkQkvLayout& layout, int h) {
  const int chunk = h / layout.q_per_chunk;
  return chunk * layout.chunk_stride + static_cast<int64_t>(h - chunk * layout.q_per_chunk) * DK;
}
template <int DK>
__device__ __forceinline__ int64_t kv_col0(const SinkQkvLayout& layout, int h, bool v) {
  const int chunk = h / layout.kv_per_chunk;
  const int64_t k0 = static_cast<int64_t>(layout.q_per_chunk) * DK;
  const int64_t v0 = k0 + static_cast<int64_t>(layout.kv_per_chunk) * DK;
  return chunk * layout.chunk_stride + (v ? v0 : k0) +
         static_cast<int64_t>(h - chunk * layout.kv_per_chunk) * (v ? DV : DK);
}

// A head row of `n` finished values, one lane per `n / 32` of them, into
// the fp8 cache: the warp's absmax, the row scale, the codes (the fp8 row
// form of kernels/latent_format.hpp), the scale by lane 0.
template <int kPerLane>
__device__ __forceinline__ void store_fp8_head(const float* x, uint8_t* dst_codes, const int* dst_off,
                                               float* dst_scale, int lane) {
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) amax = fmaxf(amax, fabsf(x[i]));
  amax = warp_max(amax);
  const LatentFp8Scale sc = latent_fp8_row_scale(amax);
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) dst_codes[dst_off[i]] = latent_fp8_encode(x[i], sc.inv);
  if (lane == 0) *dst_scale = sc.scale;
}

// A v head's lane share: dims [4 lane, 4 lane + 4): bf16(dot), then bf16(v
// x scale) — the reference's two roundings (scale 1: one).
__device__ __forceinline__ void finish_v_lane(const float* __restrict__ src, float value_scale, int lane, float* x) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float v = round_bf16(src[lane * 4 + i]);
    if (value_scale != 1.0f) v = round_bf16(v * value_scale);
    x[i] = v;
  }
}

// ---- qkv finish ---------------------------------------------------------------
// One warp per (row, head); blockDim 256 = 8 (row, head) items.
template <int DK, int R, bool kFp8>
__global__ void qkv_finish_kernel(const float* __restrict__ qkv, int64_t qkv_stride, SinkQkvLayout layout,
                                  SinkFinishParams fin, const int32_t* __restrict__ req_ids,
                                  const int64_t* __restrict__ pos, int rows, int local_heads, int kv_heads,
                                  const int32_t* __restrict__ block_tables, int blocks_per_request,
                                  int block_tokens, uint16_t* __restrict__ q_out, int64_t q_out_stride,
                                  uint16_t* __restrict__ k_cache, uint16_t* __restrict__ v_cache,
                                  float* __restrict__ k_scale, float* __restrict__ v_scale) {
  using HL = HeadLane<DK, R>;
  dgpp_pdl_wait();
  const int heads_total = local_heads + 2 * kv_heads;
  const int item = static_cast<int>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  if (item >= rows * heads_total) return;
  const int r = item / heads_total;
  const int hh = item - r * heads_total;
  const int64_t p = pos[r];
  if (p < 0) return;
  const int lane = threadIdx.x % 32;
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
  const int64_t col0 = kind == 0 ? q_col0<DK>(layout, h) : kv_col0<DK>(layout, h, kind == 2);
  const float* src = qkv + static_cast<int64_t>(r) * qkv_stride + col0;
  int64_t phys = 0;
  if (kind != 0) {
    const int32_t blk = block_tables[static_cast<int64_t>(req_ids[r]) * blocks_per_request + p / block_tokens];
    phys = static_cast<int64_t>(blk) * block_tokens + p % block_tokens;
  }
  if (kind == 2) {
    float x[4];
    finish_v_lane(src, fin.value_scale, lane, x);
    if constexpr (kFp8) {
      uint8_t* row = reinterpret_cast<uint8_t*>(v_cache) + phys * (static_cast<int64_t>(kv_heads) * DV) + h * DV;
      const int off[4] = {lane * 4, lane * 4 + 1, lane * 4 + 2, lane * 4 + 3};
      store_fp8_head<4>(x, row, off, v_scale + phys * kv_heads + h, lane);
    } else {
      *reinterpret_cast<uint2*>(v_cache + phys * (static_cast<int64_t>(kv_heads) * DV) + h * DV + lane * 4) = pack4(x);
    }
    return;
  }
  // q or k: the lane's share, finished.
  float raw[HL::kN], x[HL::kN];
  HL::load(src, lane, raw);
  HL::finish(raw, p, fin.inv_freq, kind == 0 ? fin.q_norm : fin.k_norm, fin.eps, lane, x);
  int off[HL::kN];
  HL::offsets(lane, off);
  if constexpr (kFp8) {
    if (kind == 1) {
      uint8_t* row = reinterpret_cast<uint8_t*>(k_cache) + phys * (static_cast<int64_t>(kv_heads) * DK) + h * DK;
      store_fp8_head<HL::kN>(x, row, off, k_scale + phys * kv_heads + h, lane);
      return;
    }
  }
  uint16_t* dst = kind == 0 ? q_out + static_cast<int64_t>(r) * q_out_stride + h * DK
                            : k_cache + phys * (static_cast<int64_t>(kv_heads) * DK) + h * DK;
  if constexpr (DK == 192) {
    dst[off[0]] = float_to_bf16_bits(x[0]);
    dst[off[1]] = float_to_bf16_bits(x[1]);
    *reinterpret_cast<uint2*>(dst + off[2]) = pack4(x + 2);
  } else {
#pragma unroll
    for (int i = 0; i < HL::kN; ++i) dst[off[i]] = float_to_bf16_bits(x[i]);
  }
}

// ---- the split-KV paged GQA attention ------------------------------------------
// Block: one warp per query head of the kv head (hpk warps), lane t owns
// tile token t for the scores and dims [4l, 4l + 4) of the 128 v dims for
// the accumulation. K tile rows are padded to kRowStride elements (DK +
// 8: 16-byte aligned rows, the score loop reads them as uint4; at 192 a
// row is 100 words = 4 banks, at 128 68 words — eight lanes per 128-bit
// phase cover the 32 banks once either way — conflict-free).
template <int DK>
struct Geo {
  static constexpr int kRowStride = DK + 8;
  static_assert((kRowStride * 2) % 16 == 0, "K tile rows must be 16-byte aligned");
};

// Sixteen e4m3 codes (one 16-byte piece) of a cached row, decoded with
// the row's scale into eight bf16 words.
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

// The dot of a staged K row (bf16, 16-byte aligned) with the head's fp32
// query: the fixed fp32 FMA order over d, eight elements per vector read.
template <int DK>
__device__ __forceinline__ float tile_score(const uint16_t* __restrict__ krow_bf16, const float* __restrict__ myq) {
  const uint4* krow = reinterpret_cast<const uint4*>(krow_bf16);
  const float4* q4 = reinterpret_cast<const float4*>(myq);
  float acc = 0.f;
#pragma unroll 4
  for (int v = 0; v < DK / 8; ++v) {
    const uint4 kk = krow[v];
    const float4 qa = q4[2 * v], qb = q4[2 * v + 1];
    acc = fmaf(qa.x, bf16_bits_to_float(static_cast<uint16_t>(kk.x & 0xFFFFu)), acc);
    acc = fmaf(qa.y, bf16_bits_to_float(static_cast<uint16_t>(kk.x >> 16)), acc);
    acc = fmaf(qa.z, bf16_bits_to_float(static_cast<uint16_t>(kk.y & 0xFFFFu)), acc);
    acc = fmaf(qa.w, bf16_bits_to_float(static_cast<uint16_t>(kk.y >> 16)), acc);
    acc = fmaf(qb.x, bf16_bits_to_float(static_cast<uint16_t>(kk.z & 0xFFFFu)), acc);
    acc = fmaf(qb.y, bf16_bits_to_float(static_cast<uint16_t>(kk.z >> 16)), acc);
    acc = fmaf(qb.z, bf16_bits_to_float(static_cast<uint16_t>(kk.w & 0xFFFFu)), acc);
    acc = fmaf(qb.w, bf16_bits_to_float(static_cast<uint16_t>(kk.w >> 16)), acc);
  }
  return acc;
}

template <int DK, bool kFp8>
__global__ void attn_partial_kernel(const uint16_t* __restrict__ q, int64_t q_stride,
                                    const uint16_t* __restrict__ k_cache,
                                    const uint16_t* __restrict__ v_cache,
                                    const int32_t* __restrict__ req_ids,
                                    const int64_t* __restrict__ pos, int n_split,
                                    int local_heads, int kv_heads, int block_tokens,
                                    const int32_t* __restrict__ block_tables,
                                    int blocks_per_request, int window, float scale,
                                    const float* __restrict__ sink,
                                    float* __restrict__ m_ws, float* __restrict__ l_ws,
                                    float* __restrict__ c_ws,
                                    const float* __restrict__ k_scale,
                                    const float* __restrict__ v_scale) {
  constexpr int kRowStride = Geo<DK>::kRowStride;
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint16_t smem[];
  const int hpk = local_heads / kv_heads;
  const int r = static_cast<int>(blockIdx.x);
  const int s = static_cast<int>(blockIdx.y);
  const int kvh = static_cast<int>(blockIdx.z);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int h = kvh * hpk + warp;  // this warp's query head
  const int64_t base_m = (static_cast<int64_t>(r) * n_split + s) * local_heads;
  uint16_t* kt = smem;                                    // [kTile][kRowStride]
  uint16_t* vt = kt + kTile * kRowStride;                 // [kTile][DV]
  float* qs = reinterpret_cast<float*>(vt + kTile * DV);  // [hpk][DK] the heads' queries, fp32
  float* pbuf = qs + hpk * DK;                            // [hpk][kTile] the tile's bf16-rounded probabilities

  const int64_t p = pos[r];
  int64_t t_begin = 0, t_end = 0;
  if (p >= 0) {
    const int64_t t_lo = window > 0 ? (p - window + 1 > 0 ? p - window + 1 : 0) : 0;
    const int64_t visible = p - t_lo + 1;
    const int64_t tiles = (visible + kTile - 1) / kTile;
    const int64_t chunk = (tiles + n_split - 1) / n_split;
    const int64_t tile0 = static_cast<int64_t>(s) * chunk;
    const int64_t tile1 = tile0 + chunk < tiles ? tile0 + chunk : tiles;
    if (tile0 < tile1) {
      t_begin = t_lo + tile0 * kTile;
      const int64_t e = t_lo + tile1 * kTile;
      t_end = e < p + 1 ? e : p + 1;
    }
  }
  if (t_begin >= t_end) {
    // An empty split (or a padding row): the neutral partial.
    for (int i = threadIdx.x; i < hpk * DV; i += blockDim.x)
      c_ws[(base_m + kvh * hpk + i / DV) * DV + i % DV] = 0.0f;
    if (threadIdx.x < hpk) {
      m_ws[base_m + kvh * hpk + threadIdx.x] = -INFINITY;
      l_ws[base_m + kvh * hpk + threadIdx.x] = 0.0f;
    }
    return;
  }
  // The head's query into fp32 smem (every lane reads every dim later).
  {
    const uint16_t* qrow = q + static_cast<int64_t>(r) * q_stride + static_cast<int64_t>(h) * DK;
    for (int d = lane; d < DK; d += 32) qs[warp * DK + d] = bf16_bits_to_float(qrow[d]);
  }
  __syncthreads();
  const int32_t* bt = block_tables + static_cast<int64_t>(req_ids[r]) * blocks_per_request;
  const int kwidth = kv_heads * DK, vwidth = kv_heads * DV;
  // The sink: one more softmax column with logit sink[h] and no value,
  // carried by split 0 from before its first tile.
  float m_run = -INFINITY, l_run = 0.0f;
  if (s == 0 && sink != nullptr) {
    m_run = sink[h];
    l_run = 1.0f;
  }
  float creg[4] = {0.f, 0.f, 0.f, 0.f};
  const float* myq = qs + warp * DK;
  for (int64_t t0 = t_begin; t0 < t_end; t0 += kTile) {
    const int n = t_end - t0 < kTile ? static_cast<int>(t_end - t0) : kTile;
    __syncthreads();  // the previous tile's readers are done
    if constexpr (kFp8) {
      // Stage the tile's K rows (DK / 16 16-byte pieces of 16 codes each)
      // and V rows (8), each piece decoded with its row's scale.
      const uint8_t* kc = reinterpret_cast<const uint8_t*>(k_cache);
      const uint8_t* vc = reinterpret_cast<const uint8_t*>(v_cache);
      for (int idx = threadIdx.x; idx < n * (DK / 16); idx += blockDim.x) {
        const int tt = idx / (DK / 16);
        const int c16 = idx - tt * (DK / 16);
        const int64_t tok = t0 + tt;
        const int32_t blk = bt[tok / block_tokens];
        const int64_t phys = static_cast<int64_t>(blk) * block_tokens + tok % block_tokens;
        const uint4 codes = *reinterpret_cast<const uint4*>(kc + phys * kwidth + kvh * DK + c16 * 16);
        uint32_t words[8];
        decode_fp8_piece(codes, k_scale[phys * kv_heads + kvh], words);
        uint32_t* dst = reinterpret_cast<uint32_t*>(kt + tt * kRowStride) + c16 * 8;
#pragma unroll
        for (int w = 0; w < 8; ++w) dst[w] = words[w];
      }
      for (int idx = threadIdx.x; idx < n * (DV / 16); idx += blockDim.x) {
        const int tt = idx / (DV / 16);
        const int c16 = idx - tt * (DV / 16);
        const int64_t tok = t0 + tt;
        const int32_t blk = bt[tok / block_tokens];
        const int64_t phys = static_cast<int64_t>(blk) * block_tokens + tok % block_tokens;
        const uint4 codes = *reinterpret_cast<const uint4*>(vc + phys * vwidth + kvh * DV + c16 * 16);
        uint32_t words[8];
        decode_fp8_piece(codes, v_scale[phys * kv_heads + kvh], words);
        uint4* dst = reinterpret_cast<uint4*>(vt + tt * DV + c16 * 16);
        dst[0] = make_uint4(words[0], words[1], words[2], words[3]);
        dst[1] = make_uint4(words[4], words[5], words[6], words[7]);
      }
    } else {
    // Stage the tile's K rows (DK / 8 16-byte pieces each) and V rows (16).
    for (int idx = threadIdx.x; idx < n * (DK / 8); idx += blockDim.x) {
      const int tt = idx / (DK / 8);
      const int c8 = idx - tt * (DK / 8);
      const int64_t tok = t0 + tt;
      const int32_t blk = bt[tok / block_tokens];
      const int64_t phys = static_cast<int64_t>(blk) * block_tokens + tok % block_tokens;
      const uint4 val = *reinterpret_cast<const uint4*>(k_cache + phys * kwidth + kvh * DK + c8 * 8);
      uint32_t* dst = reinterpret_cast<uint32_t*>(kt + tt * kRowStride) + c8 * 4;
      dst[0] = val.x;
      dst[1] = val.y;
      dst[2] = val.z;
      dst[3] = val.w;
    }
    for (int idx = threadIdx.x; idx < n * (DV / 8); idx += blockDim.x) {
      const int tt = idx / (DV / 8);
      const int c8 = idx - tt * (DV / 8);
      const int64_t tok = t0 + tt;
      const int32_t blk = bt[tok / block_tokens];
      const int64_t phys = static_cast<int64_t>(blk) * block_tokens + tok % block_tokens;
      *reinterpret_cast<uint4*>(vt + tt * DV + c8 * 8) =
          *reinterpret_cast<const uint4*>(v_cache + phys * vwidth + kvh * DV + c8 * 8);
    }
    }
    __syncthreads();
    // Scores: lane t against tile token t (a fixed fp32 order over d).
    float score = -INFINITY;
    if (lane < n) {
      score = tile_score<DK>(kt + lane * kRowStride, myq) * scale;
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
      const float pt = pbuf[warp * kTile + tt];
      const uint2 vv = *reinterpret_cast<const uint2*>(vt + tt * DV + lane * 4);
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
  float* crow = c_ws + (base_m + h) * DV + lane * 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) crow[i] = creg[i];
}

__global__ void attn_combine_kernel(const float* __restrict__ m_ws, const float* __restrict__ l_ws,
                                    const float* __restrict__ c_ws, int n_split, int local_heads,
                                    uint16_t* __restrict__ out) {
  dgpp_pdl_wait();
  const int r = static_cast<int>(blockIdx.x);
  const int h = static_cast<int>(blockIdx.y);
  const int d = threadIdx.x;  // DV threads
  const int64_t base = (static_cast<int64_t>(r) * n_split) * local_heads + h;
  float M = -INFINITY;
  for (int s = 0; s < n_split; ++s) M = fmaxf(M, m_ws[base + static_cast<int64_t>(s) * local_heads]);
  float L = 0.f, C = 0.f;
  if (M > -INFINITY) {
    for (int s = 0; s < n_split; ++s) {
      const int64_t i = base + static_cast<int64_t>(s) * local_heads;
      const float w = expf(m_ws[i] - M);
      L = fmaf(l_ws[i], w, L);
      C = fmaf(c_ws[i * DV + d], w, C);
    }
  }
  out[(static_cast<int64_t>(r) * local_heads + h) * DV + d] = float_to_bf16_bits(L > 0.f ? C / L : 0.f);
}

// ---- the fused decode attention ------------------------------------------------
// finish + partial + combine in one launch over a decode batch (plan §7.1):
// the partial kernel's block (row, split, kv head), which now
//   - finishes its own query heads from the fp32 projection (no q buffer),
//   - (split 0) finishes the row's K/V head into the cache,
//   - OVERLAYS, on every staged tile, the finished K/V of any batch row of
//     the same request whose position falls in the tile — computed from the
//     fp32 projection (fp8 cache: through the codec's round trip), never
//     read from the cache: those slots are being written by other blocks
//     of this launch (the batched draft rows attend their request's earlier
//     rows), and the three-kernel chain's order (all K/V appended before
//     any attention) is reproduced by recomputing instead of waiting;
//   - arrives on the (row, kv head) counter after its partials; the last
//     block to arrive combines the splits (the combine kernel's arithmetic,
//     op for op; the neutral splits contribute nothing and are skipped by
//     their m = -inf, so the empty blocks write only m and l) and resets
//     the counter for the next launch.
// The batch's (request, position) table sits in shared memory: at most
// kFusedMaxRows rows (the prefill path keeps the three kernels).

// Row `i` of the batch, kv head kvh: its finished K and V (this warp) into
// the staged tile row tt — the values the cache holds for it.
template <int DK, int R, bool kFp8>
__device__ __forceinline__ void overlay_row(const SinkAttnFusedArgs& a, int i, int64_t pi, int kvh, int tt, int lane,
                                            uint16_t* kt, uint16_t* vt) {
  using HL = HeadLane<DK, R>;
  constexpr int kRowStride = Geo<DK>::kRowStride;
  const float* krow = a.qkv + static_cast<int64_t>(i) * a.qkv_stride + kv_col0<DK>(a.layout, kvh, false);
  const float* vrow = a.qkv + static_cast<int64_t>(i) * a.qkv_stride + kv_col0<DK>(a.layout, kvh, true);
  float raw[HL::kN], kx[HL::kN], v[4];
  HL::load(krow, lane, raw);
  HL::finish(raw, pi, a.inv_freq, a.k_norm, a.eps, lane, kx);
  int off[HL::kN];
  HL::offsets(lane, off);
  finish_v_lane(vrow, a.value_scale, lane, v);
  uint16_t* kd = kt + tt * kRowStride;
  uint16_t* vd = vt + tt * DV;
  if constexpr (kFp8) {
    // The codec's round trip: the row scale from the warp's absmax, the
    // e4m3 codes, decoded with the stored scale.
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < HL::kN; ++j) amax = fmaxf(amax, fabsf(kx[j]));
    amax = warp_max(amax);
    const LatentFp8Scale ks = latent_fp8_row_scale(amax);
#pragma unroll
    for (int j = 0; j < HL::kN; ++j) kd[off[j]] = latent_fp8_decode_bf16(latent_fp8_encode(kx[j], ks.inv), ks.scale);
    float vmax = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) vmax = fmaxf(vmax, fabsf(v[j]));
    vmax = warp_max(vmax);
    const LatentFp8Scale vs = latent_fp8_row_scale(vmax);
#pragma unroll
    for (int j = 0; j < 4; ++j) vd[lane * 4 + j] = latent_fp8_decode_bf16(latent_fp8_encode(v[j], vs.inv), vs.scale);
  } else {
#pragma unroll
    for (int j = 0; j < HL::kN; ++j) kd[off[j]] = float_to_bf16_bits(kx[j]);
#pragma unroll
    for (int j = 0; j < 4; ++j) vd[lane * 4 + j] = float_to_bf16_bits(v[j]);
  }
}

// A thread's share of one tile's cache pieces (16 bytes each): the K rows
// hold DK / 16 (fp8) or DK / 8 (bf16) pieces, the V rows DV / 16 or DV / 8;
// a tile has 32 rows; the block has hpk x 32 threads with 4 <= hpk <= 16
// (the launcher's contract: 128 to 512 threads), so at most kKPieces /
// kVPieces per thread.
constexpr int kFusedMinThreads = 128;
constexpr int kFusedMaxThreads = 512;
template <int DK, bool kFp8>
struct TilePieces {
  static constexpr int kK = kFp8 ? DK / 16 : DK / 8;  // pieces per K row
  static constexpr int kV = kFp8 ? DV / 16 : DV / 8;  // pieces per V row
  static constexpr int kKPieces = (kTile * kK + kFusedMinThreads - 1) / kFusedMinThreads;
  static constexpr int kVPieces = (kTile * kV + kFusedMinThreads - 1) / kFusedMinThreads;
  uint4 k[kKPieces];
  uint4 v[kVPieces];
  float ks[kKPieces];  // fp8: the piece's row scale
  float vs[kVPieces];
};
// Block-table entries a split's range can span (checked by the launcher):
// at 32 splits over 64-token blocks a split covers ctx / 32 tokens, so
// 1024 entries reach a 2M-token pool (the 128K template needs 65, the
// 256K one 129; 4 KB of shared memory).
constexpr int kFusedMaxBt = 1024;

__device__ __forceinline__ int64_t tile_phys(const SinkAttnFusedArgs& a, const int32_t* s_bt, int64_t bt_first,
                                             int64_t tok) {
  const int32_t blk = s_bt[tok / a.block_tokens - bt_first];
  return static_cast<int64_t>(blk) * a.block_tokens + tok % a.block_tokens;
}

// The loads of tile t0 (rows [t0, min(t0 + 32, t_end))) into `pc`.
template <int DK, bool kFp8>
__device__ __forceinline__ void load_tile_pieces(const SinkAttnFusedArgs& a, const int32_t* s_bt, int64_t bt_first,
                                                 int kvh, int64_t t0, int64_t t_end, TilePieces<DK, kFp8>& pc) {
  using P = TilePieces<DK, kFp8>;
  const int n = t_end - t0 < kTile ? static_cast<int>(t_end - t0) : kTile;
  const int kwidth = a.kv_heads * DK, vwidth = a.kv_heads * DV;
  const int kpb = kFp8 ? 16 : 8;  // elements per piece
#pragma unroll
  for (int j = 0; j < P::kKPieces; ++j) {
    const int idx = j * blockDim.x + threadIdx.x;
    if (idx >= n * P::kK) continue;
    const int tt = idx / P::kK, c = idx - tt * P::kK;
    const int64_t phys = tile_phys(a, s_bt, bt_first, t0 + tt);
    if constexpr (kFp8) {
      pc.k[j] = *reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(a.k_cache) + phys * kwidth + kvh * DK + c * 16);
      pc.ks[j] = a.k_scale[phys * a.kv_heads + kvh];
    } else {
      pc.k[j] = *reinterpret_cast<const uint4*>(a.k_cache + phys * kwidth + kvh * DK + c * kpb);
    }
  }
#pragma unroll
  for (int j = 0; j < P::kVPieces; ++j) {
    const int idx = j * blockDim.x + threadIdx.x;
    if (idx >= n * P::kV) continue;
    const int tt = idx / P::kV, c = idx - tt * P::kV;
    const int64_t phys = tile_phys(a, s_bt, bt_first, t0 + tt);
    if constexpr (kFp8) {
      pc.v[j] = *reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(a.v_cache) + phys * vwidth + kvh * DV + c * 16);
      pc.vs[j] = a.v_scale[phys * a.kv_heads + kvh];
    } else {
      pc.v[j] = *reinterpret_cast<const uint4*>(a.v_cache + phys * vwidth + kvh * DV + c * kpb);
    }
  }
}

// The pieces of tile t0 into the shared tile (fp8: decoded with their row
// scales — the partial kernel's staging, op for op).
template <int DK, bool kFp8>
__device__ __forceinline__ void store_tile_pieces(int64_t t0, int64_t t_end, const TilePieces<DK, kFp8>& pc,
                                                  uint16_t* kt, uint16_t* vt) {
  using P = TilePieces<DK, kFp8>;
  constexpr int kRowStride = Geo<DK>::kRowStride;
  const int n = t_end - t0 < kTile ? static_cast<int>(t_end - t0) : kTile;
#pragma unroll
  for (int j = 0; j < P::kKPieces; ++j) {
    const int idx = j * blockDim.x + threadIdx.x;
    if (idx >= n * P::kK) continue;
    const int tt = idx / P::kK, c = idx - tt * P::kK;
    if constexpr (kFp8) {
      uint32_t words[8];
      decode_fp8_piece(pc.k[j], pc.ks[j], words);
      uint32_t* dst = reinterpret_cast<uint32_t*>(kt + tt * kRowStride) + c * 8;
#pragma unroll
      for (int w = 0; w < 8; ++w) dst[w] = words[w];
    } else {
      uint32_t* dst = reinterpret_cast<uint32_t*>(kt + tt * kRowStride) + c * 4;
      dst[0] = pc.k[j].x;
      dst[1] = pc.k[j].y;
      dst[2] = pc.k[j].z;
      dst[3] = pc.k[j].w;
    }
  }
#pragma unroll
  for (int j = 0; j < P::kVPieces; ++j) {
    const int idx = j * blockDim.x + threadIdx.x;
    if (idx >= n * P::kV) continue;
    const int tt = idx / P::kV, c = idx - tt * P::kV;
    if constexpr (kFp8) {
      uint32_t words[8];
      decode_fp8_piece(pc.v[j], pc.vs[j], words);
      uint4* dst = reinterpret_cast<uint4*>(vt + tt * DV + c * 16);
      dst[0] = make_uint4(words[0], words[1], words[2], words[3]);
      dst[1] = make_uint4(words[4], words[5], words[6], words[7]);
    } else {
      *reinterpret_cast<uint4*>(vt + tt * DV + c * 8) = pc.v[j];
    }
  }
}

template <int DK, int R, bool kFp8>
__global__ void __launch_bounds__(kFusedMaxThreads) attn_fused_kernel(SinkAttnFusedArgs a) {
  using HL = HeadLane<DK, R>;
  constexpr int kRowStride = Geo<DK>::kRowStride;
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint16_t smem[];
  __shared__ int s_last;
  const int hpk = a.local_heads / a.kv_heads;
  const int r = static_cast<int>(blockIdx.x);
  const int s = static_cast<int>(blockIdx.y);
  const int kvh = static_cast<int>(blockIdx.z);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int h = kvh * hpk + warp;  // this warp's query head
  const int64_t base_m = (static_cast<int64_t>(r) * a.n_split + s) * a.local_heads;
  uint16_t* kt = smem;                                    // [kTile][kRowStride]
  uint16_t* vt = kt + kTile * kRowStride;                 // [kTile][DV]
  float* qs = reinterpret_cast<float*>(vt + kTile * DV);  // [hpk][DK]
  float* pbuf = qs + hpk * DK;                            // [hpk][kTile]
  int32_t* s_req = reinterpret_cast<int32_t*>(pbuf + hpk * kTile);   // [kFusedMaxRows]
  int64_t* s_pos = reinterpret_cast<int64_t*>(s_req + kFusedMaxRows);  // [kFusedMaxRows]
  uint16_t* own_k = reinterpret_cast<uint16_t*>(s_pos + kFusedMaxRows);  // [DK] the row's finished K head (cache form)
  uint16_t* own_v = own_k + DK;                                          // [DV]
  // The projection loads first — they depend on nothing — so their latency
  // overlaps the batch table's and the tile geometry's: this warp's query
  // head, and (warps 0 and 1) the row's K and V heads.
  const float* qkv_row = a.qkv + static_cast<int64_t>(r) * a.qkv_stride;
  float q_raw[HL::kN], kv_raw[HL::kN > 4 ? HL::kN : 4];
  HL::load(qkv_row + q_col0<DK>(a.layout, h), lane, q_raw);
  if (warp == 0) HL::load(qkv_row + kv_col0<DK>(a.layout, kvh, false), lane, kv_raw);
  if (warp == 1) {
#pragma unroll
    for (int i = 0; i < 4; ++i) kv_raw[i] = qkv_row[kv_col0<DK>(a.layout, kvh, true) + lane * 4 + i];
  }
  for (int i = threadIdx.x; i < a.rows; i += blockDim.x) {
    s_req[i] = a.req_ids[i];
    s_pos[i] = a.pos[i];
  }
  const int64_t p = a.pos[r];
  const int32_t req = a.req_ids[r];
  int64_t t_begin = 0, t_end = 0;
  if (p >= 0) {
    const int64_t t_lo = a.window > 0 ? (p - a.window + 1 > 0 ? p - a.window + 1 : 0) : 0;
    const int64_t visible = p - t_lo + 1;
    const int64_t tiles = (visible + kTile - 1) / kTile;
    const int64_t chunk = (tiles + a.n_split - 1) / a.n_split;
    const int64_t tile0 = static_cast<int64_t>(s) * chunk;
    const int64_t tile1 = tile0 + chunk < tiles ? tile0 + chunk : tiles;
    if (tile0 < tile1) {
      t_begin = t_lo + tile0 * kTile;
      const int64_t e = t_lo + tile1 * kTile;
      t_end = e < p + 1 ? e : p + 1;
    }
  }
  const int32_t* bt = a.block_tables + static_cast<int64_t>(req < 0 ? 0 : req) * a.blocks_per_request;
  const int kwidth = a.kv_heads * DK, vwidth = a.kv_heads * DV;
  // The block-table entries of this split's range, staged once (the tile
  // loads index them; at most kMaxSplitBlocks + 1 blocks span the range).
  int32_t* s_bt = reinterpret_cast<int32_t*>(own_v + DV);  // [kFusedMaxBt]
  const int64_t bt_first = t_begin / a.block_tokens;
  if (t_begin < t_end) {
    const int nb = static_cast<int>((t_end - 1) / a.block_tokens - bt_first) + 1;
    for (int i = threadIdx.x; i < nb; i += blockDim.x) s_bt[i] = bt[bt_first + i];
  }
  // The row's K/V head (the finish kernel's values): split 0 appends it to
  // the cache; every split keeps the cache's form of it in shared memory
  // for the tile that holds the row's own token.
  if (p >= 0 && warp < 2) {
    const int32_t blk = bt[p / a.block_tokens];
    const int64_t phys = static_cast<int64_t>(blk) * a.block_tokens + p % a.block_tokens;
    if (warp == 0) {
      float x[HL::kN];
      HL::finish(kv_raw, p, a.inv_freq, a.k_norm, a.eps, lane, x);
      int off[HL::kN];
      HL::offsets(lane, off);
      if constexpr (kFp8) {
        float amax = 0.f;
#pragma unroll
        for (int j = 0; j < HL::kN; ++j) amax = fmaxf(amax, fabsf(x[j]));
        amax = warp_max(amax);
        const LatentFp8Scale sc = latent_fp8_row_scale(amax);
        uint8_t* row = reinterpret_cast<uint8_t*>(a.k_cache) + phys * kwidth + kvh * DK;
#pragma unroll
        for (int j = 0; j < HL::kN; ++j) {
          const uint8_t code = latent_fp8_encode(x[j], sc.inv);
          if (s == 0) row[off[j]] = code;
          own_k[off[j]] = latent_fp8_decode_bf16(code, sc.scale);
        }
        if (s == 0 && lane == 0) a.k_scale[phys * a.kv_heads + kvh] = sc.scale;
      } else {
        uint16_t* dst = a.k_cache + phys * kwidth + kvh * DK;
#pragma unroll
        for (int j = 0; j < HL::kN; ++j) own_k[off[j]] = float_to_bf16_bits(x[j]);
        if (s == 0) {
          if constexpr (DK == 192) {
            dst[lane] = own_k[lane];
            dst[lane + R / 2] = own_k[lane + R / 2];
            *reinterpret_cast<uint2*>(dst + R + lane * 4) = *reinterpret_cast<const uint2*>(own_k + R + lane * 4);
          } else {
#pragma unroll
            for (int j = 0; j < HL::kN; ++j) dst[off[j]] = own_k[off[j]];
          }
        }
      }
    } else {
      float v[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float x = round_bf16(kv_raw[i]);
        if (a.value_scale != 1.0f) x = round_bf16(x * a.value_scale);
        v[i] = x;
      }
      if constexpr (kFp8) {
        float amax = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) amax = fmaxf(amax, fabsf(v[j]));
        amax = warp_max(amax);
        const LatentFp8Scale sc = latent_fp8_row_scale(amax);
        uint8_t* row = reinterpret_cast<uint8_t*>(a.v_cache) + phys * vwidth + kvh * DV;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint8_t code = latent_fp8_encode(v[j], sc.inv);
          if (s == 0) row[lane * 4 + j] = code;
          own_v[lane * 4 + j] = latent_fp8_decode_bf16(code, sc.scale);
        }
        if (s == 0 && lane == 0) a.v_scale[phys * a.kv_heads + kvh] = sc.scale;
      } else {
        const uint2 packed = pack4(v);
        *reinterpret_cast<uint2*>(own_v + lane * 4) = packed;
        if (s == 0) *reinterpret_cast<uint2*>(a.v_cache + phys * vwidth + kvh * DV + lane * 4) = packed;
      }
    }
  }
  float m_run = -INFINITY, l_run = 0.0f;
  float creg[4] = {0.f, 0.f, 0.f, 0.f};
  if (t_begin < t_end) {
    // This warp's query head, finished into fp32 smem.
    {
      float x[HL::kN];
      HL::finish(q_raw, p, a.inv_freq, a.q_norm, a.eps, lane, x);
      int off[HL::kN];
      HL::offsets(lane, off);
      float* myq = qs + warp * DK;
#pragma unroll
      for (int j = 0; j < HL::kN; ++j) myq[off[j]] = x[j];
    }
    __syncthreads();  // the queries, the batch table, the row's own K/V
    if (s == 0 && a.sink != nullptr) {
      m_run = a.sink[h];
      l_run = 1.0f;
    }
    const float* myq = qs + warp * DK;
    // The tile pipeline: a tile's cache pieces are loaded into registers
    // one tile ahead (issued before the current tile's arithmetic, so the
    // cache latency hides under it), then stored to the shared tile; the
    // block's block-table entries were staged once above. A thread owns
    // up to kKPieces K pieces and kVPieces V pieces of a tile.
    TilePieces<DK, kFp8> pieces;
    int64_t t0 = t_begin;
    load_tile_pieces<DK, kFp8>(a, s_bt, bt_first, kvh, t0, t_end, pieces);
    for (; t0 < t_end; t0 += kTile) {
      const int n = t_end - t0 < kTile ? static_cast<int>(t_end - t0) : kTile;
      __syncthreads();  // the previous tile's readers are done
      store_tile_pieces<DK, kFp8>(t0, t_end, pieces, kt, vt);
      // The next tile's loads, in flight across this tile's arithmetic.
      if (t0 + kTile < t_end) load_tile_pieces<DK, kFp8>(a, s_bt, bt_first, kvh, t0 + kTile, t_end, pieces);
      __syncthreads();
      // The overlays: the batch rows of this request inside the tile — this
      // row from its shared copy, the others recomputed — one warp each
      // (block-uniform: every warp scans the same table).
      bool any = false;
      for (int i = 0; i < a.rows; ++i)
        any |= s_req[i] == req && s_pos[i] >= t0 && s_pos[i] < t0 + n;
      if (any) {
        for (int i = warp; i < a.rows; i += hpk) {
          const int64_t pi = s_pos[i];
          if (s_req[i] != req || pi < t0 || pi >= t0 + n) continue;
          const int tt = static_cast<int>(pi - t0);
          if (i == r) {
            // (K tile rows are 4-byte aligned: word stores.)
            uint16_t* kd = kt + tt * kRowStride;
            if constexpr (DK == 192) {
              kd[lane] = own_k[lane];
              kd[lane + R / 2] = own_k[lane + R / 2];
              const uint2 kw2 = *reinterpret_cast<const uint2*>(own_k + R + lane * 4);
              uint32_t* kdw = reinterpret_cast<uint32_t*>(kd + R + lane * 4);
              kdw[0] = kw2.x;
              kdw[1] = kw2.y;
            } else {
#pragma unroll
              for (int j = 0; j < HL::kN; ++j) kd[lane + 32 * j] = own_k[lane + 32 * j];
            }
            *reinterpret_cast<uint2*>(vt + tt * DV + lane * 4) = *reinterpret_cast<const uint2*>(own_v + lane * 4);
          } else {
            overlay_row<DK, R, kFp8>(a, i, pi, kvh, tt, lane, kt, vt);
          }
        }
        __syncthreads();
      }
      // Scores: lane t against tile token t (a fixed fp32 order over d).
      float score = -INFINITY;
      if (lane < n) {
        score = tile_score<DK>(kt + lane * kRowStride, myq) * a.scale;
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
      for (int tt = 0; tt < n; ++tt) {
        const float pt = pbuf[warp * kTile + tt];
        const uint2 vv = *reinterpret_cast<const uint2*>(vt + tt * DV + lane * 4);
        creg[0] = fmaf(pt, bf16_bits_to_float(static_cast<uint16_t>(vv.x & 0xFFFFu)), creg[0]);
        creg[1] = fmaf(pt, bf16_bits_to_float(static_cast<uint16_t>(vv.x >> 16)), creg[1]);
        creg[2] = fmaf(pt, bf16_bits_to_float(static_cast<uint16_t>(vv.y & 0xFFFFu)), creg[2]);
        creg[3] = fmaf(pt, bf16_bits_to_float(static_cast<uint16_t>(vv.y >> 16)), creg[3]);
      }
      m_run = m_new;
      __syncwarp();
    }
    float* crow = a.c_ws + (base_m + h) * DV + lane * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) crow[i] = creg[i];
  }
  // The partials (an empty split: the neutral m = -inf, l = 0 and no c —
  // the combine skips it).
  if (lane == 0) {
    a.m_ws[base_m + h] = m_run;
    a.l_ws[base_m + h] = l_run;
  }
  // Arrive; the last block of (row, kv head) combines.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) s_last = atomicAdd(a.counters + static_cast<int64_t>(r) * a.kv_heads + kvh, 1) == a.n_split - 1;
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  {
    // The row's (split, head) m and l staged through shared memory (one
    // coalesced pass, the K tile's space), the c rows loaded eight splits
    // at a time; the accumulation order over the splits is the combine
    // kernel's (ascending), so the result is bitwise its.
    const int64_t base_r = static_cast<int64_t>(r) * a.n_split * a.local_heads;
    const int nsl = a.n_split * a.local_heads;
    float* sm = reinterpret_cast<float*>(smem);
    float* sl = sm + nsl;
    for (int i = threadIdx.x; i < nsl; i += blockDim.x) {
      sm[i] = __ldcg(a.m_ws + base_r + i);
      sl[i] = __ldcg(a.l_ws + base_r + i);
    }
    __syncthreads();
    float M = -INFINITY;
    for (int ss = 0; ss < a.n_split; ++ss) M = fmaxf(M, sm[ss * a.local_heads + h]);
    float L = 0.f, C[4] = {0.f, 0.f, 0.f, 0.f};
    if (M > -INFINITY) {
      const float* crow = a.c_ws + (base_r + h) * DV + lane * 4;
      for (int ss0 = 0; ss0 < a.n_split; ss0 += 8) {
        float4 c[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int ss = ss0 + j;
          c[j] = make_float4(0.f, 0.f, 0.f, 0.f);
          if (ss < a.n_split && sm[ss * a.local_heads + h] > -INFINITY)
            c[j] = __ldcg(reinterpret_cast<const float4*>(crow + static_cast<int64_t>(ss) * a.local_heads * DV));
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int ss = ss0 + j;
          if (ss >= a.n_split) break;
          const float mi = sm[ss * a.local_heads + h];
          if (mi == -INFINITY) continue;  // a neutral split: its c is unwritten and would add 0
          const float w = expf(mi - M);
          L = fmaf(sl[ss * a.local_heads + h], w, L);
          C[0] = fmaf(c[j].x, w, C[0]);
          C[1] = fmaf(c[j].y, w, C[1]);
          C[2] = fmaf(c[j].z, w, C[2]);
          C[3] = fmaf(c[j].w, w, C[3]);
        }
      }
    }
    float o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = L > 0.f ? C[j] / L : 0.f;
    *reinterpret_cast<uint2*>(a.out + (static_cast<int64_t>(r) * a.local_heads + h) * DV + lane * 4) = pack4(o);
  }
  if (threadIdx.x == 0) atomicExch(a.counters + static_cast<int64_t>(r) * a.kv_heads + kvh, 0);
}

// ---- the launchers (per geometry) -------------------------------------------------
template <int DK, int R>
void qkv_finish(const float* qkv, int64_t qkv_stride, const SinkQkvLayout& layout, const SinkFinishParams& fin,
                const int32_t* req_ids, const int64_t* pos, int rows, int local_heads, int kv_heads,
                const int32_t* block_tables, int blocks_per_request, int block_tokens, uint16_t* q_out,
                int64_t q_out_stride, uint16_t* k_cache, uint16_t* v_cache, cudaStream_t stream, float* k_scale,
                float* v_scale) {
  if (rows <= 0) return;
  if (!qkv || !req_ids || !pos || !block_tables || !q_out || !k_cache || !v_cache)
    throw std::invalid_argument("sink_qkv_finish: null pointer");
  if constexpr (DK == 192) {
    if (!fin.inv_freq) throw std::invalid_argument("sink_qkv_finish: the qk-192 geometry rotates every head (inv_freq required)");
    if (fin.q_norm || fin.k_norm) throw std::invalid_argument("sink_qkv_finish: the qk-192 geometry has no head norm");
  }
  if ((k_scale == nullptr) != (v_scale == nullptr))
    throw std::invalid_argument("sink_qkv_finish: the fp8 cache's K and V scale planes come together");
  if (local_heads <= 0 || kv_heads <= 0 || local_heads % kv_heads != 0)
    throw std::invalid_argument("sink_qkv_finish: heads");
  if (layout.chunks <= 0 || layout.q_per_chunk <= 0 || layout.kv_per_chunk <= 0 ||
      layout.chunks * layout.q_per_chunk != local_heads || layout.chunks * layout.kv_per_chunk != kv_heads)
    throw std::invalid_argument("sink_qkv_finish: the chunk layout does not cover the rank's heads");
  const int64_t chunk_rows = static_cast<int64_t>(layout.q_per_chunk) * DK +
                             static_cast<int64_t>(layout.kv_per_chunk) * (DK + DV);
  if (layout.chunk_stride < chunk_rows)
    throw std::invalid_argument("sink_qkv_finish: chunk_stride must cover the chunk's rows");
  if (qkv_stride < static_cast<int64_t>(layout.chunks - 1) * layout.chunk_stride + chunk_rows)
    throw std::invalid_argument("sink_qkv_finish: qkv_stride must cover the rank's chunks");
  if (!(fin.value_scale > 0.0f)) throw std::invalid_argument("sink_qkv_finish: value_scale must be positive");
  if (!(fin.eps > 0.0f)) throw std::invalid_argument("sink_qkv_finish: eps must be positive");
  const int items = rows * (local_heads + 2 * kv_heads);
  const int blocks = (items + 7) / 8;
  if (k_scale != nullptr)
    qkv_finish_kernel<DK, R, true><<<blocks, 256, 0, stream>>>(qkv, qkv_stride, layout, fin, req_ids, pos, rows,
                                                                local_heads, kv_heads, block_tables, blocks_per_request,
                                                                block_tokens, q_out, q_out_stride, k_cache, v_cache,
                                                                k_scale, v_scale);
  else
    qkv_finish_kernel<DK, R, false><<<blocks, 256, 0, stream>>>(qkv, qkv_stride, layout, fin, req_ids, pos, rows,
                                                                 local_heads, kv_heads, block_tables, blocks_per_request,
                                                                 block_tokens, q_out, q_out_stride, k_cache, v_cache,
                                                                 nullptr, nullptr);
  DGPP_CUDA_OK(cudaGetLastError());
}

template <int DK>
void attn_partial(const uint16_t* q, int64_t q_stride, const uint16_t* k_cache, const uint16_t* v_cache,
                  const int32_t* req_ids, const int64_t* pos, int rows, int n_split, int local_heads, int kv_heads,
                  int block_tokens, const int32_t* block_tables, int blocks_per_request, int window, float scale,
                  const float* sink, float* m_ws, float* l_ws, float* c_ws, cudaStream_t stream,
                  const float* k_scale, const float* v_scale) {
  constexpr int kRowStride = Geo<DK>::kRowStride;
  if (rows <= 0) return;
  if (!q || !k_cache || !v_cache || !req_ids || !pos || !block_tables || !m_ws || !l_ws || !c_ws)
    throw std::invalid_argument("sink_attn_partial: null pointer");
  if ((k_scale == nullptr) != (v_scale == nullptr))
    throw std::invalid_argument("sink_attn_partial: the fp8 cache's K and V scale planes come together");
  if (local_heads <= 0 || kv_heads <= 0 || local_heads % kv_heads != 0)
    throw std::invalid_argument("sink_attn_partial: heads");
  if (n_split <= 0) throw std::invalid_argument("sink_attn_partial: n_split");
  if (block_tokens <= 0 || block_tokens % kTile != 0)
    throw std::invalid_argument("sink_attn_partial: block_tokens must be a multiple of the 32-token tile");
  if (window < 0) throw std::invalid_argument("sink_attn_partial: window");
  const int hpk = local_heads / kv_heads;
  if (hpk > 32) throw std::invalid_argument("sink_attn_partial: at most 32 query heads per kv head");
  const size_t smem = static_cast<size_t>(kTile) * kRowStride * 2 + static_cast<size_t>(kTile) * DV * 2 +
                      static_cast<size_t>(hpk) * DK * 4 + static_cast<size_t>(hpk) * kTile * 4;
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(n_split), static_cast<unsigned>(kv_heads));
  if (k_scale != nullptr)
    attn_partial_kernel<DK, true><<<grid, hpk * 32, smem, stream>>>(
        q, q_stride, k_cache, v_cache, req_ids, pos, n_split, local_heads, kv_heads, block_tokens, block_tables,
        blocks_per_request, window, scale, sink, m_ws, l_ws, c_ws, k_scale, v_scale);
  else
    attn_partial_kernel<DK, false><<<grid, hpk * 32, smem, stream>>>(
        q, q_stride, k_cache, v_cache, req_ids, pos, n_split, local_heads, kv_heads, block_tokens, block_tables,
        blocks_per_request, window, scale, sink, m_ws, l_ws, c_ws, nullptr, nullptr);
  DGPP_CUDA_OK(cudaGetLastError());
}

template <int DK>
size_t fused_smem_bytes(int local_heads, int kv_heads) {
  constexpr int kRowStride = Geo<DK>::kRowStride;
  const int hpk = local_heads / kv_heads;
  return static_cast<size_t>(kTile) * kRowStride * 2 + static_cast<size_t>(kTile) * DV * 2 +
         static_cast<size_t>(hpk) * DK * 4 + static_cast<size_t>(hpk) * kTile * 4 +
         static_cast<size_t>(kFusedMaxRows) * (4 + 8) + static_cast<size_t>(DK + DV) * 2 +
         static_cast<size_t>(kFusedMaxBt) * 4;
}

template <int DK, int R>
void attn_fused(const SinkAttnFusedArgs& a, cudaStream_t stream) {
  constexpr int kRowStride = Geo<DK>::kRowStride;
  if (a.rows <= 0) return;
  if (!a.qkv || !a.req_ids || !a.pos || !a.block_tables || !a.k_cache || !a.v_cache || !a.m_ws ||
      !a.l_ws || !a.c_ws || !a.counters || !a.out)
    throw std::invalid_argument("sink_attn_fused: null pointer");
  if constexpr (DK == 192) {
    if (!a.inv_freq) throw std::invalid_argument("sink_attn_fused: the qk-192 geometry rotates every head (inv_freq required)");
    if (a.q_norm || a.k_norm) throw std::invalid_argument("sink_attn_fused: the qk-192 geometry has no head norm");
  }
  if ((a.k_scale == nullptr) != (a.v_scale == nullptr))
    throw std::invalid_argument("sink_attn_fused: the fp8 cache's K and V scale planes come together");
  if (a.rows > kFusedMaxRows) throw std::invalid_argument("sink_attn_fused: rows exceed the batch table");
  if (a.local_heads <= 0 || a.kv_heads <= 0 || a.local_heads % a.kv_heads != 0)
    throw std::invalid_argument("sink_attn_fused: heads");
  const int hpk = a.local_heads / a.kv_heads;
  if (hpk * 32 < kFusedMinThreads || hpk * 32 > kFusedMaxThreads)
    throw std::invalid_argument("sink_attn_fused: four to 16 query heads per kv head (the tile pipeline's register budget)");
  if (a.n_split <= 0) throw std::invalid_argument("sink_attn_fused: n_split");
  if (static_cast<size_t>(a.n_split) * a.local_heads * 2 * sizeof(float) >
      static_cast<size_t>(kTile) * (kRowStride + DV) * 2)
    throw std::invalid_argument("sink_attn_fused: n_split x local_heads exceeds the combine's staging (the K/V tiles)");
  if (a.block_tokens <= 0 || a.block_tokens % kTile != 0)
    throw std::invalid_argument("sink_attn_fused: block_tokens must be a multiple of the 32-token tile");
  if (a.window < 0) throw std::invalid_argument("sink_attn_fused: window");
  {
    // A split's range spans at most ceil(tiles / n_split) tiles of the
    // request's context (the window's when sliding): its block-table
    // entries must fit the staged span.
    const int64_t ctx = a.window > 0 ? a.window : static_cast<int64_t>(a.blocks_per_request) * a.block_tokens;
    const int64_t tiles = (ctx + kTile - 1) / kTile;
    const int64_t chunk = (tiles + a.n_split - 1) / a.n_split;
    const int64_t span_blocks = (chunk * kTile + a.block_tokens - 1) / a.block_tokens + 1;
    if (span_blocks > kFusedMaxBt)
      throw std::invalid_argument("sink_attn_fused: a split's range spans more block-table entries than staged");
  }
  if (a.layout.chunks <= 0 || a.layout.q_per_chunk <= 0 || a.layout.kv_per_chunk <= 0 ||
      a.layout.chunks * a.layout.q_per_chunk != a.local_heads || a.layout.chunks * a.layout.kv_per_chunk != a.kv_heads)
    throw std::invalid_argument("sink_attn_fused: the chunk layout does not cover the rank's heads");
  if (!(a.value_scale > 0.0f)) throw std::invalid_argument("sink_attn_fused: value_scale must be positive");
  if (!(a.eps > 0.0f)) throw std::invalid_argument("sink_attn_fused: eps must be positive");
  const size_t smem = fused_smem_bytes<DK>(a.local_heads, a.kv_heads);
  const dim3 grid(static_cast<unsigned>(a.rows), static_cast<unsigned>(a.n_split), static_cast<unsigned>(a.kv_heads));
  if (a.k_scale != nullptr)
    attn_fused_kernel<DK, R, true><<<grid, hpk * 32, smem, stream>>>(a);
  else
    attn_fused_kernel<DK, R, false><<<grid, hpk * 32, smem, stream>>>(a);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace sink_attn
}  // namespace dgpp
