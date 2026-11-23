#include "kernels/inkling2_attn.hpp"

#include <cmath>
#include <stdexcept>

#include <cuda_bf16.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/inkling2.hpp"
#include "kernels/pdl.cuh"

namespace dgpp {
namespace {

constexpr int D = kInk2HeadDim;
constexpr int kTile = kInk2AttnTile;
// K tile rows are padded to 16-byte aligned rows of 68 words, conflict-free
// for the score loop's 128-bit reads.
constexpr int kRowStride = D + 8;
constexpr float kClipEps = 1e-6f;
static_assert(D == 4 * 32, "the lane mapping: four dims per lane");
static_assert((kRowStride * 2) % 16 == 0, "K tile rows must be 16-byte aligned");

__device__ __forceinline__ float round_bf16(float v) { return bf16_bits_to_float(float_to_bf16_bits(v)); }

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

__device__ __forceinline__ float log_tau(int64_t p, float alpha, float n_floor) {
  return 1.0f + alpha * logf(fmaxf(static_cast<float>(p + 1) / n_floor, 1.0f));
}

// ---- the query finish ---------------------------------------------------------------
struct QueryDev {
  const uint16_t* q_side;
  int64_t q_side_stride;
  const float* q_scale;
  const float* r_scale;
  const float* rel_proj;
  const int64_t* pos;
  int rows, local_heads, d_rel, extent;
  float qk_clip, rel_clip;
  bool log_scaling;
  float log_alpha, log_n_floor;
  uint16_t* q_out;
  uint16_t* bias;
};

// One warp per (row, head): lane l owns dims {l, l + 32, l + 64, l + 96}.
__global__ void __launch_bounds__(256) q_finish_kernel(QueryDev a) {
  dgpp_pdl_wait();
  const int item = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) / 32) + static_cast<int>(threadIdx.x) / 32;
  if (item >= a.rows * a.local_heads) return;
  const int row = item / a.local_heads, h = item - row * a.local_heads;
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int64_t p = a.pos[row];
  uint16_t* dst = a.q_out + static_cast<int64_t>(row) * a.local_heads * D + h * D;
  if (p < 0) {
#pragma unroll
    for (int i = 0; i < 4; ++i) dst[lane + 32 * i] = 0;
    return;
  }
  const uint16_t* src = a.q_side + static_cast<int64_t>(row) * a.q_side_stride + h * D;
  const float* scale = a.q_scale + h * D;
  float x[4];
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane + 32 * i;
    x[i] = bf16_bits_to_float(src[d]) * scale[d];
    ss = fmaf(x[i], x[i], ss);
  }
  const float f = fminf(a.qk_clip / (sqrtf(warp_sum(ss)) + kClipEps), 1.0f);
  const float tau = a.log_scaling ? log_tau(p, a.log_alpha, a.log_n_floor) : 1.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float q = round_bf16(x[i] * f);
    if (a.log_scaling) q = round_bf16(q * tau);
    dst[lane + 32 * i] = float_to_bf16_bits(q);
  }
}

// One block per (row, head): the clipped relative state, then its bias row.
__global__ void __launch_bounds__(256) rel_bias_kernel(QueryDev a) {
  dgpp_pdl_wait();
  __shared__ float r[kInk2MaxRelDim];
  const int row = static_cast<int>(blockIdx.x), h = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t p = a.pos[row];
  if (p < 0) return;
  if (tid < 32) {
    float x = 0.0f;
    if (tid < a.d_rel)
      x = bf16_bits_to_float(a.q_side[static_cast<int64_t>(row) * a.q_side_stride +
                                      static_cast<int64_t>(a.local_heads) * D + h * a.d_rel + tid]) *
          a.r_scale[h * a.d_rel + tid];
    const float f = fminf(a.rel_clip / (sqrtf(warp_sum(x * x)) + kClipEps), 1.0f);
    float v = round_bf16(x * f);
    if (a.log_scaling) v = round_bf16(v * log_tau(p, a.log_alpha, a.log_n_floor));
    r[tid] = v;
  }
  __syncthreads();
  uint16_t* dst = a.bias + (static_cast<int64_t>(row) * a.local_heads + h) * a.extent;
  for (int d = tid; d < a.extent; d += static_cast<int>(blockDim.x)) {
    float acc = 0.0f;
    for (int j = 0; j < a.d_rel; ++j) acc = fmaf(r[j], a.rel_proj[static_cast<int64_t>(j) * a.extent + d], acc);
    dst[d] = float_to_bf16_bits(acc);
  }
}

// ---- the K / V finish -----------------------------------------------------------------
struct KvDev {
  const uint16_t* in[2];
  int64_t kv_stride;
  const float* taps[2];
  bool dynamic;
  float conv_scale;
  uint16_t* ring;
  int64_t ring_request_stride, ring_slot_stride;
  int64_t ring_offset[2];
  bool store_inline;
  const int32_t* req_ids;
  const int64_t* pos;
  int rows, kv_heads;
  float qk_clip, eps;
  const int32_t* block_tables;
  int blocks_per_request, block_tokens;
  uint16_t* cache[2];
};

// One warp per (row, kind, kv head): kind 0 is K, 1 is V.
__global__ void __launch_bounds__(256) kv_finish_kernel(KvDev a) {
  dgpp_pdl_wait();
  const int per_row = 2 * a.kv_heads;
  const int item = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) / 32) + static_cast<int>(threadIdx.x) / 32;
  if (item >= a.rows * per_row) return;
  const int row = item / per_row;
  const int rem = item - row * per_row;
  const int kind = rem / a.kv_heads, h = rem - kind * a.kv_heads;
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int64_t p = a.pos[row];
  if (p < 0) return;
  const int32_t req = a.req_ids[row];
  const uint16_t* in = a.in[kind];
  const uint16_t* x_row = in + static_cast<int64_t>(row) * a.kv_stride;
  float dyn[kInk2ConvTaps] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (a.dynamic) {
    const uint16_t* d = x_row + static_cast<int64_t>(a.kv_heads) * D + h * kInk2ConvTaps;
#pragma unroll
    for (int lag = 0; lag < kInk2ConvTaps; ++lag) dyn[lag] = bf16_bits_to_float(d[lag]);
  }
  uint16_t* ring_req = a.ring + static_cast<int64_t>(req) * a.ring_request_stride + a.ring_offset[kind];
  float y[4];
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int c = h * D + lane + 32 * i;
    const float* tap = a.taps[kind] + static_cast<int64_t>(c) * kInk2ConvTaps;
    const uint16_t x_bits = x_row[c];
    const float x0 = bf16_bits_to_float(x_bits);
    float acc = 0.0f;
#pragma unroll
    for (int lag = kInk2ConvTaps - 1; lag >= 1; --lag) {
      const int64_t q = p - lag;
      float xl = 0.0f;
      if (q >= 0) {
        const int rr = row - lag;
        if (rr >= 0 && a.req_ids[rr] == req && a.pos[rr] == q)
          xl = bf16_bits_to_float(in[static_cast<int64_t>(rr) * a.kv_stride + c]);
        else
          xl = bf16_bits_to_float(ring_req[(q % kInk2RingSlots) * a.ring_slot_stride + c]);
      }
      acc = fmaf(tap[lag] + dyn[lag], xl, acc);
    }
    acc = fmaf(tap[0] + dyn[0], x0, acc);
    y[i] = round_bf16((acc + x0) * a.conv_scale);
    ss = fmaf(y[i], y[i], ss);
    if (a.store_inline) ring_req[(p % kInk2RingSlots) * a.ring_slot_stride + c] = x_bits;
  }
  ss = warp_sum(ss);
  const float f = kind == 0 ? fminf(a.qk_clip / (sqrtf(ss) + kClipEps), 1.0f)
                            : 1.0f / sqrtf(ss / static_cast<float>(D) + a.eps);
  const int32_t blk = a.block_tables[static_cast<int64_t>(req) * a.blocks_per_request + p / a.block_tokens];
  const int64_t phys = static_cast<int64_t>(blk) * a.block_tokens + p % a.block_tokens;
  uint16_t* dst = a.cache[kind] + phys * (static_cast<int64_t>(a.kv_heads) * D) + h * D;
#pragma unroll
  for (int i = 0; i < 4; ++i) dst[lane + 32 * i] = float_to_bf16_bits(y[i] * f);
}

// ---- the split-KV attention ------------------------------------------------------------
// The dot of a staged K row (BF16, 16-byte aligned) with the head's FP32
// query: a fixed FP32 FMA order over the dims, eight per vector read.
__device__ __forceinline__ float tile_score(const uint16_t* __restrict__ krow_bf16, const float* __restrict__ myq) {
  const uint4* krow = reinterpret_cast<const uint4*>(krow_bf16);
  const float4* q4 = reinterpret_cast<const float4*>(myq);
  float acc = 0.f;
#pragma unroll 4
  for (int v = 0; v < D / 8; ++v) {
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

struct AttnDev {
  const uint16_t* q;
  int64_t q_stride;
  const uint16_t* bias;
  int extent;
  const uint16_t* sink;
  int64_t sink_stride;
  const uint16_t* k_cache;
  const uint16_t* v_cache;
  const int32_t* req_ids;
  const int64_t* pos;
  int n_split, local_heads, kv_heads, block_tokens;
  const int32_t* block_tables;
  int blocks_per_request, window;
  float scale;
  float* m_ws;
  float* l_ws;
  float* c_ws;
};

// Block: one warp per query head of the kv head. Lane t owns tile token t
// for the scores and dims [4 l, 4 l + 4) of the value for the accumulation.
__global__ void attn_partial_kernel(AttnDev a) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint16_t smem[];
  const int hpk = a.local_heads / a.kv_heads;
  const int r = static_cast<int>(blockIdx.x);
  const int s = static_cast<int>(blockIdx.y);
  const int kvh = static_cast<int>(blockIdx.z);
  const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
  const int h = kvh * hpk + warp;
  const int64_t base_m = (static_cast<int64_t>(r) * a.n_split + s) * a.local_heads;
  uint16_t* kt = smem;                                   // [kTile][kRowStride]
  uint16_t* vt = kt + kTile * kRowStride;                // [kTile][D]
  float* qs = reinterpret_cast<float*>(vt + kTile * D);  // [hpk][D]
  float* pbuf = qs + hpk * D;                            // [hpk][kTile]

  const int64_t p = a.pos[r];
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
  // The sink enters split 0 whether or not the split has keys.
  const bool has_sink = p >= 0 && s == 0;
  if (t_begin >= t_end) {
    for (int i = static_cast<int>(threadIdx.x); i < hpk * D; i += static_cast<int>(blockDim.x))
      a.c_ws[(base_m + kvh * hpk + i / D) * D + i % D] = 0.0f;
    if (static_cast<int>(threadIdx.x) < hpk) {
      const int hh = kvh * hpk + static_cast<int>(threadIdx.x);
      a.m_ws[base_m + hh] = has_sink ? bf16_bits_to_float(a.sink[static_cast<int64_t>(r) * a.sink_stride + hh]) : -INFINITY;
      a.l_ws[base_m + hh] = has_sink ? 1.0f : 0.0f;
    }
    return;
  }
  {
    const uint16_t* qrow = a.q + static_cast<int64_t>(r) * a.q_stride + static_cast<int64_t>(h) * D;
    for (int d = lane; d < D; d += 32) qs[warp * D + d] = bf16_bits_to_float(qrow[d]);
  }
  __syncthreads();
  const int32_t* bt = a.block_tables + static_cast<int64_t>(a.req_ids[r]) * a.blocks_per_request;
  const int kwidth = a.kv_heads * D, vwidth = a.kv_heads * D;
  const uint16_t* brow = a.bias + (static_cast<int64_t>(r) * a.local_heads + h) * a.extent;
  float m_run = -INFINITY, l_run = 0.0f;
  if (has_sink) {
    m_run = bf16_bits_to_float(a.sink[static_cast<int64_t>(r) * a.sink_stride + h]);
    l_run = 1.0f;
  }
  float creg[4] = {0.f, 0.f, 0.f, 0.f};
  const float* myq = qs + warp * D;
  for (int64_t t0 = t_begin; t0 < t_end; t0 += kTile) {
    const int n = t_end - t0 < kTile ? static_cast<int>(t_end - t0) : kTile;
    __syncthreads();  // the previous tile's readers are done
    for (int idx = static_cast<int>(threadIdx.x); idx < n * (D / 8); idx += static_cast<int>(blockDim.x)) {
      const int tt = idx / (D / 8);
      const int c8 = idx - tt * (D / 8);
      const int64_t tok = t0 + tt;
      const int32_t blk = bt[tok / a.block_tokens];
      const int64_t phys = static_cast<int64_t>(blk) * a.block_tokens + tok % a.block_tokens;
      const uint4 val = *reinterpret_cast<const uint4*>(a.k_cache + phys * kwidth + kvh * D + c8 * 8);
      uint32_t* dst = reinterpret_cast<uint32_t*>(kt + tt * kRowStride) + c8 * 4;
      dst[0] = val.x;
      dst[1] = val.y;
      dst[2] = val.z;
      dst[3] = val.w;
      *reinterpret_cast<uint4*>(vt + tt * D + c8 * 8) =
          *reinterpret_cast<const uint4*>(a.v_cache + phys * vwidth + kvh * D + c8 * 8);
    }
    __syncthreads();
    float score = -INFINITY;
    if (lane < n) {
      score = tile_score(kt + lane * kRowStride, myq) * a.scale;
      const int64_t dist = p - (t0 + lane);
      if (dist < a.extent) score += bf16_bits_to_float(brow[dist]);
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
    a.m_ws[base_m + h] = m_run;
    a.l_ws[base_m + h] = l_run;
  }
  float* crow = a.c_ws + (base_m + h) * D + lane * 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) crow[i] = creg[i];
}

__global__ void __launch_bounds__(D) attn_combine_kernel(const float* __restrict__ m_ws, const float* __restrict__ l_ws,
                                                         const float* __restrict__ c_ws,
                                                         const uint16_t* __restrict__ gate, int64_t gate_stride,
                                                         int n_split, int local_heads, uint16_t* __restrict__ out) {
  dgpp_pdl_wait();
  const int r = static_cast<int>(blockIdx.x);
  const int h = static_cast<int>(blockIdx.y);
  const int d = static_cast<int>(threadIdx.x);
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
  const float g = bf16_bits_to_float(gate[static_cast<int64_t>(r) * gate_stride + h * D + d]);
  const float value = L > 0.f ? (C / L) * (1.0f / (1.0f + expf(-g))) : 0.f;
  out[(static_cast<int64_t>(r) * local_heads + h) * D + d] = float_to_bf16_bits(value);
}

}  // namespace

void ink2_query_finish(const Ink2QueryArgs& args, cudaStream_t stream) {
  if (args.rows <= 0) return;
  if (!args.q_side || !args.q_scale || !args.r_scale || !args.rel_proj || !args.pos || !args.q_out || !args.bias)
    throw std::invalid_argument("ink2_query_finish: null buffer");
  if (args.local_heads <= 0 || args.extent <= 0) throw std::invalid_argument("ink2_query_finish: geometry");
  if (args.d_rel <= 0 || args.d_rel > kInk2MaxRelDim)
    throw std::invalid_argument("ink2_query_finish: d_rel must be in [1, 32]");
  if (!(args.qk_clip > 0.0f) || !(args.rel_clip > 0.0f)) throw std::invalid_argument("ink2_query_finish: clips");
  if (args.log_scaling && !(args.log_n_floor > 0.0f)) throw std::invalid_argument("ink2_query_finish: n_floor");
  const QueryDev d{args.q_side, args.q_side_stride, args.q_scale, args.r_scale, args.rel_proj, args.pos, args.rows,
                   args.local_heads, args.d_rel, args.extent, args.qk_clip, args.rel_clip, args.log_scaling,
                   args.log_alpha, args.log_n_floor, args.q_out, args.bias};
  const int items = args.rows * args.local_heads;
  q_finish_kernel<<<(items + 7) / 8, 256, 0, stream>>>(d);
  DGPP_CUDA_OK(cudaGetLastError());
  rel_bias_kernel<<<dim3(static_cast<unsigned>(args.rows), static_cast<unsigned>(args.local_heads)), 256, 0, stream>>>(d);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_kv_finish(const Ink2KvArgs& args, cudaStream_t stream) {
  if (args.rows <= 0) return;
  if (!args.k_in || !args.v_in || !args.k_taps || !args.v_taps || !args.ring || !args.req_ids || !args.pos ||
      !args.block_tables || !args.k_cache || !args.v_cache)
    throw std::invalid_argument("ink2_kv_finish: null buffer");
  if (args.kv_heads <= 0 || args.block_tokens <= 0 || args.blocks_per_request <= 0)
    throw std::invalid_argument("ink2_kv_finish: geometry");
  if (args.kv_stride < static_cast<int64_t>(args.kv_heads) * (D + (args.dynamic ? kInk2ConvTaps : 0)))
    throw std::invalid_argument("ink2_kv_finish: the projection rows are narrower than the heads");
  if (!(args.conv_scale > 0.0f) || !(args.qk_clip > 0.0f) || !(args.eps > 0.0f))
    throw std::invalid_argument("ink2_kv_finish: scalars");
  const KvDev d{{args.k_in, args.v_in}, args.kv_stride, {args.k_taps, args.v_taps}, args.dynamic, args.conv_scale,
                args.ring, args.ring_request_stride, args.ring_slot_stride, {args.ring_offset_k, args.ring_offset_v},
                args.store_inline, args.req_ids, args.pos, args.rows, args.kv_heads, args.qk_clip, args.eps,
                args.block_tables, args.blocks_per_request, args.block_tokens, {args.k_cache, args.v_cache}};
  const int items = args.rows * args.kv_heads * 2;
  kv_finish_kernel<<<(items + 7) / 8, 256, 0, stream>>>(d);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_attn_partial(const Ink2AttnArgs& args, cudaStream_t stream) {
  if (args.rows <= 0) return;
  if (!args.q || !args.bias || !args.sink || !args.k_cache || !args.v_cache || !args.req_ids || !args.pos ||
      !args.block_tables || !args.m_ws || !args.l_ws || !args.c_ws)
    throw std::invalid_argument("ink2_attn_partial: null buffer");
  if (args.local_heads <= 0 || args.kv_heads <= 0 || args.local_heads % args.kv_heads != 0)
    throw std::invalid_argument("ink2_attn_partial: heads");
  if (args.n_split <= 0) throw std::invalid_argument("ink2_attn_partial: n_split");
  if (args.block_tokens <= 0 || args.block_tokens % kTile != 0)
    throw std::invalid_argument("ink2_attn_partial: block_tokens must be a multiple of the 32-token tile");
  if (args.window < 0 || args.extent <= 0) throw std::invalid_argument("ink2_attn_partial: window or extent");
  if ((args.q_stride * 2) % 16 != 0) throw std::invalid_argument("ink2_attn_partial: q rows must be 16-byte aligned");
  const int hpk = args.local_heads / args.kv_heads;
  if (hpk > 32) throw std::invalid_argument("ink2_attn_partial: at most 32 query heads per kv head");
  const size_t smem = static_cast<size_t>(kTile) * kRowStride * 2 + static_cast<size_t>(kTile) * D * 2 +
                      static_cast<size_t>(hpk) * D * 4 + static_cast<size_t>(hpk) * kTile * 4;
  const AttnDev d{args.q, args.q_stride, args.bias, args.extent, args.sink, args.sink_stride, args.k_cache,
                  args.v_cache, args.req_ids, args.pos, args.n_split, args.local_heads, args.kv_heads,
                  args.block_tokens, args.block_tables, args.blocks_per_request, args.window, args.scale, args.m_ws,
                  args.l_ws, args.c_ws};
  const dim3 grid(static_cast<unsigned>(args.rows), static_cast<unsigned>(args.n_split),
                  static_cast<unsigned>(args.kv_heads));
  attn_partial_kernel<<<grid, hpk * 32, smem, stream>>>(d);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_attn_combine(const float* m_ws, const float* l_ws, const float* c_ws, const uint16_t* gate,
                       int64_t gate_stride, int rows, int n_split, int local_heads, uint16_t* out,
                       cudaStream_t stream) {
  if (rows <= 0) return;
  if (!m_ws || !l_ws || !c_ws || !gate || !out) throw std::invalid_argument("ink2_attn_combine: null buffer");
  if (n_split <= 0 || local_heads <= 0) throw std::invalid_argument("ink2_attn_combine: geometry");
  attn_combine_kernel<<<dim3(static_cast<unsigned>(rows), static_cast<unsigned>(local_heads)), D, 0, stream>>>(
      m_ws, l_ws, c_ws, gate, gate_stride, n_split, local_heads, out);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
