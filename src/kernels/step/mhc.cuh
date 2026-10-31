#pragma once
// The hyper-connection site as step consumer units (MOONSHOT-1.md section 2.9,
// the fused chain). The arithmetic is the production kernels' op for op
// (kernels/glm_mhc.cu, the decode form: the per-coefficient dots over the
// flattened streams' norm, the fused finish with the one-rounding norm and the
// single-pass pre, the deferred comb's Sinkhorn, the fp32 stream update), so
// every output is bitwise theirs; step_consumer_test pins it.
//
//   dots    one (coefficient, token) a block: logits[t][c] = sum_d flat[t][d] * fn[c][d]
//           with flat the RMS-normalized flattened streams [4 H]
//   finish  one token a block, after its 24 dots: pre (this site's own, the
//           collapse taking the previous site's pre_in), post, comb (softmax rows
//           then Sinkhorn), the collapse sum_j pre[j] * streams[j] and its norm into
//           the sublayer's input row
//   update  a chunk of (token, d): streams_out[i] = bf16(post[i] * h + sum_j comb[j][i] * streams_in[j])
#include <cstdint>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "common/dtypes.hpp"
#include "step/device_plan.hpp"

namespace dgpp::step {

constexpr int kMhcN = 4;
constexpr int kMhcCoeffs = (2 + kMhcN) * kMhcN;  // 24: pre[4] | post[4] | comb[16]
static_assert(kMhcCoeffs % kMhcUnitCoeffs == 0, "a site's coefficients split into whole units");
constexpr int kMhcThreads = 256;

__device__ __forceinline__ float mhc_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float mhc_warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
  return v;
}
// Block-wide sum over kMhcThreads (8 warps), every thread the total: the warp
// xor trees then the warps' partials added in warp order (glm_mhc.cu block_sum).
__device__ __forceinline__ float mhc_block_sum(float v, float* scratch /* [8] */) {
  v = mhc_warp_sum(v);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  __syncthreads();
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  float total = scratch[0];
#pragma unroll
  for (int w = 1; w < kMhcThreads / 32; ++w) total += scratch[w];
  return total;
}
__device__ __forceinline__ void mhc_unpack8(uint4 q, float (&f)[8]) {
  const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    f[2 * j] = bf16_bits_to_float(static_cast<uint16_t>(w[j] & 0xFFFFu));
    f[2 * j + 1] = bf16_bits_to_float(static_cast<uint16_t>(w[j] >> 16));
  }
}

// N coefficients' logits for one token in one pass over its streams: the
// token's streams [4][hidden] bf16 at `x`, the coefficient rows [N][4 hidden]
// bf16 at `fn_rows`. Each logit's fma chain (thread t over v = t, t + 256,
// ... ascending) and its reduction are mhc_dot's, so the N logits equal N
// calls of it bitwise, for the streams read once. The thread's vectors of
// the streams stay in registers between the norm and the dots, and every
// load of a pass is issued before its first use: at one block an SM the
// unit is latency-bound, so bytes in flight set its time (2026-10-31: 12.6
// µs a unit at two vectors in flight, for 160 KB).
template <int N>
__device__ inline void mhc_dot_n(const uint16_t* __restrict__ x, const uint16_t* __restrict__ fn_rows, int hidden, float norm_eps,
                                 float* scratch, float (&out)[N]) {
  constexpr int kInFlight = 10;  // vectors a thread holds: 4 x 5120 / 8 / 256 exactly for DeepSeek-V4.1
  const int K = kMhcN * hidden, vecs = K / 8;
  const uint4* xv = reinterpret_cast<const uint4*>(x);
  float ssq = 0.f;
  float dot[N];
#pragma unroll
  for (int c = 0; c < N; ++c) dot[c] = 0.f;
  if (vecs <= kInFlight * kMhcThreads) {
    // Every vector of the thread in registers: the norm, then the dots from them.
    uint4 q[kInFlight];
#pragma unroll
    for (int i = 0; i < kInFlight; ++i) {
      const int v = i * kMhcThreads + static_cast<int>(threadIdx.x);
      q[i] = v < vecs ? xv[v] : make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int i = 0; i < kInFlight; ++i) {
      float f[8];
      mhc_unpack8(q[i], f);
#pragma unroll
      for (int j = 0; j < 8; ++j) ssq = __fmaf_rn(f[j], f[j], ssq);
    }
    const float r = rsqrtf(mhc_block_sum(ssq, scratch) / static_cast<float>(K) + norm_eps);
#pragma unroll
    for (int c = 0; c < N; ++c) {
      const uint4* wv = reinterpret_cast<const uint4*>(fn_rows + static_cast<size_t>(c) * K);
      uint4 g[kInFlight];
#pragma unroll
      for (int i = 0; i < kInFlight; ++i) {
        const int v = i * kMhcThreads + static_cast<int>(threadIdx.x);
        g[i] = v < vecs ? wv[v] : make_uint4(0u, 0u, 0u, 0u);
      }
#pragma unroll
      for (int i = 0; i < kInFlight; ++i) {
        float f[8], w[8];
        mhc_unpack8(q[i], f);
        mhc_unpack8(g[i], w);
#pragma unroll
        for (int j = 0; j < 8; ++j) dot[c] = __fmaf_rn(f[j] * r, w[j], dot[c]);  // zero past vecs: no change
      }
    }
  } else {
    // A hidden wider than the registers hold: two passes, the streams read again.
#pragma unroll 4
    for (int v = threadIdx.x; v < vecs; v += kMhcThreads) {
      float f[8];
      mhc_unpack8(xv[v], f);
#pragma unroll
      for (int j = 0; j < 8; ++j) ssq = __fmaf_rn(f[j], f[j], ssq);
    }
    const float r = rsqrtf(mhc_block_sum(ssq, scratch) / static_cast<float>(K) + norm_eps);
#pragma unroll 2
    for (int v = threadIdx.x; v < vecs; v += kMhcThreads) {
      float f[8];
      mhc_unpack8(xv[v], f);
#pragma unroll
      for (int c = 0; c < N; ++c) {
        float w[8];
        mhc_unpack8(reinterpret_cast<const uint4*>(fn_rows + static_cast<size_t>(c) * K)[v], w);
#pragma unroll
        for (int j = 0; j < 8; ++j) dot[c] = __fmaf_rn(f[j] * r, w[j], dot[c]);
      }
    }
  }
#pragma unroll
  for (int c = 0; c < N; ++c) {
    __syncthreads();  // the scratch is reused by every reduction
    out[c] = mhc_block_sum(dot[c], scratch);
  }
}

// One coefficient's logit for one token: the token's streams [4][hidden] bf16
// at `x`, the coefficient row [4 hidden] bf16 at `fn_row`. Thread t walks the
// 16-byte vectors v = t, t + 256, ... in order (the production's chain).
__device__ inline float mhc_dot(const uint16_t* __restrict__ x, const uint16_t* __restrict__ fn_row, int hidden, float norm_eps,
                                float* scratch) {
  const int K = kMhcN * hidden, vecs = K / 8;
  const uint4* xv = reinterpret_cast<const uint4*>(x);
  const uint4* wv = reinterpret_cast<const uint4*>(fn_row);
  float ssq = 0.f;
#pragma unroll 4
  for (int v = threadIdx.x; v < vecs; v += kMhcThreads) {
    float f[8];
    mhc_unpack8(xv[v], f);
#pragma unroll
    for (int j = 0; j < 8; ++j) ssq = __fmaf_rn(f[j], f[j], ssq);
  }
  const float r = rsqrtf(mhc_block_sum(ssq, scratch) / static_cast<float>(K) + norm_eps);
  float dot = 0.f;
#pragma unroll 4
  for (int v = threadIdx.x; v < vecs; v += kMhcThreads) {
    float f[8], g[8];
    mhc_unpack8(xv[v], f);
    mhc_unpack8(wv[v], g);
#pragma unroll
    for (int j = 0; j < 8; ++j) dot = __fmaf_rn(f[j] * r, g[j], dot);
  }
  return mhc_block_sum(dot, scratch);
}

// comb's seed row: softmax over row `row` of the comb logits after the affine
// transform, plus eps (glm_mhc.cu mhc_comb_seed_row).
__device__ __forceinline__ void mhc_comb_seed_row(const float* __restrict__ lg, const float* __restrict__ base, const float* __restrict__ scale,
                                                  float hc_eps, int row, float* comb_seed) {
  float m = -CUDART_INF_F;
#pragma unroll
  for (int col = 0; col < kMhcN; ++col) {
    const float v = lg[2 * kMhcN + row * kMhcN + col] * scale[2] + base[2 * kMhcN + row * kMhcN + col];
    m = fmaxf(m, v);
  }
  float c[kMhcN];
  float denom = 0.f;
#pragma unroll
  for (int col = 0; col < kMhcN; ++col) {
    const float v = lg[2 * kMhcN + row * kMhcN + col] * scale[2] + base[2 * kMhcN + row * kMhcN + col];
    c[col] = expf(v - m);
    denom += c[col];
  }
  const float inv = 1.0f / denom;
#pragma unroll
  for (int col = 0; col < kMhcN; ++col) c[col] = c[col] * inv;
#pragma unroll
  for (int col = 0; col < kMhcN; ++col) comb_seed[row * kMhcN + col] = c[col] + hc_eps;
}

// Sinkhorn on lanes 0..15 of one warp (lane = row * 4 + col): one column pass,
// then (iters - 1) row and column passes, the 4-way sums as xor butterflies
// (glm_mhc.cu mhc_sinkhorn_warp). Writes comb_f32[16].
__device__ __forceinline__ void mhc_sinkhorn_warp(const float* comb_seed, float hc_eps, int sinkhorn_iters, float* __restrict__ comb_f32, int lane) {
  const bool live = lane < kMhcN * kMhcN;
  float c = live ? comb_seed[lane] : 0.f;
  const auto col_sum = [&](float v) {
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 4);
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 8);
    return v;
  };
  const auto row_sum = [&](float v) {
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 1);
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 2);
    return v;
  };
  c = c / (col_sum(c) + hc_eps);
  for (int it = 1; it < sinkhorn_iters; ++it) {
    c = c / (row_sum(c) + hc_eps);
    c = c / (col_sum(c) + hc_eps);
  }
  if (live) comb_f32[lane] = c;
}

struct MhcSite {
  const float* base;    // [24]
  const float* scale;   // [3]
  const uint16_t* ln;   // [hidden] bf16, the sublayer's norm weight
  float hc_eps;
  int sinkhorn_iters;
  float ln_eps;
};

// The finish for one token, a block of kMhcThreads: the 24 logits at `lg`, the
// token's streams at `x` [4][hidden]; writes the token's pre_out[4], post[4],
// comb[16] (fp32) and its normed row [hidden] bf16 with the one-rounding norm.
// `smem`: kMhcN + kMhcN * kMhcN + 8 floats of scratch.
__device__ inline void mhc_finish(const uint16_t* __restrict__ x, const float* __restrict__ lg, const MhcSite& site, int hidden,
                                  const float* __restrict__ pre_in, float* __restrict__ pre_out, float* __restrict__ post_f32,
                                  float* __restrict__ comb_f32, uint16_t* __restrict__ normed, float* smem) {
  float* pre = smem;                    // [4]
  float* comb_seed = smem + kMhcN;      // [16]
  float* scratch = smem + kMhcN + kMhcN * kMhcN;  // [8]
  const float* base = site.base;
  const float* scale = site.scale;
  if (threadIdx.x < kMhcN) {
    const int i = threadIdx.x;
    const float own = mhc_sigmoid(lg[i] * scale[0] + base[i]) + site.hc_eps;
    pre_out[i] = own;
    pre[i] = pre_in != nullptr ? pre_in[i] : own;
  } else if (threadIdx.x < 2 * kMhcN) {
    const int i = threadIdx.x - kMhcN;
    post_f32[i] = 2.f * mhc_sigmoid(lg[kMhcN + i] * scale[1] + base[kMhcN + i]);
  } else if (threadIdx.x < 3 * kMhcN) {
    mhc_comb_seed_row(lg, base, scale, site.hc_eps, threadIdx.x - 2 * kMhcN, comb_seed);
  }
  __syncthreads();
  if (threadIdx.x < 32) mhc_sinkhorn_warp(comb_seed, site.hc_eps, site.sinkhorn_iters, comb_f32, threadIdx.x);
  // The collapse over runs of eight: thread t owns vectors v = t + i * 256.
  const int vecs = hidden / 8;
  const int runs = (vecs + kMhcThreads - 1) / kMhcThreads;
  const float p0 = pre[0], p1 = pre[1], p2 = pre[2], p3 = pre[3];
  float ssq = 0.f;
  for (int i = 0; i < runs; ++i) {
    const int v = threadIdx.x + i * kMhcThreads;
    if (v >= vecs) continue;
    uint4 q[kMhcN];
#pragma unroll
    for (int j = 0; j < kMhcN; ++j) q[j] = reinterpret_cast<const uint4*>(x + static_cast<size_t>(j) * hidden)[v];
    float s0[8], s1[8], s2[8], s3[8];
    mhc_unpack8(q[0], s0);
    mhc_unpack8(q[1], s1);
    mhc_unpack8(q[2], s2);
    mhc_unpack8(q[3], s3);
    uint32_t packed[4];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float val = p0 * s0[j] + p1 * s1[j] + p2 * s2[j] + p3 * s3[j];
      const uint16_t cb = float_to_bf16_bits(val);
      const float cv = bf16_bits_to_float(cb);
      ssq = __fmaf_rn(cv, cv, ssq);
      if (j % 2 == 0) packed[j / 2] = cb;
      else packed[j / 2] |= static_cast<uint32_t>(cb) << 16;
    }
    // The collapsed row is kept as bf16 in the output row until the norm rewrites it.
    reinterpret_cast<uint4*>(normed)[v] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
  }
  const float total = mhc_block_sum(ssq, scratch);
  const float rstd = rsqrtf(total / static_cast<float>(hidden) + site.ln_eps);
  for (int i = 0; i < runs; ++i) {
    const int v = threadIdx.x + i * kMhcThreads;
    if (v >= vecs) continue;
    float kept[8], lw[8];
    mhc_unpack8(reinterpret_cast<const uint4*>(normed)[v], kept);
    mhc_unpack8(reinterpret_cast<const uint4*>(site.ln)[v], lw);
    uint32_t packed[4];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float xn = kept[j] * rstd;
      const uint16_t y = float_to_bf16_bits(__fmul_rn(lw[j], xn));
      if (j % 2 == 0) packed[j / 2] = y;
      else packed[j / 2] |= static_cast<uint32_t>(y) << 16;
    }
    reinterpret_cast<uint4*>(normed)[v] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
  }
}

// The one-rounding stream update for a chunk of (token, d) outputs
// [i0, i1) of tokens * hidden: streams_out[t][i][d] = bf16(post[t][i] * h[t][d] +
// sum_j comb[t][j][i] * streams_in[t][j][d]) (glm_mhc.cu mhc_stream_update_f32_kernel).
__device__ inline void mhc_update_chunk(const float* __restrict__ post, const float* __restrict__ comb, const uint16_t* __restrict__ h,
                                        const uint16_t* __restrict__ streams_in, uint16_t* __restrict__ streams_out, int hidden, int i0,
                                        int i1) {
#pragma unroll 4
  for (int idx = i0 + static_cast<int>(threadIdx.x); idx < i1; idx += kMhcThreads) {
    const int d = idx % hidden;
    const size_t t = static_cast<size_t>(idx / hidden);
    const float hv = bf16_bits_to_float(h[t * hidden + d]);
    const uint16_t* res = streams_in + t * kMhcN * hidden;
    float rv[kMhcN];
#pragma unroll
    for (int j = 0; j < kMhcN; ++j) rv[j] = bf16_bits_to_float(res[j * hidden + d]);
#pragma unroll
    for (int i = 0; i < kMhcN; ++i) {
      float mix = 0.f;
#pragma unroll
      for (int j = 0; j < kMhcN; ++j) mix = __fmaf_rn(comb[t * kMhcN * kMhcN + j * kMhcN + i], rv[j], mix);
      const float v = __fmaf_rn(post[t * kMhcN + i], hv, mix);
      streams_out[t * kMhcN * hidden + static_cast<size_t>(i) * hidden + d] = float_to_bf16_bits(v);
    }
  }
}

}  // namespace dgpp::step
