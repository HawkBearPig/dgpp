#pragma once
// The CSA2 attention finish's block body, shared by the production kernel
// (kernels/csa2.cu) and the step engine's attention finish unit
// (kernels/step/stream.cu): the same body at a (row, head), so the unit's
// rows are the kernel's bitwise. kernels/csa2.cu's until 2026-11-01.
#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "kernels/csa2.hpp"

namespace dgpp {
namespace csa2_finish {

using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;

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
// The finish's shared memory: the production kernel's static block, the step
// engine's carved from its dynamic shared memory (kernels/step/stream.cu).
struct FinishSmem {
  float o[kCsa2Latent];
  float s_w[kFinishMaxSplits];          // exp(m - mhat) of the live splits, in order
  const float* s_c[kFinishMaxSplits];   // their c rows [kCsa2Latent]
  float s_m[kFinishMaxSplits];
  float s_l[kFinishMaxSplits];          // every split's l, staged beside its m (2026-10-20)
  float s_den;
  int s_live;
};
// The finish of one (row, head) by one block of 128 threads or more (the loops
// stride by the block; each output element's sum is one thread's, in split
// order, so the result does not depend on the block's width). The production
// kernel below and the step engine's attention finish unit run this body.
__device__ __forceinline__ void attn_finish_body(int r, int h, FinishSmem& sm, const float* m_main, const float* l_main,
                                                 const float* c_main, int n_main, const float* m_win, const float* l_win,
                                                 const float* c_win, int n_win, const float* sink, int local_heads,
                                                 const int64_t* pos, const float* inv_freq, uint16_t* out) {
  float* o = sm.o;
  float* s_w = sm.s_w;
  const float** s_c = sm.s_c;
  float* s_m = sm.s_m;
  float* s_l = sm.s_l;
  float& s_den = sm.s_den;
  int& s_live = sm.s_live;
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

}  // namespace csa2_finish
}  // namespace dgpp
