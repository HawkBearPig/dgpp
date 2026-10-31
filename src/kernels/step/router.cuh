#pragma once
// The step engine's router unit (MOONSHOT-1.md phase 3): the production
// router's dot as a consumer pool's unit, bitwise its scores. Shared by the
// stream kernel and the standalone probe in tests/cuda/step_consumer_test.cu.
#include <cstdint>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "kernels/step/mxfp4_expert.cuh"
#include "kernels/step/stream.hpp"

namespace dgpp::step {

// The form the stream kernel uses (the probe measures the others).
constexpr int kRouterBatch = 5;  // measured 2026-10-31 standalone: 11.1 us a wave at five ahead, 11.8 at ten, 17.6 one at a time with eight tokens always
constexpr bool kRouterSpecialize = true;

// ---- the router -----------------------------------------------------------
// The production router's dot (kernels/glm_moe.cu router_dot), op for op: one
// warp per expert row, each lane its strided 16-byte vectors of the row and
// of the token's activations, eight fmas a vector in the file's order, then
// the xor tree. Here a warp carries up to eight tokens' accumulators through
// one read of its row, with the activations staged in shared memory; each
// token's chain is the production's, so the scores are bitwise its.
constexpr int kRouterUnitRows = 16;  // two expert rows a warp, sharing each staged activation vector: 24 units in one wave
constexpr int kRouterTokens = 8;  // tokens a pass stages
__device__ __forceinline__ float router_sqrt_softplus(float x) {
  const float sp = x > 20.f ? x : log1pf(expf(x));
  return sqrtf(sp);
}
// The two rows' dots against NT staged tokens: the row vectors of a batch
// loaded before any is used (the unit was latency-bound at two in flight),
// each (row, token) chain in the production's lane order.
template <int kK, int NT, int kBatchWanted>
__device__ __forceinline__ void router_rows(const uint4* __restrict__ wv0, const uint4* __restrict__ wv1, const uint16_t* s_x, int lane,
                                            float (&dot0)[kRouterTokens], float (&dot1)[kRouterTokens]) {
  constexpr int kVecs = kK / 8 / 32;  // vectors a lane walks
  constexpr int kBatch = kVecs < kBatchWanted ? kVecs : kBatchWanted;
  static_assert(kVecs % kBatch == 0, "the row's vectors in whole batches");
  for (int b = 0; b < kVecs; b += kBatch) {
    uint4 q0[kBatch], q1[kBatch];
#pragma unroll
    for (int i = 0; i < kBatch; ++i) { q0[i] = __ldcs(wv0 + lane + (b + i) * 32); q1[i] = __ldcs(wv1 + lane + (b + i) * 32); }
#pragma unroll
    for (int i = 0; i < kBatch; ++i) {
      const int v = lane + (b + i) * 32;
      const uint32_t w0[4] = {q0[i].x, q0[i].y, q0[i].z, q0[i].w}, w1[4] = {q1[i].x, q1[i].y, q1[i].z, q1[i].w};
      float f0[8], f1[8];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        f0[2 * k] = bf16_bits_to_float(static_cast<uint16_t>(w0[k] & 0xFFFFu)); f0[2 * k + 1] = bf16_bits_to_float(static_cast<uint16_t>(w0[k] >> 16));
        f1[2 * k] = bf16_bits_to_float(static_cast<uint16_t>(w1[k] & 0xFFFFu)); f1[2 * k + 1] = bf16_bits_to_float(static_cast<uint16_t>(w1[k] >> 16));
      }
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        const uint4 xq = *reinterpret_cast<const uint4*>(s_x + t * kK + v * 8);
        const uint32_t xw[4] = {xq.x, xq.y, xq.z, xq.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float xl = bf16_bits_to_float(static_cast<uint16_t>(xw[k] & 0xFFFFu)), xh = bf16_bits_to_float(static_cast<uint16_t>(xw[k] >> 16));
          dot0[t] = __fmaf_rn(xl, f0[2 * k], dot0[t]);
          dot0[t] = __fmaf_rn(xh, f0[2 * k + 1], dot0[t]);
          dot1[t] = __fmaf_rn(xl, f1[2 * k], dot1[t]);
          dot1[t] = __fmaf_rn(xh, f1[2 * k + 1], dot1[t]);
        }
      }
    }
  }
}
// kBatch: row vectors a lane loads before using any (1: one at a time, the
// first form). kSpecialize: the token count specialized (false: eight tokens
// always, rows past the group staging row 0 again for finite operands).
template <int kK, int kBatch, bool kSpecialize>
__device__ void router_unit(const uint8_t* __restrict__ w_base, const DevPool& pool, int rb, const Mxfp4Group& g, const uint16_t* __restrict__ x,
                            float* __restrict__ scores, float* __restrict__ biased, int n_experts, int mode, uint8_t* smem) {
  static_assert(kK % 256 == 0, "the router's inner width");
  const uint16_t* w = reinterpret_cast<const uint16_t*>(w_base + pool.offsets[0]);
  const float* bias = reinterpret_cast<const float*>(w_base + pool.offsets[1]);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint16_t* s_x = reinterpret_cast<uint16_t*>(smem);
  const size_t xs = g.x_stride ? static_cast<size_t>(g.x_stride) : static_cast<size_t>(kK);
  for (int t0 = 0; t0 < g.count; t0 += kRouterTokens) {
    const int nt = min(kRouterTokens, g.count - t0);
    __syncthreads();  // the previous pass's readers are done with the staged rows
    const GroupTokens8 ids = group_tokens8(g, t0);  // the pass's token ids once, not a dependent load a piece
    const int staged = kSpecialize ? nt : kRouterTokens;
    for (int q = threadIdx.x; q < staged * (kK / 8); q += kStepThreads) {
      const int t = q / (kK / 8), c16 = q % (kK / 8);
      cp_async_16(s_x + t * kK + c16 * 8, x + static_cast<size_t>(ids.at(t)) * xs + c16 * 8);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    // Both of the warp's rows against every staged token through one read of each activation
    // vector; the token count specialized, so no issue slot goes to a token that is not there.
    const int e0 = rb * kRouterUnitRows + warp, e1 = e0 + (kStepThreads / 32);
    const bool live1 = e1 < n_experts;
    if (e0 < n_experts) {
      const uint4* wv0 = reinterpret_cast<const uint4*>(w + static_cast<size_t>(e0) * kK);
      const uint4* wv1 = reinterpret_cast<const uint4*>(w + static_cast<size_t>(live1 ? e1 : e0) * kK);
      float dot0[kRouterTokens], dot1[kRouterTokens];
#pragma unroll
      for (int t = 0; t < kRouterTokens; ++t) { dot0[t] = 0.f; dot1[t] = 0.f; }
      if (!kSpecialize) {
        router_rows<kK, kRouterTokens, kBatch>(wv0, wv1, s_x, lane, dot0, dot1);
      } else {
        switch (nt) {
          case 1: router_rows<kK, 1, kBatch>(wv0, wv1, s_x, lane, dot0, dot1); break;
          case 2: router_rows<kK, 2, kBatch>(wv0, wv1, s_x, lane, dot0, dot1); break;
          case 3: router_rows<kK, 3, kBatch>(wv0, wv1, s_x, lane, dot0, dot1); break;
          case 4: router_rows<kK, 4, kBatch>(wv0, wv1, s_x, lane, dot0, dot1); break;
          case 5: router_rows<kK, 5, kBatch>(wv0, wv1, s_x, lane, dot0, dot1); break;
          case 6: router_rows<kK, 6, kBatch>(wv0, wv1, s_x, lane, dot0, dot1); break;
          case 7: router_rows<kK, 7, kBatch>(wv0, wv1, s_x, lane, dot0, dot1); break;
          default: router_rows<kK, 8, kBatch>(wv0, wv1, s_x, lane, dot0, dot1); break;
        }
      }
#pragma unroll
      for (int t = 0; t < kRouterTokens; ++t) {
        float d0 = dot0[t], d1 = dot1[t];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) { d0 += __shfl_xor_sync(0xFFFFFFFFu, d0, off); d1 += __shfl_xor_sync(0xFFFFFFFFu, d1, off); }
        if (lane == 0 && t < nt) {
          const int tok = ids.at(t);
          const float s0 = mode == 2 ? router_sqrt_softplus(d0) : 1.0f / (1.0f + expf(-d0));
          scores[static_cast<size_t>(tok) * n_experts + e0] = s0;
          biased[static_cast<size_t>(tok) * n_experts + e0] = s0 + bias[e0];
          if (live1) {
            const float s1 = mode == 2 ? router_sqrt_softplus(d1) : 1.0f / (1.0f + expf(-d1));
            scores[static_cast<size_t>(tok) * n_experts + e1] = s1;
            biased[static_cast<size_t>(tok) * n_experts + e1] = s1 + bias[e1];
          }
        }
      }
    }
  }
}

}  // namespace dgpp::step
