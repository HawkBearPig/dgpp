#pragma once
// The step engine's router unit (MOONSHOT-1.md phase 3): the production
// router's dot as a consumer pool's unit, bitwise its scores. Shared by the
// stream kernel and the standalone probe in tests/cuda/step_consumer_test.cu.
#include <climits>
#include <cstdint>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "kernels/step/mxfp4_expert.cuh"
#include "kernels/step/stream.hpp"

namespace dgpp::step {

// The form the stream kernel uses (the probe measures the others).
constexpr int kRouterBatch = 5;  // measured 2026-10-31 standalone: 11.1 us a wave at five ahead, 11.8 at ten, 17.6 one at a time with eight tokens always
constexpr bool kRouterPipeline = false;  // the next batch's loads in flight under this one's fmas: 12.7 us standalone against 11.9 without, and 38 us a class inside the kernel against 23 (2026-10-31)
constexpr bool kRouterSpecialize = true;

// ---- the router -----------------------------------------------------------
// The production router's dot (kernels/glm_moe.cu router_dot), op for op: one
// warp per expert row, each lane its strided 16-byte vectors of the row and
// of the token's activations, eight fmas a vector in the file's order, then
// the xor tree. Here a warp carries up to eight tokens' accumulators through
// one read of its row, with the activations staged in shared memory; each
// token's chain is the production's, so the scores are bitwise its.
// The warp argmax with the lower id on ties, the production select's rule (kernels/glm_moe.cu).
__device__ __forceinline__ void router_argmax_lowest_id(float& v, int& id) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_xor_sync(0xFFFFFFFFu, v, off);
    const int oid = __shfl_xor_sync(0xFFFFFFFFu, id, off);
    if (ov > v || (ov == v && oid < id)) { v = ov; id = oid; }
  }
}
constexpr int kRouterUnitRows = 16;  // two expert rows a warp, sharing each staged activation vector: 24 units in one wave
constexpr int kRouterTokens = 8;  // tokens a pass stages
__device__ __forceinline__ float router_sqrt_softplus(float x) {
  const float sp = x > 20.f ? x : log1pf(expf(x));
  return sqrtf(sp);
}
// The two rows' dots against NT staged tokens: the row vectors of a batch
// loaded before any is used (the unit was latency-bound at two in flight),
// each (row, token) chain in the production's lane order.
// A unit's timeline for the trace: the global timer, and stamps at the unit's stages into
// `tr` (eight words a block: entry, activations staged, each row batch done, the scan done).
__device__ __forceinline__ unsigned long long router_timer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
constexpr int kRouterTraceWords = 8;
template <int kK, int NT, int kBatchWanted, bool kPipeline>
__device__ __forceinline__ void router_rows(const uint4* __restrict__ wv0, const uint4* __restrict__ wv1, const uint16_t* s_x, int lane,
                                            float (&dot0)[kRouterTokens], float (&dot1)[kRouterTokens], unsigned long long* tr = nullptr) {
  constexpr int kVecs = kK / 8 / 32;  // vectors a lane walks
  constexpr int kBatch = kVecs < kBatchWanted ? kVecs : kBatchWanted;
  static_assert(kVecs % kBatch == 0, "the row's vectors in whole batches");
  constexpr int kBatches = kVecs / kBatch;
  // With kPipeline the next batch's loads are in flight while this one is computed. The loads
  // are streaming (evict-first): the read-only-cache form took the unit's class from 23 to 38 µs
  // inside the kernel while the standalone probe saw no difference (2026-10-31).
  uint4 q0[2][kBatch], q1[2][kBatch];
#pragma unroll
  for (int i = 0; i < kBatch; ++i) { q0[0][i] = __ldcs(wv0 + lane + i * 32); q1[0][i] = __ldcs(wv1 + lane + i * 32); }
#pragma unroll
  for (int nb = 0; nb < kBatches; ++nb) {
    const int b = nb * kBatch, cur = kPipeline ? (nb & 1) : 0;
    if (kPipeline && nb + 1 < kBatches) {
#pragma unroll
      for (int i = 0; i < kBatch; ++i) { q0[cur ^ 1][i] = __ldcs(wv0 + lane + (b + kBatch + i) * 32); q1[cur ^ 1][i] = __ldcs(wv1 + lane + (b + kBatch + i) * 32); }
    }
#pragma unroll
    for (int i = 0; i < kBatch; ++i) {
      const int v = lane + (b + i) * 32;
      const uint32_t w0[4] = {q0[cur][i].x, q0[cur][i].y, q0[cur][i].z, q0[cur][i].w}, w1[4] = {q1[cur][i].x, q1[cur][i].y, q1[cur][i].z, q1[cur][i].w};
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
    if (!kPipeline && nb + 1 < kBatches) {
#pragma unroll
      for (int i = 0; i < kBatch; ++i) { q0[0][i] = __ldcs(wv0 + lane + (b + kBatch + i) * 32); q1[0][i] = __ldcs(wv1 + lane + (b + kBatch + i) * 32); }
    }
    if (tr && threadIdx.x == 0 && 2 + nb < kRouterTraceWords - 1) tr[2 + nb] = router_timer();  // batch nb computed (warp 0's view)
  }
}
// kBatch: row vectors a lane loads before using any (1: one at a time, the
// first form). kSpecialize: the token count specialized (false: eight tokens
// always, rows past the group staging row 0 again for finite operands).
// With `cand_id` set, the unit also writes its top-k candidates a token over its 16 rows
// (the production select's rule: the larger biased score, the lower id on ties), at
// cand_*[tok * n_experts + rb * top_k + i]; rows past the experts, or past the k-th, write id -1.
template <int kK, int kBatch, bool kSpecialize, bool kPipeline = true>
__device__ void router_unit(const uint8_t* __restrict__ w_base, const DevPool& pool, int rb, Mxfp4Group g, const uint16_t* __restrict__ x,
                            float* __restrict__ scores, float* __restrict__ biased, int n_experts, int mode, uint8_t* smem, int top_k = 0,
                            int32_t* __restrict__ cand_id = nullptr, float* __restrict__ cand_score = nullptr, float* __restrict__ cand_biased = nullptr,
                            unsigned long long* tr = nullptr) {
  if (tr && threadIdx.x == 0) tr[0] = router_timer();
  static_assert(kK % 256 == 0, "the router's inner width");
  const uint16_t* w = reinterpret_cast<const uint16_t*>(w_base + pool.offsets[0]);
  const float* bias = reinterpret_cast<const float*>(w_base + pool.offsets[1]);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint16_t* s_x = reinterpret_cast<uint16_t*>(smem);
  float* s_b = reinterpret_cast<float*>(smem + static_cast<size_t>(kRouterTokens) * kK * 2);  // [8 tokens][16 rows] biased, then scores
  float* s_s = s_b + kRouterTokens * kRouterUnitRows;
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
    if (tr && threadIdx.x == 0) tr[1] = router_timer();  // the activations staged
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
        router_rows<kK, kRouterTokens, kBatch, kPipeline>(wv0, wv1, s_x, lane, dot0, dot1, tr);
      } else {
        switch (nt) {
          case 1: router_rows<kK, 1, kBatch, kPipeline>(wv0, wv1, s_x, lane, dot0, dot1, tr); break;
          case 2: router_rows<kK, 2, kBatch, kPipeline>(wv0, wv1, s_x, lane, dot0, dot1, tr); break;
          case 3: router_rows<kK, 3, kBatch, kPipeline>(wv0, wv1, s_x, lane, dot0, dot1, tr); break;
          case 4: router_rows<kK, 4, kBatch, kPipeline>(wv0, wv1, s_x, lane, dot0, dot1, tr); break;
          case 5: router_rows<kK, 5, kBatch, kPipeline>(wv0, wv1, s_x, lane, dot0, dot1, tr); break;
          case 6: router_rows<kK, 6, kBatch, kPipeline>(wv0, wv1, s_x, lane, dot0, dot1, tr); break;
          case 7: router_rows<kK, 7, kBatch, kPipeline>(wv0, wv1, s_x, lane, dot0, dot1, tr); break;
          default: router_rows<kK, 8, kBatch, kPipeline>(wv0, wv1, s_x, lane, dot0, dot1, tr); break;
        }
      }
      // The xor trees leave every lane the sums; lane t then finishes token t (the activation and
      // its four stores) in parallel with the other tokens, where lane 0 did them all in turn.
#pragma unroll
      for (int t = 0; t < kRouterTokens; ++t) {
        float d0 = dot0[t], d1 = dot1[t];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) { d0 += __shfl_xor_sync(0xFFFFFFFFu, d0, off); d1 += __shfl_xor_sync(0xFFFFFFFFu, d1, off); }
        dot0[t] = d0;
        dot1[t] = d1;
      }
      float my0 = dot0[0], my1 = dot1[0];
#pragma unroll
      for (int t = 1; t < kRouterTokens; ++t) { my0 = lane == t ? dot0[t] : my0; my1 = lane == t ? dot1[t] : my1; }
      if (lane < nt) {
        const int t = lane, tok = ids.at(t);
        const float s0 = mode == 2 ? router_sqrt_softplus(my0) : 1.0f / (1.0f + expf(-my0));
        const float b0 = s0 + bias[e0];
        scores[static_cast<size_t>(tok) * n_experts + e0] = s0;
        biased[static_cast<size_t>(tok) * n_experts + e0] = b0;
        s_b[t * kRouterUnitRows + warp] = b0;
        s_s[t * kRouterUnitRows + warp] = s0;
        float b1 = -INFINITY, s1 = 0.f;
        if (live1) {
          s1 = mode == 2 ? router_sqrt_softplus(my1) : 1.0f / (1.0f + expf(-my1));
          b1 = s1 + bias[e1];
          scores[static_cast<size_t>(tok) * n_experts + e1] = s1;
          biased[static_cast<size_t>(tok) * n_experts + e1] = b1;
        }
        s_b[t * kRouterUnitRows + warp + (kStepThreads / 32)] = b1;
        s_s[t * kRouterUnitRows + warp + (kStepThreads / 32)] = s1;
      }
    } else if (lane == 0) {
#pragma unroll
      for (int t = 0; t < kRouterTokens; ++t) { s_b[t * kRouterUnitRows + warp] = -INFINITY; s_b[t * kRouterUnitRows + warp + (kStepThreads / 32)] = -INFINITY; }
    }
    __syncthreads();
    // The unit's top-k candidates a token over its rows, the production select's rule: one warp
    // a token, a lane a row, top_k rounds of the warp argmax with the lower id on ties (a serial
    // scan by one thread a token cost the unit several microseconds, 2026-10-31).
    if (cand_id && warp < nt && top_k > 0) {
      const int t = warp, tok = ids.at(t);
      const int row = lane, e = rb * kRouterUnitRows + row;
      const bool mine = row < kRouterUnitRows && e < n_experts;
      float v = mine ? s_b[t * kRouterUnitRows + row] : -INFINITY;
      const float sc = mine ? s_s[t * kRouterUnitRows + row] : 0.f;
      // Lane r keeps round r's pick; the k lanes store their entries together after the rounds.
      int my_id = -1;
      float my_bv = -INFINITY, my_ws = 0.f;
      for (int r = 0; r < top_k; ++r) {
        float bv = v;
        int best = (mine && v > -INFINITY) ? e : INT_MAX;
        router_argmax_lowest_id(bv, best);
        const bool found = best != INT_MAX;
        const float ws = __shfl_sync(0xFFFFFFFFu, sc, found ? best - rb * kRouterUnitRows : 0);
        if (lane == r) { my_id = found ? best : -1; my_bv = found ? bv : -INFINITY; my_ws = found ? ws : 0.f; }
        if (found && e == best) v = -INFINITY;  // taken
      }
      if (lane < top_k) {
        const size_t at = static_cast<size_t>(tok) * n_experts + static_cast<size_t>(rb) * top_k + lane;
        cand_id[at] = my_id;
        cand_biased[at] = my_bv;
        cand_score[at] = my_ws;
      }
    }
    if (tr && threadIdx.x == 0) tr[kRouterTraceWords - 1] = router_timer();  // the unit done
  }
}

}  // namespace dgpp::step
