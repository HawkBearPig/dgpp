// The distributed pick's row scan as device bodies (2026-11-01): pick.cu's
// kernels and the step kernel's pick units (kernels/step/stream.cu) run one
// code over a row's logits. The canonical order (logit desc, id asc) is a
// total order over distinct ids, so the top-2 a merge tree reaches is the
// scan's whatever the thread count.
#pragma once

#include <cstdint>
#include <cmath>

#include <cuda_runtime.h>

#include "kernels/pick.hpp"

namespace dgpp::pick_body {

constexpr int kWarp = 32;
constexpr int32_t kNoId = 0x7fffffff;  // sorts after every real id

struct Pair {
  float v;
  int32_t id;
};

// sample::candidate_before, on the device.
__host__ __device__ inline bool before(Pair a, Pair b) {
  if (a.v != b.v) return a.v > b.v;
  return a.id < b.id;
}

struct Top2 {
  Pair a;  // the row's argmax
  Pair b;  // the runner-up
};

__host__ __device__ inline Top2 empty_top2() {
  const Pair none{-INFINITY, kNoId};
  return Top2{none, none};
}

__host__ __device__ inline void push(Top2& t, Pair p) {
  if (before(p, t.a)) {
    t.b = t.a;
    t.a = p;
  } else if (before(p, t.b)) {
    t.b = p;
  }
}

__device__ inline Top2 merge(Top2 x, const Top2& y) {
  push(x, y.a);
  push(x, y.b);
  return x;
}

// xor butterfly, not shfl_down: every lane pairs with a DISTINCT lane at
// every step, so no Top2 is ever merged with itself (a self-merge would
// duplicate the argmax into the runner-up slot, and ids are unique across
// lanes only as long as no lane's pair is its own copy).
__device__ inline Top2 shfl_xor(const Top2& t, int mask) {
  Top2 r;
  r.a.v = __shfl_xor_sync(0xffffffffu, t.a.v, mask);
  r.a.id = __shfl_xor_sync(0xffffffffu, t.a.id, mask);
  r.b.v = __shfl_xor_sync(0xffffffffu, t.b.v, mask);
  r.b.id = __shfl_xor_sync(0xffffffffu, t.b.id, mask);
  return r;
}

__device__ inline void encode_digits(uint16_t* slots, uint64_t value,
                                     int digits) {
  for (int d = 0; d < digits; ++d)
    slots[d] = static_cast<uint16_t>((value >> (6 * d)) & 63);
}

__device__ inline uint64_t decode_digits(const uint16_t* slots, int digits) {
  uint64_t value = 0;
  for (int d = 0; d < digits; ++d)
    value |= static_cast<uint64_t>(slots[d] & 63) << (6 * d);
  return value;
}

// One row's top-2 under the canonical order over logits[vocab_count] whose
// ids start at vocab_begin, by a block of kThreads: a strided scan per
// thread, warp shuffles, one shared pass (`warps`: kThreads / 32 slots).
// Thread 0 holds the result; the others return an empty pair.
template <int kThreads>
__device__ inline Top2 row_top2(const float* __restrict__ slice, int vocab_count, int vocab_begin, Top2* warps) {
  const int tid = threadIdx.x;
  Top2 mine = empty_top2();
  for (int i = tid; i < vocab_count; i += kThreads) push(mine, Pair{slice[i], vocab_begin + i});
  for (int mask = kWarp / 2; mask > 0; mask >>= 1) mine = merge(mine, shfl_xor(mine, mask));
  if ((tid & (kWarp - 1)) == 0) warps[tid / kWarp] = mine;
  __syncthreads();
  if (tid != 0) return empty_top2();
  Top2 best = warps[0];
  for (int w = 1; w < kThreads / kWarp; ++w) best = merge(best, warps[w]);
  return best;
}

}  // namespace dgpp::pick_body
