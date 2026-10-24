#include "kernels/kda.hpp"
#include "kernels/pdl.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {

namespace {

// ---------------------------------------------------------------------------
// Causal depthwise conv + silu
// ---------------------------------------------------------------------------

template <int CW, bool kBatched>
__global__ void kda_conv_kernel(const uint16_t* __restrict__ src,
                                int64_t src_stride,
                                const uint16_t* __restrict__ weight,
                                uint16_t* __restrict__ state,
                                int64_t request_state_stride, int state_width,
                                uint16_t* __restrict__ dst, int tokens,
                                int channels,
                                uint16_t* __restrict__ snapshots,
                                int64_t snapshot_stride,
                                const int32_t* __restrict__ request_ids,
                                const int64_t* __restrict__ positions,
                                const int32_t* __restrict__ request_spans) {
  dgpp_pdl_wait();
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  int t0 = 0;
  int t1 = tokens;
  int req = 0;
  if constexpr (kBatched) {
    const int span = blockIdx.y;
    t0 = request_spans[span * 2];
    t1 = t0 + request_spans[span * 2 + 1];
    // An inactive fixed-shape span may consist entirely of padding. Do not
    // even form a state pointer from its sentinel request id.
    int first_real = t0;
    while (first_real < t1 && positions[first_real] < 0) ++first_real;
    if (first_real == t1) {
      for (int t = t0; t < t1; ++t)
        dst[static_cast<int64_t>(t) * channels + c] = 0;
      return;
    }
    req = request_ids[first_real];
  }
  const uint16_t* wc = weight + static_cast<int64_t>(c) * CW;
  uint16_t* sc = state + static_cast<int64_t>(req) * request_state_stride +
                 static_cast<int64_t>(c) * state_width;

  float wv[CW];
#pragma unroll
  for (int j = 0; j < CW; ++j) wv[j] = bf16_bits_to_float(wc[j]);

  // Rolling register history; starts as the incoming state, ends as the
  // last CW-1 inputs regardless of how few tokens arrive (decode).
  float hist[CW - 1];
#pragma unroll
  for (int j = 0; j < CW - 1; ++j) hist[j] = bf16_bits_to_float(sc[j]);

  for (int t = t0; t < t1; ++t) {
    if constexpr (kBatched) {
      if (positions[t] < 0) {
        dst[static_cast<int64_t>(t) * channels + c] = 0;
        continue;
      }
    }
    const float x =
        bf16_bits_to_float(src[static_cast<int64_t>(t) * src_stride + c]);
    // Same association order as the host reference so the fp32 fma chains
    // agree; only expf/silu can differ by an ulp.
    float acc = wv[CW - 1] * x;
#pragma unroll
    for (int j = 0; j < CW - 1; ++j) acc = fmaf(wv[j], hist[j], acc);
    const float y = acc / (1.0f + expf(-acc));  // silu, fp32
    dst[static_cast<int64_t>(t) * channels + c] = float_to_bf16_bits(y);
#pragma unroll
    for (int j = 0; j < CW - 2; ++j) hist[j] = hist[j + 1];
    hist[CW - 2] = x;
    // Speculative rows: the history as it stands after row t is what the
    // committed state must become if rows > t are rejected. The last row
    // lands in place below, so it needs no snapshot.
    if (snapshots && t + 1 < t1) {
      uint16_t* snap = snapshots + static_cast<int64_t>(t) * snapshot_stride +
                       static_cast<int64_t>(c) * state_width;
#pragma unroll
      for (int j = 0; j < CW - 1; ++j) snap[j] = float_to_bf16_bits(hist[j]);
    }
  }

#pragma unroll
  for (int j = 0; j < CW - 1; ++j) sc[j] = float_to_bf16_bits(hist[j]);
}

// The prefill form (2026-09-23): the conv is not recurrent -- output t needs
// only inputs t-CW+1..t -- so a single request's rows split into token tiles
// across blocks instead of one thread walking every row of the chunk (the
// 32k profile: 590 ms against SGLang's 110 for the same conv; the per-channel
// grid was ~16 blocks on a 48-SM part). Each output keeps the association
// order above, so the tiles are bitwise the serial kernel. The new state is
// written by kda_conv_state_kernel after every tile has read the old one.
constexpr int kConvTileTokens = 64;
template <int CW>
__global__ void kda_conv_tiled_kernel(const uint16_t* __restrict__ src, int64_t src_stride,
                                      const uint16_t* __restrict__ weight,
                                      const uint16_t* __restrict__ state, int state_width,
                                      uint16_t* __restrict__ dst, int tokens, int channels) {
  dgpp_pdl_wait();
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  const int a = blockIdx.y * kConvTileTokens;
  const int b = min(tokens, a + kConvTileTokens);
  const uint16_t* wc = weight + static_cast<int64_t>(c) * CW;
  float wv[CW];
#pragma unroll
  for (int j = 0; j < CW; ++j) wv[j] = bf16_bits_to_float(wc[j]);
  // hist[j] = input at position a - (CW-1) + j; negative positions come from
  // the incoming state (state[j'] is position j' - (CW-1)).
  float hist[CW - 1];
#pragma unroll
  for (int j = 0; j < CW - 1; ++j) {
    const int pos = a - (CW - 1) + j;
    hist[j] = pos >= 0 ? bf16_bits_to_float(src[static_cast<int64_t>(pos) * src_stride + c])
                       : bf16_bits_to_float(state[static_cast<int64_t>(c) * state_width + pos + (CW - 1)]);
  }
  for (int t = a; t < b; ++t) {
    const float x = bf16_bits_to_float(src[static_cast<int64_t>(t) * src_stride + c]);
    float acc = wv[CW - 1] * x;
#pragma unroll
    for (int j = 0; j < CW - 1; ++j) acc = fmaf(wv[j], hist[j], acc);
    const float y = acc / (1.0f + expf(-acc));
    dst[static_cast<int64_t>(t) * channels + c] = float_to_bf16_bits(y);
#pragma unroll
    for (int j = 0; j < CW - 2; ++j) hist[j] = hist[j + 1];
    hist[CW - 2] = x;
  }
}
// The committed state after a tiled prefill: the last CW-1 inputs (tokens >= CW-1).
template <int CW>
__global__ void kda_conv_state_kernel(const uint16_t* __restrict__ src, int64_t src_stride,
                                      uint16_t* __restrict__ state, int state_width, int tokens,
                                      int channels) {
  dgpp_pdl_wait();
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
#pragma unroll
  for (int j = 0; j < CW - 1; ++j)
    // Match the serial state's BF16 round-trip, including NaN canonicalization.
    state[static_cast<int64_t>(c) * state_width + j] =
        float_to_bf16_bits(bf16_bits_to_float(
            src[static_cast<int64_t>(tokens - (CW - 1) + j) * src_stride + c]));
}

// ---------------------------------------------------------------------------
// Gated RMSNorm (sigmoid gate)
// ---------------------------------------------------------------------------

__global__ void kda_gated_rmsnorm_kernel(const uint16_t* __restrict__ x,
                                         const uint16_t* __restrict__ gate,
                                         const uint16_t* __restrict__ weight,
                                         uint16_t* __restrict__ y, int dim,
                                         float eps) {
  dgpp_pdl_wait();
  const int64_t row = blockIdx.x;
  const uint16_t* xr = x + row * dim;
  const uint16_t* gr = gate + row * dim;
  uint16_t* yr = y + row * dim;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  extern __shared__ float sx[];  // staged x row for the second pass
  float partial = 0.0f;
  for (int i = tid; i < dim; i += nthreads) {
    const float v = bf16_bits_to_float(xr[i]);
    sx[i] = v;
    partial = fmaf(v, v, partial);
  }
  // Warp-level tree reduction, then across warps via shared memory.
  __shared__ float warp_sums[32];
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    partial += __shfl_xor_sync(0xffffffff, partial, off);
  if ((tid & 31) == 0) warp_sums[tid >> 5] = partial;
  __syncthreads();
  float total = 0.0f;
  const int nwarps = (nthreads + 31) / 32;
  for (int w = 0; w < nwarps; ++w) total += warp_sums[w];
  // Reference: 1 / sqrt(mean(x^2) + eps), fp32.
  const float rstd = 1.0f / sqrtf(total / static_cast<float>(dim) + eps);

  for (int i = tid; i < dim; i += nthreads) {
    const float xv = sx[i];
    const float g = bf16_bits_to_float(gr[i]);
    const float w = bf16_bits_to_float(weight[i]);
    const float sig = 1.0f / (1.0f + expf(-g));
    yr[i] = float_to_bf16_bits(xv * rstd * w * sig);
  }
}

// ---------------------------------------------------------------------------
// Recurrent KDA update
// ---------------------------------------------------------------------------

// 16 lanes share a v-row (8 columns each at K=128): 4 lanes x 32 columns
// left the decode kernel at 1.3 warps per scheduler with 188 registers,
// latency-bound on its own state loads (15 us for 2 MB of state traffic);
// four times the warps hide it. The lane split changes the order of the
// row reductions (q/k norms, <S,k>, <S,q>) by fp32 rounding — accepted
// 2026-09-02, the kda oracle tests measure it.
constexpr int kRecurrentLanes = 16;  // k-dim lanes cooperating on one v-row
constexpr int kRecurrentRows = 8;    // v-rows per block
constexpr int kRecurrentBlock = kRecurrentLanes * kRecurrentRows;  // 128

// A lane's kCols-wide slice of a row, loaded/stored as 16-byte vectors when
// the launcher verified alignment (kVec) and scalar otherwise. Purely the
// access pattern: the values, and every operation on them, are identical.
// With scalar accesses each warp-wide load touched 32 sectors for 128
// useful bytes, and the decode kernel spent half its cycles with the L1's
// load/store queue full (ncu: 33 cycles per issued instruction, 1.3
// active warps per scheduler) — 32 us for 2 MB of state traffic.
template <int kCols, bool kVec>
__device__ __forceinline__ void load_f32_slice(const float* __restrict__ p,
                                               float (&v)[kCols]) {
  if constexpr (kVec && kCols % 4 == 0) {
#pragma unroll
    for (int i = 0; i < kCols / 4; ++i) {
      const float4 q = reinterpret_cast<const float4*>(p)[i];
      v[4 * i] = q.x;
      v[4 * i + 1] = q.y;
      v[4 * i + 2] = q.z;
      v[4 * i + 3] = q.w;
    }
  } else {
#pragma unroll
    for (int i = 0; i < kCols; ++i) v[i] = p[i];
  }
}

template <int kCols, bool kVec>
__device__ __forceinline__ void store_f32_slice(float* __restrict__ p,
                                                const float (&v)[kCols]) {
  if constexpr (kVec && kCols % 4 == 0) {
#pragma unroll
    for (int i = 0; i < kCols / 4; ++i)
      reinterpret_cast<float4*>(p)[i] =
          make_float4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
  } else {
#pragma unroll
    for (int i = 0; i < kCols; ++i) p[i] = v[i];
  }
}

template <int kCols, bool kVec>
__device__ __forceinline__ void load_bf16_slice(const uint16_t* __restrict__ p,
                                                float (&v)[kCols]) {
  if constexpr (kVec && kCols % 4 == 0) {
#pragma unroll
    for (int i = 0; i < kCols / 4; ++i) {
      const uint2 q = reinterpret_cast<const uint2*>(p)[i];
      const uint32_t w[2] = {q.x, q.y};
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        v[4 * i + 2 * j] = bf16_bits_to_float(static_cast<uint16_t>(w[j] & 0xFFFFu));
        v[4 * i + 2 * j + 1] = bf16_bits_to_float(static_cast<uint16_t>(w[j] >> 16));
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < kCols; ++i) v[i] = bf16_bits_to_float(p[i]);
  }
}

// kScalarGate: the Gated DeltaNet variant of the same
// recurrence (docs/qwen38_flash_next_plan.md §1.3). The decay is one scalar
// per head and token, exp(-exp(A_log[h]) * softplus(a_raw[t,h] +
// dt_bias[h])), and kv_ratio value heads share one key head's q and k (the
// reference's repeat_interleave). g_raw is then a_raw [tokens, heads] with
// row stride g_stride, dt_bias is [heads], and lower_bound is unused. With
// kScalarGate false and kv_ratio 1 every expression below is the KDA
// kernel's, index for index.
template <int K, bool kVec, bool kBatched, bool kScalarGate>
__global__ __launch_bounds__(kRecurrentBlock) void kda_recurrent_kernel(
    const uint16_t* __restrict__ qkv, const uint16_t* __restrict__ g_raw,
    int64_t g_stride, const uint16_t* __restrict__ beta_raw,
    int64_t beta_stride, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ state,
    int64_t request_state_stride, uint16_t* __restrict__ out, int tokens,
    int heads, int kv_ratio, int v_dim, float lower_bound, float scale,
    float* __restrict__ snapshots, int64_t snapshot_stride,
    const int32_t* __restrict__ request_ids,
    const int64_t* __restrict__ positions,
    const int32_t* __restrict__ request_spans, KdaReplay replay) {
  dgpp_pdl_wait();
  constexpr int kCols = K / kRecurrentLanes;  // columns owned per lane
  static_assert(K % kRecurrentLanes == 0, "K must split across the lanes");

  const int h = blockIdx.y;
  const int hk = h / kv_ratio;          // the key head this value head reads
  const int heads_k = heads / kv_ratio;  // key heads
  const int v = blockIdx.x * kRecurrentRows + threadIdx.x / kRecurrentLanes;
  const int lane = threadIdx.x % kRecurrentLanes;
  const int c0 = lane * kCols;
  const bool row_valid = v < v_dim;

  int t0 = 0;
  int t1 = tokens;
  int req = 0;
  if constexpr (kBatched) {
    const int span = blockIdx.z;
    t0 = request_spans[span * 2];
    t1 = t0 + request_spans[span * 2 + 1];
    int first_real = t0;
    while (first_real < t1 && positions[first_real] < 0) ++first_real;
    if (first_real == t1) {
      if (row_valid && lane == 0) {
        for (int t = t0; t < t1; ++t)
          out[(static_cast<int64_t>(t) * heads + h) * v_dim + v] = 0;
      }
      return;
    }
    req = request_ids[first_real];
  }

  // State slice S[v, c0:c0+kCols] lives in registers for the whole call:
  // one load + one store per head per chunk — the decode hot path is pure
  // state bandwidth by design.
  float s[kCols];
  const int64_t state_elem =
      (static_cast<int64_t>(h) * v_dim + v) * K + c0;
  float* st = state + static_cast<int64_t>(req) * request_state_stride +
              state_elem;
  if (row_valid) {
    load_f32_slice<kCols, kVec>(st, s);
  } else {
#pragma unroll
    for (int i = 0; i < kCols; ++i) s[i] = 0.0f;
  }
  // The per-head decay bias slice is token-invariant: hoist it (the
  // per-dimension gate); the scalar gate's bias is one float per head.
  float bias_v[kCols];
  if constexpr (!kScalarGate) {
    load_f32_slice<kCols, kVec>(dt_bias + static_cast<int64_t>(h) * K + c0,
                                bias_v);
  } else {
#pragma unroll
    for (int i = 0; i < kCols; ++i) bias_v[i] = 0.0f;
  }
  const float bias_h = kScalarGate ? dt_bias[h] : 0.0f;

  const float a = expf(a_log[h]);
  // Fused qkv row: [q (Hk*K) | k (Hk*K) | v (H*V)].
  const int64_t qkv_stride = static_cast<int64_t>(2) * heads_k * K +
                             static_cast<int64_t>(heads) * v_dim;

  // One token of the recurrence on the register-resident slice: the same
  // sequence of operations whether the token is one of this call's rows
  // or a replayed row of the previous pass (KdaReplay), which is what
  // makes the replayed state bitwise the snapshot it stands in for.
  //   q_h, k_h: this key head's q and k rows (lane offset applied inside)
  //   grow:     the per-dimension gate row slice (KDA), unused for the scalar gate
  //   g_bits:   the scalar gate's a_raw[t, h] (GDN), unused for KDA
  //   vin:      v[t, h, v] (0 for an invalid row)
  //   beta_bits: beta_raw[t, h]
  // Returns the post-update read o[v] (every lane holds the full sum).
  const auto token = [&](const uint16_t* __restrict__ q_h, const uint16_t* __restrict__ k_h,
                         const uint16_t* __restrict__ grow, uint16_t g_bits, float vin,
                         uint16_t beta_bits) -> float {
    float q[kCols], k[kCols], gv[kCols], u = 0.0f;
    load_bf16_slice<kCols, kVec>(q_h + c0, q);
    load_bf16_slice<kCols, kVec>(k_h + c0, k);
    float decay_h = 1.0f;
    if constexpr (kScalarGate) {
      // GDN: g = -exp(A_log) * softplus(a_raw + dt_bias), the decay exp(g);
      // torch's softplus (threshold 20) in fp32.
      const float x = bf16_bits_to_float(g_bits) + bias_h;
      const float sp = x > 20.0f ? x : log1pf(expf(x));
      decay_h = expf(-(a * sp));
    } else {
      load_bf16_slice<kCols, kVec>(grow, gv);
    }
    float qs = 0.0f, ks = 0.0f;
#pragma unroll
    for (int i = 0; i < kCols; ++i) {
      qs = fmaf(q[i], q[i], qs);
      ks = fmaf(k[i], k[i], ks);
      if constexpr (kScalarGate) {
        s[i] *= decay_h;
      } else {
        // Gate: lower_bound / (1 + exp(-exp(A_log) * (g_raw + dt_bias))).
        // Bounded to (lower_bound, 0), so exp(gate) in (exp(lb), 1] — the
        // decay can never blow the state up. Computed fused with the decay.
        const float g = gv[i] + bias_v[i];
        const float gate = lower_bound / (1.0f + expf(-(a * g)));
        s[i] *= expf(gate);
      }
    }

    // l2norm(q, k) with eps inside the sqrt; butterfly across the row's
    // lanes so every lane ends with the full sum (reference tl.sum).
#pragma unroll
    for (int mask = 1; mask < kRecurrentLanes; mask <<= 1) {
      qs += __shfl_xor_sync(0xffffffff, qs, mask);
      ks += __shfl_xor_sync(0xffffffff, ks, mask);
    }
    const float qn = 1.0f / sqrtf(qs + 1e-6f);
    const float kn = 1.0f / sqrtf(ks + 1e-6f);

#pragma unroll
    for (int i = 0; i < kCols; ++i) {
      q[i] = q[i] * qn * scale;
      k[i] = k[i] * kn;
      u = fmaf(s[i], k[i], u);  // partial <S[v,:], k>
    }
#pragma unroll
    for (int mask = 1; mask < kRecurrentLanes; mask <<= 1)
      u += __shfl_xor_sync(0xffffffff, u, mask);

    const float beta = 1.0f / (1.0f + expf(-bf16_bits_to_float(beta_bits)));
    // u <- beta * (v - <S k>); the delta error.
    u = (vin - u) * beta;

    float o = 0.0f;
#pragma unroll
    for (int i = 0; i < kCols; ++i) {
      s[i] = fmaf(u, k[i], s[i]);  // rank-1 write
      o = fmaf(s[i], q[i], o);     // post-update read
    }
#pragma unroll
    for (int mask = 1; mask < kRecurrentLanes; mask <<= 1)
      o += __shfl_xor_sync(0xffffffff, o, mask);
    return o;
  };

  // The checkpoint-and-replay form (KdaReplay): the previous pass's
  // accepted rows first, from their saved inputs, then the checkpoint
  // store (the materialize form stops there).
  bool checkpoint = false;
  if constexpr (kScalarGate) {
    if (replay.in != nullptr) {
      int nrep = replay.materialize_rows >= 0
                     ? replay.materialize_rows
                     : (replay.count != nullptr ? replay.count[req] : 0);
      if (nrep > replay.rows_cap) nrep = replay.rows_cap;
      const uint16_t* rin = replay.in + static_cast<int64_t>(req) * replay.request_stride;
      for (int r = 0; r < nrep; ++r) {
        const uint16_t* row = rin + static_cast<int64_t>(r) * replay.in_stride;
        const uint16_t* q_h = row + static_cast<int64_t>(hk) * K;
        const uint16_t* k_h = q_h + static_cast<int64_t>(heads_k) * K;
        const uint16_t* v_h = row + static_cast<int64_t>(2) * heads_k * K + static_cast<int64_t>(h) * v_dim;
        const float vin = row_valid ? bf16_bits_to_float(v_h[v]) : 0.0f;
        (void)token(q_h, k_h, nullptr, row[qkv_stride + h], vin, row[qkv_stride + heads + h]);
      }
      if (replay.materialize) {
        // Nothing pending and in place: the live buffer already is the state.
        if (nrep == 0 && replay.dst == nullptr) return;
        if (row_valid) store_f32_slice<kCols, kVec>(replay.dst != nullptr ? replay.dst + state_elem : st, s);
        return;
      }
      checkpoint = replay.checkpoint;
      if (checkpoint && row_valid) store_f32_slice<kCols, kVec>(st, s);  // the state before this pass's rows
    }
  }
  uint16_t* save_base = (checkpoint && replay.save != nullptr)
                            ? replay.save + static_cast<int64_t>(req) * replay.request_stride
                            : nullptr;
  int saved = 0;

  for (int t = t0; t < t1; ++t) {
    if constexpr (kBatched) {
      if (positions[t] < 0) {
        if (row_valid && lane == 0)
          out[(static_cast<int64_t>(t) * heads + h) * v_dim + v] = 0;
        continue;
      }
    }
    const uint16_t* qrow =
        qkv + static_cast<int64_t>(t) * qkv_stride + static_cast<int64_t>(hk) * K;
    const uint16_t* krow = qrow + static_cast<int64_t>(heads_k) * K;
    const uint16_t* vrow =
        qkv + static_cast<int64_t>(t) * qkv_stride +
        static_cast<int64_t>(2) * heads_k * K + static_cast<int64_t>(h) * v_dim;
    const uint16_t* grow =
        kScalarGate ? nullptr
                    : g_raw + static_cast<int64_t>(t) * g_stride + static_cast<int64_t>(h) * K + c0;
    const uint16_t g_bits = kScalarGate ? g_raw[static_cast<int64_t>(t) * g_stride + h] : uint16_t{0};
    const uint16_t beta_bits = beta_raw[static_cast<int64_t>(t) * beta_stride + h];
    const float vin = row_valid ? bf16_bits_to_float(vrow[v]) : 0.0f;
    const float o = token(qrow, krow, grow, g_bits, vin, beta_bits);

    if (row_valid && lane == 0)
      out[(static_cast<int64_t>(t) * heads + h) * v_dim + v] =
          float_to_bf16_bits(o);
    if (save_base != nullptr && saved < replay.rows_cap) {
      // This row's inputs for the next pass's replay, each block its own
      // slices: the v element of every valid row-thread, the key head's
      // q and k slices from one block per key head (the row-thread 0 lanes,
      // kCols raw bf16 each), a_raw and beta_raw from one thread per head.
      uint16_t* srow = save_base + static_cast<int64_t>(saved) * replay.in_stride;
      if (row_valid && lane == 0)
        srow[static_cast<int64_t>(2) * heads_k * K + static_cast<int64_t>(h) * v_dim + v] = vrow[v];
      if (blockIdx.x == 0 && (h % kv_ratio) == 0 && threadIdx.x < kRecurrentLanes) {
        uint16_t* sq = srow + static_cast<int64_t>(hk) * K + c0;
        uint16_t* sk = sq + static_cast<int64_t>(heads_k) * K;
#pragma unroll
        for (int i = 0; i < kCols; ++i) {
          sq[i] = qrow[c0 + i];
          sk[i] = krow[c0 + i];
        }
      }
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        srow[qkv_stride + h] = g_bits;
        srow[qkv_stride + heads + h] = beta_bits;
      }
    }
    ++saved;
    // Speculative rows: S after row t is the state to restore if rows > t
    // are rejected (the last row's S lands in place below). One extra
    // state-sized store per speculative row; nothing on the T=1 path.
    // (The checkpoint form keeps no snapshots: the next pass replays.)
    if (snapshots && !checkpoint && t + 1 < t1 && row_valid)
      store_f32_slice<kCols, kVec>(
          snapshots + static_cast<int64_t>(t) * snapshot_stride +
              state_elem,
          s);
  }

  if (row_valid && !checkpoint) store_f32_slice<kCols, kVec>(st, s);
}

template <int CW, bool kBatched>
void conv_launch(const void* src, int64_t src_stride, const void* weight,
                 void* conv_state, int64_t request_state_stride,
                 int state_width, void* dst, int tokens, int channels,
                 const KdaRequestRows& requests,
                 const KdaConvSnapshots& snap, cudaStream_t stream) {
  constexpr int kBlock = 256;
  if constexpr (!kBatched) {
    // Single-request prefill without per-row snapshots: the token-tiled form.
    static const bool tiled = std::getenv("DGPP_KDA_CONV_SERIAL") == nullptr;
    if (tiled && snap.states == nullptr && tokens >= 2 * kConvTileTokens) {
      const dim3 tgrid(static_cast<unsigned>((channels + kBlock - 1) / kBlock),
                       static_cast<unsigned>((tokens + kConvTileTokens - 1) / kConvTileTokens), 1);
      kda_conv_tiled_kernel<CW><<<tgrid, kBlock, 0, stream>>>(
          static_cast<const uint16_t*>(src), src_stride, static_cast<const uint16_t*>(weight),
          static_cast<const uint16_t*>(conv_state), state_width, static_cast<uint16_t*>(dst), tokens, channels);
      DGPP_CUDA_OK(cudaGetLastError());
      kda_conv_state_kernel<CW><<<(channels + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
          static_cast<const uint16_t*>(src), src_stride, static_cast<uint16_t*>(conv_state), state_width,
          tokens, channels);
      DGPP_CUDA_OK(cudaGetLastError());
      return;
    }
  }
  const dim3 grid(static_cast<unsigned>((channels + kBlock - 1) / kBlock),
                  static_cast<unsigned>(kBatched ? requests.num_requests : 1),
                  1);
  kda_conv_kernel<CW, kBatched><<<grid, kBlock, 0, stream>>>(
      static_cast<const uint16_t*>(src), src_stride,
      static_cast<const uint16_t*>(weight),
      static_cast<uint16_t*>(conv_state), request_state_stride, state_width,
      static_cast<uint16_t*>(dst), tokens, channels,
      static_cast<uint16_t*>(snap.states), snap.stride_elems,
      requests.request_ids, requests.positions, requests.spans);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace

void kda_causal_conv_silu_bf16(const void* src, int64_t src_row_stride,
                               const void* weight, void* conv_state,
                               int state_width, void* dst, int tokens,
                               int channels, int conv_width,
                               cudaStream_t stream,
                               const KdaConvSnapshots& snap) {
  if (tokens <= 0 || channels <= 0)
    throw std::invalid_argument("kda conv: empty problem");
  if (conv_width < 2 || conv_width > 8)
    throw std::invalid_argument("kda conv: conv_width must be in [2, 8]");
  if (state_width < conv_width - 1)
    throw std::invalid_argument("kda conv: state narrower than history");
  if (src_row_stride < channels)
    throw std::invalid_argument("kda conv: src row stride smaller than row");
  if (snap.states && tokens > 1 &&
      snap.stride_elems < static_cast<int64_t>(channels) * state_width)
    throw std::invalid_argument("kda conv: snapshot stride smaller than a state");
  const KdaRequestRows requests{};
  const int64_t state_stride = static_cast<int64_t>(channels) * state_width;
  switch (conv_width) {
    case 2:
      conv_launch<2, false>(src, src_row_stride, weight, conv_state,
                            state_stride, state_width, dst, tokens, channels,
                            requests, snap, stream);
      return;
    case 3:
      conv_launch<3, false>(src, src_row_stride, weight, conv_state,
                            state_stride, state_width, dst, tokens, channels,
                            requests, snap, stream);
      return;
    case 4:
      conv_launch<4, false>(src, src_row_stride, weight, conv_state,
                            state_stride, state_width, dst, tokens, channels,
                            requests, snap, stream);
      return;
    case 5:
      conv_launch<5, false>(src, src_row_stride, weight, conv_state,
                            state_stride, state_width, dst, tokens, channels,
                            requests, snap, stream);
      return;
    case 6:
      conv_launch<6, false>(src, src_row_stride, weight, conv_state,
                            state_stride, state_width, dst, tokens, channels,
                            requests, snap, stream);
      return;
    case 7:
      conv_launch<7, false>(src, src_row_stride, weight, conv_state,
                            state_stride, state_width, dst, tokens, channels,
                            requests, snap, stream);
      return;
    default:
      conv_launch<8, false>(src, src_row_stride, weight, conv_state,
                            state_stride, state_width, dst, tokens, channels,
                            requests, snap, stream);
      return;
  }
}

void kda_causal_conv_silu_bf16_batched(
    const void* src, int64_t src_row_stride, const void* weight,
    void* conv_states, int64_t request_state_stride, int state_width,
    void* dst, int rows, int channels, int conv_width,
    const KdaRequestRows& requests, cudaStream_t stream,
    const KdaConvSnapshots& snap) {
  if (rows <= 0 || channels <= 0)
    throw std::invalid_argument("kda batched conv: empty problem");
  if (conv_width < 2 || conv_width > 8)
    throw std::invalid_argument(
        "kda batched conv: conv_width must be in [2, 8]");
  if (state_width < conv_width - 1)
    throw std::invalid_argument(
        "kda batched conv: state narrower than history");
  if (src_row_stride < channels)
    throw std::invalid_argument(
        "kda batched conv: src row stride smaller than row");
  const int64_t state_elems = static_cast<int64_t>(channels) * state_width;
  if (request_state_stride < state_elems)
    throw std::invalid_argument(
        "kda batched conv: request stride smaller than a state");
  if (!requests.request_ids || !requests.positions || !requests.spans ||
      requests.num_requests <= 0)
    throw std::invalid_argument("kda batched conv: incomplete request map");
  if (snap.states && snap.stride_elems < state_elems)
    throw std::invalid_argument(
        "kda batched conv: snapshot stride smaller than a state");

#define DGPP_KDA_CONV_BATCH_DISPATCH(CW)                                      \
  conv_launch<CW, true>(src, src_row_stride, weight, conv_states,             \
                        request_state_stride, state_width, dst, rows,          \
                        channels, requests, snap, stream)
  switch (conv_width) {
    case 2: DGPP_KDA_CONV_BATCH_DISPATCH(2); return;
    case 3: DGPP_KDA_CONV_BATCH_DISPATCH(3); return;
    case 4: DGPP_KDA_CONV_BATCH_DISPATCH(4); return;
    case 5: DGPP_KDA_CONV_BATCH_DISPATCH(5); return;
    case 6: DGPP_KDA_CONV_BATCH_DISPATCH(6); return;
    case 7: DGPP_KDA_CONV_BATCH_DISPATCH(7); return;
    default: DGPP_KDA_CONV_BATCH_DISPATCH(8); return;
  }
#undef DGPP_KDA_CONV_BATCH_DISPATCH
}

void kda_gated_rmsnorm_sigmoid_bf16(const void* x, const void* gate,
                                    const void* weight, void* y, int64_t rows,
                                    int dim, float eps, cudaStream_t stream) {
  if (rows <= 0 || dim <= 0)
    throw std::invalid_argument("kda gated rmsnorm: empty problem");
  if (dim > 4096)
    throw std::invalid_argument("kda gated rmsnorm: dim exceeds smem stage");
  constexpr int kBlock = 256;
  const size_t shmem = sizeof(float) * static_cast<size_t>(dim);
  if (shmem > 48 * 1024)
    throw std::invalid_argument("kda gated rmsnorm: dim too large for smem");
  cudaGetLastError();  // clear stale errors so failures attribute correctly
  kda_gated_rmsnorm_kernel<<<static_cast<unsigned>(rows), kBlock, shmem,
                             stream>>>(
      static_cast<const uint16_t*>(x), static_cast<const uint16_t*>(gate),
      static_cast<const uint16_t*>(weight), static_cast<uint16_t*>(y), dim,
      eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

namespace {

// The one launcher behind the four public entries (KDA and GDN, plain and
// request-batched): validates, picks the vector path, dispatches K.
template <bool kScalarGate>
void recurrent_launch(const char* who, const void* qkv, const void* g_raw,
                      int64_t g_row_stride, const void* beta_raw,
                      int64_t beta_row_stride, const float* a_log,
                      const float* dt_bias, float* states,
                      int64_t request_state_stride, void* out, int rows,
                      int heads, int kv_ratio, int k_dim, int v_dim,
                      float lower_bound, float scale,
                      const KdaRequestRows& requests, cudaStream_t stream,
                      const KdaStateSnapshots& snap, const KdaReplay& replay) {
  const bool batched = requests.num_requests > 0;
  const std::string w = who;
  const bool materialize = replay.in != nullptr && replay.materialize;
  if ((rows <= 0 && !materialize) || rows < 0 || heads <= 0 || v_dim <= 0)
    throw std::invalid_argument(w + ": empty problem");
  if (replay.in != nullptr) {
    if (!kScalarGate) throw std::invalid_argument(w + ": the replay form is the scalar-gate recurrence's");
    const int64_t in_row = static_cast<int64_t>(2) * (heads / std::max(kv_ratio, 1)) * k_dim +
                           static_cast<int64_t>(heads) * v_dim + 2 * static_cast<int64_t>(heads);
    if (replay.in_stride < in_row || (replay.in_stride * 2) % 8 != 0)
      throw std::invalid_argument(w + ": replay row stride");
    if (replay.rows_cap <= 0 || (replay.checkpoint && rows > replay.rows_cap))
      throw std::invalid_argument(w + ": replay rows_cap");
    if (replay.checkpoint && replay.save == nullptr)
      throw std::invalid_argument(w + ": the checkpoint form needs a save buffer");
    if (replay.checkpoint && replay.save == replay.in)
      throw std::invalid_argument(w + ": the replay's in and save buffers must differ");
    if (replay.checkpoint && snap.states != nullptr)
      throw std::invalid_argument(w + ": the checkpoint form takes no snapshots");
    if (replay.materialize_rows > replay.rows_cap)
      throw std::invalid_argument(w + ": materialize rows past the saved rows");
    if (materialize && replay.materialize_rows < 0 && replay.count == nullptr)
      throw std::invalid_argument(w + ": materialize needs a count or a row number");
    if (materialize && replay.checkpoint)
      throw std::invalid_argument(w + ": the materialize form runs no rows of its own");
  }
  if (k_dim % kRecurrentLanes != 0)
    throw std::invalid_argument(w + ": k_dim must be a multiple of 16");
  if (kv_ratio <= 0 || heads % kv_ratio != 0)
    throw std::invalid_argument(w + ": heads must divide by kv_ratio");
  if (beta_row_stride < heads)
    throw std::invalid_argument(w + ": beta stride smaller than row");
  const int64_t g_row = kScalarGate ? static_cast<int64_t>(heads)
                                    : static_cast<int64_t>(heads) * k_dim;
  if (g_row_stride < g_row)
    throw std::invalid_argument(w + ": gate stride smaller than row");
  const int64_t state_elems = static_cast<int64_t>(heads) * v_dim * k_dim;
  if (batched) {
    if (request_state_stride < state_elems)
      throw std::invalid_argument(w + ": request stride smaller than a state");
    if (!requests.request_ids || !requests.positions || !requests.spans)
      throw std::invalid_argument(w + ": incomplete request map");
    if (snap.states && snap.stride_elems < state_elems)
      throw std::invalid_argument(w + ": snapshot stride smaller than a state");
  } else if (snap.states && rows > 1 && snap.stride_elems < state_elems) {
    throw std::invalid_argument(w + ": snapshot stride smaller than a state");
  }

  const dim3 grid(static_cast<unsigned>((v_dim + kRecurrentRows - 1) /
                                        kRecurrentRows),
                  static_cast<unsigned>(heads),
                  static_cast<unsigned>(batched ? requests.num_requests : 1));
  const uint16_t* qkv16 = static_cast<const uint16_t*>(qkv);
  const uint16_t* g16 = static_cast<const uint16_t*>(g_raw);
  const uint16_t* b16 = static_cast<const uint16_t*>(beta_raw);
  uint16_t* o16 = static_cast<uint16_t*>(out);

  // Vector slices need 16-byte-aligned bases and strides: the fused qkv
  // row (2*heads_k*K + heads*v_dim bf16), the per-head K offsets, and the
  // per-lane column offsets. Per-lane slices are K/16 wide: float4 state
  // slices need K >= 64 (4 columns, 16 bytes); the bf16 slices then are
  // 8-byte uint2s. The scalar gate reads its bias per head, unvectorized.
  const auto aligned16 = [](const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
  };
  const int heads_k = heads / kv_ratio;
  const int64_t qkv_stride = static_cast<int64_t>(2) * heads_k * k_dim +
                             static_cast<int64_t>(heads) * v_dim;
  const int64_t state_stride = batched ? request_state_stride : state_elems;
  const bool vec =
      k_dim >= 64 && aligned16(qkv16) && aligned16(states) &&
      (kScalarGate || (aligned16(g16) && aligned16(dt_bias) &&
                       (g_row_stride * 2) % 16 == 0)) &&
      (qkv_stride * 2) % 16 == 0 &&
      (static_cast<int64_t>(v_dim) * k_dim * 4) % 16 == 0 &&
      (!batched || (request_state_stride * 4) % 16 == 0) &&
      (!snap.states || (aligned16(snap.states) &&
                        (snap.stride_elems * 4) % 16 == 0));

  const auto launch = [&](auto kdim_tag, auto vec_tag, auto batch_tag) {
    constexpr int KLIT = decltype(kdim_tag)::value;
    constexpr bool kVec = decltype(vec_tag)::value;
    constexpr bool kBatched = decltype(batch_tag)::value;
    kda_recurrent_kernel<KLIT, kVec, kBatched, kScalarGate>
        <<<grid, kRecurrentBlock, 0, stream>>>(
            qkv16, g16, g_row_stride, b16, beta_row_stride, a_log, dt_bias,
            states, state_stride, o16, rows, heads, kv_ratio, v_dim,
            lower_bound, scale, snap.states, snap.stride_elems,
            requests.request_ids, requests.positions, requests.spans, replay);
    DGPP_CUDA_OK(cudaGetLastError());
  };
  const auto dispatch_k = [&](auto kdim_tag) {
    if (vec) {
      if (batched) launch(kdim_tag, std::true_type{}, std::true_type{});
      else launch(kdim_tag, std::true_type{}, std::false_type{});
    } else {
      if (batched) launch(kdim_tag, std::false_type{}, std::true_type{});
      else launch(kdim_tag, std::false_type{}, std::false_type{});
    }
  };
  switch (k_dim) {
    case 32: dispatch_k(std::integral_constant<int, 32>{}); return;
    case 64: dispatch_k(std::integral_constant<int, 64>{}); return;
    case 128: dispatch_k(std::integral_constant<int, 128>{}); return;
    default:
      throw std::invalid_argument(w + ": k_dim must be one of {32, 64, 128}");
  }
}

}  // namespace

void kda_recurrent_fwd(const void* qkv, const void* g_raw, const void* beta_raw,
                       int64_t beta_row_stride, const float* a_log,
                       const float* dt_bias, float* state, void* out,
                       int tokens, int heads, int k_dim, int v_dim,
                       float lower_bound, float scale, cudaStream_t stream,
                       const KdaStateSnapshots& snap) {
  recurrent_launch<false>("kda recurrent", qkv, g_raw,
                          static_cast<int64_t>(heads) * k_dim, beta_raw,
                          beta_row_stride, a_log, dt_bias, state, 0, out,
                          tokens, heads, 1, k_dim, v_dim, lower_bound, scale,
                          KdaRequestRows{}, stream, snap, KdaReplay{});
}

void kda_recurrent_fwd_batched(
    const void* qkv, const void* g_raw, const void* beta_raw,
    int64_t beta_row_stride, const float* a_log, const float* dt_bias,
    float* states, int64_t request_state_stride, void* out, int rows,
    int heads, int k_dim, int v_dim, float lower_bound, float scale,
    const KdaRequestRows& requests, cudaStream_t stream,
    const KdaStateSnapshots& snap) {
  if (requests.num_requests <= 0)
    throw std::invalid_argument("kda batched recurrent: incomplete request map");
  recurrent_launch<false>("kda batched recurrent", qkv, g_raw,
                          static_cast<int64_t>(heads) * k_dim, beta_raw,
                          beta_row_stride, a_log, dt_bias, states,
                          request_state_stride, out, rows, heads, 1, k_dim,
                          v_dim, lower_bound, scale, requests, stream, snap, KdaReplay{});
}

void gdn_recurrent_fwd(const void* qkv, const void* a_raw, int64_t a_row_stride,
                       const void* beta_raw, int64_t beta_row_stride,
                       const float* a_log, const float* dt_bias, float* state,
                       void* out, int tokens, int heads, int kv_ratio,
                       int k_dim, int v_dim, float scale, cudaStream_t stream,
                       const KdaStateSnapshots& snap, const KdaReplay& replay) {
  recurrent_launch<true>("gdn recurrent", qkv, a_raw, a_row_stride, beta_raw,
                         beta_row_stride, a_log, dt_bias, state, 0, out,
                         tokens, heads, kv_ratio, k_dim, v_dim, 0.0f, scale,
                         KdaRequestRows{}, stream, snap, replay);
}

void gdn_recurrent_fwd_batched(
    const void* qkv, const void* a_raw, int64_t a_row_stride,
    const void* beta_raw, int64_t beta_row_stride, const float* a_log,
    const float* dt_bias, float* states, int64_t request_state_stride,
    void* out, int rows, int heads, int kv_ratio, int k_dim, int v_dim,
    float scale, const KdaRequestRows& requests, cudaStream_t stream,
    const KdaStateSnapshots& snap, const KdaReplay& replay) {
  if (requests.num_requests <= 0)
    throw std::invalid_argument("gdn batched recurrent: incomplete request map");
  recurrent_launch<true>("gdn batched recurrent", qkv, a_raw, a_row_stride,
                         beta_raw, beta_row_stride, a_log, dt_bias, states,
                         request_state_stride, out, rows, heads, kv_ratio,
                         k_dim, v_dim, 0.0f, scale, requests, stream, snap, replay);
}

}  // namespace dgpp
