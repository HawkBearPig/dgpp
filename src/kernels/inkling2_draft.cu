#include "kernels/inkling2_draft.hpp"

#include <cmath>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/pdl.cuh"
#include "kernels/sample_pick.hpp"

namespace dgpp {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float round_bf16(float v) { return bf16_bits_to_float(float_to_bf16_bits(v)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(~0u, v, off);
  return v;
}

// The block's sum of every thread's `v`: the warp trees, then the warps in
// order. `slots` holds one float per warp. Returns the sum to every thread.
__device__ __forceinline__ float block_sum(float v, float* slots) {
  v = warp_sum(v);
  if ((threadIdx.x & 31) == 0) slots[threadIdx.x >> 5] = v;
  __syncthreads();
  float total = 0.0f;
  const int warps = (blockDim.x + 31) >> 5;
  for (int w = 0; w < warps; ++w) total += slots[w];
  __syncthreads();
  return total;
}

int flat_blocks(int64_t total) {
  int64_t blocks = (total + kThreads - 1) / kThreads;
  if (blocks > 65535) blocks = 65535;
  return static_cast<int>(blocks < 1 ? 1 : blocks);
}

// ---- the feature mix ------------------------------------------------------------
struct MixDev {
  const uint16_t* tap[kInk2DraftMaxTaps];
  const float* alpha;
  const uint16_t* w;
  uint16_t* out;
  int taps;
  int hidden;
  float eps;
};

__global__ void __launch_bounds__(kThreads) draft_mix_norm_kernel(MixDev a) {
  dgpp_pdl_wait();
  extern __shared__ float mixed[];  // [hidden]
  __shared__ float slots[8];
  const int row = static_cast<int>(blockIdx.x), tid = static_cast<int>(threadIdx.x);
  const int64_t off = static_cast<int64_t>(row) * a.hidden;
  float alpha[kInk2DraftMaxTaps];
  for (int t = 0; t < a.taps; ++t) alpha[t] = a.alpha[t];
  float ss = 0.0f;
  for (int d = tid; d < a.hidden; d += kThreads) {
    float m = 0.0f;
    for (int t = 0; t < a.taps; ++t) m = fmaf(bf16_bits_to_float(a.tap[t][off + d]), alpha[t], m);
    mixed[d] = m;
    ss = fmaf(m, m, ss);
  }
  const float rstd = 1.0f / sqrtf(block_sum(ss, slots) / static_cast<float>(a.hidden) + a.eps);
  for (int d = tid; d < a.hidden; d += kThreads)
    a.out[off + d] = float_to_bf16_bits((mixed[d] * rstd) * bf16_bits_to_float(a.w[d]));
}

// ---- the block's rows ---------------------------------------------------------------
__global__ void __launch_bounds__(kThreads) draft_mask_rows_kernel(const uint16_t* __restrict__ mask,
                                                                   uint16_t* __restrict__ out, int rows,
                                                                   int block_rows, int hidden) {
  dgpp_pdl_wait();
  const int64_t total = static_cast<int64_t>(rows) * hidden;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; i < total; i += stride) {
    const int row = static_cast<int>(i / hidden);
    if (row % block_rows == 0) continue;
    out[i] = mask[i - static_cast<int64_t>(row) * hidden];
  }
}

__global__ void __launch_bounds__(kThreads) draft_add_norm_kernel(uint16_t* __restrict__ residual,
                                                                  const uint16_t* __restrict__ add,
                                                                  const uint16_t* __restrict__ w,
                                                                  uint16_t* __restrict__ out, int hidden, float eps) {
  dgpp_pdl_wait();
  extern __shared__ float sum[];  // [hidden]
  __shared__ float slots[8];
  const int row = static_cast<int>(blockIdx.x), tid = static_cast<int>(threadIdx.x);
  const int64_t off = static_cast<int64_t>(row) * hidden;
  float ss = 0.0f;
  for (int d = tid; d < hidden; d += kThreads) {
    const float s = bf16_bits_to_float(residual[off + d]) + bf16_bits_to_float(add[off + d]);
    sum[d] = s;
    ss = fmaf(s, s, ss);
  }
  const float rstd = 1.0f / sqrtf(block_sum(ss, slots) / static_cast<float>(hidden) + eps);
  for (int d = tid; d < hidden; d += kThreads) {
    residual[off + d] = float_to_bf16_bits(sum[d]);
    out[off + d] = float_to_bf16_bits((sum[d] * rstd) * bf16_bits_to_float(w[d]));
  }
}

// Grid (heads, rows), 64 threads: thread i owns the pair (i, i + 64).
__global__ void __launch_bounds__(64) draft_norm_rope_kernel(const uint16_t* __restrict__ x, int64_t x_stride,
                                                             const uint16_t* __restrict__ w,
                                                             const int64_t* __restrict__ pos,
                                                             const float* __restrict__ inv_freq, uint16_t* out,
                                                             int64_t out_stride, float eps) {
  dgpp_pdl_wait();
  __shared__ float slots[2];
  const int head = static_cast<int>(blockIdx.x), row = static_cast<int>(blockIdx.y);
  const int i = static_cast<int>(threadIdx.x);
  const int64_t p = pos[row];
  const uint16_t* in = x + static_cast<int64_t>(row) * x_stride + static_cast<int64_t>(head) * 128;
  const float a = bf16_bits_to_float(in[i]), b = bf16_bits_to_float(in[i + 64]);
  const float ss = block_sum(fmaf(a, a, b * b), slots);
  if (p < 0) return;
  const float rstd = 1.0f / sqrtf(ss / 128.0f + eps);
  const float n1 = round_bf16((a * rstd) * bf16_bits_to_float(w[i]));
  const float n2 = round_bf16((b * rstd) * bf16_bits_to_float(w[i + 64]));
  const float angle = static_cast<float>(p) * inv_freq[i];
  const float c = cosf(angle), s = sinf(angle);
  uint16_t* o = out + static_cast<int64_t>(row) * out_stride + static_cast<int64_t>(head) * 128;
  o[i] = float_to_bf16_bits(n1 * c - n2 * s);
  o[i + 64] = float_to_bf16_bits(n2 * c + n1 * s);
}

__global__ void __launch_bounds__(kThreads) draft_ring_row_kernel(const uint16_t* __restrict__ ring_request,
                                                                  int64_t slot_stride, int64_t offset, int ring_slots,
                                                                  const int64_t* __restrict__ block_pos,
                                                                  uint16_t* __restrict__ out, int width) {
  dgpp_pdl_wait();
  const int64_t p = block_pos[0] - 1;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; i < width; i += stride)
    out[i] = p >= 0 ? ring_request[(p % ring_slots) * slot_stride + offset + i] : static_cast<uint16_t>(0);
}

// ---- the recurrent head ---------------------------------------------------------------
// Grid (3 rank, slots * steps): one output per block.
__global__ void __launch_bounds__(kThreads) draft_markov_context_kernel(
    const uint16_t* __restrict__ hidden, int block_rows, int row_offset, const uint16_t* __restrict__ feature,
    const uint16_t* __restrict__ rnn_w, const float* __restrict__ rnn_bias, float* __restrict__ context, int steps,
    int hidden_size, int rank) {
  dgpp_pdl_wait();
  __shared__ float slots[8];
  const int j = static_cast<int>(blockIdx.x), row = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int s = row / steps, i = row - s * steps;
  const uint16_t* h = hidden + (static_cast<int64_t>(s) * block_rows + row_offset + i) * hidden_size;
  const uint16_t* f = feature + static_cast<int64_t>(s) * hidden_size;
  const uint16_t* w = rnn_w + static_cast<int64_t>(j) * (2 * rank + 2 * hidden_size) + 2 * rank;
  float acc = 0.0f;
  for (int d = tid; d < hidden_size; d += kThreads) {
    acc = fmaf(bf16_bits_to_float(h[d]), bf16_bits_to_float(w[d]), acc);
    acc = fmaf(bf16_bits_to_float(f[d]), bf16_bits_to_float(w[hidden_size + d]), acc);
  }
  const float total = block_sum(acc, slots);
  if (tid == 0) context[static_cast<int64_t>(row) * 3 * rank + j] = rnn_bias[j] + total;
}

// The step's projection, one block per output and slot: grid (3 rank, slots),
// 64 threads over the 2 rank inputs [state | previous token's row].
__global__ void __launch_bounds__(64) draft_markov_raw_kernel(
    const float* __restrict__ context, int steps, int step, const uint16_t* __restrict__ rnn_w, int64_t w_stride,
    const uint16_t* __restrict__ w1, const int64_t* __restrict__ prev, int prev_stride,
    const uint16_t* __restrict__ state, float* __restrict__ raw, int rank, const int32_t* __restrict__ stop) {
  dgpp_pdl_wait();
  if (*stop != 0) return;
  __shared__ float slots[2];
  const int j = static_cast<int>(blockIdx.x), s = static_cast<int>(blockIdx.y);
  const int64_t token = prev[static_cast<int64_t>(s) * prev_stride];
  const uint16_t* e = w1 + (token < 0 ? 0 : token) * rank;
  const uint16_t* st = state + static_cast<int64_t>(s) * rank;
  const uint16_t* w = rnn_w + static_cast<int64_t>(j) * w_stride;
  float acc = 0.0f;
  for (int k = static_cast<int>(threadIdx.x); k < 2 * rank; k += 64) {
    const float in = k < rank ? (step == 0 ? 0.0f : bf16_bits_to_float(st[k])) : bf16_bits_to_float(e[k - rank]);
    acc = fmaf(in, bf16_bits_to_float(w[k]), acc);
  }
  const float total = block_sum(acc, slots);
  if (threadIdx.x == 0)
    raw[static_cast<int64_t>(s) * 3 * rank + j] =
        round_bf16(context[(static_cast<int64_t>(s) * steps + step) * 3 * rank + j] + total);
}

// The gates, one block per slot.
__global__ void __launch_bounds__(kThreads) draft_markov_gate_kernel(const float* __restrict__ raw, int step,
                                                                     uint16_t* __restrict__ state,
                                                                     uint16_t* __restrict__ latent, int rank,
                                                                     const int32_t* __restrict__ stop) {
  dgpp_pdl_wait();
  if (*stop != 0) return;
  const int s = static_cast<int>(blockIdx.x);
  const float* r = raw + static_cast<int64_t>(s) * 3 * rank;
  uint16_t* st = state + static_cast<int64_t>(s) * rank;
  for (int c = static_cast<int>(threadIdx.x); c < rank; c += kThreads) {
    const float before = step == 0 ? 0.0f : bf16_bits_to_float(st[c]);
    const float gate = round_bf16(__fdiv_rn(1.0f, 1.0f + expf(-r[c])));
    const float cand = round_bf16(tanhf(r[rank + c]));
    const float keep = round_bf16(gate * before);
    const float update = round_bf16(round_bf16(1.0f - gate) * cand);
    st[c] = float_to_bf16_bits(keep + update);
    latent[static_cast<int64_t>(s) * rank + c] = float_to_bf16_bits(tanhf(r[2 * rank + c]));
  }
}

__global__ void __launch_bounds__(kThreads) draft_markov_logits_kernel(const float* __restrict__ base,
                                                                       int64_t base_slot_stride, int64_t vocab,
                                                                       const uint16_t* __restrict__ fold_b, float cap,
                                                                       float* __restrict__ logits, int slots,
                                                                       const int32_t* __restrict__ stop) {
  dgpp_pdl_wait();
  if (*stop != 0) return;
  const int64_t total = static_cast<int64_t>(slots) * vocab;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; i < total; i += stride) {
    const int64_t s = i / vocab, v = i - s * vocab;
    const float bias = round_bf16(bf16_bits_to_float(fold_b[v]) + logits[i]);
    float x = round_bf16(round_bf16(base[s * base_slot_stride + v]) + bias);
    if (cap > 0.0f) x = round_bf16(cap * tanhf(x / cap));
    logits[i] = x;
  }
}

__global__ void draft_confidence_kernel(const float* __restrict__ scores, const int32_t* __restrict__ ids,
                                        const int32_t* __restrict__ token, int k, float bias, float slope,
                                        float* __restrict__ conf, const int32_t* __restrict__ stop,
                                        float* __restrict__ survival, const SampleSpec* __restrict__ spec) {
  dgpp_pdl_wait();
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  if (*stop != 0) {
    // The step did not run: its proposal is not one, and no schedule may
    // verify it.
    conf[0] = -30.0f;
    return;
  }
  float top = -INFINITY;
  for (int i = 0; i < k; ++i) top = fmaxf(top, scores[i]);
  float mass = 0.0f, chosen = 0.0f;
  bool found = false;
  for (int i = 0; i < k; ++i) {
    const float e = expf(scores[i] - top);
    mass += e;
    if (!found && ids[i] == token[0]) {
      chosen = e;
      found = true;
    }
  }
  float logit = -30.0f;
  if (found && mass > 0.0f) {
    // The others' mass directly: 1 - p cancels when p rounds to one.
    const float rest = (mass - chosen) / mass;
    const float own = rest <= 0.0f ? 30.0f : fminf(fmaxf(logf(chosen / mass) - logf(rest), -30.0f), 30.0f);
    logit = bias + slope * own;
  }
  logit = fminf(fmaxf(logit, -30.0f), 30.0f);
  conf[0] = logit;
  // The chance every proposal so far is accepted. A sampled request's
  // proposals are verified whatever their estimate, so it never counts
  // toward stopping.
  // toward stopping: its survival is held far above any floor, so a pass
  // with a sampled request in it runs every step.
  const bool sampled = spec != nullptr && spec->temperature > 0.0f;
  survival[0] = sampled ? 1e30f : survival[0] * __fdiv_rn(1.0f, 1.0f + expf(-logit));
}

__global__ void draft_stop_reset_kernel(int32_t* __restrict__ stop, float* __restrict__ survival, int slots) {
  dgpp_pdl_wait();
  if (blockIdx.x != 0) return;
  const int t = static_cast<int>(threadIdx.x);
  if (t == 0) stop[0] = 0;
  if (t < slots) survival[t] = 1.0f;
}

__global__ void draft_stop_kernel(const float* __restrict__ survival, int slots, const float* __restrict__ floor,
                                  int32_t* __restrict__ stop) {
  dgpp_pdl_wait();
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  float sum = 0.0f;
  for (int s = 0; s < slots; ++s) sum += survival[s];
  if (sum < floor[0] * static_cast<float>(slots)) stop[0] = 1;
}

// One block per vocabulary row, one thread per output column.
__global__ void __launch_bounds__(kThreads) draft_fold_head_kernel(const uint16_t* __restrict__ head,
                                                                   const uint16_t* __restrict__ out_w,
                                                                   const uint16_t* __restrict__ out_b,
                                                                   uint16_t* __restrict__ fold_w,
                                                                   uint16_t* __restrict__ fold_b, int hidden,
                                                                   int rank) {
  dgpp_pdl_wait();
  extern __shared__ float row[];  // [hidden]
  const int64_t v = static_cast<int64_t>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const uint16_t* h = head + v * hidden;
  for (int d = tid; d < hidden; d += kThreads) row[d] = bf16_bits_to_float(h[d]);
  __syncthreads();
  for (int j = tid; j < rank; j += kThreads) {
    double acc = 0.0;
    for (int d = 0; d < hidden; ++d)
      acc = fma(static_cast<double>(row[d]), static_cast<double>(bf16_bits_to_float(out_w[static_cast<int64_t>(d) * rank + j])),
                acc);
    fold_w[v * rank + j] = float_to_bf16_bits(static_cast<float>(acc));
  }
  if (tid == 0) {
    double acc = 0.0;
    for (int d = 0; d < hidden; ++d)
      acc = fma(static_cast<double>(row[d]), static_cast<double>(bf16_bits_to_float(out_b[d])), acc);
    fold_b[v] = float_to_bf16_bits(static_cast<float>(acc));
  }
}

}  // namespace

void ink2_draft_mix_norm(const Ink2DraftMixArgs& args, cudaStream_t stream) {
  if (args.rows <= 0) return;
  if (args.taps < 1 || args.taps > kInk2DraftMaxTaps) throw std::invalid_argument("ink2_draft_mix_norm: taps");
  if (!args.alpha || !args.w || !args.out || args.hidden <= 0)
    throw std::invalid_argument("ink2_draft_mix_norm: null buffer or width");
  MixDev d{};
  for (int t = 0; t < args.taps; ++t) {
    if (!args.tap[t]) throw std::invalid_argument("ink2_draft_mix_norm: null tap");
    d.tap[t] = args.tap[t];
  }
  d.alpha = args.alpha;
  d.w = args.w;
  d.out = args.out;
  d.taps = args.taps;
  d.hidden = args.hidden;
  d.eps = args.eps;
  draft_mix_norm_kernel<<<static_cast<unsigned>(args.rows), kThreads, static_cast<size_t>(args.hidden) * sizeof(float),
                          stream>>>(d);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_draft_mask_rows(const uint16_t* mask, uint16_t* out, int rows, int block_rows, int hidden,
                          cudaStream_t stream) {
  if (rows <= 0) return;
  if (!mask || !out || block_rows < 1 || rows % block_rows != 0 || hidden <= 0)
    throw std::invalid_argument("ink2_draft_mask_rows: whole blocks of rows and their buffers");
  draft_mask_rows_kernel<<<flat_blocks(static_cast<int64_t>(rows) * hidden), kThreads, 0, stream>>>(mask, out, rows,
                                                                                                 block_rows, hidden);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_draft_add_norm(uint16_t* residual, const uint16_t* add, const uint16_t* w, uint16_t* out, int rows,
                         int hidden, float eps, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!residual || !add || !w || !out || hidden <= 0) throw std::invalid_argument("ink2_draft_add_norm: null buffer");
  draft_add_norm_kernel<<<static_cast<unsigned>(rows), kThreads, static_cast<size_t>(hidden) * sizeof(float), stream>>>(
      residual, add, w, out, hidden, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_draft_norm_rope(const uint16_t* x, int64_t x_stride, const uint16_t* w, const int64_t* pos,
                          const float* inv_freq, uint16_t* out, int64_t out_stride, int rows, int heads, float eps,
                          cudaStream_t stream) {
  if (rows <= 0 || heads <= 0) return;
  if (!x || !w || !pos || !inv_freq || !out) throw std::invalid_argument("ink2_draft_norm_rope: null buffer");
  if (x_stride < static_cast<int64_t>(heads) * 128 || out_stride < static_cast<int64_t>(heads) * 128)
    throw std::invalid_argument("ink2_draft_norm_rope: the heads do not fit the row strides");
  const dim3 grid(static_cast<unsigned>(heads), static_cast<unsigned>(rows));
  draft_norm_rope_kernel<<<grid, 64, 0, stream>>>(x, x_stride, w, pos, inv_freq, out, out_stride, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_draft_ring_row(const uint16_t* ring_request, int64_t slot_stride, int64_t offset, int ring_slots,
                         const int64_t* block_pos, uint16_t* out, int width, cudaStream_t stream) {
  if (width <= 0) return;
  if (!ring_request || !block_pos || !out || ring_slots <= 0 || offset < 0 || offset + width > slot_stride)
    throw std::invalid_argument("ink2_draft_ring_row: the row lies outside a ring slot");
  draft_ring_row_kernel<<<flat_blocks(width), kThreads, 0, stream>>>(ring_request, slot_stride, offset, ring_slots,
                                                                     block_pos, out, width);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_draft_markov_context(const uint16_t* hidden, int block_rows, int row_offset, const uint16_t* feature,
                               const uint16_t* rnn_w, const float* rnn_bias, float* context, int slots, int steps,
                               int hidden_size, int rank, cudaStream_t stream) {
  if (slots <= 0 || steps <= 0) return;
  if (!hidden || !feature || !rnn_w || !rnn_bias || !context)
    throw std::invalid_argument("ink2_draft_markov_context: null buffer");
  if (rank < 1 || rank > kInk2DraftMaxRank || hidden_size <= 0 || row_offset < 0 || row_offset + steps > block_rows)
    throw std::invalid_argument("ink2_draft_markov_context: the proposal rows lie outside the block");
  const dim3 grid(static_cast<unsigned>(3 * rank), static_cast<unsigned>(slots * steps));
  draft_markov_context_kernel<<<grid, kThreads, 0, stream>>>(hidden, block_rows, row_offset, feature, rnn_w, rnn_bias,
                                                             context, steps, hidden_size, rank);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_draft_markov_step(const float* context, int steps, int step, const uint16_t* rnn_w, int hidden_size,
                            const uint16_t* w1, const int64_t* prev, int prev_stride, uint16_t* state,
                            uint16_t* latent, float* raw, const int32_t* stop, int slots, int rank,
                            cudaStream_t stream) {
  if (slots <= 0) return;
  if (!context || !rnn_w || !w1 || !prev || !state || !latent || !raw || !stop)
    throw std::invalid_argument("ink2_draft_markov_step: null buffer");
  if (rank < 1 || rank > kInk2DraftMaxRank || step < 0 || step >= steps || prev_stride < 1)
    throw std::invalid_argument("ink2_draft_markov_step: rank, step or stride");
  const dim3 grid(static_cast<unsigned>(3 * rank), static_cast<unsigned>(slots));
  draft_markov_raw_kernel<<<grid, 64, 0, stream>>>(context, steps, step, rnn_w,
                                                   static_cast<int64_t>(2 * rank + 2 * hidden_size), w1, prev,
                                                   prev_stride, state, raw, rank, stop);
  DGPP_CUDA_OK(cudaGetLastError());
  draft_markov_gate_kernel<<<static_cast<unsigned>(slots), kThreads, 0, stream>>>(raw, step, state, latent, rank,
                                                                                 stop);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_draft_markov_logits(const float* base, int64_t base_slot_stride, int64_t vocab, const uint16_t* fold_b,
                              float cap, float* logits, const int32_t* stop, int slots, cudaStream_t stream) {
  if (slots <= 0 || vocab <= 0) return;
  if (!base || !fold_b || !logits || !stop) throw std::invalid_argument("ink2_draft_markov_logits: null buffer");
  draft_markov_logits_kernel<<<flat_blocks(static_cast<int64_t>(slots) * vocab), kThreads, 0, stream>>>(
      base, base_slot_stride, vocab, fold_b, cap, logits, slots, stop);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_draft_confidence(const float* scores, const int32_t* ids, const int32_t* token, int k,
                           const Ink2DraftConfidenceFit& fit, float* conf, const int32_t* stop, float* survival,
                           const SampleSpec* spec, cudaStream_t stream) {
  if (!scores || !ids || !token || !conf || !stop || !survival || k < 1)
    throw std::invalid_argument("ink2_draft_confidence: arguments");
  draft_confidence_kernel<<<1, 32, 0, stream>>>(scores, ids, token, k, fit.bias, fit.slope, conf, stop, survival,
                                                spec);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_draft_stop_reset(int32_t* stop, float* survival, int slots, cudaStream_t stream) {
  if (!stop || !survival || slots < 1 || slots > 32) throw std::invalid_argument("ink2_draft_stop_reset: arguments");
  draft_stop_reset_kernel<<<1, 32, 0, stream>>>(stop, survival, slots);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_draft_stop(const float* survival, int slots, const float* floor, int32_t* stop, cudaStream_t stream) {
  if (!survival || !floor || !stop || slots < 1) throw std::invalid_argument("ink2_draft_stop: arguments");
  draft_stop_kernel<<<1, 32, 0, stream>>>(survival, slots, floor, stop);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_draft_fold_head(const uint16_t* head, const uint16_t* out_w, const uint16_t* out_b, uint16_t* fold_w,
                          uint16_t* fold_b, int64_t vocab, int hidden, int rank, cudaStream_t stream) {
  if (vocab <= 0) return;
  if (!head || !out_w || !out_b || !fold_w || !fold_b) throw std::invalid_argument("ink2_draft_fold_head: null buffer");
  if (hidden <= 0 || rank < 1 || rank > kInk2DraftMaxRank)
    throw std::invalid_argument("ink2_draft_fold_head: width or rank");
  // Bounded launches: a grid of every vocabulary row at once would hold the
  // device for the whole fold.
  const int64_t chunk = 16384;
  for (int64_t v0 = 0; v0 < vocab; v0 += chunk) {
    const int64_t n = vocab - v0 < chunk ? vocab - v0 : chunk;
    draft_fold_head_kernel<<<static_cast<unsigned>(n), kThreads, static_cast<size_t>(hidden) * sizeof(float),
                             stream>>>(head + v0 * hidden, out_w, out_b, fold_w + v0 * rank, fold_b + v0, hidden,
                                       rank);
    DGPP_CUDA_OK(cudaGetLastError());
  }
}

}  // namespace dgpp
