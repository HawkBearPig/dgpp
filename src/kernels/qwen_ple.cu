#include "kernels/latent_format.hpp"
#include "kernels/pdl.cuh"
#include "kernels/qwen_ple.hpp"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {

__device__ __forceinline__ float round_bf16(float v) {
  return bf16_bits_to_float(float_to_bf16_bits(v));
}
__device__ __forceinline__ float sigmoid_f(float v) { return 1.0f / (1.0f + expf(-v)); }

constexpr int kMaxHeads = 64;

// One thread per (token, head).
__global__ void hash_ids_kernel(const int32_t* __restrict__ tokens, int n, int32_t ctx_prev1,
                                int32_t ctx_prev2, int32_t eos,
                                const int64_t* __restrict__ multipliers,
                                const int64_t* __restrict__ head_vocab,
                                const int64_t* __restrict__ head_offset, int heads,
                                int heads_per_ngram, int32_t* __restrict__ ids) {
  dgpp_pdl_wait();
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<int64_t>(n) * heads) return;
  const int t = static_cast<int>(i / heads);
  const int h = static_cast<int>(i - static_cast<int64_t>(t) * heads);
  const int64_t y0 = tokens[t];
  const int64_t prev1 = t >= 1 ? tokens[t - 1] : ctx_prev1;
  const int64_t prev2 = t >= 2 ? tokens[t - 2] : (t == 1 ? ctx_prev1 : ctx_prev2);
  const int64_t y1 = prev1;
  const int64_t y2 = prev1 == eos ? eos : prev2;
  // int64 products under torch's wrapping multiply (they stay below 2^63
  // for this vocabulary; the unsigned form makes the wrap defined).
  const uint64_t m0 = static_cast<uint64_t>(y0) * static_cast<uint64_t>(multipliers[0]);
  const uint64_t m1 = static_cast<uint64_t>(y1) * static_cast<uint64_t>(multipliers[1]);
  const uint64_t m2 = static_cast<uint64_t>(y2) * static_cast<uint64_t>(multipliers[2]);
  uint64_t mix = m0 ^ m1;
  if (h >= heads_per_ngram) mix ^= m2;
  // torch.remainder: the sign of the divisor (positive); mix is
  // non-negative here (xor of non-negative int64 values).
  const int64_t mixed = static_cast<int64_t>(mix);
  int64_t r = mixed % head_vocab[h];
  if (r < 0) r += head_vocab[h];
  ids[i] = static_cast<int32_t>(r + head_offset[h]);
}

// The span of row t: its request's first row in the batch (the row map's
// contract: <= kDecodeRows spans, so a scan is the lookup).
__device__ __forceinline__ int span_start_of(int t, const int32_t* __restrict__ spans,
                                             int num_requests) {
  for (int q = 0; q < num_requests; ++q) {
    const int s = spans[2 * q], len = spans[2 * q + 1];
    if (t >= s && t < s + len) return s;
  }
  return t;  // a row outside every span: its own start (nothing before it)
}

__device__ __forceinline__ int32_t hash_one(int64_t y0, int64_t y1, int64_t y2, int h,
                                            int heads_per_ngram,
                                            const int64_t* __restrict__ multipliers,
                                            const int64_t* __restrict__ head_vocab,
                                            const int64_t* __restrict__ head_offset) {
  const uint64_t m0 = static_cast<uint64_t>(y0) * static_cast<uint64_t>(multipliers[0]);
  const uint64_t m1 = static_cast<uint64_t>(y1) * static_cast<uint64_t>(multipliers[1]);
  const uint64_t m2 = static_cast<uint64_t>(y2) * static_cast<uint64_t>(multipliers[2]);
  uint64_t mix = m0 ^ m1;
  if (h >= heads_per_ngram) mix ^= m2;
  const int64_t mixed = static_cast<int64_t>(mix);
  int64_t r = mixed % head_vocab[h];
  if (r < 0) r += head_vocab[h];
  return static_cast<int32_t>(r + head_offset[h]);
}

// One thread per (row, head); the context off the request's stored ids.
__global__ void hash_ids_rows_kernel(const int64_t* __restrict__ tokens, int rows,
                                     const int32_t* __restrict__ req_ids,
                                     const int32_t* __restrict__ spans, int num_requests,
                                     const int32_t* __restrict__ ctx, int32_t eos,
                                     const int64_t* __restrict__ multipliers,
                                     const int64_t* __restrict__ head_vocab,
                                     const int64_t* __restrict__ head_offset, int heads,
                                     int heads_per_ngram, int32_t* __restrict__ ids) {
  dgpp_pdl_wait();
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<int64_t>(rows) * heads) return;
  const int t = static_cast<int>(i / heads);
  const int h = static_cast<int>(i - static_cast<int64_t>(t) * heads);
  const int s = span_start_of(t, spans, num_requests);
  const int p = t - s;
  if (req_ids[t] < 0) {
    ids[i] = hash_one(0, eos, eos, h, heads_per_ngram, multipliers, head_vocab, head_offset);
    return;
  }
  const int32_t* c = ctx + static_cast<int64_t>(req_ids[t]) * 4;
  const int64_t y0 = tokens[t];
  const int64_t prev1 = p >= 1 ? tokens[t - 1] : c[0];
  const int64_t prev2 = p >= 2 ? tokens[t - 2] : (p == 1 ? c[0] : c[1]);
  const int64_t y2 = prev1 == eos ? eos : prev2;
  ids[i] = hash_one(y0, prev1, y2, h, heads_per_ngram, multipliers, head_vocab, head_offset);
}

// One thread per span: the running context through its rows — every
// row's post-context into ctx_rows, the span's last into the request's
// context IN PLACE (the state after every row stood; a rollback copies an
// earlier row's back).
__global__ void context_rows_kernel(const int64_t* __restrict__ tokens,
                                    const int32_t* __restrict__ req_ids,
                                    const int64_t* __restrict__ pos,
                                    const int32_t* __restrict__ spans, int num_requests,
                                    int32_t* __restrict__ ctx, int32_t* __restrict__ ctx_rows) {
  dgpp_pdl_wait();
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_requests) return;
  const int s = spans[2 * q], len = spans[2 * q + 1];
  if (len <= 0) return;
  if (req_ids[s] < 0) {
    for (int t = s; t < s + len; ++t)
      for (int j = 0; j < 4; ++j) ctx_rows[static_cast<int64_t>(t) * 4 + j] = 0;
    return;
  }
  int32_t* c = ctx + static_cast<int64_t>(req_ids[s]) * 4;
  int32_t c0 = c[0], c1 = c[1];
  bool touched = false;
  for (int t = s; t < s + len; ++t) {
    if (pos[t] >= 0) {
      c1 = c0;
      c0 = static_cast<int32_t>(tokens[t]);
      touched = true;
    }
    ctx_rows[static_cast<int64_t>(t) * 4 + 0] = c0;
    ctx_rows[static_cast<int64_t>(t) * 4 + 1] = c1;
    ctx_rows[static_cast<int64_t>(t) * 4 + 2] = 0;
    ctx_rows[static_cast<int64_t>(t) * 4 + 3] = 0;
  }
  if (touched) {
    c[0] = c0;
    c[1] = c1;
  }
}

// One warp per (token, local head): the row's head_dim / 8 chunks of
// eight e4m3 codes, each lane one chunk -> eight bf16.
__global__ void gather_kernel(const uint8_t* __restrict__ table, int64_t row_begin,
                              int64_t rows, float scale, const int32_t* __restrict__ ids,
                              int n, int heads, int head_begin, int heads_local,
                              int head_dim, uint16_t* __restrict__ out) {
  dgpp_pdl_wait();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int64_t pair = static_cast<int64_t>(blockIdx.x) * (blockDim.x / 32) + warp;
  if (pair >= static_cast<int64_t>(n) * heads_local) return;
  const int t = static_cast<int>(pair / heads_local);
  const int hl = static_cast<int>(pair - static_cast<int64_t>(t) * heads_local);
  const int64_t id = ids[static_cast<int64_t>(t) * heads + head_begin + hl];
  const int64_t row = id - row_begin;
  if (row < 0 || row >= rows) __trap();  // the hash never leaves the head's range
  const uint8_t* src = table + row * head_dim;
  uint16_t* dst = out + (static_cast<int64_t>(t) * heads_local + hl) * head_dim;
  const int chunks = head_dim / 8;
  for (int c = lane; c < chunks; c += 32) {
    const uint2 codes = reinterpret_cast<const uint2*>(src)[c];
    const uint32_t cw[2] = {codes.x, codes.y};
    uint32_t packed[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t w = cw[j / 2] >> (16 * (j % 2));
      const uint16_t lo =
          float_to_bf16_bits(fp8_e4m3_bits_to_float(static_cast<uint8_t>(w & 0xFFu)) * scale);
      const uint16_t hi = float_to_bf16_bits(
          fp8_e4m3_bits_to_float(static_cast<uint8_t>((w >> 8) & 0xFFu)) * scale);
      packed[j] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
    }
    reinterpret_cast<uint4*>(dst)[c] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
  }
}

constexpr int kGateThreads = 256;

// One block per (token, branch): the branch's dot of the two normalized
// rows (rounded products, fp32 strided partials, a fixed tree), then the
// scalar chain on thread 0 and the broadcast multiply.
__global__ void gate_kernel(const uint16_t* __restrict__ key_n,
                            const uint16_t* __restrict__ query_n,
                            const uint16_t* __restrict__ value, uint16_t* __restrict__ gated,
                            int hc, int hidden, float inv_sqrt_h_divisor) {
  dgpp_pdl_wait();
  __shared__ float warp_sums[kGateThreads / 32];
  __shared__ float s_gate;
  const int64_t row = blockIdx.x;  // t * hc + i
  const int64_t t = row / hc;
  const int64_t base = row * hidden;
  float acc = 0.f;
  for (int d = threadIdx.x; d < hidden; d += kGateThreads)
    acc += round_bf16(bf16_bits_to_float(key_n[base + d]) * bf16_bits_to_float(query_n[base + d]));
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
  if (threadIdx.x % 32 == 0) warp_sums[threadIdx.x / 32] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = 0.f;
    for (int w = 0; w < kGateThreads / 32; ++w) s += warp_sums[w];
    float g = round_bf16(s);
    g = round_bf16(g / inv_sqrt_h_divisor);
    const float sign = g < 0.f ? -1.f : (g > 0.f ? 1.f : 0.f);
    g = round_bf16(sqrtf(fmaxf(fabsf(g), 1e-6f))) * sign;
    s_gate = round_bf16(sigmoid_f(g));
  }
  __syncthreads();
  const float gsig = s_gate;
  for (int d = threadIdx.x; d < hidden; d += kGateThreads)
    gated[base + d] = float_to_bf16_bits(gsig * bf16_bits_to_float(value[t * hidden + d]));
}

// One thread per channel, sequential over the tokens; the previous
// (W-1)*D inputs ride in registers (oldest first) and return to the state.
template <int W, int D>
__global__ void conv_kernel(const uint16_t* __restrict__ un, const uint16_t* __restrict__ gv,
                            uint16_t* __restrict__ state, const uint16_t* __restrict__ weight,
                            const uint16_t* __restrict__ residual, uint16_t* __restrict__ out,
                            int n, int channels) {
  dgpp_pdl_wait();
  constexpr int S = (W - 1) * D;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  float hist[S];
#pragma unroll
  for (int j = 0; j < S; ++j) hist[j] = bf16_bits_to_float(state[static_cast<int64_t>(c) * S + j]);
  float w[W];
#pragma unroll
  for (int k = 0; k < W; ++k) w[k] = bf16_bits_to_float(weight[static_cast<int64_t>(c) * W + k]);
  for (int t = 0; t < n; ++t) {
    const int64_t at = static_cast<int64_t>(t) * channels + c;
    const float u = bf16_bits_to_float(un[at]);
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < W - 1; ++k) acc = fmaf(w[k], hist[k * D], acc);
    acc = fmaf(w[W - 1], u, acc);
    const float cv = round_bf16(acc);
    const float act = round_bf16(cv * sigmoid_f(cv));
    const float ple = round_bf16(bf16_bits_to_float(gv[at]) + act);
    out[at] = float_to_bf16_bits(residual ? bf16_bits_to_float(residual[at]) + ple : ple);
#pragma unroll
    for (int j = 0; j + 1 < S; ++j) hist[j] = hist[j + 1];
    hist[S - 1] = u;
  }
#pragma unroll
  for (int j = 0; j < S; ++j) state[static_cast<int64_t>(c) * S + j] = float_to_bf16_bits(hist[j]);
}

// The row form: block (channel chunk, span); the span's request state.
template <int W, int D>
__global__ void conv_rows_kernel(const uint16_t* __restrict__ un, const uint16_t* __restrict__ gv,
                                 uint16_t* __restrict__ states, int64_t state_stride,
                                 const uint16_t* __restrict__ weight,
                                 const uint16_t* __restrict__ residual, uint16_t* __restrict__ out,
                                 const int32_t* __restrict__ req_ids,
                                 const int64_t* __restrict__ pos,
                                 const int32_t* __restrict__ spans, int channels,
                                 uint16_t* __restrict__ snapshots, int snapshot_rows) {
  dgpp_pdl_wait();
  constexpr int S = (W - 1) * D;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  const int q = blockIdx.y;
  const int s = spans[2 * q], len = spans[2 * q + 1];
  if (len <= 0) return;
  if (req_ids[s] < 0) {
    for (int t = s; t < s + len; ++t) {
      const int64_t at = static_cast<int64_t>(t) * channels + c;
      out[at] = residual ? residual[at] : static_cast<uint16_t>(0);
      if (snapshots && (snapshot_rows < 0 || t < snapshot_rows))
        for (int j = 0; j < S; ++j) snapshots[at * S + j] = 0;
    }
    return;
  }
  uint16_t* state = states + static_cast<int64_t>(req_ids[s]) * state_stride;
  float hist[S];
#pragma unroll
  for (int j = 0; j < S; ++j) hist[j] = bf16_bits_to_float(state[static_cast<int64_t>(c) * S + j]);
  float w[W];
#pragma unroll
  for (int k = 0; k < W; ++k) w[k] = bf16_bits_to_float(weight[static_cast<int64_t>(c) * W + k]);
  bool touched = false;
  for (int t = s; t < s + len; ++t) {
    const int64_t at = static_cast<int64_t>(t) * channels + c;
    if (pos[t] >= 0) {
      const float u = bf16_bits_to_float(un[at]);
      float acc = 0.f;
#pragma unroll
      for (int k = 0; k < W - 1; ++k) acc = fmaf(w[k], hist[k * D], acc);
      acc = fmaf(w[W - 1], u, acc);
      const float cv = round_bf16(acc);
      const float act = round_bf16(cv * sigmoid_f(cv));
      const float ple = round_bf16(bf16_bits_to_float(gv[at]) + act);
      out[at] = float_to_bf16_bits(residual ? bf16_bits_to_float(residual[at]) + ple : ple);
#pragma unroll
      for (int j = 0; j + 1 < S; ++j) hist[j] = hist[j + 1];
      hist[S - 1] = u;
      touched = true;
    } else {
      out[at] = residual ? residual[at] : static_cast<uint16_t>(0);
    }
    if (snapshots && (snapshot_rows < 0 || t < snapshot_rows)) {
      uint16_t* snap = snapshots + static_cast<int64_t>(t) * channels * S + static_cast<int64_t>(c) * S;
#pragma unroll
      for (int j = 0; j < S; ++j) snap[j] = float_to_bf16_bits(hist[j]);
    }
  }
  if (touched) {
#pragma unroll
    for (int j = 0; j < S; ++j) state[static_cast<int64_t>(c) * S + j] = float_to_bf16_bits(hist[j]);
  }
}

}  // namespace

void qwen_ple_hash_ids_rows(const int64_t* tokens, int rows, const int32_t* req_ids,
                            const int64_t* pos, const int32_t* req_spans, int num_requests,
                            const int32_t* ctx, int32_t eos, const int64_t* multipliers,
                            const int64_t* head_vocab, const int64_t* head_offset, int heads,
                            int heads_per_ngram, int32_t* ids, cudaStream_t stream) {
  if (rows <= 0) return;
  (void)pos;
  if (!tokens || !req_ids || !req_spans || !ctx || !multipliers || !head_vocab || !head_offset || !ids)
    throw std::invalid_argument("qwen_ple_hash_ids_rows: null pointer");
  if (num_requests <= 0) throw std::invalid_argument("qwen_ple_hash_ids_rows: no request spans");
  if (heads <= 0 || heads > kMaxHeads || heads_per_ngram <= 0 || heads_per_ngram > heads)
    throw std::invalid_argument("qwen_ple_hash_ids_rows: bad head geometry");
  const int64_t total = static_cast<int64_t>(rows) * heads;
  const int blocks = static_cast<int>((total + 255) / 256);
  hash_ids_rows_kernel<<<blocks, 256, 0, stream>>>(tokens, rows, req_ids, req_spans, num_requests,
                                                   ctx, eos, multipliers, head_vocab, head_offset,
                                                   heads, heads_per_ngram, ids);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_ple_context_rows(const int64_t* tokens, int rows, const int32_t* req_ids,
                           const int64_t* pos, const int32_t* req_spans, int num_requests,
                           int32_t* ctx, int32_t* ctx_rows, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!tokens || !req_ids || !pos || !req_spans || !ctx || !ctx_rows)
    throw std::invalid_argument("qwen_ple_context_rows: null pointer");
  if (num_requests <= 0) throw std::invalid_argument("qwen_ple_context_rows: no request spans");
  const int blocks = (num_requests + 127) / 128;
  context_rows_kernel<<<blocks, 128, 0, stream>>>(tokens, req_ids, pos, req_spans, num_requests,
                                                  ctx, ctx_rows);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_ple_conv_rows_bf16(const uint16_t* un, const uint16_t* gv, uint16_t* states,
                             int64_t state_stride, const uint16_t* weight,
                             const uint16_t* residual, uint16_t* out, const int32_t* req_ids,
                             const int64_t* pos, const int32_t* req_spans, int num_requests,
                             int channels, int width, int dilation, cudaStream_t stream,
                             uint16_t* snapshots, int snapshot_rows) {
  if (num_requests <= 0) return;
  if (!un || !gv || !states || !weight || !out || !req_ids || !pos || !req_spans)
    throw std::invalid_argument("qwen_ple_conv_rows: null pointer");
  if (width != 4 || dilation != 3)
    throw std::invalid_argument("qwen_ple_conv_rows: only width 4 / dilation 3 is instantiated");
  if (channels <= 0 || state_stride < static_cast<int64_t>(channels) * 9)
    throw std::invalid_argument("qwen_ple_conv_rows: bad geometry");
  const dim3 grid((channels + 255) / 256, static_cast<unsigned>(num_requests));
  conv_rows_kernel<4, 3><<<grid, 256, 0, stream>>>(un, gv, states, state_stride, weight, residual,
                                                   out, req_ids, pos, req_spans, channels, snapshots, snapshot_rows);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_ple_hash_ids(const int32_t* tokens, int n, int32_t ctx_prev1, int32_t ctx_prev2,
                       int32_t eos, const int64_t* multipliers, const int64_t* head_vocab,
                       const int64_t* head_offset, int heads, int heads_per_ngram,
                       int32_t* ids, cudaStream_t stream) {
  if (n <= 0) return;
  if (!tokens || !multipliers || !head_vocab || !head_offset || !ids)
    throw std::invalid_argument("qwen_ple_hash_ids: null pointer");
  if (heads <= 0 || heads > kMaxHeads || heads_per_ngram <= 0 || heads_per_ngram > heads)
    throw std::invalid_argument("qwen_ple_hash_ids: bad head geometry");
  const int64_t total = static_cast<int64_t>(n) * heads;
  const int blocks = static_cast<int>((total + 255) / 256);
  hash_ids_kernel<<<blocks, 256, 0, stream>>>(tokens, n, ctx_prev1, ctx_prev2, eos, multipliers,
                                              head_vocab, head_offset, heads, heads_per_ngram,
                                              ids);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_ple_gather_bf16(const uint8_t* table, int64_t row_begin, int64_t rows, float scale,
                          const int32_t* ids, int n, int heads, int head_begin,
                          int heads_local, int head_dim, uint16_t* out, cudaStream_t stream) {
  if (n <= 0) return;
  if (!table || !ids || !out) throw std::invalid_argument("qwen_ple_gather: null pointer");
  if (head_dim <= 0 || head_dim % 8 != 0 || heads_local <= 0 || head_begin < 0 ||
      head_begin + heads_local > heads)
    throw std::invalid_argument("qwen_ple_gather: bad geometry (head_dim % 8, head range)");
  if ((reinterpret_cast<uintptr_t>(table) & 7u) || (reinterpret_cast<uintptr_t>(out) & 15u))
    throw std::invalid_argument("qwen_ple_gather: table 8-byte and out 16-byte aligned required");
  const int64_t pairs = static_cast<int64_t>(n) * heads_local;
  constexpr int kWarps = 8;
  const int blocks = static_cast<int>((pairs + kWarps - 1) / kWarps);
  gather_kernel<<<blocks, kWarps * 32, 0, stream>>>(table, row_begin, rows, scale, ids, n, heads,
                                                    head_begin, heads_local, head_dim, out);
  DGPP_CUDA_OK(cudaGetLastError());
}

// The staged rows' conversion: one warp per (token, local head) row of the
// staged buffer, the chunk arithmetic gather_kernel's.
__global__ void gather_staged_kernel(const uint8_t* __restrict__ staged, float scale, int n,
                                     int heads_local, int head_dim, uint16_t* __restrict__ out) {
  dgpp_pdl_wait();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int64_t pair = static_cast<int64_t>(blockIdx.x) * (blockDim.x / 32) + warp;
  if (pair >= static_cast<int64_t>(n) * heads_local) return;
  const uint8_t* src = staged + pair * head_dim;
  uint16_t* dst = out + pair * head_dim;
  const int chunks = head_dim / 8;
  for (int c = lane; c < chunks; c += 32) {
    const uint2 codes = reinterpret_cast<const uint2*>(src)[c];
    const uint32_t cw[2] = {codes.x, codes.y};
    uint32_t packed[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t w = cw[j / 2] >> (16 * (j % 2));
      const uint16_t lo =
          float_to_bf16_bits(fp8_e4m3_bits_to_float(static_cast<uint8_t>(w & 0xFFu)) * scale);
      const uint16_t hi = float_to_bf16_bits(
          fp8_e4m3_bits_to_float(static_cast<uint8_t>((w >> 8) & 0xFFu)) * scale);
      packed[j] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
    }
    reinterpret_cast<uint4*>(dst)[c] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
  }
}

// ---- the NVFP4 table ----------------------------------------------------------
// One 16-element group of a row: eight code bytes (two e2m1 each, the low
// nibble the even element) and the group's e4m3 scale -> sixteen bf16.
// v = (e2m1 x e4m3) x global, in that order: the first product is exact in
// fp32 (six significant bits), the second rounds once, the bf16 store once
// — modelopt's dequant chain, so the values are the reference's.
__device__ __forceinline__ void fp4_group_to_bf16(const uint8_t* __restrict__ codes8, uint8_t scale,
                                                  float global, uint16_t* __restrict__ dst16) {
  const float s = fp8_e4m3_bits_to_float(scale);
  const uint2 c = *reinterpret_cast<const uint2*>(codes8);
  const uint32_t cw[2] = {c.x, c.y};
  uint32_t packed[8];
#pragma unroll
  for (int b = 0; b < 8; ++b) {
    const uint32_t byte = (cw[b / 4] >> (8 * (b % 4))) & 0xFFu;
    const float lo = (fp4_e2m1_bits_to_float(static_cast<uint8_t>(byte & 0xFu)) * s) * global;
    const float hi = (fp4_e2m1_bits_to_float(static_cast<uint8_t>(byte >> 4)) * s) * global;
    packed[b] = static_cast<uint32_t>(float_to_bf16_bits(lo)) | (static_cast<uint32_t>(float_to_bf16_bits(hi)) << 16);
  }
  uint4* d = reinterpret_cast<uint4*>(dst16);
  d[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
  d[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
}

// One warp per (token, local head): lane g converts the row's group g
// (head_dim / 16 groups; lanes past them idle).
__global__ void gather_fp4_kernel(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                                  int64_t row_begin, int64_t rows, float global,
                                  const int32_t* __restrict__ ids, int n, int heads, int head_begin,
                                  int heads_local, int head_dim, uint16_t* __restrict__ out) {
  dgpp_pdl_wait();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int64_t pair = static_cast<int64_t>(blockIdx.x) * (blockDim.x / 32) + warp;
  if (pair >= static_cast<int64_t>(n) * heads_local) return;
  const int t = static_cast<int>(pair / heads_local);
  const int hl = static_cast<int>(pair - static_cast<int64_t>(t) * heads_local);
  const int64_t id = ids[static_cast<int64_t>(t) * heads + head_begin + hl];
  const int64_t row = id - row_begin;
  if (row < 0 || row >= rows) __trap();  // the hash never leaves the head's range
  const int groups = head_dim / 16;
  if (lane >= groups) return;
  const uint8_t* crow = codes + row * (head_dim / 2);
  const uint8_t* srow = scales + row * groups;
  uint16_t* dst = out + (static_cast<int64_t>(t) * heads_local + hl) * head_dim;
  fp4_group_to_bf16(crow + lane * 8, srow[lane], global, dst + lane * 16);
}

__global__ void gather_staged_fp4_kernel(const uint8_t* __restrict__ staged_codes,
                                         const uint8_t* __restrict__ staged_scales, float global, int n,
                                         int heads_local, int head_dim, uint16_t* __restrict__ out) {
  dgpp_pdl_wait();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int64_t pair = static_cast<int64_t>(blockIdx.x) * (blockDim.x / 32) + warp;
  if (pair >= static_cast<int64_t>(n) * heads_local) return;
  const int groups = head_dim / 16;
  if (lane >= groups) return;
  const uint8_t* crow = staged_codes + pair * (head_dim / 2);
  const uint8_t* srow = staged_scales + pair * groups;
  uint16_t* dst = out + pair * head_dim;
  fp4_group_to_bf16(crow + lane * 8, srow[lane], global, dst + lane * 16);
}

namespace {
void check_fp4_geometry(int head_dim, int heads_local, const void* codes, const void* out, const char* who) {
  if (head_dim <= 0 || head_dim % 16 != 0 || head_dim > 512 || heads_local <= 0)
    throw std::invalid_argument(std::string(who) + ": bad geometry (head_dim % 16, head_dim <= 512, heads)");
  if ((reinterpret_cast<uintptr_t>(codes) & 7u) || (reinterpret_cast<uintptr_t>(out) & 15u))
    throw std::invalid_argument(std::string(who) + ": codes 8-byte and out 16-byte aligned required");
}
}  // namespace

void qwen_ple_gather_fp4_bf16(const uint8_t* codes, const uint8_t* scales, int64_t row_begin, int64_t rows,
                              float global, const int32_t* ids, int n, int heads, int head_begin,
                              int heads_local, int head_dim, uint16_t* out, cudaStream_t stream) {
  if (n <= 0) return;
  if (!codes || !scales || !ids || !out) throw std::invalid_argument("qwen_ple_gather_fp4: null pointer");
  if (head_begin < 0 || head_begin + heads_local > heads || rows <= 0)
    throw std::invalid_argument("qwen_ple_gather_fp4: bad head range");
  check_fp4_geometry(head_dim, heads_local, codes, out, "qwen_ple_gather_fp4");
  const int64_t pairs = static_cast<int64_t>(n) * heads_local;
  constexpr int kWarps = 8;
  const int blocks = static_cast<int>((pairs + kWarps - 1) / kWarps);
  gather_fp4_kernel<<<blocks, kWarps * 32, 0, stream>>>(codes, scales, row_begin, rows, global, ids, n, heads,
                                                        head_begin, heads_local, head_dim, out);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_ple_gather_staged_fp4_bf16(const uint8_t* staged_codes, const uint8_t* staged_scales, float global,
                                     int n, int heads_local, int head_dim, uint16_t* out, cudaStream_t stream) {
  if (n <= 0) return;
  if (!staged_codes || !staged_scales || !out) throw std::invalid_argument("qwen_ple_gather_staged_fp4: null pointer");
  check_fp4_geometry(head_dim, heads_local, staged_codes, out, "qwen_ple_gather_staged_fp4");
  const int64_t pairs = static_cast<int64_t>(n) * heads_local;
  constexpr int kWarps = 8;
  const int blocks = static_cast<int>((pairs + kWarps - 1) / kWarps);
  gather_staged_fp4_kernel<<<blocks, kWarps * 32, 0, stream>>>(staged_codes, staged_scales, global, n, heads_local,
                                                               head_dim, out);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_ple_gather_staged_bf16(const uint8_t* staged, float scale, int n, int heads_local,
                                 int head_dim, uint16_t* out, cudaStream_t stream) {
  if (n <= 0) return;
  if (!staged || !out) throw std::invalid_argument("qwen_ple_gather_staged: null pointer");
  if (head_dim <= 0 || head_dim % 8 != 0 || heads_local <= 0)
    throw std::invalid_argument("qwen_ple_gather_staged: bad geometry (head_dim % 8, heads)");
  if ((reinterpret_cast<uintptr_t>(staged) & 7u) || (reinterpret_cast<uintptr_t>(out) & 15u))
    throw std::invalid_argument("qwen_ple_gather_staged: staged 8-byte and out 16-byte aligned required");
  const int64_t pairs = static_cast<int64_t>(n) * heads_local;
  constexpr int kWarps = 8;
  const int blocks = static_cast<int>((pairs + kWarps - 1) / kWarps);
  gather_staged_kernel<<<blocks, kWarps * 32, 0, stream>>>(staged, scale, n, heads_local, head_dim, out);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_ple_gate_bf16(const uint16_t* key_n, const uint16_t* query_n, const uint16_t* value,
                        uint16_t* gated, int n, int hc, int hidden, cudaStream_t stream) {
  if (n <= 0) return;
  if (!key_n || !query_n || !value || !gated) throw std::invalid_argument("qwen_ple_gate: null pointer");
  if (hc <= 0 || hidden <= 0) throw std::invalid_argument("qwen_ple_gate: bad geometry");
  const int64_t rows = static_cast<int64_t>(n) * hc;
  if (rows > 0x7fffffff) throw std::invalid_argument("qwen_ple_gate: too many rows");
  // The reference divides by the Python float sqrt(hidden), which a bf16
  // op evaluates in fp32.
  const float divisor = static_cast<float>(sqrt(static_cast<double>(hidden)));
  gate_kernel<<<static_cast<int>(rows), kGateThreads, 0, stream>>>(key_n, query_n, value, gated,
                                                                    hc, hidden, divisor);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_ple_conv_bf16(const uint16_t* un, const uint16_t* gv, uint16_t* state,
                        const uint16_t* weight, const uint16_t* residual, uint16_t* out, int n,
                        int channels, int width, int dilation, cudaStream_t stream) {
  if (n <= 0) return;
  if (!un || !gv || !state || !weight || !out) throw std::invalid_argument("qwen_ple_conv: null pointer");
  if (width != 4 || dilation != 3)
    throw std::invalid_argument("qwen_ple_conv: only width 4 / dilation 3 is instantiated");
  if (channels <= 0) throw std::invalid_argument("qwen_ple_conv: bad geometry");
  const int blocks = (channels + 255) / 256;
  conv_kernel<4, 3><<<blocks, 256, 0, stream>>>(un, gv, state, weight, residual, out, n, channels);
  DGPP_CUDA_OK(cudaGetLastError());
}

namespace {
__global__ void publish_stage_kernel(uint64_t* __restrict__ device_seq, uint64_t* __restrict__ pinned_seq,
                                     int32_t* __restrict__ pinned_rows, int rows) {
  dgpp_pdl_wait();
  if (threadIdx.x != 0) return;
  asm volatile("st.relaxed.sys.global.s32 [%0], %1;" ::"l"(pinned_rows), "r"(rows) : "memory");
  __threadfence_system();
  const uint64_t v = *device_seq + 1;
  *device_seq = v;
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(pinned_seq), "l"(v) : "memory");
}
}  // namespace

void qwen_ple_publish_stage(uint64_t* device_seq, uint64_t* pinned_seq, int32_t* pinned_rows, int rows,
                            cudaStream_t stream) {
  if (!device_seq || !pinned_seq || !pinned_rows) throw std::invalid_argument("qwen_ple_publish_stage: null");
  publish_stage_kernel<<<1, 32, 0, stream>>>(device_seq, pinned_seq, pinned_rows, rows);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
