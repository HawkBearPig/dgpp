#include "kernels/dsv41_engram.hpp"
#include "kernels/pdl.cuh"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {

constexpr int kMaxNgram = 8;

// The span of row t: its request's first row in the batch (the row map's
// contract: a few spans, so a scan is the lookup).
__device__ __forceinline__ int span_start_of(int t, const int32_t* __restrict__ spans,
                                             int num_requests) {
  for (int q = 0; q < num_requests; ++q) {
    const int s = spans[2 * q], len = spans[2 * q + 1];
    if (t >= s && t < s + len) return s;
  }
  return t;
}

// One thread per (row, layer, head): the row's compressed id and its
// max_ngram - 1 predecessors (the span's earlier rows, then the request's
// context, then pad), the running XOR of the multiplied ids, one bucket
// per n-gram size for this head.
__global__ void hash_ids_rows_kernel(const int64_t* __restrict__ tokens, int rows,
                                     const int32_t* __restrict__ req_ids,
                                     const int32_t* __restrict__ spans, int num_requests,
                                     const int32_t* __restrict__ ctx, Dsv41EngramHash h,
                                     int32_t* __restrict__ ids) {
  dgpp_pdl_wait();
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t per_row = static_cast<int64_t>(h.layers) * h.heads;
  if (i >= static_cast<int64_t>(rows) * per_row) return;
  const int t = static_cast<int>(i / per_row);
  const int rem = static_cast<int>(i - static_cast<int64_t>(t) * per_row);
  const int l = rem / h.heads;
  const int hd = rem - l * h.heads;
  const int s = span_start_of(t, spans, num_requests);
  const int p = t - s;
  const int request = req_ids[t];
  const int32_t* c = request >= 0 ? ctx + static_cast<int64_t>(request) * kDsv41EngramCtx : nullptr;
  int64_t y[kMaxNgram];
  for (int k = 0; k < h.max_ngram; ++k) {
    if (request < 0) {
      y[k] = h.pad_class;
    } else if (k == 0) {
      y[k] = h.token_map[tokens[t]];
    } else if (p >= k) {
      y[k] = h.token_map[tokens[t - k]];
    } else {
      y[k] = c[k - 1 - p];  // the context's (k - p)-th newest id (pad past the start)
    }
  }
  const int64_t* mult = h.multipliers + static_cast<int64_t>(l) * h.max_ngram;
  // int64 products under torch's wrapping multiply (the unsigned form
  // makes the wrap defined; they stay below 2^63 for this vocabulary).
  uint64_t rolling = static_cast<uint64_t>(y[0]) * static_cast<uint64_t>(mult[0]);
  const int ngrams = h.max_ngram - 1;
  for (int n = 1; n <= ngrams; ++n) {
    rolling ^= static_cast<uint64_t>(y[n]) * static_cast<uint64_t>(mult[n]);
    const int64_t mixed = static_cast<int64_t>(rolling);
    const int64_t at = (static_cast<int64_t>(l) * ngrams + (n - 1)) * h.heads + hd;
    const int64_t prime = h.primes[at];
    int64_t r = mixed % prime;
    if (r < 0) r += prime;  // torch.remainder: the divisor's sign
    ids[(static_cast<int64_t>(t) * h.layers + l) * (ngrams * h.heads) + (n - 1) * h.heads + hd] =
        static_cast<int32_t>(r + h.offsets[at]);
  }
}

// One thread per span: the running context through its rows.
__global__ void context_rows_kernel(const int64_t* __restrict__ tokens,
                                    const int32_t* __restrict__ req_ids,
                                    const int64_t* __restrict__ pos,
                                    const int32_t* __restrict__ spans, int num_requests,
                                    const int32_t* __restrict__ token_map, int keep,
                                    int32_t* __restrict__ ctx, int32_t* __restrict__ ctx_rows) {
  dgpp_pdl_wait();
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_requests) return;
  const int s = spans[2 * q], len = spans[2 * q + 1];
  if (len <= 0) return;
  if (req_ids[s] < 0) {
    for (int t = s; t < s + len; ++t)
      for (int k = 0; k < kDsv41EngramCtx; ++k)
        ctx_rows[static_cast<int64_t>(t) * kDsv41EngramCtx + k] = 0;
    return;
  }
  int32_t* c = ctx + static_cast<int64_t>(req_ids[s]) * kDsv41EngramCtx;
  int32_t cur[kDsv41EngramCtx];
  for (int k = 0; k < kDsv41EngramCtx; ++k) cur[k] = c[k];
  bool touched = false;
  for (int t = s; t < s + len; ++t) {
    if (pos[t] >= 0) {
      for (int k = keep - 1; k > 0; --k) cur[k] = cur[k - 1];
      cur[0] = token_map[tokens[t]];
      touched = true;
    }
    for (int k = 0; k < kDsv41EngramCtx; ++k)
      ctx_rows[static_cast<int64_t>(t) * kDsv41EngramCtx + k] = cur[k];
  }
  if (touched)
    for (int k = 0; k < kDsv41EngramCtx; ++k) c[k] = cur[k];
}

// The e8m0 scale as fp32 (2^(byte - 127); 255 = NaN).
__device__ __forceinline__ float e8m0_to_float(uint8_t b) {
  if (b == 255) return __int_as_float(0x7FC00000);
  return b == 0 ? 0x1p-127f : __uint_as_float(static_cast<uint32_t>(b) << 23);
}

// One warp per (token, local row): the row's head_dim / 8 chunks of eight
// e4m3 codes, each lane one chunk -> eight bf16 under the chunk's block
// scale (a chunk of 8 lies inside one block of 32).
__global__ void gather_staged_kernel(const uint8_t* __restrict__ staged, int n, int rows_local,
                                     int head_dim, uint16_t* __restrict__ out) {
  dgpp_pdl_wait();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int64_t pair = static_cast<int64_t>(blockIdx.x) * (blockDim.x / 32) + warp;
  if (pair >= static_cast<int64_t>(n) * rows_local) return;
  const int row_bytes = head_dim + head_dim / 32;
  const uint8_t* src = staged + pair * row_bytes;
  const uint8_t* scales = src + head_dim;
  uint16_t* dst = out + pair * head_dim;
  const int chunks = head_dim / 8;
  for (int ch = lane; ch < chunks; ch += 32) {
    const float s = e8m0_to_float(scales[(ch * 8) / 32]);
    const uint2 codes = reinterpret_cast<const uint2*>(src)[ch];
    const uint32_t cw[2] = {codes.x, codes.y};
    uint32_t packed[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t w = cw[j / 2] >> (16 * (j % 2));
      const uint16_t lo = float_to_bf16_bits(fp8_e4m3_bits_to_float(static_cast<uint8_t>(w & 0xFFu)) * s);
      const uint16_t hi =
          float_to_bf16_bits(fp8_e4m3_bits_to_float(static_cast<uint8_t>((w >> 8) & 0xFFu)) * s);
      packed[j] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
    }
    reinterpret_cast<uint4*>(dst)[ch] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
  }
}

constexpr int kGateThreads = 256;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
  return v;
}
__device__ __forceinline__ float block_sum(float v, float* scratch) {
  v = warp_sum(v);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  __syncthreads();
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  float total = scratch[0];
#pragma unroll
  for (int w = 1; w < kGateThreads / 32; ++w) total += scratch[w];
  return total;
}

// One block per (token, stream): the two sums of squares and the weighted
// dot in one pass (fp32, fixed per-thread order + the block tree), the gate,
// the update.
__global__ void gate_rows_kernel(uint16_t* __restrict__ x, const uint16_t* __restrict__ kv,
                                 const uint16_t* __restrict__ q_weight,
                                 const uint16_t* __restrict__ k_weight, int rows, int hc,
                                 int hidden, float eps) {
  dgpp_pdl_wait();
  __shared__ float scratch[kGateThreads / 32];
  const int t = blockIdx.x / hc, i = blockIdx.x % hc;
  if (t >= rows) return;
  uint16_t* xi = x + (static_cast<size_t>(t) * hc + i) * hidden;
  const uint16_t* key = kv + static_cast<size_t>(t) * (hc + 1) * hidden + static_cast<size_t>(i) * hidden;
  const uint16_t* value = kv + static_cast<size_t>(t) * (hc + 1) * hidden + static_cast<size_t>(hc) * hidden;
  const uint16_t* qw = q_weight + static_cast<size_t>(i) * hidden;
  const uint16_t* kw = k_weight + static_cast<size_t>(i) * hidden;
  float ssx = 0.f, ssk = 0.f, dot = 0.f;
  for (int d = threadIdx.x; d < hidden; d += kGateThreads) {
    const float hv = bf16_bits_to_float(xi[d]);
    const float kvv = bf16_bits_to_float(key[d]);
    const float w = bf16_bits_to_float(qw[d]) * bf16_bits_to_float(kw[d]);
    ssx = __fmaf_rn(hv, hv, ssx);
    ssk = __fmaf_rn(kvv, kvv, ssk);
    dot = __fmaf_rn(hv * w, kvv, dot);
  }
  const float sx = block_sum(ssx, scratch);
  const float sk = block_sum(ssk, scratch);
  const float sd = block_sum(dot, scratch);
  const float rstd = rsqrtf(sx / static_cast<float>(hidden) + eps) *
                     rsqrtf(sk / static_cast<float>(hidden) + eps);
  const float g0 = sd * rstd * rsqrtf(static_cast<float>(hidden));
  const float mag = sqrtf(fmaxf(fabsf(g0), 1e-6f));
  const float signed_sqrt = copysignf(mag, g0);
  const float gate = 1.0f / (1.0f + expf(-signed_sqrt));
  for (int d = threadIdx.x; d < hidden; d += kGateThreads) {
    const float hv = bf16_bits_to_float(xi[d]);
    xi[d] = float_to_bf16_bits(__fmaf_rn(gate, bf16_bits_to_float(value[d]), hv));
  }
}

}  // namespace

void dsv41_engram_hash_ids_rows(const int64_t* tokens, int rows, const int32_t* req_ids,
                                const int64_t* pos, const int32_t* req_spans, int num_requests,
                                const int32_t* ctx, const Dsv41EngramHash& h, int32_t* ids,
                                cudaStream_t stream) {
  (void)pos;
  if (rows <= 0) return;
  if (!tokens || !req_ids || !req_spans || !ctx || !ids || !h.token_map || !h.multipliers ||
      !h.primes || !h.offsets)
    throw std::invalid_argument("dsv41_engram_hash: null pointer");
  if (h.layers <= 0 || h.heads <= 0 || h.max_ngram < 2 || h.max_ngram > kMaxNgram ||
      h.max_ngram - 1 > kDsv41EngramCtx)
    throw std::invalid_argument("dsv41_engram_hash: bad geometry (layers, heads, max_ngram)");
  const int64_t total = static_cast<int64_t>(rows) * h.layers * h.heads;
  const int blocks = static_cast<int>((total + 255) / 256);
  hash_ids_rows_kernel<<<blocks, 256, 0, stream>>>(tokens, rows, req_ids, req_spans, num_requests,
                                                   ctx, h, ids);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv41_engram_context_rows(const int64_t* tokens, int rows, const int32_t* req_ids,
                               const int64_t* pos, const int32_t* req_spans, int num_requests,
                               const int32_t* token_map, int max_ngram, int32_t* ctx,
                               int32_t* ctx_rows, cudaStream_t stream) {
  if (rows <= 0 || num_requests <= 0) return;
  if (!tokens || !req_ids || !pos || !req_spans || !token_map || !ctx || !ctx_rows)
    throw std::invalid_argument("dsv41_engram_context: null pointer");
  if (max_ngram < 2 || max_ngram - 1 > kDsv41EngramCtx)
    throw std::invalid_argument("dsv41_engram_context: max_ngram outside [2, 5]");
  const int blocks = (num_requests + 63) / 64;
  context_rows_kernel<<<blocks, 64, 0, stream>>>(tokens, req_ids, pos, req_spans, num_requests,
                                                 token_map, max_ngram - 1, ctx, ctx_rows);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv41_engram_gather_staged_bf16(const uint8_t* staged, int n, int rows_local, int head_dim,
                                     uint16_t* out, cudaStream_t stream) {
  if (n <= 0) return;
  if (!staged || !out) throw std::invalid_argument("dsv41_engram_gather: null pointer");
  if (head_dim <= 0 || head_dim % 32 != 0 || rows_local <= 0)
    throw std::invalid_argument("dsv41_engram_gather: bad geometry (head_dim % 32, rows)");
  if ((reinterpret_cast<uintptr_t>(staged) & 7u) || (reinterpret_cast<uintptr_t>(out) & 15u))
    throw std::invalid_argument("dsv41_engram_gather: staged 8-byte and out 16-byte aligned required");
  const int row_bytes = head_dim + head_dim / 32;
  if (row_bytes % 8 != 0)
    throw std::invalid_argument("dsv41_engram_gather: the staged row must be 8-byte aligned (head_dim % 64)");
  const int64_t pairs = static_cast<int64_t>(n) * rows_local;
  constexpr int kWarps = 8;
  const int blocks = static_cast<int>((pairs + kWarps - 1) / kWarps);
  gather_staged_kernel<<<blocks, kWarps * 32, 0, stream>>>(staged, n, rows_local, head_dim, out);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv41_engram_gate_rows(uint16_t* x, const uint16_t* kv, const uint16_t* q_weight,
                            const uint16_t* k_weight, int rows, int hc, int hidden, float eps,
                            cudaStream_t stream) {
  if (rows <= 0) return;
  if (!x || !kv || !q_weight || !k_weight) throw std::invalid_argument("dsv41_engram_gate: null pointer");
  if (hc <= 0 || hidden <= 0) throw std::invalid_argument("dsv41_engram_gate: bad geometry");
  if (static_cast<int64_t>(rows) * hc > 65535 * 64LL)
    throw std::invalid_argument("dsv41_engram_gate: too many rows");
  gate_rows_kernel<<<rows * hc, kGateThreads, 0, stream>>>(x, kv, q_weight, k_weight, rows, hc, hidden,
                                                           eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
