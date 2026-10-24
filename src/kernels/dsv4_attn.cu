#include "kernels/dsv4_attn.hpp"
#include "kernels/pdl.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/latent_format.hpp"

namespace dgpp {
namespace {

constexpr int kThreads = 256;
constexpr float kHadamardScale = 0.088388347648318440550f;  // 128^-0.5

// Block-wide sum in a fixed order (warp shuffles, then one warp over the
// warp sums): deterministic for a given blockDim (a multiple of 32, <= 256).
__device__ __forceinline__ float block_sum(float v, float* red /*[8]*/) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __syncthreads();
  if (lane == 0) red[warp] = v;
  __syncthreads();
  float t = 0.0f;
  if (warp == 0) {
    t = lane < (int(blockDim.x) >> 5) ? red[lane] : 0.0f;
    for (int off = 4; off > 0; off >>= 1) t += __shfl_xor_sync(0xffffffffu, t, off);
    if (lane == 0) red[0] = t;
  }
  __syncthreads();
  t = red[0];
  __syncthreads();
  return t;
}

// ---- the query's per-head normalization -------------------------------------
__global__ void q_head_rmsnorm_kernel(uint16_t* q, int dim, float eps) {
  dgpp_pdl_wait();
  __shared__ float red[8];
  uint16_t* x = q + int64_t(blockIdx.x) * dim;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = bf16_bits_to_float(x[i]);
    ss = __fmaf_rn(v, v, ss);
  }
  const float total = block_sum(ss, red);
  const float rs = rsqrtf(__fadd_rn(__fdiv_rn(total, float(dim)), eps));
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    x[i] = float_to_bf16_bits(__fmul_rn(bf16_bits_to_float(x[i]), rs));
}

// ---- act_quant on the 448 ---------------------------------------------------------
// One 64-element group's quantize-dequantize: the group's absmax over two
// warps (a pair of warps is a group), the release's scale byte, the e4m3
// code, the decode. `v` is this thread's element; red: smem [8] by warp.
__device__ __forceinline__ float act_quant_group(float v, float* red /*[8]*/) {
  float amax = fabsf(v);
  for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __syncthreads();
  if (lane == 0) red[warp] = amax;
  __syncthreads();
  amax = fmaxf(red[warp & ~1], red[warp | 1]);
  __syncthreads();
  const float s = e8m0_byte_to_float(latent_fp8_block_scale_byte(amax));
  return bf16_bits_to_float(latent_fp8_block_decode_bf16(latent_fp8_block_encode(v, s), s));
}

// The 448 of a 512-wide row held as floats in smem `row` (256 threads:
// thread t owns elements t and t + 256): elements [0, 256) are groups 0..3,
// [256, 448) groups 4..6; the tail [448, 512) is left as it is.
__device__ __forceinline__ void act_quant_row_smem(float* row /*[512]*/, float* red) {
  const float a = act_quant_group(row[threadIdx.x], red);
  const bool second = threadIdx.x < kDsv4Nope - kThreads;
  const float b = act_quant_group(second ? row[kThreads + threadIdx.x] : 0.0f, red);
  row[threadIdx.x] = a;
  if (second) row[kThreads + threadIdx.x] = b;
  __syncthreads();
}

__global__ void act_quant_nope_kernel(uint16_t* x, int64_t row_stride, const int64_t* pos) {
  dgpp_pdl_wait();
  __shared__ float row[kDsv4Latent];
  __shared__ float red[8];
  const int64_t r = blockIdx.x;
  if (pos != nullptr && pos[r] < 0) return;
  uint16_t* xr = x + r * row_stride;
  row[threadIdx.x] = bf16_bits_to_float(xr[threadIdx.x]);
  row[kThreads + threadIdx.x] = bf16_bits_to_float(xr[kThreads + threadIdx.x]);
  __syncthreads();
  act_quant_row_smem(row, red);
  xr[threadIdx.x] = float_to_bf16_bits(row[threadIdx.x]);
  if (threadIdx.x < kDsv4Nope - kThreads) xr[kThreads + threadIdx.x] = float_to_bf16_bits(row[kThreads + threadIdx.x]);
}

// ---- the decode walk's fused tails (2026-10-01) -----------------------------------------
// The K/V row's chain behind its projection — the latent's norm, the
// rotation of its last 64, act_quant on its first 448, the row into its
// ring slot and the row's window list — was six launches of 1-2.4 us each
// (csa2 rmsnorm / rope_apply, act_quant_nope, ring_slot_positions,
// dsa_latent_append, window_slots_decode) with a graph node's gap between
// them, on every layer of a pass. One block per row runs them in sequence:
// the same reductions (block_sum at 256 threads), the same expressions on
// the same values, the bf16 roundings where the separate launches wrote
// their outputs — bitwise the chain. A padding row (pos -1) keeps its
// normed latent, appends nothing and lists nothing.
__global__ void kv_tail_kernel(uint16_t* kv, const uint16_t* w, float eps, const int32_t* req_ids,
                               const int64_t* pos, const float* inv_freq, const int32_t* ring_table,
                               int ring_slots, uint8_t* ring, int window, int32_t* wlist, int32_t* wcounts) {
  dgpp_pdl_wait();
  __shared__ float row[kDsv4Latent];
  __shared__ float red[8];
  const int64_t r = blockIdx.x;
  uint16_t* xr = kv + r * kDsv4Latent;
  const int64_t p = pos[r];
  // rmsnorm_kernel (dim 512, 256 threads: thread t owns t and t + 256).
  float ss = 0.0f;
  for (int i = threadIdx.x; i < kDsv4Latent; i += blockDim.x) {
    const float v = bf16_bits_to_float(xr[i]);
    ss = __fmaf_rn(v, v, ss);
  }
  const float total = block_sum(ss, red);
  const float rs = rsqrtf(__fadd_rn(__fdiv_rn(total, float(kDsv4Latent)), eps));
  for (int i = threadIdx.x; i < kDsv4Latent; i += blockDim.x) {
    const float v = __fmul_rn(bf16_bits_to_float(xr[i]), rs);
    row[i] = bf16_bits_to_float(float_to_bf16_bits(__fmul_rn(bf16_bits_to_float(w[i]), v)));
  }
  __syncthreads();
  if (p < 0) {
    for (int i = threadIdx.x; i < kDsv4Latent; i += blockDim.x) xr[i] = float_to_bf16_bits(row[i]);
    if (wlist != nullptr) {
      for (int k = threadIdx.x; k < window; k += blockDim.x) wlist[r * window + k] = -1;
      if (threadIdx.x == 0) wcounts[r] = 0;
    }
    return;
  }
  // rope_apply_kernel on the last kDsv4Rope elements (pair i = thread i).
  if (threadIdx.x < kDsv4Rope / 2) {
    const int i = threadIdx.x;
    const int d0 = kDsv4Nope + 2 * i;
    const float ang = __fmul_rn(float(p), inv_freq[i]);
    const float c = cosf(ang);
    const float s = sinf(ang);
    const float x0 = row[d0], x1 = row[d0 + 1];
    row[d0] = bf16_bits_to_float(float_to_bf16_bits(x0 * c - x1 * s));
    row[d0 + 1] = bf16_bits_to_float(float_to_bf16_bits(x1 * c + x0 * s));
  }
  __syncthreads();
  // act_quant_nope_kernel on the first kDsv4Nope.
  act_quant_row_smem(row, red);
  // The row, and its ring slot (latent_append_kernel: the bf16 row format).
  const int64_t slot = p % ring_slots;
  const int64_t phys = int64_t(ring_table[req_ids[r]]) * ring_slots + slot;
  uint16_t* dst = reinterpret_cast<uint16_t*>(ring + phys * int64_t(kDsv4Latent) * 2);
  for (int i = threadIdx.x; i < kDsv4Latent; i += blockDim.x) {
    const uint16_t bits = float_to_bf16_bits(row[i]);
    xr[i] = bits;
    dst[i] = bits;
  }
  // window_slots_decode_kernel.
  if (wlist != nullptr) {
    const int n = int(min(int64_t(window), p + 1));
    for (int k = threadIdx.x; k < window; k += blockDim.x) {
      const int64_t q = p - (n - 1) + k;  // ascending positions
      wlist[r * window + k] = k < n ? int32_t(q % ring_slots) : -1;
    }
    if (threadIdx.x == 0) wcounts[r] = n;
  }
}

// The query's per-head norm and the rotation of each head's last 64 in one
// launch (q_head_rmsnorm_kernel, then rope_apply_kernel's pair arithmetic
// on the rounded values): one block per (row, head).
__global__ void q_head_rmsnorm_rope_kernel(uint16_t* q, int heads, float eps, const int64_t* pos,
                                           const float* inv_freq) {
  dgpp_pdl_wait();
  __shared__ float red[8];
  uint16_t* x = q + int64_t(blockIdx.x) * kDsv4Latent;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < kDsv4Latent; i += blockDim.x) {
    const float v = bf16_bits_to_float(x[i]);
    ss = __fmaf_rn(v, v, ss);
  }
  const float total = block_sum(ss, red);
  const float rs = rsqrtf(__fadd_rn(__fdiv_rn(total, float(kDsv4Latent)), eps));
  for (int i = threadIdx.x; i < kDsv4Latent; i += blockDim.x)
    x[i] = float_to_bf16_bits(__fmul_rn(bf16_bits_to_float(x[i]), rs));
  __syncthreads();
  const int64_t p = pos[blockIdx.x / heads];
  if (p < 0 || threadIdx.x >= kDsv4Rope / 2) return;
  const int i = threadIdx.x;
  uint16_t* v = x + kDsv4Nope + 2 * i;
  const float ang = __fmul_rn(float(p), inv_freq[i]);
  const float c = cosf(ang);
  const float s = sinf(ang);
  const float x0 = bf16_bits_to_float(v[0]), x1 = bf16_bits_to_float(v[1]);
  v[0] = float_to_bf16_bits(x0 * c - x1 * s);
  v[1] = float_to_bf16_bits(x1 * c + x0 * s);
}

// ---- the Hadamard rotation ----------------------------------------------------------
// The 128-point Sylvester transform of smem `a` (128 threads, thread =
// element) through the scratch `b`: seven butterfly stages, the result
// (unscaled) back in `a`.
__device__ __forceinline__ void hadamard128_smem(float* a, float* b) {
  float* src = a;
  float* dst = b;
#pragma unroll
  for (int len = 1; len < 128; len <<= 1) {
    const int t = threadIdx.x;
    const float v = (t & len) ? __fsub_rn(src[t - len], src[t]) : __fadd_rn(src[t], src[t + len]);
    dst[t] = v;
    __syncthreads();
    float* tmp = src;
    src = dst;
    dst = tmp;
  }
  // Seven stages: the result sits in `b`; bring it home.
  a[threadIdx.x] = src[threadIdx.x];
  __syncthreads();
}

__global__ void hadamard128_kernel(uint16_t* x) {
  dgpp_pdl_wait();
  __shared__ float a[128];
  __shared__ float b[128];
  uint16_t* xr = x + int64_t(blockIdx.x) * 128;
  a[threadIdx.x] = bf16_bits_to_float(xr[threadIdx.x]);
  __syncthreads();
  hadamard128_smem(a, b);
  xr[threadIdx.x] = float_to_bf16_bits(__fmul_rn(a[threadIdx.x], kHadamardScale));
}

__global__ void fold_weights_kernel(const uint16_t* w, const float* q_scale, float* out, int64_t n, float scale) {
  dgpp_pdl_wait();
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float ws = bf16_bits_to_float(float_to_bf16_bits(__fmul_rn(bf16_bits_to_float(w[i]), scale)));
  out[i] = __fmul_rn(ws, q_scale[i]);
}

// ---- the compressor rings --------------------------------------------------------------
__global__ void comp_ring_write_kernel(const float* kv, const float* score, const float* ape,
                                       const int32_t* req_ids, const int64_t* pos, int wide, int ratio, int slots,
                                       float* ring) {
  dgpp_pdl_wait();
  const int64_t r = blockIdx.x;
  const int64_t p = pos[r];
  if (p < 0) return;
  float* dst = ring + (int64_t(req_ids[r]) * slots + (p % slots)) * 2 * wide;
  const float* a = ape + (p % ratio) * wide;
  for (int c = threadIdx.x; c < wide; c += blockDim.x) {
    dst[c] = kv[r * wide + c];
    dst[wide + c] = __fadd_rn(score[r * wide + c], a[c]);
  }
}

// A group's sources. Slot i of an entry is one token: the previous
// group's tokens first (overlap), then the group's own; get() returns the
// (kv, score) pair of channel c or false when the slot does not exist (the
// first group has no previous one).
struct RingSource {  // decode: everything from the request's ring
  const float* ring;  // this request's [slots][2][wide]
  int64_t q;          // the completing position
  int ratio, dim, wide, slots;
  bool overlap;
  __device__ __forceinline__ int n_slots() const { return overlap ? 2 * ratio : ratio; }
  __device__ __forceinline__ bool get(int i, int c, float* kv, float* sc) const {
    const int own = overlap ? i - ratio : i;  // >= 0: the group's own token
    const int64_t p = q - ratio + 1 + own;
    if (p < 0) return false;
    const int ch = (overlap && own >= 0) ? dim + c : c;
    const float* s = ring + (p % slots) * 2 * wide;
    *kv = s[ch];
    *sc = s[wide + ch];
    return true;
  }
};
struct ChunkSource {  // prefill: the chunk's projections, the ring before it
  const float* kv;     // [T, wide]
  const float* score;  // [T, wide]
  const float* ape;    // [ratio, wide]
  const float* ring;   // this request's ring
  int64_t pos0;
  int group;           // the group's index within the chunk
  int ratio, dim, wide, slots;
  bool overlap;
  __device__ __forceinline__ int n_slots() const { return overlap ? 2 * ratio : ratio; }
  __device__ __forceinline__ bool get(int i, int c, float* kvo, float* sc) const {
    const int own = overlap ? i - ratio : i;
    const int tok = own >= 0 ? own : i;  // the token's index within its group
    const int ch = (overlap && own >= 0) ? dim + c : c;
    const int64_t row = int64_t(group) * ratio + own;  // the chunk row (negative: before the chunk)
    if (row >= 0) {
      *kvo = kv[row * wide + ch];
      *sc = __fadd_rn(score[row * wide + ch], ape[int64_t(tok) * wide + ch]);
      return true;
    }
    const int64_t p = pos0 + row;
    if (p < 0) return false;
    const float* s = ring + (p % slots) * 2 * wide;
    *kvo = s[ch];
    *sc = s[wide + ch];
    return true;
  }
};

// The pooled channel: the fp32 softmax over the entry's slots (max, the
// exps and their sum in slot order, the quotient per slot), the weighted
// kv sum in slot order, the bf16 rounding the reference's `.to(dtype)`
// applies before the norm.
// The slots' (kv, score) pairs are loaded eight at a time (independent
// loads issued together: a pass over the slots is one L2 round trip per
// eight, not one per slot — the ratio-4 entry's 48 dependent round trips
// were 21 us of every four-row verify walk), and an entry of at most eight
// slots (ratio 4) is loaded once for the three passes. The arithmetic and
// its order are unchanged.
template <class Src>
__device__ __forceinline__ float pool_channel(const Src& src, int c) {
  constexpr int kB = 8;
  const int n = src.n_slots();
  float kv[kB], sc[kB];
  bool ok[kB];
  const auto load = [&](int i0) {
#pragma unroll
    for (int j = 0; j < kB; ++j) {
      kv[j] = 0.0f;
      sc[j] = 0.0f;
      ok[j] = (i0 + j < n) && src.get(i0 + j, c, &kv[j], &sc[j]);
    }
  };
  const bool once = n <= kB;
  if (once) load(0);
  float m = -INFINITY;
  for (int i0 = 0; i0 < n; i0 += kB) {
    if (!once) load(i0);
#pragma unroll
    for (int j = 0; j < kB; ++j)
      if (ok[j]) m = fmaxf(m, sc[j]);
  }
  float den = 0.0f;
  float ex[kB];  // an entry loaded once keeps its exps for the weighted sum (the same values)
  for (int i0 = 0; i0 < n; i0 += kB) {
    if (!once) load(i0);
#pragma unroll
    for (int j = 0; j < kB; ++j) {
      ex[j] = ok[j] ? expf(__fsub_rn(sc[j], m)) : 0.0f;
      if (ok[j]) den = __fadd_rn(den, ex[j]);
    }
  }
  float acc = 0.0f;
  for (int i0 = 0; i0 < n; i0 += kB) {
    if (!once) load(i0);
#pragma unroll
    for (int j = 0; j < kB; ++j) {
      if (!ok[j]) continue;
      const float e = once ? ex[j] : expf(__fsub_rn(sc[j], m));
      acc = __fadd_rn(acc, __fmul_rn(kv[j], __fdiv_rn(e, den)));
    }
  }
  return bf16_bits_to_float(float_to_bf16_bits(acc));
}

// The rotation of pair i (elements base + 2i, base + 2i + 1 of smem row)
// at position p: csa2_rope_apply's arithmetic, one rounding.
__device__ __forceinline__ void rope_pair_smem(float* row, int base, int i, int64_t p, const float* inv_freq) {
  const float ang = __fmul_rn(float(p), inv_freq[i]);
  const float c = cosf(ang), s = sinf(ang);
  const float x0 = row[base + 2 * i], x1 = row[base + 2 * i + 1];
  row[base + 2 * i] = bf16_bits_to_float(float_to_bf16_bits(x0 * c - x1 * s));
  row[base + 2 * i + 1] = bf16_bits_to_float(float_to_bf16_bits(x1 * c + x0 * s));
}

// A main entry (256 threads; thread t owns channels t and t + 256): pooled,
// normed, rotated, quantized, into `out` (bf16 [512]).
template <class Src>
__device__ __forceinline__ void publish_main_entry(const Src& src, int64_t ent_pos, const uint16_t* norm_w,
                                                   float eps, const float* inv_freq, uint16_t* out, float* row,
                                                   float* red) {
  float v[2];
  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    v[j] = pool_channel(src, int(threadIdx.x) + j * kThreads);
    ss = __fmaf_rn(v[j], v[j], ss);
  }
  const float total = block_sum(ss, red);
  const float rs = rsqrtf(__fadd_rn(__fdiv_rn(total, float(kDsv4Latent)), eps));
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int c = int(threadIdx.x) + j * kThreads;
    row[c] = bf16_bits_to_float(float_to_bf16_bits(__fmul_rn(bf16_bits_to_float(norm_w[c]), __fmul_rn(v[j], rs))));
  }
  __syncthreads();
  if (threadIdx.x < kDsv4Rope / 2) rope_pair_smem(row, kDsv4Nope, threadIdx.x, ent_pos, inv_freq);
  __syncthreads();
  act_quant_row_smem(row, red);
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int c = int(threadIdx.x) + j * kThreads;
    out[c] = float_to_bf16_bits(row[c]);
  }
}

// The fp4 e8m0/32 quantization of a 128-wide row in smem (128 threads,
// warp b = block b) as exact e4m3 codes on one power-of-two row scale:
// csa2.cu's index_row_quant, reading floats.
__device__ __forceinline__ void index_row_quant_smem(const float* x, uint8_t* out_codes, float* out_scale,
                                                     unsigned* violations, int* s_kmax /*[4]*/) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const float v = x[threadIdx.x];
  float amax = fabsf(v);
  for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  const float floor_v = kLatentFp4Max * 1.1754943508222875e-38f;  // 6 * 2^-126
  amax = amax > floor_v ? amax : floor_v;
  const uint8_t kb = e8m0_ceil_log2_byte(amax * (1.0f / kLatentFp4Max));
  const float s = e8m0_byte_to_float(kb);
  const float q = fp4_e2m1_bits_to_float(float_to_fp4_e2m1_bits(v / s)) * s;  // the dequantized value
  if (lane == 0) s_kmax[warp] = int(kb);
  __syncthreads();
  const int kmax = max(max(s_kmax[0], s_kmax[1]), max(s_kmax[2], s_kmax[3]));
  const float S = ldexpf(1.0f, kmax - 127 - 6);
  const uint8_t code = float_to_fp8_e4m3_bits(q / S);
  const bool bad = fp8_e4m3_bits_to_float(code) * S != q;
  out_codes[threadIdx.x] = code;
  if (threadIdx.x == 0) *out_scale = S;
  const unsigned any = __ballot_sync(0xffffffffu, bad);
  if (violations != nullptr && lane == 0 && any != 0u) atomicAdd(violations, 1u);
}

// An index key (128 threads, thread = channel): pooled, normed, the tail
// rotated, the Hadamard rotation, the fp4 values into the planar cache.
template <class Src>
__device__ __forceinline__ void publish_index_entry(const Src& src, int64_t ent_pos, const uint16_t* norm_w,
                                                    float eps, const float* inv_freq, uint8_t* out_codes,
                                                    float* out_scale, unsigned* violations, float* a, float* b,
                                                    float* red, int* s_kmax) {
  const float v = pool_channel(src, threadIdx.x);
  const float total = block_sum(__fmul_rn(v, v), red);
  const float rs = rsqrtf(__fadd_rn(__fdiv_rn(total, float(kDsv4IndexDim)), eps));
  a[threadIdx.x] = bf16_bits_to_float(
      float_to_bf16_bits(__fmul_rn(bf16_bits_to_float(norm_w[threadIdx.x]), __fmul_rn(v, rs))));
  __syncthreads();
  if (threadIdx.x < kDsv4Rope / 2) rope_pair_smem(a, kDsv4IndexDim - kDsv4Rope, threadIdx.x, ent_pos, inv_freq);
  __syncthreads();
  hadamard128_smem(a, b);
  a[threadIdx.x] = bf16_bits_to_float(float_to_bf16_bits(__fmul_rn(a[threadIdx.x], kHadamardScale)));
  __syncthreads();
  index_row_quant_smem(a, out_codes, out_scale, violations, s_kmax);
}

__global__ void comp_publish_main_decode_kernel(const float* ring, int ratio, int dim, int wide, bool overlap,
                                                int slots, const int32_t* req_ids, const int64_t* pos,
                                                const uint16_t* norm_w, float eps, const float* inv_freq,
                                                const int32_t* block_tables, int blocks_per_request,
                                                int entries_per_block, uint16_t* cache) {
  dgpp_pdl_wait();
  __shared__ float row[kDsv4Latent];
  __shared__ float red[8];
  const int64_t r = blockIdx.x;
  const int64_t p = pos[r];
  if (p < 0 || (p + 1) % ratio != 0) return;
  const int32_t req = req_ids[r];
  const int64_t e = p / ratio;
  const int32_t blk = block_tables[int64_t(req) * blocks_per_request + e / entries_per_block];
  const int64_t slot = int64_t(blk) * entries_per_block + (e % entries_per_block);
  const RingSource src{ring + int64_t(req) * slots * 2 * wide, p, ratio, dim, wide, slots, overlap};
  publish_main_entry(src, e * ratio, norm_w, eps, inv_freq, cache + slot * kDsv4Latent, row, red);
}

// The lazy form: the completing row's group from its projections
// (kv / score [rows][ratio][wide]); no overlap, so the source never reads a
// ring.
__global__ void comp_publish_main_lazy_kernel(const float* kv, const float* score, const float* ape, int ratio,
                                              int dim, int wide, const int32_t* req_ids, const int64_t* pos,
                                              const uint16_t* norm_w, float eps, const float* inv_freq,
                                              const int32_t* block_tables, int blocks_per_request,
                                              int entries_per_block, uint16_t* cache) {
  dgpp_pdl_wait();
  __shared__ float row[kDsv4Latent];
  __shared__ float red[8];
  const int64_t r = blockIdx.x;
  const int64_t p = pos[r];
  if (p < 0 || (p + 1) % ratio != 0) return;
  const int32_t req = req_ids[r];
  const int64_t e = p / ratio;
  const int32_t blk = block_tables[int64_t(req) * blocks_per_request + e / entries_per_block];
  const int64_t slot = int64_t(blk) * entries_per_block + (e % entries_per_block);
  const int64_t g0 = r * ratio * wide;
  const ChunkSource src{kv + g0, score + g0, ape, nullptr, e * ratio, 0, ratio, dim, wide, 0, false};
  publish_main_entry(src, e * ratio, norm_w, eps, inv_freq, cache + slot * kDsv4Latent, row, red);
}

__global__ void u_ring_write_kernel(const uint16_t* hidden_rows, size_t hidden_stride, const int32_t* req_ids,
                                    const int64_t* pos, int vecs, int hidden, int slots, uint16_t* ring) {
  dgpp_pdl_wait();
  const int64_t r = blockIdx.x;
  const int64_t p = pos[r];
  if (p < 0) return;
  const uint4* src = reinterpret_cast<const uint4*>(hidden_rows + r * hidden_stride);
  uint4* dst = reinterpret_cast<uint4*>(ring + (int64_t(req_ids[r]) * slots + (p % slots)) * hidden);
  for (int c = threadIdx.x; c < vecs; c += blockDim.x) dst[c] = src[c];
}

__global__ void comp_publish_index_decode_kernel(const float* ring, int ratio, int dim, int wide, bool overlap,
                                                 int slots, const int32_t* req_ids, const int64_t* pos,
                                                 const uint16_t* norm_w, float eps, const float* inv_freq,
                                                 const int32_t* block_tables, int blocks_per_request,
                                                 int entries_per_block, uint8_t* index_k, float* index_scale,
                                                 unsigned* violations) {
  dgpp_pdl_wait();
  __shared__ float a[128];
  __shared__ float b[128];
  __shared__ float red[8];
  __shared__ int s_kmax[4];
  const int64_t r = blockIdx.x;
  const int64_t p = pos[r];
  if (p < 0 || (p + 1) % ratio != 0) return;
  const int32_t req = req_ids[r];
  const int64_t e = p / ratio;
  const int32_t blk = block_tables[int64_t(req) * blocks_per_request + e / entries_per_block];
  const int64_t slot = int64_t(blk) * entries_per_block + (e % entries_per_block);
  const RingSource src{ring + int64_t(req) * slots * 2 * wide, p, ratio, dim, wide, slots, overlap};
  publish_index_entry(src, e * ratio, norm_w, eps, inv_freq, index_k + slot * kDsv4IndexDim, index_scale + slot,
                      violations, a, b, red, s_kmax);
}

__global__ void comp_publish_main_prefill_kernel(const float* kv, const float* score, const float* ape,
                                                 const float* ring_req, int ratio, int dim, int wide, bool overlap,
                                                 int slots, int64_t pos0, const uint16_t* norm_w, float eps,
                                                 const float* inv_freq, const int32_t* block_table,
                                                 int entries_per_block, uint16_t* cache) {
  dgpp_pdl_wait();
  __shared__ float row[kDsv4Latent];
  __shared__ float red[8];
  const int g = blockIdx.x;
  const int64_t e = pos0 / ratio + g;
  const int64_t slot = int64_t(block_table[e / entries_per_block]) * entries_per_block + (e % entries_per_block);
  const ChunkSource src{kv, score, ape, ring_req, pos0, g, ratio, dim, wide, slots, overlap};
  publish_main_entry(src, e * ratio, norm_w, eps, inv_freq, cache + slot * kDsv4Latent, row, red);
}

__global__ void comp_publish_index_prefill_kernel(const float* kv, const float* score, const float* ape,
                                                  const float* ring_req, int ratio, int dim, int wide, bool overlap,
                                                  int slots, int64_t pos0, const uint16_t* norm_w, float eps,
                                                  const float* inv_freq, const int32_t* block_table,
                                                  int entries_per_block, uint8_t* index_k, float* index_scale,
                                                  unsigned* violations) {
  dgpp_pdl_wait();
  __shared__ float a[128];
  __shared__ float b[128];
  __shared__ float red[8];
  __shared__ int s_kmax[4];
  const int g = blockIdx.x;
  const int64_t e = pos0 / ratio + g;
  const int64_t slot = int64_t(block_table[e / entries_per_block]) * entries_per_block + (e % entries_per_block);
  const ChunkSource src{kv, score, ape, ring_req, pos0, g, ratio, dim, wide, slots, overlap};
  publish_index_entry(src, e * ratio, norm_w, eps, inv_freq, index_k + slot * kDsv4IndexDim, index_scale + slot,
                      violations, a, b, red, s_kmax);
}

__global__ void dense_list_kernel(const int64_t* pos_sel, int stride, int32_t* list, int32_t* counts) {
  dgpp_pdl_wait();
  const int r = blockIdx.x;
  const int64_t v = pos_sel[r] + 1;
  const int n = v < 0 ? 0 : (v > stride ? stride : int(v));
  for (int k = threadIdx.x; k < stride; k += blockDim.x) list[int64_t(r) * stride + k] = k < n ? k : -1;
  if (threadIdx.x == 0) counts[r] = n;
}

__global__ void hash_bias_rows_kernel(const int64_t* tokens, const int32_t* tid2eid, int vocab, int top_k,
                                      int n_experts, float* out) {
  dgpp_pdl_wait();
  const int r = blockIdx.x;
  float* row = out + int64_t(r) * n_experts;
  for (int e = threadIdx.x; e < n_experts; e += blockDim.x) row[e] = 0.0f;
  __syncthreads();
  int64_t tok = tokens[r];
  if (tok < 0 || tok >= vocab) tok = 0;  // a padding row's token: any row of the table, never read past it
  if (int(threadIdx.x) < top_k) row[tid2eid[tok * top_k + threadIdx.x]] = 1e30f;
}

void check_geom(const Dsv4CompGeom& g, const char* who) {
  const auto fail = [&](const char* what) { throw std::invalid_argument(std::string(who) + ": " + what); };
  if (g.ratio <= 0 || g.dim <= 0 || g.wide != (g.overlap ? 2 : 1) * g.dim) fail("compressor geometry");
  if (g.slots < (g.overlap ? 2 : 1) * g.ratio) fail("the ring must hold a whole entry's slots");
}

}  // namespace

void dsv4_q_head_rmsnorm(void* q, int64_t rows, int heads, int dim, float eps, cudaStream_t stream) {
  if (rows <= 0) return;
  if (heads <= 0 || dim <= 0 || rows * heads > 0x7fffffffLL) throw std::invalid_argument("dsv4_q_head_rmsnorm: shape");
  q_head_rmsnorm_kernel<<<unsigned(rows * heads), kThreads, 0, stream>>>(static_cast<uint16_t*>(q), dim, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv4_q_head_rmsnorm_rope(void* q, int64_t rows, int heads, float eps, const int64_t* pos,
                              const float* inv_freq, cudaStream_t stream) {
  if (rows <= 0) return;
  if (heads <= 0 || rows * heads > 0x7fffffffLL || !q || !pos || !inv_freq)
    throw std::invalid_argument("dsv4_q_head_rmsnorm_rope: shape");
  q_head_rmsnorm_rope_kernel<<<unsigned(rows * heads), kThreads, 0, stream>>>(static_cast<uint16_t*>(q), heads, eps,
                                                                               pos, inv_freq);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv4_kv_tail(void* kv, const void* norm_weight, float eps, const int32_t* req_ids, const int64_t* pos,
                  const float* inv_freq, const int32_t* ring_table, int ring_slots, void* ring, int rows, int window,
                  int32_t* wlist, int32_t* wcounts, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!kv || !norm_weight || !req_ids || !pos || !inv_freq || !ring_table || !ring || ring_slots <= 0 ||
      (wlist != nullptr && (wcounts == nullptr || window <= 0 || window > ring_slots)))
    throw std::invalid_argument("dsv4_kv_tail: shape");
  kv_tail_kernel<<<unsigned(rows), kThreads, 0, stream>>>(static_cast<uint16_t*>(kv),
                                                          static_cast<const uint16_t*>(norm_weight), eps, req_ids, pos,
                                                          inv_freq, ring_table, ring_slots, static_cast<uint8_t*>(ring),
                                                          window, wlist, wcounts);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv4_act_quant_nope(void* x, int64_t row_stride, int64_t rows, const int64_t* pos, cudaStream_t stream) {
  if (rows <= 0) return;
  if (row_stride < kDsv4Latent || rows > 0x7fffffffLL) throw std::invalid_argument("dsv4_act_quant_nope: shape");
  act_quant_nope_kernel<<<unsigned(rows), kThreads, 0, stream>>>(static_cast<uint16_t*>(x), row_stride, pos);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv4_hadamard128(void* x, int64_t rows, cudaStream_t stream) {
  if (rows <= 0) return;
  if (rows > 0x7fffffffLL) throw std::invalid_argument("dsv4_hadamard128: rows");
  hadamard128_kernel<<<unsigned(rows), 128, 0, stream>>>(static_cast<uint16_t*>(x));
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv4_fold_weights(const void* w_bf16, const float* q_scale, float* out, int64_t n, float scale,
                       cudaStream_t stream) {
  if (n <= 0) return;
  fold_weights_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(static_cast<const uint16_t*>(w_bf16), q_scale,
                                                                     out, n, scale);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv4_comp_ring_write(const float* kv, const float* score, const float* ape, const int32_t* req_ids,
                          const int64_t* pos, int rows, const Dsv4CompGeom& g, float* ring, cudaStream_t stream) {
  if (rows <= 0) return;
  check_geom(g, "dsv4_comp_ring_write");
  comp_ring_write_kernel<<<unsigned(rows), kThreads, 0, stream>>>(kv, score, ape, req_ids, pos, g.wide, g.ratio,
                                                                  g.slots, ring);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv4_comp_publish_main_decode(const float* ring, const Dsv4CompGeom& g, const int32_t* req_ids,
                                   const int64_t* pos, int rows, const void* norm_w, float eps, const float* inv_freq,
                                   const int32_t* block_tables, int blocks_per_request, int entries_per_block,
                                   void* cache, cudaStream_t stream) {
  if (rows <= 0) return;
  check_geom(g, "dsv4_comp_publish_main_decode");
  if (g.dim != kDsv4Latent) throw std::invalid_argument("dsv4_comp_publish_main_decode: the main entry is 512 wide");
  comp_publish_main_decode_kernel<<<unsigned(rows), kThreads, 0, stream>>>(
      ring, g.ratio, g.dim, g.wide, g.overlap, g.slots, req_ids, pos, static_cast<const uint16_t*>(norm_w), eps,
      inv_freq, block_tables, blocks_per_request, entries_per_block, static_cast<uint16_t*>(cache));
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv4_u_ring_write(const void* hidden_rows, size_t hidden_stride, const int32_t* req_ids, const int64_t* pos,
                       int rows, int hidden, int slots, void* ring, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!hidden_rows || !req_ids || !pos || !ring || hidden <= 0 || hidden % 8 != 0 || slots <= 0 ||
      hidden_stride % 8 != 0 || (reinterpret_cast<uintptr_t>(hidden_rows) & 15u) != 0 ||
      (reinterpret_cast<uintptr_t>(ring) & 15u) != 0)
    throw std::invalid_argument("dsv4_u_ring_write: aligned rows of a multiple of 8 elements");
  u_ring_write_kernel<<<rows, kThreads, 0, stream>>>(static_cast<const uint16_t*>(hidden_rows), hidden_stride, req_ids,
                                                     pos, hidden / 8, hidden, slots, static_cast<uint16_t*>(ring));
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv4_comp_publish_main_lazy(const float* kv, const float* score, const float* ape, const Dsv4CompGeom& g,
                                 const int32_t* req_ids, const int64_t* pos, int rows, const void* norm_w, float eps,
                                 const float* inv_freq, const int32_t* block_tables, int blocks_per_request,
                                 int entries_per_block, void* cache, cudaStream_t stream) {
  if (rows <= 0) return;
  check_geom(g, "dsv4_comp_publish_main_lazy");
  if (g.overlap || g.dim != kDsv4Latent) throw std::invalid_argument("dsv4_comp_publish_main_lazy: a plain 512-wide compressor");
  if (!kv || !score || !ape || !req_ids || !pos || !norm_w || !inv_freq || !block_tables || !cache)
    throw std::invalid_argument("dsv4_comp_publish_main_lazy: null pointer");
  comp_publish_main_lazy_kernel<<<rows, kThreads, 0, stream>>>(kv, score, ape, g.ratio, g.dim, g.wide, req_ids, pos,
                                                               static_cast<const uint16_t*>(norm_w), eps, inv_freq,
                                                               block_tables, blocks_per_request, entries_per_block,
                                                               static_cast<uint16_t*>(cache));
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv4_comp_publish_index_decode(const float* ring, const Dsv4CompGeom& g, const int32_t* req_ids,
                                    const int64_t* pos, int rows, const void* norm_w, float eps,
                                    const float* inv_freq, const int32_t* block_tables, int blocks_per_request,
                                    int entries_per_block, void* index_k, float* index_scale, unsigned* violations,
                                    cudaStream_t stream) {
  if (rows <= 0) return;
  check_geom(g, "dsv4_comp_publish_index_decode");
  if (g.dim != kDsv4IndexDim) throw std::invalid_argument("dsv4_comp_publish_index_decode: the index key is 128 wide");
  comp_publish_index_decode_kernel<<<unsigned(rows), 128, 0, stream>>>(
      ring, g.ratio, g.dim, g.wide, g.overlap, g.slots, req_ids, pos, static_cast<const uint16_t*>(norm_w), eps,
      inv_freq, block_tables, blocks_per_request, entries_per_block, static_cast<uint8_t*>(index_k), index_scale,
      violations);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv4_comp_publish_main_prefill(const float* kv, const float* score, const float* ape, const float* ring_req,
                                    const Dsv4CompGeom& g, int64_t pos0, int T, const void* norm_w, float eps,
                                    const float* inv_freq, const int32_t* block_table, int entries_per_block,
                                    void* cache, cudaStream_t stream) {
  check_geom(g, "dsv4_comp_publish_main_prefill");
  if (g.dim != kDsv4Latent) throw std::invalid_argument("dsv4_comp_publish_main_prefill: the main entry is 512 wide");
  if (pos0 < 0 || pos0 % g.ratio != 0) throw std::invalid_argument("dsv4_comp_publish_main_prefill: pos0 on a group");
  const int n = T / g.ratio;
  if (n <= 0) return;
  comp_publish_main_prefill_kernel<<<unsigned(n), kThreads, 0, stream>>>(
      kv, score, ape, ring_req, g.ratio, g.dim, g.wide, g.overlap, g.slots, pos0,
      static_cast<const uint16_t*>(norm_w), eps, inv_freq, block_table, entries_per_block,
      static_cast<uint16_t*>(cache));
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv4_comp_publish_index_prefill(const float* kv, const float* score, const float* ape, const float* ring_req,
                                     const Dsv4CompGeom& g, int64_t pos0, int T, const void* norm_w, float eps,
                                     const float* inv_freq, const int32_t* block_table, int entries_per_block,
                                     void* index_k, float* index_scale, unsigned* violations, cudaStream_t stream) {
  check_geom(g, "dsv4_comp_publish_index_prefill");
  if (g.dim != kDsv4IndexDim) throw std::invalid_argument("dsv4_comp_publish_index_prefill: the index key is 128 wide");
  if (pos0 < 0 || pos0 % g.ratio != 0) throw std::invalid_argument("dsv4_comp_publish_index_prefill: pos0 on a group");
  const int n = T / g.ratio;
  if (n <= 0) return;
  comp_publish_index_prefill_kernel<<<unsigned(n), 128, 0, stream>>>(
      kv, score, ape, ring_req, g.ratio, g.dim, g.wide, g.overlap, g.slots, pos0,
      static_cast<const uint16_t*>(norm_w), eps, inv_freq, block_table, entries_per_block,
      static_cast<uint8_t*>(index_k), index_scale, violations);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv4_dense_list(const int64_t* pos_sel, int rows, int stride, int32_t* list, int32_t* counts,
                     cudaStream_t stream) {
  if (rows <= 0) return;
  if (stride <= 0) throw std::invalid_argument("dsv4_dense_list: stride");
  dense_list_kernel<<<unsigned(rows), kThreads, 0, stream>>>(pos_sel, stride, list, counts);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv4_hash_bias_rows(const int64_t* tokens, int rows, const int32_t* tid2eid, int vocab, int top_k,
                         int n_experts, float* out, cudaStream_t stream) {
  if (rows <= 0) return;
  if (top_k <= 0 || top_k > 32 || n_experts <= 0 || vocab <= 0) throw std::invalid_argument("dsv4_hash_bias_rows: shape");
  hash_bias_rows_kernel<<<unsigned(rows), kThreads, 0, stream>>>(tokens, tid2eid, vocab, top_k, n_experts, out);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
