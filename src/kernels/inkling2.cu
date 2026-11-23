#include "kernels/inkling2.hpp"

#include <cfloat>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/pdl.cuh"

namespace dgpp {
namespace {

constexpr int kThreads = 256;

// ---- unfused FP32 steps shared by the device and host quantizers -----------------
// The released quantizers run without FMA fusion. These keep every product,
// quotient and difference a separate rounded step on both sides.
DGPP_HD inline float q_mul(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fmul_rn(a, b);
#else
  volatile float r = a * b;
  return r;
#endif
}
DGPP_HD inline float q_div(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fdiv_rn(a, b);
#else
  volatile float r = a / b;
  return r;
#endif
}
DGPP_HD inline float q_add(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fadd_rn(a, b);
#else
  volatile float r = a + b;
  return r;
#endif
}
DGPP_HD inline float q_sub(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fsub_rn(a, b);
#else
  volatile float r = a - b;
  return r;
#endif
}
DGPP_HD inline float q_min(float a, float b) { return a < b ? a : b; }
DGPP_HD inline float q_max(float a, float b) { return a > b ? a : b; }
DGPP_HD inline float q_abs(float a) { return a < 0.0f ? -a : (a == 0.0f ? 0.0f : a); }

DGPP_HD inline float round_bf16(float v) { return bf16_bits_to_float(float_to_bf16_bits(v)); }

// E2M1 round to nearest even with saturation, as a value.
DGPP_HD inline float e2m1_value(float x) {
  const float a = q_abs(x);
  const float v = a <= 0.25f ? 0.0f
                  : a < 0.75f ? 0.5f
                  : a <= 1.25f ? 1.0f
                  : a < 1.75f ? 1.5f
                  : a <= 2.5f ? 2.0f
                  : a < 3.5f ? 3.0f
                  : a <= 5.0f ? 4.0f
                  : 6.0f;
  return x < 0.0f ? -v : v;
}

struct Fp4RowScales {
  float encoder, decoder, outer;
};
DGPP_HD inline Fp4RowScales fp4_row_scales(float amax) {
  Fp4RowScales s;
  s.encoder = q_min(q_div(1536.0f, amax), FLT_MAX);
  s.decoder = q_div(1.0f, s.encoder);
  s.outer = q_mul(amax, 1.0f / 1536.0f);
  return s;
}

// One 16-value block: both candidates, the pairwise error sums, the choice.
// `values` are the chosen E2M1 values, `scale_bits` the chosen E4M3 block
// scale, `inner` their exact products (any of the three may be null).
DGPP_HD inline void fp4_block_full(const float* x, const Fp4RowScales& row, float* inner, float* values,
                                   uint8_t* scale_bits) {
  float block_max = 0.0f;
  for (int i = 0; i < 16; ++i) block_max = q_max(block_max, q_abs(x[i]));
  float cand[2][16], val[2][16];
  float err[2];
  uint8_t bits[2];
  const float factor[2] = {1.0f / 6.0f, 0.25f};
  for (int c = 0; c < 2; ++c) {
    const float raw = q_min(q_mul(block_max, q_mul(row.encoder, factor[c])), 448.0f);
    bits[c] = float_to_fp8_e4m3_bits(raw);
    const float scale = fp8_e4m3_bits_to_float(bits[c]);
    const float inverse = q_min(q_div(1.0f, q_mul(scale, row.decoder)), FLT_MAX);
    float e[16];
    for (int i = 0; i < 16; ++i) {
      const float value = e2m1_value(q_mul(x[i], inverse));
      const float in = q_mul(value, scale);  // exact: 2 by 4 significant bits
      val[c][i] = value;
      cand[c][i] = in;
      e[i] = q_abs(q_sub(q_mul(in, row.outer), x[i]));
    }
    for (int width = 8; width >= 1; width >>= 1)
      for (int j = 0; j < width; ++j) e[j] = q_add(e[2 * j], e[2 * j + 1]);
    err[c] = e[0];
  }
  const int pick = err[1] < err[0] ? 1 : 0;
  for (int i = 0; i < 16; ++i) {
    if (inner != nullptr) inner[i] = cand[pick][i];
    if (values != nullptr) values[i] = val[pick][i];
  }
  if (scale_bits != nullptr) *scale_bits = bits[pick];
}

DGPP_HD inline void fp4_block(const float* x, const Fp4RowScales& row, float* inner) {
  fp4_block_full(x, row, inner, nullptr, nullptr);
}

// An E2M1 value's 4-bit code: the sign in bit 3 over the magnitude's index.
DGPP_HD inline uint32_t e2m1_code(float value) {
  const float a = q_abs(value);
  const uint32_t m = a == 0.0f ? 0u : a == 0.5f ? 1u : a == 1.0f ? 2u : a == 1.5f ? 3u : a == 2.0f ? 4u : a == 3.0f ? 5u
                                                                                           : a == 4.0f ? 6u : 7u;
  return value < 0.0f ? (m | 8u) : m;
}

// Sixteen E2M1 values as eight bytes, the even element in the low nibble.
DGPP_HD inline void fp4_pack(const float* values, uint8_t* bytes) {
  for (int i = 0; i < 8; ++i)
    bytes[i] = static_cast<uint8_t>(e2m1_code(values[2 * i]) | (e2m1_code(values[2 * i + 1]) << 4));
}

// One 32-value MXFP8 block, dequantized.
DGPP_HD inline void mx8_block(const float* x, float* out) {
  float amax = 0.0f;
  for (int i = 0; i < 32; ++i) amax = q_max(amax, q_abs(x[i]));
  const float min_scale = 5.877471754111438e-39f;  // 2^-127
  const float raw = q_mul(amax, 1.0f / 448.0f);
  uint32_t bits;
#ifdef __CUDA_ARCH__
  bits = __float_as_uint(raw);
#else
  std::memcpy(&bits, &raw, 4);
#endif
  int exponent = static_cast<int>(((bits + 0x007FFFFFu) & 0x7F800000u) >> 23);
  exponent = raw <= min_scale ? 0 : (exponent < 254 ? exponent : 254);
  float scale = min_scale;
  if (exponent > 0) {
    const uint32_t sb = static_cast<uint32_t>(exponent) << 23;
#ifdef __CUDA_ARCH__
    scale = __uint_as_float(sb);
#else
    std::memcpy(&scale, &sb, 4);
#endif
  }
  const float inverse = q_div(1.0f, scale);  // a power of two: exact
  for (int i = 0; i < 32; ++i) {
    const float v = q_min(q_max(q_mul(x[i], inverse), -448.0f), 448.0f);
    out[i] = q_mul(fp8_e4m3_bits_to_float(float_to_fp8_e4m3_bits(v)), scale);  // exact
  }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(~0u, v, off);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, off));
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

// ---- the mix -----------------------------------------------------------------------
struct MixDev {
  const uint16_t* stack[kInk2MaxSources];
  const uint16_t* last;
  const uint16_t* addend;
  const float* queries;
  const uint16_t* norm_w[kInk2MaxQueries];
  uint16_t* out_bf16[kInk2MaxQueries];
  float* out_f32[kInk2MaxQueries];
  float norm_eps;
  int n_stack;
  int hidden;
};

__device__ __forceinline__ float mix_source(const MixDev& a, int n, int64_t at) {
  if (n < a.n_stack) return bf16_bits_to_float(a.stack[n][at]);
  float x = bf16_bits_to_float(a.last[at]);
  if (a.addend != nullptr) x += bf16_bits_to_float(a.addend[at]);
  return x;
}

// One block per (row, query).
__global__ void __launch_bounds__(kThreads) mix_kernel(MixDev a) {
  dgpp_pdl_wait();
  extern __shared__ float mixed[];  // [hidden]
  __shared__ float slots[8];
  __shared__ float coef[kInk2MaxSources + 1];
  __shared__ float logit[kInk2MaxSources + 1];
  const int row = static_cast<int>(blockIdx.x), q = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int H = a.hidden, sources = a.n_stack + 1;
  const int64_t off = static_cast<int64_t>(row) * H;
  const float* query = a.queries + static_cast<int64_t>(q) * H;
  for (int n = 0; n < sources; ++n) {
    float ss = 0.0f, dot = 0.0f;
    for (int d = tid; d < H; d += kThreads) {
      const float x = mix_source(a, n, off + d);
      ss = fmaf(x, x, ss);
      dot = fmaf(query[d], x, dot);
    }
    const float ss_all = block_sum(ss, slots);
    const float dot_all = block_sum(dot, slots);
    if (tid == 0) {
      const float rstd = 1.0f / sqrtf(ss_all / static_cast<float>(H) + kInk2MixEps);
      coef[n] = rstd;
      logit[n] = dot_all * rstd;
    }
  }
  __syncthreads();
  if (tid == 0) {
    float top = logit[0];
    for (int n = 1; n < sources; ++n) top = fmaxf(top, logit[n]);
    float p[kInk2MaxSources + 1];
    float total = 0.0f;
    for (int n = 0; n < sources; ++n) {
      p[n] = expf(logit[n] - top);
      total += p[n];
    }
    const float inv = 1.0f / total;
    for (int n = 0; n < sources; ++n) coef[n] = (p[n] * inv) * coef[n];
  }
  __syncthreads();
  float ss = 0.0f;
  for (int d = tid; d < H; d += kThreads) {
    float acc = 0.0f;
    for (int n = 0; n < sources; ++n) acc = fmaf(coef[n], mix_source(a, n, off + d), acc);
    mixed[d] = acc;
    ss = fmaf(acc, acc, ss);
  }
  if (a.norm_w[q] == nullptr) {
    float* out = a.out_f32[q];
    for (int d = tid; d < H; d += kThreads) out[off + d] = mixed[d];
    return;
  }
  const float rstd = 1.0f / sqrtf(block_sum(ss, slots) / static_cast<float>(H) + a.norm_eps);
  const uint16_t* w = a.norm_w[q];
  uint16_t* out = a.out_bf16[q];
  for (int d = tid; d < H; d += kThreads)
    out[off + d] = float_to_bf16_bits((mixed[d] * rstd) * bf16_bits_to_float(w[d]));
}

// ---- the controllers ------------------------------------------------------------
__global__ void __launch_bounds__(kThreads) pair_norm_kernel(const float* __restrict__ dest,
                                                             const float* __restrict__ message,
                                                             uint16_t* __restrict__ pair, int hidden) {
  dgpp_pdl_wait();
  __shared__ float slots[8];
  const int row = static_cast<int>(blockIdx.x), tid = static_cast<int>(threadIdx.x);
  const int64_t off = static_cast<int64_t>(row) * hidden;
  float sd = 0.0f, sm = 0.0f;
  for (int d = tid; d < hidden; d += kThreads) {
    sd = fmaf(dest[off + d], dest[off + d], sd);
    sm = fmaf(message[off + d], message[off + d], sm);
  }
  const float rd = 1.0f / sqrtf(block_sum(sd, slots) / static_cast<float>(hidden) + kInk2MixEps);
  const float rm = 1.0f / sqrtf(block_sum(sm, slots) / static_cast<float>(hidden) + kInk2MixEps);
  uint16_t* out = pair + static_cast<int64_t>(row) * 2 * hidden;
  for (int d = tid; d < hidden; d += kThreads) {
    out[d] = float_to_bf16_bits(dest[off + d] * rd);
    out[hidden + d] = float_to_bf16_bits(message[off + d] * rm);
  }
}

__global__ void __launch_bounds__(kThreads) blend_norm_kernel(const float* __restrict__ dest,
                                                              const float* __restrict__ message,
                                                              const float* __restrict__ gain_forget,
                                                              const uint16_t* __restrict__ norm_w,
                                                              uint16_t* __restrict__ out, int hidden, float norm_eps) {
  dgpp_pdl_wait();
  extern __shared__ float blended[];  // [hidden]
  __shared__ float slots[8];
  const int row = static_cast<int>(blockIdx.x), tid = static_cast<int>(threadIdx.x);
  const int64_t off = static_cast<int64_t>(row) * hidden;
  const float* gf = gain_forget + static_cast<int64_t>(row) * 2 * hidden;
  float ss = 0.0f;
  for (int d = tid; d < hidden; d += kThreads) {
    const float keep = 1.0f - tanhf(gf[hidden + d]);
    const float v = keep * dest[off + d] + tanhf(gf[d]) * message[off + d];
    blended[d] = v;
    ss = fmaf(v, v, ss);
  }
  const float rstd = 1.0f / sqrtf(block_sum(ss, slots) / static_cast<float>(hidden) + norm_eps);
  for (int d = tid; d < hidden; d += kThreads)
    out[off + d] = float_to_bf16_bits((blended[d] * rstd) * bf16_bits_to_float(norm_w[d]));
}

// ---- the convolution ----------------------------------------------------------------
struct ConvDev {
  const uint16_t* x;
  int64_t x_stride, x_offset;
  const float* taps;
  const uint16_t* dyn;
  int64_t dyn_stride, dyn_offset;
  int dyn_heads;
  float scale;
  uint16_t* ring;
  int64_t ring_request_stride, ring_slot_stride, ring_offset;
  const int32_t* req_ids;
  const int64_t* pos;
  int rows, channels;
  bool store_inline;
  uint16_t* out;
  int64_t out_stride, out_offset;
  const uint16_t* addend;
  const uint16_t* residual;
};

// The stream's input at `lag` positions before row `row` (lag >= 1).
__device__ __forceinline__ float conv_lag(const ConvDev& a, int row, int c, int lag, int32_t req, int64_t p) {
  const int64_t q = p - lag;
  if (q < 0) return 0.0f;
  const int rr = row - lag;
  if (rr >= 0 && a.req_ids[rr] == req && a.pos[rr] == q)
    return bf16_bits_to_float(a.x[static_cast<int64_t>(rr) * a.x_stride + a.x_offset + c]);
  return bf16_bits_to_float(a.ring[static_cast<int64_t>(req) * a.ring_request_stride +
                                   (q % kInk2RingSlots) * a.ring_slot_stride + a.ring_offset + c]);
}

// One element per thread, grid-strided.
__global__ void __launch_bounds__(kThreads) conv_kernel(ConvDev a) {
  dgpp_pdl_wait();
  const int64_t total = static_cast<int64_t>(a.rows) * a.channels;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
  const int per_head = a.dyn != nullptr ? a.channels / a.dyn_heads : 1;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; i < total; i += stride) {
    const int row = static_cast<int>(i / a.channels);
    const int c = static_cast<int>(i - static_cast<int64_t>(row) * a.channels);
    uint16_t* out = a.out + static_cast<int64_t>(row) * a.out_stride + a.out_offset + c;
    const int64_t p = a.pos[row];
    if (p < 0) {
      *out = 0;
      continue;
    }
    const int32_t req = a.req_ids[row];
    const uint16_t x_bits = a.x[static_cast<int64_t>(row) * a.x_stride + a.x_offset + c];
    const float x0 = bf16_bits_to_float(x_bits);
    float tap[kInk2ConvTaps];
#pragma unroll
    for (int lag = 0; lag < kInk2ConvTaps; ++lag) tap[lag] = a.taps[static_cast<int64_t>(c) * kInk2ConvTaps + lag];
    if (a.dyn != nullptr) {
      const uint16_t* dyn = a.dyn + static_cast<int64_t>(row) * a.dyn_stride + a.dyn_offset +
                            static_cast<int64_t>(c / per_head) * kInk2ConvTaps;
#pragma unroll
      for (int lag = 0; lag < kInk2ConvTaps; ++lag) tap[lag] += bf16_bits_to_float(dyn[lag]);
    }
    // The oldest tap first, the row itself last.
    float acc = 0.0f;
#pragma unroll
    for (int lag = kInk2ConvTaps - 1; lag >= 1; --lag) acc = fmaf(tap[lag], conv_lag(a, row, c, lag, req, p), acc);
    acc = fmaf(tap[0], x0, acc);
    const float y = round_bf16((acc + x0) * a.scale);
    if (a.addend != nullptr) {
      const int64_t at = static_cast<int64_t>(row) * a.channels + c;
      float base = bf16_bits_to_float(a.addend[at]);
      if (a.residual != nullptr) base = bf16_bits_to_float(a.residual[at]) + base;
      *out = float_to_bf16_bits(base + y);
    } else {
      *out = float_to_bf16_bits(y);
    }
    if (a.store_inline)
      a.ring[static_cast<int64_t>(req) * a.ring_request_stride + (p % kInk2RingSlots) * a.ring_slot_stride +
             a.ring_offset + c] = x_bits;
  }
}

__global__ void __launch_bounds__(kThreads) ring_store_kernel(const uint16_t* __restrict__ x, int64_t x_stride,
                                                              int64_t x_offset, uint16_t* __restrict__ ring,
                                                              int64_t ring_request_stride, int64_t ring_slot_stride,
                                                              int64_t ring_offset,
                                                              const int32_t* __restrict__ req_ids,
                                                              const int64_t* __restrict__ pos, int rows,
                                                              int channels) {
  dgpp_pdl_wait();
  const int64_t total = static_cast<int64_t>(rows) * channels;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; i < total; i += stride) {
    const int row = static_cast<int>(i / channels);
    const int c = static_cast<int>(i - static_cast<int64_t>(row) * channels);
    const int64_t p = pos[row];
    if (p < 0) continue;
    const int32_t req = req_ids[row];
    // A later row of the same run takes this slot: it is the slot's writer.
    const int later = row + kInk2RingSlots;
    if (later < rows && req_ids[later] == req && pos[later] == p + kInk2RingSlots) continue;
    ring[static_cast<int64_t>(req) * ring_request_stride + (p % kInk2RingSlots) * ring_slot_stride + ring_offset +
         c] = x[static_cast<int64_t>(row) * x_stride + x_offset + c];
  }
}

// The snapshot's rows after its 16-byte header, newest position first.
__global__ void __launch_bounds__(kThreads) ring_snapshot_kernel(const uint16_t* __restrict__ ring,
                                                                 int64_t slot_stride, int64_t position,
                                                                 uint8_t* __restrict__ snapshot) {
  dgpp_pdl_wait();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int64_t* header = reinterpret_cast<int64_t*>(snapshot);
    header[0] = position;
    header[1] = 0;
  }
  uint16_t* rows = reinterpret_cast<uint16_t*>(snapshot + 16);
  const int64_t total = 3 * slot_stride;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; i < total; i += stride) {
    const int64_t lag = i / slot_stride, c = i - lag * slot_stride;
    const int64_t q = position - 1 - lag;
    rows[i] = q >= 0 ? ring[(q % kInk2RingSlots) * slot_stride + c] : static_cast<uint16_t>(0);
  }
}

__global__ void __launch_bounds__(kThreads) ring_restore_kernel(const uint8_t* __restrict__ snapshot,
                                                                int64_t slot_stride, uint16_t* __restrict__ ring) {
  dgpp_pdl_wait();
  const int64_t position = reinterpret_cast<const int64_t*>(snapshot)[0];
  const uint16_t* rows = reinterpret_cast<const uint16_t*>(snapshot + 16);
  const int64_t total = 3 * slot_stride;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; i < total; i += stride) {
    const int64_t lag = i / slot_stride, c = i - lag * slot_stride;
    const int64_t q = position - 1 - lag;
    if (q >= 0) ring[(q % kInk2RingSlots) * slot_stride + c] = rows[i];
  }
}

// ---- the experts -------------------------------------------------------------------
__global__ void __launch_bounds__(kThreads) slot_rows_kernel(const int32_t* __restrict__ slot_row, int slots,
                                                             int32_t* __restrict__ row_slot) {
  dgpp_pdl_wait();
  const int s = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
  if (s < slots) row_slot[slot_row[s]] = s;
}

__device__ __forceinline__ float softcap(float x, float cap) { return cap > 0.0f ? cap * tanhf(x / cap) : x; }

__global__ void __launch_bounds__(kThreads) swiglu_kernel(const float* __restrict__ gate_up, int64_t gate_up_stride,
                                                          int rows, int inter, const float* __restrict__ row_scale,
                                                          const int32_t* __restrict__ row_token,
                                                          const float* __restrict__ slot_w,
                                                          const int32_t* __restrict__ row_slot, float gate_cap,
                                                          float up_cap, uint16_t* __restrict__ act,
                                                          int64_t act_stride) {
  dgpp_pdl_wait();
  const int64_t total = static_cast<int64_t>(rows) * inter;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; idx < total; idx += stride) {
    const int row = static_cast<int>(idx / inter);
    const int i = static_cast<int>(idx - static_cast<int64_t>(row) * inter);
    const float s = row_scale != nullptr ? row_scale[row_token != nullptr ? row_token[row] : row] : 1.0f;
    const float* gu = gate_up + static_cast<int64_t>(row) * gate_up_stride + 2 * i;
    const float g = round_bf16(gu[0] * s);
    const float u = round_bf16(gu[1] * s);
    const float sig = 1.0f / (1.0f + expf(-g));
    const float w = slot_w[row_slot[row]];
    act[static_cast<int64_t>(row) * act_stride + i] =
        float_to_bf16_bits(((softcap(g, gate_cap) * sig) * softcap(u, up_cap)) * w);
  }
}

// One block per row. A thread quantizes the 16-value blocks it owns.
__global__ void __launch_bounds__(kThreads) quantize_nvfp4_kernel(const uint16_t* __restrict__ x, int64_t x_stride,
                                                                  int k, uint16_t* __restrict__ inner,
                                                                  int64_t inner_stride, float* __restrict__ outer,
                                                                  const int32_t* __restrict__ scale_index) {
  dgpp_pdl_wait();
  __shared__ float slots[8];
  const int row = static_cast<int>(blockIdx.x), tid = static_cast<int>(threadIdx.x);
  const uint16_t* src = x + static_cast<int64_t>(row) * x_stride;
  const int groups = k / 16;
  float amax = 0.0f;
  for (int g = tid; g < groups; g += kThreads)
#pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(bf16_bits_to_float(src[g * 16 + i])));
  amax = warp_max(amax);
  if ((tid & 31) == 0) slots[tid >> 5] = amax;
  __syncthreads();
  amax = 0.0f;
  for (int w = 0; w < kThreads / 32; ++w) amax = fmaxf(amax, slots[w]);
  const Fp4RowScales scales = fp4_row_scales(amax);
  if (tid == 0) outer[scale_index != nullptr ? scale_index[row] : row] = scales.outer;
  uint16_t* dst = inner + static_cast<int64_t>(row) * inner_stride;
  for (int g = tid; g < groups; g += kThreads) {
    float v[16], in[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = bf16_bits_to_float(src[g * 16 + i]);
    fp4_block(v, scales, in);
#pragma unroll
    for (int i = 0; i < 16; ++i) dst[g * 16 + i] = float_to_bf16_bits(in[i]);
  }
}

// The same quantizer writing the format's own bytes: the E2M1 codes, the
// E4M3 block scales and the row's outer scale, for the tensor cores'
// block-scaled form. One block per row.
__global__ void __launch_bounds__(kThreads) quantize_nvfp4_codes_kernel(const uint16_t* __restrict__ x, int64_t x_stride,
                                                                        int k, uint8_t* __restrict__ codes,
                                                                        uint8_t* __restrict__ scales,
                                                                        int64_t scale_stride, float* __restrict__ gs) {
  dgpp_pdl_wait();
  __shared__ float slots[8];
  const int row = static_cast<int>(blockIdx.x), tid = static_cast<int>(threadIdx.x);
  const uint16_t* src = x + static_cast<int64_t>(row) * x_stride;
  const int groups = k / 16;
  float amax = 0.0f;
  for (int g = tid; g < groups; g += kThreads)
#pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(bf16_bits_to_float(src[g * 16 + i])));
  amax = warp_max(amax);
  if ((tid & 31) == 0) slots[tid >> 5] = amax;
  __syncthreads();
  amax = 0.0f;
  for (int w = 0; w < kThreads / 32; ++w) amax = fmaxf(amax, slots[w]);
  const Fp4RowScales row_scales = fp4_row_scales(amax);
  if (tid == 0) gs[row] = row_scales.outer;
  uint8_t* cdst = codes + static_cast<int64_t>(row) * (k / 2);
  uint8_t* sdst = scales + static_cast<int64_t>(row) * scale_stride;
  for (int g = tid; g < groups; g += kThreads) {
    float v[16], values[16];
    uint8_t bytes[8], sb;
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = bf16_bits_to_float(src[g * 16 + i]);
    fp4_block_full(v, row_scales, nullptr, values, &sb);
    fp4_pack(values, bytes);
    *reinterpret_cast<uint64_t*>(cdst + g * 8) = *reinterpret_cast<const uint64_t*>(bytes);
    sdst[g] = sb;
  }
}

__device__ __forceinline__ float swiglu_value(float g, float u, float w, float gate_cap, float up_cap) {
  const float sig = 1.0f / (1.0f + expf(-g));
  return round_bf16(((softcap(g, gate_cap) * sig) * softcap(u, up_cap)) * w);
}

// The activation from BF16 gate / up rows (already at their row scale).
__global__ void __launch_bounds__(kThreads) swiglu_bf16_kernel(const uint16_t* __restrict__ gate_up,
                                                               int64_t gate_up_stride, int rows, int inter,
                                                               const float* __restrict__ slot_w,
                                                               const int32_t* __restrict__ row_slot, float gate_cap,
                                                               float up_cap, uint16_t* __restrict__ act,
                                                               int64_t act_stride) {
  dgpp_pdl_wait();
  const int64_t total = static_cast<int64_t>(rows) * inter;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; idx < total; idx += stride) {
    const int row = static_cast<int>(idx / inter);
    const int i = static_cast<int>(idx - static_cast<int64_t>(row) * inter);
    const uint32_t gu = *reinterpret_cast<const uint32_t*>(gate_up + static_cast<int64_t>(row) * gate_up_stride + 2 * i);
    const float g = bf16_bits_to_float(static_cast<uint16_t>(gu & 0xFFFFu));
    const float u = bf16_bits_to_float(static_cast<uint16_t>(gu >> 16));
    act[static_cast<int64_t>(row) * act_stride + i] =
        float_to_bf16_bits(swiglu_value(g, u, slot_w[row_slot[row]], gate_cap, up_cap));
  }
}

// The activation and its NVFP4 quantization in one pass over a routed row:
// the activation values (rounded to BF16, as the quantizer reads them) stay
// in shared memory between the row maximum and the blocks. One block per row.
__global__ void __launch_bounds__(kThreads) swiglu_quantize_nvfp4_kernel(const uint16_t* __restrict__ gate_up,
                                                                         int64_t gate_up_stride, int inter,
                                                                         const float* __restrict__ slot_w,
                                                                         const int32_t* __restrict__ row_slot,
                                                                         float gate_cap, float up_cap,
                                                                         uint8_t* __restrict__ codes,
                                                                         uint8_t* __restrict__ scales,
                                                                         int64_t scale_stride, float* __restrict__ gs) {
  dgpp_pdl_wait();
  extern __shared__ float s_act[];
  __shared__ float slots[8];
  const int row = static_cast<int>(blockIdx.x), tid = static_cast<int>(threadIdx.x);
  const uint16_t* src = gate_up + static_cast<int64_t>(row) * gate_up_stride;
  const float w = slot_w[row_slot[row]];
  float amax = 0.0f;
  for (int i = tid; i < inter; i += kThreads) {
    const uint32_t gu = *reinterpret_cast<const uint32_t*>(src + 2 * i);
    const float v = swiglu_value(bf16_bits_to_float(static_cast<uint16_t>(gu & 0xFFFFu)),
                                 bf16_bits_to_float(static_cast<uint16_t>(gu >> 16)), w, gate_cap, up_cap);
    s_act[i] = v;
    amax = fmaxf(amax, fabsf(v));
  }
  amax = warp_max(amax);
  if ((tid & 31) == 0) slots[tid >> 5] = amax;
  __syncthreads();
  amax = 0.0f;
  for (int j = 0; j < kThreads / 32; ++j) amax = fmaxf(amax, slots[j]);
  const Fp4RowScales row_scales = fp4_row_scales(amax);
  if (tid == 0) gs[row] = row_scales.outer;
  const int groups = inter / 16;
  uint8_t* cdst = codes + static_cast<int64_t>(row) * (inter / 2);
  uint8_t* sdst = scales + static_cast<int64_t>(row) * scale_stride;
  for (int g = tid; g < groups; g += kThreads) {
    float v[16], values[16];
    uint8_t bytes[8], sb;
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = s_act[g * 16 + i];
    fp4_block_full(v, row_scales, nullptr, values, &sb);
    fp4_pack(values, bytes);
    *reinterpret_cast<uint64_t*>(cdst + g * 8) = *reinterpret_cast<const uint64_t*>(bytes);
    sdst[g] = sb;
  }
}

// out[t] = bf16(sum over the token's slots, in ascending expert id, of
// bf16(down[row] * slot_scale[slot])): every expert's output is rounded at
// its row scale before the sum, as the released recipe stores it.
constexpr int kAccumMaxTopK = 16;
__global__ void __launch_bounds__(kThreads) accum_rounded_kernel(uint16_t* __restrict__ out,
                                                                 const float* __restrict__ down, int64_t down_stride,
                                                                 const int32_t* __restrict__ slot_row,
                                                                 const int32_t* __restrict__ slot_ids,
                                                                 const float* __restrict__ slot_scale, int tokens,
                                                                 int K, int hidden,
                                                                 const int64_t* __restrict__ positions) {
  dgpp_pdl_wait();
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<int64_t>(tokens) * hidden) return;
  const int t = static_cast<int>(i / hidden);
  const int h = static_cast<int>(i - static_cast<int64_t>(t) * hidden);
  // A padding row is in no expert's row list (launch_moe_segment): no
  // launch computed its gathered row.
  if (positions != nullptr && positions[t] < 0) {
    out[i] = 0;
    return;
  }
  int order[kAccumMaxTopK];
  for (int j = 0; j < K; ++j) order[j] = j;
  for (int j = 1; j < K; ++j) {
    const int cur = order[j];
    const int key = slot_ids[t * K + cur];
    int p = j - 1;
    while (p >= 0 && slot_ids[t * K + order[p]] > key) {
      order[p + 1] = order[p];
      --p;
    }
    order[p + 1] = cur;
  }
  float acc = 0.f;
  for (int j = 0; j < K; ++j) {
    const int s = t * K + order[j];
    acc = __fadd_rn(acc, round_bf16(__fmul_rn(down[static_cast<int64_t>(slot_row[s]) * down_stride + h], slot_scale[s])));
  }
  out[i] = float_to_bf16_bits(acc);
}

// One 32-value block per thread, grid-strided over (row, block).
__global__ void __launch_bounds__(kThreads) requantize_mxfp8_kernel(const uint16_t* __restrict__ x, int64_t x_stride,
                                                                    int rows, int k, uint16_t* __restrict__ out,
                                                                    int64_t out_stride) {
  dgpp_pdl_wait();
  const int groups = k / 32;
  const int64_t total = static_cast<int64_t>(rows) * groups;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; idx < total; idx += stride) {
    const int row = static_cast<int>(idx / groups);
    const int g = static_cast<int>(idx - static_cast<int64_t>(row) * groups);
    const uint16_t* src = x + static_cast<int64_t>(row) * x_stride + g * 32;
    float v[32], o[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) v[i] = bf16_bits_to_float(src[i]);
    mx8_block(v, o);
    uint16_t* dst = out + static_cast<int64_t>(row) * out_stride + g * 32;
#pragma unroll
    for (int i = 0; i < 32; ++i) dst[i] = float_to_bf16_bits(o[i]);
  }
}

__global__ void __launch_bounds__(kThreads) logits_softcap_kernel(float* __restrict__ logits, int64_t count,
                                                                  float cap) {
  dgpp_pdl_wait();
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; i < count; i += stride) {
    const float v = logits[i];
    if (isfinite(v)) logits[i] = cap * tanhf(v / cap);
  }
}

int flat_blocks(int64_t total) {
  int64_t blocks = (total + kThreads - 1) / kThreads;
  if (blocks > 65535) blocks = 65535;
  return static_cast<int>(blocks < 1 ? 1 : blocks);
}

}  // namespace

void ink2_mix(const Ink2MixArgs& args, cudaStream_t stream) {
  if (args.rows <= 0) return;
  if (args.n_stack < 0 || args.n_stack > kInk2MaxSources) throw std::invalid_argument("ink2_mix: source count");
  if (args.n_queries < 1 || args.n_queries > kInk2MaxQueries) throw std::invalid_argument("ink2_mix: query count");
  if (args.hidden <= 0 || !args.last || !args.queries) throw std::invalid_argument("ink2_mix: null buffer");
  MixDev d{};
  for (int n = 0; n < args.n_stack; ++n) {
    if (!args.stack[n]) throw std::invalid_argument("ink2_mix: null source");
    d.stack[n] = args.stack[n];
  }
  const bool normed = args.norm_w[0] != nullptr;
  for (int q = 0; q < args.n_queries; ++q) {
    if ((args.norm_w[q] != nullptr) != normed) throw std::invalid_argument("ink2_mix: norm weights are all or none");
    if (normed ? args.out_bf16[q] == nullptr : args.out_f32[q] == nullptr)
      throw std::invalid_argument("ink2_mix: null output");
    d.norm_w[q] = args.norm_w[q];
    d.out_bf16[q] = args.out_bf16[q];
    d.out_f32[q] = args.out_f32[q];
  }
  d.last = args.last;
  d.addend = args.addend;
  d.queries = args.queries;
  d.norm_eps = args.norm_eps;
  d.n_stack = args.n_stack;
  d.hidden = args.hidden;
  const dim3 grid(static_cast<unsigned>(args.rows), static_cast<unsigned>(args.n_queries));
  mix_kernel<<<grid, kThreads, static_cast<size_t>(args.hidden) * 4, stream>>>(d);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_pair_norm(const float* dest, const float* message, uint16_t* pair, int rows, int hidden,
                    cudaStream_t stream) {
  if (rows <= 0) return;
  if (!dest || !message || !pair || hidden <= 0) throw std::invalid_argument("ink2_pair_norm: null buffer");
  pair_norm_kernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(dest, message, pair, hidden);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_blend_norm(const float* dest, const float* message, const float* gain_forget, const uint16_t* norm_w,
                     uint16_t* out, int rows, int hidden, float norm_eps, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!dest || !message || !gain_forget || !norm_w || !out || hidden <= 0)
    throw std::invalid_argument("ink2_blend_norm: null buffer");
  blend_norm_kernel<<<static_cast<unsigned>(rows), kThreads, static_cast<size_t>(hidden) * 4, stream>>>(
      dest, message, gain_forget, norm_w, out, hidden, norm_eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_conv(const Ink2ConvArgs& args, cudaStream_t stream) {
  if (args.rows <= 0) return;
  if (!args.x || !args.taps || !args.ring || !args.req_ids || !args.pos || !args.out)
    throw std::invalid_argument("ink2_conv: null buffer");
  if (args.channels <= 0 || args.x_stride < args.channels || args.out_stride < args.channels)
    throw std::invalid_argument("ink2_conv: channel geometry");
  if (args.dyn != nullptr && (args.dyn_heads <= 0 || args.channels % args.dyn_heads != 0))
    throw std::invalid_argument("ink2_conv: dynamic taps need a head count that divides the channels");
  if (args.residual != nullptr && args.addend == nullptr)
    throw std::invalid_argument("ink2_conv: a residual needs an addend");
  if (!(args.scale > 0.0f)) throw std::invalid_argument("ink2_conv: scale");
  ConvDev d{args.x, args.x_stride, args.x_offset, args.taps, args.dyn, args.dyn_stride, args.dyn_offset,
            args.dyn_heads, args.scale, args.ring, args.ring_request_stride, args.ring_slot_stride,
            args.ring_offset, args.req_ids, args.pos, args.rows, args.channels, args.store_inline, args.out,
            args.out_stride, args.out_offset, args.addend, args.residual};
  conv_kernel<<<flat_blocks(static_cast<int64_t>(args.rows) * args.channels), kThreads, 0, stream>>>(d);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_ring_store(const uint16_t* x, int64_t x_stride, int64_t x_offset, uint16_t* ring,
                     int64_t ring_request_stride, int64_t ring_slot_stride, int64_t ring_offset,
                     const int32_t* req_ids, const int64_t* pos, int rows, int channels, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!x || !ring || !req_ids || !pos || channels <= 0) throw std::invalid_argument("ink2_ring_store: null buffer");
  ring_store_kernel<<<flat_blocks(static_cast<int64_t>(rows) * channels), kThreads, 0, stream>>>(
      x, x_stride, x_offset, ring, ring_request_stride, ring_slot_stride, ring_offset, req_ids, pos, rows, channels);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_ring_snapshot(const uint16_t* ring_request, int64_t slot_stride, int64_t position, uint8_t* snapshot,
                        cudaStream_t stream) {
  if (!ring_request || !snapshot || slot_stride <= 0 || position < 0)
    throw std::invalid_argument("ink2_ring_snapshot: arguments");
  ring_snapshot_kernel<<<flat_blocks(3 * slot_stride), kThreads, 0, stream>>>(ring_request, slot_stride, position,
                                                                           snapshot);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_ring_restore(const uint8_t* snapshot, int64_t slot_stride, uint16_t* ring_request, cudaStream_t stream) {
  if (!ring_request || !snapshot || slot_stride <= 0) throw std::invalid_argument("ink2_ring_restore: arguments");
  ring_restore_kernel<<<flat_blocks(3 * slot_stride), kThreads, 0, stream>>>(snapshot, slot_stride, ring_request);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_slot_rows(const int32_t* slot_row, int slots, int32_t* row_slot, cudaStream_t stream) {
  if (slots <= 0) return;
  if (!slot_row || !row_slot) throw std::invalid_argument("ink2_slot_rows: null buffer");
  slot_rows_kernel<<<(slots + kThreads - 1) / kThreads, kThreads, 0, stream>>>(slot_row, slots, row_slot);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_swiglu(const float* gate_up, int64_t gate_up_stride, int rows, int inter, const float* row_scale,
                 const int32_t* row_token, const float* slot_w, const int32_t* row_slot, float gate_cap,
                 float up_cap, uint16_t* act, int64_t act_stride, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!gate_up || !slot_w || !row_slot || !act) throw std::invalid_argument("ink2_swiglu: null buffer");
  if (inter <= 0 || gate_up_stride < 2 * static_cast<int64_t>(inter) || act_stride < inter)
    throw std::invalid_argument("ink2_swiglu: geometry");
  if (gate_cap < 0.0f || up_cap < 0.0f) throw std::invalid_argument("ink2_swiglu: softcaps");
  swiglu_kernel<<<flat_blocks(static_cast<int64_t>(rows) * inter), kThreads, 0, stream>>>(
      gate_up, gate_up_stride, rows, inter, row_scale, row_token, slot_w, row_slot, gate_cap, up_cap, act,
      act_stride);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_quantize_nvfp4(const uint16_t* x, int64_t x_stride, int rows, int k, uint16_t* inner,
                         int64_t inner_stride, float* outer, const int32_t* scale_index, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!x || !inner || !outer) throw std::invalid_argument("ink2_quantize_nvfp4: null buffer");
  if (k <= 0 || k % 16 != 0 || x_stride < k || inner_stride < k)
    throw std::invalid_argument("ink2_quantize_nvfp4: k must be a positive multiple of 16 within the strides");
  quantize_nvfp4_kernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(x, x_stride, k, inner, inner_stride,
                                                                            outer, scale_index);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_requantize_mxfp8(const uint16_t* x, int64_t x_stride, int rows, int k, uint16_t* out, int64_t out_stride,
                           cudaStream_t stream) {
  if (rows <= 0) return;
  if (!x || !out) throw std::invalid_argument("ink2_requantize_mxfp8: null buffer");
  if (k <= 0 || k % 32 != 0 || x_stride < k || out_stride < k)
    throw std::invalid_argument("ink2_requantize_mxfp8: k must be a positive multiple of 32 within the strides");
  requantize_mxfp8_kernel<<<flat_blocks(static_cast<int64_t>(rows) * (k / 32)), kThreads, 0, stream>>>(
      x, x_stride, rows, k, out, out_stride);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_logits_softcap(float* logits, int64_t count, float cap, cudaStream_t stream) {
  if (count <= 0) return;
  if (!logits) throw std::invalid_argument("ink2_logits_softcap: null buffer");
  if (!(cap > 0.0f)) throw std::invalid_argument("ink2_logits_softcap: cap must be positive");
  logits_softcap_kernel<<<flat_blocks(count), kThreads, 0, stream>>>(logits, count, cap);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_quantize_nvfp4_host(const uint16_t* x, int k, uint16_t* inner, float* outer) {
  if (k <= 0 || k % 16 != 0) throw std::invalid_argument("ink2_quantize_nvfp4_host: k");
  float amax = 0.0f;
  for (int i = 0; i < k; ++i) amax = q_max(amax, q_abs(bf16_bits_to_float(x[i])));
  const Fp4RowScales scales = fp4_row_scales(amax);
  *outer = scales.outer;
  for (int g = 0; g < k / 16; ++g) {
    float v[16], in[16];
    for (int i = 0; i < 16; ++i) v[i] = bf16_bits_to_float(x[g * 16 + i]);
    fp4_block(v, scales, in);
    for (int i = 0; i < 16; ++i) inner[g * 16 + i] = float_to_bf16_bits(in[i]);
  }
}

void ink2_requantize_mxfp8_host(const uint16_t* x, int k, uint16_t* out) {
  if (k <= 0 || k % 32 != 0) throw std::invalid_argument("ink2_requantize_mxfp8_host: k");
  for (int g = 0; g < k / 32; ++g) {
    float v[32], o[32];
    for (int i = 0; i < 32; ++i) v[i] = bf16_bits_to_float(x[g * 32 + i]);
    mx8_block(v, o);
    for (int i = 0; i < 32; ++i) out[g * 32 + i] = float_to_bf16_bits(o[i]);
  }
}

void ink2_quantize_nvfp4_codes(const uint16_t* x, int64_t x_stride, int rows, int k, uint8_t* codes, uint8_t* scales,
                               int64_t scale_stride, float* gs, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!x || !codes || !scales || !gs) throw std::invalid_argument("ink2_quantize_nvfp4_codes: null buffer");
  if (k <= 0 || k % 16 != 0 || scale_stride < k / 16)
    throw std::invalid_argument("ink2_quantize_nvfp4_codes: k must be a multiple of 16 with a scale per block");
  quantize_nvfp4_codes_kernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(x, x_stride, k, codes, scales,
                                                                                    scale_stride, gs);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_swiglu_bf16(const uint16_t* gate_up, int64_t gate_up_stride, int rows, int inter, const float* slot_w,
                      const int32_t* row_slot, float gate_cap, float up_cap, uint16_t* act, int64_t act_stride,
                      cudaStream_t stream) {
  if (rows <= 0) return;
  if (!gate_up || !slot_w || !row_slot || !act) throw std::invalid_argument("ink2_swiglu_bf16: null buffer");
  if (inter <= 0 || (gate_up_stride * 2) % 4 != 0) throw std::invalid_argument("ink2_swiglu_bf16: geometry");
  swiglu_bf16_kernel<<<flat_blocks(static_cast<int64_t>(rows) * inter), kThreads, 0, stream>>>(
      gate_up, gate_up_stride, rows, inter, slot_w, row_slot, gate_cap, up_cap, act, act_stride);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_swiglu_quantize_nvfp4(const uint16_t* gate_up, int64_t gate_up_stride, int rows, int inter,
                                const float* slot_w, const int32_t* row_slot, float gate_cap, float up_cap,
                                uint8_t* codes, uint8_t* scales, int64_t scale_stride, float* gs,
                                cudaStream_t stream) {
  if (rows <= 0) return;
  if (!gate_up || !slot_w || !row_slot || !codes || !scales || !gs)
    throw std::invalid_argument("ink2_swiglu_quantize_nvfp4: null buffer");
  if (inter <= 0 || inter % 16 != 0 || inter > 8192 || scale_stride < inter / 16)
    throw std::invalid_argument("ink2_swiglu_quantize_nvfp4: the expert width must be a multiple of 16 up to 8192");
  swiglu_quantize_nvfp4_kernel<<<static_cast<unsigned>(rows), kThreads, static_cast<size_t>(inter) * 4, stream>>>(
      gate_up, gate_up_stride, inter, slot_w, row_slot, gate_cap, up_cap, codes, scales, scale_stride, gs);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_moe_accum_rounded(uint16_t* out, const float* down, int64_t down_stride, const int32_t* slot_row,
                            const int32_t* slot_ids, const float* slot_scale, int tokens, int top_k, int hidden,
                            cudaStream_t stream, const int64_t* positions) {
  if (tokens <= 0) return;
  if (!out || !down || !slot_row || !slot_ids || !slot_scale)
    throw std::invalid_argument("ink2_moe_accum_rounded: null buffer");
  if (top_k <= 0 || top_k > kAccumMaxTopK) throw std::invalid_argument("ink2_moe_accum_rounded: top_k out of range");
  const int64_t total = static_cast<int64_t>(tokens) * hidden;
  accum_rounded_kernel<<<static_cast<unsigned>((total + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
      out, down, down_stride, slot_row, slot_ids, slot_scale, tokens, top_k, hidden, positions);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
