#include "kernels/head_screen.hpp"
#include "kernels/pdl.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/bf12_gemv.cuh"
#include "kernels/bf16_gemv.cuh"
#include "kernels/fp8_gemv.cuh"
#include "kernels/gemv_common.cuh"
#include "kernels/packq_gemv.cuh"
#include "kernels/packq_planes.cuh"

namespace dgpp {
namespace {

constexpr int kB = kHeadScreenBlock;
constexpr int kBuildThreads = 128;  // one thread per element of a 128-block
constexpr int kPrepThreads = 256;
constexpr int kSelectThreads = 256;
constexpr float kStepAlpha = 0.9957f;             // the four-level uniform quantizer's step on a unit Gaussian
constexpr float kInvSqrtB = 0.088388347648318f;   // 1 / sqrt(128)
constexpr int kMaxPer = kHeadScreenMaxK / kBuildThreads;  // elements per thread in the butterfly

inline uint64_t splitmix64(uint64_t& s) {
  s += 0x9E3779B97F4A7C15ull;
  uint64_t z = s;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

__device__ __forceinline__ int dp4a_s32_u32(uint32_t a_s8x4, uint32_t b_u8x4, int acc) {
  int d;
  asm("dp4a.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a_s8x4), "r"(b_u8x4), "r"(acc));
  return d;
}

// In-place block-128 Walsh-Hadamard over the smem row x[k] (k % 128 == 0,
// k == per * nthreads, nthreads % 128 == 0), scaled 1 / sqrt(128): the
// element e = tid + i * nthreads pairs with e ^ s, held by thread tid ^ s,
// so every element is written by its owner only. The same stage order and
// the same (e & s) rule on the host reproduce the fp32 values bit for bit.
template <int kPer>
__device__ __forceinline__ void wht128_smem(float* x, int k, int tid, int nthreads) {
  const int per = k / nthreads;
  float p[kPer];
  for (int s = 1; s < kB; s <<= 1) {
#pragma unroll
    for (int i = 0; i < kPer; ++i)
      if (i < per) p[i] = x[(tid + i * nthreads) ^ s];
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kPer; ++i)
      if (i < per) {
        const int e = tid + i * nthreads;
        const float v = x[e];
        x[e] = (e & s) ? p[i] - v : v + p[i];
      }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < kPer; ++i)
    if (i < per) x[tid + i * nthreads] *= kInvSqrtB;
  __syncthreads();
}

// Block reduction of one float (sum) or two over `nthreads` (a multiple of 32).
__device__ __forceinline__ float block_sum(float v, float* red, int tid, int nthreads) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
  const int warps = nthreads >> 5;
  if ((tid & 31) == 0) red[tid >> 5] = v;
  __syncthreads();
  float t = 0.f;
  for (int w = 0; w < warps; ++w) t += red[w];
  __syncthreads();
  return t;
}
__device__ __forceinline__ float block_max(float v, float* red, int tid, int nthreads) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, off));
  const int warps = nthreads >> 5;
  if ((tid & 31) == 0) red[tid >> 5] = v;
  __syncthreads();
  float t = -INFINITY;
  for (int w = 0; w < warps; ++w) t = fmaxf(t, red[w]);
  __syncthreads();
  return t;
}
__device__ __forceinline__ int block_sum_int(int v, int* red, int tid, int nthreads) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
  const int warps = nthreads >> 5;
  if ((tid & 31) == 0) red[tid >> 5] = v;
  __syncthreads();
  int t = 0;
  for (int w = 0; w < warps; ++w) t += red[w];
  __syncthreads();
  return t;
}

// ---------------------------------------------------------------------------
// The build: one CTA per vocab row. smem: [k] f32 (the rotated row), then
// [k] u8 (its codes), then the reductions' words.
// ---------------------------------------------------------------------------
// The head's served value at (row, e), one functor per encoding.
struct Fp8Src {
  const uint8_t* payload;
  const float* scales;
  int rs, cs, scale_cols, k;
  __device__ __forceinline__ float operator()(int row, int e) const {
    const uint8_t code = payload[static_cast<size_t>(row) * k + e];
    const float s = scales[static_cast<size_t>(row >> rs) * scale_cols + (e >> cs)];
    // The served value: the dequant bridge's bf16(w * s).
    return fp8_gemv::dequant(fp8_gemv::e4m3x2_to_float2(static_cast<uint16_t>(code)).x, s);
  }
};
struct Bf16Src {
  const uint16_t* w;
  int k;
  __device__ __forceinline__ float operator()(int row, int e) const {
    return bf16_bits_to_float(w[static_cast<size_t>(row) * k + e]);
  }
};
struct Bf12Src {  // the 12-bit companion (bf12_gemv.cuh consume_step's rebuild, element by element)
  const uint8_t* packed;
  const uint32_t* rows;
  const uint32_t* esc;
  const uint16_t* raw;
  int k;
  size_t row_bytes;
  __device__ __forceinline__ float operator()(int row, int e) const {
    const uint32_t eb = rows[2 * row], base23 = rows[2 * row + 1], ee = rows[2 * row + 2];
    if ((base23 & kBf12RawRow) != 0u)
      return bf16_bits_to_float(raw[static_cast<size_t>(base23 & ~kBf12RawRow) * k + e]);
    const int sb = e >> 10, w = e & 1023, st = w >> 8, lane = (w >> 3) & 31, j = w & 7;
    const uint8_t* blk = packed + static_cast<size_t>(row) * row_bytes + static_cast<size_t>(sb) * kBf12SuperBytes + lane * 16;
    const uint32_t b = blk[(st < 2 ? 0 : 512) + 8 * (st & 1) + j];
    const uint32_t en = blk[1024 + 4 * st + (j >> 1)];
    const uint32_t code = (j & 1) ? (en >> 4) : (en & 0xFu);
    uint32_t bits = 0u;
    if (code == 15u) {
      for (uint32_t i = eb; i < ee; ++i) {
        const uint32_t ent = esc[i];
        if ((ent >> 16) == static_cast<uint32_t>(e)) {
          bits = (ent & 0xFFFFu) << 16;
          break;
        }
      }
    } else {
      bits = ((b & 0x80u) << 24) | ((b & 0x7Fu) << 16) | (base23 + (code << 23));
    }
    return __uint_as_float(bits);
  }
};
struct PackqSrc {  // int8 g128 in the plane layout: (code - 128) x the f16 group scale
  const uint8_t* w;
  const uint16_t* scales;
  int k, scale_cols;
  __device__ __forceinline__ float operator()(int row, int e) const {
    const float s = packq_gemv::ScaleFmt<packq_gemv::kScaleF16G128>::to_float(
        scales[static_cast<size_t>(row) * scale_cols + (e >> 7)]);
    return (static_cast<float>(packq_planes::plane_code(w, row, k, e)) - 128.f) * s;
  }
};

template <class Src>
__global__ void __launch_bounds__(kBuildThreads) head_screen_build_kernel(
    Src src, const int8_t* __restrict__ sign, int n, int k,
    uint32_t* __restrict__ codes, float* __restrict__ step_out, float* __restrict__ rms_row) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) float xs[];
  uint8_t* cs8 = reinterpret_cast<uint8_t*>(xs + k);
  float* red = reinterpret_cast<float*>(cs8 + k);
  const int row = blockIdx.x, t = threadIdx.x;
  if (row >= n) return;
  for (int e = t; e < k; e += kBuildThreads) xs[e] = src(row, e) * static_cast<float>(sign[e]);
  __syncthreads();
  wht128_smem<kMaxPer>(xs, k, t, kBuildThreads);
  float ss = 0.f, sm = 0.f;
  for (int e = t; e < k; e += kBuildThreads) {
    const float v = xs[e];
    ss += v * v;
    sm += v;
  }
  ss = block_sum(ss, red, t, kBuildThreads);
  sm = block_sum(sm, red, t, kBuildThreads);
  const float mean = sm / static_cast<float>(k);
  const float var = fmaxf(ss / static_cast<float>(k) - mean * mean, 0.f);
  float step = kStepAlpha * sqrtf(var);
  if (!(step > 0.f)) step = 1.f;
  const float inv = 1.f / step;
  float rr = 0.f;
  for (int e = t; e < k; e += kBuildThreads) {
    const float v = xs[e];
    const float c = fminf(fmaxf(floorf(__fmaf_rn(v, inv, 2.f)), 0.f), 3.f);
    const float q = (c - 1.5f) * step;
    rr += (v - q) * (v - q);
    cs8[e] = static_cast<uint8_t>(c);
  }
  __syncthreads();
  rr = block_sum(rr, red, t, kBuildThreads);
  if (t == 0) {
    step_out[row] = step;
    rms_row[row] = sqrtf(rr / static_cast<float>(k));
  }
  const int kc = k / 16, b = row / kB, c = row % kB;
  for (int g = t; g < kc; g += kBuildThreads) {
    uint32_t w = 0u;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int L = 0; L < 4; ++L) w |= static_cast<uint32_t>(cs8[16 * g + 4 * j + L]) << (8 * L + 2 * j);
    codes[(static_cast<size_t>(b) * kc + g) * kB + c] = w;
  }
}

__global__ void __launch_bounds__(kB) head_screen_rms_blk_kernel(const float* __restrict__ rms_row, int n,
                                                                 float* __restrict__ rms_blk) {
  dgpp_pdl_wait();
  __shared__ float red[kB / 32];
  const int b = blockIdx.x, t = threadIdx.x, row = b * kB + t;
  const float v = row < n ? rms_row[row] : 0.f;
  const float m = block_max(v, red, t, kB);
  if (t == 0) rms_blk[b] = m;
}

// ---------------------------------------------------------------------------
// The prep: one CTA per activation row. smem: [k] f32, then the reductions.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kPrepThreads) head_screen_prep_kernel(const uint16_t* __restrict__ act,
                                                                        size_t act_stride,
                                                                        const int8_t* __restrict__ sign, int k,
                                                                        int8_t* __restrict__ xq,
                                                                        float* __restrict__ xmeta) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) float xs[];
  float* red = xs + k;
  int* redi = reinterpret_cast<int*>(red + 8);
  const int row = blockIdx.x, t = threadIdx.x;
  for (int e = t; e < k; e += kPrepThreads)
    xs[e] = bf16_bits_to_float(act[static_cast<size_t>(row) * act_stride + e]) * static_cast<float>(sign[e]);
  __syncthreads();
  wht128_smem<kHeadScreenMaxK / kPrepThreads>(xs, k, t, kPrepThreads);
  float amax = 0.f, ss = 0.f;
  for (int e = t; e < k; e += kPrepThreads) {
    const float v = xs[e];
    amax = fmaxf(amax, fabsf(v));
    ss += v * v;
  }
  amax = block_max(amax, red, t, kPrepThreads);
  ss = block_sum(ss, red, t, kPrepThreads);
  float scale = amax / 127.f;
  if (!(scale > 0.f)) scale = 1.f;
  const float inv = 1.f / scale;
  int sum = 0;
  for (int e = t; e < k; e += kPrepThreads) {
    const int q = max(-127, min(127, __float2int_rn(__fmul_rn(xs[e], inv))));
    xq[static_cast<size_t>(row) * k + e] = static_cast<int8_t>(q);
    sum += q;
  }
  sum = block_sum_int(sum, redi, t, kPrepThreads);
  if (t == 0) {
    xmeta[row * 4 + 0] = scale;
    xmeta[row * 4 + 1] = static_cast<float>(sum);
    xmeta[row * 4 + 2] = sqrtf(ss);
    xmeta[row * 4 + 3] = 0.f;
  }
}

// ---------------------------------------------------------------------------
// The screen: one CTA per vocab block, thread t its column t, kRows
// activation rows staged in smem as int8 [kRows][k]; four code words in
// flight per thread.
// ---------------------------------------------------------------------------
template <int kRows>
__global__ void __launch_bounds__(kB) head_screen_screen_kernel(const uint32_t* __restrict__ codes,
                                                                const float* __restrict__ step,
                                                                const int8_t* __restrict__ xq,
                                                                const float* __restrict__ xmeta, int n, int k,
                                                                int row0, int rows_in, int nb,
                                                                float* __restrict__ bmax) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) int8_t sx8[];
  __shared__ float red[kB / 32][kRows];
  const int t = threadIdx.x, kc = k / 16;
  {
    const uint32_t* src = reinterpret_cast<const uint32_t*>(xq + static_cast<size_t>(row0) * k);
    uint32_t* dst = reinterpret_cast<uint32_t*>(sx8);
    const int words = rows_in * k / 4;
    for (int i = t; i < words; i += kB) dst[i] = src[i];
  }
  __syncthreads();
  float xs_r[kRows];
  int xsum[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    xs_r[r] = r < rows_in ? xmeta[(row0 + r) * 4 + 0] : 0.f;
    xsum[r] = r < rows_in ? static_cast<int>(xmeta[(row0 + r) * 4 + 1]) : 0;
  }
  // A grid-stride walk over the blocks (the grid is a few hundred CTAs):
  // the tail of 1,940 small blocks over 48 SMs no longer idles a wave.
  for (int b = blockIdx.x; b < nb; b += gridDim.x) {
    int acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0;
    const uint32_t* cw = codes + static_cast<size_t>(b) * kc * kB + t;
    int g = 0;
    for (; g + 4 <= kc; g += 4) {
      uint32_t w[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) w[u] = __ldcs(cw + static_cast<size_t>(g + u) * kB);
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        uint32_t c2[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) c2[j] = ((w[u] >> (2 * j)) & 0x03030303u) << 1;
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
          const uint4 x4 = *reinterpret_cast<const uint4*>(sx8 + static_cast<size_t>(r) * k + 16 * (g + u));
          acc[r] = dp4a_s32_u32(x4.x, c2[0], acc[r]);
          acc[r] = dp4a_s32_u32(x4.y, c2[1], acc[r]);
          acc[r] = dp4a_s32_u32(x4.z, c2[2], acc[r]);
          acc[r] = dp4a_s32_u32(x4.w, c2[3], acc[r]);
        }
      }
    }
    for (; g < kc; ++g) {
      const uint32_t w = __ldcs(cw + static_cast<size_t>(g) * kB);
      uint32_t c2[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) c2[j] = ((w >> (2 * j)) & 0x03030303u) << 1;
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        const uint4 x4 = *reinterpret_cast<const uint4*>(sx8 + static_cast<size_t>(r) * k + 16 * g);
        acc[r] = dp4a_s32_u32(x4.x, c2[0], acc[r]);
        acc[r] = dp4a_s32_u32(x4.y, c2[1], acc[r]);
        acc[r] = dp4a_s32_u32(x4.z, c2[2], acc[r]);
        acc[r] = dp4a_s32_u32(x4.w, c2[3], acc[r]);
      }
    }
    const int col = b * kB + t;
    const float st = col < n ? step[col] : 0.f;
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      float v = -INFINITY;
      if (col < n && r < rows_in)
        v = __fmul_rn(__fmul_rn(__fmul_rn(xs_r[r], st), 0.5f), static_cast<float>(acc[r] - 3 * xsum[r]));
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, off));
      if ((t & 31) == 0) red[t >> 5][r] = v;
    }
    __syncthreads();
    if (t < kRows && t < rows_in) {
      float m = -INFINITY;
#pragma unroll
      for (int w = 0; w < kB / 32; ++w) m = fmaxf(m, red[w][t]);
      bmax[static_cast<size_t>(row0 + t) * nb + b] = m;
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// The select: one CTA per activation row over its nb block maxima.
// smem: bmax [nb] f32, sigma [nb] f32, taken [nb] i32, the reductions.
// ---------------------------------------------------------------------------
struct ArgMax {
  float v;
  int i;
};
__device__ __forceinline__ ArgMax better(ArgMax a, ArgMax b) {
  if (b.v > a.v || (b.v == a.v && b.i >= 0 && (a.i < 0 || b.i < a.i))) return b;
  return a;
}
__device__ __forceinline__ ArgMax block_argmax(ArgMax a, ArgMax* red, int tid, int nthreads) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    ArgMax o;
    o.v = __shfl_xor_sync(0xFFFFFFFFu, a.v, off);
    o.i = __shfl_xor_sync(0xFFFFFFFFu, a.i, off);
    a = better(a, o);
  }
  const int warps = nthreads >> 5;
  if ((tid & 31) == 0) red[tid >> 5] = a;
  __syncthreads();
  ArgMax r = red[0];
  for (int w = 1; w < warps; ++w) r = better(r, red[w]);
  __syncthreads();
  return r;
}

__global__ void __launch_bounds__(kSelectThreads) head_screen_select_kernel(const float* __restrict__ bmax,
                                                                            const float* __restrict__ rms_blk,
                                                                            const float* __restrict__ xmeta,
                                                                            int nb, int cap, float delta,
                                                                            int32_t* __restrict__ cand) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) float sm[];
  float* bm = sm;
  float* sig = bm + nb;
  int* taken = reinterpret_cast<int*>(sig + nb);
  ArgMax* red = reinterpret_cast<ArgMax*>(taken + nb);
  const int row = blockIdx.x, t = threadIdx.x;
  const float xnorm = xmeta[row * 4 + 2];
  for (int i = t; i < nb; i += kSelectThreads) {
    bm[i] = bmax[static_cast<size_t>(row) * nb + i];
    sig[i] = __fmul_rn(xnorm, rms_blk[i]);
    taken[i] = 0;
  }
  __syncthreads();
  ArgMax a{-INFINITY, -1};
  for (int i = t; i < nb; i += kSelectThreads) a = better(a, ArgMax{bm[i], i});
  const ArgMax best = block_argmax(a, red, t, kSelectThreads);
  const float base = best.i >= 0 ? __fsub_rn(best.v, __fmul_rn(delta, sig[best.i])) : INFINITY;
  int32_t* out = cand + static_cast<size_t>(row) * cap;
  for (int c = 0; c < cap; ++c) {
    ArgMax m{-INFINITY, -1};
    for (int i = t; i < nb; i += kSelectThreads)
      if (!taken[i] && bm[i] >= __fsub_rn(base, __fmul_rn(delta, sig[i]))) m = better(m, ArgMax{bm[i], i});
    const ArgMax pick = block_argmax(m, red, t, kSelectThreads);
    if (pick.i < 0) {
      for (int j = c + t; j < cap; j += kSelectThreads) out[j] = -1;
      return;
    }
    if (t == 0) {
      out[c] = pick.i;
      taken[pick.i] = 1;
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// The rescore: one CTA per (vocab block, activation row); a candidate block
// runs its 128 rows through the production row chain, a warp a row at a
// time against the row staged in smem; any other block writes -inf.
// ---------------------------------------------------------------------------
enum class HeadForm { Fp8, Bf16, Bf12 };

struct RescoreWeights {
  const uint8_t* payload = nullptr;
  const float* scales = nullptr;
  int rs = 7, cs = 7, scale_cols = 0;
  const uint16_t* wbf16 = nullptr;
  Bf12Matrix bf12;
  size_t bf12_row_bytes = 0;
};

template <HeadForm F>
__global__ void __launch_bounds__(kB) head_screen_rescore_kernel(const int32_t* __restrict__ cand, int cap,
                                                                 const uint16_t* __restrict__ act,
                                                                 size_t act_stride, int n, int k,
                                                                 RescoreWeights w, float* __restrict__ out,
                                                                 size_t out_stride) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint16_t sx16[];
  __shared__ int is_cand;
  const int b = blockIdx.x, row = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
  if (t == 0) {
    int f = 0;
    const int32_t* c = cand + static_cast<size_t>(row) * cap;
    for (int i = 0; i < cap; ++i) {
      if (c[i] < 0) break;
      if (c[i] == b) {
        f = 1;
        break;
      }
    }
    is_cand = f;
  }
  __syncthreads();
  float* o = out + static_cast<size_t>(row) * out_stride;
  if (!is_cand) {
    const int col = b * kB + t;
    if (col < n) o[col] = -INFINITY;
    return;
  }
  gemv::stage_activations<1>(act + static_cast<size_t>(row) * act_stride, act_stride, k, sx16);
  __syncthreads();
  for (int i = 0; i < 32; ++i) {
    const int col = b * kB + warp * 32 + i;
    if (col >= n) break;
    float acc[1];
    if constexpr (F == HeadForm::Fp8) {
      fp8_gemv::row_dots<1>(w.payload + static_cast<size_t>(col) * k,
                            w.scales + static_cast<size_t>(col >> w.rs) * w.scale_cols, sx16, k, lane, acc, w.cs);
    } else if constexpr (F == HeadForm::Bf16) {
      bf16_gemv::row_dots<1>(w.wbf16 + static_cast<size_t>(col) * k, sx16, k, lane, acc);
    } else {
      bf12_gemv::row_dots<1, 2>(w.bf12.packed, w.bf12.rows, w.bf12.esc, w.bf12.raw, col, k, w.bf12_row_bytes, sx16,
                                lane, acc);
    }
    if (lane == 0) o[col] = acc[0];
  }
}

// The int8 plane form's exact rescore: one warp per candidate row of the
// head, its lanes striding the row's 16-code chunks through the packed
// core's chain (packq_gemv::consume_chunk on the reassembled words).
template <int K>
__global__ void __launch_bounds__(kB) head_screen_rescore_packq_kernel(const int32_t* __restrict__ cand, int cap,
                                                                       const uint16_t* __restrict__ act,
                                                                       size_t act_stride, int n,
                                                                       const uint8_t* __restrict__ w,
                                                                       const uint16_t* __restrict__ scales,
                                                                       float* __restrict__ out, size_t out_stride) {
  dgpp_pdl_wait();
  constexpr int SF = packq_gemv::kScaleF16G128;
  using F = packq_gemv::Fmt<8, SF>;
  constexpr int kRowChunks = K / F::codes_per_chunk;
  constexpr int kScaleCols = K / F::group;
  extern __shared__ __align__(16) uint16_t sx16[];
  __shared__ int is_cand;
  const int b = blockIdx.x, row = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
  if (t == 0) {
    int f = 0;
    const int32_t* c = cand + static_cast<size_t>(row) * cap;
    for (int i = 0; i < cap; ++i) {
      if (c[i] < 0) break;
      if (c[i] == b) {
        f = 1;
        break;
      }
    }
    is_cand = f;
  }
  __syncthreads();
  float* o = out + static_cast<size_t>(row) * out_stride;
  if (!is_cand) {
    const int col = b * kB + t;
    if (col < n) o[col] = -INFINITY;
    return;
  }
  gemv::stage_activations<1>(act + static_cast<size_t>(row) * act_stride, act_stride, K, sx16);
  __syncthreads();
  for (int i = 0; i < 32; ++i) {
    const int col = b * kB + warp * 32 + i;
    if (col >= n) break;
    float acc[1] = {0.f};
    for (int ci = lane; ci < kRowChunks; ci += 32) {
      uint2 nib;
      uint32_t mid;
      packq_planes::load_hi6(w, col, K, ci, nib, mid);
      const uint32_t lo = packq_planes::load_lo2(w, col, K, ci);
      const float s = packq_gemv::ScaleFmt<SF>::to_float(scales[static_cast<size_t>(col) * kScaleCols + ci / F::chunks_per_group]);
      uint4 xv[1][F::window_vecs];
      packq_gemv::load_window<8, 1>(sx16, K, ci * F::codes_per_chunk, xv);
      packq_gemv::consume_chunk<8, 1, SF>(packq_planes::assemble(nib, mid, lo), s, xv, acc);
    }
    gemv::warp_reduce<1>(acc);
    if (lane == 0) o[col] = acc[0];
  }
}

size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

void check_shape(int n, int k, const char* who) {
  if (n <= 0 || k <= 0 || k % kB != 0 || k > kHeadScreenMaxK)
    throw std::invalid_argument(std::string(who) + ": k must be a positive multiple of 128 up to " +
                                std::to_string(kHeadScreenMaxK) + ", n positive");
}

template <class Src>
void build(const HeadScreen& s, Src src, uint64_t seed, cudaStream_t stream) {
  check_shape(s.n, s.k, "head_screen_build");
  std::vector<int8_t> signs(static_cast<size_t>(s.k));
  head_screen_signs(seed, s.k, signs.data());
  DGPP_CUDA_OK(cudaMemcpyAsync(s.sign, signs.data(), signs.size(), cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));  // the host vector goes out of scope
  const size_t smem = static_cast<size_t>(s.k) * 4 + static_cast<size_t>(s.k) + 64;
  head_screen_build_kernel<Src><<<s.n, kBuildThreads, smem, stream>>>(src, s.sign, s.n, s.k, s.codes, s.step,
                                                                      s.rms_row);
  DGPP_CUDA_OK(cudaGetLastError());
  head_screen_rms_blk_kernel<<<s.nb, kB, 0, stream>>>(s.rms_row, s.n, s.rms_blk);
  DGPP_CUDA_OK(cudaGetLastError());
}

// The screen's shared memory (the staged int8 rows, dynamic, plus the
// kernel's static warp-reduction array `red`, 16 bytes a row) stays under the
// 48 KB default: the opt-in attribute is a context mutation the graph-capture
// paths must never make (gemv_common.cuh), and a draft head's first call may
// well be a capture. Wide heads take fewer rows per launch instead. The
// static part counts: at 6144 wide, 8 rows stage exactly 48 KB and the
// launch is refused (the full GLM-5.3's width; a 2026-10-13 fabric boot).
constexpr size_t kScreenSmemMax = gemv::kMaxSmemBytes;
constexpr size_t screen_smem_total(int rows, int k) {
  return static_cast<size_t>(rows) * k + static_cast<size_t>(kB / 32) * rows * sizeof(float);
}

template <int kRows>
void screen_chunk(const HeadScreen& s, const HeadScreenScratch& sc, int row0, int rows_in, cudaStream_t stream) {
  const size_t smem = static_cast<size_t>(kRows) * s.k;
  if (screen_smem_total(kRows, s.k) > kScreenSmemMax)
    throw std::invalid_argument("head_screen: the row chunk's staged rows exceed shared memory");
  // 48 SMs x 8 resident CTAs: every CTA walks ~5 blocks of the 1,940 at the full head.
  const int grid = std::min(s.nb, 384);
  head_screen_screen_kernel<kRows><<<grid, kB, smem, stream>>>(s.codes, s.step, sc.xq, sc.xmeta, s.n, s.k, row0,
                                                               rows_in, s.nb, sc.bmax);
  DGPP_CUDA_OK(cudaGetLastError());
}

template <HeadForm F>
void rescore(const HeadScreen& s, const HeadScreenScratch& sc, const RescoreWeights& w, const uint16_t* act,
             size_t act_stride, int rows, float* out, size_t out_stride, cudaStream_t stream) {
  if (rows <= 0 || rows > sc.rows) throw std::invalid_argument("head_screen rescore: rows");
  if (out_stride == 0) out_stride = static_cast<size_t>(s.n);
  const dim3 grid(static_cast<unsigned>(s.nb), static_cast<unsigned>(rows));
  head_screen_rescore_kernel<F><<<grid, kB, gemv::smem_bytes(1, s.k), stream>>>(sc.cand, sc.cap, act, act_stride,
                                                                                s.n, s.k, w, out, out_stride);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace

void head_screen_signs(uint64_t seed, int k, int8_t* out) {
  uint64_t s = seed ^ 0x68656164736372ull;  // "headscr"
  for (int i = 0; i < k; ++i) out[i] = (splitmix64(s) & 1u) ? 1 : -1;
}

size_t head_screen_bytes(int n, int k) {
  check_shape(n, k, "head_screen_bytes");
  const int nb = (n + kB - 1) / kB;
  size_t bytes = align_up(static_cast<size_t>(nb) * (k / 16) * kB * 4, 256);  // codes
  bytes += align_up(static_cast<size_t>(n) * 4, 256);                          // step
  bytes += align_up(static_cast<size_t>(nb) * 4, 256);                         // rms_blk
  bytes += align_up(static_cast<size_t>(k), 256);                              // sign
  bytes += align_up(static_cast<size_t>(n) * 4, 256);                          // rms_row
  return bytes;
}

HeadScreen head_screen_carve(void* storage, int n, int k) {
  check_shape(n, k, "head_screen_carve");
  HeadScreen s;
  s.n = n;
  s.k = k;
  s.nb = (n + kB - 1) / kB;
  uint8_t* p = static_cast<uint8_t*>(storage);
  s.codes = reinterpret_cast<uint32_t*>(p);
  p += align_up(static_cast<size_t>(s.nb) * (k / 16) * kB * 4, 256);
  s.step = reinterpret_cast<float*>(p);
  p += align_up(static_cast<size_t>(n) * 4, 256);
  s.rms_blk = reinterpret_cast<float*>(p);
  p += align_up(static_cast<size_t>(s.nb) * 4, 256);
  s.sign = reinterpret_cast<int8_t*>(p);
  p += align_up(static_cast<size_t>(k), 256);
  s.rms_row = reinterpret_cast<float*>(p);
  return s;
}

void head_screen_build_fp8(const HeadScreen& s, const GlmQuantMatrix& w, uint64_t seed, cudaStream_t stream) {
  if (w.payload == nullptr || w.rows != s.n || w.cols != s.k)
    throw std::invalid_argument("head_screen_build_fp8: the fp8 head does not match the screen's shape");
  int rs = 7, cs = 7;
  if (!quant_scale_shifts(w, rs, cs)) throw std::invalid_argument("head_screen_build_fp8: scale grid");
  build(s, Fp8Src{w.payload, w.scales, rs, cs, static_cast<int>(w.scale_cols()), s.k}, seed, stream);
}

void head_screen_build_bf16(const HeadScreen& s, const uint16_t* w, uint64_t seed, cudaStream_t stream) {
  if (w == nullptr) throw std::invalid_argument("head_screen_build_bf16: null head");
  build(s, Bf16Src{w, s.k}, seed, stream);
}

bool head_screen_bf12_supported(const Bf12Matrix& w) {
  return w.packed != nullptr && w.rows != nullptr && w.n > 0 && w.k > 0 && w.k % kBf12Super == 0;
}

void head_screen_build_bf12(const HeadScreen& s, const Bf12Matrix& w, uint64_t seed, cudaStream_t stream) {
  if (!head_screen_bf12_supported(w) || w.n != s.n || w.k != s.k)
    throw std::invalid_argument("head_screen_build_bf12: the companion is not a tail-free packed head of the screen's shape");
  build(s, Bf12Src{w.packed, w.rows, w.esc, w.raw, s.k, bf12_row_bytes(s.k)}, seed, stream);
}

bool head_screen_packq_supported(const GlmPackedMatrix& w) {
  return w.packed != nullptr && w.bits == 8 && w.scale_fmt == kPackedScaleF16G128 && w.layout == kPackedLayoutPlanes8 &&
         w.cols % kB == 0 && packq_gemv::k_supported<8, packq_gemv::kScaleF16G128>(static_cast<int>(w.cols)) &&
         packq_gemv::sf_compiled(packq_gemv::kScaleF16G128, static_cast<int>(w.cols));
}

void head_screen_build_packq(const HeadScreen& s, const GlmPackedMatrix& w, uint64_t seed, cudaStream_t stream) {
  if (!head_screen_packq_supported(w) || w.rows != s.n || w.cols != s.k)
    throw std::invalid_argument("head_screen_build_packq: the packed head is not the int8 g128 plane form of the screen's shape");
  build(s, PackqSrc{reinterpret_cast<const uint8_t*>(w.packed), w.scales, s.k, static_cast<int>(w.scale_cols())}, seed,
        stream);
}

size_t head_screen_scratch_bytes(int rows, int k, int nb, int cap) {
  size_t bytes = align_up(static_cast<size_t>(rows) * k, 256);        // xq
  bytes += align_up(static_cast<size_t>(rows) * 16, 256);             // xmeta
  bytes += align_up(static_cast<size_t>(rows) * nb * 4, 256);         // bmax
  bytes += align_up(static_cast<size_t>(rows) * cap * 4, 256);        // cand
  return bytes;
}

HeadScreenScratch head_screen_scratch_carve(void* storage, int rows, int k, int nb, int cap) {
  if (rows <= 0 || cap <= 0 || cap > kHeadScreenMaxBlocks)
    throw std::invalid_argument("head_screen_scratch_carve: rows / cap");
  HeadScreenScratch sc;
  sc.rows = rows;
  sc.cap = cap;
  uint8_t* p = static_cast<uint8_t*>(storage);
  sc.xq = reinterpret_cast<int8_t*>(p);
  p += align_up(static_cast<size_t>(rows) * k, 256);
  sc.xmeta = reinterpret_cast<float*>(p);
  p += align_up(static_cast<size_t>(rows) * 16, 256);
  sc.bmax = reinterpret_cast<float*>(p);
  p += align_up(static_cast<size_t>(rows) * nb * 4, 256);
  sc.cand = reinterpret_cast<int32_t*>(p);
  return sc;
}

void launch_head_screen(const HeadScreen& s, const HeadScreenScratch& sc, const uint16_t* act, size_t act_stride,
                        int rows, float delta, cudaStream_t stream) {
  if (rows <= 0 || rows > sc.rows) throw std::invalid_argument("launch_head_screen: rows");
  if (!(delta >= 0.f)) throw std::invalid_argument("launch_head_screen: delta");
  head_screen_prep_kernel<<<rows, kPrepThreads, static_cast<size_t>(s.k) * 4 + 64, stream>>>(act, act_stride, s.sign,
                                                                                            s.k, sc.xq, sc.xmeta);
  DGPP_CUDA_OK(cudaGetLastError());
  // Rows per screen launch: the most of 16 / 8 / 4 whose int8 rows and the
  // reduction array fit the smem ceiling (16 up to 2944 wide, 8 up to 6016,
  // 4 to 8192).
  int chunk = kHeadScreenMaxRows;
  while (chunk > 1 && screen_smem_total(chunk, s.k) > kScreenSmemMax) chunk >>= 1;
  for (int row0 = 0; row0 < rows; row0 += chunk) {
    const int rows_in = std::min(chunk, rows - row0);
    if (rows_in <= 1) screen_chunk<1>(s, sc, row0, rows_in, stream);
    else if (rows_in <= 2) screen_chunk<2>(s, sc, row0, rows_in, stream);
    else if (rows_in <= 4) screen_chunk<4>(s, sc, row0, rows_in, stream);
    else if (rows_in <= 8) screen_chunk<8>(s, sc, row0, rows_in, stream);
    else screen_chunk<16>(s, sc, row0, rows_in, stream);
  }
  const size_t smem = static_cast<size_t>(s.nb) * 12 + sizeof(ArgMax) * (kSelectThreads / 32);
  head_screen_select_kernel<<<rows, kSelectThreads, smem, stream>>>(sc.bmax, s.rms_blk, sc.xmeta, s.nb, sc.cap, delta,
                                                                    sc.cand);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_head_rescore_fp8(const HeadScreen& s, const HeadScreenScratch& sc, const GlmQuantMatrix& w,
                             const uint16_t* act, size_t act_stride, int rows, float* out, size_t out_stride,
                             cudaStream_t stream) {
  if (w.payload == nullptr || w.rows != s.n || w.cols != s.k)
    throw std::invalid_argument("launch_head_rescore_fp8: the fp8 head does not match the screen's shape");
  RescoreWeights rw;
  rw.payload = w.payload;
  rw.scales = w.scales;
  if (!quant_scale_shifts(w, rw.rs, rw.cs)) throw std::invalid_argument("launch_head_rescore_fp8: scale grid");
  rw.scale_cols = static_cast<int>(w.scale_cols());
  rescore<HeadForm::Fp8>(s, sc, rw, act, act_stride, rows, out, out_stride, stream);
}

void launch_head_rescore_bf16(const HeadScreen& s, const HeadScreenScratch& sc, const uint16_t* w,
                              const uint16_t* act, size_t act_stride, int rows, float* out, size_t out_stride,
                              cudaStream_t stream) {
  if (w == nullptr) throw std::invalid_argument("launch_head_rescore_bf16: null head");
  RescoreWeights rw;
  rw.wbf16 = w;
  rescore<HeadForm::Bf16>(s, sc, rw, act, act_stride, rows, out, out_stride, stream);
}

void launch_head_rescore_bf12(const HeadScreen& s, const HeadScreenScratch& sc, const Bf12Matrix& w,
                              const uint16_t* act, size_t act_stride, int rows, float* out, size_t out_stride,
                              cudaStream_t stream) {
  if (w.packed == nullptr || w.n != s.n || w.k != s.k)
    throw std::invalid_argument("launch_head_rescore_bf12: the packed head does not match the screen's shape");
  RescoreWeights rw;
  rw.bf12 = w;
  rw.bf12_row_bytes = bf12_row_bytes(s.k);
  rescore<HeadForm::Bf12>(s, sc, rw, act, act_stride, rows, out, out_stride, stream);
}

void launch_head_rescore_packq(const HeadScreen& s, const HeadScreenScratch& sc, const GlmPackedMatrix& w,
                               const uint16_t* act, size_t act_stride, int rows, float* out, size_t out_stride,
                               cudaStream_t stream) {
  if (!head_screen_packq_supported(w) || w.rows != s.n || w.cols != s.k)
    throw std::invalid_argument("launch_head_rescore_packq: the packed head is not the int8 g128 plane form of the screen's shape");
  if (rows <= 0 || rows > sc.rows) throw std::invalid_argument("launch_head_rescore_packq: rows");
  if (out_stride == 0) out_stride = static_cast<size_t>(s.n);
  const dim3 grid(static_cast<unsigned>(s.nb), static_cast<unsigned>(rows));
  packq_gemv::dispatch_k(s.k, [&](auto kc) {
    constexpr int K = decltype(kc)::value;
    if constexpr (packq_gemv::k_supported<8, packq_gemv::kScaleF16G128>(K) &&
                  packq_gemv::sf_compiled(packq_gemv::kScaleF16G128, K) && K % kB == 0) {
      head_screen_rescore_packq_kernel<K><<<grid, kB, gemv::smem_bytes(1, K), stream>>>(
          sc.cand, sc.cap, act, act_stride, s.n, reinterpret_cast<const uint8_t*>(w.packed), w.scales, out, out_stride);
      DGPP_CUDA_OK(cudaGetLastError());
    } else {
      throw std::invalid_argument("launch_head_rescore_packq: K is outside the plane layout's compiled set");
    }
  });
}

}  // namespace dgpp
