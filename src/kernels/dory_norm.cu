#include "kernels/dory_norm.hpp"
#include "kernels/pdl.cuh"

#include <algorithm>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kVecElems = 8;                                       // bf16 per 16-byte vector
constexpr int kMaxVecs = kDoryNormMaxDim / (kThreads * kVecElems);  // 4 per thread

__device__ __forceinline__ double warp_sum(double v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ uint32_t pack_bf16(uint16_t lo, uint16_t hi) {
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

__device__ __forceinline__ float lo_bf16(uint32_t w) {
  return bf16_bits_to_float(static_cast<uint16_t>(w & 0xffffu));
}
__device__ __forceinline__ float hi_bf16(uint32_t w) {
  return bf16_bits_to_float(static_cast<uint16_t>(w >> 16));
}

// s0..s3 += the squares of the eight bf16 lanes of the two words at wp[j],
// wp[j + 1], in four independent fp64 chains (add_rmsnorm's shape: GB10's
// dependent fp64 add is ~90 ns, so the chains matter).
__device__ __forceinline__ void acc_words(const uint32_t* wp, int j, double& s0, double& s1, double& s2,
                                          double& s3) {
  const float a = lo_bf16(wp[j]), b = hi_bf16(wp[j]);
  const float c = lo_bf16(wp[j + 1]), d = hi_bf16(wp[j + 1]);
  s0 = fma(static_cast<double>(a), static_cast<double>(a), s0);
  s1 = fma(static_cast<double>(b), static_cast<double>(b), s1);
  s2 = fma(static_cast<double>(c), static_cast<double>(c), s2);
  s3 = fma(static_cast<double>(d), static_cast<double>(d), s3);
}

// The row's rstd from an fp64 partial sum. `part`/`slot` are the kernel's
// shared arrays; every thread of the block must call this in the same order.
__device__ __forceinline__ float block_rstd(double ss, double* part, float* slot, float eps) {
  const int tid = threadIdx.x;
  const int warp = tid / 32, lane = tid % 32;
  const double ws = warp_sum(ss);
  if (lane == 0) part[warp] = ws;
  __syncthreads();
  if (warp == 0) {
    double t = lane < kWarps ? part[lane] : 0.0;
    t = warp_sum(t);
    if (lane == 0) *slot = 1.0f / sqrtf(static_cast<float>(t) + eps);
  }
  __syncthreads();
  return *slot;
}

// The same for two sums at once: one shared round trip for both.
__device__ __forceinline__ void block_rstd_pair(double sa, double sb, double* part_a, double* part_b,
                                                float* out_a, float* out_b, float eps) {
  const int tid = threadIdx.x;
  const int warp = tid / 32, lane = tid % 32;
  const double wa = warp_sum(sa), wb = warp_sum(sb);
  if (lane == 0) {
    part_a[warp] = wa;
    part_b[warp] = wb;
  }
  __syncthreads();
  if (warp == 0) {
    double ta = lane < kWarps ? part_a[lane] : 0.0;
    double tb = lane < kWarps ? part_b[lane] : 0.0;
    ta = warp_sum(ta);
    tb = warp_sum(tb);
    if (lane == 0) {
      *out_a = 1.0f / sqrtf(static_cast<float>(ta) + eps);
      *out_b = 1.0f / sqrtf(static_cast<float>(tb) + eps);
    }
  }
  __syncthreads();
}

// One row per block; vector v of thread t covers elements
// [(v * kThreads + t) * 8, +8). Handles dim < kThreads * kVecElems through
// the partial tail pass (nvec == 0).
__global__ void __launch_bounds__(kThreads) justnorm_kernel(const uint16_t* __restrict__ x,
                                                            uint16_t* __restrict__ y, int dim,
                                                            float eps) {
  dgpp_pdl_wait();
  const int tid = threadIdx.x;
  const size_t row = static_cast<size_t>(blockIdx.x) * dim;
  const int nvec = dim / (kThreads * kVecElems);
  const int tail = (dim - nvec * kThreads * kVecElems) / kVecElems;

  uint4 v[kMaxVecs];
  double s[4] = {0.0, 0.0, 0.0, 0.0};
#pragma unroll
  for (int i = 0; i < kMaxVecs; ++i) {
    const bool live = i < nvec || (i == nvec && tid < tail);
    if (!live) continue;
    const int e0 = (i * kThreads + tid) * kVecElems;
    v[i] = *reinterpret_cast<const uint4*>(x + row + e0);
    const uint32_t* xp = reinterpret_cast<const uint32_t*>(&v[i]);
#pragma unroll
    for (int j = 0; j < 4; j += 2) acc_words(xp, j, s[0], s[1], s[2], s[3]);
  }

  __shared__ double part[kWarps];
  __shared__ float s_rstd;
  const float rstd = block_rstd((s[0] + s[1]) + (s[2] + s[3]), part, &s_rstd, eps);

#pragma unroll
  for (int i = 0; i < kMaxVecs; ++i) {
    const bool live = i < nvec || (i == nvec && tid < tail);
    if (!live) continue;
    const int e0 = (i * kThreads + tid) * kVecElems;
    const uint32_t* xp = reinterpret_cast<const uint32_t*>(&v[i]);
    uint32_t yp[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      yp[j] = pack_bf16(float_to_bf16_bits(lo_bf16(xp[j]) * rstd),
                        float_to_bf16_bits(hi_bf16(xp[j]) * rstd));
    *reinterpret_cast<uint4*>(y + row + e0) = make_uint4(yp[0], yp[1], yp[2], yp[3]);
  }
}

// The loop's re-injection: one row per block, the fp32 sums of a + b (the
// reference's `justnorm(state + input_h)` — the add in fp32, one rounding at
// the normalization's output, exactly as justnorm_kernel rounds).
__global__ void __launch_bounds__(kThreads) add_justnorm_kernel(const uint16_t* __restrict__ a,
                                                                const uint16_t* __restrict__ b,
                                                                uint16_t* __restrict__ y, int dim,
                                                                float eps) {
  dgpp_pdl_wait();
  const int tid = threadIdx.x;
  const size_t row = static_cast<size_t>(blockIdx.x) * dim;
  const int nvec = dim / (kThreads * kVecElems);
  const int tail = (dim - nvec * kThreads * kVecElems) / kVecElems;

  float s[kMaxVecs][kVecElems];
  double acc[4] = {0.0, 0.0, 0.0, 0.0};
#pragma unroll
  for (int i = 0; i < kMaxVecs; ++i) {
    const bool live = i < nvec || (i == nvec && tid < tail);
    if (!live) continue;
    const int e0 = (i * kThreads + tid) * kVecElems;
    const uint4 av = *reinterpret_cast<const uint4*>(a + row + e0);
    const uint4 bv = *reinterpret_cast<const uint4*>(b + row + e0);
    const uint32_t* ap = reinterpret_cast<const uint32_t*>(&av);
    const uint32_t* bp = reinterpret_cast<const uint32_t*>(&bv);
#pragma unroll
    for (int k = 0; k < kVecElems; ++k) {
      const float x = (k % 2 == 0) ? lo_bf16(ap[k / 2]) : hi_bf16(ap[k / 2]);
      const float z = (k % 2 == 0) ? lo_bf16(bp[k / 2]) : hi_bf16(bp[k / 2]);
      s[i][k] = x + z;
    }
#pragma unroll
    for (int k = 0; k < kVecElems; ++k)
      acc[k % 4] = fma(static_cast<double>(s[i][k]), static_cast<double>(s[i][k]), acc[k % 4]);
  }

  __shared__ double part[kWarps];
  __shared__ float s_rstd;
  const float rstd = block_rstd((acc[0] + acc[1]) + (acc[2] + acc[3]), part, &s_rstd, eps);

#pragma unroll
  for (int i = 0; i < kMaxVecs; ++i) {
    const bool live = i < nvec || (i == nvec && tid < tail);
    if (!live) continue;
    const int e0 = (i * kThreads + tid) * kVecElems;
    uint32_t yp[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      yp[j] = pack_bf16(float_to_bf16_bits(s[i][2 * j] * rstd),
                        float_to_bf16_bits(s[i][2 * j + 1] * rstd));
    *reinterpret_cast<uint4*>(y + row + e0) = make_uint4(yp[0], yp[1], yp[2], yp[3]);
  }
}

__global__ void __launch_bounds__(kThreads)
    slerp_alpha_kernel(const uint16_t* __restrict__ a, const uint16_t* __restrict__ b,
                       const uint16_t* __restrict__ alpha, float scale, uint16_t* __restrict__ y,
                       int dim, float eps) {
  dgpp_pdl_wait();
  const int tid = threadIdx.x;
  const size_t row = static_cast<size_t>(blockIdx.x) * dim;
  const int nvec = dim / (kThreads * kVecElems);
  const int tail = (dim - nvec * kThreads * kVecElems) / kVecElems;

  // Pass 1: both input rows held in registers, their two sums in one trip.
  uint4 va[kMaxVecs], vb[kMaxVecs];
  float mixed[kMaxVecs][kVecElems];
  double sa[4] = {0.0, 0.0, 0.0, 0.0}, sb[4] = {0.0, 0.0, 0.0, 0.0};
#pragma unroll
  for (int i = 0; i < kMaxVecs; ++i) {
    const bool live = i < nvec || (i == nvec && tid < tail);
    if (!live) continue;
    const int e0 = (i * kThreads + tid) * kVecElems;
    va[i] = *reinterpret_cast<const uint4*>(a + row + e0);
    vb[i] = *reinterpret_cast<const uint4*>(b + row + e0);
    const uint32_t* ap = reinterpret_cast<const uint32_t*>(&va[i]);
    const uint32_t* bp = reinterpret_cast<const uint32_t*>(&vb[i]);
#pragma unroll
    for (int j = 0; j < 4; j += 2) {
      acc_words(ap, j, sa[0], sa[1], sa[2], sa[3]);
      acc_words(bp, j, sb[0], sb[1], sb[2], sb[3]);
    }
  }

  __shared__ double part_a[kWarps], part_b[kWarps];
  __shared__ float s_rstd_a, s_rstd_b;
  block_rstd_pair((sa[0] + sa[1]) + (sa[2] + sa[3]), (sb[0] + sb[1]) + (sb[2] + sb[3]), part_a, part_b,
                  &s_rstd_a, &s_rstd_b, eps);
  const float rstd_a = s_rstd_a, rstd_b = s_rstd_b;

  // Pass 2: the two normalized rows, the per-column slerp, and the output
  // row's sum over the FP32 combine. The reference normalizes that float
  // tensor before converting its output back to bf16.
  double sy[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
#pragma unroll
  for (int i = 0; i < kMaxVecs; ++i) {
    const bool live = i < nvec || (i == nvec && tid < tail);
    if (!live) continue;
    const int e0 = (i * kThreads + tid) * kVecElems;
    const uint32_t* ap = reinterpret_cast<const uint32_t*>(&va[i]);
    const uint32_t* bp = reinterpret_cast<const uint32_t*>(&vb[i]);
    const uint32_t* pp = reinterpret_cast<const uint32_t*>(alpha + e0);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float lr_lo = fabsf(lo_bf16(pp[j]) * scale);
      const float lr_hi = fabsf(hi_bf16(pp[j]) * scale);
      const uint16_t na_lo = float_to_bf16_bits(lo_bf16(ap[j]) * rstd_a);
      const uint16_t na_hi = float_to_bf16_bits(hi_bf16(ap[j]) * rstd_a);
      const uint16_t nb_lo = float_to_bf16_bits(lo_bf16(bp[j]) * rstd_b);
      const uint16_t nb_hi = float_to_bf16_bits(hi_bf16(bp[j]) * rstd_b);
      const float na0 = bf16_bits_to_float(na_lo), nb0 = bf16_bits_to_float(nb_lo);
      const float na1 = bf16_bits_to_float(na_hi), nb1 = bf16_bits_to_float(nb_hi);
      const float f0 = na0 + lr_lo * (nb0 - na0);
      const float f1 = na1 + lr_hi * (nb1 - na1);
      mixed[i][2 * j] = f0;
      mixed[i][2 * j + 1] = f1;
      sy[2 * j] = fma(static_cast<double>(f0), static_cast<double>(f0), sy[2 * j]);
      sy[2 * j + 1] = fma(static_cast<double>(f1), static_cast<double>(f1), sy[2 * j + 1]);
    }
  }

  const double ss = ((sy[0] + sy[1]) + (sy[2] + sy[3])) + ((sy[4] + sy[5]) + (sy[6] + sy[7]));
  const float rstd_y = block_rstd(ss, part_a, &s_rstd_a, eps);

  // Pass 3.
#pragma unroll
  for (int i = 0; i < kMaxVecs; ++i) {
    const bool live = i < nvec || (i == nvec && tid < tail);
    if (!live) continue;
    const int e0 = (i * kThreads + tid) * kVecElems;
    uint32_t zp[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      zp[j] = pack_bf16(float_to_bf16_bits(mixed[i][2 * j] * rstd_y),
                        float_to_bf16_bits(mixed[i][2 * j + 1] * rstd_y));
    *reinterpret_cast<uint4*>(y + row + e0) = make_uint4(zp[0], zp[1], zp[2], zp[3]);
  }
}

// ---- the two fused sites -------------------------------------------------------
// Both keep the fp32 interior the reference's (plan §1.3: `fp32(attn) ⊙
// sigmoid(fp32(gate) ⊙ (s_gate·s_gate_factor)²)`; §1.4: `silu(fp32(g)) ⊙
// fp32(u)`) and round once, at justnorm's output — justnorm is `bf16(fp32(x) ·
// rsqrt(sum(fp32(x)²) + 1e-6))` (plan §1.2), so the row sums run over the fp32
// products, never a rounded intermediate. `attn`/`gate`/`g`/`u` arrive as the
// bf16 Linear outputs the engine rounds once (`fp32(...)` in the plan), exactly
// like launch_moe_swiglu_clamp's bf16 gate/up.
__global__ void __launch_bounds__(kThreads)
    gate_justnorm_kernel(const uint16_t* __restrict__ attn, const uint16_t* __restrict__ gate,
                         const uint16_t* __restrict__ s_gate, float sg_factor,
                         uint16_t* __restrict__ y, int dim, float eps) {
  dgpp_pdl_wait();
  const int tid = threadIdx.x;
  const size_t row = static_cast<size_t>(blockIdx.x) * dim;
  const int nvec = dim / (kThreads * kVecElems);
  const int tail = (dim - nvec * kThreads * kVecElems) / kVecElems;

  float g[kMaxVecs][kVecElems];
  double s[4] = {0.0, 0.0, 0.0, 0.0};
#pragma unroll
  for (int i = 0; i < kMaxVecs; ++i) {
    const bool live = i < nvec || (i == nvec && tid < tail);
    if (!live) continue;
    const int e0 = (i * kThreads + tid) * kVecElems;
    const uint4 av = *reinterpret_cast<const uint4*>(attn + row + e0);
    const uint4 gv4 = *reinterpret_cast<const uint4*>(gate + row + e0);
    const uint4 sv = *reinterpret_cast<const uint4*>(s_gate + e0);
    const uint32_t* ap = reinterpret_cast<const uint32_t*>(&av);
    const uint32_t* gp = reinterpret_cast<const uint32_t*>(&gv4);
    const uint32_t* sp = reinterpret_cast<const uint32_t*>(&sv);
#pragma unroll
    for (int k = 0; k < kVecElems; ++k) {
      const float a = (k % 2 == 0) ? lo_bf16(ap[k / 2]) : hi_bf16(ap[k / 2]);
      const float gv = (k % 2 == 0) ? lo_bf16(gp[k / 2]) : hi_bf16(gp[k / 2]);
      const float sc = (k % 2 == 0) ? lo_bf16(sp[k / 2]) : hi_bf16(sp[k / 2]);
      const float m = sc * sg_factor;
      const float z = gv * (m * m);
      g[i][k] = a * (1.0f / (1.0f + expf(-z)));
    }
#pragma unroll
    for (int k = 0; k < kVecElems; ++k)
      s[k % 4] = fma(static_cast<double>(g[i][k]), static_cast<double>(g[i][k]), s[k % 4]);
  }

  __shared__ double part[kWarps];
  __shared__ float s_rstd;
  const float rstd = block_rstd((s[0] + s[1]) + (s[2] + s[3]), part, &s_rstd, eps);

#pragma unroll
  for (int i = 0; i < kMaxVecs; ++i) {
    const bool live = i < nvec || (i == nvec && tid < tail);
    if (!live) continue;
    const int e0 = (i * kThreads + tid) * kVecElems;
    uint32_t yp[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      yp[j] = pack_bf16(float_to_bf16_bits(g[i][2 * j] * rstd),
                        float_to_bf16_bits(g[i][2 * j + 1] * rstd));
    *reinterpret_cast<uint4*>(y + row + e0) = make_uint4(yp[0], yp[1], yp[2], yp[3]);
  }
}

__global__ void __launch_bounds__(kThreads)
    silu_justnorm_kernel(const uint16_t* __restrict__ g_in, const uint16_t* __restrict__ u_in,
                         const uint16_t* __restrict__ suv_gate, const uint16_t* __restrict__ suv_up,
                         float suv_factor, uint16_t* __restrict__ y, int dim, float eps) {
  dgpp_pdl_wait();
  const int tid = threadIdx.x;
  const size_t row = static_cast<size_t>(blockIdx.x) * dim;
  const int nvec = dim / (kThreads * kVecElems);
  const int tail = (dim - nvec * kThreads * kVecElems) / kVecElems;

  float m[kMaxVecs][kVecElems];
  double s[4] = {0.0, 0.0, 0.0, 0.0};
#pragma unroll
  for (int i = 0; i < kMaxVecs; ++i) {
    const bool live = i < nvec || (i == nvec && tid < tail);
    if (!live) continue;
    const int e0 = (i * kThreads + tid) * kVecElems;
    const uint4 gg = *reinterpret_cast<const uint4*>(g_in + row + e0);
    const uint4 uu = *reinterpret_cast<const uint4*>(u_in + row + e0);
    const uint4 sg = *reinterpret_cast<const uint4*>(suv_gate + e0);
    const uint4 su = *reinterpret_cast<const uint4*>(suv_up + e0);
    const uint32_t* gp = reinterpret_cast<const uint32_t*>(&gg);
    const uint32_t* up = reinterpret_cast<const uint32_t*>(&uu);
    const uint32_t* sgp = reinterpret_cast<const uint32_t*>(&sg);
    const uint32_t* sup = reinterpret_cast<const uint32_t*>(&su);
#pragma unroll
    for (int k = 0; k < kVecElems; ++k) {
      const float gv = (k % 2 == 0) ? lo_bf16(gp[k / 2]) : hi_bf16(gp[k / 2]);
      const float uv = (k % 2 == 0) ? lo_bf16(up[k / 2]) : hi_bf16(up[k / 2]);
      const float sgc = (k % 2 == 0) ? lo_bf16(sgp[k / 2]) : hi_bf16(sgp[k / 2]);
      const float suc = (k % 2 == 0) ? lo_bf16(sup[k / 2]) : hi_bf16(sup[k / 2]);
      const float gs = gv * (sgc * suv_factor);
      const float us = uv * (suc * suv_factor);
      m[i][k] = (gs / (1.0f + expf(-gs))) * us;  // silu(g) · u, fp32 interior
    }
#pragma unroll
    for (int k = 0; k < kVecElems; ++k)
      s[k % 4] = fma(static_cast<double>(m[i][k]), static_cast<double>(m[i][k]), s[k % 4]);
  }

  __shared__ double part[kWarps];
  __shared__ float s_rstd;
  const float rstd = block_rstd((s[0] + s[1]) + (s[2] + s[3]), part, &s_rstd, eps);

#pragma unroll
  for (int i = 0; i < kMaxVecs; ++i) {
    const bool live = i < nvec || (i == nvec && tid < tail);
    if (!live) continue;
    const int e0 = (i * kThreads + tid) * kVecElems;
    uint32_t yp[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      yp[j] = pack_bf16(float_to_bf16_bits(m[i][2 * j] * rstd),
                        float_to_bf16_bits(m[i][2 * j + 1] * rstd));
    *reinterpret_cast<uint4*>(y + row + e0) = make_uint4(yp[0], yp[1], yp[2], yp[3]);
  }
}

void check_shape(int rows, int dim, const char* who) {
  if (dim <= 0 || dim % kVecElems != 0 || dim > kDoryNormMaxDim)
    throw std::invalid_argument(std::string(who) + ": dim must be a multiple of 8 up to 8192");
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

}  // namespace

// The walk's ping-pong staging copy (the loop's input_h/state and the head's
// fixed capture address). A kernel, not a cudaMemcpy: the decode graph must
// be kernels-only (engine/graph_check.hpp) and a capture records a copy as
// a memcpy node.
__global__ void __launch_bounds__(256) copy_rows_kernel(const uint4* __restrict__ src,
                                                        uint4* __restrict__ dst, size_t chunks) {
  dgpp_pdl_wait();
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < chunks;
       i += stride)
    dst[i] = src[i];
}

void dory_copy_rows_bf16(const uint16_t* src, uint16_t* dst, size_t elems, cudaStream_t stream) {
  if (elems == 0) return;
  if (!src || !dst) throw std::invalid_argument("dory_copy_rows: null buffer");
  if (!aligned16(src) || !aligned16(dst))
    throw std::invalid_argument("dory_copy_rows: buffers must be 16-byte aligned");
  const size_t chunks = elems / 8;  // bf16 rows: dims are multiples of 16 (config's check)
  const size_t blocks = std::min<size_t>((chunks + 255) / 256, 4096);
  copy_rows_kernel<<<blocks, 256, 0, stream>>>(reinterpret_cast<const uint4*>(src),
                                               reinterpret_cast<uint4*>(dst), chunks);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dory_justnorm_bf16(const uint16_t* x, uint16_t* y, int rows, int dim, float eps,
                        cudaStream_t stream) {
  if (rows <= 0) return;
  if (!x || !y) throw std::invalid_argument("dory_justnorm: null buffer");
  check_shape(rows, dim, "dory_justnorm");
  if (!aligned16(x) || !aligned16(y))
    throw std::invalid_argument("dory_justnorm: rows must be 16-byte aligned");
  justnorm_kernel<<<rows, kThreads, 0, stream>>>(x, y, dim, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dory_slerp_alpha_bf16(const uint16_t* a, const uint16_t* b, const uint16_t* alpha, float scale,
                           uint16_t* y, int rows, int dim, float eps, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!a || !b || !alpha || !y) throw std::invalid_argument("dory_slerp_alpha: null buffer");
  check_shape(rows, dim, "dory_slerp_alpha");
  if (!aligned16(a) || !aligned16(b) || !aligned16(alpha) || !aligned16(y))
    throw std::invalid_argument("dory_slerp_alpha: rows must be 16-byte aligned");
  slerp_alpha_kernel<<<rows, kThreads, 0, stream>>>(a, b, alpha, scale, y, dim, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dory_add_justnorm_bf16(const uint16_t* a, const uint16_t* b, uint16_t* y, int rows, int dim,
                            float eps, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!a || !b || !y) throw std::invalid_argument("dory_add_justnorm: null buffer");
  check_shape(rows, dim, "dory_add_justnorm");
  if (!aligned16(a) || !aligned16(b) || !aligned16(y))
    throw std::invalid_argument("dory_add_justnorm: rows must be 16-byte aligned");
  add_justnorm_kernel<<<rows, kThreads, 0, stream>>>(a, b, y, dim, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dory_gate_justnorm_bf16(const uint16_t* attn, const uint16_t* gate, const uint16_t* s_gate,
                             float s_gate_factor, uint16_t* y, int rows, int dim, float eps,
                             cudaStream_t stream) {
  if (rows <= 0) return;
  if (!attn || !gate || !s_gate || !y)
    throw std::invalid_argument("dory_gate_justnorm: null buffer");
  check_shape(rows, dim, "dory_gate_justnorm");
  if (!aligned16(attn) || !aligned16(gate) || !aligned16(s_gate) || !aligned16(y))
    throw std::invalid_argument("dory_gate_justnorm: rows must be 16-byte aligned");
  gate_justnorm_kernel<<<rows, kThreads, 0, stream>>>(attn, gate, s_gate, s_gate_factor, y, dim,
                                                      eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dory_silu_justnorm_bf16(const uint16_t* g_in, const uint16_t* u_in, const uint16_t* suv_gate,
                             const uint16_t* suv_up, float suv_factor, uint16_t* y, int rows,
                             int dim, float eps, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!g_in || !u_in || !suv_gate || !suv_up || !y)
    throw std::invalid_argument("dory_silu_justnorm: null buffer");
  check_shape(rows, dim, "dory_silu_justnorm");
  if (!aligned16(g_in) || !aligned16(u_in) || !aligned16(suv_gate) || !aligned16(suv_up) ||
      !aligned16(y))
    throw std::invalid_argument("dory_silu_justnorm: rows must be 16-byte aligned");
  silu_justnorm_kernel<<<rows, kThreads, 0, stream>>>(g_in, u_in, suv_gate, suv_up, suv_factor, y,
                                                      dim, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
