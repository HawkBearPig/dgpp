#include "kernels/add_rmsnorm.hpp"
#include "kernels/pdl.cuh"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kVecElems = 8;                                   // bf16 per 16-byte vector
constexpr int kMaxVecs = kAddRmsnormMaxDim / (kThreads * kVecElems);  // 4 per thread

__device__ __forceinline__ double warp_sum(double v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// One row per block. Vector v of thread t covers elements [(v * kThreads +
// t) * 8, +8): a warp's 32 vectors are 512 contiguous bytes.
// kOne: the one-rounding form, y = bf16((x * rstd) * w) (Flint).
template <bool kAdd, bool kOne>
__global__ void __launch_bounds__(kThreads) add_rmsnorm_kernel(uint16_t* __restrict__ resid,
                                                               const uint16_t* __restrict__ add,
                                                               const uint16_t* __restrict__ w,
                                                               uint16_t* __restrict__ out, int dim,
                                                               float eps) {
  dgpp_pdl_wait();
  const int tid = threadIdx.x;
  const size_t row = static_cast<size_t>(blockIdx.x) * dim;
  uint16_t* xr = resid + row;
  const int nvec = dim / (kThreads * kVecElems);  // vectors per thread (dim % 2048 == 0)
  const int tail = (dim - nvec * kThreads * kVecElems) / kVecElems;  // vectors of the partial pass

  // Pass 1: the add (written back), the row's values kept as bf16 bits,
  // the fp64 sum of squares in four independent chains (GB10's dependent
  // fp64 add is ~90 ns); the weight vectors issued here too, so their
  // latency hides under the reduction.
  uint4 v[kMaxVecs], wv[kMaxVecs];
  double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
#pragma unroll
  for (int i = 0; i < kMaxVecs; ++i) {
    const bool live = i < nvec || (i == nvec && tid < tail);
    if (!live) continue;
    const int e0 = (i * kThreads + tid) * kVecElems;
    uint4 x = *reinterpret_cast<const uint4*>(xr + e0);
    wv[i] = *reinterpret_cast<const uint4*>(w + e0);
    if constexpr (kAdd) {
      const uint4 y = *reinterpret_cast<const uint4*>(add + row + e0);
      const uint32_t* xp = reinterpret_cast<const uint32_t*>(&x);
      const uint32_t* yp = reinterpret_cast<const uint32_t*>(&y);
      uint32_t zp[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint16_t lo = float_to_bf16_bits(bf16_bits_to_float(static_cast<uint16_t>(xp[j] & 0xffffu)) +
                                               bf16_bits_to_float(static_cast<uint16_t>(yp[j] & 0xffffu)));
        const uint16_t hi = float_to_bf16_bits(bf16_bits_to_float(static_cast<uint16_t>(xp[j] >> 16)) +
                                               bf16_bits_to_float(static_cast<uint16_t>(yp[j] >> 16)));
        zp[j] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
      }
      x = make_uint4(zp[0], zp[1], zp[2], zp[3]);
      *reinterpret_cast<uint4*>(xr + e0) = x;
    }
    v[i] = x;
    const uint32_t* xp = reinterpret_cast<const uint32_t*>(&x);
#pragma unroll
    for (int j = 0; j < 4; j += 2) {
      const float a = bf16_bits_to_float(static_cast<uint16_t>(xp[j] & 0xffffu));
      const float b = bf16_bits_to_float(static_cast<uint16_t>(xp[j] >> 16));
      const float c = bf16_bits_to_float(static_cast<uint16_t>(xp[j + 1] & 0xffffu));
      const float d = bf16_bits_to_float(static_cast<uint16_t>(xp[j + 1] >> 16));
      s0 = fma(static_cast<double>(a), static_cast<double>(a), s0);
      s1 = fma(static_cast<double>(b), static_cast<double>(b), s1);
      s2 = fma(static_cast<double>(c), static_cast<double>(c), s2);
      s3 = fma(static_cast<double>(d), static_cast<double>(d), s3);
    }
  }
  __shared__ double part[kWarps];
  __shared__ float s_rstd;
  const double ws = warp_sum((s0 + s1) + (s2 + s3));
  const int warp = tid / 32, lane = tid % 32;
  if (lane == 0) part[warp] = ws;
  __syncthreads();
  if (warp == 0) {
    double t = lane < kWarps ? part[lane] : 0.0;
    t = warp_sum(t);
    if (lane == 0) s_rstd = rsqrtf(static_cast<float>(t / dim) + eps);
  }
  __syncthreads();
  const float rstd = s_rstd;

  // Pass 2 from the registers: u = bf16(x * rstd), y = bf16(w * u) — or,
  // under kOne, y = bf16((x * rstd) * w).
#pragma unroll
  for (int i = 0; i < kMaxVecs; ++i) {
    const bool live = i < nvec || (i == nvec && tid < tail);
    if (!live) continue;
    const int e0 = (i * kThreads + tid) * kVecElems;
    const uint32_t* xp = reinterpret_cast<const uint32_t*>(&v[i]);
    const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv[i]);
    uint32_t yp[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float xlo = bf16_bits_to_float(static_cast<uint16_t>(xp[j] & 0xffffu));
      const float xhi = bf16_bits_to_float(static_cast<uint16_t>(xp[j] >> 16));
      const float wlo = bf16_bits_to_float(static_cast<uint16_t>(wp[j] & 0xffffu));
      const float whi = bf16_bits_to_float(static_cast<uint16_t>(wp[j] >> 16));
      uint16_t lo, hi;
      if constexpr (kOne) {
        lo = float_to_bf16_bits(__fmul_rn(__fmul_rn(xlo, rstd), wlo));
        hi = float_to_bf16_bits(__fmul_rn(__fmul_rn(xhi, rstd), whi));
      } else {
        const uint16_t ulo = float_to_bf16_bits(xlo * rstd);
        const uint16_t uhi = float_to_bf16_bits(xhi * rstd);
        lo = float_to_bf16_bits(wlo * bf16_bits_to_float(ulo));
        hi = float_to_bf16_bits(whi * bf16_bits_to_float(uhi));
      }
      yp[j] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
    }
    *reinterpret_cast<uint4*>(out + row + e0) = make_uint4(yp[0], yp[1], yp[2], yp[3]);
  }
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

template <bool kOne>
void launch(uint16_t* resid, const uint16_t* add, const uint16_t* weight, uint16_t* out, int rows, int dim,
            float eps, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!resid || !weight || !out) throw std::invalid_argument("add_rmsnorm: null buffer");
  if (dim <= 0 || dim % kVecElems != 0 || dim > kAddRmsnormMaxDim)
    throw std::invalid_argument("add_rmsnorm: dim must be a multiple of 8 up to 8192");
  if (!aligned16(resid) || !aligned16(weight) || !aligned16(out) || (add && !aligned16(add)))
    throw std::invalid_argument("add_rmsnorm: rows must be 16-byte aligned");
  if (add)
    add_rmsnorm_kernel<true, kOne><<<rows, kThreads, 0, stream>>>(resid, add, weight, out, dim, eps);
  else
    add_rmsnorm_kernel<false, kOne><<<rows, kThreads, 0, stream>>>(resid, nullptr, weight, out, dim, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace

void add_rmsnorm_bf16(uint16_t* resid, const uint16_t* add, const uint16_t* weight, uint16_t* out,
                      int rows, int dim, float eps, cudaStream_t stream) {
  launch<false>(resid, add, weight, out, rows, dim, eps, stream);
}

void add_rmsnorm_bf16_one(uint16_t* resid, const uint16_t* add, const uint16_t* weight, uint16_t* out,
                          int rows, int dim, float eps, cudaStream_t stream) {
  launch<true>(resid, add, weight, out, rows, dim, eps, stream);
}

}  // namespace dgpp
