#include "kernels/glm_norm.hpp"
#include "kernels/pdl.cuh"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/hadamard32.cuh"

namespace dgpp {

namespace {

constexpr int kBlock = 512;

// Loads a thread issues together. What these kernels read is cold more
// often than not (the residual stream was last touched a sublayer ago, the
// norm weight a step ago), and issued one an iteration each load waited
// out a memory round trip before the next was asked for: a 6144-wide row,
// twelve elements a thread, took 12-14 us on the GB10. A thread now asks
// for a pass's elements together, and for its norm weights before the
// first pass, so they arrive under it. The arithmetic and its order are
// unchanged: the same bits.
constexpr int kNormBatch = 16;

// A thread's norm weights for the first kNormBatch elements of the second
// pass (i = tid, tid + nthreads, ...).
__device__ __forceinline__ void norm_weights_first(const uint16_t* __restrict__ w, int dim,
                                                   uint16_t (&w0)[kNormBatch]) {
  const int tid = threadIdx.x, nthreads = blockDim.x;
#pragma unroll
  for (int k = 0; k < kNormBatch; ++k) {
    const int i = tid + k * nthreads;
    w0[k] = i < dim ? w[i] : 0;
  }
}

// The first pass over a row: its fp32 values staged in `sx` ([dim], for
// the second pass) and the thread's sum of squares, accumulated in the
// i = tid, tid + nthreads, ... order.
__device__ __forceinline__ double norm_stage_row(const uint16_t* __restrict__ xr, int dim, float* sx) {
  const int tid = threadIdx.x, nthreads = blockDim.x;
  double ssq = 0.0;
  for (int i0 = tid; i0 < dim; i0 += nthreads * kNormBatch) {
    uint16_t xv[kNormBatch];
#pragma unroll
    for (int k = 0; k < kNormBatch; ++k) {
      const int i = i0 + k * nthreads;
      xv[k] = i < dim ? xr[i] : 0;
    }
#pragma unroll
    for (int k = 0; k < kNormBatch; ++k) {
      const int i = i0 + k * nthreads;
      if (i < dim) {
        const float v = bf16_bits_to_float(xv[k]);
        sx[i] = v;
        ssq += static_cast<double>(v) * v;
      }
    }
  }
  return ssq;
}

// The first pass with the residual add fused in (2026-10-13): x[i] =
// bf16(x[i] + y[i]) exactly as glm_residual_add_kernel writes it, the sum
// staged and squared from that rounded value. Bitwise the add kernel and
// the pass above, one launch and one read of the row fewer.
__device__ __forceinline__ double norm_stage_row_add(uint16_t* __restrict__ xr, const uint16_t* __restrict__ add,
                                                     int dim, float* sx) {
  const int tid = threadIdx.x, nthreads = blockDim.x;
  double ssq = 0.0;
  for (int i0 = tid; i0 < dim; i0 += nthreads * kNormBatch) {
    uint16_t xv[kNormBatch], av[kNormBatch];
#pragma unroll
    for (int k = 0; k < kNormBatch; ++k) {
      const int i = i0 + k * nthreads;
      xv[k] = i < dim ? xr[i] : 0;
      av[k] = i < dim ? add[i] : 0;
    }
#pragma unroll
    for (int k = 0; k < kNormBatch; ++k) {
      const int i = i0 + k * nthreads;
      if (i < dim) {
        const uint16_t xb = float_to_bf16_bits(bf16_bits_to_float(xv[k]) + bf16_bits_to_float(av[k]));
        xr[i] = xb;
        const float v = bf16_bits_to_float(xb);
        sx[i] = v;
        ssq += static_cast<double>(v) * v;
      }
    }
  }
  return ssq;
}

// The block's reduction of the threads' sums of squares to the row's
// reciprocal RMS. Block-wide: every thread of the block takes part.
__device__ __forceinline__ float norm_rstd(double ssq, int dim, float eps) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  __syncthreads();
  __shared__ double part[kBlock];
  part[tid] = ssq;
  __syncthreads();
  for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride && tid + stride < blockDim.x)
      part[tid] += part[tid + stride];
    __syncthreads();
  }
  return rsqrtf(static_cast<float>(part[0] / dim) + eps);
}

// Two-rounding RMSNorm of one row (the Glm5NextTextRMSNorm choreography):
// fp32 mean of squares, u = bf16(x * rstd), y = bf16(w * u). The
// intermediate bf16 round of u is semantics, not noise — the reference
// multiplies the bf16 tensor. This is the reduction and the second pass
// over a row staged by one of the first passes above; `w0` holds the
// thread's first weights (norm_weights_first). Shared by the plain norm
// and the residual-fused one, so both produce the same bits from the same
// staged row.
__device__ void rmsnorm_row_finish(const uint16_t* __restrict__ w, uint16_t* __restrict__ yr, int dim,
                                   float eps, const float* sx, double ssq, const uint16_t (&w0)[kNormBatch]) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const float rstd = norm_rstd(ssq, dim, eps);
  bool first = true;
  for (int i0 = tid; i0 < dim; i0 += nthreads * kNormBatch) {
    uint16_t wv[kNormBatch];
#pragma unroll
    for (int k = 0; k < kNormBatch; ++k) {
      const int i = i0 + k * nthreads;
      wv[k] = first ? w0[k] : (i < dim ? w[i] : 0);
    }
    first = false;
#pragma unroll
    for (int k = 0; k < kNormBatch; ++k) {
      const int i = i0 + k * nthreads;
      if (i < dim) {
        const uint16_t u = float_to_bf16_bits(sx[i] * rstd);
        yr[i] = float_to_bf16_bits(bf16_bits_to_float(wv[k]) * bf16_bits_to_float(u));
      }
    }
  }
}

__device__ void rmsnorm_row_two_rounding(const uint16_t* __restrict__ xr,
                                         const uint16_t* __restrict__ w,
                                         uint16_t* __restrict__ yr, int dim,
                                         float eps, float* sx) {
  uint16_t w0[kNormBatch];
  norm_weights_first(w, dim, w0);
  const double ssq = norm_stage_row(xr, dim, sx);
  rmsnorm_row_finish(w, yr, dim, eps, sx, ssq, w0);
}

__global__ __launch_bounds__(kBlock) void glm_rmsnorm_kernel(const uint16_t* __restrict__ x,
                                   const uint16_t* __restrict__ w,
                                   uint16_t* __restrict__ y, int dim,
                                   float eps) {
  dgpp_pdl_wait();
  const int row = blockIdx.x;
  extern __shared__ float smem[];  // [dim] staging for the second pass
  rmsnorm_row_two_rounding(x + static_cast<size_t>(row) * dim, w,
                           y + static_cast<size_t>(row) * dim, dim, eps, smem);
}

__global__ __launch_bounds__(kBlock) void glm_residual_add_rmsnorm_kernel(uint16_t* __restrict__ x, const uint16_t* __restrict__ add,
                                                const uint16_t* __restrict__ w, uint16_t* __restrict__ y,
                                                int dim, float eps) {
  dgpp_pdl_wait();
  const int row = blockIdx.x;
  extern __shared__ float smem[];
  uint16_t w0[kNormBatch];
  norm_weights_first(w, dim, w0);
  const double ssq =
      norm_stage_row_add(x + static_cast<size_t>(row) * dim, add + static_cast<size_t>(row) * dim, dim, smem);
  rmsnorm_row_finish(w, y + static_cast<size_t>(row) * dim, dim, eps, smem, ssq, w0);
}

// The norm with the Mixed346 activation quantizer fused into its second
// pass: the block's warps take the row's 128-value groups in turn (warp w
// groups w, w + warps, ...; lane j element j of each of the group's four
// H32 blocks), write y as the kernels above do, then rotate the four bf16
// values across the lanes, round, and emit the group's scale and codes
// exactly as hadamard32_quant_int8_kernel does from y. kNormBatch / 4
// groups' weights are asked for together: the thread's first before the
// first pass (norm_quant_weights_first), the rest a batch ahead of use.
__device__ __forceinline__ void norm_quant_weights_first(const uint16_t* __restrict__ w, int dim,
                                                         uint16_t (&w0)[kNormBatch]) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
  const int groups = dim / 128;
#pragma unroll
  for (int m = 0; m < kNormBatch / 4; ++m) {
    const int g = warp + m * warps;
#pragma unroll
    for (int b = 0; b < 4; ++b) w0[m * 4 + b] = g < groups ? w[g * 128 + b * kHadamard32Block + lane] : 0;
  }
}

__device__ void rmsnorm_row_finish_quant_int8(const uint16_t* __restrict__ w, uint16_t* __restrict__ yr,
                                              int8_t* __restrict__ cr, float* __restrict__ scales_row, int dim,
                                              float eps, const float* sx, double ssq,
                                              const uint16_t (&w0)[kNormBatch]) {
  const float rstd = norm_rstd(ssq, dim, eps);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
  const int groups = dim / 128;
  bool first = true;
  for (int g0 = warp; g0 < groups; g0 += warps * (kNormBatch / 4)) {
    uint16_t wv[kNormBatch];
#pragma unroll
    for (int m = 0; m < kNormBatch / 4; ++m) {
      const int g = g0 + m * warps;
#pragma unroll
      for (int b = 0; b < 4; ++b)
        wv[m * 4 + b] = first ? w0[m * 4 + b] : (g < groups ? w[g * 128 + b * kHadamard32Block + lane] : 0);
    }
    first = false;
#pragma unroll
    for (int m = 0; m < kNormBatch / 4; ++m) {
      const int g = g0 + m * warps;
      if (g < groups) {  // uniform across the warp
        const int c = g * 128;
        float v[4];
        float amax = 0.f;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const int i = c + b * kHadamard32Block + lane;
          const uint16_t u = float_to_bf16_bits(sx[i] * rstd);
          const uint16_t yv = float_to_bf16_bits(bf16_bits_to_float(wv[m * 4 + b]) * bf16_bits_to_float(u));
          yr[i] = yv;
          v[b] = bf16_bits_to_float(hadamard32_warp_bf16(yv));
          amax = fmaxf(amax, fabsf(v[b]));
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
        const float s = fmaxf(__fdiv_rn(amax, 127.f), 1e-30f);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          float q = rintf(__fdiv_rn(v[b], s));
          q = fminf(fmaxf(q, -128.f), 127.f);
          cr[c + b * kHadamard32Block + lane] = static_cast<int8_t>(static_cast<int>(q));
        }
        if (lane == 0) scales_row[g] = s;
      }
    }
  }
}

__global__ __launch_bounds__(kBlock) void glm_rmsnorm_quant_int8_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                                              uint16_t* __restrict__ y, int8_t* __restrict__ codes,
                                              size_t code_stride, float* __restrict__ scales,
                                              size_t scale_stride, int dim, float eps) {
  dgpp_pdl_wait();
  const int row = blockIdx.x;
  extern __shared__ float smem[];
  uint16_t w0[kNormBatch];
  norm_quant_weights_first(w, dim, w0);
  const double ssq = norm_stage_row(x + static_cast<size_t>(row) * dim, dim, smem);
  rmsnorm_row_finish_quant_int8(w, y + static_cast<size_t>(row) * dim, codes + static_cast<size_t>(row) * code_stride,
                                scales + static_cast<size_t>(row) * scale_stride, dim, eps, smem, ssq, w0);
}

// The quant norm with the residual add fused into its first pass (as
// glm_residual_add_rmsnorm_kernel): bitwise the add kernel followed by the
// kernel above.
__global__ __launch_bounds__(kBlock) void glm_residual_add_rmsnorm_quant_int8_kernel(uint16_t* __restrict__ x, const uint16_t* __restrict__ add,
                                                           const uint16_t* __restrict__ w, uint16_t* __restrict__ y,
                                                           int8_t* __restrict__ codes, size_t code_stride,
                                                           float* __restrict__ scales, size_t scale_stride, int dim,
                                                           float eps) {
  dgpp_pdl_wait();
  const int row = blockIdx.x;
  extern __shared__ float smem[];
  uint16_t w0[kNormBatch];
  norm_quant_weights_first(w, dim, w0);
  const double ssq =
      norm_stage_row_add(x + static_cast<size_t>(row) * dim, add + static_cast<size_t>(row) * dim, dim, smem);
  rmsnorm_row_finish_quant_int8(w, y + static_cast<size_t>(row) * dim, codes + static_cast<size_t>(row) * code_stride,
                                scales + static_cast<size_t>(row) * scale_stride, dim, eps, smem, ssq, w0);
}

// The MTP draft block's input row: [enorm(embed[token]) | hnorm(hidden)].
// blockIdx.y selects the half — the embedding half normalizes the token's
// embedding row, the hidden half the cached main-stack hidden at the row's
// position (positions[t], or first_pos + t when positions is null).
__global__ __launch_bounds__(kBlock) void glm_mtp_input_kernel(const uint16_t* __restrict__ embed,
                                     const int64_t* __restrict__ tokens,
                                     const uint16_t* __restrict__ hidden_cache,
                                     int64_t request_cache_stride,
                                     const int32_t* __restrict__ request_ids,
                                     const int64_t* __restrict__ positions,
                                     int64_t first_pos,
                                     const uint16_t* __restrict__ enorm,
                                     const uint16_t* __restrict__ hnorm,
                                     uint16_t* __restrict__ out, int hidden,
                                     float eps) {
  dgpp_pdl_wait();
  const int t = blockIdx.x;
  extern __shared__ float smem[];
  uint16_t* dst = out + static_cast<size_t>(t) * 2 * hidden;
  const int64_t staged = positions ? positions[t] : first_pos + t;
  if (staged < 0) {
    for (int h = threadIdx.x; h < hidden; h += blockDim.x)
      dst[blockIdx.y * hidden + h] = 0;
    return;
  }
  if (blockIdx.y == 0) {
    rmsnorm_row_two_rounding(embed + tokens[t] * hidden, enorm, dst, hidden,
                             eps, smem);
  } else {
    const int req = request_ids ? request_ids[t] : 0;
    const uint16_t* cache =
        hidden_cache + static_cast<int64_t>(req) * request_cache_stride;
    rmsnorm_row_two_rounding(cache + staged * hidden, hnorm, dst + hidden,
                             hidden, eps, smem);
  }
}

// x += y in bf16: the reference's `residual + hidden_states` on bf16
// tensors (fp32 add, one rounding).
__global__ void glm_residual_add_kernel(uint16_t* __restrict__ x,
                                        const uint16_t* __restrict__ y,
                                        int64_t n) {
  dgpp_pdl_wait();
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] = float_to_bf16_bits(bf16_bits_to_float(x[i]) +
                              bf16_bits_to_float(y[i]));
}

// cache[positions[t], :] = rows[t, :] — 16-byte vectors (hidden % 8 == 0,
// aligned bases: the launcher checks).
__global__ void glm_rows_scatter_kernel(const uint16_t* __restrict__ rows,
                                        const int32_t* __restrict__ request_ids,
                                        const int64_t* __restrict__ positions,
                                        uint16_t* __restrict__ cache,
                                        int64_t request_cache_stride,
                                        int hidden) {
  dgpp_pdl_wait();
  const int t = blockIdx.x;
  const int64_t pos = positions[t];
  if (pos < 0) return;
  const int req = request_ids ? request_ids[t] : 0;
  const uint4* src =
      reinterpret_cast<const uint4*>(rows + static_cast<size_t>(t) * hidden);
  uint4* dst = reinterpret_cast<uint4*>(
      cache + static_cast<int64_t>(req) * request_cache_stride + pos * hidden);
  for (int j = threadIdx.x; j < hidden / 8; j += blockDim.x) dst[j] = src[j];
}

// streams[t, s, :] = embed[tokens[t], :] for the four residual streams —
// the reference's unsqueeze+expand of the embedding.
__global__ void glm_embed_bcast_kernel(const uint16_t* __restrict__ table,
                                       const int64_t* __restrict__ tokens,
                                       uint16_t* __restrict__ streams,
                                       int hidden) {
  dgpp_pdl_wait();
  const int t = blockIdx.x;
  const uint16_t* src =
      table + static_cast<int64_t>(tokens[t]) * hidden;
  uint16_t* dst = streams + static_cast<size_t>(t) * 4 * hidden;
  for (int h = threadIdx.x + blockIdx.y * blockDim.x; h < hidden;
       h += blockDim.x * gridDim.y) {
    const uint16_t v = src[h];
    dst[h] = v;
    dst[hidden + h] = v;
    dst[2 * hidden + h] = v;
    dst[3 * hidden + h] = v;
  }
}

}  // namespace

void glm_rmsnorm_bf16(const void* x, const void* weight, void* y, int rows,
                      int dim, float eps, cudaStream_t stream) {
  if (rows <= 0) return;
  const size_t shmem = sizeof(float) * static_cast<size_t>(dim);
  // One-time opt-in for >48 KB dynamic smem (hidden beyond 12288). The
  // request is 64 KB, not the device's 99 KB opt-in ceiling: GB10 rejects
  // larger dynamic requests for this kernel's footprint with
  // cudaErrorInvalidValue, and the tolerance fallback below would silently
  // cap us at 48 KB instead. 64 KB covers hidden <= 16384.
  static const bool attr_ok = [] {
    constexpr size_t kWant = 64ull << 10;
    const cudaError_t e = cudaFuncSetAttribute(
        glm_rmsnorm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kWant));
    if (e != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    return true;
  }();
  if (!attr_ok && shmem > 49152)
    throw std::runtime_error(
        "glm_rmsnorm: dynamic smem exceeds driver-accepted limit");
  cudaGetLastError();
  glm_rmsnorm_kernel<<<rows, kBlock, shmem, stream>>>(
      static_cast<const uint16_t*>(x),
      static_cast<const uint16_t*>(weight),
      static_cast<uint16_t*>(y), dim, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_residual_add_rmsnorm_bf16(void* x, const void* add, const void* weight, void* y, int rows, int dim,
                                   float eps, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!x || !add || !weight || !y) throw std::invalid_argument("glm_residual_add_rmsnorm: null buffer");
  const size_t shmem = sizeof(float) * static_cast<size_t>(dim);
  static const bool attr_ok = [] {
    constexpr size_t kWant = 64ull << 10;
    const cudaError_t e = cudaFuncSetAttribute(glm_residual_add_rmsnorm_kernel,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kWant));
    if (e != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    return true;
  }();
  if (!attr_ok && shmem > 49152)
    throw std::runtime_error("glm_residual_add_rmsnorm: dynamic smem exceeds driver-accepted limit");
  cudaGetLastError();
  glm_residual_add_rmsnorm_kernel<<<rows, kBlock, shmem, stream>>>(
      static_cast<uint16_t*>(x), static_cast<const uint16_t*>(add), static_cast<const uint16_t*>(weight),
      static_cast<uint16_t*>(y), dim, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_rmsnorm_bf16_quant_int8(const void* x, const void* weight, void* y, int8_t* codes, size_t code_stride,
                                 float* scales, size_t scale_stride, int rows, int dim, float eps,
                                 cudaStream_t stream) {
  if (rows <= 0) return;
  if (!x || !weight || !y || !codes || !scales) throw std::invalid_argument("glm_rmsnorm_quant_int8: null buffer");
  if (dim <= 0 || dim % 128 != 0 || code_stride < static_cast<size_t>(dim) ||
      scale_stride < static_cast<size_t>(dim / 128))
    throw std::invalid_argument("glm_rmsnorm_quant_int8: dim must be a multiple of 128 within the code / scale rows");
  const size_t shmem = sizeof(float) * static_cast<size_t>(dim);
  static const bool attr_ok = [] {
    constexpr size_t kWant = 64ull << 10;
    const cudaError_t e = cudaFuncSetAttribute(glm_rmsnorm_quant_int8_kernel,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kWant));
    if (e != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    return true;
  }();
  if (!attr_ok && shmem > 49152) throw std::runtime_error("glm_rmsnorm_quant_int8: dynamic smem exceeds the limit");
  cudaGetLastError();
  glm_rmsnorm_quant_int8_kernel<<<rows, kBlock, shmem, stream>>>(
      static_cast<const uint16_t*>(x), static_cast<const uint16_t*>(weight), static_cast<uint16_t*>(y), codes,
      code_stride, scales, scale_stride, dim, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_residual_add_rmsnorm_bf16_quant_int8(void* x, const void* add, const void* weight, void* y, int8_t* codes,
                                              size_t code_stride, float* scales, size_t scale_stride, int rows,
                                              int dim, float eps, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!x || !add || !weight || !y || !codes || !scales)
    throw std::invalid_argument("glm_residual_add_rmsnorm_quant_int8: null buffer");
  if (dim <= 0 || dim % 128 != 0 || code_stride < static_cast<size_t>(dim) ||
      scale_stride < static_cast<size_t>(dim / 128))
    throw std::invalid_argument(
        "glm_residual_add_rmsnorm_quant_int8: dim must be a multiple of 128 within the code / scale rows");
  const size_t shmem = sizeof(float) * static_cast<size_t>(dim);
  static const bool attr_ok = [] {
    constexpr size_t kWant = 64ull << 10;
    const cudaError_t e = cudaFuncSetAttribute(glm_residual_add_rmsnorm_quant_int8_kernel,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kWant));
    if (e != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    return true;
  }();
  if (!attr_ok && shmem > 49152)
    throw std::runtime_error("glm_residual_add_rmsnorm_quant_int8: dynamic smem exceeds the limit");
  cudaGetLastError();
  glm_residual_add_rmsnorm_quant_int8_kernel<<<rows, kBlock, shmem, stream>>>(
      static_cast<uint16_t*>(x), static_cast<const uint16_t*>(add), static_cast<const uint16_t*>(weight),
      static_cast<uint16_t*>(y), codes, code_stride, scales, scale_stride, dim, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_mtp_input_bf16(const void* embed_table, const int64_t* tokens,
                        const void* hidden_cache, const int64_t* positions,
                        int64_t first_pos, const void* enorm, const void* hnorm,
                        void* out, int rows, int hidden, float eps,
                        cudaStream_t stream) {
  if (rows <= 0) return;
  if (!embed_table || !tokens || !hidden_cache || !enorm || !hnorm || !out)
    throw std::invalid_argument("glm_mtp_input: null buffer");
  const size_t shmem = sizeof(float) * static_cast<size_t>(hidden);
  if (shmem > 49152)
    throw std::invalid_argument("glm_mtp_input: hidden exceeds the smem stage");
  cudaGetLastError();
  glm_mtp_input_kernel<<<dim3(static_cast<unsigned>(rows), 2), kBlock, shmem,
                         stream>>>(
      static_cast<const uint16_t*>(embed_table), tokens,
      static_cast<const uint16_t*>(hidden_cache), /*request_cache_stride=*/0,
      /*request_ids=*/nullptr, positions, first_pos,
      static_cast<const uint16_t*>(enorm), static_cast<const uint16_t*>(hnorm),
      static_cast<uint16_t*>(out), hidden, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_mtp_input_bf16_batched(
    const void* embed_table, const int64_t* tokens, const void* hidden_cache,
    int64_t request_cache_stride, const int32_t* request_ids,
    const int64_t* positions, const void* enorm, const void* hnorm, void* out,
    int rows, int hidden, float eps, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!embed_table || !tokens || !hidden_cache || !request_ids || !positions ||
      !enorm || !hnorm || !out)
    throw std::invalid_argument("glm_mtp_input_batched: null buffer");
  if (request_cache_stride < hidden)
    throw std::invalid_argument("glm_mtp_input_batched: cache stride");
  const size_t shmem = sizeof(float) * static_cast<size_t>(hidden);
  if (shmem > 49152)
    throw std::invalid_argument(
        "glm_mtp_input_batched: hidden exceeds the smem stage");
  cudaGetLastError();
  glm_mtp_input_kernel<<<dim3(static_cast<unsigned>(rows), 2), kBlock, shmem,
                         stream>>>(
      static_cast<const uint16_t*>(embed_table), tokens,
      static_cast<const uint16_t*>(hidden_cache), request_cache_stride,
      request_ids, positions, /*first_pos=*/0,
      static_cast<const uint16_t*>(enorm), static_cast<const uint16_t*>(hnorm),
      static_cast<uint16_t*>(out), hidden, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_residual_add_bf16(void* x, const void* y, int64_t n,
                           cudaStream_t stream) {
  if (n <= 0) return;
  if (!x || !y) throw std::invalid_argument("glm_residual_add: null buffer");
  const unsigned grid = static_cast<unsigned>((n + kBlock - 1) / kBlock);
  glm_residual_add_kernel<<<grid, kBlock, 0, stream>>>(
      static_cast<uint16_t*>(x), static_cast<const uint16_t*>(y), n);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_rows_scatter_bf16(const void* rows, const int64_t* positions,
                           void* cache, int num_rows, int hidden,
                           cudaStream_t stream) {
  if (num_rows <= 0) return;
  const auto aligned16 = [](const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
  };
  if (!rows || !positions || !cache || hidden % 8 != 0 ||
      !aligned16(rows) || !aligned16(cache))
    throw std::invalid_argument(
        "glm_rows_scatter: hidden must be a multiple of 8 and the buffers "
        "16-byte aligned");
  glm_rows_scatter_kernel<<<static_cast<unsigned>(num_rows), 256, 0, stream>>>(
      static_cast<const uint16_t*>(rows), /*request_ids=*/nullptr, positions,
      static_cast<uint16_t*>(cache), /*request_cache_stride=*/0, hidden);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_rows_scatter_bf16_batched(
    const void* rows, const int32_t* request_ids, const int64_t* positions,
    void* cache, int64_t request_cache_stride, int num_rows, int hidden,
    cudaStream_t stream) {
  if (num_rows <= 0) return;
  const auto aligned16 = [](const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
  };
  if (!rows || !request_ids || !positions || !cache ||
      request_cache_stride < hidden || hidden % 8 != 0 ||
      !aligned16(rows) || !aligned16(cache) ||
      (request_cache_stride * 2) % 16 != 0)
    throw std::invalid_argument(
        "glm_rows_scatter_batched: invalid map, stride, or alignment");
  glm_rows_scatter_kernel<<<static_cast<unsigned>(num_rows), 256, 0, stream>>>(
      static_cast<const uint16_t*>(rows), request_ids, positions,
      static_cast<uint16_t*>(cache), request_cache_stride, hidden);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_embed_bcast_streams(const void* embed_table, const int64_t* tokens,
                             void* streams, int num_tokens, int hidden,
                             cudaStream_t stream) {
  if (num_tokens <= 0) return;
  const int blocks_y = (hidden + kBlock - 1) / kBlock;
  dim3 grid(static_cast<unsigned>(num_tokens),
            static_cast<unsigned>(blocks_y));
  glm_embed_bcast_kernel<<<grid, kBlock, 0, stream>>>(
      static_cast<const uint16_t*>(embed_table), tokens,
      static_cast<uint16_t*>(streams), hidden);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
