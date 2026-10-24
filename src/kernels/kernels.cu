#include "kernels/kernels.hpp"
#include "kernels/pdl.cuh"

#include <math.h>

#include <algorithm>

#include "common/cuda_check.hpp"
#include "common/log.hpp"

namespace dgpp {

namespace {

__device__ __forceinline__ float sigmoidf_fast(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__global__ void rmsnorm_kernel(const uint16_t* __restrict__ x,
                               const uint16_t* __restrict__ w,
                               uint16_t* __restrict__ y, int dim, float eps) {
  dgpp_pdl_wait();
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  extern __shared__ float smem[];   // [dim] staging for re-read
  float* sx = smem;

  const uint16_t* xr = x + static_cast<size_t>(row) * dim;
  uint16_t* yr = y + static_cast<size_t>(row) * dim;

  // fp32 accumulation over squared values; second pass applies scale.
  double ssq = 0.0;
  for (int i = tid; i < dim; i += nthreads) {
    float v = bf16_bits_to_float(xr[i]);
    sx[i] = v;
    ssq += static_cast<double>(v) * v;
  }
  __syncthreads();

  __shared__ double part[1024];
  part[tid] = ssq;
  __syncthreads();
  for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride && tid + stride < blockDim.x) {
      part[tid] += part[tid + stride];
    }
    __syncthreads();
  }
  float inv_rms =
      rsqrtf(static_cast<float>(part[0] / dim) + eps);

  for (int i = tid; i < dim; i += nthreads) {
    yr[i] = float_to_bf16_bits(bf16_bits_to_float(w[i]) * sx[i] * inv_rms);
  }
}

__global__ void swiglu_limit_kernel(const uint16_t* __restrict__ gate,
                                    const uint16_t* __restrict__ up,
                                    uint16_t* __restrict__ out, int64_t n,
                                    float limit) {
  dgpp_pdl_wait();
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  float g = fminf(fmaxf(bf16_bits_to_float(gate[i]), -limit), limit);
  float u = fminf(fmaxf(bf16_bits_to_float(up[i]), -limit), limit);
  out[i] = float_to_bf16_bits(sigmoidf_fast(g) * g * u);
}

template <int S>
__global__ void mhc_mix_kernel(const uint16_t* __restrict__ x,
                               const float* __restrict__ mix,
                               uint16_t* __restrict__ out, int hidden) {
  dgpp_pdl_wait();
  // x: [B,S,H] contiguous; mix: [B,S,S]; out: [B,S,H]
  const int b = blockIdx.x;
  const int h = threadIdx.x + blockIdx.y * blockDim.x;
  if (h >= hidden) return;
  float acc[S];
#pragma unroll
  for (int s = 0; s < S; ++s) acc[s] = 0.f;
  const size_t bh = static_cast<size_t>(b) * S * hidden + h;
#pragma unroll
  for (int j = 0; j < S; ++j) {
    float xv = bf16_bits_to_float(x[bh + j * hidden]);
#pragma unroll
    for (int s = 0; s < S; ++s) {
      float m = mix[(static_cast<size_t>(b) * S + s) * S + j];
      acc[s] = fmaf(m, xv, acc[s]);
    }
  }
  size_t obh = static_cast<size_t>(b) * S * hidden + h;
#pragma unroll
  for (int s = 0; s < S; ++s)
    out[obh + s * hidden] = float_to_bf16_bits(acc[s]);
}

__global__ void argmax_rows_kernel(const float* __restrict__ logits,
                                   int64_t* __restrict__ out_idx,
                                   float* __restrict__ out_val, int cols) {
  dgpp_pdl_wait();
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const float* lr = logits + static_cast<size_t>(row) * cols;

  float best = -INFINITY;
  int64_t best_i = -1;
  for (int i = tid; i < cols; i += nthreads) {
    float v = lr[i];
    // Strict '>' plus explicit index tie-break gives first-occurrence-wins
    // semantics matching torch.argmax, independent of thread scheduling.
    if (best_i < 0 || v > best || (v == best && i < best_i)) {
      best = v;
      best_i = i;
    }
  }
  __shared__ float s_best[1024];
  __shared__ int64_t s_idx[1024];
  s_best[tid] = best;
  s_idx[tid] = best_i;
  __syncthreads();
  auto prefer = [](float vb, int64_t ib, float va, int64_t ia) {
    return vb > va || (vb == va && ib < ia);
  };
  for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      if (prefer(s_best[tid + stride], s_idx[tid + stride], s_best[tid],
                 s_idx[tid])) {
        s_best[tid] = s_best[tid + stride];
        s_idx[tid] = s_idx[tid + stride];
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    out_idx[row] = s_idx[0];
    if (out_val) out_val[row] = s_best[0];
  }
}

__global__ void embed_gather_kernel(const uint16_t* __restrict__ table,
                                    const int64_t* __restrict__ tokens,
                                    uint16_t* __restrict__ out, int hidden) {
  dgpp_pdl_wait();
  const int t = blockIdx.x;
  const long long tok = tokens[t];
  const uint16_t* src = table + static_cast<size_t>(tok) * hidden;
  uint16_t* dst = out + static_cast<size_t>(t) * hidden;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) dst[i] = src[i];
}

__global__ void embed_gather_sliced_kernel(const uint16_t* __restrict__ table,
                                           const int64_t* __restrict__ tokens,
                                           uint16_t* __restrict__ out, int hidden,
                                           int64_t vocab_begin, int64_t vocab_count) {
  dgpp_pdl_wait();
  const int t = blockIdx.x;
  const long long tok = tokens[t] - vocab_begin;
  uint16_t* dst = out + static_cast<size_t>(t) * hidden;
  if (tok < 0 || tok >= vocab_count) {
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) dst[i] = 0;
    return;
  }
  const uint16_t* src = table + static_cast<size_t>(tok) * hidden;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) dst[i] = src[i];
}

__global__ void copy_rows_kernel(uint16_t* __restrict__ dst, const uint16_t* __restrict__ src, int hidden) {
  dgpp_pdl_wait();
  const size_t base = static_cast<size_t>(blockIdx.x) * hidden;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) dst[base + i] = src[base + i];
}

// Deterministic hash-based normal sampler: one value per index.
__device__ __forceinline__ uint64_t hash_u64(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}
__device__ __forceinline__ uint32_t hash_u32_pair(uint64_t seed, uint64_t idx) {
  uint64_t h = hash_u64(seed * 0x9E3779B97F4A7C15ULL ^ idx);
  return static_cast<uint32_t>(h ^ (h >> 32));
}

__global__ void fill_normal_bf16_kernel(uint16_t* dst, uint64_t n,
                                        uint64_t seed, float stddev) {
  dgpp_pdl_wait();
  uint64_t i = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  uint32_t a = hash_u32_pair(seed, i);
  uint32_t b = hash_u32_pair(seed ^ 0x5DEECE66DULL, i);
  float u1 = (static_cast<float>(a & 0xFFFFFFu) + 1.0f) / 16777218.0f;
  float u2 = static_cast<float>(b & 0xFFFFFu) / 1048576.0f;
  // Box–Muller; only one of the pair needed.
  float z = sqrtf(-2.0f * ::logf(u1)) * cosf(6.2831853f * u2);
  dst[i] = float_to_bf16_bits(z * stddev);
}

__global__ void fill_normal_fp8_kernel(uint8_t* dst, uint64_t n,
                                       uint64_t seed, float stddev) {
  dgpp_pdl_wait();
  uint64_t i = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  uint32_t a = hash_u32_pair(seed, i);
  uint32_t b = hash_u32_pair(seed ^ 0x2545F4914F6CDD1DULL, i);
  float u1 = (static_cast<float>(a & 0xFFFFFFu) + 1.0f) / 16777218.0f;
  float u2 = static_cast<float>(b & 0xFFFFFu) / 1048576.0f;
  float z = sqrtf(-2.0f * ::logf(u1)) * cosf(6.2831853f * u2);
  dst[i] = float_to_fp8_e4m3_bits(z * stddev);
}

__global__ void fill_normal_f32_kernel(float* dst, uint64_t n, uint64_t seed,
                                       float stddev) {
  dgpp_pdl_wait();
  uint64_t i = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  uint32_t a = hash_u32_pair(seed, i);
  uint32_t b = hash_u32_pair(seed ^ 0x9E3779B9ULL, i);
  float u1 = (static_cast<float>(a & 0xFFFFFFu) + 1.0f) / 16777218.0f;
  float u2 = static_cast<float>(b & 0xFFFFFu) / 1048576.0f;
  float z = sqrtf(-2.0f * ::logf(u1)) * cosf(6.2831853f * u2);
  dst[i] = z * stddev;
}

__global__ void add_inplace_kernel(uint16_t* __restrict__ x,
                                   const uint16_t* __restrict__ y,
                                   int64_t n) {
  dgpp_pdl_wait();
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  float s = bf16_bits_to_float(x[i]) + bf16_bits_to_float(y[i]);
  x[i] = float_to_bf16_bits(s);
}

__global__ void cast_rows_kernel(const uint16_t* __restrict__ src,
                                 float* __restrict__ dst, int cols) {
  dgpp_pdl_wait();
  const int row = blockIdx.x;
  const uint16_t* sr = src + static_cast<size_t>(row) * cols;
  float* dr = dst + static_cast<size_t>(row) * cols;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    dr[i] = bf16_bits_to_float(sr[i]);
  }
}

__global__ void cast_fp8_kernel(const uint16_t* __restrict__ src,
                                uint8_t* __restrict__ dst, int64_t n) {
  dgpp_pdl_wait();
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  dst[i] = float_to_fp8_e4m3_bits(bf16_bits_to_float(src[i]));
}

__global__ void kv_append_kernel(const uint16_t* __restrict__ src,
                                 int64_t src_stride, uint16_t* __restrict__ kc,
                                 uint16_t* __restrict__ vc,
                                 const int* __restrict__ dev_abs_first,
                                 int rows, int kv_dim) {
  dgpp_pdl_wait();
  // src row layout: [K block | V block] at (base + r*src_stride) where base
  // already points at the k-column offset of the fused qkv buffer.
  const int slot = static_cast<int>(blockIdx.x);
  if (slot >= rows) return;
  const int abs_slot = *dev_abs_first + slot;
  const uint16_t* krow = src + static_cast<size_t>(slot) * src_stride;
  const uint16_t* vrow = krow + kv_dim;
  uint16_t* kdst = kc + static_cast<size_t>(abs_slot) * kv_dim;
  uint16_t* vdst = vc + static_cast<size_t>(abs_slot) * kv_dim;
  for (int d = threadIdx.x; d < kv_dim; d += blockDim.x) {
    kdst[d] = krow[d];
    vdst[d] = vrow[d];
  }
}

__global__ void bump_i32_kernel(int32_t* p, int delta) {
  dgpp_pdl_wait();
  if (threadIdx.x == 0) atomicAdd(p, delta);  // single caller => deterministic
}

__global__ void cpy_i64_kernel(const int64_t* __restrict__ s,
                               int64_t* __restrict__ d) {
  dgpp_pdl_wait();
  if (threadIdx.x == 0) d[0] = s[0];
}

__global__ void swiglu_pairs_kernel(const uint16_t* __restrict__ gu,
                                    uint16_t* __restrict__ out, int64_t total,
                                    int inter, float limit) {
  dgpp_pdl_wait();
  // Interleaved halves per row: g at (2r)*inter+c, u at (2r+1)*inter+c.
  int64_t j = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (j >= total) return;
  const int64_t r = j / inter;
  const int64_t c = j - r * inter;
  float g = fminf(fmaxf(bf16_bits_to_float(gu[(2 * r) * inter + c]), -limit),
                  limit);
  float u = fminf(
      fmaxf(bf16_bits_to_float(gu[(2 * r + 1) * inter + c]), -limit), limit);
  out[j] = float_to_bf16_bits(sigmoidf_fast(g) * g * u);
}

}  // namespace

void rmsnorm_bf16(const void* x, const void* weight, void* y, int rows,
                  int dim, float eps, cudaStream_t stream) {
  constexpr int kBlock = 512;
  size_t shmem = sizeof(float) * dim;
  // One-time opt-in for >48 KB dynamic smem. Request within the
  // driver-computed PER-KERNEL ceiling (the device-wide opt-in cap minus
  // this kernel's own footprint) so cudaFuncSetAttribute cannot fail: a
  // blind request at the device cap is rejected on GB10 for this kernel's
  // footprint, and even a swallowed-and-cleared probe reports under
  // compute-sanitizer.
  static const bool attr_ok = [] {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, rmsnorm_kernel) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    const size_t ceiling =
        static_cast<size_t>(fa.maxDynamicSharedSizeBytes);
    const size_t want =
        std::min(sizeof(float) * 24576, ceiling);
    if (want <= 49152) return false;  // nothing to opt in for
    cudaError_t e = cudaFuncSetAttribute(
        rmsnorm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(want));
    if (e != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    return true;
  }();
  if (!attr_ok && shmem > 49152)
    throw std::runtime_error(
        "rmsnorm: dynamic smem exceeds driver-accepted limit");
  cudaGetLastError();  // start the launch with a clean error slate
  rmsnorm_kernel<<<rows, kBlock, shmem, stream>>>(
      static_cast<const uint16_t*>(x), static_cast<const uint16_t*>(weight),
      static_cast<uint16_t*>(y), dim, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void swiglu_limit_bf16(const void* gate, const void* up, void* out,
                       int64_t n_elems, float limit, cudaStream_t stream) {
  constexpr int kBlock = 256;
  int64_t grid64 = (n_elems + kBlock - 1) / kBlock;
  unsigned grid = static_cast<unsigned>(grid64 > 2147483647 ? 2147483647 : grid64);
  swiglu_limit_kernel<<<grid, kBlock, 0, stream>>>(
      static_cast<const uint16_t*>(gate), static_cast<const uint16_t*>(up),
      static_cast<uint16_t*>(out), n_elems, limit);
  DGPP_CUDA_OK(cudaGetLastError());
}

void swiglu_gateup_pairs_bf16(const void* gateup, void* out, int rows,
                              int inter, float limit, cudaStream_t stream) {
  constexpr int kBlock = 256;
  int64_t total = static_cast<int64_t>(rows) * inter;
  unsigned grid = static_cast<unsigned>(
      std::min<uint64_t>((total + kBlock - 1) / kBlock, 1u << 24));
  swiglu_pairs_kernel<<<grid, kBlock, 0, stream>>>(
      static_cast<const uint16_t*>(gateup), static_cast<uint16_t*>(out),
      total, inter, limit);
  DGPP_CUDA_OK(cudaGetLastError());
}

template <int S>
void mhc_mix_launch(const void* x, const void* mix, void* out, int batch,
                    int hidden, cudaStream_t stream) {
  int block = 256;
  dim3 grid(batch, (hidden + block - 1) / block);
  mhc_mix_kernel<S><<<grid, block, 0, stream>>>(
      static_cast<const uint16_t*>(x), static_cast<const float*>(mix),
      static_cast<uint16_t*>(out), hidden);
  DGPP_CUDA_OK(cudaGetLastError());
}

void mhc_mix_bf16(const void* x, const void* mix, void* out, int batch,
                  int hidden, cudaStream_t stream) {
  mhc_mix_launch<4>(x, mix, out, batch, hidden, stream);
}

void add_inplace_bf16(void* x, const void* y, int64_t n_elems,
                      cudaStream_t stream) {
  constexpr int kBlock = 256;
  unsigned grid = static_cast<unsigned>(
      std::min<uint64_t>((n_elems + kBlock - 1) / kBlock, 1u << 24));
  add_inplace_kernel<<<grid, kBlock, 0, stream>>>(
      static_cast<uint16_t*>(x), static_cast<const uint16_t*>(y), n_elems);
  DGPP_CUDA_OK(cudaGetLastError());
}

void cast_bf16_to_f32_rows(const void* src_bf16, float* dst_f32, int rows,
                           int cols, cudaStream_t stream) {
  cast_rows_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const uint16_t*>(src_bf16), dst_f32, cols);
  DGPP_CUDA_OK(cudaGetLastError());
}

void cast_bf16_to_fp8_rows(const void* src_bf16, void* dst_fp8,
                           int64_t n_elems, cudaStream_t stream) {
  constexpr int kBlock = 256;
  unsigned grid = static_cast<unsigned>(
      std::min<uint64_t>((n_elems + kBlock - 1) / kBlock, 1u << 24));
  cast_fp8_kernel<<<grid, kBlock, 0, stream>>>(
      static_cast<const uint16_t*>(src_bf16),
      static_cast<uint8_t*>(dst_fp8), n_elems);
  DGPP_CUDA_OK(cudaGetLastError());
}

void kv_append_from_pairs(const void* src, int64_t src_stride_elems,
                          void* k_cache, void* v_cache,
                          const int* dev_abs_first, int n_rows, int kv_dim,
                          cudaStream_t stream) {
  kv_append_kernel<<<n_rows, 128, 0, stream>>>(
      static_cast<const uint16_t*>(src), src_stride_elems,
      static_cast<uint16_t*>(k_cache), static_cast<uint16_t*>(v_cache),
      dev_abs_first, n_rows, kv_dim);
  DGPP_CUDA_OK(cudaGetLastError());
}

void bump_i32_device(int32_t* p, int delta, cudaStream_t stream) {
  bump_i32_kernel<<<1, 32, 0, stream>>>(p, delta);
  DGPP_CUDA_OK(cudaGetLastError());
}

void copy_i64_device_to_device(const int64_t* src, int64_t* dst,
                               cudaStream_t stream) {
  cpy_i64_kernel<<<1, 32, 0, stream>>>(src, dst);
  DGPP_CUDA_OK(cudaGetLastError());
}

void argmax_rows_f32(const float* logits, int64_t* out_idx, float* out_val,
                     int rows, int cols, cudaStream_t stream) {
  int block = std::min(1024, ((cols + 31) / 32) * 32);
  if (block <= 0) block = 32;
  argmax_rows_kernel<<<rows, block, 0, stream>>>(logits, out_idx, out_val,
                                                 cols);
  DGPP_CUDA_OK(cudaGetLastError());
}

void embed_gather_bf16(const void* table, const int64_t* tokens, void* out,
                       int num_tokens, int hidden, cudaStream_t stream) {
  embed_gather_kernel<<<num_tokens, 128, 0, stream>>>(
      static_cast<const uint16_t*>(table), tokens,
      static_cast<uint16_t*>(out), hidden);
  DGPP_CUDA_OK(cudaGetLastError());
}

void embed_gather_sliced_bf16(const void* table, const int64_t* tokens, void* out,
                              int num_tokens, int hidden, int64_t vocab_begin,
                              int64_t vocab_count, cudaStream_t stream) {
  embed_gather_sliced_kernel<<<num_tokens, 128, 0, stream>>>(
      static_cast<const uint16_t*>(table), tokens, static_cast<uint16_t*>(out), hidden,
      vocab_begin, vocab_count);
  DGPP_CUDA_OK(cudaGetLastError());
}

void copy_rows_bf16(void* dst, const void* src, int num_tokens, int hidden,
                    cudaStream_t stream) {
  copy_rows_kernel<<<num_tokens, 128, 0, stream>>>(static_cast<uint16_t*>(dst),
                                                     static_cast<const uint16_t*>(src), hidden);
  DGPP_CUDA_OK(cudaGetLastError());
}

void fill_random_normal_bf16(void* dst, uint64_t n, uint64_t seed,
                             float stddev, cudaStream_t stream) {
  int block = 256;
  uint64_t grid = (n + block - 1) / block;
  fill_normal_bf16_kernel<<<static_cast<unsigned>(std::min<uint64_t>(grid, 1u << 24)), block, 0, stream>>>(
      static_cast<uint16_t*>(dst), n, seed, stddev);
  DGPP_CUDA_OK(cudaGetLastError());
}

void fill_random_normal_fp8(void* dst, uint64_t n, uint64_t seed,
                            float stddev, cudaStream_t stream) {
  int block = 256;
  uint64_t grid = (n + block - 1) / block;
  fill_normal_fp8_kernel<<<static_cast<unsigned>(std::min<uint64_t>(grid, 1u << 24)), block, 0, stream>>>(
      static_cast<uint8_t*>(dst), n, seed, stddev);
  DGPP_CUDA_OK(cudaGetLastError());
}

void fill_random_normal_f32(void* dst, uint64_t n, uint64_t seed,
                            float stddev, cudaStream_t stream) {
  int block = 256;
  uint64_t grid = (n + block - 1) / block;
  fill_normal_f32_kernel<<<static_cast<unsigned>(std::min<uint64_t>(grid, 1u << 24)), block, 0, stream>>>(
      static_cast<float*>(dst), n, seed, stddev);
  DGPP_CUDA_OK(cudaGetLastError());
}

__global__ void globaltimer_stamp_kernel(uint64_t* out) {
  dgpp_pdl_wait();
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  *out = t;
}

void launch_globaltimer_stamp(uint64_t* out, cudaStream_t stream) {
  globaltimer_stamp_kernel<<<1, 1, 0, stream>>>(out);
}

}  // namespace dgpp