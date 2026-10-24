// DSA/MLA forward kernels (see dsa.hpp for the layout and determinism
// contract). The streaming top-k selection is shared by the decode, prefill,
// and merge paths through a key-source functor: every path produces the same
// composite keys ((sortable_fp32 << 21) | pool_idx), so the selection spec —
// highest logit, exact ties to the lower pool index, ascending output — is
// implemented exactly once.
//
// expf (not __expf) everywhere exp appears: reference parity comes before
// the ~2 ulp a fast intrinsic would save; revisit only with M9 profiling
// evidence.
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "kernels/pdl.cuh"
#include "common/dtypes.hpp"
#include "kernels/dsa.hpp"
#include "kernels/latent_format.hpp"
#include "kernels/topk_select.cuh"

namespace dgpp {

static_assert(kDsaSelectMaxK == kSelectMaxK, "the select bound is one number");

namespace {

using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;
using dgpp::float_to_fp8_e4m3_bits;
using dgpp::fp8_e4m3_bits_to_float;

// ---- native conversion helpers ----------------------------------------
// The GB10 has hardware e4m3->f16 and bf16->f32 conversions; the software
// decoders in dtypes.hpp are branchy (NaN/denormal special cases) and were
// ~90% of the select kernel's cycles. Both paths are EXACT — e4m3 fits
// entirely inside f16, bf16 inside f32 — so these are bit-identical
// replacements, verified by the bitwise select fuzz.

__device__ inline float2 fp8x2_to_float2(uint16_t v) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(v, __NV_E4M3);
  return __half22float2(__half2(h));
}

__device__ inline float2 bf16x2_to_float2(uint32_t v) {
  __nv_bfloat16_raw lo, hi;
  lo.x = static_cast<unsigned short>(v & 0xFFFFu);
  hi.x = static_cast<unsigned short>(v >> 16);
  return __bfloat1622float2(
      __nv_bfloat162(__nv_bfloat16(lo), __nv_bfloat16(hi)));
}

// acc += sum(element-wise products of 8 bf16 pairs), accumulated in element
// order (same sequence as the scalar loop it replaces — FFMA-friendly; the
// attention path is tolerance-pinned, not bitwise).
__device__ inline float dot8_bf16(uint4 a, uint4 b, float acc) {
  const uint32_t* a32 = reinterpret_cast<const uint32_t*>(&a);
  const uint32_t* b32 = reinterpret_cast<const uint32_t*>(&b);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 x = bf16x2_to_float2(a32[j]);
    const float2 y = bf16x2_to_float2(b32[j]);
    acc += x.x * y.x;
    acc += x.y * y.y;
  }
  return acc;
}

// ---- the latent cache's formats (kernels/latent_format.hpp) ------------
// The attention kernels gather a latent tile into bf16 shared memory; a
// quantized cache dequantizes on the way in (the codes and the row scale
// -> the bf16 the host oracle sees, bitwise) and the math past the load
// is the bf16 kernel's. load8 fills 8 consecutive elements at element
// offset `e` (a multiple of 8) of physical row `phys`; load1 one element.
// A row is `row_bytes` long: the format's payload of kv_lora elements
// (latent_row_bytes) then the bf16 rope tail (plan D3) — an element
// offset at or past kv_lora reads the tail, in every format.
__device__ __forceinline__ uint32_t pack_bf16x2(uint16_t lo, uint16_t hi) {
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

template <LatentFormat F>
struct LatentTile;

template <>
struct LatentTile<LatentFormat::kBf16> {
  // The payload and the tail are one bf16 row: no branch.
  static __device__ __forceinline__ uint4 load8(const uint8_t* cache,
                                                const float*, int64_t phys,
                                                size_t row_bytes, int, int e) {
    return *reinterpret_cast<const uint4*>(cache + phys * int64_t(row_bytes) + e * 2);
  }
  static __device__ __forceinline__ uint16_t load1(const uint8_t* cache,
                                                   const float*, int64_t phys,
                                                   size_t row_bytes, int, int e) {
    return *reinterpret_cast<const uint16_t*>(cache + phys * int64_t(row_bytes) + e * 2);
  }
};

template <>
struct LatentTile<LatentFormat::kFp8> {
  // The hardware e4m3 -> f16 conversion is exact (the select kernel's pin);
  // one fp32 multiply by the row scale and a bf16 rounding follow, exactly
  // latent_fp8_decode_bf16.
  static __device__ __forceinline__ uint32_t decode2(uint16_t codes, float s) {
    const float2 v = fp8x2_to_float2(codes);
    return pack_bf16x2(float_to_bf16_bits(v.x * s), float_to_bf16_bits(v.y * s));
  }
  static __device__ __forceinline__ uint4 load8(const uint8_t* cache,
                                                const float* scales,
                                                int64_t phys, size_t row_bytes,
                                                int kv_lora, int e) {
    const uint8_t* row = cache + phys * int64_t(row_bytes);
    if (e >= kv_lora)
      return *reinterpret_cast<const uint4*>(row + kv_lora + (e - kv_lora) * 2);
    const uint2 raw = *reinterpret_cast<const uint2*>(row + e);
    const float s = scales[phys];
    uint4 out;
    out.x = decode2(static_cast<uint16_t>(raw.x & 0xFFFFu), s);
    out.y = decode2(static_cast<uint16_t>(raw.x >> 16), s);
    out.z = decode2(static_cast<uint16_t>(raw.y & 0xFFFFu), s);
    out.w = decode2(static_cast<uint16_t>(raw.y >> 16), s);
    return out;
  }
  static __device__ __forceinline__ uint16_t load1(const uint8_t* cache,
                                                   const float* scales,
                                                   int64_t phys, size_t row_bytes,
                                                   int kv_lora, int e) {
    const uint8_t* row = cache + phys * int64_t(row_bytes);
    if (e >= kv_lora)
      return *reinterpret_cast<const uint16_t*>(row + kv_lora + (e - kv_lora) * 2);
    return latent_fp8_decode_bf16(row[e], scales[phys]);
  }
};

template <>
struct LatentTile<LatentFormat::kFp4> {
  // Both e2m1 values and finite e4m3 block scales fit exactly in fp16.
  // Decode pairs in hardware, preserving the codec's two fp32 multiplies
  // and final bf16 rounding. Scalar software conversion inside every
  // gathered tile was a substantial full-GLM prefill cost.
  static __device__ __forceinline__ uint32_t decode2(uint8_t codes, float s) {
    const __half2_raw h = __nv_cvt_fp4x2_to_halfraw2(codes, __NV_E2M1);
    const float2 v = __half22float2(__half2(h));
    return pack_bf16x2(float_to_bf16_bits(v.x * s), float_to_bf16_bits(v.y * s));
  }
  static __device__ __forceinline__ float block_scale(uint8_t code, float s) {
    return fp8x2_to_float2(code).x * s;
  }
  static __device__ __forceinline__ uint4 load8(const uint8_t* cache,
                                                const float* scales,
                                                int64_t phys, size_t row_bytes,
                                                int kv_lora, int e) {
    const uint8_t* row = cache + phys * int64_t(row_bytes);
    if (e >= kv_lora)
      return *reinterpret_cast<const uint4*>(
          row + latent_row_bytes(LatentFormat::kFp4, kv_lora) + (e - kv_lora) * 2);
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(row + (e >> 1));
    const float S = block_scale(
        row[latent_fp4_scale_offset(kv_lora) + (e / kLatentFp4Block)], scales[phys]);
    uint4 out;
    out.x = decode2(static_cast<uint8_t>(raw), S);
    out.y = decode2(static_cast<uint8_t>(raw >> 8), S);
    out.z = decode2(static_cast<uint8_t>(raw >> 16), S);
    out.w = decode2(static_cast<uint8_t>(raw >> 24), S);
    return out;
  }
  static __device__ __forceinline__ uint16_t load1(const uint8_t* cache,
                                                   const float* scales,
                                                   int64_t phys, size_t row_bytes,
                                                   int kv_lora, int e) {
    const uint8_t* row = cache + phys * int64_t(row_bytes);
    if (e >= kv_lora)
      return *reinterpret_cast<const uint16_t*>(
          row + latent_row_bytes(LatentFormat::kFp4, kv_lora) + (e - kv_lora) * 2);
    const float S = block_scale(
        row[latent_fp4_scale_offset(kv_lora) + (e / kLatentFp4Block)], scales[phys]);
    const uint8_t byte = row[e >> 1];
    const uint32_t pair = decode2(byte, S);
    return static_cast<uint16_t>((e & 1) ? (pair >> 16) : pair);
  }
};

// fp8_block (2026-09-13, DeepSeek-V4.1's window rows): e4m3 codes with an
// e8m0 scale per 32 inside the row, no row scale — the decode is one
// exact fp32 multiply by a power of two and a bf16 rounding that changes
// nothing (latent_fp8_block_decode_bf16).
template <>
struct LatentTile<LatentFormat::kFp8Block> {
  static __device__ __forceinline__ uint32_t decode2(uint16_t codes, float s) {
    const float2 v = fp8x2_to_float2(codes);
    return pack_bf16x2(float_to_bf16_bits(v.x * s), float_to_bf16_bits(v.y * s));
  }
  static __device__ __forceinline__ uint4 load8(const uint8_t* cache,
                                                const float*, int64_t phys,
                                                size_t row_bytes, int kv_lora, int e) {
    const uint8_t* row = cache + phys * int64_t(row_bytes);
    if (e >= kv_lora)
      return *reinterpret_cast<const uint4*>(
          row + latent_row_bytes(LatentFormat::kFp8Block, kv_lora) + (e - kv_lora) * 2);
    const uint2 raw = *reinterpret_cast<const uint2*>(row + e);
    const float s = e8m0_byte_to_float(
        row[latent_fp8_block_scale_offset(kv_lora) + (e / kLatentFp8BlockGroup)]);
    uint4 out;
    out.x = decode2(static_cast<uint16_t>(raw.x & 0xFFFFu), s);
    out.y = decode2(static_cast<uint16_t>(raw.x >> 16), s);
    out.z = decode2(static_cast<uint16_t>(raw.y & 0xFFFFu), s);
    out.w = decode2(static_cast<uint16_t>(raw.y >> 16), s);
    return out;
  }
  static __device__ __forceinline__ uint16_t load1(const uint8_t* cache,
                                                   const float*, int64_t phys,
                                                   size_t row_bytes, int kv_lora, int e) {
    const uint8_t* row = cache + phys * int64_t(row_bytes);
    if (e >= kv_lora)
      return *reinterpret_cast<const uint16_t*>(
          row + latent_row_bytes(LatentFormat::kFp8Block, kv_lora) + (e - kv_lora) * 2);
    const float s = e8m0_byte_to_float(
        row[latent_fp8_block_scale_offset(kv_lora) + (e / kLatentFp8BlockGroup)]);
    return latent_fp8_block_decode_bf16(row[e], s);
  }
};

// fp4_block (DeepSeek-V4.1's compressed main KV): e2m1 codes with an
// ABSOLUTE e4m3 scale per 16 (no row scale); the product of two short
// mantissas is exact in fp32 and in bf16.
template <>
struct LatentTile<LatentFormat::kFp4Block> {
  static __device__ __forceinline__ uint4 load8(const uint8_t* cache,
                                                const float*, int64_t phys,
                                                size_t row_bytes, int kv_lora, int e) {
    const uint8_t* row = cache + phys * int64_t(row_bytes);
    if (e >= kv_lora)
      return *reinterpret_cast<const uint4*>(
          row + latent_row_bytes(LatentFormat::kFp4Block, kv_lora) + (e - kv_lora) * 2);
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(row + (e >> 1));
    const float S = latent_fp4_block_scale_abs(row[latent_fp4_scale_offset(kv_lora) + (e / kLatentFp4Block)]);
    uint16_t v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
      v[j] = latent_fp4_decode_bf16(static_cast<uint8_t>((raw >> (4 * j)) & 0xFu), S);
    uint4 out;
    out.x = pack_bf16x2(v[0], v[1]);
    out.y = pack_bf16x2(v[2], v[3]);
    out.z = pack_bf16x2(v[4], v[5]);
    out.w = pack_bf16x2(v[6], v[7]);
    return out;
  }
  static __device__ __forceinline__ uint16_t load1(const uint8_t* cache,
                                                   const float*, int64_t phys,
                                                   size_t row_bytes, int kv_lora, int e) {
    const uint8_t* row = cache + phys * int64_t(row_bytes);
    if (e >= kv_lora)
      return *reinterpret_cast<const uint16_t*>(
          row + latent_row_bytes(LatentFormat::kFp4Block, kv_lora) + (e - kv_lora) * 2);
    const float S = latent_fp4_block_scale_abs(row[latent_fp4_scale_offset(kv_lora) + (e / kLatentFp4Block)]);
    const uint8_t byte = row[e >> 1];
    return latent_fp4_decode_bf16(
        static_cast<uint8_t>((e & 1) ? (byte >> 4) : (byte & 0xFu)), S);
  }
};

// The cache row's stride: the format's payload plus the bf16 rope tail.
__host__ __device__ constexpr size_t latent_cache_row_bytes(LatentFormat f, int kv_lora,
                                                            int rope) {
  return latent_row_bytes(f, kv_lora) + static_cast<size_t>(rope) * 2;
}

constexpr float kFp8Max = 448.0f;
constexpr float kAbsmaxFloor = 1e-4f;
constexpr int kAttnTile = 32;  // latent rows per attention tile
constexpr int kAttnMaxDslice = 64;  // a thread's dim window (registers)
constexpr int kAttnMaxRopeSlice = 8;  // a thread's share of the 64-wide rope tail

// Smallest power of two >= v; exact powers map to themselves (see the
// reference header for why this replaces exp2f(ceilf(log2f(v)))).
__device__ inline float next_pow2_dev(float v) {
  uint32_t b = __float_as_uint(v);
  int e = int((b >> 23) & 0xFFu) - 127;
  bool exact = (b & 0x7FFFFFu) == 0;
  return exp2f(float(exact ? e : e + 1));
}

// Block-wide max over one value per thread (blockDim <= 1024).
__device__ inline float block_max(float v, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) / 32;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_xor_sync(~0u, v, off));
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  v = (threadIdx.x < nwarps) ? scratch[threadIdx.x] : -INFINITY;
  if (warp == 0) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      v = fmaxf(v, __shfl_xor_sync(~0u, v, off));
    if (lane == 0) scratch[0] = v;
  }
  __syncthreads();
  return scratch[0];
}

// In-place Hadamard-128 over smem, cooperative across the block.
__device__ inline void fwht128_smem(float* x) {
  for (int stride = 1; stride < 128; stride <<= 1) {
    for (int t = threadIdx.x; t < 64; t += blockDim.x) {
      const int p = (t / stride) * (2 * stride) + (t % stride);
      const float a = x[p];
      const float b = x[p + stride];
      x[p] = a + b;
      x[p + stride] = a - b;
    }
    __syncthreads();
  }
  if (threadIdx.x < 128) x[threadIdx.x] *= 0.08838834764831845f;  // 1/sqrt(128)
  __syncthreads();
}


// ---------------------------------------------------------------------
// Elementwise paths
// ---------------------------------------------------------------------

__global__ void fwht_quant_rows_kernel(const uint16_t* q, uint8_t* q_fp8,
                                       float* q_scale) {
  dgpp_pdl_wait();
  extern __shared__ float xs[];  // [128]
  const int64_t r = blockIdx.x;
  const uint16_t* row = q + r * 128;
  if (threadIdx.x < 128)
    xs[threadIdx.x] = bf16_bits_to_float(row[threadIdx.x]);
  __syncthreads();
  fwht128_smem(xs);
  // bf16 round before quantization (pinned boundary).
  if (threadIdx.x < 128)
    xs[threadIdx.x] =
        bf16_bits_to_float(float_to_bf16_bits(xs[threadIdx.x]));
  __syncthreads();
  float absmax = (threadIdx.x < 128) ? fabsf(xs[threadIdx.x]) : 0.0f;
  __shared__ float red[32];
  absmax = block_max(absmax, red);
  absmax = fmaxf(absmax, kAbsmaxFloor);
  const float scale = next_pow2_dev(absmax * (1.0f / kFp8Max));
  if (threadIdx.x == 0) q_scale[r] = scale;
  if (threadIdx.x < 128)
    q_fp8[r * 128 + threadIdx.x] =
        float_to_fp8_e4m3_bits(xs[threadIdx.x] / scale);
}

__global__ void fold_weights_kernel(const float* weights, const float* q_scale,
                                    float* out, int64_t n, float scale) {
  dgpp_pdl_wait();
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = (weights[i] * q_scale[i]) * scale;
}

__global__ void k_layernorm_kernel(const uint16_t* k_raw, int64_t k_stride,
                                   const uint16_t* w, const uint16_t* b,
                                   uint16_t* k_out, int dim, float eps) {
  dgpp_pdl_wait();
  const int64_t r = blockIdx.x;
  const uint16_t* row = k_raw + r * k_stride;
  __shared__ float red[32];
  float v = (threadIdx.x < dim) ? bf16_bits_to_float(row[threadIdx.x]) : 0.0f;
  float sum = warp_sum(v);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.0f;
    for (int i = 0; i < (dim + 31) / 32; ++i) t += red[i];
    red[0] = t / dim;
  }
  __syncthreads();
  const float d0 = (threadIdx.x < dim) ? v - red[0] : 0.0f;
  // Everyone has the mean now; red is reused for the variance partials.
  __syncthreads();
  float var = warp_sum(d0 * d0);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = var;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.0f;
    for (int i = 0; i < (dim + 31) / 32; ++i) t += red[i];
    red[0] = t / dim;
  }
  __syncthreads();
  if (threadIdx.x < dim) {
    const float inv = rsqrtf(red[0] + eps);
    const float y = d0 * inv * bf16_bits_to_float(w[threadIdx.x]) +
                    bf16_bits_to_float(b[threadIdx.x]);
    k_out[r * dim + threadIdx.x] = float_to_bf16_bits(y);
  }
}

__global__ void fused_qkv_rmsnorm_kernel(const uint16_t* qkv, int64_t row_stride,
                                         uint16_t* q_c, uint16_t* kv_c, int q_dim,
                                         int kv_dim, const uint16_t* q_w,
                                         const uint16_t* kv_w, float eps) {
  dgpp_pdl_wait();
  const int64_t r = blockIdx.x;
  const uint16_t* row = qkv + r * row_stride;
  __shared__ float red[32];
  __shared__ float inv[2];
  float ss = 0.0f;
  for (int d = threadIdx.x; d < q_dim; d += blockDim.x) {
    const float v = bf16_bits_to_float(row[d]);
    ss += v * v;
  }
  ss = warp_sum(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.0f;
    for (int i = 0; i < (int)((blockDim.x + 31) / 32); ++i) t += red[i];
    inv[0] = rsqrtf(t / q_dim + eps);
  }
  // red is reused for the kv-half partials only after thread 0 has read it.
  __syncthreads();
  ss = 0.0f;
  for (int d = threadIdx.x; d < kv_dim; d += blockDim.x) {
    const float v = bf16_bits_to_float(row[q_dim + d]);
    ss += v * v;
  }
  ss = warp_sum(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.0f;
    for (int i = 0; i < (int)((blockDim.x + 31) / 32); ++i) t += red[i];
    inv[1] = rsqrtf(t / kv_dim + eps);
  }
  __syncthreads();
  for (int d = threadIdx.x; d < q_dim; d += blockDim.x)
    q_c[r * q_dim + d] = float_to_bf16_bits(
        bf16_bits_to_float(row[d]) * inv[0] * bf16_bits_to_float(q_w[d]));
  for (int d = threadIdx.x; d < kv_dim; d += blockDim.x)
    kv_c[r * kv_dim + d] =
        float_to_bf16_bits(bf16_bits_to_float(row[q_dim + d]) * inv[1] *
                           bf16_bits_to_float(kv_w[d]));
}

// ---------------------------------------------------------------------
// Index cache + tail machinery
// ---------------------------------------------------------------------

// Shared compression tail: bf16 round -> Hadamard-128 -> bf16 round ->
// absmax fp8 quant, writing the fp8 row and its power-of-two scale.
// xs (smem, [dim]) holds the per-dim softmax-weighted sum on entry.
__device__ inline void compress_write_quant(float* xs, uint8_t* slot_k,
                                            float* slot_scale, int dim) {
  __shared__ float red[32];
  fwht128_smem(xs);
  if (threadIdx.x < 128)
    xs[threadIdx.x] =
        bf16_bits_to_float(float_to_bf16_bits(xs[threadIdx.x]));
  __syncthreads();
  float absmax = (threadIdx.x < dim) ? fabsf(xs[threadIdx.x]) : 0.0f;
  absmax = block_max(absmax, red);
  absmax = fmaxf(absmax, kAbsmaxFloor);
  const float scale = next_pow2_dev(absmax * (1.0f / kFp8Max));
  if (threadIdx.x == 0) *slot_scale = scale;
  if (threadIdx.x < dim)
    slot_k[threadIdx.x] = float_to_fp8_e4m3_bits(xs[threadIdx.x] / scale);
}

__global__ void kpool_compress_write_kernel(
    const uint16_t* k, int64_t k_stride, const uint16_t* gate,
    int64_t gate_stride, const float* ape, const int32_t* block_table,
    int pools_per_block, int64_t first_pool, int n_pools, uint8_t* index_k,
    float* index_scale, int kpool, int dim) {
  dgpp_pdl_wait();
  const int i = blockIdx.x;
  if (i >= n_pools) return;
  const int64_t pool = first_pool + i;
  const int32_t blk = block_table[pool / pools_per_block];
  const int64_t slot =
      int64_t(blk) * pools_per_block + (pool % pools_per_block);

  extern __shared__ float xs[];  // [dim]
  const int d = threadIdx.x;
  float scores[8];
  float maxs = -INFINITY;
  // kpool 1 (the full model): no gate, no positional bias — the softmax
  // over one slot is exactly 1 and the entry is the token's key.
  if (d < dim) {
    for (int s = 0; s < kpool; ++s) {
      const float g = gate ? bf16_bits_to_float(gate[int64_t(i * kpool + s) * gate_stride + d])
                           : 0.0f;
      const float score = g + (ape ? ape[int64_t(s) * dim + d] : 0.0f);
      scores[s] = score;
      maxs = fmaxf(maxs, score);
    }
  }
  float acc = 0.0f;
  float denom = 0.0f;
  if (d < dim) {
    for (int s = 0; s < kpool; ++s) {
      const float prob = expf(scores[s] - maxs);
      denom += prob;
      acc +=
          bf16_bits_to_float(k[int64_t(i * kpool + s) * k_stride + d]) * prob;
    }
    xs[d] = acc / denom;
  }
  __syncthreads();
  compress_write_quant(xs, index_k + slot * dim, index_scale + slot, dim);
}

__global__ void kpool_tail_seed_kernel(const uint16_t* k, int64_t k_stride,
                                       const uint16_t* gate,
                                       int64_t gate_stride,
                                       const int32_t* req_ids,
                                       const int64_t* pos, int64_t tokens,
                                       uint16_t* tail, int kpool, int dim) {
  dgpp_pdl_wait();
  const int64_t i = blockIdx.x;
  if (i >= tokens) return;
  const int32_t req = req_ids[i];
  const int64_t p = pos[i];
  if (p < 0) return;
  // Seed iff the token kpool ahead is the same request still in this batch
  // (the reference's ahead-check): only the request's last kpool tokens
  // survive in the ring. The ahead index is arithmetically clamped — the
  // compiler speculates both ternary arms' loads past any bounds check, and
  // a speculated OOB read of garbage would corrupt the seeding decision.
  const int64_t ahead_idx = min(i + kpool, tokens - 1);
  const bool ahead_in_batch = (i + kpool < tokens);
  const bool ahead_same = ahead_in_batch &&
                          req_ids[ahead_idx] == req &&
                          pos[ahead_idx] >= 0;
  if (ahead_same) return;
  const int slot = int(p % kpool);
  const int64_t kbase = (int64_t(req) * 2 * kpool + slot) * dim;
  const int64_t gbase = (int64_t(req) * 2 * kpool + kpool + slot) * dim;
  const int d = threadIdx.x;
  if (d < dim) {
    tail[kbase + d] = k[i * k_stride + d];
    tail[gbase + d] = gate ? gate[i * gate_stride + d] : uint16_t(0);
  }
}

__global__ void kpool_decode_update_kernel(
    const uint16_t* k, int64_t k_stride, const uint16_t* gate,
    int64_t gate_stride, const float* ape, const int32_t* req_ids,
    const int64_t* pos, const int32_t* req_spans,
    const int32_t* block_tables,
    int blocks_per_request, uint16_t* tail, uint8_t* index_k,
    float* index_scale, int pools_per_block, int kpool, int dim,
    uint16_t* tail_snapshots) {
  dgpp_pdl_wait();
  const int span = blockIdx.x;
  const int t0 = req_spans[span * 2];
  const int t1 = t0 + req_spans[span * 2 + 1];
  int first_real = t0;
  while (first_real < t1 && pos[first_real] < 0) ++first_real;
  if (first_real == t1) return;
  const int req = req_ids[first_real];
  const int d = threadIdx.x;
  extern __shared__ float xs[];  // [dim]
  const int ring_elems = 2 * kpool * dim;

  for (int t = t0; t < t1; ++t) {
    const int64_t p = pos[t];
    if (p < 0) continue;  // uniform across threads (p is batch metadata)
    const int slot = int(p % kpool);
    const bool completing = (slot == kpool - 1);

    if (completing) {
      // The pool spans [p-kpool+1, p]; ring slot of member s is
      // (pool_start+s) % kpool; the current token overrides its own slot
      // (which still holds one pool's stale stash — the reference's
      // is_current rule).
      const int64_t pool_start = p - (kpool - 1);
      float scores[8];
      float maxs = -INFINITY;
      if (d < dim) {
        for (int s = 0; s < kpool; ++s) {
          const int ring = int((pool_start + s) % kpool);
          float g = 0.0f;
          if (gate)
            g = (s == kpool - 1)
                    ? bf16_bits_to_float(gate[int64_t(t) * gate_stride + d])
                    : bf16_bits_to_float(
                          tail[(int64_t(req) * 2 * kpool + kpool + ring) * dim +
                               d]);
          const float score = g + (ape ? ape[int64_t(s) * dim + d] : 0.0f);
          scores[s] = score;
          maxs = fmaxf(maxs, score);
        }
      }
      float acc = 0.0f;
      float denom = 0.0f;
      if (d < dim) {
        for (int s = 0; s < kpool; ++s) {
          const int ring = int((pool_start + s) % kpool);
          const float kk =
              (s == kpool - 1)
                  ? bf16_bits_to_float(k[int64_t(t) * k_stride + d])
                  : bf16_bits_to_float(
                        tail[(int64_t(req) * 2 * kpool + ring) * dim + d]);
          const float prob = expf(scores[s] - maxs);
          denom += prob;
          acc += kk * prob;
        }
        xs[d] = acc / denom;
      }
      __syncthreads();
      const int64_t pool = p / kpool;
      const int32_t blk = block_tables[int64_t(req) * blocks_per_request +
                                       pool / pools_per_block];
      const int64_t phys =
          int64_t(blk) * pools_per_block + (pool % pools_per_block);
      compress_write_quant(xs, index_k + phys * dim, index_scale + phys, dim);
      __syncthreads();  // ring reads done; the stash may overwrite
    }

    // Stash AFTER the completion read (ordering pinned by the reference).
    if (d < dim) {
      tail[(int64_t(req) * 2 * kpool + slot) * dim + d] =
          k[int64_t(t) * k_stride + d];
      tail[(int64_t(req) * 2 * kpool + kpool + slot) * dim + d] =
          gate ? gate[int64_t(t) * gate_stride + d] : uint16_t(0);
    }
    // Speculative rows: the ring after batch row t is the state to restore
    // if rows > t are rejected (the last row's ring stays in place). The
    // ring is the one non-idempotent DSA write — latent rows and completed
    // pools are positional and a rewound position simply overwrites them.
    if (tail_snapshots && t + 1 < t1) {
      __syncthreads();  // every thread's stash is visible before the copy
      const uint16_t* ring = tail + int64_t(req) * ring_elems;
      uint16_t* snap = tail_snapshots + int64_t(t) * ring_elems;
      for (int e = threadIdx.x; e < ring_elems; e += blockDim.x)
        snap[e] = ring[e];
    }
  }
}

__device__ __forceinline__ int64_t latent_physical_slot(
    const int32_t* req_ids, const int64_t* pos, const int32_t* block_tables,
    int blocks_per_request, int block_tokens, int64_t i) {
  const int64_t p = pos[i];
  if (p < 0) return -1;
  const int32_t req = req_ids[i];
  const int32_t blk =
      block_tables[int64_t(req) * blocks_per_request + p / block_tokens];
  return int64_t(blk) * block_tokens + (p % block_tokens);
}

// The bf16 rope tail after the row's payload (rope a multiple of 8; the
// payload's byte length a multiple of 16 in every format).
__device__ __forceinline__ void latent_append_tail(const uint16_t* rope_rows,
                                                   int rope, int64_t i,
                                                   uint8_t* row_tail) {
  if (rope <= 0) return;
  const uint4* s4 = reinterpret_cast<const uint4*>(rope_rows + i * rope);
  uint4* d4 = reinterpret_cast<uint4*>(row_tail);
  for (int j = threadIdx.x; j < rope / 8; j += blockDim.x) d4[j] = s4[j];
}

__global__ void latent_append_kernel(const uint16_t* latent_rows,
                                     const int32_t* req_ids,
                                     const int64_t* pos,
                                     const int32_t* block_tables,
                                     int blocks_per_request, int block_tokens,
                                     uint8_t* latent_cache, int kv_lora,
                                     size_t row_bytes, const uint16_t* rope_rows,
                                     int rope) {
  dgpp_pdl_wait();
  const int64_t i = blockIdx.x;
  const int64_t phys = latent_physical_slot(req_ids, pos, block_tables,
                                            blocks_per_request, block_tokens, i);
  if (phys < 0) return;
  uint8_t* row = latent_cache + phys * int64_t(row_bytes);
  const uint4* s4 = reinterpret_cast<const uint4*>(latent_rows + i * kv_lora);
  uint4* d4 = reinterpret_cast<uint4*>(row);
  for (int j = threadIdx.x; j < kv_lora / 8; j += blockDim.x) d4[j] = s4[j];
  latent_append_tail(rope_rows, rope, i, row + size_t(kv_lora) * 2);
}

// The row's absmax across the block's threads (max is exact and
// order-free, so the reduction is deterministic by construction).
constexpr int kLatentAppendThreads = 128;
__device__ __forceinline__ float latent_block_absmax(float mine, float* red) {
  red[threadIdx.x] = mine;
  __syncthreads();
  for (int s = kLatentAppendThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  return red[0];
}

// fp8: thread t quantizes elements [8t, 8t+8) (kv_lora <= 1024, a
// multiple of 8 — validate_config's pin) with the row's scale.
__global__ void latent_append_fp8_kernel(const uint16_t* latent_rows,
                                         const int32_t* req_ids,
                                         const int64_t* pos,
                                         const int32_t* block_tables,
                                         int blocks_per_request,
                                         int block_tokens, uint8_t* latent_cache,
                                         float* latent_scale, int kv_lora,
                                         size_t row_bytes, const uint16_t* rope_rows,
                                         int rope) {
  dgpp_pdl_wait();
  __shared__ float red[kLatentAppendThreads];
  const int64_t i = blockIdx.x;
  const int64_t phys = latent_physical_slot(req_ids, pos, block_tables,
                                            blocks_per_request, block_tokens, i);
  if (phys < 0) return;
  uint8_t* row = latent_cache + phys * int64_t(row_bytes);
  latent_append_tail(rope_rows, rope, i, row + size_t(kv_lora));
  const int e0 = threadIdx.x * 8;
  const bool active = e0 < kv_lora;
  float v[8];
  float amax = 0.0f;
  if (active) {
    const uint4 raw = *reinterpret_cast<const uint4*>(latent_rows + i * kv_lora + e0);
    const uint32_t* w = reinterpret_cast<const uint32_t*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[2 * j] = bf16_bits_to_float(static_cast<uint16_t>(w[j] & 0xFFFFu));
      v[2 * j + 1] = bf16_bits_to_float(static_cast<uint16_t>(w[j] >> 16));
      amax = fmaxf(amax, fmaxf(fabsf(v[2 * j]), fabsf(v[2 * j + 1])));
    }
  }
  const LatentFp8Scale s = latent_fp8_row_scale(latent_block_absmax(amax, red));
  if (threadIdx.x == 0) latent_scale[phys] = s.scale;
  if (!active) return;
  uint2 packed;
  packed.x = 0;
  packed.y = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j)
    packed.x |= static_cast<uint32_t>(latent_fp8_encode(v[j], s.inv)) << (8 * j);
#pragma unroll
  for (int j = 0; j < 4; ++j)
    packed.y |= static_cast<uint32_t>(latent_fp8_encode(v[4 + j], s.inv)) << (8 * j);
  *reinterpret_cast<uint2*>(row + e0) = packed;
}

// fp4: thread b quantizes block b (16 elements) with its own e4m3 scale in
// units of the row scale (kv_lora <= 2048, a multiple of 16).
__global__ void latent_append_fp4_kernel(const uint16_t* latent_rows,
                                         const int32_t* req_ids,
                                         const int64_t* pos,
                                         const int32_t* block_tables,
                                         int blocks_per_request,
                                         int block_tokens, uint8_t* latent_cache,
                                         float* latent_scale, int kv_lora,
                                         size_t cache_row_bytes,
                                         const uint16_t* rope_rows, int rope) {
  dgpp_pdl_wait();
  __shared__ float red[kLatentAppendThreads];
  const int64_t i = blockIdx.x;
  const int64_t phys = latent_physical_slot(req_ids, pos, block_tables,
                                            blocks_per_request, block_tokens, i);
  if (phys < 0) return;
  const int b = threadIdx.x;
  const int e0 = b * kLatentFp4Block;
  const bool active = e0 < kv_lora;
  float v[kLatentFp4Block];
  float bmax = 0.0f;
  if (active) {
    const uint4* src = reinterpret_cast<const uint4*>(latent_rows + i * kv_lora + e0);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint4 raw = src[h];
      const uint32_t* w = reinterpret_cast<const uint32_t*>(&raw);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        v[8 * h + 2 * j] = bf16_bits_to_float(static_cast<uint16_t>(w[j] & 0xFFFFu));
        v[8 * h + 2 * j + 1] = bf16_bits_to_float(static_cast<uint16_t>(w[j] >> 16));
        bmax = fmaxf(bmax, fmaxf(fabsf(v[8 * h + 2 * j]), fabsf(v[8 * h + 2 * j + 1])));
      }
    }
  }
  const LatentFp4RowScale s = latent_fp4_row_scale(latent_block_absmax(bmax, red));
  if (threadIdx.x == 0) latent_scale[phys] = s.scale;
  const size_t row_bytes = latent_row_bytes(LatentFormat::kFp4, kv_lora);  // the payload
  uint8_t* row = latent_cache + phys * static_cast<int64_t>(cache_row_bytes);
  latent_append_tail(rope_rows, rope, i, row + row_bytes);
  const size_t scale_off = latent_fp4_scale_offset(kv_lora);
  // The row's alignment padding: written once so no byte of a row the block
  // copies is ever uninitialized (the initcheck discipline).
  if (threadIdx.x == 0)
    for (size_t k = scale_off + static_cast<size_t>(kv_lora / kLatentFp4Block); k < row_bytes; ++k)
      row[k] = 0;
  if (!active) return;
  const uint8_t sc = latent_fp4_block_scale_code(bmax, s.inv);
  row[scale_off + static_cast<size_t>(b)] = sc;
  const float inv = latent_fp4_block_inv(latent_fp4_block_scale(sc, s.scale));
  uint32_t w0 = 0, w1 = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) w0 |= static_cast<uint32_t>(latent_fp4_encode(v[j], inv)) << (4 * j);
#pragma unroll
  for (int j = 0; j < 8; ++j) w1 |= static_cast<uint32_t>(latent_fp4_encode(v[8 + j], inv)) << (4 * j);
  *reinterpret_cast<uint32_t*>(row + (e0 >> 1)) = w0;
  *reinterpret_cast<uint32_t*>(row + (e0 >> 1) + 4) = w1;
}

// fp8_block: thread t quantizes elements [8t, 8t+8) with the e8m0 scale of
// its 32-element group (four consecutive lanes; kv_lora <= 1024, a
// multiple of 32) — the reference's act_quant(block 32, ue8m0) bitwise.
// One row's fp8-block append (the kernel's body, 2026-10-20: the decode
// tail kernel below runs it behind the row's rope and slot work).
__device__ __forceinline__ void latent_append_fp8block_row(const uint16_t* latent_rows, const int32_t* req_ids,
                                                           const int64_t* pos, const int32_t* block_tables,
                                                           int blocks_per_request, int block_tokens,
                                                           uint8_t* latent_cache, int kv_lora, size_t cache_row_bytes,
                                                           const uint16_t* rope_rows, int rope, int64_t i) {
  const int64_t phys = latent_physical_slot(req_ids, pos, block_tables,
                                            blocks_per_request, block_tokens, i);
  if (phys < 0) return;
  const size_t row_bytes = latent_row_bytes(LatentFormat::kFp8Block, kv_lora);  // the payload
  uint8_t* row = latent_cache + phys * static_cast<int64_t>(cache_row_bytes);
  latent_append_tail(rope_rows, rope, i, row + row_bytes);
  const size_t scale_off = latent_fp8_block_scale_offset(kv_lora);
  if (threadIdx.x == 0)
    for (size_t k = scale_off + static_cast<size_t>(kv_lora / kLatentFp8BlockGroup); k < row_bytes; ++k)
      row[k] = 0;
  const int e0 = threadIdx.x * 8;
  const bool active = e0 < kv_lora;
  float v[8];
  float amax = 0.0f;
  if (active) {
    const uint4 raw = *reinterpret_cast<const uint4*>(latent_rows + i * kv_lora + e0);
    const uint32_t* w = reinterpret_cast<const uint32_t*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[2 * j] = bf16_bits_to_float(static_cast<uint16_t>(w[j] & 0xFFFFu));
      v[2 * j + 1] = bf16_bits_to_float(static_cast<uint16_t>(w[j] >> 16));
      amax = fmaxf(amax, fmaxf(fabsf(v[2 * j]), fabsf(v[2 * j + 1])));
    }
  }
  // The group's absmax across its four lanes (exact, order-free).
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  if (!active) return;
  const uint8_t sb = latent_fp8_block_scale_byte(amax);
  if ((threadIdx.x & 3) == 0) row[scale_off + static_cast<size_t>(e0 / kLatentFp8BlockGroup)] = sb;
  const float s = e8m0_byte_to_float(sb);
  uint2 packed;
  packed.x = 0;
  packed.y = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j)
    packed.x |= static_cast<uint32_t>(latent_fp8_block_encode(v[j], s)) << (8 * j);
#pragma unroll
  for (int j = 0; j < 4; ++j)
    packed.y |= static_cast<uint32_t>(latent_fp8_block_encode(v[4 + j], s)) << (8 * j);
  *reinterpret_cast<uint2*>(row + e0) = packed;
}

__global__ void latent_append_fp8block_kernel(const uint16_t* latent_rows,
                                              const int32_t* req_ids,
                                              const int64_t* pos,
                                              const int32_t* block_tables,
                                              int blocks_per_request,
                                              int block_tokens, uint8_t* latent_cache,
                                              int kv_lora, size_t cache_row_bytes,
                                              const uint16_t* rope_rows, int rope) {
  dgpp_pdl_wait();
  latent_append_fp8block_row(latent_rows, req_ids, pos, block_tables, blocks_per_request, block_tokens, latent_cache,
                             kv_lora, cache_row_bytes, rope_rows, rope, blockIdx.x);
}

// The decode walk's tail behind the q projection, one block per row
// (2026-10-20): the q rope (rope_apply_kernel's per-pair math), the row's
// ring slot (ring_slot_positions_kernel), its window slot list
// (window_slots_decode_kernel), its entry position when the layer has a
// ratio (entry_positions_kernel), then the latent row's fp8-block append
// into the ring at that slot — five launches of 1-4 us whose bodies are
// these, element for element in the same order: bitwise each kernel's.
__global__ void __launch_bounds__(256) decode_tail_fp8block_kernel(
    uint16_t* q, int64_t q_row_stride, int64_t q_head_stride, int q_heads, int q_half, const int64_t* pos,
    const float* inv_freq, int64_t* slots, int ring_slots, int window, int32_t* wlist, int32_t* wcounts,
    int64_t* entry_out, int ratio, const uint16_t* kv_rows, const int32_t* req_ids, const int32_t* ring_table,
    uint8_t* ring, int kv_lora, size_t cache_row_bytes) {
  dgpp_pdl_wait();
  const int64_t r = blockIdx.x;
  const int64_t p = pos[r];
  if (p >= 0) {
    for (int idx = threadIdx.x; idx < q_heads * q_half; idx += blockDim.x) {
      const int i = idx % q_half;
      const int h = idx / q_half;
      uint16_t* v = q + r * q_row_stride + int64_t(h) * q_head_stride + 2 * i;
      const float ang = __fmul_rn(float(p), inv_freq[i]);
      const float c = cosf(ang);
      const float sn = sinf(ang);
      const float x0 = bf16_bits_to_float(v[0]), x1 = bf16_bits_to_float(v[1]);
      v[0] = float_to_bf16_bits(x0 * c - x1 * sn);
      v[1] = float_to_bf16_bits(x1 * c + x0 * sn);
    }
  }
  if (threadIdx.x == 0) {
    slots[r] = p < 0 ? -1 : p % ring_slots;
    if (entry_out != nullptr) entry_out[r] = p < 0 ? -1 : (p + 1) / ratio - 1;
  }
  {
    const int n = p < 0 ? 0 : int(min(int64_t(window), p + 1));
    for (int k = threadIdx.x; k < window; k += blockDim.x) {
      const int64_t qp = p - (n - 1) + k;  // ascending positions
      wlist[r * window + k] = k < n ? int32_t(qp % ring_slots) : -1;
    }
    if (threadIdx.x == 0) wcounts[r] = n;
  }
  __syncthreads();  // the slot is written for the append's lookup
  latent_append_fp8block_row(kv_rows, req_ids, slots, ring_table, 1, ring_slots, ring, kv_lora, cache_row_bytes,
                             nullptr, 0, r);
}

// fp4_block: thread b quantizes block b (16 elements) with its own
// absolute e4m3 scale (kv_lora <= 2048, a multiple of 16) — the
// reference's fp4_act_quant(block 16, e4m3 scales) bitwise.
__global__ void latent_append_fp4block_kernel(const uint16_t* latent_rows,
                                              const int32_t* req_ids,
                                              const int64_t* pos,
                                              const int32_t* block_tables,
                                              int blocks_per_request,
                                              int block_tokens, uint8_t* latent_cache,
                                              int kv_lora, size_t cache_row_bytes,
                                              const uint16_t* rope_rows, int rope) {
  dgpp_pdl_wait();
  const int64_t i = blockIdx.x;
  const int64_t phys = latent_physical_slot(req_ids, pos, block_tables,
                                            blocks_per_request, block_tokens, i);
  if (phys < 0) return;
  const int b = threadIdx.x;
  const int e0 = b * kLatentFp4Block;
  const bool active = e0 < kv_lora;
  const size_t row_bytes = latent_row_bytes(LatentFormat::kFp4Block, kv_lora);  // the payload
  uint8_t* row = latent_cache + phys * static_cast<int64_t>(cache_row_bytes);
  latent_append_tail(rope_rows, rope, i, row + row_bytes);
  const size_t scale_off = latent_fp4_scale_offset(kv_lora);
  if (threadIdx.x == 0)
    for (size_t k = scale_off + static_cast<size_t>(kv_lora / kLatentFp4Block); k < row_bytes; ++k)
      row[k] = 0;
  if (!active) return;
  float v[kLatentFp4Block];
  float bmax = 0.0f;
  const uint4* src = reinterpret_cast<const uint4*>(latent_rows + i * kv_lora + e0);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint4 raw = src[h];
    const uint32_t* w = reinterpret_cast<const uint32_t*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[8 * h + 2 * j] = bf16_bits_to_float(static_cast<uint16_t>(w[j] & 0xFFFFu));
      v[8 * h + 2 * j + 1] = bf16_bits_to_float(static_cast<uint16_t>(w[j] >> 16));
      bmax = fmaxf(bmax, fmaxf(fabsf(v[8 * h + 2 * j]), fabsf(v[8 * h + 2 * j + 1])));
    }
  }
  const uint8_t sc = latent_fp4_block_scale_code_abs(bmax);
  row[scale_off + static_cast<size_t>(b)] = sc;
  const float S = latent_fp4_block_scale_abs(sc);
  uint32_t w0 = 0, w1 = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) w0 |= static_cast<uint32_t>(latent_fp4_block_encode_abs(v[j], S)) << (4 * j);
#pragma unroll
  for (int j = 0; j < 8; ++j) w1 |= static_cast<uint32_t>(latent_fp4_block_encode_abs(v[8 + j], S)) << (4 * j);
  *reinterpret_cast<uint32_t*>(row + (e0 >> 1)) = w0;
  *reinterpret_cast<uint32_t*>(row + (e0 >> 1) + 4) = w1;
}

__global__ void gather_index_pools_kernel(const int32_t* block_table,
                                          int pools_per_block,
                                          const uint8_t* index_k,
                                          const float* index_scale,
                                          int64_t n_pools, uint8_t* out_k,
                                          float* out_scale, int dim) {
  dgpp_pdl_wait();
  const int64_t j = blockIdx.x;
  if (j >= n_pools) return;
  const int32_t blk = block_table[j / pools_per_block];
  const int64_t slot =
      int64_t(blk) * pools_per_block + (j % pools_per_block);
  const uint4* s = reinterpret_cast<const uint4*>(index_k + slot * dim);
  uint4* d = reinterpret_cast<uint4*>(out_k + j * dim);
  for (int i = threadIdx.x; i < dim / 16; i += blockDim.x) d[i] = s[i];
  if (threadIdx.x == 0) out_scale[j] = index_scale[slot];
}

// ---------------------------------------------------------------------
// Selection kernels
// ---------------------------------------------------------------------

// Warp-cooperative: lane h computes head h's contribution (heads == 32).
// q stays FP8 in shared memory (4 KB/row instead of 16 KB as fp32) — the
// dot loop converts on access and is memory-bound regardless; the saving
// is what lets an 8-row MTP decode batch fit the 99 KB GB10 smem optin.
// kHG: the index heads in groups of 32 (1: 32 heads, every family before
// DeepSeek-V4-Flash; 2: its 64 — a lane computes one head of each group,
// the groups' warp sums added in group order, 2026-10-01).
template <bool kRelu, int kHG = 1>
struct DecodeKeyFn {  // warp computes one pool (lane per head)
  const uint8_t* q8;   // smem [kHG][16 chunks][32 heads] x 8 fp8 bytes (this row)
  const float* w;      // smem [heads] (this row)
  const uint8_t* index_k;
  const float* index_scale;
  const int32_t* block_table;  // this row's request
  int pools_per_block;
  int dim;

  __device__ int64_t slot_of(int64_t pool) const {
    const int32_t blk = block_table[pool / pools_per_block];
    return int64_t(blk) * pools_per_block + (pool % pools_per_block);
  }
  // The pool's 128-byte index row as one coalesced warp load (4 bytes per
  // lane) — the row goes to a per-warp smem line and every lane reads it
  // back as the same 16 uint2 it read from global before (2026-09-06: 16
  // serial same-address global loads per pool were the scoring's latency).
  __device__ uint32_t load_row_word(int64_t slot) const {
    const int lane = threadIdx.x & 31;
    return reinterpret_cast<const uint32_t*>(index_k + slot * dim)[lane];
  }
  // The key of `pool` whose row sits in `krow_s` (32 words, smem) with
  // scale `ks`. (TRIED 2026-09-06 and reverted: q and k decoded to floats
  // in smem once per row / once per pool — 40 % fewer instructions per
  // pool, but the float4 q reads quadrupled the smem bytes per pool and
  // the scoring got slower at every context; the fp8 reads below are the
  // measured optimum.)
  __device__ uint64_t key_from_row(int64_t pool, const uint32_t* krow_s,
                                   float ks) const {
    const int lane = threadIdx.x & 31;
    // Contraction-proof arithmetic (__fmul_rn/__fadd_rn, no FMA fusion):
    // the host oracle mirrors this exact sequence so pool-logit parity —
    // and therefore top-k position parity — is bitwise, not statistical.
    // The element order matches the scalar form exactly (d = 0..127); only
    // the fp8 decode is vectorized (native hardware, bit-exact — see the
    // conversion helpers above).
    // q8 is chunk-major in smem ([16 chunks][32 heads] uint2 per head
    // group): for a chunk the lanes read consecutive words — conflict-free
    // (head-major put every lane on one bank, a 32-way conflict on each of
    // the 16 loads).
    const uint2* k2 = reinterpret_cast<const uint2*>(krow_s);
    float total = 0.0f;
#pragma unroll
    for (int g = 0; g < kHG; ++g) {
      float partial = 0.0f;
      const uint2* q2 = reinterpret_cast<const uint2*>(q8) + g * 512;
#pragma unroll 8
      for (int p = 0; p < 16; ++p) {
        const uint2 qv = q2[p * 32 + lane];  // 8 fp8 values of this lane's head
        const uint2 kv = k2[p];              // 8 fp8 values of the pool row
        const uint16_t* qq = reinterpret_cast<const uint16_t*>(&qv);
        const uint16_t* kk = reinterpret_cast<const uint16_t*>(&kv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 a = fp8x2_to_float2(qq[j]);
          const float2 b = fp8x2_to_float2(kk[j]);
          partial = __fadd_rn(partial, __fmul_rn(a.x, b.x));
          partial = __fadd_rn(partial, __fmul_rn(a.y, b.y));
        }
      }
      // The full model's indexer clamps each head's score at zero before its
      // weight (relu(q_h . k) — the scales are positive powers of two, so
      // the clamp on the fp8 dot is the clamp on the score).
      if (kRelu) partial = fmaxf(partial, 0.0f);
      const float contrib = __fmul_rn(__fmul_rn(w[g * 32 + lane], ks), partial);
      const float gsum = warp_sum(contrib);
      total = g == 0 ? gsum : __fadd_rn(total, gsum);
    }
    // ~sortable reverses the ascending float order: the smallest composite
    // key is then the HIGHEST logit (ties -> lower pool index from the idx
    // bits). Without the inversion the selection picks the worst pools.
    return (uint64_t(~sortable_f32_dev(total)) << kIdxBits) | uint64_t(pool);
  }
};

template <bool kRelu>
struct PrefillKeyFn {
  static constexpr bool kWarpCooperative = true;  // warp computes one pool
  const float* dot;       // [rows * heads, dot_stride]
  const float* w_folded;  // [rows, heads]
  const float* k_scale;   // [n_pools]
  int64_t dot_stride;
  int heads;
  int row;

  __device__ uint64_t operator()(int64_t pool) const {
    const int lane = threadIdx.x & 31;
    float dv = dot[int64_t(row * heads + lane) * dot_stride + pool];
    if (kRelu) dv = fmaxf(dv, 0.0f);
    const float contrib = (w_folded[row * heads + lane] * k_scale[pool]) * dv;
    const float total = warp_sum(contrib);
    // Inverted sortable: smallest composite key == highest logit (see
    // DecodeKeyFn).
    return (uint64_t(~sortable_f32_dev(total)) << kIdxBits) | uint64_t(pool);
  }
};

// Decode select (rewritten 2026-09-06): every block scores its stripe of
// each row's visible pools and publishes the composite keys to a global
// row (keys_ws) together with a coarse histogram of the keys' top radix
// digit; the last block to finish (a ticket on counter_ws) finds the
// select_k-th smallest key by radix refinement over that histogram — one
// pass over the row's keys per further digit, stopping when the boundary
// digit's bin holds at most kSelectStopCandidates keys — gathers the keys
// below the boundary straight into best[] (the expansion sorts ids, so
// their order is free), sorts only the boundary bin's keys, and expands.
// The selection is a set under a total order (the key), so it is the
// streaming top-k's set bit for bit. What it replaces: a 2048-key bitonic
// sort per block per tile regardless of the stripe, and a serial merge of
// grid x select_k partials in the last block — 570-830 us at 4K-32K tokens
// on the 16-block grid, the decode step's only context-scaled kernel.
//
// The select counter's reset is a KERNEL, not cudaMemsetAsync. A memset
// node in the captured decode graph executes on the copy-engine queue, an
// in-order queue shared by every stream in the process; a queued node's
// dependency wait blocks everything behind it. In a one-process multi-rank
// world (the loopback gates) a peer rank's queued post-collective memset
// held this reset behind it while that peer's collective spun waiting on
// ours — the batched-MTP graph stall (docs/batched_mtp_graph_stall.md).
// Kernel nodes never share that queue; the graph engine rejects any
// non-kernel node at capture. The histogram rows reset the same way: the
// last block zeroes them after consuming them.
__global__ void select_counter_reset_kernel(int32_t* counter, int32_t* hist_rows,
                                            int hist_words) {
  dgpp_pdl_wait();
  if (threadIdx.x < 2) counter[threadIdx.x] = 0;
  // The rows' histograms start at zero every call: the
  // workspace is not zeroed at allocation, and a row's histogram used to
  // sit at a call-dependent offset (see dsa_select_decode).
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hist_words;
       i += gridDim.x * blockDim.x)
    hist_rows[i] = 0;
}

constexpr int kSelectRadixBits = 10;  // 1024 bins: 4 KB of smem, two blocks per SM
constexpr int kSelectHistBins = 1 << kSelectRadixBits;
// The key's live bits: the 32-bit inverted sortable logit above kIdxBits
// of pool index (DecodeKeyFn); the first radix digit sits just below.
constexpr int kSelectKeyBits = 32 + kIdxBits;
constexpr int kSelectTopShift = kSelectKeyBits - kSelectRadixBits;
// The boundary bin is sorted once it is this small (a 256-key bitonic sort
// is ~2 us; a further refinement pass costs a read of the row's keys).
constexpr int kSelectStopCandidates = 256;

// Phase stamps of the last block (globaltimer ns; thread 0; six stores per
// call): [0] entry, [1] phase 1 done, [2] phase 2 start, then accumulated
// over rows: [3] selection (histogram + refinement + gather + rank), [4]
// expansion. dsa_select_debug_phases reads them — the bench's breakdown.
__device__ unsigned long long g_select_phase[8];
__device__ __forceinline__ unsigned long long select_now() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ int select_digit(uint64_t key, int shift, int bits) {
  return int((key >> shift) & ((1u << bits) - 1u));
}

// The bin of the block-shared histogram (kSelectHistBins, smem) holding
// the `remaining`-th smallest key, and the count below it: a chunk sum per
// thread, a block scan, one thread walks its chunk. Uniform on return.
__device__ inline void select_find_bin(const int32_t* hist, int remaining,
                                       int* s_bin, int* s_below) {
  __shared__ int warp_tot[8];
  const int per = kSelectHistBins / int(blockDim.x);
  const int t0 = int(threadIdx.x) * per;
  int sum = 0;
  for (int i = 0; i < per; ++i) sum += hist[t0 + i];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int incl = sum;
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += v;
  }
  if (lane == 31) warp_tot[warp] = incl;
  if (threadIdx.x == 0) *s_bin = -1;
  __syncthreads();
  int wbase = 0;
  for (int wi = 0; wi < warp; ++wi) wbase += warp_tot[wi];
  const int excl = wbase + incl - sum;
  if (excl < remaining && remaining <= excl + sum) {
    int cum = excl;
    for (int i = 0; i < per; ++i) {
      const int h = hist[t0 + i];
      if (cum < remaining && remaining <= cum + h) {
        *s_bin = t0 + i;
        *s_below = cum;
        break;
      }
      cum += h;
    }
  }
  __syncthreads();
}

// The decode select's anomaly record (2026-09-06): a phase-2 fill whose
// scan disagreed with the histogram (defs != lower, or the boundary bin
// short of `remaining`), the first one's accounting and a count;
// dsa_select_anomalies() reads and clears it.
__device__ unsigned long long g_select_anomaly_count = 0;
__device__ long long g_select_anomaly[6] = {0, 0, 0, 0, 0, 0};
__device__ __forceinline__ void select_anomaly_record(int64_t visible, int lower,
                                                      int n_def, int remaining,
                                                      int cnt, int n_cand) {
  if (atomicAdd(&g_select_anomaly_count, 1ull) == 0ull) {
    g_select_anomaly[0] = visible;
    g_select_anomaly[1] = lower;
    g_select_anomaly[2] = n_def;
    g_select_anomaly[3] = remaining;
    g_select_anomaly[4] = cnt;
    g_select_anomaly[5] = n_cand;
    __threadfence();
  }
}

// ---- the tensor-core scoring of the 64-head indexer (2026-10-01) ------------
// DeepSeek-V4-Flash scores every compressed entry with 64 heads x 128 dims:
// the lane-per-head scalar chain above is 256 dependent FMAs per entry per
// row (70 us a launch at 512 entries and four rows; linear in the context).
// The 64-head instantiation scores a 16-entry tile per warp on the tensor
// cores instead: keys and queries are e4m3 codes, every one of which is an
// fp16 value, so both widen exactly (cvt e4m3x2 -> f16x2) and the dots run
// as mma.m16n8k16 f16 -> f32 — A the tile's 16 keys, B eight heads, eight k
// steps per head tile, eight head tiles. The head sum is a fixed order
// (within a lane the two heads in index order, the head tiles in order,
// then the quad's lanes pairwise), so every rank and every replay derives
// the same keys; it is not the scalar chain's order — the two-pass prefill
// select and this one already differ at rounding level.
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint16_t two) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two), __NV_E4M3);
  return static_cast<uint32_t>(h.x) | (static_cast<uint32_t>(h.y) << 16);
}
__device__ __forceinline__ void mma_f16_16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                              uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
constexpr int kMmaSelTile = 16;                                     // entries per warp tile
constexpr int kMmaSelWarps = 8;                                     // the kernel's 256 threads
constexpr size_t kMmaSelTileBytes = size_t(kMmaSelTile) * 128 * 2;  // a tile's keys as f16
// The scoring's shared memory: the current row's q as f16 [heads][128], the
// warps' key tiles and their scales.
constexpr size_t select_mma_q_bytes(int heads) { return size_t(heads) * 128 * 2; }
constexpr size_t kMmaSelKeyBytes = kMmaSelWarps * (kMmaSelTileBytes + kMmaSelTile * sizeof(float));

template <bool kRelu, int kHG = 1>
__global__ void select_decode_kernel(
    const uint8_t* q_fp8, const float* w_folded, const int32_t* req_ids,
    const int64_t* pos, int rows, const int32_t* block_tables,
    int blocks_per_request, const uint8_t* index_k, const float* index_scale,
    int pools_per_block, int heads, int select_k, int kpool, int max_selected,
    int32_t* topk_out, int32_t* out_counts, uint64_t* keys_ws,
    int64_t keys_stride, int32_t* hist_ws, int32_t* counter_ws) {
  dgpp_pdl_wait();
  extern __shared__ uint64_t smem_u64[];
  // Layout: [q8: rows x [16 chunks][32 heads] x 8 bytes][k rows: warps x 4 x
  //         32 u32][w: rows*heads f32][hist: kSelectHistBins i32]
  //         [best_hi/best_lo: select_k each]
  //         [cand_hi/cand_lo: kSelectStopCandidates each]
  //         [scratch: select_k + 1 i32].
  constexpr int kPoolsPerIter = 4;
  // kHG == 2 scores on the tensor cores: its q area holds ONE row as f16 and
  // its key area the warps' 16-entry tiles (see e4m3x2_to_f16x2 above).
  constexpr bool kMma = kHG == 2;
  uint8_t* q8 = reinterpret_cast<uint8_t*>(smem_u64);
  uint32_t* krow_s = reinterpret_cast<uint32_t*>(
      q8 + (kMma ? int64_t(select_mma_q_bytes(heads)) : int64_t(rows) * heads * 128));
  float* w = reinterpret_cast<float*>(
      reinterpret_cast<uint8_t*>(krow_s) +
      (kMma ? kMmaSelKeyBytes : size_t(256 / 32) * kPoolsPerIter * 32 * sizeof(uint32_t)));
  int32_t* hist = reinterpret_cast<int32_t*>(w + int64_t(rows) * heads);
  uint32_t* best_hi = reinterpret_cast<uint32_t*>(hist + kSelectHistBins);
  uint32_t* best_lo = best_hi + select_k;
  uint32_t* cand_hi = best_lo + select_k;
  uint32_t* cand_lo = cand_hi + kSelectStopCandidates;
  int32_t* scratch = reinterpret_cast<int32_t*>(cand_lo + kSelectStopCandidates);
  __shared__ int s_bin, s_below, s_n_def, s_n_cand;

  const unsigned long long t_entry = select_now();
  // q to smem chunk-major per row ([16 chunks][32 heads] of 8 bytes): the
  // dot's per-lane reads are then conflict-free (see key_from_row).
  if constexpr (!kMma) {
  for (int64_t c = threadIdx.x; c < int64_t(rows) * heads * 16; c += blockDim.x) {
    const int r = int(c / (heads * 16)), rem = int(c % (heads * 16));
    const int h = rem / 16, p = rem % 16;
    const uint2 v = *reinterpret_cast<const uint2*>(
        q_fp8 + (int64_t(r) * heads + h) * 128 + p * 8);
    reinterpret_cast<uint2*>(q8)[int64_t(r) * heads * 16 + (h >> 5) * 512 + p * 32 + (h & 31)] = v;
  }
  }
  for (int64_t i = threadIdx.x; i < int64_t(rows) * heads; i += blockDim.x)
    w[i] = w_folded[i];
  __syncthreads();

  // Phase 1: score the stripe, publish keys and the first radix digit's
  // histogram. Short contexts (visible <= select_k pools) select EVERY
  // pool and need no keys (the last block writes them directly).
  const int warp = threadIdx.x >> 5;
  const int nwarp = int(blockDim.x) >> 5;
  // The tensor-core form's grid is (stripes, rows): a block scores ONE row's
  // stripe, so the rows of a launch score side by side (a four-row verify
  // batch was four passes of every block: 35 us of a 48 us launch at 512
  // entries). The scalar form's grid is (stripes, 1): every block, every row.
  const int total_blocks = int(gridDim.x) * int(gridDim.y);
  for (int r = 0; r < rows; ++r) {
    if (kMma && r != int(blockIdx.y)) continue;
    const int64_t visible = (pos[r] + 1) / kpool;
    if (visible <= select_k) continue;
    // The tensor-core form's stripe is at least one tile per warp: a short
    // context then runs in its first blocks alone (the others have nothing
    // to stage: 48 blocks each converting the row's q for eleven entries was
    // the 512-entry launch's scoring).
    int64_t stripe = (visible + gridDim.x - 1) / gridDim.x;
    if (kMma && stripe < kMmaSelWarps * kMmaSelTile) stripe = kMmaSelWarps * kMmaSelTile;
    const int64_t lo = min(visible, int64_t(blockIdx.x) * stripe);
    const int64_t hi = min(visible, lo + stripe);
    if (kMma && lo >= hi) continue;  // uniform across the block
    for (int i = threadIdx.x; i < kSelectHistBins; i += blockDim.x) hist[i] = 0;
    __syncthreads();
    uint64_t* krow = keys_ws + int64_t(r) * keys_stride;
    const int lane = threadIdx.x & 31;
    if constexpr (kMma) {
      // The row's q as f16 [heads][128] (two codes per conversion).
      uint32_t* q16 = reinterpret_cast<uint32_t*>(q8);
      for (int c = threadIdx.x; c < heads * 64; c += blockDim.x)
        q16[c] = e4m3x2_to_f16x2(
            *reinterpret_cast<const uint16_t*>(q_fp8 + int64_t(r) * heads * 128 + int64_t(c) * 2));
      __syncthreads();
      const int32_t* bt = block_tables + int64_t(req_ids[r]) * blocks_per_request;
      const float* wr = w + int64_t(r) * heads;
      uint32_t* kt = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(krow_s) +
                                                 size_t(warp) * (kMmaSelTileBytes + kMmaSelTile * sizeof(float)));
      float* kscale = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(kt) + kMmaSelTileBytes);
      // Whole tiles per warp.
      const int64_t sub = ((hi - lo + nwarp - 1) / nwarp + kMmaSelTile - 1) / kMmaSelTile * kMmaSelTile;
      const int64_t wlo = min(hi, lo + int64_t(warp) * sub);
      const int64_t whi = min(hi, wlo + sub);
      const int mr = lane >> 2, mt = lane & 3;  // the fragment's row / column pair
      for (int64_t p = wlo; p < whi; p += kMmaSelTile) {
        // The tile's keys: lane l converts half hf of entry e (64 codes), all
        // four 16-byte loads in flight together; an entry past the stripe is
        // zero and publishes nothing.
        {
          const int e = lane >> 1, hf = lane & 1;
          const int64_t pool = p + e;
          const bool live = pool < whi;
          int64_t slot = 0;
          if (live) slot = int64_t(bt[pool / pools_per_block]) * pools_per_block + (pool % pools_per_block);
          uint4 raw[4];
#pragma unroll
          for (int v = 0; v < 4; ++v)
            raw[v] = live ? reinterpret_cast<const uint4*>(index_k + slot * 128 + hf * 64)[v]
                          : make_uint4(0u, 0u, 0u, 0u);
          if (hf == 0) kscale[e] = live ? index_scale[slot] : 0.f;
          uint32_t* dst = kt + e * 64 + hf * 32;  // 64 uint32 (128 f16) per entry
#pragma unroll
          for (int v = 0; v < 4; ++v) {
            const uint32_t wd[4] = {raw[v].x, raw[v].y, raw[v].z, raw[v].w};
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              dst[v * 8 + q * 2] = e4m3x2_to_f16x2(static_cast<uint16_t>(wd[q] & 0xFFFFu));
              dst[v * 8 + q * 2 + 1] = e4m3x2_to_f16x2(static_cast<uint16_t>(wd[q] >> 16));
            }
          }
        }
        __syncwarp();
        float s_lo = 0.f, s_hi = 0.f;  // entries p + mr and p + mr + 8: the weighted head sum so far
#pragma unroll 1
        for (int ht = 0; ht < heads / 8; ++ht) {
          float c[4] = {0.f, 0.f, 0.f, 0.f};
          const uint32_t* qh = q16 + (ht * 8 + mr) * 64;  // head ht * 8 + mr: 64 uint32 of f16 pairs
#pragma unroll
          for (int ks = 0; ks < 8; ++ks) {
            const int k0 = ks * 8 + mt;  // the pair (2t, 2t + 1) of this 16-k step
            mma_f16_16816(c, kt[mr * 64 + k0], kt[(mr + 8) * 64 + k0], kt[mr * 64 + k0 + 4],
                          kt[(mr + 8) * 64 + k0 + 4], qh[k0], qh[k0 + 4]);
          }
          if (kRelu) {
            c[0] = fmaxf(c[0], 0.f);
            c[1] = fmaxf(c[1], 0.f);
            c[2] = fmaxf(c[2], 0.f);
            c[3] = fmaxf(c[3], 0.f);
          }
          const float w0 = wr[ht * 8 + 2 * mt], w1 = wr[ht * 8 + 2 * mt + 1];
          s_lo = __fadd_rn(s_lo, __fmul_rn(w0, c[0]));
          s_lo = __fadd_rn(s_lo, __fmul_rn(w1, c[1]));
          s_hi = __fadd_rn(s_hi, __fmul_rn(w0, c[2]));
          s_hi = __fadd_rn(s_hi, __fmul_rn(w1, c[3]));
        }
        // The quad's four lanes hold the entry's eight column pairs: pairwise.
        s_lo = __fadd_rn(s_lo, __shfl_xor_sync(0xffffffffu, s_lo, 1));
        s_lo = __fadd_rn(s_lo, __shfl_xor_sync(0xffffffffu, s_lo, 2));
        s_hi = __fadd_rn(s_hi, __shfl_xor_sync(0xffffffffu, s_hi, 1));
        s_hi = __fadd_rn(s_hi, __shfl_xor_sync(0xffffffffu, s_hi, 2));
        if (mt == 0) {
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const int e = mr + half * 8;
            const int64_t pool = p + e;
            if (pool < whi) {
              const float total = __fmul_rn(half == 0 ? s_lo : s_hi, kscale[e]);
              const uint64_t key = (uint64_t(~sortable_f32_dev(total)) << kIdxBits) | uint64_t(pool);
              krow[pool] = key;
              atomicAdd(&hist[select_digit(key, kSelectTopShift, kSelectRadixBits)], 1);
            }
          }
        }
        __syncwarp();  // the tile is overwritten next iteration
      }
    } else {
    DecodeKeyFn<kRelu, kHG> fn{q8 + int64_t(r) * heads * 128, w + int64_t(r) * heads,
                          index_k, index_scale,
                          block_tables + int64_t(req_ids[r]) * blocks_per_request,
                          pools_per_block, 128};
    uint32_t* rows_s = krow_s + warp * (kPoolsPerIter * 32);
    // Each warp owns a contiguous sub-stripe (consecutive pools share a
    // block-table entry), four pools per iteration with every load — the
    // table entry, the row, the scale — in flight before the first dot
    // (a 128-deep dependent chain per pool; the latency, not the dot, was
    // the scoring's cost).
    const int64_t sub = (hi - lo + nwarp - 1) / nwarp;
    const int64_t wlo = min(hi, lo + int64_t(warp) * sub);
    const int64_t whi = min(hi, wlo + sub);
    for (int64_t p = wlo; p < whi; p += kPoolsPerIter) {
      int64_t slot[kPoolsPerIter];
      uint32_t word[kPoolsPerIter];
      float ks[kPoolsPerIter];
#pragma unroll
      for (int u = 0; u < kPoolsPerIter; ++u)
        slot[u] = p + u < whi ? fn.slot_of(p + u) : -1;
#pragma unroll
      for (int u = 0; u < kPoolsPerIter; ++u) {
        word[u] = slot[u] >= 0 ? fn.load_row_word(slot[u]) : 0u;
        ks[u] = slot[u] >= 0 ? index_scale[slot[u]] : 0.f;
      }
#pragma unroll
      for (int u = 0; u < kPoolsPerIter; ++u) rows_s[u * 32 + lane] = word[u];
      __syncwarp();
#pragma unroll
      for (int u = 0; u < kPoolsPerIter; ++u) {
        if (slot[u] < 0) break;  // uniform across the warp
        const uint64_t key = fn.key_from_row(p + u, rows_s + u * 32, ks[u]);
        if (lane == 0) {
          krow[p + u] = key;
          atomicAdd(&hist[select_digit(key, kSelectTopShift, kSelectRadixBits)], 1);
        }
      }
      __syncwarp();  // the rows are overwritten next iteration
    }
    }
    __syncthreads();
    int32_t* ghist = hist_ws + int64_t(r) * kSelectHistBins;
    for (int i = threadIdx.x; i < kSelectHistBins; i += blockDim.x)
      if (hist[i] != 0) atomicAdd(&ghist[i], hist[i]);
    __syncthreads();
  }
  const unsigned long long t_p1 = select_now();
  // Phase 2 is row-parallel: the last `rows` blocks to finish scoring each
  // take one row (ticket grid - rows + r), waiting until every block has
  // published (counter == grid); the others exit, so the waiters can never
  // starve a block that has not run yet. counter_ws[1] counts the rows
  // done; the last resets both counters for the next replay.
  __threadfence();
  if (threadIdx.x == 0) {
    const int ticket = atomicAdd(counter_ws, 1);
    s_bin = ticket - (total_blocks - rows);  // my row, or negative
  }
  __syncthreads();
  const int my_row = s_bin;
  if (my_row < 0) return;
  if (threadIdx.x == 0) {
    while (*reinterpret_cast<volatile int32_t*>(counter_ws) < int32_t(total_blocks)) {
    }
  }
  __syncthreads();
  __threadfence();
  unsigned long long t_sel = 0, t_exp = 0;
  if (threadIdx.x == 0 && my_row == 0) {
    g_select_phase[0] = t_entry;
    g_select_phase[1] = t_p1;
    g_select_phase[2] = select_now();
  }

  // Phase 2 (this block's row): its select_k smallest keys.
  for (int r = my_row; r <= my_row; ++r) {
    const int64_t visible = (pos[r] + 1) / kpool;
    const unsigned long long t_row = select_now();
    if (visible <= select_k) {
      // Every visible pool is selected: keys carry only the pool id in
      // their low kIdxBits — the expansion reads nothing else.
      for (int i = threadIdx.x; i < select_k; i += blockDim.x) {
        const bool real = i < visible;
        best_hi[i] = real ? 0u : 0xFFFFFFFFu;
        best_lo[i] = real ? uint32_t(i) : 0xFFFFFFFFu;
      }
      __syncthreads();
    } else {
      const uint64_t* krow = keys_ws + int64_t(r) * keys_stride;
      int32_t* ghist = hist_ws + int64_t(r) * kSelectHistBins;
      // The first digit's histogram, built by every block; consumed and
      // reset for the next call.
      for (int i = threadIdx.x; i < kSelectHistBins; i += blockDim.x) {
        hist[i] = ghist[i];
        ghist[i] = 0;
      }
      __syncthreads();
      // Refinement: prefix = the boundary key's digits so far (the keys
      // whose bits above `shift` equal it form the boundary bin); lower =
      // keys below that bin (all selected); remaining = how many of the
      // bin's `cnt` keys complete the select_k. Uniform across the block.
      uint64_t prefix = 0;
      int shift = kSelectTopShift, bits = kSelectRadixBits;
      int lower = 0, remaining = select_k, cnt = 0;
      for (;;) {
        select_find_bin(hist, remaining, &s_bin, &s_below);
        const int bin = s_bin < 0 ? kSelectHistBins - 1 : s_bin;
        const int below = s_bin < 0 ? 0 : s_below;
        cnt = hist[bin];
        lower += below;
        remaining -= below;
        prefix = (prefix << bits) | uint64_t(bin);
        if (cnt <= kSelectStopCandidates || shift == 0) break;
        const int pshift = shift;
        shift = shift >= kSelectRadixBits ? shift - kSelectRadixBits : 0;
        bits = pshift - shift;
        __syncthreads();
        for (int i = threadIdx.x; i < kSelectHistBins; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        for (int64_t p = threadIdx.x; p < visible; p += blockDim.x) {
          const uint64_t key = krow[p];
          if ((key >> pshift) == prefix)
            atomicAdd(&hist[select_digit(key, shift, bits)], 1);
        }
        __syncthreads();
      }
      // Gather: keys below the boundary bin straight into best[] (their
      // order is free — the expansion sorts ids); the bin's keys into the
      // candidate array, sorted so the `remaining` smallest complete best.
      // best[] starts EMPTY: an entry the fill leaves
      // unwritten is dropped by the expansion instead of carrying shared
      // memory's leftovers into the token list (a 62,600 in a 2,119-token
      // context faulted the listed attention on the fabric); a short fill
      // is recorded (dsa_select_anomalies).
      if (threadIdx.x == 0) {
        s_n_def = 0;
        s_n_cand = 0;
      }
      for (int i = threadIdx.x; i < select_k; i += blockDim.x) {
        best_hi[i] = 0xFFFFFFFFu;
        best_lo[i] = 0xFFFFFFFFu;
      }
      for (int i = threadIdx.x; i < kSelectStopCandidates; i += blockDim.x) {
        cand_hi[i] = 0xFFFFFFFFu;
        cand_lo[i] = 0xFFFFFFFFu;
      }
      __syncthreads();
      for (int64_t p = threadIdx.x; p < visible; p += blockDim.x) {
        const uint64_t key = krow[p];
        const uint64_t top = key >> shift;
        if (top < prefix) {
          const int s = atomicAdd(&s_n_def, 1);
          if (s < select_k) {
            best_hi[s] = uint32_t(key >> 32);
            best_lo[s] = uint32_t(key);
          }
        } else if (top == prefix) {
          const int s = atomicAdd(&s_n_cand, 1);
          if (s < kSelectStopCandidates) {
            cand_hi[s] = uint32_t(key >> 32);
            cand_lo[s] = uint32_t(key);
          }
        }
      }
      __syncthreads();
      // The bin's `remaining` smallest keys by rank (unique keys: the
      // rank is a permutation, so the keys ranked below `remaining` land
      // at best[lower + rank]) — cnt^2 / blockDim compares per thread, no
      // sorting network.
      {
        const int n = cnt < kSelectStopCandidates ? cnt : kSelectStopCandidates;
        const int i = threadIdx.x;
        const uint32_t vh = i < n ? cand_hi[i] : 0xFFFFFFFFu;
        const uint32_t vl = i < n ? cand_lo[i] : 0xFFFFFFFFu;
        int rank = 0;
#pragma unroll 8
        for (int j = 0; j < n; ++j) rank += key_less(cand_hi[j], cand_lo[j], vh, vl);
        if (i < n && rank < remaining) {
          best_hi[lower + rank] = vh;
          best_lo[lower + rank] = vl;
        }
      }
      __syncthreads();
      if (threadIdx.x == 0 &&
          (s_n_def != lower || s_n_cand < remaining || cnt != s_n_cand))
        select_anomaly_record(visible, lower, s_n_def, remaining, cnt, s_n_cand);
    }
    const unsigned long long t_mid = select_now();
    int* smem_count = reinterpret_cast<int*>(scratch + select_k);
    const int out_cnt = expand_from_best(
        best_hi, best_lo, select_k, pos[r], kpool, max_selected,
        topk_out + int64_t(r) * max_selected, scratch, smem_count);
    if (threadIdx.x == 0) out_counts[r] = out_cnt;
    __syncthreads();
    t_sel += t_mid - t_row;
    t_exp += select_now() - t_mid;
  }
  if (threadIdx.x == 0) {
    if (my_row == 0) {
      g_select_phase[3] = t_sel;
      g_select_phase[4] = t_exp;
      g_select_phase[5] = select_now();
    }
    __threadfence();
    const int done = atomicAdd(counter_ws + 1, 1);
    if (done == rows - 1) {  // the last row finished: reset for the replay
      counter_ws[1] = 0;
      __threadfence();
      counter_ws[0] = 0;
    }
  }
}

// The old prefill selector gave a warp one pool, so adjacent lanes loaded
// different head rows. At long contexts only a few query rows fit the dot
// buffer: three blocks scanned 131K pools and repeatedly sorted tiles.
// Give each thread a pool instead. Head loads are coalesced across pools;
// the register tree preserves PrefillKeyFn's exact arithmetic and order.
template <bool kRelu>
__global__ void prefill_score_keys_kernel(
    const float* dot, int64_t stride, const float* w, const float* scales,
    const int64_t* pos, int64_t n_pools, int select_k, int kpool,
    uint64_t* keys) {
  dgpp_pdl_wait();
  const int r = blockIdx.y;
  const int64_t visible = min((pos[r] + 1) / kpool, n_pools);
  const int64_t p = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (visible <= select_k || p >= visible) return;
  float v[32];
  const float ks = scales[p];
#pragma unroll
  for (int h = 0; h < 32; ++h) {
    float d = dot[(int64_t(r) * 32 + h) * stride + p];
    if (kRelu) d = fmaxf(d, 0.0f);
    v[h] = __fmul_rn(__fmul_rn(w[r * 32 + h], ks), d);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
    for (int h = 0; h < off; ++h) v[h] = __fadd_rn(v[h], v[h + off]);
  }
  keys[int64_t(r) * stride + p] =
      (uint64_t(~sortable_f32_dev(v[0])) << kIdxBits) | uint64_t(p);
}

__device__ inline void prefill_hist_add(int32_t* hist, int bin) {
  const unsigned peers = __match_any_sync(__activemask(), bin);
  if ((threadIdx.x & 31) == __ffs(peers) - 1) atomicAdd(hist + bin, __popc(peers));
}

// One query's 32 indexer heads against 128 pools. Keep the per-head
// tensor-core dots in registers and write only the reduced keys.
// The old GEMM + score chain writes and rereads 32 floats per query/pool.
template <bool kRelu>
__global__ void prefill_fused_score_keys_kernel(
    const uint8_t* q, const uint8_t* k, const float* w, const float* scales,
    const int64_t* pos, int rows, int64_t pools, int64_t stride,
    int select_k, int kpool, uint64_t* keys) {
  dgpp_pdl_wait();
  constexpr int S = 144, N = 128;
  __shared__ __align__(16) uint8_t a[32 * S];
  __shared__ __align__(16) uint8_t b[N * S];
  const int r = blockIdx.x % rows;
  const int64_t p0 = int64_t(blockIdx.x / rows) * N;
  const int64_t visible = min((pos[r] + 1) / kpool, pools);
  if (visible <= select_k || p0 >= visible) return;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  for (int i = tid; i < 32 * 8; i += 128) {
    const int row = i / 8, col = (i % 8) * 16;
    *reinterpret_cast<uint4*>(a + row * S + col) =
        *reinterpret_cast<const uint4*>(q + (int64_t(r) * 32 + row) * 128 + col);
  }
  for (int i = tid; i < N * 8; i += 128) {
    const int row = i / 8, col = (i % 8) * 16;
    const int64_t p = p0 + row;
    *reinterpret_cast<uint4*>(b + row * S + col) = p < visible
        ? *reinterpret_cast<const uint4*>(k + p * 128 + col)
        : make_uint4(0, 0, 0, 0);
  }
  __syncthreads();
  float acc[2][4][4] = {};
#pragma unroll
  for (int kk = 0; kk < 128; kk += 32) {
    uint32_t af[2][4], bf[4][2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(
          a + (i * 16 + lane % 16) * S + kk + (lane / 16) * 16));
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
          : "=r"(af[i][0]), "=r"(af[i][1]), "=r"(af[i][2]), "=r"(af[i][3]) : "r"(addr));
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int row = warp * 32 + j * 16 + (lane / 16) * 8 + lane % 8;
      const int col = kk + ((lane / 8) % 2) * 16;
      const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(b + row * S + col));
      uint32_t v[4];
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
          : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "r"(addr));
      bf[j * 2][0] = v[0]; bf[j * 2][1] = v[1];
      bf[j * 2 + 1][0] = v[2]; bf[j * 2 + 1][1] = v[3];
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
          : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]),
            "r"(bf[j][0]), "r"(bf[j][1]));
  }
  // The MMA fragments hold heads h, h+8, h+16, h+24 in each lane.
  // Fold offsets 16 and 8 in registers, then offsets 4, 2, 1 between
  // four-lane groups. This is the original head-summation order without
  // a shared-memory dot matrix or a block-wide reduction barrier.
  float weight[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int h = 0; h < 2; ++h)
      weight[i][h] = w[r * 32 + i * 16 + h * 8 + lane / 4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int v = 0; v < 2; ++v) {
      const int64_t p = p0 + warp * 32 + j * 8 + (lane % 4) * 2 + v;
      const float ks = p < visible ? scales[p] : 0.0f;
      float c[2][2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          float d = acc[i][j][h * 2 + v];
          if (kRelu) d = fmaxf(d, 0.0f);
          c[i][h] = __fmul_rn(__fmul_rn(weight[i][h], ks), d);
        }
      float sum = __fadd_rn(__fadd_rn(c[0][0], c[1][0]),
                            __fadd_rn(c[0][1], c[1][1]));
#pragma unroll
      for (int off = 16; off >= 4; off >>= 1)
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, off));
      if (lane < 4 && p < visible)
        keys[int64_t(r) * stride + p] =
            (uint64_t(~sortable_f32_dev(sum)) << kIdxBits) | uint64_t(p);
    }
}

// The boundary key under the same total order as streaming top-k. Only
// its final <=256-key bin needs ranking; every smaller key is selected.
// Pool ids make even an all-equal-score row terminate exactly.
__global__ void prefill_select_keys_kernel(
    const uint64_t* keys, int64_t stride, const int64_t* pos,
    int64_t n_pools, int select_k, int kpool, int max_selected,
    int32_t* topk_out, int32_t* out_counts, int parts, uint64_t* partials) {
  dgpp_pdl_wait();
  extern __shared__ uint32_t smem[];
  uint32_t* best_hi = smem;
  uint32_t* best_lo = best_hi + select_k;
  uint32_t* cand_hi = best_lo + select_k;
  uint32_t* cand_lo = cand_hi + kSelectHistBins;
  int32_t* scratch = reinterpret_cast<int32_t*>(cand_lo + kSelectStopCandidates);
  int* smem_count = scratch + select_k;
  int32_t* hist = reinterpret_cast<int32_t*>(cand_hi);
  __shared__ int boundary, below, definite_count, candidate_count;
  const int r = blockIdx.x;
  const int part = blockIdx.y;
  const int64_t total_visible = min((pos[r] + 1) / kpool, n_pools);
  const int64_t lo = total_visible * part / parts;
  const int64_t visible = total_visible * (part + 1) / parts - lo;
  const uint64_t* row = keys + int64_t(r) * stride + lo;
  for (int i = threadIdx.x; i < select_k; i += blockDim.x) {
    const uint64_t key = visible <= select_k && i < visible
        ? (total_visible > select_k ? row[i] : uint64_t(lo + i)) : ~uint64_t(0);
    best_hi[i] = uint32_t(key >> 32);
    best_lo[i] = uint32_t(key);
  }
  __syncthreads();
  if (visible > select_k) {
    uint64_t prefix = 0;
    int prefix_shift = kSelectKeyBits;
    int remaining = select_k, lower = 0, count = 0, shift = 0;
    for (;;) {
      shift = max(0, prefix_shift - kSelectRadixBits);
      const int bits = prefix_shift - shift;
      for (int i = threadIdx.x; i < kSelectHistBins; i += blockDim.x) hist[i] = 0;
      __syncthreads();
      for (int64_t p = threadIdx.x; p < visible; p += blockDim.x) {
        const uint64_t key = row[p];
        if ((key >> prefix_shift) == prefix)
          prefill_hist_add(hist, select_digit(key, shift, bits));
      }
      __syncthreads();
      select_find_bin(hist, remaining, &boundary, &below);
      count = hist[boundary];
      lower += below;
      remaining -= below;
      prefix = (prefix << bits) | uint64_t(boundary);
      __syncthreads();
      if (count <= kSelectStopCandidates || shift == 0) break;
      prefix_shift = shift;
    }
    if (threadIdx.x == 0) definite_count = candidate_count = 0;
    __syncthreads();
    for (int64_t p = threadIdx.x; p < visible; p += blockDim.x) {
      const uint64_t key = row[p];
      const uint64_t top = key >> shift;
      if (top < prefix) {
        const int i = atomicAdd(&definite_count, 1);
        if (i < select_k) {
          best_hi[i] = uint32_t(key >> 32);
          best_lo[i] = uint32_t(key);
        }
      } else if (top == prefix) {
        const int i = atomicAdd(&candidate_count, 1);
        if (i < kSelectStopCandidates) {
          cand_hi[i] = uint32_t(key >> 32);
          cand_lo[i] = uint32_t(key);
        }
      }
    }
    __syncthreads();
    const int i = threadIdx.x;
    const uint32_t hi = i < count ? cand_hi[i] : 0xffffffffu;
    const uint32_t lo = i < count ? cand_lo[i] : 0xffffffffu;
    int rank = 0;
#pragma unroll 8
    for (int j = 0; j < count; ++j) rank += key_less(cand_hi[j], cand_lo[j], hi, lo);
    if (i < count && rank < remaining) {
      best_hi[lower + rank] = hi;
      best_lo[lower + rank] = lo;
    }
    __syncthreads();
  }
  if (parts > 1) {
    // A pool outside its partition's top-k cannot enter the global top-k.
    // Sort only these survivors for the exact pairwise merge tree.
    bitonic_sort_asc(best_hi, best_lo, select_k);
    uint64_t* out = partials + (int64_t(r) * parts + part) * select_k;
    for (int i = threadIdx.x; i < select_k; i += blockDim.x)
      out[i] = (uint64_t(best_hi[i]) << 32) | best_lo[i];
    return;
  }
  const int cnt = expand_from_best(best_hi, best_lo, select_k, pos[r], kpool,
                                   max_selected, topk_out + int64_t(r) * max_selected,
                                   scratch, smem_count);
  if (threadIdx.x == 0) out_counts[r] = cnt;
}

// Each level owns disjoint ranges of the partial array and writes its
// winner into the left child's slot. Kernel boundaries order levels; no
// global spin barrier or second full workspace is needed. The final merge
// expands the selected pools. Its work is bounded by select_k, not context.
__global__ void prefill_merge_keys_kernel(
    uint64_t* partials, const int64_t* pos, int parts, int step,
    int select_k, int kpool, int max_selected, int32_t* topk_out, int32_t* out_counts) {
  dgpp_pdl_wait();
  extern __shared__ uint32_t smem[];
  uint32_t* hi = smem;
  uint32_t* lo = hi + 2 * select_k;
  int32_t* scratch = reinterpret_cast<int32_t*>(lo + 2 * select_k);
  const int r = blockIdx.x;
  const int part = blockIdx.y * 2 * step;
  uint64_t* left = partials + (int64_t(r) * parts + part) * select_k;
  const uint64_t* right = left + int64_t(step) * select_k;
  for (int i = threadIdx.x; i < select_k; i += blockDim.x) {
    const uint64_t a = left[i], b = right[select_k - 1 - i];
    hi[i] = uint32_t(a >> 32); lo[i] = uint32_t(a);
    hi[select_k + i] = uint32_t(b >> 32); lo[select_k + i] = uint32_t(b);
  }
  __syncthreads();
  bitonic_merge_asc(hi, lo, 2 * select_k);
  if (2 * step < parts) {
    for (int i = threadIdx.x; i < select_k; i += blockDim.x)
      left[i] = (uint64_t(hi[i]) << 32) | lo[i];
    return;
  }
  const int cnt = expand_from_best(hi, lo, select_k, pos[r], kpool, max_selected,
                                   topk_out + int64_t(r) * max_selected,
                                   scratch, scratch + select_k);
  if (threadIdx.x == 0) out_counts[r] = cnt;
}

// TILE: the streaming tile (kSelectTile for select_k <= 1024, twice that
// for the full model's 2048).
template <bool kRelu, int TILE>
__global__ void select_prefill_kernel(const float* dot, int64_t dot_stride,
                                      const float* w_folded,
                                      const float* k_scale, const int64_t* pos,
                                      int rows, int64_t n_pools, int heads,
                                      int select_k, int kpool,
                                      int max_selected, int32_t* topk_out,
                                      int32_t* out_counts) {
  dgpp_pdl_wait();
  extern __shared__ uint64_t smem_u64[];
  uint32_t* best_hi = reinterpret_cast<uint32_t*>(smem_u64);  // [select_k]
  uint32_t* best_lo = best_hi + select_k;                     // [select_k]
  uint32_t* tile_hi = best_lo + select_k;                     // [TILE]
  uint32_t* tile_lo = tile_hi + TILE;                         // [TILE]
  int32_t* scratch = reinterpret_cast<int32_t*>(tile_lo + TILE);
  int* smem_count = reinterpret_cast<int*>(scratch + select_k);

  const int r = blockIdx.x;
  const int64_t visible = (pos[r] + 1) / kpool;
  for (int i = threadIdx.x; i < select_k; i += blockDim.x) {
    best_hi[i] = 0xFFFFFFFFu;
    best_lo[i] = 0xFFFFFFFFu;
  }
  __syncthreads();
  PrefillKeyFn<kRelu> fn{dot, w_folded, k_scale, dot_stride, heads, r};
  select_topk_stream<TILE>(fn, 0, min(visible, n_pools), best_hi, best_lo, tile_hi,
                           tile_lo, select_k);
  __syncthreads();
  const int cnt = expand_from_best(best_hi, best_lo, select_k, pos[r], kpool,
                                   max_selected,
                                   topk_out + int64_t(r) * max_selected,
                                   scratch, smem_count);
  if (threadIdx.x == 0) out_counts[r] = cnt;
}

// ---------------------------------------------------------------------
// MLA absorbed attention
// ---------------------------------------------------------------------

// absorb_q: q_tilde[r, h, :] = q[r, h, :] (nope) x W_uk[h] (nope x kv_lora).
// One block of kAbsorbGroups x 128 threads per (row, head). Every thread
// owns 8 output columns (one uint4 of W per row it visits); the groups
// split the nope rows round-robin, so a block keeps kAbsorbGroups x 128 x
// (rows in flight) loads outstanding instead of 128 x 4 — the previous
// one-group form was latency-bound at ~40 us per layer for 2 MB of
// weights. The groups' partials meet in shared memory and are summed in
// group order (deterministic; the reassociation vs. the single-chain
// version is fp32 rounding, accepted 2026-09-02). kv_b is the
// checkpoint's interleaved layout: head h owns rows [h*(nope+v),
// h*(nope+v)+nope) of W_uk.
constexpr int kAbsorbGroups = 8;
constexpr int kAbsorbGroupThreads = 128;
constexpr int kAbsorbThreads = kAbsorbGroups * kAbsorbGroupThreads;

__global__ __launch_bounds__(kAbsorbThreads) void absorb_q_kernel(
    const uint16_t* q, const uint16_t* kv_b, uint16_t* q_tilde,
    int local_heads, int nope, int v, int kv_lora, int rope) {
  dgpp_pdl_wait();
  const int64_t r = blockIdx.x;
  const int h = blockIdx.y;
  const int head_rows = nope + v;
  const uint16_t* qh = q + (r * local_heads + h) * (nope + rope);
  const uint16_t* wuk = kv_b + int64_t(h) * head_rows * kv_lora;
  uint16_t* out = q_tilde + (r * local_heads + h) * (kv_lora + rope);
  __shared__ float qs[256];
  __shared__ __align__(16) float partial[kAbsorbGroups][512];
  for (int d = threadIdx.x; d < nope; d += blockDim.x)
    qs[d] = bf16_bits_to_float(qh[d]);
  // The rope tail rides along unchanged (rotated upstream).
  for (int t = threadIdx.x; t < rope; t += blockDim.x) out[kv_lora + t] = qh[nope + t];
  __syncthreads();
  const int group = threadIdx.x / kAbsorbGroupThreads;
  const int col = (threadIdx.x % kAbsorbGroupThreads) * 8;
  // kv_lora % 8 == 0 and kv_lora <= 512 validated at launch; smaller ranks
  // idle their excess threads (an unguarded write would land in the next
  // head's row — silent corruption, not an error).
  if (col < kv_lora) {
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
    for (int d = group; d < nope; d += kAbsorbGroups) {
      const float qv = qs[d];
      const uint4 wv =
          *reinterpret_cast<const uint4*>(wuk + int64_t(d) * kv_lora + col);
      const uint32_t* w32 = reinterpret_cast<const uint32_t*>(&wv);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 wf = bf16x2_to_float2(w32[j]);
        acc[2 * j] += qv * wf.x;
        acc[2 * j + 1] += qv * wf.y;
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) partial[group][col + j] = acc[j];
  }
  __syncthreads();
  if (group == 0 && col < kv_lora) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float total = partial[0][col + j];
#pragma unroll
      for (int g = 1; g < kAbsorbGroups; ++g) total += partial[g][col + j];
      out[col + j] = float_to_bf16_bits(total);
    }
  }
}

// The row-batched decode form (2026-09-26): a block owns kTile rows of one
// head, so every W_uk load feeds kTile rows' FMAs — the one-row kernel
// above reads the head's 197 KB once per (row, head) block, and at eight
// or sixteen decode rows (the batched MTP verify) those re-reads, L2 hits
// though they are, were the layer's cost. Thread (group, col) keeps the
// one-row kernel's d chain (d = group, group + kAbsorbGroups, ...) and its
// eight columns, then the groups' partials are summed IN GROUP ORDER
// through one running row (group g adds its partial in phase g), which is
// the one-row kernel's `total = partial[0]; total += partial[g]` sequence
// — so every row's bits are the one-row kernel's, whatever rows share the
// block (the decode contract: a batched row bitwise the row alone), and
// the shared memory stays static (no dynamic opt-in on a capture path).
// Rows past `rows` in the last tile stage zeros and store nothing.
template <int kTile>
__global__ __launch_bounds__(kAbsorbThreads) void absorb_q_rows_kernel(
    const uint16_t* q, const uint16_t* kv_b, uint16_t* q_tilde, int rows,
    int local_heads, int nope, int v, int kv_lora, int rope) {
  dgpp_pdl_wait();
  const int r0 = blockIdx.x * kTile;
  const int h = blockIdx.y;
  const int head_rows = nope + v;
  const int nr = min(kTile, rows - r0);
  const uint16_t* wuk = kv_b + int64_t(h) * head_rows * kv_lora;
  __shared__ float qs[kTile][256];
  __shared__ __align__(16) float run[kTile][512];
  for (int i = threadIdx.x; i < kTile * nope; i += blockDim.x) {
    const int r = i / nope, d = i - r * nope;
    qs[r][d] = r < nr ? bf16_bits_to_float(
                            q[(int64_t(r0 + r) * local_heads + h) * (nope + rope) + d])
                      : 0.0f;
  }
  // The rope tail rides along unchanged (rotated upstream).
  for (int i = threadIdx.x; i < nr * rope; i += blockDim.x) {
    const int r = i / rope, t = i - r * rope;
    q_tilde[(int64_t(r0 + r) * local_heads + h) * (kv_lora + rope) + kv_lora + t] =
        q[(int64_t(r0 + r) * local_heads + h) * (nope + rope) + nope + t];
  }
  __syncthreads();
  const int group = threadIdx.x / kAbsorbGroupThreads;
  const int col = (threadIdx.x % kAbsorbGroupThreads) * 8;
  float acc[kTile][8];
#pragma unroll
  for (int r = 0; r < kTile; ++r)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[r][j] = 0.0f;
  if (col < kv_lora) {
#pragma unroll 4
    for (int d = group; d < nope; d += kAbsorbGroups) {
      const uint4 wv =
          *reinterpret_cast<const uint4*>(wuk + int64_t(d) * kv_lora + col);
      const uint32_t* w32 = reinterpret_cast<const uint32_t*>(&wv);
      float wf[8];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = bf16x2_to_float2(w32[j]);
        wf[2 * j] = f.x;
        wf[2 * j + 1] = f.y;
      }
#pragma unroll
      for (int r = 0; r < kTile; ++r) {
        const float qv = qs[r][d];
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[r][j] += qv * wf[j];
      }
    }
  }
  for (int g = 0; g < kAbsorbGroups; ++g) {
    if (group == g && col < kv_lora) {
#pragma unroll
      for (int r = 0; r < kTile; ++r)
#pragma unroll
        for (int j = 0; j < 8; ++j)
          run[r][col + j] = g == 0 ? acc[r][j] : run[r][col + j] + acc[r][j];
    }
    __syncthreads();
  }
  if (group == 0 && col < kv_lora) {
    for (int r = 0; r < nr; ++r) {
      uint16_t* out = q_tilde + (int64_t(r0 + r) * local_heads + h) * (kv_lora + rope);
#pragma unroll
      for (int j = 0; j < 8; ++j) out[col + j] = float_to_bf16_bits(run[r][col + j]);
    }
  }
}

// One block per (row, split, head-group). Thread (h, g) within the block:
// head h, dim group g (contiguous lanes, so the group reduce is shuffles).
//
// smem strides are PADDED to break bank conflicts: with the natural strides
// (row 512 floats, group 64) every thread's address is congruent mod 32
// banks — a 32-way conflict on every access (measured: 85% of this kernel's
// cycles). Padding (16B-aligned for uint4/float4 vector access):
//   gstride    = dslice + 8            (group windows in a padded row)
//   row_stride = groups*gstride - 8 + 16
//   scores_stride = kAttnTile + 1
// A group's logical dims [g*dslice, g*dslice+dslice) live at padded offset
// [g*gstride, g*gstride+dslice); the pad words between groups are never
// read as data. All loads/stores/publish use this one mapping.
// The rope tail (plan D3) is a separate window after the value windows:
// [rope_base, rope_base + rope), each thread scoring its slice of it
// (ceil(rope / groups) dims) on top of its value window; the value
// accumulation never touches it. Flash's rope 0 leaves every rope loop
// empty — the partials are bitwise the rope-free kernel's.
template <LatentFormat F>
__global__ void attn_partial_kernel(
    const uint16_t* q_tilde, const uint8_t* latent_cache,
    const float* latent_scale, const int32_t* req_ids, const int32_t* topk,
    int topk_stride, const int32_t* counts, int n_split, int local_heads,
    int kv_lora, int rope, int block_tokens, const int32_t* block_tables,
    int blocks_per_request, float scale, float* m_ws, float* l_ws,
    float* c_ws) {
  dgpp_pdl_wait();
  const int64_t r = blockIdx.x;
  const int s = blockIdx.y;
  const int hpb = local_heads / gridDim.z;  // heads per block
  const int h0 = blockIdx.z * hpb;
  const int groups = blockDim.x / hpb;
  const int h = h0 + threadIdx.x / groups;
  const int g = threadIdx.x % groups;
  const int dslice = kv_lora / groups;
  const int gstride = dslice + 8;
  const int rope_base = groups * gstride + 8;  // the tail's window (padded row)
  const int row_stride = rope_base + (rope > 0 ? rope + 8 : 0);
  const size_t row_bytes = latent_cache_row_bytes(F, kv_lora, rope);
  const int sd = kv_lora + rope;      // a head's absorbed-query width
  const int rs = rope > 0 ? (rope + groups - 1) / groups : 0;  // rope dims per thread
  const int r0 = g * rs;
  const int rn = rope > 0 ? max(0, min(rope, r0 + rs) - r0) : 0;
  const int hl = h - h0;              // head local to this block
  const int d0 = g * gstride;         // group window start (padded row)

  const int cnt = counts[r];
  const int chunk = (cnt + n_split - 1) / n_split;
  const int t_begin = s * chunk;
  const int t_end = min(cnt, t_begin + chunk);

  // Empty split (padding rows, short contexts, counts far below the split
  // granularity): publish zeroed partials and exit before any smem traffic
  // — combine requires initialized values for every (row, split, head).
  if (t_begin >= t_end) {
    const int64_t base = (r * n_split + s) * local_heads;
    for (int dd = threadIdx.x; dd < hpb * kv_lora; dd += blockDim.x) {
      const int hh = dd / kv_lora;
      const int cc = dd % kv_lora;
      c_ws[(base + h0 + hh) * kv_lora + cc] = 0.0f;
    }
    if (threadIdx.x < hpb) {
      m_ws[base + h0 + threadIdx.x] = -INFINITY;
      l_ws[base + h0 + threadIdx.x] = 0.0f;
    }
    return;
  }

  // The thread's q window and its c accumulator live in REGISTERS
  //: the head's c row in smem (37 KB) and the q rows (19 KB)
  // put the block at 95 KB — one block per SM, the 64-block decode grid
  // in two waves at ~180 us per call. With both in registers the block
  // is ~40 KB, two per SM, one wave. Per-thread arithmetic and its order
  // are unchanged (each thread always owned exactly this window), so the
  // partials are bitwise the smem form's. The window is at most
  // kAttnMaxDslice dims (the launcher sizes the head groups so).
  extern __shared__ uint16_t sm16[];
  uint16_t* lat = sm16;  // [kAttnTile * row_stride] padded
  float* scores = reinterpret_cast<float*>(lat + kAttnTile * row_stride);
  float* l = scores + hpb * (kAttnTile + 1);  // [hpb], single writer
  const int scores_stride = kAttnTile + 1;
  const bool vec = (dslice % 8) == 0;  // vector paths need 8-wide groups
  const int ds8 = dslice / 8;

  // The running softmax max is per-lane register state: the group's
  // butterfly reduction leaves identical values in every lane, so no
  // shared m exists to order.
  float m_reg = -INFINITY;
  float creg[kAttnMaxDslice];
  uint4 qreg[kAttnMaxDslice / 8];
  float qrope[kAttnMaxRopeSlice];
#pragma unroll
  for (int i = 0; i < kAttnMaxDslice; ++i) creg[i] = 0.0f;
#pragma unroll
  for (int i = 0; i < kAttnMaxRopeSlice; ++i)
    qrope[i] = i < rn ? bf16_bits_to_float(
                            q_tilde[(r * local_heads + h) * sd + kv_lora + r0 + i])
                      : 0.0f;
  {
    const uint16_t* qrow = q_tilde + (r * local_heads + h) * sd + g * dslice;
    if (vec) {
#pragma unroll
      for (int u = 0; u < kAttnMaxDslice / 8; ++u)
        qreg[u] = u < ds8 ? *reinterpret_cast<const uint4*>(qrow + u * 8)
                          : make_uint4(0u, 0u, 0u, 0u);
    } else {
#pragma unroll
      for (int u = 0; u < kAttnMaxDslice / 8; ++u) {
        uint32_t wds[4] = {0u, 0u, 0u, 0u};
#pragma unroll
        for (int j = 0; j < 8; ++j)
          if (u * 8 + j < dslice)
            wds[j / 2] |= uint32_t(qrow[u * 8 + j]) << (16 * (j & 1));
        qreg[u] = make_uint4(wds[0], wds[1], wds[2], wds[3]);
      }
    }
  }
  const uint16_t* q16 = reinterpret_cast<const uint16_t*>(qreg);
  for (int i = threadIdx.x; i < hpb; i += blockDim.x) l[i] = 0.0f;
  __syncthreads();

  const int32_t req = req_ids[r];
  const int32_t* bt = block_tables + int64_t(req) * blocks_per_request;
  const int32_t* toks = topk + int64_t(r) * topk_stride;

  for (int t0 = t_begin; t0 < t_end; t0 += kAttnTile) {
    const int n = min(kAttnTile, t_end - t0);
    // Latent tile gather with the same padded mapping. Unrolled so the
    // token -> block-table -> row chains of several vectors overlap.
    if (vec) {
      const int total = n * groups * ds8;
#pragma unroll 4
      for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int tt = idx / (groups * ds8);
        const int gidx = (idx / ds8) % groups;
        const int c8 = idx % ds8;
        const int64_t tok = toks[t0 + tt];
        const int32_t blk = bt[tok / block_tokens];
        const int64_t phys =
            int64_t(blk) * block_tokens + (tok % block_tokens);
        *reinterpret_cast<uint4*>(&lat[int64_t(tt) * row_stride +
                                       gidx * gstride + c8 * 8]) =
            LatentTile<F>::load8(latent_cache, latent_scale, phys, row_bytes,
                                 kv_lora, gidx * dslice + c8 * 8);
      }
      if (rope > 0) {
        const int r8 = rope / 8;
        for (int idx = threadIdx.x; idx < n * r8; idx += blockDim.x) {
          const int tt = idx / r8, c8 = idx % r8;
          const int64_t tok = toks[t0 + tt];
          const int32_t blk = bt[tok / block_tokens];
          const int64_t phys =
              int64_t(blk) * block_tokens + (tok % block_tokens);
          *reinterpret_cast<uint4*>(&lat[int64_t(tt) * row_stride + rope_base + c8 * 8]) =
              LatentTile<F>::load8(latent_cache, latent_scale, phys, row_bytes,
                                   kv_lora, kv_lora + c8 * 8);
        }
      }
    } else {
      for (int i = threadIdx.x; i < n * groups * dslice; i += blockDim.x) {
        const int tt = i / (groups * dslice);
        const int gidx = (i / dslice) % groups;
        const int cc = i % dslice;
        const int64_t tok = toks[t0 + tt];
        const int32_t blk = bt[tok / block_tokens];
        const int64_t phys =
            int64_t(blk) * block_tokens + (tok % block_tokens);
        lat[tt * row_stride + gidx * gstride + cc] = LatentTile<F>::load1(
            latent_cache, latent_scale, phys, row_bytes, kv_lora, gidx * dslice + cc);
      }
      for (int i = threadIdx.x; i < n * rope; i += blockDim.x) {
        const int tt = i / rope, cc = i % rope;
        const int64_t tok = toks[t0 + tt];
        const int32_t blk = bt[tok / block_tokens];
        const int64_t phys =
            int64_t(blk) * block_tokens + (tok % block_tokens);
        lat[tt * row_stride + rope_base + cc] = LatentTile<F>::load1(
            latent_cache, latent_scale, phys, row_bytes, kv_lora, kv_lora + cc);
      }
    }
    __syncthreads();

    // Scores: each group lane dots its window, then a butterfly leaves the
    // full sum in every lane (keeps the running max per-lane consistent).
    float tile_max = -INFINITY;
    for (int tt = 0; tt < n; ++tt) {
      float partial = 0.0f;
      if (vec) {
        const uint4* l4 =
            reinterpret_cast<const uint4*>(lat + tt * row_stride + d0);
#pragma unroll
        for (int u = 0; u < kAttnMaxDslice / 8; ++u)
          if (u < ds8) partial = dot8_bf16(qreg[u], l4[u], partial);
      } else {
#pragma unroll
        for (int dd = 0; dd < kAttnMaxDslice; ++dd)
          if (dd < dslice)
            partial += bf16_bits_to_float(q16[dd]) *
                       bf16_bits_to_float(lat[tt * row_stride + d0 + dd]);
      }
      // The thread's slice of the rope tail (empty without one).
#pragma unroll
      for (int j = 0; j < kAttnMaxRopeSlice; ++j)
        if (j < rn)
          partial += qrope[j] *
                     bf16_bits_to_float(lat[tt * row_stride + rope_base + r0 + j]);
#pragma unroll
      for (int off = groups / 2; off > 0; off >>= 1)
        partial += __shfl_xor_sync(~0u, partial, off);
      const float score = partial * scale;
      if (g == 0) scores[hl * scores_stride + tt] = score;
      tile_max = fmaxf(tile_max, score);
    }
    __syncthreads();

    // Online softmax update; m in registers, l under its single writer.
    const float m_new = fmaxf(m_reg, tile_max);
    const float rescale = expf(m_reg - m_new);
#pragma unroll
    for (int dd = 0; dd < kAttnMaxDslice; ++dd)
      if (dd < dslice) creg[dd] *= rescale;
    if (g == 0) {
      float ladd = 0.0f;
      for (int tt = 0; tt < n; ++tt)
        ladd += expf(scores[hl * scores_stride + tt] - m_new);
      l[hl] = l[hl] * rescale + ladd;
    }
    // c accumulation: probs round to bf16 (pinned; l stays unrounded).
    // float4 RMW over the group window; element order preserved.
    for (int tt = 0; tt < n; ++tt) {
      const float p = bf16_bits_to_float(float_to_bf16_bits(
          expf(scores[hl * scores_stride + tt] - m_new)));
      if (vec) {
        const uint4* l4 =
            reinterpret_cast<const uint4*>(lat + tt * row_stride + d0);
#pragma unroll
        for (int u = 0; u < kAttnMaxDslice / 8; ++u) {
          if (u >= ds8) break;  // uniform across the group
          const uint4 lv = l4[u];
          const uint32_t* l32 = reinterpret_cast<const uint32_t*>(&lv);
          const float2 w0 = bf16x2_to_float2(l32[0]);  // elems 8u+0,1
          const float2 w1 = bf16x2_to_float2(l32[1]);  // 8u+2,3
          const float2 w2 = bf16x2_to_float2(l32[2]);  // 8u+4,5
          const float2 w3 = bf16x2_to_float2(l32[3]);  // 8u+6,7
          creg[u * 8 + 0] += p * w0.x;
          creg[u * 8 + 1] += p * w0.y;
          creg[u * 8 + 2] += p * w1.x;
          creg[u * 8 + 3] += p * w1.y;
          creg[u * 8 + 4] += p * w2.x;
          creg[u * 8 + 5] += p * w2.y;
          creg[u * 8 + 6] += p * w3.x;
          creg[u * 8 + 7] += p * w3.y;
        }
      } else {
#pragma unroll
        for (int dd = 0; dd < kAttnMaxDslice; ++dd)
          if (dd < dslice)
            creg[dd] += p * bf16_bits_to_float(lat[tt * row_stride + d0 + dd]);
      }
    }
    m_reg = m_new;
    __syncthreads();
  }

  // Publish partials: m/l by the group-0 lane, c by each thread for its
  // own window (the same values the smem form published, by the lanes
  // that computed them).
  const int64_t base_m = (r * n_split + s) * local_heads + h;
  if (g == 0) {
    m_ws[base_m] = m_reg;
    l_ws[base_m] = l[hl];
  }
  float* crow_ws = c_ws + (r * n_split + s) * local_heads * kv_lora +
                   int64_t(h) * kv_lora + g * dslice;
  if (vec) {
#pragma unroll
    for (int u = 0; u < kAttnMaxDslice / 8; ++u) {
      if (u >= ds8) break;
      *reinterpret_cast<float4*>(crow_ws + u * 8) =
          make_float4(creg[u * 8], creg[u * 8 + 1], creg[u * 8 + 2], creg[u * 8 + 3]);
      *reinterpret_cast<float4*>(crow_ws + u * 8 + 4) =
          make_float4(creg[u * 8 + 4], creg[u * 8 + 5], creg[u * 8 + 6], creg[u * 8 + 7]);
    }
  } else {
#pragma unroll
    for (int dd = 0; dd < kAttnMaxDslice; ++dd)
      if (dd < dslice) crow_ws[dd] = creg[dd];
  }
}

// ---------------------------------------------------------------------
// Dense causal attention on tensor cores (the prefill path, 2026-09-05)
// ---------------------------------------------------------------------
// Below index_topk tokens of context the selection is provably dense —
// every visible pool is selected and the tail appended, so a query at
// position p attends to tokens [0, p] — and the per-row gather kernel
// above spends ~130 scalar FMAs per token per thread re-reading the
// prefix once per row (a 2048-token prefill: 1,903 eight-row launches of
// 318 us per layer). This kernel is the dense case on mma.sync:
//   * the M dimension is (row, head) pairs — q_tilde is already the
//     [rows * local_heads, kv_lora] matrix — 32 M-rows per block as two
//     slabs of 16; the 8 warps are (slab, column quarter);
//   * per 32-token latent tile (gathered through the block table into
//     shared memory, shared by all 32 M-rows): S = Q~ . L^T over the
//     warp's k-quarter (16 x 32 per warp), the four quarters' partials
//     summed through shared memory in a FIXED order so every warp of a
//     slab holds identical S; then the FA2 online softmax on the C
//     fragments (row max by quad shuffles, l unrounded, probabilities
//     rounded to bf16 as the split kernel pins) and O += P . L over the
//     warp's 128 output columns, P re-used straight from the C fragments
//     as A fragments;
//   * causal masking per M-row against pos[row]; splits over the token
//     range exactly as the split kernel, publishing the same (m, l, c)
//     partials so dsa_attn_combine merges them unchanged.
// Numerics: bf16 inputs, fp32 accumulation, the same scale/exp/bf16-prob
// chain as the split kernel in a different summation order (the tensor
// core's) — tolerance-equal, not bitwise; deterministic (bitwise repeat).
namespace dense {
constexpr int kM = 32;             // (row, head) M-rows per block
constexpr int kTile = 32;          // latent tokens per tile
constexpr int kThreads = 256;      // 8 warps: 2 slabs x 4 column quarters
// KV: the value width (kv_lora); SD: the score width (kv_lora + rope, the
// cache row and the absorbed query — plan D3). Flash: SD == KV. NTOK: the
// tokens a slab's S tile spans (16 listed, kTile dense) — the exchange
// holds one warp's S partial in the C layout, 4 floats per n8 tile per
// lane. Shared memory holds the latent tile and the exchange; the Q~
// fragments live in registers (2026-10-18: 83 KB per block held one
// block per SM at the 512 widths, and the kernel was issue-latency-bound
// at two warps per scheduler; two blocks fit now).
template <int KV, int SD, int NTOK = kTile>
struct Geo {
  static_assert(SD >= KV && SD % 64 == 0 && KV % 32 == 0, "dsa flash geometry");
  static constexpr int SQ = SD + 8;        // padded smem row (u16 elements)
  static constexpr int CW = KV / 4;        // output columns per warp
  static constexpr int NT = CW / 8;        // n8 tiles per warp (PV)
  static constexpr int KW = SD / 4;        // k dims per warp (S)
  static constexpr int KS = KW / 16;       // k16 steps per warp (S)
  static constexpr int XF = NTOK * 16;     // one warp's S partial: NTOK/8 n8 tiles x 4 floats x 32 lanes
  static constexpr size_t smem_bytes = size_t(kTile) * SQ * 2 + size_t(8) * XF * 4;
};
}  // namespace dense

__device__ __forceinline__ void mma_bf16_16816(float (&c)[4], uint32_t a0,
                                               uint32_t a1, uint32_t a2,
                                               uint32_t a3, uint32_t b0,
                                               uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// kListed (2026-09-05, the sparse regime): instead of the causal range,
// each 16-row SLAB — one query row at local_heads >= 16 — walks its own
// selected-token list (topk[row], counts[row], the split kernel's inputs)
// into its own 16-token latent tile; the two slabs' loops run to the longer
// one's tile count. No union, no membership masks: the slab-is-a-row
// property makes the per-row selection the block's natural structure.
// The listed gather's anomaly record: the first out-of-range token or block
// (first writer wins) and a count; dsa_attn_anomalies() reads and clears it.
__device__ unsigned long long g_attn_anomaly_count = 0;
__device__ long long g_attn_anomaly[6] = {0, 0, 0, 0, 0, 0};
__device__ __forceinline__ void attn_anomaly_record(int64_t tok, int32_t blk,
                                                    int qrow, int split,
                                                    int index, int live) {
  if (atomicAdd(&g_attn_anomaly_count, 1ull) == 0ull) {
    g_attn_anomaly[0] = tok;
    g_attn_anomaly[1] = blk;
    g_attn_anomaly[2] = qrow;
    g_attn_anomaly[3] = split;
    g_attn_anomaly[4] = index;
    g_attn_anomaly[5] = live;
    __threadfence();
  }
}

// Two blocks per SM where two of the geometry's shared tiles fit (the 256
// widths); one at 512, where the 83 KB tile already allows one — the
// register budget then holds the prefetched tile beside the accumulators.
template <int KV, int SD, bool kListed>
constexpr int attn_flash_min_blocks() {
  // The dynamic tile, the listed walk's static address round (below) and
  // the driver's 1 KB per block, twice, within the SM's 100 KB — for the
  // rope-less widths only. The 512 + 64 tail's kernel holds nine k16 steps
  // of Q~ fragments and nine fetches per tile and does not fit the
  // two-block register budget without spilling; compiled that way it
  // faulted DeepSeek-V4's mixed walk (shared_prefill_test, 2026-10-18)
  // while the same source at one block, or the two-block build of the
  // 512-wide kernels, ran clean — so that geometry keeps one block.
  const size_t block_smem = dense::Geo<KV, SD, kListed ? 16 : dense::kTile>::smem_bytes +
                            (kListed ? size_t(2) * 512 * 4 : size_t(8)) + 1024;
  return (kListed && SD == KV && 2 * block_smem <= size_t(100) * 1024) ? 2 : 1;  // the dense tile spills at 128 too
}

// kGroup (2026-10-23, the decode batch past a few rows): the kGroup
// consecutive splits of a row pair also publish ONE merged partial per
// group (the epilogue below: the last of the group's blocks to arrive
// merges the eight per-split partials from L2 with the finish's formula,
// writes the group's partial and discards the eight dead rows from L2 so
// they never write back) — the per-split partials of forty rows at
// thirty-two main and eight window splits were fifty megabytes a layer,
// written by this kernel and read back by the finish, 160 us a layer each
// way at the DRAM rate. A cluster form (distributed shared memory) was
// measured first: it wins only past forty rows, since clusters beyond the
// twelve co-resident ones schedule in coarse waves (+35 us a layer at ten
// to twenty rows). 1 = a partial per split (the prefill tile and the
// scalar form's layout).
template <int KV, int SD, bool kListed, LatentFormat F, int kGroup = 1>
__global__ __launch_bounds__(dense::kThreads, (attn_flash_min_blocks<KV, SD, kListed>())) void attn_flash_kernel(
    const uint16_t* __restrict__ q_tilde, const uint8_t* __restrict__ latent,
    const float* __restrict__ latent_scale,
    const int32_t* __restrict__ req_ids, const int64_t* __restrict__ pos,
    const int32_t* __restrict__ topk, int topk_stride,
    const int32_t* __restrict__ counts, int rows, int n_split, int local_heads,
    int block_tokens, const int32_t* __restrict__ block_tables,
    int blocks_per_request, int blocks_total, float scale, float* __restrict__ m_ws,
    float* __restrict__ l_ws, float* __restrict__ c_ws, int32_t* __restrict__ tickets,
    float* __restrict__ mg_ws, float* __restrict__ lg_ws, float* __restrict__ cg_ws) {
  dgpp_pdl_wait();
  constexpr int NTOK = kListed ? 16 : dense::kTile;  // tokens per tile (per slab when listed)
  using G = dense::Geo<KV, SD, NTOK>;
  constexpr int SQ = G::SQ, CW = G::CW, NT = G::NT, KW = G::KW, KS = G::KS, XF = G::XF;
  constexpr int M = dense::kM;
  constexpr size_t kRowBytes = latent_cache_row_bytes(F, KV, SD - KV);
  constexpr int JT = NTOK / 8;    // n8 tiles of tokens in S
  constexpr int KK = NTOK / 16;   // k16 steps of tokens in P.L
  extern __shared__ __align__(16) uint8_t dsmem[];
  uint16_t* sLall = reinterpret_cast<uint16_t*>(dsmem);        // dense: [32][SQ]; listed: [2][16][SQ]
  float* sX = reinterpret_cast<float*>(sLall + dense::kTile * SQ);  // [8][XF]

  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int slab = warp / 4, quarter = warp % 4;
  const int r = lane / 4, cc = (lane % 4) * 2;
  const int total_m = rows * local_heads;
  const int m0 = blockIdx.x * M;
  const int s = blockIdx.y;
  uint16_t* sL = kListed ? sLall + slab * NTOK * SQ : sLall;

  // The lane's two M-rows (slab rows r and r+8): validity and positions.
  const int m_lo = m0 + slab * 16 + r;
  const int m_hi = m_lo + 8;
  const bool v_lo = m_lo < total_m, v_hi = m_hi < total_m;
  const int64_t p_lo = (!kListed && v_lo) ? pos[m_lo / local_heads] : -1;
  const int64_t p_hi = (!kListed && v_hi) ? pos[m_hi / local_heads] : -1;
  const int qrow_first = m0 / local_heads;
  const int32_t req = req_ids[qrow_first];
  const int32_t* bt = block_tables + int64_t(req) * blocks_per_request;

  // The split's token range: dense — the block's longest causal prefix,
  // split n ways, masked per M-row; listed — the slab's row's selection,
  // split n ways (the split kernel's formula), no mask beyond the count.
  int t_begin = 0, t_end = 0;
  const int32_t* list = nullptr;
  if constexpr (kListed) {
    const int slab_m = m0 + slab * 16;
    const int qrow = min(slab_m, total_m - 1) / local_heads;
    const int cnt = slab_m < total_m ? counts[qrow] : 0;
    const int chunk = (cnt + n_split - 1) / n_split;
    t_begin = s * chunk;
    t_end = min(cnt, t_begin + chunk);
    list = topk + int64_t(qrow) * topk_stride;
    // The slab's own request's block table (2026-09-06, the decode use:
    // a block's two slabs may be rows of different requests).
    bt = block_tables + int64_t(req_ids[qrow]) * blocks_per_request;
  } else {
    const int qrow_last = min(total_m, m0 + M) > m0
                              ? (min(total_m, m0 + M) - 1) / local_heads
                              : qrow_first;
    int64_t p_max = -1;
    for (int q = qrow_first; q <= qrow_last; ++q) p_max = max(p_max, pos[q]);
    const int count = static_cast<int>(p_max + 1);
    const int chunk = (count + n_split - 1) / n_split;
    t_begin = s * chunk;
    t_end = min(count, t_begin + chunk);
  }
  // The block's tile count: the longer slab's (listed) or the shared range's.
  int n_tiles = (t_end - t_begin + NTOK - 1) / NTOK;
  if (n_tiles < 0) n_tiles = 0;
  if constexpr (kListed) {
    // Both slabs must run the same number of barrier rounds.
    __shared__ int s_tiles[2];
    if (threadIdx.x % 128 == 0) s_tiles[slab] = n_tiles;
    __syncthreads();
    n_tiles = max(s_tiles[0], s_tiles[1]);
  }

  // Q~ fragments of this warp's two rows over its k-quarter, in the mma's
  // A layout, from global once per block (zero past the last M-row).
  uint32_t qa[KS][4];
  {
    const uint16_t* q_lo = q_tilde + int64_t(m_lo) * SD;
    const uint16_t* q_hi = q_tilde + int64_t(m_hi) * SD;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      const int k0 = quarter * KW + ks * 16 + cc;
      qa[ks][0] = v_lo ? *reinterpret_cast<const uint32_t*>(q_lo + k0) : 0u;
      qa[ks][1] = v_hi ? *reinterpret_cast<const uint32_t*>(q_hi + k0) : 0u;
      qa[ks][2] = v_lo ? *reinterpret_cast<const uint32_t*>(q_lo + k0 + 8) : 0u;
      qa[ks][3] = v_hi ? *reinterpret_cast<const uint32_t*>(q_hi + k0 + 8) : 0u;
    }
  }

  float acc[NT][4];
#pragma unroll
  for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  float m_lo_run = -INFINITY, m_hi_run = -INFINITY;
  float l_lo = 0.f, l_hi = 0.f;

  // ---- the gather (2026-10-18) ------------------------------------------------
  // A listed row cost three dependent global loads (the list, the block
  // table, the cache row), eight rows per thread in turn, with one block
  // resident per SM: the DeepSeek-V4.1 32K prefill's largest kernel ran at
  // a few percent of every limit. The listed addresses are now resolved
  // per round of kAddrRound tokens into shared memory — once per token,
  // not once per thread per token — and a tile's rows are fetched in two
  // halves whose loads issue together; the second block on the SM covers
  // the latency. The rows decoded and their placement are unchanged:
  // bitwise the previous walk.
  constexpr int kAddrRound = 512;
  constexpr int kTilesPerRound = kAddrRound / NTOK;
  constexpr int GTHREADS = kListed ? 128 : dense::kThreads;  // the gather's threads (per slab when listed)
  constexpr int TPT = NTOK * (SD / 8) / GTHREADS;              // uint4 per thread per tile
  constexpr int HALF = (TPT + 3) / 4;  // uint4 per pass: four passes, the registers of two blocks per SM
  static_assert(NTOK * (SD / 8) % GTHREADS == 0, "the gather's uint4s divide its threads");
  __shared__ int32_t s_phys[2][kListed ? kAddrRound : 1];
  const int gtid = threadIdx.x % GTHREADS;
  const auto resolve_round = [&](int t_round0) {
    if constexpr (kListed) {
      for (int i = gtid; i < kAddrRound; i += GTHREADS) {
        const int t = t_round0 + i;
        int32_t phys = -1;
        if (t < t_end) {
          // The gather is guarded: a token outside the table row or a
          // block outside the pool zero-fills the row and records the first
          // anomaly instead of faulting the context — the listed kernel
          // read an unmapped page on the fabric once in ~10 eager rows of
          // the fallback path; the record names the values.
          const int64_t tok = list[t];
          const int64_t bidx = tok >= 0 ? tok / block_tokens : -1;
          // The block must lie in the pool: blocks_total physical blocks
          // (the table's row width bounds the index into the row; a
          // one-entry table over many physical blocks — the decode
          // window's ring, request r's region r — is bounded by the
          // pool's count, 2026-10-19).
          const int32_t blk = (bidx >= 0 && bidx < blocks_per_request) ? bt[bidx] : -1;
          if (blk >= 0 && blk < blocks_total)
            phys = static_cast<int32_t>(int64_t(blk) * block_tokens + (tok % block_tokens));
          else
            attn_anomaly_record(tok, blk, qrow_first, s, t, t_end - t_begin);
        }
        s_phys[slab][i] = phys;
      }
    }
  };
  // The tile's rows into the shared tile (zero-filled past the slab's live
  // tokens), HALF uint4 per thread per pass.
  const auto fetch_tile = [&](int tile) {
    const int t0 = t_begin + tile * NTOK;
    const int n = max(0, min(NTOK, t_end - t0));
#pragma unroll
    for (int h = 0; h < TPT; h += HALF) {
      uint4 val[HALF];
#pragma unroll
      for (int k = 0; k < HALF; ++k) {
        const int idx = gtid + (h + k) * GTHREADS;
        const int tt = idx / (SD / 8), c8 = idx % (SD / 8);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (h + k < TPT && tt < n) {
          int64_t phys;
          if constexpr (kListed) {
            phys = s_phys[slab][(tile % kTilesPerRound) * NTOK + tt];
          } else {
            const int64_t tok = t0 + tt;
            const int32_t blk = bt[tok / block_tokens];
            phys = int64_t(blk) * block_tokens + (tok % block_tokens);
          }
          if (phys >= 0) v = LatentTile<F>::load8(latent, latent_scale, phys, kRowBytes, KV, c8 * 8);
        }
        val[k] = v;
      }
#pragma unroll
      for (int k = 0; k < HALF; ++k) {
        if (h + k >= TPT) break;
        const int idx = gtid + (h + k) * GTHREADS;
        const int tt = idx / (SD / 8), c8 = idx % (SD / 8);
        *reinterpret_cast<uint4*>(sL + tt * SQ + c8 * 8) = val[k];
      }
    }
  };
  resolve_round(t_begin);

  for (int tile = 0; tile < n_tiles; ++tile) {
    const int t0 = t_begin + tile * NTOK;
    const int n = max(0, min(NTOK, t_end - t0));  // this slab's live tokens
    (void)t0;  // the dense mask reads it
    if (tile > 0 && tile % kTilesPerRound == 0) resolve_round(t0);  // the round's addresses, ahead of its first fetch
    __syncthreads();  // the previous tile's readers are done; the round's addresses landed
    fetch_tile(tile);
    __syncthreads();

    // S partial over this warp's k-quarter: 16 rows x NTOK tokens.
    float sp[JT][4];
#pragma unroll
    for (int j = 0; j < JT; ++j) sp[j][0] = sp[j][1] = sp[j][2] = sp[j][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      const int k0 = quarter * KW + ks * 16;
      const uint32_t a0 = qa[ks][0], a1 = qa[ks][1], a2 = qa[ks][2], a3 = qa[ks][3];
#pragma unroll
      for (int j = 0; j < JT; ++j) {
        const int tn = j * 8 + r;  // token within the tile (the n index)
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sL[tn * SQ + k0 + cc]);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sL[tn * SQ + k0 + cc + 8]);
        mma_bf16_16816(sp[j], a0, a1, a2, a3, b0, b1);
      }
    }
    // Exchange: every warp of the slab sums the four quarters in the same
    // order, so the slab's warps hold bitwise-identical S.
    float* mine = sX + warp * XF + lane * (JT * 4);
#pragma unroll
    for (int j = 0; j < JT; ++j) {
      mine[j * 4 + 0] = sp[j][0];
      mine[j * 4 + 1] = sp[j][1];
      mine[j * 4 + 2] = sp[j][2];
      mine[j * 4 + 3] = sp[j][3];
    }
    __syncthreads();
    float sc[JT][4];
#pragma unroll
    for (int j = 0; j < JT; ++j) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float* q0 = sX + (slab * 4 + 0) * XF + lane * (JT * 4) + j * 4 + i;
        float v = q0[0];
        v += q0[1 * XF];
        v += q0[2 * XF];
        v += q0[3 * XF];
        sc[j][i] = v;
      }
    }
    // Scale and mask (causal per M-row when dense; the count when listed);
    // the row maxima of this tile.
    float mx_lo = -INFINITY, mx_hi = -INFINITY;
#pragma unroll
    for (int j = 0; j < JT; ++j) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int tt = j * 8 + cc + (i & 1);
        const bool hi = i >= 2;
        bool ok = tt < n && (hi ? v_hi : v_lo);
        if constexpr (!kListed) {
          const int64_t tok = t0 + tt;
          ok = ok && (hi ? tok <= p_hi : tok <= p_lo);
        }
        const float val = ok ? sc[j][i] * scale : -INFINITY;
        sc[j][i] = val;
        if (hi) mx_hi = fmaxf(mx_hi, val); else mx_lo = fmaxf(mx_lo, val);
      }
    }
    mx_lo = fmaxf(mx_lo, __shfl_xor_sync(~0u, mx_lo, 1));
    mx_lo = fmaxf(mx_lo, __shfl_xor_sync(~0u, mx_lo, 2));
    mx_hi = fmaxf(mx_hi, __shfl_xor_sync(~0u, mx_hi, 1));
    mx_hi = fmaxf(mx_hi, __shfl_xor_sync(~0u, mx_hi, 2));
    const float mn_lo = fmaxf(m_lo_run, mx_lo);
    const float mn_hi = fmaxf(m_hi_run, mx_hi);
    const float rs_lo = mn_lo == -INFINITY ? 1.f : expf(m_lo_run - mn_lo);
    const float rs_hi = mn_hi == -INFINITY ? 1.f : expf(m_hi_run - mn_hi);
    // Probabilities: fp32 for l, bf16 for the c accumulation (the pin).
    uint32_t pa[KK][4];
    float la_lo = 0.f, la_hi = 0.f;
#pragma unroll
    for (int j = 0; j < JT; ++j) {
      float pv[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const bool hi = i >= 2;
        const float mn = hi ? mn_hi : mn_lo;
        pv[i] = (sc[j][i] == -INFINITY) ? 0.f : expf(sc[j][i] - mn);
        if (hi) la_hi += pv[i]; else la_lo += pv[i];
      }
      const uint32_t lo01 = uint32_t(float_to_bf16_bits(pv[0])) |
                            (uint32_t(float_to_bf16_bits(pv[1])) << 16);
      const uint32_t hi01 = uint32_t(float_to_bf16_bits(pv[2])) |
                            (uint32_t(float_to_bf16_bits(pv[3])) << 16);
      // k16 step kk = j / 2 holds n8 tiles 2kk (regs 0,1) and 2kk+1 (2,3).
      pa[j / 2][(j % 2) * 2 + 0] = lo01;
      pa[j / 2][(j % 2) * 2 + 1] = hi01;
    }
    la_lo += __shfl_xor_sync(~0u, la_lo, 1);
    la_lo += __shfl_xor_sync(~0u, la_lo, 2);
    la_hi += __shfl_xor_sync(~0u, la_hi, 1);
    la_hi += __shfl_xor_sync(~0u, la_hi, 2);
    l_lo = l_lo * rs_lo + la_lo;
    l_hi = l_hi * rs_hi + la_hi;
    m_lo_run = mn_lo;
    m_hi_run = mn_hi;
    // O rescale, then O += P . L over the warp's columns.
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      acc[j][0] *= rs_lo;
      acc[j][1] *= rs_lo;
      acc[j][2] *= rs_hi;
      acc[j][3] *= rs_hi;
    }
#pragma unroll
    for (int kk = 0; kk < KK; ++kk) {
      const int tb = kk * 16 + cc;  // the fragment's first token (k index)
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const int d = quarter * CW + j * 8 + r;  // output column (n index)
        const uint32_t b0 = uint32_t(sL[tb * SQ + d]) | (uint32_t(sL[(tb + 1) * SQ + d]) << 16);
        const uint32_t b1 = uint32_t(sL[(tb + 8) * SQ + d]) | (uint32_t(sL[(tb + 9) * SQ + d]) << 16);
        mma_bf16_16816(acc[j], pa[kk][0], pa[kk][1], pa[kk][2], pa[kk][3], b0, b1);
      }
    }
  }

  // Publish the split partials in the combine kernel's layout.
  auto publish = [&](int mrow, float mrun, float lrun, int which) {
    if (mrow >= total_m) return;
    const int qr = mrow / local_heads, h = mrow % local_heads;
    const int64_t idx = (int64_t(qr) * n_split + s) * local_heads + h;
    if (cc == 0) {
      m_ws[idx] = mrun;
      l_ws[idx] = lrun;
    }
    float* crow = c_ws + idx * KV;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int d = quarter * CW + j * 8 + cc;
      *reinterpret_cast<float2*>(crow + d) =
          which == 0 ? make_float2(acc[j][0], acc[j][1])
                     : make_float2(acc[j][2], acc[j][3]);
    }
  };
  publish(m_lo, m_lo_run, l_lo, 0);
  publish(m_hi, m_hi_run, l_hi, 1);
  if constexpr (kGroup > 1) {
    // The group merge: splits s0 .. s0 + kGroup - 1 of this row pair take a
    // ticket (tickets[blockIdx.x * n_groups + g], zero at rest, reset by
    // the last — replay-safe), and the LAST block to arrive merges the
    // group's partials from L2 with the finish kernel's formula — mhat =
    // max m_s, w_s = exp(m_s - mhat), l = sum w_s l_s, c = sum w_s c_s in
    // split order — into the group's partial at split index g of
    // n_split / kGroup (mg_ws / lg_ws / cg_ws), then discards the eight
    // dead c rows from L2. The grouping is fixed per request (kGroup
    // splits a group whatever the batch), so a row's output does not
    // depend on the rows beside it; an empty split (m = -inf, l = 0,
    // c = 0) weighs nothing and an empty group publishes m = -inf, which
    // the finish skips. The fence / ticket order is the router's: a block
    // fences its partial before its ticket; the last fences again before
    // reading them (L2 loads: the rows were written by other SMs).
    __shared__ int s_last;
    __shared__ float s_w[kGroup][M];
    const int g = s / kGroup;
    const int n_groups = n_split / kGroup;
    const int s0 = g * kGroup;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      int32_t* t = tickets + static_cast<int64_t>(blockIdx.x) * n_groups + g;
      const int ticket = atomicAdd(t, 1);
      s_last = ticket == kGroup - 1 ? 1 : 0;
      if (s_last) *t = 0;
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    for (int mrow = static_cast<int>(threadIdx.x); mrow < M; mrow += dense::kThreads) {
      const int m = m0 + mrow;
      if (m >= total_m) continue;
      const int qr = m / local_heads, h = m % local_heads;
      float mb[kGroup], lb[kGroup];
      float mhat = -INFINITY;
#pragma unroll
      for (int b = 0; b < kGroup; ++b) {
        const int64_t idx = (int64_t(qr) * n_split + s0 + b) * local_heads + h;
        mb[b] = __ldcg(m_ws + idx);
        lb[b] = __ldcg(l_ws + idx);
        mhat = fmaxf(mhat, mb[b]);
      }
      float lsum = 0.f;
#pragma unroll
      for (int b = 0; b < kGroup; ++b) {
        const float w = (mhat == -INFINITY || mb[b] == -INFINITY) ? 0.f : expf(mb[b] - mhat);
        lsum += w * lb[b];
        s_w[b][mrow] = w;
      }
      const int64_t gidx = (int64_t(qr) * n_groups + g) * local_heads + h;
      mg_ws[gidx] = mhat;
      lg_ws[gidx] = lsum;
    }
    __syncthreads();
    // The c rows: M rows x KV columns as float4s over the block, kPer
    // outputs a thread a round with every group vector of the round
    // loaded before the first add — 16 L2 loads in flight a thread (one
    // output a round left one block reading 512 KB at ~20 GB/s: a 25 us
    // tail on every launch, 2026-10-23). A dead output (an M-row past the
    // batch) loads the block's first row and stores nothing.
    constexpr int V4 = KV / 4;
    constexpr int kPer = 2;
    const int qr0 = m0 / local_heads, h0 = m0 % local_heads;
    for (int i0 = static_cast<int>(threadIdx.x); i0 < M * V4; i0 += kPer * dense::kThreads) {
      float4 v[kPer][kGroup];
      int mrow[kPer], col[kPer], qr[kPer], h[kPer];
      bool live[kPer];
#pragma unroll
      for (int u = 0; u < kPer; ++u) {
        const int i = i0 + u * dense::kThreads;
        live[u] = i < M * V4 && m0 + i / V4 < total_m;
        mrow[u] = live[u] ? i / V4 : 0;
        col[u] = live[u] ? (i % V4) * 4 : 0;
        const int m = m0 + mrow[u];
        qr[u] = live[u] ? m / local_heads : qr0;
        h[u] = live[u] ? m % local_heads : h0;
#pragma unroll
        for (int b = 0; b < kGroup; ++b)
          v[u][b] = __ldcg(reinterpret_cast<const float4*>(
              c_ws + ((int64_t(qr[u]) * n_split + s0 + b) * local_heads + h[u]) * KV + col[u]));
      }
#pragma unroll
      for (int u = 0; u < kPer; ++u) {
        if (!live[u]) continue;
        float4 csum = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int b = 0; b < kGroup; ++b) {
          const float w = s_w[b][mrow[u]];
          csum.x += w * v[u][b].x;
          csum.y += w * v[u][b].y;
          csum.z += w * v[u][b].z;
          csum.w += w * v[u][b].w;
        }
        const int64_t gidx = (int64_t(qr[u]) * n_groups + g) * local_heads + h[u];
        *reinterpret_cast<float4*>(cg_ws + gidx * KV + col[u]) = csum;
      }
    }
    // The group's per-split c rows are dead: discard their 128-byte lines
    // from L2 (the rows are 2 KB on a 256-byte-aligned buffer) so the
    // fifty megabytes a layer never write back to DRAM — after every
    // thread's merge reads of them (a thread ahead discarded lines a
    // thread behind was still loading: the group-prefill invariance gate
    // and the 40-step decode budget caught it, 2026-10-23).
    __syncthreads();
    constexpr int kLines = KV * 4 / 128;
    for (int i = static_cast<int>(threadIdx.x); i < kGroup * M * kLines; i += dense::kThreads) {
      const int b = i / (M * kLines), rem = i % (M * kLines), mrow = rem / kLines, line = rem % kLines;
      const int m = m0 + mrow;
      if (m >= total_m) continue;
      const int qr = m / local_heads, h = m % local_heads;
      const float* row = c_ws + ((int64_t(qr) * n_split + s0 + b) * local_heads + h) * KV;
      asm volatile("discard.global.L2 [%0], 128;" ::"l"(row + line * 32) : "memory");
    }
  }
}

constexpr int kCombineStagedSplits = 128;
__global__ void attn_combine_kernel(const float* m_ws, const float* l_ws,
                                    const float* c_ws, int n_split,
                                    int local_heads, int kv_lora,
                                    float* c_out) {
  dgpp_pdl_wait();
  // One block per (row, head): the old (row)-block form launched one block
  // for single-row decode — a single SM merging 64 heads x 512 dims.
  // Every split's m and l are staged once per block, one thread per split
  // (2026-10-20: every thread read every split's m for the maximum and its
  // m and l again per element, and recomputed the exponentials; thread 0
  // now forms the maximum, the weights p_s = exp(m_s - mhat) and the
  // denominator from shared memory — the same arithmetic in the same
  // order, so the outputs are bitwise; csa2's finish has the same form);
  // the element loop reads its n_split c values and the staged weights.
  __shared__ float s_m[kCombineStagedSplits];
  __shared__ float s_l[kCombineStagedSplits];
  __shared__ float s_p[kCombineStagedSplits];
  __shared__ float s_mhat;
  __shared__ float s_den;
  const int64_t r = blockIdx.x;
  const int h = blockIdx.y;
  const int64_t base_m = (r * n_split) * local_heads + h;
  const int64_t total = int64_t(local_heads) * kv_lora;
  const bool staged = n_split <= kCombineStagedSplits;
  float mhat = -INFINITY;
  if (staged) {
    for (int s = threadIdx.x; s < n_split; s += blockDim.x) {
      s_m[s] = m_ws[base_m + int64_t(s) * local_heads];
      s_l[s] = l_ws[base_m + int64_t(s) * local_heads];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float mx = -INFINITY;
      for (int s = 0; s < n_split; ++s) mx = fmaxf(mx, s_m[s]);
      float den = 0.0f;
      if (mx != -INFINITY) {
        for (int s = 0; s < n_split; ++s) {
          const float p = expf(s_m[s] - mx);
          s_p[s] = p;
          den += p * s_l[s];
        }
      }
      s_mhat = mx;
      s_den = den;
    }
    __syncthreads();
    mhat = s_mhat;
  } else {
    for (int s = 0; s < n_split; ++s)
      mhat = fmaxf(mhat, m_ws[base_m + int64_t(s) * local_heads]);
  }
  for (int64_t i = int64_t(h) * kv_lora + threadIdx.x;
       i < int64_t(h + 1) * kv_lora; i += blockDim.x) {
    if (mhat == -INFINITY) {
      c_out[r * total + i] = 0.0f;  // empty row (padding)
      continue;
    }
    float num = 0.0f;
    float den = 0.0f;
    if (staged) {
      for (int s = 0; s < n_split; ++s) num += s_p[s] * c_ws[(r * n_split + s) * total + i];
      den = s_den;
    } else {
      for (int s = 0; s < n_split; ++s) {
        const float p =
            expf(m_ws[base_m + int64_t(s) * local_heads] - mhat);
        num += p * c_ws[(r * n_split + s) * total + i];
        den += p * l_ws[base_m + int64_t(s) * local_heads];
      }
    }
    c_out[r * total + i] = (den > 0.0f) ? num / den : 0.0f;
  }
}

// vout: out[r, h, d] = <c[r, h, :], W_uv[h][d, :]> over kv_lora, for the v
// output rows of every head. One WARP per output row: the lanes read the
// row's kv_lora bf16 as consecutive uint4s (512 contiguous bytes per warp
// instruction), multiply by the shared c row, and a shuffle tree sums the
// 32 partials. Blocks are kVoutRowsPerBlock warps of one head. The
// previous thread-per-row form streamed each row through one thread's
// L1 (34 us per layer for 4 MB); the reassociation is fp32 rounding
// (accepted 2026-09-02). kv_b interleaved: head h's W_uv rows are
// [h*(nope+v)+nope, (h+1)*(nope+v)).
constexpr int kVoutRowsPerBlock = 8;
constexpr int kVoutThreads = 32 * kVoutRowsPerBlock;

__global__ __launch_bounds__(kVoutThreads) void vout_gemm_kernel(
    const float* c, const uint16_t* kv_b, uint16_t* out, int local_heads,
    int nope, int v, int kv_lora) {
  dgpp_pdl_wait();
  const int64_t r = blockIdx.x;
  const int h = blockIdx.y;
  const int d0 = blockIdx.z * kVoutRowsPerBlock;
  const int head_rows = nope + v;
  __shared__ __align__(16) float cs[512];
  for (int cc = threadIdx.x; cc < kv_lora; cc += blockDim.x)
    cs[cc] = c[(r * local_heads + h) * kv_lora + cc];
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int d = d0 + warp;
  if (d >= v) return;
  const uint16_t* wuv =
      kv_b + (int64_t(h) * head_rows + nope) * kv_lora;
  const uint4* w4 = reinterpret_cast<const uint4*>(wuv + int64_t(d) * kv_lora);
  float acc = 0.0f;
  for (int u = lane; u < kv_lora / 8; u += 32) {
    const uint4 wv = w4[u];
    const uint32_t* w32 = reinterpret_cast<const uint32_t*>(&wv);
    const float4 cs4 = *reinterpret_cast<const float4*>(cs + u * 8);
    const float4 cs4b = *reinterpret_cast<const float4*>(cs + u * 8 + 4);
    const float2 w0 = bf16x2_to_float2(w32[0]);
    const float2 w1 = bf16x2_to_float2(w32[1]);
    const float2 w2 = bf16x2_to_float2(w32[2]);
    const float2 w3 = bf16x2_to_float2(w32[3]);
    acc += w0.x * cs4.x;
    acc += w0.y * cs4.y;
    acc += w1.x * cs4.z;
    acc += w1.y * cs4.w;
    acc += w2.x * cs4b.x;
    acc += w2.y * cs4b.y;
    acc += w3.x * cs4b.z;
    acc += w3.y * cs4b.w;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
  if (lane == 0) out[(r * local_heads + h) * v + d] = float_to_bf16_bits(acc);
}


// The row-batched decode form (2026-09-26): each warp loads its 1 KB
// W_uv row ONCE into registers (the lane's two uint4, sixteen values) and
// runs the tile's activation rows one after another, each the one-row
// kernel's chain over the staged c row — the lane / element order and the
// shuffle tree above — so every row's bits are the one-row kernel's. A
// block owns kVoutRowsPerBlock weight rows of one head for a tile of
// kVoutTileRows activation rows (grid z = the tiles): the one-row kernel's
// parallelism at a quarter of its W_uv traffic, c staged with float4
// loads, static shared memory as before.
constexpr int kVoutTileRows = 4;

__global__ __launch_bounds__(kVoutThreads) void vout_rows_kernel(
    const float* c, const uint16_t* kv_b, uint16_t* out, int rows,
    int local_heads, int nope, int v, int kv_lora) {
  dgpp_pdl_wait();
  const int h = blockIdx.y;
  const int d0 = blockIdx.x * kVoutRowsPerBlock;
  const int t0 = blockIdx.z * kVoutTileRows;
  const int nr = min(kVoutTileRows, rows - t0);
  const int head_rows = nope + v;
  __shared__ __align__(16) float cs[kVoutTileRows][512];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int d = d0 + warp;
  const bool live = d < v;
  const uint16_t* wuv = kv_b + (int64_t(h) * head_rows + nope) * kv_lora;
  const uint4* w4 = reinterpret_cast<const uint4*>(wuv + int64_t(d) * kv_lora);
  float wf[2][8];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int u = lane + 32 * i;
    uint4 wv = make_uint4(0u, 0u, 0u, 0u);
    if (live && u < kv_lora / 8) wv = w4[u];
    const uint32_t* w32 = reinterpret_cast<const uint32_t*>(&wv);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = bf16x2_to_float2(w32[j]);
      wf[i][2 * j] = f.x;
      wf[i][2 * j + 1] = f.y;
    }
  }
  const int vec_per_row = kv_lora / 4;
  for (int i = threadIdx.x; i < nr * vec_per_row; i += blockDim.x) {
    const int r = i / vec_per_row, cc = i - r * vec_per_row;
    reinterpret_cast<float4*>(cs[r])[cc] = reinterpret_cast<const float4*>(
        c + (int64_t(t0 + r) * local_heads + h) * kv_lora)[cc];
  }
  __syncthreads();
  if (!live) return;
  for (int r = 0; r < nr; ++r) {
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int u = lane + 32 * i;
      if (u < kv_lora / 8) {
        const float4 cs4 = *reinterpret_cast<const float4*>(&cs[r][u * 8]);
        const float4 cs4b = *reinterpret_cast<const float4*>(&cs[r][u * 8 + 4]);
        acc += wf[i][0] * cs4.x;
        acc += wf[i][1] * cs4.y;
        acc += wf[i][2] * cs4.z;
        acc += wf[i][3] * cs4.w;
        acc += wf[i][4] * cs4b.x;
        acc += wf[i][5] * cs4b.y;
        acc += wf[i][6] * cs4b.z;
        acc += wf[i][7] * cs4b.w;
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0)
      out[(int64_t(t0 + r) * local_heads + h) * v + d] = float_to_bf16_bits(acc);
  }
}

}  // namespace

// ---------------------------------------------------------------------
// The absorb and vout projections on tensor cores (the prefill path,
// 2026-09-05). Both are per-head GEMMs the warp kernels above ran as
// scalar dots — 56 and 67 ms per 2048-token prefill. Same tile plan as the
// MoE tensor-core kernel: 128 rows x 64 columns per block, 64-deep
// k-stages staged in shared memory, bf16 mma.sync m16n8k16 with fp32
// accumulation, eight warps of sixteen rows.
//   absorb: q_tilde[r,h,c] = sum_d q[r,h,d] W_uk[h][d][c] — B is W_uk read
//     transposed into the tile (k = d is W's row);
//   vout:   out[r,h,d] = sum_c c[r,h,c] W_uv[h][d][c] — c is fp32, carried
//     as a three-way bf16 split (c = hi + mid + lo, the full fp32 mantissa;
//     three mmas into one accumulator, so the products are exact and only
//     the fp32 summation order differs from the warp kernel's chain — a
//     two-way split left near-zero outputs formed by cancellation with
//     visible relative error), B is W_uv in its natural [d][c] layout.
// Tolerance-equal to the warp kernels (fp32 summation order); the rows
// path keeps them below 16 rows (decode).
namespace proj {
constexpr int BM = 128, BN = 64, BK = 64, BK_PAD = BK + 8, kThreads = 256;
// vout stages three A tiles (hi, mid, lo): a 32-deep k-stage keeps the
// static shared memory under 48 KB.
constexpr int VBK = 32, VBK_PAD = VBK + 8;
}  // namespace proj
// Rows at or above which the projections take the tensor-core kernels (the
// prefill tiles); the decode path passes tensor_cores=false (2026-09-13,
// its batch up to 16 rows): a batched row must be bitwise the same row
// run alone, and the tensor-core kernels are tolerance-equal, not bitwise,
// to the warp chain.
constexpr int kProjMmaMinRows = 16;

__global__ __launch_bounds__(proj::kThreads) void absorb_q_mma_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ kv_b,
    uint16_t* __restrict__ q_tilde, int rows, int local_heads, int nope, int v,
    int kv_lora, int rope) {
  dgpp_pdl_wait();
  using namespace proj;
  __shared__ __align__(16) uint16_t sA[BM][BK_PAD];
  __shared__ __align__(16) uint16_t sB[BN][BK_PAD];
  const int n0 = blockIdx.x * BN;
  const int h = blockIdx.y;
  const int m0 = blockIdx.z * BM;
  const int head_rows = nope + v;
  const int q_head = nope + rope;      // a head's q row
  const int out_head = kv_lora + rope; // a head's absorbed row
  const uint16_t* wuk = kv_b + int64_t(h) * head_rows * kv_lora;  // [nope][kv_lora]
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int r = lane / 4, cc = (lane % 4) * 2;
  // The rope tail: the first column block of every head copies its rows'.
  if (n0 == 0)
    for (int i = threadIdx.x; i < BM * rope; i += kThreads) {
      const int mm = i / rope, t = i % rope;
      if (m0 + mm < rows)
        q_tilde[(int64_t(m0 + mm) * local_heads + h) * out_head + kv_lora + t] =
            q[(int64_t(m0 + mm) * local_heads + h) * q_head + nope + t];
    }
  float acc[8][4];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  for (int k0 = 0; k0 < nope; k0 += BK) {
    // A: q rows (bf16, contiguous nope per (row, head)).
    for (int i = threadIdx.x; i < BM * (BK / 8); i += kThreads) {
      const int mm = i / (BK / 8), kq = (i % (BK / 8)) * 8;
      uint4 val = make_uint4(0, 0, 0, 0);
      const int gm = m0 + mm, gk = k0 + kq;
      if (gm < rows && gk + 8 <= nope)
        val = *reinterpret_cast<const uint4*>(q + (int64_t(gm) * local_heads + h) * q_head + gk);
      *reinterpret_cast<uint4*>(&sA[mm][kq]) = val;
    }
    // B transposed: sB[c][d] = W_uk[d][n0 + c]; read W row d along c.
    for (int i = threadIdx.x; i < BK * (BN / 8); i += kThreads) {
      const int dd = i / (BN / 8), c8 = (i % (BN / 8)) * 8;
      const int gd = k0 + dd, gc = n0 + c8;
      uint4 val = make_uint4(0, 0, 0, 0);
      if (gd < nope && gc + 8 <= kv_lora)
        val = *reinterpret_cast<const uint4*>(wuk + int64_t(gd) * kv_lora + gc);
      const uint16_t* e = reinterpret_cast<const uint16_t*>(&val);
#pragma unroll
      for (int t = 0; t < 8; ++t) sB[c8 + t][dd] = e[t];
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      const int ar = warp * 16 + r;
      const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sA[ar][kk + cc]);
      const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sA[ar + 8][kk + cc]);
      const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sA[ar][kk + cc + 8]);
      const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&sA[ar + 8][kk + cc + 8]);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int bn = j * 8 + r;
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sB[bn][kk + cc]);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sB[bn][kk + cc + 8]);
        mma_bf16_16816(acc[j], a0, a1, a2, a3, b0, b1);
      }
    }
    __syncthreads();
  }
  const int row_lo = m0 + warp * 16 + r;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int gc = n0 + j * 8 + cc;
    if (gc + 1 < kv_lora || gc < kv_lora) {
      if (row_lo < rows) {
        uint16_t* o = q_tilde + (int64_t(row_lo) * local_heads + h) * out_head + gc;
        o[0] = float_to_bf16_bits(acc[j][0]);
        if (gc + 1 < kv_lora) o[1] = float_to_bf16_bits(acc[j][1]);
      }
      if (row_lo + 8 < rows) {
        uint16_t* o = q_tilde + (int64_t(row_lo + 8) * local_heads + h) * out_head + gc;
        o[0] = float_to_bf16_bits(acc[j][2]);
        if (gc + 1 < kv_lora) o[1] = float_to_bf16_bits(acc[j][3]);
      }
    }
  }
}

__global__ __launch_bounds__(proj::kThreads) void vout_mma_kernel(
    const float* __restrict__ c, const uint16_t* __restrict__ kv_b,
    uint16_t* __restrict__ out, int rows, int local_heads, int nope, int v,
    int kv_lora) {
  dgpp_pdl_wait();
  using namespace proj;
  __shared__ __align__(16) uint16_t sHi[BM][VBK_PAD];
  __shared__ __align__(16) uint16_t sMid[BM][VBK_PAD];
  __shared__ __align__(16) uint16_t sLo[BM][VBK_PAD];
  __shared__ __align__(16) uint16_t sB[BN][VBK_PAD];
  const int n0 = blockIdx.x * BN;  // output columns d
  const int h = blockIdx.y;
  const int m0 = blockIdx.z * BM;
  const int head_rows = nope + v;
  const uint16_t* wuv = kv_b + (int64_t(h) * head_rows + nope) * kv_lora;  // [v][kv_lora]
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int r = lane / 4, cc = (lane % 4) * 2;
  float acc[8][4];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  for (int k0 = 0; k0 < kv_lora; k0 += VBK) {
    // A: c rows (fp32) split into bf16 hi + lo.
    for (int i = threadIdx.x; i < BM * (VBK / 4); i += kThreads) {
      const int mm = i / (VBK / 4), kq = (i % (VBK / 4)) * 4;
      const int gm = m0 + mm, gk = k0 + kq;
      float4 cv = make_float4(0.f, 0.f, 0.f, 0.f);
      if (gm < rows && gk + 4 <= kv_lora)
        cv = *reinterpret_cast<const float4*>(c + (int64_t(gm) * local_heads + h) * kv_lora + gk);
      const float f[4] = {cv.x, cv.y, cv.z, cv.w};
      uint16_t hi[4], mid[4], lo[4];
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        hi[t] = float_to_bf16_bits(f[t]);
        const float r1 = f[t] - bf16_bits_to_float(hi[t]);
        mid[t] = float_to_bf16_bits(r1);
        lo[t] = float_to_bf16_bits(r1 - bf16_bits_to_float(mid[t]));
      }
      *reinterpret_cast<uint2*>(&sHi[mm][kq]) =
          make_uint2(uint32_t(hi[0]) | (uint32_t(hi[1]) << 16), uint32_t(hi[2]) | (uint32_t(hi[3]) << 16));
      *reinterpret_cast<uint2*>(&sMid[mm][kq]) =
          make_uint2(uint32_t(mid[0]) | (uint32_t(mid[1]) << 16), uint32_t(mid[2]) | (uint32_t(mid[3]) << 16));
      *reinterpret_cast<uint2*>(&sLo[mm][kq]) =
          make_uint2(uint32_t(lo[0]) | (uint32_t(lo[1]) << 16), uint32_t(lo[2]) | (uint32_t(lo[3]) << 16));
    }
    // B: W_uv rows d, k = c contiguous.
    for (int i = threadIdx.x; i < BN * (VBK / 8); i += kThreads) {
      const int dd = i / (VBK / 8), kq = (i % (VBK / 8)) * 8;
      const int gd = n0 + dd, gk = k0 + kq;
      uint4 val = make_uint4(0, 0, 0, 0);
      if (gd < v && gk + 8 <= kv_lora)
        val = *reinterpret_cast<const uint4*>(wuv + int64_t(gd) * kv_lora + gk);
      *reinterpret_cast<uint4*>(&sB[dd][kq]) = val;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < VBK; kk += 16) {
      const int ar = warp * 16 + r;
      const uint32_t h0 = *reinterpret_cast<const uint32_t*>(&sHi[ar][kk + cc]);
      const uint32_t h1 = *reinterpret_cast<const uint32_t*>(&sHi[ar + 8][kk + cc]);
      const uint32_t h2 = *reinterpret_cast<const uint32_t*>(&sHi[ar][kk + cc + 8]);
      const uint32_t h3 = *reinterpret_cast<const uint32_t*>(&sHi[ar + 8][kk + cc + 8]);
      const uint32_t m0_ = *reinterpret_cast<const uint32_t*>(&sMid[ar][kk + cc]);
      const uint32_t m1_ = *reinterpret_cast<const uint32_t*>(&sMid[ar + 8][kk + cc]);
      const uint32_t m2_ = *reinterpret_cast<const uint32_t*>(&sMid[ar][kk + cc + 8]);
      const uint32_t m3_ = *reinterpret_cast<const uint32_t*>(&sMid[ar + 8][kk + cc + 8]);
      const uint32_t l0 = *reinterpret_cast<const uint32_t*>(&sLo[ar][kk + cc]);
      const uint32_t l1 = *reinterpret_cast<const uint32_t*>(&sLo[ar + 8][kk + cc]);
      const uint32_t l2 = *reinterpret_cast<const uint32_t*>(&sLo[ar][kk + cc + 8]);
      const uint32_t l3 = *reinterpret_cast<const uint32_t*>(&sLo[ar + 8][kk + cc + 8]);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int bn = j * 8 + r;
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sB[bn][kk + cc]);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sB[bn][kk + cc + 8]);
        // Small terms first: lo, mid, then hi (the classic order that keeps
        // the small parts from being absorbed by the large partial sums).
        mma_bf16_16816(acc[j], l0, l1, l2, l3, b0, b1);
        mma_bf16_16816(acc[j], m0_, m1_, m2_, m3_, b0, b1);
        mma_bf16_16816(acc[j], h0, h1, h2, h3, b0, b1);
      }
    }
    __syncthreads();
  }
  const int row_lo = m0 + warp * 16 + r;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int gd = n0 + j * 8 + cc;
    if (gd < v) {
      if (row_lo < rows) {
        uint16_t* o = out + (int64_t(row_lo) * local_heads + h) * v + gd;
        o[0] = float_to_bf16_bits(acc[j][0]);
        if (gd + 1 < v) o[1] = float_to_bf16_bits(acc[j][1]);
      }
      if (row_lo + 8 < rows) {
        uint16_t* o = out + (int64_t(row_lo + 8) * local_heads + h) * v + gd;
        o[0] = float_to_bf16_bits(acc[j][2]);
        if (gd + 1 < v) o[1] = float_to_bf16_bits(acc[j][3]);
      }
    }
  }
}

// ---------------------------------------------------------------------
// Shared-memory opt-in
// ---------------------------------------------------------------------

namespace {

// Populated by dsa_prepare_kernel_smem(); the launchers only read them.
int g_select_smem_cap = -1;
int g_attn_smem_cap = -1;

}  // namespace

void dsa_prepare_kernel_smem() {
  if (g_select_smem_cap < 0) {
    int cap = 0;
    DGPP_CUDA_OK(cudaDeviceGetAttribute(
        &cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    g_select_smem_cap = cap - 1024;  // leave room for static smem + alignment
    DGPP_CUDA_OK(cudaFuncSetAttribute(
        select_decode_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        g_select_smem_cap));
    DGPP_CUDA_OK(cudaFuncSetAttribute(
        select_decode_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        g_select_smem_cap));
    // The 64-head instantiations (DeepSeek-V4's indexer): the opt-in is per
    // kernel symbol, and four rows of 64 heads at select_k 512 pass 48 KB.
    DGPP_CUDA_OK(cudaFuncSetAttribute(
        select_decode_kernel<false, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        g_select_smem_cap));
    DGPP_CUDA_OK(cudaFuncSetAttribute(
        select_decode_kernel<true, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        g_select_smem_cap));
    // The prefill select at select_k 2048 (a 4096-key tile) needs ~56 KB.
    DGPP_CUDA_OK(cudaFuncSetAttribute(
        select_prefill_kernel<false, 2 * kSelectTile>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, g_select_smem_cap));
    DGPP_CUDA_OK(cudaFuncSetAttribute(
        select_prefill_kernel<true, 2 * kSelectTile>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, g_select_smem_cap));
  }
  if (g_attn_smem_cap < 0) {
    int cap = 0;
    DGPP_CUDA_OK(cudaDeviceGetAttribute(
        &cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    g_attn_smem_cap = cap - 1024;
    // Every format's instantiation: the opt-in is per kernel symbol.
    const auto opt_in = [&](auto tag) {
      constexpr LatentFormat F = decltype(tag)::value;
      DGPP_CUDA_OK(cudaFuncSetAttribute(
          attn_partial_kernel<F>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          g_attn_smem_cap));
      DGPP_CUDA_OK(cudaFuncSetAttribute(
          attn_flash_kernel<512, 512, false, F>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          int(dense::Geo<512, 512>::smem_bytes)));
      DGPP_CUDA_OK(cudaFuncSetAttribute(
          attn_flash_kernel<256, 256, false, F>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          int(dense::Geo<256, 256>::smem_bytes)));
      DGPP_CUDA_OK(cudaFuncSetAttribute(
          attn_flash_kernel<512, 576, false, F>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          int(dense::Geo<512, 576>::smem_bytes)));
      DGPP_CUDA_OK(cudaFuncSetAttribute(
          attn_flash_kernel<512, 512, true, F>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          int(dense::Geo<512, 512>::smem_bytes)));
      DGPP_CUDA_OK(cudaFuncSetAttribute(
          attn_flash_kernel<256, 256, true, F>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          int(dense::Geo<256, 256>::smem_bytes)));
      DGPP_CUDA_OK(cudaFuncSetAttribute(
          attn_flash_kernel<512, 576, true, F>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          int(dense::Geo<512, 576>::smem_bytes)));
    };
    opt_in(std::integral_constant<LatentFormat, LatentFormat::kBf16>{});
    opt_in(std::integral_constant<LatentFormat, LatentFormat::kFp8>{});
    opt_in(std::integral_constant<LatentFormat, LatentFormat::kFp4>{});
    opt_in(std::integral_constant<LatentFormat, LatentFormat::kFp8Block>{});
    opt_in(std::integral_constant<LatentFormat, LatentFormat::kFp4Block>{});
  }
}

// ---------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------

void dsa_fwht_quant_rows(const void* q_bf16, int64_t rows, void* q_fp8,
                         float* q_scale, cudaStream_t stream) {
  fwht_quant_rows_kernel<<<unsigned(rows), 128, 128 * sizeof(float), stream>>>(
      static_cast<const uint16_t*>(q_bf16),
      static_cast<uint8_t*>(q_fp8), q_scale);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_fold_weights(const float* weights, const float* q_scale, float* out,
                      int64_t n, float scale, cudaStream_t stream) {
  const int blocks = unsigned((n + 255) / 256);
  fold_weights_kernel<<<blocks, 256, 0, stream>>>(weights, q_scale, out, n,
                                                  scale);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_k_layernorm(const void* k_raw, int64_t k_stride, const void* w,
                     const void* b, void* k_out, int64_t rows, int dim,
                     float eps, cudaStream_t stream) {
  k_layernorm_kernel<<<unsigned(rows), 128, 0, stream>>>(
      static_cast<const uint16_t*>(k_raw), k_stride,
      static_cast<const uint16_t*>(w), static_cast<const uint16_t*>(b),
      static_cast<uint16_t*>(k_out), dim, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_fused_qkv_rmsnorm(const void* qkv, void* q_c, void* kv_c, int q_dim,
                           int kv_dim, int64_t rows, const void* q_w,
                           const void* kv_w, float eps, cudaStream_t stream,
                           int64_t row_stride) {
  if (row_stride <= 0) row_stride = q_dim + kv_dim;
  if (row_stride < q_dim + kv_dim)
    throw std::invalid_argument("dsa_fused_qkv_rmsnorm: row stride below q_dim + kv_dim");
  fused_qkv_rmsnorm_kernel<<<unsigned(rows), 256, 0, stream>>>(
      static_cast<const uint16_t*>(qkv), row_stride, static_cast<uint16_t*>(q_c),
      static_cast<uint16_t*>(kv_c), q_dim, kv_dim,
      static_cast<const uint16_t*>(q_w), static_cast<const uint16_t*>(kv_w),
      eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

// ---- interleaved RoPE ---------------------------------------------------

namespace {
// One thread per (row, head, pair): the three-rounding rotation of the
// header's contract, contraction-proof so the host table walk matches
// bit for bit.
__global__ void rope_interleave_kernel(
    const uint16_t* __restrict__ x, int64_t x_row_stride, int64_t x_head_stride,
    int heads, int half, const int64_t* __restrict__ pos,
    const uint16_t* __restrict__ table, int64_t table_positions,
    uint16_t* __restrict__ out, int64_t out_row_stride, int64_t out_head_stride,
    int64_t total) {
  dgpp_pdl_wait();
  const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int i = int(idx % half);
  const int64_t rh = idx / half;
  const int h = int(rh % heads);
  const int64_t r = rh / heads;
  const uint16_t* src = x + r * x_row_stride + int64_t(h) * x_head_stride + 2 * i;
  uint16_t* dst = out + r * out_row_stride + int64_t(h) * out_head_stride + 2 * i;
  const uint16_t x0b = src[0], x1b = src[1];
  int64_t p = pos[r];
  if (p < 0) {  // padding row: carried through unrotated
    dst[0] = x0b;
    dst[1] = x1b;
    return;
  }
  if (p >= table_positions) p = table_positions - 1;
  const float c = bf16_bits_to_float(table[(p * 2) * half + i]);
  const float s = bf16_bits_to_float(table[(p * 2 + 1) * half + i]);
  const float x0 = bf16_bits_to_float(x0b), x1 = bf16_bits_to_float(x1b);
  const float t1 = bf16_bits_to_float(float_to_bf16_bits(__fmul_rn(x0, c)));
  const float t2 = bf16_bits_to_float(float_to_bf16_bits(__fmul_rn(x1, s)));
  const float u1 = bf16_bits_to_float(float_to_bf16_bits(__fmul_rn(x1, c)));
  const float u2 = bf16_bits_to_float(float_to_bf16_bits(__fmul_rn(x0, s)));
  dst[0] = float_to_bf16_bits(__fsub_rn(t1, t2));
  dst[1] = float_to_bf16_bits(__fadd_rn(u1, u2));
}

// The same rotation over up to kRopeMaxSegments independent (x, out)
// segments in ONE launch (2026-10-13): a thread finds its segment by its
// global index against the segments' cumulative totals and then runs the
// kernel above's per-element arithmetic verbatim, so every element's bits
// are the single-segment launch's. The projection chain issues one of
// these after the GEMVs that feed the rotations instead of one launch
// per rotation (two to four a layer).
struct RopeSegDev {
  const uint16_t* x;
  uint16_t* out;
  int64_t x_row_stride, x_head_stride, out_row_stride, out_head_stride;
  int heads;
  int64_t total;  // rows * heads * half
};
struct RopeSegsDev {
  RopeSegDev seg[kRopeMaxSegments];
  int count;
};
__device__ __forceinline__ void rope_pair(const uint16_t* __restrict__ src, uint16_t* __restrict__ dst,
                                          int64_t p, int i, int half, const uint16_t* __restrict__ table,
                                          int64_t table_positions) {
  const uint16_t x0b = src[0], x1b = src[1];
  if (p < 0) {  // padding row: carried through unrotated
    dst[0] = x0b;
    dst[1] = x1b;
    return;
  }
  if (p >= table_positions) p = table_positions - 1;
  const float c = bf16_bits_to_float(table[(p * 2) * half + i]);
  const float s = bf16_bits_to_float(table[(p * 2 + 1) * half + i]);
  const float x0 = bf16_bits_to_float(x0b), x1 = bf16_bits_to_float(x1b);
  const float t1 = bf16_bits_to_float(float_to_bf16_bits(__fmul_rn(x0, c)));
  const float t2 = bf16_bits_to_float(float_to_bf16_bits(__fmul_rn(x1, s)));
  const float u1 = bf16_bits_to_float(float_to_bf16_bits(__fmul_rn(x1, c)));
  const float u2 = bf16_bits_to_float(float_to_bf16_bits(__fmul_rn(x0, s)));
  dst[0] = float_to_bf16_bits(__fsub_rn(t1, t2));
  dst[1] = float_to_bf16_bits(__fadd_rn(u1, u2));
}
__global__ void rope_interleave_multi_kernel(RopeSegsDev segs, int half, const int64_t* __restrict__ pos,
                                             const uint16_t* __restrict__ table, int64_t table_positions,
                                             int64_t total) {
  dgpp_pdl_wait();
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int s = 0;
  while (s + 1 < segs.count && idx >= segs.seg[s].total) {
    idx -= segs.seg[s].total;
    ++s;
  }
  const RopeSegDev& g = segs.seg[s];
  const int i = int(idx % half);
  const int64_t rh = idx / half;
  const int h = int(rh % g.heads);
  const int64_t r = rh / g.heads;
  rope_pair(g.x + r * g.x_row_stride + int64_t(h) * g.x_head_stride + 2 * i,
            g.out + r * g.out_row_stride + int64_t(h) * g.out_head_stride + 2 * i, pos[r], i, half, table,
            table_positions);
}
}  // namespace

void dsa_rope_interleave_multi(const DsaRopeSegment* segs, int count, int rope_dim, const int64_t* pos,
                               const void* table, int64_t table_positions, int64_t rows, cudaStream_t stream) {
  if (rows <= 0 || count <= 0) return;
  if (count > kRopeMaxSegments || !segs || !pos || !table || rope_dim <= 0 || rope_dim % 2 != 0 ||
      table_positions <= 0)
    throw std::invalid_argument("dsa_rope_interleave_multi: bad arguments");
  const int half = rope_dim / 2;
  RopeSegsDev d{};
  d.count = 0;
  int64_t total = 0;
  for (int s = 0; s < count; ++s) {
    if (!segs[s].x || !segs[s].out || segs[s].heads <= 0)
      throw std::invalid_argument("dsa_rope_interleave_multi: bad segment");
    RopeSegDev& g = d.seg[d.count++];
    g.x = static_cast<const uint16_t*>(segs[s].x);
    g.out = static_cast<uint16_t*>(segs[s].out);
    g.x_row_stride = segs[s].x_row_stride;
    g.x_head_stride = segs[s].x_head_stride;
    g.out_row_stride = segs[s].out_row_stride;
    g.out_head_stride = segs[s].out_head_stride;
    g.heads = segs[s].heads;
    g.total = rows * int64_t(segs[s].heads) * half;
    total += g.total;
  }
  const unsigned blocks = unsigned((total + 255) / 256);
  rope_interleave_multi_kernel<<<blocks, 256, 0, stream>>>(d, half, pos, static_cast<const uint16_t*>(table),
                                                           table_positions, total);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_rope_table_host(double theta, int rope_dim, int64_t positions,
                         uint16_t* out) {
  if (rope_dim <= 0 || rope_dim % 2 != 0 || positions <= 0 || out == nullptr)
    throw std::invalid_argument("dsa_rope_table_host: bad arguments");
  const int half = rope_dim / 2;
  // inv_freq = 1 / theta^(2i / rope_dim): the power in double, rounded to
  // fp32 (transformers' fp32 powf differs by at most an fp32 ulp; the
  // double form is what every host and the python reference reproduce
  // exactly — the reproducibility matters more than the last ulp of a
  // number the fp32 position product then rounds again).
  std::vector<float> inv(static_cast<size_t>(half), 0.0f);
  const float base = static_cast<float>(theta);
  for (int i = 0; i < half; ++i) {
    const float e = static_cast<float>(2 * i) / static_cast<float>(rope_dim);
    inv[size_t(i)] =
        static_cast<float>(1.0 / std::pow(static_cast<double>(base), static_cast<double>(e)));
  }
  for (int64_t p = 0; p < positions; ++p)
    for (int i = 0; i < half; ++i) {
      // The fp32 position product (the reference's), then the trig in
      // double rounded to fp32 and to bf16 — the same on every host and
      // in the python reference (np.float32(np.cos(np.float64(ang)))).
      const float ang = static_cast<float>(p) * inv[size_t(i)];
      out[(p * 2) * half + i] = float_to_bf16_bits(
          static_cast<float>(std::cos(static_cast<double>(ang))));
      out[(p * 2 + 1) * half + i] = float_to_bf16_bits(
          static_cast<float>(std::sin(static_cast<double>(ang))));
    }
}

void dsa_rope_interleave(const void* x, int64_t x_row_stride,
                         int64_t x_head_stride, int heads, int rope_dim,
                         const int64_t* pos, const void* table,
                         int64_t table_positions, void* out,
                         int64_t out_row_stride, int64_t out_head_stride,
                         int64_t rows, cudaStream_t stream) {
  if (rows <= 0 || heads <= 0) return;
  if (!x || !pos || !table || !out || rope_dim <= 0 || rope_dim % 2 != 0 ||
      table_positions <= 0)
    throw std::invalid_argument("dsa_rope_interleave: bad arguments");
  const int half = rope_dim / 2;
  const int64_t total = rows * int64_t(heads) * half;
  const unsigned blocks = unsigned((total + 255) / 256);
  rope_interleave_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const uint16_t*>(x), x_row_stride, x_head_stride, heads, half,
      pos, static_cast<const uint16_t*>(table), table_positions,
      static_cast<uint16_t*>(out), out_row_stride, out_head_stride, total);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_kpool_compress_write(const void* k, int64_t k_stride,
                              const void* gate, int64_t gate_stride,
                              const float* ape, const int32_t* block_table,
                              int pools_per_block, int64_t first_pool,
                              int n_pools, void* index_k, float* index_scale,
                              int kpool, int dim, cudaStream_t stream) {
  if (n_pools <= 0) return;
  kpool_compress_write_kernel<<<unsigned(n_pools), 128,
                                dim * sizeof(float), stream>>>(
      static_cast<const uint16_t*>(k), k_stride,
      static_cast<const uint16_t*>(gate), gate_stride, ape, block_table,
      pools_per_block, first_pool, n_pools, static_cast<uint8_t*>(index_k),
      index_scale, kpool, dim);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_kpool_tail_seed(const void* k, int64_t k_stride, const void* gate,
                         int64_t gate_stride, const int32_t* req_ids,
                         const int64_t* pos, int64_t tokens, void* tail,
                         int kpool, int dim, cudaStream_t stream) {
  if (tokens <= 0) return;
  kpool_tail_seed_kernel<<<unsigned(tokens), 128, 0, stream>>>(
      static_cast<const uint16_t*>(k), k_stride,
      static_cast<const uint16_t*>(gate), gate_stride, req_ids, pos, tokens,
      static_cast<uint16_t*>(tail), kpool, dim);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_kpool_decode_update(const void* k, int64_t k_stride,
                             const void* gate, int64_t gate_stride,
                             const float* ape, const int32_t* req_ids,
                             const int64_t* pos, const int32_t* req_spans,
                             int num_requests,
                             const int32_t* block_tables,
                             int blocks_per_request, void* tail,
                             void* index_k, float* index_scale,
                             int pools_per_block, int kpool, int dim,
                             cudaStream_t stream, void* tail_snapshots) {
  if (num_requests <= 0) return;
  kpool_decode_update_kernel<<<unsigned(num_requests), 128,
                               dim * sizeof(float), stream>>>(
      static_cast<const uint16_t*>(k), k_stride,
      static_cast<const uint16_t*>(gate), gate_stride, ape, req_ids, pos,
      req_spans, block_tables, blocks_per_request,
      static_cast<uint16_t*>(tail), static_cast<uint8_t*>(index_k),
      index_scale, pools_per_block, kpool, dim,
      static_cast<uint16_t*>(tail_snapshots));
  DGPP_CUDA_OK(cudaGetLastError());
}

__global__ void dsa_zero_padding_rows_kernel(const int64_t* __restrict__ pos,
                                             uint16_t* __restrict__ out,
                                             int hidden) {
  dgpp_pdl_wait();
  const int t = blockIdx.x;
  if (pos[t] >= 0) return;
  uint16_t* row = out + static_cast<size_t>(t) * hidden;
  for (int h = threadIdx.x; h < hidden; h += blockDim.x) row[h] = 0;
}

void dsa_zero_padding_rows(void* out, const int64_t* pos, int tokens,
                           int hidden, cudaStream_t stream) {
  if (tokens <= 0 || hidden <= 0) return;
  if (out == nullptr || pos == nullptr)
    throw std::invalid_argument("dsa_zero_padding_rows: null buffer");
  dsa_zero_padding_rows_kernel<<<unsigned(tokens), 256, 0, stream>>>(
      pos, static_cast<uint16_t*>(out), hidden);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_latent_append(const void* latent_rows, const int32_t* req_ids,
                       const int64_t* pos, int64_t tokens,
                       const int32_t* block_tables, int blocks_per_request,
                       int block_tokens, void* latent_cache, int kv_lora,
                       cudaStream_t stream, LatentFormat format,
                       float* latent_scale, const void* rope_rows, int rope_dim) {
  if (tokens <= 0) return;
  if (latent_format_has_row_scale(format) && latent_scale == nullptr)
    throw std::invalid_argument("dsa_latent_append: a quantized cache needs its scales");
  if (rope_dim < 0 || rope_dim % 8 != 0 || (rope_dim > 0 && rope_rows == nullptr) ||
      (rope_dim > 0 && kv_lora % 8 != 0))
    throw std::invalid_argument("dsa_latent_append: a rope tail needs its rows and widths in 8s");
  const size_t row_bytes = latent_cache_row_bytes(format, kv_lora, rope_dim);
  const uint16_t* rr = static_cast<const uint16_t*>(rope_rows);
  switch (format) {
    case LatentFormat::kBf16:
      if (kv_lora % 8 != 0) DGPP_CUDA_OK(cudaErrorInvalidValue);
      latent_append_kernel<<<unsigned(tokens), kLatentAppendThreads, 0, stream>>>(
          static_cast<const uint16_t*>(latent_rows), req_ids, pos, block_tables,
          blocks_per_request, block_tokens,
          static_cast<uint8_t*>(latent_cache), kv_lora, row_bytes, rr, rope_dim);
      break;
    case LatentFormat::kFp8:
      if (kv_lora % 8 != 0 || kv_lora > 8 * kLatentAppendThreads)
        DGPP_CUDA_OK(cudaErrorInvalidValue);
      latent_append_fp8_kernel<<<unsigned(tokens), kLatentAppendThreads, 0, stream>>>(
          static_cast<const uint16_t*>(latent_rows), req_ids, pos, block_tables,
          blocks_per_request, block_tokens, static_cast<uint8_t*>(latent_cache),
          latent_scale, kv_lora, row_bytes, rr, rope_dim);
      break;
    case LatentFormat::kFp4:
      if (kv_lora % kLatentFp4Block != 0 ||
          kv_lora > kLatentFp4Block * kLatentAppendThreads)
        DGPP_CUDA_OK(cudaErrorInvalidValue);
      latent_append_fp4_kernel<<<unsigned(tokens), kLatentAppendThreads, 0, stream>>>(
          static_cast<const uint16_t*>(latent_rows), req_ids, pos, block_tables,
          blocks_per_request, block_tokens, static_cast<uint8_t*>(latent_cache),
          latent_scale, kv_lora, row_bytes, rr, rope_dim);
      break;
    case LatentFormat::kFp8Block:
      if (kv_lora % kLatentFp8BlockGroup != 0 || kv_lora > 8 * kLatentAppendThreads)
        DGPP_CUDA_OK(cudaErrorInvalidValue);
      latent_append_fp8block_kernel<<<unsigned(tokens), kLatentAppendThreads, 0, stream>>>(
          static_cast<const uint16_t*>(latent_rows), req_ids, pos, block_tables,
          blocks_per_request, block_tokens, static_cast<uint8_t*>(latent_cache),
          kv_lora, row_bytes, rr, rope_dim);
      break;
    case LatentFormat::kFp4Block:
      if (kv_lora % kLatentFp4Block != 0 ||
          kv_lora > kLatentFp4Block * kLatentAppendThreads)
        DGPP_CUDA_OK(cudaErrorInvalidValue);
      latent_append_fp4block_kernel<<<unsigned(tokens), kLatentAppendThreads, 0, stream>>>(
          static_cast<const uint16_t*>(latent_rows), req_ids, pos, block_tables,
          blocks_per_request, block_tokens, static_cast<uint8_t*>(latent_cache),
          kv_lora, row_bytes, rr, rope_dim);
      break;
  }
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_decode_tail_fp8block(void* q, int64_t q_row_stride, int64_t q_head_stride, int q_heads, int q_rope_dim,
                              const int64_t* pos, const float* inv_freq, int64_t* slots, int ring_slots, int window,
                              int32_t* wlist, int32_t* wcounts, int64_t* entry_out, int ratio, const void* kv_rows,
                              const int32_t* req_ids, const int32_t* ring_table, void* ring, int kv_lora, int rows,
                              cudaStream_t stream) {
  if (rows <= 0) return;
  if (q_rope_dim <= 0 || q_rope_dim % 2 != 0) throw std::invalid_argument("dsa_decode_tail_fp8block: rope_dim must be even");
  const int q_half = q_rope_dim / 2;
  if (!q || !pos || !inv_freq || !slots || !wlist || !wcounts || !kv_rows || !req_ids || !ring_table || !ring)
    throw std::invalid_argument("dsa_decode_tail_fp8block: null buffer");
  if (entry_out != nullptr && ratio <= 0) throw std::invalid_argument("dsa_decode_tail_fp8block: a ratio for the entry positions");
  if (kv_lora % 32 != 0 || kv_lora > 1024) throw std::invalid_argument("dsa_decode_tail_fp8block: kv_lora a multiple of 32 up to 1024");
  const size_t cache_row_bytes = latent_row_bytes(LatentFormat::kFp8Block, kv_lora);
  decode_tail_fp8block_kernel<<<unsigned(rows), 256, 0, stream>>>(
      static_cast<uint16_t*>(q), q_row_stride, q_head_stride, q_heads, q_half, pos, inv_freq, slots, ring_slots, window,
      wlist, wcounts, entry_out, ratio, static_cast<const uint16_t*>(kv_rows), req_ids, ring_table,
      static_cast<uint8_t*>(ring), kv_lora, cache_row_bytes);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_gather_index_pools(const int32_t* block_table, int pools_per_block,
                            const void* index_k, const float* index_scale,
                            int64_t n_pools, void* out_k, float* out_scale,
                            int dim, cudaStream_t stream) {
  if (n_pools <= 0) return;
  gather_index_pools_kernel<<<unsigned(n_pools), 128, 0, stream>>>(
      block_table, pools_per_block, static_cast<const uint8_t*>(index_k),
      index_scale, n_pools, static_cast<uint8_t*>(out_k), out_scale, dim);
  DGPP_CUDA_OK(cudaGetLastError());
}

namespace {
constexpr int kSelectMaxRows = 8;      // rows per launch: the kernel's shared-memory bound
constexpr int kSelectLayoutRows = 96;  // rows the workspace is laid out for: the decode batch's cap (the engine's kDecodeRowsMax; 96 since 2026-10-23 for sixteen DSpark slots, 64 for twelve earlier that day, 32 from 2026-09-14 for six)
constexpr int kSelectDefaultGrid = 96;  // measured optimum (dsa_select_bench, 2026-09-06)
size_t select_align256(size_t b) { return (b + 255) / 256 * 256; }
size_t select_keys_bytes(int rows, int64_t pools) {
  return select_align256(size_t(rows) * size_t(pools) * 8);
}
size_t select_hist_bytes(int rows) {
  return select_align256(size_t(rows) * kSelectHistBins * 4);
}
}  // namespace

size_t dsa_select_workspace_bytes(int max_rows, int64_t max_pools) {
  if (max_rows <= 0 || max_rows > kSelectLayoutRows || max_pools <= 0)
    throw std::invalid_argument("dsa_select_workspace_bytes: rows in [1, 96], pools > 0");
  // Laid out for kSelectLayoutRows rows regardless of max_rows:
  // the histograms follow the keys at a FIXED offset the launcher derives
  // the same way, whatever row count a call brings. (They used to follow
  // the call's own rows of keys, so a one-row call — the sampled
  // fallback's eager verify or re-draft — put its histogram inside row 1's
  // keys, which every two-row replay writes: a garbage histogram, a
  // partial best[] fill, stale shared-memory pool ids expanded into token
  // ids, and the listed attention reading an unmapped page.)
  (void)max_rows;
  return select_keys_bytes(kSelectLayoutRows, max_pools) + select_hist_bytes(kSelectLayoutRows);
}

void dsa_select_decode(const void* q_fp8, const float* w_folded,
                       const int32_t* req_ids, const int64_t* pos, int rows,
                       const int32_t* block_tables, int blocks_per_request,
                       const void* index_k, const float* index_scale,
                       int pools_per_block, int heads, int dim, int select_k,
                       int kpool, int max_selected, int32_t* topk_out,
                       int32_t* out_counts, void* select_ws,
                       int64_t ws_max_pools, int32_t* counter_ws,
                       int grid_blocks, cudaStream_t stream, bool relu) {
  if (rows <= 0) return;
  if (heads != 32 && heads != 64) DGPP_CUDA_OK(cudaErrorInvalidValue);
  // The radix path has no tile; the expansion bounds select_k.
  if (select_k <= 0 || select_k > kSelectMaxK) DGPP_CUDA_OK(cudaErrorInvalidValue);
  if (rows > kSelectLayoutRows) DGPP_CUDA_OK(cudaErrorInvalidValue);
  if (select_ws == nullptr || ws_max_pools <= 0) DGPP_CUDA_OK(cudaErrorInvalidValue);
  if (reinterpret_cast<uintptr_t>(select_ws) % 256 != 0) DGPP_CUDA_OK(cudaErrorInvalidValue);
  dsa_prepare_kernel_smem();
  uint64_t* keys_ws = static_cast<uint64_t*>(select_ws);
  // The histograms at the workspace's fixed offset (dsa_select_workspace_bytes).
  int32_t* hist_ws = reinterpret_cast<int32_t*>(
      static_cast<char*>(select_ws) + select_keys_bytes(kSelectLayoutRows, ws_max_pools));
  // Row groups of at most kSelectMaxRows (2026-09-13, the decode batch's
  // cap lifted to 16): the kernel holds every row's fp8 q and weights in
  // shared memory (~4.2 KB per row; sixteen would exceed the block's
  // 99 KB). Every block scores a stripe of every row's own context — the
  // rows share nothing — so a group is the same work at any grouping; the
  // groups run in stream order through the same counters (each launch's
  // last block resets them). Up to eight rows, every single-stream shape,
  // is the one launch it always was.
  // The 64-head form scores on the tensor cores: one row's q as f16 and the
  // warps' key tiles in shared memory, whatever the rows of the launch.
  const int max_rows = kSelectMaxRows;
  for (int g = 0; g < rows; g += max_rows) {
    const int grows = std::min(max_rows, rows - g);
    const size_t score_smem = heads == 64 ? select_mma_q_bytes(heads) + kMmaSelKeyBytes
                                          : size_t(grows) * heads * 128 + size_t(256 / 32) * 4 * 32 * 4;
    const size_t smem = score_smem + size_t(grows) * heads * 4 + size_t(kSelectHistBins) * 4 +
                        size_t(select_k) * 8 + size_t(kSelectStopCandidates) * 8 +
                        (select_k + 1) * 4;
    if (smem > size_t(g_select_smem_cap)) DGPP_CUDA_OK(cudaErrorInvalidValue);
    // The tensor-core form's shared memory leaves one block per SM, and its
    // grid is (stripes, rows): a row's stripes are the grid's x.
    const int default_grid = heads == 64 ? kSelectDefaultGrid / 2 : kSelectDefaultGrid;
    const int want_blocks = std::max(grid_blocks > 0 ? grid_blocks : default_grid, grows);
    const int blocks = heads == 64 ? std::max(1, want_blocks / grows) : want_blocks;
    const int grid_y = heads == 64 ? grows : 1;
    int32_t* ghist = hist_ws + int64_t(g) * kSelectHistBins;
    select_counter_reset_kernel<<<4, 256, 0, stream>>>(counter_ws, ghist,
                                                       grows * kSelectHistBins);
    const auto launch = [&](auto tag, auto hg) {
      select_decode_kernel<decltype(tag)::value, decltype(hg)::value><<<dim3(blocks, grid_y), 256, smem, stream>>>(
          static_cast<const uint8_t*>(q_fp8) + int64_t(g) * heads * 128,
          w_folded + int64_t(g) * heads, req_ids + g, pos + g, grows,
          block_tables, blocks_per_request,
          static_cast<const uint8_t*>(index_k), index_scale, pools_per_block,
          heads, select_k, kpool, max_selected, topk_out + int64_t(g) * max_selected,
          out_counts + g, keys_ws + int64_t(g) * ws_max_pools, ws_max_pools, ghist,
          counter_ws);
    };
    if (heads == 64) {
      if (relu) launch(std::true_type{}, std::integral_constant<int, 2>{});
      else launch(std::false_type{}, std::integral_constant<int, 2>{});
    } else if (relu) {
      launch(std::true_type{}, std::integral_constant<int, 1>{});
    } else {
      launch(std::false_type{}, std::integral_constant<int, 1>{});
    }
    DGPP_CUDA_OK(cudaGetLastError());
  }
}

void dsa_select_debug_phases(uint64_t out[8]) {
  DGPP_CUDA_OK(cudaMemcpyFromSymbol(out, g_select_phase, sizeof(unsigned long long) * 8));
}

unsigned long long dsa_attn_anomalies(long long out[6], bool clear,
                                      cudaStream_t stream) {
  // Stream-ordered: a legacy-stream symbol copy synchronizes
  // with every blocking stream of the device — in a loopback world that
  // is the peer rank's stream, mid-replay and waiting on this rank.
  unsigned long long count = 0;
  DGPP_CUDA_OK(cudaMemcpyFromSymbolAsync(&count, g_attn_anomaly_count, sizeof(count), 0,
                                         cudaMemcpyDeviceToHost, stream));
  long long first[6] = {0, 0, 0, 0, 0, 0};
  DGPP_CUDA_OK(cudaMemcpyFromSymbolAsync(first, g_attn_anomaly, sizeof(first), 0,
                                         cudaMemcpyDeviceToHost, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  if (out != nullptr) for (int i = 0; i < 6; ++i) out[i] = first[i];
  if (clear && count != 0) {
    static const unsigned long long zero = 0;
    DGPP_CUDA_OK(cudaMemcpyToSymbolAsync(g_attn_anomaly_count, &zero, sizeof(zero), 0,
                                         cudaMemcpyHostToDevice, stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  }
  return count;
}

unsigned long long dsa_select_anomalies(long long out[6], bool clear,
                                        cudaStream_t stream) {
  unsigned long long count = 0;
  DGPP_CUDA_OK(cudaMemcpyFromSymbolAsync(&count, g_select_anomaly_count, sizeof(count), 0,
                                         cudaMemcpyDeviceToHost, stream));
  long long first[6] = {0, 0, 0, 0, 0, 0};
  DGPP_CUDA_OK(cudaMemcpyFromSymbolAsync(first, g_select_anomaly, sizeof(first), 0,
                                         cudaMemcpyDeviceToHost, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  if (out != nullptr) for (int i = 0; i < 6; ++i) out[i] = first[i];
  if (clear && count != 0) {
    static const unsigned long long zero = 0;
    DGPP_CUDA_OK(cudaMemcpyToSymbolAsync(g_select_anomaly_count, &zero, sizeof(zero), 0,
                                         cudaMemcpyHostToDevice, stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  }
  return count;
}

void dsa_prefill_score_keys(const uint8_t* q, const uint8_t* k,
                            const float* w_folded, const float* k_scale,
                            const int64_t* pos, int rows, int64_t pools,
                            int64_t stride, int select_k, int kpool,
                            uint64_t* keys, cudaStream_t stream, bool relu) {
  if (rows <= 0 || pools <= 0) return;
  if (stride < pools) throw std::invalid_argument("DSA score key stride");
  const unsigned blocks = unsigned(((pools + 127) / 128) * rows);
  if (relu)
    prefill_fused_score_keys_kernel<true><<<blocks, 128, 0, stream>>>(
        q, k, w_folded, k_scale, pos, rows, pools, stride, select_k, kpool, keys);
  else
    prefill_fused_score_keys_kernel<false><<<blocks, 128, 0, stream>>>(
        q, k, w_folded, k_scale, pos, rows, pools, stride, select_k, kpool, keys);
  DGPP_CUDA_OK(cudaGetLastError());
}

size_t dsa_select_prefill_workspace_bytes(int rows, int64_t dot_stride, int select_k) {
  // Partitioned calls have rows < 96 and a power-of-two partition count
  // bringing rows * parts to at most 190. Reserve 192 leaves, independent
  // of context. A leaf retains only select_k keys.
  return (size_t(rows) * size_t(dot_stride) + size_t(192) * size_t(select_k)) * sizeof(uint64_t);
}

void dsa_select_prefill_keys(const uint64_t* keys, int64_t stride,
                             const int64_t* pos, int rows, int64_t pools,
                             int select_k, int kpool, int max_selected,
                             int32_t* topk_out, int32_t* out_counts,
                             uint64_t* partials, cudaStream_t stream) {
  if (rows <= 0) return;
  if (select_k <= 0 || select_k > kSelectMaxK || stride < pools)
    DGPP_CUDA_OK(cudaErrorInvalidValue);
  int parts = 1;
  while (parts * rows < 96 && 2 * parts <= pools / (2 * select_k)) parts *= 2;
  const size_t bytes = (size_t(select_k) * 3 + kSelectHistBins +
                        kSelectStopCandidates + 1) * sizeof(uint32_t);
  prefill_select_keys_kernel<<<dim3(unsigned(rows), unsigned(parts)), 256, bytes, stream>>>(
      keys, stride, pos, pools, select_k, kpool, max_selected,
      topk_out, out_counts, parts, partials);
  const size_t merge_bytes = (size_t(5) * select_k + 1) * sizeof(uint32_t);
  for (int step = 1; step < parts; step *= 2)
    prefill_merge_keys_kernel<<<dim3(unsigned(rows), unsigned(parts / (2 * step))),
                               256, merge_bytes, stream>>>(
        partials, pos, parts, step, select_k, kpool, max_selected, topk_out, out_counts);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_select_prefill(const float* dot, int64_t dot_stride,
                        const float* w_folded, const float* k_scale,
                        const int64_t* pos, int rows, int64_t n_pools,
                        int heads, int select_k, int kpool, int max_selected,
                        int32_t* topk_out, int32_t* out_counts,
                        cudaStream_t stream, bool relu, void* workspace) {
  if (rows <= 0) return;
  if (heads != 32) DGPP_CUDA_OK(cudaErrorInvalidValue);
  if (select_k <= 0 || select_k > kSelectMaxK) DGPP_CUDA_OK(cudaErrorInvalidValue);
  if (workspace && n_pools > kSelectTile) {
    auto* keys = static_cast<uint64_t*>(workspace);
    auto* partials = keys + int64_t(rows) * dot_stride;
    const dim3 grid(unsigned((n_pools + 255) / 256), unsigned(rows));
    if (relu)
      prefill_score_keys_kernel<true><<<grid, 256, 0, stream>>>(
          dot, dot_stride, w_folded, k_scale, pos, n_pools, select_k, kpool, keys);
    else
      prefill_score_keys_kernel<false><<<grid, 256, 0, stream>>>(
          dot, dot_stride, w_folded, k_scale, pos, n_pools, select_k, kpool, keys);
    dsa_select_prefill_keys(keys, dot_stride, pos, rows, n_pools, select_k,
                            kpool, max_selected, topk_out, out_counts, partials, stream);
    return;
  }
  // The tile: 2 * select_k keys or more (kSelectTile up to 1024, twice
  // that for the full model's 2048).
  const bool wide = select_k > kSelectTile / 2;
  const int tile = wide ? 2 * kSelectTile : kSelectTile;
  const size_t smem = size_t(select_k) * 8 + size_t(tile) * 8 + (select_k + 1) * 4;
  if (wide) {
    dsa_prepare_kernel_smem();
    if (smem > size_t(g_select_smem_cap)) DGPP_CUDA_OK(cudaErrorInvalidValue);
  }
  const auto launch = [&](auto relu_tag, auto tile_tag) {
    select_prefill_kernel<decltype(relu_tag)::value, decltype(tile_tag)::value>
        <<<unsigned(rows), 256, smem, stream>>>(
            dot, dot_stride, w_folded, k_scale, pos, rows, n_pools, heads, select_k,
            kpool, max_selected, topk_out, out_counts);
  };
  if (relu) {
    if (wide) launch(std::true_type{}, std::integral_constant<int, 2 * kSelectTile>{});
    else launch(std::true_type{}, std::integral_constant<int, kSelectTile>{});
  } else {
    if (wide) launch(std::false_type{}, std::integral_constant<int, 2 * kSelectTile>{});
    else launch(std::false_type{}, std::integral_constant<int, kSelectTile>{});
  }
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_absorb_q(const void* q, const void* kv_b, void* q_tilde,
                  int64_t rows, int local_heads, int nope, int v, int kv_lora,
                  cudaStream_t stream, int rope, bool tensor_cores) {
  if (rows <= 0) return;
  // kv_lora % 8: uint4 weight streaming (8 bf16 per load); 512 % kv_lora:
  // static q smem + column mapping; the rope tail in 8s (uint4 q rows).
  if (nope > 256 || kv_lora % 8 != 0 || 512 % kv_lora != 0 || rope < 0 ||
      rope % 8 != 0)
    DGPP_CUDA_OK(cudaErrorInvalidValue);
  if (tensor_cores && rows >= kProjMmaMinRows && nope % 16 == 0 && kv_lora % 8 == 0 &&
      reinterpret_cast<uintptr_t>(q) % 16 == 0 &&
      (int64_t(local_heads) * (nope + rope)) % 8 == 0) {
    dim3 grid{unsigned((kv_lora + proj::BN - 1) / proj::BN), unsigned(local_heads),
              unsigned((rows + proj::BM - 1) / proj::BM)};
    absorb_q_mma_kernel<<<grid, proj::kThreads, 0, stream>>>(
        static_cast<const uint16_t*>(q), static_cast<const uint16_t*>(kv_b),
        static_cast<uint16_t*>(q_tilde), int(rows), local_heads, nope, v, kv_lora,
        rope);
    DGPP_CUDA_OK(cudaGetLastError());
    return;
  }
  if (rows == 1) {
    dim3 grid{unsigned(rows), unsigned(local_heads)};
    absorb_q_kernel<<<grid, kAbsorbThreads, 0, stream>>>(
        static_cast<const uint16_t*>(q), static_cast<const uint16_t*>(kv_b),
        static_cast<uint16_t*>(q_tilde), local_heads, nope, v, kv_lora, rope);
    DGPP_CUDA_OK(cudaGetLastError());
    return;
  }
  // The row-batched form: tiles of up to four rows per block, the exact
  // tile for two or three rows (a partial last tile only wastes FMAs).
  const auto launch = [&](auto tile) {
    constexpr int kTile = decltype(tile)::value;
    dim3 grid{unsigned((rows + kTile - 1) / kTile), unsigned(local_heads)};
    absorb_q_rows_kernel<kTile><<<grid, kAbsorbThreads, 0, stream>>>(
        static_cast<const uint16_t*>(q), static_cast<const uint16_t*>(kv_b),
        static_cast<uint16_t*>(q_tilde), int(rows), local_heads, nope, v, kv_lora, rope);
  };
  switch (rows) {
    case 2: launch(std::integral_constant<int, 2>{}); break;
    case 3: launch(std::integral_constant<int, 3>{}); break;
    default: launch(std::integral_constant<int, 4>{}); break;
  }
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_attn_partial(const void* q_tilde, const void* latent_cache,
                      const int32_t* req_ids, const int32_t* topk,
                      int topk_stride, const int32_t* counts, int rows,
                      int n_split, int local_heads, int kv_lora,
                      int block_tokens, const int32_t* block_tables,
                      int blocks_per_request, float scale, float* m_ws,
                      float* l_ws, float* c_ws, cudaStream_t stream,
                      LatentFormat format, const float* latent_scale, int rope) {
  if (rows <= 0) return;
  if (latent_format_has_row_scale(format) && latent_scale == nullptr)
    throw std::invalid_argument("dsa_attn_partial: a quantized cache needs its scales");
  if (rope < 0 || rope % 8 != 0 || rope > kAttnMaxRopeSlice * 128)
    DGPP_CUDA_OK(cudaErrorInvalidValue);
  // hpb in {1,2,4,8,16}: local_heads must divide across head-group blocks,
  // and blockDim(128)/hpb gives the per-head dim groups (the butterfly
  // reduce handles any power-of-two group count up to blockDim). kv_lora
  // must give every group a nonempty 8-multiple slice: kv_lora >= groups
  // and kv_lora % groups must keep the vec path's 8-wide windows aligned.
  // Heads per block: 16 unless a thread's dim window would exceed the
  // register window (kv_lora / (128 / hpb) <= kAttnMaxDslice).
  int hpb = local_heads < 16 ? local_heads : 16;
  while (hpb > 1 && kv_lora / (128 / hpb) > kAttnMaxDslice) hpb /= 2;
  if (local_heads % hpb != 0 || 128 % hpb != 0)
    DGPP_CUDA_OK(cudaErrorInvalidValue);
  if (kv_lora % 8 != 0) DGPP_CUDA_OK(cudaErrorInvalidValue);
  if (kv_lora < 128 / hpb) DGPP_CUDA_OK(cudaErrorInvalidValue);
  if (kv_lora / (128 / hpb) > kAttnMaxDslice) DGPP_CUDA_OK(cudaErrorInvalidValue);
  // Padded strides (see the kernel's bank-conflict note): must mirror the
  // kernel's gstride/row_stride arithmetic exactly.
  const int groups = 128 / hpb;
  const int dslice = kv_lora / groups;
  const int gstride = dslice + 8;
  const int row_stride = groups * gstride + 8 + (rope > 0 ? rope + 8 : 0);
  // A thread's rope slice (ceil(rope / groups)) must fit its registers.
  if (rope > 0 && (rope + groups - 1) / groups > kAttnMaxRopeSlice)
    DGPP_CUDA_OK(cudaErrorInvalidValue);
  const size_t smem = (kAttnTile * row_stride * 2 +
                        hpb * (kAttnTile + 1) * 4 + hpb * 4 + 15) &
                      ~size_t(15);
  dsa_prepare_kernel_smem();
  if (smem > size_t(g_attn_smem_cap)) DGPP_CUDA_OK(cudaErrorInvalidValue);
  dim3 grid{unsigned(rows), unsigned(n_split),
            unsigned(local_heads / hpb)};
  const auto launch = [&](auto tag) {
    constexpr LatentFormat F = decltype(tag)::value;
    attn_partial_kernel<F><<<grid, 128, smem, stream>>>(
        static_cast<const uint16_t*>(q_tilde),
        static_cast<const uint8_t*>(latent_cache), latent_scale, req_ids, topk,
        topk_stride, counts, n_split, local_heads, kv_lora, rope, block_tokens,
        block_tables, blocks_per_request, scale, m_ws, l_ws, c_ws);
  };
  switch (format) {
    case LatentFormat::kBf16:
      launch(std::integral_constant<LatentFormat, LatentFormat::kBf16>{});
      break;
    case LatentFormat::kFp8:
      launch(std::integral_constant<LatentFormat, LatentFormat::kFp8>{});
      break;
    case LatentFormat::kFp4:
      launch(std::integral_constant<LatentFormat, LatentFormat::kFp4>{});
      break;
    case LatentFormat::kFp8Block:
      launch(std::integral_constant<LatentFormat, LatentFormat::kFp8Block>{});
      break;
    case LatentFormat::kFp4Block:
      launch(std::integral_constant<LatentFormat, LatentFormat::kFp4Block>{});
      break;
  }
  DGPP_CUDA_OK(cudaGetLastError());
}

namespace {
template <int KV, int SD, bool kListed, LatentFormat F, int kGroup>
void launch_attn_flash_variant(dim3 grid, const void* q_tilde,
                               const void* latent_cache, const float* latent_scale,
                               const int32_t* req_ids, const int64_t* pos,
                               const int32_t* topk, int topk_stride,
                               const int32_t* counts, int rows, int n_split,
                               int local_heads, int block_tokens,
                               const int32_t* block_tables, int blocks_per_request,
                               int blocks_total, float scale, float* m_ws, float* l_ws, float* c_ws,
                               int32_t* tickets, float* mg_ws, float* lg_ws, float* cg_ws,
                               cudaStream_t stream) {
  constexpr size_t kSmem = dense::Geo<KV, SD, kListed ? 16 : dense::kTile>::smem_bytes;
  attn_flash_kernel<KV, SD, kListed, F, kGroup><<<grid, dense::kThreads, kSmem, stream>>>(
      static_cast<const uint16_t*>(q_tilde),
      static_cast<const uint8_t*>(latent_cache), latent_scale, req_ids, pos, topk,
      topk_stride, counts, rows, n_split, local_heads, block_tokens, block_tables,
      blocks_per_request, blocks_total, scale, m_ws, l_ws, c_ws, tickets, mg_ws, lg_ws, cg_ws);
}

template <int KV, int SD, bool kListed, int kGroup>
void launch_attn_flash_format(LatentFormat format, dim3 grid, const void* q_tilde,
                              const void* latent_cache, const float* latent_scale,
                              const int32_t* req_ids, const int64_t* pos,
                              const int32_t* topk, int topk_stride,
                              const int32_t* counts, int rows, int n_split,
                              int local_heads, int block_tokens,
                              const int32_t* block_tables, int blocks_per_request,
                              int blocks_total, float scale, float* m_ws, float* l_ws, float* c_ws,
                              int32_t* tickets, float* mg_ws, float* lg_ws, float* cg_ws,
                              cudaStream_t stream) {
  switch (format) {
    case LatentFormat::kBf16:
      launch_attn_flash_variant<KV, SD, kListed, LatentFormat::kBf16, kGroup>(
          grid, q_tilde, latent_cache, latent_scale, req_ids, pos, topk, topk_stride,
          counts, rows, n_split, local_heads, block_tokens, block_tables,
          blocks_per_request, blocks_total, scale, m_ws, l_ws, c_ws, tickets, mg_ws, lg_ws, cg_ws, stream);
      break;
    case LatentFormat::kFp8:
      launch_attn_flash_variant<KV, SD, kListed, LatentFormat::kFp8, kGroup>(
          grid, q_tilde, latent_cache, latent_scale, req_ids, pos, topk, topk_stride,
          counts, rows, n_split, local_heads, block_tokens, block_tables,
          blocks_per_request, blocks_total, scale, m_ws, l_ws, c_ws, tickets, mg_ws, lg_ws, cg_ws, stream);
      break;
    case LatentFormat::kFp4:
      launch_attn_flash_variant<KV, SD, kListed, LatentFormat::kFp4, kGroup>(
          grid, q_tilde, latent_cache, latent_scale, req_ids, pos, topk, topk_stride,
          counts, rows, n_split, local_heads, block_tokens, block_tables,
          blocks_per_request, blocks_total, scale, m_ws, l_ws, c_ws, tickets, mg_ws, lg_ws, cg_ws, stream);
      break;
    case LatentFormat::kFp8Block:
      launch_attn_flash_variant<KV, SD, kListed, LatentFormat::kFp8Block, kGroup>(
          grid, q_tilde, latent_cache, latent_scale, req_ids, pos, topk, topk_stride,
          counts, rows, n_split, local_heads, block_tokens, block_tables,
          blocks_per_request, blocks_total, scale, m_ws, l_ws, c_ws, tickets, mg_ws, lg_ws, cg_ws, stream);
      break;
    case LatentFormat::kFp4Block:
      launch_attn_flash_variant<KV, SD, kListed, LatentFormat::kFp4Block, kGroup>(
          grid, q_tilde, latent_cache, latent_scale, req_ids, pos, topk, topk_stride,
          counts, rows, n_split, local_heads, block_tokens, block_tables,
          blocks_per_request, blocks_total, scale, m_ws, l_ws, c_ws, tickets, mg_ws, lg_ws, cg_ws, stream);
      break;
  }
}

template <bool kListed>
bool launch_attn_flash(const void* q_tilde, const void* latent_cache,
                       const int32_t* req_ids, const int64_t* pos,
                       const int32_t* topk, int topk_stride, const int32_t* counts,
                       int rows, int n_split, int local_heads, int kv_lora,
                       int block_tokens, const int32_t* block_tables,
                       int blocks_per_request, float scale, float* m_ws,
                       float* l_ws, float* c_ws, cudaStream_t stream,
                       LatentFormat format, const float* latent_scale, int rope,
                       int blocks_total, int group, int32_t* tickets, float* mg_ws, float* lg_ws,
                       float* cg_ws) {
  if (rows <= 0) return true;
  if (blocks_total < 0) blocks_total = blocks_per_request;
  // The split groups (listed form only): 1, or kAttnSplitGroup dividing
  // n_split with the tickets and the group partials' buffers.
  if (group != 1 && (!kListed || group != kAttnSplitGroup || n_split % kAttnSplitGroup != 0 || tickets == nullptr ||
                     mg_ws == nullptr || lg_ws == nullptr || cg_ws == nullptr))
    throw std::invalid_argument(
        "dsa attention: the split group must be 1, or kAttnSplitGroup dividing n_split with tickets and group buffers (listed form)");
  // The compiled geometries: 512 and 256 without a tail, 512 + 64 with one.
  if (rope == 0 && kv_lora != 512 && kv_lora != 256) return false;  // caller falls back
  if (rope != 0 && !(kv_lora == 512 && rope == 64)) return false;
  if (kListed && (local_heads < 16 || local_heads % 16 != 0)) return false;
  if (local_heads <= 0 || n_split <= 0 || block_tokens <= 0)
    DGPP_CUDA_OK(cudaErrorInvalidValue);
  if (latent_format_has_row_scale(format) && latent_scale == nullptr)
    throw std::invalid_argument("dsa attention: a quantized cache needs its scales");
  dsa_prepare_kernel_smem();
  const int total_m = rows * local_heads;
  dim3 grid{unsigned((total_m + dense::kM - 1) / dense::kM), unsigned(n_split)};
  const auto dispatch = [&](auto group_tag) {
    constexpr int kG = decltype(group_tag)::value;
    if (rope == 64) {
      launch_attn_flash_format<512, 576, kListed, kG>(
          format, grid, q_tilde, latent_cache, latent_scale, req_ids, pos, topk,
          topk_stride, counts, rows, n_split, local_heads, block_tokens, block_tables,
          blocks_per_request, blocks_total, scale, m_ws, l_ws, c_ws, tickets, mg_ws, lg_ws, cg_ws, stream);
    } else if (kv_lora == 512) {
      launch_attn_flash_format<512, 512, kListed, kG>(
          format, grid, q_tilde, latent_cache, latent_scale, req_ids, pos, topk,
          topk_stride, counts, rows, n_split, local_heads, block_tokens, block_tables,
          blocks_per_request, blocks_total, scale, m_ws, l_ws, c_ws, tickets, mg_ws, lg_ws, cg_ws, stream);
    } else {
      launch_attn_flash_format<256, 256, kListed, kG>(
          format, grid, q_tilde, latent_cache, latent_scale, req_ids, pos, topk,
          topk_stride, counts, rows, n_split, local_heads, block_tokens, block_tables,
          blocks_per_request, blocks_total, scale, m_ws, l_ws, c_ws, tickets, mg_ws, lg_ws, cg_ws, stream);
    }
  };
  if constexpr (kListed) {
    if (group == kAttnSplitGroup)
      dispatch(std::integral_constant<int, kAttnSplitGroup>{});
    else
      dispatch(std::integral_constant<int, 1>{});
  } else {
    dispatch(std::integral_constant<int, 1>{});
  }
  DGPP_CUDA_OK(cudaGetLastError());
  return true;
}
}  // namespace

bool dsa_attn_dense(const void* q_tilde, const void* latent_cache,
                    const int32_t* req_ids, const int64_t* pos, int rows,
                    int n_split, int local_heads, int kv_lora, int block_tokens,
                    const int32_t* block_tables, int blocks_per_request,
                    float scale, float* m_ws, float* l_ws, float* c_ws,
                    cudaStream_t stream, LatentFormat format,
                    const float* latent_scale, int rope) {
  return launch_attn_flash<false>(q_tilde, latent_cache, req_ids, pos, nullptr, 0,
                                  nullptr, rows, n_split, local_heads, kv_lora,
                                  block_tokens, block_tables, blocks_per_request,
                                  scale, m_ws, l_ws, c_ws, stream, format,
                                  latent_scale, rope, -1, 1, nullptr, nullptr, nullptr, nullptr);
}

bool dsa_attn_listed(const void* q_tilde, const void* latent_cache,
                     const int32_t* req_ids, const int32_t* topk, int topk_stride,
                     const int32_t* counts, int rows, int n_split, int local_heads,
                     int kv_lora, int block_tokens, const int32_t* block_tables,
                     int blocks_per_request, float scale, float* m_ws, float* l_ws,
                     float* c_ws, cudaStream_t stream, LatentFormat format,
                     const float* latent_scale, int rope, int blocks_total, int group, int32_t* tickets,
                     float* mg_ws, float* lg_ws, float* cg_ws) {
  return launch_attn_flash<true>(q_tilde, latent_cache, req_ids, nullptr, topk,
                                 topk_stride, counts, rows, n_split, local_heads,
                                 kv_lora, block_tokens, block_tables,
                                 blocks_per_request, scale, m_ws, l_ws, c_ws, stream,
                                 format, latent_scale, rope, blocks_total, group, tickets, mg_ws, lg_ws, cg_ws);
}

void dsa_attn_combine(const float* m_ws, const float* l_ws, const float* c_ws,
                      int rows, int n_split, int local_heads, int kv_lora,
                      float* c_out, cudaStream_t stream) {
  if (rows <= 0) return;
  // One block per (row, head): 64+ blocks instead of `rows`.
  dim3 grid{unsigned(rows), unsigned(local_heads)};
  attn_combine_kernel<<<grid, 128, 0, stream>>>(
      m_ws, l_ws, c_ws, n_split, local_heads, kv_lora, c_out);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_vout_gemm(const void* c, const void* kv_b, void* out,
                   int64_t rows, int local_heads, int nope, int v,
                   int kv_lora, cudaStream_t stream, bool tensor_cores) {
  if (rows <= 0) return;
  // kv_lora % 8: uint4 weight streaming; <= 512: static c smem.
  if (kv_lora > 512 || kv_lora % 8 != 0)
    DGPP_CUDA_OK(cudaErrorInvalidValue);
  if (tensor_cores && rows >= kProjMmaMinRows && kv_lora % 16 == 0 &&
      reinterpret_cast<uintptr_t>(c) % 16 == 0) {
    dim3 grid{unsigned((v + proj::BN - 1) / proj::BN), unsigned(local_heads),
              unsigned((rows + proj::BM - 1) / proj::BM)};
    vout_mma_kernel<<<grid, proj::kThreads, 0, stream>>>(
        static_cast<const float*>(c), static_cast<const uint16_t*>(kv_b),
        static_cast<uint16_t*>(out), int(rows), local_heads, nope, v, kv_lora);
    DGPP_CUDA_OK(cudaGetLastError());
    return;
  }
  if (rows == 1) {
    dim3 grid{unsigned(rows), unsigned(local_heads),
              unsigned((v + kVoutRowsPerBlock - 1) / kVoutRowsPerBlock)};
    vout_gemm_kernel<<<grid, kVoutThreads, 0, stream>>>(
        static_cast<const float*>(c), static_cast<const uint16_t*>(kv_b),
        static_cast<uint16_t*>(out), local_heads, nope, v, kv_lora);
    DGPP_CUDA_OK(cudaGetLastError());
    return;
  }
  // The row-batched form (float4-staged c rows: c is the combine's fp32
  // output, 16-byte aligned; kv_lora % 8 == 0 keeps the vectors whole).
  if (reinterpret_cast<uintptr_t>(c) % 16 != 0) DGPP_CUDA_OK(cudaErrorInvalidValue);
  dim3 grid{unsigned((v + kVoutRowsPerBlock - 1) / kVoutRowsPerBlock), unsigned(local_heads),
            unsigned((rows + kVoutTileRows - 1) / kVoutTileRows)};
  vout_rows_kernel<<<grid, kVoutThreads, 0, stream>>>(
      static_cast<const float*>(c), static_cast<const uint16_t*>(kv_b),
      static_cast<uint16_t*>(out), int(rows), local_heads, nope, v, kv_lora);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
