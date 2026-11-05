#include <stdexcept>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "common/cuda_check.hpp"
#include "kernels/minimax_native_attention.cuh"
#include "kernels/minimax_native_attention.hpp"
#include "kernels/pdl.cuh"

namespace dgpp {
namespace {
constexpr int kThreads = 128, kStride = 144;
constexpr int kCacheBytes = 128 * kStride, kQueryBytes = 16 * kStride;
constexpr int kAttentionSmem = kCacheBytes + kQueryBytes + 16 * 128 * sizeof(float);

__device__ __forceinline__ float bf16(uint16_t value) {
  return __uint_as_float(static_cast<uint32_t>(value) << 16);
}
__device__ __forceinline__ uint32_t pack8(const float* x) {
  uint32_t packed;
  asm volatile(
      "{ .reg .b8 a,b,c,d;"
      "cvt.rn.satfinite.e2m1x2.f32 a, %2, %1;"
      "cvt.rn.satfinite.e2m1x2.f32 b, %4, %3;"
      "cvt.rn.satfinite.e2m1x2.f32 c, %6, %5;"
      "cvt.rn.satfinite.e2m1x2.f32 d, %8, %7;"
      "mov.b32 %0, {a,b,c,d}; }"
      : "=r"(packed)
      : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
  return packed;
}
__global__ void minimax_query_fp8_kernel(const uint16_t* input, uint8_t* output, int64_t count) {
  dgpp_pdl_wait();
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) output[i] = __nv_fp8_e4m3(bf16(input[i])).__x;
}

__global__ void minimax_cache_nvfp4_kernel(const uint16_t* input, uint8_t* cache,
                                           const int32_t* req, const int64_t* positions,
                                           const int32_t* table, int stride, int rows, int heads) {
  dgpp_pdl_wait();
  const int group = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = group / (heads * 8), head = (group / 8) % heads, sub = group % 8;
  if (row >= rows) return;
  const int64_t position = positions[row];
  if (position < 0) return;
  const int64_t physical =
      static_cast<int64_t>(table[static_cast<int64_t>(req[row]) * stride + position / 128]) * 128 +
      position % 128;
  const uint16_t* source = input + (static_cast<int64_t>(row) * heads + head) * 128 + sub * 16;
  float values[16], maximum = 0;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    values[i] = bf16(source[i]);
    maximum = fmaxf(maximum, fabsf(values[i]));
  }
  const __nv_fp8_e4m3 scale(maximum / 6.f);
  const float inverse = maximum != 0 ? 1.f / static_cast<float>(scale) : 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) values[i] *= inverse;
  uint8_t* dest = cache + (physical * heads + head) * kMinimaxNvfp4HeadBytes;
  *reinterpret_cast<uint2*>(dest + sub * 8) = make_uint2(pack8(values), pack8(values + 8));
  dest[64 + sub] = scale.__x;
}

__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                    uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Each of four warps owns all sixteen rows and 32 of 128 output columns.
// The identical k32 chain serves QK and PV at every query-row geometry.
__device__ __forceinline__ void product(const uint8_t* a, const uint8_t* b, float (&acc)[4][4]) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int row = lane / 4, column = (lane % 4) * 4;
#pragma unroll
  for (int kk = 0; kk < 128; kk += 32) {
    const uint8_t* ap = a + row * kStride + kk + column;
    const uint32_t av[4] = {*reinterpret_cast<const uint32_t*>(ap),
                            *reinterpret_cast<const uint32_t*>(ap + 8 * kStride),
                            *reinterpret_cast<const uint32_t*>(ap + 16),
                            *reinterpret_cast<const uint32_t*>(ap + 8 * kStride + 16)};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint8_t* bp = b + (warp * 32 + j * 8 + row) * kStride + kk + column;
      mma(acc[j], av, *reinterpret_cast<const uint32_t*>(bp),
          *reinterpret_cast<const uint32_t*>(bp + 16));
    }
  }
}

template <int QueryRows>
__global__ __launch_bounds__(kThreads) void minimax_index_nvfp4_kernel(
    const uint8_t* q, const uint8_t* cache, const int32_t* req, const int64_t* positions,
    const int32_t* table, int table_stride, int rows, int heads, int blocks, float* output) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* keys = smem;
  uint8_t* queries = smem + kCacheBytes;
  __shared__ int requests[16];
  __shared__ int64_t positions_local[16];
  __shared__ float partial[16][4];
  const int row0 = blockIdx.x * QueryRows, head = blockIdx.y;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  if (tid < 16) {
    positions_local[tid] = tid < QueryRows && row0 + tid < rows ? positions[row0 + tid] : -1;
    requests[tid] = positions_local[tid] < 0 ? -1 : req[row0 + tid];
  }
  __syncthreads();
  // Usually all sixteen prefill queries share a request. A mixed tile walks
  // its distinct requests, preserving the same matrix product for each row.
  for (int owner = 0; owner < QueryRows; ++owner) {
    const int request = requests[owner];
    bool first = request >= 0;
    for (int previous = 0; previous < owner; ++previous) first &= requests[previous] != request;
    if (!first) continue;
    int64_t last = -1;
    for (int r = owner; r < QueryRows; ++r)
      if (requests[r] == request) last = max(last, positions_local[r]);
    const int end = min(blocks, static_cast<int>(last / 128 + 1));
    if (static_cast<int>(blockIdx.z) >= end) continue;
    for (int i = tid; i < 16 * 128; i += kThreads) {
      const int r = i / 128, d = i % 128;
      queries[r * kStride + d] =
          requests[r] == request ? q[(static_cast<int64_t>(row0 + r) * heads + head) * 128 + d] : 0;
    }
    __syncthreads();
    for (int block = blockIdx.z; block < end; block += gridDim.z) {
      const int physical = table[static_cast<int64_t>(request) * table_stride + block];
      const int64_t begin = static_cast<int64_t>(physical) * 128;
      for (int i = tid; i < 128 * 32; i += kThreads) {
        const int token = i / 32, d = (i % 32) * 4;
        *reinterpret_cast<uint32_t*>(keys + token * kStride + d) =
            static_cast<int64_t>(block) * 128 + token <= last
                ? minimax_cached_fp8x4(cache + (begin + token) * kMinimaxNvfp4HeadBytes, d)
                : 0;
      }
      __syncthreads();
      float acc[4][4] = {};
      product(queries, keys, acc);
      float maximum[2] = {-INFINITY, -INFINITY};
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int component = 0; component < 4; ++component) {
          const int r = lane / 4 + (component / 2) * 8;
          const int token = warp * 32 + j * 8 + (lane % 4) * 2 + component % 2;
          if (requests[r] == request &&
              static_cast<int64_t>(block) * 128 + token <= positions_local[r])
            maximum[component / 2] = fmaxf(maximum[component / 2], acc[j][component]);
        }
#pragma unroll
      for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int offset = 2; offset; offset >>= 1)
          maximum[half] = fmaxf(maximum[half], __shfl_xor_sync(~0u, maximum[half], offset, 4));
        if ((lane & 3) == 0) partial[lane / 4 + half * 8][warp] = maximum[half];
      }
      __syncthreads();
      if (tid < QueryRows && requests[tid] == request &&
          positions_local[tid] >= static_cast<int64_t>(block) * 128) {
        const float score =
            fmaxf(fmaxf(partial[tid][0], partial[tid][1]), fmaxf(partial[tid][2], partial[tid][3]));
        output[(static_cast<int64_t>(row0 + tid) * heads + head) * blocks + block] = score;
      }
      __syncthreads();
    }
  }
}

__global__ __launch_bounds__(kThreads) void minimax_attn_nvfp4_kernel(
    const uint8_t* q, const uint8_t* k, const uint8_t* v, const int32_t* req,
    const int64_t* positions, const int32_t* table, int table_stride, const int32_t* selected,
    int heads, int kv_heads, int topk, float* maxima, float* sums, float* output) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* kv_tile = smem;
  uint8_t* q_tile = smem + kCacheBytes;
  float* scores = reinterpret_cast<float*>(q_tile + kQueryBytes);
  __shared__ float denominators[16];
  const int row = blockIdx.x, split = blockIdx.y, kv = blockIdx.z;
  const int hpk = heads / kv_heads, first_head = kv * hpk;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int block = selected[(static_cast<int64_t>(row) * kv_heads + kv) * topk + split];
  const int64_t position = positions[row];
  const int64_t out_base = (static_cast<int64_t>(row) * topk + split) * heads + first_head;
  if (block < 0 || position < static_cast<int64_t>(block) * 128) {
    if (tid < hpk) {
      maxima[out_base + tid] = -INFINITY;
      sums[out_base + tid] = 0;
    }
    for (int i = tid; i < hpk * 128; i += kThreads) output[out_base * 128 + i] = 0;
    return;
  }
  const int physical = table[static_cast<int64_t>(req[row]) * table_stride + block];
  const int visible =
      static_cast<int>(min(int64_t(128), position - static_cast<int64_t>(block) * 128 + 1));
  const int64_t cache_start = static_cast<int64_t>(physical) * 128;
  for (int i = tid; i < 16 * 128; i += kThreads) {
    const int h = i / 128, d = i % 128;
    q_tile[h * kStride + d] =
        h < hpk ? q[(static_cast<int64_t>(row) * heads + first_head + h) * 128 + d] : 0;
  }
  for (int i = tid; i < 128 * 32; i += kThreads) {
    const int token = i / 32, d = (i % 32) * 4;
    *reinterpret_cast<uint32_t*>(kv_tile + token * kStride + d) =
        token < visible
            ? minimax_cached_fp8x4(k + ((cache_start + token) * kv_heads + kv) * kMinimaxNvfp4HeadBytes, d)
            : 0;
  }
  __syncthreads();
  float qk[4][4] = {};
  product(q_tile, kv_tile, qk);
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int component = 0; component < 4; ++component) {
      const int h = lane / 4 + (component / 2) * 8;
      const int token = warp * 32 + j * 8 + (lane % 4) * 2 + component % 2;
      scores[h * 128 + token] =
          token < visible ? qk[j][component] * .08838834764831845f : -INFINITY;
    }
  __syncthreads();
  // The query tile becomes FP8 probabilities once QK has finished.
  for (int h = warp; h < hpk; h += 4) {
    float local_max = -INFINITY, probability[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) local_max = fmaxf(local_max, scores[h * 128 + lane + j * 32]);
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
      local_max = fmaxf(local_max, __shfl_xor_sync(~0u, local_max, offset));
    float total = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      probability[j] = expf(scores[h * 128 + lane + j * 32] - local_max);
      total += probability[j];
      q_tile[h * kStride + lane + j * 32] = __nv_fp8_e4m3(probability[j] * 448.f).__x;
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) total += __shfl_xor_sync(~0u, total, offset);
    if (lane == 0) {
      denominators[h] = total * 448.f;
      maxima[out_base + h] = local_max + logf(total);
      sums[out_base + h] = 1.f;
    }
  }
  // The key tile becomes transposed V for the second matrix product.
  for (int i = tid; i < 128 * 32; i += kThreads) {
    const int token = i / 32, d = (i % 32) * 4;
    const uint32_t values =
        token < visible
            ? minimax_cached_fp8x4(v + ((cache_start + token) * kv_heads + kv) * kMinimaxNvfp4HeadBytes, d)
            : 0;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      kv_tile[(d + j) * kStride + token] = static_cast<uint8_t>(values >> (j * 8));
  }
  __syncthreads();
  float pv[4][4] = {};
  product(q_tile, kv_tile, pv);
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int component = 0; component < 4; ++component) {
      const int h = lane / 4 + (component / 2) * 8;
      const int d = warp * 32 + j * 8 + (lane % 4) * 2 + component % 2;
      if (h < hpk)
        output[(out_base + h) * 128 + d] =
            __bfloat162float(__float2bfloat16_rn(pv[j][component] / denominators[h]));
    }
}
}  // namespace

void minimax_query_fp8(const uint16_t* input, uint8_t* output, int64_t count, cudaStream_t stream) {
  if (count <= 0) return;
  minimax_query_fp8_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
      input, output, count);
  DGPP_CUDA_OK(cudaGetLastError());
}
void minimax_cache_nvfp4_append(const uint16_t* input, uint8_t* cache, const int32_t* req,
                                const int64_t* positions, const int32_t* table, int stride,
                                int rows, int heads, cudaStream_t stream) {
  if (rows <= 0) return;
  if (heads <= 0 || stride <= 0) throw std::invalid_argument("MiniMax NVFP4 cache geometry");
  minimax_cache_nvfp4_kernel<<<(rows * heads * 8 + 127) / 128, 128, 0, stream>>>(
      input, cache, req, positions, table, stride, rows, heads);
  DGPP_CUDA_OK(cudaGetLastError());
}
void minimax_index_nvfp4_score(const uint8_t* q, const uint8_t* cache, const int32_t* req,
                               const int64_t* positions, const int32_t* table, int stride, int rows,
                               int heads, int blocks, float* scores, cudaStream_t stream) {
  if (rows <= 0) return;
  if (heads <= 0 || blocks <= 0 || stride < blocks)
    throw std::invalid_argument("MiniMax Q8KV4 index geometry");
  // Small decode batches can contain one request per row. Give each its
  // own CTA instead of walking those requests serially inside a 16-row
  // tile. Both paths use the same four k32 tensor-core products per score.
  const unsigned splits = static_cast<unsigned>(blocks < 64 ? blocks : 64);
  if (rows <= 16)
    minimax_index_nvfp4_kernel<1><<<dim3(rows, heads, splits), kThreads,
                                    kCacheBytes + kQueryBytes, stream>>>(
        q, cache, req, positions, table, stride, rows, heads, blocks, scores);
  else
    minimax_index_nvfp4_kernel<16><<<dim3((rows + 15) / 16, heads, splits), kThreads,
                                     kCacheBytes + kQueryBytes, stream>>>(
        q, cache, req, positions, table, stride, rows, heads, blocks, scores);
  DGPP_CUDA_OK(cudaGetLastError());
}

void minimax_attn_nvfp4_partial(const uint8_t* q, const uint8_t* k, const uint8_t* v,
                                const int32_t* req, const int64_t* positions, const int32_t* table,
                                int stride, const int32_t* selected, int rows, int heads,
                                int kv_heads, int topk, float* m, float* l, float* c,
                                cudaStream_t stream) {
  if (rows <= 0) return;
  if (kv_heads <= 0 || heads <= 0 || heads % kv_heads || heads / kv_heads > 16 || topk <= 0 ||
      topk > 16 || stride <= 0)
    throw std::invalid_argument("MiniMax Q8KV4 attention geometry");
  minimax_attn_nvfp4_kernel<<<dim3(rows, topk, kv_heads), kThreads, kAttentionSmem, stream>>>(
      q, k, v, req, positions, table, stride, selected, heads, kv_heads, topk, m, l, c);
  DGPP_CUDA_OK(cudaGetLastError());
}
}  // namespace dgpp
