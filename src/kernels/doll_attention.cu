#include "kernels/doll_attention.hpp"
#include "kernels/pdl.cuh"

#include <assert.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {

namespace {

// Shared-memory score ceiling per block. The doll caps total sequence length
// well below this; the launcher asserts it. 8192 * 4B = 32 KB/block.
constexpr int kMaxSeqCap = 8192;

__global__ void __launch_bounds__(128) gqa_causal_attn_kernel(
    const uint16_t* __restrict__ q,      // [nq, Hq*Dh] strided
    int64_t q_stride,                    // elements per q row
    const uint16_t* __restrict__ kc,     // [cap, Hkv*Dh]
    const uint16_t* __restrict__ vc,     // [cap, Hkv*Dh]
    uint16_t* __restrict__ out,          // [nq, Hq*Dh]
    const int* __restrict__ dev_nq,
    const int* __restrict__ dev_abs_first, int num_heads, int num_kv_heads,
    int head_dim, float softmax_scale) {
  dgpp_pdl_wait();
  constexpr int kDh = 128;  // fast-path specialization; launcher guards
  assert(head_dim == kDh);
  extern __shared__ float sc[];  // [kMaxSeqCap] scores -> probabilities

  const int row = static_cast<int>(blockIdx.x);
  const int nq = *dev_nq;
  if (row >= nq || row >= kMaxSeqCap) return;
  const int abs_row = *dev_abs_first + row;
  const int kh = static_cast<int>(blockIdx.y);  // kv-head index
  const int group = num_heads / num_kv_heads;
  const int tid = static_cast<int>(threadIdx.x);

  const int causal_end = min(abs_row + 1, kMaxSeqCap);
  const size_t kv_stride = static_cast<size_t>(num_kv_heads) * head_dim;
  const size_t o_stride = static_cast<size_t>(num_heads) * head_dim;
  const uint16_t* kbase = kc + kh * head_dim;
  const uint16_t* vbase = vc + kh * head_dim;

  for (int hh = 0; hh < group; ++hh) {
    const int h = kh * group + hh;
    const uint16_t* qrow = q + static_cast<size_t>(row) * q_stride +
                           static_cast<size_t>(h) * kDh;

    // ---- phase A: scores[0..causal_end) via warp-per-timestep dots ------
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nwarp = blockDim.x >> 5;
    float qr[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      qr[c] = bf16_bits_to_float(qrow[lane + 32 * c]);
    }
    __syncthreads();  // scores region reusable across heads
    for (int t = warp; t < causal_end; t += nwarp) {
      const uint16_t* kr = kbase + static_cast<size_t>(t) * kv_stride;
      float partial = 0.f;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        partial += qr[c] * bf16_bits_to_float(kr[lane + 32 * c]);
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        partial += __shfl_down_sync(0xffffffffu, partial, off);
      }
      if (lane == 0) sc[t] = partial * softmax_scale;
    }
    __syncthreads();

    // ---- softmax in place (deterministic fixed-order reductions) --------
    float lmax = -INFINITY;
    for (int t = tid; t < causal_end; t += blockDim.x) {
      lmax = fmaxf(lmax, sc[t]);
    }
    __shared__ float red[128];
    red[tid] = lmax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s && tid + s < blockDim.x) red[tid] = fmaxf(red[tid], red[tid + s]);
      __syncthreads();
    }
    const float smax = red[0];

    float lsum = 0.f;
    for (int t = tid; t < causal_end; t += blockDim.x) {
      float p = __expf(sc[t] - smax);
      sc[t] = p;
      lsum += p;
    }
    __syncthreads();
    red[tid] = lsum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s && tid + s < blockDim.x) red[tid] += red[tid + s];
      __syncthreads();
    }
    const float inv_sum = 1.0f / red[0];
    for (int t = tid; t < causal_end; t += blockDim.x) sc[t] *= inv_sum;
    __syncthreads();

    // ---- phase B: value mixup, dim-lane ownership keeps order fixed -----
    {
      const uint16_t* vr = vbase + static_cast<size_t>(0) * kv_stride + tid;
      float acc = 0.f;
      for (int t = 0; t < causal_end; ++t) {
        acc += sc[t] * bf16_bits_to_float(vr[static_cast<size_t>(t) * kv_stride]);
      }
      out[static_cast<size_t>(row) * o_stride +
          static_cast<size_t>(h) * kDh + tid] = float_to_bf16_bits(acc);
    }
    __syncthreads();
  }
}

}  // namespace

void gqa_causal_attention(const void* q, int64_t q_stride_elems,
                          const void* k_cache, const void* v_cache, void* out,
                          const int* dev_query_count, const int* dev_abs_first,
                          int max_queries_cap, int num_heads, int num_kv_heads,
                          int head_dim, float softmax_scale,
                          cudaStream_t stream) {
  (void)max_queries_cap;
  constexpr size_t kSmem = sizeof(float) * kMaxSeqCap;
  static const bool attr_ok = [] {
    // GB10 rejects opt-in requests above ~48 KB for some kernels; probe the
    // device cap and clamp instead of guessing.
    int maxoptin = 49152;
    cudaDeviceGetAttribute(&maxoptin,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    const size_t want =
        kSmem < static_cast<size_t>(maxoptin) ? kSmem : static_cast<size_t>(maxoptin);
    cudaError_t e = cudaFuncSetAttribute(
        gqa_causal_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(want));
    if (e != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    return true;
  }();
  if (!attr_ok)
    throw std::runtime_error(
        "gqa attention: dynamic smem opt-in rejected by driver");
  constexpr int kBlock = 128;
  dim3 grid(static_cast<unsigned>(max_queries_cap),
            static_cast<unsigned>(num_kv_heads));
  cudaGetLastError();  // clean error slate
  gqa_causal_attn_kernel<<<grid, kBlock, kSmem, stream>>>(
      static_cast<const uint16_t*>(q), q_stride_elems,
      static_cast<const uint16_t*>(k_cache),
      static_cast<const uint16_t*>(v_cache), static_cast<uint16_t*>(out),
      dev_query_count, dev_abs_first, num_heads, num_kv_heads, head_dim,
      softmax_scale);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp