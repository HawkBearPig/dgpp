#include <math_constants.h>
#include "kernels/pdl.cuh"

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/glm_vision.hpp"

namespace dgpp {
namespace {
constexpr int B = 256;
__global__ void patchify(const uint8_t* rgb, uint16_t* out, int width, int n) {
  dgpp_pdl_wait();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * 1176) return;
  const int patch = i / 1176, p = i % 1176, c = p / 392;
  const int y = (patch / 4 / (width / 28)) * 28 + ((patch % 4) / 2) * 14 + (p % 196) / 14;
  const int x = (patch / 4 % (width / 28)) * 28 + (patch % 2) * 14 + p % 14;
  const float mean[] = {0.48145466f, 0.4578275f, 0.40821073f};
  const float std[] = {0.26862954f, 0.26130258f, 0.27577711f};
  out[i] = float_to_bf16_bits((rgb[(y * width + x) * 3 + c] * (1.0f / 255.0f) - mean[c]) / std[c]);
}
__global__ void swiglu(uint16_t* gate, const uint16_t* up, int count, float limit) {
  dgpp_pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float g = fminf(bf16_bits_to_float(gate[i]), limit);
  const float u = fminf(fmaxf(bf16_bits_to_float(up[i]), -limit), limit);
  const float silu = bf16_bits_to_float(float_to_bf16_bits(g / (1 + expf(-g))));
  gate[i] = float_to_bf16_bits(silu * u);
}
__device__ float sum_reduce(float v, float* buf) {
  const int t = threadIdx.x;
  buf[t] = v;
  __syncthreads();
  for (int s = B / 2; s; s /= 2) {
    if (t < s) buf[t] += buf[t + s];
    __syncthreads();
  }
  return buf[0];
}
__global__ void rmsnorm(const uint16_t* x, const uint16_t* w, uint16_t* y, int rows, int dim,
                        float eps) {
  dgpp_pdl_wait();
  const int row = blockIdx.x * 4 + threadIdx.y, lane = threadIdx.x;
  if (row >= rows) return;
  x += row * dim;
  y += row * dim;
  float sums[4] = {};
  for (int j = lane * 4; j < dim; j += 128)
    for (int v = 0; v < 4; ++v) {
      const float value = bf16_bits_to_float(x[j + v]);
      sums[v] = __fadd_rn(sums[v], __fmul_rn(value, value));
    }
  float sum = ((sums[0] + sums[1]) + sums[2]) + sums[3];
  for (int s = 16; s; s /= 2) sum += __shfl_down_sync(0xffffffff, sum, s);
  const float rstd = rsqrtf(__shfl_sync(0xffffffff, sum, 0) / dim + eps);
  for (int j = lane; j < dim; j += 32) {
    const float value = bf16_bits_to_float(float_to_bf16_bits(bf16_bits_to_float(x[j]) * rstd));
    y[j] = float_to_bf16_bits(value * bf16_bits_to_float(w[j]));
  }
}
// Welford schedule adapted from PyTorch CUDA LayerNorm (BSD-3-Clause).
// See docs/licenses/pytorch.md for attribution and license.
struct Moments {
  float mean = 0, m2 = 0, count = 0;
};
__device__ Moments combine_moments(Moments left, Moments right) {
  const float count = left.count + right.count;
  if (count == 0) return {};
  const float inverse = 1.0f / count;
  const float a = right.count * inverse, b = left.count * inverse;
  const float delta = left.mean - right.mean;
  return {a * right.mean + b * left.mean, right.m2 + left.m2 + delta * delta * right.count * b,
          count};
}
__global__ void layernorm_gelu(uint16_t* x, const uint16_t* weight, const uint16_t* bias, int dim,
                               uint16_t* normalized) {
  dgpp_pdl_wait();
  // Welford moments with four adjacent values per thread and warp-pair
  // reduction, matching the CUDA LayerNorm reference's FP32 arithmetic.
  constexpr int threads = 128;
  __shared__ Moments partial[threads / 32];
  const int t = threadIdx.x, lane = t % 32, warp = t / 32;
  uint16_t* row = x + blockIdx.x * dim;
  Moments moments;
  for (int base = t * 4; base < dim; base += threads * 4)
    for (int j = 0; j < 4 && base + j < dim; ++j) {
      const float value = bf16_bits_to_float(row[base + j]);
      const float delta = value - moments.mean;
      const float count = moments.count + 1;
      const float mean = moments.mean + delta * (1.0f / count);
      moments = {mean, moments.m2 + delta * (value - mean), count};
    }
  for (int offset = 16; offset; offset /= 2) {
    Moments other{__shfl_down_sync(0xffffffff, moments.mean, offset),
                  __shfl_down_sync(0xffffffff, moments.m2, offset),
                  __shfl_down_sync(0xffffffff, moments.count, offset)};
    moments = combine_moments(moments, other);
  }
  if (lane == 0) partial[warp] = moments;
  __syncthreads();
  for (int offset = threads / 64; offset; offset /= 2) {
    if (lane == 0 && warp < offset) {
      moments = combine_moments(moments, partial[warp + offset]);
      partial[warp] = moments;
    }
    __syncthreads();
  }
  const float mean = partial[0].mean;
  const float rstd = rsqrtf(partial[0].m2 / dim + 1e-5f);
  for (int j = t; j < dim; j += threads) {
    const float norm = (bf16_bits_to_float(row[j]) - mean) * rstd;
    const float a = bf16_bits_to_float(
        float_to_bf16_bits(norm * bf16_bits_to_float(weight[j]) + bf16_bits_to_float(bias[j])));
    if (normalized) normalized[blockIdx.x * dim + j] = float_to_bf16_bits(a);
    row[j] = float_to_bf16_bits(a * 0.5f * (1 + erff(a * 0.7071067811865475f)));
  }
}
__global__ void pack_qkv(uint16_t* qkv, uint16_t* q, uint16_t* k, uint16_t* v, const uint16_t* qn,
                         const uint16_t* kn, int n, int h, int heads, int gw, float eps) {
  dgpp_pdl_wait();
  const int row = blockIdx.x, head = blockIdx.y, dim = h / heads;
  // Each block owns one head of one token, with <=256 elements.
  __shared__ float buf[B], norm[256];
  const int d = threadIdx.x;
  for (int side = 0; side < 2; ++side) {
    const float x = d < dim ? bf16_bits_to_float(qkv[row * 3 * h + side * h + head * dim + d]) : 0;
    const float rstd = rsqrtf(sum_reduce(x * x, buf) / dim + eps);
    if (d < dim)
      norm[d] =
          bf16_bits_to_float(float_to_bf16_bits(bf16_bits_to_float(float_to_bf16_bits(x * rstd)) *
                                                bf16_bits_to_float((side ? kn : qn)[d])));
    __syncthreads();
    if (d < dim) {
      const int half = dim / 2, quarter = dim / 4;
      const int y = (row / 4 / (gw / 2)) * 2 + row % 4 / 2;
      const int xcoord = (row / 4 % (gw / 2)) * 2 + row % 2;
      const int pos = d % half < quarter ? y : xcoord;
      const float freq = 1.0f / powf(10000.0f, static_cast<float>(d % quarter) / quarter);
      const float angle = pos * freq;
      const float rotated = d < half ? -norm[d + half] : norm[d - half];
      (side ? k : q)[(head * n + row) * dim + d] = float_to_bf16_bits(
          __fadd_rn(__fmul_rn(norm[d], cosf(angle)), __fmul_rn(rotated, sinf(angle))));
    }
    __syncthreads();
  }
  if (d < dim) v[(head * n + row) * dim + d] = qkv[row * 3 * h + 2 * h + head * dim + d];
}
template <bool Maximum>
__device__ float attention_reduce(float value, float* partials) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int s = 16; s; s /= 2) {
    const float other = __shfl_down_sync(0xffffffff, value, s);
    value = Maximum ? fmaxf(value, other) : value + other;
  }
  __syncthreads();
  if (lane == 0) partials[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = partials[lane];
    for (int s = 16; s; s /= 2) {
      const float other = __shfl_down_sync(0xffffffff, value, s);
      value = Maximum ? fmaxf(value, other) : value + other;
    }
    if (lane == 0) partials[0] = value;
  }
  __syncthreads();
  return partials[0];
}
// The upstream eager path rounds QK to BF16 before its FP32 softmax.
// Scaling is also a BF16 tensor operation (exact for GLM-5.3's 1/8 scale).
__device__ float attention_score(float score, float scale) {
  return bf16_bits_to_float(
      float_to_bf16_bits(bf16_bits_to_float(float_to_bf16_bits(score)) * scale));
}
__global__ void softmax(const float* scores, uint16_t* probs, int n, float scale) {
  dgpp_pdl_wait();
  __shared__ float partials[32];
  const int row = blockIdx.x, t = threadIdx.x;
  float largest = -CUDART_INF_F;
  for (int j = t; j < n; j += 1024)
    largest = fmaxf(largest, attention_score(scores[row * n + j], scale));
  largest = attention_reduce<true>(largest, partials);
  float sum = 0;
  for (int j = t; j < n; j += 1024)
    sum += expf(attention_score(scores[row * n + j], scale) - largest);
  const float denom = attention_reduce<false>(sum, partials);
  for (int j = t; j < n; j += 1024)
    probs[row * n + j] =
        float_to_bf16_bits(expf(attention_score(scores[row * n + j], scale) - largest) / denom);
}
__global__ void softmax_warp(const float* scores, uint16_t* probs, int rows, int n, float scale) {
  dgpp_pdl_wait();
  const int row = blockIdx.x * 4 + threadIdx.y, lane = threadIdx.x;
  if (row >= rows) return;
  float largest = -CUDART_INF_F;
  for (int j = lane; j < n; j += 32)
    largest = fmaxf(largest, attention_score(scores[row * n + j], scale));
  for (int s = 16; s; s /= 2) largest = fmaxf(largest, __shfl_xor_sync(0xffffffff, largest, s));
  float sum = 0;
  for (int j = lane; j < n; j += 32)
    sum += expf(attention_score(scores[row * n + j], scale) - largest);
  for (int s = 16; s; s /= 2) sum += __shfl_xor_sync(0xffffffff, sum, s);
  for (int j = lane; j < n; j += 32)
    probs[row * n + j] =
        float_to_bf16_bits(expf(attention_score(scores[row * n + j], scale) - largest) / sum);
}
__global__ void store_attention(const float* tile, uint16_t* out, int rows, int n, int dim,
                                int heads, int first) {
  dgpp_pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < heads * rows * dim) {
    const int head = i / (rows * dim), row = i / dim % rows, d = i % dim;
    if (first + row < n) out[(head * n + first + row) * dim + d] = float_to_bf16_bits(tile[i]);
  }
}
__global__ void unhead(const uint16_t* src, uint16_t* dst, int n, int h, int heads) {
  dgpp_pdl_wait();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n * h) {
    int row = i / h, d = i % h, dim = h / heads;
    dst[i] = src[((d / dim) * n + row) * dim + d % dim];
  }
}
__global__ void merge(const uint16_t* src, uint16_t* dst, int n, int h) {
  dgpp_pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n * h) {
    const int group = i / (4 * h), rem = i % (4 * h);
    dst[i] = src[(group * 4 + rem % 4) * h + rem / 4];
  }
}
__global__ void broadcast(const uint16_t* src, uint16_t* dst, int count, int h) {
  dgpp_pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) dst[i] = src[(i / (4 * h)) * h + i % h];
}
}  // namespace
void vision_patchify(const uint8_t* rgb, uint16_t* patches, int w, int h, cudaStream_t s) {
  const int n = w / 14 * (h / 14);
  patchify<<<(n * 1176 + B - 1) / B, B, 0, s>>>(rgb, patches, w, n);
  DGPP_CUDA_OK(cudaGetLastError());
}
void vision_swiglu(uint16_t* g, const uint16_t* u, int n, float limit, cudaStream_t s) {
  swiglu<<<(n + B - 1) / B, B, 0, s>>>(g, u, n, limit);
  DGPP_CUDA_OK(cudaGetLastError());
}
void vision_rmsnorm(const uint16_t* x, const uint16_t* w, uint16_t* y, int rows, int dim, float eps,
                    cudaStream_t s) {
  rmsnorm<<<(rows + 3) / 4, dim3(32, 4), 0, s>>>(x, w, y, rows, dim, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}
void vision_layernorm_gelu(uint16_t* x, const uint16_t* w, const uint16_t* b, int rows, int dim,
                           cudaStream_t s, uint16_t* normalized) {
  layernorm_gelu<<<rows, 128, 0, s>>>(x, w, b, dim, normalized);
  DGPP_CUDA_OK(cudaGetLastError());
}
void vision_qkv(uint16_t* qkv, uint16_t* q, uint16_t* k, uint16_t* v, const uint16_t* qn,
                const uint16_t* kn, int n, int h, int heads, int gw, float eps, cudaStream_t s) {
  pack_qkv<<<dim3(n, heads), B, 0, s>>>(qkv, q, k, v, qn, kn, n, h, heads, gw, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}
void vision_softmax(const float* x, uint16_t* y, int rows, int n, int dim, cudaStream_t s) {
  if (n <= 2048)
    softmax_warp<<<(rows + 3) / 4, dim3(32, 4), 0, s>>>(x, y, rows, n,
                                                        1.0f / sqrtf(static_cast<float>(dim)));
  else
    softmax<<<rows, 1024, 0, s>>>(x, y, n, 1.0f / sqrtf(static_cast<float>(dim)));
  DGPP_CUDA_OK(cudaGetLastError());
}
void vision_store_attention(const float* tile, uint16_t* out, int rows, int n, int dim, int heads,
                            int first, cudaStream_t s) {
  store_attention<<<(heads * rows * dim + B - 1) / B, B, 0, s>>>(tile, out, rows, n, dim, heads,
                                                                 first);
  DGPP_CUDA_OK(cudaGetLastError());
}
void vision_unhead(const uint16_t* x, uint16_t* y, int n, int h, int heads, cudaStream_t s) {
  unhead<<<(n * h + B - 1) / B, B, 0, s>>>(x, y, n, h, heads);
  DGPP_CUDA_OK(cudaGetLastError());
}
void vision_merge(const uint16_t* x, uint16_t* y, int n, int h, cudaStream_t s) {
  merge<<<(n * h + B - 1) / B, B, 0, s>>>(x, y, n, h);
  DGPP_CUDA_OK(cudaGetLastError());
}
void vision_broadcast(const uint16_t* x, uint16_t* y, int n, int h, cudaStream_t s) {
  broadcast<<<(n * h * 4 + B - 1) / B, B, 0, s>>>(x, y, n * h * 4, h);
  DGPP_CUDA_OK(cudaGetLastError());
}
}  // namespace dgpp
