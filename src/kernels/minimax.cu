#include <cmath>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/minimax.hpp"
#include "kernels/pdl.cuh"

namespace dgpp {
namespace {
__device__ float sum_warp(float x) {
#pragma unroll
  for (int i = 16; i; i >>= 1) x += __shfl_xor_sync(~0u, x, i);
  return x;
}
__device__ float max_warp(float x) {
#pragma unroll
  for (int i = 16; i; i >>= 1) x = fmaxf(x, __shfl_xor_sync(~0u, x, i));
  return x;
}
__device__ float rb(float x) {
  return bf16_bits_to_float(float_to_bf16_bits(x));
}
__global__ void rope_inv_freq_kernel(float* out, int count, int rotary, float theta) {
  dgpp_pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    const float exponent = __fmul_rn(float(2 * i), __fdiv_rn(1.0f, float(rotary)));
    out[i] = __fdiv_rn(1.0f, powf(theta, exponent));
  }
}
__global__ void compact_segments_kernel(const MoeSegment* input, MoeSegment* output, int experts,
                                        int capacity) {
  dgpp_pdl_wait();
  __shared__ int counts[32];
  const int e = threadIdx.x, lane = e % 32, warp = e / 32;
  const bool live = e < experts && input[e].rows > 0;
  const unsigned mask = __ballot_sync(~0u, live);
  if (lane == 0) counts[warp] = __popc(mask);
  __syncthreads();
  int before = 0, total = 0;
  for (int w = 0; w < blockDim.x / 32; ++w) {
    if (w < warp) before += counts[w];
    total += counts[w];
  }
  const int at = before + __popc(mask & ((1u << lane) - 1));
  if (live && at < capacity) output[at] = input[e];
  if (e >= total && e < capacity) output[e] = MoeSegment{};
}
__global__ void add_norm_kernel(uint16_t* residual, const uint16_t* add, const uint16_t* weight,
                                uint16_t* out, int hidden, float eps) {
  dgpp_pdl_wait();
  extern __shared__ float values[];
  __shared__ float sums[8];
  const int row = blockIdx.x, tid = threadIdx.x;
  const int64_t start = static_cast<int64_t>(row) * hidden;
  float sum = 0;
  for (int d = tid; d < hidden; d += blockDim.x) {
    const float value =
        bf16_bits_to_float(residual[start + d]) + bf16_bits_to_float(add[start + d]);
    values[d] = value;
    residual[start + d] = float_to_bf16_bits(value);
    sum = fmaf(value, value, sum);
  }
  sum = sum_warp(sum);
  if (tid % 32 == 0) sums[tid / 32] = sum;
  __syncthreads();
  sum = 0;
  for (int w = 0; w < 8; ++w) sum += sums[w];
  const float rstd = rsqrtf(sum / hidden + eps);
  for (int d = tid; d < hidden; d += blockDim.x)
    out[start + d] = float_to_bf16_bits(values[d] * rstd * (1 + bf16_bits_to_float(weight[d])));
}
__global__ void swiglu_kernel(const uint16_t* gate, const uint16_t* up, uint16_t* out,
                              int64_t count, float alpha, float limit) {
  dgpp_pdl_wait();
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float g = fminf(bf16_bits_to_float(gate[i]), limit);
  const float u = fminf(fmaxf(bf16_bits_to_float(up[i]), -limit), limit);
  out[i] = float_to_bf16_bits((g / (1.0f + expf(-alpha * g))) * (u + 1.0f));
}
__global__ void norm_rope_kernel(const uint16_t* x, const uint16_t* w, uint16_t* y,
                                 const int64_t* pos, const float* inv, int heads, int rotary,
                                 float eps, int64_t x_stride, int64_t y_stride) {
  dgpp_pdl_wait();
  const int r = blockIdx.x, h = blockIdx.y, lane = threadIdx.x;
  const int64_t at = static_cast<int64_t>(r) * x_stride + h * 128;
  const int64_t to = static_cast<int64_t>(r) * y_stride + h * 128;
  float value[4];
  float sum = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    value[j] = bf16_bits_to_float(x[at + lane + 32 * j]);
    sum = fmaf(value[j], value[j], sum);
  }
  const float inv_rms = rsqrtf(sum_warp(sum) * (1.0f / 128) + eps);
  __shared__ float normed[128];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    normed[lane + 32 * j] = rb(value[j] * inv_rms * (1.0f + bf16_bits_to_float(w[lane + 32 * j])));
  __syncwarp();
  const int64_t p = pos[r];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int d = lane + 32 * j, half = rotary / 2;
    float result = normed[d];
    if (d < rotary && p >= 0) {
      const int pair = d < half ? d + half : d - half;
      const float angle = static_cast<float>(p) * inv[d % half];
      const float product = normed[d] * cosf(angle);
      const float rotated = normed[pair] * sinf(angle);
      result = d < half ? __fsub_rn(product, rotated) : __fadd_rn(product, rotated);
    }
    y[to + d] = float_to_bf16_bits(p < 0 ? 0 : result);
  }
}
__global__ void context_norm_kernel(const float* x, const uint16_t* weight, uint16_t* out,
                                    int hidden, float eps) {
  dgpp_pdl_wait();
  __shared__ float sums[8];
  const int tid = threadIdx.x;
  const int64_t at = static_cast<int64_t>(blockIdx.x) * hidden;
  float sum = 0;
  for (int d = tid; d < hidden; d += 256) {
    const float v = rb(x[at + d]);
    sum = fmaf(v, v, sum);
  }
  sum = sum_warp(sum);
  if (tid % 32 == 0) sums[tid / 32] = sum;
  __syncthreads();
  sum = 0;
  for (int w = 0; w < 8; ++w) sum += sums[w];
  const float inv = rsqrtf(sum / hidden + eps);
  for (int d = tid; d < hidden; d += 256)
    out[at + d] = float_to_bf16_bits(rb(x[at + d]) * inv * (1 + bf16_bits_to_float(weight[d])));
}
__global__ void dspark_confidence_kernel(const uint16_t* hidden, const uint16_t* pred,
                                         const int64_t* anchor, const int32_t* proposals,
                                         const float* weight, const float* bias, float* out,
                                         int width, int rank, int vocab) {
  dgpp_pdl_wait();
  __shared__ float sums[8];
  const int row = blockIdx.x, tid = threadIdx.x;
  const int64_t previous = row == 0 ? *anchor : proposals[row - 1];
  if (previous < 0 || previous >= vocab) {
    if (tid == 0) out[row] = -INFINITY;
    return;
  }
  float sum = 0;
  for (int d = tid; d < width; d += 256)
    sum = fmaf(bf16_bits_to_float(hidden[static_cast<int64_t>(row) * width + d]), weight[d], sum);
  for (int d = tid; d < rank; d += 256)
    sum = fmaf(bf16_bits_to_float(pred[previous * rank + d]), weight[width + d], sum);
  sum = sum_warp(sum);
  if (tid % 32 == 0) sums[tid / 32] = sum;
  __syncthreads();
  if (tid == 0) {
    sum = *bias;
    for (int w = 0; w < 8; ++w) sum += sums[w];
    // The shared verification schedule consumes logits and applies sigmoid.
    out[row] = sum;
  }
}
__global__ void append_kernel(const uint16_t* x, uint16_t* cache, const int32_t* req,
                              const int64_t* pos, const int32_t* table, int stride) {
  dgpp_pdl_wait();
  const int r = blockIdx.x, d = threadIdx.x;
  const int64_t p = pos[r];
  if (p < 0) return;
  const int64_t physical =
      static_cast<int64_t>(table[static_cast<int64_t>(req[r]) * stride + p / 128]) * 128 + p % 128;
  cache[physical * 128 + d] = x[static_cast<int64_t>(r) * 128 + d];
}
__global__ void index_score_reference_kernel(const uint16_t* q, const uint16_t* cache,
                                             const int32_t* req, const int64_t* pos,
                                             const int32_t* table, int stride, int heads,
                                             int blocks, float* scores) {
  dgpp_pdl_wait();
  const int r = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int lane = threadIdx.x;
  const int64_t p = pos[r];
  float maximum = -INFINITY;
  if (p >= static_cast<int64_t>(b) * 128) {
    __shared__ float query[128];
    for (int d = lane; d < 128; d += 32)
      query[d] = bf16_bits_to_float(q[(static_cast<int64_t>(r) * heads + h) * 128 + d]);
    __syncwarp();
    const int physical = table[static_cast<int64_t>(req[r]) * stride + b];
    for (int t = lane; t < 128; t += 32) {
      if (static_cast<int64_t>(b) * 128 + t > p) break;
      const uint16_t* key = cache + (static_cast<int64_t>(physical) * 128 + t) * 128;
      float dot = 0;
#pragma unroll 4
      for (int d = 0; d < 128; ++d) dot = fmaf(query[d], bf16_bits_to_float(key[d]), dot);
      maximum = fmaxf(maximum, dot);
    }
  }
  maximum = max_warp(maximum);
  if (lane == 0) scores[(static_cast<int64_t>(r) * heads + h) * blocks + b] = maximum;
}
__global__ void index_score_bounded_kernel(const uint16_t* q, const uint16_t* cache,
                                           const int32_t* req, const int64_t* pos,
                                           const int32_t* table, int stride, int heads, int blocks,
                                           float* scores) {
  dgpp_pdl_wait();
  const int r = blockIdx.x, h = blockIdx.y;
  const int lane = threadIdx.x;
  const int64_t p = pos[r];
  // A fixed-size graph grid walks only visible logical blocks. Future scores
  // are not consumed by select_kernel, which uses this same causal bound.
  // Keep each dot and maximum's arithmetic identical to the one-block grid.
  const int end = p < 0 ? 0 : min(blocks, int(p / 128 + 1));
  if (int(blockIdx.z) >= end) return;
  __shared__ float query[128];
  if (p >= static_cast<int64_t>(blockIdx.z) * 128) {
    for (int d = lane; d < 128; d += 32)
      query[d] = bf16_bits_to_float(q[(static_cast<int64_t>(r) * heads + h) * 128 + d]);
    __syncwarp();
  }
  for (int b = blockIdx.z; b < end; b += gridDim.z) {
    float maximum = -INFINITY;
    if (p >= static_cast<int64_t>(b) * 128) {
      const int physical = table[static_cast<int64_t>(req[r]) * stride + b];
      for (int t = lane; t < 128; t += 32) {
        if (static_cast<int64_t>(b) * 128 + t > p) break;
        const uint16_t* key = cache + (static_cast<int64_t>(physical) * 128 + t) * 128;
        float dot = 0;
#pragma unroll 4
        for (int d = 0; d < 128; ++d) dot = fmaf(query[d], bf16_bits_to_float(key[d]), dot);
        maximum = fmaxf(maximum, dot);
      }
    }
    maximum = max_warp(maximum);
    if (lane == 0) scores[(static_cast<int64_t>(r) * heads + h) * blocks + b] = maximum;
  }
}
// Eight or sixteen queries share a coalesced key load. Each thread evaluates
// four or eight independent, ascending-128-term FP32 dot chains; there is no query
// quantization or reassociation. A mixed-request tile uses one warp per query
// and the original dot chain, avoiding any cross-request key reuse.
template <int tile>
__global__ void index_score_tiled_kernel(const uint16_t* q, const uint16_t* cache,
                                         const int32_t* req, const int64_t* pos,
                                         const int32_t* table, int stride, int rows, int heads,
                                         int blocks, float* scores) {
  dgpp_pdl_wait();
  const int row0 = blockIdx.x * tile, h = blockIdx.y, tid = threadIdx.x;
  const int warp = tid / 32, lane = tid % 32;
  __shared__ int64_t positions[tile], last;
  __shared__ int requests[tile], same_request, visible, owner;
  __shared__ float query[tile][128], partial[tile][4];
  // 130 halfwords per token gives successive tokens distinct shared banks
  // when lanes walk the same channel of different keys.
  __shared__ uint16_t keys[128][130];
  if (tid == 0) {
    last = -1;
    owner = -1;
    same_request = 1;
    for (int i = 0; i < tile; ++i) {
      positions[i] = row0 + i < rows ? pos[row0 + i] : -1;
      requests[i] = positions[i] >= 0 ? req[row0 + i] : -1;
      if (positions[i] < 0) continue;
      last = max(last, positions[i]);
      if (owner < 0) owner = requests[i];
      same_request &= owner == requests[i];
    }
    visible = last < 0 ? 0 : min(blocks, int(last / 128 + 1));
  }
  __syncthreads();
  if (int(blockIdx.z) >= visible) return;
  for (int i = tid; i < tile * 128; i += blockDim.x) {
    const int r = i / 128, d = i % 128;
    query[r][d] =
        positions[r] < 0
            ? 0.f
            : bf16_bits_to_float(q[(static_cast<int64_t>(row0 + r) * heads + h) * 128 + d]);
  }
  __syncthreads();
  if (!same_request) {
    for (int r = warp; r < tile; r += 8) {
      const int64_t p = positions[r];
      const int end = p < 0 ? 0 : min(blocks, int(p / 128 + 1));
      for (int b = blockIdx.z; b < end; b += gridDim.z) {
        const int physical = table[static_cast<int64_t>(requests[r]) * stride + b];
        float maximum = -INFINITY;
        for (int t = lane; t < 128; t += 32) {
          if (static_cast<int64_t>(b) * 128 + t > p) break;
          const uint16_t* key = cache + (static_cast<int64_t>(physical) * 128 + t) * 128;
          float dot = 0;
#pragma unroll 4
          for (int d = 0; d < 128; ++d) dot = fmaf(query[r][d], bf16_bits_to_float(key[d]), dot);
          maximum = fmaxf(maximum, dot);
        }
        maximum = max_warp(maximum);
        if (lane == 0) scores[(static_cast<int64_t>(row0 + r) * heads + h) * blocks + b] = maximum;
      }
    }
    return;
  }
  for (int b = blockIdx.z; b < visible; b += gridDim.z) {
    const int physical = table[static_cast<int64_t>(owner) * stride + b];
    for (int i = tid; i < 128 * 128; i += blockDim.x) {
      const int t = i / 128, d = i % 128;
      keys[t][d] = static_cast<int64_t>(b) * 128 + t > last
                       ? 0
                       : cache[static_cast<int64_t>(physical) * 128 * 128 + i];
    }
    __syncthreads();
    const int token = tid % 128, group = tid / 128;
    float dot[tile / 2]{};
#pragma unroll 4
    for (int d = 0; d < 128; ++d) {
      const float k = bf16_bits_to_float(keys[token][d]);
#pragma unroll
      for (int j = 0; j < tile / 2; ++j) dot[j] = fmaf(query[group + j * 2][d], k, dot[j]);
    }
#pragma unroll
    for (int j = 0; j < tile / 2; ++j) {
      const int r = group + j * 2;
      const float maximum =
          max_warp(static_cast<int64_t>(b) * 128 + token <= positions[r] ? dot[j] : -INFINITY);
      if (lane == 0) partial[r][warp % 4] = maximum;
    }
    __syncthreads();
    if (tid < tile && row0 + tid < rows && positions[tid] >= static_cast<int64_t>(b) * 128) {
      float maximum = partial[tid][0];
#pragma unroll
      for (int j = 1; j < 4; ++j) maximum = fmaxf(maximum, partial[tid][j]);
      scores[(static_cast<int64_t>(row0 + tid) * heads + h) * blocks + b] = maximum;
    }
    __syncthreads();
  }
}
// Deterministic score/id reduction. A tie retains the smaller logical id.
__device__ void best_warp(float& score, int& id) {
#pragma unroll
  for (int off = 16; off; off >>= 1) {
    const float other = __shfl_xor_sync(~0u, score, off);
    const int oi = __shfl_xor_sync(~0u, id, off);
    if (other > score || (other == score && oi < id)) {
      score = other;
      id = oi;
    }
  }
}
__global__ void select_kernel(const float* scores, const int64_t* pos, int heads, int blocks,
                              int topk, int32_t* selected) {
  dgpp_pdl_wait();
  const int r = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  __shared__ int chosen[16], warp_id[8];
  __shared__ float warp_score[8];
  const int visible = pos[r] < 0 ? 0 : static_cast<int>(pos[r] / 128 + 1);
  const float* row = scores + (static_cast<int64_t>(r) * heads + h) * blocks;
  for (int j = 0; j < topk; ++j) {
    float best = -INFINITY;
    int id = INT32_MAX;
    for (int b = tid; b < visible; b += blockDim.x) {
      bool used = false;
      for (int k = 0; k < j; ++k) used |= chosen[k] == b;
      if (used) continue;
      float score = b == visible - 1 ? INFINITY : row[b];
      if (isnan(score)) score = -INFINITY;
      if (score > best || (score == best && b < id)) {
        best = score;
        id = b;
      }
    }
    best_warp(best, id);
    if (lane == 0) {
      warp_score[warp] = best;
      warp_id[warp] = id;
    }
    __syncthreads();
    if (warp == 0) {
      best = lane < 8 ? warp_score[lane] : -INFINITY;
      id = lane < 8 ? warp_id[lane] : INT32_MAX;
      best_warp(best, id);
      if (lane == 0) chosen[j] = id;
    }
    __syncthreads();
  }
  if (tid == 0) {
    for (int i = 1; i < topk; ++i) {
      int id = chosen[i], j = i;
      while (j > 0 && chosen[j - 1] > id) {
        chosen[j] = chosen[j - 1];
        --j;
      }
      chosen[j] = id;
    }
    int32_t* dst = selected + (static_cast<int64_t>(r) * heads + h) * topk;
    for (int j = 0; j < topk; ++j) dst[j] = chosen[j] == INT32_MAX ? -1 : chosen[j];
  }
}
__global__ void router_dot_kernel(const uint16_t* x, const float* w, float* scores, int hidden,
                                  int experts) {
  dgpp_pdl_wait();
  const int r = blockIdx.x, e = blockIdx.y * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (e >= experts) return;
  float sum = 0;
  for (int d = lane; d < hidden; d += 32)
    sum = fmaf(bf16_bits_to_float(x[static_cast<int64_t>(r) * hidden + d]),
               w[static_cast<int64_t>(e) * hidden + d], sum);
  sum = sum_warp(sum);
  if (lane == 0) scores[static_cast<int64_t>(r) * experts + e] = 1.0f / (1.0f + expf(-sum));
}
__global__ void router_select_kernel(const float* scores, const float* bias, int experts, int topk,
                                     float factor, int32_t* ids, float* weights) {
  dgpp_pdl_wait();
  const int r = blockIdx.x, lane = threadIdx.x;
  const float* row = scores + static_cast<int64_t>(r) * experts;
  __shared__ int picked[16];
  for (int j = 0; j < topk; ++j) {
    float best = -INFINITY;
    int id = INT32_MAX;
    for (int e = lane; e < experts; e += 32) {
      bool used = false;
      for (int k = 0; k < j; ++k) used |= picked[k] == e;
      const float s = row[e] + bias[e];
      if (!used && (s > best || (s == best && e < id))) {
        best = s;
        id = e;
      }
    }
    best_warp(best, id);
    if (!lane) picked[j] = id;
    __syncwarp();
  }
  if (!lane) {
    for (int i = 1; i < topk; ++i) {
      int id = picked[i], j = i;
      while (j > 0 && picked[j - 1] > id) {
        picked[j] = picked[j - 1];
        --j;
      }
      picked[j] = id;
    }
    float sum = 0;
    for (int j = 0; j < topk; ++j) sum += row[picked[j]];
    for (int j = 0; j < topk; ++j) {
      ids[r * topk + j] = picked[j];
      weights[r * topk + j] = row[picked[j]] / fmaxf(sum, 1e-20f) * factor;
    }
  }
}
}  // namespace

void minimax_rope_inv_freq(float* out, int count, int rotary, float theta, cudaStream_t stream) {
  if (count <= 0 || count > 64 || rotary <= 0 || rotary > 128 || rotary % 2 ||
      !std::isfinite(theta) || theta <= 0)
    throw std::invalid_argument("MiniMax rotary frequency geometry");
  rope_inv_freq_kernel<<<1, 64, 0, stream>>>(out, count, rotary, theta);
  DGPP_CUDA_OK(cudaGetLastError());
}
void minimax_compact_segments(const MoeSegment* input, MoeSegment* output, int experts,
                              int capacity, cudaStream_t stream) {
  if (!input || !output || input == output || experts < 1 || experts > 1024 || capacity < 1 ||
      capacity > experts)
    throw std::invalid_argument("MiniMax segment compaction geometry");
  compact_segments_kernel<<<1, (experts + 31) / 32 * 32, 0, stream>>>(input, output, experts,
                                                                      capacity);
  DGPP_CUDA_OK(cudaGetLastError());
}

void minimax_add_norm(uint16_t* residual, const uint16_t* add, const uint16_t* weight,
                      uint16_t* out, int rows, int hidden, float eps, cudaStream_t stream) {
  if (rows <= 0) return;
  if (hidden <= 0 || hidden > 8192) throw std::invalid_argument("MiniMax add/norm hidden width");
  add_norm_kernel<<<rows, 256, static_cast<size_t>(hidden) * sizeof(float), stream>>>(
      residual, add, weight, out, hidden, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void minimax_swiglu(const uint16_t* gate, const uint16_t* up, uint16_t* out, int64_t count,
                    float alpha, float limit, cudaStream_t stream) {
  if (count <= 0) return;
  swiglu_kernel<<<(count + 255) / 256, 256, 0, stream>>>(gate, up, out, count, alpha, limit);
  DGPP_CUDA_OK(cudaGetLastError());
}
void minimax_norm_rope(const uint16_t* x, const uint16_t* weight, uint16_t* y, const int64_t* pos,
                       const float* inv_freq, int rows, int heads, int rotary_dim, float eps,
                       cudaStream_t stream, int64_t x_stride, int64_t y_stride) {
  if (rows <= 0) return;
  if (x_stride == 0) x_stride = static_cast<int64_t>(heads) * 128;
  if (y_stride == 0) y_stride = static_cast<int64_t>(heads) * 128;
  if (x_stride < static_cast<int64_t>(heads) * 128 || y_stride < static_cast<int64_t>(heads) * 128)
    throw std::invalid_argument("MiniMax norm/RoPE row stride");
  if (heads <= 0 || rotary_dim <= 0 || rotary_dim > 128 || rotary_dim % 2)
    throw std::invalid_argument("MiniMax norm/RoPE shape");
  norm_rope_kernel<<<dim3(rows, heads), 32, 0, stream>>>(x, weight, y, pos, inv_freq, heads,
                                                         rotary_dim, eps, x_stride, y_stride);
  DGPP_CUDA_OK(cudaGetLastError());
}
void minimax_context_norm(const float* x, const uint16_t* weight, uint16_t* out, int rows,
                          int hidden, float eps, cudaStream_t stream) {
  if (rows <= 0) return;
  if (hidden <= 0) throw std::invalid_argument("MiniMax context norm hidden");
  context_norm_kernel<<<rows, 256, 0, stream>>>(x, weight, out, hidden, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}
void minimax_dspark_confidence(const uint16_t* hidden, const uint16_t* pred, const int64_t* anchor,
                               const int32_t* proposals, const float* weight, const float* bias,
                               float* out, int drafts, int width, int rank, int vocab,
                               cudaStream_t stream) {
  if (drafts <= 0) return;
  if (width <= 0 || rank <= 0 || vocab <= 0)
    throw std::invalid_argument("MiniMax DSpark confidence shape");
  dspark_confidence_kernel<<<drafts, 256, 0, stream>>>(hidden, pred, anchor, proposals, weight,
                                                       bias, out, width, rank, vocab);
  DGPP_CUDA_OK(cudaGetLastError());
}
void minimax_index_append(const uint16_t* k, uint16_t* cache, const int32_t* req,
                          const int64_t* pos, const int32_t* table, int stride, int rows,
                          cudaStream_t stream) {
  if (rows <= 0) return;
  append_kernel<<<rows, 128, 0, stream>>>(k, cache, req, pos, table, stride);
  DGPP_CUDA_OK(cudaGetLastError());
}
void minimax_index_score(const uint16_t* q, const uint16_t* cache, const int32_t* req,
                         const int64_t* pos, const int32_t* table, int stride, int rows, int heads,
                         int blocks, float* scores, cudaStream_t stream, MinimaxIndexScan scan) {
  if (rows <= 0) return;
  if (blocks <= 0 || blocks > stride || heads <= 0)
    throw std::invalid_argument("MiniMax index shape");
  if (scan == MinimaxIndexScan::Tiled && rows >= 8) {
    const int tile = rows > 64 ? 16 : 8;
    const int tiles = (rows + tile - 1) / tile;
    const int groups = std::min(blocks, std::max(1, 128 / tiles / heads));
    if (tile == 16)
      index_score_tiled_kernel<16><<<dim3(tiles, heads, groups), 256, 0, stream>>>(
          q, cache, req, pos, table, stride, rows, heads, blocks, scores);
    else
      index_score_tiled_kernel<8><<<dim3(tiles, heads, groups), 256, 0, stream>>>(
          q, cache, req, pos, table, stride, rows, heads, blocks, scores);
  } else if (scan == MinimaxIndexScan::Bounded) {
    const int groups = std::min(blocks, std::max(1, 1024 / rows / heads));
    index_score_bounded_kernel<<<dim3(rows, heads, groups), 32, 0, stream>>>(
        q, cache, req, pos, table, stride, heads, blocks, scores);
  } else {
    index_score_reference_kernel<<<dim3(rows, heads, blocks), 32, 0, stream>>>(
        q, cache, req, pos, table, stride, heads, blocks, scores);
  }
  DGPP_CUDA_OK(cudaGetLastError());
}
void minimax_index_select(const float* scores, const int64_t* pos, int rows, int heads, int blocks,
                          int topk, int32_t* selected, cudaStream_t stream) {
  if (rows <= 0) return;
  if (topk <= 0 || topk > 16) throw std::invalid_argument("MiniMax index top-k");
  select_kernel<<<dim3(rows, heads), 256, 0, stream>>>(scores, pos, heads, blocks, topk, selected);
  DGPP_CUDA_OK(cudaGetLastError());
}
void minimax_router(const uint16_t* x, const float* gate, const float* bias, int rows, int hidden,
                    int experts, int topk, float factor, float* scores, int32_t* ids,
                    float* weights, cudaStream_t stream) {
  if (rows <= 0) return;
  if (experts <= 0 || topk <= 0 || topk > 16 || topk > experts)
    throw std::invalid_argument("MiniMax router shape");
  router_dot_kernel<<<dim3(rows, (experts + 3) / 4), 128, 0, stream>>>(x, gate, scores, hidden,
                                                                       experts);
  router_select_kernel<<<rows, 32, 0, stream>>>(scores, bias, experts, topk, factor, ids, weights);
  DGPP_CUDA_OK(cudaGetLastError());
}
}  // namespace dgpp
