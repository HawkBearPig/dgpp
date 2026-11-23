#include <cstdint>

#include <math_constants.h>

#include "kernels/pdl.cuh"

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/inkling2_vision.hpp"

namespace dgpp {
namespace {
constexpr int B = 256;

__device__ inline float bf(uint16_t v) { return bf16_bits_to_float(v); }
__device__ inline uint16_t fb(float v) { return float_to_bf16_bits(v); }
// Round to bf16 and continue in fp32.
__device__ inline float rb(float v) { return bf(fb(v)); }

unsigned grid_for(int64_t total) { return static_cast<unsigned>((total + B - 1) / B); }

// The processor's CLIP statistics. The released kernel normalizes with the
// processor's float32-rounded deviations, rounds to bf16 and maps back with
// the exact ones.
__device__ inline float pixel_value(float v, int channel) {
  constexpr float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
  constexpr float clip_std[3] = {0.26862954f, 0.2613026f, 0.2757771f};
  constexpr float std[3] = {0.26862954f, 0.26130258f, 0.27577711f};
  const float clip = rb(__fdiv_rn(__fsub_rn(__fdiv_rn(v, 255.0f), mean[channel]), clip_std[channel]));
  return __fsub_rn(__fmul_rn(2.0f, __fadd_rn(__fmul_rn(clip, std[channel]), mean[channel])), 1.0f);
}

__global__ void patches(const uint8_t* rgb, uint16_t* out, int width, int valid_width, int valid_height,
                        int patch, int features, int64_t total) {
  dgpp_pdl_wait();
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t p = i / features;
  const int f = static_cast<int>(i % features);
  const int py = f / (patch * 3), px = (f / 3) % patch, c = f % 3;
  const int wp = width / patch;
  const int y = static_cast<int>(p / wp) * patch + py;
  const int x = static_cast<int>(p % wp) * patch + px;
  const bool inside = x < valid_width && y < valid_height;
  const float v = inside ? static_cast<float>(rgb[(static_cast<int64_t>(y) * width + x) * 3 + c]) : -1.0f;
  out[i] = fb(pixel_value(v, c));
}

// align_corners = false source coordinates, clamped to the table.
struct Tap {
  int lo, hi;
  float w_lo, w_hi;
};
__device__ inline Tap source_tap(int index, int target, int source) {
  float coord = __fsub_rn(__fdiv_rn(__fmul_rn(static_cast<float>(source), __fadd_rn(static_cast<float>(index), 0.5f)),
                                    static_cast<float>(target)),
                          0.5f);
  coord = fmaxf(0.0f, fminf(coord, static_cast<float>(source - 1)));
  Tap t;
  t.lo = static_cast<int>(floorf(coord));
  t.hi = t.lo + 1 < source ? t.lo + 1 : source - 1;
  t.w_hi = __fsub_rn(coord, static_cast<float>(t.lo));
  t.w_lo = __fsub_rn(1.0f, t.w_hi);
  return t;
}
__device__ inline float table_at(const uint16_t* table, int sy, int sx, int side, int hidden, int e) {
  return bf(table[(static_cast<int64_t>(sy) * side + sx) * hidden + e]);
}
// The larger weight leads the fused multiply-add, as the reference's lerp.
__device__ inline float lerp(float a, float b, float wa, float wb) {
  return wa >= wb ? __fmaf_rn(a, wa, __fmul_rn(b, wb)) : __fmaf_rn(b, wb, __fmul_rn(a, wa));
}

__global__ void embed_rows(const uint16_t* patch_embed, const uint16_t* table, const uint16_t* class_embedding,
                           const uint16_t* class_position, uint16_t* out, int hp, int wp, int side, int hidden,
                           int64_t total) {
  dgpp_pdl_wait();
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t row = i / hidden;
  const int e = static_cast<int>(i % hidden);
  if (row == 0) {
    out[i] = fb(__fadd_rn(bf(class_embedding[e]), bf(class_position[e])));
    return;
  }
  const int64_t patch = row - 1;
  const int y = static_cast<int>(patch / wp), x = static_cast<int>(patch % wp);
  const Tap ty = source_tap(y, hp, side), tx = source_tap(x, wp, side);
  const float top = lerp(table_at(table, ty.lo, tx.lo, side, hidden, e), table_at(table, ty.lo, tx.hi, side, hidden, e),
                         tx.w_lo, tx.w_hi);
  const float bottom = lerp(table_at(table, ty.hi, tx.lo, side, hidden, e),
                            table_at(table, ty.hi, tx.hi, side, hidden, e), tx.w_lo, tx.w_hi);
  const float position = rb(lerp(top, bottom, ty.w_lo, ty.w_hi));
  out[i] = fb(__fadd_rn(bf(patch_embed[patch * hidden + e]), position));
}

__global__ void scaled_residual(uint16_t* x, const uint16_t* update, const uint16_t* gamma, int dim, int64_t total) {
  dgpp_pdl_wait();
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const float scaled = rb(__fmul_rn(bf(update[i]), bf(gamma[i % dim])));
  x[i] = fb(__fadd_rn(bf(x[i]), scaled));
}

// One thread per dimension pair of the token-major qkv row.
__global__ void split_rope(const uint16_t* qkv, uint16_t* q, uint16_t* k, float* vt, const float* inv_freq, int n,
                           int hidden, int dim, int wp, int64_t total) {
  dgpp_pdl_wait();
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int half_row = 3 * hidden / 2;
  const int64_t t = i / half_row;
  const int rem = static_cast<int>(i % half_row);
  const int part = rem / (hidden / 2);
  const int col = rem % (hidden / 2);
  const int head = col / (dim / 2), p = col % (dim / 2);
  const uint16_t* src = qkv + t * 3 * hidden + part * hidden + head * dim + 2 * p;
  if (part == 2) {
    float* dst = vt + (static_cast<int64_t>(head) * dim + 2 * p) * n + t;
    dst[0] = bf(src[0]);
    dst[n] = bf(src[1]);
    return;
  }
  uint16_t* dst = (part == 0 ? q : k) + (static_cast<int64_t>(head) * n + t) * dim + 2 * p;
  const int quarter = dim / 4;
  // The class row sits at 0 on both axes. A patch sits at its column and
  // row, counted from one.
  int position = 0;
  if (t > 0) {
    const int64_t patch = t - 1;
    position = (p < quarter ? static_cast<int>(patch % wp) : static_cast<int>(patch / wp)) + 1;
  }
  const float angle = __fmul_rn(static_cast<float>(position), inv_freq[p % quarter]);
  const float c = cosf(angle), s = sinf(angle);
  const float even = bf(src[0]), odd = bf(src[1]);
  dst[0] = fb(__fmaf_rn(even, c, -__fmul_rn(odd, s)));
  dst[1] = fb(__fmaf_rn(odd, c, __fmul_rn(even, s)));
}

// One block per score row: the row's largest scaled score, the sum of the
// exponentials and the probabilities, all fp32.
__global__ void softmax_rows(float* scores, int n, float scale) {
  dgpp_pdl_wait();
  __shared__ float partial[B];
  float* row = scores + static_cast<int64_t>(blockIdx.x) * n;
  const int t = threadIdx.x;
  float largest = -CUDART_INF_F;
  for (int j = t; j < n; j += B) largest = fmaxf(largest, __fmul_rn(row[j], scale));
  partial[t] = largest;
  __syncthreads();
  for (int s = B / 2; s > 0; s >>= 1) {
    if (t < s) partial[t] = fmaxf(partial[t], partial[t + s]);
    __syncthreads();
  }
  largest = partial[0];
  __syncthreads();
  float sum = 0.0f;
  for (int j = t; j < n; j += B) sum = __fadd_rn(sum, expf(__fsub_rn(__fmul_rn(row[j], scale), largest)));
  partial[t] = sum;
  __syncthreads();
  for (int s = B / 2; s > 0; s >>= 1) {
    if (t < s) partial[t] = __fadd_rn(partial[t], partial[t + s]);
    __syncthreads();
  }
  const float denom = partial[0];
  for (int j = t; j < n; j += B) row[j] = __fdiv_rn(expf(__fsub_rn(__fmul_rn(row[j], scale), largest)), denom);
}

__global__ void pool(const uint16_t* x, uint16_t* out, int wp, int merge, int hidden, int64_t total) {
  dgpp_pdl_wait();
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int width = merge * merge * hidden;
  const int64_t token = i / width;
  const int r = static_cast<int>(i % width);
  const int part = r / hidden, e = r % hidden;
  const int dy = part / merge, dx = part % merge;
  const int wt = wp / merge;
  const int64_t ty = token / wt, tx = token % wt;
  const int64_t row = 1 + (ty * merge + dy) * wp + tx * merge + dx;
  out[i] = x[row * hidden + e];
}
}  // namespace

void ink2_vision_patches(const uint8_t* rgb, uint16_t* out, int width, int height, int valid_width,
                         int valid_height, int patch, cudaStream_t stream) {
  const int features = 3 * patch * patch;
  const int64_t total = static_cast<int64_t>(width / patch) * (height / patch) * features;
  if (total <= 0) return;
  patches<<<grid_for(total), B, 0, stream>>>(rgb, out, width, valid_width, valid_height, patch, features, total);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_vision_embed_rows(const uint16_t* patch_embed, const uint16_t* position_table,
                            const uint16_t* class_embedding, const uint16_t* class_position, uint16_t* out,
                            int hp, int wp, int side, int hidden, cudaStream_t stream) {
  const int64_t total = (static_cast<int64_t>(hp) * wp + 1) * hidden;
  embed_rows<<<grid_for(total), B, 0, stream>>>(patch_embed, position_table, class_embedding, class_position, out,
                                                hp, wp, side, hidden, total);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_vision_scaled_residual(uint16_t* x, const uint16_t* update, const uint16_t* gamma, int rows, int dim,
                                 cudaStream_t stream) {
  const int64_t total = static_cast<int64_t>(rows) * dim;
  if (total <= 0) return;
  scaled_residual<<<grid_for(total), B, 0, stream>>>(x, update, gamma, dim, total);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_vision_split_rope(const uint16_t* qkv, uint16_t* q, uint16_t* k, float* vt, const float* inv_freq,
                            int n, int hidden, int heads, int wp, cudaStream_t stream) {
  const int dim = hidden / heads;
  const int64_t total = static_cast<int64_t>(n) * 3 * hidden / 2;
  if (total <= 0) return;
  split_rope<<<grid_for(total), B, 0, stream>>>(qkv, q, k, vt, inv_freq, n, hidden, dim, wp, total);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_vision_softmax(float* scores, int rows, int n, float scale, cudaStream_t stream) {
  if (rows <= 0 || n <= 0) return;
  softmax_rows<<<static_cast<unsigned>(rows), B, 0, stream>>>(scores, n, scale);
  DGPP_CUDA_OK(cudaGetLastError());
}

void ink2_vision_pool(const uint16_t* x, uint16_t* out, int hp, int wp, int merge, int hidden,
                      cudaStream_t stream) {
  const int64_t total = static_cast<int64_t>(hp / merge) * (wp / merge) * merge * merge * hidden;
  if (total <= 0) return;
  pool<<<grid_for(total), B, 0, stream>>>(x, out, wp, merge, hidden, total);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
