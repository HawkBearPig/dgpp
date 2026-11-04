#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/minimax_vision.hpp"
#include "kernels/pdl.cuh"

namespace dgpp {
namespace {
__global__ void patchify(const uint8_t* rgb, float* out, int width, int64_t count) {
  dgpp_pdl_wait();
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  constexpr int patch = 14, grid = 28, inner = 1176;
  const int token = i / inner, lane = i % inner, channel = lane / (2 * patch * patch);
  const int y =
      (token / 4 / (width / grid)) * grid + (token % 4 / 2) * patch + lane / patch % patch;
  const int x = (token / 4 % (width / grid)) * grid + (token % 2) * patch + lane % patch;
  constexpr float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
  constexpr float std[3] = {0.26862954f, 0.26130258f, 0.27577711f};
  // Match the processor's separate rescale/subtract/divide operations.
  const float rescaled = __fmul_rn(float(rgb[(int64_t(y) * width + x) * 3 + channel]), 1.0f / 255);
  const float normalized = __fdiv_rn(__fsub_rn(rescaled, mean[channel]), std[channel]);
  out[i] = bf16_bits_to_float(float_to_bf16_bits(normalized));
}
__global__ void cast(const float* x, uint16_t* y, int64_t count) {
  dgpp_pdl_wait();
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) y[i] = float_to_bf16_bits(x[i]);
}
__global__ void rope_tables(float* out, int width, int half, float theta, int count) {
  dgpp_pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int token = i / half, axis = half / 3, dim = i % half;
  const int block = token / 4;
  const int pos = dim < axis       ? 0
                  : dim < 2 * axis ? block / (width / 28) * 2 + token % 4 / 2
                                   : block % (width / 28) * 2 + token % 2;
  const float exponent = __fmul_rn(float(dim % axis), __fdiv_rn(1.0f, float(axis)));
  const float freq = __fdiv_rn(1.0f, powf(theta, exponent));
  const float angle = __fmul_rn(float(pos), freq);
  out[i] = cosf(angle);
  out[count + i] = sinf(angle);
}
__global__ void split_rope(const uint16_t* qkv, uint16_t* q, uint16_t* k, uint16_t* v,
                           const float* cos, const float* sin, int n, int hidden, int heads,
                           int rot, int64_t count) {
  dgpp_pdl_wait();
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int dim = hidden / heads, token = i / hidden, col = i % hidden, d = col % dim;
  const int64_t src = int64_t(token) * 3 * hidden + col;
  const int64_t dst = int64_t(col / dim) * n * dim + token * dim + d;
  float qv = bf16_bits_to_float(qkv[src]), kv = bf16_bits_to_float(qkv[src + hidden]);
  if (d < rot) {
    const int half = rot / 2, other = d < half ? half : -half;
    const float sign = d < half ? -1.0f : 1.0f;
    const int64_t at = int64_t(token) * half + d % half;
    const float qr = sign * bf16_bits_to_float(qkv[src + other]);
    const float kr = sign * bf16_bits_to_float(qkv[src + hidden + other]);
    qv = __fadd_rn(__fmul_rn(qv, cos[at]), __fmul_rn(qr, sin[at]));
    kv = __fadd_rn(__fmul_rn(kv, cos[at]), __fmul_rn(kr, sin[at]));
  }
  q[dst] = float_to_bf16_bits(qv);
  k[dst] = float_to_bf16_bits(kv);
  v[dst] = qkv[src + 2 * hidden];
}
}  // namespace
void minimax_vision_patchify(const uint8_t* rgb, float* out, int w, int h, cudaStream_t s) {
  const int64_t count = int64_t(w / 14) * (h / 14) * 1176;
  patchify<<<(count + 255) / 256, 256, 0, s>>>(rgb, out, w, count);
  DGPP_CUDA_OK(cudaGetLastError());
}
void minimax_vision_cast(const float* x, uint16_t* y, int64_t count, cudaStream_t s) {
  cast<<<(count + 255) / 256, 256, 0, s>>>(x, y, count);
  DGPP_CUDA_OK(cudaGetLastError());
}
void minimax_vision_rope_tables(float* out, int width, int height, int rot, float theta,
                                cudaStream_t s) {
  const int count = width / 14 * (height / 14) * (rot / 2);
  rope_tables<<<(count + 255) / 256, 256, 0, s>>>(out, width, rot / 2, theta, count);
  DGPP_CUDA_OK(cudaGetLastError());
}
void minimax_vision_split_rope(const uint16_t* qkv, uint16_t* q, uint16_t* k, uint16_t* v,
                               const float* cos, const float* sin, int n, int hidden, int heads,
                               int rot, cudaStream_t s) {
  const int64_t count = int64_t(n) * hidden;
  split_rope<<<(count + 255) / 256, 256, 0, s>>>(qkv, q, k, v, cos, sin, n, hidden, heads, rot,
                                                 count);
  DGPP_CUDA_OK(cudaGetLastError());
}
}  // namespace dgpp
