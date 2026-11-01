// The CSA2 row norms and rotation as device bodies (2026-11-01): the
// production kernels in csa2.cu (rmsnorm_kernel, norms_rope_kernel) and the
// decode tail in dsa.cu call them one block a row, and the step kernel's
// row-tail units (kernels/step/stream.cu) call the same bodies for the rows
// of a step, so the two paths are bitwise by construction. Every body runs
// on a 256-thread block: block_sum_256's order is the production's.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"

namespace dgpp::csa2_norm {

// Block-wide sum in a fixed order (warp shuffles then one warp over the
// warp sums): deterministic, launch-shape independent for a given
// blockDim.
__device__ __forceinline__ float block_sum_256(float v, float* red /*[8]*/) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __syncthreads();
  if (lane == 0) red[warp] = v;
  __syncthreads();
  float t = 0.0f;
  if (warp == 0) {
    t = lane < (int(blockDim.x) >> 5) ? red[lane] : 0.0f;
    for (int off = 4; off > 0; off >>= 1) t += __shfl_xor_sync(0xffffffffu, t, off);
    if (lane == 0) red[0] = t;
  }
  __syncthreads();
  t = red[0];
  __syncthreads();
  return t;
}

// RMSNorm of one row: each thread owns the elements t, t + 256, ...; the
// weight is applied in fp32 after the fp32 normalization (two fp32
// products, the reference's `x * rsqrt(var + eps)` then `weight * x`),
// one bf16 rounding at the end. `yr` may alias `xr`.
__device__ __forceinline__ void rmsnorm_row(const uint16_t* xr, const uint16_t* w, uint16_t* yr, int dim, float eps, float* red /*[8]*/) {
  float ss = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = bf16_bits_to_float(xr[i]);
    ss = __fmaf_rn(v, v, ss);
  }
  const float total = block_sum_256(ss, red);
  const float rs = rsqrtf(__fadd_rn(__fdiv_rn(total, float(dim)), eps));
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = __fmul_rn(bf16_bits_to_float(xr[i]), rs);
    yr[i] = float_to_bf16_bits(__fmul_rn(bf16_bits_to_float(w[i]), v));
  }
}

// One rotated pair (elements at, at + 1) of a row at position p with the
// table entry inv_freq[i] of its pair index: the production's per-pair math
// (rope_apply_kernel's). `out` may alias `in`.
__device__ __forceinline__ void rope_pair(const uint16_t* in, uint16_t* out, int64_t at, int64_t p, float inv_freq_i) {
  const float ang = __fmul_rn(float(p), inv_freq_i);
  const float c = cosf(ang);
  const float sn = sinf(ang);
  const float x0 = bf16_bits_to_float(in[at]), x1 = bf16_bits_to_float(in[at + 1]);
  out[at] = float_to_bf16_bits(x0 * c - x1 * sn);
  out[at + 1] = float_to_bf16_bits(x1 * c + x0 * sn);
}

// The rotation of one row's heads: pair i of head h at `rope_off + 2 i` of
// the head, `half` pairs a head, position p (p < 0: the row is skipped) and
// the table inv_freq [half]; each pair is read and written by one thread,
// so the thread mapping does not touch the result. `out` may alias `in`.
__device__ __forceinline__ void rope_row(const uint16_t* in, uint16_t* out, int heads, int64_t head_stride, int64_t rope_off, int half,
                                         int64_t p, const float* inv_freq) {
  if (p < 0) return;
  for (int idx = threadIdx.x; idx < heads * half; idx += blockDim.x) {
    const int i = idx % half;
    const int h = idx / half;
    rope_pair(in, out, int64_t(h) * head_stride + rope_off + 2 * i, p, inv_freq[i]);
  }
}

// The rotation of the pairs that fall in columns [n0, n1) of `rows` rows of a
// dense [rows][width] matrix whose heads are `head_stride` wide with their
// rotated tail at rope_off: a projection unit's epilogue over its own
// output block (the decode tail's q rope, the same pairs). n0, rope_off and
// head_stride even, so no pair straddles the block.
__device__ __forceinline__ void rope_block(uint16_t* y, int width, int rows, int n0, int n1, int head_stride, int rope_off, int half, const int64_t* pos,
                                           const float* inv_freq) {
  const int pairs = (n1 - n0) / 2;
  for (int q = threadIdx.x; q < rows * pairs; q += blockDim.x) {
    const int t = q / pairs;
    const int c = n0 + 2 * (q % pairs);
    const int r = c % head_stride;
    if (r < rope_off || r >= rope_off + 2 * half) continue;
    const int64_t p = pos[t];
    if (p < 0) continue;
    uint16_t* row = y + static_cast<size_t>(t) * width;
    rope_pair(row, row, c, p, inv_freq[(r - rope_off) / 2]);
  }
}

}  // namespace dgpp::csa2_norm
