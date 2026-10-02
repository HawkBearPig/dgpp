// Qwen3.5 dense SwiGLU kernel (CUDA TU: __global__ cannot live in model35.cpp).
#include <cstdint>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

// Fused SwiGLU: out[i] = silu(gate[i]) * up[i], bf16 in/out, fp32 math with
// a single bf16 rounding at the store (the gate_act convention).
__global__ void qwen35_swiglu_kernel(const uint16_t* __restrict__ gate, const uint16_t* __restrict__ up,
                                     uint16_t* __restrict__ out, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float g = dgpp::bf16_bits_to_float(gate[i]);
  const float s = g / (1.0f + expf(-g));
  out[i] = dgpp::float_to_bf16_bits(s * dgpp::bf16_bits_to_float(up[i]));
}

namespace dgpp {

void qwen35_swiglu_bf16(const uint16_t* gate, const uint16_t* up, uint16_t* out, int64_t n,
                        cudaStream_t stream) {
  if (n <= 0) return;
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((static_cast<uint64_t>(n) + kThreads - 1) / kThreads);
  qwen35_swiglu_kernel<<<blocks, kThreads, 0, stream>>>(gate, up, out, n);
  DGPP_CUDA_OK(cudaGetLastError());
}

// The MTP draft head's concat: out[t, :] = [e[t, :], h[t, :]] (bf16 rows,
// e/h [rows, hidden], out [rows, 2 * hidden]). Elementwise row copy so the
// fused fc [H, 2H] runs as one GEMM, exactly the reference's
// fc(cat(pre_fc_norm_embedding(embed), pre_fc_norm_hidden(hidden))).
__global__ void qwen35_mtp_concat_kernel(const uint16_t* __restrict__ e,
                                         const uint16_t* __restrict__ h,
                                         uint16_t* __restrict__ out, int64_t rows,
                                         int64_t hidden) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t n = rows * hidden * 2;
  if (i >= n) return;
  const int64_t row = i / (hidden * 2);
  const int64_t col = i % (hidden * 2);
  out[i] = col < hidden ? e[row * hidden + col] : h[row * hidden + (col - hidden)];
}

void qwen35_mtp_concat_bf16(const uint16_t* e, const uint16_t* h, uint16_t* out,
                             int64_t rows, int64_t hidden, cudaStream_t stream) {
  if (rows <= 0 || hidden <= 0) return;
  constexpr int kThreads = 256;
  const uint64_t n = static_cast<uint64_t>(rows) * static_cast<uint64_t>(hidden) * 2;
  const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
  qwen35_mtp_concat_kernel<<<blocks, kThreads, 0, stream>>>(e, h, out, rows, hidden);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
