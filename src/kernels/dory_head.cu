#include "kernels/dory_head.hpp"
#include "kernels/pdl.cuh"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {

namespace {

constexpr int kThreads = 256;

// One element per thread, grid-strided: logits[r, v] *= bf16(logit_scale[v]) *
// factor. The scale is re-read per element (it is one L2-resident vector, and
// a row's threads share the column).
__global__ void __launch_bounds__(kThreads) logits_scale_kernel(float* __restrict__ logits,
                                                                const uint16_t* __restrict__ logit_scale,
                                                                float factor, int64_t total,
                                                                int vocab) {
  dgpp_pdl_wait();
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; i < total; i += stride)
    logits[i] *= bf16_bits_to_float(logit_scale[i % vocab]) * factor;
}

}  // namespace

void dory_logits_scale_f32(float* logits, const uint16_t* logit_scale, float factor, int rows,
                           int vocab, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!logits || !logit_scale) throw std::invalid_argument("dory_logits_scale: null buffer");
  if (vocab <= 0) throw std::invalid_argument("dory_logits_scale: vocab must be positive");
  const int64_t total = static_cast<int64_t>(rows) * vocab;
  int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  if (blocks > 65535) blocks = 65535;
  logits_scale_kernel<<<blocks, kThreads, 0, stream>>>(logits, logit_scale, factor, total, vocab);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
