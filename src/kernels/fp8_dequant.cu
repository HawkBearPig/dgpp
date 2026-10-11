#include "kernels/fp8_dequant.hpp"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {

constexpr int kThreads = 256;

// One thread per output element: memory-bound by construction (one u8 and
// one bf16 per element plus a shared 512-byte scale row per block), so the
// scalar addressing costs nothing that matters. Vectorize only if a
// production path starts calling this in a hot loop.
//
// rs / cs: the scale grid as log2 block sizes, matching kernels/scale_gemm
// (7,7 = the 128x128 checkpoint grid; 0 / log2ceil(cols) = a channel matrix
// — one scale a row, the Qwen3.8-27B-NVFP4 release's attention half).
__global__ void fp8_dequant_blocks_kernel(const uint8_t* __restrict__ payload,
                                          const float* __restrict__ scales,
                                          uint16_t* __restrict__ out,
                                          int64_t rows, int64_t cols,
                                          int64_t scale_cols, int rs, int cs) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  const int64_t total = rows * cols;
  if (i >= total) return;
  const int64_t n = i / cols;
  const int64_t k = i - n * cols;
  const float v =
      fp8_e4m3_bits_to_float(payload[i]) *
      scales[((n >> rs) * scale_cols) + (k >> cs)];
  out[i] = float_to_bf16_bits(v);
}

}  // namespace

void launch_fp8_dequant_blocks(const uint8_t* payload, const float* scales,
                               uint16_t* out_bf16, int64_t rows, int64_t cols,
                               cudaStream_t stream, int rs, int cs) {
  if (rows <= 0 || cols <= 0)
    throw std::invalid_argument("fp8_dequant: rows and cols must be positive");
  if (!payload || !scales || !out_bf16)
    throw std::invalid_argument("fp8_dequant: null pointer");
  if (rs < 0 || cs < 4 || (1 << cs) % 16 != 0)
    throw std::invalid_argument("fp8_dequant: rs must be non-negative, cs >= 4");
  const int64_t total = rows * cols;
  const int64_t scale_cols = (cols + (1LL << cs) - 1) >> cs;
  if ((rows + (1LL << rs) - 1) >> rs > static_cast<int64_t>(1) << 31)
    throw std::runtime_error("fp8_dequant: scale grid too large");
  const int64_t blocks = (total + kThreads - 1) / kThreads;
  if (blocks > 2147483647LL)
    throw std::runtime_error("fp8_dequant: grid too large");
  fp8_dequant_blocks_kernel<<<static_cast<int>(blocks), kThreads, 0,
                              stream>>>(payload, scales, out_bf16, rows, cols,
                                        scale_cols, rs, cs);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
