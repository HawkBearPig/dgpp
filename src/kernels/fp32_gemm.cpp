#include "kernels/fp32_gemm.hpp"

#include <stdexcept>

#include "common/cuda_check.hpp"

namespace dgpp {
Fp32Gemm::Fp32Gemm(void* workspace, size_t bytes, cudaStream_t stream) {
  DGPP_CUBLAS_OK(cublasCreate(&handle_), "FP32 handle");
  try {
    DGPP_CUBLAS_OK(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH), "strict FP32 math");
    DGPP_CUBLAS_OK(cublasSetStream(handle_, stream), "FP32 stream");
    DGPP_CUBLAS_OK(cublasSetWorkspace(handle_, workspace, bytes), "FP32 workspace");
  } catch (...) {
    cublasDestroy(handle_);
    throw;
  }
}
Fp32Gemm::~Fp32Gemm() {
  cublasDestroy(handle_);
}
void Fp32Gemm::matmul(const float* act, const float* weight, float* out, int m, int n, int k,
                      int stride) {
  if (m <= 0 || n <= 0 || k <= 0 || stride < k)
    throw std::invalid_argument("FP32 GEMM: invalid dimensions");
  const float alpha = 1, beta = 0;
  DGPP_CUBLAS_OK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, weight, k, act,
                             stride, &beta, out, n),
                 "FP32 SGEMM");
}
}  // namespace dgpp
