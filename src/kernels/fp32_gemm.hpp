#pragma once
#include <cstddef>
#include <cublas_v2.h>

#include <cuda_runtime.h>

namespace dgpp {
// Strict FP32 SGEMM for patch convolutions. The caller owns workspace and
// constructs this handle at startup, before fabric collective execution.
// cuBLASLt chooses a different reduction at some patch counts; SGEMM is
// the CUDA reference's F32 linear path.
class Fp32Gemm {
 public:
  Fp32Gemm(void* workspace, size_t bytes, cudaStream_t stream);
  ~Fp32Gemm();
  Fp32Gemm(const Fp32Gemm&) = delete;
  Fp32Gemm& operator=(const Fp32Gemm&) = delete;
  void matmul(const float* act, const float* weight, float* out, int m, int n, int k,
              int act_stride);

 private:
  cublasHandle_t handle_{};
};
}  // namespace dgpp
