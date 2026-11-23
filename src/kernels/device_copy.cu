#include "kernels/device_copy.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "kernels/pdl.cuh"

namespace dgpp {
namespace {

template <typename Unit>
__global__ void device_copy_kernel(Unit* __restrict__ dst, const Unit* __restrict__ src, size_t units) {
  dgpp_pdl_wait();
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < units; i += stride) dst[i] = src[i];
}

template <typename Unit>
void launch(void* dst, const void* src, size_t bytes, unsigned max_blocks, cudaStream_t stream) {
  const size_t units = bytes / sizeof(Unit);
  const unsigned blocks = static_cast<unsigned>(std::min<size_t>((units + 255) / 256, max_blocks));
  device_copy_kernel<Unit><<<blocks, 256, 0, stream>>>(static_cast<Unit*>(dst), static_cast<const Unit*>(src), units);
  DGPP_CUDA_OK(cudaGetLastError());
}

bool whole(const void* dst, const void* src, size_t bytes, size_t unit) {
  return bytes % unit == 0 && reinterpret_cast<uintptr_t>(dst) % unit == 0 && reinterpret_cast<uintptr_t>(src) % unit == 0;
}

}  // namespace

void device_copy(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
  if (dst == nullptr || src == nullptr || bytes == 0)
    throw std::invalid_argument("device_copy: null buffer or empty copy");
  if (whole(dst, src, bytes, 16)) launch<uint4>(dst, src, bytes, 1024, stream);
  else if (whole(dst, src, bytes, 4)) launch<uint32_t>(dst, src, bytes, 4096, stream);
  else launch<uint8_t>(dst, src, bytes, 4096, stream);
}

}  // namespace dgpp
