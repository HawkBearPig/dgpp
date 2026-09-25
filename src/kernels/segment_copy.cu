#include "kernels/segment_copy.hpp"

#include <stdexcept>

#include "common/cuda_check.hpp"

namespace dgpp {

namespace {

__global__ void segment_copy_kernel(const CopySegment* __restrict__ segs, int first, int n) {
  const int i = first + blockIdx.x;
  if (i >= n) return;
  const CopySegment s = segs[i];
  const uint8_t* src = static_cast<const uint8_t*>(s.src);
  uint8_t* dst = static_cast<uint8_t*>(s.dst);
  const uint64_t bytes = s.bytes;
  const bool vec = (bytes & 15u) == 0 && (reinterpret_cast<uintptr_t>(src) & 15u) == 0 &&
                   (reinterpret_cast<uintptr_t>(dst) & 15u) == 0;
  if (vec) {
    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    uint4* d4 = reinterpret_cast<uint4*>(dst);
    const uint64_t n4 = bytes / 16;
    for (uint64_t k = threadIdx.x; k < n4; k += blockDim.x) d4[k] = s4[k];
  } else {
    for (uint64_t k = threadIdx.x; k < bytes; k += blockDim.x) dst[k] = src[k];
  }
}

}  // namespace

void segment_copy(const CopySegment* segs, int n, cudaStream_t stream) {
  if (n <= 0) return;
  if (segs == nullptr) throw std::invalid_argument("segment_copy: null segment table");
  constexpr int kMaxGrid = 65535;
  for (int first = 0; first < n; first += kMaxGrid) {
    const int count = n - first < kMaxGrid ? n - first : kMaxGrid;
    segment_copy_kernel<<<count, 256, 0, stream>>>(segs, first, n);
    DGPP_CUDA_OK(cudaGetLastError());
  }
}

}  // namespace dgpp
