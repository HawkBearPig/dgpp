#include <stdexcept>
#include "kernels/pdl.cuh"

#include "common/cuda_check.hpp"
#include "kernels/mimo_mtp_history.hpp"
namespace dgpp {
namespace {
__global__ void mtp_history_gather(const uint16_t* history, uint16_t* hidden,
                                   const int64_t* positions, int64_t* translated,
                                   const int32_t* ids, int width, int capacity, int layer) {
  dgpp_pdl_wait();
  const int row = blockIdx.x;
  const int req = ids ? ids[row] : 0;
  const int64_t p = positions[row];
  const bool valid = req >= 0 && p >= layer;
  if (threadIdx.x == 0) translated[row] = valid ? p - layer : -1;
  if (!history) return;
  for (int i = threadIdx.x; i < width; i += blockDim.x)
    hidden[size_t(row) * width + i] =
        valid ? history[(size_t(req) * capacity + (p - layer) % capacity) * width + i] : 0;
}
__global__ void mtp_history_store(uint16_t* history, const uint16_t* hidden,
                                  const int64_t* positions, const int32_t* ids, int width,
                                  int capacity, int layer) {
  dgpp_pdl_wait();
  const int row = blockIdx.x;
  const int req = ids ? ids[row] : 0;
  const int64_t p = positions[row];
  if (req < 0 || p < layer) return;
  for (int i = threadIdx.x; i < width; i += blockDim.x)
    history[(size_t(req) * capacity + p % capacity) * width + i] = hidden[size_t(row) * width + i];
}
}  // namespace
void mimo_mtp_history_gather(const uint16_t* history, uint16_t* hidden, const int64_t* positions,
                             int64_t* translated, const int32_t* ids, int rows, int width,
                             int capacity, int layer, cudaStream_t stream) {
  if (rows < 1 || width < 1 || capacity <= rows || layer < 0 || layer > 2)
    throw std::invalid_argument("MiMo MTP history gather shape");
  mtp_history_gather<<<rows, 256, 0, stream>>>(history, hidden, positions, translated, ids, width,
                                               capacity, layer);
  DGPP_CUDA_OK(cudaGetLastError());
}
void mimo_mtp_history_store(uint16_t* history, const uint16_t* hidden, const int64_t* positions,
                            const int32_t* ids, int rows, int width, int capacity, int layer,
                            cudaStream_t stream) {
  if (rows < 1 || width < 1 || capacity <= rows || layer < 0 || layer > 2)
    throw std::invalid_argument("MiMo MTP history store shape");
  mtp_history_store<<<rows, 256, 0, stream>>>(history, hidden, positions, ids, width, capacity,
                                              layer);
  DGPP_CUDA_OK(cudaGetLastError());
}
}  // namespace dgpp
