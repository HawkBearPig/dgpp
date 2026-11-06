#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "engine/graph_pdl.hpp"
#include "kernels/minimax.hpp"
#include "kernels/pdl.cuh"

namespace {
// Permit the dependent to launch early, then hold back one producer row.
// The consumer must wait for the entire producer grid before reading it.
__global__ void delayed_rows(uint16_t* x, int rows, int hidden) {
  dgpp_pdl_wait();
  cudaTriggerProgrammaticLaunchCompletion();
  if (blockIdx.x == rows - 1) {
    if (threadIdx.x == 0) {
      const auto start = clock64();
      while (clock64() - start < 2000000) {
      }
    }
    __syncthreads();
  }
  for (int d = threadIdx.x; d < hidden; d += blockDim.x)
    x[static_cast<size_t>(blockIdx.x) * hidden + d] = dgpp::float_to_bf16_bits(2.0f);
}
}  // namespace

void minimax_pdl_test() {
  constexpr int rows = 128, hidden = 6144;
  constexpr size_t count = static_cast<size_t>(rows) * hidden;
  uint16_t *x = nullptr, *add = nullptr, *weight = nullptr, *out = nullptr;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  DGPP_CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DGPP_CUDA_OK(cudaMalloc(&x, count * 2));
  DGPP_CUDA_OK(cudaMalloc(&add, count * 2));
  DGPP_CUDA_OK(cudaMalloc(&out, count * 2));
  DGPP_CUDA_OK(cudaMalloc(&weight, hidden * 2));
  DGPP_CUDA_OK(cudaMemsetAsync(add, 0, count * 2, stream));
  DGPP_CUDA_OK(cudaMemsetAsync(weight, 0, hidden * 2, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  delayed_rows<<<rows, 256, 0, stream>>>(x, rows, hidden);
  dgpp::minimax_add_norm(x, add, weight, out, rows, hidden, 1e-6f, stream);
  DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
  const auto edges = dgpp::pdl::programmatic_edges(graph);
  if (edges.converted != 1)
    throw std::runtime_error("MiniMax PDL test did not convert its dependency");
  DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  bool correct = true;
  std::vector<uint16_t> values(count);
  for (int trial = 0; trial < 8; ++trial) {
    DGPP_CUDA_OK(cudaMemsetAsync(x, 0, count * 2, stream));
    DGPP_CUDA_OK(cudaMemsetAsync(out, 0, count * 2, stream));
    DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    DGPP_CUDA_OK(cudaMemcpy(values.data(), out, count * 2, cudaMemcpyDeviceToHost));
    for (auto value : values) correct &= value == dgpp::float_to_bf16_bits(1.0f);
  }
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaFree(x);
  cudaFree(add);
  cudaFree(out);
  cudaFree(weight);
  cudaStreamDestroy(stream);
  if (!correct)
    throw std::runtime_error("MiniMax kernel read a programmatic predecessor before completion");
}
