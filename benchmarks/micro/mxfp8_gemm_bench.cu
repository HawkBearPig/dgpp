// MiniMax projection shapes, rotating at least 64 MiB of weights so a
// repeated layer does not get an unrealistically warm weight-cache timing.
// These are isolated kernel timings, not deployment throughput.
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"
#include "kernels/mxfp8_gemm.hpp"

namespace {
template <class T>
struct Device {
  T* p = nullptr;
  explicit Device(size_t count) { DGPP_CUDA_OK(cudaMalloc(&p, count * sizeof(T))); }
  ~Device() { cudaFree(p); }
  Device(const Device&) = delete;
};
struct Graph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  ~Graph() {
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
  }
};

void measure(int rows, int n, int k, cudaStream_t stream) {
  const size_t weight_codes = static_cast<size_t>(n) * k;
  const size_t weight_scales = weight_codes / 32;
  const int rotations = std::max(4, static_cast<int>((64 * 1024 * 1024 + weight_codes +
                                                     weight_scales - 1) /
                                                    (weight_codes + weight_scales)));
  Device<uint8_t> a(static_cast<size_t>(rows) * k), as(static_cast<size_t>(rows) * k / 32),
      w(weight_codes * rotations), ws(weight_scales * rotations);
  Device<float> accum(static_cast<size_t>(rows) * n);
  Device<uint16_t> result(static_cast<size_t>(rows) * n);
  std::vector<uint8_t> payload(weight_codes * rotations);
  uint32_t state = 381;
  for (auto& value : payload) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    value = static_cast<uint8_t>((state % 112) | ((state >> 16) & 128));
  }
  DGPP_CUDA_OK(cudaMemcpy(w.p, payload.data(), payload.size(), cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemset(ws.p, 120, weight_scales * rotations));
  DGPP_CUDA_OK(cudaMemset(a.p, 0x35, static_cast<size_t>(rows) * k));
  DGPP_CUDA_OK(cudaMemset(as.p, 125, static_cast<size_t>(rows) * k / 32));
  std::vector<float> reference(static_cast<size_t>(rows) * n), actual(reference.size());
  std::vector<uint16_t> reference_bf16(reference.size()), actual_bf16(reference.size());
  Graph graphs[4];
  constexpr std::array<int, 4> tiles = {64, 32, 16, 8};
  for (int mode = 0; mode < 4; ++mode) {
    dgpp::launch_mxfp8_gemm_f32(a.p, as.p, w.p, ws.p, accum.p, rows, n, k, stream, 0, tiles[mode]);
    dgpp::launch_mxfp8_gemm_bf16(a.p, as.p, w.p, ws.p, result.p, rows, n, k, stream, 0, tiles[mode]);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    DGPP_CUDA_OK(cudaMemcpy(actual.data(), accum.p, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(actual_bf16.data(), result.p, actual_bf16.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    if (mode == 0) {
      reference = actual;
      reference_bf16 = actual_bf16;
    } else if (std::memcmp(actual.data(), reference.data(), actual.size() * sizeof(float)) ||
               actual_bf16 != reference_bf16) {
      throw std::runtime_error("projection tile changed output bits");
    }
    DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int rotation = 0; rotation < rotations; ++rotation)
      dgpp::launch_mxfp8_gemm_bf16(a.p, as.p, w.p + rotation * weight_codes,
                                  ws.p + rotation * weight_scales, result.p, rows, n, k,
                                  stream, 0, tiles[mode]);
    DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graphs[mode].graph));
    DGPP_CUDA_OK(cudaGraphInstantiate(&graphs[mode].exec, graphs[mode].graph, nullptr, nullptr, 0));
    DGPP_CUDA_OK(cudaGraphLaunch(graphs[mode].exec, stream));
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  cudaEvent_t begin, end;
  DGPP_CUDA_OK(cudaEventCreate(&begin));
  DGPP_CUDA_OK(cudaEventCreate(&end));
  std::vector<float> times[4];
  const int iterations = std::max(1, 32 / rotations);
  for (int trial = 0; trial < 5; ++trial)
    for (int turn = 0; turn < 4; ++turn) {
      const int mode = (trial + turn) % 4;
      DGPP_CUDA_OK(cudaEventRecord(begin, stream));
      for (int i = 0; i < iterations; ++i)
        DGPP_CUDA_OK(cudaGraphLaunch(graphs[mode].exec, stream));
      DGPP_CUDA_OK(cudaEventRecord(end, stream));
      DGPP_CUDA_OK(cudaEventSynchronize(end));
      float ms = 0;
      DGPP_CUDA_OK(cudaEventElapsedTime(&ms, begin, end));
      times[mode].push_back(ms * 1000 / (iterations * rotations));
    }
  for (auto& values : times) std::sort(values.begin(), values.end());
  std::printf("%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,exact\n", rows, n, k, rotations,
              times[0][2], times[1][2], times[2][2], times[3][2]);
  std::fflush(stdout);
  DGPP_CUDA_OK(cudaEventDestroy(begin));
  DGPP_CUDA_OK(cudaEventDestroy(end));
}
}  // namespace

int main() {
  try {
    cudaStream_t stream;
    DGPP_CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::puts("rows,n,k,weight_rotations,tile64_us,tile32_us,tile16_us,tile8_us,outputs");
    for (int rows : {1, 8, 16})
      for (const auto& shape : std::array<std::array<int, 2>, 7>{{
               {128, 6144}, {768, 6144}, {2048, 6144}, {3072, 6144},
               {6144, 768}, {6144, 2048}, {6144, 3072}}})
        measure(rows, shape[0], shape[1], stream);
    DGPP_CUDA_OK(cudaStreamDestroy(stream));
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "mxfp8_gemm_bench: %s\n", error.what());
    return 1;
  }
}
