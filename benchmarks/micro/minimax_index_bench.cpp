// Compare MiniMax index launch schedules under graphs. These are kernel-only
// diagnostics on synthetic BF16 keys, not deployment throughput measurements.
#include <algorithm>
#include <bit>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/minimax.hpp"

namespace {
template <class T>
struct Device {
  T* p = nullptr;
  size_t size;
  explicit Device(size_t n) : size(n) { DGPP_CUDA_OK(cudaMalloc(&p, n * sizeof(T))); }
  explicit Device(const std::vector<T>& v) : Device(v.size()) { upload(v); }
  ~Device() { cudaFree(p); }
  Device(const Device&) = delete;
  void upload(const std::vector<T>& v) {
    if (v.size() != size) throw std::runtime_error("upload size");
    DGPP_CUDA_OK(cudaMemcpy(p, v.data(), size * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> get() const {
    std::vector<T> v(size);
    DGPP_CUDA_OK(cudaMemcpy(v.data(), p, size * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }
};
struct Graph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  ~Graph() {
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
  }
};
struct Clock {
  cudaEvent_t begin = nullptr, end = nullptr;
  Clock() {
    DGPP_CUDA_OK(cudaEventCreate(&begin));
    DGPP_CUDA_OK(cudaEventCreate(&end));
  }
  ~Clock() {
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
  }
  double time(cudaGraphExec_t graph, cudaStream_t stream, int iterations) {
    DGPP_CUDA_OK(cudaEventRecord(begin, stream));
    for (int i = 0; i < iterations; ++i) DGPP_CUDA_OK(cudaGraphLaunch(graph, stream));
    DGPP_CUDA_OK(cudaEventRecord(end, stream));
    DGPP_CUDA_OK(cudaEventSynchronize(end));
    float ms = 0;
    DGPP_CUDA_OK(cudaEventElapsedTime(&ms, begin, end));
    return ms * 1000.0 / iterations;
  }
};
}  // namespace

int main(int argc, char** argv) {
  try {
    const int capacity = argc > 1 ? std::stoi(argv[1]) : 786432;
    const int iterations = argc > 2 ? std::stoi(argv[2]) : 5;
    if (argc > 3 || capacity < 32768 || capacity % 128 || iterations < 1)
      throw std::invalid_argument(
          "usage: minimax_index_bench [capacity >=32768, multiple of 128] [iterations]");
    const int blocks = capacity / 128;
    std::vector<uint16_t> keys(size_t(capacity) * 128);
    for (size_t i = 0; i < keys.size(); ++i)
      keys[i] = dgpp::float_to_bf16_bits(float(int((i * 17 + i / 131) % 131) - 65) / 32);
    Device<uint16_t> dk(keys);
    std::vector<uint16_t>().swap(keys);
    std::vector<int32_t> table(blocks);
    // A permutation for every supported capacity, with no contiguous-page assumption.
    for (int i = 0; i < blocks; ++i) table[i] = blocks - i - 1;
    Device<int32_t> dt(table);
    cudaStream_t stream;
    DGPP_CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Clock clock;
    std::puts(
        "capacity,visible,rows,original_us,bounded_us,tiled_us,tiled_speedup,visible_scores_and_"
        "selection");
    for (int rows : {1, 8, 64, 512}) {
      std::vector<uint16_t> query(size_t(rows) * 128);
      for (size_t i = 0; i < query.size(); ++i)
        query[i] = dgpp::float_to_bf16_bits(float(int((i * 13 + i / 43) % 97) - 48) / 16);
      Device<uint16_t> dq(query);
      Device<int32_t> dr(std::vector<int32_t>(rows, 0)), ids(size_t(rows) * 16);
      Device<int64_t> dp(rows);
      Device<float> scores(size_t(rows) * blocks);
      for (int visible : {128, 2048, 32768, 262144, 524288}) {
        if (visible > capacity || (rows > 64 && visible > 32768)) continue;
        std::vector<int64_t> pos(rows);
        for (int r = 0; r < rows; ++r) pos[r] = std::max(0, visible - rows + r);
        dp.upload(pos);
        Graph graphs[3];
        std::vector<float> expected;
        std::vector<int32_t> expected_ids;
        for (int mode = 0; mode < 3; ++mode) {
          // Poison future entries: the bounded scan must not make selection
          // depend on either uninitialized or previously visible score storage.
          scores.upload(std::vector<float>(scores.size, 1e30f));
          DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
          dgpp::minimax_index_score(dq.p, dk.p, dr.p, dp.p, dt.p, blocks, rows, 1, blocks, scores.p,
                                    stream, static_cast<dgpp::MinimaxIndexScan>(mode));
          dgpp::minimax_index_select(scores.p, dp.p, rows, 1, blocks, 16, ids.p, stream);
          DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graphs[mode].graph));
          DGPP_CUDA_OK(
              cudaGraphInstantiate(&graphs[mode].exec, graphs[mode].graph, nullptr, nullptr, 0));
          for (int i = 0; i < 3; ++i) DGPP_CUDA_OK(cudaGraphLaunch(graphs[mode].exec, stream));
          DGPP_CUDA_OK(cudaStreamSynchronize(stream));
          const auto got = scores.get();
          if (mode == 0) {
            expected = got;
            expected_ids = ids.get();
          } else {
            if (expected_ids != ids.get()) throw std::runtime_error("index selection changed");
            for (int r = 0; r < rows; ++r)
              for (int b = 0; b <= pos[r] / 128; ++b) {
                const size_t at = size_t(r) * blocks + b;
                if (std::bit_cast<uint32_t>(got[at]) != std::bit_cast<uint32_t>(expected[at]))
                  throw std::runtime_error("visible index score changed");
              }
          }
        }
        std::vector<double> times[3];
        for (int trial = 0; trial < 5; ++trial)
          for (int turn = 0; turn < 3; ++turn) {
            const int mode = (trial + turn) % 3;
            times[mode].push_back(clock.time(graphs[mode].exec, stream, iterations));
          }
        for (auto& t : times) std::sort(t.begin(), t.end());
        std::printf("%d,%d,%d,%.3f,%.3f,%.3f,%.4f,exact\n", capacity, visible, rows, times[0][2],
                    times[1][2], times[2][2], times[0][2] / times[2][2]);
        std::fflush(stdout);
      }
    }
    DGPP_CUDA_OK(cudaStreamDestroy(stream));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "minimax_index_bench: %s\n", e.what());
    return 1;
  }
}
