// Real-checkpoint vocabulary-head diagnostics: scalar and tensor-core reductions.
// Requires exact row-geometry invariance and a sampled FP64 oracle before timing.
// These are kernel timings, not deployment throughput.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/bf12_companions.hpp"
#include "kernels/gemm.hpp"
#include "models/minimax/layers.hpp"

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
    if (argc < 3 || argc > 5)
      throw std::invalid_argument(
          "usage: minimax_head_bench CHECKPOINT RECORDED_BF16_ROWS [iterations] [bf12]");
    const std::string checkpoint = argv[1];
    const int iterations = argc >= 4 ? std::stoi(argv[3]) : 5;
    const bool packed = argc == 5 && std::string(argv[4]) == "bf12";
    if (argc == 5 && !packed) throw std::invalid_argument("unknown storage mode");
    if (iterations < 1) throw std::invalid_argument("iterations must be positive");
    const auto cfg = dgpp::MinimaxConfig::from_json_file(checkpoint + "/config.json");
    dgpp::MinimaxLayerStream loader(cfg, checkpoint, 0, 4, dgpp::LoaderResidency::Resident,
                                    dgpp::LoaderHeadSharding::VocabSharded);
    const auto& globals = loader.load_globals();
    const int k = cfg.hidden_size, n = globals.lm_vocab_count, max_rows = 64;
    std::vector<uint16_t> x(size_t(max_rows) * k);
    std::ifstream input(argv[2], std::ios::binary);
    if (!input.read(reinterpret_cast<char*>(x.data()), x.size() * 2))
      throw std::runtime_error("input must contain 64 complete BF16 hidden rows");
    Device<uint16_t> dx(x);
    Device<float> dy(size_t(max_rows) * n);
    Device<char> workspace(size_t(64) << 20);
    cudaStream_t stream;
    DGPP_CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    dgpp::CublasLtGemm gemm[2];
    dgpp::Bf12Companions companion;
    if (packed) {
      if (!companion.pack(globals.lm_head, n, k, gemm[0], stream))
        throw std::runtime_error("real head declined BF12 packing");
      gemm[1].register_bf12(globals.lm_head, *gemm[0].bf12_lookup(globals.lm_head));
    }
    for (int mode = 0; mode < 2; ++mode) {
      gemm[mode].set_decode_rows(max_rows);
      gemm[mode].set_kernel_only_rows(1, max_rows);
      gemm[mode].set_decode_mma(mode == 1);
      gemm[mode].set_bf12_wide(true);
    }
    auto launch = [&](int mode, int rows) {
      // The packed tensor-core form supports up to 32 rows per launch.
      const int chunk = packed && mode == 1 ? 32 : rows;
      for (int at = 0; at < rows; at += chunk)
        gemm[mode].matmul(dx.p + size_t(at) * k, globals.lm_head, dy.p + size_t(at) * n,
                          std::min(chunk, rows - at), n, k, dgpp::DType::BF16, dgpp::GemmOut::F32,
                          k, workspace.p, workspace.size, stream);
    };
    std::vector<float> complete[2];
    for (int mode = 0; mode < 2; ++mode) {
      launch(mode, max_rows);
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      complete[mode] = dy.get();
      uint64_t digest = 1469598103934665603ull;
      for (float value : complete[mode]) {
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite vocabulary logit");
        digest = (digest ^ std::bit_cast<uint32_t>(value)) * 1099511628211ull;
      }
      std::printf("mode %d: output_digest %016llx\n", mode,
                  static_cast<unsigned long long>(digest));
    }
    // FP64 dot products over samples spanning the real vocabulary shard.
    // The absolute bound accounts for cancellation; the relative RMS bound
    // is the existing compact-head oracle's limit.
    constexpr int samples = 128;
    std::vector<uint16_t> w(size_t(samples) * k);
    for (int sample = 0; sample < samples; ++sample) {
      const int token = sample * (n - 1) / (samples - 1);
      DGPP_CUDA_OK(cudaMemcpy(w.data() + size_t(sample) * k, globals.lm_head + size_t(token) * k,
                              size_t(k) * 2, cudaMemcpyDeviceToHost));
    }
    double norm = 0, errors[2]{};
    const double roundoff = std::ldexp(1.0, -24), gamma = k * roundoff / (1 - k * roundoff);
    for (int row = 0; row < max_rows; ++row)
      for (int sample = 0; sample < samples; ++sample) {
        double expected = 0, absolute = 0;
        for (int d = 0; d < k; ++d) {
          const double product = double(dgpp::bf16_bits_to_float(x[size_t(row) * k + d])) *
                                 dgpp::bf16_bits_to_float(w[size_t(sample) * k + d]);
          expected += product;
          absolute += std::abs(product);
        }
        norm += expected * expected;
        const int token = sample * (n - 1) / (samples - 1);
        for (int mode = 0; mode < 2; ++mode) {
          const double error = double(complete[mode][size_t(row) * n + token]) - expected;
          if (!std::isfinite(error) || std::abs(error) > gamma * absolute)
            throw std::runtime_error("head exceeds FP32 forward-error bound against FP64");
          errors[mode] += error * error;
        }
      }
    for (int mode = 0; mode < 2; ++mode) {
      const double rms = std::sqrt(errors[mode] / norm);
      if (!(rms < 2e-5)) throw std::runtime_error("head exceeds FP64 relative RMS limit");
      std::printf("mode %d: FP64 relative RMS %.9g\n", mode, rms);
    }
    double maximum = 0;
    int changed = 0;
    for (size_t i = 0; i < complete[0].size(); ++i)
      maximum = std::max(maximum, std::abs(double(complete[0][i]) - complete[1][i]));
    for (int r = 0; r < max_rows; ++r) {
      auto a = complete[0].begin() + size_t(r) * n, b = complete[1].begin() + size_t(r) * n;
      changed += std::max_element(a, a + n) - a != std::max_element(b, b + n) - b;
    }
    std::printf(
        "rank=0 world=4 columns=%d inner=%d rows=%d bf12=%d max_logit_delta=%.9g "
        "argmax_changes=%d\n",
        n, k, max_rows, int(packed), maximum, changed);
    Clock clock;
    std::puts("rows,scalar_us,mma_us,speedup,row_geometry");
    for (int rows : {1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64}) {
      Graph graphs[2];
      for (int mode = 0; mode < 2; ++mode) {
        DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch(mode, rows);
        DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graphs[mode].graph));
        DGPP_CUDA_OK(
            cudaGraphInstantiate(&graphs[mode].exec, graphs[mode].graph, nullptr, nullptr, 0));
        for (int i = 0; i < 3; ++i) DGPP_CUDA_OK(cudaGraphLaunch(graphs[mode].exec, stream));
        DGPP_CUDA_OK(cudaStreamSynchronize(stream));
        const auto got = dy.get();
        if (std::memcmp(got.data(), complete[mode].data(), size_t(rows) * n * sizeof(float)))
          throw std::runtime_error("head output changed with row geometry");
      }
      std::vector<double> times[2];
      for (int trial = 0; trial < 5; ++trial)
        for (int turn = 0; turn < 2; ++turn) {
          const int mode = (trial + turn) % 2;
          times[mode].push_back(clock.time(graphs[mode].exec, stream, iterations));
        }
      for (auto& t : times) std::sort(t.begin(), t.end());
      std::printf("%d,%.3f,%.3f,%.4f,exact\n", rows, times[0][2], times[1][2],
                  times[0][2] / times[1][2]);
      std::fflush(stdout);
    }
    DGPP_CUDA_OK(cudaStreamDestroy(stream));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "minimax_head_bench: %s\n", e.what());
    return 1;
  }
}
