// One rank's real-checkpoint MoE layer, with recorded BF16 normalized inputs.
// Compare scheduling strategies under CUDA graphs; these are kernel timings,
// not serving throughput or a deployment campaign.
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "models/minimax/layers.hpp"

namespace {
struct Device {
  uint16_t* p = nullptr;
  explicit Device(size_t elements) { DGPP_CUDA_OK(cudaMalloc(&p, elements * sizeof(uint16_t))); }
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
struct Events {
  cudaEvent_t begin = nullptr, end = nullptr;
  Events() {
    DGPP_CUDA_OK(cudaEventCreate(&begin));
    DGPP_CUDA_OK(cudaEventCreate(&end));
  }
  ~Events() {
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
    std::string checkpoint, input, dump_out;
    bool native = false;
    int rank = 0, world = 4, layer = 3, iterations = 50;
    const auto next = [&](int& i) {
      if (++i == argc) throw std::runtime_error("missing argument value");
      return std::string(argv[i]);
    };
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--checkpoint-dir")
        checkpoint = next(i);
      else if (arg == "--input")
        input = next(i);
      else if (arg == "--rank")
        rank = std::stoi(next(i));
      else if (arg == "--world")
        world = std::stoi(next(i));
      else if (arg == "--layer")
        layer = std::stoi(next(i));
      else if (arg == "--iterations")
        iterations = std::stoi(next(i));
      else if (arg == "--native-activations")
        native = true;
      else if (arg == "--dump-out")
        dump_out = next(i);
      else
        throw std::runtime_error("unknown option " + arg);
    }
    if (checkpoint.empty() || input.empty() || iterations < 1)
      throw std::runtime_error("supply --checkpoint-dir and --input (recorded ffn_norm BF16 rows)");
    const auto cfg = dgpp::MinimaxConfig::from_json_file(checkpoint + "/config.json");
    cfg.validate_tp(rank, world);
    if (layer < 0 || layer >= cfg.num_hidden_layers || !cfg.is_moe_layer(layer))
      throw std::runtime_error("--layer must select a routed layer");
    constexpr int max_rows = 64;
    const size_t elements = static_cast<size_t>(max_rows) * cfg.hidden_size;
    std::vector<uint16_t> x(elements);
    std::ifstream in(input, std::ios::binary);
    if (!in.read(reinterpret_cast<char*>(x.data()), elements * sizeof(uint16_t)))
      throw std::runtime_error("input must contain at least 64 complete BF16 hidden rows");
    const auto geometry = dgpp::MinimaxLocalGeometry::from_config(
        cfg, rank, world, dgpp::LoaderHeadSharding::VocabSharded);
    // Only this layer is loaded. Resident-image preparation and all source
    // reads finish before either graph is captured or timed.
    dgpp::MinimaxLayerStream loader(cfg, checkpoint, rank, world, dgpp::LoaderResidency::Resident,
                                    dgpp::LoaderHeadSharding::VocabSharded);
    const auto& weights = loader.load_layer(layer).moe_w;
    dgpp::MinimaxMoeLayer moe(cfg, geometry, max_rows, native);
    cudaStream_t stream = nullptr;
    DGPP_CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    moe.prepare_table(layer, weights, stream);
    Device dx(elements), dy(elements);
    DGPP_CUDA_OK(cudaMemcpy(dx.p, x.data(), elements * sizeof(uint16_t), cudaMemcpyHostToDevice));
    Events clock;
    std::printf("MiniMax MoE kernel diagnostic: rank=%d world=%d layer=%d hidden=%d inter=%lld activations=%s\n",
                rank, world, layer, cfg.hidden_size, static_cast<long long>(geometry.local_inter),
                native ? "native" : "bf16");
    if (!dump_out.empty()) std::filesystem::create_directories(dump_out);
    std::printf("rows,reference_us,compact_us,narrow_us,compact_speedup,narrow_speedup,outputs\n");
    for (int rows : {1, 2, 4, 8, 16, 17, 24, 32, 33, 64}) {
      Graph graphs[3];
      std::vector<uint16_t> reference(static_cast<size_t>(rows) * cfg.hidden_size);
      for (int mode = 0; mode < 3; ++mode) {
        for (int i = 0; i < 3; ++i)
          moe.enqueue(layer, weights, dx.p, rows, dy.p, stream, mode != 0, mode == 2);
        DGPP_CUDA_OK(cudaStreamSynchronize(stream));
        DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        moe.enqueue(layer, weights, dx.p, rows, dy.p, stream, mode != 0, mode == 2);
        DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graphs[mode].graph));
        DGPP_CUDA_OK(
            cudaGraphInstantiate(&graphs[mode].exec, graphs[mode].graph, nullptr, nullptr, 0));
        DGPP_CUDA_OK(cudaGraphLaunch(graphs[mode].exec, stream));
        DGPP_CUDA_OK(cudaStreamSynchronize(stream));
        std::vector<uint16_t> got(reference.size());
        DGPP_CUDA_OK(
            cudaMemcpy(got.data(), dy.p, got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        if (mode == 0)
          reference = std::move(got);
        else if (got != reference)
          throw std::runtime_error("expert scheduling changed a BF16 output");
      }
      if (!dump_out.empty()) {
        const auto path = std::filesystem::path(dump_out) /
                          ("layer" + std::to_string(layer) + "-rows" + std::to_string(rows) + ".bf16");
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(reference.data()), reference.size() * sizeof(uint16_t));
        if (!file) throw std::runtime_error("cannot write benchmark output");
      }
      std::vector<double> times[3];
      for (int trial = 0; trial < 5; ++trial)
        for (int turn = 0; turn < 3; ++turn) {
          const int mode = (trial + turn) % 3;
          times[mode].push_back(clock.time(graphs[mode].exec, stream, iterations));
        }
      for (auto& values : times) std::sort(values.begin(), values.end());
      std::printf("%d,%.3f,%.3f,%.3f,%.4f,%.4f,exact\n", rows, times[0][2], times[1][2],
                  times[2][2], times[0][2] / times[1][2], times[1][2] / times[2][2]);
      std::fflush(stdout);
    }
    DGPP_CUDA_OK(cudaStreamDestroy(stream));
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "minimax_moe_bench: %s\n", error.what());
    return 1;
  }
}
