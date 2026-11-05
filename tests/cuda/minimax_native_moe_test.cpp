#include "kernels/minimax_native_moe.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "engine/graph_pdl.hpp"
#include "kernels/moe_w4a4.hpp"

using namespace dgpp;
namespace fs = std::filesystem;
namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template <class T>
std::vector<T> read(const fs::path& path, size_t count) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  require(file.good() && file.tellg() == static_cast<std::streamoff>(count * sizeof(T)),
          "fixture file size");
  std::vector<T> values(count);
  file.seekg(0);
  file.read(reinterpret_cast<char*>(values.data()), count * sizeof(T));
  require(file.good(), "fixture read");
  return values;
}
template <class T>
void write(const fs::path& path, const std::vector<T>& values) {
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
  require(file.good(), "fixture output");
}
template <class T>
struct Device {
  T* p = nullptr;
  size_t count;
  explicit Device(size_t size) : count(size) {
    DGPP_CUDA_OK(cudaMalloc(&p, count * sizeof(T)));
    DGPP_CUDA_OK(cudaMemset(p, 0, count * sizeof(T)));
  }
  Device(const fs::path& path, size_t size) : Device(size) {
    const auto values = read<T>(path, size);
    DGPP_CUDA_OK(cudaMemcpy(p, values.data(), count * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~Device() { cudaFree(p); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  std::vector<T> get() const {
    std::vector<T> values(count);
    DGPP_CUDA_OK(cudaMemcpy(values.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost));
    return values;
  }
};

void check(const fs::path& folder) {
  int rows, hidden, inter, stride;
  std::ifstream params(folder / "params.txt");
  params >> rows >> hidden >> inter >> stride;
  require(params.good() && rows > 0 && hidden > 0 && stride >= hidden, "expert fixture parameters");
  Device<uint16_t> input(folder / "input.bf16", static_cast<size_t>(rows) * stride);
  Device<uint8_t> codes(static_cast<size_t>(rows) * hidden / 2),
      scales(static_cast<size_t>(rows) * hidden / 16);
  Device<float> globals(rows);
  cudaStream_t stream;
  DGPP_CUDA_OK(cudaStreamCreate(&stream));
  auto quantize = [&](int start, int count) {
    minimax_quantize_rows_nvfp4(input.p + static_cast<size_t>(start) * stride, stride, count,
                                hidden, codes.p + static_cast<size_t>(start) * hidden / 2,
                                scales.p + static_cast<size_t>(start) * hidden / 16,
                                globals.p + start, stream);
  };
  quantize(0, rows);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  const auto original_codes = codes.get(), original_scales = scales.get();
  const auto original_globals = globals.get();
  write(folder / "actual_codes.u8", original_codes);
  write(folder / "actual_scales.u8", original_scales);
  write(folder / "actual_globals.f32", original_globals);
  for (int chunk : {1, 8, 17}) {
    for (int start = 0; start < rows; start += chunk)
      quantize(start, std::min(chunk, rows - start));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    require(codes.get() == original_codes && scales.get() == original_scales &&
                globals.get() == original_globals,
            "expert quantizer row geometry");
  }
  if (inter == 0) {
    DGPP_CUDA_OK(cudaStreamDestroy(stream));
    std::printf("NVFP4 %s: quantizer row geometry exact\n", folder.filename().c_str());
    return;
  }
  Device<uint8_t> gate_w(folder / "gate.codes", static_cast<size_t>(inter) * hidden / 2);
  Device<uint8_t> gate_s(folder / "gate.scales", static_cast<size_t>(inter) * hidden / 16);
  Device<uint8_t> up_w(folder / "up.codes", gate_w.count), up_s(folder / "up.scales", gate_s.count);
  Device<uint8_t> down_w(folder / "down.codes", static_cast<size_t>(hidden) * inter / 2);
  Device<uint8_t> down_s(folder / "down.scales", static_cast<size_t>(hidden) * inter / 16);
  Device<float> weight_globals(folder / "weight_globals.f32", 3);
  Device<MoeExpertView> views(3);
  std::vector<MoeExpertView> host_views(3);
  for (int j = 0; j < 3; ++j) {
    host_views[j].payload = j == 0 ? gate_w.p : j == 1 ? up_w.p : down_w.p;
    host_views[j].fp4_scales = j == 0 ? gate_s.p : j == 1 ? up_s.p : down_s.p;
    host_views[j].fp4_global = weight_globals.p + j;
    host_views[j].fp4_group = kFp4Group;
  }
  DGPP_CUDA_OK(
      cudaMemcpy(views.p, host_views.data(), sizeof(MoeExpertView) * 3, cudaMemcpyHostToDevice));
  Device<MoeSegment> segment(1);
  Device<float> gate(static_cast<size_t>(rows) * inter), up(gate.count);
  Device<uint16_t> activation(gate.count), output(static_cast<size_t>(rows) * hidden);
  Device<uint8_t> ac(gate.count / 2), as(gate.count / 16);
  Device<float> ag(rows);
  auto run = [&](int start, int count) {
    quantize(start, count);
    const MoeSegment seg{0, count, 0};
    // Pageable-to-device metadata copies cannot be captured. The graph case
    // below prepares the full segment before capture.
    DGPP_CUDA_OK(cudaMemcpyAsync(segment.p, &seg, sizeof(seg), cudaMemcpyHostToDevice, stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    launch_moe_grouped_w4a4_f32(codes.p + static_cast<size_t>(start) * hidden / 2,
                                scales.p + static_cast<size_t>(start) * hidden / 16,
                                globals.p + start, nullptr, segment.p, 1, count, views.p, 0,
                                gate.p + static_cast<size_t>(start) * inter, inter, inter, hidden,
                                stream);
    launch_moe_grouped_w4a4_f32(codes.p + static_cast<size_t>(start) * hidden / 2,
                                scales.p + static_cast<size_t>(start) * hidden / 16,
                                globals.p + start, nullptr, segment.p, 1, count, views.p, 1,
                                up.p + static_cast<size_t>(start) * inter, inter, inter, hidden,
                                stream);
    minimax_swiglu_f32(gate.p + static_cast<size_t>(start) * inter,
                       up.p + static_cast<size_t>(start) * inter,
                       activation.p + static_cast<size_t>(start) * inter,
                       static_cast<int64_t>(count) * inter, 1.702f, 7.f, stream);
    minimax_quantize_rows_nvfp4(activation.p + static_cast<size_t>(start) * inter, inter, count,
                                inter, ac.p + static_cast<size_t>(start) * inter / 2,
                                as.p + static_cast<size_t>(start) * inter / 16, ag.p + start,
                                stream);
    launch_moe_grouped_w4a4_bf16(
        ac.p + static_cast<size_t>(start) * inter / 2,
        as.p + static_cast<size_t>(start) * inter / 16, ag.p + start, nullptr, segment.p, 1, count,
        views.p, 2, output.p + static_cast<size_t>(start) * hidden, hidden, hidden, inter, stream);
  };
  run(0, rows);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  const auto original_output = output.get(), original_activation = activation.get();
  const auto original_gate = gate.get(), original_up = up.get();
  write(folder / "actual_gate.f32", original_gate);
  write(folder / "actual_up.f32", original_up);
  write(folder / "actual_activation.bf16", original_activation);
  write(folder / "actual_act_codes.u8", ac.get());
  write(folder / "actual_act_scales.u8", as.get());
  write(folder / "actual_act_globals.f32", ag.get());
  write(folder / "actual_output.bf16", original_output);
  for (int chunk : {1, 8, 17}) {
    for (int start = 0; start < rows; start += chunk) run(start, std::min(chunk, rows - start));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    require(output.get() == original_output && activation.get() == original_activation &&
                gate.get() == original_gate && up.get() == original_up,
            "native expert row geometry");
  }
  // Graph replay of the new quantizer and activation producer, chained by
  // PDL edges. The grouped GEMMs are checked above in eager row geometries.
  cudaGraph_t graph;
  cudaGraphExec_t executable;
  DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  quantize(0, rows);
  minimax_swiglu_f32(gate.p, up.p, activation.p, static_cast<int64_t>(rows) * inter, 1.702f, 7.f,
                     stream);
  minimax_quantize_rows_nvfp4(activation.p, inter, rows, inter, ac.p, as.p, ag.p, stream);
  DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
  const auto pdl = dgpp::pdl::programmatic_edges(graph);
  require(pdl.converted > 0 && pdl.unlisted.empty(), "expert quantizer PDL graph coverage");
  DGPP_CUDA_OK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  for (int pass = 0; pass < 2; ++pass) {
    DGPP_CUDA_OK(cudaGraphLaunch(executable, stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    require(activation.get() == original_activation && codes.get() == original_codes,
            "expert graph replay");
  }
  DGPP_CUDA_OK(cudaGraphExecDestroy(executable));
  DGPP_CUDA_OK(cudaGraphDestroy(graph));
  DGPP_CUDA_OK(cudaStreamDestroy(stream));
  std::printf("NVFP4 %s: expert row geometry and PDL replay exact\n", folder.filename().c_str());
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: minimax_native_moe_test fixture-root");
    std::vector<fs::path> folders;
    for (const auto& entry : fs::directory_iterator(argv[1]))
      if (entry.is_directory() && fs::exists(entry.path() / "params.txt"))
        folders.push_back(entry.path());
    require(!folders.empty(), "no native expert fixtures");
    std::sort(folders.begin(), folders.end());
    for (const auto& folder : folders) check(folder);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "minimax_native_moe_test: %s\n", e.what());
    return 1;
  }
}
