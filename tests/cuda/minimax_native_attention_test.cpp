#include "kernels/minimax_native_attention.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "engine/graph_pdl.hpp"
#include "kernels/minimax.hpp"
#include "kernels/sink_attn.hpp"

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

void check(const fs::path& folder, bool decode) {
  int rows, appends, requests, blocks, heads, kv_heads, topk;
  std::ifstream params(folder / "params.txt");
  params >> rows >> appends >> requests >> blocks >> heads >> kv_heads >> topk;
  require(params.good() && rows > 0 && appends > 0 && requests > 0 && blocks > 0,
          "attention fixture parameters");
  const size_t cache_tokens = static_cast<size_t>(requests) * blocks * 128;
  Device<uint16_t> input_q(folder / "q.bf16", static_cast<size_t>(rows) * heads * 128);
  Device<uint16_t> input_iq(folder / "iq.bf16", static_cast<size_t>(rows) * kv_heads * 128);
  Device<uint16_t> input_k(folder / "k.bf16", static_cast<size_t>(appends) * kv_heads * 128);
  Device<uint16_t> input_v(folder / "v.bf16", input_k.count);
  Device<uint16_t> input_ik(folder / "ik.bf16", static_cast<size_t>(appends) * 128);
  Device<int32_t> req(folder / "req.i32", rows), table(folder / "table.i32", requests * blocks);
  Device<int64_t> positions(folder / "positions.i64", rows);
  Device<int32_t> append_req(folder / "append_req.i32", appends);
  Device<int64_t> append_pos(folder / "append_pos.i64", appends);
  Device<uint8_t> q(input_q.count), iq(input_iq.count);
  Device<uint8_t> k(cache_tokens * kv_heads * 72), v(k.count), ik(cache_tokens * 72);
  Device<float> scores(static_cast<size_t>(rows) * kv_heads * blocks);
  Device<int32_t> selected(static_cast<size_t>(rows) * kv_heads * topk);
  Device<float> maximum(static_cast<size_t>(rows) * topk * heads), total(maximum.count),
      partial(maximum.count * 128);
  Device<uint16_t> out(input_q.count);
  cudaStream_t stream;
  DGPP_CUDA_OK(cudaStreamCreate(&stream));
  auto append = [&] {
    minimax_cache_nvfp4_append(input_k.p, k.p, append_req.p, append_pos.p, table.p, blocks, appends,
                               kv_heads, stream);
    minimax_cache_nvfp4_append(input_v.p, v.p, append_req.p, append_pos.p, table.p, blocks, appends,
                               kv_heads, stream);
    minimax_cache_nvfp4_append(input_ik.p, ik.p, append_req.p, append_pos.p, table.p, blocks,
                               appends, 1, stream);
  };
  auto run = [&](int start, int count) {
    minimax_query_fp8(input_q.p + start * heads * 128, q.p + start * heads * 128,
                      static_cast<int64_t>(count) * heads * 128, stream);
    minimax_query_fp8(input_iq.p + start * kv_heads * 128, iq.p + start * kv_heads * 128,
                      static_cast<int64_t>(count) * kv_heads * 128, stream);
    minimax_index_nvfp4_score(iq.p + start * kv_heads * 128, ik.p, req.p + start,
                              positions.p + start, table.p, blocks, count, kv_heads, blocks,
                              scores.p + start * kv_heads * blocks, stream);
    minimax_index_select(scores.p + start * kv_heads * blocks, positions.p + start, count, kv_heads,
                         blocks, topk, selected.p + start * kv_heads * topk, stream);
    const size_t off = static_cast<size_t>(start) * topk * heads;
    if (decode) {
      minimax_attn_nvfp4_decode(q.p + start * heads * 128, k.p, v.p, req.p + start,
                                 positions.p + start, table.p, blocks,
                                 selected.p + start * kv_heads * topk, count, heads, kv_heads, topk,
                                 out.p + start * heads * 128, stream);
      return;
    }
    minimax_attn_nvfp4_partial(q.p + start * heads * 128, k.p, v.p, req.p + start,
                               positions.p + start, table.p, blocks,
                               selected.p + start * kv_heads * topk, count, heads, kv_heads, topk,
                               maximum.p + off, total.p + off, partial.p + off * 128, stream);
    sink_attn_combine(maximum.p + off, total.p + off, partial.p + off * 128, count, topk, heads,
                      out.p + start * heads * 128, stream);
  };
  append();
  run(0, rows);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  const auto original = out.get();
  const auto original_scores = scores.get();
  const auto original_selected = selected.get();
  write(folder / "actual_q.u8", q.get());
  write(folder / "actual_iq.u8", iq.get());
  write(folder / "actual_k.u8", k.get());
  write(folder / "actual_v.u8", v.get());
  write(folder / "actual_ik.u8", ik.get());
  write(folder / "actual_scores.f32", original_scores);
  write(folder / "actual_selected.i32", original_selected);
  write(folder / (decode ? "actual_decode.bf16" : "actual_out.bf16"), original);
  for (int chunk : {1, 8, 17}) {
    for (int start = 0; start < rows; start += chunk) run(start, std::min(chunk, rows - start));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    require(out.get() == original, "native attention row geometry");
    require(scores.get() == original_scores, "native index row geometry");
    require(selected.get() == original_selected, "native selection row geometry");
  }
  cudaGraph_t graph;
  cudaGraphExec_t executable;
  DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  append();
  run(0, rows);
  DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
  const auto pdl = dgpp::pdl::programmatic_edges(graph);
  require(pdl.converted > 0 && pdl.unlisted.empty(), "native attention PDL graph coverage");
  DGPP_CUDA_OK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  for (int pass = 0; pass < 2; ++pass) {
    DGPP_CUDA_OK(cudaGraphLaunch(executable, stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    require(out.get() == original, "native attention graph replay");
    require(scores.get() == original_scores && selected.get() == original_selected,
            "native index graph replay");
  }
  DGPP_CUDA_OK(cudaGraphExecDestroy(executable));
  DGPP_CUDA_OK(cudaGraphDestroy(graph));
  DGPP_CUDA_OK(cudaStreamDestroy(stream));
  std::printf("Q8KV4 %s %s: row geometry and PDL graph replay exact\n",
              folder.filename().c_str(), decode ? "decode" : "prefill");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: minimax_native_attention_test fixture-root");
    std::vector<fs::path> folders;
    for (const auto& entry : fs::directory_iterator(argv[1]))
      if (entry.is_directory() && fs::exists(entry.path() / "params.txt"))
        folders.push_back(entry.path());
    require(!folders.empty(), "no native attention fixtures");
    std::sort(folders.begin(), folders.end());
    for (const auto& folder : folders)
      for (const bool decode : {false, true}) check(folder, decode);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "minimax_native_attention_test: %s\n", e.what());
    return 1;
  }
}
