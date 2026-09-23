#pragma once
// Read-only diagnostic: actual routed-expert boundaries at selected fixture rows.
#include <cuda_runtime.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "common/cuda_check.hpp"

namespace dgpp::issue4_moe {
inline bool active = false;
inline std::filesystem::path directory;
inline std::vector<int> rows;
inline void context(int rank, int layer, int64_t pos, int tokens, bool decode, bool capture) {
  active = !decode && !capture && ((pos == 0 && tokens == 2048) || pos + tokens == 261290);
  if (!active) return;
  rows = {0, tokens / 2, tokens - 1};
  directory = std::filesystem::path("/tmp/dgpp-issue4-moe-capture-20260923") /
      ("rank" + std::to_string(rank)) / ("position" + std::to_string(pos)) /
      ("layer" + std::to_string(layer));
  std::filesystem::create_directories(directory);
}
template <class T> void write(const std::string& name, const std::vector<T>& values) {
  if (!active) return;
  const auto path = directory / (name + ".bin");
  if (std::filesystem::exists(path)) throw std::runtime_error("capture would overwrite " + path.string());
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
  if (!out) throw std::runtime_error("MoE capture write failed");
}
template <class T> std::vector<T> read(const T* pointer, size_t count, cudaStream_t stream) {
  std::vector<T> values(count);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  DGPP_CUDA_OK(cudaMemcpy(values.data(), pointer, count * sizeof(T), cudaMemcpyDeviceToHost));
  return values;
}
template <class T> void dump(const std::string& name, const T* pointer, size_t count, cudaStream_t stream) {
  if (active) write(name, read(pointer, count, stream));
}
template <class T> void sample(const std::string& name, const T* pointer, size_t width, cudaStream_t stream) {
  if (active) for (int row : rows) dump("row" + std::to_string(row) + "_" + name,
                                     pointer + static_cast<size_t>(row) * width, width, stream);
}
template <class T> void mapped(const std::string& name, const T* pointer, size_t width,
                              const std::vector<int32_t>& indices, cudaStream_t stream) {
  std::vector<T> values;
  for (int index : indices) {
    auto row = read(pointer + static_cast<size_t>(index) * width, width, stream);
    values.insert(values.end(), row.begin(), row.end());
  }
  write(name, values);
}
}
