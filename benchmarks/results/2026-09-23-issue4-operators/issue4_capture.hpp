#pragma once
// Temporary, read-only diagnostic for the exact issue-4 forced-prefix row.
// Retained as an investigation patch; not intended for production integration.
#include <cuda_runtime.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "common/cuda_check.hpp"

namespace dgpp::issue4 {
inline bool active = false;
inline std::filesystem::path directory;
inline void context(int rank, int layer, int64_t pos0, int tokens, bool decode, bool capture) {
  active = !decode && !capture && pos0 + tokens == 261290;
  if (active) {
    directory = std::filesystem::path("/tmp/dgpp-issue4-operator-capture") /
        ("rank" + std::to_string(rank)) / ("layer" + std::to_string(layer));
    std::filesystem::create_directories(directory);
  }
}
template<class T> void write(const std::string& name, const std::vector<T>& values) {
  if (!active) return;
  const auto path = directory / (name + ".bin");
  if (std::filesystem::exists(path)) throw std::runtime_error("issue4 capture would overwrite " + path.string());
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
  if (!out) throw std::runtime_error("issue4 capture write failed");
}
template<class T> std::vector<T> read(const T* pointer, size_t count, cudaStream_t stream) {
  std::vector<T> values(count);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  DGPP_CUDA_OK(cudaMemcpy(values.data(), pointer, count * sizeof(T), cudaMemcpyDeviceToHost));
  return values;
}
template<class T> void dump(const std::string& name, const T* pointer, size_t count, cudaStream_t stream) {
  if (active) write(name, read(pointer, count, stream));
}
inline void meta(const std::string& name, const std::string& value) {
  if (!active) return;
  std::ofstream out(directory / (name + ".meta"));
  out << value << '\n';
  if (!out) throw std::runtime_error("issue4 metadata write failed");
}
}
