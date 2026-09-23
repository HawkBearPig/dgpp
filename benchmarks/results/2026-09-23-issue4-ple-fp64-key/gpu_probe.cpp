// Bounded PLE diagnostic: real operands at their original 2048-row placement.
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "common/dtypes.hpp"
#include "kernels/gemm.hpp"
#include "kernels/qwen_ple_fp64.hpp"
#include "kernels/qwen_norm.hpp"
#include "kernels/qwen_ple.hpp"

namespace fs = std::filesystem;
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> std::vector<T> read(const fs::path& path, size_t count) {
  if (fs::file_size(path) != count * sizeof(T)) throw std::runtime_error("Wrong input size: " + path.string());
  std::vector<T> data(count);
  std::ifstream stream(path, std::ios::binary);
  stream.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(T)));
  if (!stream) throw std::runtime_error("Input read failed");
  return data;
}
template<class T> void write(const fs::path& path, const std::vector<T>& data) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
  if (!stream) throw std::runtime_error("Output write failed");
}
struct Buffer {
  void* data{};
  explicit Buffer(size_t bytes) { check(cudaMalloc(&data, bytes)); }
  ~Buffer() { cudaFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  template<class T> void upload(const std::vector<T>& values) {
    check(cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  template<class T> std::vector<T> download(size_t count) {
    std::vector<T> values(count);
    check(cudaMemcpy(values.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
    return values;
  }
};
int main(int argc, char** argv) {
  try {
    if (argc != 4 && argc != 5) throw std::runtime_error("usage: gpu_probe INPUT OUTPUT {bf16|fp64} [ROWS]");
    const fs::path input(argv[1]), output(argv[2]);
    const bool fp64 = std::string(argv[3]) == "fp64";
    if (!fp64 && std::string(argv[3]) != "bf16") throw std::runtime_error("Unknown mode");
    if (fs::exists(output)) throw std::runtime_error("Refusing existing output");
    fs::create_directories(output);
    constexpr int E = 1280, H = 2560, W = 10240;
    const int M = argc == 5 ? std::stoi(argv[4]) : 2048;
    if (M != 2048 && M != 1024) throw std::runtime_error("Unsupported row count");
    cudaStream_t stream{};
    check(cudaStreamCreate(&stream));
    dgpp::CublasLtGemm gemm;
    dgpp::QwenPleFp64Key accurate(M, W, E);
    Buffer workspace(64u << 20), activation(static_cast<size_t>(M) * E * 2);
    Buffer matrix(static_cast<size_t>(W) * E * 2), product(static_cast<size_t>(M) * W * 4);
    Buffer key(static_cast<size_t>(M) * W * 2), value(static_cast<size_t>(M) * H * 2);
    for (const std::string name : {"key_proj", "value_proj"}) {
      const int width = name == "key_proj" ? W : H;
      std::vector<uint16_t> parts[2];
      for (int rank = 0; rank < 2; ++rank) {
        auto embedding = read<uint16_t>(input / ("embedding" + std::to_string(rank) + ".bin"), size_t(2048) * E);
        embedding.resize(size_t(M) * E);
        activation.upload(embedding);
        matrix.upload(read<uint16_t>(input / (name + std::to_string(rank) + ".bin"), static_cast<size_t>(width) * E));
        if (fp64 && name == "key_proj") {
          accurate.project(static_cast<const uint16_t*>(activation.data), static_cast<const uint16_t*>(matrix.data),
                           static_cast<uint16_t*>(product.data), M, workspace.data, 64u << 20, stream);
        } else {
          gemm.matmul(activation.data, matrix.data, product.data, M, width, E, dgpp::DType::BF16,
                      dgpp::GemmOut::BF16, E, workspace.data, 64u << 20, stream);
        }
        check(cudaStreamSynchronize(stream));
        const size_t count = static_cast<size_t>(M) * width;
        parts[rank] = product.download<uint16_t>(count);
        write(output / (name + std::to_string(rank) + ".bin"), parts[rank]);
      }
      std::vector<uint16_t> folded(parts[0].size());
      for (size_t i = 0; i < folded.size(); ++i)
        folded[i] = dgpp::float_to_bf16_bits(dgpp::bf16_bits_to_float(parts[0][i]) + dgpp::bf16_bits_to_float(parts[1][i]));
      write(output / (name + "_folded.bin"), folded);
      (name == "key_proj" ? key : value).upload(folded);
    }
    Buffer residual(static_cast<size_t>(M) * W * 2), norm_weight(W * 2);
    Buffer kn(static_cast<size_t>(M) * W * 2), qn(static_cast<size_t>(M) * W * 2);
    Buffer gated(static_cast<size_t>(M) * W * 2), normalized(static_cast<size_t>(M) * W * 2);
    auto host_residual = read<uint16_t>(input / "residual.bin", size_t(2048) * W);
    host_residual.resize(size_t(M) * W);
    residual.upload(host_residual);
    norm_weight.upload(read<uint16_t>(input / "norm_key.bin", W));
    dgpp::qwen_group_rmsnorm_bf16(key.data, norm_weight.data, kn.data, M, 4, H, 1e-6f, stream);
    check(cudaStreamSynchronize(stream));
    norm_weight.upload(read<uint16_t>(input / "norm_query.bin", W));
    dgpp::qwen_group_rmsnorm_bf16(residual.data, norm_weight.data, qn.data, M, 4, H, 1e-6f, stream);
    dgpp::qwen_ple_gate_bf16(static_cast<const uint16_t*>(kn.data), static_cast<const uint16_t*>(qn.data),
                            static_cast<const uint16_t*>(value.data), static_cast<uint16_t*>(gated.data), M, 4, H, stream);
    check(cudaStreamSynchronize(stream));
    norm_weight.upload(read<uint16_t>(input / "norm_conv.bin", W));
    dgpp::qwen_group_rmsnorm_bf16(gated.data, norm_weight.data, normalized.data, M, 4, H, 1e-6f, stream);
    check(cudaStreamSynchronize(stream));
    write(output / "key_norm.bin", kn.download<uint16_t>(static_cast<size_t>(M) * W));
    write(output / "query_norm.bin", qn.download<uint16_t>(static_cast<size_t>(M) * W));
    write(output / "gated.bin", gated.download<uint16_t>(static_cast<size_t>(M) * W));
    write(output / "conv_norm.bin", normalized.download<uint16_t>(static_cast<size_t>(M) * W));
    check(cudaStreamDestroy(stream));
    std::cout << "Complete " << argv[3] << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
