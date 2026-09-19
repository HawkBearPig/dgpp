#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>

#include "common/cuda_check.hpp"
#include "models/glm/config.hpp"
#include "models/glm/vision.hpp"

int main(int argc, char** argv) {
  if (argc < 5 || argc > 8) {
    std::cerr << "usage: glm_vision_check CHECKPOINT WIDTH HEIGHT OUTPUT.bf16 [TRACE_DIRECTORY "
                 "[INPUT.rgb [TRACE_REGEX]]]\n";
    return 2;
  }
  try {
    auto cfg = dgpp::GlmTextConfig::from_json_file(std::string(argv[1]) + "/config.json");
    if (!cfg.vision) throw std::runtime_error("checkpoint has no vision config");
    dgpp::ImageInput image;
    image.width = std::stoi(argv[2]);
    image.height = std::stoi(argv[3]);
    if (image.width < 28 || image.height < 28 || image.width % 28 || image.height % 28 ||
        static_cast<int64_t>(image.width) * image.height >
            static_cast<int64_t>(dgpp::kMaxImagePixels))
      throw std::invalid_argument("dimensions must be multiples of 28 within the image limit");
    image.tokens = image.width / 28 * (image.height / 28);
    image.rgb.resize(static_cast<size_t>(image.width) * image.height * 3);
    for (int y = 0; y < image.height; ++y)
      for (int x = 0; x < image.width; ++x) {
        auto* pixel = image.rgb.data() + (static_cast<size_t>(y) * image.width + x) * 3;
        pixel[0] = x * 255 / (image.width - 1);
        pixel[1] = y * 255 / (image.height - 1);
        pixel[2] = (x + y) * 255 / (image.width + image.height - 2);
      }
    if (argc >= 7 && std::string_view(argv[6]) != "-") {
      std::ifstream f(argv[6], std::ios::binary | std::ios::ate);
      if (!f || f.tellg() != static_cast<std::streamoff>(image.rgb.size()))
        throw std::invalid_argument("RGB fixture size does not match dimensions");
      f.seekg(0);
      f.read(reinterpret_cast<char*>(image.rgb.data()), image.rgb.size());
      if (!f) throw std::runtime_error("cannot read RGB fixture");
    }
    cudaStream_t stream;
    DGPP_CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    {
      dgpp::GlmVisionEncoder encoder(*cfg.vision, argv[1], stream);
      dgpp::GlmVisionEncoder::Trace trace;
      const std::regex filter(argc == 8 ? argv[7] : ".*");
      if (argc >= 6 && std::string_view(argv[5]) != "-") {
        std::filesystem::create_directories(argv[5]);
        trace = [&](const std::string& name, const void* data, size_t count, dgpp::DType dtype) {
          if (!std::regex_match(name, filter)) return;
          const size_t bytes = count * (dtype == dgpp::DType::F32 ? 4 : 2);
          std::vector<uint8_t> host(bytes);
          DGPP_CUDA_OK(cudaMemcpyAsync(host.data(), data, bytes, cudaMemcpyDeviceToHost, stream));
          DGPP_CUDA_OK(cudaStreamSynchronize(stream));
          const auto suffix = dtype == dgpp::DType::F32 ? ".f32" : ".bf16";
          std::ofstream f(std::filesystem::path(argv[5]) / (name + suffix), std::ios::binary);
          f.write(reinterpret_cast<const char*>(host.data()), bytes);
          if (!f) throw std::runtime_error("cannot write vision trace");
        };
      }
      const auto* output = encoder.encode(image, trace);
      std::vector<uint16_t> data(static_cast<size_t>(image.tokens) * cfg.hidden_size);
      DGPP_CUDA_OK(
          cudaMemcpyAsync(data.data(), output, data.size() * 2, cudaMemcpyDeviceToHost, stream));
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      std::ofstream f(argv[4], std::ios::binary);
      f.write(reinterpret_cast<const char*>(data.data()), data.size() * 2);
      if (!f) throw std::runtime_error("cannot write output");
      std::cout << image.tokens << " image tokens, digest " << encoder.digest() << '\n';
    }
    DGPP_CUDA_OK(cudaStreamDestroy(stream));
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
