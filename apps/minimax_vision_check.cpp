// The MiniMax-M3.1-Flash vision tower's parity harness (docs/vision.md): one
// image in, the projected rows out, with every intermediate dumpable for a
// layer-by-layer comparison against the transformers reference.
//
//   minimax_vision_check CHECKPOINT WIDTH HEIGHT OUTPUT.bf16 [TRACE_DIR [INPUT.rgb [TRACE_REGEX]]]
//
// WIDTH and HEIGHT are the *processed* canvas, each a multiple of 28 (the
// tower's visual token is patch 14 times merge 2); the generator's synthetic
// ramp is what the reference script reproduces pixel for pixel, or read a
// real processed image from INPUT.rgb.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>

#include "common/cuda_check.hpp"
#include "loaders/minijson.hpp"
#include "models/minimax/vision.hpp"

int main(int argc, char** argv) {
  if (argc < 5 || argc > 9) {
    std::cerr << "usage: minimax_vision_check CHECKPOINT WIDTH HEIGHT OUTPUT.bf16 [TRACE_DIRECTORY "
                 "[INPUT.rgb [TRACE_REGEX [FRAMES]]]]\n";
    return 2;
  }
  try {
    std::ifstream config(std::string(argv[1]) + "/config.json");
    const std::string json((std::istreambuf_iterator<char>(config)), {});
    const auto root = dgpp::minijson::parse(json);
    const auto& text = root.root.at("text_config");
    const auto cfg = dgpp::MinimaxVisionConfig::parse(root.root, text.at("hidden_size").as_int(),
                                                      text.at("vocab_size").as_int());
    dgpp::ImageInput image;
    image.grid = dgpp::MinimaxVisionConfig::kGrid;
    image.width = std::stoi(argv[2]);
    image.height = std::stoi(argv[3]);
    image.frames = argc == 9 ? std::stoi(argv[8]) : 1;
    if (image.frames != 1 && (image.frames < 2 || image.frames > 8 || image.frames % 2))
      throw std::invalid_argument("frames must be 1, 2, 4, 6 or 8");
    const int g = image.grid;
    if (image.width < g || image.height < g || image.width % g || image.height % g ||
        static_cast<int64_t>(image.width) * image.height >
            static_cast<int64_t>(dgpp::pixels_for(g)))
      throw std::invalid_argument("dimensions must be multiples of " + std::to_string(g) +
                                  " within the image limit");
    image.tokens = image.width / g * (image.height / g) * image.spans();
    image.rgb.resize(static_cast<size_t>(image.width) * image.height * 3 * image.frames);
    for (int frame = 0; frame < image.frames; ++frame)
      for (int y = 0; y < image.height; ++y)
        for (int x = 0; x < image.width; ++x) {
          auto* pixel = image.rgb.data() +
                        ((static_cast<size_t>(frame) * image.height + y) * image.width + x) * 3;
          pixel[0] = (x * 255 / (image.width - 1) + 17 * frame) % 256;
          pixel[1] = (y * 255 / (image.height - 1) + 29 * frame) % 256;
          pixel[2] = ((x + y) * 255 / (image.width + image.height - 2) + 43 * frame) % 256;
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
      dgpp::MinimaxVisionEncoder encoder(cfg, argv[1], stream);
      dgpp::MinimaxVisionEncoder::Trace trace;
      const std::regex filter(argc >= 8 ? argv[7] : ".*");
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
      std::vector<uint16_t> data(static_cast<size_t>(image.tokens) * cfg.output);
      DGPP_CUDA_OK(
          cudaMemcpyAsync(data.data(), output, data.size() * 2, cudaMemcpyDeviceToHost, stream));
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      std::ofstream f(argv[4], std::ios::binary);
      f.write(reinterpret_cast<const char*>(data.data()), data.size() * 2);
      if (!f) throw std::runtime_error("cannot write output");
      std::cout << image.tokens << " image tokens (" << image.width / g << "x" << image.height / g
                << " grid), digest " << encoder.digest() << '\n';
    }
    DGPP_CUDA_OK(cudaStreamDestroy(stream));
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
