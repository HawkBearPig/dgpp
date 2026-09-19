#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"
#include "loaders/hf_cache.hpp"
#include "models/glm/config.hpp"
#include "models/glm/vision.hpp"

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 2;
  std::string error;
  const char* configured = std::getenv("DGPP_VISION_TEST_CHECKPOINT");
  const std::string checkpoint = configured ? configured :
      dgpp::hf::model_dir("HawkBearPig/GLM-5.3-Flash-NVFP4-FP8", &error);
  if (checkpoint.empty()) { std::cout << "[SKIP] " << error << '\n'; return 2; }
  try {
    const auto config = dgpp::GlmTextConfig::from_json_file(checkpoint + "/config.json");
    if (!config.vision) throw std::runtime_error("vision checkpoint required");
    cudaStream_t stream;
    DGPP_CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    {
      dgpp::GlmVisionEncoder encoder(*config.vision, checkpoint, stream);
      const size_t h = config.hidden_size;
      std::vector<dgpp::ImageInput> images;
      for (int i = 0; i < 12; ++i) {
        dgpp::ImageInput image{1537 + i * 1027, 1024, 896, 896,
                              std::vector<uint8_t>(896 * 896 * 3)};
        for (size_t p = 0; p < image.rgb.size(); p += 3) image.rgb[p + (i % 2 ? 2 : 0)] = 255;
        images.push_back(std::move(image));
      }
      const int64_t end = images.back().offset + images.back().tokens + 3;
      dgpp::validate_image_inputs(images, end);
      std::vector<std::vector<uint16_t>> reference(2, std::vector<uint16_t>(1024 * h));
      for (int i = 0; i < 2; ++i) {
        const auto* output = encoder.encode(images[i]);
        DGPP_CUDA_OK(cudaMemcpyAsync(reference[i].data(), output, reference[i].size() * 2,
                                     cudaMemcpyDeviceToHost, stream));
        DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      }
      // Normal chunks, busy chunks and a cache attachment inside an image.
      for (const auto& [start, chunk] : {std::pair<int64_t, int>{0, 2048}, {0, 256}, {1740, 256}}) {
        for (int64_t first = start; first < end; first += chunk) {
          const auto stop = std::min(first + chunk + 1, end);  // MTP lookahead
          const auto* window = encoder.stage(images, first, stop);
          std::vector<uint16_t> actual((stop - first) * h);
          DGPP_CUDA_OK(cudaMemcpyAsync(actual.data(), window, actual.size() * 2,
                                       cudaMemcpyDeviceToHost, stream));
          DGPP_CUDA_OK(cudaStreamSynchronize(stream));
          for (size_t i = 0; i < images.size(); ++i) {
            const auto& image = images[i];
            const auto begin = std::max(first, image.offset);
            const auto finish = std::min(stop, image.offset + image.tokens);
            if (finish <= begin) continue;
            if (!std::equal(actual.begin() + (begin - first) * h,
                            actual.begin() + (finish - first) * h,
                            reference[i % 2].begin() + (begin - image.offset) * h))
              throw std::runtime_error("streamed image rows differ from whole-image encoder output");
          }
        }
      }
      bool rejected = false;
      try { encoder.stage(images, 0, dgpp::GlmVisionConfig::kWindowTokens + 1); }
      catch (const std::invalid_argument&) { rejected = true; }
      if (!rejected) throw std::runtime_error("oversized window accepted");
    }
    DGPP_CUDA_OK(cudaStreamDestroy(stream));
    std::cout << "vision streaming rows match independent image outputs bitwise\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
