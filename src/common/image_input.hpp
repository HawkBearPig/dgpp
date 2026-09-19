#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace dgpp {

// Resized RGB bytes, owned by the admission record. All ranks receive the
// same pixels and token spans; no URL or process-local pointer crosses the wire.
struct ImageInput {
  int64_t offset = 0;
  int tokens = 0;
  int width = 0, height = 0;
  std::vector<uint8_t> rgb;
};
inline constexpr int kMaxImageTokens = 1024;
inline constexpr size_t kMaxImagePixels = 1024 * 28 * 28;
// Bound decoded host data independently of the context's visual-token count.
inline constexpr size_t kMaxRequestImageBytes = 256ull << 20;

inline void validate_image_pixels(const ImageInput& im) {
  if (im.tokens < 1 || im.tokens > kMaxImageTokens || im.width < 1 || im.height < 1 ||
      im.width % 28 != 0 || im.height % 28 != 0 ||
      static_cast<uint64_t>(im.width) * im.height > kMaxImagePixels ||
      static_cast<uint64_t>(im.width / 28) * (im.height / 28) != static_cast<uint64_t>(im.tokens) ||
      im.rgb.size() != static_cast<uint64_t>(im.width) * im.height * 3)
    throw std::invalid_argument("invalid image RGB dimensions or token geometry");
}

inline void validate_image_inputs(const std::vector<ImageInput>& images, size_t prompt_tokens) {
  if (prompt_tokens > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
    throw std::invalid_argument("image prompt length exceeds the position range");
  int64_t end = 0;
  size_t bytes = 0;
  for (const auto& im : images) {
    validate_image_pixels(im);
    if (im.offset < end || im.offset < 0 || im.tokens < 1 || im.tokens > kMaxImageTokens ||
        static_cast<uint64_t>(im.offset) > prompt_tokens ||
        static_cast<size_t>(im.tokens) > prompt_tokens - static_cast<size_t>(im.offset))
      throw std::invalid_argument("invalid image token span");
    if (im.rgb.size() > kMaxRequestImageBytes - bytes)
      throw std::invalid_argument("decoded image data exceeds the 256 MiB request byte limit");
    end = im.offset + im.tokens;
    bytes += im.rgb.size();
  }
}
}  // namespace dgpp
