#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace dgpp {

// Resized RGB bytes, owned by the admission record. All ranks receive the
// same pixels and token spans; no URL or process-local pointer crosses the
// wire. `grid` is the family's pixels per visual-token side (patch_size *
// spatial_merge_size): 28 for the GLM5-Next tower, 32 for the Qwen3-VL one,
// so the same struct describes both towers' canvases.
struct ImageInput {
  int64_t offset = 0;
  int tokens = 0;
  int width = 0, height = 0;
  std::vector<uint8_t> rgb;
  int grid = 28;
  // MiniMax video segments contain 2/4/6/8 RGB frames. Each consecutive
  // pair produces one spatial token span; timestamp text separates spans.
  // All frames remain one cache identity: attention is bidirectional over
  // the segment, so later pixels can change its first token's embedding.
  int frames = 1;
  std::vector<int64_t> frame_offsets = {};
  bool is_video() const { return frames != 1; }
  int spans() const { return is_video() ? frames / 2 : 1; }
  int span_tokens() const { return tokens / spans(); }
  int64_t span_offset(int span) const { return is_video() ? frame_offsets.at(span) : offset; }
};
inline constexpr int kMaxImageTokens = 1024;
inline constexpr int kGlmImageGrid = 28;
inline constexpr int kQwenImageGrid = 32;
// The GLM tower's workspace bound; per-family sizes follow from pixels_for().
inline constexpr size_t kMaxImagePixels = 1024ull * 28 * 28;
// Bound decoded host data independently of the context's visual-token count.
inline constexpr size_t kMaxRequestImageBytes = 256ull << 20;

// Pixels a single image may occupy at this family's grid.
inline constexpr size_t pixels_for(int grid) {
  return static_cast<size_t>(kMaxImageTokens) * grid * grid;
}
inline bool image_grid_supported(int grid) { return grid == 28 || grid == 32; }

// The checkpoint's three image delimiters, read from config.json: the
// trained start/end markers and the pad token repeated once per patch.
struct ImageTokens {
  int64_t start = -1;
  int64_t pad = -1;
  int64_t end = -1;
};

inline void validate_image_pixels(const ImageInput& im, bool allow_video = false) {
  if (!image_grid_supported(im.grid))
    throw std::invalid_argument("unsupported image pixel grid");
  if (im.frames != 1 &&
      (!allow_video || im.grid != 28 || im.frames < 2 || im.frames > 8 || im.frames % 2))
    throw std::invalid_argument("unsupported video frame geometry");
  const int64_t g = im.grid;
  const int spans = im.spans();
  if (im.tokens < 1 || im.tokens > kMaxImageTokens * spans || im.width < 1 || im.height < 1 ||
      im.width % g != 0 || im.height % g != 0 ||
      static_cast<uint64_t>(im.width) * im.height > pixels_for(g) ||
      static_cast<uint64_t>(im.width / g) * (im.height / g) * spans !=
          static_cast<uint64_t>(im.tokens) ||
      im.rgb.size() != static_cast<uint64_t>(im.width) * im.height * 3 * im.frames)
    throw std::invalid_argument("invalid image RGB dimensions or token geometry");
}

inline void validate_image_inputs(const std::vector<ImageInput>& images, size_t prompt_tokens,
                                  bool allow_video = false) {
  if (prompt_tokens > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
    throw std::invalid_argument("image prompt length exceeds the position range");
  int64_t end = 0;
  size_t bytes = 0;
  for (const auto& im : images) {
    validate_image_pixels(im, allow_video);
    if ((!im.is_video() && !im.frame_offsets.empty()) ||
        (im.is_video() &&
         (im.frame_offsets.size() != size_t(im.spans()) || im.frame_offsets.front() != im.offset)))
      throw std::invalid_argument("invalid video token spans");
    for (int span = 0; span < im.spans(); ++span) {
      const int64_t at = im.span_offset(span);
      if (at < end || at < 0 || static_cast<uint64_t>(at) > prompt_tokens ||
          static_cast<size_t>(im.span_tokens()) > prompt_tokens - static_cast<size_t>(at))
        throw std::invalid_argument("invalid image token span");
      end = at + im.span_tokens();
    }
    if (im.rgb.size() > kMaxRequestImageBytes - bytes)
      throw std::invalid_argument("decoded image data exceeds the 256 MiB request byte limit");
    bytes += im.rgb.size();
  }
}
}  // namespace dgpp
