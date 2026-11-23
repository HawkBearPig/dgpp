#pragma once

#include "common/image_input.hpp"
#include "loaders/minijson.hpp"

namespace dgpp::serve {
// Inline PNG/JPEG/WebP only. Resolving arbitrary URLs is deliberately not part of
// the HTTP/event-loop path. Decode errors carry the content part's API field.
struct ImageInputError : std::invalid_argument {
  ImageInputError(std::string message, std::string param)
      : std::invalid_argument(std::move(message)), param(std::move(param)) {}
  std::string param;
};

// How a family's processor turns decoded RGB into the patch canvas its tower
// expects. `pad_to_grid` is the GLM5-Next shape: round the canvas *up* to the
// grid and paint the remainder black. The Qwen3-VL family instead rounds to
// the *nearest* multiple (ties to even), never pads, and rejects an extreme
// aspect ratio or a side below the grid; both bounds are pixel budgets, i.e.
// tokens * grid * grid.
//
// `native_size` is the Inkling 2 shape: the image keeps its own size (no
// resample) on a canvas padded right and bottom to the grid, and the canvas
// records where the content ends (ImageInput::valid_width / valid_height),
// because the tower reads the padding as its processor's padding value and
// not as black. `min_tokens` is not a floor there: a 1 x 1 image is one
// token. The reference processor has no ceiling. DGPP's is `max_tokens`: an
// image whose padded grid holds more is scaled down, aspect kept, to the
// largest size that fits. `checkerboard_alpha` composites transparency on
// the reference's gray checkerboard (8-pixel squares of 204 and 153) in
// place of white (WebP) or dropping it (PNG).
struct ImagePreprocess {
  int grid = kGlmImageGrid;
  int min_tokens = 16;
  int max_tokens = kMaxImageTokens;
  bool pad_to_grid = true;
  int low_detail_tokens = 256;
  bool native_size = false;
  bool checkerboard_alpha = false;
};

// Inkling 2's processor (transformers' Inkling2ImageProcessor).
inline ImagePreprocess inkling_image_preprocess() {
  ImagePreprocess pp;
  pp.grid = kInklingImageGrid;
  pp.min_tokens = 1;
  pp.max_tokens = kMaxImageTokens;
  pp.pad_to_grid = false;
  pp.low_detail_tokens = 256;
  pp.native_size = true;
  pp.checkerboard_alpha = true;
  return pp;
}

ImageInput prepare_glm_image(const minijson::Value& image_url, const std::string& param);
ImageInput prepare_image(const minijson::Value& image_url, const std::string& param,
                         const ImagePreprocess& pp);
// Exposed separately for resize/patch-layout reference tests.
ImageInput resize_glm_image(const uint8_t* rgb, int width, int height, int max_tokens);
ImageInput resize_image(const uint8_t* rgb, int width, int height, const ImagePreprocess& pp);
// RGBA [height, width, 4] onto the reference checkerboard: the square of
// pixel (x, y) is 204 when (x / 8 + y / 8) is even and 153 otherwise, and a
// channel is (value * alpha + square * (255 - alpha) + 127) / 255.
void composite_on_checkerboard(const uint8_t* rgba, int width, int height, uint8_t* rgb);
}  // namespace dgpp::serve
