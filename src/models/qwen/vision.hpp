#pragma once
// The Qwen3.8-Flash-Next vision tower, replicated on every rank (the same
// policy as GLM-5.3-Flash's: it runs only on image prefills, decode graphs
// keep consuming ordinary token ids). models/qwen/vision_config.hpp describes
// the shape; this is the BF16 forward of transformers' Qwen3VLVisionModel:
//
//   patchify (32-px visual tokens, merge-block order) -> conv 1536 -> 1152
//   + bilinear-resampled learned position table
//   27 x [LN, qkv + 2-D axial RoPE, full attention, proj, +,
//         LN, fc1 1152 -> 4304, gelu-tanh, fc2 -> 1152, +]
//   LN per patch, then each 2x2 block concatenated 4608 -> fc1 -> gelu -> 2560
//
// The returned rows are the language model's image embeddings: one row of
// text hidden size per `image_token_id` in the prompt.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "common/image_input.hpp"
#include "models/qwen/vision_config.hpp"

namespace dgpp {
struct TensorInfo;
class QwenVisionEncoder {
 public:
  QwenVisionEncoder(const QwenVisionConfig& config, const std::string& checkpoint,
                    cudaStream_t stream);
  ~QwenVisionEncoder();
  using Trace = std::function<void(const std::string&, const void*, size_t, DType)>;
  // One image's rows, in prompt order. Valid until the next encode().
  const uint16_t* encode(const ImageInput& image, Trace trace = {});
  // Rows of `images` that fall in [first, first + end - first). Uses fixed
  // storage; valid until the next stage() on the same stream.
  const uint16_t* stage(const std::vector<ImageInput>& images, int64_t first, int64_t end);
  uint64_t digest() const;

  // The grid one image yields, and the token-major geometry the tower (and
  // the reference script) needs: patches per side, in merge-block order.
  struct Grid {
    int h_patches = 0, w_patches = 0;  // pixels / patch_size
    int h_blocks = 0, w_blocks = 0;    // pixels / (patch_size * merge)
    int tokens = 0, patches = 0;
  };
  static Grid grid_of(const ImageInput& image);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Load-time widening of a quantized vision linear to BF16 (vision.cpp):
// MXFP8 e4m3 x 2^(scale-127) is exact; NVFP4 e2m1 x e4m3-group x global
// rounds once (RNE). Both refuse malformed geometry and NaN codes by
// tensor name (std::runtime_error).
std::vector<uint16_t> qwen_vision_widen_mxfp8(const std::string& name, const uint8_t* weight,
                                              size_t wbytes, const uint8_t* scale, size_t sbytes,
                                              int64_t rows, int64_t cols);
std::vector<uint16_t> qwen_vision_dequant_nvfp4(const std::string& name, const uint8_t* weight,
                                               size_t wbytes, const uint8_t* scale, size_t sbytes,
                                               float global, int64_t rows, int64_t cols);
// Validate checkpoint layout before interpreting quantized bytes. The global
// scale is required only for NVFP4; MXFP8 carries its scales in the grid.
std::vector<uint16_t> qwen_vision_widen_linear(const TensorInfo& weight, const TensorInfo& scale,
                                             const TensorInfo* global, int64_t rows, int64_t cols);
}  // namespace dgpp
