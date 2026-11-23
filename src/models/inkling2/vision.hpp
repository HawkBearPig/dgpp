#pragma once
// The Inkling 2 vision tower (docs/inkling2_mini_plan.md section 1.8),
// replicated on every rank: it runs only inside an image prefill, on the
// same journaled pixels everywhere, and its rows are never broadcast.
//
//   native 20-pixel patches of the padded canvas (raster order)
//     -> linear 1200 -> 1024, + the 32 x 32 position table resampled
//        bilinearly, a class row in front, LayerNorm
//   23 x [LayerNorm, qkv with biases + axial rotation, full attention inside
//         the image, projection, learned scale, +,
//         LayerNorm, 1024 -> 4096, GELU, 4096 -> 1024, learned scale, +]
//   each visual token's 2 x 2 patch rows concatenated 4096 -> 4096, GELU,
//     -> 2304 (no norm after the last layer, no biases in the projector)
//
// The returned rows replace the normed embeddings of the image's placeholder
// tokens, one row per 40 x 40 pixel cell in raster order.
//
// Rounding follows the released kernels: bf16 wherever the reference's eager
// ops store a tensor, fp32 inside. The attention's scores and probabilities
// stay fp32, as the reference's attention kernel keeps them, and its output
// is rounded.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "common/image_input.hpp"
#include "models/inkling2/config.hpp"

namespace dgpp {

class Inkling2VisionEncoder {
 public:
  // A staged window's positions (one image at the token ceiling and a row).
  static constexpr int kWindowTokens = kMaxImageTokens + 1;
  // Query rows per attention tile.
  static constexpr int kQueryTile = 256;
  // Whole embeddings kept across chunks and turns, keyed on the pixels.
  static constexpr int kEmbeddingCacheSlots = 4;
  static constexpr size_t kGemmWorkspace = 64ull << 20;
  // The fp32 P @ V's cuBLAS workspace.
  static constexpr size_t kFp32Workspace = 32ull << 20;

  // `cfg.vision` must be present. Loads the model.visual.* tensors the
  // binding table names from `checkpoint_dir` and reserves every buffer.
  Inkling2VisionEncoder(const Inkling2Config& cfg, const std::string& checkpoint_dir, cudaStream_t stream);
  ~Inkling2VisionEncoder();
  Inkling2VisionEncoder(const Inkling2VisionEncoder&) = delete;
  Inkling2VisionEncoder& operator=(const Inkling2VisionEncoder&) = delete;

  using Trace = std::function<void(const std::string&, const void*, size_t, DType)>;
  // One image's rows [tokens, text hidden], in prompt order. Valid until the
  // next encode().
  const uint16_t* encode(const ImageInput& image, Trace trace = {});
  // The rows of `images` at prompt positions [first, end), at their offsets
  // inside the window. Fixed storage, valid until the next stage().
  const uint16_t* stage(const std::vector<ImageInput>& images, int64_t first, int64_t end);
  uint64_t digest() const;
  // Tower executions so far (a cached image is not one).
  uint64_t encode_count() const;

  // Refuses a geometry the kernels or the shared image path do not carry,
  // naming the field.
  static void validate(const Inkling2VisionConfig& v);
  static int intermediate(const Inkling2VisionConfig& v) {
    return static_cast<int>(static_cast<double>(v.hidden_size) * v.mlp_ratio);
  }
  // The memory plan's two lines, exactly what the constructor allocates.
  static size_t weight_bytes(const Inkling2Config& cfg);
  static size_t workspace_bytes(const Inkling2VisionConfig& v);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace dgpp
