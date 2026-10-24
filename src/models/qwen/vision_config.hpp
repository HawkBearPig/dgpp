#pragma once
// Qwen3.8-Flash-Next's vision tower (Qwen4ExpForConditionalGeneration).
//
// The checkpoint's root config carries a `vision_config` describing a
// Qwen3-VL-style tower: a 16-pixel spatiotemporal patch embedding, 27
// bidirectional blocks with 2-D axial RoPE over the full 72-wide head, a
// learned 48x48 position table bilinearly resampled to the image's grid, and
// a `merger` that concatenates each 2x2 block of patches and projects
// 4*1152 -> 2560. `deepstack_visual_indexes` is empty, so the projected rows
// enter the language model once, at the embedding, rather than being injected
// into several decoder layers.
//
// The two numbers a text-only reader would miss: the tower's grid is 16 *
// spatial_merge_size = 32 pixels per visual token (GLM5-Next's is 28), and
// its `out_hidden_size` must equal the text model's hidden size.
//
// Same policy as every other config here: a field is either parsed into a
// known-supported value or refused with a message naming it.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

#include "common/image_input.hpp"
#include "loaders/minijson.hpp"

namespace dgpp {
struct QwenVisionConfig {
  // The tower's geometry is fixed by the checkpoint family; the parser
  // refuses anything else. patch 16, temporal 2, merge 2 -> 32 px/token.
  static constexpr int kPatch = 16;
  static constexpr int kTemporal = 2;
  static constexpr int kMerge = 2;
  static constexpr int kGrid = kPatch * kMerge;  // = kQwenImageGrid
  // One language-model prefill chunk plus MTP's next-token lookahead.
  static constexpr int kWindowTokens = 2049;
  // The attention's query tile: it bounds the fp32 score scratch, so the
  // workspace formulas and the encoder must agree on it.
  static constexpr int kQueryTile = 128;
  // transformers' Qwen3VLVisionConfig defaults, absent from this checkpoint.
  static constexpr float kLayerNormEps = 1e-6f;
  static constexpr double kRopeTheta = 10000.0;
  // The 2304-row table is a 48x48 grid.
  static constexpr int kPositionGrid = 48;

  int depth = 27, hidden = 1152, heads = 16, intermediate = 4304, output = 2560;
  int num_position_embeddings = 2304;
  float eps = kLayerNormEps;
  double rope_theta = kRopeTheta;
  int head_dim() const { return hidden / heads; }
  int merged() const { return hidden * kMerge * kMerge; }
  int patches_per_token() const { return kMerge * kMerge; }
  // 3 channels * temporal 2 * 16 * 16, the conv's input width.
  int patch_in() const { return 3 * kTemporal * kPatch * kPatch; }

  // The three delimiters, from the root config: vision_start_token_id,
  // image_token_id (repeated once per visual token) and vision_end_token_id.
  ImageTokens tokens;

  static QwenVisionConfig parse(const minijson::Value& root, int text_hidden) {
    // Refused here rather than through minijson's at(), which raises a plain
    // runtime_error: a caller that has already decided to serve images wants
    // the same invalid_argument contract as every other bad field.
    const minijson::Value* node = root.find("vision_config");
    if (!node || node->is_null() || !node->is_object())
      throw std::invalid_argument(
          "Qwen vision: config carries no vision_config object (a text-only export ships no tower)");
    const auto& v = *node;
    QwenVisionConfig c;
    // A field the tower cannot serve without a second kernel path is
    // refused by value, with the value it wants in the message.
    const auto exact = [&](const char* key, int want) {
      const auto* f = v.find(key);
      if (!f || !f->is_number() || !std::isfinite(f->as_double()) ||
          f->as_double() != std::floor(f->as_double()) || static_cast<int>(f->as_int()) != want)
        throw std::invalid_argument(std::string("Qwen vision_config.") + key +
                                    ": this tower needs the value " + std::to_string(want));
    };
    exact("patch_size", kPatch);
    exact("temporal_patch_size", kTemporal);
    exact("spatial_merge_size", kMerge);
    exact("in_channels", 3);
    exact("num_position_embeddings", kPositionGrid * kPositionGrid);
    const auto bounded = [&](const char* key, int max) {
      const auto* f = v.find(key);
      if (!f || !f->is_number() || !std::isfinite(f->as_double()) ||
          f->as_double() != std::floor(f->as_double()) || f->as_double() < 1 ||
          f->as_double() > max)
        throw std::invalid_argument(std::string("Qwen vision_config.") + key + ": invalid integer");
      return static_cast<int>(f->as_int());
    };
    c.depth = bounded("depth", 64);
    c.hidden = bounded("hidden_size", 16384);
    c.heads = bounded("num_heads", 64);
    c.intermediate = bounded("intermediate_size", 65536);
    c.output = bounded("out_hidden_size", 65536);
    if (c.hidden % c.heads || (c.hidden / c.heads) % 2 || c.hidden / c.heads > 256)
      throw std::invalid_argument(
          "Qwen vision_config: the hidden size must split into an even, rope-able head");
    if (c.output != text_hidden)
      throw std::invalid_argument("Qwen vision_config.out_hidden_size must equal the text model's "
                                  "hidden_size");
    if (const auto* act = v.find("hidden_act"); !act || !act->is_string() ||
        act->as_string() != "gelu_pytorch_tanh")
      throw std::invalid_argument("Qwen vision_config.hidden_act: this tower is gelu_pytorch_tanh");
    if (const auto* deep = v.find("deepstack_visual_indexes");
        !deep || !deep->is_array() || !deep->items().empty())
      throw std::invalid_argument(
          "Qwen vision_config.deepstack_visual_indexes: only the no-deepstack tower is served "
          "(image rows enter at the embedding)");
    // An explicit eps/theta is honoured when present; this checkpoint omits
    // both and takes transformers' Qwen3VL defaults.
    if (const auto* f = v.find("layer_norm_eps"); f && !f->is_null()) {
      if (!f->is_number() || !std::isfinite(static_cast<float>(f->as_double())) ||
          f->as_double() <= 0)
        throw std::invalid_argument("Qwen vision_config.layer_norm_eps: must be positive");
      c.eps = static_cast<float>(f->as_double());
    }
    if (const auto* rope = v.find("rope_parameters"); rope && rope->is_object()) {
      if (const auto* theta = rope->find("rope_theta"); theta && theta->is_number())
        c.rope_theta = theta->as_double();
      if (!std::isfinite(c.rope_theta) || c.rope_theta <= 1.0)
        throw std::invalid_argument("Qwen vision_config.rope_parameters.rope_theta: invalid");
    }
    const auto token_id = [&](const char* name) {
      const auto* f = root.find(name);
      if (!f || !f->is_number() || !std::isfinite(f->as_double()) ||
          f->as_double() != std::floor(f->as_double()) || f->as_int() < 0)
        throw std::invalid_argument(std::string("Qwen vision: ") + name +
                                    ": must be a non-negative integer token id");
      return f->as_int();
    };
    c.tokens = ImageTokens{token_id("vision_start_token_id"), token_id("image_token_id"),
                           token_id("vision_end_token_id")};
    if (c.tokens.start == c.tokens.pad || c.tokens.pad == c.tokens.end ||
        c.tokens.start == c.tokens.end)
      throw std::invalid_argument("Qwen vision: the image delimiters must be three distinct ids");
    // A video placeholder id equal to the image one would let a span check
    // accept a video part this server never decodes.
    if (const auto* video = root.find("video_token_id");
        video && video->is_number() && video->as_int() == c.tokens.pad)
      throw std::invalid_argument("Qwen vision: video_token_id must differ from image_token_id");
    return c;
  }

  size_t weight_bytes() const {
    const size_t h = hidden, i = intermediate, o = output, m = merged();
    return 2 * (h * static_cast<size_t>(patch_in()) + h +
                static_cast<size_t>(num_position_embeddings) * h +
                static_cast<size_t>(depth) * (4 * h * h + 2 * h * i + 9 * h + i) +
                // merger norm (2 x h), fc1 [m, m] + [m], fc2 [o, m] + [o]
                2 * h + m * m + m + o * m + o);
  }

  size_t attention_score_bytes() const {
    const size_t patches = static_cast<size_t>(kMaxImageTokens) * patches_per_token();
    return static_cast<size_t>(heads) * kQueryTile * patches * sizeof(float);
  }

  size_t workspace_bytes() const {
    const size_t n = static_cast<size_t>(kMaxImageTokens) * patches_per_token(), h = hidden;
    const size_t rows = n * h, tok = kMaxImageTokens, hd = head_dim(), o = output;
    // One term per buffer the encoder reserves at startup (models/qwen/
    // vision.cpp), so the pre-flight plan cannot understate the sum.
    return pixels_for(kGrid) * 3 +                          // rgb
           n * static_cast<size_t>(patch_in()) * 2 +        // patchified input
           3 * rows * 2 +                                  // residual, norm, projected
           4 * rows * 2 +                                  // q, k, v, attention out
           rows * 3 * 2 +                                  // fused qkv
           attention_score_bytes() +                       // fp32 score tile
           n * n * 2 +                                     // probabilities
           n * hd * 2 * 4 +                                // rope cos | sin
           n * 4 * (sizeof(int32_t) + sizeof(float)) +     // position gather plan
           tok * static_cast<size_t>(merged()) * 2 +       // merged rows
           static_cast<size_t>(kWindowTokens + tok) * o * 2 +  // staged window
           n * static_cast<size_t>(intermediate) * 2 +     // MLP hidden
           tok * o * 2 +                                   // one image's rows
           (64ull << 20);                                  // GEMM workspace
  }
};
}  // namespace dgpp
