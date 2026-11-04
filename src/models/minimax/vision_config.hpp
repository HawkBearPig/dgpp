#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "common/image_input.hpp"
#include "loaders/minijson.hpp"

namespace dgpp {
// The published CLIP tower: FP32 spatiotemporal patch convolution, pre-LN,
// partial 3D RoPE, 32 BF16 blocks, projector, then a 2x2 patch merger.
struct MinimaxVisionConfig {
  static constexpr int kPatch = 14, kTemporal = 2, kMerge = 2, kGrid = 28;
  static constexpr int kQueryTile = 128, kWindowTokens = 513;
  static constexpr int kMaxFrames = 8, kMaxSegmentTokens = kMaxImageTokens * kMaxFrames / 2;
  static constexpr int kEmbeddingCacheSlots = 8;
  // Match the CUDA BF16 reference's cuBLASLt workspace cap. Its selected
  // split-K epilogues have different BF16 bias rounding from the 64 MiB
  // plan at the published MLP shape; this is part of the precision recipe.
  static constexpr size_t kGemmWorkspace = 1ull << 20;
  static constexpr size_t kFp32Workspace = 32ull << 20;
  int hidden = 1280, heads = 16, depth = 32, intermediate = 5120, output = 6144;
  float eps = 1e-5f;
  double rope_theta = 10000;
  ImageTokens tokens{200029, 200025, 200030};
  int64_t video_token = 200026;
  int head_dim() const { return hidden / heads; }
  int rotary_dim() const { return 6 * (head_dim() / 6); }
  int patch_in() const { return 3 * kTemporal * kPatch * kPatch; }
  static MinimaxVisionConfig parse(const minijson::Value& root, int text_hidden, int vocab) {
    const auto& v = root.at("vision_config");
    MinimaxVisionConfig c;
    const auto bad = [](const std::string& field) {
      throw std::invalid_argument("MiniMax vision config: unsupported or missing " + field);
    };
    const auto integer = [&](const minijson::Value& obj, const char* key, int low, int high) {
      const auto* p = obj.find(key);
      if (!p || p->kind() != minijson::Value::Kind::Int || p->as_int() < low || p->as_int() > high)
        bad(key);
      return static_cast<int>(p->as_int());
    };
    const auto word = [&](const minijson::Value& obj, const char* key, const char* want) {
      const auto* p = obj.find(key);
      if (!p || !p->is_string() || p->as_string() != want) bad(key);
    };
    const auto exact = [&](const minijson::Value& obj, const char* key, int want) {
      return integer(obj, key, want, want);
    };
    c.hidden = integer(v, "hidden_size", 80, 2560);
    c.heads = integer(v, "num_attention_heads", 1, 32);
    c.depth = integer(v, "num_hidden_layers", 1, 64);
    c.intermediate = integer(v, "intermediate_size", 32, 16384);
    c.output = exact(v, "projection_dim", text_hidden);
    if (c.hidden % c.heads || c.head_dim() != 80) bad("80-wide attention heads");
    exact(v, "patch_size", kPatch);
    exact(v, "num_channels", 3);
    exact(v, "vision_segment_max_frames", 4);
    word(v, "position_embedding_type", "rope");
    word(v, "rope_mode", "3d");
    word(v, "hidden_act", "gelu");
    word(root, "projector_hidden_act", "gelu");
    word(root, "vision_feature_select_strategy", "full");
    word(root, "process_image_mode", "dynamic_res");
    exact(root, "vision_feature_layer", -1);
    exact(root, "projector_hidden_size", text_hidden);
    const auto* bias = root.find("multimodal_projector_bias");
    if (!bias || !bias->is_bool() || !bias->as_bool()) bad("multimodal_projector_bias");
    for (const auto* owner : {&root, &v}) {
      const auto& m = owner->at("img_token_compression_config");
      word(m, "image_token_compression_method", "patch_merge");
      exact(m, "spatial_merge_size", kMerge);
      exact(m, "temporal_patch_size", kTemporal);
    }
    const auto positive = [&](const char* key) {
      const auto* p = v.find(key);
      if (!p || !p->is_number() || !std::isfinite(p->as_double()) || p->as_double() <= 0) bad(key);
      return p->as_double();
    };
    c.eps = static_cast<float>(positive("layer_norm_eps"));
    c.rope_theta = positive("rope_theta");
    c.tokens.pad = integer(root, "image_token_index", 0, vocab - 1);
    c.video_token = integer(root, "video_token_index", 0, vocab - 1);
    // The processor fixes start/end text. Explicit IDs support synthetic
    // vocabularies; the frontend validates the text-to-ID mapping at boot.
    if (root.find("vision_start_token_id"))
      c.tokens.start = integer(root, "vision_start_token_id", 0, vocab - 1);
    if (root.find("vision_end_token_id"))
      c.tokens.end = integer(root, "vision_end_token_id", 0, vocab - 1);
    if (c.tokens.start >= vocab || c.tokens.end >= vocab || c.tokens.start == c.tokens.pad ||
        c.tokens.end == c.tokens.pad || c.tokens.start == c.tokens.end ||
        c.video_token == c.tokens.start || c.video_token == c.tokens.end ||
        c.video_token == c.tokens.pad)
      bad("distinct in-vocabulary media token IDs");
    return c;
  }
  size_t weight_bytes() const {
    const size_t h = hidden, i = intermediate, o = output;
    return h * patch_in() * 4 + 2 * (2 * h + size_t(depth) * (4 * h * h + 2 * h * i + 9 * h + i) +
                                     o * h + o + o * o + o + 4 * o * o + o + o * o + o);
  }
  size_t score_elements() const { return size_t(heads) * kQueryTile * kMaxSegmentTokens * 4; }
  size_t workspace_bytes() const {
    const size_t n = kMaxSegmentTokens * 4, h = hidden, o = output;
    return pixels_for(kGrid) * 3 * kMaxFrames + n * patch_in() * 4 +
           n * h * 4 +                              // pixels, FP32 patch input/output
           10 * n * h * 2 + n * intermediate * 2 +  // BF16 x/norm/tmp, qkv, q/k/v/attention, MLP
           score_elements() * 6 +
           n * rotary_dim() * 4 +  // FP32 scores, BF16 probabilities, cos/sin
           2 * n * o * 2 + 2 * kMaxSegmentTokens * o * 2 +  // projector pair, merger, result
           kWindowTokens * o * 2 + kGemmWorkspace + kFp32Workspace +
           size_t(kEmbeddingCacheSlots) * kMaxSegmentTokens * o * 2;
  }
};
}  // namespace dgpp
