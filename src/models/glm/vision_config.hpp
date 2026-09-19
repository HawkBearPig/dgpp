#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "common/image_input.hpp"
#include "loaders/minijson.hpp"

namespace dgpp {
struct GlmVisionConfig {
  // One language-model chunk plus MTP's next-token lookahead.
  static constexpr int kWindowTokens = 2049;
  int depth = 24, hidden = 1024, heads = 16, intermediate = 4096;
  int output = 4096, projection = 10240;
  float eps = 1e-5f, swiglu_limit = 10.0f;
  static GlmVisionConfig parse(const minijson::Value& root, int text_hidden) {
    const auto& v = root.at("vision_config");
    GlmVisionConfig c;
    const auto integer = [&](const char* key, int max) {
      const auto* f = v.find(key);
      if (!f || !f->is_number() || !std::isfinite(f->as_double()) || f->as_double() < 1 ||
          f->as_double() > max || f->as_double() != std::floor(f->as_double()))
        throw std::invalid_argument(std::string("GLM vision_config.") + key + ": invalid integer");
      return static_cast<int>(f->as_int());
    };
    c.depth = integer("depth", 64);
    c.hidden = integer("hidden_size", 4096);
    c.heads = integer("num_heads", 64);
    c.intermediate = integer("intermediate_size", 32768);
    c.output = integer("out_hidden_size", 16384);
    c.projection = integer("projection_intermediate_size", 65536);
    if (integer("patch_size", 14) != 14 || integer("temporal_patch_size", 2) != 2 ||
        integer("spatial_merge_size", 2) != 2 || integer("in_channels", 3) != 3 ||
        c.hidden % c.heads || (c.hidden / c.heads) % 4 || c.hidden / c.heads > 256 ||
        c.output != text_hidden)
      throw std::invalid_argument("GLM vision_config: incompatible patch, head or output geometry");
    for (const char* name : {"rms_norm_eps", "swiglu_limit"}) {
      const auto* f = v.find(name);
      if (!f || !f->is_number() || !std::isfinite(static_cast<float>(f->as_double())) ||
          static_cast<float>(f->as_double()) <= 0)
        throw std::invalid_argument(std::string("GLM vision_config.") + name +
                                    ": must be positive");
      (std::string_view(name) == "rms_norm_eps" ? c.eps : c.swiglu_limit) =
          static_cast<float>(f->as_double());
    }
    if (v.at("hidden_act").as_string() != "silu" || !v.at("attention_bias").is_bool() ||
        !v.at("attention_bias").as_bool())
      throw std::invalid_argument("GLM vision_config: requires silu and attention_bias");
    const auto token_is = [&](const char* name, int id) {
      const auto* f = root.find(name);
      return f && f->is_number() && f->as_double() == id;
    };
    if (!token_is("image_token_id", 154854) || !token_is("image_start_token_id", 154830) ||
        !token_is("image_end_token_id", 154831))
      throw std::invalid_argument("GLM vision: incompatible image token ids");
    if (const auto* rope = v.find("rope_parameters"))
      if (!rope->is_object() || rope->at("rope_type").as_string() != "axial" ||
          rope->at("rope_theta").as_double() != 10000)
        throw std::invalid_argument("GLM vision: only axial RoPE with theta 10000 is supported");
    return c;
  }
  size_t weight_bytes() const {
    const size_t h = hidden, d = h / heads, o = output, i = intermediate, p = projection;
    return 2 *
           (h * (3 * 2 * 14 * 14) + h + depth * (4 * h * h + 3 * h * i + 7 * h + 2 * i + 2 * d) +
            h + o * h * 4 + o + o * o + 2 * o + 3 * o * p);
  }
  size_t workspace_bytes() const {
    const size_t n = kMaxImageTokens * 4, h = hidden, o = output;
    const size_t rows = std::max(n * h, static_cast<size_t>(kMaxImageTokens) * o);
    // Three residual/normalized/projected buffers, Q/K/V/output, MLP gates,
    // patch input, tiled attention scores and probabilities, RGB, GEMM.
    return 3 * rows * 2 + 7 * n * h * 2 +
           2 * std::max(n * intermediate, static_cast<size_t>(kMaxImageTokens) * projection) * 2 +
           n * 1176 * 2 + n * n * 2 +
           std::max({n * n, n * 3 * h, n * intermediate,
                     static_cast<size_t>(kMaxImageTokens) * output,
                     static_cast<size_t>(kMaxImageTokens) * projection}) *
               4 +
           kMaxImagePixels * 3 + (64ull << 20) +
           static_cast<size_t>(kMaxImageTokens + kWindowTokens) * output * 2;
  }
};
}  // namespace dgpp
