#pragma once
// Deterministic BF16 DFlash2 (v2) checkpoint for the Qwen3.8-Flash-Next
// tests (the 2026-10-20 port): two sliding draft layers, two target taps,
// the two-tap grouped convs and the candidate-selector codebooks — the
// tensor set load_dflash2_weights preflights (models/dflash2/dflash2.cpp).
// Sized to the qwenfx tiny target: hidden 256, vocab 512, four target
// layers, one full-attention layer (the taps land at mix sites 1 and 3).
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "qwen_fixture.hpp"

namespace qwenfx {
inline void write_dflash2_fixture(const std::string& dir, int block_rows = 8,
                                  int layers = 2) {
  fs::create_directories(dir);
  std::ofstream cfg(fs::path(dir) / "config.json");
  cfg << R"({"architectures":["DFlash2DraftModel"],"model_type":"qwen3",
    "dtype":"bfloat16","hidden_size":256,"vocab_size":512,"num_hidden_layers":)" << layers
      << R"(,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":128,
    "intermediate_size":256,"rms_norm_eps":1e-06,
    "rope_parameters":{"rope_type":"default","rope_theta":10000000},
    "use_sliding_window":true,"sliding_window":128,"max_position_embeddings":4096,
    "dflash_config":{"mask_token_id":511,"target_layer_ids":[0,2],"block_size":)"
      << block_rows
      << R"(,"conv_kernel_size":2,"conv_group_size":16,"selector_rank":8,
             "selector_top_k":16}})";
  cfg.close();
  std::string header = "{";
  std::vector<uint8_t> data;
  auto add = [&](const std::string& name, const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t s : shape) n *= s;
    std::vector<uint8_t> bytes(static_cast<size_t>(n) * 2);
    glmrng::Rng rng(glmrng::seed_for(name));
    for (int64_t i = 0; i < n; ++i) {
      // Weights ~N(0, 0.02) scale; the +1 norm fixup below lands the norms
      // at gain ~1 (the drafter's norms store the gain itself).
      const float v = 0.02f * static_cast<float>(rng.unit() - 1.0);
      const uint16_t bits = float_to_bf16_bits(v);
      std::memcpy(bytes.data() + static_cast<size_t>(i) * 2, &bits, 2);
    }
    if (name.find("norm") != std::string::npos) {
      for (size_t i = 0; i < bytes.size(); i += 2) {
        uint16_t bits;
        std::memcpy(&bits, bytes.data() + i, 2);
        bits = float_to_bf16_bits(1.f + dgpp::bf16_bits_to_float(bits));
        std::memcpy(bytes.data() + i, &bits, 2);
      }
    }
    if (!data.empty()) header += ',';
    header += '"' + name + "\":{\"dtype\":\"BF16\",\"shape\":[";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i) header += ',';
      header += std::to_string(shape[i]);
    }
    header += "],\"data_offsets\":[" + std::to_string(data.size()) + ',' +
              std::to_string(data.size() + bytes.size()) + "]}";
    data.insert(data.end(), bytes.begin(), bytes.end());
  };
  add("fc.weight", {256, 2 * 256});
  add("hidden_norm.weight", {256});
  add("norm.weight", {256});
  add("candidate_selector.predecessor_codebook", {512, 8});
  add("candidate_selector.successor_codebook", {512, 8});
  add("candidate_selector.hidden_projection.weight", {8, 256});
  for (int layer = 0; layer < layers; ++layer) {
    const std::string p = "layers." + std::to_string(layer) + '.';
    add(p + "input_layernorm.weight", {256});
    add(p + "post_attention_layernorm.weight", {256});
    add(p + "self_attn.q_proj.weight", {512, 256});
    add(p + "self_attn.k_proj.weight", {256, 256});
    add(p + "self_attn.v_proj.weight", {256, 256});
    add(p + "self_attn.o_proj.weight", {256, 512});
    add(p + "self_attn.q_norm.weight", {128});
    add(p + "self_attn.k_norm.weight", {128});
    add(p + "mlp.gate_proj.weight", {256, 256});
    add(p + "mlp.up_proj.weight", {256, 256});
    add(p + "mlp.down_proj.weight", {256, 256});
    add(p + "attention_conv.base_kernel", {2, 2, 256});
    add(p + "attention_conv.kernel_projection.weight", {64, 256});
    add(p + "mlp_conv.base_kernel", {2, 2, 256});
    add(p + "mlp_conv.kernel_projection.weight", {64, 256});
  }
  header += '}';
  const uint64_t size = header.size();
  std::ofstream shard(fs::path(dir) / "model.safetensors", std::ios::binary);
  shard.write(reinterpret_cast<const char*>(&size), sizeof(size));
  shard.write(header.data(), header.size());
  shard.write(reinterpret_cast<const char*>(data.data()), data.size());
}
}  // namespace qwenfx
