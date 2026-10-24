#pragma once
// Deterministic BF16 DFlash v1 checkpoint. Two draft layers and two target
// taps exercise sharding, feature fusion and request-local draft K/V.
#include <fstream>
#include "qwen35_fixture.hpp"

namespace qwen35fx {
inline void write_dflash_fixture(const std::string& dir, int block_rows = 16) {
  fs::create_directories(dir);
  std::ofstream cfg(fs::path(dir) / "config.json");
  cfg << R"({"architectures":["DFlashDraftModel"],"model_type":"qwen3",
    "dtype":"bfloat16","hidden_size":256,"vocab_size":512,"num_hidden_layers":2,
    "num_attention_heads":4,"num_key_value_heads":4,"head_dim":128,"intermediate_size":256,
    "rope_parameters":{"rope_type":"default","rope_theta":10000000},
    "use_sliding_window":true,"sliding_window":128,"max_position_embeddings":4096,
    "dflash_config":{"mask_token_id":511,"target_layer_ids":[0,3],"block_size":)"
      << block_rows << "}}";
  cfg.close();
  std::string header = "{";
  std::vector<uint8_t> data;
  auto add = [&](const std::string& name, const std::vector<int64_t>& shape) {
    QwenExpectedTensor e;
    e.name = name;
    e.dtype = dgpp::DType::BF16;
    e.shape = shape;
    auto bytes = tensor_bytes(e);
    // Qwen3 drafter norms store the gain itself (target norms store gain-1).
    if (name.find("norm") != std::string::npos)
      for (size_t i = 0; i < bytes.size(); i += 2) {
        uint16_t bits;
        std::memcpy(&bits, bytes.data() + i, 2);
        bits = float_to_bf16_bits(1.f + dgpp::bf16_bits_to_float(bits));
        std::memcpy(bytes.data() + i, &bits, 2);
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
  add("fc.weight", {256, 512});
  add("hidden_norm.weight", {256});
  add("norm.weight", {256});
  for (int layer = 0; layer < 2; ++layer) {
    const std::string p = "layers." + std::to_string(layer) + '.';
    add(p + "input_layernorm.weight", {256});
    add(p + "post_attention_layernorm.weight", {256});
    for (const auto* proj : {"q", "k", "v"})
      add(p + "self_attn." + proj + "_proj.weight", {512, 256});
    add(p + "self_attn.o_proj.weight", {256, 512});
    add(p + "self_attn.q_norm.weight", {128});
    add(p + "self_attn.k_norm.weight", {128});
    for (const auto* proj : {"gate", "up", "down"})
      add(p + "mlp." + proj + "_proj.weight", {256, 256});
  }
  header += '}';
  const uint64_t size = header.size();
  std::ofstream shard(fs::path(dir) / "model.safetensors", std::ios::binary);
  shard.write(reinterpret_cast<const char*>(&size), sizeof(size));
  shard.write(header.data(), header.size());
  shard.write(reinterpret_cast<const char*>(data.data()), data.size());
}
}  // namespace qwen35fx
