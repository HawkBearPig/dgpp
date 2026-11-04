#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"
#include "models/minimax/vision_config.hpp"

namespace dgpp {
struct DFlash2Config;

// MiniMax M3.1's published MXFP8 dense / NVFP4 routed-expert checkpoint.
// Dimensions remain configurable for binding-generated forward fixtures.
struct MinimaxConfig {
  std::shared_ptr<const DFlash2Config> dspark;
  std::optional<MinimaxVisionConfig> vision;
  int hidden_size = 6144;
  int vocab_size = 200064;
  std::vector<int64_t> eos_token_ids = {200020};
  int num_hidden_layers = 60;
  int num_attention_heads = 64;
  int num_key_value_heads = 4;
  int head_dim = 128;
  int rotary_dim = 64;
  int max_position_embeddings = 1048576;
  double rope_theta = 10000000;
  float rms_norm_eps = 1e-6f;
  int intermediate_size = 3072;
  int dense_intermediate_size = 12288;
  int shared_intermediate_size = 3072;
  int num_local_experts = 128;
  int num_experts_per_tok = 4;
  float routed_scaling_factor = 2;
  float swiglu_alpha = 1.702f;
  float swiglu_limit = 7;
  int index_heads = 4;
  int index_dim = 128;
  int sparse_block_size = 128;
  int sparse_topk_blocks = 16;
  int sparse_local_blocks = 1;
  std::vector<uint8_t> moe_layer;

  static MinimaxConfig parse(const minijson::Value& root);
  static MinimaxConfig from_json_file(const std::string& path);
  void validate() const;
  void validate_tp(int rank, int world) const;
  bool is_moe_layer(int l) const { return moe_layer.at(static_cast<size_t>(l)) != 0; }
  int num_moe_layers() const;
};

}  // namespace dgpp
