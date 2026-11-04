#include <fstream>
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "loaders/architecture.hpp"
#include "models/minimax/config.hpp"

namespace {
std::string source() {
  std::ifstream f(DGPP_SOURCE_DIR "/tests/data/minimax/config.json");
  if (!f) throw std::runtime_error("MiniMax config fixture missing");
  return {(std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()};
}
dgpp::MinimaxConfig parse(const std::string& s) {
  const auto json = dgpp::minijson::parse_document(s);
  return dgpp::MinimaxConfig::parse(json.root);
}
void require(bool ok, const std::string& msg) {
  if (!ok) throw std::runtime_error(msg);
}
void rejects(const std::string& before, const std::string& after, const std::string& key) {
  auto s = source();
  const auto at = s.find(before);
  require(at != std::string::npos, "fixture substitution missing: " + before);
  s.replace(at, before.size(), after);
  try {
    (void)parse(s);
  } catch (const std::invalid_argument& e) {
    require(std::string(e.what()).find(key) != std::string::npos, e.what());
    return;
  }
  throw std::runtime_error("accepted unsupported " + key);
}
}  // namespace

DGPP_TEST(minimax_config_release) {
  require(dgpp::detect_architecture(dgpp::minijson::parse_document(source()).root) ==
              dgpp::ModelArchitecture::MinimaxM3,
          "release architecture detection");
  require(dgpp::detect_architecture(
              dgpp::minijson::parse_document(R"({"model_type":"minimax_m3_vl"})").root) ==
              dgpp::ModelArchitecture::MinimaxM3,
          "model type fallback detection");
  const auto c = parse(source());
  require(c.hidden_size == 6144 && c.vocab_size == 200064 && c.num_hidden_layers == 60,
          "text shape");
  require(c.num_moe_layers() == 57 && !c.is_moe_layer(2) && c.is_moe_layer(3), "dense/MoE pattern");
  require(c.num_attention_heads == 64 && c.num_key_value_heads == 4 && c.index_heads == 4,
          "head partition");
  require(c.head_dim == 128 && c.index_dim == 128 && c.rotary_dim == 64 && c.rope_theta == 1e7,
          "partial RoPE");
  require(c.sparse_topk_blocks == 16 && c.sparse_block_size == 128 && c.sparse_local_blocks == 1,
          "MSA");
  require(c.swiglu_alpha == 1.702f && c.swiglu_limit == 7 && c.routed_scaling_factor == 2,
          "numerics");
  require(c.intermediate_size == 3072 && c.dense_intermediate_size == 12288 &&
              c.shared_intermediate_size == 3072 && c.num_local_experts == 128 &&
              c.num_experts_per_tok == 4,
          "FFN");
  for (int world : {1, 2, 4})
    for (int rank = 0; rank < world; ++rank) c.validate_tp(rank, world);
  for (int world : {0, 3, 8}) {
    bool failed = false;
    try {
      c.validate_tp(0, world);
    } catch (const std::invalid_argument&) {
      failed = true;
    }
    require(failed, "invalid TP world accepted");
  }
}

DGPP_TEST(minimax_config_refuses_semantic_changes) {
  rejects("\"use_gemma_norm\": true", "\"use_gemma_norm\": false", "use_gemma_norm");
  rejects("\"use_qk_norm\": true", "\"use_qk_norm\": false", "use_qk_norm");
  rejects("\"hidden_act\": \"swigluoai\"", "\"hidden_act\": \"silu\"", "hidden_act");
  rejects("\"scoring_func\": \"sigmoid\"", "\"scoring_func\": \"softmax\"", "scoring_func");
  rejects("\"sparse_score_type\": \"max\"", "\"sparse_score_type\": \"mean\"", "sparse_score_type");
  rejects("\"sparse_init_block\": 0", "\"sparse_init_block\": 1", "sparse_init_block");
  rejects("\"partial_rotary_factor\": 0.5", "\"partial_rotary_factor\": 1.0",
          "partial_rotary_factor");
  rejects("\"group_size\": 16", "\"group_size\": 32", "group_size");
  rejects("\"num_hidden_layers\": 60", "\"num_hidden_layers\": 59", "moe_layer_freq");
  rejects("\"hidden_size\": 6144", "\"hidden_size\": 6144.5", "hidden_size");
  rejects("\"hidden_size\": 6144", "\"hidden_size\": 6112", "hidden_size");
  rejects("\"hidden_size\": 6144", "\"hidden_size\": 8256", "hidden_size");
  rejects("\"num_key_value_heads\": 4", "\"num_key_value_heads\": 0", "num_attention_heads");
  rejects("\"swiglu_alpha\": 1.702", "\"swiglu_alpha\": 0", "numerics");
  rejects("\"rotary_dim\": 64", "\"rope_scaling\": {}, \"rotary_dim\": 64", "rope_scaling");
}
