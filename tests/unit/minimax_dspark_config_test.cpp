#include <fstream>
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "models/minimax/dspark.hpp"

namespace {
std::string source() {
  std::ifstream in(DGPP_SOURCE_DIR "/tests/data/minimax/dspark_config.json");
  if (!in) throw std::runtime_error("DSpark fixture missing");
  return {(std::istreambuf_iterator<char>(in)), {}};
}
void require(bool ok, const char* msg) {
  if (!ok) throw std::runtime_error(msg);
}
auto parse(const std::string& text) {
  return dgpp::minimax_dspark_config(dgpp::minijson::parse(text).root);
}
void rejects(const std::string& from, const std::string& to, const std::string& field) {
  auto text = source();
  const auto at = text.find(from);
  require(at != std::string::npos, "missing substitution");
  text.replace(at, from.size(), to);
  try {
    (void)parse(text);
  } catch (const std::invalid_argument& e) {
    require(std::string(e.what()).find(field) != std::string::npos, e.what());
    return;
  }
  throw std::runtime_error("accepted invalid DSpark " + field);
}
}  // namespace
DGPP_TEST(minimax_dspark_config_release) {
  const auto c = parse(source());
  require(c.backbone == dgpp::DFlash2Config::Backbone::MiniMaxM3, "native backbone");
  require(c.hidden_size == 6144 && c.vocab_size == 200064 && c.num_hidden_layers == 5,
          "backbone geometry");
  require(c.intermediate_size == 12288 && c.conv_taps == 0, "dense FFN without convolutions");
  require(c.num_attention_heads == 64 && c.num_key_value_heads == 4 && c.rotary_dim == 64,
          "attention geometry");
  require(c.query_rows() == 8 && c.drafts() == 7 && c.mask_token_id == 200058,
          "seven proposals plus anchor");
  require(c.selector_rank == 256 && c.sliding_window == 4096, "Markov rank and window");
  require(c.target_layer_ids == std::vector<int>({3, 17, 31, 45, 59}),
          "input taps in FC column order");
  c.validate_against({6144, 200064, 60, 1048576});
}
DGPP_TEST(minimax_dspark_config_rejects_semantic_changes) {
  rejects("\"DSparkMiniMaxDraftModel\"", "\"DFlash2DraftModel\"", "architectures");
  rejects("\"use_gemma_norm\": true", "\"use_gemma_norm\": false", "use_gemma_norm");
  rejects("\"use_qk_norm\": true", "\"use_qk_norm\": false", "use_qk_norm");
  rejects("\"rotary_dim\": 64", "\"rotary_dim\": 128", "rotary_dim");
  rejects("\"hidden_act\": \"swigluoai\"", "\"hidden_act\": \"silu\"", "hidden_act");
  rejects("\"num_key_value_heads\": 4", "\"num_key_value_heads\": 0", "num_key_value_heads");
  rejects("\"hidden_size\": 6144", "\"hidden_size\": 6144.5", "hidden_size");
  rejects("\"dspark_block_size\": 7", "\"dspark_block_size\": 6", "dspark_block_size");
  rejects("\"quant_method\": \"mxfp8\"", "\"quant_method\": \"fp8\"", "quant_method");
  rejects("\"sliding_window\": 4096", "\"sliding_window\": 4", "dflash_config");
  rejects("\"confidence_head_with_markov\": true", "\"confidence_head_with_markov\": false",
          "confidence_head_with_markov");
}
