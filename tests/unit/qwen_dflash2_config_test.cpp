// The DFlash2 drafter's config parse: the released checkpoint's values,
// the refusals (v1 drafter, causal layers, unsupported knobs) and the
// cross-checks against the target (widths, taps, block vs window).
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "engine/decode_outputs.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen/config35.hpp"
#include "models/qwen/dflash2.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// A trimmed fixture of z-lab/Qwen3.8-27B-DFlash2's config.json.
const char* kFixture = R"({
  "architectures": ["DFlash2DraftModel"],
  "model_type": "qwen3",
  "dtype": "bfloat16",
  "hidden_size": 5120,
  "num_hidden_layers": 5,
  "vocab_size": 248320,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "intermediate_size": 17408,
  "rms_norm_eps": 1e-6,
  "rope_parameters": {"rope_type": "default", "rope_theta": 10000000},
  "max_position_embeddings": 262144,
  "use_sliding_window": true,
  "sliding_window": 2048,
  "is_causal": false,
  "tie_word_embeddings": false,
  "dflash_config": {
    "block_size": 8,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "mask_token_id": 248070,
    "selector_rank": 256,
    "selector_top_k": 16,
    "target_layer_ids": [5, 19, 33, 47, 61]
  }
})";

dgpp::DFlash2Config parse(const std::string& text) {
  return dgpp::DFlash2Config::parse(dgpp::minijson::parse(text).root);
}

std::string refusal(const std::string& text) {
  try {
    (void)parse(text);
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}

std::string patched(const std::string& from, const std::string& to) {
  std::string s = kFixture;
  const size_t at = s.find(from);
  require(at != std::string::npos, "patch anchor missing: " + from);
  s.replace(at, from.size(), to);
  return s;
}

dgpp::Qwen35TextConfig target_like() {
  dgpp::Qwen35TextConfig t;
  t.hidden_size = 5120;
  t.vocab_size = 248320;
  t.num_hidden_layers = 64;
  t.max_position_embeddings = 262144;
  return t;
}

std::string refusal_against(const dgpp::DFlash2Config& c, const dgpp::Qwen35TextConfig& t) {
  try {
    c.validate_against(t);
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}

}  // namespace

DGPP_TEST(dflash2_config_parses_the_release) {
  const auto c = parse(kFixture);
  c.validate_against(target_like());  // must not throw
  require(c.block_size == 8 && c.drafts() == 7 && c.query_rows() == 8, "block");
  require(c.mask_token_id == 248070, "mask token");
  require(c.conv_taps == 2 && c.conv_group_size == 16 && c.conv_groups() == 320, "conv");
  require(c.selector_rank == 256 && c.selector_top_k == 16, "selector");
  require(c.target_layer_ids.size() == 5 && c.fc_in() == 5 * 5120, "taps");
  require(c.num_hidden_layers == 5 && c.kv_row() == 8 * 128 && c.q_row() == 32 * 128, "widths");
  require(c.sliding_window == 2048, "window");
  require(c.rope_theta == 1e7, "theta");
}

DGPP_TEST(dflash2_config_refuses_the_wrong_checkpoints) {
  require(!refusal(patched("\"architectures\": [\"DFlash2DraftModel\"]",
                           "\"architectures\": [\"DFlashDraftModel\"]"))
               .empty(),
          "DFlash v1 must be refused by name");
  require(!refusal(patched("\"architectures\": [\"DFlash2DraftModel\"]",
                           "\"architectures\": [\"LlamaForCausalLM\"]"))
               .empty(),
          "a foreign architecture must be refused");
  require(!refusal(patched("\"is_causal\": false", "\"is_causal\": true")).empty(),
          "causal layers must be refused");
  require(!refusal(patched("\"model_type\": \"qwen3\"", "\"model_type\": \"llama\"")).empty(),
          "a non-qwen3 backbone must be refused");
  require(!refusal(patched("\"conv_kernel_size\": 2", "\"conv_kernel_size\": 4")).empty(),
          "only the 2-tap conv is implemented");
  require(!refusal(patched("\"selector_top_k\": 16", "\"selector_top_k\": 8")).empty(),
          "only top_k 16 is implemented");
  require(!refusal(patched("\"block_size\": 8", "\"block_size\": 1")).empty(), "block_size 1");
  require(!refusal(patched("\"head_dim\": 128", "\"head_dim\": 64")).empty(), "head_dim 64");
  require(!refusal(patched("\"rope_parameters\": {\"rope_type\": \"default\", \"rope_theta\": 10000000}",
                           "\"rope_parameters\": {\"rope_type\": \"yarn\", \"rope_theta\": 10000000}"))
               .empty(),
          "scaled draft rope must be refused");
  require(!refusal(patched("\"mask_token_id\": 248070", "\"mask_token_id\": 248070, \"sample_from_anchor\": true"))
               .empty(),
          "sample_from_anchor must be refused");
}

DGPP_TEST(dflash2_config_cross_checks_the_target) {
  dgpp::Qwen35TextConfig t = target_like();
  auto c = parse(kFixture);
  require(refusal_against(c, t).empty(), "the pair must pass");
  t.hidden_size = 4096;
  require(!refusal_against(c, t).empty(), "width mismatch must be refused");
  t = target_like();
  t.num_hidden_layers = 48;
  require(!refusal_against(c, t).empty(), "a tap past the target stack must be refused");
  t = target_like();
  c.sliding_window = 8;  // a window not covering the block
  require(!refusal_against(c, t).empty(), "block past the window must be refused");
  c = parse(kFixture);
  auto big = parse(patched("\"block_size\": 8",
                           "\"block_size\": " + std::to_string(dgpp::kSpecRows + 1)));
  require(refusal_against(big, target_like()).empty() == false, "an over-deep block must be refused");
}
