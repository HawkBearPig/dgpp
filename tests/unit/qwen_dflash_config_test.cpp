// The DFlash v1 drafter's config parse: the 122B checkpoint's values, the
// refusals (v2 arch, non-qwen3, bad shapes) and the cross-checks against
// the MoE target (widths, vocab, taps, block vs window).
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen/config35.hpp"
#include "models/qwen/dflash.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// A trimmed fixture of z-lab/Qwen3.5-122B-A10B-DFlash's config.json.
const char* kFixture = R"({
  "architectures": ["DFlashDraftModel"],
  "model_type": "qwen3",
  "dtype": "bfloat16",
  "hidden_size": 3072,
  "num_hidden_layers": 6,
  "vocab_size": 248320,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "intermediate_size": 9216,
  "rms_norm_eps": 1e-6,
  "rope_parameters": {"rope_type": "default", "rope_theta": 10000000},
  "max_position_embeddings": 262144,
  "use_sliding_window": true,
  "sliding_window": 4096,
  "tie_word_embeddings": false,
  "dflash_config": {
    "block_size": 16,
    "mask_token_id": 248077,
    "target_layer_ids": [1, 7, 14, 20, 26, 32, 39, 45]
  }
})";

dgpp::DFlashConfig parse(const std::string& text) {
  const std::string owned = text;  // parsed values view the input
  return dgpp::DFlashConfig::parse(dgpp::minijson::parse(owned).root);
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
  t.is_moe = true;
  t.hidden_size = 3072;
  t.vocab_size = 248320;
  t.num_hidden_layers = 48;
  t.max_position_embeddings = 262144;
  return t;
}

std::string refusal_against(const dgpp::DFlashConfig& c, const dgpp::Qwen35TextConfig& t) {
  try {
    c.validate_against(t);
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}

}  // namespace

DGPP_TEST(dflash_config_parses_the_release) {
  const auto c = parse(kFixture);
  c.validate_against(target_like());  // must not throw
  require(c.block_size == 16 && c.drafts() == 15 && c.query_rows() == 16, "block");
  require(c.mask_token_id == 248077, "mask token");
  require(c.target_layer_ids.size() == 8 && c.fc_in() == 8 * 3072, "taps");
  require(c.num_hidden_layers == 6 && c.kv_row() == 8 * 128 && c.q_row() == 32 * 128, "widths");
  require(c.intermediate_size == 9216, "mlp");
  require(c.sliding_window == 4096, "window");
  require(c.tp_divisible(1) && c.tp_divisible(2) && c.tp_divisible(4), "tp worlds");
  require(!c.tp_divisible(3), "world 3 must fail (heads 32/8, inter 9216)");
}

DGPP_TEST(dflash_config_refuses_the_wrong_checkpoints) {
  require(!refusal(patched("\"architectures\": [\"DFlashDraftModel\"]",
                           "\"architectures\": [\"DFlash2DraftModel\"]"))
               .empty(),
          "the v2 drafter parses elsewhere");
  require(!refusal(patched("\"architectures\": [\"DFlashDraftModel\"]",
                           "\"architectures\": [\"LlamaForCausalLM\"]"))
               .empty(),
          "a foreign architecture must be refused");
  require(!refusal(patched("\"model_type\": \"qwen3\"", "\"model_type\": \"llama\"")).empty(),
          "a non-qwen3 backbone must be refused");
  require(!refusal(patched("\"head_dim\": 128", "\"head_dim\": 64")).empty(), "head_dim 64");
  require(!refusal(patched("\"block_size\": 16", "\"block_size\": 1")).empty(), "block_size 1");
  require(!refusal(patched("\"rope_parameters\": {\"rope_type\": \"default\", \"rope_theta\": 10000000}",
                           "\"rope_parameters\": {\"rope_type\": \"yarn\", \"rope_theta\": 10000000}"))
               .empty(),
          "scaled draft rope must be refused");
}

DGPP_TEST(dflash_config_cross_checks_the_target) {
  dgpp::Qwen35TextConfig t = target_like();
  const auto c = parse(kFixture);
  require(refusal_against(c, t).empty(), "the pair must pass");
  t.hidden_size = 4096;
  require(!refusal_against(c, t).empty(), "width mismatch must be refused");
  t = target_like();
  t.num_hidden_layers = 40;
  require(!refusal_against(c, t).empty(), "tap 45 past the target stack must be refused");
  t = target_like();
  auto w = parse(kFixture);
  w.sliding_window = 16;  // a window not covering the block
  require(!refusal_against(w, t).empty(), "block past the window must be refused");
}
