// The Flint config parser: the release's values parse, the derived geometry
// (the layer kinds, the no-RoPE set, the dense layers, the per-projection
// expert formats), and the unsupported shapes are refused by name.
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "flint_config_json.hpp"
#include "loaders/architecture.hpp"
#include "loaders/minijson.hpp"
#include "models/flint/config.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

dgpp::FlintTextConfig parse(const std::string& text) {
  const auto t = dgpp::minijson::parse(text);
  return dgpp::FlintTextConfig::parse(t.root);
}

std::string refusal(const std::string& text) {
  try {
    (void)parse(text);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

}  // namespace

DGPP_TEST(flint_config_parses_the_release) {
  const dgpp::FlintTextConfig c = parse(flint_test::config_json());
  require(c.hidden_size == 2048 && c.vocab_size == 150144 && c.num_hidden_layers == 49, "shape");
  require(c.num_swa_layers() == 35 && c.num_moe_layers() == 47, "layer kinds");
  require(!c.is_swa_layer(0) && !c.is_swa_layer(1) && c.is_swa_layer(2) && !c.is_swa_layer(5) && c.is_swa_layer(47) &&
              !c.is_swa_layer(48),
          "is_swa_layer");
  // Every full-attention layer is a no-RoPE layer and vice versa.
  for (int l = 0; l < c.num_hidden_layers; ++l) require(c.rope_of(l) == c.is_swa_layer(l), "rope pattern at " + std::to_string(l));
  require(!c.is_moe_layer(0) && !c.is_moe_layer(1) && c.is_moe_layer(2) && c.is_moe_layer(48), "is_moe_layer");
  require(c.num_attention_heads == 32 && c.num_key_value_heads == 4 && c.head_dim == 128, "heads");
  require(c.rope_theta == 1e4 && c.sliding_window == 513 && c.sink, "rope / window / sink");
  require(!c.attention_bias, "no attention bias");
  require(c.qkv_rows() == (32 + 8) * 128 && c.o_proj_cols() == 4096, "fused projection rows / o_proj cols");
  require(c.intermediate_size == 6144 && c.moe_intermediate_size == 512 && c.shared_expert_intermediate_size == 1024, "mlp");
  require(c.n_routed_experts == 256 && c.num_experts_per_tok == 6 && c.norm_topk_prob, "moe");
  require(c.eos_token_ids.size() == 4 && c.eos_token_ids[0] == 150001 && c.eos_token_ids[3] == 150018, "eos");
  require(c.bos_token_id == 150000 && c.pad_token_id == 150001, "bos / pad");
  require(c.rms_norm_eps == 1e-6f && c.max_position_embeddings == 262144, "eps / positions");
  require(c.fp8_block == 128 && c.nvfp4_group == 16, "quantization");
  // The precision policy: 2-11, 13-30, 32 fp8; 12 up only nvfp4; 31, 33-36,
  // 38-48 nvfp4; 37 gate/up nvfp4, down fp8.
  for (int l = 2; l <= 48; ++l) {
    const bool all_fp4 = l == 31 || (l >= 33 && l <= 36) || (l >= 38 && l <= 48);
    const uint8_t want = all_fp4 ? 7 : l == 12 ? 2 : l == 37 ? 3 : 0;
    require(c.expert_fp4_mask[static_cast<size_t>(l)] == want, "precision mask at layer " + std::to_string(l));
  }
  require(c.expert_fp4(12, 1) && !c.expert_fp4(12, 0) && !c.expert_fp4(12, 2), "layer 12's up alone is nvfp4");
  require(c.expert_fp4(37, 0) && c.expert_fp4(37, 1) && !c.expert_fp4(37, 2), "layer 37's down alone is fp8");
  require(c.expert_fp4_mask[0] == 0 && c.expert_fp4_mask[1] == 0, "the dense layers carry no mask");
  const dgpp::GlmMoeConfig m = c.moe_config(128);
  require(m.inter == 128 && m.n_experts == 256 && m.top_k == 6 && m.n_shared_experts == 0, "moe_config");
  require(m.routed_scaling_factor == 1.0f && m.norm_topk_prob, "moe_config router");
  require(m.router_mode == dgpp::MoeRouterMode::SigmoidBias && m.swiglu_limit > 1e30f, "moe_config mode");
}

DGPP_TEST(flint_config_refuses_unsupported_shapes) {
  auto has = [](const std::string& msg, const char* needle) { return msg.find(needle) != std::string::npos; };
  require(has(refusal(flint_test::config_json("\"model_type\": \"flint\"", "\"model_type\": \"flint2\"")), "model_type"),
          "model_type");
  require(has(refusal(flint_test::config_json("\"head_dim\": 128", "\"head_dim\": 192")), "head_dim"), "head_dim");
  require(has(refusal(flint_test::config_json("\"softmax_type\": \"learnable\"", "\"softmax_type\": \"vanilla\"")), "softmax_type"),
          "softmax_type");
  require(has(refusal(flint_test::config_json("\"quant_method\": \"dgpp_mixed\"", "\"quant_method\": \"fp8\"")), "quant_method"),
          "quant method");
  require(has(refusal(flint_test::config_json("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\"")), "hidden_act"), "act");
  require(has(refusal(flint_test::config_json("\"router_score_function\": \"sigmoid\"", "\"router_score_function\": \"softmax\"")),
              "router_score_function"),
          "router");
  require(has(refusal(flint_test::config_json("\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true")),
              "tie_word_embeddings"),
          "tied embeddings");
  require(has(refusal(flint_test::config_json("\"rope_theta\": 10000.0", "\"rope_theta\": 10000.0, \"rope_scaling\": {\"type\": \"yarn\"}")),
              "rope_scaling"),
          "rope scaling");
  // A layer type outside the two, a pattern of the wrong length.
  require(has(refusal(flint_test::config_json("\"sliding_attention\",\n  \"full_attention\"\n ]", "\"sliding_attention\",\n  \"chunked_attention\"\n ]")),
              "layer_types"),
          "layer type");
  require(has(refusal(flint_test::config_json("\"num_hidden_layers\": 49", "\"num_hidden_layers\": 48")), "layer_types"),
          "pattern length");
  // An expert format outside fp8 / nvfp4; a missing policy entry; a BF16 retention.
  require(has(refusal(flint_test::config_json("\"gate\": \"fp8\",\n     \"up\": \"nvfp4\"", "\"gate\": \"int4\",\n     \"up\": \"nvfp4\"")),
              "precision_by_layer"),
          "expert format");
  require(has(refusal(flint_test::config_json("\"retained_bf16_layers\": []", "\"retained_bf16_layers\": [5]")), "retained_bf16_layers"),
          "bf16 retention");
  require(has(refusal(flint_test::config_json("\"group_size\": 16", "\"group_size\": 32")), "group_size"), "nvfp4 group");
  require(has(refusal(flint_test::config_json("\"weight_block_size\": [\n     128,\n     128\n    ]", "\"weight_block_size\": [\n     64,\n     64\n    ]")),
              "weight_block_size"),
          "fp8 block");
  require(has(refusal(flint_test::config_json("\"decoder_sparse_step\": 1", "\"decoder_sparse_step\": 2")), "decoder_sparse_step"),
          "sparse step");
}

DGPP_TEST(flint_architecture_is_detected) {
  const std::string text = flint_test::config_json();
  const auto d = dgpp::minijson::parse(text);
  require(dgpp::detect_architecture(d.root) == dgpp::ModelArchitecture::Flint, "flint");
  require(std::string(dgpp::model_architecture_name(dgpp::ModelArchitecture::Flint)) == "flint", "name");
  const auto t = dgpp::minijson::parse("{\"model_type\": \"flint\"}");
  require(dgpp::detect_architecture(t.root) == dgpp::ModelArchitecture::Flint, "by model_type");
}

DGPP_TEST(flint_config_reads_the_landed_checkpoint) {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) dgpp::test::skip("HOME is unset; checkpoint cache unavailable");
  const fs::path root = fs::path(home) / ".cache/huggingface/hub/models--HawkBearPig--flint-FP8-NVFP4/snapshots";
  if (!fs::is_directory(root)) dgpp::test::skip("checkpoint snapshot directory absent: " + root.string());
  for (const auto& snap : fs::directory_iterator(root)) {
    const fs::path cfg = snap.path() / "config.json";
    if (!fs::exists(cfg)) continue;
    const dgpp::FlintTextConfig c = dgpp::FlintTextConfig::from_json_file(cfg.string());
    require(c.num_hidden_layers == 49 && c.n_routed_experts == 256 && c.num_moe_layers() == 47, "landed values");
    require(dgpp::detect_architecture_file(cfg.string()) == dgpp::ModelArchitecture::Flint, "arch");
    return;
  }
  dgpp::test::skip("no eligible checkpoint config under " + root.string());
}
