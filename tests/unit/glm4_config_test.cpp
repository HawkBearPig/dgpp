// The GLM-4.7 config parser: the real file's values parse,
// the architecture registry dispatches on it, and the unsupported shapes
// are refused by name.
#include <filesystem>
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "loaders/architecture.hpp"
#include "loaders/minijson.hpp"
#include "models/glm4/config.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// nvidia/GLM-4.7-NVFP4 @ 47fa7dc8, transcribed (the ignore list is
// generated below).
const char* kHead = R"({
  "architectures": ["Glm4MoeForCausalLM"], "attention_bias": true, "attention_dropout": 0.0,
  "dtype": "bfloat16", "eos_token_id": [151329, 151336, 151338], "first_k_dense_replace": 3,
  "head_dim": 128, "hidden_act": "silu", "hidden_size": 5120, "initializer_range": 0.02,
  "intermediate_size": 12288, "max_position_embeddings": 202752, "model_type": "glm4_moe",
  "moe_intermediate_size": 1536, "n_group": 1, "n_routed_experts": 160, "n_shared_experts": 1,
  "norm_topk_prob": true, "num_attention_heads": 96, "num_experts_per_tok": 8,
  "num_hidden_layers": 92, "num_key_value_heads": 8, "num_nextn_predict_layers": 1,
  "pad_token_id": 151329, "partial_rotary_factor": 0.5, "rms_norm_eps": 1e-05,
  "rope_scaling": null, "rope_theta": 1000000, "routed_scaling_factor": 2.5,
  "tie_word_embeddings": false, "topk_group": 1, "transformers_version": "4.57.1",
  "use_cache": true, "use_qk_norm": true, "vocab_size": 151552,
  "quantization_config": {
    "config_groups": {"group_0": {
      "input_activations": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
      "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
      "targets": ["Linear"]}},
    "ignore": [IGNORE],
    "quant_algo": "NVFP4",
    "kv_cache_scheme": {"dynamic": false, "num_bits": 8, "type": "float"},
    "producer": {"name": "modelopt", "version": "0.41.0"},
    "quant_method": "modelopt"
  }
})";

std::string ignore_json(int layers = 92, bool mtp = true) {
  std::string s = "\"lm_head\"";
  for (int l = 0; l < layers; ++l) s += ", \"model.layers." + std::to_string(l) + ".self_attn*\"";
  if (mtp) s += ", \"model.layers." + std::to_string(layers) + "*\"";
  return s;
}

std::string config_json(const std::string& patch_from = "", const std::string& patch_to = "",
                        const std::string& ignore = ignore_json()) {
  std::string s = kHead;
  s.replace(s.find("[IGNORE]"), 8, "[" + ignore + "]");
  if (!patch_from.empty()) {
    const size_t at = s.find(patch_from);
    require(at != std::string::npos, "patch anchor missing: " + patch_from);
    s.replace(at, patch_from.size(), patch_to);
  }
  return s;
}

dgpp::Glm4TextConfig parse(const std::string& text) {
  const auto t = dgpp::minijson::parse(text);
  return dgpp::Glm4TextConfig::parse(t.root);
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

DGPP_TEST(glm4_config_parses_the_release) {
  const dgpp::Glm4TextConfig c = parse(config_json());
  require(c.hidden_size == 5120 && c.vocab_size == 151552 && c.num_hidden_layers == 92, "shape");
  require(c.first_k_dense_replace == 3 && c.num_moe_layers() == 89, "dense/moe split");
  require(!c.is_moe_layer(2) && c.is_moe_layer(3) && c.is_moe_layer(92), "is_moe_layer");
  require(c.num_attention_heads == 96 && c.num_key_value_heads == 8 && c.head_dim == 128, "attention");
  require(c.attention_bias && c.use_qk_norm && c.rotary_dim == 64 && c.rope_theta == 1e6, "rope");
  require(c.q_heads_per_kv() == 12, "gqa");
  require(c.intermediate_size == 12288 && c.moe_intermediate_size == 1536, "mlp");
  require(c.n_routed_experts == 160 && c.num_experts_per_tok == 8 && c.n_shared_experts == 1, "moe");
  require(c.shared_expert_inter() == 1536, "shared inter");
  require(c.routed_scaling_factor == 2.5f && c.norm_topk_prob, "router");
  require(c.mtp_layer() == 92, "mtp layer");
  require(c.eos_token_ids.size() == 3 && c.eos_token_ids[0] == 151329, "eos");
  require(c.pad_token_id == 151329, "pad");
  require(c.fp4_group_size == 16 && c.kv_scales_present, "quantization");
  const dgpp::GlmMoeConfig m = c.moe_config(384);
  require(m.inter == 384 && m.n_experts == 160 && m.top_k == 8 && m.n_shared_experts == 1, "moe_config");
  require(m.router_mode == dgpp::MoeRouterMode::SigmoidBias && m.swiglu_limit > 1e30f, "moe_config router");
}

DGPP_TEST(glm4_config_refuses_unsupported_shapes) {
  auto has = [](const std::string& msg, const char* needle) { return msg.find(needle) != std::string::npos; };
  require(has(refusal(config_json("\"head_dim\": 128", "\"head_dim\": 64")), "head_dim"), "head_dim");
  require(has(refusal(config_json("\"n_group\": 1", "\"n_group\": 2")), "n_group"), "n_group");
  require(has(refusal(config_json("\"n_shared_experts\": 1", "\"n_shared_experts\": 2")), "n_shared_experts"),
          "shared experts");
  require(has(refusal(config_json("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\"")), "hidden_act"), "act");
  require(has(refusal(config_json("\"rope_scaling\": null", "\"rope_scaling\": {\"type\": \"yarn\"}")), "rope_scaling"),
          "rope scaling");
  require(has(refusal(config_json("\"quant_algo\": \"NVFP4\"", "\"quant_algo\": \"FP8\"")), "quant_algo"), "algo");
  require(has(refusal(config_json("\"type\": \"float\", \"group_size\": 16},\n      \"targets\"",
                                  "\"type\": \"float\", \"group_size\": 32},\n      \"targets\"")),
              "group_size"), "group");
  require(has(refusal(config_json("\"num_nextn_predict_layers\": 1", "\"num_nextn_predict_layers\": 2")),
              "num_nextn_predict_layers"), "mtp depth");
  // An ignore list missing the attention of one layer, or naming a module
  // the loader does not know, is refused.
  require(has(refusal(config_json("", "", ignore_json(91, true) + ", \"model.layers.91.mlp*\"")),
              "ignore"), "ignore coverage");
  require(has(refusal(config_json("", "", ignore_json() + ", \"model.layers.5.mlp.gate\"")), "unexpected ignored"),
          "ignore extra");
  // Without the draft layer the ignore list has no layer-92 entry.
  const dgpp::Glm4TextConfig no_mtp =
      parse(config_json("\"num_nextn_predict_layers\": 1", "\"num_nextn_predict_layers\": 0", ignore_json(92, false)));
  require(no_mtp.mtp_layer() < 0, "no mtp");
}

DGPP_TEST(glm4_architecture_registry_dispatches) {
  const auto g = dgpp::minijson::parse(R"({"architectures": ["Glm4MoeForCausalLM"], "model_type": "glm4_moe"})");
  require(dgpp::detect_architecture(g.root) == dgpp::ModelArchitecture::Glm4Moe, "glm4_moe");
  require(std::string(dgpp::model_architecture_name(dgpp::ModelArchitecture::Glm4Moe)) == "glm4_moe", "name");
  const auto t = dgpp::minijson::parse(R"({"model_type": "glm4_moe"})");
  require(dgpp::detect_architecture(t.root) == dgpp::ModelArchitecture::Glm4Moe, "by model_type");
  const auto g5 = dgpp::minijson::parse(R"({"architectures": ["Glm5ForConditionalGeneration"]})");
  require(dgpp::detect_architecture(g5.root) == dgpp::ModelArchitecture::Glm5, "glm5 unchanged");
}

DGPP_TEST(glm4_config_reads_the_landed_checkpoint) {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) dgpp::test::skip("HOME is unset; checkpoint cache unavailable");
  const fs::path root = fs::path(home) / ".cache/huggingface/hub/models--nvidia--GLM-4.7-NVFP4/snapshots";
  if (!fs::is_directory(root))
    dgpp::test::skip("checkpoint snapshot directory absent: " + root.string());
  for (const auto& snap : fs::directory_iterator(root)) {
    const fs::path cfg = snap.path() / "config.json";
    if (!fs::exists(cfg)) continue;
    require(dgpp::detect_architecture_file(cfg.string()) == dgpp::ModelArchitecture::Glm4Moe, "arch");
    const dgpp::Glm4TextConfig c = dgpp::Glm4TextConfig::from_json_file(cfg.string());
    require(c.num_hidden_layers == 92 && c.mtp_layer() == 92 && c.n_routed_experts == 160, "landed values");
    return;
  }
  dgpp::test::skip("no eligible checkpoint config under " + root.string());
}
