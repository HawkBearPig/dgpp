// The Dory config parser: the release's own values parse, the derived nGPT
// scale factors match the reference, the architecture registry dispatches on
// it, and the unsupported shapes are refused by name
// (docs/dory_plan.md §1.1, §3).
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "loaders/architecture.hpp"
#include "loaders/minijson.hpp"
#include "models/dory/config.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}
void require_near(double got, double want, double rel, const std::string& what) {
  const double tol = std::abs(want) * rel + 1e-9;
  if (!(std::abs(got - want) <= tol))
    throw std::runtime_error(what + ": got " + std::to_string(got) + ", want " + std::to_string(want));
}

// The reference's generated `layer_types` (dory_config.py __post_init__).
std::string default_layer_types() {
  std::string s = "[";
  for (int i = 0; i < 72; ++i) {
    if (i % 2) {
      s += "\"mlp\"";
    } else if (i % 8 == 6) {
      s += "\"full_attention\"";
    } else {
      s += "\"sliding_attention\"";
    }
    if (i != 71) s += ", ";
  }
  return s + "]";
}

// nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4, the release's config.json
// (2026-10-17) transcribed; `layer_types` is generated above so the literal
// stays readable.
std::string config_json(const std::string& needle = "", const std::string& repl = "") {
  std::string s = R"({
  "architectures": ["DoryForCausalLM"],
  "model_type": "dory",
  "vocab_size": 131072, "hidden_size": 2560, "num_hidden_layers": 72,
  "intermediate_size": 7168, "num_attention_heads": 16, "num_key_value_heads": 4,
  "head_dim": 256, "max_position_embeddings": 262144,
  "tie_word_embeddings": false, "attention_bias": false, "mlp_bias": false,
  "hidden_act": "silu", "initializer_range": 0.0177,
  "ngpt_alpha_init": 0.05, "attention_output_gate_logit_scale_init": null,
  "n_input_layers": 16, "n_recurrent_layers": 40, "n_output_layers": 16,
  "n_recurrent_loops": 2, "sliding_window": 1024,
  "rope_scaling": null,
  "rope_parameters": {
    "sliding_attention": {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 1.0},
    "full_attention": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 1.0}
  },
  "layer_types": __LAYER_TYPES__,
  "quantization_config": {
    "config_groups": {"group_0": {
      "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
      "targets": ["Linear"]}},
    "ignore": ["backbone.embeddings", "lm_head"],
    "quant_algo": "W4A16_NVFP4",
    "producer": {"name": "modelopt", "version": "0.0.0"},
    "quant_method": "modelopt"
  }
})";
  const std::string marker = "__LAYER_TYPES__";
  s.replace(s.find(marker), marker.size(), default_layer_types());
  if (!needle.empty()) {
    const size_t at = s.find(needle);
    if (at == std::string::npos) throw std::runtime_error("fixture needle not found: " + needle);
    s.replace(at, needle.size(), repl);
  }
  return s;
}

dgpp::DoryConfig parse(const std::string& json) {
  const auto parsed = dgpp::minijson::parse(json);
  return dgpp::DoryConfig::parse(parsed.root);
}

std::string refusal(const std::string& needle, const std::string& repl) {
  try {
    (void)parse(config_json(needle, repl));
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

bool has(const std::string& msg, const char* needle) {
  return msg.find(needle) != std::string::npos;
}

}  // namespace

DGPP_TEST(dory_config_parses_the_release) {
  const dgpp::DoryConfig c = parse(config_json());

  require(c.hidden_size == 2560 && c.vocab_size == 131072 && c.num_hidden_layers == 72, "shape");
  require(c.intermediate_size == 7168 && c.head_dim == 256, "ffn/head");
  require(c.num_attention_heads == 16 && c.num_key_value_heads == 4, "heads");
  require(c.max_position_embeddings == 262144 && c.sliding_window == 1024, "context");
  require(c.q_size() == 4096 && c.kv_size() == 1024, "q/kv size");
  require(c.gate_up_rows() == 14336, "gate_up rows");

  // The loop.
  require(c.n_input_layers == 16 && c.n_recurrent_layers == 40 && c.n_output_layers == 16, "partitions");
  require(c.n_recurrent_loops == 2, "loops");
  require(c.in_input_block(0) && c.in_input_block(15) && !c.in_input_block(16), "input block");
  require(c.in_recurrent_block(16) && c.in_recurrent_block(55) && !c.in_recurrent_block(56), "recurrent block");
  require(c.in_output_block(56) && c.in_output_block(71), "output block");

  // The layer kinds: 36 attention (27 sliding, 9 full), 36 mlp.
  require(c.num_attention_layers() == 36, "attention layers");
  require(!c.is_attention(1) && !c.is_attention(71), "odd layers are mlp");
  require(c.is_sliding(0) && c.is_full_attention(6) && c.is_full_attention(14), "kind of layer 0/6/14");
  require(c.is_full_attention(22) && c.is_full_attention(70), "kind of layer 22/70");
  int sliding = 0, full = 0;
  for (int l = 0; l < c.num_hidden_layers; ++l) {
    sliding += c.is_sliding(l);
    full += c.is_full_attention(l);
  }
  require(sliding == 27 && full == 9, "sliding/full counts");

  // The two thetas.
  require(c.rope_theta_of(0) == 1e4 && c.rope_theta_of(6) == 1e7, "per-kind rope theta");

  // The recurrent block's K/V repetition is the loop policy's, not the walk's.
  require(c.kv_repeats(0, false) == 1 && c.kv_repeats(16, false) == 2 && c.kv_repeats(55, false) == 2,
          "per_loop repeats");
  require(c.kv_repeats(16, true) == 1 && c.kv_repeats(70, true) == 1, "last_loop repeats");

  // The derived nGPT scale factors (fp32, as the reference computes them).
  require_near(c.alpha_scale(), 2.8248588, 1e-6, "alpha_scale");
  require_near(c.sqk_factor(), 903.95480, 1e-6, "sqk_factor");
  require_near(c.s_gate_factor(), 401.87107, 1e-6, "s_gate_factor");
  require_near(c.suv_factor(), 50.596443, 1e-6, "suv_factor");
  require_near(c.logit_factor(), 56.497175, 1e-6, "logit_factor");
  require(c.nvfp4_group == 16, "nvfp4 group");
}

DGPP_TEST(dory_config_generates_the_default_layer_types) {
  // Without `layer_types` the reference's generated pattern is used.
  std::string json = config_json();
  const size_t at = json.find("\"layer_types\": [");
  const size_t end = json.find("],", at);
  json.erase(at, end - at + 2);  // the key, the array and its trailing comma
  const dgpp::DoryConfig c = parse(json);
  require(c.num_attention_layers() == 36, "generated attention layers");
  require(c.is_full_attention(6) && c.is_sliding(0) && !c.is_attention(1), "generated pattern");
}

DGPP_TEST(dory_architecture_registry_dispatches) {
  const auto d = dgpp::minijson::parse(R"({"architectures": ["DoryForCausalLM"], "model_type": "dory"})");
  require(dgpp::detect_architecture(d.root) == dgpp::ModelArchitecture::Dory, "dory");
  require(std::string(dgpp::model_architecture_name(dgpp::ModelArchitecture::Dory)) == "dory", "name");
  const auto t = dgpp::minijson::parse(R"({"model_type": "dory"})");
  require(dgpp::detect_architecture(t.root) == dgpp::ModelArchitecture::Dory, "by model_type");
  const auto q = dgpp::minijson::parse(R"({"architectures": ["Qwen4ExpForConditionalGeneration"]})");
  require(dgpp::detect_architecture(q.root) == dgpp::ModelArchitecture::Qwen4Exp, "qwen unchanged");
}

DGPP_TEST(dory_config_refuses_unsupported_shapes) {
  require(has(refusal("\"model_type\": \"dory\"", "\"model_type\": \"other\""), "model_type"), "model_type");
  require(has(refusal("\"n_recurrent_loops\": 2", "\"n_recurrent_loops\": 0"), "n_recurrent_loops"), "loops");
  require(has(refusal("\"n_input_layers\": 16", "\"n_input_layers\": 17"), "n_input_layers"), "partitions odd");
  require(has(refusal("\"n_output_layers\": 16", "\"n_output_layers\": 18"), "n_input_layers"), "partitions sum");
  require(has(refusal("\"attention_bias\": false", "\"attention_bias\": true"), "attention_bias"), "attention bias");
  require(has(refusal("\"mlp_bias\": false", "\"mlp_bias\": true"), "attention_bias"), "mlp bias");
  require(has(refusal("\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true"), "tie_word_embeddings"),
          "tied embeddings");
  require(has(refusal("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\""), "hidden_act"), "activation");
  require(has(refusal("\"initializer_range\": 0.0177", "\"initializer_range\": 0"), "initializer_range"),
          "initializer range");
  require(has(refusal("\"num_attention_heads\": 16, \"num_key_value_heads\": 4",
                      "\"num_attention_heads\": 15, \"num_key_value_heads\": 4"),
              "num_key_value_heads"),
          "head divisibility");
  require(has(refusal("\"head_dim\": 256", "\"head_dim\": 255"), "head_dim"), "odd head dim");
  // RoPE: only the default recipe, only the full rotary width.
  require(has(refusal("\"rope_scaling\": null", "\"rope_scaling\": {\"type\": \"yarn\"}"), "rope_scaling"),
          "rope scaling");
  require(has(refusal("\"rope_theta\": 10000", "\"rope_theta\": 0"), "rope_theta"), "rope theta");
  require(has(refusal("\"partial_rotary_factor\": 1.0},\n    \"full_attention\"",
                      "\"partial_rotary_factor\": 0.5},\n    \"full_attention\""),
              "partial_rotary_factor"),
          "partial rotary");
  // Quantization: only the W4A16 modelopt NVFP4 release, only group 16.
  require(has(refusal("\"quant_algo\": \"W4A16_NVFP4\"", "\"quant_algo\": \"W4A4_NVFP4\""), "quant_algo"), "algo");
  require(has(refusal("\"quant_method\": \"modelopt\"", "\"quant_method\": \"compressed-tensors\""), "quant_method"),
          "method");
  require(has(refusal("\"group_size\": 16", "\"group_size\": 32"), "group_size"), "group size");
  require(has(refusal("\"ignore\": [\"backbone.embeddings\", \"lm_head\"]",
                      "\"ignore\": [\"backbone.embeddings\"]"),
              "ignore"),
          "ignore coverage");
  require(has(refusal("\"ignore\": [\"backbone.embeddings\", \"lm_head\"]",
                      "\"ignore\": [\"backbone.embeddings\", \"lm_head\", \"backbone.rotary_emb\"]"),
              "unexpected ignored"),
          "ignore extra");
}

DGPP_TEST(dory_config_reads_the_landed_checkpoint) {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) return;
  const fs::path root =
      fs::path(home) / ".cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4/snapshots";
  if (!fs::is_directory(root)) return;
  for (const auto& snap : fs::directory_iterator(root)) {
    const fs::path cfg = snap.path() / "config.json";
    if (!fs::exists(cfg)) continue;
    require(dgpp::detect_architecture_file(cfg.string()) == dgpp::ModelArchitecture::Dory, "arch");
    const dgpp::DoryConfig c = dgpp::DoryConfig::from_json_file(cfg.string());
    require(c.num_hidden_layers == 72 && c.n_recurrent_loops == 2 && c.n_recurrent_layers == 40, "landed values");
    require_near(c.alpha_scale(), 2.8248588, 1e-6, "landed alpha_scale");
    return;
  }
}
