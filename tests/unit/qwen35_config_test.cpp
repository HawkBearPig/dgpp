// The Qwen3.8-27B config parser: the release's values parse, the NVFP4
// mixed release is accepted as such, and the unsupported shapes are refused
// by name. The architecture registry detects qwen3_5.
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "loaders/architecture.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen/binding35.hpp"
#include "models/qwen/config35.hpp"
#include "qwen35_config_json.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

dgpp::Qwen35TextConfig parse(const std::string& text, const std::string& quant) {
  const auto t = dgpp::minijson::parse(text);
  const auto q = dgpp::minijson::parse(quant);
  return dgpp::Qwen35TextConfig::parse(t.root, &q.root);
}

std::string refusal(const std::string& text, const std::string& quant) {
  try {
    (void)parse(text, quant);
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}

std::string patched(const std::string& from, const std::string& to) {
  std::string s = qwen35_fixture::text_json();
  const size_t at = s.find(from);
  require(at != std::string::npos, "patch anchor missing: " + from);
  s.replace(at, from.size(), to);
  return s;
}

}  // namespace

DGPP_TEST(qwen35_config_parses_the_release) {
  const dgpp::Qwen35TextConfig c =
      parse(qwen35_fixture::text_json(), qwen35_fixture::kQuantFp8);
  require(c.hidden_size == 5120, "hidden");
  require(c.vocab_size == 248320, "vocab");
  require(c.num_hidden_layers == 64, "layers");
  require(c.num_gdn_layers() == 48, "gdn count");
  require(c.num_full_layers() == 16, "full count");
  require(c.mtp_layer() == 64, "mtp layer");
  require(c.rotary_dim == 64, "rotary");
  require(c.mrope_section == std::vector<int>({11, 11, 10}), "mrope");
  require(c.output_gate_type == "swish", "gate");
  require(c.intermediate_size == 17408, "mlp");
  require(c.num_attention_heads == 24 && c.num_key_value_heads == 4, "heads");
  require(c.head_dim == 256, "head dim");
  require(c.quant_kind == dgpp::Qwen35QuantKind::Fp8Block, "quant");
  require(c.eos_token_ids == std::vector<int64_t>({248044}), "eos");
  require(c.context_limit() == 262144, "context");
}

DGPP_TEST(qwen35_config_accepts_nvfp4_mixed) {
  const dgpp::Qwen35TextConfig c =
      parse(qwen35_fixture::text_json(), qwen35_fixture::kQuantNvfp4Mixed);
  require(c.quant_kind == dgpp::Qwen35QuantKind::Nvfp4Mixed, "quant kind");
  require(c.hidden_size == 5120, "hidden");
  // Group resolution: the ignore list, then the first group whose targets
  // match (group_0's late-MLP regex wins over group_1's blanket one).
  using Q = dgpp::Qwen35TensorQuant;
  require(c.tensor_quant("model.language_model.layers.0.self_attn.q_proj") == Q::Fp8Channel,
          "attn fp8");
  require(c.tensor_quant("model.language_model.layers.2.linear_attn.in_proj_qkv") == Q::Fp8Channel,
          "gdn fp8");
  require(c.tensor_quant("model.language_model.layers.0.mlp.gate_proj") == Q::Nvfp4, "early mlp fp4");
  require(c.tensor_quant("model.language_model.layers.56.mlp.down_proj") == Q::Fp8Channel,
          "late mlp fp8");
  require(c.tensor_quant("mtp.layers.0.mlp.up_proj") == Q::Bf16, "draft bf16");
  require(c.tensor_quant("mtp.layers.0.self_attn.o_proj") == Q::Bf16, "draft attn bf16");
  require(c.tensor_quant("lm_head") == Q::Fp8Channel, "head fp8");
  require(c.tensor_quant("model.language_model.embed_tokens") == Q::Bf16, "embed bf16");
}

DGPP_TEST(qwen35_config_plain_targets_are_exact) {
  // A plain (non-"re:") target is a literal module name: an anchored exact
  // match, its dots matching themselves. A glob target cannot be read as a
  // literal and is refused, never left to silently claim nothing.
  using Q = dgpp::Qwen35TensorQuant;
  const std::string quant = R"({"quant_method": "compressed-tensors", "format": "mixed-precision",
      "config_groups": {
        "group_0": {"format": "float-quantized",
                    "targets": ["model.language_model.layers.0.self_attn.q_proj", "re:.*lm_head"],
                    "weights": {"num_bits": 8, "type": "float", "strategy": "channel"}},
        "group_1": {"format": "nvfp4-pack-quantized",
                    "targets": ["re:.*mlp\\.(gate|up|down)_proj$"],
                    "weights": {"num_bits": 4, "type": "float", "group_size": 16}}},
      "ignore": ["re:^mtp.*"]})";
  const auto c = parse(qwen35_fixture::text_json(), quant);
  require(c.tensor_quant("model.language_model.layers.0.self_attn.q_proj") == Q::Fp8Channel,
          "plain exact target");
  require(c.tensor_quant("modelXlanguage_model.layers.0.self_attn.q_proj") == Q::Bf16,
          "the dot matches only itself");
  const std::string glob = R"({"quant_method": "compressed-tensors", "format": "mixed-precision",
      "config_groups": {
        "group_0": {"format": "float-quantized",
                    "targets": ["model.language_model.layers.*.self_attn.q_proj"],
                    "weights": {"num_bits": 8, "type": "float", "strategy": "channel"}},
        "group_1": {"format": "nvfp4-pack-quantized",
                    "targets": ["re:.*mlp\\.(gate|up|down)_proj$"],
                    "weights": {"num_bits": 4, "type": "float", "group_size": 16}}}})";
  require(refusal(qwen35_fixture::text_json(), glob).find("plain entries") != std::string::npos,
          "glob target refused by name");
}

DGPP_TEST(qwen35_config_refusals_name_the_field) {
  using namespace qwen35_fixture;
  require(refusal(patched("\"qwen3_5_text\"", "\"qwen4_exp_text\""), kQuantFp8)
              .find("model_type") != std::string::npos,
          "model_type");
  require(refusal(patched("\"linear_attention\"", "\"sliding_attention\""), kQuantFp8)
              .find("layer_types") != std::string::npos,
          "layer type");
  require(refusal(patched("\"num_hidden_layers\": 64", "\"num_hidden_layers\": 63"), kQuantFp8)
              .find("layer_types") != std::string::npos,
          "layer count");
  require(refusal(patched("\"full_attention_interval\": 4", "\"full_attention_interval\": 5"),
                  kQuantFp8)
              .find("full_attention_interval") != std::string::npos,
          "interval");
  require(refusal(patched("\"head_dim\": 256", "\"head_dim\": 128"), kQuantFp8)
              .find("head_dim") != std::string::npos,
          "head dim");
  require(refusal(patched("\"output_gate_type\": \"swish\"", "\"output_gate_type\": \"sigmoid\""),
                  kQuantFp8)
              .find("output_gate_type") != std::string::npos,
          "gate");
  require(refusal(text_json(), R"({"quant_method": "linear"})").find("quant_method") !=
              std::string::npos,
          "quant method");
  require(refusal(text_json(), R"({"quant_method": "fp8", "activation_scheme": "dynamic",
      "weight_block_size": [64, 64]})")
              .find("weight_block_size") != std::string::npos,
          "block size");
  require(refusal(patched("\"mtp_num_hidden_layers\": 1", "\"mtp_num_hidden_layers\": 2"),
                  kQuantFp8)
              .find("mtp_num_hidden_layers") != std::string::npos,
          "mtp count");
  require(refusal(patched("\"mtp_use_dedicated_embeddings\": false",
                          "\"mtp_use_dedicated_embeddings\": true"),
                  kQuantFp8)
              .find("mtp_use_dedicated_embeddings") != std::string::npos,
          "dedicated embeddings");
}

DGPP_TEST(qwen35_architecture_detects_the_release) {
  const auto root = dgpp::minijson::parse(
      R"({"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"})");
  require(dgpp::detect_architecture(root.root) == dgpp::ModelArchitecture::Qwen3_5,
          "detect qwen3_5");
  require(std::string(dgpp::model_architecture_name(dgpp::ModelArchitecture::Qwen3_5)) ==
              "qwen3_5",
          "name");
}

DGPP_TEST(qwen35_config_accepts_modelopt_moe) {
  const std::string text = qwen35_fixture::moe_text_json();
  const std::string quant = qwen35_fixture::kQuantModeloptNvfp4;
  const auto t = dgpp::minijson::parse(text);
  const auto q = dgpp::minijson::parse(quant);
  const dgpp::Qwen35TextConfig c = dgpp::Qwen35TextConfig::parse(t.root, &q.root);
  require(c.is_moe, "moe");
  require(c.hidden_size == 3072, "hidden");
  require(c.num_hidden_layers == 48, "layers");
  require(c.num_gdn_layers() == 36, "gdn count");
  require(c.num_full_layers() == 12, "full count");
  require(c.num_experts == 256 && c.num_experts_per_tok == 8, "experts");
  require(c.moe_intermediate_size == 1024 && c.shared_expert_intermediate_size == 1024,
          "inter");
  require(c.norm_topk_prob, "norm_topk default");
  require(c.quant_kind == dgpp::Qwen35QuantKind::Nvfp4Moe, "quant kind");
  using Q = dgpp::Qwen35TensorQuant;
  // ModelOpt glob ignore keeps attention/shared/head BF16; experts claim NVFP4.
  require(c.tensor_quant("model.language_model.layers.0.mlp.experts.0.gate_proj") == Q::Nvfp4,
          "expert fp4");
  require(c.tensor_quant("model.language_model.layers.0.linear_attn.in_proj_qkv") == Q::Bf16,
          "gdn bf16");
  require(c.tensor_quant("model.language_model.layers.3.self_attn.q_proj") == Q::Bf16,
          "attn bf16");
  require(c.tensor_quant("model.language_model.layers.0.mlp.shared_expert.gate_proj") ==
              Q::Bf16,
          "shared bf16");
  require(c.tensor_quant("lm_head") == Q::Bf16, "head bf16");
  require(c.tensor_quant("mtp.layers.0.mlp.experts.0.gate_proj") == Q::Bf16, "mtp bf16");
  dgpp::qwen35_tp_validate_geometry(c, 0, 1);
  dgpp::qwen35_tp_validate_geometry(c, 3, 4);
}
