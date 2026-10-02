// The Qwen3.8-27B config parser: the release's values parse, the NVFP4
// mixed release is accepted as such, and the unsupported shapes are refused
// by name. The architecture registry detects qwen3_5.
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "loaders/architecture.hpp"
#include "loaders/minijson.hpp"
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
