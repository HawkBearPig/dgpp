// Host-only tests for the GLM text-config parser (M4): every field the
// assembly consumes is parsed or rejected, the redundant layer-class
// encodings must agree, and the produced Kda/Dsa geometry configs carry the
// trained values rather than the header defaults.
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>

#include "../common/temp_dir.hpp"
#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "models/glm/config.hpp"

namespace {

using dgpp::GlmLayerKind;
using dgpp::GlmMlpKind;
using dgpp::GlmGenerationDefaults;
using dgpp::GlmTextConfig;

// A minimal, internally consistent text_config exercising every class:
// layers 0-5 with DSA at 2 and 4; dense MLP on the first two layers only.
// Field values differ from the GLM-5.3-Flash defaults where legal so a
// missed parse cannot hide behind a matching default.
const char* kTinyConfig = R"json({
  "hidden_size": 512, "vocab_size": 1000, "num_hidden_layers": 6,
  "rms_norm_eps": 1e-6, "tie_word_embeddings": false,
  "hidden_act": "silu", "swiglu_limit": 7.5,
  "layer_types": ["linear_attention", "linear_attention",
                  "deepseek_sparse_attention", "linear_attention",
                  "deepseek_sparse_attention", "linear_attention"],
  "mlp_layer_types": ["dense", "dense", "sparse", "sparse", "sparse", "sparse"],
  "indexer_types": ["full", "full", "full", "full", "full", "full"],
  "first_k_dense_replace": 2,
  "linear_attn_config": {
    "num_heads": 8, "head_dim": 64, "short_conv_kernel_size": 4,
    "gate_lower_bound": -3.5,
    "kda_layers": [0, 1, 3, 5], "full_attn_layers": [2, 4]
  },
  "num_attention_heads": 8, "q_lora_rank": 256, "kv_lora_rank": 128,
  "qk_nope_head_dim": 96, "qk_rope_head_dim": 0, "v_head_dim": 96,
  "mla_use_nope": true,
  "index_n_heads": 4, "index_head_dim": 128, "index_kpool": 4,
  "index_topk": 128, "index_kpool_compress": true,
  "index_kpool_always_select_tail": true, "indexer_rope_interleave": true,
  "intermediate_size": 1024, "moe_intermediate_size": 256,
  "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
  "scoring_func": "sigmoid", "topk_method": "noaux_tc",
  "norm_topk_prob": true, "routed_scaling_factor": 1.5,
  "n_group": 1, "topk_group": 1, "moe_router_dtype": "float32",
  "mhc": true, "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-06,
  "num_nextn_predict_layers": 1
})json";

GlmTextConfig parse_ok(const char* json = kTinyConfig) {
  auto parsed = dgpp::minijson::parse(json);
  return GlmTextConfig::parse(parsed.root);
}

GlmTextConfig parse_replacing(const std::string& find,
                              const std::string& replace) {
  std::string json = kTinyConfig;
  size_t pos = json.find(find);
  if (pos == std::string::npos)
    throw std::runtime_error("test bug: fixture lacks '" + find + "'");
  json.replace(pos, find.size(), replace);
  auto parsed = dgpp::minijson::parse(json);
  return GlmTextConfig::parse(parsed.root);
}

void expect_throw(std::function<void()> fn, const char* what) {
  try {
    fn();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(std::string("expected rejection: ") + what);
}

void require(bool cond, const char* what) {
  if (!cond) throw std::runtime_error(what);
}

}  // namespace

DGPP_TEST(glm_config_parses_every_class_and_field) {
  const GlmTextConfig c = parse_ok();
  require(c.hidden_size == 512, "assertion: c.hidden_size == 512");
  require(c.vocab_size == 1000, "assertion: c.vocab_size == 1000");
  require(c.num_hidden_layers == 6, "assertion: c.num_hidden_layers == 6");
  require(c.swiglu_limit == 7.5f, "assertion: c.swiglu_limit == 7.5f");
  require(c.layers.size() == 6 && c.mlps.size() == 6, "assertion: c.layers.size() == 6 && c.mlps.size() == 6");
  require(c.layers[2] == GlmLayerKind::Dsa, "assertion: c.layers[2] == GlmLayerKind::Dsa");
  require(c.layers[3] == GlmLayerKind::Kda, "assertion: c.layers[3] == GlmLayerKind::Kda");
  require(c.mlps[0] == GlmMlpKind::Dense && c.mlps[1] == GlmMlpKind::Dense, "assertion: c.mlps[0] == GlmMlpKind::Dense && c.mlps[1] == GlmMlpKind::Dense");
  require(c.mlps[2] == GlmMlpKind::Moe, "assertion: c.mlps[2] == GlmMlpKind::Moe");
  require(c.kda_num_heads == 8 && c.kda_head_dim == 64, "assertion: c.kda_num_heads == 8 && c.kda_head_dim == 64");
  require(c.kda_gate_lower_bound == -3.5f, "assertion: c.kda_gate_lower_bound == -3.5f");
  require(c.q_lora_rank == 256 && c.kv_lora_rank == 128, "assertion: c.q_lora_rank == 256 && c.kv_lora_rank == 128");
  require(c.qk_nope_head_dim == 96 && c.v_head_dim == 96, "assertion: c.qk_nope_head_dim == 96 && c.v_head_dim == 96");
  require(c.index_n_heads == 4 && c.index_topk == 128, "assertion: c.index_n_heads == 4 && c.index_topk == 128");
  require(c.intermediate_size == 1024, "assertion: c.intermediate_size == 1024");
  require(c.moe_intermediate_size == 256, "assertion: c.moe_intermediate_size == 256");
  require(c.n_routed_experts == 4 && c.num_experts_per_tok == 2, "assertion: c.n_routed_experts == 4 && c.num_experts_per_tok == 2");
  require(c.routed_scaling_factor == 1.5f, "assertion: c.routed_scaling_factor == 1.5f");
  require(c.hc_mult == 4 && c.mhc, "assertion: c.hc_mult == 4 && c.mhc");
  require(c.num_kda_layers() == 4 && c.num_dsa_layers() == 2, "assertion: c.num_kda_layers() == 4 && c.num_dsa_layers() == 2");
  require(c.mtp_layer() == 6, "assertion: c.mtp_layer() == 6");
}

DGPP_TEST(glm_config_geometry_configs_carry_trained_values) {
  const GlmTextConfig c = parse_ok();
  // The geometry structs must carry the parsed values, not their defaults
  // (kda_geometry.hpp explicitly warns against trusting them).
  const dgpp::KdaConfig k = c.kda_config();
  require(k.hidden == 512 && k.heads == 8 && k.head_dim == 64, "assertion: k.hidden == 512 && k.heads == 8 && k.head_dim == 64");
  require(k.lower_bound == -3.5f, "assertion: k.lower_bound == -3.5f");
  require(k.num_kda_layers == 4, "assertion: k.num_kda_layers == 4");
  const dgpp::DsaConfig d = c.dsa_config();
  require(d.hidden == 512 && d.num_heads == 8, "assertion: d.hidden == 512 && d.num_heads == 8");
  require(d.q_lora_rank == 256 && d.kv_lora_rank == 128, "assertion: d.q_lora_rank == 256 && d.kv_lora_rank == 128");
  require(d.qk_nope_head_dim == 96 && d.v_head_dim == 96, "assertion: d.qk_nope_head_dim == 96 && d.v_head_dim == 96");
  require(d.index_n_heads == 4 && d.index_topk == 128, "assertion: d.index_n_heads == 4 && d.index_topk == 128");
  require(d.num_dsa_layers == 2, "assertion: d.num_dsa_layers == 2");
}

DGPP_TEST(glm_config_rejects_inconsistent_layer_encodings) {
  // layer_types disagrees with full_attn_layers at layer 3.
  expect_throw(
      [&] {
        parse_replacing("\"linear_attention\",\n                  \"deepseek_sparse_attention\", \"linear_attention\",\n                  \"deepseek_sparse_attention\", \"linear_attention\"]",
                        "\"linear_attention\",\n                  \"deepseek_sparse_attention\", \"deepseek_sparse_attention\",\n                  \"deepseek_sparse_attention\", \"linear_attention\"]");
      },
      "layer_types vs full_attn_layers");
  // layer id lists do not cover all layers.
  expect_throw(
      [&] { parse_replacing("\"kda_layers\": [0, 1, 3, 5]", "\"kda_layers\": [0, 1, 3]"); },
      "incomplete kda/full partition");
  // overlapping id lists.
  expect_throw(
      [&] { parse_replacing("\"kda_layers\": [0, 1, 3, 5]", "\"kda_layers\": [0, 1, 2, 3, 5]"); },
      "overlapping id lists");
  // wrong layer_types length.
  expect_throw(
      [&] { parse_replacing("\"linear_attention\",\n                  \"linear_attention\",\n                  \"deepseek_sparse_attention\", \"linear_attention\",\n                  \"deepseek_sparse_attention\", \"linear_attention\"],",
                            "\"linear_attention\",\n                  \"linear_attention\",\n                  \"deepseek_sparse_attention\", \"linear_attention\",\n                  \"deepseek_sparse_attention\"],"); },
      "layer_types length");
  // dense MLP not exactly the prefix.
  expect_throw(
      [&] { parse_replacing("\"first_k_dense_replace\": 2", "\"first_k_dense_replace\": 3"); },
      "dense prefix");
  // sparse indexer variant is not implemented.
  expect_throw(
      [&] { parse_replacing("\"indexer_types\": [\"full\", \"full\", \"full\", \"full\", \"full\", \"full\"]",
                            "\"indexer_types\": [\"full\", \"full\", \"full\", \"full\", \"full\", \"sparse\"]"); },
      "sparse indexer type");
}

DGPP_TEST(glm_config_rejects_unsupported_values_loudly) {
  expect_throw([&] { parse_replacing("\"mhc\": true", "\"mhc\": false"); },
               "non-mHC checkpoint");
  expect_throw(
      [&] { parse_replacing("\"qk_rope_head_dim\": 0", "\"qk_rope_head_dim\": 64"); },
      "rope path");
  expect_throw(
      [&] { parse_replacing("\"scoring_func\": \"sigmoid\"", "\"scoring_func\": \"softmax\""); },
      "scoring_func");
  expect_throw(
      [&] { parse_replacing("\"topk_method\": \"noaux_tc\"", "\"topk_method\": \"greedy\""); },
      "topk_method");
  expect_throw(
      [&] { parse_replacing("\"n_group\": 1", "\"n_group\": 8"); },
      "grouped routing");
  expect_throw(
      [&] { parse_replacing("\"n_shared_experts\": 1", "\"n_shared_experts\": 2"); },
      "multiple shared experts");
  expect_throw(
      [&] { parse_replacing("\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true"); },
      "tied embeddings");
  expect_throw(
      [&] { parse_replacing("\"num_nextn_predict_layers\": 1", "\"num_nextn_predict_layers\": 2"); },
      "multi-draft MTP");
  expect_throw(
      [&] { parse_replacing("\"index_kpool\": 4", "\"index_kpool\": 3"); },
      "kpool template");
  expect_throw(
      [&] { parse_replacing("\"moe_router_dtype\": \"float32\"", "\"moe_router_dtype\": \"bfloat16\""); },
      "router dtype");
  expect_throw(
      [&] { parse_replacing("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\""); },
      "hidden_act");
  expect_throw([&] { parse_replacing("\"moe_intermediate_size\": 256,", ""); },
               "missing moe field");
}

DGPP_TEST(glm_config_num_nextn_zero_disables_mtp_layer) {
  const GlmTextConfig c =
      parse_replacing("\"num_nextn_predict_layers\": 1", "\"num_nextn_predict_layers\": 0");
  require(c.mtp_layer() == -1, "assertion: c.mtp_layer() == -1");
}

DGPP_TEST(glm_config_mhc_fields_parse_and_propagate) {
  const GlmTextConfig c =
      parse_replacing("\"hc_sinkhorn_iters\": 20, \"hc_eps\": 1e-06",
                      "\"hc_sinkhorn_iters\": 7, \"hc_eps\": 5e-07");
  require(c.hc_sinkhorn_iters == 7, "assertion: hc_sinkhorn_iters == 7");
  require(c.hc_eps == 5e-7f, "assertion: hc_eps == 5e-7f");
  const dgpp::GlmMhcConfig m = c.mhc_config();
  require(m.hc_mult == 4 && m.hidden == 512, "assertion: mhc geometry");
  require(m.sinkhorn_iters == 7 && m.hc_eps == 5e-7f,
          "assertion: mhc trained values");
  require(m.norm_eps == 1e-6f, "assertion: mhc norm eps");
  // hc_mult != 4 is rejected by the kernel geometry pin (like index_kpool).
  expect_throw(
      [&] { parse_replacing("\"mhc\": true, \"hc_mult\": 4", "\"mhc\": true, \"hc_mult\": 8"); },
      "hc_mult 8");
  expect_throw(
      [&] { parse_replacing("\"hc_sinkhorn_iters\": 20", "\"hc_sinkhorn_iters\": 0"); },
      "sinkhorn iters 0");
}

DGPP_TEST(glm_generation_config_parses_sampling_defaults_and_eos) {
  const auto parsed = dgpp::minijson::parse(R"json({
    "do_sample": true,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 128,
    "min_p": 0.02,
    "repetition_penalty": 1.1,
    "eos_token_id": [7, 11, 13],
    "transformers_version": "ignored cold-path metadata"
  })json");
  const GlmGenerationDefaults c =
      GlmGenerationDefaults::parse(parsed.root, /*vocab_size=*/1000);
  require(c.file_found, "direct parse represents a present file");
  require(c.do_sample.has_value() && *c.do_sample, "do_sample");
  require(c.effective_temperature() == 1.0f, "temperature");
  require(c.effective_top_p() == 0.95f, "top_p");
  require(c.effective_top_k() == 128, "top_k");
  require(c.effective_min_p() == 0.02f, "min_p");
  require(c.effective_repetition_penalty() == 1.1f,
          "repetition_penalty");
  require(c.eos_token_ids.has_value() &&
              *c.eos_token_ids == std::vector<int64_t>({7, 11, 13}),
          "eos ids");
  require(c.fallback_fields().empty(), "no parsed field should fall back");
}

DGPP_TEST(glm_generation_config_missing_fields_are_greedy_safe_and_named) {
  const auto parsed = dgpp::minijson::parse(
      R"json({"temperature": null, "top_p": null, "do_sample": false})json");
  const GlmGenerationDefaults c =
      GlmGenerationDefaults::parse(parsed.root, /*vocab_size=*/32);
  require(c.effective_temperature() == 0.0f, "missing temperature is greedy");
  require(c.effective_top_p() == 1.0f, "missing top_p is neutral");
  require(c.effective_top_k() == 0, "missing top_k is neutral");
  require(c.effective_min_p() == 0.0f, "missing min_p is neutral");
  require(c.effective_repetition_penalty() == 1.0f,
          "missing repetition penalty is neutral");
  const std::vector<std::string> missing = c.fallback_fields();
  require(missing == std::vector<std::string>(
                         {"temperature", "top_p", "top_k", "min_p",
                          "repetition_penalty"}),
          "fallback fields must be complete and stable-order");
  require(!c.eos_token_ids.has_value(), "missing eos remains distinguishable");
}

DGPP_TEST(glm_generation_config_do_sample_false_forces_greedy) {
  const auto parsed = dgpp::minijson::parse(
      R"json({"do_sample":false,"temperature":0.7,"top_p":0.9})json");
  const GlmGenerationDefaults c =
      GlmGenerationDefaults::parse(parsed.root, /*vocab_size=*/16);
  require(c.temperature.has_value() && *c.temperature == 0.7f,
          "configured temperature remains inspectable");
  require(c.effective_temperature() == 0.0f,
          "do_sample=false must select greedy semantics");
}

DGPP_TEST(glm_generation_config_rejects_invalid_present_values) {
  const auto reject_json = [](const char* json, const char* what) {
    expect_throw(
        [&] {
          const auto parsed = dgpp::minijson::parse(json);
          (void)GlmGenerationDefaults::parse(parsed.root, 32);
        },
        what);
  };
  reject_json(R"json({"temperature":-0.1})json", "negative temperature");
  reject_json(R"json({"temperature":"hot"})json", "string temperature");
  reject_json(R"json({"top_p":0})json", "zero top_p");
  reject_json(R"json({"top_p":1.01})json", "top_p above one");
  reject_json(R"json({"top_k":1.5})json", "fractional top_k");
  reject_json(R"json({"top_k":-1})json", "negative top_k");
  reject_json(R"json({"min_p":-0.1})json", "negative min_p");
  reject_json(R"json({"repetition_penalty":0})json",
              "zero repetition penalty");
  reject_json(R"json({"do_sample":1})json", "numeric do_sample");
  reject_json(R"json({"eos_token_id":[1,32]})json", "out-of-range eos");
  reject_json(R"json({"eos_token_id":1.0})json", "non-integer eos");
}

DGPP_TEST(glm_generation_config_checkpoint_load_allows_only_absence) {
  namespace fs = std::filesystem;
  const dgpp::test::TempDir fixture("dgpp_generation_config_test");
  const fs::path& dir = fixture.path();

  const GlmGenerationDefaults missing =
      GlmGenerationDefaults::from_checkpoint_dir(dir.string(), 64);
  require(!missing.file_found, "absent file must select explicit fallback");
  require(missing.effective_temperature() == 0.0f,
          "absent file must be greedy");

  {
    std::ofstream out(dir / "generation_config.json");
    out << R"json({"temperature":0.8,"top_p":0.9,"top_k":0,
                   "min_p":0.0,"repetition_penalty":1.0,
                   "eos_token_id":63})json";
  }
  const GlmGenerationDefaults loaded =
      GlmGenerationDefaults::from_checkpoint_dir(dir.string(), 64);
  require(loaded.file_found, "present file");
  require(loaded.eos_token_ids.has_value() &&
              *loaded.eos_token_ids == std::vector<int64_t>({63}),
          "scalar eos id");

  {
    std::ofstream out(dir / "generation_config.json");
    out << R"json({"temperature":"bad"})json";
  }
  expect_throw(
      [&] {
        (void)GlmGenerationDefaults::from_checkpoint_dir(dir.string(), 64);
      },
      "malformed present generation config");
  fs::remove_all(dir);
}
