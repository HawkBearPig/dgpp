// The Qwen3.8-Flash-Next config parser: the real file's
// values parse, the derived n-gram geometry reproduces the checkpoint's
// stored hash buffers (verified against the landed shards on 2026-09-09),
// and the unsupported shapes are refused by name.
#include <filesystem>
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "loaders/architecture.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen/config.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// The checkpoint's text_config and quantization_config, transcribed
// (Qwen/Qwen3.8-Flash-Next-FP8 @ 236dfdf2), vision omitted.
const char* kText = R"({
  "model_type": "qwen4_exp_text", "attention_bias": false, "attention_dropout": 0.0,
  "bos_token_id": 248044, "eos_token_id": 248044, "dtype": "bfloat16",
  "full_attention_interval": 4, "hc_count": 4, "hc_lowrank": 320, "head_dim": 256,
  "heads_per_ngram": 8, "hidden_act": "silu", "hidden_size": 2560,
  "indexer_budget": 2048, "indexer_compress_ratio": 4, "indexer_head_dim": 128,
  "indexer_kv_heads": 1, "indexer_n_heads": 4, "initializer_range": 0.02,
  "layer_types": [LAYERS],
  "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
  "linear_num_value_heads": 48, "linear_value_head_dim": 128,
  "make_ngram_vocab_size_divisible_by": 128, "mamba_ssm_dtype": "float32",
  "max_position_embeddings": 262144, "moe_intermediate_size": 640,
  "mtp": {"hybrid": true, "layer_types": ["full_attention"], "mtp_use_hidden_state_from_layer": null,
          "num_hidden_layers": 1, "rope_theta": 10000000},
  "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false,
  "ngram_size": 3, "ngram_vocab_size_base": 20000000, "num_attention_heads": 24,
  "num_experts": 512, "num_experts_per_tok": 10, "num_hidden_layers": 48,
  "num_key_value_heads": 2, "output_gate_type": "sigmoid", "output_router_logits": false,
  "pad_token_id": null, "partial_rotary_factor": 0.25, "ple_conv_kernel_size": 4,
  "ple_embed_dim": 2560, "ple_layer_ids": [2], "rms_norm_eps": 1e-06,
  "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10],
                      "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
  "router_aux_loss_coef": 0.001, "shared_expert_intermediate_size": 640,
  "split_ngram_parts": 128, "tie_word_embeddings": false, "use_cache": true,
  "vocab_size": 248320
})";
const char* kQuant = R"({"quant_method": "fp8", "activation_scheme": "dynamic",
  "weight_per_tensor": false, "act_per_tensor": false, "weight_block_size": [128, 128],
  "modules_to_not_convert": ["lm_head"], "modules_to_convert": ["ple.ple_embedding.ngram_embedding"]})";

std::string layers_json() {
  std::string s;
  for (int i = 0; i < 48; ++i) {
    if (i) s += ", ";
    s += (i + 1) % 4 == 0 ? "\"full_attention\"" : "\"linear_attention\"";
  }
  return s;
}

std::string text_json(const std::string& patch_from = "", const std::string& patch_to = "") {
  std::string s = kText;
  s.replace(s.find("[LAYERS]"), 8, "[" + layers_json() + "]");
  if (!patch_from.empty()) {
    const size_t at = s.find(patch_from);
    require(at != std::string::npos, "patch anchor missing: " + patch_from);
    s.replace(at, patch_from.size(), patch_to);
  }
  return s;
}

dgpp::QwenTextConfig parse(const std::string& text, const std::string& quant = kQuant) {
  const auto t = dgpp::minijson::parse(text);
  const auto q = dgpp::minijson::parse(quant);
  return dgpp::QwenTextConfig::parse(t.root, &q.root);
}

std::string refusal(const std::string& text, const std::string& quant = kQuant) {
  try {
    (void)parse(text, quant);
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}

}  // namespace

DGPP_TEST(qwen_config_parses_the_release) {
  const dgpp::QwenTextConfig c = parse(text_json());
  require(c.hidden_size == 2560 && c.vocab_size == 248320 && c.num_hidden_layers == 48, "shape");
  require(c.num_gdn_layers() == 36 && c.num_qsa_layers() == 12, "36 GDN + 12 QSA");
  require(c.layers[3] == dgpp::QwenLayerKind::Qsa && c.layers[47] == dgpp::QwenLayerKind::Qsa &&
              c.layers[0] == dgpp::QwenLayerKind::Gdn,
          "layer kinds");
  require(c.mtp_layer() == 48, "the draft layer index");
  require(c.ple_layer() == 1, "the PLE on layer index 1 (one-indexed 2)");
  require(c.rotary_dim == 64 && c.rope_theta == 1e7, "rope");
  require(c.mrope_section.size() == 3 && c.mrope_interleaved, "mrope");
  require(c.indexer_block_topk() == 512, "top-512 blocks");
  require(c.hyper_width() == 10240, "4 x 2560");
  require(c.eos_token_ids.size() == 1 && c.eos_token_ids[0] == 248044, "eos");
  require(c.experts_fp8 && c.ngram_table_fp8, "fp8 classes");
  // The rope knob is the engine's, never the checkpoint's: the parsed
  // config carries none and reports the checkpoint's own ceiling.
  require(!c.rope_scaling.has_value(), "no rope scaling from the checkpoint");
  require(c.context_limit() == 262144, "the checkpoint's ceiling");
}

DGPP_TEST(qwen_config_rope_scaling_knob_lifts_the_ceiling) {
  // The parsed config is the plain one; the serving layer sets
  // cfg.rope_scaling from engine.rope_scaling, and everything that bounds
  // a context reads context_limit() instead of the raw field.
  dgpp::QwenTextConfig c = parse(text_json());
  require(c.max_position_embeddings == 262144 && c.context_limit() == 262144, "plain");
  dgpp::RopeScaling rs;
  rs.factor = 2.0;
  rs.original_max_position_embeddings = 262144;
  c.rope_scaling = rs;
  require(c.context_limit() == 524288, "the 512K ceiling");
  c.rope_scaling->factor = 4.0;
  require(c.context_limit() == 1048576, "1M");
}

DGPP_TEST(qwen_config_derives_the_ngram_table_the_checkpoint_stores) {
  const dgpp::QwenTextConfig c = parse(text_json());
  const dgpp::QwenNgramGeometry g = c.ngram_geometry();
  require(g.heads == 16 && g.head_dim == 160, "16 heads of 160");
  const int64_t primes[16] = {20000003, 20000023, 20000033, 20000047, 20000059, 20000063,
                              20000069, 20000077, 20000081, 20000093, 20000107, 20000147,
                              20000153, 20000159, 20000161, 20000171};
  int64_t off = 0;
  for (int h = 0; h < 16; ++h) {
    require(g.head_vocab[h] == primes[h], "head vocab " + std::to_string(h));
    require(g.head_offset[h] == off, "head offset " + std::to_string(h));
    off += primes[h];
  }
  require(g.total_rows == 320001446 && g.padded_rows == 320001536, "rows");
  // The checkpoint's layer_multipliers (read from the landed shard 2026-09-09).
  require(g.multipliers.size() == 3 && g.multipliers[0] == 23703573157769 &&
              g.multipliers[1] == 20109073645365 && g.multipliers[2] == 8052911324071,
          "hash multipliers");
}

DGPP_TEST(qwen_config_refuses_what_the_engine_does_not_implement) {
  require(refusal(text_json("\"hc_count\": 4", "\"hc_count\": 3")).find("hc_count") != std::string::npos,
          "3 branches refused");
  require(refusal(text_json("\"output_gate_type\": \"sigmoid\"", "\"output_gate_type\": \"silu\""))
                  .find("output_gate_type") != std::string::npos,
          "silu gate refused");
  require(refusal(text_json("\"indexer_kv_heads\": 1", "\"indexer_kv_heads\": 2"))
                  .find("indexer_kv_heads") != std::string::npos,
          "two indexer key heads refused");
  require(refusal(text_json("\"num_experts_per_tok\": 10", "\"num_experts_per_tok\": 20"))
                  .find("num_experts_per_tok") != std::string::npos,
          "top-20 refused (the accumulator bound is 16)");
  require(refusal(text_json("\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true"))
                  .find("tie_word_embeddings") != std::string::npos,
          "tied embeddings refused");
  require(refusal(text_json("\"full_attention_interval\": 4", "\"full_attention_interval\": 3"))
                  .find("full_attention_interval") != std::string::npos,
          "an interval disagreeing with layer_types refused");
  require(refusal(text_json(), R"({"quant_method": "fp8", "weight_block_size": [64, 64]})")
                  .find("weight_block_size") != std::string::npos,
          "a 64-block release refused");
  require(refusal(text_json(), "null").find("quantization_config") != std::string::npos,
          "the BF16 release refused (no BF16 expert path)");
  // The checkpoint's own rope must be plain: a scaled one is the engine's
  // knob to apply (engine.rope_scaling), and taking it from here would
  // scale twice.
  require(refusal(text_json("\"rope_type\": \"default\"", "\"rope_type\": \"yarn\""))
                  .find("rope_type") != std::string::npos,
          "a checkpoint that bakes in a rope scaling refused");
  // A file without the PLE parses too (the geometry is then empty).
  const dgpp::QwenTextConfig no_ple = parse(text_json("\"ple_layer_ids\": [2]", "\"ple_layer_ids\": []"));
  require(no_ple.ple_layer() == -1 && no_ple.ngram_geometry().heads == 0, "no PLE");
}

DGPP_TEST(qwen_config_parses_the_landed_checkpoint_when_present) {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) return;
  const fs::path root = fs::path(home) / ".cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots";
  if (!fs::is_directory(root)) return;  // the gate is optional: a box without the checkpoint
  for (const auto& snap : fs::directory_iterator(root)) {
    const fs::path cfg = snap.path() / "config.json";
    if (!fs::exists(cfg)) continue;
    require(dgpp::detect_architecture_file(cfg.string()) == dgpp::ModelArchitecture::Qwen4Exp,
            "architecture detection");
    const dgpp::QwenTextConfig c = dgpp::QwenTextConfig::from_json_file(cfg.string());
    require(c.num_hidden_layers == 48 && c.num_experts == 512 && c.ple_layer() == 1, "the file's shape");
    require(c.ngram_geometry().padded_rows == 320001536, "the file's n-gram rows");
  }
}

DGPP_TEST(architecture_detection_names_the_families) {
  const auto glm = dgpp::minijson::parse(R"({"architectures": ["Glm5ForConditionalGeneration"], "model_type": "glm_moe_dsa"})");
  require(dgpp::detect_architecture(glm.root) == dgpp::ModelArchitecture::Glm5, "glm");
  const auto qwen = dgpp::minijson::parse(R"({"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp"})");
  require(dgpp::detect_architecture(qwen.root) == dgpp::ModelArchitecture::Qwen4Exp, "qwen");
  bool refused = false;
  try {
    const auto other = dgpp::minijson::parse(R"({"architectures": ["LlamaForCausalLM"]})");
    (void)dgpp::detect_architecture(other.root);
  } catch (const std::runtime_error&) {
    refused = true;
  }
  require(refused, "an unknown family is refused");
}
