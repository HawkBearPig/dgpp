// The DeepSeek-V4-Flash config parser: the 0731 release's values parse, the
// compression schedule (window / ratio 4 / ratio 128, the cache and index
// ordinals), the hash layers, the DSpark facts and the MoE configuration
// derive from the file, the architecture registry dispatches on it, and
// the unsupported shapes are refused by name.
#include <filesystem>
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "dsv4_config_json.hpp"
#include "loaders/architecture.hpp"
#include "loaders/minijson.hpp"
#include "models/dsv4/config.hpp"

namespace {

using dsv4_test::config_json;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

dgpp::Dsv4Config parse(const std::string& text) {
  const auto t = dgpp::minijson::parse(text);
  return dgpp::Dsv4Config::parse(t.root);
}

std::string refusal(const std::string& text) {
  try {
    (void)parse(text);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

bool has(const std::string& msg, const char* needle) { return msg.find(needle) != std::string::npos; }

const char* kRatios =
    "\"compress_ratios\": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, "
    "128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0, 0, 0]";

}  // namespace

DGPP_TEST(dsv4_config_parses_the_release) {
  const dgpp::Dsv4Config c = parse(config_json());
  require(c.hidden_size == 4096 && c.vocab_size == 129280 && c.num_hidden_layers == 43, "shape");
  // The config's own num_nextn_predict_layers is a stale 1: the draft
  // stages are the schedule's entries past the backbone.
  require(c.draft_stages == 3 && c.max_layer() == 46, "draft stages");
  require(c.rms_norm_eps == 1e-6f && c.max_position_embeddings == 1048576, "eps / positions");
  require(c.bos_token_id == 0 && c.eos_token_id == 1, "tokens");
  require(c.num_attention_heads == 64 && c.num_key_value_heads == 1 && c.head_dim == 512 && c.qk_rope_head_dim == 64,
          "attention dims");
  require(c.q_lora_rank == 1024 && c.o_lora_rank == 1024 && c.o_groups == 8 && c.heads_per_group() == 8, "lora ranks");
  require(c.sliding_window == 128 && c.qk_nope_head_dim() == 448, "window / nope");
  require(c.rope_theta == 10000.0 && c.compress_rope_theta == 160000.0, "thetas");
  require(c.rope_factor == 16.0 && c.original_max_position_embeddings == 65536 && c.beta_fast == 32.0 && c.beta_slow == 1.0,
          "yarn");
  // The schedule: two window layers, then ratio 4 on the even layers and
  // ratio 128 on the odd ones, every compressing layer with its own cache.
  require(c.compress_ratios.size() == 46 && c.compress_ratio(0) == 0 && c.compress_ratio(1) == 0 &&
              c.compress_ratio(2) == 4 && c.compress_ratio(3) == 128 && c.compress_ratio(41) == 128 &&
              c.compress_ratio(42) == 4 && c.compress_ratio(43) == 0 && c.compress_ratio(45) == 0,
          "ratios");
  require(c.num_caches() == 41 && c.num_index_caches() == 21, "cache counts");
  require(c.cache_ordinal(0) == -1 && c.cache_ordinal(1) == -1 && c.cache_ordinal(2) == 0 && c.cache_ordinal(3) == 1 &&
              c.cache_ordinal(42) == 40 && c.cache_ordinal(43) == -1,
          "cache ordinals");
  require(c.index_ordinal(2) == 0 && c.index_ordinal(3) == -1 && c.index_ordinal(4) == 1 && c.index_ordinal(42) == 20 &&
              c.index_ordinal(0) == -1,
          "index ordinals");
  require(c.overlapped(2) && c.indexed(2) && !c.overlapped(3) && !c.indexed(3) && !c.indexed(0), "ratio 4 facts");
  require(c.hashed(0) && c.hashed(2) && !c.hashed(3) && c.num_hash_layers == 3, "hash layers");
  require(c.is_draft(43) && c.is_draft(45) && !c.is_draft(42) && c.draft_stage(45) == 2 && c.draft_stage(3) == -1,
          "draft layers");
  require(c.index_topk == 512 && c.index_n_heads == 64 && c.index_head_dim == 128, "indexer");
  require(c.hc_mult == 4 && c.hc_sinkhorn_iters == 20 && c.hc_eps == 1e-6f && c.hc_coeff_rows() == 24, "mhc");
  require(c.moe_intermediate_size == 2048 && c.n_routed_experts == 256 && c.num_experts_per_tok == 6 &&
              c.n_shared_experts == 1 && c.shared_expert_inter() == 2048,
          "moe");
  require(c.routed_scaling_factor == 1.5f && c.norm_topk_prob && c.swiglu_limit == 10.0f &&
              c.scoring_func == "sqrtsoftplus" && c.topk_method == "noaux_tc",
          "router");
  require(c.dspark_block_size == 5 && c.dspark_noise_token_id == 128799 && c.dspark_markov_rank == 256, "dspark");
  require(c.dspark_target_layer_ids.size() == 3 && c.is_dspark_target(40) && c.is_dspark_target(42) && !c.is_dspark_target(39),
          "dspark targets");
  require(c.fp8_block_size == 128 && c.fp4_block_size == 32, "blocks");
  const dgpp::GlmMoeConfig m = c.moe_config(512);
  require(m.inter == 512 && m.n_experts == 256 && m.top_k == 6 && m.n_shared_experts == 1 && m.swiglu_limit == 10.0f,
          "moe_config");
  require(m.router_mode == dgpp::MoeRouterMode::SqrtSoftplusBias && m.routed_scaling_factor == 1.5f, "moe_config router");
}

DGPP_TEST(dsv4_config_refuses_unsupported_shapes) {
  require(has(refusal(config_json("\"model_type\": \"deepseek_v4\"", "\"model_type\": \"deepseek_v41\"")), "model_type"), "type");
  require(has(refusal(config_json("\"num_key_value_heads\": 1", "\"num_key_value_heads\": 2")), "num_key_value_heads"), "kv heads");
  require(has(refusal(config_json("\"head_dim\": 512", "\"head_dim\": 256")), "head_dim"), "head dim");
  require(has(refusal(config_json("\"qk_rope_head_dim\": 64", "\"qk_rope_head_dim\": 32")), "qk_rope_head_dim"), "rope dim");
  require(has(refusal(config_json("\"o_groups\": 8", "\"o_groups\": 7")), "o_groups"), "groups");
  require(has(refusal(config_json("\"type\": \"yarn\"", "\"type\": \"default\"")), "rope_scaling.type"), "yarn");
  require(has(refusal(config_json("\"sliding_window\": 128", "\"sliding_window\": 256")), "sliding_window"), "window");
  require(has(refusal(config_json("\"hc_mult\": 4", "\"hc_mult\": 2")), "hc_mult"), "hc_mult");
  require(has(refusal(config_json("\"scoring_func\": \"sqrtsoftplus\"", "\"scoring_func\": \"softmax\"")), "scoring_func"), "scoring");
  require(has(refusal(config_json("\"topk_method\": \"noaux_tc\"", "\"topk_method\": \"greedy\"")), "topk_method"), "topk method");
  require(has(refusal(config_json("\"n_shared_experts\": 1", "\"n_shared_experts\": 2")), "n_shared_experts"), "shared");
  require(has(refusal(config_json("\"swiglu_limit\": 10.0", "\"swiglu_limit\": 0.0")), "swiglu_limit"), "clamp");
  require(has(refusal(config_json("\"index_topk\": 512", "\"index_topk\": 500")), "index_topk"), "topk pow2");
  require(has(refusal(config_json("\"index_head_dim\": 128", "\"index_head_dim\": 64")), "index_head_dim"), "index dim");
  require(has(refusal(config_json("\"index_n_heads\": 64", "\"index_n_heads\": 48")), "index_n_heads"), "index heads");
  require(has(refusal(config_json("\"num_hash_layers\": 3", "\"num_hash_layers\": 44")), "num_hash_layers"), "hash layers");
  require(has(refusal(config_json("\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true")), "tie_word_embeddings"), "tie");
  require(has(refusal(config_json("\"attention_bias\": false", "\"attention_bias\": true")), "attention_bias"), "bias");
  // The schedule's consistency rules.
  require(config_json().find(kRatios) != std::string::npos, "ratio anchor");
  require(has(refusal(config_json(kRatios, "\"compress_ratios\": [0, 0, 4, 128]")), "compress_ratios"), "ratio length");
  {
    std::string two = kRatios;
    two.replace(two.find("[0, 0, 4, 128"), 13, "[0, 0, 2, 128");
    require(has(refusal(config_json(kRatios, two)), "ratios 0, 4"), "ratio 2 is V4.1's");
    std::string draft = kRatios;
    draft.replace(draft.rfind("0, 0, 0]"), 8, "0, 0, 4]");
    require(has(refusal(config_json(kRatios, draft)), "draft stage"), "draft ratio");
    // Nine draft stages.
    std::string many = kRatios;
    many.replace(many.rfind("0, 0, 0]"), 8, "0, 0, 0, 0, 0, 0, 0, 0, 0]");
    require(has(refusal(config_json(kRatios, many)), "draft stages"), "too many stages");
  }
  require(has(refusal(config_json("\"dspark_block_size\": 5", "\"dspark_block_size\": 0")), "dspark_block_size"),
          "draft stages without a block");
  require(has(refusal(config_json("\"dspark_target_layer_ids\": [40, 41, 42]", "\"dspark_target_layer_ids\": [40, 41, 43]")),
              "dspark_target_layer_ids"), "target outside the backbone");
  require(has(refusal(config_json("\"dspark_target_layer_ids\": [40, 41, 42]", "\"dspark_target_layer_ids\": [41, 40, 42]")),
              "dspark_target_layer_ids"), "unsorted targets");
  require(has(refusal(config_json("\"dspark_markov_rank\": 256", "\"dspark_markov_rank\": 100")), "dspark_markov_rank"), "rank");
  require(has(refusal(config_json("\"dspark_noise_token_id\": 128799", "\"dspark_noise_token_id\": 129280")),
              "dspark_noise_token_id"), "noise id");
  // The quantization contract: the release's own format only.
  require(has(refusal(config_json("\"quant_method\": \"fp8\"", "\"quant_method\": \"modelopt\"")), "quant_method"), "method");
  require(has(refusal(config_json("\"weight_block_size\": [128, 128]", "\"weight_block_size\": [32, 32]")), "weight_block_size"), "block");
  require(has(refusal(config_json("\"scale_fmt\": \"ue8m0\"", "\"scale_fmt\": null")), "scale_fmt"), "scale fmt");
  require(has(refusal(config_json("\"fmt\": \"e4m3\"", "\"fmt\": \"e5m2\"")), "quantization_config.fmt"), "fmt");
  require(has(refusal(config_json("\"expert_dtype\": \"fp4\"", "\"expert_dtype\": \"nvfp4\"")), "expert_dtype"), "nvfp4 cast");
  require(has(refusal(config_json("\"expert_dtype\": \"fp4\",", "")), "expert_dtype"), "fp8 experts");
  require(has(refusal(config_json("\"quant_method\": \"fp8\"", "\"quant_method\": \"fp8\", \"bits\": 4")), "bits"), "int4 repack");
  require(has(refusal(config_json("\"quantization_config\": {", "\"quantization_config\": null, \"x\": {")),
              "quantization_config"), "missing quantization");
}

DGPP_TEST(dsv4_architecture_registry_dispatches) {
  const auto d = dgpp::minijson::parse(R"({"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4"})");
  require(dgpp::detect_architecture(d.root) == dgpp::ModelArchitecture::DeepseekV4, "deepseek_v4");
  require(std::string(dgpp::model_architecture_name(dgpp::ModelArchitecture::DeepseekV4)) == "deepseek_v4", "name");
  const auto t = dgpp::minijson::parse(R"({"model_type": "deepseek_v4"})");
  require(dgpp::detect_architecture(t.root) == dgpp::ModelArchitecture::DeepseekV4, "by model_type");
  // The V4.1 family keeps its own dispatch (the prefix test must not swallow it).
  const auto v41 = dgpp::minijson::parse(R"({"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41"})");
  require(dgpp::detect_architecture(v41.root) == dgpp::ModelArchitecture::DeepseekV41, "deepseek_v41 unchanged");
  bool refused = false;
  try {
    const auto v3 = dgpp::minijson::parse(R"({"architectures": ["DeepseekV3ForCausalLM"], "model_type": "deepseek_v3"})");
    (void)dgpp::detect_architecture(v3.root);
  } catch (const std::runtime_error&) {
    refused = true;
  }
  require(refused, "DeepSeek-V3 is not this family");
}

DGPP_TEST(dsv4_config_reads_the_landed_checkpoint) {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) dgpp::test::skip("HOME is unset; checkpoint cache unavailable");
  const fs::path root = fs::path(home) / ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots";
  if (!fs::is_directory(root))
    dgpp::test::skip("checkpoint snapshot directory absent: " + root.string());
  for (const auto& snap : fs::directory_iterator(root)) {
    const fs::path cfg = snap.path() / "config.json";
    if (!fs::exists(cfg)) continue;
    require(dgpp::detect_architecture_file(cfg.string()) == dgpp::ModelArchitecture::DeepseekV4, "arch");
    const dgpp::Dsv4Config c = dgpp::Dsv4Config::from_json_file(cfg.string());
    require(c.num_hidden_layers == 43 && c.max_layer() == 46 && c.n_routed_experts == 256, "landed values");
    require(c.num_caches() == 41 && c.num_index_caches() == 21 && c.num_hash_layers == 3, "landed schedule");
    return;
  }
  dgpp::test::skip("no eligible checkpoint config under " + root.string());
}
