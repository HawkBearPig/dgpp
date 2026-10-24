// The DeepSeek-V4.1-Flash config parser: the release's values parse, the
// CSA2 schedule (modes, kv / index sources, the candidate pool), the Engram
// and DSpark facts and the MoE configuration derive from the file, the
// architecture registry dispatches on it, and the unsupported shapes are
// refused by name.
#include <filesystem>
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "dsv41_config_json.hpp"
#include "loaders/architecture.hpp"
#include "loaders/minijson.hpp"
#include "models/dsv41/config.hpp"

namespace {

using dsv41_test::config_json;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

dgpp::Dsv41TextConfig parse(const std::string& text) {
  const auto t = dgpp::minijson::parse(text);
  return dgpp::Dsv41TextConfig::parse(t.root);
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

}  // namespace

DGPP_TEST(dsv41_config_parses_the_release) {
  using dgpp::Dsv41LayerMode;
  const dgpp::Dsv41TextConfig c = parse(config_json());
  require(c.hidden_size == 5120 && c.vocab_size == 129280 && c.num_hidden_layers == 40, "shape");
  require(c.num_nextn_predict_layers == 3 && c.max_layer() == 43, "draft stages");
  require(c.rms_norm_eps == 1e-20f && c.max_position_embeddings == 1048576, "eps / positions");
  require(c.bos_token_id == 0 && c.eos_token_id == 1 && c.pad_token_id == 2 && c.image_token_id == 129264, "tokens");
  require(c.num_attention_heads == 64 && c.num_key_value_heads == 1 && c.head_dim == 512 && c.qk_rope_head_dim == 64,
          "attention dims");
  require(c.q_lora_rank == 1280 && c.o_lora_rank == 1024 && c.o_groups == 8 && c.heads_per_group() == 8, "lora ranks");
  require(c.sliding_window == 128 && c.qk_nope_head_dim() == 448, "window / nope");
  require(c.rope_theta == 10000.0 && c.compress_rope_theta == 160000.0, "thetas");
  require(c.rope_factor == 16.0 && c.original_max_position_embeddings == 65536 && c.beta_fast == 32.0 && c.beta_slow == 1.0,
          "yarn");
  // The CSA2 schedule.
  require(c.compress_ratios.size() == 43 && c.compress_ratio(0) == 0 && c.compress_ratio(2) == 2 &&
              c.compress_ratio(19) == 2 && c.compress_ratio(20) == 1 && c.compress_ratio(39) == 1 && c.compress_ratio(42) == 0,
          "ratios");
  require(c.num_kv_sources() == 4 && c.num_index_sources() == 8, "source counts");
  require(c.layer_mode(0) == Dsv41LayerMode::Window && c.layer_mode(1) == Dsv41LayerMode::Window, "window layers");
  require(c.layer_mode(2) == Dsv41LayerMode::Full && c.layer_mode(8) == Dsv41LayerMode::Full &&
              c.layer_mode(14) == Dsv41LayerMode::Full && c.layer_mode(20) == Dsv41LayerMode::Full,
          "full layers");
  require(c.layer_mode(24) == Dsv41LayerMode::Reindex && c.layer_mode(36) == Dsv41LayerMode::Reindex, "reindex layers");
  require(c.layer_mode(3) == Dsv41LayerMode::Reuse && c.layer_mode(21) == Dsv41LayerMode::Reuse &&
              c.layer_mode(39) == Dsv41LayerMode::Reuse,
          "reuse layers");
  require(c.layer_mode(40) == Dsv41LayerMode::Window && c.is_draft(40) && c.draft_stage(42) == 2 && !c.is_draft(39),
          "draft modes");
  require(c.kv_source_of(0) == -1 && c.kv_source_of(2) == 2 && c.kv_source_of(7) == 2 && c.kv_source_of(8) == 8 &&
              c.kv_source_of(19) == 14 && c.kv_source_of(20) == 20 && c.kv_source_of(39) == 20 && c.kv_source_of(40) == -1,
          "kv sources");
  require(c.index_source_of(7) == 2 && c.index_source_of(23) == 20 && c.index_source_of(24) == 24 &&
              c.index_source_of(27) == 24 && c.index_source_of(39) == 36,
          "index sources");
  require(c.is_candidate_source(20) && !c.uses_candidates(20) && !c.uses_candidates(21) && c.uses_candidates(24) &&
              c.uses_candidates(36) && !c.uses_candidates(14),
          "candidate pool");
  require(c.candidate_pool_entries() == 16384 && c.index_topk == 512 && c.index_n_heads == 32 && c.index_head_dim == 128,
          "indexer");
  require(c.hc_mult == 4 && c.hc_sinkhorn_iters == 20 && c.hc_eps == 1e-6f && c.hc_coeff_rows() == 24, "mhc");
  require(c.moe_intermediate_size == 2304 && c.n_routed_experts == 384 && c.num_experts_per_tok == 6 &&
              c.n_shared_experts == 1 && c.shared_expert_inter() == 2304,
          "moe");
  require(c.routed_scaling_factor == 1.5f && c.norm_topk_prob && c.swiglu_limit == 10.0f &&
              c.scoring_func == "sqrtsoftplus" && c.topk_method == "noaux_tc",
          "router");
  require(c.engram_layer_ids.size() == 2 && c.has_engram(1) && c.has_engram(14) && !c.has_engram(2) &&
              c.engram_index(14) == 1 && c.engram_index(3) == -1,
          "engram layers");
  require(c.engram_num_embeddings[0] == 384006168 && c.engram_num_embeddings[1] == 384016682, "engram rows");
  require(c.engram_max_ngram_size == 4 && c.engram_n_heads == 8 && c.engram_head_dim == 256 &&
              c.engram_rows_per_layer() == 24 && c.engram_width() == 6144,
          "engram geometry");
  require(c.engram_vocab_size == 16000000 && c.engram_pad_token_id == 2 && c.engram_compressed_vocab_size == 99092,
          "engram hash");
  require(c.dspark_block_size == 5 && c.dspark_noise_token_id == 128799 && c.dspark_markov_rank == 256, "dspark");
  require(c.dspark_target_layer_ids.size() == 3 && c.is_dspark_target(37) && c.is_dspark_target(39) && !c.is_dspark_target(36),
          "dspark targets");
  require(c.dspark_n_routed_experts == 128 && c.dspark_num_experts_per_tok == 3, "dspark experts");
  require(c.fp8_block_size == 32 && c.fp4_block_size == 32, "blocks");
  require(c.vision_present && c.vision.num_hidden_layers == 32 && c.vision.hidden_size == 1024 &&
              c.vision.intermediate_size == 2816 && c.vision.patch_size == 14 && c.vision.downsample_ratio == 3,
          "vision shape");
  const dgpp::GlmMoeConfig m = c.moe_config(576, false);
  require(m.inter == 576 && m.n_experts == 384 && m.top_k == 6 && m.n_shared_experts == 1 && m.swiglu_limit == 10.0f,
          "moe_config");
  require(m.router_mode == dgpp::MoeRouterMode::SqrtSoftplusBias && m.routed_scaling_factor == 1.5f, "moe_config router");
  const dgpp::GlmMoeConfig d = c.moe_config(576, true);
  require(d.n_experts == 128 && d.top_k == 3, "draft moe_config");
}

DGPP_TEST(dsv41_config_refuses_unsupported_shapes) {
  require(has(refusal(config_json("\"model_type\": \"deepseek_v41\"", "\"model_type\": \"deepseek_v4\"")), "model_type"), "type");
  require(has(refusal(config_json("\"model_type\": \"deepseek_v41_text\"", "\"model_type\": \"deepseek_v3\"")),
              "text_config.model_type"), "text type");
  require(has(refusal(config_json("\"num_key_value_heads\": 1", "\"num_key_value_heads\": 2")), "num_key_value_heads"), "kv heads");
  require(has(refusal(config_json("\"head_dim\": 512", "\"head_dim\": 256")), "head_dim"), "head dim");
  require(has(refusal(config_json("\"qk_rope_head_dim\": 64", "\"qk_rope_head_dim\": 32")), "qk_rope_head_dim"), "rope dim");
  require(has(refusal(config_json("\"o_groups\": 8", "\"o_groups\": 7")), "o_groups"), "groups");
  require(has(refusal(config_json("\"rope_type\": \"yarn\"", "\"rope_type\": \"default\"")), "rope_type"), "yarn");
  require(has(refusal(config_json("\"hc_mult\": 4", "\"hc_mult\": 2")), "hc_mult"), "hc_mult");
  require(has(refusal(config_json("\"scoring_func\": \"sqrtsoftplus\"", "\"scoring_func\": \"softmax\"")), "scoring_func"), "scoring");
  require(has(refusal(config_json("\"topk_method\": \"noaux_tc\"", "\"topk_method\": \"greedy\"")), "topk_method"), "topk method");
  require(has(refusal(config_json("\"n_shared_experts\": 1", "\"n_shared_experts\": 2")), "n_shared_experts"), "shared");
  require(has(refusal(config_json("\"swiglu_limit\": 10.0", "\"swiglu_limit\": 0.0")), "swiglu_limit"), "clamp");
  require(has(refusal(config_json("\"index_topk\": 512", "\"index_topk\": 500")), "index_topk"), "topk pow2");
  require(has(refusal(config_json("\"index_head_dim\": 128", "\"index_head_dim\": 64")), "index_head_dim"), "index dim");
  require(has(refusal(config_json("\"engram_head_dim\": 256", "\"engram_head_dim\": 128")), "engram_head_dim"), "engram dim");
  require(has(refusal(config_json("\"engram_num_embeddings\": [384006168, 384016682]", "\"engram_num_embeddings\": [384006168]")),
              "engram_num_embeddings"), "engram sizes");
  require(has(refusal(config_json("\"engram_layer_ids\": [1, 14]", "\"engram_layer_ids\": [1, 40]")), "engram_layer_ids"), "engram layer");
  require(has(refusal(config_json("\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true")), "tie_word_embeddings"), "tie");
  // The schedule's consistency rules.
  {
    std::string s = config_json();
    const std::string anchor = "\"compress_ratios\": [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]";
    require(s.find(anchor) != std::string::npos, "ratio anchor");
    require(has(refusal(config_json(anchor, "\"compress_ratios\": [0, 0, 2, 2]")), "compress_ratios"), "ratio length");
    require(has(refusal(config_json(anchor, "\"compress_ratios\": [0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]")),
                "compress_ratios"), "ratio 4");
    require(has(refusal(config_json(anchor, "\"compress_ratios\": [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]")),
                "draft stage"), "draft ratio");
    // Layer 19 at ratio 1 would read layer 14's ratio-2 cache.
    require(has(refusal(config_json(anchor, "\"compress_ratios\": [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]")),
                "ratio"), "ratio mismatch");
  }
  require(has(refusal(config_json("\"kv_source_layer_ids\": [2, 8, 14, 20]", "\"kv_source_layer_ids\": [2, 8, 15, 20]")),
              "kv_source_layer_ids"), "kv source not an index source");
  require(has(refusal(config_json("\"index_source_layer_ids\": [2, 8, 14, 20, 24, 28, 32, 36]",
                                  "\"index_source_layer_ids\": [2, 8, 14, 20, 24, 28, 32, 1]")),
              "index_source_layer_ids"), "unsorted / window index source");
  require(has(refusal(config_json("\"candidate_source_layer_id\": 20", "\"candidate_source_layer_id\": 21")),
              "candidate_source_layer_id"), "candidate source");
  require(has(refusal(config_json("\"candidate_topk_blocks\": 2048", "\"candidate_topk_blocks\": 32")),
              "candidate_topk_blocks"), "pool too small");
  // The quantization contract: the release's own format only.
  require(has(refusal(config_json("\"quant_method\": \"fp8\"", "\"quant_method\": \"modelopt\"")), "quant_method"), "method");
  require(has(refusal(config_json("\"weight_block_size\": [32, 32]", "\"weight_block_size\": [128, 128]")), "weight_block_size"), "block");
  require(has(refusal(config_json("\"scale_fmt\": \"ue8m0\"", "\"scale_fmt\": null")), "scale_fmt"), "scale fmt");
  require(has(refusal(config_json("\"expert_dtype\": \"fp4\"", "\"expert_dtype\": \"nvfp4\"")), "expert_dtype"), "nvfp4 cast");
  require(has(refusal(config_json("\"expert_dtype\": \"fp4\"", "\"expert_dtype\": \"fp4\", \"engram_dtype\": \"fp4\"")),
              "engram_dtype"), "engram re-pack");
  require(has(refusal(config_json("\"expert_dtype\": \"fp4\"", "\"expert_dtype\": \"fp4\", \"moe_quant_algo\": \"NVFP4\"")),
              "moe_quant_algo"), "modelopt cast");
  require(has(refusal(config_json("\"quantization_config\": {", "\"quantization_config\": null, \"x\": {")),
              "quantization_config"), "missing quantization");
}

DGPP_TEST(dsv41_architecture_registry_dispatches) {
  const auto d = dgpp::minijson::parse(R"({"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41"})");
  require(dgpp::detect_architecture(d.root) == dgpp::ModelArchitecture::DeepseekV41, "deepseek_v41");
  require(std::string(dgpp::model_architecture_name(dgpp::ModelArchitecture::DeepseekV41)) == "deepseek_v41", "name");
  const auto t = dgpp::minijson::parse(R"({"model_type": "deepseek_v41"})");
  require(dgpp::detect_architecture(t.root) == dgpp::ModelArchitecture::DeepseekV41, "by model_type");
  const auto g = dgpp::minijson::parse(R"({"architectures": ["GlmMoeDsaForCausalLM"], "model_type": "glm_moe_dsa"})");
  require(dgpp::detect_architecture(g.root) == dgpp::ModelArchitecture::GlmMoeDsa, "glm_moe_dsa unchanged");
  bool refused = false;
  try {
    const auto v3 = dgpp::minijson::parse(R"({"architectures": ["DeepseekV3ForCausalLM"], "model_type": "deepseek_v3"})");
    (void)dgpp::detect_architecture(v3.root);
  } catch (const std::runtime_error&) {
    refused = true;
  }
  require(refused, "DeepSeek-V3 is not this family");
}

DGPP_TEST(dsv41_config_reads_the_landed_checkpoint) {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) dgpp::test::skip("HOME is unset; checkpoint cache unavailable");
  // The 1017 refinement (2026-10-18) ships the same config as the original
  // release; whichever snapshot the cache holds parses to the same values.
  for (const char* repo : {"models--deepseek-ai--DeepSeek-V4.1-Flash-1017", "models--deepseek-ai--DeepSeek-V4.1-Flash"}) {
  const fs::path root = fs::path(home) / ".cache/huggingface/hub" / repo / "snapshots";
  if (!fs::is_directory(root)) continue;
  for (const auto& snap : fs::directory_iterator(root)) {
    const fs::path cfg = snap.path() / "config.json";
    if (!fs::exists(cfg)) continue;
    require(dgpp::detect_architecture_file(cfg.string()) == dgpp::ModelArchitecture::DeepseekV41, "arch");
    const dgpp::Dsv41TextConfig c = dgpp::Dsv41TextConfig::from_json_file(cfg.string());
    require(c.num_hidden_layers == 40 && c.max_layer() == 43 && c.n_routed_experts == 384, "landed values");
    require(c.num_kv_sources() == 4 && c.num_index_sources() == 8 && c.candidate_source_layer_id == 20, "landed schedule");
    return;
  }
  }
  dgpp::test::skip("no eligible DeepSeek-V4.1-Flash or DeepSeek-V4.1-Flash-1017 checkpoint config in cache");
}

DGPP_TEST(dsv41_config_applies_the_deployment_index_budget) {
  // engine.index_topk / candidate_topk_blocks / candidate_block_size
  // (2026-10-18): each value > 0 replaces the checkpoint's, 0 keeps it, and
  // the result passes the parser's own checks.
  dgpp::Dsv41TextConfig c = parse(config_json());
  c.apply_index_budget(0, 0, 0);
  require(c.index_topk == 512 && c.candidate_topk_blocks == 2048 && c.candidate_block_size == 8, "zeros keep the checkpoint's");
  c.apply_index_budget(2048, 0, 0);
  require(c.index_topk == 2048 && c.candidate_pool_entries() == 16384, "the widened budget");
  c.apply_index_budget(0, 1024, 16);
  require(c.index_topk == 2048 && c.candidate_topk_blocks == 1024 && c.candidate_block_size == 16, "the pool's shape");
  const auto refused = [&](int k, int blocks, int bs) {
    dgpp::Dsv41TextConfig d = parse(config_json());
    try {
      d.apply_index_budget(k, blocks, bs);
    } catch (const std::exception& e) {
      return std::string(e.what());
    }
    return std::string();
  };
  require(has(refused(4096, 0, 0), "engine.index_topk"), "past the select networks");
  require(has(refused(3, 0, 0), "engine.index_topk"), "not a power of two");
  require(has(refused(2048, 128, 0), "engine.candidate_topk_blocks"), "a pool of 128 x 8 cannot hold 2048");
  require(has(refused(0, 3, 0), "engine.candidate_topk_blocks"), "blocks not a power of two");
  require(has(refused(2048, 256, 4), "engine.candidate_topk_blocks"), "256 x 4 = 1024 < 2048");
  require(refused(2048, 256, 8).empty(), "256 x 8 = 2048 holds exactly 2048");
  // A checkpoint without a candidate stage takes no pool shape.
  const std::string no_candidates =
      config_json("\"candidate_source_layer_id\": 20,", "\"candidate_source_layer_id\": -1,");
  dgpp::Dsv41TextConfig n = parse(no_candidates);
  require(n.candidate_source_layer_id < 0, "no candidate stage parses");
  n.apply_index_budget(2048, 0, 0);
  require(n.index_topk == 2048, "the budget alone applies without a candidate stage");
  bool rejected = false;
  try {
    n.apply_index_budget(0, 2048, 0);
  } catch (const std::exception& e) {
    rejected = has(e.what(), "engine.candidate_topk_blocks");
  }
  require(rejected, "a pool shape without a candidate stage is refused");
}
