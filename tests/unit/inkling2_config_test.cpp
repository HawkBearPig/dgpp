// The Inkling 2 config parser: the release's values parse, the visit and
// block geometry derives from them, and the shapes the engine does not
// implement are refused by field name.
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "inkling2_config_json.hpp"
#include "loaders/architecture.hpp"
#include "loaders/minijson.hpp"
#include "models/inkling2/config.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

dgpp::Inkling2Config parse(const std::string& config, const std::string& quant = inkling2_test::quant_json()) {
  const auto c = dgpp::minijson::parse(config);
  const auto q = dgpp::minijson::parse(quant);
  return dgpp::Inkling2Config::parse(c.root, q.root);
}

std::string refusal(const std::string& config, const std::string& quant = inkling2_test::quant_json()) {
  try {
    (void)parse(config, quant);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

bool has(const std::string& s, const char* needle) { return s.find(needle) != std::string::npos; }

}  // namespace

DGPP_TEST(inkling2_config_parses_the_release) {
  const dgpp::Inkling2Config c = parse(inkling2_test::config_json());
  require(c.hidden_size == 2304 && c.num_hidden_layers == 24, "shape");
  require(c.vocab_size == 200058 && c.padded_vocab_size == 201024, "vocabulary");
  require(c.max_position_embeddings == 1048576, "positions");
  require(c.eos_token_ids.size() == 1 && c.eos_token_ids[0] == 200006, "eos");
  require(c.final_logit_softcapping == 30.0f && c.rms_norm_eps == 1e-6f, "softcap / eps");
  require(c.num_attention_heads == 24 && c.num_key_value_heads == 8 && c.head_dim == 128, "heads");
  require(c.d_rel == 16 && c.rel_extent == 1024 && c.sliding_window_size == 1024 && c.sconv_kernel_size == 4, "attention");
  require(c.log_scaling && c.log_scaling_n_floor == 8192 && c.log_scaling_alpha == 0.1f, "log scaling");
  require(c.qk_l2_clip_threshold == 22.627416997969522f && c.rel_l2_clip_threshold == 8.0f, "clips");
  require(c.n_routed_experts == 128 && c.num_experts_per_tok == 8 && c.intermediate_size == 1152, "moe");
  require(c.route_scale == 8.0f && c.mlp_gate_softcap == 4.0f && c.mlp_up_softcap == 25.0f, "moe scalars");
  require(c.block_layers == 6 && c.core_start == 6 && c.core_end == 18 && c.num_passes == 3 && c.gate_rank == 64, "loop");
  // The last layer of every block is the full-attention one.
  for (int l = 0; l < 24; ++l) require(c.is_local_layer(l) == (l % 6 != 5), "local pattern at " + std::to_string(l));
  // Layers 0 to 19 NVFP4, 20 to 23 MXFP8.
  for (int l = 0; l < 24; ++l) require(c.layer_mxfp8(l) == (l >= 20), "expert format at " + std::to_string(l));
  require(c.q_side_rows() == 24 * (2 * 128 + 16 + 1) && c.kv_rows() == 1024 && c.o_proj_cols() == 3072, "projection rows");
}

DGPP_TEST(inkling2_config_derives_the_visits) {
  const dgpp::Inkling2Config c = parse(inkling2_test::config_json());
  require(c.num_visits() == 48 && c.core_layers() == 12, "48 visits from 24 layers");
  require(c.num_full_visits() == 8 && c.num_local_visits() == 40, "visit kinds");
  require(c.max_blocks() == 4 && c.bank_rows() == 2, "blocks");
  // The prelude, three passes of the core, the coda.
  for (int v = 0; v < 6; ++v) require(c.visit(v).layer == v && c.visit(v).pass == 0, "prelude");
  for (int p = 0; p < 3; ++p)
    for (int l = 6; l < 18; ++l) {
      const int v = l + 12 * p;
      require(c.visit(v).layer == l && c.visit(v).pass == p, "core visit " + std::to_string(v));
      require(c.visit_of(l, p) == v, "visit_of");
    }
  for (int l = 18; l < 24; ++l) {
    require(c.visit(l + 24).layer == l && c.visit(l + 24).pass == 0, "coda");
    require(c.visit_of(l, 0) == l + 24, "coda visit_of");
  }
  // The draft model reads visits 5, 23 and 41 and the final-norm output.
  require(c.draft.present && c.draft.block_size == 8 && c.draft.drafts() == 7, "draft block");
  require(c.draft.target_layer_ids == std::vector<int>({5, 23, 41, 48}), "draft taps");
  require(c.draft.num_hidden_layers == 3 && c.draft.intermediate_size == 9216 && c.draft.markov_rank == 256, "draft shape");
  require(c.draft.local_layer[0] == 1 && c.draft.local_layer[1] == 0 && c.draft.local_layer[2] == 1, "draft windows");
  require(c.vision.present && c.vision.hidden_size == 1024 && c.vision.num_hidden_layers == 23 &&
              c.vision.native_patch_size == 20 && c.vision.adapter_merge_size == 2 && c.vision.posemb_grid_size == 32,
          "vision");
}

DGPP_TEST(inkling2_config_refuses_unsupported_shapes_by_name) {
  using inkling2_test::config_json;
  require(has(refusal(config_json("\"model_type\": \"inkling2_loop\"", "\"model_type\": \"inkling2\"")), "model_type"),
          "model type");
  require(has(refusal(config_json("\"head_dim\": 128,\n    \"d_rel\"", "\"head_dim\": 64,\n    \"d_rel\"")), "head_dim"),
          "head dim");
  require(has(refusal(config_json("\"d_rel\": 16", "\"d_rel\": 8")), "d_rel"), "d_rel");
  require(has(refusal(config_json("\"sconv_kernel_size\": 4", "\"sconv_kernel_size\": 3")), "sconv_kernel_size"), "taps");
  require(has(refusal(config_json("\"use_value_norm_no_gamma\": true", "\"use_value_norm_no_gamma\": false")),
              "use_value_norm_no_gamma"),
          "value norm");
  require(has(refusal(config_json("\"attn_res_normalize_values\": true", "\"attn_res_normalize_values\": false")),
              "attn_res_normalize_values"),
          "normalized mixes");
  require(has(refusal(config_json("\"n_shared_experts\": 0", "\"n_shared_experts\": 1")), "n_shared_experts"), "shared");
  require(has(refusal(config_json("\"use_gate_bias\": true", "\"use_gate_bias\": false")), "use_gate_bias"), "gate bias");
  require(has(refusal(config_json("\"is_varloop\": false", "\"is_varloop\": true")), "is_varloop"), "varloop");
  require(has(refusal(config_json("\"core_start\": 6", "\"core_start\": 5")), "core_start"), "core off a block boundary");
  require(has(refusal(config_json("\"num_passes\": 3", "\"num_passes\": 1")), "num_passes"), "one pass");
  require(has(refusal(config_json("\"swa_head_dim\": 128", "\"swa_head_dim\": 64")), "swa_head_dim"), "local geometry");
  require(has(refusal(config_json("\"markov_head_type\": \"rnn_hidden\"", "\"markov_head_type\": \"vanilla\"")),
              "markov_head_type"),
          "markov head");
  require(has(refusal(config_json("      48\n", "      47\n")), "target_layer_ids"), "the last tap is the visit count");
}

DGPP_TEST(inkling2_config_reads_the_precision_metadata) {
  using inkling2_test::quant_json;
  const std::string config = inkling2_test::config_json();
  // Every routed expert tensor must be named by a quantized group.
  require(has(refusal(config, quant_json(23, 20)), "BF16 routed experts"), "a layer without a group");
  std::string q = quant_json();
  auto swap = [&](const std::string& from, const std::string& to) {
    const size_t at = q.find(from);
    require(at != std::string::npos, "quant anchor " + from);
    std::string r = q;
    r.replace(at, from.size(), to);
    return r;
  };
  require(has(refusal(config, swap("\"weight_block_shape\": [1, 16]", "\"weight_block_shape\": [1, 32]")),
              "weight_block_shape"),
          "nvfp4 block");
  require(has(refusal(config, swap("fouroversix_mae", "static_6")), "scale_selection_rule"), "scale rule");
  require(has(refusal(config, swap("\"compute_dtype\": \"mxfp8\"", "\"compute_dtype\": \"mxfp4\"")), "compute_dtype"),
          "unknown format");
  require(has(refusal(config, swap("\"quant_method\": \"inkling2\"", "\"quant_method\": \"modelopt\"")), "quant_method"),
          "quant method");
  // A layer whose two projections disagree.
  require(has(refusal(config, swap("\"model.llm.layers.19.mlp.experts.w2_weight\"],",
                                   "\"model.llm.layers.19.mlp.experts.w13_weight\"],")),
              "listed twice"),
          "a repeated tensor");
}

DGPP_TEST(inkling2_architecture_is_detected) {
  const std::string text = inkling2_test::config_json();
  const auto d = dgpp::minijson::parse(text);
  require(dgpp::detect_architecture(d.root) == dgpp::ModelArchitecture::Inkling2Loop, "by class name");
  require(std::string(dgpp::model_architecture_name(dgpp::ModelArchitecture::Inkling2Loop)) == "inkling2_loop", "name");
  const auto t = dgpp::minijson::parse("{\"model_type\": \"inkling2_loop\"}");
  require(dgpp::detect_architecture(t.root) == dgpp::ModelArchitecture::Inkling2Loop, "by model_type");
}

DGPP_TEST(inkling2_config_reads_the_landed_checkpoint) {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) dgpp::test::skip("HOME is unset; checkpoint cache unavailable");
  const fs::path root = fs::path(home) / ".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots";
  if (!fs::is_directory(root)) dgpp::test::skip("checkpoint snapshot directory absent: " + root.string());
  for (const auto& snap : fs::directory_iterator(root)) {
    if (!fs::exists(snap.path() / "config.json") || !fs::exists(snap.path() / "hf_quant_config.json")) continue;
    const dgpp::Inkling2Config c = dgpp::Inkling2Config::from_checkpoint_dir(snap.path().string());
    require(c.num_hidden_layers == 24 && c.num_visits() == 48 && c.n_routed_experts == 128, "landed values");
    for (int l = 0; l < 24; ++l) require(c.layer_mxfp8(l) == (l >= 20), "landed expert formats");
    require(dgpp::detect_architecture_file((snap.path() / "config.json").string()) ==
                dgpp::ModelArchitecture::Inkling2Loop,
            "arch");
    return;
  }
  dgpp::test::skip("no eligible checkpoint config under " + root.string());
}
