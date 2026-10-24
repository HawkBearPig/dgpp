// The MiMo-V2.6-Flash config parser: the real file's values parse, the
// derived geometry (the hybrid pattern, the rotary slice, the fused
// projection's chunk rows and per-chunk scale rows), and the unsupported
// shapes are refused by name.
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "mimo_config_json.hpp"
#include "models/mimo/config.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

dgpp::MimoTextConfig parse(const std::string& text) {
  const auto t = dgpp::minijson::parse(text);
  return dgpp::MimoTextConfig::parse(t.root);
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

DGPP_TEST(mimo_config_parses_the_release) {
  const dgpp::MimoTextConfig c = parse(mimo_test::config_json());
  require(c.hidden_size == 4096 && c.vocab_size == 152576 && c.num_hidden_layers == 48, "shape");
  require(c.num_swa_layers() == 39 && c.num_moe_layers() == 47, "hybrid pattern / moe layers");
  require(!c.is_swa_layer(0) && c.is_swa_layer(1) && !c.is_swa_layer(5) && !c.is_swa_layer(47) && c.is_swa_layer(46),
          "is_swa_layer");
  require(!c.is_moe_layer(0) && c.is_moe_layer(1) && c.is_moe_layer(47), "is_moe_layer");
  require(c.num_attention_heads == 64 && c.num_key_value_heads == 4 && c.swa_num_key_value_heads == 8, "heads");
  require(c.head_dim == 192 && c.v_head_dim == 128 && c.rotary_dim == 64, "head dims / rotary");
  require(c.rope_theta == 1e7 && c.swa_rope_theta == 1e4, "rope thetas");
  require(c.sliding_window == 128 && c.swa_sink && !c.full_sink, "window / sinks");
  require(c.attention_value_scale == 0.707f, "value scale");
  require(!c.attention_bias, "no attention bias");
  require(c.kv_heads_of(0) == 4 && c.kv_heads_of(1) == 8 && c.sink_of(1) && !c.sink_of(0), "per-layer kinds");
  require(c.rope_theta_of(0) == 1e7 && c.rope_theta_of(1) == 1e4, "per-layer theta");
  require(c.qkv_chunks() == 4, "chunks");
  require(c.qkv_chunk_rows(0) == 3392 && c.qkv_rows(0) == 13568 && c.qkv_scale_rows(0) == 108, "global qkv rows");
  require(c.qkv_chunk_rows(1) == 3712 && c.qkv_rows(1) == 14848 && c.qkv_scale_rows(1) == 116, "swa qkv rows");
  require(c.o_proj_cols() == 8192, "o_proj cols");
  require(c.intermediate_size == 16384 && c.moe_intermediate_size == 2048, "mlp");
  require(c.n_routed_experts == 256 && c.num_experts_per_tok == 8, "moe");
  require(c.routed_scaling_factor == 1.0f && c.norm_topk_prob, "router");
  require(c.num_nextn_predict_layers == 3 && c.mtp_layers_loaded == 1 && c.mtp_layer() == 48, "mtp layer");
  require(c.is_swa_layer(48) && !c.is_moe_layer(48) && c.kv_heads_of(48) == 8 && c.sink_of(48), "the draft is SWA + dense");
  require(c.qkv_chunk_rows(48) == 3712 && c.qkv_scale_rows(48) == 116, "draft qkv rows");
  require(c.eos_token_ids.size() == 1 && c.eos_token_ids[0] == 151645, "eos");
  require(c.pad_token_id == 151643, "pad");
  require(c.rms_norm_eps == 1e-6f, "eps");
  require(c.fp8_block == 128 && c.mxfp4_block == 32, "quantization");
  const dgpp::GlmMoeConfig m = c.moe_config(512);
  require(m.inter == 512 && m.n_experts == 256 && m.top_k == 8 && m.n_shared_experts == 0, "moe_config");
  require(m.routed_scaling_factor == 1.0f && m.norm_topk_prob, "moe_config router");
  require(m.router_mode == dgpp::MoeRouterMode::SigmoidBias && m.swiglu_limit > 1e30f, "moe_config mode");
}

DGPP_TEST(mimo_config_refuses_unsupported_shapes) {
  auto has = [](const std::string& msg, const char* needle) { return msg.find(needle) != std::string::npos; };
  require(has(refusal(mimo_test::config_json("\"model_type\": \"mimo_v2\"", "\"model_type\": \"mimo_v3\"")), "model_type"),
          "model_type");
  require(has(refusal(mimo_test::config_json("\"head_dim\": 192", "\"head_dim\": 128")), "head_dim"), "head_dim");
  require(has(refusal(mimo_test::config_json("\"attention_chunk_size\": 128", "\"attention_chunk_size\": 64")),
              "attention_chunk_size"),
          "attention_chunk_size");
  require(has(refusal(mimo_test::config_json("\"store_dtype\": \"mxfp4\"", "\"store_dtype\": \"fp8\"")), "store_dtype"),
          "store_dtype");
  require(has(refusal(mimo_test::config_json("\"n_shared_experts\": null", "\"n_shared_experts\": 1")), "n_shared_experts"),
          "shared experts");
  require(has(refusal(mimo_test::config_json("\"n_group\": 1", "\"n_group\": 2")), "n_group"), "n_group");
  require(has(refusal(mimo_test::config_json("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\"")), "hidden_act"), "act");
  require(has(refusal(mimo_test::config_json("\"attention_projection_layout\": \"fused_qkv\"",
                                             "\"attention_projection_layout\": \"split\"")),
              "attention_projection_layout"),
          "projection layout");
  require(has(refusal(mimo_test::config_json("\"quant_method\": \"fp8\"", "\"quant_method\": \"modelopt\"")), "quant_method"),
          "quant method");
  require(has(refusal(mimo_test::config_json("\"weight_block_size\": [128, 128]", "\"weight_block_size\": [32, 32]")),
              "weight_block_size"),
          "block size");
  // An ignore list missing one layer's o_proj, or naming a module the
  // loader does not know, is refused.
  require(has(refusal(mimo_test::config_json("", "", mimo_test::ignore_json(47, true))), "ignored_layers"), "ignore coverage");
  require(has(refusal(mimo_test::config_json("", "", mimo_test::ignore_json() + ", \"model.layers.5.mlp.gate\"")),
              "unexpected ignored"),
          "ignore extra");
  // A window that is not a tile multiple, a pattern of the wrong length.
  require(has(refusal(mimo_test::config_json("\"sliding_window\": 128,", "\"sliding_window\": 100,")), "sliding_window"),
          "window");
  require(has(refusal(mimo_test::config_json("\"num_hidden_layers\": 48", "\"num_hidden_layers\": 47")), "hybrid_layer_pattern"),
          "pattern length");
  // Without draft layers there is no draft.
  const dgpp::MimoTextConfig no_mtp =
      parse(mimo_test::config_json("\"num_nextn_predict_layers\": 3", "\"num_nextn_predict_layers\": 0"));
  require(no_mtp.mtp_layer() < 0 && no_mtp.mtp_layers_loaded == 0, "no mtp");
}

DGPP_TEST(mimo_config_reads_the_landed_checkpoint) {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) dgpp::test::skip("HOME is unset; checkpoint cache unavailable");
  const fs::path root = fs::path(home) / ".cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-RL/snapshots";
  if (!fs::is_directory(root))
    dgpp::test::skip("checkpoint snapshot directory absent: " + root.string());
  for (const auto& snap : fs::directory_iterator(root)) {
    const fs::path cfg = snap.path() / "config.json";
    if (!fs::exists(cfg)) continue;
    const dgpp::MimoTextConfig c = dgpp::MimoTextConfig::from_json_file(cfg.string());
    require(c.num_hidden_layers == 48 && c.mtp_layer() == 48 && c.n_routed_experts == 256, "landed values");
    require(c.num_swa_layers() == 39, "landed pattern");
    return;
  }
  dgpp::test::skip("no eligible checkpoint config under " + root.string());
}

DGPP_TEST(mimo_native_mtp_config_preserves_no_draft_and_rejects_incomplete_heads) {
  struct RestoreEnv {
    bool present = std::getenv("DGPP_MIMO_NATIVE_MTP") != nullptr;
    std::string value = present ? std::getenv("DGPP_MIMO_NATIVE_MTP") : "";
    ~RestoreEnv() {
      if (present)
        setenv("DGPP_MIMO_NATIVE_MTP", value.c_str(), 1);
      else
        unsetenv("DGPP_MIMO_NATIVE_MTP");
    }
  } restore;
  setenv("DGPP_MIMO_NATIVE_MTP", "1", 1);
  require(parse(mimo_test::config_json()).mtp_layers_loaded == 3, "load native heads");
  require(parse(mimo_test::config_json("\"num_nextn_predict_layers\": 3",
                                       "\"num_nextn_predict_layers\": 0"))
                  .mtp_layers_loaded == 0,
          "native opt-in must preserve no-draft models");
  for (int count : {1, 2}) {
    const auto message =
        refusal(mimo_test::config_json("\"num_nextn_predict_layers\": 3",
                                       "\"num_nextn_predict_layers\": " + std::to_string(count)));
    require(message.find("num_nextn_predict_layers") != std::string::npos,
            "name invalid native head count");
  }
  unsetenv("DGPP_MIMO_NATIVE_MTP");
  require(parse(mimo_test::config_json()).mtp_layers_loaded == 1, "default remains recursive");
}
