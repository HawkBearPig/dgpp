#pragma once
// The config.json of thinkingmachines/Inkling-2-Mini (revision b85c3cfa) as
// shipped, and the expert precision groups of its hf_quant_config.json: the
// unit tests parse the release's own values.
#include <stdexcept>
#include <string>

namespace inkling2_test {

inline const char* kConfig = R"JSON({
  "dtype": "bfloat16",
  "architectures": [
    "Inkling2LoopForConditionalGeneration"
  ],
  "model_type": "inkling2_loop",
  "eos_token_id": 200006,
  "text_config": {
    "model_type": "inkling2_loop_text",
    "model_max_length": 1048576,
    "torch_dtype": "bfloat16",
    "hidden_size": 2304,
    "num_hidden_layers": 24,
    "vocab_size": 201024,
    "num_attention_heads": 24,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "d_rel": 16,
    "rel_extent": 1024,
    "q_bias": false,
    "o_bias": false,
    "log_scaling_n_floor": 8192,
    "log_scaling_alpha": 0.1,
    "rms_norm_eps": 1e-06,
    "use_value_norm_no_gamma": true,
    "qk_l2_clip_threshold": 22.627416997969522,
    "rel_l2_clip_threshold": 8.0,
    "local_layer_ids": [
      0,
      1,
      2,
      3,
      4,
      6,
      7,
      8,
      9,
      10,
      12,
      13,
      14,
      15,
      16,
      18,
      19,
      20,
      21,
      22
    ],
    "sconv_kernel_size": 4,
    "unpadded_vocab_size": 200058,
    "final_logit_softcapping": 30.0,
    "swa_head_dim": 128,
    "swa_num_attention_heads": 24,
    "swa_num_key_value_heads": 8,
    "sliding_window_size": 1024,
    "n_layers_per_attn_res_block": 6,
    "attn_res_normalize_values": true,
    "n_routed_experts": 128,
    "num_experts_per_tok": 8,
    "n_shared_experts": 0,
    "intermediate_size": 1152,
    "mlp_gate_softcap": 4.0,
    "mlp_up_softcap": 25.0,
    "route_scale": 8.0,
    "use_gate_bias": true,
    "loop_config": {
      "core_start": 6,
      "core_end": 18,
      "num_passes": 3,
      "gate_rank": 64,
      "is_varloop": false
    }
  },
  "vision_config": {
    "model_type": "inkling2_vision",
    "text_hidden_size": 2304,
    "temporal_patch_size": 2,
    "native_patch_size": 20,
    "adapter_merge_size": 2,
    "projection_hidden_dim": 4096,
    "projector_bias": false,
    "hidden_size": 1024,
    "num_hidden_layers": 23,
    "num_attention_heads": 16,
    "mlp_ratio": 4.0,
    "posemb_grid_size": 32,
    "layer_norm_eps": 1e-05
  },
  "mtp_config": {
    "hidden_size": 2304,
    "num_hidden_layers": 3,
    "num_attention_heads": 24,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 9216,
    "max_position_embeddings": 1048576,
    "torch_dtype": "bfloat16",
    "q_bias": false,
    "o_bias": false,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "local_layer_ids": [
      0,
      2
    ],
    "sliding_window_size": 1024,
    "block_size": 8,
    "markov_rank": 256,
    "markov_head_type": "rnn_hidden",
    "target_layer_ids": [
      5,
      23,
      41,
      48
    ]
  },
  "quantization_config": {
    "quant_method": "modelopt",
    "quant_algo": "NVFP4"
  }
})JSON";

// hf_quant_config.json reduced to what the engine reads: the precision
// group of every routed expert tensor. Layers at or after `first_mxfp8`
// are MXFP8 (20 in the release), the rest NVFP4.
inline std::string quant_json(int layers = 24, int first_mxfp8 = 20) {
  auto names = [&](int lo, int hi) {
    std::string out;
    for (int l = lo; l < hi; ++l)
      for (const char* w : {"w13", "w2"}) {
        if (!out.empty()) out += ", ";
        out += "\"model.llm.layers." + std::to_string(l) + ".mlp.experts." + w + "_weight\"";
      }
    return out;
  };
  std::string s = "{\"model\": {\"parameter_groups\": [";
  s += "{\"compute_dtype\": \"float32\", \"output_dtype\": \"float32\", \"tensors\": [], \"weight_transform\": \"cast\"},";
  s += "{\"compute_dtype\": \"bfloat16\", \"output_dtype\": \"bfloat16\", \"tensors\": [], \"weight_transform\": \"cast\"},";
  s += "{\"compute_dtype\": \"nvfp4\", \"output_dtype\": \"uint8\", \"required_activation_quantization\": "
       "{\"amax_type\": \"per_token\", \"block_size\": 16, \"format\": \"nvfp4_e2m1\", \"rounding\": \"nearest\", "
       "\"scale_dtype\": \"float8_e4m3fn\", \"scale_selection_rule\": \"fouroversix_mae\"}, \"tensors\": [" +
       names(0, first_mxfp8) +
       "], \"weight_block_shape\": [1, 16], \"weight_scale_format\": \"amax_denominator\", "
       "\"weight_transform\": \"nvfp4_nearest\"},";
  s += "{\"compute_dtype\": \"mxfp8\", \"output_dtype\": \"float8_e4m3fn\", \"required_activation_quantization\": "
       "{\"block_size\": 32, \"format\": \"mxfp8_e4m3\", \"rounding\": \"nearest\", \"scale_dtype\": \"ue8m0\"}, "
       "\"tensors\": [" +
       names(first_mxfp8, layers) + "], \"weight_block_shape\": [32, 32], \"weight_transform\": \"mxfp8_nearest\"}";
  s += "]}, \"quant_method\": \"inkling2\"}";
  return s;
}

// The config with one occurrence of `from` replaced by `to` (throws when the
// anchor is missing, so a test cannot pass on a typo).
inline std::string config_json(const std::string& from = "", const std::string& to = "") {
  std::string s = kConfig;
  if (from.empty()) return s;
  const size_t at = s.find(from);
  if (at == std::string::npos) throw std::runtime_error("inkling2_test::config_json: anchor not found: " + from);
  return s.replace(at, from.size(), to);
}

}  // namespace inkling2_test
