#pragma once
// Shared fixture: the Qwen/Qwen3.8-27B-FP8 text_config and
// quantization_config, transcribed (vision omitted). Both qwen35 test files
// build from these so the tables and the parser cannot drift apart.
#include <string>

namespace qwen35_fixture {

inline std::string layers_json() {
  std::string s;
  for (int i = 0; i < 64; ++i) {
    if (i) s += ", ";
    s += (i + 1) % 4 == 0 ? "\"full_attention\"" : "\"linear_attention\"";
  }
  return s;
}

inline std::string text_json() {
  return std::string(R"({
  "model_type": "qwen3_5_text", "attention_bias": false, "bos_token_id": 248044,
  "dtype": "bfloat16", "eos_token_id": 248044, "full_attention_interval": 4,
  "head_dim": 256, "hidden_act": "silu", "hidden_size": 5120,
  "intermediate_size": 17408,
  "layer_types": [)") + layers_json() + R"(],
  "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
  "linear_num_key_heads": 16, "linear_num_value_heads": 48,
  "linear_value_head_dim": 128, "mamba_ssm_dtype": "float32",
  "max_position_embeddings": 262144, "mtp_num_hidden_layers": 1,
  "mtp_use_dedicated_embeddings": false, "num_attention_heads": 24,
  "num_hidden_layers": 64, "num_key_value_heads": 4,
  "output_gate_type": "swish", "partial_rotary_factor": 0.25,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10],
                      "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                      "rope_type": "default"},
  "tie_word_embeddings": false, "vocab_size": 248320
})";
}

inline const char* kQuantFp8 =
    R"({"quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": ["model.visual.blocks.0.attn.proj"]})";

inline const char* kQuantNvfp4Mixed =
    R"({"quant_method": "compressed-tensors", "format": "mixed-precision",
        "config_groups": {
          "group_0": {"weights": {"num_bits": 8, "type": "float", "group_size": 128}},
          "group_1": {"weights": {"num_bits": 4, "type": "float", "group_size": 16}}}})";

}  // namespace qwen35_fixture
