#pragma once
// The Flint config.json (HawkBearPig/flint-FP8-NVFP4, revision 95e74655,
// transcribed 2026-10-22) as the unit tests build it. Shared by the config
// and binding tests; `config_json(from, to)` patches one anchor.
#include <stdexcept>
#include <string>

namespace flint_test {

inline const char* kConfig = R"JSON({
 "architectures": [
  "FlintForCausalLM"
 ],
 "attention_bias": false,
 "attention_dropout": 0.0,
 "bos_token_id": 150000,
 "decoder_sparse_step": 1,
 "eos_token_id": [
  150001,
  150003,
  150005,
  150018
 ],
 "head_dim": 128,
 "hidden_act": "silu",
 "hidden_size": 2048,
 "initializer_range": 0.02,
 "intermediate_size": 6144,
 "layer_types": [
  "full_attention",
  "full_attention",
  "sliding_attention",
  "sliding_attention",
  "sliding_attention",
  "full_attention",
  "sliding_attention",
  "sliding_attention",
  "sliding_attention",
  "full_attention",
  "sliding_attention",
  "sliding_attention",
  "sliding_attention",
  "full_attention",
  "sliding_attention",
  "sliding_attention",
  "sliding_attention",
  "full_attention",
  "sliding_attention",
  "sliding_attention",
  "sliding_attention",
  "full_attention",
  "sliding_attention",
  "sliding_attention",
  "sliding_attention",
  "full_attention",
  "sliding_attention",
  "sliding_attention",
  "sliding_attention",
  "full_attention",
  "sliding_attention",
  "sliding_attention",
  "sliding_attention",
  "full_attention",
  "sliding_attention",
  "sliding_attention",
  "sliding_attention",
  "full_attention",
  "sliding_attention",
  "sliding_attention",
  "sliding_attention",
  "full_attention",
  "sliding_attention",
  "sliding_attention",
  "sliding_attention",
  "full_attention",
  "sliding_attention",
  "sliding_attention",
  "full_attention"
 ],
 "max_position_embeddings": 262144,
 "mlp_only_layers": [
  0,
  1
 ],
 "model_type": "flint",
 "moe_intermediate_size": 512,
 "no_rope_layer_indices": [
  0,
  1,
  5,
  9,
  13,
  17,
  21,
  25,
  29,
  33,
  37,
  41,
  45,
  48
 ],
 "norm_topk_prob": true,
 "num_attention_heads": 32,
 "num_experts": 256,
 "num_experts_per_tok": 6,
 "num_hidden_layers": 49,
 "num_key_value_heads": 4,
 "output_router_logits": false,
 "pad_token_id": 150001,
 "quantization_config": {
  "activations": "bfloat16",
  "other_weights": "source BF16, byte-identical",
  "quant_method": "dgpp_mixed",
  "quantization_status": "compressed",
  "recipe": "layer-policy",
  "routed_experts": {
   "fp8_format": {
    "dequantize": "float32(weight) * weight_scale_inv[row // 128, col // 128]",
    "scale_dtype": "float32",
    "scale_suffix": "weight_scale_inv",
    "weight_block_size": [
     128,
     128
    ],
    "weight_dtype": "float8_e4m3fn"
   },
   "layers": [
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48
   ],
   "nvfp4_format": {
    "dequantize": "e2m1(code) * (float32(weight_scale) / weight_global_scale)",
    "format": "nvfp4-pack-quantized",
    "global_scale_dtype": "float32",
    "group_size": 16,
    "packing": "low nibble is even input coordinate",
    "scale_dtype": "float8_e4m3fn",
    "weight_dtype": "float4_e2m1"
   },
   "precision_by_layer": {
    "10": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "11": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "12": {
     "down": "fp8",
     "gate": "fp8",
     "up": "nvfp4"
    },
    "13": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "14": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "15": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "16": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "17": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "18": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "19": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "2": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "20": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "21": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "22": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "23": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "24": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "25": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "26": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "27": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "28": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "29": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "3": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "30": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "31": {
     "down": "nvfp4",
     "gate": "nvfp4",
     "up": "nvfp4"
    },
    "32": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "33": {
     "down": "nvfp4",
     "gate": "nvfp4",
     "up": "nvfp4"
    },
    "34": {
     "down": "nvfp4",
     "gate": "nvfp4",
     "up": "nvfp4"
    },
    "35": {
     "down": "nvfp4",
     "gate": "nvfp4",
     "up": "nvfp4"
    },
    "36": {
     "down": "nvfp4",
     "gate": "nvfp4",
     "up": "nvfp4"
    },
    "37": {
     "down": "fp8",
     "gate": "nvfp4",
     "up": "nvfp4"
    },
    "38": {
     "down": "nvfp4",
     "gate": "nvfp4",
     "up": "nvfp4"
    },
    "39": {
     "down": "nvfp4",
     "gate": "nvfp4",
     "up": "nvfp4"
    },
    "4": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "40": {
     "down": "nvfp4",
     "gate": "nvfp4",
     "up": "nvfp4"
    },
    "41": {
     "down": "nvfp4",
     "gate": "nvfp4",
     "up": "nvfp4"
    },
    "42": {
     "down": "nvfp4",
     "gate": "nvfp4",
     "up": "nvfp4"
    },
    "43": {
     "down": "nvfp4",
     "gate": "nvfp4",
     "up": "nvfp4"
    },
    "44": {
     "down": "nvfp4",
     "gate": "nvfp4",
     "up": "nvfp4"
    },
    "45": {
     "down": "nvfp4",
     "gate": "nvfp4",
     "up": "nvfp4"
    },
    "46": {
     "down": "nvfp4",
     "gate": "nvfp4",
     "up": "nvfp4"
    },
    "47": {
     "down": "nvfp4",
     "gate": "nvfp4",
     "up": "nvfp4"
    },
    "48": {
     "down": "nvfp4",
     "gate": "nvfp4",
     "up": "nvfp4"
    },
    "5": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "6": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "7": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "8": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    },
    "9": {
     "down": "fp8",
     "gate": "fp8",
     "up": "fp8"
    }
   },
   "projections": [
    "gate",
    "up",
    "down"
   ],
   "retained_bf16_layers": []
  }
 },
 "rms_norm_eps": 1e-06,
 "rope_theta": 10000.0,
 "router_aux_loss_coef": 0.0005,
 "router_score_function": "sigmoid",
 "shared_expert_intermediate_size": 1024,
 "sliding_window": 513,
 "softmax_type": "learnable",
 "tie_word_embeddings": false,
 "torch_dtype": "bfloat16",
 "transformers_version": "5.12.1",
 "use_cache": true,
 "use_sliding_window": true,
 "vocab_size": 150144
})JSON";

// The release's config with one anchor substring replaced (from -> to);
// an anchor that does not occur is a test bug.
inline std::string config_json(const std::string& from = "", const std::string& to = "") {
  std::string s = kConfig;
  if (from.empty()) return s;
  const size_t at = s.find(from);
  if (at == std::string::npos) throw std::runtime_error("flint_test::config_json: anchor not found: " + from);
  return s.replace(at, from.size(), to);
}

}  // namespace flint_test
