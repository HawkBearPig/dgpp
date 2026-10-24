#pragma once
// Synthetic mini-checkpoint writer for the Flint tests: enumerates the
// binding table for a small config and writes config.json + one
// safetensors shard, so fixture and table cannot disagree. Values are
// deterministic per tensor NAME (glm_rng's scheme), in magnitudes that
// keep the forward's nonlinearities informative; the fp8 experts carry
// random e4m3 codes (no NaN codes) under block scales that put the
// dequantized weights at ~0.05 rms, the NVFP4 experts random e2m1 nibbles
// under e4m3 scales and a global near 1 at the same rms.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "glm_rng.hpp"
#include "loaders/minijson.hpp"
#include "models/flint/binding.hpp"
#include "models/flint/config.hpp"

namespace flintfx {

namespace fs = std::filesystem;
using dgpp::float_to_bf16_bits;
using dgpp::FlintExpectedTensor;
using dgpp::FlintTensorRole;
using dgpp::FlintTextConfig;
using dgpp::FlintWeightClass;
using glmrng::Rng;
using glmrng::seed_for;

// The tiny release: 128-wide heads at 8 query / 4 kv heads (worlds 1, 2
// and 4 legal), hidden 256, six layers — full (no RoPE, dense), sliding
// (dense), sliding (fp8 MoE), full (no RoPE; gate fp8 / up NVFP4 / down
// fp8: the release's layer-12 form), sliding (gate + up NVFP4, down fp8:
// layer 37's), sliding (NVFP4 MoE) — 8 experts of 512 (a 128-wide slice
// at world 4), a 256-wide shared expert, dense intermediate 512, vocab
// 512, a 33-token window (not a tile multiple, as the release's 513; a
// 72-token prompt looks past it), eos [1, 2].
inline const char* tiny_config_json() {
  static const char* s = R"json({
  "architectures": ["FlintForCausalLM"], "model_type": "flint",
  "attention_bias": false, "bos_token_id": 0, "eos_token_id": [1, 2], "pad_token_id": 1,
  "head_dim": 128, "hidden_act": "silu", "hidden_size": 256,
  "intermediate_size": 512, "max_position_embeddings": 4096,
  "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention",
                  "sliding_attention", "sliding_attention"],
  "no_rope_layer_indices": [0, 3], "mlp_only_layers": [0, 1], "decoder_sparse_step": 1,
  "moe_intermediate_size": 512, "shared_expert_intermediate_size": 256,
  "num_experts": 8, "num_experts_per_tok": 2, "norm_topk_prob": true, "router_score_function": "sigmoid",
  "num_attention_heads": 8, "num_key_value_heads": 4, "num_hidden_layers": 6,
  "rms_norm_eps": 1e-06, "rope_theta": 10000.0, "sliding_window": 33, "use_sliding_window": true,
  "softmax_type": "learnable", "tie_word_embeddings": false, "vocab_size": 512,
  "quantization_config": {
    "quant_method": "dgpp_mixed", "activations": "bfloat16",
    "routed_experts": {
      "layers": [2, 3, 4, 5], "projections": ["gate", "up", "down"],
      "fp8_format": {"weight_dtype": "float8_e4m3fn", "weight_block_size": [128, 128],
                     "scale_dtype": "float32", "scale_suffix": "weight_scale_inv"},
      "nvfp4_format": {"weight_dtype": "float4_e2m1", "group_size": 16, "scale_dtype": "float8_e4m3fn",
                       "global_scale_dtype": "float32"},
      "precision_by_layer": {
        "2": {"gate": "fp8", "up": "fp8", "down": "fp8"},
        "3": {"gate": "fp8", "up": "nvfp4", "down": "fp8"},
        "4": {"gate": "nvfp4", "up": "nvfp4", "down": "fp8"},
        "5": {"gate": "nvfp4", "up": "nvfp4", "down": "nvfp4"}
      }
    }
  }
})json";
  return s;
}

// The window gate's two-layer stack (flint_decode_test): a sliding dense
// layer then a full MoE layer (fp8), the 33-token window, no RoPE on the
// full layer.
inline const char* window_config_json() {
  static const char* s = R"json({
  "architectures": ["FlintForCausalLM"], "model_type": "flint",
  "attention_bias": false, "bos_token_id": 0, "eos_token_id": [1, 2], "pad_token_id": 1,
  "head_dim": 128, "hidden_act": "silu", "hidden_size": 256,
  "intermediate_size": 512, "max_position_embeddings": 4096,
  "layer_types": ["sliding_attention", "full_attention"],
  "no_rope_layer_indices": [1], "mlp_only_layers": [0], "decoder_sparse_step": 1,
  "moe_intermediate_size": 512, "shared_expert_intermediate_size": 256,
  "num_experts": 8, "num_experts_per_tok": 2, "norm_topk_prob": true, "router_score_function": "sigmoid",
  "num_attention_heads": 8, "num_key_value_heads": 4, "num_hidden_layers": 2,
  "rms_norm_eps": 1e-06, "rope_theta": 10000.0, "sliding_window": 33, "use_sliding_window": true,
  "softmax_type": "learnable", "tie_word_embeddings": false, "vocab_size": 512,
  "quantization_config": {
    "quant_method": "dgpp_mixed", "activations": "bfloat16",
    "routed_experts": {
      "layers": [1], "projections": ["gate", "up", "down"],
      "fp8_format": {"weight_dtype": "float8_e4m3fn", "weight_block_size": [128, 128],
                     "scale_dtype": "float32", "scale_suffix": "weight_scale_inv"},
      "nvfp4_format": {"weight_dtype": "float4_e2m1", "group_size": 16, "scale_dtype": "float8_e4m3fn",
                       "global_scale_dtype": "float32"},
      "precision_by_layer": {"1": {"gate": "fp8", "up": "fp8", "down": "fp8"}}
    }
  }
})json";
  return s;
}

inline FlintTextConfig tiny_config() {
  const auto t = dgpp::minijson::parse(tiny_config_json());
  return FlintTextConfig::parse(t.root);
}

inline bool has(const std::string& name, const char* needle) { return name.find(needle) != std::string::npos; }

inline std::vector<uint8_t> tensor_bytes(const FlintTextConfig& cfg, const FlintExpectedTensor& e) {
  (void)cfg;
  std::vector<uint8_t> out(e.nbytes());
  const std::string& name = e.name;
  Rng rng(seed_for(name));
  const size_t n = e.numel();
  const bool is_norm = has(name, "norm") && e.shape.size() == 1;  // gains near 1
  const bool is_sink = has(name, "softmax_offset");
  for (size_t i = 0; i < n; ++i) {
    float v;
    switch (e.role) {
      case FlintTensorRole::Fp8Payload: {
        // A random e4m3 code, never a NaN (0x7F / 0xFF).
        uint8_t b = static_cast<uint8_t>(rng.next() & 0xFFu);
        if ((b & 0x7Fu) == 0x7Fu) b = static_cast<uint8_t>(b & 0xF0u);
        out[i] = b;
        continue;
      }
      case FlintTensorRole::Fp8Scale:
        // Random e4m3 codes have an rms near 100; a block scale of 4e-4 ..
        // 6e-4 puts the dequantized weights at ~0.05 rms.
        v = 4.0e-4f + 2.0e-4f * (0.5f * (rng.unit() + 1.0f));
        break;
      case FlintTensorRole::Fp4Payload:
        out[i] = static_cast<uint8_t>(rng.next() & 0xFFu);  // two random e2m1 codes
        continue;
      case FlintTensorRole::Fp4Scale:
        // e4m3 scales of 0.01 .. 0.04 (an e2m1 rms of ~2.9 and a global
        // near 1 land the experts at ~0.05).
        out[i] = dgpp::float_to_fp8_e4m3_bits(0.02f * static_cast<float>(std::exp2(rng.unit())));
        continue;
      case FlintTensorRole::Fp4Global:
        v = 1.0f + 0.5f * rng.unit();  // the reciprocal global: the kernels divide by it
        break;
      case FlintTensorRole::Plain:
      default:
        if (is_sink) v = 0.5f * rng.normal3();
        else if (is_norm) v = 0.9f + 0.2f * (0.5f * (rng.unit() + 1.0f));
        else if (e.cls == FlintWeightClass::Embed || e.cls == FlintWeightClass::LmHead) v = 0.3f * rng.normal3();
        else v = 0.05f * rng.normal3();  // the projections, the MLPs, the router
        break;
    }
    if (e.dtype == dgpp::DType::BF16) {
      const uint16_t bits = float_to_bf16_bits(v);
      std::memcpy(&out[i * 2], &bits, 2);
    } else if (e.dtype == dgpp::DType::F32) {
      std::memcpy(&out[i * 4], &v, 4);
    } else {
      throw std::runtime_error("fixture dtype not handled: " + name);
    }
  }
  return out;
}

// Writes `dir` (config.json + one safetensors shard) for `cfg`.
inline void write_fixture(const FlintTextConfig& cfg, const std::string& dir, const char* config_json) {
  fs::path root(dir);
  fs::remove_all(root);
  fs::create_directories(root);
  {
    const fs::path p = root / "config.json";
    std::FILE* f = std::fopen(p.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write config.json");
    std::fwrite(config_json, 1, std::strlen(config_json), f);
    std::fclose(f);
  }
  const auto table = dgpp::flint_expected_text_tensors(cfg);
  std::string header = "{";
  std::vector<uint8_t> data;
  size_t off = 0;
  bool first = true;
  for (const auto& e : table) {
    const auto b = tensor_bytes(cfg, e);
    std::string shape = "[";
    for (size_t i = 0; i < e.shape.size(); ++i) {
      if (i) shape += ",";
      shape += std::to_string(e.shape[i]);
    }
    shape += "]";
    if (!first) header += ",";
    first = false;
    header += "\"" + e.name + "\":{\"dtype\":\"" + std::string(dgpp::dtype_name(e.dtype)) +
              "\",\"shape\":" + shape + ",\"data_offsets\":[" + std::to_string(off) + "," +
              std::to_string(off + b.size()) + "]}";
    data.insert(data.end(), b.begin(), b.end());
    off += b.size();
  }
  header += "}";
  const fs::path shard = root / "model.safetensors";
  std::FILE* f = std::fopen(shard.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write shard");
  const uint64_t hlen = header.size();
  std::fwrite(&hlen, 8, 1, f);
  std::fwrite(header.data(), 1, hlen, f);
  std::fwrite(data.data(), 1, data.size(), f);
  std::fclose(f);
  std::printf("flint fixture written: %zu tensors, %.2f MB payload\n", table.size(),
              static_cast<double>(data.size()) / 1048576.0);
}

inline void write_fixture(const FlintTextConfig& cfg, const std::string& dir) {
  write_fixture(cfg, dir, tiny_config_json());
}

}  // namespace flintfx
