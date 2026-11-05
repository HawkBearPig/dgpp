#pragma once
// Synthetic mini-checkpoint writer for the Dory1.0 tests:
// enumerates the binding table for a small config and writes config.json
// + one safetensors shard, so fixture and table cannot disagree. Values
// are deterministic per tensor NAME (glm_rng's scheme).
//
// Magnitudes are chosen around the release's calibration constants: each
// learned scale sits within +-20% of the value that makes its factor a
// no-op (sqk * sqk_factor ~ 1, s_gate * s_gate_factor ~ 1, suv * suv_factor
// ~ 1, logit_scale * logit_factor ~ 1), which is what the factors are for;
// the NVFP4 triples carry random e2m1 nibbles, e4m3 scales without the NaN
// codes and a positive per-tensor global scale in the release's band
// (~1e-4), so the dequantized projections land at the release's ~0.01
// magnitude instead of swamping the residual.
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
#include "models/dory/binding.hpp"
#include "models/dory/config.hpp"

namespace doryfx {

namespace fs = std::filesystem;
using dgpp::float_to_bf16_bits;
using dgpp::float_to_fp8_e4m3_bits;
using dgpp::DoryConfig;
using dgpp::DoryExpectedTensor;
using dgpp::DoryTensorRole;
using dgpp::DoryWeightClass;
using glmrng::Rng;
using glmrng::seed_for;

// The tiny release: head_dim is the kernels' fixed 256, with 4 query and 2
// kv heads (hpk 2) so the whole attention block stays small; hidden 256 and
// intermediate 512 keep every justnorm width inside one block (q_size 1024,
// gate_up 1024). Eight layers reproduce the release's 8-cycle
// [sliding, mlp, sliding, mlp, sliding, mlp, full, mlp] with a 2/4/2
// input/recurrent/output split and two loops, so the fixture exercises both
// the sliding and the full attention kinds, the loop's re-injection and the
// per-loop K/V planes. The 8-token window bites on the test's sequences.
inline const char* tiny_config_json(int loops = 2, int window = 8) {
  static std::string s;
  s = std::string(R"json({
  "architectures": ["DoryForCausalLM"], "model_type": "dory",
  "vocab_size": 512, "hidden_size": 256, "num_hidden_layers": 8,
  "intermediate_size": 512, "num_attention_heads": 4, "num_key_value_heads": 2,
  "max_position_embeddings": 4096, "tie_word_embeddings": false,
  "ngpt_alpha_init": 0.05, "attention_output_gate_logit_scale_init": null,
  "mlp_bias": false, "attention_bias": false,
  "n_input_layers": 2, "n_recurrent_layers": 4, "n_output_layers": 2,
  "n_recurrent_loops": )json") + std::to_string(loops) + R"json(,
  "head_dim": 256, "hidden_act": "silu", "initializer_range": 0.0177,
  "dtype": "bfloat16", "sliding_window": )json" + std::to_string(window) + R"json(,
  "layer_types": ["sliding_attention", "mlp", "sliding_attention", "mlp",
                  "sliding_attention", "mlp", "full_attention", "mlp"],
  "rope_parameters": {
    "sliding_attention": {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 1.0},
    "full_attention": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 1.0}
  },
  "quantization_config": {
    "config_groups": {"group_0": {
      "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
      "targets": ["Linear"]}},
    "ignore": ["backbone.embeddings", "lm_head"],
    "quant_algo": "W4A16_NVFP4",
    "producer": {"name": "modelopt", "version": "0.0.0"},
    "quant_method": "modelopt"
  }
})json";
  return s.c_str();
}

inline DoryConfig tiny_config(int loops = 2, int window = 8) {
  const auto t = dgpp::minijson::parse(tiny_config_json(loops, window));
  return DoryConfig::parse(t.root);
}

// +-20% around `neutral`, the value whose factor product is one.
inline float scaled(Rng& rng, float neutral) {
  return neutral * (0.8f + 0.4f * (0.5f * (rng.unit() + 1.0f)));
}

inline std::vector<uint8_t> tensor_bytes(const DoryConfig& cfg, const DoryExpectedTensor& e) {
  std::vector<uint8_t> out(e.nbytes());
  const std::string& name = e.name;
  Rng rng(seed_for(name));
  const size_t n = e.numel();
  for (size_t i = 0; i < n; ++i) {
    float v;
    switch (e.role) {
      case DoryTensorRole::Fp4Payload:
        out[i] = static_cast<uint8_t>(rng.next() & 0xFFu);  // two random e2m1 codes
        continue;
      case DoryTensorRole::Fp4Scale: {
        // e4m3 scales in [1, 64): no NaN codes, magnitudes the payload's
        // +-6 range spreads over (dequant ~ code x scale x global).
        const float s = 1.0f + 63.0f * (0.5f * (rng.unit() + 1.0f));
        out[i] = float_to_fp8_e4m3_bits(s);
        continue;
      }
      case DoryTensorRole::Fp4Global:
        // weight_scale_2 ~ 1e-4..2e-4: with scales in [1, 64) and codes up
        // to 6 the dequantized weights sit at ~0.01 (max ~0.08), the
        // magnitude of the release's projections.
        v = 1.0e-4f + 1.0e-4f * (0.5f * (rng.unit() + 1.0f));
        break;
      default:
        switch (e.cls) {
          case DoryWeightClass::Embed:
          case DoryWeightClass::LmHead:
            v = 0.3f * rng.normal3();
            break;
          case DoryWeightClass::LogitScale:
            v = scaled(rng, 1.0f / cfg.logit_factor());
            break;
          case DoryWeightClass::AttentionScale:  // s_gate, sqk
            v = scaled(rng, 1.0f / (e.name.find("sqk") != std::string::npos ? cfg.sqk_factor()
                                                                           : cfg.s_gate_factor()));
            break;
          case DoryWeightClass::MlpScale:  // suv_gate, suv_up
            v = scaled(rng, 1.0f / cfg.suv_factor());
            break;
          case DoryWeightClass::AttentionAlpha:
          case DoryWeightClass::MlpAlpha:
            v = scaled(rng, cfg.ngpt_alpha_init);
            break;
          default:
            v = 0.05f * rng.normal3();
            break;
        }
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
inline void write_fixture(const DoryConfig& cfg, const std::string& dir, const char* config_json) {
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
  const auto table = dgpp::dory_expected_text_tensors(cfg);
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
  std::printf("dory fixture written: %zu tensors, %.2f MB payload\n", table.size(),
              static_cast<double>(data.size()) / 1048576.0);
}

inline void write_fixture(const DoryConfig& cfg, const std::string& dir) {
  write_fixture(cfg, dir, tiny_config_json());
}

}  // namespace doryfx
