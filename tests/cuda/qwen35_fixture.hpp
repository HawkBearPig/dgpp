#pragma once
// Synthetic mini-checkpoint writer for the Qwen3.8-27B (qwen3_5) tests
// (#84, 2026-10-03): enumerates the binding table for a small config and
// writes config.json + one safetensors shard, so fixture and table cannot
// disagree. Values are deterministic per tensor NAME (glm_rng's scheme),
// in magnitudes that keep the forward's nonlinearities informative; the
// block-FP8 matrices carry random e4m3 codes (no NaN codes) under BF16
// block scales that put the dequantized weights at the release's ~0.05
// rms. One vision tensor rides along in the release's naming, so the
// loader's ignore rule is exercised by every fixture gate.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "glm_rng.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen/binding35.hpp"
#include "models/qwen/config35.hpp"

namespace qwen35fx {

namespace fs = std::filesystem;
using dgpp::float_to_bf16_bits;
using dgpp::Qwen35TextConfig;
using dgpp::QwenExpectedTensor;
using dgpp::QwenTensorRole;
using dgpp::QwenWeightClass;
using glmrng::Rng;
using glmrng::seed_for;

// The tiny release: the kernels' pinned widths (128-wide GDN heads,
// 256-wide attention heads) at the smallest counts that keep worlds 1, 2
// and 4 legal (4 GDN key heads x 8 value heads, 4 query heads x 2 kv heads:
// the kv heads pair up at world 4), hidden 256, three GDN layers
// then one full-attention layer (full_attention_interval 4), the draft
// layer, dense intermediate 512, vocab 512. Every FP8 matrix's rows and
// columns are multiples of 128 except none — the 128 x 128 scale grid is
// exact — and the hidden size keeps every GEMM on the ragged-free path the
// release takes.
inline const char* tiny_config_text_json() {
  return R"json({
  "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
  "language_model_only": true, "tie_word_embeddings": false,
  "text_config": {
    "model_type": "qwen3_5_text", "attention_bias": false, "attn_output_gate": true,
    "bos_token_id": 1, "eos_token_id": 1, "full_attention_interval": 4,
    "head_dim": 256, "hidden_act": "silu", "hidden_size": 256, "intermediate_size": 512,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 4,
    "linear_num_value_heads": 8, "linear_value_head_dim": 128, "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 4096, "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false,
    "num_attention_heads": 4, "num_hidden_layers": 4, "num_key_value_heads": 2,
    "output_gate_type": "swish", "partial_rotary_factor": 0.25, "rms_norm_eps": 1e-06,
    "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10],
                        "partial_rotary_factor": 0.25, "rope_theta": 10000000.0, "rope_type": "default"},
    "tie_word_embeddings": false, "vocab_size": 512
  },
  "quantization_config": )json";
}

inline const char* tiny_quant_fp8() {
  return R"json({"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                 "weight_block_size": [128, 128]}
})json";
}

// The NVFP4 mixed release, tiny: the attention stack and lm_head ride the
// float-quantized channel group, layers 0-2's MLPs the NVFP4 group, the
// full-attention layer's (3) MLP the channel group (the release's late-layer
// regex, narrowed), and the draft layer is BF16 via the `^mtp.*` ignore.
inline const char* tiny_quant_mixed() {
  return R"json({"quant_method": "compressed-tensors", "format": "mixed-precision",
    "config_groups": {
      "group_0": {"format": "float-quantized",
        "targets": ["re:.*self_attn\\.(q|k|v|o)_proj$",
                    "re:.*linear_attn\\.(in_proj_qkv|in_proj_z|out_proj)$",
                    "re:.*layers\\.3\\.mlp\\.(gate|up|down)_proj$",
                    "re:.*lm_head"],
        "weights": {"num_bits": 8, "type": "float", "strategy": "channel"}},
      "group_1": {"format": "nvfp4-pack-quantized",
        "targets": ["re:.*mlp\\.(gate|up|down)_proj$"],
        "weights": {"num_bits": 4, "type": "float", "group_size": 16}}},
    "ignore": ["re:^mtp.*"],
    "kv_cache_scheme": {"num_bits": 8, "type": "float", "strategy": "tensor"}}
})json";
}

inline std::string tiny_config_json(bool mixed = false) {
  return std::string(tiny_config_text_json()) + (mixed ? tiny_quant_mixed() : tiny_quant_fp8());
}

// The tiny MoE release (122B layout at toy widths): 8 experts run for every
// row (top-8-of-8: no 2nd/3rd selection boundary, so no rows sit on
// knife-edges where the kernel's bf16 logit rounding flips against an fp64
// reference — both correct, outputs totally different; the full softmax +
// renorm still divides for real and every expert runs every row), ModelOpt
// NVFP4 routed experts, everything else BF16 incl. the MTP draft's experts.
// Top-k selection itself rides the shared MoE kernels (other families'
// fixtures) plus the live-checkpoint evidence (correct routed answers).
inline const char* tiny_moe_config_text_json() {
  return R"json({
  "architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe",
  "language_model_only": true, "tie_word_embeddings": false,
  "text_config": {
    "model_type": "qwen3_5_moe_text", "attention_bias": false, "attn_output_gate": true,
    "bos_token_id": 1, "eos_token_id": 1, "full_attention_interval": 4,
    "head_dim": 256, "hidden_act": "silu", "hidden_size": 256,
    "moe_intermediate_size": 64, "shared_expert_intermediate_size": 64,
    "num_experts": 8, "num_experts_per_tok": 8, "norm_topk_prob": true,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 4,
    "linear_num_value_heads": 8, "linear_value_head_dim": 128, "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 4096, "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false,
    "num_attention_heads": 4, "num_hidden_layers": 4, "num_key_value_heads": 2,
    "output_gate_type": "swish", "partial_rotary_factor": 0.25, "rms_norm_eps": 1e-06,
    "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10],
                        "partial_rotary_factor": 0.25, "rope_theta": 10000000.0, "rope_type": "default"},
    "tie_word_embeddings": false, "vocab_size": 512
  },
  "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4",
    "config_groups": {"group_0": {"targets": ["Linear"],
      "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16}}},
    "ignore": ["lm_head", "model.visual*",
               "model.language_model.layers.*.linear_attn*",
               "model.language_model.layers.*.mlp.shared_expert*",
               "model.language_model.layers.*.self_attn*",
               "mtp.layers.0*"]}
})json";
}

inline Qwen35TextConfig tiny_moe_config() {
  const std::string json = tiny_moe_config_text_json();
  const auto t = dgpp::minijson::parse(json);
  return Qwen35TextConfig::parse(*t.root.find("text_config"), t.root.find("quantization_config"));
}

inline Qwen35TextConfig tiny_config(bool mixed = false) {
  const std::string json = tiny_config_json(mixed);
  const auto t = dgpp::minijson::parse(json);
  return Qwen35TextConfig::parse(*t.root.find("text_config"), t.root.find("quantization_config"));
}

inline bool has(const std::string& name, const char* needle) {
  return name.find(needle) != std::string::npos;
}

inline std::vector<uint8_t> tensor_bytes(const QwenExpectedTensor& e, float fp4_global = 64.0f) {
  std::vector<uint8_t> out(e.nbytes());
  const std::string& name = e.name;
  Rng rng(seed_for(name));
  const size_t n = e.numel();
  const bool is_norm = has(name, "norm") && e.shape.size() == 1;  // (1 + w) gains near 1: w near 0
  const bool is_a_log = has(name, "A_log");
  const bool is_dt_bias = has(name, "dt_bias");
  const bool is_conv = has(name, "conv1d");
  // The routed gate is random; its scale comes from DGPP_FIXTURE_ROUTER_SIGMA
  // (default 1.0) so the seed scan can pick a knife-edge-free draw: too
  // narrow leaves rows where the kernel's bf16 logit rounding flips
  // 2nd-vs-3rd against an fp64 reference (both correct, outputs totally
  // different); too wide amplifies input noise into routing-weight noise.
  // The shared gate stays small random (a sigmoid scalar).
  const bool is_routed_gate =
      has(name, "mlp.gate.weight") && !has(name, "shared");
  if (is_routed_gate && e.dtype == dgpp::DType::BF16) {
    float sigma = 1.0f;
    if (const char* es = std::getenv("DGPP_FIXTURE_ROUTER_SIGMA"); es != nullptr) sigma = std::atof(es);
    const size_t n = e.numel();
    std::vector<uint16_t> wbits(n);
    for (size_t i = 0; i < n; ++i) wbits[i] = float_to_bf16_bits(sigma * rng.normal3());
    std::vector<uint8_t> out(wbits.size() * 2);
    std::memcpy(out.data(), wbits.data(), out.size());
    return out;
  }
  for (size_t i = 0; i < n; ++i) {
    float v;
    switch (e.role) {
      case QwenTensorRole::Fp8Payload: {
        uint8_t b = static_cast<uint8_t>(rng.next() & 0xFFu);  // a random e4m3 code, never a NaN
        if ((b & 0x7Fu) == 0x7Fu) b = static_cast<uint8_t>(b & 0xF0u);
        out[i] = b;
        continue;
      }
      case QwenTensorRole::Fp8Scale:
        // Random e4m3 codes have an rms near 100; a block scale of 4e-4 ..
        // 6e-4 puts the dequantized weights at the ~0.05 rms of the
        // release's projections. (Also the channel grid's [N, 1] BF16 row.)
        v = 4.0e-4f + 2.0e-4f * (0.5f * (rng.unit() + 1.0f));
        break;
      case QwenTensorRole::Fp4Payload:
        out[i] = static_cast<uint8_t>(rng.next() & 0xFFu);  // any e2m1 nibble pair
        continue;
      case QwenTensorRole::Fp4Scale: {
        // e4m3 codes around 1.0: with the 0.02 global below (and the e2m1
        // codes' ~2 rms) the fp4 weights land near the release's ~0.05 rms.
        static const uint8_t kNearOne[6] = {0x30, 0x34, 0x38, 0x3A, 0x3C, 0x40};
        out[i] = kNearOne[rng.next() % 6u];
        continue;
      }
      case QwenTensorRole::Fp4Global:
        // Compressed-tensors: the checkpoint's weight-side divisor (kernels
        // divide). ModelOpt (write_moe_fixture passes 1/64): weight_scale_2
        // is a multiplier the loader reciprocals — same served weights.
        v = fp4_global;
        break;
      case QwenTensorRole::InputScale:
        v = 1.0f;  // input_global_scale / kv scales: bound, never resident
        break;
      default:
        if (is_norm) v = 0.1f * rng.unit();                 // the zero-centered (1 + w) form
        else if (is_a_log) v = 0.5f + 0.4f * std::fabs(rng.normal3());  // A = exp(A_log) in ~[1.6, 5]
        else if (is_dt_bias) v = 0.5f * rng.normal3();
        else if (is_conv) v = 0.3f * rng.normal3();
        else if (e.cls == QwenWeightClass::Embed || e.cls == QwenWeightClass::LmHead) v = 0.3f * rng.normal3();
        else if (e.cls == QwenWeightClass::Router) v = 0.3f * rng.normal3();  // peaky router: separable top-k
        else v = 0.05f * rng.normal3();  // in_proj_a / in_proj_b, mtp.fc
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

// The unserved tensor the fixture carries beside the table: a vision
// block's norm in the release's own naming (the ignore rule admits it).
inline std::vector<QwenExpectedTensor> ignored_extras() {
  std::vector<QwenExpectedTensor> out;
  QwenExpectedTensor e;
  e.name = "model.visual.blocks.0.norm1.weight";
  e.dtype = dgpp::DType::BF16;
  e.shape = {64};
  e.cls = QwenWeightClass::Norm;
  e.role = QwenTensorRole::Plain;
  out.push_back(std::move(e));
  return out;
}

// Writes `dir` (config.json + one safetensors shard) for the tiny release.
// `mixed` swaps the FP8 quantization_config for the NVFP4 mixed release's
// (the same tensors' table comes out of the binding, in the other forms).
inline void write_fixture_from_cfg(const std::string& dir, const Qwen35TextConfig& cfg,
                                   const std::string& config_json,
                                   float fp4_global = 64.0f) {
  fs::path root(dir);
  fs::remove_all(root);
  fs::create_directories(root);
  {
    const fs::path p = root / "config.json";
    std::FILE* f = std::fopen(p.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write config.json");
    std::fwrite(config_json.data(), 1, config_json.size(), f);
    std::fclose(f);
  }
  auto table = dgpp::qwen35_expected_text_tensors(cfg);
  for (auto& e : ignored_extras()) table.push_back(std::move(e));
  std::string header = "{";
  std::vector<uint8_t> data;
  size_t off = 0;
  bool first = true;
  for (const auto& e : table) {
    const auto b = tensor_bytes(e, fp4_global);
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
  std::printf("qwen35 fixture written: %zu tensors, %.2f MB payload\n", table.size(),
              static_cast<double>(data.size()) / 1048576.0);
}

inline void write_fixture(const std::string& dir, bool mixed = false) {
  write_fixture_from_cfg(dir, tiny_config(mixed), tiny_config_json(mixed));
}

inline void write_moe_fixture(const std::string& dir) {
  // ModelOpt multiplier so the loader's reciprocal lands on the same
  // divide-by-64 the dense fixture serves.
  write_fixture_from_cfg(dir, tiny_moe_config(), tiny_moe_config_text_json(), 1.0f / 64.0f);
}

}  // namespace qwen35fx
