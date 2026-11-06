#pragma once
// Synthetic mini-checkpoint writer for the full GLM-5.3 tests: enumerates
// the binding table for a small config and writes config.json + one
// safetensors shard, so fixture and table cannot disagree. Values are
// deterministic per tensor NAME (glm_rng's scheme), in magnitudes that
// keep the forward's nonlinearities informative; the packed triples carry
// random codes, release-like bf16 group scales and exact shape records;
// the draft layer's experts are BF16 as in the release.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "glm_rng.hpp"
#include "loaders/minijson.hpp"
#include "models/glm_dsa/binding.hpp"
#include "models/glm_dsa/config.hpp"

namespace glmdsafx {

namespace fs = std::filesystem;
using dgpp::float_to_bf16_bits;
using dgpp::GlmDsaExpectedTensor;
using dgpp::GlmDsaTensorRole;
using dgpp::GlmDsaTextConfig;
using dgpp::GlmDsaWeightClass;
using glmrng::Rng;
using glmrng::seed_for;

// The tiny release: 16 heads of nope 64 + rope 64 / v 64 on latents of
// 128 (4 local heads at world 4: the split attention kernel's head-group
// floor), a 32 x 128 indexer selecting 16 tokens, one dense layer then
// four MoE layers (indexers on 0, 2 and 4: freq 2, offset 1), a draft
// layer, 8 experts of 256 (a 64-wide slice at world 4), hidden 256, vocab
// 512; layers 1-4 packed (int8 attention and shared expert, int4 experts).
// `nf4i8` (2026-10-08): the NF4I8 contract instead — the same layers, the
// routed experts as codebook `weight_indices` triples (bf16 scales per
// 128) in the H32-rotated basis; `moe_inter` then defaults to 512 so a
// world-4 slice (128) holds a whole scale group.
// attention_bits (2026-10-09): the packed attention projections' width in
// the compressed-tensors contract — 8 (the shipped checkpoints) or 4 (the
// int4-attention candidate; the shared expert keeps int8, its own group).
// `m346` (2026-10-09): the Mixed346 contract — the baseline block and the
// recipe inlined (quantization_config.baseline / .recipe), every one of the
// eight routed experts of a packed layer in its own form (expert e:
// kM346FixtureForms[e % 8], so each width pair and the existing fallback
// appear on every layer); `moe_inter` defaults to 512 as under nf4i8.
inline const char* kM346FixtureForms[8] = {"444", "334", "446", "333", "666", "443", "336", "existing"};

inline const char* tiny_config_json(int layers = 5, int dense = 1, bool mtp = true,
                                    int index_topk = 16, int experts_per_token = 2,
                                    bool nf4i8 = false, int moe_inter = 0, int attention_bits = 8,
                                    bool m346 = false) {
  static std::string s;
  const std::string last = std::to_string(layers - 1);
  std::string range = "(?:";
  for (int layer = dense; layer < layers; ++layer) {
    if (layer > dense) range += "|";
    range += std::to_string(layer);
  }
  range += ")";
  const std::string attn_bits = std::to_string(attention_bits);
  if (moe_inter == 0) moe_inter = (nf4i8 || m346) ? 512 : 256;
  const std::string head = std::string(R"json({
  "architectures": ["GlmMoeDsaForCausalLM"], "model_type": "glm_moe_dsa",
  "attention_bias": false, "eos_token_id": [1, 2], "pad_token_id": 1,
  "first_k_dense_replace": )json") +
      std::to_string(dense) + R"json(,
  "hidden_act": "silu", "hidden_size": 256, "index_head_dim": 128, "index_n_heads": 32,
  "index_share_for_mtp_iteration": true, "index_skip_topk_offset": )json" +
      std::to_string(dense) + R"json(,
  "index_topk": )json" +
      std::to_string(index_topk) + R"json(, "index_topk_freq": 2, "indexer_rope_interleave": true,
  "intermediate_size": 256, "kv_lora_rank": 128, "max_position_embeddings": 4096,
  "moe_intermediate_size": )json" +
      std::to_string(moe_inter) + R"json(, "moe_router_dtype": "float32", "n_group": 1,
  "n_routed_experts": 8, "n_shared_experts": 1, "norm_topk_prob": true,
  "num_attention_heads": 16, "num_experts_per_tok": )json" +
      std::to_string(experts_per_token) + R"json(,
  "num_hidden_layers": )json" +
      std::to_string(layers) + R"json(, "num_key_value_heads": 16,
  "num_nextn_predict_layers": )json" +
      (mtp ? "1" : "0") + R"json(,
  "q_lora_rank": 128, "qk_head_dim": 128, "qk_nope_head_dim": 64, "qk_rope_head_dim": 64,
  "rms_norm_eps": 1e-05, "rope_interleave": true,
  "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
  "routed_scaling_factor": 2.5, "scoring_func": "sigmoid", "tie_word_embeddings": false,
  "topk_group": 1, "topk_method": "noaux_tc", "v_head_dim": 64, "vocab_size": 512,
)json";
  if (nf4i8) {
    s = head + R"json(  "quantization_config": {
    "format": "mixed-codebook-packed", "format_version": 1, "quant_method": "dgpp_nf4i8",
    "retained_int8": {"group_size": 64, "layers": [)json" +
        std::to_string(dense) + ", " + last + R"json(], "scale_dtype": "bfloat16"},
    "routed_experts": {
      "bits": 4, "codebook": [-127, -88, -67, -50, -36, -23, -12, 0, 10, 20, 31, 43, 56, 71, 92, 127],
      "group_size": 128, "indices_dtype": "int32", "indices_key": "weight_indices",
      "input_transform": {"apply_to": ["gate_proj", "up_proj", "down_proj"], "axis": "input_channels",
                          "block_size": 32, "type": "normalized_hadamard"},
      "layers": [)json" +
        std::to_string(dense) + ", " + last + R"json(], "scale_dtype": "bfloat16"
    }
  }
})json";
    return s.c_str();
  }
  // The compressed-tensors block (the int4/int8 contract): config.json's
  // quantization_config, or the Mixed346 contract's baseline.
  const std::string ct = std::string(R"json({
    "config_groups": {
      "group_0": {
        "format": "pack-quantized", "input_activations": null, "output_activations": null,
        "targets": ["re:model\\.layers\\.)json") +
      range +
      R"json(\\.self_attn\\.(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)$"],
        "weights": {"actorder": null, "block_structure": null, "dynamic": false, "group_size": 64,
                    "num_bits": )json" + attn_bits + R"json(, "observer": "memoryless_minmax", "strategy": "group",
                    "symmetric": true, "type": "int", "zp_dtype": null}
      },
      "group_2": {
        "format": "pack-quantized", "input_activations": null, "output_activations": null,
        "targets": ["re:model\\.layers\\.)json" +
      range +
      R"json(\\.mlp\\.shared_experts\\.(?:gate_proj|up_proj|down_proj)$"],
        "weights": {"actorder": null, "block_structure": null, "dynamic": false, "group_size": 64,
                    "num_bits": 8, "observer": "memoryless_minmax", "strategy": "group",
                    "symmetric": true, "type": "int", "zp_dtype": null}
      },
      "group_1": {
        "format": "pack-quantized", "input_activations": null, "output_activations": null,
        "targets": ["re:model\\.layers\\.)json" +
      range + R"json(\\.mlp\\.experts\\.\\d+\\.(?:gate_proj|up_proj|down_proj)$"],
        "weights": {"actorder": null, "block_structure": null, "dynamic": false, "group_size": 64,
                    "num_bits": 4, "observer": "memoryless_minmax", "strategy": "group",
                    "symmetric": true, "type": "int", "zp_dtype": null}
      }
    },
    "format": "pack-quantized",
    "ignore": ["lm_head", "re:.*embed_tokens.*", "re:model\\.layers\\.[0-)json" +
      std::to_string(dense - 1) + R"json(]\\..*",
               "re:.*self_attn\\.indexer\\..*", "re:.*norm.*", "re:.*mlp\\.gate$"],
    "kv_cache_scheme": null, "quant_method": "compressed-tensors", "quantization_status": "compressed",
    "sparsity_config": {}, "transform_config": {}, "version": "0.18.0"
  })json";
  if (!m346) {
    s = head + "  \"quantization_config\": " + ct + "\n}";
    return s.c_str();
  }
  std::string recipes, counts;
  int n_form[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int l = dense; l < layers; ++l) {
    if (l > dense) recipes += ",\n";
    recipes += "      \"" + std::to_string(l) + "\": {";
    for (int e = 0; e < 8; ++e) {
      if (e) recipes += ", ";
      recipes += "\"" + std::to_string(e) + "\": \"" + kM346FixtureForms[e % 8] + "\"";
      ++n_form[e % 8];
    }
    recipes += "}";
  }
  for (int i = 0; i < 8; ++i) {
    if (i) counts += ", ";
    counts += "\"" + std::string(kM346FixtureForms[i]) + "\": " + std::to_string(n_form[i]);
  }
  std::string six;
  for (int i = 0; i < 64; ++i) six += (i ? ", " : "") + std::to_string(i - 32);
  s = head + R"json(  "quantization_config": {
    "activation_bits": 8, "format": "dgpp_mixed346_h32_a8_g128_v1", "group_size": 128,
    "quant_method": "dgpp_mixed346", "rotation_size": 32, "version": 1,
    "baseline": )json" + ct + R"json(,
    "recipe": {
      "activation": {"clamp": [-128, 127], "format": "int8", "group_size": 128,
                     "rotation": "H32 then BF16 rounding", "rounding": "nearest ties-to-even",
                     "scale": "max(amax/127,1e-30)", "scale_dtype": "float32"},
      "fallback": "existing: byte-exact INT4 g64, BF16 activations, no H32",
      "format": "dgpp_mixed346_h32_a8_g128_v1", "group_size": 128,
      "layer_expert_recipes": {
)json" + recipes + R"json(
      },
      "packing": "I32 [N,K*bits/32], unsigned words, low bits first, codes cross word boundaries",
      "projection_order": ["gate_proj", "up_proj", "down_proj"],
      "recipe_counts": {)json" + counts + R"json(},
      "rotation_size": 32,
      "scale_limits": {"3": 169.33333333333334, "4": 127, "6": 31},
      "version": 1,
      "weight_codebooks": {"3": [-127.0, -79.0, -45.0, -14.0, 14.0, 45.0, 79.0, 127.0],
                           "4": [-127, -88, -67, -50, -36, -23, -12, 0, 10, 20, 31, 43, 56, 71, 92, 127],
                           "6": [)json" + six + R"json(]}
    }
  }
})json";
  return s.c_str();
}

inline GlmDsaTextConfig tiny_config(int layers = 5, int dense = 1, bool mtp = true, bool nf4i8 = false,
                                    int attention_bits = 8, bool m346 = false) {
  const std::string text = tiny_config_json(layers, dense, mtp, 16, 2, nf4i8, 0, attention_bits, m346);
  const auto t = dgpp::minijson::parse(text);
  return GlmDsaTextConfig::parse(t.root);
}

inline bool has(const std::string& name, const char* needle) {
  return name.find(needle) != std::string::npos;
}

// The bytes of one tensor. `table` resolves a shape record's packed
// sibling (the record holds [N, K]).
inline std::vector<uint8_t> tensor_bytes(const GlmDsaTextConfig& cfg, const GlmDsaExpectedTensor& e,
                                         const std::vector<GlmDsaExpectedTensor>& table) {
  (void)cfg;
  std::vector<uint8_t> out(e.nbytes());
  const std::string& name = e.name;
  Rng rng(seed_for(name));
  const size_t n = e.numel();
  const bool is_norm = has(name, "norm") && e.shape.size() == 1;   // gains near 1
  const bool is_bias = has(name, ".bias");
  const bool is_router_bias = has(name, "e_score_correction_bias");
  if (e.role == GlmDsaTensorRole::IntShape) {
    const std::string base = name.substr(0, name.size() - std::strlen(".weight_shape"));
    const std::string sibling = base + dgpp::glm_dsa_packed_words_suffix(e.scale_fmt);
    for (const auto& t : table)
      if (t.name == sibling) {
        const int64_t rec[2] = {t.shape[0], t.shape[1] * 32 / t.bits};
        std::memcpy(out.data(), rec, 16);
        return out;
      }
    throw std::runtime_error("fixture: packed sibling missing for " + name);
  }
  for (size_t i = 0; i < n; ++i) {
    float v;
    switch (e.role) {
      case GlmDsaTensorRole::IntPacked: {
        const uint32_t word = static_cast<uint32_t>(rng.next());
        std::memcpy(&out[i * 4], &word, 4);
        continue;
      }
      case GlmDsaTensorRole::IntScale:
        // Release-like group scales: ~4e-3 for int4 codes, ~3e-4 for int8
        // (the checkpoint's expert and attention magnitudes); the codebook's
        // levels reach 127, so its scales sit ~16x lower for the same weights.
        // A Mixed346 triple's levels reach 127 at 3 and 4 bits, 31 at 6.
        v = (dgpp::packed_mixed346(e.scale_fmt) ? (e.bits == 6 ? 0.001f : 0.00025f)
             : e.bits == 4 ? (dgpp::packed_codebook(e.scale_fmt) ? 0.00025f : 0.004f) : 0.0003f) *
            std::exp2(rng.unit() * 1.0f);
        break;
      case GlmDsaTensorRole::Bf16Expert:
        v = 0.05f * rng.normal3();
        break;
      case GlmDsaTensorRole::Plain:
      default:
        if (is_router_bias) v = 0.02f * rng.normal3();
        else if (is_norm) v = 0.9f + 0.2f * (0.5f * (rng.unit() + 1.0f));
        else if (is_bias) v = 0.02f * rng.normal3();
        else if (e.cls == GlmDsaWeightClass::Embed || e.cls == GlmDsaWeightClass::LmHead) v = 0.3f * rng.normal3();
        else v = 0.05f * rng.normal3();  // projections, routers, indexers
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

inline std::vector<uint8_t> fixture_bytes(const GlmDsaTextConfig& cfg, const GlmDsaExpectedTensor& e,
                                          const std::vector<GlmDsaExpectedTensor>& table) {
  return tensor_bytes(cfg, e, table);
}

// Writes `dir` (config.json + one safetensors shard) for `cfg`.
inline void write_fixture(const GlmDsaTextConfig& cfg, const std::string& dir, const char* config_json) {
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
  const auto table = dgpp::glm_dsa_expected_text_tensors(cfg);
  std::string header = "{";
  std::vector<uint8_t> data;
  size_t off = 0;
  bool first = true;
  for (const auto& e : table) {
    const auto b = fixture_bytes(cfg, e, table);
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
  std::printf("glm_dsa fixture written: %zu tensors, %.2f MB payload\n", table.size(),
              static_cast<double>(data.size()) / 1048576.0);
}

inline void write_fixture(const GlmDsaTextConfig& cfg, const std::string& dir, bool nf4i8 = false,
                          int attention_bits = 8, bool m346 = false) {
  write_fixture(cfg, dir, tiny_config_json(5, 1, true, 16, 2, nf4i8, 0, attention_bits, m346));
}

}  // namespace glmdsafx
