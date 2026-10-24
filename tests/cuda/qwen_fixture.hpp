#pragma once
// Synthetic mini-checkpoint writer for the Qwen3.8-Flash-Next tests (Q2,
// 2026-09-09): enumerates the binding table for a small config and writes
// config.json + one safetensors shard, so fixture and table cannot
// disagree. Values are deterministic per tensor NAME (glm_rng's scheme), in
// magnitudes that keep the forward's nonlinearities informative; the
// n-gram hash buffers are the config's derivation (the loader refuses
// anything else) and the table shards follow the checkpoint's row split.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "glm_rng.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen/binding.hpp"
#include "models/qwen/config.hpp"

#include "qwen_csf_fixture.hpp"

namespace qwenfx {

namespace fs = std::filesystem;
using dgpp::float_to_bf16_bits;
using dgpp::float_to_fp8_e4m3_bits;
using dgpp::QwenExpectedTensor;
using dgpp::QwenTensorRole;
using dgpp::QwenTextConfig;
using dgpp::QwenWeightClass;
using glmrng::Rng;
using glmrng::seed_for;

// The tiny release: every pinned dimension (128-wide GDN and indexer heads,
// 256-wide attention heads) at the smallest legal head counts, worlds 1, 2
// and 4 all legal (the kv heads pair up at 4), one PLE layer at index 1, a
// draft layer, 8 experts of 64 (a 16-wide slice at world 4).
inline const char* tiny_text_json() {
  return R"json({
  "model_type": "qwen4_exp_text", "attention_bias": false,
  "bos_token_id": 1, "eos_token_id": 1,
  "hc_count": 4, "hc_lowrank": 32, "head_dim": 256, "heads_per_ngram": 8,
  "hidden_act": "silu", "hidden_size": 256,
  "indexer_budget": 64, "indexer_compress_ratio": 4, "indexer_head_dim": 128,
  "indexer_kv_heads": 1, "indexer_n_heads": 2,
  "layer_types": ["linear_attention", "linear_attention", "full_attention", "linear_attention"],
  "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 4,
  "linear_num_value_heads": 12, "linear_value_head_dim": 128,
  "make_ngram_vocab_size_divisible_by": 128, "mamba_ssm_dtype": "float32",
  "max_position_embeddings": 4096, "moe_intermediate_size": 64,
  "mtp": {"hybrid": true, "layer_types": ["full_attention"], "num_hidden_layers": 1},
  "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false,
  "ngram_size": 3, "ngram_vocab_size_base": 64, "num_attention_heads": 4,
  "num_experts": 8, "num_experts_per_tok": 2, "num_hidden_layers": 4,
  "num_key_value_heads": 2, "output_gate_type": "sigmoid",
  "ple_conv_kernel_size": 4, "ple_embed_dim": 256, "ple_layer_ids": [2],
  "rms_norm_eps": 1e-06,
  "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10],
                      "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
  "shared_expert_intermediate_size": 64, "split_ngram_parts": 4,
  "tie_word_embeddings": false, "vocab_size": 512
})json";
}

inline const char* tiny_quant_json() {
  return R"json({"quant_method": "fp8", "activation_scheme": "dynamic",
  "weight_block_size": [128, 128], "modules_to_not_convert": ["lm_head"],
  "modules_to_convert": ["ple.ple_embedding.ngram_embedding"]})json";
}

inline const char* tiny_nvfp4_quant_json() {
  return R"json({"config_groups":{"group_0":{"weights":{
    "num_bits":4,"type":"float","group_size":16}}}})json";
}

// The AutoRound hybrid's tiny twin (docs/qwen38_autoround_int4_plan.md):
// experts and the shared expert of 256 (whole GPTQ groups of 128 on the
// down projection's K at worlds 1 and 2; world 4 refused), the gptq block
// with the hybrid's dynamic rules.
inline const char* tiny_gptq_text_json() {
  static const std::string s = [] {
    std::string t = tiny_text_json();
    for (const char* key : {"\"moe_intermediate_size\": 64", "\"shared_expert_intermediate_size\": 64"}) {
      const std::string from = key;
      const size_t at = t.find(from);
      if (at == std::string::npos) throw std::runtime_error("fixture: intermediate size anchor");
      t.replace(at, from.size(), from.substr(0, from.size() - 2) + "256");
    }
    return t;
  }();
  return s.c_str();
}

inline const char* tiny_gptq_quant_json() {
  return R"json({"quant_method": "gptq", "bits": 4, "group_size": 128, "desc_act": false,
  "sym": true, "lm_head": true,
  "dynamic": {"+:.*lm_head$": {"bits": 8}, "-:.*linear_attn.*": {}, "-:.*self_attn.*": {},
              "-:.*hyper_connection.*": {}, "-:.*visual.*": {}, "-:.*shared_expert.*": {},
              "-:.*\\.ple\\..*": {}, "-:.*embed.*": {}, "-:.*fc_hidden.*": {}, "-:.*\\.gate$": {}}})json";
}

// The QAD export's tiny twin (local-inference-lab/Qwen3.8-Flash-Next-NVFP4,
// issue #113): a shared expert of 128 (whole 32-wide MXFP8 groups on the
// down projection's K at worlds 1, 2 and 4), and modelopt's MIXED_PRECISION
// block naming every quantized module — NVFP4 routed experts, MXFP8 dense
// stack, the NVFP4 n-gram table on the PLE layer, weight-only NVFP4 draft
// experts. Everything else (norms F32, the rest BF16) follows from the
// config's forms through the binding table.
inline const char* tiny_qad_text_json() {
  static const std::string s = [] {
    std::string t = tiny_text_json();
    const std::string from = "\"shared_expert_intermediate_size\": 64";
    const size_t at = t.find(from);
    if (at == std::string::npos) throw std::runtime_error("fixture: shared intermediate anchor");
    t.replace(at, from.size(), "\"shared_expert_intermediate_size\": 128");
    return t;
  }();
  return s.c_str();
}

inline const char* tiny_qad_quant_json() {
  static const std::string s = [] {
    // The layer kinds of tiny_text_json's layer_types.
    const bool gdn[4] = {true, true, false, true};
    std::string q = "{\"quant_method\": \"modelopt\", \"quant_algo\": \"MIXED_PRECISION\", "
                    "\"kv_cache_quant_algo\": null, \"ignore\": [\"lm_head\", \"model.visual.*\"], "
                    "\"quantized_layers\": {";
    bool first = true;
    auto add = [&](const std::string& key, const char* algo, int group) {
      if (!first) q += ", ";
      first = false;
      q += "\"" + key + "\": {\"quant_algo\": \"" + algo + "\", \"group_size\": " + std::to_string(group) + "}";
    };
    for (int l = 0; l < 4; ++l) {
      const std::string p = "model.language_model.layers." + std::to_string(l) + ".";
      add(p + "mlp.experts", "NVFP4", 16);
      if (gdn[l])
        for (const char* m : {"linear_attn.in_proj_a", "linear_attn.in_proj_b", "linear_attn.in_proj_qkv",
                              "linear_attn.in_proj_z", "linear_attn.out_proj"})
          add(p + m, "MXFP8", 32);
      else
        for (const char* m : {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                              "self_attn.indexer.index_qk_proj"})
          add(p + m, "MXFP8", 32);
      for (const char* m : {"mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj"})
        add(p + m, "MXFP8", 32);
    }
    add("model.language_model.layers.1.ple.ple_embedding.ngram_embedding", "NVFP4", 16);
    add("mtp.layers.0.mlp.experts", "W4A16_NVFP4", 16);
    q += "}}";
    return q;
  }();
  return s.c_str();
}

inline const char* tiny_fp4local_quant_json() {
  static const std::string json = [] {
    std::string q = tiny_qad_quant_json();
    const std::string from = ".mlp.experts\": {\"quant_algo\": \"NVFP4\", \"group_size\": 16}";
    size_t pos = 0;
    while ((pos = q.find(from, pos)) != std::string::npos) {
      const std::string to = from.substr(0, from.size() - 1) + ", \"weight_scale_encoding\": \"csf\"}";
      q.replace(pos, from.size(), to);
      pos += to.size();
    }
    return q;
  }();
  return json.c_str();
}

inline QwenTextConfig tiny_fp4local_config() {
  const auto t = dgpp::minijson::parse(tiny_qad_text_json());
  const auto q = dgpp::minijson::parse(tiny_fp4local_quant_json());
  return QwenTextConfig::parse(t.root, &q.root);
}

inline QwenTextConfig tiny_config() {
  const auto t = dgpp::minijson::parse(tiny_text_json());
  const auto q = dgpp::minijson::parse(tiny_quant_json());
  return QwenTextConfig::parse(t.root, &q.root);
}

inline QwenTextConfig tiny_nvfp4_config() {
  const auto t = dgpp::minijson::parse(tiny_text_json());
  const auto q = dgpp::minijson::parse(tiny_nvfp4_quant_json());
  return QwenTextConfig::parse(t.root, &q.root);
}

inline QwenTextConfig tiny_gptq_config() {
  const auto t = dgpp::minijson::parse(tiny_gptq_text_json());
  const auto q = dgpp::minijson::parse(tiny_gptq_quant_json());
  return QwenTextConfig::parse(t.root, &q.root);
}

inline QwenTextConfig tiny_qad_config() {
  const auto t = dgpp::minijson::parse(tiny_qad_text_json());
  const auto q = dgpp::minijson::parse(tiny_qad_quant_json());
  return QwenTextConfig::parse(t.root, &q.root);
}

inline bool has(const std::string& name, const char* needle) {
  return name.find(needle) != std::string::npos;
}

inline std::vector<uint8_t> tensor_bytes(const QwenTextConfig& cfg, const QwenExpectedTensor& e) {
  std::vector<uint8_t> out(e.nbytes());
  const std::string& name = e.name;
  if (e.dtype == dgpp::DType::I64) {
    // The hash buffers: the config's derivation, verbatim.
    const dgpp::QwenNgramGeometry g = cfg.ngram_geometry();
    const std::vector<int64_t>* src = nullptr;
    if (has(name, "layer_multipliers")) src = &g.multipliers;
    else if (has(name, "ngram_heads_offsets")) src = &g.head_offset;
    else if (has(name, "ngram_heads_vocab_sizes")) src = &g.head_vocab;
    else throw std::runtime_error("fixture: unknown I64 tensor " + name);
    if (src->size() != e.numel()) throw std::runtime_error("fixture: I64 length " + name);
    std::memcpy(out.data(), src->data(), out.size());
    return out;
  }
  Rng rng(seed_for(name));
  const size_t n = e.numel();
  // The GPTQ triple: random codes (every nibble / byte legal), the
  // symmetric zero constant, F16 scales in the release's range.
  if (e.role == QwenTensorRole::GptqCodes || e.role == QwenTensorRole::GptqZeros) {
    const int bits = e.cls == QwenWeightClass::LmHead ? 8 : 4;
    const uint32_t zero_word = bits == 4 ? 0x77777777u : 0x7F7F7F7Fu;
    for (size_t i = 0; i < n; ++i) {
      const uint32_t w = e.role == QwenTensorRole::GptqZeros ? zero_word
                                                             : static_cast<uint32_t>(rng.next());
      std::memcpy(&out[i * 4], &w, 4);
    }
    return out;
  }
  if (e.role == QwenTensorRole::GptqScales) {
    // The expert scale base sits where the tiny model's routing has no
    // near-tie against the pure-Python reference in the forward parity AND
    // none across the decode gates' transcripts (a factor sweep on
    // 2026-09-28: 0.004 flips two slots at layer 1, 0.0034 one decode step;
    // 0.00265 flips none in either, every layer within 5 ulps, its
    // neighbours 0.0024 and 0.003 clean too).
    const float lo = e.cls == QwenWeightClass::LmHead ? 0.001f : 0.00265f;
    for (size_t i = 0; i < n; ++i) {
      const float v = lo * (1.0f + 0.75f * (0.5f * (rng.unit() + 1.0f)));
      const uint16_t h = dgpp::float_to_fp16_bits(v);
      std::memcpy(&out[i * 2], &h, 2);
    }
    return out;
  }
  const bool is_norm = has(name, "norm") && e.shape.size() == 1;  // the (1+w) norms: w near 0
  const bool is_gdn_norm = has(name, "linear_attn.norm.weight");   // plain w near 1
  const bool is_scale_inv = has(name, "_scale_inv");
  const bool is_table_scale = has(name, "ngram_embedding.weight_scale");
  const bool is_a_log = has(name, "A_log");
  const bool is_dt_bias = has(name, "dt_bias");
  const bool is_conv = has(name, "conv1d");
  const bool is_inject = has(name, "block_inject_weight");
  for (size_t i = 0; i < n; ++i) {
    float v;
    if (e.role == QwenTensorRole::Mxfp8Scale) {
      // e8m0 scales on 2^-6 .. 2^-4 (three values, so the per-group indexing
      // is exercised) under e4m3 codes of ~1.6 x N(0, 1): weights of the
      // BF16 fixture's 0.05 x N(0, 1) magnitude, every value exact in bf16.
      out[i] = static_cast<uint8_t>(121 + (rng.next() % 3));
      continue;
    } else if (e.role == QwenTensorRole::Fp4Payload || e.dtype == dgpp::DType::U8) {
      // e2m1 code pairs (the routed experts, the NVFP4 table's shards): every byte legal.
      out[i] = static_cast<uint8_t>(rng.next() & 0xffu);
      continue;
    } else if (e.role == QwenTensorRole::NgramShardScale ||
               (e.role == QwenTensorRole::Fp4Scale && cfg.dense_mxfp8_shipped))
      // The QAD twin's e4m3 block scales: non-negative, as modelopt writes
      // them (the NVIDIA twin keeps its signed draw, which the registered
      // gates were tuned on).
      v = std::fabs(rng.normal3());
    else if (e.role == QwenTensorRole::Mxfp8Payload)
      v = 1.6f * rng.normal3();                            // e4m3 codes under the 2^-5-ish scales above
    else if (e.role == QwenTensorRole::Fp8Payload || e.role == QwenTensorRole::NgramShard ||
             e.role == QwenTensorRole::Fp4Scale)
      v = rng.normal3();                                   // e4m3 codes (scaled at dequant)
    else if (e.role == QwenTensorRole::Fp4Global)
      // The QAD twin: 2^-5, expert weights of the FP8 fixture's magnitude and
      // an exact reciprocal (the parity budget's near-tie margin); the
      // NVIDIA twin keeps the 2.0 its gates were tuned on.
      v = cfg.dense_mxfp8_shipped ? 0.03125f : 2.0f;
    else if (e.role == QwenTensorRole::InputScale) v = 1.0f;
    else if (is_scale_inv) v = 0.01f + 0.06f * (0.5f * (rng.unit() + 1.0f));
    else if (is_table_scale) v = 0.02f;
    else if (is_gdn_norm) v = 0.8f + 0.4f * (0.5f * (rng.unit() + 1.0f));
    else if (is_norm) v = 0.2f * rng.normal3();
    else if (is_a_log) v = 0.3f * rng.normal3();
    else if (is_dt_bias) v = 0.3f * rng.normal3();
    else if (is_conv) v = 0.15f * rng.normal3();
    else if (is_inject) v = 0.05f * rng.normal3();
    else if (e.cls == QwenWeightClass::Embed || e.cls == QwenWeightClass::LmHead) v = 0.3f * rng.normal3();
    else v = 0.05f * rng.normal3();                        // projections, routers, gates
    if (e.dtype == dgpp::DType::BF16) {
      const uint16_t bits = float_to_bf16_bits(v);
      std::memcpy(&out[i * 2], &bits, 2);
    } else if (e.dtype == dgpp::DType::F8_E4M3) {
      out[i] = float_to_fp8_e4m3_bits(v);
    } else if (e.dtype == dgpp::DType::F32) {
      std::memcpy(&out[i * 4], &v, 4);
    } else {
      throw std::runtime_error("fixture dtype not handled: " + name);
    }
  }
  return out;
}

inline void write_fixture(const QwenTextConfig& cfg, const std::string& dir,
                          const char* text_json, const char* quant_json,
                          const std::vector<QwenExpectedTensor>& extra_tensors);

// Writes `dir` for `cfg` with the json the config's forms came from: the
// FP8 release's (the default), the NVFP4 release's, or the AutoRound hybrid's.
inline void write_fixture_for(const QwenTextConfig& cfg, const std::string& dir) {
  if (cfg.experts_gptq_int4) write_fixture(cfg, dir, tiny_gptq_text_json(), tiny_gptq_quant_json(), {});
  else if (cfg.experts_nvfp4_csf) write_fixture(cfg, dir, tiny_qad_text_json(), tiny_fp4local_quant_json(), {});
  else if (cfg.dense_mxfp8_shipped) write_fixture(cfg, dir, tiny_qad_text_json(), tiny_qad_quant_json(), {});
  else if (cfg.experts_nvfp4) write_fixture(cfg, dir, tiny_text_json(), tiny_nvfp4_quant_json(), {});
  else write_fixture(cfg, dir, tiny_text_json(), tiny_quant_json(), {});
}

// The engine test's fixture config: DGPP_TEST_QWEN_GPTQ=1 selects the
// AutoRound hybrid (its dense stack ships as fp8: the test's main puts the
// loader in the fp8 dense mode for the process). DGPP_TEST_QWEN_QAD=1
// selects the QAD export's twin (MXFP8 dense, NVFP4 table, F32 norms,
// weight-only draft experts). DGPP_TEST_QWEN_NVFP4=1 selects NVFP4 experts,
// independently of the test's dense-weight mode.
inline QwenTextConfig test_config() {
  const char* g = std::getenv("DGPP_TEST_QWEN_GPTQ");
  if (g != nullptr && g[0] == '1') return tiny_gptq_config();
  const char* c = std::getenv("DGPP_TEST_QWEN_FP4LOCAL");
  if (c != nullptr && c[0] == '1') return tiny_fp4local_config();
  const char* q = std::getenv("DGPP_TEST_QWEN_QAD");
  if (q != nullptr && q[0] == '1') return tiny_qad_config();
  const char* n = std::getenv("DGPP_TEST_QWEN_NVFP4");
  if (n != nullptr && n[0] == '1') return tiny_nvfp4_config();
  return tiny_config();
}

// Writes `dir` (config.json + one safetensors shard) for `cfg`.
inline void write_fixture(const QwenTextConfig& cfg, const std::string& dir,
                          const char* text_json = tiny_text_json(),
                          const char* quant_json = tiny_quant_json(),
                          const std::vector<QwenExpectedTensor>& extra_tensors = {}) {
  fs::path root(dir);
  fs::remove_all(root);
  fs::create_directories(root);
  {
    const fs::path p = root / "config.json";
    std::FILE* f = std::fopen(p.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write config.json");
    const std::string json = std::string("{\"architectures\":[\"Qwen4ExpForConditionalGeneration\"],"
                                         "\"model_type\":\"qwen4_exp\",\"text_config\":") +
                             text_json + ",\"quantization_config\":" + quant_json + "}";
    std::fwrite(json.data(), 1, json.size(), f);
    std::fclose(f);
  }
  auto plain_cfg = cfg;
  plain_cfg.experts_nvfp4_csf = plain_cfg.mtp_experts_nvfp4_csf = false;
  auto table = dgpp::qwen_expected_text_tensors(plain_cfg);
  table.insert(table.end(), extra_tensors.begin(), extra_tensors.end());
  std::string header = "{";
  std::vector<uint8_t> data;
  size_t off = 0;
  std::string index = "{\"weight_map\":{";
  const bool nested = cfg.experts_nvfp4_csf || cfg.mtp_experts_nvfp4_csf;
  auto append = [&](const QwenExpectedTensor& e, const std::vector<uint8_t>& b) {
    std::string shape = "[";
    for (size_t i = 0; i < e.shape.size(); ++i) {
      if (i) shape += ",";
      shape += std::to_string(e.shape[i]);
    }
    shape += "]";
    if (header.size() > 1) { header += ","; index += ","; }
    index += "\"" + e.name + "\":\"tensors/model.safetensors\"";
    header += "\"" + e.name + "\":{\"dtype\":\"" + std::string(dgpp::dtype_name(e.dtype)) +
              "\",\"shape\":" + shape + ",\"data_offsets\":[" + std::to_string(off) + "," +
              std::to_string(off + b.size()) + "]}";
    data.insert(data.end(), b.begin(), b.end());
    off += b.size();
  };
  for (const auto& e : table) {
    const auto b = tensor_bytes(plain_cfg, e);
    if (e.role == QwenTensorRole::Fp4Scale &&
        (e.layer == cfg.mtp_layer() ? cfg.mtp_experts_nvfp4_csf : cfg.experts_nvfp4_csf)) {
      std::vector<uint8_t> fixed;
      std::vector<uint32_t> exceptions;
      encode_csf(b, e.shape[0], e.shape[1], fixed, exceptions);
      auto side = e;
      side.name += ".nvfp4_csf_fixed";
      side.dtype = dgpp::DType::U8;
      side.shape = {e.shape[0] / 16, 16 * (1 + e.shape[1] / 2)};
      append(side, fixed);
      side.name = e.name + ".nvfp4_csf_exceptions";
      side.dtype = dgpp::DType::U32;
      side.shape = {static_cast<int64_t>(exceptions.size())};
      std::vector<uint8_t> bytes(exceptions.size() * 4);
      if (!bytes.empty()) std::memcpy(bytes.data(), exceptions.data(), bytes.size());
      append(side, bytes);
    } else append(e, b);
  }
  header += "}";
  if (nested) {
    fs::create_directories(root / "tensors");
    index += "}}";
    std::FILE* f = std::fopen((root / "model.safetensors.index.json").c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write index");
    std::fwrite(index.data(), 1, index.size(), f);
    std::fclose(f);
  }
  const fs::path shard = root / (nested ? "tensors/model.safetensors" : "model.safetensors");
  std::FILE* f = std::fopen(shard.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write shard");
  const uint64_t hlen = header.size();
  std::fwrite(&hlen, 8, 1, f);
  std::fwrite(header.data(), 1, hlen, f);
  std::fwrite(data.data(), 1, data.size(), f);
  std::fclose(f);
  std::printf("qwen fixture written: %zu tensors, %.2f MB payload\n", table.size(),
              static_cast<double>(data.size()) / 1048576.0);
}

}  // namespace qwenfx
