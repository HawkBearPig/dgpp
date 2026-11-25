#pragma once
// Synthetic mini-checkpoint writer for the Inkling 2 tests: enumerates the
// binding tables for a small config and writes config.json,
// hf_quant_config.json, the main safetensors file and its index in the
// release's own layout, so fixture and table cannot disagree. Values are
// deterministic per tensor NAME, in magnitudes that keep the forward's
// nonlinearities informative: query and key projections whose products
// decide the attention, clip scales that push some heads past their
// thresholds, mix queries that make the mixes selective, controllers whose
// gates leave the linear range, routers with clear top-k margins, NVFP4
// experts with random E2M1 nibbles and MXFP8 experts with random E4M3 codes
// at about 0.05 rms.
#include <algorithm>
#include <cmath>
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
#include "models/inkling2/binding.hpp"
#include "models/inkling2/config.hpp"

namespace ink2fx {

namespace fs = std::filesystem;
using dgpp::float_to_bf16_bits;
using dgpp::Inkling2Config;
using dgpp::Inkling2ExpectedTensor;
using dgpp::Inkling2TensorRole;
using dgpp::Inkling2WeightClass;
using glmrng::Rng;
using glmrng::seed_for;

// The tiny release: 128-wide heads at 12 query / 4 kv heads (three query
// heads per kv head as the release, worlds 1, 2 and 4 legal), hidden 256,
// eight layers in blocks of two with the last layer of each block
// full-attention, layers [2, 6) looped three times (16 visits, two blocks in
// the core as the release), 8 experts of 128 with top 2 (a 32-wide slice at
// world 4), NVFP4 experts on layers 0 to 5 and MXFP8 on 6 and 7, a 40-token
// window that is not a tile multiple, a 24-token relative extent on the
// full-attention layers and log scaling from position 16, so a 72-token
// prompt looks past all three.
inline const char* tiny_config_json() {
  static const char* s = R"json({
  "architectures": ["Inkling2LoopForConditionalGeneration"], "model_type": "inkling2_loop",
  "dtype": "bfloat16", "eos_token_id": 1,
  "text_config": {
    "model_type": "inkling2_loop_text", "model_max_length": 4096,
    "hidden_size": 256, "num_hidden_layers": 8, "vocab_size": 512, "unpadded_vocab_size": 500,
    "num_attention_heads": 12, "num_key_value_heads": 4, "head_dim": 128,
    "swa_num_attention_heads": 12, "swa_num_key_value_heads": 4, "swa_head_dim": 128,
    "d_rel": 16, "rel_extent": 24, "sliding_window_size": 40, "q_bias": false, "o_bias": false,
    "log_scaling_n_floor": 16, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-06,
    "use_value_norm_no_gamma": true, "qk_l2_clip_threshold": 22.627416997969522, "rel_l2_clip_threshold": 8.0,
    "local_layer_ids": [0, 2, 4, 6], "sconv_kernel_size": 4, "final_logit_softcapping": 30.0,
    "n_layers_per_attn_res_block": 2, "attn_res_normalize_values": true,
    "n_routed_experts": 8, "num_experts_per_tok": 2, "n_shared_experts": 0, "intermediate_size": 128,
    "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 8.0, "use_gate_bias": true,
    "loop_config": {"core_start": 2, "core_end": 6, "num_passes": 3, "gate_rank": 16, "is_varloop": false}
  },
  "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}
})json";
  return s;
}

// hf_quant_config.json for a config whose layers at or after `first_mxfp8`
// carry MXFP8 experts.
inline std::string quant_json(int layers, int first_mxfp8) {
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

inline std::string tiny_quant_json() { return quant_json(8, 6); }

inline Inkling2Config tiny_config() {
  const auto c = dgpp::minijson::parse(tiny_config_json());
  const std::string q = tiny_quant_json();
  const auto qp = dgpp::minijson::parse(q);
  return Inkling2Config::parse(c.root, qp.root);
}

inline bool has(const std::string& name, const char* needle) { return name.find(needle) != std::string::npos; }

inline std::vector<uint8_t> tensor_bytes(const Inkling2ExpectedTensor& e) {
  std::vector<uint8_t> out(e.nbytes());
  const std::string& name = e.name;
  Rng rng(seed_for(name));
  const size_t n = e.numel();
  for (size_t i = 0; i < n; ++i) {
    float v;
    switch (e.role) {
      case Inkling2TensorRole::Fp4Payload:
        out[i] = static_cast<uint8_t>(rng.next() & 0xFFu);  // two random e2m1 codes
        continue;
      case Inkling2TensorRole::Fp4Scale:
        // E4M3 scales of 0.5 to 2 under a per-expert factor near 20 / 1536:
        // an e2m1 rms of about 2.9 lands the experts at about 0.04.
        out[i] = dgpp::float_to_fp8_e4m3_bits(static_cast<float>(std::exp2(rng.unit())));
        continue;
      case Inkling2TensorRole::Fp4Amax:
        v = 20.0f * (1.0f + 0.25f * rng.unit());
        break;
      case Inkling2TensorRole::Fp4Denominator:
        v = 1536.0f;
        break;
      case Inkling2TensorRole::Mx8Payload: {
        // A random E4M3 code, never a NaN (0x7F / 0xFF).
        uint8_t b = static_cast<uint8_t>(rng.next() & 0xFFu);
        if ((b & 0x7Fu) == 0x7Fu) b = static_cast<uint8_t>(b & 0xF0u);
        out[i] = b;
        continue;
      }
      case Inkling2TensorRole::Mx8Scale:
        // Random E4M3 codes have an rms near 100: 2^-11 puts them at 0.05.
        out[i] = static_cast<uint8_t>(115 + (rng.next() % 3));
        continue;
      case Inkling2TensorRole::Plain:
      default:
        switch (e.cls) {
          case Inkling2WeightClass::Embed:
          case Inkling2WeightClass::Unembed:
            v = 0.3f * rng.normal3();
            break;
          case Inkling2WeightClass::GlobalNorm:
          case Inkling2WeightClass::LayerNorm:
            v = 0.9f + 0.2f * (0.5f * (rng.unit() + 1.0f));
            break;
          case Inkling2WeightClass::MixQuery:
            v = 4.0f * rng.normal3();
            break;
          case Inkling2WeightClass::HiddenConv:
            v = 0.6f * rng.normal3();
            break;
          case Inkling2WeightClass::AttentionAux:
            if (has(name, "l2_clip.scale")) v = 0.5f + 3.5f * (0.5f * (rng.unit() + 1.0f));  // some heads clip
            else if (has(name, "rel_logits_proj")) v = 0.8f * rng.normal3();
            else if (has(name, "weight_proj")) v = 0.05f * rng.normal3();
            else v = 0.6f * rng.normal3();  // the K / V convolution taps
            break;
          case Inkling2WeightClass::Router:
            // Logits of about 2.7 rms spread the sigmoid scores over (0, 1):
            // the top-k margins sit well above a rounding difference, so two
            // launch shapes of one row route alike.
            v = (e.dtype == dgpp::DType::F32 ? 0.2f : 0.5f) * rng.normal3();
            break;
          case Inkling2WeightClass::Controller:
            v = (has(name, "down_weight") ? 0.1f : 0.4f) * rng.normal3();
            break;
          case Inkling2WeightClass::Draft:
            // The draft model. Mix weights that prefer some taps, per-head
            // norms near 2 (attention logits of about 4 rms, so the block
            // attends selectively), a recurrent head whose gates leave the
            // linear range and whose correction is as large as the base
            // logits, so a wrong step shows in the proposals.
            if (name == "fuse_weight") v = 1.5f * rng.normal3();
            else if (name == "mask_embed") v = rng.normal3();
            else if (has(name, "q_norm") || has(name, "k_norm")) v = 1.5f + 0.5f * (rng.unit() + 1.0f);
            else if (has(name, "norm.weight")) v = 0.9f + 0.2f * (0.5f * (rng.unit() + 1.0f));
            else if (has(name, "markov_w1") || has(name, "hidden_out.weight")) v = 0.3f * rng.normal3();
            else if (has(name, "rnn_proj.bias")) v = 0.2f * rng.normal3();
            else if (has(name, "rnn_proj.weight") || has(name, "hidden_out.bias")) v = 0.1f * rng.normal3();
            else if (has(name, "gate_proj") || has(name, "up_proj")) v = 0.08f * rng.normal3();
            else v = 0.05f * rng.normal3();  // the attention projections and the MLP's down projection
            break;
          default:
            // The projections. Queries and keys at three times the rest:
            // their products reach a logit rms of 0.3 to 2.5 across the
            // heads' clip scales, and the heads with the larger scales clip.
            v = (has(name, "wq_du") || has(name, "wk_dv") ? 0.15f : 0.05f) * rng.normal3();
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

inline void write_text(const fs::path& p, const std::string& text) {
  std::FILE* f = std::fopen(p.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write " + p.string());
  std::fwrite(text.data(), 1, text.size(), f);
  std::fclose(f);
}

inline void write_safetensors(const fs::path& p, const std::vector<Inkling2ExpectedTensor>& table) {
  std::string header = "{";
  std::vector<uint8_t> data;
  size_t off = 0;
  bool first = true;
  for (const auto& e : table) {
    const auto b = tensor_bytes(e);
    std::string shape = "[";
    for (size_t i = 0; i < e.shape.size(); ++i) {
      if (i) shape += ",";
      shape += std::to_string(e.shape[i]);
    }
    shape += "]";
    if (!first) header += ",";
    first = false;
    header += "\"" + e.name + "\":{\"dtype\":\"" + std::string(dgpp::dtype_name(e.dtype)) + "\",\"shape\":" + shape +
              ",\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + b.size()) + "]}";
    data.insert(data.end(), b.begin(), b.end());
    off += b.size();
  }
  header += "}";
  std::FILE* f = std::fopen(p.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write " + p.string());
  const uint64_t hlen = header.size();
  std::fwrite(&hlen, 8, 1, f);
  std::fwrite(header.data(), 1, hlen, f);
  std::fwrite(data.data(), 1, data.size(), f);
  std::fclose(f);
}

// Writes `dir` in the release's layout: config.json, hf_quant_config.json,
// model-00001-of-00001.safetensors and the index that names it.
inline void write_fixture(const Inkling2Config& cfg, const std::string& dir, const char* config_json,
                          const std::string& quant) {
  fs::path root(dir);
  fs::remove_all(root);
  fs::create_directories(root);
  write_text(root / "config.json", config_json);
  write_text(root / "hf_quant_config.json", quant);
  std::vector<Inkling2ExpectedTensor> table = dgpp::inkling2_expected_text_tensors(cfg);
  const auto vision = dgpp::inkling2_expected_vision_tensors(cfg);
  table.insert(table.end(), vision.begin(), vision.end());
  const char* shard = "model-00001-of-00001.safetensors";
  write_safetensors(root / shard, table);
  std::string index = "{\"metadata\": {}, \"weight_map\": {";
  for (size_t i = 0; i < table.size(); ++i) {
    if (i) index += ", ";
    index += "\"" + table[i].name + "\": \"" + shard + "\"";
  }
  index += "}}";
  write_text(root / "model.safetensors.index.json", index);
  std::printf("inkling2 fixture written: %zu tensors\n", table.size());
}

inline void write_fixture(const std::string& dir) {
  write_fixture(tiny_config(), dir, tiny_config_json(), tiny_quant_json());
}

// The same release with the layer kinds swapped: the first layer of every
// block attends over the whole context and the second over the window, so
// visit 0 is a full-attention layer with dynamic key and value taps.
inline std::string full_first_config_json() {
  std::string s = tiny_config_json();
  const std::string from = "\"local_layer_ids\": [0, 2, 4, 6]";
  const size_t at = s.find(from);
  if (at == std::string::npos) throw std::runtime_error("full_first_config_json: anchor not found");
  return s.replace(at, from.size(), "\"local_layer_ids\": [1, 3, 5, 7]");
}

inline void write_full_first_fixture(const std::string& dir) {
  const std::string config = full_first_config_json();
  const std::string quant = tiny_quant_json();
  const auto c = dgpp::minijson::parse(config);
  const auto q = dgpp::minijson::parse(quant);
  write_fixture(Inkling2Config::parse(c.root, q.root), dir, config.c_str(), quant);
}

// The tiny release with a draft model (mtp.safetensors): three layers of 12
// query and 4 key/value heads (worlds 1, 2 and 4 legal), the first and last
// over a 24-position window that a 40-token prompt looks past, a block of
// the anchor row and seven mask rows, a recurrent head of rank 64 (a width the tensor-core GEMV takes), and taps
// at the entry of visits 2, 7 and 12 and on the final-norm output.
// `full_vocabulary` serves all 512 stored rows, for the tests that draw
// their prompts over 512 ids.
inline std::string draft_config_json(bool full_vocabulary = false) {
  std::string s = tiny_config_json();
  if (full_vocabulary) {
    const std::string from = "\"unpadded_vocab_size\": 500";
    const size_t v = s.find(from);
    if (v == std::string::npos) throw std::runtime_error("draft_config_json: vocabulary anchor not found");
    s.replace(v, from.size(), "\"unpadded_vocab_size\": 512");
  }
  const std::string anchor = "\"quantization_config\"";
  const size_t at = s.find(anchor);
  if (at == std::string::npos) throw std::runtime_error("draft_config_json: anchor not found");
  return s.insert(at,
                  "\"mtp_config\": {\"hidden_size\": 256, \"num_hidden_layers\": 3, \"num_attention_heads\": 12, "
                  "\"num_key_value_heads\": 4, \"head_dim\": 128, \"intermediate_size\": 512, "
                  "\"rope_theta\": 1000000.0, \"rms_norm_eps\": 1e-06, \"local_layer_ids\": [0, 2], "
                  "\"sliding_window_size\": 24, \"block_size\": 8, \"markov_rank\": 64, "
                  "\"markov_head_type\": \"rnn_hidden\", \"target_layer_ids\": [2, 7, 12, 16]},\n  ");
}

inline Inkling2Config draft_config(bool full_vocabulary = false) {
  const std::string config = draft_config_json(full_vocabulary);
  const std::string quant = tiny_quant_json();
  const auto c = dgpp::minijson::parse(config);
  const auto q = dgpp::minijson::parse(quant);
  return Inkling2Config::parse(c.root, q.root);
}

// A test run under loop guidance (engine.loop_guidance): DGPP_TEST_INK2_LOOP
// names it as "guidance=1,strength=0.3", either or both. The session,
// engine and draft gates compare the engine with itself, so they hold under
// guidance too, and a run of them with the variable set is how guidance is
// held to them. Unset, a config is returned as it is.
inline dgpp::Inkling2LoopOptions test_loop_options() {
  dgpp::Inkling2LoopOptions o;
  const char* spec = std::getenv("DGPP_TEST_INK2_LOOP");
  if (spec == nullptr || *spec == 0) return o;
  std::string s(spec);
  size_t at = 0;
  while (at < s.size()) {
    const size_t end = std::min(s.find(',', at), s.size());
    const std::string item = s.substr(at, end - at);
    at = end + 1;
    const size_t eq = item.find('=');
    if (eq == std::string::npos) throw std::runtime_error("DGPP_TEST_INK2_LOOP: expected key=value, got " + item);
    const std::string key = item.substr(0, eq), value = item.substr(eq + 1);
    if (key == "guidance") o.guidance = value == "1";
    else if (key == "strength") o.strength = std::stof(value);
    else throw std::runtime_error("DGPP_TEST_INK2_LOOP: unknown item " + item);
  }
  return o;
}

inline Inkling2Config with_test_loop(Inkling2Config cfg) {
  cfg.apply_loop_options(test_loop_options());
  return cfg;
}

inline void write_draft_fixture(const std::string& dir, bool full_vocabulary = false) {
  const Inkling2Config cfg = draft_config(full_vocabulary);
  const std::string config = draft_config_json(full_vocabulary);
  write_fixture(cfg, dir, config.c_str(), tiny_quant_json());
  write_safetensors(fs::path(dir) / "mtp.safetensors", dgpp::inkling2_expected_draft_tensors(cfg));
}

}  // namespace ink2fx
