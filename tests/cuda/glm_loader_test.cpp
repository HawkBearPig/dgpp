// Loader tests for the M4 streaming resident loader (glm_loader.cu). A
// synthetic mini-checkpoint is generated ON DISK from the expected-tensor
// table itself (config.json + one safetensors shard), so fixture and table
// can never disagree. The tests then verify, through the loader's managed
// memory (CPU-readable):
//   * KDA merged in_proj [f_a|g_a|q|k|v|b] and merged conv are byte-exact
//     concatenations of the six separate checkpoint tensors;
//   * DSA fused qkv_a, q_b, o_proj are exact block-scale dequants (bitwise
//     vs the host oracle, including the rounding of the single
//     decode-multiply then BF16 round);
//   * the APE arrives as F32 (converted from checkpoint BF16);
//   * MLP/MoE matrices stay COMPRESSED: payload bytes identical, scale
//     bytes identical, geometry recorded;
//   * globals and MTP head tensors are byte-exact;
//   * layer_bytes() equals actual bump usage for every layer (the
//     no-drift contract), and streaming frees the previous layer.
// The dequant kernel's ragged-tail handling is pinned by a direct
// [1000, 1000] test against the host oracle.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "common/bf16_residency.hpp"
#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/fp8_dequant.hpp"
#include "kernels/latent_format.hpp"
#include "loaders/minijson.hpp"
#include "models/glm/binding.hpp"
#include "models/glm/config.hpp"
#include "models/glm/loader.hpp"
#include "models/glm/resident_image.hpp"
#include "resident_image_fixture.hpp"

namespace {

namespace fs = std::filesystem;
using dgpp::DType;
using dgpp::GlmExpectedTensor;
using dgpp::GlmTextConfig;
using dgpp::GlmTensorDesc;

// Same tiny config as the binding tests, with one ragged edge:
// intermediate_size 1000 exercises non-128-multiple dense-MLP shapes.
const char* kTinyJson = R"json({
  "hidden_size": 512, "vocab_size": 1000, "num_hidden_layers": 6,
  "rms_norm_eps": 1e-6, "tie_word_embeddings": false,
  "hidden_act": "silu", "swiglu_limit": 7.5,
  "layer_types": ["linear_attention", "linear_attention",
                  "deepseek_sparse_attention", "linear_attention",
                  "deepseek_sparse_attention", "linear_attention"],
  "mlp_layer_types": ["dense", "dense", "sparse", "sparse", "sparse", "sparse"],
  "indexer_types": ["full", "full", "full", "full", "full", "full"],
  "first_k_dense_replace": 2,
  "linear_attn_config": {
    "num_heads": 8, "head_dim": 64, "short_conv_kernel_size": 4,
    "gate_lower_bound": -3.5,
    "kda_layers": [0, 1, 3, 5], "full_attn_layers": [2, 4]
  },
  "num_attention_heads": 8, "q_lora_rank": 256, "kv_lora_rank": 128,
  "qk_nope_head_dim": 96, "qk_rope_head_dim": 0, "v_head_dim": 96,
  "mla_use_nope": true,
  "index_n_heads": 4, "index_head_dim": 128, "index_kpool": 4,
  "index_topk": 128, "index_kpool_compress": true,
  "index_kpool_always_select_tail": true, "indexer_rope_interleave": true,
  "intermediate_size": 1000, "moe_intermediate_size": 256,
  "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
  "scoring_func": "sigmoid", "topk_method": "noaux_tc",
  "norm_topk_prob": true, "routed_scaling_factor": 1.5,
  "n_group": 1, "topk_group": 1, "moe_router_dtype": "float32",
  "mhc": true, "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-06,
  "num_nextn_predict_layers": 1
})json";

GlmTextConfig tiny_config() {
  auto parsed = dgpp::minijson::parse(kTinyJson);
  return GlmTextConfig::parse(parsed.root);
}

// The same config with a TP-legal dense width (1024: 512 at world 2), for
// the sharded NVFP4 gate; the 1000-wide original is world-1 only by design
// (the ragged-edge coverage above).
std::string tiny_tp_json() {
  std::string j = kTinyJson;
  const std::string from = "\"intermediate_size\": 1000";
  const size_t at = j.find(from);
  if (at == std::string::npos) throw std::runtime_error("tiny json drifted");
  j.replace(at, from.size(), "\"intermediate_size\": 1024");
  return j;
}

// ---- deterministic fixture values ------------------------------------
// xorshift64 seeded per tensor name; every tensor is reproducible and
// independent of load order.
uint64_t seed_for(const std::string& name) {
  uint64_t h = 1469598103934665603ull;
  for (char c : name) {
    h ^= static_cast<uint8_t>(c);
    h *= 1099511628211ull;
  }
  return h ^ 0x9e3779b97f4a7c15ull;
}

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed | 1) {}
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  float unit() {  // [-1, 1)
    return static_cast<float>(next() >> 11) / static_cast<float>(1ull << 52) -
           1.0f;
  }
};

// NVFP4 triples (the hybrid's routed experts): per matrix a global scale g
// in [0.5, 4) seeded by the base name, e4m3 block scales in (0.01..0.07) x g,
// and e2m1 codes two per byte, low nibble first.
float fp4_global_for(const std::string& base) {
  Rng g(seed_for(base) ^ 0x5bd1e995u);
  return 0.5f + 1.75f * (g.unit() + 1.0f);
}
bool fp4_tensor_bytes(const GlmExpectedTensor& e, std::vector<uint8_t>& out) {
  using dgpp::GlmTensorRole;
  const char* suffix = e.role == GlmTensorRole::Fp4Packed   ? "_packed"
                       : e.role == GlmTensorRole::Fp4Scale  ? "_scale"
                       : e.role == GlmTensorRole::Fp4Global ? "_global_scale"
                                                             : nullptr;
  if (!suffix) return false;
  const std::string base = e.name.substr(0, e.name.size() - std::strlen(suffix));
  const float g = fp4_global_for(base);
  Rng rng(seed_for(e.name));
  out.assign(e.nbytes(), 0);
  if (e.role == GlmTensorRole::Fp4Global) {
    std::memcpy(out.data(), &g, 4);
  } else if (e.role == GlmTensorRole::Fp4Scale) {
    for (auto& b : out)
      b = dgpp::float_to_fp8_e4m3_bits((0.01f + 0.03f * (rng.unit() + 1.0f)) * g);
  } else {
    for (auto& b : out) {
      const uint8_t lo = dgpp::float_to_fp4_e2m1_bits(3.0f * rng.unit());
      const uint8_t hi = dgpp::float_to_fp4_e2m1_bits(3.0f * rng.unit());
      b = static_cast<uint8_t>(lo | (hi << 4));
    }
  }
  return true;
}

std::vector<uint8_t> tensor_bytes(const GlmExpectedTensor& e) {
  std::vector<uint8_t> fp4;
  if (fp4_tensor_bytes(e, fp4)) return fp4;
  Rng rng(seed_for(e.name));
  std::vector<uint8_t> out(e.nbytes());
  const size_t n = e.numel();
  const size_t ds = dgpp::dtype_size(e.dtype);
  for (size_t i = 0; i < n; ++i) {
    if (e.dtype == DType::BF16) {
      const float v = rng.unit();
      uint16_t bits = dgpp::float_to_bf16_bits(v);
      std::memcpy(&out[i * 2], &bits, 2);
    } else if (e.dtype == DType::F32) {
      const float v = 0.25f + 1.5f * (rng.unit() + 1.0f);  // [0.25, 2)
      std::memcpy(&out[i * 4], &v, 4);
    } else if (e.dtype == DType::F8_E4M3) {
      const float v = 2.0f * rng.unit();
      out[i] = dgpp::float_to_fp8_e4m3_bits(v);
    } else {
      throw std::runtime_error("fixture dtype not handled");
    }
  }
  (void)ds;
  return out;
}

// Host dequant oracle: decode-multiply, one BF16 round — the kernel's
// exact operation.
std::vector<uint16_t> dequant_oracle(const std::vector<uint8_t>& payload,
                                     const std::vector<uint8_t>& scales,
                                     int64_t rows, int64_t cols) {
  const int64_t sc = (cols + 127) / 128;
  std::vector<uint16_t> out(rows * cols);
  for (int64_t i = 0; i < rows * cols; ++i) {
    const int64_t n = i / cols, k = i % cols;
    float s;
    std::memcpy(&s, &scales[((n / 128) * sc + k / 128) * 4], 4);
    out[i] = dgpp::float_to_bf16_bits(
        dgpp::fp8_e4m3_bits_to_float(payload[i]) * s);
  }
  return out;
}

// ---- synthetic checkpoint on disk ------------------------------------
struct Fixture {
  fs::path dir;
  GlmTextConfig cfg;
  std::unordered_map<std::string, std::vector<uint8_t>> bytes;

  const std::vector<uint8_t>& at(const std::string& name) const {
    auto it = bytes.find(name);
    if (it == bytes.end())
      throw std::runtime_error("fixture missing tensor: " + name);
    return it->second;
  }
};

// The composed hybrid's quantization_config spelling, for the NVFP4 fixture.
const char* kNvfp4QuantJson = R"json({"quant_method":"dgpp_mixed",
  "routed_experts":{"format":"nvfp4-pack-quantized","num_bits":4,"group_size":16},
  "fp8":{"quant_method":"fp8","fmt":"e4m3","weight_block_size":[128,128]},
  "mtp_layer":{"source":"base"},"dsa_attention":{"source":"base"}})json";

Fixture write_fixture(bool nvfp4 = false, bool tp_legal = false) {
  Fixture fx;
  fx.dir = fs::temp_directory_path() /
           (std::string(nvfp4 ? "dgpp_glm_loader_test_fp4" : "dgpp_glm_loader_test") +
            (tp_legal ? "_tp" : ""));
  fs::remove_all(fx.dir);
  fs::create_directories(fx.dir);
  const std::string text_json = tp_legal ? tiny_tp_json() : std::string(kTinyJson);
  if (!nvfp4 && !tp_legal) {
    fx.cfg = tiny_config();
  } else {
    const auto text = dgpp::minijson::parse(text_json);
    if (nvfp4) {
      const auto q = dgpp::minijson::parse(kNvfp4QuantJson);
      fx.cfg = GlmTextConfig::parse(text.root, &q.root);
    } else {
      fx.cfg = GlmTextConfig::parse(text.root);
    }
  }

  // config.json: root object with text_config (the loader's parse shape),
  // plus the quantization_config under the NVFP4 profile.
  {
    fs::path p = fx.dir / "config.json";
    std::FILE* f = std::fopen(p.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write config.json");
    std::string json = std::string("{\"text_config\":") + text_json;
    if (nvfp4) json += std::string(",\"quantization_config\":") + kNvfp4QuantJson;
    json += "}";
    std::fwrite(json.data(), 1, json.size(), f);
    std::fclose(f);
  }

  // All expected tensors, sequential layout, one shard.
  const auto table = dgpp::glm_expected_text_tensors(fx.cfg);
  std::string header = "{";
  std::vector<uint8_t> data;
  size_t off = 0;
  auto append = [&](const GlmExpectedTensor& e) {
    auto b = tensor_bytes(e);
    std::string shape = "[";
    for (size_t i = 0; i < e.shape.size(); ++i) {
      if (i) shape += ",";
      shape += std::to_string(e.shape[i]);
    }
    shape += "]";
    if (off) header += ",";
    header += "\"" + e.name + "\":{" + "\"dtype\":\"" +
              std::string(dgpp::dtype_name(e.dtype)) +
              "\",\"shape\":" + shape + ",\"data_offsets\":[" +
              std::to_string(off) + "," + std::to_string(off + b.size()) +
              "]}";
    data.insert(data.end(), b.begin(), b.end());
    off += b.size();
    fx.bytes.emplace(e.name, std::move(b));
  };
  for (const auto& e : table) append(e);
  header += "}";

  fs::path shard = fx.dir / "model.safetensors";
  std::FILE* f = std::fopen(shard.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write shard");
  const uint64_t hlen = header.size();
  std::fwrite(&hlen, 8, 1, f);
  std::fwrite(header.data(), 1, hlen, f);
  std::fwrite(data.data(), 1, data.size(), f);
  std::fclose(f);
  return fx;
}

void require(bool cond, const char* what) {
  if (!cond) throw std::runtime_error(what);
}

// The resident pointers are DEVICE addresses (the bump is cudaMalloc'd —
// see GlmLayerBump): fetch before comparing.
std::vector<uint8_t> fetch(const void* dev, size_t bytes) {
  std::vector<uint8_t> h(bytes);
  if (bytes)
    DGPP_CUDA_OK(cudaMemcpy(h.data(), dev, bytes, cudaMemcpyDeviceToHost));
  return h;
}

bool device_bytes_equal(const void* a, const void* b, size_t bytes) {
  return fetch(a, bytes) == fetch(b, bytes);
}

template <typename T>
void require_bytes_eq(const T* got_dev, const std::vector<uint8_t>& want,
                      size_t count, const char* what) {
  const std::vector<uint8_t> got = fetch(got_dev, count * sizeof(T));
  if (std::memcmp(got.data(), want.data(), count * sizeof(T)) != 0) {
    std::printf("mismatch at %s: first differing offset %zu\n", what,
                [&] {
                  for (size_t i = 0; i < count; ++i)
                    if (std::memcmp(got.data() + i * sizeof(T),
                                    want.data() + i * sizeof(T), sizeof(T)))
                      return i;
                  return count;
                }());
    throw std::runtime_error(what);
  }
}

}  // namespace

DGPP_TEST(glm_loader_streams_kda_dense_layer_byte_exact) {
  const Fixture fx = write_fixture();
  dgpp::GlmLayerStream stream(fx.cfg, fx.dir.string());
  const dgpp::GlmLayerResident& r = stream.load_layer(0);
  require(r.kind == dgpp::GlmLayerKind::Kda, "layer 0 is KDA");
  require(r.bytes == dgpp::GlmLayerStream::layer_bytes(fx.cfg, 0),
          "layer bytes match formula");

  const std::string p = "model.language_model.layers.0.self_attn.";
  // Merged in_proj rows [f_a | g_a | q | k | v | b].
  {
    const int64_t head_dim = fx.cfg.kda_head_dim;   // 64
    const int64_t proj = 8 * 64;                    // 512
    const int64_t heads = 8;
    const int64_t hidden = 512;
    const uint16_t* in = static_cast<const uint16_t*>(r.kda.in_proj);
    const struct {
      const char* name;
      int64_t rows;
    } pieces[] = {{"f_a_proj.weight", head_dim},
                  {"g_a_proj.weight", head_dim},
                  {"q_proj.weight", proj},
                  {"k_proj.weight", proj},
                  {"v_proj.weight", proj},
                  {"b_proj.weight", heads}};
    int64_t row = 0;
    for (const auto& piece : pieces) {
      require_bytes_eq(in + row * hidden, fx.at(p + piece.name),
                       static_cast<size_t>(piece.rows * hidden), piece.name);
      row += piece.rows;
    }
  }
  // Merged conv channels [q | k | v].
  {
    const int64_t proj = 512, w = 4;
    const uint16_t* conv = static_cast<const uint16_t*>(r.kda.conv);
    for (int i = 0; i < 3; ++i) {
      static const char* convs[3] = {"q_conv1d.weight", "k_conv1d.weight",
                                     "v_conv1d.weight"};
      require_bytes_eq(conv + i * proj * w, fx.at(p + convs[i]),
                       static_cast<size_t>(proj * w), convs[i]);
    }
  }
  // Direct tensors.
  require_bytes_eq(static_cast<const uint16_t*>(r.kda.f_b),
                   fx.at(p + "f_b_proj.weight"), 512 * 64, "f_b");
  require_bytes_eq(static_cast<const uint16_t*>(r.kda.g_b),
                   fx.at(p + "g_b_proj.weight"), 512 * 64, "g_b");
  require_bytes_eq(r.kda.a_log, fx.at(p + "A_log"), 8, "A_log");
  require_bytes_eq(r.kda.dt_bias, fx.at(p + "dt_bias"), 512, "dt_bias");
  require_bytes_eq(static_cast<const uint16_t*>(r.kda.o_norm),
                   fx.at(p + "o_norm.weight"), 64, "o_norm");
  require_bytes_eq(static_cast<const uint16_t*>(r.kda.o_proj),
                   fx.at(p + "o_proj.weight"), 512 * 512, "o_proj");
  // Dense MLP stays compressed: byte-identical payload and scales.
  static const char* dense[3] = {"gate_proj", "up_proj", "down_proj"};
  for (int i = 0; i < 3; ++i) {
    const std::string name =
        "model.language_model.layers.0.mlp." + std::string(dense[i]) +
        ".weight";
    require_bytes_eq(r.dense[i].payload, fx.at(name), 1000 * 512,
                    dense[i]);
    require_bytes_eq(reinterpret_cast<const uint8_t*>(r.dense[i].scales),
                     fx.at(name + "_scale_inv"), 8 * 4 * 4, "dense scale");
    require(r.dense[i].rows == (i == 2 ? 512 : 1000) &&
                r.dense[i].cols == (i == 2 ? 1000 : 512),
            "dense geometry");
  }
  // Norms + mHC.
  require_bytes_eq(r.ln1, fx.at("model.language_model.layers.0.input_layernorm.weight"),
                   512, "ln1");
  require_bytes_eq(r.ln2,
                   fx.at("model.language_model.layers.0.post_attention_layernorm.weight"),
                   512, "ln2");
  require_bytes_eq(r.mhc.attn_base,
                   fx.at("model.language_model.layers.0.hc_attn_base"), 24,
                   "hc attn base");
  require_bytes_eq(r.mhc.ffn_fn,
                   fx.at("model.language_model.layers.0.hc_ffn_fn"), 24 * 2048,
                   "hc ffn fn");
  // MoE view must be empty on a dense layer.
  require(r.moe.experts.empty() && r.moe.router_gate == nullptr,
          "dense layer has empty moe view");
}

DGPP_TEST(glm_loader_streams_dsa_moe_layer_with_dequant_bridge) {
  const Fixture fx = write_fixture();
  dgpp::GlmLayerStream stream(fx.cfg, fx.dir.string());
  const dgpp::GlmLayerResident& r = stream.load_layer(2);
  require(r.kind == dgpp::GlmLayerKind::Dsa, "layer 2 is DSA");
  require(r.bytes == dgpp::GlmLayerStream::layer_bytes(fx.cfg, 2),
          "layer bytes match formula");

  const std::string p = "model.language_model.layers.2.self_attn.";
  const int64_t hidden = 512, q_lora = 256, kv_lora = 128;

  // The form the loader took: at world 1 every slice is
  // whole, so the resident carries the FP8 pairs as they are — payload and
  // scale grid byte-exact from the fixture — and no bf16 bridge exists.
  // (The bridge, checked below, is the misaligned worlds' form.)
  if (r.dsa.quantized()) {
    const auto pair = [&](const char* tensor, const dgpp::GlmQuantMatrix& q,
                          int64_t rows, int64_t cols, const char* what) {
      require(q.rows == rows && q.cols == cols, what);
      require_bytes_eq(q.payload, fx.at(p + tensor + ".weight"),
                       static_cast<size_t>(rows) * cols, what);
      require_bytes_eq(q.scales, fx.at(p + tensor + ".weight_scale_inv"),
                       static_cast<size_t>((rows + 127) / 128) * ((cols + 127) / 128),
                       what);
    };
    pair("q_a_proj", r.dsa.q_a_q, q_lora, hidden, "q_a pair");
    pair("kv_a_proj_with_mqa", r.dsa.kv_a_q, kv_lora, hidden, "kv_a pair");
    pair("q_b_proj", r.dsa.q_b_q, 8 * 96, q_lora, "q_b pair");
    pair("o_proj", r.dsa.o_proj_q, hidden, 8 * 96, "o_proj pair");
    require(r.dsa.qkv_a == nullptr && r.dsa.q_b == nullptr && r.dsa.o_proj == nullptr,
            "no bf16 bridge beside the pairs");
  }
  // Fused qkv_a: q_a rows dequantized, then kv_a rows dequantized.
  if (!r.dsa.quantized()) {
    const uint16_t* qkv = static_cast<const uint16_t*>(r.dsa.qkv_a);
    const std::vector<uint16_t> qa = dequant_oracle(
        fx.at(p + "q_a_proj.weight"), fx.at(p + "q_a_proj.weight_scale_inv"),
        q_lora, hidden);
    const std::vector<uint16_t> kva = dequant_oracle(
        fx.at(p + "kv_a_proj_with_mqa.weight"),
        fx.at(p + "kv_a_proj_with_mqa.weight_scale_inv"), kv_lora, hidden);
    require_bytes_eq(qkv, {reinterpret_cast<const uint8_t*>(qa.data()),
                           reinterpret_cast<const uint8_t*>(qa.data()) +
                               qa.size() * 2},
                     qa.size(), "qkv_a q_a rows");
    require_bytes_eq(qkv + q_lora * hidden,
                     {reinterpret_cast<const uint8_t*>(kva.data()),
                      reinterpret_cast<const uint8_t*>(kva.data()) +
                          kva.size() * 2},
                     kva.size(), "qkv_a kv_a rows");
  }
  // q_b / o_proj: standalone dequants, bitwise vs oracle.
  if (!r.dsa.quantized()) {
    const std::vector<uint16_t> qb = dequant_oracle(
        fx.at(p + "q_b_proj.weight"), fx.at(p + "q_b_proj.weight_scale_inv"),
        8 * 96, q_lora);
    require_bytes_eq(static_cast<const uint16_t*>(r.dsa.q_b),
                     {reinterpret_cast<const uint8_t*>(qb.data()),
                      reinterpret_cast<const uint8_t*>(qb.data()) +
                          qb.size() * 2},
                     qb.size(), "q_b dequant");
    const std::vector<uint16_t> op = dequant_oracle(
        fx.at(p + "o_proj.weight"), fx.at(p + "o_proj.weight_scale_inv"),
        hidden, 8 * 96);
    require_bytes_eq(static_cast<const uint16_t*>(r.dsa.o_proj),
                     {reinterpret_cast<const uint8_t*>(op.data()),
                      reinterpret_cast<const uint8_t*>(op.data()) +
                          op.size() * 2},
                     op.size(), "o_proj dequant");
  }
  // kv_b arrives BF16 byte-exact (not quantized in this checkpoint family).
  require_bytes_eq(static_cast<const uint16_t*>(r.dsa.kv_b),
                   fx.at(p + "kv_b_proj.weight"), 8 * 192 * kv_lora, "kv_b");
  // APE: BF16 checkpoint tensor converted to F32.
  {
    const auto& src = fx.at(p + "indexer.index_kpool_compress_ape");
    const uint16_t* s16 =
        reinterpret_cast<const uint16_t*>(src.data());
    const std::vector<uint8_t> ape_bytes = fetch(r.dsa.ape, 4 * 128 * 4);
    const float* ape = reinterpret_cast<const float*>(ape_bytes.data());
    for (size_t i = 0; i < 4 * 128; ++i)
      if (ape[i] != dgpp::bf16_bits_to_float(s16[i]))
        throw std::runtime_error("ape f32 conversion");
  }
  // Indexer tensors byte-exact.
  require_bytes_eq(static_cast<const uint16_t*>(r.dsa.wq_b),
                   fx.at(p + "indexer.wq_b.weight"), 512 * q_lora, "wq_b");
  require_bytes_eq(static_cast<const uint16_t*>(r.dsa.wk),
                   fx.at(p + "indexer.wk.weight"), 128 * hidden, "wk");
  require_bytes_eq(static_cast<const uint16_t*>(r.dsa.wp),
                   fx.at(p + "indexer.weights_proj.weight"), 4 * hidden,
                   "wp");
  require_bytes_eq(static_cast<const uint16_t*>(r.dsa.gate),
                   fx.at(p + "indexer.index_kpool_compress_gate"),
                   128 * hidden, "gate");
  require_bytes_eq(static_cast<const uint16_t*>(r.dsa.k_norm_w),
                   fx.at(p + "indexer.k_norm.weight"), 128, "k_norm w");
  require_bytes_eq(static_cast<const uint16_t*>(r.dsa.k_norm_b),
                   fx.at(p + "indexer.k_norm.bias"), 128, "k_norm b");
  // Router + experts stay compressed, byte-exact.
  const std::string m = "model.language_model.layers.2.mlp.";
  require_bytes_eq(r.moe.router_gate, fx.at(m + "gate.weight"), 4 * hidden,
                   "router gate");
  require_bytes_eq(r.moe.router_bias, fx.at(m + "gate.e_score_correction_bias"),
                   4, "router bias");
  static const char* mats[3] = {"gate_proj", "up_proj", "down_proj"};
  for (int i = 0; i < 3; ++i) {
    const std::string name =
        m + "shared_experts." + std::string(mats[i]) + ".weight";
    require_bytes_eq(r.moe.shared[i].payload, fx.at(name), 256 * 512,
                    "shared payload");
    require_bytes_eq(reinterpret_cast<const uint8_t*>(r.moe.shared[i].scales),
                     fx.at(name + "_scale_inv"), 2 * 4 * 4, "shared scale");
    const std::string e2 = m + "experts.2." + std::string(mats[i]) +
                           ".weight";
    require_bytes_eq(r.moe.expert(2, i).payload, fx.at(e2), 256 * 512,
                    "expert 2 payload");
    // gate/up are [inter, hidden]; down is [hidden, inter].
    const int64_t want_rows = (i == 2) ? 512 : 256;
    const int64_t want_cols = (i == 2) ? 256 : 512;
    require(r.moe.expert(2, i).rows == want_rows &&
                r.moe.expert(2, i).cols == want_cols,
            "expert geometry");
  }
  require(r.moe.experts.size() == 4 * 3, "expert count");
  // Dense view empty on a MoE layer.
  require(r.dense[0].payload == nullptr, "moe layer has empty dense view");
}

DGPP_TEST(glm_loader_streams_mtp_layer_and_globals) {
  const Fixture fx = write_fixture();
  dgpp::GlmLayerStream stream(fx.cfg, fx.dir.string());
  const dgpp::GlmLayerResident& r = stream.load_layer(6);  // MTP draft
  require(r.kind == dgpp::GlmLayerKind::Dsa, "mtp is DSA");
  require(r.mhc.attn_base == nullptr, "mtp has no mhc");
  require(r.bytes == dgpp::GlmLayerStream::layer_bytes(fx.cfg, 6),
          "mtp bytes match formula");
  const std::string p = "model.language_model.layers.6.";
  require_bytes_eq(r.enorm, fx.at(p + "enorm.weight"), 512, "enorm");
  require_bytes_eq(r.eh_proj, fx.at(p + "eh_proj.weight"), 512 * 1024,
                   "eh_proj");
  require_bytes_eq(r.shared_head_norm, fx.at(p + "shared_head.norm.weight"),
                   512, "shared head norm");
  require(r.moe.experts.size() == 12, "mtp moe present");

  const dgpp::GlmGlobalsResident& g = stream.load_globals();
  require_bytes_eq(g.embed, fx.at("model.language_model.embed_tokens.weight"),
                   1000 * 512, "embed");
  require_bytes_eq(g.lm_head, fx.at("lm_head.weight"), 1000 * 512, "lm_head");
  require_bytes_eq(g.final_norm, fx.at("model.language_model.norm.weight"),
                   512, "final norm");
  require(g.bytes == dgpp::GlmLayerStream::globals_bytes(fx.cfg),
          "globals bytes match formula");
}

DGPP_TEST(glm_loader_streaming_replaces_layers_and_counts_every_kind) {
  const Fixture fx = write_fixture();
  dgpp::GlmLayerStream stream(fx.cfg, fx.dir.string());
  // Every layer (0..6): formula == usage enforced inside load_layer; a
  // throw here IS the failure. Streaming replacement: 0 -> 2 -> 6 -> 1.
  const int seq[] = {0, 2, 6, 1, 3, 4, 5, 0};
  for (int layer : seq) {
    const dgpp::GlmLayerResident& r = stream.load_layer(layer);
    require(r.layer == layer, "resident layer id");
  }
  // Layer capacity = max over layers.
  size_t maxb = 0;
  for (int i = 0; i <= 6; ++i)
    maxb = std::max(maxb, dgpp::GlmLayerStream::layer_bytes(fx.cfg, i));
  require(stream.layer_capacity() == maxb, "capacity is max layer");
  // MTP-only config: no layer 6 when num_nextn_predict_layers == 0.
  bool threw = false;
  try {
    (void)stream.load_layer(7);
  } catch (const std::exception&) {
    threw = true;
  }
  require(threw, "layer out of range rejected");
}

DGPP_TEST(glm_loader_resident_mode_serves_cache_hits_without_storage_reads) {
  const Fixture fx = write_fixture();
  const int max_layer =
      fx.cfg.num_hidden_layers + (fx.cfg.mtp_layer() >= 0 ? 1 : 0);

  // GIVEN a resident stream alongside a streaming one (same checkpoint):
  // layer 0 is KDA/dense, layer 2 is DSA/sparse — both kinds covered.
  dgpp::GlmLayerStream streaming(fx.cfg, fx.dir.string());
  dgpp::GlmLayerStream resident(fx.cfg, fx.dir.string(), 0, 1,
                                dgpp::GlmResidency::Resident);

  // WHEN layer 0 materializes on both streams, THEN the bytes are
  // bitwise-identical (the same build code; the parity driver pins the
  // full surface set, this pins the two builds meet byte-for-byte here).
  const dgpp::GlmLayerResident& r0 = resident.load_layer(0);
  const dgpp::GlmLayerResident& s0 = streaming.load_layer(0);
  require(r0.bytes == s0.bytes, "resident layer 0 formula differs");
  require(r0.bytes == dgpp::GlmLayerStream::layer_bytes(fx.cfg, 0),
          "resident layer 0 bytes match formula");
  require(device_bytes_equal(r0.ln1, s0.ln1,
                             static_cast<size_t>(fx.cfg.hidden_size) * 2),
          "resident ln1 differs from streaming");
  {
    // Merged in_proj rows [f_a | g_a | q | k | v | b] — same sizes the
    // byte-exact test below derives from this fixture's config.
    const int64_t head_dim = fx.cfg.kda_head_dim;  // 64
    const int64_t proj = 8 * head_dim;             // 512
    const int64_t heads = 8;
    const size_t in_proj_bytes =
        static_cast<size_t>(2 * head_dim + 3 * proj + heads) *
        static_cast<size_t>(fx.cfg.hidden_size) * 2;
    require(device_bytes_equal(r0.kda.in_proj, s0.kda.in_proj, in_proj_bytes),
            "resident in_proj differs from streaming");
  }

  // WHEN a different layer loads (which EVICTS layer 0 in streaming
  // mode), THEN layer 0's resident view stays at the same addresses —
  // the production residency contract's shape.
  const dgpp::GlmLayerResident& r2 = resident.load_layer(2);
  require(r2.layer == 2, "resident layer 2 id");
  const dgpp::GlmLayerResident& r0_again = resident.load_layer(0);
  require(r0_again.ln1 == r0.ln1 && r0_again.kda.in_proj == r0.kda.in_proj,
          "resident re-load moved the view (rebuild, not cache)");

  // WHEN release_layer() is called, THEN it is a no-op: the model's
  // preconstruct pass calls it unconditionally, and a streaming-style
  // release would make every resident view a dangling pointer.
  resident.release_layer();
  require(resident.load_layer(0).ln1 == r0.ln1,
          "resident release_layer freed a view");

  // THEN the storage counters freeze — but only once every layer is
  // materialized (the re-pass below also serves first loads for layers
  // this test never touched, which are LEGITIMATE storage reads). Pass 1:
  // materialize everything; pass 2: re-serve everything in mixed order.
  for (int l = 0; l < max_layer; ++l) (void)resident.load_layer(l);
  const uint64_t before = resident.source_bytes_read();
  for (int layer : {5, 1, 4, 0, 6, 2, 3, 0}) (void)resident.load_layer(layer);
  require(resident.source_bytes_read() == before,
          "resident re-pass re-read storage — contract broken");

  // AND the formula total is exactly layers + globals — the number a
  // deployment refuses a world on BEFORE the first allocation.
  size_t formula = dgpp::GlmLayerStream::globals_bytes(fx.cfg);
  for (int l = 0; l < max_layer; ++l)
    formula += dgpp::GlmLayerStream::layer_bytes(fx.cfg, l);
  require(dgpp::GlmLayerStream::resident_bytes(fx.cfg) == formula,
          "resident_bytes formula differs from the per-layer sum");
}

// bf12-only residency's loader half (LayerBump's side grants): a packable
// matrix gets its own releasable range while the STAGING layout — the byte
// formula, the mirror, the image — stays the grant order.
DGPP_TEST(layer_bump_side_grants_keep_the_staging_layout_and_release) {
  const size_t gran = dgpp::ReleasableRange::granularity();
  require(gran >= 4096 && (gran & (gran - 1)) == 0, "the mapping granularity is a power of two");
  // Grants: main 1000 B | side 3 MiB | main 300 B, main 5000 B | side 100 KiB | main 64 B.
  const size_t sizes[] = {1000, size_t{3} << 20, 300, 5000, size_t{100} << 10, 64};
  const bool aside[] = {false, true, false, false, true, false};
  size_t total = 0, side_total = 0;
  for (int i = 0; i < 6; ++i) {
    total += dgpp::align_up_256(sizes[i]);
    if (aside[i]) side_total += dgpp::align_up_256(sizes[i]);
  }
  // The counting pass: the same totals without touching memory.
  {
    dgpp::LayerBump count;
    count.counting = true;
    count.side_mode = true;
    count.capacity = SIZE_MAX;
    for (int i = 0; i < 6; ++i) (void)(aside[i] ? count.alloc_side(sizes[i]) : count.alloc(sizes[i]));
    require(count.cursor == total && count.side_bytes == side_total && count.dev_cursor == total - side_total,
            "the counting pass splits the layer's bytes");
  }
  std::vector<uint8_t> stage(total), back(total, 0);
  for (size_t i = 0; i < total; ++i) stage[i] = static_cast<uint8_t>(i * 2654435761u >> 11);
  for (const bool side_mode : {false, true}) {
    dgpp::LayerBump bump;
    bump.side_mode = side_mode;
    bump.init(side_mode ? total - side_total : total);
    bump.stage = stage.data();
    void* grant[6];
    size_t off = 0;
    for (int i = 0; i < 6; ++i) {
      grant[i] = aside[i] ? bump.alloc_side(sizes[i]) : bump.alloc(sizes[i]);
      // The build writes through host(): always the grant-order offset.
      require(bump.host(static_cast<uint8_t*>(grant[i])) == stage.data() + off,
              "host() of a grant is its grant-order staging offset");
      require(bump.host(static_cast<uint8_t*>(grant[i]) + sizes[i] - 1) == stage.data() + off + sizes[i] - 1,
              "host() inside a grant");
      off += dgpp::align_up_256(sizes[i]);
    }
    require(bump.cursor == total, "the cursor walks every grant");
    require(bump.dev_cursor == (side_mode ? total - side_total : total), "the main allocation's bytes");
    const char* base = static_cast<const char*>(bump.base);
    for (int i = 0; i < 6; ++i) {
      const char* g = static_cast<const char*>(grant[i]);
      const bool inside = g >= base && g < base + bump.capacity;
      require(inside == !(side_mode && aside[i]), "a side grant lives outside the layer's allocation");
    }
    bump.upload(nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    // Each grant's device bytes are its staging bytes...
    off = 0;
    for (int i = 0; i < 6; ++i) {
      std::vector<uint8_t> got(sizes[i]);
      DGPP_CUDA_OK(cudaMemcpy(got.data(), grant[i], sizes[i], cudaMemcpyDeviceToHost));
      require(std::memcmp(got.data(), stage.data() + off, sizes[i]) == 0, "a grant's device bytes");
      off += dgpp::align_up_256(sizes[i]);
    }
    // ...and the download is the mirror again, whatever the placement.
    std::fill(back.begin(), back.end(), 0);
    bump.download(back.data());
    require(back == stage, "download returns the grant-order bytes");
    if (!side_mode) {
      require(bump.release_side(grant[1]) == 0, "nothing to release without side grants");
      continue;
    }
    size_t free_before = 0, free_after = 0, device_total = 0;
    DGPP_CUDA_OK(cudaMemGetInfo(&free_before, &device_total));
    const size_t freed = bump.release_side(grant[1]);
    DGPP_CUDA_OK(cudaMemGetInfo(&free_after, &device_total));
    require(freed >= dgpp::align_up_256(sizes[1]) && freed % gran == 0, "the release returns the mapped range");
    require(free_after + (gran << 1) >= free_before + freed,
            "the device's free memory grows by the released bytes");
    require(bump.release_side(grant[1]) == 0, "a second release is a no-op");
    require(bump.release_side(grant[0]) == 0, "a main grant is not releasable");
    bool threw = false;
    try {
      bump.download(back.data());
    } catch (const std::logic_error&) {
      threw = true;
    }
    require(threw, "a download after a release is refused");
    // The other side grant and the main spans are untouched.
    std::vector<uint8_t> got(sizes[4]);
    DGPP_CUDA_OK(cudaMemcpy(got.data(), grant[4], sizes[4], cudaMemcpyDeviceToHost));
    size_t off4 = 0;
    for (int i = 0; i < 4; ++i) off4 += dgpp::align_up_256(sizes[i]);
    require(std::memcmp(got.data(), stage.data() + off4, sizes[4]) == 0, "the other side grant survives");
  }
}

// An image captured in one residency mode restores in the other: the
// packable matrices — every KDA layer's in and out projections and the draft
// layer's eh_proj ([512, 1024]); format v2 packs any row of a multiple of
// eight columns — are granted aside under bf12-only residency, the image
// bytes are the same, and a range gives its memory back.
DGPP_TEST(glm_loader_side_grants_share_the_image_and_release) {
  const Fixture fx = write_fixture();
  const int max_layer = fx.cfg.num_hidden_layers + (fx.cfg.mtp_layer() >= 0 ? 1 : 0);
  const int mtp = fx.cfg.mtp_layer();
  require(mtp >= 0, "the fixture carries a draft layer");
  const fs::path cache = fx.dir / "resident-cache-side";
  const std::string saved = dgpp::GlmLayerStream::resident_image_dir();
  dgpp::GlmLayerStream::set_resident_image_dir(cache.string());
  struct Restore {
    std::string dir;
    ~Restore() {
      dgpp::set_bf16_residency(dgpp::Bf16Residency::Checkpoint);
      dgpp::GlmLayerStream::set_resident_image_dir(dir);
    }
  } restore{saved};

  const size_t eh_bytes = dgpp::align_up_256(size_t{512} * 1024 * 2);
  require(dgpp::GlmLayerStream::layer_side_bytes(fx.cfg, mtp) == eh_bytes,
          "the draft layer's side bytes are eh_proj's");
  // A KDA layer's: the fused in_proj [2 head_dim + 3 proj + heads, hidden]
  // and o_proj [hidden, proj]; a DSA layer grants nothing aside.
  const size_t kda_hidden = static_cast<size_t>(fx.cfg.hidden_size);
  const size_t kda_proj = static_cast<size_t>(fx.cfg.kda_num_heads) * fx.cfg.kda_head_dim;
  const size_t kda_side =
      dgpp::align_up_256((2 * static_cast<size_t>(fx.cfg.kda_head_dim) + 3 * kda_proj + fx.cfg.kda_num_heads) *
                         kda_hidden * 2) +
      dgpp::align_up_256(kda_hidden * kda_proj * 2);
  for (int l = 0; l < fx.cfg.num_hidden_layers; ++l)
    require(dgpp::GlmLayerStream::layer_side_bytes(fx.cfg, l) ==
                (fx.cfg.layers[static_cast<size_t>(l)] == dgpp::GlmLayerKind::Kda ? kda_side : size_t{0}),
            ("layer " + std::to_string(l) + ": the side bytes are the KDA projections'").c_str());

  // Built and captured with the bf16 bytes in place...
  std::vector<std::vector<uint8_t>> built(static_cast<size_t>(max_layer));
  {
    dgpp::GlmLayerStream first(fx.cfg, fx.dir.string(), 0, 1, dgpp::GlmResidency::Resident);
    require(!first.side_grants(), "checkpoint residency grants nothing aside");
    for (int l = 0; l < max_layer; ++l) {
      (void)first.load_layer(l);
      built[static_cast<size_t>(l)].resize(dgpp::GlmLayerStream::layer_bytes(fx.cfg, l));
      first.copy_resident_layer(l, built[static_cast<size_t>(l)].data());
    }
    require(first.image_layers_captured() == max_layer, "the first stream captures every layer");
  }
  // ...restored under bf12-only residency from the SAME image...
  dgpp::set_bf16_residency(dgpp::Bf16Residency::Bf12);
  {
    dgpp::GlmLayerStream second(fx.cfg, fx.dir.string(), 0, 1, dgpp::GlmResidency::Resident);
    require(second.side_grants(), "bf12-only residency grants aside");
    for (int l = 0; l < max_layer; ++l) {
      const dgpp::GlmLayerResident& r = second.load_layer(l);
      std::vector<uint8_t> got(built[static_cast<size_t>(l)].size());
      second.copy_resident_layer(l, got.data());
      require(got == built[static_cast<size_t>(l)],
              ("layer " + std::to_string(l) + ": a side-grant restore is the built bytes").c_str());
      const auto [base, bytes] = second.resident_layer_span(l);
      require(bytes == built[static_cast<size_t>(l)].size() - dgpp::GlmLayerStream::layer_side_bytes(fx.cfg, l),
              "the layer's own allocation excludes its side grants");
      if (l != mtp) continue;
      const char* eh = reinterpret_cast<const char*>(r.eh_proj);
      require(eh < static_cast<const char*>(base) || eh >= static_cast<const char*>(base) + bytes,
              "eh_proj lives in its own range");
      require_bytes_eq(r.eh_proj, fx.at("model.language_model.layers." + std::to_string(mtp) + ".eh_proj.weight"),
                       512 * 1024, "eh_proj (side grant)");
      require(second.release_packed(mtp, r.eh_proj) >= eh_bytes, "eh_proj's range returns its memory");
      require(second.release_packed(mtp, r.eh_proj) == 0, "once");
      require(second.release_packed(mtp, r.ln1) == 0, "a main grant is not releasable");
    }
    require(second.image_layers_restored() == max_layer && second.image_layers_captured() == 0,
            "the side-grant stream restores every layer from the other mode's image");
  }
  // ...and built under bf12-only residency (a fresh cache): the same bytes,
  // an image the bf16 mode restores.
  const fs::path cache2 = fx.dir / "resident-cache-side-built";
  dgpp::GlmLayerStream::set_resident_image_dir(cache2.string());
  {
    dgpp::GlmLayerStream third(fx.cfg, fx.dir.string(), 0, 1, dgpp::GlmResidency::Resident);
    for (int l = 0; l < max_layer; ++l) {
      (void)third.load_layer(l);
      std::vector<uint8_t> got(built[static_cast<size_t>(l)].size());
      third.copy_resident_layer(l, got.data());
      require(got == built[static_cast<size_t>(l)],
              ("layer " + std::to_string(l) + ": a side-grant build is the bf16-mode build").c_str());
    }
    require(third.image_layers_captured() == max_layer, "the side-grant stream captures every layer");
  }
  dgpp::set_bf16_residency(dgpp::Bf16Residency::Checkpoint);
  {
    dgpp::GlmLayerStream fourth(fx.cfg, fx.dir.string(), 0, 1, dgpp::GlmResidency::Resident);
    for (int l = 0; l < max_layer; ++l) {
      (void)fourth.load_layer(l);
      std::vector<uint8_t> got(built[static_cast<size_t>(l)].size());
      fourth.copy_resident_layer(l, got.data());
      require(got == built[static_cast<size_t>(l)], "a bf16-mode restore of a side-grant image");
    }
    require(fourth.image_layers_restored() == max_layer, "restored across modes");
  }
}

DGPP_TEST(glm_loader_resident_image_restore_is_bitwise_and_reads_no_source) {
  const Fixture fx = write_fixture();
  const int max_layer =
      fx.cfg.num_hidden_layers + (fx.cfg.mtp_layer() >= 0 ? 1 : 0);
  const fs::path cache = fx.dir / "resident-cache";
  const std::string saved = dgpp::GlmLayerStream::resident_image_dir();
  dgpp::GlmLayerStream::set_resident_image_dir(cache.string());

  // GIVEN a resident stream that builds every layer from the checkpoint
  // with the image cache enabled (so it CAPTURES each one):
  std::vector<std::vector<uint8_t>> built(static_cast<size_t>(max_layer));
  dgpp::GlmReplicatedDigest computed;
  {
    dgpp::GlmLayerStream first(fx.cfg, fx.dir.string(), 0, 1,
                               dgpp::GlmResidency::Resident);
    computed = first.hash_replicated();  // from the shards; publishes the note
    for (int l = 0; l < max_layer; ++l) {
      (void)first.load_layer(l);
      const auto [base, bytes] = first.resident_layer_span(l);
      require(base != nullptr && bytes ==
                                     dgpp::GlmLayerStream::layer_bytes(fx.cfg, l),
              "built layer span");
      built[static_cast<size_t>(l)].resize(bytes);
      DGPP_CUDA_OK(cudaMemcpy(built[static_cast<size_t>(l)].data(), base, bytes,
                              cudaMemcpyDeviceToHost));
    }
    require(first.image_layers_captured() == max_layer &&
                first.image_layers_restored() == 0,
            "first stream should capture every layer");
  }
  bool note_seen = false;
  for (const auto& e : fs::directory_iterator(cache))
    note_seen |= e.path().extension() == ".digest";
  require(note_seen, "hash_replicated publishes its digest note beside the image");

  // WHEN a second stream opens the same checkpoint with the same cache,
  // THEN every layer restores from the image — zero checkpoint bytes read
  // for layers (globals are not cached) — and the device bytes are
  // bitwise the built ones, view pointers laid out identically.
  {
    dgpp::GlmLayerStream second(fx.cfg, fx.dir.string(), 0, 1,
                                dgpp::GlmResidency::Resident);
    // The digest comes back from the note, equal in every word:
    const dgpp::GlmReplicatedDigest noted = second.hash_replicated();
    require(noted.layer == computed.layer && noted.globals == computed.globals &&
                noted.bytes == computed.bytes && noted.tensors == computed.tensors,
            "digest from the image note must equal the computed digest");
    const uint64_t before = second.source_bytes_read();
    for (int l = 0; l < max_layer; ++l) {
      const dgpp::GlmLayerResident& r = second.load_layer(l);
      require(r.layer == l && r.bytes == built[static_cast<size_t>(l)].size(),
              "restored layer identity/bytes");
      const auto [base, bytes] = second.resident_layer_span(l);
      std::vector<uint8_t> got(bytes);
      DGPP_CUDA_OK(cudaMemcpy(got.data(), base, bytes, cudaMemcpyDeviceToHost));
      require(got == built[static_cast<size_t>(l)],
              ("restored layer " + std::to_string(l) + " differs from built").c_str());
      // The layout pass must hand out the same relative offsets: ln1 is a
      // grant in every layer but the MTP draft's mhc-less shape still has
      // one; compare its offset from the bump base.
      require(r.ln1 != nullptr &&
                  static_cast<const uint8_t*>(static_cast<const void*>(r.ln1)) >=
                      static_cast<const uint8_t*>(base),
              "restored view points into the bump");
    }
    require(second.source_bytes_read() == before,
            "restore read checkpoint bytes for layers");
    require(second.image_layers_restored() == max_layer &&
                second.image_layers_captured() == 0,
            "second stream should restore every layer");
  }

  // AND a checkpoint whose config differs gets its own image — the key
  // covers config.json (and world/rank/headers) — so nothing restores
  // across checkpoints: touch the config, expect a fresh capture.
  {
    const fs::path cfg_path = fx.dir / "config.json";
    std::string text;
    {
      std::ifstream in(cfg_path);
      text.assign(std::istreambuf_iterator<char>(in), {});
    }
    { std::ofstream out(cfg_path); out << text << "\n"; }  // same JSON, new bytes
    dgpp::GlmLayerStream other(fx.cfg, fx.dir.string(), 0, 1,
                               dgpp::GlmResidency::Resident);
    (void)other.load_layer(0);
    require(other.image_layers_restored() == 0 &&
                other.image_layers_captured() == 1,
            "a changed config.json must not restore the old image");
    { std::ofstream out(cfg_path); out << text; }
  }
  dgpp::GlmLayerStream::set_resident_image_dir(saved);
}

DGPP_TEST(glm_loader_rebuilds_rejected_resident_images) {
  const Fixture fx = write_fixture();
  imagefx::Cache<dgpp::GlmLayerStream> cache(fx.dir / "rejected-cache");
  struct RestoreResidency {
    dgpp::Bf16Residency saved = dgpp::bf16_residency();
    ~RestoreResidency() { dgpp::set_bf16_residency(saved); }
  } restore;
  dgpp::set_bf16_residency(dgpp::Bf16Residency::Checkpoint);
  for (const size_t field : {0, 8, 16}) {
    std::vector<uint8_t> expected(dgpp::GlmLayerStream::layer_bytes(fx.cfg, 0));
    {
      dgpp::GlmLayerStream stream(fx.cfg, fx.dir.string(), 0, 1, dgpp::GlmResidency::Resident);
      (void)stream.load_layer(0);
      (void)stream.load_layer(1);
      stream.copy_resident_layer(0, expected.data());
    }
    cache.corrupt(field);
    // A failed checksum occurs after the layout allocated side grants.
    // Recovery must release them and construct fresh, valid views.
    if (field == 16) dgpp::set_bf16_residency(dgpp::Bf16Residency::Bf12);
    {
      dgpp::GlmLayerStream stream(fx.cfg, fx.dir.string(), 0, 1, dgpp::GlmResidency::Resident);
      const auto& layer = stream.load_layer(0);
      (void)stream.load_layer(1);
      std::vector<uint8_t> got(expected.size());
      stream.copy_resident_layer(0, got.data());
      require(stream.side_grants() == (field == 16), "checksum recovery exercises side grants");
      require(got == expected && layer.layer == 0, "rebuilt GLM layer matches checkpoint");
      require(stream.image_layers_captured() == 1 && stream.image_layers_restored() == 1,
              "rebuild rejected GLM layer and restore its neighbor");
      require(stream.source_bytes_read() > 0, "rebuild reads source weights");
    }
    {
      dgpp::GlmLayerStream stream(fx.cfg, fx.dir.string(), 0, 1, dgpp::GlmResidency::Resident);
      (void)stream.load_layer(0);
      require(stream.image_layers_restored() == 1 && stream.source_bytes_read() == 0,
              "repaired GLM layer restores on next startup");
    }
  }
}

DGPP_TEST(glm_resident_image_direct_and_buffered_paths_agree_on_odd_sizes) {
  // GIVEN an image in a real filesystem directory (ext4 in CI: O_DIRECT is
  // available, so aligned prefixes go direct and sub-page tails buffered)
  // and blobs whose sizes straddle the page boundary in every way:
  const fs::path dir = fs::temp_directory_path() / "dgpp_resident_image_test";
  fs::remove_all(dir);
  const std::vector<size_t> sizes = {4096 * 3 + 17, 100, 4096 * 2, 1, 4096 + 4095};
  std::vector<std::vector<uint8_t>> blobs;
  for (size_t i = 0; i < sizes.size(); ++i) {
    std::vector<uint8_t> b(sizes[i]);
    for (size_t k = 0; k < b.size(); ++k)
      b[k] = static_cast<uint8_t>((k * 7 + i * 131) & 0xFF);
    blobs.push_back(std::move(b));
  }
  // Page-aligned source/destination (the loader's pinned staging is) so the
  // direct path is actually taken; +1 for the deliberately unaligned case.
  void* aligned = nullptr;
  require(posix_memalign(&aligned, 4096, 64 * 1024) == 0, "posix_memalign");
  auto* al = static_cast<uint8_t*>(aligned);
  {
    dgpp::GlmResidentImage img(dir.string(), 0xABCDEFULL, static_cast<int>(sizes.size()));
    require(img.present() == 0, "fresh image has no layers");

    // WHEN each blob is written from an aligned buffer and read back into
    // both an aligned buffer (direct prefix + buffered tail) and an
    // unaligned one (all buffered),
    for (size_t i = 0; i < sizes.size(); ++i) {
      std::memcpy(al, blobs[i].data(), sizes[i]);
      img.write_layer(static_cast<int>(i), al, sizes[i]);
    }
    for (size_t i = 0; i < sizes.size(); ++i) {
      // THEN both reads are bitwise the blob and pass verification:
      std::memset(al, 0, 64 * 1024);
      img.read_layer(static_cast<int>(i), al, sizes[i], /*verify=*/true);
      require(std::memcmp(al, blobs[i].data(), sizes[i]) == 0,
              ("aligned read of layer " + std::to_string(i)).c_str());
      std::memset(al, 0, 64 * 1024);
      img.read_layer(static_cast<int>(i), al + 1, sizes[i], /*verify=*/true);
      require(std::memcmp(al + 1, blobs[i].data(), sizes[i]) == 0,
              ("unaligned read of layer " + std::to_string(i)).c_str());
    }
    require(img.present() == static_cast<int>(sizes.size()), "all layers present");
  }
  // AND a reopened image still serves them (the table round-trips):
  {
    dgpp::GlmResidentImage again(dir.string(), 0xABCDEFULL, static_cast<int>(sizes.size()));
    require(again.present() == static_cast<int>(sizes.size()), "reopen keeps layers");
    again.read_layer(0, al, sizes[0], true);
    require(std::memcmp(al, blobs[0].data(), sizes[0]) == 0, "reopen read");
    bool threw = false;
    try { again.read_layer(0, al, sizes[0] + 1, false); } catch (const std::exception&) { threw = true; }
    require(threw, "size mismatch must throw");
  }
  free(aligned);
  fs::remove_all(dir);
}

DGPP_TEST(fp8_dequant_blocks_handles_ragged_tails_bitwise) {
  // [1000, 1000]: 7 full row/col blocks + 104-wide ragged tails on both
  // axes — the geometry the real checkpoint never produces but the kernel
  // must still get right.
  const int64_t rows = 1000, cols = 1000;
  Rng rng(0xC0FFEE);
  std::vector<uint8_t> payload(rows * cols);
  for (auto& b : payload) {
    const float v = 2.0f * rng.unit();
    b = dgpp::float_to_fp8_e4m3_bits(v);
  }
  const int64_t sr = (rows + 127) / 128, sc = (cols + 127) / 128;
  std::vector<uint8_t> scales(sr * sc * 4);
  for (auto& b : scales) b = 0;
  {
    std::vector<float> s(sr * sc);
    for (auto& v : s) v = 0.25f + 1.5f * (rng.unit() + 1.0f);
    std::memcpy(scales.data(), s.data(), scales.size());
  }
  const std::vector<uint16_t> oracle =
      dequant_oracle(payload, scales, rows, cols);

  uint8_t* dev_p = nullptr;
  float* dev_s = nullptr;
  uint16_t* dev_o = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&dev_p, payload.size()));
  DGPP_CUDA_OK(cudaMallocManaged(&dev_s, scales.size()));
  DGPP_CUDA_OK(
      cudaMallocManaged(&dev_o, static_cast<size_t>(rows * cols) * 2));
  std::memcpy(dev_p, payload.data(), payload.size());
  std::memcpy(dev_s, scales.data(), scales.size());
  dgpp::launch_fp8_dequant_blocks(dev_p, dev_s, dev_o, rows, cols, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  require_bytes_eq(dev_o,
                   {reinterpret_cast<const uint8_t*>(oracle.data()),
                    reinterpret_cast<const uint8_t*>(oracle.data()) +
                        oracle.size() * 2},
                   oracle.size(), "ragged dequant");
  DGPP_CUDA_OK(cudaFree(dev_p));
  DGPP_CUDA_OK(cudaFree(dev_s));
  DGPP_CUDA_OK(cudaFree(dev_o));
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  return dgpp::test::run_all();
}

// ---- the NVFP4 profile (docs/nvfp4_plan.md §5, gate 5) ---------------------

DGPP_TEST(glm_loader_streams_nvfp4_moe_layer_byte_exact) {
  const Fixture fx = write_fixture(/*nvfp4=*/true);
  require(fx.cfg.routed_expert_format == dgpp::GlmExpertFormat::Nvfp4Group16,
          "fixture config carries the NVFP4 profile");
  dgpp::GlmLayerStream stream(fx.cfg, fx.dir.string());
  const auto& r = stream.load_layer(2);  // DSA + MoE
  require(r.bytes == dgpp::GlmLayerStream::layer_bytes(fx.cfg, 2),
          "nvfp4 layer byte formula");
  require(r.moe.nvfp4() && r.moe.experts.empty() && r.moe.experts_fp4.size() == 12,
          "routed experts resident as NVFP4 triples");
  require(r.moe.expert_global_scales != nullptr, "global scales gathered");
  static const char* kProj[3] = {"gate_proj", "up_proj", "down_proj"};
  const std::string ep = "model.language_model.layers.2.mlp.experts.2.";
  for (int i = 0; i < 3; ++i) {
    const dgpp::GlmFp4Matrix& m = r.moe.expert_fp4(2, i);
    const std::string base = ep + kProj[i] + ".weight";
    const int64_t want_rows = i == 2 ? 512 : 256, want_cols = i == 2 ? 256 : 512;
    require(m.rows == want_rows && m.cols == want_cols, "nvfp4 expert dims");
    require(m.payload_bytes() == fx.at(base + "_packed").size() &&
                m.scale_bytes() == fx.at(base + "_scale").size(),
            "nvfp4 expert byte counts");
    require_bytes_eq(m.payload, fx.at(base + "_packed"), m.payload_bytes(),
                     "nvfp4 expert packed payload");
    require_bytes_eq(m.scales, fx.at(base + "_scale"), m.scale_bytes(),
                     "nvfp4 expert block scales");
    require(m.global_scale == r.moe.expert_global_scales + 2 * 3 + i,
            "global scale points into the layer's gathered array");
    require_bytes_eq(m.global_scale, fx.at(base + "_global_scale"), 1,
                     "nvfp4 expert global scale");
  }
  // The shared expert and the dense MLPs stay FP8 (the release's bytes).
  const std::string sp = "model.language_model.layers.2.mlp.shared_experts.gate_proj.weight";
  require_bytes_eq(r.moe.shared[0].payload, fx.at(sp), 256 * 512, "shared payload still fp8");
  // The MTP layer's experts stay FP8 under the profile.
  const auto& mtp = stream.load_layer(6);
  require(!mtp.moe.nvfp4() && mtp.moe.experts.size() == 12 && mtp.moe.experts_fp4.empty(),
          "MTP experts stay FP8");
  require(mtp.bytes == dgpp::GlmLayerStream::layer_bytes(fx.cfg, 6), "mtp byte formula");
}

DGPP_TEST(glm_loader_shards_nvfp4_experts_on_the_inter_slice) {
  // World 2, rank 1: gate/up keep rows [128, 256) of the packed payload and
  // the per-row scales; down keeps columns [128, 256) — packed bytes
  // [64, 128) and scale bytes [8, 16) of every row. Oracle slices are cut
  // from the fixture bytes on the host.
  const Fixture fx = write_fixture(/*nvfp4=*/true, /*tp_legal=*/true);
  dgpp::GlmLayerStream stream(fx.cfg, fx.dir.string(), /*rank=*/1, /*world=*/2);
  const auto& r = stream.load_layer(3);  // KDA + MoE
  require(r.moe.nvfp4() && r.moe.experts_fp4.size() == 12, "sharded nvfp4 experts");
  const std::string ep = "model.language_model.layers.3.mlp.experts.1.";
  const int64_t M = 128;  // inter 256 / world 2
  // gate: [256, 512] -> rows [128, 256): packed row = 256 B, scale row = 32 B.
  {
    const dgpp::GlmFp4Matrix& g = r.moe.expert_fp4(1, 0);
    require(g.rows == M && g.cols == 512, "gate slice dims");
    const auto& pk = fx.at(ep + "gate_proj.weight_packed");
    const auto& sc = fx.at(ep + "gate_proj.weight_scale");
    std::vector<uint8_t> want_pk(pk.begin() + 128 * 256, pk.begin() + 256 * 256);
    std::vector<uint8_t> want_sc(sc.begin() + 128 * 32, sc.begin() + 256 * 32);
    require_bytes_eq(g.payload, want_pk, want_pk.size(), "gate row-slice payload");
    require_bytes_eq(g.scales, want_sc, want_sc.size(), "gate row-slice scales");
  }
  // down: [512, 256] -> cols [128, 256): per row packed bytes [64, 128) of
  // 128, scale bytes [8, 16) of 16.
  {
    const dgpp::GlmFp4Matrix& d = r.moe.expert_fp4(1, 2);
    require(d.rows == 512 && d.cols == M, "down slice dims");
    const auto& pk = fx.at(ep + "down_proj.weight_packed");
    const auto& sc = fx.at(ep + "down_proj.weight_scale");
    std::vector<uint8_t> want_pk, want_sc;
    for (int64_t row = 0; row < 512; ++row) {
      want_pk.insert(want_pk.end(), pk.begin() + row * 128 + 64, pk.begin() + row * 128 + 128);
      want_sc.insert(want_sc.end(), sc.begin() + row * 16 + 8, sc.begin() + row * 16 + 16);
    }
    require_bytes_eq(d.payload, want_pk, want_pk.size(), "down column-pack payload");
    require_bytes_eq(d.scales, want_sc, want_sc.size(), "down column-pack scales");
    require_bytes_eq(d.global_scale, fx.at(ep + "down_proj.weight_global_scale"), 1,
                     "down global scale");
  }
  require(r.bytes == dgpp::GlmLayerStream::layer_bytes(fx.cfg, 3, 1, 2),
          "sharded nvfp4 layer byte formula");
  // The rank's bytes are less than a full load's and the global scales are
  // in the verbatim (full-read) set — the reconcile's arithmetic.
  dgpp::GlmLayerStream full(fx.cfg, fx.dir.string());
  full.load_layer(3);
  require(stream.source_bytes_read() < full.source_bytes_read(),
          "a rank reads less than the full layer");
  require(stream.verbatim_source_bytes() >= 12 * 4, "global scales counted as full reads");
}

DGPP_TEST(glm_loader_nvfp4_resident_image_round_trip_is_bitwise) {
  const Fixture fx = write_fixture(/*nvfp4=*/true);
  const fs::path cache = fs::temp_directory_path() / "dgpp_glm_loader_test_fp4_image";
  fs::remove_all(cache);
  const std::string saved = dgpp::GlmLayerStream::resident_image_dir();
  dgpp::GlmLayerStream::set_resident_image_dir(cache.string());
  const int max_layer = fx.cfg.num_hidden_layers + 1;
  std::vector<std::vector<uint8_t>> built(static_cast<size_t>(max_layer));
  {
    dgpp::GlmLayerStream first(fx.cfg, fx.dir.string(), 0, 1,
                               dgpp::GlmResidency::Resident);
    for (int l = 0; l < max_layer; ++l) {
      first.load_layer(l);
      const auto [base, bytes] = first.resident_layer_span(l);
      require(base != nullptr && bytes == dgpp::GlmLayerStream::layer_bytes(fx.cfg, l),
              "resident span");
      built[static_cast<size_t>(l)] = fetch(base, bytes);
    }
    require(first.image_layers_captured() == max_layer, "every layer captured");
  }
  {
    dgpp::GlmLayerStream second(fx.cfg, fx.dir.string(), 0, 1,
                                dgpp::GlmResidency::Resident);
    const uint64_t before = second.source_bytes_read();
    for (int l = 0; l < max_layer; ++l) {
      const auto& r = second.load_layer(l);
      const auto [base, bytes] = second.resident_layer_span(l);
      require(fetch(base, bytes) == built[static_cast<size_t>(l)],
              "restored nvfp4 layer bitwise the built one");
      if (l >= 2 && l < 6)
        require(r.moe.nvfp4() && r.moe.expert_fp4(0, 0).payload != nullptr,
                "restored layer's views are NVFP4");
    }
    require(second.image_layers_restored() == max_layer &&
                second.source_bytes_read() == before,
            "every layer restored from the image, no source reads");
  }
  dgpp::GlmLayerStream::set_resident_image_dir(saved);
  fs::remove_all(cache);
}
