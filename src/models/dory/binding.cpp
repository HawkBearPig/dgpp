#include "models/dory/binding.hpp"

#include <format>
#include <stdexcept>

namespace dgpp {
namespace {

using TensorList = std::vector<DoryExpectedTensor>;

void add(TensorList& out, std::string name, DType dtype, std::vector<int64_t> shape,
         DoryWeightClass cls, int layer, DoryTensorRole role = DoryTensorRole::Plain) {
  out.push_back(DoryExpectedTensor{std::move(name), dtype, std::move(shape), cls, layer, role});
}

void add_bf16(TensorList& out, const std::string& name, std::vector<int64_t> shape,
              DoryWeightClass cls, int layer) {
  add(out, name, DType::BF16, std::move(shape), cls, layer);
}

// One NVFP4 [rows, cols] projection: the packed codes, the per-16 e4m3 scale
// grid, and the fp32 per-tensor scale. There is no activation scale: the
// release is W4A16 (no `input_scale` tensor exists).
void add_fp4(TensorList& out, const std::string& base, int64_t rows, int64_t cols,
             DoryWeightClass cls, int layer, int group) {
  if (cols % group != 0 || cols % 2 != 0)
    throw std::invalid_argument("dory binding: NVFP4 K must be a multiple of the group on " + base);
  add(out, base + ".weight", DType::U8, {rows, cols / 2}, cls, layer, DoryTensorRole::Fp4Payload);
  add(out, base + ".weight_scale", DType::F8_E4M3, {rows, cols / group}, cls, layer,
      DoryTensorRole::Fp4Scale);
  add(out, base + ".weight_scale_2", DType::F32, {}, cls, layer, DoryTensorRole::Fp4Global);
}

}  // namespace

std::string dory_layer_prefix(const DoryConfig&, int layer) {
  return "backbone.layers." + std::to_string(layer) + ".";
}

std::vector<DoryExpectedTensor> dory_expected_attention_tensors(const DoryConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size;
  const int64_t Q = cfg.q_size(), KV = cfg.kv_size();
  const int g = cfg.nvfp4_group;
  const std::string p = dory_layer_prefix(cfg, layer);
  const std::string m = p + "mixer.";
  TensorList out;
  add_fp4(out, m + "q_proj", Q, H, DoryWeightClass::AttentionProj, layer, g);
  add_fp4(out, m + "k_proj", KV, H, DoryWeightClass::AttentionProj, layer, g);
  add_fp4(out, m + "v_proj", KV, H, DoryWeightClass::AttentionProj, layer, g);
  add_fp4(out, m + "o_proj", H, Q, DoryWeightClass::AttentionProj, layer, g);
  add_fp4(out, m + "g_proj", Q, H, DoryWeightClass::AttentionProj, layer, g);
  // The learned nGPT scales: the per-dimension qk scale and the output gate.
  add_bf16(out, m + "s_gate", {Q}, DoryWeightClass::AttentionScale, layer);
  add_bf16(out, m + "sqk", {Q}, DoryWeightClass::AttentionScale, layer);
  // The residual slerp scale lives at the layer root, not under `.mixer.`.
  add_bf16(out, p + "attn_alpha", {H}, DoryWeightClass::AttentionAlpha, layer);
  return out;
}

std::vector<DoryExpectedTensor> dory_expected_mlp_tensors(const DoryConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size, I = cfg.intermediate_size;
  const int g = cfg.nvfp4_group;
  const std::string p = dory_layer_prefix(cfg, layer);
  const std::string m = p + "mixer.";
  TensorList out;
  add_fp4(out, m + "gate_proj", I, H, DoryWeightClass::MlpProj, layer, g);
  add_fp4(out, m + "up_proj", I, H, DoryWeightClass::MlpProj, layer, g);
  add_fp4(out, m + "down_proj", H, I, DoryWeightClass::MlpProj, layer, g);
  // The learned SwiGLU input scales, one per branch.
  add_bf16(out, m + "suv_gate", {I}, DoryWeightClass::MlpScale, layer);
  add_bf16(out, m + "suv_up", {I}, DoryWeightClass::MlpScale, layer);
  add_bf16(out, p + "mlp_alpha", {H}, DoryWeightClass::MlpAlpha, layer);
  return out;
}

std::vector<DoryExpectedTensor> dory_expected_layer_tensors(const DoryConfig& cfg, int layer) {
  if (layer < 0 || layer >= cfg.num_hidden_layers)
    throw std::invalid_argument("dory_expected_layer_tensors: layer out of range");
  return cfg.is_attention(layer) ? dory_expected_attention_tensors(cfg, layer)
                                 : dory_expected_mlp_tensors(cfg, layer);
}

std::vector<DoryExpectedTensor> dory_expected_global_tensors(const DoryConfig& cfg) {
  TensorList out;
  const int64_t H = cfg.hidden_size;
  add_bf16(out, "backbone.embeddings.weight", {cfg.vocab_size, H}, DoryWeightClass::Embed, -1);
  add_bf16(out, "lm_head.weight", {cfg.vocab_size, H}, DoryWeightClass::LmHead, -1);
  add_bf16(out, "lm_head.logit_scale", {cfg.vocab_size}, DoryWeightClass::LogitScale, -1);
  return out;
}

std::vector<DoryExpectedTensor> dory_expected_text_tensors(const DoryConfig& cfg) {
  TensorList out = dory_expected_global_tensors(cfg);
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    TensorList layer = dory_expected_layer_tensors(cfg, l);
    out.insert(out.end(), std::make_move_iterator(layer.begin()), std::make_move_iterator(layer.end()));
  }
  return out;
}

DoryBindReport dory_validate_text_binding(
    const DoryConfig& cfg, const std::unordered_map<std::string, DoryTensorDesc>& present,
    size_t max_errors) {
  DoryBindReport rep;
  const auto expected = dory_expected_text_tensors(cfg);
  rep.expected = expected.size();
  auto push_error = [&](std::string msg) {
    if (rep.errors.size() < max_errors) rep.errors.push_back(std::move(msg));
  };
  auto shape_str = [](const std::vector<int64_t>& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); ++i) {
      if (i) out += ",";
      out += std::to_string(s[i]);
    }
    return out + "]";
  };
  std::unordered_map<std::string, int8_t> consumed;
  consumed.reserve(present.size());
  for (const auto& e : expected) {
    auto it = present.find(e.name);
    if (it == present.end()) {
      ++rep.missing;
      push_error(std::format("missing tensor '{}'", e.name));
      continue;
    }
    consumed.emplace(e.name, 1);
    if (it->second.dtype != e.dtype) {
      ++rep.dtype_mismatch;
      push_error(std::format("'{}' dtype {} != expected {}", e.name, dtype_name(it->second.dtype),
                             dtype_name(e.dtype)));
      continue;
    }
    if (it->second.shape != e.shape) {
      ++rep.shape_mismatch;
      push_error(std::format("'{}' shape {} != expected {}", e.name, shape_str(it->second.shape),
                             shape_str(e.shape)));
      continue;
    }
    ++rep.matched;
    if (e.role == DoryTensorRole::Fp4Payload) ++rep.fp4_matrices;
  }
  for (const auto& [name, desc] : present) {
    if (consumed.count(name)) continue;
    ++rep.unexpected;
    push_error(std::format("unexpected tensor '{}'", name));
  }
  return rep;
}

void dory_validate_world(int world) {
  if (world != 1)
    throw std::invalid_argument(
        "dory: the release is a one-rank image (the reference refuses tensor_parallel_size != 1); "
        "world must be 1, got " +
        std::to_string(world));
}

}  // namespace dgpp
