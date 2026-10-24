#include "models/flint/binding.hpp"

#include <format>
#include <stdexcept>

namespace dgpp {
namespace {

using TensorList = std::vector<FlintExpectedTensor>;

void add(TensorList& out, std::string name, DType dtype, std::vector<int64_t> shape,
         FlintWeightClass cls, int layer, int expert = -1,
         FlintTensorRole role = FlintTensorRole::Plain) {
  out.push_back(FlintExpectedTensor{std::move(name), dtype, std::move(shape), cls, layer, expert, role});
}

void add_bf16(TensorList& out, const std::string& name, std::vector<int64_t> shape,
              FlintWeightClass cls, int layer) {
  add(out, name, DType::BF16, std::move(shape), cls, layer);
}

// One fp8 [rows, cols] matrix on 128 x 128 blocks: the codes and the F32
// scale grid (`weight` / `weight_scale_inv`).
void add_fp8(TensorList& out, const std::string& base, int64_t rows, int64_t cols,
             FlintWeightClass cls, int layer, int expert) {
  if (cols % 128 != 0 || rows % 128 != 0)
    throw std::invalid_argument("flint binding: fp8 N and K must be multiples of 128 on " + base);
  add(out, base + ".weight", DType::F8_E4M3, {rows, cols}, cls, layer, expert, FlintTensorRole::Fp8Payload);
  add(out, base + ".weight_scale_inv", DType::F32, {rows / 128, cols / 128}, cls, layer, expert,
      FlintTensorRole::Fp8Scale);
}

// One NVFP4 [rows, cols] matrix: the packed codes, the per-16 e4m3 scales
// and the F32 global (`weight_packed` / `weight_scale` / `weight_global_scale`).
void add_nvfp4(TensorList& out, const std::string& base, int64_t rows, int64_t cols,
               FlintWeightClass cls, int layer, int expert) {
  if (cols % 16 != 0)
    throw std::invalid_argument("flint binding: NVFP4 K must be a multiple of 16 on " + base);
  add(out, base + ".weight_packed", DType::U8, {rows, cols / 2}, cls, layer, expert, FlintTensorRole::Fp4Payload);
  add(out, base + ".weight_scale", DType::F8_E4M3, {rows, cols / 16}, cls, layer, expert, FlintTensorRole::Fp4Scale);
  add(out, base + ".weight_global_scale", DType::F32, {1}, cls, layer, expert, FlintTensorRole::Fp4Global);
}

void expect_attention(TensorList& out, const std::string& p, const FlintTextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size, D = cfg.head_dim;
  const FlintWeightClass c = FlintWeightClass::Attention;
  add_bf16(out, p + "q_proj.weight", {cfg.num_attention_heads * D, H}, c, layer);
  add_bf16(out, p + "k_proj.weight", {cfg.num_key_value_heads * D, H}, c, layer);
  add_bf16(out, p + "v_proj.weight", {cfg.num_key_value_heads * D, H}, c, layer);
  add_bf16(out, p + "o_proj.weight", {H, cfg.o_proj_cols()}, c, layer);
  add_bf16(out, p + "q_norm.weight", {D}, c, layer);
  add_bf16(out, p + "k_norm.weight", {D}, c, layer);
  if (cfg.sink) add_bf16(out, p + "softmax_offset", {cfg.num_attention_heads}, c, layer);
}

void expect_dense_mlp(TensorList& out, const std::string& p, const FlintTextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size, I = cfg.intermediate_size;
  const FlintWeightClass c = FlintWeightClass::DenseMlp;
  add_bf16(out, p + "gate_proj.weight", {I, H}, c, layer);
  add_bf16(out, p + "up_proj.weight", {I, H}, c, layer);
  add_bf16(out, p + "down_proj.weight", {H, I}, c, layer);
}

void expect_moe(TensorList& out, const std::string& p, const FlintTextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size, I = cfg.moe_intermediate_size, S = cfg.shared_expert_intermediate_size;
  add_bf16(out, p + "gate.weight", {cfg.n_routed_experts, H}, FlintWeightClass::Router, layer);
  add_bf16(out, p + "shared_expert.gate_proj.weight", {S, H}, FlintWeightClass::SharedExpert, layer);
  add_bf16(out, p + "shared_expert.up_proj.weight", {S, H}, FlintWeightClass::SharedExpert, layer);
  add_bf16(out, p + "shared_expert.down_proj.weight", {H, S}, FlintWeightClass::SharedExpert, layer);
  for (int e = 0; e < cfg.n_routed_experts; ++e) {
    const std::string ep = p + "experts." + std::to_string(e) + ".";
    const char* names[3] = {"gate_proj", "up_proj", "down_proj"};
    for (int w = 0; w < 3; ++w) {
      const int64_t rows = w == 2 ? H : I, cols = w == 2 ? I : H;
      if (cfg.expert_fp4(layer, w))
        add_nvfp4(out, ep + names[w], rows, cols, FlintWeightClass::RoutedExpert, layer, e);
      else
        add_fp8(out, ep + names[w], rows, cols, FlintWeightClass::RoutedExpert, layer, e);
    }
  }
}

}  // namespace

std::string flint_layer_prefix(int layer) { return "model.layers." + std::to_string(layer) + "."; }

std::vector<FlintExpectedTensor> flint_expected_layer_tensors(const FlintTextConfig& cfg, int layer) {
  if (layer < 0 || layer >= cfg.num_hidden_layers)
    throw std::invalid_argument("flint_expected_layer_tensors: layer out of range");
  const std::string p = flint_layer_prefix(layer);
  const int64_t H = cfg.hidden_size;
  TensorList out;
  add_bf16(out, p + "input_layernorm.weight", {H}, FlintWeightClass::LayerNorm, layer);
  add_bf16(out, p + "post_attention_layernorm.weight", {H}, FlintWeightClass::LayerNorm, layer);
  expect_attention(out, p + "self_attn.", cfg, layer);
  if (cfg.is_moe_layer(layer))
    expect_moe(out, p + "mlp.", cfg, layer);
  else
    expect_dense_mlp(out, p + "mlp.", cfg, layer);
  return out;
}

std::vector<FlintExpectedTensor> flint_expected_global_tensors(const FlintTextConfig& cfg) {
  TensorList out;
  const int64_t H = cfg.hidden_size;
  add_bf16(out, "model.embed_tokens.weight", {cfg.vocab_size, H}, FlintWeightClass::Embed, -1);
  add_bf16(out, "model.norm.weight", {H}, FlintWeightClass::FinalNorm, -1);
  add_bf16(out, "lm_head.weight", {cfg.vocab_size, H}, FlintWeightClass::LmHead, -1);
  return out;
}

std::vector<FlintExpectedTensor> flint_expected_text_tensors(const FlintTextConfig& cfg) {
  TensorList out = flint_expected_global_tensors(cfg);
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    TensorList layer = flint_expected_layer_tensors(cfg, l);
    out.insert(out.end(), std::make_move_iterator(layer.begin()), std::make_move_iterator(layer.end()));
  }
  return out;
}

FlintBindReport flint_validate_text_binding(
    const FlintTextConfig& cfg, const std::unordered_map<std::string, FlintTensorDesc>& present,
    size_t max_errors) {
  FlintBindReport rep;
  const auto expected = flint_expected_text_tensors(cfg);
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
      push_error(std::format("'{}' dtype {} != expected {}", e.name, dtype_name(it->second.dtype), dtype_name(e.dtype)));
      continue;
    }
    if (it->second.shape != e.shape) {
      ++rep.shape_mismatch;
      push_error(std::format("'{}' shape {} != expected {}", e.name, shape_str(it->second.shape), shape_str(e.shape)));
      continue;
    }
    ++rep.matched;
    if (e.role == FlintTensorRole::Fp8Payload) ++rep.fp8_matrices;
    if (e.role == FlintTensorRole::Fp4Payload) ++rep.fp4_matrices;
  }
  for (const auto& [name, desc] : present) {
    if (consumed.count(name)) continue;
    ++rep.unexpected;
    push_error(std::format("unexpected tensor '{}'", name));
  }
  return rep;
}

void flint_tp_validate_geometry(const FlintTextConfig& cfg, int rank, int world) {
  auto fail = [](const std::string& what) { throw std::invalid_argument("flint tp geometry: " + what); };
  if (world < 1 || rank < 0 || rank >= world) fail("rank/world out of range");
  if (cfg.num_key_value_heads % world != 0)
    fail("world must divide num_key_value_heads (a rank's query heads sit on its own kv heads: worlds 1, 2 and 4 for the release)");
  if (cfg.num_attention_heads % world != 0) fail("num_attention_heads must divide by world");
  for (const int inter : {cfg.intermediate_size, cfg.moe_intermediate_size, cfg.shared_expert_intermediate_size})
    if (inter % world != 0) fail("an intermediate size must divide by world");
  if ((cfg.moe_intermediate_size / world) % 128 != 0)
    fail("the routed expert slice must be a multiple of 128 (the fp8 scale block)");
  if ((cfg.intermediate_size / world) % 8 != 0 || (cfg.shared_expert_intermediate_size / world) % 8 != 0)
    fail("the dense and shared expert slices must be multiples of 8");
}

}  // namespace dgpp
