#include "models/minimax/binding.hpp"

#include <format>
#include <stdexcept>

namespace dgpp {
namespace {
using List = std::vector<MinimaxExpectedTensor>;
using Class = MinimaxWeightClass;
using Role = MinimaxTensorRole;
void add(List& out, const std::string& name, DType type, std::vector<int64_t> shape, Class cls,
         int layer, Role role = Role::Plain, int expert = -1) {
  out.push_back({name, type, std::move(shape), cls, layer, expert, role});
}
void mxfp8(List& out, const std::string& base, int64_t n, int64_t k, Class cls, int layer) {
  if (k % 32) throw std::invalid_argument("MiniMax MXFP8 width must align to 32: " + base);
  add(out, base + ".weight", DType::F8_E4M3, {n, k}, cls, layer, Role::Fp8Payload);
  add(out, base + ".weight_scale_inv", DType::U8, {n, k / 32}, cls, layer, Role::Fp8Scale);
}
void nvfp4(List& out, const std::string& base, int64_t n, int64_t k, int layer, int expert) {
  add(out, base + ".weight_packed", DType::U8, {n, k / 2}, Class::RoutedExpert, layer,
      Role::Fp4Payload, expert);
  add(out, base + ".weight_scale", DType::F8_E4M3, {n, k / 16}, Class::RoutedExpert, layer,
      Role::Fp4Scale, expert);
  add(out, base + ".weight_global_scale", DType::F32, {}, Class::RoutedExpert, layer,
      Role::Fp4Global, expert);
  add(out, base + ".input_global_scale", DType::F32, {}, Class::RoutedExpert, layer,
      Role::Fp4InputGlobal, expert);
}
void mlp(List& out, const std::string& base, int64_t h, int64_t i, Class cls, int layer) {
  mxfp8(out, base + "gate_proj", i, h, cls, layer);
  mxfp8(out, base + "up_proj", i, h, cls, layer);
  mxfp8(out, base + "down_proj", h, i, cls, layer);
}
}  // namespace

std::string minimax_layer_prefix(int layer) {
  return "language_model.model.layers." + std::to_string(layer) + ".";
}

std::vector<MinimaxExpectedTensor> minimax_expected_layer_tensors(const MinimaxConfig& c, int l) {
  if (l < 0 || l >= c.num_hidden_layers)
    throw std::invalid_argument("MiniMax binding: layer out of range");
  List out;
  const auto p = minimax_layer_prefix(l);
  const int64_t h = c.hidden_size, d = c.head_dim;
  add(out, p + "input_layernorm.weight", DType::BF16, {h}, Class::LayerNorm, l);
  add(out, p + "post_attention_layernorm.weight", DType::BF16, {h}, Class::LayerNorm, l);
  const auto a = p + "self_attn.";
  mxfp8(out, a + "q_proj", c.num_attention_heads * d, h, Class::Attention, l);
  mxfp8(out, a + "k_proj", c.num_key_value_heads * d, h, Class::Attention, l);
  mxfp8(out, a + "v_proj", c.num_key_value_heads * d, h, Class::Attention, l);
  mxfp8(out, a + "o_proj", h, c.num_attention_heads * d, Class::Attention, l);
  mxfp8(out, a + "index_q_proj", c.index_heads * c.index_dim, h, Class::Attention, l);
  mxfp8(out, a + "index_k_proj", c.index_dim, h, Class::Attention, l);
  for (const char* name : {"q_norm", "k_norm", "index_q_norm", "index_k_norm"})
    add(out, a + name + ".weight", DType::BF16, {d}, Class::Attention, l);
  if (!c.is_moe_layer(l)) {
    mlp(out, p + "mlp.", h, c.dense_intermediate_size, Class::DenseMlp, l);
  } else {
    const auto m = p + "block_sparse_moe.";
    add(out, m + "gate.weight", DType::F32, {c.num_local_experts, h}, Class::Router, l);
    add(out, m + "e_score_correction_bias", DType::F32, {c.num_local_experts}, Class::Router, l);
    mlp(out, m + "shared_experts.", h, c.shared_intermediate_size, Class::SharedExpert, l);
    for (int e = 0; e < c.num_local_experts; ++e) {
      const auto ep = m + "experts." + std::to_string(e) + ".";
      nvfp4(out, ep + "w1", c.intermediate_size, h, l, e);
      nvfp4(out, ep + "w3", c.intermediate_size, h, l, e);
      nvfp4(out, ep + "w2", h, c.intermediate_size, l, e);
    }
  }
  return out;
}

std::vector<MinimaxExpectedTensor> minimax_expected_global_tensors(const MinimaxConfig& c) {
  List out;
  add(out, "language_model.model.embed_tokens.weight", DType::BF16, {c.vocab_size, c.hidden_size},
      Class::Embed, -1);
  add(out, "language_model.model.norm.weight", DType::BF16, {c.hidden_size}, Class::FinalNorm, -1);
  add(out, "language_model.lm_head.weight", DType::BF16, {c.vocab_size, c.hidden_size},
      Class::LmHead, -1);
  return out;
}

std::vector<MinimaxExpectedTensor> minimax_expected_text_tensors(const MinimaxConfig& c) {
  c.validate();
  auto out = minimax_expected_global_tensors(c);
  for (int l = 0; l < c.num_hidden_layers; ++l) {
    auto v = minimax_expected_layer_tensors(c, l);
    out.insert(out.end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
  }
  return out;
}

void minimax_tp_validate_geometry(const MinimaxConfig& c, int rank, int world) {
  c.validate_tp(rank, world);
}

MinimaxBindReport minimax_validate_text_binding(
    const MinimaxConfig& cfg, const std::unordered_map<std::string, MinimaxTensorDesc>& present,
    size_t max_errors) {
  MinimaxBindReport rep;
  const auto expected = minimax_expected_text_tensors(cfg);
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
    if (e.role == MinimaxTensorRole::Fp8Payload) ++rep.fp8_matrices;
    if (e.role == MinimaxTensorRole::Fp4Payload) ++rep.fp4_matrices;
  }
  for (const auto& [name, desc] : present) {
    if (consumed.count(name)) continue;
    if (name.starts_with("vision_tower.") || name.starts_with("multi_modal_projector.") ||
        name.starts_with("patch_merge_mlp."))
      continue;
    ++rep.unexpected;
    push_error(std::format("unexpected tensor '{}'", name));
  }
  return rep;
}

}  // namespace dgpp
