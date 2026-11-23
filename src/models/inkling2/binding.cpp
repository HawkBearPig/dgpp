#include "models/inkling2/binding.hpp"

#include <format>
#include <stdexcept>

namespace dgpp {
namespace {

using TensorList = std::vector<Inkling2ExpectedTensor>;

void add(TensorList& out, std::string name, DType dtype, std::vector<int64_t> shape, Inkling2WeightClass cls,
         int layer, int projection = -1, Inkling2TensorRole role = Inkling2TensorRole::Plain) {
  out.push_back(Inkling2ExpectedTensor{std::move(name), dtype, std::move(shape), cls, layer, projection, role});
}

void add_bf16(TensorList& out, const std::string& name, std::vector<int64_t> shape, Inkling2WeightClass cls, int layer) {
  add(out, name, DType::BF16, std::move(shape), cls, layer);
}

void add_f32(TensorList& out, const std::string& name, std::vector<int64_t> shape, Inkling2WeightClass cls, int layer) {
  add(out, name, DType::F32, std::move(shape), cls, layer);
}

// One projection of all E experts, [rows, cols] each.
void expect_experts(TensorList& out, const std::string& base, const Inkling2Config& cfg, int layer, int projection,
                    int64_t rows, int64_t cols) {
  const int64_t E = cfg.n_routed_experts;
  const Inkling2WeightClass c = Inkling2WeightClass::RoutedExpert;
  if (cfg.format_of(layer, projection) == Inkling2ExpertFormat::Nvfp4) {
    if (cols % 16 != 0) throw std::invalid_argument("inkling2 binding: NVFP4 K must be a multiple of 16 on " + base);
    add(out, base + "_weight", DType::U8, {E, rows, cols / 2}, c, layer, projection, Inkling2TensorRole::Fp4Payload);
    add(out, base + "_weight_scale", DType::F8_E4M3, {E, rows, cols / 16}, c, layer, projection, Inkling2TensorRole::Fp4Scale);
    add(out, base + "_weight_amax", DType::F32, {E}, c, layer, projection, Inkling2TensorRole::Fp4Amax);
    add(out, base + "_weight_outer_scale_denominator", DType::F32, {1}, c, layer, projection,
        Inkling2TensorRole::Fp4Denominator);
  } else {
    if (cols % 32 != 0) throw std::invalid_argument("inkling2 binding: MXFP8 K must be a multiple of 32 on " + base);
    add(out, base + "_weight", DType::F8_E4M3, {E, rows, cols}, c, layer, projection, Inkling2TensorRole::Mx8Payload);
    add(out, base + "_weight_scale", DType::U8, {E, rows, cols / 32}, c, layer, projection, Inkling2TensorRole::Mx8Scale);
  }
}

}  // namespace

std::string inkling2_layer_prefix(int layer) { return "model.llm.layers." + std::to_string(layer) + "."; }

std::vector<Inkling2ExpectedTensor> inkling2_expected_layer_tensors(const Inkling2Config& cfg, int layer) {
  if (layer < 0 || layer >= cfg.num_hidden_layers)
    throw std::invalid_argument("inkling2_expected_layer_tensors: layer out of range");
  const std::string p = inkling2_layer_prefix(layer);
  const std::string a = p + "attn.";
  const int64_t H = cfg.hidden_size, D = cfg.head_dim, R = cfg.d_rel, W = cfg.sconv_kernel_size;
  const int64_t Hq = cfg.num_attention_heads, Hkv = cfg.num_key_value_heads;
  const bool local = cfg.is_local_layer(layer);
  TensorList out;

  if (local) {
    add_bf16(out, p + "attn_norm.weight", {H}, Inkling2WeightClass::LayerNorm, layer);
  } else {
    for (const char* s : {"q", "k", "v"})
      add_bf16(out, p + s + "_input_norm.weight", {H}, Inkling2WeightClass::LayerNorm, layer);
  }
  add_bf16(out, p + "mlp_norm.weight", {H}, Inkling2WeightClass::LayerNorm, layer);

  const Inkling2WeightClass attn = Inkling2WeightClass::Attention, aux = Inkling2WeightClass::AttentionAux;
  add_bf16(out, a + "wq_du.weight", {Hq * D, H}, attn, layer);
  add_bf16(out, a + "wr_du.weight", {Hq * R, H}, attn, layer);
  add_bf16(out, a + "gate_attn.weight", {Hq * D, H}, attn, layer);
  add_bf16(out, a + "attn_sink_proj.weight", {Hq, H}, attn, layer);
  add_bf16(out, a + "wk_dv.weight", {Hkv * D, H}, attn, layer);
  add_bf16(out, a + "wv_dv.weight", {Hkv * D, H}, attn, layer);
  add_bf16(out, a + "wo_ud.weight", {H, Hq * D}, attn, layer);
  add_bf16(out, a + "q_l2_clip.scale", {Hq, D}, aux, layer);
  add_bf16(out, a + "rel_l2_clip.scale", {Hq, R}, aux, layer);
  add_bf16(out, a + "rel_logits_proj.proj", {R, cfg.rel_extent_of(layer)}, aux, layer);
  for (const char* s : {"k", "v"}) {
    if (local) {
      add_bf16(out, a + s + "_sconv.weight", {Hkv * D, 1, W}, aux, layer);
    } else {
      add_bf16(out, a + s + "_sconv.static_w", {W, Hkv * D}, aux, layer);
      add_bf16(out, a + s + "_sconv.weight_proj", {Hkv * W, H}, aux, layer);
    }
  }
  add_bf16(out, p + "attn_sconv.weight", {H, 1, W}, Inkling2WeightClass::HiddenConv, layer);
  add_bf16(out, p + "mlp_sconv.weight", {H, 1, W}, Inkling2WeightClass::HiddenConv, layer);

  add_bf16(out, p + "mlp.gate.weight", {cfg.n_routed_experts, H}, Inkling2WeightClass::Router, layer);
  add_f32(out, p + "mlp.gate.bias", {cfg.n_routed_experts}, Inkling2WeightClass::Router, layer);
  expect_experts(out, p + "mlp.experts.w13", cfg, layer, 0, 2 * static_cast<int64_t>(cfg.intermediate_size), H);
  expect_experts(out, p + "mlp.experts.w2", cfg, layer, 1, H, cfg.intermediate_size);

  // The layer's mix queries live under attn_res: attn_queries.i belongs to
  // layer i + 1 (one query for a local layer, three for a full-attention
  // layer, none for layer 0), mlp_queries.i to layer i.
  if (layer > 0) {
    const std::string name = "model.llm.attn_res.attn_queries." + std::to_string(layer - 1);
    if (local) add_f32(out, name, {H}, Inkling2WeightClass::MixQuery, layer);
    else add_f32(out, name, {3, H}, Inkling2WeightClass::MixQuery, layer);
  }
  add_f32(out, "model.llm.attn_res.mlp_queries." + std::to_string(layer), {H}, Inkling2WeightClass::MixQuery, layer);

  if (cfg.is_core_layer(layer)) {
    for (const char* s : {"attn", "k", "v", "mlp"}) {
      const std::string c = p + "controllers." + s + ".";
      add_bf16(out, c + "down_weight", {cfg.gate_rank, 2 * H}, Inkling2WeightClass::Controller, layer);
      add_bf16(out, c + "gain_weight", {H, cfg.gate_rank}, Inkling2WeightClass::Controller, layer);
      add_bf16(out, c + "forget_weight", {H, cfg.gate_rank}, Inkling2WeightClass::Controller, layer);
    }
  }
  return out;
}

std::vector<Inkling2ExpectedTensor> inkling2_expected_global_tensors(const Inkling2Config& cfg) {
  TensorList out;
  const int64_t H = cfg.hidden_size;
  const std::string p = "model.llm.";
  add_bf16(out, p + "embed.weight", {cfg.padded_vocab_size, H}, Inkling2WeightClass::Embed, -1);
  add_bf16(out, p + "embed_norm.weight", {H}, Inkling2WeightClass::GlobalNorm, -1);
  add_bf16(out, p + "norm.weight", {H}, Inkling2WeightClass::GlobalNorm, -1);
  add_bf16(out, p + "unembed.weight", {cfg.padded_vocab_size, H}, Inkling2WeightClass::Unembed, -1);
  add_f32(out, p + "attn_res.output_query", {H}, Inkling2WeightClass::MixQuery, -1);
  // Two rows per core layer: (attention, MLP), then (key, value).
  const int64_t rows = 2 * static_cast<int64_t>(cfg.core_layers());
  add_f32(out, p + "middle_cycle_cross_pass_queries", {rows, H}, Inkling2WeightClass::MixQuery, -1);
  add_f32(out, p + "middle_cycle_cross_pass_kv_queries", {rows, H}, Inkling2WeightClass::MixQuery, -1);
  return out;
}

std::vector<Inkling2ExpectedTensor> inkling2_expected_text_tensors(const Inkling2Config& cfg) {
  TensorList out = inkling2_expected_global_tensors(cfg);
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    TensorList layer = inkling2_expected_layer_tensors(cfg, l);
    out.insert(out.end(), std::make_move_iterator(layer.begin()), std::make_move_iterator(layer.end()));
  }
  return out;
}

std::vector<Inkling2ExpectedTensor> inkling2_expected_vision_tensors(const Inkling2Config& cfg) {
  TensorList out;
  const Inkling2VisionConfig& v = cfg.vision;
  if (!v.present) return out;
  const Inkling2WeightClass c = Inkling2WeightClass::Vision;
  const int64_t H = v.hidden_size;
  const int64_t mlp = static_cast<int64_t>(static_cast<double>(H) * v.mlp_ratio);
  const int64_t patch = 3 * static_cast<int64_t>(v.native_patch_size) * v.native_patch_size;
  const int64_t grid = static_cast<int64_t>(v.posemb_grid_size) * v.posemb_grid_size;
  const int64_t merged = H * v.adapter_merge_size * v.adapter_merge_size;
  const std::string b = "model.visual.backbone.";
  add_bf16(out, b + "embeddings.class_embedding", {H}, c, -1);
  add_bf16(out, b + "embeddings.cls_position_embedding", {H}, c, -1);
  add_bf16(out, b + "embeddings.patch_embedding.weight", {H, patch}, c, -1);
  add_bf16(out, b + "embeddings.position_embedding", {grid, H}, c, -1);
  add_bf16(out, b + "ln_pre.weight", {H}, c, -1);
  add_bf16(out, b + "ln_pre.bias", {H}, c, -1);
  for (int l = 0; l < v.num_hidden_layers; ++l) {
    const std::string p = b + "layers." + std::to_string(l) + ".";
    for (const char* n : {"ln_1", "ln_2"}) {
      add_bf16(out, p + n + ".weight", {H}, c, l);
      add_bf16(out, p + n + ".bias", {H}, c, l);
    }
    add_bf16(out, p + "ls_1.gamma", {H}, c, l);
    add_bf16(out, p + "ls_2.gamma", {H}, c, l);
    for (const char* n : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
      add_bf16(out, p + "self_attn." + n + ".weight", {H, H}, c, l);
      add_bf16(out, p + "self_attn." + n + ".bias", {H}, c, l);
    }
    add_bf16(out, p + "mlp.c_fc.weight", {mlp, H}, c, l);
    add_bf16(out, p + "mlp.c_fc.bias", {mlp}, c, l);
    add_bf16(out, p + "mlp.c_proj.weight", {H, mlp}, c, l);
    add_bf16(out, p + "mlp.c_proj.bias", {H}, c, l);
  }
  add_bf16(out, "model.visual.proj_in.weight", {v.projection_hidden_dim, merged}, c, -1);
  add_bf16(out, "model.visual.proj_out.weight", {v.text_hidden_size, v.projection_hidden_dim}, c, -1);
  return out;
}

std::vector<Inkling2ExpectedTensor> inkling2_expected_draft_tensors(const Inkling2Config& cfg) {
  TensorList out;
  const Inkling2DraftConfig& d = cfg.draft;
  if (!d.present) return out;
  const Inkling2WeightClass c = Inkling2WeightClass::Draft;
  const int64_t H = d.hidden_size, D = d.head_dim, I = d.intermediate_size, R = d.markov_rank;
  const int64_t Hq = d.num_attention_heads, Hkv = d.num_key_value_heads;
  // One fusion row per draft layer (its context keys and values), then the
  // Markov head's row, over the target taps.
  add_bf16(out, "fuse_weight", {d.num_hidden_layers + 1, static_cast<int64_t>(d.target_layer_ids.size())}, c, -1);
  add_bf16(out, "mask_embed", {H}, c, -1);
  add_bf16(out, "norm.weight", {H}, c, -1);
  for (int l = 0; l < d.num_hidden_layers; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    for (const char* n : {"dflare_norm", "input_layernorm", "post_attention_layernorm"})
      add_bf16(out, p + n + ".weight", {H}, c, l);
    const std::string a = p + "self_attn.";
    add_bf16(out, a + "q_proj.weight", {Hq * D, H}, c, l);
    for (const char* n : {"k_proj", "v_proj", "k_proj_ctx", "v_proj_ctx"})
      add_bf16(out, a + n + ".weight", {Hkv * D, H}, c, l);
    add_bf16(out, a + "o_proj.weight", {H, Hq * D}, c, l);
    for (const char* n : {"q_norm", "k_norm", "k_norm_ctx"}) add_bf16(out, a + n + ".weight", {D}, c, l);
    add_bf16(out, p + "mlp.gate_proj.weight", {I, H}, c, l);
    add_bf16(out, p + "mlp.up_proj.weight", {I, H}, c, l);
    add_bf16(out, p + "mlp.down_proj.weight", {H, I}, c, l);
  }
  // rnn_proj reads [state | markov_w1[previous] | hidden | fused features].
  add_bf16(out, "markov_head.dflare_norm.weight", {H}, c, -1);
  add_bf16(out, "markov_head.markov_w1.weight", {cfg.padded_vocab_size, R}, c, -1);
  add_bf16(out, "markov_head.rnn_proj.weight", {3 * R, 2 * R + 2 * H}, c, -1);
  add_bf16(out, "markov_head.rnn_proj.bias", {3 * R}, c, -1);
  add_bf16(out, "markov_head.hidden_out.weight", {H, R}, c, -1);
  add_bf16(out, "markov_head.hidden_out.bias", {H}, c, -1);
  return out;
}

Inkling2BindReport inkling2_validate_binding(const std::vector<Inkling2ExpectedTensor>& expected,
                                             const std::unordered_map<std::string, Inkling2TensorDesc>& present,
                                             size_t max_errors) {
  Inkling2BindReport rep;
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
    if (e.role == Inkling2TensorRole::Fp4Payload) ++rep.fp4_tensors;
    if (e.role == Inkling2TensorRole::Mx8Payload) ++rep.mxfp8_tensors;
  }
  for (const auto& [name, desc] : present) {
    if (consumed.count(name)) continue;
    ++rep.unexpected;
    push_error(std::format("unexpected tensor '{}'", name));
  }
  return rep;
}

void inkling2_tp_validate_geometry(const Inkling2Config& cfg, int rank, int world) {
  auto fail = [](const std::string& what) { throw std::invalid_argument("inkling2 tp geometry: " + what); };
  if (world < 1 || rank < 0 || rank >= world) fail("rank/world out of range");
  if (cfg.num_key_value_heads % world != 0)
    fail("world must divide num_key_value_heads (a rank's query heads sit on its own key/value heads)");
  if (cfg.num_attention_heads % world != 0) fail("num_attention_heads must divide by world");
  if (cfg.intermediate_size % world != 0 || (cfg.intermediate_size / world) % 32 != 0)
    fail("the expert slice must be a whole number of 32-value scale blocks");
  if (cfg.draft.present) {
    if (cfg.draft.num_key_value_heads % world != 0 || cfg.draft.num_attention_heads % world != 0)
      fail("world must divide the draft model's head counts");
    if (cfg.draft.intermediate_size % world != 0 || (cfg.draft.intermediate_size / world) % 8 != 0)
      fail("the draft model's MLP slice must be a multiple of 8");
  }
}

}  // namespace dgpp
