#include "models/qwen/binding.hpp"

#include <cstdlib>
#include <format>
#include <stdexcept>

namespace dgpp {
namespace {

using TensorList = std::vector<QwenExpectedTensor>;

void add(TensorList& out, std::string name, DType dtype, std::vector<int64_t> shape,
         QwenWeightClass cls, int layer, int expert = -1,
         QwenTensorRole role = QwenTensorRole::Plain) {
  out.push_back(QwenExpectedTensor{std::move(name), dtype, std::move(shape), cls, layer,
                                   expert, role});
}

void add_bf16(TensorList& out, const std::string& name, std::vector<int64_t> shape,
              QwenWeightClass cls, int layer) {
  add(out, name, DType::BF16, std::move(shape), cls, layer);
}

// The e4m3 payload + its block-scale partner, as one call: BF16 scales (the
// Qwen FP8 release) or F32 (the AutoRound hybrid's side layers, DeepSeek's
// convention) by `f32_scale`.
void add_quantized(TensorList& out, const std::string& name, int64_t rows, int64_t cols,
                   QwenWeightClass cls, int layer, int expert, bool nvfp4 = false,
                   bool f32_scale = false, bool input_scale = true, bool csf_scales = false) {
  if (nvfp4) {
    // The modelopt NVFP4 triple (plus its activation scale, which the
    // weight-only W4A16_NVFP4 form of the QAD export's draft experts does
    // not carry): `name` is "...proj.weight". The FP4-Local export wraps
    // the backbone experts' per-group scales as CSF sidecars instead of
    // the plain E4M3 weight_scale (config experts_nvfp4_csf): the sidecars
    // are counted, not tabled (their shapes are data-dependent), and the
    // loader expands them once.
    const std::string base = name.substr(0, name.size() - std::string(".weight").size());
    add(out, name, DType::U8, {rows, cols / 2}, cls, layer, expert, QwenTensorRole::Fp4Payload);
    if (!csf_scales)
      add(out, base + ".weight_scale", DType::F8_E4M3, {rows, cols / 16}, cls, layer, expert,
          QwenTensorRole::Fp4Scale);
    add(out, base + ".weight_scale_2", DType::F32, {}, cls, layer, expert, QwenTensorRole::Fp4Global);
    if (input_scale)
      add(out, base + ".input_scale", DType::F32, {}, cls, layer, expert, QwenTensorRole::InputScale);
    return;
  }
  add(out, name, DType::F8_E4M3, {rows, cols}, cls, layer, expert, QwenTensorRole::Fp8Payload);
  add(out, name + "_scale_inv", f32_scale ? DType::F32 : DType::BF16, qwen_scale_shape({rows, cols}),
      cls, layer, expert, QwenTensorRole::Fp8Scale);
}

// An MXFP8 matrix (the QAD export's dense stack): e4m3 `name` [rows, cols]
// and one e8m0 byte per 32 columns of a row, `base.weight_scale` U8
// [rows, cols/32]; `name` is "...proj.weight".
void add_mxfp8(TensorList& out, const std::string& name, int64_t rows, int64_t cols,
               QwenWeightClass cls, int layer) {
  const std::string base = name.substr(0, name.size() - std::string(".weight").size());
  add(out, name, DType::F8_E4M3, {rows, cols}, cls, layer, -1, QwenTensorRole::Mxfp8Payload);
  add(out, base + ".weight_scale", DType::U8, {rows, cols / kQwenMxfp8Group}, cls, layer, -1,
      QwenTensorRole::Mxfp8Scale);
}

// A dense projection in the form the checkpoint ships it: BF16, (the
// AutoRound hybrid's 48 backbone layers) block FP8 with F32 scales, or (the
// QAD export's 48 backbone layers) MXFP8 — both hybrids' draft layer keeps
// its projections and shared expert in BF16.
void add_dense(TensorList& out, const std::string& name, int64_t rows, int64_t cols,
               QwenWeightClass cls, int layer, const QwenTextConfig& cfg) {
  if (cfg.dense_mxfp8_shipped && layer != cfg.mtp_layer())
    add_mxfp8(out, name, rows, cols, cls, layer);
  else if (cfg.dense_fp8_shipped && layer != cfg.mtp_layer())
    add_quantized(out, name, rows, cols, cls, layer, -1, false, true);
  else
    add_bf16(out, name, {rows, cols}, cls, layer);
}

// A (1 + w) norm vector: BF16, or F32 for the QAD export's trained norms
// (the backbone's and the model mixer's; the draft layer's stay BF16).
void add_norm(TensorList& out, const std::string& name, int64_t n, QwenWeightClass cls, int layer,
              const QwenTextConfig& cfg) {
  const bool f32 = cfg.norms_f32 && layer != cfg.mtp_layer();
  add(out, name, f32 ? DType::F32 : DType::BF16, {n}, cls, layer);
}

// The GPTQ triple of the logical [N, K] matrix `base` ("...gate_proj",
// "lm_head") at `bits` (4 or 8), group `group`.
void add_gptq(TensorList& out, const std::string& base, int64_t N, int64_t K, int bits, int group,
              QwenWeightClass cls, int layer, int expert) {
  add(out, base + ".qweight", DType::I32, {K * bits / 32, N}, cls, layer, expert,
      QwenTensorRole::GptqCodes);
  add(out, base + ".scales", DType::F16, {K / group, N}, cls, layer, expert,
      QwenTensorRole::GptqScales);
  add(out, base + ".qzeros", DType::I32, {K / group, N * bits / 32}, cls, layer, expert,
      QwenTensorRole::GptqZeros);
}

// A gated-residual site: hc_norm [W], down [r, W], up [W, r], inject [n, W]
// (the model-level mixer has no inject).
void expect_gr(TensorList& out, const std::string& p, const QwenTextConfig& cfg, int layer,
               QwenWeightClass cls, bool combine) {
  const int64_t W = cfg.hyper_width();
  add_norm(out, p + "hc_norm.weight", W, cls, layer, cfg);
  add_bf16(out, p + "input_mix_weight_down.weight", {cfg.hc_lowrank, W}, cls, layer);
  add_bf16(out, p + "input_mix_weight_up.weight", {W, cfg.hc_lowrank}, cls, layer);
  if (combine) add_bf16(out, p + "block_inject_weight.weight", {cfg.hc_count, W}, cls, layer);
}

void expect_gdn(TensorList& out, const std::string& p, const QwenTextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size;
  const int64_t kdim = static_cast<int64_t>(cfg.gdn_key_heads) * cfg.gdn_key_head_dim;
  const int64_t vdim = static_cast<int64_t>(cfg.gdn_value_heads) * cfg.gdn_value_head_dim;
  const int64_t vh = cfg.gdn_value_heads;
  const QwenWeightClass c = QwenWeightClass::Gdn;
  add_bf16(out, p + "A_log", {vh}, c, layer);
  add_bf16(out, p + "dt_bias", {vh}, c, layer);
  add_bf16(out, p + "conv1d.weight", {2 * kdim + vdim, 1, cfg.gdn_conv_width}, c, layer);
  // The AutoRound hybrid ships a and b in BF16 beside its FP8 stack; the
  // QAD export quantizes them with the rest (MXFP8).
  if (cfg.dense_mxfp8_shipped) {
    add_dense(out, p + "in_proj_a.weight", vh, H, c, layer, cfg);
    add_dense(out, p + "in_proj_b.weight", vh, H, c, layer, cfg);
  } else {
    add_bf16(out, p + "in_proj_a.weight", {vh, H}, c, layer);
    add_bf16(out, p + "in_proj_b.weight", {vh, H}, c, layer);
  }
  add_dense(out, p + "in_proj_qkv.weight", 2 * kdim + vdim, H, c, layer, cfg);
  add_dense(out, p + "in_proj_z.weight", vdim, H, c, layer, cfg);
  add_bf16(out, p + "norm.weight", {cfg.gdn_value_head_dim}, c, layer);
  add_dense(out, p + "out_proj.weight", H, vdim, c, layer, cfg);
}

void expect_qsa(TensorList& out, const std::string& p, const QwenTextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size;
  const int64_t qh = cfg.num_attention_heads, kvh = cfg.num_key_value_heads;
  const int64_t d = cfg.head_dim;
  const QwenWeightClass c = QwenWeightClass::Qsa;
  add_dense(out, p + "q_proj.weight", qh * d * 2, H, c, layer, cfg);  // [q | gate] per head
  add_dense(out, p + "k_proj.weight", kvh * d, H, c, layer, cfg);
  add_dense(out, p + "v_proj.weight", kvh * d, H, c, layer, cfg);
  add_dense(out, p + "o_proj.weight", H, qh * d, c, layer, cfg);
  add_bf16(out, p + "q_norm.weight", {d}, c, layer);
  add_bf16(out, p + "k_norm.weight", {d}, c, layer);
  const int64_t id = cfg.indexer_head_dim;
  // The indexer projection: BF16 (the AutoRound hybrid's too), or the QAD
  // export's MXFP8 with the rest of the attention form.
  if (cfg.dense_mxfp8_shipped && layer != cfg.mtp_layer())
    add_mxfp8(out, p + "indexer.index_qk_proj.weight", (cfg.indexer_n_heads + cfg.indexer_kv_heads) * id, H,
              QwenWeightClass::QsaIndexer, layer);
  else
    add_bf16(out, p + "indexer.index_qk_proj.weight",
             {(cfg.indexer_n_heads + cfg.indexer_kv_heads) * id, H}, QwenWeightClass::QsaIndexer,
             layer);
  add_bf16(out, p + "indexer.q_layernorm.weight", {id}, QwenWeightClass::QsaIndexer, layer);
  add_bf16(out, p + "indexer.k_layernorm.weight", {id}, QwenWeightClass::QsaIndexer, layer);
}

void expect_moe(TensorList& out, const std::string& p, const QwenTextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size;
  add_bf16(out, p + "gate.weight", {cfg.num_experts, H}, QwenWeightClass::Router, layer);
  add_bf16(out, p + "shared_expert_gate.weight", {1, H}, QwenWeightClass::Router, layer);
  const int64_t S = cfg.shared_expert_intermediate_size;
  add_dense(out, p + "shared_expert.gate_proj.weight", S, H, QwenWeightClass::SharedExpert, layer, cfg);
  add_dense(out, p + "shared_expert.up_proj.weight", S, H, QwenWeightClass::SharedExpert, layer, cfg);
  add_dense(out, p + "shared_expert.down_proj.weight", H, S, QwenWeightClass::SharedExpert, layer, cfg);
  const int64_t I = cfg.moe_intermediate_size;
  const int64_t E = cfg.num_experts;
  // The AutoRound hybrid: every layer's experts (the draft layer's too) as
  // int4 GPTQ triples.
  if (cfg.experts_gptq_int4) {
    for (int e = 0; e < E; ++e) {
      const std::string ep = p + "experts." + std::to_string(e) + ".";
      add_gptq(out, ep + "gate_proj", I, H, 4, cfg.gptq_group, QwenWeightClass::RoutedExpert, layer, e);
      add_gptq(out, ep + "up_proj", I, H, 4, cfg.gptq_group, QwenWeightClass::RoutedExpert, layer, e);
      add_gptq(out, ep + "down_proj", H, I, 4, cfg.gptq_group, QwenWeightClass::RoutedExpert, layer, e);
    }
    return;
  }
  // The NVFP4 release quantizes the backbone's routed experts only; the MTP
  // layer's keep the FP8 block form. The QAD export's draft experts are
  // NVFP4 too, weight-only (no input_scale).
  const bool is_mtp = layer == cfg.mtp_layer();
  if (cfg.mtp_experts_nvfp4 && is_mtp) {
    for (int e = 0; e < E; ++e) {
      const std::string ep = p + "experts." + std::to_string(e) + ".";
      add_quantized(out, ep + "gate_proj.weight", I, H, QwenWeightClass::RoutedExpert, layer, e, true, false, false,
                    cfg.mtp_experts_nvfp4_csf);
      add_quantized(out, ep + "up_proj.weight", I, H, QwenWeightClass::RoutedExpert, layer, e, true, false, false,
                    cfg.mtp_experts_nvfp4_csf);
      add_quantized(out, ep + "down_proj.weight", H, I, QwenWeightClass::RoutedExpert, layer, e, true, false, false,
                    cfg.mtp_experts_nvfp4_csf);
    }
    return;
  }
  const bool nvfp4 = cfg.experts_nvfp4 && !is_mtp;
  // The RadixArk release stores the draft layer's 512 experts as one fused
  // BF16 pair (gate_up_proj [E, 2*I, H] and down_proj [E, H, I]) — the gate
  // and up projections stacked on dim 1, no per-expert index — which the
  // loader encodes to FP8 at load. Skip the per-expert FP8/NVFP4 table.
  if (cfg.mtp_experts_bf16_fused && layer == cfg.mtp_layer()) {
    add_bf16(out, p + "experts.gate_up_proj", {E, 2 * I, H}, QwenWeightClass::RoutedExpert, layer);
    add_bf16(out, p + "experts.down_proj", {E, H, I}, QwenWeightClass::RoutedExpert, layer);
    return;
  }
  for (int e = 0; e < E; ++e) {
    const std::string ep = p + "experts." + std::to_string(e) + ".";
    // The FP4-Local export's CSF-wrapped scales (config experts_nvfp4_csf):
    // no plain weight_scale; the sidecars are counted at validation and
    // expanded once at load.
    const bool csf = nvfp4 && cfg.experts_nvfp4_csf;
    add_quantized(out, ep + "gate_proj.weight", I, H, QwenWeightClass::RoutedExpert, layer, e, nvfp4,
                  false, true, csf);
    add_quantized(out, ep + "up_proj.weight", I, H, QwenWeightClass::RoutedExpert, layer, e, nvfp4,
                  false, true, csf);
    add_quantized(out, ep + "down_proj.weight", H, I, QwenWeightClass::RoutedExpert, layer, e, nvfp4,
                  false, true, csf);
  }
}

void expect_ple(TensorList& out, const std::string& p, const QwenTextConfig& cfg, int layer) {
  const int64_t W = cfg.hyper_width();
  const int64_t E = cfg.ple_embed_dim;
  const QwenWeightClass c = QwenWeightClass::Ple;
  add_bf16(out, p + "conv1d.weight", {W, 1, cfg.ple_conv_kernel_size}, c, layer);
  add_bf16(out, p + "key_proj.weight", {W, E}, c, layer);
  add_bf16(out, p + "value_proj.weight", {cfg.hidden_size, E}, c, layer);
  add_norm(out, p + "norm_conv.weight", W, c, layer, cfg);
  add_norm(out, p + "norm_key.weight", W, c, layer, cfg);
  add_norm(out, p + "norm_query.weight", W, c, layer, cfg);
  const QwenNgramGeometry g = cfg.ngram_geometry();
  const std::string ep = p + "ple_embedding.";
  add(out, ep + "layer_multipliers", DType::I64, {cfg.ngram_size}, c, layer);
  add(out, ep + "ngram_heads_offsets", DType::I64, {g.heads}, c, layer);
  add(out, ep + "ngram_heads_vocab_sizes", DType::I64, {g.heads}, c, layer);
  if (cfg.ngram_table_nvfp4) {
    // The QAD export's table: e2m1 codes with e4m3 scales per 16 and one
    // F32 global (modelopt's layout, the experts' triple at table scale).
    add(out, ep + "ngram_embedding.weight_scale_2", DType::F32, {1}, QwenWeightClass::PleTable, layer, -1,
        QwenTensorRole::NgramScale);
    for (int s = 0; s < cfg.split_ngram_parts; ++s) {
      const std::string sp = ep + "ngram_embedding.shard_" + std::to_string(s);
      const int64_t rows = qwen_ngram_shard_rows(cfg, s);
      add(out, sp + ".weight", DType::U8, {rows, g.head_dim / 2}, QwenWeightClass::PleTable, layer, s,
          QwenTensorRole::NgramShard);
      add(out, sp + ".weight_scale", DType::F8_E4M3, {rows, g.head_dim / kQwenNvfp4Group},
          QwenWeightClass::PleTable, layer, s, QwenTensorRole::NgramShardScale);
    }
    return;
  }
  add(out, ep + "ngram_embedding.weight_scale", DType::BF16, {1}, QwenWeightClass::PleTable, layer,
      -1, QwenTensorRole::NgramScale);
  for (int s = 0; s < cfg.split_ngram_parts; ++s)
    add(out, ep + "ngram_embedding.shard_" + std::to_string(s) + ".weight", DType::F8_E4M3,
        {qwen_ngram_shard_rows(cfg, s), g.head_dim}, QwenWeightClass::PleTable, layer, s,
        QwenTensorRole::NgramShard);
}

int max_layer(const QwenTextConfig& cfg) {
  return cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
}

}  // namespace

std::vector<int64_t> qwen_scale_shape(const std::vector<int64_t>& payload) {
  if (payload.size() != 2) throw std::invalid_argument("qwen_scale_shape: payload must be 2-D");
  return {(payload[0] + 127) / 128, (payload[1] + 127) / 128};
}

int64_t qwen_ngram_shard_capacity(const QwenTextConfig& cfg) {
  const int64_t padded = cfg.ngram_geometry().padded_rows;
  return (padded + cfg.split_ngram_parts - 1) / cfg.split_ngram_parts;
}

int64_t qwen_ngram_shard_rows(const QwenTextConfig& cfg, int shard) {
  const int64_t padded = cfg.ngram_geometry().padded_rows;
  const int64_t cap = qwen_ngram_shard_capacity(cfg);
  const int64_t begin = static_cast<int64_t>(shard) * cap;
  if (shard < 0 || begin >= padded)
    throw std::invalid_argument("qwen_ngram_shard_rows: shard out of range");
  return std::min(cap, padded - begin);
}

std::string qwen_layer_prefix(const QwenTextConfig& cfg, int layer) {
  if (layer == cfg.mtp_layer()) return "mtp.layers.0.";
  return "model.language_model.layers." + std::to_string(layer) + ".";
}

std::vector<QwenExpectedTensor> qwen_expected_layer_tensors(const QwenTextConfig& cfg,
                                                            int layer) {
  if (layer < 0 || layer >= max_layer(cfg))
    throw std::invalid_argument("qwen_expected_layer_tensors: layer out of range");
  const bool is_mtp = layer == cfg.mtp_layer();
  const QwenLayerKind kind = is_mtp ? QwenLayerKind::Qsa : cfg.layers[layer];
  const std::string p = qwen_layer_prefix(cfg, layer);
  TensorList out;
  if (!is_mtp && layer == cfg.ple_layer()) expect_ple(out, p + "ple.", cfg, layer);
  expect_gr(out, p + "attn_hyper_connection.", cfg, layer, QwenWeightClass::Gr, true);
  if (kind == QwenLayerKind::Gdn)
    expect_gdn(out, p + "linear_attn.", cfg, layer);
  else
    expect_qsa(out, p + "self_attn.", cfg, layer);
  expect_gr(out, p + "mlp_hyper_connection.", cfg, layer, QwenWeightClass::Gr, true);
  expect_moe(out, p + "mlp.", cfg, layer);
  return out;
}

std::vector<QwenExpectedTensor> qwen_expected_global_tensors(const QwenTextConfig& cfg) {
  TensorList out;
  const int64_t H = cfg.hidden_size;
  add_bf16(out, "model.language_model.embed_tokens.weight", {cfg.vocab_size, H},
           QwenWeightClass::Embed, -1);
  if (cfg.lm_head_gptq_int8)
    add_gptq(out, "lm_head", cfg.vocab_size, H, 8, cfg.gptq_group, QwenWeightClass::LmHead, -1, -1);
  else
    add_bf16(out, "lm_head.weight", {cfg.vocab_size, H}, QwenWeightClass::LmHead, -1);
  expect_gr(out, "model.language_model.hyper_connection_mixer.", cfg, -1, QwenWeightClass::Mixer,
            false);
  if (cfg.mtp_layer() >= 0) {
    add_bf16(out, "mtp.fc_embedding.weight", {H, H}, QwenWeightClass::Mtp, cfg.mtp_layer());
    add_bf16(out, "mtp.fc_hidden.weight", {H, H}, QwenWeightClass::Mtp, cfg.mtp_layer());
    add_bf16(out, "mtp.pre_fc_norm_embedding.weight", {H}, QwenWeightClass::Mtp, cfg.mtp_layer());
    add_bf16(out, "mtp.pre_fc_norm_hidden.weight", {cfg.hyper_width()}, QwenWeightClass::Mtp,
             cfg.mtp_layer());
    expect_gr(out, "mtp.hyper_connection_mixer.", cfg, cfg.mtp_layer(), QwenWeightClass::Mtp,
              false);
  }
  return out;
}

std::vector<QwenExpectedTensor> qwen_expected_text_tensors(const QwenTextConfig& cfg) {
  TensorList out = qwen_expected_global_tensors(cfg);
  for (int l = 0; l < max_layer(cfg); ++l) {
    TensorList layer = qwen_expected_layer_tensors(cfg, l);
    out.insert(out.end(), std::make_move_iterator(layer.begin()),
               std::make_move_iterator(layer.end()));
  }
  return out;
}

QwenBindReport qwen_validate_text_binding(
    const QwenTextConfig& cfg, const std::unordered_map<std::string, QwenTensorDesc>& present,
    size_t max_errors) {
  QwenBindReport rep;
  const auto expected = qwen_expected_text_tensors(cfg);
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
      push_error(std::format("'{}' dtype {} != expected {}", e.name,
                             dtype_name(it->second.dtype), dtype_name(e.dtype)));
      continue;
    }
    if (it->second.shape != e.shape) {
      ++rep.shape_mismatch;
      push_error(std::format("'{}' shape {} != expected {}", e.name,
                             shape_str(it->second.shape), shape_str(e.shape)));
      continue;
    }
    ++rep.matched;
    if (e.quantized()) ++rep.quantized_matrices;
    if (e.role == QwenTensorRole::NgramShard) ++rep.ngram_shards;
  }
  // Sidecars belong to a specific routed matrix, not to an allowlisted
  // suffix. Validate both before allocating the resident model; only the
  // exception count is data-dependent.
  for (const auto& e : expected) {
    if (e.role != QwenTensorRole::Fp4Payload || e.cls != QwenWeightClass::RoutedExpert) continue;
    if (!(e.layer == cfg.mtp_layer() ? cfg.mtp_experts_nvfp4_csf : cfg.experts_nvfp4_csf)) continue;
    const int64_t rows = e.shape[0], cols = e.shape[1] / 8;  // K/16 scales
    for (const bool fixed : {true, false}) {
      const std::string name = e.name + "_scale.nvfp4_csf_" + (fixed ? "fixed" : "exceptions");
      const auto it = present.find(name);
      if (it == present.end()) {
        ++rep.missing;
        push_error(std::format("missing tensor '{}'", name));
        continue;
      }
      consumed.emplace(name, 1);
      const auto& d = it->second;
      if (d.dtype != (fixed ? DType::U8 : DType::U32)) {
        ++rep.dtype_mismatch;
        push_error(std::format("'{}' has the wrong CSF sidecar dtype", name));
      } else if (fixed ? d.shape != std::vector<int64_t>{rows / 16, 16 * (1 + cols / 2)}
                       : d.shape.size() != 1 || d.shape[0] < 0 || d.shape[0] > rows * cols) {
        ++rep.shape_mismatch;
        push_error(std::format("'{}' has an invalid CSF sidecar shape {}", name, shape_str(d.shape)));
      } else {
        ++rep.csf_sidecars;
      }
    }
  }
  for (const auto& [name, desc] : present) {
    if (consumed.count(name)) continue;
    if (name.rfind("model.visual.", 0) == 0) {
      ++rep.vision;
      continue;
    }
    // A truncated config (the check apps' --layers N): the layers past it
    // are out of scope, not unexpected.
    {
      static const std::string kLayers = "model.language_model.layers.";
      if (name.rfind(kLayers, 0) == 0) {
        const size_t dot = name.find('.', kLayers.size());
        const int64_t idx = dot == std::string::npos ? -1 : std::atoll(name.substr(kLayers.size(), dot - kLayers.size()).c_str());
        if (idx >= cfg.num_hidden_layers) {
          ++rep.out_of_scope;
          continue;
        }
      }
    }
    ++rep.unexpected;
    push_error(std::format("unexpected tensor '{}'", name));
  }
  return rep;
}

void qwen_tp_validate_geometry(const QwenTextConfig& cfg, int rank, int world) {
  auto fail = [](const std::string& what) { throw std::invalid_argument("qwen tp geometry: " + what); };
  if (world < 1 || rank < 0 || rank >= world) fail("rank/world out of range");
  if (cfg.gdn_key_heads % world != 0) fail("linear_num_key_heads must divide by world");
  if (cfg.gdn_value_heads % world != 0) fail("linear_num_value_heads must divide by world");
  if (cfg.num_attention_heads % world != 0) fail("num_attention_heads must divide by world");
  if (cfg.num_key_value_heads % world != 0 && world % cfg.num_key_value_heads != 0)
    fail("num_key_value_heads must divide world or be divided by it");
  if (cfg.num_attention_heads / cfg.num_key_value_heads * cfg.num_key_value_heads !=
      cfg.num_attention_heads)
    fail("query heads per kv head");
  if (!cfg.ple_layer_ids.empty() && cfg.ngram_geometry().heads % world != 0)
    fail("the n-gram hash heads must divide by world");
  for (const int inter : {cfg.moe_intermediate_size, cfg.shared_expert_intermediate_size}) {
    if (inter % world != 0) fail("an expert intermediate size must divide by world");
    if ((inter / world) % 16 != 0) fail("an expert intermediate slice must be a multiple of 16");
  }
  if (cfg.dense_mxfp8_shipped) {
    // The K-sliced MXFP8 matrices slice on whole 32-wide scale groups.
    if ((cfg.gdn_value_heads / world * cfg.gdn_value_head_dim) % kQwenMxfp8Group != 0)
      fail("the GDN out_proj's local columns must be a multiple of the MXFP8 group (32)");
    if ((cfg.num_attention_heads / world * cfg.head_dim) % kQwenMxfp8Group != 0)
      fail("the QSA o_proj's local columns must be a multiple of the MXFP8 group (32)");
    if ((cfg.shared_expert_intermediate_size / world) % kQwenMxfp8Group != 0)
      fail("the shared expert's intermediate slice must be a multiple of the MXFP8 group (32)");
  }
}

}  // namespace dgpp
