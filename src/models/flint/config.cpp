#include "models/flint/config.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <limits>
#include <set>
#include <stdexcept>

namespace dgpp {
namespace {

[[noreturn]] void reject(std::string_view field, std::string_view why) {
  throw std::runtime_error(std::format("Flint config.{}: {}", field, why));
}

const minijson::Value& require(const minijson::Value& v, std::string_view field) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) reject(field, "missing");
  return *f;
}
int require_int(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_number()) reject(field, "not a number");
  return static_cast<int>(f.as_int());
}
int optional_int(const minijson::Value& v, std::string_view field, int dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_number()) reject(field, "not a number");
  return static_cast<int>(f->as_int());
}
double optional_double(const minijson::Value& v, std::string_view field, double dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_number()) reject(field, "not a number");
  const double d = f->as_double();
  if (!std::isfinite(d)) reject(field, "not finite");
  return d;
}
bool optional_bool(const minijson::Value& v, std::string_view field, bool dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_bool()) reject(field, "not a bool");
  return f->as_bool();
}
std::string optional_string(const minijson::Value& v, std::string_view field, const std::string& dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_string()) reject(field, "not a string");
  return std::string(f->as_string());
}
std::vector<int> int_list(const minijson::Value& v, std::string_view field, bool required) {
  const minijson::Value* f = v.find(field);
  std::vector<int> out;
  if (!f || f->is_null()) {
    if (required) reject(field, "missing");
    return out;
  }
  if (!f->is_array()) reject(field, "not an array");
  for (const auto& item : f->items()) {
    if (!item.is_number()) reject(field, "non-numeric element");
    out.push_back(static_cast<int>(item.as_int()));
  }
  return out;
}

std::string read_file(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error(std::format("cannot open config {}: {}", path, std::strerror(errno)));
  std::string text;
  char buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
  std::fclose(f);
  return text;
}

// The release's contract: quant_method dgpp_mixed, the routed experts of
// every MoE layer in fp8 (128 x 128 blocks, the fp32 multiplicative
// weight_scale_inv) or nvfp4 (group 16, e4m3 scales, the fp32 reciprocal
// global) per projection as precision_by_layer names them, nothing kept
// in BF16, every other weight BF16.
void parse_quantization(const minijson::Value& root, FlintTextConfig& c) {
  const minijson::Value* qc = root.find("quantization_config");
  if (!qc || qc->is_null())
    reject("quantization_config",
           "missing — the engine implements the FP8 / NVFP4 mixed export as shipped; the BF16 source "
           "has no expert path here");
  if (!qc->is_object()) reject("quantization_config", "not an object");
  if (const std::string method = optional_string(*qc, "quant_method", ""); method != "dgpp_mixed")
    reject("quantization_config.quant_method", "only dgpp_mixed is implemented, got '" + method + "'");
  if (const std::string act = optional_string(*qc, "activations", "bfloat16"); act != "bfloat16")
    reject("quantization_config.activations", "only bfloat16 activations are implemented, got '" + act + "'");
  const minijson::Value* re = qc->find("routed_experts");
  if (!re || !re->is_object()) reject("quantization_config.routed_experts", "missing");
  if (const minijson::Value* f8 = re->find("fp8_format"); f8 && f8->is_object()) {
    if (const std::string d = optional_string(*f8, "weight_dtype", "float8_e4m3fn"); d != "float8_e4m3fn")
      reject("quantization_config.routed_experts.fp8_format.weight_dtype", "only float8_e4m3fn is implemented");
    const std::vector<int> wb = int_list(*f8, "weight_block_size", false);
    if (!wb.empty() && (wb.size() != 2 || wb[0] != 128 || wb[1] != 128))
      reject("quantization_config.routed_experts.fp8_format.weight_block_size", "the fp8 kernels implement 128 x 128 blocks");
    if (const std::string s = optional_string(*f8, "scale_dtype", "float32"); s != "float32")
      reject("quantization_config.routed_experts.fp8_format.scale_dtype", "only float32 scales are implemented");
    if (const std::string s = optional_string(*f8, "scale_suffix", "weight_scale_inv"); s != "weight_scale_inv")
      reject("quantization_config.routed_experts.fp8_format.scale_suffix", "the loader binds weight_scale_inv");
  }
  if (const minijson::Value* f4 = re->find("nvfp4_format"); f4 && f4->is_object()) {
    if (const int g = optional_int(*f4, "group_size", 16); g != 16)
      reject("quantization_config.routed_experts.nvfp4_format.group_size", "the NVFP4 kernels implement group 16");
    if (const std::string d = optional_string(*f4, "weight_dtype", "float4_e2m1"); d != "float4_e2m1")
      reject("quantization_config.routed_experts.nvfp4_format.weight_dtype", "only float4_e2m1 is implemented");
    if (const std::string s = optional_string(*f4, "scale_dtype", "float8_e4m3fn"); s != "float8_e4m3fn")
      reject("quantization_config.routed_experts.nvfp4_format.scale_dtype", "only e4m3 block scales are implemented");
    if (const std::string s = optional_string(*f4, "global_scale_dtype", "float32"); s != "float32")
      reject("quantization_config.routed_experts.nvfp4_format.global_scale_dtype", "only an fp32 global is implemented");
  }
  c.fp8_block = 128;
  c.nvfp4_group = 16;
  if (const std::vector<int> kept = int_list(*re, "retained_bf16_layers", false); !kept.empty())
    reject("quantization_config.routed_experts.retained_bf16_layers", "BF16 routed experts are not implemented");
  // The listed layers must be exactly the MoE layers, the projections the
  // three, and every MoE layer must name every projection's format.
  const std::vector<int> layers = int_list(*re, "layers", true);
  std::set<int> listed(layers.begin(), layers.end());
  for (int l = 0; l < c.num_hidden_layers; ++l)
    if (c.is_moe_layer(l) != (listed.count(l) > 0))
      reject("quantization_config.routed_experts.layers", "differs from the MoE layers (layer " + std::to_string(l) + ")");
  {
    const minijson::Value* pr = re->find("projections");
    if (!pr || !pr->is_array() || pr->items().size() != 3)
      reject("quantization_config.routed_experts.projections", "expected [gate, up, down]");
    const char* want[3] = {"gate", "up", "down"};
    for (int i = 0; i < 3; ++i)
      if (!pr->items()[static_cast<size_t>(i)].is_string() || pr->items()[static_cast<size_t>(i)].as_string() != want[i])
        reject("quantization_config.routed_experts.projections", "expected [gate, up, down]");
  }
  const minijson::Value* pbl = re->find("precision_by_layer");
  if (!pbl || !pbl->is_object()) reject("quantization_config.routed_experts.precision_by_layer", "missing");
  c.expert_fp4_mask.assign(static_cast<size_t>(c.num_hidden_layers), 0);
  for (int l = 0; l < c.num_hidden_layers; ++l) {
    if (!c.is_moe_layer(l)) continue;
    const minijson::Value* entry = pbl->find(std::to_string(l));
    if (!entry || !entry->is_object())
      reject("quantization_config.routed_experts.precision_by_layer", "no entry for MoE layer " + std::to_string(l));
    const char* names[3] = {"gate", "up", "down"};
    uint8_t mask = 0;
    for (int w = 0; w < 3; ++w) {
      const std::string fmt = optional_string(*entry, names[w], "");
      if (fmt == "nvfp4") mask |= static_cast<uint8_t>(1u << w);
      else if (fmt != "fp8")
        reject("quantization_config.routed_experts.precision_by_layer",
               std::string("layer ") + std::to_string(l) + " " + names[w] + ": only fp8 and nvfp4 are implemented, got '" + fmt + "'");
    }
    c.expert_fp4_mask[static_cast<size_t>(l)] = mask;
  }
}

}  // namespace

FlintTextConfig FlintTextConfig::parse(const minijson::Value& root) {
  if (!root.is_object()) reject("", "root is not an object");
  FlintTextConfig c;
  const std::string model_type = optional_string(root, "model_type", "flint");
  if (model_type != "flint") reject("model_type", "expected flint, got " + model_type);

  c.hidden_size = require_int(root, "hidden_size");
  c.vocab_size = require_int(root, "vocab_size");
  c.num_hidden_layers = require_int(root, "num_hidden_layers");
  c.rms_norm_eps = static_cast<float>(optional_double(root, "rms_norm_eps", 1e-6));
  c.tie_word_embeddings = optional_bool(root, "tie_word_embeddings", false);
  c.hidden_act = optional_string(root, "hidden_act", "silu");
  c.max_position_embeddings = require_int(root, "max_position_embeddings");
  if (c.hidden_size <= 0 || c.hidden_size % 128 != 0)
    reject("hidden_size", "must be a positive multiple of 128 (the fp8 scale block)");
  if (c.vocab_size <= 0) reject("vocab_size", "must be positive");
  if (c.num_hidden_layers <= 0) reject("num_hidden_layers", "must be positive");
  if (c.hidden_act != "silu") reject("hidden_act", "only silu is implemented, got " + c.hidden_act);
  if (c.tie_word_embeddings) reject("tie_word_embeddings", "tied embeddings are not implemented");
  if (c.max_position_embeddings <= 0) reject("max_position_embeddings", "must be positive");
  if (!(c.rms_norm_eps > 0)) reject("rms_norm_eps", "must be positive");

  // --- the layer kinds ---------------------------------------------------------
  {
    const minijson::Value& lt = require(root, "layer_types");
    if (!lt.is_array()) reject("layer_types", "not an array");
    if (static_cast<int>(lt.items().size()) != c.num_hidden_layers) reject("layer_types", "length differs from num_hidden_layers");
    for (const auto& item : lt.items()) {
      if (!item.is_string()) reject("layer_types", "non-string element");
      const std::string_view s = item.as_string();
      if (s == "sliding_attention") c.swa_layer.push_back(1);
      else if (s == "full_attention") c.swa_layer.push_back(0);
      else reject("layer_types", "unknown layer type '" + std::string(s) + "'");
    }
  }
  c.rope_layer.assign(static_cast<size_t>(c.num_hidden_layers), 1);
  for (const int l : int_list(root, "no_rope_layer_indices", false)) {
    if (l < 0 || l >= c.num_hidden_layers) reject("no_rope_layer_indices", "index out of range");
    c.rope_layer[static_cast<size_t>(l)] = 0;
  }
  c.moe_layer.assign(static_cast<size_t>(c.num_hidden_layers), 1);
  for (const int l : int_list(root, "mlp_only_layers", false)) {
    if (l < 0 || l >= c.num_hidden_layers) reject("mlp_only_layers", "index out of range");
    c.moe_layer[static_cast<size_t>(l)] = 0;
  }
  if (const int step = optional_int(root, "decoder_sparse_step", 1); step != 1)
    reject("decoder_sparse_step", "only 1 is implemented (every layer outside mlp_only_layers is an MoE layer)");
  if (c.num_moe_layers() == 0) reject("mlp_only_layers", "no MoE layer");

  // --- tokens ---------------------------------------------------------------
  if (const minijson::Value* eos = root.find("eos_token_id"); eos && !eos->is_null()) {
    if (eos->is_array()) {
      for (const auto& item : eos->items()) {
        if (!item.is_number()) reject("eos_token_id", "non-numeric element");
        c.eos_token_ids.push_back(item.as_int());
      }
    } else if (eos->is_number()) {
      c.eos_token_ids.push_back(eos->as_int());
    } else {
      reject("eos_token_id", "not a number or array");
    }
    for (int64_t id : c.eos_token_ids)
      if (id < 0 || id >= c.vocab_size) reject("eos_token_id", "id outside [0, vocab_size)");
  }
  if (c.eos_token_ids.empty()) reject("eos_token_id", "missing");
  c.bos_token_id = optional_int(root, "bos_token_id", -1);
  c.pad_token_id = optional_int(root, "pad_token_id", -1);

  // --- attention ------------------------------------------------------------
  c.num_attention_heads = require_int(root, "num_attention_heads");
  c.num_key_value_heads = require_int(root, "num_key_value_heads");
  c.head_dim = optional_int(root, "head_dim", c.hidden_size / std::max(c.num_attention_heads, 1));
  c.attention_bias = optional_bool(root, "attention_bias", false);
  c.rope_theta = optional_double(root, "rope_theta", 10000.0);
  if (const minijson::Value* rs = root.find("rope_scaling"); rs && !rs->is_null())
    reject("rope_scaling", "only the default rope is implemented");
  if (const minijson::Value* rp = root.find("rope_parameters"); rp && rp->is_object()) {
    if (optional_string(*rp, "rope_type", optional_string(*rp, "type", "default")) != "default")
      reject("rope_parameters.rope_type", "only the default rope is implemented");
    c.rope_theta = optional_double(*rp, "rope_theta", c.rope_theta);
  }
  if (c.num_attention_heads <= 0 || c.num_key_value_heads <= 0) reject("num_attention_heads", "head counts must be positive");
  if (c.num_attention_heads % c.num_key_value_heads != 0) reject("num_key_value_heads", "must divide num_attention_heads");
  if (c.head_dim != 128) reject("head_dim", "the attention kernels implement 128-wide heads");
  if (c.attention_bias) reject("attention_bias", "biased projections are not implemented");
  if (!(c.rope_theta > 0)) reject("rope_theta", "must be positive");
  c.sliding_window = optional_int(root, "sliding_window", 0);
  if (c.num_swa_layers() > 0) {
    if (!optional_bool(root, "use_sliding_window", true)) reject("use_sliding_window", "false with sliding_attention layers");
    if (c.sliding_window <= 0) reject("sliding_window", "must be positive with sliding_attention layers");
  }
  if (const std::string st = optional_string(root, "softmax_type", "learnable"); st == "learnable") c.sink = true;
  else reject("softmax_type", "only the learnable (sink) softmax is implemented, got " + st);

  // --- MLP / MoE --------------------------------------------------------------
  c.intermediate_size = require_int(root, "intermediate_size");
  c.moe_intermediate_size = require_int(root, "moe_intermediate_size");
  c.shared_expert_intermediate_size = require_int(root, "shared_expert_intermediate_size");
  c.n_routed_experts = require_int(root, "num_experts");
  c.num_experts_per_tok = require_int(root, "num_experts_per_tok");
  c.norm_topk_prob = optional_bool(root, "norm_topk_prob", true);
  if (c.intermediate_size <= 0 || c.intermediate_size % 8 != 0) reject("intermediate_size", "must be a positive multiple of 8");
  if (c.moe_intermediate_size <= 0 || c.moe_intermediate_size % 128 != 0)
    reject("moe_intermediate_size", "must be a positive multiple of 128 (the fp8 scale block)");
  if (c.shared_expert_intermediate_size <= 0 || c.shared_expert_intermediate_size % 8 != 0)
    reject("shared_expert_intermediate_size", "must be a positive multiple of 8");
  if (c.n_routed_experts <= 0 || c.n_routed_experts > 4096) reject("num_experts", "must be in [1, 4096]");
  if (c.num_experts_per_tok <= 0 || c.num_experts_per_tok > 16 || c.num_experts_per_tok > c.n_routed_experts)
    reject("num_experts_per_tok", "must be in [1, min(num_experts, 16)]");
  if (const std::string sf = optional_string(root, "router_score_function", "sigmoid"); sf != "sigmoid")
    reject("router_score_function", "only sigmoid is implemented");

  parse_quantization(root, c);
  return c;
}

FlintTextConfig FlintTextConfig::from_json_file(const std::string& path) {
  const std::string json = read_file(path);
  const auto parsed = minijson::parse(json);
  return parse(parsed.root);
}

GlmMoeConfig FlintTextConfig::moe_config(int local_inter) const {
  GlmMoeConfig m;
  m.hidden = hidden_size;
  m.inter = local_inter;
  m.n_experts = n_routed_experts;
  m.top_k = num_experts_per_tok;
  m.n_shared_experts = 0;
  m.routed_scaling_factor = 1.0f;
  m.norm_topk_prob = norm_topk_prob;
  m.swiglu_limit = std::numeric_limits<float>::infinity();  // FlintMLP: no clamps
  m.router_mode = MoeRouterMode::SigmoidBias;
  GlmMoeConfig::validate_config(m);
  return m;
}

}  // namespace dgpp
