#include "models/dory/config.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <stdexcept>

namespace dgpp {
namespace {

[[noreturn]] void reject(std::string_view field, std::string_view why) {
  throw std::runtime_error(std::format("Dory config.{}: {}", field, why));
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
double require_double(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_number()) reject(field, "not a number");
  const double d = f.as_double();
  if (!std::isfinite(d)) reject(field, "not finite");
  return d;
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
std::string optional_string(const minijson::Value& v, std::string_view field,
                            const std::string& dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_string()) reject(field, "not a string");
  return std::string(f->as_string());
}

std::string read_file(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f)
    throw std::runtime_error(std::format("cannot open config {}: {}", path, std::strerror(errno)));
  std::string text;
  char buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
  std::fclose(f);
  return text;
}

DoryLayerKind kind_from_string(std::string_view field, const std::string& s) {
  if (s == "sliding_attention") return DoryLayerKind::SlidingAttention;
  if (s == "full_attention") return DoryLayerKind::FullAttention;
  if (s == "mlp") return DoryLayerKind::Mlp;
  reject(field, "must be sliding_attention, full_attention or mlp, got '" + s + "'");
}

// `layer_types`, or the reference's generated default (dory_config.py
// __post_init__): mlp on odd i, else full_attention when i % 8 == 6, else
// sliding_attention.
std::vector<DoryLayerKind> parse_layer_types(const minijson::Value& root, int layers) {
  std::vector<DoryLayerKind> kinds;
  const minijson::Value* lt = root.find("layer_types");
  if (!lt || lt->is_null()) {
    for (int i = 0; i < layers; ++i) {
      if (i % 2) {
        kinds.push_back(DoryLayerKind::Mlp);
      } else if (i % 8 == 6) {
        kinds.push_back(DoryLayerKind::FullAttention);
      } else {
        kinds.push_back(DoryLayerKind::SlidingAttention);
      }
    }
    return kinds;
  }
  if (!lt->is_array()) reject("layer_types", "not an array");
  for (const auto& item : lt->items()) {
    if (!item.is_string()) reject("layer_types", "non-string element");
    kinds.push_back(kind_from_string("layer_types", std::string(item.as_string())));
  }
  if (static_cast<int>(kinds.size()) != layers) reject("layer_types", "length differs from num_hidden_layers");
  return kinds;
}

// The per-layer-type RoPE contract: `rope_type` default (no scaling), a
// positive `rope_theta` per attention kind, and a full-width rotary slice
// (partial_rotary_factor 1 — the attention kernels rotate the whole head).
void parse_rope(const minijson::Value& root, DoryConfig& c) {
  if (const minijson::Value* rs = root.find("rope_scaling"); rs && !rs->is_null())
    reject("rope_scaling", "only the default rope is implemented");
  const double default_theta = 10000.0;  // DoryConfig.default_theta
  bool need_sliding = false, need_full = false;
  for (DoryLayerKind k : c.layer_kind) {
    if (k == DoryLayerKind::SlidingAttention) need_sliding = true;
    if (k == DoryLayerKind::FullAttention) need_full = true;
  }
  const minijson::Value* rp = root.find("rope_parameters");
  if (!rp || rp->is_null()) {
    // The reference generates the defaults per kind; both kinds fall back to
    // `default_theta` because no explicit theta is written.
    c.rope_theta_sliding = default_theta;
    c.rope_theta_full = default_theta;
    return;
  }
  if (!rp->is_object()) reject("rope_parameters", "not an object");
  bool saw_sliding = false, saw_full = false;
  for (const auto& m : rp->members()) {
    const std::string& key = m.key;
    const minijson::Value& entry = m.value;
    if (key != "sliding_attention" && key != "full_attention")
      reject("rope_parameters", "unexpected attention kind '" + key + "'");
    if (!entry.is_object()) reject("rope_parameters." + key, "not an object");
    const std::string type = optional_string(entry, "rope_type", optional_string(entry, "type", "default"));
    if (type != "default") reject("rope_parameters." + key + ".rope_type", "only the default rope is implemented, got '" + type + "'");
    const double theta = optional_double(entry, "rope_theta", default_theta);
    if (!(theta > 0)) reject("rope_parameters." + key + ".rope_theta", "must be positive");
    const double factor = optional_double(entry, "partial_rotary_factor", 1.0);
    // int(head_dim * factor) is the reference's rotary width (truncation).
    const int rotary_dim = static_cast<int>(static_cast<double>(c.head_dim) * factor);
    if (rotary_dim != c.head_dim)
      reject("rope_parameters." + key + ".partial_rotary_factor",
             "the attention kernels rotate the full head_dim");
    if (key == "sliding_attention") {
      c.rope_theta_sliding = theta;
      saw_sliding = true;
    } else {
      c.rope_theta_full = theta;
      saw_full = true;
    }
  }
  if (need_sliding && !saw_sliding) reject("rope_parameters", "sliding_attention is missing");
  if (need_full && !saw_full) reject("rope_parameters", "full_attention is missing");
}

// The release's quantization: modelopt W4A16_NVFP4, group_size 16, the
// embedding and the head on the ignore list. A release with activation
// scales (W4A4) has no kernel here, so it is rejected rather than guessed.
void parse_quantization(const minijson::Value& root, DoryConfig& c) {
  const minijson::Value* qc = root.find("quantization_config");
  if (!qc || qc->is_null())
    reject("quantization_config",
           "missing — the engine implements the modelopt W4A16_NVFP4 release as shipped");
  if (!qc->is_object()) reject("quantization_config", "not an object");
  const std::string method = optional_string(*qc, "quant_method", "");
  if (method != "modelopt")
    reject("quantization_config.quant_method", "only modelopt is implemented, got '" + method + "'");
  const std::string algo = optional_string(*qc, "quant_algo", "");
  if (algo != "W4A16_NVFP4")
    reject("quantization_config.quant_algo",
           "only W4A16_NVFP4 is implemented (there is no activation-scale path), got '" + algo + "'");
  // group_size lives under config_groups.<name>.weights in the release.
  int group = optional_int(*qc, "group_size", 0);
  if (const minijson::Value* cg = qc->find("config_groups"); cg && cg->is_object()) {
    for (const auto& m : cg->members()) {
      const std::string& name = m.key;
      const minijson::Value& group0 = m.value;
      if (!group0.is_object()) reject("quantization_config.config_groups." + name, "not an object");
      const minijson::Value* w = group0.find("weights");
      if (!w || !w->is_object()) reject("quantization_config.config_groups." + name + ".weights", "missing");
      const int bits = optional_int(*w, "num_bits", 4);
      if (bits != 4)
        reject("quantization_config.config_groups." + name + ".weights.num_bits",
               "only 4-bit weights are implemented");
      const std::string type = optional_string(*w, "type", "float");
      if (type != "float")
        reject("quantization_config.config_groups." + name + ".weights.type",
               "only float (e2m1) weights are implemented, got '" + type + "'");
      if (optional_bool(*w, "dynamic", false))
        reject("quantization_config.config_groups." + name + ".weights.dynamic",
               "dynamic weight quantization is not implemented");
      group = optional_int(*w, "group_size", group);
    }
  }
  if (group != 16) reject("quantization_config.group_size", "the NVFP4 kernels implement group_size 16");
  c.nvfp4_group = group;
  // The ignore list: the embedding and the head are BF16; anything else
  // ignored would be a class the loader does not implement in BF16.
  const minijson::Value* ig = qc->find("ignore");
  bool saw_embed = false, saw_head = false;
  if (ig && !ig->is_null()) {
    if (!ig->is_array()) reject("quantization_config.ignore", "not an array");
    for (const auto& item : ig->items()) {
      if (!item.is_string()) reject("quantization_config.ignore", "non-string entry");
      const std::string s(item.as_string());
      if (s == "backbone.embeddings") {
        saw_embed = true;
      } else if (s == "lm_head") {
        saw_head = true;
      } else {
        reject("quantization_config.ignore",
               "unexpected ignored module '" + s + "' (a BF16 class the loader does not implement)");
      }
    }
  }
  if (!saw_embed || !saw_head)
    reject("quantization_config.ignore", "the embedding and lm_head must be ignored (BF16)");
}

}  // namespace

DoryConfig DoryConfig::parse(const minijson::Value& root) {
  if (!root.is_object()) reject("", "root is not an object");
  DoryConfig c;
  const std::string model_type = optional_string(root, "model_type", "dory");
  if (model_type != "dory") reject("model_type", "expected dory, got " + model_type);

  // --- shape ---------------------------------------------------------------
  c.hidden_size = require_int(root, "hidden_size");
  c.vocab_size = require_int(root, "vocab_size");
  c.num_hidden_layers = require_int(root, "num_hidden_layers");
  c.intermediate_size = require_int(root, "intermediate_size");
  c.max_position_embeddings = require_int(root, "max_position_embeddings");
  c.hidden_act = optional_string(root, "hidden_act", "silu");
  c.tie_word_embeddings = optional_bool(root, "tie_word_embeddings", false);
  c.attention_bias = optional_bool(root, "attention_bias", false);
  c.mlp_bias = optional_bool(root, "mlp_bias", false);
  if (c.hidden_size <= 0 || c.hidden_size % 16 != 0)
    reject("hidden_size", "must be a positive multiple of 16 (the NVFP4 group)");
  if (c.vocab_size <= 0) reject("vocab_size", "must be positive");
  if (c.num_hidden_layers <= 0) reject("num_hidden_layers", "must be positive");
  if (c.intermediate_size <= 0 || c.intermediate_size % 16 != 0)
    reject("intermediate_size", "must be a positive multiple of 16 (the NVFP4 group)");
  if (c.max_position_embeddings <= 0) reject("max_position_embeddings", "must be positive");
  if (c.hidden_act != "silu") reject("hidden_act", "only silu is implemented, got " + c.hidden_act);
  if (c.tie_word_embeddings) reject("tie_word_embeddings", "tied embeddings are not implemented");
  if (c.attention_bias || c.mlp_bias) reject("attention_bias", "bias-free projections are required");

  // --- tokens ---------------------------------------------------------------
  // Optional here: the release names its stop tokens in
  // generation_config.json ([2, 11]), which GlmGenerationDefaults reads as
  // the serving authority; this field is the fixtures' and older exports'
  // fallback, like the other families'.
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

  // --- the nGPT scales ------------------------------------------------------
  // The release writes a fractional initializer_range (0.0177), so it is
  // parsed as a double; it divides every learned scale and is required.
  c.initializer_range = static_cast<float>(require_double(root, "initializer_range"));
  if (!(c.initializer_range > 0)) reject("initializer_range", "must be positive");
  c.ngpt_alpha_init = static_cast<float>(optional_double(root, "ngpt_alpha_init", 0.05));
  if (!(c.ngpt_alpha_init > 0)) reject("ngpt_alpha_init", "must be positive");
  c.attention_output_gate_logit_scale_init =
      static_cast<float>(optional_double(root, "attention_output_gate_logit_scale_init", 0.0));

  // --- the loop -------------------------------------------------------------
  c.n_input_layers = require_int(root, "n_input_layers");
  c.n_output_layers = require_int(root, "n_output_layers");
  c.n_recurrent_layers = optional_int(root, "n_recurrent_layers", -1);
  if (c.n_recurrent_layers < 0)
    c.n_recurrent_layers = c.num_hidden_layers - c.n_input_layers - c.n_output_layers;
  c.n_recurrent_loops = optional_int(root, "n_recurrent_loops", 1);
  if (c.n_input_layers < 0 || c.n_output_layers < 0 || c.n_recurrent_layers < 0 ||
      c.n_input_layers + c.n_recurrent_layers + c.n_output_layers != c.num_hidden_layers)
    reject("n_input_layers",
           "the input, recurrent and output blocks must be non-negative and sum to num_hidden_layers");
  if (c.n_recurrent_loops < 1) reject("n_recurrent_loops", "must be a positive integer");
  // The release's blocks are whole attention+mlp pairs, so every boundary
  // falls on an even index (the layer_types pattern is indexed from 0).
  if (c.n_input_layers % 2 != 0 || (c.n_input_layers + c.n_recurrent_layers) % 2 != 0)
    reject("n_input_layers", "the block boundaries must fall on an even layer index");

  // --- attention ------------------------------------------------------------
  c.num_attention_heads = require_int(root, "num_attention_heads");
  c.num_key_value_heads = require_int(root, "num_key_value_heads");
  c.head_dim = require_int(root, "head_dim");
  if (c.num_attention_heads <= 0 || c.num_key_value_heads <= 0 || c.head_dim <= 0)
    reject("num_attention_heads", "head counts and head_dim must be positive");
  if (c.num_attention_heads % c.num_key_value_heads != 0)
    reject("num_key_value_heads", "must divide num_attention_heads");
  if (c.head_dim % 2 != 0) reject("head_dim", "must be even (the rotary pairs)");
  c.layer_kind = parse_layer_types(root, c.num_hidden_layers);

  c.sliding_window = optional_int(root, "sliding_window", 0);
  if (c.num_attention_layers() == 0) reject("layer_types", "no attention layer");
  bool any_sliding = false;
  for (DoryLayerKind k : c.layer_kind) any_sliding = any_sliding || k == DoryLayerKind::SlidingAttention;
  if (any_sliding && c.sliding_window <= 0)
    reject("sliding_window", "must be positive with sliding_attention layers");
  parse_rope(root, c);

  parse_quantization(root, c);
  return c;
}

DoryConfig DoryConfig::from_json_file(const std::string& path) {
  const std::string json = read_file(path);
  const auto parsed = minijson::parse(json);
  return parse(parsed.root);
}

}  // namespace dgpp
