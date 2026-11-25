#include "models/inkling2/config.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <set>
#include <stdexcept>

namespace dgpp {
namespace {

[[noreturn]] void reject(std::string_view field, std::string_view why) {
  throw std::runtime_error(std::format("Inkling2 config.{}: {}", field, why));
}

const minijson::Value& require(const minijson::Value& v, std::string_view field, std::string_view path) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) reject(path, "missing");
  return *f;
}
int require_int(const minijson::Value& v, std::string_view field, std::string_view path) {
  const minijson::Value& f = require(v, field, path);
  if (!f.is_number()) reject(path, "not a number");
  return static_cast<int>(f.as_int());
}
int optional_int(const minijson::Value& v, std::string_view field, std::string_view path, int dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_number()) reject(path, "not a number");
  return static_cast<int>(f->as_int());
}
int64_t optional_int64(const minijson::Value& v, std::string_view field, std::string_view path, int64_t dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_number()) reject(path, "not a number");
  return f->as_int();
}
double optional_double(const minijson::Value& v, std::string_view field, std::string_view path, double dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_number()) reject(path, "not a number");
  const double d = f->as_double();
  if (!std::isfinite(d)) reject(path, "not finite");
  return d;
}
bool optional_bool(const minijson::Value& v, std::string_view field, std::string_view path, bool dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_bool()) reject(path, "not a bool");
  return f->as_bool();
}
std::string optional_string(const minijson::Value& v, std::string_view field, std::string_view path,
                            const std::string& dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_string()) reject(path, "not a string");
  return std::string(f->as_string());
}
std::vector<int> int_list(const minijson::Value& v, std::string_view field, std::string_view path, bool required) {
  const minijson::Value* f = v.find(field);
  std::vector<int> out;
  if (!f || f->is_null()) {
    if (required) reject(path, "missing");
    return out;
  }
  if (!f->is_array()) reject(path, "not an array");
  for (const auto& item : f->items()) {
    if (!item.is_number()) reject(path, "non-numeric element");
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

std::vector<uint8_t> layer_flags(const std::vector<int>& ids, int layers, std::string_view path) {
  std::vector<uint8_t> flags(static_cast<size_t>(layers), 0);
  for (const int l : ids) {
    if (l < 0 || l >= layers) reject(path, "index out of range");
    if (flags[static_cast<size_t>(l)]) reject(path, "repeated index");
    flags[static_cast<size_t>(l)] = 1;
  }
  return flags;
}

// hf_quant_config.json: `model.parameter_groups` names every tensor's
// precision. The engine implements the release's contract: the routed
// experts in NVFP4 (E2M1 pairs, one E4M3 scale per 16 values along K, the
// per-expert amax over one denominator) or MXFP8 (E4M3 values, one
// power-of-two scale per 32 values along K), every other tensor BF16 or
// FP32 as stored.
void parse_quantization(const minijson::Value& quant, Inkling2Config& c) {
  if (!quant.is_object()) reject("hf_quant_config", "root is not an object");
  if (const std::string m = optional_string(quant, "quant_method", "hf_quant_config.quant_method", "inkling2");
      m != "inkling2")
    reject("hf_quant_config.quant_method", "only inkling2 is implemented, got '" + m + "'");
  const minijson::Value& model = require(quant, "model", "hf_quant_config.model");
  if (!model.is_object()) reject("hf_quant_config.model", "not an object");
  const minijson::Value& groups = require(model, "parameter_groups", "hf_quant_config.model.parameter_groups");
  if (!groups.is_array()) reject("hf_quant_config.model.parameter_groups", "not an array");
  const size_t slots = static_cast<size_t>(c.num_hidden_layers) * 2;
  std::vector<int8_t> seen(slots, -1);
  for (const auto& group : groups.items()) {
    if (!group.is_object()) reject("hf_quant_config.model.parameter_groups", "non-object element");
    const std::string compute = optional_string(group, "compute_dtype", "parameter_groups.compute_dtype", "");
    if (compute == "float32" || compute == "bfloat16") continue;
    Inkling2ExpertFormat format;
    if (compute == "nvfp4") {
      format = Inkling2ExpertFormat::Nvfp4;
      const std::vector<int> block = int_list(group, "weight_block_shape", "parameter_groups.weight_block_shape", true);
      if (block.size() != 2 || block[0] != 1 || block[1] != 16)
        reject("parameter_groups.weight_block_shape", "the NVFP4 kernels implement one scale per 16 values along K");
      if (const std::string s = optional_string(group, "weight_scale_format", "parameter_groups.weight_scale_format",
                                                "amax_denominator");
          s != "amax_denominator")
        reject("parameter_groups.weight_scale_format", "only amax_denominator is implemented, got '" + s + "'");
      if (const minijson::Value* act = group.find("required_activation_quantization"); act && act->is_object()) {
        const char* path = "parameter_groups.required_activation_quantization";
        if (optional_string(*act, "format", path, "nvfp4_e2m1") != "nvfp4_e2m1") reject(path, "format is not nvfp4_e2m1");
        if (optional_int(*act, "block_size", path, 16) != 16) reject(path, "block_size is not 16");
        if (optional_string(*act, "amax_type", path, "per_token") != "per_token") reject(path, "amax_type is not per_token");
        if (optional_string(*act, "scale_dtype", path, "float8_e4m3fn") != "float8_e4m3fn")
          reject(path, "scale_dtype is not float8_e4m3fn");
        if (optional_string(*act, "scale_selection_rule", path, "fouroversix_mae") != "fouroversix_mae")
          reject(path, "scale_selection_rule is not fouroversix_mae");
        if (optional_string(*act, "rounding", path, "nearest") != "nearest") reject(path, "rounding is not nearest");
      }
    } else if (compute == "mxfp8") {
      format = Inkling2ExpertFormat::Mxfp8;
      if (const minijson::Value* act = group.find("required_activation_quantization"); act && act->is_object()) {
        const char* path = "parameter_groups.required_activation_quantization";
        if (optional_string(*act, "format", path, "mxfp8_e4m3") != "mxfp8_e4m3") reject(path, "format is not mxfp8_e4m3");
        if (optional_int(*act, "block_size", path, 32) != 32) reject(path, "block_size is not 32");
        if (optional_string(*act, "scale_dtype", path, "ue8m0") != "ue8m0") reject(path, "scale_dtype is not ue8m0");
        if (optional_string(*act, "rounding", path, "nearest") != "nearest") reject(path, "rounding is not nearest");
      }
    } else {
      reject("parameter_groups.compute_dtype", "only float32, bfloat16, nvfp4 and mxfp8 are implemented, got '" + compute + "'");
    }
    const minijson::Value& tensors = require(group, "tensors", "parameter_groups.tensors");
    if (!tensors.is_array()) reject("parameter_groups.tensors", "not an array");
    for (const auto& item : tensors.items()) {
      if (!item.is_string()) reject("parameter_groups.tensors", "non-string element");
      const std::string name(item.as_string());
      // model.llm.layers.<N>.mlp.experts.{w13,w2}_weight
      static constexpr std::string_view prefix = "model.llm.layers.";
      int layer = -1, projection = -1;
      if (name.rfind(prefix, 0) == 0) {
        const size_t dot = name.find('.', prefix.size());
        if (dot != std::string::npos) {
          const std::string_view tail = std::string_view(name).substr(dot);
          if (tail == ".mlp.experts.w13_weight") projection = 0;
          else if (tail == ".mlp.experts.w2_weight") projection = 1;
          if (projection >= 0) {
            layer = 0;
            for (size_t i = prefix.size(); i < dot; ++i) {
              if (name[i] < '0' || name[i] > '9') reject("parameter_groups.tensors", "unreadable layer in " + name);
              layer = layer * 10 + (name[i] - '0');
            }
          }
        }
      }
      if (projection < 0) reject("parameter_groups.tensors", "a quantized tensor that is not a routed expert weight: " + name);
      if (layer >= c.num_hidden_layers) reject("parameter_groups.tensors", "layer out of range in " + name);
      const size_t slot = static_cast<size_t>(layer) * 2 + static_cast<size_t>(projection);
      if (seen[slot] >= 0) reject("parameter_groups.tensors", "listed twice: " + name);
      seen[slot] = static_cast<int8_t>(format);
    }
  }
  c.expert_format.resize(slots);
  for (size_t s = 0; s < slots; ++s) {
    if (seen[s] < 0)
      reject("parameter_groups", std::format("layer {} {} is not a quantized tensor: BF16 routed experts are not implemented",
                                             s / 2, s % 2 ? "w2_weight" : "w13_weight"));
    c.expert_format[s] = static_cast<Inkling2ExpertFormat>(seen[s]);
  }
  for (int l = 0; l < c.num_hidden_layers; ++l)
    if (c.format_of(l, 0) != c.format_of(l, 1))
      reject("parameter_groups", std::format("layer {} mixes expert formats between w13 and w2: not implemented", l));
}

void parse_draft(const minijson::Value& root, Inkling2Config& c) {
  const minijson::Value* m = root.find("mtp_config");
  if (!m || m->is_null()) return;
  if (!m->is_object()) reject("mtp_config", "not an object");
  Inkling2DraftConfig& d = c.draft;
  d.present = true;
  d.hidden_size = require_int(*m, "hidden_size", "mtp_config.hidden_size");
  d.num_hidden_layers = require_int(*m, "num_hidden_layers", "mtp_config.num_hidden_layers");
  d.num_attention_heads = require_int(*m, "num_attention_heads", "mtp_config.num_attention_heads");
  d.num_key_value_heads = require_int(*m, "num_key_value_heads", "mtp_config.num_key_value_heads");
  d.head_dim = require_int(*m, "head_dim", "mtp_config.head_dim");
  d.intermediate_size = require_int(*m, "intermediate_size", "mtp_config.intermediate_size");
  d.max_position_embeddings = optional_int64(*m, "max_position_embeddings", "mtp_config.max_position_embeddings",
                                             c.max_position_embeddings);
  d.rope_theta = optional_double(*m, "rope_theta", "mtp_config.rope_theta", 1e6);
  d.rms_norm_eps = static_cast<float>(optional_double(*m, "rms_norm_eps", "mtp_config.rms_norm_eps", 1e-6));
  d.sliding_window_size = require_int(*m, "sliding_window_size", "mtp_config.sliding_window_size");
  d.block_size = require_int(*m, "block_size", "mtp_config.block_size");
  d.markov_rank = require_int(*m, "markov_rank", "mtp_config.markov_rank");
  d.target_layer_ids = int_list(*m, "target_layer_ids", "mtp_config.target_layer_ids", true);
  if (d.hidden_size != c.hidden_size) reject("mtp_config.hidden_size", "differs from the target's hidden_size");
  if (d.num_hidden_layers <= 0 || d.num_hidden_layers > 8) reject("mtp_config.num_hidden_layers", "must be in [1, 8]");
  d.local_layer = layer_flags(int_list(*m, "local_layer_ids", "mtp_config.local_layer_ids", false), d.num_hidden_layers,
                              "mtp_config.local_layer_ids");
  if (d.head_dim != 128) reject("mtp_config.head_dim", "the block attention kernel implements 128-wide heads");
  if (d.num_attention_heads <= 0 || d.num_key_value_heads <= 0 || d.num_attention_heads % d.num_key_value_heads != 0)
    reject("mtp_config.num_key_value_heads", "must divide num_attention_heads");
  if (d.num_attention_heads / d.num_key_value_heads > 16)
    reject("mtp_config.num_attention_heads", "the block attention kernel implements at most 16 query heads per key head");
  if (d.intermediate_size <= 0 || d.intermediate_size % 8 != 0) reject("mtp_config.intermediate_size", "must be a positive multiple of 8");
  if (optional_bool(*m, "q_bias", "mtp_config.q_bias", false)) reject("mtp_config.q_bias", "biased projections are not implemented");
  if (optional_bool(*m, "o_bias", "mtp_config.o_bias", false)) reject("mtp_config.o_bias", "biased projections are not implemented");
  if (!(d.rope_theta > 0)) reject("mtp_config.rope_theta", "must be positive");
  if (!(d.rms_norm_eps > 0)) reject("mtp_config.rms_norm_eps", "must be positive");
  if (d.sliding_window_size <= 0) reject("mtp_config.sliding_window_size", "must be positive");
  if (d.block_size < 2 || d.block_size > 8)
    reject("mtp_config.block_size", "must be in [2, 8] (the anchor row plus at most seven drafts)");
  if (d.markov_rank <= 0 || d.markov_rank > 512 || (d.markov_rank & (d.markov_rank - 1)) != 0)
    reject("mtp_config.markov_rank", "must be a power of two up to 512");
  if (const std::string t = optional_string(*m, "markov_head_type", "mtp_config.markov_head_type", "rnn_hidden");
      t != "rnn_hidden")
    reject("mtp_config.markov_head_type", "only rnn_hidden is implemented, got '" + t + "'");
  // Sorted target visits, then the visit count for the final-norm output.
  const std::vector<int>& ids = d.target_layer_ids;
  if (ids.size() < 2 || ids.size() > 8) reject("mtp_config.target_layer_ids", "expected 2 to 8 ids");
  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] < 0 || (i > 0 && ids[i] <= ids[i - 1])) reject("mtp_config.target_layer_ids", "must be sorted and unique");
  }
  if (ids.back() != c.num_visits())
    reject("mtp_config.target_layer_ids", std::format("the last id must be the target's visit count {} (the final-norm output)",
                                                      c.num_visits()));
}

void parse_vision(const minijson::Value& root, Inkling2Config& c) {
  const minijson::Value* v = root.find("vision_config");
  if (!v || v->is_null()) return;
  if (!v->is_object()) reject("vision_config", "not an object");
  Inkling2VisionConfig& o = c.vision;
  o.present = true;
  o.text_hidden_size = optional_int(*v, "text_hidden_size", "vision_config.text_hidden_size", c.hidden_size);
  o.hidden_size = require_int(*v, "hidden_size", "vision_config.hidden_size");
  o.num_hidden_layers = require_int(*v, "num_hidden_layers", "vision_config.num_hidden_layers");
  o.num_attention_heads = require_int(*v, "num_attention_heads", "vision_config.num_attention_heads");
  o.native_patch_size = require_int(*v, "native_patch_size", "vision_config.native_patch_size");
  o.temporal_patch_size = require_int(*v, "temporal_patch_size", "vision_config.temporal_patch_size");
  o.adapter_merge_size = require_int(*v, "adapter_merge_size", "vision_config.adapter_merge_size");
  o.projection_hidden_dim = require_int(*v, "projection_hidden_dim", "vision_config.projection_hidden_dim");
  o.posemb_grid_size = require_int(*v, "posemb_grid_size", "vision_config.posemb_grid_size");
  o.mlp_ratio = optional_double(*v, "mlp_ratio", "vision_config.mlp_ratio", 4.0);
  o.layer_norm_eps = static_cast<float>(optional_double(*v, "layer_norm_eps", "vision_config.layer_norm_eps", 1e-5));
  o.projector_bias = optional_bool(*v, "projector_bias", "vision_config.projector_bias", false);
  if (o.text_hidden_size != c.hidden_size) reject("vision_config.text_hidden_size", "differs from the text hidden_size");
  if (o.hidden_size <= 0 || o.num_hidden_layers <= 0 || o.num_attention_heads <= 0 ||
      o.hidden_size % o.num_attention_heads != 0)
    reject("vision_config.hidden_size", "must be a positive multiple of num_attention_heads");
  if (o.native_patch_size <= 0 || o.temporal_patch_size <= 0 || o.adapter_merge_size <= 0 || o.posemb_grid_size <= 0 ||
      o.projection_hidden_dim <= 0)
    reject("vision_config", "patch, merge, grid and projector sizes must be positive");
  if (o.projector_bias) reject("vision_config.projector_bias", "a biased projector is not implemented");
}

}  // namespace

Inkling2Visit Inkling2Config::visit(int v) const {
  if (v < core_start) return {v, 0};
  const int core = core_layers();
  if (v < core_start + num_passes * core) return {core_start + (v - core_start) % core, (v - core_start) / core};
  if (v < num_visits()) return {v - (num_passes - 1) * core, 0};
  // The reference readout repeats the layers after the core.
  return {core_end + (v - num_visits()), 0};
}

void Inkling2Config::apply_loop_options(const Inkling2LoopOptions& o) {
  const auto bad = [](const char* key, const std::string& why) {
    throw std::invalid_argument(std::string("engine.") + key + ": " + why);
  };
  if (!o.guidance) {
    if (o.strength != 0.0f) bad("loop_guidance_strength", "set without engine.loop_guidance");
    return;
  }
  if (num_passes < 2) bad("loop_guidance", "the loop runs once: there is no earlier pass to read out");
  if (!std::isfinite(o.strength) || o.strength < -1.0f || o.strength > 4.0f)
    bad("loop_guidance_strength", "must be in [-1, 4]");
  loop_guidance = true;
  loop_guidance_strength = o.strength;
}

int Inkling2Config::visit_of(int layer, int pass) const {
  if (layer < core_start) return layer;
  if (layer >= core_end) return layer + (num_passes - 1) * core_layers();
  return layer + pass * core_layers();
}

int Inkling2Config::max_blocks() const {
  // The prelude's blocks, the core's, then the coda's on top of one pass.
  int blocks = 0;
  for (int l = 0; l < num_hidden_layers; ++l) blocks += first_of_block(l) ? 1 : 0;
  return blocks;
}

int Inkling2Config::num_full_visits() const {
  int n = 0;
  for (int v = 0; v < num_visits(); ++v) n += is_local_layer(visit(v).layer) ? 0 : 1;
  return n;
}

Inkling2Config Inkling2Config::parse(const minijson::Value& root, const minijson::Value& quant) {
  if (!root.is_object()) reject("", "root is not an object");
  Inkling2Config c;
  if (const std::string mt = optional_string(root, "model_type", "model_type", "inkling2_loop"); mt != "inkling2_loop")
    reject("model_type", "expected inkling2_loop, got " + mt);
  const minijson::Value& t = require(root, "text_config", "text_config");
  if (!t.is_object()) reject("text_config", "not an object");

  c.hidden_size = require_int(t, "hidden_size", "text_config.hidden_size");
  c.num_hidden_layers = require_int(t, "num_hidden_layers", "text_config.num_hidden_layers");
  c.padded_vocab_size = require_int(t, "vocab_size", "text_config.vocab_size");
  c.vocab_size = optional_int(t, "unpadded_vocab_size", "text_config.unpadded_vocab_size", c.padded_vocab_size);
  c.max_position_embeddings = optional_int64(t, "model_max_length", "text_config.model_max_length", 1048576);
  c.rms_norm_eps = static_cast<float>(optional_double(t, "rms_norm_eps", "text_config.rms_norm_eps", 1e-6));
  // NVFP4 scales group 16 values and MXFP8 scales 32 along the hidden size.
  if (c.hidden_size <= 0 || c.hidden_size % 32 != 0) reject("text_config.hidden_size", "must be a positive multiple of 32");
  if (c.num_hidden_layers <= 0) reject("text_config.num_hidden_layers", "must be positive");
  if (c.padded_vocab_size <= 0) reject("text_config.vocab_size", "must be positive");
  if (c.vocab_size <= 0 || c.vocab_size > c.padded_vocab_size)
    reject("text_config.unpadded_vocab_size", "must be in [1, vocab_size]");
  if (c.max_position_embeddings <= 0) reject("text_config.model_max_length", "must be positive");
  if (!(c.rms_norm_eps > 0)) reject("text_config.rms_norm_eps", "must be positive");
  if (optional_bool(t, "pre_norm_pre_affine", "text_config.pre_norm_pre_affine", false))
    reject("text_config.pre_norm_pre_affine", "pre-affine norms are not implemented");
  if (optional_bool(t, "final_norm_pre_affine", "text_config.final_norm_pre_affine", false))
    reject("text_config.final_norm_pre_affine", "pre-affine norms are not implemented");
  if (optional_bool(root, "tie_word_embeddings", "tie_word_embeddings", false))
    reject("tie_word_embeddings", "tied embeddings are not implemented");

  // --- tokens ---------------------------------------------------------------
  for (const minijson::Value* owner : {&root, &t}) {
    const minijson::Value* eos = owner->find("eos_token_id");
    if (!eos || eos->is_null() || !c.eos_token_ids.empty()) continue;
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
  }
  if (c.eos_token_ids.empty()) reject("eos_token_id", "missing");
  for (int64_t id : c.eos_token_ids)
    if (id < 0 || id >= c.vocab_size) reject("eos_token_id", "id outside [0, unpadded_vocab_size)");
  c.pad_token_id = optional_int64(root, "pad_token_id", "pad_token_id", -1);
  c.image_token_id = optional_int64(root, "image_token_id", "image_token_id", 200054);
  {
    const minijson::Value* cap = t.find("final_logit_softcapping");
    c.final_logit_softcapping = cap && !cap->is_null()
                                    ? static_cast<float>(optional_double(t, "final_logit_softcapping",
                                                                         "text_config.final_logit_softcapping", 0.0))
                                    : 0.0f;
    if (c.final_logit_softcapping < 0) reject("text_config.final_logit_softcapping", "must not be negative");
  }

  // --- blocks and the loop ------------------------------------------------------
  c.block_layers = optional_int(t, "n_layers_per_attn_res_block", "text_config.n_layers_per_attn_res_block", 6);
  if (t.find("n_layers_per_attnres_block"))
    reject("text_config.n_layers_per_attnres_block", "renamed to n_layers_per_attn_res_block");
  if (c.block_layers <= 0) reject("text_config.n_layers_per_attn_res_block", "must be positive");
  if (!optional_bool(t, "attn_res_normalize_values", "text_config.attn_res_normalize_values", true))
    reject("text_config.attn_res_normalize_values", "only normalized mix values are implemented");
  {
    const minijson::Value& loop = require(t, "loop_config", "text_config.loop_config");
    if (!loop.is_object()) reject("text_config.loop_config", "not an object");
    c.core_start = require_int(loop, "core_start", "text_config.loop_config.core_start");
    c.core_end = require_int(loop, "core_end", "text_config.loop_config.core_end");
    c.num_passes = require_int(loop, "num_passes", "text_config.loop_config.num_passes");
    c.gate_rank = require_int(loop, "gate_rank", "text_config.loop_config.gate_rank");
    if (optional_bool(loop, "is_varloop", "text_config.loop_config.is_varloop", false))
      reject("text_config.loop_config.is_varloop", "variable-loop checkpoints are not implemented");
    if (!(0 < c.core_start && c.core_start < c.core_end && c.core_end <= c.num_hidden_layers))
      reject("text_config.loop_config.core_start", "the core layers are out of range");
    if (c.core_start % c.block_layers != 0 || c.core_end % c.block_layers != 0)
      reject("text_config.loop_config.core_start", "the core must start and end on a block boundary");
    if (c.num_passes < 2 || c.num_passes > 8) reject("text_config.loop_config.num_passes", "must be in [2, 8]");
    if (c.gate_rank <= 0 || c.gate_rank > 512) reject("text_config.loop_config.gate_rank", "must be in [1, 512]");
    if (t.find("loop_num_passes") && !t.find("loop_num_passes")->is_null())
      reject("text_config.loop_num_passes", "a pass override needs a variable-loop checkpoint");
    if (optional_bool(t, "loop_share_cache", "text_config.loop_share_cache", false))
      reject("text_config.loop_share_cache", "a shared loop cache needs a variable-loop checkpoint");
  }

  // --- attention ------------------------------------------------------------
  c.num_attention_heads = require_int(t, "num_attention_heads", "text_config.num_attention_heads");
  c.num_key_value_heads = require_int(t, "num_key_value_heads", "text_config.num_key_value_heads");
  c.head_dim = require_int(t, "head_dim", "text_config.head_dim");
  if (c.num_attention_heads <= 0 || c.num_key_value_heads <= 0)
    reject("text_config.num_attention_heads", "head counts must be positive");
  if (c.num_attention_heads % c.num_key_value_heads != 0)
    reject("text_config.num_key_value_heads", "must divide num_attention_heads");
  if (c.head_dim != 128) reject("text_config.head_dim", "the attention kernels implement 128-wide heads");
  // One geometry serves both layer kinds.
  if (optional_int(t, "swa_num_attention_heads", "text_config.swa_num_attention_heads", c.num_attention_heads) !=
      c.num_attention_heads)
    reject("text_config.swa_num_attention_heads", "must equal num_attention_heads");
  if (optional_int(t, "swa_num_key_value_heads", "text_config.swa_num_key_value_heads", c.num_key_value_heads) !=
      c.num_key_value_heads)
    reject("text_config.swa_num_key_value_heads", "must equal num_key_value_heads");
  if (optional_int(t, "swa_head_dim", "text_config.swa_head_dim", c.head_dim) != c.head_dim)
    reject("text_config.swa_head_dim", "must equal head_dim");
  if (optional_bool(t, "q_bias", "text_config.q_bias", false)) reject("text_config.q_bias", "biased projections are not implemented");
  if (optional_bool(t, "o_bias", "text_config.o_bias", false)) reject("text_config.o_bias", "biased projections are not implemented");
  c.d_rel = require_int(t, "d_rel", "text_config.d_rel");
  c.rel_extent = require_int(t, "rel_extent", "text_config.rel_extent");
  c.sliding_window_size = require_int(t, "sliding_window_size", "text_config.sliding_window_size");
  c.sconv_kernel_size = require_int(t, "sconv_kernel_size", "text_config.sconv_kernel_size");
  c.qk_l2_clip_threshold = static_cast<float>(optional_double(t, "qk_l2_clip_threshold", "text_config.qk_l2_clip_threshold",
                                                              22.627416997969522));
  c.rel_l2_clip_threshold = static_cast<float>(optional_double(t, "rel_l2_clip_threshold", "text_config.rel_l2_clip_threshold", 8.0));
  if (c.d_rel != 16) reject("text_config.d_rel", "the relative bias kernels implement 16 relative dimensions");
  if (c.rel_extent <= 0) reject("text_config.rel_extent", "must be positive");
  if (c.sliding_window_size <= 0) reject("text_config.sliding_window_size", "must be positive");
  if (c.sconv_kernel_size != 4) reject("text_config.sconv_kernel_size", "the convolution kernels implement four taps");
  if (!(c.qk_l2_clip_threshold > 0) || !(c.rel_l2_clip_threshold > 0))
    reject("text_config.qk_l2_clip_threshold", "clip thresholds must be positive");
  if (!optional_bool(t, "use_value_norm_no_gamma", "text_config.use_value_norm_no_gamma", true))
    reject("text_config.use_value_norm_no_gamma", "only the weightless value norm is implemented");
  {
    const minijson::Value* floor = t.find("log_scaling_n_floor");
    c.log_scaling = floor && !floor->is_null();
    c.log_scaling_n_floor = optional_int64(t, "log_scaling_n_floor", "text_config.log_scaling_n_floor", 0);
    c.log_scaling_alpha = static_cast<float>(optional_double(t, "log_scaling_alpha", "text_config.log_scaling_alpha", 0.1));
    if (c.log_scaling && c.log_scaling_n_floor <= 0) reject("text_config.log_scaling_n_floor", "must be positive");
  }
  c.local_layer = layer_flags(int_list(t, "local_layer_ids", "text_config.local_layer_ids", true), c.num_hidden_layers,
                              "text_config.local_layer_ids");

  // --- MoE ----------------------------------------------------------------------
  c.n_routed_experts = require_int(t, "n_routed_experts", "text_config.n_routed_experts");
  c.num_experts_per_tok = require_int(t, "num_experts_per_tok", "text_config.num_experts_per_tok");
  c.intermediate_size = require_int(t, "intermediate_size", "text_config.intermediate_size");
  c.route_scale = static_cast<float>(optional_double(t, "route_scale", "text_config.route_scale", 8.0));
  c.mlp_gate_softcap = static_cast<float>(optional_double(t, "mlp_gate_softcap", "text_config.mlp_gate_softcap", 0.0));
  c.mlp_up_softcap = static_cast<float>(optional_double(t, "mlp_up_softcap", "text_config.mlp_up_softcap", 0.0));
  if (c.n_routed_experts <= 0 || c.n_routed_experts > 4096) reject("text_config.n_routed_experts", "must be in [1, 4096]");
  if (c.num_experts_per_tok <= 0 || c.num_experts_per_tok > 16 || c.num_experts_per_tok > c.n_routed_experts)
    reject("text_config.num_experts_per_tok", "must be in [1, min(n_routed_experts, 16)]");
  if (c.intermediate_size <= 0 || c.intermediate_size % 32 != 0)
    reject("text_config.intermediate_size", "must be a positive multiple of 32");
  if (optional_int(t, "n_shared_experts", "text_config.n_shared_experts", 0) != 0)
    reject("text_config.n_shared_experts", "shared experts are not implemented");
  if (!optional_bool(t, "use_gate_bias", "text_config.use_gate_bias", true))
    reject("text_config.use_gate_bias", "the router without its selection bias is not implemented");
  if (!(c.route_scale > 0)) reject("text_config.route_scale", "must be positive");
  if (c.mlp_gate_softcap < 0 || c.mlp_up_softcap < 0) reject("text_config.mlp_gate_softcap", "softcaps must not be negative");

  parse_quantization(quant, c);
  parse_draft(root, c);
  parse_vision(root, c);
  return c;
}

Inkling2Config Inkling2Config::from_checkpoint_dir(const std::string& dir) {
  const std::string config = read_file(dir + "/config.json");
  const std::string quant = read_file(dir + "/hf_quant_config.json");
  const auto parsed = minijson::parse(config);
  const auto parsed_quant = minijson::parse(quant);
  return parse(parsed.root, parsed_quant.root);
}

}  // namespace dgpp
