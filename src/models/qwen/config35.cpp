#include "models/qwen/config35.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <stdexcept>

namespace dgpp {
namespace {

[[noreturn]] void reject(std::string_view field, std::string_view why) {
  throw std::runtime_error(std::format("Qwen3.5 text_config.{}: {}", field, why));
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
int64_t optional_int64(const minijson::Value& v, std::string_view field, int64_t dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_number()) reject(field, "not a number");
  return f->as_int();
}
double require_double(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_number()) reject(field, "not a number");
  const double d = f.as_double();
  if (!std::isfinite(d)) reject(field, "not finite");
  return d;
}
bool require_bool(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_bool()) reject(field, "not a bool");
  return f.as_bool();
}
bool optional_bool(const minijson::Value& v, std::string_view field, bool dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_bool()) reject(field, "not a bool");
  return f->as_bool();
}
std::string require_string(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_string()) reject(field, "not a string");
  return std::string(f.as_string());
}
std::string optional_string(const minijson::Value& v, std::string_view field,
                            const std::string& dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_string()) reject(field, "not a string");
  return std::string(f->as_string());
}
std::vector<int64_t> require_int_array(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_array()) reject(field, "not an array");
  std::vector<int64_t> out;
  for (const auto& item : f.items()) {
    if (!item.is_number()) reject(field, "non-numeric element");
    out.push_back(item.as_int());
  }
  return out;
}

std::string read_file(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f)
    throw std::runtime_error(std::format("cannot open config {}: {}", path,
                                         std::strerror(errno)));
  std::string text;
  char buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
  std::fclose(f);
  return text;
}

}  // namespace

Qwen35TextConfig Qwen35TextConfig::parse(const minijson::Value& tc,
                                        const minijson::Value* quantization_config) {
  if (!tc.is_object()) reject("text_config", "not an object");
  Qwen35TextConfig c;
  const std::string model_type = optional_string(tc, "model_type", "qwen3_5_text");
  if (model_type != "qwen3_5_text")
    reject("model_type", "expected qwen3_5_text, got " + model_type);

  c.hidden_size = require_int(tc, "hidden_size");
  c.vocab_size = require_int(tc, "vocab_size");
  c.num_hidden_layers = require_int(tc, "num_hidden_layers");
  c.rms_norm_eps = static_cast<float>(require_double(tc, "rms_norm_eps"));
  c.tie_word_embeddings = require_bool(tc, "tie_word_embeddings");
  c.hidden_act = require_string(tc, "hidden_act");
  c.max_position_embeddings = require_int(tc, "max_position_embeddings");
  if (c.hidden_size <= 0 || c.hidden_size % 8 != 0)
    reject("hidden_size", "must be a positive multiple of 8");
  if (c.vocab_size <= 0) reject("vocab_size", "must be positive");
  if (c.num_hidden_layers <= 0) reject("num_hidden_layers", "must be positive");
  if (c.hidden_act != "silu")
    reject("hidden_act", "only silu is implemented, got " + c.hidden_act);
  if (c.tie_word_embeddings)
    reject("tie_word_embeddings", "tied embeddings are not implemented");
  if (const minijson::Value* ab = tc.find("attention_bias"))
    if (ab->is_bool() && ab->as_bool())
      reject("attention_bias", "biased attention projections are not implemented");

  // --- layer kinds ------------------------------------------------------------
  {
    const minijson::Value& lt = require(tc, "layer_types");
    if (!lt.is_array()) reject("layer_types", "not an array");
    for (const auto& item : lt.items()) {
      const std::string s = item.is_string() ? std::string(item.as_string()) : "";
      if (s == "linear_attention") c.layers.push_back(Qwen35LayerKind::Gdn);
      else if (s == "full_attention") c.layers.push_back(Qwen35LayerKind::Full);
      else reject("layer_types", "unsupported layer type '" + s + "'");
    }
    if (static_cast<int>(c.layers.size()) != c.num_hidden_layers)
      reject("layer_types", "length does not match num_hidden_layers");
    const int interval = optional_int(tc, "full_attention_interval", 0);
    if (interval > 0)
      for (int i = 0; i < c.num_hidden_layers; ++i)
        if (((i + 1) % interval == 0) != (c.layers[i] == Qwen35LayerKind::Full))
          reject("full_attention_interval",
                 "disagrees with layer_types at layer " + std::to_string(i));
  }

  // --- tokens -----------------------------------------------------------------
  if (const minijson::Value* eos = tc.find("eos_token_id"); eos && !eos->is_null()) {
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
  c.bos_token_id = optional_int64(tc, "bos_token_id", -1);

  // --- Gated DeltaNet ---------------------------------------------------------
  c.gdn_key_heads = require_int(tc, "linear_num_key_heads");
  c.gdn_value_heads = require_int(tc, "linear_num_value_heads");
  c.gdn_key_head_dim = require_int(tc, "linear_key_head_dim");
  c.gdn_value_head_dim = require_int(tc, "linear_value_head_dim");
  c.gdn_conv_width = require_int(tc, "linear_conv_kernel_dim");
  // The swish output gate's kernels land with the layer slice; the config
  // records (and binds) it now so the checkpoint is loadable end to end.
  c.output_gate_type = optional_string(tc, "output_gate_type", "swish");
  if (c.gdn_key_heads <= 0 || c.gdn_value_heads <= 0 ||
      c.gdn_value_heads % c.gdn_key_heads != 0)
    reject("linear_num_value_heads", "must be a positive multiple of linear_num_key_heads");
  if (c.gdn_key_head_dim != 128 || c.gdn_value_head_dim != 128)
    reject("linear_key_head_dim", "the GDN kernels implement 128-wide heads");
  if (c.gdn_conv_width < 2 || c.gdn_conv_width > 8)
    reject("linear_conv_kernel_dim", "must be in [2, 8]");
  if (c.output_gate_type != "swish")
    reject("output_gate_type", "only the swish output gate is implemented, got " +
                                   c.output_gate_type);
  if (const std::string dt = optional_string(tc, "mamba_ssm_dtype", "float32");
      dt != "float32")
    reject("mamba_ssm_dtype", "the recurrent state is float32, got " + dt);

  // --- full attention -----------------------------------------------------------
  c.num_attention_heads = require_int(tc, "num_attention_heads");
  c.num_key_value_heads = require_int(tc, "num_key_value_heads");
  c.head_dim = require_int(tc, "head_dim");
  if (c.num_attention_heads <= 0 || c.num_key_value_heads <= 0 ||
      c.num_attention_heads % c.num_key_value_heads != 0)
    reject("num_key_value_heads", "must divide num_attention_heads");
  if (c.head_dim != 256) reject("head_dim", "the full-attention kernels implement 256-wide heads");
  {
    const minijson::Value* rp = tc.find("rope_parameters");
    if (!rp || !rp->is_object()) reject("rope_parameters", "missing");
    c.rope_theta = require_double(*rp, "rope_theta");
    const double factor = require_double(*rp, "partial_rotary_factor");
    const double rd = c.head_dim * factor;
    if (!(rd > 0) || rd != std::floor(rd) || static_cast<int>(rd) % 2 != 0)
      reject("rope_parameters.partial_rotary_factor", "rotary dim must be a positive even integer");
    c.rotary_dim = static_cast<int>(rd);
    if (const std::string rt = optional_string(*rp, "rope_type", "default"); rt != "default")
      // Same checkpoint-vs-engine precedence as Qwen4Exp: the YaRN ramp is
      // the engine's rope_scaling knob, so a checkpoint that bakes one in
      // would be scaled twice.
      reject("rope_parameters.rope_type",
             "the checkpoint's rope must be \"default\": the YaRN ramp is the engine's "
             "rope_scaling knob, got " + rt);
    c.mrope_interleaved = optional_bool(*rp, "mrope_interleaved", false);
    if (const minijson::Value* ms = rp->find("mrope_section"); ms && !ms->is_null()) {
      for (const int64_t v : require_int_array(*rp, "mrope_section"))
        c.mrope_section.push_back(static_cast<int>(v));
      int64_t sum = 0;
      for (int v : c.mrope_section) sum += v;
      if (c.mrope_section.size() != 3 || sum * 2 != c.rotary_dim)
        reject("rope_parameters.mrope_section", "must be three sections summing to rotary_dim / 2");
    }
  }
  c.attn_output_gate = optional_bool(tc, "attn_output_gate", true);
  if (!c.attn_output_gate)
    reject("attn_output_gate", "the checkpoint stacks [q | gate] in q_proj");

  // --- dense MLP ----------------------------------------------------------------
  c.intermediate_size = require_int(tc, "intermediate_size");
  if (c.intermediate_size <= 0) reject("intermediate_size", "must be positive");

  // --- MTP ----------------------------------------------------------------------
  c.mtp_num_layers = optional_int(tc, "mtp_num_hidden_layers", 0);
  if (const minijson::Value* m = tc.find("mtp"); m && m->is_object()) {
    const int n = optional_int(*m, "num_hidden_layers", c.mtp_num_layers);
    if (n != c.mtp_num_layers) reject("mtp.num_hidden_layers", "disagrees with mtp_num_hidden_layers");
  }
  if (c.mtp_num_layers != 0 && c.mtp_num_layers != 1)
    reject("mtp_num_hidden_layers", "only the single draft layer is implemented");
  if (optional_bool(tc, "mtp_use_dedicated_embeddings", false))
    reject("mtp_use_dedicated_embeddings", "the draft shares the embeddings");

  // --- quantization ---------------------------------------------------------------
  if (quantization_config == nullptr || quantization_config->is_null())
    throw std::runtime_error(
        "Qwen3.5 quantization_config: missing — the engine implements the FP8 "
        "block release (e4m3 + BF16 128x128 scales) and the NVFP4 mixed release");
  {
    const minijson::Value& q = *quantization_config;
    if (const minijson::Value* groups = q.find("config_groups");
        groups != nullptr && groups->is_object()) {
      // The NVFP4 mixed release (compressed-tensors / modelopt): group_1 is
      // the MLP's 4-bit float per 16; the loader slice interprets the rest.
      const minijson::Value* g1 = groups->find("group_1");
      const minijson::Value* w = g1 != nullptr ? g1->find("weights") : nullptr;
      if (w == nullptr || !w->is_object())
        throw std::runtime_error("Qwen3.5 quantization_config.config_groups: group_1.weights missing");
      const int64_t bits = require_int(*w, "num_bits");
      const int64_t group = require_int(*w, "group_size");
      if (bits != 4 || group != 16)
        throw std::runtime_error("Qwen3.5 quantization_config.config_groups: only NVFP4 (4-bit, group 16) is implemented");
      c.quant_kind = Qwen35QuantKind::Nvfp4Mixed;
      return c;
    }
    const std::string method = optional_string(q, "quant_method", "");
    if (method != "fp8")
      throw std::runtime_error("Qwen3.5 quantization_config.quant_method: only fp8 is implemented, got '" + method + "'");
    const std::vector<int64_t> bs = require_int_array(q, "weight_block_size");
    if (bs.size() != 2 || bs[0] != 128 || bs[1] != 128)
      throw std::runtime_error("Qwen3.5 quantization_config.weight_block_size: only [128, 128] is implemented");
    if (const std::string scheme = optional_string(q, "activation_scheme", "dynamic"); scheme != "dynamic")
      throw std::runtime_error("Qwen3.5 quantization_config.activation_scheme: only dynamic is implemented");
    c.quant_kind = Qwen35QuantKind::Fp8Block;
  }
  return c;
}

Qwen35TextConfig Qwen35TextConfig::from_json_file(const std::string& path) {
  // The parsed values view the text: it must outlive the parse.
  const std::string json = read_file(path);
  const auto parsed = minijson::parse(json);
  const minijson::Value* tc = parsed.root.find("text_config");
  if (!tc) throw std::runtime_error("config " + path + ": missing text_config object");
  return parse(*tc, parsed.root.find("quantization_config"));
}

int Qwen35TextConfig::num_gdn_layers() const {
  int n = 0;
  for (auto k : layers) n += k == Qwen35LayerKind::Gdn;
  return n;
}
int Qwen35TextConfig::num_full_layers() const {
  int n = 0;
  for (auto k : layers) n += k == Qwen35LayerKind::Full;
  return n;
}

}  // namespace dgpp
