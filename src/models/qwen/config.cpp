#include "models/qwen/config.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <stdexcept>

namespace dgpp {
namespace {

[[noreturn]] void reject(std::string_view field, std::string_view why) {
  throw std::runtime_error(std::format("Qwen text_config.{}: {}", field, why));
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

bool is_prime(int64_t v) {
  if (v < 2) return false;
  if (v % 2 == 0) return v == 2;
  for (int64_t d = 3; d * d <= v; d += 2)
    if (v % d == 0) return false;
  return true;
}

// splitmix64 and the reference's multiplier derivation
// (modular_qwen4_exp.py _build_layer_multipliers): unsigned 64-bit wraparound.
uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

}  // namespace

QwenTextConfig QwenTextConfig::parse(const minijson::Value& tc,
                                     const minijson::Value* quantization_config) {
  if (!tc.is_object()) reject("text_config", "not an object");
  QwenTextConfig c;
  const std::string model_type = optional_string(tc, "model_type", "qwen4_exp_text");
  if (model_type != "qwen4_exp_text")
    reject("model_type", "expected qwen4_exp_text, got " + model_type);

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
      if (s == "linear_attention") c.layers.push_back(QwenLayerKind::Gdn);
      else if (s == "full_attention" || s == "qwen_sparse_attention")
        c.layers.push_back(QwenLayerKind::Qsa);
      else reject("layer_types", "unsupported layer type '" + s + "'");
    }
    if (static_cast<int>(c.layers.size()) != c.num_hidden_layers)
      reject("layer_types", "length does not match num_hidden_layers");
    const int interval = optional_int(tc, "full_attention_interval", 0);
    if (interval > 0)
      for (int i = 0; i < c.num_hidden_layers; ++i)
        if (((i + 1) % interval == 0) != (c.layers[i] == QwenLayerKind::Qsa))
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

  // --- gated residual ---------------------------------------------------------
  c.hc_count = require_int(tc, "hc_count");
  c.hc_lowrank = require_int(tc, "hc_lowrank");
  if (c.hc_count != 4)
    reject("hc_count", "the gated-residual kernels are 4-branch (got " +
                           std::to_string(c.hc_count) + ")");
  if (c.hc_lowrank <= 0 || c.hc_lowrank % 8 != 0)
    reject("hc_lowrank", "must be a positive multiple of 8");

  // --- Gated DeltaNet ---------------------------------------------------------
  c.gdn_key_heads = require_int(tc, "linear_num_key_heads");
  c.gdn_value_heads = require_int(tc, "linear_num_value_heads");
  c.gdn_key_head_dim = require_int(tc, "linear_key_head_dim");
  c.gdn_value_head_dim = require_int(tc, "linear_value_head_dim");
  c.gdn_conv_width = require_int(tc, "linear_conv_kernel_dim");
  c.output_gate_type = optional_string(tc, "output_gate_type", c.hidden_act);
  if (c.gdn_key_heads <= 0 || c.gdn_value_heads <= 0 ||
      c.gdn_value_heads % c.gdn_key_heads != 0)
    reject("linear_num_value_heads", "must be a positive multiple of linear_num_key_heads");
  if (c.gdn_key_head_dim != 128 || c.gdn_value_head_dim != 128)
    reject("linear_key_head_dim", "the GDN kernels implement 128-wide heads");
  if (c.gdn_conv_width < 2 || c.gdn_conv_width > 8)
    reject("linear_conv_kernel_dim", "must be in [2, 8]");
  if (c.output_gate_type != "sigmoid")
    reject("output_gate_type", "only the sigmoid output gate is implemented, got " +
                                   c.output_gate_type);
  if (const std::string dt = optional_string(tc, "mamba_ssm_dtype", "float32");
      dt != "float32")
    reject("mamba_ssm_dtype", "the recurrent state is float32, got " + dt);

  // --- attention + indexer ----------------------------------------------------
  c.num_attention_heads = require_int(tc, "num_attention_heads");
  c.num_key_value_heads = require_int(tc, "num_key_value_heads");
  c.head_dim = require_int(tc, "head_dim");
  if (c.num_attention_heads <= 0 || c.num_key_value_heads <= 0 ||
      c.num_attention_heads % c.num_key_value_heads != 0)
    reject("num_key_value_heads", "must divide num_attention_heads");
  if (c.head_dim != 256) reject("head_dim", "the QSA kernels implement 256-wide heads");
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
      // The checkpoint's own rope must be plain: the YaRN ramp this family
      // serves 512K with is the ENGINE's knob (engine.rope_scaling,
      // kernels/rope_scaling.hpp), because the NVFP4 release carries no
      // scaling and vLLM's recipe applies it from --hf-overrides. A
      // checkpoint that bakes one in would otherwise be mis-scaled twice.
      reject("rope_parameters.rope_type",
             "the checkpoint's rope must be default (the YaRN ramp is the engine's rope_scaling knob), got " + rt);
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
  c.indexer_n_heads = require_int(tc, "indexer_n_heads");
  c.indexer_kv_heads = require_int(tc, "indexer_kv_heads");
  c.indexer_head_dim = require_int(tc, "indexer_head_dim");
  c.indexer_budget = require_int(tc, "indexer_budget");
  c.indexer_compress_ratio = require_int(tc, "indexer_compress_ratio");
  if (c.indexer_kv_heads != 1) reject("indexer_kv_heads", "QSA requires one indexer key head");
  if (c.indexer_n_heads <= 0) reject("indexer_n_heads", "must be positive");
  if (c.indexer_head_dim != 128) reject("indexer_head_dim", "the indexer kernels implement 128");
  if (c.indexer_compress_ratio < 2) reject("indexer_compress_ratio", "must be at least 2");
  if (c.indexer_budget <= 0 || c.indexer_budget % c.indexer_compress_ratio != 0)
    reject("indexer_budget", "must be a positive multiple of indexer_compress_ratio");
  if (c.rotary_dim > c.indexer_head_dim)
    reject("indexer_head_dim", "the attention's rotary dims must fit the indexer head");

  // --- MoE --------------------------------------------------------------------
  c.num_experts = require_int(tc, "num_experts");
  c.num_experts_per_tok = require_int(tc, "num_experts_per_tok");
  c.moe_intermediate_size = require_int(tc, "moe_intermediate_size");
  c.shared_expert_intermediate_size = require_int(tc, "shared_expert_intermediate_size");
  c.norm_topk_prob = optional_bool(tc, "norm_topk_prob", true);
  if (c.num_experts <= 0 || c.num_experts > 4096) reject("num_experts", "must be in [1, 4096]");
  if (c.num_experts_per_tok <= 0 || c.num_experts_per_tok > c.num_experts ||
      c.num_experts_per_tok > 16)
    reject("num_experts_per_tok", "must be in [1, min(num_experts, 16)]");
  if (c.moe_intermediate_size <= 0 || c.shared_expert_intermediate_size <= 0)
    reject("moe_intermediate_size", "must be positive");
  if (!c.norm_topk_prob) reject("norm_topk_prob", "the router renormalizes the top-k (true)");

  // --- PLE --------------------------------------------------------------------
  if (const minijson::Value* pl = tc.find("ple_layer_ids"); pl && !pl->is_null()) {
    for (const int64_t v : require_int_array(tc, "ple_layer_ids"))
      c.ple_layer_ids.push_back(static_cast<int>(v));
  }
  c.ple_embed_dim = optional_int(tc, "ple_embed_dim", c.hidden_size);
  c.ple_conv_kernel_size = optional_int(tc, "ple_conv_kernel_size", 4);
  c.ngram_size = optional_int(tc, "ngram_size", 3);
  c.heads_per_ngram = optional_int(tc, "heads_per_ngram", 8);
  c.ngram_vocab_size_base = optional_int64(tc, "ngram_vocab_size_base", 20000000);
  c.ngram_vocab_divisor = optional_int(tc, "make_ngram_vocab_size_divisible_by", 128);
  c.ngram_seed = optional_int64(tc, "seed", 1234);
  c.split_ngram_parts = optional_int(tc, "split_ngram_parts", 512);
  if (c.ple_layer_ids.size() > 1)
    reject("ple_layer_ids", "one n-gram embedding layer is implemented");
  if (!c.ple_layer_ids.empty()) {
    const int id = c.ple_layer_ids[0];
    if (id < 1 || id > c.num_hidden_layers) reject("ple_layer_ids", "one-indexed id out of range");
    if (c.layers[id - 1] != QwenLayerKind::Gdn)
      reject("ple_layer_ids", "the PLE sits on a linear_attention layer");
    if (c.ngram_size < 2) reject("ngram_size", "must be at least 2");
    if (c.heads_per_ngram <= 0) reject("heads_per_ngram", "must be positive");
    const int heads = (c.ngram_size - 1) * c.heads_per_ngram;
    if (c.ple_embed_dim <= 0 || c.ple_embed_dim % heads != 0)
      reject("ple_embed_dim", "must be a positive multiple of the n-gram head count");
    if (c.ngram_vocab_size_base <= 0) reject("ngram_vocab_size_base", "must be positive");
    if (c.ngram_vocab_divisor <= 0) reject("make_ngram_vocab_size_divisible_by", "must be positive");
    if (c.ple_conv_kernel_size < 1 || c.ple_conv_kernel_size > 8)
      reject("ple_conv_kernel_size", "must be in [1, 8]");
    if (c.split_ngram_parts <= 0) reject("split_ngram_parts", "must be positive");
    if (c.eos_token_ids.empty())
      reject("eos_token_id", "required: the n-gram context resets on it");
  }

  // --- MTP --------------------------------------------------------------------
  c.mtp_num_layers = optional_int(tc, "mtp_num_hidden_layers", 0);
  if (const minijson::Value* m = tc.find("mtp"); m && m->is_object()) {
    const int n = optional_int(*m, "num_hidden_layers", c.mtp_num_layers);
    if (n != c.mtp_num_layers) reject("mtp.num_hidden_layers", "disagrees with mtp_num_hidden_layers");
    if (const minijson::Value* lt = m->find("layer_types"); lt && lt->is_array()) {
      if (static_cast<int>(lt->items().size()) != n) reject("mtp.layer_types", "length");
      for (const auto& item : lt->items())
        if (!item.is_string() || (item.as_string() != "full_attention" &&
                                  item.as_string() != "qwen_sparse_attention"))
          reject("mtp.layer_types", "the draft layer must be a full_attention (QSA) layer");
    }
  }
  if (c.mtp_num_layers != 0 && c.mtp_num_layers != 1)
    reject("mtp_num_hidden_layers", "only the single draft layer is implemented");
  if (optional_bool(tc, "mtp_use_dedicated_embeddings", false))
    reject("mtp_use_dedicated_embeddings", "the draft shares the embeddings");

  // --- quantization -----------------------------------------------------------
  if (quantization_config == nullptr || quantization_config->is_null())
    throw std::runtime_error(
        "Qwen quantization_config: missing — the engine implements the FP8 "
        "release (routed experts and the n-gram table in e4m3)");
  {
    const minijson::Value& q = *quantization_config;
    const std::string method = optional_string(q, "quant_method", "");
    // The NVIDIA NVFP4 release carries a modelopt/compressed-tensors block
    // instead of quant_method: config_groups.group_0.weights = 4-bit float
    // per 16, targets = the backbone's expert modules.
    if (const minijson::Value* groups = q.find("config_groups"); groups != nullptr && groups->is_object()) {
      const minijson::Value* g0 = groups->find("group_0");
      const minijson::Value* w = g0 != nullptr ? g0->find("weights") : nullptr;
      if (w == nullptr || !w->is_object())
        throw std::runtime_error("Qwen quantization_config.config_groups: group_0.weights missing");
      const int64_t bits = require_int(*w, "num_bits");
      const std::string type = optional_string(*w, "type", "");
      const int64_t group = require_int(*w, "group_size");
      if (bits != 4 || type != "float" || group != 16)
        throw std::runtime_error("Qwen quantization_config.config_groups: only NVFP4 (4-bit float, group 16) is implemented");
      c.experts_fp8 = false;
      c.experts_nvfp4 = true;
      c.ngram_table_fp8 = false;
      return c;
    }
    if (method != "fp8")
      throw std::runtime_error("Qwen quantization_config.quant_method: only fp8 is implemented, got '" + method + "'");
    const std::vector<int64_t> bs = require_int_array(q, "weight_block_size");
    if (bs.size() != 2 || bs[0] != 128 || bs[1] != 128)
      throw std::runtime_error("Qwen quantization_config.weight_block_size: only [128, 128] is implemented");
    if (const std::string scheme = optional_string(q, "activation_scheme", "dynamic"); scheme != "dynamic")
      throw std::runtime_error("Qwen quantization_config.activation_scheme: only dynamic is implemented");
    c.experts_fp8 = true;
    c.ngram_table_fp8 = false;
    if (const minijson::Value* mc = q.find("modules_to_convert"); mc && mc->is_array())
      for (const auto& item : mc->items())
        if (item.is_string() && item.as_string().find("ngram_embedding") != std::string_view::npos)
          c.ngram_table_fp8 = true;
    if (!c.ple_layer_ids.empty() && !c.ngram_table_fp8)
      throw std::runtime_error("Qwen quantization_config: the n-gram table must be listed in modules_to_convert (e4m3)");
  }
  (void)c.ngram_geometry();  // validates the derivation (throws on overflow)
  return c;
}

QwenTextConfig QwenTextConfig::from_json_file(const std::string& path) {
  // The parsed values view the text: it must outlive the parse.
  const std::string json = read_file(path);
  const auto parsed = minijson::parse(json);
  const minijson::Value* tc = parsed.root.find("text_config");
  if (!tc) throw std::runtime_error("config " + path + ": missing text_config object");
  return parse(*tc, parsed.root.find("quantization_config"));
}

int QwenTextConfig::num_gdn_layers() const {
  int n = 0;
  for (auto k : layers) n += k == QwenLayerKind::Gdn;
  return n;
}
int QwenTextConfig::num_qsa_layers() const {
  int n = 0;
  for (auto k : layers) n += k == QwenLayerKind::Qsa;
  return n;
}

QwenNgramGeometry QwenTextConfig::ngram_geometry() const {
  QwenNgramGeometry g;
  if (ple_layer_ids.empty()) return g;
  g.heads = (ngram_size - 1) * heads_per_ngram;
  g.head_dim = ple_embed_dim / g.heads;
  int64_t prime = ngram_vocab_size_base - 1;
  for (int h = 0; h < g.heads; ++h) {
    // ple_layer_index is 0 (one PLE layer): global head index == h.
    ++prime;
    while (!is_prime(prime)) ++prime;
    g.head_vocab.push_back(prime);
    g.head_offset.push_back(g.total_rows);
    g.total_rows += prime;
  }
  g.padded_rows = (g.total_rows + ngram_vocab_divisor - 1) / ngram_vocab_divisor *
                  ngram_vocab_divisor;
  // The multipliers (the reference: max_long // vocab, halved; 2*(mix % half)+1).
  const uint64_t max_long = (uint64_t{1} << 63) - 1;
  const uint64_t m_max = max_long / static_cast<uint64_t>(vocab_size > 0 ? vocab_size : 1);
  const uint64_t half = m_max / 2 > 0 ? m_max / 2 : 1;
  const uint64_t base_seed = static_cast<uint64_t>(ngram_seed);  // + 10007 * ple_layer_index (0)
  for (int i = 0; i < ngram_size; ++i) {
    const uint64_t x0 = base_seed + 0x9E3779B97F4A7C15ull * static_cast<uint64_t>(i + 1);
    const uint64_t mixed = splitmix64(x0);
    g.multipliers.push_back(static_cast<int64_t>(2 * (mixed % half) + 1));
  }
  // The hash's products (multiplier * token id) must stay below 2^63 so
  // the int64 arithmetic never wraps (the reference relies on it too).
  for (const int64_t m : g.multipliers)
    if (static_cast<uint64_t>(m) > max_long / static_cast<uint64_t>(vocab_size))
      throw std::runtime_error("Qwen n-gram hash: multiplier * vocab overflows int64");
  return g;
}

}  // namespace dgpp
