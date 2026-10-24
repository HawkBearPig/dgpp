#include "models/qwen/config.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <stdexcept>
#include <vector>

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
      // checkpoint that bakes one in would otherwise be mis-scaled twice —
      // its table is already the ramped one, and the band would land past
      // the end of it. The reason rides the message as well as the comment
      // (review item 9): an operator holding an otherwise compatible
      // checkpoint reads this as "the engine will not scale it twice", not
      // as "unimplemented". Accepting such a checkpoint, with an explicit
      // checkpoint-vs-engine precedence, is the follow-up
      // (docs/qwen38_flash_next_plan.md §1.9.1).
      reject("rope_parameters.rope_type",
             "the checkpoint's rope must be \"default\": the YaRN ramp is the engine's "
             "rope_scaling knob, and applying it over a checkpoint that already declares "
             "YaRN would scale the rope twice — the follow-up in "
             "docs/qwen38_flash_next_plan.md §1.9.1. rope_type = " + rt + " is not "
             "implemented for this family");
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
    // The QAD export (local-inference-lab/Qwen3.8-Flash-Next-NVFP4, issue
    // #113): modelopt's MIXED_PRECISION block — no config_groups, every
    // quantized module listed under quantized_layers with its algorithm
    // and group. Strict: the backbone experts NVFP4/16 in every layer, the
    // dense stack MXFP8/32 in every layer (all five projections of the
    // layer's attention form and the three shared-expert matrices), the
    // n-gram table NVFP4/16 when a PLE exists, the draft experts weight-only
    // NVFP4/16 when the draft layer exists, nothing else quantized. A
    // module the engine does not read in that form is refused by name.
    if (method == "modelopt") {
      auto bad = [](const std::string& what) {
        throw std::runtime_error("Qwen quantization_config (modelopt): " + what);
      };
      if (const std::string algo = optional_string(q, "quant_algo", ""); algo != "MIXED_PRECISION")
        bad("only quant_algo MIXED_PRECISION is implemented, got '" + algo + "'");
      if (const minijson::Value* kv = q.find("kv_cache_quant_algo"); kv != nullptr && !kv->is_null())
        bad("a quantized KV cache (kv_cache_quant_algo) is not implemented");
      const minijson::Value* ql = q.find("quantized_layers");
      if (ql == nullptr || !ql->is_object()) bad("quantized_layers is missing");
      const int L = c.num_hidden_layers;
      std::vector<bool> experts_seen(static_cast<size_t>(L), false);
      // Per layer, the dense modules declared MXFP8/32 (a bitmask over the
      // eight names below: five of the attention form, three shared).
      std::vector<unsigned> dense_seen(static_cast<size_t>(L), 0u);
      static const char* kGdn[] = {"linear_attn.in_proj_a", "linear_attn.in_proj_b", "linear_attn.in_proj_qkv",
                                   "linear_attn.in_proj_z", "linear_attn.out_proj"};
      static const char* kQsa[] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                                   "self_attn.indexer.index_qk_proj"};
      static const char* kShared[] = {"mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj",
                                      "mlp.shared_expert.down_proj"};
      bool table_seen = false, mtp_seen = false;
      bool csf_seen = false, plain_seen = false, mtp_csf_seen = false, mtp_plain_seen = false;
      const std::string layers_prefix = "model.language_model.layers.";
      for (const auto& m : ql->members()) {
        const std::string key = m.key;
        if (!m.value.is_object()) bad("quantized_layers['" + key + "'] is not an object");
        // The vision tower is never served (text-only; its tensors are
        // counted, not validated, at binding): note its presence, skip
        // its form. The FP4-Local export quantizes it (W4A16_NVFP4 +
        // MXFP8); other releases leave it out entirely.
        if (key.rfind("model.visual.", 0) == 0) continue;
        const std::string algo = optional_string(m.value, "quant_algo", "");
        const int64_t group = optional_int64(m.value, "group_size", -1);
        auto want = [&](const char* a, int64_t g, const char* what) {
          if (algo != a || group != g)
            bad("'" + key + "' is " + algo + "/" + std::to_string(group) + "; the engine reads " + what + " as " +
                std::string(a) + " with group " + std::to_string(g));
        };
        // The draft experts: the file names them under mtp.layers.0 and
        // again under the absolute index (mtp.layers.48); the tensors live
        // under mtp.layers.0 alone.
        if (key.rfind("mtp.layers.", 0) == 0 && key.size() > 11 &&
            key.compare(key.find('.', 11), std::string::npos, ".mlp.experts") == 0 &&
            key.substr(11, key.find('.', 11) - 11).find_first_not_of("0123456789") == std::string::npos) {
          if (c.mtp_layer() < 0) bad("'" + key + "' declared without a draft layer");
          if (algo != "W4A16_NVFP4" && algo != "NVFP4") want("W4A16_NVFP4", 16, "the draft experts");
          if (group != 16) want(algo.c_str(), 16, "the draft experts");
          // The draft experts carry their own scale form: the FP4-Local
          // export wraps only the backbone's as CSF, the draft keeping the
          // plain E4M3 weight_scale (the loader picks the form per scope).
          const std::string mtp_enc = optional_string(m.value, "weight_scale_encoding", "");
          if (mtp_enc == "csf")
            mtp_csf_seen = true;
          else if (mtp_enc.empty())
            mtp_plain_seen = true;
          else
            bad("'" + key + "' declares weight_scale_encoding '" + mtp_enc +
                "' (only \"csf\" is implemented)");
          mtp_seen = true;
          continue;
        }
        if (key.rfind(layers_prefix, 0) != 0) bad("quantized module '" + key + "' is not implemented");
        const size_t dot = key.find('.', layers_prefix.size());
        if (dot == std::string::npos) bad("quantized module '" + key + "' is not implemented");
        const int64_t layer = std::atoll(key.substr(layers_prefix.size(), dot - layers_prefix.size()).c_str());
        if (layer < 0 || layer >= L) bad("'" + key + "' names a layer outside num_hidden_layers");
        const std::string module = key.substr(dot + 1);
        if (module == "mlp.experts") {
          want("NVFP4", 16, "the routed experts");
          // The FP4-Local export wraps every backbone experts entry's
          // per-group scales as ModelOpt CSF sidecars
          // (weight_scale_encoding "csf"); the plain E4M3 weight_scale is
          // then absent and the loader expands the sidecars once. Mixed
          // encodings across layers refuse: one export, one form.
          const std::string enc = optional_string(m.value, "weight_scale_encoding", "");
          if (enc == "csf")
            csf_seen = true;
          else if (enc.empty())
            plain_seen = true;
          else
            bad("'" + key + "' declares weight_scale_encoding '" + enc + "' (only \"csf\" is implemented)");
          experts_seen[static_cast<size_t>(layer)] = true;
          continue;
        }
        if (module == "ple.ple_embedding.ngram_embedding") {
          if (static_cast<int>(layer) != c.ple_layer()) bad("'" + key + "' is not on the PLE layer");
          want("NVFP4", 16, "the n-gram table");
          table_seen = true;
          continue;
        }
        const bool gdn = c.layers[static_cast<size_t>(layer)] == QwenLayerKind::Gdn;
        int bit = -1;
        for (int i = 0; i < 5 && bit < 0; ++i)
          if (module == (gdn ? kGdn : kQsa)[i]) bit = i;
        for (int i = 0; i < 3 && bit < 0; ++i)
          if (module == kShared[i]) bit = 5 + i;
        if (bit < 0) bad("quantized module '" + key + "' is not implemented (the engine reads it in BF16)");
        want("MXFP8", 32, "the dense stack");
        dense_seen[static_cast<size_t>(layer)] |= 1u << bit;
      }
      for (int l = 0; l < L; ++l) {
        if (!experts_seen[static_cast<size_t>(l)])
          bad("layer " + std::to_string(l) + "'s routed experts are not declared NVFP4");
        if (dense_seen[static_cast<size_t>(l)] != 0xFFu) {
          const bool gdn = c.layers[static_cast<size_t>(l)] == QwenLayerKind::Gdn;
          for (int i = 0; i < 8; ++i)
            if (!(dense_seen[static_cast<size_t>(l)] >> i & 1u))
              bad("layer " + std::to_string(l) + "." + std::string(i < 5 ? (gdn ? kGdn : kQsa)[i] : kShared[i - 5]) +
                  " is not declared MXFP8 (the engine reads the whole dense stack in the one shipped form)");
        }
      }
      if (!c.ple_layer_ids.empty() && !table_seen) bad("the n-gram table's form is not declared");
      if (c.mtp_layer() >= 0 && !mtp_seen) bad("the draft layer's experts are not declared");
      if (csf_seen && plain_seen)
        bad("some routed-experts entries declare weight_scale_encoding \"csf\" and some do not (one export, one form)");
      if (mtp_csf_seen && mtp_plain_seen)
        bad("the draft experts mix weight_scale_encoding \"csf\" and plain entries (one export, one form)");
      c.experts_fp8 = false;
      c.experts_nvfp4 = true;
      c.experts_nvfp4_csf = csf_seen;
      c.mtp_experts_nvfp4_csf = mtp_csf_seen;
      c.dense_mxfp8_shipped = true;
      c.mtp_experts_nvfp4 = c.mtp_layer() >= 0;
      c.ngram_table_nvfp4 = table_seen;
      c.ngram_table_fp8 = false;
      c.norms_f32 = true;
      (void)c.ngram_geometry();
      return c;
    }
    // The AutoRound hybrid (docs/qwen38_autoround_int4_plan.md D7): the
    // GPTQ block the hybrid's tools write. Strict on purpose: the codes and
    // scales the packed core reads, the one `+` rule lifting the lm_head to
    // 8 bits, and `-` rules excluding exactly the classes the engine reads
    // unpacked. Intel's raw release (BF16 head and side layers) and the
    // base hybrid (BF16 draft experts, `-:.*layers\.48\..*`) are refused
    // by name: the served checkpoint is the -MTP_int4RTN variant.
    if (method == "gptq") {
      auto bad = [](const std::string& what) {
        throw std::runtime_error("Qwen quantization_config (gptq): " + what);
      };
      if (require_int(q, "bits") != 4) bad("only 4-bit codes are implemented");
      if (require_int(q, "group_size") != 128) bad("only group 128 is implemented");
      if (!optional_bool(q, "sym", false)) bad("only symmetric codes are implemented");
      if (optional_bool(q, "desc_act", false)) bad("act-order (desc_act) is not implemented");
      if (!optional_bool(q, "lm_head", false))
        bad("the hybrid's int8 lm_head (lm_head: true) is required — Intel's raw AutoRound release is not served");
      const minijson::Value* dyn = q.find("dynamic");
      if (dyn == nullptr || !dyn->is_object()) bad("the dynamic rule map is missing");
      bool head8 = false;
      std::vector<std::string> excluded;
      for (const auto& m : dyn->members()) {
        if (m.key.rfind("+:", 0) == 0) {
          if (m.key != "+:.*lm_head$") bad("unsupported + rule '" + m.key + "'");
          if (!m.value.is_object() || require_int(m.value, "bits") != 8)
            bad("the lm_head rule must lift it to 8 bits");
          head8 = true;
        } else if (m.key.rfind("-:", 0) == 0) {
          excluded.push_back(m.key.substr(2));
        } else {
          bad("unknown dynamic rule '" + m.key + "'");
        }
      }
      if (!head8) bad("the int8 lm_head rule (+:.*lm_head$) is missing — Intel's raw AutoRound release is not served");
      for (const std::string& e : excluded)
        if (e.find("layers\\.") != std::string::npos)
          bad("a whole layer's experts are excluded ('" + e +
              "': the base hybrid's BF16 draft experts) — serve the -MTP_int4RTN variant");
      for (const char* need : {"linear_attn", "self_attn", "hyper_connection", "shared_expert",
                               "\\.ple\\.", "embed", "fc_hidden", "\\.gate$"}) {
        bool found = false;
        for (const std::string& e : excluded) found = found || e.find(need) != std::string::npos;
        if (!found)
          bad(std::string("the exclusion of '") + need + "' is missing (the engine reads that class unpacked)");
      }
      c.experts_fp8 = false;
      c.experts_nvfp4 = false;
      c.experts_gptq_int4 = true;
      c.lm_head_gptq_int8 = true;
      c.dense_fp8_shipped = true;
      c.gptq_group = 128;
      c.ngram_table_fp8 = true;  // served from the FP8 release's shards (engine.ngram_table_dir)
      (void)c.ngram_geometry();
      return c;
    }
    if (method != "fp8")
      throw std::runtime_error("Qwen quantization_config.quant_method: only fp8, gptq and modelopt are implemented, got '" + method + "'");
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
  QwenTextConfig c = parse(*tc, parsed.root.find("quantization_config"));
  // A checkpoint that ships a vision tower serves images with it; the root
  // object holds both the tower's config and the image token ids.
  if (const auto* vision = parsed.root.find("vision_config"); vision && !vision->is_null())
    c.vision = QwenVisionConfig::parse(parsed.root, c.hidden_size);
  return c;
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
