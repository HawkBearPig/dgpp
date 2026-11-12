#include "models/dsv4/config.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <stdexcept>

namespace dgpp {
namespace {

[[noreturn]] void reject(std::string_view field, std::string_view why) {
  throw std::runtime_error(std::format("DeepSeek-V4 config.{}: {}", field, why));
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
int64_t require_int64(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_number()) reject(field, "not a number");
  return f.as_int();
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
std::vector<int> require_int_list(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_array()) reject(field, "not an array");
  std::vector<int> out;
  for (const auto& item : f.items()) {
    if (!item.is_number()) reject(field, "non-numeric element");
    out.push_back(static_cast<int>(item.as_int()));
  }
  return out;
}
// A field that must be absent or null (a feature the engine does not
// implement and must not silently ignore).
void require_absent(const minijson::Value& v, std::string_view field, std::string_view why) {
  const minijson::Value* f = v.find(field);
  if (f && !f->is_null()) reject(field, why);
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

bool contains(const std::vector<int>& v, int x) { return std::find(v.begin(), v.end(), x) != v.end(); }
bool strictly_increasing(const std::vector<int>& v) {
  for (size_t i = 1; i < v.size(); ++i)
    if (v[i] <= v[i - 1]) return false;
  return true;
}

// --- the quantization contract --------------------------------------------

void parse_quantization(const minijson::Value& root, Dsv4Config& c) {
  const minijson::Value* qc = root.find("quantization_config");
  if (!qc || qc->is_null())
    reject("quantization_config",
           "missing — the engine serves the release's own mixed fp8 / fp4 format; a BF16 checkpoint has no "
           "expert path here");
  if (!qc->is_object()) reject("quantization_config", "not an object");
  if (const std::string m = optional_string(*qc, "quant_method", ""); m != "fp8")
    reject("quantization_config.quant_method",
           "only the release's fp8 method is implemented, got '" + m +
               "' (a re-quantized repack — int4, EXL3, NVFP4 — is not the format this loader reads)");
  if (const std::string f = optional_string(*qc, "fmt", "e4m3"); f != "e4m3")
    reject("quantization_config.fmt", "the dense matrices must be e4m3, got '" + f + "'");
  if (const std::string a = optional_string(*qc, "activation_scheme", "dynamic"); a != "dynamic")
    reject("quantization_config.activation_scheme", "only dynamic is implemented, got '" + a + "'");
  if (const std::string s = optional_string(*qc, "scale_fmt", ""); s != "ue8m0")
    reject("quantization_config.scale_fmt", "the loader reads e8m0 (power-of-two) scales, got '" + s + "'");
  const minijson::Value& wb = require(*qc, "weight_block_size");
  if (!wb.is_array() || wb.items().size() != 2 || !wb.items()[0].is_number() || !wb.items()[1].is_number())
    reject("quantization_config.weight_block_size", "must be [rows, cols]");
  const int br = static_cast<int>(wb.items()[0].as_int()), bc = static_cast<int>(wb.items()[1].as_int());
  if (br != 128 || bc != 128)
    reject("quantization_config.weight_block_size",
           std::format("the release's dense grid is 128x128, got [{}, {}]", br, bc));
  c.fp8_block_size = br;
  // The routed experts' format sits at the root in this release.
  const std::string e = optional_string(root, "expert_dtype", optional_string(*qc, "expert_dtype", ""));
  if (e != "fp4")
    reject("expert_dtype", "the routed experts must be the release's MXFP4 (e2m1 + e8m0 per 32), got '" + e + "'");
  c.fp4_block_size = 32;  // the release's MXFP4 block (model.py fp4_block_size)
  require_absent(*qc, "config_groups", "a compressed-tensors layout is not the release format the loader reads");
  require_absent(*qc, "bits", "a re-quantized repack is not the release format the loader reads");
}

}  // namespace

Dsv4Config Dsv4Config::parse(const minijson::Value& root) {
  if (!root.is_object()) reject("", "root is not an object");
  Dsv4Config c;
  const std::string model_type = optional_string(root, "model_type", "deepseek_v4");
  if (model_type != "deepseek_v4") reject("model_type", "expected deepseek_v4, got " + model_type);
  require_absent(root, "text_config", "a nested text_config is the DeepSeek-V4.1 layout, not this family's");
  require_absent(root, "vision_config", "the vision variant of this family is not implemented");
  const minijson::Value& t = root;

  // --- tokens -------------------------------------------------------------------
  c.bos_token_id = optional_int64(t, "bos_token_id", 0);
  c.eos_token_id = require_int64(t, "eos_token_id");

  // --- shape --------------------------------------------------------------------
  c.hidden_size = require_int(t, "hidden_size");
  c.vocab_size = require_int(t, "vocab_size");
  c.num_hidden_layers = require_int(t, "num_hidden_layers");
  c.rms_norm_eps = static_cast<float>(require_double(t, "rms_norm_eps"));
  c.tie_word_embeddings = optional_bool(t, "tie_word_embeddings", false);
  c.hidden_act = optional_string(t, "hidden_act", "silu");
  c.max_position_embeddings = require_int(t, "max_position_embeddings");
  if (c.hidden_size <= 0 || c.hidden_size % 128 != 0)
    reject("hidden_size", "must be a positive multiple of 128 (the fp8 scale block)");
  if (c.vocab_size <= 0) reject("vocab_size", "must be positive");
  if (c.num_hidden_layers <= 0) reject("num_hidden_layers", "must be positive");
  if (!(c.rms_norm_eps > 0.f)) reject("rms_norm_eps", "must be positive");
  if (c.hidden_act != "silu") reject("hidden_act", "only silu is implemented, got " + c.hidden_act);
  if (c.tie_word_embeddings) reject("tie_word_embeddings", "tied embeddings are not implemented");
  if (c.max_position_embeddings <= 0) reject("max_position_embeddings", "must be positive");
  for (const int64_t id : {c.bos_token_id, c.eos_token_id})
    if (id < 0 || id >= c.vocab_size) reject("eos_token_id", "id outside [0, vocab_size)");

  // --- attention ------------------------------------------------------------
  c.num_attention_heads = require_int(t, "num_attention_heads");
  c.num_key_value_heads = optional_int(t, "num_key_value_heads", 1);
  c.head_dim = require_int(t, "head_dim");
  c.qk_rope_head_dim = require_int(t, "qk_rope_head_dim");
  c.q_lora_rank = require_int(t, "q_lora_rank");
  c.o_lora_rank = require_int(t, "o_lora_rank");
  c.o_groups = require_int(t, "o_groups");
  c.attention_bias = optional_bool(t, "attention_bias", false);
  c.sliding_window = require_int(t, "sliding_window");
  if (c.num_attention_heads <= 0) reject("num_attention_heads", "must be positive");
  if (c.num_key_value_heads != 1)
    reject("num_key_value_heads", "the attention keeps one latent per token that is both key and value: must be 1");
  if (c.head_dim != 512) reject("head_dim", "the attention kernels are compiled for the release's 512-wide latent");
  if (c.qk_rope_head_dim != 64) reject("qk_rope_head_dim", "the rope kernels rotate the release's 64-wide tail");
  if (c.q_lora_rank <= 0 || c.q_lora_rank % 128 != 0)
    reject("q_lora_rank", "must be a positive multiple of 128 (an fp8 block)");
  if (c.o_lora_rank <= 0 || c.o_lora_rank % 128 != 0)
    reject("o_lora_rank", "must be a positive multiple of 128 (an fp8 block)");
  if (c.o_groups <= 0 || c.num_attention_heads % c.o_groups != 0)
    reject("o_groups", "must divide num_attention_heads");
  if (c.attention_bias) reject("attention_bias", "biased attention projections are not implemented");
  if (c.sliding_window <= 0 || c.sliding_window > 128)
    reject("sliding_window", "must be in [1, 128] (the window ring and its attention lists)");
  c.rope_theta = optional_double(t, "rope_theta", 10000.0);
  c.compress_rope_theta = optional_double(t, "compress_rope_theta", c.rope_theta);
  if (!(c.rope_theta > 0) || !(c.compress_rope_theta > 0)) reject("rope_theta", "must be positive");
  {
    const minijson::Value& rs = require(t, "rope_scaling");
    if (!rs.is_object()) reject("rope_scaling", "not an object");
    const std::string rt = optional_string(rs, "type", optional_string(rs, "rope_type", "default"));
    if (rt != "yarn") reject("rope_scaling.type", "the compressed layers' rope is YaRN, got '" + rt + "'");
    c.rope_factor = require_double(rs, "factor");
    c.original_max_position_embeddings = require_int(rs, "original_max_position_embeddings");
    c.beta_fast = optional_double(rs, "beta_fast", 32.0);
    c.beta_slow = optional_double(rs, "beta_slow", 1.0);
    if (!(c.rope_factor >= 1.0)) reject("rope_scaling.factor", "must be >= 1");
    if (c.original_max_position_embeddings <= 0)
      reject("rope_scaling.original_max_position_embeddings", "must be positive");
    if (!(c.beta_fast > c.beta_slow) || !(c.beta_slow > 0))
      reject("rope_scaling.beta_fast", "must exceed beta_slow > 0");
    require_absent(rs, "mscale", "a YaRN attention mscale is not part of the release (softmax scale is head_dim^-0.5)");
    require_absent(rs, "mscale_all_dim", "a YaRN attention mscale is not part of the release");
  }

  // --- compression schedule and the draft stages -------------------------------
  c.compress_ratios = require_int_list(t, "compress_ratios");
  c.dspark_block_size = optional_int(t, "dspark_block_size", 0);
  if (static_cast<int>(c.compress_ratios.size()) < c.num_hidden_layers)
    reject("compress_ratios", std::format("must list every backbone layer ({} entries at least), got {}",
                                          c.num_hidden_layers, c.compress_ratios.size()));
  c.draft_stages = static_cast<int>(c.compress_ratios.size()) - c.num_hidden_layers;
  if (c.dspark_block_size <= 0 && c.draft_stages != 0)
    reject("compress_ratios", "lists draft stages but dspark_block_size is absent");
  if (c.draft_stages > 8) reject("compress_ratios", "more than 8 draft stages");
  for (int l = 0; l < c.max_layer(); ++l) {
    const int r = c.compress_ratios[static_cast<size_t>(l)];
    if (r != 0 && r != 4 && r != 128)
      reject("compress_ratios",
             std::format("layer {}: the compressor implements ratios 0, 4 (overlapping, indexed) and 128, got {}", l, r));
    if (c.is_draft(l) && r != 0)
      reject("compress_ratios", std::format("draft stage {} must be window-only (ratio 0)", c.draft_stage(l)));
  }
  c.index_n_heads = require_int(t, "index_n_heads");
  c.index_head_dim = require_int(t, "index_head_dim");
  c.index_topk = require_int(t, "index_topk");
  if (c.index_n_heads != 32 && c.index_n_heads != 64)
    reject("index_n_heads", "the selection kernels implement 32 or 64 index heads");
  if (c.index_head_dim != 128) reject("index_head_dim", "the indexer implements 128 (the index cache row)");
  if (c.index_topk <= 0 || (c.index_topk & (c.index_topk - 1)) != 0 || c.index_topk > 1024)
    reject("index_topk", "must be a power of two <= 1024 (the select networks)");

  // --- mHC ------------------------------------------------------------------------
  c.hc_mult = require_int(t, "hc_mult");
  c.hc_sinkhorn_iters = require_int(t, "hc_sinkhorn_iters");
  c.hc_eps = static_cast<float>(require_double(t, "hc_eps"));
  if (c.hc_mult != 4) reject("hc_mult", "the mHC kernel is pinned to 4 streams");
  if (c.hc_sinkhorn_iters < 1) reject("hc_sinkhorn_iters", "must be >= 1");
  if (!(c.hc_eps > 0)) reject("hc_eps", "must be positive");

  // --- MoE ------------------------------------------------------------------------
  c.moe_intermediate_size = require_int(t, "moe_intermediate_size");
  c.n_routed_experts = require_int(t, "n_routed_experts");
  c.n_shared_experts = optional_int(t, "n_shared_experts", 0);
  c.num_experts_per_tok = require_int(t, "num_experts_per_tok");
  c.norm_topk_prob = optional_bool(t, "norm_topk_prob", true);
  c.routed_scaling_factor = static_cast<float>(optional_double(t, "routed_scaling_factor", 1.0));
  c.swiglu_limit = static_cast<float>(optional_double(t, "swiglu_limit", 0.0));
  c.scoring_func = optional_string(t, "scoring_func", "sqrtsoftplus");
  c.topk_method = optional_string(t, "topk_method", "noaux_tc");
  c.num_hash_layers = optional_int(t, "num_hash_layers", 0);
  if (c.moe_intermediate_size <= 0 || c.moe_intermediate_size % 128 != 0)
    reject("moe_intermediate_size", "must be a positive multiple of 128 (the shared expert's fp8 block)");
  if (c.n_routed_experts <= 0 || c.n_routed_experts > 4096) reject("n_routed_experts", "must be in [1, 4096]");
  if (c.num_experts_per_tok <= 0 || c.num_experts_per_tok > 16 || c.num_experts_per_tok > c.n_routed_experts)
    reject("num_experts_per_tok", "must be in [1, min(n_routed_experts, 16)]");
  if (c.n_shared_experts != 1) reject("n_shared_experts", "the MoE chain implements exactly one shared expert");
  if (!c.norm_topk_prob) reject("norm_topk_prob", "the release normalizes the picked scores; false is not implemented");
  if (!(c.routed_scaling_factor > 0)) reject("routed_scaling_factor", "must be positive");
  if (!(c.swiglu_limit > 0)) reject("swiglu_limit", "the release clamps its SwiGLU; a positive limit is required");
  if (c.scoring_func != "sqrtsoftplus")
    reject("scoring_func", "sqrtsoftplus is the release's rule, got '" + c.scoring_func + "'");
  if (c.topk_method != "noaux_tc") reject("topk_method", "only noaux_tc is implemented");
  if (c.num_hash_layers < 0 || c.num_hash_layers > c.num_hidden_layers)
    reject("num_hash_layers", "must be in [0, num_hidden_layers]");
  if (const int ng = optional_int(t, "n_group", 1); ng != 1) reject("n_group", "group-limited routing is not implemented");
  if (const int tg = optional_int(t, "topk_group", 1); tg != 1) reject("topk_group", "group-limited routing is not implemented");

  // --- DSpark ----------------------------------------------------------------------
  c.dspark_noise_token_id = optional_int64(t, "dspark_noise_token_id", -1);
  c.dspark_target_layer_ids =
      t.find("dspark_target_layer_ids") ? require_int_list(t, "dspark_target_layer_ids") : std::vector<int>{};
  c.dspark_markov_rank = optional_int(t, "dspark_markov_rank", 0);
  if (c.draft_stages > 0) {
    if (c.dspark_noise_token_id < 0 || c.dspark_noise_token_id >= c.vocab_size)
      reject("dspark_noise_token_id", "id outside [0, vocab_size)");
    if (c.dspark_target_layer_ids.empty()) reject("dspark_target_layer_ids", "DSpark needs target layers");
    if (!strictly_increasing(c.dspark_target_layer_ids)) reject("dspark_target_layer_ids", "must be strictly increasing");
    for (const int l : c.dspark_target_layer_ids)
      if (l < 0 || l >= c.num_hidden_layers)
        reject("dspark_target_layer_ids", std::format("layer {} outside the backbone", l));
    if (c.dspark_markov_rank <= 0 || c.dspark_markov_rank % 32 != 0)
      reject("dspark_markov_rank", "must be a positive multiple of 32");
  }

  parse_quantization(root, c);
  return c;
}

Dsv4Config Dsv4Config::from_json_file(const std::string& path) {
  const std::string json = read_file(path);
  const auto parsed = minijson::parse(json);
  return parse(parsed.root);
}

int Dsv4Config::cache_ordinal(int l) const {
  if (l < 0 || l >= max_layer() || compress_ratio(l) == 0) return -1;
  int ord = 0;
  for (int i = 0; i < l; ++i)
    if (compress_ratio(i) > 0) ++ord;
  return ord;
}

int Dsv4Config::index_ordinal(int l) const {
  if (l < 0 || l >= max_layer() || !indexed(l)) return -1;
  int ord = 0;
  for (int i = 0; i < l; ++i)
    if (indexed(i)) ++ord;
  return ord;
}

int Dsv4Config::num_caches() const {
  int n = 0;
  for (int l = 0; l < max_layer(); ++l)
    if (compress_ratio(l) > 0) ++n;
  return n;
}

int Dsv4Config::num_index_caches() const {
  int n = 0;
  for (int l = 0; l < max_layer(); ++l)
    if (indexed(l)) ++n;
  return n;
}

bool Dsv4Config::is_dspark_target(int l) const { return contains(dspark_target_layer_ids, l); }

GlmMoeConfig Dsv4Config::moe_config(int local_inter) const {
  GlmMoeConfig m;
  m.hidden = hidden_size;
  m.inter = local_inter;
  m.n_experts = n_routed_experts;
  m.top_k = num_experts_per_tok;
  m.n_shared_experts = n_shared_experts;
  m.routed_scaling_factor = routed_scaling_factor;
  m.norm_topk_prob = norm_topk_prob;
  m.swiglu_limit = swiglu_limit;
  m.router_mode = MoeRouterMode::SqrtSoftplusBias;
  // The decode walk's shared expert: aside on the tensor cores where the
  // shapes allow (GlmMoeConfig::shared_mma_aside), else in the slots, beside
  // the first routed ones as in every family.
  m.shared_mma_aside = true;
  GlmMoeConfig::validate_config(m);
  return m;
}

}  // namespace dgpp
