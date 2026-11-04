#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

#include "engine/decode_outputs.hpp"
#include "models/minimax/dspark.hpp"

namespace dgpp {
namespace {
using V = minijson::Value;
[[noreturn]] void bad(std::string_view field, const char* why) {
  throw std::invalid_argument("MiniMax DSpark config." + std::string(field) + ": " + why);
}
const V& need(const V& v, std::string_view field) {
  const auto* p = v.find(field);
  if (!p) bad(field, "missing");
  return *p;
}
int integer(const V& v, const std::string& field, int low, int high) {
  const auto& p = need(v, field);
  if (p.kind() != V::Kind::Int || p.as_int() < low || p.as_int() > high)
    bad(field, "integer outside the supported range");
  return static_cast<int>(p.as_int());
}
double number(const V& v, const std::string& field) {
  const auto& p = need(v, field);
  if (!p.is_number() || !std::isfinite(p.as_double()) || p.as_double() <= 0)
    bad(field, "positive finite number required");
  return p.as_double();
}
void flag(const V& v, const std::string& field, bool expected) {
  const auto& p = need(v, field);
  if (!p.is_bool() || p.as_bool() != expected) bad(field, "unsupported value");
}
void word(const V& v, const std::string& field, const char* expected) {
  const auto& p = need(v, field);
  if (!p.is_string() || p.as_string() != expected) bad(field, "unsupported value");
}
void architecture(const V& v) {
  const auto& p = need(v, "architectures");
  if (!p.is_array() || p.items().size() != 1 || !p.items()[0].is_string() ||
      p.items()[0].as_string() != "DSparkMiniMaxDraftModel")
    bad("architectures", "DSparkMiniMaxDraftModel required");
}
// Duplicated root/text metadata is common in the published config. Reject
// disagreement rather than silently selecting one of two execution recipes.
bool equal(const V& a, const V& b) {
  if (a.is_number() && b.is_number()) return a.as_double() == b.as_double();
  if (a.kind() != b.kind()) return false;
  if (a.is_null()) return true;
  if (a.is_bool()) return a.as_bool() == b.as_bool();
  if (a.is_string()) return a.as_string() == b.as_string();
  if (a.is_array()) {
    if (a.items().size() != b.items().size()) return false;
    for (size_t i = 0; i < a.items().size(); ++i)
      if (!equal(a.items()[i], b.items()[i])) return false;
    return true;
  }
  if (a.members().size() != b.members().size()) return false;
  for (const auto& m : a.members()) {
    const auto* n = b.find(m.key);
    if (!n || !equal(m.value, *n)) return false;
  }
  return true;
}
}  // namespace

DFlash2Config minimax_dspark_config(const V& root) {
  architecture(root);
  word(root, "model_type", "minimax_m3_vl");
  word(root, "dtype", "bfloat16");
  const auto& t = need(root, "text_config");
  architecture(t);
  word(t, "dtype", "bfloat16");
  for (const auto& m : t.members())
    if (m.key.starts_with("dspark_") || m.key == "dflash_config" || m.key == "hidden_size" ||
        m.key == "vocab_size" || m.key == "num_hidden_layers" ||
        m.key == "enable_confidence_head" || m.key == "confidence_head_with_markov")
      if (const auto* other = root.find(m.key); other && !equal(*other, m.value))
        bad(m.key, "root and text_config disagree");
  DFlash2Config c;
  c.backbone = DFlash2Config::Backbone::MiniMaxM3;
  c.conv_taps = 0;
  c.hidden_size = integer(t, "hidden_size", 32, 16384);
  c.vocab_size = integer(t, "vocab_size", 16, 1048576);
  c.num_hidden_layers = integer(t, "num_hidden_layers", 1, 16);
  c.num_attention_heads = integer(t, "num_attention_heads", 1, 256);
  c.num_key_value_heads = integer(t, "num_key_value_heads", 1, c.num_attention_heads);
  c.head_dim = integer(t, "head_dim", 128, 128);
  c.rotary_dim = integer(t, "rotary_dim", 64, 64);
  c.intermediate_size = integer(t, "dense_intermediate_size", 32, 65536);
  if (c.hidden_size % 32 || c.intermediate_size % 32)
    bad("hidden_size/dense_intermediate_size", "must align to 32");
  if (c.num_attention_heads % c.num_key_value_heads ||
      c.num_attention_heads / c.num_key_value_heads > 16)
    bad("num_key_value_heads", "query/kv head ratio must be an integer in 1..16");
  c.rms_norm_eps = static_cast<float>(number(t, "rms_norm_eps"));
  c.rope_theta = number(t, "rope_theta");
  c.swiglu_alpha = static_cast<float>(number(t, "swiglu_alpha"));
  c.swiglu_limit = static_cast<float>(number(t, "swiglu_limit"));
  c.max_position_embeddings = integer(t, "max_position_embeddings", 128, 1048576);
  if (number(t, "partial_rotary_factor") != 0.5) bad("partial_rotary_factor", "0.5 required");
  for (const auto* f : {"use_gemma_norm", "use_qk_norm", "dspark_use_dense_ffn",
                        "enable_confidence_head", "confidence_head_with_markov"})
    flag(t, f, true);
  for (const auto* f :
       {"attention_output_gate", "tie_word_embeddings", "dspark_hybrid_context_fusion", "use_mtp"})
    flag(t, f, false);
  word(t, "hidden_act", "swigluoai");
  word(t, "qk_norm_type", "per_head");
  word(t, "dspark_markov_head_type", "vanilla");
  word(t, "dspark_aux_gemm_precision", "bf16");
  const auto& d = need(t, "dflash_config");
  c.block_size = 1 + integer(d, "block_size", 1, kSpecRows - 1);
  c.mask_token_id = integer(d, "mask_token_id", 0, c.vocab_size - 1);
  c.selector_rank = integer(t, "dspark_markov_rank", 256, 256);
  c.selector_top_k = 16;  // Proposal recipe; the target's exact verify remains authoritative.
  c.sliding_window = integer(d, "sliding_window", c.block_size + 1, c.max_position_embeddings);
  word(d, "context_source", "target_layers");
  const auto& taps = need(d, "target_layer_ids");
  if (!taps.is_array() || taps.items().empty() || taps.items().size() > 16)
    bad("target_layer_ids", "1..16 taps required");
  std::unordered_set<int> seen;
  for (const auto& v : taps.items()) {
    if (v.kind() != V::Kind::Int || v.as_int() < 0 || v.as_int() >= 256 ||
        !seen.insert(v.as_int()).second)
      bad("target_layer_ids", "distinct target input layer indices required");
    c.target_layer_ids.push_back(
        static_cast<int>(v.as_int()));  // FC columns retain checkpoint order.
  }
  for (const auto& [field, canonical] : {std::pair{"dspark_block_size", "block_size"},
                                         {"dspark_noise_token_id", "mask_token_id"},
                                         {"dspark_target_layer_ids", "target_layer_ids"}})
    if (!equal(need(t, field), need(d, canonical))) bad(field, "disagrees with dflash_config");
  if (integer(t, "dspark_ffn_hidden_size", 32, 65536) != c.intermediate_size ||
      integer(d, "num_hidden_layers", 1, 16) != c.num_hidden_layers ||
      integer(t, "sliding_window", 1, 1048576) != c.sliding_window)
    bad("dflash_config", "inconsistent backbone geometry");
  const auto& dc = need(t, "dspark_config");
  for (const auto& [field, canonical] : {std::pair{"markov_rank", "dspark_markov_rank"},
                                         {"markov_head_type", "dspark_markov_head_type"},
                                         {"mask_token_id", "dspark_noise_token_id"},
                                         {"hybrid_context_fusion", "dspark_hybrid_context_fusion"},
                                         {"aux_gemm_precision", "dspark_aux_gemm_precision"},
                                         {"use_dense_ffn", "dspark_use_dense_ffn"},
                                         {"ffn_hidden_size", "dspark_ffn_hidden_size"}})
    if (!equal(need(dc, field), need(t, canonical)))
      bad("dspark_config." + std::string(field), "inconsistent alias");
  const auto& moe = need(t, "moe_layer_freq");
  if (!moe.is_array() || moe.items().size() != static_cast<size_t>(c.num_hidden_layers))
    bad("moe_layer_freq", "dense layer list required");
  for (const auto& v : moe.items())
    if (v.kind() != V::Kind::Int || v.as_int() != 0)
      bad("moe_layer_freq", "only dense draft layers supported");
  for (const auto* owner : {&t, &d}) {
    const auto& types = need(*owner, "layer_types");
    if (!types.is_array() || types.items().size() != static_cast<size_t>(c.num_hidden_layers))
      bad("layer_types", "wrong layer count");
    for (const auto& v : types.items())
      if (!v.is_string() || v.as_string() != "sliding_attention")
        bad("layer_types", "sliding_attention required");
  }
  const auto& q = need(root, "quantization_config");
  word(q, "quant_method", "mxfp8");
  word(q, "activation_scheme", "dynamic");
  const auto& block = need(q, "weight_block_size");
  if (!block.is_array() || block.items().size() != 2 || block.items()[0].as_int() != 1 ||
      block.items()[1].as_int() != 32)
    bad("quantization_config.weight_block_size", "[1,32] required");
  return c;
}

DFlash2Config minimax_dspark_config_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::invalid_argument("MiniMax DSpark config: cannot open " + path);
  const std::string text((std::istreambuf_iterator<char>(in)), {});
  return minimax_dspark_config(minijson::parse(text).root);
}
}  // namespace dgpp
