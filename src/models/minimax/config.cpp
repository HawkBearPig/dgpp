#include "models/minimax/config.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "models/minimax/dspark.hpp"

namespace dgpp {
namespace {
using V = minijson::Value;
[[noreturn]] void bad(const std::string& key, const std::string& reason) {
  throw std::invalid_argument("MiniMax config." + key + ": " + reason);
}
const V& field(const V& obj, const char* key) {
  const auto* v = obj.find(key);
  if (!v || v->is_null()) bad(key, "missing");
  return *v;
}
int integer(const V& obj, const char* key) {
  const auto& v = field(obj, key);
  if (v.kind() != V::Kind::Int || v.as_int() < 0 || v.as_int() > std::numeric_limits<int>::max())
    bad(key, "expected a nonnegative integer in int range");
  return static_cast<int>(v.as_int());
}
double real(const V& obj, const char* key) {
  const auto& v = field(obj, key);
  if (!v.is_number() || !std::isfinite(v.as_double())) bad(key, "expected a finite number");
  return v.as_double();
}
void text(const V& obj, const char* key, const char* expected) {
  const auto& v = field(obj, key);
  if (!v.is_string() || v.as_string() != expected) bad(key, std::string("expected ") + expected);
}
void boolean(const V& obj, const char* key, bool expected) {
  const auto& v = field(obj, key);
  if (!v.is_bool() || v.as_bool() != expected)
    bad(key, expected ? "must be true" : "must be false");
}
void ones(const V& obj, const char* key, int n) {
  const auto& v = field(obj, key);
  if (!v.is_array() || v.items().size() != static_cast<size_t>(n)) bad(key, "wrong layer count");
  for (const auto& x : v.items())
    if (x.kind() != V::Kind::Int || x.as_int() != 1)
      bad(key, "M3.1 requires every entry to be one");
}
}  // namespace

MinimaxConfig MinimaxConfig::parse(const V& root) {
  text(root, "model_type", "minimax_m3_vl");
  if (const auto* arch = root.find("architectures")) {
    if (!arch->is_array() || arch->items().size() != 1 || !arch->items()[0].is_string() ||
        arch->items()[0].as_string() != "MiniMaxM3SparseForConditionalGeneration")
      bad("architectures", "expected MiniMaxM3SparseForConditionalGeneration");
  }
  const auto& t = field(root, "text_config");
  if (!t.is_object()) bad("text_config", "expected object");
  MinimaxConfig c;
#define READ_INT(name) c.name = integer(t, #name)
  READ_INT(hidden_size);
  READ_INT(vocab_size);
  READ_INT(num_hidden_layers);
  READ_INT(num_attention_heads);
  READ_INT(num_key_value_heads);
  READ_INT(head_dim);
  READ_INT(rotary_dim);
  READ_INT(max_position_embeddings);
  READ_INT(intermediate_size);
  READ_INT(dense_intermediate_size);
  READ_INT(shared_intermediate_size);
  READ_INT(num_local_experts);
  READ_INT(num_experts_per_tok);
#undef READ_INT
  if (t.find("eos_token_id")) c.eos_token_ids = {integer(t, "eos_token_id")};
  c.rope_theta = real(t, "rope_theta");
  c.rms_norm_eps = static_cast<float>(real(t, "rms_norm_eps"));
  c.routed_scaling_factor = static_cast<float>(real(t, "routed_scaling_factor"));
  c.swiglu_alpha = static_cast<float>(real(t, "swiglu_alpha"));
  c.swiglu_limit = static_cast<float>(real(t, "swiglu_limit"));
  boolean(t, "use_gemma_norm", true);
  boolean(t, "use_qk_norm", true);
  boolean(t, "attention_output_gate", false);
  boolean(t, "tie_word_embeddings", false);
  boolean(t, "use_routing_bias", true);
  text(t, "qk_norm_type", "per_head");
  text(t, "hidden_act", "swigluoai");
  text(t, "scoring_func", "sigmoid");
  if (integer(t, "n_shared_experts") != 1) bad("n_shared_experts", "expected one");
  if (real(t, "partial_rotary_factor") * c.head_dim != c.rotary_dim)
    bad("partial_rotary_factor", "does not match rotary_dim");
  if (const auto* r = t.find("rope_scaling"); r && !r->is_null())
    bad("rope_scaling", "unsupported");
  const auto& pattern = field(t, "moe_layer_freq");
  if (!pattern.is_array() || pattern.items().size() != static_cast<size_t>(c.num_hidden_layers))
    bad("moe_layer_freq", "wrong layer count");
  for (const auto& x : pattern.items()) {
    if (x.kind() != V::Kind::Int || (x.as_int() != 0 && x.as_int() != 1))
      bad("moe_layer_freq", "expected 0 or 1");
    c.moe_layer.push_back(static_cast<uint8_t>(x.as_int()));
  }
  const auto& a = field(t, "sparse_attention_config");
  boolean(a, "use_sparse_attention", true);
  text(a, "sparse_score_type", "max");
  c.index_heads = integer(a, "sparse_num_index_heads");
  c.index_dim = integer(a, "sparse_index_dim");
  c.sparse_block_size = integer(a, "sparse_block_size");
  c.sparse_topk_blocks = integer(a, "sparse_topk_blocks");
  c.sparse_local_blocks = integer(a, "sparse_local_block");
  if (integer(a, "sparse_init_block") != 0) bad("sparse_init_block", "only zero is implemented");
  ones(a, "sparse_attention_freq", c.num_hidden_layers);
  ones(a, "sparse_disable_index_value", c.num_hidden_layers);
  const auto& q = field(root, "quantization_config");
  text(q, "quant_method", "mxfp8");
  text(q, "activation_scheme", "dynamic");
  text(q, "moe_quant_algo", "NVFP4");
  text(q, "moe_quant_format", "nvfp4-pack-quantized");
  if (integer(q, "group_size") != 16) bad("group_size", "NVFP4 requires 16");
  const auto& blocks = field(q, "weight_block_size");
  if (!blocks.is_array() || blocks.items().size() != 2 ||
      blocks.items()[0].kind() != V::Kind::Int || blocks.items()[1].kind() != V::Kind::Int ||
      blocks.items()[0].as_int() != 1 || blocks.items()[1].as_int() != 32)
    bad("weight_block_size", "MXFP8 requires [1,32]");
  c.validate();
  if (const auto* v = root.find("vision_config"); v && !v->is_null())
    c.vision = MinimaxVisionConfig::parse(root, c.hidden_size, c.vocab_size);
  return c;
}

MinimaxConfig MinimaxConfig::from_json_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open MiniMax config: " + path);
  const std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  auto c = parse(minijson::parse_document(data).root);
  const auto slash = path.find_last_of('/');
  std::ifstream gen((slash == std::string::npos ? std::string{} : path.substr(0, slash + 1)) +
                    "generation_config.json");
  if (gen) {
    const std::string generation((std::istreambuf_iterator<char>(gen)),
                                 std::istreambuf_iterator<char>());
    const auto json = minijson::parse_document(generation);
    if (json.root.find("eos_token_id")) c.eos_token_ids = {integer(json.root, "eos_token_id")};
  }
  c.validate();
  const auto draft = std::filesystem::path(path).parent_path() / "dspark" / "config.json";
  if (std::filesystem::exists(draft)) {
    c.dspark = std::make_shared<DFlash2Config>(minimax_dspark_config_file(draft.string()));
    c.dspark->validate_against(
        {c.hidden_size, c.vocab_size, c.num_hidden_layers, c.max_position_embeddings});
  }
  return c;
}

void MinimaxConfig::validate() const {
  if (hidden_size <= 0 || hidden_size > 8192 || hidden_size % 64)
    bad("hidden_size", "must be a positive multiple of 64 up to 8192");
  if (vocab_size <= 0) bad("vocab_size", "must be positive");
  if (eos_token_ids.empty()) bad("eos_token_id", "must be present");
  for (int64_t eos : eos_token_ids)
    if (eos < 0 || eos >= vocab_size) bad("eos_token_id", "outside vocabulary");
  if (num_hidden_layers <= 0 || moe_layer.size() != static_cast<size_t>(num_hidden_layers))
    bad("num_hidden_layers", "must match the nonempty layer pattern");
  for (const auto v : moe_layer)
    if (v > 1) bad("moe_layer_freq", "expected 0 or 1");
  if (num_attention_heads <= 0 || num_key_value_heads <= 0 ||
      num_attention_heads % num_key_value_heads)
    bad("num_attention_heads", "must be a positive multiple of num_key_value_heads");
  if (head_dim != 128 || index_dim != 128) bad("head_dim", "main and index heads must be 128 wide");
  if (rotary_dim <= 0 || rotary_dim > head_dim || rotary_dim % 2)
    bad("rotary_dim", "invalid partial RoPE width");
  if (index_heads <= 0 || num_attention_heads % index_heads)
    bad("sparse_num_index_heads", "must divide query heads");
  if (index_heads != num_key_value_heads || num_attention_heads / num_key_value_heads > 32)
    bad("sparse_num_index_heads",
        "requires one index head per KV head, at most 32 query heads per group");
  if (sparse_block_size != 128 || sparse_topk_blocks <= 0 || sparse_topk_blocks > 16 ||
      sparse_local_blocks != 1)
    bad("sparse_attention_config", "requires block 128, top-k in [1,16], one local block");
  for (const int n : {intermediate_size, dense_intermediate_size, shared_intermediate_size})
    if (n <= 0 || n % 32) bad("intermediate_size", "must be positive multiples of 32");
  if (num_local_experts <= 0 || num_local_experts > 1024 || num_experts_per_tok <= 0 ||
      num_experts_per_tok > 16 || num_experts_per_tok > num_local_experts)
    bad("num_experts_per_tok", "invalid expert/top-k dimensions");
  if (max_position_embeddings <= 0 || !(rope_theta > 0) || !std::isfinite(rope_theta))
    bad("rope_theta", "invalid position range or base");
  for (float v : {rms_norm_eps, swiglu_alpha, swiglu_limit, routed_scaling_factor})
    if (!(v > 0) || !std::isfinite(v))
      bad("numerics", "norm, activation and routing constants must be finite and positive");
}

void MinimaxConfig::validate_tp(int rank, int world) const {
  validate();
  if (world < 1 || rank < 0 || rank >= world) bad("tp", "rank/world out of range");
  if (num_attention_heads % world || num_key_value_heads % world || index_heads % world)
    bad("tp", "world must divide query, KV and index head counts");
  for (int n : {intermediate_size, dense_intermediate_size, shared_intermediate_size})
    if (n % world || (n / world) % 64)
      bad("tp", "intermediate slices must align to 64-wide kernel tiles");
}

int MinimaxConfig::num_moe_layers() const {
  return static_cast<int>(std::count(moe_layer.begin(), moe_layer.end(), uint8_t{1}));
}
}  // namespace dgpp
