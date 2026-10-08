#include "models/qwen/dflash.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "loaders/safetensors.hpp"

namespace dgpp {
namespace {

namespace fs = std::filesystem;

[[noreturn]] void reject(const std::string& field, const std::string& why) {
  throw std::runtime_error("DFlash v1 config " + field + ": " + why);
}

const minijson::Value& require(const minijson::Value& v, const char* key) {
  if (const minijson::Value* f = v.find(key); f != nullptr && !f->is_null()) return *f;
  reject(key, "missing");
  throw std::runtime_error("unreachable");
}

std::string str_at(const minijson::Value& v, const char* key, const std::string& dflt) {
  if (const minijson::Value* f = v.find(key); f != nullptr && !f->is_null()) {
    if (!f->is_string()) reject(key, "not a string");
    return std::string(f->as_string());
  }
  return dflt;
}

int int_at(const minijson::Value& v, const char* key, int dflt, bool required = false) {
  if (const minijson::Value* f = v.find(key); f != nullptr && !f->is_null()) {
    if (!f->is_number()) reject(key, "not a number");
    return static_cast<int>(f->as_int());
  }
  if (required) reject(key, "missing");
  return dflt;
}

int64_t i64_at(const minijson::Value& v, const char* key, int64_t dflt, bool required = false) {
  if (const minijson::Value* f = v.find(key); f != nullptr && !f->is_null()) {
    if (!f->is_number()) reject(key, "not a number");
    return f->as_int();
  }
  if (required) reject(key, "missing");
  return dflt;
}

double double_at(const minijson::Value& v, const char* key, double dflt) {
  if (const minijson::Value* f = v.find(key); f != nullptr && !f->is_null()) {
    if (!f->is_number()) reject(key, "not a number");
    return f->as_double();
  }
  return dflt;
}

}  // namespace

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

DFlashConfig DFlashConfig::from_json_file(const std::string& path) {
  const std::string text = read_file(path);
  minijson::Value root;
  try {
    root = minijson::parse(text).root;
  } catch (const std::exception& e) {
    throw std::runtime_error("DFlash v1 config " + path + ": " + e.what());
  }
  return parse(root);
}

DFlashConfig DFlashConfig::parse(const minijson::Value& root) {
  DFlashConfig c;
  const std::string arch =
      root.find("architectures") && root.at("architectures").is_array() &&
              !root.at("architectures").items().empty()
          ? std::string(root.at("architectures").items().front().as_string())
          : "";
  // v1 only: the DFlash2 draft (DFlash2DraftModel, selector + conv) parses
  // through DFlash2Config, never here.
  if (arch != "DFlashDraftModel") reject("architectures", "DFlashDraftModel required");
  if (str_at(root, "model_type", "") != "qwen3") reject("model_type", "qwen3 required");
  if (str_at(root, "dtype", "bfloat16") != "bfloat16") reject("dtype", "bfloat16 required");
  const minijson::Value& d = require(root, "dflash_config");
  c.block_size = int_at(d, "block_size", 16, true);
  c.mask_token_id = int_at(d, "mask_token_id", -1, true);
  const minijson::Value& taps = require(d, "target_layer_ids");
  if (!taps.is_array() || taps.items().empty()) reject("dflash_config.target_layer_ids", "empty");
  for (const auto& t : taps.items()) {
    if (!t.is_number()) reject("dflash_config.target_layer_ids", "non-numeric element");
    c.target_layer_ids.push_back(static_cast<int>(t.as_int()));
  }
  std::sort(c.target_layer_ids.begin(), c.target_layer_ids.end());
  c.num_hidden_layers = int_at(root, "num_hidden_layers", 6, true);
  c.hidden_size = int_at(root, "hidden_size", 3072, true);
  c.vocab_size = int_at(root, "vocab_size", 248320, true);
  c.num_attention_heads = int_at(root, "num_attention_heads", 32, true);
  c.num_key_value_heads = int_at(root, "num_key_value_heads", 8, true);
  c.head_dim = int_at(root, "head_dim", c.hidden_size / c.num_attention_heads, true);
  c.intermediate_size = int_at(root, "intermediate_size", 9216, true);
  c.rms_norm_eps = static_cast<float>(double_at(root, "rms_norm_eps", 1e-6));
  const minijson::Value& rp = require(root, "rope_parameters");
  if (str_at(rp, "rope_type", "default") != "default")
    reject("rope_parameters.rope_type", "only the plain default scaling is supported");
  c.rope_theta = double_at(rp, "rope_theta", 1e7);
  c.max_position_embeddings = i64_at(root, "max_position_embeddings", 262144);
  {
    const minijson::Value* sw = root.find("use_sliding_window");
    const bool sliding = sw && !sw->is_null() ? sw->as_bool(true) : true;
    c.sliding_window = sliding ? i64_at(root, "sliding_window", 4096, true) : INT64_MAX;
  }
  if (c.block_size < 2) reject("dflash_config.block_size", "at least 2");
  if (c.num_hidden_layers < 1 || c.num_hidden_layers > 16)
    reject("num_hidden_layers", "1..16 supported");
  if (c.head_dim != 128) reject("head_dim", "only 128 is supported");
  if (c.num_attention_heads % c.num_key_value_heads ||
      c.num_attention_heads / c.num_key_value_heads > 8)
    reject("num_key_value_heads", "heads/kv_heads must be in [1, 8]");
  if (const minijson::Value* tw = root.find("tie_word_embeddings"))
    if (tw->as_bool(false))
      reject("tie_word_embeddings", "the draft shares the target's head; ties are not loaded");
  return c;
}

void DFlashConfig::validate_against(const Qwen35TextConfig& target) const {
  if (hidden_size != target.hidden_size)
    reject("hidden_size", "the draft must share the target's width (fc and the shared head)");
  if (vocab_size != target.vocab_size)
    reject("vocab_size", "the draft must share the target's vocabulary (mask id, shared head)");
  if (mask_token_id < 0 || mask_token_id >= target.vocab_size)
    reject("dflash_config.mask_token_id", "outside the target vocabulary");
  if (block_size - 1 > 15)
    reject("dflash_config.block_size", "the engine verifies at most 15 drafts per step");
  for (size_t i = 0; i < target_layer_ids.size(); ++i) {
    if (target_layer_ids[i] < 0 || target_layer_ids[i] >= target.num_hidden_layers)
      reject("dflash_config.target_layer_ids",
             "layer " + std::to_string(target_layer_ids[i]) + " does not exist in the target");
    if (i && target_layer_ids[i] == target_layer_ids[i - 1])
      reject("dflash_config.target_layer_ids", "duplicate tap layer");
  }
  if (sliding_window <= block_size)
    reject("sliding_window", "the window must cover the draft block");
}

size_t dflash_weights_bytes(const DFlashConfig& cfg, int world) {
  if (!cfg.tp_divisible(world))
    throw std::invalid_argument("dflash_weights_bytes: the drafter does not split across this world");
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t QW = static_cast<size_t>(cfg.local_q_row(world));
  const size_t KW = static_cast<size_t>(cfg.local_kv_row(world));
  const size_t I = static_cast<size_t>(cfg.local_intermediate(world));
  const size_t per_layer = (QW + 2 * KW) * H + H * QW + 2 * I * H + H * I + 2 * H + 2 * cfg.head_dim;
  const size_t total =
      static_cast<size_t>(cfg.num_hidden_layers) * per_layer +
      static_cast<size_t>(cfg.target_layer_ids.size()) * H * H + 2 * H;
  return total * 2;
}

DFlashWeights load_dflash_weights(const DFlashConfig& cfg, const std::string& dir,
                                  cudaStream_t stream, int rank, int world) {
  if (!cfg.tp_divisible(world))
    throw std::invalid_argument("load_dflash_weights: the drafter does not split across this world");
  std::vector<std::string> shards;
  for (const auto& e : fs::directory_iterator(dir))
    if (e.path().extension() == ".safetensors") shards.push_back(e.path().string());
  if (shards.empty()) throw std::runtime_error("load_dflash_weights: no shards in " + dir);
  std::sort(shards.begin(), shards.end());
  std::vector<std::unique_ptr<SafetensorsFile>> files;
  for (const auto& s : shards) files.push_back(SafetensorsFile::open(s));
  auto find = [&](const std::string& name) -> const TensorInfo* {
    for (const auto& f : files)
      if (const TensorInfo* t = f->find(name)) return t;
    return nullptr;
  };
  auto need = [&](const std::string& name) -> const TensorInfo* {
    if (const TensorInfo* t = find(name)) return t;
    throw std::runtime_error("load_dflash_weights: missing tensor '" + name + "'");
  };
  auto check = [&](const TensorInfo* t, DType dtype, std::vector<int64_t> shape) {
    if (t->dtype != dtype || t->shape != shape)
      throw std::runtime_error("load_dflash_weights: tensor '" + t->name + "' shape/dtype mismatch");
  };

  const int64_t H = cfg.hidden_size;
  const int64_t QW = cfg.local_q_row(world), KW = cfg.local_kv_row(world);
  const int64_t I = cfg.local_intermediate(world);
  const int64_t r = rank;
  const int64_t nTaps = static_cast<int64_t>(cfg.target_layer_ids.size());

  DFlashWeights w;
  DGPP_CUDA_OK(cudaMalloc(&w.arena, dflash_weights_bytes(cfg, world)));
  w.bytes = dflash_weights_bytes(cfg, world);
  uint16_t* base = static_cast<uint16_t*>(w.arena);
  size_t off = 0;
  auto bump = [&](size_t n) {
    uint16_t* p = base + off;
    off += n;
    return p;
  };
  auto copy_full = [&](const std::string& name, uint16_t* dst, std::vector<int64_t> shape) {
    const TensorInfo* t = need(name);
    check(t, DType::BF16, shape);
    DGPP_CUDA_OK(cudaMemcpyAsync(dst, t->data, t->nbytes(), cudaMemcpyHostToDevice, stream));
  };
  auto copy_rows = [&](const std::string& name, uint16_t* dst, int64_t rows, int64_t cols,
                       int64_t row0, int64_t count) {
    const TensorInfo* t = need(name);
    check(t, DType::BF16, {rows, cols});
    const auto* src = static_cast<const uint16_t*>(t->data) + static_cast<size_t>(row0) * cols;
    DGPP_CUDA_OK(cudaMemcpyAsync(dst, src, static_cast<size_t>(count) * cols * 2,
                                 cudaMemcpyHostToDevice, stream));
  };
  std::vector<uint16_t> colbuf;
  auto copy_cols = [&](const std::string& name, uint16_t* dst, int64_t rows, int64_t cols,
                       int64_t col0, int64_t count) {
    const TensorInfo* t = need(name);
    check(t, DType::BF16, {rows, cols});
    colbuf.resize(static_cast<size_t>(rows) * count);
    const auto* src = static_cast<const uint16_t*>(t->data);
    for (int64_t i = 0; i < rows; ++i)
      std::memcpy(colbuf.data() + static_cast<size_t>(i) * count,
                  src + static_cast<size_t>(i) * cols + col0, static_cast<size_t>(count) * 2);
    DGPP_CUDA_OK(cudaMemcpyAsync(dst, colbuf.data(), colbuf.size() * 2, cudaMemcpyHostToDevice,
                                 stream));
  };

  w.layers.resize(static_cast<size_t>(cfg.num_hidden_layers));
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    DFlashLayerWeights& x = w.layers[static_cast<size_t>(l)];
    x.input_norm = bump(H);
    copy_full(p + "input_layernorm.weight", const_cast<uint16_t*>(x.input_norm), {H});
    x.post_norm = bump(H);
    copy_full(p + "post_attention_layernorm.weight", const_cast<uint16_t*>(x.post_norm), {H});
    x.q = bump(static_cast<size_t>(QW) * H);
    copy_rows(p + "self_attn.q_proj.weight", const_cast<uint16_t*>(x.q), cfg.q_row(), H,
              r * QW, QW);
    x.k = bump(static_cast<size_t>(KW) * H);
    copy_rows(p + "self_attn.k_proj.weight", const_cast<uint16_t*>(x.k), cfg.kv_row(), H,
              r * KW, KW);
    x.v = bump(static_cast<size_t>(KW) * H);
    copy_rows(p + "self_attn.v_proj.weight", const_cast<uint16_t*>(x.v), cfg.kv_row(), H,
              r * KW, KW);
    x.o = bump(static_cast<size_t>(H) * QW);
    copy_cols(p + "self_attn.o_proj.weight", const_cast<uint16_t*>(x.o), H, cfg.q_row(),
              r * QW, QW);
    x.gate = bump(static_cast<size_t>(I) * H);
    copy_rows(p + "mlp.gate_proj.weight", const_cast<uint16_t*>(x.gate), cfg.intermediate_size,
              H, r * I, I);
    x.up = bump(static_cast<size_t>(I) * H);
    copy_rows(p + "mlp.up_proj.weight", const_cast<uint16_t*>(x.up), cfg.intermediate_size, H,
              r * I, I);
    x.down = bump(static_cast<size_t>(H) * I);
    copy_cols(p + "mlp.down_proj.weight", const_cast<uint16_t*>(x.down), H,
              cfg.intermediate_size, r * I, I);
    x.q_norm = bump(cfg.head_dim);
    copy_full(p + "self_attn.q_norm.weight", const_cast<uint16_t*>(x.q_norm), {cfg.head_dim});
    x.k_norm = bump(cfg.head_dim);
    copy_full(p + "self_attn.k_norm.weight", const_cast<uint16_t*>(x.k_norm), {cfg.head_dim});
  }
  w.fc = bump(static_cast<size_t>(nTaps) * H * H);
  {
    const TensorInfo* t = need("fc.weight");
    check(t, DType::BF16, {H, nTaps * H});
    colbuf.resize(static_cast<size_t>(nTaps) * H * H);
    const auto* src = static_cast<const uint16_t*>(t->data);
    for (int64_t i = 0; i < H; ++i)
      for (int64_t s = 0; s < nTaps; ++s)
        std::memcpy(colbuf.data() + (static_cast<size_t>(s) * H + i) * H,
                    src + static_cast<size_t>(i) * nTaps * H + static_cast<size_t>(s) * H,
                    static_cast<size_t>(H) * 2);
    DGPP_CUDA_OK(cudaMemcpyAsync(const_cast<uint16_t*>(w.fc), colbuf.data(), colbuf.size() * 2,
                                 cudaMemcpyHostToDevice, stream));
  }
  w.hidden_norm = bump(H);
  copy_full("hidden_norm.weight", const_cast<uint16_t*>(w.hidden_norm), {H});
  w.norm = bump(H);
  copy_full("norm.weight", const_cast<uint16_t*>(w.norm), {H});
  if (off * 2 != w.bytes)
    throw std::logic_error("load_dflash_weights: arena accounting mismatch");
  return w;
}

DFlashWeights::~DFlashWeights() {
  cudaFree(arena);
  arena = nullptr;
  bytes = 0;
}

}  // namespace dgpp
