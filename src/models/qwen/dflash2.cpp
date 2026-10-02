// The DFlash2 drafter: config parse and the standalone weight load. The
// tensors the drafter needs from its checkpoint (verified against
// z-lab/Qwen3.8-27B-DFlash2): per-layer q/k/v/o projections + q/k norms +
// two norms + SwiGLU + the attention_conv / mlp_conv dynamic-conv pairs,
// and the globals fc, hidden_norm, norm and the candidate_selector
// (codebooks + hidden projection). No embeddings and no lm head: the
// draft shares the target's.
#include "models/qwen/dflash2.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "engine/decode_outputs.hpp"
#include "loaders/safetensors.hpp"

namespace dgpp {
namespace {

namespace fs = std::filesystem;

[[noreturn]] void reject(std::string_view field, std::string_view why) {
  throw std::runtime_error(std::format("DFlash2 config.{}: {}", field, why));
}

const minijson::Value& require(const minijson::Value& v, std::string_view field) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) reject(field, "missing");
  return *f;
}
int int_at(const minijson::Value& v, std::string_view field, int dflt, bool required = false) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) {
    if (required) reject(field, "missing");
    return dflt;
  }
  if (!f->is_number()) reject(field, "not a number");
  return static_cast<int>(f->as_int());
}
int64_t i64_at(const minijson::Value& v, std::string_view field, int64_t dflt, bool required = false) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) {
    if (required) reject(field, "missing");
    return dflt;
  }
  if (!f->is_number()) reject(field, "not a number");
  return f->as_int();
}
double double_at(const minijson::Value& v, std::string_view field, double dflt, bool required = false) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) {
    if (required) reject(field, "missing");
    return dflt;
  }
  if (!f->is_number() || !std::isfinite(f->as_double())) reject(field, "not a finite number");
  return f->as_double();
}
std::string str_at(const minijson::Value& v, std::string_view field, const std::string& dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_string()) reject(field, "not a string");
  return std::string(f->as_string());
}

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// ---- the weight load ----------------------------------------------------------

struct Plan {
  void* arena = nullptr;
  size_t cursor = 0;
  size_t reserve_bytes = 0;
};

// One bf16 tensor: existence, dtype and shape checked; copied H2D verbatim.
void upload_bf16(const SafetensorsFile& f, const std::string& name,
                 const std::vector<int64_t>& shape, void* dst, cudaStream_t stream) {
  const TensorInfo* t = f.find(name);
  if (!t) throw std::runtime_error("DFlash2 checkpoint is missing " + name);
  if (t->dtype != DType::BF16)
    throw std::runtime_error("DFlash2 tensor " + name + ": BF16 required");
  if (t->shape != shape) {
    std::string got;
    for (auto d : t->shape) got += std::to_string(d) + "x";
    throw std::runtime_error("DFlash2 tensor " + name + ": unexpected shape " + got);
  }
  DGPP_CUDA_OK(cudaMemcpyAsync(dst, t->data, t->nbytes(), cudaMemcpyHostToDevice, stream));
}

}  // namespace

DFlash2Config DFlash2Config::from_json_file(const std::string& path) {
  const std::string text = read_file(path);
  minijson::Value root;
  try {
    root = minijson::parse(text).root;
  } catch (const std::exception& e) {
    throw std::runtime_error("DFlash2 config " + path + ": " + e.what());
  }
  return parse(root);
}

DFlash2Config DFlash2Config::parse(const minijson::Value& root) {
  DFlash2Config c;
  const std::string arch =
      root.find("architectures") && root.at("architectures").is_array() &&
              !root.at("architectures").items().empty()
          ? std::string(root.at("architectures").items().front().as_string())
          : "";
  if (arch != "DFlash2DraftModel")
    reject("architectures",
           arch == "DFlashDraftModel"
               ? "DFlash v1 (no convolutions / no selector) is not supported; DFlash2 required"
               : "DFlash2DraftModel required");
  if (str_at(root, "model_type", "") != "qwen3") reject("model_type", "qwen3 required");
  if (str_at(root, "dtype", "bfloat16") != "bfloat16") reject("dtype", "bfloat16 required");
  const minijson::Value& d = require(root, "dflash_config");
  if (d.find("sample_from_anchor") && d.at("sample_from_anchor").as_bool())
    reject("dflash_config.sample_from_anchor", "the anchor-as-bonus layout is required");
  if (double_at(d, "input_embedding_scale", 1.0) != 1.0)
    reject("dflash_config.input_embedding_scale", "only 1.0 is supported");
  for (const char* k : {"final_logit_softcapping", "output_multiplier"})
    if (const minijson::Value* s = d.find(k))
      if (s->as_double(0.0) != 0.0 && s->as_double(1.0) != 1.0)
        reject(std::string("dflash_config.") + k, "only 1.0 / unset is supported");
  if (d.find("use_aux_hidden_state") && !d.at("use_aux_hidden_state").as_bool(true))
    reject("dflash_config.use_aux_hidden_state", "the fused feature input is required (fc)");
  if (d.find("causal") && d.at("causal").as_bool(false))
    reject("dflash_config.causal", "causal drafter layers are not supported");
  if (root.find("is_causal") && root.at("is_causal").as_bool(false))
    reject("is_causal", "causal drafter layers are not supported");
  c.block_size = int_at(d, "block_size", 8, true);
  c.mask_token_id = int_at(d, "mask_token_id", -1, true);
  c.conv_taps = int_at(d, "conv_kernel_size", 2);
  c.conv_group_size = int_at(d, "conv_group_size", 16);
  c.selector_rank = int_at(d, "selector_rank", 256, true);
  c.selector_top_k = int_at(d, "selector_top_k", 16, true);
  const minijson::Value& taps = require(d, "target_layer_ids");
  if (!taps.is_array() || taps.items().empty()) reject("dflash_config.target_layer_ids", "empty");
  for (const auto& t : taps.items()) {
    if (!t.is_number()) reject("dflash_config.target_layer_ids", "non-numeric element");
    c.target_layer_ids.push_back(static_cast<int>(t.as_int()));
  }
  std::sort(c.target_layer_ids.begin(), c.target_layer_ids.end());
  c.num_hidden_layers = int_at(root, "num_hidden_layers", 5, true);
  c.hidden_size = int_at(root, "hidden_size", 5120, true);
  c.vocab_size = int_at(root, "vocab_size", 248320, true);
  c.num_attention_heads = int_at(root, "num_attention_heads", 32, true);
  c.num_key_value_heads = int_at(root, "num_key_value_heads", 8, true);
  c.head_dim = int_at(root, "head_dim", c.hidden_size / c.num_attention_heads, true);
  c.intermediate_size = int_at(root, "intermediate_size", 0, true);
  c.rms_norm_eps = static_cast<float>(double_at(root, "rms_norm_eps", 1e-6));
  const minijson::Value& rp = require(root, "rope_parameters");
  if (str_at(rp, "rope_type", "default") != "default")
    reject("rope_parameters.rope_type", "only the plain default scaling is supported");
  c.rope_theta = double_at(rp, "rope_theta", 1e7);
  c.max_position_embeddings = i64_at(root, "max_position_embeddings", 262144);
  {
    const minijson::Value* sw = root.find("use_sliding_window");
    const bool sliding = sw && !sw->is_null() ? sw->as_bool(true) : true;
    c.sliding_window = sliding ? i64_at(root, "sliding_window", 2048, true) : INT64_MAX;
  }
  if (c.block_size < 2) reject("dflash_config.block_size", "at least 2");
  if (c.num_hidden_layers < 1 || c.num_hidden_layers > 16)
    reject("num_hidden_layers", "1..16 supported");
  if (c.conv_taps != 2) reject("dflash_config.conv_kernel_size", "only the 2-tap conv is supported");
  if (c.selector_top_k != 16) reject("dflash_config.selector_top_k", "only 16 is supported");
  if (c.head_dim != 128) reject("head_dim", "only 128 is supported");
  if (c.hidden_size % c.conv_group_size)
    reject("dflash_config.conv_group_size", "must divide hidden_size");
  if (c.num_attention_heads % c.num_key_value_heads ||
      c.num_attention_heads / c.num_key_value_heads > 8)
    reject("num_key_value_heads", "heads/kv_heads must be in [1, 8]");
  if (const minijson::Value* tw = root.find("tie_word_embeddings"))
    if (tw->as_bool(false))
      reject("tie_word_embeddings", "the draft shares the target's head; ties are not loaded");
  return c;
}

void DFlash2Config::validate_against(const Qwen35TextConfig& target) const {
  if (hidden_size != target.hidden_size)
    reject("hidden_size", "the draft must share the target's width (fc and the shared head)");
  if (vocab_size != target.vocab_size)
    reject("vocab_size", "the draft shares the target's embeddings and lm head");
  if (mask_token_id < 0 || mask_token_id >= target.vocab_size)
    reject("dflash_config.mask_token_id", "outside the target vocabulary");
  if (drafts() > kSpecRows - 1)
    reject("dflash_config.block_size", "the draft depth exceeds the speculative row budget");
  for (size_t i = 0; i < target_layer_ids.size(); ++i) {
    if (target_layer_ids[i] < 0 || target_layer_ids[i] >= target.num_hidden_layers)
      reject("dflash_config.target_layer_ids",
             "layer " + std::to_string(target_layer_ids[i]) + " does not exist in the target");
    if (i && target_layer_ids[i] == target_layer_ids[i - 1])
      reject("dflash_config.target_layer_ids", "duplicate tap layer");
  }
  if (sliding_window <= block_size)
    reject("sliding_window", "the window must cover the whole query block");
  if (max_position_embeddings > target.max_position_embeddings)
    reject("max_position_embeddings", "the draft must not exceed the target's");
}

DFlash2Weights::~DFlash2Weights() { cudaFree(arena); }

size_t dflash2_weights_bytes(const DFlash2Config& cfg) {
  // Element offsets with 128-element (256-byte) alignment — the same bump
  // the loader uses, so this is exactly the arena's byte size.
  const size_t H = cfg.hidden_size, I = cfg.intermediate_size, HD = cfg.head_dim;
  const size_t QW = cfg.q_row(), KW = cfg.kv_row(), G = cfg.conv_groups(), T = cfg.conv_taps;
  size_t total = 0;
  const auto bump = [&](size_t elems) {
    const size_t o = total;
    total = (total + elems + 127) & ~size_t{127};
    return o;
  };
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    bump(H); bump(H);
    bump(static_cast<size_t>(QW + 2 * KW) * H);
    bump(H * QW); bump(HD); bump(HD);
    bump(I * H); bump(I * H); bump(H * I);
    bump(2 * T * H); bump(2 * T * G * H); bump(2 * T * H); bump(2 * T * G * H);
  }
  bump(static_cast<size_t>(cfg.target_layer_ids.size()) * H * H);
  bump(H); bump(H);
  bump(static_cast<size_t>(cfg.vocab_size) * cfg.selector_rank);
  bump(static_cast<size_t>(cfg.vocab_size) * cfg.selector_rank);
  bump(static_cast<size_t>(cfg.selector_rank) * H);
  return total * 2;
}

DFlash2Weights load_dflash2_weights(const DFlash2Config& cfg, const std::string& dir,
                                    cudaStream_t stream) {
  std::vector<std::string> shards;
  for (const auto& e : fs::directory_iterator(dir))
    if (e.path().extension() == ".safetensors") shards.push_back(e.path().string());
  std::sort(shards.begin(), shards.end());
  if (shards.empty()) throw std::runtime_error("DFlash2: no .safetensors shards in " + dir);
  std::vector<std::unique_ptr<SafetensorsFile>> files;
  for (const auto& s : shards) files.push_back(SafetensorsFile::open(s));

  const int H = cfg.hidden_size, I = cfg.intermediate_size, HD = cfg.head_dim;
  const int QW = cfg.q_row(), KW = cfg.kv_row(), G = cfg.conv_groups(), T = cfg.conv_taps;
  const int64_t V = cfg.vocab_size;
  const int nTaps = static_cast<int>(cfg.target_layer_ids.size());
  const int L = cfg.num_hidden_layers;

  // One arena, layer-major. Sizes in bf16 elements.
  auto file_for = [&](const std::string& name) -> const SafetensorsFile* {
    for (const auto& f : files)
      if (f->find(name)) return f.get();
    throw std::runtime_error("DFlash2 checkpoint is missing " + name);
  };
  auto need = [&](const std::string& name) {
    const SafetensorsFile* f = file_for(name);
    const TensorInfo* t = f->find(name);
    if (t->dtype != DType::BF16)
      throw std::runtime_error("DFlash2 tensor " + name + ": BF16 required");
    return t;
  };

  // Pre-flight: every required name and shape, so a wrong drafter fails
  // before the allocation. (The codebooks name the vocab; fc names the taps.)
  need("fc.weight");
  need("hidden_norm.weight");
  need("norm.weight");
  need("candidate_selector.predecessor_codebook");
  need("candidate_selector.successor_codebook");
  need("candidate_selector.hidden_projection.weight");
  for (int l = 0; l < L; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    for (const char* n : {"input_layernorm.weight", "post_attention_layernorm.weight",
                          "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                          "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                          "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                          "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
                          "attention_conv.base_kernel", "attention_conv.kernel_projection.weight",
                          "mlp_conv.base_kernel", "mlp_conv.kernel_projection.weight"})
      need(p + n);
  }
  {
    const TensorInfo* fcw = need("fc.weight");
    if (fcw->shape != std::vector<int64_t>{H, int64_t(nTaps) * H})
      throw std::runtime_error(
          "DFlash2 fc.weight: the target_layer_ids count disagrees with the checkpoint");
    if (need("candidate_selector.predecessor_codebook")->shape !=
            std::vector<int64_t>{V, cfg.selector_rank} ||
        need("candidate_selector.successor_codebook")->shape !=
            std::vector<int64_t>{V, cfg.selector_rank})
      throw std::runtime_error("DFlash2 candidate_selector codebook shape");
    if (need("candidate_selector.hidden_projection.weight")->shape !=
        std::vector<int64_t>{cfg.selector_rank, H})
      throw std::runtime_error("DFlash2 candidate_selector.hidden_projection shape");
  }

  // The arena layout: element offsets (uint16_t) with 128-element
  // (256-byte) alignment — the same bump dflash2_weights_bytes uses.
  struct Off {
    size_t input_norm, post_norm, qkv, o, qn, kn, gate, up, down;
    size_t acb, ack, mcb, mck;
  };
  std::vector<Off> off(L);
  size_t total = 0;
  const auto bump = [&](size_t elems) {
    const size_t o = total;
    total = (total + elems + 127) & ~size_t{127};
    return o;
  };
  for (int l = 0; l < L; ++l) {
    Off& x = off[l];
    x.input_norm = bump(H);
    x.post_norm = bump(H);
    x.qkv = bump(static_cast<size_t>(QW + 2 * KW) * H);  // q, k, v rows contiguous
    x.o = bump(static_cast<size_t>(H) * QW);
    x.qn = bump(HD);
    x.kn = bump(HD);
    x.gate = bump(static_cast<size_t>(I) * H);
    x.up = bump(static_cast<size_t>(I) * H);
    x.down = bump(static_cast<size_t>(H) * I);
    x.acb = bump(2ull * T * H);
    x.ack = bump(2ull * T * G * H);
    x.mcb = bump(2ull * T * H);
    x.mck = bump(2ull * T * G * H);
  }
  const size_t g_fc = bump(static_cast<size_t>(nTaps) * H * H);
  const size_t g_hidden_norm = bump(H);
  const size_t g_norm = bump(H);
  const size_t g_pred = bump(static_cast<size_t>(V) * cfg.selector_rank);
  const size_t g_succ = bump(static_cast<size_t>(V) * cfg.selector_rank);
  const size_t g_hp = bump(static_cast<size_t>(cfg.selector_rank) * H);

  DFlash2Weights w;
  DGPP_CUDA_OK(cudaMalloc(&w.arena, total * 2));
  w.bytes = total * 2;
  auto* base = static_cast<uint16_t*>(w.arena);
  const auto dev = [&](size_t off_elems) { return base + off_elems; };

  auto copy = [&](const char* name, size_t off_elems, const std::vector<int64_t>& shape) {
    upload_bf16(*file_for(name), name, shape, dev(off_elems), stream);
  };
  for (int l = 0; l < L; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    const Off& x = off[l];
    copy((p + "input_layernorm.weight").c_str(), x.input_norm, {H});
    copy((p + "post_attention_layernorm.weight").c_str(), x.post_norm, {H});
    copy((p + "self_attn.q_proj.weight").c_str(), x.qkv, {QW, H});
    copy((p + "self_attn.k_proj.weight").c_str(), x.qkv + static_cast<size_t>(QW) * H, {KW, H});
    copy((p + "self_attn.v_proj.weight").c_str(), x.qkv + static_cast<size_t>(QW + KW) * H,
         {KW, H});
    copy((p + "self_attn.o_proj.weight").c_str(), x.o, {H, QW});
    copy((p + "self_attn.q_norm.weight").c_str(), x.qn, {HD});
    copy((p + "self_attn.k_norm.weight").c_str(), x.kn, {HD});
    copy((p + "mlp.gate_proj.weight").c_str(), x.gate, {I, H});
    copy((p + "mlp.up_proj.weight").c_str(), x.up, {I, H});
    copy((p + "mlp.down_proj.weight").c_str(), x.down, {H, I});
    copy((p + "attention_conv.base_kernel").c_str(), x.acb, {2, T, H});
    copy((p + "attention_conv.kernel_projection.weight").c_str(), x.ack, {2ll * T * G, H});
    copy((p + "mlp_conv.base_kernel").c_str(), x.mcb, {2, T, H});
    copy((p + "mlp_conv.kernel_projection.weight").c_str(), x.mck, {2ll * T * G, H});
    DFlash2LayerWeights& lw = w.layers.emplace_back();
    lw.input_norm = dev(x.input_norm);
    lw.post_norm = dev(x.post_norm);
    lw.qkv = dev(x.qkv);
    lw.o = dev(x.o);
    lw.gate = dev(x.gate);
    lw.up = dev(x.up);
    lw.down = dev(x.down);
    lw.q_norm = dev(x.qn);
    lw.k_norm = dev(x.kn);
    lw.attn_conv_base = dev(x.acb);
    lw.attn_conv_kp = dev(x.ack);
    lw.mlp_conv_base = dev(x.mcb);
    lw.mlp_conv_kp = dev(x.mck);
  }
  // fc: the [H, nTaps*H] rows are split into nTaps [H, H] GEMM weights
  // (the [N, K] view of the columns fc[o, t*H:(t+1)*H]) — a host transpose
  // through one staging buffer, so each tap GEMM streams a contiguous
  // weight with the same bytes the reference's one wide GEMM reads.
  {
    const TensorInfo* t = need("fc.weight");
    std::vector<uint16_t> host(static_cast<size_t>(H) * nTaps * H);
    std::memcpy(host.data(), t->data, host.size() * 2);
    std::vector<uint16_t> slice(static_cast<size_t>(H) * H);
    for (int tt = 0; tt < nTaps; ++tt) {
      for (int o = 0; o < H; ++o)
        std::memcpy(slice.data() + static_cast<size_t>(o) * H,
                    host.data() + static_cast<size_t>(o) * nTaps * H + static_cast<size_t>(tt) * H,
                    static_cast<size_t>(H) * 2);
      DGPP_CUDA_OK(cudaMemcpy(dev(g_fc) + static_cast<size_t>(tt) * H * H, slice.data(),
                              slice.size() * 2, cudaMemcpyHostToDevice));
    }
  }
  copy("hidden_norm.weight", g_hidden_norm, {H});
  copy("norm.weight", g_norm, {H});
  copy("candidate_selector.predecessor_codebook", g_pred, {V, cfg.selector_rank});
  copy("candidate_selector.successor_codebook", g_succ, {V, cfg.selector_rank});
  copy("candidate_selector.hidden_projection.weight", g_hp, {cfg.selector_rank, H});

  w.fc = dev(g_fc);
  w.hidden_norm = dev(g_hidden_norm);
  w.norm = dev(g_norm);
  w.pred_codebook = dev(g_pred);
  w.succ_codebook = dev(g_succ);
  w.hidden_projection = dev(g_hp);

  // The shard mappings are no longer needed once every byte is copied.
  for (auto& f : files) f->close_mapping(/*drop_page_cache=*/true);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  return w;
}

}  // namespace dgpp
