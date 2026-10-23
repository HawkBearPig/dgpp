// The DFlash2 drafter: config parse and the standalone weight load. The
// tensors the drafter needs from its checkpoint (verified against
// z-lab/Qwen3.8-27B-DFlash2): per-layer q/k/v/o projections + q/k norms +
// two norms + SwiGLU + the attention_conv / mlp_conv dynamic-conv pairs,
// and the globals fc, hidden_norm, norm and the candidate_selector
// (codebooks + hidden projection). No embeddings and no lm head: the
// draft shares the target's.
#include "models/dflash2/dflash2.hpp"

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
#include "loaders/fp8_quant.hpp"
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

std::string drafter_architecture(const std::string& dir) {
  const std::string path = dir + "/config.json";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::invalid_argument("DFlash2: cannot open the drafter config " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  const std::string text = ss.str();  // values view the input: it must outlive the parse
  minijson::Value root;
  try {
    root = minijson::parse(text).root;
  } catch (const std::exception&) {
    throw std::invalid_argument("DFlash2: cannot parse the drafter config " + path);
  }
  if (root.find("architectures") && root.at("architectures").is_array() &&
      !root.at("architectures").items().empty())
    return std::string(root.at("architectures").items().front().as_string());
  return "";
}

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

void DFlash2Config::validate_against(const DFlash2TargetShape& target) const {
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
  if (max_position_embeddings > target.context_limit)
    reject("max_position_embeddings", "the draft must not exceed the target's");
}

DFlash2Weights::~DFlash2Weights() {
  cudaFree(arena);
  cudaFree(fp8_arena);
}

// The fp8 region (engine.dflash_weights fp8): per layer the five block
// matrices' E4M3 payloads and fp32 block-128 scales, 256-byte aligned.
namespace {
struct Fp8Off {
  size_t qkv_p, qkv_s, o_p, o_s, gate_p, gate_s, up_p, up_s, down_p, down_s;
};
size_t fp8_layout(const DFlash2Config& cfg, int world, std::vector<Fp8Off>* offs) {
  const size_t H = cfg.hidden_size, I = cfg.local_intermediate(world);
  const size_t QW = cfg.local_q_row(world), KW = cfg.local_kv_row(world);
  size_t total = 0;
  const auto bump = [&](size_t bytes) {
    const size_t o = total;
    total = (total + bytes + 255) & ~size_t{255};
    return o;
  };
  const auto mat = [&](size_t rows, size_t cols, size_t* p, size_t* s) {
    *p = bump(rows * cols);
    *s = bump(static_cast<size_t>(fp8_quant::scale_rows(rows)) * fp8_quant::scale_cols(cols) * 4);
  };
  if (offs) offs->resize(cfg.num_hidden_layers);
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    Fp8Off x{};
    mat(QW + 2 * KW, H, &x.qkv_p, &x.qkv_s);
    mat(H, QW, &x.o_p, &x.o_s);
    mat(I, H, &x.gate_p, &x.gate_s);
    mat(I, H, &x.up_p, &x.up_s);
    mat(H, I, &x.down_p, &x.down_s);
    if (offs) (*offs)[l] = x;
  }
  return total;
}
}  // namespace

size_t dflash2_weights_bytes(const DFlash2Config& cfg, int world, bool fp8) {
  // fp8: the five block matrices live in the fp8 arena only (no bf16 copy).
  // Element offsets with 128-element (256-byte) alignment — the same bump
  // the loader uses, so this is exactly the arena's byte size.
  if (!cfg.tp_divisible(world)) throw std::invalid_argument("dflash2_weights_bytes: the drafter does not split across this world");
  const size_t H = cfg.hidden_size, I = cfg.local_intermediate(world), HD = cfg.head_dim;
  const size_t QW = cfg.local_q_row(world), KW = cfg.local_kv_row(world), G = cfg.conv_groups(), T = cfg.conv_taps;
  size_t total = 0;
  const auto bump = [&](size_t elems) {
    const size_t o = total;
    total = (total + elems + 127) & ~size_t{127};
    return o;
  };
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    bump(H); bump(H);
    if (!fp8) {
      bump(static_cast<size_t>(QW + 2 * KW) * H);
      bump(H * QW);
    } else {
      bump(static_cast<size_t>(2 * KW) * H);  // the k|v rows alone
    }
    bump(HD); bump(HD);
    if (!fp8) {
      bump(I * H); bump(I * H); bump(H * I);
    }
    bump(2 * T * H); bump(2 * T * G * H); bump(2 * T * H); bump(2 * T * G * H);
  }
  bump(static_cast<size_t>(cfg.target_layer_ids.size()) * H * H);
  bump(H); bump(H);
  bump(static_cast<size_t>(cfg.vocab_size) * cfg.selector_rank);
  bump(static_cast<size_t>(cfg.vocab_size) * cfg.selector_rank);
  bump(static_cast<size_t>(cfg.selector_rank) * H);
  return total * 2 + (fp8 ? fp8_layout(cfg, world, nullptr) : 0);
}

DFlash2Weights load_dflash2_weights(const DFlash2Config& cfg, const std::string& dir,
                                    cudaStream_t stream, int rank, int world, bool fp8) {
  if (!cfg.tp_divisible(world) || rank < 0 || rank >= world)
    throw std::invalid_argument("DFlash2: the drafter's heads / kv heads / MLP rows must divide across the world");
  std::vector<std::string> shards;
  for (const auto& e : fs::directory_iterator(dir))
    if (e.path().extension() == ".safetensors") shards.push_back(e.path().string());
  std::sort(shards.begin(), shards.end());
  if (shards.empty()) throw std::runtime_error("DFlash2: no .safetensors shards in " + dir);
  std::vector<std::unique_ptr<SafetensorsFile>> files;
  for (const auto& s : shards) files.push_back(SafetensorsFile::open(s));

  // The checkpoint's widths (the shapes checked) and this rank's slices.
  const int H = cfg.hidden_size, HD = cfg.head_dim;
  const int gI = cfg.intermediate_size, gQW = cfg.q_row(), gKW = cfg.kv_row();
  const int I = cfg.local_intermediate(world), QW = cfg.local_q_row(world), KW = cfg.local_kv_row(world);
  const int G = cfg.conv_groups(), T = cfg.conv_taps;
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
      throw std::runtime_error("DFlash2 candidate_selector hidden_projection shape");
  }

  // The arena layout: element offsets (uint16_t) with 128-element
  // (256-byte) alignment — the same bump dflash2_weights_bytes uses.
  struct Off {
    size_t input_norm, post_norm, qkv, o, qn, kn, gate, up, down;
    size_t kv;  // fp8: the k|v rows' own bf16 slot
    size_t acb, ack, mcb, mck;
  };
  std::vector<Off> off(L);
  constexpr size_t kNoBf16 = ~size_t{0};  // fp8: the block matrices take no bf16 slot
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
    x.qkv = fp8 ? kNoBf16 : bump(static_cast<size_t>(QW + 2 * KW) * H);  // q, k, v rows contiguous
    x.o = fp8 ? kNoBf16 : bump(static_cast<size_t>(H) * QW);
    x.kv = fp8 ? bump(static_cast<size_t>(2 * KW) * H) : kNoBf16;
    x.qn = bump(HD);
    x.kn = bump(HD);
    x.gate = fp8 ? kNoBf16 : bump(static_cast<size_t>(I) * H);
    x.up = fp8 ? kNoBf16 : bump(static_cast<size_t>(I) * H);
    x.down = fp8 ? kNoBf16 : bump(static_cast<size_t>(H) * I);
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
  std::vector<Fp8Off> f8;
  if (fp8) {
    w.fp8_bytes = fp8_layout(cfg, world, &f8);
    DGPP_CUDA_OK(cudaMalloc(&w.fp8_arena, w.fp8_bytes));
  }
  // The fp8 encode: block-128 E4M3 of a host slice (row-major [rows, cols]
  // at `stride`), uploaded to the fp8 arena at the matrix's offsets.
  std::vector<uint8_t> pay;
  std::vector<float> scl;
  const auto encode_to = [&](const uint16_t* src, size_t stride, int64_t rows, int64_t cols, size_t p_off,
                             size_t s_off) {
    pay.resize(static_cast<size_t>(rows) * cols);
    scl.resize(static_cast<size_t>(fp8_quant::scale_rows(rows)) * fp8_quant::scale_cols(cols));
    fp8_quant::encode_block128(src, stride, rows, cols, pay.data(), scl.data());
    DGPP_CUDA_OK(cudaMemcpy(static_cast<uint8_t*>(w.fp8_arena) + p_off, pay.data(), pay.size(), cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(static_cast<uint8_t*>(w.fp8_arena) + s_off, scl.data(), scl.size() * 4, cudaMemcpyHostToDevice));
  };
  auto* base = static_cast<uint16_t*>(w.arena);
  const auto dev = [&](size_t off_elems) { return base + off_elems; };

  auto copy = [&](const char* name, size_t off_elems, const std::vector<int64_t>& shape) {
    upload_bf16(*file_for(name), name, shape, dev(off_elems), stream);
  };
  // A row range of a [rows, cols] tensor (contiguous bytes): this rank's
  // heads or MLP rows.
  auto copy_rows = [&](const char* name, size_t off_elems, int64_t rows, int64_t cols, int64_t row0,
                       int64_t count) {
    const TensorInfo* t = need(name);
    if (t->shape != std::vector<int64_t>{rows, cols})
      throw std::runtime_error(std::string("DFlash2 tensor ") + name + ": unexpected shape");
    const auto* src = static_cast<const uint16_t*>(t->data) + static_cast<size_t>(row0) * cols;
    DGPP_CUDA_OK(cudaMemcpyAsync(dev(off_elems), src, static_cast<size_t>(count) * cols * 2,
                                 cudaMemcpyHostToDevice, stream));
  };
  // A column range of a [rows, cols] tensor (this rank's input slice of o /
  // down): gathered on the host, then one upload.
  std::vector<uint16_t> colbuf;
  auto copy_cols = [&](const char* name, size_t off_elems, int64_t rows, int64_t cols, int64_t col0,
                       int64_t count) {
    const TensorInfo* t = need(name);
    if (t->shape != std::vector<int64_t>{rows, cols})
      throw std::runtime_error(std::string("DFlash2 tensor ") + name + ": unexpected shape");
    colbuf.resize(static_cast<size_t>(rows) * count);
    const auto* src = static_cast<const uint16_t*>(t->data);
    for (int64_t r = 0; r < rows; ++r)
      std::memcpy(colbuf.data() + static_cast<size_t>(r) * count, src + static_cast<size_t>(r) * cols + col0,
                  static_cast<size_t>(count) * 2);
    DGPP_CUDA_OK(cudaMemcpy(dev(off_elems), colbuf.data(), colbuf.size() * 2, cudaMemcpyHostToDevice));
  };
  const int64_t q0 = static_cast<int64_t>(rank) * QW, kv0 = static_cast<int64_t>(rank) * KW;
  const int64_t i0 = static_cast<int64_t>(rank) * I;
  for (int l = 0; l < L; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    const Off& x = off[l];
    copy((p + "input_layernorm.weight").c_str(), x.input_norm, {H});
    copy((p + "post_attention_layernorm.weight").c_str(), x.post_norm, {H});
    if (!fp8) {
      copy_rows((p + "self_attn.q_proj.weight").c_str(), x.qkv, gQW, H, q0, QW);
      copy_rows((p + "self_attn.k_proj.weight").c_str(), x.qkv + static_cast<size_t>(QW) * H, gKW, H, kv0, KW);
      copy_rows((p + "self_attn.v_proj.weight").c_str(), x.qkv + static_cast<size_t>(QW + KW) * H, gKW, H, kv0,
                KW);
      copy_cols((p + "self_attn.o_proj.weight").c_str(), x.o, H, gQW, q0, QW);
    } else {
      copy_rows((p + "self_attn.k_proj.weight").c_str(), x.kv, gKW, H, kv0, KW);
      copy_rows((p + "self_attn.v_proj.weight").c_str(), x.kv + static_cast<size_t>(KW) * H, gKW, H, kv0, KW);
    }
    copy((p + "self_attn.q_norm.weight").c_str(), x.qn, {HD});
    copy((p + "self_attn.k_norm.weight").c_str(), x.kn, {HD});
    if (!fp8) {
      copy_rows((p + "mlp.gate_proj.weight").c_str(), x.gate, gI, H, i0, I);
      copy_rows((p + "mlp.up_proj.weight").c_str(), x.up, gI, H, i0, I);
      copy_cols((p + "mlp.down_proj.weight").c_str(), x.down, H, gI, i0, I);
    }
    if (fp8) {
      // The five block matrices: the row slices straight from the shards,
      // the column slices (o, down) from their gathered host copies, the
      // stacked q|k|v rows from a host stack.
      const Fp8Off& y = f8[l];
      const auto rows_of = [&](const char* name) {
        return static_cast<const uint16_t*>(need(name)->data);
      };
      std::vector<uint16_t> qkv(static_cast<size_t>(QW + 2 * KW) * H);
      std::memcpy(qkv.data(), rows_of((p + "self_attn.q_proj.weight").c_str()) + static_cast<size_t>(q0) * H,
                  static_cast<size_t>(QW) * H * 2);
      std::memcpy(qkv.data() + static_cast<size_t>(QW) * H,
                  rows_of((p + "self_attn.k_proj.weight").c_str()) + static_cast<size_t>(kv0) * H,
                  static_cast<size_t>(KW) * H * 2);
      std::memcpy(qkv.data() + static_cast<size_t>(QW + KW) * H,
                  rows_of((p + "self_attn.v_proj.weight").c_str()) + static_cast<size_t>(kv0) * H,
                  static_cast<size_t>(KW) * H * 2);
      encode_to(qkv.data(), static_cast<size_t>(H), QW + 2 * KW, H, y.qkv_p, y.qkv_s);
      // o: the column slice gathered again (colbuf holds down's by now).
      {
        const TensorInfo* t = need((p + "self_attn.o_proj.weight").c_str());
        std::vector<uint16_t> ob(static_cast<size_t>(H) * QW);
        const auto* src = static_cast<const uint16_t*>(t->data);
        for (int64_t r = 0; r < H; ++r)
          std::memcpy(ob.data() + static_cast<size_t>(r) * QW, src + static_cast<size_t>(r) * gQW + q0,
                      static_cast<size_t>(QW) * 2);
        encode_to(ob.data(), static_cast<size_t>(QW), H, QW, y.o_p, y.o_s);
      }
      encode_to(rows_of((p + "mlp.gate_proj.weight").c_str()) + static_cast<size_t>(i0) * H, static_cast<size_t>(H), I, H,
                y.gate_p, y.gate_s);
      encode_to(rows_of((p + "mlp.up_proj.weight").c_str()) + static_cast<size_t>(i0) * H, static_cast<size_t>(H), I, H,
                y.up_p, y.up_s);
      {
        // down: this rank's column slice, gathered on the host.
        const TensorInfo* t = need((p + "mlp.down_proj.weight").c_str());
        if (t->shape != std::vector<int64_t>{H, gI})
          throw std::runtime_error("DFlash2 tensor " + p + "mlp.down_proj.weight: unexpected shape");
        std::vector<uint16_t> db(static_cast<size_t>(H) * I);
        const auto* src = static_cast<const uint16_t*>(t->data);
        for (int64_t r = 0; r < H; ++r)
          std::memcpy(db.data() + static_cast<size_t>(r) * I, src + static_cast<size_t>(r) * gI + i0,
                      static_cast<size_t>(I) * 2);
        encode_to(db.data(), static_cast<size_t>(I), H, I, y.down_p, y.down_s);
      }
    }
    copy((p + "attention_conv.base_kernel").c_str(), x.acb, {2, T, H});
    copy((p + "attention_conv.kernel_projection.weight").c_str(), x.ack, {2ll * T * G, H});
    copy((p + "mlp_conv.base_kernel").c_str(), x.mcb, {2, T, H});
    copy((p + "mlp_conv.kernel_projection.weight").c_str(), x.mck, {2ll * T * G, H});
    DFlash2LayerWeights& lw = w.layers.emplace_back();
    lw.input_norm = dev(x.input_norm);
    lw.post_norm = dev(x.post_norm);
    lw.qkv = fp8 ? nullptr : dev(x.qkv);
    lw.kv_rows = fp8 ? dev(x.kv) : dev(x.qkv) + static_cast<size_t>(QW) * H;
    lw.o = fp8 ? nullptr : dev(x.o);
    lw.gate = fp8 ? nullptr : dev(x.gate);
    lw.up = fp8 ? nullptr : dev(x.up);
    lw.down = fp8 ? nullptr : dev(x.down);
    lw.q_norm = dev(x.qn);
    lw.k_norm = dev(x.kn);
    lw.attn_conv_base = dev(x.acb);
    lw.attn_conv_kp = dev(x.ack);
    lw.mlp_conv_base = dev(x.mcb);
    lw.mlp_conv_kp = dev(x.mck);
    if (fp8) {
      const Fp8Off& y = f8[static_cast<size_t>(l)];
      const auto* fb = static_cast<const uint8_t*>(w.fp8_arena);
      lw.qkv_fp8 = fb + y.qkv_p;   lw.qkv_scales = reinterpret_cast<const float*>(fb + y.qkv_s);
      lw.o_fp8 = fb + y.o_p;       lw.o_scales = reinterpret_cast<const float*>(fb + y.o_s);
      lw.gate_fp8 = fb + y.gate_p; lw.gate_scales = reinterpret_cast<const float*>(fb + y.gate_s);
      lw.up_fp8 = fb + y.up_p;     lw.up_scales = reinterpret_cast<const float*>(fb + y.up_s);
      lw.down_fp8 = fb + y.down_p; lw.down_scales = reinterpret_cast<const float*>(fb + y.down_s);
    }
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
