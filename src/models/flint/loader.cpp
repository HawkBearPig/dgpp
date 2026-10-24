#include "models/flint/loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/log.hpp"

namespace dgpp {
namespace {

bool ends_with(const std::string& s, const char* suffix) {
  const size_t n = std::strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// The replicated set (rank-invariant reads at world > 1): the layer norms,
// the per-head q / k norms, the router, the embedding and the final norm.
// Everything else is a slice: the fused projection's head rows, o_proj's
// columns, the sink's heads, every MLP matrix, every routed expert, the lm
// head under VocabSharded.
bool is_replicated(const FlintExpectedTensor& e) {
  switch (e.cls) {
    case FlintWeightClass::Embed:
    case FlintWeightClass::FinalNorm:
    case FlintWeightClass::LayerNorm:
    case FlintWeightClass::Router:
      return true;
    case FlintWeightClass::Attention:
      return ends_with(e.name, "q_norm.weight") || ends_with(e.name, "k_norm.weight");
    case FlintWeightClass::LmHead:
    case FlintWeightClass::SharedExpert:
    case FlintWeightClass::DenseMlp:
    case FlintWeightClass::RoutedExpert:
      return false;
  }
  return false;
}

}  // namespace

// The per-class builders (loaders/weight_build.hpp's primitives).
struct FlintLoaderFamily::Builder : WeightBuilder<FlintExpectedTensor> {
  const FlintTextConfig& cfg;
  const FlintLocalGeometry& geo;
  FlintLayerResident& out;

  Builder(const FlintTextConfig& cfg_, const FlintLocalGeometry& geo_, const std::vector<FlintExpectedTensor>& table_,
          const std::unordered_map<std::string, const FlintExpectedTensor*>& by_name_, LayerBump& bump_,
          FlintLayerResident& out_, const std::unordered_map<std::string, const TensorInfo*>& tensors_,
          std::vector<DequantJob>& jobs_, std::vector<PackJob>& packs_, bool copy_)
      : WeightBuilder<FlintExpectedTensor>(table_, by_name_, bump_, tensors_, jobs_, packs_, copy_, geo_.rank,
                                           geo_.world, "flint loader"),
        cfg(cfg_), geo(geo_), out(out_) {}

  bool replicated(const FlintExpectedTensor& e) const override { return is_replicated(e); }
  bool fp4_global(const FlintExpectedTensor& e) const override { return e.role == FlintTensorRole::Fp4Global; }

  // ---- the fused [Q | K | V] projection ----------------------------------------
  // One packable bf16 grant holding the rank's q rows, then its k rows,
  // then its v rows, each a contiguous row range of its source matrix.
  const uint16_t* load_qkv(const std::string& p) {
    const int64_t H = cfg.hidden_size, D = cfg.head_dim;
    const int64_t q_rows = static_cast<int64_t>(geo.local_heads) * D;
    const int64_t kv_rows = static_cast<int64_t>(geo.local_kv_heads) * D;
    const int64_t rows = q_rows + 2 * kv_rows;
    uint16_t* dst = static_cast<uint16_t*>(grant_bf16(rows, H, /*packable=*/true));
    struct Part {
      const char* name;
      int64_t row0, n;
    };
    const Part parts[3] = {{"q_proj.weight", static_cast<int64_t>(geo.head_begin) * D, q_rows},
                           {"k_proj.weight", static_cast<int64_t>(geo.kv_head_begin) * D, kv_rows},
                           {"v_proj.weight", static_cast<int64_t>(geo.kv_head_begin) * D, kv_rows}};
    int64_t at = 0;
    for (const Part& part : parts) {
      const FlintExpectedTensor& e = expected(p + part.name);
      check_range(e.name, part.row0, part.n, e.shape[0]);
      if (e.shape[1] != H) fail("'" + e.name + "' is not [rows, hidden]");
      if (copy) {
        const TensorInfo& t = source(e.name);
        std::memcpy(bump.host(dst) + static_cast<size_t>(at) * H,
                    static_cast<const uint8_t*>(t.data) + static_cast<size_t>(part.row0) * H * 2,
                    static_cast<size_t>(part.n) * H * 2);
        consumed(t);
      }
      note_read(e, static_cast<size_t>(part.n) * H * 2);
      at += part.n;
    }
    return dst;
  }

  // ---- the NVFP4 experts: the checkpoint's reciprocal global as the kernels' divisor
  float* load_global(const std::string& base) {
    float* slot = static_cast<float*>(bump.alloc(4));
    load_fp4_global(base, slot);
    return slot;
  }

  // ---- the classes ---------------------------------------------------------
  void build_attention(const std::string& p, int layer) {
    FlintAttnResident& a = out.attn;
    a.swa = cfg.is_swa_layer(layer);
    a.rope = cfg.rope_of(layer);
    a.local_heads = geo.local_heads;
    a.head_begin = geo.head_begin;
    a.local_kv_heads = geo.local_kv_heads;
    a.kv_head_begin = geo.kv_head_begin;
    a.qkv = load_qkv(p);
    const int64_t o0 = static_cast<int64_t>(geo.head_begin) * cfg.head_dim;
    const int64_t on = static_cast<int64_t>(geo.local_heads) * cfg.head_dim;
    a.o_proj = load_bf16_cols(p + "o_proj.weight", o0, on, /*packable=*/true);
    a.q_norm = load_bf16(p + "q_norm.weight");
    a.k_norm = load_bf16(p + "k_norm.weight");
    a.sink = cfg.sink ? load_bf16_as_f32(p + "softmax_offset", geo.head_begin, geo.local_heads) : nullptr;
  }

  // packable: the dense layers' matrices run through the GEMM interface
  // and take 12-bit companions (engine.bf16_weights); the shared expert's
  // are read raw by the MoE tail's own decode kernels (kernels/qwen_moe).
  void build_bf16_mlp(const std::string& p, int64_t I, FlintBf16MlpResident& m, bool packable) {
    const int64_t r = rank;
    m.local_inter = I;
    m.gate = load_bf16_rows(p + "gate_proj.weight", r * I, I, packable);
    m.up = load_bf16_rows(p + "up_proj.weight", r * I, I, packable);
    m.down = load_bf16_cols(p + "down_proj.weight", r * I, I, packable);
  }

  void build_moe(const std::string& p, int layer) {
    FlintMoeResident& m = out.moe_w;
    m.router = load_bf16(p + "gate.weight");
    // The zero selection bias (docs/flint_plan.md D7): the router kernels'
    // SigmoidBias mode adds it to the scores; zero keeps the key the score.
    {
      const size_t bytes = static_cast<size_t>(cfg.n_routed_experts) * 4;
      float* bias = static_cast<float*>(bump.alloc(bytes));
      if (copy) std::memset(bump.host(bias), 0, bytes);
      m.router_bias = bias;
    }
    build_bf16_mlp(p + "shared_expert.", geo.local_shared_inter, m.shared, /*packable=*/false);
    const int64_t I = geo.local_inter, r = rank;
    m.local_inter = I;
    m.fp4_mask = cfg.expert_fp4_mask[static_cast<size_t>(layer)];
    const int E = cfg.n_routed_experts;
    m.experts_fp8.assign(static_cast<size_t>(E) * 3, GlmQuantMatrix{});
    m.experts_fp4.assign(static_cast<size_t>(E) * 3, GlmFp4Matrix{});
    const char* names[3] = {"gate_proj.weight", "up_proj.weight", "down_proj.weight"};
    for (int e = 0; e < E; ++e) {
      const std::string ep = p + "experts." + std::to_string(e) + ".";
      for (int w = 0; w < 3; ++w) {
        const size_t at = static_cast<size_t>(e) * 3 + static_cast<size_t>(w);
        const std::string base = ep + names[w];
        if (m.fp4_of(w)) {
          float* g = load_global(base);
          m.experts_fp4[at] = w == 2 ? load_fp4_cols(base, r * I, I, g) : load_fp4_rows(base, r * I, I, g);
        } else {
          m.experts_fp8[at] = w == 2 ? load_quant_cols(base, r * I, I) : load_quant_rows(base, r * I, I);
        }
      }
    }
  }

  void build_layer(int layer) {
    if (layer < 0 || layer >= cfg.num_hidden_layers) fail("layer index out of range: " + std::to_string(layer));
    const std::string p = flint_layer_prefix(layer);
    out.layer = layer;
    out.moe = cfg.is_moe_layer(layer);
    out.swa = cfg.is_swa_layer(layer);
    out.rope = cfg.rope_of(layer);
    out.input_norm = load_bf16(p + "input_layernorm.weight");
    out.post_norm = load_bf16(p + "post_attention_layernorm.weight");
    build_attention(p + "self_attn.", layer);
    if (out.moe)
      build_moe(p + "mlp.", layer);
    else
      build_bf16_mlp(p + "mlp.", geo.local_dense_inter, out.dense, /*packable=*/true);
  }
};

namespace {

std::pair<int, int> lm_head_slice(const FlintTextConfig& cfg, int rank, int world) {
  const int64_t V = cfg.vocab_size;
  const int64_t begin = V * rank / world;
  const int64_t end = V * (rank + 1) / world;
  return {static_cast<int>(begin), static_cast<int>(end - begin)};
}

std::string& resident_image_dir_storage() {
  static std::string dir;
  return dir;
}

}  // namespace

// ---------------------------------------------------------------------------

FlintLocalGeometry FlintLocalGeometry::from_config(const FlintTextConfig& cfg, int rank, int world,
                                                   FlintHeadSharding head) {
  if (world > 1) flint_tp_validate_geometry(cfg, rank, world);
  if (world < 1 || rank < 0 || rank >= world) throw std::invalid_argument("flint loader: rank/world out of range");
  FlintLocalGeometry g;
  g.world = world;
  g.rank = rank;
  g.local_heads = cfg.num_attention_heads / world;
  g.head_begin = g.local_heads * rank;
  g.local_kv_heads = cfg.num_key_value_heads / world;
  g.kv_head_begin = g.local_kv_heads * rank;
  g.local_inter = cfg.moe_intermediate_size / world;
  g.local_dense_inter = cfg.intermediate_size / world;
  g.local_shared_inter = cfg.shared_expert_intermediate_size / world;
  if (head == FlintHeadSharding::VocabSharded) {
    const auto [b, n] = lm_head_slice(cfg, rank, world);
    g.lm_vocab_begin = b;
    g.lm_vocab_count = n;
  } else {
    g.lm_vocab_begin = 0;
    g.lm_vocab_count = cfg.vocab_size;
  }
  return g;
}

// ---- the family hooks (loaders/resident_stream.hpp) -------------------------

void FlintLoaderFamily::validate_binding(const FlintTextConfig& cfg, const PresentMap& present) {
  const FlintBindReport rep = flint_validate_text_binding(cfg, present);
  if (rep.ok()) return;
  std::string msg = "flint loader: checkpoint binding failed: ";
  for (size_t i = 0; i < rep.errors.size() && i < 8; ++i) {
    if (i) msg += "; ";
    msg += rep.errors[i];
  }
  throw std::runtime_error(msg);
}

bool FlintLoaderFamily::digest_included(const FlintExpectedTensor& e) { return is_replicated(e) && !e.unused(); }

// The packed column slices (o_proj, the MLPs' down projections) read
// their sources after the builder returns: those classes' sources are
// dropped after the packs.
bool FlintLoaderFamily::discard_after_pack(const FlintExpectedTensor& e) {
  return e.cls == FlintWeightClass::Attention || e.cls == FlintWeightClass::DenseMlp ||
         e.cls == FlintWeightClass::SharedExpert;
}

size_t FlintLoaderFamily::globals_bytes(const FlintTextConfig& cfg, int rank, int world, LoaderHeadSharding head) {
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  size_t b = 0;
  b += align_up_256(static_cast<size_t>(cfg.vocab_size) * H * 2);  // embed
  b += align_up_256(H * 2);                                        // final norm
  b += align_up_256(static_cast<size_t>(FlintLocalGeometry::from_config(cfg, rank, world, head).lm_vocab_count) * H * 2);
  return b;
}

size_t FlintLoaderFamily::globals_side_bytes(const FlintTextConfig& cfg, int rank, int world, LoaderHeadSharding head) {
  const int count = FlintLocalGeometry::from_config(cfg, rank, world, head).lm_vocab_count;
  if (!bf12_shape_ok(count, cfg.hidden_size)) return 0;
  return align_up_256(static_cast<size_t>(count) * static_cast<size_t>(cfg.hidden_size) * 2);
}

void FlintLoaderFamily::build_globals(const FlintTextConfig& cfg, const FlintLocalGeometry& geo,
                                      const LoaderTensorMap& tensors, LayerBump& bump, FlintGlobalsResident& out,
                                      uint64_t& source_bytes, uint64_t& verbatim_bytes, LoaderHeadSharding head) {
  auto lookup = [&](const std::string& name) -> const TensorInfo& {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second) throw std::runtime_error("flint loader: global tensor missing: " + name);
    return *it->second;
  };
  auto copy_global = [&](const std::string& name) -> uint16_t* {
    const TensorInfo& t = lookup(name);
    uint16_t* dst = static_cast<uint16_t*>(bump.alloc(t.nbytes()));
    std::memcpy(bump.host(dst), t.data, t.nbytes());
    source_bytes += t.nbytes();
    verbatim_bytes += t.nbytes();
    return dst;
  };
  out.embed = copy_global("model.embed_tokens.weight");
  out.final_norm = copy_global("model.norm.weight");
  const TensorInfo& t = lookup("lm_head.weight");
  const size_t row_bytes = static_cast<size_t>(cfg.hidden_size) * 2;
  const int begin = geo.lm_vocab_begin, count = geo.lm_vocab_count;
  uint16_t* dst = static_cast<uint16_t*>(bf12_shape_ok(count, cfg.hidden_size)
                                             ? bump.alloc_side(static_cast<size_t>(count) * row_bytes)
                                             : bump.alloc(static_cast<size_t>(count) * row_bytes));
  std::memcpy(bump.host(dst), static_cast<const uint8_t*>(t.data) + static_cast<size_t>(begin) * row_bytes,
              static_cast<size_t>(count) * row_bytes);
  source_bytes += static_cast<size_t>(count) * row_bytes;
  if (head == LoaderHeadSharding::Full) verbatim_bytes += static_cast<size_t>(count) * row_bytes;
  out.lm_head = dst;
  out.lm_vocab_begin = begin;
  out.lm_vocab_count = count;
}

template class ResidentLayerStream<FlintLoaderFamily>;

// ---------------------------------------------------------------------------

FlintLayerStream::FlintLayerStream(const FlintTextConfig& cfg, const std::string& checkpoint_dir, int rank, int world,
                                   FlintResidency residency, FlintHeadSharding head)
    : ResidentLayerStream<FlintLoaderFamily>(cfg, checkpoint_dir, rank, world, residency, head, /*resident_mtp=*/false) {
  open_resident_image();
}

void FlintLayerStream::set_resident_image_dir(const std::string& dir) { resident_image_dir_storage() = dir; }
const std::string& FlintLayerStream::resident_image_dir() { return resident_image_dir_storage(); }

}  // namespace dgpp
