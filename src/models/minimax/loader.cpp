#include "models/minimax/loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace dgpp {
namespace {
bool is_replicated(const MinimaxExpectedTensor& e) {
  if (e.role == MinimaxTensorRole::Fp4Global || e.role == MinimaxTensorRole::Fp4InputGlobal)
    return true;
  switch (e.cls) {
    case MinimaxWeightClass::Embed:
    case MinimaxWeightClass::FinalNorm:
    case MinimaxWeightClass::LayerNorm:
    case MinimaxWeightClass::Router:
      return true;
    case MinimaxWeightClass::Attention:
      return e.name.ends_with("_norm.weight") || e.name.find(".index_k_proj.") != std::string::npos;
    default:
      return false;
  }
}
std::string& image_directory() {
  static std::string dir;
  return dir;
}
}  // namespace

struct MinimaxLoaderFamily::Builder : WeightBuilder<MinimaxExpectedTensor> {
  const MinimaxConfig& cfg;
  const MinimaxLocalGeometry& geo;
  MinimaxLayerResident& out;
  Builder(const MinimaxConfig& c, const MinimaxLocalGeometry& g,
          const std::vector<MinimaxExpectedTensor>& tab,
          const std::unordered_map<std::string, const MinimaxExpectedTensor*>& names, LayerBump& b,
          MinimaxLayerResident& dst, const LoaderTensorMap& ts, std::vector<DequantJob>& jobs,
          std::vector<PackJob>& packs, bool do_copy)
      : WeightBuilder(tab, names, b, ts, jobs, packs, do_copy, g.rank, g.world, "minimax loader"),
        cfg(c),
        geo(g),
        out(dst) {}
  bool replicated(const MinimaxExpectedTensor& e) const override { return is_replicated(e); }
  bool fp4_global(const MinimaxExpectedTensor& e) const override {
    return e.role == MinimaxTensorRole::Fp4Global;
  }

  // Copy a rectangular MXFP8 slice, preserving both E4M3 payload and the
  // E8M0 scale grid. No scale inversion or conversion occurs at load time.
  GlmQuantMatrix mxfp8(const std::string& base, int64_t row0, int64_t rows, int64_t col0,
                       int64_t cols) {
    const auto& e = expected(base + ".weight");
    const auto& es = expected(base + ".weight_scale_inv");
    check_range(e.name, row0, rows, e.shape.at(0));
    check_range(e.name, col0, cols, e.shape.at(1));
    if (col0 % 32 || cols <= 0 || cols % 32 || rows <= 0) fail("unaligned MXFP8 slice: " + base);
    GlmQuantMatrix m;
    m.rows = rows;
    m.cols = cols;
    m.scale_block_rows = 1;
    m.scale_block_cols = 32;
    const auto payload_bytes = static_cast<size_t>(rows) * cols;
    m.payload = static_cast<uint8_t*>(bump.alloc(payload_bytes));
    m.scales_e8m0 = static_cast<uint8_t*>(bump.alloc(payload_bytes / 32));
    if (copy) {
      const auto& p = source(e.name);
      const auto& s = source(es.name);
      const auto* sp = static_cast<const uint8_t*>(p.data);
      const auto* ss = static_cast<const uint8_t*>(s.data);
      auto* dp = bump.host(const_cast<uint8_t*>(m.payload));
      auto* ds = bump.host(const_cast<uint8_t*>(m.scales_e8m0));
      const int64_t full = e.shape[1];
      for (int64_t r = 0; r < rows; ++r) {
        std::memcpy(dp + r * cols, sp + (row0 + r) * full + col0, static_cast<size_t>(cols));
        const auto* scales = ss + ((row0 + r) * full + col0) / 32;
        for (int64_t j = 0; j < cols / 32; ++j)
          if (scales[j] == 255) fail("NaN E8M0 scale in " + es.name);
        std::memcpy(ds + r * cols / 32, scales, static_cast<size_t>(cols / 32));
      }
      consumed(p);
      consumed(s);
    }
    note_read(e, payload_bytes);
    note_read(es, payload_bytes / 32);
    return m;
  }
  GlmQuantMatrix rows(const std::string& base, int64_t start, int64_t count) {
    return mxfp8(base, start, count, 0, expected(base + ".weight").shape.at(1));
  }
  GlmQuantMatrix cols(const std::string& base, int64_t start, int64_t count) {
    return mxfp8(base, 0, expected(base + ".weight").shape.at(0), start, count);
  }
  void mlp(const std::string& p, int64_t i, MinimaxMlpResident& m) {
    m.local_inter = i;
    m.gate = rows(p + "gate_proj", geo.rank * i, i);
    m.up = rows(p + "up_proj", geo.rank * i, i);
    m.down = cols(p + "down_proj", geo.rank * i, i);
  }
  const float* checked_global(const std::string& name) {
    // Reading scalars through load_f32 also includes them in replicated
    // digests and byte reconciliation, including the activation metadata.
    const float* p = load_f32(name);
    if (copy) {
      const float v = *bump.host(const_cast<float*>(p));
      if (!(v > 0) || !std::isfinite(v)) fail("nonpositive or nonfinite global scale: " + name);
    }
    return p;
  }
  void build_layer(int l) {
    const auto p = minimax_layer_prefix(l);
    out.layer = l;
    out.moe = cfg.is_moe_layer(l);
    out.input_norm = load_bf16(p + "input_layernorm.weight");
    out.post_norm = load_bf16(p + "post_attention_layernorm.weight");
    const auto a = p + "self_attn.";
    auto& w = out.attn;
    w.local_heads = geo.local_heads;
    w.local_kv_heads = geo.local_kv_heads;
    w.local_index_heads = geo.local_index_heads;
    w.q = rows(a + "q_proj", geo.head_begin * cfg.head_dim, geo.local_heads * cfg.head_dim);
    w.k = rows(a + "k_proj", geo.kv_head_begin * cfg.head_dim, geo.local_kv_heads * cfg.head_dim);
    w.v = rows(a + "v_proj", geo.kv_head_begin * cfg.head_dim, geo.local_kv_heads * cfg.head_dim);
    w.o = cols(a + "o_proj", geo.head_begin * cfg.head_dim, geo.local_heads * cfg.head_dim);
    w.index_q = rows(a + "index_q_proj", geo.index_head_begin * cfg.index_dim,
                     geo.local_index_heads * cfg.index_dim);
    w.index_k = rows(a + "index_k_proj", 0, cfg.index_dim);
    w.q_norm = load_bf16(a + "q_norm.weight");
    w.k_norm = load_bf16(a + "k_norm.weight");
    w.index_q_norm = load_bf16(a + "index_q_norm.weight");
    w.index_k_norm = load_bf16(a + "index_k_norm.weight");
    if (!out.moe) {
      mlp(p + "mlp.", geo.local_dense_inter, out.dense);
      return;
    }
    const auto ep = p + "block_sparse_moe.";
    auto& m = out.moe_w;
    m.router = load_f32(ep + "gate.weight");
    m.router_bias = load_f32(ep + "e_score_correction_bias");
    mlp(ep + "shared_experts.", geo.local_shared_inter, m.shared);
    m.local_inter = geo.local_inter;
    m.experts.resize(static_cast<size_t>(cfg.num_local_experts) * 3);
    m.input_globals.resize(m.experts.size());
    const char* names[] = {"w1", "w3", "w2"};
    for (int e = 0; e < cfg.num_local_experts; ++e)
      for (int j = 0; j < 3; ++j) {
        const auto b = ep + "experts." + std::to_string(e) + "." + names[j];
        const auto* global = checked_global(b + ".weight_global_scale");
        const size_t at = static_cast<size_t>(e) * 3 + j;
        m.input_globals[at] = checked_global(b + ".input_global_scale");
        m.experts[at] =
            j == 2
                ? load_fp4_cols(b + ".weight", geo.rank * geo.local_inter, geo.local_inter, global)
                : load_fp4_rows(b + ".weight", geo.rank * geo.local_inter, geo.local_inter, global);
      }
  }
};

MinimaxLocalGeometry MinimaxLocalGeometry::from_config(const MinimaxConfig& c, int rank, int world,
                                                       LoaderHeadSharding head) {
  c.validate_tp(rank, world);
  MinimaxLocalGeometry g;
  g.rank = rank;
  g.world = world;
  g.local_heads = c.num_attention_heads / world;
  g.head_begin = rank * g.local_heads;
  g.local_kv_heads = c.num_key_value_heads / world;
  g.kv_head_begin = rank * g.local_kv_heads;
  g.local_index_heads = c.index_heads / world;
  g.index_head_begin = rank * g.local_index_heads;
  g.local_inter = c.intermediate_size / world;
  g.local_dense_inter = c.dense_intermediate_size / world;
  g.local_shared_inter = c.shared_intermediate_size / world;
  g.lm_vocab_begin = head == LoaderHeadSharding::VocabSharded
                         ? static_cast<int>(int64_t(c.vocab_size) * rank / world)
                         : 0;
  const int end = head == LoaderHeadSharding::VocabSharded
                      ? static_cast<int>(int64_t(c.vocab_size) * (rank + 1) / world)
                      : c.vocab_size;
  g.lm_vocab_count = end - g.lm_vocab_begin;
  return g;
}
void MinimaxLoaderFamily::validate_binding(const MinimaxConfig& c, const PresentMap& p) {
  const auto r = minimax_validate_text_binding(c, p);
  if (r.ok()) return;
  std::string message = "minimax loader: checkpoint binding failed";
  for (const auto& e : r.errors) message += "; " + e;
  throw std::runtime_error(message);
}
bool MinimaxLoaderFamily::digest_included(const MinimaxExpectedTensor& e) {
  return is_replicated(e);
}

size_t MinimaxLoaderFamily::globals_bytes(const MinimaxConfig& cfg, int rank, int world,
                                          LoaderHeadSharding head) {
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  size_t b = 0;
  b += align_up_256(static_cast<size_t>(cfg.vocab_size) * H * 2);  // embed
  b += align_up_256(H * 2);                                        // final norm
  b += align_up_256(static_cast<size_t>(
                        MinimaxLocalGeometry::from_config(cfg, rank, world, head).lm_vocab_count) *
                    H * 2);
  return b;
}

size_t MinimaxLoaderFamily::globals_side_bytes(const MinimaxConfig& cfg, int rank, int world,
                                               LoaderHeadSharding head) {
  const int count = MinimaxLocalGeometry::from_config(cfg, rank, world, head).lm_vocab_count;
  if (!bf12_shape_ok(count, cfg.hidden_size)) return 0;
  return align_up_256(static_cast<size_t>(count) * static_cast<size_t>(cfg.hidden_size) * 2);
}

void MinimaxLoaderFamily::build_globals(const MinimaxConfig& cfg, const MinimaxLocalGeometry& geo,
                                        const LoaderTensorMap& tensors, LayerBump& bump,
                                        MinimaxGlobalsResident& out, uint64_t& source_bytes,
                                        uint64_t& verbatim_bytes, LoaderHeadSharding head) {
  auto lookup = [&](const std::string& name) -> const TensorInfo& {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second)
      throw std::runtime_error("minimax loader: global tensor missing: " + name);
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
  out.embed = copy_global("language_model.model.embed_tokens.weight");
  out.final_norm = copy_global("language_model.model.norm.weight");
  const TensorInfo& t = lookup("language_model.lm_head.weight");
  const size_t row_bytes = static_cast<size_t>(cfg.hidden_size) * 2;
  const int begin = geo.lm_vocab_begin, count = geo.lm_vocab_count;
  uint16_t* dst =
      static_cast<uint16_t*>(bf12_shape_ok(count, cfg.hidden_size)
                                 ? bump.alloc_side(static_cast<size_t>(count) * row_bytes)
                                 : bump.alloc(static_cast<size_t>(count) * row_bytes));
  std::memcpy(bump.host(dst),
              static_cast<const uint8_t*>(t.data) + static_cast<size_t>(begin) * row_bytes,
              static_cast<size_t>(count) * row_bytes);
  source_bytes += static_cast<size_t>(count) * row_bytes;
  if (head == LoaderHeadSharding::Full) verbatim_bytes += static_cast<size_t>(count) * row_bytes;
  out.lm_head = dst;
  out.lm_vocab_begin = begin;
  out.lm_vocab_count = count;
}

template class ResidentLayerStream<MinimaxLoaderFamily>;

// ---------------------------------------------------------------------------

MinimaxLayerStream::MinimaxLayerStream(const MinimaxConfig& cfg, const std::string& checkpoint_dir,
                                       int rank, int world, LoaderResidency residency,
                                       LoaderHeadSharding head)
    : ResidentLayerStream<MinimaxLoaderFamily>(cfg, checkpoint_dir, rank, world, residency, head,
                                               /*resident_mtp=*/false) {
  open_resident_image();
}

void MinimaxLayerStream::set_resident_image_dir(const std::string& dir) {
  image_directory() = dir;
}
const std::string& MinimaxLayerStream::resident_image_dir() {
  return image_directory();
}

}  // namespace dgpp
