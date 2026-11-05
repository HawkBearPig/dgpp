#include "models/dory/loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace dgpp {
namespace {

// The rank-invariant set. The family is one rank (DoryGeometry::from_config
// refuses any world but 1, plan §2), so every tensor is rank-invariant; the
// classifier still names each class so the switch stays exhaustive under
// -Werror and a future world > 1 has one place to change. Every NVFP4
// projection would be a row/column slice at world > 1; the learned scales,
// the per-tensor global scales, the embedding and the head are replicated.
bool is_replicated(const DoryExpectedTensor& e) {
  if (e.role == DoryTensorRole::Fp4Global) return true;
  switch (e.cls) {
    case DoryWeightClass::Embed:
    case DoryWeightClass::LmHead:
    case DoryWeightClass::LogitScale:
    case DoryWeightClass::AttentionScale:  // s_gate, sqk
    case DoryWeightClass::AttentionAlpha:  // attn_alpha
    case DoryWeightClass::MlpScale:        // suv_gate, suv_up
    case DoryWeightClass::MlpAlpha:        // mlp_alpha
      return true;
    case DoryWeightClass::AttentionProj:  // one rank: kept whole
    case DoryWeightClass::MlpProj:
      return true;
  }
  return false;
}

std::string& resident_image_dir_storage() {
  static std::string dir;
  return dir;
}

}  // namespace

// The per-class builders (loaders/weight_build.hpp's primitives).
struct DoryLoaderFamily::Builder : WeightBuilder<DoryExpectedTensor> {
  const DoryConfig& cfg;
  DoryLayerResident& out;
  float* globals_ = nullptr;  // the layer's gathered fp4 global scales (device)
  int globals_used_ = 0;
  int globals_total_ = 0;

  Builder(const DoryConfig& cfg_, const DoryGeometry& geo_,
          const std::vector<DoryExpectedTensor>& table_,
          const std::unordered_map<std::string, const DoryExpectedTensor*>& by_name_,
          LayerBump& bump_, DoryLayerResident& out_,
          const std::unordered_map<std::string, const TensorInfo*>& tensors_,
          std::vector<DequantJob>& jobs_, std::vector<PackJob>& packs_, bool copy_)
      : WeightBuilder<DoryExpectedTensor>(table_, by_name_, bump_, tensors_, jobs_, packs_, copy_,
                                           geo_.rank, geo_.world, "dory loader"),
        cfg(cfg_), out(out_) {}

  bool replicated(const DoryExpectedTensor& e) const override { return is_replicated(e); }
  bool fp4_global(const DoryExpectedTensor& e) const override {
    return e.role == DoryTensorRole::Fp4Global;
  }

  // ---- the gathered global scales ----------------------------------------
  void reserve_globals(int n) {
    globals_total_ = n;
    globals_used_ = 0;
    globals_ = static_cast<float*>(bump.alloc(static_cast<size_t>(std::max(n, 1)) * 4));
  }
  // The next slot: the device address the view points at (a real address
  // whenever the bump is not counting — the image restore's layout pass
  // allocates without copying); `*host` its staging mirror (copy mode).
  float* next_global(float** host) {
    if (globals_used_ >= globals_total_) fail("global scale slots exhausted (builder bug)");
    const int i = globals_used_++;
    *host = copy ? bump.host(globals_) + i : nullptr;
    return globals_ ? globals_ + i : reinterpret_cast<float*>(static_cast<size_t>(i) * 4);
  }
  // The modelopt per-tensor scale of `base` into a slot as 1 / weight_scale_2
  // (the kernels divide the finished dot by it once).
  float* load_global_reciprocal(const std::string& base) {
    const DoryExpectedTensor& eg = expected(base + ".weight_scale_2");
    float* host = nullptr;
    float* slot = next_global(&host);
    if (copy) {
      const TensorInfo& t = source(eg.name);
      float ws2;
      std::memcpy(&ws2, t.data, 4);
      if (!(ws2 > 0.0f) || !std::isfinite(ws2))
        fail("'" + eg.name + "' is not a positive finite scale");
      *host = 1.0f / ws2;
      consumed(t);
    }
    note_read(eg, 4);
    return slot;
  }

  // ---- NVFP4 matrices in the modelopt layout ------------------------------
  // The whole [N, K] matrix `base`: the U8 e2m1 pairs, the e4m3 scale grid,
  // and the gathered F32 reciprocal global scale. No slicing: the family is
  // one rank.
  GlmFp4Matrix load_fp4(const std::string& base) {
    const DoryExpectedTensor& ep = expected(base + ".weight");
    const DoryExpectedTensor& es = expected(base + ".weight_scale");
    const int64_t N = ep.shape[0];
    const int64_t cols = ep.shape[1] * 2;
    fp4_check_cols(cols, who.c_str());
    if (es.shape[0] != N || es.shape[1] != cols / kFp4Group)
      fail("NVFP4 scale geometry mismatch on " + base);
    const size_t pc = static_cast<size_t>(cols / 2), sc = static_cast<size_t>(cols / kFp4Group);
    GlmFp4Matrix q;
    q.rows = N;
    q.cols = cols;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(N) * pc));
    q.scales = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(N) * sc));
    if (copy) {
      const TensorInfo& tp = source(ep.name);
      const TensorInfo& ts = source(es.name);
      std::memcpy(bump.host(const_cast<uint8_t*>(q.payload)), tp.data, static_cast<size_t>(N) * pc);
      std::memcpy(bump.host(const_cast<uint8_t*>(q.scales)), ts.data, static_cast<size_t>(N) * sc);
      consumed(tp);
      consumed(ts);
    }
    note_read(ep, static_cast<size_t>(N) * pc);
    note_read(es, static_cast<size_t>(N) * sc);
    q.global_scale = load_global_reciprocal(base);
    return q;
  }

  // ---- the classes ---------------------------------------------------------
  void build_attention(const std::string& p) {
    const std::string m = p + "mixer.";
    DoryAttentionResident& a = out.attn;
    a.q_proj = load_fp4(m + "q_proj");
    a.k_proj = load_fp4(m + "k_proj");
    a.v_proj = load_fp4(m + "v_proj");
    a.o_proj = load_fp4(m + "o_proj");
    a.g_proj = load_fp4(m + "g_proj");
    a.s_gate = load_bf16(m + "s_gate");
    a.sqk = load_bf16(m + "sqk");
    a.attn_alpha = load_bf16(p + "attn_alpha");
  }

  void build_mlp(const std::string& p) {
    const std::string m = p + "mixer.";
    DoryMlpResident& r = out.mlp;
    r.gate = load_fp4(m + "gate_proj");
    r.up = load_fp4(m + "up_proj");
    r.down = load_fp4(m + "down_proj");
    r.suv_gate = load_bf16(m + "suv_gate");
    r.suv_up = load_bf16(m + "suv_up");
    r.mlp_alpha = load_bf16(p + "mlp_alpha");
  }

  void build_layer(int layer) {
    if (layer < 0 || layer >= cfg.num_hidden_layers)
      fail("layer index out of range: " + std::to_string(layer));
    const std::string p = dory_layer_prefix(cfg, layer);
    out.layer = layer;
    out.kind = cfg.kind_of(layer);
    // One global-scale slot per NVFP4 matrix of the layer.
    reserve_globals(cfg.is_attention(layer) ? 5 : 3);
    if (cfg.is_attention(layer))
      build_attention(p);
    else
      build_mlp(p);
    if (globals_used_ != globals_total_) fail("global scale slots left unused (builder bug)");
  }
};

// ---------------------------------------------------------------------------

DoryGeometry DoryGeometry::from_config(const DoryConfig& cfg, int rank, int world,
                                       LoaderHeadSharding /*head*/) {
  dory_validate_world(world);
  if (rank < 0 || rank >= world)
    throw std::invalid_argument("dory loader: rank/world out of range");
  DoryGeometry g;
  g.world = world;
  g.rank = rank;
  g.lm_vocab_begin = 0;
  g.lm_vocab_count = cfg.vocab_size;
  return g;
}

// ---- the family hooks (loaders/resident_stream.hpp) -------------------------

void DoryLoaderFamily::validate_binding(const DoryConfig& cfg, const PresentMap& present) {
  const DoryBindReport rep = dory_validate_text_binding(cfg, present);
  if (rep.ok()) return;
  std::string msg = "dory loader: checkpoint binding failed: ";
  for (size_t i = 0; i < rep.errors.size() && i < 8; ++i) {
    if (i) msg += "; ";
    msg += rep.errors[i];
  }
  throw std::runtime_error(msg);
}

// No duplicated tensors in this release (no MTP draft, no tied embedding),
// so there is nothing beyond the binding to cross-check.
void DoryLoaderFamily::check_sources(const DoryConfig&, const LoaderTensorMap&) {}

bool DoryLoaderFamily::digest_included(const DoryExpectedTensor& e) {
  return is_replicated(e);
}

size_t DoryLoaderFamily::globals_bytes(const DoryConfig& cfg, int /*rank*/, int /*world*/,
                                       LoaderHeadSharding /*head*/) {
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t V = static_cast<size_t>(cfg.vocab_size);
  size_t b = 0;
  b += align_up_256(V * H * 2);  // embed
  b += align_up_256(V * H * 2);  // lm_head
  b += align_up_256(V * 2);      // logit_scale
  return b;
}

void DoryLoaderFamily::build_globals(const DoryConfig& cfg, const DoryGeometry& /*geo*/,
                                     const LoaderTensorMap& tensors, LayerBump& bump,
                                     DoryGlobalsResident& out, uint64_t& source_bytes,
                                     uint64_t& verbatim_bytes, LoaderHeadSharding /*head*/) {
  auto lookup = [&](const std::string& name) -> const TensorInfo& {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second)
      throw std::runtime_error("dory loader: global tensor missing: " + name);
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
  out.embed = copy_global("backbone.embeddings.weight");
  out.lm_head = copy_global("lm_head.weight");
  out.logit_scale = copy_global("lm_head.logit_scale");
  (void)cfg;
}

template class ResidentLayerStream<DoryLoaderFamily>;

// ---------------------------------------------------------------------------

DoryLayerStream::DoryLayerStream(const DoryConfig& cfg, const std::string& checkpoint_dir,
                                 int rank, int world, DoryResidency residency,
                                 LoaderHeadSharding head, bool resident_mtp)
    : ResidentLayerStream<DoryLoaderFamily>(cfg, checkpoint_dir, rank, world, residency, head,
                                            resident_mtp) {
  open_resident_image();
}

void DoryLayerStream::set_resident_image_dir(const std::string& dir) {
  resident_image_dir_storage() = dir;
}
const std::string& DoryLayerStream::resident_image_dir() { return resident_image_dir_storage(); }

}  // namespace dgpp
