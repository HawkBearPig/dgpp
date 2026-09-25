#include "models/mimo/forward.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "common/bf16_residency.hpp"
#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "kernels/add_rmsnorm.hpp"
#include "kernels/bf12_companions.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/kernels.hpp"
#include "kernels/mimo_attn.hpp"
#include "kernels/mimo_mtp_history.hpp"

namespace dgpp {
namespace {

template <class T>
T* dev_alloc(size_t n) {
  T* p = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  return p;
}

template <class T>
T* pinned_alloc(size_t n) {
  T* p = nullptr;
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&p), std::max<size_t>(n, 1) * sizeof(T),
                             cudaHostAllocMapped));
  return p;
}

// The MoE layers' ordinals (the graph table slots and the trace slots):
// main MoE layers in order, the draft (dense) at -1.
std::vector<int> moe_ordinals(const MimoTextConfig& cfg) {
  std::vector<int> o(static_cast<size_t>(cfg.num_hidden_layers + cfg.mtp_layers_loaded), -1);
  int n = 0;
  for (int l = 0; l < cfg.num_hidden_layers; ++l)
    if (cfg.is_moe_layer(l)) o[static_cast<size_t>(l)] = n++;
  return o;
}

}  // namespace

std::vector<int> MimoModel::pool_kv_heads(const MimoTextConfig& cfg, const MimoLocalGeometry& geo, bool mtp) {
  std::vector<int> heads;
  for (int l = 0; l < cfg.num_hidden_layers; ++l) heads.push_back(geo.kv_heads_of(cfg, l));
  if (mtp)
    for (int d = 0; d < cfg.mtp_layers_loaded; ++d) heads.push_back(geo.local_swa_kv_heads);
  return heads;
}

MimoModel::MimoModel(const MimoTextConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
                     int64_t max_cache_tokens, MimoResidency residency, BoundaryReducer* boundary,
                     int tp_rank, int tp_world, int max_requests, bool mtp, int decode_rows, LatentFormat kv_format)
    : cfg_(cfg),
      loader_(cfg, checkpoint_dir, tp_rank, tp_world, residency,
              tp_world > 1 ? MimoHeadSharding::VocabSharded : MimoHeadSharding::Full,
              mtp && residency == MimoResidency::Resident),
      moe_ordinal_(moe_ordinals(cfg)) {
  if (decode_rows > decode_rows_cap())
    throw std::invalid_argument("MimoModel: decode_rows exceeds the MiMo-V2 limit of 32");
  if (max_tokens <= 0) throw std::invalid_argument("MimoModel: max_tokens must be positive");
  if (mtp && cfg_.mtp_layers_loaded == 3 && max_tokens > kPrefillChunkTokens)
    throw std::invalid_argument("native MiMo MTP supports at most 2048 rows per walk");
  if (mtp && cfg_.mtp_layer() < 0) throw std::invalid_argument("MimoModel: the config has no draft layer (mtp)");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("MimoModel: max_requests must be in [1, kPickMaxRequests]");
  if ((tp_world > 1) != (boundary != nullptr))
    throw std::invalid_argument(
        "MimoModel: a boundary reducer is required exactly when tp_world > 1 — without one the "
        "block-boundary partials would be returned silently as results");
  if (cfg_.eos_token_ids.empty()) throw std::invalid_argument("MimoModel: the config names no EOS token");
  if (cfg_.head_dim != kMimoQkDim || cfg_.v_head_dim != kMimoVDim)
    throw std::invalid_argument("MimoModel: qk 192 / v 128 heads (the kernels' shape)");
  if (kv_format != LatentFormat::kBf16 && kv_format != LatentFormat::kFp8)
    throw std::invalid_argument("MimoModel: the K/V cache format must be bf16 or fp8");
  init_stream();
  loader_.set_reader_stream(stream_);
  const int H = cfg_.hidden_size;
  globals_ = loader_.load_globals();
  {
    SessionParams sp;
    sp.max_tokens = max_tokens;
    sp.max_cache_tokens =
        ((std::max<int64_t>(max_cache_tokens, max_tokens) + kBlockTokens - 1) / kBlockTokens) * kBlockTokens;
    sp.rank = tp_rank;
    sp.world = tp_world;
    sp.boundary = boundary;
    sp.max_requests = max_requests;
    sp.decode_rows = decode_rows;
    sp.mtp = mtp;
    sp.vocab_size = cfg_.vocab_size;
    sp.hidden = H;
    sp.lm_vocab_begin = globals_.lm_vocab_begin;
    sp.lm_vocab_count = globals_.lm_vocab_count > 0 ? globals_.lm_vocab_count : cfg_.vocab_size;
    sp.max_position_embeddings = cfg_.max_position_embeddings;
    sp.block_tokens = kBlockTokens;
    sp.snapshot_align = 1;  // no recurrent state, any position snapshots
    sp.draft_width = H;
    sp.eos = static_cast<int32_t>(cfg_.eos_token_ids[0]);
    init_session(sp);
  }
  gemm_ws_bytes_ = std::max<size_t>(64u << 20, gemm_.query_workspace_bytes(max_tokens_, lm_vocab_count_, H, DType::BF16));
  gemm_ws_ = dev_alloc<char>(gemm_ws_bytes_);
  gw_ = MimoGemmWorkspace{&gemm_, gemm_ws_, gemm_ws_bytes_};
  // The bf16 sites' lowering (o_proj, the head, eh_proj — kernels/gemm.hpp
  // dense_gemv_rows): the GEMV chunks (the 12-bit companions) to the
  // bound, cuBLASLt's algorithm above it.
  gemm_.set_decode_rows(std::min(max_decode_rows_, dense_gemv_rows()));
  // The fp8 sites (the fused projection, the dense MLPs) take the streaming
  // tensor-core GEMM at decode rows: a row's chain the same whatever rows
  // share the launch (DGPP_MIMO_DENSE_GEMV=1 restores the 4-row chunks).
  dense_mma_ = std::getenv("DGPP_MIMO_DENSE_GEMV") == nullptr;
  if (const char* v = std::getenv("DGPP_MIMO_PREFILL_LAST_HEAD"))
    prefill_last_head_ = std::string(v) == "1";
  if (const char* v = std::getenv("DGPP_MIMO_MTP_CACHE_ONLY"))
    mtp_cache_only_ = std::string(v) == "1";
  bf16_side_grants_ = residency == MimoResidency::Resident && bf16_side_grants();
  moe_cfg_ = cfg_.moe_config(static_cast<int>(loader_.geometry().local_inter));
  n_split_ = MimoAttentionLayer::default_decode_splits();

  const MimoLocalGeometry& geo = loader_.geometry();
  {
    MimoKvPoolShape shape;
    shape.kv_heads = pool_kv_heads(cfg_, geo, mtp_);
    shape.k_dim = cfg_.head_dim;
    shape.v_dim = cfg_.v_head_dim;
    shape.block_tokens = kBlockTokens;
    shape.max_requests = max_requests_;
    shape.token_slots = max_cache_tokens_;
    shape.format = kv_format;
    pool_.init(shape);
  }
  const size_t M = static_cast<size_t>(max_tokens_);
  resid_ = dev_alloc<uint16_t>(M * H);
  x_ = dev_alloc<uint16_t>(M * H);
  y_ = dev_alloc<uint16_t>(M * H);
  moe_acc_ = dev_alloc<float>(M * H);
  {
    const size_t layers = static_cast<size_t>(cfg_.num_moe_layers());
    const size_t K = static_cast<size_t>(cfg_.num_experts_per_tok);
    h_route_ids_ = pinned_alloc<int32_t>(layers * M * K);
    h_route_weights_ = pinned_alloc<float>(layers * M * K);
    h_route_biased_ = pinned_alloc<float>(layers * static_cast<size_t>(max_decode_rows_) *
                                          static_cast<size_t>(cfg_.n_routed_experts));
    // The boundary windows' budget: 20 MB as the GLM-4.7 walk's;
    // DGPP_L2_PREFETCH_MB overrides.
    prefetch_window_bytes_ = std::getenv("DGPP_L2_PREFETCH_MB") ? 0 : (size_t{20} << 20);
    // The early projection prefetch experiment (plan §7.1, REJECTED on the
    // fabric 2026-09-22: +7-10 % per step at 8 and 16 MiB — the window runs
    // through the expert stream at the Light rate and competes there, and
    // the o_proj window loses its L2): the next layer's fused qkv read into
    // the L2 persisting set-aside from the attention fold on.
    // DGPP_MIMO_L2_PERSIST_MB=<MiB> re-arms it (0 or unset: off; the device
    // grants at most 18 MB here).
    long mb = 0;
    if (const char* v = std::getenv("DGPP_MIMO_L2_PERSIST_MB"); v && *v) mb = std::strtol(v, nullptr, 10);
    if (const char* v = std::getenv("DGPP_MIMO_EARLY_QKV_MB"); v && *v && prefetch_.enabled()) {
      const long e = std::strtol(v, nullptr, 10);
      if (e > 0) early_qkv_bytes_ = static_cast<size_t>(e) << 20;
    }
    if (mb > 0 && prefetch_.enabled()) {
      l2_persist_bytes_ = l2_set_persisting_limit(static_cast<size_t>(mb) << 20);
      DGPP_LOG_INFO("mimo: early projection prefetch into a {} MiB persisting L2 set-aside",
                    l2_persist_bytes_ >> 20);
    }
  }
  if (mtp_) {
    mtp_h_ = dev_alloc<uint16_t>(M * H);
    mtp_in_ = dev_alloc<uint16_t>(M * 2 * H);
    mtp_r_ = dev_alloc<uint16_t>(M * H);
    if (native_mtp()) {
      native_history_ = dev_alloc<uint16_t>(size_t(max_requests_) * kNativeHistoryRows * H);
      native_backup_ = dev_alloc<uint8_t>(size_t(max_requests_) * draft_state_bytes());
      native_positions_ = dev_alloc<int64_t>(M);
      native_output_ = dev_alloc<uint16_t>(M * H);
      DGPP_CUDA_OK(cudaMemsetAsync(native_history_, 0, size_t(max_requests_) * draft_state_bytes(),
                                   stream_));
    }
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}

MimoModel::~MimoModel() {
  cudaFree(resid_);
  cudaFree(x_);
  cudaFree(y_);
  cudaFree(moe_acc_);
  cudaFreeHost(h_route_ids_);
  cudaFreeHost(h_route_weights_);
  cudaFreeHost(h_route_biased_);
  cudaFree(gemm_ws_);
  cudaFree(mtp_h_);
  cudaFree(mtp_in_);
  cudaFree(mtp_r_);
  cudaFree(native_history_);
  cudaFree(native_backup_);
  cudaFree(native_positions_);
  cudaFree(native_output_);
}

int MimoModel::table_slots() const {
  return loader_.residency() == MimoResidency::Resident ? cfg_.num_moe_layers() : 0;
}

size_t MimoModel::kv_block_bytes_static(const MimoTextConfig& cfg, int tp_rank, int tp_world, bool mtp,
                                        LatentFormat kv_format) {
  const MimoHeadSharding head = tp_world > 1 ? MimoHeadSharding::VocabSharded : MimoHeadSharding::Full;
  const MimoLocalGeometry geo = MimoLocalGeometry::from_config(cfg, tp_rank, tp_world, head);
  const auto bytes = [&](int64_t tokens) {
    MimoKvPoolShape shape;
    shape.kv_heads = pool_kv_heads(cfg, geo, mtp);
    shape.k_dim = cfg.head_dim;
    shape.v_dim = cfg.v_head_dim;
    shape.block_tokens = kBlockTokens;
    shape.max_requests = 1;
    shape.token_slots = tokens;
    shape.format = kv_format;
    return MimoKvPool::cache_bytes(shape) - PagedBlockTable::table_bytes(1, tokens / kBlockTokens);
  };
  return bytes(2 * kBlockTokens) - bytes(kBlockTokens);
}

MimoModel::MemoryPlan MimoModel::plan_memory(const MimoTextConfig& cfg, int max_tokens, int64_t max_cache_tokens,
                                             int tp_rank, int tp_world, MimoResidency residency, int max_requests,
                                             bool mtp, int decode_rows, LatentFormat kv_format) {
  if (max_tokens <= 0) throw std::invalid_argument("plan_memory: max_tokens must be positive");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("plan_memory: max_requests must be in [1, kPickMaxRequests]");
  if (decode_rows > decode_rows_cap())
    throw std::invalid_argument("plan_memory: decode_rows exceeds the MiMo-V2 limit of 32");
  // The fixed batch's row ceiling, floored as the session core floors it.
  const int rows = std::max({kDecodeRows, decode_rows, max_requests});
  if (mtp && cfg.mtp_layer() < 0) throw std::invalid_argument("plan_memory: the config has no draft layer");
  const MimoHeadSharding head = tp_world > 1 ? MimoHeadSharding::VocabSharded : MimoHeadSharding::Full;
  const MimoLocalGeometry geo = MimoLocalGeometry::from_config(cfg, tp_rank, tp_world, head);
  const int64_t cache_tokens =
      ((std::max<int64_t>(max_cache_tokens, max_tokens) + kBlockTokens - 1) / kBlockTokens) * kBlockTokens;
  MemoryPlan plan;
  plan.context_tokens = std::min<int64_t>(cache_tokens, cfg.max_position_embeddings);
  const size_t M = static_cast<size_t>(max_tokens);
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t V = static_cast<size_t>(MimoLayerStream::lm_vocab_count(cfg, tp_rank, tp_world, head));
  const int pool_layers = cfg.num_hidden_layers + (mtp ? cfg.mtp_layers_loaded : 0);
  const int n_split = MimoAttentionLayer::default_decode_splits();

  // bf12-only residency: the packable bf16 matrices load aside and give
  // their bytes back as each layer is packed (graph_prepare).
  const bool bf12_only = Bf12Companions::packed_only() && residency == MimoResidency::Resident;
  if (residency == MimoResidency::Resident) {
    size_t resident = MimoLayerStream::resident_bytes(cfg, tp_rank, tp_world, head, mtp);
    if (bf12_only) resident -= MimoLayerStream::side_bytes(cfg, tp_rank, tp_world, head, mtp);
    plan.add(bf12_only ? "model weights (resident, packed bf16 matrices released)" : "model weights (resident)",
             resident);
    plan.add("loader staging (pinned host, freed when the last layer is resident)", 0,
             MimoLayerStream::staging_plan_bytes(cfg, tp_rank, tp_world, head, mtp));
  } else {
    size_t largest = 0;
    const int layers_total = cfg.num_hidden_layers + cfg.mtp_layers_loaded;
    for (int l = 0; l < layers_total; ++l)
      largest = std::max(largest, MimoLayerStream::layer_bytes(cfg, l, tp_rank, tp_world));
    plan.add("model weights (one streamed layer + globals)",
             largest + MimoLayerStream::globals_bytes(cfg, tp_rank, tp_world, head));
  }
  if (Bf12Companions::enabled() && residency == MimoResidency::Resident) {
    // The lossless 12-bit companions of the bf16 o_proj of every layer, the
    // head and the draft's eh_proj (kernels/bf12_companions.hpp).
    const int64_t O = static_cast<int64_t>(geo.local_heads) * cfg.v_head_dim;
    const int64_t Hh = cfg.hidden_size;
    size_t packed = static_cast<size_t>(pool_layers) * Bf12Companions::planned_bytes(Hh, O) +
                    Bf12Companions::planned_bytes(static_cast<int64_t>(V), Hh);
    if (mtp) packed += cfg.mtp_layers_loaded * Bf12Companions::planned_bytes(Hh, 2 * Hh);
    plan.add("bf16 decode packing (12-bit companions)", packed);
    if (bf12_only) {
      const size_t h = static_cast<size_t>(Hh);
      plan.add("bf16 prefill expansion scratch",
               static_cast<size_t>(kBf12ExpandSlots) *
                   Bf12Companions::planned_slot_bytes({h * static_cast<size_t>(O) * 2, mtp ? h * 2 * h * 2 : 0, V * h * 2}));
    }
  }
  plan.add("gemm workspace (at least)", size_t{64} << 20);
  {
    MimoKvPoolShape shape;
    shape.kv_heads = pool_kv_heads(cfg, geo, mtp);
    shape.k_dim = cfg.head_dim;
    shape.v_dim = cfg.v_head_dim;
    shape.block_tokens = kBlockTokens;
    shape.max_requests = max_requests;
    shape.token_slots = cache_tokens;
    shape.format = kv_format;
    plan.add(kv_format == LatentFormat::kFp8 ? "kv cache pool (K/V fp8 rows + scales, every layer, the draft's included)"
                                             : "kv cache pool (K/V bf16, every layer, the draft's included)",
             MimoKvPool::cache_bytes(shape));
  }
  {
    size_t core_dev = 0, core_pin = 0;
    session_core_plan_bytes(max_tokens, max_requests, cfg.hidden_size, static_cast<int>(V), mtp,
                            cfg.hidden_size, &core_dev, &core_pin, rows);
    const size_t moe_layers = static_cast<size_t>(cfg.num_moe_layers());
    const size_t K = static_cast<size_t>(cfg.num_experts_per_tok);
    const size_t E = static_cast<size_t>(cfg.n_routed_experts);
    plan.add("activations (session core, residual, block io, the moe chain's fp32 sum, route staging)",
             core_dev + 3 * M * H * 2 + M * H * 4,
             core_pin + moe_layers * M * K * 8 + moe_layers * static_cast<size_t>(rows) * E * 4);
  }
  {
    size_t layers = MimoAttentionLayer::scratch_bytes(cfg, geo.local_heads, max_tokens, rows, n_split);
    layers += MimoDenseMlp::scratch_bytes(cfg, geo.local_dense_inter, max_tokens);
    plan.add("layer objects (attention, dense MLP)", layers);
    const GlmMoeConfig moe_cfg = cfg.moe_config(static_cast<int>(geo.local_inter));
    const int slots = residency == MimoResidency::Resident ? cfg.num_moe_layers() : 0;
    size_t moe_pinned = 0;
    const size_t moe_dev = GlmMoeLayer::scratch_bytes(moe_cfg, max_tokens, rows, slots, &moe_pinned);
    plan.add("moe scratch (routed slots, graph tables)", moe_dev, moe_pinned);
  }
  if (mtp && cfg.mtp_layers_loaded == 3) {
    if (max_tokens > kPrefillChunkTokens)
      throw std::invalid_argument("native MiMo MTP supports at most 2048 rows per walk");
    plan.add("native MTP history, rollback and scratch",
             size_t(max_requests) * kNativeHistoryRows * H * 4 + M * (H * 2 + 8));
  }
  if (mtp) plan.add("draft block (gathered hidden, the fused input, its residual)", M * H * 2 + M * 2 * H * 2 + M * H * 2);
  return plan;
}

size_t MimoModel::session_snapshot_bytes(const MimoTextConfig& cfg, int, bool mtp) {
  size_t bytes = kSnapshotStamp;
  if (mtp && cfg.mtp_layers_loaded == 3) bytes += size_t(kNativeHistoryRows) * cfg.hidden_size * 2;
  if (mtp) bytes += static_cast<size_t>(cfg.hidden_size) * 2;  // the draft's hidden row
  return bytes;
}

GlmMoeWeights MimoModel::moe_view(const MimoMoeResident& m) {
  GlmMoeWeights w;
  w.router_gate = m.router;
  w.router_bias = m.router_bias;
  w.experts = nullptr;
  w.experts_fp4 = m.experts.data();
  return w;
}

void MimoModel::build_layer_objects(const MimoLayerResident& r) {
  if (!attn_) {
    attn_ = std::make_unique<MimoAttentionLayer>(r.attn, gw_, cfg_, max_tokens_, max_decode_rows_, n_split_, dense_mma_);
  } else {
    attn_->rebind(r.attn);
  }
  if (r.moe) {
    if (!moe_) {
      moe_ = std::make_unique<GlmMoeLayer>(moe_view(r.moe_w), moe_cfg_, max_tokens_, max_decode_rows_, table_slots());
    } else {
      moe_->rebind(moe_view(r.moe_w));
    }
  } else {
    if (!dense_) {
      dense_ = std::make_unique<MimoDenseMlp>(r.dense, cfg_, max_tokens_, max_decode_rows_, dense_mma_);
    } else {
      dense_->rebind(r.dense);
    }
  }
}

// The state every row walk starts from on a fresh request: no blocks
// (the K/V rows are written before any row reads them).
void MimoModel::reset_slot_state(int req) {
  pool_.release_request_blocks(req, stream_);
  if (native_mtp())
    DGPP_CUDA_OK(
        cudaMemsetAsync(native_history_ + size_t(req) * kNativeHistoryRows * cfg_.hidden_size, 0,
                        draft_state_bytes(), stream_));
}

GlmSpecSegments MimoModel::spec_segments(int, int) const { return GlmSpecSegments{}; }

void MimoModel::write_state_snapshot(int, uint8_t* d, int) {
  DGPP_CUDA_OK(cudaMemsetAsync(d, 0, kSnapshotStamp, stream_));
}

// ---------------------------------------------------------------------------
// The boundary prefetch windows (bit-identical on or off).
// ---------------------------------------------------------------------------
void MimoModel::prefetch_fp8(const GlmQuantMatrix& m) {
  if (m.payload) prefetch_.add(m.payload, static_cast<size_t>(m.rows) * static_cast<size_t>(m.cols));
  if (m.scales) prefetch_.add(m.scales, m.scale_bytes());
}

// A bf16 matrix through its resident view: the packed companion's bytes
// when the GEMM holds one (its own allocation: isolated), else the
// matrix's own — which under bf12-only residency (common/bf16_residency.hpp)
// is a SIDE grant, its own allocation too, so it is isolated as well: the
// window's range merge bridges gaps of up to 2 MB between adds, and the
// gap between the globals image and the head's side grant is unmapped
// (compute-sanitizer 2026-09-22: the first-token draft after the warm
// prefill runs before the capture packs the companions, so the head's
// view is still the raw matrix).
void MimoModel::prefetch_bf16_view(const uint16_t* w, size_t bytes) {
  if (w == nullptr) return;
  const void* view = nullptr;
  size_t view_bytes = 0;
  gemm_.resident_view(w, bytes, walk_rows_, &view, &view_bytes);
  if (view != w || bf16_side_grants_)
    prefetch_.add_isolated(view, view_bytes);
  else
    prefetch_.add(view, view_bytes);
}

void MimoModel::prefetch_attn(const MimoAttnResident& a) {
  prefetch_fp8(a.qkv);
  if (a.sink) prefetch_.add(a.sink, static_cast<size_t>(a.local_heads) * 4);
  // The o_proj's bytes the rows' launches stream.
  prefetch_bf16_view(a.o_proj, static_cast<size_t>(cfg_.hidden_size) * static_cast<size_t>(a.local_heads) * cfg_.v_head_dim * 2);
}

// Before the attention fold: this layer's post norm, the router (and its
// bias) — or the dense MLP — one image. With the persisting set-aside,
// the next layer's input norm and fused projection follow in the same
// window: the fold and the norm / router behind it idle the DRAM, and
// the expert kernels' own ramps leave it slack; the persisting lines ride
// out the expert stream (the fold-2 window then carries only the o_proj).
void MimoModel::prefetch_ffn_side(const MimoLayerResident& r) {
  if (!prefetch_.enabled()) return;
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  const bool resident = loader_.residency() == MimoResidency::Resident;
  const bool early = l2_persist_bytes_ > 0 && resident;
  const bool early_plain = !early && early_qkv_bytes_ > 0 && resident;
  // The normal-priority early window: the router's bytes plus the budget.
  const size_t router_bytes = r.moe ? static_cast<size_t>(cfg_.n_routed_experts) * (H * 2 + 4) + H * 2 : prefetch_window_bytes_;
  prefetch_.open_window(stream_, early ? std::max(prefetch_window_bytes_, l2_persist_bytes_)
                                       : early_plain ? router_bytes + early_qkv_bytes_ : prefetch_window_bytes_,
                        prefetch_.boundary_rate(), early);
  if (r.post_norm) prefetch_.add(r.post_norm, H * 2);
  if (r.moe) {
    if (r.moe_w.router) prefetch_.add(r.moe_w.router, static_cast<size_t>(cfg_.n_routed_experts) * H * 2);
    if (r.moe_w.router_bias) prefetch_.add(r.moe_w.router_bias, static_cast<size_t>(cfg_.n_routed_experts) * 4);
  } else {
    prefetch_fp8(r.dense.gate);
    prefetch_fp8(r.dense.up);
    prefetch_fp8(r.dense.down);
  }
  if ((early || early_plain) && r.layer + 1 < cfg_.num_hidden_layers) {
    const MimoLayerResident& n = loader_.load_layer(r.layer + 1);
    if (n.input_norm) prefetch_.add(n.input_norm, H * 2);
    prefetch_fp8(n.attn.qkv);  // clamped to the window's budget
    if (n.attn.sink) prefetch_.add(n.attn.sink, static_cast<size_t>(n.attn.local_heads) * 4);
  }
}

// Before the MLP fold: the next layer's input norm and its projection (or
// the head) — with the early window, its o_proj alone (the projection is
// already persisting; the normal region's traffic keeps the o_proj lines
// only briefly, so its window stays a normal one). Resident stacks only
// (load_layer is a lookup there).
void MimoModel::prefetch_attention_side(int layer) {
  if (!prefetch_.enabled()) return;
  if (layer >= cfg_.num_hidden_layers) {
    prefetch_head();
    return;
  }
  if (loader_.residency() != MimoResidency::Resident) return;
  const MimoLayerResident& r = loader_.load_layer(layer);
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, prefetch_window_bytes_, prefetch_.boundary_rate());
  if (l2_persist_bytes_ > 0) {
    prefetch_bf16_view(r.attn.o_proj, H * static_cast<size_t>(r.attn.local_heads) * cfg_.v_head_dim * 2);
    return;
  }
  if (r.input_norm) prefetch_.add(r.input_norm, H * 2);
  prefetch_attn(r.attn);
}

void MimoModel::prefetch_head() {
  if (!prefetch_.enabled()) return;
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, prefetch_window_bytes_, prefetch_.boundary_rate());
  if (globals_.final_norm) prefetch_.add(globals_.final_norm, H * 2);
  prefetch_bf16_view(globals_.lm_head, static_cast<size_t>(lm_vocab_count_) * H * 2);
}

// ---------------------------------------------------------------------------
// The row walk.
// ---------------------------------------------------------------------------
// The boundary folds (plan D2): the producer writes its partial into the
// reducer's staged buffer when the shape fits (the collective sends
// straight from there; under capture the recorder's one stable buffer),
// else into `fallback`. The eager producer quiesces before the collective;
// under capture the fold is a recorded node and the stream order IS the
// drain.
uint16_t* MimoModel::stage(uint16_t* fallback, int T, int width, bool capture) {
  if (!boundary_) return fallback;
  uint16_t* s = boundary_->stage(T, width);
  if (s == nullptr && capture) throw std::runtime_error("run_rows: a capture fold does not fit the recorder's staged buffer");
  return s ? s : fallback;
}

void MimoModel::fold(uint16_t* buf, int T, int width, bool capture) {
  if (!boundary_) return;
  if (!capture) DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  boundary_->reduce(buf, T, width);
}

void MimoModel::add_pending(uint16_t* resid, const uint16_t* pending, int row0, int n) {
  if (pending == nullptr || n <= 0) return;
  const size_t off = static_cast<size_t>(row0) * cfg_.hidden_size;
  glm_residual_add_bf16(resid + off, pending + off, static_cast<int64_t>(n) * cfg_.hidden_size, stream_);
}

// The block boundaries run the fused add + two-rounding norm
// (kernels/add_rmsnorm.hpp, plan §7.1: one launch where the chain had
// two, the row register-resident; the arithmetic is the chain's).
const uint16_t* MimoModel::enqueue_layer(const MimoLayerResident& r, int pool_layer, uint16_t* resid, int T,
                                         const WalkRows& rows, const uint16_t* pending) {
  const int H = cfg_.hidden_size;
  const float eps = cfg_.rms_norm_eps;
  build_layer_objects(r);
  // ---- the attention block --------------------------------------------------
  add_rmsnorm_bf16(resid, pending, r.input_norm, x_, T, H, eps, stream_);
  uint16_t* attn_out = stage(y_, T, H, rows.capture);
  {
    MimoAttnRows arows;
    arows.req_ids = rows.req_ids;
    arows.pos = rows.pos;
    arows.decode = rows.decode;
    attn_->enqueue(x_, T, arows, pool_.view(pool_layer), attn_out, stream_);
  }
  if (rows.decode) prefetch_ffn_side(r);
  fold(attn_out, T, H, rows.capture);  // block boundary 1: o_proj's partial
  // ---- the MLP / MoE block ----------------------------------------------------
  add_rmsnorm_bf16(resid, attn_out, r.post_norm, x_, T, H, eps, stream_);
  uint16_t* ffn_out = stage(y_, T, H, rows.capture);
  if (r.moe) {
    // The routed chain alone (no shared expert): the fp32 sum in ascending
    // expert order, rounded once here (the Qwen MoE's contract with the
    // shared layer; GLM's enqueue_decode / enqueue_prefill fold their
    // shared expert into the chain and refuse without one).
    if (rows.decode)
      moe_->enqueue_decode_f32(x_, moe_acc_, T, rows.trace, stream_, rows.moe_table_slot);
    else
      moe_->enqueue_prefill_f32(x_, moe_acc_, T, rows.trace, stream_,
                                moe_->mma_takes_grid() ? MoeExpertKernel::kMma : MoeExpertKernel::kGemv);
    launch_moe_round_bf16(ffn_out, moe_acc_, static_cast<int64_t>(T) * H, stream_);
  } else {
    dense_->enqueue(x_, T, ffn_out, stream_);
  }
  if (rows.decode) prefetch_attention_side(r.layer + 1);
  fold(ffn_out, T, H, rows.capture);  // block boundary 2: the sliced down projections
  return ffn_out;                      // pending: the next norm adds it
}

MimoModel::Outputs MimoModel::run_rows(const RowRun& run) {
  const int T = run.T, req = run.req;
  if (run.capture && loader_.residency() != MimoResidency::Resident)
    throw std::logic_error("run_rows: a capture needs a resident stack");
  const int H = cfg_.hidden_size;
  // The packed companions' wide launches are the decode batch's alone (a
  // short prefill chunk keeps its Lt algorithm: kernels/gemm.hpp).
  gemm_.set_bf12_wide(run.decode);
  walk_rows_ = T;
  // A prefill walk gets the whole L2 back (the decode replays refill the
  // persisting set-aside).
  if (!run.decode && l2_persist_bytes_ > 0) prefetch_.release_persisting(stream_);
  const RowInputs in = begin_run(run);
  embed_gather_bf16(globals_.embed, in.tokens, resid_, T, H, stream_);
  Outputs out;
  // The route traces: the prefill rows' always, the decode rows' while the
  // eager engines keep them on (never under capture: no host copies in a
  // recorded step).
  const bool traces = route_traces_ && !run.capture && (!run.decode || decode_row_traces_);
  const size_t K = static_cast<size_t>(cfg_.num_experts_per_tok);
  const size_t E = static_cast<size_t>(cfg_.n_routed_experts);
  WalkRows rows;
  rows.req_ids = in.req_ids;
  rows.pos = in.pos;
  rows.req = req;
  rows.decode = run.decode;
  rows.capture = run.capture;
  const uint16_t* pending = nullptr;  // the block output the next norm adds
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const MimoLayerResident& r = loader_.load_layer(layer);
    MoeTraceStaging trace;
    rows.trace = nullptr;
    rows.moe_table_slot = -1;
    if (r.moe) {
      const int ordinal = moe_ordinal(layer);
      rows.moe_table_slot = run.capture ? ordinal : -1;
      if (traces) {
        const size_t slot = static_cast<size_t>(ordinal) * static_cast<size_t>(max_tokens_) * K;
        trace.ids = h_route_ids_ + slot;
        trace.weights = h_route_weights_ + slot;
        trace.biased = run.decode ? h_route_biased_ + static_cast<size_t>(ordinal) * static_cast<size_t>(max_decode_rows_) * E
                                  : nullptr;
        rows.trace = &trace;
        out.route_ids.emplace_back();  // filled after the sync
        out.route_weights.emplace_back();
      }
    }
    pending = enqueue_layer(r, layer, resid_, T, rows, pending);
    if (run.capture_layers) {
      add_pending(resid_, pending, 0, T);
      pending = nullptr;
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      std::vector<uint16_t> snap(static_cast<size_t>(T) * H);
      DGPP_CUDA_OK(cudaMemcpy(snap.data(), resid_, snap.size() * 2, cudaMemcpyDeviceToHost));
      out.layer_states.push_back(std::move(snap));
    }
  }
  // Normalize every row: MTP consumes the complete hidden-state chunk.
  // By default the head retains the diagnostic forward's m=T shape.
  add_rmsnorm_bf16(resid_, pending, globals_.final_norm, h_, T, H, cfg_.rms_norm_eps, stream_);
  // Keep all hidden rows for MTP, but project only the row consumed by a
  // scalar serving prefill. Diagnostics, grouped prefill and verification
  // retain every row and the common finish_run output layout is unchanged.
  const int head_first =
      prefill_last_head_ && !run.decode && !run.all_rows && run.num_spans == 0 ? T - 1 : 0;
  gemm_.matmul(h_ + static_cast<size_t>(head_first) * H, globals_.lm_head,
               logits_ + static_cast<size_t>(head_first) * lm_vocab_count_, T - head_first,
               lm_vocab_count_, H, DType::BF16, GemmOut::F32, static_cast<size_t>(H), gemm_ws_,
               gemm_ws_bytes_, stream_);
  // The draft block's input: the last rows' POST-final-norm hidden (the
  // model's output hidden state, vLLM's mimo_v2_mtp convention — the draft
  // applies hnorm to it) into the slots' windows by position (the last
  // window rows of a prefill chunk, every decode row — distinct slots
  // within one launch). A group prefill (several requests' spans in one
  // walk) stores every row: each span's last window rows land in its own
  // slot.
  if (mtp_) {
    const int n = run.num_spans > 0 ? T : std::min(T, max_decode_rows_);
    store_draft_hidden(h_ + static_cast<size_t>(T - n) * H, in.req_ids + (T - n), in.pos + (T - n), n);
  }
  if (run.decode) prefetch_.join(stream_);
  out = finish_run(run, std::move(out));
  if (!run.capture && traces) {
    for (size_t l = 0; l < out.route_ids.size(); ++l) {
      const size_t slot = l * static_cast<size_t>(max_tokens_) * K;
      out.route_ids[l].assign(h_route_ids_ + slot, h_route_ids_ + slot + static_cast<size_t>(T) * K);
      out.route_weights[l].assign(h_route_weights_ + slot, h_route_weights_ + slot + static_cast<size_t>(T) * K);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// The cold diagnostic forward: slot 0, fresh state, every row.
// ---------------------------------------------------------------------------
MimoModel::Outputs MimoModel::forward(const std::vector<int64_t>& token_ids, bool capture_layers) {
  const int T = static_cast<int>(token_ids.size());
  if (T <= 0) throw std::invalid_argument("forward: empty token batch");
  if (T > max_tokens_) throw std::invalid_argument("forward: tokens exceed max_tokens");
  if (T > max_context_) throw std::invalid_argument("forward: tokens exceed the context bound");
  for (int64_t id : token_ids)
    if (id < 0 || id >= cfg_.vocab_size) throw std::invalid_argument("forward: token id out of range");
  if (session_pos_[0] != 0) throw std::logic_error("forward: slot 0 holds an open session (close it first)");
  open_slot(0);
  if (!pool_.ensure_request_blocks(0, T, stream_)) throw std::runtime_error("forward: the cache pool cannot cover the batch");
  RowRun run;
  run.req = 0;
  run.ids = token_ids.data();
  run.T = T;
  run.pos0 = 0;
  run.decode = false;
  run.all_rows = true;
  run.capture_layers = capture_layers;
  Outputs out = run_rows(run);
  session_close(0);
  return out;
}

// ---------------------------------------------------------------------------
// The graph era.
// ---------------------------------------------------------------------------
void MimoModel::graph_prepare() {
  if (loader_.residency() != MimoResidency::Resident)
    throw std::logic_error("session_graph_prepare: the decode graph needs a resident stack");
  // The bf16 decode weights' 12-bit companions are packed as each layer
  // lands (the caches already exist: under bf12-only residency the layer's
  // bf16 bytes go back at once, so no more than one layer's sit beside
  // their companions).
  const bool pack = Bf12Companions::enabled() && !bf12_built_;
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const MimoLayerResident& r = loader_.load_layer(layer);
    build_layer_objects(r);
    if (r.moe) moe_->prepare_graph_table(moe_ordinal(layer), stream_);
    if (pack) pack_layer_companions(layer, r);
  }
  if (mtp_)
    for (int d = 0; d < cfg_.mtp_layers_loaded; ++d) {
      const int layer = cfg_.mtp_layer() + d;
      const MimoLayerResident& r = loader_.load_layer(layer);
      build_layer_objects(r);
      if (pack) pack_layer_companions(layer, r);
    }
  if (pack) finish_companions();
}

// The lossless 12-bit companions of the decode GEMV's bf16 weights
// (engine.bf16_weights = "bf12"): the o_proj of every layer, the head and
// the draft's eh_proj, packed as each layer lands and before any capture.
// Prefill and the batches past eight rows keep the bf16 bytes — or, under
// bf12-only residency (the bytes returned here, layer by layer), expand
// them and take the packed launches (kernels/gemm.hpp).
void MimoModel::pack_layer_companions(int layer, const MimoLayerResident& r) {
  const auto t0 = std::chrono::steady_clock::now();
  const int64_t H = cfg_.hidden_size;
  const auto release = [&](const void* w) { return loader_.release_packed(layer, w); };
  const auto pack = [&](const uint16_t* w, int64_t n, int64_t k) {
    bf12_.pack_and_release(w, n, k, gemm_, stream_, release);
  };
  const MimoAttnResident& a = r.attn;
  const int64_t O = static_cast<int64_t>(a.local_heads) * cfg_.v_head_dim;
  pack(a.o_proj, H, O);
  if (cfg_.is_mtp_layer(layer) && r.eh_proj != nullptr) pack(r.eh_proj, H, 2 * H);
  bf12_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// The lm head's, the expansion scratch (one slot: no walk of this family
// calls a matrix twice), and the summary.
void MimoModel::finish_companions() {
  const auto t0 = std::chrono::steady_clock::now();
  bf12_built_ = true;
  bf12_.pack_and_release(globals_.lm_head, lm_vocab_count_, cfg_.hidden_size, gemm_, stream_,
                         [&](const void* w) { return loader_.release_packed(-1, w); });
  bf12_.finish(gemm_, kBf12ExpandSlots);
  bf12_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  bf12_.log_summary(loader_.rank(), bf12_s_);
}

// ---------------------------------------------------------------------------
// The MTP draft block: eh_proj over [enorm(embed(tok_{q+1})) | hnorm(h_q)],
// one sliding-window layer with the dense MLP (its own pool layer),
// final_layernorm, the shared head.
// ---------------------------------------------------------------------------
void MimoModel::mtp_run_rows(int req, const int64_t* tokens, int64_t first_pos, int T, bool decode_row,
                             bool capture, int head_rows, int batch_requests) {
  if (!mtp_) throw std::logic_error("mtp_run_rows: MTP is not enabled");
  if (native_mtp()) {
    native_mtp_rows(req, tokens, first_pos, T, decode_row, capture, head_rows);
    return;
  }
  if (T <= 0 || T > max_tokens_) throw std::invalid_argument("mtp_run_rows: rows");
  gemm_.set_bf12_wide(decode_row);  // the decode batch's alone (kernels/gemm.hpp)
  walk_rows_ = T;
  if (head_rows < 0 || head_rows > T) throw std::invalid_argument("mtp_run_rows: head_rows");
  const int H = cfg_.hidden_size;
  const float eps = cfg_.rms_norm_eps;
  const int64_t* d_pos = decode_row ? d_step_pos_ : d_prefill_pos_;
  const int32_t* d_req = decode_row ? d_req_ids_ : d_prefill_req_;
  if (!decode_row) stage_prefill_meta(req, first_pos, T);
  const MimoLayerResident& r = loader_.load_layer(cfg_.mtp_layer());
  if (!r.enorm || !r.hnorm || !r.eh_proj || !r.final_norm)
    throw std::runtime_error("mtp_run_rows: the draft layer's head tensors are unbound");
  // ---- the input fusion --------------------------------------------------
  // Decode rows gather their hidden from the slots' windows by position;
  // prefill rows read the main chunk's rows in place (h_, the chunk's
  // post-final-norm hidden at rows 0 .. T-1; the head below overwrites
  // h_ only after the fusion has read it — stream order).
  const uint16_t* hin = h_;
  if (decode_row) {
    gather_draft_hidden(d_req, d_pos, mtp_h_, T);
    hin = mtp_h_;
  }
  glm_mtp_input_bf16(globals_.embed, tokens, hin, nullptr, 0, r.enorm, r.hnorm, mtp_in_, T, H, eps, stream_);
  gemm_.matmul(mtp_in_, r.eh_proj, mtp_r_, T, H, 2 * H, DType::BF16, GemmOut::BF16, static_cast<size_t>(2 * H),
               gemm_ws_, gemm_ws_bytes_, stream_);
  // ---- the draft layer (the stack's objects rebound to its weights) ------
  WalkRows rows;
  rows.req_ids = d_req;
  rows.pos = d_pos;
  rows.req = req;
  rows.decode = decode_row;
  rows.capture = capture;
  rows.moe_table_slot = -1;
  rows.trace = nullptr;
  (void)batch_requests;
  // No caller consumes the residual or logits of a history-only prefill.
  // Preserve the input fusion, normalization, projection and paged K/V
  // append, including FP8 scale planes. Decode chain rows still run fully.
  if (mtp_cache_only_ && head_rows == 0 && !decode_row) {
    build_layer_objects(r);
    add_rmsnorm_bf16(mtp_r_, nullptr, r.input_norm, x_, T, H, eps, stream_);
    MimoAttnRows arows;
    arows.req_ids = d_req;
    arows.pos = d_pos;
    arows.cache_only = true;
    attn_->enqueue(x_, T, arows, pool_.view(cfg_.num_hidden_layers), y_, stream_);
    return;
  }
  const uint16_t* pending = enqueue_layer(r, cfg_.num_hidden_layers, mtp_r_, T, rows, nullptr);
  // The draft's residual is the chain rows' input at depth >= 2
  // (draft_hidden_rows): every row completes — the head rows through the
  // fused add + final norm, the rest through the plain add.
  add_pending(mtp_r_, pending, 0, T - head_rows);
  if (head_rows == 0) {  // prefill rows fill the cache; no head
    if (decode_row) prefetch_.join(stream_);
    return;
  }
  // ---- head: the draft distribution over the last head_rows rows --------
  const size_t head_off = static_cast<size_t>(T - head_rows) * H;
  add_rmsnorm_bf16(mtp_r_ + head_off, pending + head_off, r.final_norm, h_, head_rows, H, eps, stream_);
  gemm_.matmul(h_, globals_.lm_head, logits_, head_rows, lm_vocab_count_, H, DType::BF16, GemmOut::F32,
               static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
  if (decode_row) prefetch_.join(stream_);
  if (decode_row && (!capture || decode_tail_mirrors_) && head_rows <= max_decode_rows_)
    DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_logits_, logits_, static_cast<size_t>(head_rows) * lm_vocab_count_ * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
}

// Native head d uses backbone hidden p-d and token p+1. The ordinary
// recursive path and all other model families retain their original contract.
void MimoModel::mtp_select_block(int index) {
  if (!native_mtp()) return;
  if (index < 0 || index >= cfg_.mtp_layers_loaded)
    throw std::invalid_argument("native MTP head index");
  draft_block_ = index;
}
void MimoModel::snapshot_draft_state(int req) {
  if (native_mtp())
    write_draft_snapshot(req, native_backup_ + size_t(req) * draft_state_bytes(), true, 0);
}
void MimoModel::restore_draft_state(int req) {
  if (native_mtp()) read_draft_snapshot(req, native_backup_ + size_t(req) * draft_state_bytes());
}
void MimoModel::write_draft_snapshot(int req, uint8_t* dst, bool live, int64_t pos) {
  if (!native_mtp()) return;
  const void* src =
      !live && mtp_pos_[req] > pos
          ? static_cast<const void*>(native_backup_ + size_t(req) * draft_state_bytes())
          : static_cast<const void*>(native_history_ +
                                     size_t(req) * kNativeHistoryRows * cfg_.hidden_size);
  glm_device_copy(dst, src, draft_state_bytes(), stream_);
}
void MimoModel::read_draft_snapshot(int req, const uint8_t* src) {
  if (!native_mtp()) return;
  glm_device_copy(native_history_ + size_t(req) * kNativeHistoryRows * cfg_.hidden_size, src,
                  draft_state_bytes(), stream_);
}
void MimoModel::native_mtp_rows(int req, const int64_t* tokens, int64_t first_pos, int T,
                                bool decode, bool capture, int head_rows) {
  if (T <= 0 || T > max_tokens_ || head_rows < 0 || head_rows > T)
    throw std::invalid_argument("native MTP rows");
  gemm_.set_bf12_wide(decode);
  walk_rows_ = T;
  const int H = cfg_.hidden_size;
  const float eps = cfg_.rms_norm_eps;
  if (!decode) stage_prefill_meta(req, first_pos, T);
  const int64_t* pos = decode ? d_step_pos_ : d_prefill_pos_;
  const int32_t* ids = decode ? d_req_ids_ : d_prefill_req_;
  if (draft_block_ == 0) {
    // Graph entry points snapshot before capture; eager drafting needs its
    // own pre-call history for session_draft_rollback and prefix cuts.
    if (decode && !capture) snapshot_draft_state(req);
    const uint16_t* hidden = h_;
    if (decode) {
      gather_draft_hidden(ids, pos, mtp_h_, T);
      hidden = mtp_h_;
    }
    mimo_mtp_history_store(native_history_, hidden, pos, ids, T, H, kNativeHistoryRows, 0, stream_);
  }
  for (int d = draft_block_; d < cfg_.mtp_layers_loaded; ++d) {
    mimo_mtp_history_gather(native_history_, mtp_h_, pos, native_positions_, ids, T, H,
                            kNativeHistoryRows, d, stream_);
    const int layer = cfg_.mtp_layer() + d;
    const auto& r = loader_.load_layer(layer);
    glm_mtp_input_bf16(globals_.embed, tokens, mtp_h_, nullptr, 0, r.enorm, r.hnorm, mtp_in_, T, H,
                       eps, stream_);
    gemm_.matmul(mtp_in_, r.eh_proj, mtp_r_, T, H, 2 * H, DType::BF16, GemmOut::BF16, size_t(2 * H),
                 gemm_ws_, gemm_ws_bytes_, stream_);
    // Deeper heads only need K/V until selected. No head consumes another
    // head's residual. Padded negative positions never append K/V.
    const bool history_only = head_rows == 0 || d != draft_block_;
    WalkRows rows;
    rows.req_ids = ids;
    rows.pos = native_positions_;
    rows.req = req;
    rows.decode = decode;
    rows.capture = capture;
    if (history_only) {
      build_layer_objects(r);
      add_rmsnorm_bf16(mtp_r_, nullptr, r.input_norm, x_, T, H, eps, stream_);
      MimoAttnRows ar;
      ar.req_ids = ids;
      ar.pos = native_positions_;
      ar.cache_only = true;
      ar.decode = decode;
      attn_->enqueue(x_, T, ar, pool_.view(layer), y_, stream_);
    } else {
      const uint16_t* pending = enqueue_layer(r, layer, mtp_r_, T, rows, nullptr);
      const size_t off = size_t(T - head_rows) * H;
      add_rmsnorm_bf16(mtp_r_ + off, pending + off, r.final_norm, native_output_, head_rows, H, eps,
                       stream_);
    }
  }
  if (head_rows) {
    gemm_.matmul(native_output_, globals_.lm_head, logits_, head_rows, lm_vocab_count_, H,
                 DType::BF16, GemmOut::F32, size_t(H), gemm_ws_, gemm_ws_bytes_, stream_);
    // Keep the diagnostic mtp_forward hidden-output surface valid.
    glm_device_copy(h_, native_output_, size_t(head_rows) * H * 2, stream_);
  }
  if (decode) prefetch_.join(stream_);
  if (decode && head_rows && (!capture || decode_tail_mirrors_))
    DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_logits_, logits_,
                                 size_t(head_rows) * lm_vocab_count_ * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
}

MimoModel::Outputs MimoModel::mtp_forward(const std::vector<int64_t>& token_ids) {
  if (!mtp_) refuse_mtp("mtp_forward");
  const int T = static_cast<int>(token_ids.size());
  if (T < 2) throw std::invalid_argument("mtp_forward: at least two tokens");
  if (T > max_tokens_) throw std::invalid_argument("mtp_forward: tokens exceed max_tokens");
  for (int64_t id : token_ids)
    if (id < 0 || id >= cfg_.vocab_size) throw std::invalid_argument("mtp_forward: token id out of range");
  if (session_pos_[0] != 0) throw std::logic_error("mtp_forward: slot 0 holds an open session");
  open_slot(0);
  if (!pool_.ensure_request_blocks(0, T, stream_)) throw std::runtime_error("mtp_forward: the cache pool cannot cover the batch");
  RowRun run;
  run.req = 0;
  run.ids = token_ids.data();
  run.T = T;
  run.decode = false;
  run.all_rows = true;
  (void)run_rows(run);
  const int rows = T - 1;
  DGPP_CUDA_OK(cudaMemcpyAsync(d_tokens_, token_ids.data() + 1, static_cast<size_t>(rows) * 8, cudaMemcpyHostToDevice,
                               stream_));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  mtp_select_block(0);
  mtp_run_rows(0, d_tokens_, 0, rows, /*decode_row=*/false, /*capture=*/false, /*head_rows=*/rows, 0);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  const int H = cfg_.hidden_size;
  Outputs out;
  out.lm_vocab_begin = lm_vocab_begin_;
  out.lm_vocab_count = lm_vocab_count_;
  out.final_hidden_bits.resize(static_cast<size_t>(rows) * H);
  out.logits.resize(static_cast<size_t>(rows) * lm_vocab_count_);
  DGPP_CUDA_OK(cudaMemcpy(out.final_hidden_bits.data(), h_, out.final_hidden_bits.size() * 2, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(out.logits.data(), logits_, out.logits.size() * 4, cudaMemcpyDeviceToHost));
  session_close(0);
  return out;
}

}  // namespace dgpp
