#include "models/flint/forward.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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

namespace dgpp {
namespace {

constexpr int kQk = 128;

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

// The MoE layers' ordinals (the graph table slots and the trace slots).
std::vector<int> moe_ordinals(const FlintTextConfig& cfg) {
  std::vector<int> o(static_cast<size_t>(cfg.num_hidden_layers), -1);
  int n = 0;
  for (int l = 0; l < cfg.num_hidden_layers; ++l)
    if (cfg.is_moe_layer(l)) o[static_cast<size_t>(l)] = n++;
  return o;
}

MimoKvPoolShape pool_shape(const FlintTextConfig& cfg, int local_kv_heads, int max_requests, int64_t cache_tokens,
                           LatentFormat kv_format) {
  MimoKvPoolShape shape;
  shape.kv_heads.assign(static_cast<size_t>(cfg.num_hidden_layers), local_kv_heads);
  shape.k_dim = kQk;
  shape.v_dim = kQk;
  shape.block_tokens = 64;
  shape.max_requests = max_requests;
  shape.token_slots = cache_tokens;
  shape.format = kv_format;
  return shape;
}

}  // namespace

FlintModel::FlintModel(const FlintTextConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
                       int64_t max_cache_tokens, FlintResidency residency, BoundaryReducer* boundary, int tp_rank,
                       int tp_world, int max_requests, bool mtp, int decode_rows, LatentFormat kv_format,
                       bool serving_logits)
    : cfg_(cfg),
      loader_(cfg, checkpoint_dir, tp_rank, tp_world, residency,
              tp_world > 1 ? FlintHeadSharding::VocabSharded : FlintHeadSharding::Full),
      moe_ordinal_(moe_ordinals(cfg)) {
  if (mtp) throw std::invalid_argument("FlintModel: the family has no draft block (engine.mtp must be off)");
  if (decode_rows > decode_rows_cap())
    throw std::invalid_argument("FlintModel: decode_rows exceeds the family's limit of 32");
  if (max_tokens <= 0) throw std::invalid_argument("FlintModel: max_tokens must be positive");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("FlintModel: max_requests must be in [1, kPickMaxRequests]");
  if ((tp_world > 1) != (boundary != nullptr))
    throw std::invalid_argument(
        "FlintModel: a boundary reducer is required exactly when tp_world > 1 — without one the "
        "block-boundary partials would be returned silently as results");
  if (cfg_.eos_token_ids.empty()) throw std::invalid_argument("FlintModel: the config names no EOS token");
  if (cfg_.head_dim != kQk) throw std::invalid_argument("FlintModel: 128-wide heads (the kernels' shape)");
  if (kv_format != LatentFormat::kBf16 && kv_format != LatentFormat::kFp8)
    throw std::invalid_argument("FlintModel: the K/V cache format must be bf16 or fp8");
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
    sp.logits_rows = compact_logits_rows(serving_logits, decode_rows, max_requests);
    sp.mtp = false;
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
  gw_ = FlintGemmWorkspace{&gemm_, gemm_ws_, gemm_ws_bytes_};
  // The bf16 sites' lowering (kernels/gemm.hpp dense_gemv_rows): the GEMV
  // chunks (the 12-bit companions) to the bound, cuBLASLt's algorithm
  // above it.
  gemm_.set_decode_rows(std::min(max_decode_rows_, dense_gemv_rows()));
  bf16_side_grants_ = residency == FlintResidency::Resident && bf16_side_grants();
  moe_cfg_ = cfg_.moe_config(static_cast<int>(loader_.geometry().local_inter));

  const FlintLocalGeometry& geo = loader_.geometry();
  pool_.init(pool_shape(cfg_, geo.local_kv_heads, max_requests_, max_cache_tokens_, kv_format));
  const size_t M = static_cast<size_t>(max_tokens_);
  resid_ = dev_alloc<uint16_t>(M * H);
  x_ = dev_alloc<uint16_t>(M * H);
  y_ = dev_alloc<uint16_t>(M * H);
  {
    const size_t layers = static_cast<size_t>(cfg_.num_moe_layers());
    const size_t K = static_cast<size_t>(cfg_.num_experts_per_tok);
    h_route_ids_ = pinned_alloc<int32_t>(layers * M * K);
    h_route_weights_ = pinned_alloc<float>(layers * M * K);
    h_route_biased_ = pinned_alloc<float>(layers * static_cast<size_t>(max_decode_rows_) *
                                          static_cast<size_t>(cfg_.n_routed_experts));
    // The boundary windows' budget (engine.l2_prefetch_boundary_window_mib).
    prefetch_window_bytes_ = l2_prefetch_settings().boundary_window_bytes;
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}

FlintModel::~FlintModel() {
  cudaFree(resid_);
  cudaFree(x_);
  cudaFree(y_);
  cudaFreeHost(h_route_ids_);
  cudaFreeHost(h_route_weights_);
  cudaFreeHost(h_route_biased_);
  cudaFree(gemm_ws_);
}

int FlintModel::table_slots() const {
  return loader_.residency() == FlintResidency::Resident ? cfg_.num_moe_layers() : 0;
}

size_t FlintModel::kv_block_bytes_static(const FlintTextConfig& cfg, int tp_rank, int tp_world,
                                         LatentFormat kv_format) {
  const FlintHeadSharding head = tp_world > 1 ? FlintHeadSharding::VocabSharded : FlintHeadSharding::Full;
  const FlintLocalGeometry geo = FlintLocalGeometry::from_config(cfg, tp_rank, tp_world, head);
  const auto bytes = [&](int64_t tokens) {
    const MimoKvPoolShape shape = pool_shape(cfg, geo.local_kv_heads, 1, tokens, kv_format);
    return MimoKvPool::cache_bytes(shape) - PagedBlockTable::table_bytes(1, tokens / kBlockTokens);
  };
  return bytes(2 * kBlockTokens) - bytes(kBlockTokens);
}

FlintModel::MemoryPlan FlintModel::plan_memory(const FlintTextConfig& cfg, int max_tokens, int64_t max_cache_tokens,
                                               int tp_rank, int tp_world, FlintResidency residency,
                                               int max_requests, bool mtp, int decode_rows, LatentFormat kv_format,
                                               bool serving_logits) {
  if (mtp) throw std::invalid_argument("plan_memory: the Flint family has no draft block (engine.mtp must be off)");
  if (max_tokens <= 0) throw std::invalid_argument("plan_memory: max_tokens must be positive");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("plan_memory: max_requests must be in [1, kPickMaxRequests]");
  if (decode_rows > decode_rows_cap())
    throw std::invalid_argument("plan_memory: decode_rows exceeds the family's limit of 32");
  // The fixed batch's row ceiling, floored as the session core floors it.
  const int rows = std::max({kDecodeRows, decode_rows, max_requests});
  const FlintHeadSharding head = tp_world > 1 ? FlintHeadSharding::VocabSharded : FlintHeadSharding::Full;
  const FlintLocalGeometry geo = FlintLocalGeometry::from_config(cfg, tp_rank, tp_world, head);
  const int64_t cache_tokens =
      ((std::max<int64_t>(max_cache_tokens, max_tokens) + kBlockTokens - 1) / kBlockTokens) * kBlockTokens;
  MemoryPlan plan;
  plan.context_tokens = std::min<int64_t>(cache_tokens, cfg.max_position_embeddings);
  const size_t M = static_cast<size_t>(max_tokens);
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t V = static_cast<size_t>(FlintLayerStream::lm_vocab_count(cfg, tp_rank, tp_world, head));
  const int n_split = FlintAttentionLayer::default_decode_splits();

  // bf12-only residency: the packable bf16 matrices load aside and give
  // their bytes back as each layer is packed (graph_prepare).
  const bool bf12_only = Bf12Companions::packed_only() && residency == FlintResidency::Resident;
  if (residency == FlintResidency::Resident) {
    size_t resident = FlintLayerStream::resident_bytes(cfg, tp_rank, tp_world, head, false);
    if (bf12_only) resident -= FlintLayerStream::side_bytes(cfg, tp_rank, tp_world, head, false);
    plan.add(bf12_only ? "model weights (resident, packed bf16 matrices released)" : "model weights (resident)",
             resident);
    plan.add("loader staging (pinned host, freed when the last layer is resident)", 0,
             FlintLayerStream::staging_plan_bytes(cfg, tp_rank, tp_world, head, false));
  } else {
    size_t largest = 0;
    for (int l = 0; l < cfg.num_hidden_layers; ++l)
      largest = std::max(largest, FlintLayerStream::layer_bytes(cfg, l, tp_rank, tp_world));
    plan.add("model weights (one streamed layer + globals)",
             largest + FlintLayerStream::globals_bytes(cfg, tp_rank, tp_world, head));
  }
  const int64_t Hh = cfg.hidden_size;
  const int64_t qkv_rows = static_cast<int64_t>(geo.local_heads + 2 * geo.local_kv_heads) * cfg.head_dim;
  const int64_t O = static_cast<int64_t>(geo.local_heads) * cfg.head_dim;
  const int64_t Id = geo.local_dense_inter;
  if (Bf12Companions::enabled() && residency == FlintResidency::Resident) {
    // The lossless 12-bit companions of the fused projection and o_proj of
    // every layer, the dense layers' three matrices and the head.
    const int dense_layers = cfg.num_hidden_layers - cfg.num_moe_layers();
    size_t packed = static_cast<size_t>(cfg.num_hidden_layers) *
                        (Bf12Companions::planned_bytes(qkv_rows, Hh) + Bf12Companions::planned_bytes(Hh, O)) +
                    static_cast<size_t>(dense_layers) *
                        (2 * Bf12Companions::planned_bytes(Id, Hh) + Bf12Companions::planned_bytes(Hh, Id)) +
                    Bf12Companions::planned_bytes(static_cast<int64_t>(V), Hh);
    plan.add("bf16 decode packing (12-bit companions)", packed);
    if (bf12_only) {
      const size_t h = static_cast<size_t>(Hh);
      plan.add("bf16 prefill expansion scratch",
               static_cast<size_t>(kBf12ExpandSlots) *
                   Bf12Companions::planned_slot_bytes({static_cast<size_t>(qkv_rows) * h * 2, h * static_cast<size_t>(O) * 2,
                                                       static_cast<size_t>(Id) * h * 2, V * h * 2}));
    }
  }
  plan.add("gemm workspace (at least)", size_t{64} << 20);
  {
    const MimoKvPoolShape shape = pool_shape(cfg, geo.local_kv_heads, max_requests, cache_tokens, kv_format);
    plan.add(kv_format == LatentFormat::kFp8 ? "kv cache pool (K/V fp8 rows + scales, every layer)"
                                             : "kv cache pool (K/V bf16, every layer)",
             MimoKvPool::cache_bytes(shape));
  }
  {
    size_t core_dev = 0, core_pin = 0;
    session_core_plan_bytes(max_tokens, max_requests, cfg.hidden_size, static_cast<int>(V), false,
                            cfg.hidden_size, &core_dev, &core_pin, rows, serving_logits);
    const size_t moe_layers = static_cast<size_t>(cfg.num_moe_layers());
    const size_t K = static_cast<size_t>(cfg.num_experts_per_tok);
    const size_t E = static_cast<size_t>(cfg.n_routed_experts);
    plan.add("activations (session core, residual, block io, route staging)", core_dev + 3 * M * H * 2,
             core_pin + moe_layers * M * K * 8 + moe_layers * static_cast<size_t>(rows) * E * 4);
  }
  {
    size_t layers = FlintAttentionLayer::scratch_bytes(cfg, geo.local_heads, geo.local_kv_heads, max_tokens, rows,
                                                       n_split);
    layers += FlintBf16Mlp::scratch_bytes(geo.local_dense_inter, max_tokens);
    plan.add("layer objects (attention, dense MLP)", layers);
    const GlmMoeConfig moe_cfg = cfg.moe_config(static_cast<int>(geo.local_inter));
    const int slots = residency == FlintResidency::Resident ? cfg.num_moe_layers() : 0;
    size_t moe_pinned = 0;
    const size_t moe_dev =
        QwenMoeLayer::scratch_bytes(moe_cfg, geo.local_shared_inter, max_tokens, &moe_pinned, rows, slots);
    plan.add("moe scratch (routed slots, graph tables, the shared expert's tail)", moe_dev, moe_pinned);
  }
  return plan;
}

QwenMoeWeights FlintModel::moe_view(const FlintMoeResident& m) {
  QwenMoeWeights w;
  w.router = m.router;
  w.router_bias = m.router_bias;
  w.shared_gate = nullptr;  // the ungated shared expert: weight 1
  w.shared_gate_proj = m.shared.gate;
  w.shared_up_proj = m.shared.up;
  w.shared_down_proj = m.shared.down;
  w.shared_inter = m.shared.local_inter;
  // The layer's per-projection formats: a uniform layer binds one table,
  // a mixed one both with the mask (models/glm/moe.hpp).
  const uint8_t mask = m.fp4_mask;
  w.experts = mask == 7 ? nullptr : m.experts_fp8.data();
  w.experts_fp4 = mask == 0 ? nullptr : m.experts_fp4.data();
  w.routed_fp4_mask = (mask == 0 || mask == 7) ? 0 : mask;
  return w;
}

void FlintModel::build_layer_objects(const FlintLayerResident& r) {
  if (!attn_) {
    attn_ = std::make_unique<FlintAttentionLayer>(r.attn, gw_, cfg_, max_tokens_, max_decode_rows_, n_split_);
  } else {
    attn_->rebind(r.attn);
  }
  if (r.moe) {
    if (!moe_) {
      moe_ = std::make_unique<QwenMoeLayer>(moe_view(r.moe_w), moe_cfg_, gemm_, max_tokens_, max_decode_rows_,
                                            table_slots());
    } else {
      moe_->rebind(moe_view(r.moe_w));
    }
  } else {
    if (!dense_) {
      dense_ = std::make_unique<FlintBf16Mlp>(r.dense, gw_, cfg_, max_tokens_);
    } else {
      dense_->rebind(r.dense);
    }
  }
}

// The state every row walk starts from on a fresh request: no blocks
// (the K/V rows are written before any row reads them).
void FlintModel::reset_slot_state(int req) { pool_.release_request_blocks(req, stream_); }

void FlintModel::write_state_snapshot(int, uint8_t* d, int) {
  DGPP_CUDA_OK(cudaMemsetAsync(d, 0, kSnapshotStamp, stream_));
}

// ---------------------------------------------------------------------------
// The boundary prefetch windows (bit-identical on or off).
// ---------------------------------------------------------------------------
// A bf16 matrix through its resident view: the packed companion's bytes
// when the GEMM holds one (its own allocation: isolated), else the
// matrix's own — which under bf12-only residency is a SIDE grant, its own
// allocation too.
void FlintModel::prefetch_bf16_view(const uint16_t* w, size_t bytes) {
  if (w == nullptr) return;
  const void* view = nullptr;
  size_t view_bytes = 0;
  gemm_.resident_view(w, bytes, walk_rows_, &view, &view_bytes);
  if (view != w || bf16_side_grants_)
    prefetch_.add_isolated(view, view_bytes);
  else
    prefetch_.add(view, view_bytes);
}

// A raw bf16 matrix inside the layer image (never packed).
void FlintModel::prefetch_bf16_raw(const uint16_t* w, size_t bytes) {
  if (w != nullptr) prefetch_.add(w, bytes);
}

void FlintModel::prefetch_attn(const FlintAttnResident& a) {
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_bf16_view(a.qkv, static_cast<size_t>(a.qkv_rows()) * H * 2);
  prefetch_bf16_raw(a.q_norm, kQk * 2);
  prefetch_bf16_raw(a.k_norm, kQk * 2);
  if (a.sink) prefetch_.add(a.sink, static_cast<size_t>(a.local_heads) * 4);
  prefetch_bf16_view(a.o_proj, H * static_cast<size_t>(a.local_heads) * kQk * 2);
}

// Before the attention fold: this layer's post norm and the router (its
// zero bias) with the shared expert's matrices — or the dense MLP's.
void FlintModel::prefetch_ffn_side(const FlintLayerResident& r) {
  if (!prefetch_.enabled()) return;
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, prefetch_window_bytes_, prefetch_.boundary_rate());
  if (r.post_norm) prefetch_.add(r.post_norm, H * 2);
  if (r.moe) {
    const FlintMoeResident& m = r.moe_w;
    if (m.router) prefetch_.add(m.router, static_cast<size_t>(cfg_.n_routed_experts) * H * 2);
    if (m.router_bias) prefetch_.add(m.router_bias, static_cast<size_t>(cfg_.n_routed_experts) * 4);
    const size_t S = static_cast<size_t>(m.shared.local_inter);
    prefetch_bf16_raw(m.shared.gate, S * H * 2);
    prefetch_bf16_raw(m.shared.up, S * H * 2);
    prefetch_bf16_raw(m.shared.down, H * S * 2);
  } else {
    const size_t I = static_cast<size_t>(r.dense.local_inter);
    prefetch_bf16_view(r.dense.gate, I * H * 2);
    prefetch_bf16_view(r.dense.up, I * H * 2);
    prefetch_bf16_view(r.dense.down, H * I * 2);
  }
}

// Before the MLP fold: the next layer's input norm and its attention
// matrices (or the head). Resident stacks only (load_layer is a lookup).
void FlintModel::prefetch_attention_side(int layer) {
  if (!prefetch_.enabled()) return;
  if (layer >= cfg_.num_hidden_layers) {
    prefetch_head();
    return;
  }
  if (loader_.residency() != FlintResidency::Resident) return;
  const FlintLayerResident& r = loader_.load_layer(layer);
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, prefetch_window_bytes_, prefetch_.boundary_rate());
  if (r.input_norm) prefetch_.add(r.input_norm, H * 2);
  prefetch_attn(r.attn);
}

void FlintModel::prefetch_head() {
  if (!prefetch_.enabled()) return;
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, prefetch_window_bytes_, prefetch_.boundary_rate());
  if (globals_.final_norm) prefetch_.add(globals_.final_norm, H * 2);
  prefetch_bf16_view(globals_.lm_head, static_cast<size_t>(lm_vocab_count_) * H * 2);
}

// ---------------------------------------------------------------------------
// The row walk.
// ---------------------------------------------------------------------------
uint16_t* FlintModel::stage(uint16_t* fallback, int T, int width, bool capture) {
  if (!boundary_) return fallback;
  uint16_t* s = boundary_->stage(T, width);
  if (s == nullptr && capture) throw std::runtime_error("run_rows: a capture fold does not fit the recorder's staged buffer");
  return s ? s : fallback;
}

void FlintModel::fold(uint16_t* buf, int T, int width, bool capture) {
  if (!boundary_) return;
  if (!capture) DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  boundary_->reduce(buf, T, width);
}

void FlintModel::add_pending(uint16_t* resid, const uint16_t* pending, int row0, int n) {
  if (pending == nullptr || n <= 0) return;
  const size_t off = static_cast<size_t>(row0) * cfg_.hidden_size;
  glm_residual_add_bf16(resid + off, pending + off, static_cast<int64_t>(n) * cfg_.hidden_size, stream_);
}

// The block boundaries run the fused add + one-rounding norm
// (kernels/add_rmsnorm.hpp, D4).
const uint16_t* FlintModel::enqueue_layer(const FlintLayerResident& r, int pool_layer, uint16_t* resid, int T,
                                          const WalkRows& rows, const uint16_t* pending) {
  const int H = cfg_.hidden_size;
  const float eps = cfg_.rms_norm_eps;
  build_layer_objects(r);
  const auto groups = [&](bool combine_readers, auto&& enqueue) {
    if (rows.run)
      rows.run->for_each_kernel_group(
          [&](int count, int next) { return combine_readers || (count > 128 && next > 128); }, enqueue);
    else
      enqueue(0, T, rows.decode, 0);
  };
  // ---- the attention block --------------------------------------------------
  add_rmsnorm_bf16_one(resid, pending, r.input_norm, x_, T, H, eps, stream_);
  uint16_t* attn_out = stage(y_, T, H, rows.capture);
  groups(false, [&](int row0, int count, bool decode, int) {
    FlintAttnRows arows;
    arows.req_ids = rows.req_ids + row0;
    arows.pos = rows.pos + row0;
    arows.decode = decode;
    gemm_.set_bf12_wide(decode);
    const size_t offset = static_cast<size_t>(row0) * H;
    attn_->enqueue(x_ + offset, count, arows, pool_.view(pool_layer), attn_out + offset, stream_);
  });
  gemm_.set_bf12_wide(rows.decode);
  if (rows.decode) prefetch_ffn_side(r);
  fold(attn_out, T, H, rows.capture);  // block boundary 1: o_proj's partial
  // ---- the MLP / MoE block ----------------------------------------------------
  add_rmsnorm_bf16_one(resid, attn_out, r.post_norm, x_, T, H, eps, stream_);
  uint16_t* ffn_out = stage(y_, T, H, rows.capture);
  groups(r.moe, [&](int row0, int count, bool decode, int) {
    const size_t offset = static_cast<size_t>(row0) * H;
    gemm_.set_bf12_wide(decode);
    if (r.moe) {
      MoeTraceStaging trace;
      if (rows.trace) {
        trace.ids = rows.trace->ids + static_cast<size_t>(row0) * cfg_.num_experts_per_tok;
        trace.weights = rows.trace->weights + static_cast<size_t>(row0) * cfg_.num_experts_per_tok;
        trace.biased = rows.trace->biased ? rows.trace->biased + static_cast<size_t>(row0) * cfg_.n_routed_experts
                                          : nullptr;
      }
      // The MoE layer's tail (the ungated shared expert, the one rounding)
      // writes the block output directly.
      if (decode)
        moe_->enqueue_decode(x_ + offset, ffn_out + offset, count, stream_, rows.moe_table_slot,
                             rows.trace ? &trace : nullptr);
      else
        moe_->enqueue_prefill(x_ + offset, ffn_out + offset, count, stream_, rows.trace ? &trace : nullptr);
    } else {
      dense_->enqueue(x_ + offset, count, ffn_out + offset, stream_);
    }
  });
  gemm_.set_bf12_wide(rows.decode);
  if (rows.decode) prefetch_attention_side(r.layer + 1);
  fold(ffn_out, T, H, rows.capture);  // block boundary 2: the sliced down projections
  return ffn_out;                      // pending: the next norm adds it
}

FlintModel::Outputs FlintModel::run_rows(const RowRun& run) {
  const int T = run.T, req = run.req;
  if (run.capture && loader_.residency() != FlintResidency::Resident)
    throw std::logic_error("run_rows: a capture needs a resident stack");
  const int H = cfg_.hidden_size;
  // The packed companions' wide launches are the decode batch's alone (a
  // short prefill chunk keeps its Lt algorithm: kernels/gemm.hpp).
  gemm_.set_bf12_wide(run.decode);
  walk_rows_ = T;
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
  rows.run = &run;
  rows.req_ids = in.req_ids;
  rows.pos = in.pos;
  rows.req = req;
  rows.decode = run.decode;
  rows.capture = run.capture;
  const uint16_t* pending = nullptr;  // the block output the next norm adds
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const FlintLayerResident& r = loader_.load_layer(layer);
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
  // The final norm over every row (the diagnostic forward keeps the full
  // head shape; serving packs the prefill tails).
  add_rmsnorm_bf16_one(resid_, pending, globals_.final_norm, h_, T, H, cfg_.rms_norm_eps, stream_);
  const bool packed_logits = project_head_rows(run, [&](int input_row, int output_row, int n) {
    gemm_.set_bf12_wide(run.decode || input_row < run.decode_rows());
    gemm_.matmul(h_ + static_cast<size_t>(input_row) * H, globals_.lm_head,
                 logits_ + static_cast<size_t>(output_row) * lm_vocab_count_, n, lm_vocab_count_, H, DType::BF16,
                 GemmOut::F32, static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
  });
  if (run.decode) prefetch_.join(stream_);
  out = finish_run(run, std::move(out), packed_logits);
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
FlintModel::Outputs FlintModel::forward(const std::vector<int64_t>& token_ids, bool capture_layers) {
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
void FlintModel::graph_prepare() {
  if (loader_.residency() != FlintResidency::Resident)
    throw std::logic_error("session_graph_prepare: the decode graph needs a resident stack");
  // The bf16 decode weights' 12-bit companions are packed as each layer
  // lands (under bf12-only residency the layer's bf16 bytes go back at
  // once, so no more than one layer's sit beside their companions).
  const bool pack = Bf12Companions::enabled() && !bf12_built_;
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const FlintLayerResident& r = loader_.load_layer(layer);
    build_layer_objects(r);
    if (r.moe) moe_->prepare_graph_table(moe_ordinal(layer), stream_);
    if (pack) pack_layer_companions(layer, r);
  }
  if (pack) finish_companions();
}

// The lossless 12-bit companions of the decode GEMV's bf16 weights
// (engine.bf16_weights = "bf12"): the fused projection and o_proj of every
// layer, the dense layers' MLP, the head — the matrices the GEMM interface
// serves. The shared expert's matrices stay raw (the MoE tail's kernels).
void FlintModel::pack_layer_companions(int layer, const FlintLayerResident& r) {
  const auto t0 = std::chrono::steady_clock::now();
  const int64_t H = cfg_.hidden_size;
  const auto release = [&](const void* w) { return loader_.release_packed(layer, w); };
  const auto pack = [&](const uint16_t* w, int64_t n, int64_t k) {
    if (w != nullptr) bf12_.pack_and_release(w, n, k, gemm_, stream_, release);
  };
  const FlintAttnResident& a = r.attn;
  pack(a.qkv, a.qkv_rows(), H);
  pack(a.o_proj, H, static_cast<int64_t>(a.local_heads) * kQk);
  if (!r.moe) {
    pack(r.dense.gate, r.dense.local_inter, H);
    pack(r.dense.up, r.dense.local_inter, H);
    pack(r.dense.down, H, r.dense.local_inter);
  }
  bf12_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void FlintModel::finish_companions() {
  const auto t0 = std::chrono::steady_clock::now();
  bf12_built_ = true;
  bf12_.pack_and_release(globals_.lm_head, lm_vocab_count_, cfg_.hidden_size, gemm_, stream_,
                         [&](const void* w) { return loader_.release_packed(-1, w); });
  bf12_.finish(gemm_, kBf12ExpandSlots);
  bf12_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  bf12_.log_summary(loader_.rank(), bf12_s_);
}

}  // namespace dgpp
