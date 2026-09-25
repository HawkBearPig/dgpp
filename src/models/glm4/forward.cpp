#include "models/glm4/forward.hpp"

#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "kernels/bf12_companions.hpp"
#include "kernels/glm4_attn.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/kernels.hpp"

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

}  // namespace

Glm4Model::Glm4Model(const Glm4TextConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
                     int64_t max_cache_tokens, Glm4Residency residency, BoundaryReducer* boundary,
                     int tp_rank, int tp_world, int max_requests, bool mtp, int decode_rows)
    : cfg_(cfg),
      loader_(cfg, checkpoint_dir, tp_rank, tp_world, residency,
              tp_world > 1 ? Glm4HeadSharding::VocabSharded : Glm4HeadSharding::Full,
              mtp && residency == Glm4Residency::Resident) {
  if (decode_rows > decode_rows_cap())
    throw std::invalid_argument("Glm4Model: decode_rows exceeds the GLM-4.7 limit of 32");
  if (max_tokens <= 0) throw std::invalid_argument("Glm4Model: max_tokens must be positive");
  if (mtp && cfg_.mtp_layer() < 0) throw std::invalid_argument("Glm4Model: the config has no draft layer (mtp)");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("Glm4Model: max_requests must be in [1, kPickMaxRequests]");
  if ((tp_world > 1) != (boundary != nullptr))
    throw std::invalid_argument(
        "Glm4Model: a boundary reducer is required exactly when tp_world > 1 — without one the "
        "block-boundary partials would be returned silently as results");
  if (cfg_.eos_token_ids.empty()) throw std::invalid_argument("Glm4Model: the config names no EOS token");
  if (cfg_.head_dim != kGlm4HeadDim) throw std::invalid_argument("Glm4Model: 128-wide heads (the kernels' shape)");
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
    sp.snapshot_align = 1;  // plan D5: no recurrent state, any position snapshots
    sp.draft_width = H;
    sp.eos = static_cast<int32_t>(cfg_.eos_token_ids[0]);
    init_session(sp);
  }
  gemm_ws_bytes_ = std::max<size_t>(64u << 20, gemm_.query_workspace_bytes(max_tokens_, lm_vocab_count_, H, DType::BF16));
  gemm_ws_ = dev_alloc<char>(gemm_ws_bytes_);
  gw_ = Glm4GemmWorkspace{&gemm_, gemm_ws_, gemm_ws_bytes_};
  // The attention projections' lowering (kernels/gemm.hpp dense_gemv_rows):
  // the GEMV chunks to the bound, cuBLASLt's algorithm above it (the
  // chunks re-read the BF16 attention weights per four rows: the batched
  // step's bound before 2026-09-14).
  gemm_.set_decode_rows(std::min(max_decode_rows_, dense_gemv_rows()));
  moe_cfg_ = cfg_.moe_config(static_cast<int>(loader_.geometry().local_inter));
  n_split_ = Glm4AttentionLayer::default_decode_splits();

  const Glm4LocalGeometry& geo = loader_.geometry();
  {
    Glm4KvPoolShape shape;
    shape.layers = pool_layers();
    shape.kv_heads = geo.local_kv_heads;
    shape.dim = cfg_.head_dim;
    shape.block_tokens = kBlockTokens;
    shape.max_requests = max_requests_;
    shape.token_slots = max_cache_tokens_;
    pool_.init(shape);
  }
  const size_t M = static_cast<size_t>(max_tokens_);
  resid_ = dev_alloc<uint16_t>(M * H);
  x_ = dev_alloc<uint16_t>(M * H);
  y_ = dev_alloc<uint16_t>(M * H);
  {
    const size_t layers = static_cast<size_t>(cfg_.num_moe_layers()) + (mtp_ ? 1 : 0);
    const size_t K = static_cast<size_t>(cfg_.num_experts_per_tok);
    h_route_ids_ = pinned_alloc<int32_t>(layers * M * K);
    h_route_weights_ = pinned_alloc<float>(layers * M * K);
    // The boundary windows' budget: 20 MB as the Qwen walk's (the fabric
    // sweep of 2026-09-09); DGPP_L2_PREFETCH_MB overrides.
    prefetch_window_bytes_ = std::getenv("DGPP_L2_PREFETCH_MB") ? 0 : (size_t{20} << 20);
  }
  if (mtp_) {
    mtp_h_ = dev_alloc<uint16_t>(M * H);
    mtp_in_ = dev_alloc<uint16_t>(M * 2 * H);
    mtp_r_ = dev_alloc<uint16_t>(M * H);
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}

Glm4Model::~Glm4Model() {
  cudaFree(resid_);
  cudaFree(x_);
  cudaFree(y_);
  cudaFreeHost(h_route_ids_);
  cudaFreeHost(h_route_weights_);
  cudaFree(gemm_ws_);
  cudaFree(mtp_h_);
  cudaFree(mtp_in_);
  cudaFree(mtp_r_);
}

int Glm4Model::table_slots() const {
  return loader_.residency() == Glm4Residency::Resident ? cfg_.num_moe_layers() + (mtp_ ? 1 : 0) : 0;
}

size_t Glm4Model::kv_block_bytes_static(const Glm4TextConfig& cfg, int tp_rank, int tp_world, bool mtp) {
  const Glm4HeadSharding head = tp_world > 1 ? Glm4HeadSharding::VocabSharded : Glm4HeadSharding::Full;
  const Glm4LocalGeometry geo = Glm4LocalGeometry::from_config(cfg, tp_rank, tp_world, head);
  const int pool_layers = cfg.num_hidden_layers + (mtp ? 1 : 0);
  const auto bytes = [&](int64_t tokens) {
    Glm4KvPoolShape shape;
    shape.layers = pool_layers;
    shape.kv_heads = geo.local_kv_heads;
    shape.dim = cfg.head_dim;
    shape.block_tokens = kBlockTokens;
    shape.max_requests = 1;
    shape.token_slots = tokens;
    return Glm4KvPool::cache_bytes(shape) - PagedBlockTable::table_bytes(1, tokens / kBlockTokens);
  };
  return bytes(2 * kBlockTokens) - bytes(kBlockTokens);
}

Glm4Model::MemoryPlan Glm4Model::plan_memory(const Glm4TextConfig& cfg, int max_tokens, int64_t max_cache_tokens,
                                             int tp_rank, int tp_world, Glm4Residency residency, int max_requests,
                                             bool mtp, int decode_rows) {
  if (max_tokens <= 0) throw std::invalid_argument("plan_memory: max_tokens must be positive");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("plan_memory: max_requests must be in [1, kPickMaxRequests]");
  if (decode_rows > decode_rows_cap())
    throw std::invalid_argument("plan_memory: decode_rows exceeds the GLM-4.7 limit of 32");
  // The fixed batch's row ceiling, floored as the session core floors it.
  const int rows = std::max({kDecodeRows, decode_rows, max_requests});
  if (mtp && cfg.mtp_layer() < 0) throw std::invalid_argument("plan_memory: the config has no draft layer");
  const Glm4HeadSharding head = tp_world > 1 ? Glm4HeadSharding::VocabSharded : Glm4HeadSharding::Full;
  const Glm4LocalGeometry geo = Glm4LocalGeometry::from_config(cfg, tp_rank, tp_world, head);
  const int64_t cache_tokens =
      ((std::max<int64_t>(max_cache_tokens, max_tokens) + kBlockTokens - 1) / kBlockTokens) * kBlockTokens;
  MemoryPlan plan;
  plan.context_tokens = std::min<int64_t>(cache_tokens, cfg.max_position_embeddings);
  const size_t M = static_cast<size_t>(max_tokens);
  const size_t R = static_cast<size_t>(max_requests);
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t V = static_cast<size_t>(Glm4LayerStream::lm_vocab_count(cfg, tp_rank, tp_world, head));
  const int pool_layers = cfg.num_hidden_layers + (mtp ? 1 : 0);
  const int n_split = Glm4AttentionLayer::default_decode_splits();

  // bf12-only residency: the packable bf16 matrices load aside and give
  // their bytes back as each layer is packed (graph_prepare).
  const bool bf12_only = Bf12Companions::packed_only() && residency == Glm4Residency::Resident;
  if (residency == Glm4Residency::Resident) {
    size_t resident = Glm4LayerStream::resident_bytes(cfg, tp_rank, tp_world, head, mtp);
    if (bf12_only) resident -= Glm4LayerStream::side_bytes(cfg, tp_rank, tp_world, head, mtp);
    plan.add(bf12_only ? "model weights (resident, packed bf16 matrices released)" : "model weights (resident)",
             resident);
    plan.add("loader staging (pinned host, freed when the last layer is resident)", 0,
             Glm4LayerStream::staging_plan_bytes(cfg, tp_rank, tp_world, head, mtp));
  } else {
    size_t largest = 0;
    const int layers_total = cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
    for (int l = 0; l < layers_total; ++l)
      largest = std::max(largest, Glm4LayerStream::layer_bytes(cfg, l, tp_rank, tp_world));
    plan.add("model weights (one streamed layer + globals)",
             largest + Glm4LayerStream::globals_bytes(cfg, tp_rank, tp_world, head));
  }
  if (Bf12Companions::enabled() && residency == Glm4Residency::Resident) {
    // The lossless 12-bit companions of the bf16 attention projections, the
    // head and the draft's eh_proj (kernels/bf12_companions.hpp).
    const int64_t Q = static_cast<int64_t>(geo.local_heads) * cfg.head_dim;
    const int64_t KV = static_cast<int64_t>(geo.local_kv_heads) * cfg.head_dim;
    const int64_t Hh = cfg.hidden_size;
    const size_t per_layer = Bf12Companions::planned_bytes(Q, Hh) + 2 * Bf12Companions::planned_bytes(KV, Hh) +
                             Bf12Companions::planned_bytes(Hh, Q);
    size_t packed = static_cast<size_t>(pool_layers) * per_layer +
                    Bf12Companions::planned_bytes(static_cast<int64_t>(V), Hh);
    if (mtp) packed += Bf12Companions::planned_bytes(Hh, 2 * Hh);
    plan.add("bf16 decode packing (12-bit companions)", packed);
    if (bf12_only) {
      const size_t h = static_cast<size_t>(Hh);
      plan.add("bf16 prefill expansion scratch",
               static_cast<size_t>(kBf12ExpandSlots) *
                   Bf12Companions::planned_slot_bytes({static_cast<size_t>(Q) * h * 2, static_cast<size_t>(KV) * h * 2,
                                                       mtp ? h * 2 * h * 2 : 0, V * h * 2}));
    }
  }
  plan.add("gemm workspace (at least)", size_t{64} << 20);
  {
    Glm4KvPoolShape shape;
    shape.layers = pool_layers;
    shape.kv_heads = geo.local_kv_heads;
    shape.dim = cfg.head_dim;
    shape.block_tokens = kBlockTokens;
    shape.max_requests = max_requests;
    shape.token_slots = cache_tokens;
    plan.add("kv cache pool (K/V bf16, the draft layer's included)", Glm4KvPool::cache_bytes(shape));
  }
  {
    size_t core_dev = 0, core_pin = 0;
    session_core_plan_bytes(max_tokens, max_requests, cfg.hidden_size, static_cast<int>(V), mtp,
                            cfg.hidden_size, &core_dev, &core_pin, rows);
    const size_t moe_layers = static_cast<size_t>(cfg.num_moe_layers()) + (mtp ? 1 : 0);
    const size_t K = static_cast<size_t>(cfg.num_experts_per_tok);
    plan.add("activations (session core, residual, block io, route staging)", core_dev + 3 * M * H * 2,
             core_pin + moe_layers * M * K * 8);
  }
  {
    size_t layers = Glm4AttentionLayer::scratch_bytes(cfg, geo.local_heads, geo.local_kv_heads, max_tokens, rows,
                                                      n_split);
    if (cfg.first_k_dense_replace > 0) layers += Glm4DenseMlp::scratch_bytes(cfg, geo.local_dense_inter, max_tokens);
    plan.add("layer objects (attention, dense MLP)", layers);
    const GlmMoeConfig moe_cfg = cfg.moe_config(static_cast<int>(geo.local_inter));
    const int slots = residency == Glm4Residency::Resident ? cfg.num_moe_layers() + (mtp ? 1 : 0) : 0;
    size_t moe_pinned = 0;
    const size_t moe_dev = GlmMoeLayer::scratch_bytes(moe_cfg, max_tokens, rows, slots, &moe_pinned);
    plan.add("moe scratch (routed slots, shared expert, graph tables)", moe_dev, moe_pinned);
    plan.add("moe W4A4 activation workspace", GlmMoeLayer::w4a4_scratch_bytes(moe_cfg, max_tokens));
  }
  if (mtp) plan.add("draft block (gathered hidden, the fused input, its residual)", M * H * 2 + M * 2 * H * 2 + M * H * 2);
  (void)R;
  return plan;
}

size_t Glm4Model::session_snapshot_bytes(const Glm4TextConfig& cfg, int, bool mtp) {
  size_t bytes = kSnapshotStamp;
  if (mtp) bytes += static_cast<size_t>(cfg.hidden_size) * 2;  // the draft's hidden row
  return bytes;
}

GlmMoeWeights Glm4Model::moe_view(const Glm4MoeResident& m) {
  GlmMoeWeights w;
  w.router_gate = m.router;
  w.router_bias = m.router_bias;
  for (int i = 0; i < 3; ++i) w.shared_fp4[i] = m.shared[i];
  w.experts = nullptr;
  w.experts_fp4 = m.experts.data();
  return w;
}

void Glm4Model::build_layer_objects(const Glm4LayerResident& r) {
  if (!attn_) {
    attn_ = std::make_unique<Glm4AttentionLayer>(r.attn, gw_, cfg_, max_tokens_, max_decode_rows_, n_split_);
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
      dense_ = std::make_unique<Glm4DenseMlp>(r.dense, cfg_, max_tokens_, max_decode_rows_);
    } else {
      dense_->rebind(r.dense);
    }
  }
}

// The state every row walk starts from on a fresh request: no blocks
// (the K/V rows are written before any row reads them).
void Glm4Model::reset_slot_state(int req) { pool_.release_request_blocks(req, stream_); }

GlmSpecSegments Glm4Model::spec_segments(int, int) const { return GlmSpecSegments{}; }

void Glm4Model::write_state_snapshot(int, uint8_t* d, int) {
  DGPP_CUDA_OK(cudaMemsetAsync(d, 0, kSnapshotStamp, stream_));
}

// ---------------------------------------------------------------------------
// The boundary prefetch windows (bit-identical on or off).
// ---------------------------------------------------------------------------
void Glm4Model::prefetch_fp4(const GlmFp4Matrix& m) {
  if (m.payload) prefetch_.add(m.payload, m.payload_bytes());
  if (m.scales) prefetch_.add(m.scales, m.scale_bytes());
}

void Glm4Model::prefetch_attn(const Glm4AttnResident& a) {
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  const size_t Q = static_cast<size_t>(a.local_heads) * cfg_.head_dim;
  const size_t KV = static_cast<size_t>(a.local_kv_heads) * cfg_.head_dim;
  // The bytes the rows' launches stream: a packed companion's when the
  // GEMM holds one (kernels/bf12_gemv.hpp).
  const auto add = [&](const uint16_t* w, size_t bytes) {
    if (w == nullptr) return;
    const void* view = nullptr;
    size_t view_bytes = 0;
    gemm_.resident_view(w, bytes, walk_rows_, &view, &view_bytes);
    prefetch_.add_view(w, view, view_bytes);  // a companion is its own allocation
  };
  add(a.q_proj, Q * H * 2);
  add(a.k_proj, KV * H * 2);
  add(a.v_proj, KV * H * 2);
}

// Before the attention fold: this layer's post norm, the router (and its
// bias) and the shared expert — or the dense MLP — one image.
void Glm4Model::prefetch_ffn_side(const Glm4LayerResident& r) {
  if (!prefetch_.enabled()) return;
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, prefetch_window_bytes_, prefetch_.boundary_rate());
  if (r.post_norm) prefetch_.add(r.post_norm, H * 2);
  if (r.moe) {
    if (r.moe_w.router) prefetch_.add(r.moe_w.router, static_cast<size_t>(cfg_.n_routed_experts) * H * 2);
    if (r.moe_w.router_bias) prefetch_.add(r.moe_w.router_bias, static_cast<size_t>(cfg_.n_routed_experts) * 4);
    for (int i = 0; i < 3; ++i) prefetch_fp4(r.moe_w.shared[i]);
  } else {
    prefetch_fp4(r.dense.gate);
    prefetch_fp4(r.dense.up);
    prefetch_fp4(r.dense.down);
  }
}

// Before the MLP fold: the next layer's input norm and its projections (or
// the head). Resident stacks only (load_layer is a lookup there).
void Glm4Model::prefetch_attention_side(int layer) {
  if (!prefetch_.enabled()) return;
  if (layer >= cfg_.num_hidden_layers) {
    prefetch_head();
    return;
  }
  if (loader_.residency() != Glm4Residency::Resident) return;
  const Glm4LayerResident& r = loader_.load_layer(layer);
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, prefetch_window_bytes_, prefetch_.boundary_rate());
  if (r.input_norm) prefetch_.add(r.input_norm, H * 2);
  prefetch_attn(r.attn);
}

void Glm4Model::prefetch_head() {
  if (!prefetch_.enabled()) return;
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, prefetch_window_bytes_, prefetch_.boundary_rate());
  if (globals_.final_norm) prefetch_.add(globals_.final_norm, H * 2);
  if (globals_.lm_head) {
    const void* view = nullptr;
    size_t view_bytes = 0;
    gemm_.resident_view(globals_.lm_head, static_cast<size_t>(lm_vocab_count_) * H * 2, walk_rows_, &view,
                        &view_bytes);
    prefetch_.add_view(globals_.lm_head, view, view_bytes);
  }
}

// ---------------------------------------------------------------------------
// The row walk.
// ---------------------------------------------------------------------------
// The boundary folds (plan D7): the producer writes its partial into the
// reducer's staged buffer when the shape fits (the collective sends
// straight from there; under capture the recorder's one stable buffer),
// else into `fallback`. The eager producer quiesces before the collective;
// under capture the fold is a recorded node and the stream order IS the
// drain.
uint16_t* Glm4Model::stage(uint16_t* fallback, int T, int width, bool capture) {
  if (!boundary_) return fallback;
  uint16_t* s = boundary_->stage(T, width);
  if (s == nullptr && capture) throw std::runtime_error("run_rows: a capture fold does not fit the recorder's staged buffer");
  return s ? s : fallback;
}

void Glm4Model::fold(uint16_t* buf, int T, int width, bool capture) {
  if (!boundary_) return;
  if (!capture) DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  boundary_->reduce(buf, T, width);
}

void Glm4Model::enqueue_layer(const Glm4LayerResident& r, int pool_layer, uint16_t* resid, int T,
                              const WalkRows& rows) {
  const int H = cfg_.hidden_size;
  const float eps = cfg_.rms_norm_eps;
  const int64_t n = static_cast<int64_t>(T) * H;
  build_layer_objects(r);
  // ---- the attention block --------------------------------------------------
  glm_rmsnorm_bf16(resid, r.input_norm, x_, T, H, eps, stream_);
  uint16_t* attn_out = stage(y_, T, H, rows.capture);
  {
    Glm4AttnRows arows;
    arows.req_ids = rows.req_ids;
    arows.pos = rows.pos;
    arows.decode = rows.decode;
    attn_->enqueue(x_, T, arows, pool_.view(pool_layer), attn_out, stream_);
  }
  if (rows.decode) prefetch_ffn_side(r);
  fold(attn_out, T, H, rows.capture);  // block boundary 1: o_proj's partial
  glm_residual_add_bf16(resid, attn_out, n, stream_);
  // ---- the MLP / MoE block ----------------------------------------------------
  glm_rmsnorm_bf16(resid, r.post_norm, x_, T, H, eps, stream_);
  uint16_t* ffn_out = stage(y_, T, H, rows.capture);
  if (r.moe) {
    if (rows.decode)
      moe_->enqueue_decode(x_, ffn_out, T, nullptr, stream_, rows.moe_table_slot);
    else
      moe_->enqueue_prefill(x_, ffn_out, T, rows.trace, stream_);
  } else {
    dense_->enqueue(x_, T, ffn_out, stream_);
  }
  if (rows.decode) prefetch_attention_side(r.layer + 1);
  fold(ffn_out, T, H, rows.capture);  // block boundary 2: the sliced down projections
  glm_residual_add_bf16(resid, ffn_out, n, stream_);
}

Glm4Model::Outputs Glm4Model::run_rows(const RowRun& run) {
  const int T = run.T, req = run.req;
  if (run.capture && loader_.residency() != Glm4Residency::Resident)
    throw std::logic_error("run_rows: a capture needs a resident stack");
  const int H = cfg_.hidden_size;
  // The packed companions' wide launches are the decode batch's alone (a
  // short prefill chunk keeps its Lt algorithm: kernels/gemm.hpp).
  gemm_.set_bf12_wide(run.decode);
  walk_rows_ = T;
  const RowInputs in = begin_run(run);
  embed_gather_bf16(globals_.embed, in.tokens, resid_, T, H, stream_);
  Outputs out;
  const bool traces = !run.decode && route_traces_;
  const size_t K = static_cast<size_t>(cfg_.num_experts_per_tok);
  WalkRows rows;
  rows.req_ids = in.req_ids;
  rows.pos = in.pos;
  rows.req = req;
  rows.decode = run.decode;
  rows.capture = run.capture;
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const Glm4LayerResident& r = loader_.load_layer(layer);
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
        trace.biased = nullptr;
        rows.trace = &trace;
        out.route_ids.emplace_back();  // filled after the sync
        out.route_weights.emplace_back();
      }
    }
    enqueue_layer(r, layer, resid_, T, rows);
    if (run.capture_layers) {
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      std::vector<uint16_t> snap(static_cast<size_t>(T) * H);
      DGPP_CUDA_OK(cudaMemcpy(snap.data(), resid_, snap.size() * 2, cudaMemcpyDeviceToHost));
      out.layer_states.push_back(std::move(snap));
    }
  }
  // The head on every row: a prefill chunk's last row comes off the same
  // m=T GEMM the diagnostic forward runs (the prefill == forward bitwise
  // gate).
  glm_rmsnorm_bf16(resid_, globals_.final_norm, h_, T, H, cfg_.rms_norm_eps, stream_);
  gemm_.matmul(h_, globals_.lm_head, logits_, T, lm_vocab_count_, H, DType::BF16, GemmOut::F32,
               static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
  // The draft block's input: the last rows' POST-final-norm hidden (the
  // model's output hidden state, vLLM's glm4_moe_mtp convention — the
  // draft applies hnorm to it; the pre-norm residual halved the draft's
  // acceptance on the real checkpoint, 2026-09-10) into the slots'
  // windows by position (the last window rows of a prefill chunk, every
  // decode row — distinct slots within one launch).
  // A group prefill (several requests' spans in one walk, 2026-09-14)
  // stores every row: each span's last window rows land in its own slot.
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
Glm4Model::Outputs Glm4Model::forward(const std::vector<int64_t>& token_ids, bool capture_layers) {
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
void Glm4Model::graph_prepare() {
  if (loader_.residency() != Glm4Residency::Resident)
    throw std::logic_error("session_graph_prepare: the decode graph needs a resident stack");
  // The bf16 decode weights' 12-bit companions are packed as each layer
  // lands (the caches already exist: under bf12-only residency the layer's
  // bf16 bytes go back at once, so no more than one layer's sit beside
  // their companions).
  const bool pack = Bf12Companions::enabled() && !bf12_built_;
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const Glm4LayerResident& r = loader_.load_layer(layer);
    build_layer_objects(r);
    if (r.moe) moe_->prepare_graph_table(moe_ordinal(layer), stream_);
    if (pack) pack_layer_companions(layer, r);
  }
  if (mtp_) {
    const Glm4LayerResident& r = loader_.load_layer(cfg_.mtp_layer());
    build_layer_objects(r);
    moe_->prepare_graph_table(cfg_.num_moe_layers(), stream_);
    if (pack) pack_layer_companions(cfg_.mtp_layer(), r);
  }
  if (pack) finish_companions();
}

// The lossless 12-bit companions of the decode GEMV's bf16 weights
// (engine.bf16_weights = "bf12"): the attention projections of every layer
// — 6.3 of the 9.7 GB a decode step reads per rank at world 4 — the head
// and the draft's eh_proj, packed as each layer lands and before any
// capture. Prefill and the batches past eight rows keep the bf16 bytes —
// or, under bf12-only residency (the bytes returned here, layer by layer),
// expand them and take the packed launches (kernels/gemm.hpp).
void Glm4Model::pack_layer_companions(int layer, const Glm4LayerResident& r) {
  const auto t0 = std::chrono::steady_clock::now();
  const int64_t H = cfg_.hidden_size;
  const auto release = [&](const void* w) { return loader_.release_packed(layer, w); };
  const auto pack = [&](const uint16_t* w, int64_t n, int64_t k) {
    bf12_.pack_and_release(w, n, k, gemm_, stream_, release);
  };
  const Glm4AttnResident& a = r.attn;
  const int64_t Q = static_cast<int64_t>(a.local_heads) * cfg_.head_dim;
  const int64_t KV = static_cast<int64_t>(a.local_kv_heads) * cfg_.head_dim;
  pack(a.q_proj, Q, H);
  pack(a.k_proj, KV, H);
  pack(a.v_proj, KV, H);
  pack(a.o_proj, H, Q);
  if (layer == cfg_.mtp_layer() && r.eh_proj != nullptr) pack(r.eh_proj, H, 2 * H);
  bf12_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// The lm head's, the expansion scratch (one slot: no walk of this family
// calls a matrix twice), and the summary.
void Glm4Model::finish_companions() {
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
// one full layer (its own pool layer), shared_head.norm, the shared head.
// ---------------------------------------------------------------------------
void Glm4Model::mtp_run_rows(int req, const int64_t* tokens, int64_t first_pos, int T, bool decode_row,
                             bool capture, int head_rows, int batch_requests) {
  if (!mtp_) throw std::logic_error("mtp_run_rows: MTP is not enabled");
  if (T <= 0 || T > max_tokens_) throw std::invalid_argument("mtp_run_rows: rows");
  gemm_.set_bf12_wide(decode_row);  // the decode batch's alone (kernels/gemm.hpp)
  walk_rows_ = T;
  if (head_rows < 0 || head_rows > T) throw std::invalid_argument("mtp_run_rows: head_rows");
  const int H = cfg_.hidden_size;
  const float eps = cfg_.rms_norm_eps;
  const int64_t* d_pos = decode_row ? d_step_pos_ : d_prefill_pos_;
  const int32_t* d_req = decode_row ? d_req_ids_ : d_prefill_req_;
  if (!decode_row) stage_prefill_meta(req, first_pos, T);
  const Glm4LayerResident& r = loader_.load_layer(cfg_.mtp_layer());
  if (!r.enorm || !r.hnorm || !r.eh_proj || !r.shared_head_norm)
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
  rows.moe_table_slot = capture ? cfg_.num_moe_layers() : -1;
  rows.trace = nullptr;
  (void)batch_requests;
  enqueue_layer(r, cfg_.num_hidden_layers, mtp_r_, T, rows);
  if (head_rows == 0) {  // prefill rows fill the cache; no head
    if (decode_row) prefetch_.join(stream_);
    return;
  }
  // ---- head: the draft distribution over the last head_rows rows --------
  const uint16_t* head_in = mtp_r_ + static_cast<size_t>(T - head_rows) * H;
  glm_rmsnorm_bf16(head_in, r.shared_head_norm, h_, head_rows, H, eps, stream_);
  gemm_.matmul(h_, globals_.lm_head, logits_, head_rows, lm_vocab_count_, H, DType::BF16, GemmOut::F32,
               static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
  if (decode_row) prefetch_.join(stream_);
  if (decode_row && (!capture || decode_tail_mirrors_) && head_rows <= max_decode_rows_)
    DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_logits_, logits_, static_cast<size_t>(head_rows) * lm_vocab_count_ * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
}

Glm4Model::Outputs Glm4Model::mtp_forward(const std::vector<int64_t>& token_ids) {
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
