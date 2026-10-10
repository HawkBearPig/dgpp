#include "models/dsv4/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "kernels/csa2.hpp"
#include "kernels/dsv41_dspark.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/kernels.hpp"
#include "kernels/mma_gemv.hpp"
#include "kernels/scale_gemm.hpp"

namespace dgpp {
using session_detail::dev_alloc;
using session_detail::pinned_alloc;

namespace {
// The fp8 matrices' scale grid as log2 block sizes (the release's 128 x 128).
constexpr int kGrid = 7;
}  // namespace

// ---------------------------------------------------------------------------
// The shapes.
// ---------------------------------------------------------------------------
Dsv4AttnConfig Dsv4Model::attn_config(const Dsv4Config& cfg, int tp_world) {
  Dsv4AttnConfig c;
  c.hidden = cfg.hidden_size;
  c.q_lora = cfg.q_lora_rank;
  c.o_lora = cfg.o_lora_rank;
  c.num_heads = cfg.num_attention_heads;
  c.o_groups = cfg.o_groups;
  c.index_heads = cfg.index_n_heads;
  c.index_topk = cfg.index_topk;
  c.window = cfg.sliding_window;
  c.ring_slots = std::max(kRingSlots, cfg.sliding_window + 32);
  c.block_tokens = kBlockTokens;
  c.eps = cfg.rms_norm_eps;
  c.tp = tp_world;
  Dsv4AttnConfig::validate(c);
  return c;
}

Dsv4PoolShape Dsv4Model::pool_shape(const Dsv4Config& cfg, int max_requests, int64_t cache_tokens) {
  Dsv4PoolShape s;
  s.layers = cfg.max_layer();
  for (int l = 0; l < cfg.max_layer(); ++l)
    if (cfg.compress_ratio(l) > 0) s.cache_ratio.push_back(cfg.compress_ratio(l));
  s.max_requests = max_requests;
  s.token_slots = cache_tokens;
  s.block_tokens = kBlockTokens;
  s.ring_slots = std::max(kRingSlots, cfg.sliding_window + 32);
  // The ratio-128 compressors pool lazily wherever the group GEMM takes the
  // width (kernels/dense_mma_bf16w.hpp: k a multiple of 32).
  s.lazy_hidden = cfg.hidden_size % 32 == 0 ? cfg.hidden_size : 0;
  return s;
}

// ---------------------------------------------------------------------------
// Construction.
// ---------------------------------------------------------------------------
Dsv4Model::Dsv4Model(const Dsv4Config& cfg, const std::string& checkpoint_dir, int max_tokens,
                     int64_t max_cache_tokens, Dsv4Residency residency, BoundaryReducer* boundary, int tp_rank,
                     int tp_world, int max_requests, bool mtp, int decode_rows, bool serving_logits)
    : cfg_(cfg),
      loader_(cfg, checkpoint_dir, tp_rank, tp_world, residency,
              tp_world > 1 ? Dsv4HeadSharding::VocabSharded : Dsv4HeadSharding::Full,
              mtp && residency == Dsv4Residency::Resident),
      world_(tp_world) {
  if (max_tokens <= 0) throw std::invalid_argument("Dsv4Model: max_tokens must be positive");
  if (mtp) {
    if (cfg_.draft_stages < 1 || cfg_.dspark_target_layer_ids.empty())
      throw std::invalid_argument("Dsv4Model: mtp needs the DSpark draft stages and target layers");
    if (cfg_.dspark_block_size < 1 || cfg_.dspark_block_size > Dsv4AttnLayer::kMaxDraftBlock ||
        cfg_.dspark_block_size > kSpecRows - 1)
      throw std::invalid_argument("Dsv4Model: dspark_block_size must be in [1, min(kMaxDraftBlock, kSpecRows - 1)]");
    if (cfg_.dspark_markov_rank < 1 || cfg_.dspark_markov_rank > 512 || cfg_.dspark_noise_token_id < 0 ||
        cfg_.dspark_noise_token_id >= cfg_.vocab_size)
      throw std::invalid_argument("Dsv4Model: the DSpark Markov rank or noise token is out of range");
    targets_ = static_cast<int>(cfg_.dspark_target_layer_ids.size());
  }
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("Dsv4Model: max_requests must be in [1, kPickMaxRequests]");
  if (max_requests > decode_rows_cap() || decode_rows > decode_rows_cap())
    throw std::invalid_argument("Dsv4Model: the decode batch is bounded at " + std::to_string(decode_rows_cap()) +
                                " rows: max_requests and decode_rows must not exceed it");
  if ((tp_world > 1) != (boundary != nullptr))
    throw std::invalid_argument("Dsv4Model: a boundary reducer is required exactly when tp_world > 1");
  if (cfg_.sliding_window > max_tokens)
    throw std::invalid_argument("Dsv4Model: max_tokens must cover the sliding window");
  init_stream();
  loader_.set_reader_stream(stream_);
  log_memory_ledger("dsv4: loader opened");
  const int H = cfg_.hidden_size;
  globals_ = loader_.load_globals();
  embed_sharded_ = globals_.embed_vocab_count < cfg_.vocab_size;
  if (embed_sharded_ && boundary == nullptr)
    throw std::invalid_argument("Dsv4Model: a vocab-sharded embedding needs the boundary reducer (world > 1)");
  if (residency == Dsv4Residency::Resident) {
    // Eager load, parallel when DGPP_PRELOAD_THREADS asks (unset keeps the
    // serial loop); release_sources() still runs once everything is up.
    loader_.load_layers_parallel(0, (mtp ? cfg_.max_layer() : cfg_.num_hidden_layers),
                                 loader_preload_threads());
    loader_.release_sources();
  }
  log_memory_ledger("dsv4: layers resident, sources released");
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
    sp.mtp = mtp;
    sp.vocab_size = cfg_.vocab_size;
    sp.hidden = H;
    sp.lm_vocab_begin = globals_.lm_vocab_begin;
    sp.lm_vocab_count = globals_.lm_vocab_count > 0 ? globals_.lm_vocab_count : cfg_.vocab_size;
    sp.max_position_embeddings = cfg_.max_position_embeddings;
    sp.block_tokens = kBlockTokens;
    // A whole block: every ratio-128 group is complete at a cut, so a
    // snapshot carries no ratio-128 compressor state.
    sp.snapshot_align = kBlockTokens;
    sp.draft_width = mtp ? targets_ * H : H;  // the draft window holds [h_t1 | h_t2 | h_t3] per row
    sp.eos = static_cast<int32_t>(cfg_.eos_token_id);
    init_session(sp);
  }
  if (max_decode_rows_ > decode_rows_cap())
    throw std::logic_error("Dsv4Model: the session core widened the decode batch past the family's cap");
  gemm_ws_bytes_ = std::max<size_t>(64u << 20, gemm_.query_workspace_bytes(max_tokens_, lm_vocab_count_, H, DType::BF16));
  gemm_ws_ = dev_alloc<char>(gemm_ws_bytes_);
  gemm_.set_decode_rows(max_decode_rows_);
  const Dsv4LocalGeometry& geo = loader_.geometry();
  moe_cfg_ = cfg_.moe_config(static_cast<int>(geo.local_inter));
  mhc_cfg_.hc_mult = cfg_.hc_mult;
  mhc_cfg_.hidden = H;
  mhc_cfg_.sinkhorn_iters = cfg_.hc_sinkhorn_iters;
  mhc_cfg_.hc_eps = cfg_.hc_eps;
  mhc_cfg_.norm_eps = cfg_.rms_norm_eps;
  GlmMhcConfig::validate_config(mhc_cfg_);
  attn_cfg_ = attn_config(cfg_, tp_world);
  if (boundary_) boundary_->bind_stream(stream_);  // the stream-ordered reducer's stream
  attn_cfg_.dense_mma = dense_mma_;
  gemm_.set_decode_mma(dense_mma_);
  // The mHC dots take the tiled form at every prefill row count: a row's
  // collapse coefficients are then one chain whatever rows share the
  // launch, and a prompt prefilled in a group is bitwise the prompt alone.
  mhc_set_tile_min_tokens(1);
  // The pool and the attention scratch.
  const int64_t slots = max_cache_tokens_;
  pool_.init(pool_shape(cfg_, max_requests_, slots));
  attn_scratch_bytes_ =
      Dsv4AttnLayer::scratch_bytes(attn_cfg_, max_tokens_, slots, max_decode_rows_, kDecodeSplit, kDotBudget);
  attn_scratch_ = dev_alloc<char>(attn_scratch_bytes_);
  log_memory_ledger("dsv4: pool and scratch");
  // The rotary tables: the window-only layers' (theta, no YaRN) and the
  // compressing layers' (compress theta with YaRN).
  {
    std::vector<float> win(32), comp(32);
    csa2_rope_inv_freq_host(cfg_.qk_rope_head_dim, cfg_.rope_theta, 0, cfg_.rope_factor, cfg_.beta_fast, cfg_.beta_slow,
                            win.data());
    csa2_rope_inv_freq_host(cfg_.qk_rope_head_dim, cfg_.compress_rope_theta, cfg_.original_max_position_embeddings,
                            cfg_.rope_factor, cfg_.beta_fast, cfg_.beta_slow, comp.data());
    inv_freq_window_ = dev_alloc<float>(32);
    inv_freq_compressed_ = dev_alloc<float>(32);
    DGPP_CUDA_OK(cudaMemcpy(inv_freq_window_, win.data(), 32 * 4, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(inv_freq_compressed_, comp.data(), 32 * 4, cudaMemcpyHostToDevice));
  }
  const size_t M = static_cast<size_t>(max_tokens_);
  const size_t R = static_cast<size_t>(max_requests_);
  streams_a_ = dev_alloc<uint16_t>(M * 4 * H);
  streams_b_ = dev_alloc<uint16_t>(M * 4 * H);
  x_ = dev_alloc<uint16_t>(M * H);
  y_ = dev_alloc<uint16_t>(M * H);
  collapsed_ = dev_alloc<uint16_t>(M * H);
  post_bf16_ = dev_alloc<uint16_t>(M * 4);
  comb_bf16_ = dev_alloc<uint16_t>(M * 16);
  post_f32_ = dev_alloc<float>(M * 4);
  comb_f32_ = dev_alloc<float>(M * 16);
  mhc_logits_ = dev_alloc<float>(M * static_cast<size_t>(mhc_cfg_.coeff_rows()));
  // The decode sites' fused mHC: the finish tickets and the comb's side stream.
  mhc_counters_ = dev_alloc<int>(M);
  DGPP_CUDA_OK(cudaMemset(mhc_counters_, 0, M * sizeof(int)));
  DGPP_CUDA_OK(cudaStreamCreateWithFlags(&mhc_side_, cudaStreamNonBlocking));
  DGPP_CUDA_OK(cudaEventCreateWithFlags(&mhc_fork_, cudaEventDisableTiming));
  DGPP_CUDA_OK(cudaEventCreateWithFlags(&mhc_join_, cudaEventDisableTiming));
  zero_bias_ = dev_alloc<float>(static_cast<size_t>(cfg_.n_routed_experts));
  DGPP_CUDA_OK(cudaMemset(zero_bias_, 0, static_cast<size_t>(cfg_.n_routed_experts) * 4));
  if (mtp) {
    const size_t W = static_cast<size_t>(targets_) * H;
    const size_t D = static_cast<size_t>(max_decode_rows_);
    const size_t block = static_cast<size_t>(cfg_.dspark_block_size);
    main_hidden_ = dev_alloc<uint16_t>(M * W);
    main_gather_ = dev_alloc<uint16_t>(D * W);
    main_x_ = dev_alloc<uint16_t>(std::max(M, D) * H);
    draft_collapsed_ = dev_alloc<uint16_t>(D * H);
    blk_pos_ = dev_alloc<int64_t>(D);
    blk_tok_ = dev_alloc<int64_t>(D);
    blk_req_ = dev_alloc<int32_t>(D);
    blk_spans_ = dev_alloc<int32_t>(D + 1);
    base_logits_ = dev_alloc<float>(D * static_cast<size_t>(lm_vocab_count_));
    conf_ = dev_alloc<float>(R * block);
    d_draft_pos_ = dev_alloc<int64_t>(M);
    d_draft_req_ = dev_alloc<int32_t>(M);
    draft_row_max_ = dev_alloc<float>(D * ((static_cast<size_t>(lm_vocab_count_) + 255) / 256));
    DGPP_CUDA_OK(cudaMemsetAsync(conf_, 0, R * block * 4, stream_));
    build_draft_head_screen();
  }
  {
    const size_t layers = static_cast<size_t>(cfg_.num_hidden_layers);
    const size_t K = static_cast<size_t>(cfg_.num_experts_per_tok);
    h_route_ids_ = pinned_alloc<int32_t>(layers * M * K);
    h_route_weights_ = pinned_alloc<float>(layers * M * K);
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  log_memory_ledger("dsv4: model ready");
}

Dsv4Model::~Dsv4Model() {
  cudaStreamSynchronize(stream_);
  attn_.reset();
  moe_.reset();
  cudaFree(gemm_ws_);
  cudaFree(attn_scratch_);
  cudaFree(inv_freq_window_);
  cudaFree(inv_freq_compressed_);
  cudaFree(streams_a_);
  cudaFree(streams_b_);
  cudaFree(x_);
  cudaFree(y_);
  cudaFree(collapsed_);
  cudaFree(post_bf16_);
  cudaFree(comb_bf16_);
  cudaFree(post_f32_);
  cudaFree(comb_f32_);
  cudaFree(mhc_logits_);
  cudaFree(mhc_counters_);
  if (mhc_join_) cudaEventDestroy(mhc_join_);
  if (mhc_fork_) cudaEventDestroy(mhc_fork_);
  if (mhc_side_) {
    cudaStreamSynchronize(mhc_side_);
    cudaStreamDestroy(mhc_side_);
  }
  cudaFree(zero_bias_);
  cudaFree(main_hidden_);
  cudaFree(main_gather_);
  cudaFree(main_x_);
  cudaFree(draft_collapsed_);
  cudaFree(blk_pos_);
  cudaFree(blk_tok_);
  cudaFree(blk_req_);
  cudaFree(blk_spans_);
  cudaFree(base_logits_);
  cudaFree(conf_);
  cudaFree(d_draft_pos_);
  cudaFree(d_draft_req_);
  cudaFree(draft_row_max_);
  cudaFree(draft_head_fp8_);
  cudaFree(draft_head_scales_);
  cudaFreeHost(h_route_ids_);
  cudaFreeHost(h_route_weights_);
}

// ---------------------------------------------------------------------------
// The memory plan.
// ---------------------------------------------------------------------------
Dsv4Model::MemoryPlan Dsv4Model::plan_memory(const Dsv4Config& cfg, int max_tokens, int64_t max_cache_tokens,
                                             int tp_rank, int tp_world, Dsv4Residency residency, int max_requests,
                                             bool mtp, int decode_rows, bool serving_logits) {
  if (max_tokens <= 0) throw std::invalid_argument("plan_memory: max_tokens must be positive");
  if (mtp && (cfg.draft_stages < 1 || cfg.dspark_target_layer_ids.empty()))
    throw std::invalid_argument("plan_memory: mtp needs the DSpark draft stages and target layers");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("plan_memory: max_requests must be in [1, kPickMaxRequests]");
  if (max_requests > decode_rows_cap() || decode_rows > decode_rows_cap())
    throw std::invalid_argument("plan_memory: the decode batch is bounded at " + std::to_string(decode_rows_cap()) + " rows");
  const int rows = std::max({kDecodeRows, decode_rows, max_requests});
  const Dsv4HeadSharding head = tp_world > 1 ? Dsv4HeadSharding::VocabSharded : Dsv4HeadSharding::Full;
  const Dsv4LocalGeometry geo = Dsv4LocalGeometry::from_config(cfg, tp_rank, tp_world, head);
  const int64_t cache_tokens =
      ((std::max<int64_t>(max_cache_tokens, max_tokens) + kBlockTokens - 1) / kBlockTokens) * kBlockTokens;
  MemoryPlan plan;
  plan.context_tokens = std::min<int64_t>(cache_tokens, cfg.max_position_embeddings);
  const size_t M = static_cast<size_t>(max_tokens);
  const size_t R = static_cast<size_t>(max_requests);
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t V = static_cast<size_t>(Dsv4LayerStream::lm_vocab_count(cfg, tp_rank, tp_world, head));
  size_t staging = 0;
  if (residency == Dsv4Residency::Resident) {
    plan.add("model weights (resident)", Dsv4LayerStream::resident_bytes(cfg, tp_rank, tp_world, head, mtp));
    staging = Dsv4LayerStream::staging_plan_bytes(cfg, tp_rank, tp_world, head, mtp);
  } else {
    size_t largest = 0;
    for (int l = 0; l < (mtp ? cfg.max_layer() : cfg.num_hidden_layers); ++l)
      largest = std::max(largest, Dsv4LayerStream::layer_bytes(cfg, l, tp_rank, tp_world));
    plan.add("model weights (one streamed layer + globals)",
             largest + Dsv4LayerStream::globals_bytes(cfg, tp_rank, tp_world, head));
  }
  const size_t after_weights = plan.total_bytes();
  plan.add("gemm workspace (at least)", size_t{64} << 20);
  const Dsv4AttnConfig ac = attn_config(cfg, tp_world);
  const Dsv4PoolShape shape = pool_shape(cfg, max_requests, cache_tokens);
  plan.add("attention cache pool (every compressing layer's bf16 KV, the ratio-4 index keys, the window and "
           "compressor rings)",
           Dsv4StatePool::cache_bytes(shape));
  plan.add("attention scratch (projections, selection, attention, the prefill dot tiles)",
           Dsv4AttnLayer::scratch_bytes(ac, max_tokens, cache_tokens, rows, kDecodeSplit, kDotBudget));
  {
    size_t core_dev = 0, core_pin = 0;
    const size_t targets = cfg.dspark_target_layer_ids.size();
    const int draft_width = mtp ? static_cast<int>(targets) * cfg.hidden_size : cfg.hidden_size;
    session_core_plan_bytes(max_tokens, max_requests, cfg.hidden_size, static_cast<int>(V), mtp, draft_width, &core_dev,
                            &core_pin, rows, serving_logits);
    const size_t K = static_cast<size_t>(cfg.num_experts_per_tok);
    size_t act = core_dev + 2 * M * 4 * H * 2 + 3 * M * H * 2 + M * (4 + 16) * 2 + M * (4 + 16) * 4 + M * 24 * 4 +
                 static_cast<size_t>(cfg.n_routed_experts) * 4;
    if (mtp) {
      const size_t D = static_cast<size_t>(rows);
      act += M * targets * H * 2 + D * targets * H * 2 + std::max(M, D) * H * 2 + D * H * 2 + D * (8 + 8 + 4 + 4) + 4 +
             D * V * 4 + R * static_cast<size_t>(cfg.dspark_block_size) * 4 + M * 12 + D * ((V + 255) / 256) * 4;
    }
    plan.add("activations (session core, the four residual streams, block io, mHC coefficients, route staging)", act,
             core_pin + static_cast<size_t>(cfg.num_hidden_layers) * M * K * 8);
  }
  if (mtp)
    plan.add("draft head screen (a block-FP8 copy of the lm head slice)",
             V * H + ((V + 127) / 128) * ((H + 127) / 128) * sizeof(float));
  {
    const GlmMoeConfig moe_cfg = cfg.moe_config(static_cast<int>(geo.local_inter));
    const int slots = residency == Dsv4Residency::Resident ? (mtp ? cfg.max_layer() : cfg.num_hidden_layers) : 0;
    size_t moe_pinned = 0;
    const size_t moe_dev = GlmMoeLayer::scratch_bytes(moe_cfg, max_tokens, rows, slots, &moe_pinned);
    plan.add("moe scratch (routed slots, shared expert, graph tables)", moe_dev, moe_pinned);
  }
  if (staging > 0) {
    const size_t later = plan.total_bytes() - after_weights;
    plan.add("loader staging beyond the caches that replace it (pinned host, transient)", 0,
             staging > later ? staging - later : 0);
  }
  return plan;
}

// A prefix snapshot: every layer's window ring and the ratio-4 layers'
// compressor rings (main and index). A cut sits on a 128-token block, where
// every ratio-128 group is complete — those rings carry nothing forward.
size_t Dsv4Model::session_snapshot_bytes(const Dsv4Config& cfg, int, bool) {
  const size_t ring = static_cast<size_t>(std::max(kRingSlots, cfg.sliding_window + 32)) * Dsv4StatePool::kRowBytes;
  size_t b = static_cast<size_t>(cfg.max_layer()) * ring;
  for (int l = 0; l < cfg.max_layer(); ++l)
    if (cfg.compress_ratio(l) == 4)
      b += (Dsv4StatePool::main_geom(4).ring_elems() + Dsv4StatePool::index_geom(4).ring_elems()) * sizeof(float);
  return b + 16;
}

size_t Dsv4Model::snapshot_state_bytes() const { return session_snapshot_bytes(cfg_, world_, false); }

// ---------------------------------------------------------------------------
// Views and layer objects.
// ---------------------------------------------------------------------------
Dsv4AttnWeights Dsv4Model::attn_view(const Dsv4LayerResident& r, int layer) const {
  const Dsv4AttnResident& a = r.attn;
  Dsv4AttnWeights w;
  w.wq_a = a.wq_a;
  w.wkv = a.wkv;
  w.wq_b = a.wq_b;
  w.wo_a = a.wo_a;
  w.wo_b = a.wo_b;
  w.q_norm = a.q_norm;
  w.kv_norm = a.kv_norm;
  w.attn_sink = a.attn_sink;
  w.comp = a.comp;
  w.idx_wq_b = a.idx_wq_b;
  w.idx_wp = a.idx_wp;
  w.idx_comp = a.idx_comp;
  w.ratio = cfg_.compress_ratio(layer);
  w.inv_freq = w.ratio > 0 ? inv_freq_compressed_ : inv_freq_window_;
  w.cache_ord = cfg_.cache_ordinal(layer);
  return w;
}

GlmMoeWeights Dsv4Model::moe_view(const Dsv4MoeResident& m) const {
  GlmMoeWeights w;
  w.router_gate = m.router;
  // A hash layer has no selection bias (its experts come from the table).
  w.router_bias = m.router_bias != nullptr ? m.router_bias : zero_bias_;
  for (int i = 0; i < 3; ++i) w.shared[i] = m.shared[i];
  w.experts_fp4 = m.experts.data();
  return w;
}

void Dsv4Model::build_layer_objects(const Dsv4LayerResident& r) {
  if (!attn_)
    attn_ = std::make_unique<Dsv4AttnLayer>(gemm_, attn_cfg_, max_tokens_, max_cache_tokens_, attn_scratch_,
                                            attn_scratch_bytes_, gemm_ws_, gemm_ws_bytes_, max_decode_rows_,
                                            kDecodeSplit, kDotBudget);
  // The in-layer windows read the layer's resident image; a streamed layer's
  // staging is recycled under them.
  attn_->set_prefetcher(loader_.residency() == Dsv4Residency::Resident ? &prefetch_ : nullptr);
  attn_->rebind(attn_view(r, r.layer), r.layer);
  if (!moe_) {
    const int slots =
        loader_.residency() == Dsv4Residency::Resident ? (mtp_ ? cfg_.max_layer() : cfg_.num_hidden_layers) : 0;
    moe_ = std::make_unique<GlmMoeLayer>(moe_view(r.moe), moe_cfg_, max_tokens_, max_decode_rows_, slots);
  } else {
    moe_->rebind(moe_view(r.moe));
  }
}

int Dsv4Model::target_ordinal(int layer) const {
  for (int i = 0; i < targets_; ++i)
    if (cfg_.dspark_target_layer_ids[static_cast<size_t>(i)] == layer) return i;
  return -1;
}

const Dsv4DraftResident& Dsv4Model::draft_stage(int stage) {
  return loader_.load_layer(cfg_.num_hidden_layers + stage).draft;
}

// ---------------------------------------------------------------------------
// The session core's state hooks.
// ---------------------------------------------------------------------------
void Dsv4Model::reset_slot_state(int req) { pool_.reset_request(req, stream_); }

void Dsv4Model::write_state_snapshot(int req, uint8_t* d, int) {
  // Every ring is positional: any spec row's view is the ring itself (a
  // later row overwrote only slots no query or group of that position reads).
  const size_t r = static_cast<size_t>(req);
  const size_t rb = pool_.ring_bytes_per_request();
  for (int l = 0; l < pool_.shape().layers; ++l) {
    DGPP_CUDA_OK(cudaMemcpyAsync(d, pool_.ring(l) + r * rb, rb, cudaMemcpyDeviceToDevice, stream_));
    d += rb;
  }
  for (int o = 0; o < pool_.caches(); ++o) {
    if (!pool_.indexed(o)) continue;
    const size_t cb = pool_.comp_ring_bytes_per_request(o), ib = pool_.index_comp_ring_bytes_per_request(o);
    DGPP_CUDA_OK(cudaMemcpyAsync(d, reinterpret_cast<const uint8_t*>(pool_.comp_ring(o)) + r * cb, cb,
                                 cudaMemcpyDeviceToDevice, stream_));
    d += cb;
    DGPP_CUDA_OK(cudaMemcpyAsync(d, reinterpret_cast<const uint8_t*>(pool_.index_comp_ring(o)) + r * ib, ib,
                                 cudaMemcpyDeviceToDevice, stream_));
    d += ib;
  }
  DGPP_CUDA_OK(cudaMemsetAsync(d, 0, 16, stream_));
}

void Dsv4Model::read_state_snapshot(int req, const uint8_t* s) {
  const size_t r = static_cast<size_t>(req);
  const size_t rb = pool_.ring_bytes_per_request();
  for (int l = 0; l < pool_.shape().layers; ++l) {
    DGPP_CUDA_OK(cudaMemcpyAsync(pool_.ring(l) + r * rb, s, rb, cudaMemcpyDeviceToDevice, stream_));
    s += rb;
  }
  for (int o = 0; o < pool_.caches(); ++o) {
    if (!pool_.indexed(o)) continue;
    const size_t cb = pool_.comp_ring_bytes_per_request(o), ib = pool_.index_comp_ring_bytes_per_request(o);
    DGPP_CUDA_OK(cudaMemcpyAsync(reinterpret_cast<uint8_t*>(pool_.comp_ring(o)) + r * cb, s, cb,
                                 cudaMemcpyDeviceToDevice, stream_));
    s += cb;
    DGPP_CUDA_OK(cudaMemcpyAsync(reinterpret_cast<uint8_t*>(pool_.index_comp_ring(o)) + r * ib, s, ib,
                                 cudaMemcpyDeviceToDevice, stream_));
    s += ib;
  }
}

void Dsv4Model::graph_prepare() {
  if (loader_.residency() != Dsv4Residency::Resident)
    throw std::logic_error("session_graph_prepare: the decode graph needs a resident stack");
  for (int layer = 0; layer < (mtp_ ? cfg_.max_layer() : cfg_.num_hidden_layers); ++layer) {
    const Dsv4LayerResident& r = loader_.load_layer(layer);
    build_layer_objects(r);
    for (int rows = 1; rows <= max_decode_rows_; ++rows)
      if (!attn_->prepare(rows)) throw std::runtime_error("session_graph_prepare: attention GEMM plans unavailable");
    moe_->prepare_graph_table(layer, stream_);
  }
}

// ---------------------------------------------------------------------------
// The DSpark draft.
// ---------------------------------------------------------------------------
void Dsv4Model::mtp_run_rows(int req, const int64_t* tokens, int64_t first_pos, int T, bool decode_row, bool capture,
                             int head_rows, int batch_requests) {
  if (head_rows > logits_capacity_rows_) throw std::invalid_argument("draft head exceeds logits capacity");
  if (!mtp_) throw std::logic_error("mtp_run_rows: MTP is not enabled");
  if (T <= 0 || T > max_tokens_) throw std::invalid_argument("mtp_run_rows: rows");
  int src_row0 = 0;
  if (!decode_row && head_rows == 0) {
    // The prefill's draft rows: the rows the last walk carried target
    // hidden for, up to the caller's end — of them the last ring's worth
    // (the draft stages attend their window only: an older row is past
    // every window, and its projection was work for nothing), read in
    // place from the request's span of the walk.
    const WalkSegment seg = segment_of(req);
    const int64_t hi = std::min<int64_t>(first_pos + T, seg.pos0 + seg.rows);
    if (hi <= seg.pos0) return;
    const int rows = static_cast<int>(hi - seg.pos0);
    const int skip = rows - std::min(rows, attn_cfg_.ring_slots);
    first_pos = seg.pos0 + skip;
    T = rows - skip;
    src_row0 = seg.row0 + skip;
  }
  if (head_rows < 0 || head_rows > T) throw std::invalid_argument("mtp_run_rows: head_rows");
  const int groups = batch_requests > 0 ? batch_requests : 1;
  if (T % groups != 0 || (head_rows > 0 && head_rows % groups != 0))
    throw std::invalid_argument("mtp_run_rows: rows must divide into the batch's groups");
  // A chain row: every call after the step's first draft until the next
  // walk (the capture sequence is host-ordered; the eager path names the
  // row by its position past the session's).
  const bool chain = decode_row && head_rows > 0 && draft_row_ > 0 &&
                     (capture ? T == groups : first_pos >= session_pos_[static_cast<size_t>(req)]);
  if (chain) {
    draft_chain_row(req, tokens, capture, head_rows, batch_requests);
    return;
  }
  const int64_t* d_pos = d_step_pos_;
  const int32_t* d_req = d_req_ids_;
  if (!decode_row) {
    std::vector<int64_t> pos(static_cast<size_t>(T));
    std::vector<int32_t> ids(static_cast<size_t>(T), req);
    for (int i = 0; i < T; ++i) pos[static_cast<size_t>(i)] = first_pos + i;
    DGPP_CUDA_OK(cudaMemcpyAsync(d_draft_pos_, pos.data(), static_cast<size_t>(T) * 8, cudaMemcpyHostToDevice, stream_));
    DGPP_CUDA_OK(cudaMemcpyAsync(d_draft_req_, ids.data(), static_cast<size_t>(T) * 4, cudaMemcpyHostToDevice, stream_));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
    d_pos = d_draft_pos_;
    d_req = d_draft_req_;
  }
  draft_first(req, tokens, d_pos, d_req, T, decode_row, capture, head_rows, batch_requests, src_row0);
}

void Dsv4Model::draft_first(int req, const int64_t* tokens, const int64_t* d_pos, const int32_t* d_req, int T,
                            bool decode_row, bool capture, int head_rows, int batch_requests, int src_row0) {
  const int H = cfg_.hidden_size;
  const int W = targets_ * H;
  const int block = cfg_.dspark_block_size;
  const int stages = cfg_.draft_stages;
  const float eps = cfg_.rms_norm_eps;
  // ---- the accepted rows' main hidden -> main_x --------------------------------
  // Decode rows gather from the slots' windows by position (a padding row,
  // pos -1, gathers zeros and appends nothing); prefill rows read the last
  // walk's rows in place.
  const uint16_t* src = main_hidden_ + static_cast<size_t>(src_row0) * W;
  if (decode_row) {
    gather_draft_hidden(d_req, d_pos, main_gather_, T);
    src = main_gather_;
  }
  {
    const Dsv4DraftResident& d0 = draft_stage(0);
    if (!d0.main_proj.payload || !d0.main_norm) throw std::runtime_error("mtp_run_rows: the DSpark main projection is unbound");
    launch_scale_gemm_grid_bf16(src, static_cast<size_t>(W), d0.main_proj.payload, d0.main_proj.scales, main_x_, T, H, W,
                                stream_, 0, kGrid, kGrid, dense_mma_);
    csa2_rmsnorm_bf16(main_x_, H, d0.main_norm, main_x_, H, T, H, eps, stream_);
  }
  // ---- the draft rings: each stage's window latent of the real rows ------------
  // A prefill chunk appends its last min(T, ring) rows (one ring slot per
  // row; the older rows are past every window).
  const int n = decode_row ? T : std::min(T, attn_cfg_.ring_slots);
  for (int s = 0; s < stages; ++s) {
    const Dsv4LayerResident& r = loader_.load_layer(cfg_.num_hidden_layers + s);
    build_layer_objects(r);
    attn_->append_window_rows(main_x_ + static_cast<size_t>(T - n) * H, pool_, d_req + (T - n), d_pos + (T - n), n,
                              stream_);
  }
  if (head_rows == 0) return;  // the prefill fills the rings; no block
  gemm_.set_decode_split_k(true);  // the block is a decode walk
  // ---- the block: [next, noise x (block - 1)] at the next block positions ---------
  const int rpg = T / (batch_requests > 0 ? batch_requests : 1);
  const int groups = T / rpg;
  const int rows = groups * block;
  if (rows > max_decode_rows_) throw std::invalid_argument("mtp_run_rows: the draft blocks exceed the decode batch");
  dsv41_dspark_block_rows(d_pos, tokens, d_req, groups, rpg, block, cfg_.dspark_noise_token_id, blk_pos_, blk_tok_,
                          blk_req_, blk_spans_, stream_);
  cur_ = streams_a_;
  nxt_ = streams_b_;
  // DGPP_DSV4_CAPTURE_DRAFT=1: the eager block walk keeps its sites (a
  // diagnostic: the parity localizer of the forward test).
  if (!capture && std::getenv("DGPP_DSV4_CAPTURE_DRAFT")) {
    debug_capture_ = true;
    debug_sites_.clear();
  }
  gather_embedding(blk_tok_, rows, capture);
  if (prefetching(true)) prefetch_attention_side(cfg_.num_hidden_layers);
  WalkRows wr;
  wr.tokens = blk_tok_;
  wr.req_ids = blk_req_;
  wr.pos = blk_pos_;
  wr.num_requests = groups;
  wr.req = req;
  wr.decode = true;
  wr.capture = capture;
  for (int s = 0; s < stages; ++s) {
    const Dsv4LayerResident& r = loader_.load_layer(cfg_.num_hidden_layers + s);
    wr.moe_table_slot = capture ? cfg_.num_hidden_layers + s : -1;
    build_layer_objects(r);
    if (!attn_->prepare(rows)) throw std::runtime_error("mtp_run_rows: attention GEMM plans unavailable");
    enqueue_layer(r, cfg_.num_hidden_layers + s, rows, wr);
  }
  debug_capture_ = false;
  // ---- the head over the block: the draft's own hc_head, its norm, the
  // shared lm head -> base_logits_ [rows, vocab slice] ---------------------------
  const Dsv4DraftResident& dl = draft_stage(stages - 1);
  if (!dl.norm || !dl.markov_embed || !dl.markov_head || !dl.confidence || !dl.hc_head.fn)
    throw std::runtime_error("mtp_run_rows: the DSpark head tensors are unbound");
  hc_head(dl.hc_head, draft_collapsed_, rows, /*decode=*/true);
  csa2_rmsnorm_bf16(draft_collapsed_, H, dl.norm, h_, H, rows, H, eps, stream_);
  // The screened head: the block's base logits off the FP8 copy (half the
  // bf16 head's bytes, which the verify rows already read once this pass);
  // each pick's biased row then has every logit within kDraftRescoreDelta
  // of its maximum — the only ones the pick can land on — recomputed from
  // the bf16 rows (draft_rescore). (The copy
  // ALONE, tried 2026-10-01: the read halves, 39.0 -> 38.4 ms per depth-3
  // pass, and the drafts get worse by more — paired on 40 greedy prompts,
  // 1.1 % more passes. The exact pass over the candidates keeps the drafts
  // the bf16 head's.) Without the copy (a shape the tensor-core form does
  // not take): the bf16 head itself.
  draft_screened_ = draft_head_fp8_ != nullptr && mma_gemv_shape_ok(draft_head_fp8_, h_, static_cast<size_t>(H), rows, H);
  if (draft_screened_)
    launch_mma_gemv_fp8_f32(h_, static_cast<size_t>(H), draft_head_fp8_, draft_head_scales_, base_logits_, rows,
                            lm_vocab_count_, H, static_cast<size_t>(lm_vocab_count_), 7, 7, stream_);
  else
    gemm_.matmul(h_, globals_.lm_head, base_logits_, rows, lm_vocab_count_, H, DType::BF16, GemmOut::F32,
                 static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
  // ---- block row 0, biased by the Markov head of `next`, into the head rows ----
  const int rows_out = head_rows / groups;
  dsv41_dspark_markov_bias(base_logits_, static_cast<int64_t>(block) * lm_vocab_count_, 0, dl.markov_embed, dl.markov_head,
                           cfg_.dspark_markov_rank, lm_vocab_begin_, lm_vocab_count_, blk_tok_, block, groups, logits_,
                           static_cast<int64_t>(rows_out) * lm_vocab_count_, rows_out, stream_,
                           draft_screened_ ? draft_row_max_ : nullptr);
  draft_rescore(0, groups, rows_out);
  dsv41_dspark_confidence(draft_collapsed_, static_cast<int64_t>(block) * H, 0, H, dl.markov_embed,
                          cfg_.dspark_markov_rank, blk_tok_, block, dl.confidence, groups,
                          conf_ + static_cast<size_t>(batch_requests > 0 ? 0 : req) * block, block, stream_);
  draft_row_ = 1;
  prefetch_.join(stream_);
  if (decode_row && (!capture || decode_tail_mirrors_) && head_rows <= max_decode_rows_)
    DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_logits_, logits_, static_cast<size_t>(head_rows) * lm_vocab_count_ * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
}

// The exact pass over a pick's candidates (see the screened head above):
// logits_ holds the pick's biased row, `rows_out` copies per group.
void Dsv4Model::draft_rescore(int block_row, int groups, int rows_out) {
  if (!draft_screened_) return;
  const int H = cfg_.hidden_size;
  const int block = cfg_.dspark_block_size;
  dsv41_dspark_rescore(logits_, static_cast<int64_t>(rows_out) * lm_vocab_count_, rows_out, base_logits_,
                       static_cast<int64_t>(block) * lm_vocab_count_, block_row, lm_vocab_count_, h_,
                       static_cast<int64_t>(block) * H, H, globals_.lm_head, draft_row_max_, kDraftRescoreDelta, groups,
                       stream_);
}

void Dsv4Model::draft_chain_row(int req, const int64_t* tokens, bool capture, int head_rows, int batch_requests) {
  const int H = cfg_.hidden_size;
  const int block = cfg_.dspark_block_size;
  const int groups = batch_requests > 0 ? batch_requests : 1;
  if (draft_row_ >= block) throw std::logic_error("mtp_run_rows: more chain rows than the block holds");
  if (head_rows != groups) throw std::invalid_argument("mtp_run_rows: a chain call heads one row per request");
  const Dsv4DraftResident& dl = draft_stage(cfg_.draft_stages - 1);
  const int row = draft_row_++;
  dsv41_dspark_markov_bias(base_logits_, static_cast<int64_t>(block) * lm_vocab_count_, row, dl.markov_embed, dl.markov_head,
                           cfg_.dspark_markov_rank, lm_vocab_begin_, lm_vocab_count_, tokens, 1, groups, logits_,
                           lm_vocab_count_, 1, stream_, draft_screened_ ? draft_row_max_ : nullptr);
  draft_rescore(row, groups, 1);
  dsv41_dspark_confidence(draft_collapsed_, static_cast<int64_t>(block) * H, row, H, dl.markov_embed,
                          cfg_.dspark_markov_rank, tokens, 1, dl.confidence, groups,
                          conf_ + static_cast<size_t>(batch_requests > 0 ? 0 : req) * block + row, block, stream_);
  if (!capture || decode_tail_mirrors_)
    DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_logits_, logits_, static_cast<size_t>(head_rows) * lm_vocab_count_ * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
}

// ---------------------------------------------------------------------------
// The decode walk's boundary prefetch windows (bit-identical on or off).
// ---------------------------------------------------------------------------
void Dsv4Model::prefetch_fp8(const GlmQuantMatrix& m) {
  if (m.payload) prefetch_.add(m.payload, static_cast<size_t>(m.rows) * static_cast<size_t>(m.cols));
  if (m.scales) prefetch_.add(m.scales, m.scale_bytes());
}

// An mHC site's coefficient matrix (bf16 [24, 4H]) and the sublayer's norm.
void Dsv4Model::prefetch_mhc(const uint16_t* fn, const uint16_t* norm) {
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  if (fn) prefetch_.add(fn, static_cast<size_t>(cfg_.hc_coeff_rows()) * 4 * H * 2);
  if (norm) prefetch_.add(norm, H * 2);
}

// Before the attention fold: this layer's MoE site — the mHC coefficients,
// the norm, the router and the shared expert's slices. The routed experts
// are known only behind the router.
void Dsv4Model::prefetch_ffn_side(const Dsv4LayerResident& r) {
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, kPrefetchWindowBytes, prefetch_.boundary_rate());
  prefetch_mhc(r.mhc.ffn_fn, r.ffn_norm);
  if (r.moe.router) prefetch_.add(r.moe.router, static_cast<size_t>(cfg_.n_routed_experts) * H * 2);
  if (r.moe.router_bias) prefetch_.add(r.moe.router_bias, static_cast<size_t>(cfg_.n_routed_experts) * 4);
  for (const GlmQuantMatrix& m : r.moe.shared) prefetch_fp8(m);
}

// Before the MoE fold (and at the walk's start for the first layer): the
// next layer's attention head — its mHC coefficients, the norm, wq_a, wkv
// and wq_b in consumption order, clamped to the window — or the model
// head's behind the last layer.
void Dsv4Model::prefetch_attention_side(int layer) {
  const Dsv4LayerResident& r = loader_.load_layer(layer);
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, kPrefetchWindowBytes, prefetch_.boundary_rate());
  prefetch_mhc(r.mhc.attn_fn, r.attn_norm);
  prefetch_fp8(r.attn.wq_a);
  if (r.attn.q_norm) prefetch_.add(r.attn.q_norm, static_cast<size_t>(cfg_.q_lora_rank) * 2);
  prefetch_fp8(r.attn.wkv);
  prefetch_fp8(r.attn.wq_b);
  (void)H;
}

// The head: the collapse's coefficients, the norm, the lm head's first rows.
void Dsv4Model::prefetch_head(const Dsv4HcHeadResident& hc, const uint16_t* norm, bool draft) {
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, kPrefetchWindowBytes, prefetch_.boundary_rate());
  prefetch_mhc(hc.fn, norm);
  if (draft && draft_head_fp8_ != nullptr)
    prefetch_.add(draft_head_fp8_, static_cast<size_t>(lm_vocab_count_) * H);
  else if (globals_.lm_head)
    prefetch_.add(globals_.lm_head, static_cast<size_t>(lm_vocab_count_) * H * 2);
}

void Dsv4Model::debug_drop_draft_screen() {
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  cudaFree(draft_head_fp8_);
  cudaFree(draft_head_scales_);
  draft_head_fp8_ = nullptr;
  draft_head_scales_ = nullptr;
  draft_screened_ = false;
}

// The draft head's screen: the local lm head slice as e4m3 codes on
// power-of-two scales per 128 x 128 block (the release's own FP8 layout),
// quantized once at load. Left unbuilt where the tensor-core GEMV does not
// take the shape (the draft head then reads the bf16 rows).
void Dsv4Model::build_draft_head_screen() {
  const size_t V = static_cast<size_t>(lm_vocab_count_), H = static_cast<size_t>(cfg_.hidden_size);
  if (globals_.lm_head == nullptr || H % 128 != 0 || (reinterpret_cast<uintptr_t>(globals_.lm_head) & 15u) != 0) return;
  std::vector<uint16_t> w(V * H);
  DGPP_CUDA_OK(cudaMemcpy(w.data(), globals_.lm_head, V * H * 2, cudaMemcpyDeviceToHost));
  const size_t sr = (V + 127) / 128, sc = H / 128;
  std::vector<float> scales(sr * sc);
  std::vector<uint8_t> codes(V * H);
  const auto band = [&](size_t br0, size_t br1) {
    for (size_t br = br0; br < br1; ++br) {
      const size_t r0 = br * 128, r1 = std::min(V, r0 + 128);
      for (size_t bc = 0; bc < sc; ++bc) {
        float amax = 0.f;
        for (size_t r = r0; r < r1; ++r)
          for (size_t c = bc * 128; c < bc * 128 + 128; ++c) amax = std::max(amax, std::fabs(bf16_bits_to_float(w[r * H + c])));
        const float s = amax > 0.f ? std::exp2(std::ceil(std::log2(amax / 448.0f))) : 1.0f;
        scales[br * sc + bc] = s;
        const float inv = 1.0f / s;
        for (size_t r = r0; r < r1; ++r)
          for (size_t c = bc * 128; c < bc * 128 + 128; ++c)
            codes[r * H + c] = float_to_fp8_e4m3_bits(bf16_bits_to_float(w[r * H + c]) * inv);
      }
    }
  };
  const size_t threads = std::max<size_t>(1, std::min<size_t>({sr, 16, std::thread::hardware_concurrency()}));
  std::vector<std::thread> pool;
  for (size_t t = 0; t < threads; ++t) pool.emplace_back(band, sr * t / threads, sr * (t + 1) / threads);
  for (auto& t : pool) t.join();
  draft_head_fp8_ = dev_alloc<uint8_t>(V * H);
  draft_head_scales_ = dev_alloc<float>(sr * sc);
  DGPP_CUDA_OK(cudaMemcpy(draft_head_fp8_, codes.data(), V * H, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(draft_head_scales_, scales.data(), sr * sc * sizeof(float), cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------
// The walk.
// ---------------------------------------------------------------------------
uint16_t* Dsv4Model::stage(uint16_t* fallback, int T, int width, bool capture) {
  if (!boundary_) return fallback;
  uint16_t* s = boundary_->stage(T, width);
  if (s == nullptr && capture) throw std::runtime_error("run_rows: a capture fold does not fit the recorder's staged buffer");
  return s ? s : fallback;
}

void Dsv4Model::fold(uint16_t* buf, int T, int width, bool capture) {
  if (!boundary_) return;
  // The host-driven reducer folds after the producing kernels have
  // quiesced; the stream-ordered one launches the fold on this stream
  // behind them and needs no drain.
  if (!capture && !boundary_->stream_ordered()) DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  boundary_->reduce(buf, T, width);
}

// The embedding on all four streams: the whole table's gather straight
// into the streams, or — vocab-sharded — this rank's rows, one fold, and
// the rows broadcast to the streams.
void Dsv4Model::gather_embedding(const int64_t* tokens, int T, bool capture) {
  const int H = cfg_.hidden_size;
  if (!embed_sharded_) {
    glm_embed_bcast_streams(globals_.embed, tokens, cur_, T, H, stream_);
    return;
  }
  uint16_t* e = stage(x_, T, H, capture);
  embed_gather_sliced_bf16(globals_.embed, tokens, e, T, H, globals_.embed_vocab_begin, globals_.embed_vocab_count, stream_);
  fold(e, T, H, capture);
  for (int s = 0; s < 4; ++s)
    DGPP_CUDA_OK(cudaMemcpy2DAsync(cur_ + static_cast<size_t>(s) * H, static_cast<size_t>(4) * H * 2, e,
                                   static_cast<size_t>(H) * 2, static_cast<size_t>(H) * 2, static_cast<size_t>(T),
                                   cudaMemcpyDefault, stream_));
}

// One mHC site (the reference's two-pass hc_pre): this site's coefficients
// from the streams, the collapse with its OWN pre, the sublayer's
// one-rounding norm into x_; post / comb in fp32 for the stream update.
void Dsv4Model::mhc_site(const uint16_t* streams, const GlmMhcWeights& w, const uint16_t* ln, int T, bool decode) {
  MhcSinglePass sp;
  sp.pre_in = nullptr;
  sp.pre_out = nullptr;
  sp.post_f32 = post_f32_;
  sp.comb_f32 = comb_f32_;
  if (decode && mhc_counters_ != nullptr) {
    // The fused decode form: one launch for the dots, the finish and the
    // sublayer's norm; the comb on the side stream (joined by stream_update).
    sp.comb_f32 = nullptr;
    sp.one_round_norm = true;
    const bool deferred =
        launch_mhc_compute_normed(streams, w, mhc_cfg_, nullptr, post_bf16_, comb_bf16_, mhc_logits_, ln, x_,
                                  cfg_.rms_norm_eps, T, stream_, mhc_counters_, /*defer_comb=*/true, &sp, true);
    if (!deferred) throw std::logic_error("mhc_site: the fused decode form must defer its comb");
    DGPP_CUDA_OK(cudaEventRecord(mhc_fork_, stream_));
    DGPP_CUDA_OK(cudaStreamWaitEvent(mhc_side_, mhc_fork_, 0));
    launch_mhc_comb(mhc_logits_, w, mhc_cfg_, comb_bf16_, T, mhc_side_, comb_f32_);
    DGPP_CUDA_OK(cudaEventRecord(mhc_join_, mhc_side_));
    comb_pending_ = true;
    return;
  }
  // Decode rows stay on the per-coefficient form at any count (a batched
  // row bitwise the row alone; the >= 16-token prefill forms are not).
  (void)launch_mhc_compute_normed(streams, w, mhc_cfg_, collapsed_, post_bf16_, comb_bf16_, mhc_logits_, nullptr, nullptr,
                                  cfg_.rms_norm_eps, T, stream_, nullptr, false, &sp, decode);
  csa2_rmsnorm_bf16(collapsed_, cfg_.hidden_size, ln, x_, cfg_.hidden_size, T, cfg_.hidden_size, cfg_.rms_norm_eps, stream_);
}

// The head's collapse: pre = sigmoid(mix * scale + base) + eps over the
// streams — the mHC site's pre rule on the padded head weights (the post /
// comb outputs are scratch).
void Dsv4Model::hc_head(const Dsv4HcHeadResident& hw, uint16_t* out, int T, bool decode) {
  GlmMhcWeights w;
  w.fn = hw.fn;
  w.base = hw.base;
  w.scale = hw.scale;
  (void)launch_mhc_compute_normed(cur_, w, mhc_cfg_, out, post_bf16_, comb_bf16_, mhc_logits_, nullptr, nullptr,
                                  cfg_.rms_norm_eps, T, stream_, nullptr, false, nullptr, decode);
}

void Dsv4Model::stream_update(const uint16_t* sublayer_out, int T) {
  if (comb_pending_) {
    DGPP_CUDA_OK(cudaStreamWaitEvent(stream_, mhc_join_, 0));
    comb_pending_ = false;
  }
  launch_mhc_stream_update_f32(post_f32_, comb_f32_, sublayer_out, cur_, nxt_, mhc_cfg_, T, stream_);
  std::swap(cur_, nxt_);
}

// ---- the prefill's fold overlap (model.hpp) -------------------------------------
void Dsv4Model::mhc_site_rows(const uint16_t* streams, const GlmMhcWeights& w, const uint16_t* ln, int r0, int rows) {
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  const size_t n = static_cast<size_t>(mhc_cfg_.hc_mult), R = static_cast<size_t>(r0);
  MhcSinglePass sp;
  sp.pre_in = nullptr;
  sp.pre_out = nullptr;
  sp.post_f32 = post_f32_ + R * n;
  sp.comb_f32 = comb_f32_ + R * n * n;
  (void)launch_mhc_compute_normed(streams + R * n * H, w, mhc_cfg_, collapsed_ + R * H, post_bf16_ + R * n,
                                  comb_bf16_ + R * n * n, mhc_logits_ + R * static_cast<size_t>(mhc_cfg_.coeff_rows()),
                                  nullptr, nullptr, cfg_.rms_norm_eps, rows, stream_, nullptr, false, &sp, false);
  csa2_rmsnorm_bf16(collapsed_ + R * H, cfg_.hidden_size, ln, x_ + R * H, cfg_.hidden_size, rows, cfg_.hidden_size,
                    cfg_.rms_norm_eps, stream_);
}

void Dsv4Model::update_rows(const uint16_t* sublayer_out, int r0, int rows) {
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  const size_t n = static_cast<size_t>(mhc_cfg_.hc_mult), R = static_cast<size_t>(r0);
  launch_mhc_stream_update_f32(post_f32_ + R * n, comb_f32_ + R * n * n, sublayer_out + R * H, cur_ + R * n * H,
                               nxt_ + R * n * H, mhc_cfg_, rows, stream_);
}

// Block B's outstanding FFN fold lands: its stream update, the swap.
void Dsv4Model::flush_overlap(int T, int TA) {
  if (!ffn_b_pending_) return;
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  boundary_->end_async();
  update_rows(y_, TA, T - TA);
  std::swap(cur_, nxt_);
  ffn_b_pending_ = false;
}

void Dsv4Model::enqueue_layer_overlap(const Dsv4LayerResident& r, int layer, int T, int TA, const WalkRows& rows,
                                      bool flush) {
  const int H = cfg_.hidden_size;
  const int TB = T - TA;
  build_layer_objects(r);
  const auto drain = [&] { DGPP_CUDA_OK(cudaStreamSynchronize(stream_)); };
  // A fold of rows [r0, r0 + n) of `buf` begun (true) or done in place.
  const auto begin_fold = [&](uint16_t* buf, int r0, int n) {
    uint16_t* at = buf + static_cast<size_t>(r0) * H;
    drain();
    if (boundary_->begin_async(at, n, H)) {
      ++overlap_async_folds_;
      return true;
    }
    boundary_->reduce(at, n, H);
    return false;
  };
  const auto end_fold = [&](bool begun) {
    if (!begun) return;
    drain();
    boundary_->end_async();
  };
  // ---- the attention site ----------------------------------------------------
  GlmMhcWeights aw;
  aw.fn = r.mhc.attn_fn;
  aw.base = r.mhc.attn_base;
  aw.scale = r.mhc.attn_scale;
  // Block A's streams are updated (in nxt_) while block B's wait for the
  // previous layer's FFN fold.
  if (ffn_b_pending_)
    mhc_site_rows(nxt_, aw, r.attn_norm, 0, TA);
  else
    mhc_site_rows(cur_, aw, r.attn_norm, 0, T);
  uint16_t* attn_out = y_;  // rows [TA, T) may still hold the previous FFN's block B: A writes [0, TA) only
  // The attention of walk rows [r0, r0 + n): one request's rows, or the
  // part of every span of a group inside them (a span the block boundary
  // cuts runs in two calls — split-invariant, as at any chunk cut).
  const auto attention = [&](int r0, int n) {
    if (rows.num_spans == 0) {
      if (!attn_->prepare(n)) throw std::runtime_error("run_rows: attention GEMM plans unavailable");
      attn_->enqueue_prefill(x_ + static_cast<size_t>(r0) * H, pool_, rows.req, rows.pos0 + r0, n,
                             attn_out + static_cast<size_t>(r0) * H, stream_);
      return;
    }
    int span0 = 0;
    for (int sp = 0; sp < rows.num_spans; ++sp) {
      const int len = rows.span_lens[sp];
      const int lo = std::max(span0, r0), hi = std::min(span0 + len, r0 + n);
      if (hi > lo) {
        if (!attn_->prepare(hi - lo)) throw std::runtime_error("run_rows: attention GEMM plans unavailable");
        attn_->enqueue_prefill(x_ + static_cast<size_t>(lo) * H, pool_, rows.span_reqs[sp],
                               rows.span_pos0[sp] + (lo - span0), hi - lo, attn_out + static_cast<size_t>(lo) * H,
                               stream_);
      }
      span0 += len;
    }
  };
  attention(0, TA);
  if (ffn_b_pending_) {
    flush_overlap(T, TA);
    mhc_site_rows(cur_, aw, r.attn_norm, TA, TB);
  }
  const bool fold_a = begin_fold(attn_out, 0, TA);
  attention(TA, TB);
  end_fold(fold_a);
  // ---- the MoE site: A's update and site under B's attention fold ---------------
  GlmMhcWeights fw;
  fw.fn = r.mhc.ffn_fn;
  fw.base = r.mhc.ffn_base;
  fw.scale = r.mhc.ffn_scale;
  if (r.moe.tid2eid != nullptr) moe_->set_hash_routing(r.moe.tid2eid, rows.tokens, cfg_.vocab_size);
  const bool fold_b = begin_fold(attn_out, TA, TB);
  update_rows(attn_out, 0, TA);
  mhc_site_rows(nxt_, fw, r.ffn_norm, 0, TA);
  moe_->route_prefill_rows(x_, 0, TA, stream_);  // A's router too, under B's fold
  end_fold(fold_b);
  update_rows(attn_out, TA, TB);
  std::swap(cur_, nxt_);
  mhc_site_rows(cur_, fw, r.ffn_norm, TA, TB);
  moe_->route_prefill_rows(x_, TA, TB, stream_);
  // The expert chain over every row (its cost is the weights it decodes);
  // then the accumulation per block, so block B's runs under A's fold.
  uint16_t* ffn_out = y_;
  moe_->enqueue_prefill_phased(x_, T, stream_);
  moe_->set_hash_routing(nullptr, nullptr, 0);
  moe_->accumulate_prefill_rows(ffn_out, T, 0, TA, stream_);
  const bool ffn_a = begin_fold(ffn_out, 0, TA);
  moe_->accumulate_prefill_rows(ffn_out, T, TA, TB, stream_);
  end_fold(ffn_a);
  // Block A's update; block B's fold under the next layer's site and
  // attention of A (or landed here: the walk's last layer, a DSpark
  // target layer).
  update_rows(ffn_out, 0, TA);
  const bool target = mtp_ && target_ordinal(layer) >= 0;
  ffn_b_pending_ = begin_fold(ffn_out, TA, TB);
  if (!ffn_b_pending_) {
    update_rows(ffn_out, TA, TB);
    std::swap(cur_, nxt_);
  } else if (flush || target) {
    flush_overlap(T, TA);
  }
  if (target)
    dsv41_stream_mean_bf16(cur_, 4, H, T, main_hidden_ + static_cast<size_t>(target_ordinal(layer)) * H,
                           static_cast<int64_t>(targets_) * H, stream_);
}

void Dsv4Model::enqueue_layer(const Dsv4LayerResident& r, int layer, int T, const WalkRows& rows) {
  const int H = cfg_.hidden_size;
  build_layer_objects(r);
  const bool draft = cfg_.is_draft(layer);
  // ---- the attention site ----------------------------------------------------
  GlmMhcWeights aw;
  aw.fn = r.mhc.attn_fn;
  aw.base = r.mhc.attn_base;
  aw.scale = r.mhc.attn_scale;
  mhc_site(cur_, aw, r.attn_norm, T, rows.decode);
  const auto grab = [&](const uint16_t* p, size_t n) {
    std::vector<uint16_t> v(n);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
    DGPP_CUDA_OK(cudaMemcpy(v.data(), p, n * 2, cudaMemcpyDeviceToHost));
    return v;
  };
  SiteCapture cap;
  if (debug_capture_) cap.x_attn = grab(x_, static_cast<size_t>(T) * H);
  uint16_t* attn_out = stage(y_, T, H, rows.capture);
  if (!attn_->prepare(T)) throw std::runtime_error("run_rows: attention GEMM plans unavailable");
  if (draft) {
    attn_->enqueue_draft_block(x_, pool_, rows.req_ids, rows.pos, T, cfg_.dspark_block_size, attn_out, stream_);
  } else if (rows.decode) {
    attn_->enqueue_decode(x_, pool_, rows.req_ids, rows.pos, T, attn_out, stream_);
  } else if (rows.num_spans > 0) {
    int row0 = 0;
    for (int s = 0; s < rows.num_spans; ++s) {
      const int len = rows.span_lens[s];
      if (!attn_->prepare(len)) throw std::runtime_error("run_rows: attention GEMM plans unavailable");
      attn_->enqueue_prefill(x_ + static_cast<size_t>(row0) * H, pool_, rows.span_reqs[s], rows.span_pos0[s], len,
                             attn_out + static_cast<size_t>(row0) * H, stream_);
      row0 += len;
    }
  } else {
    attn_->enqueue_prefill(x_, pool_, rows.req, rows.pos0, T, attn_out, stream_);
  }
  const bool pf = prefetching(rows.decode);
  if (pf) prefetch_ffn_side(r);
  fold(attn_out, T, H, rows.capture);  // block boundary 1: wo_b's partial
  if (debug_capture_) cap.attn_out = grab(attn_out, static_cast<size_t>(T) * H);
  stream_update(attn_out, T);
  if (debug_capture_) cap.streams_after_attn = grab(cur_, static_cast<size_t>(T) * 4 * H);
  // ---- the MoE site ------------------------------------------------------------
  GlmMhcWeights fw;
  fw.fn = r.mhc.ffn_fn;
  fw.base = r.mhc.ffn_base;
  fw.scale = r.mhc.ffn_scale;
  mhc_site(cur_, fw, r.ffn_norm, T, rows.decode);
  if (debug_capture_) cap.x_ffn = grab(x_, static_cast<size_t>(T) * H);
  uint16_t* ffn_out = stage(y_, T, H, rows.capture);
  // A hash layer's experts come from its token table (the rows' tokens).
  if (r.moe.tid2eid != nullptr) moe_->set_hash_routing(r.moe.tid2eid, rows.tokens, cfg_.vocab_size);
  if (rows.decode)
    moe_->enqueue_decode(x_, ffn_out, T, nullptr, stream_, rows.moe_table_slot);
  else
    moe_->enqueue_prefill(x_, ffn_out, T, rows.trace, stream_);
  moe_->set_hash_routing(nullptr, nullptr, 0);
  if (pf) {
    // The next layer of the walk: a backbone layer's successor, a draft
    // stage's next stage, or the head behind the last of either.
    const int last = draft ? cfg_.max_layer() - 1 : cfg_.num_hidden_layers - 1;
    if (layer < last) {
      prefetch_attention_side(layer + 1);
    } else if (draft) {
      const Dsv4DraftResident& dl = r.draft;
      prefetch_head(dl.hc_head, dl.norm, /*draft=*/true);
    } else {
      prefetch_head(globals_.hc_head, globals_.final_norm, /*draft=*/false);
    }
  }
  fold(ffn_out, T, H, rows.capture);  // block boundary 2: the sliced experts
  if (debug_capture_) cap.ffn_out = grab(ffn_out, static_cast<size_t>(T) * H);
  stream_update(ffn_out, T);
  if (debug_capture_) debug_sites_.push_back(std::move(cap));
  // DSpark: the target layers' OUTPUT, its stream mean, per row into
  // [h_t1 | h_t2 | h_t3] (the reference appends h.mean(dim=2) after the layer).
  if (mtp_ && !draft) {
    const int ord = target_ordinal(layer);
    if (ord >= 0)
      dsv41_stream_mean_bf16(cur_, 4, H, T, main_hidden_ + static_cast<size_t>(ord) * H,
                             static_cast<int64_t>(targets_) * H, stream_);
  }
}

Dsv4Model::Outputs Dsv4Model::run_rows(const RowRun& run) {
  const int T = run.T, req = run.req;
  if (run.capture && loader_.residency() != Dsv4Residency::Resident)
    throw std::logic_error("run_rows: a capture needs a resident stack");
  const int H = cfg_.hidden_size;
  const RowInputs in = begin_run(run);
  cur_ = streams_a_;
  nxt_ = streams_b_;
  ffn_b_pending_ = false;
  // The decode walks' bf16 projections split their k range across the part
  // (a prefill chunk keeps the unsplit chain: bitwise at any cut).
  gemm_.set_decode_split_k(run.decode);
  gather_embedding(in.tokens, T, run.capture);
  if (prefetching(run.decode)) prefetch_attention_side(0);
  Outputs out;
  const bool traces = !run.decode && route_traces_;
  const size_t K = static_cast<size_t>(cfg_.num_experts_per_tok);
  WalkRows rows;
  rows.tokens = in.tokens;
  rows.req_ids = in.req_ids;
  rows.pos = in.pos;
  rows.num_requests = in.num_requests;
  rows.req = req;
  rows.pos0 = run.pos0;
  rows.decode = run.decode;
  rows.capture = run.capture;
  rows.span_reqs = run.span_reqs;
  rows.span_pos0 = run.span_pos0;
  rows.span_lens = run.span_lens;
  rows.num_spans = run.num_spans;
  // The group prefill: several requests' cold prompts as the spans of one
  // walk. The dense sites, the MoE, the norms and the head are per row; the
  // attention (its compressors, selections and window scratch) runs per
  // span. Not captured; a span starts at any position its request's state
  // stands at (0: a cold prompt; a cut: the next chunk of a prompt the
  // scheduler reads in beside others).
  const bool group = run.num_spans > 0;
  std::vector<int> span_row0;
  if (group) {
    if (run.decode || run.capture) throw std::logic_error("run_rows: a group prefill is neither a decode nor a capture");
    int at = 0;
    for (int s = 0; s < run.num_spans; ++s) {
      const int len = run.span_lens[s];
      if (len <= 0 || len > prefill_group_span_limit())
        throw std::invalid_argument("run_rows: a group prefill span of " + std::to_string(len) +
                                    " rows exceeds the span limit " + std::to_string(prefill_group_span_limit()));
      if (run.span_pos0[s] < 0) throw std::invalid_argument("run_rows: a group prefill span at a negative position");
      span_row0.push_back(at);
      at += len;
    }
    if (at != T) throw std::invalid_argument("run_rows: the group's spans do not cover the walk's rows");
  }
  // DGPP_DSV4_CAPTURE_DECODE=1 / DGPP_DSV4_CAPTURE_PREFILL=1: an eager decode
  // walk / a session prefill's chunks keep every layer's rows too
  // (diagnostics: the decode-vs-prefill localizer of the model test).
  const bool capture_decode_env = std::getenv("DGPP_DSV4_CAPTURE_DECODE") != nullptr;
  const bool capture_prefill_env = std::getenv("DGPP_DSV4_CAPTURE_PREFILL") != nullptr;
  const bool capture_layers = run.capture_layers || (run.decode && !run.capture && capture_decode_env) ||
                              (!run.decode && !run.capture && capture_prefill_env);
  debug_capture_ = capture_layers;
  debug_sites_.clear();
  debug_index_logits_.clear();
  // The teacher's streams replace the live ones before a layer (the
  // diagnostic forward only; the walk is synchronized per layer there).
  const auto teach = [&](int i) {
    const std::vector<uint16_t>& src = (*debug_teacher_)[static_cast<size_t>(i)];
    if (src.size() != static_cast<size_t>(T) * 4 * H) throw std::invalid_argument("forward: teacher streams shape");
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
    DGPP_CUDA_OK(cudaMemcpy(cur_, src.data(), src.size() * 2, cudaMemcpyHostToDevice));
  };
  std::vector<int> trace_rows;
  bool head = true;
  const int last_layer = (debug_layer_limit_ > 0 && debug_layer_limit_ < cfg_.num_hidden_layers && !run.decode)
                             ? debug_layer_limit_
                             : cfg_.num_hidden_layers;
  if (last_layer < cfg_.num_hidden_layers) head = false;  // the limited diagnostic walk: no head
  // The fold overlap's block boundary: the walk row nearest the middle at
  // which the attention may start a chunk — a span's start, or a row of a
  // span on its request's block grid (the attention takes a chunk at a
  // block boundary; a group's spans begin at any walk row, so the middle
  // row itself may stand inside a block of its request) — with at least a
  // block of rows on either side. 0: none (the one-block walk).
  int overlap_boundary = 0;
  {
    const auto consider = [&](int c) {
      if (c < kBlockTokens || T - c < kBlockTokens) return;
      if (overlap_boundary == 0 || std::abs(c - T / 2) < std::abs(overlap_boundary - T / 2)) overlap_boundary = c;
    };
    if (!group) {
      if (run.pos0 % kBlockTokens == 0) consider((T / 2 / kBlockTokens) * kBlockTokens);
    } else {
      for (int s = 0; s < run.num_spans; ++s) {
        const int row0 = span_row0[static_cast<size_t>(s)];
        consider(row0);
        if (run.span_pos0[s] % kBlockTokens != 0) continue;
        for (int off = kBlockTokens; off < run.span_lens[s]; off += kBlockTokens) consider(row0 + off);
      }
    }
  }
  for (int layer = 0; layer < last_layer; ++layer) {
    const Dsv4LayerResident& r = loader_.load_layer(layer);
    if (debug_teacher_ && layer > 0) teach(layer - 1);
    MoeTraceStaging trace;
    rows.trace = nullptr;
    rows.moe_table_slot = run.capture ? layer : -1;
    if (traces) {
      const size_t slot = static_cast<size_t>(layer) * static_cast<size_t>(max_tokens_) * K;
      trace.ids = h_route_ids_ + slot;
      trace.weights = h_route_weights_ + slot;
      trace.biased = nullptr;
      rows.trace = &trace;
      out.route_ids.emplace_back();
      out.route_weights.emplace_back();
    }
    // The fold overlap: a prefill walk (one request's chunk or a group's
    // spans) on a multi-rank world, never a capture or a diagnostic walk.
    const int overlap_ta = (fold_overlap_ && !run.decode && !run.capture && !capture_layers && debug_teacher_ == nullptr &&
                            boundary_ != nullptr && !traces && T >= kFoldOverlapMinRows)
                               ? overlap_boundary
                               : 0;
    if (overlap_ta > 0)
      enqueue_layer_overlap(r, layer, T, overlap_ta, rows, /*flush=*/layer + 1 == last_layer);
    else
      enqueue_layer(r, layer, T, rows);
    trace_rows.push_back(T);
    if (capture_layers) {
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      std::vector<uint16_t> snap(static_cast<size_t>(T) * 4 * H);
      DGPP_CUDA_OK(cudaMemcpy(snap.data(), cur_, snap.size() * 2, cudaMemcpyDeviceToHost));
      out.layer_states.push_back(std::move(snap));
      if (cfg_.indexed(layer) && !group) {
        std::vector<int32_t> sel(static_cast<size_t>(T) * static_cast<size_t>(cfg_.index_topk));
        DGPP_CUDA_OK(cudaMemcpy(sel.data(), attn_->debug_topk(), sel.size() * 4, cudaMemcpyDeviceToHost));
        out.dsa_selections.push_back(std::move(sel));
        IndexLogits lg;
        lg.rows = run.decode ? 0 : attn_->debug_logits_rows();
        lg.stride = attn_->debug_logits_stride();
        lg.entries = attn_->debug_logits_entries();
        if (lg.rows > 0) {
          lg.values.resize(static_cast<size_t>(lg.rows) * static_cast<size_t>(lg.stride));
          DGPP_CUDA_OK(cudaMemcpy(lg.values.data(), attn_->debug_logits(), lg.values.size() * 4, cudaMemcpyDeviceToHost));
        }
        lg.q_rows = T;
        lg.q_heads = attn_cfg_.index_heads;
        lg.q_codes.resize(static_cast<size_t>(T) * static_cast<size_t>(lg.q_heads) * kDsv4IndexDim);
        lg.q_scales.resize(static_cast<size_t>(T) * static_cast<size_t>(lg.q_heads));
        DGPP_CUDA_OK(cudaMemcpy(lg.q_codes.data(), attn_->debug_index_q_codes(), lg.q_codes.size(), cudaMemcpyDeviceToHost));
        DGPP_CUDA_OK(cudaMemcpy(lg.q_scales.data(), attn_->debug_index_q_scales(), lg.q_scales.size() * 4,
                                cudaMemcpyDeviceToHost));
        debug_index_logits_.push_back(std::move(lg));
      }
    }
  }
  const bool packed_logits = packed_prefill_logits(run);
  if (head) {
    // The head: the head's own collapse, the final norm, the lm head.
    if (debug_teacher_ && debug_teacher_->size() >= static_cast<size_t>(cfg_.num_hidden_layers))
      teach(cfg_.num_hidden_layers - 1);
    hc_head(globals_.hc_head, collapsed_, T, run.decode);
    csa2_rmsnorm_bf16(collapsed_, H, globals_.final_norm, h_, H, T, H, cfg_.rms_norm_eps, stream_);
    if (packed_logits) {
      project_head_rows(run, [&](int input_row, int output_row, int count) {
        gemm_.matmul(h_ + static_cast<size_t>(input_row) * H, globals_.lm_head,
                     logits_ + static_cast<size_t>(output_row) * lm_vocab_count_, count, lm_vocab_count_, H,
                     DType::BF16, GemmOut::F32, static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
      });
    } else {
      gemm_.matmul(h_, globals_.lm_head, logits_, T, lm_vocab_count_, H, DType::BF16, GemmOut::F32,
                   static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
    }
    // DSpark: the last rows' target hidden into the slots' windows by
    // position (the draft gathers its accepted rows' from there); a
    // prefill's walked rows are the draft's rows (mtp_run_rows).
    if (mtp_) {
      const size_t W = static_cast<size_t>(targets_) * H;
      if (group) {
        for (int s = 0; s < run.num_spans; ++s) {
          const int len = run.span_lens[s], n = std::min(len, max_decode_rows_);
          const int last = span_row0[static_cast<size_t>(s)] + len - n;
          store_draft_hidden(main_hidden_ + static_cast<size_t>(last) * W, in.req_ids + last, in.pos + last, n);
        }
      } else {
        const int n = std::min(T, max_decode_rows_);
        store_draft_hidden(main_hidden_ + static_cast<size_t>(T - n) * W, in.req_ids + (T - n), in.pos + (T - n), n);
      }
      draft_row_ = 0;
    }
    if (!run.decode) {
      if (group) {
        for (int s = 0; s < run.num_spans; ++s)
          segment_of(run.span_reqs[s]) = {run.span_pos0[s], run.span_lens[s], span_row0[static_cast<size_t>(s)]};
      } else {
        segment_of(req) = {run.pos0, T, 0};
      }
    }
  }
  // The prefetch side stream rejoins the walk (a capture ends on one stream).
  if (run.decode) prefetch_.join(stream_);
  // The stream-ordered reducer's verdict for this pass's folds (a failed
  // collective is an error here, not a wrong number read later).
  if (!run.capture && boundary_) boundary_->settle();
  out = finish_run(run, std::move(out), packed_logits);
  if (!run.capture && traces) {
    for (size_t l = 0; l < out.route_ids.size(); ++l) {
      const size_t slot = l * static_cast<size_t>(max_tokens_) * K;
      const size_t n = static_cast<size_t>(trace_rows[l]) * K;
      out.route_ids[l].assign(h_route_ids_ + slot, h_route_ids_ + slot + n);
      out.route_weights[l].assign(h_route_weights_ + slot, h_route_weights_ + slot + n);
    }
  }
  return out;
}

Dsv4Model::Outputs Dsv4Model::forward(const std::vector<int64_t>& token_ids, bool capture_layers,
                                      const std::vector<std::vector<uint16_t>>* teacher) {
  const int T = static_cast<int>(token_ids.size());
  if (T <= 0) throw std::invalid_argument("forward: empty token batch");
  if (teacher && !capture_layers) throw std::invalid_argument("forward: a teacher-forced walk captures its layers");
  if (teacher && teacher->size() + 1 < static_cast<size_t>(cfg_.num_hidden_layers))
    throw std::invalid_argument("forward: the teacher covers fewer layers than the model");
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
  debug_teacher_ = teacher;
  Outputs out;
  try {
    out = run_rows(run);
  } catch (...) {
    debug_teacher_ = nullptr;
    throw;
  }
  debug_teacher_ = nullptr;
  session_close(0);
  return out;
}

}  // namespace dgpp
