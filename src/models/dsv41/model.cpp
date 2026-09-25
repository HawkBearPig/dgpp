#include "models/dsv41/model.hpp"

#include "kernels/dsv41_dspark.hpp"
#include "kernels/scale_gemm.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/kernels.hpp"

namespace dgpp {
using session_detail::dev_alloc;
using session_detail::pinned_alloc;

// ---------------------------------------------------------------------------
// The shapes.
// ---------------------------------------------------------------------------
Csa2Config Dsv41Model::csa2_config(const Dsv41TextConfig& cfg, int tp_world) {
  Csa2Config c;
  c.hidden = cfg.hidden_size;
  c.q_lora = cfg.q_lora_rank;
  c.o_lora = cfg.o_lora_rank;
  c.num_heads = cfg.num_attention_heads;
  c.o_groups = cfg.o_groups;
  c.index_heads = cfg.index_n_heads;
  c.index_topk = cfg.index_topk;
  c.candidate_block = cfg.candidate_block_size > 0 ? cfg.candidate_block_size : 8;
  c.candidate_blocks = cfg.candidate_topk_blocks > 0 ? cfg.candidate_topk_blocks : 2048;
  c.window = cfg.sliding_window;
  c.ring_slots = std::max(kRingSlots, cfg.sliding_window + 32);
  c.block_tokens = kBlockTokens;
  c.eps = cfg.rms_norm_eps;
  c.tp = tp_world;
  Csa2Config::validate(c);
  return c;
}

Csa2PoolShape Dsv41Model::pool_shape(const Dsv41TextConfig& cfg, int max_requests, int64_t cache_tokens) {
  Csa2PoolShape s;
  s.layers = cfg.max_layer();
  int tails = 0;
  for (const int l : cfg.kv_source_layer_ids) {
    s.cache_ratio.push_back(cfg.compress_ratio(l));
    if (cfg.compress_ratio(l) == 2) ++tails;
  }
  s.tail_ordinals = tails;
  s.max_requests = max_requests;
  s.token_slots = cache_tokens;
  s.block_tokens = kBlockTokens;
  s.ring_slots = std::max(kRingSlots, cfg.sliding_window + 32);
  return s;
}

size_t Dsv41Model::kv_block_bytes_static(const Dsv41TextConfig& cfg) {
  const auto bytes = [&](int64_t tokens) {
    const Csa2PoolShape s = pool_shape(cfg, 1, tokens);
    size_t total = 0;
    for (const int r : s.cache_ratio) {
      const size_t slots = static_cast<size_t>(tokens / s.block_tokens) * static_cast<size_t>(s.block_tokens / r);
      total += slots * latent_row_bytes(Csa2StatePool::kMainFormat, kCsa2Latent) + slots * kCsa2IndexDim +
               slots * sizeof(float);
    }
    return total;
  };
  return bytes(2 * kBlockTokens) - bytes(kBlockTokens);
}

int Dsv41Model::cache_ordinal(int layer) const { return cache_ord_[static_cast<size_t>(layer)]; }
int Dsv41Model::tail_ordinal(int layer) const { return tail_ord_[static_cast<size_t>(layer)]; }

// ---------------------------------------------------------------------------
// Construction.
// ---------------------------------------------------------------------------
Dsv41Model::Dsv41Model(const Dsv41TextConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
                       int64_t max_cache_tokens, Dsv41Residency residency, BoundaryReducer* boundary, int tp_rank,
                       int tp_world, int max_requests, bool mtp, int decode_rows)
    : cfg_(cfg),
      loader_(cfg, checkpoint_dir, tp_rank, tp_world, residency,
              tp_world > 1 ? Dsv41HeadSharding::VocabSharded : Dsv41HeadSharding::Full,
              mtp && residency == Dsv41Residency::Resident),
      world_(tp_world) {
  if (max_tokens <= 0) throw std::invalid_argument("Dsv41Model: max_tokens must be positive");
  if (mtp) {
    if (cfg_.num_nextn_predict_layers < 1 || cfg_.dspark_target_layer_ids.empty())
      throw std::invalid_argument("Dsv41Model: mtp needs the DSpark draft stages and target layers");
    if (cfg_.dspark_block_size < 1 || cfg_.dspark_block_size > Csa2Layer::kMaxDraftBlock ||
        cfg_.dspark_block_size > kSpecRows - 1)
      throw std::invalid_argument("Dsv41Model: dspark_block_size must be in [1, min(kMaxDraftBlock, kSpecRows - 1)]");
    if (cfg_.dspark_markov_rank < 1 || cfg_.dspark_markov_rank > 512 || cfg_.dspark_noise_token_id < 0 ||
        cfg_.dspark_noise_token_id >= cfg_.vocab_size)
      throw std::invalid_argument("Dsv41Model: the DSpark Markov rank or noise token is out of range");
    for (const int l : cfg_.dspark_target_layer_ids)
      if (l < 0 || l >= cfg_.num_hidden_layers) throw std::invalid_argument("Dsv41Model: a DSpark target layer is out of range");
    targets_ = static_cast<int>(cfg_.dspark_target_layer_ids.size());
  }
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("Dsv41Model: max_requests must be in [1, kPickMaxRequests]");
  if (max_requests > decode_rows_cap() || decode_rows > decode_rows_cap())
    throw std::invalid_argument("Dsv41Model: the decode batch is bounded at " + std::to_string(decode_rows_cap()) +
                                " rows: max_requests and decode_rows must not exceed it");
  if ((tp_world > 1) != (boundary != nullptr))
    throw std::invalid_argument("Dsv41Model: a boundary reducer is required exactly when tp_world > 1");
  init_stream();
  loader_.set_reader_stream(stream_);
  log_memory_ledger("dsv41: loader opened");
  const int H = cfg_.hidden_size;
  globals_ = loader_.load_globals();
  embed_sharded_ = globals_.embed_vocab_count < cfg_.vocab_size;
  if (embed_sharded_ && boundary == nullptr)
    throw std::invalid_argument("Dsv41Model: a vocab-sharded embedding needs the boundary reducer (world > 1)");
  // The Engram tables and their constants: mapped before the sources go.
  const bool has_engram = !cfg_.engram_layer_ids.empty();
  if (has_engram) {
    sidecar_ = dsv41_load_engram_sidecar_for(checkpoint_dir, cfg_);
    (void)loader_.load_engram_tables();
  }
  if (residency == Dsv41Residency::Resident) {
    for (int l = 0; l < (mtp ? cfg_.max_layer() : cfg_.num_hidden_layers); ++l) (void)loader_.load_layer(l);
    loader_.release_sources();
  }
  log_memory_ledger("dsv41: layers resident, sources released");
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
    sp.snapshot_align = kBlockTokens;  // a whole block of the ratio-1 cache (two of the ratio-2 caches')
    // A sub-block cut (align 2, to cache prompts shorter than a block) is
    // deferred: the partial-block copy lands on the CSA2 state pool, a
    // second paged pool the scheduler's block reservation does not size,
    // so a sub-block snapshot under real load acquires a block the pool
    // still needs and a later kernel writes out of range (surfaced at
    // reset_request's ring memset on the four-slot plain world, 2026-09-14).
    // The fix is to size the CSA2 pool's snapshot headroom; the plan §8.
    sp.draft_width = mtp ? targets_ * H : H;  // the draft window holds [h_t1 | h_t2 | h_t3] per row
    sp.eos = static_cast<int32_t>(cfg_.eos_token_id);
    init_session(sp);
  }
  if (max_decode_rows_ > decode_rows_cap())
    throw std::logic_error("Dsv41Model: the session core widened the decode batch past the family's cap");
  gemm_ws_bytes_ = std::max<size_t>(64u << 20, gemm_.query_workspace_bytes(max_tokens_, lm_vocab_count_, H, DType::BF16));
  gemm_ws_ = dev_alloc<char>(gemm_ws_bytes_);
  gemm_.set_decode_rows(max_decode_rows_);
  const Dsv41LocalGeometry& geo = loader_.geometry();
  moe_cfg_ = cfg_.moe_config(static_cast<int>(geo.local_inter), /*draft=*/false);
  if (mtp) draft_moe_cfg_ = cfg_.moe_config(static_cast<int>(geo.local_inter), /*draft=*/true);
  mhc_cfg_.hc_mult = cfg_.hc_mult;
  mhc_cfg_.hidden = H;
  mhc_cfg_.sinkhorn_iters = cfg_.hc_sinkhorn_iters;
  mhc_cfg_.hc_eps = cfg_.hc_eps;
  mhc_cfg_.norm_eps = cfg_.rms_norm_eps;
  GlmMhcConfig::validate_config(mhc_cfg_);
  csa2_cfg_ = csa2_config(cfg_, tp_world);
  // The decode form of every dense bf16/fp8 site (the csa2 projections, the
  // engram wkv, the draft's main_proj, the lm head): the streaming
  // tensor-core GEMM unless the environment asks for the GEMV chunks (an
  // A/B switch: the two forms are tolerance-equal, not bitwise, and a
  // request's rows must meet the same form batched and alone).
  dense_mma_ = std::getenv("DGPP_DSV41_DENSE_GEMV") == nullptr;
  if (boundary_) boundary_->bind_stream(stream_);  // the stream-ordered reducer's stream (plan D9)
  csa2_cfg_.dense_mma = dense_mma_;
  gemm_.set_decode_mma(dense_mma_);
  // The mHC dots take the tiled form at every prefill row count: a row's
  // collapse coefficients are then one chain whatever rows share the
  // launch, and a prompt prefilled in a group (session_prefill_group) is
  // bitwise the prompt alone (the vector form under 16 rows was not).
  mhc_set_tile_min_tokens(1);
  // The layer -> cache / tail ordinals.
  cache_ord_.assign(static_cast<size_t>(cfg_.max_layer()), -1);
  tail_ord_.assign(static_cast<size_t>(cfg_.max_layer()), -1);
  {
    std::vector<int> tail_of_source(cfg_.kv_source_layer_ids.size(), -1);
    int t = 0;
    for (size_t i = 0; i < cfg_.kv_source_layer_ids.size(); ++i)
      if (cfg_.compress_ratio(cfg_.kv_source_layer_ids[i]) == 2) tail_of_source[i] = t++;
    tails_ = t;
    for (int l = 0; l < cfg_.max_layer(); ++l) {
      const int kv = cfg_.kv_source_of(l);
      if (kv < 0) continue;
      for (size_t i = 0; i < cfg_.kv_source_layer_ids.size(); ++i)
        if (cfg_.kv_source_layer_ids[i] == kv) {
          cache_ord_[static_cast<size_t>(l)] = static_cast<int>(i);
          if (cfg_.is_kv_source(l)) tail_ord_[static_cast<size_t>(l)] = tail_of_source[i];
        }
    }
  }
  // The pool and the CSA2 scratch.
  const int64_t slots = max_cache_tokens_;
  pool_.init(pool_shape(cfg_, max_requests_, slots));
  csa2_scratch_bytes_ = Csa2Layer::scratch_bytes(csa2_cfg_, max_tokens_, slots, max_decode_rows_, kDecodeSplit, kDotBudget);
  csa2_scratch_ = dev_alloc<char>(csa2_scratch_bytes_);
  log_memory_ledger("dsv41: pool and scratch");
  // The rotary tables: the window's (theta, no YaRN) and the compressed
  // layers' (compress theta with YaRN).
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
  const size_t rows = static_cast<size_t>(std::max(max_decode_rows_, max_tokens_));
  streams_a_ = dev_alloc<uint16_t>(M * 4 * H);
  streams_b_ = dev_alloc<uint16_t>(M * 4 * H);
  x_ = dev_alloc<uint16_t>(M * H);
  y_ = dev_alloc<uint16_t>(M * H);
  collapsed_ = dev_alloc<uint16_t>(M * H);
  post_bf16_ = dev_alloc<uint16_t>(M * 4);
  comb_bf16_ = dev_alloc<uint16_t>(M * 16);
  post_f32_ = dev_alloc<float>(M * 4);
  comb_f32_ = dev_alloc<float>(M * 16);
  pre_a_ = dev_alloc<float>(M * 4);
  pre_b_ = dev_alloc<float>(M * 4);
  one_hot_ = dev_alloc<float>(M * 4);
  mhc_logits_ = dev_alloc<float>(M * static_cast<size_t>(mhc_cfg_.coeff_rows()));
  {
    std::vector<float> oh(M * 4, 0.0f);
    for (size_t t = 0; t < M; ++t) oh[t * 4] = 1.0f;
    DGPP_CUDA_OK(cudaMemcpy(one_hot_, oh.data(), oh.size() * 4, cudaMemcpyHostToDevice));
  }
  if (has_engram) {
    engram_ = std::make_unique<Dsv41EngramLayer>(cfg_, loader_.load_engram_tables(), sidecar_, max_tokens_, max_requests_);
    engram_->set_dense_mma(dense_mma_);
    engram_kv_ = dev_alloc<uint16_t>(M * 5 * H);
    d_ctx_ = dev_alloc<int32_t>(R * 4);
    spec_ctx_ = dev_alloc<int32_t>(rows * 4);
    std::vector<int32_t> pad(R * 4, sidecar_.pad_class);
    DGPP_CUDA_OK(cudaMemcpy(d_ctx_, pad.data(), pad.size() * 4, cudaMemcpyHostToDevice));
  }
  if (tails_ > 0) spec_tails_ = dev_alloc<float>(static_cast<size_t>(tails_) * static_cast<size_t>(max_decode_rows_) * 2 * kCsa2Latent);
  {
    // The bounded prefill's tail and segment coefficients (window rows).
    const size_t win = static_cast<size_t>(cfg_.sliding_window);
    tail_streams_ = dev_alloc<uint16_t>(win * 4 * H);
    tail_pre_ = dev_alloc<float>(win * 4);
    seg_pre_ = dev_alloc<float>(win * 4);
    if (cfg_.sliding_window > max_tokens_)
      throw std::invalid_argument("Dsv41Model: max_tokens must cover the sliding window (the bounded prefill's segment)");
    const int dec0 = cfg_.decoder_first_layer();
    for (int l = dec0; l < cfg_.num_hidden_layers; ++l)
      if (cfg_.compress_ratio(l) > 1 || (cfg_.is_kv_source(l) && l != dec0))
        throw std::invalid_argument("Dsv41Model: the decoder (from the last kv source on) must read that source at ratio 1");
  }
  if (mtp) {
    const size_t W = static_cast<size_t>(targets_) * H;
    const size_t D = static_cast<size_t>(max_decode_rows_);
    const size_t block = static_cast<size_t>(cfg_.dspark_block_size);
    main_hidden_ = dev_alloc<uint16_t>(M * W);
    main_gather_ = dev_alloc<uint16_t>(D * W);
    main_x_ = dev_alloc<uint16_t>(std::max(M, D) * H);
    blk_pos_ = dev_alloc<int64_t>(D);
    blk_tok_ = dev_alloc<int64_t>(D);
    blk_req_ = dev_alloc<int32_t>(D);
    blk_spans_ = dev_alloc<int32_t>(D + 1);
    base_logits_ = dev_alloc<float>(D * static_cast<size_t>(lm_vocab_count_));
    conf_ = dev_alloc<float>(R * block);
    d_draft_pos_ = dev_alloc<int64_t>(M);
    d_draft_req_ = dev_alloc<int32_t>(M);
    DGPP_CUDA_OK(cudaMemsetAsync(conf_, 0, R * block * 4, stream_));
  }
  {
    const size_t layers = static_cast<size_t>(cfg_.num_hidden_layers);
    const size_t K = static_cast<size_t>(cfg_.num_experts_per_tok);
    h_route_ids_ = pinned_alloc<int32_t>(layers * M * K);
    h_route_weights_ = pinned_alloc<float>(layers * M * K);
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  log_memory_ledger("dsv41: model ready");
}

Dsv41Model::~Dsv41Model() {
  cudaStreamSynchronize(stream_);
  csa2_.reset();
  moe_.reset();
  engram_.reset();
  cudaFree(gemm_ws_);
  cudaFree(csa2_scratch_);
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
  cudaFree(pre_a_);
  cudaFree(pre_b_);
  cudaFree(one_hot_);
  cudaFree(mhc_logits_);
  cudaFree(engram_kv_);
  cudaFree(d_ctx_);
  cudaFree(spec_ctx_);
  cudaFree(spec_tails_);
  cudaFree(tail_streams_);
  cudaFree(tail_pre_);
  cudaFree(seg_pre_);
  cudaFree(main_hidden_);
  cudaFree(main_gather_);
  cudaFree(main_x_);
  cudaFree(blk_pos_);
  cudaFree(blk_tok_);
  cudaFree(blk_req_);
  cudaFree(blk_spans_);
  cudaFree(base_logits_);
  cudaFree(conf_);
  cudaFree(d_draft_pos_);
  cudaFree(d_draft_req_);
  draft_moe_.reset();
  cudaFreeHost(h_route_ids_);
  cudaFreeHost(h_route_weights_);
}

// ---------------------------------------------------------------------------
// The memory plan.
// ---------------------------------------------------------------------------
Dsv41Model::MemoryPlan Dsv41Model::plan_memory(const Dsv41TextConfig& cfg, int max_tokens, int64_t max_cache_tokens,
                                               int tp_rank, int tp_world, Dsv41Residency residency, int max_requests,
                                               bool mtp, int decode_rows) {
  if (max_tokens <= 0) throw std::invalid_argument("plan_memory: max_tokens must be positive");
  if (mtp && (cfg.num_nextn_predict_layers < 1 || cfg.dspark_target_layer_ids.empty()))
    throw std::invalid_argument("plan_memory: mtp needs the DSpark draft stages and target layers");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("plan_memory: max_requests must be in [1, kPickMaxRequests]");
  if (max_requests > decode_rows_cap() || decode_rows > decode_rows_cap())
    throw std::invalid_argument("plan_memory: the decode batch is bounded at " + std::to_string(decode_rows_cap()) + " rows");
  const int rows = std::max({kDecodeRows, decode_rows, max_requests});
  const Dsv41HeadSharding head = tp_world > 1 ? Dsv41HeadSharding::VocabSharded : Dsv41HeadSharding::Full;
  const Dsv41LocalGeometry geo = Dsv41LocalGeometry::from_config(cfg, tp_rank, tp_world, head);
  const int64_t cache_tokens =
      ((std::max<int64_t>(max_cache_tokens, max_tokens) + kBlockTokens - 1) / kBlockTokens) * kBlockTokens;
  MemoryPlan plan;
  plan.context_tokens = std::min<int64_t>(cache_tokens, cfg.max_position_embeddings);
  const size_t M = static_cast<size_t>(max_tokens);
  const size_t R = static_cast<size_t>(max_requests);
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t V = static_cast<size_t>(Dsv41LayerStream::lm_vocab_count(cfg, tp_rank, tp_world, head));
  size_t staging = 0;
  if (residency == Dsv41Residency::Resident) {
    plan.add("model weights (resident)", Dsv41LayerStream::resident_bytes(cfg, tp_rank, tp_world, head, mtp));
    staging = Dsv41LayerStream::staging_plan_bytes(cfg, tp_rank, tp_world, head, mtp);
  } else {
    size_t largest = 0;
    for (int l = 0; l < (mtp ? cfg.max_layer() : cfg.num_hidden_layers); ++l)
      largest = std::max(largest, Dsv41LayerStream::layer_bytes(cfg, l, tp_rank, tp_world));
    plan.add("model weights (one streamed layer + globals)",
             largest + Dsv41LayerStream::globals_bytes(cfg, tp_rank, tp_world, head));
  }
  const size_t after_weights = plan.total_bytes();
  plan.add("gemm workspace (at least)", size_t{64} << 20);
  const Csa2Config c2 = csa2_config(cfg, tp_world);
  const Csa2PoolShape shape = pool_shape(cfg, max_requests, cache_tokens);
  plan.add("csa2 cache pool (fp4_block main KV + index keys on the kv sources, the window rings, the compressor tails)",
           Csa2StatePool::cache_bytes(shape));
  plan.add("csa2 scratch (projections, selection, attention, the prefill dot tiles)",
           Csa2Layer::scratch_bytes(c2, max_tokens, cache_tokens, rows, kDecodeSplit, kDotBudget));
  {
    size_t core_dev = 0, core_pin = 0;
    const size_t targets = cfg.dspark_target_layer_ids.size();
    const int draft_width = mtp ? static_cast<int>(targets) * cfg.hidden_size : cfg.hidden_size;
    session_core_plan_bytes(max_tokens, max_requests, cfg.hidden_size, static_cast<int>(V), mtp, draft_width, &core_dev,
                            &core_pin, rows);
    const size_t K = static_cast<size_t>(cfg.num_experts_per_tok);
    size_t act = core_dev + 2 * M * 4 * H * 2 + 3 * M * H * 2 + M * (4 + 16) * 2 + M * (4 + 16 + 4 + 4 + 4) * 4 +
                 M * 24 * 4;
    if (mtp) {
      const size_t D = static_cast<size_t>(rows);
      act += M * targets * H * 2 + D * targets * H * 2 + std::max(M, D) * H * 2 + D * (8 + 8 + 4 + 4) + 4 + D * V * 4 +
             R * static_cast<size_t>(cfg.dspark_block_size) * 4 + M * 12;
    }
    if (!cfg.engram_layer_ids.empty()) act += M * 5 * H * 2 + R * 16 + std::max(M, static_cast<size_t>(rows)) * 16;
    act += static_cast<size_t>(shape.tail_ordinals) * static_cast<size_t>(rows) * 2 * kCsa2Latent * 4;
    // The bounded prefill's tail streams and the tail/segment coefficients.
    act += static_cast<size_t>(cfg.sliding_window) * 4 * H * 2 + 2 * static_cast<size_t>(cfg.sliding_window) * 4 * 4;
    plan.add("activations (session core, the four residual streams, block io, mHC coefficients, route staging)", act,
             core_pin + static_cast<size_t>(cfg.num_hidden_layers) * M * K * 8);
  }
  {
    const GlmMoeConfig moe_cfg = cfg.moe_config(static_cast<int>(geo.local_inter), false);
    const int slots = residency == Dsv41Residency::Resident ? cfg.num_hidden_layers : 0;
    size_t moe_pinned = 0;
    const size_t moe_dev = GlmMoeLayer::scratch_bytes(moe_cfg, max_tokens, rows, slots, &moe_pinned);
    plan.add("moe scratch (routed slots, shared expert, graph tables)", moe_dev, moe_pinned);
    if (mtp) {
      const GlmMoeConfig draft_cfg = cfg.moe_config(static_cast<int>(geo.local_inter), true);
      const int draft_slots = residency == Dsv41Residency::Resident ? cfg.num_nextn_predict_layers : 0;
      size_t draft_pinned = 0;
      const size_t draft_dev = GlmMoeLayer::scratch_bytes(draft_cfg, max_tokens, rows, draft_slots, &draft_pinned);
      plan.add("dspark moe scratch (the draft stages' routed chain)", draft_dev, draft_pinned);
    }
  }
  if (!cfg.engram_layer_ids.empty())
    plan.add("engram (the mmap'ed tables' staging, the hash ids, the embedded rows, the kv partial)",
             Dsv41EngramLayer::device_bytes(cfg, tp_world, max_tokens),
             Dsv41EngramLayer::staging_bytes(cfg, tp_world, max_tokens));
  if (staging > 0) {
    const size_t later = plan.total_bytes() - after_weights;
    plan.add("loader staging beyond the caches that replace it (pinned host, transient)", 0,
             staging > later ? staging - later : 0);
  }
  return plan;
}

size_t Dsv41Model::session_snapshot_bytes(const Dsv41TextConfig& cfg, int, bool) {
  const size_t ring = static_cast<size_t>(std::max(kRingSlots, cfg.sliding_window + 32)) *
                      latent_row_bytes(Csa2StatePool::kRingFormat, kCsa2Latent);
  size_t tails = 0;
  for (const int l : cfg.kv_source_layer_ids)
    if (cfg.compress_ratio(l) == 2) ++tails;
  return static_cast<size_t>(cfg.max_layer()) * ring + tails * 2 * kCsa2Latent * sizeof(float) + 16;
}

size_t Dsv41Model::snapshot_state_bytes() const { return session_snapshot_bytes(cfg_, world_, false); }

// ---------------------------------------------------------------------------
// Views and layer objects.
// ---------------------------------------------------------------------------
Csa2LayerWeights Dsv41Model::csa2_view(const Dsv41LayerResident& r, int layer) const {
  const Dsv41AttnResident& a = r.attn;
  Csa2LayerWeights w;
  w.wq_a = a.wq_a;
  w.wkv = a.wkv;
  w.wq_b = a.wq_b;
  w.wo_a = a.wo_a;
  w.wo_b = a.wo_b;
  w.q_norm = a.q_norm;
  w.kv_norm = a.kv_norm;
  w.attn_sink = a.attn_sink;
  w.idx_wq_b = a.idx_wq_b;
  w.idx_wp = a.idx_wp;
  w.idx_wk = a.idx_wk;
  w.idx_k_norm = a.idx_k_norm;
  w.comp_wkv = a.comp_wkv;
  w.comp_wgate = a.comp_wgate;
  w.comp_norm = a.comp_norm;
  w.ratio = cfg_.compress_ratio(layer);
  w.inv_freq = w.ratio > 0 ? inv_freq_compressed_ : inv_freq_window_;
  w.cache_ord = cache_ordinal(layer);
  w.tail_ord = tail_ordinal(layer);
  w.kv_source = cfg_.is_kv_source(layer);
  w.index_source = cfg_.is_index_source(layer);
  w.candidate_source = layer == cfg_.candidate_source_layer_id;
  w.uses_candidates = cfg_.uses_candidates(layer);
  return w;
}

GlmMoeWeights Dsv41Model::moe_view(const Dsv41MoeResident& m) {
  GlmMoeWeights w;
  w.router_gate = m.router;
  w.router_bias = m.router_bias;
  for (int i = 0; i < 3; ++i) w.shared[i] = m.shared[i];
  w.experts_fp4 = m.experts.data();
  return w;
}

Dsv41EngramLayerWeights Dsv41Model::engram_view(const Dsv41EngramResident& e) const {
  Dsv41EngramLayerWeights w;
  w.wkv = e.wkv;
  w.q_weight = e.q_weight;
  w.k_weight = e.k_weight;
  w.table_index = e.table_index;
  return w;
}

void Dsv41Model::build_layer_objects(const Dsv41LayerResident& r) {
  if (!csa2_) {
    csa2_ = std::make_unique<Csa2Layer>(gemm_, csa2_cfg_, max_tokens_, max_cache_tokens_, csa2_scratch_, csa2_scratch_bytes_,
                                        gemm_ws_, gemm_ws_bytes_, max_decode_rows_, kDecodeSplit, kDotBudget);
  }
  csa2_->rebind(csa2_view(r, r.layer), r.layer);
  if (cfg_.is_draft(r.layer)) {
    if (!draft_moe_) {
      const int slots = loader_.residency() == Dsv41Residency::Resident ? cfg_.num_nextn_predict_layers : 0;
      draft_moe_ = std::make_unique<GlmMoeLayer>(moe_view(r.moe), draft_moe_cfg_, max_tokens_, max_decode_rows_, slots);
    } else {
      draft_moe_->rebind(moe_view(r.moe));
    }
    return;
  }
  if (!moe_) {
    const int slots = loader_.residency() == Dsv41Residency::Resident ? cfg_.num_hidden_layers : 0;
    moe_ = std::make_unique<GlmMoeLayer>(moe_view(r.moe), moe_cfg_, max_tokens_, max_decode_rows_, slots);
  } else {
    moe_->rebind(moe_view(r.moe));
  }
}

int Dsv41Model::target_ordinal(int layer) const {
  for (int i = 0; i < targets_; ++i)
    if (cfg_.dspark_target_layer_ids[static_cast<size_t>(i)] == layer) return i;
  return -1;
}

const Dsv41DraftResident& Dsv41Model::draft_stage(int stage) {
  return loader_.load_layer(cfg_.num_hidden_layers + stage).draft;
}

// ---------------------------------------------------------------------------
// The session core's state hooks.
// ---------------------------------------------------------------------------
void Dsv41Model::reset_slot_state(int req) {
  pool_.reset_request(req, stream_);
  if (engram_) {
    std::vector<int32_t> pad(4, sidecar_.pad_class);
    DGPP_CUDA_OK(cudaMemcpyAsync(ctx(req), pad.data(), 16, cudaMemcpyHostToDevice, stream_));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  }
}

GlmSpecSegments Dsv41Model::spec_segments(int req, int snapshot_row0) const {
  GlmSpecSegments segs;
  const auto add = [&](void* dst, const void* snapshots, size_t row_stride, size_t bytes) {
    if (segs.count >= kSpecMaxSegments) throw std::logic_error("spec_segments: too many state families");
    segs.seg[segs.count++] = GlmSpecSegment{dst, snapshots, row_stride, bytes};
  };
  const size_t tail_bytes = 2 * kCsa2Latent * sizeof(float);
  for (int t = 0; t < tails_; ++t)
    add(pool_.tails(t) + static_cast<size_t>(req) * 2 * kCsa2Latent, spec_tails(t, snapshot_row0), tail_bytes, tail_bytes);
  if (engram_) add(ctx(req), spec_ctx_ + static_cast<size_t>(snapshot_row0) * 4, 16, 16);
  return segs;
}

void Dsv41Model::write_state_snapshot(int req, uint8_t* d, int spec_row) {
  const bool live = spec_row < 0;
  const size_t row = live ? 0 : static_cast<size_t>(spec_row);
  const size_t rb = pool_.ring_bytes_per_request();
  // The rings as they stand: positional, so any spec row's view is the ring
  // itself (a later row overwrote only slots no query of that position reads).
  for (int l = 0; l < pool_.shape().layers; ++l) {
    DGPP_CUDA_OK(cudaMemcpyAsync(d, pool_.ring(l) + static_cast<size_t>(req) * rb, rb, cudaMemcpyDeviceToDevice, stream_));
    d += rb;
  }
  const size_t tail_bytes = 2 * kCsa2Latent * sizeof(float);
  for (int t = 0; t < tails_; ++t) {
    DGPP_CUDA_OK(cudaMemcpyAsync(d, live ? pool_.tails(t) + static_cast<size_t>(req) * 2 * kCsa2Latent : spec_tails(t, static_cast<int>(row)),
                                 tail_bytes, cudaMemcpyDeviceToDevice, stream_));
    d += tail_bytes;
  }
  if (engram_) DGPP_CUDA_OK(cudaMemcpyAsync(d, live ? ctx(req) : spec_ctx_ + row * 4, 16, cudaMemcpyDeviceToDevice, stream_));
  else DGPP_CUDA_OK(cudaMemsetAsync(d, 0, 16, stream_));
}

void Dsv41Model::read_state_snapshot(int req, const uint8_t* s) {
  const size_t rb = pool_.ring_bytes_per_request();
  for (int l = 0; l < pool_.shape().layers; ++l) {
    DGPP_CUDA_OK(cudaMemcpyAsync(pool_.ring(l) + static_cast<size_t>(req) * rb, s, rb, cudaMemcpyDeviceToDevice, stream_));
    s += rb;
  }
  const size_t tail_bytes = 2 * kCsa2Latent * sizeof(float);
  for (int t = 0; t < tails_; ++t) {
    DGPP_CUDA_OK(cudaMemcpyAsync(pool_.tails(t) + static_cast<size_t>(req) * 2 * kCsa2Latent, s, tail_bytes,
                                 cudaMemcpyDeviceToDevice, stream_));
    s += tail_bytes;
  }
  if (engram_) DGPP_CUDA_OK(cudaMemcpyAsync(ctx(req), s, 16, cudaMemcpyDeviceToDevice, stream_));
}

void Dsv41Model::graph_prepare() {
  if (loader_.residency() != Dsv41Residency::Resident)
    throw std::logic_error("session_graph_prepare: the decode graph needs a resident stack");
  for (int layer = 0; layer < (mtp_ ? cfg_.max_layer() : cfg_.num_hidden_layers); ++layer) {
    const Dsv41LayerResident& r = loader_.load_layer(layer);
    build_layer_objects(r);
    for (int rows = 1; rows <= max_decode_rows_; ++rows)
      if (!csa2_->prepare(rows)) throw std::runtime_error("session_graph_prepare: CSA2 GEMM plans unavailable");
    if (cfg_.is_draft(layer)) draft_moe_->prepare_graph_table(cfg_.draft_stage(layer), stream_);
    else moe_->prepare_graph_table(layer, stream_);
  }
}

// ---------------------------------------------------------------------------
// The DSpark draft (plan D8).
// ---------------------------------------------------------------------------
void Dsv41Model::mtp_run_rows(int req, const int64_t* tokens, int64_t first_pos, int T, bool decode_row, bool capture,
                              int head_rows, int batch_requests) {
  if (!mtp_) throw std::logic_error("mtp_run_rows: MTP is not enabled");
  if (T <= 0 || T > max_tokens_) throw std::invalid_argument("mtp_run_rows: rows");
  if (!decode_row && head_rows == 0) {
    // The prefill's draft rows: the rows the last walk carried target
    // hidden for — the chunk in exact mode, the replay segment in bounded
    // mode (its tail rows included: the draft rings hold the last window
    // positions) — up to the caller's end (the prompt's last row drafts
    // nothing; the session names its chunk's rows, which start before the
    // segment or, after a tail, inside it). A skipped bounded chunk has no
    // rows.
    const auto [seg_pos0, seg_rows] = segment_of(req);
    const int64_t hi = std::min<int64_t>(first_pos + T, seg_pos0 + seg_rows);
    if (hi <= seg_pos0) return;
    first_pos = seg_pos0;
    T = static_cast<int>(hi - seg_pos0);
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
  draft_first(req, tokens, d_pos, d_req, T, decode_row, capture, head_rows, batch_requests);
}

void Dsv41Model::draft_first(int req, const int64_t* tokens, const int64_t* d_pos, const int32_t* d_req, int T,
                             bool decode_row, bool capture, int head_rows, int batch_requests) {
  const int H = cfg_.hidden_size;
  const int W = targets_ * H;
  const int block = cfg_.dspark_block_size;
  const int stages = cfg_.num_nextn_predict_layers;
  const float eps = cfg_.rms_norm_eps;
  // ---- the accepted rows' main hidden -> main_x --------------------------------
  // Decode rows gather from the slots' windows by position (a padding row,
  // pos -1, gathers zeros and appends nothing); prefill rows read the
  // last walk's rows in place (row 0 at seg_pos0_; mtp_run_rows clamped
  // them to the walk's segment). (A streamed stack holds
  // one layer at a time: every stage's tensors are used right after its
  // load.)
  const uint16_t* src = main_hidden_;
  if (decode_row) {
    gather_draft_hidden(d_req, d_pos, main_gather_, T);
    src = main_gather_;
  }
  {
    const Dsv41DraftResident& d0 = draft_stage(0);
    if (!d0.main_proj.payload || !d0.main_norm) throw std::runtime_error("mtp_run_rows: the DSpark main projection is unbound");
    launch_scale_gemm_grid_bf16(src, static_cast<size_t>(W), d0.main_proj.payload, d0.main_proj.scales, main_x_, T, H, W,
                                stream_, 0, 5, 5, dense_mma_);
    csa2_rmsnorm_bf16(main_x_, H, d0.main_norm, main_x_, H, T, H, eps, stream_);
  }
  // ---- the draft rings: each stage's window latent of the real rows ------------
  // A prefill chunk appends its last min(T, ring) rows (one ring slot per
  // row; the older rows are past every window).
  const int n = decode_row ? T : std::min(T, csa2_cfg_.ring_slots);
  for (int s = 0; s < stages; ++s) {
    const Dsv41LayerResident& r = loader_.load_layer(cfg_.num_hidden_layers + s);
    build_layer_objects(r);
    csa2_->append_window_rows(main_x_ + static_cast<size_t>(T - n) * H, pool_, d_req + (T - n), d_pos + (T - n), n,
                              stream_);
  }
  if (head_rows == 0) return;  // the prefill fills the rings; no block
  // ---- the block: [next, noise x (block - 1)] at the next block positions ---------
  const int rpg = T / (batch_requests > 0 ? batch_requests : 1);
  const int groups = T / rpg;
  const int rows = groups * block;
  if (rows > max_decode_rows_) throw std::invalid_argument("mtp_run_rows: the draft blocks exceed the decode batch");
  dsv41_dspark_block_rows(d_pos, tokens, d_req, groups, rpg, block, cfg_.dspark_noise_token_id, blk_pos_, blk_tok_,
                          blk_req_, blk_spans_, stream_);
  cur_ = streams_a_;
  nxt_ = streams_b_;
  pre_cur_ = one_hot_;
  pre_nxt_ = pre_a_;
  // DGPP_DSV41_CAPTURE_DRAFT=1: the eager block walk keeps its sites (the
  // parity localizer in dsv41_forward_test).
  if (!capture && std::getenv("DGPP_DSV41_CAPTURE_DRAFT")) {
    debug_capture_ = true;
    debug_sites_.clear();
  }
  gather_embedding(blk_tok_, rows, capture);
  WalkRows wr;
  wr.tokens = blk_tok_;
  wr.req_ids = blk_req_;
  wr.pos = blk_pos_;
  wr.spans = blk_spans_;
  wr.num_requests = groups;
  wr.req = req;
  wr.decode = true;
  wr.capture = capture;
  for (int s = 0; s < stages; ++s) {
    const Dsv41LayerResident& r = loader_.load_layer(cfg_.num_hidden_layers + s);
    wr.moe_table_slot = capture ? s : -1;
    if (!csa2_->prepare(rows)) throw std::runtime_error("mtp_run_rows: CSA2 GEMM plans unavailable");
    enqueue_layer(r, cfg_.num_hidden_layers + s, rows, wr);
  }
  debug_capture_ = false;
  // ---- the head over the block: hc_pre with the last stage's pre, the draft's
  // norm, the shared lm head -> base_logits_ [rows, vocab slice] --------------
  const Dsv41DraftResident& dl = draft_stage(stages - 1);
  if (!dl.norm || !dl.markov_embed || !dl.markov_head || !dl.confidence)
    throw std::runtime_error("mtp_run_rows: the DSpark head tensors are unbound");
  launch_mhc_collapse_normed(cur_, pre_cur_, nullptr, eps, collapsed_, nullptr, mhc_cfg_, rows, stream_);
  csa2_rmsnorm_bf16(collapsed_, H, dl.norm, h_, H, rows, H, eps, stream_);
  gemm_.matmul(h_, globals_.lm_head, base_logits_, rows, lm_vocab_count_, H, DType::BF16, GemmOut::F32,
               static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
  // ---- block row 0, biased by the Markov head of `next`, into the head rows ----
  const int rows_out = head_rows / groups;
  dsv41_dspark_markov_bias(base_logits_, static_cast<int64_t>(block) * lm_vocab_count_, 0, dl.markov_embed, dl.markov_head,
                           cfg_.dspark_markov_rank, lm_vocab_begin_, lm_vocab_count_, blk_tok_, block, groups, logits_,
                           static_cast<int64_t>(rows_out) * lm_vocab_count_, rows_out, stream_);
  dsv41_dspark_confidence(collapsed_, static_cast<int64_t>(block) * H, 0, H, dl.markov_embed, cfg_.dspark_markov_rank,
                          blk_tok_, block, dl.confidence, groups, conf_ + static_cast<size_t>(batch_requests > 0 ? 0 : req) * block,
                          block, stream_);
  draft_row_ = 1;
  if (decode_row && (!capture || decode_tail_mirrors_) && head_rows <= max_decode_rows_)
    DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_logits_, logits_, static_cast<size_t>(head_rows) * lm_vocab_count_ * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
}

void Dsv41Model::draft_chain_row(int req, const int64_t* tokens, bool capture, int head_rows, int batch_requests) {
  const int H = cfg_.hidden_size;
  const int block = cfg_.dspark_block_size;
  const int groups = batch_requests > 0 ? batch_requests : 1;
  if (draft_row_ >= block) throw std::logic_error("mtp_run_rows: more chain rows than the block holds");
  if (head_rows != groups) throw std::invalid_argument("mtp_run_rows: a chain call heads one row per request");
  const Dsv41DraftResident& dl = draft_stage(cfg_.num_nextn_predict_layers - 1);
  const int row = draft_row_++;
  dsv41_dspark_markov_bias(base_logits_, static_cast<int64_t>(block) * lm_vocab_count_, row, dl.markov_embed, dl.markov_head,
                           cfg_.dspark_markov_rank, lm_vocab_begin_, lm_vocab_count_, tokens, 1, groups, logits_,
                           lm_vocab_count_, 1, stream_);
  dsv41_dspark_confidence(collapsed_, static_cast<int64_t>(block) * H, row, H, dl.markov_embed, cfg_.dspark_markov_rank,
                          tokens, 1, dl.confidence, groups, conf_ + static_cast<size_t>(batch_requests > 0 ? 0 : req) * block + row,
                          block, stream_);
  if (!capture || decode_tail_mirrors_)
    DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_logits_, logits_, static_cast<size_t>(head_rows) * lm_vocab_count_ * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
}

// ---------------------------------------------------------------------------
// The walk.
// ---------------------------------------------------------------------------
uint16_t* Dsv41Model::stage(uint16_t* fallback, int T, int width, bool capture) {
  if (!boundary_) return fallback;
  uint16_t* s = boundary_->stage(T, width);
  if (s == nullptr && capture) throw std::runtime_error("run_rows: a capture fold does not fit the recorder's staged buffer");
  return s ? s : fallback;
}

void Dsv41Model::fold(uint16_t* buf, int T, int width, bool capture) {
  if (!boundary_) return;
  // The host-driven reducer folds after the producing kernels have
  // quiesced; the stream-ordered one (plan D9) launches the fold on this
  // stream behind them and needs no drain.
  if (!capture && !boundary_->stream_ordered()) DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  boundary_->reduce(buf, T, width);
}

// The embedding on all four streams: the whole table's gather straight
// into the streams, or — vocab-sharded — this rank's rows, one fold, and
// the rows broadcast to the streams.
void Dsv41Model::gather_embedding(const int64_t* tokens, int T, bool capture) {
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

// One mHC site: the coefficients of this site (pre_nxt_, post, comb) from
// the streams, the collapse with the coefficients in use (pre_cur_), the
// sublayer's one-rounding norm into x_.
void Dsv41Model::mhc_site(const uint16_t* streams, const GlmMhcWeights& w, const uint16_t* ln, int T, bool decode) {
  MhcSinglePass sp;
  sp.pre_in = pre_cur_;
  sp.pre_out = pre_nxt_;
  sp.post_f32 = post_f32_;
  sp.comb_f32 = comb_f32_;
  // Decode rows stay on the per-coefficient form at any count (a batched
  // row bitwise the row alone; the >= 16-token prefill forms are not).
  (void)launch_mhc_compute_normed(streams, w, mhc_cfg_, collapsed_, post_bf16_, comb_bf16_, mhc_logits_, nullptr, nullptr,
                                  cfg_.rms_norm_eps, T, stream_, nullptr, false, &sp, decode);
  csa2_rmsnorm_bf16(collapsed_, cfg_.hidden_size, ln, x_, cfg_.hidden_size, T, cfg_.hidden_size, cfg_.rms_norm_eps, stream_);
  pre_cur_ = pre_nxt_;
  pre_nxt_ = pre_cur_ == pre_a_ ? pre_b_ : pre_a_;
}

void Dsv41Model::stream_update(const uint16_t* sublayer_out, int T) {
  launch_mhc_stream_update_f32(post_f32_, comb_f32_, sublayer_out, cur_, nxt_, mhc_cfg_, T, stream_);
  std::swap(cur_, nxt_);
}

void Dsv41Model::enqueue_layer(const Dsv41LayerResident& r, int layer, int T, const WalkRows& rows) {
  const int H = cfg_.hidden_size;
  build_layer_objects(r);
  // Engram: the hashed rows (staged at the walk's start) gate the streams
  // before this layer's coefficients read them.
  if (r.engram.present) {
    engram_->embed(r.engram.table_index, T, stream_);
    const Dsv41EngramLayerWeights ew = engram_view(r.engram);
    uint16_t* kv = stage(engram_kv_, T, 5 * H, rows.capture);
    engram_->project(ew, T, kv, stream_);
    fold(kv, T, 5 * H, rows.capture);
    engram_->gate(ew, cur_, kv, T, stream_);
  }
  // DSpark: the target layers' attention INPUT, its stream mean, per row
  // into [h_t1 | h_t2 | h_t3] (the reference reads it before the layer).
  if (mtp_) {
    const int ord = target_ordinal(layer);
    if (ord >= 0)
      dsv41_stream_mean_bf16(cur_, 4, H, T, main_hidden_ + static_cast<size_t>(ord) * H,
                             static_cast<int64_t>(targets_) * H, stream_);
  }
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
  if (!csa2_->prepare(T)) throw std::runtime_error("run_rows: CSA2 GEMM plans unavailable");
  const int tail = tail_ordinal(layer);
  if (draft) {
    csa2_->enqueue_draft_block(x_, pool_, rows.req_ids, rows.pos, T, cfg_.dspark_block_size, attn_out, stream_);
  } else if (rows.decode) {
    csa2_->enqueue_decode(x_, pool_, rows.req_ids, rows.pos, rows.spans, rows.num_requests, T, attn_out, stream_,
                          (rows.snapshots && tail >= 0) ? spec_tails(tail, 0) : nullptr);
  } else if (rows.num_spans > 0) {
    int row0 = 0;
    for (int s = 0; s < rows.num_spans; ++s) {
      const int len = rows.span_lens[s];
      if (!csa2_->prepare(len)) throw std::runtime_error("run_rows: CSA2 GEMM plans unavailable");
      const int64_t floor = rows.span_floor ? rows.span_floor[s] : rows.window_floor;
      csa2_->enqueue_prefill(x_ + static_cast<size_t>(row0) * H, pool_, rows.span_reqs[s], rows.span_pos0[s], len,
                             attn_out + static_cast<size_t>(row0) * H, stream_, floor, rows.publish, row0);
      row0 += len;
    }
  } else {
    csa2_->enqueue_prefill(x_, pool_, rows.req, rows.pos0, T, attn_out, stream_, rows.window_floor, rows.publish);
  }
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
  if (draft)
    draft_moe_->enqueue_decode(x_, ffn_out, T, nullptr, stream_, rows.moe_table_slot);
  else if (rows.decode)
    moe_->enqueue_decode(x_, ffn_out, T, nullptr, stream_, rows.moe_table_slot);
  else
    moe_->enqueue_prefill(x_, ffn_out, T, rows.trace, stream_);
  fold(ffn_out, T, H, rows.capture);  // block boundary 2: the sliced experts
  if (debug_capture_) cap.ffn_out = grab(ffn_out, static_cast<size_t>(T) * H);
  stream_update(ffn_out, T);
  if (debug_capture_) debug_sites_.push_back(std::move(cap));
}

Dsv41Model::Outputs Dsv41Model::run_rows(const RowRun& run) {
  const int T = run.T, req = run.req;
  if (run.capture && loader_.residency() != Dsv41Residency::Resident)
    throw std::logic_error("run_rows: a capture needs a resident stack");
  const int H = cfg_.hidden_size;
  const RowInputs in = begin_run(run);
  cur_ = streams_a_;
  nxt_ = streams_b_;
  // Layer 0's collapse reads the constant one-hot (1, 0, 0, 0) directly:
  // the sites rotate through pre_a_ / pre_b_ from there (no copy node —
  // the decode graph is kernels-only by contract).
  pre_cur_ = one_hot_;
  pre_nxt_ = pre_a_;
  // The Engram staging forked before the first layer (both tables' rows
  // gathered by the host while layer 0 runs); the contexts advance now.
  if (engram_) {
    engram_->stage(in.tokens, T, in.req_ids, in.pos, in.spans, in.num_requests, d_ctx_, stream_);
    engram_->context_rows(in.tokens, T, in.req_ids, in.pos, in.spans, in.num_requests, d_ctx_, spec_ctx_, stream_);
  }
  gather_embedding(in.tokens, T, run.capture);
  Outputs out;
  const bool traces = !run.decode && route_traces_;
  const size_t K = static_cast<size_t>(cfg_.num_experts_per_tok);
  WalkRows rows;
  rows.tokens = in.tokens;
  rows.req_ids = in.req_ids;
  rows.pos = in.pos;
  rows.spans = in.spans;
  rows.num_requests = in.num_requests;
  rows.req = req;
  rows.pos0 = run.pos0;
  rows.decode = run.decode;
  rows.capture = run.capture;
  rows.snapshots = run.decode && run.snapshots;
  rows.span_reqs = run.span_reqs;
  rows.span_pos0 = run.span_pos0;
  rows.span_lens = run.span_lens;
  rows.num_spans = run.num_spans;
  // The group prefill: several requests' cold prompts as the spans of one
  // walk. The dense sites, the MoE, the norms, the Engram and the head are
  // per row; the CSA2 attention and the kv publication run per span; each
  // span is its own draft segment. Not captured, positions from 0. Under
  // the bounded prefill the decoder walks each span's last window rows
  // (its replay segment), packed span after span; a span within one
  // window is whole and floors at 0.
  const bool group = run.num_spans > 0;
  std::vector<int> span_row0;
  std::vector<int64_t> span_floor(static_cast<size_t>(std::max(run.num_spans, 0)), 0);
  std::vector<int32_t> dec_lens;   // the decoder phase's span tables (bounded group)
  std::vector<int64_t> dec_pos0;
  std::vector<int> dec_row0;
  if (group) {
    if (run.decode || run.capture) throw std::logic_error("run_rows: a group prefill is neither a decode nor a capture");
    if (!run.first_chunk || !run.last_chunk) throw std::logic_error("run_rows: a group prefill is one chunk per span");
    int at = 0;
    for (int s = 0; s < run.num_spans; ++s) {
      const int len = run.span_lens[s];
      if (len <= 0 || len > prefill_group_span_limit())
        throw std::invalid_argument("run_rows: a group prefill span of " + std::to_string(len) +
                                    " rows exceeds the span limit " + std::to_string(prefill_group_span_limit()));
      if (run.span_pos0[s] != 0) throw std::invalid_argument("run_rows: a group prefill takes cold prompts (position 0)");
      span_row0.push_back(at);
      at += len;
    }
    if (at != T) throw std::invalid_argument("run_rows: the group's spans do not cover the walk's rows");
    rows.span_floor = span_floor.data();
  }
  // DGPP_DSV41_CAPTURE_DECODE=1: an eager decode walk keeps every layer's
  // rows too (the decode-vs-prefill localizer in dsv41_model_test).
  // DGPP_DSV41_CAPTURE_PREFILL=1 likewise keeps a session prefill's
  // chunks' rows (the chunked-prefill gate's selection comparison).
  const bool capture_decode_env = std::getenv("DGPP_DSV41_CAPTURE_DECODE") != nullptr;
  const bool capture_prefill_env = std::getenv("DGPP_DSV41_CAPTURE_PREFILL") != nullptr;
  const bool capture_layers = run.capture_layers || (run.decode && !run.capture && capture_decode_env) ||
                              (!run.decode && !run.capture && capture_prefill_env);
  debug_capture_ = capture_layers;
  debug_sites_.clear();
  debug_index_logits_.clear();
  debug_layer_pre_.clear();
  // The teacher's streams replace the live ones before a layer (the
  // diagnostic forward only; the walk is synchronized per layer there).
  // The bounded prefill (plan §1.8, D8): the encoder layers over the chunk's
  // rows, the decoder's global KV published for every row at its first
  // layer, then the decoder over the replay segment only — the call's last
  // `sliding_window` rows (a chunk's tail carries over to the call's last
  // chunk, where the segment is assembled). `walk_rows` is the row count
  // the current layers walk; the head's rows land at `row_off` within the
  // chunk so finish_run's last row is the segment's.
  const bool bounded = prefill_bounded_ && !run.decode;
  const int dec0 = cfg_.decoder_first_layer();
  const int win = cfg_.sliding_window;
  int walk_rows = T;
  int row_off = 0;
  if (bounded) {
    if (run.capture) throw std::logic_error("run_rows: the bounded prefill is never captured");
    for (int e : cfg_.engram_layer_ids)
      if (e >= dec0) throw std::logic_error("run_rows: the bounded prefill needs the Engram layers inside the encoder");
    if (group) {
      // Every span is whole and within a window: no tail, the segment is the walk.
    } else if (run.first_chunk) {
      tail_rows_ = 0;
      tail_end_ = run.pos0;
      call_pos0_ = run.pos0;
    } else if (tail_end_ != run.pos0) {
      throw std::logic_error("run_rows: a bounded chunk must follow its call's previous chunk");
    }
  }
  // The teacher's streams replace the live ones before a layer (the
  // diagnostic forward only; the walk is synchronized per layer there).
  const auto teach = [&](int i) {
    const std::vector<uint16_t>& src = (*debug_teacher_)[static_cast<size_t>(i)];
    if (src.size() != static_cast<size_t>(walk_rows) * 4 * H) throw std::invalid_argument("forward: teacher streams shape");
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
    DGPP_CUDA_OK(cudaMemcpy(cur_, src.data(), src.size() * 2, cudaMemcpyHostToDevice));
    if (debug_teacher_pre_) {
      const std::vector<float>& pre = (*debug_teacher_pre_)[static_cast<size_t>(i)];
      if (pre.size() != static_cast<size_t>(walk_rows) * 4) throw std::invalid_argument("forward: teacher pre shape");
      if (pre_cur_ == one_hot_ || pre_cur_ == seg_pre_) pre_cur_ = pre_nxt_ == pre_a_ ? pre_b_ : pre_a_;
      DGPP_CUDA_OK(cudaMemcpy(pre_cur_, pre.data(), pre.size() * 4, cudaMemcpyHostToDevice));
    }
  };
  const auto rows_d2d = [&](void* dst, const void* src, size_t rows, size_t row_bytes) {
    if (rows > 0)
      DGPP_CUDA_OK(cudaMemcpyAsync(dst, src, rows * row_bytes, cudaMemcpyDeviceToDevice, stream_));
  };
  const size_t sb = static_cast<size_t>(4) * H * 2;  // a streams row's bytes
  const size_t pb = 4 * sizeof(float);               // a pre row's bytes
  std::vector<int> trace_rows;
  bool head = true;
  const int last_layer = (debug_layer_limit_ > 0 && debug_layer_limit_ < cfg_.num_hidden_layers && !run.decode)
                             ? debug_layer_limit_
                             : cfg_.num_hidden_layers;
  if (last_layer < cfg_.num_hidden_layers) head = false;  // the limited diagnostic walk: no head
  for (int layer = 0; layer < last_layer; ++layer) {
    const Dsv41LayerResident& r = loader_.load_layer(layer);
    if (debug_teacher_ && layer > 0) teach(layer - 1);
    if (bounded && layer == dec0) {
      // The decoder's global KV: this layer's attention-site input over the
      // chunk's rows (the site's coefficients are recomputed by the segment
      // walk below: the pointers stand). The compressor and index keys of
      // every row go to the pool now; the segment walk reads, never publishes.
      build_layer_objects(r);
      {
        float* const pc = pre_cur_;
        float* const pn = pre_nxt_;
        GlmMhcWeights aw;
        aw.fn = r.mhc.attn_fn;
        aw.base = r.mhc.attn_base;
        aw.scale = r.mhc.attn_scale;
        mhc_site(cur_, aw, r.attn_norm, T, rows.decode);
        pre_cur_ = pc;
        pre_nxt_ = pn;
      }
      if (!csa2_->prepare(T)) throw std::runtime_error("run_rows: CSA2 GEMM plans unavailable");
      if (group) {
        // Every span's rows published; then each span's segment — its last
        // `window` rows (the whole span when it fits) — packed into the
        // free streams buffer with its collapse coefficients: the decoder
        // walks the packed rows, span by span, never publishing again; a
        // span wider than a window floors its window at the segment's
        // start (the rows before it stand in the ring), a whole span at 0.
        for (int s = 0; s < run.num_spans; ++s) {
          if (!csa2_->prepare(run.span_lens[s])) throw std::runtime_error("run_rows: CSA2 GEMM plans unavailable");
          csa2_->publish_prefill(x_ + static_cast<size_t>(span_row0[static_cast<size_t>(s)]) * H, pool_, run.span_reqs[s],
                                 run.span_pos0[s], run.span_lens[s], stream_);
        }
        // The packed segments' coefficients go to the free rotating
        // buffer (seg_pre_ holds one window; a group's segments hold up to
        // a walk's rows).
        float* const packed_pre = pre_cur_ == pre_a_ ? pre_b_ : pre_a_;
        int packed = 0;
        for (int s = 0; s < run.num_spans; ++s) {
          const int len = run.span_lens[s], take = std::min(len, win);
          const int from = span_row0[static_cast<size_t>(s)] + len - take;
          rows_d2d(nxt_ + static_cast<size_t>(packed) * 4 * H, cur_ + static_cast<size_t>(from) * 4 * H,
                   static_cast<size_t>(take), sb);
          rows_d2d(packed_pre + static_cast<size_t>(packed) * 4, pre_cur_ + static_cast<size_t>(from) * 4,
                   static_cast<size_t>(take), pb);
          dec_lens.push_back(take);
          dec_pos0.push_back(run.span_pos0[s] + len - take);
          dec_row0.push_back(packed);
          span_floor[static_cast<size_t>(s)] = take < len ? dec_pos0.back() : 0;
          packed += take;
        }
        std::swap(cur_, nxt_);
        pre_cur_ = packed_pre;
        pre_nxt_ = packed_pre == pre_a_ ? pre_b_ : pre_a_;
        walk_rows = packed;
        row_off = 0;
        rows.span_lens = dec_lens.data();
        rows.span_pos0 = dec_pos0.data();
        rows.window_floor = 0;
        rows.publish = false;
        tail_rows_ = 0;
      } else {
      csa2_->publish_prefill(x_, pool_, req, run.pos0, T, stream_);
      const int take = std::min(T, win);
      if (!run.last_chunk) {
        // The tail: the last `window` rows of (old tail, this chunk) — the
        // encoder output and its collapse coefficients — for the call's last
        // chunk. No decoder, no head: this chunk's logits are unspecified.
        const int keep = std::min(tail_rows_, win - take);
        const int from = tail_rows_ - keep;
        if (keep > 0 && from > 0) {
          rows_d2d(nxt_, tail_streams_ + static_cast<size_t>(from) * 4 * H, static_cast<size_t>(keep), sb);
          rows_d2d(tail_streams_, nxt_, static_cast<size_t>(keep), sb);
          rows_d2d(seg_pre_, tail_pre_ + static_cast<size_t>(from) * 4, static_cast<size_t>(keep), pb);
          rows_d2d(tail_pre_, seg_pre_, static_cast<size_t>(keep), pb);
        }
        rows_d2d(tail_streams_ + static_cast<size_t>(keep) * 4 * H, cur_ + static_cast<size_t>(T - take) * 4 * H,
                 static_cast<size_t>(take), sb);
        rows_d2d(tail_pre_ + static_cast<size_t>(keep) * 4, pre_cur_ + static_cast<size_t>(T - take) * 4,
                 static_cast<size_t>(take), pb);
        tail_rows_ = keep + take;
        tail_end_ = run.pos0 + T;
        seg_rows_ = 0;
        segment_of(req) = {run.pos0, 0};
        head = false;
        break;
      }
      // The segment: the tail's last rows before this chunk's last rows,
      // `window` at most, into the free streams buffer; its coefficients
      // into seg_pre_. The window floors at the segment's start unless the
      // segment holds every row of the call (then the rows before it stand
      // in the ring: the previous call's segment or decode wrote them).
      const int from_tail = std::min(tail_rows_, win - take);
      const int R = from_tail + take;
      rows_d2d(nxt_, tail_streams_ + static_cast<size_t>(tail_rows_ - from_tail) * 4 * H, static_cast<size_t>(from_tail), sb);
      rows_d2d(seg_pre_, tail_pre_ + static_cast<size_t>(tail_rows_ - from_tail) * 4, static_cast<size_t>(from_tail), pb);
      rows_d2d(nxt_ + static_cast<size_t>(from_tail) * 4 * H, cur_ + static_cast<size_t>(T - take) * 4 * H,
               static_cast<size_t>(take), sb);
      rows_d2d(seg_pre_ + static_cast<size_t>(from_tail) * 4, pre_cur_ + static_cast<size_t>(T - take) * 4,
               static_cast<size_t>(take), pb);
      std::swap(cur_, nxt_);
      pre_cur_ = seg_pre_;
      pre_nxt_ = pre_a_;
      walk_rows = R;
      row_off = T - R;
      rows.pos0 = run.pos0 + T - R;
      rows.window_floor = rows.pos0 > call_pos0_ ? rows.pos0 : 0;
      rows.publish = false;
      tail_rows_ = 0;
      }  // !group
    }
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
    enqueue_layer(r, layer, walk_rows, rows);
    trace_rows.push_back(walk_rows);
    if (capture_layers) {
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      std::vector<uint16_t> snap(static_cast<size_t>(walk_rows) * 4 * H);
      DGPP_CUDA_OK(cudaMemcpy(snap.data(), cur_, snap.size() * 2, cudaMemcpyDeviceToHost));
      out.layer_states.push_back(std::move(snap));
      std::vector<float> pre(static_cast<size_t>(walk_rows) * 4);
      DGPP_CUDA_OK(cudaMemcpy(pre.data(), pre_cur_, pre.size() * 4, cudaMemcpyDeviceToHost));
      debug_layer_pre_.push_back(std::move(pre));
      if (cfg_.is_index_source(layer)) {
        std::vector<int32_t> sel(static_cast<size_t>(walk_rows) * static_cast<size_t>(cfg_.index_topk));
        DGPP_CUDA_OK(cudaMemcpy(sel.data(), csa2_->debug_topk(), sel.size() * 4, cudaMemcpyDeviceToHost));
        out.dsa_selections.push_back(std::move(sel));
        IndexLogits lg;
        lg.rows = run.decode ? 0 : csa2_->debug_logits_rows();
        lg.stride = csa2_->debug_logits_stride();
        lg.entries = csa2_->debug_logits_entries();
        if (lg.rows > 0) {
          lg.values.resize(static_cast<size_t>(lg.rows) * static_cast<size_t>(lg.stride));
          DGPP_CUDA_OK(cudaMemcpy(lg.values.data(), csa2_->debug_logits(), lg.values.size() * 4, cudaMemcpyDeviceToHost));
        }
        lg.q_rows = walk_rows;
        lg.q_heads = csa2_cfg_.index_heads;
        lg.q_codes.resize(static_cast<size_t>(walk_rows) * static_cast<size_t>(lg.q_heads) * kCsa2IndexDim);
        lg.q_scales.resize(static_cast<size_t>(walk_rows) * static_cast<size_t>(lg.q_heads));
        DGPP_CUDA_OK(cudaMemcpy(lg.q_codes.data(), csa2_->debug_index_q_codes(), lg.q_codes.size(), cudaMemcpyDeviceToHost));
        DGPP_CUDA_OK(cudaMemcpy(lg.q_scales.data(), csa2_->debug_index_q_scales(), lg.q_scales.size() * 4, cudaMemcpyDeviceToHost));
        if (layer == cfg_.candidate_source_layer_id || cfg_.uses_candidates(layer)) {
          lg.cand.resize(static_cast<size_t>(walk_rows) * static_cast<size_t>(csa2_cfg_.candidate_blocks));
          lg.cand_counts.resize(static_cast<size_t>(walk_rows));
          DGPP_CUDA_OK(cudaMemcpy(lg.cand.data(), csa2_->debug_cand(), lg.cand.size() * 4, cudaMemcpyDeviceToHost));
          DGPP_CUDA_OK(cudaMemcpy(lg.cand_counts.data(), csa2_->debug_cand_counts(), lg.cand_counts.size() * 4, cudaMemcpyDeviceToHost));
        }
        debug_index_logits_.push_back(std::move(lg));
      }
    }
  }
  if (head) {
    // The head: the weighted collapse with the last site's pre, the final
    // norm, the lm head on every walked row (bounded: the segment's rows,
    // at the chunk's last rows; the rows before them read as NaN).
    if (debug_teacher_ && debug_teacher_->size() >= static_cast<size_t>(cfg_.num_hidden_layers))
      teach(cfg_.num_hidden_layers - 1);
    if (row_off > 0 && run.all_rows) {
      DGPP_CUDA_OK(cudaMemsetAsync(logits_, 0xFF, static_cast<size_t>(row_off) * lm_vocab_count_ * sizeof(float), stream_));
      DGPP_CUDA_OK(cudaMemsetAsync(h_, 0xFF, static_cast<size_t>(row_off) * H * 2, stream_));
    }
    launch_mhc_collapse_normed(cur_, pre_cur_, nullptr, cfg_.rms_norm_eps, collapsed_ + static_cast<size_t>(row_off) * H,
                               nullptr, mhc_cfg_, walk_rows, stream_);
    csa2_rmsnorm_bf16(collapsed_ + static_cast<size_t>(row_off) * H, H, globals_.final_norm,
                      h_ + static_cast<size_t>(row_off) * H, H, walk_rows, H, cfg_.rms_norm_eps, stream_);
    gemm_.matmul(h_ + static_cast<size_t>(row_off) * H, globals_.lm_head,
                 logits_ + static_cast<size_t>(row_off) * lm_vocab_count_, walk_rows, lm_vocab_count_, H, DType::BF16,
                 GemmOut::F32, static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
    if (group && !dec_row0.empty()) {
      // The packed decoder's last row of each span into the row the
      // session core reads (the span's last prompt row), last span first:
      // every destination sits at or past its source and past the earlier
      // spans' sources.
      for (int s = run.num_spans - 1; s >= 0; --s) {
        const size_t src = static_cast<size_t>(dec_row0[static_cast<size_t>(s)] + dec_lens[static_cast<size_t>(s)] - 1);
        const size_t dst = static_cast<size_t>(span_row0[static_cast<size_t>(s)] + run.span_lens[s] - 1);
        if (src == dst) continue;
        rows_d2d(logits_ + dst * lm_vocab_count_, logits_ + src * lm_vocab_count_, 1,
                 static_cast<size_t>(lm_vocab_count_) * sizeof(float));
        rows_d2d(h_ + dst * H, h_ + src * H, 1, static_cast<size_t>(H) * 2);
      }
    }
    // DSpark: the last rows' target hidden into the slots' windows by
    // position (the draft gathers its accepted rows' from there); a
    // prefill's walked rows are the draft's rows (mtp_run_rows).
    if (mtp_) {
      const size_t W = static_cast<size_t>(targets_) * H;
      if (group) {
        // Per span: its last n walked rows — in the packed decoder space
        // under the bounded prefill (their positions are the span's last),
        // in place otherwise.
        for (int s = 0; s < run.num_spans; ++s) {
          const int len = run.span_lens[s], n = std::min(len, max_decode_rows_);
          const int walked = dec_row0.empty() ? span_row0[static_cast<size_t>(s)] : dec_row0[static_cast<size_t>(s)];
          const int wlen = dec_row0.empty() ? len : dec_lens[static_cast<size_t>(s)];
          const int nn = std::min(n, wlen);
          const int last_walked = walked + wlen - nn;
          const int last_orig = span_row0[static_cast<size_t>(s)] + run.span_lens[s] - nn;
          store_draft_hidden(main_hidden_ + static_cast<size_t>(last_walked) * W, in.req_ids + last_orig,
                             in.pos + last_orig, nn);
        }
      } else {
        const int n = std::min(walk_rows, max_decode_rows_);
        store_draft_hidden(main_hidden_ + static_cast<size_t>(walk_rows - n) * W, in.req_ids + (T - n), in.pos + (T - n), n);
      }
      draft_row_ = 0;
    }
    if (!run.decode) {
      seg_pos0_ = rows.pos0;
      seg_rows_ = walk_rows;
      if (group) {
        for (int s = 0; s < run.num_spans; ++s)
          segment_of(run.span_reqs[s]) = dec_row0.empty()
                                             ? std::pair<int64_t, int>{run.span_pos0[s], run.span_lens[s]}
                                             : std::pair<int64_t, int>{dec_pos0[static_cast<size_t>(s)], dec_lens[static_cast<size_t>(s)]};
      } else {
        segment_of(req) = {rows.pos0, walk_rows};
      }
    }
  }
  // The stream-ordered reducer's verdict for this pass's folds (a failed
  // collective is an error here, not a wrong number read later).
  if (!run.capture && boundary_) boundary_->settle();
  out = finish_run(run, std::move(out));
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

Dsv41Model::Outputs Dsv41Model::forward(const std::vector<int64_t>& token_ids, bool capture_layers,
                                        const std::vector<std::vector<uint16_t>>* teacher,
                                        const std::vector<std::vector<float>>* teacher_pre) {
  const int T = static_cast<int>(token_ids.size());
  if (T <= 0) throw std::invalid_argument("forward: empty token batch");
  if (teacher && !capture_layers) throw std::invalid_argument("forward: a teacher-forced walk captures its layers");
  if (teacher && teacher->size() + 1 < static_cast<size_t>(cfg_.num_hidden_layers))
    throw std::invalid_argument("forward: the teacher covers fewer layers than the model");
  if (teacher_pre && (!teacher || teacher_pre->size() < teacher->size()))
    throw std::invalid_argument("forward: the teacher's pre coefficients must match its streams");
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
  debug_teacher_pre_ = teacher_pre;
  Outputs out;
  try {
    out = run_rows(run);
  } catch (...) {
    debug_teacher_ = nullptr;
    debug_teacher_pre_ = nullptr;
    throw;
  }
  debug_teacher_ = nullptr;
  debug_teacher_pre_ = nullptr;
  session_close(0);
  return out;
}

}  // namespace dgpp
