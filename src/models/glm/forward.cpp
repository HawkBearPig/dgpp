#include "models/glm/forward.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "kernels/bf12_companions.hpp"
#include "kernels/glm_mhc_launch.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/scale_gemm.hpp"

namespace dgpp {

namespace {

double ms_between(std::chrono::steady_clock::time_point a,
                  std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

constexpr size_t kGemmWsBase = 64ull << 20;

// Activation scratch lives in DEVICE memory (2026-09-02; it was managed).
// Managed pages that both sides touch are a UVM fault factory: the host's
// per-step read of the logits row had the GPU re-faulting the page next
// step, and ~2% of decode steps paid a 9-10 ms fault-servicing stall that
// every other rank then waited out at the pick. Every host touch below
// goes through an explicit copy behind a stream sync (the diagnostic
// forward's captures) or a pinned mirror (the decode tail).
void* alloc_device(size_t bytes) {
  void* p = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&p, bytes));
  return p;
}

// Host vector of `n` bf16 from a device buffer, through the model's own
// stream — never the synchronous cudaMemcpy: that one waits on the legacy
// default stream, which in the in-process multi-rank tests means a PEER's
// collective kernel spinning on our doorbell (the deadlock class the
// lazy-loading fix documented). Our stream, our sync, nobody else's.
std::vector<uint16_t> fetch_bf16(const uint16_t* dev, size_t n,
                                 cudaStream_t stream) {
  std::vector<uint16_t> host(n);
  if (n) {
    DGPP_CUDA_OK(cudaMemcpyAsync(host.data(), dev, n * sizeof(uint16_t),
                                 cudaMemcpyDeviceToHost, stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  }
  return host;
}

std::vector<float> fetch_f32(const float* dev, size_t n,
                             cudaStream_t stream) {
  std::vector<float> host(n);
  if (n) {
    DGPP_CUDA_OK(cudaMemcpyAsync(host.data(), dev, n * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  }
  return host;
}

void upload_bf16(uint16_t* dev, const uint16_t* host, size_t n,
                 cudaStream_t stream) {
  DGPP_CUDA_OK(cudaMemcpyAsync(dev, host, n * sizeof(uint16_t),
                               cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
}

// TP configs: the geometry validators run inside from_config, so the
// head-divisibility rejection happens at construction, before any load.
template <typename Cfg>
Cfg with_tp(Cfg c, int world) {
  c.tp_size = world;
  return c;
}

}  // namespace

GlmDiagnosticModel::GlmDiagnosticModel(const GlmTextConfig& cfg,
                                        const std::string& checkpoint_dir,
                                        int max_tokens,
                                        int64_t max_cache_tokens,
                                        GlmBoundaryReducer* boundary,
                                        int tp_rank, int tp_world,
                                        GlmResidency residency,
                                        GlmHeadSharding head,
                                        int max_requests, bool mtp,
                                        LatentFormat kv_format)
    : cfg_(cfg),
      kda_cfg_(with_tp(cfg.kda_config(), tp_world)),
      dsa_cfg_(with_tp(cfg.dsa_config(), tp_world)),
      mhc_cfg_(cfg.mhc_config()),
      moe_cfg_(cfg.moe_config()),
      kda_geo_(KdaGeometry::from_config(kda_cfg_)),
      max_tokens_(max_tokens),
      loader_(cfg, checkpoint_dir, tp_rank, tp_world, residency, head,
              /*resident_mtp=*/mtp),
      mtp_(mtp) {
  if (max_tokens_ <= 0)
    throw std::invalid_argument("GlmDiagnosticModel: max_tokens must be positive");
  if (mtp_ && cfg_.mtp_layer() < 0)
    throw std::invalid_argument(
        "GlmDiagnosticModel: mtp requested but the config has no draft layer "
        "(num_nextn_predict_layers == 0)");
  // The draft layer is a DSA layer with its own cache: one more pool
  // ordinal, after the main stack's (the pool and the layer object share
  // the config, so both see the extra ordinal).
  main_dsa_layers_ = dsa_cfg_.num_dsa_layers;
  if (mtp_) dsa_cfg_.num_dsa_layers += 1;
  // The latent cache's format (validated against the geometry: the
  // quantized rows' alignment pins).
  dsa_cfg_.latent_format = kv_format;
  // The dense sites' lowering (kernels/gemm.hpp dense_gemv_rows): the GEMV
  // chunks to the bound, cuBLASLt's algorithm (bf16) or the streaming
  // tensor-core GEMM (fp8: the DSA projections, the dense MLP) above it.
  dsa_cfg_.gemm_mma_from_rows = dense_gemv_rows() + 1;
  dense_mma_from_rows_ = dense_gemv_rows() + 1;
  gemm_.set_decode_rows(std::min(kDecodeRows, dense_gemv_rows()));
  DsaConfig::validate_config(dsa_cfg_);
  if (max_cache_tokens < max_tokens_)
    max_cache_tokens = max_tokens_;
  if (max_requests <= 0)
    throw std::invalid_argument(
        "GlmDiagnosticModel: max_requests must be positive");
  max_requests_ = max_requests;
  if ((tp_world > 1) != (boundary != nullptr))
    throw std::invalid_argument(
        "GlmDiagnosticModel: a boundary reducer is required exactly when "
        "tp_world > 1 — without one the block-boundary partials would be "
        "returned silently as results");
  boundary_ = boundary;

  // Boot check (§5.2): hash every replicated tensor BEFORE anything else
  // runs — runners exchange this across ranks at startup, and a mismatch
  // pinpoints the diverging layer. Pure mmap reads; no residency needed.
  const auto t_digest = std::chrono::steady_clock::now();
  if (tp_world > 1) boot_digest_ = loader_.hash_replicated();
  const auto t_globals = std::chrono::steady_clock::now();
  globals_ = loader_.load_globals();
  boot_digest_ms_ = ms_between(t_digest, t_globals);
  boot_globals_ms_ = ms_between(t_globals, std::chrono::steady_clock::now());
  lm_vocab_begin_ = globals_.lm_vocab_begin;
  lm_vocab_count_ =
      globals_.lm_vocab_count > 0 ? globals_.lm_vocab_count : cfg_.vocab_size;

  // Highest priority: the decode chain's kernels take SM slots ahead of
  // the L2 prefetch stream's (kernels/l2_prefetch.hpp), which is created
  // at the lowest. Blocking w.r.t. the legacy stream, as before.
  {
    int least = 0, greatest = 0;
    DGPP_CUDA_OK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    DGPP_CUDA_OK(cudaStreamCreateWithPriority(&stream_, cudaStreamDefault,
                                              greatest));
    // The comb side stream at the LOWEST priority: its one-warp kernel
    // takes a block slot only where the sublayer's kernels leave one (at
    // the highest it stole a slot from the KDA GEMV it ran beside: +0.2 ms
    // of GEMV time per step in the 2026-09-08 trace), and the update that
    // joins it is hundreds of microseconds away.
    DGPP_CUDA_OK(cudaStreamCreateWithPriority(&mhc_side_, cudaStreamNonBlocking,
                                              least));
    DGPP_CUDA_OK(cudaEventCreateWithFlags(&mhc_fork_, cudaEventDisableTiming));
    DGPP_CUDA_OK(cudaEventCreateWithFlags(&mhc_join_, cudaEventDisableTiming));
    if (const char* e = std::getenv("DGPP_MHC_COMB_SIDE"))
      mhc_comb_side_ = !(e[0] == '0');
  }
  // The model's kernels are the resident layers' readers: load boundaries
  // synchronize exactly this stream (+ the loader's dequant stream), not
  // the whole device — a device-wide wait in a one-process multi-rank
  // world deadlocks against a peer's spinning collective kernel (the
  // loopback first-collective stall; see set_reader_stream).
  loader_.set_reader_stream(stream_);

  if (tp_world > 1)
    tp_ = std::make_unique<GlmTpViews>(cfg, tp_rank, tp_world, stream_);

  // GEMM workspace: 64 MB covers every M2/M3 shape; the lm head's N
  // (this rank's vocab slice in sharded mode) is the only dimension
  // larger than anything tested there.
  gemm_ws_bytes_ = kGemmWsBase;
  {
    const size_t head_ws = gemm_.query_workspace_bytes(
        max_tokens_, lm_vocab_count_, cfg_.hidden_size, DType::BF16);
    if (head_ws > gemm_ws_bytes_) gemm_ws_bytes_ = head_ws;
  }
  DGPP_CUDA_OK(cudaMalloc(&gemm_ws_, gemm_ws_bytes_));

  // The DSA pool shares the arena with the KDA layer scratch (both take
  // persistent-hot grants at construction/init).
  const int64_t dsa_slots =
      ((max_cache_tokens + dsa_cfg_.block_tokens - 1) /
       dsa_cfg_.block_tokens) *
      dsa_cfg_.block_tokens;
  // The context bound: the pool's slots (every position a session reaches
  // has a latent row) — a model without DSA layers is bounded by the
  // larger of the two numbers the caller gave.
  max_context_ = dsa_cfg_.num_dsa_layers > 0
                     ? dsa_slots
                     : std::max<int64_t>(max_cache_tokens, max_tokens_);
  Arena::Config ac;
  ac.persistent_hot = KdaLayer::persistent_hot_bytes(kda_cfg_, max_tokens_);
  if (dsa_cfg_.num_dsa_layers > 0) {
    ac.persistent_hot += DsaStatePool::cache_bytes(dsa_cfg_, max_requests_,
                                                   dsa_slots);
    arena_.init(ac);
    pool_.init(arena_, dsa_cfg_, max_requests_, dsa_slots);
    dsa_scratch_ = static_cast<uint8_t*>(
        alloc_device(DsaLayer::scratch_bytes(dsa_cfg_, max_tokens_,
                                              dsa_slots)));
  } else {
    arena_.init(ac);
  }

  // Decode-session row metadata is shared by DSA, request-indexed KDA, and
  // the Phase-2 position/pick kernels. Allocate it even on a KDA-only
  // configuration; it is lifetime-stable because CUDA graphs bake it.
  d_req_ids_ = static_cast<int32_t*>(
      alloc_device(sizeof(int32_t) * kDecodeRows));
  d_step_pos_ = static_cast<int64_t*>(
      alloc_device(sizeof(int64_t) * kDecodeRows));
  d_req_spans_ = static_cast<int32_t*>(
      alloc_device(sizeof(int32_t) * 2 * kDecodeRows));

  // The decode path's H2D upload sources, PINNED at construction
  // (pageable async copies stream-sync before initiating — a per-step
  // pipeline drain the decode path refuses). Addresses are stable for the
  // model's lifetime — the graph bakes them. All four are DEVICE-MAPPED:
  // the graph uploads them with kernels that read the pinned buffers
  // directly (glm_upload_i32/i64), never memcpy nodes — a memcpy node
  // rides the process-shared copy-engine queue, the batched-MTP graph
  // stall (docs/batched_mtp_graph_stall.md). Under UVA the host pointer
  // IS the device pointer; the check below pins that.
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_req_ids_),
                             sizeof(int32_t) * kDecodeRows,
                             cudaHostAllocMapped));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_step_pos_),
                             sizeof(int64_t) * kDecodeRows,
                             cudaHostAllocMapped));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_req_spans_),
                             sizeof(int32_t) * 2 * kDecodeRows,
                             cudaHostAllocMapped));
  // The pinned token rows: the decode rows, or every slot's feed rows
  // (session_graph_seed_feed) when those are more.
  DGPP_CUDA_OK(cudaHostAlloc(
      reinterpret_cast<void**>(&h_token_),
      sizeof(int64_t) * (static_cast<size_t>(kDecodeRows) +
                         static_cast<size_t>(max_requests_) *
                             static_cast<size_t>(kSpecRows)),
      cudaHostAllocMapped));
  for (void* host : {static_cast<void*>(h_req_ids_),
                     static_cast<void*>(h_step_pos_),
                     static_cast<void*>(h_req_spans_),
                     static_cast<void*>(h_token_)}) {
    void* dev = nullptr;
    DGPP_CUDA_OK(cudaHostGetDevicePointer(&dev, host, 0));
    if (dev != host)
      throw std::runtime_error(
          "decode upload sources: the mapped pinned buffer's device address "
          "differs from its host address (no UVA?) — the kernel upload "
          "needs one address");
  }
  // The device-side session positions (the device-driven graph's source of
  // truth; see push_position) and their pinned upload mirror.
  d_session_pos_ = static_cast<int64_t*>(
      alloc_device(sizeof(int64_t) * static_cast<size_t>(max_requests_)));
  DGPP_CUDA_OK(cudaMemset(d_session_pos_, 0,
                          sizeof(int64_t) * static_cast<size_t>(max_requests_)));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_session_pos_),
                             sizeof(int64_t) * static_cast<size_t>(max_requests_),
                             cudaHostAllocDefault));
  d_next_ = static_cast<int64_t*>(
      alloc_device(sizeof(int64_t) * static_cast<size_t>(max_requests_)));
  DGPP_CUDA_OK(cudaMemset(d_next_, 0,
                          sizeof(int64_t) * static_cast<size_t>(max_requests_)));

  // Per-request, per-layer KDA state (slot-major: one memset pair per
  // request open — see the header's layout note).
  if (kda_cfg_.num_kda_layers > 0) {
    kda_rec_ = static_cast<float*>(alloc_device(
        static_cast<size_t>(kda_cfg_.num_kda_layers) * max_requests_ *
        kda_geo_.recurrent_bytes));
    kda_conv_ = static_cast<uint16_t*>(alloc_device(
        static_cast<size_t>(kda_cfg_.num_kda_layers) * max_requests_ *
        kda_geo_.conv_committed_bytes));
  }
  session_pos_.assign(static_cast<size_t>(max_requests_), 0);
  prefill_epochs_.assign(static_cast<size_t>(max_requests_), 0);
  // Speculative verify snapshots (one in-flight verify; see the header).
  if (kda_cfg_.num_kda_layers > 0) {
    const size_t per_row = static_cast<size_t>(kda_cfg_.num_kda_layers);
    spec_rec_ = static_cast<float*>(
        alloc_device(kDecodeRows * per_row * kda_geo_.recurrent_bytes));
    spec_conv_ = static_cast<uint16_t*>(alloc_device(
        kDecodeRows * per_row * kda_geo_.conv_committed_bytes));
  }
  if (main_dsa_layers_ > 0)
    spec_tail_ = static_cast<uint16_t*>(
        alloc_device(static_cast<size_t>(main_dsa_layers_) * kDecodeRows *
                     spec_tail_ring_elems() * 2));
  if (mtp_) {
    const size_t H = static_cast<size_t>(cfg_.hidden_size);
    mtp_pos_.assign(static_cast<size_t>(max_requests_), 0);
    mtp_ring_snapshot_ = static_cast<uint16_t*>(alloc_device(
        static_cast<size_t>(max_requests_) * spec_tail_ring_elems() * 2));
    mtp_chain_ring_ = static_cast<uint16_t*>(alloc_device(
        static_cast<size_t>(max_requests_) * spec_tail_ring_elems() * 2));
    d_mtp_pos_ = static_cast<int64_t*>(
        alloc_device(sizeof(int64_t) * static_cast<size_t>(max_requests_)));
    DGPP_CUDA_OK(cudaMemset(d_mtp_pos_, 0,
                            sizeof(int64_t) * static_cast<size_t>(max_requests_)));
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_mtp_pos_),
                               sizeof(int64_t) * static_cast<size_t>(max_requests_),
                               cudaHostAllocDefault));
    mtp_hidden_ = static_cast<uint16_t*>(alloc_device(
        static_cast<size_t>(max_requests_) *
        static_cast<size_t>(max_context_) * H * 2));
    mtp_cat_ = static_cast<uint16_t*>(
        alloc_device(static_cast<size_t>(max_tokens_) * 2 * H * 2));
    mtp_x_ = static_cast<uint16_t*>(
        alloc_device(static_cast<size_t>(max_tokens_) * H * 2));
  }

  // Decode-path route traces: pinned staging the decode MoE's async
  // copies land in (see glm_moe_layer.hpp's MoeTraceStaging). Pinned —
  // the copies are issued mid-step with no sync, and pinned destinations
  // keep them true async D2H.
  n_moe_layers_ = static_cast<int>(std::count_if(
      cfg_.mlps.begin(), cfg_.mlps.end(),
      [](GlmMlpKind k) { return k == GlmMlpKind::Moe; }));
  if (n_moe_layers_ > 0) {
    const size_t rows = static_cast<size_t>(n_moe_layers_) * kDecodeRows;
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&moe_trace_ids_),
                               rows * moe_cfg_.top_k * sizeof(int32_t),
                               cudaHostAllocDefault));
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&moe_trace_weights_),
                               rows * moe_cfg_.top_k * sizeof(float),
                               cudaHostAllocDefault));
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&moe_trace_biased_),
                               rows * moe_cfg_.n_experts * sizeof(float),
                               cudaHostAllocDefault));
    const size_t prows =
        static_cast<size_t>(n_moe_layers_) * static_cast<size_t>(max_tokens_);
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&moe_prefill_trace_ids_),
                               prows * moe_cfg_.top_k * sizeof(int32_t),
                               cudaHostAllocDefault));
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&moe_prefill_trace_weights_),
                               prows * moe_cfg_.top_k * sizeof(float),
                               cudaHostAllocDefault));
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&moe_prefill_trace_biased_),
                               prows * moe_cfg_.n_experts * sizeof(float),
                               cudaHostAllocDefault));
  }

  // Activations.
  const size_t T = static_cast<size_t>(max_tokens_);
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  d_tokens_ = static_cast<int64_t*>(alloc_device(T * 8));
  DGPP_CUDA_OK(cudaMemset(d_tokens_, 0, T * 8));
  streams_[0] = static_cast<uint16_t*>(alloc_device(T * 4 * H * 2));
  streams_[1] = static_cast<uint16_t*>(alloc_device(T * 4 * H * 2));
  post_ = static_cast<uint16_t*>(alloc_device(T * 4 * 2));
  mhc_logits_ = static_cast<float*>(
      alloc_device(T * static_cast<size_t>(mhc_cfg_.coeff_rows()) * 4));
  mhc_counters_ = static_cast<int*>(alloc_device(T * sizeof(int)));
  DGPP_CUDA_OK(cudaMemset(mhc_counters_, 0, T * sizeof(int)));
  comb_ = static_cast<uint16_t*>(alloc_device(T * 16 * 2));
  collapsed_ = static_cast<uint16_t*>(alloc_device(T * H * 2));
  normed_ = static_cast<uint16_t*>(alloc_device(T * H * 2));
  sub_out_ = static_cast<uint16_t*>(alloc_device(T * H * 2));
  if (cfg_.first_k_dense_replace > 0) {
    const size_t I = static_cast<size_t>(cfg_.intermediate_size);
    dense_g_ = static_cast<uint16_t*>(alloc_device(T * I * 2));
    dense_u_ = static_cast<uint16_t*>(alloc_device(T * I * 2));
    dense_act_ = static_cast<uint16_t*>(alloc_device(T * I * 2));
  }
  logits_ = static_cast<float*>(
      alloc_device(T * lm_vocab_count_ * sizeof(float)));
  DGPP_CUDA_OK(cudaHostAlloc(
      reinterpret_cast<void**>(&h_tail_logits_),
      static_cast<size_t>(kDecodeRows) * lm_vocab_count_ * sizeof(float),
      cudaHostAllocDefault));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_tail_hidden_),
                             static_cast<size_t>(kDecodeRows) * H * 2,
                             cudaHostAllocDefault));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_graph_start_gt_),
                             sizeof(uint64_t), cudaHostAllocDefault));
  *h_graph_start_gt_ = 0;

  // Every device allocation happens above (see preconstruct_layers): the
  // TP runners barrier after construction so no rank's first collective
  // can spin while a peer is still inside allocation-phase device syncs.
  const auto t_layers = std::chrono::steady_clock::now();
  preconstruct_layers();
  if (cfg_.vision) {
    vision_ = std::make_unique<GlmVisionEncoder>(*cfg_.vision, checkpoint_dir, stream_);
    boot_digest_.globals ^= vision_->digest();
    boot_digest_.bytes += cfg_.vision->weight_bytes();
    boot_digest_.tensors += 14 * cfg_.vision->depth + 11;
  }
  // The startup budget, itemized — what the resident image cache leaves
  // behind is the digest and the globals, both still read from the shards.
  DGPP_LOG_INFO("rank {} boot phases: digest {:.1f} s ({:.2f} GiB hashed), "
                "globals {:.1f} s, layers {:.1f} s",
                tp_rank, boot_digest_ms_ / 1000.0,
                static_cast<double>(boot_digest_.bytes) /
                    (1024.0 * 1024.0 * 1024.0),
                boot_globals_ms_ / 1000.0,
                ms_between(t_layers, std::chrono::steady_clock::now()) / 1000.0);
}

// The lossless 12-bit companions of the decode GEMV's bf16 weights
// (kernels/bf12_gemv.hpp): a resident layer's KDA in/out projections and
// the draft block's eh_proj are packed as the layer lands — read back
// through the model's stream, encoded on the host, uploaded, registered
// with the GEMM — so the decode launches stream 0.75 of the bytes and
// produce the same bits. Under bf12-only residency the matrix's own bytes
// go back to the device at once (the caches already exist when the layers
// load: only one layer's bf16 bytes ever sit beside their companions), and
// a prefill GEMM expands what it reads (kernels/gemm.hpp); otherwise wider
// calls and prefill keep the bf16 bytes. A matrix outside the contract
// (shape, or too many rows with a pathological escape count) simply keeps
// its bf16 form.
void GlmDiagnosticModel::pack_layer_companions(int layer, const GlmLayerResident& r) {
  const auto t0 = std::chrono::steady_clock::now();
  const int H = cfg_.hidden_size;
  const auto release = [&](const void* w) { return loader_.release_packed(layer, w); };
  if (r.kind == GlmLayerKind::Kda) {
    bf12_.pack_and_release(static_cast<const uint16_t*>(r.kda.in_proj), kda_geo_.in_proj_cols, H, gemm_,
                           stream_, release);
    bf12_.pack_and_release(static_cast<const uint16_t*>(r.kda.o_proj), H, kda_geo_.local_proj, gemm_,
                           stream_, release);
  }
  if (layer == cfg_.mtp_layer() && r.eh_proj != nullptr)
    bf12_.pack_and_release(r.eh_proj, H, 2 * H, gemm_, stream_, release);
  bf12_ms_ += ms_between(t0, std::chrono::steady_clock::now());
}

// The lm head's, and the summary; the expansion scratch takes two slots —
// the fold overlap's row blocks call a site's in and o projections twice.
void GlmDiagnosticModel::finish_companions() {
  const auto t0 = std::chrono::steady_clock::now();
  bf12_.pack_and_release(static_cast<const uint16_t*>(globals_.lm_head), lm_vocab_count_,
                         cfg_.hidden_size, gemm_, stream_,
                         [&](const void* w) { return loader_.release_packed(-1, w); });
  bf12_.finish(gemm_, kBf12ExpandSlots);
  bf12_ms_ += ms_between(t0, std::chrono::steady_clock::now());
  bf12_.log_summary(loader_.rank(), bf12_ms_ / 1000.0);
}

size_t GlmDiagnosticModel::kv_block_bytes_static(const GlmTextConfig& cfg, int tp_world, bool mtp,
                                                 LatentFormat kv_format) {
  DsaConfig dsa_cfg = with_tp(cfg.dsa_config(), tp_world);
  if (dsa_cfg.num_dsa_layers == 0) return 0;
  if (mtp) dsa_cfg.num_dsa_layers += 1;
  dsa_cfg.latent_format = kv_format;
  DsaConfig::validate_config(dsa_cfg);
  const DsaGeometry g = DsaGeometry::from_config(dsa_cfg);
  const size_t layers = static_cast<size_t>(dsa_cfg.num_dsa_layers);
  const size_t ilayers = static_cast<size_t>(g.index_layers);
  const size_t pools = static_cast<size_t>(g.pools_per_block);
  return layers * static_cast<size_t>(dsa_cfg.block_tokens) *
             (g.latent_bytes_per_token + g.latent_scale_bytes_per_token) +
         ilayers * pools * (g.index_k_bytes_per_pool + sizeof(float));
}

GlmDiagnosticModel::MemoryPlan GlmDiagnosticModel::plan_memory(
    const GlmTextConfig& cfg, int max_tokens, int64_t max_cache_tokens,
    int tp_rank, int tp_world, GlmResidency residency, GlmHeadSharding head,
    int max_requests, bool mtp, LatentFormat kv_format) {
  if (max_tokens <= 0)
    throw std::invalid_argument("plan_memory: max_tokens must be positive");
  if (max_requests <= 0)
    throw std::invalid_argument("plan_memory: max_requests must be positive");
  MemoryPlan plan;
  if (cfg.vision) {
    plan.add("vision weights (replicated)", cfg.vision->weight_bytes());
    plan.add("vision workspace (bounded image prefill)", cfg.vision->workspace_bytes());
  }
  const KdaConfig kda_cfg = with_tp(cfg.kda_config(), tp_world);
  DsaConfig dsa_cfg = with_tp(cfg.dsa_config(), tp_world);
  const int main_dsa_layers = dsa_cfg.num_dsa_layers;
  if (mtp) dsa_cfg.num_dsa_layers += 1;
  dsa_cfg.latent_format = kv_format;
  DsaConfig::validate_config(dsa_cfg);
  const KdaGeometry kda_geo = KdaGeometry::from_config(kda_cfg);
  const GlmMoeConfig moe_cfg = cfg.moe_config();
  const GlmMhcConfig mhc_cfg = cfg.mhc_config();
  if (max_cache_tokens < max_tokens) max_cache_tokens = max_tokens;
  const int64_t dsa_slots =
      ((max_cache_tokens + dsa_cfg.block_tokens - 1) / dsa_cfg.block_tokens) *
      dsa_cfg.block_tokens;
  const int64_t max_context =
      dsa_cfg.num_dsa_layers > 0 ? dsa_slots
                                 : std::max<int64_t>(max_cache_tokens, max_tokens);
  plan.context_tokens = max_context;
  const size_t T = static_cast<size_t>(max_tokens);
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t R = static_cast<size_t>(max_requests);
  const size_t V = static_cast<size_t>(
      GlmLayerStream::lm_vocab_count(cfg, tp_rank, tp_world, head));
  int n_moe = 0;
  for (GlmMlpKind k : cfg.mlps) n_moe += k == GlmMlpKind::Moe ? 1 : 0;

  // The weights: every layer resident, or one layer's bump when streaming;
  // the loader's pinned staging mirror is the largest of them.
  size_t largest_layer = 0;
  const int layers_total = cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
  for (int l = 0; l < layers_total; ++l)
    largest_layer = std::max(largest_layer,
                             GlmLayerStream::layer_bytes(cfg, l, tp_rank, tp_world));
  const size_t globals = GlmLayerStream::globals_bytes(cfg, tp_rank, tp_world, head);
  // bf12-only residency: the packable bf16 matrices load aside and give
  // their bytes back once packed (the companions are their own line below).
  const bool bf12_only = Bf12Companions::packed_only() && residency == GlmResidency::Resident;
  if (residency == GlmResidency::Resident) {
    size_t resident = globals;
    for (int l = 0; l < cfg.num_hidden_layers; ++l)
      resident += GlmLayerStream::layer_bytes(cfg, l, tp_rank, tp_world);
    if (mtp && cfg.mtp_layer() >= 0)
      resident += GlmLayerStream::layer_bytes(cfg, cfg.mtp_layer(), tp_rank, tp_world);
    if (bf12_only) resident -= GlmLayerStream::side_bytes(cfg, tp_rank, tp_world, head, mtp);
    plan.add(bf12_only ? "model weights (resident, packed bf16 matrices released)"
                       : "model weights (resident)",
             resident, std::max(largest_layer, globals));
  } else {
    plan.add("model weights (one streamed layer + globals)", largest_layer + globals,
             std::max(largest_layer, globals));
  }
  if (tp_world > 1)
    plan.add("tensor-parallel slice views",
             GlmTpViews::slice_bytes(cfg, tp_world) +
                 GlmTpViews::expert_pack_bytes(cfg, tp_world));
  if (Bf12Companions::enabled() && residency == GlmResidency::Resident) {
    size_t packed = Bf12Companions::planned_bytes(static_cast<int64_t>(V), cfg.hidden_size);
    const size_t per_kda = Bf12Companions::planned_bytes(kda_geo.in_proj_cols, cfg.hidden_size) +
                           Bf12Companions::planned_bytes(cfg.hidden_size, kda_geo.local_proj);
    packed += static_cast<size_t>(kda_cfg.num_kda_layers) * per_kda;
    if (mtp) packed += Bf12Companions::planned_bytes(cfg.hidden_size, 2 * cfg.hidden_size);
    plan.add("bf16 decode packing (12-bit companions)", packed);
    if (bf12_only) {
      const size_t Hh = static_cast<size_t>(cfg.hidden_size);
      plan.add("bf16 prefill expansion scratch",
               static_cast<size_t>(kBf12ExpandSlots) *
                   Bf12Companions::planned_slot_bytes(
                       {static_cast<size_t>(kda_geo.in_proj_cols) * Hh * 2,
                        Hh * static_cast<size_t>(kda_geo.local_proj) * 2, mtp ? Hh * 2 * Hh * 2 : 0,
                        V * Hh * 2}));
    }
  }
  plan.add("gemm workspace (at least)", kGemmWsBase);
  plan.add("kda scratch", KdaLayer::persistent_hot_bytes(kda_cfg, max_tokens));
  if (dsa_cfg.num_dsa_layers > 0) {
    plan.add(std::string("kv cache pool (") + latent_format_name(kv_format) + ")",
             DsaStatePool::cache_bytes(dsa_cfg, max_requests, dsa_slots));
    plan.add("dsa scratch", DsaLayer::scratch_bytes(dsa_cfg, max_tokens, dsa_slots));
  }
  if (n_moe > 0) {
    size_t pinned = 0;
    const size_t dev = GlmMoeLayer::scratch_bytes(moe_cfg, max_tokens, kDecodeRows,
                                                  n_moe + (mtp ? 1 : 0), &pinned);
    plan.add("moe scratch", dev, pinned);
    if (cfg.routed_expert_format == GlmExpertFormat::Nvfp4Group16) {
      GlmMoeConfig local_moe = moe_cfg;
      local_moe.inter /= tp_world;
      plan.add("moe W4A4 activation workspace",
               GlmMoeLayer::w4a4_scratch_bytes(local_moe, max_tokens));
    }
  }
  if (kda_cfg.num_kda_layers > 0) {
    const size_t layers = static_cast<size_t>(kda_cfg.num_kda_layers);
    const size_t per_slot =
        layers * (kda_geo.recurrent_bytes + kda_geo.conv_committed_bytes);
    plan.add("kda state (request slots)", R * per_slot);
    plan.add("kda speculative snapshots", static_cast<size_t>(kDecodeRows) * per_slot);
  }
  if (main_dsa_layers > 0) {
    const size_t ring =
        static_cast<size_t>(2) * dsa_cfg.index_kpool * dsa_cfg.index_head_dim * 2;
    size_t bytes = static_cast<size_t>(main_dsa_layers) * kDecodeRows * ring;
    if (mtp) bytes += R * ring;
    plan.add("dsa tail-ring snapshots", bytes);
  }
  if (mtp) {
    plan.add("draft hidden cache (per position)",
             R * static_cast<size_t>(max_context) * H * 2);
    plan.add("draft activations", T * 2 * H * 2 + T * H * 2);
  }
  {
    size_t act = T * 8;                                   // d_tokens_
    act += 2 * (T * 4 * H * 2);                           // streams_
    act += T * 4 * 2 + T * static_cast<size_t>(mhc_cfg.coeff_rows()) * 4 +
           T * 4 + T * 16 * 2;                            // post_, mhc, comb_
    act += 3 * (T * H * 2);                               // collapsed_, normed_, sub_out_
    if (cfg.first_k_dense_replace > 0)
      act += 3 * (T * static_cast<size_t>(cfg.intermediate_size) * 2);
    act += T * V * 4;                                     // logits_
    act += R * 8 * 3 + static_cast<size_t>(kDecodeRows) * (4 + 8 + 8);
    plan.add("activations (per-forward rows)", act,
             static_cast<size_t>(kDecodeRows) * V * 4 +
                 static_cast<size_t>(kDecodeRows) * H * 2);
  }
  if (n_moe > 0) {
    const size_t K = static_cast<size_t>(moe_cfg.top_k);
    const size_t E = static_cast<size_t>(moe_cfg.n_experts);
    const size_t per_row = K * 4 * 2 + E * 4;
    plan.add("route-trace staging", 0,
             static_cast<size_t>(n_moe) * (static_cast<size_t>(kDecodeRows) + T) * per_row);
  }
  return plan;
}

GlmDiagnosticModel::~GlmDiagnosticModel() {
  vision_.reset();
  cudaFree(gemm_ws_);
  cudaFree(dsa_scratch_);
  cudaFree(d_req_ids_);
  cudaFree(d_step_pos_);
  cudaFree(d_req_spans_);
  cudaFreeHost(h_req_ids_);
  cudaFreeHost(h_step_pos_);
  cudaFreeHost(h_req_spans_);
  cudaFreeHost(h_token_);
  cudaFree(d_session_pos_);
  cudaFreeHost(h_session_pos_);
  cudaFree(d_next_);
  cudaFree(d_mtp_pos_);
  cudaFreeHost(h_mtp_pos_);
  cudaFree(kda_rec_);
  cudaFree(kda_conv_);
  cudaFree(spec_rec_);
  cudaFree(spec_conv_);
  cudaFree(spec_tail_);
  cudaFree(mtp_hidden_);
  cudaFree(mtp_ring_snapshot_);
  cudaFree(mtp_chain_ring_);
  cudaFree(mtp_cat_);
  cudaFree(mtp_x_);
  cudaFree(d_tokens_);
  cudaFree(streams_[0]);
  cudaFree(streams_[1]);
  cudaFree(post_);
  cudaFree(mhc_logits_);
  cudaFree(mhc_counters_);
  cudaFree(comb_);
  cudaFree(collapsed_);
  cudaFree(normed_);
  cudaFree(sub_out_);
  cudaFree(dense_g_);
  cudaFree(dense_u_);
  cudaFree(dense_act_);
  cudaFree(logits_);
  cudaFreeHost(h_tail_logits_);
  cudaFreeHost(h_tail_hidden_);
  cudaFreeHost(h_graph_start_gt_);
  cudaFreeHost(moe_trace_ids_);
  cudaFreeHost(moe_trace_weights_);
  cudaFreeHost(moe_trace_biased_);
  if (moe_prefill_trace_ids_) cudaFreeHost(moe_prefill_trace_ids_);
  if (moe_prefill_trace_weights_) cudaFreeHost(moe_prefill_trace_weights_);
  if (moe_prefill_trace_biased_) cudaFreeHost(moe_prefill_trace_biased_);
  if (mhc_join_) cudaEventDestroy(mhc_join_);
  if (mhc_fork_) cudaEventDestroy(mhc_fork_);
  if (mhc_side_) cudaStreamDestroy(mhc_side_);
  if (stream_) cudaStreamDestroy(stream_);
}

// ---- DSA admission meters (the scheduler's budget interface, Stage 2b) ------
// A no-DSA model reports an unbounded pool: admission then keys on the
// engine slot count alone. INT64_MAX (not "huge") so the scheduler's
// subtraction arithmetic cannot overflow a real capacity.
int64_t GlmDiagnosticModel::kv_blocks_total() const {
  return dsa_cfg_.num_dsa_layers > 0 ? pool_.total_blocks() : INT64_MAX;
}
int64_t GlmDiagnosticModel::kv_blocks_in_use() const {
  return dsa_cfg_.num_dsa_layers > 0 ? pool_.blocks_in_use() : int64_t(0);
}
int64_t GlmDiagnosticModel::kv_blocks_for_tokens(int64_t tokens) const {
  return dsa_cfg_.num_dsa_layers > 0 ? pool_.block_count_for_tokens(tokens)
                                    : int64_t(0);
}

GlmMoeWeights GlmDiagnosticModel::moe_weights(const GlmMoeResident& r) {
  GlmMoeWeights w;
  w.router_gate = r.router_gate;
  w.router_bias = r.router_bias;
  w.experts = r.experts.empty() ? nullptr : r.experts.data();
  w.experts_fp4 = r.experts_fp4.empty() ? nullptr : r.experts_fp4.data();
  for (int i = 0; i < 3; ++i) w.shared[i] = r.shared[i];
  return w;
}

const GlmLayerResident& GlmDiagnosticModel::stack_layer(int layer) {
  const GlmLayerResident& r = loader_.load_layer(layer);
  // The last layer this model will ever read: the main stack's last, or
  // the draft layer when MTP is on (preconstruct_layers walks it after
  // the main stack).
  const int last_layer = mtp_ ? cfg_.mtp_layer() : cfg_.num_hidden_layers - 1;
  if (layer == last_layer &&
      loader_.residency() == GlmResidency::Resident &&
      !loader_.sources_released())
    loader_.release_sources();
  return r;
}

GlmLayerBound GlmDiagnosticModel::bind_layer(const GlmLayerResident& r,
                                              bool dense_mlp) {
  // The sharded path (d4): the resident layer is already this rank's
  // geometry, so binding is identity wiring. The views' full-load bind()
  // remains the parity reference the shard test pins bitwise.
  if (tp_) return tp_->bind_sharded(r, dense_mlp);
  // World=1: direct resident views, byte-identical to the M4 path.
  full_ = GlmLayerBound{};
  full_.mhc = &r.mhc;
  full_.ln1 = r.ln1;
  full_.ln2 = r.ln2;
  if (r.kind == GlmLayerKind::Kda) full_.kda = &r.kda;
  else full_.dsa = &r.dsa;
  if (dense_mlp) {
    full_.dense = r.dense;
  } else {
    moe_full_ = moe_weights(r.moe);  // partition defaults: every expert
    full_.moe = &moe_full_;
  }
  return full_;
}

// Constructs every layer object (KDA/DSA/MoE) with throwaway layer-0..k
// loads so NO allocation ever happens inside run_stack. Allocations are
// implicit device syncs, and a rank constructing lazily while a peer's
// first collective kernel spins on doorbells is the process-wide
// deadlock the M5 loopback bring-up measured: the lagging rank's
// cudaMallocManaged waits for the spinning kernel, which waits for the
// lagging rank's post, which requires the lagging rank to finish
// constructing. Post-startup (all objects resident, engines posting from
// free threads) the shape is safe — staggered per-layer loads and
// collectives coexist by construction.
void GlmDiagnosticModel::preconstruct_layers() {
  bool need_kda = kda_cfg_.num_kda_layers > 0;
  bool need_dsa = dsa_cfg_.num_dsa_layers > 0;
  bool need_moe =
      std::any_of(cfg_.mlps.begin(), cfg_.mlps.end(),
                  [](GlmMlpKind k) { return k == GlmMlpKind::Moe; });
  // RESIDENT: every layer materializes here, eagerly — a serving process
  // is ready when its constructor returns, not four minutes into its first
  // request (the lazy load put ~260 s inside the first prefill while every
  // peer's bulk collective sat waiting on the slowest disk, 2026-09-02).
  // Walking all layers also lets stack_layer release the checkpoint
  // sources before any forward. STREAMING keeps the historical shape:
  // throwaway loads until every layer kind has been seen.
  const bool eager_all = loader_.residency() == GlmResidency::Resident;
  // The bf16 decode weights' 12-bit companions: packed as each layer lands.
  const bool pack = eager_all && Bf12Companions::enabled();
  for (int layer = 0; layer < cfg_.num_hidden_layers &&
                       (eager_all || need_kda || need_dsa || need_moe);
       ++layer) {
    const GlmLayerResident& r = stack_layer(layer);
    const GlmLayerBound b =
        bind_layer(r, cfg_.mlps[layer] == GlmMlpKind::Dense);
    if (r.kind == GlmLayerKind::Kda && need_kda && !kda_) {
      kda_ = std::make_unique<KdaLayer>(arena_, gemm_, *b.kda, kda_cfg_,
                                        max_tokens_, gemm_ws_,
                                        gemm_ws_bytes_, cfg_.rms_norm_eps);
      need_kda = false;
    }
    if (r.kind == GlmLayerKind::Dsa && need_dsa && !dsa_) {
      dsa_ = std::make_unique<DsaLayer>(
          gemm_, *b.dsa, dsa_cfg_, max_tokens_, pool_.max_token_slots(),
          dsa_scratch_,
          DsaLayer::scratch_bytes(dsa_cfg_, max_tokens_,
                                  pool_.max_token_slots()),
          gemm_ws_, gemm_ws_bytes_);
      need_dsa = false;
    }
    if (need_moe && !moe_ && cfg_.mlps[layer] == GlmMlpKind::Moe) {
      // kDecodeRows: the decode fast path's physical row bound (scalar
      // session_step rows or the Phase-2 fixed batch). The graph
      // table slots (one per MoE layer) provision the capture path's
      // per-layer pinned expert-table sources unconditionally — a few
      // hundred KB of pinned memory; the eager path never touches them.
      moe_ = std::make_unique<GlmMoeLayer>(*b.moe, moe_cfg_, max_tokens_,
                                           kDecodeRows, moe_graph_slots());
      need_moe = false;
    }
    if (pack) pack_layer_companions(layer, r);
  }
  // The draft layer materializes last (stack_layer releases the sources
  // on it); it is DSA + MoE, so the layer objects above already exist.
  if (mtp_) {
    const GlmLayerResident& r = stack_layer(cfg_.mtp_layer());
    if (pack) pack_layer_companions(cfg_.mtp_layer(), r);
  }
  if (pack) finish_companions();
  loader_.release_layer();
}

void GlmDiagnosticModel::enqueue_dense_mlp(
    const uint16_t* x, uint16_t* out, const GlmQuantMatrix* dense,
    int tokens, cudaStream_t stream) {
  // The inter dim comes from the views: the full matrix at world=1, this
  // rank's column/row shard under TP (scratch is sized for the full I).
  const int H = cfg_.hidden_size;
  const int I = static_cast<int>(dense[0].rows);
  if (dense[1].rows != dense[0].rows || dense[2].cols != dense[0].rows)
    throw std::runtime_error("forward: inconsistent dense matrices");
  launch_scale_gemm_bf16(x, H, dense[0].payload, dense[0].scales, dense_g_,
                          tokens, I, H, stream, 0, dense_mma_from_rows_);
  launch_scale_gemm_bf16(x, H, dense[1].payload, dense[1].scales, dense_u_,
                          tokens, I, H, stream, 0, dense_mma_from_rows_);
  // Same asymmetric swiglu clamps as the experts (the reference MLP and
  // experts share the clamp choreography; DESIGN §7.4).
  launch_moe_swiglu_clamp(dense_g_, dense_u_, dense_act_,
                          static_cast<int64_t>(tokens) * I,
                          cfg_.swiglu_limit, stream);
  launch_scale_gemm_bf16(dense_act_, I, dense[2].payload, dense[2].scales,
                          out, tokens, H, I, stream, 0, dense_mma_from_rows_);
}

// Shared stack runner. `layer_inputs` (isolated mode) overrides the stream
// state entering EVERY layer (index L feeds layer L); `capture` (isolated
// mode) receives each layer's output streams plus the initial state at
// index 0; `boundary_capture` (isolated mode) additionally receives the two
// post-fold block-boundary outputs per layer (attn, FFN) — the raw surface
// where slicing errors surface unattenuated (the mHC stream update in
// `capture`'s snapshots compresses boundary errors below assertion budgets;
// the dense scale-grid slice bug hid exactly there). Free-run forward
// passes null for all three.
GlmDiagnosticModel::Outputs GlmDiagnosticModel::run_stack(
    const std::vector<int64_t>& token_ids, const uint16_t* const* layer_inputs,
    std::vector<std::vector<uint16_t>>* capture,
    std::vector<std::vector<uint16_t>>* boundary_capture) {
  // The re-forward and an open decode session SHARE the KDA/DSA state
  // pools — running one mid-session silently clobbers the other's state.
  // That failure mode is exactly the kind a gate would absorb as noise;
  // it throws loudly instead (a parity harness uses separate model
  // instances for engine and reference).
  for (int64_t pos : session_pos_) {
    if (pos > 0)
      throw std::runtime_error(
          "forward: a decode session is open on this model instance — a "
          "re-forward would clobber its state (use a second model for the "
          "reference)");
  }
  const int T = static_cast<int>(token_ids.size());
  if (T <= 0) throw std::invalid_argument("forward: empty token batch");
  if (T > max_tokens_)
    throw std::invalid_argument("forward: tokens exceed max_tokens");
  for (int64_t id : token_ids)
    if (id < 0 || id >= cfg_.vocab_size)
      throw std::invalid_argument("forward: token id out of range");
  const int H = cfg_.hidden_size;
  const float eps = cfg_.rms_norm_eps;

  // Shape-keyed plans (cache hits after the first forward of this size).
  if (!gemm_.ensure_plan(T, lm_vocab_count_, H, DType::BF16, GemmOut::F32,
                         H))
    throw std::runtime_error("forward: lm head GEMM plan unavailable");
  // A prefill-shaped walk (kernels/gemm.hpp): never the decode batch's
  // packed launches, whatever the last session walk left set.
  gemm_.set_bf12_wide(false);

  // Fresh per-request state.
  if (kda_rec_) {
    DGPP_CUDA_OK(cudaMemsetAsync(
        kda_rec_, 0,
        static_cast<size_t>(kda_cfg_.num_kda_layers) *
            kda_geo_.recurrent_bytes,
        stream_));
    DGPP_CUDA_OK(cudaMemsetAsync(
        kda_conv_, 0,
        static_cast<size_t>(kda_cfg_.num_kda_layers) *
            kda_geo_.conv_committed_bytes,
        stream_));
  }
  if (dsa_cfg_.num_dsa_layers > 0) {
    // Cold start every forward: fresh block tables and zeroed caches (the
    // diagnostic model owns no cross-call state).
    pool_.reset_all(stream_);
  }
  DGPP_CUDA_OK(cudaMemcpyAsync(d_tokens_, token_ids.data(),
                               static_cast<size_t>(T) * 8,
                               cudaMemcpyHostToDevice, stream_));
  glm_embed_bcast_streams(globals_.embed, d_tokens_, streams_[0], T, H,
                          stream_);
  if (layer_inputs) {
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
    upload_bf16(streams_[0], layer_inputs[0], static_cast<size_t>(T) * 4 * H,
                stream_);
    if (capture)
      capture->push_back(
          fetch_bf16(streams_[0], static_cast<size_t>(T) * 4 * H, stream_));
  }

  Outputs out;
  out.routes.reserve(static_cast<size_t>(cfg_.num_hidden_layers));
  uint16_t* cur = streams_[0];
  uint16_t* nxt = streams_[1];
  int dsa_ordinal = 0;
  int kda_ordinal = 0;

  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    if (layer_inputs && layer > 0) {
      // Isolated mode: every layer starts from the reference trajectory.
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      upload_bf16(cur, layer_inputs[layer], static_cast<size_t>(T) * 4 * H,
                  stream_);
    }
    const GlmLayerResident& r = stack_layer(layer);
    const GlmLayerBound b = bind_layer(r, cfg_.mlps[layer] == GlmMlpKind::Dense);

    // ---- attention site --------------------------------------------
    GlmMhcWeights hw;
    hw.fn = b.mhc->attn_fn;
    hw.base = b.mhc->attn_base;
    hw.scale = b.mhc->attn_scale;
    launch_mhc_compute(cur, hw, mhc_cfg_, collapsed_, post_, comb_,
                       mhc_logits_, T,
                        stream_);    glm_rmsnorm_bf16(collapsed_, b.ln1, normed_, T, H, eps, stream_);
    // Block boundary 1 (DESIGN §5.1): the attention output projection is
    // row-parallel over this rank's heads, so the block output is a partial
    // sum until folded. With pre-stage support the attention writes the
    // pinned staging buffer directly (the §6.3 interface — no staging copy;
    // decode T qualifies, prefill-sized T falls back to the device
    // buffer). The decision precedes the enqueue so the GEMM's destination
    // is the transport's send source.
    uint16_t* attn_out = sub_out_;
    if (boundary_) {
      if (uint16_t* staged = boundary_->stage(T, H)) attn_out = staged;
    }
    if (r.kind == GlmLayerKind::Kda) {
      if (!kda_) {
        kda_ = std::make_unique<KdaLayer>(arena_, gemm_, *b.kda, kda_cfg_,
                                          max_tokens_, gemm_ws_,
                                          gemm_ws_bytes_, eps);
      } else {
        kda_->rebind(*b.kda);
      }
      if (!kda_->prepare(T))
        throw std::runtime_error("forward: KDA GEMM plans unavailable");
      float* rec = kda_rec_ +
                   static_cast<size_t>(kda_ordinal) *
                       kda_geo_.recurrent_elems;
      // Conv state stride is the COMMITTED width (conv_hist): the
      // diagnostic forward reserves no MTP spec region, so the slot width
      // and the committed width coincide.
      uint16_t* conv =
          kda_conv_ + static_cast<size_t>(kda_ordinal) *
                          (kda_geo_.conv_committed_bytes / 2);
      kda_->enqueue(normed_, rec, conv, kda_geo_.conv_hist, attn_out, T,
                    stream_);
      ++kda_ordinal;
    } else {
      if (!dsa_) {
        dsa_ = std::make_unique<DsaLayer>(
            gemm_, *b.dsa, dsa_cfg_, max_tokens_,
            pool_.max_token_slots(), dsa_scratch_,
            DsaLayer::scratch_bytes(dsa_cfg_, max_tokens_,
                                    pool_.max_token_slots()),
            gemm_ws_, gemm_ws_bytes_);
      } else {
        dsa_->rebind(*b.dsa);
      }
      if (!dsa_->prepare(T))
        throw std::runtime_error("forward: DSA GEMM plans unavailable");
      dsa_->enqueue_prefill(normed_, pool_, dsa_ordinal, 0, 0, T, attn_out,
                             stream_);
      ++dsa_ordinal;
    }
    // The collective kernel runs on the bus's stream, so the producer
    // quiesces first (stream order cannot cover a cross-stream consumer;
    // the graph-mode decode path restores this as a stream-ordered node).
    if (boundary_) {
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      DGPP_LOG_DEBUG("TP boundary (attn) layer={} rows={} — reducing", layer,
                     T);
      boundary_->reduce(attn_out, T, H);
    }
    if (boundary_capture) {
      // Post-fold attention boundary (world=1: the unfolded full output —
      // the comparison target). Captured before the FFN site reuses the
      // device buffer.
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      boundary_capture->push_back(
          fetch_bf16(attn_out, static_cast<size_t>(T) * H, stream_));
    }
    launch_mhc_stream_update(post_, comb_, attn_out, cur, nxt, mhc_cfg_, T,
                             stream_);
    std::swap(cur, nxt);

    // ---- feed-forward site -----------------------------------------
    GlmMhcWeights fw;
    fw.fn = b.mhc->ffn_fn;
    fw.base = b.mhc->ffn_base;
    fw.scale = b.mhc->ffn_scale;
    launch_mhc_compute(cur, fw, mhc_cfg_, collapsed_, post_, comb_,
                       mhc_logits_, T,
                        stream_);
    glm_rmsnorm_bf16(collapsed_, b.ln2, normed_, T, H, eps, stream_);
    // Block boundary 2 destination: same staging-interface decision before the
    // FFN enqueue (dense down-proj, shared expert, and this rank's routed
    // experts are all partial until the fold).
    uint16_t* ffn_out = sub_out_;
    if (boundary_) {
      if (uint16_t* staged = boundary_->stage(T, H)) ffn_out = staged;
    }
    if (cfg_.mlps[layer] == GlmMlpKind::Dense) {
      enqueue_dense_mlp(normed_, ffn_out, b.dense, T, stream_);    } else {
      if (!moe_) {
        moe_ = std::make_unique<GlmMoeLayer>(*b.moe, moe_cfg_,
                                             max_tokens_);
      } else {
        moe_->rebind(*b.moe);
      }
      // The forward is prefill-class: the tensor-core expert kernel, the
      // same the session prefill runs (glm_tp_test pins the two bitwise);
      // only the decode-class paths keep the GEMV core.
      moe_->enqueue(normed_, ffn_out, T, stream_, MoeExpertKernel::kMma);
      GlmRouteTraceLayer route;
      route.layer_idx = static_cast<uint32_t>(layer);
      route.top_k = static_cast<uint32_t>(moe_cfg_.top_k);
      route.tokens = static_cast<uint64_t>(T);
      route.ids = moe_->last_ids();
      route.weights = moe_->last_weights();
      out.routes.push_back(std::move(route));
      out.route_biased.push_back(moe_->last_biased());
    }
    if (boundary_) {
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      DGPP_LOG_DEBUG("TP boundary (ffn)  layer={} rows={} — reducing", layer,
                     T);
      boundary_->reduce(ffn_out, T, H);
    }
    if (boundary_capture) {
      // Post-fold FFN boundary (world=1: the unfolded full output). The
      // raw fold surface — dense/shared-expert slicing errors appear here
      // at full magnitude.
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      boundary_capture->push_back(
          fetch_bf16(ffn_out, static_cast<size_t>(T) * H, stream_));
    }
    launch_mhc_stream_update(post_, comb_, ffn_out, cur, nxt, mhc_cfg_, T,
                             stream_);
    std::swap(cur, nxt);
    if (capture) {
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      capture->push_back(
          fetch_bf16(cur, static_cast<size_t>(T) * 4 * H, stream_));
    }
  }

  // ---- head: mean over streams, final norm, lm head -----------------
  launch_mhc_final_mean(cur, collapsed_, mhc_cfg_, T, stream_);
  glm_rmsnorm_bf16(collapsed_, globals_.final_norm, normed_, T, H, eps,
                   stream_);
  gemm_.matmul(normed_, globals_.lm_head, logits_, T, lm_vocab_count_, H,
               DType::BF16, GemmOut::F32, H, gemm_ws_, gemm_ws_bytes_,
               stream_);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));

    const size_t TH = static_cast<size_t>(T) * H;
  const size_t TV = static_cast<size_t>(T) * lm_vocab_count_;
  out.final_hidden_bits = fetch_bf16(normed_, TH, stream_);
  out.logits = fetch_f32(logits_, TV, stream_);
  out.lm_vocab_begin = lm_vocab_begin_;
  out.lm_vocab_count = lm_vocab_count_;
  return out;
}

GlmDiagnosticModel::Outputs GlmDiagnosticModel::forward(
    const std::vector<int64_t>& token_ids) {
  return run_stack(token_ids, nullptr, nullptr, nullptr);
}

GlmDiagnosticModel::Outputs GlmDiagnosticModel::forward_isolated(
    const std::vector<int64_t>& token_ids,
    const std::vector<const uint16_t*>& layer_inputs,
    std::vector<std::vector<uint16_t>>& capture,
    std::vector<std::vector<uint16_t>>* boundary_capture) {
  if (layer_inputs.size() != static_cast<size_t>(cfg_.num_hidden_layers) + 1)
    throw std::invalid_argument(
        "forward_isolated: needs num_layers+1 input snapshots");
  return run_stack(token_ids, layer_inputs.data(), &capture,
                   boundary_capture);
}

std::vector<std::vector<std::pair<int32_t, float>>> GlmDiagnosticModel::topk(
    const std::vector<float>& logits, int64_t rows, int vocab, int k) {
  if (k <= 0 || k > 64 || k > vocab)
    throw std::invalid_argument("topk: k out of range");
  std::vector<std::vector<std::pair<int32_t, float>>> out(
      static_cast<size_t>(rows));
  for (int64_t r = 0; r < rows; ++r) {
    const float* row = logits.data() + static_cast<size_t>(r) * vocab;
    std::vector<std::pair<float, int32_t>> best;
    best.reserve(static_cast<size_t>(k));
    for (int c = 0; c < vocab; ++c) {
      const float v = row[c];
      // Lowest-id tie-break: a later column only displaces on a strictly
      // greater value (insertion into the sorted-descending prefix).
      bool placed = false;
      for (size_t i = 0; i < best.size(); ++i) {
        if (v > best[i].first) {
          if (best.size() < static_cast<size_t>(k))
            best.emplace_back(0.f, 0);
          for (size_t j = best.size() - 1; j > i; --j) best[j] = best[j - 1];
          best[i] = {v, c};
          placed = true;
          break;
        }
      }
      if (!placed && best.size() < static_cast<size_t>(k))
        best.emplace_back(v, c);
    }
    auto& row_out = out[static_cast<size_t>(r)];
    row_out.reserve(best.size());
    for (const auto& [v, c] : best) row_out.emplace_back(c, v);
  }
  return out;
}

}  // namespace dgpp
