#include "models/qwen/forward.hpp"

#include "kernels/scale_gemm.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/qwen_mtp.hpp"
#include "kernels/qwen_norm.hpp"
#include "kernels/qwen_ple.hpp"
#include "kernels/qwen_vision.hpp"
#include "models/qwen/vision.hpp"

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

void d2d(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
  if (bytes == 0) return;
  DGPP_CUDA_OK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream));
}

}  // namespace

QwenModel::QwenModel(const QwenTextConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
                     int64_t max_cache_tokens, QwenResidency residency, BoundaryReducer* boundary,
                     int tp_rank, int tp_world, int max_requests, bool mtp, int decode_rows,
                     bool fp8_head_mma)
    : fp8_head_mma_(fp8_head_mma),
      cfg_(cfg),
      loader_(cfg, checkpoint_dir, tp_rank, tp_world, residency,
              tp_world > 1 ? QwenHeadSharding::VocabSharded : QwenHeadSharding::Full,
              mtp && residency == QwenResidency::Resident) {
  if (max_tokens <= 0) throw std::invalid_argument("QwenModel: max_tokens must be positive");
  if (mtp && cfg_.mtp_layer() < 0)
    throw std::invalid_argument("QwenModel: the config has no draft layer (mtp)");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("QwenModel: max_requests must be in [1, kPickMaxRequests]");
  if ((tp_world > 1) != (boundary != nullptr))
    throw std::invalid_argument(
        "QwenModel: a boundary reducer is required exactly when tp_world > 1 — without one the "
        "block-boundary partials would be returned silently as results");
  if (cfg_.eos_token_ids.empty()) throw std::invalid_argument("QwenModel: the config names no EOS token");
  init_stream();
  loader_.set_reader_stream(stream_);
  const int H = cfg_.hidden_size, W = cfg_.hc_count * H;
  globals_ = loader_.load_globals();
  // The generic session core (engine/session_model.hpp) over this family's
  // geometry: the pool's 64-token blocks, snapshots at pool boundaries,
  // the draft block's hyper-state window.
  {
    SessionParams sp;
    sp.max_tokens = max_tokens;
    sp.max_cache_tokens = ((std::max<int64_t>(max_cache_tokens, max_tokens) + kBlockTokens - 1) / kBlockTokens) * kBlockTokens;
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
    sp.max_position_embeddings = cfg_.context_limit();
    sp.block_tokens = kBlockTokens;
    sp.snapshot_align = cfg_.indexer_compress_ratio;
    sp.draft_width = W;
    sp.eos = static_cast<int32_t>(cfg_.eos_token_ids[0]);
    init_session(sp);
  }
  gemm_ws_bytes_ = std::max<size_t>(64u << 20, gemm_.query_workspace_bytes(max_tokens_, lm_vocab_count_, H, DType::BF16));
  gemm_ws_ = dev_alloc<char>(gemm_ws_bytes_);
  gw_ = QwenGemmWorkspace{&gemm_, gemm_ws_, gemm_ws_bytes_};
  if (QwenLayerStream::dense_weights_fp8()) {
    // The FP8 dense stack's prefill bridge: the largest dense matrix of
    // this rank's slice, in BF16.
    dense_bridge_bytes_ = dense_bridge_bytes(cfg_, loader_.geometry());
    gw_.dequant = dev_alloc<uint16_t>(dense_bridge_bytes_ / 2);
    gw_.dequant_bytes = dense_bridge_bytes_;
  }
  // The dense sites' lowering (kernels/gemm.hpp dense_gemv_rows): the GEMV
  // chunks (and the fused multi-problem launches) to the bound, cuBLASLt's
  // algorithm (bf16) or the streaming tensor-core GEMM (fp8) above it.
  gemm_.set_decode_rows(std::min(max_decode_rows_, dense_gemv_rows()));
  gw_.gemv_rows = dense_gemv_rows();
  gw_.mma_from_rows = dense_gemv_rows() + 1;
  has_ple_ = !cfg_.ple_layer_ids.empty();
  if (has_ple_) table_ = loader_.load_ngram_table();
  for (int l = 0; l < cfg_.num_hidden_layers; ++l)
    (cfg_.layers[static_cast<size_t>(l)] == QwenLayerKind::Gdn ? num_gdn_ : num_qsa_) += 1;
  n_moe_layers_ = cfg_.num_hidden_layers;  // every layer carries the MoE
  // The routed chain on this rank's slice of the intermediate dim.
  moe_cfg_ = QwenMoeLayer::routed_config(H, static_cast<int>(loader_.geometry().local_inter),
                                         cfg_.num_experts, cfg_.num_experts_per_tok, cfg_.norm_topk_prob);

  // Per-slot state from the local geometry (the layer objects agree — they
  // are built from the same numbers).
  const QwenLocalGeometry& geo = loader_.geometry();
  const size_t R = static_cast<size_t>(max_requests_);
  const size_t rows = static_cast<size_t>(max_decode_rows_);
  if (num_gdn_ > 0) {
    const int64_t lk = geo.local_key_heads, lv = geo.local_value_heads;
    const int64_t K = cfg_.gdn_key_head_dim, V = cfg_.gdn_value_head_dim;
    const int64_t C = 2 * lk * K + lv * V;
    gdn_rec_elems_ = lv * V * K;
    gdn_conv_elems_ = C * (cfg_.gdn_conv_width - 1);
    gdn_rec_ = dev_alloc<float>(R * num_gdn_ * static_cast<size_t>(gdn_rec_elems_));
    gdn_conv_ = dev_alloc<uint16_t>(R * num_gdn_ * static_cast<size_t>(gdn_conv_elems_));
    spec_rec_ = dev_alloc<float>(rows * num_gdn_ * static_cast<size_t>(gdn_rec_elems_));
    spec_conv_ = dev_alloc<uint16_t>(rows * num_gdn_ * static_cast<size_t>(gdn_conv_elems_));
  }
  if (has_ple_) {
    ple_conv_elems_ = static_cast<int64_t>(W) * (cfg_.ple_conv_kernel_size - 1) * cfg_.ngram_size;
    ple_conv_state_ = dev_alloc<uint16_t>(R * static_cast<size_t>(ple_conv_elems_));
    spec_ple_ = dev_alloc<uint16_t>(rows * static_cast<size_t>(ple_conv_elems_));
    if ((static_cast<size_t>(ple_conv_elems_) * 2) % 16 != 0)
      throw std::logic_error("QwenModel: the PLE conv state is not a 16-byte multiple");
  }
  d_ctx_ = dev_alloc<int32_t>(R * 4);
  h_ctx_ = pinned_alloc<int32_t>(R * 4);
  // The context kernel leaves a post-row snapshot for EVERY row of a walk
  // (a prefill chunk's rows included — the rollback reads decode rows
  // only, but the kernel writes them all): sized to the row bound.
  spec_ctx_ = dev_alloc<int32_t>(std::max(static_cast<size_t>(max_tokens_), rows) * 4);
  if (num_qsa_ > 0 || mtp_) {
    QwenKvPoolShape shape;
    shape.layers = pool_layers();
    shape.kv_heads = geo.local_kv_heads;
    shape.dim = cfg_.head_dim;
    shape.idx_dim = cfg_.indexer_head_dim;
    shape.kpool = cfg_.indexer_compress_ratio;
    shape.block_tokens = kBlockTokens;
    shape.max_requests = max_requests_;
    shape.token_slots = max_cache_tokens_;
    pool_.init(shape);
    spec_ring_ = dev_alloc<uint16_t>(static_cast<size_t>(num_qsa_) * rows * ring_elems());
  }

  const size_t M = static_cast<size_t>(max_tokens_);
  r_ = dev_alloc<uint16_t>(M * W);
  x_ = dev_alloc<uint16_t>(M * H);
  y_ = dev_alloc<uint16_t>(M * H);
  {
    const size_t layers = static_cast<size_t>(cfg_.num_hidden_layers) + (mtp_ ? 1 : 0);
    const size_t K = static_cast<size_t>(cfg_.num_experts_per_tok);
    h_route_ids_ = pinned_alloc<int32_t>(layers * M * K);
    h_route_weights_ = pinned_alloc<float>(layers * M * K);
    if (const char* v = std::getenv("DGPP_QWEN_MOE_PREFILL"); v && std::string(v) == "host")
      moe_prefill_host_path_ = true;
    // The boundary windows' budget: 20 MB on the Qwen walk (the fabric
    // sweep of 2026-09-09: 12 MB 25.1 ms per step, 20 MB 23.6, 24–32 MB
    // 23.0–23.1 — the GR site's 13 MB pair plus the next projection fit
    // beside the chain's traffic in the 24 MB L2); DGPP_L2_PREFETCH_MB
    // overrides through the prefetcher's own default.
    prefetch_window_bytes_ = std::getenv("DGPP_L2_PREFETCH_MB") ? 0 : (size_t{20} << 20);
  }
  // The mixer (the final read) is a global.
  mixer_ = std::make_unique<QwenGrSite>(globals_.mixer, gw_, cfg_.hc_count, H, cfg_.hc_lowrank, max_tokens_, cfg_.rms_norm_eps);
  for (int req = 0; req < max_requests_; ++req) push_context(req, eos_, eos_);
  if (mtp_) {
    if (!globals_.mtp_fc_embedding || !globals_.mtp_fc_hidden || !globals_.mtp_pre_fc_norm_embedding ||
        !globals_.mtp_pre_fc_norm_hidden || !globals_.mtp_mixer.hc_norm)
      throw std::runtime_error("QwenModel: the draft head's globals are unbound");
    mtp_ring_snapshot_ = dev_alloc<uint16_t>(R * ring_elems());
    mtp_chain_ring_ = dev_alloc<uint16_t>(R * ring_elems());
    mtp_hin_ = dev_alloc<uint16_t>(M * W);
    mtp_hn_ = dev_alloc<uint16_t>(M * W);
    mtp_enc_ = dev_alloc<uint16_t>(M * W);
    mtp_r_ = dev_alloc<uint16_t>(M * W);
    mtp_e_ = dev_alloc<uint16_t>(M * H);
    mtp_en_ = dev_alloc<uint16_t>(M * H);
    mtp_ein_ = dev_alloc<uint16_t>(M * H);
    mtp_mixer_ = std::make_unique<QwenGrSite>(globals_.mtp_mixer, gw_, cfg_.hc_count, H, cfg_.hc_lowrank,
                                              max_tokens_, cfg_.rms_norm_eps);
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  // The tower is replicated (it runs only on image prefills, never inside a
  // captured graph), so its weights are this rank's own copy and its digest
  // belongs to the boot identity like any other global.
  if (cfg_.vision) vision_ = std::make_unique<QwenVisionEncoder>(*cfg_.vision, checkpoint_dir, stream_);
}

size_t QwenModel::kv_block_bytes_static(const QwenTextConfig& cfg, int tp_rank, int tp_world, bool mtp) {
  const QwenHeadSharding head = tp_world > 1 ? QwenHeadSharding::VocabSharded : QwenHeadSharding::Full;
  const QwenLocalGeometry geo = QwenLocalGeometry::from_config(cfg, tp_rank, tp_world, head);
  int num_qsa = 0;
  for (QwenLayerKind k : cfg.layers) num_qsa += k == QwenLayerKind::Gdn ? 0 : 1;
  const int pool_layers = num_qsa + (mtp ? 1 : 0);
  if (pool_layers <= 0) return 0;
  const auto bytes = [&](int64_t tokens) {
    QwenKvPoolShape shape;
    shape.layers = pool_layers;
    shape.kv_heads = geo.local_kv_heads;
    shape.dim = cfg.head_dim;
    shape.idx_dim = cfg.indexer_head_dim;
    shape.kpool = cfg.indexer_compress_ratio;
    shape.block_tokens = kBlockTokens;
    shape.max_requests = 1;
    shape.token_slots = tokens;
    return QwenKvPool::cache_bytes(shape) - PagedBlockTable::table_bytes(1, tokens / kBlockTokens);
  };
  return bytes(2 * kBlockTokens) - bytes(kBlockTokens);
}

QwenModel::MemoryPlan QwenModel::plan_memory(const QwenTextConfig& cfg, int max_tokens,
                                             int64_t max_cache_tokens, int tp_rank, int tp_world,
                                             QwenResidency residency, int max_requests, bool mtp,
                                             int decode_rows) {
  if (max_tokens <= 0) throw std::invalid_argument("plan_memory: max_tokens must be positive");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("plan_memory: max_requests must be in [1, kPickMaxRequests]");
  if (decode_rows > kDecodeRowsMax) throw std::invalid_argument("plan_memory: decode_rows exceeds kDecodeRowsMax");
  if (mtp && cfg.mtp_layer() < 0) throw std::invalid_argument("plan_memory: the config has no draft layer");
  const QwenHeadSharding head = tp_world > 1 ? QwenHeadSharding::VocabSharded : QwenHeadSharding::Full;
  const QwenLocalGeometry geo = QwenLocalGeometry::from_config(cfg, tp_rank, tp_world, head);
  const int64_t cache_tokens =
      ((std::max<int64_t>(max_cache_tokens, max_tokens) + kBlockTokens - 1) / kBlockTokens) * kBlockTokens;
  MemoryPlan plan;
  plan.context_tokens = std::min<int64_t>(cache_tokens, cfg.context_limit());
  const size_t M = static_cast<size_t>(max_tokens);
  const size_t R = static_cast<size_t>(max_requests);
  // The fixed batch's row ceiling, floored as the session core floors it.
  const size_t rows = static_cast<size_t>(std::max({kDecodeRows, decode_rows, max_requests}));
  const size_t H = static_cast<size_t>(cfg.hidden_size), W = static_cast<size_t>(cfg.hc_count) * H;
  const size_t V = static_cast<size_t>(QwenLayerStream::lm_vocab_count(cfg, tp_rank, tp_world, head));
  int num_gdn = 0, num_qsa = 0;
  for (QwenLayerKind k : cfg.layers) (k == QwenLayerKind::Gdn ? num_gdn : num_qsa) += 1;
  const int pool_layers = num_qsa + (mtp ? 1 : 0);
  const bool has_ple = !cfg.ple_layer_ids.empty();

  // The weights: every layer resident (the n-gram table always is), or
  // one streamed layer beside the globals and the table.
  if (residency == QwenResidency::Resident) {
    plan.add(QwenLayerStream::ngram_table_mmap() ? "model weights (resident; the n-gram table mmap'ed from the checkpoint)"
                                                 : "model weights (resident, n-gram table included)",
             QwenLayerStream::resident_bytes(cfg, tp_rank, tp_world, head, mtp));
    plan.add("loader staging (pinned host, freed when the last layer is resident)", 0,
             QwenLayerStream::staging_plan_bytes(cfg, tp_rank, tp_world, head, mtp));
  } else {
    size_t largest = 0;
    const int layers_total = cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
    for (int l = 0; l < layers_total; ++l)
      largest = std::max(largest, QwenLayerStream::layer_bytes(cfg, l, tp_rank, tp_world));
    plan.add("model weights (one streamed layer + globals + n-gram table)",
             largest + QwenLayerStream::globals_bytes(cfg, tp_rank, tp_world, head) +
                 QwenLayerStream::ngram_table_bytes(cfg, tp_rank, tp_world));
  }
  if (Bf12Companions::enabled() && residency == QwenResidency::Resident) {
    const size_t packed = bf12_plan_bytes(cfg, geo, mtp);
    if (packed > 0) plan.add("bf16 decode packing (12-bit companions)", packed);
  }
  plan.add("gemm workspace (at least)", size_t{64} << 20);
  if (QwenLayerStream::dense_weights_fp8())
    plan.add("dense fp8 prefill bridge (the largest dense matrix in BF16)",
             dense_bridge_bytes(cfg, QwenLocalGeometry::from_config(cfg, tp_rank, tp_world, head)));
  if (has_ple && QwenLayerStream::ngram_table_mmap()) {
    // The table stays on the NVMe behind the page cache (nothing
    // reserved); the walk's ids and staged rows are pinned.
    const int hash_heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram / std::max(tp_world, 1);
    plan.add("n-gram table staging (pinned; the table mmap'ed from the checkpoint)", 0,
             QwenPleLayer::staging_bytes(cfg, hash_heads, max_tokens));
  }

  // Per-slot state and the spec snapshot rows.
  const size_t rec_elems = static_cast<size_t>(geo.local_value_heads) * cfg.gdn_value_head_dim * cfg.gdn_key_head_dim;
  const size_t conv_ch = 2 * static_cast<size_t>(geo.local_key_heads) * cfg.gdn_key_head_dim +
                         static_cast<size_t>(geo.local_value_heads) * cfg.gdn_value_head_dim;
  const size_t conv_elems = conv_ch * static_cast<size_t>(cfg.gdn_conv_width - 1);
  const size_t ple_elems = has_ple ? W * static_cast<size_t>((cfg.ple_conv_kernel_size - 1) * cfg.ngram_size) : 0;
  const size_t ring_elems = static_cast<size_t>(cfg.indexer_compress_ratio) * cfg.indexer_head_dim;
  const size_t family_bytes = static_cast<size_t>(num_gdn) * (rec_elems * 4 + conv_elems * 2) + ple_elems * 2 + 16;
  plan.add("request slot state (GDN recurrent/conv, PLE conv, context)", R * family_bytes);
  plan.add("spec snapshot rows", rows * family_bytes + static_cast<size_t>(num_qsa) * rows * ring_elems * 2);
  if (pool_layers > 0) {
    QwenKvPoolShape shape;
    shape.layers = pool_layers;
    shape.kv_heads = geo.local_kv_heads;
    shape.dim = cfg.head_dim;
    shape.idx_dim = cfg.indexer_head_dim;
    shape.kpool = cfg.indexer_compress_ratio;
    shape.block_tokens = kBlockTokens;
    shape.max_requests = max_requests;
    shape.token_slots = cache_tokens;
    plan.add("kv cache pool (K/V bf16, compressed index keys, rings)", QwenKvPool::cache_bytes(shape));
  }
  // Activations and the token rows.
  const size_t token_rows = std::max(M, rows + R * static_cast<size_t>(kSpecRows));
  plan.add("activations (hyper state, rows, head)",
           token_rows * 8 + M * 12 + 8 + M * W * 2 + 3 * M * H * 2 + M * V * 4 + rows * 64,
           rows * V * 4 + rows * H * 2 + token_rows * 8 + rows * 32 + R * 40);
  // The layer objects (built once, rebound per layer).
  const int64_t max_pools = cache_tokens / cfg.indexer_compress_ratio;
  size_t layers = 0;
  layers += 3 * QwenGrSite::scratch_bytes(cfg.hc_count, cfg.hidden_size, cfg.hc_lowrank, max_tokens);
  if (num_gdn > 0) layers += QwenGdnLayer::scratch_bytes(cfg, geo.local_key_heads, geo.local_value_heads, max_tokens);
  if (pool_layers > 0) layers += QwenQsaLayer::scratch_bytes(cfg, geo.local_heads, geo.local_kv_heads, max_tokens, max_pools);
  if (has_ple) layers += QwenPleLayer::scratch_bytes(cfg, geo.hash_heads, max_tokens);
  plan.add("layer objects (GR sites, GDN, QSA, PLE)", layers);
  const GlmMoeConfig moe_cfg = QwenMoeLayer::routed_config(cfg.hidden_size, static_cast<int>(geo.local_inter),
                                                           cfg.num_experts, cfg.num_experts_per_tok, cfg.norm_topk_prob);
  size_t moe_pinned = 0;
  const int table_slots = residency == QwenResidency::Resident ? cfg.num_hidden_layers + (mtp ? 1 : 0) : 0;
  const size_t moe_dev = QwenMoeLayer::scratch_bytes(moe_cfg, geo.local_shared_inter, max_tokens, &moe_pinned,
                                                     static_cast<int>(rows), table_slots);
  plan.add("moe scratch (routed slots, shared expert, graph tables)", moe_dev, moe_pinned);
  if (cfg.experts_nvfp4)
    plan.add("moe W4A4 activation workspace",
             GlmMoeLayer::w4a4_scratch_bytes(moe_cfg, max_tokens, true));
  if (mtp) {
    plan.add("draft block (hyper-state window, ring snapshot, fusion scratch, mixer)",
             R * rows * W * 2 + R * ring_elems * 2 + 4 * M * W * 2 + 3 * M * H * 2 + R * 24 +
                 QwenGrSite::scratch_bytes(cfg.hc_count, cfg.hidden_size, cfg.hc_lowrank, max_tokens),
             R * 8);
  }
  if (cfg.vision) {
    plan.add("vision tower weights (bf16, replicated)", cfg.vision->weight_bytes());
    plan.add("vision workspace (patches, attention tiles, staged image rows)",
             cfg.vision->workspace_bytes());
  }
  return plan;
}

QwenModel::~QwenModel() {
  cudaFree(gdn_rec_);
  cudaFree(gdn_conv_);
  cudaFree(ple_conv_state_);
  cudaFree(d_ctx_);
  cudaFreeHost(h_ctx_);
  cudaFree(spec_rec_);
  cudaFree(spec_conv_);
  cudaFree(spec_ple_);
  cudaFree(spec_ctx_);
  cudaFree(spec_ring_);
  cudaFree(r_);
  cudaFree(x_);
  cudaFree(y_);
  cudaFreeHost(h_route_ids_);
  cudaFreeHost(h_route_weights_);
  cudaFree(gemm_ws_);
  if (gw_.dequant) cudaFree(gw_.dequant);
  cudaFree(mtp_ring_snapshot_);
  cudaFree(mtp_chain_ring_);
  cudaFree(mtp_hin_);
  cudaFree(mtp_hn_);
  cudaFree(mtp_enc_);
  cudaFree(mtp_r_);
  cudaFree(mtp_e_);
  cudaFree(mtp_en_);
  cudaFree(mtp_ein_);
}

QwenMoeWeights QwenModel::moe_view(const QwenMoeResident& m) {
  QwenMoeWeights w;
  w.router = m.router;
  w.shared_gate = m.shared_gate;
  w.shared_gate_proj = m.shared[0];
  w.shared_up_proj = m.shared[1];
  w.shared_down_proj = m.shared[2];
  w.shared_fp8 = m.shared_fp8[0].payload ? m.shared_fp8 : nullptr;
  w.shared_inter = m.local_shared_inter;
  w.experts = m.experts.empty() ? nullptr : m.experts.data();
  w.experts_fp4 = m.experts_fp4.empty() ? nullptr : m.experts_fp4.data();
  w.act_scale_w13 = m.act_scale_w13;
  w.act_scale_w2 = m.act_scale_w2;
  w.act_scales_dev = m.act_scales;
  return w;
}

// The largest dense matrix of the rank's slice in BF16 (the FP8 prefill bridge).
size_t QwenModel::dense_bridge_bytes(const QwenTextConfig& cfg, const QwenLocalGeometry& geo) {
  const size_t H = static_cast<size_t>(cfg.hidden_size), W = static_cast<size_t>(cfg.hyper_width());
  const size_t r = static_cast<size_t>(cfg.hc_lowrank);
  size_t elems = 0;
  const auto take = [&](size_t n, size_t k) { elems = std::max(elems, n * k); };
  take(static_cast<size_t>(geo.local_heads) * 2 * cfg.head_dim, H);                      // q_proj
  take(static_cast<size_t>(geo.local_kv_heads) * cfg.head_dim, H);                       // k / v
  take(H, static_cast<size_t>(geo.local_heads) * cfg.head_dim);                          // o_proj
  take(static_cast<size_t>((cfg.indexer_n_heads + 1) * cfg.indexer_head_dim), H);        // indexer
  take(static_cast<size_t>(2 * geo.local_key_heads * cfg.gdn_key_head_dim + geo.local_value_heads * cfg.gdn_value_head_dim), H);
  take(static_cast<size_t>(geo.local_value_heads) * cfg.gdn_value_head_dim, H);          // z
  take(H, static_cast<size_t>(geo.local_value_heads) * cfg.gdn_value_head_dim);          // out_proj
  take(r, W);                                                                            // GR down
  take(W, r);                                                                            // GR up
  take(W, static_cast<size_t>(cfg.ple_embed_dim / std::max(geo.world, 1)));              // PLE key
  take(H, static_cast<size_t>(cfg.ple_embed_dim / std::max(geo.world, 1)));              // PLE value
  return elems * 2;
}

// The lm_head product into logits_ (f32): the checkpoint's BF16 through the
// GEMM interface, or the block-FP8 form (engine.dense_weights) through the scale
// GEMM.
void QwenModel::lm_head_logits(const uint16_t* hidden, int rows, cudaStream_t stream, bool last_row_only) {
  const int H = cfg_.hidden_size;
  // Opt in to weight-tile reuse only within the configured decode envelope;
  // this bounds the optimization to verification shapes covered by its gates.
  // Short prefill chunks/tails in that envelope also use MMA, so their logits
  // depend on decode capacity as well as the head setting. Bitwise eager/graph
  // gates must construct both models with the same capacity and head setting.
  // Streaming MMA preserves weight values but reassociates FP32 sums.
  if (globals_.lm_head_fp8.payload)
    launch_scale_gemm_f32(hidden, static_cast<size_t>(H), globals_.lm_head_fp8.payload,
                          globals_.lm_head_fp8.scales, logits_, rows, lm_vocab_count_, H, stream,
                          static_cast<size_t>(lm_vocab_count_),
                          fp8_head_mma_ && rows <= max_decode_rows_ ? dense_gemv_rows() + 1 : 0,
                          last_row_only);
  else
    gemm_.matmul(hidden, globals_.lm_head, logits_, rows, lm_vocab_count_, H, DType::BF16, GemmOut::F32,
                 static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream);
}

float* QwenModel::gdn_rec(int req, int ordinal) const {
  return gdn_rec_ + (static_cast<size_t>(req) * num_gdn_ + ordinal) * static_cast<size_t>(gdn_rec_elems_);
}

uint16_t* QwenModel::gdn_conv(int req, int ordinal) const {
  return gdn_conv_ + (static_cast<size_t>(req) * num_gdn_ + ordinal) * static_cast<size_t>(gdn_conv_elems_);
}

uint16_t* QwenModel::ple_conv(int req) const {
  return ple_conv_state_ + static_cast<size_t>(req) * static_cast<size_t>(ple_conv_elems_);
}

size_t QwenModel::ring_elems() const {
  return static_cast<size_t>(cfg_.indexer_compress_ratio) * cfg_.indexer_head_dim;
}

void QwenModel::push_context(int req, int32_t prev1, int32_t prev2) {
  int32_t* h = h_ctx_ + static_cast<size_t>(req) * 4;
  h[0] = prev1;
  h[1] = prev2;
  h[2] = 0;
  h[3] = 0;
  DGPP_CUDA_OK(cudaMemcpyAsync(ctx(req), h, 4 * sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
}

void QwenModel::build_layer_objects(const QwenLayerResident& r) {
  const int H = cfg_.hidden_size;
  if (!attn_gr_) {
    attn_gr_ = std::make_unique<QwenGrSite>(r.attn_gr, gw_, cfg_.hc_count, H, cfg_.hc_lowrank, max_tokens_, cfg_.rms_norm_eps);
    mlp_gr_ = std::make_unique<QwenGrSite>(r.mlp_gr, gw_, cfg_.hc_count, H, cfg_.hc_lowrank, max_tokens_, cfg_.rms_norm_eps);
  } else {
    attn_gr_->rebind(r.attn_gr);
    mlp_gr_->rebind(r.mlp_gr);
  }
  if (r.kind == QwenLayerKind::Gdn) {
    if (!gdn_) {
      gdn_ = std::make_unique<QwenGdnLayer>(r.gdn, gw_, cfg_, max_tokens_);
      if (gdn_->recurrent_elems() != gdn_rec_elems_ || gdn_->conv_state_elems() != gdn_conv_elems_)
        throw std::logic_error("QwenModel: the GDN layer's state geometry disagrees with the slots'");
    } else {
      gdn_->rebind(r.gdn);
    }
  } else {
    if (!qsa_) {
      qsa_ = std::make_unique<QwenQsaLayer>(r.qsa, gw_, cfg_, max_tokens_, pool_.pool_slots());
    } else {
      qsa_->rebind(r.qsa);
    }
  }
  if (!moe_) {
    // The decode fast path's provisioning: the decode-row ceiling's slot rows and one
    // graph table slot per MoE layer (resident stacks bake them in).
    const int table_slots =
        loader_.residency() == QwenResidency::Resident ? n_moe_layers_ + (mtp_ ? 1 : 0) : 0;
    moe_ = std::make_unique<QwenMoeLayer>(moe_view(r.moe), moe_cfg_, gemm_, max_tokens_, max_decode_rows_,
                                          table_slots);
    moe_->set_mma_from_rows(gw_.mma_from_rows);
  } else {
    moe_->rebind(moe_view(r.moe));
  }
  if (r.has_ple) {
    if (!ple_) {
      ple_ = std::make_unique<QwenPleLayer>(r.ple, table_, gw_, cfg_, max_tokens_);
      if (ple_->conv_state_elems() != ple_conv_elems_)
        throw std::logic_error("QwenModel: the PLE layer's state geometry disagrees with the slots'");
    } else {
      ple_->rebind(r.ple, table_);
    }
  }
}

// The state every row walk starts from on a fresh request (the session
// core's open_slot / close): zero GDN recurrent/conv states, zero PLE conv
// state, EOS n-gram context, zero rings, no blocks.
void QwenModel::reset_slot_state(int req) {
  if (num_gdn_ > 0) {
    DGPP_CUDA_OK(cudaMemsetAsync(gdn_rec(req, 0), 0, static_cast<size_t>(num_gdn_) * gdn_rec_elems_ * 4, stream_));
    DGPP_CUDA_OK(cudaMemsetAsync(gdn_conv(req, 0), 0, static_cast<size_t>(num_gdn_) * gdn_conv_elems_ * 2, stream_));
  }
  if (has_ple_)
    DGPP_CUDA_OK(cudaMemsetAsync(ple_conv(req), 0, static_cast<size_t>(ple_conv_elems_) * 2, stream_));
  if (num_qsa_ > 0) pool_.reset_request(req, stream_);
  push_context(req, eos_, eos_);
}

// ---------------------------------------------------------------------------
// The row walk.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// The boundary prefetch windows. Each fold is a bus
// all-reduce the chain waits on, followed by the GR combine and the next
// site's norm before the next weight-streaming GEMV — latency the memory
// system would idle through. A window opened just before the fold runs
// through it on the prefetcher's side stream (a graph branch under
// capture), so the other side's first ~12 MB of weights are in L2 when
// their GEMVs arrive. Bit-identical on or off: nothing is written.
// ---------------------------------------------------------------------------
namespace {
// DGPP_QWEN_PREFETCH_DEBUG=1: every add's pointer and bytes on stderr.
bool prefetch_debug() {
  static const bool on = [] {
    const char* v = std::getenv("DGPP_QWEN_PREFETCH_DEBUG");
    return v && *v && std::string(v) != "0";
  }();
  return on;
}
}  // namespace

void QwenModel::prefetch_add(const char* what, const void* p, size_t bytes) {
  if (prefetch_debug()) std::fprintf(stderr, "prefetch add %-18s %p %zu\n", what, p, bytes);
  prefetch_.add(p, bytes);
}

void QwenModel::prefetch_bf16(const char* what, const uint16_t* w, size_t bytes) {
  const void* view = nullptr;
  size_t view_bytes = 0;
  gemm_.resident_view(w, bytes, walk_rows_, &view, &view_bytes);
  if (prefetch_debug()) std::fprintf(stderr, "prefetch add %-18s %p %zu\n", what, view, view_bytes);
  prefetch_.add_view(w, view, view_bytes);  // a companion is its own allocation
}

// The FP8 form's payload and scale grid (adjacent grants of one image).
void QwenModel::prefetch_fp8(const char* what, const GlmQuantMatrix& q) {
  if (!q.payload) return;
  prefetch_add(what, q.payload, static_cast<size_t>(q.rows) * static_cast<size_t>(q.cols));
  prefetch_add(what, q.scales, static_cast<size_t>(q.scale_rows()) * static_cast<size_t>(q.scale_cols()) * 4);
}

void QwenModel::prefetch_gr(const QwenGrResident& g, bool inject) {
  const size_t W = static_cast<size_t>(cfg_.hyper_width());
  const size_t r = static_cast<size_t>(cfg_.hc_lowrank);
  const size_t hc = static_cast<size_t>(cfg_.hc_count);
  if (inject && g.inject) prefetch_add("g.inject", g.inject, hc * W * 2);
  if (g.hc_norm) prefetch_add("g.hc_norm", g.hc_norm, W * 2);
  if (g.down) prefetch_add("g.down", g.down, r * W * 2);
  else prefetch_fp8("g.down_fp8", g.down_fp8);
  if (g.up) prefetch_add("g.up", g.up, W * r * 2);
  else prefetch_fp8("g.up_fp8", g.up_fp8);
}

// Before the attention fold: the MLP-side GR mix WITH its inject (since
// 2026-09-10 the inject dots ride the mix's down GEMV, so the 82 KB is
// read at the mix, not at the combine), the router and the shared expert
// (the routed experts wait for the router) — one image.
void QwenModel::prefetch_ffn_side(const QwenLayerResident& r) {
  if (!prefetch_.enabled()) return;
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, prefetch_window_bytes_, prefetch_.boundary_rate());
  prefetch_gr(r.mlp_gr, /*inject=*/true);
  if (r.moe.router) prefetch_add("r.moe.router", r.moe.router, static_cast<size_t>(cfg_.num_experts) * H * 2);
  if (r.moe.shared_gate) prefetch_add("r.moe.shared_gate", r.moe.shared_gate, H * 2);
  const size_t S = static_cast<size_t>(cfg_.shared_expert_intermediate_size / world_);
  for (int i = 0; i < 3; ++i) {
    if (r.moe.shared[i]) prefetch_add("r.moe.shared[i]", r.moe.shared[i], S * H * 2);
    else prefetch_fp8("r.moe.shared_fp8[i]", r.moe.shared_fp8[i]);
  }
}

// Before the MoE fold: the next layer's attention-side GR mix and its
// first projection (or the head).
void QwenModel::prefetch_attention_side(int layer) {
  if (!prefetch_.enabled()) return;
  if (layer >= cfg_.num_hidden_layers) {
    prefetch_head(globals_.mixer);
    return;
  }
  // Resident stacks only: load_layer is a lookup there; a streaming stack
  // loads on demand, one layer at a time.
  if (loader_.residency() != QwenResidency::Resident) return;
  const QwenLayerResident& r = loader_.load_layer(layer);
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, prefetch_window_bytes_, prefetch_.boundary_rate());
  prefetch_gr(r.attn_gr, /*inject=*/true);
  if (r.kind == QwenLayerKind::Gdn && r.gdn.in_proj_qkv) {
    const size_t rows = static_cast<size_t>(r.gdn.local_key_heads) * cfg_.gdn_key_head_dim * 2 +
                        static_cast<size_t>(r.gdn.local_value_heads) * cfg_.gdn_value_head_dim;
    prefetch_bf16("r.gdn.in_proj_qkv", r.gdn.in_proj_qkv, rows * H * 2);
  } else if (r.kind == QwenLayerKind::Gdn && r.gdn.in_proj_qkv_fp8.payload) {
    prefetch_fp8("r.gdn.in_proj_qkv_fp8", r.gdn.in_proj_qkv_fp8);
  } else if (r.qsa.q_proj) {
    const size_t rows = static_cast<size_t>(r.qsa.local_heads) * 2 * cfg_.head_dim;
    prefetch_bf16("r.qsa.q_proj", r.qsa.q_proj, rows * H * 2);
  } else if (r.qsa.q_proj_fp8.payload) {
    prefetch_fp8("r.qsa.q_proj_fp8", r.qsa.q_proj_fp8);
  }
}

// A window's adds must come from one allocation: the prefetcher bridges
// gaps up to 2 MB between adjacent adds, and two images (a layer's and
// its neighbour's, or a layer's and the globals') can sit closer than
// that — the fixture's do; a bridged hole is an out-of-bounds read
// (compute-sanitizer, 2026-09-09). So the current layer's 82 KB inject
// is never added beside the next image's weights.
void QwenModel::prefetch_head(const QwenGrResident& mixer) {
  if (!prefetch_.enabled()) return;
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, prefetch_window_bytes_, prefetch_.boundary_rate());
  prefetch_gr(mixer, /*inject=*/false);
  // Far larger than the window: add() clamps, the GEMV's leading rows hit.
  if (globals_.lm_head) prefetch_bf16("globals_.lm_head", globals_.lm_head, static_cast<size_t>(lm_vocab_count_) * H * 2);
  else prefetch_fp8("globals_.lm_head_fp8", globals_.lm_head_fp8);
}

// The PLE layer's two folds (its K-sliced key and value partials).
void QwenModel::prefetch_ple_key_side(const QwenLayerResident& r) {
  if (!prefetch_.enabled()) return;
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  const size_t W = static_cast<size_t>(cfg_.hyper_width());
  const size_t cols = static_cast<size_t>(cfg_.ple_embed_dim / world_);
  prefetch_.open_window(stream_, prefetch_window_bytes_, prefetch_.boundary_rate());
  if (r.ple.norm_key) prefetch_add("r.ple.norm_key", r.ple.norm_key, W * 2);
  if (r.ple.value_proj) prefetch_add("r.ple.value_proj", r.ple.value_proj, H * cols * 2);
  else prefetch_fp8("r.ple.value_proj_fp8", r.ple.value_proj_fp8);
}

void QwenModel::prefetch_ple_value_side(const QwenLayerResident& r) {
  if (!prefetch_.enabled()) return;
  const size_t W = static_cast<size_t>(cfg_.hyper_width());
  prefetch_.open_window(stream_, prefetch_window_bytes_, prefetch_.boundary_rate());
  if (r.ple.norm_conv) prefetch_add("r.ple.norm_conv", r.ple.norm_conv, W * 2);
  if (r.ple.conv) prefetch_add("r.ple.conv", r.ple.conv, W * static_cast<size_t>(cfg_.ple_conv_kernel_size) * 2);
  prefetch_gr(r.attn_gr, /*inject=*/true);
}

QwenModel::Outputs QwenModel::run_rows(const RowRun& run) {
  const int T = run.T, req = run.req;
  walk_rows_ = T;
  qwen_configure_gemm_rows(gemm_, T, run.decode);
  if (run.capture && loader_.residency() != QwenResidency::Resident)
    throw std::logic_error("run_rows: a capture needs a resident stack");
  const int H = cfg_.hidden_size, W = cfg_.hc_count * H;
  const RowInputs in = begin_run(run);
  const bool batched = in.batched;
  const int num_requests = in.num_requests;
  const int64_t* tokens = in.tokens;
  const int64_t* d_pos = in.pos;
  const int32_t* d_req = in.req_ids;
  const int32_t* d_spans = in.spans;
  // The mmap'ed n-gram table: the walk's hash ids and the
  // host's gather forked off here, joined at the PLE layer's turn.
  if (has_ple_ && table_.mmap) {
    if (!ple_) build_layer_objects(loader_.load_layer(cfg_.ple_layer()));
    ple_->stage(tokens, T, d_req, d_pos, d_spans, num_requests, d_ctx_, stream_);
  }
  glm_embed_bcast_streams(globals_.embed, tokens, r_, T, H, stream_);
  // Image rows replace the embedding at their prompt positions; the tower
  // already projected them (see apply_image_embeddings). Only a single
  // request's prefill walk carries images.
  if (prefill_images_) {
    if (run.decode || run.batch_requests || run.num_spans)
      throw std::logic_error("Qwen: image prefill reached a non-prefill walk");
    apply_image_embeddings(r_, run.pos0, T, 0, cfg_.hc_count);
  }

  // The state families' per-row snapshots (the rollback's source).
  const bool snapshots = run.decode && run.snapshots;
  KdaRequestRows rows;
  rows.request_ids = d_req;
  rows.positions = d_pos;
  rows.spans = d_spans;
  rows.num_requests = num_requests;
  // The boundary folds (plan D1): the producer writes its partial into
  // the reducer's staged buffer when the shape fits (the collective sends
  // straight from there; under capture the recorder's one stable buffer,
  // consumed before the next handout), else into `fallback`. The eager
  // producer quiesces before the collective; under capture the fold is a
  // recorded node and the stream order IS the drain.
  const auto stage = [&](uint16_t* fallback, int width) -> uint16_t* {
    if (!boundary_) return fallback;
    uint16_t* s = boundary_->stage(T, width);
    if (s == nullptr && run.capture)
      throw std::runtime_error("run_rows: a capture fold does not fit the recorder's staged buffer");
    return s ? s : fallback;
  };
  const auto fold = [&](uint16_t* buf, int width) {
    if (!boundary_) return;
    if (!run.capture) DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
    boundary_->reduce(buf, T, width);
  };

  Outputs out;
  const bool traces = !run.decode && route_traces_;
  int gdn_ordinal = 0, qsa_ordinal = 0;
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const QwenLayerResident& r = loader_.load_layer(layer);
    build_layer_objects(r);
    if (r.has_ple) {
      ple_->embed(tokens, T, d_req, d_pos, d_spans, num_requests, d_ctx_, stream_);
      // The context AFTER every row into the spec rows, the last real
      // row's in place (the hash above read the incoming context).
      qwen_ple_context_rows(tokens, T, d_req, d_pos, d_spans, num_requests, d_ctx_, spec_ctx_, stream_);
      // The K-sliced projections are partial sums (plan D4): each folded
      // before its norm, the key consumed before the value's handout.
      uint16_t* key = stage(ple_->key_partial(), W);
      ple_->project_key(key == ple_->key_partial() ? nullptr : key, T, stream_);
      if (run.decode) prefetch_ple_key_side(r);
      fold(key, W);
      ple_->norm_key(key, T, stream_);
      uint16_t* value = stage(ple_->value_partial(), H);
      ple_->project_value(value == ple_->value_partial() ? nullptr : value, T, stream_);
      if (run.decode) prefetch_ple_value_side(r);
      fold(value, H);
      ple_->finish(r_, value, ple_conv_state_, ple_conv_elems_, d_req, d_pos, d_spans, num_requests, T,
                   stream_, snapshots ? spec_ple_ : nullptr);
    }
    attn_gr_->mix(r_, x_, T, stream_);
    uint16_t* attn_out = stage(y_, H);
    if (r.kind == QwenLayerKind::Gdn) {
      KdaStateSnapshots rec_snap;
      KdaConvSnapshots conv_snap;
      if (snapshots) {
        rec_snap.states = spec_rec_ + static_cast<size_t>(gdn_ordinal) * gdn_rec_elems_;
        rec_snap.stride_elems = static_cast<int64_t>(num_gdn_) * gdn_rec_elems_;
        conv_snap.states = spec_conv_ + static_cast<size_t>(gdn_ordinal) * gdn_conv_elems_;
        conv_snap.stride_elems = static_cast<int64_t>(num_gdn_) * gdn_conv_elems_;
      }
      if (batched) {
        gdn_->enqueue_rows(x_, gdn_rec(0, gdn_ordinal), static_cast<int64_t>(num_gdn_) * gdn_rec_elems_,
                           gdn_conv(0, gdn_ordinal), static_cast<int64_t>(num_gdn_) * gdn_conv_elems_,
                           attn_out, T, rows, stream_, rec_snap, conv_snap);
      } else if (run.num_spans > 0) {
        // A group prefill (2026-09-14): each span's rows scan its own
        // request's recurrent and conv state (the pointers step by rows).
        int64_t row0 = 0;
        for (int sp = 0; sp < run.num_spans; ++sp) {
          const int len = run.span_lens[sp], sreq = run.span_reqs[sp];
          gdn_->enqueue(x_ + static_cast<size_t>(row0) * H, gdn_rec(sreq, gdn_ordinal), gdn_conv(sreq, gdn_ordinal),
                        attn_out + static_cast<size_t>(row0) * H, len, stream_);
          row0 += len;
        }
      } else {
        gdn_->enqueue(x_, gdn_rec(req, gdn_ordinal), gdn_conv(req, gdn_ordinal), attn_out, T, stream_,
                      rec_snap, conv_snap);
      }
      ++gdn_ordinal;
    } else {
      QwenQsaCache cache = pool_.view(qsa_ordinal);
      QwenQsaRows qrows;
      qrows.req_ids = d_req;
      qrows.pos = d_pos;
      qrows.decode = run.decode;
      qrows.request = req;
      qrows.pos0 = run.pos0;
      qrows.spans = d_spans;
      qrows.num_requests = num_requests;
      if (snapshots)
        qrows.ring_snapshots = spec_ring_ + static_cast<size_t>(qsa_ordinal) * static_cast<size_t>(max_decode_rows_) * ring_elems();
      if (!run.decode && run.num_spans > 0) {
        // A group prefill: each span attends over its own request's cache
        // and writes its pools and ring (the row metadata steps with the rows).
        int64_t row0 = 0;
        for (int sp = 0; sp < run.num_spans; ++sp) {
          const int len = run.span_lens[sp];
          QwenQsaRows srows = qrows;
          srows.req_ids = d_req + row0;
          srows.pos = d_pos + row0;
          srows.request = run.span_reqs[sp];
          srows.pos0 = run.span_pos0[sp];
          qsa_->enqueue(x_ + static_cast<size_t>(row0) * H, len, srows, cache, attn_out + static_cast<size_t>(row0) * H, stream_);
          row0 += len;
        }
      } else {
        qsa_->enqueue(x_, T, qrows, cache, attn_out, stream_);
      }
      ++qsa_ordinal;
    }
    if (run.decode) prefetch_ffn_side(r);
    fold(attn_out, H);  // block boundary 1: the attention output projection's partial
    attn_gr_->combine(r_, attn_out, T, stream_);
    mlp_gr_->mix(r_, x_, T, stream_);
    uint16_t* ffn_out = stage(y_, H);
    if (run.decode) {
      moe_->enqueue_decode(x_, ffn_out, T, stream_, run.capture ? layer : -1);
    } else if (moe_prefill_host_path_) {
      moe_->enqueue(x_, ffn_out, T, stream_);  // the host-orchestrated reference chain
      if (traces) {
        out.route_ids.push_back(moe_->routed().last_ids());
        out.route_weights.push_back(moe_->routed().last_weights());
      }
    } else {
      // The prefill's device-segmented tensor-core chain (Q7): no host
      // sync per layer; the routing rides async copies into this layer's
      // pinned slot and is materialized after the final sync.
      MoeTraceStaging trace;
      const size_t K = static_cast<size_t>(cfg_.num_experts_per_tok);
      const size_t slot = static_cast<size_t>(layer) * static_cast<size_t>(max_tokens_) * K;
      trace.ids = h_route_ids_ + slot;
      trace.weights = h_route_weights_ + slot;
      trace.biased = nullptr;
      moe_->enqueue_prefill(x_, ffn_out, T, stream_, traces ? &trace : nullptr);
      if (traces) {
        out.route_ids.emplace_back();  // filled after the sync
        out.route_weights.emplace_back();
      }
    }
    if (run.decode) prefetch_attention_side(layer + 1);
    fold(ffn_out, H);  // block boundary 2: the experts' sliced down projections
    mlp_gr_->combine(r_, ffn_out, T, stream_);
    if (run.capture_layers) {
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      std::vector<uint16_t> snap(static_cast<size_t>(T) * W);
      DGPP_CUDA_OK(cudaMemcpy(snap.data(), r_, snap.size() * 2, cudaMemcpyDeviceToHost));
      out.layer_states.push_back(std::move(snap));
    }
  }
  // A plain prefill chunk reads back only its last row's logits for the
  // scheduler's next-token pick, so only that row takes the head.
  // Keep the original T-row dispatch: GEMV and the large-m tensor-core
  // kernel accumulate differently. The FP8 launcher computes only row T-1
  // through the same kernel family the full head would use. Streaming-MMA
  // chunks, BF16 heads (cuBLAS is m-dependent), decode, all-row runs and
  // group prefills keep the full head. DGPP_PREFILL_HEAD_ALL_ROWS=1 keeps
  // it everywhere.
  mixer_->mix(r_, h_, T, stream_);
  static const bool head_all_rows = [] {
    const char* e = std::getenv("DGPP_PREFILL_HEAD_ALL_ROWS");
    return e != nullptr && e[0] == '1';
  }();
  const bool mma_envelope = fp8_head_mma_ && T <= max_decode_rows_;
  const bool last_row_only = !run.decode && !run.all_rows && run.num_spans == 0 && T > 1 &&
                             globals_.lm_head_fp8.payload != nullptr && !mma_envelope && !head_all_rows;
  lm_head_logits(h_, T, stream_, last_row_only);
  // The draft block's input: the last rows' hyper states into the slots'
  // windows by position (the last window rows of a prefill chunk, every
  // decode row — distinct slots within one launch).
  if (mtp_) {
    const int n = run.num_spans > 0 ? T : std::min(T, max_decode_rows_);  // a group prefill stores every row
    store_draft_hidden(r_ + static_cast<size_t>(T - n) * W, d_req + (T - n), d_pos + (T - n), n);
  }
  // The prefetch side stream rejoins here: a capture must end with every
  // forked stream joined, and the eager tail's sync (finish_run) covers
  // the prefetches too (they read weights, nothing else).
  if (run.decode) prefetch_.join(stream_);
  // The tail mirrors, the sync and the host copies are the core's; the
  // route traces of the device-segmented prefill materialize after it.
  out = finish_run(run, std::move(out));
  // An eager walk has synced: a staging that failed on the host surfaces
  // here rather than as a silent embedding.
  if (has_ple_ && table_.mmap && !run.capture) ple_->check_staged();
  if (!run.capture && traces && !run.decode && !moe_prefill_host_path_) {
    const size_t K = static_cast<size_t>(cfg_.num_experts_per_tok);
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
QwenModel::Outputs QwenModel::forward(const std::vector<int64_t>& token_ids, bool capture_layers) {
  const int T = static_cast<int>(token_ids.size());
  if (T <= 0) throw std::invalid_argument("forward: empty token batch");
  if (T > max_tokens_) throw std::invalid_argument("forward: tokens exceed max_tokens");
  if (T > max_context_) throw std::invalid_argument("forward: tokens exceed the context bound");
  for (int64_t id : token_ids)
    if (id < 0 || id >= cfg_.vocab_size) throw std::invalid_argument("forward: token id out of range");
  if (session_pos_[0] != 0)
    throw std::logic_error("forward: slot 0 holds an open session (close it first)");
  open_slot(0);
  if (num_qsa_ > 0 && !pool_.ensure_request_blocks(0, T, stream_))
    throw std::runtime_error("forward: the cache pool cannot cover the batch");
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
// Sessions: the prefill.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Sessions: the decode rows.
// ---------------------------------------------------------------------------

GlmSpecSegments QwenModel::spec_segments(int req, int snapshot_row0) const {
  GlmSpecSegments segs;
  const auto add = [&](void* dst, const void* snapshots, size_t row_stride, size_t bytes) {
    if (segs.count >= kSpecMaxSegments) throw std::logic_error("spec_segments: too many state families");
    segs.seg[segs.count++] = GlmSpecSegment{dst, snapshots, row_stride, bytes};
  };
  const size_t row0 = static_cast<size_t>(snapshot_row0);
  if (num_gdn_ > 0) {
    const size_t rec_bytes = static_cast<size_t>(num_gdn_) * gdn_rec_elems_ * 4;
    const size_t conv_bytes = static_cast<size_t>(num_gdn_) * gdn_conv_elems_ * 2;
    add(gdn_rec(req, 0), spec_rec_ + row0 * num_gdn_ * gdn_rec_elems_, rec_bytes, rec_bytes);
    add(gdn_conv(req, 0), spec_conv_ + row0 * num_gdn_ * gdn_conv_elems_, conv_bytes, conv_bytes);
  }
  if (has_ple_) {
    const size_t bytes = static_cast<size_t>(ple_conv_elems_) * 2;
    add(ple_conv(req), spec_ple_ + row0 * ple_conv_elems_, bytes, bytes);
  }
  add(ctx(req), spec_ctx_ + row0 * 4, 16, 16);
  for (int l = 0; l < num_qsa_; ++l)
    add(pool_.ring(l, req), spec_ring_ + (static_cast<size_t>(l) * static_cast<size_t>(max_decode_rows_) + row0) * ring_elems(),
        ring_elems() * 2, ring_elems() * 2);
  return segs;
}

// ---------------------------------------------------------------------------
// The prefix cache: snapshots and attach.
// ---------------------------------------------------------------------------

size_t QwenModel::session_snapshot_bytes(const QwenTextConfig& cfg, int tp_world, bool mtp) {
  const QwenLocalGeometry geo = QwenLocalGeometry::from_config(
      cfg, 0, tp_world, tp_world > 1 ? QwenHeadSharding::VocabSharded : QwenHeadSharding::Full);
  int num_gdn = 0, num_qsa = 0;
  for (QwenLayerKind k : cfg.layers) (k == QwenLayerKind::Gdn ? num_gdn : num_qsa) += 1;
  const size_t H = static_cast<size_t>(cfg.hidden_size), W = static_cast<size_t>(cfg.hc_count) * H;
  const size_t rec = static_cast<size_t>(geo.local_value_heads) * cfg.gdn_value_head_dim * cfg.gdn_key_head_dim;
  const size_t conv_ch = 2 * static_cast<size_t>(geo.local_key_heads) * cfg.gdn_key_head_dim +
                         static_cast<size_t>(geo.local_value_heads) * cfg.gdn_value_head_dim;
  const size_t conv = conv_ch * static_cast<size_t>(cfg.gdn_conv_width - 1);
  const size_t ring = static_cast<size_t>(cfg.indexer_compress_ratio) * cfg.indexer_head_dim;
  size_t bytes = static_cast<size_t>(num_gdn) * (rec * 4 + conv * 2);
  if (!cfg.ple_layer_ids.empty()) bytes += W * static_cast<size_t>((cfg.ple_conv_kernel_size - 1) * cfg.ngram_size) * 2;
  bytes += 16;
  bytes += static_cast<size_t>(num_qsa + (mtp ? 1 : 0)) * ring * 2;
  if (mtp) bytes += W * 2;
  return bytes;
}

void QwenModel::write_state_snapshot(int req, uint8_t* d, int spec_row) {
  const bool live = spec_row < 0;
  const size_t row = live ? 0 : static_cast<size_t>(spec_row);
  if (num_gdn_ > 0) {
    const size_t rec_bytes = static_cast<size_t>(num_gdn_) * gdn_rec_elems_ * 4;
    const size_t conv_bytes = static_cast<size_t>(num_gdn_) * gdn_conv_elems_ * 2;
    d2d(d, live ? gdn_rec(req, 0) : spec_rec_ + row * num_gdn_ * gdn_rec_elems_, rec_bytes, stream_);
    d += rec_bytes;
    d2d(d, live ? gdn_conv(req, 0) : spec_conv_ + row * num_gdn_ * gdn_conv_elems_, conv_bytes, stream_);
    d += conv_bytes;
  }
  if (has_ple_) {
    const size_t bytes = static_cast<size_t>(ple_conv_elems_) * 2;
    d2d(d, live ? ple_conv(req) : spec_ple_ + row * ple_conv_elems_, bytes, stream_);
    d += bytes;
  }
  d2d(d, live ? ctx(req) : spec_ctx_ + row * 4, 16, stream_);
  d += 16;
  for (int l = 0; l < num_qsa_; ++l) {
    d2d(d, live ? pool_.ring(l, req) : spec_ring_ + (static_cast<size_t>(l) * static_cast<size_t>(max_decode_rows_) + row) * ring_elems(),
        ring_elems() * 2, stream_);
    d += ring_elems() * 2;
  }
}

// The draft's ring: as it stands, or — a hop snapshot after an in-graph
// draft ran past the position — the ring before its rows.
void QwenModel::write_draft_snapshot(int req, uint8_t* d, bool live, int64_t pos) {
  const bool draft_ran_past = !live && mtp_pos_[static_cast<size_t>(req)] > pos;
  d2d(d, draft_ran_past ? mtp_ring_snapshot_ + static_cast<size_t>(req) * ring_elems() : pool_.ring(num_qsa_, req),
      ring_elems() * 2, stream_);
}

size_t QwenModel::snapshot_state_bytes() const {
  size_t bytes = 0;
  if (num_gdn_ > 0) bytes += static_cast<size_t>(num_gdn_) * (gdn_rec_elems_ * 4 + gdn_conv_elems_ * 2);
  if (has_ple_) bytes += static_cast<size_t>(ple_conv_elems_) * 2;
  bytes += 16;  // the n-gram context
  bytes += static_cast<size_t>(num_qsa_) * ring_elems() * 2;
  return bytes;
}

size_t QwenModel::draft_state_bytes() const { return mtp_ ? ring_elems() * 2 : 0; }

void QwenModel::read_state_snapshot(int req, const uint8_t* d) {
  if (num_gdn_ > 0) {
    const size_t rec_bytes = static_cast<size_t>(num_gdn_) * gdn_rec_elems_ * 4;
    const size_t conv_bytes = static_cast<size_t>(num_gdn_) * gdn_conv_elems_ * 2;
    d2d(gdn_rec(req, 0), d, rec_bytes, stream_);
    d += rec_bytes;
    d2d(gdn_conv(req, 0), d, conv_bytes, stream_);
    d += conv_bytes;
  }
  if (has_ple_) {
    const size_t bytes = static_cast<size_t>(ple_conv_elems_) * 2;
    d2d(ple_conv(req), d, bytes, stream_);
    d += bytes;
  }
  d2d(ctx(req), d, 16, stream_);
  d += 16;
  for (int l = 0; l < num_qsa_; ++l) {
    d2d(pool_.ring(l, req), d, ring_elems() * 2, stream_);
    d += ring_elems() * 2;
  }
}

void QwenModel::read_draft_snapshot(int req, const uint8_t* d) {
  d2d(pool_.ring(num_qsa_, req), d, ring_elems() * 2, stream_);
}

// ---------------------------------------------------------------------------
// The graph era.
// ---------------------------------------------------------------------------
void QwenModel::graph_prepare() {
  if (loader_.residency() != QwenResidency::Resident)
    throw std::logic_error("session_graph_prepare: the decode graph needs a resident stack");
  // The bf16 decode weights' 12-bit companions are packed as each layer
  // lands, before any capture.
  const bool pack = Bf12Companions::enabled() && !bf12_built_;
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const QwenLayerResident& r = loader_.load_layer(layer);
    build_layer_objects(r);
    moe_->prepare_graph_table(layer, stream_);
    if (pack) pack_layer_companions(layer, r);
  }
  if (mtp_) {
    // The draft layer's MoE takes the slot after the main stack's.
    const QwenLayerResident& r = loader_.load_layer(cfg_.mtp_layer());
    build_layer_objects(r);
    moe_->prepare_graph_table(n_moe_layers_, stream_);
    if (pack) pack_layer_companions(cfg_.mtp_layer(), r);
  }
  if (pack) finish_companions();
}

// The lossless 12-bit companions of the decode GEMV's bf16 weights
// (engine.bf16_weights; kernels/bf12_gemv.hpp, format v2: the hidden 2560
// and the 1536-wide slices carry a 512-column tail): the GDN's four input
// projections (one multi-problem launch — QwenGdnLayer::in_projections picks
// its packed twin) and its out projection, the QSA's q / k / v / index / o,
// the draft block's, its two fc matrices and the head. Measured on the real
// weights: 2-7 escapes per 10,000, no raw rows; cold GEMVs -15 to -26 %;
// the four-node FP8 recipe +6.4 / +3.8 / +2.4 / +1.3 % at one to four live
// requests, transcripts identical. Under engine.dense_weights = "fp8" every
// one of those matrices is already block-FP8 (the head included): nothing
// is packed. This family keeps both forms resident under either value of
// the key (its loader grants nothing aside: every recipe has the room).
void QwenModel::pack_layer_companions(int layer, const QwenLayerResident& r) {
  if (QwenLayerStream::dense_weights_fp8()) return;
  const auto t0 = std::chrono::steady_clock::now();
  const int64_t H = cfg_.hidden_size;
  const auto release = [&](const void* w) { return loader_.release_packed(layer, w); };
  const auto pack = [&](const uint16_t* w, int64_t n, int64_t k) {
    if (w != nullptr) bf12_.pack_and_release(w, n, k, gemm_, stream_, release);
  };
  if (r.kind == QwenLayerKind::Gdn) {
    const int64_t lk = r.gdn.local_key_heads, lv = r.gdn.local_value_heads;
    const int64_t LV = lv * cfg_.gdn_value_head_dim;
    pack(r.gdn.in_proj_qkv, 2 * lk * cfg_.gdn_key_head_dim + LV, H);
    pack(r.gdn.in_proj_z, LV, H);
    pack(r.gdn.in_proj_a, lv, H);
    pack(r.gdn.in_proj_b, lv, H);
    pack(r.gdn.out_proj, H, LV);
  } else {
    const int64_t D = cfg_.head_dim, lh = r.qsa.local_heads, lkv = r.qsa.local_kv_heads;
    pack(r.qsa.q_proj, lh * 2 * D, H);
    pack(r.qsa.k_proj, lkv * D, H);
    pack(r.qsa.v_proj, lkv * D, H);
    pack(r.qsa.index_qk_proj, static_cast<int64_t>(cfg_.indexer_n_heads + 1) * cfg_.indexer_head_dim, H);
    pack(r.qsa.o_proj, H, lh * D);
  }
  bf12_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void QwenModel::finish_companions() {
  bf12_built_ = true;
  if (QwenLayerStream::dense_weights_fp8()) {
    DGPP_LOG_INFO("rank {} bf12: nothing to pack — engine.dense_weights = fp8 already holds the dense "
                  "projections and the head as block-FP8",
                  loader_.rank());
    return;
  }
  const auto t0 = std::chrono::steady_clock::now();
  const int64_t H = cfg_.hidden_size;
  const auto release = [&](const void* w) { return loader_.release_packed(-1, w); };
  if (mtp_) {
    if (globals_.mtp_fc_embedding) bf12_.pack_and_release(globals_.mtp_fc_embedding, H, H, gemm_, stream_, release);
    if (globals_.mtp_fc_hidden) bf12_.pack_and_release(globals_.mtp_fc_hidden, H, H, gemm_, stream_, release);
  }
  if (globals_.lm_head) bf12_.pack_and_release(globals_.lm_head, lm_vocab_count_, H, gemm_, stream_, release);
  bf12_.finish(gemm_, /*slots=*/1);
  bf12_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  bf12_.log_summary(loader_.rank(), bf12_s_);
}

// The companions' planned bytes: pack_layer_companions' matrices by formula
// (nothing under dense_weights = "fp8").
size_t QwenModel::bf12_plan_bytes(const QwenTextConfig& cfg, const QwenLocalGeometry& geo, bool mtp) {
  if (QwenLayerStream::dense_weights_fp8()) return 0;
  const int64_t H = cfg.hidden_size;
  size_t bytes = Bf12Companions::planned_bytes(geo.lm_vocab_count, H);
  const int64_t lk = geo.local_key_heads, lv = geo.local_value_heads, LV = lv * cfg.gdn_value_head_dim;
  const size_t gdn = Bf12Companions::planned_bytes(2 * lk * cfg.gdn_key_head_dim + LV, H) +
                     Bf12Companions::planned_bytes(LV, H) + 2 * Bf12Companions::planned_bytes(lv, H) +
                     Bf12Companions::planned_bytes(H, LV);
  const int64_t D = cfg.head_dim, lh = geo.local_heads, lkv = geo.local_kv_heads;
  const size_t qsa = Bf12Companions::planned_bytes(lh * 2 * D, H) + 2 * Bf12Companions::planned_bytes(lkv * D, H) +
                     Bf12Companions::planned_bytes(static_cast<int64_t>(cfg.indexer_n_heads + 1) * cfg.indexer_head_dim, H) +
                     Bf12Companions::planned_bytes(H, lh * D);
  for (QwenLayerKind k : cfg.layers) bytes += k == QwenLayerKind::Gdn ? gdn : qsa;
  if (mtp) bytes += qsa + 2 * Bf12Companions::planned_bytes(H, H);
  return bytes;
}

// ---------------------------------------------------------------------------
// The MTP draft block (§1.8): one QSA layer with its own GR sites and MoE
// over R_mtp = fc_h(norm_W(R)) per branch + fc_e(norm_H(embed(tok))),
// then its own mixer and the shared head.
// ---------------------------------------------------------------------------
void QwenModel::mtp_run_rows(int req, const int64_t* tokens, int64_t first_pos, int T, bool decode_row,
                             bool capture, int head_rows, int batch_requests) {
  walk_rows_ = T;
  // The hidden projection has T * hc matrix rows: a shipped 6/8-token
  // walk must not enter the wide guard just because hc expands it to 24/32.
  qwen_configure_gemm_rows(gemm_, T, decode_row);
  if (!mtp_) throw std::logic_error("mtp_run_rows: MTP is not enabled");
  if (T <= 0 || T > max_tokens_) throw std::invalid_argument("mtp_run_rows: rows");
  if (head_rows < 0 || head_rows > T) throw std::invalid_argument("mtp_run_rows: head_rows");
  const int H = cfg_.hidden_size, W = cfg_.hc_count * H, hc = cfg_.hc_count;
  const float eps = cfg_.rms_norm_eps;
  const bool batched = batch_requests > 0;
  const int num_requests = batched ? batch_requests : 1;
  const int64_t* d_pos = decode_row ? d_step_pos_ : d_prefill_pos_;
  const int32_t* d_req = decode_row ? d_req_ids_ : d_prefill_req_;
  const int32_t* d_spans = decode_row ? d_req_spans_ : d_prefill_spans_;
  if (!decode_row) stage_prefill_meta(req, first_pos, T);

  // ---- the input fusion --------------------------------------------------
  // Decode rows gather their hyper states from the slots' windows by
  // position; prefill rows read the main chunk's rows in place (r_).
  const uint16_t* hin = r_;
  if (decode_row) {
    gather_draft_hidden(d_req, d_pos, mtp_hin_, T);
    hin = mtp_hin_;
  }
  qwen_rmsnorm_bf16(hin, globals_.mtp_pre_fc_norm_hidden, mtp_hn_, T, W, eps, stream_);
  qwen_mtp_hidden_projection(gw_, mtp_hn_, globals_.mtp_fc_hidden, mtp_enc_, T, hc, H, decode_row, stream_);
  qwen_mtp_embed_gather_bf16(globals_.embed, tokens, mtp_e_, T, H, stream_);
  // The draft's row at position p embeds token p + 1, so its image window is
  // the main walk's shifted by one (and one row past a chunk's end, which is
  // why staging covers chunk + 1).
  if (prefill_images_) apply_image_embeddings(mtp_e_, first_pos, T, 1, 1);
  qwen_rmsnorm_bf16(mtp_e_, globals_.mtp_pre_fc_norm_embedding, mtp_en_, T, H, eps, stream_);
  // Wide embedding projections can capture an Lt memset node, which is
  // unsafe for collective graph replay. Use MMA above 32 tokens; smaller
  // shapes retain this helper's existing dispatch, with the shared GEMM
  // guard keeping 17-32 decode rows kernel-only too.
  qwen_mtp_hidden_projection(gw_, mtp_en_, globals_.mtp_fc_embedding, mtp_ein_, T, 1, H,
                             decode_row && T > 32, stream_);
  qwen_mtp_fuse_bf16(mtp_ein_, mtp_enc_, mtp_r_, T, hc, H, stream_);

  // ---- the draft layer (the stack's objects rebound to its weights) ------
  const QwenLayerResident& r = loader_.load_layer(cfg_.mtp_layer());
  build_layer_objects(r);
  const auto stage = [&](uint16_t* fallback, int width) -> uint16_t* {
    if (!boundary_) return fallback;
    uint16_t* s = boundary_->stage(T, width);
    if (s == nullptr && capture)
      throw std::runtime_error("mtp_run_rows: a capture fold does not fit the recorder's staged buffer");
    return s ? s : fallback;
  };
  const auto fold = [&](uint16_t* buf, int width) {
    if (!boundary_) return;
    if (!capture) DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
    boundary_->reduce(buf, T, width);
  };
  attn_gr_->mix(mtp_r_, x_, T, stream_);
  uint16_t* attn_out = stage(y_, H);
  {
    QwenQsaCache cache = pool_.view(num_qsa_);
    QwenQsaRows qrows;
    qrows.req_ids = d_req;
    qrows.pos = d_pos;
    qrows.decode = decode_row;
    qrows.request = req;
    qrows.pos0 = first_pos;
    qrows.spans = d_spans;
    qrows.num_requests = num_requests;
    qsa_->enqueue(x_, T, qrows, cache, attn_out, stream_);
  }
  if (decode_row) prefetch_ffn_side(r);
  fold(attn_out, H);
  attn_gr_->combine(mtp_r_, attn_out, T, stream_);
  mlp_gr_->mix(mtp_r_, x_, T, stream_);
  uint16_t* ffn_out = stage(y_, H);
  if (decode_row)
    moe_->enqueue_decode(x_, ffn_out, T, stream_, capture ? n_moe_layers_ : -1);
  else if (moe_prefill_host_path_)
    moe_->enqueue(x_, ffn_out, T, stream_);
  else
    moe_->enqueue_prefill(x_, ffn_out, T, stream_);
  if (decode_row) prefetch_head(globals_.mtp_mixer);
  fold(ffn_out, H);
  mlp_gr_->combine(mtp_r_, ffn_out, T, stream_);
  if (head_rows == 0) {  // prefill rows fill the cache; no head
    if (decode_row) prefetch_.join(stream_);
    return;
  }

  // ---- head: the draft distribution over the last head_rows rows --------
  const uint16_t* head_in = mtp_r_ + static_cast<size_t>(T - head_rows) * W;
  mtp_mixer_->mix(head_in, h_, head_rows, stream_);
  lm_head_logits(h_, head_rows, stream_);
  if (decode_row) prefetch_.join(stream_);  // every forked prefetch back on the main stream
  if (decode_row && (!capture || decode_tail_mirrors_) && head_rows <= max_decode_rows_)
    DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_logits_, logits_,
                                 static_cast<size_t>(head_rows) * lm_vocab_count_ * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
}

QwenModel::Outputs QwenModel::mtp_forward(const std::vector<int64_t>& token_ids) {
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
  DGPP_CUDA_OK(cudaMemcpyAsync(d_tokens_, token_ids.data() + 1, static_cast<size_t>(rows) * 8,
                               cudaMemcpyHostToDevice, stream_));
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

void QwenModel::snapshot_draft_state(int req) {
  glm_device_copy(mtp_ring_snapshot_ + static_cast<size_t>(req) * ring_elems(), pool_.ring(num_qsa_, req),
                  ring_elems() * 2, stream_);
}

void QwenModel::restore_draft_state(int req) {
  glm_device_copy(pool_.ring(num_qsa_, req), mtp_ring_snapshot_ + static_cast<size_t>(req) * ring_elems(),
                  ring_elems() * 2, stream_);
}

// The chain rows' ring copy (depth >= 2): aside before the first chain row,
// back after the last — its own buffer, so the draft snapshot above (the
// fallback's rollback point) stays the pre-draft ring.
void QwenModel::snapshot_chain_state(int req) {
  glm_device_copy(mtp_chain_ring_ + static_cast<size_t>(req) * ring_elems(), pool_.ring(num_qsa_, req),
                  ring_elems() * 2, stream_);
}

void QwenModel::restore_chain_state(int req) {
  glm_device_copy(pool_.ring(num_qsa_, req), mtp_chain_ring_ + static_cast<size_t>(req) * ring_elems(),
                  ring_elems() * 2, stream_);
}

uint64_t QwenModel::vision_digest() const { return vision_ ? vision_->digest() : 0; }

QwenModel::Outputs QwenModel::session_prefill_images(
    int req, const std::vector<int64_t>& prompt, const std::vector<ImageInput>& images) {
  return session_prefill_images(req, prompt, images, {}, nullptr);
}

QwenModel::Outputs QwenModel::session_prefill_images(
    int req, const std::vector<int64_t>& prompt, const std::vector<ImageInput>& images,
    const std::vector<int64_t>& boundaries, SnapshotRequest* snap) {
  return session_prefill_with_images(req, prompt, images, boundaries, snap, false);
}

QwenModel::Outputs QwenModel::session_prefill_resume_images(
    int req, const std::vector<int64_t>& suffix, const std::vector<ImageInput>& images,
    const std::vector<int64_t>& boundaries, SnapshotRequest* snap) {
  return session_prefill_with_images(req, suffix, images, boundaries, snap, true);
}

QwenModel::Outputs QwenModel::session_prefill_with_images(
    int req, const std::vector<int64_t>& ids, const std::vector<ImageInput>& images,
    const std::vector<int64_t>& boundaries, SnapshotRequest* snap, bool resume) {
  if (!vision_) throw std::invalid_argument("Qwen: checkpoint has no vision tower");
  if (req < 0 || req >= max_requests_) throw std::out_of_range("Qwen image prefill: request slot");
  const int64_t start = resume ? session_pos_[static_cast<size_t>(req)] : 0;
  if (start < 0) throw std::invalid_argument("Qwen image prefill: closed session");
  validate_image_inputs(images, start + static_cast<size_t>(ids.size()));
  if (images.empty())
    return resume ? session_prefill_resume(req, ids, boundaries, snap)
                  : session_prefill(req, ids, boundaries, snap);
  // The frontend rendered one pad id per visual token; a prompt that names
  // a different span (or none) cannot be given the tower's rows.
  for (const auto& im : images)
    for (int64_t pos = std::max(start, im.offset); pos < im.offset + im.tokens; ++pos) {
      if (ids[static_cast<size_t>(pos - start)] != image_pad_id())
        throw std::invalid_argument("Qwen: image span does not contain image tokens");
    }
  prefill_images_ = &images;
  image_embeddings_ = nullptr;
  image_window_first_ = image_window_end_ = 0;
  try {
    auto out = resume ? session_prefill_resume(req, ids, boundaries, snap)
                      : session_prefill(req, ids, boundaries, snap);
    prefill_images_ = nullptr;
    image_embeddings_ = nullptr;
    return out;
  } catch (...) {
    prefill_images_ = nullptr;
    image_embeddings_ = nullptr;
    throw;
  }
}

void QwenModel::stage_image_embeddings(int64_t first, int64_t end) {
  image_window_first_ = first;
  image_window_end_ = end;
  image_embeddings_ = prefill_images_ && !prefill_images_->empty() && end > first
                          ? vision_->stage(*prefill_images_, first, end)
                          : nullptr;
}

void QwenModel::apply_image_embeddings(uint16_t* dst, int64_t first, int rows, int shift,
                                       int branches) {
  if (!prefill_images_ || prefill_images_->empty() || rows <= 0) return;
  const int64_t begin_all = first + shift, end_all = begin_all + rows;
  if (begin_all < image_window_first_ || end_all > image_window_end_)
    stage_image_embeddings(begin_all, end_all);
  const int h = cfg_.hidden_size;
  for (const auto& im : *prefill_images_) {
    const int64_t begin = std::max(begin_all, im.offset),
                  end = std::min(end_all, im.offset + im.tokens);
    if (end <= begin) continue;
    if (!image_embeddings_ || begin < image_window_first_ || end > image_window_end_)
      throw std::logic_error("Qwen: image consumer escaped its staged window");
    const uint16_t* source = image_embeddings_ + (begin - image_window_first_) * h;
    const int n = static_cast<int>(end - begin);
    if (branches == 1)
      qwen_image_copy(source, dst + (begin - begin_all) * h, n, h, stream_);
    else
      qwen_image_broadcast(source, dst + (begin - begin_all) * branches * h, n, h, stream_);
  }
}

}  // namespace dgpp
