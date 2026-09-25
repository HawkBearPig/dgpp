#include "models/glm_dsa/model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "kernels/bf12_companions.hpp"
#include "kernels/dsa.hpp"
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

size_t align256(size_t b) { return (b + 255) / 256 * 256; }

}  // namespace

// ---------------------------------------------------------------------------
// The DSA configuration and the pool's index-cache map.
// ---------------------------------------------------------------------------

DsaConfig GlmDsaModel::dsa_config(const GlmDsaTextConfig& cfg, int tp_world, bool mtp,
                                  LatentFormat format) {
  DsaConfig d;
  d.hidden = cfg.hidden_size;
  d.num_heads = cfg.num_attention_heads;
  d.q_lora_rank = cfg.q_lora_rank;
  d.kv_lora_rank = cfg.kv_lora_rank;
  d.qk_nope_head_dim = cfg.qk_nope_head_dim;
  d.qk_rope_head_dim = cfg.qk_rope_head_dim;
  d.v_head_dim = cfg.v_head_dim;
  d.index_n_heads = cfg.index_n_heads;
  d.index_head_dim = cfg.index_head_dim;
  d.index_topk = cfg.index_topk;
  d.index_kpool = 1;  // per-token selection (plan D8)
  d.always_select_tail = 1;
  d.index_relu = 1;   // relu(q_h . k) per indexer head (plan §1.3)
  d.num_dsa_layers = cfg.num_hidden_layers + (mtp ? 1 : 0);
  d.num_index_layers = cfg.num_indexer_layers() + (mtp ? 1 : 0);
  d.block_tokens = kBlockTokens;
  d.tp_size = tp_world;
  d.rms_norm_eps = cfg.rms_norm_eps;
  d.latent_format = format;
  d.gemm_mma_from_rows = dense_gemv_rows() + 1;
  DsaConfig::validate_config(d);
  return d;
}

std::vector<int> GlmDsaModel::index_ordinals(const GlmDsaTextConfig& cfg, bool mtp) {
  std::vector<int> ord;
  int next = 0;
  for (int l = 0; l < cfg.num_hidden_layers; ++l) ord.push_back(cfg.owns_indexer(l) ? next++ : -1);
  if (mtp) ord.push_back(next++);  // the draft layer's own indexer
  return ord;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

GlmDsaModel::GlmDsaModel(const GlmDsaTextConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
                         int64_t max_cache_tokens, GlmDsaResidency residency, BoundaryReducer* boundary,
                         int tp_rank, int tp_world, int max_requests, bool mtp, int decode_rows,
                         LatentFormat latent_format)
    : cfg_(cfg),
      loader_(cfg, checkpoint_dir, tp_rank, tp_world, residency,
              tp_world > 1 ? GlmDsaHeadSharding::VocabSharded : GlmDsaHeadSharding::Full,
              mtp && residency == GlmDsaResidency::Resident),
      latent_format_(latent_format) {
  if (max_tokens <= 0) throw std::invalid_argument("GlmDsaModel: max_tokens must be positive");
  if (mtp && cfg_.mtp_layer() < 0) throw std::invalid_argument("GlmDsaModel: the config has no draft layer (mtp)");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("GlmDsaModel: max_requests must be in [1, kPickMaxRequests]");
  // Plan D9: the fixed batch (slots x verify rows) is capped at
  // decode_rows_cap() rows (the select's row groups and the pick kernels'
  // request slots are sized there).
  if (max_requests > decode_rows_cap() || decode_rows > decode_rows_cap())
    throw std::invalid_argument("GlmDsaModel: the decode batch is bounded at " +
                                std::to_string(decode_rows_cap()) +
                                " rows (plan D9): max_requests and decode_rows must not exceed it");
  if ((tp_world > 1) != (boundary != nullptr))
    throw std::invalid_argument(
        "GlmDsaModel: a boundary reducer is required exactly when tp_world > 1 — without one the "
        "block-boundary partials would be returned silently as results");
  if (cfg_.eos_token_ids.empty()) throw std::invalid_argument("GlmDsaModel: the config names no EOS token");
  init_stream();
  loader_.set_reader_stream(stream_);
  log_memory_ledger("glm_dsa: loader opened");
  const int H = cfg_.hidden_size;
  globals_ = loader_.load_globals();
  log_memory_ledger("glm_dsa: globals resident");
  embed_sharded_ = globals_.embed_vocab_count < cfg_.vocab_size;
  if (embed_sharded_ && boundary == nullptr)
    throw std::invalid_argument("GlmDsaModel: a vocab-sharded embedding needs the boundary reducer (world > 1)");
  // A resident stack materializes every layer here, before any cache or
  // scratch exists, and releases the checkpoint mappings, their page cache
  // and the loader's pinned staging mirror at once — before the bus has a
  // collective in flight (a release mid-forward would wait on its
  // persistent kernels) — so the caches below take the memory the load
  // used rather than sit beside it (the plan counts the staging only
  // beyond what the caches replace).
  // The bf16 decode weights' 12-bit companions are packed as each layer
  // lands (under bf12-only residency the layer's bf16 bytes go back at once:
  // no more than one layer's sit beside their companions).
  if (residency == GlmDsaResidency::Resident) {
    const int layers = cfg_.num_hidden_layers + (mtp && cfg_.mtp_layer() >= 0 ? 1 : 0);
    const bool pack = Bf12Companions::enabled();
    for (int l = 0; l < layers; ++l) {
      const GlmDsaLayerResident& r = loader_.load_layer(l);
      if (pack) pack_layer_companions(l, r);
    }
    if (pack) finish_companions();
    loader_.release_sources();
  }
  log_memory_ledger("glm_dsa: layers resident, sources released");
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
    sp.snapshot_align = 1;  // per-token selection: any position snapshots
    sp.draft_width = H;
    sp.eos = static_cast<int32_t>(cfg_.eos_token_ids[0]);
    init_session(sp);
  }
  if (max_decode_rows_ > decode_rows_cap())
    throw std::logic_error("GlmDsaModel: the session core widened the decode batch past the family's cap");
  gemm_ws_bytes_ = std::max<size_t>(64u << 20, gemm_.query_workspace_bytes(max_tokens_, lm_vocab_count_, H, DType::BF16));
  gemm_ws_ = dev_alloc<char>(gemm_ws_bytes_);
  // The dense sites' lowering (kernels/gemm.hpp dense_gemv_rows): the GEMV
  // chunks to the bound, cuBLASLt's algorithm (bf16) or the streaming
  // tensor-core GEMM (the DSA layer's fp8 projections) above it.
  gemm_.set_decode_rows(std::min(max_decode_rows_, dense_gemv_rows()));
  const GlmDsaLocalGeometry& geo = loader_.geometry();
  moe_cfg_ = cfg_.moe_config(static_cast<int>(geo.local_inter));
  dsa_cfg_ = dsa_config(cfg_, tp_world, mtp_, latent_format_);

  // The DSA pool and the layer's scratch from one arena.
  const int64_t slots = max_cache_tokens_;
  dsa_scratch_bytes_ = DsaLayer::scratch_bytes(dsa_cfg_, max_tokens_, slots, max_decode_rows_,
                                               kDsaDecodeSplit, kDotBudget);
  {
    Arena::Config ac;
    ac.persistent_hot = DsaStatePool::cache_bytes(dsa_cfg_, max_requests_, slots) + align256(dsa_scratch_bytes_) + 4096;
    arena_.init(ac);
    pool_.init(arena_, dsa_cfg_, max_requests_, slots, index_ordinals(cfg_, mtp_));
    dsa_scratch_ = arena_.alloc_persistent(MemClass::DeviceHot, dsa_scratch_bytes_, 256);
    log_memory_ledger("glm_dsa: arena, pool and scratch");
  }
  // The rotary table over the pool's positions (the layer requires it to
  // cover max_cache_tokens).
  if (cfg_.qk_rope_head_dim > 0) {
    rope_positions_ = slots;
    const size_t n = static_cast<size_t>(rope_positions_) * 2 * static_cast<size_t>(cfg_.qk_rope_head_dim / 2);
    std::vector<uint16_t> host(n);
    dsa_rope_table_host(cfg_.rope_theta, cfg_.qk_rope_head_dim, rope_positions_, host.data());
    rope_table_ = dev_alloc<uint16_t>(n);
    DGPP_CUDA_OK(cudaMemcpyAsync(rope_table_, host.data(), n * 2, cudaMemcpyHostToDevice, stream_));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  }

  const size_t M = static_cast<size_t>(max_tokens_);
  resid_ = dev_alloc<uint16_t>(M * H);
  x_ = dev_alloc<uint16_t>(M * H);
  y_ = dev_alloc<uint16_t>(M * H);
  if (cfg_.first_k_dense_replace > 0) {
    const size_t I = static_cast<size_t>(geo.local_dense_inter);
    dense_g_ = dev_alloc<uint16_t>(M * I);
    dense_u_ = dev_alloc<uint16_t>(M * I);
    dense_act_ = dev_alloc<uint16_t>(M * I);
  }
  {
    const size_t layers = static_cast<size_t>(cfg_.num_moe_layers()) + (mtp_ ? 1 : 0);
    const size_t K = static_cast<size_t>(cfg_.num_experts_per_tok);
    h_route_ids_ = pinned_alloc<int32_t>(layers * M * K);
    h_route_weights_ = pinned_alloc<float>(layers * M * K);
    prefetch_window_bytes_ = std::getenv("DGPP_L2_PREFETCH_MB") ? 0 : (size_t{20} << 20);
  }
  if (mtp_) {
    mtp_h_ = dev_alloc<uint16_t>(M * H);
    mtp_in_ = dev_alloc<uint16_t>(M * 2 * H);
    if (embed_sharded_) {
      mtp_embed_rows_ = dev_alloc<uint16_t>(M * H);
      d_iota_ = dev_alloc<int64_t>(M);
      std::vector<int64_t> iota(M);
      for (size_t i = 0; i < M; ++i) iota[i] = static_cast<int64_t>(i);
      DGPP_CUDA_OK(cudaMemcpy(d_iota_, iota.data(), M * sizeof(int64_t), cudaMemcpyHostToDevice));
    }
    mtp_r_ = dev_alloc<uint16_t>(M * H);
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}

// The lossless 12-bit companions of the decode GEMV's bf16 weights
// (engine.bf16_weights = "bf12"; kernels/bf12_companions.hpp): what this
// checkpoint leaves bf16 on the matmul seam — the dense layers' and the
// draft's attention projections and MLPs, every indexer, the head and
// eh_proj (2.25 of the 3.65 GB of bf16 a step reads per rank at world 4;
// kv_b and the routers ride their own kernels), packed as each layer lands.
// Prefill and the batches past eight rows keep the bf16 bytes — or, under
// bf12-only residency (the bytes returned here, layer by layer), expand
// them and take the packed launches (kernels/gemm.hpp).
void GlmDsaModel::pack_layer_companions(int layer, const GlmDsaLayerResident& r) {
  const auto t0 = std::chrono::steady_clock::now();
  const int64_t H = cfg_.hidden_size;
  const auto release = [&](const void* w) { return loader_.release_packed(layer, w); };
  const auto pack = [&](const uint16_t* w, int64_t n, int64_t k) {
    bf12_.pack_and_release(w, n, k, gemm_, stream_, release);
  };
  const GlmDsaAttnResident& a = r.attn;
  const int64_t lh = a.local_heads;
  if (!a.packed()) {
    pack(a.qkv_a, cfg_.q_lora_rank + cfg_.kv_lora_rank + cfg_.qk_rope_head_dim, H);
    pack(a.q_b, lh * cfg_.qk_head_dim(), cfg_.q_lora_rank);
  }
  if (a.o_proj != nullptr && a.o_proj_packed.packed == nullptr) pack(a.o_proj, H, lh * cfg_.v_head_dim);
  if (a.owns_indexer()) {
    pack(a.wq_b, static_cast<int64_t>(cfg_.index_n_heads) * cfg_.index_head_dim, cfg_.q_lora_rank);
    pack(a.wk, cfg_.index_head_dim, H);
    pack(a.wp, cfg_.index_n_heads, H);
  }
  if (!r.moe && r.dense.gate != nullptr) {
    pack(r.dense.gate, r.dense.local_inter, H);
    pack(r.dense.up, r.dense.local_inter, H);
    pack(r.dense.down, H, r.dense.local_inter);
  }
  if (r.eh_proj != nullptr) pack(r.eh_proj, H, 2 * H);
  bf12_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// The lm head's, the expansion scratch (one slot: no walk of this family
// calls a matrix twice), and the summary.
void GlmDsaModel::finish_companions() {
  const auto t0 = std::chrono::steady_clock::now();
  // (The session core learns the head's rows after the load: the globals' here.)
  const int head_rows = globals_.lm_vocab_count > 0 ? globals_.lm_vocab_count : cfg_.vocab_size;
  bf12_.pack_and_release(globals_.lm_head, head_rows, cfg_.hidden_size, gemm_, stream_,
                         [&](const void* w) { return loader_.release_packed(-1, w); });
  bf12_.finish(gemm_, kBf12ExpandSlots);
  bf12_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  bf12_.log_summary(loader_.rank(), bf12_s_);
}

// bf12-only residency's expansion slot: the largest packed matrix under the
// cap, by formula (pack_layer_companions' matrices).
size_t GlmDsaModel::bf12_slot_bytes(const GlmDsaTextConfig& cfg, int tp_rank, int tp_world,
                                    GlmDsaHeadSharding head, bool mtp) {
  const GlmDsaLocalGeometry geo = GlmDsaLocalGeometry::from_config(cfg, tp_rank, tp_world, head);
  const size_t H = static_cast<size_t>(cfg.hidden_size), lh = static_cast<size_t>(geo.local_heads);
  const auto ok = [](size_t n, size_t k) {
    return bf12_shape_ok(static_cast<int>(n), static_cast<int>(k)) ? n * k * 2 : size_t{0};
  };
  const size_t q_lora = static_cast<size_t>(cfg.q_lora_rank);
  bool bf16_attn = mtp && cfg.mtp_layer() >= 0, dense = false, indexer = false;
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    bf16_attn = bf16_attn || !cfg.packed_layer(l);
    dense = dense || !cfg.is_moe_layer(l);
    indexer = indexer || cfg.owns_indexer(l);
  }
  const size_t I = static_cast<size_t>(geo.local_dense_inter);
  return Bf12Companions::planned_slot_bytes(
      {bf16_attn ? ok(static_cast<size_t>(cfg.q_lora_rank + cfg.kv_lora_rank + cfg.qk_rope_head_dim), H) : 0,
       bf16_attn ? ok(lh * static_cast<size_t>(cfg.qk_head_dim()), q_lora) : 0,
       bf16_attn ? ok(H, lh * static_cast<size_t>(cfg.v_head_dim)) : 0,
       indexer ? ok(static_cast<size_t>(cfg.index_n_heads) * static_cast<size_t>(cfg.index_head_dim), q_lora) : 0,
       indexer ? ok(static_cast<size_t>(cfg.index_head_dim), H) : 0,
       indexer ? ok(static_cast<size_t>(cfg.index_n_heads), H) : 0, dense ? ok(I, H) : 0, dense ? ok(H, I) : 0,
       mtp && cfg.mtp_layer() >= 0 ? ok(H, 2 * H) : 0, ok(static_cast<size_t>(geo.lm_vocab_count), H)});
}

// The companions' planned bytes: build_bf12_companions' matrices by formula.
size_t GlmDsaModel::bf12_plan_bytes(const GlmDsaTextConfig& cfg, int tp_rank, int tp_world,
                                    GlmDsaHeadSharding head, bool mtp) {
  const GlmDsaLocalGeometry geo = GlmDsaLocalGeometry::from_config(cfg, tp_rank, tp_world, head);
  const int64_t H = cfg.hidden_size, lh = geo.local_heads;
  const size_t attn = Bf12Companions::planned_bytes(cfg.q_lora_rank + cfg.kv_lora_rank + cfg.qk_rope_head_dim, H) +
                      Bf12Companions::planned_bytes(lh * cfg.qk_head_dim(), cfg.q_lora_rank) +
                      Bf12Companions::planned_bytes(H, lh * cfg.v_head_dim);
  const size_t indexer =
      Bf12Companions::planned_bytes(static_cast<int64_t>(cfg.index_n_heads) * cfg.index_head_dim, cfg.q_lora_rank) +
      Bf12Companions::planned_bytes(cfg.index_head_dim, H) + Bf12Companions::planned_bytes(cfg.index_n_heads, H);
  const size_t dense = 2 * Bf12Companions::planned_bytes(geo.local_dense_inter, H) +
                       Bf12Companions::planned_bytes(H, geo.local_dense_inter);
  size_t bytes = Bf12Companions::planned_bytes(geo.lm_vocab_count, H);
  const int layers = cfg.num_hidden_layers + (mtp && cfg.mtp_layer() >= 0 ? 1 : 0);
  for (int l = 0; l < layers; ++l) {
    const bool draft = l == cfg.mtp_layer();
    if (draft || !cfg.packed_layer(l)) bytes += attn;
    if (cfg.owns_indexer(l)) bytes += indexer;
    if (!draft && !cfg.is_moe_layer(l)) bytes += dense;
    if (draft) bytes += Bf12Companions::planned_bytes(H, 2 * H);
  }
  return bytes;
}

GlmDsaModel::~GlmDsaModel() {
  cudaFree(resid_);
  cudaFree(x_);
  cudaFree(y_);
  cudaFree(dense_g_);
  cudaFree(dense_u_);
  cudaFree(dense_act_);
  cudaFreeHost(h_route_ids_);
  cudaFreeHost(h_route_weights_);
  cudaFree(gemm_ws_);
  cudaFree(rope_table_);
  cudaFree(mtp_h_);
  cudaFree(mtp_in_);
  cudaFree(mtp_r_);
  cudaFree(mtp_embed_rows_);
  cudaFree(d_iota_);
}

int GlmDsaModel::table_slots() const {
  return loader_.residency() == GlmDsaResidency::Resident ? cfg_.num_moe_layers() + (mtp_ ? 1 : 0) : 0;
}

size_t GlmDsaModel::kv_block_bytes_static(const GlmDsaTextConfig& cfg, int tp_world, bool mtp,
                                          LatentFormat format) {
  const DsaConfig dsa = dsa_config(cfg, tp_world, mtp, format);
  const DsaGeometry g = DsaGeometry::from_config(dsa);
  const size_t layers = static_cast<size_t>(dsa.num_dsa_layers);
  const size_t ilayers = static_cast<size_t>(g.index_layers);
  const size_t pools = static_cast<size_t>(g.pools_per_block);
  return layers * static_cast<size_t>(dsa.block_tokens) * (g.latent_bytes_per_token + g.latent_scale_bytes_per_token) +
         ilayers * pools * (g.index_k_bytes_per_pool + sizeof(float));
}

GlmDsaModel::MemoryPlan GlmDsaModel::plan_memory(const GlmDsaTextConfig& cfg, int max_tokens,
                                                 int64_t max_cache_tokens, int tp_rank, int tp_world,
                                                 GlmDsaResidency residency, int max_requests, bool mtp,
                                                 int decode_rows, LatentFormat latent_format) {
  if (max_tokens <= 0) throw std::invalid_argument("plan_memory: max_tokens must be positive");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("plan_memory: max_requests must be in [1, kPickMaxRequests]");
  if (max_requests > decode_rows_cap() || decode_rows > decode_rows_cap())
    throw std::invalid_argument("plan_memory: the decode batch is bounded at " +
                                std::to_string(decode_rows_cap()) + " rows (plan D9)");
  const int rows = std::max({kDecodeRows, decode_rows, max_requests});
  if (mtp && cfg.mtp_layer() < 0) throw std::invalid_argument("plan_memory: the config has no draft layer");
  const GlmDsaHeadSharding head = tp_world > 1 ? GlmDsaHeadSharding::VocabSharded : GlmDsaHeadSharding::Full;
  const GlmDsaLocalGeometry geo = GlmDsaLocalGeometry::from_config(cfg, tp_rank, tp_world, head);
  const int64_t cache_tokens =
      ((std::max<int64_t>(max_cache_tokens, max_tokens) + kBlockTokens - 1) / kBlockTokens) * kBlockTokens;
  MemoryPlan plan;
  plan.context_tokens = std::min<int64_t>(cache_tokens, cfg.max_position_embeddings);
  const size_t M = static_cast<size_t>(max_tokens);
  const size_t R = static_cast<size_t>(max_requests);
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t V = static_cast<size_t>(GlmDsaLayerStream::lm_vocab_count(cfg, tp_rank, tp_world, head));
  const DsaConfig dsa = dsa_config(cfg, tp_world, mtp, latent_format);

  size_t staging = 0;
  // bf12-only residency: the packable bf16 matrices load aside and give
  // their bytes back as each layer is packed (the companions and the
  // prefill scratch are their own lines; all of it precedes the caches).
  const bool bf12_only = Bf12Companions::packed_only() && residency == GlmDsaResidency::Resident;
  if (residency == GlmDsaResidency::Resident) {
    size_t resident = GlmDsaLayerStream::resident_bytes(cfg, tp_rank, tp_world, head, mtp);
    if (bf12_only) resident -= GlmDsaLayerStream::side_bytes(cfg, tp_rank, tp_world, head, mtp);
    plan.add(bf12_only ? "model weights (resident, packed bf16 matrices released)" : "model weights (resident)",
             resident);
    staging = GlmDsaLayerStream::staging_plan_bytes(cfg, tp_rank, tp_world, head, mtp);
  } else {
    size_t largest = 0;
    const int layers_total = cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
    for (int l = 0; l < layers_total; ++l)
      largest = std::max(largest, GlmDsaLayerStream::layer_bytes(cfg, l, tp_rank, tp_world));
    plan.add("model weights (one streamed layer + globals)",
             largest + GlmDsaLayerStream::globals_bytes(cfg, tp_rank, tp_world, head));
  }
  if (Bf12Companions::enabled() && residency == GlmDsaResidency::Resident)
    plan.add("bf16 decode packing (12-bit companions)", bf12_plan_bytes(cfg, tp_rank, tp_world, head, mtp));
  if (bf12_only)
    plan.add("bf16 prefill expansion scratch",
             static_cast<size_t>(kBf12ExpandSlots) * bf12_slot_bytes(cfg, tp_rank, tp_world, head, mtp));
  const size_t after_weights = plan.total_bytes();
  plan.add("gemm workspace (at least)", size_t{64} << 20);
  plan.add(std::string("dsa cache pool (latent ") + latent_format_name(latent_format) +
               " + rope keys on every layer, fp8 index caches on the indexed layers)",
           DsaStatePool::cache_bytes(dsa, max_requests, cache_tokens));
  plan.add("dsa scratch (projections, selection, attention, the prefill dot tiles)",
           DsaLayer::scratch_bytes(dsa, max_tokens, cache_tokens, rows, kDsaDecodeSplit, kDotBudget));
  if (cfg.qk_rope_head_dim > 0)
    plan.add("rotary table (bf16 cos/sin per position)",
             static_cast<size_t>(cache_tokens) * static_cast<size_t>(cfg.qk_rope_head_dim) * 2);
  {
    size_t core_dev = 0, core_pin = 0;
    session_core_plan_bytes(max_tokens, max_requests, cfg.hidden_size, static_cast<int>(V), mtp,
                            cfg.hidden_size, &core_dev, &core_pin, rows);
    const size_t moe_layers = static_cast<size_t>(cfg.num_moe_layers()) + (mtp ? 1 : 0);
    const size_t K = static_cast<size_t>(cfg.num_experts_per_tok);
    size_t act = core_dev + 3 * M * H * 2;
    if (cfg.first_k_dense_replace > 0) act += 3 * M * static_cast<size_t>(geo.local_dense_inter) * 2;
    plan.add("activations (session core, residual, block io, dense scratch, route staging)", act,
             core_pin + moe_layers * M * K * 8);
  }
  {
    const GlmMoeConfig moe_cfg = cfg.moe_config(static_cast<int>(geo.local_inter));
    const int slots = residency == GlmDsaResidency::Resident ? cfg.num_moe_layers() + (mtp ? 1 : 0) : 0;
    size_t moe_pinned = 0;
    const size_t moe_dev = GlmMoeLayer::scratch_bytes(moe_cfg, max_tokens, rows, slots, &moe_pinned);
    plan.add("moe scratch (routed slots, shared expert, graph tables)", moe_dev, moe_pinned);
  }
  if (mtp) plan.add("draft block (gathered hidden, the fused input, its residual, the sharded embedding's rows)",
                    M * H * 2 + M * 2 * H * 2 + M * H * 2 + M * H * 2 + M * 8);
  (void)R;
  if (staging > 0) {
    // The constructor materializes the stack before the caches, so the
    // staging mirror is gone when they are allocated: only the part the
    // caches do not replace stacks with the weights at the boot's peak.
    const size_t later = plan.total_bytes() - after_weights;
    plan.add("loader staging beyond the caches that replace it (pinned host, transient)", 0,
             staging > later ? staging - later : 0);
  }
  return plan;
}

size_t GlmDsaModel::session_snapshot_bytes(const GlmDsaTextConfig& cfg, int, bool mtp) {
  size_t bytes = kSnapshotStamp;
  if (mtp) bytes += static_cast<size_t>(cfg.hidden_size) * 2;  // the draft's hidden row
  return bytes;
}

// ---------------------------------------------------------------------------
// The weight views.
// ---------------------------------------------------------------------------

DsaLayerWeights GlmDsaModel::dsa_view(const GlmDsaAttnResident& a) const {
  DsaLayerWeights w;
  // The indexer (null on a shared-selection layer: DsaLayer then reuses).
  w.wq_b = a.wq_b;
  w.wk = a.wk;
  w.wp = a.wp;
  w.k_norm_w = a.k_norm_w;
  w.k_norm_b = a.k_norm_b;
  w.gate = nullptr;  // kpool 1: no compression gate, no APE
  w.ape = nullptr;
  // The projections: the packed triples on the packed layers, bf16 else.
  if (a.packed()) {
    w.qkv_a_p = a.qkv_a_packed;
    w.q_b_p = a.q_b_packed;
    w.o_proj_p = a.o_proj_packed;
  } else {
    w.qkv_a = a.qkv_a;
    w.q_b = a.q_b;
    w.o_proj = a.o_proj;
  }
  w.q_aln = a.q_aln;
  w.kv_aln = a.kv_aln;
  w.kv_b = a.kv_b;
  w.rope_table = rope_table_;
  w.rope_table_positions = rope_positions_;
  return w;
}

GlmMoeWeights GlmDsaModel::moe_view(const GlmDsaMoeResident& m) {
  GlmMoeWeights w;
  w.router_gate = m.router;
  w.router_bias = m.router_bias;
  for (int i = 0; i < 3; ++i) w.shared_packed[i] = m.shared[i];
  w.experts = nullptr;
  w.experts_fp4 = nullptr;
  w.experts_packed = m.experts.data();
  return w;
}

void GlmDsaModel::build_layer_objects(const GlmDsaLayerResident& r) {
  if (!dsa_) {
    dsa_ = std::make_unique<DsaLayer>(gemm_, dsa_view(r.attn), dsa_cfg_, max_tokens_, pool_.max_token_slots(),
                                      dsa_scratch_, dsa_scratch_bytes_, gemm_ws_, gemm_ws_bytes_,
                                      max_decode_rows_, kDsaDecodeSplit, kDotBudget);
  } else {
    dsa_->rebind(dsa_view(r.attn));
  }
  if (r.moe) {
    if (!moe_) {
      moe_ = std::make_unique<GlmMoeLayer>(moe_view(r.moe_w), moe_cfg_, max_tokens_, max_decode_rows_, table_slots());
    } else {
      moe_->rebind(moe_view(r.moe_w));
    }
  }
}

// The state every row walk starts from on a fresh request: no blocks (the
// cache rows are written before any row reads them), zeroed rings.
void GlmDsaModel::reset_slot_state(int req) { pool_.reset_request(req, stream_); }

GlmSpecSegments GlmDsaModel::spec_segments(int, int) const { return GlmSpecSegments{}; }

void GlmDsaModel::write_state_snapshot(int, uint8_t* d, int) {
  DGPP_CUDA_OK(cudaMemsetAsync(d, 0, kSnapshotStamp, stream_));
}

// ---------------------------------------------------------------------------
// The boundary prefetch windows (bit-identical on or off).
// ---------------------------------------------------------------------------
void GlmDsaModel::prefetch_packed(const GlmPackedMatrix& m) {
  if (m.packed) prefetch_.add(m.packed, m.packed_bytes());
  if (m.scales) prefetch_.add(m.scales, m.scale_bytes());
}

// Before the attention fold: this layer's post norm, the router (and its
// bias) and the shared expert — or the dense MLP — one image.
void GlmDsaModel::prefetch_ffn_side(const GlmDsaLayerResident& r) {
  if (!prefetch_.enabled()) return;
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, prefetch_window_bytes_, prefetch_.boundary_rate());
  if (r.post_norm) prefetch_.add(r.post_norm, H * 2);
  if (r.moe) {
    if (r.moe_w.router) prefetch_.add(r.moe_w.router, static_cast<size_t>(cfg_.n_routed_experts) * H * 2);
    if (r.moe_w.router_bias) prefetch_.add(r.moe_w.router_bias, static_cast<size_t>(cfg_.n_routed_experts) * 4);
    for (int i = 0; i < 3; ++i) prefetch_packed(r.moe_w.shared[i]);
  } else {
    const size_t I = static_cast<size_t>(r.dense.local_inter);
    if (r.dense.gate) prefetch_bf16(r.dense.gate, I * H * 2);
    if (r.dense.up) prefetch_bf16(r.dense.up, I * H * 2);
    if (r.dense.down) prefetch_bf16(r.dense.down, I * H * 2);
  }
}

// Before the MLP fold: the next layer's input norm and its first
// projections (or the head). Resident stacks only (load_layer is a lookup
// there).
void GlmDsaModel::prefetch_attention_side(int layer) {
  if (!prefetch_.enabled()) return;
  if (layer >= cfg_.num_hidden_layers) {
    prefetch_head();
    return;
  }
  if (loader_.residency() != GlmDsaResidency::Resident) return;
  const GlmDsaLayerResident& r = loader_.load_layer(layer);
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, prefetch_window_bytes_, prefetch_.boundary_rate());
  if (r.input_norm) prefetch_.add(r.input_norm, H * 2);
  const GlmDsaAttnResident& a = r.attn;
  const size_t qkv_rows = static_cast<size_t>(cfg_.q_lora_rank + cfg_.kv_lora_rank + cfg_.qk_rope_head_dim);
  if (a.packed()) {
    prefetch_packed(a.qkv_a_packed);
    prefetch_packed(a.q_b_packed);
  } else {
    if (a.qkv_a) prefetch_bf16(a.qkv_a, qkv_rows * H * 2);
    if (a.q_b)
      prefetch_bf16(a.q_b, static_cast<size_t>(a.local_heads) * static_cast<size_t>(cfg_.qk_head_dim()) *
                               static_cast<size_t>(cfg_.q_lora_rank) * 2);
  }
  if (a.owns_indexer()) {
    prefetch_bf16(a.wq_b, static_cast<size_t>(cfg_.index_n_heads) * cfg_.index_head_dim *
                              static_cast<size_t>(cfg_.q_lora_rank) * 2);
    prefetch_bf16(a.wk, static_cast<size_t>(cfg_.index_head_dim) * H * 2);
    prefetch_bf16(a.wp, static_cast<size_t>(cfg_.index_n_heads) * H * 2);
  }
}

// A bf16 matmul weight into the open window: the bytes the walk's launch
// streams — its packed companion's when the GEMM holds one.
void GlmDsaModel::prefetch_bf16(const uint16_t* w, size_t bytes) {
  const void* view = nullptr;
  size_t view_bytes = 0;
  gemm_.resident_view(w, bytes, walk_rows_, &view, &view_bytes);
  prefetch_.add_view(w, view, view_bytes);  // a companion is its own allocation
}

void GlmDsaModel::prefetch_head() {
  if (!prefetch_.enabled()) return;
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, prefetch_window_bytes_, prefetch_.boundary_rate());
  if (globals_.final_norm) prefetch_.add(globals_.final_norm, H * 2);
  if (globals_.lm_head) prefetch_bf16(globals_.lm_head, static_cast<size_t>(lm_vocab_count_) * H * 2);
}

// ---------------------------------------------------------------------------
// The row walk.
// ---------------------------------------------------------------------------
// The embedding rows of `tokens` into `dst` (device): the whole table's
// gather, or — vocab-sharded — this rank's rows with zeros elsewhere, the
// ranks' partials summed by one fold (a row plus zeros is exact in bf16,
// so the result is bitwise the replicated lookup). The fold's staged
// buffer, when the recorder hands one out, is pinned host memory: the rows
// are copied back to the device residual afterwards.
void GlmDsaModel::gather_embedding(const int64_t* tokens, uint16_t* dst, int T, bool capture) {
  const int H = cfg_.hidden_size;
  if (!embed_sharded_) {
    embed_gather_bf16(globals_.embed, tokens, dst, T, H, stream_);
    return;
  }
  uint16_t* e = stage(dst, T, H, capture);
  embed_gather_sliced_bf16(globals_.embed, tokens, e, T, H, globals_.embed_vocab_begin, globals_.embed_vocab_count,
                           stream_);
  fold(e, T, H, capture);
  if (e != dst) copy_rows_bf16(dst, e, T, H, stream_);
}

uint16_t* GlmDsaModel::stage(uint16_t* fallback, int T, int width, bool capture) {
  if (!boundary_) return fallback;
  uint16_t* s = boundary_->stage(T, width);
  if (s == nullptr && capture) throw std::runtime_error("run_rows: a capture fold does not fit the recorder's staged buffer");
  return s ? s : fallback;
}

void GlmDsaModel::fold(uint16_t* buf, int T, int width, bool capture) {
  if (!boundary_) return;
  if (!capture) DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  boundary_->reduce(buf, T, width);
}

// The BF16 dense MLP of the first layers: out = down(silu(gate(x)) * up(x)),
// this rank's intermediate slice, no clamps (the swiglu kernel at an
// infinite limit).
void GlmDsaModel::enqueue_dense(const GlmDsaDenseMlpResident& d, const uint16_t* x, int T, uint16_t* out) {
  const int H = cfg_.hidden_size;
  const int I = static_cast<int>(d.local_inter);
  if (!d.gate || !d.up || !d.down || I <= 0) throw std::runtime_error("run_rows: the dense MLP is unbound");
  gemm_.matmul(x, d.gate, dense_g_, T, I, H, DType::BF16, GemmOut::BF16, static_cast<size_t>(H), gemm_ws_,
               gemm_ws_bytes_, stream_);
  gemm_.matmul(x, d.up, dense_u_, T, I, H, DType::BF16, GemmOut::BF16, static_cast<size_t>(H), gemm_ws_,
               gemm_ws_bytes_, stream_);
  swiglu_limit_bf16(dense_g_, dense_u_, dense_act_, static_cast<int64_t>(T) * I,
                    std::numeric_limits<float>::infinity(), stream_);
  gemm_.matmul(dense_act_, d.down, out, T, H, I, DType::BF16, GemmOut::BF16, static_cast<size_t>(I), gemm_ws_,
               gemm_ws_bytes_, stream_);
}

void GlmDsaModel::enqueue_layer(const GlmDsaLayerResident& r, int pool_layer, uint16_t* resid, int T,
                                const WalkRows& rows) {
  const int H = cfg_.hidden_size;
  const float eps = cfg_.rms_norm_eps;
  const int64_t n = static_cast<int64_t>(T) * H;
  build_layer_objects(r);
  // ---- the attention block --------------------------------------------------
  glm_rmsnorm_bf16(resid, r.input_norm, x_, T, H, eps, stream_);
  uint16_t* attn_out = stage(y_, T, H, rows.capture);
  if (!dsa_->prepare(T)) throw std::runtime_error("run_rows: DSA GEMM plans unavailable");
  if (rows.decode) {
    dsa_->enqueue_decode(x_, pool_, pool_layer, rows.req_ids, rows.pos, rows.spans, rows.num_requests, T,
                         attn_out, stream_, &prefetch_, /*tail_snapshots=*/nullptr);
  } else if (rows.num_spans > 0) {
    // The group's spans one by one: each span's rows attend to their own
    // request's cache and publish to it (the DSA layer's scratch is
    // per call; the pointers step by the span's rows).
    int64_t row0 = 0;
    for (int s = 0; s < rows.num_spans; ++s) {
      const int len = rows.span_lens[s];
      if (!dsa_->prepare(len)) throw std::runtime_error("run_rows: DSA GEMM plans unavailable");
      dsa_->enqueue_prefill(x_ + static_cast<size_t>(row0) * H, pool_, pool_layer, rows.span_reqs[s], rows.span_pos0[s],
                            len, attn_out + static_cast<size_t>(row0) * H, stream_, static_cast<int>(row0));
      row0 += len;
    }
  } else {
    dsa_->enqueue_prefill(x_, pool_, pool_layer, rows.req, rows.pos0, T, attn_out, stream_);
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
    enqueue_dense(r.dense, x_, T, ffn_out);
  }
  if (rows.decode) prefetch_attention_side(r.layer + 1);
  fold(ffn_out, T, H, rows.capture);  // block boundary 2: the sliced down projections
  glm_residual_add_bf16(resid, ffn_out, n, stream_);
}

GlmDsaModel::Outputs GlmDsaModel::run_rows(const RowRun& run) {
  const int T = run.T, req = run.req;
  // The packed companions' wide launches are the decode batch's alone (a
  // short prefill chunk keeps its Lt algorithm: kernels/gemm.hpp).
  gemm_.set_bf12_wide(run.decode);
  walk_rows_ = T;
  if (run.capture && loader_.residency() != GlmDsaResidency::Resident)
    throw std::logic_error("run_rows: a capture needs a resident stack");
  const int H = cfg_.hidden_size;
  const RowInputs in = begin_run(run);
  gather_embedding(in.tokens, resid_, T, run.capture);
  Outputs out;
  const bool traces = !run.decode && route_traces_;
  const size_t K = static_cast<size_t>(cfg_.num_experts_per_tok);
  WalkRows rows;
  rows.req_ids = in.req_ids;
  rows.pos = in.pos;
  rows.spans = in.spans;
  rows.num_requests = in.num_requests;
  rows.req = req;
  rows.pos0 = run.pos0;
  rows.decode = run.decode;
  rows.capture = run.capture;
  rows.span_reqs = run.span_reqs;
  rows.span_pos0 = run.span_pos0;
  rows.span_lens = run.span_lens;
  rows.num_spans = run.num_spans;
  // DGPP_GLM_DSA_CAPTURE_DECODE=1: an eager decode walk keeps every layer's
  // rows too (the decode-vs-prefill localizer in glm_dsa_decode_test).
  static const bool capture_decode_env = std::getenv("DGPP_GLM_DSA_CAPTURE_DECODE") != nullptr;
  const bool capture_layers = run.capture_layers || (run.decode && !run.capture && capture_decode_env);
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const GlmDsaLayerResident& r = loader_.load_layer(layer);
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
    if (capture_layers) {
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      std::vector<uint16_t> snap(static_cast<size_t>(T) * H);
      DGPP_CUDA_OK(cudaMemcpy(snap.data(), resid_, snap.size() * 2, cudaMemcpyDeviceToHost));
      out.layer_states.push_back(std::move(snap));
      if (r.attn.owns_indexer()) {  // the layer's selection (its dependants reuse it)
        const size_t ms = static_cast<size_t>(dsa_->geometry().max_selected);
        std::vector<int32_t> sel(static_cast<size_t>(T) * ms);
        DGPP_CUDA_OK(cudaMemcpy(sel.data(), dsa_->debug_topk(), sel.size() * 4, cudaMemcpyDeviceToHost));
        out.dsa_selections.push_back(std::move(sel));
      }
    }
  }
  // The head on every row: a prefill chunk's last row comes off the same
  // m=T GEMM the diagnostic forward runs (the prefill == forward bitwise
  // gate).
  glm_rmsnorm_bf16(resid_, globals_.final_norm, h_, T, H, cfg_.rms_norm_eps, stream_);
  gemm_.matmul(h_, globals_.lm_head, logits_, T, lm_vocab_count_, H, DType::BF16, GemmOut::F32,
               static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
  // The draft block's input: the last rows' POST-final-norm hidden into
  // the slots' windows by position.
  // A group prefill stores every row: each span's last window rows land
  // in its own slot (2026-09-14).
  if (mtp_) {
    const int nrows = run.num_spans > 0 ? T : std::min(T, max_decode_rows_);
    store_draft_hidden(h_ + static_cast<size_t>(T - nrows) * H, in.req_ids + (T - nrows), in.pos + (T - nrows), nrows);
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
GlmDsaModel::Outputs GlmDsaModel::forward(const std::vector<int64_t>& token_ids, bool capture_layers) {
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
void GlmDsaModel::graph_prepare() {
  if (loader_.residency() != GlmDsaResidency::Resident)
    throw std::logic_error("session_graph_prepare: the decode graph needs a resident stack");
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const GlmDsaLayerResident& r = loader_.load_layer(layer);
    build_layer_objects(r);
    for (int rows = 1; rows <= max_decode_rows_; ++rows)
      if (!dsa_->prepare(rows)) throw std::runtime_error("session_graph_prepare: DSA GEMM plans unavailable");
    if (r.moe) moe_->prepare_graph_table(moe_ordinal(layer), stream_);
  }
  if (mtp_) {
    const GlmDsaLayerResident& r = loader_.load_layer(cfg_.mtp_layer());
    build_layer_objects(r);
    for (int rows = 1; rows <= max_decode_rows_; ++rows)
      if (!dsa_->prepare(rows)) throw std::runtime_error("session_graph_prepare: DSA GEMM plans unavailable");
    moe_->prepare_graph_table(cfg_.num_moe_layers(), stream_);
  }
}

// ---------------------------------------------------------------------------
// The MTP draft block: eh_proj over [enorm(embed(tok_{q+1})) | hnorm(h_q)],
// one full layer (its own pool layer and indexer), shared_head.norm, the
// shared head.
// ---------------------------------------------------------------------------
void GlmDsaModel::mtp_run_rows(int req, const int64_t* tokens, int64_t first_pos, int T, bool decode_row,
                               bool capture, int head_rows, int batch_requests) {
  if (!mtp_) throw std::logic_error("mtp_run_rows: MTP is not enabled");
  if (T <= 0 || T > max_tokens_) throw std::invalid_argument("mtp_run_rows: rows");
  if (head_rows < 0 || head_rows > T) throw std::invalid_argument("mtp_run_rows: head_rows");
  gemm_.set_bf12_wide(decode_row);  // the decode batch's alone (kernels/gemm.hpp)
  walk_rows_ = T;
  const int H = cfg_.hidden_size;
  const float eps = cfg_.rms_norm_eps;
  const int64_t* d_pos = decode_row ? d_step_pos_ : d_prefill_pos_;
  const int32_t* d_req = decode_row ? d_req_ids_ : d_prefill_req_;
  const int32_t* d_spans = decode_row ? d_req_spans_ : d_prefill_spans_;
  if (!decode_row) stage_prefill_meta(req, first_pos, T);
  const GlmDsaLayerResident& r = loader_.load_layer(cfg_.mtp_layer());
  if (!r.enorm || !r.hnorm || !r.eh_proj || !r.shared_head_norm)
    throw std::runtime_error("mtp_run_rows: the draft layer's head tensors are unbound");
  // ---- the input fusion --------------------------------------------------
  const uint16_t* hin = h_;
  if (decode_row) {
    gather_draft_hidden(d_req, d_pos, mtp_h_, T);
    hin = mtp_h_;
  }
  const uint16_t* embed_table = globals_.embed;
  const int64_t* embed_ids = tokens;
  if (embed_sharded_) {
    // The draft's rows gathered and folded into the scratch; the fused
    // input kernel then indexes that scratch by row.
    gather_embedding(tokens, mtp_embed_rows_, T, capture);
    embed_table = mtp_embed_rows_;
    embed_ids = d_iota_;
  }
  glm_mtp_input_bf16(embed_table, embed_ids, hin, nullptr, 0, r.enorm, r.hnorm, mtp_in_, T, H, eps, stream_);
  gemm_.matmul(mtp_in_, r.eh_proj, mtp_r_, T, H, 2 * H, DType::BF16, GemmOut::BF16, static_cast<size_t>(2 * H),
               gemm_ws_, gemm_ws_bytes_, stream_);
  // ---- the draft layer (the stack's objects rebound to its weights) ------
  WalkRows rows;
  rows.req_ids = d_req;
  rows.pos = d_pos;
  rows.spans = d_spans;
  rows.num_requests = batch_requests > 0 ? batch_requests : 1;
  rows.req = req;
  rows.pos0 = first_pos;
  rows.decode = decode_row;
  rows.capture = capture;
  rows.moe_table_slot = capture ? cfg_.num_moe_layers() : -1;
  rows.trace = nullptr;
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

GlmDsaModel::Outputs GlmDsaModel::mtp_forward(const std::vector<int64_t>& token_ids) {
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
