#include "models/inkling2/forward.hpp"

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
#include "kernels/inkling2_attn.hpp"
#include "kernels/inkling2_draft.hpp"
#include "kernels/kernels.hpp"
#include "kernels/mma_gemv.hpp"
#include "models/inkling2/draft.hpp"

namespace dgpp {
namespace {

template <class T>
T* dev_alloc_zero(size_t n) {
  T* p = nullptr;
  const size_t bytes = std::max<size_t>(n, 1) * sizeof(T);
  DGPP_CUDA_OK(cudaMalloc(&p, bytes));
  DGPP_CUDA_OK(cudaMemset(p, 0, bytes));
  return p;
}

// The draft model's planes follow the visits': one per draft layer, at this
// rank's share of the draft's key/value heads.
int draft_kv_heads(const Inkling2Config& cfg, int world) { return cfg.draft.num_key_value_heads / std::max(world, 1); }

void check_draft(const Inkling2Config& cfg, int world, int max_tokens) {
  const DFlash2Config d = inkling2_draft_config(cfg);
  if (!d.tp_divisible(world))
    throw std::invalid_argument("engine.mtp: the Inkling 2 draft's heads, key/value heads and MLP rows must divide "
                                "across the world");
  if (max_tokens < d.query_rows())
    throw std::invalid_argument("engine.mtp: max_tokens must cover the draft's block");
  if (d.query_rows() > kInk2RingSlots - 3)
    throw std::invalid_argument("engine.mtp: the draft's block is wider than the convolution ring covers");
}

MimoKvPoolShape pool_shape(const Inkling2Config& cfg, int local_kv_heads, int max_requests, int64_t cache_tokens,
                           LatentFormat kv_format, bool mtp, int world) {
  MimoKvPoolShape shape;
  // One plane per visit: a core layer's passes keep separate keys and values.
  shape.kv_heads.assign(static_cast<size_t>(cfg.num_visits()), local_kv_heads);
  if (mtp)
    shape.kv_heads.insert(shape.kv_heads.end(), static_cast<size_t>(cfg.draft.num_hidden_layers),
                          draft_kv_heads(cfg, world));
  // The reference readout's visits (logit guidance) follow, so the planes
  // before them sit where they do without guidance.
  shape.kv_heads.insert(shape.kv_heads.end(), static_cast<size_t>(cfg.reference_visits()), local_kv_heads);
  shape.k_dim = kInk2HeadDim;
  shape.v_dim = kInk2HeadDim;
  shape.block_tokens = 64;
  shape.max_requests = max_requests;
  shape.token_slots = cache_tokens;
  shape.format = kv_format;
  return shape;
}

int64_t ring_visit_width(const Inkling2Config& cfg, int local_kv_heads) {
  return Inkling2Ring::visit_width(static_cast<int64_t>(local_kv_heads) * kInk2HeadDim, cfg.hidden_size);
}

// A ring slot: every visit's convolution inputs, then the draft model's
// feature row of that position, then the reference readout's visits.
int64_t ring_slot_stride(const Inkling2Config& cfg, int local_kv_heads, bool mtp) {
  return ring_visit_width(cfg, local_kv_heads) * (cfg.num_visits() + cfg.reference_visits()) +
         (mtp ? cfg.hidden_size : 0);
}

size_t snapshot_bytes_of(const Inkling2Config& cfg, int local_kv_heads, bool mtp) {
  const size_t raw = ink2_ring_snapshot_bytes(ring_slot_stride(cfg, local_kv_heads, mtp));
  return (raw + 15) / 16 * 16;
}

size_t activation_bytes(const Inkling2Config& cfg, int max_tokens) {
  const size_t M = static_cast<size_t>(max_tokens), H = static_cast<size_t>(cfg.hidden_size);
  const size_t planes = static_cast<size_t>(cfg.max_blocks() + 1 + cfg.bank_rows());
  size_t bytes = planes * M * H * 2      // the stack
                 + 6 * M * H * 2         // the normed inputs, the sublayer output, the attention contribution
                 + 7 * M * H * 4         // the FP32 mixes and messages
                 + M * 2 * H * 2 + M * static_cast<size_t>(cfg.gate_rank) * 2 + M * 2 * H * 4;  // the controllers
  // Loop guidance: the first pass's states and the readout's hidden rows
  // (plan_memory adds its logits).
  if (cfg.loop_guidance) bytes += static_cast<size_t>(cfg.bank_rows() + 1) * M * H * 2;
  return bytes;
}

}  // namespace

Inkling2Model::Inkling2Model(const Inkling2Config& cfg, const std::string& checkpoint_dir, int max_tokens,
                             int64_t max_cache_tokens, Inkling2Residency residency, BoundaryReducer* boundary,
                             int tp_rank, int tp_world, int max_requests, bool mtp, int decode_rows,
                             LatentFormat kv_format, bool serving_logits, bool native_activations)
    : cfg_(cfg),
      loader_(cfg, checkpoint_dir, tp_rank, tp_world, residency,
              tp_world > 1 ? Inkling2HeadSharding::VocabSharded : Inkling2HeadSharding::Full),
      native_(native_activations) {
  if (mtp) check_draft(cfg_, tp_world, max_tokens);
  if (decode_rows > decode_rows_cap())
    throw std::invalid_argument("Inkling2Model: decode_rows exceeds the family's limit of 64");
  if (max_tokens <= 0) throw std::invalid_argument("Inkling2Model: max_tokens must be positive");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("Inkling2Model: max_requests must be in [1, kPickMaxRequests]");
  if ((tp_world > 1) != (boundary != nullptr))
    throw std::invalid_argument(
        "Inkling2Model: a boundary reducer is required exactly when tp_world > 1, or the sublayer partials "
        "would be returned as results");
  if (cfg_.eos_token_ids.empty()) throw std::invalid_argument("Inkling2Model: the config names no EOS token");
  if (kv_format != LatentFormat::kBf16)
    throw std::invalid_argument("Inkling2Model: the K/V cache format must be bf16");
  if (cfg_.max_blocks() + 1 > kInk2MaxSources)
    throw std::invalid_argument("Inkling2Model: more blocks than a mix holds");
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
    sp.snapshot_align = 1;
    sp.draft_width = H;
    sp.eos = static_cast<int32_t>(cfg_.eos_token_ids[0]);
    init_session(sp);
  }
  gemm_ws_bytes_ =
      std::max<size_t>(64u << 20, gemm_.query_workspace_bytes(max_tokens_, lm_vocab_count_, H, DType::BF16));
  DGPP_CUDA_OK(cudaMalloc(&gemm_ws_, gemm_ws_bytes_));
  gw_ = Inkling2GemmWorkspace{&gemm_, gemm_ws_, gemm_ws_bytes_};
  // One request's pass holds up to eight rows, the pending token and the
  // draft's seven proposals, and a verify row must round as the plain step
  // at its position does. Every BF16 product of up to eight rows therefore
  // takes the row-independent GEMV chain, the matrices without a 12-bit
  // companion included (the controllers' gain product).
  gemm_.set_decode_rows(std::min(max_decode_rows_, kRequestRows));

  const Inkling2LocalGeometry& geo = loader_.geometry();
  visits_ = cfg_.num_visits();
  bank_rows_ = cfg_.bank_rows();
  max_blocks_ = cfg_.max_blocks();
  ref_visits_ = cfg_.reference_visits();
  draft_planes_ = mtp ? cfg_.draft.num_hidden_layers : 0;
  if (cfg_.loop_guidance) keep_visit_ = cfg_.visit_of(cfg_.core_end - 1, 0);
  pool_.init(pool_shape(cfg_, geo.local_kv_heads, max_requests_, max_cache_tokens_, kv_format, mtp, tp_world));
  ring_visit_width_ = ring_visit_width(cfg_, geo.local_kv_heads);
  ring_draft_offset_ = ring_visit_width_ * visits_;
  ring_slot_stride_ = ring_slot_stride(cfg_, geo.local_kv_heads, mtp);
  ring_ = dev_alloc_zero<uint16_t>(static_cast<size_t>(max_requests_) * kInk2RingSlots *
                                   static_cast<size_t>(ring_slot_stride_));
  snapshot_bytes_ = snapshot_bytes_of(cfg_, geo.local_kv_heads, mtp);

  // Zeroed: a padding row of a fixed batch reads its own untouched rows.
  const size_t M = static_cast<size_t>(max_tokens_), Hs = static_cast<size_t>(H);
  stack_ = dev_alloc_zero<uint16_t>(static_cast<size_t>(max_blocks_ + 1 + bank_rows_) * M * Hs);
  if (cfg_.loop_guidance) {
    reference_ = dev_alloc_zero<uint16_t>(static_cast<size_t>(bank_rows_) * M * Hs);
    reference_hidden_ = dev_alloc_zero<uint16_t>(M * Hs);
    reference_logits_ =
        dev_alloc_zero<float>(static_cast<size_t>(kReferenceLogitRows) * static_cast<size_t>(lm_vocab_count_));
    // Every slot starts at the deployment's strength.
    guidance_strength_host_.assign(static_cast<size_t>(max_requests_), cfg_.loop_guidance_strength);
    guidance_strength_ = dev_alloc_zero<float>(static_cast<size_t>(max_requests_));
    DGPP_CUDA_OK(cudaMemcpy(guidance_strength_, guidance_strength_host_.data(),
                            guidance_strength_host_.size() * sizeof(float), cudaMemcpyHostToDevice));
  }
  for (auto& x : x_) x = dev_alloc_zero<uint16_t>(M * Hs);
  xm_ = dev_alloc_zero<uint16_t>(M * Hs);
  y_ = dev_alloc_zero<uint16_t>(M * Hs);
  a_ = dev_alloc_zero<uint16_t>(M * Hs);
  for (auto& d : dest_) d = dev_alloc_zero<float>(M * Hs);
  for (auto& m : msg_) m = dev_alloc_zero<float>(M * Hs);
  pair_ = dev_alloc_zero<uint16_t>(M * 2 * Hs);
  ctrl_h_ = dev_alloc_zero<uint16_t>(M * static_cast<size_t>(cfg_.gate_rank));
  gain_forget_ = dev_alloc_zero<float>(M * 2 * Hs);

  attn_ = std::make_unique<Inkling2AttentionLayer>(cfg_, geo, gw_, max_tokens_, max_decode_rows_, n_split_);
  moe_ = std::make_unique<Inkling2MoeLayer>(cfg_, geo, max_tokens_, native_, residency == Inkling2Residency::Resident);
  if (mtp) init_drafter(checkpoint_dir, tp_rank, tp_world);
  // The vision tower is replicated on every rank and reserves its buffers here.
  request_images_.assign(static_cast<size_t>(max_requests_), nullptr);
  if (cfg_.vision.present) vision_ = std::make_unique<Inkling2VisionEncoder>(cfg_, checkpoint_dir, stream_);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}

// The draft model on the shared block drafter: its planes are the pool's
// last ones, its head is the target's unembedding without the cap (the
// recurrent head adds its bias before the cap), and the feature its
// recurrent head reads lives in the ring slot of each position.
void Inkling2Model::init_drafter(const std::string& checkpoint_dir, int tp_rank, int tp_world) {
  const DFlash2Config dcfg = inkling2_draft_config(cfg_);
  const int H = cfg_.hidden_size;
  DFlash2Host host;
  host.hidden = H;
  // The taps name visits, and the visit count names the final-norm output.
  host.num_target_layers = visits_ + 1;
  host.target_vocab = cfg_.vocab_size;
  host.vocab_count = lm_vocab_count_;
  host.vocab_begin = lm_vocab_begin_;
  host.rank = tp_rank;
  host.world = tp_world;
  host.stream = stream_;
  host.max_requests = max_requests_;
  host.max_tokens = max_tokens_;
  host.max_decode_rows = max_decode_rows_;
  host.max_context = static_cast<int>(max_context_);
  host.position_limit = cfg_.max_position_embeddings;
  host.session_pos = d_session_pos_;
  host.decode_tokens = d_tokens_;
  const MimoKvCache first = pool_.view(0);
  host.device_tables = first.block_tables;
  host.total_blocks = first.blocks_per_request;
  host.block_tokens = kBlockTokens;
  host.ensure_request_blocks = [this](int req, int64_t pos, cudaStream_t st) {
    return pool_.ensure_request_blocks(req, pos, st);
  };
  host.plane = [this](int l) {
    const MimoKvCache v = pool_.view(visits_ + l);
    return DFlash2Host::Plane{v.k_cache, v.v_cache};
  };
  host.session_position = [this](int req) { return session_position(req); };
  host.embed = globals_.embed;
  // The draft's projections take the tensor-core form on their own GEMM
  // object: a matrix is read once for all the rows of a pass and a row's
  // sums do not depend on the slots stacked beside it. cuBLASLt changes its
  // algorithm with the row count and leaves memset nodes in a wider
  // capture. The two objects share the workspace: their calls are serial
  // on one stream.
  host.gemm = &draft_gemm_;
  host.gemm_ws = gemm_ws_;
  host.gemm_ws_bytes = gemm_ws_bytes_;
  host.gemm_rows = [this](int, bool decode) { draft_gemm_.set_decode_mma(decode); };
  host.boundary = [this] { return boundary_; };
  // The mask rows' base logits are the exact head's. The recurrent head's
  // correction reorders the whole vocabulary, so a shortlist taken from the
  // base logits (engine.draft_head screen) drops tokens it would have
  // chosen: the screen is not read here.
  host.head = [this](const uint16_t* hidden, float* out, int rows) {
    const int H = cfg_.hidden_size;
    // One block's rows fit the packed head's launch, at three quarters of
    // the bytes. Several slots' rows take the tensor-core form, one read for
    // every row: over the packed head up to its 32 rows, over the BF16
    // bytes past that when those are resident. With the packed head alone
    // (engine.bf16_weights bf12) the widest blocks take its launches, one
    // read for every eight rows.
    if (packed_head_wide(hidden, out, rows)) return;
    const bool packed = gemm_.bf12_lookup(globals_.unembed) != nullptr;
    const bool released = gemm_.bf12_raw_released(globals_.unembed);
    if (packed && (released || rows <= kBf12MaxRows)) {
      const bool wide = gemm_.decode_mma();
      if (wide) gemm_.set_decode_mma(false);
      gemm_.set_bf12_wide(true);
      gemm_.matmul(hidden, globals_.unembed, out, rows, lm_vocab_count_, H, DType::BF16, GemmOut::F32,
                   static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
      if (wide) gemm_.set_decode_mma(true, kBf12MaxRows + 1, 0);
      return;
    }
    draft_gemm_.set_decode_mma(true);
    draft_gemm_.matmul(hidden, globals_.unembed, out, rows, lm_vocab_count_, H, DType::BF16, GemmOut::F32,
                       static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
  };
  host.store_step_feature = [this](const uint16_t* rows, int count, const int32_t* d_req, const int64_t* d_pos) {
    ink2_ring_store(rows, cfg_.hidden_size, 0, ring_, static_cast<int64_t>(kInk2RingSlots) * ring_slot_stride_,
                    ring_slot_stride_, ring_draft_offset_, d_req, d_pos, count, cfg_.hidden_size, stream_);
  };
  host.load_step_feature = [this](int req, const int64_t* d_block_pos, uint16_t* d_out) {
    ink2_draft_ring_row(ring_ + static_cast<size_t>(req) * kInk2RingSlots * static_cast<size_t>(ring_slot_stride_),
                        ring_slot_stride_, ring_draft_offset_, kInk2RingSlots, d_block_pos, d_out, cfg_.hidden_size,
                        stream_);
  };
  Inkling2DraftTarget target;
  target.unembed = globals_.unembed;
  target.vocab_count = lm_vocab_count_;
  target.embed_norm = globals_.embed_norm;
  target.padded_vocab = cfg_.padded_vocab_size;
  df_.init(host, dcfg, load_inkling2_draft(dcfg, checkpoint_dir, stream_, tp_rank, tp_world, target));
}

DFlash2CaptureCtx Inkling2Model::df_capture_ctx() const {
  return {step_tokens_,
          decode_rows_,
          graph_feed_rows_,
          graph_batch_requests_,
          graph_rows_per_request_,
          graph_batch_map_source_,
          graph_device_positions_,
          graph_device_tokens_};
}

Inkling2Model::~Inkling2Model() {
  df_.free();
  cudaFree(ring_);
  cudaFree(stack_);
  cudaFree(reference_);
  cudaFree(reference_hidden_);
  cudaFree(reference_logits_);
  cudaFree(guidance_strength_);
  for (auto* x : x_) cudaFree(x);
  cudaFree(xm_);
  cudaFree(y_);
  cudaFree(a_);
  for (auto* d : dest_) cudaFree(d);
  for (auto* m : msg_) cudaFree(m);
  cudaFree(pair_);
  cudaFree(ctrl_h_);
  cudaFree(gain_forget_);
  cudaFree(gemm_ws_);
}

size_t Inkling2Model::session_snapshot_bytes(const Inkling2Config& cfg, int tp_world, bool mtp) {
  return snapshot_bytes_of(cfg, cfg.num_key_value_heads / std::max(tp_world, 1), mtp);
}

size_t Inkling2Model::kv_block_bytes_static(const Inkling2Config& cfg, int tp_rank, int tp_world,
                                            LatentFormat kv_format, bool mtp) {
  const Inkling2HeadSharding head = tp_world > 1 ? Inkling2HeadSharding::VocabSharded : Inkling2HeadSharding::Full;
  const Inkling2LocalGeometry geo = Inkling2LocalGeometry::from_config(cfg, tp_rank, tp_world, head);
  const auto bytes = [&](int64_t tokens) {
    const MimoKvPoolShape shape = pool_shape(cfg, geo.local_kv_heads, 1, tokens, kv_format, mtp, tp_world);
    return MimoKvPool::cache_bytes(shape) - PagedBlockTable::table_bytes(1, tokens / kBlockTokens);
  };
  return bytes(2 * kBlockTokens) - bytes(kBlockTokens);
}

Inkling2Model::MemoryPlan Inkling2Model::plan_memory(const Inkling2Config& cfg, int max_tokens,
                                                     int64_t max_cache_tokens, int tp_rank, int tp_world,
                                                     Inkling2Residency residency, int max_requests, bool mtp,
                                                     int decode_rows, LatentFormat kv_format, bool serving_logits,
                                                     bool native_activations) {
  if (max_tokens <= 0) throw std::invalid_argument("plan_memory: max_tokens must be positive");
  if (mtp) check_draft(cfg, tp_world, max_tokens);
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("plan_memory: max_requests must be in [1, kPickMaxRequests]");
  if (decode_rows > decode_rows_cap())
    throw std::invalid_argument("plan_memory: decode_rows exceeds the family's limit of 64");
  if (kv_format != LatentFormat::kBf16) throw std::invalid_argument("plan_memory: the K/V cache format must be bf16");
  const int rows = std::max({kDecodeRows, decode_rows, max_requests});
  const Inkling2HeadSharding head = tp_world > 1 ? Inkling2HeadSharding::VocabSharded : Inkling2HeadSharding::Full;
  const Inkling2LocalGeometry geo = Inkling2LocalGeometry::from_config(cfg, tp_rank, tp_world, head);
  const int64_t cache_tokens =
      ((std::max<int64_t>(max_cache_tokens, max_tokens) + kBlockTokens - 1) / kBlockTokens) * kBlockTokens;
  MemoryPlan plan;
  plan.context_tokens = std::min<int64_t>(cache_tokens, cfg.max_position_embeddings);
  const size_t V = static_cast<size_t>(Inkling2LayerStream::lm_vocab_count(cfg, tp_rank, tp_world, head));
  const int n_split = Inkling2AttentionLayer::default_decode_splits();

  const bool bf12_only = Bf12Companions::packed_only() && residency == Inkling2Residency::Resident;
  if (residency == Inkling2Residency::Resident) {
    size_t resident = Inkling2LayerStream::resident_bytes(cfg, tp_rank, tp_world, head, false);
    if (bf12_only) resident -= Inkling2LayerStream::side_bytes(cfg, tp_rank, tp_world, head, false);
    plan.add(bf12_only ? "model weights (resident, packed bf16 matrices released)" : "model weights (resident)",
             resident);
    plan.add("loader staging (pinned host, freed when the last layer is resident)", 0,
             Inkling2LayerStream::staging_plan_bytes(cfg, tp_rank, tp_world, head, false));
  } else {
    size_t largest = 0;
    for (int l = 0; l < cfg.num_hidden_layers; ++l)
      largest = std::max(largest, Inkling2LayerStream::layer_bytes(cfg, l, tp_rank, tp_world));
    plan.add("model weights (one streamed layer + globals)",
             largest + Inkling2LayerStream::globals_bytes(cfg, tp_rank, tp_world, head));
  }
  const int64_t H = cfg.hidden_size;
  const int64_t q_side = static_cast<int64_t>(geo.local_heads) * (2 * kInk2HeadDim + cfg.d_rel + 1);
  const int64_t O = static_cast<int64_t>(geo.local_heads) * kInk2HeadDim;
  if (Bf12Companions::enabled() && residency == Inkling2Residency::Resident) {
    size_t packed = Bf12Companions::planned_bytes(static_cast<int64_t>(V), H);
    size_t widest_kv = 0;
    for (int l = 0; l < cfg.num_hidden_layers; ++l) {
      const int64_t kv = static_cast<int64_t>(geo.local_kv_heads) *
                         (kInk2HeadDim + (cfg.is_local_layer(l) ? 0 : cfg.sconv_kernel_size));
      widest_kv = std::max(widest_kv, static_cast<size_t>(kv));
      packed += Bf12Companions::planned_bytes(q_side, H) + 2 * Bf12Companions::planned_bytes(kv, H) +
                Bf12Companions::planned_bytes(H, O);
      if (cfg.is_core_layer(l)) packed += 4 * Bf12Companions::planned_bytes(cfg.gate_rank, 2 * H);
    }
    plan.add("bf16 decode packing (12-bit companions)", packed);
    if (bf12_only) {
      const size_t h = static_cast<size_t>(H);
      plan.add("bf16 prefill expansion scratch",
               static_cast<size_t>(kBf12ExpandSlots) *
                   Bf12Companions::planned_slot_bytes({static_cast<size_t>(q_side) * h * 2, widest_kv * h * 2,
                                                       h * static_cast<size_t>(O) * 2,
                                                       static_cast<size_t>(cfg.gate_rank) * 2 * h * 2, V * h * 2}));
    }
  }
  plan.add("gemm workspace (at least)", size_t{64} << 20);
  {
    const MimoKvPoolShape shape =
        pool_shape(cfg, geo.local_kv_heads, max_requests, cache_tokens, kv_format, mtp, tp_world);
    plan.add(mtp ? "kv cache pool (K/V bf16, one plane per visit and per draft layer)"
                 : "kv cache pool (K/V bf16, one plane per visit)",
             MimoKvPool::cache_bytes(shape));
  }
  plan.add(mtp ? "convolution rings (with the draft's feature rows)" : "convolution rings",
           static_cast<size_t>(max_requests) * kInk2RingSlots *
               static_cast<size_t>(ring_slot_stride(cfg, geo.local_kv_heads, mtp)) * 2);
  {
    size_t core_dev = 0, core_pin = 0;
    session_core_plan_bytes(max_tokens, max_requests, cfg.hidden_size, static_cast<int>(V), false, cfg.hidden_size,
                            &core_dev, &core_pin, rows, serving_logits);
    plan.add("activations (session core, block stack, sublayer io, controllers)",
             core_dev + activation_bytes(cfg, max_tokens), core_pin);
  }
  if (cfg.loop_guidance)
    plan.add("loop guidance (the first-pass readout's logits, the requests' strengths)",
             static_cast<size_t>(kReferenceLogitRows) * V * 4 + static_cast<size_t>(max_requests) * 4);
  plan.add("layer objects (attention)",
           Inkling2AttentionLayer::scratch_bytes(cfg, geo, max_tokens, rows, n_split));
  plan.add("moe scratch (routes, expert io, view tables)",
           Inkling2MoeLayer::scratch_bytes(cfg, geo, max_tokens, native_activations));
  if (mtp) {
    const DFlash2Config dcfg = inkling2_draft_config(cfg);
    std::vector<std::pair<std::string, size_t>> entries;
    DFlash2Drafter::plan_entries(
        dcfg, tp_world, decode_rows, max_tokens, max_requests, static_cast<int64_t>(V), false, &entries,
        inkling2_draft_weights_bytes(dcfg, tp_world, static_cast<int64_t>(V), cfg.padded_vocab_size));
    for (const auto& [name, bytes] : entries) plan.add(name, bytes);
  }
  if (cfg.vision.present) {
    plan.add("vision tower weights (bf16, replicated)", Inkling2VisionEncoder::weight_bytes(cfg));
    plan.add("vision workspace (patches, attention tiles, staged image rows)",
             Inkling2VisionEncoder::workspace_bytes(cfg.vision));
  }
  return plan;
}

// ---------------------------------------------------------------------------
// State.
// ---------------------------------------------------------------------------
void Inkling2Model::reset_slot_state(int req) { pool_.release_request_blocks(req, stream_); }

void Inkling2Model::write_state_snapshot(int req, uint8_t* dst, int spec_row) {
  // The live position, or the position after the step's row `spec_row`.
  const int64_t pos = session_pos_[static_cast<size_t>(req)] - (spec_row < 0 ? 0 : rows_after_for_snapshot_);
  ink2_ring_snapshot(ring_ + static_cast<size_t>(req) * kInk2RingSlots * static_cast<size_t>(ring_slot_stride_),
                     ring_slot_stride_, pos, dst, stream_);
}

void Inkling2Model::read_state_snapshot(int req, const uint8_t* src) {
  ink2_ring_restore(src, ring_slot_stride_,
                    ring_ + static_cast<size_t>(req) * kInk2RingSlots * static_cast<size_t>(ring_slot_stride_),
                    stream_);
}

Inkling2Ring Inkling2Model::ring_of(int visit) const {
  Inkling2Ring ring;
  ring.base = ring_;
  ring.request_stride = static_cast<int64_t>(kInk2RingSlots) * ring_slot_stride_;
  ring.slot_stride = ring_slot_stride_;
  // The reference readout's entries follow the draft's feature row.
  ring.visit_offset = static_cast<int64_t>(visit) * ring_visit_width_ +
                      (visit >= visits_ && draft_planes_ > 0 ? cfg_.hidden_size : 0);
  ring.kv_width = static_cast<int64_t>(loader_.geometry().local_kv_heads) * kInk2HeadDim;
  ring.hidden = cfg_.hidden_size;
  return ring;
}

// ---------------------------------------------------------------------------
// The visit walk.
// ---------------------------------------------------------------------------
uint16_t* Inkling2Model::slot(int n, int parity) const {
  const size_t plane = static_cast<size_t>(max_tokens_) * static_cast<size_t>(cfg_.hidden_size);
  const int core0 = cfg_.core_start / cfg_.block_layers + 1;  // the first index a pass writes
  if (parity != 0 && n >= core0 && n < core0 + bank_rows_)
    return stack_ + static_cast<size_t>(max_blocks_ + 1 + (n - core0)) * plane;
  return stack_ + static_cast<size_t>(n) * plane;
}

uint16_t* Inkling2Model::stage(uint16_t* fallback, int T, int width, bool capture) {
  if (!boundary_) return fallback;
  uint16_t* s = boundary_->stage(T, width);
  if (s == nullptr && capture)
    throw std::runtime_error("run_rows: a capture fold does not fit the recorder's staged buffer");
  return s ? s : fallback;
}

void Inkling2Model::fold(uint16_t* buf, int T, int width, bool capture) {
  if (!boundary_) return;
  if (!capture) DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  boundary_->reduce(buf, T, width);
}

// The dense projections' form for a group of rows. Decode rows up to the
// companions' launch take the packed launches, each row on the scalar chain.
// A wider decode pass (the draft model's verify rows of several requests)
// takes the tensor-core form when the BF16 bytes are resident: a matrix is
// read once for all its rows, where the packed launches read it again every
// eight. Prefill rows keep cuBLASLt's algorithm.
void Inkling2Model::decode_gemm(bool decode) {
  gemm_.set_bf12_wide(decode);
  gemm_.set_decode_mma(decode && !Bf12Companions::packed_only(), kBf12MaxRows + 1, 0);
}

// The vocabulary head over 9 to 32 decode rows (several requests' verify
// rows, several slots' draft blocks): the tensor-core form over the packed
// head, one read at three quarters of the bytes, each row's chain the BF16
// tensor-core launch's. The head is the one matrix here large enough for
// that form to pay (kernels/mma_gemv.hpp). False when it does not apply and
// nothing was launched.
bool Inkling2Model::packed_head_wide(const uint16_t* hidden, float* out, int rows) {
  if (rows <= kBf12MaxRows || rows > kMmaGemvMaxRows) return false;
  const Bf12Matrix* packed = gemm_.bf12_lookup(globals_.unembed);
  if (packed == nullptr) return false;
  const int H = cfg_.hidden_size;
  return launch_mma_gemv_bf12_f32(hidden, static_cast<size_t>(H), *packed, out, rows, lm_vocab_count_, H,
                                  static_cast<size_t>(lm_vocab_count_), stream_);
}

// The decoder prefix is its own group and short readers stay apart, so a
// row's projection recipe does not depend on the rows beside it.
template <class Fn>
void Inkling2Model::groups(const Walk& w, Fn&& fn) const {
  w.run->for_each_kernel_group([](int count, int next) { return count > 128 && next > 128; },
                               [&](int row0, int count, bool decode, int) { fn(row0, count, decode); });
}

// out = bf16(rmsnorm((1 - tanh(forget(h))) * dest + tanh(gain(h)) * message) * w),
// h = bf16(bf16([rms(dest), rms(message)]) @ down^T).
void Inkling2Model::blend(const float* dest, const float* message, const Inkling2ControllerResident& c,
                          const uint16_t* norm_w, uint16_t* out, int rows) {
  const int H = cfg_.hidden_size, R = cfg_.gate_rank;
  ink2_pair_norm(dest, message, pair_, rows, H, stream_);
  gemm_.matmul(pair_, c.down, ctrl_h_, rows, R, 2 * H, DType::BF16, GemmOut::BF16, static_cast<size_t>(2 * H),
               gemm_ws_, gemm_ws_bytes_, stream_);
  // The gains' product is a gate-rank wide: the tensor-core form's launch
  // costs more than the matrix, so a wide pass keeps cuBLASLt's kernel here.
  const bool wide = gemm_.decode_mma();
  if (wide) gemm_.set_decode_mma(false);
  gemm_.matmul(ctrl_h_, c.gain_forget, gain_forget_, rows, 2 * H, R, DType::BF16, GemmOut::F32,
               static_cast<size_t>(R), gemm_ws_, gemm_ws_bytes_, stream_);
  if (wide) gemm_.set_decode_mma(true, kBf12MaxRows + 1, 0);
  ink2_blend_norm(dest, message, gain_forget_, norm_w, out, rows, H, cfg_.rms_norm_eps, stream_);
}

void Inkling2Model::visit(int v, Walk& w, Outputs* capture_out) {
  const Inkling2Visit vi = cfg_.visit(v);
  const int l = vi.layer, p = vi.pass;
  const int H = cfg_.hidden_size, T = w.T;
  const float eps = cfg_.rms_norm_eps;
  const int W = cfg_.sconv_kernel_size;
  const float conv_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(W + 1)));
  const Inkling2LayerResident& r = loader_.load_layer(l);
  moe_->bind(l, r.moe, stream_);
  const bool core = cfg_.is_core_layer(l);
  const bool first = cfg_.first_of_block(l);
  const bool messages = p > 0;
  // Every pass restarts from the state at the core's entry.
  if (l == cfg_.core_start) {
    if (p == 0) w.prelude_blocks = w.num_blocks;
    else w.num_blocks = w.prelude_blocks;
  }
  w.parity = core ? (p & 1) : (l < cfg_.core_start ? 0 : ((cfg_.num_passes - 1) & 1));
  const int nb = w.num_blocks;
  const uint16_t* src[kInk2MaxSources] = {};
  for (int n = 0; n < nb; ++n) src[n] = slot(n, w.parity);
  uint16_t* partial = slot(nb, w.parity);
  // The previous pass's states: the blocks it completed after its first,
  // then its partial.
  const int core0 = cfg_.core_start / cfg_.block_layers + 1;
  const int bank_parity = (p - 1) & 1;
  const Inkling2Ring ring = ring_of(v);
  const int nq = r.attn_query_count;
  // The draft model reads the partial state at this visit's entry. It reads
  // the model's own walk, never the reference readout.
  if (df_.on() && !w.reference) {
    const int tap = df_.tap(v);
    if (tap >= 0) copy_rows_bf16(df_.tap_rows(tap), partial, T, H, stream_);
  }

  // ---- the attention ----------------------------------------------------------
  uint16_t* attn_out = stage(y_, T, H, w.capture);
  groups(w, [&](int row0, int count, bool decode) {
    const size_t off = static_cast<size_t>(row0) * H;
    decode_gemm(decode);
    const uint16_t* xq = x_[0] + off;
    const uint16_t* xk = xq;
    const uint16_t* xv = xq;
    if (l == 0) {
      // Nothing to mix yet: the partial state through the layer's norms.
      const int norms = r.local ? 1 : 3;
      for (int j = 0; j < norms; ++j)
        add_rmsnorm_bf16_one(partial + off, nullptr, r.attn_norm[j], x_[j] + off, count, H, eps, stream_);
      if (!r.local) {
        xk = x_[1] + off;
        xv = x_[2] + off;
      }
    } else {
      Ink2MixArgs m;
      for (int n = 0; n < nb; ++n) m.stack[n] = src[n] + off;
      m.n_stack = nb;
      m.last = partial + off;
      m.queries = r.attn_queries;
      m.n_queries = nq;
      m.norm_eps = eps;
      m.rows = count;
      m.hidden = H;
      if (!messages) {
        for (int j = 0; j < nq; ++j) {
          m.norm_w[j] = r.attn_norm[j];
          m.out_bf16[j] = x_[j] + off;
        }
        ink2_mix(m, stream_);
        if (nq == 3) {
          xk = x_[1] + off;
          xv = x_[2] + off;
        }
      } else {
        for (int j = 0; j < nq; ++j) m.out_f32[j] = dest_[j] + off;
        ink2_mix(m, stream_);
        Ink2MixArgs b;
        for (int j = 0; j + 1 < bank_rows_; ++j) b.stack[j] = slot(core0 + j, bank_parity) + off;
        b.n_stack = bank_rows_ - 1;
        b.last = slot(core0 + bank_rows_ - 1, bank_parity) + off;
        b.queries = globals_.message_queries + static_cast<size_t>(l - cfg_.core_start) * 4 * H;
        b.n_queries = 4;
        b.rows = count;
        b.hidden = H;
        for (int s = 0; s < 4; ++s) b.out_f32[s] = msg_[s] + off;
        ink2_mix(b, stream_);
        // A local layer's one mix is the destination of all three messages.
        for (int s = 0; s < 3; ++s)
          blend(dest_[nq == 3 ? s : 0] + off, msg_[s] + off, r.ctrl[s], r.attn_norm[s], x_[s] + off, count);
        xk = x_[1] + off;
        xv = x_[2] + off;
      }
    }
    Inkling2AttnRows rows;
    rows.req_ids = w.req_ids + row0;
    rows.pos = w.pos + row0;
    rows.decode = decode;
    attn_->enqueue(r.attn, xq, xk, xv, count, rows, pool_.view(plane_of(v)), ring, attn_out + off, stream_);
  });
  fold(attn_out, T, H, w.capture);

  // ---- the attention convolution, the MLP input, the experts ----------------------
  // A block's first layer pushes the partial state as a completed block.
  const int nb2 = first ? nb + 1 : nb;
  if (first) src[nb] = partial;
  uint16_t* new_partial = first ? slot(nb + 1, w.parity) : partial;
  uint16_t* ffn_out = stage(y_, T, H, w.capture);
  const auto conv_args = [&](const uint16_t* x, const float* taps, int64_t ring_offset, int row0, int count,
                             bool decode) {
    Ink2ConvArgs c;
    c.x = x + static_cast<size_t>(row0) * H;
    c.x_stride = H;
    c.taps = taps;
    c.scale = conv_scale;
    c.ring = ring.base;
    c.ring_request_stride = ring.request_stride;
    c.ring_slot_stride = ring.slot_stride;
    c.ring_offset = ring_offset;
    c.req_ids = w.req_ids + row0;
    c.pos = w.pos + row0;
    c.rows = count;
    c.channels = H;
    c.store_inline = decode;
    c.out_stride = H;
    return c;
  };
  const auto store = [&](const uint16_t* x, int64_t ring_offset, int row0, int count) {
    ink2_ring_store(x + static_cast<size_t>(row0) * H, H, 0, ring.base, ring.request_stride, ring.slot_stride,
                    ring_offset, w.req_ids + row0, w.pos + row0, count, H, stream_);
  };
  groups(w, [&](int row0, int count, bool decode) {
    const size_t off = static_cast<size_t>(row0) * H;
    decode_gemm(decode);
    Ink2ConvArgs c = conv_args(attn_out, r.attn_conv, ring.attn_offset(), row0, count, decode);
    c.out = a_ + off;
    ink2_conv(c, stream_);
    if (!decode) store(attn_out, ring.attn_offset(), row0, count);
    Ink2MixArgs m;
    for (int n = 0; n < nb2; ++n) m.stack[n] = src[n] + off;
    m.n_stack = nb2;
    m.last = first ? a_ + off : partial + off;
    m.addend = first ? nullptr : a_ + off;
    m.queries = r.mlp_query;
    m.n_queries = 1;
    m.norm_eps = eps;
    m.rows = count;
    m.hidden = H;
    if (!messages) {
      m.norm_w[0] = r.mlp_norm;
      m.out_bf16[0] = xm_ + off;
      ink2_mix(m, stream_);
    } else {
      m.out_f32[0] = dest_[0] + off;
      ink2_mix(m, stream_);
      blend(dest_[0] + off, msg_[kInk2StreamMlp] + off, r.ctrl[kInk2StreamMlp], r.mlp_norm, xm_ + off, count);
    }
    moe_->enqueue(l, r.moe, xm_ + off, count, decode, ffn_out + off, stream_, decode ? w.pos + row0 : nullptr);
  });
  fold(ffn_out, T, H, w.capture);

  // ---- the MLP convolution and the new partial state ---------------------------------
  groups(w, [&](int row0, int count, bool decode) {
    const size_t off = static_cast<size_t>(row0) * H;
    Ink2ConvArgs c = conv_args(ffn_out, r.mlp_conv, ring.mlp_offset(), row0, count, decode);
    c.out = new_partial + off;
    c.addend = a_ + off;
    c.residual = first ? nullptr : partial + off;
    ink2_conv(c, stream_);
    if (!decode) store(ffn_out, ring.mlp_offset(), row0, count);
  });
  if (first) w.num_blocks = nb + 1;

  if (capture_out != nullptr) {
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
    const size_t n = static_cast<size_t>(T) * H;
    // Copies on the model's own stream: a loopback world's ranks share the
    // process, and a device-wide wait would sit on a peer's collective.
    std::vector<uint16_t> state(n), attn(n), mlp_in(n);
    DGPP_CUDA_OK(cudaMemcpyAsync(state.data(), new_partial, n * 2, cudaMemcpyDeviceToHost, stream_));
    DGPP_CUDA_OK(cudaMemcpyAsync(attn.data(), a_, n * 2, cudaMemcpyDeviceToHost, stream_));
    DGPP_CUDA_OK(cudaMemcpyAsync(mlp_in.data(), xm_, n * 2, cudaMemcpyDeviceToHost, stream_));
    capture_out->layer_states.push_back(std::move(state));
    visit_attn_.push_back(std::move(attn));
    visit_mlp_in_.push_back(std::move(mlp_in));
    const size_t K = static_cast<size_t>(cfg_.num_experts_per_tok);
    std::vector<int32_t> ids(static_cast<size_t>(T) * K);
    std::vector<float> weights(static_cast<size_t>(T) * K);
    DGPP_CUDA_OK(cudaMemcpyAsync(ids.data(), moe_->route_ids(), ids.size() * 4, cudaMemcpyDeviceToHost, stream_));
    DGPP_CUDA_OK(cudaMemcpyAsync(weights.data(), moe_->route_weights(), weights.size() * 4, cudaMemcpyDeviceToHost, stream_));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
    capture_out->route_ids.push_back(std::move(ids));
    capture_out->route_weights.push_back(std::move(weights));
  }
}

Inkling2Model::Outputs Inkling2Model::run_rows(const RowRun& run) {
  const int T = run.T;
  if (run.capture && loader_.residency() != Inkling2Residency::Resident)
    throw std::logic_error("run_rows: a capture needs a resident stack");
  if (run.capture_layers && run.num_spans > 0)
    throw std::invalid_argument("run_rows: layer capture is the scalar diagnostic forward's");
  const int H = cfg_.hidden_size;
  decode_gemm(run.decode);
  const RowInputs in = begin_run(run);
  // The embedding through its norm is the first partial state.
  embed_gather_bf16(globals_.embed, in.tokens, y_, T, H, stream_);
  add_rmsnorm_bf16_one(y_, nullptr, globals_.embed_norm, slot(0, 0), T, H, cfg_.rms_norm_eps, stream_);
  // An image row is the vision tower's row in place of the normed embedding.
  apply_image_embeddings(slot(0, 0), run);
  Walk w;
  w.run = &run;
  w.req_ids = in.req_ids;
  w.pos = in.pos;
  w.T = T;
  w.capture = run.capture;
  Outputs out;
  // A decode run under row traces records what the diagnostic forward does:
  // a test's view of one step, never a captured walk.
  const bool trace = run.capture_layers || (decode_row_traces_ && run.decode && !run.capture && run.num_spans == 0);
  if (trace) {
    visit_attn_.clear();
    visit_mlp_in_.clear();
  }
  for (int v = 0; v < visits_; ++v) {
    visit(v, w, trace ? &out : nullptr);
    // Loop guidance: the first pass's states, kept as its last core layer
    // leaves them.
    if (v == keep_visit_) keep_reference(w);
  }
  // The output: one more mix through the final norm.
  {
    Ink2MixArgs m;
    for (int n = 0; n < w.num_blocks; ++n) m.stack[n] = slot(n, w.parity);
    m.n_stack = w.num_blocks;
    m.last = slot(w.num_blocks, w.parity);
    m.queries = globals_.output_query;
    m.n_queries = 1;
    m.norm_w[0] = globals_.final_norm;
    m.out_bf16[0] = h_;
    m.norm_eps = cfg_.rms_norm_eps;
    m.rows = T;
    m.hidden = H;
    ink2_mix(m, stream_);
  }
  const bool packed_logits = project_head_rows(run, [&](int input_row, int output_row, int n) {
    const bool decode = run.decode || input_row < run.decode_rows();
    decode_gemm(decode);
    float* logits = logits_ + static_cast<size_t>(output_row) * lm_vocab_count_;
    const uint16_t* hidden = h_ + static_cast<size_t>(input_row) * H;
    if (!(decode && packed_head_wide(hidden, logits, n)))
      gemm_.matmul(hidden, globals_.unembed, logits, n, lm_vocab_count_, H, DType::BF16, GemmOut::F32,
                   static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
    if (cfg_.final_logit_softcapping > 0.0f)
      ink2_logits_softcap(logits, static_cast<int64_t>(n) * lm_vocab_count_, cfg_.final_logit_softcapping, stream_);
  });
  // The draft model's last tap is the final-norm output. The run's rows
  // feed its planes at their positions, rejected verify rows included: the
  // next rows at those positions overwrite them.
  if (df_.on()) {
    copy_rows_bf16(df_.tap_rows(df_.tap(visits_)), h_, T, H, stream_);
    // A decode pass's context projections are recorded with the pass.
    draft_gemm_.set_decode_mma(run.decode);
    df_.store_features(T, in.req_ids, in.pos);
  }
  // Loop guidance, after everything that reads the model's own walk.
  if (cfg_.loop_guidance) reference_readout(run, w);
  return finish_run(run, std::move(out), packed_logits);
}

// ---------------------------------------------------------------------------
// Loop guidance (engine.loop_guidance).
// ---------------------------------------------------------------------------
void Inkling2Model::configure_loop_guidance(int req, float strength) {
  if (req < 0 || req >= max_requests_) throw std::out_of_range("configure_loop_guidance: request slot");
  if (!loop_guidance()) {
    if (strength != 0.0f)
      throw std::logic_error("configure_loop_guidance: this deployment runs without engine.loop_guidance");
    return;
  }
  if (!std::isfinite(strength) || strength < -1.0f || strength > 4.0f)
    throw std::invalid_argument("loop_guidance_strength must be in [-1, 4]");
  guidance_strength_host_[static_cast<size_t>(req)] = strength;
  DGPP_CUDA_OK(cudaMemcpyAsync(guidance_strength_ + req, &guidance_strength_host_[static_cast<size_t>(req)],
                               sizeof(float), cudaMemcpyHostToDevice, stream_));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}

// The states a pass hands on sit in the core's stack slots of its parity:
// the blocks it completed after its first, then its partial.
void Inkling2Model::keep_reference(const Walk& w) {
  const int core0 = cfg_.core_start / cfg_.block_layers + 1;
  const size_t plane = static_cast<size_t>(max_tokens_) * static_cast<size_t>(cfg_.hidden_size);
  for (int j = 0; j < bank_rows_; ++j)
    copy_rows_bf16(reference_ + static_cast<size_t>(j) * plane, slot(core0 + j, 0), w.T, cfg_.hidden_size, stream_);
}

// The stack is a walk's scratch and the model's own output is taken, so the
// kept states take the last pass's slots and the layers after the core run
// from there as they did the first time, over the readout's own planes and
// ring entries. Every row of the walk takes part: a later row's readout
// attends over the keys and values this one leaves.
void Inkling2Model::reference_readout(const RowRun& run, Walk& w) {
  const int T = run.T, H = cfg_.hidden_size;
  const int core0 = cfg_.core_start / cfg_.block_layers + 1;
  const int parity = (cfg_.num_passes - 1) & 1;
  const size_t plane = static_cast<size_t>(max_tokens_) * static_cast<size_t>(H);
  for (int j = 0; j < bank_rows_; ++j)
    copy_rows_bf16(slot(core0 + j, parity), reference_ + static_cast<size_t>(j) * plane, T, H, stream_);
  w.reference = true;
  w.parity = parity;
  w.num_blocks = core0 + bank_rows_ - 1;  // the blocks at the core's exit: its partial is the next index
  for (int v = visits_; v < visits_ + ref_visits_; ++v) visit(v, w, nullptr);
  {
    Ink2MixArgs m;
    for (int n = 0; n < w.num_blocks; ++n) m.stack[n] = slot(n, w.parity);
    m.n_stack = w.num_blocks;
    m.last = slot(w.num_blocks, w.parity);
    m.queries = globals_.output_query;
    m.n_queries = 1;
    m.norm_w[0] = globals_.final_norm;
    m.out_bf16[0] = reference_hidden_;
    m.norm_eps = cfg_.rms_norm_eps;
    m.rows = T;
    m.hidden = H;
    ink2_mix(m, stream_);
  }
  // The rows the model's head projected, through the same head in the same
  // launches, then the contrast in place on the model's logits, each row by
  // its request's strength.
  project_head_rows(run, [&](int input_row, int output_row, int n) {
    const bool decode = run.decode || input_row < run.decode_rows();
    for (int done = 0; done < n; done += kReferenceLogitRows) {
      const int rows = std::min(kReferenceLogitRows, n - done);
      const uint16_t* hidden = reference_hidden_ + static_cast<size_t>(input_row + done) * H;
      float* logits = logits_ + static_cast<size_t>(output_row + done) * lm_vocab_count_;
      decode_gemm(decode);
      if (!(decode && packed_head_wide(hidden, reference_logits_, rows)))
        gemm_.matmul(hidden, globals_.unembed, reference_logits_, rows, lm_vocab_count_, H, DType::BF16, GemmOut::F32,
                     static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
      if (cfg_.final_logit_softcapping > 0.0f)
        ink2_logits_softcap(reference_logits_, static_cast<int64_t>(rows) * lm_vocab_count_,
                            cfg_.final_logit_softcapping, stream_);
      ink2_logits_guide(logits, reference_logits_, rows, lm_vocab_count_, w.req_ids + input_row + done,
                        guidance_strength_, max_requests_, stream_);
    }
  });
}

// ---------------------------------------------------------------------------
// The cold diagnostic forward: slot 0, fresh state, every row.
// ---------------------------------------------------------------------------
Inkling2Model::Outputs Inkling2Model::forward(const std::vector<int64_t>& token_ids, bool capture_layers) {
  const int T = static_cast<int>(token_ids.size());
  if (T <= 0) throw std::invalid_argument("forward: empty token batch");
  if (T > max_tokens_) throw std::invalid_argument("forward: tokens exceed max_tokens");
  if (T > max_context_) throw std::invalid_argument("forward: tokens exceed the context bound");
  for (int64_t id : token_ids)
    if (id < 0 || id >= cfg_.vocab_size) throw std::invalid_argument("forward: token id out of range");
  if (session_pos_[0] != 0) throw std::logic_error("forward: slot 0 holds an open session (close it first)");
  open_slot(0);
  if (!pool_.ensure_request_blocks(0, T, stream_))
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
// The graph era.
// ---------------------------------------------------------------------------
void Inkling2Model::graph_prepare() {
  if (loader_.residency() != Inkling2Residency::Resident)
    throw std::logic_error("session_graph_prepare: the decode graph needs a resident stack");
  const bool pack = Bf12Companions::enabled() && !bf12_built_;
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const Inkling2LayerResident& r = loader_.load_layer(layer);
    moe_->bind(layer, r.moe, stream_);
    if (pack) pack_layer_companions(layer, r);
  }
  if (pack) finish_companions();
}

// The lossless 12-bit companions of the BF16 matrices the GEMM interface
// serves (engine.bf16_weights): the three attention projections, the output
// projection, the controllers' down projections and the unembedding.
void Inkling2Model::pack_layer_companions(int layer, const Inkling2LayerResident& r) {
  const auto t0 = std::chrono::steady_clock::now();
  const int64_t H = cfg_.hidden_size;
  const auto release = [&](const void* w) { return loader_.release_packed(layer, w); };
  const auto pack = [&](const uint16_t* w, int64_t n, int64_t k) {
    if (w != nullptr) bf12_.pack_and_release(w, n, k, gemm_, stream_, release);
  };
  const Inkling2AttnResident& a = r.attn;
  pack(a.q_side, a.q_side_rows, H);
  pack(a.k, a.kv_rows, H);
  pack(a.v, a.kv_rows, H);
  pack(a.o_proj, H, static_cast<int64_t>(a.local_heads) * kInk2HeadDim);
  if (r.core)
    for (const auto& c : r.ctrl) pack(c.down, cfg_.gate_rank, 2 * H);
  bf12_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void Inkling2Model::finish_companions() {
  const auto t0 = std::chrono::steady_clock::now();
  bf12_built_ = true;
  bf12_.pack_and_release(globals_.unembed, lm_vocab_count_, cfg_.hidden_size, gemm_, stream_,
                         [&](const void* w) { return loader_.release_packed(-1, w); });
  bf12_.finish(gemm_, kBf12ExpandSlots);
  bf12_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  bf12_.log_summary(loader_.rank(), bf12_s_);
}

}  // namespace dgpp
