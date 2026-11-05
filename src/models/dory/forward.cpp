#include "models/dory/forward.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "kernels/dory_attn.hpp"
#include "kernels/dory_head.hpp"
#include "kernels/dory_norm.hpp"
#include "kernels/kernels.hpp"
#include "kernels/pick.hpp"
#include "models/dory/reference.hpp"

namespace dgpp {

namespace {

template <class T>
T* dev_alloc(size_t n) {
  T* p = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  return p;
}

// A layer's kv planes: none on an MLP layer (the pool's own contract), the
// loop count on a recurrent attention layer under `per_loop`, one otherwise.
int kv_repeats_of(const DoryConfig& cfg, int layer, bool last_loop) {
  if (!cfg.is_attention(layer)) return 0;
  return cfg.kv_repeats(layer, last_loop);
}

}  // namespace

int DoryModel::recurrent_loops_ = 0;
DoryKvPolicy DoryModel::kv_policy_ = DoryKvPolicy::PerLoop;

void DoryModel::set_recurrent_loops(int loops) {
  if (loops <= 0 || loops > 8)
    throw std::invalid_argument("engine.dory_loops: the loop count must be in [1, 8]");
  recurrent_loops_ = loops;
}

void DoryModel::clear_recurrent_loops() { recurrent_loops_ = 0; }

size_t DoryModel::kv_block_bytes_static(const DoryConfig& config, DoryKvPolicy policy) {
  DoryConfig cfg = config;
  if (recurrent_loops_ > 0) cfg.n_recurrent_loops = recurrent_loops_;
  size_t planes = 0;
  for (int l = 0; l < cfg.num_hidden_layers; ++l)
    planes += kv_repeats_of(cfg, l, policy == DoryKvPolicy::LastLoop);
  return 2 * planes * kBlockTokens * static_cast<size_t>(cfg.kv_size()) * sizeof(uint16_t);
}

DoryKvPolicy DoryModel::parse_kv_policy(const std::string& name) {
  if (name == "per_loop") return DoryKvPolicy::PerLoop;
  if (name == "last_loop") return DoryKvPolicy::LastLoop;
  throw std::invalid_argument("engine.dory_kv_cache: unknown policy '" + name +
                              "' (per_loop, last_loop)");
}

DoryModel::DoryModel(const DoryConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
                     int64_t max_cache_tokens, DoryResidency residency, int max_requests,
                     int decode_rows, bool serving_logits, DoryKvPolicy kv_policy)
    : cfg_(cfg),
      loader_(cfg, checkpoint_dir, 0, 1, residency, LoaderHeadSharding::Full, false),
      policy_(kv_policy) {
  if (recurrent_loops_ > 0) cfg_.n_recurrent_loops = recurrent_loops_;
  if (max_tokens <= 0) throw std::invalid_argument("DoryModel: max_tokens must be positive");
  if (decode_rows > decode_rows_cap())
    throw std::invalid_argument("DoryModel: decode_rows exceeds the Dory limit of 32");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("DoryModel: max_requests must be in [1, kPickMaxRequests]");
  if (cfg_.head_dim != kDoryHeadDim)
    throw std::invalid_argument("DoryModel: 256-wide heads (the kernels' shape)");
  if (cfg_.n_recurrent_loops < 1)
    throw std::invalid_argument("DoryModel: n_recurrent_loops must be positive");
  if (static_cast<int>(cfg_.layer_kind.size()) != cfg_.num_hidden_layers)
    throw std::invalid_argument("DoryModel: layer_types must cover num_hidden_layers");
  if (cfg_.n_input_layers <= 0 || cfg_.n_recurrent_layers <= 0 ||
      cfg_.n_input_layers + cfg_.n_recurrent_layers >= cfg_.num_hidden_layers)
    throw std::invalid_argument("DoryModel: the input/recurrent/output partition does not fit");
  if (!cfg_.in_output_block(cfg_.num_hidden_layers - 1))
    throw std::invalid_argument("DoryModel: the output block is empty");
  init_stream();
  loader_.set_reader_stream(stream_);
  const int H = cfg_.hidden_size;
  globals_ = loader_.load_globals();
  {
    SessionParams sp;
    sp.max_tokens = max_tokens;
    sp.max_cache_tokens =
        ((std::max<int64_t>(max_cache_tokens, max_tokens) + kBlockTokens - 1) / kBlockTokens) * kBlockTokens;
    sp.rank = 0;
    sp.world = 1;
    sp.boundary = nullptr;
    sp.max_requests = max_requests;
    sp.decode_rows = decode_rows;
    sp.logits_rows = compact_logits_rows(serving_logits, decode_rows, max_requests);
    sp.mtp = false;  // plan D4: the checkpoint has no draft layer
    sp.vocab_size = cfg_.vocab_size;
    sp.hidden = H;
    sp.lm_vocab_begin = 0;
    sp.lm_vocab_count = cfg_.vocab_size;
    sp.max_position_embeddings = cfg_.max_position_embeddings;
    sp.block_tokens = kBlockTokens;
    // Every plane holds the full history (the sliding window is applied at
    // read time), so any position snapshots.
    sp.snapshot_align = 1;
    sp.draft_width = 0;
    init_session(sp);
  }
  gemm_ws_bytes_ =
      std::max<size_t>(64u << 20, gemm_.query_workspace_bytes(max_tokens_, lm_vocab_count_, H, DType::BF16));
  gemm_ws_ = dev_alloc<char>(gemm_ws_bytes_);
  n_split_ = DoryAttentionLayer::default_decode_splits();

  {
    DoryKvPoolShape shape;
    shape.repeats.resize(static_cast<size_t>(pool_layers()), 0);
    for (int l = 0; l < pool_layers(); ++l)
      shape.repeats[static_cast<size_t>(l)] = kv_repeats_of(cfg_, l, last_loop_policy());
    shape.kv_heads = cfg_.num_key_value_heads;
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
  input_h_ = dev_alloc<uint16_t>(M * H);
  state_ = dev_alloc<uint16_t>(M * H);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}

DoryModel::~DoryModel() {
  attn_.reset();
  mlp_.reset();
  cudaFree(resid_);
  cudaFree(x_);
  cudaFree(y_);
  cudaFree(input_h_);
  cudaFree(state_);
  cudaFree(gemm_ws_);
  cudaFree(snap_dev_);
}

DoryModel::MemoryPlan DoryModel::plan_memory(const DoryConfig& cfg_in, int max_tokens,
                                             int64_t max_cache_tokens, DoryResidency residency,
                                             int max_requests, int decode_rows,
                                             bool serving_logits, DoryKvPolicy kv_policy) {
  DoryConfig cfg = cfg_in;
  if (recurrent_loops_ > 0) cfg.n_recurrent_loops = recurrent_loops_;
  if (max_tokens <= 0) throw std::invalid_argument("plan_memory: max_tokens must be positive");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("plan_memory: max_requests must be in [1, kPickMaxRequests]");
  if (decode_rows > decode_rows_cap())
    throw std::invalid_argument("plan_memory: decode_rows exceeds the Dory limit of 32");
  const int64_t cache_tokens =
      ((std::max<int64_t>(max_cache_tokens, max_tokens) + kBlockTokens - 1) / kBlockTokens) * kBlockTokens;
  const bool last_loop = kv_policy == DoryKvPolicy::LastLoop;
  MemoryPlan plan;
  plan.context_tokens = std::min<int64_t>(cache_tokens, cfg.max_position_embeddings);
  const size_t M = static_cast<size_t>(max_tokens);
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t V = static_cast<size_t>(cfg.vocab_size);
  const int n_split = DoryAttentionLayer::default_decode_splits();
  const int rows = std::max({kDecodeRows, decode_rows, max_requests});

  if (residency == DoryResidency::Resident) {
    plan.add("model weights (resident)", DoryLayerStream::resident_bytes(cfg, 0, 1, LoaderHeadSharding::Full, false));
    plan.add("loader staging (pinned host, freed when the last layer is resident)", 0,
             DoryLayerStream::staging_plan_bytes(cfg, 0, 1, LoaderHeadSharding::Full, false));
  } else {
    size_t largest = 0;
    for (int l = 0; l < cfg.num_hidden_layers; ++l)
      largest = std::max(largest, DoryLayerStream::layer_bytes(cfg, l, 0, 1));
    plan.add("model weights (one streamed layer + globals)",
             largest + DoryLayerStream::globals_bytes(cfg, 0, 1, LoaderHeadSharding::Full));
  }
  plan.add("gemm workspace (at least)", size_t{64} << 20);
  {
    DoryKvPoolShape shape;
    shape.repeats.resize(static_cast<size_t>(cfg.num_hidden_layers), 0);
    for (int l = 0; l < cfg.num_hidden_layers; ++l)
      shape.repeats[static_cast<size_t>(l)] = kv_repeats_of(cfg, l, last_loop);
    shape.kv_heads = cfg.num_key_value_heads;
    shape.dim = cfg.head_dim;
    shape.block_tokens = kBlockTokens;
    shape.max_requests = max_requests;
    shape.token_slots = cache_tokens;
    plan.add(last_loop ? "kv cache pool (K/V bf16, last_loop: one plane per layer)"
                       : "kv cache pool (K/V bf16, per_loop: one plane per loop)",
             DoryKvPool::cache_bytes(shape));
  }
  {
    size_t core_dev = 0, core_pin = 0;
    session_core_plan_bytes(max_tokens, max_requests, cfg.hidden_size, static_cast<int>(V), false, 0,
                            &core_dev, &core_pin, rows, serving_logits);
    plan.add("activations (session core, the walk's buffers)", core_dev + 5 * M * H * 2, core_pin);
  }
  {
    size_t layers = DoryAttentionLayer::scratch_bytes(cfg, max_tokens, rows, n_split);
    layers += DoryMlpLayer::scratch_bytes(cfg, max_tokens);
    plan.add("layer objects (attention, MLP)", layers);
  }
  return plan;
}

// Both layer objects exist before a capture (nothing may allocate inside a
// captured region): graph_prepare() builds them by walking the stack.
void DoryModel::graph_prepare() {
  if (loader_.residency() != DoryResidency::Resident)
    throw std::logic_error("session_graph_prepare: the decode graph needs a resident stack");
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const DoryLayerResident& r = loader_.load_layer(layer);
    if (r.kind == DoryLayerKind::Mlp) {
      if (!mlp_) mlp_ = std::make_unique<DoryMlpLayer>(r.mlp, cfg_, max_tokens_, max_decode_rows_);
      else mlp_->rebind(r.mlp);
    } else {
      if (!attn_)
        attn_ = std::make_unique<DoryAttentionLayer>(r.attn, cfg_, layer, max_tokens_, max_decode_rows_, n_split_,
                                            stream_);
      else attn_->rebind(r.attn, layer, stream_);
    }
  }
}

void DoryModel::reset_slot_state(int req) { pool_.release_request_blocks(req, stream_); }

GlmSpecSegments DoryModel::spec_segments(int, int) const { return GlmSpecSegments{}; }

void DoryModel::write_state_snapshot(int, uint8_t* d, int) {
  DGPP_CUDA_OK(cudaMemsetAsync(d, 0, kSnapshotStamp, stream_));
}

void DoryModel::mtp_run_rows(int, const int64_t*, int64_t, int, bool, bool, int, int, int) {
  throw std::logic_error(
      "mtp_run_rows: the Dory family has no draft layer (the checkpoint ships none, plan D4)");
}

// ---------------------------------------------------------------------------
// The row walk (the plan's §1.5 loop).
// ---------------------------------------------------------------------------
void DoryModel::enqueue_layer(int layer, int loop, const uint16_t* in, int T, uint16_t* out,
                              const WalkRows& rows) {
  if (T <= 0) return;
  const int H = cfg_.hidden_size;
  const DoryLayerResident& r = loader_.load_layer(layer);
  const bool is_mlp = r.kind == DoryLayerKind::Mlp;
  if (is_mlp) {
    if (!mlp_) mlp_ = std::make_unique<DoryMlpLayer>(r.mlp, cfg_, max_tokens_, max_decode_rows_);
    else mlp_->rebind(r.mlp);
  } else {
    if (!attn_)
      attn_ = std::make_unique<DoryAttentionLayer>(r.attn, cfg_, layer, max_tokens_, max_decode_rows_, n_split_,
                                            stream_);
    else attn_->rebind(r.attn, layer, stream_);
  }
  const DoryKvCache cache = is_mlp ? DoryKvCache{} : pool_.view(layer, loop);
  const auto visit = [&](int row0, int count, bool decode, int) {
    if (count <= 0) return;
    const size_t offset = static_cast<size_t>(row0) * H;
    if (is_mlp) {
      mlp_->enqueue(in + offset, count, out + offset, stream_);
      return;
    }
    DoryAttnRows arows;
    arows.req_ids = rows.req_ids + row0;
    arows.pos = rows.pos + row0;
    arows.decode = decode;
    attn_->enqueue(in + offset, count, arows, cache, out + offset, stream_);
  };
  // Every row is addressed by its own request id and position, so the spans
  // of a grouped walk combine into one group per decoder/prefill class.
  if (rows.run)
    rows.run->for_each_kernel_group([](int, int) { return true; }, visit);
  else
    visit(0, T, rows.decode, 0);
}

DoryModel::Outputs DoryModel::run_rows(const RowRun& run) {
  const int T = run.T;
  if (T <= 0) throw std::invalid_argument("run_rows: empty row batch");
  if (run.capture && loader_.residency() != DoryResidency::Resident)
    throw std::logic_error("run_rows: a capture needs a resident stack");
  const int H = cfg_.hidden_size;
  const float eps = kDoryJustnormEps;
  const RowInputs in = begin_run(run);
  embed_gather_bf16(globals_.embed, in.tokens, resid_, T, H, stream_);
  Outputs out;
  WalkRows rows;
  rows.run = &run;
  rows.req_ids = in.req_ids;
  rows.pos = in.pos;
  rows.decode = run.decode;
  rows.capture = run.capture;
  // A capture stages its rows on the device and hands them over in one go at
  // the end of the walk, so it does not perturb the arithmetic (a per-layer
  // synchronize would). A capture too large to stage keeps the host path.
  const size_t app_bytes = static_cast<size_t>(T) * H * 2;
  const int max_apps = cfg_.num_hidden_layers * std::max(cfg_.n_recurrent_loops, 1) + 2;
  bool staged = app_bytes * static_cast<size_t>(max_apps) <= kCaptureStagingCap;
  if (run.capture_layers && staged) {
    const size_t need = app_bytes * static_cast<size_t>(max_apps);
    if (snap_dev_bytes_ < need) {
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      if (snap_dev_) DGPP_CUDA_OK(cudaFree(snap_dev_));
      DGPP_CUDA_OK(cudaMalloc(&snap_dev_, need));
      snap_dev_bytes_ = need;
    }
    snap_apps_ = 0;
  } else if (!run.capture_layers) {
    staged = false;
  }
  const auto capture_state = [&](const uint16_t* buf) {
    if (!run.capture_layers) return;
    if (staged) {
      DGPP_CUDA_OK(cudaMemcpyAsync(snap_dev_ + static_cast<size_t>(snap_apps_) * (app_bytes / 2),
                                   buf, app_bytes, cudaMemcpyDeviceToDevice, stream_));
      ++snap_apps_;
      return;
    }
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
    std::vector<uint16_t> snap(static_cast<size_t>(T) * H);
    DGPP_CUDA_OK(cudaMemcpy(snap.data(), buf, snap.size() * 2, cudaMemcpyDeviceToHost));
    out.layer_states.push_back(std::move(snap));
  };
  const auto copy_rows = [&](uint16_t* dst, const uint16_t* src) {
    // The kernel form, not a cudaMemcpyAsync: the decode-graph capture
    // refuses any memcpy node (engine/graph_check.hpp).
    dory_copy_rows_bf16(src, dst, static_cast<size_t>(T) * H, stream_);
  };

  // ---- the input block ------------------------------------------------------
  uint16_t* cur = resid_;
  uint16_t* alt = x_;
  for (int l = 0; l < cfg_.n_input_layers; ++l) {
    enqueue_layer(l, 0, cur, T, alt, rows);
    std::swap(cur, alt);
    capture_state(cur);
  }
  // input_h = h, and the state the first loop re-injects is input_h itself.
  copy_rows(input_h_, cur);
  copy_rows(state_, cur);

  // ---- the loop (plan §1.5) --------------------------------------------------
  for (int i = 0; i < cfg_.n_recurrent_loops; ++i) {
    // h = justnorm(state + input_h): the add in fp32, one rounding.
    dory_add_justnorm_bf16(state_, input_h_, state_, T, H, eps, stream_);
    cur = state_;
    alt = y_;
    for (int l = cfg_.n_input_layers; l < cfg_.n_input_layers + cfg_.n_recurrent_layers; ++l) {
      enqueue_layer(l, i, cur, T, alt, rows);
      std::swap(cur, alt);
      capture_state(cur);
    }
    // state = justnorm(h); h = state. The norm kernels load every element they
    // own before writing any, so the in-place case (an even layer count) is
    // exact.
    dory_justnorm_bf16(cur, state_, T, H, eps, stream_);
  }

  // ---- the output block ------------------------------------------------------
  cur = state_;
  alt = x_;
  for (int l = cfg_.n_input_layers + cfg_.n_recurrent_layers; l < cfg_.num_hidden_layers; ++l) {
    enqueue_layer(l, 0, cur, T, alt, rows);
    std::swap(cur, alt);
    capture_state(cur);
  }

  // ---- the head (plan §1.6) --------------------------------------------------
  // There is no final norm: the last sublayer's slerp already normalized, and
  // the reference projects the walk's output as it stands.
  copy_rows(h_, cur);
  const bool packed_logits = project_head_rows(run, [&](int input_row, int output_row, int head_rows) {
    gemm_.matmul(h_ + static_cast<size_t>(input_row) * H, globals_.lm_head,
                 logits_ + static_cast<size_t>(output_row) * lm_vocab_count_, head_rows,
                 lm_vocab_count_, H, DType::BF16, GemmOut::F32, static_cast<size_t>(H), gemm_ws_,
                 gemm_ws_bytes_, stream_);
    dory_logits_scale_f32(logits_ + static_cast<size_t>(output_row) * lm_vocab_count_,
                          globals_.logit_scale, cfg_.logit_factor(), head_rows, lm_vocab_count_,
                          stream_);
  });
  out = finish_run(run, std::move(out), packed_logits);
  if (run.capture_layers && staged) {
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
    out.layer_states.reserve(static_cast<size_t>(snap_apps_));
    std::vector<uint16_t> snap(static_cast<size_t>(T) * H);
    for (int a = 0; a < snap_apps_; ++a) {
      DGPP_CUDA_OK(cudaMemcpy(snap.data(),
                              snap_dev_ + static_cast<size_t>(a) * (app_bytes / 2), app_bytes,
                              cudaMemcpyDeviceToHost));
      out.layer_states.push_back(snap);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// The cold diagnostic forward: slot 0, fresh state, every row.
// ---------------------------------------------------------------------------
DoryModel::Outputs DoryModel::forward(const std::vector<int64_t>& token_ids, bool capture_layers) {
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

}  // namespace dgpp
