#include "models/minimax/forward.hpp"

#include <algorithm>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/kernels.hpp"
#include "kernels/minimax.hpp"
#include "kernels/qwen_norm.hpp"

namespace dgpp {
namespace {
template <class T>
T* device(size_t n) {
  T* p = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  return p;
}
template <class T>
T* pinned(size_t n) {
  T* p = nullptr;
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&p), std::max<size_t>(n, 1) * sizeof(T),
                             cudaHostAllocMapped));
  return p;
}
void validate_run(const MinimaxConfig& c, int t, int rank, int world, int slots, bool mtp,
                  int decode, LatentFormat format) {
  c.validate_tp(rank, world);
  if (t <= 0 || slots <= 0 || slots > kPickMaxRequests || decode > MinimaxModel::decode_rows_cap())
    throw std::invalid_argument("MiniMax: invalid forward/session capacity");
  if (mtp)
    throw std::invalid_argument(
        "MiniMax: DSpark execution is not enabled yet; engine.mtp must be false");
  if (format != LatentFormat::kBf16)
    throw std::invalid_argument("MiniMax: this baseline requires BF16 KV and index caches");
}
int64_t capacity(int64_t cache, int tokens) {
  return ((std::max<int64_t>(cache, tokens) + 127) / 128) * 128;
}
MinimaxKvPoolShape pool_shape(const MinimaxConfig& c, const MinimaxLocalGeometry& g, int slots,
                              int64_t cache) {
  return {c.num_hidden_layers, g.local_kv_heads, 128, 128, slots, cache};
}
}  // namespace
MinimaxModel::MinimaxModel(const MinimaxConfig& c, const std::string& checkpoint, int max_tokens,
                           int64_t cache, LoaderResidency residency, BoundaryReducer* boundary,
                           int rank, int world, int slots, bool mtp, int decode_rows,
                           LatentFormat format, bool serving_logits)
    : cfg_(c),
      loader_(c, checkpoint, rank, world, residency,
              world > 1 ? LoaderHeadSharding::VocabSharded : LoaderHeadSharding::Full) {
  validate_run(c, max_tokens, rank, world, slots, mtp, decode_rows, format);
  if ((world > 1) != (boundary != nullptr))
    throw std::invalid_argument("MiniMax TP requires a boundary reducer");
  init_stream();
  loader_.set_reader_stream(stream_);
  globals_ = loader_.load_globals();
  SessionParams p;
  p.max_tokens = max_tokens;
  p.max_cache_tokens = capacity(cache, max_tokens);
  p.rank = rank;
  p.world = world;
  p.boundary = boundary;
  p.max_requests = slots;
  p.decode_rows = decode_rows;
  p.logits_rows = compact_logits_rows(serving_logits, decode_rows, slots);
  p.mtp = false;
  p.vocab_size = c.vocab_size;
  p.hidden = c.hidden_size;
  p.lm_vocab_begin = globals_.lm_vocab_begin;
  p.lm_vocab_count = globals_.lm_vocab_count;
  p.max_position_embeddings = c.max_position_embeddings;
  p.block_tokens = 128;
  p.snapshot_align = 1;
  p.draft_width = c.hidden_size;
  p.eos = static_cast<int32_t>(c.eos_token_ids.at(0));
  init_session(p);
  const auto& g = loader_.geometry();
  pool_.init(pool_shape(c, g, slots, max_cache_tokens_));
  const size_t elements = static_cast<size_t>(max_tokens_) * c.hidden_size;
  resid_ = device<uint16_t>(elements);
  x_ = device<uint16_t>(elements);
  y_ = device<uint16_t>(elements);
  gemm_ws_bytes_ = std::max<size_t>(
      64u << 20,
      gemm_.query_workspace_bytes(max_tokens_, lm_vocab_count_, c.hidden_size, DType::BF16));
  gemm_ws_ = device<char>(gemm_ws_bytes_);
  gemm_.set_decode_rows(std::min(max_decode_rows_, dense_gemv_rows()));
  attn_ = std::make_unique<MinimaxAttentionLayer>(c, g, max_tokens_, max_cache_tokens_);
  if (c.num_moe_layers() < c.num_hidden_layers)
    dense_ = std::make_unique<MinimaxMlp>(c, static_cast<int>(g.local_dense_inter), max_tokens_);
  if (c.num_moe_layers()) moe_ = std::make_unique<MinimaxMoeLayer>(c, g, max_tokens_);
  const size_t routes =
      static_cast<size_t>(c.num_moe_layers()) * max_tokens_ * c.num_experts_per_tok;
  trace_ids_ = pinned<int32_t>(routes);
  trace_weights_ = pinned<float>(routes);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}
MinimaxModel::~MinimaxModel() {
  cudaFree(resid_);
  cudaFree(x_);
  cudaFree(y_);
  cudaFree(gemm_ws_);
  cudaFreeHost(trace_ids_);
  cudaFreeHost(trace_weights_);
}
size_t MinimaxModel::kv_block_bytes_static(const MinimaxConfig& c, int rank, int world,
                                           LatentFormat format) {
  validate_run(c, 1, rank, world, 1, false, 0, format);
  return static_cast<size_t>(c.num_hidden_layers) * 128 *
         (2 * (c.num_key_value_heads / world) * 128 + 128) * 2;
}
MinimaxModel::MemoryPlan MinimaxModel::plan_memory(const MinimaxConfig& c, int t, int64_t cache,
                                                   int rank, int world, LoaderResidency residency,
                                                   int slots, bool mtp, int decode,
                                                   LatentFormat format, bool serving) {
  validate_run(c, t, rank, world, slots, mtp, decode, format);
  const auto head = world > 1 ? LoaderHeadSharding::VocabSharded : LoaderHeadSharding::Full;
  const auto g = MinimaxLocalGeometry::from_config(c, rank, world, head);
  const int64_t cap = capacity(cache, t);
  const size_t H = c.hidden_size, V = g.lm_vocab_count, T = t;
  MemoryPlan plan;
  plan.context_tokens = std::min<int64_t>(cap, c.max_position_embeddings);
  if (residency == LoaderResidency::Resident) {
    size_t weights = MinimaxLayerStream::resident_bytes(c, rank, world, head, false);
    if (Bf12Companions::packed_only())
      weights -= MinimaxLayerStream::side_bytes(c, rank, world, head, false);
    plan.add("model weights (published MXFP8/NVFP4)", weights);
    plan.add("loader staging (released after residency)", 0,
             MinimaxLayerStream::staging_plan_bytes(c, rank, world, head, false));
  } else {
    size_t largest = 0;
    for (int l = 0; l < c.num_hidden_layers; ++l)
      largest = std::max(largest, MinimaxLayerStream::layer_bytes(c, l, rank, world));
    plan.add("streamed layer and globals",
             largest + MinimaxLayerStream::globals_bytes(c, rank, world, head));
  }
  if (Bf12Companions::enabled() && residency == LoaderResidency::Resident) {
    plan.add("lossless BF12 output head", Bf12Companions::planned_bytes(V, H));
    if (Bf12Companions::packed_only())
      plan.add("head expansion", Bf12Companions::planned_slot_bytes({V * H * 2}));
  }
  plan.add("GEMM workspace (at least)", size_t{64} << 20);
  plan.add("main KV and unpooled index K",
           MinimaxKvPool::cache_bytes(pool_shape(c, g, slots, cap)));
  size_t core = 0, pin = 0;
  session_core_plan_bytes(t, slots, c.hidden_size, g.lm_vocab_count, false, c.hidden_size, &core,
                          &pin, std::max({kDecodeRows, decode, slots}), serving);
  plan.add("session and residual buffers", core + T * H * 6,
           pin + static_cast<size_t>(c.num_moe_layers()) * T * c.num_experts_per_tok * 8);
  plan.add("sparse attention workspace", MinimaxAttentionLayer::scratch_bytes(c, g, t, cap));
  if (c.num_moe_layers() < c.num_hidden_layers)
    plan.add("dense MLP workspace",
             MinimaxMlp::scratch_bytes(static_cast<int>(g.local_dense_inter), t));
  if (c.num_moe_layers())
    plan.add("routed and shared expert workspace", MinimaxMoeLayer::scratch_bytes(c, g, t));
  return plan;
}
void MinimaxModel::reset_slot_state(int req) {
  pool_.release_request_blocks(req, stream_);
}
void MinimaxModel::write_state_snapshot(int, uint8_t* p, int) {
  DGPP_CUDA_OK(cudaMemsetAsync(p, 0, 16, stream_));
}
uint16_t* MinimaxModel::stage(int rows, bool capture) {
  if (!boundary_) return y_;
  auto* p = boundary_->stage(rows, cfg_.hidden_size);
  if (!p && capture)
    throw std::runtime_error("MiniMax capture boundary does not fit the staging buffer");
  return p ? p : y_;
}
void MinimaxModel::fold(uint16_t* p, int rows, bool capture) {
  if (!boundary_) return;
  if (!capture) DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  boundary_->reduce(p, rows, cfg_.hidden_size);
}
void MinimaxModel::norm(uint16_t* resid, const uint16_t* add, const uint16_t* w, uint16_t* out,
                        int rows) {
  if (add)
    minimax_add_norm(resid, add, w, out, rows, cfg_.hidden_size, cfg_.rms_norm_eps, stream_);
  else
    qwen_rmsnorm_bf16(resid, w, out, rows, cfg_.hidden_size, cfg_.rms_norm_eps, stream_);
}
MinimaxModel::Outputs MinimaxModel::run_rows(const RowRun& run) {
  if (run.capture && loader_.residency() != LoaderResidency::Resident)
    throw std::logic_error("MiniMax graph requires resident weights");
  const int T = run.T, H = cfg_.hidden_size, K = cfg_.num_experts_per_tok;
  const auto in = begin_run(run);
  embed_gather_bf16(globals_.embed, in.tokens, resid_, T, H, stream_);
  Outputs out;
  const uint16_t* pending = nullptr;
  const bool traces = route_traces_ && !run.capture && (!run.decode || decode_row_traces_);
  int ordinal = 0;
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const auto& r = loader_.load_layer(layer);
    if (r.moe && !run.capture) moe_->prepare_table(layer, r.moe_w, stream_);
    norm(resid_, pending, r.input_norm, x_, T);
    if (observer_ && !run.capture) observer_(layer, "input_norm", x_, T, stream_);
    auto* attn = stage(T, run.capture);
    run.for_each_kernel_group([](int a, int b) { return a > 128 && b > 128; },
                              [&](int row0, int n, bool decode, int) {
                                PagedAttentionRows rows;
                                rows.req_ids = in.req_ids + row0;
                                rows.pos = in.pos + row0;
                                rows.decode = decode;
                                attn_->enqueue(r.attn, x_ + static_cast<size_t>(row0) * H, n, rows,
                                               pool_.view(layer),
                                               attn + static_cast<size_t>(row0) * H, stream_);
                              });
    fold(attn, T, run.capture);
    if (observer_ && !run.capture) observer_(layer, "attention", attn, T, stream_);
    norm(resid_, attn, r.post_norm, x_, T);
    if (observer_ && !run.capture) observer_(layer, "ffn_norm", x_, T, stream_);
    auto* ffn = stage(T, run.capture);
    run.for_each_kernel_group(
        [&](int a, int b) { return r.moe || (a > 128 && b > 128); },
        [&](int row0, int n, bool, int) {
          const size_t off = static_cast<size_t>(row0) * H;
          if (r.moe) {
            moe_->enqueue(layer, r.moe_w, x_ + off, n, ffn + off, stream_);
            if (traces) {
              const size_t at = (static_cast<size_t>(ordinal) * max_tokens_ + row0) * K;
              DGPP_CUDA_OK(cudaMemcpyAsync(trace_ids_ + at, moe_->route_ids(),
                                           static_cast<size_t>(n) * K * 4, cudaMemcpyDeviceToHost,
                                           stream_));
              DGPP_CUDA_OK(cudaMemcpyAsync(trace_weights_ + at, moe_->route_weights(),
                                           static_cast<size_t>(n) * K * 4, cudaMemcpyDeviceToHost,
                                           stream_));
            }
          } else
            dense_->enqueue(r.dense, x_ + off, n, ffn + off, stream_);
        });
    fold(ffn, T, run.capture);
    if (observer_ && !run.capture) observer_(layer, "ffn", ffn, T, stream_);
    pending = ffn;
    if (r.moe) ++ordinal;
    if (run.capture_layers) {
      // Capturing a residual must not materialize the sum on the execution
      // path: the next fused norm needs its unrounded FP32 value.
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      out.layer_states.emplace_back(static_cast<size_t>(T) * H);
      std::vector<uint16_t> add(static_cast<size_t>(T) * H);
      DGPP_CUDA_OK(cudaMemcpy(out.layer_states.back().data(), resid_,
                              static_cast<size_t>(T) * H * 2, cudaMemcpyDeviceToHost));
      DGPP_CUDA_OK(cudaMemcpy(add.data(), pending, add.size() * 2, cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < add.size(); ++i)
        out.layer_states.back()[i] = float_to_bf16_bits(
            bf16_bits_to_float(out.layer_states.back()[i]) + bf16_bits_to_float(add[i]));
    }
  }
  norm(resid_, pending, globals_.final_norm, h_, T);
  const bool packed = project_head_rows(run, [&](int input, int output, int n) {
    gemm_.set_bf12_wide(run.decode || input < run.decode_rows());
    gemm_.matmul(h_ + static_cast<size_t>(input) * H, globals_.lm_head,
                 logits_ + static_cast<size_t>(output) * lm_vocab_count_, n, lm_vocab_count_, H,
                 DType::BF16, GemmOut::F32, H, gemm_ws_, gemm_ws_bytes_, stream_);
  });
  out = finish_run(run, std::move(out), packed);
  if (traces)
    for (int l = 0; l < ordinal; ++l) {
      const size_t at = static_cast<size_t>(l) * max_tokens_ * K;
      out.route_ids.emplace_back(trace_ids_ + at, trace_ids_ + at + static_cast<size_t>(T) * K);
      out.route_weights.emplace_back(trace_weights_ + at,
                                     trace_weights_ + at + static_cast<size_t>(T) * K);
    }
  return out;
}
MinimaxModel::Outputs MinimaxModel::forward(const std::vector<int64_t>& ids, bool layers) {
  const int T = static_cast<int>(ids.size());
  if (T <= 0 || T > max_tokens_ || T > max_context_)
    throw std::invalid_argument("MiniMax forward token count");
  for (auto id : ids)
    if (id < 0 || id >= cfg_.vocab_size) throw std::invalid_argument("MiniMax forward token id");
  if (session_pos_[0] != 0) throw std::logic_error("MiniMax forward slot already open");
  open_slot(0);
  if (!pool_.ensure_request_blocks(0, T, stream_))
    throw std::runtime_error("MiniMax forward KV capacity");
  RowRun run;
  run.req = 0;
  run.ids = ids.data();
  run.T = T;
  run.pos0 = 0;
  run.decode = false;
  run.all_rows = true;
  run.capture_layers = layers;
  auto out = run_rows(run);
  session_close(0);
  return out;
}
void MinimaxModel::graph_prepare() {
  if (loader_.residency() != LoaderResidency::Resident)
    throw std::logic_error("MiniMax graph requires residency");
  for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
    const auto& r = loader_.load_layer(l);
    if (r.moe) moe_->prepare_table(l, r.moe_w, stream_);
  }
  if (Bf12Companions::enabled() && !bf12_built_) {
    bf12_.pack_and_release(globals_.lm_head, lm_vocab_count_, cfg_.hidden_size, gemm_, stream_,
                           [&](const void* p) { return loader_.release_packed(-1, p); });
    bf12_.finish(gemm_, 1);
    bf12_built_ = true;
  }
}
}  // namespace dgpp
