#include "models/minimax/forward.hpp"

#include <algorithm>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "kernels/dflash2.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/kernels.hpp"
#include "kernels/minimax.hpp"
#include "kernels/mma_gemv.hpp"
#include "kernels/qwen_norm.hpp"
#include "kernels/qwen_vision.hpp"
#include "models/minimax/dspark.hpp"

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
  if (mtp) {
    if (!c.dspark)
      throw std::invalid_argument("engine.mtp: MiniMax requires its bundled dspark/config.json");
    c.dspark->validate_against(
        {c.hidden_size, c.vocab_size, c.num_hidden_layers, c.max_position_embeddings});
    if (!c.dspark->tp_divisible(world) || c.dspark->local_intermediate(world) % 32)
      throw std::invalid_argument("MiniMax DSpark: TP world does not divide the draft geometry");
    if (t < c.dspark->query_rows())
      throw std::invalid_argument("MiniMax DSpark: max_tokens must cover the query block");
  }
  (void)MinimaxKvPool::head_bytes(format);
}
int64_t capacity(int64_t cache, int tokens) {
  return ((std::max<int64_t>(cache, tokens) + 127) / 128) * 128;
}
MinimaxKvPoolShape pool_shape(const MinimaxConfig& c, const MinimaxLocalGeometry& g, int slots,
                              int64_t cache, bool draft, LatentFormat format) {
  MinimaxKvPoolShape shape{c.num_hidden_layers, g.local_kv_heads, 128, 128, slots, cache};
  shape.format = format;
  if (draft) {
    shape.draft_layers = c.dspark->num_hidden_layers;
    shape.draft_kv_heads = c.dspark->local_kv_heads(g.world);
    shape.draft_dim = c.dspark->head_dim;
  }
  return shape;
}
}  // namespace
MinimaxModel::MinimaxModel(const MinimaxConfig& c, const std::string& checkpoint, int max_tokens,
                           int64_t cache, LoaderResidency residency, BoundaryReducer* boundary,
                           int rank, int world, int slots, bool mtp, int decode_rows,
                           LatentFormat format, bool serving_logits, bool native)
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
  pool_.init(pool_shape(c, g, slots, max_cache_tokens_, mtp, format));
  const size_t elements = static_cast<size_t>(max_tokens_) * c.hidden_size;
  resid_ = device<uint16_t>(elements);
  x_ = device<uint16_t>(elements);
  y_ = device<uint16_t>(elements);
  gemm_ws_bytes_ = std::max<size_t>(
      64u << 20,
      gemm_.query_workspace_bytes(max_tokens_, lm_vocab_count_, c.hidden_size, DType::BF16));
  gemm_ws_ = device<char>(gemm_ws_bytes_);
  gemm_.set_decode_rows(std::min(max_decode_rows_, dense_gemv_rows()));
  attn_ =
      std::make_unique<MinimaxAttentionLayer>(c, g, max_tokens_, max_cache_tokens_, native, format);
  if (c.num_moe_layers() < c.num_hidden_layers)
    dense_ =
        std::make_unique<MinimaxMlp>(c, static_cast<int>(g.local_dense_inter), max_tokens_, native);
  if (c.num_moe_layers()) moe_ = std::make_unique<MinimaxMoeLayer>(c, g, max_tokens_, native);
  const size_t routes =
      static_cast<size_t>(c.num_moe_layers()) * max_tokens_ * c.num_experts_per_tok;
  trace_ids_ = pinned<int32_t>(routes);
  trace_weights_ = pinned<float>(routes);
  if (mtp) {
    DFlash2Host host;
    host.hidden = c.hidden_size;
    host.num_target_layers = c.num_hidden_layers;
    host.target_vocab = c.vocab_size;
    host.vocab_count = lm_vocab_count_;
    host.vocab_begin = lm_vocab_begin_;
    host.rank = rank;
    host.world = world;
    host.native_activations = native;
    host.stream = stream_;
    host.max_requests = max_requests_;
    host.max_tokens = max_tokens_;
    host.max_decode_rows = max_decode_rows_;
    host.max_context = max_context();
    host.position_limit = c.max_position_embeddings;
    host.session_pos = d_session_pos_;
    host.decode_tokens = d_tokens_;
    host.device_tables = pool_.blocks().device_tables();
    host.total_blocks = static_cast<int>(pool_.total_blocks());
    host.block_tokens = 128;
    host.ensure_request_blocks = [this](int req, int64_t pos, cudaStream_t st) {
      return pool_.ensure_request_blocks(req, pos, st);
    };
    host.plane = [this](int l) {
      const auto v = pool_.draft_view(l);
      return DFlash2Host::Plane{v.k, v.v};
    };
    host.session_position = [this](int req) { return session_position(req); };
    host.embed = globals_.embed;
    host.gemm = &gemm_;
    host.gemm_ws = gemm_ws_;
    host.gemm_ws_bytes = gemm_ws_bytes_;
    host.gemm_rows = [](int, bool) {};
    host.boundary = [this] { return boundary_; };
    host.head = [this](const uint16_t* hidden, float* out, int rows) {
      project_head(hidden, out, rows, true);
    };
    df_.init(host, *c.dspark,
             load_minimax_dspark(*c.dspark, checkpoint + "/dspark", stream_, rank, world));
  }
  request_images_.resize(slots, nullptr);
  if (c.vision) vision_ = std::make_unique<MinimaxVisionEncoder>(*c.vision, checkpoint, stream_);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}
MinimaxModel::~MinimaxModel() {
  df_.free();
  cudaFree(resid_);
  cudaFree(x_);
  cudaFree(y_);
  cudaFree(gemm_ws_);
  cudaFreeHost(trace_ids_);
  cudaFreeHost(trace_weights_);
}
size_t MinimaxModel::kv_block_bytes_static(const MinimaxConfig& c, int rank, int world,
                                           LatentFormat format, bool mtp) {
  validate_run(c, mtp && c.dspark ? c.dspark->query_rows() : 1, rank, world, 1, mtp, 0, format);
  return static_cast<size_t>(c.num_hidden_layers) * 128 *
             (2 * (c.num_key_value_heads / world) + 1) * MinimaxKvPool::head_bytes(format) +
         (mtp ? DraftKvPlanes::bytes(c.dspark->num_hidden_layers, c.dspark->local_kv_heads(world),
                                     c.dspark->head_dim, 128)
              : 0);
}
MinimaxModel::MemoryPlan MinimaxModel::plan_memory(const MinimaxConfig& c, int t, int64_t cache,
                                                   int rank, int world, LoaderResidency residency,
                                                   int slots, bool mtp, int decode,
                                                   LatentFormat format, bool serving, bool native) {
  validate_run(c, t, rank, world, slots, mtp, decode, format);
  const auto head = world > 1 ? LoaderHeadSharding::VocabSharded : LoaderHeadSharding::Full;
  const auto g = MinimaxLocalGeometry::from_config(c, rank, world, head);
  if (native && (c.hidden_size % 128 || g.local_inter % 128))
    throw std::invalid_argument("MiniMax native experts require 128-aligned TP slices");
  const int64_t cap = capacity(cache, t);
  const size_t H = c.hidden_size, V = g.lm_vocab_count, T = t;
  MemoryPlan plan;
  plan.context_tokens = std::min<int64_t>(cap, c.max_position_embeddings);
  if (c.vision) {
    plan.add("vision tower weights (BF16 and FP32, replicated)", c.vision->weight_bytes());
    plan.add("vision workspace (patches, attention tiles, projected rows)",
             c.vision->workspace_bytes());
  }
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
           MinimaxKvPool::cache_bytes(pool_shape(c, g, slots, cap, mtp, format)));
  size_t core = 0, pin = 0;
  session_core_plan_bytes(t, slots, c.hidden_size, g.lm_vocab_count, false, c.hidden_size, &core,
                          &pin, std::max({kDecodeRows, decode, slots}), serving);
  plan.add("session and residual buffers", core + T * H * 6,
           pin + static_cast<size_t>(c.num_moe_layers()) * T * c.num_experts_per_tok * 8);
  plan.add("sparse attention workspace",
           MinimaxAttentionLayer::scratch_bytes(c, g, t, cap, native, format));
  if (c.num_moe_layers() < c.num_hidden_layers)
    plan.add("dense MLP workspace",
             MinimaxMlp::scratch_bytes(c, static_cast<int>(g.local_dense_inter), t, native));
  if (c.num_moe_layers())
    plan.add("routed and shared expert workspace", MinimaxMoeLayer::scratch_bytes(c, g, t, native));
  if (mtp) {
    std::vector<std::pair<std::string, size_t>> entries;
    DFlash2Drafter::plan_entries(*c.dspark, world, decode, t, slots, V, false, &entries,
                                 minimax_dspark_weights_bytes(*c.dspark, world), native);
    for (const auto& [name, bytes] : entries) plan.add(name, bytes);
  }
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
void MinimaxModel::accumulate_draft_tap(int completed_layer, int rows) {
  if (!df_.on()) return;
  const int tap = df_.tap(completed_layer);
  if (tap < 0) return;
  const int H = cfg_.hidden_size;
  const auto* fc = df_.weights().fc + static_cast<size_t>(tap) * H * H;
  // One accumulation chain across prefill, resume and every verification
  // width. cuBLAS introduces memset graph nodes for this 6144-square FC
  // at 24 rows; serving captures must remain kernel-only for RDMA.
  launch_mma_gemv_bf16_f32(resid_, H, fc, df_.tap_gemm_out(), rows, H, H, H, stream_);
  const auto& taps = cfg_.dspark->target_layer_ids;
  dflash2_acc_f32(df_.acc(), df_.tap_gemm_out(), static_cast<int64_t>(rows) * H,
                  completed_layer == *std::min_element(taps.begin(), taps.end()) ? 0 : 1, stream_);
}
MinimaxModel::Outputs MinimaxModel::run_rows(const RowRun& run) {
  if (run.capture && loader_.residency() != LoaderResidency::Resident)
    throw std::logic_error("MiniMax graph requires resident weights");
  const int T = run.T, H = cfg_.hidden_size, K = cfg_.num_experts_per_tok;
  const auto in = begin_run(run);
  embed_gather_bf16(globals_.embed, in.tokens, resid_, T, H, stream_);
  apply_image_embeddings(resid_, run);
  Outputs out;
  const uint16_t* pending = nullptr;
  const bool traces = route_traces_ && !run.capture && (!run.decode || decode_row_traces_);
  int ordinal = 0;
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const auto& r = loader_.load_layer(layer);
    if (r.moe && !run.capture) moe_->prepare_table(layer, r.moe_w, stream_);
    norm(resid_, pending, r.input_norm, x_, T);
    // Checkpoint tap L names the OUTPUT of layer L. The next input norm
    // materializes that residual with one BF16 rounding while preserving
    // its unrounded sum for the target's norm. Upstream converts L to L+1
    // at its capture interface; it does not tap entry of layer L itself.
    accumulate_draft_tap(layer - 1, T);
    if (observer_ && !run.capture && df_.on() && df_.tap(layer - 1) >= 0)
      observer_(layer - 1, "draft_tap", resid_, T, stream_);
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
  // There is no next decoder layer for the final tap (59 in the release).
  // final_norm has saved its updated residual, before normalization, too.
  accumulate_draft_tap(cfg_.num_hidden_layers - 1, T);
  if (observer_ && !run.capture && df_.on() && df_.tap(cfg_.num_hidden_layers - 1) >= 0)
    observer_(cfg_.num_hidden_layers - 1, "draft_tap", resid_, T, stream_);
  const bool packed = project_head_rows(run, [&](int input, int output, int n) {
    project_head(h_ + static_cast<size_t>(input) * H,
                 logits_ + static_cast<size_t>(output) * lm_vocab_count_, n,
                 run.decode || input < run.decode_rows());
  });
  if (df_.on()) df_.store_features(T, in.req_ids, in.pos);
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
void MinimaxModel::project_head(const uint16_t* hidden, float* out, int rows, bool decode) {
  // Keep every verification width on the same FP32 reduction chain, including
  // checkpoint BF16 weights and shapes the lossless BF12 packer declines.
  // A cuBLAS dispatch here changes logits as occupancy changes and can insert
  // non-kernel graph nodes. The tensor-core form reads the head once per batch;
  // its BF12 and BF16 forms have identical chains at every supported width.
  // Packed-only residency retains the scalar chain: wider tensor-core calls
  // may need the raw weight. The drafter shares this head and GEMM object.
  gemm_.set_bf12_wide(decode);
  gemm_.set_decode_mma(decode && !gemm_.bf12_raw_released(globals_.lm_head));
  gemm_.set_kernel_only_rows(decode ? 1 : 0, decode ? max_decode_rows_ : 0);
  gemm_.matmul(hidden, globals_.lm_head, out, rows, lm_vocab_count_, cfg_.hidden_size, DType::BF16,
               GemmOut::F32, cfg_.hidden_size, gemm_ws_, gemm_ws_bytes_, stream_);
  gemm_.set_kernel_only_rows(0, 0);
  gemm_.set_decode_mma(false);
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
DFlash2CaptureCtx MinimaxModel::df_capture_ctx() const {
  return {step_tokens_,
          decode_rows_,
          graph_feed_rows_,
          graph_batch_requests_,
          graph_rows_per_request_,
          graph_batch_map_source_,
          graph_device_positions_,
          graph_device_tokens_};
}
}  // namespace dgpp
