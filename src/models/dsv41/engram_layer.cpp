#include "models/dsv41/engram_layer.hpp"
#include "kernels/glm_spec.hpp"

#include <cstring>

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "kernels/exl3_gemv.hpp"
#include "kernels/scale_gemm.hpp"

namespace dgpp {

struct Dsv41EngramLayer::StageArgs {
  const Dsv41EngramTables* tables = nullptr;
  // The walk's staging publishes each table's rows as it finishes them
  // (2026-10-20): ready[l] takes the staging's count after table l, the
  // consumer kernel of that layer spins on it (glm_stage_wait) instead of
  // the whole staging's join — the first Engram layer no longer waits for
  // the second's rows, and the decode graph keeps its one host node.
  uint64_t* ready = nullptr;  // pinned [layers], null: the prestage's (joined by event)
  uint64_t* count = nullptr;  // host [layers]: stagings published per table
  const int32_t* ids = nullptr;
  uint8_t* dst = nullptr;
  size_t layer_bytes = 0;
  int n = 0, entries = 0;
  std::atomic<int> error{0};
  std::string what;
};

void Dsv41EngramLayer::stage_callback(void* user) {
  StageArgs* a = static_cast<StageArgs*>(user);
  try {
    const int L = static_cast<int>(a->tables->tables.size());
    for (int l = 0; l < L; ++l) {
      a->tables->tables[static_cast<size_t>(l)]->gather(a->ids + static_cast<size_t>(l) * a->entries,
                                                        static_cast<int64_t>(L) * a->entries, a->tables->sel.data(),
                                                        a->tables->rows_local(), a->n,
                                                        a->dst + static_cast<size_t>(l) * a->layer_bytes);
      if (a->ready != nullptr) {
        ++a->count[l];
        __atomic_store_n(a->ready + l, a->count[l], __ATOMIC_RELEASE);
      }
    }
  } catch (const std::exception& e) {
    a->what = e.what();
    a->error.store(1, std::memory_order_release);
    DGPP_LOG_ERROR("Dsv41EngramLayer: the Engram staging failed: {}", e.what());
  }
}

size_t Dsv41EngramLayer::staging_bytes(const Dsv41TextConfig& cfg, int world, int max_tokens) {
  const size_t L = cfg.engram_layer_ids.size();
  const size_t entries = static_cast<size_t>(cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads;
  const size_t rows_local = static_cast<size_t>(cfg.engram_max_ngram_size - 1) * (cfg.engram_n_heads / world);
  const size_t row_bytes = static_cast<size_t>(cfg.engram_head_dim) + cfg.engram_head_dim / 32;
  // Both channels: the walk's own and the chunk-ahead prestage's (its ids
  // hold the prefix rows too).
  return static_cast<size_t>(max_tokens) * L * entries * 4 + L * static_cast<size_t>(max_tokens) * rows_local * row_bytes +
         static_cast<size_t>(max_tokens + 4) * L * entries * 4 + L * static_cast<size_t>(max_tokens) * rows_local * row_bytes;
}

namespace {
bool g_engram_prestage = true;
}
void Dsv41EngramLayer::set_prestage(bool on) { g_engram_prestage = on; }
bool Dsv41EngramLayer::prestage_enabled() { return g_engram_prestage; }
size_t Dsv41EngramLayer::device_bytes(const Dsv41TextConfig& cfg, int world, int max_tokens) {
  const size_t rows_local = static_cast<size_t>(cfg.engram_max_ngram_size - 1) * (cfg.engram_n_heads / world);
  return static_cast<size_t>(max_tokens) * rows_local * cfg.engram_head_dim * 2 +
         static_cast<size_t>(max_tokens) * (cfg.hc_mult + 1) * cfg.hidden_size * 2 +
         static_cast<size_t>(cfg.vocab_size) * 4 + 4096;
}

Dsv41EngramLayer::Dsv41EngramLayer(const Dsv41TextConfig& cfg, const Dsv41EngramTables& tables,
                                   const Dsv41EngramSidecar& sidecar, int max_tokens, int max_requests)
    : cfg_(cfg), tables_(tables), max_tokens_(max_tokens) {
  layers_ = static_cast<int>(cfg.engram_layer_ids.size());
  if (layers_ <= 0 || static_cast<int>(tables.tables.size()) != layers_)
    throw std::invalid_argument("Dsv41EngramLayer: the tables must cover every Engram layer");
  if (max_tokens <= 0 || max_requests <= 0) throw std::invalid_argument("Dsv41EngramLayer: shape");
  if (tables.head_dim != cfg.engram_head_dim || tables.heads != cfg.engram_n_heads ||
      tables.ngrams != cfg.engram_max_ngram_size - 1)
    throw std::invalid_argument("Dsv41EngramLayer: the tables' geometry disagrees with the config");
  entries_ = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads;
  // The hash constants.
  const size_t L = static_cast<size_t>(layers_);
  if (sidecar.token_map.size() != static_cast<size_t>(cfg.vocab_size) ||
      sidecar.multipliers.size() != L * cfg.engram_max_ngram_size ||
      sidecar.primes.size() != L * entries_ || sidecar.offsets.size() != L * entries_)
    throw std::invalid_argument("Dsv41EngramLayer: the sidecar's shape disagrees with the config");
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_token_map_), sidecar.token_map.size() * 4));
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_mult_), sidecar.multipliers.size() * 8));
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_primes_), sidecar.primes.size() * 8));
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_offsets_), sidecar.offsets.size() * 8));
  DGPP_CUDA_OK(cudaMemcpy(d_token_map_, sidecar.token_map.data(), sidecar.token_map.size() * 4, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_mult_, sidecar.multipliers.data(), sidecar.multipliers.size() * 8, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_primes_, sidecar.primes.data(), sidecar.primes.size() * 8, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_offsets_, sidecar.offsets.data(), sidecar.offsets.size() * 8, cudaMemcpyHostToDevice));
  hash_.token_map = d_token_map_;
  hash_.multipliers = d_mult_;
  hash_.primes = d_primes_;
  hash_.offsets = d_offsets_;
  hash_.pad_class = sidecar.pad_class;
  hash_.layers = layers_;
  hash_.max_ngram = cfg.engram_max_ngram_size;
  hash_.heads = cfg.engram_n_heads;
  hash_.vocab = cfg.vocab_size;
  // The pinned ids (device-mapped: the hash kernel writes them, the host
  // callback reads them) and the staged rows.
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_ids_), static_cast<size_t>(max_tokens) * L * entries_ * 4,
                             cudaHostAllocMapped));
  staged_layer_bytes_ = static_cast<size_t>(max_tokens) * tables.rows_local() * tables.tables[0]->row_bytes();
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&staged_), L * staged_layer_bytes_, cudaHostAllocMapped));
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&e_), static_cast<size_t>(max_tokens) * width_local() * 2));
  if (cfg.weights_exl3) {
    exl3_ws_bytes_ = exl3_gemm_ws_bytes_for((cfg.hc_mult + 1) * cfg.hidden_size, width_local());
    DGPP_CUDA_OK(cudaMalloc(&exl3_ws_, exl3_ws_bytes_));
    DGPP_CUDA_OK(cudaMemset(exl3_ws_, 0, exl3_ws_bytes_));  // the GEMV's arrival counters start at zero
  }
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&kv_),
                          static_cast<size_t>(max_tokens) * (cfg.hc_mult + 1) * cfg.hidden_size * 2));
  DGPP_CUDA_OK(cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking));
  DGPP_CUDA_OK(cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming));
  DGPP_CUDA_OK(cudaEventCreateWithFlags(&join_, cudaEventDisableTiming));
  // The decode walk's staging gathers one table per host callback, each
  // with its own join (2026-10-20): the first Engram layer's consumer waited
  // for every table's rows — on the fabric the gather's page faults put
  // 200-1200 us on one rank at 12 % of the Engram collectives. Two eager
  // slots per table; a capture takes fresh arguments.
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_ready_), static_cast<size_t>(layers_) * 8, cudaHostAllocMapped));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_late_), static_cast<size_t>(layers_) * 4, cudaHostAllocMapped));
  for (int l = 0; l < layers_; ++l) { h_ready_[l] = 0; h_late_[l] = 0; }
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_ready_), static_cast<size_t>(layers_) * 8));
  DGPP_CUDA_OK(cudaMemset(d_ready_, 0, static_cast<size_t>(layers_) * 8));
  ready_count_.assign(static_cast<size_t>(layers_), 0);
  stage_args_.push_back(std::make_unique<StageArgs>());
  stage_args_.push_back(std::make_unique<StageArgs>());
  // The prestage channel.
  const size_t pre_rows = static_cast<size_t>(max_tokens) + 4;
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_pre_tokens_), pre_rows * 8, cudaHostAllocDefault));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_pre_pos_), pre_rows * 8, cudaHostAllocDefault));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_pre_meta_), 2 * 4, cudaHostAllocDefault));
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_pre_tokens_), pre_rows * 8));
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_pre_pos_), pre_rows * 8));
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_pre_req_), pre_rows * 4));
  DGPP_CUDA_OK(cudaMemset(d_pre_req_, 0, pre_rows * 4));
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_pre_spans_), 2 * 4));
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_pre_ctx_), 4 * 4));
  {
    const int32_t pad[4] = {sidecar.pad_class, sidecar.pad_class, sidecar.pad_class, sidecar.pad_class};
    DGPP_CUDA_OK(cudaMemcpy(d_pre_ctx_, pad, 4 * 4, cudaMemcpyHostToDevice));
  }
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&pre_ids_), pre_rows * L * entries_ * 4, cudaHostAllocMapped));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&pre_staged_), L * staged_layer_bytes_, cudaHostAllocMapped));
  DGPP_CUDA_OK(cudaEventCreateWithFlags(&pre_fork_, cudaEventDisableTiming));
  DGPP_CUDA_OK(cudaEventCreateWithFlags(&pre_join_, cudaEventDisableTiming));
  pre_args_ = std::make_unique<StageArgs>();
}

Dsv41EngramLayer::~Dsv41EngramLayer() {
  cudaFree(exl3_ws_);
  if (side_) cudaStreamSynchronize(side_);
  if (fork_) cudaEventDestroy(fork_);
  if (join_) cudaEventDestroy(join_);
  if (h_ready_) cudaFreeHost(h_ready_);
  if (h_late_) cudaFreeHost(h_late_);
  if (d_ready_) cudaFree(d_ready_);
  if (pre_fork_) cudaEventDestroy(pre_fork_);
  if (pre_join_) cudaEventDestroy(pre_join_);
  if (side_) cudaStreamDestroy(side_);
  cudaFreeHost(h_ids_);
  cudaFreeHost(staged_);
  cudaFreeHost(h_pre_tokens_);
  cudaFreeHost(h_pre_pos_);
  cudaFreeHost(h_pre_meta_);
  cudaFreeHost(pre_ids_);
  cudaFreeHost(pre_staged_);
  cudaFree(d_pre_tokens_);
  cudaFree(d_pre_pos_);
  cudaFree(d_pre_req_);
  cudaFree(d_pre_spans_);
  cudaFree(d_pre_ctx_);
  cudaFree(e_);
  cudaFree(kv_);
  cudaFree(d_token_map_);
  cudaFree(d_mult_);
  cudaFree(d_primes_);
  cudaFree(d_offsets_);
}

void Dsv41EngramLayer::check_staged() const {
  for (const auto& a : stage_args_)
    if (a->error.load(std::memory_order_acquire))
      throw std::runtime_error("Dsv41EngramLayer: an Engram staging failed on the host: " + a->what);
  if (pre_args_ && pre_args_->error.load(std::memory_order_acquire))
    throw std::runtime_error("Dsv41EngramLayer: an Engram prestage failed on the host: " + pre_args_->what);
}

void Dsv41EngramLayer::stage(const int64_t* tokens, int rows, const int32_t* req_ids, const int64_t* pos,
                             const int32_t* req_spans, int num_requests, const int32_t* ctx, cudaStream_t stream,
                             const int64_t* host_ids, int64_t pos0, int req) {
  if (rows <= 0) return;
  if (rows > max_tokens_) throw std::invalid_argument("Dsv41EngramLayer: rows exceed max_tokens");
  // A new staging may follow a consumed or an abandoned one: the side
  // stream serializes the host gathers, so the pinned buffers are free
  // once the previous callback has run, which the fork below waits for.
  check_staged();
  use_pre_ = false;
  if (pre_pending_) {
    // The chunk a prestage gathered: its rows are on the prestage channel.
    // The key: the same slot, rows and first position, the same host ids
    // pointer AND the same first and last token (a prompt vector's address
    // can be reused by a later request).
    pre_pending_ = false;
    if (host_ids != nullptr && host_ids == pre_host_ids_ && rows == pre_rows_ && pos0 == pre_pos0_ && req == pre_req_ &&
        host_ids[0] == pre_first_ && host_ids[rows - 1] == pre_last_) {
      use_pre_ = true;
      staged_rows_ = rows;
      ++pre_claims_;
      return;
    }
  }
  int32_t* d_ids = nullptr;
  DGPP_CUDA_OK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_ids), h_ids_, 0));
  dsv41_engram_hash_ids_rows(tokens, rows, req_ids, pos, req_spans, num_requests, ctx, hash_, d_ids, stream);
  cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
  DGPP_CUDA_OK(cudaStreamIsCapturing(stream, &cs));
  StageArgs* a = nullptr;
  if (cs != cudaStreamCaptureStatusNone) {
    stage_args_.push_back(std::make_unique<StageArgs>());
    a = stage_args_.back().get();
  } else {
    a = stage_args_[static_cast<size_t>(eager_slot_)].get();
    eager_slot_ ^= 1;
  }
  a->tables = &tables_;
  a->ids = h_ids_;
  a->dst = staged_;
  a->layer_bytes = staged_layer_bytes_;
  a->n = rows;
  a->entries = entries_;
  a->ready = h_ready_;
  a->count = ready_count_.data();
  DGPP_CUDA_OK(cudaEventRecord(fork_, stream));
  DGPP_CUDA_OK(cudaStreamWaitEvent(side_, fork_, 0));
  DGPP_CUDA_OK(cudaLaunchHostFunc(side_, &Dsv41EngramLayer::stage_callback, a));
  DGPP_CUDA_OK(cudaEventRecord(join_, side_));
  staged_rows_ = rows;
}

void Dsv41EngramLayer::prestage(const int64_t* host_ids, int rows, int64_t pos0, int req, const int64_t* prefix,
                                int n_prefix, cudaStream_t stream) {
  if (!prestage_enabled() || host_ids == nullptr || rows <= 0 || rows > max_tokens_ || req < 0) return;
  if (n_prefix != cfg_.engram_max_ngram_size - 1 || prefix == nullptr) return;  // every row's context must be in the span
  cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
  DGPP_CUDA_OK(cudaStreamIsCapturing(stream, &cs));
  if (cs != cudaStreamCaptureStatusNone) return;  // the decode graphs stage on the main channel only
  check_staged();
  // The previous prestage, claimed or dropped: its gather ran or runs to
  // completion on the side stream; the hash below (and the host copies it
  // reads) wait for it before reusing the channel's buffers.
  pre_pending_ = false;
  DGPP_CUDA_OK(cudaStreamWaitEvent(stream, pre_join_, 0));
  const int n = n_prefix + rows;
  std::memcpy(h_pre_tokens_, prefix, static_cast<size_t>(n_prefix) * sizeof(int64_t));
  std::memcpy(h_pre_tokens_ + n_prefix, host_ids, static_cast<size_t>(rows) * sizeof(int64_t));
  for (int i = 0; i < n; ++i) h_pre_pos_[i] = pos0 - n_prefix + i;
  h_pre_meta_[0] = 0;
  h_pre_meta_[1] = n;
  DGPP_CUDA_OK(cudaMemcpyAsync(d_pre_tokens_, h_pre_tokens_, static_cast<size_t>(n) * 8, cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(d_pre_pos_, h_pre_pos_, static_cast<size_t>(n) * 8, cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(d_pre_spans_, h_pre_meta_, 2 * 4, cudaMemcpyHostToDevice, stream));
  int32_t* d_ids = nullptr;
  DGPP_CUDA_OK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_ids), pre_ids_, 0));
  // One span of n rows on the private slot 0 whose context is pad_class:
  // the prefix rows hash from nothing (unused), the chunk's rows from the
  // prefix and each other — the ids the walk's own stage() would produce.
  dsv41_engram_hash_ids_rows(d_pre_tokens_, n, d_pre_req_, d_pre_pos_, d_pre_spans_, 1, d_pre_ctx_, hash_, d_ids, stream);
  StageArgs* a = pre_args_.get();
  a->ready = nullptr;  // the prestage's channel joins by event
  a->tables = &tables_;
  a->ids = pre_ids_ + static_cast<size_t>(n_prefix) * static_cast<size_t>(layers_) * entries_;
  a->dst = pre_staged_;
  a->layer_bytes = staged_layer_bytes_;
  a->n = rows;
  a->entries = entries_;
  DGPP_CUDA_OK(cudaEventRecord(pre_fork_, stream));
  DGPP_CUDA_OK(cudaStreamWaitEvent(side_, pre_fork_, 0));
  DGPP_CUDA_OK(cudaLaunchHostFunc(side_, &Dsv41EngramLayer::stage_callback, a));
  DGPP_CUDA_OK(cudaEventRecord(pre_join_, side_));
  pre_host_ids_ = host_ids;
  pre_rows_ = rows;
  pre_pos0_ = pos0;
  pre_req_ = req;
  pre_first_ = host_ids[0];
  pre_last_ = host_ids[rows - 1];
  pre_pending_ = true;
  ++pre_publishes_;
}

void Dsv41EngramLayer::context_rows(const int64_t* tokens, int rows, const int32_t* req_ids, const int64_t* pos,
                                    const int32_t* req_spans, int num_requests, int32_t* ctx, int32_t* ctx_rows,
                                    cudaStream_t stream) {
  dsv41_engram_context_rows(tokens, rows, req_ids, pos, req_spans, num_requests, d_token_map_, cfg_.engram_max_ngram_size,
                            ctx, ctx_rows, stream);
}

void Dsv41EngramLayer::embed(int table_index, int rows, cudaStream_t stream) {
  if (rows <= 0) return;
  if (table_index < 0 || table_index >= layers_) throw std::invalid_argument("Dsv41EngramLayer: table index");
  if (staged_rows_ != rows) throw std::logic_error("Dsv41EngramLayer: embed() without a matching stage()");
  if (use_pre_) {
    DGPP_CUDA_OK(cudaStreamWaitEvent(stream, pre_join_, 0));
  } else if (table_index + 1 < layers_) {
    // An earlier table's rows: the wait kernel bumps the device's count of
    // stagings consumed for the table and spins until the host published
    // that many (the gather's own pace; 30 s is a failure).
    glm_stage_wait(h_ready_ + table_index, d_ready_ + table_index, h_late_ + table_index, 30'000'000'000ll, stream);
  } else {
    // The last table's consumer joins the staging's side stream (its
    // gather is the callback's last work; a capture needs the fork joined).
    DGPP_CUDA_OK(cudaStreamWaitEvent(stream, join_, 0));
  }
  uint8_t* d_staged = nullptr;
  DGPP_CUDA_OK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_staged), use_pre_ ? pre_staged_ : staged_, 0));
  dsv41_engram_gather_staged_bf16(d_staged + static_cast<size_t>(table_index) * staged_layer_bytes_, rows,
                                  tables_.rows_local(), cfg_.engram_head_dim, e_, stream);
}

void Dsv41EngramLayer::project(const Dsv41EngramLayerWeights& w, int rows, uint16_t* kv_dst, cudaStream_t stream,
                               bool prefill_tile, int r0) {
  if (rows <= 0) return;
  const int K = width_local();
  const int N = (cfg_.hc_mult + 1) * cfg_.hidden_size;
  const bool exl3 = w.wkv_x.tiles != nullptr;
  if (exl3 ? (w.wkv_x.rows != N || w.wkv_x.cols != K) : (w.wkv.payload == nullptr || w.wkv.rows != N || w.wkv.cols != K))
    throw std::invalid_argument("Dsv41EngramLayer: wkv geometry disagrees with the rank's rows");
  if (r0 < 0 || r0 + rows > max_tokens_) throw std::invalid_argument("Dsv41EngramLayer: projection rows out of range");
  uint16_t* dst = (kv_dst ? kv_dst : kv_) + static_cast<size_t>(r0) * N;
  if (exl3) {
    if (exl3_ws_ == nullptr) throw std::logic_error("Dsv41EngramLayer: an EXL3 wkv on a layer built without its workspace");
    launch_exl3_gemm_bf16(e_ + static_cast<size_t>(r0) * K, static_cast<size_t>(K), w.wkv_x, dst, rows, N, K, 0, stream,
                          exl3_ws_, exl3_ws_bytes_);
    return;
  }
  launch_scale_gemm_grid_bf16(e_ + static_cast<size_t>(r0) * K, static_cast<size_t>(K), w.wkv.payload, w.wkv.scales, dst,
                              rows, N, K, stream, 0, 5, 5, dense_mma_, nullptr, 0, prefill_tile);
}

void Dsv41EngramLayer::gate(const Dsv41EngramLayerWeights& w, uint16_t* x, const uint16_t* kv, int rows,
                            cudaStream_t stream) {
  if (rows <= 0) return;
  if (w.q_weight == nullptr || w.k_weight == nullptr) throw std::invalid_argument("Dsv41EngramLayer: null gate weights");
  dsv41_engram_gate_rows(x, kv, w.q_weight, w.k_weight, rows, cfg_.hc_mult, cfg_.hidden_size, cfg_.rms_norm_eps, stream);
}

}  // namespace dgpp
