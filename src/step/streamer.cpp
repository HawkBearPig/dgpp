// The streamer's host side (step/streamer.hpp).
#include "step/streamer.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "common/cuda_check.hpp"

namespace dgpp::step {

namespace {

template <typename T>
T* upload(const std::vector<T>& v) {
  T* d = nullptr;
  const size_t bytes = std::max<size_t>(v.size(), 1) * sizeof(T);
  DGPP_CUDA_OK(cudaMalloc(&d, bytes));
  if (!v.empty()) DGPP_CUDA_OK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

template <typename T>
void pinned(T** host, T** dev, size_t n) {
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(host), n * sizeof(T), cudaHostAllocMapped));
  std::memset(static_cast<void*>(*host), 0, n * sizeof(T));
  DGPP_CUDA_OK(cudaHostGetDevicePointer(reinterpret_cast<void**>(dev), *host, 0));
}

void store_release(uint32_t* p, uint32_t v) { std::atomic_ref<uint32_t>(*p).store(v, std::memory_order_release); }
void store_release(uint64_t* p, uint64_t v) { std::atomic_ref<uint64_t>(*p).store(v, std::memory_order_release); }
uint64_t load_acquire(const uint64_t* p) { return std::atomic_ref<const uint64_t>(*p).load(std::memory_order_acquire); }

}  // namespace

Streamer::Streamer(const DevicePlan& dp, const std::vector<const uint8_t*>& bases, const std::vector<LayerPlan>& plans)
    : dp_(dp) {
  if (bases.size() != dp.layers.size() || plans.size() != dp.layers.size())
    throw std::invalid_argument("Streamer: one base and one plan per layer");
  if (dp.layers.size() > static_cast<size_t>(kStepMaxLayers) || dp.phases.size() > static_cast<size_t>(kStepMaxPhases))
    throw std::invalid_argument("Streamer: the plan exceeds the control block's layers or phases");
  for (const LayerPlan& lp : plans)
    if (lp.groups > kStepMaxGroups) throw std::invalid_argument("Streamer: a layer's groups exceed the run list");
  for (const LayerPlan& lp : plans)
    for (const Phase& ph : lp.phases) phase_names_.push_back(ph.name);
  bases_ = upload(bases);
  layers_ = upload(dp.layers);
  phases_ = upload(dp.phases);
  segments_ = upload(dp.segments);
  pools_ = upload(dp.pools);
  DGPP_CUDA_OK(cudaMalloc(&sink_, sizeof(unsigned) * kStepMaxWorkers * 8));
  DGPP_CUDA_OK(cudaMemset(sink_, 0, sizeof(unsigned) * kStepMaxWorkers * 8));
  DGPP_CUDA_OK(cudaMalloc(&token_feed_, sizeof(int32_t) * 2 * kStepMaxRows));
  DGPP_CUDA_OK(cudaMemset(token_feed_, 0, sizeof(int32_t) * 2 * kStepMaxRows));
  DGPP_CUDA_OK(cudaMallocManaged(&ctl_, sizeof(StepCtl)));
  std::memset(ctl_, 0, sizeof(StepCtl));
  pinned(&desc_, &desc_dev_, static_cast<size_t>(kStepRingSlots));
  pinned(&verdict_, &verdict_dev_, static_cast<size_t>(kStepRingSlots));
  pinned(&host_seq_, &host_seq_dev_, 1);
  DGPP_CUDA_OK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  // The consumers' operands, when the plan has consumer pools of any class:
  // the routed experts (the MXFP4 forms), the shared expert and the fp8
  // projections (the fp8 forms). A plan with fp8 pools and no routed pool
  // launched with no dynamic shared memory and no activations once, and
  // faulted (2026-10-31).
  bool mxfp4 = false, fp8 = false;
  for (const DevPool& pool : dp.pools) {
    if (pool.kind == kPoolBytes) continue;
    consumers_ = true;
    if (pool_is_gate_up(pool.kind) && pool.kind != kPoolFp8GateUp) { hidden_ = pool.k; inter_ = pool.rows; mxfp4 = true; }
    else if (pool.kind == kPoolFp8GateUp) { if (!hidden_) hidden_ = pool.k; if (!inter_) inter_ = pool.rows; fp8 = true; }
    else if (pool.kind == kPoolFp8Matrix || pool.kind == kPoolFp8Down) fp8 = true;
  }
  if (consumers_) {
    if (mxfp4 && !step_stream_has_mxfp4_consumer(hidden_, inter_))
      throw std::invalid_argument("Streamer: the kernel has no MXFP4 consumer for hidden " + std::to_string(hidden_) + ", inter " +
                                  std::to_string(inter_));
    for (const DevPool& pool : dp.pools)
      if (pool.kind == kPoolFp8Matrix && !step_stream_has_fp8_matrix(pool.k, pool.rows))
        throw std::invalid_argument("Streamer: the kernel has no fp8 projection form for [" + std::to_string(pool.rows) + ", " +
                                    std::to_string(pool.k) + "]");
    consumer_smem_ = step_stream_consumer_smem(mxfp4 ? inter_ : 0, fp8);
    if (!hidden_) hidden_ = dp.max_k;  // projections only: the activations' width is the widest inner width
    if (!inter_) inter_ = 1;
    const size_t slots = static_cast<size_t>(kStepMaxRows) * 9;  // top_k up to 8, plus the shared slot
    // The activations: every consumer reads a token's row at its own inner
    // width from the same buffer, so it holds the widest.
    x_stride_ = std::max(hidden_, dp.max_k);
    DGPP_CUDA_OK(cudaMalloc(&x_, static_cast<size_t>(kStepMaxRows) * x_stride_ * 2));
    if (dp.proj_elems) {
      DGPP_CUDA_OK(cudaMalloc(&proj_, static_cast<size_t>(dp.proj_elems) * 2));
      DGPP_CUDA_OK(cudaMemset(proj_, 0, static_cast<size_t>(dp.proj_elems) * 2));
    }
    if (dp.part_elems) {
      DGPP_CUDA_OK(cudaMalloc(&proj_part_, static_cast<size_t>(dp.part_elems) * 4));
      DGPP_CUDA_OK(cudaMemset(proj_part_, 0, static_cast<size_t>(dp.part_elems) * 4));
    }
    DGPP_CUDA_OK(cudaMalloc(&act_, slots * inter_ * 2));
    DGPP_CUDA_OK(cudaMalloc(&out_, slots * hidden_ * 4));
    DGPP_CUDA_OK(cudaMalloc(&y_, static_cast<size_t>(kStepMaxRows) * hidden_ * 2));
    DGPP_CUDA_OK(cudaMemset(act_, 0, slots * inter_ * 2));
    DGPP_CUDA_OK(cudaMemset(out_, 0, slots * hidden_ * 4));
    DGPP_CUDA_OK(cudaMemset(y_, 0, static_cast<size_t>(kStepMaxRows) * hidden_ * 2));
    step_stream_init_x(x_, kStepMaxRows * x_stride_, 0x51e5u, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
  }
}

Streamer::~Streamer() {
  cudaStreamDestroy(stream_);
  cudaFreeHost(desc_);
  cudaFreeHost(verdict_);
  cudaFreeHost(host_seq_);
  cudaFree(const_cast<uint8_t**>(bases_));
  cudaFree(layers_);
  cudaFree(phases_);
  cudaFree(segments_);
  cudaFree(pools_);
  cudaFree(sink_);
  cudaFree(token_feed_);
  cudaFree(ctl_);
  cudaFree(x_);
  cudaFree(act_);
  cudaFree(out_);
  cudaFree(y_);
  cudaFree(proj_);
  cudaFree(proj_part_);
}

void Streamer::poison() { std::atomic_ref<unsigned>(ctl_->poison).store(1u, std::memory_order_release); }

StreamReport Streamer::run(const StreamerConfig& cfg) {
  if (cfg.steps < 1) throw std::invalid_argument("Streamer: at least one step");
  if (cfg.rows < 1 || cfg.rows > kStepMaxRows) throw std::invalid_argument("Streamer: rows out of range");
  std::memset(ctl_, 0, sizeof(StepCtl));
  for (int s = 0; s < kStepRingSlots; ++s) {
    desc_[s] = StepDescriptor{};
    verdict_[s] = StepVerdict{};
    verdict_[s].seq = ~uint64_t{0};
  }
  *host_seq_ = 0;
  std::vector<int32_t> feed(static_cast<size_t>(cfg.rows), 0);
  for (size_t i = 0; i < feed.size() && i < cfg.tokens.size(); ++i) feed[i] = cfg.tokens[i];

  StepStreamParams p;
  p.desc = desc_dev_;
  p.verdict = verdict_dev_;
  p.token_feed = token_feed_;
  p.host_seq = host_seq_dev_;
  p.max_steps = cfg.steps;
  p.bases = bases_;
  p.layers = layers_;
  p.phases = phases_;
  p.segments = segments_;
  p.pools = pools_;
  p.n_layers = static_cast<int>(dp_.layers.size());
  p.unit_bytes = dp_.unit_bytes;
  p.lead_units = dp_.lead_units;
  p.ctl = ctl_;
  p.sink = sink_;
  p.lookahead_layers = cfg.lookahead_layers;
  p.lead_bytes_per_us = cfg.lead_bytes_per_us;
  p.route_pool = cfg.route_pool;
  p.lead_cap_bytes = cfg.lead_cap_bytes;
  p.max_leaders = cfg.max_leaders;
  p.deadline_ns = static_cast<unsigned long long>(cfg.deadline_s * 1e9);
  p.pending_deadline_ns = static_cast<unsigned long long>(cfg.pending_deadline_s * 1e9);
  p.x = x_;
  p.act = act_;
  p.out = out_;
  p.y = y_;
  p.proj = proj_;
  p.proj_part = proj_part_;
  p.x_stride = x_stride_;
  p.hidden = hidden_;
  p.inter = inter_;
  p.consumer_smem = consumer_smem_;
  if (consumers_ && cfg.top_k > 8) throw std::invalid_argument("Streamer: the consumers' partials hold at most eight picks a row");
  cudaEvent_t e0 = nullptr, e1 = nullptr;
  DGPP_CUDA_OK(cudaEventCreate(&e0));
  DGPP_CUDA_OK(cudaEventCreate(&e1));
  DGPP_CUDA_OK(cudaEventRecord(e0, stream_));
  const cudaError_t launch = launch_step_stream(p, cfg.workers, stream_);
  if (launch != cudaSuccess) {
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    throw std::runtime_error(std::string("Streamer: launch failed: ") + cudaGetErrorString(launch) +
                             " (workers " + std::to_string(cfg.workers) + ", capacity " + std::to_string(step_stream_capacity(consumer_smem_)) + ")");
  }
  DGPP_CUDA_OK(cudaEventRecord(e1, stream_));

  // ---- the host's side of the protocol: one descriptor at a time, in order
  StreamReport r;
  uint64_t fold_seed = 0x9e3779b97f4a7c15ull;
  (void)fold_seed;
  uint64_t last_end_ns = 0;
  bool ended = false;
  for (int step = 0; step < cfg.steps && !ended; ++step) {
    const uint64_t seq = cfg.first_seq + static_cast<uint64_t>(step);
    const int slot = step & 1;
    StepDescriptor& d = desc_[slot];
    // The slot is the host's: its previous verdict was read two steps ago.
    store_release(&d.seq, seq);
    if (cfg.between_steps) cfg.between_steps(step);
    d.rows = static_cast<uint32_t>(cfg.rows);
    d.top_k = static_cast<uint32_t>(cfg.top_k);
    d.seed = cfg.seed;
    d.epoch = cfg.epoch;
    d.gather_seq = cfg.gather_delay_us > 0 ? seq + 1 : 0;
    d.tokens_from_host = step == 0 ? 1u : 0u;
    if (step == 0)
      for (int i = 0; i < cfg.rows; ++i) d.tokens[i] = feed[static_cast<size_t>(i)];
    const bool skip = step == cfg.skip_at_step;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    store_release(&d.state, skip ? kStepSkip : kStepExecute);
    std::thread gather;
    if (!skip && cfg.gather_delay_us > 0) {
      const uint64_t publish = seq + 1;
      const double delay = cfg.gather_delay_us;
      gather = std::thread([this, publish, delay] {
        std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long>(delay)));
        store_release(host_seq_, publish);
      });
    }
    // The verdict, or the kernel's end.
    for (;;) {
      if (load_acquire(&verdict_[slot].seq) == seq) break;
      if (cudaStreamQuery(stream_) == cudaSuccess) { ended = true; break; }
      std::this_thread::yield();
    }
    if (gather.joinable()) gather.join();
    if (load_acquire(&verdict_[slot].seq) != seq) break;  // the kernel ended before this step's verdict
    std::atomic_thread_fence(std::memory_order_acquire);
    StepVerdict v = verdict_[slot];
    // The host's replica of the token feed and its fold.
    if (v.status == kVerdictOk) {
      uint64_t fold = 0x9e3779b97f4a7c15ull;
      for (int i = 0; i < cfg.rows; ++i) {
        feed[static_cast<size_t>(i)] = step_next_token(feed[static_cast<size_t>(i)], seq, i);
        fold = step_fold_mix(fold, static_cast<uint64_t>(static_cast<uint32_t>(feed[static_cast<size_t>(i)])));
      }
      if (fold != v.tokens_fold) r.folds_match = false;
      r.pending_us.push_back(last_end_ns ? static_cast<double>(v.published_ns - last_end_ns) / 1e3 : 0.0);
      r.bind_us.push_back(static_cast<double>(v.begin_ns - v.published_ns) / 1e3);
      r.step_us.push_back(static_cast<double>(v.end_ns - v.begin_ns) / 1e3);
      last_end_ns = v.end_ns;
    } else {
      ended = true;
    }
    r.verdicts.push_back(v);
    // The slot returns to the host; reset its state for the next use.
    store_release(&d.state, kStepPending);
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  DGPP_CUDA_OK(cudaEventElapsedTime(&r.kernel_ms, e0, e1));
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  r.status = ctl_->status;
  r.steps_run = ctl_->steps_run;
  switch (r.status & 0xff) {
    case kStepOk: r.status_text = "ok"; break;
    case kStepDeadline:
      r.status_text = (r.status >> 8) == 0xfffu ? "deadline waiting for the host's descriptor"
                                                : "deadline in phase " + std::to_string(r.status >> 8);
      break;
    case kStepPoisoned: r.status_text = "poisoned"; break;
    case kStepBadUnit: r.status_text = "a unit outside its phase"; break;
    default: r.status_text = "unknown";
  }
  if (!r.verdicts.empty() && r.verdicts.back().status == kVerdictSkipped) r.status_text += ", skipped";
  // The last completed step's phase trace.
  unsigned long long origin = ~0ull;
  for (const DevLayer& dl : dp_.layers)
    for (int P = dl.phase_begin; P < dl.phase_begin + dl.phase_count; ++P) origin = std::min(origin, ctl_->stamp_begin[P]);
  r.run_counts.assign(ctl_->run_count, ctl_->run_count + dp_.layers.size());
  for (size_t l = 0; l < dp_.layers.size(); ++l) {
    const DevLayer& dl = dp_.layers[l];
    unsigned long long first = ~0ull, last = 0;
    double bytes = 0;
    for (int P = dl.phase_begin; P < dl.phase_begin + dl.phase_count; ++P) {
      const DevPhase& ph = dp_.phases[static_cast<size_t>(P)];
      PhaseTiming t;
      t.layer = static_cast<int>(l);
      t.phase = P;
      t.name = phase_names_[static_cast<size_t>(P)];
      t.stream = ph.kind == 1;
      t.begin_us = static_cast<double>(ctl_->stamp_begin[P] - origin) / 1e3;
      t.end_us = static_cast<double>(ctl_->stamp_end[P] - origin) / 1e3;
      t.units = ctl_->done[P];
      t.pf_units = ctl_->pf_units[P];
      if (ctl_->idle_first[P] && ctl_->idle_first[P] < ctl_->stamp_end[P])
        t.drained_us = static_cast<double>(ctl_->stamp_end[P] - ctl_->idle_first[P]) / 1e3;
      t.busy_us = static_cast<double>(ctl_->busy_ns[P]) / 1e3;
      t.handout_us = static_cast<double>(ctl_->handout_ns[P]) / 1e3;
      t.lead_in_mb = static_cast<double>(ctl_->lead_in_bytes[P]) / 1e6;
      t.staged_mb = static_cast<double>(ctl_->pf_staged[P]) / 1e6;
      t.wait_us = static_cast<double>(ctl_->wait_ns[P]) / 1e3;
      t.ahead_mb = static_cast<double>(ctl_->ahead_begin[P]) / 1e6;
      t.cursor_layer = ctl_->cursor_begin[P] < dp_.phases.size() ? dp_.phases[ctl_->cursor_begin[P]].layer : -1;
      t.look_blocked = ctl_->look_blocked[P];
      t.grabs = ctl_->grabs[P];
      first = std::min(first, ctl_->stamp_begin[P]);
      last = std::max(last, ctl_->stamp_end[P]);
      for (int i = 0; i < ph.seg_count; ++i) bytes += static_cast<double>(dp_.segments[static_cast<size_t>(ph.seg_begin + i)].bytes);
      for (int i = 0; i < ph.n_pools; ++i) {
        const DevPool& pool = dp_.pools[static_cast<size_t>(ph.pools[i])];
        bytes += static_cast<double>(pool.bytes) * (pool.group_kind == kGroupRouted ? ctl_->run_count[l] : 1);
      }
      r.phases.push_back(std::move(t));
    }
    r.layer_us.push_back(static_cast<double>(last - first) / 1e3);
    r.layer_bytes.push_back(bytes);
  }
  r.lead_bytes = static_cast<double>(ctl_->pf_bytes);
  r.workers = cfg.workers;
  return r;
}

}  // namespace dgpp::step
