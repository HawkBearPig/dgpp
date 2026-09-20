#pragma once
// Graph-based scheduler adapter shared by serving and diagnostic tools.
// It coordinates model capture/replay, distributed picks, speculative
// state, sampling fallbacks and prefix snapshots. All ranks must select
// the same graph variant and execute collectives in the same order.
//
// This header depends on CUDA and bus/verbs types. Host-only service tests
// use a fake engine instead.
#include <algorithm>
#include <array>
#include <deque>
#include <thread>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/log.hpp"
#include "common/process_memory.hpp"
#include "engine/decode_outputs.hpp"
#include "engine/eager_engine.hpp"
#include "engine/graph_check.hpp"
#include "engine/image_prefill.hpp"
#include "engine/prefix_arena.hpp"
#include "engine/speculative.hpp"
#include "engine/step_timing.hpp"
#include "engine/tp_bus.hpp"
#include "engine/verify_schedule.hpp"
#include "kernels/glm_spec.hpp"
#include "net/bus_kernel.hpp"
#include "net/collective_bus.hpp"

namespace dgpp {

// Pins the process's current pages BEFORE the model is constructed — the
// decode loop's host state (tokenizer tables, the bus, this binary) once
// had to survive a load phase that drove every box to its memory
// watermark and had the kernel swapping exactly those pages out
// (2026-09-02: a ~10 ms swap-in fault per shared token, in lockstep across
// ranks). The one-pass loader removed that pressure, and a 1000-step run
// with the pin OFF was indistinguishable (p99 46, 0 stalls, no swap
// traffic) — so this is a belt-and-braces safety net, not a requirement:
// a box whose RLIMIT_MEMLOCK refuses it serves exactly as well, and the
// refusal is informational. Before construction on purpose: the loader's
// checkpoint mmaps do not exist yet, so MCL_CURRENT cannot try to pin
// them. DGPP_MLOCK=off skips the attempt.
inline void pin_serving_process(int rank) {
  if (const char* m = std::getenv("DGPP_MLOCK"); m && std::string(m) == "off") {
    DGPP_LOG_INFO("rank {}: process memory not locked (DGPP_MLOCK=off)", rank);
    return;
  }
  std::string why;
  size_t locked = 0;
  if (lock_process_memory(&why, &locked))
    DGPP_LOG_INFO("rank {}: process memory locked (mlockall MCL_CURRENT, "
                  "{:.0f} MiB)",
                  rank, static_cast<double>(locked) / (1024.0 * 1024.0));
  else
    DGPP_LOG_INFO("rank {}: process memory not locked ({}); serving "
                  "proceeds — the pin is optional",
                  rank, why);
}

// Real-mesh budgets, not loopback budgets (found by the first fabric
// gate run, 2026-08-30: a cold peer's first boundary waits behind
// seconds of cold NVMe weight streaming). The lane watchdog arms from
// each POST, so these bounds measure genuine in-flight stalls only.
// launch_consumers=false: the app drives every collective itself.
// `lat_slot_bytes`: the latency slot sized for the model's widest decode
// boundary — the decode-row ceiling's bf16 rows of its boundary width (GLM: 4096; a
// model whose boundary carries more than the hidden row says so here).
inline net::BusOptions fabric_bus_options(int rank, int world, uint16_t port,
                                          const std::string& peer,
                                          int rendezvous_timeout_ms,
                                          size_t lat_slot_bytes) {
  net::BusOptions o;
  o.world_size = world;
  o.my_rank = rank;
  o.rendezvous_port = port;
  o.rendezvous_host = rank == 0 ? "" : peer;
  o.rendezvous_timeout_ms = rendezvous_timeout_ms;
  o.lat_slots = 8;
  // Phase 2 folds the full fixed row batch in one collective (GLM: eight
  // bf16 hidden-4096 rows = 64 KiB). This is only a transport sizing knob;
  // the latency protocol and element-driven kernels are unchanged.
  o.lat_slot_bytes = lat_slot_bytes;
  o.bulk_slots = 8;
  o.bulk_slot_bytes = 262144;
  o.qp_depth = 1024;
  o.completion_timeout_ms = 120000;
  o.consumer_deadline_s = 60.0;
  o.launch_consumers = false;
  return o;
}

// The fabric pick: this rank's vocab-slice argmax, then the winner
// through bus_greedy_pick — the collective every rank joins in the
// same tick, with the readback invariant that makes a corrupt
// broadcast loud on the exact rank. The float row is hoisted inside
// the returned closure (no per-token device-adjacent allocation; the
// pick scratch is caller-pinned BEFORE the world forms).
inline DecodePick make_fabric_pick(net::CollectiveBus* bus,
                                               int rank, int world,
                                               uint16_t* pick_scratch,
                                               int64_t vocab,
                                               int pick_timeout_ms = 60000) {
  return [bus, rank, world, pick_scratch, vocab, pick_timeout_ms](
             const DecodeOutputs& out) -> int32_t {
    const sample::Candidate local = sample::local_max(
        out.logits.data(), static_cast<int>(out.lm_vocab_count),
        out.lm_vocab_begin);
    const int32_t t = bus_greedy_pick(*bus, rank, world, local,
                                      pick_scratch, pick_timeout_ms);
    if (t < 0 || t >= vocab)
      throw std::runtime_error("fabric pick out of range: " +
                               std::to_string(t));
    return t;
  };
}

// The fabric SAMPLER (M6 6b, the eager engines): the request's penalized
// slice through the exact candidate/LSE fold (bus_sampling_prefix at
// kSamplingCandidates per rank); when the prefix cannot decide, the exact
// fallback — the penalized slices gathered as one bulk collective and the
// same decision over the complete list under the transported normalizer,
// with the same draw — and rank 0's digest echoed either way. Both scratch
// buffers are caller-pinned BEFORE the world forms:
// fabric_sampling_prefix_scratch_elems(world) and
// sampling_gather_scratch_elems(vocab) words.
inline size_t fabric_sampling_prefix_scratch_elems(int world) {
  return sampling_prefix_scratch_elems(world, kSamplingCandidates);
}

inline DecodeSample make_fabric_sample(
    net::CollectiveBus* bus, int rank, int world, uint16_t* prefix_scratch,
    uint16_t* gather_scratch, int64_t vocab, int pick_timeout_ms = 60000) {
  if (bus == nullptr || prefix_scratch == nullptr || gather_scratch == nullptr)
    throw std::invalid_argument("fabric sample: null bus/scratch");
  auto gather_buffer = std::make_shared<std::vector<float>>();
  return [bus, rank, world, prefix_scratch, gather_scratch, vocab,
          pick_timeout_ms, gather_buffer](
             const DecodeOutputs& out,
             const sample::Params& p, sample::Rng& rng,
             const std::vector<int32_t>& context,
             const text::TokenMask* mask, const float* bias) -> sample::Result {
    step_timing::Scope tick(step_timing::kPick);
    const sample::Result r = bus_sample_row(
        *bus, rank, world, out.logits.data(),
        static_cast<int>(out.lm_vocab_count), out.lm_vocab_begin,
        static_cast<int>(vocab), p, rng, context, kSamplingCandidates,
        prefix_scratch, gather_scratch, pick_timeout_ms, gather_buffer.get(),
        mask, bias);
    if (r.token < 0 || r.token >= vocab)
      throw std::runtime_error("fabric sample out of range: " +
                               std::to_string(r.token));
    return r;
  };
}

// The fabric row deciders of the eager SAMPLED speculator (glm_speculative.hpp
// SampledSpeculator): row 0's accept/residual and row 1's sample, each one
// fold plus the gather fallback plus rank 0's digest, in the same order on
// every rank.
inline SpecRow0 make_fabric_spec_row0(
    net::CollectiveBus* bus, int rank, int world, uint16_t* prefix_scratch,
    uint16_t* gather_scratch, int64_t vocab, int pick_timeout_ms = 60000) {
  auto gather_buffer = std::make_shared<std::vector<float>>();
  return [bus, rank, world, prefix_scratch, gather_scratch, vocab,
          pick_timeout_ms, gather_buffer](
             const DecodeOutputs& row0, int32_t draft,
             const sample::Params& p, sample::Rng& rng,
             const std::vector<int32_t>& context)
             -> sample::SpecPrefixDecision {
    step_timing::Scope tick(step_timing::kPick);
    return bus_spec_accept(*bus, rank, world, row0.logits.data(),
                           static_cast<int>(row0.lm_vocab_count),
                           row0.lm_vocab_begin, static_cast<int>(vocab), draft,
                           p, rng, context, kSamplingCandidates,
                           prefix_scratch, gather_scratch, pick_timeout_ms,
                           gather_buffer.get());
  };
}

// M6.6a Phase 2: adaptive scalar/row-batched graphs. Each physical request
// slot lazily gets the exact Phase-1 scalar capture; a fixed batch covers a
// prefix of the slots, with rows [slot, speculative-row] (T=1 plain, T=2
// MTP). Below the crossover the adapter replays each live scalar variant;
// at and above it, one batch replay advances them all. Closed batch slots
// derive position -1 and remain padding. Both paths return one independently
// judged token vector per live slot.
// THE BATCH FAMILY. The 8-row batch at two live requests
// stepped in 96.5 ms against 83 for two scalar replays (41.5 solo): the
// padding rows pay the fixed graph's stateless compute, expert reads and
// collective width, which is why the crossover sat at four. So the batch
// is recorded per occupancy — 2 slots, 3 slots, and every slot — and the
// smallest family whose slots cover the live ones replays. The scheduler
// fills the lowest free slot, so two live requests usually sit in 0 and
// 1 and pay four rows; a hole from churn (slots 0 and 2) falls to the next
// family up. Each family has its own two parities, end events, stage and
// verdict indices (slots_ + family) and bus variants (2 * slots_ + 2 *
// family + parity). A single live request never touches a batch.
// SAMPLING (M6 6b, the device path): with both sampler scratch tables the
// plain (T=1) graphs carry the on-device sampling pick
// (kernels/glm_sample_pick.hpp) — per-slot device specs and count tables,
// the exact local top-k + slice normalizer, and the verdict that decides
// bit for bit what the host oracle decides or flags a fallback. The
// candidate width is the widest that fits the fixed batch's rows in one
// latency slot (at most kSamplingCandidates). A fallback is served between
// windows exactly as the eager engine's: the penalized row to the host, the
// bulk gather, the complete decision under the transported normalizer with
// the reserved draw, rank 0's digest — and the true token overrides the
// provisional one in the graph's token feed before the next replay. The
// MTP graphs sample too (the T=2 verify's accept test and row-1 sample on
// the device); a fallback there rolls the in-graph draft back to its ring
// snapshot, decides on the host (re-running the verify's second row eagerly
// when a provisionally rejected draft turns out to stand), re-drafts
// eagerly on the true rows and reseeds the [next, draft] feed.
template <class Model>
class GraphEngineAdapter final : public sched::SchedulerEngine {
 private:
  // A replay in flight (2026-09-06, the pipelined replay): launched, its
  // end not yet settled.
  struct Replay {
    int req = -1;             // the scalar slot, or -1 for the batch
    bool batched = false;
    int family = -1;          // the batch family (batched only)
    int parity = 0;
    std::vector<int> reqs;    // the live slots the replay decided for
    std::vector<int> redrafted;  // slots the host re-drafted: skip their draft verdicts
    uint64_t verdict_seq = 0; // the pinned sequence its verdict node publishes
    // The scheduled verify depth (scalar MTP replays only): the option
    // replayed (-1: the full-depth variant, every replay before 2026-09-14)
    // and the rows its verify decided (1 + the drafts verified).
    int depth_option = -1;
    int rows = 0;
  };

 public:
  // `grammar_vocab` (optional, M6 6g): the tokenizer's token table; with
  // it and the device sampler the adapter constrains the pick per slot
  // (configure_constraint) — the masks live in a device table the captures
  // bake in, staged before every replay.
  GraphEngineAdapter(Model* model, net::CollectiveBus* bus,
                        int rank, int world, uint16_t* pick_scratch,
                        int64_t vocab, int pick_timeout_ms = 60000,
                        int batch_min_live = 4,
                        uint16_t* sample_prefix_scratch = nullptr,
                        uint16_t* sample_gather_scratch = nullptr,
                        int sampling_candidates_cap = kSamplingCandidates,
                        const text::GrammarVocab* grammar_vocab = nullptr,
                        int prefix_slots = 0, int mtp_depth = 1)
      : model_(model),
        bus_(bus),
        rank_(rank),
        world_(world),
        vocab_(vocab),
        pick_timeout_ms_(pick_timeout_ms),
        sample_prefix_scratch_(sample_prefix_scratch),
        sample_gather_scratch_(sample_gather_scratch),
        grammar_vocab_(grammar_vocab),
        arena_(model, model == nullptr ? 0 : prefix_slots) {
    if (model_ == nullptr || bus_ == nullptr || pick_scratch == nullptr)
      throw std::invalid_argument("graph engine: null model/bus/pick scratch");
    if constexpr (requires { model_->set_prefill_monitor(prefill_monitor()); })
      model_->set_prefill_monitor(prefill_monitor());
    static_assert(kPickMaxRequests <= sched::SchedulerEngine::DecodeBatchStats::kMaxSlots);
    slots_ = model_->max_session_requests();
    // The verify's rows: the pending token plus `mtp_depth` drafts
    // (2026-09-06; depth 1 is the two-row step as built).
    if (mtp_depth < 1 || 1 + mtp_depth > kSpecRows)
      throw std::invalid_argument("graph engine: mtp depth must be in [1, " +
                                  std::to_string(kSpecRows - 1) + "]");
    rows_per_request_ = model_->mtp_enabled() ? 1 + mtp_depth : 1;
    depth_ = rows_per_request_ - 1;
    if (slots_ < 1)
      throw std::invalid_argument("graph engine: no request slots");
    prefills_.resize(static_cast<size_t>(slots_));
    // Build the fitting slot prefixes even when the configured capacity
    // exceeds one verify pass. Sparse live sets beyond those prefixes use
    // scalar replays; a family must cover every live physical slot.
    // Past depth 1 the batch needs the family's batched draft chain.
    max_rows_ = model_->max_decode_rows();
    if (slots_ > kPickMaxRequests || rows_per_request_ > max_rows_)
      throw std::invalid_argument("graph engine: request or verify capacity exceeds the model/picker limit");
    const int batch_slots = std::min(slots_, max_rows_ / rows_per_request_);
    batch_unavailable_ =
        batch_slots < 2 ||
        (depth_ > 1 && !Model::kBatchedDraftChain);
    if (!batch_unavailable_) {
      // 2 and 3 slots, then 4 and 6 for the wider recipes (the full GLM-5.3's
      // sixteen-row batch, 2026-09-13: eight slots at depth 1 replayed four
      // live requests on the sixteen-row family — 26–28 tok/s aggregate
      // against the four-slot template's 38–43 — until the 4-slot family
      // covered them), then every slot. The bus bounds the variants:
      // 2 x slots + 2 x families (x the scheduled depth options) <= kBusMaxGraphVariants (64).
      for (const int k : {2, 3, 4, 6})
        if (k < batch_slots) {
          BatchFamily f;
          f.requests = k;
          families_.push_back(f);
        }
      BatchFamily full;
      full.requests = batch_slots;
      families_.push_back(full);
      family_steps_.assign(families_.size(), 0);
    }
    slot_mtp_attempts_.assign(static_cast<size_t>(slots_), {});
    slot_mtp_accepts_.assign(static_cast<size_t>(slots_), {});
    prefill_pick_ = make_fabric_pick(bus_, rank_, world, pick_scratch, vocab_,
                                     pick_timeout_ms_);
    if (sample_prefix_scratch_ != nullptr && sample_gather_scratch_ != nullptr) {
      // The planned width, or a lower cap (the loopback gates narrow it so
      // the tiny fixture vocabulary still exercises the fallback).
      if (sampling_candidates_cap < 1 ||
          sampling_candidates_cap > kSampleMaxCandidates)
        throw std::invalid_argument("graph engine: sampling candidates cap");
      candidates_ = device_sample_candidates_that_fit(
          slots_ * rows_per_request_, world_,
          bus_->slot_bytes(net::BusMessageClass::kLatency),
          sampling_candidates_cap);
      if (candidates_ > 0) {
        sampling_ = true;
        prefill_sample_ = make_fabric_sample(
            bus_, rank_, world_, sample_prefix_scratch_,
            sample_gather_scratch_, vocab_, pick_timeout_ms_);
        DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_specs_),
                                sizeof(SampleSpec) * slots_));
        DGPP_CUDA_OK(cudaMemset(d_specs_, 0, sizeof(SampleSpec) * slots_));
        DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_draft_specs_),
                                sizeof(SampleSpec) * slots_));
        DGPP_CUDA_OK(cudaMemset(d_draft_specs_, 0, sizeof(SampleSpec) * slots_));
        DGPP_CUDA_OK(cudaMallocHost(reinterpret_cast<void**>(&h_specs_),
                                    sizeof(SampleSpec) * slots_));
        for (int i = 0; i < slots_; ++i) h_specs_[i] = SampleSpec{};
        DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_counts_),
                                sizeof(int32_t) * slots_ * vocab_));
        DGPP_CUDA_OK(cudaMemset(d_counts_, 0, sizeof(int32_t) * slots_ * vocab_));
        // The logit bias table: one dense row per slot.
        DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_bias_),
                                sizeof(float) * slots_ * vocab_));
        DGPP_CUDA_OK(cudaMemset(d_bias_, 0, sizeof(float) * slots_ * vocab_));
        DGPP_CUDA_OK(cudaMallocHost(reinterpret_cast<void**>(&h_bias_row_),
                                    sizeof(float) * vocab_));
        DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_prompt_ids_),
                                sizeof(int64_t) * model_->max_context()));
        DGPP_CUDA_OK(cudaMallocHost(reinterpret_cast<void**>(&h_prompt_ids_),
                                    sizeof(int64_t) * model_->max_context()));
        DGPP_CUDA_OK(cudaMallocHost(
            reinterpret_cast<void**>(&h_fallback_row_),
            sizeof(float) * model_->lm_vocab_count()));
        // The drafts' proposals: what distribution each draft
        // was drawn from, written by the draft pick and read by the next
        // step's verify (kernels/sample_pick.hpp). The pinned mirror is the
        // host fallback's copy.
        if (model_->mtp_enabled()) {
          const size_t n = static_cast<size_t>(slots_) * kSampleProposalSlots;
          DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_proposals_),
                                  sizeof(DraftProposal) * n));
          DGPP_CUDA_OK(cudaMemset(d_proposals_, 0, sizeof(DraftProposal) * n));
          DGPP_CUDA_OK(cudaMallocHost(reinterpret_cast<void**>(&h_proposals_),
                                      sizeof(DraftProposal) * n));
          for (size_t i = 0; i < n; ++i) h_proposals_[i] = DraftProposal{};
          draft_sampled_ = proposal_drafts_enabled();
        }
        // The verify rows as the pick left them (penalized, masked), kept
        // for the host's MTP fallback: the in-graph draft's head reuses the
        // logits buffer, so after a replay the buffer holds the DRAFT's
        // rows — a fallback deciding over those decides over the wrong
        // distribution. A copy kernel node after the verify pick, before
        // the draft, preserves them ([slots][rows_per_request][count]).
        if (model_->mtp_enabled())
          DGPP_CUDA_OK(cudaMalloc(
              reinterpret_cast<void**>(&d_verify_logits_),
              sizeof(float) * static_cast<size_t>(slots_) * rows_per_request_ *
                  model_->lm_vocab_count()));
        // The token masks: one row per physical decode row, the header
        // word 0 (unconstrained) until a grammar stages one.
        mask_stride_ = device_sample_mask_words(static_cast<int>(vocab_));
        const size_t mask_words =
            static_cast<size_t>(slots_) * rows_per_request_ * mask_stride_;
        DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_masks_),
                                sizeof(uint32_t) * mask_words));
        DGPP_CUDA_OK(cudaMemset(d_masks_, 0, sizeof(uint32_t) * mask_words));
        DGPP_CUDA_OK(cudaMallocHost(reinterpret_cast<void**>(&h_masks_),
                                    sizeof(uint32_t) * mask_words));
        std::fill_n(h_masks_, mask_words, 0u);
        DGPP_LOG_INFO(
            "rank {}: graph engine samples on the device — {} candidates per "
            "rank ({} rows x {} ranks in a {}-byte latency slot; the planned "
            "width is {}); constrained decoding {}",
            rank_, candidates_, slots_ * rows_per_request_, world_,
            bus_->slot_bytes(net::BusMessageClass::kLatency),
            sampling_candidates_cap,
            grammar_vocab_ != nullptr && grammar_vocab_->usable()
                ? "available"
                : "unavailable (no grammar vocabulary)");
      } else {
        DGPP_LOG_WARN(
            "rank {}: graph engine cannot sample — not even one candidate per "
            "rank fits the latency slot for {} rows; greedy only",
            rank_, slots_ * rows_per_request_);
      }
    }
    picker_ = std::make_unique<DevicePicker>(*bus_, rank_, world,
                                                pick_timeout_ms_, candidates_,
                                                max_rows_);
    params_.assign(static_cast<size_t>(slots_), sample::greedy_params());
    rng_.assign(static_cast<size_t>(slots_), sample::Rng{});
    grammar_.resize(static_cast<size_t>(slots_));
    bias_.resize(static_cast<size_t>(slots_));
    masks_.assign(static_cast<size_t>(slots_) * rows_per_request_,
                  text::TokenMask{});
    context_.assign(static_cast<size_t>(slots_), {});
    report_.assign(static_cast<size_t>(slots_), false);
    pending_logprobs_.assign(static_cast<size_t>(slots_), {});
    slot_sampled_.assign(static_cast<size_t>(slots_), 0);
    slot_fallbacks_.assign(static_cast<size_t>(slots_), 0);
    pending_.assign(static_cast<size_t>(slots_), -1);
    drafts_.assign(static_cast<size_t>(slots_),
                   std::vector<int32_t>(static_cast<size_t>(depth_), -1));
    hop_slot_.assign(static_cast<size_t>(slots_), -1);
    hop_position_.assign(static_cast<size_t>(slots_), 0);
    live_.assign(static_cast<size_t>(slots_), false);
    reserved_.assign(static_cast<size_t>(slots_), false);
    scalar_execs_.assign(static_cast<size_t>(slots_), {{nullptr, nullptr}});
    scalar_parity_.assign(static_cast<size_t>(slots_), 0);
    end_events_.assign(static_cast<size_t>(slots_), {{nullptr, nullptr}});
    for (int req = 0; req < slots_; ++req)
      for (int p = 0; p < 2; ++p)
        DGPP_CUDA_OK(cudaEventCreateWithFlags(
            &end_events_[static_cast<size_t>(req)][static_cast<size_t>(p)],
            cudaEventDisableTiming));
    for (BatchFamily& f : families_)
      for (int p = 0; p < 2; ++p)
        DGPP_CUDA_OK(cudaEventCreateWithFlags(
            &f.end_events[static_cast<size_t>(p)], cudaEventDisableTiming));
    {
      // Per slot, and per batch family at slots_ + family.
      const size_t n = static_cast<size_t>(slots_) +
                       std::max<size_t>(families_.size(), 1);
      DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_verdict_seq_),
                                 sizeof(uint64_t) * n, cudaHostAllocMapped));
      for (size_t i = 0; i < n; ++i) h_verdict_seq_[i] = 0;
      DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_verdict_seq_),
                              sizeof(uint64_t) * n));
      DGPP_CUDA_OK(cudaMemset(d_verdict_seq_, 0, sizeof(uint64_t) * n));
      verdict_seq_.assign(n, 0);
      DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_stage_seq_),
                                 sizeof(uint64_t) * n, cudaHostAllocMapped));
      DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_stage_late_),
                                 sizeof(uint32_t) * n, cudaHostAllocMapped));
      for (size_t i = 0; i < n; ++i) {
        h_stage_seq_[i] = 0;
        h_stage_late_[i] = 0;
      }
      DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_stage_seq_),
                              sizeof(uint64_t) * n));
      DGPP_CUDA_OK(cudaMemset(d_stage_seq_, 0, sizeof(uint64_t) * n));
      stage_seq_.assign(n, 0);
    }
    batch_min_live_ = slots_ == 1
                          ? 1
                          : std::clamp(batch_min_live, 1, slots_);
    if (batch_unavailable_) {
      // Never reached by the live count: every step replays a scalar graph.
      batch_min_live_ = slots_ + 1;
      if (depth_ > 1 && !Model::kBatchedDraftChain)
        DGPP_LOG_INFO(
            "rank {}: graph row batch unavailable at mtp depth {} (this "
            "family has no batched chain); every step replays a scalar graph",
            rank_, depth_);
      else
        DGPP_LOG_INFO(
            "rank {}: graph row batch unavailable — {} slot{} x {} rows "
            "exceed the {}-row decode ceiling; every step replays a scalar "
            "graph",
            rank_, slots_, slots_ == 1 ? "" : "s", rows_per_request_,
            max_rows_);
    } else if (batch_min_live_ != batch_min_live)
      DGPP_LOG_INFO(
          "rank {}: graph batch crossover {} clamped to {} — the fixed batch "
          "has {} slot{}, so the batch is selected only at full occupancy",
          rank_, batch_min_live, batch_min_live_, slots_,
          slots_ == 1 ? "" : "s");
  }

  ~GraphEngineAdapter() override {
    if (sampling_)
      DGPP_LOG_INFO(
          "rank {}: graph engine sampling summary — {} sampled decode steps, "
          "{} served by the exact gather fallback ({:.1f}%), {} candidates "
          "per rank",
          rank_, sampled_steps_, fallbacks_,
          sampled_steps_ ? 100.0 * static_cast<double>(fallbacks_) /
                               static_cast<double>(sampled_steps_)
                         : 0.0,
          candidates_);
    try {
      drain();
    } catch (const std::exception& e) {
      DGPP_LOG_WARN("rank {}: graph engine drain at teardown: {}", rank_,
                    e.what());
    }
    if (schedule_) {
      std::string hist;
      for (size_t i = 0; i < depth_options_.size(); ++i)
        hist += std::format("{}{}:{}", i ? " " : "", depth_options_[i],
                            sched_hist_[i]);
      std::string bhist;
      for (const BatchFamily& f : families_) {
        bhist += std::format("{}{}-slot batch [", bhist.empty() ? "; " : ", ", f.requests);
        for (size_t i = 0; i < f.sched_hist.size(); ++i)
          bhist += std::format("{}{}:{}", i ? " " : "", depth_options_[i], f.sched_hist[i]);
        bhist += "]";
      }
      DGPP_LOG_INFO(
          "rank {}: scheduled verify depth summary — scalar replays per depth [{}] "
          "({} steps held at the full block: a fresh or sampled slot){}{}",
          rank_, hist, sched_full_forced_, bhist,
          sched_adapt_ ? std::format("; lambda ended at {:.4f} tok/ms after {} steps (floor {:.4f})",
                                     lambda_now(), sched_lambda_steps_, sched_lambda_)
                       : std::string());
    }
    for (std::array<cudaGraphExec_t, 2>& execs : scalar_execs_)
      for (cudaGraphExec_t exec : execs)
        if (exec != nullptr) cudaGraphExecDestroy(exec);
    for (std::vector<std::array<cudaGraphExec_t, 2>>& per_slot : sched_execs_)
      for (std::array<cudaGraphExec_t, 2>& execs : per_slot)
        for (cudaGraphExec_t exec : execs)
          if (exec != nullptr) cudaGraphExecDestroy(exec);
    for (BatchFamily& f : families_)
      for (std::array<cudaGraphExec_t, 2>& execs : f.sched_execs)
        for (cudaGraphExec_t exec : execs)
          if (exec != nullptr) cudaGraphExecDestroy(exec);
    for (BatchFamily& f : families_)
      for (cudaGraphExec_t exec : f.execs)
        if (exec != nullptr) cudaGraphExecDestroy(exec);
    for (std::array<cudaEvent_t, 2>& events : end_events_)
      for (cudaEvent_t ev : events)
        if (ev != nullptr) cudaEventDestroy(ev);
    for (BatchFamily& f : families_)
      for (cudaEvent_t ev : f.end_events)
        if (ev != nullptr) cudaEventDestroy(ev);
    if (h_verdict_seq_) cudaFreeHost(h_verdict_seq_);
    if (d_verdict_seq_) cudaFree(d_verdict_seq_);
    if (h_stage_seq_) cudaFreeHost(h_stage_seq_);
    if (h_stage_late_) cudaFreeHost(h_stage_late_);
    if (d_stage_seq_) cudaFree(d_stage_seq_);
    if (d_proposals_) cudaFree(d_proposals_);
    if (h_proposals_) cudaFreeHost(h_proposals_);
    if (d_specs_) cudaFree(d_specs_);
    if (h_specs_) cudaFreeHost(h_specs_);
    if (d_counts_) cudaFree(d_counts_);
    if (d_bias_) cudaFree(d_bias_);
    if (h_bias_row_) cudaFreeHost(h_bias_row_);
    if (d_prompt_ids_) cudaFree(d_prompt_ids_);
    if (h_prompt_ids_) cudaFreeHost(h_prompt_ids_);
    if (h_fallback_row_) cudaFreeHost(h_fallback_row_);
    if (d_masks_) cudaFree(d_masks_);
    if (h_masks_) cudaFreeHost(h_masks_);
    if (d_verify_logits_) cudaFree(d_verify_logits_);
    if (h_conf_) cudaFreeHost(h_conf_);
    if (h_conf_seq_) cudaFreeHost(h_conf_seq_);
    if (d_conf_seq_) cudaFree(d_conf_seq_);
    if (d_draft_specs_) cudaFree(d_draft_specs_);
    if (d_draft_conf_) cudaFree(d_draft_conf_);
  }
  GraphEngineAdapter(const GraphEngineAdapter&) = delete;
  GraphEngineAdapter& operator=(const GraphEngineAdapter&) = delete;

  int max_concurrent_requests() const override { return slots_; }
  int decode_batch_capacity() const override { return slots_; }
  // The MTP verify writes two rows per step (next and the draft).
  int max_tokens_per_step() const override { return rows_per_request_; }
  MtpAcceptance mtp_acceptance() const override {
    MtpAcceptance a;
    a.depth = rows_per_request_ - 1;
    for (int p = 0; p < a.depth; ++p) {
      a.attempts[p] = mtp_attempts_[static_cast<size_t>(p)];
      a.accepts[p] = mtp_accepts_[static_cast<size_t>(p)];
    }
    return a;
  }
  MtpAcceptance mtp_acceptance(int req) const override {
    MtpAcceptance a;
    if (req < 0 || req >= slots_) return a;
    a.depth = rows_per_request_ - 1;
    for (int p = 0; p < a.depth; ++p) {
      a.attempts[p] = slot_mtp_attempts_[static_cast<size_t>(req)][static_cast<size_t>(p)];
      a.accepts[p] = slot_mtp_accepts_[static_cast<size_t>(req)][static_cast<size_t>(p)];
    }
    return a;
  }
  sched::SchedulerEngine::DecodeBatchStats decode_batch_stats() const override {
    return decode_batch_stats_;
  }
  int batch_min_live() const { return batch_min_live_; }
  // The batch families' slot counts, ascending (empty: scalar only), and
  // the steps each replayed — the gates' evidence that a family ran.
  std::vector<int> batch_families() const {
    std::vector<int> out;
    for (const BatchFamily& f : families_) out.push_back(f.requests);
    return out;
  }
  uint64_t batch_family_steps(int family) const {
    return family_steps_.at(static_cast<size_t>(family));
  }
  int sampling_candidates() const { return candidates_; }

  // ---- confidence-scheduled verify depth (2026-09-14) -----------------------
  // engine/verify_schedule.hpp over the family's confidence head
  // (Model::kVerifyConfidence): a scalar MTP replay verifies only the
  // leading drafts whose prefix survival beats the value of a verify row,
  // on a variant captured at that depth; the block still drafts its full
  // width and the committed transcript is the plain greedy one at every
  // depth (a draft not verified is decoded next step). OFF unless
  // configured here — every existing configuration keeps its captures and
  // its replay path unchanged — and only for greedy slots (a sampled slot
  // verifies its whole block; its fallback machinery is untouched). Call
  // before the first capture (warm_captures). `row_ms` is the cost of one
  // verify row, `lambda_tok_per_ms` the value of decode time (the achieved
  // throughput; the reservation rate verify_reservation_lambda(base, row)
  // before it is measured) — the same constants on every rank, so every
  // rank derives the same depth from the replicated confidence.
  // `min_depth`: never verify fewer drafts than this (>= 1).
  // The depth options: every depth in [min_depth, depth] when the bus's
  // graph-variant budget (kBusMaxGraphVariants: two per slot per option
  // plus two per batch family) holds them, else an even spread that always
  // keeps the full block; a policy depth rounds UP to the next option
  // (never fewer drafts than the policy asked — exact either way).
  // base_ms + adapt: lambda follows the modeled throughput (an EWMA of
  // committed / (base + rows.row) over the MTP steps, floored at
  // lambda_tok_per_ms; verify_schedule.hpp) instead of standing at the
  // configured constant — the fixed point differs per concurrency.
  void configure_verify_schedule(bool on, float row_ms, float lambda_tok_per_ms,
                                 int min_depth = 1, float base_ms = 0.f, bool adapt = false) {
    drain();
    for (const std::array<cudaGraphExec_t, 2>& e : scalar_execs_)
      if (e[0] != nullptr)
        throw std::logic_error(
            "graph engine: configure_verify_schedule after a capture");
    if (!on) {
      schedule_ = false;
      depth_options_.clear();
      return;
    }
    {
      if (!model_->mtp_enabled() || depth_ < 1)
        throw std::invalid_argument(
            "graph engine: the scheduled verify depth needs MTP");
      if (!(row_ms > 0.f) || !(lambda_tok_per_ms > 0.f))
        throw std::invalid_argument(
            "graph engine: the scheduled verify depth needs row_ms > 0 and "
            "lambda > 0");
      if (min_depth < 1 || min_depth > depth_)
        throw std::invalid_argument(
            "graph engine: the scheduled verify depth's min_depth must be in "
            "[1, depth]");
      if constexpr (Model::kVerifyConfidence) {
        conf_rows_ = model_->confidence_rows();
        draft_full_path_ = false;
      } else {
        // No confidence head: the draft head's own probability of its
        // pick, off the sampler's full path (needs the device sampler).
        if (!sampling_)
          throw std::invalid_argument(
              "graph engine: this family has no confidence head; the "
              "scheduled verify depth takes the draft head's probabilities "
              "from the device sampler, which this engine has not");
        conf_rows_ = depth_;
        draft_full_path_ = true;
        if (d_draft_conf_ == nullptr)
          DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_draft_conf_),
                                  sizeof(float) * static_cast<size_t>(slots_) * conf_rows_));
        // The draft picks' spec table: every slot's spec reporting logprobs.
        for (int req = 0; req < slots_; ++req) push_spec(req, h_specs_[req]);
      }
      if (conf_rows_ < depth_ || conf_rows_ > 32)
        throw std::invalid_argument(
            "graph engine: the confidence covers " +
            std::to_string(conf_rows_) + " positions, the verify depth is " +
            std::to_string(depth_));
      // The options that fit the bus: two variants per slot per option
      // and, with the batch, two per family per option.
      const int fit = net::kBusMaxGraphVariants /
                      (2 * (slots_ + static_cast<int>(families_.size())));
      const int span = depth_ - min_depth + 1;
      const int options = std::min(span, fit);
      if (options < 2)
        throw std::invalid_argument(
            "graph engine: the bus's " + std::to_string(net::kBusMaxGraphVariants) +
            " graph variants leave no room for a second verify depth at " +
            std::to_string(slots_) + " slots and " +
            std::to_string(families_.size()) + " batch families");
      depth_options_.clear();
      for (int i = 0; i < options; ++i) {
        const int d = options == 1
                          ? depth_
                          : min_depth + static_cast<int>(std::lround(
                                            static_cast<double>(i) * (depth_ - min_depth) /
                                            static_cast<double>(options - 1)));
        if (depth_options_.empty() || depth_options_.back() != d)
          depth_options_.push_back(d);
      }
      if (depth_options_.back() != depth_) depth_options_.push_back(depth_);
      if (adapt && !(base_ms >= 0.f))
        throw std::invalid_argument("graph engine: the adaptive lambda needs base_ms >= 0");
      schedule_ = true;
      sched_row_ms_ = row_ms;
      sched_lambda_ = lambda_tok_per_ms;
      sched_min_depth_ = min_depth;
      sched_base_ms_ = base_ms;
      sched_adapt_ = adapt;
      sched_lambda_live_ = lambda_tok_per_ms;
      sched_lambda_steps_ = 0;
      if (h_conf_ == nullptr) {
        DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_conf_),
                                   sizeof(float) * static_cast<size_t>(slots_) * conf_rows_,
                                   cudaHostAllocMapped));
        std::fill_n(h_conf_, static_cast<size_t>(slots_) * conf_rows_, 0.f);
        DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_conf_seq_),
                                   sizeof(uint64_t) * slots_, cudaHostAllocMapped));
        for (int i = 0; i < slots_; ++i) h_conf_seq_[i] = 0;
        DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_conf_seq_),
                                sizeof(uint64_t) * slots_));
        DGPP_CUDA_OK(cudaMemset(d_conf_seq_, 0, sizeof(uint64_t) * slots_));
      }
      conf_seq_.assign(static_cast<size_t>(slots_), 0);
      conf_stale_.assign(static_cast<size_t>(slots_), true);
      sched_execs_.assign(
          static_cast<size_t>(slots_),
          std::vector<std::array<cudaGraphExec_t, 2>>(
              depth_options_.size() - 1, {{nullptr, nullptr}}));
      sched_hist_.assign(depth_options_.size(), 0);
      for (BatchFamily& f : families_) {
        f.sched_execs.assign(depth_options_.size() - 1, {{nullptr, nullptr}});
        f.sched_hist.assign(depth_options_.size(), 0);
      }
      sched_full_forced_ = 0;
      std::string opts;
      for (const int d : depth_options_) opts += (opts.empty() ? "" : ",") + std::to_string(d);
      DGPP_LOG_INFO(
          "rank {}: scheduled verify depth on — options [{}] of the {}-draft "
          "block (threshold: prefix survival > {:.3f} = {:.4f} tok/ms x {:.2f} "
          "ms/row{}), greedy slots only; the confidence is {}",
          rank_, opts, depth_, sched_lambda_ * sched_row_ms_, sched_lambda_,
          sched_row_ms_,
          sched_adapt_ ? std::format("; lambda adaptive over base {:.1f} + rows x row ms, floored there",
                                     sched_base_ms_)
                       : std::string(),
          draft_full_path_ ? "the draft head's own probabilities"
                           : "the model's confidence head");
    }
  }
  // Per option, the batched replays of family `family` at that depth.
  std::vector<uint64_t> verify_depth_histogram_batch(int family) const {
    return families_.at(static_cast<size_t>(family)).sched_hist;
  }
  bool verify_schedule() const { return schedule_; }
  const std::vector<int>& verify_depth_options() const { return depth_options_; }
  // Per option, the scalar replays that verified at that depth.
  std::vector<uint64_t> verify_depth_histogram() const { return sched_hist_; }
  // Tests: replaces the policy's depth (req, the policy's depth, the slot's
  // confidence logits, their count) -> the depth to verify (clamped to
  // [min_depth, depth], then rounded up to an option).
  void set_verify_depth_hook(
      std::function<int(int, int, const float*, int)> hook) {
    depth_hook_ = std::move(hook);
  }

  bool supports_sampling() const override { return sampling_; }
  bool supports_logprobs() const override { return sampling_; }
  void configure_sampling(int req, const sample::Params& sampling,
                          uint64_t seed) override {
    check_req(req);
    sample::validate_params(sampling);
    if (sampling.temperature > 0.0f && !sampling_)
      throw std::logic_error(
          "graph engine: no device sampler (engines without the sampler "
          "scratch are greedy-only)");
    params_[static_cast<size_t>(req)] = sampling;
    rng_[static_cast<size_t>(req)] = sample::Rng{seed, 0};
    context_[static_cast<size_t>(req)].clear();
    report_[static_cast<size_t>(req)] = false;
    pending_logprobs_[static_cast<size_t>(req)].clear();
    if (!sampling_) return;
    SampleSpec spec;
    spec.temperature = sampling.temperature;
    spec.top_p = sampling.top_p;
    spec.min_p = sampling.min_p;
    spec.repetition_penalty = sampling.repetition_penalty;
    spec.frequency_penalty = sampling.frequency_penalty;
    spec.presence_penalty = sampling.presence_penalty;
    spec.top_k = sampling.top_k;
    spec.logprobs = -1;
    spec.seed = seed;
    spec.counter = 0;
    spec.draft_temperature = sampling.temperature * proposal_temperature_scale();
    push_spec(req, spec);
    // A fresh context: the prompt's counts arrive with the prefill.
    DGPP_CUDA_OK(cudaMemsetAsync(d_counts_ + static_cast<size_t>(req) * vocab_,
                                 0, sizeof(int32_t) * vocab_,
                                 model_->stream()));
    DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
  }
  void configure_logprobs(int req, int logprobs) override {
    check_req(req);
    if (logprobs >= 0 && !sampling_)
      throw std::logic_error("graph engine: no device sampler — no logprobs");
    report_[static_cast<size_t>(req)] = logprobs >= 0;
    pending_logprobs_[static_cast<size_t>(req)].clear();
    if (!sampling_) return;
    SampleSpec spec = h_specs_[req];
    spec.logprobs = logprobs;
    push_spec(req, spec);
  }
  std::vector<sample::Result> take_logprobs(int req) override {
    check_req(req);
    std::vector<sample::Result> out;
    out.swap(pending_logprobs_[static_cast<size_t>(req)]);
    return out;
  }
  // The logit bias (OpenAI's logit_bias, 2026-09-06): the slot's dense row
  // on the device (the pick kernels add it after the penalties, in place,
  // for the rows whose spec says biased — the graphs carry the table's
  // pointer, the spec flag turns a row on) and on the host (the prefill's
  // decision and the MTP fallback's row-1 sample go through the host
  // sampler). Needs the device sampler: a biased slot takes the full path.
  bool supports_logit_bias() const override { return sampling_; }
  void configure_logit_bias(int req,
                            const std::vector<sched::LogitBias>& bias) override {
    check_req(req);
    std::vector<float>& row = bias_[static_cast<size_t>(req)];
    if (bias.empty()) {
      if (row.empty()) return;
      row.clear();
      clear_bias_row(req);
      return;
    }
    if (!sampling_)
      throw std::logic_error(
          "graph engine: no device sampler — this engine cannot bias the pick");
    row.assign(static_cast<size_t>(vocab_), 0.0f);
    for (const sched::LogitBias& b : bias) {
      if (b.token < 0 || b.token >= vocab_)
        throw std::invalid_argument(
            "graph engine: logit_bias token outside the vocabulary");
      row[static_cast<size_t>(b.token)] = b.bias;
    }
    std::copy(row.begin(), row.end(), h_bias_row_);
    DGPP_CUDA_OK(cudaMemcpyAsync(d_bias_ + static_cast<size_t>(req) * vocab_,
                                 h_bias_row_, sizeof(float) * vocab_,
                                 cudaMemcpyHostToDevice, model_->stream()));
    DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
    SampleSpec spec = h_specs_[req];
    spec.biased = 1;
    push_spec(req, spec);
  }
  void clear_bias_row(int req) {
    if (!sampling_ || d_bias_ == nullptr) return;
    DGPP_CUDA_OK(cudaMemsetAsync(d_bias_ + static_cast<size_t>(req) * vocab_, 0,
                                 sizeof(float) * vocab_, model_->stream()));
    DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
    SampleSpec spec = h_specs_[req];
    spec.biased = 0;
    push_spec(req, spec);
  }
  const float* bias_row(int req) const {
    const std::vector<float>& row = bias_[static_cast<size_t>(req)];
    return row.empty() ? nullptr : row.data();
  }
  bool supports_constraints() const override {
    return sampling_ && grammar_vocab_ != nullptr && grammar_vocab_->usable();
  }
  void configure_constraint(int req, const text::GrammarSpec& grammar) override {
    check_req(req);
    grammar_[static_cast<size_t>(req)].reset();
    if (sampling_) clear_masks(req);
    if (!grammar.active()) return;
    if (!supports_constraints())
      throw std::logic_error(
          "graph engine: no grammar vocabulary — this engine cannot "
          "constrain the pick");
    // The opening state is re-derived at the prefill from the prompt.
    grammar_[static_cast<size_t>(req)] = std::make_unique<text::GrammarState>(
        grammar_vocab_, grammar, /*prompt_opens_thinking=*/true);
  }

  // Startup warm-up: records every scalar variant and, above one slot, the
  // row batch BEFORE the first client request, so no capture (record +
  // instantiate, ~10 ms each, up to slots+1 of them) lands on a live
  // stream. Opens one throwaway session at a time — slot by slot, with
  // `warm_prompt` (eager prefill: collectives on the bus, so every rank must
  // call this at the same point) — captures that slot's scalar variant (and
  // the row batch while the first slot is open; a capture bakes addresses,
  // occupancy is derived at replay), then closes it. One live warm session
  // keeps the pool footprint at a single prompt. Nothing of the warm
  // sessions survives: close pushes position 0 to the device and releases
  // the blocks, and a slot's next prefill rewrites its state. Requires an
  // idle engine.
  void warm_captures(const std::vector<int64_t>& warm_prompt) {
    if (warm_prompt.empty())
      throw std::invalid_argument("graph engine: warm prompt must not be empty");
    for (int req = 0; req < slots_; ++req)
      if (live_[static_cast<size_t>(req)])
        throw std::logic_error("graph engine: warm_captures needs an idle "
                               "engine");
    for (int req = 0; req < slots_; ++req) {
      (void)prefill(req, warm_prompt);
      try {
        ensure_scalar_graph(req);
        if (schedule_)
          for (size_t di = 0; di + 1 < depth_options_.size(); ++di)
            ensure_sched_graph(req, static_cast<int>(di));
        if (req == 0)
          for (size_t f = 0; f < families_.size(); ++f) {
            ensure_batch_graph(static_cast<int>(f));
            if (schedule_ && model_->mtp_enabled())
              for (size_t di = 0; di + 1 < depth_options_.size(); ++di)
                ensure_sched_batch_graph(static_cast<int>(f), static_cast<int>(di));
          }
      } catch (...) {
        close(req);
        throw;
      }
      close(req);
    }
    std::string batches;
    for (const BatchFamily& f : families_)
      batches += (batches.empty() ? " and the row batches for " : ", ") +
                 std::to_string(f.requests) + " slots";
    if (schedule_)
      batches += std::format(" and {} reduced-depth variant{} per slot",
                             depth_options_.size() - 1,
                             depth_options_.size() == 2 ? "" : "s");
    DGPP_LOG_INFO(
        "rank {}: warm capture complete — {} scalar variant{}{} recorded "
        "before the first request",
        rank_, slots_, slots_ == 1 ? "" : "s", batches);
  }
  int64_t pool_blocks_total() const override {
    return model_->kv_blocks_total();
  }
  int64_t pool_blocks_in_use() const override {
    return model_->kv_blocks_in_use();
  }
  int64_t blocks_for_tokens(int64_t tokens) const override {
    return model_->kv_blocks_for_tokens(tokens);
  }

  int32_t prefill(int req, const std::vector<int64_t>& prompt) override {
    return open_slot(req, prompt, [&] { return model_->session_prefill(req, prompt); });
  }
  bool supports_images() const override {
    if constexpr (requires { model_->supports_images(); }) return model_->supports_images();
    return false;
  }
  bool supports_image_prefix_cache() const override {
    return supports_images() && kImagePrefixCache<Model>;
  }
  int32_t prefill_images(int req, const std::vector<int64_t>& prompt,
                         const std::vector<ImageInput>& images) override {
    if constexpr (requires { model_->session_prefill_images(req, prompt, images); })
      return open_slot(req, prompt,
                       [&] { return model_->session_prefill_images(req, prompt, images); });
    return sched::SchedulerEngine::prefill_images(req, prompt, images);
  }

  int64_t prefill_chunk_alignment() const override {
    if constexpr (requires { Model::kResumablePrefill; }) {
      if constexpr (Model::kResumablePrefill) return model_->session_snapshot_align();
    }
    return 0;
  }
  int64_t prefill_chunk_limit() const override {
    if constexpr (requires { Model::kResumablePrefill; }) {
      if constexpr (Model::kResumablePrefill) return model_->max_tokens();
    }
    return 0;
  }
  bool supports_image_chunked_prefill() const override {
    if constexpr (requires { typename Model::PrefillCursor; }) {
      if constexpr (requires(int req, const std::vector<int64_t>& prompt,
                              const std::vector<ImageInput>* images) {
        model_->session_prefill_begin(req, prompt, int64_t{}, int64_t{}, prompt,
                                      nullptr, int64_t{}, images);
      }) return supports_images() && prefill_chunk_alignment() > 0;
    }
    return false;
  }
  void begin_prefill(int req, const std::vector<int64_t>& prompt, int64_t reserve_tokens,
                     int64_t chunk_tokens, const sched::SchedulerEngine::PrefixPrefill& plan) override {
    if constexpr (requires { typename Model::PrefillCursor; }) {
      if (prefill_chunk_alignment() == 0)
        throw std::logic_error("graph engine: this family cannot yield prefill");
      drain();
      check_req(req);
      if (live_[static_cast<size_t>(req)] || prefills_[static_cast<size_t>(req)])
        throw std::logic_error("graph engine: prefill on an open request");
      auto task = std::make_unique<PendingPrefill>();
      task->prompt = prompt;
      if (plan.boundaries) task->boundaries = *plan.boundaries;
      task->plan = plan;
      task->plan.boundaries = &task->boundaries;
      if (plan.images) task->images = *plan.images;
      task->plan.images = &task->images;
      try {
        open_slot_grammar(req, prompt);
        typename Model::SnapshotRequest* snap = nullptr;
        if (plan.snap_slot >= 0) {
          task->snap = arena_.request(plan.snap_slot, plan.snap_position);
          snap = &task->snap;
        }
        if (plan.attach_slot >= 0) {
          if (arena_.position(plan.attach_slot) != plan.attach_position)
            throw std::logic_error("graph engine: attached prefix differs from the plan");
          arena_.attach(req, plan.attach_slot);
        }
        auto cursor = std::make_shared<typename Model::PrefillCursor>([&] {
          const auto reserved = std::min<int64_t>(reserve_tokens + std::max(0, depth_ - 1), model_->max_context());
          if constexpr (requires { model_->session_prefill_begin(req, task->prompt, reserved,
              chunk_tokens, task->boundaries, snap, plan.attach_position, &task->images); }) {
            return model_->session_prefill_begin(req, task->prompt, reserved, chunk_tokens,
                task->boundaries, snap, plan.attach_position, &task->images);
          } else {
            if (!task->images.empty()) throw std::logic_error("graph engine: image prefill cannot yield");
            return model_->session_prefill_begin(req, task->prompt, reserved, chunk_tokens,
                task->boundaries, snap, plan.attach_position);
          }
        }());
        task->advance = [this, req, cursor, task = task.get()](int64_t budget) {
          const int64_t start = cursor->next;
          const bool done = model_->session_prefill_advance(*cursor, budget);
          if (task->snap.taken && !task->plan.snap_taken) {
            arena_.commit(task->plan.snap_slot, task->snap);
            task->plan.snap_taken = true;
          }
          sched::SchedulerEngine::PrefillProgress progress;
          progress.computed_tokens = cursor->next - start;
          progress.snap_taken = task->plan.snap_taken;
          if (done) progress.first_token = open_slot_finish(req, task->prompt, cursor->output);
          return progress;
        };
        prefills_[static_cast<size_t>(req)] = std::move(task);
      } catch (...) {
        close_failed_slot(req);
        throw;
      }
    } else {
      throw std::logic_error("graph engine: this family cannot yield prefill");
    }
  }
  sched::SchedulerEngine::PrefillProgress advance_prefill(int req, int64_t budget = 0) override {
    drain();
    check_req(req);
    auto& task = prefills_[static_cast<size_t>(req)];
    if (!task) throw std::logic_error("graph engine: no pending prefill");
    try {
      auto progress = task->advance(budget);
      reseed_live_feeds();
      if (progress.first_token >= 0) task.reset();
      return progress;
    } catch (...) {
      if (task->snap.taken && !task->plan.snap_taken) arena_.commit(task->plan.snap_slot, task->snap);
      task.reset();
      close_failed_slot(req);
      reseed_live_feeds();
      throw;
    }
  }
  // The group prefill: several cold prompts as the spans of one forward
  // (session_prefill_group), each slot's opening work per request around
  // it. A family without span support prefills them one by one.
  int64_t prefill_group_span_limit() const override {
    if constexpr (requires { model_->prefill_group_span_limit(); })
      return model_->prefill_group_span_limit();
    else
      return 0;
  }
  int64_t prefill_group_total_limit() const override {
    if constexpr (requires { model_->max_tokens(); })
      return model_->max_tokens();
    else
      return 0;
  }
  std::vector<int32_t> prefill_group(const std::vector<int>& reqs,
                                     const std::vector<const std::vector<int64_t>*>& prompts) override {
    if constexpr (requires { model_->session_prefill_group(reqs, prompts); }) {
      if (reqs.size() != prompts.size() || reqs.empty())
        throw std::invalid_argument("graph engine: prefill_group takes one prompt per request");
      if (reqs.size() == 1 || prefill_group_span_limit() <= 0)
        return sched::SchedulerEngine::prefill_group(reqs, prompts);
      drain();
      for (size_t i = 0; i < reqs.size(); ++i) {
        check_req(reqs[i]);
        if (live_[static_cast<size_t>(reqs[i])])
          throw std::logic_error("graph engine: prefill on a live request");
      }
      std::vector<int32_t> firsts(reqs.size(), -1);
      size_t opened = 0;
      try {
        for (size_t i = 0; i < reqs.size(); ++i) open_slot_grammar(reqs[i], *prompts[i]);
        const std::vector<typename Model::Outputs> outs = model_->session_prefill_group(reqs, prompts);
        opened = reqs.size();  // every slot is open on the model from here
        for (size_t i = 0; i < reqs.size(); ++i)
          firsts[i] = open_slot_finish(reqs[i], *prompts[i], outs[i]);
        reseed_live_feeds();
        return firsts;
      } catch (...) {
        // A failed group admission closes every member (session_prefill_group
        // opens every slot before the walk, so the closes are always due).
        for (size_t i = 0; i < reqs.size(); ++i) close_failed_slot(reqs[i]);
        (void)opened;
        throw;
      }
    } else {
      return sched::SchedulerEngine::prefill_group(reqs, prompts);
    }
  }

  // ---- prefix cache (M7) --------------------------------------------------
  sched::SchedulerEngine::PrefixInfo prefix_info() const override {
    sched::SchedulerEngine::PrefixInfo info;
    info.arena_slots = arena_.slots();
    info.step_tokens_max = rows_per_request_;
    info.align = model_->session_snapshot_align();
    info.block_tokens = model_->kv_block_tokens();
    info.chunk_tokens = Model::prefill_chunk_tokens();
    return info;
  }
  int32_t prefill_cached(int req, const std::vector<int64_t>& prompt,
                         sched::SchedulerEngine::PrefixPrefill* plan) override {
    if (plan == nullptr || plan->boundaries == nullptr)
      throw std::invalid_argument("graph engine: prefill_cached without a plan");
    return open_slot(req, prompt, [&] {
      typename Model::SnapshotRequest snap;
      typename Model::SnapshotRequest* snap_ptr = nullptr;
      if (plan->snap_slot >= 0) {
        snap = arena_.request(plan->snap_slot, plan->snap_position);
        snap_ptr = &snap;
      }
      typename Model::Outputs out;
      if (plan->attach_slot >= 0) {
        if (arena_.position(plan->attach_slot) != plan->attach_position)
          throw std::logic_error(
              "graph engine: the attach slot's position differs from the plan");
        arena_.attach(req, plan->attach_slot);
        const std::vector<int64_t> suffix(prompt.begin() + plan->attach_position,
                                          prompt.end());
        out = cached_model_prefill(model_, req, suffix, *plan->boundaries,
                                   snap_ptr, plan->images, true);
      } else {
        out = cached_model_prefill(model_, req, prompt, *plan->boundaries,
                                   snap_ptr, plan->images, false);
      }
      if (snap_ptr != nullptr) {
        arena_.commit(plan->snap_slot, snap);
        plan->snap_taken = snap.taken;
      }
      return out;
    });
  }
  void prefix_snapshot(int req, int slot, int64_t position) override {
    check_live(req, "prefix_snapshot");
    arena_.snapshot(req, slot, position);
  }
  void prefix_arm_hop(int req, int slot, int64_t position) override {
    check_live(req, "prefix_arm_hop");
    hop_slot_[static_cast<size_t>(req)] = slot;
    hop_position_[static_cast<size_t>(req)] = position;
  }
  void prefix_release(int slot) override { arena_.release(slot); }
  sched::SchedulerEngine::PrefixEngineStats prefix_engine_stats() const override {
    sched::SchedulerEngine::PrefixEngineStats st;
    st.snapshots = arena_.snapshots();
    st.snapshot_ms = arena_.snapshot_ms();
    st.attaches = arena_.attaches();
    st.attach_ms = arena_.attach_ms();
    st.snapshot_bytes = static_cast<int64_t>(arena_.bytes());
    return st;
  }

 private:
  // The prefill's slot-side work around the model call that yields the
  // last row's logits (a cold prefill, or an attach + resume): the grammar's
  // opening state, the pick or the sampled draw, the sampled context and
  // its device count table, the draft block's first proposal, the batch
  // feeds. `run` opens the slot on the model; a failure after it closes
  // the slot again (a failed admission must not leak its blocks).
  template <typename Run>
  int32_t open_slot(int req, const std::vector<int64_t>& prompt, Run&& run) {
    // Eager work ahead (the prefill's folds, the draft's picks): every
    // replay in flight settles first — the bus's eager gate opens only
    // once their windows are finished.
    drain();
    check_req(req);
    if (live_[static_cast<size_t>(req)] || prefills_[static_cast<size_t>(req)])
      throw std::logic_error("graph engine: prefill on a live request");
    try {
      open_slot_grammar(req, prompt);
      const typename Model::Outputs out = run();
      const int32_t first = open_slot_finish(req, prompt, out);
      // The prefill wrote its chunks over the device token rows, the feeds
      // included: restore every live slot's persistent feed (the scalar
      // variants and the batch read the same rows).
      reseed_live_feeds();
      return first;
    } catch (...) {
      close_failed_slot(req);
      throw;
    }
  }
  // The grammar's opening state: thinking iff the prompt ends in <think>
  // (the template's generation prompt does).
  void open_slot_grammar(int req, const std::vector<int64_t>& prompt) {
    std::unique_ptr<text::GrammarState>& grammar = grammar_[static_cast<size_t>(req)];
    if (grammar) {
      const text::ChatMarkers& m = grammar_vocab_->markers();
      const bool opens = m.prompt_opens_thinking(prompt);
      grammar = std::make_unique<text::GrammarState>(grammar_vocab_, grammar->spec(), opens);
    }
  }
  // session_prefill opens the slot before any later pick/draft can fail:
  // a failed admission is made recoverable rather than leaking its blocks.
  void close_failed_slot(int req) {
    model_->session_close(req);
    pending_[static_cast<size_t>(req)] = -1;
    drafts_[static_cast<size_t>(req)].assign(static_cast<size_t>(depth_), -1);
    live_[static_cast<size_t>(req)] = false;
    reserved_[static_cast<size_t>(req)] = false;
  }
  // The slot-side work after the model's prefill of `req` (its last-row
  // outputs in `out`): the pick or the sampled draw, the sampled context
  // and its device count table, the draft block's first proposal; the
  // slot goes live. The caller reseeds the live feeds after.
  int32_t open_slot_finish(int req, const std::vector<int64_t>& prompt, const typename Model::Outputs& out) {
    std::unique_ptr<text::GrammarState>& grammar = grammar_[static_cast<size_t>(req)];
    const bool sampled = full_path_slot(req);
    int32_t first = -1;
    {
      {
        if (sampled) {
          std::vector<int32_t> context(prompt.begin(), prompt.end());
          const text::TokenMask* mask = nullptr;
          if (grammar && grammar->active()) {
            text::TokenMask& m0 =
                masks_[static_cast<size_t>(req) * rows_per_request_];
            grammar->mask(&m0);
            if (m0.constrained()) mask = &m0;
          }
          const sample::Result r =
              prefill_sample_(out, params_[static_cast<size_t>(req)],
                              rng_[static_cast<size_t>(req)], context, mask,
                              bias_row(req));
          first = r.token;
          if (report_[static_cast<size_t>(req)])
            pending_logprobs_[static_cast<size_t>(req)].push_back(r);
        } else {
          first = prefill_pick_(out);
        }
        if (grammar) grammar->advance(first);
      }
      if (sampled) {
        // The prompt is the request's context on the device (the first
        // token is the next step's fed token and counts itself there), and
        // the prefill's draw moved the counter. The host mirror of the
        // context serves the MTP fallbacks' penalties.
        std::vector<int32_t>& context = context_[static_cast<size_t>(req)];
        context.assign(prompt.begin(), prompt.end());
        context.push_back(first);
        const int n = static_cast<int>(prompt.size());
        if (n > model_->max_context())
          throw std::invalid_argument("graph engine: prompt exceeds the context bound");
        std::copy(prompt.begin(), prompt.end(), h_prompt_ids_);
        DGPP_CUDA_OK(cudaMemcpyAsync(d_prompt_ids_, h_prompt_ids_,
                                     sizeof(int64_t) * n,
                                     cudaMemcpyHostToDevice, model_->stream()));
        device_sample_count_tokens(d_counts_ + static_cast<size_t>(req) * vocab_,
                                d_prompt_ids_, n, static_cast<int>(vocab_),
                                model_->stream());
        DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
        push_counter(req);
      }
      pending_[static_cast<size_t>(req)] = first;
      if (model_->mtp_enabled()) {
        // session_prefill filled the draft cache through the prompt. Advance
        // its one-row lag over the first generated token and pick the initial
        // proposal directly from device logits (decode mirrors may already be
        // disabled after this graph's first capture).
        (void)model_->session_draft(req, {first});
        std::vector<int32_t>& drafts = drafts_[static_cast<size_t>(req)];
        drafts.assign(static_cast<size_t>(depth_), -1);
        DevicePicker::Inputs first_draft = scalar_pick_inputs(1);
        arm_draft_sampling(first_draft, req, /*draft_index=*/0);
        drafts[0] = picker_->run(model_->stream(), first_draft).next;
        chain_drafts_eagerly(req);
        refresh_confidence_eagerly(req);
      }
      reserved_[static_cast<size_t>(req)] = false;
      live_[static_cast<size_t>(req)] = true;
      return first;
    }
  }

 public:
  void reserve(int req, int64_t tokens) override {
    check_live(req, "reserve");
    // The chained drafts write depth - 1 rows past the verify's last row.
    model_->session_reserve_blocks(
        req, std::min<int64_t>(tokens + std::max(0, depth_ - 1),
                               model_->max_context()));
    reserved_[static_cast<size_t>(req)] = true;
  }

  std::vector<int32_t> step(int req) override {
    std::vector<std::vector<int32_t>> out = step_batch({req});
    return std::move(out[0]);
  }

  std::vector<std::vector<int32_t>> step_batch(
      const std::vector<int>& reqs) override {
    validate_batch(reqs);
    const int family = reqs.size() >= static_cast<size_t>(batch_min_live_)
                           ? family_for(reqs) : -1;
    const bool batched = family >= 0;
    log_mode_change(batched, reqs.size(), family);
    if (!batched) {
      std::vector<std::vector<int32_t>> batches;
      batches.reserve(reqs.size());
      for (const int req : reqs) batches.push_back(step_scalar(req));
      return batches;
    }

    ensure_batch_graph(family);
    BatchFamily& fam = families_[static_cast<size_t>(family)];
    ++family_steps_[static_cast<size_t>(family)];
    // The scheduled verify depth: one depth for the batch — the mean-
    // survival rule over the live slots (verify_schedule.hpp; a slot
    // verifying more or fewer drafts than its own policy would is exact
    // either way); the whole block when any of them is fresh or sampled.
    int di = -1;
    int rows = rows_per_request_;
    if (schedule_ && model_->mtp_enabled()) {
      di = choose_batch_depth_option(reqs);
      rows = 1 + depth_options_[static_cast<size_t>(di)];
      if (di < full_depth_option()) ensure_sched_batch_graph(family, di);
      ++fam.sched_hist[static_cast<size_t>(di)];
    }
    model_->session_graph_use_batch_contract(rows, fam.requests);
    model_->session_graph_stage_batch();
    // Launch first (the window armed behind whatever runs), settle the
    // older replay while this one runs, stage the pick's masks for it,
    // then wait for its verdict.
    Replay r;
    r.batched = true;
    r.family = family;
    r.parity = fam.parity;
    fam.parity ^= 1;
    r.reqs = reqs;
    r.depth_option = di;
    r.rows = rows;
    launch(std::move(r));
    settle_older();
    if (rows < rows_per_request_)
      stage_masks_compact(fam.requests, rows);
    else
      for (const int req : reqs) stage_masks(req);
    publish_stage(batch_index(family));
    if (!pipeline_) drain();  // after the publish: the replay's gate waits on it
    wait_verdict(inflight_.back());

    std::vector<std::vector<int32_t>> batches;
    batches.reserve(reqs.size());
    int committed = 0;
    for (const int req : reqs) {
      batches.push_back(
          collect_verdict(req, /*verdict_request=*/req, /*batched=*/true, rows));
      committed += static_cast<int>(batches.back().size());
    }
    if (schedule_ && model_->mtp_enabled())
      note_step_for_lambda(committed, rows * static_cast<int>(reqs.size()));
    return batches;
  }

  void close(int req) override {
    drain();
    check_req(req);
    if (prefills_[static_cast<size_t>(req)]) prefills_[static_cast<size_t>(req)].reset();
    else check_live(req, "close");
    // The slot's sampling tally, for the width sweep and the record: how
    // many of its stochastic steps the exact gather fallback served.
    if (sampling_ && slot_sampled_[static_cast<size_t>(req)] > 0)
      DGPP_LOG_INFO(
          "rank {}: slot {} closed: {} sampled decode steps, {} fallbacks",
          rank_, req, slot_sampled_[static_cast<size_t>(req)],
          slot_fallbacks_[static_cast<size_t>(req)]);
    slot_sampled_[static_cast<size_t>(req)] = 0;
    slot_fallbacks_[static_cast<size_t>(req)] = 0;
    slot_mtp_attempts_[static_cast<size_t>(req)] = {};
    slot_mtp_accepts_[static_cast<size_t>(req)] = {};
    // The model reports its own slot diagnostics on close (GLM: the listed
    // attention's guarded gather and the select's short fills).
    model_->session_close(req);
    live_[static_cast<size_t>(req)] = false;
    reserved_[static_cast<size_t>(req)] = false;
    pending_[static_cast<size_t>(req)] = -1;
    drafts_[static_cast<size_t>(req)].assign(static_cast<size_t>(depth_), -1);
    hop_slot_[static_cast<size_t>(req)] = -1;
    if (schedule_) conf_stale_[static_cast<size_t>(req)] = true;
    // A reopened slot is greedy until the scheduler arms it again — on the
    // device too, so a padded replay of this slot never draws.
    params_[static_cast<size_t>(req)] = sample::greedy_params();
    rng_[static_cast<size_t>(req)] = sample::Rng{};
    context_[static_cast<size_t>(req)].clear();
    report_[static_cast<size_t>(req)] = false;
    pending_logprobs_[static_cast<size_t>(req)].clear();
    grammar_[static_cast<size_t>(req)].reset();
    if (sampling_) {
      push_spec(req, SampleSpec{});
      clear_masks(req);
    }
    if (!bias_[static_cast<size_t>(req)].empty()) {
      bias_[static_cast<size_t>(req)].clear();
      if (sampling_ && d_bias_ != nullptr) {
        DGPP_CUDA_OK(cudaMemsetAsync(d_bias_ + static_cast<size_t>(req) * vocab_,
                                     0, sizeof(float) * vocab_, model_->stream()));
        DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
      }
    }
  }

  // The slot's RNG state — the audit's view of the draws consumed.
  const sample::Rng& rng(int req) const {
    check_req(req);
    return rng_[static_cast<size_t>(req)];
  }

 private:
  struct PendingPrefill {
    std::vector<int64_t> prompt, boundaries;
    std::vector<ImageInput> images;
    sched::SchedulerEngine::PrefixPrefill plan;
    typename Model::SnapshotRequest snap;
    std::function<sched::SchedulerEngine::PrefillProgress(int64_t)> advance;
  };
  std::vector<std::unique_ptr<PendingPrefill>> prefills_;

  void check_req(int req) const {
    if (req < 0 || req >= slots_)
      throw std::out_of_range("graph engine: request slot " +
                              std::to_string(req));
  }
  void check_live(int req, const char* op) const {
    check_req(req);
    if (!live_[static_cast<size_t>(req)])
      throw std::logic_error(std::string("graph engine: ") + op +
                             " on a closed request");
  }

  DevicePicker::Inputs scalar_pick_inputs(int slot) const {
    DevicePicker::Inputs in;
    in.logits = model_->device_logits();
    in.rows = 1;
    in.vocab_count = model_->lm_vocab_count();
    in.vocab_begin = model_->lm_vocab_begin();
    in.fed = model_->device_tokens();
    in.slot = slot;
    return in;
  }

  // The scalar variant of request slot `req`: its own spec and count table
  // are baked into the recorded nodes (the verdict indexes request 0).
  DevicePicker::Inputs scalar_sampling_inputs(int req, int rows = 1) const {
    DevicePicker::Inputs in = scalar_pick_inputs(/*slot=*/0);
    in.rows = rows;
    in.fed = model_->device_feed(req, rows_per_request_);
    if (sampling_) {
      in.specs = d_specs_ + req;
      in.counts = d_counts_ + static_cast<size_t>(req) * vocab_;
      in.vocab_size = static_cast<int>(vocab_);
      in.masks = d_masks_ + static_cast<size_t>(req) * rows_per_request_ *
                                mask_stride_;
      in.mask_stride = mask_stride_;
      in.bias = d_bias_ + static_cast<size_t>(req) * vocab_;
      in.proposals_in =
          d_proposals_ ? d_proposals_ + static_cast<size_t>(req) * kSampleProposalSlots
                       : nullptr;
    }
    return in;
  }

  // `rows`: the verify's rows per request (rows_per_request_, or a reduced-
  // depth batch's; then the fed tokens are the compacted copy in the
  // front scratch, glm_spec_gather_feed).
  DevicePicker::Inputs verify_pick_inputs(int requests, int rows = -1) const {
    if (rows < 0) rows = rows_per_request_;
    DevicePicker::Inputs in = scalar_pick_inputs(/*slot=*/0);
    in.rows = requests * rows;
    in.requests = requests;
    in.rows_per_request = rows;
    in.fed = rows == rows_per_request_ ? model_->device_feed(0, rows_per_request_)
                                       : model_->device_tokens();
    in.positions = model_->device_positions();
    in.position_stride = rows;
    if (sampling_) {
      in.specs = d_specs_;
      in.counts = d_counts_;
      in.vocab_size = static_cast<int>(vocab_);
      in.masks = d_masks_;
      in.mask_stride = mask_stride_;
      in.bias = d_bias_;
      in.proposals_in = d_proposals_;
    }
    return in;
  }

  // A draft pick draws from the draft head's own distribution instead of
  // taking its argmax, and keeps the final set it drew from
  // for the next step's verify: with a proposal in hand the accept test is
  // min(1, P/Q), whose rate is 1 - TV(P, Q) rather than the deterministic
  // rule's ceiling of P(mode). No count table — a draft commits no context,
  // and no mask or bias: any proposal is exact, so the cheapest one that
  // resembles P is the right one. A greedy request (temperature 0) takes
  // the same path and writes no proposal, so its rule is unchanged.
  bool draft_sampled_ = false;  // the draft pick draws (and proposes)
  // DGPP_SPEC_PROPOSAL_TEMP: the draft's temperature as a multiple of the
  // request's (default 1). Exact at any value; a calibration knob for the
  // draft head's overlap with the target.
  static float proposal_temperature_scale() {
    static const float scale = [] {
      const char* v = std::getenv("DGPP_SPEC_PROPOSAL_TEMP");
      if (v == nullptr) return 1.0f;
      const float f = std::strtof(v, nullptr);
      return (f > 0.0f && f < 10.0f) ? f : 1.0f;
    }();
    return scale;
  }
  static bool proposal_drafts_enabled() {
    static const bool on = [] {
      const char* v = std::getenv("DGPP_SPEC_PROPOSAL");
      return !(v != nullptr && std::string(v) == "off");
    }();
    return on;
  }
  void arm_draft_sampling(DevicePicker::Inputs& in, int req,
                          int draft_index) const {
    if (!sampling_ || d_proposals_ == nullptr || !proposal_drafts_enabled())
      return;
    in.specs = (draft_full_path_ ? d_draft_specs_ : d_specs_) + req;
    in.counts = nullptr;
    in.vocab_size = static_cast<int>(vocab_);
    in.proposals_out =
        d_proposals_ + static_cast<size_t>(req) * kSampleProposalSlots;
    in.proposals_out_host =
        h_proposals_ + static_cast<size_t>(req) * kSampleProposalSlots;
    in.draft_index = draft_index;
  }
  void arm_draft_sampling_batch(DevicePicker::Inputs& in, int draft_index) const {
    if (!sampling_ || d_proposals_ == nullptr || !proposal_drafts_enabled())
      return;
    in.specs = draft_full_path_ ? d_draft_specs_ : d_specs_;
    in.counts = nullptr;
    in.vocab_size = static_cast<int>(vocab_);
    in.proposals_out = d_proposals_;
    in.proposals_out_host = h_proposals_;
    in.draft_index = draft_index;
  }

  DevicePicker::Inputs draft_pick_inputs(
      const PickVerdict* verify_verdicts, int requests, int rows = -1) const {
    if (rows < 0) rows = rows_per_request_;
    DevicePicker::Inputs in = scalar_pick_inputs(/*slot=*/1);
    in.rows = requests;
    in.requests = requests;
    in.rows_per_request = 1;
    in.positions = model_->device_positions();
    in.position_stride = rows;
    in.row_select = verify_verdicts;
    in.source_row_stride = rows;
    return in;
  }

  // The batched chain rows' pick (depth >= 2, 2026-09-10): one candidate
  // per request at rows 0 .. k-1 of the chain run's head, in slot order.
  // No occupancy positions, as the scalar chain pick: a padding row's
  // verdict is some valid id (a closed slot's feed is zeroed off the first
  // draft's inactive verdict; a live slot's row past the context cannot
  // stand).
  DevicePicker::Inputs chain_pick_inputs(int requests, int slot) const {
    DevicePicker::Inputs in = scalar_pick_inputs(slot);
    in.rows = requests;
    in.requests = requests;
    in.rows_per_request = 1;
    return in;
  }

  void validate_batch(const std::vector<int>& reqs) const {
    if (reqs.empty() || reqs.size() > static_cast<size_t>(slots_))
      throw std::invalid_argument("graph engine: invalid decode batch size");
    std::vector<bool> seen(static_cast<size_t>(slots_), false);
    for (const int req : reqs) {
      check_live(req, "step");
      if (!reserved_[static_cast<size_t>(req)])
        throw std::logic_error(
            "graph engine: step before lifetime reservation on slot " +
            std::to_string(req));
      if (seen[static_cast<size_t>(req)])
        throw std::invalid_argument("graph engine: duplicate request slot");
      seen[static_cast<size_t>(req)] = true;
    }
    for (int req = 0; req < slots_; ++req) {
      if (live_[static_cast<size_t>(req)] != seen[static_cast<size_t>(req)])
        throw std::logic_error(
            "graph engine: a physical replay must include every live slot");
    }
  }


  // Every live slot's persistent feed rows from the host's pending token
  // and drafts (after eager work wrote over the device token rows).
  void reseed_live_feeds() {
    for (int req = 0; req < slots_; ++req) {
      if (!live_[static_cast<size_t>(req)]) continue;
      model_->session_graph_seed_feed(req, feed_of(req));
    }
  }

  void ensure_capture_resources() {
    if (recorder_ != nullptr) return;
    model_->set_decode_route_traces(false);
    model_->set_decode_tail_mirrors(false);
    model_->session_graph_prepare();
    recorder_ =
        std::make_unique<GraphRecordReducer>(*bus_, model_->stream());
  }

  template <typename Build>
  cudaGraphExec_t capture_variant(int variant, Build&& build) {
    drain();  // recording needs a quiet bus: no window live
    ensure_capture_resources();
    BoundaryReducer* eager = model_->set_boundary(recorder_.get());
    bool record_open = false;
    bool capture_open = false;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    try {
      std::string err;
      if (!bus_->graph_record_begin(&err, variant))
        throw std::runtime_error("graph engine record begin: " + err);
      record_open = true;
      DGPP_CUDA_OK(cudaStreamBeginCapture(model_->stream(),
                                          cudaStreamCaptureModeThreadLocal));
      capture_open = true;
      build();
      const cudaError_t capture_status =
          cudaStreamEndCapture(model_->stream(), &graph);
      capture_open = false;
      DGPP_CUDA_OK(capture_status);
      if (graph == nullptr)
        throw std::runtime_error("graph engine capture produced no graph");
      if (!bus_->graph_record_end(&err))
        throw std::runtime_error("graph engine record end: " + err);
      record_open = false;
      model_->set_boundary(eager);
      eager = nullptr;
      check_decode_graph(graph, rank_,
                             "graph variant " + std::to_string(variant), model_->session_graph_host_nodes());
      // The executable's device memory (the residual ledger, 2026-09-13):
      // ~45 KB per node on this driver, so a 2,500-node variant costs
      // ~110 MB and fourteen of them 1.5 GiB — the largest item the memory
      // plan did not name.
      size_t free_before = 0, free_after = 0, total = 0;
      (void)cudaMemGetInfo(&free_before, &total);
      DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
      cudaGraphDestroy(graph);
      (void)cudaMemGetInfo(&free_after, &total);
      DGPP_LOG_INFO("rank {}: graph variant {} instantiated — {:.1f} MiB of device memory ({:.1f} GiB free)", rank_,
                    variant, static_cast<double>(free_before - free_after) / (1024.0 * 1024.0),
                    static_cast<double>(free_after) / (1024.0 * 1024.0 * 1024.0));
      return exec;
    } catch (...) {
      if (capture_open) {
        cudaGraph_t abandoned = nullptr;
        (void)cudaStreamEndCapture(model_->stream(), &abandoned);
        if (abandoned != nullptr) cudaGraphDestroy(abandoned);
      }
      if (record_open) {
        std::string ignored;
        (void)bus_->graph_record_end(&ignored);
      }
      if (eager != nullptr) model_->set_boundary(eager);
      if (graph != nullptr) cudaGraphDestroy(graph);
      if (exec != nullptr) cudaGraphExecDestroy(exec);
      throw;
    }
  }

  int scalar_variant(int req, int parity) const { return 2 * req + parity; }
  int batch_variant(int family, int parity) const {
    return 2 * slots_ + 2 * family + parity;
  }
  // The reduced-depth scalar variants (the scheduled verify depth), after
  // every scalar and batch variant: option `di` of slot `req` (the last
  // option, the full block, is the slot's scalar variant itself).
  int sched_variant(int req, int di, int parity) const {
    const int reduced = static_cast<int>(depth_options_.size()) - 1;
    return 2 * slots_ + 2 * static_cast<int>(families_.size()) +
           2 * (req * reduced + di) + parity;
  }
  int full_depth_option() const {
    return static_cast<int>(depth_options_.size()) - 1;
  }
  // The reduced-depth batch variants, after the reduced-depth scalar ones.
  int sched_batch_variant(int family, int di, int parity) const {
    const int reduced = static_cast<int>(depth_options_.size()) - 1;
    return 2 * slots_ + 2 * static_cast<int>(families_.size()) +
           2 * slots_ * reduced + 2 * (family * reduced + di) + parity;
  }
  // The stage handshake's and the verdict's index of a family.
  int batch_index(int family) const { return slots_ + family; }
  bool batch_captured(int family) const {
    return families_[static_cast<size_t>(family)].execs[0] != nullptr;
  }
  bool any_batch_captured() const {
    for (size_t f = 0; f < families_.size(); ++f)
      if (batch_captured(static_cast<int>(f))) return true;
    return false;
  }
  // The smallest family whose slots [0, requests) cover every live slot.
  int family_for(const std::vector<int>& reqs) const {
    int top = 0;
    for (const int req : reqs) top = std::max(top, req);
    for (size_t f = 0; f < families_.size(); ++f)
      if (families_[f].requests > top) return static_cast<int>(f);
    return -1;
  }

  // The stage handshake and the masks' upload, recorded ahead of the
  // pick: the wait node bumps the replay counter of `index` (a slot, or
  // slots_ for the batch) and spins until the host published that many
  // stages; the upload copies the pinned mask rows [row0, row0 + rows) to
  // the device table. Only a sampling engine has masks.
  void record_stage_gate(int index, int row0, int rows) {
    if (!sampling_) return;
    glm_stage_wait(h_stage_seq_ + index, d_stage_seq_ + index,
                   h_stage_late_ + index, /*timeout_ns=*/30'000'000'000ll,
                   model_->stream());
    const size_t words = static_cast<size_t>(rows) * mask_stride_;
    const size_t off = static_cast<size_t>(row0) * mask_stride_;
    glm_upload_words(h_masks_ + off, d_masks_ + off, words, model_->stream());
  }
  void publish_stage(int index) {
    if (!sampling_) return;
    const size_t i = static_cast<size_t>(index);
    ++stage_seq_[i];
    __atomic_store_n(h_stage_seq_ + i, stage_seq_[i], __ATOMIC_RELEASE);
  }

  void ensure_scalar_graph(int req) {
    std::array<cudaGraphExec_t, 2>& execs = scalar_execs_[static_cast<size_t>(req)];
    if (execs[0] != nullptr) return;
    const bool mtp = model_->mtp_enabled();
    const auto build = [&](int parity) {
      if (mtp) {
        (void)parity;
        record_scalar_mtp(req, rows_per_request_);
      } else {
        model_->session_graph_capture_step(
            req, std::vector<int64_t>{pending_[static_cast<size_t>(req)]},
            /*device_positions=*/true, /*device_tokens=*/false);
        record_stage_gate(req, req * rows_per_request_, rows_per_request_);
        picker_->record(model_->stream(), scalar_sampling_inputs(req));
        (void)parity;
        glm_publish_seq(d_verdict_seq_ + req, h_verdict_seq_ + req,
                        model_->stream());
        model_->session_graph_capture_commit(
            req, picker_->device_verdict(0));
      }
    };
    for (int parity = 0; parity < 2; ++parity)
      execs[static_cast<size_t>(parity)] =
          capture_variant(scalar_variant(req, parity), [&] { build(parity); });
    if (any_batch_captured())
      model_->session_graph_use_batch_contract(rows_per_request_,
                                               families_.back().requests);
    DGPP_LOG_INFO(
        "rank {}: serving scalar graph variants {}/{} captured for request "
        "slot {} ({} rows{})",
        rank_, scalar_variant(req, 0), scalar_variant(req, 1), req,
        rows_per_request_, mtp ? ", MTP" : "");
  }

  // The scalar MTP replay of slot `req` verifying `rows` rows (the pending
  // token and rows - 1 drafts; rows_per_request_ for the full block, fewer
  // for a reduced-depth variant): the T-row verify, the verdict, the
  // commit, the draft block off the verdict with its pick at the last
  // accepted row, then — depth >= 2 — the chained rows, one pick each, the
  // feed of every draft (the block's full width, whatever the verify's
  // rows: the slot's persistent feed keeps rows_per_request_ rows and a
  // reduced variant reads its prefix), and — scheduling — the block
  // confidence published to the host for the next step's depth.
  void record_scalar_mtp(int req, int rows) {
    const bool reduced = rows != rows_per_request_;
    std::vector<int64_t> feed = feed_of(req);
    if (reduced) feed.resize(static_cast<size_t>(rows));
    model_->session_graph_capture_step(req, feed,
                                       /*device_positions=*/true,
                                       /*device_tokens=*/true,
                                       /*feed_rows=*/reduced ? rows_per_request_ : 0);
    record_stage_gate(req, req * rows_per_request_, rows);
    picker_->record(model_->stream(), scalar_sampling_inputs(req, rows));
    glm_publish_seq(d_verdict_seq_ + req, h_verdict_seq_ + req,
                    model_->stream());
    snapshot_verify_rows(req, rows, /*first_row=*/0);
    model_->session_graph_capture_commit(req, picker_->device_verdict(0));
    model_->session_graph_capture_draft(req, picker_->device_verdict(0));
    DevicePicker::Inputs draft = scalar_pick_inputs(/*slot=*/1);
    draft.row_select = picker_->device_verdict(0);
    draft.source_row_stride = rows;
    arm_draft_sampling(draft, req, /*draft_index=*/0);
    picker_->record(model_->stream(), draft);
    std::vector<const PickVerdict*> drafts{picker_->device_verdict(1)};
    for (int c = 1; c < depth_; ++c) {
      model_->session_graph_capture_draft_chain(
          req, picker_->device_verdict(0), picker_->device_verdict(c),
          /*index=*/c - 1, /*first=*/c == 1, /*last=*/c == depth_ - 1);
      DevicePicker::Inputs chain = scalar_pick_inputs(1 + c);
      arm_draft_sampling(chain, req, /*draft_index=*/c);
      picker_->record(model_->stream(), chain);
      drafts.push_back(picker_->device_verdict(1 + c));
    }
    model_->session_graph_capture_next_tokens(req, drafts);
    record_confidence_publish(req, /*request=*/0);
  }

  // The reduced-depth scalar variant `di` of slot `req` (the scheduled
  // verify depth): the same replay as the slot's scalar variant over
  // 1 + depth_options_[di] rows, its own bus variant and executable.
  void ensure_sched_graph(int req, int di) {
    std::array<cudaGraphExec_t, 2>& execs =
        sched_execs_.at(static_cast<size_t>(req)).at(static_cast<size_t>(di));
    if (execs[0] != nullptr) return;
    const int rows = 1 + depth_options_.at(static_cast<size_t>(di));
    for (int parity = 0; parity < 2; ++parity)
      execs[static_cast<size_t>(parity)] = capture_variant(
          sched_variant(req, di, parity), [&] { record_scalar_mtp(req, rows); });
    if (any_batch_captured())
      model_->session_graph_use_batch_contract(rows_per_request_,
                                               families_.back().requests);
    DGPP_LOG_INFO(
        "rank {}: serving reduced-depth graph variants {}/{} captured for "
        "request slot {} ({} rows: {} of the {} drafts)",
        rank_, sched_variant(req, di, 0), sched_variant(req, di, 1), req,
        rows, rows - 1, depth_);
  }

  // Scheduling: the slot's block confidence to its pinned mirror at the
  // replay's tail (a kernel node; the decode graph is kernels-only).
  // `request`: the picker request index the replay decided slot `req` at
  // (0 for a scalar replay, the slot for a batch).
  void record_confidence_publish(int req, int request) {
    if (!schedule_) return;
    const float* src = nullptr;
    if constexpr (Model::kVerifyConfidence) {
      src = model_->device_confidence() + static_cast<size_t>(req) * conf_rows_;
    } else {
      // The draft picks' outcomes (slots 1 .. depth) -> the logits.
      float* dst = d_draft_conf_ + static_cast<size_t>(req) * conf_rows_;
      device_sample_draft_confidence(picker_->outcomes_table(),
                                     DevicePicker::outcomes_slot_stride(),
                                     request, depth_, dst, model_->stream());
      src = dst;
    }
    glm_publish_f32(src, h_conf_ + static_cast<size_t>(req) * conf_rows_, conf_rows_,
                    d_conf_seq_ + req, h_conf_seq_ + req, model_->stream());
  }
  // The eager draft's confidence (a slot's open, or a host re-draft): no
  // replay of the slot is in flight, so the mirror is simply copied.
  void refresh_confidence_eagerly(int req) {
    if (!schedule_) return;
    if constexpr (Model::kVerifyConfidence) {
      DGPP_CUDA_OK(cudaMemcpyAsync(
          h_conf_ + static_cast<size_t>(req) * conf_rows_,
          model_->device_confidence() + static_cast<size_t>(req) * conf_rows_,
          sizeof(float) * static_cast<size_t>(conf_rows_), cudaMemcpyDeviceToHost,
          model_->stream()));
      DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
      conf_stale_[static_cast<size_t>(req)] = false;
    } else {
      // The eager chain picks reuse one picker slot: no per-position
      // outcomes to read. The next replay publishes; this step verifies
      // the whole block.
      conf_stale_[static_cast<size_t>(req)] = true;
    }
  }
  // The depth option the next scalar replay of `req` verifies at: the
  // full block for a slot whose drafts no publication describes (fresh
  // after a host re-draft, or sampled), else the policy over the block
  // confidence the slot's last replay published — waited for here, at its
  // tail, so this step's launch follows that replay's end (the scheduled
  // path gives up the launch-ahead of the pipelined replay: the graph to
  // launch is not known until the confidence is).
  // The slot's published block confidence for the coming step, or null
  // when the slot must verify the whole block (fresh, re-drafted eagerly,
  // or sampled: no confidence stands for its drafts). Waits for the last
  // replay's publication (the pinned sequence).
  const float* wait_confidence(int req) {
    if (conf_stale_[static_cast<size_t>(req)] || sampled_slot(req)) return nullptr;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(pick_timeout_ms_);
    uint64_t spins = 0;
    while (__atomic_load_n(h_conf_seq_ + req, __ATOMIC_ACQUIRE) <
           conf_seq_[static_cast<size_t>(req)]) {
      if ((++spins & 255) == 0) {
        if (std::chrono::steady_clock::now() > deadline)
          throw std::runtime_error(
              "graph engine: the replay's confidence did not publish within "
              "the pick timeout");
        std::this_thread::yield();
      }
    }
    return h_conf_ + static_cast<size_t>(req) * conf_rows_;
  }
  // The option index whose depth covers k drafts (rounding up).
  int depth_option_for(int k) const {
    k = std::clamp(k, sched_min_depth_, depth_);
    for (size_t di = 0; di < depth_options_.size(); ++di)
      if (depth_options_[di] >= k) return static_cast<int>(di);
    return full_depth_option();
  }

  // The lambda in force: the configured constant, or the adaptive EWMA
  // floored at it.
  float lambda_now() const {
    return sched_adapt_ ? std::max(sched_lambda_, static_cast<float>(sched_lambda_live_))
                        : sched_lambda_;
  }
  // After an MTP step (scalar or batched, any depth): fold its committed
  // tokens over its modeled time into the adaptive lambda. `rows` is the
  // step's verify rows over all its slots. Replicated inputs, one fixed
  // arithmetic: every rank derives the same value.
  void note_step_for_lambda(int committed, int rows) {
    if (!sched_adapt_) return;
    sched_lambda_live_ = verify_lambda_update(sched_lambda_live_, committed, rows, sched_base_ms_,
                                              sched_row_ms_, kSchedLambdaAlpha);
    ++sched_lambda_steps_;
  }

  int choose_depth_option(int req) {
    const float* conf = wait_confidence(req);
    if (conf == nullptr) {
      ++sched_full_forced_;
      return full_depth_option();
    }
    int k = scheduled_verify_depth(conf, depth_, sched_row_ms_, lambda_now());
    if (depth_hook_) k = depth_hook_(req, k, conf, depth_);
    return depth_option_for(k);
  }

  // The batch's one depth (verify_schedule.hpp, the mean-survival rule over
  // the live slots): the whole block when any slot must verify it; a test
  // hook forces per slot, the batch taking the deepest.
  int choose_batch_depth_option(const std::vector<int>& reqs) {
    std::vector<const float*> confs;
    confs.reserve(reqs.size());
    for (const int req : reqs) {
      const float* conf = wait_confidence(req);
      if (conf == nullptr) {
        ++sched_full_forced_;
        return full_depth_option();
      }
      confs.push_back(conf);
    }
    int k = scheduled_verify_depth_batch(confs.data(), static_cast<int>(confs.size()), depth_,
                                         sched_row_ms_, lambda_now());
    if (depth_hook_) {
      int forced = 0;
      for (size_t i = 0; i < reqs.size(); ++i)
        forced = std::max(forced, depth_hook_(reqs[i], k, confs[i], depth_));
      k = forced;
    }
    return depth_option_for(k);
  }

  // The MTP replay of batch family `family` verifying `rows` rows per
  // slot (rows_per_request_ for the full block, fewer for a reduced-depth
  // variant: the feeds compacted, the draft over those rows, the block's
  // full width drafted and fed back, every slot's confidence published).
  void record_batch_mtp(int family, int rows) {
    BatchFamily& fam = families_.at(static_cast<size_t>(family));
    const int k = fam.requests;
    const int index = batch_index(family);
    const bool reduced = rows != rows_per_request_;
    const int total = k * rows;
    model_->session_graph_capture_batch(rows, k, reduced ? rows_per_request_ : 0);
    record_stage_gate(index, 0, total);
    picker_->record(model_->stream(), verify_pick_inputs(k, rows));
    glm_publish_seq(d_verdict_seq_ + index, h_verdict_seq_ + index,
                    model_->stream());
    snapshot_verify_rows(/*req=*/0, total, /*first_row=*/0);
    model_->session_graph_capture_commit_batch(picker_->device_verdict(0));
    model_->session_graph_capture_draft_batch(picker_->device_verdict(0));
    DevicePicker::Inputs draft_b =
        draft_pick_inputs(picker_->device_verdict(0), k, rows);
    arm_draft_sampling_batch(draft_b, /*draft_index=*/0);
    picker_->record(model_->stream(), draft_b);
    if constexpr (Model::kBatchedDraftChain) {
      // Depth >= 2: every slot's chain rows, one pick each, and the
      // feed of every draft per request.
      std::vector<const PickVerdict*> drafts{picker_->device_verdict(1)};
      for (int c = 1; c < depth_; ++c) {
        model_->session_graph_capture_draft_chain_batch(
            picker_->device_verdict(0), picker_->device_verdict(c),
            /*index=*/c - 1, /*first=*/c == 1, /*last=*/c == depth_ - 1);
        DevicePicker::Inputs chain_b = chain_pick_inputs(k, 1 + c);
        arm_draft_sampling_batch(chain_b, /*draft_index=*/c);
        picker_->record(model_->stream(), chain_b);
        drafts.push_back(picker_->device_verdict(1 + c));
      }
      model_->session_graph_capture_next_tokens_batch(drafts);
    } else {
      model_->session_graph_capture_next_tokens_batch(
          picker_->device_verdict(1));
    }
    // Scheduling: every slot of the batch publishes its block confidence
    // (the next replay of any of them, scalar or batched, is scheduled).
    for (int q = 0; q < k; ++q) record_confidence_publish(q, /*request=*/q);
  }

  // The reduced-depth variant `di` of batch family `family`.
  void ensure_sched_batch_graph(int family, int di) {
    BatchFamily& fam = families_.at(static_cast<size_t>(family));
    std::array<cudaGraphExec_t, 2>& execs = fam.sched_execs.at(static_cast<size_t>(di));
    if (execs[0] != nullptr) return;
    const int rows = 1 + depth_options_.at(static_cast<size_t>(di));
    for (int parity = 0; parity < 2; ++parity)
      execs[static_cast<size_t>(parity)] = capture_variant(
          sched_batch_variant(family, di, parity),
          [&] { record_batch_mtp(family, rows); });
    model_->session_graph_use_batch_contract(rows_per_request_, fam.requests);
    DGPP_LOG_INFO(
        "rank {}: serving reduced-depth row-batched graph variants {}/{} "
        "captured for {} slots x {} rows ({} of the {} drafts)",
        rank_, sched_batch_variant(family, di, 0), sched_batch_variant(family, di, 1),
        fam.requests, rows, rows - 1, depth_);
  }

  void ensure_batch_graph(int family) {
    BatchFamily& fam = families_.at(static_cast<size_t>(family));
    if (fam.execs[0] != nullptr) return;
    const bool mtp = model_->mtp_enabled();
    const int k = fam.requests;
    const int rows = k * rows_per_request_;
    const int index = batch_index(family);
    const auto build = [&](int parity) {
      (void)parity;
      if (mtp) {
        record_batch_mtp(family, rows_per_request_);
        return;
      }
      model_->session_graph_capture_batch(rows_per_request_, k);
      record_stage_gate(index, 0, rows);
      picker_->record(model_->stream(), verify_pick_inputs(k));
      glm_publish_seq(d_verdict_seq_ + index, h_verdict_seq_ + index,
                      model_->stream());
      model_->session_graph_capture_commit_batch(picker_->device_verdict(0));
      model_->session_graph_capture_verify_next_tokens_batch(
          picker_->device_verdict(0));
    };
    for (int parity = 0; parity < 2; ++parity)
      fam.execs[static_cast<size_t>(parity)] = capture_variant(
          batch_variant(family, parity), [&] { build(parity); });
    model_->session_graph_use_batch_contract(rows_per_request_, k);
    DGPP_LOG_INFO(
        "rank {}: serving row-batched graph variants {}/{} captured for {} "
        "slots x {} rows = {} fixed rows{} (a batch is selected at {}+ live "
        "requests, the smallest that covers the live slots)",
        rank_, batch_variant(family, 0), batch_variant(family, 1), k,
        rows_per_request_, rows, mtp ? ", MTP" : "", batch_min_live_);
  }

  // Polls the slot's pinned verdict sequence until the replay's verdict
  // node published it (the verify's pick is done; the tail runs on).
  void wait_verdict(const Replay& r) const {
    const size_t index =
        static_cast<size_t>(r.batched ? batch_index(r.family) : r.req);
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(pick_timeout_ms_);
    uint64_t spins = 0;
    while (__atomic_load_n(h_verdict_seq_ + index, __ATOMIC_ACQUIRE) <
           r.verdict_seq) {
      if ((++spins & 255) == 0) {
        if (std::chrono::steady_clock::now() > deadline)
          throw std::runtime_error(
              "graph engine: the replay's verdict did not publish within the "
              "pick timeout");
        std::this_thread::yield();
      }
    }
  }
  cudaEvent_t end_event(const Replay& r) const {
    return r.batched ? families_[static_cast<size_t>(r.family)]
                           .end_events[static_cast<size_t>(r.parity)]
                     : end_events_[static_cast<size_t>(r.req)]
                                  [static_cast<size_t>(r.parity)];
  }

  // Arms the replay's bus window and enqueues its graph behind whatever
  // runs on the model stream; its end event follows. At most one older
  // replay stays in flight (the bus holds two windows).
  void launch(Replay r) {
    while (inflight_.size() >= 2) settle_front();
    const bool reduced =
        r.depth_option >= 0 && r.depth_option < full_depth_option();
    const int variant =
        r.batched ? (reduced ? sched_batch_variant(r.family, r.depth_option, r.parity)
                             : batch_variant(r.family, r.parity))
        : reduced ? sched_variant(r.req, r.depth_option, r.parity)
                  : scalar_variant(r.req, r.parity);
    cudaGraphExec_t exec =
        r.batched ? (reduced ? families_[static_cast<size_t>(r.family)]
                                   .sched_execs[static_cast<size_t>(r.depth_option)]
                                               [static_cast<size_t>(r.parity)]
                             : families_[static_cast<size_t>(r.family)]
                                   .execs[static_cast<size_t>(r.parity)])
        : reduced ? sched_execs_[static_cast<size_t>(r.req)]
                                [static_cast<size_t>(r.depth_option)]
                                [static_cast<size_t>(r.parity)]
                  : scalar_execs_[static_cast<size_t>(r.req)]
                                 [static_cast<size_t>(r.parity)];
    r.verdict_seq = ++verdict_seq_[static_cast<size_t>(
        r.batched ? batch_index(r.family) : r.req)];
    if (schedule_ && model_->mtp_enabled()) {
      // Every MTP replay publishes the block confidence of the slots it
      // drafts for; the next scalar step of such a slot waits for it.
      if (r.batched) {
        const int k = families_[static_cast<size_t>(r.family)].requests;
        for (int q = 0; q < k; ++q) {
          ++conf_seq_[static_cast<size_t>(q)];
          conf_stale_[static_cast<size_t>(q)] = false;
        }
      } else {
        ++conf_seq_[static_cast<size_t>(r.req)];
        conf_stale_[static_cast<size_t>(r.req)] = false;
      }
    }
    std::string err;
    if (!bus_->graph_replay_arm(&err, variant))
      throw std::runtime_error("graph engine replay arm: " + err);
    if (trace_)
      DGPP_LOG_INFO("rank {}: pipeline launch slot {} parity {} variant {} "
                    "(inflight {})",
                    rank_, r.req, r.parity, variant, inflight_.size());
    DGPP_CUDA_OK(cudaGraphLaunch(exec, model_->stream()));
    DGPP_CUDA_OK(cudaEventRecord(end_event(r), model_->stream()));
    const int slots = r.batched ? families_.at(static_cast<size_t>(r.family)).requests : 1;
    decode_batch_stats_.slots = slots;
    decode_batch_stats_.active = static_cast<int>(r.reqs.size());
    decode_batch_stats_.rows_per_request = r.rows;
    ++decode_batch_stats_.replays;
    ++decode_batch_stats_.replays_by_slots[slots];
    decode_batch_stats_.rows += slots * r.rows;
    decode_batch_stats_.padded_rows += (slots - static_cast<int>(r.reqs.size())) * r.rows;

    if (trace_)
      DGPP_LOG_INFO("rank {}: pipeline launched slot {} parity {}", rank_,
                    r.req, r.parity);
    // (TRIED 2026-09-09 and removed: cudaGraphUpload of the other parity's
    // exec on a side stream while this replay runs, to leave the next
    // launch only its enqueue. The fabric read no gain on Qwen T=1
    // (22.0–22.1 ms off vs 22.1–22.2 on) and GLM slipped — T=1 29.9 → 30.4,
    // MTP 40.4 → 41.5 ms per step — so the upload contends with the running
    // replay rather than hiding behind it.)
    inflight_.push_back(std::move(r));
  }
  // The oldest replay in flight: its end, its bus window (finished in arm
  // order), the stage handshake's verdict, and its draft verdicts (the
  // block's picks at its tail) into the slots' drafts.
  void settle_front() {
    Replay r = std::move(inflight_.front());
    inflight_.pop_front();
    if (trace_)
      DGPP_LOG_INFO("rank {}: pipeline settle slot {} parity {}: waiting end",
                    rank_, r.req, r.parity);
    DGPP_CUDA_OK(cudaEventSynchronize(end_event(r)));
    if (trace_)
      DGPP_LOG_INFO("rank {}: pipeline settle slot {} parity {}: ended", rank_,
                    r.req, r.parity);
    std::string err;
    if (!bus_->graph_replay_finish(pick_timeout_ms_, &err)) {
      bus_->dump_graph_cells("engine finish");
      throw std::runtime_error("graph engine replay finish: " + err);
    }
    if (sampling_) {
      const size_t index =
          static_cast<size_t>(r.batched ? batch_index(r.family) : r.req);
      if (__atomic_load_n(h_stage_late_ + index, __ATOMIC_ACQUIRE) != 0u)
        throw std::runtime_error(
            "graph engine: the stage handshake timed out — the replay's "
            "pick ran before the host staged its masks");
    }
    if (model_->mtp_enabled()) {
      for (const int req : r.reqs) {
        // A slot the host re-drafted (its sampled fallback) carries the
        // true drafts already; the replay's provisional picks are its
        // alone — the other slots' picks stand (2026-09-10: one slot's
        // fallback used to drop every slot's drafts of a batched replay).
        if (std::find(r.redrafted.begin(), r.redrafted.end(), req) !=
            r.redrafted.end())
          continue;
        std::vector<int32_t>& drafts = drafts_[static_cast<size_t>(req)];
        for (int c = 0; c < depth_; ++c) {
          const PickVerdict draft =
              picker_->verdict(1 + c, r.batched ? req : 0);
          if (draft.rows != 1 || draft.accepted != 1 || draft.next < 0 ||
              draft.next >= vocab_)
            throw std::runtime_error(
                "graph engine: invalid draft verdict " + std::to_string(c + 1) +
                " for slot " + std::to_string(req));
          drafts[static_cast<size_t>(c)] = draft.next;
        }
      }
    }
  }
  void settle_older() {
    while (inflight_.size() > 1) settle_front();
  }

 public:
  // Settles every replay in flight: the bus's eager gate is open after
  // this, and every slot's drafts are the block's latest. Any eager use
  // of the bus (a prefill, an eager verify or draft, the host's fallback
  // gathers) must follow a drain; the gates' eager oracles call it.
  void drain() {
    while (!inflight_.empty()) settle_front();
  }

 private:

  // The step's fed rows for slot `req`: the pending token and its drafts.
  std::vector<int64_t> feed_of(int req) const {
    std::vector<int64_t> feed{pending_[static_cast<size_t>(req)]};
    for (const int32_t d : drafts_[static_cast<size_t>(req)]) feed.push_back(d);
    return feed;
  }

  // Drafts 2..depth eagerly off the block's recursion (the open, and the
  // sampled fallback's re-draft), each a greedy pick of the chain row's
  // head. A chain row past the context is skipped: its draft repeats the
  // previous one (any valid id; it cannot stand).
  // Every draft row's proposal for slot `req` says "none" from here on
  // (n = 0): the host re-drafts the whole chain.
  void invalidate_proposal(int req) {
    if (d_proposals_ == nullptr) return;
    for (int t = 0; t < kSampleProposalSlots; ++t) {
      DraftProposal* d = d_proposals_ + static_cast<size_t>(req) * kSampleProposalSlots + t;
      DGPP_CUDA_OK(cudaMemsetAsync(&d->n, 0, sizeof(int32_t), model_->stream()));
      if (h_proposals_ != nullptr)
        h_proposals_[static_cast<size_t>(req) * kSampleProposalSlots + t].n = 0;
    }
    DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
  }

  void chain_drafts_eagerly(int req) {
    std::vector<int32_t>& drafts = drafts_[static_cast<size_t>(req)];
    int runs = 0;
    for (int c = 1; c < depth_; ++c)
      if (model_->session_draft_chain_fits(req, c - 1)) runs = c;
    for (int c = 1; c < depth_; ++c) {
      if (c > runs) {
        drafts[static_cast<size_t>(c)] = drafts[static_cast<size_t>(c - 1)];
        continue;
      }
      (void)model_->session_draft_chain(req, drafts[static_cast<size_t>(c - 1)],
                                        /*index=*/c - 1, /*first=*/c == 1,
                                        /*last=*/c == runs);
      drafts[static_cast<size_t>(c)] =
          picker_->run(model_->stream(), scalar_pick_inputs(1)).next;
    }
    // These chain rows are argmax drafts: no proposal describes them.
    if (depth_ > 1) invalidate_chain_proposals(req);
  }

  void invalidate_chain_proposals(int req) {
    if (d_proposals_ == nullptr) return;
    for (int t = 1; t < kSampleProposalSlots; ++t) {
      DraftProposal* d = d_proposals_ + static_cast<size_t>(req) * kSampleProposalSlots + t;
      DGPP_CUDA_OK(cudaMemsetAsync(&d->n, 0, sizeof(int32_t), model_->stream()));
      if (h_proposals_ != nullptr)
        h_proposals_[static_cast<size_t>(req) * kSampleProposalSlots + t].n = 0;
    }
    DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
  }

  // `rows`: the verify's rows this replay decided (rows_per_request_, or a
  // scheduled scalar replay's 1 + verified drafts).
  std::vector<int32_t> collect_verdict(int req, int verdict_request,
                                       bool batched, int rows = -1) {
    if (rows < 0) rows = rows_per_request_;
    const PickVerdict verify = picker_->verdict(0, verdict_request);
    if (verify.rows != rows || verify.accepted < 1 ||
        verify.accepted > rows)
      throw std::runtime_error(
          "graph engine: invalid verdict for slot " + std::to_string(req) +
          " (rows " + std::to_string(verify.rows) + ", accepted " +
          std::to_string(verify.accepted) + ")");
    model_->session_graph_settle(req, verify.accepted, rows);
    // The prefix cache's hop snapshot (M7 under the multi-row step): armed
    // by the scheduler for the aligned position row 1 sat on; taken from
    // the state after row 0 when row 1 stood — here, before another slot's
    // scalar step can reuse the spec snapshot rows. A one-row verdict
    // landed ON the position; the scheduler's rolling snapshot follows.
    if (hop_slot_[static_cast<size_t>(req)] >= 0) {
      const int slot = hop_slot_[static_cast<size_t>(req)];
      const int64_t hop = hop_position_[static_cast<size_t>(req)];
      hop_slot_[static_cast<size_t>(req)] = -1;
      if (verify.accepted >= 2) {
        const int rows_after = verify.accepted - 1;
        if (model_->session_position(req) != hop + rows_after)
          throw std::runtime_error(
              "graph engine: slot " + std::to_string(req) + " sits at " +
              std::to_string(model_->session_position(req)) + " after a " +
              std::to_string(verify.accepted) + "-row step, the armed hop expects " +
              std::to_string(hop + rows_after));
        arena_.snapshot_post_row0(req, slot, hop,
                                  batched ? req * rows : 0,
                                  rows_after);
      }
    }

    std::vector<int32_t> decided;
    decided.reserve(static_cast<size_t>(verify.accepted));
    for (int row = 0; row < verify.accepted; ++row) {
      const int32_t token = verify.winners[row];
      if (token < 0 || token >= vocab_)
        throw std::runtime_error(
            "graph engine pick out of range for slot " +
            std::to_string(req) + ": " + std::to_string(token));
      decided.push_back(token);
    }
    int32_t next = verify.next;
    // The drafts this step fed (the verify's rows after the first); the
    // slot's drafts are replaced below by the block's new ones.
    const std::vector<int32_t> fed_drafts = drafts_[static_cast<size_t>(req)];
    const bool stochastic = sampled_slot(req);
    const bool full_path = full_path_slot(req);
    if (stochastic) {
      ++sampled_steps_;
      ++slot_sampled_[static_cast<size_t>(req)];
    }
    std::vector<sample::Result>& report =
        pending_logprobs_[static_cast<size_t>(req)];
    if (full_path && !stochastic) {
      // The greedy request through the full path (logprobs or penalties):
      // the device decided the argmax rows under the raw normalizer; the
      // context mirror and the report follow.
      const SampleOutcome& o = picker_->outcome(0, verdict_request);
      std::vector<int32_t>& context = context_[static_cast<size_t>(req)];
      for (int t = 1; t < verify.accepted; ++t)
        context.push_back(fed_drafts[static_cast<size_t>(t - 1)]);
      context.push_back(next);
      if (report_[static_cast<size_t>(req)])
        for (int row = 0; row < verify.accepted; ++row)
          report.push_back(device_result(o, row, verify.winners[row]));
    }
    if (stochastic && model_->mtp_enabled()) {
      // The T-row verify's sampled verdict. The context mirror follows the
      // device count table: a draft joins it only when it stands.
      const SampleOutcome& o = picker_->outcome(0, verdict_request);
      sample::Rng& rng = rng_[static_cast<size_t>(req)];
      std::vector<int32_t>& context = context_[static_cast<size_t>(req)];
      if (!o.sampled)
        throw std::runtime_error(
            "graph engine: the device made no stochastic decision for a "
            "sampled MTP slot " + std::to_string(req));
      DGPP_LOG_DEBUG(
          "rank {}: slot {} device MTP sampling outcome: fallback_row {} "
          "accepted_draft {} counter {} winners {}/{} accepted {} of {}",
          rank_, req, o.fallback_row, o.accepted_draft, o.counter,
          verify.winners[0], verify.winners[1], verify.accepted,
          rows);
      if (o.fallback_row < 0) {
        // The device's draws: one per draft that stood, two for the reject
        // that ended the chain (the test and the residual), one for the
        // last row's plain sample when every draft stood.
        const uint64_t draws = verify.accepted < rows
                                   ? static_cast<uint64_t>(verify.accepted) + 1
                                   : static_cast<uint64_t>(rows);
        if (o.counter != rng.counter + draws)
          throw std::runtime_error(
              "graph engine: the device consumed " +
              std::to_string(o.counter - rng.counter) + " draws for a " +
              std::to_string(rows) + "-row step that committed " +
              std::to_string(verify.accepted) + " (expected " +
              std::to_string(draws) + ")");
        rng.counter = o.counter;  // the draft pick draws on its own stream
        for (int t = 1; t < verify.accepted; ++t)
          context.push_back(fed_drafts[static_cast<size_t>(t - 1)]);
        context.push_back(next);
        if (report_[static_cast<size_t>(req)])
          for (int row = 0; row < verify.accepted; ++row)
            report.push_back(device_result(o, row, verify.winners[row]));
      } else {
        // The host decides between windows: the replay in flight (this
        // one) ends first — its window finished, its provisional draft
        // picks skipped (the host re-drafts below).
        if (!inflight_.empty()) inflight_.back().redrafted.push_back(req);
        drain();
        next = serve_mtp_fallback(req, verdict_request, o, verify, fed_drafts,
                                  &decided, batched, &report);
      }
    } else if (stochastic) {
      const SampleOutcome& o = picker_->outcome(0, verdict_request);
      sample::Rng& rng = rng_[static_cast<size_t>(req)];
      if (!o.sampled)
        throw std::runtime_error(
            "graph engine: the device made no stochastic decision for a "
            "sampled slot " + std::to_string(req));
      DGPP_LOG_DEBUG(
          "rank {}: slot {} device sampling outcome: fallback {} counter {} "
          "normalizer {:.6f} covered {:.6f} next {} logprob {:.4f}",
          rank_, req, o.fallback, o.counter, o.normalizer[0], o.covered_mass[0],
          verify.next, o.logprob[0]);
      if (o.fallback) {
        if (o.counter != rng.counter)
          throw std::runtime_error(
              "graph engine: the device's counter drifted from the host's "
              "on a fallback");
        drain();
        const sample::Result r = serve_fallback(req, verdict_request, o);
        next = r.token;
        decided.back() = next;
        if (report_[static_cast<size_t>(req)]) report.push_back(r);
        // The graph fed itself the provisional token; the true one replaces
        // it in the slot's persistent feed before the next replay.
        model_->session_graph_seed_feed(req, {next});
      } else {
        if (o.counter != rng.counter + 1)
          throw std::runtime_error(
              "graph engine: the device consumed " +
              std::to_string(o.counter - rng.counter) +
              " draws for one T=1 step");
        rng.counter = o.counter;
        if (report_[static_cast<size_t>(req)])
          report.push_back(device_result(o, 0, next));
      }
      context_[static_cast<size_t>(req)].push_back(next);
    }
    // Count the final verdict: a sampled fallback can accept drafts that
    // the device provisionally rejected. The decided block contains one
    // non-speculative token plus its accepted draft prefix. Attempts still
    // cover only the positions verified by this scheduled replay.
    for (int p = 0; p + 1 < rows; ++p) {
      const size_t pi = static_cast<size_t>(p);
      ++mtp_attempts_[pi];
      ++slot_mtp_attempts_[static_cast<size_t>(req)][pi];
      if (decided.size() > static_cast<size_t>(p + 1)) {
        ++mtp_accepts_[pi];
        ++slot_mtp_accepts_[static_cast<size_t>(req)][pi];
      }
    }
    pending_[static_cast<size_t>(req)] = next;
    if (std::unique_ptr<text::GrammarState>& grammar =
            grammar_[static_cast<size_t>(req)]) {
      // The committed tokens advance the grammar in transcript order; the
      // sampler never produced one outside its mask, so the state stays
      // live (a dead state would mean a contract breach upstream).
      for (const int32_t token : decided) grammar->advance(token);
      if (!grammar->active() && grammar->spec().active())
        DGPP_LOG_WARN(
            "rank {}: slot {} grammar died on a committed token — the pick "
            "left its mask (state {})",
            rank_, req, grammar->state_name());
    }
    // The block's new drafts land when the replay's tail settles
    // (settle_front), unless the host re-drafted.
    mtp_redrafted_ = false;
    return decided;
  }

  // The sampled MTP step's fallback (DESIGN §9/§10) at row t = the outcome's
  // fallback_row: the device committed rows [0, t] (the consumed token and
  // the drafts before row t, which stood), and the in-graph draft ran on
  // those provisional rows, so the block rolls back to its ring snapshot
  // first. The host gathers row t from the verify snapshot and decides it
  // under the device's normalizer: a row before the last tests its draft —
  // a reject is the residual token; a stand means the draft joins the
  // count table and the context, and the verify's next row re-runs eagerly
  // and is decided the same way (the chain continues on the host exactly
  // as the device would have, draw for draw); the last row is sampled
  // plainly. Then the true rows re-draft eagerly (the chain included), the
  // feed is reseeded, the counter is pushed. Returns the new next token
  // and rewrites `decided`.
  int32_t serve_mtp_fallback(int req, int verdict_request,
                             const SampleOutcome& o,
                             const PickVerdict& verify,
                             const std::vector<int32_t>& fed_drafts,
                             std::vector<int32_t>* decided, bool batched,
                             std::vector<sample::Result>* report) {
    const bool reporting = report_[static_cast<size_t>(req)];
    sample::Rng& rng = rng_[static_cast<size_t>(req)];
    std::vector<int32_t>& context = context_[static_cast<size_t>(req)];
    const sample::Params& p = params_[static_cast<size_t>(req)];
    const int count = model_->lm_vocab_count();
    const int begin = model_->lm_vocab_begin();
    const int T = rows_per_request_;
    const int t0 = o.fallback_row;
    (void)verdict_request;
    // The device's draws: one per draft that stood before row t0 (the
    // fallback row's own draw is the host's).
    if (t0 < 0 || t0 >= T || verify.accepted != t0 + 1 ||
        o.counter != rng.counter + static_cast<uint64_t>(t0))
      throw std::runtime_error(
          "graph engine: a row-" + std::to_string(t0) +
          " fallback must commit the rows before it (" +
          std::to_string(verify.accepted) + " committed) with their draws "
          "consumed (" + std::to_string(o.counter - rng.counter) + ")");
    rng.counter = o.counter;
    // The draft block ran on the provisional rows: back to its snapshot.
    model_->session_draft_rollback(req, verify.accepted);
    // The drafts the device accepted before row t0 joined its count table;
    // the mirror and the report follow.
    std::vector<int32_t> rows;  // the rows' tokens: the block's next inputs
    for (int t = 0; t < t0; ++t) {
      const int32_t d = fed_drafts[static_cast<size_t>(t)];
      context.push_back(d);
      rows.push_back(d);
      if (reporting) report->push_back(device_result(o, t, d));
    }
    // The verify rows from the snapshot the graph took before its draft
    // (the logits buffer itself holds the draft head's rows now); the
    // snapshot is indexed [slot][row], the scalar and the batch alike.
    const auto gather_row = [&](size_t t) {
      const float* src = d_verify_logits_ +
                         (static_cast<size_t>(req) * rows_per_request_ + t) * count;
      DGPP_CUDA_OK(cudaMemcpyAsync(h_fallback_row_, src, sizeof(float) * count,
                                   cudaMemcpyDeviceToHost, model_->stream()));
      DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
      bus_gather_logits(*bus_, rank_, world_, h_fallback_row_, count, begin,
                        static_cast<int>(vocab_), sample_gather_scratch_,
                        pick_timeout_ms_, &fallback_full_);
    };
    gather_row(static_cast<size_t>(t0));
    check_gathered_row(o.covered_mass[t0], o.normalizer[t0], p,
                       t0 == 0 ? "row 0" : "row");
    int32_t next = -1;
    int t = t0;
    bool gathered = true;  // row t is the gathered snapshot row
    typename Model::Outputs eager;  // else the eagerly re-run row t
    for (;;) {
      const text::TokenMask& m =
          masks_[static_cast<size_t>(req) * rows_per_request_ + t];
      const text::TokenMask* mask = m.constrained() ? &m : nullptr;
      if (t + 1 < T) {
        const int32_t draft = fed_drafts[static_cast<size_t>(t)];
        sample::SpecPrefixDecision d;
        if (gathered) {
          d = sample::spec_accept_complete(fallback_full_.data(),
                                               static_cast<int>(vocab_),
                                               o.normalizer[t], draft, p, rng);
          bus_check_decision_digest(*bus_, rank_, d.accepted, d.result,
                                    o.normalizer[t], sample_prefix_scratch_,
                                    pick_timeout_ms_, "graph MTP fallback row");
        } else {
          // Row t under the mask staged for it (the grammar advanced by
          // the drafts), the bias, and the context as it stands.
          step_timing::Scope tick(step_timing::kPick);
          d = bus_spec_accept(*bus_, rank_, world_, eager.logits.data(),
                              static_cast<int>(eager.lm_vocab_count),
                              eager.lm_vocab_begin, static_cast<int>(vocab_),
                              draft, p, rng, context, kSamplingCandidates,
                              sample_prefix_scratch_, sample_gather_scratch_,
                              pick_timeout_ms_, &fallback_full_, mask,
                              bias_row(req));
          if (!d.resolved)
            throw std::runtime_error(
                "graph engine: the eager fallback row did not resolve");
        }
        if (reporting) report->push_back(d.result);
        if (!d.accepted) {
          next = d.result.token;
          rows.push_back(next);
          break;
        }
        // The draft stood after all: it joins the device count table (the
        // verdict committed only the rows before it), the context and the
        // rows, then the verify's next row runs eagerly.
        device_sample_adjust_count(d_counts_ + static_cast<size_t>(req) * vocab_,
                                draft, +1, static_cast<int>(vocab_),
                                model_->stream());
        DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
        context.push_back(draft);
        rows.push_back(draft);
        eager = model_->session_verify(req, std::vector<int64_t>{draft});
        gathered = false;
        ++t;
      } else {
        sample::Result r;
        if (gathered) {
          r = sample::sample_complete_logits(fallback_full_.data(),
                                             static_cast<int>(vocab_),
                                             o.normalizer[t], p, rng);
          bus_check_decision_digest(*bus_, rank_, true, r, o.normalizer[t],
                                    sample_prefix_scratch_, pick_timeout_ms_,
                                    "graph MTP fallback last row");
        } else {
          r = prefill_sample_(eager, p, rng, context, mask, bias_row(req));
        }
        if (reporting) report->push_back(r);
        next = r.token;
        rows.push_back(next);
        break;
      }
    }
    if (next < 0 || next >= vocab_)
      throw std::runtime_error("graph engine MTP fallback token out of range");
    context.push_back(next);
    *decided = rows;
    // The true rows through the draft block, eagerly; the greedy draft
    // picks, the chain included.
    std::vector<int32_t>& drafts = drafts_[static_cast<size_t>(req)];
    drafts[0] = prefill_pick_(model_->session_draft(
        req, std::vector<int64_t>(rows.begin(), rows.end())));
    // This draft is the host's argmax, not a draw from the draft head's
    // distribution, so the proposal the graph's draft pick left behind no
    // longer describes it: clear it and let the next verify use the plain
    // rule (2026-09-10; the ratio rule is only exact for a draft actually
    // drawn from the proposal it is tested against).
    invalidate_proposal(req);
    chain_drafts_eagerly(req);
    // The graph published the provisional drafts' confidence; the true
    // drafts' is on the device only. Copy it (no replay of this slot is
    // in flight: the fallback drained).
    refresh_confidence_eagerly(req);
    mtp_redrafted_ = true;
    {
      std::vector<int64_t> feed{next};
      for (const int32_t d : drafts) feed.push_back(d);
      model_->session_graph_seed_feed(req, feed);
    }
    (void)batched;
    push_counter(req);
    ++fallbacks_;
    ++slot_fallbacks_[static_cast<size_t>(req)];
    return next;
  }

  std::vector<int32_t> step_scalar(int req) {
    ensure_scalar_graph(req);
    // The scheduled verify depth: the option (and its rows) this replay
    // verifies at; the full block when scheduling is off.
    int di = -1;
    int rows = rows_per_request_;
    if (schedule_ && model_->mtp_enabled()) {
      di = choose_depth_option(req);
      rows = 1 + depth_options_[static_cast<size_t>(di)];
      if (di < full_depth_option()) ensure_sched_graph(req, di);
      ++sched_hist_[static_cast<size_t>(di)];
    }
    // The pinned row metadata (the MTP feed itself is device-resident; a
    // plain T=1 graph uploads the staged token at its start).
    if (model_->mtp_enabled()) {
      std::vector<int64_t> feed = feed_of(req);
      if (rows < static_cast<int>(feed.size())) feed.resize(static_cast<size_t>(rows));
      model_->session_graph_stage(req, feed);
    } else {
      model_->session_graph_stage(req, pending_[static_cast<size_t>(req)]);
    }
    // Launch first: the window armed and the graph enqueued behind the
    // previous replay's tail. Then settle that tail (its drafts are this
    // replay's fed rows, already on the device; the masks need them),
    // stage the masks for this replay's pick and publish, and wait for
    // the verdict — the draft block's tail runs on while the host works.
    Replay r;
    r.req = req;
    r.parity = scalar_parity_[static_cast<size_t>(req)];
    scalar_parity_[static_cast<size_t>(req)] ^= 1;
    r.reqs = {req};
    r.depth_option = di;
    r.rows = rows;
    launch(std::move(r));
    settle_older();
    stage_masks(req);
    publish_stage(req);
    if (!pipeline_) drain();  // after the publish: the replay's gate waits on it
    wait_verdict(inflight_.back());
    std::vector<int32_t> out = collect_verdict(req, /*verdict_request=*/0, /*batched=*/false, rows);
    if (schedule_ && model_->mtp_enabled()) note_step_for_lambda(static_cast<int>(out.size()), rows);
    return out;
  }

  // The slot's masks for the coming replay (M6 6g): row 0 under the
  // grammar's current state, row 1 (MTP) under the state advanced by the
  // pending draft (row 1 is used only when the draft stands, in which case
  // that is exactly its position; a draft outside the mask kills the copy
  // and leaves row 1 unconstrained — it is discarded either way). Written
  // to the device table on the model stream ahead of the graph launch; an
  // unconstrained position writes a zero header.
  void stage_masks(int req) {
    if (!sampling_) return;
    std::unique_ptr<text::GrammarState>& grammar = grammar_[static_cast<size_t>(req)];
    uint32_t* h = h_masks_ + static_cast<size_t>(req) * rows_per_request_ * mask_stride_;
    if (!grammar) {
      // The headers are zero (configure/close) — unless a reduced-depth
      // batch staged its compact layout over these rows meanwhile.
      if (schedule_)
        for (int t = 0; t < rows_per_request_; ++t) h[static_cast<size_t>(t) * mask_stride_] = 0u;
      return;
    }
    stage_mask_rows(req, rows_per_request_, h);
    // The device table takes the rows through the replay's upload node,
    // behind the stage handshake (record_stage_gate).
  }
  // Slot `req`'s masks for `rows` rows into the pinned rows at `h`: row 0
  // under the grammar's current state, row t under the state advanced by
  // the drafts before it (a draft outside its mask kills the copy: the
  // rows after stay unconstrained and are discarded with it).
  void stage_mask_rows(int req, int rows, uint32_t* h) {
    std::unique_ptr<text::GrammarState>& grammar = grammar_[static_cast<size_t>(req)];
    text::TokenMask* m = &masks_[static_cast<size_t>(req) * rows_per_request_];
    grammar->mask(&m[0]);
    if (rows > 1) {
      text::GrammarState after = *grammar;
      const std::vector<int32_t>& drafts = drafts_[static_cast<size_t>(req)];
      for (int t = 1; t < rows; ++t) {
        after.advance(drafts[static_cast<size_t>(t - 1)]);
        after.mask(&m[t]);
      }
    }
    for (int t = 0; t < rows; ++t) {
      uint32_t* row = h + static_cast<size_t>(t) * mask_stride_;
      row[0] = m[t].constrained() ? static_cast<uint32_t>(m[t].allowed) : 0u;
      if (m[t].constrained())
        std::copy(m[t].words.begin(), m[t].words.end(), row + 1);
    }
  }
  // The reduced-depth batch's masks: the k slots' rows compacted at
  // `rows` per slot (the pick's row r is mask row r), every header
  // written (zero for an unconstrained or closed slot).
  void stage_masks_compact(int k, int rows) {
    if (!sampling_) return;
    for (int q = 0; q < k; ++q) {
      uint32_t* h = h_masks_ + static_cast<size_t>(q) * rows * mask_stride_;
      if (live_[static_cast<size_t>(q)] && grammar_[static_cast<size_t>(q)]) {
        stage_mask_rows(q, rows, h);
      } else {
        for (int t = 0; t < rows; ++t) h[static_cast<size_t>(t) * mask_stride_] = 0u;
      }
    }
  }
  // The gathered row is the row the device decided over iff its prefix's
  // covered mass under the device's normalizer is the device's, bit for
  // bit (the same candidates, the same masses, the same order) — the
  // invariant a wrong row breaks (the draft head's row did, silently,
  // until the verify snapshot). Loud, never papered over.
  void check_gathered_row(double device_covered, double normalizer,
                          const sample::Params& p, const char* what) const {
    const std::vector<sample::Candidate> top = sample::local_topk(
        fallback_full_.data(), static_cast<int>(vocab_), 0, candidates_);
    double covered = 0.0;
    for (const sample::Candidate& c : top)
      covered += detmath::exp_d(
          static_cast<double>(c.logit / p.temperature) - normalizer);
    if (std::memcmp(&covered, &device_covered, sizeof(double)) != 0)
      throw std::runtime_error(std::format(
          "graph engine: the gathered fallback {} is not the row the device "
          "decided over (covered mass {:.9g} vs the device's {:.9g}) — the "
          "verify snapshot and the pick disagree",
          what, covered, device_covered));
  }

  // Records the copy of the verify's logits rows [first_row, first_row +
  // rows) — as the pick left them — into the snapshot at slot `req`'s
  // rows, before the draft block overwrites the buffer (a kernel node:
  // the decode graph is kernels-only).
  void snapshot_verify_rows(int req, int rows, int first_row) {
    if (d_verify_logits_ == nullptr) return;
    const size_t count = static_cast<size_t>(model_->lm_vocab_count());
    glm_device_copy(
        d_verify_logits_ + static_cast<size_t>(req) * rows_per_request_ * count,
        model_->device_logits() + static_cast<size_t>(first_row) * count,
        sizeof(float) * static_cast<size_t>(rows) * count, model_->stream());
  }
  // Zero headers for the slot's rows: unconstrained until staged again.
  void clear_masks(int req) {
    if (!sampling_ || d_masks_ == nullptr) return;
    uint32_t* h = h_masks_ + static_cast<size_t>(req) * rows_per_request_ * mask_stride_;
    for (int t = 0; t < rows_per_request_; ++t) h[static_cast<size_t>(t) * mask_stride_] = 0u;
    for (int t = 0; t < rows_per_request_; ++t) {
      const size_t off =
          (static_cast<size_t>(req) * rows_per_request_ + t) * mask_stride_;
      DGPP_CUDA_OK(cudaMemcpyAsync(d_masks_ + off, h_masks_ + off,
                                   sizeof(uint32_t), cudaMemcpyHostToDevice,
                                   model_->stream()));
    }
    DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
  }

  bool sampled_slot(int req) const {
    return sampling_ && params_[static_cast<size_t>(req)].temperature > 0.0f;
  }
  // The device outcome's report for one decided row, as the host's Result.
  static sample::Result device_result(const SampleOutcome& o, int row,
                                          int32_t token) {
    sample::Result r;
    r.token = token;
    r.logprob = o.logprob[row];
    for (int i = 0; i < o.top_count[row]; ++i)
      r.top_logprobs.emplace_back(o.top_ids[row][i], o.top_logprobs[row][i]);
    return r;
  }
  // Slots that take the full sampling path on the device and the host
  // sampler at the prefill: stochastic ones, and greedy ones that report
  // logprobs or carry penalties (decided as the argmax under the raw
  // distribution, bitwise the greedy pick's token).
  bool full_path_slot(int req) const {
    if (!sampling_) return false;
    const sample::Params& p = params_[static_cast<size_t>(req)];
    return p.temperature > 0.0f || report_[static_cast<size_t>(req)] ||
           p.repetition_penalty != 1.0f || p.frequency_penalty != 0.0f ||
           p.presence_penalty != 0.0f ||
           grammar_[static_cast<size_t>(req)] != nullptr ||
           !bias_[static_cast<size_t>(req)].empty();
  }

  void push_spec(int req, const SampleSpec& spec) {
    h_specs_[req] = spec;
    DGPP_CUDA_OK(cudaMemcpyAsync(d_specs_ + req, h_specs_ + req,
                                 sizeof(SampleSpec), cudaMemcpyHostToDevice,
                                 model_->stream()));
    if (d_draft_specs_ != nullptr) {
      // The draft picks' spec: the slot's, reporting logprobs when the
      // scheduled verify depth reads the draft head's probabilities (the
      // full path decides the same argmax under the raw normalizer).
      SampleSpec draft = spec;
      if (draft_full_path_) draft.logprobs = std::max(0, spec.logprobs);
      DGPP_CUDA_OK(cudaMemcpyAsync(d_draft_specs_ + req, &draft,
                                   sizeof(SampleSpec), cudaMemcpyHostToDevice,
                                   model_->stream()));
    }
    DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
  }
  void push_counter(int req) {
    SampleSpec spec = h_specs_[req];
    spec.counter = rng_[static_cast<size_t>(req)].counter;
    push_spec(req, spec);
  }

  // The exact fallback between windows (DESIGN §10): the device penalized
  // the request's logits row in place and folded the normalizer; the host
  // gathers the row from every rank, decides over the complete list with
  // the reserved draw, echoes rank 0's digest, and pushes the advanced
  // counter back to the device spec.
  sample::Result serve_fallback(int req, int verdict_request,
                                    const SampleOutcome& o) {
    const int count = model_->lm_vocab_count();
    const int begin = model_->lm_vocab_begin();
    const size_t row = static_cast<size_t>(verdict_request) * rows_per_request_;
    DGPP_CUDA_OK(cudaMemcpyAsync(h_fallback_row_,
                                 model_->device_logits() + row * count,
                                 sizeof(float) * count, cudaMemcpyDeviceToHost,
                                 model_->stream()));
    DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
    bus_gather_logits(*bus_, rank_, world_, h_fallback_row_, count, begin,
                      static_cast<int>(vocab_), sample_gather_scratch_,
                      pick_timeout_ms_, &fallback_full_);
    sample::Rng& rng = rng_[static_cast<size_t>(req)];
    check_gathered_row(o.covered_mass[0], o.normalizer[0],
                       params_[static_cast<size_t>(req)], "row");
    const sample::Result r = sample::sample_complete_logits(
        fallback_full_.data(), static_cast<int>(vocab_), o.normalizer[0],
        params_[static_cast<size_t>(req)], rng);
    bus_check_decision_digest(*bus_, rank_, /*resolved=*/true, r,
                              o.normalizer[0], sample_prefix_scratch_,
                              pick_timeout_ms_, "graph sample fallback");
    if (r.token < 0 || r.token >= vocab_)
      throw std::runtime_error("graph engine fallback token out of range: " +
                               std::to_string(r.token));
    push_counter(req);
    ++fallbacks_;
    ++slot_fallbacks_[static_cast<size_t>(req)];
    return r;
  }

 public:
  // Fallbacks served so far (the measurement record's fallback rate).
  uint64_t fallbacks() const { return fallbacks_; }

 private:

  void log_mode_change(bool batched, size_t live, int family) {
    const int mode = batched ? 1 + family : 0;
    if (last_mode_ == mode) return;
    last_mode_ = mode;
    const int rows = batched ? families_[static_cast<size_t>(family)].requests *
                                   rows_per_request_
                             : rows_per_request_;
    DGPP_LOG_DEBUG(
        "rank {}: adaptive decode selected {} graph ({} rows) at {} live "
        "request{} (batch crossover {})",
        rank_, batched ? "row-batched" : "scalar", rows, live,
        live == 1 ? "" : "s", batch_min_live_);
  }

  Model* model_ = nullptr;
  net::CollectiveBus* bus_ = nullptr;
  int rank_ = 0;
  int world_ = 1;
  int64_t vocab_ = 0;
  int pick_timeout_ms_ = 60000;
  uint16_t* sample_prefix_scratch_ = nullptr;
  uint16_t* sample_gather_scratch_ = nullptr;
  bool sampling_ = false;
  int candidates_ = 0;
  DraftProposal* d_proposals_ = nullptr;  // device [slots][proposal slots]
  DraftProposal* h_proposals_ = nullptr;  // pinned mirror, the fallback's
  SampleSpec* d_specs_ = nullptr;   // device [slots]
  SampleSpec* h_specs_ = nullptr;   // pinned mirror
  int32_t* d_counts_ = nullptr;        // device [slots][vocab]
  int64_t* d_prompt_ids_ = nullptr;    // device [max_context]
  int64_t* h_prompt_ids_ = nullptr;    // pinned [max_context]
  float* h_fallback_row_ = nullptr;    // pinned [lm_vocab_count]
  std::vector<float> fallback_full_;
  // Constrained decoding (M6 6g): the grammar per slot, the two staged
  // masks per slot, the device/pinned mask table [slots*rows][stride].
  const text::GrammarVocab* grammar_vocab_ = nullptr;
  std::vector<std::unique_ptr<text::GrammarState>> grammar_;
  std::vector<text::TokenMask> masks_;
  uint32_t* d_masks_ = nullptr;
  uint32_t* h_masks_ = nullptr;
  int mask_stride_ = 0;
  // The logit bias: the device table [slots][vocab] the pick
  // reads for the rows whose spec says biased, a pinned row for uploads,
  // and the host copy per slot for the host-side decisions.
  float* d_bias_ = nullptr;
  float* h_bias_row_ = nullptr;
  std::vector<std::vector<float>> bias_;
  float* d_verify_logits_ = nullptr;  // device [slots][rows][count]: the
                                      // verify rows the MTP fallback decides over
  std::vector<sample::Params> params_;
  std::vector<sample::Rng> rng_;
  std::vector<std::vector<int32_t>> context_;  // prompt + decided, per slot
  std::vector<bool> report_;                    // logprobs asked, per slot
  std::vector<std::vector<sample::Result>> pending_logprobs_;
  DecodeSample prefill_sample_;
  uint64_t fallbacks_ = 0;
  uint64_t sampled_steps_ = 0;  // stochastic collects (the fallback rate's base)
  std::vector<uint64_t> slot_sampled_, slot_fallbacks_;  // per slot, reset at close
  bool mtp_redrafted_ = false;  // this collect re-drafted on the host
  int slots_ = 0;
  int rows_per_request_ = 1;
  int max_rows_ = kDecodeRows;      // the model's decode-row ceiling
  bool batch_unavailable_ = false;  // slots * rows past it (or no batched chain): scalar only
  // Per-position draft acceptance: engine-wide and per slot.
  std::array<uint64_t, 8> mtp_attempts_{}, mtp_accepts_{};
  std::vector<std::array<uint64_t, 8>> slot_mtp_attempts_, slot_mtp_accepts_;
  std::vector<int> hop_slot_;          // per slot: the armed hop's arena slot, -1 none
  std::vector<int64_t> hop_position_;  // per slot: the armed hop's position
  int batch_min_live_ = 1;
  int last_mode_ = -1;  // 0 scalar variants, 1 + family for a row batch
  DecodePick prefill_pick_;
  std::unique_ptr<DevicePicker> picker_;
  // Recorder storage is referenced by graph nodes and must outlive the exec.
  std::unique_ptr<GraphRecordReducer> recorder_;
  // The pipelined replay: two graph variants per slot and two
  // for the batch, alternating per replay, so the next replay's bus window
  // is armed on a variant whose cells are not in flight while the previous
  // replay of the same shape still runs. A replay is launched, its verdict
  // awaited on an event recorded in the graph right after the verify's
  // pick, and its END (the draft block's tail) settled after the NEXT
  // replay was launched — that is where the host's interface hides.
  std::vector<std::array<cudaGraphExec_t, 2>> scalar_execs_;
  std::vector<int> scalar_parity_;
  std::vector<std::array<cudaEvent_t, 2>> end_events_;
  // The batch families: by slot count, ascending — 2, 3 and
  // every slot where the rows allow — each with its two parities' execs
  // and end events and its own parity clock; the steps each replayed.
  struct BatchFamily {
    int requests = 0;
    std::array<cudaGraphExec_t, 2> execs{{nullptr, nullptr}};
    std::array<cudaEvent_t, 2> end_events{{nullptr, nullptr}};
    int parity = 0;
    // The scheduled verify depth: the family's reduced-depth variants
    // [option] (the last option is `execs`) and its replays per option.
    std::vector<std::array<cudaGraphExec_t, 2>> sched_execs;
    std::vector<uint64_t> sched_hist;
  };
  sched::SchedulerEngine::DecodeBatchStats decode_batch_stats_;
  std::vector<BatchFamily> families_;
  std::vector<uint64_t> family_steps_;
  // The verdict's publication: per slot (and per batch family, at index
  // slots_ + family)
  // the device replay counter the verdict node bumps, its pinned mirror
  // the host polls, and the host's expected count.
  uint64_t* h_verdict_seq_ = nullptr;
  uint64_t* d_verdict_seq_ = nullptr;
  std::vector<uint64_t> verdict_seq_;
  std::deque<Replay> inflight_;  // launched, end not yet settled (<= 2)
  // DGPP_PIPELINE=0 settles every replay right after its launch (no
  // overlap): the bisecting knob and the operational escape hatch.
  bool pipeline_ = [] {
    const char* v = std::getenv("DGPP_PIPELINE");
    return v == nullptr || std::string(v) != "0";
  }();

  bool trace_ = std::getenv("DGPP_PIPELINE_TRACE") != nullptr;

  // The stage handshake: per slot (and per batch family, at slots_ + f) the
  // host's published stage sequence (pinned), the device's replay counter,
  // and the late flag the wait node sets on a timeout.
  uint64_t* h_stage_seq_ = nullptr;
  uint64_t* d_stage_seq_ = nullptr;
  uint32_t* h_stage_late_ = nullptr;
  std::vector<uint64_t> stage_seq_;
  std::vector<int64_t> pending_;
  // Per slot: the drafts fed with the pending token, one per position
  // (depth_ of them; depth 1 is the two-row step as built).
  std::vector<std::vector<int32_t>> drafts_;
  int depth_ = 0;
  // ---- the confidence-scheduled verify depth (configure_verify_schedule) ----
  bool schedule_ = false;
  float sched_row_ms_ = 0.f;
  float sched_lambda_ = 0.f;          // the configured lambda (the floor when adaptive)
  float sched_base_ms_ = 0.f;         // the modeled step's fixed part (adaptive lambda)
  bool sched_adapt_ = false;          // lambda follows the modeled throughput
  double sched_lambda_live_ = 0.0;    // the EWMA (replicated arithmetic)
  uint64_t sched_lambda_steps_ = 0;
  static constexpr double kSchedLambdaAlpha = 1.0 / 64.0;
  int sched_min_depth_ = 1;
  std::vector<int> depth_options_;  // ascending; the last is depth_ (the full block)
  int conf_rows_ = 0;               // confidence entries per slot (the block)
  float* h_conf_ = nullptr;         // pinned, mapped [slots][conf_rows_]: the published logits
  uint64_t* h_conf_seq_ = nullptr;  // pinned, mapped [slots]: the publication counter
  uint64_t* d_conf_seq_ = nullptr;  // device [slots]
  std::vector<uint64_t> conf_seq_;  // per slot: the publications the host expects
  std::vector<bool> conf_stale_;    // per slot: no publication describes its drafts
  // The reduced-depth scalar variants [slot][option] (the last option is
  // the slot's scalar variant in scalar_execs_).
  std::vector<std::vector<std::array<cudaGraphExec_t, 2>>> sched_execs_;
  std::vector<uint64_t> sched_hist_;  // per option: the scalar replays at that depth
  uint64_t sched_full_forced_ = 0;    // steps held at the full block (fresh/sampled slot)
  std::function<int(int, int, const float*, int)> depth_hook_;  // tests
  // A family without a confidence head takes the draft-probability
  // confidence (device_sample_draft_confidence): its draft picks run the
  // sampler's full path so their outcomes carry the draft's log-probability
  // — the picks read a spec table whose rows report logprobs.
  bool draft_full_path_ = false;
  SampleSpec* d_draft_specs_ = nullptr;  // device [slots]: the draft picks' specs
  float* d_draft_conf_ = nullptr;        // device [slots][conf_rows_]: the gathered logits
  std::vector<bool> live_;
  std::vector<bool> reserved_;
  PrefixArena<Model> arena_;  // the prefix cache's snapshot slots (M7)
};

}  // namespace dgpp
