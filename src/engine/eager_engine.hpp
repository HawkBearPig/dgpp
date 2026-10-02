#pragma once
// Scheduler adapter for eager model execution and token selection.
// The caller supplies pick callbacks for its world: a full-vocabulary pick
// at world 1 or a distributed pick over rank-local logits. Distributed
// picks are collectives, so every rank must call them in the same order.
//
// An optional Sample callback enables stochastic sampling. Each request
// slot retains its parameters, counter RNG and token context for penalties.
// configure_sampling() initializes that state before the prefill pick.
// Nonpositive temperature uses the greedy callback; an adapter without a
// Sample callback reports that it supports greedy decoding only.
//
// Allocate pick buffers before decoding. Allocating device-related memory
// inside a pick can synchronize with another rank's spinning collective.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "engine/decode_outputs.hpp"
#include "engine/image_prefill.hpp"
#include "engine/prefix_arena.hpp"
#include "sample/sampler.hpp"
#include "sched/scheduler.hpp"
#include "text/tool_grammar.hpp"
#include "engine/speculative.hpp"

namespace dgpp {

// CONSTRAINED DECODING (M6 6g): an adapter built with a GrammarVocab keeps
// a grammar state per slot (configure_constraint), hands the next
// position's mask to the sampler closure with every decision — a
// constrained slot decides through the closure whatever its temperature,
// the masked argmax at 0 — and advances the state with the committed
// token. The mask is null for unconstrained slots, so nothing changes for
// them.
template <class Model>
class EagerEngineAdapter : public sched::SchedulerEngine {
 public:
  using Pick = DecodePick;
  // `mask` is null when the position is unconstrained; `bias`
  // is the request's dense logit_bias row over [0, this rank's slice end),
  // null when the request has none.
  using Sample = DecodeSample;

  // `prefix_slots` (M7): snapshot slots of the prefix cache's arena this
  // engine holds (0: no cache; the scheduler then never calls the prefix
  // ops and the op stream is the pre-cache one).
  EagerEngineAdapter(Model* model, int max_requests, Pick pick,
                   Sample sample = nullptr,
                   const text::GrammarVocab* grammar_vocab = nullptr,
                   int prefix_slots = 0)
      : model_(model),
        slots_(max_requests),
        pick_(std::move(pick)),
        sample_(std::move(sample)),
        grammar_vocab_(grammar_vocab),
        pending_(static_cast<size_t>(max_requests), -1),
        state_(static_cast<size_t>(max_requests)),
        arena_(model, prefix_slots) {
    spec_.resize(static_cast<size_t>(max_requests));
    if constexpr (requires { model_->set_prefill_monitor(prefill_monitor()); })
      model_->set_prefill_monitor(prefill_monitor());
  }

  int max_concurrent_requests() const override { return slots_; }
  int64_t pool_blocks_total() const override {
    return model_->kv_blocks_total();
  }
  int64_t pool_blocks_in_use() const override {
    return model_->kv_blocks_in_use();
  }
  int64_t blocks_for_tokens(int64_t tokens) const override {
    return model_->kv_blocks_for_tokens(tokens);
  }

  bool supports_sampling() const override {
    return static_cast<bool>(sample_);
  }
  bool supports_logprobs() const override {
    return static_cast<bool>(sample_);
  }
  void configure_sampling(int req, const sample::Params& sampling,
                          uint64_t seed) override {
    sample::validate_params(sampling);
    if (sampling.temperature > 0.0f && !sample_)
      throw std::logic_error(
          "generation engine: no sampler bound — this engine is greedy-only");
    SlotState& s = state_.at(static_cast<size_t>(req));
    s.params = sampling;
    s.rng = sample::Rng{seed, 0};
    s.context.clear();
    s.logprobs.clear();
  }
  void configure_logprobs(int req, int logprobs) override {
    if (logprobs >= 0 && !sample_)
      throw std::logic_error(
          "generation engine: no sampler bound — no logprobs either");
    SlotState& s = state_.at(static_cast<size_t>(req));
    s.report_logprobs = logprobs >= 0;
    s.logprobs.clear();
  }
  std::vector<sample::Result> take_logprobs(int req) override {
    SlotState& s = state_.at(static_cast<size_t>(req));
    std::vector<sample::Result> out;
    out.swap(s.logprobs);
    return out;
  }
  bool supports_constraints() const override {
    return grammar_vocab_ != nullptr && grammar_vocab_->usable() &&
           static_cast<bool>(sample_);
  }
  void configure_constraint(int req, const text::GrammarSpec& grammar) override {
    SlotState& s = state_.at(static_cast<size_t>(req));
    s.grammar.reset();
    if (!grammar.active()) return;
    if (!supports_constraints())
      throw std::logic_error(
          "generation engine: no grammar vocabulary — this engine cannot "
          "constrain the pick");
    s.grammar = std::make_unique<text::GrammarState>(
        grammar_vocab_, grammar, /*prompt_opens_thinking=*/true);
  }

  int32_t prefill(int req, const std::vector<int64_t>& prompt) override {
    return open_slot(req, prompt, [&] { return model_->session_prefill(req, prompt); });
  }
  bool supports_images() const override {
    if constexpr (requires { model_->supports_images(); }) return model_->supports_images();
    return false;
  }
  ImageTokens image_token_ids() const override {
    if constexpr (requires { model_->image_tokens(); }) return model_->image_tokens();
    return {};
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

  // ---- prefix cache (M7) --------------------------------------------------
  sched::SchedulerEngine::PrefixInfo prefix_info() const override {
    sched::SchedulerEngine::PrefixInfo info;
    info.arena_slots = arena_.slots();
    info.align = model_->session_snapshot_align();
    info.block_tokens = model_->kv_block_tokens();
    info.chunk_tokens = Model::prefill_chunk_tokens();
    if constexpr (requires { model_->mtp_enabled(); })
      info.prefill_lookahead = model_->mtp_enabled();
    // The DFlash2 planes at position p are functions of token p too.
    if constexpr (requires { model_->dflash2_enabled(); })
      info.prefill_lookahead = info.prefill_lookahead || model_->dflash2_enabled();
    if constexpr (requires { model_->prefill_bounded(); })
      info.body_snapshots = !model_->prefill_bounded();
    return info;
  }
  int32_t prefill_cached(int req, const std::vector<int64_t>& prompt,
                         sched::SchedulerEngine::PrefixPrefill* plan) override {
    if (plan == nullptr || plan->boundaries == nullptr)
      throw std::invalid_argument("generation engine: prefill_cached without a plan");
    return open_slot(req, prompt, [&] {
      typename Model::SnapshotRequest snap, body_snap, head_snap;
      typename Model::SnapshotRequest* snap_ptr = nullptr;
      if (plan->snap_slot >= 0) {
        snap = arena_.request(plan->snap_slot, plan->snap_position);
        snap_ptr = &snap;
      }
      if (plan->body_snap_slot >= 0) {
        body_snap = arena_.request(plan->body_snap_slot, plan->body_snap_position);
        body_snap.next = snap_ptr;
        snap_ptr = &body_snap;
      }
      if (plan->head_snap_slot >= 0) {
        head_snap = arena_.request(plan->head_snap_slot, plan->head_snap_position);
        head_snap.next = snap_ptr;
        snap_ptr = &head_snap;
      }
      const auto commit = [&] {
        if (plan->snap_slot >= 0) {
          arena_.commit(plan->snap_slot, snap);
          plan->snap_taken = snap.taken;
        }
        if (plan->body_snap_slot >= 0) {
          arena_.commit(plan->body_snap_slot, body_snap);
          plan->body_snap_taken = body_snap.taken;
        }
        if (plan->head_snap_slot >= 0) {
          arena_.commit(plan->head_snap_slot, head_snap);
          plan->head_snap_taken = head_snap.taken;
        }
      };
      typename Model::Outputs out;
      try {
        if (plan->attach_slot >= 0) {
          if (arena_.position(plan->attach_slot) != plan->attach_position)
            throw std::logic_error(
                "generation engine: the attach slot's position differs from the plan");
          arena_.attach(req, plan->attach_slot);
          try {
            const std::vector<int64_t> suffix(prompt.begin() + plan->attach_position, prompt.end());
            out = cached_model_prefill(model_, req, suffix, *plan->boundaries, snap_ptr,
                                       plan->images, true);
          } catch (...) {
            model_->session_close(req);  // the attach opened it
            throw;
          }
        } else {
          out = cached_model_prefill(model_, req, prompt, *plan->boundaries, snap_ptr, plan->images,
                                     false);
        }
      } catch (...) {
        commit();  // a completed earlier snapshot still owns its pinned blocks
        throw;
      }
      commit();
      return out;
    });
  }
  void prefix_snapshot(int req, int slot, int64_t position) override {
    arena_.snapshot(req, slot, position);
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
  void reserve(int req, int64_t tokens) override {
    model_->session_reserve_blocks(req, tokens);
  }
  std::vector<int32_t> step(int req) override {
    int64_t& pending = pending_.at(static_cast<size_t>(req));
    if (pending < 0)
      throw std::logic_error("generation step on a slot without a pending "
                             "token");
    SlotState& s = state_.at(static_cast<size_t>(req));
    if constexpr (requires { model_->dflash2_enabled(); }) {
      if (model_->dflash2_enabled()) {
        // Keep a live speculation only while the slot stays plain greedy and
        // one full block of verify rows fits the context; anything else
        // runs the exact plain step (and retries later).
        const std::vector<int64_t> fed = dflash_fed(req);
        if (!fed.empty()) {
          auto& sp = spec_.at(static_cast<size_t>(req));
          const int T = static_cast<int>(fed.size());
          const auto out = model_->session_verify(req, fed);
          const std::vector<int32_t> winners = rows_pick()(local_row_maxes(out, T));
          std::vector<int32_t> committed = sp->commit(fed, winners, 0);
          pending = sp->next();
          for (int32_t t : committed) s.context.push_back(t);
          s.context.push_back(static_cast<int32_t>(pending));
          return committed;
        }
      }
    }
    const int32_t next = decide(s, model_->session_step(req, pending));
    s.context.push_back(next);
    pending = next;
    return {next};
  }
  // The batched speculative pass (plan §7's concurrency gate): every
  // arriving slot's verify rows ride ONE physical target pass (slot-major,
  // <= max_decode_rows() rows). Drafts stay per-slot — the block is a few
  // % of the target's step. Per-slot decisions, rollbacks, publishes and
  // the eligibility rule are exactly the scalar step()'s; a single-slot
  // batch takes the scalar path, so a C1 transcript keeps the scalar
  // kernel sequence bit-for-bit.
  std::vector<std::vector<int32_t>> step_batch(const std::vector<int>& reqs) override {
    if constexpr (requires { model_->dflash2_enabled(); }) {
      const int graph_level = dflash_verify_graph();
      if (!model_->dflash2_enabled() || reqs.size() < (graph_level >= 2 ? size_t{1} : size_t{2}))
        return SchedulerEngine::step_batch(reqs);
      const bool ph = dflash_phases();
      const auto ns_now = [] {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
                    .count();
      };
      const long long ph0 = ph ? ns_now() : 0;
      std::vector<std::vector<int64_t>> feds(reqs.size());
      std::vector<char> is_spec(reqs.size(), 0);
      for (size_t i = 0; i < reqs.size(); ++i) {
        if (pending_.at(static_cast<size_t>(reqs[i])) < 0)
          throw std::logic_error("generation step on a slot without a pending "
                                 "token");
        feds[i] = dflash_fed(reqs[i]);
        if (!feds[i].empty()) {
          is_spec[i] = 1;
        } else {
          // Ineligible slot: feeds its pending token, decided through the
          // sampler closure like the scalar plain step.
          feds[i].push_back(pending_.at(static_cast<size_t>(reqs[i])));
        }
      }
      const long long ph1 = ph ? ns_now() : 0;
      std::vector<int> offs;
      // The verify, per DGPP_DFLASH2_VERIFY_GRAPH level: 0 the packed
      // eager batch (the opt-out baseline, no graph); 1 multi-slot batches
      // replay a captured static verify (the shipped default, the measured
      // best); 2 lone slots too; 3 the bisection control — the same static
      // padded staging executed eagerly. A lone slot is the scalar path at
      // every level < 2 (the line-289 gate), so a C1 transcript keeps the
      // scalar kernel sequence. Drafts, judge, rollback and redrafts are
      // unchanged around it; any capture breakage falls back to the eager
      // batch inside the model call. BATCH_EAGER forces the packed eager
      // batch at any level.
      const bool padded = !dflash_batch_eager() && graph_level == 3 &&
                          model_->dflash_graph_verify_available();
      const bool want_graph = !dflash_batch_eager() && !padded &&
                              graph_level >= 1 &&
                              (graph_level >= 2 || reqs.size() >= 2) &&
                              model_->dflash_graph_verify_available();
      auto outs = want_graph
                      ? model_->session_verify_batch_graph(reqs, feds, &offs)
                      : padded ? model_->session_verify_batch_padded(reqs, feds, &offs)
                               : model_->session_verify_batch(reqs, feds, &offs);
      const long long ph2 = ph ? ns_now() : 0;
      std::vector<std::vector<int32_t>> out(reqs.size());
      // Batched redrafts (DGPP_DFLASH2_DRAFT_BATCH=1): one stacked block
      // forward for every spec slot instead of one per slot. Otherwise
      // each slot redrafts alone (the shipped behavior).
      std::vector<size_t> batch_idx;
      if (dflash_draft_batch()) {
        for (size_t i = 0; i < reqs.size(); ++i)
          if (is_spec[i]) batch_idx.push_back(i);
      }
      const bool use_batch = batch_idx.size() >= 2;
      for (size_t i = 0; i < reqs.size(); ++i) {
        SlotState& s = state_.at(static_cast<size_t>(reqs[i]));
        int64_t& pending = pending_.at(static_cast<size_t>(reqs[i]));
        if (is_spec[i]) {
          auto& sp = spec_.at(static_cast<size_t>(reqs[i]));
          const int T = static_cast<int>(feds[i].size());
          const std::vector<int32_t> winners = rows_pick()(local_row_maxes(outs[i], T));
          if (use_batch) {
            out[i] = sp->commit_verify(feds[i], winners, offs[i]);
          } else {
            out[i] = sp->commit(feds[i], winners, offs[i]);
          }
          pending = sp->next();
          for (int32_t t : out[i]) s.context.push_back(t);
          s.context.push_back(static_cast<int32_t>(pending));
        } else {
          const int32_t next = decide(s, outs[i]);
          s.context.push_back(next);
          pending = next;
          out[i] = {next};
        }
      }
      const long long ph3 = ph ? ns_now() : 0;
      if (use_batch) {
        std::vector<int> breqs;
        std::vector<int64_t> bonuses;
        breqs.reserve(batch_idx.size());
        bonuses.reserve(batch_idx.size());
        for (size_t i : batch_idx) {
          breqs.push_back(reqs[i]);
          bonuses.push_back(spec_.at(static_cast<size_t>(reqs[i]))->next());
        }
        std::vector<std::vector<int32_t>> bdrafts;
        model_->dflash2_draft_batch(breqs, bonuses, &bdrafts);
        for (size_t k = 0; k < batch_idx.size(); ++k)
          spec_.at(static_cast<size_t>(reqs[batch_idx[k]]))->set_drafts(std::move(bdrafts[k]));
      } else {
        for (size_t i : batch_idx)
          spec_.at(static_cast<size_t>(reqs[i]))->redraft();
      }
      if (ph) {
        const long long ph4 = ns_now();
        PhaseAcc& a = dfph_acc();
        a.fed_ns += ph1 - ph0;
        a.verify_ns += ph2 - ph1;
        a.commit_ns += ph3 - ph2;
        a.draft_ns += ph4 - ph3;
        a.total_ns += ph4 - ph0;
        ++a.steps;
        if (a.steps % 25 == 0) {
          const int n = a.steps;
          DGPP_LOG_INFO("dflash phases ({} steps): fed {:.1f} us, verify {:.2f} ms, "
                        "commit {:.1f} us, draft {:.2f} ms, total {:.2f} ms/pass; "
                        "slots this step {}",
                        n, a.fed_ns * 1e-3 / n, a.verify_ns * 1e-6 / n,
                        a.commit_ns * 1e-3 / n, a.draft_ns * 1e-6 / n,
                        a.total_ns * 1e-6 / n, static_cast<int>(reqs.size()));
        }
      }
      return out;
    } else {
      return SchedulerEngine::step_batch(reqs);
    }
  }
  // One physical pass advances every arriving spec slot; the row budget
  // (max_decode_rows verify rows, a full block per spec slot) bounds it.
  int decode_batch_capacity() const override {
    if constexpr (requires { model_->dflash2_enabled(); model_->max_decode_rows(); })
      if (model_->dflash2_enabled()) {
        const int cap = model_->max_decode_rows() / kSpecRows;
        return cap < 1 ? 1 : (cap > slots_ ? slots_ : cap);
      }
    return 1;
  }
  // A spec step writes up to a full block of positions (the scheduler's
  // reservation window grows accordingly); plain engines keep 1.
  int max_tokens_per_step() const override {
    if constexpr (requires { model_->dflash2_enabled(); })
      if (model_->dflash2_enabled()) return kSpecRows;
    return 1;
  }
  // The logit bias (OpenAI's logit_bias, 2026-09-06): a dense row per slot
  // over [0, this rank's slice end) — the sampler closure adds it after the
  // penalties; a biased slot decides through the closure whatever its
  // temperature (the biased argmax at 0). An id past this rank's slice is
  // another rank's to apply.
  bool supports_logit_bias() const override { return sample_ != nullptr; }
  void configure_logit_bias(int req,
                            const std::vector<sched::LogitBias>& bias) override {
    SlotState& s = state_.at(static_cast<size_t>(req));
    s.bias.clear();
    if (bias.empty()) return;
    if (!sample_)
      throw std::logic_error(
          "gen engine: no sampler — this engine cannot bias the pick");
    const size_t cover = static_cast<size_t>(model_->lm_vocab_begin()) +
                         static_cast<size_t>(model_->lm_vocab_count());
    s.bias.assign(cover, 0.0f);
    for (const sched::LogitBias& b : bias) {
      if (b.token < 0)
        throw std::invalid_argument("gen engine: logit_bias token below 0");
      if (static_cast<size_t>(b.token) < cover)
        s.bias[static_cast<size_t>(b.token)] = b.bias;
    }
  }
  void close(int req) override {
    pending_.at(static_cast<size_t>(req)) = -1;
    if constexpr (requires { model_->dflash2_enabled(); }) retire_spec(spec_.at(static_cast<size_t>(req)));
    // A reopened slot is greedy until the scheduler arms it again.
    state_.at(static_cast<size_t>(req)) = SlotState{};
    model_->session_close(req);
  }

  // The slot's RNG state (seed, counter) — the journal/audit view of how
  // many draws the request has consumed.
  const sample::Rng& rng(int req) const {
    return state_.at(static_cast<size_t>(req)).rng;
  }

 private:
  struct SlotState {
    sample::Params params = sample::greedy_params();
    sample::Rng rng;
    std::vector<int32_t> context;  // prompt + generated: the count table
    std::vector<sample::Result> logprobs;  // the last op's, when asked
    bool report_logprobs = false;
    std::unique_ptr<text::GrammarState> grammar;  // constrained decoding
    text::TokenMask mask;                          // the next position's
    std::vector<float> bias;  // the logit_bias row (empty: none)
  };

  // The prefill's slot-side work around the model call that produces the
  // last row's logits (a cold prefill, or an attach + resume): the context
  // for the penalties, the grammar's opening state, the decision.
  template <typename Run>
  int32_t open_slot(int req, const std::vector<int64_t>& prompt, Run&& run) {
    SlotState& s = state_.at(static_cast<size_t>(req));
    s.context.clear();
    s.context.reserve(prompt.size() + 64);
    for (int64_t id : prompt) s.context.push_back(static_cast<int32_t>(id));
    // The grammar's opening state: thinking iff the prompt ends in <think>.
    if (s.grammar) {
      const text::ChatMarkers& m = grammar_vocab_->markers();
      const bool opens = m.prompt_opens_thinking(prompt);
      s.grammar = std::make_unique<text::GrammarState>(
          grammar_vocab_, s.grammar->spec(), opens, m.prompt_leaves_thinking_to_model(prompt));
    }
    const int32_t token = decide(s, run());
    s.context.push_back(token);
    pending_.at(static_cast<size_t>(req)) = token;
    return token;
  }

  // The plain greedy pick serves temperature 0 with neither logprobs nor
  // penalties nor a grammar (zero cost, the exact op stream every gate
  // pins); anything else goes through the sampler closure, which handles
  // temperature 0 as the greedy decision under the raw distribution — the
  // masked argmax when a grammar constrains the position.
  int32_t decide(SlotState& s, const DecodeOutputs& out) {
    const bool penalized = s.params.repetition_penalty != 1.0f ||
                           s.params.frequency_penalty != 0.0f ||
                           s.params.presence_penalty != 0.0f;
    const text::TokenMask* mask = nullptr;
    if (s.grammar && s.grammar->active()) {
      s.grammar->mask(&s.mask);
      if (s.mask.constrained()) mask = &s.mask;
    }
    const float* bias = s.bias.empty() ? nullptr : s.bias.data();
    int32_t token = -1;
    if (s.params.temperature <= 0.0f && !s.report_logprobs && !penalized &&
        mask == nullptr && bias == nullptr) {
      token = pick_(out);
    } else {
      const sample::Result r =
          sample_(out, s.params, s.rng, s.context, mask, bias);
      if (s.report_logprobs) s.logprobs.push_back(r);
      token = r.token;
    }
    if (s.grammar) s.grammar->advance(token);
    return token;
  }

  Model* model_;
  int slots_;
  Pick pick_;
  Sample sample_;
  const text::GrammarVocab* grammar_vocab_ = nullptr;
  std::vector<int64_t> pending_;
  std::vector<SlotState> state_;
  PrefixArena<Model> arena_;  // the prefix cache's snapshot slots (M7)
  // DFlash2 drafter drivers, one per slot (only used when the model has a
  // drafter; eager greedy semantics, exact under greedy verify).
  std::vector<std::unique_ptr<DFlash2Speculator<Model>>> spec_;
  // The drafter's per-position acceptance counters, engine-wide: retired
  // drivers fold in here, live ones read on top (the MTP stats group).
  uint64_t spec_att_[8] = {}, spec_acc_[8] = {};

  // The eligibility rule of step()'s spec branch, shared with step_batch:
  // the slot's verify-fed rows when it rides speculation this step (plain
  // greedy, one full block fits the context; the speculator is created and
  // started as needed), empty when it runs the exact plain step instead
  // (retiring any live speculation, retried next step).
  std::vector<int64_t> dflash_fed(int req) {
    SlotState& s = state_.at(static_cast<size_t>(req));
    auto& sp = spec_.at(static_cast<size_t>(req));
    const bool plain_greedy = s.params.temperature <= 0.0f && !s.report_logprobs &&
                              s.bias.empty() && !s.grammar &&
                              s.params.repetition_penalty == 1.0f &&
                              s.params.frequency_penalty == 0.0f &&
                              s.params.presence_penalty == 0.0f;
    const bool fits = model_->session_position(req) + 1 + model_->dflash2_drafts() <=
                      model_->max_context();
    if (sp && (!plain_greedy || !fits)) retire_spec(sp);
    if (!(plain_greedy && fits)) {
      // DGPP_DFLASH2_TRACE=1 logs why a slot runs plain (one line per
      // slot-step): the engagement split that sizes every spec-side
      // investment. Default off: zero behavior change.
      if (dflash_trace()) {
        logf(LogLevel::Info, "dflash: slot {} runs plain (temp={} logprobs={} bias={} grammar={} reppen={} freqpen={} prespen={})",
             req, s.params.temperature, s.report_logprobs ? 1 : 0, s.bias.empty() ? 0 : 1,
             s.grammar ? 1 : 0, s.params.repetition_penalty, s.params.frequency_penalty,
             s.params.presence_penalty);
      }
      return {};
    }
    if (!sp) {
      sp = std::make_unique<DFlash2Speculator<Model>>(*model_, req, rows_pick());
      // Experimental verify-depth cap (long-context throughput work):
      // DGPP_DFLASH2_DEPTH=k verifies only the first k drafts per step
      // (exact transcripts — unverified drafts re-draft next step).
      // Unset (the default) verifies the whole block.
      const int depth_cap = dflash_depth_cap();
      if (depth_cap >= 0) {
        const int D = model_->dflash2_drafts();
        int k = depth_cap < D ? depth_cap : D;
        sp->set_depth_policy([k](const std::vector<int32_t>& d) {
          const int n = static_cast<int>(d.size());
          return k < n ? k : n;
        });
      }
      sp->start(static_cast<int32_t>(pending_.at(static_cast<size_t>(req))));
    }
    return sp->fed_rows();
  }

  // Cached env gates for the dflash throughput work (both default off,
  // so the shipped path is untouched unless the operator opts in).
  static int dflash_depth_cap() {
    static const int v = [] {
      const char* e = std::getenv("DGPP_DFLASH2_DEPTH");
      if (!e || !*e) return -1;
      const int k = std::atoi(e);
      return k < 0 ? -1 : k;
    }();
    return v;
  }
  static bool dflash_draft_batch() {
    // Batched redrafts (one stacked forward per step): default ON —
    // validated +5–9% at c4 with no errors, C1-identical by construction
    // (a single spec slot keeps the scalar redraft). DGPP_DFLASH2_DRAFT_BATCH=0
    // opts out to the per-slot redrafts.
    static const bool v = [] {
      const char* e = std::getenv("DGPP_DFLASH2_DRAFT_BATCH");
      return !(e && *e == '0');
    }();
    return v;
  }
  static bool dflash_trace() {
    static const bool v = [] {
      const char* e = std::getenv("DGPP_DFLASH2_TRACE");
      return e && *e && *e != '0';
    }();
    return v;
  }
  static int dflash_verify_graph() {
    // 0: the packed eager batch — no graph (the opt-out baseline). 1:
    // multi-slot batches replay the captured static verify (the shipped
    // default — the measured best: ~5–8% over the eager batch at
    // short-context c4/c8, a tie at 8K). 2: lone slots too (their own
    // 8-row capture — the capture-mechanism control: no padding involved
    // at all). 3: the bisection control — the same static padded staging
    // executed eagerly.
    static const int v = [] {
      const char* e = std::getenv("DGPP_DFLASH2_VERIFY_GRAPH");
      if (!e || !*e) return 1;   // shipped default: the multi-slot graph
      if (*e == '0') return 0;   // opt out: the packed eager batch
      return std::atoi(e);
    }();
    return v;
  }
  static bool dflash_batch_eager() {
    // Force the eager batch inside the graph method (the 3-way bisection:
    // scalar vs eager-batch vs captured-batch on identical staging).
    static const bool v = [] {
      const char* e = std::getenv("DGPP_DFLASH2_BATCH_EAGER");
      return e && *e && *e != '0';
    }();
    return v;
  }
  // DGPP_DFLASH2_PHASES=1: accumulate step_batch's phase wall times and
  // log a rolling average every 25 steps (the 8K profiling split:
  // fed / verify / commit / draft per pass).
  static bool dflash_phases() {
    static const bool v = [] {
      const char* e = std::getenv("DGPP_DFLASH2_PHASES");
      return e && *e && *e != '0';
    }();
    return v;
  }
  struct PhaseAcc {
    long long verify_ns = 0, draft_ns = 0, commit_ns = 0, fed_ns = 0, total_ns = 0;
    int steps = 0;
  };
  static PhaseAcc& dfph_acc() {
    static PhaseAcc a;
    return a;
  }

  template <class S>
  void retire_spec(S& sp) {
    if constexpr (requires { sp->attempts(0); }) {
      if (sp) {
        for (int p = 0; p < 8; ++p) {
          spec_att_[p] += sp->attempts(p);
          spec_acc_[p] += sp->accepts(p);
        }
        sp.reset();
      }
    }
  }

  sched::SchedulerEngine::MtpAcceptance mtp_acceptance() const override {
    sched::SchedulerEngine::MtpAcceptance a;
    if constexpr (requires { model_->dflash2_drafts(); }) {
      if (model_->dflash2_enabled()) a.depth = model_->dflash2_drafts();
      for (int p = 0; p < 8; ++p) {
        a.attempts[p] = spec_att_[p];
        a.accepts[p] = spec_acc_[p];
        for (const auto& sp : spec_)
          if (sp) {
            a.attempts[p] += sp->attempts(p);
            a.accepts[p] += sp->accepts(p);
          }
      }
    }
    return a;
  }

  sched::SchedulerEngine::MtpAcceptance mtp_acceptance(int req) const override {
    sched::SchedulerEngine::MtpAcceptance a;
    if constexpr (requires { model_->dflash2_drafts(); }) {
      const auto& sp = spec_.at(static_cast<size_t>(req));
      if (sp) {
        a.depth = model_->dflash2_drafts();
        for (int p = 0; p < 8; ++p) {
          a.attempts[p] = sp->attempts(p);
          a.accepts[p] = sp->accepts(p);
        }
      }
    }
    (void)req;
    return a;
  }


  // The world-1 rows pick: each row's local max is already the winner.
  static SpecPickRows rows_pick() {
    return [](const std::vector<sample::Candidate>& locals) {
      std::vector<int32_t> w;
      w.reserve(locals.size());
      for (const auto& c : locals) w.push_back(c.id);
      return w;
    };
  }
};

// The world-1 pick: full-vocab argmax over the fp32 logits row. The closure
// keeps the decode path allocation-free.
inline DecodePick make_w1_pick(int64_t vocab) {
  return [vocab](const DecodeOutputs& out) -> int32_t {
    const int32_t t =
        sample::local_max(out.logits.data(),
                              static_cast<int>(out.lm_vocab_count),
                              /*vocab_begin=*/0)
            .id;
    if (t < 0 || t >= vocab)
      throw std::runtime_error("w1 pick out of range: " + std::to_string(t));
    return t;
  };
}

// The world-1 sampler: the complete distribution is on this host, so the
// decision is sample_full_logits at the one-slice layout — bitwise the
// sharded reference at world 1.
inline DecodeSample make_w1_sample(int64_t vocab) {
  const std::vector<sample::VocabSlice> layout =
      sample::vocab_layout(static_cast<int>(vocab), 1);
  return [vocab, layout](const DecodeOutputs& out,
                         const sample::Params& p, sample::Rng& rng,
                         const std::vector<int32_t>& context,
                         const text::TokenMask* mask,
                         const float* bias) -> sample::Result {
    if (out.lm_vocab_begin != 0 || out.lm_vocab_count != vocab)
      throw std::runtime_error("w1 sample: the head is not the full vocab");
    const uint32_t* words =
        mask != nullptr && mask->constrained() ? mask->words.data() : nullptr;
    if (p.temperature <= 0.0f) {
      // The greedy decision with logprobs, penalties or a mask: penalties,
      // the mask, the raw normalizer, the canonical argmax under it.
      std::vector<float> v(out.logits.begin(), out.logits.begin() + vocab);
      sample::apply_penalties(v.data(), static_cast<int>(vocab), 0, p,
                                  sample::count_context(context));
      sample::apply_bias(v.data(), static_cast<int>(vocab), 0, bias);
      sample::apply_mask(v.data(), static_cast<int>(vocab), 0, words,
                             static_cast<int>(vocab));
      const double lse =
          sample::sharded_scaled_logsumexp(v.data(), layout, 1.0f);
      const int n = std::max(1, p.logprobs);
      const std::vector<sample::Candidate> top =
          sample::local_topk(v.data(), static_cast<int>(vocab), 0, n);
      return sample::greedy_from_prefix(top, lse, p.logprobs);
    }
    sample::Result r = sample::sample_full_logits(
        out.logits.data(), static_cast<int>(vocab), layout, p, rng, context,
        words, bias);
    if (r.token < 0 || r.token >= vocab)
      throw std::runtime_error("w1 sample out of range: " +
                               std::to_string(r.token));
    return r;
  };
}

}  // namespace dgpp
