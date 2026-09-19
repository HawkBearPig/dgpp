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
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/dtypes.hpp"
#include "engine/decode_outputs.hpp"
#include "engine/image_prefill.hpp"
#include "engine/prefix_arena.hpp"
#include "sample/sampler.hpp"
#include "sched/scheduler.hpp"
#include "text/tool_grammar.hpp"

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
    return info;
  }
  int32_t prefill_cached(int req, const std::vector<int64_t>& prompt,
                         sched::SchedulerEngine::PrefixPrefill* plan) override {
    if (plan == nullptr || plan->boundaries == nullptr)
      throw std::invalid_argument("generation engine: prefill_cached without a plan");
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
          throw std::logic_error("generation engine: the attach slot's position differs from the plan");
        arena_.attach(req, plan->attach_slot);
        try {
          const std::vector<int64_t> suffix(
              prompt.begin() + plan->attach_position, prompt.end());
          out = cached_model_prefill(model_, req, suffix, *plan->boundaries,
                                   snap_ptr, plan->images, true);
        } catch (...) {
          model_->session_close(req);  // the attach opened it
          throw;
        }
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
    const int32_t next = decide(s, model_->session_step(req, pending));
    s.context.push_back(next);
    pending = next;
    return {next};
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
          grammar_vocab_, s.grammar->spec(), opens);
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
