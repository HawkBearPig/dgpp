#pragma once
// Greedy speculative decode (DESIGN §9) over a model built
// with mtp = true.
//
// Per step the main stack verifies [next, draft] — the token it was going
// to consume anyway plus the MTP block's guess for the one after — in one
// T=2 call. Row 0's argmax is what a plain step would have produced for
// `next`; if it equals the draft, row 1 was a legitimate step too and its
// argmax is the next `next`: two tokens for one weight sweep. Otherwise
// row 1 is retracted (session_rollback) and row 0's argmax is the next
// `next`, exactly a plain step. Either way the block then drafts over the
// accepted rows. The transcript is bitwise the plain greedy transcript:
// verify rows are the T=1 rows (GlmDiagnosticModel::session_verify), and
// the pick is the same canonical argmax — MTP changes the cost, never the
// output.
//
// judge_verify is the pure core; GreedySpeculator is the eager driver the
// tests run; the graph-era loop in glm_gen_check replays graphs around the
// same judge.
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/decode_outputs.hpp"
#include "sample/sampler.hpp"
#include "text/tool_grammar.hpp"

namespace dgpp {

struct SpecVerdict {
  int accepted = 0;                  // rows [0, accepted) of the verify stand
  std::vector<int32_t> committed;    // the tokens those rows consumed (final)
  int32_t next = -1;                 // the token the main stack consumes next
  std::vector<int64_t> draft_rows;   // the block's next inputs: rows'
                                     // argmaxes, i.e. committed[1..] + next
};

// fed: the verify's tokens [next, draft_1, ...]; winners: each row's
// argmax. Row r stands if every row before it stood and row r-1 predicted
// fed[r] (row 0 always stands: it consumed a token already decided).
inline SpecVerdict judge_verify(const std::vector<int64_t>& fed,
                                const std::vector<int32_t>& winners) {
  if (fed.empty() || winners.size() != fed.size())
    throw std::invalid_argument("judge_verify: fed/winners shape");
  SpecVerdict v;
  int a = 1;
  while (a < static_cast<int>(fed.size()) &&
         winners[static_cast<size_t>(a - 1)] == fed[static_cast<size_t>(a)])
    ++a;
  v.accepted = a;
  v.committed.assign(fed.begin(), fed.begin() + a);
  v.next = winners[static_cast<size_t>(a - 1)];
  v.draft_rows.assign(winners.begin(), winners.begin() + a);
  return v;
}

// Per-row local argmax over a rank's [rows, count] logits slice.
inline std::vector<sample::Candidate> local_row_maxes(
    const DecodeOutputs& out, int rows) {
  std::vector<sample::Candidate> locals;
  locals.reserve(static_cast<size_t>(rows));
  for (int r = 0; r < rows; ++r)
    locals.push_back(sample::local_max(
        out.logits.data() + static_cast<size_t>(r) * out.lm_vocab_count,
        out.lm_vocab_count, out.lm_vocab_begin));
  return locals;
}

// The chained drafts (depth >= 2): drafts[c] for c >= 1 is the greedy pick
// of the block's chain row fed drafts[c - 1] (GlmDiagnosticModel::
// session_draft_chain); a row past the context repeats the previous draft.
template <class Model, typename Pick>
inline void chain_drafts(Model& model, int req, int depth,
                         std::vector<int32_t>* drafts, Pick pick) {
  int runs = 0;
  for (int c = 1; c < depth; ++c)
    if (model.session_draft_chain_fits(req, c - 1)) runs = c;
  for (int c = 1; c < depth; ++c) {
    if (c > runs) {
      (*drafts)[static_cast<size_t>(c)] = (*drafts)[static_cast<size_t>(c - 1)];
      continue;
    }
    const auto o = model.session_draft_chain(
        req, (*drafts)[static_cast<size_t>(c - 1)], c - 1, c == 1, c == runs);
    (*drafts)[static_cast<size_t>(c)] = pick(o);
  }
}

// The eager driver. `pick_rows` turns every rank's per-row local maxes
// into the per-row global winners (bus_greedy_pick_rows on the fabric, the
// identity argmax at world 1).
template <class Model>
class GreedySpeculator {
 public:
  using PickRows = SpecPickRows;

  // `depth`: drafts per step (1..kSpecRows-1); past the first, the block's
  // chained rows (session_draft_chain) propose the tokens after it.
  GreedySpeculator(Model& model, int req, PickRows pick_rows,
                   int depth = 1)
      : model_(model), req_(req), pick_rows_(std::move(pick_rows)),
        depth_(depth) {
    if (!model_.mtp_enabled())
      throw std::invalid_argument("GreedySpeculator: the model has no MTP");
    if (depth_ < 1 || depth_ >= kSpecRows)
      throw std::invalid_argument("GreedySpeculator: depth");
  }

  // After session_prefill: `first` is the pick off the prefill logits.
  // Drafts the first proposal (the block's row P-1).
  void start(int32_t first) {
    next_ = first;
    drafts_ = draft_after({first});
  }
  // Test interface: the block still drafts its first row (its counter must
  // move), but the proposal fed to the first verify is `forced_draft` —
  // feeding the true next token makes the first step an accept-all.
  void start(int32_t first, int32_t forced_draft) {
    start(first);
    drafts_[0] = forced_draft;
  }

  // Confidence-scheduled verify depth (engine/verify_schedule.hpp; DSpark). The
  // block always drafts its full width, but only the first `policy(drafts_)`
  // drafts are fed to the verify — the rest are recomputed next step. This
  // changes throughput only: a draft is committed iff it equals the target's
  // argmax at that position, and any draft not verified is decoded plainly
  // next step, so the committed transcript is the plain greedy one at every
  // per-step depth (the exactness the graph-engine port relies on). null
  // (the default) verifies the whole block, unchanged. The policy returns
  // the number of drafts to verify; it is clamped to [0, depth].
  void set_depth_policy(std::function<int(const std::vector<int32_t>&)> p) {
    depth_policy_ = std::move(p);
  }
  int last_verify_depth() const { return last_verify_depth_; }

  // One speculative step. Returns the tokens that became final this step
  // (1 to 1 + depth); next() is then the following token, already decided
  // but not yet consumed by the main stack.
  std::vector<int32_t> step() {
    int k = depth_;
    if (depth_policy_) {
      k = depth_policy_(drafts_);
      if (k < 0) k = 0;
      if (k > depth_) k = depth_;
    }
    last_verify_depth_ = k;
    std::vector<int64_t> fed{next_};
    for (int i = 0; i < k; ++i) fed.push_back(drafts_[static_cast<size_t>(i)]);
    const int T = static_cast<int>(fed.size());
    const auto out = model_.session_verify(req_, fed);
    const std::vector<int32_t> winners = pick_rows_(local_row_maxes(out, T));
    const SpecVerdict v = judge_verify(fed, winners);
    model_.session_rollback(req_, v.accepted);  // every verify: a replay family records the accepted count
    ++steps_;
    accepted_drafts_ += v.accepted - 1;
    next_ = v.next;
    drafts_ = draft_after(v.draft_rows);
    return v.committed;
  }

  int32_t next() const { return next_; }
  // The block's current proposal for the token after next().
  int32_t draft() const { return drafts_[0]; }
  // Every proposal: the tokens after next(), in order.
  const std::vector<int32_t>& drafts() const { return drafts_; }
  int steps() const { return steps_; }
  int accepted_drafts() const { return accepted_drafts_; }

 private:
  std::vector<int32_t> draft_after(const std::vector<int64_t>& rows) {
    std::vector<int32_t> drafts(static_cast<size_t>(depth_), -1);
    const auto d = model_.session_draft(req_, rows);
    drafts[0] = pick_rows_(local_row_maxes(d, 1))[0];
    chain_drafts(model_, req_, depth_, &drafts, [&](const DecodeOutputs& o) {
      return pick_rows_(local_row_maxes(o, 1))[0];
    });
    return drafts;
  }

  Model& model_;
  int req_ = 0;
  PickRows pick_rows_;
  int depth_ = 1;
  std::function<int(const std::vector<int32_t>&)> depth_policy_;
  int last_verify_depth_ = 0;
  int32_t next_ = -1;
  std::vector<int32_t> drafts_;
  int steps_ = 0;
  int accepted_drafts_ = 0;
};

// ---------------------------------------------------------------------------
// Exact speculative SAMPLING with a deterministic draft (DESIGN §9/§10, M6
// 6b): the eager driver, and the oracle the on-device T=2 verdict is held
// to. Per step the main stack verifies [next, draft]; row 0's final
// distribution P0 (the plain sampler's, penalties over the context that
// includes `next`) accepts the draft with probability P0(draft) or yields
// the residual sample (sample::spec_select_from_sorted, over the bus
// bus_spec_accept); when the draft stands, row 1 is sampled as any step's
// token would be (context + draft). Two draws per step. The draft itself
// stays the block's argmax — deterministic, so the accept test is the whole
// correction. The marginal distribution of every committed token is the
// plain sampler's, so a transcript is a legitimate sample; only the cost
// changes with the acceptance rate. The context (the penalties' count
// table) is prompt + committed + next, and the draft joins it only when it
// stands — the device count table follows the same rule.
// ---------------------------------------------------------------------------
template <class Model>
class SampledSpeculator {
 public:
  using PickRows = SpecPickRows;
  using Row0 = SpecRow0;
  // Row1 is the engines' Sample closure; the eager speculator has no
  // grammar and no logit bias and passes null for both.
  using Row1 = DecodeSample;

  SampledSpeculator(Model& model, int req, PickRows draft_pick,
                    Row0 row0, Row1 row1, const sample::Params& params,
                    sample::Rng rng, const std::vector<int64_t>& prompt,
                    int depth = 1)
      : model_(model),
        req_(req),
        draft_pick_(std::move(draft_pick)),
        row0_(std::move(row0)),
        row1_(std::move(row1)),
        params_(params),
        rng_(rng),
        depth_(depth) {
    if (!model_.mtp_enabled())
      throw std::invalid_argument("SampledSpeculator: the model has no MTP");
    if (depth_ < 1 || depth_ >= kSpecRows)
      throw std::invalid_argument("SampledSpeculator: depth");
    if (!(params_.temperature > 0.0f))
      throw std::invalid_argument("SampledSpeculator: temperature must be > 0 "
                                  "(the greedy driver is GreedySpeculator)");
    context_.assign(prompt.begin(), prompt.end());
  }

  // After session_prefill: `first` is the sampled pick off the prefill
  // logits (already drawn with this speculator's RNG by the caller, who
  // passes the advanced state in). Drafts the first proposal.
  void start(int32_t first) {
    next_ = first;
    context_.push_back(first);
    drafts_ = draft_after({first});
  }

  // One speculative step: the tokens that became final (1 to 1 + depth);
  // next() is then the following token, decided but not yet consumed. Row
  // t < T-1 tests the draft fed to row t+1 (a stand moves on, a reject ends
  // the step on the residual); the last row reached is sampled plainly —
  // the device verdict's chain, draw for draw.
  std::vector<int32_t> step() {
    std::vector<int64_t> fed{next_};
    for (const int32_t d : drafts_) fed.push_back(d);
    const int T = static_cast<int>(fed.size());
    const auto out = model_.session_verify(req_, fed);
    std::vector<int32_t> winners;
    int32_t next_new = -1;
    for (int t = 0; t < T; ++t) {
      if (t + 1 < T) {
        const int32_t draft = drafts_[static_cast<size_t>(t)];
        const sample::SpecPrefixDecision d =
            row0_(row_view(out, t), draft, params_, rng_, context_);
        if (!d.resolved)
          throw std::logic_error("SampledSpeculator: a verify row must resolve");
        if (!d.accepted) {
          winners.push_back(d.result.token);
          next_new = d.result.token;
          break;
        }
        context_.push_back(draft);
        winners.push_back(draft);
      } else {
        const sample::Result r =
            row1_(row_view(out, t), params_, rng_, context_, nullptr, nullptr);
        winners.push_back(r.token);
        next_new = r.token;
      }
    }
    const int accepted = static_cast<int>(winners.size());
    model_.session_rollback(req_, accepted);  // every verify: a replay family records the accepted count
    std::vector<int32_t> committed(fed.begin(), fed.begin() + accepted);
    ++steps_;
    accepted_drafts_ += accepted - 1;
    context_.push_back(next_new);
    next_ = next_new;
    drafts_ = draft_after(std::vector<int64_t>(winners.begin(), winners.end()));
    return committed;
  }

  int32_t next() const { return next_; }
  int32_t draft() const { return drafts_[0]; }
  const std::vector<int32_t>& drafts() const { return drafts_; }
  int steps() const { return steps_; }
  int accepted_drafts() const { return accepted_drafts_; }
  const sample::Rng& rng() const { return rng_; }
  const std::vector<int32_t>& context() const { return context_; }

 private:
  static DecodeOutputs row_view(const DecodeOutputs& out, int row) {
    DecodeOutputs v;
    const size_t count = static_cast<size_t>(out.lm_vocab_count);
    v.logits.assign(out.logits.begin() + static_cast<long>(row * count),
                    out.logits.begin() + static_cast<long>((row + 1) * count));
    v.lm_vocab_begin = out.lm_vocab_begin;
    v.lm_vocab_count = out.lm_vocab_count;
    return v;
  }
  std::vector<int32_t> draft_after(const std::vector<int64_t>& rows) {
    std::vector<int32_t> drafts(static_cast<size_t>(depth_), -1);
    const auto d = model_.session_draft(req_, rows);
    drafts[0] = draft_pick_(local_row_maxes(d, 1))[0];
    chain_drafts(model_, req_, depth_, &drafts, [&](const DecodeOutputs& o) {
      return draft_pick_(local_row_maxes(o, 1))[0];
    });
    return drafts;
  }

  Model& model_;
  int req_ = 0;
  PickRows draft_pick_;
  Row0 row0_;
  Row1 row1_;
  sample::Params params_;
  sample::Rng rng_;
  int depth_ = 1;
  std::vector<int32_t> context_;
  int32_t next_ = -1;
  std::vector<int32_t> drafts_;
  int steps_ = 0;
  int accepted_drafts_ = 0;
};

// The DFlash2 drafter's eager greedy driver (models/qwen/dflash2.hpp):
// the same exact verify/judge/rollback as GreedySpeculator — the fed rows
// are the pending token plus the drafter's block proposals, the accepted
// prefix commits, the rest rolls back — but the proposal comes from one
// block forward + selector walk (model.dflash2_draft) instead of the MTP
// block's chained rows. When the drafter cannot propose (pool exhausted,
// context bound) it proposes nothing: the step degrades to the plain T=1
// decode and the draft is retried next step.
template <class Model>
class DFlash2Speculator {
 public:
  using PickRows = SpecPickRows;

  DFlash2Speculator(Model& model, int req, PickRows pick_rows)
      : model_(model), req_(req), pick_rows_(std::move(pick_rows)) {
    if (!model_.dflash2_enabled())
      throw std::invalid_argument("DFlash2Speculator: the model has no DFlash2 drafter");
  }

  // After session_prefill: `first` is the pick off the prefill logits; it
  // anchors the first block.
  void start(int32_t first) {
    next_ = first;
    drafts_.clear();
    model_.dflash2_draft(req_, next_, &drafts_);
  }

  std::vector<int32_t> step() {
    const std::vector<int64_t> fed = fed_rows();
    const int T = static_cast<int>(fed.size());
    const auto out = model_.session_verify(req_, fed);
    const std::vector<int32_t> winners = pick_rows_(local_row_maxes(out, T));
    return commit(fed, winners, 0);
  }

  // Confidence-scheduled verify depth (mirrors GreedySpeculator): the
  // block always drafts its full width, but only the first
  // `policy(drafts_)` drafts are fed to the verify — the rest are
  // recomputed next step. Throughput only: a draft commits iff it equals
  // the target's argmax at that position, and an unverified draft is
  // decoded plainly next step, so the transcript is exact at every
  // per-step depth. null (the default) verifies the whole block. The
  // policy returns the number of drafts to verify, clamped to [0, depth].
  // (Experimental lever for long-context work, where each tail row scores
  // the full KV for a shrinking acceptance: DGPP_DFLASH2_DEPTH caps it —
  // see EagerEngine's dflash wiring.)
  void set_depth_policy(std::function<int(const std::vector<int32_t>&)> p) {
    depth_policy_ = std::move(p);
  }
  int last_verify_depth() const { return last_verify_depth_; }

  // The verify's fed tokens for the pending step: [next_, drafts...]
  // (depth-capped when a policy is set).
  std::vector<int64_t> fed_rows() {
    int k = static_cast<int>(drafts_.size());
    if (depth_policy_) {
      k = depth_policy_(drafts_);
      if (k < 0) k = 0;
      if (k > static_cast<int>(drafts_.size())) k = static_cast<int>(drafts_.size());
    }
    last_verify_depth_ = k;
    std::vector<int64_t> fed{next_};
    for (int i = 0; i < k; ++i) fed.push_back(drafts_[static_cast<size_t>(i)]);
    return fed;
  }

  // The batch driver's half: this slot's rows were verified inside a
  // shared pass (its snapshot rows start at `snapshot_base`). Judges the
  // verdict, rolls back, recounts, and re-drafts — byte-identical to the
  // scalar step()'s tail.
  std::vector<int32_t> commit(const std::vector<int64_t>& fed,
                              const std::vector<int32_t>& winners,
                              int snapshot_base) {
    std::vector<int32_t> committed = commit_verify(fed, winners, snapshot_base);
    redraft();
    return committed;
  }

  // The batch driver's halves: judge/rollback/recount without redrafting
  // (the engine batches redrafts across slots), then the redraft itself.
  std::vector<int32_t> commit_verify(const std::vector<int64_t>& fed,
                                      const std::vector<int32_t>& winners,
                                      int snapshot_base) {
    const int T = static_cast<int>(fed.size());
    const SpecVerdict v = judge_verify(fed, winners);
    if (T > 1) model_.session_rollback(req_, v.accepted, T, snapshot_base);
    ++steps_;
    accepted_drafts_ += v.accepted - 1;
    // The scheduler's MTP group, per draft position (sched MtpAcceptance):
    // only verified positions count (a depth-capped fed leaves the tail
    // for the next step).
    for (int p = 0; p < T - 1; ++p) {
      ++attempts_[p & 7];
      if (p < v.accepted - 1) ++accepts_[p & 7];
    }
    next_ = v.next;
    drafts_.clear();
    return v.committed;
  }
  void redraft() { model_.dflash2_draft(req_, next_, &drafts_); }
  void set_drafts(std::vector<int32_t> d) { drafts_ = std::move(d); }

  int32_t next() const { return next_; }
  const std::vector<int32_t>& drafts() const { return drafts_; }
  int steps() const { return steps_; }
  int accepted_drafts() const { return accepted_drafts_; }
  uint64_t attempts(int p) const { return attempts_[p]; }
  uint64_t accepts(int p) const { return accepts_[p]; }

 private:
  Model& model_;
  int req_ = 0;
  PickRows pick_rows_;
  int32_t next_ = -1;
  std::vector<int32_t> drafts_;
  int steps_ = 0;
  int accepted_drafts_ = 0;
  uint64_t attempts_[8] = {};
  uint64_t accepts_[8] = {};
  std::function<int(const std::vector<int32_t>&)> depth_policy_;
  int last_verify_depth_ = 0;
};

}  // namespace dgpp
