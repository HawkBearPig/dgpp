#pragma once
// Confidence-scheduled verification depth (docs/deepseek_v41_flash_plan.md
// §D8a; DeepSeek-V4.1-Flash tech report §2.4.3, the DSpark confidence head).
// Engine-level and model-agnostic: any draft that emits a per-position
// acceptance logit can drive it (Model::kVerifyConfidence in the graph
// engine); DSpark is the first.
//
// The draft always computes the whole block of `block` positions in one pass
// (that pass is what produces the confidence, so it is sunk once we are here).
// The confidence head emits a per-position acceptance LOGIT c_i, so
// sigmoid(c_i) = P(draft i accepts | the prefix before it survived), and
// S_i = prod_{j<i} sigmoid(c_j) is the probability the first i drafts all
// survive. S_i is monotone non-increasing in i.
//
// A verify pass over k of the block's drafts runs 1+k rows (the fed token plus
// k drafts) and commits the greedy-matching prefix. That prefix is EXACTLY the
// one a full-block verify would commit, so k changes throughput only, never the
// output.
//
// Objective. We maximize aggregate decode throughput, sum(committed)/sum(time)
// over the run, NOT the per-step ratio committed/time (maximizing a sum of
// ratios is the wrong surrogate for a ratio of sums, and the two disagree
// exactly when confidence varies across steps -- chat vs counting, our case).
// The Dinkelbach optimum of a ratio of sums is a per-step LINEAR tradeoff at a
// single global multiplier lambda* (the achieved throughput, tokens/ms): each
// step maximizes E[committed|k] - lambda*.time(k). With
//   E[committed|k] = 1 + sum_{i=1..k} S_i,   time(k) = base + row.(1+k),
// the gain of extending k->k+1 is S_{k+1} - lambda*.row, so we verify draft i
// while S_i > lambda*.row and stop at the first that falls below (S monotone).
// The fixed base cost drops out of the depth decision; it only sets lambda*.
//
// lambda is supplied by the caller as the value of decode time -- the engine's
// realized throughput, tracked as an EWMA of committed/time, which converges to
// the self-consistent lambda*. Before it warms up, the reservation rate below
// (a plain single-token step's throughput) is a safe floor: speculation is
// taken only when a row beats just decoding one more token the plain way.
//
// Confidence is replicated across ranks and lambda is a scalar the caller keeps
// identical on every rank, so all ranks derive the same k.
#include <cmath>
#include <vector>

namespace dgpp {

// Throughput of a plain single-token decode step: 1 token over the fixed base
// plus one row. A safe lower bound for lambda (tokens/ms) before an EWMA of
// realized throughput is available.
inline float verify_reservation_lambda(float base_ms, float row_ms) {
  const float t = base_ms + row_ms;
  return t > 0.f ? 1.f / t : 0.f;
}

// conf_logit: [block] the block's per-position confidence logits (replicated
// across ranks). row_ms: cost of one verify row. lambda_tok_per_ms: the value
// of decode time (realized throughput EWMA, floored at the reservation rate).
// Returns k in [0, block]: the number of leading drafts to verify.
// accept_scale: the slot's acceptance per position as a fraction of the
// head's (1 for a greedy slot, whose draft is accepted exactly when it is
// the target's argmax — what the head predicts; under 1 for a sampled slot
// with argmax drafts, accepted with the target's PROBABILITY of the draft.
// Traced on DeepSeek-V4-Flash, 2026-10-02, the whole block verified every
// pass: the head over-states a sampled slot's FIRST draft — 0.75 predicted
// against 0.67 accepted on prose — and is near calibrated behind it; over
// 3,140 prose passes and 819 mixed-class ones the rule at 0.93 commits
// within 0.2 % of the best first-position-only correction and 1.5-4 %
// ahead of 0.8, which stops the deeper drafts short.)
inline int scheduled_verify_depth(const float* conf_logit, int block, float row_ms,
                              float lambda_tok_per_ms, double accept_scale = 1.0) {
  if (block <= 0) return 0;
  const double threshold = static_cast<double>(lambda_tok_per_ms) * row_ms;
  double survival = 1.0;  // S_i, the running prefix-survival product
  int k = 0;
  for (int i = 1; i <= block; ++i) {
    const double p = accept_scale / (1.0 + std::exp(-static_cast<double>(conf_logit[i - 1])));
    survival *= p;  // S_i includes sigmoid(c_{i-1})
    if (survival > threshold)
      k = i;  // this row's expected tokens still beat the value of its time
    else
      break;  // S monotone non-increasing: no later row can qualify
  }
  return k;
}

// The batched replay's one depth for `slots` live slots (conf[s]: slot s's
// block of logits). Extending the batch by one draft position costs one
// verify row PER SLOT (slots * row_ms) and yields the sum over the slots
// of their prefix survivals at that position, so the Dinkelbach gain is
// sum_s S_i(s) - slots * lambda * row: verify position i while the MEAN
// survival over the slots exceeds lambda * row (the scalar rule with the
// mean in place of one slot's S_i; each mean is non-increasing in i, so
// the kept positions are a prefix). The deepest slot's own depth (the
// earlier rule) verifies rows for every slot that a confident one alone
// justifies; the mean rule stops where the batch as a whole stops paying.
// Every slot commits its greedy prefix at any depth, so this is throughput
// only, never output. Deterministic across ranks like the scalar rule.
// The adaptive lambda (D8a): the Dinkelbach multiplier is the achieved
// aggregate throughput, which grows with concurrency (0.048 tok/ms at one
// stream, 0.087 at six on the 2026-09-14 fabric), so one configured
// constant serves one concurrency. Tracked as an EWMA of each MTP step's
// committed tokens over its MODELED time base + rows.row (never wall
// clock: the ranks' clocks differ and would derive different depths) —
// every input is replicated (the verdict's committed counts, the step's
// rows, the configured constants) and the arithmetic is one fixed
// sequence of doubles, so every rank holds the same lambda. The caller
// floors it at the configured lambda (a plain step's reservation rate at
// the least).
inline double verify_lambda_update(double lambda, int committed, int rows, float base_ms,
                                   float row_ms, double alpha) {
  const double t = static_cast<double>(base_ms) + static_cast<double>(rows) * row_ms;
  if (!(t > 0.0) || committed < 0) return lambda;
  const double sample = static_cast<double>(committed) / t;
  return lambda + alpha * (sample - lambda);
}

// accept_scale: per slot (null: 1 for every slot), as the scalar rule's.
inline int scheduled_verify_depth_batch(const float* const* conf_logit, int slots, int block,
                                        float row_ms, float lambda_tok_per_ms,
                                        const double* accept_scale = nullptr) {
  if (block <= 0 || slots <= 0) return 0;
  const double threshold = static_cast<double>(lambda_tok_per_ms) * row_ms;
  std::vector<double> survival(static_cast<size_t>(slots), 1.0);
  int k = 0;
  for (int i = 1; i <= block; ++i) {
    double sum = 0.0;
    for (int s = 0; s < slots; ++s) {
      const float c = conf_logit[s][i - 1];
      survival[static_cast<size_t>(s)] *=
          (accept_scale != nullptr ? accept_scale[s] : 1.0) / (1.0 + std::exp(-static_cast<double>(c)));
      sum += survival[static_cast<size_t>(s)];
    }
    if (sum / slots > threshold)
      k = i;
    else
      break;
  }
  return k;
}

// The depth options a scheduled step chooses among, in the order the engine
// indexes them: [narrow] [refill] policy options.
//
// The policy's options are `options` depths spread evenly over [min_depth,
// depth], ascending, the last always the full block (one option is the full
// block alone). A policy depth rounds UP to the next of them, and each of
// them drafts the whole block.
//
// The narrow option (engine.mtp_schedule_max_live) verifies and drafts the
// minimum depth only, and every step past the live limit takes it. A slot
// leaves such a step holding min_depth drafts, so its next step may verify no
// more than those, and has to draft the whole block again before a deeper
// option can follow. The policy's own minimum-depth option does that when the
// policy has one. When it has none (a single option: the full block) the set
// carries the REFILL option: the minimum depth verified, the whole block
// drafted, taken only by a step whose slot holds fewer drafts than the
// policy's lowest option covers.
struct VerifyDepthOptions {
  std::vector<int> depths;  // the drafts each option verifies
  int first_block = 0;      // the first option that drafts the whole block (1 behind the narrow option)
  int first_policy = 0;     // the first option a policy depth rounds to (past the refill option)
  bool narrow() const { return first_block > 0; }
  bool refill() const { return first_policy > first_block; }
};

inline VerifyDepthOptions verify_depth_option_set(int depth, int min_depth, int options, bool narrow) {
  std::vector<int> policy;
  for (int i = 0; i < options; ++i) {
    const int d = options == 1
                      ? depth
                      : min_depth + static_cast<int>(std::lround(static_cast<double>(i) * (depth - min_depth) /
                                                                 static_cast<double>(options - 1)));
    if (policy.empty() || policy.back() != d) policy.push_back(d);
  }
  if (policy.empty() || policy.back() != depth) policy.push_back(depth);
  VerifyDepthOptions set;
  set.depths.reserve(policy.size() + 2);
  if (narrow) {
    set.depths.push_back(min_depth);
    set.first_block = 1;
    if (policy.front() > min_depth) set.depths.push_back(min_depth);  // the refill option
  }
  set.first_policy = static_cast<int>(set.depths.size());
  set.depths.insert(set.depths.end(), policy.begin(), policy.end());
  return set;
}

// The drafts option `option` of a `depth`-draft block leaves in the slot's
// feed for the next step: its own depth for the narrow option, the block for
// every other.
inline int verify_depth_option_drafts(const std::vector<int>& depths, int first_block, int option, int depth) {
  return option < first_block ? depths[static_cast<size_t>(option)] : depth;
}

// The option a step replays when the policy asks for `k` drafts and the slot
// holds `available` (what its last replay drafted): the policy's lowest
// option covering k, the full block when none does, and never an option
// deeper than the drafts available, in whose place the deepest block-drafting
// option within them stands. A draft row past the available ones is an older
// step's token and proposal, which no verify may test. The caller checks the
// result against `available`: it exceeds them only when no block-drafting
// option fits, which a set built by verify_depth_option_set never leaves.
inline int verify_depth_option(const std::vector<int>& depths, int first_block, int first_policy, int k,
                               int available) {
  const int count = static_cast<int>(depths.size());
  int option = count - 1;
  for (int i = first_policy; i < count; ++i)
    if (depths[static_cast<size_t>(i)] >= k) {
      option = i;
      break;
    }
  while (option > first_block && depths[static_cast<size_t>(option)] > available) --option;
  return option;
}

}  // namespace dgpp
