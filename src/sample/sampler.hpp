#pragma once
// Exact sampling over the (possibly vocab-sharded) lm head — M6 d3,
// DESIGN §10. Three execution paths, one selection semantics:
//
//   greedy  : reduce the per-rank (value, token_id) maxima
//   top_k   : merge each rank's exact local top-k, filter/sample at rank 0
//   gather  : pull the full FP32 vocab to rank 0, sample centrally
//   prefix  : (M6 6b) merge each rank's exact local top-k AND fold each
//             slice's log-sum-exp; every rank decides identically whether
//             the request resolves inside the prefix, else falls back to
//             the gather — see sample_from_prefix() for its width-
//             independence contract and sample_reference_sharded() for
//             the reference it is measured against
//
// The distributed paths are bitwise-EQUAL to the centralized oracle by
// construction, not by tolerance: every path funnels into
// select_from_sorted() over a candidate list in the canonical order
// (logit descending, then token id ascending — a total order), and the
// merge provably produces the same set in the same order as sorting the
// full vocabulary (a token in the global top-k is, per slice, inside that
// slice's local top-k; the merged list is the k-way prefix of the same
// total order). The oracle "match" gate therefore asserts exact float
// equality, and any arithmetic that could drift (softmax denominators,
// top-p cumulative sums) is defined to run in that same listed order in
// the one shared implementation.
//
// Numerics (the contract, in HF warper order):
//   1. penalties on raw logits: repetition (sign-based divide/multiply),
//      then frequency (additive per count), then presence (additive once)
//   2. temperature scaling (T <= 0 -> greedy; sampling requires T > 0)
//   3. top-k truncation to the first k candidates (k >= vocab: no-op)
//   4. min-p: p_i = e_i/den_k >= min_p * p_max (den over the post-top-k
//      survivors; p_max = p_0, which every later filter keeps)
//   5. top-p: keep the smallest prefix with cumulative p >= top_p
//      (the crossing token stays in — the HF warper's rule)
//   6. final probabilities over the surviving set; one uniform draw
//      selects (fp64 walk over fp32 probabilities)
//
// The RNG is COUNTER-based, not stream-based: one draw per sampled token
// from (seed, counter). The counter is the resumption token DESIGN §10
// broadcasts with the chosen id — any rank can reconstruct any request's
// draw sequence from the seed plus the step count, which is what makes
// speculative verification (M8) and rank-consistency checks cheap.
// Greedy draws NOTHING: the counter advances only on stochastic draws.
//
// Host-side by design: the logits buffers are managed memory (the
// loader contract already grants CPU-readable resident buffers), so a
// rank samples its vocab slice without a staging copy.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/det_math.hpp"

// ARITHMETIC CONTRACT (M6 6b): every transcendental here is the
// deterministic host/device implementation (common/det_math.hpp), every
// multiply-add an explicit fma, and every reduction a fixed order — so the
// device verdict kernel reproduces this file bit for bit, and so do two
// fabric nodes with different libm builds. The slice normalizer's sum runs
// in fixed pairwise trees (kLseChunk-id chunks halved recursively, the
// chunk partials halved the same way), the order a parallel kernel keeps
// exactly with a short dependent chain.

namespace dgpp::sample {

// ---------------------------------------------------------------------------
// RNG
// ---------------------------------------------------------------------------

// splitmix64 finalizer: a PractRand-clean scramble, so a linear counter
// yields a well-mixed stream (this is the same construction splitmix64
// itself uses over sequential state).
inline uint64_t splitmix64_step(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

// The sampling RNG state. `counter` advances exactly once per stochastic
// draw; `seed` is the request's fixed-seed identity.
struct Rng {
  uint64_t seed = 0;
  uint64_t counter = 0;
};

// One uniform in [0, 1): the top 53 bits of the draw, as an fp64.
inline double uniform01(const Rng& rng) {
  const uint64_t draw = splitmix64_step(splitmix64_step(rng.counter) ^ rng.seed);
  return static_cast<double>(draw >> 11) * (1.0 / 9007199254740992.0);
}

// ---------------------------------------------------------------------------
// Request spec
// ---------------------------------------------------------------------------

struct Params {
  float temperature = 1.0f;        // <= 0 -> greedy
  int top_k = 0;                    // 0 = disabled (the gather path)
  float top_p = 1.0f;               // >= 1 = disabled
  float min_p = 0.0f;               // <= 0 = disabled
  float repetition_penalty = 1.0f;  // 1 = disabled (multiplicative)
  float frequency_penalty = 0.0f;   // additive per context count
  float presence_penalty = 0.0f;    // additive once for seen ids
  int logprobs = 0;                 // report top-N (id, logprob) pairs
};

struct Candidate {
  int32_t id = -1;
  float logit = 0.0f;
};

struct Result {
  int32_t token = -1;
  float logprob = 0.0f;  // log-softmax of the chosen token, final dist
  std::vector<std::pair<int32_t, float>> top_logprobs;  // if logprobs > 0
};

// Result of the bounded candidate-table path used by the distributed
// sampler. A fallback is not an approximation: it says that the prefix does
// not contain enough of the distribution to make the exact decision, so the
// caller must gather the full logits and run the same decision over the
// complete list (sample_reference_sharded()). The RNG is deliberately left
// untouched on fallback, so that slow path consumes the very same
// (seed, counter) draw and the request's outcome does not depend on the
// transported width.
struct PrefixDecision {
  bool resolved = false;
  Result result;
  double covered_mass = 0.0;  // pre-filter mass represented by the prefix
  double normalizer = 0.0;    // the fold log-sum-exp the decision used; the
                              // fallback re-runs the decision under it
};

// The greedy spec: no draw, no seed consumed, the exact argmax path.
inline Params greedy_params() {
  Params p;
  p.temperature = 0.0f;
  return p;
}

// The request-spec contract every path assumes. Throws invalid_argument so a
// bad spec fails identically on every rank (the scheduler validates at
// submit; the service turns the same failure into a 400).
inline void validate_params(const Params& p) {
  if (!std::isfinite(p.temperature) || p.temperature < 0.0f)
    throw std::invalid_argument("temperature must be finite and >= 0");
  if (!std::isfinite(p.top_p) || !(p.top_p > 0.0f && p.top_p <= 1.0f))
    throw std::invalid_argument("top_p must be in (0, 1]");
  if (!std::isfinite(p.min_p) || p.min_p < 0.0f || p.min_p > 1.0f)
    throw std::invalid_argument("min_p must be in [0, 1]");
  if (p.top_k < 0) throw std::invalid_argument("top_k must be >= 0");
  if (!std::isfinite(p.repetition_penalty) || !(p.repetition_penalty > 0.0f))
    throw std::invalid_argument("repetition_penalty must be > 0");
  if (!std::isfinite(p.frequency_penalty) || !std::isfinite(p.presence_penalty))
    throw std::invalid_argument("penalties must be finite");
  if (p.logprobs < 0) throw std::invalid_argument("logprobs must be >= 0");
}

// ---------------------------------------------------------------------------
// Penalties (warper stage 1)
// ---------------------------------------------------------------------------

// Negative-zero additive penalties retain the original signed-zero arithmetic.
// Decide from the effective request parameters, after defaults and overrides.
inline bool needs_context_penalties(const Params& p) {
  return p.repetition_penalty != 1.0f || p.frequency_penalty != 0.0f ||
         p.presence_penalty != 0.0f || std::signbit(p.frequency_penalty) ||
         std::signbit(p.presence_penalty);
}

// Token counts over the request context (prompt + generated so far).
// Same on every rank — the scheduler's rank-consistency guarantee.
inline std::unordered_map<int32_t, int32_t> count_context(
    const std::vector<int32_t>& ids) {
  std::unordered_map<int32_t, int32_t> counts;
  for (int32_t id : ids) ++counts[id];
  return counts;
}

// Applies the three penalties to one rank's vocab slice, in HF warper
// order, on the RAW (unscaled) logits. Ids outside
// [slice_begin, slice_begin + n) are ignored: a token lives on exactly
// one rank, so the union of slices applies each penalty exactly once.
inline void apply_penalties(float* logits, int n, int slice_begin,
                            const Params& p,
                            const std::unordered_map<int32_t, int32_t>& counts) {
  for (int i = 0; i < n; ++i) {
    const auto it = counts.find(slice_begin + i);
    if (it == counts.end()) continue;
    const int32_t count = it->second;
    float v = logits[i];
    if (p.repetition_penalty != 1.0f) {
      // HF semantics: divide positive logits, multiply negative ones.
      v = v > 0.0f ? v / p.repetition_penalty : v * p.repetition_penalty;
    }
    // One fused step (never a separate product and subtraction the two
    // sides could round differently).
    v = std::fma(-p.frequency_penalty, static_cast<float>(count), v);
    v -= p.presence_penalty;
    logits[i] = v;
  }
}

// ---------------------------------------------------------------------------
// Neutral preparation is an identity for non-NaN logits under round-to-nearest
// with gradual underflow. Exception flags/traps are not part of the sampling
// contract. Guard before counting: neutral requests need neither the context
// map nor a vocabulary scan. The reference samplers retain the full arithmetic.
inline void apply_context_penalties(float* logits, int n, int slice_begin, const Params& p,
                                    const std::vector<int32_t>& context_ids) {
  if (!needs_context_penalties(p)) return;
  apply_penalties(logits, n, slice_begin, p, count_context(context_ids));
}

// ---------------------------------------------------------------------------
// The token mask (M6 6g, constrained decoding)
// ---------------------------------------------------------------------------

// A masked id is an ABSENT candidate: its logit is -inf, it is never listed
// by local_topk/sort_slice, it contributes nothing to any normalizer, and
// the decision's "vocabulary size" is the count of PRESENT ids (the
// allowed set) — so a constrained row samples the distribution restricted
// to the mask and renormalized, exactly, through the unchanged selector.
// `mask_words` is a bitmask over [0, vocab): bit id set = allowed. Ids of
// the slice outside the mask become -inf in place.
inline void apply_mask(float* logits, int n, int slice_begin,
                       const uint32_t* mask_words, int vocab) {
  if (mask_words == nullptr) return;
  for (int i = 0; i < n; ++i) {
    const int id = slice_begin + i;
    const bool allowed =
        id < vocab && ((mask_words[id >> 5] >> (id & 31)) & 1u) != 0u;
    if (!allowed) logits[i] = -INFINITY;
  }
}

// The request's logit bias (OpenAI's logit_bias, 2026-09-06): `bias` is a
// dense table over [0, vocab) — null: none — added to the slice's logits
// AFTER the penalties and BEFORE the mask and the temperature. The device
// kernel adds the same float in the same place (bitwise the same sum).
inline void apply_bias(float* logits, int n, int slice_begin,
                       const float* bias) {
  if (bias == nullptr) return;
  for (int i = 0; i < n; ++i) logits[i] += bias[slice_begin + i];
}

inline bool present_logit(float v) { return v != -INFINITY; }

// The number of present (not -inf) logits — a constrained decision's
// vocabulary size.
inline int count_present(const float* logits, int n) {
  int present = 0;
  for (int i = 0; i < n; ++i) present += present_logit(logits[i]) ? 1 : 0;
  return present;
}

// ---------------------------------------------------------------------------
// Canonical order
// ---------------------------------------------------------------------------

// The total order shared by every path (and the parity proof's anchor).
inline bool candidate_before(const Candidate& a, const Candidate& b) {
  if (a.logit != b.logit) return a.logit > b.logit;
  return a.id < b.id;
}

// Sorts a full slice into the canonical order (the oracle's candidate
// list; also the shard test's reference).
inline std::vector<Candidate> sort_slice(const float* logits, int n,
                                        int slice_begin) {
  std::vector<Candidate> out;
  out.reserve(n);
  for (int i = 0; i < n; ++i)
    if (present_logit(logits[i])) out.push_back({slice_begin + i, logits[i]});
  std::sort(out.begin(), out.end(), candidate_before);
  return out;
}

// ---------------------------------------------------------------------------
// Selection core (shared by every path — the bitwise-parity anchor)
// ---------------------------------------------------------------------------

// Stages 2-5 of the pipeline over a canonical list: the temperature scale,
// the top-k/min-p/top-p survivors, the final distribution's fp32 exps and
// denominator, and its log-sum-exp. select_from_sorted draws over it; the
// speculative accept/residual (spec_select_from_sorted) evaluates one
// candidate's probability and walks the rest over the same state.
struct SelectorState {
  std::vector<float> scaled;  // logit / temperature, every candidate
  std::vector<float> exps;    // the survivors' exp(scaled - scaled[0])
  size_t final_count = 0;     // the final set is the first final_count
  float final_den = 0.0f;
  float lse = 0.0f;

  Result result_for(const std::vector<Candidate>& sorted, size_t chosen,
                    int logprobs) const {
    Result out;
    out.token = sorted[chosen].id;
    out.logprob = scaled[chosen] - lse;
    const int n = std::min<int>(logprobs, static_cast<int>(final_count));
    for (int i = 0; i < n; ++i)
      out.top_logprobs.emplace_back(sorted[static_cast<size_t>(i)].id,
                                    scaled[static_cast<size_t>(i)] - lse);
    return out;
  }
};

inline SelectorState selector_state(const std::vector<Candidate>& sorted,
                                    const Params& p) {
  SelectorState s;
  // Temperature scale (fp32 division — IEEE-deterministic everywhere).
  s.scaled.reserve(sorted.size());
  for (const Candidate& c : sorted) s.scaled.push_back(c.logit / p.temperature);
  std::vector<float>& scaled = s.scaled;

  // top-k truncation (k >= size: no-op).
  size_t kept = sorted.size();
  if (p.top_k > 0) kept = std::min<size_t>(kept, static_cast<size_t>(p.top_k));

  // min-p over the post-top-k distribution. exps are computed once —
  // the max is scaled[0] and every filter keeps index 0, so no recompute
  // can drift. Denominators always sum survivors in listed order.
  std::vector<float> exps(kept);
  for (size_t i = 0; i < kept; ++i) {
    exps[i] = detmath::exp_f(scaled[i] - scaled[0]);
  }
  float den = 0.0f;
  for (size_t i = 0; i < kept; ++i) den += exps[i];
  if (p.min_p > 0.0f) {
    const float threshold = p.min_p * (exps[0] / den);
    std::vector<float> survivors;
    for (size_t i = 0; i < kept; ++i) {
      if (exps[i] / den >= threshold) survivors.push_back(exps[i]);
    }
    // index 0 always survives (p_max is itself), so the set is non-empty;
    // the explicit compaction keeps the survivor order = listed order.
    exps = std::move(survivors);
  }

  // top-p: smallest prefix with cumulative probability >= top_p, the
  // crossing token included.
  den = 0.0f;
  for (float e : exps) den += e;
  size_t final_count = exps.size();
  if (p.top_p < 1.0f) {
    float cum = 0.0f;
    size_t cut = exps.size();  // default: keep all (sum may fall short)
    for (size_t i = 0; i < exps.size(); ++i) {
      cum += exps[i] / den;
      if (cum >= p.top_p) {
        cut = i + 1;
        break;
      }
    }
    final_count = cut;
  }

  // Final distribution over the surviving set.
  float final_den = 0.0f;
  for (size_t i = 0; i < final_count; ++i) final_den += exps[i];
  s.exps = std::move(exps);
  s.final_count = final_count;
  s.final_den = final_den;
  s.lse = scaled[0] + detmath::log_f(final_den);
  return s;
}

// `sorted` is the (already canonical-order) candidate list covering the
// global top-k-or-more set with penalties applied and is not modified.
// Selects per the documented pipeline; returns the chosen token with
// its logprob (and the top-N survivors' logprobs when requested).
// Throws on an empty list.
inline Result select_from_sorted(const std::vector<Candidate>& sorted,
                                const Params& p, Rng& rng) {
  if (sorted.empty()) {
    throw std::runtime_error("glm_sample: empty candidate list");
  }

  // Greedy: argmax under the canonical order; reported probabilities are
  // the RAW distribution's (what a greedy consumer expects logprobs to
  // mean), not a renormalized singleton.
  if (p.temperature <= 0.0f) {
    Result r;
    r.token = sorted[0].id;
    float mx = sorted[0].logit;
    double acc = 0.0;
    for (const Candidate& c : sorted) acc += detmath::exp_f(c.logit - mx);
    const float lse = mx + static_cast<float>(detmath::log_d(acc));
    r.logprob = sorted[0].logit - lse;
    const int n = std::min<int>(p.logprobs, static_cast<int>(sorted.size()));
    for (int i = 0; i < n; ++i) {
      r.top_logprobs.emplace_back(sorted[i].id, sorted[i].logit - lse);
    }
    return r;
  }

  const SelectorState s = selector_state(sorted, p);

  // One draw. fp64 walk over the fp32 probabilities: the walk order is
  // the listed order, so every rank that computes this computes it
  // identically.
  const double r = uniform01(rng);
  ++rng.counter;
  double cum = 0.0;
  size_t chosen = s.final_count;
  size_t last_positive = 0;  // the rounding guard: the last token with mass
  for (size_t i = 0; i < s.final_count; ++i) {
    if (s.exps[i] > 0.0f) last_positive = i;
    cum += s.exps[i] / s.final_den;
    if (cum > r) {
      chosen = i;
      break;
    }
  }
  if (chosen == s.final_count) chosen = last_positive;
  return s.result_for(sorted, chosen, p.logprobs);
}

// The speculative step's row-0 decision over a materialized final set
// (DESIGN §9/§10, exact speculative sampling with a deterministic draft):
// with P the final distribution the plain sampler would draw from, accept
// `draft` iff u1 < P(draft) (a token outside the final set has P = 0), else
// sample the residual — P with the draft removed and renormalized — with
// u2. Two draws every step, whichever way the test falls (the accepted
// row's own next token is drawn from row 1 with u2 by the caller). The
// marginal over the row-0 token is exactly P; the residual walk mirrors
// select_from_sorted's (fp32 quotients over the residual denominator,
// fp64 accumulation, listed order), logprobs are reported under P.
struct SpecOutcome {
  bool accepted = false;
  Result result;  // the draft when accepted, else the residual sample
};

// The proposal the draft was drawn from: Q as (id, mass) over
// the draft head's own final set, mass summing to 1. A DETERMINISTIC draft
// (the head's argmax) carries no proposal — the rule above, whose accept
// rate is P(draft) and so, at temperature, is capped by the target's own
// mode. Given a proposal the rule becomes the standard one: accept with
// min(1, P(x)/Q(x)) and, on rejection, draw the normalized residual
// (P - Q)+; the marginal over the emitted token is still exactly P (the
// same rejection-sampling identity), while the accept rate rises to
// 1 - TV(P, Q). Both halves must use the same Q, which is why the draft's
// final set travels with the draft.
struct Proposal {
  std::vector<std::pair<int32_t, float>> mass;  // id -> Q(id)
  bool empty() const { return mass.empty(); }
  double at(int32_t id) const {
    for (const auto& e : mass)
      if (e.first == id) return static_cast<double>(e.second);
    return 0.0;
  }
};

// Every id the proposal gives mass is in the candidate list.
inline bool proposal_inside(const Proposal& q, const std::vector<Candidate>& sorted) {
  for (const auto& e : q.mass) {
    if (!(e.second > 0.0f)) continue;
    bool found = false;
    for (const Candidate& c : sorted)
      if (c.id == e.first) { found = true; break; }
    if (!found) return false;
  }
  return true;
}

inline SpecOutcome spec_select_from_sorted(const std::vector<Candidate>& sorted,
                                           const Params& p, int32_t draft,
                                           Rng& rng, const Proposal* q = nullptr) {
  if (sorted.empty()) throw std::runtime_error("glm_sample: empty candidate list");
  if (!(p.temperature > 0.0f))
    throw std::invalid_argument("glm_sample: speculative sampling needs T > 0");
  const SelectorState s = selector_state(sorted, p);
  size_t j = s.final_count;  // the draft's index in the final set, if any
  for (size_t i = 0; i < s.final_count; ++i)
    if (sorted[i].id == draft) {
      j = i;
      break;
    }
  const float p_draft = j < s.final_count ? s.exps[j] / s.final_den : 0.0f;
  // The proposal's mass at the draft: 0 when the draft is outside Q (it
  // cannot be, having been drawn from Q — but a re-drafted row can arrive
  // that way, and then the deterministic rule is the right one).
  const double q_draft = (q != nullptr && !q->empty()) ? q->at(draft) : 0.0;
  const bool ratio = q_draft > 0.0;
  const double u1 = uniform01(rng);
  ++rng.counter;
  SpecOutcome out;
  // accept iff u1 < min(1, P/Q) — as u1 < 1, P >= Q always stands.
  if (ratio ? (static_cast<double>(p_draft) > u1 * q_draft)
            : (static_cast<double>(p_draft) > u1)) {
    out.accepted = true;
    out.result = s.result_for(sorted, j, p.logprobs);
    return out;
  }
  const double u2 = uniform01(rng);
  ++rng.counter;
  if (ratio) {
    // The residual (P - Q)+ over the final set, in the listed order and the
    // same fp64 accumulation the plain walk uses.
    double res_den = 0.0;
    for (size_t i = 0; i < s.final_count; ++i) {
      const double pi = static_cast<double>(s.exps[i]) / s.final_den;
      const double qi = q->at(sorted[i].id);
      if (pi > qi) res_den += pi - qi;
    }
    double cum = 0.0;
    size_t chosen = s.final_count, last = s.final_count;
    if (res_den > 0.0) {
      for (size_t i = 0; i < s.final_count; ++i) {
        const double pi = static_cast<double>(s.exps[i]) / s.final_den;
        const double qi = q->at(sorted[i].id);
        const double ri = pi > qi ? pi - qi : 0.0;
        if (ri > 0.0 || last == s.final_count) last = i;
        cum += ri / res_den;
        if (cum > u2) {
          chosen = i;
          break;
        }
      }
    }
    if (chosen == s.final_count) chosen = last;
    if (chosen == s.final_count)
      throw std::logic_error("glm_sample: the residual has no candidate");
    out.result = s.result_for(sorted, chosen, p.logprobs);
    return out;
  }
  const float res_den = j < s.final_count ? s.final_den - s.exps[j] : s.final_den;
  double cum = 0.0;
  size_t chosen = s.final_count;
  size_t last = s.final_count;  // the last residual token with mass
  for (size_t i = 0; i < s.final_count; ++i) {
    if (i == j) continue;
    if (s.exps[i] > 0.0f || last == s.final_count) last = i;
    cum += s.exps[i] / res_den;
    if (cum > u2) {
      chosen = i;
      break;
    }
  }
  if (chosen == s.final_count) chosen = last;
  if (chosen == s.final_count)
    throw std::logic_error("glm_sample: the residual has no candidate");
  out.result = s.result_for(sorted, chosen, p.logprobs);
  return out;
}

// ---------------------------------------------------------------------------
// The centralized oracle (full vocab on one host)
// ---------------------------------------------------------------------------

inline Result sample_reference(const float* logits, int vocab,
                               const Params& p, Rng& rng,
                               const std::vector<int32_t>& context_ids) {
  std::vector<float> v(logits, logits + vocab);
  apply_penalties(v.data(), vocab, 0, p, count_context(context_ids));
  const std::vector<Candidate> sorted = sort_slice(v.data(), vocab, 0);
  return select_from_sorted(sorted, p, rng);
}

// ---------------------------------------------------------------------------
// Distributed fast paths (per-rank host code over vocab slices)
// ---------------------------------------------------------------------------

// Path 1 — greedy. Each rank contributes its slice maximum.
inline Candidate local_max(const float* logits, int n, int slice_begin) {
  // Ascending scan with strict '>' keeps the LOWEST id on ties — the
  // canonical order's tie-break, for free.
  Candidate best{slice_begin, logits[0]};
  for (int i = 1; i < n; ++i) {
    if (logits[i] > best.logit) best = {slice_begin + i, logits[i]};
  }
  return best;
}

inline int32_t merge_greedy(const std::vector<Candidate>& local_maxima) {
  Candidate best = local_maxima[0];
  for (const Candidate& c : local_maxima) {
    if (candidate_before(c, best)) best = c;
  }
  return best.id;
}

// Path 2 — finite top-k. Each rank contributes its exact local top-k
// (canonical order); the merge is a sort of the union under the same
// total order, truncated to k — provably the global top-k in the global
// canonical order. k <= 0 is invalid here (that is the gather path).
inline std::vector<Candidate> local_topk(const float* logits, int n,
                                        int slice_begin, int k) {
  std::vector<Candidate> all;
  all.reserve(n);
  for (int i = 0; i < n; ++i)
    if (present_logit(logits[i])) all.push_back({slice_begin + i, logits[i]});
  // partial_sort: O(n log k) — this runs per decode step on a ~38k-entry
  // vocab slice, so the full O(n log n) sort would show up in the
  // step-time budget.
  const size_t k_eff = std::min<size_t>(all.size(), static_cast<size_t>(k));
  std::partial_sort(all.begin(), all.begin() + k_eff, all.end(),
                    candidate_before);
  all.resize(k_eff);
  return all;
}

inline std::vector<Candidate> merge_topk(
    std::vector<std::vector<Candidate>> local_topks, int k) {
  std::vector<Candidate> merged;
  for (auto& l : local_topks) {
    merged.insert(merged.end(), l.begin(), l.end());
  }
  std::sort(merged.begin(), merged.end(), candidate_before);
  merged.resize(std::min<size_t>(merged.size(), static_cast<size_t>(k)));
  return merged;
}

// Exact full-distribution normalizer for a host-visible logit slice. The
// teacher-forced sampling profiler uses one value per vocab shard and folds
// those with logaddexp; this is intentionally fp64 measurement arithmetic,
// separate from the fp32 device verdict that the production sampler will use.
// The slice normalizer's summation order, shared with the device kernels
// that compute it in parallel: the terms of each kLseChunk-id chunk (ids
// past the end are zeros) summed by recursive halving — t[i] += t[i + h]
// for h = kLseChunk/2 down to 1 — and the chunk partials, padded with
// zeros to a power of two, summed the same way. Every add's operands are
// fixed by position, so any thread mapping that keeps the trees is
// bitwise this; the dependent chain is 8 + log2(chunks) adds where a
// sequential sum's was 256 + chunks (a dependent fp64 add is ~90 ns on
// GB10, which made the sequential order ~30 µs per slice on the device).
inline constexpr int kLseChunk = 256;

// t[0, width) summed by recursive halving; width a power of two.
inline double halving_sum(double* t, int width) {
  for (int h = width >> 1; h >= 1; h >>= 1)
    for (int i = 0; i < h; ++i) t[i] += t[i + h];
  return t[0];
}

template <typename ScaledAt>
inline double chunked_exp_sum(int n, double top, ScaledAt scaled_at) {
  const int nchunks = (n + kLseChunk - 1) / kLseChunk;
  int width = 1;
  while (width < nchunks) width <<= 1;
  std::vector<double> partials(static_cast<size_t>(width), 0.0);
  double terms[kLseChunk];
  for (int c = 0; c < nchunks; ++c) {
    const int c0 = c * kLseChunk;
    for (int j = 0; j < kLseChunk; ++j)
      terms[j] = c0 + j < n
                     ? detmath::exp_d(static_cast<double>(scaled_at(c0 + j)) - top)
                     : 0.0;
    partials[static_cast<size_t>(c)] = halving_sum(terms, kLseChunk);
  }
  return halving_sum(partials.data(), width);
}

inline double slice_logsumexp(const float* logits, int n) {
  if (logits == nullptr || n <= 0)
    throw std::invalid_argument("glm_sample: empty logit slice");
  double top = -INFINITY;
  for (int i = 0; i < n; ++i)
    top = std::max(top, static_cast<double>(logits[i]));
  if (top == -INFINITY) return -INFINITY;  // every id masked: no mass here
  const double sum =
      chunked_exp_sum(n, top, [&](int i) { return logits[i]; });
  return top + detmath::log_d(sum);
}

// The production candidate path needs the normalizer after temperature.
// Match select_from_sorted()'s fp32 division before promoting to fp64 for the
// measurement/fold arithmetic; doing the division in fp64 would describe a
// subtly different distribution.
inline double slice_logsumexp(const float* logits, int n,
                              float temperature) {
  if (logits == nullptr || n <= 0)
    throw std::invalid_argument("glm_sample: empty logit slice");
  if (!(temperature > 0.0f) || !std::isfinite(temperature))
    throw std::invalid_argument(
        "glm_sample: temperature-scaled lse requires finite temperature > 0");
  double top = -INFINITY;
  for (int i = 0; i < n; ++i) {
    const float scaled = logits[i] / temperature;
    top = std::max(top, static_cast<double>(scaled));
  }
  if (top == -INFINITY) return -INFINITY;  // every id masked: no mass here
  const double sum = chunked_exp_sum(
      n, top, [&](int i) { return logits[i] / temperature; });
  return top + detmath::log_d(sum);
}

// Canonical rank-order fold of per-slice log-sum-exp values. The centralized
// test oracle uses this same fold, so the candidate decision is judged
// against the exact normalizer the distributed path actually transports.
// A slice whose every id is masked folds as -inf (a zero term); at least
// one slice must hold mass.
inline double merge_logsumexp(const std::vector<double>& slice_lses) {
  if (slice_lses.empty())
    throw std::invalid_argument("glm_sample: no slice log-sum-exp values");
  const double top =
      *std::max_element(slice_lses.begin(), slice_lses.end());
  if (!std::isfinite(top))
    throw std::invalid_argument("glm_sample: non-finite slice log-sum-exp");
  double sum = 0.0;
  for (double lse : slice_lses) {
    if (lse == -INFINITY) continue;  // exp_d(-inf) is 0: a masked slice
    if (!std::isfinite(lse))
      throw std::invalid_argument("glm_sample: non-finite slice log-sum-exp");
    sum += detmath::exp_d(lse - top);
  }
  return top + detmath::log_d(sum);
}

// Attempts an EXACT stochastic decision from the canonical global top-k
// prefix plus the fold normalizer of the complete temperature-scaled
// distribution (merge_logsumexp over the per-slice values in rank order).
//
// This is the host oracle for M6.6b's device verdict kernel, and it is
// WIDTH-INDEPENDENT by construction: every step whose value depends on the
// unseen tail is expressed through the fold normalizer, never through a
// materialized full-vocabulary sum, and the same code runs whether the list
// is a prefix or the complete vocabulary. A prefix that resolves therefore
// yields bitwise the result the complete list yields, and the complete list
// is the full-logit fallback (sample_reference_sharded) — so a request's
// outcome never depends on the transported width k, only on whether the
// fast path could decide it. (A shortcut that hands a complete list to a
// different arithmetic breaks exactly this: at an exact-tie crossing the
// fp32 and fp64 cumulative sums disagree on the nucleus, and the token with
// it.)
//
// Regimes, in HF warper order:
//   finite top_k : the post-top-k support is materialized once the prefix
//                  holds top_k candidates and the shared selector runs over
//                  it with the request unchanged — bitwise sample_reference()
//                  by the M6 d3 merge proof; below top_k candidates, fallback.
//   min_p > 0    : survivors are a prefix (descending candidates, positive
//                  temperature), decided against p_max in the scaled-logit
//                  domain; the first failure proves every later token fails.
//                  The survivor set is then materialized and the shared
//                  selector finishes it (top-p over the survivors' own
//                  denominator, the walk). No failure inside an incomplete
//                  prefix: fallback.
//   top_p < 1    : the crossing is decided on the fp64 fold masses in listed
//                  order; inside the prefix, the nucleus is materialized and
//                  the shared selector finishes it. No crossing inside an
//                  incomplete prefix: fallback.
//   otherwise    : pure temperature sampling. The survivor set is the whole
//                  vocabulary and is never materialized: the draw walks the
//                  fp64 fold masses and resolves when it lands inside the
//                  prefix (logprobs against the fold normalizer); in the
//                  unseen tail, fallback.
// A fallback consumes NO RNG draw: the full-logit path re-runs this decision
// over the complete list with the same (seed, counter).
// What a canonical prefix knows about the request's final support under
// the fold normalizer: materialized (the first `n` candidates with the
// selector's params over them), pure (no truncation: the walk decides
// against the fold masses), or nothing yet (fallback).
struct PrefixSupport {
  enum class Kind { kFallback, kMaterialized, kPure };
  Kind kind = Kind::kFallback;
  size_t n = 0;
  Params exact;
  double covered_mass = 0.0;
};

inline PrefixSupport resolve_support(const std::vector<Candidate>& sorted_prefix,
                                     int vocab_size,
                                     double global_scaled_logsumexp,
                                     const Params& p) {
  if (sorted_prefix.empty())
    throw std::invalid_argument("glm_sample: empty sampling prefix");
  if (vocab_size <= 0 || sorted_prefix.size() > static_cast<size_t>(vocab_size))
    throw std::invalid_argument("glm_sample: sampling prefix/vocab mismatch");
  if (!(p.temperature > 0.0f) || !std::isfinite(p.temperature))
    throw std::invalid_argument(
        "glm_sample: prefix sampling requires finite temperature > 0");
  if (!(p.top_p > 0.0f && p.top_p <= 1.0f) || !std::isfinite(p.top_p))
    throw std::invalid_argument("glm_sample: top_p must be in (0, 1]");
  if (!(p.min_p >= 0.0f && p.min_p <= 1.0f) || !std::isfinite(p.min_p))
    throw std::invalid_argument("glm_sample: min_p must be in [0, 1]");
  if (p.top_k < 0)
    throw std::invalid_argument("glm_sample: top_k must be nonnegative");
  if (!std::isfinite(global_scaled_logsumexp))
    throw std::invalid_argument(
        "glm_sample: non-finite global sampling log-sum-exp");
  for (size_t i = 1; i < sorted_prefix.size(); ++i) {
    if (candidate_before(sorted_prefix[i], sorted_prefix[i - 1]))
      throw std::invalid_argument(
          "glm_sample: sampling prefix is not in canonical order");
  }

  const size_t held = sorted_prefix.size();
  const bool complete = held == static_cast<size_t>(vocab_size);
  const auto scaled_at = [&](size_t i) {
    return sorted_prefix[i].logit / p.temperature;
  };
  const auto mass_at = [&](size_t i) {
    return detmath::exp_d(static_cast<double>(scaled_at(i)) -
                          global_scaled_logsumexp);
  };
  PrefixSupport out;
  for (size_t i = 0; i < held; ++i) out.covered_mass += mass_at(i);
  out.exact = p;

  if (p.top_k > 0) {
    const size_t required = std::min(static_cast<size_t>(p.top_k),
                                     static_cast<size_t>(vocab_size));
    if (held < required) return out;
    out.kind = PrefixSupport::Kind::kMaterialized;
    out.n = required;
    return out;
  }
  if (p.min_p > 0.0f) {
    const float threshold = scaled_at(0) + detmath::log_f(p.min_p);
    size_t survivors = held;
    bool bounded = complete;
    for (size_t i = 0; i < held; ++i) {
      if (scaled_at(i) < threshold) {
        survivors = i;  // index 0 never fails: log(min_p) <= 0
        bounded = true;
        break;
      }
    }
    if (!bounded) return out;
    out.kind = PrefixSupport::Kind::kMaterialized;
    out.n = survivors;
    out.exact.min_p = 0.0f;
    return out;
  }
  if (p.top_p < 1.0f) {
    const double top_p = static_cast<double>(p.top_p);
    double cumulative = 0.0;
    size_t nucleus = 0;
    for (size_t i = 0; i < held; ++i) {
      cumulative += mass_at(i);
      if (cumulative >= top_p) {
        nucleus = i + 1;
        break;
      }
    }
    if (nucleus == 0) {
      if (!complete) return out;
      nucleus = held;  // the selector's rule: a sum that falls short keeps all
    }
    out.kind = PrefixSupport::Kind::kMaterialized;
    out.n = nucleus;
    out.exact.top_p = 1.0f;
    return out;
  }
  out.kind = PrefixSupport::Kind::kPure;
  out.n = held;
  return out;
}

// Attempts an EXACT stochastic decision from the canonical global top-k
// prefix plus the fold normalizer of the complete temperature-scaled
// distribution (merge_logsumexp over the per-slice values in rank order).
//
// This is the host oracle for M6.6b's device verdict kernel, and it is
// WIDTH-INDEPENDENT by construction: every step whose value depends on the
// unseen tail is expressed through the fold normalizer, never through a
// materialized full-vocabulary sum, and the same code runs whether the list
// is a prefix or the complete vocabulary. A prefix that resolves therefore
// yields bitwise the result the complete list yields, and the complete list
// is the full-logit fallback (sample_reference_sharded) — so a request's
// outcome never depends on the transported width k, only on whether the
// fast path could decide it. (A shortcut that hands a complete list to a
// different arithmetic breaks exactly this: at an exact-tie crossing the
// fp32 and fp64 cumulative sums disagree on the nucleus, and the token with
// it.)
//
// Regimes, in HF warper order (resolve_support):
//   finite top_k : the post-top-k support is materialized once the prefix
//                  holds top_k candidates and the shared selector runs over
//                  it with the request unchanged — bitwise sample_reference()
//                  by the M6 d3 merge proof; below top_k candidates, fallback.
//   min_p > 0    : survivors are a prefix (descending candidates, positive
//                  temperature), decided against p_max in the scaled-logit
//                  domain; the first failure proves every later token fails.
//                  The survivor set is then materialized and the shared
//                  selector finishes it (top-p over the survivors' own
//                  denominator, the walk). No failure inside an incomplete
//                  prefix: fallback.
//   top_p < 1    : the crossing is decided on the fp64 fold masses in listed
//                  order; inside the prefix, the nucleus is materialized and
//                  the shared selector finishes it. No crossing inside an
//                  incomplete prefix: fallback.
//   otherwise    : pure temperature sampling. The survivor set is the whole
//                  vocabulary and is never materialized: the draw walks the
//                  fp64 fold masses and resolves when it lands inside the
//                  prefix (logprobs against the fold normalizer); in the
//                  unseen tail, fallback.
// A fallback consumes NO RNG draw: the full-logit path re-runs this decision
// over the complete list with the same (seed, counter).
inline PrefixDecision sample_from_prefix(
    const std::vector<Candidate>& sorted_prefix, int vocab_size,
    double global_scaled_logsumexp, const Params& p, Rng& rng) {
  const PrefixSupport support =
      resolve_support(sorted_prefix, vocab_size, global_scaled_logsumexp, p);
  PrefixDecision decision;
  decision.normalizer = global_scaled_logsumexp;
  decision.covered_mass = support.covered_mass;
  if (support.kind == PrefixSupport::Kind::kFallback) return decision;
  if (support.kind == PrefixSupport::Kind::kMaterialized) {
    // The survivor set is known and materialized: the shared selector
    // finishes over precisely that list, so the final denominator, the
    // logprobs and the counter semantics are its own — identical for a
    // prefix and for the complete list, because both hand it the same
    // candidates.
    const std::vector<Candidate> materialized(
        sorted_prefix.begin(), sorted_prefix.begin() + support.n);
    decision.result = select_from_sorted(materialized, support.exact, rng);
    decision.resolved = true;
    return decision;
  }

  // Pure temperature sampling over the whole vocabulary.
  const size_t held = sorted_prefix.size();
  const bool complete = held == static_cast<size_t>(vocab_size);
  const auto scaled_at = [&](size_t i) {
    return sorted_prefix[i].logit / p.temperature;
  };
  const auto mass_at = [&](size_t i) {
    return detmath::exp_d(static_cast<double>(scaled_at(i)) -
                          global_scaled_logsumexp);
  };
  if (!complete && p.logprobs > static_cast<int>(held)) return decision;
  const double draw = uniform01(rng);
  double cumulative = 0.0;
  size_t chosen = held;
  size_t last_positive = 0;  // the rounding guard: the last token with mass
  for (size_t i = 0; i < held; ++i) {
    const double m = mass_at(i);
    if (m > 0.0) last_positive = i;
    cumulative += m;
    if (cumulative > draw) {
      chosen = i;
      break;
    }
  }
  if (chosen == held) {
    if (!complete) return decision;  // in the unseen tail; the RNG is untouched
    chosen = last_positive;          // the selector's rounding guard
  }
  ++rng.counter;
  const float lse = static_cast<float>(global_scaled_logsumexp);
  decision.result.token = sorted_prefix[chosen].id;
  decision.result.logprob = scaled_at(chosen) - lse;
  const int n = std::min<int>(p.logprobs, static_cast<int>(held));
  for (int j = 0; j < n; ++j) {
    decision.result.top_logprobs.emplace_back(
        sorted_prefix[static_cast<size_t>(j)].id,
        scaled_at(static_cast<size_t>(j)) - lse);
  }
  decision.resolved = true;
  return decision;
}

// The speculative step's row-0 decision from a prefix (DESIGN §9): accept
// `draft` with its exact probability under the final distribution, else
// sample the residual — width-independent exactly as sample_from_prefix
// is, and the host oracle for the device's T=2 verdict. Materialized
// supports hand the same list to spec_select_from_sorted; the pure regime
// evaluates the draft's fold mass (the draft must be inside the prefix or
// the list complete — else fallback) and walks the residual against
// u2 * (1 - P(draft)). A fallback consumes NO draw (the complete-list
// re-run redoes the accept test with the same u1).
struct SpecPrefixDecision {
  bool resolved = false;
  bool accepted = false;
  Result result;  // the draft when accepted, else the residual sample
  double covered_mass = 0.0;
  double normalizer = 0.0;
};

// `draft_excluded`: the caller knows the draft is outside the row's
// support (a masked id): its probability is 0 without the list having to
// show it, so an incomplete prefix still decides (M6 6g).
// `proposal`: the distribution the draft was drawn from — the ratio accept
// min(1, P/Q) and the (P - Q)+ residual in both regimes (2026-10-05; the
// pure temperature walk kept the deterministic rule before). In the pure
// regime the residual's total mass is 1 - sum_x min(P, Q), exact only when
// every id the proposal gives mass is inside the prefix (an unseen one
// hides min(P, Q) there): such a row keeps the deterministic P(draft) rule,
// exact for a draft drawn from anything (a choice the draws never see);
// a residual walk past the prefix falls back as the deterministic walk's
// does. The device mirrors the split exactly.
inline SpecPrefixDecision spec_accept_from_prefix(
    const std::vector<Candidate>& sorted_prefix, int vocab_size,
    double global_scaled_logsumexp, int32_t draft, const Params& p,
    Rng& rng, bool draft_excluded = false, const Proposal* proposal = nullptr) {
  const PrefixSupport support =
      resolve_support(sorted_prefix, vocab_size, global_scaled_logsumexp, p);
  SpecPrefixDecision decision;
  decision.normalizer = global_scaled_logsumexp;
  decision.covered_mass = support.covered_mass;
  if (support.kind == PrefixSupport::Kind::kFallback) return decision;
  if (support.kind == PrefixSupport::Kind::kMaterialized) {
    const std::vector<Candidate> materialized(
        sorted_prefix.begin(), sorted_prefix.begin() + support.n);
    const SpecOutcome o =
        spec_select_from_sorted(materialized, support.exact, draft, rng, proposal);
    decision.resolved = true;
    decision.accepted = o.accepted;
    decision.result = o.result;
    return decision;
  }

  // Pure temperature sampling.
  const size_t held = sorted_prefix.size();
  const bool complete = held == static_cast<size_t>(vocab_size);
  const auto scaled_at = [&](size_t i) {
    return sorted_prefix[i].logit / p.temperature;
  };
  const auto mass_at = [&](size_t i) {
    return detmath::exp_d(static_cast<double>(scaled_at(i)) -
                          global_scaled_logsumexp);
  };
  size_t j = held;
  if (!draft_excluded)
    for (size_t i = 0; i < held; ++i)
      if (sorted_prefix[i].id == draft) {
        j = i;
        break;
      }
  if (j == held && !complete && !draft_excluded)
    return decision;  // the draft's mass is unseen
  const double p_draft = j < held ? mass_at(j) : 0.0;
  // The ratio rule when the proposal carries the draft: the accept test
  // against Q(draft), the residual (P - Q)+ over the prefix against the
  // exact total 1 - sum min(P, Q), which needs every proposed id inside
  // the prefix (else the tail hides part of the overlap: fallback).
  const double q_draft = (proposal != nullptr && !proposal->empty()) ? proposal->at(draft) : 0.0;
  // The ratio rule needs every id the proposal gives mass inside the prefix
  // (an exact set test; two float sums in different orders are not), else
  // a proposed id's overlap is unseen and the row keeps the deterministic rule.
  const bool ratio = q_draft > 0.0 && (complete || proposal_inside(*proposal, sorted_prefix));
  double residual_total = 1.0 - p_draft;
  if (ratio) {
    double overlap = 0.0;
    for (size_t i = 0; i < held; ++i) {
      const double qi = proposal->at(sorted_prefix[i].id);
      if (qi <= 0.0) continue;
      overlap += std::min(mass_at(i), qi);
    }
    residual_total = 1.0 - overlap;
  }
  const Rng entry = rng;
  const double u1 = uniform01(rng);
  ++rng.counter;
  const float lse = static_cast<float>(global_scaled_logsumexp);
  if (ratio ? (p_draft > u1 * q_draft) : (p_draft > u1)) {
    decision.resolved = true;
    decision.accepted = true;
    decision.result.token = draft;
    decision.result.logprob = scaled_at(j) - lse;
    return decision;
  }
  const double u2 = uniform01(rng);
  ++rng.counter;
  const double threshold = u2 * residual_total;
  double cumulative = 0.0;
  size_t chosen = held;
  size_t last = held;  // the last residual token with mass
  for (size_t i = 0; i < held; ++i) {
    if (!ratio && i == j) continue;
    double m = mass_at(i);
    if (ratio) {
      const double qi = proposal->at(sorted_prefix[i].id);
      m = m > qi ? m - qi : 0.0;
    }
    if (m > 0.0 || last == held) last = i;
    cumulative += m;
    if (cumulative > threshold) {
      chosen = i;
      break;
    }
  }
  if (chosen == held) {
    if (!complete) {
      rng = entry;  // the residual lies in the unseen tail: nothing consumed
      return decision;
    }
    chosen = last;
  }
  if (chosen == held)
    throw std::logic_error("glm_sample: the residual has no candidate");
  decision.resolved = true;
  decision.result.token = sorted_prefix[chosen].id;
  decision.result.logprob = scaled_at(chosen) - lse;
  return decision;
}

// The GREEDY decision with logprobs (temperature 0 requests that ask for
// them, or carry penalties): the canonical first candidate of the prefix —
// the global argmax, bitwise the greedy pick's token — reported under the
// raw distribution (the fold normalizer at temperature 1, as
// select_from_sorted's greedy branch reports over a complete list). No
// draw. `logprobs` top entries, all inside the prefix.
inline Result greedy_from_prefix(const std::vector<Candidate>& sorted_prefix,
                                 double raw_logsumexp, int logprobs) {
  if (sorted_prefix.empty())
    throw std::invalid_argument("glm_sample: empty greedy prefix");
  if (!std::isfinite(raw_logsumexp))
    throw std::invalid_argument("glm_sample: non-finite raw log-sum-exp");
  const float lse = static_cast<float>(raw_logsumexp);
  Result out;
  out.token = sorted_prefix[0].id;
  out.logprob = sorted_prefix[0].logit - lse;
  const int n = std::min<int>(logprobs, static_cast<int>(sorted_prefix.size()));
  for (int i = 0; i < n; ++i)
    out.top_logprobs.emplace_back(sorted_prefix[static_cast<size_t>(i)].id,
                                  sorted_prefix[static_cast<size_t>(i)].logit - lse);
  return out;
}

// One contiguous vocabulary slice of the sharded lm head: rank r owns
// [begin, begin + count). The fold normalizer is defined over the slices in
// rank order, so the layout is part of the sharded sampler's semantics, not
// an implementation detail.
struct VocabSlice {
  int begin = 0;
  int count = 0;
};

inline void validate_vocab_slices(const std::vector<VocabSlice>& slices,
                                  int vocab) {
  if (slices.empty() || vocab <= 0)
    throw std::invalid_argument("glm_sample: empty vocabulary layout");
  int next = 0;
  for (const VocabSlice& s : slices) {
    if (s.begin != next || s.count <= 0)
      throw std::invalid_argument(
          "glm_sample: vocabulary slices must be contiguous, non-empty and "
          "in rank order");
    next += s.count;
  }
  if (next != vocab)
    throw std::invalid_argument(
        "glm_sample: vocabulary slices must cover the vocabulary exactly");
}

// The sharded lm head's layout — the loader's lm_head_slice rule (an
// integer split, gap-free and overlap-free at any vocab/world) — which is
// the fold order the sharded sampler is defined over. World 1 is one slice.
inline std::vector<VocabSlice> vocab_layout(int vocab, int world) {
  if (vocab <= 0 || world <= 0 || world > vocab)
    throw std::invalid_argument("glm_sample: vocab/world layout");
  std::vector<VocabSlice> out;
  out.reserve(static_cast<size_t>(world));
  for (int r = 0; r < world; ++r) {
    const int begin =
        static_cast<int>(static_cast<int64_t>(vocab) * r / world);
    const int end =
        static_cast<int>(static_cast<int64_t>(vocab) * (r + 1) / world);
    out.push_back({begin, end - begin});
  }
  return out;
}

// The fold normalizer of the complete temperature-scaled distribution: each
// slice's fp64 log-sum-exp, folded in rank order — bitwise what the bus
// transports and folds.
inline double sharded_scaled_logsumexp(const float* logits,
                                       const std::vector<VocabSlice>& slices,
                                       float temperature) {
  std::vector<double> lses;
  lses.reserve(slices.size());
  for (const VocabSlice& s : slices)
    lses.push_back(slice_logsumexp(logits + s.begin, s.count, temperature));
  return merge_logsumexp(lses);
}

// The centralized reference for the sharded sampler at a given layout, and
// the full-logit FALLBACK's exact computation: penalties, the canonical
// sort, the fold normalizer, then the very same decision over the complete
// list — which always resolves. For finite top_k it equals
// sample_reference() bitwise (the shared selector over the same materialized
// support); in the unbounded regimes it coincides with sample_reference()
// except where the fp64 fold and that oracle's single fp32 denominator
// disagree on a crossing — a boundary event whose outcome must not depend
// on k, which is why the fold, not the fp32 sum, is the definition there.
inline Result sample_reference_sharded(
    const float* logits, int vocab, const std::vector<VocabSlice>& slices,
    const Params& p, Rng& rng, const std::vector<int32_t>& context_ids,
    const uint32_t* mask = nullptr) {
  if (!(p.temperature > 0.0f) || !std::isfinite(p.temperature))
    throw std::invalid_argument(
        "glm_sample: the sharded reference is the stochastic path; "
        "temperature <= 0 is the greedy pick's");
  validate_vocab_slices(slices, vocab);
  std::vector<float> v(logits, logits + vocab);
  apply_penalties(v.data(), vocab, 0, p, count_context(context_ids));
  apply_mask(v.data(), vocab, 0, mask, vocab);
  const double lse = sharded_scaled_logsumexp(v.data(), slices, p.temperature);
  const std::vector<Candidate> sorted = sort_slice(v.data(), vocab, 0);
  const PrefixDecision decision = sample_from_prefix(
      sorted, static_cast<int>(sorted.size()), lse, p, rng);
  if (!decision.resolved)
    throw std::logic_error(
        "glm_sample: the complete candidate list must resolve");
  return decision.result;
}

// The full-logit FALLBACK's decision, given the complete PENALIZED logits
// (every rank's slice gathered) and the fold normalizer the prefix step
// already transported. Bitwise sample_reference_sharded()'s result, by width
// independence: widening exact prefixes are tried first (a partial sort of
// the vocabulary costs O(V log k), the complete sort O(V log V)), and the
// first width whose decision resolves is the complete list's decision; the
// complete list is the last width and always resolves. A fallback here
// consumes no draw until it resolves, so the counter advances exactly once.
// Masked (-inf) logits are absent: the decision's vocabulary is the count
// of present ids, so the complete list is complete over them.
inline Result sample_complete_logits(const float* adjusted, int vocab,
                                     double normalizer, const Params& p,
                                     Rng& rng) {
  if (adjusted == nullptr || vocab <= 0)
    throw std::invalid_argument("glm_sample: empty complete logits");
  const int present = count_present(adjusted, vocab);
  if (present <= 0)
    throw std::invalid_argument("glm_sample: every logit is masked");
  static constexpr int kWidths[] = {1024, 8192, 65536};
  for (int width : kWidths) {
    if (width >= present) break;
    const std::vector<Candidate> prefix = local_topk(adjusted, vocab, 0, width);
    const PrefixDecision d = sample_from_prefix(prefix, present, normalizer, p, rng);
    if (d.resolved) return d.result;
  }
  const std::vector<Candidate> sorted = sort_slice(adjusted, vocab, 0);
  const PrefixDecision d = sample_from_prefix(sorted, present, normalizer, p, rng);
  if (!d.resolved)
    throw std::logic_error("glm_sample: the complete logits must resolve");
  return d.result;
}

// The speculative row-0 decision over the complete penalized logits (the
// fallback's, and the reference's) — widening exact prefixes then the full
// sort, bitwise what a resolving prefix decides.
inline SpecPrefixDecision spec_accept_complete(const float* adjusted, int vocab,
                                               double normalizer, int32_t draft,
                                               const Params& p, Rng& rng) {
  if (adjusted == nullptr || vocab <= 0)
    throw std::invalid_argument("glm_sample: empty complete logits");
  const int present = count_present(adjusted, vocab);
  if (present <= 0)
    throw std::invalid_argument("glm_sample: every logit is masked");
  // A masked draft is known excluded: probability 0 whatever the list shows.
  const bool draft_excluded =
      draft < 0 || draft >= vocab || !present_logit(adjusted[draft]);
  static constexpr int kWidths[] = {1024, 8192, 65536};
  for (int width : kWidths) {
    if (width >= present) break;
    const std::vector<Candidate> prefix = local_topk(adjusted, vocab, 0, width);
    const SpecPrefixDecision d = spec_accept_from_prefix(
        prefix, present, normalizer, draft, p, rng, draft_excluded);
    if (d.resolved) return d;
  }
  const std::vector<Candidate> sorted = sort_slice(adjusted, vocab, 0);
  const SpecPrefixDecision d = spec_accept_from_prefix(
      sorted, present, normalizer, draft, p, rng, draft_excluded);
  if (!d.resolved)
    throw std::logic_error("glm_sample: the complete logits must resolve");
  return d;
}

// The sampler for a host that holds EVERY logit (world 1's full head, or a
// diagnostic over gathered slices): penalties, the fold normalizer over the
// layout, the widening decision. Bitwise sample_reference_sharded() at the
// same layout, in O(V log k) rather than a full sort on most calls.
// `mask` (optional): the constrained decoding mask over [0, vocab).
inline Result sample_full_logits(const float* logits, int vocab,
                                 const std::vector<VocabSlice>& layout,
                                 const Params& p, Rng& rng,
                                 const std::vector<int32_t>& context_ids,
                                 const uint32_t* mask = nullptr,
                                 const float* bias = nullptr) {
  if (!(p.temperature > 0.0f) || !std::isfinite(p.temperature))
    throw std::invalid_argument(
        "glm_sample: sample_full_logits is the stochastic path");
  validate_vocab_slices(layout, vocab);
  std::vector<float> v(logits, logits + vocab);
  apply_context_penalties(v.data(), vocab, 0, p, context_ids);
  apply_bias(v.data(), vocab, 0, bias);
  apply_mask(v.data(), vocab, 0, mask, vocab);
  const double lse = sharded_scaled_logsumexp(v.data(), layout, p.temperature);
  return sample_complete_logits(v.data(), vocab, lse, p, rng);
}

// The speculative T=2 step's reference at a layout (DESIGN §9): row 0's
// accept/residual under context C (prompt + committed + the fed token),
// then — when the draft stands — row 1's ordinary sample under C + draft.
// Two draws per step. `winners` is the verify's shape: winners[0] the row-0
// outcome, winners[1] the row-1 sample when accepted; accepted is 2 or 1.
struct SpecStepReference {
  bool accepted = false;
  int32_t winners[2] = {-1, -1};
  Result row0;
  Result row1;
};

// `mask0`/`mask1` (optional): the rows' constrained decoding masks.
inline SpecStepReference spec_reference_sharded(
    const float* row0, const float* row1, int vocab,
    const std::vector<VocabSlice>& layout, int32_t draft, const Params& p,
    Rng& rng, const std::vector<int32_t>& context_ids,
    const uint32_t* mask0 = nullptr, const uint32_t* mask1 = nullptr) {
  if (!(p.temperature > 0.0f) || !std::isfinite(p.temperature))
    throw std::invalid_argument("glm_sample: the speculative reference is the "
                                "stochastic path");
  validate_vocab_slices(layout, vocab);
  SpecStepReference out;
  {
    std::vector<float> v(row0, row0 + vocab);
    apply_penalties(v.data(), vocab, 0, p, count_context(context_ids));
    apply_mask(v.data(), vocab, 0, mask0, vocab);
    const double lse = sharded_scaled_logsumexp(v.data(), layout, p.temperature);
    const SpecPrefixDecision d =
        spec_accept_complete(v.data(), vocab, lse, draft, p, rng);
    out.accepted = d.accepted;
    out.row0 = d.result;
    out.winners[0] = d.result.token;
  }
  if (out.accepted) {
    std::vector<int32_t> context = context_ids;
    context.push_back(draft);
    std::vector<float> v(row1, row1 + vocab);
    apply_penalties(v.data(), vocab, 0, p, count_context(context));
    apply_mask(v.data(), vocab, 0, mask1, vocab);
    const double lse = sharded_scaled_logsumexp(v.data(), layout, p.temperature);
    out.row1 = sample_complete_logits(v.data(), vocab, lse, p, rng);
    out.winners[1] = out.row1.token;
  }
  return out;
}

// Probability mass covered by each requested prefix of an already
// canonical global candidate list, under the full vocabulary normalizer.
// This is the sizing instrument for the device pick table: at top_p=0.95 a
// prefix whose mass is below 0.95 cannot resolve nucleus sampling locally
// and must take the exact full-logit fallback.
inline std::vector<double> topk_probability_masses(
    const std::vector<Candidate>& sorted, double global_logsumexp,
    const std::vector<int>& ks) {
  if (sorted.empty())
    throw std::invalid_argument("glm_sample: empty top-k candidate list");
  if (!std::isfinite(global_logsumexp))
    throw std::invalid_argument("glm_sample: non-finite global log-sum-exp");
  int previous = 0;
  for (int k : ks) {
    if (k <= previous)
      throw std::invalid_argument(
          "glm_sample: top-k mass prefixes must be positive and increasing");
    previous = k;
  }

  std::vector<double> out;
  out.reserve(ks.size());
  double mass = 0.0;
  size_t next_k = 0;
  for (size_t i = 0; i < sorted.size() && next_k < ks.size(); ++i) {
    mass += detmath::exp_d(static_cast<double>(sorted[i].logit) -
                           global_logsumexp);
    while (next_k < ks.size() && i + 1 >= static_cast<size_t>(ks[next_k])) {
      out.push_back(mass);
      ++next_k;
    }
  }
  // A diagnostic over a tiny synthetic vocabulary may request a prefix
  // wider than the vocabulary. Its mass is simply the complete available
  // candidate set, matching top-k's k>=vocab no-op semantics.
  while (next_k < ks.size()) {
    out.push_back(mass);
    ++next_k;
  }
  return out;
}


// ---------------------------------------------------------------------------
// Block verification (Sun, Mendlovic, Leviathan, Aharoni, Ro, Beirami,
// Suresh 2024, "Block Verification Accelerates Speculative Decoding",
// Algorithm 2; 2026-10-05). The sampled chain decided jointly instead of
// token by token: along the drafts X_1..X_gamma with the rows' target
// distributions P_t and the drafts' proposals Q_t (a point mass at the
// draft when none), p_i = min(1, p_{i-1} P_{i-1}(X_i) / Q_{i-1}(X_i)); the
// acceptance of the sub-block of length i is h_i = S_i / (S_i + 1 - p_i)
// with S_i = sum_x max(p_i P_i(x) - Q_i(x), 0) (h_gamma = p_gamma); tau is
// the LARGEST i with eta_i <= h_i over gamma uniforms; the block's next
// token is drawn from the residual (p_tau P_tau - Q_tau)+ at row tau, or
// from P_gamma plainly when every draft stood. Exact for the output
// distribution and never fewer tokens in expectation than the token rule
// (the paper's Theorems 1 and 2). The device's arithmetic (kernels/
// sample_pick.cu, the verdict's block path) is this one: fp32 masses of a
// materialized final set, fp64 fold masses in the pure regime, the sums in
// candidate order, the pure regime's unseen tail counted as p_i (1 -
// covered) in S_i (it carries no proposal mass once every proposed id is
// inside the prefix — the condition, with every draft priced, for the
// block rule to run at all; otherwise the caller takes the token rule, a
// choice the draws never see).
struct BlockDecision {
  bool decidable = false;   // the block rule could run (else the token rule)
  bool resolved = false;    // the chain's next token drawn (else fallback_row)
  int fallback_row = -1;    // tau when the pure regime's residual lies in the
                            // unseen tail (block_p set), gamma when the last
                            // row's plain sample does
  int tau = 0;              // accepted drafts
  double block_p = 1.0;     // p_tau, the fallback residual's scale
  std::vector<int32_t> winners;  // rows [0, tau]: the drafts, then the next token
  std::vector<float> logprobs;   // per decided row
  int32_t next = -1;
  const char* reason = "";       // why the block rule could not run (diagnostics)
};

// `rows[t]` is row t's merged prefix in canonical order (t = 0..gamma), with
// its fold log-sum-exp `lses[t]`; `drafts[t]` the draft fed to row t + 1;
// `proposals[t]` the distribution it was drawn from (null: a point mass);
// `excluded[t]`: the draft is masked (probability 0 without the prefix
// having to show it). The draws: gamma uniforms at rng.counter .. + gamma
// - 1, then the residual's or the last row's.
inline BlockDecision block_verify_from_prefixes(
    const std::vector<std::vector<Candidate>>& rows, int vocab_size,
    const std::vector<double>& lses, const std::vector<int32_t>& drafts,
    const std::vector<const Proposal*>& proposals, const Params& p, Rng& rng,
    const std::vector<bool>& excluded = {}) {
  BlockDecision out;
  const int gamma = static_cast<int>(drafts.size());
  if (gamma < 1 || rows.size() != static_cast<size_t>(gamma + 1) || lses.size() != rows.size() ||
      proposals.size() != drafts.size())
    throw std::invalid_argument("block_verify: rows, lses, drafts and proposals must agree");
  if (!(p.temperature > 0.0f)) throw std::invalid_argument("block_verify: needs T > 0");
  struct Row {
    PrefixSupport support;
    SelectorState state;  // the materialized regime's final set
    bool materialized = false, complete = false, live = false;
    size_t fin = 0, j = 0;
    float lse = 0.0f;
    double q_at = 1.0;
  };
  std::vector<Row> r(static_cast<size_t>(gamma));
  std::vector<double> pcum(static_cast<size_t>(gamma) + 1, 1.0);
  const auto mass_at = [&](int t, size_t i) {
    return detmath::exp_d(static_cast<double>(rows[t][i].logit / p.temperature) - lses[t]);
  };
  for (int t = 0; t < gamma; ++t) {
    Row& row = r[static_cast<size_t>(t)];
    if (rows[t].empty()) { out.reason = "empty row"; return out; }
    row.support = resolve_support(rows[t], vocab_size, lses[t], p);
    if (row.support.kind == PrefixSupport::Kind::kFallback) { out.reason = "unresolved support"; return out; }
    row.materialized = row.support.kind == PrefixSupport::Kind::kMaterialized;
    row.complete = rows[t].size() == static_cast<size_t>(vocab_size);
    const bool is_excluded = t < static_cast<int>(excluded.size()) && excluded[static_cast<size_t>(t)];
    if (row.materialized) {
      const std::vector<Candidate> materialized(rows[t].begin(), rows[t].begin() + row.support.n);
      row.state = selector_state(materialized, row.support.exact);
      row.fin = row.state.final_count;
      row.lse = row.state.lse;
    } else {
      row.fin = rows[t].size();
      row.lse = static_cast<float>(lses[t]);
    }
    row.j = row.fin;
    if (!is_excluded)
      for (size_t i = 0; i < row.fin; ++i)
        if (rows[t][i].id == drafts[static_cast<size_t>(t)]) { row.j = i; break; }
    if (row.j == row.fin && !is_excluded && !row.materialized && !row.complete) { out.reason = "draft unseen"; return out; }
    const Proposal* q = proposals[static_cast<size_t>(t)];
    row.live = q != nullptr && !q->empty() && q->at(drafts[static_cast<size_t>(t)]) > 0.0;
    if (row.live && !row.materialized && !row.complete && !proposal_inside(*q, rows[t])) {
      // An unseen proposed id: the row's draft counts as a point mass (exact
      // for a draft drawn from anything; the device does the same).
      row.live = false;
    }
    if (row.live) row.q_at = q->at(drafts[static_cast<size_t>(t)]);
    const double p_x = row.j < row.fin
        ? (row.materialized ? static_cast<double>(row.state.exps[row.j] / row.state.final_den) : mass_at(t, row.j))
        : 0.0;
    double pi = pcum[static_cast<size_t>(t)] * (p_x / row.q_at);
    if (pi > 1.0) pi = 1.0;
    pcum[static_cast<size_t>(t) + 1] = pi;
  }
  out.decidable = true;
  const auto p_of = [&](int t, size_t c) {
    const Row& row = r[static_cast<size_t>(t)];
    return row.materialized ? static_cast<double>(row.state.exps[c] / row.state.final_den) : mass_at(t, c);
  };
  const auto q_of = [&](int t, size_t c) {
    const Row& row = r[static_cast<size_t>(t)];
    if (row.live) return proposals[static_cast<size_t>(t)]->at(rows[t][c].id);
    return rows[t][c].id == drafts[static_cast<size_t>(t)] ? 1.0 : 0.0;
  };
  int tau = 0;
  for (int i = 1; i <= gamma; ++i) {
    const double pi = pcum[static_cast<size_t>(i)];
    double h;
    if (i == gamma) {
      h = pi;
    } else {
      const int t = i;
      const Row& row = r[static_cast<size_t>(t)];
      double S = 0.0;
      for (size_t c = 0; c < row.fin; ++c) {
        const double d = pi * p_of(t, c) - q_of(t, c);
        if (d > 0.0) S += d;
      }
      if (!row.materialized) S += pi * (1.0 - row.support.covered_mass);
      h = S / (S + 1.0 - pi);
    }
    const Rng draw{rng.seed, rng.counter + static_cast<uint64_t>(i - 1)};
    if (uniform01(draw) <= h) tau = i;
  }
  rng.counter += static_cast<uint64_t>(gamma);
  out.tau = tau;
  for (int t = 0; t < tau; ++t) {
    const Row& row = r[static_cast<size_t>(t)];
    out.winners.push_back(drafts[static_cast<size_t>(t)]);
    const float scaled = rows[t][row.j].logit / p.temperature;
    out.logprobs.push_back(row.j < row.fin ? scaled - row.lse : 0.0f);
  }
  if (tau < gamma) {
    const int t = tau;
    const Row& row = r[static_cast<size_t>(t)];
    const double pt = pcum[static_cast<size_t>(t)];
    double total = 0.0;
    for (size_t c = 0; c < row.fin; ++c) {
      const double d = pt * p_of(t, c) - q_of(t, c);
      if (d > 0.0) total += d;
    }
    if (!row.materialized) total += pt * (1.0 - row.support.covered_mass);
    const double u = uniform01(rng);
    const double threshold = u * total;
    double cum = 0.0;
    size_t chosen = row.fin, last = row.fin;
    for (size_t c = 0; c < row.fin; ++c) {
      double d = pt * p_of(t, c) - q_of(t, c);
      if (d < 0.0) d = 0.0;
      if (d > 0.0 || last == row.fin) last = c;
      cum += d;
      if (cum > threshold) { chosen = c; break; }
    }
    if (chosen == row.fin && !(row.materialized || row.complete)) {
      out.fallback_row = t;
      out.block_p = pt;
      return out;
    }
    if (chosen == row.fin) chosen = last;
    ++rng.counter;
    out.winners.push_back(rows[t][chosen].id);
    out.logprobs.push_back(rows[t][chosen].logit / p.temperature - row.lse);
    out.next = rows[t][chosen].id;
    out.resolved = true;
    return out;
  }
  const PrefixDecision last = sample_from_prefix(rows[gamma], vocab_size, lses[gamma], p, rng);
  if (!last.resolved) {
    out.fallback_row = gamma;
    return out;
  }
  out.winners.push_back(last.result.token);
  out.logprobs.push_back(last.result.logprob);
  out.next = last.result.token;
  out.resolved = true;
  return out;
}

// The host's draw of a block fallback's residual at row tau over the
// COMPLETE logits (the pure regime only: a materialized final set never
// falls back): (block_p P - Q)+ over every present token, the draft a
// point mass when it carries no proposal.
inline Result block_residual_complete(const float* adjusted, int vocab, double normalizer,
                                      double block_p, int32_t draft, const Proposal* q,
                                      const Params& p, Rng& rng) {
  if (adjusted == nullptr || vocab <= 0) throw std::invalid_argument("block_residual: empty complete logits");
  if (p.top_k > 0 || p.top_p < 1.0f || p.min_p > 0.0f)
    throw std::invalid_argument("block_residual: a materialized final set never falls back");
  const int present = count_present(adjusted, vocab);
  if (present <= 0) throw std::invalid_argument("block_residual: every logit is masked");
  const std::vector<Candidate> sorted = sort_slice(adjusted, vocab, 0);
  const bool live = q != nullptr && !q->empty() && q->at(draft) > 0.0;
  const auto q_of = [&](size_t c) {
    return live ? q->at(sorted[c].id) : (sorted[c].id == draft ? 1.0 : 0.0);
  };
  const auto mass_at = [&](size_t c) {
    return detmath::exp_d(static_cast<double>(sorted[c].logit / p.temperature) - normalizer);
  };
  const size_t held = static_cast<size_t>(present);
  double total = 0.0;
  for (size_t c = 0; c < held; ++c) {
    const double d = block_p * mass_at(c) - q_of(c);
    if (d > 0.0) total += d;
  }
  const double u = uniform01(rng);
  const double threshold = u * total;
  double cum = 0.0;
  size_t chosen = held, last = held;
  for (size_t c = 0; c < held; ++c) {
    double d = block_p * mass_at(c) - q_of(c);
    if (d < 0.0) d = 0.0;
    if (d > 0.0 || last == held) last = c;
    cum += d;
    if (cum > threshold) { chosen = c; break; }
  }
  if (chosen == held) chosen = last;
  if (chosen == held) throw std::logic_error("block_residual: the residual has no candidate");
  ++rng.counter;
  Result out;
  out.token = sorted[chosen].id;
  out.logprob = sorted[chosen].logit / p.temperature - static_cast<float>(normalizer);
  return out;
}

}  // namespace dgpp::sample
