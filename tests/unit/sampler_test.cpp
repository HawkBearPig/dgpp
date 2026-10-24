// Sampler tests (host-only): the three distributed fast paths must be
// bitwise-EQUAL to the centralized oracle (the M6 d3 exit criterion —
// no tolerance is permitted, the design's canonical candidate order
// makes the paths converge on identical inputs to the shared selection
// core). These pin: the merge is the exact global top-k under ties, the
// RNG is counter-reproducible, penalties follow the HF warper formulas,
// and the documented filter edge cases hold.
#include "sample/sampler.hpp"

#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/test.hpp"
#include "engine/eager_engine.hpp"

namespace {

using dgpp::sample::apply_penalties;
using dgpp::sample::Candidate;
using dgpp::sample::count_context;
using dgpp::sample::merge_greedy;
using dgpp::sample::merge_topk;
using dgpp::sample::local_max;
using dgpp::sample::local_topk;
using dgpp::sample::Params;
using dgpp::sample::Rng;
using dgpp::sample::Result;
using dgpp::sample::sample_reference;
using dgpp::sample::sample_from_prefix;
using dgpp::sample::sample_reference_sharded;
using dgpp::sample::sample_full_logits;
using dgpp::sample::validate_params;
using dgpp::sample::vocab_layout;
using dgpp::sample::sharded_scaled_logsumexp;
using dgpp::sample::VocabSlice;
using dgpp::sample::select_from_sorted;
using dgpp::sample::sort_slice;
using dgpp::sample::slice_logsumexp;
using dgpp::sample::topk_probability_masses;
using dgpp::sample::uniform01;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// Bitwise float compare (uint32 pattern): the parity contract is exact,
// so == on floats is too loose an instrument for a -0.0/+0.0 or NaN mix.
bool same_bits(float a, float b) {
  uint32_t ua, ub;
  std::memcpy(&ua, &a, 4);
  std::memcpy(&ub, &b, 4);
  return ua == ub;
}

// A deterministic fixture RNG (not the sampler's): builds test vectors.
struct FixtureRng {
  uint64_t s = 0x1234567;
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  float unit() {  // [0, 1): 53-bit draw scaled by 2^-53
    return static_cast<float>(next() >> 11) * (1.0f / 9007199254740992.0f);
  }
};

std::vector<float> make_logits(int n, FixtureRng& fx, float spread = 8.0f) {
  std::vector<float> v(n);
  for (float& x : v) x = (fx.unit() - 0.5f) * spread;
  return v;
}

// Splits a vector into `world` contiguous slices (uneven on purpose —
// vocab/world is rarely integral).
std::vector<std::pair<int, int>> slices_for(int n, int world) {
  std::vector<std::pair<int, int>> out;
  int begin = 0;
  for (int r = 0; r < world; ++r) {
    const int len = (n - begin + (world - r - 1)) / (world - r);
    out.push_back({begin, len});
    begin += len;
  }
  return out;
}

// The same split as a vocabulary layout for the sharded sampler.
std::vector<VocabSlice> layout_for(int n, int world) {
  std::vector<VocabSlice> out;
  for (const auto& [begin, len] : slices_for(n, world))
    out.push_back({begin, len});
  return out;
}

// The distributed top-k pipeline over simulated shards: per-slice
// penalties, per-slice exact local top-k, merge, shared selection.
Result distributed_topk(const std::vector<float>& logits, int world,
                        const Params& p, Rng& rng,
                        const std::vector<int32_t>& context) {
  const std::vector<Candidate> no_candidates;  // unused sentinel
  (void)no_candidates;
  const auto counts = count_context(context);
  const int k_eff = p.top_k;
  std::vector<std::vector<Candidate>> locals;
  for (const auto& [begin, len] : slices_for(static_cast<int>(logits.size()), world)) {
    std::vector<float> slice(logits.begin() + begin, logits.begin() + begin + len);
    apply_penalties(slice.data(), len, begin, p, counts);
    locals.push_back(local_topk(slice.data(), len, begin, k_eff));
  }
  return select_from_sorted(merge_topk(std::move(locals), k_eff), p, rng);
}

// The distributed greedy pipeline over simulated shards.
int32_t distributed_greedy(const std::vector<float>& logits, int world,
                           const Params& p,
                           const std::vector<int32_t>& context) {
  const auto counts = count_context(context);
  std::vector<Candidate> maxima;
  for (const auto& [begin, len] : slices_for(static_cast<int>(logits.size()), world)) {
    std::vector<float> slice(logits.begin() + begin, logits.begin() + begin + len);
    apply_penalties(slice.data(), len, begin, p, counts);
    maxima.push_back(local_max(slice.data(), len, begin));
  }
  return merge_greedy(maxima);
}

bool results_identical(const Result& a, const Result& b) {
  if (a.token != b.token) return false;
  if (!same_bits(a.logprob, b.logprob)) return false;
  if (a.top_logprobs.size() != b.top_logprobs.size()) return false;
  for (size_t i = 0; i < a.top_logprobs.size(); ++i) {
    if (a.top_logprobs[i].first != b.top_logprobs[i].first) return false;
    if (!same_bits(a.top_logprobs[i].second, b.top_logprobs[i].second)) {
      return false;
    }
  }
  return true;
}


// M6 6g: masked ids are absent everywhere in the host sampler. A masked id
// is never the token; the fold tolerates a slice with every id masked; a
// finite-top-k masked sharded decision is bitwise the plain oracle over
// the masked logits (the merge proof, unchanged by absence); a mask
// leaving one id makes it the token whatever the draw; the greedy full
// path picks the masked argmax; a masked draft is rejected outright by the
// complete accept without a fallback.
DGPP_TEST(sampler_maskedIdsAreAbsentEverywhere) {
  using dgpp::sample::apply_mask;
  using dgpp::sample::count_present;
  using dgpp::sample::greedy_from_prefix;
  using dgpp::sample::spec_accept_complete;
  using dgpp::sample::SpecPrefixDecision;
  constexpr int vocab = 640;
  const std::vector<VocabSlice> layout = vocab_layout(vocab, 4);
  std::vector<float> logits(vocab);
  uint64_t x = 0x9e37u;
  for (int i = 0; i < vocab; ++i) {
    x = x * 6364136223846793005ull + 1442695040888963407ull;
    logits[static_cast<size_t>(i)] = static_cast<float>((x >> 33) % 97) * 0.1f - 4.0f;
  }
  logits[17] = 6.0f;  // a peak the masks may or may not admit
  const int words = (vocab + 31) / 32;
  // An allow-list on three of the four slices (slice 3 fully masked).
  std::vector<uint32_t> allow(static_cast<size_t>(words), 0u);
  const std::vector<int> allowed = {3, 17, 200, 331, 400};
  for (const int id : allowed) allow[static_cast<size_t>(id >> 5)] |= 1u << (id & 31);
  Params p;
  p.temperature = 1.0f;
  p.top_p = 0.95f;
  for (uint64_t seed = 1; seed <= 200; ++seed) {
    Rng rng{seed, 0};
    const Result r = sample_reference_sharded(logits.data(), vocab, layout, p, rng, {}, allow.data());
    bool in = false;
    for (const int id : allowed) in = in || id == r.token;
    require(in, "the masked sharded reference never leaves the allow-list");
    require(rng.counter == 1, "one draw");
    Rng rng2{seed, 0};
    const Result f = sample_full_logits(logits.data(), vocab, layout, p, rng2, {}, allow.data());
    require(f.token == r.token && f.logprob == r.logprob,
            "the full-logit sampler under the mask is the sharded reference");
  }
  // The fully masked slice: lse -inf, the fold skips it.
  {
    std::vector<float> v = logits;
    apply_mask(v.data(), vocab, 0, allow.data(), vocab);
    const VocabSlice& last = layout[3];
    require(slice_logsumexp(v.data() + last.begin, last.count, 1.0f) == -INFINITY,
            "a slice with every id masked has lse -inf");
    const double z = sharded_scaled_logsumexp(v.data(), layout, 1.0f);
    // Bitwise the one-slice normalizer over the five allowed logits, in the
    // fold's order: the three finite slice lses folded.
    require(std::isfinite(z), "the fold over a masked slice is finite");
    require(count_present(v.data(), vocab) == 5, "five present ids");
    require(local_topk(v.data(), vocab, 0, 10).size() == 5,
            "local_topk lists only the present ids");
    require(sort_slice(v.data(), vocab, 0).size() == 5,
            "sort_slice lists only the present ids");
  }
  // Finite top_k: the masked sharded decision is the plain oracle over the
  // masked logits, bitwise (the merge proof holds over absent ids).
  {
    Params k = p;
    k.top_k = 4;
    k.top_p = 1.0f;
    std::vector<float> v = logits;
    apply_mask(v.data(), vocab, 0, allow.data(), vocab);
    for (uint64_t seed = 1; seed <= 50; ++seed) {
      Rng a{seed, 0}, b{seed, 0};
      const Result ra = sample_reference_sharded(logits.data(), vocab, layout, k, a, {}, allow.data());
      const Result rb = sample_reference(v.data(), vocab, k, b, {});
      require(ra.token == rb.token && ra.logprob == rb.logprob,
              "masked sharded top-k == the plain oracle over masked logits");
    }
  }
  // One allowed id: the token, whatever the draw; its logprob is 0.
  {
    std::vector<uint32_t> one(static_cast<size_t>(words), 0u);
    one[static_cast<size_t>(331 >> 5)] |= 1u << (331 & 31);
    for (uint64_t seed = 1; seed <= 20; ++seed) {
      Rng rng{seed, 0};
      const Result r = sample_reference_sharded(logits.data(), vocab, layout, p, rng, {}, one.data());
      require(r.token == 331 && r.logprob == 0.0f, "one allowed id is certain");
    }
  }
  // The greedy full path under a mask: the masked argmax under the raw
  // normalizer (the peak at 17 is masked out here).
  {
    std::vector<uint32_t> no17 = allow;
    no17[17 >> 5] &= ~(1u << (17 & 31));
    std::vector<float> v = logits;
    apply_mask(v.data(), vocab, 0, no17.data(), vocab);
    const double lse = sharded_scaled_logsumexp(v.data(), layout, 1.0f);
    const std::vector<Candidate> top = local_topk(v.data(), vocab, 0, 3);
    const Result g = greedy_from_prefix(top, lse, 2);
    require(g.token != 17 && top.size() == 3, "the masked argmax skips the masked peak");
    for (const auto& [id, lp] : g.top_logprobs) require(id != 17, "no masked alternative");
  }
  // A masked draft over the complete masked logits: rejected outright,
  // two draws consumed (the accept test with p = 0, then the residual),
  // the residual inside the mask.
  {
    std::vector<float> v = logits;
    apply_mask(v.data(), vocab, 0, allow.data(), vocab);
    const double z = sharded_scaled_logsumexp(v.data(), layout, 1.0f);
    Params pure = p;
    pure.top_p = 1.0f;
    for (uint64_t seed = 1; seed <= 20; ++seed) {
      Rng rng{seed, 0};
      const SpecPrefixDecision d = spec_accept_complete(v.data(), vocab, z, /*draft=*/99, pure, rng);
      require(d.resolved && !d.accepted && rng.counter == 2, "masked draft: rejected, two draws");
      bool in = false;
      for (const int id : allowed) in = in || id == d.result.token;
      require(in, "the residual is inside the mask");
    }
    // An allowed draft at the peak mostly stands.
    int stood = 0;
    for (uint64_t seed = 1; seed <= 40; ++seed) {
      Rng rng{seed, 0};
      const SpecPrefixDecision d = spec_accept_complete(v.data(), vocab, z, /*draft=*/17, pure, rng);
      stood += d.accepted ? 1 : 0;
    }
    require(stood > 20, "the allowed peak stands most of the time");
  }
}

}  // namespace

DGPP_TEST(rng_is_counter_reproducible_and_seed_sensitive) {
  Rng a{42, 0}, b{42, 0}, c{42, 1}, d{43, 0};
  for (int i = 0; i < 100; ++i) {
    const double da = uniform01(a), db = uniform01(b);
    require(da == db, "same seed+counter must reproduce the draw");
    require(da != uniform01(c), "counter must change the draw");
    require(da != uniform01(d), "seed must change the draw");
    ++a.counter; ++b.counter; ++c.counter; ++d.counter;
  }
  Rng r{42, 0};
  for (int i = 0; i < 1000; ++i) {
    const double u = uniform01(r);
    require(u >= 0.0 && u < 1.0, "uniform01 out of [0,1)");
    ++r.counter;
  }
}

DGPP_TEST(greedy_takes_argmax_lowest_id_on_ties) {
  const std::vector<float> v{0.5f, 3.25f, 3.25f, -1.0f, 3.25f, 0.0f};
  const std::vector<Candidate> sorted = sort_slice(v.data(), v.size(), 0);
  Params p;  // temperature 1: not the greedy branch — use the greedy one:
  p.temperature = 0.0f;
  p.logprobs = 3;
  Rng rng{1, 0};
  const uint64_t counter_before = rng.counter;
  const Result r = select_from_sorted(sorted, p, rng);
  require(r.token == 1, "tie must break to the LOWEST id");
  require(rng.counter == counter_before, "greedy must not consume RNG");
  require(r.top_logprobs.size() == 3, "logprobs count");
  require(r.top_logprobs[0].first == 1, "top logprob is the argmax");
}

DGPP_TEST(distributed_greedy_matches_oracle_with_cross_shard_ties) {
  FixtureRng fx;
  for (int world : {1, 2, 3, 5, 7}) {
    // Duplicate the max value across MANY positions so ties land across
    // shard boundaries for every split.
    std::vector<float> v = make_logits(29, fx);
    for (int i : {0, 3, 8, 14, 22, 28}) v[i] = 9.5f;
    Params p;
    p.temperature = 0.0f;
    Rng oracle_rng{1, 0};
    const int32_t oracle =
        select_from_sorted(sort_slice(v.data(), v.size(), 0), p, oracle_rng)
            .token;
    const int32_t dist = distributed_greedy(v, world, p, {});
    require(oracle == 0, "fixture sanity: lowest tied id wins");
    require(dist == oracle, "distributed greedy != oracle at world " + std::to_string(world));
  }
}

DGPP_TEST(distributed_topk_matches_oracle_bitwise_across_params) {
  FixtureRng fx;
  int cases = 0;
  for (int world : {1, 2, 3, 5}) {
    for (int n : {1, 13, 29}) {
      const std::vector<float> v = make_logits(n, fx);
      // context for penalties, deliberately including ids that fall in
      // different shards
      const std::vector<int32_t> context{0, n / 2, n - 1, n / 2, 0};
      for (float temperature : {1.0f, 0.7f}) {
        for (int k : {1, 3, 40}) {
          for (float top_p : {1.0f, 0.5f, 0.9f}) {
            for (float min_p : {0.0f, 0.05f}) {
              for (int seed : {0, 1, 2}) {
                for (int use_penalties : {0, 1}) {
                  Params p;
                  p.temperature = temperature;
                  p.top_k = k;
                  p.top_p = top_p;
                  p.min_p = min_p;
                  p.logprobs = 5;
                  if (use_penalties) {
                    p.repetition_penalty = 1.3f;
                    p.frequency_penalty = 0.2f;
                    p.presence_penalty = 0.1f;
                  }
                  Rng ra{static_cast<uint64_t>(seed), 0};
                  Rng rb{static_cast<uint64_t>(seed), 0};
                  const Result oracle = sample_reference(v.data(), n, p, ra, context);
                  const Result dist = distributed_topk(v, world, p, rb, context);
                  require(results_identical(oracle, dist),
                          "top-k path drifted from the oracle: world=" +
                              std::to_string(world) + " n=" +
                              std::to_string(n) + " seed=" +
                              std::to_string(seed));
                  require(ra.counter == rb.counter,
                          "RNG counters must advance identically");
                  ++cases;
                }
              }
            }
          }
        }
      }
    }
  }
  require(cases > 2000, "expected a broad sweep, got " + std::to_string(cases));
}

DGPP_TEST(topk_merge_is_exact_under_ties_spanning_shards) {
  const std::vector<float> v{5.0f, 1.0f, 5.0f, 5.0f, 0.5f, 2.0f, 5.0f};
  // Global top-3 under (value desc, id asc) = {0, 2, 3}.
  std::vector<std::vector<Candidate>> locals;
  locals.push_back(local_topk(v.data(), 3, 0, 3));  // ids 0,1,2
  locals.push_back(local_topk(v.data() + 3, 4, 3, 3));  // ids 3,4,5,6
  const std::vector<Candidate> merged = merge_topk(std::move(locals), 3);
  require(merged.size() == 3, "merge keeps k");
  require(merged[0].id == 0 && merged[0].logit == 5.0f, "tie order 0");
  require(merged[1].id == 2 && merged[1].logit == 5.0f, "tie order 1");
  require(merged[2].id == 3 && merged[2].logit == 5.0f, "tie order 2");
}

DGPP_TEST(penalties_follow_hf_warper_formulas) {
  // logit 4.0f at id 0: positive -> rep divides; id 1 negative -> rep
  // multiplies; frequency is additive per count; presence additive once.
  std::vector<float> v{4.0f, -2.0f, 1.0f};
  std::vector<int32_t> context{0, 0, 0, 1, 2};
  Params p;
  p.repetition_penalty = 2.0f;
  p.frequency_penalty = 0.25f;
  p.presence_penalty = 0.5f;
  apply_penalties(v.data(), 3, 0, p, count_context(context));
  // id 0: 4/2 - 0.25*3 - 0.5 = 2 - 0.75 - 0.5 = 0.75
  require(same_bits(v[0], 0.75f), "id 0 penalty math");
  // id 1: -2*2 - 0.25*1 - 0.5 = -4.75
  require(same_bits(v[1], -4.75f), "id 1 penalty math");
  // id 2: 1/2 - 0.25*1 - 0.5 = -0.25
  require(same_bits(v[2], -0.25f), "id 2 penalty math");
}

DGPP_TEST(apply_penalties_touches_only_its_slice) {
  std::vector<float> v{1.0f, 1.0f, 1.0f, 1.0f};
  const auto counts = count_context({1});  // global id 1
  Params p;
  p.presence_penalty = 1.0f;
  // Slice [2, 4): global id 1 is not in it — nothing changes.
  apply_penalties(v.data() + 2, 2, 2, p, counts);
  require(same_bits(v[2], 1.0f) && same_bits(v[3], 1.0f),
          "out-of-slice ids must be ignored");
  // Slice [0, 2): global id 1 IS in it — only v[1] changes.
  apply_penalties(v.data(), 2, 0, p, counts);
  require(same_bits(v[0], 1.0f), "id 0 untouched");
  require(same_bits(v[1], 0.0f), "id 1 penalized");
}

DGPP_TEST(context_penalties_preserve_active_and_signed_zero_arithmetic) {
  for (const int begin : {0, 17}) {
    const std::vector<int32_t> context{begin - 1, begin,     begin,    begin + 1,
                                       begin + 2, begin + 3, begin + 6};
    for (int regime = 0; regime < 8; ++regime) {
      Params p;
      if (regime == 1) p.frequency_penalty = -0.0f;
      if (regime == 2) p.presence_penalty = -0.0f;
      if (regime == 3) p.repetition_penalty = 1.1f;
      if (regime == 4) p.frequency_penalty = 0.25f;
      if (regime == 5) p.presence_penalty = -0.5f;
      if (regime == 6) p.repetition_penalty = 0.9f;
      if (regime == 7) {
        p.repetition_penalty = 2.0f;
        p.frequency_penalty = 0.25f;
        p.presence_penalty = 0.5f;
      }
      require(dgpp::sample::needs_context_penalties(p) == (regime != 0),
              "only positive-zero neutral parameters omit penalty preparation");
      const std::vector<float> input{
          123.0f, -0.0f, -0.0f, 4.0f, -2.0f, 0.0f, -std::numeric_limits<float>::infinity(), 456.0f};
      auto want = input, got = input;
      apply_penalties(want.data() + 1, 6, begin, p, count_context(context));
      dgpp::sample::apply_context_penalties(got.data() + 1, 6, begin, p, context);
      for (size_t i = 0; i < got.size(); ++i)
        require(same_bits(got[i], want[i]), "context penalty slice or arithmetic changed");
    }
  }
}

DGPP_TEST(context_penalties_neutral_omits_arithmetic) {
  // A signaling NaN is a diagnostic sentinel, not a valid inference logit:
  // the old neutral fma quiets it, making a missed shortcut observable.
  std::fenv_t saved;
  require(std::feholdexcept(&saved) == 0, "save the floating-point environment");
  struct RestoreEnvironment {
    const std::fenv_t* saved;
    ~RestoreEnvironment() { std::fesetenv(saved); }
  } restore{&saved};
  const float sentinel = std::bit_cast<float>(0x7f800001u);
  float logit = sentinel;
  dgpp::sample::apply_context_penalties(&logit, 1, 0, Params{}, {0, 0});
  require(same_bits(logit, sentinel), "neutral preparation must leave logits untouched");
}

DGPP_TEST(context_penalties_neutral_preserves_subnormals_zeros_and_infinities) {
  const float tiny = std::numeric_limits<float>::denorm_min();
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<float> got{0.0f,
                         -0.0f,
                         tiny,
                         -tiny,
                         std::numeric_limits<float>::min(),
                         std::numeric_limits<float>::max(),
                         inf,
                         -inf};
  auto want = got;
  const std::vector<int32_t> context{0, 1, 1, 2, 3, 4, 5, 6, 7};
  Params p;
  apply_penalties(want.data(), static_cast<int>(want.size()), 0, p, count_context(context));
  dgpp::sample::apply_context_penalties(got.data(), static_cast<int>(got.size()), 0, p, context);
  for (size_t i = 0; i < got.size(); ++i)
    require(same_bits(got[i], want[i]), "neutral arithmetic preserves every supported float class");
}

DGPP_TEST(host_greedy_penalties_follow_effective_parameters_without_consuming_rng) {
  const auto sample = dgpp::make_w1_sample(4);
  dgpp::DecodeOutputs out;
  out.lm_vocab_begin = 0;
  out.lm_vocab_count = 4;
  out.logits = {5.0f, 4.0f, 3.0f, -0.0f};
  const std::vector<int32_t> context{0, 0, 0, 3};
  const uint32_t words = 0xe;  // exclude the unpenalized winner
  dgpp::text::TokenMask mask;
  mask.vocab = 4;
  mask.allowed = 3;
  mask.words = {words};
  const float bias[] = {0.0f, 0.0f, 2.0f, 0.0f};
  for (int regime = 0; regime < 6; ++regime) {
    auto p = dgpp::sample::greedy_params();
    p.logprobs = 3;
    if (regime == 1) p.repetition_penalty = 2.0f;
    if (regime == 2) p.frequency_penalty = 0.5f;
    if (regime == 3) p.presence_penalty = 2.0f;
    if (regime == 4) p.frequency_penalty = -0.0f;
    if (regime == 5) p.presence_penalty = -0.0f;
    for (bool constrained : {false, true}) {
      auto adjusted = out.logits;
      apply_penalties(adjusted.data(), 4, 0, p, count_context(context));
      dgpp::sample::apply_bias(adjusted.data(), 4, 0, constrained ? bias : nullptr);
      dgpp::sample::apply_mask(adjusted.data(), 4, 0, constrained ? &words : nullptr, 4);
      const auto want = dgpp::sample::greedy_from_prefix(
          sort_slice(adjusted.data(), 4, 0),
          sharded_scaled_logsumexp(adjusted.data(), {{0, 4}}, 1.0f), 3);
      Rng rng{42, 7};
      const auto got =
          sample(out, p, rng, context, constrained ? &mask : nullptr, constrained ? bias : nullptr);
      require(results_identical(got, want) && rng.counter == 7,
              "greedy tokens and reports match original arithmetic without a draw");
      require(got.token == (constrained ? 2 : (regime >= 1 && regime <= 3 ? 1 : 0)),
              "active penalties change the winner; neutral values retain it");
    }
  }
}

DGPP_TEST(neutral_host_sampling_preserves_bias_mask_reports_and_rng) {
  constexpr int vocab = 17;
  const std::vector<VocabSlice> layout = vocab_layout(vocab, 3);
  FixtureRng fx;
  const auto logits = make_logits(vocab, fx);
  std::vector<float> bias(vocab, 0.0f);
  bias[3] = 10.0f;
  const uint32_t mask = (1u << vocab) - 2;  // token zero is absent
  for (int regime = 0; regime < 6; ++regime) {
    Params p;
    p.temperature = 0.8f;
    p.logprobs = 5;
    if (regime == 1) p.top_k = 4;
    if (regime == 2) p.top_p = 0.9f;
    if (regime == 3) p.min_p = 0.1f;
    if (regime == 4) p.frequency_penalty = -0.0f;
    if (regime == 5) p.presence_penalty = 0.3f;
    for (uint64_t seed = 0; seed < 4; ++seed) {
      Rng a{seed, 3}, b{seed, 3};
      std::vector<int32_t> context{0, 0, 1, 3, 3, 16};
      for (int step = 0; step < 4; ++step) {
        auto adjusted = logits;
        apply_penalties(adjusted.data(), vocab, 0, p, count_context(context));
        dgpp::sample::apply_bias(adjusted.data(), vocab, 0, bias.data());
        dgpp::sample::apply_mask(adjusted.data(), vocab, 0, &mask, vocab);
        Params already_adjusted = p;
        already_adjusted.repetition_penalty = 1.0f;
        already_adjusted.frequency_penalty = already_adjusted.presence_penalty = 0.0f;
        const Result want =
            sample_reference_sharded(adjusted.data(), vocab, layout, already_adjusted, a, {});
        const Result got =
            sample_full_logits(logits.data(), vocab, layout, p, b, context, &mask, bias.data());
        require(results_identical(got, want) && a.counter == b.counter &&
                    b.counter == static_cast<uint64_t>(step + 4) && got.token != 0,
                "host sampling changed bias, mask, reports or RNG draws");
        context.push_back(got.token);
      }
    }
  }
}

DGPP_TEST(min_p_drops_tail_never_sampled) {
  // Three well-separated levels; min_p 0.5 must make the tail level
  // unsampleable regardless of draw.
  std::vector<float> v{4.0f, 3.9f, 3.5f, -6.0f, -6.5f, -7.0f};
  Params p;
  p.temperature = 1.0f;
  p.top_k = 6;
  p.min_p = 0.5f;
  for (int seed = 0; seed < 200; ++seed) {
    Rng rng{static_cast<uint64_t>(seed), 0};
    const Result r = select_from_sorted(sort_slice(v.data(), v.size(), 0), p, rng);
    require(r.token < 3, "min_p must make the tail unsampleable");
  }
}

DGPP_TEST(top_p_keeps_the_crossing_token) {
  // Crafted so only two tokens can survive top-p; assert the third is
  // never chosen and never appears in the logprobs list.
  std::vector<float> v{5.0f, 4.0f, 3.0f, 2.0f};
  Params p;
  p.temperature = 1.0f;
  p.top_k = 4;
  p.top_p = 0.9f;
  p.logprobs = 4;
  for (int seed = 0; seed < 200; ++seed) {
    Rng rng{static_cast<uint64_t>(seed), 0};
    const Result r = select_from_sorted(sort_slice(v.data(), v.size(), 0), p, rng);
    require(r.token < 3, "top-p tail must be unsampleable");
    for (const auto& lp : r.top_logprobs) {
      require(lp.first < 3, "top-p must drop the tail from logprobs");
    }
  }
}

DGPP_TEST(temperature_zero_equals_greedy) {
  FixtureRng fx;
  const std::vector<float> v = make_logits(31, fx);
  Params sampled;
  sampled.temperature = 0.0f;  // sampling spec with T=0 -> greedy branch
  Params greedy;
  greedy.temperature = 0.0f;
  Rng ra{7, 0}, rb{7, 0};
  const Result a = sample_reference(v.data(), v.size(), sampled, ra, {});
  const Result b = select_from_sorted(sort_slice(v.data(), v.size(), 0), greedy, rb);
  require(results_identical(a, b), "T=0 sample_reference must be greedy");
}

DGPP_TEST(empty_candidates_rejected) {
  Params p;
  Rng rng{1, 0};
  bool threw = false;
  try {
    select_from_sorted({}, p, rng);
  } catch (const std::exception&) {
    threw = true;
  }
  require(threw, "empty candidate list must throw");
}

DGPP_TEST(topk_probability_mass_uses_the_full_distribution_normalizer) {
  const std::vector<float> logits = {4.0f, 3.0f, 2.0f, 1.0f, -1.0f};
  const double lse = slice_logsumexp(logits.data(), logits.size());
  const std::vector<Candidate> sorted =
      sort_slice(logits.data(), logits.size(), 0);
  const std::vector<double> masses =
      topk_probability_masses(sorted, lse, {1, 2, 4, 8});
  require(masses.size() == 4, "mass count");
  double den = 0.0;
  for (float logit : logits) den += std::exp(static_cast<double>(logit) - 4.0);
  require(std::abs(masses[0] - 1.0 / den) < 1e-15, "top-1 mass");
  require(std::abs(masses[1] -
                   (1.0 + std::exp(-1.0)) / den) < 1e-15,
          "top-2 mass");
  require(masses[1] < masses[2], "prefix mass must increase");
  require(std::abs(masses[3] - 1.0) < 1e-15,
          "k beyond vocab means full mass");
}

DGPP_TEST(topk_probability_mass_rejects_bad_measurement_inputs) {
  const std::vector<Candidate> candidates{{0, 1.0f}, {1, 0.0f}};
  bool threw = false;
  try {
    (void)topk_probability_masses(candidates, 1.0, {2, 1});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "non-increasing k list must fail");
  threw = false;
  try {
    (void)slice_logsumexp(nullptr, 0);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "empty lse slice must fail");
}

DGPP_TEST(sampling_prefix_resolves_exact_top_p_nucleus) {
  // The first three tokens carry >95% of the full distribution, while the
  // first two do not. A three-candidate prefix therefore knows the exact
  // nucleus and must produce the same result/counter as the full oracle for
  // every draw.
  std::vector<float> logits{9.0f, 8.0f, 7.0f};
  logits.resize(100, -10.0f);
  const std::vector<Candidate> sorted =
      sort_slice(logits.data(), logits.size(), 0);
  const std::vector<Candidate> prefix(sorted.begin(), sorted.begin() + 3);
  Params p;
  p.temperature = 1.0f;
  p.top_p = 0.95f;
  p.logprobs = 3;
  const double lse = slice_logsumexp(logits.data(), logits.size(),
                                     p.temperature);
  for (uint64_t seed = 0; seed < 100; ++seed) {
    Rng expected_rng{seed, 7};
    Rng actual_rng = expected_rng;
    const Result expected = sample_reference(logits.data(), logits.size(), p,
                                             expected_rng, {});
    const auto actual = sample_from_prefix(prefix, logits.size(), lse, p,
                                           actual_rng);
    require(actual.resolved, "top-p crossing inside prefix must resolve");
    require(results_identical(actual.result, expected),
            "resolved prefix result differs from full oracle");
    require(actual_rng.counter == expected_rng.counter,
            "resolved prefix must consume exactly one oracle draw");
  }
}

DGPP_TEST(sampling_prefix_fallback_preserves_rng_draw) {
  // A flat 100-token distribution gives this 32-token prefix only 32% mass,
  // nowhere near the model default's 95% nucleus.
  const std::vector<float> logits(100, 0.0f);
  const std::vector<Candidate> sorted =
      sort_slice(logits.data(), logits.size(), 0);
  const std::vector<Candidate> prefix(sorted.begin(), sorted.begin() + 32);
  Params p;
  p.temperature = 1.0f;
  p.top_p = 0.95f;
  Rng rng{1234, 19};
  const auto decision = sample_from_prefix(
      prefix, logits.size(),
      slice_logsumexp(logits.data(), logits.size(), p.temperature), p, rng);
  require(!decision.resolved, "incomplete top-p prefix must request fallback");
  require(std::abs(decision.covered_mass - 0.32) < 1e-14,
          "fallback must report the exact transported mass");
  require(rng.counter == 19,
          "fallback must preserve the RNG draw for the full gather");
}

DGPP_TEST(sampling_prefix_finite_topk_is_complete_support) {
  FixtureRng fx;
  const std::vector<float> logits = make_logits(73, fx);
  const std::vector<int32_t> context{1, 1, 8, 55};
  Params p;
  p.temperature = 0.8f;
  p.top_k = 12;
  p.top_p = 0.87f;
  p.min_p = 0.03f;
  p.repetition_penalty = 1.2f;
  p.frequency_penalty = 0.1f;
  p.presence_penalty = 0.2f;

  std::vector<float> adjusted = logits;
  apply_penalties(adjusted.data(), adjusted.size(), 0, p,
                  count_context(context));
  const std::vector<Candidate> prefix =
      local_topk(adjusted.data(), adjusted.size(), 0, p.top_k);
  const double lse =
      slice_logsumexp(adjusted.data(), adjusted.size(), p.temperature);
  for (uint64_t seed = 0; seed < 32; ++seed) {
    Rng expected_rng{seed, 0};
    Rng actual_rng = expected_rng;
    const Result expected = sample_reference(logits.data(), logits.size(), p,
                                             expected_rng, context);
    const auto actual = sample_from_prefix(prefix, logits.size(), lse, p,
                                           actual_rng);
    require(actual.resolved, "complete finite top-k support must resolve");
    require(results_identical(actual.result, expected),
            "finite top-k prefix differs from full oracle");
    require(actual_rng.counter == expected_rng.counter,
            "finite top-k prefix counter differs from oracle");
  }
}

// The load-bearing property of the interface: a prefix that resolves is bitwise
// the complete list's result (which is the full-logit fallback), for every
// regime a request can ask for, at several widths, on flat and on peaked
// distributions with penalties across shards. A fallback leaves the counter
// where it was and the complete list then consumes that same draw.
DGPP_TEST(sampling_prefix_is_width_independent_in_every_regime) {
  struct Regime {
    const char* name;
    Params p;
  };
  std::vector<Regime> regimes;
  {
    Params p;  // GLM-5.3-Flash-FP8's generation_config.json
    p.temperature = 1.0f;
    p.top_p = 0.95f;
    p.logprobs = 2;
    regimes.push_back({"model default", p});
  }
  {
    Params p;
    p.temperature = 0.7f;
    p.top_p = 0.9f;
    p.logprobs = 2;
    regimes.push_back({"cool nucleus", p});
  }
  {
    Params p;
    p.temperature = 1.0f;
    p.top_p = 1.0f;
    p.logprobs = 2;
    regimes.push_back({"pure temperature", p});
  }
  {
    Params p;
    p.temperature = 0.8f;
    p.top_p = 0.9f;
    p.min_p = 0.05f;
    p.logprobs = 2;
    regimes.push_back({"min-p", p});
  }
  const std::vector<int32_t> context{3, 3, 40, 77, 128};
  FixtureRng fx;
  for (const Regime& regime : regimes) {
    int resolved = 0;
    int fallbacks = 0;
    for (int fixture = 0; fixture < 40; ++fixture) {
      const int n = 129 + fixture * 3;  // uneven three-way layouts
      const float spread = (fixture % 2) ? 1.5f : 8.0f;  // flat / peaked
      const std::vector<float> logits = make_logits(n, fx, spread);
      const std::vector<VocabSlice> layout = layout_for(n, 3);
      // What the sharded pipeline feeds the decision: penalties, the
      // canonical order, the fold normalizer.
      std::vector<float> adjusted = logits;
      apply_penalties(adjusted.data(), n, 0, regime.p, count_context(context));
      const std::vector<Candidate> sorted = sort_slice(adjusted.data(), n, 0);
      const double lse =
          sharded_scaled_logsumexp(adjusted.data(), layout, regime.p.temperature);
      for (int k : {4, 16, 64}) {
        const std::vector<Candidate> prefix(sorted.begin(),
                                            sorted.begin() + std::min(k, n));
        for (uint64_t seed = 0; seed < 8; ++seed) {
          const std::string where = std::string(regime.name) + " n=" +
                                    std::to_string(n) + " k=" +
                                    std::to_string(k) + " seed=" +
                                    std::to_string(seed);
          Rng oracle_rng{seed, 11};
          Rng rng = oracle_rng;
          const Result oracle = sample_reference_sharded(
              logits.data(), n, layout, regime.p, oracle_rng, context);
          require(oracle_rng.counter == 12,
                  "the sharded reference draws exactly once: " + where);
          const auto got = sample_from_prefix(prefix, n, lse, regime.p, rng);
          if (got.resolved) {
            ++resolved;
            require(results_identical(got.result, oracle),
                    "resolved prefix differs from the complete list: " + where);
            require(rng.counter == 12,
                    "resolved prefix must consume exactly one draw: " + where);
          } else {
            ++fallbacks;
            require(rng.counter == 11,
                    "fallback must leave the draw for the gather: " + where);
            Rng again{seed, 11};
            const auto full =
                sample_from_prefix(sorted, n, lse, regime.p, again);
            require(full.resolved && results_identical(full.result, oracle) &&
                        again.counter == 12,
                    "the complete list must resolve the fallback: " + where);
          }
        }
      }
    }
    require(resolved > 0 && fallbacks > 0,
            std::string("regime must exercise both outcomes: ") + regime.name +
                " resolved=" + std::to_string(resolved) +
                " fallbacks=" + std::to_string(fallbacks));
  }
}

// The case a complete-list shortcut got wrong: twenty equal logits at
// top_p=0.5 put the nucleus crossing on an exact tie, where a single fp32
// cumulative sum and the fp64 fold disagree (10 vs 11 survivors, and the
// token with them on half the draws). One arithmetic, one nucleus.
DGPP_TEST(sampling_prefix_exact_tie_crossing_is_width_independent) {
  const int n = 20;
  const std::vector<float> logits(n, 0.25f);
  const std::vector<VocabSlice> layout{{0, 10}, {10, 10}};
  Params p;
  p.temperature = 1.0f;
  p.top_p = 0.5f;
  p.logprobs = 11;  // reports min(11, nucleus): the nucleus size is pinned
  const std::vector<Candidate> sorted = sort_slice(logits.data(), n, 0);
  const std::vector<Candidate> prefix(sorted.begin(), sorted.begin() + 15);
  const double lse = sharded_scaled_logsumexp(logits.data(), layout, 1.0f);
  size_t nucleus = 0;
  for (uint64_t seed = 0; seed < 200; ++seed) {
    Rng oracle_rng{seed, 0};
    Rng rng{seed, 0};
    const Result oracle =
        sample_reference_sharded(logits.data(), n, layout, p, oracle_rng, {});
    const auto got = sample_from_prefix(prefix, n, lse, p, rng);
    require(got.resolved, "the crossing lies inside a 15-wide prefix");
    require(results_identical(got.result, oracle),
            "tie crossing: prefix and complete list disagree at seed " +
                std::to_string(seed));
    require(rng.counter == 1 && oracle_rng.counter == 1, "one draw each");
    nucleus = oracle.top_logprobs.size();
  }
  require(nucleus == 10 || nucleus == 11,
          "fixture sanity: the nucleus is the boundary set");
}

DGPP_TEST(sampling_prefix_pure_walk_needs_the_requested_logprobs) {
  // Pure temperature sampling can only report logprobs it holds; asking for
  // more than the prefix width is a fallback before any draw is consumed.
  const std::vector<float> logits{3.0f, 2.0f, 1.0f, 0.0f, -1.0f};
  const std::vector<VocabSlice> layout{{0, 5}};
  const std::vector<Candidate> sorted = sort_slice(logits.data(), 5, 0);
  const std::vector<Candidate> prefix(sorted.begin(), sorted.begin() + 2);
  const double lse = sharded_scaled_logsumexp(logits.data(), layout, 1.0f);
  Params p;
  p.temperature = 1.0f;
  p.top_p = 1.0f;
  p.logprobs = 3;
  Rng rng{5, 21};
  const auto got = sample_from_prefix(prefix, 5, lse, p, rng);
  require(!got.resolved && rng.counter == 21,
          "logprobs beyond the prefix must fall back without a draw");
  p.logprobs = 2;
  int inside = 0;
  for (uint64_t seed = 0; seed < 64; ++seed) {
    Rng oracle_rng{seed, 21};
    Rng walk{seed, 21};
    const Result oracle =
        sample_reference_sharded(logits.data(), 5, layout, p, oracle_rng, {});
    const auto d = sample_from_prefix(prefix, 5, lse, p, walk);
    if (d.resolved) {
      ++inside;
      require(results_identical(d.result, oracle) && walk.counter == 22,
              "pure walk inside the prefix must equal the complete list");
    } else {
      require(walk.counter == 21, "pure walk in the tail must not draw");
    }
  }
  require(inside > 0 && inside < 64, "the walk must land on both sides");
}

DGPP_TEST(sample_reference_sharded_rejects_bad_layouts) {
  FixtureRng fx;
  const std::vector<float> logits = make_logits(30, fx);
  Params p;
  p.temperature = 1.0f;
  const auto rejects = [&](const std::vector<VocabSlice>& layout) {
    Rng rng{1, 0};
    try {
      (void)sample_reference_sharded(logits.data(), 30, layout, p, rng, {});
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  require(rejects({}), "empty layout");
  require(rejects({{0, 10}, {11, 19}}), "a gap between slices");
  require(rejects({{0, 10}, {5, 25}}), "overlapping slices");
  require(rejects({{0, 10}, {10, 10}}), "a layout short of the vocabulary");
  require(rejects({{10, 20}, {0, 10}}), "slices out of rank order");
  require(!rejects({{0, 10}, {10, 20}}), "a covering layout is accepted");
  Rng greedy{1, 0};
  bool threw = false;
  try {
    Params g;
    g.temperature = 0.0f;
    (void)sample_reference_sharded(logits.data(), 30, {{0, 30}}, g, greedy, {});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "temperature <= 0 is not the sharded sampler's path");
}

DGPP_TEST(vocab_layout_matches_the_loader_rule) {
  // The loader's lm_head_slice: begin = V*r/W in integer arithmetic,
  // contiguous and covering for any V/W.
  for (const auto& [vocab, world] : {std::pair{154880, 4}, std::pair{7, 3},
                                    std::pair{600, 2}, std::pair{5, 5},
                                    std::pair{1000, 1}}) {
    const std::vector<VocabSlice> layout = vocab_layout(vocab, world);
    require(static_cast<int>(layout.size()) == world, "one slice per rank");
    int next = 0;
    for (int r = 0; r < world; ++r) {
      require(layout[static_cast<size_t>(r)].begin == next, "contiguous");
      require(layout[static_cast<size_t>(r)].begin ==
                  static_cast<int>(static_cast<int64_t>(vocab) * r / world),
              "the loader's begin");
      require(layout[static_cast<size_t>(r)].count > 0, "non-empty");
      next += layout[static_cast<size_t>(r)].count;
    }
    require(next == vocab, "covers the vocabulary");
  }
  bool threw = false;
  try {
    (void)vocab_layout(3, 4);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "more ranks than ids is not a layout");
}

// The fallback's decision over the complete logits — widening partial sorts,
// then the full sort — is the sharded reference bitwise, in every regime,
// with penalties, at vocabularies both below and above the first width.
DGPP_TEST(sample_full_logits_is_the_sharded_reference_bitwise) {
  FixtureRng fx;
  std::vector<Params> regimes;
  {
    Params p;
    p.temperature = 1.0f;
    p.top_p = 0.95f;
    p.logprobs = 2;
    regimes.push_back(p);
  }
  {
    Params p;
    p.temperature = 1.0f;
    p.top_p = 1.0f;  // the pure walk: the widening actually widens
    p.logprobs = 2;
    regimes.push_back(p);
  }
  {
    Params p;
    p.temperature = 0.6f;
    p.top_p = 0.9f;
    p.min_p = 0.02f;
    p.frequency_penalty = 0.3f;
    regimes.push_back(p);
  }
  {
    Params p;
    p.temperature = 1.3f;
    p.top_k = 50;
    p.top_p = 0.8f;
    regimes.push_back(p);
  }
  const std::vector<int32_t> context{1, 1, 2, 500, 1500, 2999};
  int cases = 0;
  for (const int vocab : {300, 3000}) {
    for (const int world : {1, 3}) {
      const std::vector<VocabSlice> layout = vocab_layout(vocab, world);
      for (int fixture = 0; fixture < 6; ++fixture) {
        const float spread = (fixture % 2) ? 1.0f : 10.0f;
        const std::vector<float> logits = make_logits(vocab, fx, spread);
        for (const Params& p : regimes) {
          for (uint64_t seed = 0; seed < 4; ++seed) {
            Rng a{seed, 3}, b{seed, 3};
            const Result oracle = sample_reference_sharded(
                logits.data(), vocab, layout, p, a, context);
            const Result got = sample_full_logits(logits.data(), vocab,
                                                  layout, p, b, context);
            require(results_identical(got, oracle) && a.counter == b.counter,
                    "sample_full_logits drifted from the sharded reference "
                    "at vocab " + std::to_string(vocab) + " world " +
                        std::to_string(world) + " seed " +
                        std::to_string(seed));
            ++cases;
          }
        }
      }
    }
  }
  require(cases == 2 * 2 * 6 * 4 * 4, "the sweep ran");
}

DGPP_TEST(validate_params_rejects_bad_specs_and_accepts_the_defaults) {
  const auto rejects = [](const Params& p) {
    try {
      validate_params(p);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  Params ok;
  require(!rejects(ok), "the neutral spec is valid");
  require(!rejects(dgpp::sample::greedy_params()), "greedy is valid");
  Params p = ok;
  p.temperature = -0.1f;
  require(rejects(p), "negative temperature");
  p = ok;
  p.temperature = std::numeric_limits<float>::infinity();
  require(rejects(p), "infinite temperature");
  p = ok;
  p.top_p = 0.0f;
  require(rejects(p), "top_p 0");
  p = ok;
  p.top_p = 1.5f;
  require(rejects(p), "top_p above 1");
  p = ok;
  p.min_p = 1.5f;
  require(rejects(p), "min_p above 1");
  p = ok;
  p.top_k = -1;
  require(rejects(p), "negative top_k");
  p = ok;
  p.repetition_penalty = 0.0f;
  require(rejects(p), "zero repetition penalty");
  p = ok;
  p.presence_penalty = std::nanf("");
  require(rejects(p), "NaN penalty");
}

// The speculative row-0 decision (accept the draft with its exact
// probability, else the residual) is width-independent exactly as the plain
// decision: a prefix that resolves equals the complete list bitwise, in
// every regime, for drafts inside and outside the nucleus; a fallback leaves
// the RNG untouched; a resolved step consumes exactly two draws on a reject
// and one on an accept (row 1's draw is the caller's).
DGPP_TEST(spec_accept_is_width_independent_in_every_regime) {
  using dgpp::sample::spec_accept_complete;
  using dgpp::sample::spec_accept_from_prefix;
  using dgpp::sample::SpecPrefixDecision;
  std::vector<Params> regimes;
  {
    Params p;
    p.temperature = 1.0f;
    p.top_p = 0.95f;
    regimes.push_back(p);
  }
  {
    Params p;
    p.temperature = 0.9f;
    p.top_p = 1.0f;  // pure: the draft's fold mass and the residual walk
    regimes.push_back(p);
  }
  {
    Params p;
    p.temperature = 0.7f;
    p.top_k = 30;
    p.min_p = 0.02f;
    regimes.push_back(p);
  }
  FixtureRng fx;
  int resolved = 0, fallbacks = 0, accepts = 0, rejects = 0;
  for (const Params& p : regimes) {
    for (int fixture = 0; fixture < 24; ++fixture) {
      const int n = 200 + fixture * 5;
      const float spread = (fixture % 2) ? 2.0f : 9.0f;
      std::vector<float> logits = make_logits(n, fx, spread);
      const std::vector<VocabSlice> layout = vocab_layout(n, 3);
      const std::vector<Candidate> sorted = sort_slice(logits.data(), n, 0);
      const double lse = sharded_scaled_logsumexp(logits.data(), layout, p.temperature);
      // Drafts: the argmax (often accepted), a mid-rank token, a tail token.
      const int32_t drafts[3] = {sorted[0].id, sorted[static_cast<size_t>(n / 4)].id,
                                 sorted[static_cast<size_t>(n - 1)].id};
      for (int di = 0; di < 3; ++di) {
        const int32_t draft = drafts[di];
        for (int k : {8, 32, 96}) {
          const std::vector<Candidate> prefix(sorted.begin(), sorted.begin() + k);
          for (uint64_t s = 0; s < 6; ++s) {
            // Distinct draws per fixture and draft (a fixed seed set would
            // reuse the same six accept tests everywhere).
            const uint64_t seed = s + 1000 * static_cast<uint64_t>(fixture) +
                                  77 * static_cast<uint64_t>(di) +
                                  9001 * static_cast<uint64_t>(k);
            Rng full_rng{seed, 5};
            Rng rng{seed, 5};
            const SpecPrefixDecision want =
                spec_accept_complete(logits.data(), n, lse, draft, p, full_rng);
            require(want.resolved, "the complete list resolves");
            require(full_rng.counter == (want.accepted ? 6 : 7),
                    "an accept draws once, a reject twice");
            const SpecPrefixDecision got =
                spec_accept_from_prefix(prefix, n, lse, draft, p, rng);
            if (got.resolved) {
              ++resolved;
              require(got.accepted == want.accepted &&
                          results_identical(got.result, want.result) &&
                          rng.counter == full_rng.counter,
                      "resolved prefix differs from the complete list (k " +
                          std::to_string(k) + ")");
            } else {
              ++fallbacks;
              require(rng.counter == 5, "a fallback consumes no draw");
            }
            if (want.accepted) ++accepts; else ++rejects;
          }
        }
      }
    }
  }
  require(resolved > 0 && fallbacks > 0 && accepts > 0 && rejects > 0,
          "the sweep must exercise every outcome (resolved " +
              std::to_string(resolved) + ", fallbacks " +
              std::to_string(fallbacks) + ", accepts " +
              std::to_string(accepts) + ", rejects " +
              std::to_string(rejects) + ")");
}

// Exact speculative sampling: whichever token the draft proposes, the
// row-0 outcome's distribution is the plain sampler's. Over many seeds the
// outcome frequencies must track the target distribution for a draft that
// is likely and one that is unlikely (a loose, deterministic statistical
// sanity check; the bitwise property is the gate above).
DGPP_TEST(spec_accept_marginal_tracks_the_target_distribution) {
  using dgpp::sample::spec_accept_complete;
  const std::vector<float> logits{2.0f, 1.5f, 1.0f, 0.0f, -3.0f, -3.0f};
  const int n = static_cast<int>(logits.size());
  const std::vector<VocabSlice> layout{{0, n}};
  Params p;
  p.temperature = 1.0f;
  p.top_p = 1.0f;  // the full distribution: every token reachable
  const double lse = sharded_scaled_logsumexp(logits.data(), layout, 1.0f);
  std::vector<double> target(logits.size());
  for (int i = 0; i < n; ++i)
    target[static_cast<size_t>(i)] = std::exp(logits[static_cast<size_t>(i)] - lse);
  constexpr int kTrials = 4000;
  for (const int32_t draft : {0, 3, 5}) {
    std::vector<int> counts(logits.size(), 0);
    int accepted = 0;
    for (uint64_t seed = 0; seed < kTrials; ++seed) {
      Rng rng{seed * 7919 + 1, 0};
      const auto d = spec_accept_complete(logits.data(), n, lse, draft, p, rng);
      counts[static_cast<size_t>(d.result.token)] += 1;
      accepted += d.accepted ? 1 : 0;
    }
    for (int i = 0; i < n; ++i) {
      const double freq = static_cast<double>(counts[static_cast<size_t>(i)]) / kTrials;
      require(std::abs(freq - target[static_cast<size_t>(i)]) < 0.03,
              "draft " + std::to_string(draft) + ": token " + std::to_string(i) +
                  " frequency " + std::to_string(freq) + " vs target " +
                  std::to_string(target[static_cast<size_t>(i)]));
    }
    const double accept_rate = static_cast<double>(accepted) / kTrials;
    require(std::abs(accept_rate - target[static_cast<size_t>(draft)]) < 0.03,
            "the acceptance rate is the draft's probability");
  }
}


// The greedy decision with logprobs: the canonical first candidate (the
// greedy pick's token) reported under the raw normalizer — exactly what
// select_from_sorted's greedy branch reports over a complete list — with
// the top-N alternatives inside the prefix.
DGPP_TEST(greedy_from_prefix_reports_the_raw_distribution) {
  using dgpp::sample::greedy_from_prefix;
  FixtureRng fx;
  const std::vector<float> logits = make_logits(300, fx, 6.0f);
  const std::vector<VocabSlice> layout = vocab_layout(300, 2);
  const std::vector<Candidate> sorted = sort_slice(logits.data(), 300, 0);
  const double lse = sharded_scaled_logsumexp(logits.data(), layout, 1.0f);
  Params greedy;
  greedy.temperature = 0.0f;
  greedy.logprobs = 5;
  Rng rng{1, 0};
  const Result complete = select_from_sorted(sorted, greedy, rng);
  const std::vector<Candidate> prefix(sorted.begin(), sorted.begin() + 16);
  const Result got = greedy_from_prefix(prefix, lse, 5);
  require(got.token == complete.token, "the greedy token");
  require(got.top_logprobs.size() == 5, "five alternatives");
  for (size_t i = 0; i < 5; ++i)
    require(got.top_logprobs[i].first == complete.top_logprobs[i].first,
            "the alternatives are the canonical prefix");
  // The raw normalizer: exp(logprob) sums to one over the vocabulary.
  double mass = 0.0;
  for (const Candidate& c : sorted)
    mass += std::exp(static_cast<double>(c.logit) - lse);
  require(std::abs(mass - 1.0) < 1e-9, "the fold normalizer is the raw one");
  require(std::abs(static_cast<double>(got.logprob) -
                   (static_cast<double>(sorted[0].logit) - lse)) < 1e-6,
          "the token's logprob under the raw distribution");
  require(rng.counter == 0, "no draw");
}

// The proposal-aware speculative rule: a draft DRAWN from Q,
// accepted with min(1, P/Q) and otherwise replaced by the (P - Q)+
// residual, still emits exactly P — and accepts at 1 - TV(P, Q), which is
// what a deterministic draft cannot reach (its ceiling is P(mode)).
DGPP_TEST(spec_accept_with_a_proposal_keeps_the_marginal_and_lifts_acceptance) {
  using dgpp::sample::Proposal;
  using dgpp::sample::spec_select_from_sorted;
  const std::vector<float> logits{2.0f, 1.5f, 1.0f, 0.2f, -0.4f, -3.0f};
  const int n = static_cast<int>(logits.size());
  Params p;
  p.temperature = 1.0f;
  p.top_p = 1.0f;
  const std::vector<Candidate> sorted = sort_slice(logits.data(), n, 0);
  // P, as the selector sees it.
  const std::vector<VocabSlice> layout{{0, n}};
  const double lse = sharded_scaled_logsumexp(logits.data(), layout, 1.0f);
  std::vector<double> target(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i)
    target[static_cast<size_t>(i)] = std::exp(logits[static_cast<size_t>(i)] - lse);
  // Q: a draft head that agrees on the shape but not the detail.
  const std::vector<float> draft_logits{1.6f, 1.7f, 0.9f, 0.1f, -0.2f, -2.0f};
  double qz = 0.0;
  for (int i = 0; i < n; ++i) qz += std::exp(draft_logits[static_cast<size_t>(i)]);
  Proposal q;
  std::vector<double> qmass(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    qmass[static_cast<size_t>(i)] = std::exp(draft_logits[static_cast<size_t>(i)]) / qz;
    q.mass.emplace_back(i, static_cast<float>(qmass[static_cast<size_t>(i)]));
  }
  double overlap = 0.0;  // 1 - TV(P, Q) = the acceptance the rule reaches
  double mode_mass = 0.0;
  for (int i = 0; i < n; ++i) {
    overlap += std::min(target[static_cast<size_t>(i)], qmass[static_cast<size_t>(i)]);
    mode_mass = std::max(mode_mass, target[static_cast<size_t>(i)]);
  }
  constexpr int kTrials = 60000;
  std::vector<int> counts(static_cast<size_t>(n), 0);
  int accepted = 0;
  for (uint64_t t = 0; t < kTrials; ++t) {
    // The draft, drawn from Q with its own stream.
    Rng qrng{t * 6364136223846793005ull + 1442695040888963407ull, 0};
    const double u = dgpp::sample::uniform01(qrng);
    double cum = 0.0;
    int32_t draft = n - 1;
    for (int i = 0; i < n; ++i) {
      cum += qmass[static_cast<size_t>(i)];
      if (cum > u) { draft = i; break; }
    }
    Rng rng{t * 7919 + 13, 0};
    const auto out = spec_select_from_sorted(sorted, p, draft, rng, &q);
    counts[static_cast<size_t>(out.result.token)] += 1;
    accepted += out.accepted ? 1 : 0;
    require(!out.accepted || out.result.token == draft, "an accept emits the draft");
  }
  for (int i = 0; i < n; ++i) {
    const double freq = static_cast<double>(counts[static_cast<size_t>(i)]) / kTrials;
    require(std::abs(freq - target[static_cast<size_t>(i)]) < 0.012,
            "token " + std::to_string(i) + " frequency " + std::to_string(freq) +
                " vs target " + std::to_string(target[static_cast<size_t>(i)]));
  }
  const double rate = static_cast<double>(accepted) / kTrials;
  require(std::abs(rate - overlap) < 0.012,
          "the accept rate " + std::to_string(rate) + " is the overlap " +
              std::to_string(overlap));
  require(overlap > mode_mass + 0.05,
          "the proposal must beat a deterministic draft's ceiling (overlap " +
              std::to_string(overlap) + " vs mode " + std::to_string(mode_mass) + ")");
}

// The pure regime (no top-k / top-p / min-p) takes the ratio rule too
// (2026-10-05): over a complete list the marginal is P and the accept rate
// the overlap; an incomplete prefix that hides an id the proposal gives
// mass to cannot price the residual and falls back without a draw.
DGPP_TEST(spec_accept_pure_regime_uses_the_proposal) {
  using dgpp::sample::Proposal;
  using dgpp::sample::spec_accept_from_prefix;
  const std::vector<float> logits{2.0f, 1.5f, 1.0f, 0.2f, -0.4f, -3.0f};
  const int n = static_cast<int>(logits.size());
  Params p;  // temperature 1, no truncation: the pure regime
  const std::vector<Candidate> sorted = sort_slice(logits.data(), n, 0);
  const std::vector<VocabSlice> layout{{0, n}};
  const double lse = sharded_scaled_logsumexp(logits.data(), layout, 1.0f);
  std::vector<double> target(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i)
    target[static_cast<size_t>(i)] = std::exp(logits[static_cast<size_t>(i)] - lse);
  const std::vector<float> draft_logits{1.6f, 1.7f, 0.9f, 0.1f, -0.2f, -2.0f};
  double qz = 0.0;
  for (int i = 0; i < n; ++i) qz += std::exp(draft_logits[static_cast<size_t>(i)]);
  Proposal q;
  std::vector<double> qmass(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    qmass[static_cast<size_t>(i)] = std::exp(draft_logits[static_cast<size_t>(i)]) / qz;
    q.mass.emplace_back(i, static_cast<float>(qmass[static_cast<size_t>(i)]));
  }
  double overlap = 0.0, mode_mass = 0.0;
  for (int i = 0; i < n; ++i) {
    overlap += std::min(target[static_cast<size_t>(i)], qmass[static_cast<size_t>(i)]);
    mode_mass = std::max(mode_mass, target[static_cast<size_t>(i)]);
  }
  constexpr int kTrials = 60000;
  std::vector<int> counts(static_cast<size_t>(n), 0);
  int accepted = 0;
  for (uint64_t t = 0; t < kTrials; ++t) {
    Rng qrng{t * 6364136223846793005ull + 1442695040888963407ull, 0};
    const double u = dgpp::sample::uniform01(qrng);
    double cum = 0.0;
    int32_t draft = n - 1;
    for (int i = 0; i < n; ++i) {
      cum += qmass[static_cast<size_t>(i)];
      if (cum > u) { draft = i; break; }
    }
    Rng rng{t * 7919 + 13, 0};
    const auto d = spec_accept_from_prefix(sorted, n, lse, draft, p, rng, false, &q);
    require(d.resolved, "a complete list always decides");
    counts[static_cast<size_t>(d.result.token)] += 1;
    accepted += d.accepted ? 1 : 0;
    require(!d.accepted || d.result.token == draft, "an accept emits the draft");
  }
  for (int i = 0; i < n; ++i) {
    const double freq = static_cast<double>(counts[static_cast<size_t>(i)]) / kTrials;
    require(std::abs(freq - target[static_cast<size_t>(i)]) < 0.012,
            "pure regime: token " + std::to_string(i) + " frequency " + std::to_string(freq) +
                " vs target " + std::to_string(target[static_cast<size_t>(i)]));
  }
  const double rate = static_cast<double>(accepted) / kTrials;
  require(std::abs(rate - overlap) < 0.012,
          "pure regime: the accept rate " + std::to_string(rate) + " is the overlap " +
              std::to_string(overlap));
  require(overlap > mode_mass + 0.05, "the proposal must beat the deterministic ceiling");
  // An incomplete prefix (the top three of six) with the proposal's mass
  // partly outside it: the overlap is unseen, so the row keeps the
  // deterministic rule — the same decision and draws as without a proposal.
  const std::vector<Candidate> prefix(sorted.begin(), sorted.begin() + 3);
  for (uint64_t t = 0; t < 50; ++t) {
    Rng with{99 + t, 5}, without{99 + t, 5};
    const auto d = spec_accept_from_prefix(prefix, n, lse, prefix[0].id, p, with, false, &q);
    const auto e = spec_accept_from_prefix(prefix, n, lse, prefix[0].id, p, without, false, nullptr);
    require(d.resolved == e.resolved && with.counter == without.counter &&
                (!d.resolved || (d.accepted == e.accepted && d.result.token == e.result.token)),
            "a hidden overlap keeps the deterministic rule's decision and draws");
  }
  // The same prefix with a proposal confined to it decides (the tail holds
  // no proposal mass): the residual total is exact.
  Proposal inside;
  for (int i = 0; i < 3; ++i) inside.mass.emplace_back(prefix[static_cast<size_t>(i)].id, i == 0 ? 0.6f : 0.2f);
  int decided = 0;
  for (uint64_t t = 0; t < 200; ++t) {
    Rng r{t + 1, 0};
    const auto e = spec_accept_from_prefix(prefix, n, lse, prefix[0].id, p, r, false, &inside);
    decided += e.resolved ? 1 : 0;
    require(!e.resolved || e.accepted || e.result.token != prefix[0].id || target[0] > 0.6,
            "a rejected draft re-emerges only with residual mass");
  }
  require(decided > 100, "most draws decide inside the prefix (" + std::to_string(decided) + ")");
}

// Block verification on the paper's two-token example (Sun et al. 2024,
// Section 2): M_b = (1/3, 2/3), M_s = (2/3, 1/3), gamma = 2. Token
// verification accepts 10/9 drafts in expectation, block verification
// 11/9; both emit the first token with M_b's marginal.
DGPP_TEST(block_verify_matches_the_papers_two_token_example) {
  using dgpp::sample::BlockDecision;
  using dgpp::sample::Proposal;
  using dgpp::sample::block_verify_from_prefixes;
  const std::vector<float> target_logits{std::log(1.0f / 3.0f), std::log(2.0f / 3.0f)};  // A, B
  const double qa = 2.0 / 3.0;  // M_s(A)
  Params p;  // temperature 1, the pure regime (a complete two-token list)
  const std::vector<Candidate> row = sort_slice(target_logits.data(), 2, 0);  // B first (canonical)
  const std::vector<VocabSlice> layout{{0, 2}};
  const double lse = sharded_scaled_logsumexp(target_logits.data(), layout, 1.0f);
  Proposal q;
  q.mass.emplace_back(0, static_cast<float>(qa));
  q.mass.emplace_back(1, static_cast<float>(1.0 - qa));
  constexpr int kTrials = 60000;
  double tau_sum = 0.0;
  int first_a = 0, unresolved = 0;
  for (uint64_t t = 0; t < kTrials; ++t) {
    Rng qrng{t * 6364136223846793005ull + 97, 0};
    std::vector<int32_t> drafts(2);
    for (int i = 0; i < 2; ++i) {
      const double u = dgpp::sample::uniform01(qrng);
      ++qrng.counter;
      drafts[static_cast<size_t>(i)] = u < qa ? 0 : 1;
    }
    Rng rng{t * 7919 + 3, 0};
    const BlockDecision d = block_verify_from_prefixes({row, row, row}, 2, {lse, lse, lse}, drafts,
                                                       {&q, &q}, p, rng);
    require(d.decidable, "a complete list always decides");
    if (!d.resolved) { ++unresolved; continue; }
    tau_sum += d.tau;
    require(static_cast<int>(d.winners.size()) == d.tau + 1, "the winners are the drafts and the next token");
    first_a += d.winners[0] == 0 ? 1 : 0;
    require(rng.counter == 3, "two etas and one final draw");
  }
  require(unresolved == 0, "no fallback on a complete list");
  const double mean_tau = tau_sum / kTrials;
  require(std::abs(mean_tau - 11.0 / 9.0) < 0.02,
          "block verification accepts 11/9 drafts in expectation, got " + std::to_string(mean_tau));
  require(std::abs(static_cast<double>(first_a) / kTrials - 1.0 / 3.0) < 0.012,
          "the first emitted token keeps the target's marginal");
}

// A proposal that is absent, or that never proposed this draft, leaves the
// rule exactly where it was: the same decision, the same draws.
DGPP_TEST(spec_accept_without_a_proposal_is_the_deterministic_rule) {
  using dgpp::sample::Proposal;
  using dgpp::sample::spec_select_from_sorted;
  FixtureRng fx;
  const int n = 240;
  const std::vector<float> logits = make_logits(n, fx, 3.0f);
  const std::vector<Candidate> sorted = sort_slice(logits.data(), n, 0);
  Params p;
  p.temperature = 0.8f;
  p.top_p = 0.95f;
  Proposal other;  // a proposal that does not carry the draft
  other.mass.emplace_back(sorted[static_cast<size_t>(n - 1)].id, 1.0f);
  for (uint64_t seed = 0; seed < 64; ++seed) {
    const int32_t draft = sorted[static_cast<size_t>(seed % 5)].id;
    Rng a{seed + 1, 3}, b{seed + 1, 3}, c{seed + 1, 3};
    const auto want = spec_select_from_sorted(sorted, p, draft, a);
    const auto null_q = spec_select_from_sorted(sorted, p, draft, b, nullptr);
    const auto absent = spec_select_from_sorted(sorted, p, draft, c, &other);
    require(want.accepted == null_q.accepted && want.accepted == absent.accepted,
            "the same accept decision");
    require(want.result.token == null_q.result.token &&
                want.result.token == absent.result.token,
            "the same token");
    require(a.counter == b.counter && a.counter == c.counter, "the same draws");
  }
}
