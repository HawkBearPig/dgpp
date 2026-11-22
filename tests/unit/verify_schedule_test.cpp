// The confidence-scheduled verify-depth policy (engine/verify_schedule.hpp): the Dinkelbach-optimal
// depth for aggregate decode throughput. Exactness is not at stake (every k
// commits the same greedy prefix); these pin the economics.
//
// The policy verifies draft i while its prefix-survival S_i exceeds the tokens
// that one row of verify time is worth elsewhere (lambda * row_ms), and stops
// at the first draft that falls below -- S_i is monotone, so kept drafts are a
// prefix. Larger lambda (time more valuable) => shallower; a larger fixed base
// lowers the reservation lambda => deeper (amortization).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "engine/verify_schedule.hpp"

namespace {
void require(bool c, const char* w) {
  if (!c) throw std::runtime_error(w);
}
float logit(double p) { return static_cast<float>(std::log(p / (1.0 - p))); }
}  // namespace

DGPP_TEST(verify_schedule_hot_stream_verifies_whole_block) {
  // Every position ~99% likely, lambda at the plain-decode reservation rate
  // (profiled curve: base ~20 ms, ~9 ms/row): the whole block clears.
  std::vector<float> c(5, logit(0.99));
  const float lam = dgpp::verify_reservation_lambda(20.0f, 9.0f);
  require(dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, lam) == 5,
          "hot draft verifies all five");
}

DGPP_TEST(verify_schedule_cold_stream_verifies_nothing) {
  // Drafts almost certain to be rejected: verify none, decode the target token.
  std::vector<float> c(5, logit(0.02));
  const float lam = dgpp::verify_reservation_lambda(20.0f, 9.0f);
  require(dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, lam) == 0,
          "a cold stream verifies zero drafts");
}

DGPP_TEST(verify_schedule_stops_at_the_survival_cliff) {
  // Likely, then a cliff: verify the surviving prefix only.
  const std::vector<float> c = {logit(0.9), logit(0.8), logit(0.1),
                                logit(0.9), logit(0.9)};
  const float lam = dgpp::verify_reservation_lambda(20.0f, 9.0f);
  // S = .9, .72, .072, ... threshold = lam*9 = 9/29 = .310. Prefix over .310 is
  // {.9,.72}; .072 falls below and stops even though a later logit is high.
  require(dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, lam) == 2,
          "stops at the first sub-threshold survival, ignoring later spikes");
}

DGPP_TEST(verify_schedule_accept_scale_shortens_a_sampled_slot) {
  // A sampled slot with argmax drafts accepts at a fraction of the head's
  // rate per position: the scaled survival falls under the threshold sooner
  // (never later), scale 1 is the greedy rule, and the batch rule takes a
  // scale per slot.
  std::vector<float> c(5, logit(0.8));
  const float lam = dgpp::verify_reservation_lambda(20.0f, 9.0f);  // threshold .310
  // S = .8, .64, .512, .410, .328: all five clear at scale 1.
  require(dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, lam) == 5, "the greedy rule clears the block");
  require(dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, lam, 1.0) == 5, "scale 1 is the greedy rule");
  // At 0.9: p = .72 -> S = .72, .518, .373, .269: three.
  require(dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, lam, 0.9) == 3, "a scaled slot stops sooner");
  for (double s = 0.5; s < 1.0; s += 0.1)
    require(dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, lam, s) <=
                dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, lam, s + 0.1),
            "the depth is non-decreasing in the scale");
  const float* confs[2] = {c.data(), c.data()};
  const double ones[2] = {1.0, 1.0}, mixed[2] = {1.0, 0.9};
  require(dgpp::scheduled_verify_depth_batch(confs, 2, 5, 9.0f, lam, ones) ==
              dgpp::scheduled_verify_depth_batch(confs, 2, 5, 9.0f, lam),
          "unit scales are the unscaled batch rule");
  // Mean survival of (1.0, 0.9): (.8+.72)/2 .. -> .76, .579, .442, .340, .261: four.
  require(dgpp::scheduled_verify_depth_batch(confs, 2, 5, 9.0f, lam, mixed) == 4,
          "the batch rule averages the slots' scaled survivals");
}

DGPP_TEST(verify_schedule_larger_lambda_verifies_shallower) {
  // Monotone in lambda: the more valuable decode time is, the fewer drafts.
  std::vector<float> c(5, logit(0.85));
  const int a = dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, 0.02f);
  const int b = dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, 0.05f);
  const int d = dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, 0.09f);
  require(a >= b && b >= d, "depth is non-increasing in lambda");
  require(a == 5, "cheap time verifies the whole block");
  require(d < 5, "dear time verifies less than the whole block");
}

DGPP_TEST(verify_schedule_larger_base_verifies_deeper) {
  // Amortization via the reservation rate: a larger fixed base lowers lambda,
  // which lowers the threshold, which verifies deeper.
  std::vector<float> c(5, logit(0.6));
  const float lo_base = dgpp::verify_reservation_lambda(5.0f, 9.0f);
  const float hi_base = dgpp::verify_reservation_lambda(200.0f, 9.0f);
  const int shallow = dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, lo_base);
  const int deep = dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, hi_base);
  require(deep >= shallow, "a larger fixed base never verifies shallower");
}

DGPP_TEST(verify_schedule_batch_takes_the_mean_survival) {
  // Two slots at threshold 0.3 (lambda 0.1/ms x 3 ms): alone, the hot slot
  // verifies 2 (S 0.9, 0.45 > 0.3; 0.09 stops) and the cold one 0; the
  // batch's mean survival (0.5, 0.25) keeps one position: the second
  // position's row would cost two rows of time for 0.5 expected tokens.
  const std::vector<float> hot = {logit(0.9), logit(0.5), logit(0.2)};
  const std::vector<float> cold = {logit(0.1), logit(0.5), logit(0.5)};
  const float* confs[2] = {hot.data(), cold.data()};
  require(dgpp::scheduled_verify_depth(hot.data(), 3, 3.0f, 0.1f) == 2, "the hot slot alone verifies 2");
  require(dgpp::scheduled_verify_depth(cold.data(), 3, 3.0f, 0.1f) == 0, "the cold slot alone verifies 0");
  require(dgpp::scheduled_verify_depth_batch(confs, 2, 3, 3.0f, 0.1f) == 1, "the batch verifies 1");
  // One slot: the batch rule is the scalar rule.
  require(dgpp::scheduled_verify_depth_batch(confs, 1, 3, 3.0f, 0.1f) == 2, "a batch of one is the scalar rule");
  // Two hot slots: the whole block of 2 as alone (the mean is each slot's S).
  const float* hots[2] = {hot.data(), hot.data()};
  require(dgpp::scheduled_verify_depth_batch(hots, 2, 3, 3.0f, 0.1f) == 2, "twin hot slots verify as one");
  require(dgpp::scheduled_verify_depth_batch(confs, 0, 3, 3.0f, 0.1f) == 0, "no slots: nothing");
}

DGPP_TEST(verify_schedule_adaptive_lambda_converges_to_the_modeled_throughput) {
  // Six slots at depth 4 (30 rows, base 28 + 30 x 8 = 268 ms) committing 19
  // tokens a step: the EWMA converges to 19 / 268 = 0.0709 from the
  // one-stream constant 0.045; one slot at 5 rows committing 3.3 lands at
  // 3.3 / 68 = 0.0485. The update is a fixed sequence of doubles on
  // replicated inputs (the ranks agree by construction).
  double lam = 0.045;
  for (int i = 0; i < 2000; ++i) lam = dgpp::verify_lambda_update(lam, 19, 30, 28.f, 8.f, 1.0 / 64.0);
  require(std::fabs(lam - 19.0 / 268.0) < 1e-6, "six slots: the modeled six-stream rate");
  double lam1 = 0.045;
  for (int i = 0; i < 2000; ++i) lam1 = dgpp::verify_lambda_update(lam1, (i % 10 < 3) ? 4 : 3, 5, 28.f, 8.f, 1.0 / 64.0);
  require(std::fabs(lam1 - 3.3 / 68.0) < 2e-3, "one slot: the modeled one-stream rate");
  require(dgpp::verify_lambda_update(0.05, 3, 0, 0.f, 0.f, 0.5) == 0.05, "a degenerate time leaves lambda alone");
  double a = 0.045, b = 0.045;
  for (int i = 0; i < 100; ++i) { a = dgpp::verify_lambda_update(a, 7, 12, 28.f, 8.f, 1.0 / 64.0); b = dgpp::verify_lambda_update(b, 7, 12, 28.f, 8.f, 1.0 / 64.0); }
  require(a == b, "the same inputs give the same doubles");
}

// ---- The option set and the option a step replays ----
// engine.mtp_schedule_options and engine.mtp_schedule_max_live decide the
// set; a step's option follows the policy's depth and the drafts its slot
// holds. These pin that a step never verifies a draft its slot's last replay
// did not write, on every set the engine can build.

namespace {
using dgpp::VerifyDepthOptions;

int option_of(const VerifyDepthOptions& s, int k, int available) {
  return dgpp::verify_depth_option(s.depths, s.first_block, s.first_policy, k, available);
}
int drafts_of(const VerifyDepthOptions& s, int option, int depth) {
  return dgpp::verify_depth_option_drafts(s.depths, s.first_block, option, depth);
}
// The policy's own choice: its lowest option covering k, the full block when
// none does.
int policy_choice(const VerifyDepthOptions& s, int k) {
  for (int i = s.first_policy; i < static_cast<int>(s.depths.size()); ++i)
    if (s.depths[static_cast<size_t>(i)] >= k) return i;
  return static_cast<int>(s.depths.size()) - 1;
}
std::string set_text(int depth, int min_depth, int options, bool narrow) {
  return "depth " + std::to_string(depth) + " min " + std::to_string(min_depth) + " options " +
         std::to_string(options) + (narrow ? " with the live limit" : "");
}
void require_at(bool c, const std::string& where, const char* w) {
  if (!c) throw std::runtime_error(where + ": " + w);
}
// Every set the engine builds: a block of 2 to 7 drafts, a minimum under it,
// one option up to every depth of the span, with and without the live limit.
template <class F>
void for_every_set(F&& f) {
  for (int depth = 2; depth <= 7; ++depth)
    for (int min_depth = 1; min_depth < depth; ++min_depth)
      for (int options = 1; options <= depth - min_depth + 1; ++options)
        for (const bool narrow : {false, true})
          f(depth, min_depth, options, narrow, dgpp::verify_depth_option_set(depth, min_depth, options, narrow));
}
uint64_t next_random(uint64_t* s) {
  *s ^= *s << 13;
  *s ^= *s >> 7;
  *s ^= *s << 17;
  return *s;
}
}  // namespace

DGPP_TEST(verify_schedule_options_spread_evenly_and_keep_the_full_block) {
  const auto depths = [](int depth, int min_depth, int options, bool narrow) {
    return dgpp::verify_depth_option_set(depth, min_depth, options, narrow).depths;
  };
  require(depths(4, 1, 3, false) == std::vector<int>{1, 3, 4}, "three options on a 4-draft block are 1, 3 and 4");
  require(depths(2, 1, 2, false) == std::vector<int>{1, 2}, "two options on a 2-draft block are both depths");
  require(depths(2, 1, 1, false) == std::vector<int>{2}, "one option is the full block");
  require(depths(7, 1, 4, false) == std::vector<int>{1, 3, 5, 7}, "four options on a 7-draft block");
  require(depths(5, 3, 2, false) == std::vector<int>{3, 5}, "the spread starts at the minimum depth");
  // The live limit puts the narrow option first. The policy's options follow
  // it, and a set whose policy has a minimum-depth option needs no other.
  const VerifyDepthOptions two = dgpp::verify_depth_option_set(2, 1, 2, true);
  require(two.depths == std::vector<int>{1, 1, 2} && two.first_block == 1 && two.first_policy == 1 && two.narrow() &&
              !two.refill(),
          "the narrow option stands before the policy's two");
  const VerifyDepthOptions four = dgpp::verify_depth_option_set(4, 1, 3, true);
  require(four.depths == std::vector<int>{1, 1, 3, 4} && four.first_policy == 1 && !four.refill(),
          "the narrow option stands before the policy's three");
  // A single option under the limit: the refill option between the two.
  const VerifyDepthOptions one = dgpp::verify_depth_option_set(2, 1, 1, true);
  require(one.depths == std::vector<int>{1, 1, 2} && one.first_block == 1 && one.first_policy == 2 && one.refill(),
          "a single option under the limit carries the refill option");
  const VerifyDepthOptions deep = dgpp::verify_depth_option_set(4, 2, 1, true);
  require(deep.depths == std::vector<int>{2, 2, 4} && deep.first_policy == 2 && deep.refill(),
          "the refill option verifies the minimum depth");
  const VerifyDepthOptions plain = dgpp::verify_depth_option_set(2, 1, 1, false);
  require(plain.first_block == 0 && plain.first_policy == 0 && !plain.narrow() && !plain.refill(),
          "no limit: the policy's options alone");
}

DGPP_TEST(verify_schedule_single_option_refills_the_block_after_a_narrow_step) {
  // Two drafts, a minimum of one, a single option and a live limit: two
  // requests share narrow steps that draft one token each. When one retires
  // the other holds one draft. Its first step alone verifies that one and
  // drafts the block, and the full depth follows.
  const int depth = 2;
  const VerifyDepthOptions s = dgpp::verify_depth_option_set(depth, 1, 1, true);
  for (int k = 0; k <= depth + 1; ++k)
    require(option_of(s, k, depth) == 2, "a slot holding the block verifies it whatever the policy asks");
  const int after_narrow = drafts_of(s, 0, depth);
  require(after_narrow == 1, "the narrow option drafts the minimum depth");
  for (int k = 0; k <= depth + 1; ++k) {
    const int first_alone = option_of(s, k, after_narrow);
    require(first_alone == 1 && s.depths[1] == 1, "the first step alone verifies the one draft it holds");
    require(drafts_of(s, first_alone, depth) == depth, "and drafts the whole block");
  }
  require(option_of(s, depth, drafts_of(s, 1, depth)) == 2, "the step after it verifies the full depth");
}

DGPP_TEST(verify_schedule_option_stays_within_the_drafts_a_slot_holds) {
  int sets = 0, choices = 0;
  for_every_set([&](int depth, int min_depth, int options, bool narrow, const VerifyDepthOptions& s) {
    const std::string where = set_text(depth, min_depth, options, narrow);
    const int count = static_cast<int>(s.depths.size());
    ++sets;
    // The set's shape.
    require_at(s.depths.back() == depth, where, "the last option is the full block");
    require_at(s.first_block == (narrow ? 1 : 0) && s.first_policy >= s.first_block && s.first_policy < count, where,
               "the narrow option, then the block-drafting ones");
    require_at(s.refill() == (narrow && options == 1), where,
               "the refill option stands exactly beside a single option under the limit");
    require_at(s.first_policy - s.first_block <= 1, where, "at most one refill option");
    if (narrow)
      require_at(s.depths[0] == min_depth && s.depths[static_cast<size_t>(s.first_block)] == min_depth, where,
                 "a block-drafting option at the narrow option's depth");
    for (int i = s.first_policy; i < count; ++i) {
      require_at(s.depths[static_cast<size_t>(i)] >= min_depth && s.depths[static_cast<size_t>(i)] <= depth, where,
                 "a policy option outside [min_depth, depth]");
      if (i > s.first_policy)
        require_at(s.depths[static_cast<size_t>(i)] > s.depths[static_cast<size_t>(i - 1)], where,
                   "the policy's options ascend");
    }
    require_at(count - s.first_policy <= options, where, "more policy options than asked for");
    if (options > 1)
      require_at(s.depths[static_cast<size_t>(s.first_policy)] == min_depth, where,
                 "two or more options start at the minimum depth");
    // Every choice. A slot holds the block, or what a narrow step drafted
    // (the minimum depth). The depths between are checked under the limit
    // too: the rule holds for any count the minimum-depth options cover.
    for (int available = narrow ? min_depth : depth; available <= depth; ++available)
      for (int k = -1; k <= depth + 2; ++k) {
        ++choices;
        const int option = option_of(s, k, available);
        require_at(option >= s.first_block && option < count, where, "a policy depth took the narrow option");
        require_at(s.depths[static_cast<size_t>(option)] <= available, where,
                   "a step verifies more drafts than its slot holds");
        const int policy = policy_choice(s, k);
        if (s.depths[static_cast<size_t>(policy)] <= available) {
          require_at(option == policy, where, "the policy's own option fits the drafts available and was not taken");
        } else {
          require_at(option < policy, where, "an option the drafts available do not cover");
          require_at(s.depths[static_cast<size_t>(option + 1)] > available, where,
                     "a deeper block-drafting option fits the drafts available");
        }
        require_at(drafts_of(s, option, depth) == depth, where, "a step under the limit drafts the whole block");
        // A set whose policy has an option at the minimum depth, and any set
        // without the limit, rounds up exactly as it did before the drafts
        // available bounded the chosen option. Only the states a slot
        // reaches are compared.
        if (!s.refill() && (available == depth || available == min_depth)) {
          const int clamped = std::min(std::max(k, min_depth), depth);
          require_at(option == policy_choice(s, std::min(clamped, available)), where,
                     "a set with a minimum-depth policy option chooses differently");
        }
      }
  });
  require(sets == 2 * (2 + 5 + 9 + 14 + 20 + 27), "every option set was built");
  require(choices > 2000, "every choice was checked");
}

DGPP_TEST(verify_schedule_option_walk_returns_to_the_policy_after_a_narrow_step) {
  // Random runs of steps over every set: a step past the live limit takes
  // the narrow option, any other the option the rule gives for a random
  // policy depth. No step verifies past what the step before it drafted, a
  // slot is back on the policy's own options one step after it leaves the
  // limit, and the refill option is taken only by that one step.
  const char* env = std::getenv("DGPP_TEST_SCHEDULE_WALK_RUNS");
  const int runs = env != nullptr ? std::max(1, std::atoi(env)) : 64;
  uint64_t rng = 0x9E3779B97F4A7C15ull;
  uint64_t steps = 0, narrow_steps = 0, refill_steps = 0;
  for_every_set([&](int depth, int min_depth, int options, bool narrow, const VerifyDepthOptions& s) {
    const std::string where = set_text(depth, min_depth, options, narrow);
    for (int run = 0; run < runs; ++run) {
      int available = depth;  // a request opens holding the block
      bool last_narrow = false;
      for (int step = 0; step < 48; ++step) {
        const bool gated = narrow && next_random(&rng) % 3 == 0;
        const int k = static_cast<int>(next_random(&rng) % static_cast<uint64_t>(depth + 2));
        const int option = gated ? 0 : option_of(s, std::min(k, available), available);
        ++steps;
        require_at(s.depths[static_cast<size_t>(option)] <= available, where,
                   "a step verifies more drafts than the step before it drafted");
        if (gated) {
          ++narrow_steps;
        } else {
          require_at(option >= s.first_block, where, "a step under the limit took the narrow option");
          const bool refill = s.refill() && option == s.first_block;
          if (refill) ++refill_steps;
          require_at(refill == (s.refill() && last_narrow), where,
                     "the refill option belongs to the step after a narrow one, and to no other");
          if (!last_narrow)
            require_at(option == policy_choice(s, k), where, "a slot holding the block takes the policy's own option");
        }
        available = drafts_of(s, option, depth);
        require_at(available == (gated ? min_depth : depth), where, "the drafts a step leaves");
        last_narrow = gated;
      }
    }
  });
  require(steps > 0 && narrow_steps > 0 && refill_steps > 0, "the walk reached the narrow and the refill options");
}

DGPP_TEST(verify_schedule_degenerate_block) {
  const float lam = dgpp::verify_reservation_lambda(20.0f, 9.0f);
  require(dgpp::scheduled_verify_depth(nullptr, 0, 9.0f, lam) == 0,
          "empty block verifies nothing");
  std::vector<float> c(1, logit(0.99));
  require(dgpp::scheduled_verify_depth(c.data(), 1, 9.0f, lam) == 1,
          "a single confident draft is worth verifying");
}
