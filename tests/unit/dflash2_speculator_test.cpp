// The DFlash2 speculator's host-side contract: the depth-cap policy, the
// commit_verify/redraft split (the batched redraft's halves) and the
// per-position counters — all against a scripted host model, no device.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "engine/decode_outputs.hpp"
#include "engine/speculative.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// Scripted winners per verify call; empty script row means accept-all.
struct FakeModel {
  bool dflash2_enabled() const { return true; }
  std::vector<std::vector<int32_t>> winner_script;
  size_t calls = 0;
  struct Rollback {
    int accepted = 0, rows = 0, base = 0;
  };
  std::vector<Rollback> rollbacks;
  std::vector<int32_t> next_drafts = {11, 12, 13, 14, 15, 16, 17};
  int draft_calls = 0;

  bool dflash2_draft(int, int64_t, std::vector<int32_t>* d) {
    ++draft_calls;
    *d = next_drafts;
    return true;
  }
  dgpp::DecodeOutputs session_verify(int, const std::vector<int64_t>& fed) {
    const int T = static_cast<int>(fed.size());
    std::vector<int32_t> winners;
    if (calls < winner_script.size() && !winner_script[calls].empty()) {
      winners = winner_script[calls];
      require(static_cast<int>(winners.size()) == T, "script/fed shape");
    } else {
      // Accept-all: row r's winner is fed[r], except the last row which
      // picks 99 (the next bonus).
      for (int r = 0; r < T; ++r)
        winners.push_back(r + 1 < T ? static_cast<int32_t>(fed[static_cast<size_t>(r + 1)]) : 99);
    }
    ++calls;
    dgpp::DecodeOutputs out;
    out.lm_vocab_begin = 0;
    out.lm_vocab_count = 128;
    out.logits.assign(static_cast<size_t>(T) * 128, -1.0f);
    for (int r = 0; r < T; ++r)
      out.logits[static_cast<size_t>(r) * 128 + winners[static_cast<size_t>(r)]] = 1.0f;
    return out;
  }
  void session_rollback(int, int accepted, int rows, int base) {
    rollbacks.push_back({accepted, rows, base});
  }
};

dgpp::SpecPickRows identity_pick() {
  return [](const std::vector<dgpp::sample::Candidate>& locals) {
    std::vector<int32_t> w;
    for (const auto& c : locals) w.push_back(c.id);
    return w;
  };
}

DGPP_TEST(dflash2_speculator_full_width_step_is_exact) {
  FakeModel m;
  dgpp::DFlash2Speculator<FakeModel> sp(m, 0, identity_pick());
  sp.start(10);
  require(sp.fed_rows() == std::vector<int64_t>({10, 11, 12, 13, 14, 15, 16, 17}), "full fed");
  const std::vector<int32_t> got = sp.step();
  require(got == std::vector<int32_t>({10, 11, 12, 13, 14, 15, 16, 17}), "accept-all commits the block");
  require(sp.next() == 99, "next is the last row's winner");
  require(m.rollbacks.size() == 1 && m.rollbacks[0].accepted == 8 && m.rollbacks[0].rows == 8 &&
              m.rollbacks[0].base == 0,
          "rollback carries accepted/rows/base");
  for (int p = 0; p < 7; ++p)
    require(sp.attempts(p) == 1 && sp.accepts(p) == 1, "all positions attempt and accept");
}

DGPP_TEST(dflash2_speculator_depth_cap_truncates_fed_and_counters) {
  FakeModel m;
  dgpp::DFlash2Speculator<FakeModel> sp(m, 0, identity_pick());
  sp.set_depth_policy([](const std::vector<int32_t>& d) { return d.empty() ? 0 : 3; });
  sp.start(10);
  require(sp.fed_rows() == std::vector<int64_t>({10, 11, 12, 13}), "fed capped at 1+3");
  require(sp.last_verify_depth() == 3, "depth recorded");
  const std::vector<int32_t> got = sp.step();
  require(got == std::vector<int32_t>({10, 11, 12, 13}), "capped accept-all commits");
  require(m.rollbacks.size() == 1 && m.rollbacks[0].accepted == 4 && m.rollbacks[0].rows == 4,
          "rollback sees the capped width");
  for (int p = 0; p < 3; ++p)
    require(sp.attempts(p) == 1 && sp.accepts(p) == 1, "verified positions count");
  for (int p = 3; p < 7; ++p)
    require(sp.attempts(p) == 0, "unverified tail does not count");
  // The block re-drafts full width after the capped step.
  require(sp.fed_rows() == std::vector<int64_t>({99, 11, 12, 13}), "next step re-drafts");
}

DGPP_TEST(dflash2_speculator_capped_reject_stays_exact) {
  FakeModel m;
  // Row 1 mispredicts fed[2]: winners[1] != fed[2] stops the accept at 2.
  m.winner_script.push_back({{11, 42, 13, 99}});
  dgpp::DFlash2Speculator<FakeModel> sp(m, 0, identity_pick());
  sp.set_depth_policy([](const std::vector<int32_t>&) { return 3; });
  sp.start(10);
  const std::vector<int32_t> got = sp.step();
  require(got == std::vector<int32_t>({10, 11}), "reject truncates the commit");
  require(sp.next() == 42, "next is the rejecting row's winner");
  require(sp.accepted_drafts() == 1, "one draft accepted");
}

DGPP_TEST(dflash2_speculator_commit_split_matches_commit) {
  FakeModel a, b;
  dgpp::DFlash2Speculator<FakeModel> sa(a, 0, identity_pick());
  dgpp::DFlash2Speculator<FakeModel> sb(b, 0, identity_pick());
  sa.start(10);
  sb.start(10);
  const std::vector<int64_t> fed = sa.fed_rows();
  const auto out = a.session_verify(0, fed);
  const std::vector<int32_t> winners = identity_pick()(dgpp::local_row_maxes(out, 8));
  const std::vector<int32_t> whole = sa.commit(fed, winners, 0);
  const std::vector<int32_t> split = sb.commit_verify(fed, winners, 3);
  require(whole == split, "commit_verify judges like commit");
  require(sb.next() == sa.next(), "next matches before the redraft");
  // The batched redraft's assignment path.
  std::vector<int32_t> d = {21, 22, 23, 24, 25, 26, 27};
  sb.set_drafts(d);
  require(sb.drafts() == d, "set_drafts installs the batch result");
  require(sa.drafts() == b.next_drafts, "solo commit redrafts from the model");
}

}  // namespace
