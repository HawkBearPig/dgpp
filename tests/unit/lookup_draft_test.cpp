// Context-lookup drafting (engine/lookup_draft.hpp): the history scan, the
// MTP fusion rule, the copy trigger, and the GreedySpeculator hook — all on
// scripted host models, no device.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "engine/decode_outputs.hpp"
#include "engine/lookup_draft.hpp"
#include "engine/speculative.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// history: prompt [10..19] then a repeat of [10..15] currently ending at 15.
DGPP_TEST(lookup_propose_verbatim_copy) {
  const std::vector<int64_t> h = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                  10, 11, 12, 13, 14, 15};
  const dgpp::LookupDraft lu = dgpp::lookup_propose(h.data(), 16, 4);
  require(lu.match_len == 6, "six-token suffix match");
  require(lu.match_end == 5, "earlier occurrence");
  require(!lu.strong, "six is below the strong threshold");
  require(lu.drafts == std::vector<int32_t>({16, 17, 18, 19}), "continuation follows the match");
}

DGPP_TEST(lookup_propose_no_match_is_empty) {
  const std::vector<int64_t> h = {1, 2, 3, 4, 5, 6, 7, 8};
  const dgpp::LookupDraft lu = dgpp::lookup_propose(h.data(), 8, 4);
  require(lu.drafts.empty() && lu.match_len == 0, "no proposal without a repeat");
}

DGPP_TEST(lookup_propose_short_match_rejected) {
  // Only a 2-token repeat at the end; nmin=6 rejects it (prose protection).
  const std::vector<int64_t> h = {1, 2, 3, 4, 5, 6, 5, 6};
  const dgpp::LookupDraft lu = dgpp::lookup_propose(h.data(), 8, 4);
  require(lu.drafts.empty(), "coincidental short match proposes nothing");
}

DGPP_TEST(lookup_propose_most_recent_wins_ties) {
  // Suffix [7,8] occurs ending at 1 and at 5; the most recent (5) proposes 9.
  const std::vector<int64_t> h = {7, 8, 1, 2, 7, 8, 9, 10, 7, 8};
  const dgpp::LookupDraft lu = dgpp::lookup_propose(h.data(), 10, 2, /*nmin=*/2);
  require(lu.match_end == 5, "most recent occurrence");
  require(lu.drafts == std::vector<int32_t>({9, 10}), "its continuation");
}

DGPP_TEST(lookup_propose_overlapping_period) {
  // Period-2 repeat: the suffix also ends inside itself; proposes the period.
  const std::vector<int64_t> h = {5, 6, 5, 6, 5, 6};
  const dgpp::LookupDraft lu = dgpp::lookup_propose(h.data(), 6, 2, /*nmin=*/2);
  require(!lu.drafts.empty(), "overlap proposes");
  require(lu.drafts[0] == 5, "from its own period");
}

DGPP_TEST(lookup_fuse_strong_takes_lookup) {
  const std::vector<int32_t> mtp = {1, 2, 3};
  dgpp::LookupDraft lu;
  lu.match_len = 9;
  lu.match_end = 3;
  lu.strong = true;
  lu.drafts = {7, 8, 9};
  const dgpp::LookupDraft f = dgpp::lookup_fuse(mtp, lu);
  require(f.drafts == std::vector<int32_t>({7, 8, 9}), "strong match stands alone");
}

DGPP_TEST(lookup_fuse_weak_needs_agreement) {
  const std::vector<int32_t> mtp = {7, 8, 1};
  dgpp::LookupDraft lu;
  lu.match_len = 6;
  lu.drafts = {7, 8, 9};
  const dgpp::LookupDraft agree = dgpp::lookup_fuse(mtp, lu);
  require(agree.drafts == std::vector<int32_t>({7, 8, 9}), "agreeing weak match taken");
  const std::vector<int32_t> other = {1, 2, 3};
  const dgpp::LookupDraft disagree = dgpp::lookup_fuse(other, lu);
  require(disagree.drafts == other, "disagreeing weak match keeps MTP");
}

DGPP_TEST(lookup_trigger_needs_two_saturated_steps) {
  dgpp::LookupTrigger t;
  require(!t.update(3, 3), "one saturated step is not a copy");
  require(t.update(3, 3), "two in a row is");
  require(!t.update(3, 1), "a rejection resets");
  require(!t.update(3, 3), "streak restarts at one");
}

// Scripted copy target: the transcript repeats `pattern` forever. The MTP
// drafter always proposes `wrong`, so MTP-only acceptance is zero and any
// accepted draft comes from the lookup path. pos is the truth index of the
// pending token.
struct CopyModel {
  bool mtp_enabled() const { return true; }
  std::vector<int32_t> pattern = {5, 6, 7, 8};
  int32_t wrong = 1;
  int64_t pos = 0;
  int depth = 2;

  int32_t truth(int64_t idx) const {
    const size_t p = static_cast<size_t>((pos + idx) % static_cast<int64_t>(pattern.size()));
    return pattern[p];
  }
  dgpp::DecodeOutputs session_draft(int, const std::vector<int64_t>&) {
    dgpp::DecodeOutputs o;
    o.lm_vocab_begin = 0;
    o.lm_vocab_count = 16;
    o.logits.assign(16, -1.0f);
    o.logits[static_cast<size_t>(wrong)] = 1.0f;
    return o;
  }
  bool session_draft_chain_fits(int, int) const { return true; }
  dgpp::DecodeOutputs session_draft_chain(int, int32_t, int, bool, bool) { return session_draft(0, {}); }
  dgpp::DecodeOutputs session_verify(int, const std::vector<int64_t>& fed) {
    const int T = static_cast<int>(fed.size());
    dgpp::DecodeOutputs o;
    o.lm_vocab_begin = 0;
    o.lm_vocab_count = 16;
    o.logits.assign(static_cast<size_t>(T) * 16, -1.0f);
    // Row r predicts position pos+r+1: accept iff fed[r+1] is the truth.
    // Either way the winner is the truth (a reject ends the step on it).
    for (int r = 0; r < T; ++r) o.logits[static_cast<size_t>(r) * 16 + truth(r + 1)] = 1.0f;
    return o;
  }
  void session_rollback(int, int accepted) { pos += accepted; }
};

dgpp::SpecPickRows identity_pick() {
  return [](const std::vector<dgpp::sample::Candidate>& locals) {
    std::vector<int32_t> w;
    for (const auto& c : locals) w.push_back(c.id);
    return w;
  };
}

DGPP_TEST(lookup_fused_speculator_copies_exactly) {
  // History seeds the copy: two full periods before decoding starts.
  // history_fn carries prompt + committed; the hook appends the pending.
  std::vector<int64_t> hist = {5, 6, 7, 8, 5, 6, 7, 8};
  CopyModel m;
  dgpp::GreedySpeculator<CopyModel> spec(m, 0, identity_pick(), /*depth=*/2);
  spec.set_lookup([&hist]() { return hist; }, /*nmin=*/2, /*nstrong=*/4);
  spec.start(5);
  std::vector<int32_t> out;
  for (int i = 0; i < 6; ++i) {
    for (int32_t t : spec.step()) {
      out.push_back(t);
      hist.push_back(t);
    }
  }
  // The copy target emits the period; every step verifies both drafts.
  for (size_t i = 0; i < out.size(); ++i)
    require(out[i] == std::vector<int32_t>({5, 6, 7, 8})[i % 4], "lookup transcript follows the period");
  require(spec.accepted_drafts() == spec.steps() * 2, "every fused draft accepted");
}

DGPP_TEST(lookup_off_mtp_only_rejects_on_copy_workload) {
  std::vector<int64_t> hist = {5, 6, 7, 8, 5, 6, 7, 8};
  CopyModel m;
  dgpp::GreedySpeculator<CopyModel> spec(m, 0, identity_pick(), /*depth=*/2);
  spec.start(5);
  int accepted = 0;
  for (int i = 0; i < 6; ++i) {
    const int before = spec.accepted_drafts();
    for (int32_t t : spec.step()) hist.push_back(t);
    accepted += spec.accepted_drafts() - before;
  }
  require(accepted == 0, "the wrong drafter never verifies on a copy workload");
}

}  // namespace
