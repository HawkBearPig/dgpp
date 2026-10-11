#pragma once
// Context-lookup drafting (HyperQwen's `dflash2-lookup-drafting` idea,
// adapted to dgpp's MTP path): propose the continuation of the most recent
// earlier occurrence of the suffix just generated, from the request's own
// token history. Drafts are point masses, so the greedy verify stays exact:
// a draft commits iff it equals the target's argmax at that position, and an
// unverified or rejected tail is decoded plainly next step.
//
// v1 scope: same-depth fusion (lookup replaces MTP drafts within the
// configured depth) plus engine.lookup_tail: extra verify rows past the MTP
// block (a pinned mode; the device chain spans the full width and a strong
// lookup match fuses over it). Adaptive (triggered) tail sizing is
// follow-up work.
#include <cstdint>
#include <functional>
#include <vector>

namespace dgpp {

// What the history scan found.
struct LookupDraft {
  std::vector<int32_t> drafts;  // continuation tokens after the match (up to max_drafts)
  int match_len = 0;            // length of the matched suffix (0: no usable match)
  int64_t match_end = -1;       // history index where the matched occurrence ends
  bool strong = false;          // match_len >= nstrong
};

// Propose a continuation from token history.
//
// history[0..n): prompt ids + committed + pending tokens (the full sequence
//   the request has consumed so far, in order).
// max_drafts: continuation tokens to propose (clamped by what follows the match).
// nmin: minimum suffix-match length to propose anything (a coincidental short
//   match costs acceptance on prose).
// max_suffix: longest suffix considered (the scan compares at most this many
//   back from the end).
// nstrong: threshold marking a match takable on its own (see lookup_fuse).
//
// Picks the most recent (largest match_end < n - 1) occurrence of the longest
// suffix; ties go to the most recent. Overlapping occurrences are allowed, so
// a repeating period proposes from its own period. O(n * max_suffix) worst
// case with an early-out on first mismatch — fine for C1-C8 host-side use at
// moderate contexts; a 256K-context production path wants an index, not this.
inline LookupDraft lookup_propose(const int64_t* history, int64_t n, int max_drafts, int nmin = 6,
                                  int max_suffix = 64, int nstrong = 8) {
  LookupDraft out;
  if (history == nullptr || n < 2 || max_drafts <= 0 || nmin <= 0) return out;
  if (max_suffix > n - 1) max_suffix = static_cast<int>(n - 1);
  const int64_t last = history[n - 1];
  int best_len = 0;
  int64_t best_end = -1;
  // Most-recent-first: strictly-greater comparison keeps the first (most
  // recent) end on ties.
  for (int64_t e = n - 2; e >= 0; --e) {
    if (history[e] != last) continue;
    // Max length this end could reach: bounded by history start, the suffix
    // cap, and the suffix itself (the occurrence must end before the current
    // end... it may overlap, so only by the buffer start).
    int64_t cap = e + 1;  // occurrence start >= 0
    if (cap > max_suffix) cap = max_suffix;
    if (cap > n - 1) cap = n - 1;
    if (cap <= best_len) continue;  // cannot beat the incumbent
    int len = 1;
    while (len < cap && history[e - len] == history[n - 1 - len]) ++len;
    if (len > best_len) {
      best_len = len;
      best_end = e;
    }
  }
  if (best_len < nmin || best_end < 0) {
    // Report what the scan found even when it proposes nothing (below the
    // floor): the counter and the log distinguish "no repeat" from a weak
    // match the fusion rule skips.
    out.match_len = best_len;
    out.match_end = best_end;
    out.strong = best_len >= nstrong;
    return out;
  }
  out.match_len = best_len;
  out.match_end = best_end;
  out.strong = best_len >= nstrong;
  for (int i = 0; i < max_drafts && best_end + 1 + i < n; ++i)
    out.drafts.push_back(static_cast<int32_t>(history[best_end + 1 + i]));
  if (out.drafts.empty()) {
    out.match_len = 0;
    out.match_end = -1;
    out.strong = false;
  }
  return out;
}

// Fuse MTP drafts with a lookup proposal (same depth, no extra rows).
//
// A strong match is taken on its own. A weak one only if the drafter
// independently proposed the same first `agree` tokens — two sources
// agreeing (hidden state + text) is the cheap confidence signal that stops a
// coincidental short match from costing acceptance on prose. Otherwise the
// MTP drafts stand. Lengths: the fused block has mtp.size() entries (lookup
// tail truncated or zero-padded with the MTP tail when short).
inline LookupDraft lookup_fuse(const std::vector<int32_t>& mtp, const LookupDraft& lu, int nstrong = 8,
                               int agree = 2) {
  LookupDraft out;
  out.match_len = lu.match_len;
  out.match_end = lu.match_end;
  out.drafts = mtp;
  if (lu.drafts.empty() || lu.match_len < 0) return out;
  bool take = lu.match_len >= nstrong;
  if (!take && lu.match_len > 0 && agree > 0) {
    take = true;
    for (int i = 0; i < agree; ++i) {
      if (i >= static_cast<int>(mtp.size()) || i >= static_cast<int>(lu.drafts.size()) ||
          mtp[static_cast<size_t>(i)] != lu.drafts[static_cast<size_t>(i)]) {
        take = false;
        break;
      }
    }
  }
  if (!take) return out;
  out.strong = lu.strong;
  for (size_t i = 0; i < out.drafts.size() && i < lu.drafts.size(); ++i)
    out.drafts[i] = lu.drafts[i];
  return out;
}

// Schedules the long verify block only while a copy is running: each extra
// verify row costs attention over the full context, so the long block is
// worth it only when steps saturate. A single saturated step happens inside
// ordinary prose; two in a row is a copy (HyperQwen's rule).
class LookupTrigger {
 public:
  explicit LookupTrigger(int need = 2) : need_(need < 1 ? 1 : need) {}
  // verified: rows fed this step (drafts); accepted: drafts accepted.
  // Returns true when the next step may use the long block.
  bool update(int verified, int accepted) {
    if (verified > 0 && accepted >= verified)
      ++streak_;
    else
      streak_ = 0;
    return streak_ >= need_;
  }
  void reset() { streak_ = 0; }
  int streak() const { return streak_; }

 private:
  int need_ = 2;
  int streak_ = 0;
};

}  // namespace dgpp
