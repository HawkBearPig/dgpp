// The DeepSeek-V4.1 session surface's gate (plan G5's `dsv41_decode_test`):
// incremental decode over Dsv41Model's request slots against the cold
// re-forward, two interleaved slots, the chunked prefill across the
// pool's 128-token blocks (a ratio-2 pair boundary at every cut), the
// prefix snapshots (hot == cold at a cut and mid-decode: the rings, the
// compressor tails and the Engram context travel with the entry), a
// closed and reopened slot, and the eager greedy speculator through the
// DSpark block (the committed transcript is the plain greedy one).
//
// Why the decode and chunked audits certify selection flips instead of
// demanding agreement (2026-09-13): the decode row's projections run the
// GEMV chain, the prefill's the tile kernels, and a prefill's GEMMs
// accumulate in a row-count dependent order — rounding-level differences
// (0.4 % of a row; one bf16 ulp in a few elements). The window rows and
// the compressed entries are fp8- and fp4-coded, so a row appended by the
// other path re-quantizes that difference into code flips, and the index
// queries are fp4-coded too: at the kv sources the index logits move by
// percents of their range, and on this random-weight fixture the top-16
// boundary gaps are that close for most rows. A flip's attention then
// differs by O(1), its cached entries and, through the 16-row window,
// every later row's state are perturbed — flips cascade. The gates that
// follow from this: rows before the first flip hold the tight budget; the
// first flip is certified (its reference boundary gap within 5 % of the
// range, the row's streams within 1 % before the first kv source and 10 %
// after); a chunked prefill's flips are each certified against the
// chunks' own index logits (the gap within twice the measured deviation);
// after a certified flip the flipped rows are exempt and the kept rows
// hold the long-audit budget. The exact gates around this (prefill ==
// forward, the interleaved slots, the snapshots, the graph engine, the
// TP twin) are the decode path's bitwise evidence.
//
//   --fixture DIR   the fixture gates (tests/cuda/dsv41_fixture.hpp's
//                   checkpoint with its Engram sidecar, written by
//                   dsv41_forward_test --write-fixture)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "engine/speculative.hpp"
#include "engine/verify_schedule.hpp"
#include "models/dsv41/config.hpp"
#include "models/dsv41/model.hpp"

namespace fs = std::filesystem;
using dgpp::Dsv41Model;
using dgpp::Dsv41Residency;
using dgpp::Dsv41TextConfig;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

std::vector<int64_t> smoke_tokens(const Dsv41TextConfig& cfg, int n, uint64_t seed) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = seed;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    t[static_cast<size_t>(i)] = static_cast<int64_t>(s % static_cast<uint64_t>(cfg.vocab_size));
  }
  return t;
}

int32_t argmax(const float* row, int n) {
  int32_t best = 0;
  for (int i = 1; i < n; ++i)
    if (row[i] > row[best]) best = i;
  return best;
}

struct RowCompare {
  double l2 = 0;
  bool top1_equal = false;
  bool near_tie = false;  // the reference's top-1 and the candidate within 2 %
};

RowCompare compare_row(const float* got, const float* want, int n) {
  RowCompare c;
  double d2 = 0, w2 = 0;
  for (int i = 0; i < n; ++i) {
    const double d = static_cast<double>(got[i]) - want[i];
    d2 += d * d;
    w2 += static_cast<double>(want[i]) * want[i];
  }
  c.l2 = std::sqrt(d2) / std::sqrt(w2 + 1e-30);
  const int32_t a = argmax(want, n), b = argmax(got, n);
  c.top1_equal = a == b;
  if (!c.top1_equal) {
    const double v1 = want[a], v2 = want[b];
    c.near_tie = std::fabs(v1 - v2) / (std::fabs(v1) + 1e-30) < 0.02;
  }
  return c;
}

bool bitwise(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (std::memcmp(&a[i], &b[i], 4) != 0) return false;
  return true;
}

// Greedy steps from an open slot: the pending token is the prefill's
// argmax; every step's logits row is kept, with the step's CSA2 selections
// (DGPP_DSV41_CAPTURE_DECODE=1 keeps them on the decode walk).
struct Transcript {
  std::vector<int64_t> tokens;
  std::vector<std::vector<float>> rows;
  std::vector<std::vector<std::vector<int32_t>>> sels;    // per row: per index source [index_topk]
  std::vector<std::vector<std::vector<uint16_t>>> states;  // per row: per layer [4H] the decode row's streams
  std::vector<std::vector<Dsv41Model::IndexLogits>> qcodes;  // per row: per index source the coded query (q_codes)
};

Transcript greedy(Dsv41Model& m, int req, const std::vector<int64_t>& prompt, int steps) {
  Transcript t;
  Dsv41Model::Outputs o = m.session_prefill(req, prompt);
  int64_t pending = argmax(o.logits.data(), o.lm_vocab_count);
  t.tokens.push_back(pending);
  t.rows.push_back(o.logits);
  t.sels.emplace_back();  // the prefill's row: its bitwise gate stands in
  t.states.emplace_back();
  setenv("DGPP_DSV41_CAPTURE_DECODE", "1", 1);
  for (int s = 0; s < steps; ++s) {
    o = m.session_step(req, pending);
    pending = argmax(o.logits.data(), o.lm_vocab_count);
    t.tokens.push_back(pending);
    t.rows.push_back(o.logits);
    t.sels.push_back(o.dsa_selections);
    t.states.push_back(o.layer_states);
    t.qcodes.push_back(m.debug_index_logits());
  }
  unsetenv("DGPP_DSV41_CAPTURE_DECODE");
  return t;
}

std::string ids_text(const std::vector<int64_t>& ids) {
  std::string s;
  for (size_t i = 0; i < ids.size(); ++i) s += (i ? "," : "") + std::to_string(ids[i]);
  return s;
}

// A selection flip's certificate: the reference's own index logits of the
// row (the forward's capture, -inf past the visible entries) must put the
// entries the two paths disagree on within `gap_budget` of the row's range
// — the smallest reference logit among the entries only the reference
// selected against the largest among the entries only the other path
// selected. A flip wider than that is a defect, not a near tie.
struct FlipCert {
  bool certified = true;
  double gap = 0;   // the widest gap over the flipped sources, of the range
  int source = -1;  // the source with the widest gap
  int coded = 0;    // codes (+ scales) of the row's index query that differ between the paths at the flipped sources
};

// The coding evidence: the two paths' e4m3 index-query codes and row
// scales of one row at one source (Dsv41Model::IndexLogits::q_codes). The
// selection is the top-k of the codes' dot products with the cached keys,
// so two paths with identical codes select from identical logits (up to
// the dot products' own rounding, far under any gap here); a differing
// code is the fp8 coding's discontinuity — the reason a flip can sit on a
// reference gap no fp32 reordering could bridge (a code step moves an
// element by 1/8 or more).
int coded_query_diffs(const Dsv41Model::IndexLogits& a, size_t row_a, const Dsv41Model::IndexLogits& b, size_t row_b) {
  if (a.q_rows <= static_cast<int>(row_a) || b.q_rows <= static_cast<int>(row_b) || a.q_heads != b.q_heads || a.q_heads == 0) return -1;
  const size_t w = static_cast<size_t>(a.q_heads) * 128;
  int n = 0;
  for (size_t i = 0; i < w; ++i) n += a.q_codes[row_a * w + i] != b.q_codes[row_b * w + i];
  for (int h = 0; h < a.q_heads; ++h)
    n += a.q_scales[row_a * static_cast<size_t>(a.q_heads) + h] != b.q_scales[row_b * static_cast<size_t>(b.q_heads) + h];
  return n;
}

// One source's gap for a flipped row (`r`, `g`: the two paths' [ms]
// selections; `row` indexes the reference's logits).
double source_flip_gap(const Dsv41Model::IndexLogits& lg, const int32_t* r, const int32_t* g, size_t row, size_t ms) {
  require(static_cast<size_t>(lg.rows) > row, "certify_flip: the reference's logits cover the row");
  const float* v = lg.values.data() + row * static_cast<size_t>(lg.stride);
  double lo = 1e300, hi = -1e300;
  for (int64_t j = 0; j < lg.stride; ++j)
    if (std::isfinite(v[j])) { lo = std::min<double>(lo, v[j]); hi = std::max<double>(hi, v[j]); }
  double ref_only_min = 1e300, got_only_max = -1e300;
  for (size_t i = 0; i < ms; ++i) {
    if (r[i] >= 0 && std::find(g, g + ms, r[i]) == g + ms && r[i] < lg.stride) ref_only_min = std::min<double>(ref_only_min, v[r[i]]);
    if (g[i] >= 0 && std::find(r, r + ms, g[i]) == r + ms && g[i] < lg.stride) got_only_max = std::max<double>(got_only_max, v[g[i]]);
  }
  const double range = std::max({hi - lo, std::fabs(hi), 1e-30});
  return (ref_only_min < 1e300 && got_only_max > -1e300) ? (ref_only_min - got_only_max) / range : 0.0;
}

// A flip is certified by the reference's gap (within `gap_budget` of the
// range: a near tie under the paths' rounding), or by the coding: the
// other path's index query codes differ from the reference's at that
// source (`got_q`, the other path's capture of the row, when given).
FlipCert certify_flip(const std::vector<Dsv41Model::IndexLogits>& logits, const std::vector<std::vector<int32_t>>& ref_sels,
                      const std::vector<std::vector<int32_t>>& got_sels, size_t row, size_t ms, double gap_budget,
                      const std::vector<Dsv41Model::IndexLogits>* got_q = nullptr, size_t got_row = 0) {
  FlipCert cert;
  for (size_t l = 0; l < ref_sels.size(); ++l) {
    const int32_t* r = ref_sels[l].data() + row * ms;
    const int32_t* g = got_sels[l].data();
    if (std::memcmp(r, g, ms * 4) == 0) continue;
    const double gap = source_flip_gap(logits[l], r, g, row, ms);
    const int coded = (got_q && l < got_q->size()) ? coded_query_diffs(logits[l], row, (*got_q)[l], got_row) : 0;
    if (coded > 0) cert.coded += coded;
    if (gap > cert.gap) { cert.gap = gap; cert.source = static_cast<int>(l); }
    if (gap > gap_budget && coded <= 0) cert.certified = false;
  }
  return cert;
}

// Greedy (or teacher-forced, `force`: the tokens to feed) steps from an
// open slot whose prefill row is `first`; the decode rows' selections
// captured.
Transcript decode_rows(Dsv41Model& m, int req, const std::vector<float>& prefill_logits, int steps,
                       const std::vector<int64_t>* force = nullptr) {
  Transcript t;
  int64_t pending = force ? (*force)[0] : argmax(prefill_logits.data(), static_cast<int>(prefill_logits.size()));
  t.tokens.push_back(pending);
  t.rows.push_back(prefill_logits);
  t.sels.emplace_back();
  t.states.emplace_back();
  setenv("DGPP_DSV41_CAPTURE_DECODE", "1", 1);
  for (int s = 0; s < steps; ++s) {
    const Dsv41Model::Outputs o = m.session_step(req, pending);
    pending = force ? (*force)[static_cast<size_t>(s) + 1] : argmax(o.logits.data(), o.lm_vocab_count);
    t.tokens.push_back(pending);
    t.rows.push_back(o.logits);
    t.sels.push_back(o.dsa_selections);
    t.states.push_back(o.layer_states);
    t.qcodes.push_back(m.debug_index_logits());
  }
  unsetenv("DGPP_DSV41_CAPTURE_DECODE");
  return t;
}

// The forward test's rule for a flip whose two logit rows are BOTH at
// hand: the reference's boundary gap must lie within twice the rows'
// logit deviation (of the range) plus the fp32 noise — the flip is then
// the measured deviation's doing.
double logit_deviation(const Dsv41Model::IndexLogits& ref, size_t ref_row, const Dsv41Model::IndexLogits& got,
                       size_t got_row) {
  double lo = 1e300, hi = -1e300, d = 0;
  for (int64_t j = 0; j < std::min(ref.stride, got.stride); ++j) {
    const float a = ref.values[ref_row * static_cast<size_t>(ref.stride) + static_cast<size_t>(j)];
    const float b = got.values[got_row * static_cast<size_t>(got.stride) + static_cast<size_t>(j)];
    if (!std::isfinite(a) || !std::isfinite(b)) continue;
    lo = std::min<double>(lo, a);
    hi = std::max<double>(hi, a);
    d = std::max<double>(d, std::fabs(a - b));
  }
  return d / std::max({hi - lo, std::fabs(hi), 1e-30});
}

// The transcript's rows against the cold re-forward of prompt + tokens.
// The decode path's rows sit within ~1 % of the prefill's (the GEMV chain
// against the tile kernels, the fp4-coded queries and keys), so a
// selection boundary that close moves: the FIRST row whose CSA2 selection
// differs from the re-forward's at any index source must be a certified
// near tie (certify_flip: the disagreeing entries within kFlipGapBudget
// of the row's logit range in the reference's own logits — a wider gap
// is a defect and fails the audit). Every row before it holds the tight
// budget. That row's attention then differs by O(1): its cached entries
// and, through the 16-row window, every later row's state are perturbed
// — flips cascade on this random-weight fixture and the reference's
// logits no longer measure them — so after the first flip the flipped
// rows are exempt (reported with their gaps), the kept rows hold the
// long-audit budget and the top-1 rule, and must number enough.
constexpr double kFlipGapBudget = 0.05;
constexpr double kCleanBudget = 2e-2;

int audit(Dsv41Model& ref, const std::vector<int64_t>& prompt, const Transcript& t, const char* what,
          double l2_budget, double clean_budget = kCleanBudget, int min_kept = -1) {
  std::vector<int64_t> all(prompt);
  all.insert(all.end(), t.tokens.begin(), t.tokens.end() - 1);
  const Dsv41Model::Outputs f = ref.forward(all, true);
  const std::vector<Dsv41Model::IndexLogits> ref_logits = ref.debug_index_logits();
  const int V = f.lm_vocab_count;
  const size_t P = prompt.size();
  int hard = 0, soft = 0, flipped = 0, clean_rows = 0;
  int first_flip = -1;
  double first_gap = 0;
  int first_source = -1;
  int first_coded = 0;
  bool first_certified = true;
  double worst_l2 = 0, worst_flipped = 0, worst_clean = 0;
  std::string profile, gaps;
  for (size_t i = 0; i < t.rows.size(); ++i) {
    const size_t row = P - 1 + i;
    bool flip = false;
    if (i < t.sels.size() && !t.sels[i].empty()) {
      require(t.sels[i].size() == f.dsa_selections.size(), std::string(what) + ": selection captures");
      const size_t ms = t.sels[i][0].size();
      for (size_t l = 0; l < f.dsa_selections.size() && !flip; ++l)
        flip = std::memcmp(t.sels[i][l].data(), f.dsa_selections[l].data() + row * ms, ms * 4) != 0;
      if (flip) {
        const FlipCert cert = certify_flip(ref_logits, f.dsa_selections, t.sels[i], row, ms, kFlipGapBudget,
                                           i < t.qcodes.size() ? &t.qcodes[i] : nullptr, 0);
        if (first_flip < 0) {
          first_flip = static_cast<int>(P + i);
          first_gap = cert.gap;
          first_source = cert.source;
          first_coded = cert.coded;
          first_certified = cert.certified;
          // The evidence behind the certificate: the decode row's streams
          // against the forward's row, per layer — the perturbation that
          // reaches the flipped source's fp4-coded index query.
          // Its layers before the first kv source carry the GEMV-vs-tile
          // difference alone (within 1 %); from the first compressed layer
          // on, the fp8 window rows and fp4 entries the decode path
          // appended re-quantize that difference into code flips (single
          // elements moving by a quarter or more), which the fp4-coded
          // index query turns into the boundary move. Past the flipped
          // source the row attends to other entries: its move is the
          // flip's, which reaches 0.4 on these random fixtures (the later
          // flipped rows' worst below; the EXL3 fixture's first flip moved
          // 0.24–0.39 at layers 4–7, the release fixture's 0.02) — bounded
          // at 0.5, well under a defect's order of magnitude.
          if (i < t.states.size() && !t.states[i].empty() && !f.layer_states.empty()) {
            const size_t H4 = f.layer_states[0].size() / (P + t.rows.size() - 1);
            std::string per_layer;
            double worst_pre = 0, worst_post = 0;
            for (size_t l = 0; l < t.states[i].size() && l < f.layer_states.size(); ++l) {
              double d2 = 0, r2 = 0;
              for (size_t k = 0; k < H4; ++k) {
                const double a = dgpp::bf16_bits_to_float(t.states[i][l][k]);
                const double b = dgpp::bf16_bits_to_float(f.layer_states[l][row * H4 + k]);
                d2 += (a - b) * (a - b);
                r2 += b * b;
              }
              const double l2 = std::sqrt(d2) / std::max(std::sqrt(r2), 1e-30);
              if (l < 2) worst_pre = std::max(worst_pre, l2); else worst_post = std::max(worst_post, l2);
              char buf[96];
              std::snprintf(buf, sizeof(buf), " L%zu:%.3g", l, l2);
              per_layer += buf;
            }
            std::printf("[ .. ] %s: the first flipped row's decode streams vs the forward's, relative l2 per layer%s\n", what,
                        per_layer.c_str());
            require(worst_pre < 1e-2, std::string(what) + ": the first flipped row differs before the first kv source");
            require(worst_post < 0.5, std::string(what) + ": the first flipped row differs beyond a flip's move");
          }
        }
        char buf[48];
        std::snprintf(buf, sizeof(buf), " p%zu:%.3g", P + i, cert.gap);
        gaps += buf;
      }
    }
    if (i >= 1 && i <= 2 && i < t.states.size() && !t.states[i].empty() && std::getenv("DGPP_DSV41_AUDIT_DEBUG")) {
      const size_t H4 = f.layer_states[0].size() / (P + t.rows.size() - 1);
      std::printf("[ .. ] debug row i=%zu (position %zu, %s): per-layer stream l2", i, P + i - 1, flip ? "flipped" : "clean");
      for (size_t l = 0; l < t.states[i].size(); ++l) {
        double d2 = 0, r2 = 0;
        for (size_t k = 0; k < H4; ++k) {
          const double a = dgpp::bf16_bits_to_float(t.states[i][l][k]);
          const double b = dgpp::bf16_bits_to_float(f.layer_states[l][row * H4 + k]);
          d2 += (a - b) * (a - b); r2 += b * b;
        }
        std::printf(" %.3g", std::sqrt(d2) / std::max(std::sqrt(r2), 1e-30));
      }
      std::printf(" (decode state size %zu, H4 %zu)\n", t.states[i][0].size(), H4);
    }
    const float* want = f.logits.data() + row * static_cast<size_t>(V);
    const RowCompare c = compare_row(t.rows[i].data(), want, V);
    if (flip) {
      ++flipped;
      worst_flipped = std::max(worst_flipped, c.l2);
      continue;
    }
    if (first_flip < 0) {
      ++clean_rows;
      worst_clean = std::max(worst_clean, c.l2);
    }
    worst_l2 = std::max(worst_l2, c.l2);
    if (!c.top1_equal) (c.near_tie ? soft : hard) += 1;
    if (t.rows.size() > 16 && (i % 4 == 0 || c.l2 > 0.02)) {
      char buf[64];
      std::snprintf(buf, sizeof(buf), " p%zu:%.3f%s", P + i, c.l2, c.top1_equal ? "" : (c.near_tie ? "~" : "!"));
      profile += buf;
    }
  }
  if (!profile.empty()) std::printf("[ .. ] %s: per-position relative l2 (kept rows)%s\n", what, profile.c_str());
  if (!gaps.empty()) std::printf("[ .. ] %s: flipped rows' reference boundary gaps (of the range)%s\n", what, gaps.c_str());
  std::printf("[ .. ] %s: %zu rows vs the re-forward — %d clean rows before the first flip (worst l2 %.3g); the first flip at "
              "position %d, source %d, gap %.3g of the range%s; %d flipped rows after it (worst l2 %.3g); kept rows: worst "
              "relative l2 %.3g, top-1 hard %d near-tie %d\n",
              what, t.rows.size(), clean_rows, worst_clean, first_flip, first_source, first_gap,
              first_flip < 0 ? "" : (first_gap <= kFlipGapBudget ? " (a certified near tie)" : first_coded > 0 ? " (the coded query differs: certified)" : " (UNCERTIFIED)"), flipped,
              worst_flipped, worst_l2, hard, soft);
  if (first_flip >= 0 && first_coded > 0)
    std::printf("[ .. ] %s: the first flip's index query differs from the re-forward's in %d codes/scales at the flipped sources\n", what, first_coded);
  require(first_flip < 0 || first_certified,
          std::string(what) + ": the first selection flip is not a near tie and the coded queries agree");
  require(worst_clean < clean_budget, std::string(what) + ": a row before the first flip is over the tight budget");
  require(hard == 0, std::string(what) + ": a top-1 mismatch beyond the near-tie margin on a kept row");
  require(worst_l2 < l2_budget, std::string(what) + ": relative l2 over budget on a kept row");
  const int rows_n = static_cast<int>(t.rows.size());
  require(rows_n - flipped >= (min_kept >= 0 ? min_kept : std::min(8, (rows_n + 1) / 2)),
          std::string(what) + ": too few rows without a selection flip to audit");
  return soft;
}

int run_fixture(const std::string& dir) {
  const Dsv41TextConfig cfg = Dsv41TextConfig::from_json_file((fs::path(dir) / "config.json").string());
  const std::vector<int64_t> A = smoke_tokens(cfg, 40, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 27, 0xD1B54A32D192ED03ull);
  const std::vector<int64_t> Cl = smoke_tokens(cfg, 300, 0x2545F4914F6CDD1Dull);  // three pool blocks
  const std::vector<int64_t> Cm = smoke_tokens(cfg, 140, 0x6A09E667F3BCC908ull);  // one cut at 128
  Dsv41Model m(cfg, dir, /*max_tokens=*/512, /*max_cache_tokens=*/1024, Dsv41Residency::Resident, nullptr, 0, 1,
               /*max_requests=*/2);
  const int V = m.lm_vocab_count();
  const int block = static_cast<int>(m.kv_block_tokens());

  // 1. prefill == forward, bitwise (the same m=T GEMMs on the same state).
  {
    const Dsv41Model::Outputs f = m.forward(A);
    const Dsv41Model::Outputs p = m.session_prefill(0, A);
    require(bitwise(p.logits, std::vector<float>(f.logits.end() - V, f.logits.end())),
            "prefill: the last row differs from the forward's");
    m.session_close(0);
    std::printf("[ OK ] the session prefill's last row is bitwise the forward's\n");
  }

  // 2. Incremental decode vs the re-forward: 40 steps from the 40-token
  //    prompt (positions 40 .. 79, the ratio-2 pair cadence and the 16-entry
  //    select horizon inside).
  const Transcript tA = greedy(m, 0, A, 40);
  m.session_close(0);
  const int soft = audit(m, A, tA, "40-step decode", 5e-2);
  std::printf("[ OK ] 40-step decode matches the re-forward (%d near ties)\n", soft);

  // 3. Two slots interleaved reproduce their solo runs bitwise.
  const Transcript tB = greedy(m, 0, B, 8);
  m.session_close(0);
  {
    Dsv41Model::Outputs oa = m.session_prefill(0, A);
    Dsv41Model::Outputs ob = m.session_prefill(1, B);
    require(bitwise(oa.logits, tA.rows[0]), "interleaved: slot 0's prefill differs from the solo run");
    require(bitwise(ob.logits, tB.rows[0]), "interleaved: slot 1's prefill differs from the solo run");
    int64_t pa = tA.tokens[0], pb = tB.tokens[0];
    for (int s = 0; s < 8; ++s) {
      oa = m.session_step(0, pa);
      ob = m.session_step(1, pb);
      require(bitwise(oa.logits, tA.rows[static_cast<size_t>(s) + 1]),
              "interleaved: slot 0's step " + std::to_string(s) + " differs from the solo run");
      require(bitwise(ob.logits, tB.rows[static_cast<size_t>(s) + 1]),
              "interleaved: slot 1's step " + std::to_string(s) + " differs from the solo run");
      pa = argmax(oa.logits.data(), V);
      pb = argmax(ob.logits.data(), V);
    }
    m.session_close(1);
    for (int s = 8; s < 12; ++s) {
      oa = m.session_step(0, pa);
      require(bitwise(oa.logits, tA.rows[static_cast<size_t>(s) + 1]),
              "interleaved: slot 0's step " + std::to_string(s) + " after slot 1 closed");
      pa = argmax(oa.logits.data(), V);
    }
    m.session_close(0);
    std::printf("[ OK ] two interleaved slots reproduce their solo transcripts bitwise\n");
  }

  // 4. Chunked prefill: a 140-token prompt through 128-token chunks (the
  //    core cuts at max_tokens, pool-aligned; the cut is a ratio-2 pair
  //    boundary and crosses the compressed caches' blocks) agrees with the
  //    one-shot; decode continues from the chunked state. (A longer prompt
  //    flips nearly every later selection on this fixture — the cascade.)
  {
    Dsv41Model c(cfg, dir, /*max_tokens=*/block, /*max_cache_tokens=*/1024, Dsv41Residency::Resident, nullptr, 0, 1,
                 /*max_requests=*/1);
    const Dsv41Model::Outputs one = m.session_prefill(0, Cm);
    m.session_close(0);
    // The one-shot's per-row selections and index logits (the forward's
    // capture), the chunked walk's per-row selections (the prefill capture
    // switch): the first row where they differ is where the chunked state
    // departs — a certified near tie under the chunks' GEMM rounding — and
    // the last row is held tight only when no row before it flipped.
    const Dsv41Model::Outputs one_f = m.forward(Cm, true);
    const std::vector<Dsv41Model::IndexLogits> one_logits = m.debug_index_logits();
    // Every chunk's index logits: the first chunk's are the 128-row prefill's
    // (bitwise the same walk), the last chunk's the chunked run's own last
    // select tile. The projections' GEMMs accumulate in a row-count
    // dependent order (a one-ulp flip in a few elements per 128 rows —
    // layer 0 shows 4 such rows between the 140- and 128-row walks), and
    // the fp4-coded index queries turn such a flip into percent-level
    // logit moves at the kv sources: a selection then flips when its
    // boundary gap lies within the measured logit deviation — every flip
    // must pass that certificate (the forward test's rule).
    setenv("DGPP_DSV41_CAPTURE_PREFILL", "1", 1);
    const Dsv41Model::Outputs p0 = c.session_prefill(0, std::vector<int64_t>(Cm.begin(), Cm.begin() + block));
    const std::vector<Dsv41Model::IndexLogits> chunk0_logits = c.debug_index_logits();
    c.session_close(0);
    const Dsv41Model::Outputs p = c.session_prefill(0, Cm);
    const std::vector<Dsv41Model::IndexLogits> chunk1_logits = c.debug_index_logits();
    unsetenv("DGPP_DSV41_CAPTURE_PREFILL");
    (void)p0;
    require(p.dsa_selections.size() == one_f.dsa_selections.size() && !p.dsa_selections.empty(), "chunked: selection captures");
    const size_t ms = static_cast<size_t>(cfg.index_topk);
    int first_flip = -1, flips = 0, unc = 0;
    double worst_dev = 0, worst_gap = 0;
    for (size_t row = 0; row < Cm.size(); ++row) {
      bool flip = false;
      std::vector<std::vector<int32_t>> got(p.dsa_selections.size());
      for (size_t l = 0; l < p.dsa_selections.size(); ++l) {
        require(p.dsa_selections[l].size() == Cm.size() * ms, "chunked: selection rows");
        got[l].assign(p.dsa_selections[l].begin() + row * ms, p.dsa_selections[l].begin() + (row + 1) * ms);
        if (std::memcmp(got[l].data(), one_f.dsa_selections[l].data() + row * ms, ms * 4) != 0) flip = true;
      }
      if (!flip) continue;
      ++flips;
      if (first_flip < 0) first_flip = static_cast<int>(row);
      const FlipCert cert = certify_flip(one_logits, one_f.dsa_selections, got, row, ms, 1e300);
      const bool second = row >= static_cast<size_t>(block);
      const std::vector<Dsv41Model::IndexLogits>& chunk = second ? chunk1_logits : chunk0_logits;
      const size_t chunk_row = second ? row - static_cast<size_t>(block) : row;
      double dev = 0;
      for (size_t l = 0; l < one_logits.size(); ++l)
        dev = std::max(dev, logit_deviation(one_logits[l], row, chunk[l], chunk_row));
      worst_dev = std::max(worst_dev, dev);
      worst_gap = std::max(worst_gap, cert.gap);
      if (cert.gap > 2.0 * dev + 2e-3) {
        ++unc;
        std::printf("[ .. ] chunked prefill: UNCERTIFIED flip at row %zu (source %d): gap %.3g against a logit deviation of %.3g\n",
                    row, cert.source, cert.gap, dev);
      }
    }
    RowCompare r = compare_row(p.logits.data(), one.logits.data(), V);
    std::printf("[ .. ] chunked prefill (%d-row chunks): relative l2 %.3g, top-1 %s; %d selection flips vs the one-shot "
                "(the first at row %d; the widest gap %.3g, the logit deviation up to %.3g of the range; %d uncertified)\n",
                block, r.l2, r.top1_equal ? "equal" : (r.near_tie ? "near tie" : "MISMATCH"), flips, first_flip, worst_gap,
                worst_dev, unc);
    require(unc == 0, "chunked prefill: a selection flip beyond the measured logit deviation");
    require(r.top1_equal || r.near_tie, "chunked prefill: top-1 mismatch");
    // No flip before the last row: the chunks' GEMM rounding alone (the
    // tight budget); a flipped earlier row's cached entries moved the last
    // row's attention (the long audit's budget).
    require(r.l2 < (first_flip < 0 || first_flip == static_cast<int>(Cm.size()) - 1 ? 2e-2 : 1e-1),
            "chunked prefill: relative l2 over budget");
    int64_t pending = argmax(p.logits.data(), V);
    Transcript tc;
    tc.tokens.push_back(pending);
    tc.rows.push_back(p.logits);
    tc.sels.emplace_back();
    tc.states.emplace_back();
    setenv("DGPP_DSV41_CAPTURE_DECODE", "1", 1);
    for (int s = 0; s < 48; ++s) {
      const Dsv41Model::Outputs o = c.session_step(0, pending);
      pending = argmax(o.logits.data(), V);
      tc.tokens.push_back(pending);
      tc.rows.push_back(o.logits);
      tc.sels.push_back(o.dsa_selections);
      tc.states.push_back(o.layer_states);
    }
    unsetenv("DGPP_DSV41_CAPTURE_DECODE");
    c.session_close(0);
    // The chunked state's rows start from the certified chunked prefill
    // (its last row 0.033 from the one-shot's), so its clean rows hold the
    // chunked budget rather than the fresh-session one.
    audit(m, Cm, tc, "decode after the chunked prefill", 1e-1, /*clean_budget=*/1e-1);  // the chunked state's cascade (the long audit's budget)
    std::printf("[ OK ] chunked prefill and its decode agree with the one-shot walk\n");
  }

  // 5. Close and reopen restarts bitwise; the pool's accounting.
  {
    const Dsv41Model::Outputs p = m.session_prefill(1, A);
    require(bitwise(p.logits, tA.rows[0]), "reopen: the prefill differs");
    const Dsv41Model::Outputs o = m.session_step(1, tA.tokens[0]);
    require(bitwise(o.logits, tA.rows[1]), "reopen: the step differs");
    m.session_close(1);
    require(m.kv_blocks_in_use() == 0, "reopen: every block released");
    std::printf("[ OK ] a closed slot reopens bitwise; the pool is empty after the close\n");
  }

  // 6. The prefix cache: hot == cold bitwise. A snapshot at the pool-
  //    aligned cut 256 of the 300-token prompt (the rings, the compressor
  //    tails and the Engram context at the cut), attached in another
  //    slot, the suffix resumed; then a mid-decode snapshot at an aligned
  //    position (the 40-token prompt + 88 steps = 128).
  {
    const std::vector<int64_t> bounds{2 * block};
    std::vector<uint8_t*> arena(2, nullptr);
    const size_t bytes = m.session_snapshot_bytes();
    require(bytes > 0, "prefix: the snapshot has bytes");
    for (uint8_t*& p : arena) require(cudaMalloc(reinterpret_cast<void**>(&p), bytes) == cudaSuccess, "prefix: arena");
    Dsv41Model::SessionSnapshotMeta meta;
    Dsv41Model::SnapshotRequest snap;
    snap.position = 2 * block;
    snap.dst = arena[0];
    snap.meta = &meta;
    const Dsv41Model::Outputs cold = m.session_prefill(0, Cl, bounds, &snap);
    require(snap.taken && meta.position == 2 * block, "prefix: the snapshot was taken at the cut");
    const Dsv41Model::Outputs cold_step = m.session_step(0, 7);
    const Dsv41Model::Outputs cold_step2 = m.session_step(0, 11);
    m.session_close(0);
    m.session_attach(1, arena[0], meta);
    require(m.session_position(1) == 2 * block, "prefix: attached at the snapshot position");
    const Dsv41Model::Outputs hot =
        m.session_prefill_resume(1, std::vector<int64_t>(Cl.begin() + 2 * block, Cl.end()), bounds);
    require(bitwise(hot.logits, cold.logits), "prefix: the hot prefill's last row differs from the cold one's");
    const Dsv41Model::Outputs hot_step = m.session_step(1, 7);
    require(bitwise(hot_step.logits, cold_step.logits), "prefix: the first step after the attach differs");
    const Dsv41Model::Outputs hot_step2 = m.session_step(1, 11);
    require(bitwise(hot_step2.logits, cold_step2.logits), "prefix: the second step after the attach differs");
    m.session_close(1);
    // Mid-decode at an aligned position: 40 + 88 steps = 128.
    Dsv41Model::Outputs o = m.session_prefill(1, A);
    int64_t pending = argmax(o.logits.data(), V);
    for (int s = 0; s < block - static_cast<int>(A.size()); ++s) {
      o = m.session_step(1, pending);
      pending = argmax(o.logits.data(), V);
    }
    require(m.session_position(1) == block, "prefix: position 128");
    Dsv41Model::SessionSnapshotMeta meta2 = m.session_snapshot(1, arena[1]);
    const Dsv41Model::Outputs cont = m.session_step(1, pending);
    const Dsv41Model::Outputs cont2 = m.session_step(1, 5);
    m.session_close(1);
    m.session_attach(0, arena[1], meta2);
    const Dsv41Model::Outputs re = m.session_step(0, pending);
    require(bitwise(re.logits, cont.logits), "prefix: the step after a mid-decode attach differs");
    const Dsv41Model::Outputs re2 = m.session_step(0, 5);
    require(bitwise(re2.logits, cont2.logits), "prefix: the second step after a mid-decode attach differs");
    m.session_close(0);
    m.session_release_snapshot(meta);
    m.session_release_snapshot(meta2);
    require(m.kv_blocks_in_use() == 0, "prefix: every block released with the entries");
    for (uint8_t* p : arena) cudaFree(p);
    std::printf("[ OK ] prefix snapshots: hot == cold bitwise at a cut and mid-decode\n");
  }

  // 7. The DSpark block eagerly (the greedy speculator at the full block):
  //    the committed transcript is the plain greedy one exactly; a forced
  //    correct first draft is accepted.
  {
    Dsv41Model d(cfg, dir, /*max_tokens=*/512, /*max_cache_tokens=*/1024, Dsv41Residency::Resident, nullptr, 0, 1,
                 /*max_requests=*/2, /*mtp=*/true, /*decode_rows=*/1 + cfg.dspark_block_size);
    require(d.mtp_enabled(), "mtp: enabled");
    const auto pick_rows = [](const std::vector<dgpp::sample::Candidate>& c) {
      std::vector<int32_t> ids;
      for (const auto& x : c) ids.push_back(x.id);
      return ids;
    };
    for (int round = 0; round < 2; ++round) {
      const Dsv41Model::Outputs p = d.session_prefill(1, A);
      require(bitwise(p.logits, tA.rows[0]), "mtp: the prefill's last row differs from the plain model's");
      require(d.session_draft_position(1) == static_cast<int64_t>(A.size()) - 1, "mtp: the draft trails by one after the prefill");
      dgpp::GreedySpeculator<Dsv41Model> spec(d, 1, pick_rows, cfg.dspark_block_size);
      spec.start(argmax(p.logits.data(), V), static_cast<int32_t>(tA.tokens[1]));
      std::vector<int64_t> committed;
      int first_step_committed = 0;
      while (committed.size() < tA.tokens.size()) {
        const std::vector<int32_t> got = spec.step();
        require(!got.empty(), "mtp: a step commits at least one token");
        if (spec.steps() == 1) first_step_committed = static_cast<int>(got.size());
        for (const int32_t t : got) committed.push_back(t);
      }
      committed.resize(tA.tokens.size());
      require(std::equal(committed.begin(), committed.end(), tA.tokens.begin()),
              "mtp: the speculative transcript differs from the plain greedy one: " + ids_text(committed));
      require(first_step_committed >= 2, "mtp: the forced correct draft was not accepted");
      std::printf("[ .. ] mtp round %d: %d steps for %zu tokens, %d drafts accepted (the forced one included)\n", round,
                  spec.steps(), committed.size(), spec.accepted_drafts());
      d.session_close(1);
    }
    std::printf("[ OK ] the eager speculator reproduces the greedy transcript through the DSpark block\n");
  }
  // 7b. Confidence-scheduled verify depth (DSpark; scheduled_verify_depth) is
  //     EXACT at any per-step depth. The block always drafts its full width,
  //     but only a policy-chosen prefix of the drafts is fed to the verify;
  //     a draft not verified is decoded plainly next step. So the committed
  //     transcript is the plain greedy one whatever depth each step picks —
  //     the exactness the graph-engine port relies on, proven on the model
  //     code (draft conditioning + rollback), not only argued. Three
  //     policies must all reproduce tA: full depth, adversarially shallow
  //     (one draft/step), and the confidence policy over the head's own
  //     logits (arbitrary on this random fixture — it exercises the
  //     debug_confidence read and, above all, exactness).
  {
    Dsv41Model d(cfg, dir, /*max_tokens=*/512, /*max_cache_tokens=*/1024, Dsv41Residency::Resident, nullptr, 0, 1,
                 /*max_requests=*/2, /*mtp=*/true, /*decode_rows=*/1 + cfg.dspark_block_size);
    require(d.mtp_enabled(), "sched: mtp enabled");
    const auto pick_rows = [](const std::vector<dgpp::sample::Candidate>& c) {
      std::vector<int32_t> ids;
      for (const auto& x : c) ids.push_back(x.id);
      return ids;
    };
    const int blk = cfg.dspark_block_size;  // the block width == the speculator depth
    auto run_policy = [&](const std::function<int(const std::vector<int32_t>&)>& policy,
                          double* mean_depth) {
      const Dsv41Model::Outputs p = d.session_prefill(1, A);
      dgpp::GreedySpeculator<Dsv41Model> spec(d, 1, pick_rows, blk);
      if (policy) spec.set_depth_policy(policy);
      spec.start(argmax(p.logits.data(), V));
      std::vector<int64_t> committed;
      long depth_sum = 0;
      int nsteps = 0;
      while (committed.size() < tA.tokens.size()) {
        const std::vector<int32_t> got = spec.step();
        require(!got.empty(), "sched: a step commits at least one token");
        depth_sum += spec.last_verify_depth();
        ++nsteps;
        for (const int32_t t : got) committed.push_back(t);
      }
      committed.resize(tA.tokens.size());
      d.session_close(1);
      if (mean_depth) *mean_depth = nsteps ? static_cast<double>(depth_sum) / nsteps : 0.0;
      return committed;
    };
    double md_full = 0, md_one = 0, md_conf = 0;
    const auto full = run_policy(nullptr, &md_full);
    require(std::equal(full.begin(), full.end(), tA.tokens.begin()),
            "sched: full-depth transcript differs from plain greedy: " + ids_text(full));
    const auto one = run_policy([](const std::vector<int32_t>&) { return 1; }, &md_one);
    require(std::equal(one.begin(), one.end(), tA.tokens.begin()),
            "sched: shallow (k=1) transcript differs from plain greedy: " + ids_text(one));
    std::vector<float> conf(static_cast<size_t>(blk));
    const auto conf_policy = [&](const std::vector<int32_t>&) {
      // conf_ is [max_requests, block]; slot 1's block starts at 1 * block.
      const cudaError_t e = cudaMemcpy(conf.data(), d.debug_confidence() + static_cast<size_t>(blk),
                                       sizeof(float) * static_cast<size_t>(blk), cudaMemcpyDeviceToHost);
      require(e == cudaSuccess, "sched: confidence copy failed");
      const float lam = dgpp::verify_reservation_lambda(20.0f, 9.0f);  // profiled curve
      return dgpp::scheduled_verify_depth(conf.data(), blk, /*row_ms=*/9.0f, lam);
    };
    const auto sched = run_policy(conf_policy, &md_conf);
    require(std::equal(sched.begin(), sched.end(), tA.tokens.begin()),
            "sched: confidence-scheduled transcript differs from plain greedy: " + ids_text(sched));
    require(md_full > md_one, "sched: k=1 must verify fewer rows than full depth");
    std::printf("[ OK ] scheduled verify depth is exact at any per-step depth "
                "(mean verified depth: full %.2f, k=1 %.2f, confidence %.2f)\n",
                md_full, md_one, md_conf);
  }
  // 8. The bounded prefill (plan §1.8, the production mode): the decoder
  //    over the call's last `window` rows (the segment), the decoder's
  //    global KV published for every row.
  {
    const std::vector<int64_t> S = smoke_tokens(cfg, 12, 0x3C6EF372FE94F82Bull);  // within the window
    const int win = cfg.sliding_window;
    require(win < block && static_cast<int>(S.size()) <= win, "bounded: the fixture's window vs the prompts");
    const size_t ms = static_cast<size_t>(cfg.index_topk);
    // 8a. A prompt within the window walks the same rows with the window
    //     floored at 0 (the call's start): bitwise the exact mode's, and
    //     so is the decode from that state.
    {
      const Transcript te = greedy(m, 0, S, 8);
      m.session_close(0);
      m.set_prefill_bounded(true);
      const Transcript tb = greedy(m, 0, S, 8);
      m.session_close(0);
      m.set_prefill_bounded(false);
      require(tb.tokens == te.tokens, "bounded: a prompt within the window decodes differently");
      for (size_t i = 0; i < te.rows.size(); ++i)
        require(bitwise(tb.rows[i], te.rows[i]),
                "bounded: row " + std::to_string(i) + " of a prompt within the window is not the exact mode's");
      std::printf("[ OK ] bounded prefill: a %zu-token prompt (within the window) and its decode are bitwise the exact mode's\n",
                  S.size());
    }
    // 8b. Beyond the window: the one-shot bounded walk vs the exact walk
    //     (informational: the approximation's own distance), then the
    //     chunked bounded walk — the first chunk's encoder rows carried as
    //     the tail, the segment assembled on the last chunk — against the
    //     one-shot: every segment row's selection equal or a flip certified
    //     by the measured logit deviation (the chunked gate's rule), the
    //     last row within the chunked budget; then 24 decode steps from
    //     both states fed the one-shot's tokens: the rows whose selections
    //     agree hold the long-audit budget (a row whose selections differ
    //     inherits the certified prefill difference and is exempt).
    const Dsv41Model::Outputs one_exact = m.session_prefill(0, Cm);
    m.session_close(0);
    m.set_prefill_bounded(true);
    {
      const Dsv41Model::Outputs one_f = m.forward(Cm, true);  // the one-shot walk's captures (a cold slot)
      const std::vector<Dsv41Model::IndexLogits> one_logits = m.debug_index_logits();
      const Dsv41Model::Outputs one = m.session_prefill(0, Cm);
      {
        const RowCompare r = compare_row(one.logits.data(), one_exact.logits.data(), V);
        std::printf("[ .. ] bounded vs exact on the %zu-token prompt (segment of %d rows): last-row relative l2 %.3g, top-1 %s\n",
                    Cm.size(), win, r.l2, r.top1_equal ? "equal" : (r.near_tie ? "near tie" : "different"));
      }
      Dsv41Model cb(cfg, dir, /*max_tokens=*/block, /*max_cache_tokens=*/1024, Dsv41Residency::Resident, nullptr, 0, 1,
                    /*max_requests=*/1);
      cb.set_prefill_bounded(true);
      setenv("DGPP_DSV41_CAPTURE_PREFILL", "1", 1);
      const Dsv41Model::Outputs p0 = cb.session_prefill(0, std::vector<int64_t>(Cm.begin(), Cm.begin() + block));
      const std::vector<Dsv41Model::IndexLogits> chunk0_logits = cb.debug_index_logits();  // the encoder sources' rows are chunk 0's
      cb.session_close(0);
      const Dsv41Model::Outputs p = cb.session_prefill(0, Cm);
      const std::vector<Dsv41Model::IndexLogits> chunk1_logits = cb.debug_index_logits();
      unsetenv("DGPP_DSV41_CAPTURE_PREFILL");
      (void)p0;
      require(p.dsa_selections.size() == one_f.dsa_selections.size() && p.dsa_selections.size() == one_logits.size(),
              "bounded chunked: selection captures");
      int flips = 0, unc = 0, first_flip = -1;
      double worst_dev = 0, worst_gap = 0;
      const int dec0 = cfg.decoder_first_layer();
      for (size_t l = 0; l < p.dsa_selections.size(); ++l) {
        const int layer = cfg.index_source_layer_ids[l];
        const size_t rows = p.dsa_selections[l].size() / ms;
        require(rows == (layer >= dec0 ? static_cast<size_t>(win) : Cm.size()) && one_f.dsa_selections[l].size() == rows * ms,
                "bounded chunked: source " + std::to_string(l) + " rows");
        for (size_t row = 0; row < rows; ++row) {
          const int32_t* g = p.dsa_selections[l].data() + row * ms;
          const int32_t* r = one_f.dsa_selections[l].data() + row * ms;
          if (std::memcmp(g, r, ms * 4) == 0) continue;
          const size_t position = Cm.size() - rows + row;
          ++flips;
          if (first_flip < 0 || static_cast<int>(position) < first_flip) first_flip = static_cast<int>(position);
          const double gap = source_flip_gap(one_logits[l], r, g, row, ms);
          // The chunked path's own logits of the row: an encoder source's
          // first-chunk rows are the 128-row prefill's, the rest the last
          // chunk's (the segment rows at its segment-relative index).
          const bool in_chunk0 = layer < dec0 && position < static_cast<size_t>(block);
          const Dsv41Model::IndexLogits& mine = in_chunk0 ? chunk0_logits[l] : chunk1_logits[l];
          const size_t my_row = layer >= dec0 ? row : (in_chunk0 ? position : position - static_cast<size_t>(block));
          const double dev = logit_deviation(one_logits[l], row, mine, my_row);
          worst_dev = std::max(worst_dev, dev);
          worst_gap = std::max(worst_gap, gap);
          if (gap > 2.0 * dev + 2e-3) {
            ++unc;
            std::printf("[ .. ] bounded chunked: UNCERTIFIED flip at position %zu (source %zu): gap %.3g against a logit deviation of %.3g\n",
                        position, l, gap, dev);
          }
        }
      }
      const RowCompare r = compare_row(p.logits.data(), one.logits.data(), V);
      std::printf("[ .. ] bounded chunked prefill (%d-row chunks, a %d-row tail): relative l2 %.3g, top-1 %s; %d selection flips vs "
                  "the one-shot (the first at position %d; the widest gap %.3g, the logit deviation up to %.3g; %d uncertified)\n",
                  block, win - static_cast<int>(Cm.size()) + block, r.l2, r.top1_equal ? "equal" : (r.near_tie ? "near tie" : "MISMATCH"),
                  flips, first_flip, worst_gap, worst_dev, unc);
      require(unc == 0, "bounded chunked: a selection flip beyond the measured logit deviation");
      require(r.top1_equal || r.near_tie, "bounded chunked: top-1 mismatch");
      require(r.l2 < (first_flip < 0 || first_flip == static_cast<int>(Cm.size()) - 1 ? 2e-2 : 1e-1),
              "bounded chunked: relative l2 over budget");
      // The decodes, the chunked one fed the one-shot's tokens.
      const Transcript to = decode_rows(m, 0, one.logits, 24);
      m.session_close(0);
      const Transcript tc = decode_rows(cb, 0, p.logits, 24, &to.tokens);
      cb.session_close(0);
      int kept = 0, differ = 0, hard = 0, soft = 0;
      double worst = 0;
      for (size_t i = 1; i < to.rows.size(); ++i) {
        bool same_sel = to.sels[i].size() == tc.sels[i].size();
        for (size_t l = 0; same_sel && l < to.sels[i].size(); ++l) same_sel = to.sels[i][l] == tc.sels[i][l];
        if (!same_sel) { ++differ; continue; }
        ++kept;
        const RowCompare rc = compare_row(tc.rows[i].data(), to.rows[i].data(), V);
        worst = std::max(worst, rc.l2);
        if (!rc.top1_equal) (rc.near_tie ? soft : hard) += 1;
      }
      std::printf("[ .. ] decode after the bounded chunked prefill vs after the one-shot (24 rows, the same tokens): %d rows with equal "
                  "selections — worst relative l2 %.3g, top-1 hard %d near-tie %d; %d rows with a selection difference exempt\n",
                  kept, worst, hard, soft, differ);
      require(hard == 0, "bounded chunked decode: a top-1 mismatch beyond the near-tie margin");
      require(worst < 1e-1, "bounded chunked decode: relative l2 over the long-audit budget");
      require(kept >= 8, "bounded chunked decode: too few rows with equal selections");
      std::printf("[ OK ] the bounded chunked prefill (with a tail) and its decode agree with the one-shot bounded walk\n");
    }
    // 8c. The prefix cache in bounded mode: a snapshot position closes a
    //     span (the decoder runs over the rows up to it, so the saved
    //     state is complete) and the next chunk opens one; hot == cold
    //     bitwise at the cut and through two steps.
    {
      const std::vector<int64_t> bounds{2 * block};
      uint8_t* arena = nullptr;
      const size_t bytes = m.session_snapshot_bytes();
      require(cudaMalloc(reinterpret_cast<void**>(&arena), bytes) == cudaSuccess, "bounded prefix: arena");
      Dsv41Model::SessionSnapshotMeta meta;
      Dsv41Model::SnapshotRequest snap;
      snap.position = 2 * block;
      snap.dst = arena;
      snap.meta = &meta;
      const Dsv41Model::Outputs cold = m.session_prefill(0, Cl, bounds, &snap);
      require(snap.taken && meta.position == 2 * block, "bounded prefix: the snapshot was taken at the cut");
      const Dsv41Model::Outputs cold_step = m.session_step(0, 7);
      const Dsv41Model::Outputs cold_step2 = m.session_step(0, 11);
      m.session_close(0);
      m.session_attach(1, arena, meta);
      const Dsv41Model::Outputs hot =
          m.session_prefill_resume(1, std::vector<int64_t>(Cl.begin() + 2 * block, Cl.end()), bounds);
      require(bitwise(hot.logits, cold.logits), "bounded prefix: the hot prefill's last row differs from the cold one's");
      require(bitwise(m.session_step(1, 7).logits, cold_step.logits), "bounded prefix: the first step after the attach differs");
      require(bitwise(m.session_step(1, 11).logits, cold_step2.logits), "bounded prefix: the second step after the attach differs");
      m.session_close(1);
      m.session_release_snapshot(meta);
      require(m.kv_blocks_in_use() == 0, "bounded prefix: every block released");
      cudaFree(arena);
      std::printf("[ OK ] bounded prefix snapshots: hot == cold bitwise at a cut and through two steps\n");
    }
    // 8d. The DSpark speculator after a bounded prefill of the 300-token
    //     prompt (the draft rows are the segment's): the committed
    //     transcript is the plain bounded greedy one; the forced first
    //     draft is accepted.
    {
      const Transcript tl = greedy(m, 0, Cl, 24);
      m.session_close(0);
      Dsv41Model d(cfg, dir, /*max_tokens=*/512, /*max_cache_tokens=*/1024, Dsv41Residency::Resident, nullptr, 0, 1,
                   /*max_requests=*/2, /*mtp=*/true, /*decode_rows=*/1 + cfg.dspark_block_size);
      d.set_prefill_bounded(true);
      const auto pick_rows = [](const std::vector<dgpp::sample::Candidate>& c) {
        std::vector<int32_t> ids;
        for (const auto& x : c) ids.push_back(x.id);
        return ids;
      };
      const Dsv41Model::Outputs p = d.session_prefill(1, Cl);
      require(bitwise(p.logits, tl.rows[0]), "bounded mtp: the prefill's last row differs from the plain model's");
      require(d.session_draft_position(1) == static_cast<int64_t>(Cl.size()) - 1, "bounded mtp: the draft trails by one");
      dgpp::GreedySpeculator<Dsv41Model> spec(d, 1, pick_rows, cfg.dspark_block_size);
      spec.start(argmax(p.logits.data(), V), static_cast<int32_t>(tl.tokens[1]));
      std::vector<int64_t> committed;
      int first_step_committed = 0;
      while (committed.size() < tl.tokens.size()) {
        const std::vector<int32_t> got = spec.step();
        require(!got.empty(), "bounded mtp: a step commits at least one token");
        if (spec.steps() == 1) first_step_committed = static_cast<int>(got.size());
        for (const int32_t t : got) committed.push_back(t);
      }
      committed.resize(tl.tokens.size());
      require(std::equal(committed.begin(), committed.end(), tl.tokens.begin()),
              "bounded mtp: the speculative transcript differs from the plain greedy one: " + ids_text(committed));
      require(first_step_committed >= 2, "bounded mtp: the forced correct draft was not accepted");
      d.session_close(1);
      std::printf("[ OK ] the eager speculator after a bounded prefill reproduces the bounded greedy transcript (%d steps, %d drafts accepted)\n",
                  spec.steps(), spec.accepted_drafts());
    }
    m.set_prefill_bounded(false);
  }
  std::printf("[ OK ] dsv41_decode_test\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string fixture;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--fixture" && i + 1 < argc) fixture = argv[++i];
  }
  if (fixture.empty()) {
    std::fprintf(stderr, "usage: dsv41_decode_test --fixture DIR\n");
    return 2;
  }
  try {
    return run_fixture(fixture);
  } catch (const std::exception& e) {
    std::printf("[FAIL] %s\n", e.what());
    return 1;
  }
}
