// The MiMo-V2.6-Flash session surface's gate (2026-09-22,
// docs/mimo_v26_flash_plan.md G4): incremental decode over MimoModel's
// request slots against the cold re-forward, the prefix snapshots (any
// position: no recurrent state), the verify rows against the scalar steps,
// the eager speculator through the draft block, and the sliding window at
// the session level.
//
//   --fixture DIR                the fixture gates (tests/cuda/mimo_fixture.hpp's
//                                checkpoint, written by mimo_forward_test; absent:
//                                written into ./mimo_decode_fixture)
//   --checkpoint-dir DIR --ids 1,2,... [--steps N]
//                                the real checkpoint (streaming, world 1): a
//                                greedy transcript and its re-forward audit
//
// Gates on the fixture: a one-shot prefill's last row is bitwise the cold
// forward's (the same m=T launches on the same state); T=1 steps agree with
// the re-forward's rows at every position under the near-tie rule (their
// m=1 launches reassociate); a verify of T rows reproduces the T scalar
// steps' rows — bitwise up to the GEMV chunk's four rows (the fp8 sites'
// streaming form and the bf16 sites' GEMV chunks keep a row's chain
// whatever rows share the launch), tolerance-equal above (the bf16 sites
// take cuBLASLt there) — and a rollback restarts the scalar chain bitwise;
// two interleaved slots reproduce their solo runs bitwise; a chunked
// prefill (chunk == max_tokens, pool-aligned boundary cuts) agrees with the
// one-shot; a closed and reopened slot restarts bitwise; the pool's block
// accounting; the draft block through the greedy speculator; and the
// window: on a one-SWA-layer stack, a change to token 0 of a 40-token
// prompt leaves the rows 32 and later of that layer's residual bitwise
// (they see [pos - 31, pos]) while every earlier row moves.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "engine/speculative.hpp"
#include "models/mimo/config.hpp"
#include "kernels/latent_format.hpp"
#include "models/mimo/forward.hpp"
#include "mimo_fixture.hpp"

namespace fs = std::filesystem;
using dgpp::MimoModel;
using dgpp::MimoResidency;
using dgpp::MimoTextConfig;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

std::vector<int64_t> smoke_tokens(const MimoTextConfig& cfg, int n, uint64_t seed) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = seed;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    t[static_cast<size_t>(i)] = static_cast<int64_t>(s % static_cast<uint64_t>(cfg.vocab_size));
  }
  if (n > 9) t[9] = cfg.eos_token_ids.empty() ? 0 : cfg.eos_token_ids[0];
  return t;
}

int32_t argmax(const float* row, int n) {
  int32_t best = 0;
  for (int i = 1; i < n; ++i)
    if (row[i] > row[best]) best = i;
  return best;
}

struct RowCompare {
  double l2 = 0;        // relative l2 of the row
  bool top1_equal = false;
  bool near_tie = false;  // the reference's top-1 and the candidate within 2 %
};

// The re-forward's row `want` against `got`: relative l2, the top-1 with
// near-tie certification (the forward test's rule).
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
// argmax; every step's logits row is kept, with the row's routing (every
// MoE layer's K expert ids: the prefill's last row, then each step's).
struct Transcript {
  std::vector<int64_t> tokens;             // generated ids (the prefill's pick first)
  std::vector<std::vector<float>> rows;    // the prefill's row, then every step's
  std::vector<std::vector<int32_t>> routes;  // per row: [moe layers x K] ids
};

// The last row's routing of an outputs record (a prefill's rows or a
// decode step's one row) as one flat [layers x K] vector.
std::vector<int32_t> last_row_routes(const MimoModel::Outputs& o, int K) {
  std::vector<int32_t> r;
  for (const auto& layer : o.route_ids) {
    if (layer.size() < static_cast<size_t>(K)) continue;
    r.insert(r.end(), layer.end() - K, layer.end());
  }
  return r;
}

void push_row(Transcript& t, const MimoModel::Outputs& o, int K) {
  t.tokens.push_back(argmax(o.logits.data(), o.lm_vocab_count));
  t.rows.push_back(o.logits);
  t.routes.push_back(last_row_routes(o, K));
}

Transcript greedy(MimoModel& m, int req, const std::vector<int64_t>& prompt, int steps) {
  const int K = m.config().num_experts_per_tok;
  Transcript t;
  MimoModel::Outputs o = m.session_prefill(req, prompt);
  push_row(t, o, K);
  for (int s = 0; s < steps; ++s) {
    o = m.session_step(req, t.tokens.back());
    push_row(t, o, K);
  }
  return t;
}

std::string ids_text(const std::vector<int64_t>& ids) {
  std::string s;
  for (size_t i = 0; i < ids.size(); ++i) s += (i ? "," : "") + std::to_string(ids[i]);
  return s;
}

// The transcript's rows against the cold re-forward of prompt + tokens:
// row P-1+i of the re-forward is step i's input at the same position. A
// row the decode path ROUTED differently from the re-forward (a near tie
// of the sigmoid router: the decode row's chain differs from the prefill
// row's at rounding level, and on this fixture's 2-of-8 routing without a
// shared expert a flip moves the row's logits by up to half their norm)
// is a certified flip: exempt from the l2 and top-1 rules, counted and
// bounded — every other row must hold both. The routing comes from the
// decode rows' own traces (models/mimo/forward.cpp: the eager walk stages
// them while route_traces_ is on).
int audit(MimoModel& ref, const std::vector<int64_t>& prompt, const Transcript& t, const char* what,
          double l2_budget, int max_flips = 3) {
  std::vector<int64_t> all(prompt);
  all.insert(all.end(), t.tokens.begin(), t.tokens.end() - 1);
  const MimoModel::Outputs f = ref.forward(all);
  const int V = f.lm_vocab_count;
  const int K = ref.config().num_experts_per_tok;
  const size_t P = prompt.size();
  int hard = 0, soft = 0, flips = 0;
  double worst_l2 = 0, worst_clean_l2 = 0;
  std::string profile;
  for (size_t i = 0; i < t.rows.size(); ++i) {
    const float* want = f.logits.data() + (P - 1 + i) * static_cast<size_t>(V);
    const RowCompare c = compare_row(t.rows[i].data(), want, V);
    // The row's routing against the re-forward's row at the same position.
    std::vector<int32_t> ref_routes;
    for (const auto& layer : f.route_ids)
      ref_routes.insert(ref_routes.end(), layer.begin() + static_cast<std::ptrdiff_t>((P - 1 + i) * static_cast<size_t>(K)),
                        layer.begin() + static_cast<std::ptrdiff_t>((P + i) * static_cast<size_t>(K)));
    // A row without a trace (a group prefill's span row) keeps the strict rule.
    const bool flipped = t.routes[i].size() == ref_routes.size() && t.routes[i] != ref_routes;
    worst_l2 = std::max(worst_l2, c.l2);
    if (flipped) {
      ++flips;
      char buf[96];
      std::snprintf(buf, sizeof(buf), " p%zu:%.3f(flip%s)", P + i, c.l2, c.top1_equal ? "" : c.near_tie ? ",near tie" : ",top-1");
      profile += buf;
      continue;
    }
    worst_clean_l2 = std::max(worst_clean_l2, c.l2);
    if (!c.top1_equal) (c.near_tie ? soft : hard) += 1;
    if (t.rows.size() > 16 && (i % 4 == 0 || c.l2 > 0.02)) {
      char buf[48];
      std::snprintf(buf, sizeof(buf), " p%zu:%.3f", P + i, c.l2);
      profile += buf;
    }
  }
  if (!profile.empty()) std::printf("[ .. ] %s: per-position relative l2%s\n", what, profile.c_str());
  std::printf("[ .. ] %s: %zu rows vs the re-forward — worst relative l2 %.3g (%.3g outside %d routing flips), top-1 hard %d "
              "near-tie %d\n",
              what, t.rows.size(), worst_l2, worst_clean_l2, flips, hard, soft);
  require(hard == 0, std::string(what) + ": a top-1 mismatch beyond the near-tie margin");
  require(worst_clean_l2 < l2_budget, std::string(what) + ": relative l2 over budget");
  require(flips <= max_flips, std::string(what) + ": too many routing flips against the re-forward");
  return soft;
}

// The sliding window at the session level: a stack of one SWA dense layer
// and one GA MoE layer (hybrid_layer_pattern [1, 0]), a 40-token prompt
// against the same prompt with token 0 changed. Layer 0's residual: rows 32
// .. 39 attend [pos - 31, pos] and must be bitwise unmoved; rows 1 .. 31
// see token 0 and move (row 0 is the changed embedding itself). Layer 1
// (global) moves on every row — the control that the change reached it.
void run_window_gate(const std::string& base_dir) {
  std::string json = mimofx::tiny_config_json(/*layers=*/2, /*mtp=*/false);
  const std::string from = "\"hybrid_layer_pattern\": [0, 0]";
  const size_t at = json.find(from);
  require(at != std::string::npos, "window gate: the two-layer fixture config's pattern");
  json.replace(at, from.size(), "\"hybrid_layer_pattern\": [1, 0]");
  const MimoTextConfig cfg = MimoTextConfig::parse(dgpp::minijson::parse(json).root);
  require(cfg.is_swa_layer(0) && !cfg.is_swa_layer(1) && cfg.sliding_window == 32, "window gate: the config");
  const std::string dir = (fs::path(base_dir) / "mimo_window_fixture").string();
  mimofx::write_fixture(cfg, dir, json.c_str());
  const int T = 40, W = cfg.sliding_window, H = cfg.hidden_size;
  std::vector<int64_t> a = smoke_tokens(cfg, T, 0x5851F42D4C957F2Dull);
  std::vector<int64_t> b = a;
  b[0] = (a[0] + 7) % cfg.vocab_size;
  MimoModel m(cfg, dir, /*max_tokens=*/T, /*max_cache_tokens=*/128, MimoResidency::Resident, nullptr, 0, 1, 1);
  m.set_decode_row_traces(true);
  const MimoModel::Outputs fa = m.forward(a, true);
  const MimoModel::Outputs fb = m.forward(b, true);
  require(fa.layer_states.size() == 2 && fb.layer_states.size() == 2, "window gate: two layer captures");
  int moved_early = 0, moved_late = 0, moved_global = 0;
  for (int t = 0; t < T; ++t) {
    const bool same0 = std::memcmp(fa.layer_states[0].data() + static_cast<size_t>(t) * H,
                                   fb.layer_states[0].data() + static_cast<size_t>(t) * H, static_cast<size_t>(H) * 2) == 0;
    const bool same1 = std::memcmp(fa.layer_states[1].data() + static_cast<size_t>(t) * H,
                                   fb.layer_states[1].data() + static_cast<size_t>(t) * H, static_cast<size_t>(H) * 2) == 0;
    if (t < W) moved_early += !same0;
    else moved_late += !same0;
    moved_global += !same1;
  }
  std::printf("[ .. ] window gate: layer 0 (SWA, window %d) rows moved by a token-0 change: %d of %d before the window's "
              "reach, %d of %d past it; layer 1 (global): %d of %d\n",
              W, moved_early, W, moved_late, T - W, moved_global, T);
  require(moved_late == 0, "window gate: a row past the window saw token 0 through the SWA layer");
  require(moved_early >= W - 2, "window gate: the rows inside the window did not see token 0");
  require(moved_global >= T - 2, "window gate: the global layer did not see the change");
  std::printf("[ OK ] the sliding window is applied through the session's paged cache (rows %d.. unmoved)\n", W);
}

// ---- the fixture gates ----------------------------------------------------------
int run_fixture(const std::string& dir) {
  const MimoTextConfig cfg = MimoTextConfig::from_json_file((fs::path(dir) / "config.json").string());
  const std::vector<int64_t> A = smoke_tokens(cfg, 23, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 17, 0xD1B54A32D192ED03ull);
  MimoModel m(cfg, dir, /*max_tokens=*/64, /*max_cache_tokens=*/256, MimoResidency::Resident, nullptr,
              0, 1, /*max_requests=*/2);
  m.set_decode_row_traces(true);
  const int V = m.lm_vocab_count();

  // 1. prefill == forward, bitwise.
  {
    const MimoModel::Outputs f = m.forward(A);
    const MimoModel::Outputs p = m.session_prefill(0, A);
    require(p.logits.size() == static_cast<size_t>(V), "prefill: one row of logits");
    const std::vector<float> last(f.logits.end() - V, f.logits.end());
    require(bitwise(p.logits, last), "prefill's last row is not bitwise the forward's");
    require(std::equal(p.final_hidden_bits.begin(), p.final_hidden_bits.end(),
                       f.final_hidden_bits.end() - cfg.hidden_size),
            "prefill's final hidden is not bitwise the forward's");
    require(m.session_position(0) == static_cast<int64_t>(A.size()), "prefill: position");
    m.session_close(0);
    require(m.session_position(0) == 0, "close: position");
    std::printf("[ OK ] prefill last row bitwise the cold forward (%zu tokens)\n", A.size());
  }
  // 1b. A group prefill — A and B as the spans of one walk — against the
  //     prefills alone: the attention rows carry their own request ids and
  //     positions; the dense sites see 40 rows instead of 23 and 17 (the
  //     fp8 GEMV chunks / cuBLASLt at each), so the rows are
  //     tolerance-equal under the row compare's l2 and near-tie rule, and a
  //     decode off the group's cache is audited against the re-forward
  //     like any other.
  {
    const MimoModel::Outputs pa = m.session_prefill(0, A);
    m.session_close(0);
    const MimoModel::Outputs pb = m.session_prefill(1, B);
    m.session_close(1);
    const std::vector<MimoModel::Outputs> g = m.session_prefill_group({0, 1}, {&A, &B});
    require(g.size() == 2 && g[0].logits.size() == static_cast<size_t>(V) && g[1].logits.size() == static_cast<size_t>(V),
            "group prefill: one row of logits per span");
    require(m.session_position(0) == static_cast<int64_t>(A.size()) && m.session_position(1) == static_cast<int64_t>(B.size()),
            "group prefill: positions");
    const RowCompare ca = compare_row(g[0].logits.data(), pa.logits.data(), V);
    const RowCompare cb = compare_row(g[1].logits.data(), pb.logits.data(), V);
    std::printf("[ .. ] group prefill (23 + 17 rows) vs the prefills alone: relative l2 %.3g / %.3g, top-1 %s / %s\n", ca.l2, cb.l2,
                ca.top1_equal ? "equal" : ca.near_tie ? "near tie" : "DIFFERS", cb.top1_equal ? "equal" : cb.near_tie ? "near tie" : "DIFFERS");
    require((ca.top1_equal || ca.near_tie) && (cb.top1_equal || cb.near_tie), "group prefill: a top-1 mismatch beyond the near-tie margin");
    require(ca.l2 < 1e-1 && cb.l2 < 1e-1, "group prefill: relative l2 over budget");
    Transcript tb;
    push_row(tb, g[1], cfg.num_experts_per_tok);
    for (int s = 0; s < 8; ++s) {
      const MimoModel::Outputs o = m.session_step(1, tb.tokens.back());
      push_row(tb, o, cfg.num_experts_per_tok);
    }
    m.session_close(0);
    m.session_close(1);
    const int soft = audit(m, B, tb, "decode after the group prefill", 1e-1);
    std::printf("[ OK ] group prefill: the spans' rows and an 8-step decode off the group's cache (%d near ties)\n", soft);
  }

  // 2. Incremental decode vs the re-forward — 60 steps from the 23-token
  //    prompt, so the decode rows cross the pool's 64-token block boundary
  //    (position 64 lands on step 42) and every SWA row's window (32) slides
  //    across blocks.
  const Transcript tL = greedy(m, 0, A, 60);
  m.session_close(0);
  {
    MimoModel wide(cfg, dir, /*max_tokens=*/128, /*max_cache_tokens=*/256, MimoResidency::Resident, nullptr, 0, 1, 1);
    wide.set_decode_row_traces(true);
    // The l2 budget admits routing near-tie flips over 60 rows; the top-1
    // rule is the gate.
    const int soft = audit(wide, A, tL, "decode 60 steps", 1e-1);
    std::printf("[ OK ] 60-step decode across the block boundary matches the re-forward (%d near ties)\n", soft);
  }
  // 2b. The serving scheduler's rolling prefix snapshot fires on EVERY
  //     decode step at this family's align of 1 (a live snapshot: the
  //     block list pinned, the partial block copied into a pinned block,
  //     the previous entry released). The transcript under per-step
  //     snapshots must be bitwise the plain one.
  {
    std::vector<uint8_t*> arena(2, nullptr);
    const size_t bytes = m.session_snapshot_bytes();
    for (uint8_t*& p : arena) require(cudaMalloc(reinterpret_cast<void**>(&p), bytes) == cudaSuccess, "snap arena");
    MimoModel::Outputs o = m.session_prefill(0, A);
    require(bitwise(o.logits, tL.rows[0]), "per-step snapshots: the prefill differs");
    int64_t pending = argmax(o.logits.data(), V);
    MimoModel::SessionSnapshotMeta metas[2];
    bool filled[2] = {false, false};
    for (int s = 0; s < 60; ++s) {
      const int slot = s % 2;
      if (filled[slot]) m.session_release_snapshot(metas[slot]);
      metas[slot] = m.session_snapshot(0, arena[slot]);
      filled[slot] = true;
      o = m.session_step(0, pending);
      require(bitwise(o.logits, tL.rows[static_cast<size_t>(s) + 1]),
              "per-step snapshots: step " + std::to_string(s) + " differs from the plain transcript");
      pending = argmax(o.logits.data(), V);
    }
    m.session_close(0);
    for (int i = 0; i < 2; ++i) if (filled[i]) m.session_release_snapshot(metas[i]);
    require(m.kv_blocks_in_use() == 0, "per-step snapshots: every block released");
    for (uint8_t* p : arena) cudaFree(p);
    std::printf("[ OK ] 60 steps under per-step rolling snapshots reproduce the plain transcript bitwise\n");
  }
  const Transcript tA = greedy(m, 0, A, 12);
  m.session_close(0);
  {
    const int soft = audit(m, A, tA, "decode 12 steps", 2e-2);
    std::printf("[ OK ] incremental decode matches the re-forward (%d near ties); ids %s\n", soft,
                ids_text(tA.tokens).c_str());
  }
  // 2c. The verify rows: T scalar steps as one T-row verify (the same
  //     tokens at the same positions, the rows' K/V appended before the
  //     attention reads them). The fp8 sites' streaming tensor-core form
  //     and the bf16 sites' GEMV chunks keep a row's chain whatever rows
  //     share the launch, so rows 1 .. 4 are bitwise the scalar steps; at
  //     5 and 6 rows the bf16 sites take cuBLASLt's algorithm (kernels/
  //     gemm.hpp dense_gemv_rows) and the rows are tolerance-equal. A
  //     rollback to one accepted row restarts the scalar chain bitwise.
  {
    const int gemv_rows = std::min(dgpp::dense_gemv_rows(), 4);
    for (int T = 1; T <= dgpp::kSpecRows; ++T) {
      MimoModel::Outputs p = m.session_prefill(0, A);
      require(bitwise(p.logits, tA.rows[0]), "verify rows: the prefill differs");
      const std::vector<int64_t> ids(tA.tokens.begin(), tA.tokens.begin() + T);
      const MimoModel::Outputs v = m.session_verify(0, ids);
      require(v.logits.size() == static_cast<size_t>(T) * V, "verify rows: T rows of logits");
      require(m.session_position(0) == static_cast<int64_t>(A.size()) + T, "verify rows: the position advanced by T");
      int exact = 0;
      for (int r = 0; r < T; ++r) {
        const std::vector<float> row(v.logits.begin() + static_cast<size_t>(r) * V, v.logits.begin() + static_cast<size_t>(r + 1) * V);
        const std::vector<float>& want = tA.rows[static_cast<size_t>(r) + 1];
        if (bitwise(row, want)) ++exact;
        else {
          const RowCompare c = compare_row(row.data(), want.data(), V);
          require(T > gemv_rows, "verify of " + std::to_string(T) + " rows: row " + std::to_string(r) +
                                     " is not bitwise the scalar step (relative l2 " + std::to_string(c.l2) + ")");
          require(c.top1_equal || c.near_tie, "verify of " + std::to_string(T) + " rows: row " + std::to_string(r) +
                                                  " top-1 differs from the scalar step beyond the near-tie margin");
          require(c.l2 < 2e-2, "verify of " + std::to_string(T) + " rows: row " + std::to_string(r) + " relative l2 over budget");
        }
      }
      // Accept row 0 only: the scalar chain continues bitwise from there.
      m.session_rollback(0, 1);
      require(m.session_position(0) == static_cast<int64_t>(A.size()) + 1, "verify rows: the rollback's position");
      const MimoModel::Outputs next = m.session_step(0, tA.tokens[1]);
      if (T <= gemv_rows) {
        require(bitwise(next.logits, tA.rows[2]),
                "verify of " + std::to_string(T) + " rows: the step after the rollback differs");
      } else {
        // Beyond the GEMV rows the verify's row 0 (whose K/V the step
        // re-reads) went through cuBLASLt: tolerance-equal, not bitwise.
        const RowCompare c = compare_row(next.logits.data(), tA.rows[2].data(), V);
        require(c.top1_equal || c.near_tie,
                "verify of " + std::to_string(T) + " rows: the rollback's next step top-1 differs");
        require(c.l2 < 2e-2, "verify of " + std::to_string(T) + " rows: the rollback's next step over budget");
      }
      m.session_close(0);
      std::printf("[ .. ] verify of %d rows: %d bitwise the scalar steps%s; the rollback's next step %s\n", T, exact,
                  T > gemv_rows && exact < T ? " (the rest tolerance-equal: cuBLASLt on the bf16 sites)" : "",
                  T > gemv_rows ? "tolerance-equal" : "bitwise");
    }
    std::printf("[ OK ] verify rows reproduce the scalar steps (bitwise through %d rows) and roll back bitwise\n", gemv_rows);
  }

  // 3. Two slots interleaved reproduce their solo runs bitwise.
  const Transcript tB = greedy(m, 0, B, 8);
  m.session_close(0);
  {
    MimoModel::Outputs oa = m.session_prefill(0, A);
    MimoModel::Outputs ob = m.session_prefill(1, B);
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
    require(m.kv_blocks_in_use() == 2, "interleaved: two slots hold two blocks");
    m.session_close(1);
    // Slot 0 keeps going after slot 1 closes.
    for (int s = 8; s < 12; ++s) {
      oa = m.session_step(0, pa);
      require(bitwise(oa.logits, tA.rows[static_cast<size_t>(s) + 1]),
              "interleaved: slot 0's step " + std::to_string(s) + " after slot 1 closed");
      pa = argmax(oa.logits.data(), V);
    }
    m.session_close(0);
    std::printf("[ OK ] two interleaved slots reproduce their solo transcripts bitwise\n");
  }

  // 4. Chunked prefill (chunk == max_tokens == 8) agrees with the one-shot.
  {
    MimoModel c(cfg, dir, /*max_tokens=*/8, /*max_cache_tokens=*/256, MimoResidency::Resident, nullptr,
                0, 1, /*max_requests=*/1);
    const MimoModel::Outputs one = m.session_prefill(0, A);
    m.session_close(0);
    const MimoModel::Outputs p = c.session_prefill(0, A);
    RowCompare r = compare_row(p.logits.data(), one.logits.data(), V);
    std::printf("[ .. ] chunked prefill (8-row chunks): relative l2 %.3g, top-1 %s\n", r.l2,
                r.top1_equal ? "equal" : (r.near_tie ? "near tie" : "MISMATCH"));
    require(r.top1_equal || r.near_tie, "chunked prefill: top-1 mismatch");
    require(r.l2 < 2e-2, "chunked prefill: relative l2 over budget");
    // Boundary cuts: the pool-aligned image of 13 (12) joins the 8-multiples.
    c.session_close(0);
    const MimoModel::Outputs pb = c.session_prefill(0, A, std::vector<int64_t>{13});
    r = compare_row(pb.logits.data(), one.logits.data(), V);
    require(r.top1_equal || r.near_tie, "chunked prefill with a boundary cut: top-1 mismatch");
    require(r.l2 < 2e-2, "chunked prefill with a boundary cut: relative l2 over budget");
    // Decode continues from the chunked state.
    Transcript tc;
    push_row(tc, pb, cfg.num_experts_per_tok);
    for (int s = 0; s < 6; ++s) {
      const MimoModel::Outputs o = c.session_step(0, tc.tokens.back());
      push_row(tc, o, cfg.num_experts_per_tok);
    }
    c.session_close(0);
    audit(m, A, tc, "decode after the chunked prefill", 2e-2);
    std::printf("[ OK ] chunked prefill and its decode agree with the one-shot walk\n");
  }

  // 5. Close and reopen restarts bitwise; the pool's accounting.
  {
    require(m.kv_blocks_total() == 4, "pool: 256 tokens are 4 blocks of 64");
    require(m.kv_blocks_in_use() == 0, "pool: nothing held after the closes");
    require(m.kv_blocks_for_tokens(65) == 2, "pool: 65 tokens take 2 blocks");
    const MimoModel::Outputs p = m.session_prefill(0, A);
    require(bitwise(p.logits, tA.rows[0]), "reopen: the prefill differs from the first run");
    require(m.kv_blocks_in_use() == 1, "pool: one block for 23 tokens");
    m.session_reserve_blocks(0, 130);
    require(m.kv_blocks_in_use() == 3, "pool: the reserve grew the slot to 3 blocks");
    bool refused = false;
    try {
      m.session_reserve_blocks(0, 257);
    } catch (const std::exception&) {
      refused = true;
    }
    require(refused, "pool: a reserve beyond the context bound is refused");
    MimoModel::Outputs o = m.session_step(0, tA.tokens[0]);
    require(bitwise(o.logits, tA.rows[1]), "reopen: the first step differs from the first run");
    m.session_close(0);
    require(m.kv_blocks_in_use() == 0, "pool: the close released every block");
    // The forward refuses an open slot 0 and works after the close.
    (void)m.session_prefill(0, B);
    refused = false;
    try {
      (void)m.forward(B);
    } catch (const std::logic_error&) {
      refused = true;
    }
    require(refused, "forward: must refuse while slot 0 is open");
    m.session_close(0);
    const MimoModel::Outputs f = m.forward(B);
    require(bitwise(std::vector<float>(f.logits.end() - V, f.logits.end()), tB.rows[0]),
            "forward after the sessions differs from the solo prefill");
    std::printf("[ OK ] close/reopen restarts bitwise; pool accounting; forward guarded\n");
  }
  // 6. The prefix cache: hot == cold bitwise. A snapshot at the pool-
  //    aligned cut 12 of A's prefill, attached in another slot, the
  //    suffix resumed — the last row and the steps after it bitwise the
  //    cold session's; a mid-decode snapshot likewise.
  {
    const std::vector<int64_t> bounds{12};
    std::vector<uint8_t*> arena(2, nullptr);
    const size_t bytes = m.session_snapshot_bytes();
    require(bytes > 0, "prefix: the snapshot has bytes");
    for (uint8_t*& p : arena) require(cudaMalloc(reinterpret_cast<void**>(&p), bytes) == cudaSuccess, "prefix: arena");
    MimoModel::SessionSnapshotMeta meta;
    MimoModel::SnapshotRequest snap;
    snap.position = 12;
    snap.dst = arena[0];
    snap.meta = &meta;
    const MimoModel::Outputs cold = m.session_prefill(0, A, bounds, &snap);
    require(snap.taken && meta.position == 12, "prefix: the snapshot was taken at the cut");
    // Position 12 sits inside block 0: the entry owns a COPY of the partial
    // block beside the request's own (two in use), and keeps it after the
    // request closes.
    require(m.kv_blocks_in_use() == 2, "prefix: the entry copied the partial block");
    const MimoModel::Outputs cold_step = m.session_step(0, tA.tokens[0]);
    m.session_close(0);
    require(m.kv_blocks_in_use() == 1, "prefix: the entry keeps its partial block after the close");
    m.session_attach(1, arena[0], meta);
    require(m.session_position(1) == 12, "prefix: attached at the snapshot position");
    const MimoModel::Outputs hot = m.session_prefill_resume(1, std::vector<int64_t>(A.begin() + 12, A.end()), bounds);
    // Hot == cold bitwise (the same chunks on the same state). The plain
    // one-chunk prefill is a different launch shape (its m), so it is only
    // near the two-chunk one — the chunking gate above covers that.
    require(bitwise(hot.logits, cold.logits), "prefix: the hot prefill's last row differs from the cold one's");
    const MimoModel::Outputs hot_step = m.session_step(1, tA.tokens[0]);
    require(bitwise(hot_step.logits, cold_step.logits), "prefix: the first step after the attach differs");
    // A mid-decode snapshot at an unaligned position: 23 + 5 steps = 28.
    for (int s = 1; s < 5; ++s) (void)m.session_step(1, tA.tokens[static_cast<size_t>(s)]);
    require(m.session_position(1) == 28, "prefix: position 28");
    MimoModel::SessionSnapshotMeta meta2 = m.session_snapshot(1, arena[1]);
    const MimoModel::Outputs cont = m.session_step(1, tA.tokens[5]);
    m.session_close(1);
    m.session_attach(0, arena[1], meta2);
    const MimoModel::Outputs re = m.session_step(0, tA.tokens[5]);
    require(bitwise(re.logits, cont.logits), "prefix: the step after a mid-decode attach differs");
    m.session_close(0);
    m.session_release_snapshot(meta);
    m.session_release_snapshot(meta2);
    require(m.kv_blocks_in_use() == 0, "prefix: every block released with the entries");
    for (uint8_t* p : arena) cudaFree(p);
    std::printf("[ OK ] prefix snapshots: hot == cold bitwise at a cut and mid-decode\n");
  }
  // 7. The MTP draft block, eagerly (the greedy speculator): the committed
  //    transcript is the plain greedy one exactly (a verify's rows are the
  //    steps' rows; a rejected draft's rows roll back), the draft rate
  //    reported. Then close/reopen through the draft's counter.
  {
    MimoModel d(cfg, dir, /*max_tokens=*/64, /*max_cache_tokens=*/256, MimoResidency::Resident, nullptr, 0, 1,
                /*max_requests=*/2, /*mtp=*/true);
    require(d.mtp_enabled(), "mtp: enabled");
    const auto pick_rows = [](const std::vector<dgpp::sample::Candidate>& c) {
      std::vector<int32_t> ids;
      for (const auto& x : c) ids.push_back(x.id);
      return ids;
    };
    for (int round = 0; round < 2; ++round) {
      const MimoModel::Outputs p = d.session_prefill(1, A);
      require(bitwise(p.logits, tA.rows[0]), "mtp: the prefill's last row differs from the plain model's");
      require(d.session_draft_position(1) == static_cast<int64_t>(A.size()) - 1, "mtp: the draft trails by one after the prefill");
      // The speculator commits the pending token with each verify; a
      // random-weight fixture drafts by chance only, so the first draft is
      // FORCED to the known next token: that step must accept it (two
      // tokens committed, no rollback), later ones roll their rejects back.
      dgpp::GreedySpeculator<MimoModel> spec(d, 1, pick_rows);
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
      require(first_step_committed == 2, "mtp: the forced correct draft was not accepted");
      std::printf("[ .. ] mtp round %d: %d steps for %zu tokens, %d drafts accepted (the forced one included)\n",
                  round, spec.steps(), committed.size(), spec.accepted_drafts());
      d.session_close(1);
    }
    // The draft block over the prompt (mtp_forward: rows q = 0 .. T-2,
    // token q+1 on the hidden at q) agrees with the session's draft row:
    // a prefill of A[0 .. T-2] then the draft fed A[T-1] is row T-2 on the
    // same fusion — tolerance-equal (the hidden comes off an m = T-1
    // walk against mtp_forward's m = T, the draft row off a decode-row
    // launch against the prefill-row one).
    {
      const MimoModel::Outputs df = d.mtp_forward(A);
      require(df.logits.size() == static_cast<size_t>(A.size() - 1) * V, "mtp_forward: T-1 rows of draft logits");
      for (float x : df.logits) require(std::isfinite(x), "mtp_forward: a non-finite draft logit");
      const std::vector<int64_t> head(A.begin(), A.end() - 1);
      (void)d.session_prefill(1, head);
      require(d.session_draft_position(1) == static_cast<int64_t>(head.size()) - 1, "session_draft: the draft's counter");
      const MimoModel::Outputs sd = d.session_draft(1, std::vector<int64_t>{A.back()});
      require(sd.logits.size() == static_cast<size_t>(V), "session_draft: one row");
      const std::vector<float> last(df.logits.end() - V, df.logits.end());
      const RowCompare c = compare_row(sd.logits.data(), last.data(), V);
      std::printf("[ .. ] the session's draft row vs mtp_forward's last row: relative l2 %.3g, top-1 %s\n", c.l2,
                  c.top1_equal ? "equal" : c.near_tie ? "near tie" : "DIFFERS");
      require(c.top1_equal || c.near_tie, "session_draft: top-1 differs from mtp_forward's beyond the near-tie margin");
      require(c.l2 < 5e-2, "session_draft: relative l2 over budget vs mtp_forward");
      d.session_close(1);
    }
    std::printf("[ OK ] the eager speculator reproduces the greedy transcript through the draft block\n");
  }
  // 8. The sliding window at the session level.
  run_window_gate(fs::path(dir).parent_path().string());
  // 9. The fp8 K/V pool (engine.kv_dtype fp8): the same prompt and steps
  // through a model whose pool keeps the fp8 row form — its transcript
  // against the bf16 pool's under the fp8 cache's tolerance (every row's
  // relative l2 small; a top-1 that differs is a near tie), and its own
  // prefill == forward and chunked == one-shot equalities, which the format
  // must keep (the same rows are quantized either way).
  {
    const std::vector<int64_t> A = smoke_tokens(cfg, 40, 0x51ED270693F0A1B3ull);
    MimoModel b16(cfg, dir, /*max_tokens=*/64, /*max_cache_tokens=*/256, MimoResidency::Resident, nullptr, 0, 1, 1);
    MimoModel f8(cfg, dir, /*max_tokens=*/64, /*max_cache_tokens=*/256, MimoResidency::Resident, nullptr, 0, 1, 1, false,
                 0, dgpp::LatentFormat::kFp8);
    require(f8.kv_format() == dgpp::LatentFormat::kFp8 && b16.kv_format() == dgpp::LatentFormat::kBf16, "the pools' formats");
    const MimoModel::Outputs fa = f8.forward(A);
    const MimoModel::Outputs pa = f8.session_prefill(0, A);
    require(pa.logits == std::vector<float>(fa.logits.end() - V, fa.logits.end()), "fp8 pool: prefill last row == forward");
    f8.session_close(0);
    const Transcript tb = greedy(b16, 0, A, 24);
    const Transcript tf = greedy(f8, 0, A, 24);
    b16.session_close(0);
    f8.session_close(0);
    // The rows up to the first token the two transcripts pick differently:
    // from there the transcripts are different prompts (on this random
    // fixture one fp8-rounded near tie of the 2-of-8 router moves a row's
    // logits by half their norm, and the pick with it).
    size_t agree = 0;
    while (agree < tf.tokens.size() && agree < tb.tokens.size() && tf.tokens[agree] == tb.tokens[agree]) ++agree;
    double worst = 0;
    for (size_t i = 0; i < agree && i < tf.rows.size() && i < tb.rows.size(); ++i)
      worst = std::max(worst, compare_row(tf.rows[i].data(), tb.rows[i].data(), V).l2);
    std::printf("[ .. ] fp8 pool vs bf16 pool: the first %zu of %zu greedy picks agree, worst relative l2 over them %.3g; "
                "ids %s | %s\n",
                agree, tf.tokens.size(), worst, ids_text(tb.tokens).c_str(), ids_text(tf.tokens).c_str());
    require(agree >= 8, "fp8 pool: fewer than eight greedy picks agree with the bf16 pool");
    require(worst < 0.1, "fp8 pool: relative l2 over the fp8 cache's budget on the agreeing rows");
    std::printf("[ OK ] the fp8 K/V pool decodes within the format's budget of the bf16 pool\n");
  }
  std::printf("[ OK ] mimo_decode_test\n");
  return 0;
}

// ---- the real checkpoint ---------------------------------------------------------
// Streaming walks: the prefill against the cold forward of the same prompt
// (bitwise expected: the same m=P launches), the greedy steps, the
// re-forward of prompt + transcript against every row — and the
// re-forward's prompt rows against the P-row forward's, the localizer for
// a hard mismatch.
void report_row(const char* what, size_t i, const float* got, const float* want, int V) {
  const RowCompare c = compare_row(got, want, V);
  const int32_t a = argmax(want, V), b = argmax(got, V);
  std::printf("[ .. ] %s row %zu: relative l2 %.3g; top-1 want %d (%.3f) got %d (%.3f; want's logit there %.3f) %s\n",
              what, i, c.l2, a, want[a], b, got[b], want[b],
              c.top1_equal ? "equal" : (c.near_tie ? "near tie" : "HARD MISMATCH"));
}

// The MoE layer's ordinal among the MoE layers (the route captures' slot).
int moe_ordinal(const MimoTextConfig& cfg, int layer) {
  int n = 0;
  for (int l = 0; l < layer; ++l) n += cfg.is_moe_layer(l);
  return n;
}

// The divergence profile of two cold forwards of one prompt at T=P and
// T=P+extra: per layer, the residual's relative l2 on the shared rows and
// the MoE routing flips — a smooth growth with flips is amplification
// through the stack, a jump at one layer kind is a bug in that path.
void profile_pair(const MimoTextConfig& cfg, const MimoModel::Outputs& a, const MimoModel::Outputs& b, int P,
                  const char* what) {
  const int W = cfg.hidden_size;
  const int K = cfg.num_experts_per_tok;
  std::printf("[ .. ] %s\n", what);
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    if (l > 2 && l + 1 < cfg.num_hidden_layers && l % 8 != 7) continue;
    std::string line = "[ .. ] layer " + std::to_string(l) + (cfg.is_swa_layer(l) ? " swa" : " global") +
                       (cfg.is_moe_layer(l) ? " moe" : " dense") + ": l2";
    for (int r = 0; r < P; ++r) {
      double d2 = 0, w2 = 0;
      for (int c = 0; c < W; ++c) {
        const double x = dgpp::bf16_bits_to_float(a.layer_states[static_cast<size_t>(l)][static_cast<size_t>(r) * W + c]);
        const double y = dgpp::bf16_bits_to_float(b.layer_states[static_cast<size_t>(l)][static_cast<size_t>(r) * W + c]);
        d2 += (x - y) * (x - y);
        w2 += x * x;
      }
      char buf[32];
      std::snprintf(buf, sizeof(buf), " %.2e", std::sqrt(d2) / std::sqrt(w2 + 1e-30));
      line += buf;
    }
    if (cfg.is_moe_layer(l)) {
      const size_t m = static_cast<size_t>(moe_ordinal(cfg, l));
      int flips = 0;
      for (int r = 0; r < P; ++r)
        for (int j = 0; j < K; ++j)
          if (a.route_ids[m][static_cast<size_t>(r) * K + j] != b.route_ids[m][static_cast<size_t>(r) * K + j]) ++flips;
      line += "; route flips " + std::to_string(flips) + "/" + std::to_string(P * K);
    }
    std::printf("%s\n", line.c_str());
    std::fflush(stdout);
  }
}

int run_layers(const std::string& dir, const std::vector<int64_t>& ids, int extra) {
  const MimoTextConfig cfg = MimoTextConfig::from_json_file((fs::path(dir) / "config.json").string());
  const int P = static_cast<int>(ids.size());
  const auto longer = [&](int n) {
    std::vector<int64_t> v(ids);
    for (int i = 0; i < n; ++i) v.push_back(ids[static_cast<size_t>(i % P)]);
    return v;
  };
  const int top = std::max(extra, 7);
  MimoModel m(cfg, dir, /*max_tokens=*/P + top, /*max_cache_tokens=*/P + top + 64,
              MimoResidency::Streaming, nullptr, 0, 1, 1);
  const MimoModel::Outputs a = m.forward(ids, true);
  const MimoModel::Outputs a1 = m.forward(longer(1), true);
  const MimoModel::Outputs b = m.forward(longer(extra), true);
  const MimoModel::Outputs b7 = m.forward(longer(7), true);
  profile_pair(cfg, a, a1, P, "forward(P) vs forward(P+1)");
  profile_pair(cfg, b, b7, P, "forward(P+extra) vs forward(P+7)");
  profile_pair(cfg, a, b, P, "forward(P) vs forward(P+extra)");
  return 0;
}

int run_checkpoint(const std::string& dir, const std::vector<int64_t>& ids, int steps) {
  const MimoTextConfig cfg = MimoTextConfig::from_json_file((fs::path(dir) / "config.json").string());
  const int P = static_cast<int>(ids.size());
  MimoModel m(cfg, dir, /*max_tokens=*/P + steps, /*max_cache_tokens=*/P + steps + 64,
              MimoResidency::Streaming, nullptr, 0, 1, 1);
  m.set_decode_row_traces(true);
  const int V = m.lm_vocab_count();
  std::printf("[ .. ] %s: %d prompt tokens, %d greedy steps (streaming world 1)\n", dir.c_str(), P, steps);
  const MimoModel::Outputs fP = m.forward(ids);
  const Transcript t = greedy(m, 0, ids, steps);
  m.session_close(0);
  {
    const std::vector<float> last(fP.logits.end() - V, fP.logits.end());
    std::printf("[ .. ] prefill(%d) vs forward(%d) last row: %s\n", P, P,
                bitwise(t.rows[0], last) ? "BITWISE" : "DIFFERENT");
    report_row("prefill vs forward", static_cast<size_t>(P - 1), t.rows[0].data(), last.data(), V);
  }
  std::printf("[ .. ] transcript: %s\n", ids_text(t.tokens).c_str());
  std::vector<int64_t> all(ids);
  all.insert(all.end(), t.tokens.begin(), t.tokens.end() - 1);
  const MimoModel::Outputs fA = m.forward(all);
  for (int r = 0; r < P; ++r)
    report_row("forward(P+steps) vs forward(P)", static_cast<size_t>(r),
               fA.logits.data() + static_cast<size_t>(r) * V, fP.logits.data() + static_cast<size_t>(r) * V, V);
  int hard = 0;
  for (size_t i = 0; i < t.rows.size(); ++i) {
    const float* want = fA.logits.data() + (static_cast<size_t>(P) - 1 + i) * static_cast<size_t>(V);
    report_row("decode vs re-forward", static_cast<size_t>(P) - 1 + i, t.rows[i].data(), want, V);
    const RowCompare c = compare_row(t.rows[i].data(), want, V);
    if (!c.top1_equal && !c.near_tie) ++hard;
  }
  require(hard == 0, "real-checkpoint decode: a top-1 mismatch beyond the near-tie margin");
  std::printf("[ OK ] mimo_decode_test (real checkpoint)\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string fixture, checkpoint, ids_text;
  int steps = 4;
  int layers_extra = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--fixture" && i + 1 < argc) fixture = argv[++i];
    else if (a == "--checkpoint-dir" && i + 1 < argc) checkpoint = argv[++i];
    else if (a == "--ids" && i + 1 < argc) ids_text = argv[++i];
    else if (a == "--steps" && i + 1 < argc) steps = std::stoi(argv[++i]);
    else if (a == "--layers" && i + 1 < argc) layers_extra = std::stoi(argv[++i]);
  }
  try {
    if (!checkpoint.empty()) {
      std::vector<int64_t> ids;
      std::stringstream ss(ids_text);
      std::string item;
      while (std::getline(ss, item, ',')) if (!item.empty()) ids.push_back(std::stoll(item));
      if (ids.empty()) throw std::runtime_error("--ids is required with --checkpoint-dir");
      if (layers_extra >= 0) return run_layers(checkpoint, ids, layers_extra);
      return run_checkpoint(checkpoint, ids, steps);
    }
    if (fixture.empty() || !fs::is_regular_file(fs::path(fixture) / "config.json")) {
      // No fixture given (or not written yet): the tiny release into a
      // directory of our own.
      fixture = (fs::current_path() / "mimo_decode_fixture").string();
      mimofx::write_fixture(mimofx::tiny_config(), fixture);
    }
    return run_fixture(fixture);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[FAIL] %s\n", e.what());
    return 1;
  }
}
