// The GLM-4.7 session surface's gate (2026-09-09, docs/glm47_plan.md G4):
// incremental decode over Glm4Model's request slots against the cold
// re-forward, the prefix snapshots (any position: plan D5), the eager
// speculator through the draft block.
//
//   --fixture DIR                the fixture gates (tests/cuda/glm4_fixture.hpp's
//                                checkpoint, written by glm4_forward_test)
//   --checkpoint-dir DIR --ids 1,2,... [--steps N]
//                                the real checkpoint (streaming, world 1): a
//                                greedy transcript and its re-forward audit
//
// Gates on the fixture: a one-shot prefill's last row is bitwise the cold
// forward's (the same m=T GEMMs on the same state); T=1 steps agree with
// the re-forward's rows at every position under the near-tie rule (their
// m=1 GEMMs reassociate); two interleaved slots reproduce their solo runs
// bitwise; a chunked prefill (chunk == max_tokens, pool-aligned boundary
// cuts) agrees with the one-shot; a closed and reopened slot restarts
// bitwise; the pool's block accounting.
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
#include "models/glm4/config.hpp"
#include "models/glm4/forward.hpp"
#include "attach_within_gate.hpp"

namespace fs = std::filesystem;
using dgpp::Glm4Model;
using dgpp::Glm4Residency;
using dgpp::Glm4TextConfig;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

std::vector<int64_t> smoke_tokens(const Glm4TextConfig& cfg, int n, uint64_t seed) {
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
// near-tie certification (glm4_forward_test's rule).
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
// argmax; every step's logits row is kept.
struct Transcript {
  std::vector<int64_t> tokens;             // generated ids (the prefill's pick first)
  std::vector<std::vector<float>> rows;    // the prefill's row, then every step's
};

Transcript greedy(Glm4Model& m, int req, const std::vector<int64_t>& prompt, int steps) {
  Transcript t;
  Glm4Model::Outputs o = m.session_prefill(req, prompt);
  int64_t pending = argmax(o.logits.data(), o.lm_vocab_count);
  t.tokens.push_back(pending);
  t.rows.push_back(o.logits);
  for (int s = 0; s < steps; ++s) {
    o = m.session_step(req, pending);
    pending = argmax(o.logits.data(), o.lm_vocab_count);
    t.tokens.push_back(pending);
    t.rows.push_back(o.logits);
  }
  return t;
}

std::string ids_text(const std::vector<int64_t>& ids) {
  std::string s;
  for (size_t i = 0; i < ids.size(); ++i) s += (i ? "," : "") + std::to_string(ids[i]);
  return s;
}

// The transcript's rows against the cold re-forward of prompt + tokens:
// row P-1+i of the re-forward is step i's input at the same position.
int audit(Glm4Model& ref, const std::vector<int64_t>& prompt, const Transcript& t, const char* what,
          double l2_budget) {
  std::vector<int64_t> all(prompt);
  all.insert(all.end(), t.tokens.begin(), t.tokens.end() - 1);
  const Glm4Model::Outputs f = ref.forward(all);
  const int V = f.lm_vocab_count;
  const size_t P = prompt.size();
  int hard = 0, soft = 0;
  double worst_l2 = 0;
  std::string profile;
  for (size_t i = 0; i < t.rows.size(); ++i) {
    const float* want = f.logits.data() + (P - 1 + i) * static_cast<size_t>(V);
    const RowCompare c = compare_row(t.rows[i].data(), want, V);
    worst_l2 = std::max(worst_l2, c.l2);
    if (!c.top1_equal) (c.near_tie ? soft : hard) += 1;
    if (t.rows.size() > 16 && (i % 4 == 0 || c.l2 > 0.02)) {
      char buf[48];
      std::snprintf(buf, sizeof(buf), " p%zu:%.3f", P + i, c.l2);
      profile += buf;
    }
  }
  if (!profile.empty()) std::printf("[ .. ] %s: per-position relative l2%s\n", what, profile.c_str());
  std::printf("[ .. ] %s: %zu rows vs the re-forward — worst relative l2 %.3g, top-1 hard %d near-tie %d\n",
              what, t.rows.size(), worst_l2, hard, soft);
  require(hard == 0, std::string(what) + ": a top-1 mismatch beyond the near-tie margin");
  require(worst_l2 < l2_budget, std::string(what) + ": relative l2 over budget");
  return soft;
}

// ---- the fixture gates ----------------------------------------------------------
int run_fixture(const std::string& dir) {
  const Glm4TextConfig cfg = Glm4TextConfig::from_json_file((fs::path(dir) / "config.json").string());
  const std::vector<int64_t> A = smoke_tokens(cfg, 23, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 17, 0xD1B54A32D192ED03ull);
  Glm4Model m(cfg, dir, /*max_tokens=*/64, /*max_cache_tokens=*/256, Glm4Residency::Resident, nullptr,
              0, 1, /*max_requests=*/2);
  const int V = m.lm_vocab_count();

  // 1. prefill == forward, bitwise.
  {
    const Glm4Model::Outputs f = m.forward(A);
    const Glm4Model::Outputs p = m.session_prefill(0, A);
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
  //     prefills alone (2026-09-14): the attention rows carry their own request ids and positions;
  //     the dense sites see 40 rows instead of 23 and 17 (cuBLASLt's
  //     algorithm at each: kernels/gemm.hpp dense_gemv_rows), so the rows
  //     are tolerance-equal under the row compare's l2 and near-tie rule,
  //     and a decode off the group's cache is audited against the
  //     re-forward like any other.
  {
    const Glm4Model::Outputs pa = m.session_prefill(0, A);
    m.session_close(0);
    const Glm4Model::Outputs pb = m.session_prefill(1, B);
    m.session_close(1);
    const std::vector<Glm4Model::Outputs> g = m.session_prefill_group({0, 1}, {&A, &B});
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
    int64_t pending = argmax(g[1].logits.data(), V);
    tb.tokens.push_back(pending);
    tb.rows.push_back(g[1].logits);
    for (int s = 0; s < 8; ++s) {
      const Glm4Model::Outputs o = m.session_step(1, pending);
      pending = argmax(o.logits.data(), V);
      tb.tokens.push_back(pending);
      tb.rows.push_back(o.logits);
    }
    m.session_close(0);
    m.session_close(1);
    const int soft = audit(m, B, tb, "decode after the group prefill", 1e-1);
    std::printf("[ OK ] group prefill: the spans' rows and an 8-step decode off the group's cache (%d near ties)\n", soft);
  }

  // 2. Incremental decode vs the re-forward — 60 steps from the 23-token
  //    prompt, so the decode rows cross the pool's 64-token block boundary
  //    (position 64 lands on step 42; every other gate stays inside block
  //    0 — the real checkpoint's transcripts degenerated after ~30 tokens,
  //    2026-09-10).
  const Transcript tL = greedy(m, 0, A, 60);
  m.session_close(0);
  {
    Glm4Model wide(cfg, dir, /*max_tokens=*/128, /*max_cache_tokens=*/256, Glm4Residency::Resident, nullptr, 0, 1, 1);
    // The l2 budget admits one routing near-tie flip in 60 rows (position
    // 81's row sits at 0.05 on this fixture, its top-1 a certified near
    // tie); the top-1 rule is the gate.
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
    Glm4Model::Outputs o = m.session_prefill(0, A);
    require(bitwise(o.logits, tL.rows[0]), "per-step snapshots: the prefill differs");
    int64_t pending = argmax(o.logits.data(), V);
    Glm4Model::SessionSnapshotMeta metas[2];
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

  // 3. Two slots interleaved reproduce their solo runs bitwise.
  const Transcript tB = greedy(m, 0, B, 8);
  m.session_close(0);
  {
    Glm4Model::Outputs oa = m.session_prefill(0, A);
    Glm4Model::Outputs ob = m.session_prefill(1, B);
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
    Glm4Model c(cfg, dir, /*max_tokens=*/8, /*max_cache_tokens=*/256, Glm4Residency::Resident, nullptr,
                0, 1, /*max_requests=*/1);
    const Glm4Model::Outputs one = m.session_prefill(0, A);
    m.session_close(0);
    const Glm4Model::Outputs p = c.session_prefill(0, A);
    RowCompare r = compare_row(p.logits.data(), one.logits.data(), V);
    std::printf("[ .. ] chunked prefill (8-row chunks): relative l2 %.3g, top-1 %s\n", r.l2,
                r.top1_equal ? "equal" : (r.near_tie ? "near tie" : "MISMATCH"));
    require(r.top1_equal || r.near_tie, "chunked prefill: top-1 mismatch");
    require(r.l2 < 2e-2, "chunked prefill: relative l2 over budget");
    // Boundary cuts: the pool-aligned image of 13 (12) joins the 8-multiples.
    c.session_close(0);
    const Glm4Model::Outputs pb = c.session_prefill(0, A, std::vector<int64_t>{13});
    r = compare_row(pb.logits.data(), one.logits.data(), V);
    require(r.top1_equal || r.near_tie, "chunked prefill with a boundary cut: top-1 mismatch");
    require(r.l2 < 2e-2, "chunked prefill with a boundary cut: relative l2 over budget");
    // Decode continues from the chunked state.
    int64_t pending = argmax(pb.logits.data(), V);
    Transcript tc;
    tc.tokens.push_back(pending);
    tc.rows.push_back(pb.logits);
    for (int s = 0; s < 6; ++s) {
      const Glm4Model::Outputs o = c.session_step(0, pending);
      pending = argmax(o.logits.data(), V);
      tc.tokens.push_back(pending);
      tc.rows.push_back(o.logits);
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
    const Glm4Model::Outputs p = m.session_prefill(0, A);
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
    Glm4Model::Outputs o = m.session_step(0, tA.tokens[0]);
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
    const Glm4Model::Outputs f = m.forward(B);
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
    Glm4Model::SessionSnapshotMeta meta;
    Glm4Model::SnapshotRequest snap;
    snap.position = 12;
    snap.dst = arena[0];
    snap.meta = &meta;
    const Glm4Model::Outputs cold = m.session_prefill(0, A, bounds, &snap);
    require(snap.taken && meta.position == 12, "prefix: the snapshot was taken at the cut");
    // Position 12 sits inside block 0: the entry owns a COPY of the partial
    // block beside the request's own (two in use), and keeps it after the
    // request closes.
    require(m.kv_blocks_in_use() == 2, "prefix: the entry copied the partial block");
    const Glm4Model::Outputs cold_step = m.session_step(0, tA.tokens[0]);
    m.session_close(0);
    require(m.kv_blocks_in_use() == 1, "prefix: the entry keeps its partial block after the close");
    m.session_attach(1, arena[0], meta);
    require(m.session_position(1) == 12, "prefix: attached at the snapshot position");
    const Glm4Model::Outputs hot = m.session_prefill_resume(1, std::vector<int64_t>(A.begin() + 12, A.end()), bounds);
    // Hot == cold bitwise (the same chunks on the same state). The plain
    // one-chunk prefill is a different GEMM shape (its m), so it is only
    // near the two-chunk one — the chunking gate above covers that.
    require(bitwise(hot.logits, cold.logits), "prefix: the hot prefill's last row differs from the cold one's");
    const Glm4Model::Outputs hot_step = m.session_step(1, tA.tokens[0]);
    require(bitwise(hot_step.logits, cold_step.logits), "prefix: the first step after the attach differs");
    // A mid-decode snapshot at a pool-aligned position: 23 + 5 steps = 28.
    for (int s = 1; s < 5; ++s) (void)m.session_step(1, tA.tokens[static_cast<size_t>(s)]);
    require(m.session_position(1) == 28, "prefix: position 28");
    Glm4Model::SessionSnapshotMeta meta2 = m.session_snapshot(1, arena[1]);
    const Glm4Model::Outputs cont = m.session_step(1, tA.tokens[5]);
    m.session_close(1);
    m.session_attach(0, arena[1], meta2);
    const Glm4Model::Outputs re = m.session_step(0, tA.tokens[5]);
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
    Glm4Model d(cfg, dir, /*max_tokens=*/64, /*max_cache_tokens=*/256, Glm4Residency::Resident, nullptr, 0, 1,
                /*max_requests=*/2, /*mtp=*/true);
    require(d.mtp_enabled(), "mtp: enabled");
    const auto pick_rows = [](const std::vector<dgpp::sample::Candidate>& c) {
      std::vector<int32_t> ids;
      for (const auto& x : c) ids.push_back(x.id);
      return ids;
    };
    for (int round = 0; round < 2; ++round) {
      const Glm4Model::Outputs p = d.session_prefill(1, A);
      require(bitwise(p.logits, tA.rows[0]), "mtp: the prefill's last row differs from the plain model's");
      require(d.session_draft_position(1) == static_cast<int64_t>(A.size()) - 1, "mtp: the draft trails by one after the prefill");
      // The speculator commits the pending token with each verify; a
      // random-weight fixture drafts by chance only, so the first draft is
      // FORCED to the known next token: that step must accept it (two
      // tokens committed, no rollback), later ones roll their rejects back.
      dgpp::GreedySpeculator<Glm4Model> spec(d, 1, pick_rows);
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
    std::printf("[ OK ] the eager speculator reproduces the greedy transcript through the draft block\n");
  }
  // The prefix cache's hit inside an entry (the shared gate).
  for (const bool mtp : {false, true}) {
    Glm4Model w(cfg, dir, /*max_tokens=*/64, /*max_cache_tokens=*/2048, Glm4Residency::Resident, nullptr, 0, 1,
            /*max_requests=*/2, mtp);
    dgpp::test::attach_within_gate(w, [&](int n, uint64_t seed) { return smoke_tokens(cfg, n, seed); },
                                   cfg.vocab_size, mtp, "glm4");
  }
  std::printf("[ OK ] glm4_decode_test\n");
  return 0;
}

// ---- the real checkpoint ---------------------------------------------------------
// Seven streaming walks: the prefill against the cold forward of the same
// prompt (bitwise expected: the same m=P GEMMs), the greedy steps, the
// re-forward of prompt + transcript against every row — and the re-
// forward's prompt rows against the P-row forward's (the m=P+steps GEMM
// path against the m=P one), the localizer for a hard mismatch.
void report_row(const char* what, size_t i, const float* got, const float* want, int V) {
  const RowCompare c = compare_row(got, want, V);
  const int32_t a = argmax(want, V), b = argmax(got, V);
  std::printf("[ .. ] %s row %zu: relative l2 %.3g; top-1 want %d (%.3f) got %d (%.3f; want's logit there %.3f) %s\n",
              what, i, c.l2, a, want[a], b, got[b], want[b],
              c.top1_equal ? "equal" : (c.near_tie ? "near tie" : "HARD MISMATCH"));
}

// The divergence profile of two cold forwards of one prompt at T=P and
// T=P+extra (the interface's GEMV family vs cuBLASLt's): per layer, the hyper
// state's relative l2 on the shared rows and the MoE routing flips — a
// smooth growth with flips is amplification through the stack, a jump at
// one layer kind is a bug in that path.
void profile_pair(const Glm4TextConfig& cfg, const Glm4Model::Outputs& a, const Glm4Model::Outputs& b, int P,
                  const char* what) {
  const int W = cfg.hidden_size;
  const int K = cfg.num_experts_per_tok;
  std::printf("[ .. ] %s\n", what);
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    if (l > 2 && l + 1 < cfg.num_hidden_layers && l % 8 != 7) continue;
    std::string line = "[ .. ] layer " + std::to_string(l) + (cfg.is_moe_layer(l) ? " moe" : " dense") + ": l2";
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
      const size_t m = static_cast<size_t>(l - cfg.first_k_dense_replace);
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
  const Glm4TextConfig cfg = Glm4TextConfig::from_json_file((fs::path(dir) / "config.json").string());
  const int P = static_cast<int>(ids.size());
  const auto longer = [&](int n) {
    std::vector<int64_t> v(ids);
    for (int i = 0; i < n; ++i) v.push_back(ids[static_cast<size_t>(i % P)]);
    return v;
  };
  const int top = std::max(extra, 7);
  Glm4Model m(cfg, dir, /*max_tokens=*/P + top, /*max_cache_tokens=*/P + top + 64,
              Glm4Residency::Streaming, nullptr, 0, 1, 1);
  // Four cold forwards: P and P+1 (both the interface's GEMV family when P+1 <= 8),
  // P+extra and P+7 (cuBLASLt when P+extra > 8) — the pairs within a family
  // isolate the interface from everything else that could depend on T.
  const Glm4Model::Outputs a = m.forward(ids, true);
  const Glm4Model::Outputs a1 = m.forward(longer(1), true);
  const Glm4Model::Outputs b = m.forward(longer(extra), true);
  const Glm4Model::Outputs b7 = m.forward(longer(7), true);
  profile_pair(cfg, a, a1, P, "forward(P) vs forward(P+1)");
  profile_pair(cfg, b, b7, P, "forward(P+extra) vs forward(P+7)");
  profile_pair(cfg, a, b, P, "forward(P) vs forward(P+extra)");
  return 0;
}

int run_checkpoint(const std::string& dir, const std::vector<int64_t>& ids, int steps) {
  const Glm4TextConfig cfg = Glm4TextConfig::from_json_file((fs::path(dir) / "config.json").string());
  const int P = static_cast<int>(ids.size());
  Glm4Model m(cfg, dir, /*max_tokens=*/P + steps, /*max_cache_tokens=*/P + steps + 64,
              Glm4Residency::Streaming, nullptr, 0, 1, 1);
  const int V = m.lm_vocab_count();
  std::printf("[ .. ] %s: %d prompt tokens, %d greedy steps (streaming world 1)\n", dir.c_str(), P, steps);
  const Glm4Model::Outputs fP = m.forward(ids);
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
  const Glm4Model::Outputs fA = m.forward(all);
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
  std::printf("[ OK ] glm4_decode_test (real checkpoint)\n");
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
    if (!fixture.empty()) return run_fixture(fixture);
    if (!checkpoint.empty()) {
      std::vector<int64_t> ids;
      std::stringstream ss(ids_text);
      std::string item;
      while (std::getline(ss, item, ',')) if (!item.empty()) ids.push_back(std::stoll(item));
      if (ids.empty()) throw std::runtime_error("--ids is required with --checkpoint-dir");
      if (layers_extra >= 0) return run_layers(checkpoint, ids, layers_extra);
      return run_checkpoint(checkpoint, ids, steps);
    }
    std::fprintf(stderr, "usage: --fixture DIR | --checkpoint-dir DIR --ids 1,2,... [--steps N]\n");
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[FAIL] %s\n", e.what());
    return 1;
  }
}
