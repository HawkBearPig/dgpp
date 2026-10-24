// The Qwen session surface's gate (Q6 stage A, 2026-09-09): incremental
// decode over QwenModel's request slots against the cold re-forward.
//
//   --fixture DIR                the fixture gates (tests/cuda/qwen_fixture.hpp's
//                                checkpoint, written by qwen_forward_test)
//   --checkpoint-dir DIR --ids 1,2,... [--steps N]
//                                the real checkpoint (streaming, world 1): a
//                                greedy transcript and its re-forward audit
//
// Gates on the fixture: a one-shot prefill's last row is bitwise the cold
// forward's (the same accumulation order on the same state); T=1 steps
// agree with the re-forward's rows at every position under the near-tie rule (their
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
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/bf16_residency.hpp"
#include "common/dtypes.hpp"
#include "engine/speculative.hpp"
#include "kernels/draft_head.hpp"
#include "kernels/gemm.hpp"
#include "models/qwen/config.hpp"
#include "models/qwen/forward.hpp"

namespace fs = std::filesystem;
using dgpp::QwenModel;
using dgpp::QwenResidency;
using dgpp::QwenTextConfig;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

std::vector<int64_t> smoke_tokens(const QwenTextConfig& cfg, int n, uint64_t seed) {
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
// near-tie certification (qwen_forward_test's rule). A draft row through
// the head screen (--draft-screen) holds -inf outside its candidate
// blocks, so both measures run over the columns finite in BOTH rows: the
// screen of a tolerance-equal input may keep a slightly different block
// set, and a column one row masked says nothing about the other's value.
// A row without -inf compares exactly as before.
RowCompare compare_row(const float* got, const float* want, int n) {
  RowCompare c;
  double d2 = 0, w2 = 0;
  int a = -1, b = -1;
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(got[i]) || !std::isfinite(want[i])) continue;
    const double d = static_cast<double>(got[i]) - want[i];
    d2 += d * d;
    w2 += static_cast<double>(want[i]) * want[i];
    if (a < 0 || want[i] > want[a]) a = i;
    if (b < 0 || got[i] > got[b]) b = i;
  }
  c.l2 = std::sqrt(d2) / std::sqrt(w2 + 1e-30);
  c.top1_equal = a == b;
  if (!c.top1_equal && a >= 0 && b >= 0) {
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

Transcript greedy(QwenModel& m, int req, const std::vector<int64_t>& prompt, int steps) {
  Transcript t;
  QwenModel::Outputs o = m.session_prefill(req, prompt);
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
int audit(QwenModel& ref, const std::vector<int64_t>& prompt, const Transcript& t, const char* what,
          double l2_budget) {
  std::vector<int64_t> all(prompt);
  all.insert(all.end(), t.tokens.begin(), t.tokens.end() - 1);
  const QwenModel::Outputs f = ref.forward(all);
  const int V = f.lm_vocab_count;
  const size_t P = prompt.size();
  int hard = 0, soft = 0;
  double worst_l2 = 0;
  std::string per_row;
  for (size_t i = 0; i < t.rows.size(); ++i) {
    const float* want = f.logits.data() + (P - 1 + i) * static_cast<size_t>(V);
    const RowCompare c = compare_row(t.rows[i].data(), want, V);
    worst_l2 = std::max(worst_l2, c.l2);
    if (!c.top1_equal) (c.near_tie ? soft : hard) += 1;
    char buf[32];
    std::snprintf(buf, sizeof buf, "%s%.2g", i ? " " : "", c.l2);
    per_row += buf;
  }
  std::printf("[ .. ] %s: %zu rows vs the re-forward — worst relative l2 %.3g, top-1 hard %d near-tie %d (per row: %s)\n",
              what, t.rows.size(), worst_l2, hard, soft, per_row.c_str());
  require(hard == 0, std::string(what) + ": a top-1 mismatch beyond the near-tie margin");
  require(worst_l2 < l2_budget, std::string(what) + ": relative l2 over budget");
  return soft;
}

// FP8/packed heads retain the full product arithmetic. BF16 heads may
// reassociate, but hidden states and every subsequent fixed-token decode
// must remain exact; the isolated BF16 head is also checked against FP64.
int run_prefill_head(const std::string& dir) {
  const QwenTextConfig cfg = QwenTextConfig::from_json_file((fs::path(dir) / "config.json").string());
  const bool old_fp8 = dgpp::QwenLayerStream::dense_weights_fp8();
  struct RestoreDenseWeights {
    bool fp8;
    ~RestoreDenseWeights() { dgpp::QwenLayerStream::set_dense_weights_fp8(fp8); }
  } restore{old_fp8};
  for (const bool fp8 : {false, true}) {
    if (cfg.dense_fp8_shipped && !fp8) continue;  // the hybrid ships fp8: no bf16 dense form
    dgpp::QwenLayerStream::set_dense_weights_fp8(fp8);
    for (const bool mma : {false, true}) {
      if (!fp8 && mma) continue;
      // Capacity must exceed the longest prompt after alignment: 257 would
      // round the prefill chunk to 256 and compare two walks with one.
      QwenModel model(cfg, dir, 4096, 8192, QwenResidency::Resident, nullptr, 0, 1, 2, false, 16, mma);
      QwenModel compact(cfg, dir, 4096, 8192, QwenResidency::Resident, nullptr, 0, 1, 2, false, 16, mma, true);
      const char* all = std::getenv("DGPP_PREFILL_HEAD_ALL_ROWS");
      const bool packed = !(all && all[0] == '1');
      require(compact.logits_capacity_rows() == (packed ? 17 : 4096), "compact head capacity");
      const auto plan_full = QwenModel::plan_memory(cfg, 4096, 8192, 0, 1, QwenResidency::Resident, 2, false, 16);
      const auto plan_compact = QwenModel::plan_memory(cfg, 4096, 8192, 0, 1, QwenResidency::Resident, 2, false, 16, true);
      const int vocab = model.lm_vocab_count();
      require(plan_full.total_bytes() - plan_compact.total_bytes() ==
              (packed ? size_t{4096 - 17} * vocab * sizeof(float) : 0), "compact head memory plan");
      for (const int length : {1, 4, 5, 8, 9, 16, 17, 127, 128, 129, 257, 1024, 4096}) {
        const auto prompt = smoke_tokens(cfg, length, 0x4300 + length);
        const auto full = model.forward(prompt);
        const auto prefill = model.session_prefill(0, prompt);
        const auto selected = compact.session_prefill(0, prompt);
        const auto cmp = compare_row(selected.logits.data(), prefill.logits.data(), vocab);
        require(fp8 ? bitwise(selected.logits, prefill.logits) : cmp.l2 < 1e-5,
                "compact prefill head exceeded rounding budget");
        require(selected.final_hidden_bits == prefill.final_hidden_bits, "compact prefill changed hidden state");
        compact.session_close(0);
        require(bitwise(prefill.logits, std::vector<float>(full.logits.end() - vocab, full.logits.end())),
                "prefill head differs from full head: T=" + std::to_string(length) +
                    " fp8=" + std::to_string(fp8) + " mma=" + std::to_string(mma));
        require(std::equal(prefill.final_hidden_bits.begin(), prefill.final_hidden_bits.end(),
                           full.final_hidden_bits.end() - cfg.hidden_size),
                "prefill head comparison received different hidden states");
        model.session_close(0);
      }
      for (const int length : {1, 8, 17, 129, 257}) {
        const auto a = smoke_tokens(cfg, length, 0xAB00 + length);
        const auto b = smoke_tokens(cfg, length + 1, 0xCD00 + length);
        const auto full = model.session_prefill_group({0, 1}, {&a, &b});
        const auto selected = compact.session_prefill_group({0, 1}, {&a, &b});
        for (int slot = 0; slot < 2; ++slot) {
          const auto cmp =
              compare_row(selected[slot].logits.data(), full[slot].logits.data(), vocab);
          require(fp8 ? bitwise(full[slot].logits, selected[slot].logits) : cmp.l2 < 1e-5,
                  "compact grouped head exceeded rounding budget");
          require(full[slot].final_hidden_bits == selected[slot].final_hidden_bits, "compact grouped head hidden state");
          int64_t token = argmax(full[slot].logits.data(), vocab);
          for (int step = 0; step < 4; ++step) {
            const auto f = model.session_step(slot, token), c = compact.session_step(slot, token);
            require(bitwise(f.logits, c.logits), "decode after compact grouped prefill changed logits");
            token = argmax(f.logits.data(), vocab);
          }
          model.session_close(slot);
          compact.session_close(slot);
        }
      }
    }
  }
  std::printf("[ OK ] BF16/FP8 compact and full prefill heads match through 4096 rows, groups and decode\n");
  return 0;
}

// ---- the fixture gates ----------------------------------------------------------
int run_fixture(const std::string& dir, bool fp8_head = false) {
  const QwenTextConfig cfg = QwenTextConfig::from_json_file((fs::path(dir) / "config.json").string());
  if (cfg.dense_fp8_shipped) dgpp::QwenLayerStream::set_dense_weights_fp8(true);
  const std::vector<int64_t> A = smoke_tokens(cfg, 23, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 17, 0xD1B54A32D192ED03ull);
  QwenModel m(cfg, dir, /*max_tokens=*/64, /*max_cache_tokens=*/256, QwenResidency::Resident, nullptr,
              0, 1, /*max_requests=*/2);
  const int V = m.lm_vocab_count();

  // 1. prefill == forward, bitwise.
  {
    const QwenModel::Outputs f = m.forward(A);
    const QwenModel::Outputs p = m.session_prefill(0, A);
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
  //     prefills alone (2026-09-14): the GDN scan and the QSA attention run
  //     per span over their own request's state and cache; the dense sites
  //     see 40 rows instead of 23 and 17 (cuBLASLt's algorithm at each:
  //     kernels/gemm.hpp dense_gemv_rows), so the rows are tolerance-equal
  //     under the row compare's l2 and near-tie rule, and a decode off the
  //     group's cache is audited against the re-forward like any other.
  {
    const QwenModel::Outputs pa = m.session_prefill(0, A);
    m.session_close(0);
    const QwenModel::Outputs pb = m.session_prefill(1, B);
    m.session_close(1);
    const std::vector<QwenModel::Outputs> g = m.session_prefill_group({0, 1}, {&A, &B});
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
      const QwenModel::Outputs o = m.session_step(1, pending);
      pending = argmax(o.logits.data(), V);
      tb.tokens.push_back(pending);
      tb.rows.push_back(o.logits);
    }
    m.session_close(0);
    m.session_close(1);
    const int soft = audit(m, B, tb, "decode after the group prefill", 1e-1);
    std::printf("[ OK ] group prefill: the spans' rows and an 8-step decode off the group's cache (%d near ties)\n", soft);
  }

  // 2. Incremental decode vs the re-forward.
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
    QwenModel::Outputs oa = m.session_prefill(0, A);
    QwenModel::Outputs ob = m.session_prefill(1, B);
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
    QwenModel c(cfg, dir, /*max_tokens=*/8, /*max_cache_tokens=*/256, QwenResidency::Resident, nullptr,
                0, 1, /*max_requests=*/1);
    const QwenModel::Outputs one = m.session_prefill(0, A);
    m.session_close(0);
    const QwenModel::Outputs p = c.session_prefill(0, A);
    RowCompare r = compare_row(p.logits.data(), one.logits.data(), V);
    std::printf("[ .. ] chunked prefill (8-row chunks): relative l2 %.3g, top-1 %s\n", r.l2,
                r.top1_equal ? "equal" : (r.near_tie ? "near tie" : "MISMATCH"));
    require(r.top1_equal || r.near_tie, "chunked prefill: top-1 mismatch");
    require(r.l2 < 2e-2, "chunked prefill: relative l2 over budget");
    // Boundary cuts: the pool-aligned image of 13 (12) joins the 8-multiples.
    c.session_close(0);
    const QwenModel::Outputs pb = c.session_prefill(0, A, std::vector<int64_t>{13});
    r = compare_row(pb.logits.data(), one.logits.data(), V);
    require(r.top1_equal || r.near_tie, "chunked prefill with a boundary cut: top-1 mismatch");
    require(r.l2 < 2e-2, "chunked prefill with a boundary cut: relative l2 over budget");
    // Decode continues from the chunked state.
    int64_t pending = argmax(pb.logits.data(), V);
    Transcript tc;
    tc.tokens.push_back(pending);
    tc.rows.push_back(pb.logits);
    for (int s = 0; s < 6; ++s) {
      const QwenModel::Outputs o = c.session_step(0, pending);
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
    const QwenModel::Outputs p = m.session_prefill(0, A);
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
    QwenModel::Outputs o = m.session_step(0, tA.tokens[0]);
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
    const QwenModel::Outputs f = m.forward(B);
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
    QwenModel::SessionSnapshotMeta meta;
    QwenModel::SnapshotRequest snap;
    snap.position = 12;
    snap.dst = arena[0];
    snap.meta = &meta;
    const QwenModel::Outputs cold = m.session_prefill(0, A, bounds, &snap);
    require(snap.taken && meta.position == 12, "prefix: the snapshot was taken at the cut");
    // Position 12 sits inside block 0: the entry owns a COPY of the partial
    // block beside the request's own (two in use), and keeps it after the
    // request closes.
    require(m.kv_blocks_in_use() == 2, "prefix: the entry copied the partial block");
    const QwenModel::Outputs cold_step = m.session_step(0, tA.tokens[0]);
    m.session_close(0);
    require(m.kv_blocks_in_use() == 1, "prefix: the entry keeps its partial block after the close");
    m.session_attach(1, arena[0], meta);
    require(m.session_position(1) == 12, "prefix: attached at the snapshot position");
    const QwenModel::Outputs hot = m.session_prefill_resume(1, std::vector<int64_t>(A.begin() + 12, A.end()), bounds);
    // Hot == cold bitwise (the same chunks on the same state). The plain
    // one-chunk prefill is a different GEMM shape (its m), so it is only
    // near the two-chunk one — the chunking gate above covers that.
    require(bitwise(hot.logits, cold.logits), "prefix: the hot prefill's last row differs from the cold one's");
    const QwenModel::Outputs hot_step = m.session_step(1, tA.tokens[0]);
    require(bitwise(hot_step.logits, cold_step.logits), "prefix: the first step after the attach differs");
    // A mid-decode snapshot at a pool-aligned position: 23 + 5 steps = 28.
    for (int s = 1; s < 5; ++s) (void)m.session_step(1, tA.tokens[static_cast<size_t>(s)]);
    require(m.session_position(1) == 28, "prefix: position 28");
    QwenModel::SessionSnapshotMeta meta2 = m.session_snapshot(1, arena[1]);
    const QwenModel::Outputs cont = m.session_step(1, tA.tokens[5]);
    m.session_close(1);
    m.session_attach(0, arena[1], meta2);
    const QwenModel::Outputs re = m.session_step(0, tA.tokens[5]);
    require(bitwise(re.logits, cont.logits), "prefix: the step after a mid-decode attach differs");
    m.session_close(0);
    m.session_release_snapshot(meta);
    m.session_release_snapshot(meta2);
    require(m.kv_blocks_in_use() == 0, "prefix: every block released with the entries");
    for (uint8_t* p : arena) cudaFree(p);
    std::printf("[ OK ] prefix snapshots: hot == cold bitwise at a cut and mid-decode\n");
  }
  // Two cuts in one prefill: an earlier document state for changed tails,
  // and the deepest state for identical repeats. Both include MTP state.
  {
    QwenModel cached(cfg, dir, 64, 512, QwenResidency::Resident, nullptr, 0, 1, 2, true);
    std::vector<void*> arena(2, nullptr);
    for (auto& p : arena) DGPP_CUDA_OK(cudaMalloc(&p, cached.session_snapshot_bytes()));
    QwenModel::SessionSnapshotMeta body_meta, final_meta;
    QwenModel::SnapshotRequest final{20, arena[1], &final_meta, false};
    QwenModel::SnapshotRequest body{8, arena[0], &body_meta, false, &final};
    const std::vector<int64_t> cuts{8, 16, 20};
    const auto original = cached.session_prefill(0, A, cuts, &body);
    require(body.taken && final.taken, "both prefill snapshots were taken");
    cached.session_close(0);
    for (int cut : {8, 20}) {
      auto prompt = A;
      if (cut == 8)
        for (size_t i = 9; i < prompt.size(); ++i) prompt[i] = (prompt[i] + 17) % cfg.vocab_size;
      const auto cold = cached.session_prefill(0, prompt, cuts);
      if (cut == 20)
        require(bitwise(cold.logits, original.logits), "taking two snapshots is invisible");
      cached.session_attach(1, arena[cut == 8 ? 0 : 1], cut == 8 ? body_meta : final_meta);
      const auto hot = cached.session_prefill_resume(
          1, std::vector<int64_t>(prompt.begin() + cut, prompt.end()), cuts);
      require(bitwise(hot.logits, cold.logits),
              "two snapshots: changed-tail and identical target parity");
      const auto token = argmax(cold.logits.data(), V);
      require(
          bitwise(cached.session_draft(0, {token}).logits, cached.session_draft(1, {token}).logits),
          "two snapshots: changed-tail and identical MTP parity");
      require(bitwise(cached.session_step(0, token).logits, cached.session_step(1, token).logits),
              "two snapshots: subsequent decode parity");
      cached.session_close(0);
      cached.session_close(1);
    }
    cached.session_release_snapshot(body_meta);
    cached.session_release_snapshot(final_meta);
    body.taken = final.taken = false;
    auto cursor = cached.session_prefill_begin(0, A, 64, 4, {}, &body);
    for (const auto& [budget, position] :
         std::vector<std::pair<int64_t, int64_t>>{{64, 8}, {8, 16}, {64, 20}, {64, 23}}) {
      const bool done = cached.session_prefill_advance(cursor, budget);
      require(cursor.next == position && done == (position == 23),
              "coalescing preserves both requested cuts");
    }
    require(body.taken && final.taken && bitwise(cursor.output.logits, original.logits),
            "resizing a continuation takes both snapshots without changing its state");
    cached.session_close(0);
    cached.session_release_snapshot(body_meta);
    cached.session_release_snapshot(final_meta);
    require(cached.kv_blocks_in_use() == 0, "two snapshots release every KV reference");
    for (auto p : arena) DGPP_CUDA_OK(cudaFree(p));
    std::printf(
        "[ OK ] two prefill snapshots: changed tails and identical repeats, target/draft "
        "bitwise\n");
  }
  // 7. The MTP draft block, eagerly (the greedy speculator): the committed
  //    transcript is the plain greedy one exactly (a verify's rows are the
  //    steps' rows; a rejected draft's rows roll back), the draft rate
  //    reported. Then close/reopen through the draft's counter.
  {
    QwenModel d(cfg, dir, /*max_tokens=*/64, /*max_cache_tokens=*/256, QwenResidency::Resident, nullptr, 0, 1,
                /*max_requests=*/2, /*mtp=*/true);
    require(d.mtp_enabled(), "mtp: enabled");
    const auto pick_rows = [](const std::vector<dgpp::sample::Candidate>& c) {
      std::vector<int32_t> ids;
      for (const auto& x : c) ids.push_back(x.id);
      return ids;
    };
    for (int round = 0; round < 2; ++round) {
      const QwenModel::Outputs p = d.session_prefill(1, A);
      require(bitwise(p.logits, tA.rows[0]), "mtp: the prefill's last row differs from the plain model's");
      require(d.session_draft_position(1) == static_cast<int64_t>(A.size()) - 1, "mtp: the draft trails by one after the prefill");
      // The speculator commits the pending token with each verify; a
      // random-weight fixture drafts by chance only, so the first draft is
      // FORCED to the known next token: that step must accept it (two
      // tokens committed, no rollback), later ones roll their rejects back.
      dgpp::GreedySpeculator<QwenModel> spec(d, 1, pick_rows);
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
  // Teacher-forced target verification across the GEMV boundary through sixteen rows. The
  // production dense GEMM lowering reassociates the sums across shapes;
  // use the existing logit/margin budget and bound the mean loss change.
  // The separate row-independent graph lane retains exact MTP/plain parity.
  {
    const bool old_fp8 = dgpp::QwenLayerStream::dense_weights_fp8();
    struct RestoreDenseWeights {
      bool fp8;
      ~RestoreDenseWeights() { dgpp::QwenLayerStream::set_dense_weights_fp8(fp8); }
    } restore_dense_weights{old_fp8};
    if (fp8_head) dgpp::QwenLayerStream::set_dense_weights_fp8(true);
    QwenModel reference(cfg, dir, 64, 2048, QwenResidency::Resident, nullptr, 0, 1, 8, false, 16,
                        fp8_head);
    QwenModel wide(cfg, dir, 64, 2048, QwenResidency::Resident, nullptr, 0, 1, 8, false, 16,
                   fp8_head);
    require(reference.max_decode_rows() == wide.max_decode_rows() &&
                reference.fp8_head_mma() == wide.fp8_head_mma(),
            "eager/graph parity requires matching decode capacity and head mode");
    wide.session_graph_prepare();
    const auto nll = [V](const float* row, int64_t token) {
      const double top = *std::max_element(row, row + V);
      double sum = 0;
      for (int i = 0; i < V; ++i) sum += std::exp(static_cast<double>(row[i]) - top);
      return top + std::log(sum) - row[token];
    };
    if (fp8_head && dgpp::dense_gemv_rows() == 4) {
      // The default head retains the GEMV chunks; the mma head takes the
      // streaming form at every row count (2026-10-05: one chain whatever the
      // prompt length or the batch; the GEMV chunks below five rows and
      // above the decode ceiling were two more chains). The minimum-capacity
      // MMA control must read bitwise the wide one at every length. Compare
      // hidden states to establish that these comparisons isolate the head.
      QwenModel old_head(cfg, dir, 64, 2048, QwenResidency::Resident, nullptr, 0, 1, 8, false, 16);
      QwenModel narrow(cfg, dir, 64, 2048, QwenResidency::Resident, nullptr, 0, 1, 1, false, 4,
                       true);
      require(!old_head.fp8_head_mma(), "the default head must remain GEMV");
      for (const int length : {1, 4, 5, 8, 9, 16, 17}) {
        double prefill_loss_delta = 0;
        for (int trial = 0; trial < 8; ++trial) {
          auto prompt = smoke_tokens(cfg, length + 1, 9000 + 100 * length + trial);
          const int64_t label = prompt.back();
          prompt.pop_back();
          const auto expected = old_head.session_prefill(0, prompt);
          const auto actual = wide.session_prefill(0, prompt);
          const auto narrow_output = narrow.session_prefill(0, prompt);
          require(narrow.max_decode_rows() == dgpp::kDecodeRows,
                  "minimum-capacity control must expose the session-core floor");
          require(narrow_output.final_hidden_bits == actual.final_hidden_bits &&
                      bitwise(narrow_output.logits, actual.logits),
                  "the MMA head must be one chain whatever the decode capacity");
          narrow.session_close(0);
          require(actual.logits.size() == static_cast<size_t>(V) &&
                      expected.logits.size() == static_cast<size_t>(V),
                  "FP8 short prefill: one vocabulary row");
          require(actual.final_hidden_bits.size() == static_cast<size_t>(cfg.hidden_size) &&
                      actual.final_hidden_bits == expected.final_hidden_bits,
                  "FP8 short prefill: head controls received different hidden states");
          for (int col = 0; col < V; ++col)
            require(std::isfinite(actual.logits[col]) && std::isfinite(expected.logits[col]),
                    "FP8 short prefill: nonfinite logit");
          const auto comparison = compare_row(actual.logits.data(), expected.logits.data(), V);
          require(comparison.l2 < 2e-2, "FP8 short prefill exceeds the logit L2 budget");
          require(comparison.top1_equal || comparison.near_tie,
                  "FP8 short prefill changes top-1 beyond the near-tie margin");
          prefill_loss_delta +=
              nll(actual.logits.data(), label) - nll(expected.logits.data(), label);
          wide.session_close(0);
          const auto repeated = wide.session_prefill(0, prompt);
          require(bitwise(actual.logits, repeated.logits), "FP8 short prefill must repeat bitwise");
          old_head.session_close(0);
          wide.session_close(0);
        }
        std::printf("[ .. ] FP8 prefill length %d: mean NLL delta %.6g\n", length,
                    prefill_loss_delta / 8);
        require(std::abs(prefill_loss_delta / 8) < 0.02,
                "FP8 short prefill exceeds the mean NLL budget");
      }
    }
    double loss_delta = 0, worst_l2 = 0;
    int rows = 0, near_ties = 0;
    for (const int count : {2, 3, 4, 6, 8})
      for (int trial = 0; trial < 8; ++trial) {
        std::vector<std::vector<float>> expected;
        std::vector<std::vector<int64_t>> feeds;
        std::vector<int64_t> labels;
        for (int req = 0; req < count; ++req) {
          const auto prompt = smoke_tokens(cfg, 11 + req, 4000 + 100 * trial + req);
          const auto teacher = smoke_tokens(cfg, 3, 7000 + 100 * trial + req);
          (void)reference.session_prefill(req, prompt);
          (void)wide.session_prefill(req, prompt);
          wide.session_reserve_blocks(req, 64);
          for (int row = 0; row < 2; ++row) {
            expected.push_back(reference.session_step(req, teacher[row]).logits);
            labels.push_back(teacher[row + 1]);
          }
          feeds.push_back({teacher[0], teacher[1]});
        }
        // Prefill uses the token-feed buffer as scratch. Seed every feed
        // after the last prefill, as GraphEngineAdapter does before replay.
        for (int req = 0; req < count; ++req) wide.session_graph_seed_feed(req, feeds[req]);
        // Read target logits before any picker or draft can overwrite them.
        // The FP8 lane also validates capture and the selected head kernel.
        if (fp8_head) {
          cudaGraph_t graph = nullptr;
          cudaGraphExec_t executable = nullptr;
          DGPP_CUDA_OK(cudaStreamBeginCapture(wide.stream(), cudaStreamCaptureModeGlobal));
          wide.session_graph_capture_batch(2, count);
          DGPP_CUDA_OK(cudaStreamEndCapture(wide.stream(), &graph));
          // Find the streaming kernel writing the vocabulary logits, rather
          // than counting MMA kernels used by other dense projections.
          size_t node_count = 0;
          DGPP_CUDA_OK(cudaGraphGetNodes(graph, nullptr, &node_count));
          std::vector<cudaGraphNode_t> nodes(node_count);
          DGPP_CUDA_OK(cudaGraphGetNodes(graph, nodes.data(), &node_count));
          int streaming_heads = 0;
          for (const auto node : nodes) {
            cudaGraphNodeType type;
            DGPP_CUDA_OK(cudaGraphNodeGetType(node, &type));
            if (type != cudaGraphNodeTypeKernel) continue;
            cudaKernelNodeParams params{};
            // cuBLAS may capture driver-loaded kernels without a registered
            // runtime function. They cannot be inspected through this API;
            // our statically linked head kernel must still be found below.
            const auto status = cudaGraphKernelNodeGetParams(node, &params);
            if (status == cudaErrorInvalidDeviceFunction) {
              (void)cudaGetLastError();
              continue;
            }
            DGPP_CUDA_OK(status);
            const char* name = nullptr;
            DGPP_CUDA_OK(cudaFuncGetName(&name, params.func));
            if (std::string(name).find("mma_gemv_kernel") == std::string::npos) continue;
            // mma_gemv_kernel's fifth argument is its output pointer.
            const auto output = *static_cast<void**>(params.kernelParams[4]);
            if (output == wide.device_logits()) ++streaming_heads;
          }
          // engine.fp8_head mma: the streaming head at every decode row
          // count inside the envelope (2026-10-05: one chain 1..64, so a
          // request's logits are the same alone and in a batch); the GEMV
          // chunks were the rows up to dense_gemv_rows before.
          const bool expect_streaming = 2 * count <= wide.max_decode_rows();
          const bool dispatch_ok = streaming_heads == (expect_streaming ? 1 : 0);
          DGPP_CUDA_OK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
          DGPP_CUDA_OK(cudaGraphLaunch(executable, wide.stream()));
          DGPP_CUDA_OK(cudaStreamSynchronize(wide.stream()));
          DGPP_CUDA_OK(cudaGraphExecDestroy(executable));
          DGPP_CUDA_OK(cudaGraphDestroy(graph));
          require(dispatch_ok, "FP8 vocabulary head did not take the streaming form at every decode row count");
        } else {
          wide.session_graph_capture_batch(2, count);
        }
        DGPP_CUDA_OK(cudaStreamSynchronize(wide.stream()));
        std::vector<float> got(static_cast<size_t>(2 * count) * V);
        DGPP_CUDA_OK(cudaMemcpy(got.data(), wide.device_logits(), got.size() * sizeof(float),
                                cudaMemcpyDeviceToHost));
        for (int row = 0; row < 2 * count; ++row) {
          const float* actual = got.data() + static_cast<size_t>(row) * V;
          const auto comparison = compare_row(actual, expected[row].data(), V);
          require(comparison.l2 < 2e-2, "wide target verification exceeds the logit l2 budget: " +
                                            std::to_string(comparison.l2) + " at width " +
                                            std::to_string(2 * count) + " row " +
                                            std::to_string(row));
          require(comparison.top1_equal || comparison.near_tie,
                  "wide target verification changes a top-1 decision beyond the near-tie margin");
          worst_l2 = std::max(worst_l2, comparison.l2);
          near_ties += !comparison.top1_equal;
          loss_delta += nll(actual, labels[row]) - nll(expected[row].data(), labels[row]);
          ++rows;
        }
        for (int req = 0; req < count; ++req) {
          reference.session_close(req);
          wide.session_close(req);
        }
      }
    std::printf("[ .. ] wide teacher-forced target: %d rows, worst l2 %.5g, %d near ties, mean NLL delta %.6g\n",
                rows, worst_l2, near_ties, loss_delta / rows);
    require(std::abs(loss_delta / rows) < 0.02, "wide target verification exceeds the mean NLL budget");
  }
  // A continuation sees the same cuts as the monolithic reference, with
  // another request decoding between every pair of chunks. Check logits
  // and draft state, not just the generated text.
  {
    QwenModel reference(cfg, dir, 64, 512, QwenResidency::Resident, nullptr, 0, 1, 2, true);
    QwenModel yielded(cfg, dir, 64, 512, QwenResidency::Resident, nullptr, 0, 1, 2, true);
    const std::vector<int64_t> cuts{4, 8, 12, 16, 20};
    const auto expected = reference.session_prefill(0, A, cuts);
    const auto peer = yielded.session_prefill(1, B);
    int64_t next = argmax(peer.logits.data(), V);
    auto cursor = yielded.session_prefill_begin(0, A, 64, 4);
    int chunks = 0;
    while (!yielded.session_prefill_advance(cursor)) {
      require(yielded.session_position(0) == 4 * ++chunks, "continuation position after a yield");
      const auto row = yielded.session_step(1, next);
      next = argmax(row.logits.data(), V);
    }
    require(bitwise(cursor.output.logits, expected.logits), "yielding prefill changes target logits");
    const int64_t first = argmax(expected.logits.data(), V);
    const auto draft_ref = reference.session_draft(0, {first});
    const auto draft_yielded = yielded.session_draft(0, {first});
    require(bitwise(draft_ref.logits, draft_yielded.logits), "yielding prefill changes draft logits");
    const auto step_ref = reference.session_step(0, first);
    const auto step_yielded = yielded.session_step(0, first);
    require(bitwise(step_ref.logits, step_yielded.logits), "yielding prefill changes continuation state");
    yielded.session_close(0);
    auto cancelled = yielded.session_prefill_begin(0, A, 64, 4);
    require(!yielded.session_prefill_advance(cancelled), "long prompt yields");
    yielded.session_close(0);
    auto reused = yielded.session_prefill_begin(0, A, 64, 4);
    while (!yielded.session_prefill_advance(reused)) {}
    require(bitwise(reused.output.logits, expected.logits), "cancelled prefill leaves stale state on slot reuse");
    yielded.session_close(0);
    yielded.session_close(1);
    reference.session_close(0);
    std::printf("[ OK ] resumable prefill: target/draft logits, interleaving and cancellation\n");
  }
  // The second grouped prompt's MTP input starts after the first span,
  // and a prompt longer than the hidden window must retain its final rows.
  // The observable is the draft's INPUT row through the whole head, so
  // these two models read the exact head whatever --draft-screen says: a
  // masked row (the screen's -inf outside its candidate blocks) cannot be
  // compared by relative l2 to another masked row of a tolerance-equal
  // input, and the screen's own gates are the bitwise ones above.
  {
    const dgpp::DraftHeadSettings screen = dgpp::draft_head_settings();
    dgpp::draft_head_configure(dgpp::DraftHeadSettings{0});
    QwenModel reference(cfg, dir, 128, 512, QwenResidency::Resident, nullptr, 0, 1, 2, true);
    QwenModel grouped(cfg, dir, 128, 512, QwenResidency::Resident, nullptr, 0, 1, 2, true);
    dgpp::draft_head_configure(screen);
    const auto long_b = smoke_tokens(cfg, 95, 107);
    (void)reference.session_prefill(1, long_b);
    const auto expected = reference.session_draft(1, {10});
    (void)grouped.session_prefill_group({0, 1}, {&A, &long_b});
    const auto got = grouped.session_draft(1, {10});
    const auto c = compare_row(got.logits.data(), expected.logits.data(), V);
    require(c.l2 < 2e-2 && (c.top1_equal || c.near_tie), "group MTP reads its own span and final window rows");
    grouped.session_close(0);
    grouped.session_close(1);
    reference.session_close(1);
    require(grouped.kv_blocks_in_use() == 0, "group MTP releases reservations");
    std::printf("[ OK ] grouped MTP: second span and wrapped hidden window\n");
  }
  // A mixed walk keeps every decoder's rollback rows even though a prefill
  // span follows them and then runs its MTP block. Exercise nonzero snapshot
  // offsets, every acceptance length, a QSA pool boundary and resumed chunks.
  for (int accepted = 1; accepted <= 4; ++accepted) {
    QwenModel reference(cfg, dir, 128, 1024, QwenResidency::Resident, nullptr, 0, 1, 3, true);
    QwenModel mixed(cfg, dir, 128, 1024, QwenResidency::Resident, nullptr, 0, 1, 4, true);
    QwenModel restored(cfg, dir, 128, 1024, QwenResidency::Resident, nullptr, 0, 1, 4, true);
    void* arena = nullptr;
    DGPP_CUDA_OK(cudaMalloc(&arena, restored.session_snapshot_bytes()));
    QwenModel::SessionSnapshotMeta meta;
    QwenModel::SnapshotRequest snapshot{48, arena, &meta, false};
    const auto prompt = smoke_tokens(cfg, 95, 104);
    const std::vector<int64_t> cuts{16, 48, 80};
    auto expected_cursor = reference.session_prefill_begin(0, prompt, 128, 32, cuts);
    auto cursor = mixed.session_prefill_begin(0, prompt, 128, 32, cuts);
    auto restored_cursor = restored.session_prefill_begin(0, prompt, 128, 32, cuts, &snapshot);
    bool attached = false;
    std::vector<std::vector<int64_t>> feeds{{1, 2, 3, 4}, {5, 6, 7, 8}};
    for (int req = 1; req <= 2; ++req) {
      (void)reference.session_prefill(req, req == 1 ? A : B);
      (void)mixed.session_prefill(req, req == 1 ? A : B);
      (void)restored.session_prefill(req, req == 1 ? A : B);
      reference.session_reserve_blocks(req, 128);
      mixed.session_reserve_blocks(req, 128);
      restored.session_reserve_blocks(req, 128);
    }
    for (int rows : {-1, dgpp::kSpecRows + 1}) {
      bool refused = false;
      try { (void)mixed.session_prefill_verify(cursor, 32, {1, 2}, feeds, rows); }
      catch (const std::invalid_argument&) { refused = true; }
      require(refused && cursor.next == 0 && mixed.session_position(0) == 0 &&
                  mixed.session_position(1) == static_cast<int64_t>(A.size()) &&
                  mixed.session_position(2) == static_cast<int64_t>(B.size()),
              "invalid mixed graph geometry must fail before advancing any request");
    }
    const auto compare = [&](const QwenModel::Outputs& got, const QwenModel::Outputs& want, const char* label) {
      require(got.logits.size() == want.logits.size(), std::string(label) + ": output rows");
      for (size_t at = 0; at < got.logits.size(); at += V) {
        const auto c = compare_row(got.logits.data() + at, want.logits.data() + at, V);
        require(c.l2 < 2e-2 && (c.top1_equal || c.near_tie),
                std::string(label) + ": l2=" + std::to_string(c.l2));
      }
    };
    while (cursor.next < cursor.end) {
      // The reader keeps its 32-token budget: the eight decode rows ride on
      // scratch beyond it (prefill_batch.hpp trims readers only past max_tokens).
      (void)reference.session_prefill_advance(expected_cursor, 32);
      const auto expected = reference.session_verify_batch({1, 2}, feeds);
      const auto actual = mixed.session_prefill_verify(cursor, 32, {1, 2}, feeds);
      const auto resumed = restored.session_prefill_verify(restored_cursor, 32, {1, 2}, feeds);
      require(cursor.next == expected_cursor.next, "mixed prefill preserves cuts and work budget");
      require(restored_cursor.next == cursor.next &&
                  bitwise(restored_cursor.output.logits, cursor.output.logits),
              "mixed snapshot restore: prefill logits differ at the same execution shape");
      compare(cursor.output, expected_cursor.output, "mixed prefill");
      for (int s = 0; s < 2; ++s) {
        compare(actual[s], expected[s], "mixed verify");
        require(bitwise(resumed[s].logits, actual[s].logits),
                "mixed snapshot restore changed a peer's verify logits");
        reference.session_rollback(s + 1, accepted, 4, s * 4);
        mixed.session_rollback(s + 1, accepted, 4, s * 4);
        restored.session_rollback(s + 1, accepted, 4, s * 4);
        require(mixed.session_position(s + 1) == reference.session_position(s + 1), "mixed rollback position");
      }
      for (int s = 0; s < 2; ++s) {
        const auto next = reference.session_step(s + 1, 9);
        const auto mixed_next = mixed.session_step(s + 1, 9);
        compare(mixed_next, next, "mixed rollback continuation");
        require(bitwise(restored.session_step(s + 1, 9).logits, mixed_next.logits),
                "mixed snapshot restore changed a peer's rollback continuation");
      }
      if (!attached && cursor.next == 48) {
        require(snapshot.taken && meta.position == 48, "mixed prefill snapshot was taken");
        restored.session_close(0);
        restored.session_attach(3, arena, meta);
        restored_cursor = restored.session_prefill_begin(3, prompt, 128, 32, cuts, nullptr, 48);
        attached = true;
      }
    }
    require(attached, "mixed snapshot restore was exercised in a different request slot");
    const auto draft = mixed.session_draft(0, {10});
    compare(draft, reference.session_draft(0, {10}), "mixed prefill draft");
    require(bitwise(restored.session_draft(3, {10}).logits, draft.logits),
            "mixed snapshot restore changed the resumed MTP logits");
    for (int req = 0; req < 3; ++req) {
      mixed.session_close(req);
      reference.session_close(req);
    }
    for (int req = 1; req <= 3; ++req) restored.session_close(req);
    restored.session_release_snapshot(meta);
    DGPP_CUDA_OK(cudaFree(arena));
    require(mixed.kv_blocks_in_use() == 0, "mixed prefill releases every reservation");
    require(restored.kv_blocks_in_use() == 0, "mixed snapshot restore releases every KV reference");
  }
  std::printf("[ OK ] mixed prefill/verify: chunk cuts, logits, MTP, rollback and bitwise snapshot restoration in another slot\n");
  // A donated share shifts the cursor off the model's original grid.
  // Subsequent work fills its cap except at a real structural/snapshot cut.
  {
    QwenModel reference(cfg, dir, 256, 512, QwenResidency::Resident, nullptr, 0, 1, 2, true);
    QwenModel yielded(cfg, dir, 32, 512, QwenResidency::Resident, nullptr, 0, 1, 2, true);
    std::vector<int64_t> prompt;
    for (int i = 0; i < 155; ++i) prompt.push_back(A[static_cast<size_t>(i) % A.size()]);
    const auto expected = reference.session_prefill(0, prompt, {32, 56, 80, 112, 144});
    uint8_t* arena = nullptr;
    DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&arena), yielded.session_snapshot_bytes()));
    QwenModel::SessionSnapshotMeta meta;
    QwenModel::SnapshotRequest snap{80, arena, &meta};
    auto cursor = yielded.session_prefill_begin(0, prompt, 192, 32, {80}, &snap);
    cursor.final_output_only = true;
    for (const auto& [budget, position] :
         std::vector<std::pair<int64_t, int64_t>>{{32, 32}, {24, 56}, {32, 80}, {32, 112}, {32, 144}, {32, 155}}) {
      const bool done = yielded.session_prefill_advance(cursor, budget);
      require(done || (cursor.output.logits.empty() && cursor.output.final_hidden_bits.empty()),
              "serving continuation must defer unused intermediate outputs");
      require(cursor.next == position && done == (position == 155),
              "work budget must not split again at an automatic model grid");
    }
    require(snap.taken && bitwise(cursor.output.logits, expected.logits),
            "work-budget continuation preserves the snapshot and target state");
    const int64_t token = argmax(expected.logits.data(), V);
    require(bitwise(reference.session_draft(0, {token}).logits, yielded.session_draft(0, {token}).logits),
            "work-budget continuation preserves draft state");
    yielded.session_release_snapshot(meta);
    yielded.session_close(0);
    reference.session_close(0);
    require(yielded.kv_blocks_in_use() == 0, "work-budget cursor releases every KV reference");
    DGPP_CUDA_OK(cudaFree(arena));
    std::printf("[ OK ] work-budget prefill: full shares, structural cuts, snapshot and target/draft parity\n");
  }
  // Change the budget in both directions while preserving a snapshot that
  // was legal only on the original budget grid. The oracle uses the actual
  // resulting cuts, so this checks continuation state rather than GEMM drift.
  {
    QwenModel reference(cfg, dir, 64, 512, QwenResidency::Resident, nullptr, 0, 1, 2, true);
    QwenModel yielded(cfg, dir, 64, 512, QwenResidency::Resident, nullptr, 0, 1, 2, true);
    uint8_t *ref_arena = nullptr, *yield_arena = nullptr;
    DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&ref_arena), reference.session_snapshot_bytes()));
    DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&yield_arena), yielded.session_snapshot_bytes()));
    QwenModel::SessionSnapshotMeta ref_meta, yield_meta;
    QwenModel::SnapshotRequest ref_snap, yield_snap;
    ref_snap.position = yield_snap.position = 12;
    ref_snap.dst = ref_arena;
    ref_snap.meta = &ref_meta;
    yield_snap.dst = yield_arena;
    yield_snap.meta = &yield_meta;
    const auto expected = reference.session_prefill(0, A, {4, 12, 16}, &ref_snap);
    auto peer = yielded.session_prefill(1, B);
    auto cursor = yielded.session_prefill_begin(0, A, 64, 4, {}, &yield_snap);
    for (const auto& [budget, position] : std::vector<std::pair<int64_t, int64_t>>{{0, 4}, {16, 12}, {0, 16}, {16, 23}}) {
      if (cursor.next == 4) {
        bool rejected = false;
        try { (void)yielded.session_prefill_advance(cursor, 3); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && cursor.next == 4 && cursor.suspended,
                "invalid resizing must not expose or advance a suspended slot");
      }
      const bool done = yielded.session_prefill_advance(cursor, budget);
      require(cursor.next == position && done == (position == 23), "resizing preserves mandatory cuts");
      if (!done) peer = yielded.session_step(1, argmax(peer.logits.data(), V));
    }
    require(yield_snap.taken && yield_meta.position == 12, "coalescing retains the snapshot cut");
    require(bitwise(cursor.output.logits, expected.logits), "resizing changes target logits");
    const int64_t first = argmax(expected.logits.data(), V);
    const auto expected_draft = reference.session_draft(0, {first});
    require(bitwise(yielded.session_draft(0, {first}).logits, expected_draft.logits),
            "resizing changes draft state");
    yielded.session_close(0);
    yielded.session_close(1);
    yielded.session_attach(0, yield_arena, yield_meta);
    auto resumed = yielded.session_prefill_begin(0, A, 64, 4, {}, nullptr, 12);
    require(!yielded.session_prefill_advance(resumed, 4) && resumed.next == 16, "attached first cut");
    require(yielded.session_prefill_advance(resumed, 16), "attached continuation completes");
    require(bitwise(resumed.output.logits, expected.logits), "resizing snapshot changes resumed target");
    require(bitwise(yielded.session_draft(0, {first}).logits, expected_draft.logits),
            "resizing snapshot changes resumed draft");
    yielded.session_close(0);
    reference.session_close(0);
    yielded.session_release_snapshot(yield_meta);
    reference.session_release_snapshot(ref_meta);
    DGPP_CUDA_OK(cudaFree(yield_arena));
    DGPP_CUDA_OK(cudaFree(ref_arena));
    require(yielded.kv_blocks_in_use() == 0 && reference.kv_blocks_in_use() == 0,
            "resized continuation releases all blocks");
    std::printf("[ OK ] resized prefill: target/draft logits, required snapshot and attach\n");
  }
  // engine.bf16_weights: the same checkpoint with the lossless 12-bit
  // companions of its GDN / QSA / draft projections and the head (packed at
  // session_graph_prepare; format v2 — the fixture's rows are all tail). The
  // GDN's four input projections then take the packed multi launch, every
  // other site the GEMM seam's packed GEMV; prefill keeps the bf16 bytes.
  // Prefill, the draft block and the decode that continues are bitwise the
  // bf16 build's, under either value of the key (this family keeps both
  // forms resident).
  // Under the fp8 dense stack (the AutoRound hybrid's fixture) nothing is
  // offered to the companions — the block is the FP8 fixture's gate.
  if (dgpp::QwenLayerStream::dense_weights_fp8()) {
    std::printf("[ .. ] bf16_weights: nothing packs under the fp8 dense stack; the FP8 fixture gates the companions\n");
  } else {
    struct Restore {
      ~Restore() { dgpp::set_bf16_residency(dgpp::Bf16Residency::Checkpoint); }
    } restore;
    const auto build = [&](dgpp::Bf16Residency mode) {
      dgpp::set_bf16_residency(mode);
      auto m = std::make_unique<QwenModel>(cfg, dir, 64, 512, QwenResidency::Resident, nullptr, 0, 1, 2, true);
      m->session_graph_prepare();
      return m;
    };
    const auto plain = build(dgpp::Bf16Residency::Checkpoint);
    const auto both = build(dgpp::Bf16Residency::Bf12AndBf16);
    const auto only = build(dgpp::Bf16Residency::Bf12);
    require(plain->bf12_companions().matrices() == 0, "bf12: checkpoint residency packs nothing");
    require(both->bf12_companions().matrices() > 0 && both->bf12_companions().kept_bf16() == 0,
            "bf12: every offered matrix packs");
    require(only->bf12_companions().matrices() == both->bf12_companions().matrices() &&
                only->bf12_companions().released() == 0,
            "bf12: this family keeps the bf16 bytes under either value");
    for (const std::vector<int64_t>* prompt : {&A, &B}) {
      const QwenModel::Outputs want = plain->session_prefill(0, *prompt);
      require(bitwise(want.logits, both->session_prefill(0, *prompt).logits), "bf12+bf16: prefill logits are bitwise");
      require(bitwise(want.logits, only->session_prefill(0, *prompt).logits), "bf12: prefill logits are bitwise");
      int64_t token = argmax(want.logits.data(), V);
      for (int step = 0; step < 6; ++step) {
        const QwenModel::Outputs d = plain->session_draft(0, {token});
        require(bitwise(d.logits, both->session_draft(0, {token}).logits), "bf12+bf16: the draft block is bitwise");
        require(bitwise(d.logits, only->session_draft(0, {token}).logits), "bf12: the draft block is bitwise");
        const QwenModel::Outputs o = plain->session_step(0, token);
        require(bitwise(o.logits, both->session_step(0, token).logits), "bf12+bf16: decode continues bitwise");
        require(bitwise(o.logits, only->session_step(0, token).logits), "bf12: decode continues bitwise");
        token = argmax(o.logits.data(), V);
      }
      plain->session_close(0);
      both->session_close(0);
      only->session_close(0);
    }
    std::printf("[ OK ] bf16_weights: %zu matrices packed; prefill, draft and decode bitwise the bf16 build\n",
                both->bf12_companions().matrices());
  }
  std::printf("[ OK ] qwen_decode_test\n");
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
void profile_pair(const QwenTextConfig& cfg, const QwenModel::Outputs& a, const QwenModel::Outputs& b, int P,
                  const char* what) {
  const int W = cfg.hc_count * cfg.hidden_size;
  const int K = cfg.num_experts_per_tok;
  std::printf("[ .. ] %s\n", what);
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    if (l > 2 && l + 1 < cfg.num_hidden_layers && l % 8 != 7) continue;
    std::string line = "[ .. ] layer " + std::to_string(l) + (cfg.layers[static_cast<size_t>(l)] == dgpp::QwenLayerKind::Gdn ? " gdn" : " qsa") + ": l2";
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
    int flips = 0;
    for (int r = 0; r < P; ++r)
      for (int j = 0; j < K; ++j)
        if (a.route_ids[static_cast<size_t>(l)][static_cast<size_t>(r) * K + j] !=
            b.route_ids[static_cast<size_t>(l)][static_cast<size_t>(r) * K + j])
          ++flips;
    line += "; route flips " + std::to_string(flips) + "/" + std::to_string(P * K);
    std::printf("%s\n", line.c_str());
    std::fflush(stdout);
  }
}

int run_layers(const std::string& dir, const std::vector<int64_t>& ids, int extra) {
  const QwenTextConfig cfg = QwenTextConfig::from_json_file((fs::path(dir) / "config.json").string());
  const int P = static_cast<int>(ids.size());
  const auto longer = [&](int n) {
    std::vector<int64_t> v(ids);
    for (int i = 0; i < n; ++i) v.push_back(ids[static_cast<size_t>(i % P)]);
    return v;
  };
  const int top = std::max(extra, 7);
  QwenModel m(cfg, dir, /*max_tokens=*/P + top, /*max_cache_tokens=*/P + top + 64,
              QwenResidency::Streaming, nullptr, 0, 1, 1);
  // Four cold forwards: P and P+1 (both the interface's GEMV family when P+1 <= 8),
  // P+extra and P+7 (cuBLASLt when P+extra > 8) — the pairs within a family
  // isolate the interface from everything else that could depend on T.
  const QwenModel::Outputs a = m.forward(ids, true);
  const QwenModel::Outputs a1 = m.forward(longer(1), true);
  const QwenModel::Outputs b = m.forward(longer(extra), true);
  const QwenModel::Outputs b7 = m.forward(longer(7), true);
  profile_pair(cfg, a, a1, P, "forward(P) vs forward(P+1)");
  profile_pair(cfg, b, b7, P, "forward(P+extra) vs forward(P+7)");
  profile_pair(cfg, a, b, P, "forward(P) vs forward(P+extra)");
  return 0;
}

int run_checkpoint(const std::string& dir, const std::vector<int64_t>& ids, int steps) {
  const QwenTextConfig cfg = QwenTextConfig::from_json_file((fs::path(dir) / "config.json").string());
  const int P = static_cast<int>(ids.size());
  QwenModel m(cfg, dir, /*max_tokens=*/P + steps, /*max_cache_tokens=*/P + steps + 64,
              QwenResidency::Streaming, nullptr, 0, 1, 1);
  const int V = m.lm_vocab_count();
  std::printf("[ .. ] %s: %d prompt tokens, %d greedy steps (streaming world 1)\n", dir.c_str(), P, steps);
  const QwenModel::Outputs fP = m.forward(ids);
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
  const QwenModel::Outputs fA = m.forward(all);
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
  std::printf("[ OK ] qwen_decode_test (real checkpoint)\n");
  return 0;
}

}  // namespace

// A decode step's rows are bitwise the same whatever the step's row count
// (2026-10-05, the Qwen3.8-Flash-Next family: the QSA projections took the
// fp8 GEMV core up to dense_gemv_rows rows and the streaming MMA above): the
// same context verified as 8 rows and as 1..7 rows, the first rows' logits
// and hidden bits compared bitwise.
int run_rows_invariance(const std::string& dir) {
  const QwenTextConfig cfg = QwenTextConfig::from_json_file((fs::path(dir) / "config.json").string());
  // The shipped recipes serve the dense projections as block FP8
  // (engine.dense_weights fp8): the fixture's bf16 dense sites would run on
  // cuBLASLt at every width, a different question.
  dgpp::QwenLayerStream::set_dense_weights_fp8(true);
  QwenModel::set_session_capture_layers(true);
  // Sixteen decode rows (the four-slot MTP-depth-3 batch's width, past the
  // MoE layer's eight-row fused tails and the GR site's fused rows): a
  // scalar verify is at most eight rows; the wider forms are reached by the
  // batched verify (two slots x eight rows, four x four).
  constexpr int kRows = 16, kSlots = 4;
  QwenModel m(cfg, dir, /*max_tokens=*/64, /*max_cache_tokens=*/512, QwenResidency::Resident, nullptr, 0, 1,
              /*max_requests=*/kSlots, /*mtp=*/false, /*decode_rows=*/kRows, /*fp8_head_mma=*/true);
  const int V = m.lm_vocab_count();
  const int H = cfg.hidden_size;
  const std::vector<int64_t> prompt = smoke_tokens(cfg, 23, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> block = smoke_tokens(cfg, 8, 0xD1B54A32D192ED03ull);
  const auto reset = [&](int slots) {
    for (int q = 0; q < slots; ++q) { m.session_close(q); (void)m.session_prefill(q, prompt); }
  };
  const auto solo = [&](int T) {
    reset(1);
    return m.session_verify(0, std::vector<int64_t>(block.begin(), block.begin() + T));
  };
  const QwenModel::Outputs full = solo(8);
  require(full.logits.size() == static_cast<size_t>(8) * V, "rows: the eight-row verify's logits");
  const QwenModel::Outputs again = solo(8);
  require(again.logits == full.logits, "rows: the eight-row verify is not deterministic across calls");
  std::vector<std::string> failures;
  const auto check = [&](const QwenModel::Outputs& got, int T, const std::string& what) {
    int bad = -1;
    for (int r = 0; r < T && bad < 0; ++r)
      if (std::memcmp(got.logits.data() + static_cast<size_t>(r) * V, full.logits.data() + static_cast<size_t>(r) * V,
                      static_cast<size_t>(V) * sizeof(float)) != 0 ||
          std::memcmp(got.final_hidden_bits.data() + static_cast<size_t>(r) * H,
                      full.final_hidden_bits.data() + static_cast<size_t>(r) * H, static_cast<size_t>(H) * 2) != 0)
        bad = r;
    if (bad >= 0) {
      int first_layer = -1;
      const int W = static_cast<int>(got.layer_states.empty() ? 0 : got.layer_states[0].size() / static_cast<size_t>(T));
      for (size_t l = 0; l < got.layer_states.size() && first_layer < 0; ++l)
        if (W > 0 && std::memcmp(got.layer_states[l].data() + static_cast<size_t>(bad) * W,
                                 full.layer_states[l].data() + static_cast<size_t>(bad) * W, static_cast<size_t>(W) * 2) != 0)
          first_layer = static_cast<int>(l);
      std::printf("[ !! ] %s: row %d differs from the eight-row solo verify's (first differing layer %d of %zu)\n", what.c_str(),
                  bad, first_layer, got.layer_states.size());
      failures.push_back(what);
    } else {
      std::printf("[ .. ] %s is bitwise the eight-row solo verify's rows\n", what.c_str());
    }
  };
  for (const int T : {1, 2, 3, 4, 5, 6, 7}) check(solo(T), T, "a " + std::to_string(T) + "-row solo verify");
  // The batched forms: slot 0 beside one slot (2 x 8 = 16 rows) and three (4 x 4 = 16 rows).
  for (const int slots : {2, 4}) {
    const int T = kRows / slots;
    reset(slots);
    std::vector<int> reqs;
    std::vector<std::vector<int64_t>> feds;
    for (int q = 0; q < slots; ++q) { reqs.push_back(q); feds.emplace_back(block.begin(), block.begin() + T); }
    const std::vector<QwenModel::Outputs> outs = m.session_verify_batch(reqs, feds);
    QwenModel::Outputs slot0 = outs[0];
    // The batched walk's layer capture covers the whole batch: slot 0's rows are the first T.
    check(slot0, T, "slot 0 of a " + std::to_string(slots) + " x " + std::to_string(T) + "-row batched verify");
  }
  require(failures.empty(), "rows: " + std::to_string(failures.size()) + " forms differ from the eight-row solo verify");
  std::printf("[ OK ] qwen_decode_rows_invariance\n");
  return 0;
}

int main(int argc, char** argv) {
  if (const char* tile = std::getenv("DGPP_TEST_QWEN_PREFILL_TILE"); tile && tile[0] == '1')
    dgpp::QwenModel::set_default_prefill_dense_tile(true);

  std::string fixture, checkpoint, ids_text;
  bool fp8_head = false, prefill_head = false, dense_fp8 = false, rows_invariance = false;
  int steps = 4;
  int layers_extra = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--fixture" && i + 1 < argc) fixture = argv[++i];
    else if (a == "--fp8-head")
      fp8_head = true;
    else if (a == "--draft-screen" && i + 1 < argc)  // the draft head's screen with this candidate cap
      dgpp::draft_head_configure(dgpp::DraftHeadSettings{std::stoi(argv[++i])});
    else if (a == "--prefill-head")
      prefill_head = true;
    else if (a == "--rows-invariance")
      rows_invariance = true;
    else if (a == "--dense-fp8")  // the whole run under engine.dense_weights = fp8
      dense_fp8 = true;
    else if (a == "--checkpoint-dir" && i + 1 < argc) checkpoint = argv[++i];
    else if (a == "--ids" && i + 1 < argc) ids_text = argv[++i];
    else if (a == "--steps" && i + 1 < argc) steps = std::stoi(argv[++i]);
    else if (a == "--layers" && i + 1 < argc) layers_extra = std::stoi(argv[++i]);
  }
  try {
    if (dense_fp8) dgpp::QwenLayerStream::set_dense_weights_fp8(true);
    if (!fixture.empty() && rows_invariance) return run_rows_invariance(fixture);
    if (!fixture.empty()) return prefill_head ? run_prefill_head(fixture) : run_fixture(fixture, fp8_head);
    if (!checkpoint.empty()) {
      std::vector<int64_t> ids;
      std::stringstream ss(ids_text);
      std::string item;
      while (std::getline(ss, item, ',')) if (!item.empty()) ids.push_back(std::stoll(item));
      if (ids.empty()) throw std::runtime_error("--ids is required with --checkpoint-dir");
      if (layers_extra >= 0) return run_layers(checkpoint, ids, layers_extra);
      return run_checkpoint(checkpoint, ids, steps);
    }
    std::fprintf(
        stderr,
        "usage: --fixture DIR [--fp8-head | --prefill-head] | --checkpoint-dir DIR --ids 1,2,... [--steps N]\n");
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[FAIL] %s\n", e.what());
    return 1;
  }
}
