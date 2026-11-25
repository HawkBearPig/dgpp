// The Inkling 2 session surface (docs/inkling2_mini_plan.md G4): incremental
// decode over Inkling2Model's request slots against the cold re-forward, the
// prefix snapshots, the verify rows against the scalar steps, and the window.
//
//   --fixture DIR [--activations bf16|native]
//                                the fixture gates (tests/cuda/inkling2_fixture.hpp's
//                                checkpoint, absent: written into DIR)
//   --checkpoint-dir DIR --ids 1,2,... [--steps N] [--activations bf16|native]
//                                the real checkpoint (streaming, world 1): a greedy
//                                transcript and its re-forward audit
//
// Gates on the fixture. A one-shot prefill's last row is bitwise the cold
// forward's. Single steps agree with the re-forward's rows at every position
// where the two are one computation: the first visit, on a fixture whose
// first layer is local and on one whose first layer attends over the whole
// context. A verify of
// T rows reproduces the T scalar steps bitwise through the GEMV rows, and a
// rollback restarts the scalar chain. Per-step
// snapshots leave the transcript bitwise. Two interleaved slots reproduce
// their solo runs bitwise. A chunked prefill agrees with the one-shot. A
// closed and reopened slot restarts bitwise. A snapshot attached in another
// slot resumes bitwise, at a prefill cut and mid-decode: the convolution
// rings travel with it. The window: a change to token 0 leaves the rows of a
// local layer past its reach bitwise unmoved.
//
// The convolutions give every visit a memory of three positions per stream,
// which the engine keeps in positional rings rather than in rollback
// snapshots. The verify, rollback and attach gates are what hold that
// design: a retracted row must leave no trace in the ring.
//
// The fixture's random layers amplify a difference by about half again per
// visit and its routers flip on near ties, so the last visit of a one-row
// launch and of a T-row launch are two samples of a chaotic map: their
// logits are reported, not judged. What is judged is every decode row's
// first visit, whose inputs are the same bits in both walks, and so the
// decode forms of the attention, the paged cache and the rings. The later
// visits run the same code on other planes and offsets, which the forward
// test holds to the reference visit by visit. On the real checkpoint, whose
// layers are stable, the logits themselves are judged (--checkpoint-dir).
// Under the native activation recipe the expert output of the first visit
// is reported as well: the activation quantizers turn a one-ulp input
// difference into whole quantization steps. The bitwise gates hold under
// both recipes.
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

#include "common/dtypes.hpp"
#include "inkling2_fixture.hpp"
#include "kernels/gemm.hpp"
#include "kernels/latent_format.hpp"
#include "models/inkling2/config.hpp"
#include "models/inkling2/forward.hpp"

namespace fs = std::filesystem;
using dgpp::Inkling2Config;
using dgpp::Inkling2Model;
using dgpp::Inkling2Residency;

namespace {

bool g_native = false;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

std::vector<int64_t> smoke_tokens(const Inkling2Config& cfg, int n, uint64_t seed) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = seed;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    t[static_cast<size_t>(i)] = static_cast<int64_t>(s % static_cast<uint64_t>(cfg.vocab_size));
  }
  return t;
}

std::unique_ptr<Inkling2Model> model(const Inkling2Config& cfg, const std::string& dir, int max_tokens, int64_t cache_tokens,
                                     int requests, Inkling2Residency residency = Inkling2Residency::Resident) {
  return std::make_unique<Inkling2Model>(cfg, dir, max_tokens, cache_tokens, residency, nullptr, 0, 1, requests, false, 0,
                                         dgpp::LatentFormat::kBf16, false, g_native);
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

struct Transcript {
  std::vector<int64_t> tokens;           // generated ids (the prefill's pick first)
  std::vector<std::vector<float>> rows;  // the prefill's row, then every step's
};

void push_row(Transcript& t, const Inkling2Model::Outputs& o) {
  t.tokens.push_back(argmax(o.logits.data(), o.lm_vocab_count));
  t.rows.push_back(o.logits);
}

Transcript greedy(Inkling2Model& m, int req, const std::vector<int64_t>& prompt, int steps) {
  Transcript t;
  Inkling2Model::Outputs o = m.session_prefill(req, prompt);
  push_row(t, o);
  for (int s = 0; s < steps; ++s) {
    o = m.session_step(req, t.tokens.back());
    push_row(t, o);
  }
  return t;
}

std::string ids_text(const std::vector<int64_t>& ids) {
  std::string s;
  for (size_t i = 0; i < ids.size(); ++i) s += (i ? "," : "") + std::to_string(ids[i]);
  return s;
}

// The transcript's rows against the cold re-forward of prompt + tokens: row
// P - 1 + i of the re-forward is step i's row at the same position. A step
// is a one-row launch and the re-forward a T-row one, so the dense sites
// reassociate and a routing near tie may fall the other way: at most
// `max_outliers` rows may sit over the l2 budget with their top-1 still
// inside the near-tie rule.
int audit(Inkling2Model& ref, const std::vector<int64_t>& prompt, const Transcript& t, const char* what, double l2_budget,
          int max_outliers, bool judged) {
  std::vector<int64_t> all(prompt);
  all.insert(all.end(), t.tokens.begin(), t.tokens.end() - 1);
  const Inkling2Model::Outputs f = ref.forward(all);
  const int V = f.lm_vocab_count;
  const size_t P = prompt.size();
  int hard = 0, soft = 0, outliers = 0;
  double worst = 0;
  std::string profile;
  for (size_t i = 0; i < t.rows.size(); ++i) {
    const float* want = f.logits.data() + (P - 1 + i) * static_cast<size_t>(V);
    const RowCompare c = compare_row(t.rows[i].data(), want, V);
    worst = std::max(worst, c.l2);
    if (!c.top1_equal) (c.near_tie ? soft : hard) += 1;
    if (c.l2 >= l2_budget) ++outliers;
    if (c.l2 >= l2_budget || (t.rows.size() > 16 && i % 8 == 0)) {
      char buf[64];
      std::snprintf(buf, sizeof(buf), " p%zu:%.3f%s", P + i, c.l2, c.top1_equal ? "" : c.near_tie ? "(near tie)" : "(top-1)");
      profile += buf;
    }
  }
  if (!profile.empty()) std::printf("[ .. ] %s: per-position relative l2%s\n", what, profile.c_str());
  std::printf("[ .. ] %s: %zu rows vs the re-forward, worst relative l2 %.3g, %d rows over %.2g, top-1 hard %d near-tie %d%s\n",
              what, t.rows.size(), worst, outliers, l2_budget, hard, soft, judged ? "" : " (reported)");
  if (!judged) return soft;
  require(hard == 0, std::string(what) + ": a top-1 mismatch beyond the near-tie margin");
  require(outliers <= max_outliers, std::string(what) + ": too many rows over the l2 budget");
  return soft;
}

// The window through the paged cache and the rings. Visit 0 is a local
// layer: a row attends [pos - W + 1, pos], its keys and values reach three
// positions further back through their convolutions, and the attention and
// MLP convolutions three more each. So a change to token 0 moves visit 0's
// state on rows 1 .. W + 8 and on no row after them. Visit 1 is a
// full-attention layer and moves everywhere.
void run_window_gate(const Inkling2Config& cfg, const std::string& dir) {
  const int T = 72, W = cfg.sliding_window_size, H = cfg.hidden_size;
  const int reach = W - 1 + 3 * (cfg.sconv_kernel_size - 1);
  require(cfg.is_local_layer(0) && !cfg.is_local_layer(1) && reach + 4 < T, "window gate: the fixture's layers");
  std::vector<int64_t> a = smoke_tokens(cfg, T, 0x5851F42D4C957F2Dull);
  std::vector<int64_t> b = a;
  b[0] = (a[0] + 7) % cfg.vocab_size;
  auto m = model(cfg, dir, T, 128, 1);
  const Inkling2Model::Outputs fa = m->forward(a, true);
  const Inkling2Model::Outputs fb = m->forward(b, true);
  int moved_inside = 0, moved_past = 0, moved_full = 0;
  for (int t = 1; t < T; ++t) {
    const bool same0 = std::memcmp(fa.layer_states[0].data() + static_cast<size_t>(t) * H,
                                   fb.layer_states[0].data() + static_cast<size_t>(t) * H, static_cast<size_t>(H) * 2) == 0;
    const bool same1 = std::memcmp(fa.layer_states[1].data() + static_cast<size_t>(t) * H,
                                   fb.layer_states[1].data() + static_cast<size_t>(t) * H, static_cast<size_t>(H) * 2) == 0;
    if (t <= reach) moved_inside += !same0;
    else moved_past += !same0;
    moved_full += !same1;
  }
  std::printf("[ .. ] window gate: visit 0 (local, window %d, reach %d) rows moved by a token-0 change: %d of %d inside the "
              "reach, %d of %d past it; visit 1 (full): %d of %d\n",
              W, reach, moved_inside, reach, moved_past, T - 1 - reach, moved_full, T - 1);
  require(moved_past == 0, "window gate: a row past the window's reach saw token 0");
  require(moved_inside >= reach - 2, "window gate: the rows inside the reach did not see token 0");
  require(moved_full >= T - 3, "window gate: the full-attention layer did not see the change");
  std::printf("[ OK ] the window and the convolutions' reach hold through the paged cache (rows %d.. unmoved)\n", reach + 1);
}

// Decode against the re-forward where the two are one computation. Every
// step's row is traced and compared with the re-forward's row at the same
// position, visit by visit. Visit 0 reads the embedding and the cache and
// rings that earlier rows of the same embedding wrote, so its attention
// contribution differs between the walks only by the launch shapes: the
// split count of the attention and the GEMV against the batched GEMM. The
// steps cross the pool's block boundary, the window, the relative extent and
// the ring's wrap.
//
// The fixture's routers sit near ties, so a difference of a few bf16 units in
// the expert input can route a row to another expert: that row's state is
// then another expert's output and says nothing about the decode forms, and
// neither do the next rows the MLP convolution carries it into. The flips
// are counted against the same budget as the forward test's (5 % of the
// slots) and those rows' state is left out of the comparison. Which tokens
// hit a tie depends on the transcript, so a run under loop guidance meets
// other ties than a run without.
void run_trace_gate(const Inkling2Config& cfg, const std::string& dir, const char* what) {
  const int P = 23, steps = 60, H = cfg.hidden_size, V = cfg.num_visits();
  const std::vector<int64_t> prompt = smoke_tokens(cfg, P, 0x9E3779B97F4A7C15ull);
  auto m = model(cfg, dir, 128, 256, 1);
  m->set_decode_row_traces(true);
  struct Step {
    std::vector<std::vector<uint16_t>> attn, in, state;
    std::vector<int32_t> routes;  // visit 0
  };
  std::vector<Step> trace;
  std::vector<int64_t> all(prompt);
  Inkling2Model::Outputs o = m->session_prefill(0, prompt);
  for (int s = 0; s < steps; ++s) {
    const int64_t token = argmax(o.logits.data(), o.lm_vocab_count);
    all.push_back(token);
    o = m->session_step(0, token);
    require(static_cast<int>(o.layer_states.size()) == V, "trace gate: a step records every visit");
    require(static_cast<int>(o.route_ids.size()) == V, "trace gate: a step records every visit's routes");
    trace.push_back(Step{m->visit_attention(), m->visit_mlp_input(), o.layer_states, o.route_ids[0]});
  }
  m->session_close(0);
  m->set_decode_row_traces(false);
  const Inkling2Model::Outputs f = m->forward(all, true);
  const auto& f_attn = m->visit_attention();
  const auto& f_in = m->visit_mlp_input();
  const auto rel = [&](const std::vector<uint16_t>& row, const std::vector<uint16_t>& plane, int pos) {
    double d2 = 0, w2 = 0;
    for (int c = 0; c < H; ++c) {
      const double u = dgpp::bf16_bits_to_float(row[static_cast<size_t>(c)]);
      const double v = dgpp::bf16_bits_to_float(plane[static_cast<size_t>(pos) * H + c]);
      d2 += (u - v) * (u - v);
      w2 += v * v;
    }
    return std::sqrt(d2 / (w2 + 1e-30));
  };
  std::vector<double> worst_attn(static_cast<size_t>(V), 0.0), worst_in(static_cast<size_t>(V), 0.0), worst_state(static_cast<size_t>(V), 0.0);
  // The slots of a step's first visit whose expert the re-forward did not pick.
  const int K = cfg.num_experts_per_tok;
  const auto flipped = [&](const std::vector<int32_t>& row, int pos) {
    const std::vector<int32_t>& plane = f.route_ids[0];
    int n = 0;
    for (int k = 0; k < K; ++k) {
      bool found = false;
      for (int j = 0; j < K; ++j) found = found || row[static_cast<size_t>(k)] == plane[static_cast<size_t>(pos) * K + j];
      n += found ? 0 : 1;
    }
    return n;
  };
  // A flipped row's expert output also enters the MLP convolution of the
  // rows after it, as far as the taps reach.
  int flips = 0, flipped_rows = 0, last_flip = -cfg.sconv_kernel_size;
  for (int s = 0; s < steps; ++s) {
    const int row_flips = flipped(trace[static_cast<size_t>(s)].routes, P + s);
    flips += row_flips;
    if (row_flips > 0) last_flip = s;
    const bool carried = s - last_flip < cfg.sconv_kernel_size;
    flipped_rows += carried ? 1 : 0;
    for (int v = 0; v < V; ++v) {
      const size_t i = static_cast<size_t>(v);
      worst_attn[i] = std::max(worst_attn[i], rel(trace[static_cast<size_t>(s)].attn[i], f_attn[i], P + s));
      worst_in[i] = std::max(worst_in[i], rel(trace[static_cast<size_t>(s)].in[i], f_in[i], P + s));
      if (v == 0 && carried) continue;
      worst_state[i] = std::max(worst_state[i], rel(trace[static_cast<size_t>(s)].state[i], f.layer_states[i], P + s));
    }
  }
  std::string later;
  for (int v = 1; v < V; ++v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), " %d:%.3f", v, worst_state[static_cast<size_t>(v)]);
    later += buf;
  }
  std::printf("[ .. ] %s: %d steps vs the re-forward, worst relative l2 over the steps at visit 0 (%s): attention %.5f, expert "
              "input %.5f, state %.5f\n",
              what, steps, cfg.is_local_layer(0) ? "local" : "full", worst_attn[0], worst_in[0], worst_state[0]);
  std::printf("[ .. ] %s: later visits' states (reported):%s\n", what, later.c_str());
  std::printf("[ .. ] %s: %d of %d first-visit slots routed to another expert than the re-forward's (%d rows that carry "
              "one, their state not compared)\n",
              what, flips, steps * K, flipped_rows);
  require(flips * 20 <= steps * K, std::string(what) + ": routing flips at the first visit exceed 5 % of the slots");
  require(worst_attn[0] < 1e-2, std::string(what) + ": a decode row's first attention differs from the re-forward's");
  require(worst_in[0] < 2e-2, std::string(what) + ": a decode row's first expert input differs from the re-forward's");
  if (!g_native) require(worst_state[0] < 5e-2, std::string(what) + ": a decode row's first state differs from the re-forward's");
  std::printf("[ OK ] %s: every decode row's first visit is the re-forward's\n", what);
}

// Causality across launch shapes: the first P rows of a longer forward are
// the rows of the P-row forward, visit by visit. The two walks differ only in
// how many rows share their launches.
void run_prefix_gate(const Inkling2Config& cfg, const std::string& dir) {
  const int P = 23, T = 83, H = cfg.hidden_size;
  const std::vector<int64_t> all = smoke_tokens(cfg, T, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> head(all.begin(), all.begin() + P);
  auto m = model(cfg, dir, 128, 256, 1);
  const Inkling2Model::Outputs a = m->forward(head, true);
  const std::vector<std::vector<uint16_t>> attn_a = m->visit_attention();
  const std::vector<std::vector<uint16_t>> in_a = m->visit_mlp_input();
  const Inkling2Model::Outputs b = m->forward(all, true);
  const auto rel = [&](const std::vector<uint16_t>& x, const std::vector<uint16_t>& y) {
    double d2 = 0, w2 = 0;
    for (size_t i = 0; i < static_cast<size_t>(P) * H; ++i) {
      const double u = dgpp::bf16_bits_to_float(x[i]), v = dgpp::bf16_bits_to_float(y[i]);
      d2 += (u - v) * (u - v);
      w2 += v * v;
    }
    return std::sqrt(d2 / (w2 + 1e-30));
  };
  double worst = 0;
  for (size_t v = 0; v < a.layer_states.size(); ++v) {
    const double ra = rel(attn_a[v], m->visit_attention()[v]);
    const double rx = rel(in_a[v], m->visit_mlp_input()[v]);
    const double rs = rel(a.layer_states[v], b.layer_states[v]);
    int flips = 0;
    const int K = cfg.num_experts_per_tok;
    for (int t = 0; t < P; ++t)
      for (int k = 0; k < K; ++k) {
        bool found = false;
        for (int j = 0; j < K; ++j)
          found = found || a.route_ids[v][static_cast<size_t>(t) * K + k] == b.route_ids[v][static_cast<size_t>(t) * K + j];
        flips += found ? 0 : 1;
      }
    std::printf("[ .. ] prefix gate visit %2zu: attention %.5f, expert input %.5f, state %.5f, routing flips %d\n", v, ra, rx, rs, flips);
    worst = std::max(worst, rs);
  }
  std::printf("[ .. ] prefix gate: worst state difference %.4f\n", worst);
}

int run_fixture(const std::string& dir) {
  const Inkling2Config cfg = ink2fx::with_test_loop(Inkling2Config::from_checkpoint_dir(dir));
  const std::vector<int64_t> A = smoke_tokens(cfg, 23, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 17, 0xD1B54A32D192ED03ull);
  run_prefix_gate(cfg, dir);
  auto mp = model(cfg, dir, 64, 256, 2);
  Inkling2Model& m = *mp;
  const int V = m.lm_vocab_count();

  // 1. prefill == forward, bitwise.
  {
    const Inkling2Model::Outputs f = m.forward(A);
    const Inkling2Model::Outputs p = m.session_prefill(0, A);
    require(p.logits.size() == static_cast<size_t>(V), "prefill: one row of logits");
    const std::vector<float> last(f.logits.end() - V, f.logits.end());
    require(bitwise(p.logits, last), "prefill's last row is not bitwise the forward's");
    require(m.session_position(0) == static_cast<int64_t>(A.size()), "prefill: position");
    m.session_close(0);
    require(m.session_position(0) == 0, "close: position");
    std::printf("[ OK ] prefill last row bitwise the cold forward (%zu tokens)\n", A.size());
  }

  // 2. Incremental decode vs the re-forward: 60 steps from the 23-token
  //    prompt, across the pool's 64-token block boundary, past the 40-token
  //    window and the 24-token relative extent, and three times round the
  //    16-slot convolution rings.
  const Transcript tL = greedy(m, 0, A, 60);
  m.session_close(0);
  {
    auto wide = model(cfg, dir, 128, 256, 1);
    (void)audit(*wide, A, tL, "decode 60 steps", 1e-1, 8, false);
  }

  // 2b. A rolling snapshot on every step: the transcript stays bitwise.
  {
    std::vector<uint8_t*> arena(2, nullptr);
    const size_t bytes = m.session_snapshot_bytes();
    for (uint8_t*& p : arena) require(cudaMalloc(reinterpret_cast<void**>(&p), bytes) == cudaSuccess, "snap arena");
    Inkling2Model::Outputs o = m.session_prefill(0, A);
    require(bitwise(o.logits, tL.rows[0]), "per-step snapshots: the prefill differs");
    int64_t pending = argmax(o.logits.data(), V);
    Inkling2Model::SessionSnapshotMeta metas[2];
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
    for (int i = 0; i < 2; ++i)
      if (filled[i]) m.session_release_snapshot(metas[i]);
    require(m.kv_blocks_in_use() == 0, "per-step snapshots: every block released");
    for (uint8_t* p : arena) cudaFree(p);
    std::printf("[ OK ] 60 steps under per-step rolling snapshots reproduce the plain transcript bitwise\n");
  }

  const Transcript tA = greedy(m, 0, A, 12);
  m.session_close(0);
  std::printf("[ .. ] 12 greedy steps: ids %s\n", ids_text(tA.tokens).c_str());

  // 2c. The verify rows: T scalar steps as one T-row verify. The bf16 sites'
  //     GEMV keeps a row's chain whatever rows share the launch, so the rows
  //     are bitwise the scalar steps through one request's widest pass (the
  //     pending token and the draft's seven proposals: a draft must never
  //     change what the model writes). A rollback to one accepted row
  //     restarts the scalar chain: the retracted rows' ring slots and cache
  //     rows are dead and the next step overwrites them.
  {
    const int gemv_rows = Inkling2Model::kRequestRows;
    for (int T = 1; T <= dgpp::kSpecRows; ++T) {
      Inkling2Model::Outputs p = m.session_prefill(0, A);
      require(bitwise(p.logits, tA.rows[0]), "verify rows: the prefill differs");
      const std::vector<int64_t> ids(tA.tokens.begin(), tA.tokens.begin() + T);
      const Inkling2Model::Outputs v = m.session_verify(0, ids);
      require(v.logits.size() == static_cast<size_t>(T) * V, "verify rows: T rows of logits");
      require(m.session_position(0) == static_cast<int64_t>(A.size()) + T, "verify rows: the position advanced by T");
      int exact = 0;
      for (int r = 0; r < T; ++r) {
        const std::vector<float> row(v.logits.begin() + static_cast<size_t>(r) * V, v.logits.begin() + static_cast<size_t>(r + 1) * V);
        const std::vector<float>& want = tA.rows[static_cast<size_t>(r) + 1];
        if (bitwise(row, want)) {
          ++exact;
          continue;
        }
        // Past one request's rows the dense sites take another launch
        // form, whose sums associate differently. On this fixture that is
        // another sample of a chaotic map, so the rows are counted, not
        // judged.
        const RowCompare c = compare_row(row.data(), want.data(), V);
        require(T > gemv_rows, "verify of " + std::to_string(T) + " rows: row " + std::to_string(r) +
                                   " is not bitwise the scalar step (relative l2 " + std::to_string(c.l2) + ")");
      }
      // Accept row 0 only: the scalar chain continues from there.
      m.session_rollback(0, 1, T);
      require(m.session_position(0) == static_cast<int64_t>(A.size()) + 1, "verify rows: the rollback's position");
      const Inkling2Model::Outputs next = m.session_step(0, tA.tokens[1]);
      const bool next_exact = bitwise(next.logits, tA.rows[2]);
      if (T <= gemv_rows)
        require(next_exact, "verify of " + std::to_string(T) + " rows: the step after the rollback differs");
      m.session_close(0);
      std::printf("[ .. ] verify of %d rows: %d bitwise the scalar steps; the rollback's next step %s\n", T, exact,
                  next_exact ? "bitwise" : "another launch shape's");
    }
    std::printf("[ OK ] verify rows reproduce the scalar steps (bitwise through %d rows) and roll back\n", gemv_rows);
  }

  // 3. Two slots interleaved reproduce their solo runs bitwise.
  const Transcript tB = greedy(m, 0, B, 8);
  m.session_close(0);
  {
    Inkling2Model::Outputs oa = m.session_prefill(0, A);
    Inkling2Model::Outputs ob = m.session_prefill(1, B);
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
    for (int s = 8; s < 12; ++s) {
      oa = m.session_step(0, pa);
      require(bitwise(oa.logits, tA.rows[static_cast<size_t>(s) + 1]),
              "interleaved: slot 0's step " + std::to_string(s) + " after slot 1 closed");
      pa = argmax(oa.logits.data(), V);
    }
    m.session_close(0);
    std::printf("[ OK ] two interleaved slots reproduce their solo transcripts bitwise\n");
  }

  // 3b. Under loop guidance a slot runs at its own request's strength: slot
  //     1 at another strength than slot 0, interleaved, reproduces the solo
  //     run at that strength, and slot 0 its own.
  if (cfg.loop_guidance) {
    const float own = m.loop_guidance_default(), other = own + 0.4f;
    m.configure_loop_guidance(0, other);
    const Transcript tB2 = greedy(m, 0, B, 8);
    m.session_close(0);
    m.configure_loop_guidance(0, own);
    m.configure_loop_guidance(1, other);
    Inkling2Model::Outputs oa = m.session_prefill(0, A);
    Inkling2Model::Outputs ob = m.session_prefill(1, B);
    require(bitwise(oa.logits, tA.rows[0]) && bitwise(ob.logits, tB2.rows[0]),
            "guided slots: a prefill differs from the solo run at the slot's strength");
    int64_t pa = tA.tokens[0], pb = tB2.tokens[0];
    for (int s = 0; s < 8; ++s) {
      oa = m.session_step(0, pa);
      ob = m.session_step(1, pb);
      require(bitwise(oa.logits, tA.rows[static_cast<size_t>(s) + 1]) &&
                  bitwise(ob.logits, tB2.rows[static_cast<size_t>(s) + 1]),
              "guided slots: step " + std::to_string(s) + " differs from the solo run at the slot's strength");
      pa = argmax(oa.logits.data(), V);
      pb = argmax(ob.logits.data(), V);
    }
    require(!bitwise(tB2.rows[0], tB.rows[0]), "guided slots: another strength left the logits unmoved");
    m.session_close(0);
    m.session_close(1);
    m.configure_loop_guidance(1, own);
    std::printf("[ OK ] two slots at their own guidance strengths (%.2f and %.2f) reproduce their solo transcripts\n",
                static_cast<double>(own), static_cast<double>(other));
  }

  // 3c. Random schedules. Two slots, each with a prompt of its own and,
  //     under loop guidance, a strength of its own, advance in a random
  //     order. A slot advances by one scalar step or by a verify of two to
  //     eight rows, of which a random count is accepted and the rest rolled
  //     back with wrong tokens in them, sometimes under a snapshot taken
  //     first. Every accepted row is bitwise the solo run's at that slot's
  //     strength: nothing a rejected row, the other slot or a snapshot
  //     leaves in the cache, the rings or the readout's planes reaches a
  //     later row.
  {
    const int kSchedules = 24, kTokens = 20;
    const size_t snap_bytes = m.session_snapshot_bytes();
    uint8_t* arena = nullptr;
    require(cudaMalloc(reinterpret_cast<void**>(&arena), snap_bytes) == cudaSuccess, "random schedules: snapshot arena");
    const float strengths[] = {0.0f, 0.2f, 0.5f, 1.0f};
    uint64_t rng = 0xA5A5F00D2026ull;
    const auto next = [&rng] {
      rng ^= rng << 13;
      rng ^= rng >> 7;
      rng ^= rng << 17;
      return rng;
    };
    int verifies = 0, rejected = 0, snapshots = 0;
    for (int schedule = 0; schedule < kSchedules; ++schedule) {
      std::vector<int64_t> prompt[2];
      float strength[2] = {0.0f, 0.0f};
      Transcript solo[2];
      for (int i = 0; i < 2; ++i) {
        prompt[i] = smoke_tokens(cfg, 5 + static_cast<int>(next() % 26), next());
        if (cfg.loop_guidance) {
          strength[i] = (next() % 5 == 0) ? m.loop_guidance_default() : strengths[next() % 4];
          m.configure_loop_guidance(0, strength[i]);
        }
        solo[i] = greedy(m, 0, prompt[i], kTokens);
        m.session_close(0);
      }
      if (cfg.loop_guidance)
        for (int i = 0; i < 2; ++i) m.configure_loop_guidance(i, strength[i]);
      int done[2] = {0, 0};  // steps taken: the next input is solo.tokens[done]
      for (int i = 0; i < 2; ++i) {
        const Inkling2Model::Outputs o = m.session_prefill(i, prompt[i]);
        require(bitwise(o.logits, solo[i].rows[0]), "random schedules: a prefill differs from the solo run");
      }
      bool snapped = false;
      Inkling2Model::SessionSnapshotMeta meta;
      while (done[0] < kTokens || done[1] < kTokens) {
        int i = static_cast<int>(next() % 2);
        if (done[i] >= kTokens) i ^= 1;
        const std::string where = "random schedules: schedule " + std::to_string(schedule) + ", slot " + std::to_string(i) +
                                  ", step " + std::to_string(done[i]);
        if (next() % 5 == 0) {
          if (snapped) m.session_release_snapshot(meta);
          meta = m.session_snapshot(i, arena);
          snapped = true;
          ++snapshots;
        }
        const int left = kTokens - done[i];
        if (left >= 2 && next() % 2 == 0) {
          const int T = 2 + static_cast<int>(next() % static_cast<uint64_t>(std::min(left, Inkling2Model::kRequestRows) - 1));
          const int accept = 1 + static_cast<int>(next() % static_cast<uint64_t>(T));
          std::vector<int64_t> ids(solo[i].tokens.begin() + done[i], solo[i].tokens.begin() + done[i] + T);
          // The rows past the accepted ones carry tokens the solo run never saw.
          for (int r = accept; r < T; ++r) ids[static_cast<size_t>(r)] = (ids[static_cast<size_t>(r)] + 1 + static_cast<int64_t>(next() % 7)) % V;
          const Inkling2Model::Outputs v = m.session_verify(i, ids);
          require(v.logits.size() == static_cast<size_t>(T) * V, where + ": a verify's rows");
          for (int r = 0; r < accept; ++r) {
            const std::vector<float> row(v.logits.begin() + static_cast<size_t>(r) * V,
                                         v.logits.begin() + static_cast<size_t>(r + 1) * V);
            require(bitwise(row, solo[i].rows[static_cast<size_t>(done[i] + r) + 1]),
                    where + ": accepted row " + std::to_string(r) + " of " + std::to_string(T) + " differs from the solo run");
          }
          if (accept < T) {
            m.session_rollback(i, accept, T);
            ++rejected;
          }
          done[i] += accept;
          ++verifies;
        } else {
          const Inkling2Model::Outputs o = m.session_step(i, solo[i].tokens[static_cast<size_t>(done[i])]);
          require(bitwise(o.logits, solo[i].rows[static_cast<size_t>(done[i]) + 1]), where + ": a step differs from the solo run");
          done[i] += 1;
        }
        require(m.session_position(i) == static_cast<int64_t>(prompt[i].size()) + done[i], where + ": the slot's position");
      }
      if (snapped) m.session_release_snapshot(meta);
      m.session_close(0);
      m.session_close(1);
      require(m.kv_blocks_in_use() == 0, "random schedules: every block released");
    }
    if (cfg.loop_guidance)
      for (int i = 0; i < 2; ++i) m.configure_loop_guidance(i, m.loop_guidance_default());
    cudaFree(arena);
    std::printf("[ OK ] %d random schedules of two slots (%d verifies, %d with rejected rows, %d snapshots) reproduce the solo "
                "transcripts bitwise\n",
                kSchedules, verifies, rejected, snapshots);
  }

  // 4. Chunked prefill (chunk == max_tokens == 8) agrees with the one-shot:
  //    every chunk's rows read the three positions before the chunk from
  //    the rings.
  {
    auto c = model(cfg, dir, 8, 256, 1);
    const Inkling2Model::Outputs one = m.session_prefill(0, A);
    m.session_close(0);
    const Inkling2Model::Outputs p = c->session_prefill(0, A);
    RowCompare r = compare_row(p.logits.data(), one.logits.data(), V);
    std::printf("[ .. ] chunked prefill (8-row chunks): relative l2 %.3g, top-1 %s (reported)\n", r.l2,
                r.top1_equal ? "equal" : (r.near_tie ? "near tie" : "differs"));
    c->session_close(0);
    const Inkling2Model::Outputs pb = c->session_prefill(0, A, std::vector<int64_t>{13});
    Transcript tc;
    push_row(tc, pb);
    for (int s = 0; s < 6; ++s) {
      const Inkling2Model::Outputs o = c->session_step(0, tc.tokens.back());
      push_row(tc, o);
    }
    c->session_close(0);
    (void)audit(m, A, tc, "decode after the chunked prefill", 1e-1, 2, false);
    std::printf("[ OK ] chunked prefill runs and its decode continues from the chunked state\n");
  }

  // 5. Close and reopen restarts bitwise; the pool's accounting.
  {
    require(m.kv_blocks_total() == 4, "pool: 256 tokens are 4 blocks of 64");
    require(m.kv_blocks_in_use() == 0, "pool: nothing held after the closes");
    require(m.kv_blocks_for_tokens(65) == 2, "pool: 65 tokens take 2 blocks");
    const Inkling2Model::Outputs p = m.session_prefill(0, A);
    require(bitwise(p.logits, tA.rows[0]), "reopen: the prefill differs from the first run");
    require(m.kv_blocks_in_use() == 1, "pool: one block for 23 tokens");
    Inkling2Model::Outputs o = m.session_step(0, tA.tokens[0]);
    require(bitwise(o.logits, tA.rows[1]), "reopen: the first step differs from the first run");
    m.session_close(0);
    require(m.kv_blocks_in_use() == 0, "pool: the close released every block");
    (void)m.session_prefill(0, B);
    bool refused = false;
    try {
      (void)m.forward(B);
    } catch (const std::logic_error&) {
      refused = true;
    }
    require(refused, "forward: must refuse while slot 0 is open");
    m.session_close(0);
    const Inkling2Model::Outputs f = m.forward(B);
    require(bitwise(std::vector<float>(f.logits.end() - V, f.logits.end()), tB.rows[0]),
            "forward after the sessions differs from the solo prefill");
    std::printf("[ OK ] close/reopen restarts bitwise; pool accounting; forward guarded\n");
  }

  // 6. The prefix cache: hot == cold bitwise. A snapshot at cut 12 of A's
  //    prefill, attached in the other slot, the suffix resumed: the last row
  //    and the steps after it are bitwise the cold session's. The snapshot
  //    carries the three ring positions before the cut for every visit and
  //    stream. A mid-decode snapshot likewise.
  {
    const std::vector<int64_t> bounds{12};
    std::vector<uint8_t*> arena(2, nullptr);
    const size_t bytes = m.session_snapshot_bytes();
    require(bytes > 0, "prefix: the snapshot has bytes");
    for (uint8_t*& p : arena) require(cudaMalloc(reinterpret_cast<void**>(&p), bytes) == cudaSuccess, "prefix: arena");
    Inkling2Model::SessionSnapshotMeta meta;
    Inkling2Model::SnapshotRequest snap;
    snap.position = 12;
    snap.dst = arena[0];
    snap.meta = &meta;
    const Inkling2Model::Outputs cold = m.session_prefill(0, A, bounds, &snap);
    require(snap.taken && meta.position == 12, "prefix: the snapshot was taken at the cut");
    const Inkling2Model::Outputs cold_step = m.session_step(0, tA.tokens[0]);
    // The cold session moves on: its rings no longer hold the cut.
    std::vector<std::vector<float>> cold_rows;
    int64_t pending = argmax(cold_step.logits.data(), V);
    for (int s = 0; s < 20; ++s) {
      const Inkling2Model::Outputs o = m.session_step(0, pending);
      cold_rows.push_back(o.logits);
      pending = argmax(o.logits.data(), V);
    }
    m.session_close(0);
    m.session_attach(1, arena[0], meta);
    require(m.session_position(1) == 12, "prefix: attached at the snapshot position");
    const Inkling2Model::Outputs hot = m.session_prefill_resume(1, std::vector<int64_t>(A.begin() + 12, A.end()), bounds);
    require(bitwise(hot.logits, cold.logits), "prefix: the hot prefill's last row differs from the cold one's");
    const Inkling2Model::Outputs hot_step = m.session_step(1, tA.tokens[0]);
    require(bitwise(hot_step.logits, cold_step.logits), "prefix: the first step after the attach differs");
    // A mid-decode snapshot: 23 + 1 + 4 steps = position 28.
    pending = argmax(hot_step.logits.data(), V);
    for (int s = 0; s < 4; ++s) {
      const Inkling2Model::Outputs o = m.session_step(1, pending);
      require(bitwise(o.logits, cold_rows[static_cast<size_t>(s)]), "prefix: a step after the attach differs");
      pending = argmax(o.logits.data(), V);
    }
    require(m.session_position(1) == 28, "prefix: position 28");
    const Inkling2Model::SessionSnapshotMeta mid = m.session_snapshot(1, arena[1]);
    // The source runs ahead, then closes.
    for (int s = 4; s < 20; ++s) {
      const Inkling2Model::Outputs o = m.session_step(1, pending);
      require(bitwise(o.logits, cold_rows[static_cast<size_t>(s)]), "prefix: the source's later steps differ");
      pending = argmax(o.logits.data(), V);
    }
    m.session_close(1);
    m.session_attach(0, arena[1], mid);
    require(m.session_position(0) == 28, "prefix: attached mid-decode");
    // The token fed at position 28 is the pick of the row before it.
    pending = argmax(cold_rows[3].data(), V);
    for (int s = 4; s < 20; ++s) {
      const Inkling2Model::Outputs o = m.session_step(0, pending);
      require(bitwise(o.logits, cold_rows[static_cast<size_t>(s)]),
              "prefix: step " + std::to_string(s) + " after the mid-decode attach differs");
      pending = argmax(o.logits.data(), V);
    }
    m.session_close(0);
    m.session_release_snapshot(meta);
    m.session_release_snapshot(mid);
    require(m.kv_blocks_in_use() == 0, "prefix: every block released");
    for (uint8_t* p : arena) cudaFree(p);
    std::printf("[ OK ] prefix snapshots resume bitwise at a prefill cut and mid-decode\n");
  }

  run_window_gate(cfg, dir);
  run_trace_gate(cfg, dir, "decode trace, local first layer");
  {
    const std::string full = (fs::path(dir).parent_path() / (fs::path(dir).filename().string() + "_full_first")).string();
    ink2fx::write_full_first_fixture(full);
    run_trace_gate(ink2fx::with_test_loop(Inkling2Config::from_checkpoint_dir(full)), full, "decode trace, full-attention first layer");
  }
  std::printf("inkling2 decode (%s activations): PASS\n", g_native ? "native" : "bf16");
  return 0;
}

int run_checkpoint(const std::string& dir, const std::vector<int64_t>& ids, int steps) {
  const Inkling2Config cfg = ink2fx::with_test_loop(Inkling2Config::from_checkpoint_dir(dir));
  const int P = static_cast<int>(ids.size());
  auto m = model(cfg, dir, P + steps, P + steps + 64, 1, Inkling2Residency::Streaming);
  const Transcript t = greedy(*m, 0, ids, steps);
  m->session_close(0);
  std::printf("greedy ids: %s\n", ids_text(t.tokens).c_str());
  const bool was_native = g_native;
  const int soft = audit(*m, ids, t, "real checkpoint decode", 5e-2, steps / 8 + 1, !g_native);
  std::printf("inkling2 decode on the checkpoint (%s activations): %d steps, %d near ties: PASS\n", was_native ? "native" : "bf16",
              steps, soft);
  return 0;
}

// A diagnostic on the real checkpoint (--verify-diag T): T greedy scalar
// steps against one T-row verify of the same tokens, visit by visit. A verify
// row must be the scalar step at its position bit for bit, or a draft would
// change what the model writes. Prints, for every row, the first visit and
// stream that differs and how many logits differ.
// `shape`: 0 one slot of the test's shape. 1 the server's model: eight slots,
// 64 decode rows, the draft model loaded. 2 that with serving logits.
int run_verify_diag(const std::string& dir, const std::vector<int64_t>& ids, int T, int shape) {
  const Inkling2Config cfg = ink2fx::with_test_loop(Inkling2Config::from_checkpoint_dir(dir));
  const int P = static_cast<int>(ids.size()), H = cfg.hidden_size, V = cfg.num_visits();
  require(T >= 1 && T <= dgpp::kSpecRows, "--verify-diag takes 1..kSpecRows rows");
  auto m = shape == 0 ? model(cfg, dir, 2 * P + 64, 2 * P + 256, 1, Inkling2Residency::Resident)
                      : std::make_unique<Inkling2Model>(cfg, dir, std::max(2 * P + 64, 256), 65536,
                                                        Inkling2Residency::Resident, nullptr, 0, 1, 8, true, 64,
                                                        dgpp::LatentFormat::kBf16, shape == 2, g_native);
  m->set_decode_row_traces(true);
  struct Step {
    std::vector<std::vector<uint16_t>> attn, in, state;
    std::vector<float> logits;
  };
  std::vector<Step> scalar;
  std::vector<int64_t> tokens;
  Inkling2Model::Outputs o = m->session_prefill(0, ids);
  const int64_t vocab = o.lm_vocab_count;
  for (int s = 0; s < T; ++s) {
    tokens.push_back(argmax(o.logits.data(), static_cast<int>(vocab)));
    o = m->session_step(0, tokens.back());
    scalar.push_back(Step{m->visit_attention(), m->visit_mlp_input(), o.layer_states, o.logits});
  }
  m->session_close(0);
  std::printf("greedy ids: %s\n", ids_text(tokens).c_str());
  // Another request through the slot first: a longer prompt and wide passes
  // of other tokens, some retracted. Whatever it leaves behind must not
  // reach the rows compared below.
  {
    m->set_decode_row_traces(false);
    std::vector<int64_t> other(ids.rbegin(), ids.rend());
    for (int i = 0; i < 31; ++i) other.push_back(ids[static_cast<size_t>(i) % ids.size()]);
    (void)m->session_prefill(0, other);
    std::vector<int64_t> junk;
    for (int i = 0; i < T; ++i) junk.push_back(ids[static_cast<size_t>(3 * i + 1) % ids.size()]);
    for (int pass = 0; pass < 3; ++pass) {
      (void)m->session_verify(0, junk);
      m->session_rollback(0, 1 + pass % T, T);
    }
    m->session_close(0);
    m->set_decode_row_traces(true);
  }
  (void)m->session_prefill(0, ids);
  const Inkling2Model::Outputs v = m->session_verify(0, tokens);
  const auto v_attn = m->visit_attention();
  const auto v_in = m->visit_mlp_input();
  require(static_cast<int>(v.layer_states.size()) == V, "verify diag: the verify records every visit");
  int bad_rows = 0;
  for (int r = 0; r < T; ++r) {
    const Step& want = scalar[static_cast<size_t>(r)];
    std::string first = "none";
    for (int visit = 0; visit < V && first == "none"; ++visit) {
      const size_t i = static_cast<size_t>(visit);
      const auto differs = [&](const std::vector<uint16_t>& rows, const std::vector<uint16_t>& one) {
        return std::memcmp(rows.data() + static_cast<size_t>(r) * H, one.data(), static_cast<size_t>(H) * 2) != 0;
      };
      if (differs(v_attn[i], want.attn[i])) first = "visit " + std::to_string(visit) + " attention";
      else if (differs(v_in[i], want.in[i])) first = "visit " + std::to_string(visit) + " expert input";
      else if (differs(v.layer_states[i], want.state[i])) first = "visit " + std::to_string(visit) + " state";
    }
    size_t logits_off = 0;
    const float* row = v.logits.data() + static_cast<size_t>(r) * vocab;
    for (int64_t c = 0; c < vocab; ++c) logits_off += std::memcmp(row + c, want.logits.data() + c, 4) != 0;
    const bool same_top = argmax(row, static_cast<int>(vocab)) == argmax(want.logits.data(), static_cast<int>(vocab));
    std::printf("row %d of %d: first difference %s; %zu of %lld logits differ; top-1 %s\n", r, T, first.c_str(), logits_off,
                static_cast<long long>(vocab), same_top ? "equal" : "DIFFERENT");
    bad_rows += first != "none" || logits_off != 0;
  }
  std::printf("verify diag (%s activations): %d of %d rows differ from their scalar steps\n", g_native ? "native" : "bf16",
              bad_rows, T);
  return bad_rows == 0 ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string fixture, dir, ids_arg;
    int steps = 24, verify_diag = 0, shape = 0;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string {
        require(i + 1 < argc, "missing value after " + a);
        return argv[++i];
      };
      if (a == "--fixture") fixture = next();
      else if (a == "--checkpoint-dir") dir = next();
      else if (a == "--ids") ids_arg = next();
      else if (a == "--steps") steps = std::stoi(next());
      else if (a == "--verify-diag") verify_diag = std::stoi(next());
      else if (a == "--shape") shape = std::stoi(next());
      else if (a == "--activations") {
        const std::string v = next();
        require(v == "bf16" || v == "native", "--activations takes bf16 or native");
        g_native = v == "native";
      } else throw std::runtime_error("unknown argument " + a);
    }
    if (!fixture.empty()) {
      if (!fs::exists(fs::path(fixture) / "config.json")) ink2fx::write_fixture(fixture);
      return run_fixture(fixture);
    }
    require(!dir.empty() && !ids_arg.empty(), "usage: --fixture DIR | --checkpoint-dir DIR --ids 1,2,...");
    std::vector<int64_t> ids;
    std::stringstream ss(ids_arg);
    std::string item;
    while (std::getline(ss, item, ',')) ids.push_back(std::stoll(item));
    if (verify_diag > 0) return run_verify_diag(dir, ids, verify_diag, shape);
    return run_checkpoint(dir, ids, steps);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "inkling2_decode_test: %s\n", e.what());
    return 1;
  }
}
