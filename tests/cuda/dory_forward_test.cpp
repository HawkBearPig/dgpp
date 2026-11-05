// The Dory1.0 walk over a synthetic mini-checkpoint (tests/cuda/dory_fixture.hpp):
// the loop's re-injection, the per-loop K/V planes, the sliding window, the head's
// logit scale, and the invariants a session path must keep.
//
// The reference's full transcription is the plan's §1 — these rungs check the
// walk's own contract (determinism, causality, the window biting, the K/V
// policies) before they check numbers: a fixture this small cannot stand in for
// the release's transcript, but every one of these fails loudly on a wrong
// plane, a stale cache row or a missing scale.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "dory_fixture.hpp"
#include "dory_host_reference.hpp"
#include "models/dory/forward.hpp"

namespace {

namespace fs = std::filesystem;
using namespace dgpp;

void require(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(std::string("dory forward: ") + what);
}

struct Fixture {
  DoryConfig cfg;
  std::string dir;
  std::vector<int64_t> ids;
};

Fixture write_fixture(const char* name, int loops = 2, int window = 8, int tokens = 12) {
  Fixture fx;
  fx.cfg = doryfx::tiny_config(loops, window);
  fx.dir = (fs::current_path() / name).string();
  doryfx::write_fixture(fx.cfg, fx.dir);
  fx.ids.resize(tokens);
  for (int i = 0; i < tokens; ++i) fx.ids[i] = 7 + 37 * i;  // inside the tiny vocab
  return fx;
}

bool all_finite(const std::vector<float>& v) {
  for (float x : v)
    if (!std::isfinite(x)) return false;
  return true;
}

bool all_finite_bits(const std::vector<uint16_t>& v) {
  for (uint16_t b : v) {
    const float x = dgpp::bf16_bits_to_float(b);
    if (!std::isfinite(x)) return false;
  }
  return true;
}

// Relative L2 and the largest element-wise relative difference, with an abs
// floor (the logits of a fresh head cross zero).
struct Diff {
  double l2_rel = 0;
  double max_rel = 0;
  size_t n = 0;
};

Diff diff_f32(const std::vector<float>& a, const std::vector<float>& b) {
  Diff d;
  d.n = a.size();
  if (a.size() != b.size()) {
    d.l2_rel = d.max_rel = 1e30;
    return d;
  }
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double x = a[i], y = b[i];
    num += (x - y) * (x - y);
    den += y * y;
    const double scale = std::max(std::abs(x), std::abs(y));
    const double rel = std::abs(x - y) / std::max(scale, 1e-6);
    d.max_rel = std::max(d.max_rel, rel);
  }
  d.l2_rel = den > 0 ? std::sqrt(num / den) : (num > 0 ? 1e30 : 0.0);
  return d;
}

size_t count_differing(const std::vector<float>& a, const std::vector<float>& b) {
  size_t n = 0;
  for (size_t i = 0; i < a.size(); ++i) n += a[i] != b[i];
  return n;
}

}  // namespace

DGPP_TEST(dory_forward_fixture_smoke) {
  const Fixture fx = write_fixture("dory_forward_fixture");
  DoryModel model(fx.cfg, fx.dir, 32, 256, DoryResidency::Resident, 1, 0, false);
  const auto v = fx.cfg.vocab_size;
  const int T = static_cast<int>(fx.ids.size());

  const auto a = model.forward(fx.ids);
  require(static_cast<int>(a.logits.size()) == T * v, "one logits row per token");
  require(a.lm_vocab_count == v, "the head is unsharded");
  require(all_finite(a.logits), "logits finite");
  require(a.final_hidden_bits.size() == static_cast<size_t>(T) * fx.cfg.hidden_size,
          "the hidden state's rows");
  require(all_finite_bits(a.final_hidden_bits), "hidden finite");
  {
    double lo = 1e30, hi = -1e30;
    for (float x : a.logits) {
      lo = std::min<double>(lo, x);
      hi = std::max<double>(hi, x);
    }
    std::printf("[ .. ] dory fixture: T=%d vocab=%d logits in [%.4g, %.4g]\n", T, v, lo, hi);
    require(hi > lo, "the head's logits are not constant");
  }

  // The same walk twice: the pool's rows are released and rebuilt, so any
  // state the walk leaves behind shows up here.
  const auto b = model.forward(fx.ids);
  {
    double m = 0;
    for (size_t i = 0; i < a.logits.size(); ++i) m = std::max(m, std::abs((double)a.logits[i] - b.logits[i]));
    std::printf("[ .. ] dory fixture: forward 1 vs 2: %zu/%zu differ, max abs %.3g\n",
                count_differing(a.logits, b.logits), a.logits.size(), m);
  }
  {
    DoryModel other(fx.cfg, fx.dir, 32, 256, DoryResidency::Resident, 1, 0, false);
    const auto c = other.forward(fx.ids);
    double m = 0;
    for (size_t i = 0; i < a.logits.size(); ++i) m = std::max(m, std::abs((double)a.logits[i] - c.logits[i]));
    std::printf("[ .. ] dory fixture: instance A vs B, one call each: %zu/%zu differ, max abs %.3g\n",
                count_differing(a.logits, c.logits), a.logits.size(), m);
  }
  model.pool().reset_all(model.stream());
  ::cudaStreamSynchronize(model.stream());
  const auto d2 = model.forward(fx.ids);
  {
    double m = 0;
    for (size_t i = 0; i < a.logits.size(); ++i) m = std::max(m, std::abs((double)a.logits[i] - d2.logits[i]));
    std::printf("[ .. ] dory fixture: forward 1 vs reset+forward: %zu/%zu differ, max abs %.3g\n",
                count_differing(a.logits, d2.logits), a.logits.size(), m);
  }
  {
    double m = 0;
    for (size_t i = 0; i < b.logits.size(); ++i) m = std::max(m, std::abs((double)b.logits[i] - d2.logits[i]));
    std::printf("[ .. ] dory fixture: forward 2 vs reset+forward: %zu/%zu differ, max abs %.3g\n",
                count_differing(b.logits, d2.logits), b.logits.size(), m);
  }
  {
    DoryModel probe(fx.cfg, fx.dir, 32, 256, DoryResidency::Resident, 1, 0, false);
    const auto s1 = probe.forward(fx.ids, true);
    const auto s2 = probe.forward(fx.ids, true);
    for (size_t l = 0; l < s1.layer_states.size(); ++l) {
      size_t c = 0;
      const auto& x = s1.layer_states[l];
      const auto& y = s2.layer_states[l];
      for (size_t i = 0; i < x.size(); ++i) c += x[i] != y[i];
      std::printf("[ .. ] dory fixture: application %zu: %zu/%zu differ\n", l, c, x.size());
      if (c == 0) continue;
      break;
    }
  }
  const auto d3 = model.forward(fx.ids);
  {
    double m = 0;
    for (size_t i = 0; i < a.logits.size(); ++i) m = std::max(m, std::abs((double)a.logits[i] - d3.logits[i]));
    std::printf("[ .. ] dory fixture: forward 1 vs 3: %zu/%zu differ, max abs %.3g\n",
                count_differing(a.logits, d3.logits), a.logits.size(), m);
  }
  {
    double m = 0;
    for (size_t i = 0; i < b.logits.size(); ++i) m = std::max(m, std::abs((double)b.logits[i] - d3.logits[i]));
    std::printf("[ .. ] dory fixture: forward 2 vs 3: %zu/%zu differ, max abs %.3g\n",
                count_differing(b.logits, d3.logits), b.logits.size(), m);
  }
  require(count_differing(a.logits, b.logits) == 0, "a repeated forward is bitwise identical");
  require(a.final_hidden_bits == b.final_hidden_bits, "the hidden rows repeat bitwise");

  // Causality: a shorter prompt's rows must not move. The window, the
  // positions and the per-loop planes all enter row t alone, so a mismatch
  // means one of them leaks across rows.
  DoryModel shorter(fx.cfg, fx.dir, 32, 256, DoryResidency::Resident, 1, 0, false);
  std::vector<int64_t> cut(fx.ids.begin(), fx.ids.end() - 1);
  const auto c = shorter.forward(cut);
  const int Tc = static_cast<int>(cut.size());
  {
    const double d = diff_f32(std::vector<float>(c.logits.begin(), c.logits.end()),
                             std::vector<float>(a.logits.begin(), a.logits.begin() + Tc * v))
                         .l2_rel;
    std::printf("[ .. ] dory fixture: T-1 prefix vs the T-token walk, rows 0..%d relative L2 %.3g\n",
                Tc - 1, d);
    require(count_differing(std::vector<float>(c.logits.begin(), c.logits.end()),
                            std::vector<float>(a.logits.begin(), a.logits.begin() + Tc * v)) == 0,
            "the prefix's rows are bitwise");
  }
}

DGPP_TEST(dory_forward_layer_states) {
  const Fixture fx = write_fixture("dory_forward_states_fixture");
  DoryModel model(fx.cfg, fx.dir, 32, 256, DoryResidency::Resident, 1, 0, false);
  const auto out = model.forward(fx.ids, /*capture_layers=*/true);
  std::printf("[ .. ] dory fixture: %zu layer states captured\n", out.layer_states.size());
  // One state per sublayer application: the input block, the recurrent block
  // once per loop, then the output block (plan §1.5).
  const size_t apps = static_cast<size_t>(fx.cfg.n_input_layers) +
                      static_cast<size_t>(fx.cfg.n_recurrent_layers) *
                          static_cast<size_t>(fx.cfg.n_recurrent_loops) +
                      static_cast<size_t>(fx.cfg.n_output_layers);
  require(out.layer_states.size() == apps, "one state per sublayer application");
  const size_t row = static_cast<size_t>(fx.ids.size()) * fx.cfg.hidden_size;
  size_t changed = 0;
  for (const auto& s : out.layer_states) {
    require(s.size() == row, "a layer state per row");
    require(all_finite_bits(s), "layer states finite");
    const auto& first = out.layer_states.front();
    for (size_t i = 0; i < s.size(); ++i) changed += s[i] != first[i];
  }
  // Every layer's output differs from the embedding's raw rows: a layer that
  // silently passed its input through would collapse the walk.
  require(changed > 0, "the layers move the state");
  std::printf("[ .. ] dory fixture: %zu layer states, %zu of %zu values differ from layer 0's\n",
              out.layer_states.size(), changed, row * out.layer_states.size());
}

DGPP_TEST(dory_forward_sliding_window_bites) {
  const Fixture narrow = write_fixture("dory_forward_window_fixture", 2, /*window=*/8);
  const Fixture wide = write_fixture("dory_forward_nowindow_fixture", 2, /*window=*/4096);
  DoryModel a(narrow.cfg, narrow.dir, 32, 256, DoryResidency::Resident, 1, 0, false);
  DoryModel b(wide.cfg, wide.dir, 32, 256, DoryResidency::Resident, 1, 0, false);
  const auto la = a.forward(narrow.ids);
  const auto lb = b.forward(wide.ids);
  require(la.logits.size() == lb.logits.size(), "the same shapes");
  const size_t differ = count_differing(la.logits, lb.logits);
  std::printf("[ .. ] dory fixture: window 8 over 12 tokens vs window 4096: %zu/%zu logits differ\n",
              differ, la.logits.size());
  require(differ > la.logits.size() / 10, "the sliding window masks the old rows");
}

DGPP_TEST(dory_forward_kv_policies) {
  const Fixture fx = write_fixture("dory_forward_policy_fixture");
  DoryModel per_loop(fx.cfg, fx.dir, 32, 256, DoryResidency::Resident, 1, 0, false,
                     DoryKvPolicy::PerLoop);
  DoryModel last_loop(fx.cfg, fx.dir, 32, 256, DoryResidency::Resident, 1, 0, false,
                      DoryKvPolicy::LastLoop);
  // One plane per recurrent attention layer per loop, against one shared.
  const int loops = fx.cfg.n_recurrent_loops;
  int recurrent_attn = 0, other_attn = 0;
  for (int l = 0; l < fx.cfg.num_hidden_layers; ++l) {
    if (!fx.cfg.is_attention(l)) continue;
    (fx.cfg.in_recurrent_block(l) ? recurrent_attn : other_attn)++;
  }
  const int want_per_loop = recurrent_attn * loops + other_attn;
  std::printf("[ .. ] dory fixture: planes per_loop=%d last_loop=%d\n",
              per_loop.kv_pool().shape().planes(), last_loop.kv_pool().shape().planes());
  require(per_loop.kv_pool().shape().planes() == want_per_loop, "per_loop planes");
  require(last_loop.kv_pool().shape().planes() == recurrent_attn + other_attn, "last_loop planes");
  std::printf("[ .. ] dory fixture: planes per_loop=%d last_loop=%d (recurrent attention %d, loops %d)\n",
              per_loop.kv_pool().shape().planes(), last_loop.kv_pool().shape().planes(),
              recurrent_attn, loops);

  // One cold walk reads only rows it wrote in that walk, so both policies
  // compute the same logits; a plane mix-up (loop i reading loop j's rows)
  // would show up here.
  const auto a = per_loop.forward(fx.ids);
  const auto b = last_loop.forward(fx.ids);
  require(count_differing(a.logits, b.logits) == 0, "one walk is policy-invariant");
}

DGPP_TEST(dory_forward_loop_override) {
  const Fixture fx = write_fixture("dory_forward_loops_fixture");
  DoryModel two(fx.cfg, fx.dir, 32, 256, DoryResidency::Resident, 1, 0, false);
  const auto a = two.forward(fx.ids);
  DoryModel::set_recurrent_loops(1);
  struct Clear {
    ~Clear() { DoryModel::clear_recurrent_loops(); }
  } clear;
  DoryModel one(fx.cfg, fx.dir, 32, 256, DoryResidency::Resident, 1, 0, false);
  require(one.config().n_recurrent_loops == 1, "the override reaches the config");
  require(one.config().n_recurrent_loops != two.config().n_recurrent_loops, "the two models differ");
  const auto b = one.forward(fx.ids);
  const size_t differ = count_differing(a.logits, b.logits);
  std::printf("[ .. ] dory fixture: loops 2 vs 1: %zu/%zu logits differ\n", differ, a.logits.size());
  require(differ > 0, "the loop count changes the result");
  DoryModel::clear_recurrent_loops();
  require(DoryModel::recurrent_loops_override() == 0, "the override clears");
}

DGPP_TEST(dory_forward_decode_matches_prefill) {
  const Fixture fx = write_fixture("dory_forward_decode_fixture");
  DoryModel model(fx.cfg, fx.dir, 32, 256, DoryResidency::Resident, 1, 1, false);
  const auto v = fx.cfg.vocab_size;
  const int64_t next = 313;
  const auto prefill = model.session_prefill(0, fx.ids);
  const auto step = model.session_step(0, next);
  model.session_close(0);
  std::vector<int64_t> whole = fx.ids;
  whole.push_back(next);
  DoryModel fresh(fx.cfg, fx.dir, 32, 256, DoryResidency::Resident, 1, 1, false);
  const auto both = fresh.forward(whole);
  require(step.logits.size() == static_cast<size_t>(v), "one decode row");
  require(all_finite(step.logits), "decode logits finite");
  require(prefill.logits.size() >= static_cast<size_t>(v), "the prefill's last row");
  // The decode row's logits are the whole-prompt walk's last row: the paged
  // read and the fresh prefill are the same arithmetic (different split-KV
  // orders, so not bitwise).
  const std::vector<float> want(both.logits.end() - v, both.logits.end());
  const Diff d = diff_f32(step.logits, want);
  // Near-zero logits make a per-element relative bound meaningless, so the
  // worst absolute deviation is measured against the row's peak magnitude.
  double peak = 0, worst = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    peak = std::max(peak, static_cast<double>(std::fabs(want[i])));
    worst = std::max(worst, static_cast<double>(std::fabs(step.logits[i] - want[i])));
  }
  std::printf(
      "[ .. ] dory fixture: decode row vs the fresh prefill's last row: rel L2 %.3g, worst abs "
      "%.3g of peak %.3g\n",
      d.l2_rel, worst, peak);
  require(d.l2_rel < 2e-2, "the decode row matches the prefill's arithmetic");
  require(worst < 5e-2 * peak, "no element diverges from the row's scale");
  const size_t pick_a = std::max_element(step.logits.begin(), step.logits.end()) - step.logits.begin();
  const size_t pick_b = std::max_element(want.begin(), want.end()) - want.begin();
  require(pick_a == pick_b, "the greedy pick agrees");
}

// The fixture ladder's rungs (docs/adding_a_model.md §6), run over a
// directory written by `--write-fixture`: the tiny checkpoint in the
// release's own layout, generated from the binding table itself.
//   --smoke DIR          the CUDA model over the fixture: finite, non-constant,
//                        bitwise application by application
//   --reference DIR      every logits row against dory_host_reference.hpp's
//                        CPU fp64 walk (the generate rung), and the session
//                        decode row against that walk's last row (the decode
//                        rung)
int run_smoke(const std::string& dir) {
  const DoryConfig cfg = DoryConfig::from_json_file((fs::path(dir) / "config.json").string());
  std::vector<int64_t> ids(12);
  for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int64_t>(7 + 37 * i) % cfg.vocab_size;
  DoryModel model(cfg, dir, 32, 256, DoryResidency::Resident, 1, 0, false);
  const auto a = model.forward(ids);
  require(all_finite(a.logits), "smoke: finite logits");
  require(a.logits.size() == ids.size() * static_cast<size_t>(cfg.vocab_size), "smoke: one row per token");
  double lo = 1e30, hi = -1e30, top = -1e30;
  size_t pick = 0;
  for (size_t i = 0; i < a.logits.size(); ++i) {
    lo = std::min(lo, static_cast<double>(a.logits[i]));
    hi = std::max(hi, static_cast<double>(a.logits[i]));
  }
  for (int64_t v = 0; v < cfg.vocab_size; ++v) {
    const float x = a.logits[a.logits.size() - static_cast<size_t>(cfg.vocab_size) + v];
    if (x > top) {
      top = x;
      pick = static_cast<size_t>(v);
    }
  }
  require(hi > lo, "smoke: the head is not constant");
  const auto b = model.forward(ids);
  require(a.logits == b.logits, "smoke: a repeated forward is bitwise identical");
  std::printf("[ .. ] smoke: %zu tokens, %d layers, top-1 of the last row %zu (%.4g); deterministic\n",
              ids.size(), cfg.num_hidden_layers, pick, top);
  std::printf("[ OK ] dory_forward_smoke\n");
  return 0;
}

int run_reference(const std::string& dir) {
  const DoryConfig cfg = DoryConfig::from_json_file((fs::path(dir) / "config.json").string());
  const int T = 12;
  std::vector<int64_t> ids(T);
  for (int i = 0; i < T; ++i) ids[i] = static_cast<int64_t>(7 + 37 * i) % cfg.vocab_size;
  const auto want = doryfx::host_forward(cfg, dir, ids, false);

  // The generate rung: the cold walk's every logits row.
  {
    DoryModel model(cfg, dir, 32, 256, DoryResidency::Resident, 1, 0, false);
    const auto got = model.forward(ids);
    const Diff d = diff_f32(got.logits, want);
    // Near-zero logits make a per-element relative bound meaningless (the
    // decode rung's own finding), so the element bound is absolute against
    // the walk's peak magnitude.
    double peak = 0, worst = 0;
    for (size_t i = 0; i < want.size(); ++i) {
      peak = std::max(peak, std::fabs(static_cast<double>(want[i])));
      worst = std::max(worst, std::fabs(static_cast<double>(got.logits[i] - want[i])));
    }
    std::printf("[ .. ] generate: %zu rows against the host reference: rel L2 %.3g, worst abs "
                "%.3g of peak %.3g\n",
                ids.size(), d.l2_rel, worst, peak);
    require(d.l2_rel < 2e-2, "the walk matches the host reference (rel L2)");
    require(worst < 5e-2 * peak, "no logits element diverges beyond the reference's scale");
  }

  // The decode rung: a T-1 prefill and one session step, against the same
  // host walk over all T tokens. The paged read's split-KV order differs
  // from the fresh prefill's, so the bound is the decode test's own.
  {
    std::vector<int64_t> pre(ids.begin(), ids.end() - 1);
    std::vector<int64_t> whole = pre;
    whole.push_back(ids[T - 1]);
    const auto want_last = doryfx::host_forward(cfg, dir, whole, false);
    DoryModel model(cfg, dir, 32, 256, DoryResidency::Resident, 1, 1, false);
    model.session_prefill(0, pre);
    const auto step = model.session_step(0, ids[T - 1]);
    model.session_close(0);
    const std::vector<float> tail(want_last.end() - cfg.vocab_size, want_last.end());
    const Diff d = diff_f32(step.logits, tail);
    double peak = 0, worst = 0;
    for (size_t i = 0; i < tail.size(); ++i) {
      peak = std::max(peak, std::fabs(static_cast<double>(tail[i])));
      worst = std::max(worst, std::fabs(static_cast<double>(step.logits[i] - tail[i])));
    }
    std::printf("[ .. ] decode: the session row against the host reference: rel L2 %.3g, worst abs "
                "%.3g of peak %.3g\n",
                d.l2_rel, worst, peak);
    require(d.l2_rel < 2e-2, "the decode row matches the host reference (rel L2)");
    require(worst < 5e-2 * peak, "no element diverges from the row's scale");
  }
  std::printf("[ OK ] dory_forward_reference\n");
  return 0;
}

// `--write-fixture DIR` writes the tiny checkpoint for manual runs; with no
// argument the DGPP_TEST filter (DGPP_TEST_FILTER) selects the cases.
int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--write-fixture") {
      const DoryConfig cfg = doryfx::tiny_config();
      doryfx::write_fixture(cfg, argv[2]);
      std::printf("[ OK ] wrote the fixture to %s\n", argv[2]);
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--smoke") return run_smoke(argv[2]);
    if (argc == 3 && std::string(argv[1]) == "--reference") return run_reference(argv[2]);
    if (argc != 1) {
      std::fprintf(stderr, "usage: dory_forward_test [--write-fixture DIR | --smoke DIR | --reference DIR]\n");
      return 2;
    }
    return dgpp::test::run_all();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[FAIL] %s\n", e.what());
    return 1;
  }
}

