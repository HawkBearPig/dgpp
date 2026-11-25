// The Inkling 2 world-1 forward (docs/inkling2_mini_plan.md G3).
//
// Modes:
//   --write-fixture DIR          the tiny synthetic checkpoint (tests/cuda/inkling2_fixture.hpp)
//   --smoke DIR                  Inkling2Model over the fixture: finite outputs, a bitwise repeat
//   --checkpoint-dir DIR
//   --dump-file FILE             Inkling2Model against tools/inkling2_reference_dump.py's numpy
//                                reference: the partial state after every visit, every visit's
//                                attention contribution, the final hidden state, the per-token
//                                top-k logits, the routing decisions
//   --activations bf16|native    the expert input recipe (both modes)
//   --relaxed                    the end-to-end budget
//   --report                     another implementation's dump (transformers): print the
//                                comparison, judge nothing, accept a dump of the other recipe
//   --loop-guidance [--loop-strength W]
//                                loop guidance (engine.loop_guidance*), in every mode
//   --passes N                   a test's walk of N passes over the core (no engine key sets
//                                it): the one-pass walk is what the first pass's readout is
//                                held to
//   --loop-check DIR             loop guidance against the checkpoint's own walk over the
//                                fixture: the gates are listed at loop_check()
//
// The ctest chain: fixture -> smoke, and fixture -> generate (python, end to
// end) -> test_e2e (--relaxed, writes the engine's states) -> generate_teacher
// (python, --teacher: every reference visit starts from the engine's own
// state) -> test (strict). The chain runs once per activation recipe.
//
// Budgets: 16 / 128 bf16 ulps soft / hard with a 2 % RMS cancellation floor,
// relative l2 1 %, 0.5 % hard elements, top-1 exact outside near ties and
// routing flips under 5 % of the slots on the teacher-forced dump. The
// end-to-end run is the relaxed yardstick (l2 3 %, 2 % hard) with the visits
// from the first routing flip on reported, not judged. Under the native
// recipe the teacher also supplies every visit's expert input, and the
// end-to-end run judges the first visit up to its expert input only.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "inkling2_fixture.hpp"
#include "inkling2_reference_compare.hpp"
#include "kernels/inkling2.hpp"
#include "loaders/minijson.hpp"
#include "models/inkling2/config.hpp"
#include "models/inkling2/forward.hpp"

namespace fs = std::filesystem;
using dgpp::bf16_bits_to_float;
using dgpp::Inkling2Config;
using dgpp::Inkling2Model;

namespace {

using ink2ref::Dump;
using ink2ref::require;

std::vector<int64_t> smoke_tokens(const Inkling2Config& cfg, int n) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = 0x9E3779B97F4A7C15ull;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    t[static_cast<size_t>(i)] = static_cast<int64_t>(s % static_cast<uint64_t>(cfg.vocab_size));
  }
  return t;
}

// A run's reading of the loop: the test's pass count, then the deployment's guidance.
struct LoopReading {
  dgpp::Inkling2LoopOptions options;
  int passes = 0;  // 0: the checkpoint's
};

Inkling2Config config_of(const std::string& dir, const LoopReading& loop) {
  Inkling2Config cfg = Inkling2Config::from_checkpoint_dir(dir);
  if (loop.passes > 0) cfg.num_passes = loop.passes;
  cfg.apply_loop_options(loop.options);
  return cfg;
}

int smoke(const std::string& dir, bool native, const LoopReading& loop) {
  const Inkling2Config cfg = config_of(dir, loop);
  const int T = 72;
  const std::vector<int64_t> tokens = smoke_tokens(cfg, T);
  Inkling2Model model(cfg, dir, T, 4096, dgpp::Inkling2Residency::Streaming, nullptr, 0, 1, 1, false, 0,
                      dgpp::LatentFormat::kBf16, false, native);
  const auto a = model.forward(tokens, true);
  const auto b = model.forward(tokens, true);
  require(static_cast<int>(a.layer_states.size()) == cfg.num_visits(), "smoke: one state per visit");
  for (size_t v = 0; v < a.layer_states.size(); ++v) {
    for (uint16_t bits : a.layer_states[v])
      require(std::isfinite(bf16_bits_to_float(bits)), "smoke: non-finite state at visit " + std::to_string(v));
    require(a.layer_states[v] == b.layer_states[v], "smoke: the repeat differs at visit " + std::to_string(v));
  }
  require(a.logits.size() == static_cast<size_t>(T) * cfg.vocab_size, "smoke: logits shape");
  for (float x : a.logits) require(std::isfinite(x) && std::fabs(x) <= cfg.final_logit_softcapping + 1e-3f, "smoke: logit");
  require(a.logits == b.logits, "smoke: the repeat's logits differ");
  require(a.final_hidden_bits == b.final_hidden_bits, "smoke: the repeat's hidden differs");
  std::printf("inkling2 forward smoke (%s activations): %d visits, %d tokens, finite and repeatable\n",
              native ? "native" : "bf16", cfg.num_visits(), T);
  return 0;
}

int compare(const std::string& dir, const std::string& dump_path, bool relaxed, bool native, bool report,
            const LoopReading& loop) {
  const Inkling2Config cfg = config_of(dir, loop);
  const Dump d = Dump::load(dump_path);
  const int T = static_cast<int>(d.token_count);
  std::vector<int64_t> tokens(static_cast<size_t>(T));
  std::memcpy(tokens.data(), d.tensor("tokens").data, static_cast<size_t>(T) * 8);
  Inkling2Model model(cfg, dir, T, 4096, dgpp::Inkling2Residency::Streaming, nullptr, 0, 1, 1, false, 0,
                      dgpp::LatentFormat::kBf16, false, native);
  const auto out = model.forward(tokens, true);
  ink2ref::EngineRun run;
  run.states = out.layer_states;
  run.attention = model.visit_attention();
  run.mlp_input = model.visit_mlp_input();
  run.route_ids = out.route_ids;
  run.final_hidden = out.final_hidden_bits;
  run.logits = out.logits;
  if (const char* path = std::getenv("DGPP_INKLING2_DUMP_ENGINE_STATES")) ink2ref::write_states(path, run, T, cfg.hidden_size);
  return ink2ref::judge(cfg, d, run, relaxed, native, report, "inkling2 forward");
}

// The teacher-forced likelihood of a token file on the real checkpoint (a
// measurement, not a gate): compare two runs position by position.
int likelihood(const std::string& dir, const std::string& ids_path, const std::string& out_path, bool native,
               const LoopReading& loop) {
  const Inkling2Config cfg = config_of(dir, loop);
  const std::vector<int64_t> tokens = ink2ref::read_ids(ids_path);
  const int T = static_cast<int>(tokens.size());
  require(T <= 4096, "--ids: at most 4096 tokens in one forward");
  Inkling2Model model(cfg, dir, T, 4096, dgpp::Inkling2Residency::Streaming, nullptr, 0, 1, 1, false, 0,
                      dgpp::LatentFormat::kBf16, false, native);
  const auto out = model.forward(tokens, false);
  char reading[160];
  std::snprintf(reading, sizeof reading, ", %d passes, loop guidance %s strength %g", cfg.num_passes,
                cfg.loop_guidance ? "on" : "off", static_cast<double>(cfg.loop_guidance_strength));
  const std::string label =
      std::string("inkling2 world 1 (") + (native ? "native" : "bf16") + " activations" + reading + ")";
  return ink2ref::likelihood(cfg, out.logits, tokens, out_path, label.c_str());
}

// Loop guidance against the checkpoint's own walk on the fixture
// (engine.loop_guidance). The unguided walk is the anchor, which the
// reference chain holds visit by visit.
//   - At strength 0 a guided deployment gives the unguided output bit for
//     bit: states, final hidden rows and logits. The second run of the
//     layers after the core touches nothing the model's own walk reads.
//   - At strength -1 the guided logits are the first pass's own readout. A
//     walk of one pass runs the layers after the core on the first pass's
//     states in the model's planes, and the readout does the same in its
//     own: the two logit sets are equal bit for bit. That holds the
//     readout's visits, planes, ring entries and head to a walk the
//     reference chain judges (--passes 1).
//   - A guided logit is the host statement of the model's logit and the
//     readout's, bit for bit.
//   - A request's strength set on its slot gives the output of a deployment
//     whose default is that strength, bit for bit, and the next request's
//     strength replaces it.
constexpr float kUnset = -100.0f;  // no strength set on the slot

int loop_check(const std::string& dir, bool native) {
  const Inkling2Config base = Inkling2Config::from_checkpoint_dir(dir);
  const int T = 48;  // one chunk of the readout's logits
  const std::vector<int64_t> tokens = smoke_tokens(base, T);
  const int V = base.vocab_size;
  // One forward: the deployment's guidance, the test's pass count, and
  // optionally a strength set on the request's slot.
  const auto run = [&](bool guidance, float strength, int passes = 0, float slot = kUnset, float slot_again = kUnset) {
    LoopReading r;
    r.options.guidance = guidance;
    r.options.strength = strength;
    r.passes = passes;
    const Inkling2Config cfg = config_of(dir, r);
    Inkling2Model model(cfg, dir, T, 4096, dgpp::Inkling2Residency::Streaming, nullptr, 0, 1, 1, false, 0,
                        dgpp::LatentFormat::kBf16, false, native);
    if (slot != kUnset) model.configure_loop_guidance(0, slot);
    auto out = model.forward(tokens, true);
    if (slot_again != kUnset) {
      model.configure_loop_guidance(0, slot_again);
      out = model.forward(tokens, true);
    }
    require(static_cast<int>(out.layer_states.size()) == cfg.num_visits(), "loop check: one state per visit");
    require(out.logits.size() == static_cast<size_t>(T) * V, "loop check: logits shape");
    for (float x : out.logits) require(std::isfinite(x), "loop check: a non-finite logit");
    return out;
  };
  const auto same_bits = [](const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
  };
  const auto plain = run(false, 0.0f);

  // Strength 0.
  const auto zero = run(true, 0.0f);
  require(zero.layer_states == plain.layer_states, "loop check: guidance moved a state of the model's walk");
  require(zero.final_hidden_bits == plain.final_hidden_bits, "loop check: guidance moved the final hidden rows");
  require(same_bits(zero.logits, plain.logits), "loop check: guidance at strength 0 moved a logit");

  // The first pass's own readout against the one-pass walk.
  const auto one_pass = run(false, 0.0f, 1);
  const auto readout = run(true, -1.0f);
  require(readout.layer_states == plain.layer_states, "loop check: the readout moved a state of the model's walk");
  require(same_bits(readout.logits, one_pass.logits), "loop check: the first pass's readout is not the one-pass walk's logits");
  require(!same_bits(readout.logits, plain.logits), "loop check: the first pass reads out as the last one");

  // A guided logit from the model's and the readout's.
  const float w = 0.5f;
  const auto guided = run(true, w);
  for (size_t i = 0; i < guided.logits.size(); ++i) {
    const float want = dgpp::ink2_guided_value_host(w, plain.logits[i], readout.logits[i]);
    require(std::memcmp(&guided.logits[i], &want, 4) == 0, "loop check: a guided logit is not its statement");
  }
  // A request's strength on its slot.
  const auto by_slot = run(true, 0.0f, 0, w);
  require(same_bits(by_slot.logits, guided.logits), "loop check: a slot's strength is not a deployment's with that default");
  const auto replaced = run(true, w, 0, 0.9f, 0.0f);
  require(same_bits(replaced.logits, plain.logits), "loop check: the next request's strength did not replace the last one's");

  std::printf("inkling2 loop check (%s activations): strength 0, the first pass's readout, the guided statement, a "
              "request's own strength\n",
              native ? "native" : "bf16");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string fixture, smoke_dir, dir, dump, ids, nll_out, loop_dir;
    bool relaxed = false, native = false, report = false;
    LoopReading loop;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string {
        require(i + 1 < argc, "missing value after " + a);
        return argv[++i];
      };
      if (a == "--write-fixture") fixture = next();
      else if (a == "--smoke") smoke_dir = next();
      else if (a == "--checkpoint-dir") dir = next();
      else if (a == "--dump-file") dump = next();
      else if (a == "--ids") ids = next();
      else if (a == "--nll-out") nll_out = next();
      else if (a == "--loop-check") loop_dir = next();
      else if (a == "--passes") loop.passes = std::stoi(next());
      else if (a == "--loop-guidance") loop.options.guidance = true;
      else if (a == "--loop-strength") loop.options.strength = std::stof(next());
      else if (a == "--relaxed") relaxed = true;
      else if (a == "--report") report = relaxed = true;
      else if (a == "--activations") {
        const std::string v = next();
        require(v == "bf16" || v == "native", "--activations takes bf16 or native");
        native = v == "native";
      } else throw std::runtime_error("unknown argument " + a);
    }
    if (!loop_dir.empty()) return loop_check(loop_dir, native);
    if (!dir.empty() && !ids.empty()) return likelihood(dir, ids, nll_out, native, loop);
    if (!fixture.empty()) {
      ink2fx::write_fixture(fixture);
      return 0;
    }
    if (!smoke_dir.empty()) return smoke(smoke_dir, native, loop);
    require(!dir.empty() && !dump.empty(),
            "usage: --write-fixture DIR | --smoke DIR | --checkpoint-dir DIR --dump-file FILE | --checkpoint-dir DIR --ids FILE "
            "[--nll-out FILE]");
    return compare(dir, dump, relaxed, native, report, loop);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "inkling2_forward_test: %s\n", e.what());
    return 1;
  }
}
