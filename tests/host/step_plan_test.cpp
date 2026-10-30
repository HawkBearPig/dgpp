// The step engine's plan gate (MOONSHOT-1.md section 5, phase 2). The
// DeepSeek-V4.1 plan is built from the config alone and must (1) tile every
// layer's resident span exactly as the loader lays it out, (2) read the
// bytes the hand count from the checkpoint's own safetensors headers
// gives, within one percent, at the two C1 shapes of the phase-0 record,
// and (3) be the same on every rank of the world. Host only, no device.
//
// The hand count (handcount_dsv41.py over DeepSeek-V4.1-Flash-1017's
// headers, world 4, 2026-10-30, in the resident-read form: the draft
// stages pick their own top-k, the mHC coefficients are the bf16 the
// loader rounds them to, the fp8 scale grids are the F32 the fp8 core
// reads, main_proj is held whole on every rank, the KV rows are the
// pool's 528-byte window rows and 288-byte entries) per rank per pass:
//   five verify rows, a five-row draft block, the head read twice:
//     8.886 GB = fixed 3.211 + routed experts 5.633 (main 5.432, draft 0.200) + KV 0.043
//   two verify rows, the same draft block and head:
//     5.653 GB = fixed 3.211 + routed experts 2.424 + KV 0.018
//   one rank's slice of one routed expert: 4,700,160 bytes.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "dsv41_config_json.hpp"
#include "loaders/minijson.hpp"
#include "models/dsv41/config.hpp"
#include "models/dsv41/loader.hpp"
#include "step/dsv41/arena.hpp"
#include "step/dsv41/plan.hpp"

namespace {

int g_failures = 0;
int g_cases = 0;

void check(bool ok, const std::string& what) {
  ++g_cases;
  if (ok) {
    std::printf("[ OK ] %s\n", what.c_str());
  } else {
    ++g_failures;
    std::printf("[FAIL] %s\n", what.c_str());
  }
}

std::string mb(double bytes) {
  char buf[48];
  std::snprintf(buf, sizeof buf, "%.1f MB", bytes / 1e6);
  return buf;
}

// |a - b| within `rel` of b, reported with both figures.
void check_near(double a, double b, double rel, const std::string& what) {
  const bool ok = std::fabs(a - b) <= rel * std::fabs(b);
  char buf[96];
  std::snprintf(buf, sizeof buf, " (plan %s, reference %s, %.2f%%)", mb(a).c_str(), mb(b).c_str(),
                b != 0 ? 100.0 * std::fabs(a - b) / std::fabs(b) : 0.0);
  check(ok, what + buf);
}

}  // namespace

int main() {
  using namespace dgpp;
  using namespace dgpp::step;
  const auto doc = minijson::parse(dsv41_test::config_json());
  const Dsv41TextConfig cfg = Dsv41TextConfig::parse(doc.root);
  constexpr int kWorld = 4;

  // ---- the uniform expectation of distinct experts ---------------------------
  check(std::fabs(dsv41_expected_distinct(384, 5, 6) - 28.9) < 0.05, "five rows of six picks reach 28.9 of 384 experts");
  check(std::fabs(dsv41_expected_distinct(384, 2, 6) - 11.8) < 0.05, "two rows of six picks reach 11.8 of 384 experts");
  check(std::fabs(dsv41_expected_distinct(128, 5, 3) - 14.2) < 0.05, "a five-row draft block of three picks reaches 14.2 of 128");

  // ---- the C1 code shape: five verify rows, a five-row draft block ---------------
  Dsv41StepShape code;
  code.rows = 5;
  code.draft_rows = 5;
  code.head_reads = 2;
  const StepPlan plan = dsv41_step_plan(cfg, 0, kWorld, code);
  std::printf("%s", plan_summary(plan).c_str());
  check(static_cast<int>(plan.layers.size()) == cfg.max_layer(), "the plan walks every main layer and draft stage");

  // (1) structure: every layer's resident ranges tile the loader's span
  {
    const std::vector<std::string> problems = plan_check(plan);
    for (const std::string& p : problems) std::printf("       %s\n", p.c_str());
    check(problems.empty(), "plan_check finds no problem in any layer or the globals");
    bool spans = true;
    for (const LayerPlan& lp : plan.layers)
      spans = spans && lp.span_bytes == Dsv41LayerStream::layer_bytes(cfg, lp.layer, 0, kWorld);
    check(spans, "every layer's span is the loader's layer_bytes");
    check(plan.globals.span_bytes == Dsv41LayerStream::globals_bytes(cfg, 0, kWorld, LoaderHeadSharding::VocabSharded),
          "the globals' span is the loader's globals_bytes");
    // The phases of a main layer, in walk order.
    const LayerPlan& l3 = plan.layers[3];
    std::string names;
    for (const Phase& ph : l3.phases) names += ph.name + " ";
    check(names == "attn_in qkv attend attn_out ffn_in router gate_up down accumulate ",
          "a reusing layer's phases run attn_in, qkv, attend, attn_out, ffn_in, router, gate_up, down, accumulate (" + names + ")");
    const LayerPlan& l2 = plan.layers[2];
    names.clear();
    for (const Phase& ph : l2.phases) names += ph.name + " ";
    check(names.find("select") != std::string::npos, "an index source has a select phase (" + names + ")");
    const LayerPlan& l1 = plan.layers[1];
    check(!l1.phases.empty() && l1.phases[0].name == "engram", "an Engram layer begins with its Engram phase");
    const LayerPlan& d0 = plan.layers[static_cast<size_t>(cfg.num_hidden_layers)];
    check(!d0.phases.empty() && d0.phases[0].name == "draft_in", "the first draft stage begins with its input projection");
    const LayerPlan& dl = plan.layers.back();
    check(!dl.phases.empty() && dl.phases.back().name == "draft_head", "the last draft stage ends with the draft head");
    check(l3.groups == 384 && std::fabs(l3.expected_groups - 28.9) < 0.05, "a main layer expects 28.9 of its 384 experts");
    check(d0.groups == 128 && std::fabs(d0.expected_groups - 14.2) < 0.05, "a draft stage expects 14.2 of its 128 experts");
  }

  // (2) bytes: the hand count from the checkpoint's headers
  {
    const ByteTotals t = step_bytes(plan);
    // The attention consumer's bytes are the KV rows (the sink vectors beside them are 64 floats a layer).
    const double kv = consumer_step_bytes(plan, Consumer::Attention);
    check_near(t.total(), 8.886e9, 0.01, "the five-row step reads the hand count's bytes");
    check_near(t.total() - kv - t.router, 3.211e9, 0.01, "the fixed bytes match the hand count");
    check_near(t.router, 5.633e9, 0.01, "the routed experts' expectation matches the hand count");
    check_near(kv, 0.043e9, 0.03, "the KV rows match the hand count (window rows plus selected entries)");
    // By class.
    check_near(consumer_step_bytes(plan, Consumer::Projection), (1179.0 + 368.4 + 79.0 + 88.4 + 27.6 + 79.0) * 1e6, 0.01,
               "the projections (attention, Engram, DSpark input) match");
    check_near(consumer_step_bytes(plan, Consumer::SharedGateUp) + consumer_step_bytes(plan, Consumer::SharedDown),
               (355.3 + 26.6) * 1e6, 0.01, "the shared experts match");
    check_near(consumer_step_bytes(plan, Consumer::Head), 661.9e6, 0.01, "the head read twice matches");
    check_near(consumer_step_bytes(plan, Consumer::Router), (157.4 + 3.9) * 1e6, 0.01, "the routers match");
    check_near(consumer_step_bytes(plan, Consumer::HyperConn), (78.7 + 5.9) * 1e6, 0.01, "the mHC coefficients match");
    check_near(consumer_step_bytes(plan, Consumer::Indexer), 45.3e6, 0.01, "the indexers match");
    check_near(consumer_step_bytes(plan, Consumer::Compressor), 36.7e6, 0.01, "the compressors match");
    check_near(consumer_step_bytes(plan, Consumer::DraftHead), 16.5e6, 0.02, "the Markov head's rows match");
    // One expert's slice: gate, up and down with their scales.
    double slice = 0;
    for (const Range& r : plan.layers[3].ranges)
      if (r.group == 0) slice += static_cast<double>(r.bytes);
    check(static_cast<uint64_t>(slice) == 4700160u, "one rank's slice of one expert is 4,700,160 bytes");
    // The prose shape: two verify rows.
    Dsv41StepShape prose = code;
    prose.rows = 2;
    const ByteTotals p = step_bytes(dsv41_step_plan(cfg, 0, kWorld, prose));
    check_near(p.total(), 5.653e9, 0.01, "the two-row step reads the hand count's bytes");
    check_near(p.router, 2.424e9, 0.01, "the two-row routed experts match the hand count");
    // The measured overlap of consecutive rows (2026-10-30, four texts through the real checkpoint): five rows read
    // 19.8 distinct experts a layer and two rows 10.1, where rows picking independently would read 28.9 and 11.8.
    Dsv41StepShape code_measured = code;
    code_measured.distinct_main = 19.8;
    code_measured.bf12_head = true;
    const ByteTotals cm = step_bytes(dsv41_step_plan(cfg, 0, kWorld, code_measured));
    check_near(cm.router, 5.633e9 + (19.8 - 28.9) * 4700160.0 * 40, 0.002, "five rows at the measured overlap read 3.92 GB of experts");
    Dsv41StepShape prose_measured = prose;
    prose_measured.distinct_main = 10.1;
    prose_measured.bf12_head = true;
    const ByteTotals pm = step_bytes(dsv41_step_plan(cfg, 0, kWorld, prose_measured));
    check_near(pm.router, 2.424e9 + (10.1 - 11.83) * 4700160.0 * 40, 0.002, "two rows at the measured overlap read 2.10 GB of experts");
    std::printf("   floors at the measured overlap, bf12 head: five rows %.3f GB = %.1f ms, two rows %.3f GB = %.1f ms at 240 GB/s\n",
                cm.total() / 1e9, cm.total() / 240e6, pm.total() / 1e9, pm.total() / 240e6);
    // A measured run list replaces the expectation.
    Dsv41StepShape counted = code;
    counted.distinct_main = 143;
    const ByteTotals c = step_bytes(dsv41_step_plan(cfg, 0, kWorld, counted));
    check_near(c.router - t.router, (143.0 - 28.9) * 4700160.0 * 40, 0.002, "143 distinct experts a layer add their slices");
    // The production template's bf12 head: each read is three quarters of the bf16 bytes.
    Dsv41StepShape packed = code;
    packed.bf12_head = true;
    const ByteTotals b = step_bytes(dsv41_step_plan(cfg, 0, kWorld, packed));
    check_near(t.total() - b.total(), 661.9e6 / 4, 0.002, "the bf12 head saves a quarter of its two reads");
    // The index scan is a context term.
    Dsv41StepShape scanned = code;
    scanned.scanned_keys = 16384;
    const ByteTotals s = step_bytes(dsv41_step_plan(cfg, 0, kWorld, scanned));
    check_near(s.token - t.token, 16384.0 * 132 * 5 * cfg.num_index_sources(), 0.001,
               "scanned index keys add 132 bytes a key a row on each index source");
  }

  // (3) every rank reads the same bytes
  {
    const ByteTotals t0 = step_bytes(plan);
    bool same = true;
    for (int rank = 1; rank < kWorld; ++rank) {
      const StepPlan pr = dsv41_step_plan(cfg, rank, kWorld, code);
      const ByteTotals tr = step_bytes(pr);
      same = same && std::fabs(tr.total() - t0.total()) < 1.0 && plan_check(pr).empty();
    }
    check(same, "ranks 1 to 3 read rank 0's bytes and pass the checks");
    // World 1 holds every expert slice whole.
    Dsv41StepShape one = code;
    const StepPlan w1 = dsv41_step_plan(cfg, 0, 1, one);
    check(plan_check(w1).empty(), "the world-1 plan passes the checks");
    double slice = 0;
    for (const Range& r : w1.layers[3].ranges)
      if (r.group == 0) slice += static_cast<double>(r.bytes);
    check(static_cast<uint64_t>(slice) == 4u * 4700160u, "world 1 holds four rank-slices of each expert");
    bool spans = true;
    for (const LayerPlan& lp : w1.layers) spans = spans && lp.span_bytes == Dsv41LayerStream::layer_bytes(cfg, lp.layer, 0, 1);
    check(spans, "every world-1 span is the loader's layer_bytes");
  }

  // (4) the arena: a permutation of the grant-order layout in consumption order
  {
    const Dsv41LocalGeometry geo = Dsv41LocalGeometry::from_config(cfg, 0, kWorld, LoaderHeadSharding::VocabSharded);
    void* base = reinterpret_cast<void*>(static_cast<uintptr_t>(1) << 40);
    bool sound = true, differs = false, ordered = true, pooled = true;
    for (int l : {0, 1, 2, 20, cfg.num_hidden_layers, cfg.max_layer() - 1}) {
      const std::vector<size_t> placement = dsv41_arena_placement(cfg, l, 0, kWorld);
      Dsv41LayerResident r;
      const size_t span = Dsv41LayerStream::layout_layer(cfg, l, 0, kWorld, base, r, &placement);
      const LayerPlan arena = dsv41_layer_plan(cfg, geo, r, base, span, code);
      sound = sound && plan_check(arena).empty() && arena.span_bytes == Dsv41LayerStream::layer_bytes(cfg, l, 0, kWorld);
      // The ranges in arena order: phases ascend, fixed before routed inside a
      // phase, and the routed experts of a phase in expert order with gate and
      // up of one expert adjacent (one pool), down in the next phase's pool.
      std::vector<const Range*> by_offset;
      for (const Range& rg : arena.ranges)
        if (rg.source == Source::Resident) by_offset.push_back(&rg);
      std::sort(by_offset.begin(), by_offset.end(), [](const Range* a, const Range* b) { return a->offset < b->offset; });
      int last_phase = -1, last_group = -1;
      bool routed_seen = false;
      for (const Range* rg : by_offset) {
        if (rg->phase < last_phase) ordered = false;
        if (rg->phase != last_phase) {
          last_phase = rg->phase;
          routed_seen = false;
          last_group = -1;
        }
        if (rg->dep == Dependency::Router) {
          routed_seen = true;
          if (rg->group < last_group) pooled = false;
          last_group = rg->group;
        } else if (routed_seen) {
          ordered = false;  // a fixed range after the phase's routed pool
        }
      }
      Dsv41LayerResident g;
      Dsv41LayerStream::layout_layer(cfg, l, 0, kWorld, base, g);
      differs = differs || (g.moe.experts.empty() ? false : g.moe.experts[0].payload != r.moe.experts[0].payload);
    }
    check(sound, "the arena layout of six layers tiles the loader's span and passes the checks");
    check(ordered, "arena ranges ascend by phase with a phase's fixed ranges before its routed pool");
    check(pooled, "a phase's routed experts sit in expert order");
    check(differs, "the arena moves the experts from their grant-order places");
    const ResidentLayout layout = dsv41_resident_layout(cfg, 0, kWorld);
    check(layout.version == kDsv41ArenaVersion && layout.placement && layout.placement(3) == dsv41_arena_placement(cfg, 3, 0, kWorld),
          "the resident layout carries the arena's version and placement");
  }

  std::printf("%d cases, %d failed\n", g_cases, g_failures);
  return g_failures == 0 ? 0 : 1;
}
