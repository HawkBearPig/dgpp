// The DeepSeek-V4.1 TP forwards (plan G5): loopback worlds of 2 and 4
// ranks in one process (real verbs QPs over 127.0.0.1, the bus the fabric
// uses) over the tiny fixture, against the world-1 model as the oracle.
// Asserted:
//   * cross-rank: every layer's residual streams, the final read, the
//     routing decisions and every index source's selection are bitwise
//     identical on every rank (the canonical rank-order fold; the indexer,
//     the router and the mHC coefficients are replicated);
//   * vs world 1, layer-local: the world-1 model walks the same tokens
//     with every layer started from the TP world's streams (the twin), so
//     each layer's output differs by that layer's folds alone (the
//     partial sums fold in bf16 on the wire: the attention's wo_b, the
//     MoE and the Engram kv) — within the numerics budget, no hard
//     element, the selections bitwise, the final read likewise, the
//     merged top-1 equal or a certified near tie;
//   * vs world 1, end to end: reported on the clean prefix (the rows
//     before the first routing or selection flip) and gated loosely —
//     the random-weight fixture amplifies rounding-level differences
//     layer over layer; layer 0 (every fold site once) is the tight
//     slice-and-fold gate;
//   * the Engram head sharding: world 4 puts one of the fixture's four
//     hash heads per rank, world 2 two — the kv partials fold to the
//     world-1 projection.
#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "common/bf16_residency.hpp"
#include "common/test.hpp"
#include "dsv41_fixture.hpp"
#include "engine/tp_bus.hpp"
#include "models/dsv41/config.hpp"
#include "models/dsv41/model.hpp"
#include "net/collective_bus.hpp"

namespace fs = std::filesystem;
using dgpp::bf16_bits_to_float;
using dgpp::BoundaryReducer;
using dgpp::BusBoundaryReducer;
using dgpp::BusStreamReducer;
using dgpp::Dsv41Model;
using dgpp::Dsv41Residency;
using dgpp::Dsv41TextConfig;
using dgpp::net::BusOptions;
using dgpp::net::CollectiveBus;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

int wait_timeout_ms() {
  const char* v = std::getenv("DGPP_TEST_WAIT_TIMEOUT_MS");
  return v ? std::atoi(v) : 60000;
}

BusOptions loop_options(int rank, int world, uint16_t port) {
  BusOptions o;
  o.world_size = world;
  o.my_rank = rank;
  o.rendezvous_port = port;
  o.rendezvous_host = rank == 0 ? "" : "127.0.0.1";
  o.rendezvous_timeout_ms = 20000;
  o.lat_slots = 8;
  // Every fold of this test rides the chunked latency path (qwen_tp_test's
  // note on the bulk machine at world 4 with sub-stripe buffers).
  o.lat_slot_bytes = 262144;
  o.bulk_slots = 8;
  o.bulk_slot_bytes = 262144;
  o.qp_depth = 1024;
  o.completion_timeout_ms = [] {
    const char* ms = std::getenv("DGPP_TEST_BUS_TIMEOUT_MS");
    return ms ? std::atoi(ms) : 5000;
  }();
  o.consumer_deadline_s = [] {
    const char* s = std::getenv("DGPP_TEST_CONSUMER_DEADLINE_S");
    return s ? std::atof(s) : 20.0;
  }();
  o.launch_consumers = false;
  return o;
}

std::vector<std::unique_ptr<CollectiveBus>> start_world(int world, uint16_t port) {
  std::vector<std::unique_ptr<CollectiveBus>> out;
  for (int r = 0; r < world; ++r) out.push_back(std::make_unique<CollectiveBus>(loop_options(r, world, port)));
  std::vector<std::string> errors(static_cast<size_t>(world));
  std::thread listener([&] {
    if (!out[0]->start(&errors[0])) DGPP_LOG_ERROR("dsv41 tp world rank 0: {}", errors[0]);
  });
  std::vector<std::thread> connectors;
  for (int r = 1; r < world; ++r)
    connectors.emplace_back([&, r] {
      if (!out[static_cast<size_t>(r)]->start(&errors[static_cast<size_t>(r)]))
        DGPP_LOG_ERROR("dsv41 tp world rank {}: {}", r, errors[static_cast<size_t>(r)]);
    });
  listener.join();
  for (auto& t : connectors) t.join();
  for (int r = 0; r < world; ++r)
    if (!errors[static_cast<size_t>(r)].empty()) return {};
  return out;
}

struct ConstructBarrier {
  std::mutex mu;
  std::condition_variable cv;
  int left;
  explicit ConstructBarrier(int world) : left(world) {}
  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mu);
    if (--left == 0) cv.notify_all();
    else cv.wait(lock, [this] { return left == 0; });
  }
};

struct RankOutcome {
  std::string error;
  Dsv41Model::Outputs out;
  std::vector<std::vector<float>> pre;  // per layer, the collapse coefficients the next site read
};

void rank_work(int rank, int world, const std::string& dir, const Dsv41TextConfig& cfg,
               const std::vector<int64_t>& tokens, CollectiveBus* bus, ConstructBarrier* barrier,
               RankOutcome* out) {
  bool arrived = false;
  auto arrive_once = [&] {
    if (arrived) return;
    arrived = true;
    barrier->arrive_and_wait();
  };
  try {
    BusBoundaryReducer reducer(*bus, wait_timeout_ms());
    Dsv41Model model(cfg, dir, static_cast<int>(tokens.size()), 512, Dsv41Residency::Streaming, &reducer, rank,
                     world);
    arrive_once();
    out->out = model.forward(tokens, true);
    out->pre = model.debug_layer_pre();
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("dsv41 tp rank {} failed: {}", rank, e.what());
    out->error = "rank " + std::to_string(rank) + ": " + e.what();
    arrive_once();
  }
}

std::vector<int64_t> make_tokens(const Dsv41TextConfig& cfg, int n) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = 0xA5A5F00DBEEF1234ull;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    t[static_cast<size_t>(i)] = static_cast<int64_t>(s % static_cast<uint64_t>(cfg.vocab_size));
  }
  return t;
}

bool bits_equal(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * 2) == 0);
}

struct Stats {
  double l2 = 0, max_ulps = 0;
  long hard = 0, total = 0;
};

int bf16_ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) -> int32_t {
    return (v & 0x8000u) ? -static_cast<int32_t>(v & 0x7FFFu) : static_cast<int32_t>(v & 0x7FFFu);
  };
  return std::abs(static_cast<int>(key(a) - key(b)));
}

Stats compare(const std::vector<uint16_t>& got, const std::vector<uint16_t>& want) {
  Stats s;
  s.total = static_cast<long>(got.size());
  double rms = 0;
  for (uint16_t v : want) rms += std::pow(bf16_bits_to_float(v), 2);
  rms = std::sqrt(rms / std::max<size_t>(want.size(), 1));
  double d2 = 0, o2 = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double g = bf16_bits_to_float(got[i]), w = bf16_bits_to_float(want[i]);
    int u = bf16_ulps(got[i], want[i]);
    // The floor under the ulp count: a world's fold sums the ranks' bf16
    // partials, each rounded at its own magnitude, so an element near
    // zero whose partials cancel carries the partials' rounding (four
    // partials of a few times the rms: a few percent of it) — 5 % of the
    // layer's rms, which a defect's whole-row error dwarfs.
    if (std::fabs(g - w) <= 0.05 * rms) u = 0;
    s.max_ulps = std::max(s.max_ulps, static_cast<double>(u));
    if (u > 128) ++s.hard;
    d2 += (g - w) * (g - w);
    o2 += w * w;
  }
  s.l2 = std::sqrt(d2) / std::sqrt(o2 + 1e-30);
  return s;
}

// The merged top-1 over the ranks' vocab slices: the highest logit, ties
// to the lower id.
std::vector<std::pair<int32_t, float>> merged_top1(const std::vector<RankOutcome>& ranks, int T) {
  std::vector<std::pair<int32_t, float>> best(static_cast<size_t>(T), {-1, -INFINITY});
  for (const RankOutcome& r : ranks)
    for (int t = 0; t < T; ++t)
      for (int c = 0; c < r.out.lm_vocab_count; ++c) {
        const float v = r.out.logits[static_cast<size_t>(t) * r.out.lm_vocab_count + c];
        const int32_t id = r.out.lm_vocab_begin + c;
        auto& b = best[static_cast<size_t>(t)];
        if (v > b.second || (v == b.second && id < b.first)) b = {id, v};
      }
  return best;
}

void check_world(int world, uint16_t port, const std::string& dir, const Dsv41TextConfig& cfg,
                 const std::vector<int64_t>& tokens, const Dsv41Model::Outputs& ref) {
  DGPP_LOG_INFO("dsv41 TP loopback world={} tokens={}", world, tokens.size());
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(world, port);
  require(!buses.empty(), "tp bus world failed to start");
  std::vector<RankOutcome> ranks(static_cast<size_t>(world));
  ConstructBarrier barrier(world);
  std::vector<std::thread> workers;
  for (int r = 0; r < world; ++r)
    workers.emplace_back(rank_work, r, world, dir, std::cref(cfg), std::cref(tokens),
                         buses[static_cast<size_t>(r)].get(), &barrier, &ranks[static_cast<size_t>(r)]);
  for (auto& t : workers) t.join();
  for (int r = 0; r < world; ++r) require(ranks[static_cast<size_t>(r)].error.empty(), ranks[static_cast<size_t>(r)].error);
  const int T = static_cast<int>(tokens.size());
  const int L = cfg.num_hidden_layers;
  const int W = 4 * cfg.hidden_size;  // the four residual streams per row
  // ---- cross-rank bitwise ------------------------------------------------------
  for (int r = 1; r < world; ++r) {
    const Dsv41Model::Outputs& a = ranks[0].out;
    const Dsv41Model::Outputs& b = ranks[static_cast<size_t>(r)].out;
    require(bits_equal(a.final_hidden_bits, b.final_hidden_bits), "final hidden differs across ranks");
    require(a.layer_states.size() == static_cast<size_t>(L) && b.layer_states.size() == static_cast<size_t>(L),
            "layer captures missing");
    for (int l = 0; l < L; ++l)
      require(bits_equal(a.layer_states[static_cast<size_t>(l)], b.layer_states[static_cast<size_t>(l)]),
              "layer " + std::to_string(l) + " streams differ across ranks");
    require(a.route_ids == b.route_ids, "routing differs across ranks");
    require(a.dsa_selections == b.dsa_selections, "the CSA2 selections differ across ranks");
  }
  // ---- vs world 1, end to end -----------------------------------------------------
  // The folds reassociate (bf16 on the wire), so every element differs at
  // rounding level, and the random-weight network amplifies any such
  // difference some 1.5x per layer (the pure-python reference of
  // dsv41_forward_test departs from the engine the same way: 6e-4 at
  // layer 0, 1.3e-2 at layer 7), the fp4 block cache flipping codes on the
  // way (a value element moves by a quarter or more per flipped e2m1
  // code). The end-to-end comparison is therefore REPORTED on the clean
  // prefix (the rows before the first routing or selection flip: a
  // flipped row's cached entries perturb every later query) and gated
  // loosely; layer 0 — every fold site once, no cache of its own yet
  // amplifying — is the slice-and-fold gate (a wrong slice or scale grid
  // reads as a tenfold l2). The layer-local gate follows below.
  const Dsv41Model::Outputs& o = ranks[0].out;
  const int K = cfg.num_experts_per_tok;
  require(o.route_ids.size() == static_cast<size_t>(L) && ref.route_ids.size() == o.route_ids.size(), "route captures");
  require(o.dsa_selections.size() == static_cast<size_t>(cfg.num_index_sources()) &&
              ref.dsa_selections.size() == o.dsa_selections.size(),
          "selection captures");
  std::vector<std::string> failures;
  const auto budget = [&](bool ok, const std::string& what) {
    if (!ok) failures.push_back(what);
  };
  {
    std::vector<bool> flipped(static_cast<size_t>(T), false);
    long flips = 0, slots = 0, sel_flips = 0;
    int index_ordinal = 0;
    for (int l = 0; l < L; ++l) {
      if (cfg.is_index_source(l)) {
        const std::vector<int32_t>& a = o.dsa_selections[static_cast<size_t>(index_ordinal)];
        const std::vector<int32_t>& b = ref.dsa_selections[static_cast<size_t>(index_ordinal)];
        const size_t ms = a.size() / static_cast<size_t>(T);
        for (int t = 0; t < T; ++t)
          if (std::memcmp(a.data() + static_cast<size_t>(t) * ms, b.data() + static_cast<size_t>(t) * ms, ms * 4) != 0) {
            ++sel_flips;
            flipped[static_cast<size_t>(t)] = true;
          }
        ++index_ordinal;
      }
      for (int t = 0; t < T; ++t)
        for (int i = 0; i < K; ++i, ++slots)
          if (o.route_ids[static_cast<size_t>(l)][static_cast<size_t>(t) * K + i] !=
              ref.route_ids[static_cast<size_t>(l)][static_cast<size_t>(t) * K + i]) {
            ++flips;
            flipped[static_cast<size_t>(t)] = true;
          }
      int first_flip = T;
      for (int t = 0; t < T; ++t)
        if (flipped[static_cast<size_t>(t)]) {
          first_flip = t;
          break;
        }
      std::vector<uint16_t> kept_got(o.layer_states[static_cast<size_t>(l)].begin(),
                                     o.layer_states[static_cast<size_t>(l)].begin() + static_cast<size_t>(first_flip) * W);
      std::vector<uint16_t> kept_want(ref.layer_states[static_cast<size_t>(l)].begin(),
                                      ref.layer_states[static_cast<size_t>(l)].begin() + static_cast<size_t>(first_flip) * W);
      const Stats s = compare(kept_got, kept_want);
      std::printf("[ .. ] world %d end to end, layer %d (clean prefix %d of %d rows): l2 %.3g max %g ulps hard %ld of %ld\n",
                  world, l, first_flip, T, s.l2, s.max_ulps, s.hard, s.total);
      if (l == 0)
        budget(first_flip == T && s.l2 < 0.01 && s.hard == 0,
               "world " + std::to_string(world) + " layer 0 (the slice-and-fold gate) outside the numerics budget");
      else if (first_flip >= 2)
        budget(s.l2 < 0.1, "world " + std::to_string(world) + " layer " + std::to_string(l) + " end to end beyond 10 %");
    }
    std::printf("[ .. ] world %d end to end: %ld of %ld routing slots differ from world 1; %ld selection flips\n", world,
                flips, slots, sel_flips);
    budget(flips <= slots / 20, "too many routing flips against world 1");
  }
  // ---- the layer-local gate: the world-1 twin ---------------------------------------
  // The world-1 model walks the same tokens with every layer l > 0 (and the
  // head) started from the TP world's streams after layer l - 1 and the
  // collapse coefficients its next site read: each layer's output then
  // differs from the TP world's by that layer's folds alone (the
  // attention's wo_b, the MoE, the Engram kv — bf16 on the wire once
  // each) and the layer's own amplification of them — no cascade. The
  // caches, the compressor, the indexer and the candidate pool read the
  // identical input, so every selection is bitwise; the router reads the
  // streams after the attention fold, so a near tie may still flip a row
  // (exempt at that layer, bounded).
  Dsv41Model::Outputs tw;
  {
    Dsv41Model twin(cfg, dir, T, 512);
    tw = twin.forward(tokens, true, &o.layer_states, &ranks[0].pre);
  }
  require(tw.layer_states.size() == static_cast<size_t>(L) && tw.dsa_selections.size() == o.dsa_selections.size(),
          "twin captures");
  {
    long flips = 0, slots = 0;
    int index_ordinal = 0;
    for (int l = 0; l < L; ++l) {
      if (cfg.is_index_source(l)) {
        budget(tw.dsa_selections[static_cast<size_t>(index_ordinal)] == o.dsa_selections[static_cast<size_t>(index_ordinal)],
               "world " + std::to_string(world) + " layer " + std::to_string(l) +
                   ": the selections differ from the twin's on the same input");
        ++index_ordinal;
      }
      std::vector<bool> row_flip(static_cast<size_t>(T), false);
      for (int t = 0; t < T; ++t)
        for (int i = 0; i < K; ++i, ++slots)
          if (o.route_ids[static_cast<size_t>(l)][static_cast<size_t>(t) * K + i] !=
              tw.route_ids[static_cast<size_t>(l)][static_cast<size_t>(t) * K + i]) {
            ++flips;
            row_flip[static_cast<size_t>(t)] = true;
          }
      std::vector<uint16_t> kept_got, kept_want;
      int kept = 0;
      double worst = 0;
      int worst_row = -1;
      for (int t = 0; t < T; ++t) {
        if (row_flip[static_cast<size_t>(t)]) continue;
        const std::vector<uint16_t> got(o.layer_states[static_cast<size_t>(l)].begin() + static_cast<size_t>(t) * W,
                                        o.layer_states[static_cast<size_t>(l)].begin() + static_cast<size_t>(t + 1) * W);
        const std::vector<uint16_t> want(tw.layer_states[static_cast<size_t>(l)].begin() + static_cast<size_t>(t) * W,
                                         tw.layer_states[static_cast<size_t>(l)].begin() + static_cast<size_t>(t + 1) * W);
        const Stats r = compare(got, want);
        if (r.l2 > worst) {
          worst = r.l2;
          worst_row = t;
        }
        kept_got.insert(kept_got.end(), got.begin(), got.end());
        kept_want.insert(kept_want.end(), want.begin(), want.end());
        ++kept;
      }
      const Stats s = compare(kept_got, kept_want);
      std::printf("[ .. ] world %d layer-local, layer %d (%d of %d rows routed alike): l2 %.3g max %g ulps hard %ld of %ld; worst row t%d %.3g\n",
                  world, l, kept, T, s.l2, s.max_ulps, s.hard, s.total, worst_row, worst);
      if (s.hard > 0) {
        // The hard elements themselves (the evidence: a value against its twin and the layer's rms).
        double rms = 0;
        for (uint16_t v : kept_want) rms += std::pow(bf16_bits_to_float(v), 2);
        rms = std::sqrt(rms / std::max<size_t>(kept_want.size(), 1));
        int shown = 0;
        for (size_t i = 0; i < kept_got.size() && shown < 6; ++i) {
          const double g = bf16_bits_to_float(kept_got[i]), w = bf16_bits_to_float(kept_want[i]);
          if (std::fabs(g - w) <= 0.02 * rms || bf16_ulps(kept_got[i], kept_want[i]) <= 128) continue;
          std::printf("[ .. ]   hard element %zu (row %zu, stream %zu, channel %zu): got %.5g want %.5g (rms %.4g)\n", i, i / W,
                      (i % W) / (W / 4), (i % W) % (W / 4), g, w, rms);
          ++shown;
        }
      }
      budget(kept >= T - 2 && s.l2 < 0.01 && s.hard == 0,
             "world " + std::to_string(world) + " layer " + std::to_string(l) + " layer-local outside the numerics budget");
    }
    std::printf("[ .. ] world %d layer-local: %ld of %ld routing slots differ from the twin\n", world, flips, slots);
    budget(flips <= slots / 100, "too many layer-local routing flips against the twin");
    // The head from the TP world's last streams: the replicated collapse
    // and norm (the collapse coefficients come from the twin's own last
    // site, rounding level apart), the vocab-sharded lm head.
    const Stats f = compare(o.final_hidden_bits, tw.final_hidden_bits);
    std::printf("[ .. ] world %d layer-local final hidden: l2 %.3g max %g ulps hard %ld of %ld\n", world, f.l2, f.max_ulps,
                f.hard, f.total);
    budget(f.l2 < 0.01 && f.hard == 0, "world " + std::to_string(world) + " final hidden outside the layer-local budget");
    const auto merged = merged_top1(ranks, T);
    const std::vector<std::pair<int32_t, float>> single = merged_top1({RankOutcome{"", tw, {}}}, T);
    int mism = 0;
    for (int t = 0; t < T; ++t) {
      if (merged[static_cast<size_t>(t)].first == single[static_cast<size_t>(t)].first) continue;
      const float v_tp = tw.logits[static_cast<size_t>(t) * tw.lm_vocab_count + merged[static_cast<size_t>(t)].first];
      const float v_1 = single[static_cast<size_t>(t)].second;
      if (std::fabs(v_1 - v_tp) > 0.02 * std::fabs(v_1)) ++mism;
    }
    std::printf("[ .. ] world %d layer-local top-1: %d of %d rows beyond a near tie\n", world, mism, T);
    budget(mism == 0, "world " + std::to_string(world) + " top-1 differs from the twin's beyond a near tie");
  }
  std::string all;
  for (const std::string& f : failures) all += (all.empty() ? "" : "; ") + f;
  require(failures.empty(), all);
}

}  // namespace

DGPP_TEST(dsv41_tp_loopback_worlds_2_and_4_match_world_1) {
  const std::string dir = (fs::current_path() / "dsv41_tp_fixture").string();
  const Dsv41TextConfig cfg = dsv41fx::tiny_config();
  dsv41fx::write_fixture(cfg, dir);
  (void)dsv41fx::fixture_sidecar(cfg, dir);
  // 70 rows: past the 16-entry select horizon at ratio 2 and the candidate
  // pool's blocks at ratio 1.
  const std::vector<int64_t> tokens = make_tokens(cfg, 70);
  Dsv41Model::Outputs ref;
  {
    Dsv41Model single(cfg, dir, static_cast<int>(tokens.size()), 512);
    ref = single.forward(tokens, true);
  }
  check_world(2, 29958, dir, cfg, tokens, ref);
  check_world(4, 29959, dir, cfg, tokens, ref);
}

// The decode-shaped prefill: 24 rows take the dense projections' and the
// head's decode form (the streaming tensor-core GEMM, rows 1..32 in one
// launch) — the sharded slices' row offsets, per-rank k and scale grids
// under that kernel against world 1's, layer-locally, with the same
// gates. The 70-row test above prefills through the tile kernels.
// The group prefill at world 2: three cold prompts as one walk against
// the shortest prefilled alone — per layer, its rows' streams bitwise
// (world 1 is bitwise by dsv41_model_test; the TP folds must keep it so
// whichever bus path the wider boundary takes). A small group (12 rows,
// inside one latency slot) and a wide one (51 rows, the bulk path).
namespace {
struct GroupOutcome {
  std::string error;
  std::vector<std::vector<uint16_t>> alone;   // per layer [T_c x 4H]
  std::vector<std::vector<uint16_t>> group;   // per layer [T_total x 4H]
  std::vector<float> alone_logits, group_logits;
  std::vector<Dsv41Model::SiteCapture> alone_sites, group_sites;  // per layer
  int rows_c = 0, rows_total = 0;
};
void rank_work_group(int rank, int world, const std::string& dir, const Dsv41TextConfig& cfg,
                     const std::vector<std::vector<int64_t>>& prompts, CollectiveBus* bus, ConstructBarrier* barrier,
                     GroupOutcome* out, bool stream_folds = false) {
  bool arrived = false;
  auto arrive_once = [&] {
    if (arrived) return;
    arrived = true;
    barrier->arrive_and_wait();
  };
  try {
    // The host-driven reducer, or the stream-ordered one (plan D9: the
    // folds launch on the model stream; the same chain, so the walk's
    // every state is bitwise the host-driven walk's).
    std::unique_ptr<BoundaryReducer> reducer;
    if (stream_folds)
      reducer = std::make_unique<BusStreamReducer>(*bus, wait_timeout_ms());
    else
      reducer = std::make_unique<BusBoundaryReducer>(*bus, wait_timeout_ms());
    Dsv41Model model(cfg, dir, 96, 512, Dsv41Residency::Streaming, reducer.get(), rank, world, 3);
    arrive_once();
    setenv("DGPP_DSV41_CAPTURE_PREFILL", "1", 1);
    const Dsv41Model::Outputs a = model.session_prefill(2, prompts[2]);
    out->alone = a.layer_states;
    out->alone_logits = a.logits;
    out->alone_sites = model.debug_sites();
    model.session_close(2);
    const std::vector<const std::vector<int64_t>*> pp = {&prompts[0], &prompts[1], &prompts[2]};
    const std::vector<Dsv41Model::Outputs> g = model.session_prefill_group({0, 1, 2}, pp);
    unsetenv("DGPP_DSV41_CAPTURE_PREFILL");
    out->group = g[0].layer_states;
    out->group_logits = g[2].logits;
    out->group_sites = model.debug_sites();
    out->rows_c = static_cast<int>(prompts[2].size());
    out->rows_total = static_cast<int>(prompts[0].size() + prompts[1].size() + prompts[2].size());
  } catch (const std::exception& e) {
    out->error = "rank " + std::to_string(rank) + ": " + e.what();
    arrive_once();
  }
}
}  // namespace

DGPP_TEST(dsv41_tp_group_prefill_is_bitwise_the_prefill_alone_at_world_2) {
  const std::string dir = (fs::current_path() / "dsv41_tp_fixture_group").string();
  const Dsv41TextConfig cfg = dsv41fx::tiny_config();
  dsv41fx::write_fixture(cfg, dir);
  (void)dsv41fx::fixture_sidecar(cfg, dir);
  const int W = 4 * cfg.hidden_size;
  // Four loopback worlds from here (2026-09-30): the block 29968–29971 it
  // used before is the deployment defaults' (fabric 29970, journal 29971),
  // so the fourth world could not bind beside a live deployment on the node.
  uint16_t port = 29980;
  for (const auto lens : {std::array<int, 3>{5, 4, 3}, std::array<int, 3>{23, 17, 11}}) {
    std::vector<std::vector<int64_t>> prompts;
    for (int i = 0; i < 3; ++i) {
      std::vector<int64_t> p = make_tokens(cfg, 64);
      p.resize(static_cast<size_t>(lens[static_cast<size_t>(i)]));
      for (auto& t : p) t = (t + 17 * i) % cfg.vocab_size;
      prompts.push_back(p);
    }
    std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(2, port++);
    require(!buses.empty(), "tp bus world failed to start");
    std::vector<GroupOutcome> ranks(2);
    ConstructBarrier barrier(2);
    std::vector<std::thread> workers;
    for (int r = 0; r < 2; ++r)
      workers.emplace_back(rank_work_group, r, 2, dir, std::cref(cfg), std::cref(prompts), buses[static_cast<size_t>(r)].get(),
                           &barrier, &ranks[static_cast<size_t>(r)], false);
    for (auto& t : workers) t.join();
    for (int r = 0; r < 2; ++r) require(ranks[static_cast<size_t>(r)].error.empty(), ranks[static_cast<size_t>(r)].error);
    const GroupOutcome& o = ranks[0];
    require(o.alone.size() == o.group.size() && !o.alone.empty(), "layer captures on both walks");
    const int L = static_cast<int>(o.alone.size());
    int first_bad = -1;
    for (int l = 0; l < L; ++l) {
      const std::vector<uint16_t>& a = o.alone[static_cast<size_t>(l)];
      const std::vector<uint16_t>& g = o.group[static_cast<size_t>(l)];
      require(a.size() == static_cast<size_t>(o.rows_c) * W && g.size() == static_cast<size_t>(o.rows_total) * W, "capture shapes");
      const std::vector<uint16_t> gc(g.begin() + static_cast<std::ptrdiff_t>(o.rows_total - o.rows_c) * W, g.end());
      const Stats s = compare(gc, a);
      std::printf("[ .. ] group %d+%d+%d rows, layer %d: C's rows vs alone l2 %.3g max %g ulps%s\n", lens[0], lens[1], lens[2],
                  l, s.l2, s.max_ulps, bits_equal(gc, a) ? " (bitwise)" : "");
      if (!bits_equal(gc, a)) {
        int shown = 0;
        for (size_t i = 0; i < gc.size() && shown < 4; ++i)
          if (gc[i] != a[i]) {
            std::printf("[ .. ]     row %zu stream %zu dim %zu: group 0x%04x (%g) vs alone 0x%04x (%g)\n", i / W,
                        (i % W) / static_cast<size_t>(cfg.hidden_size), i % static_cast<size_t>(cfg.hidden_size), gc[i],
                        bf16_bits_to_float(gc[i]), a[i], bf16_bits_to_float(a[i]));
            ++shown;
          }
        if (first_bad < 0) first_bad = l;
        // The layer's sites for C's rows: which one first departs.
        if (o.alone_sites.size() > static_cast<size_t>(l) && o.group_sites.size() > static_cast<size_t>(l)) {
          const Dsv41Model::SiteCapture& sa = o.alone_sites[static_cast<size_t>(l)];
          const Dsv41Model::SiteCapture& sg = o.group_sites[static_cast<size_t>(l)];
          const size_t H = static_cast<size_t>(cfg.hidden_size);
          auto site = [&](const char* name, const std::vector<uint16_t>& av, const std::vector<uint16_t>& gv, size_t width) {
            if (av.size() != static_cast<size_t>(o.rows_c) * width || gv.size() != static_cast<size_t>(o.rows_total) * width) {
              std::printf("[ .. ]     site %s: shapes %zu / %zu\n", name, av.size(), gv.size());
              return;
            }
            const std::vector<uint16_t> gcv(gv.begin() + static_cast<std::ptrdiff_t>((o.rows_total - o.rows_c) * width), gv.end());
            size_t nd = 0, firsti = 0;
            for (size_t i = 0; i < av.size(); ++i)
              if (av[i] != gcv[i]) { if (nd == 0) firsti = i; ++nd; }
            std::printf("[ .. ]     site %s: %zu differing elements%s\n", name, nd,
                        nd ? (" (first row " + std::to_string(firsti / width) + " col " + std::to_string(firsti % width) + ")").c_str() : "");
          };
          site("x_attn", sa.x_attn, sg.x_attn, H);
          site("attn_out", sa.attn_out, sg.attn_out, H);
          site("streams_after_attn", sa.streams_after_attn, sg.streams_after_attn, 4 * H);
          site("x_ffn", sa.x_ffn, sg.x_ffn, H);
          site("ffn_out", sa.ffn_out, sg.ffn_out, H);
        }
      }
    }
    require(o.alone_logits == o.group_logits, "the last row's logits bitwise");
    require(first_bad < 0, "C's rows differ from its prefill alone from layer " + std::to_string(first_bad));
    // The same group through the stream-ordered reducer (plan D9): every
    // layer's rows and the logits bitwise the host-driven walk's.
    {
      std::vector<std::unique_ptr<CollectiveBus>> sbuses = start_world(2, port++);
      require(!sbuses.empty(), "tp bus world (stream folds) failed to start");
      std::vector<GroupOutcome> sranks(2);
      ConstructBarrier sbarrier(2);
      std::vector<std::thread> sworkers;
      for (int r = 0; r < 2; ++r)
        sworkers.emplace_back(rank_work_group, r, 2, dir, std::cref(cfg), std::cref(prompts),
                              sbuses[static_cast<size_t>(r)].get(), &sbarrier, &sranks[static_cast<size_t>(r)], true);
      for (auto& t : sworkers) t.join();
      for (int r = 0; r < 2; ++r) require(sranks[static_cast<size_t>(r)].error.empty(), sranks[static_cast<size_t>(r)].error);
      const GroupOutcome& so = sranks[0];
      require(so.group.size() == o.group.size() && so.alone.size() == o.alone.size(), "stream folds: layer captures");
      for (size_t l = 0; l < o.group.size(); ++l) {
        require(bits_equal(so.group[l], o.group[l]), "stream folds: the group's layer " + std::to_string(l) +
                                                          " rows differ from the host-driven walk's");
        require(bits_equal(so.alone[l], o.alone[l]), "stream folds: the alone walk's layer " + std::to_string(l) +
                                                          " rows differ from the host-driven walk's");
      }
      require(so.group_logits == o.group_logits && so.alone_logits == o.alone_logits,
              "stream folds: the logits differ from the host-driven walk's");
      std::printf("[ OK ] group %d+%d+%d rows through the stream-ordered reducer: bitwise the host-driven walk\n", lens[0],
                  lens[1], lens[2]);
    }
  }
}

DGPP_TEST(dsv41_tp_loopback_worlds_2_and_4_match_world_1_at_decode_rows) {
  const std::string dir = (fs::current_path() / "dsv41_tp_fixture_rows").string();
  const Dsv41TextConfig cfg = dsv41fx::tiny_config();
  dsv41fx::write_fixture(cfg, dir);
  (void)dsv41fx::fixture_sidecar(cfg, dir);
  const std::vector<int64_t> tokens = make_tokens(cfg, 24);
  Dsv41Model::Outputs ref;
  {
    Dsv41Model single(cfg, dir, static_cast<int>(tokens.size()), 512);
    ref = single.forward(tokens, true);
  }
  check_world(2, 29966, dir, cfg, tokens, ref);
  check_world(4, 29967, dir, cfg, tokens, ref);
}

namespace {
// A reducer at world 1 for the fold-overlap gates: every fold an identity,
// the asynchronous ones counted (dsv4_tp_test's).
struct LocalAsyncReducer : dgpp::BoundaryReducer {
  void reduce(uint16_t*, int, int) override { ++folds; }
  bool begin_async(uint16_t*, int, int) override {
    if (open) throw std::logic_error("local reducer: one fold outstanding at a time");
    open = true;
    ++folds;
    return true;
  }
  void end_async() override { open = false; }
  bool open = false;
  int folds = 0;
};
}  // namespace

// The prefill's fold overlap (engine.prefill_fold_overlap, 2026-10-18): a
// 701-row prompt walks each layer as the row blocks 384 + 317 (the
// boundary on the request's block grid nearest the middle); its logits and
// the four decode steps behind it bitwise the one-block walk's, in the
// exact prefill (every layer two blocks) and the bounded one (the encoder
// layers two blocks; the decoder's prologue at its first layer lands the
// pending fold before it reads the streams). Then three prompts of 141,
// 701 and 300 rows as one group walk whose middle row stands inside the
// second span: the boundary moves to that span's block grid, and each
// prompt's logits and steps are bitwise its prefill alone.
DGPP_TEST(dsv41_tp_prefill_fold_overlap_is_bitwise_the_one_block_walk) {
  const std::string dir = (fs::current_path() / "dsv41_tp_fixture_overlap").string();
  const Dsv41TextConfig cfg = dsv41fx::tiny_config();
  dsv41fx::write_fixture(cfg, dir);
  (void)dsv41fx::fixture_sidecar(cfg, dir);
  std::vector<int64_t> prompt = make_tokens(cfg, 160);
  while (prompt.size() < 701) prompt.push_back((prompt[prompt.size() - 160] * 31 + 11) % cfg.vocab_size);
  for (const bool bounded : {false, true}) {
    LocalAsyncReducer reducer;
    Dsv41Model model(cfg, dir, 1280, 4096, Dsv41Residency::Streaming, nullptr, 0, 1, /*max_requests=*/3);
    (void)model.set_boundary(&reducer);    // (the constructor takes a reducer at tp_world > 1 only)
    model.set_decode_route_traces(false);  // as the engine serves it (a traced walk stays one block)
    model.set_prefill_bounded(bounded);
    const auto run = [&](int slot, const std::vector<int64_t>& p, bool overlap, std::vector<float>* logits,
                         std::vector<std::vector<float>>* steps) {
      model.set_prefill_fold_overlap(overlap);
      Dsv41Model::Outputs o = model.session_prefill(slot, p);
      *logits = o.logits;
      for (int k = 0; k < 4; ++k) {
        o = model.session_step(slot, (p[static_cast<size_t>(k)] + 7) % cfg.vocab_size);
        steps->push_back(o.logits);
      }
      model.session_close(slot);
    };
    // 1) one prompt: 701 rows as blocks of 384 and 317.
    std::vector<float> on_logits, off_logits;
    std::vector<std::vector<float>> on_steps, off_steps;
    run(0, prompt, true, &on_logits, &on_steps);
    const uint64_t async_folds = model.debug_overlap_async_folds();
    run(1, prompt, false, &off_logits, &off_steps);
    require(model.debug_overlap_async_folds() == async_folds, "the one-block walk began an asynchronous fold");
    require(async_folds > 0, "fold overlap: no fold ran asynchronously");
    require(!on_logits.empty() && on_logits == off_logits,
            "fold overlap: the prefill's logits differ from the one-block walk's");
    require(on_steps == off_steps, "fold overlap: decode steps behind the prefill differ (the state)");
    std::printf("[ OK ] %s prefill: 701 rows in two row blocks (%llu asynchronous folds): logits and 4 decode steps "
                "bitwise the one-block walk\n", bounded ? "bounded" : "exact", static_cast<unsigned long long>(async_folds));
    // 2) a group's spans.
    const std::vector<std::vector<int64_t>> group = {
        std::vector<int64_t>(prompt.begin(), prompt.begin() + 141), prompt,
        std::vector<int64_t>(prompt.begin() + 200, prompt.begin() + 500)};
    std::vector<std::vector<float>> alone_logits(3);
    std::vector<std::vector<std::vector<float>>> alone_steps(3), group_steps(3);
    for (int r = 0; r < 3; ++r)
      run(r, group[static_cast<size_t>(r)], false, &alone_logits[static_cast<size_t>(r)], &alone_steps[static_cast<size_t>(r)]);
    model.set_prefill_fold_overlap(true);
    const std::vector<const std::vector<int64_t>*> pp = {&group[0], &group[1], &group[2]};
    const std::vector<Dsv41Model::Outputs> g = model.session_prefill_group({0, 1, 2}, pp);
    const uint64_t group_folds = model.debug_overlap_async_folds() - async_folds;
    require(g.size() == 3, "three prompts prefilled as a group");
    require(group_folds > 0, "fold overlap: the group walk ran no fold asynchronously");
    for (int r = 0; r < 3; ++r) {
      const std::vector<int64_t>& p = group[static_cast<size_t>(r)];
      require(g[static_cast<size_t>(r)].logits == alone_logits[static_cast<size_t>(r)],
              "fold overlap: a group walk's logits differ from the prompt's one-block prefill alone");
      for (int k = 0; k < 4; ++k)
        group_steps[static_cast<size_t>(r)].push_back(
            model.session_step(r, (p[static_cast<size_t>(k)] + 7) % cfg.vocab_size).logits);
      require(group_steps[static_cast<size_t>(r)] == alone_steps[static_cast<size_t>(r)],
              "fold overlap: the steps behind a group walk differ (the state)");
      model.session_close(r);
    }
    std::printf("[ OK ] %s prefill: 141 + 701 + 300 rows as one group walk in two row blocks (%llu asynchronous folds, "
                "the boundary inside the second span on its block grid): logits and steps bitwise the prefills alone\n",
                bounded ? "bounded" : "exact", static_cast<unsigned long long>(group_folds));
  }
}

// Every decode form of the 2026-10-19 round repeats itself bitwise run to
// run across request slots at world 1: the scalar and the flash window
// (in line and beside the main source) and the fused mHC site. (The flash
// window's first form read a zero-filled ring for every slot but 0 — the
// listed kernel bounded a ring block by the table's row width — and this
// gate's slot pair is what caught it.)
DGPP_TEST(dsv41_decode_forms_repeat_bitwise_across_slots) {
  const std::string dir = (fs::current_path() / "dsv41_tp_fixture_forms").string();
  const Dsv41TextConfig cfg = dsv41fx::tiny_config();
  dsv41fx::write_fixture(cfg, dir);
  (void)dsv41fx::fixture_sidecar(cfg, dir);
  const std::vector<int64_t> prompt = make_tokens(cfg, 96);
  LocalAsyncReducer reducer;
  Dsv41Model model(cfg, dir, 1280, 4096, Dsv41Residency::Streaming, nullptr, 0, 1, /*max_requests=*/2);
  (void)model.set_boundary(&reducer);
  model.set_decode_route_traces(false);
  const auto run = [&](int slot, std::vector<std::vector<float>>* steps) {
    Dsv41Model::Outputs o = model.session_prefill(slot, prompt);
    steps->push_back(o.logits);
    for (int k = 0; k < 6; ++k) {
      o = model.session_step(slot, (prompt[static_cast<size_t>(k)] + 7) % cfg.vocab_size);
      steps->push_back(o.logits);
    }
    model.session_close(slot);
  };
  struct Arm { bool fused, flash, beside; };
  for (const Arm arm : {Arm{false, false, false}, Arm{false, true, false}, Arm{false, true, true}, Arm{true, true, true}}) {
    std::vector<std::vector<float>> a, b;
    model.set_decode_mhc_fused(arm.fused);
    model.set_decode_window_flash(arm.flash);
    model.set_decode_window_beside(arm.beside);
    run(0, &a);
    run(1, &b);
    size_t first = a.size();
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
      if (a[i] != b[i]) { first = i; break; }
    std::printf("  determinism: mhc %s, window %s %s: %s\n", arm.fused ? "fused" : "unfused", arm.flash ? "flash" : "scalar",
                arm.beside ? "beside" : "in line", first == a.size() ? "bitwise run to run" : ("step " + std::to_string(first) + " differs").c_str());
    require(first == a.size() && !a.empty(), std::string("the decode step is not bitwise across slots (mhc ") +
                                                 (arm.fused ? "fused" : "unfused") + ", window " + (arm.flash ? "flash" : "scalar") +
                                                 (arm.beside ? " beside" : " in line") + ")");
  }
  model.set_decode_mhc_fused(true);
  model.set_decode_window_flash(true);
  model.set_decode_window_beside(true);
  std::printf("[ OK ] decode forms: the prefill and 6 decode steps bitwise across slots for every form\n");
}

// The overlap on a loopback world of 2: the bus's bulk all-reduce runs
// asynchronously under the other block's compute (block A's fold exceeds
// the reducer's asynchronous threshold of two latency slots at the
// fixture's width); every rank's logits bitwise its one-block walk's.
DGPP_TEST(dsv41_tp_prefill_fold_overlap_at_world_2_is_bitwise_the_one_block_walk) {
  const std::string dir = (fs::current_path() / "dsv41_tp_fixture_overlap_w2").string();
  const Dsv41TextConfig cfg = dsv41fx::tiny_config();
  dsv41fx::write_fixture(cfg, dir);
  (void)dsv41fx::fixture_sidecar(cfg, dir);
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(2, 29989);
  require(!buses.empty(), "tp bus world failed to start");
  const size_t slot = dgpp::bus_latency_slot_elems(*buses[0]);
  int TA = 128;
  while (static_cast<size_t>(TA) * static_cast<size_t>(cfg.hidden_size) <= 2 * slot) TA += 128;
  const int T = 2 * TA;
  require(T <= 4096, "the rows of an asynchronous fold exceed the fixture's positions");
  const std::vector<int64_t> tokens = make_tokens(cfg, T);
  struct Outcome {
    std::vector<float> one, two;
    uint64_t async_folds = 0;
    std::string error;
  };
  std::vector<Outcome> ranks(2);
  ConstructBarrier barrier(2);
  std::vector<std::thread> workers;
  for (int r = 0; r < 2; ++r)
    workers.emplace_back([&, r] {
      bool arrived = false;
      auto arrive_once = [&] {
        if (arrived) return;
        arrived = true;
        barrier.arrive_and_wait();
      };
      Outcome& o = ranks[static_cast<size_t>(r)];
      try {
        BusBoundaryReducer reducer(*buses[static_cast<size_t>(r)], wait_timeout_ms());
        Dsv41Model model(cfg, dir, T, 4096, Dsv41Residency::Streaming, &reducer, r, 2);
        arrive_once();
        model.set_decode_route_traces(false);  // as the engine serves it (a traced walk stays one block)
        model.set_prefill_fold_overlap(false);
        o.one = model.forward(tokens).logits;
        model.set_prefill_fold_overlap(true, /*min_rows=*/256);
        o.two = model.forward(tokens).logits;
        o.async_folds = model.debug_overlap_async_folds();
      } catch (const std::exception& e) {
        o.error = "rank " + std::to_string(r) + ": " + e.what();
        arrive_once();
      }
    });
  for (auto& t : workers) t.join();
  for (int r = 0; r < 2; ++r) require(ranks[static_cast<size_t>(r)].error.empty(), ranks[static_cast<size_t>(r)].error);
  for (int r = 0; r < 2; ++r) {
    const Outcome& o = ranks[static_cast<size_t>(r)];
    require(!o.one.empty() && o.one == o.two, "fold overlap at world 2: rank " + std::to_string(r) +
                                                  "'s logits differ from its one-block walk's");
    require(o.async_folds > 0, "fold overlap at world 2: rank " + std::to_string(r) + " ran no fold asynchronously");
  }
  std::printf("[ OK ] world 2: %d rows as blocks of %d (%llu asynchronous bulk folds per rank): logits bitwise the "
              "one-block walk on both ranks\n", T, TA, static_cast<unsigned long long>(ranks[0].async_folds));
}

// The Engram chunk-ahead staging (engine.ngram_prestage, 2026-10-18): a
// 701-row prompt prefilled in 256-row chunks on the cursor path (the base
// names the next chunk at the same budget), every chunk after the first
// taking the rows the previous chunk's prestage gathered on the side
// stream; the logits and the steps behind it bitwise the same chunks
// staged in turn (the rows gathered are the same rows).
DGPP_TEST(dsv41_engram_prestage_is_bitwise_the_sequential_staging) {
  const std::string dir = (fs::current_path() / "dsv41_tp_fixture_prestage").string();
  const Dsv41TextConfig cfg = dsv41fx::tiny_config();
  dsv41fx::write_fixture(cfg, dir);
  (void)dsv41fx::fixture_sidecar(cfg, dir);
  std::vector<int64_t> prompt = make_tokens(cfg, 160);
  while (prompt.size() < 701) prompt.push_back((prompt[prompt.size() - 160] * 31 + 11) % cfg.vocab_size);
  const auto run = [&](bool prestage, bool group_api, std::vector<float>* logits, std::vector<std::vector<float>>* steps,
                       int* chunks, uint64_t* claims) {
    dgpp::Dsv41EngramLayer::set_prestage(prestage);
    Dsv41Model model(cfg, dir, 1280, 4096, Dsv41Residency::Streaming, nullptr, 0, 1, /*max_requests=*/2);
    model.set_decode_route_traces(false);
    Dsv41Model::PrefillCursor cursor = model.session_prefill_begin(0, prompt, static_cast<int64_t>(prompt.size()) + 8, 256);
    *chunks = 0;
    if (group_api) {  // the scheduler's path: one cursor through the batch walk
      std::vector<Dsv41Model::PrefillCursor*> open = {&cursor};
      while (!model.session_prefill_advance_group(open, {256})[0]) ++*chunks;
    } else {
      while (!model.session_prefill_advance(cursor)) ++*chunks;
    }
    ++*chunks;
    *logits = cursor.output.logits;
    for (int k = 0; k < 4; ++k) steps->push_back(model.session_step(0, (prompt[static_cast<size_t>(k)] + 7) % cfg.vocab_size).logits);
    *claims = model.debug_engram_prestage_claims();
    require(model.debug_engram_prestage_publishes() == (prestage ? uint64_t(*chunks - 1) : 0),
            "the prestage publishes once per chunk that has a next (or never, off)");
    model.session_close(0);
  };
  for (const bool group_api : {false, true}) {
    std::vector<float> on_logits, off_logits;
    std::vector<std::vector<float>> on_steps, off_steps;
    int on_chunks = 0, off_chunks = 0;
    uint64_t on_claims = 0, off_claims = 0;
    run(true, group_api, &on_logits, &on_steps, &on_chunks, &on_claims);
    run(false, group_api, &off_logits, &off_steps, &off_chunks, &off_claims);
    dgpp::Dsv41EngramLayer::set_prestage(true);
    require(on_chunks == 3 && off_chunks == 3, "701 rows at a 256-row budget walk as three chunks");
    require(on_claims == 2 && off_claims == 0, "every chunk after the first takes the prestaged rows");
    require(!on_logits.empty() && on_logits == off_logits, "prestage: the prefill's logits differ from the sequential staging's");
    require(on_steps == off_steps, "prestage: the decode steps behind the prefill differ (the state)");
    std::printf("[ OK ] 701 rows in three 256-row chunks (%s): 2 prestages claimed, logits and 4 decode steps bitwise "
                "the sequential staging\n", group_api ? "the scheduler's batch walk" : "the cursor advance");
  }
}

// The lm head's 12-bit companion (engine.bf16_weights, 2026-10-19): under
// bf12 | bf12+bf16 the model packs its head slice after the load and the
// decode head GEMV reads the companion. The packed values are the bf16
// values; the companion GEMV sums each row in another fp32 order, so the
// logits are the bf16 head's within rounding and the argmax is equal — the
// prefill's last row and four decode steps behind it, against the
// checkpoint residency; the companions-only residency packs the same.
DGPP_TEST(dsv41_lm_head_bf12_companion_matches_the_bf16_head) {
  const std::string dir = (fs::current_path() / "dsv41_tp_fixture_bf12").string();
  const Dsv41TextConfig cfg = dsv41fx::tiny_config();
  if (cfg.weights_exl3) {
    std::printf("[ .. ] the EXL3 pack's head is a trellis matrix (no bf16 bytes, no bf12 companion): nothing to pack\n");
    return;
  }
  dsv41fx::write_fixture(cfg, dir);
  (void)dsv41fx::fixture_sidecar(cfg, dir);
  const std::vector<int64_t> prompt = make_tokens(cfg, 160);
  struct Run {
    std::vector<float> logits;
    std::vector<std::vector<float>> steps;
    size_t matrices = 0, released = 0;
  };
  const auto run = [&](dgpp::Bf16Residency mode) {
    dgpp::set_bf16_residency(mode);
    Run r;
    Dsv41Model model(cfg, dir, 512, 2048, Dsv41Residency::Streaming, nullptr, 0, 1, /*max_requests=*/2);
    model.set_decode_route_traces(false);
    r.logits = model.session_prefill(0, prompt).logits;
    for (int k = 0; k < 4; ++k) r.steps.push_back(model.session_step(0, (prompt[static_cast<size_t>(k)] + 7) % cfg.vocab_size).logits);
    model.session_close(0);
    r.matrices = model.bf12_companions().matrices();
    r.released = model.bf12_companions().released();
    return r;
  };
  const Run plain = run(dgpp::Bf16Residency::Checkpoint);
  const Run both = run(dgpp::Bf16Residency::Bf12AndBf16);
  const Run only = run(dgpp::Bf16Residency::Bf12);
  dgpp::set_bf16_residency(dgpp::Bf16Residency::Checkpoint);
  require(plain.matrices == 0, "the checkpoint residency packs nothing");
  require(both.matrices == 1 && only.matrices == 1, "both companion residencies pack the lm head");
  const int V = cfg.vocab_size;
  const auto compare = [&](const std::vector<float>& a, const std::vector<float>& b, const char* what) {
    require(!a.empty() && a.size() == b.size(), std::string(what) + ": logits shape");
    const size_t rows = a.size() / static_cast<size_t>(V);
    const float* pa = a.data() + (rows - 1) * static_cast<size_t>(V);
    const float* pb = b.data() + (rows - 1) * static_cast<size_t>(V);
    int ia = 0, ib = 0;
    double d = 0, m = 0;
    for (int i = 0; i < V; ++i) {
      if (pa[i] > pa[ia]) ia = i;
      if (pb[i] > pb[ib]) ib = i;
      d += double(pa[i] - pb[i]) * double(pa[i] - pb[i]);
      m += double(pb[i]) * double(pb[i]);
    }
    const double l2 = std::sqrt(d) / std::max(std::sqrt(m), 1e-30);
    require(ia == ib, std::string(what) + ": the argmax differs from the bf16 head's");
    require(l2 < 0.02, std::string(what) + ": the logits beyond rounding of the bf16 head's");
    return l2;
  };
  double worst = 0;
  for (const Run* r : {&both, &only}) {
    worst = std::max(worst, compare(r->logits, plain.logits, "prefill last row"));
    for (int k = 0; k < 4; ++k) worst = std::max(worst, compare(r->steps[static_cast<size_t>(k)], plain.steps[static_cast<size_t>(k)], "decode step"));
  }
  std::printf("[ OK ] lm head bf12 companion: the prefill's last row and 4 decode steps within rel l2 %.2e of the bf16 head "
              "(argmax equal), both residencies packed, %zu released under the companions-only one\n", worst, only.released);
}

int main() { return dgpp::test::run_all(); }
