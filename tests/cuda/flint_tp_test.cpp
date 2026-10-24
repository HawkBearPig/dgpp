// The Flint TP forwards (2026-10-22, docs/flint_plan.md §2): loopback
// worlds of 2 and 4 ranks in one process (real verbs QPs over 127.0.0.1,
// the bus the fabric uses) over the tiny fixture, against the world-1
// model as the oracle. Asserted:
//   * cross-rank: every layer's residual, the final read, the routing
//     decisions and the lm-head slices' top-1 are bitwise identical on
//     every rank (the canonical rank-order fold; a divergence is the
//     silent-corruption class);
//   * vs world 1: per-layer residuals within the numerics budget (the
//     partial sums fold in bf16 on the wire), the final read likewise, the
//     top-1 per token equal or a certified near tie;
//   * the head geometry: 8 query heads over 4 kv heads slice to 4 heads
//     with 2 kv heads (world 2) and 2 heads with 1 kv head (world 4), the
//     fused [Q | K | V] projection re-stacked per rank; the experts'
//     512-wide intermediate slices to 256 and 128 (the fp8 scale block;
//     the NVFP4 experts' 16-groups with it), the dense MLP's 512 to 256
//     and 128, the shared expert's 256 to 128 and 64;
//   * 72 tokens: past the 33-token window, so the sliding rows read their
//     window through the paged cache at every world.
#include <algorithm>
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
#include "common/test.hpp"
#include "engine/tp_bus.hpp"
#include "models/flint/config.hpp"
#include "models/flint/forward.hpp"
#include "net/collective_bus.hpp"
#include "flint_fixture.hpp"

namespace fs = std::filesystem;
using dgpp::bf16_bits_to_float;
using dgpp::BusBoundaryReducer;
using dgpp::FlintLocalGeometry;
using dgpp::FlintModel;
using dgpp::FlintResidency;
using dgpp::FlintTextConfig;
using dgpp::net::BusOptions;
using dgpp::net::CollectiveBus;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// A numerics budget: DGPP_TEST_REPORT_ONLY=1 reports a miss and carries on
// (the whole picture of both worlds while a budget is being set).
void budget(bool cond, const std::string& what) {
  if (cond) return;
  if (std::getenv("DGPP_TEST_REPORT_ONLY")) {
    std::printf("[MISS] %s\n", what.c_str());
    return;
  }
  throw std::runtime_error(what);
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
  // A 256 KB latency slot: every fold of this test (72 rows x 256 bf16)
  // rides the chunked latency path (glm4_tp_test's note on the bulk
  // machine's stripe count at world 4).
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
    if (!out[0]->start(&errors[0])) DGPP_LOG_ERROR("flint tp world rank 0: {}", errors[0]);
  });
  std::vector<std::thread> connectors;
  for (int r = 1; r < world; ++r)
    connectors.emplace_back([&, r] {
      if (!out[static_cast<size_t>(r)]->start(&errors[static_cast<size_t>(r)]))
        DGPP_LOG_ERROR("flint tp world rank {}: {}", r, errors[static_cast<size_t>(r)]);
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
  FlintModel::Outputs out;
};

void rank_work(int rank, int world, const std::string& dir, const FlintTextConfig& cfg,
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
    FlintModel model(cfg, dir, static_cast<int>(tokens.size()), 256, FlintResidency::Streaming, &reducer,
                    rank, world);
    arrive_once();
    out->out = model.forward(tokens, true);
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("flint tp rank {} failed: {}", rank, e.what());
    out->error = "rank " + std::to_string(rank) + ": " + e.what();
    arrive_once();
  }
}

std::vector<int64_t> make_tokens(const FlintTextConfig& cfg, int n) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = 0xA5A5F00DBEEF1234ull;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    t[static_cast<size_t>(i)] = static_cast<int64_t>(s % static_cast<uint64_t>(cfg.vocab_size));
  }
  if (n > 9) t[9] = cfg.eos_token_ids.empty() ? 0 : cfg.eos_token_ids[0];
  return t;
}

bool bits_equal(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * 2) == 0);
}

struct Stats {
  double l2 = 0, max_ulps = 0;
  long hard = 0, total = 0;
  std::string worst;  // the hard elements (index, want, got), the first few
};

int bf16_ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) -> int32_t {
    return (v & 0x8000u) ? -static_cast<int32_t>(v & 0x7FFFu) : static_cast<int32_t>(v & 0x7FFFu);
  };
  return std::abs(static_cast<int>(key(a) - key(b)));
}

// An element is hard when it is more than 128 bf16 ulps (a binade) AND
// more than 3 % of the tensor's rms from the reference: a fold's partials
// round to bf16 on the wire (2^-8 relative each), so an element that
// cancels between two partials of 3 sigma carries ~2.3 % of the rms as
// legitimate reassociation error (world 2 layer 0's element [3147] on this
// fixture: 0.003 against -0.007 under an rms of 0.46 — 2.1 %). GLM-4.7's
// gate holds 2 %; its shared expert damps the folds' partials.
constexpr double kHardRms = 0.03;

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
    if (std::fabs(g - w) <= kHardRms * rms) u = 0;
    s.max_ulps = std::max(s.max_ulps, static_cast<double>(u));
    if (u > 128) {
      ++s.hard;
      if (s.hard <= 4) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), " [%zu] want %.4g got %.4g (rms %.3g)", i, w, g, rms);
        s.worst += buf;
      }
    }
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

// The MoE layer's ordinal among the MoE layers (the route captures' slot).
int moe_ordinal(const FlintTextConfig& cfg, int layer) {
  int n = 0;
  for (int l = 0; l < layer; ++l) n += cfg.is_moe_layer(l);
  return n;
}

// The rank's slices at every world: the heads, the intermediate slices —
// the loader's geometry against the plan's formulas.
void check_geometry(const FlintTextConfig& cfg, int world) {
  for (int r = 0; r < world; ++r) {
    const FlintLocalGeometry g = FlintLocalGeometry::from_config(cfg, r, world, dgpp::FlintHeadSharding::VocabSharded);
    require(g.local_heads == cfg.num_attention_heads / world && g.head_begin == r * g.local_heads, "geometry: the query heads");
    require(g.local_kv_heads == cfg.num_key_value_heads / world && g.kv_head_begin == r * g.local_kv_heads,
            "geometry: the kv heads");
    require(g.local_inter == cfg.moe_intermediate_size / world && g.local_inter % 128 == 0, "geometry: the expert slice");
    require(g.local_dense_inter == cfg.intermediate_size / world && g.local_dense_inter % 8 == 0, "geometry: the dense slice");
    require(g.local_shared_inter == cfg.shared_expert_intermediate_size / world && g.local_shared_inter % 8 == 0,
            "geometry: the shared slice");
    // Every rank's query heads sit on its own kv heads (GQA groups never
    // straddle a rank).
    require(g.head_begin / (cfg.num_attention_heads / cfg.num_key_value_heads) == g.kv_head_begin,
            "geometry: the query heads straddle kv heads");
  }
  const FlintLocalGeometry g0 = FlintLocalGeometry::from_config(cfg, 0, world, dgpp::FlintHeadSharding::VocabSharded);
  std::printf("[ .. ] world %d geometry: %d query heads, %d kv heads, expert slice %lld, dense slice %lld, shared slice %lld\n",
              world, g0.local_heads, g0.local_kv_heads, static_cast<long long>(g0.local_inter),
              static_cast<long long>(g0.local_dense_inter), static_cast<long long>(g0.local_shared_inter));
}

void check_world(int world, uint16_t port, const std::string& dir, const FlintTextConfig& cfg,
                 const std::vector<int64_t>& tokens, const FlintModel::Outputs& ref) {
  DGPP_LOG_INFO("flint TP loopback world={} tokens={}", world, tokens.size());
  check_geometry(cfg, world);
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
  // ---- cross-rank bitwise ------------------------------------------------------
  for (int r = 1; r < world; ++r) {
    const FlintModel::Outputs& a = ranks[0].out;
    const FlintModel::Outputs& b = ranks[static_cast<size_t>(r)].out;
    require(bits_equal(a.final_hidden_bits, b.final_hidden_bits), "final hidden differs across ranks");
    require(a.layer_states.size() == static_cast<size_t>(L) && b.layer_states.size() == static_cast<size_t>(L),
            "layer captures missing");
    for (int l = 0; l < L; ++l)
      require(bits_equal(a.layer_states[static_cast<size_t>(l)], b.layer_states[static_cast<size_t>(l)]),
              "layer " + std::to_string(l) + " residual differs across ranks");
    require(a.route_ids == b.route_ids, "routing differs across ranks");
  }
  // ---- vs world 1 ---------------------------------------------------------------
  // The folds reassociate (bf16 on the wire), so every element differs at
  // rounding level; a routing flip on a near tie (the replicated router
  // reads a slightly different input) sends that ROW to different experts
  // — on this family's 2-of-8 routing without a shared expert a large
  // divergence (half the row's norm) confined to the flipped row and, from
  // the next layer on, carried to the rows that attend to it (its K/V
  // moved: a 1/32 share of a window row's attention). So the budget is
  // taken over the CLEAN rows — those before the first flipped row at
  // this and every earlier MoE layer (a flipped row's K/V moves every
  // later row's attention, so the rows after it are its shadow, not a
  // measurement of the fold) — as an aggregate l2, a per-row worst, and a
  // bound on the hard elements as a share of the clean set: the
  // silent-corruption class (a wrong slice, a garbage expert, a
  // mis-stacked projection) fails whole rows or heads — thousands of hard
  // elements — on every row, the first ones included.
  const FlintModel::Outputs& o = ranks[0].out;
  const int H = cfg.hidden_size;
  const int K = cfg.num_experts_per_tok;
  require(o.route_ids.size() == static_cast<size_t>(cfg.num_moe_layers()) && ref.route_ids.size() == o.route_ids.size(),
          "route captures");
  std::vector<bool> flipped(static_cast<size_t>(T), false);
  long flips = 0, slots = 0;
  // The clean rows' aggregate: relative l2 over them, their hard elements,
  // the worst row.
  struct Clean {
    double l2 = 0, worst_row = 0;
    long hard = 0, elems = 0;
    int worst_t = -1, rows = 0;
  };
  const auto clean_stats = [&](const std::vector<uint16_t>& got, const std::vector<uint16_t>& want) {
    Clean c;
    double d2 = 0, w2 = 0;
    for (int t = 0; t < T; ++t) {
      if (flipped[static_cast<size_t>(t)]) continue;
      const std::vector<uint16_t> g(got.begin() + static_cast<size_t>(t) * H, got.begin() + static_cast<size_t>(t + 1) * H);
      const std::vector<uint16_t> w(want.begin() + static_cast<size_t>(t) * H, want.begin() + static_cast<size_t>(t + 1) * H);
      const Stats r = compare(g, w);
      c.hard += r.hard;
      c.elems += r.total;
      ++c.rows;
      if (r.l2 > c.worst_row) {
        c.worst_row = r.l2;
        c.worst_t = t;
      }
      for (int i = 0; i < H; ++i) {
        const double x = bf16_bits_to_float(g[static_cast<size_t>(i)]), y = bf16_bits_to_float(w[static_cast<size_t>(i)]);
        d2 += (x - y) * (x - y);
        w2 += y * y;
      }
    }
    c.l2 = std::sqrt(d2) / std::sqrt(w2 + 1e-30);
    return c;
  };
  const auto within = [](const Clean& c) {
    return c.l2 < 0.02 && c.worst_row < 0.06 && c.hard * 200 <= c.elems;  // hard elements: 0.5 % of the clean set
  };
  int judged = 0, judged_moe = 0;
  for (int l = 0; l < L; ++l) {
    if (cfg.is_moe_layer(l)) {
      const size_t m = static_cast<size_t>(moe_ordinal(cfg, l));
      for (int t = 0; t < T; ++t)
        for (int i = 0; i < K; ++i, ++slots)
          if (o.route_ids[m][static_cast<size_t>(t) * K + i] != ref.route_ids[m][static_cast<size_t>(t) * K + i]) {
            ++flips;
            if (!flipped[static_cast<size_t>(t)]) std::printf("[ .. ] world %d layer %d: routing flip on row %d\n", world, l, t);
            // The flip's shadow: every later row attends to the flipped
            // row (the full layers see every earlier row; the sliding
            // ones their window), so from this layer on the clean set is
            // the rows BEFORE the first flip — those untouched by any
            // flip, which is what the budget measures.
            for (int u = t; u < T; ++u) flipped[static_cast<size_t>(u)] = true;
          }
    }
    const Stats s = compare(o.layer_states[static_cast<size_t>(l)], ref.layer_states[static_cast<size_t>(l)]);
    const Clean c = clean_stats(o.layer_states[static_cast<size_t>(l)], ref.layer_states[static_cast<size_t>(l)]);
    std::printf("[ .. ] world %d layer %d (%s %s): l2 %.3g max %g ulps hard %ld of %ld; clean rows %d: l2 %.3g hard %ld of %ld "
                "worst t%d %.3g%s\n",
                world, l, cfg.is_swa_layer(l) ? "swa" : "full", cfg.is_moe_layer(l) ? "moe" : "dense", s.l2, s.max_ulps,
                s.hard, s.total, c.rows, c.l2, c.hard, c.elems, c.worst_t, c.worst_row, c.hard ? s.worst.c_str() : "");
    // The budget judges a layer while at least 16 rows stand before the
    // first flip; on this fixture the random router's near ties flip
    // early (every bf16 site reassociates across the fold), so the later
    // layers are reported. The first MoE layers must be judged: the fold
    // of the routed slices and the shared expert's.
    if (c.rows >= 16) {
      budget(within(c), "world " + std::to_string(world) + " layer " + std::to_string(l) + " outside the numerics budget");
      ++judged;
      if (cfg.is_moe_layer(l)) ++judged_moe;
    } else {
      std::printf("[ .. ] world %d layer %d: %d clean rows — reported, not judged\n", world, l, c.rows);
    }
  }
  budget(judged_moe >= 2, "world " + std::to_string(world) + ": fewer than two MoE layers judged on 16+ clean rows");
  {
    const Stats f = compare(o.final_hidden_bits, ref.final_hidden_bits);
    const Clean c = clean_stats(o.final_hidden_bits, ref.final_hidden_bits);
    std::printf("[ .. ] world %d final hidden: l2 %.3g max %g ulps hard %ld of %ld; clean rows %d: l2 %.3g hard %ld of %ld worst t%d %.3g\n",
                world, f.l2, f.max_ulps, f.hard, f.total, c.rows, c.l2, c.hard, c.elems, c.worst_t, c.worst_row);
    if (c.rows >= 16) budget(within(c), "world " + std::to_string(world) + " final hidden outside the budget");
    else std::printf("[ .. ] world %d final hidden: %d clean rows — reported, not judged\n", world, c.rows);
  }
  std::printf("[ .. ] world %d routing: %ld of %ld slots differ from world 1\n", world, flips, slots);
  budget(flips <= slots / 20, "too many routing flips against world 1");
  // The merged top-1 vs world 1's: equal, or a near tie in world 1's logits.
  const auto merged = merged_top1(ranks, T);
  const std::vector<std::pair<int32_t, float>> single = merged_top1({RankOutcome{"", ref}}, T);
  int mism = 0;
  for (int t = 0; t < T; ++t) {
    if (flipped[static_cast<size_t>(t)]) continue;  // its logits legitimately differ
    if (merged[static_cast<size_t>(t)].first == single[static_cast<size_t>(t)].first) continue;
    // world 1's logit for the TP top-1 must be within 2 % of its own top-1.
    const float v_tp = ref.logits[static_cast<size_t>(t) * ref.lm_vocab_count + merged[static_cast<size_t>(t)].first];
    const float v_1 = single[static_cast<size_t>(t)].second;
    if (std::fabs(v_1 - v_tp) > 0.02 * std::fabs(v_1)) ++mism;
  }
  std::printf("[ .. ] world %d top-1: %d of %d rows beyond a near tie (the flipped rows exempt)\n", world, mism, T);
  budget(mism == 0, "world " + std::to_string(world) + " top-1 differs beyond a near tie");
}

}  // namespace

DGPP_TEST(flint_tp_loopback_worlds_2_and_4_match_world_1) {
  const std::string dir = (fs::current_path() / "flint_tp_fixture").string();
  const FlintTextConfig cfg = flintfx::tiny_config();
  flintfx::write_fixture(cfg, dir);
  const std::vector<int64_t> tokens = make_tokens(cfg, 72);
  require(static_cast<int>(tokens.size()) > cfg.sliding_window, "the prompt reaches past the window");
  FlintModel::Outputs ref;
  {
    FlintModel single(cfg, dir, static_cast<int>(tokens.size()), 256);
    ref = single.forward(tokens, true);
  }
  check_world(2, 29964, dir, cfg, tokens, ref);
  check_world(4, 29965, dir, cfg, tokens, ref);
}

int main() { return dgpp::test::run_all(); }
