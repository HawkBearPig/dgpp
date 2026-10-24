// The Flint engines (2026-10-22, docs/flint_plan.md G6):
// EagerEngineAdapter<FlintModel> and GraphEngineAdapter<FlintModel> on the
// fixture, a loopback world of 2 (real verbs QPs over 127.0.0.1) against
// the world-1 eager engine — the graph engine's scalar and row-batched
// replays bitwise the eager engine's transcripts at the same world, the
// world-2 eager transcripts against world 1's under the near-tie rule
// (engine_test_ties.hpp). No draft head: the MTP engines have no Flint
// form (the family refuses engine.mtp).
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
#include "engine/eager_engine.hpp"
#include "engine/verify_schedule.hpp"
#include "engine/graph_engine.hpp"
#include "engine_test_ties.hpp"
#include "engine/tp_bus.hpp"

namespace fs = std::filesystem;
using dgpp::BusBoundaryReducer;
using dgpp::EagerEngineAdapter;
using dgpp::GraphEngineAdapter;
using dgpp::FlintModel;
using dgpp::FlintResidency;
using dgpp::FlintTextConfig;
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
  o.lat_slot_bytes = 262144;  // every fold on the latency path (mimo_tp_test's note)
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
    if (!out[0]->start(&errors[0])) DGPP_LOG_ERROR("flint engine world rank 0: {}", errors[0]);
  });
  std::vector<std::thread> connectors;
  for (int r = 1; r < world; ++r)
    connectors.emplace_back([&, r] {
      if (!out[static_cast<size_t>(r)]->start(&errors[static_cast<size_t>(r)]))
        DGPP_LOG_ERROR("flint engine world rank {}: {}", r, errors[static_cast<size_t>(r)]);
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

std::vector<int64_t> smoke_tokens(const FlintTextConfig& cfg, int n, uint64_t seed) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = seed;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    t[static_cast<size_t>(i)] = static_cast<int64_t>(s % static_cast<uint64_t>(cfg.vocab_size));
  }
  return t;
}

std::string ids_text(const std::vector<int32_t>& ids) {
  std::string s;
  for (size_t i = 0; i < ids.size(); ++i) s += (i ? "," : "") + std::to_string(ids[i]);
  return s;
}

// The engine's greedy transcript for one prompt in slot `req`: the
// prefill's pick then `steps` scalar steps.
template <class Engine>
std::vector<int32_t> solo(Engine& eng, int req, const std::vector<int64_t>& prompt, int steps) {
  std::vector<int32_t> out;
  out.push_back(eng.prefill(req, prompt));
  eng.reserve(req, static_cast<int64_t>(prompt.size()) + steps + 1);
  for (int s = 0; s < steps; ++s) {
    const std::vector<int32_t> t = eng.step(req);
    require(t.size() == 1, "a scalar step decides one token");
    out.push_back(t[0]);
  }
  eng.close(req);
  return out;
}

size_t agreeing_prefix(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
  size_t n = 0;
  while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
  return n;
}

constexpr int kSlots = 3;
constexpr int kSteps = 60;
constexpr int kMaxTokens = 64;
constexpr int64_t kCache = 512;
constexpr int kWorld = 2;
constexpr uint16_t kPort = 29969;


// The batched engines against the eager one at the same world (2026-09-14):
// the dense sites' lowering follows the rows of a launch (kernels/gemm.hpp
// dense_gemv_rows — the GEMV chunks to four rows, cuBLASLt's algorithm or
// the streaming tensor-core form above), so a batched step's rows and the
// eager scalar's are tolerance-equal, not bitwise; on this fixture a near
// tie flips and every later token follows. The rule (engine_test_ties.hpp):
// the first kAgreePositions decisions agree, or the first difference among
// them sits on a near tie of the world-1 reference's own decision (its
// top-2 margin under kTieMargin); later differences are reported.
constexpr float kTieMargin = 0.1f;
constexpr size_t kAgreePositions = 3;

struct Ref {
  std::vector<int32_t> a, b, c;
  std::vector<float> am, bm, cm;  // the world-1 reference's top-2 margin at every decision
};

Ref world1_reference(const FlintTextConfig& cfg, const std::string& dir, const std::vector<int64_t>& A,
                     const std::vector<int64_t>& B, const std::vector<int64_t>& C) {
  FlintModel m(cfg, dir, kMaxTokens, kCache, FlintResidency::Resident, nullptr, 0, 1, kSlots);
  std::vector<float> margins;
  EagerEngineAdapter<FlintModel> eng(&m, kSlots, engine_ties::margin_pick(cfg.vocab_size, &margins));
  Ref r;
  r.a = solo(eng, 0, A, kSteps);
  r.am = margins; margins.clear();
  r.b = solo(eng, 1, B, kSteps);
  r.bm = margins; margins.clear();
  r.c = solo(eng, 2, C, kSteps);
  r.cm = margins;
  return r;
}

struct RankOutcome {
  std::string error;
  std::vector<int32_t> ea, eb, ec;   // the eager engine, world 2
  std::vector<int32_t> ga;           // the graph engine, scalar
  std::vector<int32_t> ba, bb, bc;   // the graph engine, batched
};

void rank_work(int r, const FlintTextConfig& cfg, const std::string& dir, const std::vector<int64_t>& A,
               const std::vector<int64_t>& B, const std::vector<int64_t>& C, CollectiveBus* bus,
               ConstructBarrier* barrier, RankOutcome* out) {
  bool arrived = false;
  const auto arrive_once = [&] {
    if (arrived) return;
    arrived = true;
    barrier->arrive_and_wait();
  };
  uint16_t* scratch = nullptr;
  try {
    BusBoundaryReducer reducer(*bus, wait_timeout_ms());
    FlintModel eager(cfg, dir, kMaxTokens, kCache, FlintResidency::Resident, &reducer, r, kWorld, kSlots);
    FlintModel graph(
        cfg, dir, kMaxTokens, kCache, FlintResidency::Resident, &reducer, r, kWorld, kSlots,
        /*mtp=*/false, /*decode_rows=*/0, dgpp::LatentFormat::kBf16, /*serving_logits=*/true);
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                               sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld), cudaHostAllocDefault));
    arrive_once();
    EagerEngineAdapter<FlintModel> eager_engine(
        &eager, kSlots, dgpp::make_fabric_pick(bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms()));
    GraphEngineAdapter<FlintModel> graph_engine(&graph, bus, r, kWorld, scratch, cfg.vocab_size,
                                               wait_timeout_ms(), /*batch_min_live=*/2);

    // 1. The eager engine at world 2.
    out->ea = solo(eager_engine, 0, A, kSteps);
    out->eb = solo(eager_engine, 1, B, kSteps);
    out->ec = solo(eager_engine, 2, C, kSteps);

    // 2. The graph engine, scalar replays (one live slot).
    out->ga = solo(graph_engine, 1, A, kSteps);
    graph_engine.drain();

    // 3. The row-batched replays: A and B live (the 2-slot family), then
    //    C joins (the full family), then B closes (slot 1 pads).
    out->ba.push_back(graph_engine.prefill(0, A));
    graph_engine.reserve(0, static_cast<int64_t>(A.size()) + kSteps + 1);
    out->bb.push_back(graph_engine.prefill(1, B));
    graph_engine.reserve(1, static_cast<int64_t>(B.size()) + kSteps + 1);
    for (int s = 0; s < 25; ++s) {
      const auto t = graph_engine.step_batch({0, 1});
      require(t.size() == 2 && t[0].size() == 1 && t[1].size() == 1, "batch step shape");
      out->ba.push_back(t[0][0]);
      out->bb.push_back(t[1][0]);
    }
    graph_engine.drain();
    out->bc.push_back(graph_engine.prefill(2, C));
    graph_engine.reserve(2, static_cast<int64_t>(C.size()) + kSteps + 1);
    for (int s = 0; s < 20; ++s) {
      const auto t = graph_engine.step_batch({0, 1, 2});
      require(t.size() == 3, "batch step shape");
      out->ba.push_back(t[0][0]);
      out->bb.push_back(t[1][0]);
      out->bc.push_back(t[2][0]);
    }
    graph_engine.close(1);
    for (int s = 0; s < 15; ++s) {
      const auto t = graph_engine.step_batch({0, 2});
      require(t.size() == 2, "batch step shape");
      out->ba.push_back(t[0][0]);
      out->bc.push_back(t[1][0]);
    }
    graph_engine.close(0);
    graph_engine.close(2);
    graph_engine.drain();
    cudaFreeHost(scratch);
  } catch (const std::exception& e) {
    if (scratch) cudaFreeHost(scratch);
    out->error = "rank " + std::to_string(r) + ": " + e.what();
    arrive_once();
  }
}

}  // namespace

DGPP_TEST(flint_model_rejects_decode_rows_above_family_cap) {
  const auto cfg = flintfx::tiny_config();
  const std::string dir = "flint_engine_cap_fixture";
  flintfx::write_fixture(cfg, dir);
  require(FlintModel::decode_rows_cap() == 32, "Flint retains its family cap");
  for (const int rows : {33, 64}) {
    for (const bool construct : {false, true}) {
      bool rejected = false;
      try {
        if (construct) {
          FlintModel model(cfg, dir, 128, 2048, FlintResidency::Resident, nullptr, 0, 1, 16, false,
                          rows);
        } else {
          (void)FlintModel::plan_memory(cfg, 128, 2048, 0, 1, FlintResidency::Resident, 16, false,
                                       rows);
        }
      } catch (const std::invalid_argument& e) {
        const std::string message = e.what();
        rejected = message.find("decode_rows") != std::string::npos &&
                   message.find("32") != std::string::npos;
      }
      require(rejected, "Flint rejects oversized decode_rows at its model boundary");
    }
  }
}

DGPP_TEST(flint_engines_loopback_world_2_graph_matches_eager) {
  const FlintTextConfig cfg = flintfx::tiny_config();
  const std::string dir = "flint_engine_fixture";
  flintfx::write_fixture(cfg, dir);
  const std::vector<int64_t> A = smoke_tokens(cfg, 23, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 17, 0xD1B54A32D192ED03ull);
  const std::vector<int64_t> C = smoke_tokens(cfg, 11, 0x2545F4914F6CDD1Dull);
  const Ref ref = world1_reference(cfg, dir, A, B, C);
  DGPP_LOG_INFO("world 1 eager: A {} | B {} | C {}", ids_text(ref.a), ids_text(ref.b), ids_text(ref.c));

  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, kPort);
  require(!buses.empty(), "the loopback bus world failed to start");
  std::vector<RankOutcome> outs(kWorld);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int r = 0; r < kWorld; ++r)
    workers.emplace_back(rank_work, r, std::cref(cfg), std::cref(dir), std::cref(A), std::cref(B), std::cref(C),
                         buses[static_cast<size_t>(r)].get(), &barrier, &outs[static_cast<size_t>(r)]);
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r) require(outs[static_cast<size_t>(r)].error.empty(), outs[static_cast<size_t>(r)].error);
  for (int r = 1; r < kWorld; ++r) {
    require(outs[static_cast<size_t>(r)].ea == outs[0].ea && outs[static_cast<size_t>(r)].ga == outs[0].ga &&
                outs[static_cast<size_t>(r)].ba == outs[0].ba,
            "the ranks' transcripts differ");
  }
  const RankOutcome& o = outs[0];
  DGPP_LOG_INFO("world 2 eager: A {} | B {} | C {}", ids_text(o.ea), ids_text(o.eb), ids_text(o.ec));
  DGPP_LOG_INFO("world 2 graph scalar A {}", ids_text(o.ga));
  DGPP_LOG_INFO("world 2 graph batched: A {} | B {} | C {}", ids_text(o.ba), ids_text(o.bb), ids_text(o.bc));
  // The world-2 eager engine against world 1: reported, the first tokens
  // must agree or differ on a near tie of world 1's own decision (the
  // folds reassociate; on this family's 2-of-8 routing without a shared
  // expert a near-tie routing flip moves a row's logits by half their
  // norm, so a flipped early token is the world-1 near tie the rule admits
  // — engine_test_ties.hpp).
  const size_t pa = agreeing_prefix(o.ea, ref.a), pb = agreeing_prefix(o.eb, ref.b), pc = agreeing_prefix(o.ec, ref.c);
  DGPP_LOG_INFO("world 2 vs world 1 agreeing prefixes: A {}/{} B {}/{} C {}/{}", pa, ref.a.size(), pb, ref.b.size(),
                pc, ref.c.size());
  engine_ties::require_agrees_or_tie(o.ea, ref.a, ref.am, kTieMargin, kAgreePositions, "the world-2 eager transcript of A");
  engine_ties::require_agrees_or_tie(o.eb, ref.b, ref.bm, kTieMargin, kAgreePositions, "the world-2 eager transcript of B");
  engine_ties::require_agrees_or_tie(o.ec, ref.c, ref.cm, kTieMargin, kAgreePositions, "the world-2 eager transcript of C");
  // The graph engine against the eager engine at the same world: exact.
  require(o.ga == o.ea, "the scalar graph transcript differs from the eager engine's");
  // 61 tokens for A (23 + 60: the batched rows cross the pool's 64-token
  // block), 46 for B, 36 for C — each a prefix of the eager transcript.
  require(o.ba == o.ea, "the batched graph transcript of A differs from the eager engine's");
  require(std::equal(o.bb.begin(), o.bb.end(), o.eb.begin()), "the batched graph transcript of B differs");
  require(std::equal(o.bc.begin(), o.bc.end(), o.ec.begin()), "the batched graph transcript of C differs");
  require(o.bb.size() == 46 && o.bc.size() == 36, "the batched transcripts' lengths");
}

int main() { return dgpp::test::run_all(); }
