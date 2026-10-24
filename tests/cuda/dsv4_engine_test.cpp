// The DeepSeek-V4-Flash decode engines over the engine core:
// EagerEngineAdapter<Dsv4Model> and GraphEngineAdapter<Dsv4Model> on the
// fixture, a loopback world of 2 (real verbs QPs over 127.0.0.1) against
// the world-1 eager engine.
//
// Gates: the graph engine's SCALAR replays (the device-driven T=1 graph:
// device positions, the recorded commit, the pinned token upload — the
// hash layers' routes read the same device tokens) and its ROW-BATCHED
// replays (the fixed slot-major batch off the persistent feeds, closed
// slots padding at -1, the 2-slot and full families; the attention's
// rings and compressors per slot) produce the eager engine's transcripts
// at the same world exactly (the recorded kernels are the eager kernels);
// the world-2 eager transcripts follow the world-1 ones (reported; the
// folds reassociate, so a near tie may flip a late token — the first
// tokens must agree). The prompts are sized to the compressors: A crosses
// a 128-token group during its decode steps (a ratio-128 entry published
// by a decode row) and holds more ratio-4 entries than the fixture's
// top-16 selection.
// Graph models use the compact serving allocation; eager references retain
// full heads. Cache, batching and speculative contracts keep their strict gates.
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
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
#include "dsv4_fixture.hpp"
#include "engine/eager_engine.hpp"
#include "engine/verify_schedule.hpp"
#include "engine/graph_engine.hpp"
#include "engine/tp_bus.hpp"
#include "models/dsv4/config.hpp"
#include "models/dsv4/model.hpp"
#include "net/collective_bus.hpp"
#include "serve/prefill_policy.hpp"

namespace fs = std::filesystem;
using dgpp::BusBoundaryReducer;
using dgpp::Dsv4Model;
using dgpp::Dsv4Residency;
using dgpp::Dsv4Config;
using dgpp::EagerEngineAdapter;
using dgpp::GraphEngineAdapter;
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
  o.lat_slot_bytes = 262144;  // every fold on the latency path (qwen_tp_test's note)
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
    if (!out[0]->start(&errors[0])) DGPP_LOG_ERROR("dsv4 engine world rank 0: {}", errors[0]);
  });
  std::vector<std::thread> connectors;
  for (int r = 1; r < world; ++r)
    connectors.emplace_back([&, r] {
      if (!out[static_cast<size_t>(r)]->start(&errors[static_cast<size_t>(r)]))
        DGPP_LOG_ERROR("dsv4 engine world rank {}: {}", r, errors[static_cast<size_t>(r)]);
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

std::vector<int64_t> smoke_tokens(const Dsv4Config& cfg, int n, uint64_t seed) {
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
constexpr int kSteps = 12;
constexpr int kMaxTokens = 256;
constexpr int64_t kCache = 2048;
constexpr int kWorld = 2;
constexpr uint16_t kPort = 29986;
// A world-1 top-2 logit margin under this is a near tie for the world-2
// engine (see Ref::am). Calibration: the folds reassociate in bf16 on the
// wire and the random-weight fixture amplifies the difference ~1.5x per
// layer (dsv4_tp_test: 6e-4 at layer 0, 1.3e-2 at layer 7, the fp4 cache
// flipping codes on the way), so world 2's logits sit ~1e-2 from world
// 1's; the early flips read on the fixture had world-1 margins 7.5e-3,
// 8.5e-3 and 5.2e-2 while the unflipped early positions run 1e-1..1e+0.
// kAgreePositions: the decisions held to it (the prefill's pick and the
// first scalar steps: one fold round each, no amplified history yet).
constexpr float kTieMargin = 0.1f;
constexpr size_t kAgreePositions = 3;

struct Ref {
  std::vector<int32_t> a, b, c;
  // The world-1 engine's top-2 logit margin at every decision (read by the
  // pick closure on the decode's own logits): the evidence a token world 2
  // flipped was a near tie.
  std::vector<float> am, bm, cm;
};

// The world-1 pick that also records the decision's top-2 margin.
dgpp::DecodePick margin_pick(int64_t vocab, std::vector<float>* margins) {
  return [vocab, margins](const dgpp::DecodeOutputs& out) -> int32_t {
    const int n = static_cast<int>(out.lm_vocab_count);
    int best = -1;
    float top = -INFINITY, second = -INFINITY;
    for (int v = 0; v < n; ++v) {
      const float x = out.logits[static_cast<size_t>(v)];
      if (x > top) { second = top; top = x; best = v; }
      else if (x > second) second = x;
    }
    if (best < 0 || best >= vocab) throw std::runtime_error("w1 pick out of range");
    margins->push_back(top - second);
    return best;
  };
}

// World 2's transcript against world 1's: the first kAgreePositions
// decisions agree, or the first difference among them sits on a
// demonstrated near tie of world 1's own decision (the folds reassociate
// in bf16 on the wire, so world 2's logits sit ~1e-2 from world 1's on
// this fixture). Past those positions the transcripts are reported only:
// the random-weight network amplifies the fold noise ~1.5x per layer and
// a routing or selection flip moves a later decision by whole logits
// (dsv4_tp_test gates the tensors layer-locally for that reason). A
// defect reads as a flip inside the first positions at a margin well
// above the bound (a wrong slice moves the logits tenfold).
void require_agrees_or_tie(const std::vector<int32_t>& w2, const std::vector<int32_t>& w1,
                           const std::vector<float>& margins, float tie, size_t positions, const char* what) {
  const size_t p = agreeing_prefix(w2, w1);
  if (p == w1.size()) return;
  require(p < margins.size(), what);
  DGPP_LOG_INFO("{}: world 2 differs from world 1 at position {} (world-1 margin {:.3e}, tie bound {:.1e} over the first {})",
                what, p, margins[p], tie, positions);
  if (p < positions)
    require(margins[p] < tie, std::string(what) + ": the world-2 engine flips an early token that is not a near tie");
}

Ref world1_reference(const Dsv4Config& cfg, const std::string& dir, const std::vector<int64_t>& A,
                     const std::vector<int64_t>& B, const std::vector<int64_t>& C) {
  Dsv4Model m(cfg, dir, kMaxTokens, kCache, Dsv4Residency::Resident, nullptr, 0, 1, kSlots);
  std::vector<float> margins;
  EagerEngineAdapter<Dsv4Model> eng(&m, kSlots, margin_pick(cfg.vocab_size, &margins));
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
  std::vector<int32_t> pa, pb, pc;  // the group prefill's transcripts (phase 4)
  std::vector<int32_t> ea, eb, ec;   // the eager engine, world 2
  std::vector<int32_t> ga;           // the graph engine, scalar
  std::vector<int32_t> ba, bb, bc;   // the graph engine, batched
};

void rank_work(int r, const Dsv4Config& cfg, const std::string& dir, const std::vector<int64_t>& A,
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
    Dsv4Model eager(cfg, dir, kMaxTokens, kCache, Dsv4Residency::Resident, &reducer, r, kWorld, kSlots);
    Dsv4Model graph(cfg, dir, kMaxTokens, kCache, Dsv4Residency::Resident, &reducer, r, kWorld,
                     kSlots,
                     /*mtp=*/false, /*decode_rows=*/0, /*serving_logits=*/true);
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                               sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld), cudaHostAllocDefault));
    arrive_once();
    EagerEngineAdapter<Dsv4Model> eager_engine(
        &eager, kSlots, dgpp::make_fabric_pick(bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms()));
    require(eager_engine.prefix_info().body_snapshots, "the prefill can snapshot existing cuts");
    GraphEngineAdapter<Dsv4Model> graph_engine(&graph, bus, r, kWorld, scratch, cfg.vocab_size,
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
    for (int s = 0; s < 5; ++s) {
      const auto t = graph_engine.step_batch({0, 1});
      require(t.size() == 2 && t[0].size() == 1 && t[1].size() == 1, "batch step shape");
      out->ba.push_back(t[0][0]);
      out->bb.push_back(t[1][0]);
    }
    graph_engine.drain();
    out->bc.push_back(graph_engine.prefill(2, C));
    graph_engine.reserve(2, static_cast<int64_t>(C.size()) + kSteps + 1);
    for (int s = 0; s < 4; ++s) {
      const auto t = graph_engine.step_batch({0, 1, 2});
      require(t.size() == 3, "batch step shape");
      out->ba.push_back(t[0][0]);
      out->bb.push_back(t[1][0]);
      out->bc.push_back(t[2][0]);
    }
    graph_engine.close(1);
    for (int s = 0; s < 3; ++s) {
      const auto t = graph_engine.step_batch({0, 2});
      require(t.size() == 2, "batch step shape");
      out->ba.push_back(t[0][0]);
      out->bc.push_back(t[1][0]);
    }
    graph_engine.close(0);
    graph_engine.close(2);
    graph_engine.drain();

    // 4. The group prefill: A, B and C as the spans of one forward, then
    //    scalar steps per slot — every transcript the eager engine's.
    const std::vector<int32_t> firsts = graph_engine.prefill_group({0, 1, 2}, {&A, &B, &C});
    require(firsts.size() == 3, "one first token per request");
    graph_engine.reserve(0, static_cast<int64_t>(A.size()) + kSteps + 1);
    graph_engine.reserve(1, static_cast<int64_t>(B.size()) + kSteps + 1);
    graph_engine.reserve(2, static_cast<int64_t>(C.size()) + kSteps + 1);
    out->pa = {firsts[0]};
    out->pb = {firsts[1]};
    out->pc = {firsts[2]};
    // Three live slots: the physical replay is the full family's batch.
    for (int s = 0; s < kSteps; ++s) {
      const auto t = graph_engine.step_batch({0, 1, 2});
      require(t.size() == 3 && t[0].size() == 1 && t[1].size() == 1 && t[2].size() == 1, "group batch step shape");
      out->pa.push_back(t[0][0]);
      out->pb.push_back(t[1][0]);
      out->pc.push_back(t[2][0]);
    }
    graph_engine.close(0);
    graph_engine.close(1);
    graph_engine.close(2);
    graph_engine.drain();
    cudaFreeHost(scratch);
  } catch (const std::exception& e) {
    if (scratch) cudaFreeHost(scratch);
    out->error = "rank " + std::to_string(r) + ": " + e.what();
    arrive_once();
  }
}

// The DSpark graph engine (plan D8): the T = 1 + depth verify with the
// recorded commit, the in-graph block draft (the accepted rows' main hidden
// into the draft rings, the five-row block, the Markov-biased picks —
// depth - 1 chain rows re-biasing the block's precomputed rows) and the
// [next, drafts] feed — scalar for A, then the batched families for B and
// C — against the plain eager engine: the greedy transcripts must be the
// plain engine's exactly (a draft only ever proposes; the verify decides).
struct MtpOutcome {
  std::string error;
  std::vector<int32_t> ea, eb, ec;   // the plain eager engine, world 2
  std::vector<int32_t> ma, mb, mc;   // the MTP graph engine: scalar A, batched B and C
  int mtp_steps_a = 0;
};

void rank_work_mtp(int r, const Dsv4Config& cfg, const std::string& dir, const std::vector<int64_t>& A,
                   const std::vector<int64_t>& B, const std::vector<int64_t>& C, CollectiveBus* bus,
                   ConstructBarrier* barrier, MtpOutcome* out, int depth) {
  bool arrived = false;
  const auto arrive_once = [&] {
    if (arrived) return;
    arrived = true;
    barrier->arrive_and_wait();
  };
  uint16_t* scratch = nullptr;
  try {
    BusBoundaryReducer reducer(*bus, wait_timeout_ms());
    // Two slots x (1 + depth) rows fit the family's 16-row decode cap.
    const int decode_rows = 2 * (1 + depth);
    Dsv4Model eager(cfg, dir, kMaxTokens, kCache, Dsv4Residency::Resident, &reducer, r, kWorld, kSlots);
    Dsv4Model mtp(cfg, dir, kMaxTokens, kCache, Dsv4Residency::Resident, &reducer, r, kWorld,
                   kSlots,
                   /*mtp=*/true, decode_rows, /*serving_logits=*/true);
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                               sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld), cudaHostAllocDefault));
    arrive_once();
    EagerEngineAdapter<Dsv4Model> eager_engine(
        &eager, kSlots, dgpp::make_fabric_pick(bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms()));
    out->ea = solo(eager_engine, 0, A, kSteps);
    out->eb = solo(eager_engine, 1, B, kSteps);
    out->ec = solo(eager_engine, 2, C, kSteps);
    {
      GraphEngineAdapter<Dsv4Model> mtp_engine(&mtp, bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms(),
                                                /*batch_min_live=*/2, /*prefix_scratch=*/nullptr,
                                                /*gather_scratch=*/nullptr, /*candidates=*/0, /*grammar=*/nullptr,
                                                /*prefix_slots=*/0, /*mtp_depth=*/depth);
      out->ma.push_back(mtp_engine.prefill(0, A));
      mtp_engine.reserve(0, static_cast<int64_t>(A.size()) + kSteps + 2 + depth);
      while (out->ma.size() < static_cast<size_t>(kSteps) + 1) {
        const std::vector<int32_t> t = mtp_engine.step(0);
        require(!t.empty() && t.size() <= static_cast<size_t>(1 + depth), "mtp step shape");
        out->ma.insert(out->ma.end(), t.begin(), t.end());
        ++out->mtp_steps_a;
      }
      out->ma.resize(static_cast<size_t>(kSteps) + 1);
      mtp_engine.close(0);
      out->mb.push_back(mtp_engine.prefill(1, B));
      mtp_engine.reserve(1, static_cast<int64_t>(B.size()) + kSteps + 2 + depth);
      out->mc.push_back(mtp_engine.prefill(2, C));
      mtp_engine.reserve(2, static_cast<int64_t>(C.size()) + kSteps + 2 + depth);
      while (out->mb.size() < static_cast<size_t>(kSteps) + 1 || out->mc.size() < static_cast<size_t>(kSteps) + 1) {
        const auto t = mtp_engine.step_batch({1, 2});
        require(t.size() == 2, "mtp batch step shape");
        out->mb.insert(out->mb.end(), t[0].begin(), t[0].end());
        out->mc.insert(out->mc.end(), t[1].begin(), t[1].end());
      }
      out->mb.resize(static_cast<size_t>(kSteps) + 1);
      out->mc.resize(static_cast<size_t>(kSteps) + 1);
      mtp_engine.close(1);
      mtp_engine.close(2);
      mtp_engine.drain();
    }
    cudaFreeHost(scratch);
  } catch (const std::exception& e) {
    if (scratch) cudaFreeHost(scratch);
    out->error = "rank " + std::to_string(r) + ": " + e.what();
    arrive_once();
  }
}

// The scheduled verify depth (plan D8a, engine/verify_schedule.hpp): the
// scalar MTP replay verifies a per-step prefix of the block on a variant
// captured at that depth. Exactness is the gate — the committed transcript
// must be the plain eager engine's whatever depth each step picks — so A
// runs under a forced, varied depth sequence (every reduced variant
// replays regardless of the fixture's arbitrary confidence: the hook
// returns the policy-independent 1 + 3 * step mod 5, i.e. 1,4,2,5,3,...),
// then B and C run batched (the batch verifies the full block and
// publishes each slot's confidence) and B continues alone under the real
// policy over the confidence the batch published. The reduced variants
// are the bus variants after the scalar and batch ones; the scheduled
// engine has two slots (its 12 rows fit the cap, so a batch family exists
// and the batched steps really replay the batch) and every depth is an
// option (22 of the bus's 32 variants).
struct SchedOutcome {
  std::string error;
  std::vector<int32_t> ea, eb, ec;  // the plain eager engine, world 2
  std::vector<int32_t> sa, sb, sc;  // the scheduled graph engine: scalar A, batched B/C, then B alone
  std::vector<int> options;
  std::vector<uint64_t> hist;
  std::vector<uint64_t> batch_hist;  // the two-slot family's replays per option
  std::vector<int> depths_a;        // the depth option each A step replayed at
  int steps_a = 0;
  int steps_batch = 0;              // B and C's batched steps
  int steps_b_alone = 0;            // B's scalar steps after the batch
};

void rank_work_sched(int r, const Dsv4Config& cfg, const std::string& dir, const std::vector<int64_t>& A,
                     const std::vector<int64_t>& B, const std::vector<int64_t>& C, CollectiveBus* bus,
                     ConstructBarrier* barrier, SchedOutcome* out) {
  bool arrived = false;
  const auto arrive_once = [&] {
    if (arrived) return;
    arrived = true;
    barrier->arrive_and_wait();
  };
  uint16_t* scratch = nullptr;
  try {
    const int depth = cfg.dspark_block_size;  // 5
    BusBoundaryReducer reducer(*bus, wait_timeout_ms());
    // Two request slots x (1 + depth) rows = 12 fit the model's decode-row
    // cap, so the engine has a batch family (three slots would not: 18 >
    // 16, and every "batched" step would fall back to scalar replays).
    const int sched_slots = 2;
    const int decode_rows = sched_slots * (1 + depth);
    Dsv4Model eager(cfg, dir, kMaxTokens, kCache, Dsv4Residency::Resident, &reducer, r, kWorld, kSlots);
    Dsv4Model mtp(cfg, dir, kMaxTokens, kCache, Dsv4Residency::Resident, &reducer, r, kWorld,
                   sched_slots,
                   /*mtp=*/true, decode_rows, /*serving_logits=*/true);
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                               sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld), cudaHostAllocDefault));
    arrive_once();
    EagerEngineAdapter<Dsv4Model> eager_engine(
        &eager, kSlots, dgpp::make_fabric_pick(bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms()));
    out->ea = solo(eager_engine, 0, A, kSteps);
    out->eb = solo(eager_engine, 1, B, kSteps);
    out->ec = solo(eager_engine, 2, C, kSteps);
    {
      GraphEngineAdapter<Dsv4Model> eng(&mtp, bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms(),
                                         /*batch_min_live=*/2, /*prefix_scratch=*/nullptr,
                                         /*gather_scratch=*/nullptr, /*candidates=*/0, /*grammar=*/nullptr,
                                         /*prefix_slots=*/0, /*mtp_depth=*/depth);
      eng.configure_verify_schedule(true, /*row_ms=*/9.0f, dgpp::verify_reservation_lambda(20.0f, 9.0f),
                                    /*min_depth=*/1);
      out->options = eng.verify_depth_options();
      // A: the forced, varied depth sequence.
      int step = 0;
      eng.set_verify_depth_hook([&](int, int, const float*, int) { return 1 + (3 * step++) % 5; });
      out->sa.push_back(eng.prefill(0, A));
      eng.reserve(0, static_cast<int64_t>(A.size()) + kSteps + 2 + depth);
      std::vector<uint64_t> before = eng.verify_depth_histogram();
      while (out->sa.size() < static_cast<size_t>(kSteps) + 1) {
        const std::vector<int32_t> t = eng.step(0);
        require(!t.empty() && t.size() <= static_cast<size_t>(1 + depth), "sched step shape");
        out->sa.insert(out->sa.end(), t.begin(), t.end());
        ++out->steps_a;
        const std::vector<uint64_t> after = eng.verify_depth_histogram();
        for (size_t i = 0; i < after.size(); ++i)
          if (after[i] != before[i]) out->depths_a.push_back(out->options[i]);
        before = after;
      }
      out->sa.resize(static_cast<size_t>(kSteps) + 1);
      eng.close(0);
      // B (slot 0) and C (slot 1) batched half way under forced, varied
      // depths per slot (the batch takes the deepest: the reduced-depth
      // batch variants replay — the compacted feeds, the compact masks,
      // one depth for both slots) and publishes both slots' confidence.
      int bstep = 0;
      eng.set_verify_depth_hook([&](int req, int, const float*, int) {
        const int k = req == 0 ? 1 + (3 * bstep) % 5 : 1 + (2 * bstep) % 5;
        if (req == 1) ++bstep;  // both slots asked per step; advance after the second
        return k;
      });
      out->sb.push_back(eng.prefill(0, B));
      eng.reserve(0, static_cast<int64_t>(B.size()) + kSteps + 2 + depth);
      out->sc.push_back(eng.prefill(1, C));
      eng.reserve(1, static_cast<int64_t>(C.size()) + kSteps + 2 + depth);
      const size_t half = static_cast<size_t>(kSteps) / 2 + 1;
      while (out->sb.size() < half || out->sc.size() < half) {
        const auto t = eng.step_batch({0, 1});
        require(t.size() == 2, "sched batch step shape");
        out->sb.insert(out->sb.end(), t[0].begin(), t[0].end());
        out->sc.insert(out->sc.end(), t[1].begin(), t[1].end());
        ++out->steps_batch;
      }
      require(eng.batch_families().size() == 1 && eng.batch_family_steps(0) > 0,
              "the two-slot batch family replayed (its confidence publication is exercised)");
      out->batch_hist = eng.verify_depth_histogram_batch(0);
      eng.close(1);
      // B alone: first one step forced to the whole block right after a
      // reduced-depth batch (the settle's bound must be the replay's own
      // rows, not the batch contract's — the 2026-09-14 fabric failure),
      // then the real policy over the confidence the batch published.
      bool first_alone = true;
      eng.set_verify_depth_hook([&](int, int suggested, const float*, int) {
        const int k = first_alone ? depth : suggested;
        first_alone = false;
        return k;
      });
      while (out->sb.size() < static_cast<size_t>(kSteps) + 1) {
        const std::vector<int32_t> t = eng.step(0);
        require(!t.empty() && t.size() <= static_cast<size_t>(1 + depth), "sched post-batch step shape");
        out->sb.insert(out->sb.end(), t.begin(), t.end());
        ++out->steps_b_alone;
      }
      out->sb.resize(static_cast<size_t>(kSteps) + 1);
      eng.close(0);
      eng.drain();
      out->hist = eng.verify_depth_histogram();
    }
    cudaFreeHost(scratch);
  } catch (const std::exception& e) {
    if (scratch) cudaFreeHost(scratch);
    out->error = "rank " + std::to_string(r) + ": " + e.what();
    arrive_once();
  }
}

// Six request slots at depth 4 (30 rows, the family's 32-row cap since
// 2026-09-14; the six-stream serving shape): every slot batched in
// the 6-slot family under the scheduled depth (forced per-slot depths so
// the reduced batch variants replay), every transcript the plain eager
// engine's, both ranks identical.
struct WideOutcome {
  std::string error;
  std::vector<std::vector<int32_t>> eager, wide;  // per slot
  std::vector<int> families;
  uint64_t steps6 = 0;
  std::vector<uint64_t> batch_hist;
};

void rank_work_wide(int r, const Dsv4Config& cfg, const std::string& dir,
                    const std::vector<std::vector<int64_t>>& prompts, CollectiveBus* bus, ConstructBarrier* barrier,
                    WideOutcome* out, int depth, bool scheduled) {
  bool arrived = false;
  const auto arrive_once = [&] {
    if (arrived) return;
    arrived = true;
    barrier->arrive_and_wait();
  };
  uint16_t* scratch = nullptr;
  try {
    const int slots = 6;
    // 30 rows at depth 4; at depth 5 the family's 32-row cap (36 asked).
    const int decode_rows = std::min(Dsv4Model::decode_rows_cap(), slots * (1 + depth));
    // Six slots x two 128-token blocks each (the prompt and its decode,
    // plus the chain rows) need more pool than the three-slot worlds' 512.
    const int64_t cache = 4 * kCache;
    BusBoundaryReducer reducer(*bus, wait_timeout_ms());
    // Both models at the same decode rows: the GEMM lowering follows the
    // decode rows into short prefills (glm53-sixteen-row trap 1), so an
    // oracle at the default rows would prefill these prompts on another
    // kernel and diverge at near-ties.
    Dsv4Model eager(cfg, dir, kMaxTokens, cache, Dsv4Residency::Resident, &reducer, r, kWorld, slots, /*mtp=*/false,
                     decode_rows);
    Dsv4Model mtp(cfg, dir, kMaxTokens, cache, Dsv4Residency::Resident, &reducer, r, kWorld,
                   slots,
                   /*mtp=*/true, decode_rows, /*serving_logits=*/true);
    require(mtp.max_decode_rows() == decode_rows, "the model takes the wide decode batch");
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                               sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld), cudaHostAllocDefault));
    arrive_once();
    EagerEngineAdapter<Dsv4Model> eager_engine(
        &eager, slots, dgpp::make_fabric_pick(bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms()));
    for (int q = 0; q < slots; ++q) out->eager.push_back(solo(eager_engine, q, prompts[static_cast<size_t>(q)], kSteps));
    {
      GraphEngineAdapter<Dsv4Model> eng(&mtp, bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms(),
                                         /*batch_min_live=*/2, nullptr, nullptr, 0, nullptr, 0, /*mtp_depth=*/depth);
      out->families = eng.batch_families();
      if (scheduled) eng.configure_verify_schedule(true, 9.0f, dgpp::verify_reservation_lambda(20.0f, 9.0f), 1);
      // One depth per step for every slot (the batch takes the deepest
      // slot's, so per-slot variety would always reach the full block).
      int bstep = 0;
      if (scheduled)
        eng.set_verify_depth_hook([&](int req, int, const float*, int) {
          const int k = 1 + bstep % depth;
          if (req == slots - 1) ++bstep;
          return k;
        });
      out->wide.assign(static_cast<size_t>(slots), {});
      std::vector<int> reqs;
      for (int q = 0; q < slots; ++q) {
        out->wide[static_cast<size_t>(q)].push_back(eng.prefill(q, prompts[static_cast<size_t>(q)]));
        eng.reserve(q, static_cast<int64_t>(prompts[static_cast<size_t>(q)].size()) + kSteps + 2 + depth);
        reqs.push_back(q);
      }
      bool done = false;
      while (!done) {
        const auto t = eng.step_batch(reqs);
        require(t.size() == static_cast<size_t>(slots), "wide batch step shape");
        ++out->steps6;
        done = true;
        for (int q = 0; q < slots; ++q) {
          out->wide[static_cast<size_t>(q)].insert(out->wide[static_cast<size_t>(q)].end(), t[static_cast<size_t>(q)].begin(),
                                                    t[static_cast<size_t>(q)].end());
          if (out->wide[static_cast<size_t>(q)].size() < static_cast<size_t>(kSteps) + 1) done = false;
        }
      }
      for (int q = 0; q < slots; ++q) out->wide[static_cast<size_t>(q)].resize(static_cast<size_t>(kSteps) + 1);
      const int fam6 = static_cast<int>(out->families.size()) - 1;
      if (scheduled) out->batch_hist = eng.verify_depth_histogram_batch(fam6);
      for (int q = 0; q < slots; ++q) eng.close(q);
      eng.drain();
    }
    cudaFreeHost(scratch);
  } catch (const std::exception& e) {
    if (scratch) cudaFreeHost(scratch);
    out->error = "rank " + std::to_string(r) + ": " + e.what();
    arrive_once();
  }
}

void check_wide_world(uint16_t port, int depth = 4, bool scheduled = true) {
  const Dsv4Config cfg = dsv4fx::tiny_config();
  const std::string dir = (fs::current_path() / "dsv4_engine_fixture").string();
  dsv4fx::write_fixture(cfg, dir);
  std::vector<std::vector<int64_t>> prompts;
  const uint64_t seeds[6] = {0x9E3779B97F4A7C15ull, 0xD1B54A32D192ED03ull, 0x2545F4914F6CDD1Dull,
                             0x3C6EF372FE94F82Bull, 0x1F83D9ABFB41BD6Bull, 0x5BE0CD19137E2179ull};
  for (int q = 0; q < 6; ++q) prompts.push_back(smoke_tokens(cfg, 40 + 17 * q, seeds[q]));
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, port);
  require(!buses.empty(), "the loopback bus world failed to start");
  std::vector<WideOutcome> outs(kWorld);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int r = 0; r < kWorld; ++r)
    workers.emplace_back(rank_work_wide, r, std::cref(cfg), std::cref(dir), std::cref(prompts),
                         buses[static_cast<size_t>(r)].get(), &barrier, &outs[static_cast<size_t>(r)], depth, scheduled);
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r) require(outs[static_cast<size_t>(r)].error.empty(), outs[static_cast<size_t>(r)].error);
  for (int r = 1; r < kWorld; ++r)
    require(outs[static_cast<size_t>(r)].wide == outs[0].wide, "the ranks' six-slot transcripts differ");
  const WideOutcome& o = outs[0];
  std::string hist;
  for (size_t i = 0; i < o.batch_hist.size(); ++i) hist += (i ? " " : "") + std::to_string(o.batch_hist[i]);
  DGPP_LOG_INFO("world 2 six slots x depth {}: families {} | {} batched steps, the widest family's replays per depth option [{}]",
                depth, ids_text(o.families), o.steps6, hist);
  // Depth 4: 30 rows, every slot in the 6-slot family. Depth 5: 36 rows
  // against the 32-row cap — the widest full-block family holds five
  // slots, and the depth-capped 6-slot family takes all six at up to four
  // drafts (a step the policy asks five of verifies four: exact).
  require(o.families == (depth == 4 ? std::vector<int>{2, 3, 4, 6} : std::vector<int>{2, 3, 4, 5, 6}),
          "the six-slot world's batch families");
  for (int q = 0; q < 6; ++q)
    require(o.wide[static_cast<size_t>(q)] == o.eager[static_cast<size_t>(q)],
            "slot " + std::to_string(q) + "'s six-slot batched transcript differs from the plain eager engine's");
  if (!scheduled) return;  // a fixed depth: the transcripts are the gate
  if (depth == 5) require(o.batch_hist.size() == 5 && o.batch_hist[4] == 0, "the depth-capped family never replays the full block");
  uint64_t total = 0;
  int used = 0;
  for (const uint64_t h : o.batch_hist) {
    total += h;
    used += h > 0 ? 1 : 0;
  }
  require(total == o.steps6 && used >= 2, "the 6-slot family replayed at more than one scheduled depth");
}

// A SAMPLED request under the scheduled verify depth (2026-10-02,
// engine.mtp_schedule_sampled_scale): with argmax drafts
// (set_proposal_drafts(false)) a sampled slot follows the schedule — its
// accept test is per row, so the rows left unverified change the pace, not
// the distribution. The gates: the slot's depth is asked of the policy
// (the hook is called: a sampled slot used to verify the whole block
// unasked), every option replays, the ranks commit the same tokens, and
// the same seed commits the same tokens again.
struct SampledSchedOutcome {
  std::string error;
  std::vector<int32_t> first, second;
  std::vector<uint64_t> hist;
  int hook_calls = 0;
  // Two sampled slots batched at reduced depths (the fallback's snapshot
  // rows of a slot past the first): per run, per slot.
  std::vector<std::vector<int32_t>> pair_first, pair_second;
  uint64_t pair_fallbacks = 0;
};

void rank_work_sampled_sched(int r, const Dsv4Config& cfg, const std::string& dir, const std::vector<int64_t>& A,
                             CollectiveBus* bus, ConstructBarrier* barrier, SampledSchedOutcome* out) {
  bool arrived = false;
  uint16_t* scratch = nullptr;
  uint16_t* prefix_scratch = nullptr;
  uint16_t* gather_scratch = nullptr;
  try {
    const int depth = cfg.dspark_block_size;
    BusBoundaryReducer reducer(*bus, wait_timeout_ms());
    const int slots = 2;
    Dsv4Model mtp(cfg, dir, kMaxTokens, kCache, Dsv4Residency::Resident, &reducer, r, kWorld, slots, /*mtp=*/true,
                   slots * (1 + depth), /*serving_logits=*/true);
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                               sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld), cudaHostAllocDefault));
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&prefix_scratch),
                               2 * dgpp::fabric_sampling_prefix_scratch_elems(kWorld), cudaHostAllocDefault));
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&gather_scratch),
                               2 * dgpp::sampling_gather_scratch_elems(cfg.vocab_size), cudaHostAllocDefault));
    arrived = true;
    barrier->arrive_and_wait();
    {
      GraphEngineAdapter<Dsv4Model> eng(&mtp, bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms(),
                                         /*batch_min_live=*/2, prefix_scratch, gather_scratch, /*candidates=*/32,
                                         /*grammar=*/nullptr, /*prefix_slots=*/0, /*mtp_depth=*/depth);
      eng.set_proposal_drafts(false);
      eng.set_sampled_schedule_scale(0.93f);
      eng.configure_verify_schedule(true, /*row_ms=*/9.0f, dgpp::verify_reservation_lambda(20.0f, 9.0f),
                                    /*min_depth=*/1);
      for (int run = 0; run < 2; ++run) {
        std::vector<int32_t>& t = run == 0 ? out->first : out->second;
        dgpp::sample::Params params;
        params.temperature = 1.0f;
        eng.configure_sampling(0, params, 4242);
        int step = 0;
        eng.set_verify_depth_hook([&](int, int, const float*, int) {
          ++out->hook_calls;
          return 1 + (3 * step++) % 5;
        });
        t.push_back(eng.prefill(0, A));
        eng.reserve(0, static_cast<int64_t>(A.size()) + kSteps + 2 + depth);
        while (t.size() < static_cast<size_t>(kSteps) + 1) {
          const std::vector<int32_t> got = eng.step(0);
          require(!got.empty() && got.size() <= static_cast<size_t>(1 + depth), "sampled sched step shape");
          t.insert(t.end(), got.begin(), got.end());
        }
        t.resize(static_cast<size_t>(kSteps) + 1);
        eng.close(0);
      }
      eng.drain();
      out->hist = eng.verify_depth_histogram();
      // Both slots sampled and batched, the batch's depth forced through
      // every option: a fallback of slot 1 in a reduced-depth batch reads
      // its verify snapshot at the replay's rows per request (2026-10-02:
      // read at the full block's stride it gathered another slot's row,
      // which the engine refuses — every sampled multi-request step under
      // the schedule could fail).
      const uint64_t fallbacks_before = eng.fallbacks();
      for (int run = 0; run < 2; ++run) {
        std::vector<std::vector<int32_t>>& t = run == 0 ? out->pair_first : out->pair_second;
        t.assign(2, {});
        dgpp::sample::Params params;
        params.temperature = 1.0f;
        int step = 0;
        eng.set_verify_depth_hook([&](int req, int, const float*, int) {
          const int k = 1 + step % 5;
          if (req == 1) ++step;
          return k;
        });
        for (int q = 0; q < 2; ++q) {
          eng.configure_sampling(q, params, 977 + 31 * q);
          t[static_cast<size_t>(q)].push_back(eng.prefill(q, A));
          eng.reserve(q, static_cast<int64_t>(A.size()) + 3 * kSteps + 2 + depth);
        }
        while (t[0].size() < static_cast<size_t>(3 * kSteps) + 1 || t[1].size() < static_cast<size_t>(3 * kSteps) + 1) {
          const auto got = eng.step_batch({0, 1});
          require(got.size() == 2, "sampled pair step shape");
          for (int q = 0; q < 2; ++q)
            t[static_cast<size_t>(q)].insert(t[static_cast<size_t>(q)].end(), got[static_cast<size_t>(q)].begin(),
                                             got[static_cast<size_t>(q)].end());
        }
        for (int q = 0; q < 2; ++q) {
          t[static_cast<size_t>(q)].resize(static_cast<size_t>(3 * kSteps) + 1);
          eng.close(q);
        }
      }
      eng.drain();
      out->pair_fallbacks = eng.fallbacks() - fallbacks_before;
    }
  } catch (const std::exception& e) {
    out->error = e.what();
    if (!arrived) barrier->arrive_and_wait();
  }
  if (scratch) cudaFreeHost(scratch);
  if (prefix_scratch) cudaFreeHost(prefix_scratch);
  if (gather_scratch) cudaFreeHost(gather_scratch);
}

void check_sampled_sched_world(uint16_t port) {
  const Dsv4Config cfg = dsv4fx::tiny_config();
  const std::string dir = (fs::current_path() / "dsv4_engine_fixture").string();
  dsv4fx::write_fixture(cfg, dir);
  const std::vector<int64_t> A = smoke_tokens(cfg, 120, 0x9E3779B97F4A7C15ull);
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, port);
  require(!buses.empty(), "the loopback bus world failed to start");
  std::vector<SampledSchedOutcome> outs(kWorld);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int r = 0; r < kWorld; ++r)
    workers.emplace_back(rank_work_sampled_sched, r, std::cref(cfg), std::cref(dir), std::cref(A),
                         buses[static_cast<size_t>(r)].get(), &barrier, &outs[static_cast<size_t>(r)]);
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r) require(outs[static_cast<size_t>(r)].error.empty(), outs[static_cast<size_t>(r)].error);
  const SampledSchedOutcome& o = outs[0];
  for (int r = 1; r < kWorld; ++r)
    require(outs[static_cast<size_t>(r)].first == o.first && outs[static_cast<size_t>(r)].second == o.second,
            "the ranks' sampled scheduled transcripts differ");
  std::string hist;
  for (size_t i = 0; i < o.hist.size(); ++i) hist += (i ? " " : "") + std::to_string(o.hist[i]);
  DGPP_LOG_INFO("world 2 sampled scheduled depth: {} | again {} | policy asked {} times | replays per option [{}]",
                ids_text(o.first), ids_text(o.second), o.hook_calls, hist);
  require(o.hook_calls > 0, "a sampled slot with argmax drafts asks the depth policy");
  require(o.first == o.second, "the same seed commits the same sampled tokens under the same depths");
  uint64_t reduced = 0;
  for (size_t i = 0; i + 1 < o.hist.size(); ++i) reduced += o.hist[i];
  require(reduced > 0, "the sampled slot replayed reduced-depth variants");
  for (int r = 1; r < kWorld; ++r)
    require(outs[static_cast<size_t>(r)].pair_first == o.pair_first && outs[static_cast<size_t>(r)].pair_second == o.pair_second,
            "the ranks' batched sampled transcripts differ");
  require(o.pair_first == o.pair_second, "the same seeds commit the same batched sampled tokens");
  DGPP_LOG_INFO("world 2 two sampled slots batched under the schedule: {} fallbacks served over two runs", o.pair_fallbacks);
  require(o.pair_fallbacks > 0, "the batched sampled runs exercised the host fallback");
}

void check_sched_world(uint16_t port) {
  const Dsv4Config cfg = dsv4fx::tiny_config();
  const std::string dir = (fs::current_path() / "dsv4_engine_fixture").string();
  dsv4fx::write_fixture(cfg, dir);
  const std::vector<int64_t> A = smoke_tokens(cfg, 120, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 70, 0xD1B54A32D192ED03ull);
  const std::vector<int64_t> C = smoke_tokens(cfg, 37, 0x2545F4914F6CDD1Dull);
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, port);
  require(!buses.empty(), "the loopback bus world failed to start");
  std::vector<SchedOutcome> outs(kWorld);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int r = 0; r < kWorld; ++r)
    workers.emplace_back(rank_work_sched, r, std::cref(cfg), std::cref(dir), std::cref(A), std::cref(B), std::cref(C),
                         buses[static_cast<size_t>(r)].get(), &barrier, &outs[static_cast<size_t>(r)]);
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r) require(outs[static_cast<size_t>(r)].error.empty(), outs[static_cast<size_t>(r)].error);
  for (int r = 1; r < kWorld; ++r)
    require(outs[static_cast<size_t>(r)].sa == outs[0].sa && outs[static_cast<size_t>(r)].sb == outs[0].sb &&
                outs[static_cast<size_t>(r)].sc == outs[0].sc && outs[static_cast<size_t>(r)].depths_a == outs[0].depths_a,
            "the ranks' scheduled transcripts or depths differ");
  const SchedOutcome& o = outs[0];
  std::string depths;
  for (const int d : o.depths_a) depths += (depths.empty() ? "" : ",") + std::to_string(d);
  std::string hist, bhist;
  for (size_t i = 0; i < o.hist.size(); ++i)
    hist += (i ? " " : "") + std::to_string(o.options[i]) + ":" + std::to_string(o.hist[i]);
  for (size_t i = 0; i < o.batch_hist.size(); ++i)
    bhist += (i ? " " : "") + std::to_string(o.options[i]) + ":" + std::to_string(o.batch_hist[i]);
  DGPP_LOG_INFO("world 2 scheduled depth: options {} | A {} ({} steps at depths {}) | B {} | C {} | scalar replays per depth [{}] | batched [{}]",
                ids_text(o.options), ids_text(o.sa), o.steps_a, depths, ids_text(o.sb), ids_text(o.sc), hist, bhist);
  require(o.options == std::vector<int>{1, 2, 3, 4, 5},
          "every depth is an option (two slots and one family: 2 x 2 x 5 + 2 = 22 of the bus's 32 variants)");
  require(o.sa == o.ea, "the scheduled scalar transcript of A differs from the plain eager engine's");
  require(o.sb == o.eb, "the scheduled transcript of B (batched, then alone) differs from the plain eager engine's");
  require(std::equal(o.sc.begin(), o.sc.end(), o.ec.begin()), "the batched transcript of C differs");
  // Every option replayed for A (the forced sequence 1,4,2,5,3 rounds 3 up to 4).
  for (size_t i = 0; i < o.hist.size(); ++i) require(o.hist[i] > 0, "an option never replayed");
  uint64_t total = 0;
  for (const uint64_t h : o.hist) total += h;
  require(total == static_cast<uint64_t>(o.steps_a) + static_cast<uint64_t>(o.steps_b_alone),
          "the histogram counts every scalar replay (A's, then B's after the batch)");
  require(o.steps_b_alone > 0, "B stepped alone after the batch (the batch's confidence publication is exercised)");
  uint64_t btotal = 0;
  int boptions = 0;
  for (const uint64_t h : o.batch_hist) {
    btotal += h;
    boptions += h > 0 ? 1 : 0;
  }
  require(btotal == static_cast<uint64_t>(o.steps_batch), "the batch histogram counts every batched replay");
  require(boptions >= 2 && o.batch_hist.back() < btotal,
          "the batch replayed reduced-depth variants (the compacted feeds and masks) as well as the full block");
}

void check_mtp_world(int depth, uint16_t port) {
  const Dsv4Config cfg = dsv4fx::tiny_config();
  const std::string dir = (fs::current_path() / "dsv4_engine_fixture").string();
  dsv4fx::write_fixture(cfg, dir);
  const std::vector<int64_t> A = smoke_tokens(cfg, 120, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 70, 0xD1B54A32D192ED03ull);
  const std::vector<int64_t> C = smoke_tokens(cfg, 37, 0x2545F4914F6CDD1Dull);
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, port);
  require(!buses.empty(), "the loopback bus world failed to start");
  std::vector<MtpOutcome> outs(kWorld);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int r = 0; r < kWorld; ++r)
    workers.emplace_back(rank_work_mtp, r, std::cref(cfg), std::cref(dir), std::cref(A), std::cref(B), std::cref(C),
                         buses[static_cast<size_t>(r)].get(), &barrier, &outs[static_cast<size_t>(r)], depth);
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r) require(outs[static_cast<size_t>(r)].error.empty(), outs[static_cast<size_t>(r)].error);
  for (int r = 1; r < kWorld; ++r)
    require(outs[static_cast<size_t>(r)].ma == outs[0].ma && outs[static_cast<size_t>(r)].mb == outs[0].mb &&
                outs[static_cast<size_t>(r)].mc == outs[0].mc,
            "the ranks' MTP transcripts differ");
  const MtpOutcome& o = outs[0];
  DGPP_LOG_INFO("world 2 DSpark depth {}: A {} ({} steps) | B {} | C {}", depth, ids_text(o.ma), o.mtp_steps_a,
                ids_text(o.mb), ids_text(o.mc));
  require(o.ma == o.ea, "the DSpark scalar transcript differs from the plain eager engine's");
  require(o.mb == o.eb, "the DSpark batched transcript of B differs from the plain eager engine's");
  require(o.mc == o.ec, "the DSpark batched transcript of C differs from the plain eager engine's");
  // A random-weight fixture drafts by chance only (the acceptance rate is
  // the real checkpoint's measurement).
  DGPP_LOG_INFO("world 2 DSpark depth {}: A took {} steps for {} tokens", depth, o.mtp_steps_a, kSteps);
}

}  // namespace

DGPP_TEST(dsv4_engines_adaptive_prefill_mixed_cancel_and_reuse) {
  const auto cfg = dsv4fx::tiny_config();
  const std::string dir = "dsv4_engine_fixture/adaptive";
  dsv4fx::write_fixture(cfg, dir);
  for (bool graph : {false, true}) {
    auto buses = start_world(1, kPort + 20);
    require(buses.size() == 1, "adaptive fixture bus");
    Dsv4Model model(cfg, dir, 512, 4096, Dsv4Residency::Resident,
                    nullptr, 0, 1, 4, true, 32, true);
    uint16_t* scratch = nullptr;
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
        2 * dgpp::kPickScratchElems(1), cudaHostAllocDefault));
    try {
      {
        std::unique_ptr<dgpp::sched::SchedulerEngine> engine;
        if (graph) {
          auto adapter = std::make_unique<GraphEngineAdapter<Dsv4Model>>(
              &model, buses[0].get(), 0, 1, scratch, cfg.vocab_size, wait_timeout_ms(),
              2, nullptr, nullptr, 1, nullptr, 0, 2, true);
          adapter->configure_verify_schedule(true, 1.f, .5f, 1);
          engine = std::move(adapter);
        } else {
          engine = std::make_unique<EagerEngineAdapter<Dsv4Model>>(&model, 4,
              [](const dgpp::DecodeOutputs& out) {
                return static_cast<int32_t>(std::max_element(out.logits.begin(), out.logits.end()) - out.logits.begin());
              });
        }
        dgpp::sched::AdmissionPolicy policy;
        policy.prefill_budget_tokens = -1;
        policy.prefill_latency.initial_tokens = 128;
        policy.decode_gap_target_ms = dgpp::sched::kDefaultDecodeGapTargetMs;
        policy.first_token_target_ms = dgpp::sched::kDefaultFirstTokenTargetMs;
        policy = dgpp::serve::resolve_prefill_policy(policy, *engine, std::nullopt);
        require(policy.prefill_budget_automatic && policy.prefill_early_admission,
                "non-Qwen adapter resolves automatic admission and budgets");
        require(engine->prefill_chunk_alignment() == 128, "exercise compressor alignment");
        struct Observer : dgpp::sched::SchedulerObserver {
          int short_tokens = 0;
          void on_token(const std::string& id, int64_t, int) override {
            if (id == "short") ++short_tokens;
          }
          void on_retire(const std::string&, const dgpp::sched::Scheduler::Result&) override {}
        } observer;
        dgpp::sched::Scheduler sched(engine.get(), {}, 0, policy, 0);
        sched.set_observer(&observer);
        auto request = [&](const char* id, int length, int outputs, int salt) {
          dgpp::sched::SchedulerRequest r;
          r.id = id; r.prompt = smoke_tokens(cfg, length, salt);
          r.max_steps = outputs; r.no_cache = true;
          return r;
        };
        sched.submit(request("decoder", 17, 24, 101));
        sched.tick();
        sched.submit(request("long", 1801, 2, 102));
        sched.tick();
        require(sched.meters().prefilling == 1, "long reader yields before completion");
        sched.submit(request("short", 13, 8, 103));
        const auto mixed = sched.meters().mixed_steps;
        sched.tick();
        require(sched.meters().mixed_steps > mixed && sched.meters().queued == 0,
                "adaptive mixed walk admits the later short reader");
        require(observer.short_tokens > 0 && sched.meters().prefilling == 1,
                "short reader emits before the long reader completes");
        sched.cancel("long");
        sched.tick();
        require(sched.meters().prefilling == 0, "cancelled continuation releases its slot");
        require(sched.find("long")->reason == dgpp::sched::Scheduler::Result::Reason::kCancelled,
                "unfinished reader retires through cancellation");
        sched.submit(request("reuse", 901, 4, 104));
        bool grew = false;
        for (int i = 0; i < 100 && sched.has_pending(); ++i) {
          const auto limit = sched.next_prefill_budget();
          require(limit && *limit >= 128 && *limit <= 512 && *limit % 128 == 0,
                  "learned budget preserves native geometry");
          grew |= *limit > 128;
          sched.tick(limit);
        }
        require(grew && !sched.has_pending(), "adaptive growth and bounded completion");
        for (const auto& expected : {std::pair{"decoder", 24}, {"short", 8}, {"reuse", 4}})
          require(sched.find(expected.first)->steps_done == expected.second,
                  "uncancelled requests produce their full output after slot reuse");
        require(sched.meters().pool_blocks_in_use == 0, "adaptive cancellation and completion release KV");
      }
      require(model.kv_blocks_in_use() == 0, "adapter retains no request reservations");
    } catch (...) { cudaFreeHost(scratch); throw; }
    DGPP_CUDA_OK(cudaFreeHost(scratch));
  }
}

DGPP_TEST(dsv4_engines_loopback_world_2_dspark_block_matches_plain_decode) { check_mtp_world(5, kPort + 1); }

DGPP_TEST(dsv4_engines_loopback_world_2_dspark_depth_2_matches_plain_decode) { check_mtp_world(2, kPort + 2); }

DGPP_TEST(dsv4_engines_loopback_world_2_scheduled_verify_depth_is_exact) { check_sched_world(kPort + 3); }

DGPP_TEST(dsv4_engines_loopback_world_2_sampled_slot_follows_the_schedule) { check_sampled_sched_world(30005); }

DGPP_TEST(dsv4_engines_loopback_world_2_six_slots_at_depth_4_batched_scheduled_is_exact) { check_wide_world(kPort + 4); }

// Six slots at the full block (the four-node template's shape): 36 rows
// exceed the 32-row decode batch, so under the schedule six live requests
// ride the depth-capped 6-slot family (five rows each) — every transcript
// still the plain eager engine's. (Until 2026-10-02 such a step replayed
// six scalar graphs.)
DGPP_TEST(dsv4_engines_loopback_world_2_six_slots_at_depth_5_ride_the_depth_capped_family_exactly) {
  check_wide_world(30006, 5);
}

// ... and at a FIXED depth 5 (no schedule, so no reduced variants): the
// five-slot family with the sixth slot stepped alone.
DGPP_TEST(dsv4_engines_loopback_world_2_six_slots_at_fixed_depth_5_split_past_the_widest_family_is_exact) {
  check_wide_world(30007, 5, /*scheduled=*/false);
}

DGPP_TEST(dsv4_engines_loopback_world_2_graph_matches_eager) {
  const Dsv4Config cfg = dsv4fx::tiny_config();
  const std::string dir = (fs::current_path() / "dsv4_engine_fixture").string();
  dsv4fx::write_fixture(cfg, dir);
  const std::vector<int64_t> A = smoke_tokens(cfg, 120, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 70, 0xD1B54A32D192ED03ull);
  const std::vector<int64_t> C = smoke_tokens(cfg, 37, 0x2545F4914F6CDD1Dull);
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
  // The world-2 eager engine against world 1: the first decisions agree
  // or flip on a demonstrated near tie (require_agrees_or_tie; the bare
  // prefix rule this replaces passed or failed on which side of a tie a
  // kernel's rounding fell).
  const size_t pa = agreeing_prefix(o.ea, ref.a), pb = agreeing_prefix(o.eb, ref.b), pc = agreeing_prefix(o.ec, ref.c);
  DGPP_LOG_INFO("world 2 vs world 1 agreeing prefixes: A {}/{} B {}/{} C {}/{}", pa, ref.a.size(), pb, ref.b.size(),
                pc, ref.c.size());
  auto margins_text = [](const std::vector<float>& m) {
    std::string s;
    for (size_t i = 0; i < m.size(); ++i) s += (i ? " " : "") + std::format("{:.2e}", m[i]);
    return s;
  };
  DGPP_LOG_INFO("world 1 margins: A [{}] | B [{}] | C [{}]", margins_text(ref.am), margins_text(ref.bm),
                margins_text(ref.cm));
  require_agrees_or_tie(o.ea, ref.a, ref.am, kTieMargin, kAgreePositions, "A");
  require_agrees_or_tie(o.eb, ref.b, ref.bm, kTieMargin, kAgreePositions, "B");
  require_agrees_or_tie(o.ec, ref.c, ref.cm, kTieMargin, kAgreePositions, "C");
  // The graph engine against the eager engine at the same world: exact.
  require(o.ga == o.ea, "the scalar graph transcript differs from the eager engine's");
  require(o.ba == o.ea, "the batched graph transcript of A differs from the eager engine's");
  require(std::equal(o.bb.begin(), o.bb.end(), o.eb.begin()), "the batched graph transcript of B differs");
  require(std::equal(o.bc.begin(), o.bc.end(), o.ec.begin()), "the batched graph transcript of C differs");
  // The group prefill (one forward over A, B and C) then scalar steps:
  // bitwise the eager engine's prefills alone.
  DGPP_LOG_INFO("world 2 group prefill: A {} | B {} | C {}", ids_text(o.pa), ids_text(o.pb), ids_text(o.pc));
  require(o.pa == o.ea, "the group prefill's transcript of A differs from the eager engine's");
  require(o.pb == o.eb, "the group prefill's transcript of B differs from the eager engine's");
  require(o.pc == o.ec, "the group prefill's transcript of C differs from the eager engine's");
  for (int r = 1; r < kWorld; ++r)
    require(outs[static_cast<size_t>(r)].pa == outs[0].pa && outs[static_cast<size_t>(r)].pb == outs[0].pb &&
                outs[static_cast<size_t>(r)].pc == outs[0].pc,
            "the ranks' group prefill transcripts differ");
  require(o.bb.size() == 10 && o.bc.size() == 8, "the batched transcripts' lengths");
}

int main() { return dgpp::test::run_all(); }
