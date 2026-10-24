// The DeepSeek-V4.1 decode engines over the engine core (plan G5):
// EagerEngineAdapter<Dsv41Model> and GraphEngineAdapter<Dsv41Model> on the
// fixture, a loopback world of 2 (real verbs QPs over 127.0.0.1) against
// the world-1 eager engine.
//
// Gates: the graph engine's SCALAR replays (the device-driven T=1 graph:
// device positions, the recorded commit, the pinned token upload, the
// Engram host node forked inside the walk — `session_graph_host_nodes()`
// declares it) and its ROW-BATCHED replays (the fixed slot-major batch off
// the persistent feeds, closed slots padding at -1, the 2-slot and full
// families; the CSA2 decode over request spans, the window rings per
// slot) produce the eager engine's transcripts at the same world exactly
// (the recorded kernels are the eager kernels, the host node's rows the
// same gather); the world-2 eager transcripts follow the world-1 ones
// (reported; the folds reassociate, so a near tie may flip a late token —
// the first tokens must agree).
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
#include "kernels/draft_head.hpp"
#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "common/test.hpp"
#include "dsv41_fixture.hpp"
#include "engine/eager_engine.hpp"
#include "engine/verify_schedule.hpp"
#include "engine/graph_engine.hpp"
#include "engine/tp_bus.hpp"
#include "models/dsv41/config.hpp"
#include "models/dsv41/model.hpp"
#include "net/collective_bus.hpp"

namespace fs = std::filesystem;
using dgpp::BusBoundaryReducer;
using dgpp::Dsv41Model;
using dgpp::Dsv41Residency;
using dgpp::Dsv41TextConfig;
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
    if (!out[0]->start(&errors[0])) DGPP_LOG_ERROR("dsv41 engine world rank 0: {}", errors[0]);
  });
  std::vector<std::thread> connectors;
  for (int r = 1; r < world; ++r)
    connectors.emplace_back([&, r] {
      if (!out[static_cast<size_t>(r)]->start(&errors[static_cast<size_t>(r)]))
        DGPP_LOG_ERROR("dsv41 engine world rank {}: {}", r, errors[static_cast<size_t>(r)]);
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

std::vector<int64_t> smoke_tokens(const Dsv41TextConfig& cfg, int n, uint64_t seed) {
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
constexpr int kMaxTokens = 64;
constexpr int64_t kCache = 512;
constexpr int kWorld = 2;
constexpr uint16_t kPort = 29961;
// A world-1 top-2 logit margin under this is a near tie for the world-2
// engine (see Ref::am). Calibration: the folds reassociate in bf16 on the
// wire and the random-weight fixture amplifies the difference ~1.5x per
// layer (dsv41_tp_test: 6e-4 at layer 0, 1.3e-2 at layer 7, the fp4 cache
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
// (dsv41_tp_test gates the tensors layer-locally for that reason). A
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

Ref world1_reference(const Dsv41TextConfig& cfg, const std::string& dir, const std::vector<int64_t>& A,
                     const std::vector<int64_t>& B, const std::vector<int64_t>& C) {
  Dsv41Model m(cfg, dir, kMaxTokens, kCache, Dsv41Residency::Resident, nullptr, 0, 1, kSlots);
  std::vector<float> margins;
  EagerEngineAdapter<Dsv41Model> eng(&m, kSlots, margin_pick(cfg.vocab_size, &margins));
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

void rank_work(int r, const Dsv41TextConfig& cfg, const std::string& dir, const std::vector<int64_t>& A,
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
    Dsv41Model eager(cfg, dir, kMaxTokens, kCache, Dsv41Residency::Resident, &reducer, r, kWorld, kSlots);
    Dsv41Model graph(cfg, dir, kMaxTokens, kCache, Dsv41Residency::Resident, &reducer, r, kWorld,
                     kSlots,
                     /*mtp=*/false, /*decode_rows=*/0, /*serving_logits=*/true);
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                               sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld), cudaHostAllocDefault));
    arrive_once();
    EagerEngineAdapter<Dsv41Model> eager_engine(
        &eager, kSlots, dgpp::make_fabric_pick(bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms()));
    eager.set_prefill_bounded(true);
    require(!eager_engine.prefix_info().body_snapshots,
            "bounded prefill must not gain extra decoder spans");
    eager.set_prefill_bounded(false);
    require(eager_engine.prefix_info().body_snapshots, "exact prefill can snapshot existing cuts");
    GraphEngineAdapter<Dsv41Model> graph_engine(&graph, bus, r, kWorld, scratch, cfg.vocab_size,
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

void rank_work_mtp(int r, const Dsv41TextConfig& cfg, const std::string& dir, const std::vector<int64_t>& A,
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
    Dsv41Model eager(cfg, dir, kMaxTokens, kCache, Dsv41Residency::Resident, &reducer, r, kWorld, kSlots);
    Dsv41Model mtp(cfg, dir, kMaxTokens, kCache, Dsv41Residency::Resident, &reducer, r, kWorld,
                   kSlots,
                   /*mtp=*/true, decode_rows, /*serving_logits=*/true);
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                               sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld), cudaHostAllocDefault));
    arrive_once();
    EagerEngineAdapter<Dsv41Model> eager_engine(
        &eager, kSlots, dgpp::make_fabric_pick(bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms()));
    out->ea = solo(eager_engine, 0, A, kSteps);
    out->eb = solo(eager_engine, 1, B, kSteps);
    out->ec = solo(eager_engine, 2, C, kSteps);
    {
      GraphEngineAdapter<Dsv41Model> mtp_engine(&mtp, bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms(),
                                                /*batch_min_live=*/2, /*prefix_scratch=*/nullptr,
                                                /*gather_scratch=*/nullptr, /*candidates=*/0, /*grammar=*/nullptr,
                                                /*prefix_slots=*/0, /*mtp_depth=*/depth);
      mtp.set_prefill_bounded(true);
      require(!mtp_engine.prefix_info().body_snapshots,
              "bounded graph prefill keeps its decoder spans");
      mtp.set_prefill_bounded(false);
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

void rank_work_sched(int r, const Dsv41TextConfig& cfg, const std::string& dir, const std::vector<int64_t>& A,
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
    Dsv41Model eager(cfg, dir, kMaxTokens, kCache, Dsv41Residency::Resident, &reducer, r, kWorld, kSlots);
    Dsv41Model mtp(cfg, dir, kMaxTokens, kCache, Dsv41Residency::Resident, &reducer, r, kWorld,
                   sched_slots,
                   /*mtp=*/true, decode_rows, /*serving_logits=*/true);
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                               sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld), cudaHostAllocDefault));
    arrive_once();
    EagerEngineAdapter<Dsv41Model> eager_engine(
        &eager, kSlots, dgpp::make_fabric_pick(bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms()));
    out->ea = solo(eager_engine, 0, A, kSteps);
    out->eb = solo(eager_engine, 1, B, kSteps);
    out->ec = solo(eager_engine, 2, C, kSteps);
    {
      GraphEngineAdapter<Dsv41Model> eng(&mtp, bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms(),
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

void rank_work_wide(int r, const Dsv41TextConfig& cfg, const std::string& dir,
                    const std::vector<std::vector<int64_t>>& prompts, int slots, CollectiveBus* bus,
                    ConstructBarrier* barrier, WideOutcome* out) {
  bool arrived = false;
  const auto arrive_once = [&] {
    if (arrived) return;
    arrived = true;
    barrier->arrive_and_wait();
  };
  uint16_t* scratch = nullptr;
  try {
    const int depth = 4;
    const int decode_rows = slots * (1 + depth);  // 30 at six slots, 60 at twelve (the 64-row cap since 2026-10-23)
    // Six slots x two 128-token blocks each (the prompt and its decode,
    // plus the chain rows) need more pool than the three-slot worlds' 512.
    const int64_t cache = static_cast<int64_t>((slots + 2) / 2) * kCache;
    BusBoundaryReducer reducer(*bus, wait_timeout_ms());
    // Both models at the same decode rows: the GEMM lowering follows the
    // decode rows into short prefills (glm53-sixteen-row trap 1), so an
    // oracle at the default rows would prefill these prompts on another
    // kernel and diverge at near-ties.
    // The forward's rows hold the decode batch (sixteen slots x 5 = 80 > kMaxTokens).
    const int max_tokens = std::max(kMaxTokens, decode_rows);
    Dsv41Model eager(cfg, dir, max_tokens, cache, Dsv41Residency::Resident, &reducer, r, kWorld, slots, /*mtp=*/false,
                     decode_rows);
    Dsv41Model mtp(cfg, dir, max_tokens, cache, Dsv41Residency::Resident, &reducer, r, kWorld,
                   slots,
                   /*mtp=*/true, decode_rows, /*serving_logits=*/true);
    require(mtp.max_decode_rows() == decode_rows, "the model takes the slots x 5 decode batch");
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                               sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld), cudaHostAllocDefault));
    arrive_once();
    EagerEngineAdapter<Dsv41Model> eager_engine(
        &eager, slots, dgpp::make_fabric_pick(bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms()));
    for (int q = 0; q < slots; ++q) out->eager.push_back(solo(eager_engine, q, prompts[static_cast<size_t>(q)], kSteps));
    {
      GraphEngineAdapter<Dsv41Model> eng(&mtp, bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms(),
                                         /*batch_min_live=*/2, nullptr, nullptr, 0, nullptr, 0, /*mtp_depth=*/depth);
      out->families = eng.batch_families();
      eng.configure_verify_schedule(true, 9.0f, dgpp::verify_reservation_lambda(20.0f, 9.0f), 1);
      // One depth per step for every slot (the batch takes the deepest
      // slot's, so per-slot variety would always reach the full block).
      int bstep = 0;
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
      out->batch_hist = eng.verify_depth_histogram_batch(fam6);
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

void check_wide_world(uint16_t port, int slots) {
  const Dsv41TextConfig cfg = dsv41fx::tiny_config();
  const std::string dir = (fs::current_path() / "dsv41_engine_fixture").string();
  dsv41fx::write_fixture(cfg, dir);
  (void)dsv41fx::fixture_sidecar(cfg, dir);
  std::vector<std::vector<int64_t>> prompts;
  const uint64_t seeds[6] = {0x9E3779B97F4A7C15ull, 0xD1B54A32D192ED03ull, 0x2545F4914F6CDD1Dull,
                             0x3C6EF372FE94F82Bull, 0x1F83D9ABFB41BD6Bull, 0x5BE0CD19137E2179ull};
  for (int q = 0; q < slots; ++q)
    prompts.push_back(smoke_tokens(cfg, 11 + 2 * (q % 6), seeds[q % 6] + 0x1000003ull * static_cast<uint64_t>(q / 6)));
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, port);
  require(!buses.empty(), "the loopback bus world failed to start");
  std::vector<WideOutcome> outs(kWorld);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int r = 0; r < kWorld; ++r)
    workers.emplace_back(rank_work_wide, r, std::cref(cfg), std::cref(dir), std::cref(prompts), slots,
                         buses[static_cast<size_t>(r)].get(), &barrier, &outs[static_cast<size_t>(r)]);
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r) require(outs[static_cast<size_t>(r)].error.empty(), outs[static_cast<size_t>(r)].error);
  for (int r = 1; r < kWorld; ++r)
    require(outs[static_cast<size_t>(r)].wide == outs[0].wide, "the ranks' six-slot transcripts differ");
  const WideOutcome& o = outs[0];
  std::string hist;
  for (size_t i = 0; i < o.batch_hist.size(); ++i) hist += (i ? " " : "") + std::to_string(o.batch_hist[i]);
  DGPP_LOG_INFO("world 2 {} slots x depth 4 ({} rows): families {} | {} batched steps, widest family replays per depth option [{}]",
                slots, slots * 5, ids_text(o.families), o.steps6, hist);
  if (slots == 6)
    require(o.families == std::vector<int>{2, 3, 4, 6}, "the six-slot world builds the 2/3/4/6 families");
  else
    require(!o.families.empty() && o.families.back() == slots, "the wide world's widest family is the slot count");
  for (int q = 0; q < slots; ++q)
    require(o.wide[static_cast<size_t>(q)] == o.eager[static_cast<size_t>(q)],
            "slot " + std::to_string(q) + "'s six-slot batched transcript differs from the plain eager engine's");
  uint64_t total = 0;
  int used = 0;
  for (const uint64_t h : o.batch_hist) {
    total += h;
    used += h > 0 ? 1 : 0;
  }
  require(total == o.steps6 && used >= 2, "the 6-slot family replayed at more than one scheduled depth");
}

void check_sched_world(uint16_t port) {
  const Dsv41TextConfig cfg = dsv41fx::tiny_config();
  const std::string dir = (fs::current_path() / "dsv41_engine_fixture").string();
  dsv41fx::write_fixture(cfg, dir);
  (void)dsv41fx::fixture_sidecar(cfg, dir);
  const std::vector<int64_t> A = smoke_tokens(cfg, 23, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 17, 0xD1B54A32D192ED03ull);
  const std::vector<int64_t> C = smoke_tokens(cfg, 11, 0x2545F4914F6CDD1Dull);
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
  const Dsv41TextConfig cfg = dsv41fx::tiny_config();
  const std::string dir = (fs::current_path() / "dsv41_engine_fixture").string();
  dsv41fx::write_fixture(cfg, dir);
  (void)dsv41fx::fixture_sidecar(cfg, dir);
  const std::vector<int64_t> A = smoke_tokens(cfg, 23, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 17, 0xD1B54A32D192ED03ull);
  const std::vector<int64_t> C = smoke_tokens(cfg, 11, 0x2545F4914F6CDD1Dull);
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

DGPP_TEST(dsv41_engines_loopback_world_2_dspark_block_matches_plain_decode) { check_mtp_world(5, kPort + 1); }

DGPP_TEST(dsv41_engines_loopback_world_2_dspark_depth_2_matches_plain_decode) { check_mtp_world(2, kPort + 2); }

DGPP_TEST(dsv41_engines_loopback_world_2_scheduled_verify_depth_is_exact) { check_sched_world(kPort + 3); }

DGPP_TEST(dsv41_engines_loopback_world_2_six_slots_at_depth_4_batched_scheduled_is_exact) { check_wide_world(kPort + 4, 6); }
// Twelve slots x depth 4 = 60 rows: the family's 64-row decode batch
// (2026-10-23) — the same gates as the six-slot world at its widest.
DGPP_TEST(dsv41_engines_loopback_world_2_twelve_slots_at_depth_4_batched_scheduled_is_exact) {
  check_wide_world(kPort + 12, 12);  // 29973: the dsv41 TP test holds kPort + 5
}
// Sixteen slots x depth 4 = 80 rows: the family's 96-row decode batch
// (2026-10-23 pm; the cap left 64 for sixteen five-row draft blocks) —
// the long expert runs cut into 64-row groups, the capped family unused.
DGPP_TEST(dsv41_engines_loopback_world_2_sixteen_slots_at_depth_4_batched_scheduled_is_exact) {
  check_wide_world(kPort + 13, 16);  // 29974
}

DGPP_TEST(dsv41_engines_loopback_world_2_graph_matches_eager) {
  const Dsv41TextConfig cfg = dsv41fx::tiny_config();
  const std::string dir = (fs::current_path() / "dsv41_engine_fixture").string();
  dsv41fx::write_fixture(cfg, dir);
  (void)dsv41fx::fixture_sidecar(cfg, dir);
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

int main() {
  // DGPP_TEST_DRAFT_SCREEN=N: every case drafts through the head screen
  // (kernels/draft_head.hpp) with an N-block candidate cap; every gate
  // holds unchanged (the verify reads the exact head; only drafts move).
  if (const char* d = std::getenv("DGPP_TEST_DRAFT_SCREEN"); d && d[0] != '\0')
    dgpp::draft_head_configure(dgpp::DraftHeadSettings{std::atoi(d)});
  return dgpp::test::run_all();
}
