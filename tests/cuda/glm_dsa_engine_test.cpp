// The full GLM-5.3 decode engines over the engine core (docs/glm53_plan.md
// G5): EagerEngineAdapter<GlmDsaModel> and GraphEngineAdapter<GlmDsaModel>
// on the fixture, a loopback world of 2 (real verbs QPs over 127.0.0.1)
// against the world-1 eager engine. The decode batch is capped at eight
// rows (plan D9), so the depth-2 world runs two slots (2 x 3 rows).
//
// Gates: the graph engine's SCALAR replays (the device-driven T=1 graph:
// device positions, the recorded commit, the pinned token upload) and its
// ROW-BATCHED replays (the fixed slot-major batch off the persistent
// feeds, closed slots padding at -1, the 2-slot and full families) produce
// the eager engine's transcripts at the same world exactly (the recorded
// kernels are the eager kernels); the world-2 eager transcripts follow the
// world-1 ones (reported; the folds reassociate, so a near tie may flip a
// late token — the first tokens must agree).
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
#include "engine/tp_bus.hpp"
#include "models/glm_dsa/config.hpp"
#include "models/glm_dsa/model.hpp"
#include "net/collective_bus.hpp"
#include "glm_dsa_fixture.hpp"
#include "engine/eager_engine.hpp"
#include "engine/graph_engine.hpp"
#include "engine/verify_schedule.hpp"
#include "engine_test_ties.hpp"
#include "engine/tp_bus.hpp"

namespace fs = std::filesystem;
using dgpp::BusBoundaryReducer;
using dgpp::EagerEngineAdapter;
using dgpp::GraphEngineAdapter;
using dgpp::GlmDsaModel;
using dgpp::GlmDsaResidency;
using dgpp::GlmDsaTextConfig;
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
  o.lat_slot_bytes = 262144;  // every fold on the latency path (glm4_tp_test's note)
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
    if (!out[0]->start(&errors[0])) DGPP_LOG_ERROR("glm_dsa engine world rank 0: {}", errors[0]);
  });
  std::vector<std::thread> connectors;
  for (int r = 1; r < world; ++r)
    connectors.emplace_back([&, r] {
      if (!out[static_cast<size_t>(r)]->start(&errors[static_cast<size_t>(r)]))
        DGPP_LOG_ERROR("glm_dsa engine world rank {}: {}", r, errors[static_cast<size_t>(r)]);
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

std::vector<int64_t> smoke_tokens(const GlmDsaTextConfig& cfg, int n, uint64_t seed) {
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
constexpr int kDepth2Slots = 2;  // 2 x 3 verify rows within the 8-row cap
constexpr int kSteps = 60;
constexpr int kMaxTokens = 64;
constexpr int64_t kCache = 512;
constexpr int kWorld = 2;
constexpr uint16_t kPort = 29954;  // 29954 (graph vs eager), 29955 (mtp), 29956 (depth 2), 29957 (sharded), 29960 (sixteen rows), 29976 (nf4i8 graph vs eager)


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

Ref world1_reference(const GlmDsaTextConfig& cfg, const std::string& dir, const std::vector<int64_t>& A,
                     const std::vector<int64_t>& B, const std::vector<int64_t>& C) {
  GlmDsaModel m(cfg, dir, kMaxTokens, kCache, GlmDsaResidency::Resident, nullptr, 0, 1, kSlots);
  std::vector<float> margins;
  EagerEngineAdapter<GlmDsaModel> eng(&m, kSlots, engine_ties::margin_pick(cfg.vocab_size, &margins));
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
  std::vector<int32_t> ma, mb, mc;   // the MTP graph engine: scalar A, batched B and C
  int mtp_steps_a = 0;               // scalar MTP steps A took (< kSteps: drafts stood)
};

void rank_work(int r, const GlmDsaTextConfig& cfg, const std::string& dir, const std::vector<int64_t>& A,
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
    GlmDsaModel eager(cfg, dir, kMaxTokens, kCache, GlmDsaResidency::Resident, &reducer, r, kWorld, kSlots);
    GlmDsaModel graph(
        cfg, dir, kMaxTokens, kCache, GlmDsaResidency::Resident, &reducer, r, kWorld, kSlots,
        /*mtp=*/false, /*decode_rows=*/0, dgpp::LatentFormat::kBf16, /*serving_logits=*/true);
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                               sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld), cudaHostAllocDefault));
    arrive_once();
    EagerEngineAdapter<GlmDsaModel> eager_engine(
        &eager, kSlots, dgpp::make_fabric_pick(bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms()));
    GraphEngineAdapter<GlmDsaModel> graph_engine(&graph, bus, r, kWorld, scratch, cfg.vocab_size,
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

// The MTP graph engine (the one-graph draft) in its own bus world (one
// graph era per bus): the T=2 verify with the recorded commit, the in-graph
// draft and its pick, the [next, draft] feed — scalar for A, then the
// batched families for B and C — against the plain eager engine.
void rank_work_mtp(int r, const GlmDsaTextConfig& cfg, const std::string& dir, const std::vector<int64_t>& A,
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
    GlmDsaModel eager(cfg, dir, kMaxTokens, kCache, GlmDsaResidency::Resident, &reducer, r, kWorld, kSlots);
    GlmDsaModel mtp(
        cfg, dir, kMaxTokens, kCache, GlmDsaResidency::Resident, &reducer, r, kWorld, kSlots,
        /*mtp=*/true, /*decode_rows=*/0, dgpp::LatentFormat::kBf16, /*serving_logits=*/true);
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                               sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld), cudaHostAllocDefault));
    arrive_once();
    EagerEngineAdapter<GlmDsaModel> eager_engine(
        &eager, kSlots, dgpp::make_fabric_pick(bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms()));
    out->ea = solo(eager_engine, 0, A, kSteps);
    out->eb = solo(eager_engine, 1, B, kSteps);
    out->ec = solo(eager_engine, 2, C, kSteps);
    {
      GraphEngineAdapter<GlmDsaModel> mtp_engine(&mtp, bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms(),
                                               /*batch_min_live=*/2);
      out->ma.push_back(mtp_engine.prefill(0, A));
      mtp_engine.reserve(0, static_cast<int64_t>(A.size()) + kSteps + 2);
      while (out->ma.size() < static_cast<size_t>(kSteps) + 1) {
        const std::vector<int32_t> t = mtp_engine.step(0);
        require(!t.empty() && t.size() <= 2, "mtp step shape");
        out->ma.insert(out->ma.end(), t.begin(), t.end());
        ++out->mtp_steps_a;
      }
      out->ma.resize(static_cast<size_t>(kSteps) + 1);
      mtp_engine.close(0);
      out->mb.push_back(mtp_engine.prefill(1, B));
      mtp_engine.reserve(1, static_cast<int64_t>(B.size()) + kSteps + 2);
      out->mc.push_back(mtp_engine.prefill(2, C));
      mtp_engine.reserve(2, static_cast<int64_t>(C.size()) + kSteps + 2);
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

// Sixteen decode rows (2026-09-13, the batch's cap lifted from eight):
// eight slots at depth 1, all live, on the every-slot family — the select's
// two row groups, the pick's sixteen request slots and the sixteen-row
// folds in one world. Each slot's batched transcript is the eager engine's,
// bitwise, with BOTH models configured for the same decode rows: the GEMM
// interface lowers every bf16 call up to the model's decode rows to the
// row-independent GEMV chain (kernels/gemm.hpp), so a prompt of 9–16 tokens
// prefills through it on a sixteen-row model and through cuBLASLt on an
// eight-row one — last-bit differences that flip a near tie thirty tokens
// later (found 2026-09-13 with the eager model at eight rows; the
// DGPP_WIDE_PREFILL_PROBE=1 aid prints which prompts' prefill logits agree).
// DGPP_WIDE_SLOTS=n and DGPP_WIDE_DEPTH=d reshape the gate (n slots at
// depth d; the bus allows 2n + 2 x families <= 32 variants), and
// DGPP_WIDE_ROTATE=k puts prompt k in slot 0.
constexpr int kWideSlotsMax = 16;
constexpr int64_t kWideCache = 4096;  // sixteen sessions x up to three 64-token blocks
inline int wide_slots() {
  const char* v = std::getenv("DGPP_WIDE_SLOTS");
  const int n = v ? std::atoi(v) : 8;
  return n < 2 ? 2 : n > kWideSlotsMax ? kWideSlotsMax : n;
}
inline int wide_depth() {
  const char* v = std::getenv("DGPP_WIDE_DEPTH");
  const int d = v ? std::atoi(v) : 1;
  return d < 1 ? 1 : d > 3 ? 3 : d;
}
struct WideOutcome {
  std::vector<std::vector<int32_t>> eager, batched;
  uint64_t full_steps = 0;
  std::string error;
};
void rank_work_wide(int r, const GlmDsaTextConfig& cfg, const std::string& dir,
                    const std::vector<std::vector<int64_t>>& prompts, CollectiveBus* bus, ConstructBarrier* barrier,
                    WideOutcome* out) {
  bool arrived = false;
  const auto arrive_once = [&] {
    if (arrived) return;
    arrived = true;
    barrier->arrive_and_wait();
  };
  const int slots = wide_slots();
  const int rows_per_slot = 1 + wide_depth();
  const int decode_rows = slots * rows_per_slot;
  uint16_t* scratch = nullptr;
  try {
    BusBoundaryReducer reducer(*bus, wait_timeout_ms());
    GlmDsaModel eager(cfg, dir, kMaxTokens, kWideCache, GlmDsaResidency::Resident, &reducer, r, kWorld, slots,
                      /*mtp=*/false, decode_rows);
    GlmDsaModel mtp(cfg, dir, kMaxTokens, kWideCache, GlmDsaResidency::Resident, &reducer, r,
                    kWorld, slots,
                    /*mtp=*/true, decode_rows, dgpp::LatentFormat::kBf16, /*serving_logits=*/true);
    require(mtp.max_decode_rows() == std::max(8, decode_rows), "the slots make the batch");
    if (std::getenv("DGPP_WIDE_PREFILL_PROBE")) {
      for (int i = 0; i < slots; ++i) {
        const GlmDsaModel::Outputs a = eager.session_prefill(0, prompts[static_cast<size_t>(i)]);
        eager.session_close(0);
        const GlmDsaModel::Outputs b = mtp.session_prefill(0, prompts[static_cast<size_t>(i)]);
        mtp.session_close(0);
        size_t first = a.logits.size();
        for (size_t k = 0; k < a.logits.size() && k < b.logits.size(); ++k)
          if (a.logits[k] != b.logits[k]) { first = k; break; }
        DGPP_LOG_INFO("rank {} prefill probe: prompt {} ({} tokens): logits {} ({} elems, first difference at {})", r, i,
                      prompts[static_cast<size_t>(i)].size(), first == a.logits.size() ? "bitwise" : "DIFFER",
                      a.logits.size(), first);
      }
    }
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                               sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld), cudaHostAllocDefault));
    arrive_once();
    EagerEngineAdapter<GlmDsaModel> eager_engine(
        &eager, slots, dgpp::make_fabric_pick(bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms()));
    for (int i = 0; i < slots; ++i) out->eager.push_back(solo(eager_engine, i, prompts[static_cast<size_t>(i)], kSteps));
    {
      GraphEngineAdapter<GlmDsaModel> mtp_engine(&mtp, bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms(),
                                               /*batch_min_live=*/2, nullptr, nullptr, dgpp::kSamplingCandidates, nullptr, 0,
                                               /*mtp_depth=*/wide_depth());
      out->batched.assign(static_cast<size_t>(slots), {});
      std::vector<int> reqs;
      for (int i = 0; i < slots; ++i) {
        const std::vector<int64_t>& p = prompts[static_cast<size_t>(i)];
        out->batched[static_cast<size_t>(i)].push_back(mtp_engine.prefill(i, p));
        // Room for a run that accepts every draft (rows_per_slot tokens a step).
        mtp_engine.reserve(i, static_cast<int64_t>(p.size()) + rows_per_slot * kSteps + 2);
        reqs.push_back(i);
      }
      const auto all_done = [&] {
        for (const auto& t : out->batched)
          if (t.size() < static_cast<size_t>(kSteps) + 1) return false;
        return true;
      };
      while (!all_done()) {
        const auto t = mtp_engine.step_batch(reqs);
        require(t.size() == static_cast<size_t>(slots), "wide batch step shape");
        for (int i = 0; i < slots; ++i)
          out->batched[static_cast<size_t>(i)].insert(out->batched[static_cast<size_t>(i)].end(),
                                                      t[static_cast<size_t>(i)].begin(), t[static_cast<size_t>(i)].end());
      }
      for (auto& t : out->batched) t.resize(static_cast<size_t>(kSteps) + 1);
      // The every-slot family is the last (graph_engine.hpp).
      out->full_steps = mtp_engine.batch_family_steps(static_cast<int>(mtp_engine.batch_families().size()) - 1);
      for (int i = 0; i < slots; ++i) mtp_engine.close(i);
      mtp_engine.drain();
    }
    cudaFreeHost(scratch);
  } catch (const std::exception& e) {
    if (scratch) cudaFreeHost(scratch);
    out->error = "rank " + std::to_string(r) + ": " + e.what();
    arrive_once();
  }
}

}  // namespace

DGPP_TEST(glm_dsa_engines_loopback_world_2_sixteen_row_batch_matches_eager) {
  const GlmDsaTextConfig cfg = glmdsafx::tiny_config();
  const std::string dir = "glm_dsa_engine_fixture";
  glmdsafx::write_fixture(cfg, dir);
  const int slots = wide_slots();
  std::vector<std::vector<int64_t>> prompts;
  const int lengths[kWideSlotsMax] = {23, 17, 11, 19, 13, 21, 15, 9, 25, 14, 10, 18, 12, 20, 16, 22};
  for (int i = 0; i < kWideSlotsMax; ++i)
    prompts.push_back(smoke_tokens(cfg, lengths[i], 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull * static_cast<uint64_t>(i + 1)));
  if (const char* rot = std::getenv("DGPP_WIDE_ROTATE"))
    std::rotate(prompts.begin(), prompts.begin() + (std::atoi(rot) % kWideSlotsMax), prompts.end());
  prompts.resize(static_cast<size_t>(slots));
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, 29960);
  require(!buses.empty(), "the loopback bus world failed to start");
  std::vector<WideOutcome> outs(kWorld);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int r = 0; r < kWorld; ++r)
    workers.emplace_back(rank_work_wide, r, std::cref(cfg), std::cref(dir), std::cref(prompts),
                         buses[static_cast<size_t>(r)].get(), &barrier, &outs[static_cast<size_t>(r)]);
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r) require(outs[static_cast<size_t>(r)].error.empty(), outs[static_cast<size_t>(r)].error);
  for (int r = 1; r < kWorld; ++r)
    require(outs[static_cast<size_t>(r)].eager == outs[0].eager && outs[static_cast<size_t>(r)].batched == outs[0].batched,
            "the ranks' transcripts differ");
  const WideOutcome& o = outs[0];
  DGPP_LOG_INFO("world 2 {}-row batch ({} slots x {} rows): {} every-slot steps; slot 0 eager {} | batched {}",
                slots * (1 + wide_depth()), slots, 1 + wide_depth(), o.full_steps, ids_text(o.eager[0]), ids_text(o.batched[0]));
  require(o.full_steps > 0, "the every-slot family replayed");
  // The world-1 reference's margins per prompt (the near-tie rule's evidence).
  std::vector<std::vector<float>> margins(static_cast<size_t>(slots));
  {
    GlmDsaModel w1(cfg, dir, kMaxTokens, kWideCache, GlmDsaResidency::Resident, nullptr, 0, 1, slots);
    std::vector<float> m;
    EagerEngineAdapter<GlmDsaModel> eng(&w1, slots, engine_ties::margin_pick(cfg.vocab_size, &m));
    for (int i = 0; i < slots; ++i) {
      (void)solo(eng, i, prompts[static_cast<size_t>(i)], kSteps);
      margins[static_cast<size_t>(i)] = m;
      m.clear();
    }
  }
  int differ = 0;
  for (int i = 0; i < slots; ++i) {
    const auto& e = o.eager[static_cast<size_t>(i)];
    const auto& b = o.batched[static_cast<size_t>(i)];
    if (b != e) {
      ++differ;
      DGPP_LOG_INFO("slot {} (prompt {} tokens): batched differs from eager at index {}/{}: eager {} | batched {}", i,
                    prompts[static_cast<size_t>(i)].size(), agreeing_prefix(b, e), e.size(), ids_text(e), ids_text(b));
    }
    engine_ties::require_agrees_or_tie(b, e, margins[static_cast<size_t>(i)], kTieMargin, kAgreePositions,
                                       ("slot " + std::to_string(i) + " of the wide batch").c_str());
  }
  DGPP_LOG_INFO("wide batch: {} of {} slots differ from the eager engine past the near-tie rule's positions", differ, slots);
}

// Depth 2: the T=3 verify with two drafts, the second off the
// block's chain row (its own output row as the hidden, the first draft's
// pick as the token), in its own bus world — A on the scalar graph, then B
// and C together on the ROW-BATCHED graph (the batched chain: every slot's
// chain row in one draft-block run). The decode batch is capped at eight
// rows (the fused DSA select, plan D9), so this world runs two slots at
// 3 rows each (6 rows: slots 1 and 2 share the 2-slot family). The
// transcripts must be the plain eager engine's exactly; the pass count
// reports the second draft's worth.
void rank_work_mtp_depth2(int r, const GlmDsaTextConfig& cfg, const std::string& dir, const std::vector<int64_t>& A,
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
    GlmDsaModel eager(cfg, dir, kMaxTokens, kCache, GlmDsaResidency::Resident, &reducer, r, kWorld, kSlots);
    // The decode-row ceiling of the depth-2 shape: kDepth2Slots x 3 rows = 6
    // (the model's cap is 8, plan D9).
    GlmDsaModel mtp(cfg, dir, kMaxTokens, kCache, GlmDsaResidency::Resident, &reducer, r, kWorld,
                    kDepth2Slots,
                    /*mtp=*/true, /*decode_rows=*/kDepth2Slots * 3, dgpp::LatentFormat::kBf16,
                    /*serving_logits=*/true);
    require(mtp.max_decode_rows() == 8, "the model floors the decode rows at eight (the batch's cap)");
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                               sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld), cudaHostAllocDefault));
    arrive_once();
    EagerEngineAdapter<GlmDsaModel> eager_engine(
        &eager, kSlots, dgpp::make_fabric_pick(bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms()));
    out->ea = solo(eager_engine, 0, A, kSteps);
    out->eb = solo(eager_engine, 1, B, kSteps);
    out->ec = solo(eager_engine, 2, C, kSteps);
    {
      GraphEngineAdapter<GlmDsaModel> d2(&mtp, bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms(),
                                       /*batch_min_live=*/2, nullptr, nullptr, dgpp::kSamplingCandidates, nullptr, 0,
                                       /*mtp_depth=*/2);
      require(!d2.batch_families().empty() && d2.batch_families().front() == 2,
              "the depth-2 world builds the 2-slot batch family over the 8-row ceiling");
      // A alone: the scalar depth-2 graph.
      out->ma.push_back(d2.prefill(0, A));
      d2.reserve(0, static_cast<int64_t>(A.size()) + kSteps + 3);
      while (out->ma.size() < static_cast<size_t>(kSteps) + 1) {
        const std::vector<int32_t> t = d2.step(0);
        require(!t.empty() && t.size() <= 3, "mtp depth-2 step shape");
        out->ma.insert(out->ma.end(), t.begin(), t.end());
        ++out->mtp_steps_a;
      }
      out->ma.resize(static_cast<size_t>(kSteps) + 1);
      d2.close(0);
      // B and C together: the row-batched depth-2 graph (the two slots of
      // this world form the 2-slot family).
      out->mb.push_back(d2.prefill(0, B));
      d2.reserve(0, static_cast<int64_t>(B.size()) + kSteps + 3);
      out->mc.push_back(d2.prefill(1, C));
      d2.reserve(1, static_cast<int64_t>(C.size()) + kSteps + 3);
      while (out->mb.size() < static_cast<size_t>(kSteps) + 1 || out->mc.size() < static_cast<size_t>(kSteps) + 1) {
        const auto t = d2.step_batch({0, 1});
        require(t.size() == 2 && !t[0].empty() && t[0].size() <= 3 && !t[1].empty() && t[1].size() <= 3,
                "mtp depth-2 batch step shape");
        out->mb.insert(out->mb.end(), t[0].begin(), t[0].end());
        out->mc.insert(out->mc.end(), t[1].begin(), t[1].end());
      }
      out->mb.resize(static_cast<size_t>(kSteps) + 1);
      out->mc.resize(static_cast<size_t>(kSteps) + 1);
      d2.close(0);
      d2.close(1);
      d2.drain();
      require(d2.batch_family_steps(0) > 0, "the batched depth-2 steps ran on the 2-slot family");
    }
    cudaFreeHost(scratch);
  } catch (const std::exception& e) {
    if (scratch) cudaFreeHost(scratch);
    out->error = "rank " + std::to_string(r) + ": " + e.what();
    arrive_once();
  }
}

// The scheduled verify depth (engine/verify_schedule.hpp) on the full
// GLM-5.3 (2026-10-13): this family has no confidence head, so the graph
// engine takes the draft head's own probability of each draft from the
// device sampler, as GLM-4.7's gate does. A forced per-step depth (so
// every reduced variant replays, scalar and batched) and then the engine's
// own policy; the scheduled transcripts must be the plain eager engine's
// (A bitwise, B and C under the near-tie rule), the histograms must count
// every replay.
struct SchedOutcome {
  std::string error;
  std::vector<int32_t> ea, eb, ec;
  std::vector<int32_t> sa, sb, sc;
  std::vector<int> options;
  std::vector<uint64_t> hist, batch_hist;
  int steps_a = 0, steps_batch = 0, steps_b_alone = 0;
  int first_alone_option = -1;  // the option B's first step alone replayed
};

void rank_work_sched(int r, const GlmDsaTextConfig& cfg, const std::string& dir, const std::vector<int64_t>& A,
                     const std::vector<int64_t>& B, const std::vector<int64_t>& C, CollectiveBus* bus,
                     ConstructBarrier* barrier, SchedOutcome* out, int max_live = 0, int max_options = 0) {
  bool arrived = false;
  const auto arrive_once = [&] {
    if (arrived) return;
    arrived = true;
    barrier->arrive_and_wait();
  };
  uint16_t* scratch = nullptr;
  uint16_t* prefix_scratch = nullptr;
  uint16_t* gather_scratch = nullptr;
  try {
    BusBoundaryReducer reducer(*bus, wait_timeout_ms());
    GlmDsaModel eager(cfg, dir, kMaxTokens, kCache, GlmDsaResidency::Resident, &reducer, r, kWorld, kSlots);
    GlmDsaModel mtp(cfg, dir, kMaxTokens, kCache, GlmDsaResidency::Resident, &reducer, r, kWorld, kSlots,
                    /*mtp=*/true, /*decode_rows=*/kSlots * 3, dgpp::LatentFormat::kBf16, /*serving_logits=*/true);
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                               sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld), cudaHostAllocDefault));
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&prefix_scratch),
                               sizeof(uint16_t) * std::max<size_t>(dgpp::fabric_sampling_prefix_scratch_elems(kWorld), 2),
                               cudaHostAllocDefault));
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&gather_scratch),
                               sizeof(uint16_t) * std::max<size_t>(dgpp::sampling_gather_scratch_elems(cfg.vocab_size), 2),
                               cudaHostAllocDefault));
    arrive_once();
    EagerEngineAdapter<GlmDsaModel> eager_engine(
        &eager, kSlots, dgpp::make_fabric_pick(bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms()));
    out->ea = solo(eager_engine, 0, A, kSteps);
    out->eb = solo(eager_engine, 1, B, kSteps);
    out->ec = solo(eager_engine, 2, C, kSteps);
    {
      const int depth = 2;
      GraphEngineAdapter<GlmDsaModel> eng(&mtp, bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms(),
                                          /*batch_min_live=*/2, prefix_scratch, gather_scratch,
                                          dgpp::kSamplingCandidates, nullptr, 0, /*mtp_depth=*/depth);
      require(eng.supports_sampling(), "the world has the device sampler (the draft probabilities' source)");
      eng.configure_verify_schedule(true, /*row_ms=*/12.0f, dgpp::verify_reservation_lambda(50.0f, 12.0f), 1,
                                    /*base_ms=*/0.f, /*adapt=*/false, max_options, max_live);
      out->options = eng.verify_depth_options();
      int step = 0;
      eng.set_verify_depth_hook([&](int, int, const float*, int) { return 1 + (step++ % 2); });
      out->sa.push_back(eng.prefill(0, A));
      eng.reserve(0, static_cast<int64_t>(A.size()) + kSteps + 3);
      while (out->sa.size() < static_cast<size_t>(kSteps) + 1) {
        const std::vector<int32_t> t = eng.step(0);
        require(!t.empty() && t.size() <= 3, "sched step shape");
        out->sa.insert(out->sa.end(), t.begin(), t.end());
        ++out->steps_a;
      }
      out->sa.resize(static_cast<size_t>(kSteps) + 1);
      eng.close(0);
      // B (slot 1) and C (slot 2) in the batch under forced per-slot depths:
      // the batch takes the deeper, so both batch variants replay.
      int bstep = 0;
      eng.set_verify_depth_hook([&](int req, int, const float*, int) {
        const int k = req == 1 ? 1 + (bstep % 2) : 1 + ((bstep / 2) % 2);
        if (req == 2) ++bstep;
        return k;
      });
      out->sb.push_back(eng.prefill(1, B));
      eng.reserve(1, static_cast<int64_t>(B.size()) + kSteps + 3);
      out->sc.push_back(eng.prefill(2, C));
      eng.reserve(2, static_cast<int64_t>(C.size()) + kSteps + 3);
      const size_t half = static_cast<size_t>(kSteps) / 2 + 1;
      while (out->sb.size() < half || out->sc.size() < half) {
        const auto t = eng.step_batch({1, 2});
        require(t.size() == 2, "sched batch step shape");
        out->sb.insert(out->sb.end(), t[0].begin(), t[0].end());
        out->sc.insert(out->sc.end(), t[1].begin(), t[1].end());
        ++out->steps_batch;
      }
      out->batch_hist = eng.verify_depth_histogram_batch(1);
      eng.close(2);
      // B alone: the first step forced to the whole block right after the
      // reduced batch, then the engine's own policy over the published
      // draft probabilities.
      bool first_alone = true;
      eng.set_verify_depth_hook([&](int, int suggested, const float*, int) {
        const int k = first_alone ? depth : suggested;
        first_alone = false;
        return k;
      });
      while (out->sb.size() < static_cast<size_t>(kSteps) + 1) {
        const std::vector<uint64_t> before = eng.verify_depth_histogram();
        const std::vector<int32_t> t = eng.step(1);
        if (out->steps_b_alone == 0) {
          const std::vector<uint64_t> after = eng.verify_depth_histogram();
          for (size_t i = 0; i < after.size(); ++i)
            if (after[i] != before[i]) out->first_alone_option = static_cast<int>(i);
        }
        require(!t.empty() && t.size() <= 3, "sched post-batch step shape");
        out->sb.insert(out->sb.end(), t.begin(), t.end());
        ++out->steps_b_alone;
      }
      out->sb.resize(static_cast<size_t>(kSteps) + 1);
      eng.close(1);
      eng.drain();
      out->hist = eng.verify_depth_histogram();
    }
    cudaFreeHost(scratch);
    cudaFreeHost(prefix_scratch);
    cudaFreeHost(gather_scratch);
  } catch (const std::exception& e) {
    if (scratch) cudaFreeHost(scratch);
    if (prefix_scratch) cudaFreeHost(prefix_scratch);
    if (gather_scratch) cudaFreeHost(gather_scratch);
    out->error = "rank " + std::to_string(r) + ": " + e.what();
    arrive_once();
  }
}

DGPP_TEST(glm_dsa_engines_loopback_world_2_scheduled_verify_depth_from_draft_probabilities_is_exact) {
  const GlmDsaTextConfig cfg = glmdsafx::tiny_config();
  const std::string dir = "glm_dsa_engine_fixture";
  glmdsafx::write_fixture(cfg, dir);
  const std::vector<int64_t> A = smoke_tokens(cfg, 23, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 17, 0xD1B54A32D192ED03ull);
  const std::vector<int64_t> C = smoke_tokens(cfg, 11, 0x2545F4914F6CDD1Dull);
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, kPort + 8);
  require(!buses.empty(), "the loopback bus world failed to start");
  std::vector<SchedOutcome> outs(kWorld);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int r = 0; r < kWorld; ++r)
    workers.emplace_back(rank_work_sched, r, std::cref(cfg), std::cref(dir), std::cref(A), std::cref(B), std::cref(C),
                         buses[static_cast<size_t>(r)].get(), &barrier, &outs[static_cast<size_t>(r)], /*max_live=*/0, /*max_options=*/0);
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r) require(outs[static_cast<size_t>(r)].error.empty(), outs[static_cast<size_t>(r)].error);
  for (int r = 1; r < kWorld; ++r)
    require(outs[static_cast<size_t>(r)].sa == outs[0].sa && outs[static_cast<size_t>(r)].sb == outs[0].sb &&
                outs[static_cast<size_t>(r)].sc == outs[0].sc && outs[static_cast<size_t>(r)].hist == outs[0].hist,
            "the ranks' scheduled transcripts or depths differ");
  const Ref ref = world1_reference(cfg, dir, A, B, C);
  const SchedOutcome& o = outs[0];
  std::string hist, bhist;
  for (size_t i = 0; i < o.hist.size(); ++i)
    hist += (i ? " " : "") + std::to_string(o.options[i]) + ":" + std::to_string(o.hist[i]);
  for (size_t i = 0; i < o.batch_hist.size(); ++i)
    bhist += (i ? " " : "") + std::to_string(o.options[i]) + ":" + std::to_string(o.batch_hist[i]);
  DGPP_LOG_INFO("world 2 scheduled depth (draft probabilities): options {} | A {} ({} steps) | B {} | C {} | scalar [{}] | batched [{}]",
                ids_text(o.options), ids_text(o.sa), o.steps_a, ids_text(o.sb), ids_text(o.sc), hist, bhist);
  require(o.options == std::vector<int>{1, 2}, "depth 2 offers the two options");
  require(o.sa == o.ea, "the scheduled scalar transcript of A differs from the plain eager engine's");
  engine_ties::require_agrees_or_tie(o.sb, o.eb, ref.bm, kTieMargin, kAgreePositions, "the scheduled transcript of B");
  engine_ties::require_agrees_or_tie(o.sc, o.ec, ref.cm, kTieMargin, kAgreePositions, "the scheduled batched transcript of C");
  require(o.hist[0] > 0 && o.hist[1] > 0, "both scalar depth variants replayed");
  require(o.batch_hist[0] > 0 && o.batch_hist[1] > 0, "both batch depth variants replayed");
  uint64_t total = 0, btotal = 0;
  for (const uint64_t h : o.hist) total += h;
  for (const uint64_t h : o.batch_hist) btotal += h;
  require(total == static_cast<uint64_t>(o.steps_a + o.steps_b_alone) && btotal == static_cast<uint64_t>(o.steps_batch),
          "the histograms count every replay");
  require(o.steps_b_alone > 0, "B stepped alone after the batch over the batch's published draft probabilities");
}

// engine.mtp_schedule_max_live 1: a step with more than one live request
// takes the narrow option (one draft verified AND drafted, the fixed
// depth-1 replay), whatever the policy or the hook asks; the step after it
// has one draft to verify, so a forced whole block verifies one; and a
// request alone is scheduled as before. Every transcript is the plain eager
// engine's.
DGPP_TEST(glm_dsa_engines_loopback_world_2_scheduled_verify_depth_narrows_past_one_live_request) {
  const GlmDsaTextConfig cfg = glmdsafx::tiny_config();
  const std::string dir = "glm_dsa_engine_fixture";
  glmdsafx::write_fixture(cfg, dir);
  const std::vector<int64_t> A = smoke_tokens(cfg, 23, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 17, 0xD1B54A32D192ED03ull);
  const std::vector<int64_t> C = smoke_tokens(cfg, 11, 0x2545F4914F6CDD1Dull);
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, kPort + 12);
  require(!buses.empty(), "the loopback bus world failed to start");
  std::vector<SchedOutcome> outs(kWorld);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int r = 0; r < kWorld; ++r)
    workers.emplace_back(rank_work_sched, r, std::cref(cfg), std::cref(dir), std::cref(A), std::cref(B), std::cref(C),
                         buses[static_cast<size_t>(r)].get(), &barrier, &outs[static_cast<size_t>(r)], /*max_live=*/1, /*max_options=*/0);
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r) require(outs[static_cast<size_t>(r)].error.empty(), outs[static_cast<size_t>(r)].error);
  for (int r = 1; r < kWorld; ++r)
    require(outs[static_cast<size_t>(r)].sa == outs[0].sa && outs[static_cast<size_t>(r)].sb == outs[0].sb &&
                outs[static_cast<size_t>(r)].sc == outs[0].sc && outs[static_cast<size_t>(r)].hist == outs[0].hist,
            "the ranks' narrowed transcripts or depths differ");
  const Ref ref = world1_reference(cfg, dir, A, B, C);
  const SchedOutcome& o = outs[0];
  std::string hist, bhist;
  for (size_t i = 0; i < o.hist.size(); ++i)
    hist += (i ? " " : "") + std::to_string(o.options[i]) + ":" + std::to_string(o.hist[i]);
  for (size_t i = 0; i < o.batch_hist.size(); ++i)
    bhist += (i ? " " : "") + std::to_string(o.options[i]) + ":" + std::to_string(o.batch_hist[i]);
  DGPP_LOG_INFO("world 2 scheduled depth, one live request at most: options {} | A {} ({} steps) | B {} | C {} | scalar [{}] | batched [{}]",
                ids_text(o.options), ids_text(o.sa), o.steps_a, ids_text(o.sb), ids_text(o.sc), hist, bhist);
  require(o.options == std::vector<int>{1, 1, 2}, "the narrow option stands before the policy's two");
  require(o.sa == o.ea, "the scheduled scalar transcript of A differs from the plain eager engine's");
  engine_ties::require_agrees_or_tie(o.sb, o.eb, ref.bm, kTieMargin, kAgreePositions, "the narrowed transcript of B");
  engine_ties::require_agrees_or_tie(o.sc, o.ec, ref.cm, kTieMargin, kAgreePositions, "the narrowed batched transcript of C");
  require(o.batch_hist.size() == 3 && o.batch_hist[0] == static_cast<uint64_t>(o.steps_batch) && o.batch_hist[1] == 0 &&
              o.batch_hist[2] == 0,
          "every two-request step took the narrow option");
  require(o.hist.size() == 3 && o.hist[0] == 0 && o.hist[1] > 0 && o.hist[2] > 0,
          "a request alone replays the policy's options and never the narrow one");
  uint64_t total = 0;
  for (const uint64_t h : o.hist) total += h;
  require(total == static_cast<uint64_t>(o.steps_a + o.steps_b_alone), "the histogram counts every scalar replay");
  require(o.steps_b_alone > 0, "B stepped alone after the narrow batch");
  require(o.first_alone_option == 1,
          "B's first step alone, forced to the whole block, verifies the one draft the narrow batch left it");
}

// One scheduled depth under the live limit (engine.mtp_schedule_options 1
// with engine.mtp_schedule_max_live 1): the policy's only option is the full
// block, and a narrow batch leaves each request one draft. The first step a
// request takes alone replays the refill option, which verifies that draft
// and drafts the block, and every step after it verifies the full depth.
DGPP_TEST(glm_dsa_engines_loopback_world_2_scheduled_verify_depth_single_option_refills_after_the_narrow_batch) {
  const GlmDsaTextConfig cfg = glmdsafx::tiny_config();
  const std::string dir = "glm_dsa_engine_fixture";
  glmdsafx::write_fixture(cfg, dir);
  const std::vector<int64_t> A = smoke_tokens(cfg, 23, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 17, 0xD1B54A32D192ED03ull);
  const std::vector<int64_t> C = smoke_tokens(cfg, 11, 0x2545F4914F6CDD1Dull);
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, 30020);
  require(!buses.empty(), "the loopback bus world failed to start");
  std::vector<SchedOutcome> outs(kWorld);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int r = 0; r < kWorld; ++r)
    workers.emplace_back(rank_work_sched, r, std::cref(cfg), std::cref(dir), std::cref(A), std::cref(B), std::cref(C),
                         buses[static_cast<size_t>(r)].get(), &barrier, &outs[static_cast<size_t>(r)], /*max_live=*/1,
                         /*max_options=*/1);
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r) require(outs[static_cast<size_t>(r)].error.empty(), outs[static_cast<size_t>(r)].error);
  for (int r = 1; r < kWorld; ++r)
    require(outs[static_cast<size_t>(r)].sa == outs[0].sa && outs[static_cast<size_t>(r)].sb == outs[0].sb &&
                outs[static_cast<size_t>(r)].sc == outs[0].sc && outs[static_cast<size_t>(r)].hist == outs[0].hist &&
                outs[static_cast<size_t>(r)].first_alone_option == outs[0].first_alone_option,
            "the ranks' single-option transcripts or depths differ");
  const Ref ref = world1_reference(cfg, dir, A, B, C);
  const SchedOutcome& o = outs[0];
  std::string hist, bhist;
  for (size_t i = 0; i < o.hist.size(); ++i)
    hist += (i ? " " : "") + std::to_string(o.options[i]) + ":" + std::to_string(o.hist[i]);
  for (size_t i = 0; i < o.batch_hist.size(); ++i)
    bhist += (i ? " " : "") + std::to_string(o.options[i]) + ":" + std::to_string(o.batch_hist[i]);
  DGPP_LOG_INFO("world 2 scheduled depth, one option and one live request at most: options {} | A {} ({} steps) | B {} "
                "({} alone, the first at option {}) | C {} | scalar [{}] | batched [{}]",
                ids_text(o.options), ids_text(o.sa), o.steps_a, ids_text(o.sb), o.steps_b_alone, o.first_alone_option,
                ids_text(o.sc), hist, bhist);
  require(o.options == std::vector<int>{1, 1, 2}, "the narrow option, the refill option, the full block");
  require(o.sa == o.ea, "the single-option scalar transcript of A differs from the plain eager engine's");
  engine_ties::require_agrees_or_tie(o.sb, o.eb, ref.bm, kTieMargin, kAgreePositions, "the single-option transcript of B");
  engine_ties::require_agrees_or_tie(o.sc, o.ec, ref.cm, kTieMargin, kAgreePositions,
                                     "the single-option batched transcript of C");
  require(o.batch_hist.size() == 3 && o.batch_hist[0] == static_cast<uint64_t>(o.steps_batch) && o.batch_hist[1] == 0 &&
              o.batch_hist[2] == 0,
          "every two-request step took the narrow option");
  require(o.steps_b_alone > 1, "B stepped alone after the narrow batch, more than once");
  require(o.first_alone_option == 1, "B's first step alone verifies the one draft the narrow batch left it");
  require(o.hist.size() == 3 && o.hist[0] == 0 && o.hist[1] == 1 &&
              o.hist[2] == static_cast<uint64_t>(o.steps_a + o.steps_b_alone - 1),
          "one refill step, and the full block on every other step a request takes alone");
}

// ---- The scheduled verify depth under random arrivals ----
// Requests open, decode together and retire in a seeded random order on an
// engine with a live limit, greedy ones and sampled ones. The walk keeps its
// own count of the drafts every slot holds, from the option each replay took
// (read off the depth histograms), and holds every step to it: a step past
// the limit takes the narrow option and no other does, no step verifies a
// draft its slot's last replay did not write, and a slot is back on the
// policy's options one step after it leaves the limit. The same walk runs
// twice on the one engine, where the slots start from different leftovers,
// and every transcript repeats. A greedy request that only scalar replays
// carried is the eager engine's transcript exactly.
struct WalkConfig {
  const char* name;
  uint16_t port;
  int max_options;     // engine.mtp_schedule_options
  int max_live;        // engine.mtp_schedule_max_live
  int batch_min_live;  // the row batch's crossover
  int max_open;        // the walk's live requests at most
};

struct WalkRequest {
  std::vector<int64_t> prompt;
  int tokens = 0;  // decided tokens after the prefill's
  bool sampled = false;
  uint64_t seed = 0;
};

struct WalkOutcome {
  std::string error;
  std::vector<int> options;
  std::vector<std::vector<int32_t>> eager;    // per request: the eager engine's transcript (greedy requests)
  std::vector<std::vector<int32_t>> pass[2];  // per request: the graph engine's, the walk run twice
  std::vector<char> sampled, batched;         // per request: sampled; a row-batched replay carried it
  uint64_t steps = 0, narrow = 0, refill = 0, fallbacks = 0;
};

constexpr int kWalkRequests = 5;
constexpr int kWalkDepth = 2;
constexpr int kWalkMinDepth = 1;

uint64_t walk_random(uint64_t* s) {
  *s ^= *s << 13;
  *s ^= *s >> 7;
  *s ^= *s << 17;
  return *s;
}

uint64_t walk_seed(int walk) { return 0xA5B35705F1E4D3C7ull + 0x9E3779B97F4A7C15ull * static_cast<uint64_t>(walk + 1); }

std::vector<WalkRequest> walk_plan(const GlmDsaTextConfig& cfg, uint64_t seed) {
  std::vector<WalkRequest> plan(kWalkRequests);
  uint64_t s = seed;
  for (WalkRequest& q : plan) {
    const int n = 8 + static_cast<int>(walk_random(&s) % 14);
    q.prompt = smoke_tokens(cfg, n, walk_random(&s) | 1);
    q.tokens = 8 + static_cast<int>(walk_random(&s) % 18);
    q.sampled = walk_random(&s) % 3 == 0;
    q.seed = walk_random(&s);
  }
  return plan;
}

void run_walk(GraphEngineAdapter<GlmDsaModel>& eng, const WalkConfig& wc, int families,
              const std::vector<WalkRequest>& plan, uint64_t seed, std::vector<int>* forced,
              std::vector<std::vector<int32_t>>* transcripts, std::vector<char>* batched, WalkOutcome* out) {
  const int depth = kWalkDepth, min_depth = kWalkMinDepth;
  const std::vector<int> options = eng.verify_depth_options();
  const int count = static_cast<int>(options.size());
  const int first_block = wc.max_live > 0 ? 1 : 0;
  const bool has_refill = wc.max_live > 0 && wc.max_options == 1;
  const int first_policy = first_block + (has_refill ? 1 : 0);
  // The option the engine owes a step whose policy depth is k and whose
  // slots hold `held` drafts: the policy's lowest option covering k, or the
  // block-drafting option at the minimum depth when the slots hold less.
  const auto owed = [&](int k, int held) {
    k = std::clamp(std::min(k, held), min_depth, depth);
    int option = count - 1;
    for (int i = first_policy; i < count; ++i)
      if (options[static_cast<size_t>(i)] >= k) {
        option = i;
        break;
      }
    if (options[static_cast<size_t>(option)] > held) option = first_block;
    return option;
  };
  std::vector<int> slot_request(kSlots, -1), held(kSlots, depth);
  std::vector<char> fresh(kSlots, 1);
  uint64_t rng = seed;
  size_t next = 0;
  int tick = 0;
  const auto where = [&] {
    return std::string(wc.name) + ", walk seed " + std::to_string(seed) + ", tick " + std::to_string(tick);
  };
  const auto histograms = [&] {
    std::vector<std::vector<uint64_t>> h;
    h.push_back(eng.verify_depth_histogram());
    for (int f = 0; f < families; ++f) h.push_back(eng.verify_depth_histogram_batch(f));
    return h;
  };
  for (;; ++tick) {
    require(tick < 4000, where() + ": the walk does not end");
    int live = 0, sampled_slot = -1;
    for (int s = 0; s < kSlots; ++s)
      if (slot_request[static_cast<size_t>(s)] >= 0) {
        ++live;
        if (plan[static_cast<size_t>(slot_request[static_cast<size_t>(s)])].sampled) sampled_slot = s;
      }
    if (next == plan.size() && live == 0) break;
    // An arrival: always into an idle engine, else one tick in three. A
    // sampled request waits for the one before it, so a fallback in a step
    // is the one sampled slot's.
    if (next < plan.size() && live < wc.max_open && !(plan[next].sampled && sampled_slot >= 0) &&
        (live == 0 || walk_random(&rng) % 3 == 0)) {
      int slot = 0;
      while (slot_request[static_cast<size_t>(slot)] >= 0) ++slot;
      const WalkRequest& q = plan[next];
      eng.drain();
      if (q.sampled) {
        // Half the sampled requests keep a top-k the device's candidates
        // hold, so the device decides their steps. The others draw from the
        // whole vocabulary, which on this fixture mostly ends in the host's
        // fallback and its re-draft.
        dgpp::sample::Params params;
        params.temperature = 1.0f;
        if (q.seed & 1) params.top_k = 20;
        eng.configure_sampling(slot, params, q.seed);
        sampled_slot = slot;
      }
      (*transcripts)[next].push_back(eng.prefill(slot, q.prompt));
      eng.reserve(slot, static_cast<int64_t>(q.prompt.size()) + q.tokens + depth + 2);
      slot_request[static_cast<size_t>(slot)] = static_cast<int>(next++);
      held[static_cast<size_t>(slot)] = depth;  // the eager first drafts are the block
      fresh[static_cast<size_t>(slot)] = 1;
    }
    std::vector<int> reqs;
    for (int s = 0; s < kSlots; ++s)
      if (slot_request[static_cast<size_t>(s)] >= 0) reqs.push_back(s);
    if (reqs.empty()) continue;
    const bool gated = wc.max_live > 0 && static_cast<int>(reqs.size()) > wc.max_live;
    for (const int s : reqs)
      (*forced)[static_cast<size_t>(s)] = static_cast<int>(walk_random(&rng) % static_cast<uint64_t>(depth + 2));
    const std::vector<std::vector<uint64_t>> before = histograms();
    const uint64_t fallbacks_before = eng.fallbacks();
    const std::vector<std::vector<int32_t>> got = eng.step_batch(reqs);
    const std::vector<std::vector<uint64_t>> after = histograms();
    require(got.size() == reqs.size(), where() + ": the step's shape");
    ++out->steps;
    // The step's replays, by the options the histograms counted.
    std::vector<int> scalar(static_cast<size_t>(count), 0);
    int batch_option = -1, batch_replays = 0, scalar_replays = 0;
    for (int i = 0; i < count; ++i) {
      scalar[static_cast<size_t>(i)] = static_cast<int>(after[0][static_cast<size_t>(i)] - before[0][static_cast<size_t>(i)]);
      scalar_replays += scalar[static_cast<size_t>(i)];
      for (int f = 0; f < families; ++f) {
        const int d = static_cast<int>(after[static_cast<size_t>(1 + f)][static_cast<size_t>(i)] -
                                       before[static_cast<size_t>(1 + f)][static_cast<size_t>(i)]);
        if (d > 0) batch_option = i;
        batch_replays += d;
      }
    }
    const bool rode_batch = batch_replays > 0;
    require(rode_batch ? batch_replays == 1 && scalar_replays == 0 : scalar_replays == static_cast<int>(reqs.size()),
            where() + ": one row-batched replay, or one scalar replay a request");
    // What the engine owed each slot, and what it replayed.
    std::vector<int> option_of(kSlots, -1);
    std::vector<int> held_sorted, depth_sorted;
    if (rode_batch) {
      int low = depth, k = 0;
      bool whole = false;
      for (const int s : reqs) {
        low = std::min(low, held[static_cast<size_t>(s)]);
        whole = whole || fresh[static_cast<size_t>(s)] ||
                plan[static_cast<size_t>(slot_request[static_cast<size_t>(s)])].sampled;
        k = std::max(k, (*forced)[static_cast<size_t>(s)]);
      }
      require(options[static_cast<size_t>(batch_option)] <= low,
              where() + ": the batch verified " + std::to_string(options[static_cast<size_t>(batch_option)]) +
                  " drafts where a slot held " + std::to_string(low));
      const int want = gated ? 0 : owed(whole ? low : k, low);
      require(batch_option == want, where() + ": the batch replayed option " + std::to_string(batch_option) +
                                        ", owed " + std::to_string(want));
      for (const int s : reqs) option_of[static_cast<size_t>(s)] = batch_option;
    } else {
      std::vector<int> want(static_cast<size_t>(count), 0);
      for (const int s : reqs) {
        const bool whole = fresh[static_cast<size_t>(s)] ||
                           plan[static_cast<size_t>(slot_request[static_cast<size_t>(s)])].sampled;
        const int h = held[static_cast<size_t>(s)];
        option_of[static_cast<size_t>(s)] = gated ? 0 : owed(whole ? h : (*forced)[static_cast<size_t>(s)], h);
        ++want[static_cast<size_t>(option_of[static_cast<size_t>(s)])];
        held_sorted.push_back(h);
      }
      // Whatever the walk expects: the depths the scalar replays verified
      // fit the drafts the slots held, slot for slot in some order.
      for (int i = 0; i < count; ++i)
        for (int n = 0; n < scalar[static_cast<size_t>(i)]; ++n) depth_sorted.push_back(options[static_cast<size_t>(i)]);
      std::sort(held_sorted.begin(), held_sorted.end());
      std::sort(depth_sorted.begin(), depth_sorted.end());
      for (size_t i = 0; i < depth_sorted.size(); ++i)
        require(depth_sorted[i] <= held_sorted[i],
                where() + ": a scalar replay verified " + std::to_string(depth_sorted[i]) + " drafts where its slot held " +
                    std::to_string(held_sorted[i]));
      std::string seen, owes;
      for (int i = 0; i < count; ++i) {
        seen += (i ? " " : "") + std::to_string(scalar[static_cast<size_t>(i)]);
        owes += (i ? " " : "") + std::to_string(want[static_cast<size_t>(i)]);
      }
      require(scalar == want, where() + ": the scalar replays per option were [" + seen + "], owed [" + owes + "]");
    }
    for (const int s : reqs)
      require((option_of[static_cast<size_t>(s)] < first_block) == gated,
              where() + ": the narrow option belongs to the steps past the live limit, and to every one of them");
    if (gated) ++out->narrow;
    bool refilled = false;
    for (const int s : reqs) refilled = refilled || (has_refill && option_of[static_cast<size_t>(s)] == first_block);
    if (refilled) ++out->refill;
    // The drafts each slot holds now, and a host re-draft's block.
    const uint64_t fell = eng.fallbacks() - fallbacks_before;
    require(fell == 0 || sampled_slot >= 0, where() + ": a fallback without a sampled request");
    out->fallbacks += fell;
    for (size_t i = 0; i < reqs.size(); ++i) {
      const int s = reqs[i];
      const int option = option_of[static_cast<size_t>(s)];
      require(!got[i].empty() && static_cast<int>(got[i].size()) <= 1 + options[static_cast<size_t>(option)],
              where() + ": a step commits its token and at most the drafts it verified");
      held[static_cast<size_t>(s)] = option < first_block ? options[static_cast<size_t>(option)] : depth;
      fresh[static_cast<size_t>(s)] = 0;
    }
    if (fell > 0) {
      held[static_cast<size_t>(sampled_slot)] = depth;
      fresh[static_cast<size_t>(sampled_slot)] = 1;
    }
    for (size_t i = 0; i < reqs.size(); ++i) {
      const int s = reqs[i];
      const size_t q = static_cast<size_t>(slot_request[static_cast<size_t>(s)]);
      std::vector<int32_t>& t = (*transcripts)[q];
      t.insert(t.end(), got[i].begin(), got[i].end());
      if (rode_batch) (*batched)[q] = 1;
      if (static_cast<int>(t.size()) >= 1 + plan[q].tokens) {
        t.resize(static_cast<size_t>(1 + plan[q].tokens));
        eng.close(s);
        slot_request[static_cast<size_t>(s)] = -1;
      }
    }
  }
  eng.drain();
}

void rank_work_walk(int r, const GlmDsaTextConfig& cfg, const std::string& dir, const WalkConfig& wc, int walks,
                    CollectiveBus* bus, ConstructBarrier* barrier, WalkOutcome* out) {
  bool arrived = false;
  const auto arrive_once = [&] {
    if (arrived) return;
    arrived = true;
    barrier->arrive_and_wait();
  };
  uint16_t* scratch = nullptr;
  uint16_t* prefix_scratch = nullptr;
  uint16_t* gather_scratch = nullptr;
  try {
    BusBoundaryReducer reducer(*bus, wait_timeout_ms());
    GlmDsaModel eager(cfg, dir, kMaxTokens, kCache, GlmDsaResidency::Resident, &reducer, r, kWorld, kSlots);
    GlmDsaModel mtp(cfg, dir, kMaxTokens, kCache, GlmDsaResidency::Resident, &reducer, r, kWorld, kSlots,
                    /*mtp=*/true, /*decode_rows=*/kSlots * 3, dgpp::LatentFormat::kBf16, /*serving_logits=*/true);
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                               sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld), cudaHostAllocDefault));
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&prefix_scratch),
                               sizeof(uint16_t) * std::max<size_t>(dgpp::fabric_sampling_prefix_scratch_elems(kWorld), 2),
                               cudaHostAllocDefault));
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&gather_scratch),
                               sizeof(uint16_t) * std::max<size_t>(dgpp::sampling_gather_scratch_elems(cfg.vocab_size), 2),
                               cudaHostAllocDefault));
    arrive_once();
    EagerEngineAdapter<GlmDsaModel> eager_engine(
        &eager, kSlots, dgpp::make_fabric_pick(bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms()));
    std::vector<std::vector<WalkRequest>> plans;
    for (int w = 0; w < walks; ++w) {
      plans.push_back(walk_plan(cfg, walk_seed(w)));
      for (const WalkRequest& q : plans.back()) {
        out->sampled.push_back(q.sampled ? 1 : 0);
        out->eager.push_back(q.sampled ? std::vector<int32_t>{} : solo(eager_engine, 0, q.prompt, q.tokens));
      }
    }
    {
      GraphEngineAdapter<GlmDsaModel> eng(&mtp, bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms(),
                                          wc.batch_min_live, prefix_scratch, gather_scratch,
                                          dgpp::kSamplingCandidates, nullptr, 0, /*mtp_depth=*/kWalkDepth);
      require(eng.supports_sampling(), "the world has the device sampler (the draft probabilities' source)");
      eng.configure_verify_schedule(true, /*row_ms=*/12.0f, dgpp::verify_reservation_lambda(50.0f, 12.0f),
                                    kWalkMinDepth, /*base_ms=*/0.f, /*adapt=*/false, wc.max_options, wc.max_live);
      out->options = eng.verify_depth_options();
      int families = 0;
      for (;; ++families) {
        try {
          (void)eng.verify_depth_histogram_batch(families);
        } catch (const std::out_of_range&) {
          break;
        }
      }
      std::vector<int> forced(kSlots, 0);
      eng.set_verify_depth_hook([&](int req, int, const float*, int) { return forced[static_cast<size_t>(req)]; });
      for (int pass = 0; pass < 2; ++pass)
        for (int w = 0; w < walks; ++w) {
          std::vector<std::vector<int32_t>> transcripts(kWalkRequests);
          std::vector<char> batched(kWalkRequests, 0);
          run_walk(eng, wc, families, plans[static_cast<size_t>(w)], walk_seed(w), &forced, &transcripts, &batched, out);
          for (int q = 0; q < kWalkRequests; ++q) {
            out->pass[pass].push_back(transcripts[static_cast<size_t>(q)]);
            if (pass == 0) out->batched.push_back(batched[static_cast<size_t>(q)]);
          }
        }
      eng.drain();
    }
    cudaFreeHost(scratch);
    cudaFreeHost(prefix_scratch);
    cudaFreeHost(gather_scratch);
  } catch (const std::exception& e) {
    if (scratch) cudaFreeHost(scratch);
    if (prefix_scratch) cudaFreeHost(prefix_scratch);
    if (gather_scratch) cudaFreeHost(gather_scratch);
    out->error = "rank " + std::to_string(r) + ": " + e.what();
    arrive_once();
  }
}

DGPP_TEST(glm_dsa_engines_loopback_world_2_scheduled_verify_depth_random_walks_verify_only_the_drafts_held) {
  const GlmDsaTextConfig cfg = glmdsafx::tiny_config();
  const std::string dir = "glm_dsa_engine_fixture";
  glmdsafx::write_fixture(cfg, dir);
  const char* env = std::getenv("DGPP_TEST_SCHEDULE_WALKS");
  const int walks = env != nullptr ? std::max(1, std::atoi(env)) : 3;
  const WalkConfig configs[] = {
      {"one option, scalar narrow steps", 30021, /*max_options=*/1, /*max_live=*/1, /*batch_min_live=*/3, /*max_open=*/2},
      {"one option, batched narrow steps", 30022, 1, 1, 2, 3},
      {"one option, a limit of two", 30023, 1, 2, 2, 3},
      {"every depth, batched narrow steps", 30024, 0, 1, 2, 3},
  };
  for (const WalkConfig& wc : configs) {
    std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, wc.port);
    require(!buses.empty(), "the loopback bus world failed to start");
    std::vector<WalkOutcome> outs(kWorld);
    ConstructBarrier barrier(kWorld);
    std::vector<std::thread> workers;
    for (int r = 0; r < kWorld; ++r)
      workers.emplace_back(rank_work_walk, r, std::cref(cfg), std::cref(dir), std::cref(wc), walks,
                           buses[static_cast<size_t>(r)].get(), &barrier, &outs[static_cast<size_t>(r)]);
    for (auto& t : workers) t.join();
    for (int r = 0; r < kWorld; ++r)
      require(outs[static_cast<size_t>(r)].error.empty(), std::string(wc.name) + ": " + outs[static_cast<size_t>(r)].error);
    for (int r = 1; r < kWorld; ++r)
      require(outs[static_cast<size_t>(r)].pass[0] == outs[0].pass[0] && outs[static_cast<size_t>(r)].pass[1] == outs[0].pass[1] &&
                  outs[static_cast<size_t>(r)].eager == outs[0].eager && outs[static_cast<size_t>(r)].steps == outs[0].steps &&
                  outs[static_cast<size_t>(r)].narrow == outs[0].narrow && outs[static_cast<size_t>(r)].refill == outs[0].refill,
              std::string(wc.name) + ": the ranks' walks differ");
    const WalkOutcome& o = outs[0];
    require(o.options == std::vector<int>{1, 1, 2}, std::string(wc.name) + ": the narrow option and two behind it");
    int greedy = 0, exact = 0, sampled = 0;
    for (size_t q = 0; q < o.pass[0].size(); ++q) {
      const std::string which = std::string(wc.name) + ", request " + std::to_string(q);
      require(o.pass[0][q] == o.pass[1][q], which + ": the walk run again decides differently");
      if (o.sampled[q]) {
        ++sampled;
        continue;
      }
      ++greedy;
      if (o.batched[q]) continue;  // the batched rows' lowering differs: the eager transcript is a tolerance there
      ++exact;
      require(o.pass[0][q] == o.eager[q], which + ": the scalar transcript differs from the plain eager engine's");
    }
    DGPP_LOG_INFO("world 2 scheduled depth walk ({}): {} walks twice, {} steps, {} past the limit, {} refill steps, "
                  "{} fallbacks | {} greedy requests ({} the eager transcript exactly), {} sampled",
                  wc.name, walks, o.steps, o.narrow, o.refill, o.fallbacks, greedy, exact, sampled);
    require(o.narrow > 0, std::string(wc.name) + ": no step went past the live limit");
    require((o.refill > 0) == (wc.max_options == 1),
            std::string(wc.name) + ": the refill option belongs to a single option under the limit");
    if (wc.batch_min_live > wc.max_open)
      require(exact == greedy, std::string(wc.name) + ": every greedy request stepped on scalar replays");
  }
}

DGPP_TEST(glm_dsa_engines_loopback_world_2_mtp_depth2_graph_matches_plain_decode) {
  const GlmDsaTextConfig cfg = glmdsafx::tiny_config();
  const std::string dir = "glm_dsa_engine_fixture";
  glmdsafx::write_fixture(cfg, dir);
  const std::vector<int64_t> A = smoke_tokens(cfg, 23, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 17, 0xD1B54A32D192ED03ull);
  const std::vector<int64_t> C = smoke_tokens(cfg, 11, 0x2545F4914F6CDD1Dull);
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, kPort + 2);
  require(!buses.empty(), "the loopback bus world failed to start");
  std::vector<RankOutcome> outs(kWorld);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int r = 0; r < kWorld; ++r)
    workers.emplace_back(rank_work_mtp_depth2, r, std::cref(cfg), std::cref(dir), std::cref(A), std::cref(B),
                         std::cref(C), buses[static_cast<size_t>(r)].get(), &barrier, &outs[static_cast<size_t>(r)]);
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r) require(outs[static_cast<size_t>(r)].error.empty(), outs[static_cast<size_t>(r)].error);
  for (int r = 1; r < kWorld; ++r)
    require(outs[static_cast<size_t>(r)].ma == outs[0].ma && outs[static_cast<size_t>(r)].mb == outs[0].mb &&
                outs[static_cast<size_t>(r)].mc == outs[0].mc,
            "the ranks' depth-2 transcripts differ");
  const Ref ref = world1_reference(cfg, dir, A, B, C);
  const RankOutcome& o = outs[0];
  DGPP_LOG_INFO("world 2 MTP depth 2: scalar A {} ({} steps for {} tokens) | batched B {} | C {}", ids_text(o.ma),
                o.mtp_steps_a, kSteps, ids_text(o.mb), ids_text(o.mc));
  require(o.ma == o.ea, "the scalar depth-2 transcript of A differs from the plain eager engine's");
  engine_ties::require_agrees_or_tie(o.mb, o.eb, ref.bm, kTieMargin, kAgreePositions, "the batched depth-2 transcript of B");
  engine_ties::require_agrees_or_tie(o.mc, o.ec, ref.cm, kTieMargin, kAgreePositions, "the batched depth-2 transcript of C");
}

DGPP_TEST(glm_dsa_engines_loopback_world_2_mtp_graph_matches_plain_decode) {
  const GlmDsaTextConfig cfg = glmdsafx::tiny_config();
  const std::string dir = "glm_dsa_engine_fixture";
  glmdsafx::write_fixture(cfg, dir);
  const std::vector<int64_t> A = smoke_tokens(cfg, 23, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 17, 0xD1B54A32D192ED03ull);
  const std::vector<int64_t> C = smoke_tokens(cfg, 11, 0x2545F4914F6CDD1Dull);
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, kPort + 1);
  require(!buses.empty(), "the loopback bus world failed to start");
  std::vector<RankOutcome> outs(kWorld);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int r = 0; r < kWorld; ++r)
    workers.emplace_back(rank_work_mtp, r, std::cref(cfg), std::cref(dir), std::cref(A), std::cref(B),
                         std::cref(C), buses[static_cast<size_t>(r)].get(), &barrier,
                         &outs[static_cast<size_t>(r)]);
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r) require(outs[static_cast<size_t>(r)].error.empty(), outs[static_cast<size_t>(r)].error);
  for (int r = 1; r < kWorld; ++r)
    require(outs[static_cast<size_t>(r)].ma == outs[0].ma && outs[static_cast<size_t>(r)].mb == outs[0].mb,
            "the ranks' MTP transcripts differ");
  const Ref ref = world1_reference(cfg, dir, A, B, C);
  const RankOutcome& o = outs[0];
  DGPP_LOG_INFO("world 2 MTP graph: A {} ({} steps) | B {} | C {}", ids_text(o.ma), o.mtp_steps_a, ids_text(o.mb),
                ids_text(o.mc));
  require(o.ma == o.ea, "the MTP scalar transcript differs from the plain eager engine's");
  engine_ties::require_agrees_or_tie(o.mb, o.eb, ref.bm, kTieMargin, kAgreePositions, "the MTP batched transcript of B");
  engine_ties::require_agrees_or_tie(o.mc, o.ec, ref.cm, kTieMargin, kAgreePositions, "the MTP batched transcript of C");
  // A random-weight fixture drafts by chance only (the acceptance rate is
  // the real checkpoint's measurement, scripts/fabric_mtp_classes.sh).
  DGPP_LOG_INFO("world 2 MTP graph: A took {} steps for {} tokens", o.mtp_steps_a, kSteps);
}

RankOutcome graph_world(const GlmDsaTextConfig& cfg, const std::string& dir, const std::vector<int64_t>& A,
                        const std::vector<int64_t>& B, const std::vector<int64_t>& C, const Ref& ref, uint16_t port) {
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, port);
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
  // must agree (the folds reassociate; a near tie may flip a late token).
  const size_t pa = agreeing_prefix(o.ea, ref.a), pb = agreeing_prefix(o.eb, ref.b), pc = agreeing_prefix(o.ec, ref.c);
  DGPP_LOG_INFO("world 2 vs world 1 agreeing prefixes: A {}/{} B {}/{} C {}/{}", pa, ref.a.size(), pb, ref.b.size(),
                pc, ref.c.size());
  require(pa >= 3 && pb >= 3 && pc >= 3, "the world-2 eager engine diverges from world 1 at the start");
  // The graph engine against the eager engine at the same world: exact.
  require(o.ga == o.ea, "the scalar graph transcript differs from the eager engine's");
  // 61 tokens for A (23 + 60: the batched rows cross the pool's 64-token
  // block), 46 for B, 36 for C — each a prefix of the eager transcript.
  require(o.ba == o.ea, "the batched graph transcript of A differs from the eager engine's");
  require(std::equal(o.bb.begin(), o.bb.end(), o.eb.begin()), "the batched graph transcript of B differs");
  require(std::equal(o.bc.begin(), o.bc.end(), o.ec.begin()), "the batched graph transcript of C differs");
  require(o.bb.size() == 46 && o.bc.size() == 36, "the batched transcripts' lengths");
  return o;
}

DGPP_TEST(glm_dsa_engines_loopback_world_2_graph_matches_eager) {
  const GlmDsaTextConfig cfg = glmdsafx::tiny_config();
  const std::string dir = "glm_dsa_engine_fixture";
  glmdsafx::write_fixture(cfg, dir);
  const std::vector<int64_t> A = smoke_tokens(cfg, 23, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 17, 0xD1B54A32D192ED03ull);
  const std::vector<int64_t> C = smoke_tokens(cfg, 11, 0x2545F4914F6CDD1Dull);
  const Ref ref = world1_reference(cfg, dir, A, B, C);
  DGPP_LOG_INFO("world 1 eager: A {} | B {} | C {}", ids_text(ref.a), ids_text(ref.b), ids_text(ref.c));
  const RankOutcome replicated = graph_world(cfg, dir, A, B, C, ref, kPort);
  // engine.embed_sharding = vocab: the same world with each rank's rows of
  // the embedding summed by a recorded fold (the main path's and the
  // draft's) — eager and graph transcripts bitwise the replicated ones.
  dgpp::GlmDsaLayerStream::set_embed_vocab_sharded(true);
  const RankOutcome sharded = graph_world(cfg, dir, A, B, C, ref, static_cast<uint16_t>(kPort + 3));
  dgpp::GlmDsaLayerStream::set_embed_vocab_sharded(false);
  require(sharded.ea == replicated.ea && sharded.eb == replicated.eb && sharded.ec == replicated.ec,
          "the vocab-sharded embedding changes the eager transcripts");
  require(sharded.ga == replicated.ga && sharded.ba == replicated.ba && sharded.bb == replicated.bb &&
              sharded.bc == replicated.bc,
          "the vocab-sharded embedding changes the graph transcripts");
}

// The NF4I8 contract's fixture: the recorded decode step's codebook slot
// kernels with their in-smem rotation, bitwise the eager transcripts.
DGPP_TEST(glm_dsa_engines_loopback_world_2_graph_matches_eager_nf4i8) {
  const GlmDsaTextConfig cfg = glmdsafx::tiny_config(5, 1, true, true);
  const std::string dir = "glm_dsa_engine_fixture_nf4i8";
  glmdsafx::write_fixture(cfg, dir, true);
  const std::vector<int64_t> A = smoke_tokens(cfg, 23, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 17, 0xD1B54A32D192ED03ull);
  const std::vector<int64_t> C = smoke_tokens(cfg, 11, 0x2545F4914F6CDD1Dull);
  const Ref ref = world1_reference(cfg, dir, A, B, C);
  DGPP_LOG_INFO("world 1 eager (nf4i8): A {} | B {} | C {}", ids_text(ref.a), ids_text(ref.b), ids_text(ref.c));
  (void)graph_world(cfg, dir, A, B, C, ref, static_cast<uint16_t>(29976));
}

// The Mixed346 contract's fixture: the recorded decode step's quantizer
// and mixed-form slot kernels, bitwise the eager transcripts.
DGPP_TEST(glm_dsa_engines_loopback_world_2_graph_matches_eager_m346) {
  const GlmDsaTextConfig cfg = glmdsafx::tiny_config(5, 1, true, false, 8, true);
  const std::string dir = "glm_dsa_engine_fixture_m346";
  glmdsafx::write_fixture(cfg, dir, false, 8, true);
  const std::vector<int64_t> A = smoke_tokens(cfg, 23, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 17, 0xD1B54A32D192ED03ull);
  const std::vector<int64_t> C = smoke_tokens(cfg, 11, 0x2545F4914F6CDD1Dull);
  const Ref ref = world1_reference(cfg, dir, A, B, C);
  DGPP_LOG_INFO("world 1 eager (mixed346): A {} | B {} | C {}", ids_text(ref.a), ids_text(ref.b), ids_text(ref.c));
  (void)graph_world(cfg, dir, A, B, C, ref, static_cast<uint16_t>(29998));
}

int main() {
  // DGPP_TEST_DRAFT_SCREEN=N: every case drafts through the head screen
  // (kernels/draft_head.hpp) with an N-block candidate cap; every gate
  // holds unchanged (the verify reads the exact head; only drafts move).
  if (const char* d = std::getenv("DGPP_TEST_DRAFT_SCREEN"); d && d[0] != '\0')
    dgpp::draft_head_configure(dgpp::DraftHeadSettings{std::atoi(d)});
  return dgpp::test::run_all();
}
