// The Inkling 2 draft model (docs/inkling2_mini_plan.md section 1.7) on the
// shared block drafter.
//
//   --write-fixture DIR
//       the tiny release with a draft model (mtp.safetensors).
//   --fixture DIR
//       the structural checks: the draft model leaves the target's rows
//       alone at every accepted length, the memory plan and the pool carry
//       its planes, and the eager engine decodes with it.
//   --fixture DIR --dump FILE [--world N --port P]
//   --checkpoint-dir DIR --dump FILE [--ids FILE]
//       the cases tools/inkling2_reference_dump.py check-draft recomputes:
//       per case the states the draft tapped, the context keys and values
//       it stored, its block's final rows and its proposals. At a world
//       above 1 the ranks run on a loopback bus and their key/value slices
//       are joined in head order.
//   --checkpoint-dir DIR --ids FILE --steps N [--activations native]
//                    [--conf-fit BIAS,SLOPE --conf-out FILE]
//       greedy decoding of a release with and without the draft model: the
//       accepted lengths, the agreement of the two transcripts and the wall
//       time a token. --conf-out records every step's committed tokens and
//       the proposals' confidence, the calibration's input.
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "engine/eager_engine.hpp"
#include "engine/graph_engine.hpp"
#include "engine/tp_bus.hpp"
#include "inkling2_fixture.hpp"
#include "inkling2_reference_compare.hpp"
#include "models/inkling2/config.hpp"
#include "models/inkling2/forward.hpp"
#include "net/collective_bus.hpp"

using dgpp::BusBoundaryReducer;
using dgpp::DFlash2Drafter;
using dgpp::EagerEngineAdapter;
using dgpp::Inkling2Config;
using dgpp::Inkling2Model;
using dgpp::Inkling2Residency;
using dgpp::net::BusOptions;
using dgpp::net::CollectiveBus;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

std::vector<int64_t> smoke_tokens(const Inkling2Config& cfg, int n, uint64_t seed) {
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

std::unique_ptr<Inkling2Model> make_model(const Inkling2Config& cfg, const std::string& dir, int max_tokens,
                                          int64_t cache_tokens, dgpp::BoundaryReducer* reducer, int rank, int world,
                                          int slots, bool mtp, bool native) {
  return std::make_unique<Inkling2Model>(cfg, dir, max_tokens, cache_tokens, Inkling2Residency::Resident, reducer,
                                         rank, world, slots, mtp, slots * dgpp::kSpecRows, dgpp::LatentFormat::kBf16,
                                         /*serving_logits=*/false, native);
}

// ---- the reference cases ----------------------------------------------------------

// A context built the ways serving builds one: a prefill, scalar steps, and
// a verify of which only the first rows are kept.
struct Case {
  int prefill = 0;
  int steps = 0;
  int verify = 0;
  int accept = 0;
  int rows() const { return prefill + steps + verify + 1; }  // with the anchor token
};

struct CaseOut {
  int p = 0;
  int64_t anchor = 0;
  std::vector<std::vector<uint16_t>> taps;  // per tap [p, H]
  std::vector<uint16_t> kv;                 // [layers][keys, values][p][this rank's kv row]
  std::vector<uint16_t> hidden;             // [block, H]
  std::vector<int32_t> picks;               // [block - 1]
};

CaseOut run_case(Inkling2Model& m, const Inkling2Config& cfg, int world, const std::vector<int64_t>& ids,
                 const Case& c) {
  const DFlash2Drafter& df = m.drafter();
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const int taps = static_cast<int>(df.cfg().target_layer_ids.size());
  const int B = df.query_rows();
  require(static_cast<int>(ids.size()) >= c.rows(), "the case needs more token ids");
  CaseOut out;
  out.taps.assign(static_cast<size_t>(taps), {});
  // The first `rows` rows of the run that just finished are committed.
  const auto keep = [&](int rows) {
    DGPP_CUDA_OK(cudaStreamSynchronize(m.stream()));
    for (int t = 0; t < taps; ++t) {
      std::vector<uint16_t>& dst = out.taps[static_cast<size_t>(t)];
      const size_t at = dst.size();
      dst.resize(at + static_cast<size_t>(rows) * H);
      DGPP_CUDA_OK(cudaMemcpy(dst.data() + at, df.tap_rows(t), static_cast<size_t>(rows) * H * 2,
                              cudaMemcpyDeviceToHost));
    }
  };
  size_t at = 0;
  m.session_prefill(0, std::vector<int64_t>(ids.begin(), ids.begin() + c.prefill));
  keep(c.prefill);
  at += static_cast<size_t>(c.prefill);
  for (int s = 0; s < c.steps; ++s) {
    m.session_step(0, ids[at++]);
    keep(1);
  }
  if (c.verify > 0) {
    m.session_verify(0, std::vector<int64_t>(ids.begin() + static_cast<std::ptrdiff_t>(at),
                                             ids.begin() + static_cast<std::ptrdiff_t>(at) + c.verify));
    m.session_rollback(0, c.accept, c.verify);
    keep(c.accept);
    at += static_cast<size_t>(c.accept);
  }
  out.p = static_cast<int>(at);
  require(m.session_position(0) == static_cast<int64_t>(at), "the case's committed position");
  out.anchor = ids[at];

  std::vector<int32_t> drafts;
  require(m.dflash2_draft(0, out.anchor, &drafts), "the draft model produced no block");
  require(static_cast<int>(drafts.size()) == B - 1, "the block's proposals");
  out.picks = drafts;
  DGPP_CUDA_OK(cudaStreamSynchronize(m.stream()));
  out.hidden.resize(static_cast<size_t>(B) * H);
  DGPP_CUDA_OK(cudaMemcpy(out.hidden.data(), df.block_hidden(), out.hidden.size() * 2, cudaMemcpyDeviceToHost));

  // The context rows of the committed positions, through the block table.
  const int L = df.cfg().num_hidden_layers;
  const size_t kvw = static_cast<size_t>(df.cfg().local_kv_row(world));
  const int32_t* table = m.pool().request_table_row(0);
  const int block = static_cast<int>(m.kv_block_tokens());
  out.kv.resize(static_cast<size_t>(L) * 2 * static_cast<size_t>(out.p) * kvw);
  size_t w = 0;
  for (int l = 0; l < L; ++l) {
    const dgpp::MimoKvCache plane = m.pool().view(cfg.num_visits() + l);
    for (const uint16_t* data : {plane.k_cache, plane.v_cache})
      for (int r = 0; r < out.p; ++r) {
        const size_t physical = static_cast<size_t>(table[r / block]) * static_cast<size_t>(block) +
                                static_cast<size_t>(r % block);
        DGPP_CUDA_OK(cudaMemcpy(out.kv.data() + w, data + physical * kvw, kvw * 2, cudaMemcpyDeviceToHost));
        w += kvw;
      }
  }
  m.session_close(0);
  require(m.kv_blocks_in_use() == 0, "the case left blocks in the pool");
  return out;
}

std::vector<Case> fixture_cases() {
  return {
      {1, 0, 0, 0},    // one committed position
      {5, 0, 0, 0},    // shorter than the block and the window
      {40, 0, 0, 0},   // past the sliding layers' window
      {70, 0, 0, 0},   // past the pool's first block and around the ring
      {30, 7, 8, 3},   // decode rows, then a verify that keeps three of eight
      {61, 0, 8, 8},   // a verify across the block boundary, all kept
  };
}

BusOptions loop_options(int rank, int world, uint16_t port) {
  BusOptions o;
  o.world_size = world;
  o.my_rank = rank;
  o.rendezvous_port = port;
  o.rendezvous_host = rank == 0 ? "" : "127.0.0.1";
  o.rendezvous_timeout_ms = 20000;
  o.lat_slots = 8;
  o.lat_slot_bytes = 262144;
  o.bulk_slots = 8;
  o.bulk_slot_bytes = 262144;
  o.qp_depth = 1024;
  o.completion_timeout_ms = 10000;
  o.consumer_deadline_s = 60;
  o.launch_consumers = false;
  return o;
}

struct Barrier {
  std::mutex mu;
  std::condition_variable cv;
  int left;
  explicit Barrier(int world) : left(world) {}
  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mu);
    if (--left == 0) cv.notify_all();
    else cv.wait(lock, [this] { return left == 0; });
  }
};

// Every rank's outputs of every case, [rank][case].
std::vector<std::vector<CaseOut>> run_cases(const Inkling2Config& cfg, const std::string& dir, int world, uint16_t port,
                                            const std::vector<Case>& cases, const std::vector<int64_t>& ids,
                                            int max_tokens, int64_t cache_tokens, bool native) {
  std::vector<std::vector<CaseOut>> outs(static_cast<size_t>(world));
  if (world == 1) {
    auto m = make_model(cfg, dir, max_tokens, cache_tokens, nullptr, 0, 1, 1, true, native);
    for (const Case& c : cases) outs[0].push_back(run_case(*m, cfg, 1, ids, c));
    return outs;
  }
  std::vector<std::unique_ptr<CollectiveBus>> buses;
  for (int r = 0; r < world; ++r) buses.push_back(std::make_unique<CollectiveBus>(loop_options(r, world, port)));
  std::vector<std::string> errors(static_cast<size_t>(world));
  {
    std::vector<std::thread> starters;
    for (int r = 0; r < world; ++r)
      starters.emplace_back([&, r] { (void)buses[static_cast<size_t>(r)]->start(&errors[static_cast<size_t>(r)]); });
    for (auto& t : starters) t.join();
  }
  for (const std::string& e : errors) require(e.empty(), "the loopback bus: " + e);
  Barrier barrier(world);
  std::vector<std::thread> workers;
  for (int r = 0; r < world; ++r)
    workers.emplace_back([&, r] {
      bool arrived = false;
      try {
        BusBoundaryReducer reducer(*buses[static_cast<size_t>(r)], 60000);
        auto m = make_model(cfg, dir, max_tokens, cache_tokens, &reducer, r, world, 1, true, native);
        arrived = true;
        barrier.arrive_and_wait();
        for (const Case& c : cases) outs[static_cast<size_t>(r)].push_back(run_case(*m, cfg, world, ids, c));
      } catch (const std::exception& e) {
        errors[static_cast<size_t>(r)] = "rank " + std::to_string(r) + ": " + e.what();
        if (!arrived) barrier.arrive_and_wait();
      }
    });
  for (auto& t : workers) t.join();
  for (auto& bus : buses) bus->stop();
  for (const std::string& e : errors) require(e.empty(), e);
  return outs;
}

// The dump check-draft reads: the replicated values from rank 0 (the ranks
// must agree on them) and the key/value slices joined in head order.
void write_dump(const std::string& path, const Inkling2Config& cfg, int world,
                const std::vector<std::vector<CaseOut>>& outs) {
  const dgpp::Inkling2DraftConfig& d = cfg.draft;
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const int taps = static_cast<int>(d.target_layer_ids.size());
  const size_t full = static_cast<size_t>(d.num_key_value_heads) * static_cast<size_t>(d.head_dim);
  const size_t local = full / static_cast<size_t>(world);
  std::ofstream f(path, std::ios::binary);
  f.write("DGPPINKD", 8);
  const int32_t header[7] = {1, cfg.hidden_size, taps, d.num_hidden_layers, static_cast<int32_t>(full), d.block_size,
                             static_cast<int32_t>(outs[0].size())};
  f.write(reinterpret_cast<const char*>(header), sizeof(header));
  for (size_t c = 0; c < outs[0].size(); ++c) {
    const CaseOut& o = outs[0][c];
    for (int r = 1; r < world; ++r) {
      const CaseOut& other = outs[static_cast<size_t>(r)][c];
      require(other.p == o.p && other.taps == o.taps, "the ranks tapped different states");
      require(other.hidden == o.hidden, "the ranks' block rows differ");
      require(other.picks == o.picks, "the ranks' proposals differ");
    }
    const int32_t head[2] = {o.p, 0};
    f.write(reinterpret_cast<const char*>(head), sizeof(head));
    f.write(reinterpret_cast<const char*>(&o.anchor), sizeof(o.anchor));
    for (const auto& t : o.taps) {
      require(t.size() == static_cast<size_t>(o.p) * H, "a tap's rows");
      f.write(reinterpret_cast<const char*>(t.data()), static_cast<std::streamsize>(t.size() * 2));
    }
    const size_t rows = static_cast<size_t>(d.num_hidden_layers) * 2 * static_cast<size_t>(o.p);
    for (size_t row = 0; row < rows; ++row)
      for (int r = 0; r < world; ++r)
        f.write(reinterpret_cast<const char*>(outs[static_cast<size_t>(r)][c].kv.data() + row * local),
                static_cast<std::streamsize>(local * 2));
    f.write(reinterpret_cast<const char*>(o.hidden.data()), static_cast<std::streamsize>(o.hidden.size() * 2));
    f.write(reinterpret_cast<const char*>(o.picks.data()), static_cast<std::streamsize>(o.picks.size() * 4));
    std::printf("[ .. ] case %zu: context %d, anchor %lld, proposals %s\n", c, o.p, static_cast<long long>(o.anchor),
                ids_text(o.picks).c_str());
  }
  require(f.good(), "cannot write " + path);
  std::printf("[ OK ] wrote %zu draft cases at world %d to %s\n", outs[0].size(), world, path.c_str());
}

// ---- the structural checks --------------------------------------------------------

bool same(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

int run_checks(const std::string& dir, bool native) {
  const Inkling2Config cfg = ink2fx::draft_config();
  const int B = cfg.draft.block_size;
  constexpr int kMaxTokens = 128;
  constexpr int64_t kCache = 512;

  // 1. The memory plan and the pool carry the draft model.
  {
    const auto plan = [&](bool mtp) {
      return Inkling2Model::plan_memory(cfg, kMaxTokens, kCache, 0, 1, Inkling2Residency::Resident, 2, mtp,
                                        2 * dgpp::kSpecRows, dgpp::LatentFormat::kBf16, false, native);
    };
    const dgpp::MemoryPlan plain = plan(false), with = plan(true);
    require(with.items.size() > plain.items.size(), "the plan has no lines for the draft model");
    const size_t block_plain = Inkling2Model::kv_block_bytes_static(cfg, 0, 1, dgpp::LatentFormat::kBf16, false);
    const size_t block_draft = Inkling2Model::kv_block_bytes_static(cfg, 0, 1, dgpp::LatentFormat::kBf16, true);
    const size_t planes = static_cast<size_t>(cfg.draft.num_hidden_layers) * 2 *
                          static_cast<size_t>(cfg.draft.num_key_value_heads) *
                          static_cast<size_t>(cfg.draft.head_dim) * 2 *
                          static_cast<size_t>(Inkling2Model::kv_block_tokens_static());
    require(block_draft == block_plain + planes, "a cache block grows by the draft layers' keys and values");
    require(Inkling2Model::session_snapshot_bytes(cfg, 1, true) > Inkling2Model::session_snapshot_bytes(cfg, 1, false),
            "a session snapshot carries the draft's feature rows");
    std::printf("[ OK ] the plan, a cache block (+%zu bytes) and a snapshot carry the draft model\n", planes);
  }

  // 2. The draft model reads the target and never moves it: prefill, verify
  //    and the step after a rollback are bitwise the plain model's at every
  //    accepted length, and a block comes out after each.
  {
    auto plain = make_model(cfg, dir, kMaxTokens, kCache, nullptr, 0, 1, 2, false, native);
    auto draft = make_model(cfg, dir, kMaxTokens, kCache, nullptr, 0, 1, 2, true, native);
    require(draft->dflash2_enabled() && draft->dflash2_drafts() == B - 1 && draft->block_draft_rows() == B,
            "the draft model's block");
    require(!plain->dflash2_enabled() && plain->dflash2_drafts() == 0, "the plain model drafts nothing");
    for (int accepted = 1; accepted <= B; ++accepted) {
      const std::vector<int64_t> prompt = smoke_tokens(cfg, 70 + accepted, 0x9E3779B97F4A7C15ull + accepted);
      const std::vector<int64_t> feed = smoke_tokens(cfg, B, 0xD1B54A32D192ED03ull + accepted);
      require(same(plain->session_prefill(0, prompt).logits, draft->session_prefill(0, prompt).logits),
              "the taps changed the target's prefill");
      std::vector<int32_t> proposals;
      require(draft->dflash2_draft(0, feed[0], &proposals) && static_cast<int>(proposals.size()) == B - 1,
              "a block after the prefill");
      for (const int32_t t : proposals) require(t >= 0 && t < cfg.vocab_size, "a proposal outside the vocabulary");
      std::vector<int32_t> again;
      require(draft->dflash2_draft(0, feed[0], &again) && again == proposals, "a redraft of one state differs");
      require(same(plain->session_verify(0, feed).logits, draft->session_verify(0, feed).logits),
              "the draft model changed the target's verify rows");
      plain->session_rollback(0, accepted, B);
      draft->session_rollback(0, accepted, B);
      require(same(plain->session_step(0, 117).logits, draft->session_step(0, 117).logits),
              "the draft model changed the step after a rollback");
      require(draft->dflash2_draft(0, 211, &proposals), "a block after the rollback");
      plain->session_close(0);
      draft->session_close(0);
    }
    require(draft->kv_blocks_in_use() == 0, "the draft model leaked blocks");
    std::printf("[ OK ] the target's rows are bitwise the plain model's at accepted lengths 1 to %d\n", B);

    // 3. Two slots draft together what each drafts alone.
    {
      const std::vector<int64_t> pa = smoke_tokens(cfg, 37, 11), pb = smoke_tokens(cfg, 66, 12);
      draft->session_prefill(0, pa);
      draft->session_prefill(1, pb);
      std::vector<int32_t> a, b;
      require(draft->dflash2_draft(0, 5, &a) && draft->dflash2_draft(1, 9, &b), "the slots' own blocks");
      std::vector<std::vector<int32_t>> both;
      draft->dflash2_draft_batch({0, 1}, {5, 9}, &both);
      require(both.size() == 2 && both[0].size() == a.size() && both[1].size() == b.size(), "the stacked blocks");
      int differing = 0;
      for (size_t i = 0; i < a.size(); ++i) differing += (both[0][i] != a[i]) + (both[1][i] != b[i]);
      std::printf("[ .. ] two stacked blocks: slot 0 %s | slot 1 %s (%d proposals differ from the slots' own)\n",
                  ids_text(both[0]).c_str(), ids_text(both[1]).c_str(), differing);
      // The stacked rows take another GEMM shape, so a proposal may move at
      // a near tie. A slot reading its neighbour's context would move all.
      require(both[0][0] == a[0] && both[1][0] == b[0], "a stacked block's first proposal differs from the slot's own");
      require(differing <= 2, "the stacked blocks differ from the slots' own");

      // The recurrent head stops once no later proposal can be verified. A
      // floor above any survival stops it after its first step: the first
      // proposal and its confidence are the full block's, every later
      // confidence is the floor value no schedule verifies, and a zero
      // floor reproduces the full block.
      const auto confidence = [&] {
        DGPP_CUDA_OK(cudaStreamSynchronize(draft->stream()));
        std::vector<float> c(static_cast<size_t>(B - 1));
        DGPP_CUDA_OK(cudaMemcpy(c.data(), draft->dflash2_confidence(), c.size() * sizeof(float), cudaMemcpyDeviceToHost));
        return c;
      };
      std::vector<int32_t> stopped, again;
      require(draft->dflash2_draft(0, 5, &again) && again == a, "the block before the stop check");
      const std::vector<float> full_conf = confidence();
      draft->drafter().set_eager_stop_floor(2.0f);
      require(draft->dflash2_draft(0, 5, &stopped), "a stopped block");
      const std::vector<float> stopped_conf = confidence();
      draft->drafter().set_eager_stop_floor(0.0f);
      require(stopped[0] == a[0] && stopped_conf[0] == full_conf[0], "the first proposal runs before any stop");
      for (size_t i = 1; i < stopped_conf.size(); ++i)
        require(stopped_conf[i] == -30.0f, "a stopped step's confidence is the value no schedule verifies");
      for (const int32_t t : stopped) require(t >= 0 && t < cfg.vocab_size, "a stopped step left an invalid token");
      require(draft->dflash2_draft(0, 5, &again) && again == a && confidence() == full_conf,
              "a zero floor reproduces the full block");
      draft->session_close(0);
      draft->session_close(1);
    }
    std::printf("[ OK ] stacked blocks follow the slots' own, and the recurrent head stops at its floor\n");
  }

  // 4. The eager engine with the draft model: it advances and commits the
  //    plain engine's tokens. A verify row is the plain step at its position
  //    bit for bit through a request's eight rows, so the draft changes how
  //    many tokens a step yields and nothing else.
  {
    const std::vector<int64_t> prompt = smoke_tokens(cfg, 63, 0x2545F4914F6CDD1Dull);
    const auto transcript = [&](bool mtp, std::vector<int>* lengths) {
      auto m = make_model(cfg, dir, kMaxTokens, kCache, nullptr, 0, 1, 2, mtp, native);
      EagerEngineAdapter<Inkling2Model> engine(m.get(), 2, dgpp::make_w1_pick(cfg.vocab_size));
      std::vector<int32_t> out{engine.prefill(0, prompt)};
      while (out.size() < 40) {
        engine.reserve(0, m->session_position(0) + B);
        const std::vector<int32_t> next = engine.step(0);
        require(!next.empty(), "the engine made no progress");
        if (lengths) lengths->push_back(static_cast<int>(next.size()));
        out.insert(out.end(), next.begin(), next.end());
      }
      engine.close(0);
      require(m->kv_blocks_in_use() == 0, "the engine left blocks in the pool");
      out.resize(40);
      return out;
    };
    std::vector<int> lengths;
    const std::vector<int32_t> plain = transcript(false, nullptr), spec = transcript(true, &lengths);
    size_t agree = 0;
    while (agree < plain.size() && plain[agree] == spec[agree]) ++agree;
    int longest = 0;
    for (const int n : lengths) longest = std::max(longest, n);
    std::printf("[ .. ] eager engine: %zu steps for 40 tokens with the draft model, longest step %d\n", lengths.size(),
                longest);
    require(agree == plain.size(), "the draft model changed the committed tokens after " + std::to_string(agree) +
                                       " of " + std::to_string(plain.size()));
    std::printf("[ OK ] the eager engine commits the plain engine's tokens with the draft model\n");
  }
  return 0;
}

// ---- a release through the server's decode path -----------------------------------------

// --graph DEPTH: the recorded passes and the device-side pick, as a serving
// world runs them (a loopback world of two, one request at a time, greedy
// through the sampling pick), with and without the draft. The committed
// tokens must be the same. DEPTH: the drafts a pass verifies (7: the whole
// block every pass, -1: the depths cycle). `before`: another request through
// the slot first.
int run_graph_gate(const std::string& dir, const std::vector<int64_t>& ids, const std::vector<int64_t>& before, int steps,
                   bool native, int depth, uint16_t port, int world, int slots, bool warm, int forward_rows,
                   int64_t pool_tokens) {
  const Inkling2Config cfg = Inkling2Config::from_checkpoint_dir(dir);
  const int B = cfg.draft.block_size;
  const int max_tokens = forward_rows > 0 ? forward_rows : std::max(256, static_cast<int>(std::max(ids.size(), before.size())));
  const int64_t cache = pool_tokens > 0 ? pool_tokens
                                        : ((static_cast<int64_t>(std::max(ids.size(), before.size())) + steps + 4 * B) / 4096 + 1) * 4096;
  const auto run = [&](bool mtp) {
    std::vector<std::unique_ptr<CollectiveBus>> buses;
    for (int r = 0; r < world; ++r) {
      BusOptions o = loop_options(r, world, port);
      // The widest decode boundary: every slot's block of BF16 hidden rows.
      o.lat_slot_bytes = std::max<size_t>(o.lat_slot_bytes, static_cast<size_t>(slots) * dgpp::kSpecRows * cfg.hidden_size * 2);
      buses.push_back(std::make_unique<CollectiveBus>(o));
    }
    std::vector<std::string> errors(static_cast<size_t>(world));
    {
      std::vector<std::thread> starters;
      for (int r = 0; r < world; ++r)
        starters.emplace_back([&, r] { (void)buses[static_cast<size_t>(r)]->start(&errors[static_cast<size_t>(r)]); });
      for (auto& t : starters) t.join();
    }
    for (const std::string& e : errors) require(e.empty(), "the loopback bus: " + e);
    Barrier barrier(world);
    std::vector<std::vector<std::vector<int32_t>>> transcripts(static_cast<size_t>(world));
    std::vector<std::thread> workers;
    for (int r = 0; r < world; ++r)
      workers.emplace_back([&, r] {
        uint16_t *scratch = nullptr, *prefix = nullptr, *gather = nullptr;
        bool arrived = false;
        try {
          BusBoundaryReducer reducer(*buses[static_cast<size_t>(r)], 60000);
          auto m = std::make_unique<Inkling2Model>(cfg, dir, max_tokens, cache, Inkling2Residency::Resident,
                                                   world > 1 ? &reducer : nullptr, r, world, slots, mtp,
                                                   slots * dgpp::kSpecRows, dgpp::LatentFormat::kBf16,
                                                   /*serving_logits=*/true, native);
          DGPP_CUDA_OK(cudaHostAlloc(&scratch, sizeof(uint16_t) * dgpp::kPickScratchElems(world), cudaHostAllocDefault));
          DGPP_CUDA_OK(cudaHostAlloc(&prefix, 2 * dgpp::fabric_sampling_prefix_scratch_elems(world), cudaHostAllocDefault));
          DGPP_CUDA_OK(cudaHostAlloc(&gather, 2 * dgpp::sampling_gather_scratch_elems(cfg.vocab_size), cudaHostAllocDefault));
          {
            dgpp::GraphEngineAdapter<Inkling2Model> engine(m.get(), buses[static_cast<size_t>(r)].get(), r, world, scratch,
                                                           cfg.vocab_size, 60000, 2, prefix, gather, 128);
            int tick = 0;
            if (mtp && depth != B - 1) {
              engine.configure_verify_schedule(true, 8.f, 0.045f, 0, 28.f, false, 6);
              engine.set_verify_depth_hook([&, depth](int, int, const float*, int) {
                const int cycle[] = {7, 4, 0, 1, 5, 2, 6, 3};
                return depth >= 0 ? depth : cycle[tick++ % 8];
              });
            }
            arrived = true;
            barrier.arrive_and_wait();
            // A serving world records its passes at boot from a dummy prompt.
            if (warm) engine.warm_captures(std::vector<int64_t>(4, 0));
            for (const std::vector<int64_t>* prompt : {&before, &ids}) {
              if (prompt->empty()) continue;
              dgpp::sample::Params greedy;
              greedy.temperature = 0.f;
              engine.configure_sampling(0, greedy, 1234);
              // The scheduler's resumable prefill, as a serving world admits
              // a prompt (--warm), or the one-shot prefill.
              int32_t first = -1;
              if (warm) {
                engine.begin_prefill(0, *prompt, static_cast<int64_t>(prompt->size()) + steps + 4 * B, 256,
                                     dgpp::sched::SchedulerEngine::PrefixPrefill{});
                while (first < 0) first = engine.advance_prefill(0, 0).first_token;
              } else {
                first = engine.prefill(0, *prompt);
              }
              std::vector<int32_t> out{first};
              while (static_cast<int>(out.size()) < steps + 1) {
                engine.reserve(0, m->session_position(0) + B);
                const auto generated = engine.step_batch(std::vector<int>{0});
                engine.drain();
                require(!generated[0].empty(), "the graph engine made no progress");
                out.insert(out.end(), generated[0].begin(), generated[0].end());
              }
              engine.close(0);
              out.resize(static_cast<size_t>(steps) + 1);
              transcripts[static_cast<size_t>(r)].push_back(std::move(out));
            }
          }
        } catch (const std::exception& e) {
          errors[static_cast<size_t>(r)] = "rank " + std::to_string(r) + ": " + e.what();
          if (!arrived) barrier.arrive_and_wait();
        }
        if (scratch) cudaFreeHost(scratch);
        if (prefix) cudaFreeHost(prefix);
        if (gather) cudaFreeHost(gather);
      });
    for (auto& t : workers) t.join();
    for (auto& bus : buses) bus->stop();
    for (const std::string& e : errors) require(e.empty(), e);
    for (int r = 1; r < world; ++r)
      require(transcripts[0] == transcripts[static_cast<size_t>(r)], "the ranks committed different tokens");
    return transcripts[0];
  };
  const auto plain = run(false);
  const auto drafted = run(true);
  bool same = true;
  for (size_t q = 0; q < plain.size(); ++q) {
    size_t agree = 0;
    while (agree < plain[q].size() && plain[q][agree] == drafted[q][agree]) ++agree;
    std::printf("request %zu: the transcripts agree on the first %zu of %zu tokens\n", q, agree, plain[q].size());
    std::string head;
    for (size_t i = 0; i < std::min<size_t>(8, plain[q].size()); ++i) head += " " + std::to_string(plain[q][i]);
    std::printf("request %zu: the plain transcript starts%s\n", q, head.c_str());
    same = same && agree == plain[q].size();
  }
  std::printf("graph gate (depth %d, %s activations): %s\n", depth, native ? "native" : "bf16", same ? "PASS" : "FAIL");
  return same ? 0 : 2;
}

// ---- a release: greedy decoding with and without the draft model -----------------------

// `fit` (bias, slope) replaces the draft's confidence fit, and `conf_out`
// takes one line per step: the tokens the step committed and the confidence
// of each proposal in its block. A run with the identity fit (0,1) records
// the raw logit a calibration regresses the accepted proposals against.
// `before` (--before FILE): another request decoded through the slot first.
// Whatever it leaves in the slot must not change the transcript.
int run_release(const std::string& dir, const std::vector<int64_t>& ids, int steps, bool native,
                const std::vector<float>& fit, const std::string& conf_out, const std::vector<int64_t>& before) {
  const Inkling2Config cfg = Inkling2Config::from_checkpoint_dir(dir);
  require(cfg.draft.present, "the checkpoint bundles no draft model");
  const int B = cfg.draft.block_size;
  const int max_tokens = std::max(256, static_cast<int>(std::max(ids.size(), before.size())));
  const int64_t cache = ((static_cast<int64_t>(std::max(ids.size(), before.size())) + steps + 4 * B) / 4096 + 1) * 4096;
  struct Result {
    std::vector<int32_t> tokens;
    std::vector<int> lengths;
    double seconds = 0;
  };
  const auto decode = [&](bool mtp) {
    Result r;
    auto m = make_model(cfg, dir, max_tokens, cache, nullptr, 0, 1, 1, mtp, native);
    if (mtp && fit.size() == 2) m->drafter().set_confidence_fit(fit[0], fit[1]);
    std::FILE* records = mtp && !conf_out.empty() ? std::fopen(conf_out.c_str(), "w") : nullptr;
    require(!mtp || conf_out.empty() || records != nullptr, "cannot write " + conf_out);
    EagerEngineAdapter<Inkling2Model> engine(m.get(), 1, dgpp::make_w1_pick(cfg.vocab_size));
    if (!before.empty()) {
      int made = 1;
      (void)engine.prefill(0, before);
      while (made < steps / 2) {
        engine.reserve(0, m->session_position(0) + B);
        made += static_cast<int>(engine.step(0).size());
      }
      engine.close(0);
    }
    r.tokens.push_back(engine.prefill(0, ids));
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> conf(static_cast<size_t>(B - 1));
    bool drafted = false;  // the first step drafts inside itself: no estimate to read before it
    while (static_cast<int>(r.tokens.size()) < steps + 1) {
      engine.reserve(0, m->session_position(0) + B);
      if (records && drafted) {
        DGPP_CUDA_OK(cudaStreamSynchronize(m->stream()));
        DGPP_CUDA_OK(cudaMemcpy(conf.data(), m->dflash2_confidence(), conf.size() * sizeof(float), cudaMemcpyDeviceToHost));
      }
      const std::vector<int32_t> next = engine.step(0);
      require(!next.empty(), "the engine made no progress");
      if (records && drafted) {
        std::fprintf(records, "%zu", next.size());
        for (const float c : conf) std::fprintf(records, " %.6g", c);
        std::fprintf(records, "\n");
      }
      drafted = true;
      r.lengths.push_back(static_cast<int>(next.size()));
      r.tokens.insert(r.tokens.end(), next.begin(), next.end());
    }
    if (records) std::fclose(records);
    DGPP_CUDA_OK(cudaStreamSynchronize(m->stream()));
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    engine.close(0);
    return r;
  };
  const Result plain = decode(false);
  const Result spec = decode(true);
  const size_t n = std::min(plain.tokens.size(), spec.tokens.size());
  size_t agree = 0;
  while (agree < n && plain.tokens[agree] == spec.tokens[agree]) ++agree;
  std::vector<int> histogram(static_cast<size_t>(B) + 1, 0);
  for (const int len : spec.lengths) ++histogram[static_cast<size_t>(std::min(len, B))];
  std::string h;
  for (int len = 1; len <= B; ++len) h += " " + std::to_string(len) + ":" + std::to_string(histogram[static_cast<size_t>(len)]);
  const double plain_tokens = static_cast<double>(plain.tokens.size() - 1);
  const double spec_tokens = static_cast<double>(spec.tokens.size() - 1);
  std::printf("plain: %zu tokens in %.3f s, %.2f ms a token\n", plain.tokens.size() - 1, plain.seconds,
              1e3 * plain.seconds / plain_tokens);
  std::printf("draft: %zu tokens in %zu steps (%.2f tokens a step), %.3f s, %.2f ms a token, %.2f ms a step\n",
              spec.tokens.size() - 1, spec.lengths.size(), spec_tokens / static_cast<double>(spec.lengths.size()),
              spec.seconds, 1e3 * spec.seconds / spec_tokens, 1e3 * spec.seconds / static_cast<double>(spec.lengths.size()));
  std::printf("tokens a step:%s\n", h.c_str());
  std::printf("the transcripts agree on the first %zu of %zu tokens%s\n", agree, n,
              before.empty() ? "" : " (another request went through the slot first)");
  return agree == n ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  std::string write_dir, fixture, checkpoint, dump, ids_path, activations = "bf16", conf_out, before_path;
  int graph_depth = -2, slots = 1, forward_rows = 0;
  int64_t pool_tokens = 0;
  bool warm = false;
  std::vector<float> fit;
  int world = 1, port = 30070, steps = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
      return argv[++i];
    };
    try {
      if (a == "--write-fixture") write_dir = value();
      else if (a == "--fixture") fixture = value();
      else if (a == "--checkpoint-dir") checkpoint = value();
      else if (a == "--dump") dump = value();
      else if (a == "--ids") ids_path = value();
      else if (a == "--world") world = std::stoi(value());
      else if (a == "--port") port = std::stoi(value());
      else if (a == "--steps") steps = std::stoi(value());
      else if (a == "--activations") activations = value();
      else if (a == "--conf-out") conf_out = value();
      else if (a == "--before") before_path = value();
      else if (a == "--graph") graph_depth = std::stoi(value());
      else if (a == "--slots") slots = std::stoi(value());
      else if (a == "--warm") warm = true;
      else if (a == "--forward-rows") forward_rows = std::stoi(value());
      else if (a == "--pool") pool_tokens = std::stoll(value());
      else if (a == "--conf-fit") {
        std::string v = value();
        for (size_t at = 0; at <= v.size();) {
          const size_t end = std::min(v.find(',', at), v.size());
          fit.push_back(std::stof(v.substr(at, end - at)));
          at = end + 1;
        }
        if (fit.size() != 2) throw std::invalid_argument("--conf-fit takes bias,slope");
      }
      else throw std::invalid_argument("unknown argument " + a);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "inkling2_draft_test: %s\n", e.what());
      return 2;
    }
  }
  if (activations != "bf16" && activations != "native") {
    std::fprintf(stderr, "inkling2_draft_test: --activations is bf16 or native\n");
    return 2;
  }
  const bool native = activations == "native";
  try {
    if (!write_dir.empty()) {
      ink2fx::write_draft_fixture(write_dir);
      return 0;
    }
    if (!fixture.empty() && dump.empty()) return run_checks(fixture, native);
    if (!fixture.empty()) {
      const Inkling2Config cfg = ink2fx::draft_config();
      const std::vector<Case> cases = fixture_cases();
      const std::vector<int64_t> ids = smoke_tokens(cfg, 96, 0x9E3779B97F4A7C15ull);
      write_dump(dump, cfg, world,
                 run_cases(cfg, fixture, world, static_cast<uint16_t>(port), cases, ids, 128, 512, native));
      return 0;
    }
    if (checkpoint.empty()) {
      std::fprintf(stderr, "inkling2_draft_test: --write-fixture, --fixture or --checkpoint-dir\n");
      return 2;
    }
    const Inkling2Config cfg = Inkling2Config::from_checkpoint_dir(checkpoint);
    if (!dump.empty()) {
      const std::vector<int64_t> ids = ids_path.empty() ? smoke_tokens(cfg, 2200, 0x9E3779B97F4A7C15ull)
                                                        : ink2ref::read_ids(ids_path);
      // A context inside the window, one past it through decode rows and a
      // partly kept verify, and one whose first positions the sliding
      // layers no longer see.
      std::vector<Case> cases = {{24, 0, 0, 0}, {300, 5, 8, 3}};
      if (static_cast<int>(ids.size()) > cfg.draft.sliding_window_size + 200)
        cases.push_back({cfg.draft.sliding_window_size + 150, 0, 0, 0});
      write_dump(dump, cfg, world,
                 run_cases(cfg, checkpoint, world, static_cast<uint16_t>(port), cases, ids, 2048, 8192, native));
      return 0;
    }
    if (ids_path.empty() || steps <= 0) {
      std::fprintf(stderr, "inkling2_draft_test: --checkpoint-dir takes --dump FILE, or --ids FILE --steps N\n");
      return 2;
    }
    const std::vector<int64_t> before_ids = before_path.empty() ? std::vector<int64_t>{} : ink2ref::read_ids(before_path);
    if (graph_depth > -2)
      return run_graph_gate(checkpoint, ink2ref::read_ids(ids_path), before_ids, steps, native, graph_depth,
                            static_cast<uint16_t>(port > 0 ? port : 30079), world, slots, warm, forward_rows, pool_tokens);
    return run_release(checkpoint, ink2ref::read_ids(ids_path), steps, native, fit, conf_out, before_ids);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[FAIL] %s\n", e.what());
    return 1;
  }
}
