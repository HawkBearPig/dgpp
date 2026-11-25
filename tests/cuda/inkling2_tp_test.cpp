// The Inkling 2 tensor-parallel forwards (docs/inkling2_mini_plan.md section 2):
// a loopback world of 2 or 4 ranks in one process (real verbs QPs over
// 127.0.0.1, the bus the fabric uses) over a checkpoint directory.
//
//   --checkpoint-dir DIR --dump-file FILE --world W --port P
//       [--activations bf16|native] [--relaxed | --report]
//   --checkpoint-dir DIR --loop-check --world W --port P [--activations bf16|native]
//       loop guidance (engine.loop_guidance) on a world of W ranks, held as
//       the world-1 forward test holds it: the gates are listed at
//       loop_check(). Five worlds in turn, on ports P to P + 4.
//
// Asserted:
//   * the geometry: a rank holds whole key/value heads with their query
//     heads, the same slice of every expert's intermediate width on whole
//     32-value scale blocks, and its share of the vocabulary;
//   * cross-rank: every visit's state, attention contribution and expert
//     input, the routes and the final hidden state are bitwise identical on
//     every rank (the canonical rank-order fold; a divergence is the
//     silent-corruption class);
//   * against the reference dump, as the world-1 test judges it: FILE is a
//     teacher-forced dump generated with --world W, so the reference rounds
//     each rank's partial sums and quantizes each rank's activation slice as
//     the world does. The logits are the ranks' vocabulary slices put back
//     together.
//
// DGPP_INKLING2_DUMP_ENGINE_STATES=PATH writes rank 0's states for the
// teacher. The ctest chain per world and recipe: this test with --report on
// the world-1 dump (writes the states) -> the reference with --teacher and
// --world -> this test, strict.
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "common/log.hpp"
#include "engine/tp_bus.hpp"
#include "inkling2_reference_compare.hpp"
#include "kernels/inkling2.hpp"
#include "kernels/latent_format.hpp"
#include "models/inkling2/binding.hpp"
#include "models/inkling2/config.hpp"
#include "models/inkling2/forward.hpp"
#include "net/collective_bus.hpp"

using dgpp::BusBoundaryReducer;
using dgpp::Inkling2Config;
using dgpp::Inkling2LocalGeometry;
using dgpp::Inkling2Model;
using dgpp::net::BusOptions;
using dgpp::net::CollectiveBus;
using ink2ref::require;

namespace {

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
  // A 2 MiB latency slot holds a whole fold of a few hundred rows.
  o.lat_slot_bytes = 2 << 20;
  o.bulk_slots = 8;
  o.bulk_slot_bytes = 2 << 20;
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
    if (!out[0]->start(&errors[0])) DGPP_LOG_ERROR("inkling2 tp world rank 0: {}", errors[0]);
  });
  std::vector<std::thread> connectors;
  for (int r = 1; r < world; ++r)
    connectors.emplace_back([&, r] {
      if (!out[static_cast<size_t>(r)]->start(&errors[static_cast<size_t>(r)]))
        DGPP_LOG_ERROR("inkling2 tp world rank {}: {}", r, errors[static_cast<size_t>(r)]);
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
  Inkling2Model::Outputs out;
  std::vector<std::vector<uint16_t>> attention, mlp_input;
};

void rank_work(int rank, int world, const std::string& dir, const Inkling2Config& cfg, const std::vector<int64_t>& tokens,
               bool native, CollectiveBus* bus, ConstructBarrier* barrier, RankOutcome* out) {
  bool arrived = false;
  auto arrive_once = [&] {
    if (arrived) return;
    arrived = true;
    barrier->arrive_and_wait();
  };
  try {
    BusBoundaryReducer reducer(*bus, wait_timeout_ms());
    Inkling2Model model(cfg, dir, static_cast<int>(tokens.size()), 4096, dgpp::Inkling2Residency::Streaming, &reducer, rank,
                        world, 1, false, 0, dgpp::LatentFormat::kBf16, false, native);
    arrive_once();
    out->out = model.forward(tokens, true);
    out->attention = model.visit_attention();
    out->mlp_input = model.visit_mlp_input();
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("inkling2 tp rank {} failed: {}", rank, e.what());
    out->error = "rank " + std::to_string(rank) + ": " + e.what();
    arrive_once();
  }
}

void check_geometry(const Inkling2Config& cfg, int world) {
  int vocab = 0;
  for (int r = 0; r < world; ++r) {
    dgpp::inkling2_tp_validate_geometry(cfg, r, world);
    const Inkling2LocalGeometry g =
        Inkling2LocalGeometry::from_config(cfg, r, world, dgpp::Inkling2HeadSharding::VocabSharded);
    require(g.local_heads == cfg.num_attention_heads / world && g.head_begin == r * g.local_heads, "geometry: the query heads");
    require(g.local_kv_heads == cfg.num_key_value_heads / world && g.kv_head_begin == r * g.local_kv_heads,
            "geometry: the kv heads");
    require(g.head_begin / (cfg.num_attention_heads / cfg.num_key_value_heads) == g.kv_head_begin,
            "geometry: the query heads straddle kv heads");
    require(g.local_inter == cfg.intermediate_size / world && g.local_inter % 32 == 0 && g.inter_begin == r * g.local_inter,
            "geometry: the expert slice");
    require(g.lm_vocab_begin == vocab && g.lm_vocab_count > 0, "geometry: the vocabulary slices are not contiguous");
    vocab += g.lm_vocab_count;
  }
  require(vocab == cfg.vocab_size, "geometry: the vocabulary slices do not cover the head's rows");
  const Inkling2LocalGeometry g0 = Inkling2LocalGeometry::from_config(cfg, 0, world, dgpp::Inkling2HeadSharding::VocabSharded);
  std::printf("[ .. ] world %d geometry: %d query heads on %d kv heads, expert slice %lld of %d, %d vocabulary rows on rank 0\n",
              world, g0.local_heads, g0.local_kv_heads, static_cast<long long>(g0.local_inter), cfg.intermediate_size,
              g0.lm_vocab_count);
}

bool bits_equal(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * 2) == 0);
}

// One forward of `tokens` on a loopback world of `world` ranks: every rank's outcome.
std::vector<RankOutcome> forward_world(const std::string& dir, const Inkling2Config& cfg,
                                       const std::vector<int64_t>& tokens, int world, uint16_t port, bool native) {
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(world, port);
  require(!buses.empty(), "tp bus world failed to start");
  std::vector<RankOutcome> ranks(static_cast<size_t>(world));
  ConstructBarrier barrier(world);
  std::vector<std::thread> workers;
  for (int r = 0; r < world; ++r)
    workers.emplace_back(rank_work, r, world, dir, std::cref(cfg), std::cref(tokens), native, buses[static_cast<size_t>(r)].get(),
                         &barrier, &ranks[static_cast<size_t>(r)]);
  for (auto& t : workers) t.join();
  for (int r = 0; r < world; ++r) require(ranks[static_cast<size_t>(r)].error.empty(), ranks[static_cast<size_t>(r)].error);
  return ranks;
}

// The ranks' vocabulary slices put back together: [T, vocabulary].
std::vector<float> joined_logits(const std::vector<RankOutcome>& ranks, const Inkling2Config& cfg, int T) {
  std::vector<float> logits(static_cast<size_t>(T) * cfg.vocab_size, 0.0f);
  int covered = 0;
  for (const RankOutcome& r : ranks) {
    const int begin = r.out.lm_vocab_begin, count = r.out.lm_vocab_count;
    require(begin >= 0 && begin + count <= cfg.vocab_size && r.out.logits.size() == static_cast<size_t>(T) * count,
            "a rank's logits are not its vocabulary slice");
    for (int t = 0; t < T; ++t)
      std::memcpy(logits.data() + static_cast<size_t>(t) * cfg.vocab_size + begin,
                  r.out.logits.data() + static_cast<size_t>(t) * count, static_cast<size_t>(count) * 4);
    covered += count;
  }
  require(covered == cfg.vocab_size, "the ranks' logits do not cover the vocabulary");
  return logits;
}

// Loop guidance on a world of ranks, with the ranks' logits put back
// together. The gates are the world-1 forward test's, at this world:
//   - at strength 0 a guided world gives the unguided world's states and
//     logits bit for bit: the readout's own folds touch nothing the model's
//     walk reads;
//   - at strength -1 the logits are those of a walk of one pass, bit for
//     bit: the readout's visits fold as the model's do;
//   - a guided logit is the host statement of the model's logit and the
//     readout's, bit for bit, every rank moving its own share.
int loop_check(const std::string& dir, int world, uint16_t port, bool native) {
  const Inkling2Config base = Inkling2Config::from_checkpoint_dir(dir);
  const int T = 48;
  std::vector<int64_t> tokens(static_cast<size_t>(T));
  uint64_t seed = 0x9E3779B97F4A7C15ull;
  for (int i = 0; i < T; ++i) {
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    tokens[static_cast<size_t>(i)] = static_cast<int64_t>(seed % static_cast<uint64_t>(base.vocab_size));
  }
  check_geometry(base, world);
  int next_port = port;
  const auto run_world = [&](bool guidance, float strength, int passes = 0) {
    Inkling2Config cfg = base;
    if (passes > 0) cfg.num_passes = passes;
    dgpp::Inkling2LoopOptions o;
    o.guidance = guidance;
    o.strength = strength;
    cfg.apply_loop_options(o);
    return forward_world(dir, cfg, tokens, world, static_cast<uint16_t>(next_port++), native);
  };
  const auto same_bits = [](const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
  };
  const std::vector<RankOutcome> plain_ranks = run_world(false, 0.0f);
  const std::vector<float> plain = joined_logits(plain_ranks, base, T);

  const std::vector<RankOutcome> zero_ranks = run_world(true, 0.0f);
  for (int r = 0; r < world; ++r)
    require(zero_ranks[static_cast<size_t>(r)].out.layer_states == plain_ranks[static_cast<size_t>(r)].out.layer_states,
            "loop check: guidance moved a state of the model's walk on rank " + std::to_string(r));
  require(same_bits(joined_logits(zero_ranks, base, T), plain), "loop check: guidance at strength 0 moved a logit");

  const std::vector<float> one_pass = joined_logits(run_world(false, 0.0f, 1), base, T);
  const std::vector<float> readout = joined_logits(run_world(true, -1.0f), base, T);
  require(same_bits(readout, one_pass), "loop check: the first pass's readout is not the one-pass walk's logits");
  require(!same_bits(readout, plain), "loop check: the first pass reads out as the last one");

  const float w = 0.5f;
  const std::vector<float> guided = joined_logits(run_world(true, w), base, T);
  for (size_t i = 0; i < guided.size(); ++i) {
    const float want = dgpp::ink2_guided_value_host(w, plain[i], readout[i]);
    require(std::memcmp(&guided[i], &want, 4) == 0, "loop check: a guided logit is not its statement");
  }

  std::printf("inkling2 tp loop check, world %d (%s activations): strength 0, the first pass's readout and the guided "
              "statement over the joined vocabulary\n",
              world, native ? "native" : "bf16");
  return 0;
}

// With `ids_path` the run is the teacher-forced likelihood of that token
// file at this world size (a measurement, written to `nll_out` position by
// position); otherwise it is the comparison with the reference dump.
int run(const std::string& dir, const std::string& dump_path, int world, uint16_t port, bool native, bool relaxed, bool report,
        const std::string& ids_path = "", const std::string& nll_out = "") {
  const Inkling2Config cfg = Inkling2Config::from_checkpoint_dir(dir);
  const bool measure = !ids_path.empty();
  ink2ref::Dump d;
  std::vector<int64_t> tokens;
  if (measure) {
    tokens = ink2ref::read_ids(ids_path);
    require(tokens.size() <= 4096, "--ids: at most 4096 tokens in one forward");
  } else {
    d = ink2ref::Dump::load(dump_path);
    tokens.resize(static_cast<size_t>(d.token_count));
    std::memcpy(tokens.data(), d.tensor("tokens").data, tokens.size() * 8);
  }
  const int T = static_cast<int>(tokens.size()), V = cfg.num_visits();
  check_geometry(cfg, world);
  const std::vector<RankOutcome> ranks = forward_world(dir, cfg, tokens, world, port, native);

  // ---- cross-rank bitwise ---------------------------------------------------------
  const RankOutcome& a = ranks[0];
  require(static_cast<int>(a.out.layer_states.size()) == V, "rank 0: one state per visit");
  for (int r = 1; r < world; ++r) {
    const RankOutcome& b = ranks[static_cast<size_t>(r)];
    const std::string who = " differs between rank 0 and rank " + std::to_string(r);
    require(static_cast<int>(b.out.layer_states.size()) == V, "a rank's visit captures are missing");
    for (int v = 0; v < V; ++v) {
      const size_t i = static_cast<size_t>(v);
      require(bits_equal(a.attention[i], b.attention[i]), "visit " + std::to_string(v) + " attention" + who);
      require(bits_equal(a.mlp_input[i], b.mlp_input[i]), "visit " + std::to_string(v) + " expert input" + who);
      require(bits_equal(a.out.layer_states[i], b.out.layer_states[i]), "visit " + std::to_string(v) + " state" + who);
    }
    require(a.out.route_ids == b.out.route_ids, "routing" + who);
    require(bits_equal(a.out.final_hidden_bits, b.out.final_hidden_bits), "final hidden" + who);
  }
  std::printf("[ OK ] world %d: %d visits' states, attention, expert inputs and routes are bitwise identical on every rank\n", world,
              V);

  // ---- against the reference ---------------------------------------------------------
  ink2ref::EngineRun er;
  er.states = a.out.layer_states;
  er.attention = a.attention;
  er.mlp_input = a.mlp_input;
  er.route_ids = a.out.route_ids;
  er.final_hidden = a.out.final_hidden_bits;
  er.logits = joined_logits(ranks, cfg, T);
  if (const char* path = std::getenv("DGPP_INKLING2_DUMP_ENGINE_STATES")) ink2ref::write_states(path, er, T, cfg.hidden_size);
  if (measure) {
    const std::string what =
        "inkling2 world " + std::to_string(world) + " (" + (native ? "native" : "bf16") + " activations)";
    return ink2ref::likelihood(cfg, er.logits, tokens, nll_out, what.c_str());
  }
  const std::string label = "inkling2 tp world " + std::to_string(world);
  return ink2ref::judge(cfg, d, er, relaxed, native, report, label.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string dir, dump, ids, nll_out;
    int world = 0, port = 0;
    bool native = false, relaxed = false, report = false, loop = false;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string {
        require(i + 1 < argc, "missing value after " + a);
        return argv[++i];
      };
      if (a == "--checkpoint-dir") dir = next();
      else if (a == "--dump-file") dump = next();
      else if (a == "--ids") ids = next();
      else if (a == "--nll-out") nll_out = next();
      else if (a == "--world") world = std::stoi(next());
      else if (a == "--port") port = std::stoi(next());
      else if (a == "--loop-check") loop = true;
      else if (a == "--relaxed") relaxed = true;
      else if (a == "--report") report = relaxed = true;
      else if (a == "--activations") {
        const std::string v = next();
        require(v == "bf16" || v == "native", "--activations takes bf16 or native");
        native = v == "native";
      } else throw std::runtime_error("unknown argument " + a);
    }
    if (loop) {
      require(!dir.empty() && (world == 2 || world == 4) && port > 0,
              "usage: --checkpoint-dir DIR --loop-check --world 2|4 --port P [--activations bf16|native]");
      return loop_check(dir, world, static_cast<uint16_t>(port), native);
    }
    require(!dir.empty() && (!dump.empty() || !ids.empty()) && (world == 2 || world == 4) && port > 0,
            "usage: --checkpoint-dir DIR (--dump-file FILE | --ids FILE [--nll-out FILE]) --world 2|4 --port P "
            "[--activations bf16|native] [--report]");
    return run(dir, dump, world, static_cast<uint16_t>(port), native, relaxed, report, ids, nll_out);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "inkling2_tp_test: %s\n", e.what());
    return 1;
  }
}
