#include <algorithm>
#include <barrier>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "adapters/bus_sum.hpp"
#include "common/bf16_residency.hpp"
#include "common/cuda_check.hpp"
#include "engine/eager_engine.hpp"
#include "engine/graph_engine.hpp"
#include "glm_dsa_fixture.hpp"
#include "kernels/low_rank.hpp"
#include "models/glm_dsa/model.hpp"

namespace {
using namespace dgpp;
constexpr int kSlots = 8, kRows = 256, kDecodeRows = 16, kSteps = 12;
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
template <class T>
void exact(const std::vector<T>& a, const std::vector<T>& b, const char* message) {
  require(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0,
          message);
}

// The graph boundary's two fused kernels against the kernels they replace,
// bit for bit, at shapes the model fixture does not reach: the projection
// that writes its own digits against project + pack_digits (in two row
// ranges, so the second starts mid rider), and the kernel behind the
// collective against sum_digits + add. The shapes cover every block layout
// (two, four and eight rows of threads a column), rows that loop a thread,
// a partial last block, and a rank too wide for shared memory.
void kernel_check(int world) {
  if (world == 1) return;
  struct Shape { int rows, rank, outputs, inputs; };
  const Shape shapes[] = {{1, 50, 6144, 1536}, {2, 50, 6144, 4096}, {3, 50, 6144, 1536}, {5, 50, 6144, 1536},
                          {8, 50, 6144, 1536}, {20, 50, 6144, 257}, {2, 7, 100, 33},      {4, 96, 6144, 512},
                          {2, 200, 1000, 64},  {2, 400, 512, 64}};
  uint32_t seed = 0x9e3779b9u;
  const auto rnd = [&] { return seed = seed * 1664525u + 1013904223u; };
  const auto unit = [&] { return static_cast<float>(rnd() >> 8) / 8388608.0f - 1.0f; };
  for (const Shape& sh : shapes) {
    const size_t count = static_cast<size_t>(sh.rows) * sh.rank, digit_elems = count * world * 4;
    std::vector<uint16_t> x(static_cast<size_t>(sh.rows) * sh.inputs), base(static_cast<size_t>(sh.rows) * sh.outputs);
    std::vector<float> b(static_cast<size_t>(sh.outputs) * sh.rank);
    for (auto& v : b) v = unit();
    for (auto& v : base) v = float_to_bf16_bits(4.0f * unit());
    uint16_t *dx, *dout, *ddigits;
    float *da, *db, *dz;
    DGPP_CUDA_OK(cudaMalloc(&dx, x.size() * 2));
    DGPP_CUDA_OK(cudaMalloc(&dout, base.size() * 2));
    DGPP_CUDA_OK(cudaMalloc(&ddigits, digit_elems * 2));
    DGPP_CUDA_OK(cudaMalloc(&da, static_cast<size_t>(sh.rank) * sh.inputs * 4));
    DGPP_CUDA_OK(cudaMalloc(&db, b.size() * 4));
    DGPP_CUDA_OK(cudaMalloc(&dz, count * 4));
    DGPP_CUDA_OK(cudaMemcpy(db, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
    std::vector<float> folded(digit_elems, 0.0f);
    for (int r = 0; r < world; ++r) {
      std::vector<float> a(static_cast<size_t>(sh.rank) * sh.inputs);
      for (auto& v : a) v = unit();
      for (auto& v : x) v = float_to_bf16_bits(unit());
      DGPP_CUDA_OK(cudaMemcpy(da, a.data(), a.size() * 4, cudaMemcpyHostToDevice));
      DGPP_CUDA_OK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
      std::vector<uint16_t> packed(digit_elems), written(digit_elems);
      low_rank_project(dx, sh.inputs, da, dz, sh.rows, sh.inputs, sh.rank, nullptr);
      low_rank_pack_digits(dz, ddigits, count, r, world, nullptr);
      DGPP_CUDA_OK(cudaMemcpy(packed.data(), ddigits, digit_elems * 2, cudaMemcpyDeviceToHost));
      DGPP_CUDA_OK(cudaMemset(ddigits, 0xff, digit_elems * 2));
      const int head = sh.rows / 2;  // the rows in two launches, as a boundary's spans project them
      if (head)
        low_rank_project_digits(dx, sh.inputs, da, dz, head, sh.inputs, sh.rank, ddigits, 0, count, r, world, nullptr);
      low_rank_project_digits(dx + static_cast<size_t>(head) * sh.inputs, sh.inputs, da,
                              dz + static_cast<size_t>(head) * sh.rank, sh.rows - head, sh.inputs, sh.rank, ddigits,
                              static_cast<size_t>(head) * sh.rank, count, r, world, nullptr);
      DGPP_CUDA_OK(cudaMemcpy(written.data(), ddigits, digit_elems * 2, cudaMemcpyDeviceToHost));
      exact(written, packed, "the projection's own digits differ from pack_digits");
      for (size_t i = 0; i < digit_elems; ++i) folded[i] += bf16_bits_to_float(packed[i]);
    }
    std::vector<uint16_t> digits(digit_elems);  // what the collective leaves: the ranks' digits summed in bf16
    for (size_t i = 0; i < digit_elems; ++i) digits[i] = float_to_bf16_bits(folded[i]);
    DGPP_CUDA_OK(cudaMemcpy(ddigits, digits.data(), digit_elems * 2, cudaMemcpyHostToDevice));
    std::vector<uint16_t> want(base.size()), got(base.size());
    std::vector<float> z_want(count), z_got(count);
    DGPP_CUDA_OK(cudaMemcpy(dout, base.data(), base.size() * 2, cudaMemcpyHostToDevice));
    low_rank_sum_digits(ddigits, dz, count, world, nullptr);
    low_rank_add(dz, db, dout, sh.rows, sh.outputs, sh.rank, nullptr);
    DGPP_CUDA_OK(cudaMemcpy(want.data(), dout, want.size() * 2, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(z_want.data(), dz, count * 4, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(dout, base.data(), base.size() * 2, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemset(dz, 0, count * 4));
    low_rank_add_digits(ddigits, world, dz, db, dout, sh.rows, sh.outputs, sh.rank, nullptr);
    DGPP_CUDA_OK(cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(z_got.data(), dz, count * 4, cudaMemcpyDeviceToHost));
    exact(got, want, "the fused boundary kernel differs from sum_digits + add");
    exact(z_got, z_want, "the fused boundary kernel leaves other rank-space rows than sum_digits");
    require(std::memcmp(want.data(), base.data(), want.size() * 2) != 0, "the adapter kernels added nothing");
    for (void* q : {static_cast<void*>(dx), static_cast<void*>(dout), static_cast<void*>(ddigits),
                    static_cast<void*>(da), static_cast<void*>(db), static_cast<void*>(dz)})
      DGPP_CUDA_OK(cudaFree(q));
  }
}

template <class Work>
void world_run(int world, int port, Work work, int extra_graph_nodes = 0,
               size_t latency_bytes = 8192) {
  std::vector<std::unique_ptr<net::CollectiveBus>> buses;
  for (int r = 0; r < world; ++r) {
    net::BusOptions o;
    o.my_rank = r;
    o.world_size = world;
    o.rendezvous_host = r ? "127.0.0.1" : "";
    o.rendezvous_port = port;
    o.rendezvous_timeout_ms = 20000;
    o.lat_slots = 8;
    o.lat_slot_bytes = latency_bytes;
    o.bulk_slots = 8;
    o.bulk_slot_bytes = 262144;
    o.launch_consumers = false;
    o.latency_transport = net::BusLatencyTransport::kWrite;
    o.graph_nodes_per_variant += extra_graph_nodes;
    buses.push_back(std::make_unique<net::CollectiveBus>(o));
  }
  std::vector<std::string> errors(world);
  std::vector<std::thread> threads;
  for (int r = 0; r < world; ++r) threads.emplace_back([&, r] { buses[r]->start(&errors[r]); });
  for (auto& t : threads) t.join();
  for (const auto& error : errors)
    if (!error.empty()) throw std::runtime_error(error);
  threads.clear();
  std::barrier ready(world);
  for (int r = 0; r < world; ++r)
    threads.emplace_back([&, r] {
      bool arrived = false;
      const auto sync = [&] {
        arrived = true;
        ready.arrive_and_wait();
      };
      try {
        work(r, *buses[r], sync);
      } catch (const std::exception& e) {
        errors[r] = e.what();
        if (!arrived) ready.arrive_and_drop();
      }
    });
  for (auto& t : threads) t.join();
  std::string failures;
  for (int r = 0; r < world; ++r)
    if (!errors[r].empty()) failures += "rank " + std::to_string(r) + ": " + errors[r] + "\n";
  if (!failures.empty()) throw std::runtime_error(failures);
}

void transport_check(int world) {
  if (world == 1) return;
  constexpr size_t count = 4099;  // odd shape, several latency slots, nonzero tail
  world_run(
      world, 29982,
      [&](int r, net::CollectiveBus& bus, auto sync) {
        cudaStream_t stream;
        DGPP_CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        float* values = nullptr;
        DGPP_CUDA_OK(cudaMalloc(&values, count * sizeof(float)));
        uint16_t* padding = nullptr;
        DGPP_CUDA_OK(cudaMalloc(&padding, 2 * sizeof(uint16_t)));
        DGPP_CUDA_OK(cudaMemset(padding, 0, 2 * sizeof(uint16_t)));
        auto fold = adapters::bus_sum_stream(bus, r, world, count, 60000);
        std::vector<float> input(count), expected(count), actual(count);
        const auto value = [](int rank, size_t i) {
          // Include cancellation, signs and exponents that would be lost by a
          // BF16 conversion of the rank-space value itself.
          return std::ldexp(
              static_cast<float>(static_cast<int>((i * 17 + rank * 3) % 127) - 63) / 37.0f,
              static_cast<int>(i % 41) - 20);
        };
        for (size_t i = 0; i < count; ++i) {
          input[i] = value(r, i);
          expected[i] = value(0, i);
          for (int p = 1; p < world; ++p) expected[i] += value(p, i);
        }
        sync();
        DGPP_CUDA_OK(cudaMemcpyAsync(values, input.data(), count * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));
        fold(values, count, stream, false);
        DGPP_CUDA_OK(cudaMemcpyAsync(actual.data(), values, count * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
        DGPP_CUDA_OK(cudaStreamSynchronize(stream));
        exact(actual, expected, "eager adapter fold loses FP32 values");
        std::string error;
        require(bus.graph_record_begin(&error), error.c_str());
        DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        // A full model with adapters can exceed the ordinary 256-node budget.
        // Extra identity nodes exercise the extended allocation and indexing.
        for (int node = 0; node < net::kBusMaxGraphGens; ++node)
          require(bus.allreduce_record(stream, padding, padding, 2, &error), error.c_str());
        fold(values, count, stream, true);
        cudaGraph_t graph;
        DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
        require(bus.graph_record_end(&error), error.c_str());
        cudaGraphExec_t exec;
        DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int replay = 0; replay < 3; ++replay) {
          for (size_t i = 0; i < count; ++i) {
            const float sign = replay % 2 ? -1.0f : 1.0f;
            input[i] = sign * value(r, i);
            expected[i] = sign * value(0, i);
            for (int p = 1; p < world; ++p) expected[i] += sign * value(p, i);
          }
          DGPP_CUDA_OK(cudaMemcpyAsync(values, input.data(), count * sizeof(float),
                                       cudaMemcpyHostToDevice, stream));
          require(bus.graph_replay_arm(&error), error.c_str());
          DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
          DGPP_CUDA_OK(cudaStreamSynchronize(stream));
          require(bus.graph_replay_finish(60000, &error), error.c_str());
          DGPP_CUDA_OK(
              cudaMemcpy(actual.data(), values, count * sizeof(float), cudaMemcpyDeviceToHost));
          exact(actual, expected, "recorded adapter fold differs from the FP32 oracle");
        }
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaFree(values);
        cudaFree(padding);
        cudaStreamDestroy(stream);
      },
      net::kBusMaxGraphGens);
}

adapters::Manifest factors(const GlmDsaTextConfig& cfg, int world, bool zero) {
  adapters::Manifest manifest;
  manifest.world = world;
  manifest.id = zero ? "zero" : "nonzero";
  for (const auto& target : glm_dsa_modification_targets(cfg, 0, world)) {
    adapters::ProjectionFactors p;
    p.target = target;
    p.rank = 3;
    p.a.resize(p.rank * target.input_features);
    p.b.resize(p.rank * target.output_features);
    for (size_t i = 0; i < p.a.size(); ++i)
      p.a[i] = 0.01f * (static_cast<int>((i * 7 + target.layer) % 19) - 9);
    for (size_t i = 0; i < p.b.size(); ++i)
      p.b[i] = zero ? 0 : 0.03f * (static_cast<int>((i * 3 + target.layer) % 17) - 8);
    manifest.projections.push_back(std::move(p));
  }
  return manifest;
}

void model_check(const GlmDsaTextConfig& cfg, const std::string& dir, int world) {
  const auto nonzero = factors(cfg, world, false), zero = factors(cfg, world, true);
  std::vector<std::vector<std::vector<int32_t>>> transcripts(world);
  world_run(
      world, 29983,
      [&](int r, net::CollectiveBus& bus, auto sync) {
        BusBoundaryReducer reducer(bus);
        auto make = [&](bool mtp, const adapters::Manifest* manifest, bool native) {
          auto m = std::make_unique<GlmDsaModel>(cfg, dir, kRows, 4096, GlmDsaResidency::Resident,
                                                 world > 1 ? &reducer : nullptr, r, world, kSlots,
                                                 mtp, kDecodeRows, LatentFormat::kFp8, true);
          if (manifest) {
            adapters::SumFp32 old_fold;
            adapters::SumFp32Stream fold;
            if (world > 1) {
              if (native)
                fold = adapters::bus_sum_stream(bus, r, world, kRows * 3, 60000);
              else
                old_fold = adapters::bus_sum(bus, r, world, kRows * 3, 60000);
            }
            m->set_runtime_adapter(
                std::make_unique<adapters::Runtime>(*manifest, r, world, kRows, cfg.hidden_size,
                                                    cfg.num_hidden_layers, std::move(old_fold),
                                                    true, std::move(fold)),
                true);
          }
          return m;
        };
        auto base = make(false, nullptr, false), identity = make(false, &zero, true);
        auto eager = make(false, &nonzero, false), graph = make(true, &nonzero, true);
        uint16_t* picks = nullptr;
        DGPP_CUDA_OK(cudaHostAlloc(&picks, kPickScratchElems(world) * sizeof(uint16_t),
                                   cudaHostAllocDefault));
        sync();
        std::vector<std::vector<int64_t>> prompts(kSlots);
        for (int i = 0; i < kSlots; ++i)
          for (int j = 0; j < 9 + i; ++j)
            prompts[i].push_back((j * 17 + i * 3 + 7) % cfg.vocab_size);
        const auto plain = base->session_prefill(0, prompts[0]);
        exact(plain.logits, identity->session_prefill(0, prompts[0]).logits,
              "zero adapter changes native prefill");
        const auto reference = eager->session_prefill(0, prompts[0]);
        exact(reference.logits, graph->session_prefill(0, prompts[0]).logits,
              "GPU adapter fold differs from research fold");
        require(plain.logits != reference.logits, "nonzero adapter has no effect");
        base->session_close(0);
        identity->session_close(0);
        eager->session_close(0);
        graph->session_close(0);
        EagerEngineAdapter<GlmDsaModel> oracle(
            eager.get(), kSlots, make_fabric_pick(&bus, r, world, picks, cfg.vocab_size, 60000));
        std::vector<std::vector<int32_t>> want(kSlots);
        for (int i = 0; i < kSlots; ++i) {
          want[i].push_back(oracle.prefill(i, prompts[i]));
          oracle.reserve(i, prompts[i].size() + kSteps + 4);
          while (want[i].size() <= kSteps) {
            const auto next = oracle.step(i);
            want[i].insert(want[i].end(), next.begin(), next.end());
          }
          oracle.close(i);
        }
        {
          GraphEngineAdapter<GlmDsaModel> engine(graph.get(), &bus, r, world, picks, cfg.vocab_size,
                                                 60000, 2, nullptr, nullptr, kSamplingCandidates,
                                                 nullptr, 2);
          std::vector<int> slots;
          auto& got = transcripts[r];
          got.resize(kSlots);
          for (int i = 0; i < kSlots; ++i) {
            slots.push_back(i);
            got[i].push_back(engine.prefill(i, prompts[i]));
            engine.reserve(i, prompts[i].size() + kSteps + 4);
          }
          while (std::any_of(got.begin(), got.end(),
                             [](const auto& row) { return row.size() <= kSteps; })) {
            const auto next = engine.step_batch(slots);
            for (int i = 0; i < kSlots; ++i)
              got[i].insert(got[i].end(), next[i].begin(), next[i].end());
          }
          for (int i = 0; i < kSlots; ++i) {
            got[i].resize(kSteps + 1);
            exact(got[i], want[i], "adapted MTP graph transcript differs from plain eager decode");
            engine.close(i);
          }
          engine.drain();
          std::fprintf(stderr, "native test rank %d: C8 transcript checked, saving prefix\n", r);
          std::vector<int64_t> prompt(97);
          for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = (i * 13 + 3) % cfg.vocab_size;
          std::vector<int64_t> boundaries{64};
          sched::SchedulerEngine::PrefixPrefill save;
          save.boundaries = &boundaries;
          save.snap_slot = 0;
          save.snap_position = 64;
          const auto cold = engine.prefill_cached(0, prompt, &save);
          std::fprintf(stderr, "native test rank %d: prefix prefill complete\n", r);
          require(save.snap_taken, "adapted prefix snapshot was not taken");
          engine.close(0);
          sched::SchedulerEngine::PrefixPrefill restore;
          restore.boundaries = &boundaries;
          restore.attach_slot = 0;
          restore.attach_position = 64;
          require(engine.prefill_cached(7, prompt, &restore) == cold,
                  "adapted prefix reuse changed the first token");
          std::fprintf(stderr, "native test rank %d: prefix reuse checked\n", r);
          engine.close(7);
          engine.prefix_release(0);
          engine.drain();
        }
        cudaFreeHost(picks);
      },
      // Match the existing TP fixture geometry: tiny model prefills fit
      // latency slots instead of making bulk shards empty on some ranks.
      adapters::Runtime::extra_graph_nodes(nonzero, world, cfg.hidden_size), 262144);
  for (int r = 1; r < world; ++r)
    require(transcripts[r] == transcripts[0], "ranks disagree on adapted transcripts");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 4 || argc == 5,
            "usage: runtime_adapter_native_test FIXTURE_DIR int8|m346 WORLD [LAYERS]");
    const int world = std::stoi(argv[3]);
    require(world == 1 || world == 2 || world == 4, "unsupported test world");
    const bool m346 = std::string(argv[2]) == "m346";
    // Dense attention/all experts isolate execution paths from sparse near-tie
    // amplification, as in the existing grouped-prefill numerical fixture.
    const int layers = argc == 5 ? std::stoi(argv[4]) : 5;
    const std::string json = glmdsafx::tiny_config_json(layers, 1, true, 256, 8, false, 0, 8, m346);
    const auto cfg = GlmDsaTextConfig::parse(minijson::parse(json).root);
    glmdsafx::write_fixture(cfg, argv[1], json.c_str());
    set_bf16_residency(Bf16Residency::Bf12);
    GlmDsaLayerStream::set_embed_vocab_sharded(true);
    draft_head_configure(DraftHeadSettings{2});
    kernel_check(world);
    transport_check(world);
    model_check(cfg, argv[1], world);
    std::printf(
        "native adapter %s world %d: boundary kernels, FP32 fold, C8 MTP graphs, FP8 KV, BF12, vocab embeddings and "
        "prefix reuse passed\n",
        argv[2], world);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "runtime_adapter_native_test: %s\n", e.what());
    return 1;
  }
}
