// The streamer's gate (MOONSHOT-1.md section 5, phase 2). The DeepSeek-V4.1
// plan at world 4 under the arena is streamed from a device arena holding
// its layers' bytes, with the design chain's latencies interleaved and the
// synthetic router picking the experts, and must read a layer's bytes at
// 95 percent of the ceiling. The hand-off exits are gated too: a deadline
// ends the kernel with its status, and the host's poison ends it from
// outside. Needs the GPU and about 4 GB of device memory free; skips
// otherwise.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/test.hpp"
#include "dsv41_config_json.hpp"
#include "loaders/minijson.hpp"
#include "models/dsv41/config.hpp"
#include "models/dsv41/loader.hpp"
#include "step/device_plan.hpp"
#include "step/dsv41/arena.hpp"
#include "step/dsv41/plan.hpp"
#include "step/streamer.hpp"
#include "step_reference_host.hpp"

namespace {

using namespace dgpp;
using namespace dgpp::step;

constexpr double kCeilingGBs = 240.0;
constexpr int kLayers = 8;      // the first eight main layers: Engram layers 1 and 14 aside, every phase kind appears
constexpr int kCopies = 2;      // arena copies the layers cycle through, so a layer's bytes are cold
constexpr int kWorld = 4;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

struct Rig {
  Dsv41TextConfig cfg;
  std::vector<LayerPlan> plans;
  DevicePlan dp;
  std::vector<uint8_t*> copies;
  std::vector<const uint8_t*> bases;
  Dsv41StepShape shape;
  ~Rig() {
    for (uint8_t* c : copies) cudaFree(c);
  }
};

// Builds the plan of the first kLayers main layers under the arena and the
// device arena they stream from.
std::unique_ptr<Rig> make_rig(int layers = kLayers, bool consumers = false, bool tile_major = false) {
  auto rig = std::make_unique<Rig>();
  const auto doc = minijson::parse(dsv41_test::config_json());
  rig->cfg = Dsv41TextConfig::parse(doc.root);
  rig->shape.rows = 5;
  rig->shape.draft_rows = 5;
  rig->shape.bf12_head = true;
  const Dsv41LocalGeometry geo = Dsv41LocalGeometry::from_config(rig->cfg, 0, kWorld, LoaderHeadSharding::VocabSharded);
  void* base = reinterpret_cast<void*>(static_cast<uintptr_t>(1) << 40);
  uint64_t max_span = 0;
  for (int l = 0; l < layers; ++l) {
    const std::vector<size_t> placement = dsv41_arena_placement(rig->cfg, l, 0, kWorld);
    Dsv41LayerResident r;
    const size_t span = Dsv41LayerStream::layout_layer(rig->cfg, l, 0, kWorld, base, r, &placement);
    rig->plans.push_back(dsv41_layer_plan(rig->cfg, geo, r, base, span, rig->shape));
    if (tile_major)  // the arena's second layout version, as the loader will write it
      for (Range& rg : rig->plans.back().ranges)
        if (rg.dep == Dependency::Router && rg.format == Format::Fp4E2m1) rg.layout = Layout::Tile16;
    max_span = std::max<uint64_t>(max_span, span);
  }
  rig->dp = lower_plan(rig->plans, 65536, consumers);
  size_t free_bytes = 0, total = 0;
  DGPP_CUDA_OK(cudaMemGetInfo(&free_bytes, &total));
  const size_t need = static_cast<size_t>(max_span) * kCopies + (size_t{256} << 20);
  if (free_bytes < need)
    dgpp::test::skip("needs " + std::to_string(need >> 20) + " MB of device memory, " + std::to_string(free_bytes >> 20) + " free");
  for (int c = 0; c < kCopies; ++c) {
    uint8_t* d = nullptr;
    DGPP_CUDA_OK(cudaMalloc(&d, max_span));
    DGPP_CUDA_OK(cudaMemset(d, 0x3c + c, max_span));
    rig->copies.push_back(d);
  }
  for (int l = 0; l < layers; ++l) rig->bases.push_back(rig->copies[static_cast<size_t>(l % kCopies)]);
  if (consumers) {
    // Payloads hashed over the whole copy, then every layer's e8m0 scale
    // ranges in [112, 136) so no dot product overflows.
    for (int c = 0; c < kCopies; ++c) {
      unsigned long long *offsets = nullptr, *bytes = nullptr;
      std::vector<unsigned long long> h_off = {0}, h_bytes = {max_span};
      DGPP_CUDA_OK(cudaMalloc(&offsets, sizeof(unsigned long long)));
      DGPP_CUDA_OK(cudaMalloc(&bytes, sizeof(unsigned long long)));
      DGPP_CUDA_OK(cudaMemcpy(offsets, h_off.data(), 8, cudaMemcpyHostToDevice));
      DGPP_CUDA_OK(cudaMemcpy(bytes, h_bytes.data(), 8, cudaMemcpyHostToDevice));
      step_stream_fill_ranges(rig->copies[static_cast<size_t>(c)], offsets, bytes, 1, 0x9e37u + c, 0, 256, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      cudaFree(offsets);
      cudaFree(bytes);
      // Then the formats whose bytes must decode finitely: e8m0 scales in
      // [112, 136), e4m3 payloads as normals, F32 scale grids in [0.5, 2).
      struct Pass { Format format; StepFillMode mode; unsigned lo, span, seed; };
      const Pass passes[3] = {{Format::ScaleE8m0, kFillBytes, 112, 24, 0x7f4au}, {Format::Fp8E4m3, kFillE4m3, 0, 0, 0x1357u},
                              {Format::ScaleF32, kFillF32Scales, 0, 0, 0x2468u}};
      for (const Pass& pass : passes) {
        h_off.clear();
        h_bytes.clear();
        for (int l = c; l < layers; l += kCopies)
          for (const Range& r : rig->plans[static_cast<size_t>(l)].ranges)
            if (r.source == Source::Resident && r.format == pass.format) { h_off.push_back(r.offset); h_bytes.push_back(r.bytes); }
        if (h_off.empty()) continue;
        DGPP_CUDA_OK(cudaMalloc(&offsets, h_off.size() * 8));
        DGPP_CUDA_OK(cudaMalloc(&bytes, h_bytes.size() * 8));
        DGPP_CUDA_OK(cudaMemcpy(offsets, h_off.data(), h_off.size() * 8, cudaMemcpyHostToDevice));
        DGPP_CUDA_OK(cudaMemcpy(bytes, h_bytes.data(), h_bytes.size() * 8, cudaMemcpyHostToDevice));
        step_stream_fill_ranges(rig->copies[static_cast<size_t>(c)], offsets, bytes, static_cast<int>(h_off.size()), pass.seed + c, pass.lo,
                                pass.span, nullptr, pass.mode);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        cudaFree(offsets);
        cudaFree(bytes);
      }
    }
  }
  return rig;
}

StreamerConfig design_config() {
  StreamerConfig cfg;
  cfg.workers = std::min(47, step_stream_capacity() - 1);
  cfg.steps = 3;
  cfg.lookahead_layers = 2;
  cfg.lead_cap_bytes = uint64_t{16} << 20;  // the sweep's optimum with every consumer real (8 MB with synthetic ones)
  cfg.lead_bytes_per_us = 240e3f;           // the idle budget at the ceiling
  cfg.max_leaders = 0;
  cfg.deadline_s = 5.0;
  cfg.rows = 5;
  cfg.top_k = 6;
  cfg.seed = 7;
  return cfg;
}

void print_report(const char* title, const StreamReport& r, int layer) {
  std::printf("   %s: status %s, kernel %.1f ms, lead %.1f MB\n", title, r.status_text.c_str(), r.kernel_ms, r.lead_bytes / 1e6);
  for (const PhaseTiming& t : r.phases) {
    if (t.layer != layer) continue;
    const double window = t.end_us - t.begin_us;
    const double util = (t.stream && window > 0 && r.workers > 0) ? t.busy_us / (window * r.workers) : 0.0;
    std::printf("      layer %d %-10s %s %8.1f .. %8.1f us  units %5u  lead %4u  drained %6.1f us  busy %5.1f %%  %5.1f us a grab (%u), hand-out %4.1f us\n",
                t.layer, t.name.c_str(), t.stream ? "stream " : "latency", t.begin_us, t.end_us, t.units, t.pf_units, t.drained_us,
                util * 100.0, t.grabs ? t.busy_us / t.grabs : 0.0, t.grabs, t.grabs ? t.handout_us / t.grabs : 0.0);
  }
}

}  // namespace

DGPP_TEST(step_stream_rate_at_the_design_chain) {
  auto rig = make_rig();
  std::printf("   %s", device_plan_summary(rig->dp).c_str());
  Streamer s(rig->dp, rig->bases, rig->plans);
  const StreamerConfig cfg = design_config();
  require(cfg.workers >= 16, "fewer than 16 worker blocks can be co-resident");
  const StreamReport r = s.run(cfg);
  print_report("design chain, lead 16 MB", r, 3);
  require(r.status == kStepOk, "the kernel ended on " + r.status_text);
  // Every stream phase of the last pass completed its units.
  for (const PhaseTiming& t : r.phases)
    if (t.stream) require(t.units > 0 || rig->dp.phases[static_cast<size_t>(t.phase)].fixed_units == 0, "phase " + t.name + " streamed no units");
  std::vector<double> pct;
  for (size_t l = 0; l < r.layer_us.size(); ++l) {
    const double floor_us = r.layer_bytes[l] / kCeilingGBs * 1e-3;
    pct.push_back(100.0 * floor_us / r.layer_us[l]);
    std::printf("   layer %zu: %.0f MB (%d experts) in %.0f us, floor %.0f us: %.0f %% of the ceiling\n", l, r.layer_bytes[l] / 1e6,
                r.run_counts[l], r.layer_us[l], floor_us, pct.back());
  }
  std::sort(pct.begin(), pct.end());
  const double median = pct[pct.size() / 2];
  std::printf("   median layer at %.1f %% of the ceiling with the design chain interleaved\n", median);
  require(median >= 95.0, "the median layer streams at " + std::to_string(median) + " percent of the ceiling, under 95");
  // Without the lead the same chain shows its holes: the layer is slower.
  StreamerConfig no_lead = cfg;
  no_lead.lead_cap_bytes = 0;
  const StreamReport r0 = s.run(no_lead);
  print_report("design chain, no lead", r0, 3);
  require(r0.status == kStepOk, "the no-lead kernel ended on " + r0.status_text);
  std::vector<double> pct0;
  for (size_t l = 0; l < r0.layer_us.size(); ++l) pct0.push_back(100.0 * (r0.layer_bytes[l] / kCeilingGBs * 1e-3) / r0.layer_us[l]);
  std::sort(pct0.begin(), pct0.end());
  std::printf("   without the lead: median layer at %.1f %% of the ceiling\n", pct0[pct0.size() / 2]);
  require(r.lead_bytes > 0 && pct0[pct0.size() / 2] <= median + 0.5, "the lead did not pay for itself");
}

DGPP_TEST(step_stream_governor_bounds_the_leaders) {
  auto rig = make_rig();
  Streamer s(rig->dp, rig->bases, rig->plans);
  StreamerConfig cfg = design_config();
  cfg.steps = 2;
  cfg.max_leaders = 4;
  const StreamReport r = s.run(cfg);
  print_report("governor: 4 leaders", r, 3);
  require(r.status == kStepOk, "the governed kernel ended on " + r.status_text);
  require(r.lead_bytes > 0, "the governed lead read nothing");
}

DGPP_TEST(step_stream_deadline_ends_the_kernel) {
  auto rig = make_rig();
  Streamer s(rig->dp, rig->bases, rig->plans);
  StreamerConfig cfg = design_config();
  cfg.steps = 1;
  cfg.deadline_s = 50e-6;  // a 190 MB phase takes 800 us at the ceiling
  const auto t0 = std::chrono::steady_clock::now();
  const StreamReport r = s.run(cfg);
  const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::printf("   deadline run: status %s after %.1f ms\n", r.status_text.c_str(), wall_ms);
  require((r.status & 0xff) == kStepDeadline, "a 50 us deadline did not end the kernel: " + r.status_text);
  require(wall_ms < 2000.0, "the deadline exit took too long");
}

DGPP_TEST(step_stream_poison_ends_the_kernel) {
  auto rig = make_rig();
  Streamer s(rig->dp, rig->bases, rig->plans);
  StreamerConfig cfg = design_config();
  cfg.steps = 1000;  // seconds of work the poison cuts short
  cfg.deadline_s = 0;
  std::thread killer([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    s.poison();
  });
  const auto t0 = std::chrono::steady_clock::now();
  const StreamReport r = s.run(cfg);
  const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  killer.join();
  std::printf("   poison run: status %s after %.1f ms\n", r.status_text.c_str(), wall_ms);
  require((r.status & 0xff) == kStepPoisoned, "the poison did not end the kernel: " + r.status_text);
  require(wall_ms < 3000.0, "the poison exit took too long");
}

// ---- the protocol (section 2.2, phase 2 item 5)

DGPP_TEST(step_protocol_soak_without_host_work) {
  auto rig = make_rig(2);
  Streamer s(rig->dp, rig->bases, rig->plans);
  StreamerConfig cfg = design_config();
  cfg.steps = 1000;
  cfg.tokens = {11, 22, 33, 44, 55};
  const StreamReport r = s.run(cfg);
  std::printf("   soak: status %s, %u steps in %.1f ms, %zu verdicts, folds %s\n", r.status_text.c_str(), r.steps_run, r.kernel_ms,
              r.verdicts.size(), r.folds_match ? "match" : "DIFFER");
  require(r.status == kStepOk && r.steps_run == 1000 && r.verdicts.size() == 1000, "a thousand steps ran: " + r.status_text);
  require(r.folds_match, "the device feed carried the tokens across every step");
  for (size_t k = 0; k < r.verdicts.size(); ++k)
    require(r.verdicts[k].status == kVerdictOk && r.verdicts[k].seq == k && r.verdicts[k].epoch == 0 && r.verdicts[k].rows == 5,
            "verdict " + std::to_string(k) + " is the step's");
  std::vector<double> pend(r.pending_us.begin() + 1, r.pending_us.end()), bind = r.bind_us, step = r.step_us;
  std::sort(pend.begin(), pend.end());
  std::sort(bind.begin(), bind.end());
  std::sort(step.begin(), step.end());
  std::printf("   between steps: the kernel saw the next execute %.1f us after the verdict (median), %.1f (p99), %.1f (max); bind %.1f us; step %.0f us\n",
              pend[pend.size() / 2], pend[pend.size() * 99 / 100], pend.back(), bind[bind.size() / 2], step[step.size() / 2]);
  require(pend[pend.size() / 2] < 100.0, "the host's turnaround between steps is over 100 us at the median");
}

DGPP_TEST(step_protocol_skip_and_host_work_beside_a_pending_kernel) {
  auto rig = make_rig(2);
  Streamer s(rig->dp, rig->bases, rig->plans);
  cudaStream_t side = nullptr;
  DGPP_CUDA_OK(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));
  uint8_t* scratch = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&scratch, size_t{16} << 20));
  uint8_t* pinned = nullptr;
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&pinned), size_t{16} << 20, cudaHostAllocDefault));
  std::memset(pinned, 0x5a, size_t{16} << 20);
  int worked = 0;
  StreamerConfig cfg = design_config();
  cfg.steps = 4;
  cfg.skip_at_step = 3;
  cfg.between_steps = [&](int step) {
    if (step == 0) return;
    // The kernel is resident and pending, holding every SM. The host's
    // between-step work that moves data through pinned memory, the token
    // and mask traffic of section 2.2, goes through the copy engines and
    // must run; work that needs SMs is what the skip state is for.
    DGPP_CUDA_OK(cudaMemcpyAsync(scratch, pinned, size_t{16} << 20, cudaMemcpyHostToDevice, side));
    DGPP_CUDA_OK(cudaMemcpyAsync(pinned, scratch, size_t{16} << 20, cudaMemcpyDeviceToHost, side));
    DGPP_CUDA_OK(cudaStreamSynchronize(side));
    ++worked;
  };
  const StreamReport r = s.run(cfg);
  std::printf("   skip run: status %s, %u steps, %zu verdicts, host work %d times\n", r.status_text.c_str(), r.steps_run, r.verdicts.size(), worked);
  require(worked == 3, "the host's work ran beside the pending kernel before steps 1, 2 and 3");
  require(r.steps_run == 3 && r.verdicts.size() == 4, "three steps ran and the fourth was answered");
  require(r.verdicts[3].status == kVerdictSkipped && r.verdicts[3].seq == 3, "the skipped step's verdict says so");
  require(r.status == kStepOk, "a skip is not an error: " + r.status_text);
  cudaFree(scratch);
  cudaFreeHost(pinned);
  cudaStreamDestroy(side);
}

DGPP_TEST(step_protocol_sequence_crosses_the_epoch_boundaries) {
  auto rig = make_rig(1);
  Streamer s(rig->dp, rig->bases, rig->plans);
  for (const uint64_t first : {(uint64_t{1} << 32) - 2, ~uint64_t{0} - 2}) {
    StreamerConfig cfg = design_config();
    cfg.steps = 4;
    cfg.first_seq = first;
    cfg.epoch = first ^ 0x5555555555555555ull;
    const StreamReport r = s.run(cfg);
    require(r.status == kStepOk && r.verdicts.size() == 4 && r.folds_match, "four steps across the boundary: " + r.status_text);
    for (int k = 0; k < 4; ++k)
      require(r.verdicts[static_cast<size_t>(k)].seq == first + static_cast<uint64_t>(k) &&
                  r.verdicts[static_cast<size_t>(k)].epoch == cfg.epoch,
              "verdict " + std::to_string(k) + " echoes the sequence and the epoch across the boundary");
    std::printf("   sequence %llu .. %llu: four verdicts in order, folds match\n", static_cast<unsigned long long>(first),
                static_cast<unsigned long long>(first + 3));
  }
}

DGPP_TEST(step_protocol_gather_phase_waits_for_the_host_sequence) {
  auto rig = make_rig(2);  // layer 1 is an Engram layer: its first phase gathers host rows
  Streamer s(rig->dp, rig->bases, rig->plans);
  StreamerConfig cfg = design_config();
  cfg.steps = 2;
  const StreamReport prompt = s.run(cfg);
  cfg.gather_delay_us = 3000;  // well past layer 0's 800 us: the phase must wait
  const StreamReport late = s.run(cfg);
  require(prompt.status == kStepOk && late.status == kStepOk, "both runs completed");
  auto engram = [&](const StreamReport& r) {
    for (const PhaseTiming& t : r.phases)
      if (t.layer == 1 && t.name == "engram") return t.end_us - t.begin_us;
    throw std::runtime_error("no engram phase on layer 1");
  };
  const double wait_prompt = engram(prompt), wait_late = engram(late);
  const double step_prompt = prompt.step_us.back(), step_late = late.step_us.back();
  std::printf("   engram phase: %.0f us with the rows published at once, %.0f us with them published 3000 us after the descriptor; "
              "the step %.0f and %.0f us\n", wait_prompt, wait_late, step_prompt, step_late);
  require(wait_prompt < 1000.0, "the prompt gather should not have waited");
  require(wait_late > 1500.0, "the gather phase did not wait for the host's sequence");
  require(step_late > step_prompt + 1500.0, "the late sequence did not lengthen the step");
}

// ---- the consumers inside the stream (phase 3 items 1 and 2)

void consumers_match_the_reference(bool tile_major) {
  auto rig = make_rig(2, /*consumers=*/true, tile_major);
  Streamer s(rig->dp, rig->bases, rig->plans);
  require(s.consumers(), "the plan's expert pools took the MXFP4 consumer");
  StreamerConfig cfg = design_config();
  cfg.steps = 1;
  cfg.rows = 5;
  cfg.top_k = 6;
  const StreamReport r = s.run(cfg);
  print_report("consumers, one step", r, 1);
  require(r.status == kStepOk && r.verdicts.size() == 1 && r.verdicts[0].status == kVerdictOk, "the step ran: " + r.status_text);
  const int H = s.hidden(), I = s.inter(), spt = cfg.top_k + 1, L = 1;
  const StepCtl* ctl = s.ctl();
  const int groups = ctl->run_count[L];
  require(groups > 0, "the last layer routed no experts");
  // The pools the layer's gate_up and down phases use.
  const DevPool* gate_up = nullptr;
  const DevPool* down = nullptr;
  const DevPool* sh_gate_up = nullptr;
  const DevPool* sh_down = nullptr;
  std::vector<const DevPool*> projections;
  const DevLayer& dl = rig->dp.layers[static_cast<size_t>(L)];
  for (int P = dl.phase_begin; P < dl.phase_begin + dl.phase_count; ++P) {
    const DevPhase& ph = rig->dp.phases[static_cast<size_t>(P)];
    for (int i = 0; i < ph.n_pools; ++i) {
      const DevPool& pool = rig->dp.pools[static_cast<size_t>(ph.pools[i])];
      if (pool.group_kind == kGroupRouted && pool_is_gate_up(pool.kind)) gate_up = &pool;
      if (pool.group_kind == kGroupRouted && pool_is_down(pool.kind)) down = &pool;
      if (pool.kind == kPoolFp8GateUp) sh_gate_up = &pool;
      if (pool.kind == kPoolFp8Down) sh_down = &pool;
      if (pool.kind == kPoolFp8Matrix) projections.push_back(&pool);
    }
  }
  require(gate_up && down, "the layer has a gate_up and a down consumer pool");
  require(sh_gate_up && sh_down, "the layer has the shared expert's fp8 pools");
  require(!projections.empty(), "the layer has fp8 projection pools");
  require(s.x_stride() == H, "the expert references read x at the hidden width, which this rig's buffer uses");
  const size_t slots = static_cast<size_t>(cfg.rows) * spt;
  const size_t act_n = slots * I, out_n = slots * H;
  std::vector<uint16_t> act(act_n);
  std::vector<float> out(out_n);
  std::vector<uint16_t> y(static_cast<size_t>(cfg.rows) * H);
  DGPP_CUDA_OK(cudaMemcpy(act.data(), s.act(), act_n * 2, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(out.data(), s.out(), out_n * 4, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(y.data(), s.y(), y.size() * 2, cudaMemcpyDeviceToHost));
  // The reference over the same bytes: gate and up from x, down from the consumer's act.
  uint16_t* act_ref = nullptr;
  float *gu = nullptr, *gu_mag = nullptr, *out_ref = nullptr, *out_mag = nullptr;
  uint16_t* act_dev = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&act_ref, act_n * 2));
  DGPP_CUDA_OK(cudaMalloc(&gu, act_n * 2 * 4));
  DGPP_CUDA_OK(cudaMalloc(&gu_mag, act_n * 2 * 4));
  DGPP_CUDA_OK(cudaMalloc(&out_ref, out_n * 4));
  DGPP_CUDA_OK(cudaMalloc(&out_mag, out_n * 4));
  DGPP_CUDA_OK(cudaMalloc(&act_dev, act_n * 2));
  DGPP_CUDA_OK(cudaMemcpy(act_dev, act.data(), act_n * 2, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemset(act_ref, 0, act_n * 2));
  DGPP_CUDA_OK(cudaMemset(out_ref, 0, out_n * 4));
  std::vector<bool> live(slots, false);
  for (int g = 0; g < groups; ++g) {
    const int expert = ctl->run_list[L][g];
    Mxfp4Group grp;
    grp.slots = ctl->group_slots[L][g];
    grp.tokens = ctl->group_tokens[L][g];
    grp.count = ctl->group_count[L][g];
    for (int i = 0; i < grp.count; ++i) live[static_cast<size_t>(grp.slots[i])] = true;
    Mxfp4Expert e;
    const uint8_t* gb = rig->bases[static_cast<size_t>(L)] + gate_up->base + static_cast<uint64_t>(expert) * gate_up->stride;
    e.gate = gb + gate_up->offsets[0];
    e.gate_scales = gb + gate_up->offsets[1];
    e.up = gb + gate_up->offsets[2];
    e.up_scales = gb + gate_up->offsets[3];
    const uint8_t* db = rig->bases[static_cast<size_t>(L)] + down->base + static_cast<uint64_t>(expert) * down->stride;
    e.down = db + down->offsets[0];
    e.down_scales = db + down->offsets[1];
    if (gate_up->kind == kPoolMxfp4GateUp) {
      step_ref::launch_gate_up(e, grp, s.x(), H, I, act_ref, gu, gu_mag, nullptr);
      step_ref::launch_down(e, grp, act_dev, H, I, spt, out_ref, out_mag, nullptr);
    } else {
      step_ref::launch_mxfp4_row_gate_up(e, grp, s.x(), H, I, act_ref, gu, gu_mag, nullptr);
      step_ref::launch_mxfp4_row_down(e, grp, act_dev, H, I, spt, out_ref, out_mag, nullptr);
    }
  }
  {
    // The shared expert: every token's shared slot through the fp8 pools.
    Mxfp4Group grp;
    grp.slots = ctl->shared_slots[L];
    grp.tokens = ctl->shared_tokens[L];
    grp.count = cfg.rows;
    for (int i = 0; i < grp.count; ++i) live[static_cast<size_t>(grp.slots[i])] = true;
    Fp8Expert f;
    const uint8_t* gb = rig->bases[static_cast<size_t>(L)] + sh_gate_up->base;
    f.gate = gb + sh_gate_up->offsets[0];
    f.gate_scales = reinterpret_cast<const float*>(gb + sh_gate_up->offsets[1]);
    f.up = gb + sh_gate_up->offsets[2];
    f.up_scales = reinterpret_cast<const float*>(gb + sh_gate_up->offsets[3]);
    const uint8_t* db = rig->bases[static_cast<size_t>(L)] + sh_down->base;
    f.down = db + sh_down->offsets[0];
    f.down_scales = reinterpret_cast<const float*>(db + sh_down->offsets[1]);
    step_ref::launch_fp8_gate_up(f, grp, s.x(), H, I, 32, act_ref, gu, gu_mag, nullptr);
    step_ref::launch_fp8_down(f, grp, act_dev, H, I, 32, spt, out_ref, out_mag, nullptr);
  }
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> act_r(act_n);
  std::vector<float> gu_h(act_n * 2), gu_m(act_n * 2), out_r(out_n), out_m(out_n);
  DGPP_CUDA_OK(cudaMemcpy(act_r.data(), act_ref, act_n * 2, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(gu_h.data(), gu, act_n * 8, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(gu_m.data(), gu_mag, act_n * 8, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(out_r.data(), out_ref, out_n * 4, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(out_m.data(), out_mag, out_n * 4, cudaMemcpyDeviceToHost));
  cudaFree(act_ref); cudaFree(gu); cudaFree(gu_mag); cudaFree(out_ref); cudaFree(out_mag); cudaFree(act_dev);
  size_t exact = 0, ok = 0, bad = 0, n = 0;
  for (size_t i = 0; i < act_n; ++i) {
    if (!live[i / I]) continue;
    ++n;
    if (act[i] == act_r[i]) ++exact;
    else if (step_ref::act_ok(act[i], act_r[i], gu_h[2 * i], gu_h[2 * i + 1], gu_m[2 * i], gu_m[2 * i + 1])) ++ok;
    else ++bad;
  }
  size_t out_bad = 0, out_n_live = 0;
  for (size_t i = 0; i < out_n; ++i) {
    const int slot = static_cast<int>((i / spt) / H) * spt + static_cast<int>(i % spt);
    if (!live[static_cast<size_t>(slot)]) continue;
    ++out_n_live;
    if (!step_ref::partial_ok(out[i], out_r[i], out_m[i])) ++out_bad;
  }
  // The accumulation, in the kernel's fma order, from the consumer's own partials.
  size_t y_bad = 0;
  for (int t = 0; t < cfg.rows; ++t)
    for (int c = 0; c < H; ++c) {
      float a = 0.f;
      for (int j = 0; j < cfg.top_k; ++j) a = std::fma(1.0f / cfg.top_k, out[step_out_index(t * spt + j, c, H, spt)], a);
      a = std::fma(1.0f, out[step_out_index(t * spt + cfg.top_k, c, H, spt)], a);
      if (float_to_bf16_bits(a) != y[static_cast<size_t>(t) * H + c]) ++y_bad;
    }
  std::printf("   %s experts, layer %d, %d routed and the shared over %d rows: act %zu exact, %zu within the envelope, %zu beyond of %zu; partials %zu beyond the bound of %zu; y %zu of %zu differ from the ordered sum\n",
              tile_major ? "tile-major" : "row-major", L, groups, cfg.rows, exact, ok, bad, n, out_bad, out_n_live, y_bad, y.size());
  require(bad == 0, "gate and up outputs beyond the reference's envelope inside the stream");
  require(out_bad == 0, "down partials beyond the fp32 bound inside the stream");
  require(y_bad == 0, "the accumulated output is not the ordered sum of the partials");
  // The projections: every token through each fp8 matrix of the layer.
  size_t proj_exact = 0, proj_ok = 0, proj_bad = 0, proj_n = 0;
  for (const DevPool* pool : projections) {
    const size_t elems = static_cast<size_t>(cfg.rows) * pool->rows;
    std::vector<uint16_t> got(elems);
    DGPP_CUDA_OK(cudaMemcpy(got.data(), s.proj() + pool->out_offset, elems * 2, cudaMemcpyDeviceToHost));
    uint16_t* y_ref = nullptr;
    float *yf = nullptr, *ymag = nullptr;
    DGPP_CUDA_OK(cudaMalloc(&y_ref, elems * 2));
    DGPP_CUDA_OK(cudaMalloc(&yf, elems * 4));
    DGPP_CUDA_OK(cudaMalloc(&ymag, elems * 4));
    Mxfp4Group grp;
    grp.slots = ctl->token_slots;
    grp.tokens = ctl->token_slots;
    grp.count = cfg.rows;
    const uint8_t* b = rig->bases[static_cast<size_t>(L)] + pool->base;
    step_ref::launch_fp8_matrix(b + pool->offsets[0], reinterpret_cast<const float*>(b + pool->offsets[1]), grp, s.x(), s.x_stride(), pool->k,
                                pool->rows, 32, y_ref, yf, ymag, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint16_t> ref(elems);
    std::vector<float> exact_f(elems), mag_f(elems);
    DGPP_CUDA_OK(cudaMemcpy(ref.data(), y_ref, elems * 2, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(exact_f.data(), yf, elems * 4, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(mag_f.data(), ymag, elems * 4, cudaMemcpyDeviceToHost));
    cudaFree(y_ref); cudaFree(yf); cudaFree(ymag);
    size_t pb = 0;
    for (size_t i = 0; i < elems; ++i) {
      ++proj_n;
      if (got[i] == ref[i]) ++proj_exact;
      else if (step_ref::value_ok(got[i], exact_f[i], mag_f[i])) ++proj_ok;
      else { ++proj_bad; ++pb; }
    }
    std::printf("      projection [%d, %d]: %zu outputs, %zu beyond\n", pool->rows, pool->k, elems, pb);
  }
  std::printf("   projections of layer %d: %zu exact, %zu within the envelope, %zu beyond of %zu\n", L, proj_exact, proj_ok, proj_bad, proj_n);
  require(proj_bad == 0, "projection outputs beyond the reference's envelope inside the stream");
}

DGPP_TEST(step_consumers_in_the_stream_match_the_reference) {
  consumers_match_the_reference(/*tile_major=*/false);  // the checkpoint's layout, as the loader writes it today
  consumers_match_the_reference(/*tile_major=*/true);   // the arena's tile-major experts
}

double consumers_rate(bool tile_major) {
  auto rig = make_rig(kLayers, /*consumers=*/true, tile_major);
  Streamer s(rig->dp, rig->bases, rig->plans);
  StreamerConfig cfg = design_config();
  cfg.steps = 3;
  cfg.rows = 5;
  cfg.top_k = 6;
  const StreamReport r = s.run(cfg);
  print_report(tile_major ? "tile-major experts, design chain, lead 16 MB" : "row-major experts, design chain, lead 16 MB", r, 3);
  require(r.status == kStepOk, "the kernel ended on " + r.status_text);
  std::vector<double> pct;
  for (size_t l = 0; l < r.layer_us.size(); ++l) {
    const double floor_us = r.layer_bytes[l] / kCeilingGBs * 1e-3;
    pct.push_back(100.0 * floor_us / r.layer_us[l]);
    std::printf("   layer %zu: %.0f MB (%d experts) in %.0f us, floor %.0f us: %.0f %% of the ceiling\n", l, r.layer_bytes[l] / 1e6,
                r.run_counts[l], r.layer_us[l], floor_us, pct.back());
  }
  std::sort(pct.begin(), pct.end());
  std::printf("   %s experts computed by the consumers: median layer at %.1f %% of the ceiling\n", tile_major ? "tile-major" : "row-major",
              pct[pct.size() / 2]);
  return pct[pct.size() / 2];
}

DGPP_TEST(step_consumers_rate_at_the_design_chain) {
  const double row = consumers_rate(/*tile_major=*/false);
  const double tile = consumers_rate(/*tile_major=*/true);
  // Interim gates at the measured plateau less the run-to-run noise (tile-major
  // 86 to 90 percent, row-major 78 to 88 with every weight computed; the
  // fused chain's structure is where the rest comes back, section 1.3).
  require(tile >= 82.0, "the tile-major consumers' layers stream under 82 percent of the ceiling");
  require(row >= 74.0, "the row-major consumers' layers stream under 74 percent of the ceiling");
}

DGPP_TEST(step_consumers_lead_sweep) {
  // The lead's cap against the consumers' own L2 use, with every consumer real (report only).
  auto rig = make_rig(kLayers, /*consumers=*/true, /*tile_major=*/true);
  Streamer s(rig->dp, rig->bases, rig->plans);
  for (const int mib : {4, 8, 12, 16, 20}) {
    StreamerConfig cfg = design_config();
    cfg.steps = 3;
    cfg.rows = 5;
    cfg.top_k = 6;
    cfg.lead_cap_bytes = static_cast<uint64_t>(mib) << 20;
    const StreamReport r = s.run(cfg);
    require(r.status == kStepOk, "the kernel ended on " + r.status_text);
    std::vector<double> pct;
    for (size_t l = 0; l < r.layer_us.size(); ++l) pct.push_back(100.0 * (r.layer_bytes[l] / kCeilingGBs * 1e-3) / r.layer_us[l]);
    std::sort(pct.begin(), pct.end());
    double qkv = 0, attn_out = 0, experts = 0;
    for (const PhaseTiming& t : r.phases)
      if (t.layer == 3) {
        if (t.name == "qkv") qkv = t.end_us - t.begin_us;
        if (t.name == "attn_out") attn_out = t.end_us - t.begin_us;
        if (t.name == "gate_up") experts = t.end_us - t.begin_us;
      }
    std::printf("   lead %2d MB: median layer %.1f %% of the ceiling; layer 3 qkv %.0f us, attn_out %.0f us, experts %.0f us; lead read %.0f MB\n", mib,
                pct[pct.size() / 2], qkv, attn_out, experts, r.lead_bytes / 1e6);
  }
}

int main() { return dgpp::test::run_all(); }
