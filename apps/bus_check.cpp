// bus_check: M5 deliverable 2 deployment driver — CollectiveBus over RoCE.
//
// Modes:
//   serve   [--port N] [--dev D]...        rank 0 rendezvous listener; runs
//          [--world N] [--duration-ms N]    the receiver, prints stats
//          [--lat-slots N] [--lat-bytes B] [--bulk-slots N] [--bulk-bytes B]
//   ping   --peer HOST [--port N] ...      rank 1 connector; sends messages
//          [--iters N] [--bytes B] [--class latency|bulk]
//          [--contend] [--lat-iters N]     bulk flood while latency pings
//   selftest                               loopback, in-process (ctest)
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/log.hpp"
#include "net/bus_kernel.hpp"
#include "net/collective_bus.hpp"

namespace {

using dgpp::net::BusMessageClass;
using dgpp::net::BusOptions;
using dgpp::net::CollectiveBus;

bool parse_long(const std::string& text, long minimum, long maximum,
                long* result) {
  if (text.empty()) return false;
  char* end = nullptr;
  errno = 0;
  const long value = std::strtol(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0' || value < minimum ||
      value > maximum)
    return false;
  *result = value;
  return true;
}

bool parse_size(const std::string& text, size_t minimum, size_t maximum,
                size_t* result) {
  if (text.empty() || text.front() == '-') return false;
  errno = 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0' ||
      value < minimum || value > maximum)
    return false;
  *result = static_cast<size_t>(value);
  return true;
}

double percentile(std::vector<double> sorted, double frac) {
  if (sorted.empty()) return 0.0;
  std::sort(sorted.begin(), sorted.end());
  const size_t idx = std::min(sorted.size() - 1,
                              static_cast<size_t>(frac * sorted.size()));
  return sorted[idx];
}

// Deterministic payload varying per message index: stale-buffer bugs (a
// slot reused without a fresh copy, a fold of last generation's bytes) fail
// the hash instead of passing silently.
void fill_payload(uint64_t* data, size_t words, uint32_t salt) {
  uint64_t x = 0x9E3779B97F4A7C15ULL ^ (0x100000001B3ULL * salt);
  for (size_t i = 0; i < words; ++i) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    data[i] = x;
  }
}

// Verifies a completed send: each stripe's consumer hash equals the fold of
// its contiguous chunk (latency: one stripe, whole message; bulk: chunk i
// is the i-th bulk_slot_bytes slice of the payload).
bool verify_result(const dgpp::net::BusSendResult& r, const void* data,
                   size_t bytes, size_t stripe_bytes, std::string* why) {
  if (!r.ok) {
    *why = "send failed: " + r.error;
    return false;
  }
  if (r.stripe_hashes.empty()) {
    *why = "no stripes credited";
    return false;
  }
  uint64_t got_xor = 0, want_xor = 0;
  for (size_t i = 0; i < r.stripe_hashes.size(); ++i) {
    const size_t off = i * stripe_bytes;
    const size_t chunk = std::min(stripe_bytes, bytes - off);
    const uint64_t want = dgpp::net::bus_fold(
        static_cast<const uint8_t*>(data) + off, chunk);
    got_xor ^= r.stripe_hashes[i];
    want_xor ^= want;
  }
  if (got_xor != want_xor) {
    *why = "payload hash mismatch (got " + std::to_string(got_xor) +
           " want " + std::to_string(want_xor) + ")";
    return false;
  }
  return true;
}

void print_bus_stats(const dgpp::net::BusStats& s) {
  for (const dgpp::net::BusLaneStats& l : s.lanes) {
    DGPP_LOG_INFO(
        "lane peer={} lane={} posts={} sent={}B recvs={} recv={}B "
        "credits_out={} credits_in={} failed={}",
        l.peer_rank, l.lane, l.posts, l.bytes_sent, l.doorbell_recvs,
        l.bytes_recv, l.credits_returned, l.credits_received,
        l.failed ? 1 : 0);
  }
}

struct CommonArgs {
  uint16_t port = 29600;
  std::vector<std::string> devs;
  int world = 2;
  int rank = 1;  // connector rank for ping; serve is always rank 0
  int lat_slots = 32;
  size_t lat_bytes = 8192;
  int bulk_slots = 16;
  size_t bulk_bytes = 262144;
  int timeout_ms = 5000;
};


BusOptions options_for(const CommonArgs& c, int my_rank,
                       const std::string& host) {
  BusOptions o;
  o.world_size = c.world;
  o.my_rank = my_rank;
  o.lane_devices =
      c.devs.empty()
          ? dgpp::net::configured_lane_devices()
          : c.devs;
  o.rendezvous_port = c.port;
  o.rendezvous_host = host;
  o.rendezvous_timeout_ms = 20000;
  o.lat_slots = c.lat_slots;
  o.lat_slot_bytes = c.lat_bytes;
  o.bulk_slots = c.bulk_slots;
  o.bulk_slot_bytes = c.bulk_bytes;
  o.completion_timeout_ms = c.timeout_ms;
  o.consumer_deadline_s = 60.0;
  return o;
}

// ---- serve ------------------------------------------------------------------

int run_serve(CommonArgs c, long duration_ms) {
  CollectiveBus bus(options_for(c, 0, ""));
  std::string error;
  if (!bus.start(&error)) {
    DGPP_LOG_ERROR("serve: {}", error);
    return 1;
  }
  DGPP_LOG_INFO("SERVE-READY port={} duration_ms={}", c.port, duration_ms);
  std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
  print_bus_stats(bus.stats());
  bus.stop();
  DGPP_LOG_INFO("serve: stopped cleanly");
  return 0;
}

// ---- ping -------------------------------------------------------------------

struct PingStats {
  int ok = 0;
  int fail = 0;
  std::vector<double> latency_us;
  double bytes = 0;
  double elapsed_s = 0;
};

PingStats ping_class(CollectiveBus& bus, int peer, BusMessageClass cls,
                     int iters, size_t bytes, int wait_ms) {
  PingStats st;
  std::vector<uint64_t> payload((bytes + 7) / 8);
  const size_t stripe_bytes = bus.slot_bytes(cls);
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    fill_payload(payload.data(), payload.size(), static_cast<uint32_t>(i));
    std::string error;
    const uint64_t id = bus.send(peer, payload.data(), bytes, cls, &error);
    if (id == 0) {
      DGPP_LOG_ERROR("send rejected: {}", error);
      ++st.fail;
      continue;
    }
    const dgpp::net::BusSendResult r = bus.wait(id, wait_ms);
    std::string why;
    if (!verify_result(r, payload.data(), bytes, stripe_bytes, &why)) {
      DGPP_LOG_ERROR("iter {}: {}", i, why);
      ++st.fail;
      continue;
    }
    ++st.ok;
    st.latency_us.push_back(r.elapsed_us);
    st.bytes += static_cast<double>(bytes);
  }
  st.elapsed_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  return st;
}

// Windowed bulk flood: keeps `window` requests in flight and defers hash
// verification out of the timed section. This measures the transport's
// steady state — engine pipelining, ring recycling, consumer rate — without
// the per-message host costs (payload fill, fold verification) that the
// sequential ping path pays inside its clock.
PingStats bulk_flood(CollectiveBus& bus, int peer, int iters, size_t bytes,
                     int window, int wait_ms) {
  PingStats st;
  std::vector<std::vector<uint64_t>> payloads(
      static_cast<size_t>(window));
  for (int i = 0; i < window; ++i) {
    payloads[static_cast<size_t>(i)].resize(bytes / 8);
    fill_payload(payloads[static_cast<size_t>(i)].data(), bytes / 8,
                 static_cast<uint32_t>(i));
  }
  std::vector<dgpp::net::BusSendResult> results;
  results.reserve(static_cast<size_t>(iters));
  std::vector<uint64_t> ids;
  ids.reserve(static_cast<size_t>(iters));

  int submitted = 0;
  int reaped = 0;
  const auto t0 = std::chrono::steady_clock::now();
  while (reaped < iters) {
    while (submitted < iters && submitted - reaped < window) {
      std::string error;
      const uint64_t id =
          bus.send(peer, payloads[static_cast<size_t>(submitted % window)]
                       .data(),
                   bytes, BusMessageClass::kBulk, &error);
      if (id == 0) {
        DGPP_LOG_ERROR("flood send rejected: {}", error);
        ++st.fail;
        results.push_back(dgpp::net::BusSendResult{});
        ids.push_back(0);
      } else {
        ids.push_back(id);
      }
      ++submitted;
    }
    if (ids[static_cast<size_t>(reaped)] == 0) {
      ++reaped;
      continue;
    }
    dgpp::net::BusSendResult r =
        bus.wait(ids[static_cast<size_t>(reaped)], wait_ms);
    if (r.ok) {
      ++st.ok;
      st.latency_us.push_back(r.elapsed_us);
      st.bytes += static_cast<double>(bytes);
    } else {
      DGPP_LOG_ERROR("flood request {} failed: {}", reaped, r.error);
      ++st.fail;
    }
    results.push_back(std::move(r));
    ++reaped;
  }
  st.elapsed_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();

  // Verification runs after the clock: every message is still hash-checked,
  // but the host cost no longer pollutes the throughput measurement.
  const size_t stripe_bytes = bus.slot_bytes(BusMessageClass::kBulk);
  for (size_t i = 0; i < results.size(); ++i) {
    if (!results[i].ok) continue;
    std::string why;
    if (!verify_result(results[i], payloads[i % payloads.size()].data(),
                       bytes, stripe_bytes, &why)) {
      DGPP_LOG_ERROR("flood verification failed at request {}: {}", i, why);
      --st.ok;
      ++st.fail;
    }
  }
  return st;
}

// Mixed-class soak: bulk flood and latency probes run CONCURRENTLY until
// the deadline (`ping --contend --soak-ms N`). The fixed --iters/--lat-iters
// caps size smoke runs — a one-hour exit-gate soak needs a duration-bounded
// workload driver instead. Latency samples are stored every
// kSoakSampleStride-th probe (uniform subsample: percentiles stay
// representative; counts and running max are exact).
struct SoakStats {
  std::atomic<uint64_t> ok{0};
  std::atomic<uint64_t> fail{0};
  std::atomic<uint64_t> bytes{0};
  double max_us = 0.0;              // guarded by samples_mu
  std::vector<double> samples;      // every kSoakSampleStride-th latency
  std::mutex samples_mu;
};

constexpr size_t kSoakSampleStride = 16;

// One worker thread's loop: batches of ping_class until stop. Batch sizes
// bound stop latency (~1 s) while amortizing per-batch setup.
void soak_thread(CollectiveBus& bus, BusMessageClass cls, size_t bytes,
                 long batch, int wait_ms, std::atomic<bool>& stop,
                 SoakStats& st) {
  while (!stop.load()) {
    const PingStats s =
        ping_class(bus, 0, cls, static_cast<int>(batch), bytes, wait_ms);
    st.ok.fetch_add(static_cast<uint64_t>(s.ok));
    st.fail.fetch_add(static_cast<uint64_t>(s.fail));
    st.bytes.fetch_add(static_cast<uint64_t>(s.bytes));
    std::lock_guard<std::mutex> lock(st.samples_mu);
    if (!s.latency_us.empty())
      st.max_us = std::max(st.max_us, *std::max_element(
                                          s.latency_us.begin(),
                                          s.latency_us.end()));
    for (size_t i = 0; i < s.latency_us.size(); i += kSoakSampleStride)
      st.samples.push_back(s.latency_us[i]);
  }
}

void soak_report(const char* what, SoakStats& st, double elapsed_s) {
  std::vector<double> sorted;
  double max_us;
  {
    std::lock_guard<std::mutex> lock(st.samples_mu);
    sorted = st.samples;
    max_us = st.max_us;
  }
  const double gbps =
      elapsed_s > 0 ? static_cast<double>(st.bytes.load()) * 8.0 /
                         elapsed_s / 1e9
                   : 0.0;
  DGPP_LOG_INFO(
      "SOAK {}: ok={} fail={} max_us={:.1f} p50_us={:.1f} p99_us={:.1f} "
      "samples={} gbps={:.2f}",
      what, st.ok.load(), st.fail.load(), max_us,
      percentile(sorted, 0.5), percentile(sorted, 0.99), sorted.size(),
      gbps);
}

int run_soak(CommonArgs c, const std::string& peer_host, size_t bulk_bytes,
             long soak_ms) {
  CollectiveBus bus(options_for(c, c.rank, peer_host));
  std::string error;
  if (!bus.start(&error)) {
    DGPP_LOG_ERROR("soak: {}", error);
    return 1;
  }
  std::atomic<bool> stop{false};
  SoakStats bulk, lat;
  // Batches sized for ~1 s stop latency at the measured fabric rates
  // (bulk ~50-400 us/iter by size, lat ~15 us/probe).
  const long bulk_batch = 2000, lat_batch = 20000;
  const int wait_ms = c.timeout_ms + 2000;
  std::thread bulk_thread(soak_thread, std::ref(bus),
                          BusMessageClass::kBulk, bulk_bytes, bulk_batch,
                          wait_ms, std::ref(stop), std::ref(bulk));
  std::thread lat_thread(soak_thread, std::ref(bus),
                         BusMessageClass::kLatency, c.lat_bytes, lat_batch,
                         wait_ms, std::ref(stop), std::ref(lat));
  const auto t0 = std::chrono::steady_clock::now();
  const auto deadline = t0 + std::chrono::milliseconds(soak_ms);
  uint64_t next_report_s = 60;
  while (std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::seconds(5));
    const double elapsed_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    if (elapsed_s >= static_cast<double>(next_report_s)) {
      DGPP_LOG_INFO("SOAK progress: {:.0f}s elapsed", elapsed_s);
      next_report_s += 60;
    }
  }
  stop.store(true);
  bulk_thread.join();
  lat_thread.join();
  const double elapsed_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  soak_report("bulk", bulk, elapsed_s);
  soak_report("lat", lat, elapsed_s);
  print_bus_stats(bus.stats());
  bus.stop();
  const uint64_t fails = bulk.fail.load() + lat.fail.load();
  DGPP_LOG_INFO("soak: stopped cleanly (failures={})", fails);
  return fails == 0 ? 0 : 1;
}

int run_ping(CommonArgs c, std::string peer_host, int iters, size_t bytes,
             const std::string& cls_text, bool contend, int lat_iters,
             long window, long hold_ms) {
  CollectiveBus bus(options_for(c, c.rank, peer_host));
  std::string error;
  if (!bus.start(&error)) {
    DGPP_LOG_ERROR("ping: {}", error);
    return 1;
  }

  int failures = 0;

  // Mesh smoke (world > 2): every rank exchanges with every peer, small
  // latency class plus one bulk message per pair — validates all pair
  // QPs, the endpoint-table distribution, and per-peer doorbell counts.
  // Ranks run at different speeds, so hold the bus up for slower peers:
  // a rank that tears down early makes its in-flight partners' last pair
  // watchdog against a dead peer (by design, but noise for a smoke).
  if (c.world > 2) {
    for (int peer = 0; peer < c.world; ++peer) {
      if (peer == c.rank) continue;
      const PingStats lat =
          ping_class(bus, peer, BusMessageClass::kLatency, iters,
                     c.lat_bytes, c.timeout_ms + 2000);
      const PingStats bulk =
          ping_class(bus, peer, BusMessageClass::kBulk, 2, 4u << 20,
                     c.timeout_ms + 2000);
      DGPP_LOG_INFO(
          "MESH rank {} -> rank {}: lat ok={}/{} p50_us={:.1f} | bulk ok={}/{}",
          c.rank, peer, lat.ok, iters,
          percentile(lat.latency_us, 0.5), bulk.ok, 2);
      failures += lat.fail + bulk.fail;
    }
    print_bus_stats(bus.stats());
    DGPP_LOG_INFO("MESH rank {} holding {} ms for slower peers", c.rank,
                  hold_ms);
    std::this_thread::sleep_for(std::chrono::milliseconds(hold_ms));
    bus.stop();
    DGPP_LOG_INFO("mesh smoke: stopped cleanly (failures={})", failures);
    return failures == 0 ? 0 : 1;
  }

  if (contend) {
    // Bulk flood on one thread, latency probes on another: the §6.1
    // concurrency contract — latency must hold budget under bulk load.
    std::atomic<int> bulk_fail{0};
    std::thread bulk_thread([&] {
      const PingStats bulk =
          ping_class(bus, 0, BusMessageClass::kBulk, iters, bytes,
                     c.timeout_ms + 2000);
      bulk_fail.store(bulk.fail);
    });
    const PingStats lat =
        ping_class(bus, 0, BusMessageClass::kLatency, lat_iters, c.lat_bytes,
                   c.timeout_ms + 2000);
    bulk_thread.join();
    DGPP_LOG_INFO(
        "CONTEND lat ok={}/{} p50_us={:.1f} p99_us={:.1f} max_us={:.1f}",
        lat.ok, lat_iters, percentile(lat.latency_us, 0.5),
        percentile(lat.latency_us, 0.99),
        lat.latency_us.empty()
            ? 0.0
            : *std::max_element(lat.latency_us.begin(),
                                lat.latency_us.end()));
    failures = lat.fail + bulk_fail.load();
  } else {
    const BusMessageClass cls =
        cls_text == "bulk" ? BusMessageClass::kBulk : BusMessageClass::kLatency;
    const PingStats st =
        (cls == BusMessageClass::kBulk && window > 1)
            ? bulk_flood(bus, 0, iters, bytes, window, c.timeout_ms + 5000)
            : ping_class(bus, 0, cls, iters, bytes, c.timeout_ms + 2000);
    if (st.fail != 0) failures += st.fail;
    const double gbps =
        st.elapsed_s > 0 ? st.bytes * 8.0 / st.elapsed_s / 1e9 : 0.0;
    DGPP_LOG_INFO(
        "PING class={} iters={} bytes={} ok={} fail={} min_us={:.1f} "
        "p50_us={:.1f} p99_us={:.1f} max_us={:.1f} gbps={:.2f}",
        cls_text, iters, bytes, st.ok, st.fail,
        st.latency_us.empty()
            ? 0.0
            : *std::min_element(st.latency_us.begin(), st.latency_us.end()),
        percentile(st.latency_us, 0.5), percentile(st.latency_us, 0.99),
        st.latency_us.empty()
            ? 0.0
            : *std::max_element(st.latency_us.begin(), st.latency_us.end()),
        gbps);
  }

  print_bus_stats(bus.stats());
  bus.stop();
  DGPP_LOG_INFO("ping: stopped cleanly (failures={})", failures);
  return failures == 0 ? 0 : 1;
}

// ---- allreduce --------------------------------------------------------------

// Deterministic per-rank bf16 pattern (the same LCG shape as the loopback
// test; every rank can regenerate every rank's vector, so each node
// verifies its own destination bitwise against the canonical chain).
void fill_rank_bf16(std::vector<uint16_t>* out, size_t elems, int rank) {
  out->assign(elems, 0);
  uint64_t x = 0x9E3779B97F4A7C15ULL ^ (0x100000001B3ULL * (rank + 1));
  for (size_t i = 0; i < elems; ++i) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    const float v =
        static_cast<float>(static_cast<int32_t>(x >> 40)) * (1.0f / 8388608.0f) - 1.0f;
    (*out)[i] = dgpp::net::bf16_from_f32_rne(v);
  }
}

int run_allreduce(CommonArgs c, const std::string& peer, int iters,
                  long hold_ms, int graph_gens) {
  const int my_rank = c.rank;
  if (my_rank != 0 && peer.empty()) {
    DGPP_LOG_ERROR("allreduce: --peer is required for ranks 1..N-1");
    return 2;
  }
  if (my_rank >= c.world) {
    DGPP_LOG_ERROR("allreduce: --rank must be below --world");
    return 2;
  }
  BusOptions o = options_for(c, my_rank, my_rank == 0 ? "" : peer);
  o.launch_consumers = false;  // the per-collective kernel owns the claims

  CollectiveBus bus(o);
  std::string error;
  if (!bus.start(&error)) {
    DGPP_LOG_ERROR("allreduce: {}", error);
    return 1;
  }
  DGPP_LOG_INFO("ALLREDUCE-READY rank={} world={}", my_rank, c.world);

  const size_t elems = o.lat_slot_bytes / 2;
  std::vector<uint16_t> host_src, host_dst(elems, 0), got(elems, 0);
  fill_rank_bf16(&host_src, elems, my_rank);
  std::vector<std::vector<uint16_t>> all_src(static_cast<size_t>(c.world));
  for (int r = 0; r < c.world; ++r)
    fill_rank_bf16(&all_src[static_cast<size_t>(r)], elems, r);

  uint16_t* dev_src = nullptr;
  uint16_t* dev_dst = nullptr;
  float* warm_buf = nullptr;
  cudaStream_t warm_stream = nullptr;
  if (cudaMalloc(&dev_src, elems * 2) != cudaSuccess ||
      cudaMalloc(&dev_dst, elems * 2) != cudaSuccess ||
      cudaMalloc(&warm_buf, 256 * 4) != cudaSuccess ||
      cudaStreamCreateWithFlags(&warm_stream, cudaStreamNonBlocking) !=
          cudaSuccess) {
    DGPP_LOG_ERROR("allreduce: device alloc failed");
    bus.stop();
    return 1;
  }
  cudaMemcpy(dev_src, host_src.data(), elems * 2, cudaMemcpyHostToDevice);
  cudaMemset(warm_buf, 0, 256 * 4);

  // Isolation probe: same-thread back-to-back kernel launches, timed.
  for (int i = 0; i < 12; ++i) {
    const auto w0 = std::chrono::steady_clock::now();
    dgpp::net::launch_bus_warm_work(warm_stream, warm_buf, 2000);
    cudaStreamSynchronize(warm_stream);
    DGPP_LOG_INFO("probe: app-thread launch+sync #{} = {:.1f}us", i,
                  std::chrono::duration<double, std::micro>(
                      std::chrono::steady_clock::now() - w0)
                      .count());
  }
  // Graph probe (the §6.2 decode answer): capture one launch, replay it.
  // If replay is uniform and cheap, per-collective kernels inside graphs
  // dodge the isolated-launch tail entirely.
  {
    cudaStreamBeginCapture(warm_stream, cudaStreamCaptureModeThreadLocal);
    dgpp::net::launch_bus_warm_work(warm_stream, warm_buf, 2000);
    cudaGraph_t graph;
    cudaStreamEndCapture(warm_stream, &graph);
    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    for (int i = 0; i < 3; ++i) {  // warm
      cudaGraphLaunch(exec, warm_stream);
      cudaStreamSynchronize(warm_stream);
    }
    for (int i = 0; i < 12; ++i) {
      const auto g0 = std::chrono::steady_clock::now();
      cudaGraphLaunch(exec, warm_stream);
      cudaStreamSynchronize(warm_stream);
      DGPP_LOG_INFO("probe: graph replay+sync #{} = {:.1f}us", i,
                    std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - g0)
                        .count());
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
  }

  // One collective is a full round trip; verification needs a D2H copy, so
  // each iteration is: submit, wait, verify. Warmup absorbs the first-
  // launch costs (module load, doorbell paths) like the mesh smoke does.
  const int warmup = 4;
  int verified = 0;
  int failed = 0;
  std::vector<double> latency_us;
  const char* spins_env = std::getenv("DGPP_WARM_SPINS");
  const int warm_spins = spins_env ? std::atoi(spins_env) : 200000;
  auto prev_end = std::chrono::steady_clock::now();
  for (int iter = 0; iter < warmup + iters; ++iter) {
    // Spins >= 1,000,000: one long keep-alive at iter 0 (device never
    // sleeps for the whole run — isolates GPU wake latency).
    if (warm_spins >= 1000000) {
      if (iter == 0)
        dgpp::net::launch_bus_warm_work(warm_stream, warm_buf, warm_spins);
    } else if (warm_spins > 0) {
      dgpp::net::launch_bus_warm_work(warm_stream, warm_buf, warm_spins);
    }
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t id = bus.allreduce(dev_src, dev_dst, elems, &error);
    if (id == 0) {
      DGPP_LOG_ERROR("allreduce rejected: {}", error);
      ++failed;
      break;
    }
    const dgpp::net::BusAllReduceResult r = bus.wait_allreduce(id, 30000);
    const double us = std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    prev_end = std::chrono::steady_clock::now();
    if (iter >= warmup && iter < warmup + 6)
      DGPP_LOG_INFO("iter {}: submit->wait={:.1f}us", iter, us);
    if (!r.ok) {
      DGPP_LOG_ERROR("allreduce iter {} failed: {}", iter, r.error);
      ++failed;
      break;
    }
    if (iter < warmup) continue;
    latency_us.push_back(us);
    const bool do_verify = iter == warmup + iters - 1;
    if (!do_verify) continue;
    if (cudaMemcpy(got.data(), dev_dst, elems * 2, cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
      DGPP_LOG_ERROR("allreduce iter {}: D2H failed", iter);
      ++failed;
      break;
    }
    size_t mismatches = 0;
    for (size_t i = 0; i < elems; ++i) {
      float acc = 0.0f;
      for (int rr = 0; rr < c.world; ++rr)
        acc += dgpp::net::bf16_to_f32(all_src[static_cast<size_t>(rr)][i]);
      if (got[i] != dgpp::net::bf16_from_f32_rne(acc)) {
        if (mismatches < 3)
          DGPP_LOG_ERROR("iter {} elem {}: got {:04x} want {:04x}", iter, i,
                         got[i], dgpp::net::bf16_from_f32_rne(acc));
        ++mismatches;
      }
    }
    if (mismatches != 0) {
      DGPP_LOG_ERROR("allreduce iter {}: {} mismatches", iter, mismatches);
      ++failed;
    } else {
      ++verified;
    }
  }

  std::sort(latency_us.begin(), latency_us.end());
  const auto pick = [&](double frac) {
    return latency_us.empty()
               ? 0.0
               : latency_us[std::min(latency_us.size() - 1,
                                     static_cast<size_t>(
                                         frac * latency_us.size()))];
  };
  DGPP_LOG_INFO(
      "ALLREDUCE rank={} world={} verified={}/{} min={:.1f}us p50={:.1f}us "
      "p99={:.1f}us",
      my_rank, c.world, verified, iters, pick(0.0), pick(0.5), pick(0.99));

  // ---- graph amortization probe (§6.2) -----------------------------------
  // The same collective, recorded into a decode-shaped step (GEMM
  // stand-ins between the nodes, the same warm-spin cadence the eager
  // loop pays per collective — the A/B is per-collective amortized) and
  // replayed. The eager p50 above is the baseline; the numbers here are
  // the amortization verdict the fabric measurement exists for.
  if (graph_gens > 0 && failed == 0) {
    if (!bus.graph_record_begin(&error)) {
      DGPP_LOG_ERROR("graph record begin rejected: {}", error);
      ++failed;
    } else {
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t exec = nullptr;
      // Keep-alive spin (>= 1M means the eager loop used one long spin
      // for the whole run; the probe wants the per-collective cadence).
      const int spins_per_node = warm_spins >= 1000000 ? 200000 : warm_spins;
      if (cudaStreamBeginCapture(warm_stream,
                                 cudaStreamCaptureModeThreadLocal) !=
          cudaSuccess) {
        DGPP_LOG_ERROR("graph probe: capture begin failed");
        ++failed;
      } else {
        for (int g = 0; g < graph_gens; ++g) {
          dgpp::net::launch_bus_warm_work(warm_stream, warm_buf,
                                          spins_per_node);
          if (!bus.allreduce_record(warm_stream, dev_src, dev_dst, elems,
                                    &error)) {
            DGPP_LOG_ERROR("allreduce_record rejected: {}", error);
            ++failed;
            break;
          }
        }
        dgpp::net::launch_bus_warm_work(warm_stream, warm_buf,
                                        spins_per_node);
        if (cudaStreamEndCapture(warm_stream, &graph) != cudaSuccess ||
            !graph ||
            cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) !=
                cudaSuccess) {
          DGPP_LOG_ERROR("graph probe: capture/instantiate failed");
          ++failed;
        }
      }
      if (failed == 0) {
        if (!bus.graph_record_end(&error)) {
          DGPP_LOG_ERROR("graph record end rejected: {}", error);
          ++failed;
        }
      } else {
        std::string close_error;
        bus.graph_record_end(&close_error);
      }
      if (exec != nullptr) {
        // Warm replays absorb the instantiation/module costs, then the
        // measured ones: arm -> launch -> sync -> finish is the step the
        // decode path pays per token.
        for (int warm = 0; warm < 2 && failed == 0; ++warm) {
          if (!bus.graph_replay_arm(&error)) {
            DGPP_LOG_ERROR("graph arm rejected: {}", error);
            ++failed;
            break;
          }
          cudaGraphLaunch(exec, warm_stream);
          cudaStreamSynchronize(warm_stream);
          if (!bus.graph_replay_finish(30000, &error)) {
            DGPP_LOG_ERROR("graph finish rejected: {}", error);
            ++failed;
          }
        }
        std::vector<double> step_us;
        int replay_fail = 0;
        for (int replay = 0; replay < iters && failed == 0; ++replay) {
          const auto t0 = std::chrono::steady_clock::now();
          if (!bus.graph_replay_arm(&error)) {
            DGPP_LOG_ERROR("graph arm rejected: {}", error);
            replay_fail = 1;
            break;
          }
          if (cudaGraphLaunch(exec, warm_stream) != cudaSuccess ||
              cudaStreamSynchronize(warm_stream) != cudaSuccess) {
            DGPP_LOG_ERROR("graph replay {} failed to run", replay);
            replay_fail = 1;
            break;
          }
          if (!bus.graph_replay_finish(30000, &error)) {
            DGPP_LOG_ERROR("graph finish rejected: {}", error);
            replay_fail = 1;
            break;
          }
          step_us.push_back(std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - t0)
                                .count());
        }
        failed += replay_fail;
        if (!step_us.empty()) {
          std::sort(step_us.begin(), step_us.end());
          const double step_p50 =
              step_us[std::min(step_us.size() - 1, step_us.size() / 2)];
          const double step_min = step_us.front();
          DGPP_LOG_INFO(
              "GRAPH-PROBE rank={} gens={} replays={} step min={:.1f}us "
              "p50={:.1f}us | per-collective min={:.1f}us p50={:.1f}us "
              "(eager p50={:.1f}us)",
              my_rank, graph_gens, step_us.size(), step_min, step_p50,
              step_min / graph_gens, step_p50 / graph_gens, pick(0.5));
        }
        // Bitwise: the last replay must equal the canonical chain.
        if (cudaMemcpy(got.data(), dev_dst, elems * 2,
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
          DGPP_LOG_ERROR("graph probe: D2H failed");
          ++failed;
        } else {
          size_t mismatches = 0;
          for (size_t i = 0; i < elems; ++i) {
            float acc = 0.0f;
            for (int rr = 0; rr < c.world; ++rr)
              acc += dgpp::net::bf16_to_f32(
                  all_src[static_cast<size_t>(rr)][i]);
            if (got[i] != dgpp::net::bf16_from_f32_rne(acc)) {
              if (mismatches < 3)
                DGPP_LOG_ERROR("graph probe elem {}: got {:04x} want {:04x}",
                               i, got[i], dgpp::net::bf16_from_f32_rne(acc));
              ++mismatches;
            }
          }
          if (mismatches != 0) {
            DGPP_LOG_ERROR("graph probe: {} mismatches", mismatches);
            ++failed;
          }
        }
        cudaGraphExecDestroy(exec);
      }
      if (graph) cudaGraphDestroy(graph);
    }
  }

  cudaFree(dev_src);
  cudaFree(dev_dst);
  cudaFree(warm_buf);
  cudaStreamDestroy(warm_stream);

  print_bus_stats(bus.stats());
  if (hold_ms > 0)
    DGPP_LOG_INFO("allreduce: rank {} holding {} ms for slower peers",
                  my_rank, hold_ms);
  std::this_thread::sleep_for(std::chrono::milliseconds(hold_ms));
  bus.stop();
  DGPP_LOG_INFO("allreduce: stopped cleanly (failures={})", failed);
  // `verified` counts the last-iteration check (the loop verifies once, by
  // design — the eager p50 would otherwise pay a D2H per collective).
  return failed == 0 && verified == 1 ? 0 : 1;
}

// ---- selftest ----------------------------------------------------------------

// Runs rank-0 (listener) and rank-1 (connector) buses in-process against a
// loopback rendezvous on the fabric (validated: an RC pair through the
// switch to ourselves).
int run_selftest() {
  int failures = 0;
  CommonArgs c;
  c.port = 29710;

  auto start_pair = [&](BusOptions a_opt, BusOptions b_opt,
                        std::unique_ptr<CollectiveBus>* a_out,
                        std::unique_ptr<CollectiveBus>* b_out,
                        std::string* error) {
    *a_out = std::make_unique<CollectiveBus>(a_opt);
    *b_out = std::make_unique<CollectiveBus>(b_opt);
    std::string a_error;
    std::thread a_thread([&] {
      if (!(*a_out)->start(&a_error)) DGPP_LOG_ERROR("selftest A: {}", a_error);
    });
    const bool b_ok = (*b_out)->start(error);
    a_thread.join();
    if (!b_ok) return false;
    if (!a_error.empty()) {
      *error = a_error;
      return false;
    }
    return true;
  };

  {
    // Scenario 1: latency ping integrity.
    std::unique_ptr<CollectiveBus> a, b;
    std::string error;
    if (!start_pair(options_for(c, 0, ""), options_for(c, 1, "127.0.0.1"),
                    &a, &b, &error)) {
      DGPP_LOG_ERROR("selftest startup: {}", error);
      return 1;
    }
    const PingStats st =
        ping_class(*b, 0, BusMessageClass::kLatency, 32, c.lat_bytes,
                   c.timeout_ms + 2000);
    if (st.ok != 32) {
      DGPP_LOG_ERROR("selftest latency: ok={} fail={}", st.ok, st.fail);
      ++failures;
    } else {
      DGPP_LOG_INFO("selftest latency ok (p99_us={:.1f})",
                    percentile(st.latency_us, 0.99));
    }
    a->quiesce();
    b->quiesce();
    a->stop();
    b->stop();
  }

  {
    // Scenario 2: bulk striping across both lanes.
    std::unique_ptr<CollectiveBus> a, b;
    std::string error;
    if (!start_pair(options_for(c, 0, ""), options_for(c, 1, "127.0.0.1"),
                    &a, &b, &error)) {
      DGPP_LOG_ERROR("selftest startup: {}", error);
      return 1;
    }
    const size_t bytes = 4u << 20;
    const PingStats st = ping_class(*b, 0, BusMessageClass::kBulk, 4, bytes,
                                    c.timeout_ms + 2000);
    const dgpp::net::BusStats sender = b->stats();
    int lanes_used = 0;
    for (const auto& l : sender.lanes)
      if (l.credits_received > 0) ++lanes_used;
    if (st.ok != 4 || lanes_used < 2) {
      DGPP_LOG_ERROR("selftest bulk: ok={} fail={} lanes_used={}", st.ok,
                     st.fail, lanes_used);
      ++failures;
    } else {
      DGPP_LOG_INFO("selftest bulk ok ({:.0f} MB, {} lanes, gbps={:.2f})",
                    4.0 * 4, lanes_used,
                    st.elapsed_s > 0 ? st.bytes * 8.0 / st.elapsed_s / 1e9 : 0.0);
    }
    a->quiesce();
    b->quiesce();
    a->stop();
    b->stop();
  }

  {
    // Scenario 3: latency under concurrent bulk.
    std::unique_ptr<CollectiveBus> a, b;
    std::string error;
    if (!start_pair(options_for(c, 0, ""), options_for(c, 1, "127.0.0.1"),
                    &a, &b, &error)) {
      DGPP_LOG_ERROR("selftest startup: {}", error);
      return 1;
    }
    std::thread bulk_thread([&] {
      ping_class(*b, 0, BusMessageClass::kBulk, 8, 2u << 20,
                 c.timeout_ms + 2000);
    });
    const PingStats lat = ping_class(*b, 0, BusMessageClass::kLatency, 64,
                                     c.lat_bytes, c.timeout_ms + 2000);
    bulk_thread.join();
    if (lat.ok != 64) {
      DGPP_LOG_ERROR("selftest contention: ok={} fail={}", lat.ok, lat.fail);
      ++failures;
    } else {
      DGPP_LOG_INFO("selftest contention ok (lat p99_us={:.1f})",
                    percentile(lat.latency_us, 0.99));
    }
    a->quiesce();
    b->quiesce();
    a->stop();
    b->stop();
  }

  DGPP_LOG_INFO("bus selftest: {} failures", failures);
  return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage:\n"
                 "  bus_check serve [--port N] [--dev D]... [--world N] "
                 "[--duration-ms N]\n"
                 "                  [--lat-slots N] [--lat-bytes B] "
                 "[--bulk-slots N] [--bulk-bytes B]\n"
                 "  bus_check ping --peer HOST [--port N] [--dev D]... "
                 "[--iters N] [--bytes B]\n"
                 "                 [--class latency|bulk] [--contend] "
                 "[--window N] [--lat-iters N] [--timeout-ms N]\n"
                 "                 [--soak-ms N] (with --contend: bulk flood "
                 "+ latency probes until the deadline)\n"
                 "                 (--window > 1 with --class bulk: pipelined "
                 "flood, verification deferred)\n"
                 "  bus_check allreduce --peer HOST [--rank R] [--world N] "
                 "[--iters N] [--hold-ms N] [--graph-gens N]\n"
                 "                        (rank 0 listens; every rank "
                 "reduces and verifies bitwise)\n"
                 "  bus_check selftest\n");
    return 2;
  }
  const std::string mode = argv[1];
  if (mode == "selftest" && argc == 2) return run_selftest();

  CommonArgs c;
  std::string peer;
  long iters = 64;
  long duration_ms = 30000;
  long soak_ms = 0;
  long lat_iters = 1000;
  long window = 1;
  long hold_ms = 20000;
  long graph_gens = 0;
  size_t bytes = 8192;
  std::string cls_text = "latency";
  bool contend = false;
  bool args_ok = true;

  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) {
        args_ok = false;
        return {};
      }
      return argv[++i];
    };
    long n = 0;
    size_t sz = 0;
    if (a == "--port") {
      if (!parse_long(val(), 1, 65535, &n)) args_ok = false;
      else c.port = static_cast<uint16_t>(n);
    } else if (a == "--dev") {
      const std::string d = val();
      if (d.empty()) args_ok = false;
      else c.devs.push_back(d);
    } else if (a == "--world") {
      if (!parse_long(val(), 2, 4, &n)) args_ok = false;
      else c.world = static_cast<int>(n);
    } else if (a == "--duration-ms") {
      if (!parse_long(val(), 100, 3600000, &n)) args_ok = false;
      else duration_ms = n;
    } else if (a == "--soak-ms") {
      // ping --contend --soak-ms N: duration-bounded mixed-class soak
      // (bulk flood + latency probes concurrent until the deadline).
      if (!parse_long(val(), 1000, 3600000, &n)) args_ok = false;
      else soak_ms = n;
    } else if (a == "--lat-slots") {
      if (!parse_long(val(), 1, 240, &n)) args_ok = false;
      else c.lat_slots = static_cast<int>(n);
    } else if (a == "--lat-bytes") {
      if (!parse_size(val(), 64, 1 << 20, &sz)) args_ok = false;
      else c.lat_bytes = sz;
    } else if (a == "--bulk-slots") {
      if (!parse_long(val(), 1, 240, &n)) args_ok = false;
      else c.bulk_slots = static_cast<int>(n);
    } else if (a == "--bulk-bytes") {
      if (!parse_size(val(), 64, 4 << 20, &sz)) args_ok = false;
      else c.bulk_bytes = sz;
    } else if (a == "--iters") {
      if (!parse_long(val(), 1, 1000000, &n)) args_ok = false;
      else iters = n;
    } else if (a == "--bytes") {
      if (!parse_size(val(), 8, 64 << 20, &sz)) args_ok = false;
      else bytes = sz;
    } else if (a == "--class") {
      cls_text = val();
      if (cls_text != "latency" && cls_text != "bulk") args_ok = false;
    } else if (a == "--contend") {
      contend = true;
    } else if (a == "--lat-iters") {
      if (!parse_long(val(), 1, 1000000, &n)) args_ok = false;
      else lat_iters = n;
    } else if (a == "--window") {
      if (!parse_long(val(), 1, 256, &n)) args_ok = false;
      else window = n;
    } else if (a == "--hold-ms") {
      if (!parse_long(val(), 0, 3600000, &n)) args_ok = false;
      else hold_ms = n;
    } else if (a == "--graph-gens") {
      if (!parse_long(val(), 0, 64, &n)) args_ok = false;
      else graph_gens = n;
    } else if (a == "--rank") {
      if (!parse_long(val(), 0, 3, &n)) args_ok = false;
      else c.rank = static_cast<int>(n);
    } else if (a == "--timeout-ms") {
      if (!parse_long(val(), 100, 600000, &n)) args_ok = false;
      else c.timeout_ms = static_cast<int>(n);
    } else if (a == "--peer") {
      peer = val();
      if (peer.empty()) args_ok = false;
    } else {
      DGPP_LOG_ERROR("unexpected argument {}", a);
      args_ok = false;
    }
  }
  if (!args_ok) return 2;

  if (mode == "serve") return run_serve(c, duration_ms);
  if (mode == "allreduce") {
    if (c.world < 2 || c.rank >= c.world) {
      DGPP_LOG_ERROR("allreduce: need --world >= 2 and --rank < world");
      return 2;
    }
    return run_allreduce(c, peer, static_cast<int>(iters), hold_ms,
                          static_cast<int>(graph_gens));
  }
  if (mode == "ping") {
    if (peer.empty() || c.rank == 0) return 2;
    if (soak_ms > 0) {
      if (!contend) {
        DGPP_LOG_ERROR("--soak-ms requires --contend (mixed-class soak)");
        return 2;
      }
      return run_soak(c, peer, bytes, soak_ms);
    }
    return run_ping(c, peer, static_cast<int>(iters), bytes, cls_text,
                    contend, static_cast<int>(lat_iters), window, hold_ms);
  }
  DGPP_LOG_ERROR("unknown mode {}", mode);
  return 2;
}
