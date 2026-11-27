// CollectiveBus deep scenarios (M5 deliverable 2). Everything runs against
// a real loopback pair: two bus endpoints in one process, real verbs QPs
// through the fabric switch to ourselves (validated by the M0 smoke), real
// pinned slabs, real consumer kernels. The scenarios pin the §6 contract:
// payload integrity via the fold hash, credit recycling, dual-lane striping,
// latency-under-bulk contention, watchdog behavior, config-error
// legibility, and orderly stop.

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "common/cpu_wait.hpp"
#include "common/log.hpp"
#include "net/bus_kernel.hpp"
#include "net/collective_bus.hpp"

void test_stream_boundary_reduce(dgpp::net::CollectiveBus& bus, uint16_t* data, size_t elems, cudaStream_t stream);

namespace {

using dgpp::net::BusMessageClass;
using dgpp::net::BusOptions;
using dgpp::net::BusLatencyTransport;
using dgpp::net::BusRecvView;
using dgpp::net::BusSendResult;
using dgpp::net::BusStats;
using dgpp::net::CollectiveBus;

int g_failures = 0;

#define CHECK(cond, msg)                                              \
  do {                                                                \
    if (!(cond)) {                                                    \
      DGPP_LOG_ERROR("FAIL {}:{} — {}", __FILE__, __LINE__, msg);     \
      ++g_failures;                                                    \
    }                                                                 \
  } while (0)

double percentile(std::vector<double> v, double frac) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<size_t>(frac * v.size()))];
}

// The graph/stream scenarios run under both latency transports (the
// 2026-10-19 one-sided write ring and the send pairs); the eager and
// harness scenarios use the send machinery either way.
BusLatencyTransport g_transport = BusLatencyTransport::kSend;
const char* transport_name() { return g_transport == BusLatencyTransport::kWrite ? "write" : "send"; }

BusOptions base_options(int rank, uint16_t port) {
  BusOptions o;
  o.world_size = 2;
  o.my_rank = rank;
  o.rendezvous_port = port;
  o.rendezvous_host = rank == 0 ? "" : "127.0.0.1";
  o.rendezvous_timeout_ms = 20000;
  o.lat_slots = 8;
  o.lat_slot_bytes = 8192;
  o.bulk_slots = 8;
  o.bulk_slot_bytes = 262144;
  o.qp_depth = 1024;
  o.completion_timeout_ms = 5000;
  o.consumer_deadline_s = 300.0;  // scenarios pause between phases
  o.latency_transport = g_transport;
  return o;
}

// Starts a rank-0 listener bus (in a thread) and a rank-1 connector bus,
// both fully up, or fails the scenario.
std::pair<std::unique_ptr<CollectiveBus>, std::unique_ptr<CollectiveBus>>
start_pair(const BusOptions& a_opt, const BusOptions& b_opt) {
  auto a = std::make_unique<CollectiveBus>(a_opt);
  auto b = std::make_unique<CollectiveBus>(b_opt);
  std::string a_error;
  std::thread a_thread([&] {
    if (!a->start(&a_error)) DGPP_LOG_ERROR("rank A start: {}", a_error);
  });
  std::string b_error;
  const bool b_ok = b->start(&b_error);
  a_thread.join();
  CHECK(a_error.empty(), "rank A start failed: " + a_error);
  CHECK(b_ok, "rank B start failed: " + b_error);
  if (!a_error.empty() || !b_ok) return {nullptr, nullptr};
  return {std::move(a), std::move(b)};
}

void fill_payload(uint64_t* data, size_t words, uint32_t salt) {
  uint64_t x = 0x9E3779B97F4A7C15ULL ^ (0x100000001B3ULL * salt);
  for (size_t i = 0; i < words; ++i) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    data[i] = x;
  }
}

bool send_and_verify(CollectiveBus& bus, int peer, BusMessageClass cls,
                     const void* data, size_t bytes, int wait_ms,
                     BusSendResult* out) {
  std::string error;
  const uint64_t id = bus.send(peer, data, bytes, cls, &error);
  if (id == 0) {
    DGPP_LOG_ERROR("send rejected: {}", error);
    return false;
  }
  *out = bus.wait(id, wait_ms);
  if (!out->ok) return false;
  // Stripe i covers bytes [i*stripe_bytes, min((i+1)*stripe_bytes, bytes)).
  const size_t stripe_bytes = bus.slot_bytes(cls);
  uint64_t got_xor = 0, want_xor = 0;
  for (size_t i = 0; i < out->stripe_hashes.size(); ++i) {
    const size_t off = i * stripe_bytes;
    const size_t chunk = std::min(stripe_bytes, bytes - off);
    got_xor ^= out->stripe_hashes[i];
    want_xor ^= dgpp::net::bus_fold(
        static_cast<const uint8_t*>(data) + off, chunk);
  }
  if (got_xor != want_xor) {
    DGPP_LOG_ERROR("hash mismatch: got {:x} want {:x}", got_xor, want_xor);
    return false;
  }
  return true;
}

// ---- scenarios ----------------------------------------------------------------

void scenario_latency_ping() {
  auto [a, b] = start_pair(base_options(0, 29810), base_options(1, 29810));
  if (!a || !b) return;
  std::vector<uint64_t> payload(8192 / 8);
  int ok = 0;
  std::vector<double> latency;
  for (int i = 0; i < 64; ++i) {
    fill_payload(payload.data(), payload.size(), static_cast<uint32_t>(i));
    BusSendResult r;
    if (send_and_verify(*b, 0, BusMessageClass::kLatency, payload.data(),
                        8192, 15000, &r)) {
      ++ok;
      latency.push_back(r.elapsed_us);
    }
  }
  CHECK(ok == 64, "latency ping: only " + std::to_string(ok) + "/64 verified");
  DGPP_LOG_INFO("scenario latency_ping: p50={:.1f}us p99={:.1f}us",
                percentile(latency, 0.5), percentile(latency, 0.99));
  a->quiesce();
  b->quiesce();
  a->stop();
  b->stop();
}

void scenario_credit_recycle() {
  // 4 latency slots per lane, 96 messages: recycling must keep the pool
  // flowing or the run stalls (and the watchdog would fail it loudly).
  BusOptions a_opt = base_options(0, 29820);
  BusOptions b_opt = base_options(1, 29820);
  a_opt.lat_slots = 4;
  b_opt.lat_slots = 4;
  auto [a, b] = start_pair(a_opt, b_opt);
  if (!a || !b) return;
  std::vector<uint64_t> payload(8192 / 8);
  int ok = 0;
  for (int i = 0; i < 96; ++i) {
    fill_payload(payload.data(), payload.size(), static_cast<uint32_t>(i));
    BusSendResult r;
    if (send_and_verify(*b, 0, BusMessageClass::kLatency, payload.data(),
                        8192, 15000, &r))
      ++ok;
  }
  CHECK(ok == 96, "credit recycle: only " + std::to_string(ok) + "/96");
  a->quiesce();
  b->quiesce();
  a->stop();
  b->stop();
}

void scenario_bulk_dual_lane() {
  auto [a, b] = start_pair(base_options(0, 29830), base_options(1, 29830));
  if (!a || !b) return;
  std::vector<uint64_t> payload(8u << 20);  // 64 MiB / 8
  fill_payload(payload.data(), payload.size(), 7u);
  const size_t bytes = payload.size() * 8;
  const auto t0 = std::chrono::steady_clock::now();
  BusSendResult r;
  const bool ok = send_and_verify(*b, 0, BusMessageClass::kBulk,
                                  payload.data(), bytes, 60000, &r);
  const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  CHECK(ok, "bulk 64MiB message failed verification");
  CHECK(r.stripe_hashes.size() >= 4,
        "expected >= 4 stripes for 64MiB, got " +
            std::to_string(r.stripe_hashes.size()));
  // Both lanes must contribute: credits received per lane on the sender,
  // doorbells received per lane on the receiver.
  int sender_lanes = 0, receiver_lanes = 0;
  for (const auto& l : b->stats().lanes)
    if (l.credits_received > 0) ++sender_lanes;
  for (const auto& l : a->stats().lanes)
    if (l.doorbell_recvs > 0) ++receiver_lanes;
  CHECK(sender_lanes == 2, "striping used " + std::to_string(sender_lanes) +
                                " lanes (sender view)");
  CHECK(receiver_lanes == 2, "striping used " +
                                 std::to_string(receiver_lanes) +
                                 " lanes (receiver view)");
  DGPP_LOG_INFO("scenario bulk_dual_lane: {:.1f} MiB in {:.3f}s = {:.2f} GB/s",
                bytes / 1048576.0, secs,
                secs > 0 ? static_cast<double>(bytes) / secs / 1e9 : 0.0);
  a->quiesce();
  b->quiesce();
  a->stop();
  b->stop();
}

void scenario_latency_under_bulk() {
  auto [a, b] = start_pair(base_options(0, 29840), base_options(1, 29840));
  if (!a || !b) return;

  std::vector<uint64_t> bulk(2u << 20);  // 16 MiB payloads
  std::vector<uint64_t> lat(8192 / 8);
  std::atomic<int> bulk_fail{0};
  const int bulk_iters = 24;
  const int lat_iters = 256;

  std::thread bulk_thread([&] {
    for (int i = 0; i < bulk_iters; ++i) {
      fill_payload(bulk.data(), bulk.size(), static_cast<uint32_t>(i));
      BusSendResult r;
      if (!send_and_verify(*b, 0, BusMessageClass::kBulk, bulk.data(),
                           bulk.size() * 8, 30000, &r))
        bulk_fail.fetch_add(1);
    }
  });

  std::vector<double> latency;
  int ok = 0;
  for (int i = 0; i < lat_iters; ++i) {
    fill_payload(lat.data(), lat.size(), 0xF000 + static_cast<uint32_t>(i));
    BusSendResult r;
    if (send_and_verify(*b, 0, BusMessageClass::kLatency, lat.data(), 8192,
                        30000, &r)) {
      ++ok;
      latency.push_back(r.elapsed_us);
    }
  }
  bulk_thread.join();

  CHECK(bulk_fail.load() == 0, "bulk stream had failures under contention");
  CHECK(ok == lat_iters, "latency stream lost " +
                             std::to_string(lat_iters - ok) + " messages");
  const double p99 = percentile(latency, 0.99);
  // Generous CI budget (headless node, two threads); the recorded number is
  // what matters for the exit-criterion record.
  CHECK(p99 < 20000.0, "latency p99 under bulk = " + std::to_string(p99) +
                           "us, over the 20ms budget");
  DGPP_LOG_INFO("scenario latency_under_bulk: lat p50={:.1f}us p99={:.1f}us "
                "while {}x16MiB bulk streamed",
                percentile(latency, 0.5), p99, bulk_iters);
  a->quiesce();
  b->quiesce();
  a->stop();
  b->stop();
}

void scenario_completion_timeout() {
  // No consumer on the RECEIVER: the sender's request must fail loudly via
  // the per-request watchdog, not hang. (The sender's own consumers are
  // irrelevant — it is the receiving rank's GPU that acks.)
  BusOptions a_opt = base_options(0, 29850);
  BusOptions b_opt = base_options(1, 29850);
  a_opt.launch_consumers = false;
  a_opt.completion_timeout_ms = 600;
  b_opt.completion_timeout_ms = 600;
  auto [a, b] = start_pair(a_opt, b_opt);
  if (!a || !b) return;
  std::vector<uint64_t> payload(8192 / 8);
  fill_payload(payload.data(), payload.size(), 1);
  std::string error;
  const uint64_t id =
      b->send(0, payload.data(), 8192, BusMessageClass::kLatency, &error);
  CHECK(id != 0, "send rejected: " + error);
  const BusSendResult r = b->wait(id, 10000);
  CHECK(!r.ok, "request should have failed without a receiver consumer");
  CHECK(r.error.find("completion timeout") != std::string::npos ||
            r.error.find("watchdog") != std::string::npos,
        "expected a watchdog reason, got: " + r.error);
  DGPP_LOG_INFO("scenario completion_timeout: failed as designed ({})",
                r.error);
  a->quiesce();
  b->quiesce();
  a->stop();
  b->stop();
}

void scenario_consumer_inactivity_exit() {
  // A consumer with a short deadline and no traffic must exit cleanly via
  // the inactivity watchdog; the stream sync must return promptly.
  BusOptions a_opt = base_options(0, 29860);
  BusOptions b_opt = base_options(1, 29860);
  b_opt.launch_consumers = false;
  auto [a, b] = start_pair(a_opt, b_opt);
  if (!a || !b) return;

  cudaStream_t stream;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) ==
            cudaSuccess,
        "stream create failed");
  const uint64_t deadline = dgpp::net::bus_consumer_deadline_cycles(0.3);
  CHECK(deadline > 0, "deadline cycles unavailable");
  const BusRecvView view = b->recv_view(0, 0);
  CHECK(view.control != nullptr, "recv view is empty");
  CHECK(dgpp::net::launch_bus_consumer(view, deadline, stream) ==
            cudaSuccess,
        "consumer launch failed");

  const auto t0 = std::chrono::steady_clock::now();
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess,
        "consumer stream sync failed");
  const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  CHECK(secs < 5.0, "consumer exit took " + std::to_string(secs) +
                        "s; watchdog did not bound inactivity");
  cudaStreamDestroy(stream);
  DGPP_LOG_INFO("scenario consumer_inactivity_exit: exited after {:.2f}s",
                secs);
  a->quiesce();
  b->quiesce();
  a->stop();
  b->stop();
}

void scenario_mesh_three_way() {
  // World=3 loopback mesh: rank 0 listens, ranks 1 and 2 connect, and every
  // rank exchanges with every peer. Exercises the full endpoint-table
  // distribution (the multi-peer path the two-rank scenarios never touch)
  // on one node, one log.
  const uint16_t port = 29890;
  BusOptions o0 = base_options(0, port);
  BusOptions o1 = base_options(1, port);
  BusOptions o2 = base_options(2, port);
  o0.world_size = 3;
  o1.world_size = 3;
  o2.world_size = 3;
  auto a = std::make_unique<CollectiveBus>(o0);  // rank 0, listener
  auto b = std::make_unique<CollectiveBus>(o1);
  auto c = std::make_unique<CollectiveBus>(o2);

  std::string a_error;
  std::thread a_thread([&] {
    if (!a->start(&a_error)) DGPP_LOG_ERROR("mesh A: {}", a_error);
  });
  auto start_rank = [](CollectiveBus* bus, std::string* error) {
    if (!bus->start(error)) DGPP_LOG_ERROR("mesh: {}", *error);
  };
  std::string b_error, c_error;
  std::thread b_thread(start_rank, b.get(), &b_error);
  std::thread c_thread(start_rank, c.get(), &c_error);
  a_thread.join();
  b_thread.join();
  c_thread.join();
  const bool started =
      a_error.empty() && b_error.empty() && c_error.empty();
  CHECK(started, "mesh startup failed: A=" + a_error + " B=" + b_error +
                     " C=" + c_error);
  if (!started) {
    // Error path must quiesce all consumers before any teardown frees
    // (cudaFreeHost synchronizes the device implicitly).
    a->quiesce();
    b->quiesce();
    c->quiesce();
    a->stop();
    b->stop();
    c->stop();
    return;
  }

  // Every rank sends to every peer; verify each.
  struct Lane { int from; int to; };
  const Lane lanes[] = {{1, 0}, {2, 0}, {0, 1}, {2, 1}, {0, 2}, {1, 2}};
  int failures = 0;
  for (const Lane& l : lanes) {
    CollectiveBus* from = l.from == 0 ? a.get() : (l.from == 1 ? b.get() : c.get());
    std::vector<uint64_t> payload(8192 / 8);
    fill_payload(payload.data(), payload.size(),
                 static_cast<uint32_t>(l.from * 10 + l.to));
    for (int i = 0; i < 4; ++i) {
      BusSendResult r;
      if (!send_and_verify(*from, l.to, BusMessageClass::kLatency,
                           payload.data(), 8192, 15000, &r)) {
        DGPP_LOG_ERROR("mesh {}->{} iter {} failed", l.from, l.to, i);
        ++failures;
      }
    }
  }
  CHECK(failures == 0,
        "three-way mesh had " + std::to_string(failures) + " failures");
  DGPP_LOG_INFO("scenario mesh_three_way: {} failures over 6 directed pairs",
                failures);
  a->quiesce();
  b->quiesce();
  c->quiesce();
  a->stop();
  b->stop();
  c->stop();
}

// ---- bf16 helpers: shared in bus_kernel.hpp (match RNE exactly) -----------

// Deterministic per-rank bf16 pattern with arbitrary mantissas (the
// rounding path is exercised; the oracle chain matters).
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

// Device twin of fill_rank_bf16 — the producing-kernel half of the
// pre-stage interface: a GPU kernel writing the NIC-registered pinned buffer
// directly (what the boundary GEMM does in the model interface). Single thread
// so the sequential LCG chain matches the host pattern element-for-element.
__global__ void staged_fill_kernel(uint16_t* dst, size_t elems, int rank) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  uint64_t x = 0x9E3779B97F4A7C15ULL ^ (0x100000001B3ULL * (rank + 1));
  for (size_t i = 0; i < elems; ++i) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    const float v =
        static_cast<float>(static_cast<int32_t>(x >> 40)) * (1.0f / 8388608.0f) - 1.0f;
    const uint32_t bits = __float_as_uint(v);
    dst[i] = static_cast<uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1)) >> 16);
  }
}

// Starts a loopback world: rank 0 listens, ranks 1..N-1 connect.
// Collective mode: no persistent consumers (they would race the
// per-collective kernel — and their 300s deadlines would turn any
// implicit-sync call in the workers into a hang).
std::vector<std::unique_ptr<CollectiveBus>> start_world(
    int world, uint16_t port, bool one_lane = false,
    int rank0_pass_delay_us = 0, size_t lat_slot_bytes = 8192,
    bool rank0_flight_park_decoy = false) {
  std::vector<std::unique_ptr<CollectiveBus>> out;
  for (int r = 0; r < world; ++r) {
    BusOptions o = base_options(r, port);
    o.world_size = world;
    o.launch_consumers = false;
    o.lat_slot_bytes = lat_slot_bytes;
    if (one_lane) o.lane_devices = {o.lane_devices.front()};
    // Fault injection (scenario_allreduce_done_before_posted): rank 0's
    // engine dawdles between its control-cell reads.
    if (r == 0) o.debug_pass_delay_us = rank0_pass_delay_us;
    // Fault injection (the flight park's breaker): rank 0's engine parks
    // beside its kernel's block records.
    if (r == 0) o.debug_flight_park_decoy = rank0_flight_park_decoy;
    out.push_back(std::make_unique<CollectiveBus>(o));
  }
  std::vector<std::string> errors(world);
  std::thread listener([&] {
    if (!out[0]->start(&errors[0])) DGPP_LOG_ERROR("world rank 0: {}", errors[0]);
  });
  std::vector<std::thread> connectors;
  for (int r = 1; r < world; ++r)
    connectors.emplace_back([&, r] {
      if (!out[static_cast<size_t>(r)]->start(&errors[static_cast<size_t>(r)]))
        DGPP_LOG_ERROR("world rank {}: {}", r, errors[static_cast<size_t>(r)]);
    });
  listener.join();
  for (auto& t : connectors) t.join();
  for (int r = 0; r < world; ++r)
    if (!errors[static_cast<size_t>(r)].empty()) return {};
  return out;
}

// One rank's share of a collective run: `iters` collectives over the same
// buffers (slot generations advance, recycling is exercised), each result
// checked bitwise against the canonical-chain oracle. Returns the number
// of local failures (CHECK is main-thread-only; workers report home).
int allreduce_rank_work(CollectiveBus& bus, int world, int my_rank,
                        size_t elems, int iters) {
  std::vector<uint16_t> host_src, host_dst(elems, 0);
  fill_rank_bf16(&host_src, elems, my_rank);
  // Any rank can regenerate any rank's pattern — the oracle needs them all.
  std::vector<std::vector<uint16_t>> all_src(world);
  for (int r = 0; r < world; ++r) fill_rank_bf16(&all_src[static_cast<size_t>(r)], elems, r);

  uint16_t* dev_src = nullptr;
  uint16_t* dev_dst = nullptr;
  if (cudaMalloc(&dev_src, elems * 2) != cudaSuccess ||
      cudaMalloc(&dev_dst, elems * 2) != cudaSuccess) {
    DGPP_LOG_ERROR("rank {}: device alloc failed", my_rank);
    return 1;
  }
  if (cudaMemcpy(dev_src, host_src.data(), elems * 2, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    DGPP_LOG_ERROR("rank {}: H2D failed", my_rank);
    cudaFree(dev_src);
    cudaFree(dev_dst);
    return 1;
  }

  int failures = 0;
  std::vector<uint16_t> got(elems, 0);
  for (int iter = 0; iter < iters; ++iter) {
    std::string error;
    const uint64_t id = bus.allreduce(dev_src, dev_dst, elems, &error);
    if (id == 0) {
      DGPP_LOG_ERROR("rank {}: allreduce rejected: {}", my_rank, error);
      ++failures;
      break;
    }
    const dgpp::net::BusAllReduceResult r = bus.wait_allreduce(id, 15000);
    if (!r.ok) {
      DGPP_LOG_ERROR("rank {}: collective failed: {}", my_rank, r.error);
      ++failures;
      break;
    }
    if (cudaMemcpy(got.data(), dev_dst, elems * 2, cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
      DGPP_LOG_ERROR("rank {}: D2H failed", my_rank);
      ++failures;
      break;
    }
    // Canonical chain: acc = 0; for r in 0..world-1: acc += f32(src_r[i]).
    size_t mismatches = 0;
    for (size_t i = 0; i < elems; ++i) {
      float acc = 0.0f;
      for (int r = 0; r < world; ++r)
        acc += dgpp::net::bf16_to_f32(all_src[static_cast<size_t>(r)][i]);
      const uint16_t want = dgpp::net::bf16_from_f32_rne(acc);
      if (got[i] != want) {
        if (mismatches < 3)
          DGPP_LOG_ERROR("rank {} iter {} elem {}: got {:04x} want {:04x}",
                         my_rank, iter, i, got[i], want);
        ++mismatches;
      }
    }
    if (mismatches != 0) {
      DGPP_LOG_ERROR("rank {} iter {}: {} mismatches", my_rank, iter,
                     mismatches);
      ++failures;
    }
  }

  cudaFree(dev_src);
  cudaFree(dev_dst);
  return failures;
}

void scenario_collective_progress() {
  auto buses = start_world(2, 29908);
  CHECK(buses.size() == 2, "progress world failed to start");
  if (buses.size() != 2) return;
  constexpr size_t bulk_elems = 16384;
  uint16_t* data[2]{};
  for (int rank = 0; rank < 2; ++rank) {
    CHECK(cudaMalloc(&data[rank], bulk_elems * 2) == cudaSuccess, "progress alloc");
    CHECK(cudaMemset(data[rank], 0, bulk_elems * 2) == cudaSuccess, "progress zero");
    CHECK(buses[rank]->completion_epoch().load() == 0, "startup is not completed work");
  }
  std::string error;
  const uint64_t first = buses[0]->allreduce(data[0], data[0], 4096, &error);
  CHECK(first != 0, "progress submit: " + error);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  CHECK(buses[0]->completion_epoch().load() == 0, "submission/polling is not completion");
  const uint64_t second = buses[1]->allreduce(data[1], data[1], 4096, &error);
  CHECK(second != 0, "progress peer submit: " + error);
  CHECK(buses[0]->wait_allreduce(first, 15000).ok, "progress rank 0 completion");
  CHECK(buses[1]->wait_allreduce(second, 15000).ok, "progress rank 1 completion");
  for (const auto& bus : buses)
    CHECK(bus->completion_epoch().load() == 1, "eager completion must advance once");

  uint64_t ids[2]{};
  for (int rank = 0; rank < 2; ++rank)
    ids[rank] = buses[rank]->allreduce_bulk(data[rank], data[rank], bulk_elems, &error);
  for (int rank = 0; rank < 2; ++rank) {
    CHECK(ids[rank] != 0 && buses[rank]->wait_allreduce(ids[rank], 15000).ok,
          "bulk progress completion: " + error);
    CHECK(buses[rank]->completion_epoch().load() == 2, "bulk completion must advance once");
  }
  // A missing peer causes a real failed flight, not a successful heartbeat.
  const uint64_t stalled = buses[0]->allreduce(data[0], data[0], 4096, &error);
  CHECK(stalled != 0, "stalled progress submit: " + error);
  CHECK(!buses[0]->wait_allreduce(stalled, 15000).ok, "missing peer should fail");
  CHECK(buses[0]->completion_epoch().load() == 2, "failed flight must not advance progress");
  for (auto& bus : buses) bus->quiesce();
  for (auto& bus : buses) bus->stop();
  for (auto* buffer : data) cudaFree(buffer);
  DGPP_LOG_INFO("scenario collective_progress: successful completions only");
}

void scenario_allreduce() {
  // One-shot all-to-all all-reduce over loopback worlds: bitwise-identical
  // results on every rank, across generations (slot recycling), plus the
  // v1 contract pins (consumers-running rejection, mode-closed sends).
  const size_t elems = 4096;  // 8 KiB, the decode unit
  int failures = 0;

  for (const int world : {2, 4}) {
    const uint16_t port = world == 2 ? 29895 : 29896;
    std::vector<std::unique_ptr<CollectiveBus>> world_buses =
        start_world(world, port);
    if (world_buses.empty()) {
      DGPP_LOG_ERROR("world {} failed to start", world);
      ++failures;
      continue;
    }
    std::vector<std::thread> workers;
    std::vector<int> rank_failures(static_cast<size_t>(world), 0);
    for (int r = 0; r < world; ++r)
      workers.emplace_back([&, r] {
        rank_failures[static_cast<size_t>(r)] = allreduce_rank_work(
            *world_buses[static_cast<size_t>(r)], world, r, elems, 8);
      });
    for (auto& t : workers) t.join();
    int world_failures = 0;
    for (int r = 0; r < world; ++r) {
      if (rank_failures[static_cast<size_t>(r)] != 0)
        DGPP_LOG_ERROR("world {} rank {} reported failures", world, r);
      world_failures += rank_failures[static_cast<size_t>(r)];
    }
    CHECK(world_failures == 0,
          "world " + std::to_string(world) + " had " +
              std::to_string(world_failures) + " collective failures");
    for (auto& bus : world_buses) bus->quiesce();
    for (auto& bus : world_buses) bus->stop();
    DGPP_LOG_INFO("scenario allreduce: world {} clean", world);
  }

  // Contract pins on a dedicated pair: allreduce is rejected while harness
  // consumers run, and send() is closed once the bus is in collective mode.
  {
    auto [a, b] = start_pair(base_options(0, 29897), base_options(1, 29897));
    if (!a || !b) {
      ++failures;
    } else {
      std::string error;
      CHECK(a->allreduce(nullptr, nullptr, elems, &error) == 0 &&
                error.find("launch_consumers") != std::string::npos,
            "allreduce must be rejected while consumers run: " + error);
      CHECK(b->allreduce(nullptr, nullptr, elems, &error) == 0 &&
                error.find("launch_consumers") != std::string::npos,
            "peer allreduce must be rejected while consumers run: " + error);
      a->quiesce();
      b->quiesce();
      a->stop();
      b->stop();
    }
  }

  {
    // Mode pin: once a collective has been picked up, harness sends fail
    // legibly (their messages would be folded into a peer's reduce). Both
    // ranks participate; rank 0 probes send() mid-flight, after giving the
    // engine a moment to enter collective mode.
    std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(2, 29898);
    if (buses.empty()) {
      ++failures;
    } else {
      uint16_t* dev = nullptr;
      CHECK(cudaMalloc(&dev, elems * 2) == cudaSuccess, "device alloc failed");
      std::thread peer_worker([&, elems] {
        uint16_t* peer_dev = nullptr;
        cudaMalloc(&peer_dev, elems * 2);
        std::string peer_error;
        const uint64_t peer_id =
            buses[1]->allreduce(peer_dev, peer_dev, elems, &peer_error);
        if (peer_id != 0)
          buses[1]->wait_allreduce(peer_id, 15000);
        cudaFree(peer_dev);
      });
      std::string error;
      const uint64_t id = buses[0]->allreduce(dev, dev, elems, &error);
      CHECK(id != 0, "allreduce rejected: " + error);
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      std::vector<uint64_t> payload(8192 / 8);
      const uint64_t send_id = buses[0]->send(
          1, payload.data(), 8192, BusMessageClass::kLatency, &error);
      CHECK(send_id == 0 && error.find("collective mode") != std::string::npos,
            "send() must be closed in collective mode: " + error);
      const dgpp::net::BusAllReduceResult r =
          buses[0]->wait_allreduce(id, 15000);
      CHECK(r.ok, "collective failed: " + r.error);
      peer_worker.join();
      cudaFree(dev);
      for (auto& bus : buses) bus->quiesce();
      for (auto& bus : buses) bus->stop();
    }
  }

  g_failures += failures;
  DGPP_LOG_INFO("scenario allreduce: {} total failures", g_failures);
}

// One rank's share of a pre-staged run: stage_next(), a GPU kernel writes
// the pinned buffer (the interface's producing half), allreduce_staged(), wait,
// and an in-place bitwise check against the canonical-chain oracle. The
// result never leaves the pinned buffer — exactly the model's shape.
int allreduce_staged_rank_work(CollectiveBus& bus, int world, int my_rank,
                               size_t elems, int iters,
                               std::vector<uint16_t>* want) {
  // The producing kernel runs on a private stream and the worker
  // STREAM-syncs before submit — never cudaDeviceSynchronize, which is a
  // whole-device barrier and deadlocks against peer collective kernels
  // spinning on this device in the loopback process (DESIGN §6.3's
  // allocation-phase ordering rule, same class).
  cudaStream_t fill_stream = nullptr;
  if (cudaStreamCreateWithFlags(&fill_stream, cudaStreamNonBlocking) !=
      cudaSuccess) {
    DGPP_LOG_ERROR("rank {}: fill stream create failed", my_rank);
    return 1;
  }
  int failures = 0;
  for (int iter = 0; iter < iters; ++iter) {
    std::string error;
    uint16_t* staged = static_cast<uint16_t*>(bus.stage_next(&error));
    if (staged == nullptr) {
      DGPP_LOG_ERROR("rank {}: stage_next rejected: {}", my_rank, error);
      ++failures;
      break;
    }
    // Contract pins (rank 0, first iteration, before consuming): one
    // handout at a time, and a device-source collective must not stomp
    // it. Counted, not CHECKed — workers report home (g_failures is
    // main-thread state).
    if (my_rank == 0 && iter == 0) {
      std::string pin_error;
      if (bus.stage_next(&pin_error) != nullptr ||
          pin_error.find("one pre-stage handout") == std::string::npos) {
        DGPP_LOG_ERROR("rank 0: double stage_next not rejected: {}", pin_error);
        ++failures;
      }
      if (bus.allreduce(nullptr, nullptr, elems, &pin_error) != 0 ||
          pin_error.find("pre-stage handout is held") == std::string::npos) {
        DGPP_LOG_ERROR("rank 0: allreduce under a held handout not rejected: {}",
                       pin_error);
        ++failures;
      }
    }
    // The producing kernel writes the send source directly; a stream sync
    // (not a device sync — see the loopback note above) orders it before
    // the submit, which is the interface's contract.
    staged_fill_kernel<<<1, 1, 0, fill_stream>>>(staged, elems, my_rank);
    if (cudaStreamSynchronize(fill_stream) != cudaSuccess) {
      DGPP_LOG_ERROR("rank {}: staged fill kernel failed", my_rank);
      ++failures;
      break;
    }
    const uint64_t id = bus.allreduce_staged(elems, &error);
    if (id == 0) {
      DGPP_LOG_ERROR("rank {}: allreduce_staged rejected: {}", my_rank, error);
      ++failures;
      break;
    }
    const dgpp::net::BusAllReduceResult r = bus.wait_allreduce(id, 15000);
    if (!r.ok) {
      DGPP_LOG_ERROR("rank {}: staged collective failed: {}", my_rank,
                     r.error);
      ++failures;
      break;
    }
    size_t mismatches = 0;
    for (size_t i = 0; i < elems; ++i)
      if (staged[i] != (*want)[i]) ++mismatches;
    if (mismatches != 0) {
      DGPP_LOG_ERROR("rank {} iter {}: {} staged mismatches", my_rank, iter,
                     mismatches);
      ++failures;
      break;
    }
    if (my_rank == 0 && iter == 0) {
      // The handout was consumed; a staged submit without one must fail.
      std::string pin_error;
      if (bus.allreduce_staged(elems, &pin_error) != 0 ||
          pin_error.find("no held pre-stage handout") == std::string::npos) {
        DGPP_LOG_ERROR("rank 0: staged submit without handout not rejected: {}",
                       pin_error);
        ++failures;
      }
    }
  }
  cudaStreamDestroy(fill_stream);
  return failures;
}

void scenario_allreduce_done_before_posted() {
  // DONE DOES not IMPLY POSTED, the one-shot form: rank 0's
  // engine sleeps 300 us between reading the control cell's done stamp
  // and its ready bits, so its kernel — whose peers post at once, unpaced
  // — stages, claims, folds and stamps done inside that gap on every
  // collective. Before the fix the engine read ready first (0), slept,
  // read done (== seq) and completed the flight with rank 0's own stripe
  // never posted: the peers' kernels waited on it forever and rank 0's
  // next generation parked behind their generation gate (glm_tp_test's
  // two-rank loopback, once in fifteen runs). Now the pass reads done
  // first and refuses to finish a reduced flight before every peer's
  // stripe is out; every collective completes bitwise on every rank.
  const size_t elems = 4096;
  int failures = 0;
  for (const int world : {2, 4}) {
    const uint16_t port = world == 2 ? 29906 : 29907;
    std::vector<std::unique_ptr<CollectiveBus>> world_buses =
        start_world(world, port, /*one_lane=*/false,
                    /*rank0_pass_delay_us=*/300);
    if (world_buses.empty()) {
      DGPP_LOG_ERROR("done-before-posted world {} failed to start", world);
      ++failures;
      continue;
    }
    std::vector<std::thread> workers;
    std::vector<int> rank_failures(static_cast<size_t>(world), 0);
    for (int r = 0; r < world; ++r)
      workers.emplace_back([&, r] {
        rank_failures[static_cast<size_t>(r)] = allreduce_rank_work(
            *world_buses[static_cast<size_t>(r)], world, r, elems, 40);
      });
    for (auto& t : workers) t.join();
    int world_failures = 0;
    for (int r = 0; r < world; ++r)
      world_failures += rank_failures[static_cast<size_t>(r)];
    CHECK(world_failures == 0,
          "done-before-posted world " + std::to_string(world) + " had " +
              std::to_string(world_failures) + " collective failures");
    for (auto& bus : world_buses) bus->quiesce();
    for (auto& bus : world_buses) bus->stop();
    DGPP_LOG_INFO("scenario allreduce_done_before_posted: world {} clean "
                  "(40 collectives with rank 0's engine 300 us late)",
                  world);
  }
  g_failures += failures;
  DGPP_LOG_INFO("scenario allreduce_done_before_posted: {} total failures",
                g_failures);
}

void scenario_allreduce_staged() {
  // Pre-stage interface (§6.3 evolution): the producing kernel writes the
  // transport's pinned send source directly and the collective runs with
  // zero staging copies (world 2) or only the pinned fan-out (world 4).
  // Bitwise against the canonical-chain oracle, across ring generations.
  const size_t elems = 4096;
  int failures = 0;

  for (const int world : {2, 4}) {
    const uint16_t port = world == 2 ? 29901 : 29902;
    std::vector<std::unique_ptr<CollectiveBus>> world_buses =
        start_world(world, port);
    if (world_buses.empty()) {
      DGPP_LOG_ERROR("staged world {} failed to start", world);
      ++failures;
      continue;
    }
    // The oracle chain over every rank's pattern.
    std::vector<std::vector<uint16_t>> all_src(static_cast<size_t>(world));
    for (int r = 0; r < world; ++r)
      fill_rank_bf16(&all_src[static_cast<size_t>(r)], elems, r);
    std::vector<uint16_t> want(elems, 0);
    for (size_t i = 0; i < elems; ++i) {
      float acc = 0.0f;
      for (int r = 0; r < world; ++r)
        acc += dgpp::net::bf16_to_f32(all_src[static_cast<size_t>(r)][i]);
      want[i] = dgpp::net::bf16_from_f32_rne(acc);
    }
    std::vector<std::thread> workers;
    std::vector<int> rank_failures(static_cast<size_t>(world), 0);
    for (int r = 0; r < world; ++r)
      workers.emplace_back([&, r] {
        rank_failures[static_cast<size_t>(r)] = allreduce_staged_rank_work(
            *world_buses[static_cast<size_t>(r)], world, r, elems, 6, &want);
      });
    for (auto& t : workers) t.join();
    int world_failures = 0;
    for (int r = 0; r < world; ++r)
      world_failures += rank_failures[static_cast<size_t>(r)];
    CHECK(world_failures == 0,
          "staged world " + std::to_string(world) + " had " +
              std::to_string(world_failures) + " failures");
    for (auto& bus : world_buses) bus->quiesce();
    for (auto& bus : world_buses) bus->stop();
    DGPP_LOG_INFO("scenario allreduce_staged: world {} clean", world);
  }

  g_failures += failures;
  DGPP_LOG_INFO("scenario allreduce_staged: {} total failures", g_failures);
}

// One rank's share of a bulk-collective run: the machine over N iterations
// of one size — (1) allreduce_bulk verified bitwise against the canonical
// chain, (2) the latency-chunked path over the same buffer, verified
// bitwise against both the oracle and the bulk run's bytes (the two paths
// compute the identical per-element chain; any divergence is a protocol
// bug, not noise). Returns failures home.
int allreduce_bulk_rank_work(CollectiveBus& bus, int world, int my_rank,
                             size_t elems, int iters,
                             std::vector<uint16_t>* want) {
  // Loopback writer's discipline: EVERY device op in a rank worker runs on
  // a private stream. Synchronous cudaMemcpy/cudaMemset are device-wide
  // barriers (legacy default stream) and deadlock against a peer rank's
  // spinning collective kernel — the §6.3 rule, which has now bitten this
  // harness four separate ways (cudaDeviceSynchronize, sync H2D, memset,
  // and cudaEventCreate: a reducer that created its events in its
  // constructor stalled every second stream collective here, 2026-10-19).
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
      cudaSuccess) {
    DGPP_LOG_ERROR("rank {}: bulk stream create failed", my_rank);
    return 1;
  }
  std::vector<uint16_t> host_src;
  fill_rank_bf16(&host_src, elems, my_rank);
  uint16_t* dev_src = nullptr;
  uint16_t* dev_dst = nullptr;
  if (cudaMalloc(&dev_src, elems * 2) != cudaSuccess ||
      cudaMalloc(&dev_dst, elems * 2) != cudaSuccess) {
    DGPP_LOG_ERROR("rank {}: bulk device alloc failed", my_rank);
    return 1;
  }
  if (cudaMemcpyAsync(dev_src, host_src.data(), elems * 2,
                      cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    DGPP_LOG_ERROR("rank {}: bulk H2D failed", my_rank);
    cudaFree(dev_src);
    cudaFree(dev_dst);
    return 1;
  }

  int failures = 0;
  std::vector<uint16_t> got(elems, 0), bulk_run(elems, 0);
  for (int iter = 0; iter < iters; ++iter) {
    // ---- the bulk machine (RS + AG, segments internal) -----------------
    cudaMemsetAsync(dev_dst, 0, elems * 2, stream);
    cudaStreamSynchronize(stream);
    {
      std::string error;
      const uint64_t id = bus.allreduce_bulk(dev_src, dev_dst, elems, &error);
      if (id == 0) {
        DGPP_LOG_ERROR("rank {}: allreduce_bulk rejected: {}", my_rank,
                       error);
        ++failures;
        break;
      }
      const dgpp::net::BusAllReduceResult r = bus.wait_allreduce(id, 60000);
      if (!r.ok) {
        DGPP_LOG_ERROR("rank {}: bulk collective failed: {}", my_rank,
                       r.error);
        ++failures;
        break;
      }
    }
    if (cudaMemcpyAsync(got.data(), dev_dst, elems * 2,
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      DGPP_LOG_ERROR("rank {}: bulk D2H failed", my_rank);
      ++failures;
      break;
    }
    size_t mismatches = 0;
    // Bring-up fingerprint: which stripes mismatch (0=clean 1=partial
    // 2=fully wrong) plus the first/last mismatching element — the
    // distribution localizes a mapping bug faster than a count alone.
    size_t first_bad = elems, last_bad = 0;
    for (size_t i = 0; i < elems; ++i) {
      if (got[i] == (*want)[i]) continue;
      ++mismatches;
      first_bad = std::min(first_bad, i);
      last_bad = i;
    }
    if (mismatches != 0) {
      const size_t stripe = 131072;  // loopback bulk_slot_bytes / 2
      std::string map;
      for (size_t s = 0; s * stripe < elems; ++s) {
        const size_t b = s * stripe;
        const size_t e = std::min(elems, b + stripe);
        size_t bad = 0;
        for (size_t i = b; i < e; ++i)
          if (got[i] != (*want)[i]) ++bad;
        map += bad == 0 ? "0" : (bad == e - b ? "2" : "1");
      }
      DGPP_LOG_ERROR("rank {} bulk iter {}: {} oracle mismatches "
                     "[{},{}] stripes {}",
                     my_rank, iter, mismatches, first_bad, last_bad, map);
      for (size_t i = first_bad; i < first_bad + 4 && i < elems; ++i)
        DGPP_LOG_ERROR("rank {} elem {}: got {:04x} want {:04x} (src {:04x})",
                       my_rank, i, got[i], (*want)[i], host_src[i]);
      ++failures;
      break;
    }
    if (iter == 0) bulk_run = got;

    // ---- the latency-chunked path over the same buffer ------------------
    // Same chain per element, chunked into latency slots; the result must
    // be bitwise identical to the bulk machine's.
    cudaMemsetAsync(dev_dst, 0, elems * 2, stream);
    cudaStreamSynchronize(stream);
    bool chunk_failed = false;
    for (size_t base = 0; base < elems; base += 4096) {
      const size_t n = std::min<size_t>(4096, elems - base);
      std::string error;
      const uint64_t id =
          bus.allreduce(dev_src + base, dev_dst + base, n, &error);
      if (id == 0) {
        DGPP_LOG_ERROR("rank {}: chunked allreduce rejected: {}", my_rank,
                       error);
        ++failures;
        chunk_failed = true;
        break;
      }
      const dgpp::net::BusAllReduceResult r = bus.wait_allreduce(id, 30000);
      if (!r.ok) {
        DGPP_LOG_ERROR("rank {}: chunked collective failed: {}", my_rank,
                       r.error);
        ++failures;
        chunk_failed = true;
        break;
      }
    }
    if (chunk_failed) break;
    if (cudaMemcpyAsync(got.data(), dev_dst, elems * 2,
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      DGPP_LOG_ERROR("rank {}: chunked D2H failed", my_rank);
      ++failures;
      break;
    }
    if (std::memcmp(got.data(), bulk_run.data(), elems * 2) != 0) {
      DGPP_LOG_ERROR("rank {} iter {}: bulk and chunked paths diverged "
                     "bitwise",
                     my_rank, iter);
      ++failures;
      break;
    }
    // The serving reducer may use bulk for complete segments and the
    // model stream for a small tail. Compare that composition with the
    // full bulk oracle, including the segment/latency boundary cases.
    cudaMemcpyAsync(dev_dst, dev_src, elems * 2, cudaMemcpyDeviceToDevice, stream);
    test_stream_boundary_reduce(bus, dev_dst, elems, stream);
    cudaMemcpyAsync(got.data(), dev_dst, elems * 2, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::string settle_error;
    if (!bus.allreduce_settle(60000, &settle_error)) {
      DGPP_LOG_ERROR("rank {}: stream boundary settle failed: {}", my_rank, settle_error);
      ++failures;
      break;
    }
    if (got != bulk_run) {
      DGPP_LOG_ERROR("rank {} iter {}: stream boundary and bulk paths diverged bitwise", my_rank, iter);
      ++failures;
      break;
    }
  }

  cudaFree(dev_src);
  cudaFree(dev_dst);
  cudaStreamDestroy(stream);
  return failures;
}

std::vector<size_t> bulk_case_sizes(bool one_lane) {
  const size_t stripe = 262144 / 2;
  const size_t segment = stripe * 8 * (one_lane ? 1 : 2);
  return {2, 4, 6, 8, 2048, 4094, 4096, 4098, 8190, 8192, 8194, 2 * stripe + 100,
          segment, segment + 2, segment + 6, segment + 4096, segment + 4098,
          segment + 8192, segment + 8194, 41 * stripe - 32};
}

// One process per GPU, as in serving. This also lets Compute Sanitizer check
// the real distributed bulk/latency/stream composition without placing every
// rank's spinning kernels in the same instrumented CUDA context.
int distributed_bulk_cases(int rank, int world, const std::string& peer,
                           uint16_t port, bool one_lane) {
  auto options = base_options(rank, port);
  options.world_size = world;
  options.rendezvous_host = rank == 0 ? "" : peer;
  options.launch_consumers = false;
  if (one_lane) options.lane_devices.resize(1);
  CollectiveBus bus(options);
  std::string error;
  if (!bus.start(&error)) {
    DGPP_LOG_ERROR("distributed bulk rank {}: {}", rank, error);
    return 1;
  }
  int failures = 0;
  for (const size_t elems : bulk_case_sizes(one_lane)) {
    std::vector<std::vector<uint16_t>> sources(static_cast<size_t>(world));
    for (int r = 0; r < world; ++r) fill_rank_bf16(&sources[static_cast<size_t>(r)], elems, r);
    std::vector<uint16_t> want(elems);
    for (size_t i = 0; i < elems; ++i) {
      float value = 0;
      for (int r = 0; r < world; ++r) value += dgpp::net::bf16_to_f32(sources[static_cast<size_t>(r)][i]);
      want[i] = dgpp::net::bf16_from_f32_rne(value);
    }
    const int bad = allreduce_bulk_rank_work(bus, world, rank, elems, 2, &want);
    failures += bad;
    DGPP_LOG_INFO("distributed bulk rank={} world={} lanes={} elems={}: {} failures",
                  rank, world, bus.lane_count(), elems, bad);
    if (bad) break;
  }
  bus.quiesce();
  bus.stop();
  return failures == 0 ? 0 : 1;
}

void scenario_allreduce_bulk() {
  // The prefill class (§6.3): segment-quantized reduce-scatter + allgather
  // over the bulk pool. Sizes chosen to exercise the degenerate geometry:
  // below one stripe (empty shards at W=4), two stripes (more empty
  // shards), and a multi-segment buffer with a partial tail stripe and
  // shard/segment straddling — every result bitwise against the canonical
  // chain and against the latency-chunked path (identical chains).
  // DGPP_BULK_ONE_LANE=1 bisects lane-dependent failures (bring-up).
  int failures = 0;
  const bool one_lane = std::getenv("DGPP_BULK_ONE_LANE") != nullptr;

  for (const int world : {2, 4}) {
    const uint16_t port = world == 2 ? 29904 : 29905;
    for (const size_t elems : bulk_case_sizes(one_lane)) {
      std::vector<std::unique_ptr<CollectiveBus>> world_buses =
          start_world(world, port, one_lane);
      if (world_buses.empty()) {
        DGPP_LOG_ERROR("bulk world {} failed to start", world);
        ++failures;
        break;
      }
      // Oracle chain over every rank's pattern.
      std::vector<std::vector<uint16_t>> all_src(static_cast<size_t>(world));
      for (int r = 0; r < world; ++r)
        fill_rank_bf16(&all_src[static_cast<size_t>(r)], elems, r);
      std::vector<uint16_t> want(elems, 0);
      for (size_t i = 0; i < elems; ++i) {
        float acc = 0.0f;
        for (int r = 0; r < world; ++r)
          acc += dgpp::net::bf16_to_f32(all_src[static_cast<size_t>(r)][i]);
        want[i] = dgpp::net::bf16_from_f32_rne(acc);
      }
      std::vector<std::thread> workers;
      std::vector<int> rank_failures(static_cast<size_t>(world), 0);
      const auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < world; ++r)
        workers.emplace_back([&, r] {
          rank_failures[static_cast<size_t>(r)] = allreduce_bulk_rank_work(
              *world_buses[static_cast<size_t>(r)], world, r, elems, 2,
              &want);
        });
      for (auto& t : workers) t.join();
      int world_failures = 0;
      for (int r = 0; r < world; ++r)
        world_failures += rank_failures[static_cast<size_t>(r)];
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
      CHECK(world_failures == 0,
            "bulk world " + std::to_string(world) + " elems " +
                std::to_string(elems) + " had " +
                std::to_string(world_failures) + " failures");
      DGPP_LOG_INFO("bulk case world={} elems={}: clean ({:.0f}ms for 2 "
                    "iterations of both paths)",
                    world, elems, ms);
      for (auto& bus : world_buses) bus->quiesce();
      for (auto& bus : world_buses) bus->stop();
    }
  }

  g_failures += failures;
  DGPP_LOG_INFO("scenario allreduce_bulk: {} total failures", g_failures);
}

// One rank's share of a graph-era run (§6.2): the decode step's fixed
// launch sequence — collectives with GEMM-stand-in compute between them —
// recorded once, replayed per step. Every replay's result must be bitwise
// the canonical-chain oracle AND the eager machine's bytes (the recorded
// fold is the same chain; the doorbell/claim machinery is what differs).
// The replay loop is also the MIXED-ERA gate: eager collectives (the pick
// class, the staged interface, the prefill bulk class) run BETWEEN windows —
// consuming generations from the shared counter — and every later window
// must adopt around them and stay bitwise.
// Returns failures home (CHECK is main-thread-only).
int allreduce_graph_rank_work(CollectiveBus& bus, int world, int my_rank,
                              size_t elems, int gens_per_step, int replays,
                              const std::vector<uint16_t>* want) {
  // Loopback discipline: every device op on a private stream; the capture
  // is ThreadLocal (other rank workers capture concurrently).
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
      cudaSuccess) {
    DGPP_LOG_ERROR("rank {}: graph stream create failed", my_rank);
    return 1;
  }
  std::vector<uint16_t> host_src;
  fill_rank_bf16(&host_src, elems, my_rank);
  uint16_t* dev_src = nullptr;
  uint16_t* dev_dst = nullptr;
  float* warm_buf = nullptr;
  // The prefill-class interlude buffer: the bulk machine shards on the
  // bulk-slot grid, so it needs at least one stripe per rank (world x
  // stripe elems; 4 x 131072 = 1 MiB at the default geometry).
  const size_t bulk_elems =
      static_cast<size_t>(world) * (262144 / 2) * 4;
  uint16_t* dev_bsrc = nullptr;
  uint16_t* dev_bdst = nullptr;
  if (cudaMalloc(&dev_src, elems * 2) != cudaSuccess ||
      cudaMalloc(&dev_dst, elems * 2) != cudaSuccess ||
      cudaMalloc(&warm_buf, 256 * 4) != cudaSuccess ||
      cudaMalloc(&dev_bsrc, bulk_elems * 2) != cudaSuccess ||
      cudaMalloc(&dev_bdst, bulk_elems * 2) != cudaSuccess) {
    DGPP_LOG_ERROR("rank {}: graph device alloc failed", my_rank);
    cudaFree(dev_src);
    cudaFree(dev_dst);
    cudaFree(warm_buf);
    cudaFree(dev_bsrc);
    cudaFree(dev_bdst);
    cudaStreamDestroy(stream);
    return 1;
  }
  // Bulk oracle: the canonical chain over every rank's fill, computed
  // locally (each rank knows every rank's deterministic input).
  std::vector<uint16_t> bulk_want(bulk_elems, 0);
  {
    std::vector<std::vector<uint16_t>> all_src(static_cast<size_t>(world));
    for (int r = 0; r < world; ++r)
      fill_rank_bf16(&all_src[static_cast<size_t>(r)], bulk_elems, r);
    for (size_t i = 0; i < bulk_elems; ++i) {
      float acc = 0.0f;
      for (int r = 0; r < world; ++r)
        acc += dgpp::net::bf16_to_f32(all_src[static_cast<size_t>(r)][i]);
      bulk_want[i] = dgpp::net::bf16_from_f32_rne(acc);
    }
    // My own slice up to the device source.
    if (cudaMemcpyAsync(dev_bsrc, all_src[static_cast<size_t>(my_rank)].data(),
                        bulk_elems * 2, cudaMemcpyHostToDevice, stream) !=
        cudaSuccess) {
      DGPP_LOG_ERROR("rank {}: bulk H2D failed", my_rank);
      cudaFree(dev_src);
      cudaFree(dev_dst);
      cudaFree(warm_buf);
      cudaFree(dev_bsrc);
      cudaFree(dev_bdst);
      cudaStreamDestroy(stream);
      return 1;
    }
  }
  if (cudaMemcpyAsync(dev_src, host_src.data(), elems * 2,
                      cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemsetAsync(warm_buf, 0, 256 * 4, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    DGPP_LOG_ERROR("rank {}: graph H2D failed", my_rank);
    cudaFree(dev_src);
    cudaFree(dev_dst);
    cudaFree(warm_buf);
    cudaFree(dev_bsrc);
    cudaFree(dev_bdst);
    cudaStreamDestroy(stream);
    return 1;
  }

  int failures = 0;
  auto fail = [&](const std::string& why) {
    DGPP_LOG_ERROR("rank {}: {}", my_rank, why);
    ++failures;
  };

  // ---- eager baseline: the same fold before the era opens ----------------
  std::vector<uint16_t> eager_bytes(elems, 0);
  {
    std::string error;
    const uint64_t id = bus.allreduce(dev_src, dev_dst, elems, &error);
    if (id == 0) {
      fail("eager baseline rejected: " + error);
    } else {
      const dgpp::net::BusAllReduceResult r = bus.wait_allreduce(id, 30000);
      if (!r.ok) {
        fail("eager baseline failed: " + r.error);
      } else if (cudaMemcpyAsync(eager_bytes.data(), dev_dst, elems * 2,
                                 cudaMemcpyDeviceToHost, stream) !=
                     cudaSuccess ||
                 cudaStreamSynchronize(stream) != cudaSuccess) {
        fail("eager baseline D2H failed");
      }
    }
  }
  if (failures != 0) {
    cudaFree(dev_src);
    cudaFree(dev_dst);
    cudaFree(warm_buf);
    cudaFree(dev_bsrc);
    cudaFree(dev_bdst);
    cudaStreamDestroy(stream);
    return failures;
  }

  // ---- contract pins (rank 0, pre-era) -------------------------------------
  if (my_rank == 0) {
    std::string error;
    if (bus.allreduce_record(stream, dev_src, dev_dst, elems, &error) ||
        error.find("no open graph session") == std::string::npos) {
      fail("record without an open session not rejected: " + error);
    }
    if (bus.graph_replay_arm(&error) ||
        error.find("no recorded graph") == std::string::npos) {
      fail("arm without a recorded graph not rejected: " + error);
    }
  }

  // ---- capture the step ----------------------------------------------------
  // G collectives with dependent compute between them (the GEMM stand-in
  // keeps the clocks honest and forces the stream-order serialization the
  // walk's one-gen-at-a-time assumption is built on), plus trailing
  // compute — the decode step's shape.
  std::string error;
  if (!bus.graph_record_begin(&error)) {
    fail("graph_record_begin rejected: " + error);
  } else {
    if (my_rank == 0) {
      std::string pin_error;
      if (bus.graph_replay_arm(&pin_error) ||
          pin_error.find("no recorded graph") == std::string::npos)
        fail("arm during an open session not rejected: " + pin_error);
      // The eager gate while RECORDING: rejections are host-side (no CUDA
      // call, no registry entry), so they are capture-safe.
      if (bus.allreduce(dev_src, dev_dst, elems, &pin_error) != 0 ||
          pin_error.find("recording") == std::string::npos)
        fail("eager allreduce during recording not rejected: " + pin_error);
      if (bus.stage_next(&pin_error) != nullptr ||
          pin_error.find("recording") == std::string::npos)
        fail("stage_next during recording not rejected: " + pin_error);
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t cap =
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (cap != cudaSuccess) {
      fail("stream capture begin failed");
    } else {
      for (int g = 0; g < gens_per_step; ++g) {
        dgpp::net::launch_bus_warm_work(stream, warm_buf, 20000);
        if (!bus.allreduce_record(stream, dev_src, dev_dst, elems, &error)) {
          fail("allreduce_record rejected: " + error);
          break;
        }
      }
      dgpp::net::launch_bus_warm_work(stream, warm_buf, 20000);
    }
    if (cap == cudaSuccess) {
      if (cudaStreamEndCapture(stream, &graph) != cudaSuccess || !graph) {
        fail("stream capture end failed");
      }
    }
    if (failures == 0) {
      if (!bus.graph_record_end(&error)) {
        fail("graph_record_end rejected: " + error);
      }
    } else {
      // Close the half-open session so the bus state stays defined; the
      // run reports the failures above.
      std::string close_error;
      bus.graph_record_end(&close_error);
    }

    // ---- era pins ----------------------------------------------------------
    if (my_rank == 0 && failures == 0) {
      // Harness sends stay closed for the era's lifetime (coll_mode);
      // eager collectives are the ones that reopen between windows.
      std::string pin_error;
      std::vector<uint64_t> payload(8);
      if (bus.send(world == 2 ? 1 - my_rank : 1, payload.data(), 64,
                   BusMessageClass::kLatency, &pin_error) != 0 ||
          pin_error.find("collective mode") == std::string::npos)
        fail("send during the graph era not rejected: " + pin_error);
    }

    // ---- replays: warm 2, then measured ------------------------------------
    cudaGraphExec_t exec = nullptr;
    if (failures == 0) {
      if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) !=
          cudaSuccess) {
        fail("graph instantiate failed");
      }
    }
    // A second, shorter/narrower shape gets its own cell slab. Replaying it
    // once before variant 0 pins selection in both directions and, because
    // its generation count differs, catches using the newly selected G when
    // waiting for the previous window.
    cudaGraph_t alt_graph = nullptr;
    cudaGraphExec_t alt_exec = nullptr;
    const int alt_gens = gens_per_step - 3;
    const size_t alt_elems = std::max<size_t>(2, (elems / 2) & ~size_t{1});
    if (failures == 0) {
      if (!bus.graph_record_begin(&error, /*variant=*/1)) {
        fail("second graph variant record_begin rejected: " + error);
      } else {
        const cudaError_t cap = cudaStreamBeginCapture(
            stream, cudaStreamCaptureModeThreadLocal);
        if (cap != cudaSuccess) {
          fail("second graph variant capture begin failed");
        } else {
          for (int g = 0; g < alt_gens; ++g) {
            dgpp::net::launch_bus_warm_work(stream, warm_buf, 12000);
            if (!bus.allreduce_record(stream, dev_src, dev_dst, alt_elems,
                                      &error)) {
              fail("second graph variant allreduce_record rejected: " +
                   error);
              break;
            }
          }
          dgpp::net::launch_bus_warm_work(stream, warm_buf, 12000);
          if (cudaStreamEndCapture(stream, &alt_graph) != cudaSuccess ||
              alt_graph == nullptr)
            fail("second graph variant capture end failed");
        }
        if (failures == 0 && !bus.graph_record_end(&error))
          fail("second graph variant record_end rejected: " + error);
      }
      if (failures == 0 &&
          cudaGraphInstantiate(&alt_exec, alt_graph, nullptr, nullptr, 0) !=
              cudaSuccess)
        fail("second graph variant instantiate failed");
    }
    if (exec != nullptr && alt_exec != nullptr) {
      std::vector<double> step_us;
      std::vector<uint16_t> got(elems, 0);
      std::vector<uint16_t> bulk_got(bulk_elems, 0);
      const uint64_t completions_before = bus.completion_epoch().load();
      if (!bus.graph_replay_arm(&error, /*variant=*/1)) {
        fail("second graph variant arm rejected: " + error);
      } else if (cudaGraphLaunch(alt_exec, stream) != cudaSuccess ||
                 cudaStreamSynchronize(stream) != cudaSuccess) {
        fail("second graph variant replay failed");
      } else if (!bus.graph_replay_finish(30000, &error)) {
        fail("second graph variant finish rejected: " + error);
      } else if (cudaMemcpyAsync(got.data(), dev_dst, alt_elems * 2,
                                 cudaMemcpyDeviceToHost, stream) !=
                     cudaSuccess ||
                 cudaStreamSynchronize(stream) != cudaSuccess) {
        fail("second graph variant D2H failed");
      } else if (std::memcmp(got.data(), eager_bytes.data(), alt_elems * 2) !=
                 0) {
        fail("second graph variant diverged from eager bitwise");
      }
      if (failures == 0 && bus.completion_epoch().load() != completions_before + alt_gens)
        fail("graph completions did not advance progress once per generation");
      for (int replay = 0; replay < 2 + replays && failures == 0; ++replay) {
        if (!bus.graph_replay_arm(&error)) {
          fail("arm rejected: " + error);
          break;
        }
        // The eager gate while a window is ARMED (arm .. finish): rank 0
        // must be rejected loudly, mid-flight — the gate is host-side
        // and coll_mu-serialized, so pinning here is race-free.
        if (replay == 3 && my_rank == 0) {
          std::string pin_error;
          if (bus.allreduce(dev_src, dev_dst, elems, &pin_error) != 0 ||
              pin_error.find("armed") == std::string::npos)
            fail("eager allreduce during an armed window not rejected: " +
                 pin_error);
          if (bus.stage_next(&pin_error) != nullptr ||
              pin_error.find("armed") == std::string::npos)
            fail("stage_next during an armed window not rejected: " +
                 pin_error);
          if (bus.allreduce_bulk(dev_bsrc, dev_bdst, bulk_elems,
                                 &pin_error) != 0 ||
              pin_error.find("armed") == std::string::npos)
            fail("allreduce_bulk during an armed window not rejected: " +
                 pin_error);
        }
        const auto t0 = std::chrono::steady_clock::now();
        if (cudaGraphLaunch(exec, stream) != cudaSuccess) {
          fail("graph launch failed");
          break;
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
          fail("replay stream sync failed");
          break;
        }
        if (!bus.graph_replay_finish(30000, &error)) {
          fail("finish rejected: " + error);
          // TEMP bring-up: the per-gen cells as this rank's engine last
          // saw them — gen/ready/done/status localize a dead kernel vs a
          // blind engine instantly.
          bus.dump_graph_cells("finish");
          break;
        }
        const double us =
            std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0)
                .count();
        if (replay >= 2) step_us.push_back(us);
        // Bitwise: every replay against the oracle AND the eager bytes.
        if (cudaMemcpyAsync(got.data(), dev_dst, elems * 2,
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
          fail("replay D2H failed");
          break;
        }
        if (std::memcmp(got.data(), eager_bytes.data(), elems * 2) != 0) {
          fail("replay " + std::to_string(replay) +
               " diverged from the eager machine bitwise");
          break;
        }
        if (std::memcmp(got.data(), want->data(), elems * 2) != 0) {
          fail("replay " + std::to_string(replay) +
               " diverged from the canonical-chain oracle");
          break;
        }
        // ---- the mixed-era interlude (eager between windows) ------------
        // After the warmups: eager collectives run between replay windows,
        // consuming generations from the shared counter exactly as the
        // serving process interleaves prefill folds and the pick between
        // decode steps. Every LATER window is the proof — it must adopt
        // around the consumed numbers and stay bitwise.
        if (replay == 2) {
          std::string eager_error;
          // (a) the pick's class: a latency one-shot.
          {
            const uint64_t id =
                bus.allreduce(dev_src, dev_dst, elems, &eager_error);
            if (id == 0) {
              fail("mixed-era eager allreduce rejected: " + eager_error);
            } else {
              const dgpp::net::BusAllReduceResult r =
                  bus.wait_allreduce(id, 30000);
              if (!r.ok)
                fail("mixed-era eager allreduce failed: " + r.error);
            }
          }
          // (b) the prefill interface: a staged handout, folded in place.
          {
            void* handout = bus.stage_next(&eager_error);
            if (handout == nullptr) {
              fail("mixed-era stage_next rejected: " + eager_error);
            } else {
              std::memcpy(handout, host_src.data(), elems * 2);
              const uint64_t id = bus.allreduce_staged(elems, &eager_error);
              if (id == 0) {
                fail("mixed-era allreduce_staged rejected: " + eager_error);
              } else {
                const dgpp::net::BusAllReduceResult r =
                    bus.wait_allreduce(id, 30000);
                if (!r.ok) {
                  fail("mixed-era allreduce_staged failed: " + r.error);
                } else if (std::memcmp(handout, eager_bytes.data(),
                                       elems * 2) != 0) {
                  fail("mixed-era staged fold diverged bitwise");
                }
              }
            }
          }
          // (c) the prefill folds' class: a bulk machine run.
          {
            const uint64_t id = bus.allreduce_bulk(dev_bsrc, dev_bdst,
                                                   bulk_elems, &eager_error);
            if (id == 0) {
              fail("mixed-era allreduce_bulk rejected: " + eager_error);
            } else {
              const dgpp::net::BusAllReduceResult r =
                  bus.wait_allreduce(id, 30000);
              if (!r.ok) {
                fail("mixed-era allreduce_bulk failed: " + r.error);
              } else if (
                  cudaMemcpyAsync(bulk_got.data(), dev_bdst, bulk_elems * 2,
                                  cudaMemcpyDeviceToHost, stream) !=
                      cudaSuccess ||
                  cudaStreamSynchronize(stream) != cudaSuccess) {
                fail("mixed-era bulk D2H failed");
              } else if (std::memcmp(bulk_got.data(), bulk_want.data(),
                                     bulk_elems * 2) != 0) {
                fail("mixed-era bulk fold diverged bitwise");
              }
            }
          }
        }
      }
      if (failures == 0) {
        DGPP_LOG_INFO(
            "rank {}: {} replays x {} gens clean (eager interlude after "
            "warmup); step p50={:.1f}us ({:.1f}us per collective)",
            my_rank, replays, gens_per_step, percentile(step_us, 0.5),
            percentile(step_us, 0.5) / gens_per_step);
      }
    }
    // ---- the pipelined replay: two windows armed at once --
    // The engine arms the NEXT replay's window while the previous replay
    // still runs, on the other variant. Variant 1 then variant 0, both
    // launched back to back on the stream before either is finished; a
    // third arm is rejected (two windows live), an arm of a live variant
    // is rejected, the eager gate stays closed, and the finishes retire
    // the windows in arm order. Bitwise against the eager bytes every
    // round.
    if (exec != nullptr && alt_exec != nullptr && failures == 0) {
      std::vector<uint16_t> got(elems, 0);
      for (int round = 0; round < 6 && failures == 0; ++round) {
        if (!bus.graph_replay_arm(&error, /*variant=*/1)) {
          fail("pipelined arm (variant 1) rejected: " + error);
          break;
        }
        if (cudaGraphLaunch(alt_exec, stream) != cudaSuccess) {
          fail("pipelined launch (variant 1) failed");
          break;
        }
        if (!bus.graph_replay_arm(&error, /*variant=*/0)) {
          fail("pipelined arm (variant 0) while variant 1 is live rejected: " +
               error);
          break;
        }
        if (cudaGraphLaunch(exec, stream) != cudaSuccess) {
          fail("pipelined launch (variant 0) failed");
          break;
        }
        if (round == 0 && my_rank == 0) {
          std::string pin_error;
          if (bus.graph_replay_arm(&pin_error, /*variant=*/1) ||
              pin_error.find("already armed") == std::string::npos)
            fail("a third window was not rejected: " + pin_error);
          if (bus.allreduce(dev_src, dev_dst, elems, &pin_error) != 0 ||
              pin_error.find("armed") == std::string::npos)
            fail("eager allreduce with two windows armed not rejected: " +
                 pin_error);
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
          fail("pipelined replay stream sync failed");
          break;
        }
        if (!bus.graph_replay_finish(30000, &error)) {
          fail("pipelined finish (variant 1's window) rejected: " + error);
          bus.dump_graph_cells("pipelined finish 1");
          break;
        }
        if (round == 0 && my_rank == 0) {
          // One window still live: variant 0 cannot be armed again, the
          // gate is still closed.
          std::string pin_error;
          if (bus.graph_replay_arm(&pin_error, /*variant=*/0) ||
              pin_error.find("live replay window") == std::string::npos)
            fail("arming the live variant was not rejected: " + pin_error);
        }
        if (!bus.graph_replay_finish(30000, &error)) {
          fail("pipelined finish (variant 0's window) rejected: " + error);
          bus.dump_graph_cells("pipelined finish 0");
          break;
        }
        if (cudaMemcpyAsync(got.data(), dev_dst, elems * 2,
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
          fail("pipelined replay D2H failed");
          break;
        }
        if (std::memcmp(got.data(), eager_bytes.data(), elems * 2) != 0) {
          fail("pipelined round " + std::to_string(round) +
               " diverged from the eager machine bitwise");
          break;
        }
      }
      if (failures == 0) {
        std::string pin_error;
        if (!bus.graph_replay_finish(30000, &pin_error) &&
            pin_error.find("no armed window") == std::string::npos)
          fail("a finish with no armed window reported: " + pin_error);
        else if (pin_error.empty())
          fail("a finish with no armed window succeeded");
        DGPP_LOG_INFO("rank {}: 6 pipelined rounds (two windows armed at "
                      "once) clean and bitwise",
                      my_rank);
      }
    }
    if (exec != nullptr) cudaGraphExecDestroy(exec);
    if (alt_exec != nullptr) cudaGraphExecDestroy(alt_exec);
    if (alt_graph) cudaGraphDestroy(alt_graph);
    if (graph) cudaGraphDestroy(graph);
  }

  cudaFree(dev_src);
  cudaFree(dev_dst);
  cudaFree(warm_buf);
  cudaFree(dev_bsrc);
  cudaFree(dev_bdst);
  cudaStreamDestroy(stream);
  return failures;
}

void scenario_allreduce_graph() {
  // The decode step (§6.2): capture the fixed launch sequence once,
  // replay it per step. Pins: bitwise vs the oracle and the eager
  // machine across replays and ring generations, the contract pins
  // (record/arm/eager/send rejections), and the walk's determinism
  // across worlds.
  const size_t elems = 4096;  // 8 KiB, the decode unit
  int failures = 0;
  // Above the staging-ring depth (8) on purpose: intra-window row reuse
  // is the walk's serialized-reuse discipline, and the decode graph is
  // 90 nodes deep in it.
  const int gens_per_step = 12;
  const int replays = 8;

  for (const int world : {2, 4}) {
    const uint16_t port = world == 2 ? 29906 : 29907;
    std::vector<std::unique_ptr<CollectiveBus>> world_buses =
        start_world(world, port);
    if (world_buses.empty()) {
      DGPP_LOG_ERROR("graph world {} failed to start", world);
      ++failures;
      continue;
    }
    std::vector<std::vector<uint16_t>> all_src(static_cast<size_t>(world));
    for (int r = 0; r < world; ++r)
      fill_rank_bf16(&all_src[static_cast<size_t>(r)], elems, r);
    std::vector<uint16_t> want(elems, 0);
    for (size_t i = 0; i < elems; ++i) {
      float acc = 0.0f;
      for (int r = 0; r < world; ++r)
        acc += dgpp::net::bf16_to_f32(all_src[static_cast<size_t>(r)][i]);
      want[i] = dgpp::net::bf16_from_f32_rne(acc);
    }
    std::vector<std::thread> workers;
    std::vector<int> rank_failures(static_cast<size_t>(world), 0);
    for (int r = 0; r < world; ++r)
      workers.emplace_back([&, r] {
        rank_failures[static_cast<size_t>(r)] = allreduce_graph_rank_work(
            *world_buses[static_cast<size_t>(r)], world, r, elems,
            gens_per_step, replays, &want);
      });
    for (auto& t : workers) t.join();
    int world_failures = 0;
    for (int r = 0; r < world; ++r) {
      if (rank_failures[static_cast<size_t>(r)] != 0)
        DGPP_LOG_ERROR("graph world {} rank {} reported failures", world, r);
      world_failures += rank_failures[static_cast<size_t>(r)];
    }
    CHECK(world_failures == 0,
          "graph world " + std::to_string(world) + " had " +
              std::to_string(world_failures) + " failures");
    for (auto& bus : world_buses) bus->quiesce();
    for (auto& bus : world_buses) bus->stop();
    DGPP_LOG_INFO("scenario allreduce_graph ({}): world {} clean", transport_name(), world);
  }

  g_failures += failures;
  DGPP_LOG_INFO("scenario allreduce_graph: {} total failures", g_failures);
}

void scenario_graph_snapshot_hash() {
  // Independent CPU hash oracle, including u32 tails, non-vector-aligned
  // sources, staging-ring wrap, and all three graph block sizes. No fabric
  // is needed to exercise the producer's snapshot/publication contract.
  using dgpp::net::BusAllReduceCtl;
  using dgpp::net::BusAllReduceGraphView;
  for (const uint32_t elems : {2u, 4u, 6u, 8u, 510u, 5120u, 20480u, 40960u, 65536u}) {
    for (const uint32_t offset : {0u, 2u}) {
      uint16_t* device = nullptr;
      uint16_t* stage = nullptr;
      BusAllReduceCtl* cell = nullptr;
      const uint32_t row_bytes = (elems * 2 + 63) / 64 * 64;
      if (cudaMalloc(&device, (elems + offset) * 2) != cudaSuccess ||
          cudaMallocHost(&stage, row_bytes * 2) != cudaSuccess ||
          cudaMallocHost(&cell, sizeof(*cell)) != cudaSuccess) {
        CHECK(false, "snapshot hash allocation failed");
        cudaFree(device);
        cudaFreeHost(stage);
        cudaFreeHost(cell);
        return;
      }
      BusAllReduceGraphView view{};
      view.stage_row_base = stage;
      view.stage_ring = 2;
      view.stage_row_bytes = row_bytes;
      for (uint64_t gen = 1; gen <= 3; ++gen) {
        std::vector<uint16_t> source(elems);
        for (uint32_t i = 0; i < elems; ++i)
          source[i] = static_cast<uint16_t>(0x3800 + ((i * 73 + gen * 101) & 0x7ff));
        std::vector<uint64_t> words((elems + 3) / 4, 0);
        if (!words.empty())
          std::memcpy(words.data(), source.data(), elems * sizeof(uint16_t));
        const uint64_t want = dgpp::net::bus_fold64(words.data(), words.size());
        CHECK(dgpp::net::bus_fold(source.data(), elems * 2) == want, "CPU hash omits payload tail");
        *cell = {};
        cell->gen_seq = gen;
        cell->stage_hash = ~want;
        CHECK(cudaMemcpy(device + offset, source.data(), elems * 2, cudaMemcpyHostToDevice) ==
                  cudaSuccess,
              "snapshot hash source upload failed");
        auto* src = reinterpret_cast<__nv_bfloat16*>(device + offset);
        CHECK(dgpp::net::launch_bus_allreduce_graph(view, 0, src, src, elems, cell,
                                                    dgpp::net::bus_consumer_deadline_cycles(5),
                                                    nullptr) == cudaSuccess,
              "snapshot hash kernel launch failed");
        CHECK(cudaDeviceSynchronize() == cudaSuccess, "snapshot hash kernel failed");
        CHECK(cell->done_seq == gen && cell->status == 0, "snapshot hash did not complete");
        CHECK(cell->stage_hash == want, "snapshot hash differs from CPU bus_fold64");
        CHECK(std::memcmp(stage + ((gen - 1) % 2) * row_bytes / 2, source.data(), elems * 2) == 0,
              "snapshot bytes differ from source");
      }
      cudaFree(device);
      cudaFreeHost(stage);
      cudaFreeHost(cell);
    }
  }
  DGPP_LOG_INFO("scenario graph_snapshot_hash: {} total failures", g_failures);
}

void scenario_allreduce_graph_rows() {
  // Staged and unstaged folds, both sides of the wider-block thresholds,
  // a non-vector-aligned row, and DeepSeek's 30-row hidden/Engram shapes.
  // Every destination must match the independent rank-order oracle
  // bitwise across replays and mixed eager/graph generations.
  constexpr int world = 4;
  const int gens_per_step = 12;
  const int replays = 4;
  const size_t rows_elems[] = {2, 6, 510, 6144, 12288, 16380, 16384, 16388, 18432,
                               49152, 65532, 65536, 65540, 153600, 768000};
  const uint16_t ports[] = {29909, 29910, 29911, 29891, 29892, 29893, 29894, 29895, 29896,
                            29897, 29898, 29899, 29900, 29901, 29902};
  int failures = 0;
  for (size_t c = 0; c < std::size(rows_elems); ++c) {
    const size_t elems = rows_elems[c];
    // The allocation has 64-byte geometry even for a ragged payload.
    const size_t lat_slot_bytes = std::max<size_t>(98304, (elems * 2 + 63) / 64 * 64);
    std::vector<std::unique_ptr<CollectiveBus>> world_buses =
        start_world(world, ports[c], /*one_lane=*/false, /*rank0_pass_delay_us=*/0,
                    lat_slot_bytes);
    if (world_buses.empty()) {
      DGPP_LOG_ERROR("graph rows world (elems {}) failed to start", elems);
      ++failures;
      continue;
    }
    std::vector<std::vector<uint16_t>> all_src(static_cast<size_t>(world));
    for (int r = 0; r < world; ++r)
      fill_rank_bf16(&all_src[static_cast<size_t>(r)], elems, r);
    std::vector<uint16_t> want(elems, 0);
    for (size_t i = 0; i < elems; ++i) {
      float acc = 0.0f;
      for (int r = 0; r < world; ++r)
        acc += dgpp::net::bf16_to_f32(all_src[static_cast<size_t>(r)][i]);
      want[i] = dgpp::net::bf16_from_f32_rne(acc);
    }
    std::vector<std::thread> workers;
    std::vector<int> rank_failures(static_cast<size_t>(world), 0);
    for (int r = 0; r < world; ++r)
      workers.emplace_back([&, r] {
        rank_failures[static_cast<size_t>(r)] = allreduce_graph_rank_work(
            *world_buses[static_cast<size_t>(r)], world, r, elems,
            gens_per_step, replays, &want);
      });
    for (auto& t : workers) t.join();
    int world_failures = 0;
    for (int r = 0; r < world; ++r) world_failures += rank_failures[static_cast<size_t>(r)];
    CHECK(world_failures == 0,
          "graph rows world (elems " + std::to_string(elems) + ") had " +
              std::to_string(world_failures) + " failures");
    for (auto& bus : world_buses) bus->quiesce();
    for (auto& bus : world_buses) bus->stop();
    DGPP_LOG_INFO("scenario allreduce_graph_rows ({}): elems {} ({} KB per row) clean", transport_name(), elems,
                  elems * 2 / 1024);
  }
  g_failures += failures;
  DGPP_LOG_INFO("scenario allreduce_graph_rows: {} total failures", g_failures);
}

// The stream collectives (2026-09-14, plan D9): the eager fold launched on
// the caller's stream in the graph kernel form. Pins: every generation's
// destination bitwise the oracle and the eager one-shot (the same chain),
// generations far beyond the staging ring and the stream ring's depth in
// one pass, the gates (a host-driven collective and a handout rejected
// while stream generations are outstanding; a stream issue rejected while
// a replay window is armed), and the mixed era (stream passes between
// replay windows, the shared counter keeping the order).
int allreduce_stream_rank_work(CollectiveBus& bus, int world, int my_rank,
                               size_t elems, int gens, int passes) {
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
    DGPP_LOG_ERROR("rank {}: stream create failed", my_rank);
    return 1;
  }
  int failures = 0;
  auto fail = [&](const std::string& why) {
    DGPP_LOG_ERROR("rank {}: {}", my_rank, why);
    ++failures;
  };
  // Sources: generation g folds fill(rank) + g (the oracle knows every rank's fill).
  std::vector<uint16_t> base;
  fill_rank_bf16(&base, elems, my_rank);
  std::vector<std::vector<uint16_t>> want(static_cast<size_t>(gens));
  {
    std::vector<std::vector<uint16_t>> all(static_cast<size_t>(world));
    for (int r = 0; r < world; ++r) fill_rank_bf16(&all[static_cast<size_t>(r)], elems, r);
    for (int g = 0; g < gens; ++g) {
      want[static_cast<size_t>(g)].assign(elems, 0);
      for (size_t i = 0; i < elems; ++i) {
        float acc = 0.0f;
        for (int r = 0; r < world; ++r)
          acc += dgpp::net::bf16_to_f32(dgpp::net::bf16_from_f32_rne(
              dgpp::net::bf16_to_f32(all[static_cast<size_t>(r)][i]) + static_cast<float>(g)));
        want[static_cast<size_t>(g)][i] = dgpp::net::bf16_from_f32_rne(acc);
      }
    }
  }
  std::vector<uint16_t*> dev(static_cast<size_t>(gens), nullptr);
  for (int g = 0; g < gens; ++g) {
    std::vector<uint16_t> src(elems);
    for (size_t i = 0; i < elems; ++i)
      src[i] = dgpp::net::bf16_from_f32_rne(dgpp::net::bf16_to_f32(base[i]) + static_cast<float>(g));
    if (cudaMalloc(&dev[static_cast<size_t>(g)], elems * 2) != cudaSuccess ||
        cudaMemcpy(dev[static_cast<size_t>(g)], src.data(), elems * 2, cudaMemcpyHostToDevice) != cudaSuccess) {
      fail("device alloc/H2D failed");
      break;
    }
  }
  std::vector<uint16_t> got(elems);
  // The oracle for generation g: the canonical chain over every rank's
  // bf16(fill + g) — the source each rank uploads, rounded the same way.
  for (int pass = 0; pass < passes && failures == 0; ++pass) {
    const uint64_t completions_before = bus.completion_epoch().load();
    // Re-upload (the folds ran in place).
    for (int g = 0; g < gens; ++g) {
      std::vector<uint16_t> src(elems);
      for (size_t i = 0; i < elems; ++i)
        src[i] = dgpp::net::bf16_from_f32_rne(dgpp::net::bf16_to_f32(base[i]) + static_cast<float>(g));
      if (cudaMemcpyAsync(dev[static_cast<size_t>(g)], src.data(), elems * 2, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        fail("H2D failed");
        break;
      }
    }
    if (failures) break;
    std::string error;
    for (int g = 0; g < gens; ++g) {
      if (bus.allreduce_stream(stream, dev[static_cast<size_t>(g)], dev[static_cast<size_t>(g)], elems, &error) == 0) {
        fail("stream collective " + std::to_string(g) + " rejected: " + error);
        break;
      }
      if (g == 0 && pass == 0) {
        // The gates while a stream generation is outstanding.
        uint16_t* dummy = dev[0];
        if (bus.allreduce(dummy, dummy, elems, &error) != 0 || error.find("stream collectives are in flight") == std::string::npos)
          fail("a host-driven collective was not rejected with stream generations outstanding: " + error);
        if (bus.stage_next(&error) != nullptr || error.find("stream collectives are in flight") == std::string::npos)
          fail("a handout was not rejected with stream generations outstanding: " + error);
      }
    }
    if (failures) break;
    if (!bus.allreduce_settle(30000, &error)) {
      fail("settle failed: " + error);
      break;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      fail("stream sync failed");
      break;
    }
    if (bus.completion_epoch().load() != completions_before + gens) {
      fail("stream completions did not advance progress once per generation");
      break;
    }
    for (int g = 0; g < gens; ++g) {
      if (cudaMemcpy(got.data(), dev[static_cast<size_t>(g)], elems * 2, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fail("D2H failed");
        break;
      }
      if (got != want[static_cast<size_t>(g)]) {
        size_t first = 0;
        while (first < elems && got[first] == want[static_cast<size_t>(g)][first]) ++first;
        fail("pass " + std::to_string(pass) + " generation " + std::to_string(g) + " differs from the oracle at element " +
             std::to_string(first));
        break;
      }
    }
    // Between passes: a host-driven one-shot runs again (the gate reopened).
    {
      std::vector<uint16_t> src(elems);
      for (size_t i = 0; i < elems; ++i) src[i] = base[i];
      if (cudaMemcpy(dev[0], src.data(), elems * 2, cudaMemcpyHostToDevice) != cudaSuccess) { fail("H2D failed"); break; }
      const uint64_t id = bus.allreduce(dev[0], dev[0], elems, &error);
      if (id == 0) { fail("eager one-shot after settle rejected: " + error); break; }
      const dgpp::net::BusAllReduceResult r = bus.wait_allreduce(id, 30000);
      if (!r.ok) { fail("eager one-shot after settle failed: " + r.error); break; }
      if (cudaMemcpy(got.data(), dev[0], elems * 2, cudaMemcpyDeviceToHost) != cudaSuccess) { fail("D2H failed"); break; }
      if (got != want[0]) { fail("the eager one-shot after settle differs from the oracle"); break; }
    }
  }
  for (uint16_t* d : dev) cudaFree(d);
  cudaStreamDestroy(stream);
  return failures;
}

void scenario_allreduce_stream() {
  const size_t elems = 4096;
  // Beyond the staging ring (8) and the stream ring (64) in one pass.
  const int gens = 100;
  const int passes = 3;
  int failures = 0;
  for (const int world : {2, 4}) {
    const uint16_t port = world == 2 ? 29896 : 29897;
    std::vector<std::unique_ptr<CollectiveBus>> world_buses = start_world(world, port);
    if (world_buses.empty()) {
      DGPP_LOG_ERROR("stream world {} failed to start", world);
      ++failures;
      continue;
    }
    std::vector<std::thread> workers;
    std::vector<int> rank_failures(static_cast<size_t>(world), 0);
    for (int r = 0; r < world; ++r)
      workers.emplace_back([&, r] {
        rank_failures[static_cast<size_t>(r)] =
            allreduce_stream_rank_work(*world_buses[static_cast<size_t>(r)], world, r, elems, gens, passes);
      });
    for (auto& t : workers) t.join();
    int world_failures = 0;
    for (int r = 0; r < world; ++r) world_failures += rank_failures[static_cast<size_t>(r)];
    CHECK(world_failures == 0, "stream world " + std::to_string(world) + " had " + std::to_string(world_failures) + " failures");
    for (auto& bus : world_buses) bus->quiesce();
    for (auto& bus : world_buses) bus->stop();
    DGPP_LOG_INFO("scenario allreduce_stream ({}): world {} clean ({} gens x {} passes)", transport_name(), world, gens, passes);
  }
  g_failures += failures;
}

// ---- quiet rounds ----------------------------------------------------------
//
// Once a decode graph is recorded, an engine with nothing to do parks on its
// kick word instead of polling (where the CPU parks; collective_bus.cpp,
// engine_loop). Every hand-off must end that park: a window arm, an eager
// collective, a stream generation. The rounds below hand work to engines in
// every state. A gap of a few hundred microseconds finds the engine still
// polling, a gap of a few milliseconds finds it about to park or just
// parked, and a long gap finds it parked for good. In the first half every
// rank starts a round at the same instant, which makes a round's duration a
// measurement of what the park costs. In the second half each rank adds a
// delay of its own, so one rank's window is live while its peers are still
// parked.
//
// On the write transport the engine also parks while a generation is in
// flight, on the block record its kernel stores next. The same rounds cover
// that park: a window's and a stream collective's generations each wait for
// the kernel twice, and in the second half a rank's kernel waits on peers
// that have not started. The row length sets how many blocks a generation
// has, and so how many records the engine parks on in turn.

enum class QuietKind { kWindow, kEager, kStream };

struct QuietRound {
  QuietKind kind = QuietKind::kWindow;
  double gap_us = 0;  // quiet time before the round, the same on every rank
};

struct QuietShared {
  int world = 0;
  std::vector<QuietRound> rounds;
  int aligned_rounds = 0;  // the first rounds start on every rank at one instant
  std::atomic<int> finished{0};  // rounds finished, summed over the ranks
  std::vector<std::atomic<int64_t>> finish_ns;  // per rank: when its last round ended
  std::atomic<bool> abort{false};
  // Every rank stops before this round until `resume` is set, so that the
  // scenario can read the buses while no collective is in flight (-1: no
  // pause). `paused` counts the ranks that have arrived.
  int pause_round = -1;
  std::atomic<int> paused{0};
  std::atomic<bool> resume{false};
  explicit QuietShared(int w) : world(w), finish_ns(static_cast<size_t>(w)) {}
};

struct QuietRankBuffers {
  cudaStream_t stream = nullptr;
  uint16_t* dev_src = nullptr;
  uint16_t* dev_dst = nullptr;
  float* warm_buf = nullptr;
};

// One sample of the aligned half: the quiet time before a window and how
// long the window took from arm to finish.
struct QuietSample {
  double gap_us = 0;
  double window_us = 0;
};

int64_t quiet_now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::vector<QuietRound> quiet_schedule(int rounds, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<QuietRound> out(static_cast<size_t>(rounds));
  for (QuietRound& round : out) {
    const uint64_t kind = rng() % 10;
    round.kind = kind < 6 ? QuietKind::kWindow : kind < 8 ? QuietKind::kEager : QuietKind::kStream;
    const uint64_t gap_class = rng() % 100;
    const double unit = static_cast<double>(rng() % 1000000) / 1e6;
    if (gap_class < 30) round.gap_us = 0;                               // back to back
    else if (gap_class < 45) round.gap_us = 300.0 * unit;               // still polling
    else if (gap_class < 75) round.gap_us = 300.0 + 5700.0 * unit;      // about to park, or just parked
    else if (gap_class < 97) round.gap_us = 10000.0 + 30000.0 * unit;   // parked
    else round.gap_us = 100000.0 + 150000.0 * unit;                     // parked for long
  }
  return out;
}

// One rank's share of the quiet rounds. The device buffers come from the
// scenario: an allocation in a rank's thread would wait on a peer's
// spinning collective kernel.
int quiet_rank_work(CollectiveBus& bus, QuietShared& shared, int my_rank, size_t elems,
                    int gens_per_window, uint64_t seed, const QuietRankBuffers& buf,
                    const std::vector<uint16_t>* want, std::vector<QuietSample>* samples) {
  int failures = 0;
  auto fail = [&](const std::string& why) {
    DGPP_LOG_ERROR("rank {}: {}", my_rank, why);
    ++failures;
    shared.abort.store(true, std::memory_order_release);
  };
  std::vector<uint16_t> host_src;
  fill_rank_bf16(&host_src, elems, my_rank);
  if (cudaMemcpyAsync(buf.dev_src, host_src.data(), elems * 2, cudaMemcpyHostToDevice,
                      buf.stream) != cudaSuccess ||
      cudaMemsetAsync(buf.warm_buf, 0, 256 * 4, buf.stream) != cudaSuccess ||
      cudaStreamSynchronize(buf.stream) != cudaSuccess) {
    fail("quiet rounds H2D failed");
    return failures;
  }

  // Record one window. From here on the engine parks when it is quiet.
  std::string error;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  if (!bus.graph_record_begin(&error)) {
    fail("graph_record_begin rejected: " + error);
  } else {
    if (cudaStreamBeginCapture(buf.stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
      fail("stream capture begin failed");
    } else {
      for (int g = 0; g < gens_per_window; ++g) {
        dgpp::net::launch_bus_warm_work(buf.stream, buf.warm_buf, 4000);
        if (!bus.allreduce_record(buf.stream, buf.dev_src, buf.dev_dst, elems, &error)) {
          fail("allreduce_record rejected: " + error);
          break;
        }
      }
      dgpp::net::launch_bus_warm_work(buf.stream, buf.warm_buf, 4000);
      if (cudaStreamEndCapture(buf.stream, &graph) != cudaSuccess || graph == nullptr)
        fail("stream capture end failed");
    }
    std::string end_error;
    if (!bus.graph_record_end(&end_error) && failures == 0)
      fail("graph_record_end rejected: " + end_error);
  }
  if (failures == 0 && cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess)
    fail("graph instantiate failed");

  // This rank's own delays in the second half.
  std::mt19937_64 own(seed * 0x9E3779B97F4A7C15ull + static_cast<uint64_t>(my_rank) + 1);
  std::vector<uint16_t> got(elems, 0);
  const int total = static_cast<int>(shared.rounds.size());
  for (int r = 0; r < total && failures == 0; ++r) {
    const QuietRound& round = shared.rounds[static_cast<size_t>(r)];
    // The round starts gap_us after the last rank finished the round before.
    while (shared.finished.load(std::memory_order_acquire) < r * shared.world) {
      if (shared.abort.load(std::memory_order_acquire)) break;
      dgpp::cpu_wait::spin_hint();
    }
    if (shared.abort.load(std::memory_order_acquire)) break;
    if (r == shared.pause_round) {
      shared.paused.fetch_add(1, std::memory_order_release);
      while (!shared.resume.load(std::memory_order_acquire) && !shared.abort.load(std::memory_order_acquire))
        std::this_thread::sleep_for(std::chrono::microseconds(200));
      if (shared.abort.load(std::memory_order_acquire)) break;
    }
    int64_t start = 0;
    for (const auto& f : shared.finish_ns) start = std::max(start, f.load(std::memory_order_relaxed));
    if (start == 0) start = quiet_now_ns();
    start += static_cast<int64_t>(round.gap_us * 1000.0) + 100000;  // 100 us for every rank to get here
    if (r >= shared.aligned_rounds) start += static_cast<int64_t>(own() % 3000000);
    // Sleep through most of a long gap, as an idle forward thread does, and
    // poll the clock for the rest so the ranks start together.
    const int64_t ahead = start - quiet_now_ns();
    if (ahead > 2000000)
      std::this_thread::sleep_for(std::chrono::nanoseconds(ahead - 1000000));
    while (quiet_now_ns() < start) dgpp::cpu_wait::spin_hint();

    const auto t0 = std::chrono::steady_clock::now();
    if (round.kind == QuietKind::kWindow) {
      if (!bus.graph_replay_arm(&error)) {
        fail("round " + std::to_string(r) + ": arm rejected: " + error);
        break;
      }
      if (cudaGraphLaunch(exec, buf.stream) != cudaSuccess ||
          cudaStreamSynchronize(buf.stream) != cudaSuccess) {
        fail("round " + std::to_string(r) + ": replay failed");
        break;
      }
      if (!bus.graph_replay_finish(30000, &error)) {
        fail("round " + std::to_string(r) + ": finish rejected: " + error);
        bus.dump_graph_cells("quiet round finish");
        break;
      }
    } else if (round.kind == QuietKind::kEager) {
      const uint64_t id = bus.allreduce(buf.dev_src, buf.dev_dst, elems, &error);
      if (id == 0) {
        fail("round " + std::to_string(r) + ": eager collective rejected: " + error);
        break;
      }
      const dgpp::net::BusAllReduceResult result = bus.wait_allreduce(id, 30000);
      if (!result.ok) {
        fail("round " + std::to_string(r) + ": eager collective failed: " + result.error);
        break;
      }
    } else {
      if (bus.allreduce_stream(buf.stream, buf.dev_src, buf.dev_dst, elems, &error) == 0) {
        fail("round " + std::to_string(r) + ": stream collective rejected: " + error);
        break;
      }
      if (!bus.allreduce_settle(30000, &error)) {
        fail("round " + std::to_string(r) + ": settle failed: " + error);
        break;
      }
      if (cudaStreamSynchronize(buf.stream) != cudaSuccess) {
        fail("round " + std::to_string(r) + ": stream sync failed");
        break;
      }
    }
    const double us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    if (cudaMemcpyAsync(got.data(), buf.dev_dst, elems * 2, cudaMemcpyDeviceToHost, buf.stream) !=
            cudaSuccess ||
        cudaStreamSynchronize(buf.stream) != cudaSuccess) {
      fail("round " + std::to_string(r) + ": D2H failed");
      break;
    }
    if (std::memcmp(got.data(), want->data(), elems * 2) != 0) {
      fail("round " + std::to_string(r) + " diverged from the canonical-chain oracle");
      break;
    }
    if (round.kind == QuietKind::kWindow && r < shared.aligned_rounds)
      samples->push_back({round.gap_us, us});
    shared.finish_ns[static_cast<size_t>(my_rank)].store(quiet_now_ns(), std::memory_order_relaxed);
    shared.finished.fetch_add(1, std::memory_order_release);
  }
  if (exec != nullptr) cudaGraphExecDestroy(exec);
  if (graph != nullptr) cudaGraphDestroy(graph);
  return failures;
}

// The shape of the quiet rounds' collectives. `decoy` puts rank 0's engine
// on the fault injection of the flight park (it parks beside the records),
// so its breaker turns the flight parks off somewhere in the schedule.
struct QuietShape {
  size_t elems = 4096;
  int gens_per_window = 4;
  bool decoy = false;
};

// `rounds` quiet rounds in one world, half aligned and half with each
// rank's own delay. Returns the failures.
int quiet_rounds_world(int world, uint16_t port, int rounds, uint64_t seed, const QuietShape& shape) {
  const size_t elems = shape.elems;
  const int gens_per_window = shape.gens_per_window;
  // The suite's 4096-element rows fit the default slot. A longer row takes
  // the row scenario's geometry.
  const size_t lat_slot_bytes =
      elems <= 4096 ? 8192 : std::max<size_t>(98304, (elems * 2 + 63) / 64 * 64);
  std::vector<std::unique_ptr<CollectiveBus>> buses =
      start_world(world, port, /*one_lane=*/false, /*rank0_pass_delay_us=*/0, lat_slot_bytes, shape.decoy);
  if (buses.empty()) {
    DGPP_LOG_ERROR("quiet rounds: world {} failed to start", world);
    return 1;
  }
  std::vector<std::vector<uint16_t>> all_src(static_cast<size_t>(world));
  for (int r = 0; r < world; ++r) fill_rank_bf16(&all_src[static_cast<size_t>(r)], elems, r);
  std::vector<uint16_t> want(elems, 0);
  for (size_t i = 0; i < elems; ++i) {
    float acc = 0.0f;
    for (int r = 0; r < world; ++r) acc += dgpp::net::bf16_to_f32(all_src[static_cast<size_t>(r)][i]);
    want[i] = dgpp::net::bf16_from_f32_rne(acc);
  }
  int failures = 0;
  std::vector<QuietRankBuffers> bufs(static_cast<size_t>(world));
  for (QuietRankBuffers& b : bufs) {
    if (cudaStreamCreateWithFlags(&b.stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaMalloc(&b.dev_src, elems * 2) != cudaSuccess ||
        cudaMalloc(&b.dev_dst, elems * 2) != cudaSuccess ||
        cudaMalloc(&b.warm_buf, 256 * 4) != cudaSuccess) {
      DGPP_LOG_ERROR("quiet rounds: device allocation failed");
      ++failures;
    }
  }
  if (failures == 0) {
    QuietShared shared(world);
    shared.rounds = quiet_schedule(rounds, seed);
    shared.aligned_rounds = rounds / 2;
    std::vector<int> rank_failures(static_cast<size_t>(world), 0);
    std::vector<std::vector<QuietSample>> samples(static_cast<size_t>(world));
    std::vector<std::thread> workers;
    for (int r = 0; r < world; ++r)
      workers.emplace_back([&, r] {
        rank_failures[static_cast<size_t>(r)] = quiet_rank_work(
            *buses[static_cast<size_t>(r)], shared, r, elems, gens_per_window, seed,
            bufs[static_cast<size_t>(r)], &want, &samples[static_cast<size_t>(r)]);
      });
    for (auto& t : workers) t.join();
    for (int r = 0; r < world; ++r) failures += rank_failures[static_cast<size_t>(r)];

    // Every engine parked where the CPU parks, and none did elsewhere.
    for (int r = 0; r < world; ++r) {
      const uint64_t parks = buses[static_cast<size_t>(r)]->stats().engine_parks;
      if (dgpp::cpu_wait::supported() ? parks == 0 : parks != 0) {
        DGPP_LOG_ERROR("quiet rounds: rank {} counted {} engine parks on a CPU that {}", r, parks,
                       dgpp::cpu_wait::supported() ? "parks" : "cannot park");
        ++failures;
      }
    }
    // Flight parks. On the write transport, where the CPU parks, every
    // engine parked while its generations walked and none turned the parks
    // off. A rank on the decoy is the exception: it has to turn them off
    // once enough of its kernel's stores have landed during its parks.
    // Elsewhere no engine parks in a flight.
    int walked_rounds = 0;
    for (const QuietRound& round : shared.rounds) walked_rounds += round.kind != QuietKind::kEager ? 1 : 0;
    const bool flight_parks = dgpp::cpu_wait::supported() && g_transport == BusLatencyTransport::kWrite;
    uint64_t store_parks = 0, late_wakes = 0, posts = 0, post_wakes = 0;
    for (int r = 0; r < world; ++r) {
      const dgpp::net::BusStats st = buses[static_cast<size_t>(r)]->stats();
      const bool on_decoy = shape.decoy && r == 0;
      if (!on_decoy) {
        store_parks += st.engine_flight_store_parks;
        late_wakes += st.engine_flight_late_wakes;
        posts += st.engine_flight_posts;
        post_wakes += st.engine_flight_post_wakes;
      }
      if (!flight_parks && st.engine_flight_parks != 0) {
        DGPP_LOG_ERROR("quiet rounds: rank {} counted {} flight parks where no engine parks in a flight", r,
                       st.engine_flight_parks);
        ++failures;
      }
      if (flight_parks && walked_rounds >= 16 && st.engine_flight_parks == 0) {
        DGPP_LOG_ERROR("quiet rounds: rank {} never parked in a flight over {} walked rounds", r, walked_rounds);
        ++failures;
      }
      if (st.engine_flight_parks_off && !on_decoy) {
        DGPP_LOG_ERROR("quiet rounds: rank {} turned its flight parks off ({} late wakes over {} store parks)",
                       r, st.engine_flight_late_wakes, st.engine_flight_store_parks);
        ++failures;
      }
      if (on_decoy && flight_parks && st.engine_flight_store_parks >= 2 * 4096 && !st.engine_flight_parks_off) {
        DGPP_LOG_ERROR("quiet rounds: rank {} on the decoy kept its flight parks on ({} late wakes over {} "
                       "store parks)", r, st.engine_flight_late_wakes, st.engine_flight_store_parks);
        ++failures;
      }
    }
    // What the aligned windows took, by the quiet time before them.
    std::vector<double> polling, edge, parked;
    for (const auto& rank_samples : samples)
      for (const QuietSample& sample : rank_samples)
        (sample.gap_us < 300.0 ? polling : sample.gap_us < 10000.0 ? edge : parked)
            .push_back(sample.window_us);
    DGPP_LOG_INFO(
        "quiet rounds ({}, world {}, seed {}, {} elems, {} gens a window{}): {} rounds, {} failures; "
        "window us p50/p99/max "
        "after under 0.3 ms quiet {:.1f}/{:.1f}/{:.1f} (n={}), after 0.3-6 ms {:.1f}/{:.1f}/{:.1f} "
        "(n={}), after 10 ms or more {:.1f}/{:.1f}/{:.1f} (n={}); engine parks on rank 0: {} quiet, "
        "{} in flights; the world's kernels stored during {} flight parks, {} of them woke late; the first "
        "pass after a park found the ready store of {} of {} generations",
        transport_name(), world, seed, elems, gens_per_window, shape.decoy ? ", rank 0 on the decoy" : "",
        rounds, failures, percentile(polling, 0.5),
        percentile(polling, 0.99), percentile(polling, 1.0), polling.size(), percentile(edge, 0.5),
        percentile(edge, 0.99), percentile(edge, 1.0), edge.size(), percentile(parked, 0.5),
        percentile(parked, 0.99), percentile(parked, 1.0), parked.size(),
        buses[0]->stats().engine_parks, buses[0]->stats().engine_flight_parks, store_parks, late_wakes,
        post_wakes, posts);
  }
  for (auto& bus : buses) bus->quiesce();
  for (auto& bus : buses) bus->stop();
  for (QuietRankBuffers& b : bufs) {
    if (b.dev_src) cudaFree(b.dev_src);
    if (b.dev_dst) cudaFree(b.dev_dst);
    if (b.warm_buf) cudaFree(b.warm_buf);
    if (b.stream) cudaStreamDestroy(b.stream);
  }
  return failures;
}

// Both worlds under both latency transports.
void scenario_quiet_rounds(int rounds, uint64_t seed, const QuietShape& shape = {}) {
  const BusLatencyTransport before = g_transport;
  int failures = 0;
  for (const BusLatencyTransport transport : {BusLatencyTransport::kSend, BusLatencyTransport::kWrite}) {
    g_transport = transport;
    failures += quiet_rounds_world(2, 29861, rounds, seed, shape);
    failures += quiet_rounds_world(4, 29862, rounds, seed + 1, shape);
  }
  g_transport = before;
  CHECK(failures == 0, "quiet rounds had " + std::to_string(failures) + " failures");
  DGPP_LOG_INFO("scenario quiet_rounds: {} rounds a world, {} failures", rounds, failures);
}

// ---- the flight park's breaker ---------------------------------------------
//
// While a write-transport generation is in flight the engine parks on the
// block record its kernel stores next, and that store ends the park
// (collective_bus.cpp, engine_loop). That a GPU store ends a host park is a
// property of the platform. On one that lost it, every such park would run
// to its bound and every collective would wait for it. Rank 0 runs as that
// platform here: its engine parks on a word nothing writes. Its breaker has
// to turn the flight parks off, every window has to stay exact before and
// after, the engine must not park in a flight again, and rank 1, which
// parks on the records, has to keep its parks.
void scenario_flight_park_breaker() {
  if (!dgpp::cpu_wait::supported()) {
    DGPP_LOG_INFO("scenario flight_park_breaker: skipped (this CPU cannot park)");
    return;
  }
  constexpr int world = 2;
  const size_t elems = 4096;
  const int gens_per_window = 16;
  const int windows = 800;
  const BusLatencyTransport before = g_transport;
  g_transport = BusLatencyTransport::kWrite;
  std::vector<std::unique_ptr<CollectiveBus>> buses =
      start_world(world, 29863, /*one_lane=*/false, /*rank0_pass_delay_us=*/0, 8192,
                  /*rank0_flight_park_decoy=*/true);
  g_transport = before;
  if (buses.empty()) {
    CHECK(false, "flight park breaker: the world failed to start");
    return;
  }
  std::vector<std::vector<uint16_t>> all_src(static_cast<size_t>(world));
  for (int r = 0; r < world; ++r) fill_rank_bf16(&all_src[static_cast<size_t>(r)], elems, r);
  std::vector<uint16_t> want(elems, 0);
  for (size_t i = 0; i < elems; ++i) {
    float acc = 0.0f;
    for (int r = 0; r < world; ++r) acc += dgpp::net::bf16_to_f32(all_src[static_cast<size_t>(r)][i]);
    want[i] = dgpp::net::bf16_from_f32_rne(acc);
  }
  int failures = 0;
  std::vector<QuietRankBuffers> bufs(static_cast<size_t>(world));
  for (QuietRankBuffers& b : bufs) {
    if (cudaStreamCreateWithFlags(&b.stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaMalloc(&b.dev_src, elems * 2) != cudaSuccess ||
        cudaMalloc(&b.dev_dst, elems * 2) != cudaSuccess ||
        cudaMalloc(&b.warm_buf, 256 * 4) != cudaSuccess) {
      DGPP_LOG_ERROR("flight park breaker: device allocation failed");
      ++failures;
    }
  }
  if (failures == 0) {
    // Windows only, back to back, every rank starting a window together.
    // The ranks pause once, late in the run, with no collective in flight.
    // By then rank 0's kernel has stored during several times the parks its
    // breaker judges at a time: its start stamp and its `done` store of
    // every generation land during a park.
    const int pause_round = windows - 100;
    QuietShared shared(world);
    shared.rounds.assign(static_cast<size_t>(windows), QuietRound{QuietKind::kWindow, 0.0});
    shared.aligned_rounds = windows;
    shared.pause_round = pause_round;
    std::vector<int> rank_failures(static_cast<size_t>(world), 0);
    std::vector<std::vector<QuietSample>> samples(static_cast<size_t>(world));
    std::vector<std::thread> workers;
    for (int r = 0; r < world; ++r)
      workers.emplace_back([&, r] {
        rank_failures[static_cast<size_t>(r)] = quiet_rank_work(
            *buses[static_cast<size_t>(r)], shared, r, elems, gens_per_window, /*seed=*/1,
            bufs[static_cast<size_t>(r)], &want, &samples[static_cast<size_t>(r)]);
      });
    while (shared.paused.load(std::memory_order_acquire) < world &&
           !shared.abort.load(std::memory_order_acquire))
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const dgpp::net::BusStats decoy_at_pause = buses[0]->stats();
    const dgpp::net::BusStats peer_at_pause = buses[1]->stats();
    shared.resume.store(true, std::memory_order_release);
    for (auto& t : workers) t.join();
    for (int r = 0; r < world; ++r) failures += rank_failures[static_cast<size_t>(r)];
    const dgpp::net::BusStats decoy = buses[0]->stats();
    const dgpp::net::BusStats peer = buses[1]->stats();
    CHECK(failures == 0, "flight park breaker: " + std::to_string(failures) + " window failures");
    CHECK(decoy_at_pause.engine_flight_parks_off,
          "flight park breaker: rank 0 on the decoy kept its flight parks on over " +
              std::to_string(pause_round) + " windows (" +
              std::to_string(decoy_at_pause.engine_flight_late_wakes) + " late wakes over " +
              std::to_string(decoy_at_pause.engine_flight_store_parks) + " store parks)");
    CHECK(decoy.engine_flight_parks == decoy_at_pause.engine_flight_parks,
          "flight park breaker: rank 0 parked in a flight after its breaker tripped (" +
              std::to_string(decoy_at_pause.engine_flight_parks) + " parks at the pause, " +
              std::to_string(decoy.engine_flight_parks) + " at the end)");
    CHECK(!peer.engine_flight_parks_off && peer.engine_flight_parks > peer_at_pause.engine_flight_parks &&
              peer_at_pause.engine_flight_parks != 0,
          "flight park breaker: rank 1 parks on the records and should have kept parking (" +
              std::to_string(peer_at_pause.engine_flight_parks) + " parks at the pause, " +
              std::to_string(peer.engine_flight_parks) + " at the end, " +
              std::to_string(peer.engine_flight_late_wakes) + " late wakes over " +
              std::to_string(peer.engine_flight_store_parks) + " store parks)");
    DGPP_LOG_INFO(
        "scenario flight_park_breaker: {} windows of {} generations, {} failures; rank 0 (decoy) turned its "
        "flight parks off after {} parks ({} late wakes over {} store parks) and walked the last {} windows "
        "without one; rank 1 parked {} times ({} late wakes over {} store parks; the first pass after a park "
        "found the ready store of {} of {} generations)",
        windows, gens_per_window, failures, decoy.engine_flight_parks, decoy.engine_flight_late_wakes,
        decoy.engine_flight_store_parks, windows - pause_round, peer.engine_flight_parks,
        peer.engine_flight_late_wakes, peer.engine_flight_store_parks, peer.engine_flight_post_wakes,
        peer.engine_flight_posts);
  } else {
    CHECK(false, "flight park breaker: setup failed");
  }
  for (auto& bus : buses) bus->quiesce();
  for (auto& bus : buses) bus->stop();
  for (QuietRankBuffers& b : bufs) {
    if (b.dev_src) cudaFree(b.dev_src);
    if (b.dev_dst) cudaFree(b.dev_dst);
    if (b.warm_buf) cudaFree(b.warm_buf);
    if (b.stream) cudaStreamDestroy(b.stream);
  }
}

void scenario_idle_gap_collective() {
  // The real-mesh shape, found by the M5 exit-gate fabric run:
  // buses sit IDLE for seconds while a host loads weights (a cold peer's
  // first inter-collective gap — cold NVMe reads of replicated globals
  // before its first boundary), then the first collective posts against a
  // lane whose last traffic flowed before the gap. The lane watchdog must
  // arm from the POST, not inherit the idle period as a stall — otherwise
  // every cold-start fabric forward dies on its first collective.
  // Tight budget on purpose: the gaps below exceed it by a wide margin, so
  // stale arming cannot pass by accident.
  BusOptions a_opt = base_options(0, 29903);
  BusOptions b_opt = base_options(1, 29903);
  // Collective mode (the per-collective kernel owns claims) — and with it
  // no persistent consumer kernels spinning on the device, which would
  // wedge a worker's cudaMalloc mid-scenario (the bring-up hazard class).
  a_opt.launch_consumers = false;
  b_opt.launch_consumers = false;
  a_opt.completion_timeout_ms = 800;
  b_opt.completion_timeout_ms = 800;
  auto [a, b] = start_pair(a_opt, b_opt);
  if (!a || !b) return;

  const size_t elems = 4096;
  std::vector<int> failures(2, 0);
  for (int round = 0; round < 3; ++round) {
    // The idle gap — longer than the watchdog budget, before the first
    // collective AND between rounds.
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    std::vector<std::thread> workers;
    workers.emplace_back(
        [&] { failures[0] += allreduce_rank_work(*a, 2, 0, elems, 1); });
    workers.emplace_back(
        [&] { failures[1] += allreduce_rank_work(*b, 2, 1, elems, 1); });
    for (auto& t : workers) t.join();
  }
  const int total = failures[0] + failures[1];
  CHECK(total == 0, "idle-gap collectives: " + std::to_string(total) +
                        " failures across 3 rounds (lane watchdog fired on "
                        "an idle gap?)");
  a->quiesce();
  b->quiesce();
  a->stop();
  b->stop();
  DGPP_LOG_INFO("scenario idle_gap_collective: 3 collectives across 2s "
                "idle gaps, clean");
}

void scenario_geometry_mismatch() {
  // Config errors must be legible over the rendezvous (roster precedent),
  // never a bare close or a hang.
  BusOptions a_opt = base_options(0, 29870);
  BusOptions b_opt = base_options(1, 29870);
  a_opt.lat_slots = 8;
  b_opt.lat_slots = 16;
  auto a = std::make_unique<CollectiveBus>(a_opt);
  auto b = std::make_unique<CollectiveBus>(b_opt);
  std::string a_error;
  std::thread a_thread([&] {
    if (!a->start(&a_error)) DGPP_LOG_INFO("A rejected: {}", a_error);
  });
  std::string b_error;
  const bool b_ok = b->start(&b_error);
  a_thread.join();
  CHECK(!b_ok, "mismatched geometry must fail startup");
  CHECK(b_error.find("geometry") != std::string::npos ||
            b_error.find("mismatch") != std::string::npos,
        "expected a geometry-mismatch reason, got: " + b_error);
  CHECK(a_error.find("geometry") != std::string::npos ||
            !a_error.empty(),
        "rank A should also fail legibly, got: " + a_error);
  DGPP_LOG_INFO("scenario geometry_mismatch: rejected as designed");
  a->quiesce();
  b->quiesce();
  a->stop();
  b->stop();
}

void scenario_stop_releases_waiters() {
  // Stopping while a message is in flight must resolve its waiter — either
  // completed or failed — and never hang.
  auto [a, b] = start_pair(base_options(0, 29880), base_options(1, 29880));
  if (!a || !b) return;
  std::vector<uint64_t> payload(8192 / 8);
  fill_payload(payload.data(), payload.size(), 3);
  std::string error;
  const uint64_t id =
      b->send(0, payload.data(), 8192, BusMessageClass::kLatency, &error);
  CHECK(id != 0, "send rejected: " + error);
  // Quiesce first: the engine drains the in-flight request (completes it
  // via the peer, or the watchdog fails it), so the waiter resolves before
  // stop() reaps the registry — never a hang, never a lost notification.
  b->quiesce();
  a->quiesce();
  const BusSendResult r = b->wait(id, 15000);
  CHECK(r.ok || !r.error.empty(),
        "waiter neither completed nor failed cleanly: " + r.error);
  DGPP_LOG_INFO("scenario stop_releases_waiters: resolved ({})",
                r.ok ? "completed" : r.error);
  a->stop();
  b->stop();
}

}  // namespace

// A world of one (2026-09-10, the single-Spark serve): the bus starts
// without lanes or a rendezvous, every collective is the identity (a copy
// when the destination differs), the staging handout is one pinned slot,
// the graph hooks succeed in session order, and the peer surface refuses.
void scenario_world_of_one() {
  DGPP_LOG_INFO("scenario: a world of one");
  dgpp::net::BusRankExchange frame{};
  frame.lane_count = 1;
  frame.peer_count = 1;
  std::vector<uint8_t> wire(dgpp::net::kBusMaxExchangeBytes);
  const size_t wire_size = dgpp::net::bus_rank_exchange_encode(frame, wire.data(), wire.size());
  dgpp::net::BusRankExchange decoded{};
  CHECK(wire_size > 0 && dgpp::net::bus_rank_exchange_decode(wire.data(), wire_size, &decoded),
        "current collective protocol must round-trip");
  wire[4] = 1;
  CHECK(!dgpp::net::bus_rank_exchange_decode(wire.data(), wire_size, &decoded),
        "old peers must not use the incomplete placement hash");
  BusOptions o = base_options(0, 29899);
  o.world_size = 1;
  CollectiveBus bus(o);
  std::string err;
  if (!bus.start(&err)) {
    CHECK(false, "world-of-one start: " + err);
    return;
  }
  // No peers.
  CHECK(bus.send(0, nullptr, 8, BusMessageClass::kLatency, &err) == 0, "send must refuse");
  // The identity, in place and as a copy.
  constexpr size_t elems = 1024;
  std::vector<uint16_t> h(elems);
  for (size_t i = 0; i < elems; ++i) h[i] = static_cast<uint16_t>(0x3C00 + (i % 97));
  uint16_t* src = nullptr;
  uint16_t* dst = nullptr;
  CHECK(cudaMalloc(&src, elems * 2) == cudaSuccess && cudaMalloc(&dst, elems * 2) == cudaSuccess, "alloc");
  CHECK(cudaMemcpy(src, h.data(), elems * 2, cudaMemcpyHostToDevice) == cudaSuccess, "upload");
  CHECK(cudaMemset(dst, 0, elems * 2) == cudaSuccess, "zero");
  uint64_t id = bus.allreduce(src, src, elems, &err);
  CHECK(id != 0 && bus.wait_allreduce(id, 1000).ok, "in-place allreduce: " + err);
  id = bus.allreduce(src, dst, elems, &err);
  CHECK(id != 0 && bus.wait_allreduce(id, 1000).ok, "copy allreduce: " + err);
  std::vector<uint16_t> back(elems);
  CHECK(cudaMemcpy(back.data(), dst, elems * 2, cudaMemcpyDeviceToHost) == cudaSuccess, "download");
  CHECK(back == h, "the copy allreduce must be the identity");
  CHECK(!bus.wait_allreduce(id, 1000).ok, "a waited id is gone");
  id = bus.allreduce_bulk(src, dst, elems, &err);
  CHECK(id != 0 && bus.wait_allreduce(id, 1000).ok, "bulk allreduce: " + err);
  // The staging handout: one at a time, consumed by the staged submit.
  void* slot = bus.stage_next(&err);
  CHECK(slot != nullptr, "stage_next: " + err);
  CHECK(bus.stage_next(&err) == nullptr, "a second handout must refuse");
  CHECK(bus.allreduce(src, src, elems, &err) == 0, "an eager collective under a held handout must refuse");
  std::memset(slot, 0x5A, o.lat_slot_bytes);
  id = bus.allreduce_staged(elems, &err);
  CHECK(id != 0 && bus.wait_allreduce(id, 1000).ok, "staged allreduce: " + err);
  CHECK(static_cast<const uint8_t*>(slot)[0] == 0x5A, "the staged fold is in place");
  // The graph session: record, arm, finish, in order.
  CHECK(bus.graph_record_begin(&err, 0), "record begin: " + err);
  CHECK(!bus.graph_replay_arm(&err, 0), "arm while recording must refuse");
  cudaStream_t stream = nullptr;
  CHECK(cudaStreamCreate(&stream) == cudaSuccess, "stream");
  CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess, "capture begin");
  CHECK(bus.allreduce_record(stream, src, src, elems, &err), "allreduce_record: " + err);
  cudaGraph_t graph = nullptr;
  CHECK(cudaStreamEndCapture(stream, &graph) == cudaSuccess, "capture end");
  size_t nodes = 0;
  CHECK(cudaGraphGetNodes(graph, nullptr, &nodes) == cudaSuccess && nodes == 0, "an in-place record adds no node");
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  CHECK(bus.graph_record_end(&err), "record end: " + err);
  CHECK(!bus.graph_replay_finish(1000, &err), "finish without an armed window must refuse");
  CHECK(bus.graph_replay_arm(&err, 0), "arm: " + err);
  CHECK(bus.graph_replay_arm(&err, 0), "a second arm (the pipelined replay): " + err);
  CHECK(bus.stage_next(&err) == nullptr, "no handout while a window is armed");
  CHECK(bus.graph_replay_finish(1000, &err), "finish 1: " + err);
  CHECK(bus.graph_replay_finish(1000, &err), "finish 2: " + err);
  CHECK(bus.graph_record_begin(&err, 1), "a second variant records between windows: " + err);
  CHECK(bus.graph_record_end(&err), "record end 2: " + err);
  bus.stop();
  CHECK(bus.allreduce(src, src, elems, &err) == 0, "a stopped bus refuses");
  cudaFree(src);
  cudaFree(dst);
  DGPP_LOG_INFO("scenario: a world of one OK");
}

int main(int argc, char** argv) {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  // One process, several ranks, each with kernels that spin on a peer's
  // doorbell: CUDA's default LAZY module loading deadlocks that shape (a
  // first launch waits for an idle device — see bus_kernel.hpp,
  // bus_preload_kernels). The bus preloads its own kernels; the model's
  // compute kernels launched beside a live collective are loaded eagerly
  // here, before the first CUDA call. Production is one rank per box.
  setenv("CUDA_MODULE_LOADING", "EAGER", /*overwrite=*/0);

  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices == 0) {
    DGPP_LOG_INFO("no CUDA device visible; skipping bus_test");
    return 2;
  }

  // bus_test --quiet-rounds N [--seed S] [--elems E] [--gens G] [--decoy]:
  // the quiet rounds alone, N a world, for long runs over many schedules
  // and shapes. E is the row length (it sets the blocks a generation has),
  // G the generations of a window, and --decoy puts rank 0's engine on the
  // flight park's fault injection.
  if (argc > 1 && std::string(argv[1]) == "--quiet-rounds") {
    try {
      const char* usage = "use --quiet-rounds N [--seed S] [--elems E] [--gens G] [--decoy]";
      if (argc < 3) throw std::invalid_argument(usage);
      const int rounds = std::stoi(argv[2]);
      uint64_t seed = 1;
      QuietShape shape;
      for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--decoy") { shape.decoy = true; continue; }
        if (i + 1 >= argc) throw std::invalid_argument(usage);
        const std::string value = argv[++i];
        if (arg == "--seed") seed = std::stoull(value);
        else if (arg == "--elems") shape.elems = std::stoull(value);
        else if (arg == "--gens") shape.gens_per_window = std::stoi(value);
        else throw std::invalid_argument(usage);
      }
      if (rounds < 2) throw std::invalid_argument("at least two rounds");
      if (shape.elems < 1 || shape.elems > 768000) throw std::invalid_argument("--elems from 1 to 768000");
      if (shape.gens_per_window < 1 || shape.gens_per_window > 64)
        throw std::invalid_argument("--gens from 1 to 64");
      scenario_quiet_rounds(rounds, seed, shape);
      return g_failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
      DGPP_LOG_ERROR("quiet rounds: {}", e.what());
      return 2;
    }
  }
  if (argc > 1) {
    int rank = -1, world = 0, port = 29924;
    std::string peer;
    bool one_lane = false;
    try {
      for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--one-lane") { one_lane = true; continue; }
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
        const std::string value = argv[++i];
        if (arg == "--bulk-rank") rank = std::stoi(value);
        else if (arg == "--world") world = std::stoi(value);
        else if (arg == "--peer") peer = value;
        else if (arg == "--port") port = std::stoi(value);
        else throw std::invalid_argument("unknown option " + arg);
      }
      if ((world != 2 && world != 4) || rank < 0 || rank >= world ||
          (rank != 0 && peer.empty()) || port < 1 || port > 65535)
        throw std::invalid_argument("use --bulk-rank R --world 2|4 --peer HOST [--port N] [--one-lane]");
      return distributed_bulk_cases(rank, world, peer, static_cast<uint16_t>(port), one_lane);
    } catch (const std::exception& e) {
      DGPP_LOG_ERROR("distributed bulk: {}", e.what());
      return 2;
    }
  }

  scenario_world_of_one();
  // DGPP_TEST_FILTER=world_of_one: the world-of-one gate alone (no lanes,
  // no loopback fabric needed).
  if (const char* f = std::getenv("DGPP_TEST_FILTER"); f && std::string(f) == "world_of_one")
    return g_failures == 0 ? 0 : 1;
  scenario_graph_snapshot_hash();
  if (const char* f = std::getenv("DGPP_TEST_FILTER"); f && std::string(f) == "graph_snapshot_hash")
    return g_failures == 0 ? 0 : 1;
  // DGPP_TEST_FILTER=write: the one-sided transport's graph, row and
  // stream scenarios alone.
  if (const char* f = std::getenv("DGPP_TEST_FILTER"); f && std::string(f) == "write") {
    g_transport = BusLatencyTransport::kWrite;
    scenario_allreduce_graph();
    scenario_allreduce_graph_rows();
    scenario_allreduce_stream();
    return g_failures == 0 ? 0 : 1;
  }
  if (const char* f = std::getenv("DGPP_TEST_FILTER"); f && std::string(f) == "quiet_rounds") {
    scenario_quiet_rounds(200, 1);
    return g_failures == 0 ? 0 : 1;
  }
  if (const char* f = std::getenv("DGPP_TEST_FILTER"); f && std::string(f) == "flight_park_breaker") {
    scenario_flight_park_breaker();
    return g_failures == 0 ? 0 : 1;
  }
  if (const char* f = std::getenv("DGPP_TEST_FILTER");
      f && std::string(f) == "allreduce_graph_rows") {
    scenario_allreduce_graph_rows();
    return g_failures == 0 ? 0 : 1;
  }
  if (const char* f = std::getenv("DGPP_TEST_FILTER");
      f && std::string(f) == "allreduce_bulk") {
    scenario_allreduce_bulk();
    return g_failures == 0 ? 0 : 1;
  }
  if (const char* f = std::getenv("DGPP_TEST_FILTER");
      f && std::string(f) == "collective_progress") {
    scenario_collective_progress();
    scenario_allreduce_graph();
    scenario_allreduce_stream();
    return g_failures == 0 ? 0 : 1;
  }
  scenario_latency_ping();
  scenario_credit_recycle();
  scenario_bulk_dual_lane();
  scenario_latency_under_bulk();
  scenario_completion_timeout();
  scenario_consumer_inactivity_exit();
  scenario_mesh_three_way();
  scenario_collective_progress();
  scenario_allreduce();
  scenario_allreduce_done_before_posted();
  scenario_allreduce_staged();
  scenario_allreduce_bulk();
  scenario_allreduce_graph();
  scenario_allreduce_graph_rows();
  scenario_allreduce_stream();
  g_transport = BusLatencyTransport::kWrite;
  scenario_allreduce_graph();
  scenario_allreduce_graph_rows();
  scenario_allreduce_stream();
  g_transport = BusLatencyTransport::kSend;
  scenario_quiet_rounds(200, 1);
  scenario_flight_park_breaker();
  scenario_idle_gap_collective();
  scenario_geometry_mismatch();
  scenario_stop_releases_waiters();

  DGPP_LOG_INFO("bus_test: {} failures", g_failures);
  return g_failures == 0 ? 0 : 1;
}
