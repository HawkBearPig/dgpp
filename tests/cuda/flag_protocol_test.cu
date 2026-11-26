// Regression tests for the pinned-memory flag protocol (flag_protocol.cuh).
// These pin the ordering contract CollectiveBus will depend on: payload
// released before a flag must be fully visible when the flag is observed,
// and lost wakeups must degrade to detectable timeouts, never a wedged
// device. A driver update silently breaking fence semantics should fail
// here before it fails in production.
#include <cuda_runtime.h>

#include <cstdlib>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#include "common/cpu_wait.hpp"
#include "common/test.hpp"
#include "kernels/flag_protocol.cuh"

// Acks every new start sequence `delay_cycles` of the device clock after it
// saw it, so that a host thread waiting for the ack has parked by the time
// the ack is stored.
__global__ void flag_delayed_ack_kernel(uint32_t* start, dgpp::FlagAck* ack,
                                        uint64_t delay_cycles, uint64_t deadline_cycles) {
  dgpp_pdl_wait();
  uint64_t idle_since = clock64();
  uint32_t last_seen = 0;
  for (;;) {
    uint32_t s = dgpp::flag_load_acquire(start);
    while (s == last_seen) {
      if (clock64() - idle_since > deadline_cycles) return;
      dgpp::flag_poll_pause();
      s = dgpp::flag_load_acquire(start);
    }
    if (s == dgpp::kFlagStopSequence) return;
    last_seen = s;
    const uint64_t seen_at = clock64();
    while (clock64() - seen_at < delay_cycles) dgpp::flag_poll_pause();
    ack->cycles = clock64();
    ack->hash = 0;
    dgpp::flag_store_release(&ack->seq, s);
    idle_since = clock64();
  }
}

namespace {

using dgpp::FlagAck;

constexpr int kMsgs = 256;  // fast enough for CI, long enough to be meaningful
// Watchdog budget for kernels under test. Cycles are clock-dependent, so the
// tests use generous time margins (GB10 clocks keep this well under 1 s).
constexpr uint64_t kDeadlineCycles = 800'000'000ull;
// Host-side per-message bound; must comfortably exceed the kernel deadline.
constexpr double kHostTimeoutSec = 6.0;

#if defined(__aarch64__)
inline void cpu_pause() { asm volatile("yield" ::: "memory"); }
#else
inline void cpu_pause() {}
#endif

struct Fixture {
  dgpp::StartSlot* start = nullptr;
  FlagAck* ack = nullptr;
  uint32_t* payload = nullptr;  // one 64 B line: 16 u32 words
  cudaStream_t stream = nullptr;

  Fixture() {
    if (cudaHostAlloc(&start, sizeof(dgpp::StartSlot), cudaHostAllocDefault) !=
            cudaSuccess ||
        cudaHostAlloc(&ack, sizeof(FlagAck), cudaHostAllocDefault) !=
            cudaSuccess ||
        cudaHostAlloc(&payload, 16 * sizeof(uint32_t), cudaHostAllocDefault) !=
            cudaSuccess ||
        !start || !ack || !payload) {
      if (payload) cudaFreeHost(payload);
      if (ack) cudaFreeHost(ack);
      if (start) cudaFreeHost(start);
      throw std::runtime_error("pinned alloc");
    }
    *start = {};
    *ack = {};
    std::fill_n(payload, 16, 0u);
    if (cudaStreamCreate(&stream) != cudaSuccess) {
      cudaFreeHost(payload);
      cudaFreeHost(ack);
      cudaFreeHost(start);
      payload = nullptr;
      ack = nullptr;
      start = nullptr;
      throw std::runtime_error("stream create");
    }
  }
  ~Fixture() {
    cudaStreamDestroy(stream);
    cudaFreeHost(payload);
    cudaFreeHost(ack);
    cudaFreeHost(start);
  }
};

// Host-side mirror of the device checksum in flag_payload_kernel. MUST stay
// in lockstep with the fold documented there.
uint64_t host_fold(const uint32_t* p) {
  uint64_t h = 0;
  for (int i = 0; i < 16; i += 2) {
    const uint64_t pair = (static_cast<uint64_t>(p[i]) << 32) |
                          static_cast<uint64_t>(p[i + 1]);
    h ^= pair * 0x9E3779B97F4A7C15ull;
  }
  return h;
}

// Waits until ack.seq == want or times out; returns true on success.
bool wait_ack(const FlagAck& ack, uint32_t want) {
  const auto t0 = std::chrono::steady_clock::now();
  while (__atomic_load_n(&ack.seq, __ATOMIC_ACQUIRE) != want) {
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count() > kHostTimeoutSec)
      return false;
    cpu_pause();
  }
  return true;
}

void stop_and_sync(Fixture& fixture) {
  __atomic_store_n(&fixture.start->seq, dgpp::kFlagStopSequence,
                   __ATOMIC_RELEASE);
  if (cudaStreamSynchronize(fixture.stream) != cudaSuccess)
    throw std::runtime_error("orderly flag-kernel stop failed");
}

bool wait_stream_complete(cudaStream_t stream, double timeout_seconds) {
  const auto start = std::chrono::steady_clock::now();
  for (;;) {
    const cudaError_t status = cudaStreamQuery(stream);
    if (status == cudaSuccess) return true;
    if (status != cudaErrorNotReady)
      throw std::runtime_error("stream query failed while waiting for watchdog");
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count() > timeout_seconds)
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

}  // namespace

DGPP_TEST(flag_protocol_with_sequential_writes_delivers_every_ack_in_order) {
  // GIVEN a heartbeat kernel spinning on a pinned start slot:
  Fixture f;
  flag_heartbeat_kernel<<<1, 1, 0, f.stream>>>(&f.start->seq, f.ack,
                                               kDeadlineCycles);

  // WHEN we issue kMsgs monotonically increasing sequences and wait for the
  // matching ack of each:
  uint64_t last_cycles = 0;
  for (uint32_t i = 1; i <= kMsgs; ++i) {
    __atomic_store_n(&f.start->seq, i, __ATOMIC_RELEASE);
    if (!wait_ack(*f.ack, i))
      throw std::runtime_error("ack for seq not observed in order");
    // THEN every stamp is present and non-decreasing (protocol sanity):
    const uint64_t c = f.ack->cycles;
    if (c == 0) throw std::runtime_error("ack cycles missing");
    if (c < last_cycles) throw std::runtime_error("device stamps regressed");
    last_cycles = c;
  }
  stop_and_sync(f);
}

DGPP_TEST(flag_protocol_with_payload_released_before_flag_shows_payload_at_ack) {
  // GIVEN the payload variant of the protocol kernel:
  Fixture f;
  flag_payload_kernel<<<1, 1, 0, f.stream>>>(&f.start->seq, f.payload, f.ack,
                                             kDeadlineCycles);

  // WHEN each message writes a fresh 64 B payload BEFORE releasing the flag:
  for (uint32_t i = 1; i <= kMsgs; ++i) {
    for (int w = 0; w < 16; ++w)
      f.payload[w] = i * 2654435761u + static_cast<uint32_t>(w);
    std::atomic_thread_fence(std::memory_order_release);
    __atomic_store_n(&f.start->seq, i, __ATOMIC_RELEASE);

    // THEN the acked checksum must equal the host's independent fold —
    // proving payload-before-flag visibility end to end:
    if (!wait_ack(*f.ack, i)) throw std::runtime_error("payload ack missing");
    if (f.ack->hash != host_fold(f.payload))
      throw std::runtime_error("payload not fully visible when ack observed");
  }
  stop_and_sync(f);
}

DGPP_TEST(flag_protocol_with_continuous_traffic_resets_inactivity_watchdog) {
  // GIVEN a watchdog shorter than this test's total runtime but longer than
  // the interval between messages:
  Fixture f;
  int clock_khz = 0;
  if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0) !=
          cudaSuccess ||
      clock_khz <= 0)
    throw std::runtime_error("device clock unavailable");
  constexpr uint64_t kIdleBudgetMs = 250;
  const uint64_t deadline =
      static_cast<uint64_t>(clock_khz) * kIdleBudgetMs;
  flag_heartbeat_kernel<<<1, 1, 0, f.stream>>>(&f.start->seq, f.ack, deadline);

  // WHEN messages remain active for more than one full watchdog budget as
  // measured by the GPU's own clock64 domain:
  uint64_t first_cycles = 0;
  uint64_t last_cycles = 0;
  uint32_t last_seq = 0;
  for (uint32_t seq = 1; seq <= 100; ++seq) {
    __atomic_store_n(&f.start->seq, seq, __ATOMIC_RELEASE);
    if (!wait_ack(*f.ack, seq))
      throw std::runtime_error("active watchdog expired from kernel age");
    if (first_cycles == 0) first_cycles = f.ack->cycles;
    last_cycles = f.ack->cycles;
    last_seq = seq;
    if (last_cycles - first_cycles > deadline) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(75));
  }
  if (last_cycles - first_cycles <= deadline)
    throw std::runtime_error("traffic did not span the device watchdog budget");

  // THEN all messages are acknowledged. A final quiet period still causes
  // the intended clean watchdog exit.
  if (!wait_stream_complete(f.stream, kHostTimeoutSec)) {
    stop_and_sync(f);
    throw std::runtime_error("inactivity watchdog did not expire in time");
  }
  if (cudaStreamSynchronize(f.stream) != cudaSuccess)
    throw std::runtime_error("inactivity watchdog failed to drain stream");
  if (__atomic_load_n(&f.ack->seq, __ATOMIC_ACQUIRE) != last_seq)
    throw std::runtime_error("last active sequence was not acknowledged");
}

DGPP_TEST(flag_protocol_with_dead_wakeup_times_out_and_device_remains_usable) {
  // GIVEN a heartbeat kernel whose watchdog expires after a quiet period:
  Fixture f;
  flag_heartbeat_kernel<<<1, 1, 0, f.stream>>>(&f.start->seq, f.ack,
                                               kDeadlineCycles);
  __atomic_store_n(&f.start->seq, 1, __ATOMIC_RELEASE);
  if (!wait_ack(*f.ack, 1)) throw std::runtime_error("healthy ack missing");

  // WHEN traffic stops until the stream proves that the watchdog exited, and
  // a new sequence is then sent to the dead kernel:
  if (!wait_stream_complete(f.stream, kHostTimeoutSec)) {
    stop_and_sync(f);
    throw std::runtime_error("watchdog did not exit within host bound");
  }
  __atomic_store_n(&f.start->seq, 2, __ATOMIC_RELEASE);

  // THEN the stream drains (kernel exited) and seq 2 is never acked —
  // detection, not an infinite hang:
  const cudaError_t err = cudaStreamSynchronize(f.stream);
  if (err != cudaSuccess)
    throw std::runtime_error("stream did not drain after watchdog exit");
  if (__atomic_load_n(&f.ack->seq, __ATOMIC_ACQUIRE) != 1)
    throw std::runtime_error("kernel failed to watchdog-exit");

  // AND the device remains fully usable afterwards:
  float* probe = nullptr;
  cudaMalloc(&probe, 1 << 20);
  cudaMemsetAsync(probe, 0x3F, 1 << 20, f.stream);
  cudaStreamSynchronize(f.stream);
  std::vector<char> back(1 << 20);
  cudaMemcpy(back.data(), probe, back.size(), cudaMemcpyDeviceToHost);
  if (back[123] != 0x3F || back.back() != 0x3F)
    throw std::runtime_error("device unusable after watchdog exit");
  cudaFree(probe);
}

DGPP_TEST(flag_protocol_gpu_release_store_ends_a_host_park) {
  // The serving thread parks on words the GPU publishes (the verdict and
  // confidence sequences) where the CPU can park. That a store from another
  // core ends a park is architectural. That a GPU store into pinned host
  // memory does is a property of this platform, and this case measures it.
  if (!dgpp::cpu_wait::supported()) dgpp::test::skip("this CPU has no WFET");
  Fixture f;
  int clock_khz = 0;
  if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0) != cudaSuccess || clock_khz <= 0)
    throw std::runtime_error("device clock unavailable");
  // The device holds every ack back 300 us: the host is in its wait long
  // before the store, in both forms of the wait.
  constexpr int kDelayUs = 300;
  constexpr int kTrips = 300;
  const uint64_t delay_cycles = static_cast<uint64_t>(clock_khz) * kDelayUs / 1000;
  flag_delayed_ack_kernel<<<1, 1, 0, f.stream>>>(&f.start->seq, f.ack, delay_cycles,
                                                 kDeadlineCycles);

  using Clock = std::chrono::steady_clock;
  auto pause_us = [](double us) {
    const auto t0 = Clock::now();
    while (std::chrono::duration<double, std::micro>(Clock::now() - t0).count() < us) cpu_pause();
  };
  // One round trip: the host's start store, the device's delayed ack, the
  // host seeing the ack. Only the last leg differs between the two forms.
  auto round_trip_us = [&](uint32_t seq, bool park) {
    const auto t0 = Clock::now();
    __atomic_store_n(&f.start->seq, seq, __ATOMIC_RELEASE);
    if (park) {
      // One long park: the store, an event or the deadline ends it.
      const uint64_t deadline = dgpp::cpu_wait::deadline_in_us(2000000);
      for (uint32_t seen; (seen = __atomic_load_n(&f.ack->seq, __ATOMIC_ACQUIRE)) != seq;) {
        if (dgpp::cpu_wait::ticks() >= deadline) throw std::runtime_error("parked wait: ack missing");
        dgpp::cpu_wait::park_while_equal(&f.ack->seq, seen, deadline);
      }
    } else if (!wait_ack(*f.ack, seq)) {
      throw std::runtime_error("polling wait: ack missing");
    }
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
  };

  std::vector<double> polled, parked;
  uint32_t seq = 0;
  for (int i = 0; i < 16; ++i) round_trip_us(++seq, (i & 1) != 0);  // warm both forms
  for (int i = 0; i < kTrips; ++i) {
    // A different phase against the timer event stream every trip.
    pause_us(static_cast<double>((i * 37) % 100));
    polled.push_back(round_trip_us(++seq, /*park=*/false));
    pause_us(static_cast<double>((i * 53) % 100));
    parked.push_back(round_trip_us(++seq, /*park=*/true));
  }
  stop_and_sync(f);

  std::sort(polled.begin(), polled.end());
  std::sort(parked.begin(), parked.end());
  auto at = [](const std::vector<double>& v, double frac) {
    return v[std::min(v.size() - 1, static_cast<size_t>(frac * static_cast<double>(v.size())))];
  };
  std::printf(
      "host wake after a GPU release-store (%d round trips each, the device holding its ack "
      "%d us): polling p50 %.2f p90 %.2f p99 %.2f us, parked p50 %.2f p90 %.2f p99 %.2f us; "
      "parking adds %.2f us at the median\n",
      kTrips, kDelayUs, at(polled, 0.5), at(polled, 0.9), at(polled, 0.99), at(parked, 0.5),
      at(parked, 0.9), at(parked, 0.99), at(parked, 0.5) - at(polled, 0.5));
  // A park the store does not end lasts until the timer event stream's
  // next tick, about 50 us later at the median, or until the deadline.
  if (at(parked, 0.5) - at(polled, 0.5) > 10.0)
    throw std::runtime_error("a GPU release-store no longer ends a host park promptly (median)");
  if (at(parked, 0.9) - at(polled, 0.9) > 50.0)
    throw std::runtime_error("a GPU release-store no longer ends a host park promptly (p90)");
}

int main() {
  // One process, several ranks, each with kernels that spin on a peer's
  // doorbell: CUDA's default LAZY module loading deadlocks that shape (a
  // first launch waits for an idle device — see bus_kernel.hpp,
  // bus_preload_kernels). The bus preloads its own kernels; the model's
  // compute kernels launched beside a live collective are loaded eagerly
  // here, before the first CUDA call. Production is one rank per box.
  setenv("CUDA_MODULE_LOADING", "EAGER", /*overwrite=*/0);
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  return dgpp::test::run_all();
}
