// The host waits on a word another agent writes (common/cpu_wait.hpp): the
// park itself where the CPU has one, and both forms of the wait loops on
// every CPU. The wake by a GPU store is measured in
// tests/cuda/flag_protocol_test.cu.
#include "common/cpu_wait.hpp"
#include "common/test.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace cw = dgpp::cpu_wait;
using Clock = std::chrono::steady_clock;

double us_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

void poll_for_us(double us) {
  const auto t0 = Clock::now();
  while (us_since(t0) < us) cw::spin_hint();
}

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error(what);
}

void require_parks() {
  if (!cw::supported()) dgpp::test::skip("this CPU has no WFET");
}

// A word alone on its line, so that nothing but the test writes there.
template <class T>
struct alignas(64) Lone {
  T value;
  char pad[64 - sizeof(T)];
};

// 2000 parks on a word that already differs must not wait. One that did
// wait would last until the timer event stream's next tick, tens of
// microseconds each.
template <class T>
void park_on_a_changed_word(const char* name) {
  Lone<T> word{};
  word.value = T{5};
  const auto t0 = Clock::now();
  for (int i = 0; i < 2000; ++i)
    cw::park_while_equal(&word.value, T{4}, cw::deadline_in_us(2000000));
  const double us = us_since(t0);
  require(us < 10000.0, std::string(name) + ": 2000 parks on a changed word took " +
                            std::to_string(us) + " us");
}

template <class T>
void park_on_a_changed_atomic(const char* name, T now, T old) {
  Lone<std::atomic<T>> word{};
  word.value.store(now);
  const auto t0 = Clock::now();
  for (int i = 0; i < 2000; ++i) cw::park_while_equal(word.value, old, cw::deadline_in_us(2000000));
  const double us = us_since(t0);
  require(us < 10000.0, std::string(name) + ": 2000 parks on a changed atomic took " +
                            std::to_string(us) + " us");
}

// Both wait loops, in the form `park` asks for.
void wait_loops_reach_the_word_and_time_out(bool park) {
  Lone<uint64_t> counter{};
  counter.value = 3;
  using std::chrono::milliseconds;
  require(cw::wait_until_at_least(&counter.value, 3, milliseconds(50), park),
          "a count already reached must return true");
  require(cw::wait_until_at_least(&counter.value, 2, milliseconds(0), park),
          "a count already passed must return true with no time left");

  auto t0 = Clock::now();
  require(!cw::wait_until_at_least(&counter.value, 4, milliseconds(30), park),
          "a count never reached must time out");
  double us = us_since(t0);
  require(us > 29000.0 && us < 500000.0,
          "the 30 ms timeout returned after " + std::to_string(us) + " us");

  // The writer passes a value below the target first: the wait must go on.
  std::thread writer([&] {
    std::this_thread::sleep_for(milliseconds(3));
    __atomic_store_n(&counter.value, uint64_t{4}, __ATOMIC_RELEASE);
    std::this_thread::sleep_for(milliseconds(3));
    __atomic_store_n(&counter.value, uint64_t{6}, __ATOMIC_RELEASE);
  });
  t0 = Clock::now();
  const bool reached = cw::wait_until_at_least(&counter.value, 5, milliseconds(5000), park);
  us = us_since(t0);
  writer.join();
  require(reached, "a count another thread advances must be reached");
  require(us > 5000.0 && us < 2000000.0,
          "the wait for the second store returned after " + std::to_string(us) + " us");
  require(__atomic_load_n(&counter.value, __ATOMIC_ACQUIRE) == 6, "the counter's last value");

  Lone<std::atomic<bool>> flag{};
  flag.value.store(false);
  std::thread setter([&] {
    std::this_thread::sleep_for(milliseconds(3));
    flag.value.store(true, std::memory_order_release);
  });
  t0 = Clock::now();
  while (!flag.value.load(std::memory_order_acquire)) {
    if (us_since(t0) > 5e6) break;
    cw::wait_step(flag.value, false, park);
  }
  setter.join();
  require(flag.value.load(), "a flag another thread sets must end the step loop");
}

}  // namespace

DGPP_TEST(cpu_wait_park_returns_at_once_when_the_word_already_differs) {
  require_parks();
  park_on_a_changed_word<uint8_t>("uint8_t");
  park_on_a_changed_word<uint32_t>("uint32_t");
  park_on_a_changed_word<uint64_t>("uint64_t");
  park_on_a_changed_word<int64_t>("int64_t");
  park_on_a_changed_atomic<bool>("atomic<bool>", true, false);
  park_on_a_changed_atomic<uint32_t>("atomic<uint32_t>", 7u, 6u);
  park_on_a_changed_atomic<uint64_t>("atomic<uint64_t>", uint64_t{1} << 40, uint64_t{0});
}

DGPP_TEST(cpu_wait_park_ends_by_its_deadline_when_nothing_writes) {
  require_parks();
  // The tick rate turns microseconds into the counter's units.
  const uint64_t a = cw::ticks();
  const uint64_t d = cw::deadline_in_us(1000);
  const double ticks_per_ms = static_cast<double>(cw::ticks_hz()) / 1000.0;
  require(static_cast<double>(d - a) > 0.95 * ticks_per_ms &&
              static_cast<double>(d - a) < 1.5 * ticks_per_ms,
          "a 1000 us deadline is about one millisecond of ticks away");

  Lone<uint64_t> word{};
  word.value = 9;
  const auto t0 = Clock::now();
  const uint64_t deadline = cw::deadline_in_us(3000);
  int parks = 0;
  while (cw::ticks() < deadline) {
    cw::park_while_equal(&word.value, uint64_t{9}, deadline);
    ++parks;
  }
  const double us = us_since(t0);
  require(us > 2900.0 && us < 200000.0,
          "parks to a 3 ms deadline ended after " + std::to_string(us) + " us");
  // A park that returned without waiting would loop here millions of times.
  require(parks < 3000, "the parks to a 3 ms deadline did not wait (" + std::to_string(parks) + ")");
}

DGPP_TEST(cpu_wait_a_store_from_another_thread_ends_the_park) {
  require_parks();
  constexpr int kTrials = 300;
  Lone<std::atomic<uint64_t>> word{};
  Lone<std::atomic<int>> parked_for{};  // the trial the waiter is waiting in, plus one
  Lone<std::atomic<bool>> gave_up{};
  word.value.store(0);
  parked_for.value.store(0);
  gave_up.value.store(false);
  std::vector<int64_t> seen_ns(kTrials, 0);
  auto now_ns = [] {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
        .count();
  };
  std::thread waiter([&] {
    for (int trial = 0; trial < kTrials; ++trial) {
      const uint64_t old = static_cast<uint64_t>(trial);
      parked_for.value.store(trial + 1, std::memory_order_release);
      const uint64_t deadline = cw::deadline_in_us(2000000);
      while (word.value.load(std::memory_order_acquire) == old) {
        if (cw::ticks() >= deadline) {
          gave_up.value.store(true, std::memory_order_release);
          return;
        }
        cw::park_while_equal(word.value, old, deadline);
      }
      seen_ns[static_cast<size_t>(trial)] = now_ns();
    }
  });
  std::vector<double> wake_us;
  for (int trial = 0; trial < kTrials && !gave_up.value.load(std::memory_order_acquire); ++trial) {
    while (parked_for.value.load(std::memory_order_acquire) != trial + 1 &&
           !gave_up.value.load(std::memory_order_acquire))
      cw::spin_hint();
    // Long enough for the waiter to be in its park, and a different phase
    // against the timer event stream every trial.
    poll_for_us(150.0 + static_cast<double>((trial * 37) % 100));
    const int64_t stored = now_ns();
    word.value.store(static_cast<uint64_t>(trial) + 1, std::memory_order_release);
    // The waiter's next announcement follows its timestamp for this trial.
    while (trial + 1 < kTrials &&
           parked_for.value.load(std::memory_order_acquire) != trial + 2 &&
           !gave_up.value.load(std::memory_order_acquire))
      cw::spin_hint();
    if (trial + 1 == kTrials) break;
    wake_us.push_back(static_cast<double>(seen_ns[static_cast<size_t>(trial)] - stored) / 1000.0);
  }
  waiter.join();
  require(!gave_up.value.load(), "a parked waiter never saw the other thread's store");
  std::sort(wake_us.begin(), wake_us.end());
  const double p50 = wake_us[wake_us.size() / 2];
  const double p99 = wake_us[wake_us.size() * 99 / 100];
  std::printf("a store from another thread ended a park in p50 %.2f us, p99 %.2f us (%zu trials)\n",
              p50, p99, wake_us.size());
  // Without the wake, a park ends at the timer event stream's next tick:
  // tens of microseconds at the median.
  require(p50 < 20.0, "the median wake took " + std::to_string(p50) + " us");
}

DGPP_TEST(cpu_wait_polling_loops_reach_the_word_and_time_out) {
  wait_loops_reach_the_word_and_time_out(/*park=*/false);
}

DGPP_TEST(cpu_wait_parking_loops_reach_the_word_and_time_out) {
  require_parks();
  wait_loops_reach_the_word_and_time_out(/*park=*/true);
}
