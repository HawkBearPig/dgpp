#pragma once

// Waits on a word in memory that another agent writes: a CPU thread, the GPU
// or the NIC.
//
// A waiter that polls such a word keeps its core at the full clock for as
// long as it waits. On AArch64 with FEAT_WFxT (GB10's Cortex-X925 and A725)
// the waiter parks instead. It arms the exclusive monitor on the word's
// cache line (LDXR) and executes WFET, which holds the core in a low-power
// state until the line is written, an event or an interrupt arrives, or an
// absolute counter deadline passes. This is the Linux kernel's __cmpwait
// sequence with a timeout. WFET is an instruction, not a system call. The
// thread never leaves its core, so the scheduler sees no idle core to give
// to another thread and nothing has to be woken.
//
// A park can end early for reasons that have nothing to do with the word:
// the kernel's timer event stream, an interrupt, a write to a neighbour on
// the same line. A preemption between the load and the wait can also lose
// the wake, and then the deadline or the next event ends the park. Callers
// therefore loop: read the word again, check every other exit condition,
// park again. The deadline bounds the instruction, not the scheduling delay
// around it.
//
// That a store from another core ends a park is architectural. That a GPU
// store into pinned host memory does is a property of the platform, and
// tests/cuda/flag_protocol_test.cu measures it, so a driver or firmware
// change that loses it fails a test.
//
// The word is read through its object representation with an exclusive
// load, as the kernel's smp_cond_load_acquire reads one. It must be a
// naturally aligned integer of 1, 4 or 8 bytes (or a lock-free std::atomic
// of one), written only by stores or read-modify-writes of the whole word:
// CPU atomics, the GPU's st.release.sys, NIC DMA. On the LP64 Linux ABI a
// lock-free std::atomic<T> has T's size, alignment and representation. The
// assembly is invisible to TSan, and the callers' acquire re-reads carry the
// ordering.
//
// supported() is false off AArch64 Linux and on a CPU without FEAT_WFxT.
// The loops below then poll as they did before parking existed, and a
// caller can ask for that path by name, which is how the tests reach it on
// hardware that parks.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <type_traits>
#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#endif

namespace dgpp::cpu_wait {

#if defined(__aarch64__) && defined(__linux__)

#ifndef HWCAP2_WFXT
#define HWCAP2_WFXT (1UL << 31)
#endif

inline bool supported() {
  static const bool on = (getauxval(AT_HWCAP2) & HWCAP2_WFXT) != 0;
  return on;
}

// The counter WFET's deadline is compared against.
inline uint64_t ticks() {
  uint64_t t;
  asm volatile("isb\n\tmrs %0, cntvct_el0" : "=r"(t)::"memory");
  return t;
}

inline uint64_t ticks_hz() {
  static const uint64_t f = [] {
    uint64_t hz;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(hz));
    return hz != 0 ? hz : uint64_t{1000000000};
  }();
  return f;
}

// Parks while *p still holds `old`, until `deadline` (in ticks()) at the
// latest. Returns at once when the value already differs. Requires
// supported().
template <class T>
inline void park_while_equal(const T* p, T old, uint64_t deadline) {
  static_assert(std::is_integral_v<T>, "integral words only");
  static_assert(alignof(T) == sizeof(T), "naturally aligned words only");
  uint64_t tmp;
  if constexpr (sizeof(T) == 8) {
    asm volatile(
        "mov x9, %[dl]\n\t"
        "sevl\n\t"
        "wfe\n\t"                      // consume a stale event
        "ldxr %[tmp], %[v]\n\t"        // arm the monitor on the line
        "eor %[tmp], %[tmp], %[old]\n\t"
        "cbnz %[tmp], 1f\n\t"          // already changed: do not park
        ".inst 0xd5031009\n"           // wfet x9
        "1:"
        : [tmp] "=&r"(tmp)
        : [v] "Q"(*p), [old] "r"(static_cast<uint64_t>(old)), [dl] "r"(deadline)
        : "x9", "memory");
  } else if constexpr (sizeof(T) == 4) {
    asm volatile(
        "mov x9, %[dl]\n\t"
        "sevl\n\t"
        "wfe\n\t"
        "ldxr %w[tmp], %[v]\n\t"
        "eor %w[tmp], %w[tmp], %w[old]\n\t"
        "cbnz %w[tmp], 1f\n\t"
        ".inst 0xd5031009\n"
        "1:"
        : [tmp] "=&r"(tmp)
        : [v] "Q"(*p), [old] "r"(static_cast<uint32_t>(old)), [dl] "r"(deadline)
        : "x9", "memory");
  } else if constexpr (sizeof(T) == 1) {
    asm volatile(
        "mov x9, %[dl]\n\t"
        "sevl\n\t"
        "wfe\n\t"
        "ldxrb %w[tmp], %[v]\n\t"
        "eor %w[tmp], %w[tmp], %w[old]\n\t"
        "cbnz %w[tmp], 1f\n\t"
        ".inst 0xd5031009\n"
        "1:"
        : [tmp] "=&r"(tmp)
        : [v] "Q"(*p), [old] "r"(static_cast<uint32_t>(static_cast<uint8_t>(old))),
          [dl] "r"(deadline)
        : "x9", "memory");
  } else {
    static_assert(sizeof(T) == 1 || sizeof(T) == 4 || sizeof(T) == 8, "1, 4 or 8 bytes");
  }
  (void)tmp;
}

#else  // no WFET: the loops below poll

inline bool supported() { return false; }
inline uint64_t ticks() { return 0; }
inline uint64_t ticks_hz() { return 1; }
template <class T>
inline void park_while_equal(const T*, T, uint64_t) {}

#endif

// An absolute park deadline `us` microseconds from now (at most a few
// seconds: the product must fit 64 bits).
inline uint64_t deadline_in_us(uint64_t us) { return ticks() + us * ticks_hz() / 1000000; }

// The same park on a lock-free std::atomic, read through its object
// representation.
template <class T>
inline void park_while_equal(const std::atomic<T>& a, T old, uint64_t deadline) {
  static_assert(std::atomic<T>::is_always_lock_free && sizeof(std::atomic<T>) == sizeof(T) &&
                    alignof(std::atomic<T>) == sizeof(T),
                "a plain, naturally aligned lock-free word");
  if constexpr (std::is_same_v<T, bool>) {
    park_while_equal(reinterpret_cast<const uint8_t*>(&a), static_cast<uint8_t>(old), deadline);
  } else {
    park_while_equal(reinterpret_cast<const T*>(&a), old, deadline);
  }
}

// The longest one park lasts inside a wait loop. A loop's other exit
// conditions (a failure flag, a stop, a timeout) do not live on the parked
// line, so this is how late the loop can notice one of them.
inline constexpr uint64_t kWaitSliceUs = 1000;

// The processor's hint for a polling loop. It costs a few cycles and gives
// up nothing to the scheduler.
inline void spin_hint() {
#if defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#elif defined(__x86_64__)
  asm volatile("pause" ::: "memory");
#endif
}

// One step of a wait loop on `word`, last read as `seen`: a park of at most
// kWaitSliceUs where the CPU parks, the spin hint elsewhere. The caller
// re-reads the word and re-checks its exit conditions after every step.
template <class T>
inline void wait_step(const std::atomic<T>& word, T seen, bool park = supported()) {
  if (park && supported())
    park_while_equal(word, seen, deadline_in_us(kWaitSliceUs));
  else
    spin_hint();
}

// Waits until the counter at `word` reaches `want`. Returns false when
// `timeout` passes first. The counter only grows, and its writer stores the
// whole word (the GPU's publish kernels do, with st.release.sys).
//
// Where the CPU parks, the thread parks on the counter a slice at a time.
// Elsewhere it polls and offers its core to the scheduler every 256 polls.
// `park` false asks for the polling loop on any CPU.
inline bool wait_until_at_least(const uint64_t* word, uint64_t want,
                                std::chrono::milliseconds timeout, bool park = supported()) {
  if (park && supported()) {
    const uint64_t hz = ticks_hz();
    const uint64_t slice = kWaitSliceUs * hz / 1000000;
    const uint64_t end = ticks() + static_cast<uint64_t>(std::max<int64_t>(timeout.count(), 0)) * hz / 1000;
    for (;;) {
      const uint64_t seen = __atomic_load_n(word, __ATOMIC_ACQUIRE);
      if (seen >= want) return true;
      const uint64_t now = ticks();
      if (now >= end) return false;
      park_while_equal(word, seen, std::min(now + slice, end));
    }
  }
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  uint64_t polls = 0;
  for (;;) {
    if (__atomic_load_n(word, __ATOMIC_ACQUIRE) >= want) return true;
    if ((++polls & 255) == 0) {
      if (std::chrono::steady_clock::now() > deadline) return false;
      std::this_thread::yield();
    }
  }
}

}  // namespace dgpp::cpu_wait
