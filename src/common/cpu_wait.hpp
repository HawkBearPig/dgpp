#pragma once

// Low-power waits on a word that another agent (a CPU thread, the GPU or
// the NIC) writes. On aarch64 with FEAT_WFxT (GB10's Cortex-X925/A725) the
// waiter arms the exclusive monitor on the word's line (LDXR) and parks in
// WFET until that line is written, an event or interrupt arrives, or an
// absolute CNTVCT deadline passes: the Linux kernel's __cmpwait sequence
// with a timeout. The thread never leaves its CPU (WFET is an instruction,
// not a syscall), so the scheduler sees no idle core and nothing has to be
// woken: on GB10 a GPU store reached a parked waiter in 1.50/2.25 us
// (p50/p99) against a spin's 1.43/1.73 us, at roughly a sixth of the
// spin's extra wall power. Every park may return early for unrelated
// reasons (other events, interrupts, the timer event stream, a preemption
// or migration that drops the reservation, after which only the deadline
// or another event ends the park); callers re-read the word and
// re-check their exit conditions after each one. The deadline bounds the
// instruction, not scheduling delay around it.
//
// The words are read through their object representation with an
// exclusive load, as the kernel's smp_cond_load_acquire does: naturally
// aligned 1/4/8-byte integers (or lock-free std::atomic of them) written
// only by stores or read-modify-writes of the whole word (CPU atomics,
// GPU st.release.sys, NIC DMA), on the LP64 Linux ABI where a lock-free
// std::atomic<T> has T's size, alignment and representation. The asm is
// invisible to TSan; the callers' acquire re-reads carry the ordering.
//
// enabled() is false without FEAT_WFxT, off aarch64 Linux, or with
// DGPP_WFE_WAITS=0; callers then keep their existing spin.

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#endif

namespace dgpp::cpu_wait {

#if defined(__aarch64__) && defined(__linux__)

#ifndef HWCAP2_WFXT
#define HWCAP2_WFXT (1UL << 31)
#endif

inline bool enabled() {
  static const bool on = [] {
    const char* e = std::getenv("DGPP_WFE_WAITS");
    if (e != nullptr && std::strcmp(e, "0") == 0) return false;
    return (getauxval(AT_HWCAP2) & HWCAP2_WFXT) != 0;
  }();
  return on;
}

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

// An absolute park deadline `us` (at most a few seconds) microseconds from now.
inline uint64_t deadline_in_us(uint64_t us) { return ticks() + us * ticks_hz() / 1000000; }

// Park while *p still holds `old`, until `deadline` (CNTVCT ticks) at the
// latest. Returns at once when the value already differs. Requires
// enabled().
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

#else  // no WFET: callers keep their spin

inline bool enabled() { return false; }
inline uint64_t ticks() { return 0; }
inline uint64_t deadline_in_us(uint64_t) { return 0; }
template <class T>
inline void park_while_equal(const T*, T, uint64_t) {}

#endif

// The same on a lock-free std::atomic (read through its object
// representation, as the kernel's smp_cond_load_acquire reads a word).
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

// Default bound for one park in a wait loop whose other exit conditions
// (failure flags, stop, backstop deadlines) are not on the parked line:
// they are noticed within this long.
inline constexpr uint64_t kWaitSliceUs = 1000;

}  // namespace dgpp::cpu_wait
