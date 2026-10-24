// Pinned-memory flag protocol: the completion-notification contract for
// CollectiveBus on GB10's unified-memory fabric.
//
// Contract (validated by tests/cuda/flag_protocol_test.cu):
//   * A producer writes payload first, then bumps `start->seq`. Host producers
//     use a release store; the device observes with a system-scope acquire,
//     consumes payload, stamps cycles/hash, and publishes `ack->seq` with a
//     system-scope release. The host observes acks with acquire loads.
//   * Consequence: if a host thread sees ack.seq == S, then the payload for
//     S was fully visible to the device when it ran — the ordering invariant
//     the bus receive path relies on.
//   * NIC doorbells are a hardware-coherency contract rather than a C++
//     release operation; the M0 bench (micro_ibv_smoke verify) and the
//     transport regression command (nic_regress, via src/net/nic_visibility)
//     validate ordered RC DMA payload + doorbell visibility on the deployed
//     driver/firmware stack.
//   * The persistent kernels are inactivity-watchdog bounded. The deadline
//     resets after traffic, and a reserved sequence shuts them down cleanly.
//
// Header-only by design: included from exactly one TU each (bench, test,
// later the bus itself). Keep the kernels single-thread/single-block —
// the protocol must stay minimal; bulk data movement is the caller's job.
#pragma once

#include <cuda/atomic>
#include "kernels/pdl.cuh"
#include <cuda_runtime.h>

#include <cstdint>

#include "kernels/flag_protocol_types.hpp"

namespace dgpp {

__device__ inline void flag_poll_pause() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  __nanosleep(32);
#endif
}

__device__ inline uint32_t flag_load_acquire(uint32_t* seq) {
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ref(*seq);
  return ref.load(cuda::memory_order_acquire);
}

__device__ inline void flag_store_release(uint32_t* seq, uint32_t value) {
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ref(*seq);
  ref.store(value, cuda::memory_order_release);
}

// Persistent heartbeat: for every new start seq S, acks with S and a device
// cycle stamp. Exits silently once `deadline_cycles` elapse without traffic.
// Sequences start at 1; 0 means "idle".
__global__ inline void flag_heartbeat_kernel(uint32_t* start,
                                             FlagAck* ack,
                                             uint64_t deadline_cycles) {
  dgpp_pdl_wait();
  uint64_t idle_since = clock64();
  uint32_t last_seen = 0;
  for (;;) {
    uint32_t s = flag_load_acquire(start);
    while (s == last_seen) {
      if (clock64() - idle_since > deadline_cycles) return;
      flag_poll_pause();
      s = flag_load_acquire(start);
    }
    if (s == kFlagStopSequence) return;
    last_seen = s;
    ack->cycles = clock64();
    ack->hash = 0;
    flag_store_release(&ack->seq, s);
    idle_since = clock64();
  }
}

// Persistent payload variant: on new seq, folds 16 u32 words (64 B, one
// cache line) into a checksum before acking — the receive-path shape where
// payload visibility must be proven before completion is signalled.
__global__ inline void flag_payload_kernel(uint32_t* start,
                                           const uint32_t* payload,
                                           FlagAck* ack,
                                           uint64_t deadline_cycles) {
  dgpp_pdl_wait();
  uint64_t idle_since = clock64();
  uint32_t last_seen = 0;
  for (;;) {
    uint32_t s = flag_load_acquire(start);
    while (s == last_seen) {
      if (clock64() - idle_since > deadline_cycles) return;
      flag_poll_pause();
      s = flag_load_acquire(start);
    }
    if (s == kFlagStopSequence) return;
    last_seen = s;
    uint64_t h = 0;
    // Checksum: xor of (hi<<32 | lo) per u32 pair, mixed with the golden
    // ratio constant. Host side MUST mirror this fold exactly (see test).
#pragma unroll
    for (int i = 0; i < 16; i += 2) {
      const uint64_t pair = (static_cast<uint64_t>(payload[i]) << 32) |
                            static_cast<uint64_t>(payload[i + 1]);
      h ^= pair * 0x9E3779B97F4A7C15ull;
    }
    ack->cycles = clock64();
    ack->hash = h;
    flag_store_release(&ack->seq, s);
    idle_since = clock64();
  }
}

}  // namespace dgpp
