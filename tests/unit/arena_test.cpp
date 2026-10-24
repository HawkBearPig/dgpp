#include <cuda_runtime.h>

#include <cstring>
#include <limits>

#include "common/cuda_check.hpp"
#include "common/test.hpp"
#include "core/arena.hpp"

namespace {
bool cuda_available() {
  int n = 0;
  cudaError_t e = cudaGetDeviceCount(&n);
  return e == cudaSuccess && n > 0;
}
}  // namespace

// The steady-state invariant M1 promises: repeated scratch usage after a
// reset performs no new underlying allocations — asserted by
// arena_slab_accounting_stable_across_reset_cycle below.

namespace {

// One warmup call performs the persistent allocations once; every later call
// only exercises the resettable scratch region.
void arena_body(dgpp::Arena& a, bool warm) {
  if (warm) {
    void* p1 = a.alloc_persistent(dgpp::MemClass::HostPinned, 1024);
    std::memset(p1, 0xAB, 1024);
  }
  void* s1 = a.alloc_scratch(dgpp::MemClass::DeviceHot, 4096);
  void* s2 = a.alloc_scratch(dgpp::MemClass::DeviceHot, 8192);
  a.reset_scratch();
  void* s1b = a.alloc_scratch(dgpp::MemClass::DeviceHot, 4096);
  // Deterministic bump placement after reset:
  if (s1 != s1b) throw std::runtime_error("scratch not reset to base");
  (void)s2;
  a.reset_scratch();
}
}  // namespace

DGPP_TEST(arena_slab_accounting_stable_across_reset_cycle) {
  if (!cuda_available()) {
    dgpp::test::skip("no CUDA device");
  }
  dgpp::Arena a;
  a.init({.persistent_hot = 0,
          .scratch_hot = 1 << 20,
          .persistent_cold = 0,
          .scratch_cold = 0,
          .host_pinned = 1 << 16});
  arena_body(a, /*warm=*/true);

  uint64_t slabs_before = 0;
  size_t used_before = 0;
  for (auto& [name, st] : a.stats()) {
    (void)name;
    slabs_before += st->slab_mallocs;
    used_before += st->used;
  }
  for (int i = 0; i < 1000; ++i) arena_body(a, false);

  uint64_t slabs_after = 0;
  size_t used_after = 0;
  for (auto& [name, st] : a.stats()) {
    (void)name;
    slabs_after += st->slab_mallocs;
    used_after += st->used;
  }
  if (slabs_before != slabs_after)
    throw std::runtime_error("slab allocations grew in steady state");
  // Used bytes at reset points must be identical too (same bump trajectory).
  if (used_before != used_after)
    throw std::runtime_error("steady-state usage drifted");
}

DGPP_TEST(arena_alignment_guarantees) {
  if (!cuda_available()) {
    dgpp::test::skip("no CUDA device");
  }
  dgpp::Arena a;
  a.init({.persistent_hot = 0,
          .scratch_hot = 1 << 20,
          .host_pinned = 1 << 16});
  auto is_aligned = [](const void* p, size_t align) {
    return reinterpret_cast<uintptr_t>(p) % align == 0;
  };
  // odd-sized request followed by aligned request
  (void)a.alloc_persistent(dgpp::MemClass::HostPinned, 137);
  void* q = a.alloc_persistent(dgpp::MemClass::HostPinned, 8);
  if (!is_aligned(q, 256)) throw std::runtime_error("pinned alignment lost");
  void* d = a.alloc_scratch(dgpp::MemClass::DeviceHot, 33);
  if (!is_aligned(d, 256)) throw std::runtime_error("device alignment lost");
  (void)a.alloc_scratch(dgpp::MemClass::DeviceHot, 137, 256);
  void* page_aligned =
      a.alloc_scratch(dgpp::MemClass::DeviceHot, 8, 4096);
  if (!is_aligned(page_aligned, 4096))
    throw std::runtime_error("alignment ignored slab base address");
}

DGPP_TEST(arena_rejects_invalid_geometry_and_double_init) {
  if (!cuda_available()) {
    dgpp::test::skip("no CUDA device");
  }
  dgpp::Arena a;
  a.init({.scratch_hot = 1 << 20});

  for (size_t alignment : {size_t{0}, size_t{3}}) {
    bool rejected = false;
    try {
      (void)a.alloc_scratch(dgpp::MemClass::DeviceHot, 8, alignment);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected) throw std::runtime_error("invalid alignment was accepted");
  }

  bool overflow_rejected = false;
  try {
    (void)a.alloc_array_scratch<uint64_t>(
        dgpp::MemClass::DeviceHot, std::numeric_limits<size_t>::max());
  } catch (const std::overflow_error&) {
    overflow_rejected = true;
  }
  if (!overflow_rejected) throw std::runtime_error("array overflow was accepted");

  bool reinit_rejected = false;
  try {
    a.init({.scratch_hot = 1 << 20});
  } catch (const std::logic_error&) {
    reinit_rejected = true;
  }
  if (!reinit_rejected) throw std::runtime_error("double init was accepted");

  a.release_all();
  a.init({.scratch_hot = 4096});
  (void)a.alloc_scratch(dgpp::MemClass::DeviceHot, 16);
}
