#pragma once
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "adapters/runtime.hpp"
#include "engine/tp_bus.hpp"

namespace dgpp::adapters {
// Preserve FP32 partials through the existing lossless byte-digit transport,
// then sum in rank order. Allocated only for an enabled runtime adapter.
inline SumFp32 bus_sum(net::CollectiveBus& bus, int rank, int world, size_t capacity,
                       int timeout_ms) {
  struct Scratch {
    uint16_t* digits = nullptr;
    std::vector<float> gathered;
    ~Scratch() { cudaFreeHost(digits); }
  };
  if (world < 2 || world > 4 || rank < 0 || rank >= world || capacity == 0 ||
      capacity > 2048u * kMaxAdapterRank)
    throw std::invalid_argument("runtime adapter fold geometry");
  auto scratch = std::make_shared<Scratch>();
  DGPP_CUDA_OK(cudaHostAlloc(&scratch->digits,
                             sampling_gather_scratch_elems(capacity * world) * sizeof(uint16_t),
                             cudaHostAllocDefault));
  return [&bus, rank, world, capacity, timeout_ms, scratch](float* values, size_t count) {
    if (!values || count == 0 || count > capacity)
      throw std::invalid_argument("runtime adapter fold exceeds budget");
    bus_gather_logits(bus, rank, world, values, count, rank * count, world * count, scratch->digits,
                      timeout_ms, &scratch->gathered);
    for (size_t i = 0; i < count; ++i) {
      float value = scratch->gathered[i];
      for (int peer = 1; peer < world; ++peer) value += scratch->gathered[peer * count + i];
      if (!std::isfinite(value)) throw std::runtime_error("non-finite runtime adapter fold");
      values[i] = value;
    }
  };
}
}  // namespace dgpp::adapters
