#pragma once
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "adapters/runtime.hpp"
#include "engine/tp_bus.hpp"
#include "kernels/low_rank.hpp"

namespace dgpp::adapters {
// Serving uses the same FP32 rank order as the research fold, with packing
// and reconstruction on the GPU so decode can record the existing bus nodes.
inline SumFp32Stream bus_sum_stream(net::CollectiveBus& bus, int rank, int world, size_t capacity,
                                    int timeout_ms) {
  struct Scratch {
    uint16_t* digits = nullptr;
    ~Scratch() { cudaFree(digits); }
  };
  if (world < 2 || world > 4 || rank < 0 || rank >= world || capacity == 0 ||
      capacity > SIZE_MAX / (world * 4 * sizeof(uint16_t)))
    throw std::invalid_argument("runtime adapter stream fold geometry");
  auto scratch = std::make_shared<Scratch>();
  DGPP_CUDA_OK(cudaMalloc(&scratch->digits, capacity * world * 4 * sizeof(uint16_t)));
  return [&bus, rank, world, capacity, timeout_ms, scratch](float* values, size_t count,
                                                            cudaStream_t stream, bool capture) {
    if (!values || count == 0 || count > capacity)
      throw std::invalid_argument("runtime adapter stream fold exceeds budget");
    low_rank_pack_digits(values, scratch->digits, count, rank, world, stream);
    const size_t elems = count * world * 4;
    std::string error;
    if (capture) {
      const size_t chunk = bus_latency_slot_elems(bus);
      for (size_t offset = 0; offset < elems; offset += chunk)
        if (!bus.allreduce_record(stream, scratch->digits + offset, scratch->digits + offset,
                                  std::min(chunk, elems - offset), &error))
          throw std::runtime_error("runtime adapter graph fold: " + error);
    } else {
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      // Rank-space buffers are small even for long prefills. Use latency
      // chunks: the bulk machine assumes a shard for every participating
      // rank, which these buffers need not provide.
      const size_t chunk = bus_latency_slot_elems(bus);
      for (size_t offset = 0; offset < elems; offset += chunk) {
        const auto id = bus.allreduce(scratch->digits + offset, scratch->digits + offset,
                                      std::min(chunk, elems - offset), &error);
        if (!id) throw std::runtime_error("runtime adapter fold: " + error);
        const auto result = bus.wait_allreduce(id, timeout_ms);
        if (!result.ok) throw std::runtime_error("runtime adapter fold: " + result.error);
      }
    }
    low_rank_sum_digits(scratch->digits, values, count, world, stream);
  };
}

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
