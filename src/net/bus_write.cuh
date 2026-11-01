#pragma once
// The one-sided latency transport's device side (2026-10-19; kernels/bus_kernel.cu
// until 2026-11-01): a row of self-describing 512-byte chunks (31 data lanes and a
// trailer lane of hash, generation and chunk index) staged into the generation's
// pinned row for the engine's RDMA WRITE, then every peer's copy polled in its
// write ring, proven by its trailer, folded in canonical rank order. The two
// phases are block bodies taking the block's index in the row's grid, so the
// production kernel (bus_allreduce_write_kernel) and the step engine's fold
// units (kernels/step/stream.cu) run the same code: a fold's result is bitwise
// the eager machine's whichever kernel carries it.
#include <cstdint>

#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "net/bus_kernel.hpp"
#include "net/bus_types.hpp"

namespace dgpp::net::bus_write {

constexpr int kThreads = 32 * static_cast<int>(kBusWriteWarpsPerBlock);
constexpr uint32_t kPollSleepNs = 400;  // between a warp's polls (the poll RATE is what slows the NIC's landing)

__device__ __forceinline__ uint64_t globaltimer_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ inline uint64_t sys_load_u64(const uint64_t* p) {
  cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ref(*const_cast<uint64_t*>(p));
  return ref.load(cuda::memory_order_relaxed);
}
// 16-byte system-scope load of NIC-placed memory (the placement-visibility discipline).
__device__ inline uint4 sys_load_u128(const uint4* p) {
  uint4 v;
  asm volatile("ld.relaxed.sys.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ inline bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }
__device__ __forceinline__ float bf16_to_f32(uint16_t v) {
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&v));
}

// A data lane's two words mixed with their positions (the positional fold
// of bus_fold64, per lane): XOR-combined over the warp, order-free.
__device__ __forceinline__ uint64_t lane_mix(const uint4& v, uint32_t lane) {
  const uint64_t w0 = static_cast<uint64_t>(v.x) | (static_cast<uint64_t>(v.y) << 32);
  const uint64_t w1 = static_cast<uint64_t>(v.z) | (static_cast<uint64_t>(v.w) << 32);
  return ((w0 + 2 * lane + 1) * kFoldMultiplier) ^ ((w1 + 2 * lane + 2) * kFoldMultiplier);
}
__device__ __forceinline__ uint64_t warp_xor_u64(uint64_t h) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) h ^= __shfl_xor_sync(0xffffffffu, h, o);
  return h;
}
__device__ __forceinline__ uint64_t chunk_hash(uint64_t lanes_xor, uint32_t gen, uint32_t chunk) {
  return lanes_xor ^ (((static_cast<uint64_t>(gen) << 32) | chunk) * kFoldMultiplier);
}
// A data lane's 16 bytes of chunk `chunk` from an `elems`-long bf16 vector
// (zero past the end; lane 31, the trailer lane, reads nothing).
__device__ __forceinline__ uint4 load_lane(const uint16_t* src, uint32_t elems, uint32_t chunk, uint32_t lane, bool vec) {
  const uint32_t e0 = chunk * kBusWriteChunkElems + lane * 8;
  if (lane >= kBusWriteDataLanes || e0 >= elems) return make_uint4(0, 0, 0, 0);
  if (vec && e0 + 8 <= elems) return reinterpret_cast<const uint4*>(src)[e0 / 8];
  uint32_t w[4] = {0, 0, 0, 0};
  for (uint32_t i = 0; i < 8 && e0 + i < elems; ++i) w[i / 2] |= static_cast<uint32_t>(src[e0 + i]) << (16 * (i & 1));
  return make_uint4(w[0], w[1], w[2], w[3]);
}
__device__ __forceinline__ void store_lane(uint16_t* dst, uint32_t elems, uint32_t chunk, uint32_t lane, bool vec, const uint4& v) {
  const uint32_t e0 = chunk * kBusWriteChunkElems + lane * 8;
  if (lane >= kBusWriteDataLanes || e0 >= elems) return;
  if (vec && e0 + 8 <= elems) {
    reinterpret_cast<uint4*>(dst)[e0 / 8] = v;
    return;
  }
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
  for (uint32_t i = 0; i < 8 && e0 + i < elems; ++i) dst[e0 + i] = static_cast<uint16_t>(w[i / 2] >> (16 * (i & 1)));
}
// A chunk as read: 2 = proven (its trailer carries (gen, chunk) and the data
// lanes fold to its hash), 1 = the trailer is here but the data is not yet (a
// torn read), 0 = not landed.
__device__ __forceinline__ int chunk_state(const uint4& val, uint32_t lane, uint32_t gen, uint32_t c) {
  const uint64_t h = chunk_hash(warp_xor_u64(lane < kBusWriteDataLanes ? lane_mix(val, lane) : 0), gen, c);
  const uint32_t t_lo = __shfl_sync(0xffffffffu, val.x, kBusWriteDataLanes);
  const uint32_t t_hi = __shfl_sync(0xffffffffu, val.y, kBusWriteDataLanes);
  const uint32_t t_gen = __shfl_sync(0xffffffffu, val.z, kBusWriteDataLanes);
  const uint32_t t_chunk = __shfl_sync(0xffffffffu, val.w, kBusWriteDataLanes);
  if (t_gen != gen || t_chunk != c) return 0;
  return ((static_cast<uint64_t>(t_hi) << 32) | t_lo) == h ? 2 : 1;
}

// The block's chunk count in the row's grid of `nblocks` blocks: c = gwarp + k * nwarps over its warps.
__device__ __forceinline__ int block_chunks(int bx, int nblocks, uint32_t chunks) {
  constexpr int kWarps = kThreads / 32;
  const uint32_t nwarps = static_cast<uint32_t>(nblocks) * kWarps;
  int n = 0;
  for (int w = 0; w < kWarps; ++w) {
    const uint32_t g = static_cast<uint32_t>(bx) * kWarps + w;
    if (g < chunks) n += static_cast<int>((chunks - 1 - g) / nwarps) + 1;
  }
  return n;
}

// Phase 1: block `bx` of `nblocks` lays its chunks of `src` (an `elems`-long
// bf16 vector) out as self-describing chunks in generation `gen`'s staging row,
// fences once and publishes its record's ready (the engine posts on every
// block's). Call from every thread of a kThreads block.
__device__ inline void stage_block(int bx, int nblocks, const BusAllReduceGraphView& v, uint32_t gen, const uint16_t* src,
                                   uint32_t elems, BusWriteBlockRecord& rec) {
  constexpr int kWarps = kThreads / 32;
  const int warp = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t gwarp = static_cast<uint32_t>(bx) * kWarps + warp;
  const uint32_t nwarps = static_cast<uint32_t>(nblocks) * kWarps;
  const uint32_t chunks = (elems + kBusWriteChunkElems - 1) / kBusWriteChunkElems;
  const bool vec = aligned16(src);
  const uint32_t row = (gen - 1) % kBusWriteRing;
  uint8_t* stage = reinterpret_cast<uint8_t*>(v.stage_row_base) + static_cast<size_t>(row) * v.stage_row_bytes;
  for (uint32_t c = gwarp; c < chunks; c += nwarps) {
    uint4 val = load_lane(src, elems, c, lane, vec);
    const uint64_t h = chunk_hash(warp_xor_u64(lane < kBusWriteDataLanes ? lane_mix(val, lane) : 0), gen, c);
    if (lane == kBusWriteDataLanes) val = make_uint4(static_cast<uint32_t>(h), static_cast<uint32_t>(h >> 32), gen, c);
    reinterpret_cast<uint4*>(stage + static_cast<size_t>(c) * kBusWriteChunkBytes)[lane] = val;
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    rec.gt_stage = globaltimer_ns();
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ready(rec.ready);
    ready.store(gen, cuda::memory_order_release);
  }
}

// Phase 2's shared memory (the production kernel's static block; the step engine's carved).
struct FoldSmem {
  int peer_done[kBusMaxPeers];  // chunks of this block proved per peer
  int abort;
  unsigned gate_waits, gate_spins;
};

// Phase 2: block `bx`'s chunks, every peer's copy polled in flight in its
// write ring, proven by its trailer, folded in canonical rank order with the
// block's own `src` and stored to `dst`; the record's done published with the
// status. Returns false when the block gave up (the deadline from
// `start_clock`, or the cell's done_seq poisoned to `gen`). Call from every
// thread of a kThreads block, `sm` zeroed by the caller.
__device__ inline bool fold_block(int bx, int nblocks, const BusAllReduceGraphView& v, int my_rank, uint32_t gen, const uint16_t* src,
                                  uint16_t* dst, uint32_t elems, BusAllReduceCtl* cell, BusWriteBlockRecord& rec,
                                  uint64_t deadline_cycles, uint64_t start_clock, FoldSmem& sm) {
  constexpr int kWarps = kThreads / 32;
  const int warp = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t gwarp = static_cast<uint32_t>(bx) * kWarps + warp;
  const uint32_t nwarps = static_cast<uint32_t>(nblocks) * kWarps;
  const uint32_t chunks = (elems + kBusWriteChunkElems - 1) / kBusWriteChunkElems;
  const bool vec = aligned16(src) && aligned16(dst);
  const int world = v.send_peers + 1;
  const int lanes = v.lanes_per_peer;
  const uint32_t row = (gen - 1) % kBusWriteRing;
  const int nchunks = block_chunks(bx, nblocks, chunks);
  auto abort_check = [&](uint32_t spin, uint32_t every_mask) -> bool {
    if ((spin & every_mask) != every_mask) return false;
    int abort = 0;
    if (lane == 0) {
      abort = *reinterpret_cast<volatile int*>(&sm.abort);
      if (abort == 0 && (clock64() - start_clock > deadline_cycles || sys_load_u64(&cell->done_seq) == gen)) {
        abort = 1;
        *reinterpret_cast<volatile int*>(&sm.abort) = 1;
      }
    }
    return __shfl_sync(0xffffffffu, abort, 0) != 0;
  };
  bool failed = false;
  for (uint32_t c = gwarp; c < chunks && !failed; c += nwarps) {
    const int cl = bus_write_lane_of(chunks, lanes, c);
    uint4 in[kBusMaxPeers];
    const uint4* ptrs[kBusMaxPeers];
    bool proven[kBusMaxPeers];
    for (int p = 0; p < static_cast<int>(kBusMaxPeers); ++p) {
      ptrs[p] = nullptr;
      proven[p] = p >= v.send_peers;
      if (p >= v.send_peers) continue;
      const BusRecvView& rv = v.recv[p * lanes + cl];
      ptrs[p] = reinterpret_cast<const uint4*>(rv.write_ring + static_cast<size_t>(row) * rv.write_row_bytes +
                                               static_cast<size_t>(c) * kBusWriteChunkBytes) + lane;
    }
    const uint4 mine = load_lane(src, elems, c, lane, vec);
    for (uint32_t spin = 0;; ++spin) {
      for (int p = 0; p < static_cast<int>(kBusMaxPeers); ++p)
        if (!proven[p]) in[p] = sys_load_u128(ptrs[p]);
      bool all = true;
      bool torn = false;
      for (int p = 0; p < static_cast<int>(kBusMaxPeers); ++p) {
        if (proven[p]) continue;
        const int state = chunk_state(in[p], lane, gen, c);
        if (state == 1) torn = true;
        if (state == 2) {
          proven[p] = true;
          if (lane == 0) {
            const int before = atomicAdd(&sm.peer_done[p], 1);
            if (before == 0 && bx == 0 && spin > 0) cell->gt_landed[p] = globaltimer_ns();  // the row landed while this grid watched
            if (before == nchunks - 1) rec.gt_peer[p] = globaltimer_ns();  // this block's last chunk of p
          }
        } else {
          all = false;
        }
      }
      if (all) break;
      if (torn && lane == 0) {
        atomicAdd(&sm.gate_waits, 1u);
        atomicAdd(&sm.gate_spins, 1u);
      }
      if (abort_check(spin, 15u)) {
        failed = true;
        break;
      }
      __nanosleep(kPollSleepNs);
    }
    if (failed) break;
    float acc[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
    for (int r = 0; r < world; ++r) {
      const uint4 val = r == my_rank ? mine : in[r < my_rank ? r : r - 1];
      if (lane < kBusWriteDataLanes) {
        const uint32_t w[4] = {val.x, val.y, val.z, val.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] += bf16_to_f32(static_cast<uint16_t>((e & 1) ? (w[e >> 1] >> 16) : (w[e >> 1] & 0xFFFFu)));
      }
    }
    uint4 out;
    uint32_t o[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const __nv_bfloat16 lo = __float2bfloat16(acc[2 * q]);
      const __nv_bfloat16 hi = __float2bfloat16(acc[2 * q + 1]);
      o[q] = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&lo)) |
             (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&hi)) << 16);
    }
    out.x = o[0]; out.y = o[1]; out.z = o[2]; out.w = o[3];
    store_lane(dst, elems, c, lane, vec, out);
  }
  if (failed && lane == 0) *reinterpret_cast<volatile int*>(&sm.abort) = 1;
  __syncthreads();
  if (threadIdx.x == 0) {
    if (sm.gate_waits != 0) {
      atomicAdd_system(&cell->dbg_gate_waits, sm.gate_waits);
      atomicAdd_system(&cell->dbg_gate_spins, sm.gate_spins);
    }
    rec.status = sm.abort != 0 ? 1u : 0u;
    rec.gt_done = globaltimer_ns();
    __threadfence_system();
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system> done(rec.done);
    done.store(gen, cuda::memory_order_release);
  }
  return sm.abort == 0;
}

}  // namespace dgpp::net::bus_write
