// The latent cache append and the decode tail's row work as device bodies
// (2026-11-01): the production kernels in dsa.cu (latent_append_fp8block_kernel,
// decode_tail_fp8block_kernel) run them one block a row, and the step kernel's
// kv row-tail unit (kernels/step/stream.cu) runs the same bodies for a step's
// rows, so the ring a step appends is bitwise the production's.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "kernels/latent_format.hpp"

namespace dgpp::dsa_tail {

__device__ __forceinline__ int64_t latent_physical_slot(const int32_t* req_ids, const int64_t* pos, const int32_t* block_tables,
                                                        int blocks_per_request, int block_tokens, int64_t i) {
  const int64_t p = pos[i];
  if (p < 0) return -1;
  const int32_t req = req_ids[i];
  const int32_t blk = block_tables[int64_t(req) * blocks_per_request + p / block_tokens];
  return int64_t(blk) * block_tokens + (p % block_tokens);
}

// The bf16 rope tail after the row's payload (rope a multiple of 8; the
// payload's byte length a multiple of 16 in every format).
__device__ __forceinline__ void latent_append_tail(const uint16_t* rope_rows, int rope, int64_t i, uint8_t* row_tail) {
  if (rope <= 0) return;
  const uint4* s4 = reinterpret_cast<const uint4*>(rope_rows + i * rope);
  uint4* d4 = reinterpret_cast<uint4*>(row_tail);
  for (int j = threadIdx.x; j < rope / 8; j += blockDim.x) d4[j] = s4[j];
}

// fp8_block: thread t quantizes elements [8t, 8t+8) with the e8m0 scale of
// its 32-element group (four consecutive lanes; kv_lora <= 1024, a
// multiple of 32) — the reference's act_quant(block 32, ue8m0) bitwise.
// One row's fp8-block append (2026-10-20: the decode tail runs it behind
// the row's rope and slot work).
__device__ __forceinline__ void latent_append_fp8block_row(const uint16_t* latent_rows, const int32_t* req_ids, const int64_t* pos,
                                                           const int32_t* block_tables, int blocks_per_request, int block_tokens,
                                                           uint8_t* latent_cache, int kv_lora, size_t cache_row_bytes,
                                                           const uint16_t* rope_rows, int rope, int64_t i) {
  const int64_t phys = latent_physical_slot(req_ids, pos, block_tables, blocks_per_request, block_tokens, i);
  if (phys < 0) return;
  const size_t row_bytes = latent_row_bytes(LatentFormat::kFp8Block, kv_lora);  // the payload
  uint8_t* row = latent_cache + phys * static_cast<int64_t>(cache_row_bytes);
  latent_append_tail(rope_rows, rope, i, row + row_bytes);
  const size_t scale_off = latent_fp8_block_scale_offset(kv_lora);
  if (threadIdx.x == 0)
    for (size_t k = scale_off + static_cast<size_t>(kv_lora / kLatentFp8BlockGroup); k < row_bytes; ++k) row[k] = 0;
  const int e0 = threadIdx.x * 8;
  const bool active = e0 < kv_lora;
  float v[8];
  float amax = 0.0f;
  if (active) {
    const uint4 raw = *reinterpret_cast<const uint4*>(latent_rows + i * kv_lora + e0);
    const uint32_t* w = reinterpret_cast<const uint32_t*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[2 * j] = bf16_bits_to_float(static_cast<uint16_t>(w[j] & 0xFFFFu));
      v[2 * j + 1] = bf16_bits_to_float(static_cast<uint16_t>(w[j] >> 16));
      amax = fmaxf(amax, fmaxf(fabsf(v[2 * j]), fabsf(v[2 * j + 1])));
    }
  }
  // The group's absmax across its four lanes (exact, order-free).
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  if (!active) return;
  const uint8_t sb = latent_fp8_block_scale_byte(amax);
  if ((threadIdx.x & 3) == 0) row[scale_off + static_cast<size_t>(e0 / kLatentFp8BlockGroup)] = sb;
  const float s = e8m0_byte_to_float(sb);
  uint2 packed;
  packed.x = 0;
  packed.y = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) packed.x |= static_cast<uint32_t>(latent_fp8_block_encode(v[j], s)) << (8 * j);
#pragma unroll
  for (int j = 0; j < 4; ++j) packed.y |= static_cast<uint32_t>(latent_fp8_block_encode(v[4 + j], s)) << (8 * j);
  *reinterpret_cast<uint2*>(row + e0) = packed;
}

// The decode tail's slot work for row r at position p: the row's ring slot
// (ring_slot_positions_kernel), its window slot list (window_slots_decode_kernel:
// the last `window` positions ascending, -1 past the count) and its entry
// position when the layer has a ratio (entry_positions_kernel; entry_out
// nullptr otherwise). The caller syncs before the append reads the slot.
__device__ __forceinline__ void decode_tail_slots(int64_t r, int64_t p, int64_t* slots, int ring_slots, int window, int32_t* wlist,
                                                  int32_t* wcounts, int64_t* entry_out, int ratio) {
  if (threadIdx.x == 0) {
    slots[r] = p < 0 ? -1 : p % ring_slots;
    if (entry_out != nullptr) entry_out[r] = p < 0 ? -1 : (p + 1) / ratio - 1;
  }
  const int n = p < 0 ? 0 : int(min(int64_t(window), p + 1));
  for (int k = threadIdx.x; k < window; k += blockDim.x) {
    const int64_t qp = p - (n - 1) + k;  // ascending positions
    wlist[r * window + k] = k < n ? int32_t(qp % ring_slots) : -1;
  }
  if (threadIdx.x == 0) wcounts[r] = n;
}

}  // namespace dgpp::dsa_tail
