// The step kernel's bf16-row consumer over a 12-bit companion (2026-11-01):
// the lm head. One block of kStepThreads runs 64 rows of the packed matrix,
// a warp kPerWarp rows at once, the activation rows staged a super-block
// (1024 columns) at a time in shared memory and every warp consuming its
// rows' super-block from registers: bf12_gemv_wide_kernel's form
// (kernels/bf12_gemv.cu) with the same bodies, consume_step and raw_window,
// so a logit is bitwise launch_bf12_gemv's and the bf16 GEMV launch's. The
// production's tensor-core form (mma_gemv.cu's bf12 decode) was the first
// head unit: its block body needs the kernel's whole register budget, and
// behind the latency units' call it cost the expert stream three points
// whether it ran or not (the hot loop spilled around the call, 55 percent
// more local loads; docs/step_engine.md, the token path). This form keeps
// about 120 registers a lane. The next row group's packed rows are
// prefetched into L2 while the current one is consumed, so the unit streams
// at the memory's share rather than a lane's loads in flight.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "kernels/bf12_gemv.cuh"
#include "kernels/gemv_common.cuh"
#include "kernels/step/fp8_dense.cuh"

namespace dgpp::step {

constexpr int kBf12RowsWarps = 8;                                // the step block's warps
constexpr int kBf12RowsPerWarp = 2;                              // packed rows a warp holds at once (acc[2][kRows])
constexpr int kBf12RowsGroup = kBf12RowsWarps * kBf12RowsPerWarp;  // rows a staging serves: 16
constexpr int kBf12RowsUnit = 64;                                // rows a unit
constexpr int kBf12RowsMaxRows = 32;                             // activation rows (kStepMaxRows)
static_assert(kBf12RowsUnit % kBf12RowsGroup == 0, "a unit is whole row groups");

// Shared memory: the activation window [kBf12RowsMaxRows][kBf12Super] bf16.
__host__ __device__ constexpr unsigned bf12_rows_smem_bytes() { return kBf12RowsMaxRows * kBf12Super * 2u; }

// Rows [n0, n0 + kBf12RowsUnit) of the packed matrix (n rows of k columns, k a
// whole number of super-blocks) over `rows` activation rows at `act`
// (stride act_stride, kRows rows readable), into out[r * out_stride + row]
// fp32. kRows (8, 16 or 32, the step's rows rounded up) sizes the staged
// window and the accumulators: at five rows the window is 16 KB and the
// chain reads a quarter of the shared memory it would at 32. The chain of a
// (row, activation row) pair is the production GEMV's whatever kRows:
// consume_step over the super-blocks in column order, then the warp's xor
// reduction.
template <int kRows>
__device__ __forceinline__ void bf12_rows_unit(const uint8_t* __restrict__ packed, const uint32_t* __restrict__ rowtab,
                                               const uint32_t* __restrict__ esc, const uint16_t* __restrict__ raw, int n, int k,
                                               const uint16_t* __restrict__ act, size_t act_stride, int rows, int n0, float* __restrict__ out,
                                               size_t out_stride, uint8_t* smem) {
  static_assert(kRows == 8 || kRows == 16 || kRows == 32, "the window forms");
  uint16_t* sx = reinterpret_cast<uint16_t*>(smem);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int nsb = k / kBf12Super;
  const size_t row_bytes = static_cast<size_t>(nsb) * kBf12SuperBytes;  // whole super-blocks, no tail (the caller requires k % kBf12Super == 0)
  for (int g0 = n0; g0 < n0 + kBf12RowsUnit; g0 += kBf12RowsGroup) {
    // The next group's packed rows into L2 while this one streams (the last group: the next unit's first).
    if (lane < kBf12RowsGroup / kBf12RowsWarps) {
      const int next = g0 + kBf12RowsGroup + warp * kBf12RowsPerWarp + lane;
      if (next < n) prefetch_l2_bulk(packed + static_cast<size_t>(next) * row_bytes, static_cast<unsigned>(row_bytes));
    }
    int row[kBf12RowsPerWarp];
    bool live[kBf12RowsPerWarp], raw_row[kBf12RowsPerWarp];
    const uint8_t* wr[kBf12RowsPerWarp];
    const uint16_t* raw_w[kBf12RowsPerWarp];
    uint32_t eb[kBf12RowsPerWarp], base23[kBf12RowsPerWarp], ee[kBf12RowsPerWarp];
    float acc[kBf12RowsPerWarp][kRows];
#pragma unroll
    for (int i = 0; i < kBf12RowsPerWarp; ++i) {
      row[i] = g0 + warp * kBf12RowsPerWarp + i;
      live[i] = row[i] < n;
      const int r = live[i] ? row[i] : 0;
      wr[i] = packed + static_cast<size_t>(r) * row_bytes;
      eb[i] = rowtab[2 * r];
      base23[i] = rowtab[2 * r + 1];
      ee[i] = rowtab[2 * r + 2];
      raw_row[i] = (base23[i] & kBf12RawRow) != 0u;
      raw_w[i] = raw + static_cast<size_t>(base23[i] & ~kBf12RawRow) * k;
#pragma unroll
      for (int a = 0; a < kRows; ++a) acc[i][a] = 0.f;
    }
    for (int wi = 0; wi < nsb; ++wi) {
      // The rows' super-blocks first (the DRAM loads in flight across the staging), then the window.
      uint4 pa[kBf12RowsPerWarp], pb[kBf12RowsPerWarp], pe[kBf12RowsPerWarp];
#pragma unroll
      for (int i = 0; i < kBf12RowsPerWarp; ++i) {
        const uint8_t* p = wr[i] + static_cast<size_t>(wi) * kBf12SuperBytes + lane * 16;
        pa[i] = raw_row[i] ? make_uint4(0u, 0u, 0u, 0u) : __ldcs(reinterpret_cast<const uint4*>(p));
        pb[i] = raw_row[i] ? make_uint4(0u, 0u, 0u, 0u) : __ldcs(reinterpret_cast<const uint4*>(p + 512));
        pe[i] = raw_row[i] ? make_uint4(0u, 0u, 0u, 0u) : __ldcs(reinterpret_cast<const uint4*>(p + 1024));
      }
      gemv::stage_activations<kRows>(act + static_cast<size_t>(wi) * kBf12Super, act_stride, kBf12Super, sx);
      __syncthreads();
      const int cw = lane * 8, cb = wi * kBf12Super + cw;
#pragma unroll
      for (int i = 0; i < kBf12RowsPerWarp; ++i) {
        if (!live[i]) continue;
        if (raw_row[i]) {
          bf12_gemv::raw_window<kRows>(raw_w[i], wi * kBf12Super, kBf12Super, sx, kBf12Super, lane, acc[i]);
        } else {
          bf12_gemv::consume_step<kRows>(pa[i].x, pa[i].y, pe[i].x, base23[i], sx, kBf12Super, cw, cb, esc, eb[i], ee[i], acc[i]);
          bf12_gemv::consume_step<kRows>(pa[i].z, pa[i].w, pe[i].y, base23[i], sx, kBf12Super, cw + 256, cb + 256, esc, eb[i], ee[i], acc[i]);
          bf12_gemv::consume_step<kRows>(pb[i].x, pb[i].y, pe[i].z, base23[i], sx, kBf12Super, cw + 512, cb + 512, esc, eb[i], ee[i], acc[i]);
          bf12_gemv::consume_step<kRows>(pb[i].z, pb[i].w, pe[i].w, base23[i], sx, kBf12Super, cw + 768, cb + 768, esc, eb[i], ee[i], acc[i]);
        }
      }
      __syncthreads();  // every warp is done with the window before it is restaged
    }
#pragma unroll
    for (int i = 0; i < kBf12RowsPerWarp; ++i) {
      gemv::warp_reduce<kRows>(acc[i]);  // every lane holds every row's sum
      if (!live[i]) continue;
      // Lane a stores activation row a: static indices only (a lane-indexed read of acc would put
      // the accumulators in local memory, and every FMA of the chain with them).
#pragma unroll
      for (int a = 0; a < kRows; ++a)
        if (a < rows && lane == (a & 31)) out[static_cast<size_t>(a) * out_stride + row[i]] = acc[i][a];
    }
  }
}

}  // namespace dgpp::step
