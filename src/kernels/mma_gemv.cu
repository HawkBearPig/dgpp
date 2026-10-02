// The streaming tensor-core decode GEMM — see mma_gemv.hpp.
#include "kernels/mma_gemv.hpp"

#include <algorithm>
#include <cstdlib>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/fp8_gemv.cuh"

namespace dgpp {
namespace {

constexpr int kWideWarps = 8;          // the wide forms' block: 8 warps x 8 weight rows
constexpr int kRowsPerWarp = 8;        // the mma's n
constexpr int kBatch = 8;              // 16-byte vectors in flight per lane
// A lane's run: kVecs consecutive 16-byte vectors of its weight row, so a
// quad's loads cover 4 x kVecs x 16 B contiguous of a row per group: fp8
// one vector (a quad = 64 B; 64-byte runs, four vectors per lane with the
// lines filled across instructions, halved the stream to 112 GB/s), bf16
// two (a quad = one 128-byte line).
template <bool kFp8> struct Fmt {
  static constexpr int kVecs = kFp8 ? 1 : 2;         // 16-byte vectors per lane per group
  static constexpr int kPerVec = kFp8 ? 16 : 8;      // k per vector
  static constexpr int kLaneK = kVecs * kPerVec;     // k per lane per group: 16
  static constexpr int kQuadSpan = 4 * kLaneK;       // k per quad per group: 64
  static constexpr int kLaneUnits = kLaneK / 8;      // the lane's A run in 16-byte units: 2
};
// A's shared-memory unit (16 B = 8 bf16) swizzle: fp8 lanes read 128-byte
// runs 128 B apart (a 4-way bank conflict per LDS.128 phase); XOR-ing the
// unit's low bits with its run index spreads the four lanes over the
// banks. bf16's 32-byte runs already interleave. Stage and read agree.
template <int kLaneUnits>
__device__ __forceinline__ int swz(int u) { return kLaneUnits == 8 ? (u ^ ((u >> 3) & 3)) : u; }
// The activation window [tiles*16 rows, kWindowK(tiles) k] staged in
// dynamic shared memory per batch of chunks, double-buffered (window i+1
// staged while window i is consumed: one barrier per window, and the
// weight loads of the next batch issued before it): every warp of the
// block reads the same A fragments from shared memory instead of a
// dependent global load per k slice (that stall halved the m = 1 rate).
// The window shrinks with the tile count so two buffers stay inside the
// 48 KB the capture paths allow without an attribute opt-in: 1/2/4/8
// tiles (16/32/64/128 rows) take 512/256/128/64-k windows, 8/4/2/1
// chunks in flight per lane. The wider forms are the prefill's (33..128
// rows per launch, the weights read once per 128 rows instead of once
// per 4-row chunk); a row's chain is the same in every form (the set of
// physical k folded per slice is a function of the 64-k group, whatever
// the window), so the forms are bitwise each other's.
template <int kTiles, bool kFp8> struct Win {
  // The one-tile form takes 256-k windows like the two-tile one: its
  // warp-contiguous weight buffer (WarpLoads) and two A buffers then fit
  // two blocks per SM. Tried and not taken (2026-09-14, the bf16 head at
  // one row, GEMV 1406..1431 us): 512-k bf16 windows at one block per SM
  // (97 KB) 1453; 128-k windows at three blocks per SM 1470 (and the fp8
  // one-row form 133 against its 120). The format parameter stays for the
  // window's byte geometry (WarpLoads).
  static constexpr int kBatchW = kTiles == 1 ? kBatch / 2 : kBatch / kTiles;  // vectors per lane per window
  static constexpr int kK = 64 * kBatchW;                             // k per window: 256 / 256 / 128 / 64
  static constexpr int kStride = kK + 8;                              // staged row stride (rows shift 16 B: no bank conflicts)
  static constexpr int kRows = kTiles * 16;
  static constexpr size_t kBufBytes = static_cast<size_t>(kRows) * kStride * 2;
  static constexpr size_t kSmemBytes = 2 * kBufBytes;                 // double-buffered
};
static_assert(Win<1, true>::kSmemBytes <= 48 * 1024 && Win<1, false>::kSmemBytes <= 48 * 1024 &&
              Win<2, true>::kSmemBytes <= 48 * 1024 && Win<4, true>::kSmemBytes <= 48 * 1024 &&
              Win<8, true>::kSmemBytes <= 48 * 1024, "the A windows fit the default smem");

// The decode forms' weight loads are warp-contiguous and asynchronous
// (2026-09-14): a warp reads one row's whole window slice per instruction
// — 256 or 512 contiguous bytes, the DRAM pattern of the GEMV core —
// straight into a per-warp shared-memory ring of kStages window slices
// (cp.async: no register staging, so the depth costs shared memory, not
// registers), laid out in the quad order the mma wants (odd rows shifted
// half a bank line so the quad reads are conflict-free). kStages - 1
// windows are in flight while one is consumed: the per-window DRAM round
// trip is then hidden inside the warp, not only by other resident blocks.
// The ring's block budget is 24 KB (the stages from it, 2..8: two at eight
// warps, so two blocks share an SM; eight at one warp) — a 48 KB budget
// measured no better at any width and cost the second block (2026-09-14).
// The wide forms keep the direct loads (their cost is the tensor work, not
// the stream).
template <int kTiles, bool kFp8, int kW> struct WarpLoads {
  static constexpr int kBudgetKB = 24;
  static constexpr bool kOn = kTiles <= 2;  // the 1..32-row forms
  static constexpr int kRowBytes = Win<kTiles, kFp8>::kK * (kFp8 ? 1 : 2);  // one row's window slice
  static constexpr int kRowVecs = kRowBytes / 16;                         // 16-byte vectors per row slice
  static constexpr int kLanesPerRow = kRowVecs < 32 ? kRowVecs : 32;      // lanes a row's slice spans
  static constexpr int kVecsPerLane = kRowVecs / kLanesPerRow;            // vectors per lane per row (1 or 2)
  static constexpr int kRowsPerInstr = 32 / kLanesPerRow;                 // rows one warp instruction covers
  static constexpr int kInstrs = kRowsPerWarp / kRowsPerInstr;            // instructions per window per warp
  static constexpr int kSwz = kFp8 ? 4 : 1;                               // odd rows' 16-byte-unit XOR (a 64-byte shift)
  static constexpr size_t kWarpBytes = static_cast<size_t>(kRowsPerWarp) * kRowBytes;  // one stage of one warp
  static constexpr size_t kStageBytes = kWarpBytes * kW;                  // one stage of the block
  static constexpr int kStagesRaw = static_cast<int>((static_cast<size_t>(kBudgetKB) * 1024) / kStageBytes);
  static constexpr int kStages = !kOn ? 1 : kStagesRaw < 2 ? 2 : kStagesRaw > 8 ? 8 : kStagesRaw;
  static constexpr size_t kBlockBytes = kStageBytes * kStages;
  static constexpr size_t kSmemBytes = Win<kTiles, kFp8>::kSmemBytes + (kOn ? kBlockBytes : 0);
  // The geometry's invariants (and every member referenced in every instantiation).
  static_assert(kRowVecs * 16 == kRowBytes && kLanesPerRow * kVecsPerLane == kRowVecs &&
                kRowsPerInstr * kLanesPerRow == 32 && kInstrs * kRowsPerInstr == kRowsPerWarp && kSwz > 0 &&
                kBlockBytes == kStageBytes * kStages && kSmemBytes >= Win<kTiles, kFp8>::kSmemBytes &&
                kStages >= 1 && kSmemBytes <= 99 * 1024,
                "the warp-contiguous load geometry");
};

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src, bool pred) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  const int src_size = pred ? 16 : 0;  // 0: the 16 bytes zero-filled, nothing read
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(gmem_src), "r"(src_size));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int kPending>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(kPending)); }

// Stage rows [0, min(m, tiles*16)) x [base, base + W::kK) of act into sA
// (columns past k read as zero); the whole block participates, the caller
// syncs. Rows past m are zeroed once by zero_rows (they never change).
template <int kTiles, bool kFp8, int kLaneUnits, int kW>
__device__ __forceinline__ void stage_a(const uint16_t* __restrict__ act, size_t act_stride, int m,
                                        int k, int base, uint16_t* __restrict__ sA) {
  using W = Win<kTiles, kFp8>;
  constexpr int kUnits = W::kK / 8;  // 16-byte units per row
  if (base >= k) return;
  const int rows = m < W::kRows ? m : W::kRows;
  for (int i = threadIdx.x; i < rows * kUnits; i += kW * 32) {
    const int row = i / kUnits, u = i - row * kUnits;
    const int col = base + u * 8;
    uint4 x = make_uint4(0u, 0u, 0u, 0u);
    if (col + 8 <= k)
      x = *reinterpret_cast<const uint4*>(act + static_cast<size_t>(row) * act_stride + col);
    *reinterpret_cast<uint4*>(sA + static_cast<size_t>(row) * W::kStride + swz<kLaneUnits>(u) * 8) = x;
  }
}
template <int kTiles, bool kFp8, int kLaneUnits, int kW>
__device__ __forceinline__ void zero_rows(int m, uint16_t* __restrict__ sA) {
  using W = Win<kTiles, kFp8>;
  constexpr int kUnits = W::kK / 8;
  const int rows = m < W::kRows ? m : W::kRows;
  for (int i = rows * kUnits + static_cast<int>(threadIdx.x); i < W::kRows * kUnits; i += kW * 32) {
    const int row = i / kUnits, u = i - row * kUnits;
    *reinterpret_cast<uint4*>(sA + static_cast<size_t>(row) * W::kStride + swz<kLaneUnits>(u) * 8) =
        make_uint4(0u, 0u, 0u, 0u);
  }
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One 16-byte fp8 chunk (16 k of one weight row, scale s) -> 8 packed bf16
// pairs (k 0..15 in order), the dequant bridge's values bf16(w * s) — exact
// for the e8m0 (power-of-two) scales the fp8 matrices carry.
__device__ __forceinline__ void chunk_to_bf16x8(const uint4& c, float s, uint32_t (&w)[8]) {
  const uint32_t words[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const float2 w01 = fp8_gemv::e4m3x2_to_float2(static_cast<uint16_t>(words[q] & 0xFFFFu));
    const float2 w23 = fp8_gemv::e4m3x2_to_float2(static_cast<uint16_t>(words[q] >> 16));
    const uint32_t d0 = float_to_bf16_bits(w01.x * s), d1 = float_to_bf16_bits(w01.y * s);
    const uint32_t d2 = float_to_bf16_bits(w23.x * s), d3 = float_to_bf16_bits(w23.y * s);
    w[2 * q] = d0 | (d1 << 16);
    w[2 * q + 1] = d2 | (d3 << 16);
  }
}

// The k permutation. The mma's B fragment gives lane (r = lane/4, t =
// lane%4) weight row r at the slice's k 2t, 2t+1 (b0) and 2t+8, 2t+9 (b1);
// its A fragment gives the same lane rows r and r+8 at the same k. A dot
// product is invariant to which physical k sits in which slot as long as
// A and B agree, so lane t's own 16-element chunk (physical k c0_t..+15,
// c0_t = window + 16 t) fills its four slots of four slices: slice q takes
// chunk elements 4q..4q+3 as b0 = (4q, 4q+1), b1 = (4q+2, 4q+3), and the
// A words are the same four activation elements — 32 contiguous bytes per
// row per chunk, two 16-byte shared loads. No shuffles: the earlier
// owner-lane form (8 shuffles per slice) was issue-bound at ~4 GB/s per SM.
// The chain of a weight row (the set of k folded per slice, the order of
// slices) is a function of (k, the window layout) only: rows batched and
// rows alone see the same arithmetic.

// kTiles: 1 (m <= 16), 2 (m <= 32), 4 (m <= 64), 8 (m <= 128). kFp8: the
// weight format. kW: warps per block (8 weight rows each) — the decode
// forms narrow to 4 / 2 / 1 on a small n so the grid still fills the SMs
// (a [576 x 6144] site is 9 wide blocks: 46 us against the GEMV's 16 at one
// row, 2026-09-14); the rows' chains do not depend on the width.
template <int kTiles, bool kFp8, int kW, typename OutT>
__global__ __launch_bounds__(kW * 32, 2) void mma_gemv_kernel(const uint16_t* __restrict__ act,
                                                            size_t act_stride,
                                                            const void* __restrict__ wv,
                                                            const float* __restrict__ scales,
                                                            OutT* __restrict__ out, int m, int n,
                                                            int k, size_t out_stride, int rs,
                                                            int cs, float* __restrict__ part) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int r = lane / 4, t = lane % 4;
  const int n0 = (blockIdx.x * kW + warp) * kRowsPerWarp;  // this warp's first weight row
  const int row = n0 + r;                                        // this lane's weight row
  const bool row_live = row < n;
  const int scale_cols = (k + (1 << cs) - 1) >> cs;
  const float* scale_row = scales + (row_live ? static_cast<size_t>(row >> rs) * scale_cols : 0);
  const uint8_t* w8 = static_cast<const uint8_t*>(wv);
  const uint16_t* w16 = static_cast<const uint16_t*>(wv);

  float c[kTiles][4];
#pragma unroll
  for (int tt = 0; tt < kTiles; ++tt)
#pragma unroll
    for (int i = 0; i < 4; ++i) c[tt][i] = 0.f;

  using W = Win<kTiles, kFp8>;
  using F = Fmt<kFp8>;
  using L = WarpLoads<kTiles, kFp8, kW>;
  constexpr int kGroups = W::kK / F::kQuadSpan;  // lane runs per window
  extern __shared__ __align__(16) uint16_t sA_raw[];
  uint16_t* sA[2] = {sA_raw, sA_raw + W::kBufBytes / 2};
  // The warp's weight ring (past the A windows): stage s at s * kWarpBytes, row b's slice at b * kRowBytes.
  uint8_t* wring = reinterpret_cast<uint8_t*>(sA_raw) + W::kSmemBytes +
                   static_cast<size_t>(warp) * L::kStages * L::kWarpBytes;
  const int nwin = (k + W::kK - 1) / W::kK;
  // Split-K (the decode forms at a small n, grid.y > 1): this block folds
  // windows [win0, win1) of the k range and stores fp32 partials to
  // part[blockIdx.y][m][n]; mma_gemv_split_reduce_kernel sums the splits in
  // split order. grid.y == 1: the whole range, the output stored directly.
  const int wper = (nwin + static_cast<int>(gridDim.y) - 1) / static_cast<int>(gridDim.y);
  const int win0 = static_cast<int>(blockIdx.y) * wper;
  const int win1 = nwin < win0 + wper ? nwin : win0 + wper;
  // Window `win` of the warp's eight rows into its ring stage, as one
  // asynchronous group (an empty group past the last window keeps the
  // group accounting uniform). Instruction i covers rows i*kRowsPerInstr..
  // of the eight, lane l the bytes [l' * 16 * kVecsPerLane, ...) of its
  // row's slice (l' = lane within the row's lanes); k % 64 == 0 keeps a
  // row's slice whole or absent past k.
  auto issue = [&](int win) {
    if constexpr (L::kOn) {
      if (win < win1) {
        uint8_t* dst = wring + static_cast<size_t>(win % L::kStages) * L::kWarpBytes;
        const size_t kb = static_cast<size_t>(win) * W::kK * (kFp8 ? 1 : 2);  // the window's byte offset in a row
#pragma unroll
        for (int i = 0; i < L::kInstrs; ++i) {
          const int rb = i * L::kRowsPerInstr + lane / L::kLanesPerRow;  // the row within the warp's eight
          const int u = (lane % L::kLanesPerRow) * L::kVecsPerLane;      // the lane's first 16-byte unit in the slice
          const int wrow = n0 + rb;
          const bool in = wrow < n && (kb + static_cast<size_t>(u) * 16) < static_cast<size_t>(k) * (kFp8 ? 1 : 2);
          const uint8_t* src = static_cast<const uint8_t*>(wv) +
                               (in ? static_cast<size_t>(wrow) * k * (kFp8 ? 1 : 2) + kb + static_cast<size_t>(u) * 16 : 0);
#pragma unroll
          for (int v = 0; v < L::kVecsPerLane; ++v)
            cp_async_16(dst + static_cast<size_t>(rb) * L::kRowBytes +
                            static_cast<size_t>((u + v) ^ ((rb & 1) * L::kSwz)) * 16,
                        src + v * 16, in);
        }
      }
      cp_async_commit();
    }
  };
  if constexpr (L::kOn) {
#pragma unroll
    for (int s0 = 0; s0 < L::kStages - 1; ++s0) issue(win0 + s0);
  }
  // The first window's activations, then per window: issue the weight
  // loads, stage the NEXT window's activations, consume, one barrier.
  // Rows past m never change: zeroed once in both buffers, so a window's
  // staging touches only the live rows (a sixteenth of the stores at one row).
  zero_rows<kTiles, kFp8, F::kLaneUnits, kW>(m, sA[0]);
  zero_rows<kTiles, kFp8, F::kLaneUnits, kW>(m, sA[1]);
  stage_a<kTiles, kFp8, F::kLaneUnits, kW>(act, act_stride, m, k, win0 * W::kK, sA[0]);
  __syncthreads();
  int buf = 0;
  for (int base = win0 * W::kK, base_end = win1 * W::kK; base < base_end; base += W::kK, buf ^= 1) {
    uint4 wv4[kGroups][F::kVecs];
    if constexpr (L::kOn) {
      // Keep kStages - 1 windows in flight: window base + (kStages - 1)
      // into the stage consumed last window (its reads finished before the
      // previous window's barrier), then wait for this window's group.
      const int win = base / W::kK;
      issue(win + L::kStages - 1);
      cp_async_wait<L::kStages - 1>();
      __syncwarp();  // every lane's copies of this window visible to the warp
      stage_a<kTiles, kFp8, F::kLaneUnits, kW>(act, act_stride, m, k, base + W::kK, sA[buf ^ 1]);
      // The quad layout: lane (r, t) takes its run of group g (16 fp8 or 8 bf16
      // k per vector) out of row r's slice.
      const uint8_t* wbuf = wring + static_cast<size_t>(win % L::kStages) * L::kWarpBytes;
#pragma unroll
      for (int g = 0; g < kGroups; ++g) {
        const int u0 = (g * F::kQuadSpan + t * F::kLaneK) * (kFp8 ? 1 : 2) / 16;
#pragma unroll
        for (int v = 0; v < F::kVecs; ++v)
          wv4[g][v] = *reinterpret_cast<const uint4*>(wbuf + static_cast<size_t>(r) * L::kRowBytes +
                                                      static_cast<size_t>((u0 + v) ^ ((r & 1) * L::kSwz)) * 16);
      }
    } else {
#pragma unroll
    for (int g = 0; g < kGroups; ++g) {
      const int c0 = base + g * F::kQuadSpan + t * F::kLaneK;  // this lane's run (k % 64 == 0: whole or none)
      const bool in = row_live && c0 < k;
      const uint4* p = kFp8 ? reinterpret_cast<const uint4*>(w8 + static_cast<size_t>(row) * k + c0)
                            : reinterpret_cast<const uint4*>(w16 + static_cast<size_t>(row) * k + c0);
#pragma unroll
      for (int v = 0; v < F::kVecs; ++v) wv4[g][v] = in ? p[v] : make_uint4(0u, 0u, 0u, 0u);
    }
    stage_a<kTiles, kFp8, F::kLaneUnits, kW>(act, act_stride, m, k, base + W::kK, sA[buf ^ 1]);
    }
    const uint16_t* sAc = sA[buf];
#pragma unroll
    for (int g = 0; g < kGroups; ++g) {
      if (base + g * F::kQuadSpan >= k) break;
      const int c0 = base + g * F::kQuadSpan + t * F::kLaneK;
      const int u0 = (c0 - base) / 8;  // the run's first A unit (window-local)
#pragma unroll
      for (int v = 0; v < F::kVecs; ++v) {
        // This vector's k (16 fp8 or 8 bf16) as bf16 words, the B words of
        // its slices (4 k per slice); the A words are the same k of rows r
        // and r+8: one 16-byte unit per 8 k.
        uint32_t w[8];
        if (kFp8) {
          const float s = (row_live && c0 < k) ? scale_row[(c0 + v * F::kPerVec) >> cs] : 0.f;
          chunk_to_bf16x8(wv4[g][v], s, w);
        } else {
          w[0] = wv4[g][v].x; w[1] = wv4[g][v].y; w[2] = wv4[g][v].z; w[3] = wv4[g][v].w;
        }
        constexpr int kUnitsPerVec = F::kPerVec / 8;  // 2 or 1
#pragma unroll
        for (int tt = 0; tt < kTiles; ++tt) {
          const uint16_t* row0 = sAc + static_cast<size_t>(tt * 16 + r) * W::kStride;
          const uint16_t* row8 = row0 + static_cast<size_t>(8) * W::kStride;
#pragma unroll
          for (int j = 0; j < kUnitsPerVec; ++j) {
            const int u = swz<F::kLaneUnits>(u0 + v * kUnitsPerVec + j) * 8;
            const uint4 a0 = *reinterpret_cast<const uint4*>(row0 + u);
            const uint4 a8 = *reinterpret_cast<const uint4*>(row8 + u);
            mma_bf16(c[tt], a0.x, a8.x, a0.y, a8.y, w[4 * j], w[4 * j + 1]);
            mma_bf16(c[tt], a0.z, a8.z, a0.w, a8.w, w[4 * j + 2], w[4 * j + 3]);
          }
        }
      }
    }
    __syncthreads();  // this window consumed, the next one staged
  }
  // Epilogue: C fragment (m16n8): (r, 2t), (r, 2t+1), (r+8, 2t), (r+8, 2t+1)
  // of the warp's [16 x 8] slice; the slice's n columns are the warp's rows.
  const int col0 = n0 + 2 * t;
  if (part) {
    float* prow = part + static_cast<size_t>(blockIdx.y) * m * n;
#pragma unroll
    for (int tt = 0; tt < kTiles; ++tt) {
      const int mrow = tt * 16 + r;
      if (mrow < m) {
        if (col0 < n) prow[static_cast<size_t>(mrow) * n + col0] = c[tt][0];
        if (col0 + 1 < n) prow[static_cast<size_t>(mrow) * n + col0 + 1] = c[tt][1];
      }
      if (mrow + 8 < m) {
        if (col0 < n) prow[static_cast<size_t>(mrow + 8) * n + col0] = c[tt][2];
        if (col0 + 1 < n) prow[static_cast<size_t>(mrow + 8) * n + col0 + 1] = c[tt][3];
      }
    }
    return;
  }
#pragma unroll
  for (int tt = 0; tt < kTiles; ++tt) {
    const int mrow = tt * 16 + r;
    if (mrow < m) {
      if (col0 < n) fp8_gemv::store_dot(out + static_cast<size_t>(mrow) * out_stride + col0, c[tt][0]);
      if (col0 + 1 < n) fp8_gemv::store_dot(out + static_cast<size_t>(mrow) * out_stride + col0 + 1, c[tt][1]);
    }
    if (mrow + 8 < m) {
      if (col0 < n) fp8_gemv::store_dot(out + static_cast<size_t>(mrow + 8) * out_stride + col0, c[tt][2]);
      if (col0 + 1 < n) fp8_gemv::store_dot(out + static_cast<size_t>(mrow + 8) * out_stride + col0 + 1, c[tt][3]);
    }
  }
}

// The split-K reduce: out[m, n] = sum over the splits, in split order, of
// the fp32 partials (deterministic; the split count is a function of the
// shape, never of m).
template <typename OutT>
__global__ void mma_gemv_split_reduce_kernel(const float* __restrict__ part, int splits, int m, int n,
                                             OutT* __restrict__ out, size_t out_stride) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m * n) return;
  const int mrow = i / n, col = i - mrow * n;
  float acc = 0.f;
  for (int s = 0; s < splits; ++s) acc += part[(static_cast<size_t>(s) * m + mrow) * n + col];
  fp8_gemv::store_dot(out + static_cast<size_t>(mrow) * out_stride + col, acc);
}

// Split-K for the decode forms (2026-09-21): a small-n site's grid is n / 16
// blocks at two warps (the [320 x 10240] hyperconnection down projection: 20
// blocks on a 48-SM part), and each warp then walks its k windows in
// sequence with nothing else resident — the c=4 profile has that site at
// 68 us for 3.3 MB (48 GB/s) and the shared expert's [320 x 2560] gate and
// up at 19-24 us (under 45 GB/s). Splitting the k range across grid.y
// blocks (fp32 partials in the caller's GEMM workspace, one reduce launch)
// fills the part. The split count is a function of (n, k, width) only, so a
// row's chain is the same whatever m rides in the launch (the decode
// forms' invariant); it is not the unsplit chain (tolerance-equal, the
// contract above the GEMV chunk bound). On by default -- the same trade
// cuBLAS makes by heuristic for small-n shapes; DGPP_MMA_SPLITK=0 keeps the
// unsplit chain. Measured on two GB10s (Qwen3.8-Flash-Next, TP=2, MTP depth
// 3): the decode pass shortens 119.0 -> 113.5 ms (-4.6%); tokens per pass
// also move a little, because the summation order (and so the text) does.
// DGPP_MMA_SPLITK_FILL overrides the block target (default two resident
// blocks per SM).
constexpr int kMaxSplit = 16;
int split_k_target_blocks() {
  static const int target = [] {
    if (const char* v = std::getenv("DGPP_MMA_SPLITK_FILL")) {
      const int t = std::atoi(v);
      if (t >= 1) return t;
    }
    int dev = 0, sms = 48;
    if (cudaGetDevice(&dev) == cudaSuccess) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    return 2 * sms;
  }();
  return target;
}
bool split_k_enabled() {
  static const bool on = [] {
    const char* v = std::getenv("DGPP_MMA_SPLITK");
    return !(v && v[0] == '0' && v[1] == '\0');
  }();
  return on;
}
int split_k_for(int blocks, int n, int k, int win_k, const void* ws, size_t ws_bytes) {
  if (!ws || !split_k_enabled()) return 1;
  const int fill = split_k_target_blocks();
  if (blocks >= fill) return 1;
  const int nwin = (k + win_k - 1) / win_k;
  const int s = std::min({(fill + blocks - 1) / blocks, nwin, kMaxSplit});
  // The workspace check at the decode forms' widest m, so the split (and the
  // chain) never depends on the rows in the launch.
  if (s < 2 || static_cast<size_t>(s) * kMmaGemvMaxRows * static_cast<size_t>(n) * sizeof(float) > ws_bytes)
    return 1;
  return s;
}

// One decode form at one width: the shared-memory opt-in once per
// instantiation (the A windows + the warp weight buffers exceed the default
// budget at the wide widths).
template <int kTiles, bool kFp8, int kW, typename OutT>
void launch_decode_form(const uint16_t* a, size_t act_stride, const void* w, const float* scales, OutT* o,
                        int rows, int n, int k, size_t out_stride, int rs, int cs, cudaStream_t stream,
                        void* ws, size_t ws_bytes) {
  using L = WarpLoads<kTiles, kFp8, kW>;
  static bool opted_in = false;
  if (!opted_in) {
    DGPP_CUDA_OK(cudaFuncSetAttribute(mma_gemv_kernel<kTiles, kFp8, kW, OutT>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(L::kSmemBytes)));
    opted_in = true;
  }
  const int blocks = (n + kW * kRowsPerWarp - 1) / (kW * kRowsPerWarp);
  const int splits = split_k_for(blocks, n, k, Win<kTiles, kFp8>::kK, ws, ws_bytes);
  if (splits > 1) {
    float* part = static_cast<float*>(ws);
    mma_gemv_kernel<kTiles, kFp8, kW, OutT><<<dim3(blocks, splits), kW * 32, L::kSmemBytes, stream>>>(
        a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs, part);
    const int total = rows * n;
    mma_gemv_split_reduce_kernel<OutT><<<(total + 255) / 256, 256, 0, stream>>>(part, splits, rows, n, o,
                                                                                 out_stride);
    return;
  }
  mma_gemv_kernel<kTiles, kFp8, kW, OutT><<<dim3(blocks), kW * 32, L::kSmemBytes, stream>>>(
      a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs, nullptr);
}
// The decode forms' width from n: the widest whose grid fills the resident
// blocks (n >= 6144 rows: eight warps; [2048, 6144): four; [1024, 2048):
// two; under 1024: one).
// The decode forms' width from n (mma_gemv_test's sweep over the DSA and
// attention shapes, 2026-09-14): eight warps from 4096 rows, four from
// 1024, two under (never one: a lone warp per block is issue-bound — the
// [576 x 6144] site at 16 rows 53 us at one warp, 37 at two, 44 at four).
int g_mma_decode_width = 0;  // mma_gemv_set_decode_width: 0 the rule, else a fixed width (the sweep)
template <int kTiles, bool kFp8, typename OutT>
void launch_decode(const uint16_t* a, size_t act_stride, const void* w, const float* scales, OutT* o, int rows,
                   int n, int k, size_t out_stride, int rs, int cs, cudaStream_t stream, void* ws,
                   size_t ws_bytes) {
  const int warps = (n + kRowsPerWarp - 1) / kRowsPerWarp;
  const int width = g_mma_decode_width > 0 ? g_mma_decode_width : warps >= 512 ? 8 : warps >= 128 ? 4 : 2;
  if (width == 8)
    launch_decode_form<kTiles, kFp8, 8, OutT>(a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs, stream, ws, ws_bytes);
  else if (width == 4)
    launch_decode_form<kTiles, kFp8, 4, OutT>(a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs, stream, ws, ws_bytes);
  else if (width == 2)
    launch_decode_form<kTiles, kFp8, 2, OutT>(a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs, stream, ws, ws_bytes);
  else
    launch_decode_form<kTiles, kFp8, 1, OutT>(a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs, stream, ws, ws_bytes);
}

// DGPP_MMA_TRACE=n: print the first n shape decisions (the dispatch
// bisection log: which (m, n, k) lands on which tiles form).
static int g_mma_trace_budget() {
  static const int v = [] {
    const char* e = std::getenv("DGPP_MMA_TRACE");
    if (!e || !*e || *e == '0') return 0;
    return std::atoi(e) > 0 ? std::atoi(e) : 64;
  }();
  return v;
}
static int& g_mma_trace_left() {
  static int v = -1;  // -1: not yet read from the budget
  if (v < 0) v = g_mma_trace_budget();
  return v;
}

template <bool kFp8, typename OutT>
void launch(const uint16_t* act, size_t act_stride, const void* w, const float* scales, OutT* out,
            int m, int n, int k, size_t out_stride, int rs, int cs, cudaStream_t stream, void* ws = nullptr,
            size_t ws_bytes = 0) {
  if (m < 1) throw std::invalid_argument("mma_gemv: m outside [1, ...)");
  if (n <= 0 || k <= 0 || (k % 64) != 0) throw std::invalid_argument("mma_gemv: k a multiple of 64");
  if (!mma_gemv_shape_ok(w, act, act_stride, m, k)) throw std::invalid_argument("mma_gemv: alignment");
  if (out_stride == 0) out_stride = static_cast<size_t>(n);
  if (out_stride < static_cast<size_t>(n)) throw std::invalid_argument("mma_gemv: out stride");
  if (kFp8 && (scales == nullptr || cs < 4)) throw std::invalid_argument("mma_gemv: fp8 scales");
  constexpr int kWideThreads = kWideWarps * 32;
  const dim3 grid((n + kWideWarps * kRowsPerWarp - 1) / (kWideWarps * kRowsPerWarp));
  // Groups of at most kMmaGemvMaxRowsPerLaunch rows; the form by the group's rows.
  for (int row0 = 0; row0 < m; row0 += kMmaGemvMaxRowsPerLaunch) {
    const int rows = std::min(kMmaGemvMaxRowsPerLaunch, m - row0);
    const uint16_t* a = act + static_cast<size_t>(row0) * act_stride;
    OutT* o = out + static_cast<size_t>(row0) * out_stride;
    if (g_mma_trace_left() > 0) {
      --g_mma_trace_left();
      std::fprintf(stderr, "mma_trace fp8=%d out=%s m=%d n=%d k=%d tiles=%d\n",
                   static_cast<int>(kFp8), std::is_same<OutT, float>::value ? "f32" : "bf16", m, n, k,
                   rows <= 16 ? 1 : rows <= 32 ? 2 : rows <= 64 ? 4 : 8);
      std::fflush(stderr);
    }
    if (rows <= 16)
      launch_decode<1, kFp8, OutT>(a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs, stream, ws, ws_bytes);
    else if (rows <= 32)
      launch_decode<2, kFp8, OutT>(a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs, stream, ws, ws_bytes);
    else if (rows <= 64)
      mma_gemv_kernel<4, kFp8, kWideWarps, OutT><<<grid, kWideThreads, Win<4, kFp8>::kSmemBytes, stream>>>(a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs, nullptr);
    else
      mma_gemv_kernel<8, kFp8, kWideWarps, OutT><<<grid, kWideThreads, Win<8, kFp8>::kSmemBytes, stream>>>(a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs, nullptr);
    DGPP_CUDA_OK(cudaGetLastError());
  }
}
}  // namespace

void mma_gemv_set_decode_width(int warps) {
  if (warps != 0 && warps != 1 && warps != 2 && warps != 4 && warps != 8)
    throw std::invalid_argument("mma_gemv_set_decode_width: 0, 1, 2, 4 or 8");
  g_mma_decode_width = warps;
}

bool mma_gemv_shape_ok(const void* w, const void* act, size_t act_stride, int m, int k) {
  return m >= 1 && k > 0 && (k % 64) == 0 &&
         (reinterpret_cast<uintptr_t>(w) & 15u) == 0 && (reinterpret_cast<uintptr_t>(act) & 15u) == 0 &&
         ((act_stride * 2) % 16) == 0;
}

void launch_mma_gemv_fp8_bf16(const uint16_t* act, size_t act_stride, const uint8_t* w,
                              const float* scales, uint16_t* out, int m, int n, int k,
                              size_t out_stride, int rs, int cs, cudaStream_t stream, void* ws,
                              size_t ws_bytes) {
  launch<true, uint16_t>(act, act_stride, w, scales, out, m, n, k, out_stride, rs, cs, stream, ws, ws_bytes);
}
void launch_mma_gemv_fp8_f32(const uint16_t* act, size_t act_stride, const uint8_t* w,
                             const float* scales, float* out, int m, int n, int k,
                             size_t out_stride, int rs, int cs, cudaStream_t stream, void* ws,
                             size_t ws_bytes) {
  launch<true, float>(act, act_stride, w, scales, out, m, n, k, out_stride, rs, cs, stream, ws, ws_bytes);
}
void launch_mma_gemv_bf16_bf16(const uint16_t* act, size_t act_stride, const uint16_t* w,
                               uint16_t* out, int m, int n, int k, size_t out_stride,
                               cudaStream_t stream) {
  launch<false, uint16_t>(act, act_stride, w, nullptr, out, m, n, k, out_stride, 7, 7, stream);
}
void launch_mma_gemv_bf16_f32(const uint16_t* act, size_t act_stride, const uint16_t* w,
                              float* out, int m, int n, int k, size_t out_stride,
                              cudaStream_t stream) {
  launch<false, float>(act, act_stride, w, nullptr, out, m, n, k, out_stride, 7, 7, stream);
}

}  // namespace dgpp
