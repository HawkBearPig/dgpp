// The EXL3 streaming decode GEMM and the MoE kernels on its core — see
// exl3_gemv.hpp.
#include "kernels/exl3_gemv.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/exl3_codec.cuh"
#include "kernels/pdl.cuh"

namespace dgpp {
namespace {

constexpr int kWarps = 8;                               // a block: eight 16-wide n tiles = one 128-output block
constexpr int kThreads = kWarps * 32;
constexpr int kWindowK = kExl3Block;                    // 128 k per window: one rotation block
constexpr int kTilesPerWindow = kWindowK / kExl3Tile;   // 8
constexpr int kAStride = kWindowK + 8;                  // staged fp16 row: 272 B (16-byte aligned, ldmatrix conflict-free)
constexpr int kAChunksPerRow = kWindowK * 2 / 16;       // 16-byte cp.async chunks per staged row: 16
constexpr int kMaxSplit = 16;
constexpr size_t kSmemBudget = 99 * 1024;               // the opt-in ceiling the other decode forms use

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src, bool pred) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  const int src_size = pred ? 16 : 0;  // 0: the 16 bytes zero-filled, nothing read
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(gmem_src), "r"(src_size));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int kPending>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(kPending));
}

__device__ __forceinline__ void store_out(uint16_t* p, float v) { *p = float_to_bf16_bits(v); }
__device__ __forceinline__ void store_out(float* p, float v) { *p = v; }
__device__ __forceinline__ float sigmoid_acc(float x) { return 1.0f / (1.0f + expf(-x)); }

// The block core: one 128-output block of one EXL3 matrix against up to
// kRows rotated activation rows (xh, fp16, stride xh_stride), over the
// windows [win0, win1) of the k range. A stage holds one window: the A
// rows (fp16, kRows x kAStride; rows past `rows` zero-filled) then the
// eight warps' B tiles; kStages - 1 windows are in flight while one is
// consumed, both operands by cp.async (no compute on the staging path).
// The accumulators come back as mma C fragments; the epilogues below turn
// them into pre-rotation partials or rotated, scaled outputs.
template <int kBits, int kMT, int kStageCap = 8>
struct Core {
  static constexpr int kRows = kMT * 16;
  static constexpr int kTileBytes = 32 * kBits;
  static constexpr int kWindowBytes = kTilesPerWindow * kTileBytes;  // one warp's B per window
  static constexpr int kBChunks = kWindowBytes / 16;                 // per warp per window
  static constexpr size_t kABytes = static_cast<size_t>(kRows) * kAStride * 2;
  static constexpr size_t kStageBytes = kABytes + static_cast<size_t>(kWarps) * kWindowBytes;
  static constexpr int kStagesRaw = static_cast<int>(kSmemBudget / kStageBytes);
  // As many stages as the budget holds, up to kStageCap: the slot forms
  // take eight (one-row forms: 3-bit 8, 4-bit 8, 6-bit 6 — at one decode
  // row the slot kernels fill 35 of 48 SMs and ran at 72 % of the DRAM rate
  // with three windows in flight per block, 2026-10-23 trace, 95 % with
  // the deep ring), the dense GEMV four (its split grids fill the SMs and
  // its launches ran 16 % longer under the deep ring's smem). Bitwise: the
  // chain is the same at every depth.
  static constexpr int kStages = kStagesRaw > kStageCap ? kStageCap : kStagesRaw;
  static constexpr size_t kSmemBytes = static_cast<size_t>(kStages) * kStageBytes;
  // The epilogues' C tile [kRows][128] fp32 reuses the stages.
  static constexpr size_t kCBytes = static_cast<size_t>(kRows) * kExl3Block * 4;
  static_assert(kStages >= 2, "two stages at least");
  static_assert(kCBytes <= kSmemBytes, "the C tile fits");
  static_assert(kWindowBytes % 16 == 0 && kABytes % 16 == 0, "16-byte stages");

  // tiles_block: the block's first n tile's stream (n-tile-major); its
  // eight warps' streams follow at n_tile_bytes each. The A rows are the
  // rotated fp16 rows xh (a rotate kernel's: rotating them in the stage
  // was measured 2026-10-23 and cost 30-60 % of every kernel's time — the
  // rotating warp also computes, so its rotation sits on the block's
  // critical path at every window; the plan's §7 record).
  __device__ static void run(uint8_t* smem, const uint16_t* __restrict__ xh, size_t xh_stride, int rows,
                             const uint8_t* __restrict__ tiles_block, size_t n_tile_bytes, int win0, int win1,
                             float (&c)[kMT][2][4]) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const uint8_t* wtiles = tiles_block + static_cast<size_t>(warp) * n_tile_bytes;
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) c[mt][j][i] = 0.f;
    auto stage_a = [&](int s) { return reinterpret_cast<uint16_t*>(smem + static_cast<size_t>(s) * kStageBytes); };
    auto stage_b = [&](int s, int w) {
      return smem + static_cast<size_t>(s) * kStageBytes + kABytes + static_cast<size_t>(w) * kWindowBytes;
    };
    auto issue = [&](int win) {
      if (win < win1) {
        const int s = win % kStages;
        const int base = win * kWindowK;
        uint16_t* sa = stage_a(s);
        for (int i = static_cast<int>(threadIdx.x); i < kRows * kAChunksPerRow; i += kThreads) {
          const int r = i / kAChunksPerRow, ch = i - r * kAChunksPerRow;
          const bool live = r < rows;
          const uint16_t* src = live ? xh + static_cast<size_t>(r) * xh_stride + base + ch * 8 : xh;
          cp_async_16(sa + static_cast<size_t>(r) * kAStride + ch * 8, src, live);
        }
        uint8_t* sb = stage_b(s, warp);
        const uint8_t* src = wtiles + static_cast<size_t>(win) * kWindowBytes;
        for (int ch = lane; ch < kBChunks; ch += 32) cp_async_16(sb + ch * 16, src + ch * 16, true);
      }
      cp_async_commit();
    };
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) issue(win0 + s);
    for (int win = win0; win < win1; ++win) {
      issue(win + kStages - 1);
      cp_async_wait<kStages - 1>();
      __syncthreads();  // this window's A (every thread's chunks) and B visible
      const int s = win % kStages;
      const uint16_t* sa = stage_a(s);
      const uint8_t* wb = stage_b(s, warp);
#pragma unroll
      for (int kt = 0; kt < kTilesPerWindow; ++kt) {
        uint32_t b[4];
        exl3::lane_fragments<kBits>(reinterpret_cast<const uint32_t*>(wb + kt * kTileBytes), lane, b);
#pragma unroll
        for (int mt = 0; mt < kMT; ++mt) {
          uint32_t a[4];
          exl3::ldmatrix_a(a, sa + static_cast<size_t>(mt * 16) * kAStride + kt * kExl3Tile, kAStride * 2, lane);
          exl3::mma_f16(c[mt][0], a[0], a[1], a[2], a[3], b[0], b[1]);
          exl3::mma_f16(c[mt][1], a[0], a[1], a[2], a[3], b[2], b[3]);
        }
      }
      __syncthreads();  // the stage is free for the window issued next iteration
    }
    // (win0 == win1: nothing in flight; the empty groups cost nothing.)
  }

  // The fragments as the block's [kRows][128] pre-rotation tile in shared
  // memory (the stages, free after run). Caller syncs after.
  __device__ static void fragments_to_smem(float* sC, const float (&c)[kMT][2][4]) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int g = lane / 4, t = lane % 4;
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int col = warp * kExl3Tile + j * 8 + 2 * t;
        const int r0 = mt * 16 + g, r1 = r0 + 8;
        sC[r0 * kExl3Block + col] = c[mt][j][0];
        sC[r0 * kExl3Block + col + 1] = c[mt][j][1];
        sC[r1 * kExl3Block + col] = c[mt][j][2];
        sC[r1 * kExl3Block + col + 1] = c[mt][j][3];
      }
  }
  // The rotated, scaled output rows [0, rows) of the smem tile: one warp
  // per row, four outputs per lane at out + r * out_stride + 4 lane.
  template <typename OutT>
  __device__ static void rotate_store(const float* sC, int rows, const uint16_t* __restrict__ svh_block,
                                      OutT* __restrict__ out, size_t out_stride) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int r = warp; r < rows && r < kRows; r += kWarps) {
      const float4 v = *reinterpret_cast<const float4*>(sC + r * kExl3Block + 4 * lane);
      float v0 = v.x, v1 = v.y, v2 = v.z, v3 = v.w;
      exl3::rotate_out_warp(v0, v1, v2, v3, svh_block, lane);
      OutT* o = out + static_cast<size_t>(r) * out_stride + 4 * lane;
      store_out(o, v0);
      store_out(o + 1, v1);
      store_out(o + 2, v2);
      store_out(o + 3, v3);
    }
  }
};

// ---- the dense form -------------------------------------------------------

// The A operand: xh[m, k] fp16 = rotate_in(act rows) — one warp per (row,
// 128-k block); the GEMV stages it by cp.async. act rows 8-byte aligned.
__global__ void exl3_rotate_rows_kernel(const uint16_t* __restrict__ act, size_t act_stride,
                                        const uint16_t* __restrict__ suh, uint16_t* __restrict__ xh, int m, int k) {
  dgpp_pdl_wait();
  const int lane = threadIdx.x % 32;
  const int gw = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) / 32) + static_cast<int>(threadIdx.x) / 32;
  const int blocks = k / kWindowK;
  if (gw >= m * blocks) return;
  const int row = gw / blocks, base = (gw - row * blocks) * kWindowK;
  exl3::rotate_in_warp(act + static_cast<size_t>(row) * act_stride + base, suh + base,
                       xh + static_cast<size_t>(row) * k + base, lane);
}

// kBits: the trellis width; kMT: 16-row A tiles (1, 2, 4, 8 -> m <= 16, 32, 64, 128).
// grid.x = n / 128 output blocks, grid.y = the splits. xh: the rotated
// fp16 rows (stride k). part: null for the unsplit form (the rotation in
// the epilogue), else the
// pre-rotation fp32 partials [grid.y][m][n]; the LAST split block of an
// output block to arrive (counters[blockIdx.x], zero between launches:
// each split block counts itself in, the last one — the count grid.y - 1
// — resets it) sums the splits in split order, rotates, scales and
// stores — the reduce launch folded into the GEMV (2026-10-23: the
// three-kernel chain cost 1.3 ms of launches a decode pass). The
// workspace owner zeroes the counters once at allocation (the hpp).
template <int kBits, int kMT, typename OutT>
__global__ __launch_bounds__(kThreads) void exl3_gemv_kernel(const uint16_t* __restrict__ xh,
                                                             const uint8_t* __restrict__ tiles,
                                                             const uint16_t* __restrict__ svh, OutT* __restrict__ out,
                                                             size_t out_stride, int m, int n, int k,
                                                             float* __restrict__ part,
                                                             unsigned* __restrict__ counters) {
  dgpp_pdl_wait();
  using C = Core<kBits, kMT, 4>;
  extern __shared__ __align__(16) uint8_t smem[];
  const int n0 = static_cast<int>(blockIdx.x) * kExl3Block;
  const size_t n_tile_bytes = static_cast<size_t>(k / kExl3Tile) * C::kTileBytes;
  const int nwin = k / kWindowK;
  const int wper = (nwin + static_cast<int>(gridDim.y) - 1) / static_cast<int>(gridDim.y);
  const int win0 = static_cast<int>(blockIdx.y) * wper;
  const int win1 = nwin < win0 + wper ? nwin : win0 + wper;
  float c[kMT][2][4];
  C::run(smem, xh, static_cast<size_t>(k), m, tiles + static_cast<size_t>(n0 / kExl3Tile) * n_tile_bytes,
         n_tile_bytes, win0, win1, c);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  if (part != nullptr) {
    {
      const int g = lane / 4, t = lane % 4;
      float* prow = part + static_cast<size_t>(blockIdx.y) * m * n;
#pragma unroll
      for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int col = n0 + warp * kExl3Tile + j * 8 + 2 * t;
          const int r0 = mt * 16 + g, r1 = r0 + 8;
          if (r0 < m) {
            prow[static_cast<size_t>(r0) * n + col] = c[mt][j][0];
            prow[static_cast<size_t>(r0) * n + col + 1] = c[mt][j][1];
          }
          if (r1 < m) {
            prow[static_cast<size_t>(r1) * n + col] = c[mt][j][2];
            prow[static_cast<size_t>(r1) * n + col + 1] = c[mt][j][3];
          }
        }
    }
    // This block's partial is visible device-wide before its count lands.
    // The count's broadcast word lives in the stages (free after run; a
    // static __shared__ would push the 99 KB forms past the opt-in limit).
    unsigned* s_prev = reinterpret_cast<unsigned*>(smem);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) *s_prev = atomicAdd(counters + blockIdx.x, 1u);
    __syncthreads();
    if (*s_prev != gridDim.y - 1) return;
    if (threadIdx.x == 0) counters[blockIdx.x] = 0u;  // every split block has counted: zero for the next launch
    __threadfence();
    // The last split block: the splits' partials (L2, not this SM's L1)
    // summed in split order from zero — the reduce the separate launch did,
    // the same sums in the same order.
    const int splits = static_cast<int>(gridDim.y);
    for (int r = warp; r < m; r += kWarps) {
      // Every split's partial in flight at once (a loop bounded at run
      // time serialized the L2 round trips: 2.5 us a launch), then the
      // sums in split order from zero — the reduce kernel's arithmetic.
      float4 p[kMaxSplit];
#pragma unroll
      for (int s = 0; s < kMaxSplit; ++s)
        if (s < splits)
          p[s] = __ldcg(reinterpret_cast<const float4*>(part + (static_cast<size_t>(s) * m + r) * n + n0 + 4 * lane));
      float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;
#pragma unroll
      for (int s = 0; s < kMaxSplit; ++s) {
        if (s < splits) {
          v0 += p[s].x;
          v1 += p[s].y;
          v2 += p[s].z;
          v3 += p[s].w;
        }
      }
      exl3::rotate_out_warp(v0, v1, v2, v3, svh + n0, lane);
      OutT* o = out + static_cast<size_t>(r) * out_stride + n0 + 4 * lane;
      store_out(o, v0);
      store_out(o + 1, v1);
      store_out(o + 2, v2);
      store_out(o + 3, v3);
    }
    return;
  }
  float* sC = reinterpret_cast<float*>(smem);
  C::fragments_to_smem(sC, c);
  __syncthreads();
  C::rotate_store(sC, m, svh + n0, out + n0, out_stride);
}

int split_target_blocks() {
  static const int target = [] {
    int dev = 0, sms = 48;
    if (cudaGetDevice(&dev) == cudaSuccess) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    return 2 * sms;
  }();
  return target;
}

// The workspace layout: the arrival counters first — a FIXED kCountersBytes
// region, one counter per 128-output block of the launch, whatever the
// site (sites of different widths share a workspace, so the counters may
// not move with the partials' size: a site's partials would land on
// another's counters) — then the rotated rows xh [kExl3GemvSplitRows, k]
// fp16, then the split partials
// [splits][kExl3GemvSplitRows][n] fp32 (none unsplit). The counters are
// zero between launches: the owner zeroes them at allocation, the last
// split block of a launch resets its own. A split launch has at most
// kMaxCounters output blocks (a wider matrix never splits: natural_splits).
constexpr int kMaxCounters = 256;
constexpr size_t kCountersBytes = static_cast<size_t>(kMaxCounters) * 4;
size_t xh_bytes(int k) { return static_cast<size_t>(kExl3GemvSplitRows) * static_cast<size_t>(k) * 2; }
size_t parts_bytes(int n, int splits) {
  return splits > 1 ? static_cast<size_t>(splits) * kExl3GemvSplitRows * static_cast<size_t>(n) * 4 : 0;
}

template <int kBits, int kMT, typename OutT>
void launch_form(const uint16_t* xh, const Exl3Matrix& w, OutT* out, int rows, int n, int k, size_t out_stride,
                 int splits, float* part, unsigned* counters, cudaStream_t stream) {
  using C = Core<kBits, kMT, 4>;
  static bool opted_in = false;  // the attribute set before this instantiation's first launch
  if (!opted_in) {
    DGPP_CUDA_OK(cudaFuncSetAttribute(exl3_gemv_kernel<kBits, kMT, OutT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(C::kSmemBytes)));
    opted_in = true;
  }
  const dim3 grid(n / kExl3Block, splits);
  exl3_gemv_kernel<kBits, kMT, OutT><<<grid, kThreads, C::kSmemBytes, stream>>>(
      xh, w.tiles, w.svh, out, out_stride, rows, n, k, splits > 1 ? part : nullptr, counters);
  DGPP_CUDA_OK(cudaGetLastError());
}

template <int kBits, typename OutT>
void launch_bits(const uint16_t* act, size_t act_stride, const Exl3Matrix& w, OutT* out, int m, int n, int k,
                 size_t out_stride, int splits, void* ws, cudaStream_t stream) {
  unsigned* counters = static_cast<unsigned*>(ws);
  uint16_t* xh = reinterpret_cast<uint16_t*>(static_cast<uint8_t*>(ws) + kCountersBytes);
  float* part = reinterpret_cast<float*>(static_cast<uint8_t*>(ws) + kCountersBytes + xh_bytes(k));
  if (splits > 1 && n / kExl3Block > kMaxCounters)
    throw std::logic_error("exl3_gemm: a split launch wider than the counter region");  // natural_splits forbids it
  // The 128-row A tile holds two stages of a 6-bit stream in the smem
  // budget, not of an 8-bit one (the indexer's wk): 8-bit launches keep
  // 64-row groups.
  constexpr int kGroup = kBits <= 6 ? kExl3GemvSplitRows : 64;
  for (int row0 = 0; row0 < m; row0 += kGroup) {
    const int rows = std::min(kGroup, m - row0);
    const uint16_t* a = act + static_cast<size_t>(row0) * act_stride;
    OutT* o = out + static_cast<size_t>(row0) * out_stride;
    const int rwarps = rows * (k / kWindowK);
    exl3_rotate_rows_kernel<<<(rwarps + 7) / 8, 256, 0, stream>>>(a, act_stride, w.suh, xh, rows, k);
    DGPP_CUDA_OK(cudaGetLastError());
    if (rows <= 16)
      launch_form<kBits, 1, OutT>(xh, w, o, rows, n, k, out_stride, splits, part, counters, stream);
    else if (rows <= 32)
      launch_form<kBits, 2, OutT>(xh, w, o, rows, n, k, out_stride, splits, part, counters, stream);
    else if (rows <= 64)
      launch_form<kBits, 4, OutT>(xh, w, o, rows, n, k, out_stride, splits, part, counters, stream);
    else if constexpr (kBits <= 6)
      launch_form<kBits, 8, OutT>(xh, w, o, rows, n, k, out_stride, splits, part, counters, stream);
  }
}

template <typename OutT>
void launch(const uint16_t* act, size_t act_stride, const Exl3Matrix& w, OutT* out, int m, int n, int k,
            size_t out_stride, cudaStream_t stream, void* ws, size_t ws_bytes) {
  if (!exl3_gemm_shape_ok(w, act, act_stride, m, n, k))
    throw std::invalid_argument("exl3_gemm: shape or alignment outside the contract (n == rows, k == cols, "
                                "multiples of 128; 16-byte aligned act / tiles; bits 3, 4, 6 or 8)");
  if (out_stride == 0) out_stride = static_cast<size_t>(n);
  if (out_stride < static_cast<size_t>(n)) throw std::invalid_argument("exl3_gemm: out stride");
  if (ws == nullptr || ws_bytes < exl3_gemm_ws_bytes(n, k, 1) || (reinterpret_cast<uintptr_t>(ws) & 15u) != 0)
    throw std::invalid_argument("exl3_gemm: a 16-byte aligned workspace of exl3_gemm_ws_bytes(n, k, splits) is required");
  const int splits = exl3_gemm_splits(n, k, ws, ws_bytes);
  switch (w.bits) {
    case 3: launch_bits<3, OutT>(act, act_stride, w, out, m, n, k, out_stride, splits, ws, stream); break;
    case 4: launch_bits<4, OutT>(act, act_stride, w, out, m, n, k, out_stride, splits, ws, stream); break;
    case 6: launch_bits<6, OutT>(act, act_stride, w, out, m, n, k, out_stride, splits, ws, stream); break;
    case 8: launch_bits<8, OutT>(act, act_stride, w, out, m, n, k, out_stride, splits, ws, stream); break;
    default: throw std::invalid_argument("exl3_gemm: bit width not compiled");
  }
}

// ---- the MoE forms ----------------------------------------------------------

// The view-table base of a slot: its expert's triple, or the shared triple.
__device__ __forceinline__ int view_base_of(const int32_t* __restrict__ ids, int slot, int top_k,
                                            int shared_view_base, bool& shared) {
  const int t = slot / (top_k + 1), j = slot - t * (top_k + 1);
  shared = j == top_k;
  return shared ? shared_view_base : ids[static_cast<size_t>(t) * top_k + j] * 3;
}

// The slots' rotated inputs: for which == 0 the token rows against the
// slot's expert's gate (xh0) and up (xh1) suh; for which == 2 the slot's own
// activation row against its down suh (xh0). One warp per (position,
// 128-k block) over k_max blocks. The rows are laid out by POSITION p of
// the expert-sorted order (order[p] = the logical slot; null = identity),
// so an equal-expert run's rows are contiguous for the run forms below —
// the activation rows (which == 2) are read by position too, as the
// gate-up run form wrote them. A slot whose matrix is narrower (the shared
// expert's inter, an odd expert's slice) stops at its own width.
__global__ void exl3_moe_rotate_slots_kernel(const uint16_t* __restrict__ x, size_t x_stride,
                                             const int32_t* __restrict__ ids, const int32_t* __restrict__ order,
                                             const MoeExpertView* __restrict__ views, int slots, int top_k,
                                             int k_max, int shared_view_base, int n_shared, int which,
                                             uint16_t* __restrict__ xh0, uint16_t* __restrict__ xh1,
                                             size_t xh_stride) {
  dgpp_pdl_wait();
  const int lane = threadIdx.x % 32;
  const int gw = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) / 32) + static_cast<int>(threadIdx.x) / 32;
  const int blocks = k_max / kWindowK;
  if (gw >= slots * blocks) return;
  const int p = gw / blocks, base = (gw - p * blocks) * kWindowK;
  const int slot = order ? order[p] : p;
  bool shared = false;
  const int vb = view_base_of(ids, slot, top_k, shared_view_base, shared);
  if (shared && n_shared == 0) return;
  const int t = slot / (top_k + 1);
  const int xrow = which == 2 ? p : t;
  const MoeExpertView v0 = views[vb + which];
  if (base >= v0.exl3_cols) return;
  const uint16_t* src = x + static_cast<size_t>(xrow) * x_stride + base;
  exl3::rotate_in_warp(src, v0.exl3_suh + base, xh0 + static_cast<size_t>(p) * xh_stride + base, lane);
  if (which == 0) {
    const MoeExpertView v1 = views[vb + 1];
    exl3::rotate_in_warp(src, v1.exl3_suh + base, xh1 + static_cast<size_t>(p) * xh_stride + base, lane);
  }
}

// The run a block of the slot forms takes: position p's equal-expert run
// of L rows (runs[p] from launch_moe_slot_order; 0 inside a run, whose
// blocks return; null runs = every position a run of one, the per-slot
// form). A launch serves one run class — kMT 1 the runs of up to 16 rows,
// kMT 4 the longer ones (up to 64, the decode batch's bound) — and a block
// of the other class returns.
// The block's run: the 16-row tile (kMT 1) at position blockIdx.y, which
// computes when it leads a run of up to kMoeRunTileRows rows (every
// position when `runs` is null); the 64-row tile (kMT 4) from the order
// kernel's group list after the run lengths (glm_moe_launch.hpp): entry
// blockIdx.y = position | rows << 16, -1 past the last group.
template <int kMT>
__device__ __forceinline__ bool run_block_rows(const int32_t* __restrict__ runs, int slots, int* p, int* L) {
  if constexpr (kMT == 1) {
    *p = static_cast<int>(blockIdx.y);
    if (*p >= slots) return false;
    *L = runs ? runs[*p] : 1;
    return *L > 0 && *L <= kMoeRunTileRows;
  } else {
    const int e = runs[slots + static_cast<int>(blockIdx.y)];
    if (e < 0) return false;
    *p = e & 0xFFFF;
    *L = e >> 16;
    return true;
  }
}

// Gate + up + SwiGLU per expert run: block (x, y) = output block x of
// POSITION y of the expert-sorted order; the run leader's block takes its
// run's L rows [y, y + L) — contiguous in xh by position — as one A tile
// of up to kMT x 16 rows, so an expert L rows share is streamed and
// decoded once (2026-10-23: eight identical streams cost the per-slot
// kernels 4.9x one stream's time, the duplicates re-read past what L2
// served; the plan's §7). The gate's and the up's 128 outputs per row
// through the core, rotated and scaled, rounded to bf16 as the other
// formats' slot kernels round their dots, then the SwiGLU op for op (the
// fp8 / fp4 kernels' form) — per row the chain of the one-row form, so
// the act rows are bitwise the per-slot kernel's; written by position.
// kRBits: the routed experts' width; kSBits: the shared expert's.
template <int kRBits, int kSBits, int kMT>
__global__ __launch_bounds__(kThreads) void exl3_moe_slot_gate_up_kernel(
    const uint16_t* __restrict__ xh_g, const uint16_t* __restrict__ xh_u, size_t xh_stride,
    const int32_t* __restrict__ ids, const int32_t* __restrict__ order, const int32_t* __restrict__ runs,
    const MoeExpertView* __restrict__ views, int k, uint16_t* __restrict__ act, int act_stride, int slots,
    int top_k, float limit, int shared_view_base, int n_shared) {
  dgpp_pdl_wait();
  using CR = Core<kRBits, kMT>;
  using CS = Core<kSBits, kMT>;
  static_assert(CR::kRows == CS::kRows, "one A tile height");
  extern __shared__ __align__(16) uint8_t smem[];
  int p, L;
  if (!run_block_rows<kMT>(runs, slots, &p, &L)) return;
  const int slot = order ? order[p] : p;
  bool shared = false;
  const int vb = view_base_of(ids, slot, top_k, shared_view_base, shared);
  if (shared && n_shared == 0) return;
  const MoeExpertView vg = views[vb + 0];
  const MoeExpertView vu = views[vb + 1];
  const int n = vg.exl3_rows;
  const int n0 = static_cast<int>(blockIdx.x) * kExl3Block;
  if (n0 >= n) return;
  const int nwin = k / kWindowK;
  const uint16_t* ag = xh_g + static_cast<size_t>(p) * xh_stride;
  const uint16_t* au = xh_u + static_cast<size_t>(p) * xh_stride;
  float cg[kMT][2][4], cu[kMT][2][4];
  if (shared) {
    const size_t ntb = static_cast<size_t>(k / kExl3Tile) * CS::kTileBytes;
    CS::run(smem, ag, xh_stride, L, vg.payload + static_cast<size_t>(n0 / kExl3Tile) * ntb, ntb, 0, nwin, cg);
    CS::run(smem, au, xh_stride, L, vu.payload + static_cast<size_t>(n0 / kExl3Tile) * ntb, ntb, 0, nwin, cu);
  } else {
    const size_t ntb = static_cast<size_t>(k / kExl3Tile) * CR::kTileBytes;
    CR::run(smem, ag, xh_stride, L, vg.payload + static_cast<size_t>(n0 / kExl3Tile) * ntb, ntb, 0, nwin, cg);
    CR::run(smem, au, xh_stride, L, vu.payload + static_cast<size_t>(n0 / kExl3Tile) * ntb, ntb, 0, nwin, cu);
  }
  // The two [kRows][128] tiles in the stages (free after run): the gate's
  // then the up's.
  float* sG = reinterpret_cast<float*>(smem);
  float* sU = sG + CR::kRows * kExl3Block;
  CR::fragments_to_smem(sG, cg);
  CR::fragments_to_smem(sU, cu);
  __syncthreads();
  {
    // One warp per (row, matrix): the rotation and scale of its 128 outputs.
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int i = warp; i < 2 * L; i += kWarps) {
      const int r = i >> 1;
      float* s = ((i & 1) ? sU : sG) + r * kExl3Block;
      const uint16_t* svh = ((i & 1) ? vu.exl3_svh : vg.exl3_svh) + n0;
      const float4 v4 = *reinterpret_cast<const float4*>(s + 4 * lane);
      float v0 = v4.x, v1 = v4.y, v2 = v4.z, v3 = v4.w;
      exl3::rotate_out_warp(v0, v1, v2, v3, svh, lane);
      *reinterpret_cast<float4*>(s + 4 * lane) = make_float4(v0, v1, v2, v3);
    }
  }
  __syncthreads();
  for (int i = static_cast<int>(threadIdx.x); i < L * kExl3Block; i += kThreads) {
    const int r = i / kExl3Block, c = i - r * kExl3Block;
    float gv = bf16_bits_to_float(float_to_bf16_bits(sG[r * kExl3Block + c]));
    float uv = bf16_bits_to_float(float_to_bf16_bits(sU[r * kExl3Block + c]));
    if (gv > limit) gv = limit;
    uv = fminf(fmaxf(uv, -limit), limit);
    const uint16_t tt = float_to_bf16_bits(gv * sigmoid_acc(gv));
    act[static_cast<size_t>(p + r) * act_stride + n0 + c] = float_to_bf16_bits(bf16_bits_to_float(tt) * uv);
  }
}

// The down projection per expert run: block (x, y) = output block x of
// position y (the run form above; the leader's block takes its run's L
// rotated SwiGLU rows by position); the rows against the run's expert's
// w2 over the matrix's own k (the routed slice or the shared expert's),
// rotated and scaled, stored unrounded (the accumulation chain owns the
// rounding) at the LOGICAL slots' rows — out's layout is the
// accumulation's, whatever the dispatch order.
template <int kRBits, int kSBits, int kMT>
__global__ __launch_bounds__(kThreads) void exl3_moe_slot_down_kernel(
    const uint16_t* __restrict__ xh_d, size_t xh_stride, const int32_t* __restrict__ ids,
    const int32_t* __restrict__ order, const int32_t* __restrict__ runs, const MoeExpertView* __restrict__ views,
    float* __restrict__ out, int out_stride, int slots, int top_k, int shared_view_base, int n_shared) {
  dgpp_pdl_wait();
  using CR = Core<kRBits, kMT>;
  using CS = Core<kSBits, kMT>;
  extern __shared__ __align__(16) uint8_t smem[];
  int p, L;
  if (!run_block_rows<kMT>(runs, slots, &p, &L)) return;
  const int slot = order ? order[p] : p;
  bool shared = false;
  const int vb = view_base_of(ids, slot, top_k, shared_view_base, shared);
  if (shared && n_shared == 0) return;
  const MoeExpertView v = views[vb + 2];
  const int n0 = static_cast<int>(blockIdx.x) * kExl3Block;
  if (n0 >= v.exl3_rows) return;
  const int k = v.exl3_cols;
  const int nwin = k / kWindowK;
  const uint16_t* a = xh_d + static_cast<size_t>(p) * xh_stride;
  float c[kMT][2][4];
  float* sC = reinterpret_cast<float*>(smem);
  if (shared) {
    const size_t ntb = static_cast<size_t>(k / kExl3Tile) * CS::kTileBytes;
    CS::run(smem, a, xh_stride, L, v.payload + static_cast<size_t>(n0 / kExl3Tile) * ntb, ntb, 0, nwin, c);
    CS::fragments_to_smem(sC, c);
  } else {
    const size_t ntb = static_cast<size_t>(k / kExl3Tile) * CR::kTileBytes;
    CR::run(smem, a, xh_stride, L, v.payload + static_cast<size_t>(n0 / kExl3Tile) * ntb, ntb, 0, nwin, c);
    CR::fragments_to_smem(sC, c);
  }
  __syncthreads();
  {
    // One warp per row: rotated, scaled, stored at the row's logical slot.
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int r = warp; r < L; r += kWarps) {
      const float4 v4 = *reinterpret_cast<const float4*>(sC + r * kExl3Block + 4 * lane);
      float v0 = v4.x, v1 = v4.y, v2 = v4.z, v3 = v4.w;
      exl3::rotate_out_warp(v0, v1, v2, v3, v.exl3_svh + n0, lane);
      const int dst = order ? order[p + r] : p + r;
      float* o = out + static_cast<size_t>(dst) * out_stride + n0 + 4 * lane;
      o[0] = v0;
      o[1] = v1;
      o[2] = v2;
      o[3] = v3;
    }
  }
}

// The grouped segments' rotated inputs: segment y's rows (already in
// segment order in `act`) against the segment's expert's suh for matrix
// `which` (and, for which == 0, the up matrix's into xh1). One warp per
// (row, 128-k block) of the segment; grid.x covers max_rows x blocks.
__global__ void exl3_moe_rotate_segments_kernel(const uint16_t* __restrict__ act, size_t act_stride,
                                                const MoeSegment* __restrict__ segs,
                                                const MoeExpertView* __restrict__ views, int which, int k,
                                                uint16_t* __restrict__ xh0, uint16_t* __restrict__ xh1,
                                                size_t xh_stride) {
  dgpp_pdl_wait();
  const int lane = threadIdx.x % 32;
  const MoeSegment seg = segs[blockIdx.y];
  if (seg.rows <= 0) return;
  const int gw = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) / 32) + static_cast<int>(threadIdx.x) / 32;
  const int blocks = k / kWindowK;
  if (gw >= seg.rows * blocks) return;
  const int i = gw / blocks, base = (gw - i * blocks) * kWindowK;
  const size_t row = static_cast<size_t>(seg.row0) + i;
  const MoeExpertView v0 = views[seg.expert * 3 + which];
  if (base >= v0.exl3_cols) return;  // the launch's k is the widest matrix's; this expert's may be narrower
  const uint16_t* src = act + row * act_stride + base;
  exl3::rotate_in_warp(src, v0.exl3_suh + base, xh0 + row * xh_stride + base, lane);
  if (which == 0) {
    const MoeExpertView v1 = views[seg.expert * 3 + 1];
    exl3::rotate_in_warp(src, v1.exl3_suh + base, xh1 + row * xh_stride + base, lane);
  }
}

// The grouped kernel: block (x, y) = output block (x % n_blocks) of 64-row
// group (x / n_blocks) of segment y — the segment's rows against its
// expert's matrix `which`, the core at up to four A tiles, the rows
// rotated and scaled into out rows [seg.row0 + 64 g, ...). One width per
// launch (the routed segments' or the shared segment's).
template <int kBits, typename OutT>
__global__ __launch_bounds__(kThreads) void exl3_moe_grouped_kernel(const uint16_t* __restrict__ xh, size_t xh_stride,
                                                                    const MoeSegment* __restrict__ segs,
                                                                    const MoeExpertView* __restrict__ views, int which,
                                                                    int n_blocks, OutT* __restrict__ out,
                                                                    size_t out_stride, int k) {
  dgpp_pdl_wait();
  using C = Core<kBits, 4>;
  extern __shared__ __align__(16) uint8_t smem[];
  const MoeSegment seg = segs[blockIdx.y];
  const int mg = static_cast<int>(blockIdx.x) / n_blocks, nb = static_cast<int>(blockIdx.x) - mg * n_blocks;
  const int row0 = mg * C::kRows;
  if (seg.rows <= 0 || row0 >= seg.rows) return;
  const int rows = seg.rows - row0 < C::kRows ? seg.rows - row0 : C::kRows;
  const MoeExpertView v = views[seg.expert * 3 + which];
  const int n0 = nb * kExl3Block;
  if (n0 >= v.exl3_rows) return;
  // The segment's own K: the launch's k is the widest matrix's, and the
  // down projection's inter slice varies by expert (exl3_expert_block_split).
  const int kk = static_cast<int>(v.exl3_cols) < k ? static_cast<int>(v.exl3_cols) : k;
  const size_t ntb = static_cast<size_t>(kk / kExl3Tile) * C::kTileBytes;
  const size_t grow = static_cast<size_t>(seg.row0) + row0;
  float c[4][2][4];
  C::run(smem, xh + grow * xh_stride, xh_stride, rows, v.payload + static_cast<size_t>(n0 / kExl3Tile) * ntb, ntb, 0,
         kk / kWindowK, c);
  float* sC = reinterpret_cast<float*>(smem);
  C::fragments_to_smem(sC, c);
  __syncthreads();
  C::rotate_store(sC, rows, v.exl3_svh + n0, out + grow * out_stride + n0, out_stride);
}

// The slot forms' two run classes: the 16-row tile (kMT 1, the deep ring)
// serves every run of up to 16 rows — every block when `runs` is null —
// and the 64-row tile (kMT 4) the longer runs' groups of up to 64 rows,
// which exist only when the batch has more than 16 token rows (a routed
// expert appears at most once per row; the shared expert's run is every
// row). The 64-row launch's grid is the group list's bound (a run that
// spans more than 64 rows is several groups), so its blocks all compute.
template <int kR, int kS, int kMT>
void launch_slot_gate_up_mt(const uint16_t* xh_g, const uint16_t* xh_u, size_t xh_stride, const int32_t* ids,
                            const int32_t* order, const int32_t* runs, const MoeExpertView* views, int n_max, int k,
                            uint16_t* act, int act_stride, int slots, int top_k, float limit, int shared_view_base,
                            int n_shared, cudaStream_t stream) {
  using CR = Core<kR, kMT>;
  using CS = Core<kS, kMT>;
  constexpr size_t kSmem = CR::kSmemBytes > CS::kSmemBytes ? CR::kSmemBytes : CS::kSmemBytes;
  static_assert(kSmem >= 2 * CR::kCBytes && kSmem >= 2 * CS::kCBytes, "the gate and up tiles fit the stages");
  static bool opted_in = false;
  if (!opted_in) {
    DGPP_CUDA_OK(cudaFuncSetAttribute(exl3_moe_slot_gate_up_kernel<kR, kS, kMT>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmem)));
    opted_in = true;
  }
  const dim3 grid(n_max / kExl3Block, kMT == 1 ? slots : moe_run_groups_max(slots));
  exl3_moe_slot_gate_up_kernel<kR, kS, kMT><<<grid, kThreads, kSmem, stream>>>(
      xh_g, xh_u, xh_stride, ids, order, runs, views, k, act, act_stride, slots, top_k, limit, shared_view_base,
      n_shared);
  DGPP_CUDA_OK(cudaGetLastError());
}
template <int kR, int kS>
void launch_slot_gate_up(const uint16_t* xh_g, const uint16_t* xh_u, size_t xh_stride, const int32_t* ids,
                         const int32_t* order, const int32_t* runs, const MoeExpertView* views, int n_max, int k,
                         uint16_t* act, int act_stride, int slots, int top_k, float limit, int shared_view_base,
                         int n_shared, cudaStream_t stream) {
  launch_slot_gate_up_mt<kR, kS, 1>(xh_g, xh_u, xh_stride, ids, order, runs, views, n_max, k, act, act_stride, slots,
                                    top_k, limit, shared_view_base, n_shared, stream);
  if (runs && slots / (top_k + 1) > kMoeRunTileRows)
    launch_slot_gate_up_mt<kR, kS, 4>(xh_g, xh_u, xh_stride, ids, order, runs, views, n_max, k, act, act_stride,
                                      slots, top_k, limit, shared_view_base, n_shared, stream);
}
template <int kR, int kS, int kMT>
void launch_slot_down_mt(const uint16_t* xh_d, size_t xh_stride, const int32_t* ids, const int32_t* order,
                         const int32_t* runs, const MoeExpertView* views, int n, float* out, int out_stride,
                         int slots, int top_k, int shared_view_base, int n_shared, cudaStream_t stream) {
  using CR = Core<kR, kMT>;
  using CS = Core<kS, kMT>;
  constexpr size_t kSmem = CR::kSmemBytes > CS::kSmemBytes ? CR::kSmemBytes : CS::kSmemBytes;
  static bool opted_in = false;
  if (!opted_in) {
    DGPP_CUDA_OK(cudaFuncSetAttribute(exl3_moe_slot_down_kernel<kR, kS, kMT>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmem)));
    opted_in = true;
  }
  const dim3 grid(n / kExl3Block, kMT == 1 ? slots : moe_run_groups_max(slots));
  exl3_moe_slot_down_kernel<kR, kS, kMT><<<grid, kThreads, kSmem, stream>>>(
      xh_d, xh_stride, ids, order, runs, views, out, out_stride, slots, top_k, shared_view_base, n_shared);
  DGPP_CUDA_OK(cudaGetLastError());
}
template <int kR, int kS>
void launch_slot_down(const uint16_t* xh_d, size_t xh_stride, const int32_t* ids, const int32_t* order,
                      const int32_t* runs, const MoeExpertView* views, int n, float* out, int out_stride, int slots,
                      int top_k, int shared_view_base, int n_shared, cudaStream_t stream) {
  launch_slot_down_mt<kR, kS, 1>(xh_d, xh_stride, ids, order, runs, views, n, out, out_stride, slots, top_k,
                                 shared_view_base, n_shared, stream);
  if (runs && slots / (top_k + 1) > kMoeRunTileRows)
    launch_slot_down_mt<kR, kS, 4>(xh_d, xh_stride, ids, order, runs, views, n, out, out_stride, slots, top_k,
                                   shared_view_base, n_shared, stream);
}
// The compiled (routed, shared) width pairs: the pack's experts at 3 or 4
// bits with the shared expert at 6, the drafts at 6 / 6, and the uniform
// forms the tests use.
template <class F>
void dispatch_pair(int rb, int sb, F&& f) {
  if (rb == 3 && sb == 6) return f(std::integral_constant<int, 3>{}, std::integral_constant<int, 6>{});
  if (rb == 4 && sb == 6) return f(std::integral_constant<int, 4>{}, std::integral_constant<int, 6>{});
  if (rb == 6 && sb == 6) return f(std::integral_constant<int, 6>{}, std::integral_constant<int, 6>{});
  if (rb == 3 && sb == 3) return f(std::integral_constant<int, 3>{}, std::integral_constant<int, 3>{});
  if (rb == 4 && sb == 4) return f(std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{});
  if (rb == 8 && sb == 8) return f(std::integral_constant<int, 8>{}, std::integral_constant<int, 8>{});
  if (rb == 4 && sb == 8) return f(std::integral_constant<int, 4>{}, std::integral_constant<int, 8>{});
  throw std::invalid_argument("exl3 moe: the (routed, shared) width pair " + std::to_string(rb) + "/" +
                              std::to_string(sb) + " is not compiled");
}
template <int kBits, typename OutT>
void launch_grouped(const uint16_t* xh, size_t xh_stride, const MoeSegment* segs, int n_segs, int max_rows,
                    const MoeExpertView* views, int which, OutT* out, size_t out_stride, int n, int k,
                    cudaStream_t stream) {
  using C = Core<kBits, 4>;
  static bool opted_in = false;
  if (!opted_in) {
    DGPP_CUDA_OK(cudaFuncSetAttribute(exl3_moe_grouped_kernel<kBits, OutT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(C::kSmemBytes)));
    opted_in = true;
  }
  const int n_blocks = n / kExl3Block;
  const int groups = (max_rows + C::kRows - 1) / C::kRows;
  const dim3 grid(static_cast<unsigned>(n_blocks * groups), static_cast<unsigned>(n_segs));
  exl3_moe_grouped_kernel<kBits, OutT><<<grid, kThreads, C::kSmemBytes, stream>>>(xh, xh_stride, segs, views, which,
                                                                                   n_blocks, out, out_stride, k);
  DGPP_CUDA_OK(cudaGetLastError());
}
template <typename OutT>
void launch_grouped_bits(int bits, const uint16_t* xh, size_t xh_stride, const MoeSegment* segs, int n_segs,
                         int max_rows, const MoeExpertView* views, int which, OutT* out, size_t out_stride, int n,
                         int k, cudaStream_t stream) {
  if (n_segs <= 0 || max_rows <= 0) return;
  if (!xh || !segs || !views || !out || n <= 0 || k <= 0 || n % kExl3Block != 0 || k % kExl3Block != 0 ||
      (xh_stride * 2) % 16 != 0 || out_stride < static_cast<size_t>(n))
    throw std::invalid_argument("exl3_moe_grouped: shape outside the contract");
  switch (bits) {
    case 3: launch_grouped<3, OutT>(xh, xh_stride, segs, n_segs, max_rows, views, which, out, out_stride, n, k, stream); break;
    case 4: launch_grouped<4, OutT>(xh, xh_stride, segs, n_segs, max_rows, views, which, out, out_stride, n, k, stream); break;
    case 6: launch_grouped<6, OutT>(xh, xh_stride, segs, n_segs, max_rows, views, which, out, out_stride, n, k, stream); break;
    case 8: launch_grouped<8, OutT>(xh, xh_stride, segs, n_segs, max_rows, views, which, out, out_stride, n, k, stream); break;
    default: throw std::invalid_argument("exl3_moe_grouped: bit width not compiled");
  }
}

}  // namespace

bool exl3_gemm_shape_ok(const Exl3Matrix& w, const void* act, size_t act_stride, int m, int n, int k) {
  if (m < 1 || n <= 0 || k <= 0 || w.rows != n || w.cols != k) return false;
  if (n % kExl3Block != 0 || k % kExl3Block != 0) return false;
  if (w.bits != 3 && w.bits != 4 && w.bits != 6 && w.bits != 8) return false;
  if (w.tiles == nullptr || w.suh == nullptr || w.svh == nullptr || act == nullptr) return false;
  if ((reinterpret_cast<uintptr_t>(w.tiles) & 15u) != 0 || (reinterpret_cast<uintptr_t>(act) & 15u) != 0) return false;
  if ((act_stride * 2) % 16 != 0) return false;
  if ((reinterpret_cast<uintptr_t>(w.suh) & 7u) != 0 || (reinterpret_cast<uintptr_t>(w.svh) & 7u) != 0) return false;
  return true;
}

size_t exl3_gemm_ws_bytes(int n, int k, int splits) { return kCountersBytes + xh_bytes(k) + parts_bytes(n, splits); }

namespace {
int natural_splits(int n, int k) {
  const int blocks = n / kExl3Block;
  const int fill = split_target_blocks();
  if (blocks >= fill) return 1;
  const int windows = k / kWindowK;
  const int s = std::min({(fill + blocks - 1) / blocks, windows, kMaxSplit});
  return s < 2 ? 1 : s;
}
}  // namespace

int exl3_gemm_splits(int n, int k, const void* ws, size_t ws_bytes) {
  if (ws == nullptr) return 1;
  const int s = natural_splits(n, k);
  if (s < 2 || exl3_gemm_ws_bytes(n, k, s) > ws_bytes) return 1;
  return s;
}

size_t exl3_gemm_ws_bytes_for(int n, int k) { return exl3_gemm_ws_bytes(n, k, natural_splits(n, k)); }

void launch_exl3_gemm_bf16(const uint16_t* act, size_t act_stride, const Exl3Matrix& w, uint16_t* out, int m, int n,
                           int k, size_t out_stride, cudaStream_t stream, void* ws, size_t ws_bytes) {
  launch<uint16_t>(act, act_stride, w, out, m, n, k, out_stride, stream, ws, ws_bytes);
}
void launch_exl3_gemm_f32(const uint16_t* act, size_t act_stride, const Exl3Matrix& w, float* out, int m, int n, int k,
                          size_t out_stride, cudaStream_t stream, void* ws, size_t ws_bytes) {
  launch<float>(act, act_stride, w, out, m, n, k, out_stride, stream, ws, ws_bytes);
}

void launch_exl3_moe_rotate_slots(const uint16_t* x, size_t x_stride, const int32_t* ids, const int32_t* order,
                                  const MoeExpertView* views, int slots, int top_k, int k_max, int shared_view_base,
                                  int n_shared, int which, uint16_t* xh0, uint16_t* xh1, size_t xh_stride,
                                  cudaStream_t stream) {
  if (slots <= 0) return;
  if (!x || !ids || !views || !xh0 || (which == 0 && !xh1) || (which != 0 && which != 2) || k_max <= 0 ||
      k_max % kExl3Block != 0 || (x_stride * 2) % 8 != 0 || (xh_stride * 2) % 8 != 0 || shared_view_base < 0)
    throw std::invalid_argument("exl3_moe_rotate_slots: arguments outside the contract");
  const int warps = slots * (k_max / kWindowK);
  exl3_moe_rotate_slots_kernel<<<(warps + 7) / 8, 256, 0, stream>>>(
      x, x_stride, ids, order, views, slots, top_k, k_max, shared_view_base, n_shared, which, xh0, xh1, xh_stride);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_exl3_moe_slot_gate_up_swiglu(const uint16_t* xh_g, const uint16_t* xh_u, size_t xh_stride,
                                         const int32_t* ids, const int32_t* order, const int32_t* runs,
                                         const MoeExpertView* views, int n_max, int k, int routed_bits,
                                         int shared_bits, uint16_t* act, int act_stride, int slots, int top_k,
                                         float limit, int shared_view_base, int n_shared, cudaStream_t stream) {
  if (slots <= 0) return;
  if (!xh_g || !xh_u || !ids || !views || !act || n_max <= 0 || n_max % kExl3Block != 0 || k <= 0 ||
      k % kExl3Block != 0 || (xh_stride * 2) % 16 != 0 || act_stride < n_max || shared_view_base < 0 ||
      (runs && !order) || slots > 1024)
    throw std::invalid_argument("exl3_moe_slot_gate_up_swiglu: arguments outside the contract");
  dispatch_pair(routed_bits, shared_bits, [&](auto r, auto s) {
    launch_slot_gate_up<decltype(r)::value, decltype(s)::value>(xh_g, xh_u, xh_stride, ids, order, runs, views, n_max,
                                                               k, act, act_stride, slots, top_k, limit,
                                                               shared_view_base, n_shared, stream);
  });
}

void launch_exl3_moe_slot_down(const uint16_t* xh_d, size_t xh_stride, const int32_t* ids, const int32_t* order,
                               const int32_t* runs, const MoeExpertView* views, int n, int routed_bits,
                               int shared_bits, float* out, int out_stride, int slots, int top_k,
                               int shared_view_base, int n_shared, cudaStream_t stream) {
  if (slots <= 0) return;
  if (!xh_d || !ids || !views || !out || n <= 0 || n % kExl3Block != 0 || (xh_stride * 2) % 16 != 0 ||
      out_stride < n || shared_view_base < 0 || (runs && !order) || slots > 1024)
    throw std::invalid_argument("exl3_moe_slot_down: arguments outside the contract");
  dispatch_pair(routed_bits, shared_bits, [&](auto r, auto s) {
    launch_slot_down<decltype(r)::value, decltype(s)::value>(xh_d, xh_stride, ids, order, runs, views, n, out,
                                                            out_stride, slots, top_k, shared_view_base, n_shared,
                                                            stream);
  });
}

void launch_exl3_moe_rotate_segments(const uint16_t* act, size_t act_stride, const MoeSegment* segs, int n_segs,
                                     int max_rows, const MoeExpertView* views, int which, int k, uint16_t* xh0,
                                     uint16_t* xh1, size_t xh_stride, cudaStream_t stream) {
  if (n_segs <= 0 || max_rows <= 0) return;
  if (!act || !segs || !views || !xh0 || (which == 0 && !xh1) || (which != 0 && which != 2) || k <= 0 ||
      k % kExl3Block != 0 || (act_stride * 2) % 8 != 0 || (xh_stride * 2) % 8 != 0)
    throw std::invalid_argument("exl3_moe_rotate_segments: arguments outside the contract");
  const int warps = max_rows * (k / kWindowK);
  const dim3 grid(static_cast<unsigned>((warps + 7) / 8), static_cast<unsigned>(n_segs));
  exl3_moe_rotate_segments_kernel<<<grid, 256, 0, stream>>>(act, act_stride, segs, views, which, k, xh0, xh1,
                                                            xh_stride);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_exl3_moe_grouped_bf16(const uint16_t* xh, size_t xh_stride, const MoeSegment* segs, int n_segs,
                                  int max_rows, const MoeExpertView* views, int which, int bits, uint16_t* out,
                                  size_t out_stride, int n, int k, cudaStream_t stream) {
  launch_grouped_bits<uint16_t>(bits, xh, xh_stride, segs, n_segs, max_rows, views, which, out, out_stride, n, k, stream);
}
void launch_exl3_moe_grouped_f32(const uint16_t* xh, size_t xh_stride, const MoeSegment* segs, int n_segs,
                                 int max_rows, const MoeExpertView* views, int which, int bits, float* out,
                                 size_t out_stride, int n, int k, cudaStream_t stream) {
  launch_grouped_bits<float>(bits, xh, xh_stride, segs, n_segs, max_rows, views, which, out, out_stride, n, k, stream);
}

}  // namespace dgpp
