#include "kernels/scale_gemm.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/bf16_gemv.cuh"
#include "kernels/fp8_blockwise_dense.hpp"
#include "kernels/fp8_gemv.cuh"
#include "kernels/fp8_gemv.cuh"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/mma_gemv.hpp"

namespace dgpp {
namespace {

// Tile geometry. BN and BK both divide the 128-wide scale block, which is
// what lets every stage apply one scalar scale: the n-range [n0, n0+BN)
// (n0 a multiple of BN, BN | 128) lies inside scale row n0/128, and each
// k-slice [k0, k0+BK) (BK | 128) lies inside scale column k0/128.
constexpr int BM = 16;
constexpr int BN = 64;
// Rows at or below which the chunked GEMV beats the tile kernel (measured
// 2026-09-04: the crossover sits near 128-256 rows for the MoE experts'
// n=512 slabs; the tile kernel's grid is n/64 x m/16 blocks).
constexpr int kGemvMaxM = 128;
constexpr int BK = 32;
constexpr int BK_PAD = BK + 8;  // u16 pad breaks the worst bank conflicts
constexpr int kBlockThreads = (BN / 8) * 32;  // 8 warps, one n8 group each

// rs / cs: the scale grid as log2 block sizes (7 = the checkpoint's 128;
// a TP slice's re-blocked axis 6 or 5 — docs/qwen38_flash_next_plan.md
// D2). Rows read their own scale row (n >> rs); a 32-deep stage lies in
// one scale column for cs >= 5.
template <typename OutT>
__global__ void scale_gemm_kernel(const uint16_t* __restrict__ act,
                                  size_t act_stride,
                                  const uint8_t* __restrict__ w,
                                  const float* __restrict__ scales,
                                  OutT* __restrict__ out, int m, int n,
                                  int k, size_t out_stride, int rs, int cs) {
  const int n0 = blockIdx.x * BN;
  const int m0 = blockIdx.y * BM;
  const int scale_cols = (k + (1 << cs) - 1) >> cs;

  __shared__ uint16_t sA[BM][BK_PAD];
  __shared__ uint16_t sB[BN][BK_PAD];

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  // m16n8k16 fragment coordinates (PTX register layouts): A rows lane/4
  // (+8), k columns (lane%4)*2 (+1, +8); B n = lane/4 over the same k
  // columns; C rows lane/4 (+8), n columns (lane%4)*2 (+1).
  const int r = lane / 4;
  const int cc = (lane % 4) * 2;
  const int bnr = warp * 8 + r;  // this thread's B row (n) within the tile

  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

  for (int k0 = 0; k0 < k; k0 += BK) {
    const int scale_col = k0 >> cs;  // one scale column per stage (BK | 2^cs)

    // Weight tile: decode + scale + one BF16 round — the exact operation
    // of the dequant bridge, so weight-tile bits cannot diverge from it.
    for (int idx = threadIdx.x; idx < BN * BK; idx += kBlockThreads) {
      const int nn = idx / BK, kk = idx % BK;
      const int gn = n0 + nn, gk = k0 + kk;
      const float s = scales[(size_t)(gn >> rs) * scale_cols + scale_col];
      sB[nn][kk] =
          (gn < n && gk < k)
              ? float_to_bf16_bits(
                    fp8_e4m3_bits_to_float(w[(size_t)gn * k + gk]) * s)
              : 0;
    }
    // Activation tile (zero-fill outside [M, K)).
    for (int idx = threadIdx.x; idx < BM * BK; idx += kBlockThreads) {
      const int mm = idx / BK, kk = idx % BK;
      const int gm = m0 + mm, gk = k0 + kk;
      sA[mm][kk] =
          (gm < m && gk < k) ? act[(size_t)gm * act_stride + gk] : 0;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      const uint32_t a0 = static_cast<uint32_t>(sA[r][kk + cc]) |
                          (static_cast<uint32_t>(sA[r][kk + cc + 1]) << 16);
      const uint32_t a1 =
          static_cast<uint32_t>(sA[r + 8][kk + cc]) |
          (static_cast<uint32_t>(sA[r + 8][kk + cc + 1]) << 16);
      const uint32_t a2 =
          static_cast<uint32_t>(sA[r][kk + cc + 8]) |
          (static_cast<uint32_t>(sA[r][kk + cc + 9]) << 16);
      const uint32_t a3 =
          static_cast<uint32_t>(sA[r + 8][kk + cc + 8]) |
          (static_cast<uint32_t>(sA[r + 8][kk + cc + 9]) << 16);
      const uint32_t b0 =
          static_cast<uint32_t>(sB[bnr][kk + cc]) |
          (static_cast<uint32_t>(sB[bnr][kk + cc + 1]) << 16);
      const uint32_t b1 =
          static_cast<uint32_t>(sB[bnr][kk + cc + 8]) |
          (static_cast<uint32_t>(sB[bnr][kk + cc + 9]) << 16);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    __syncthreads();  // tile reads done before the next stage overwrites
  }

  // Epilogue: C fragment (m16n8) — thread holds (r, cc), (r, cc+1),
  // (r+8, cc), (r+8, cc+1) within its warp's [16 x 8] output slice.
  const int out_col = n0 + warp * 8 + cc;
  auto store = [&](int row_off, int col_off, float v) {
    const int gm = m0 + row_off;
    const int gn = out_col + col_off;
    if (gm < m && gn < n)
      fp8_gemv::store_dot(out + (size_t)gm * out_stride + gn, v);
  };
  store(r, 0, c0);
  store(r, 1, c1);
  store(r + 8, 0, c2);
  store(r + 8, 1, c3);
}

// The small-m path (m <= fp8_gemv::kMaxRows): the bandwidth GEMV core, one
// warp per weight row, activations staged in dynamic smem. Same dequant
// values as the tile kernel above, a different (deterministic) fp32
// accumulation order — see fp8_gemv.cuh.
// rs / cs: the scale grid as log2 block sizes (7 = 128 x 128; 5 = the
// DeepSeek-V4.1 release's 32 x 32, 2026-09-13; the GEMV core reads any
// grid whose column block covers a 16-element chunk, i.e. cs >= 4).
// A block streams a CONTIGUOUS span of rows, kWarps at a time, over one
// staging of the activations (2026-09-29): the model's 65 GB of weights per
// pass are 32K 2 MB pages, and a grid whose blocks interleave across the SMs
// has every SM walk every page of a matrix — an 8 GiB cold ring in the
// isolated bench reproduces the step's kernel times to the microsecond
// (the multi GEMV 167 → 191 us, the out projection 68 → 78). Spans were
// meant to keep each SM's pages to its own few and stage the rows once
// per span instead of once per eight rows; measured on the 8 GiB ring
// (DGPP_GEMV_BLOCKS 96 / 192 / 384 / 768 against 0): within ±4 % of the
// one-group grid on every shape — the misses are served by a shared
// walker, whichever SM issues them. The form stays for experiments
// (gemv_span_blocks(): the target grid; 0, the default, is one row group
// per block). Every row's chain is block_rows' — bitwise either way.
int gemv_span_blocks() {
  static const int blocks = [] {
    const char* v = std::getenv("DGPP_GEMV_BLOCKS");
    if (v == nullptr) return 0;
    const long b = std::strtol(v, nullptr, 10);
    return (b >= 0 && b <= 4096) ? static_cast<int>(b) : 0;
  }();
  return blocks;
}
// The span (rows per block, a multiple of kWarps) that puts `groups`
// row groups on about gemv_span_blocks() blocks.
inline int gemv_span_rows(int groups) {
  const int G = gemv_span_blocks();
  if (G <= 0 || groups <= G) return fp8_gemv::kWarps;
  return ((groups + G - 1) / G) * fp8_gemv::kWarps;
}

template <int kRows, typename OutT>
__global__ void scale_gemv_kernel(const uint16_t* __restrict__ act,
                                  size_t act_stride,
                                  const uint8_t* __restrict__ w,
                                  const float* __restrict__ scales,
                                  OutT* __restrict__ out, int n, int k,
                                  size_t out_stride, int rs, int cs, int span) {
  extern __shared__ __align__(16) uint16_t sx[];
  fp8_gemv::stage_activations<kRows>(act, act_stride, k, sx);
  __syncthreads();
  const int b0 = static_cast<int>(blockIdx.x) * span;
  const int b1 = min(n, b0 + span);
  for (int n0 = b0; n0 < b1; n0 += fp8_gemv::kWarps)
    fp8_gemv::block_rows<kRows>(w, scales, sx, n0, n, k, out, out_stride, rs, cs);
}

template <int kRows, typename OutT>
void launch_scale_gemv(const uint16_t* act, size_t act_stride,
                       const uint8_t* w, const float* scales, OutT* out, int n,
                       int k, size_t out_stride, cudaStream_t stream, int rs = 7,
                       int cs = 7) {
  const int groups = (n + fp8_gemv::kWarps - 1) / fp8_gemv::kWarps;
  const int span = gemv_span_rows(groups);
  const dim3 grid((n + span - 1) / span);
  scale_gemv_kernel<kRows, OutT>
      <<<grid, fp8_gemv::kThreads, fp8_gemv::smem_bytes(kRows, k), stream>>>(
          act, act_stride, w, scales, out, n, k, out_stride, rs, cs, span);
  DGPP_CUDA_OK(cudaGetLastError());
}

template <typename OutT>
void launch_scale_gemv_rows(const uint16_t* act, size_t act_stride,
                            const uint8_t* w, const float* scales, OutT* out,
                            int rows, int n, int k, size_t out_stride,
                            cudaStream_t stream, int rs = 7, int cs = 7) {
  switch (rows) {
    case 1:
      launch_scale_gemv<1, OutT>(act, act_stride, w, scales, out, n, k,
                                 out_stride, stream, rs, cs);
      return;
    case 2:
      launch_scale_gemv<2, OutT>(act, act_stride, w, scales, out, n, k,
                                 out_stride, stream, rs, cs);
      return;
    case 3:
      launch_scale_gemv<3, OutT>(act, act_stride, w, scales, out, n, k,
                                 out_stride, stream, rs, cs);
      return;
    case 4:
      launch_scale_gemv<4, OutT>(act, act_stride, w, scales, out, n, k,
                                 out_stride, stream, rs, cs);
      return;
    default:
      throw std::invalid_argument("scale_gemv: rows outside [1,4]");
  }
}

// One dispatch for both output dtypes: the epilogue store is the only
// difference between the bf16 and fp32 products (same tiles, same GEMV
// core, same accumulation order), so a value that rounds to bf16 in one
// is the unrounded fp32 of the other.
inline void launch_dense_mma(const uint16_t* act, size_t act_stride,
                             const uint8_t* payload, const float* scales,
                             uint16_t* out, int m, int n, int k,
                             size_t out_stride, cudaStream_t stream) {
  launch_dense_mma_bf16(act, act_stride, payload, scales, out, m, n, k, stream,
                        out_stride);
}
inline void launch_dense_mma(const uint16_t* act, size_t act_stride,
                             const uint8_t* payload, const float* scales,
                             float* out, int m, int n, int k,
                             size_t out_stride, cudaStream_t stream) {
  launch_dense_mma_f32(act, act_stride, payload, scales, out, m, n, k, stream,
                       out_stride);
}

template <typename OutT>
void launch_scale_gemm(const uint16_t* act, size_t act_row_stride_elems,
                       const uint8_t* w_payload, const float* w_scales,
                       OutT* out, int m, int n, int k, cudaStream_t stream,
                       size_t out_stride, int mma_from_rows, bool last_row_only = false,
                       void* ws = nullptr, size_t ws_bytes = 0, int compact_row = -1) {
  if (m <= 0 || n <= 0) return;  // empty output by definition
  if (!act || !w_payload || !w_scales || !out)
    throw std::invalid_argument("scale_gemm: null pointer");
  if (out_stride == 0) out_stride = static_cast<size_t>(n);
  if (out_stride < static_cast<size_t>(n))
    throw std::invalid_argument("scale_gemm: output row stride narrower than n");
  if (compact_row < -1 || compact_row >= m || (compact_row >= 0 && last_row_only))
    throw std::invalid_argument("scale_gemm: invalid compact row selection");
  const int dispatch_rows = m;
  const bool selected_only = last_row_only || compact_row >= 0;
  const int selected_row = compact_row >= 0 ? compact_row : m - 1;
  const bool streaming_mma = mma_from_rows > 0 && m >= mma_from_rows && m <= kScaleGemmMmaMaxRows &&
                            mma_gemv_shape_ok(w_payload, act, act_row_stride_elems, m, k);
  if (selected_only) {
    act += static_cast<size_t>(selected_row) * act_row_stride_elems;
    if (last_row_only) out += static_cast<size_t>(selected_row) * out_stride;
    m = 1;
  }
  if (k <= 0) {
    // Degenerate contraction: zero outputs (matches the fp64 oracle).
    DGPP_CUDA_OK(cudaMemset2DAsync(out, out_stride * sizeof(OutT), 0,
                                   static_cast<size_t>(n) * sizeof(OutT),
                                   static_cast<size_t>(m), stream));
    return;
  }
  // The streaming tensor-core form between its bounds (the header's
  // table); a shape it cannot take (k % 64, alignment) falls through.
  if (streaming_mma) {
    // A last-row request must retain the full product's reduction order.
    // Only its final group can contain the selected row; groups above the
    // decode bound run unsplit even when a workspace was supplied.
    const int group_start = selected_row / kMmaGemvMaxRowsPerLaunch * kMmaGemvMaxRowsPerLaunch;
    const int selected_group_rows = std::min(kMmaGemvMaxRowsPerLaunch, dispatch_rows - group_start);
    if (selected_only && selected_group_rows > kMmaGemvMaxRows) {
      ws = nullptr;
      ws_bytes = 0;
    }
    if constexpr (std::is_same_v<OutT, float>)
      launch_mma_gemv_fp8_f32(act, act_row_stride_elems, w_payload, w_scales, out, m, n, k,
                              out_stride, 7, 7, stream, ws, ws_bytes);
    else
      launch_mma_gemv_fp8_bf16(act, act_row_stride_elems, w_payload, w_scales, out, m, n, k,
                               out_stride, 7, 7, stream, ws, ws_bytes);
    return;
  }
  // Small-M calls take the row-independent bandwidth GEMV (the tile below
  // is latency-bound at small m — see fp8_gemv.cuh): one GEMV launch
  // carries at most four rows (fewer when K fills the 48-KiB smem budget),
  // and the rows are chunked through the same scalar-order core, so each
  // output row is bitwise invariant within the GEMV path. The tensor-core
  // paths have different accumulation orders. The threshold was 8 (the
  // decode-row bound); the 2026-09-04 prefill profile found the tile kernel
  // at 578 us for the MoE experts'
  // 9..30-row segments (grid 8 x 1..2 blocks on a 48-SM part) against 15 us
  // per four-row GEMV launch, so prefill-sized segments go this way too.
  // The tile kernel keeps the large-m shapes where its grid fills the GPU.
  if (dispatch_rows <= kGemvMaxM && fp8_gemv::shape_ok(w_payload, /*rows=*/1, k)) {
    {
      static int trace_left = -1;  // the DGPP_MMA_TRACE budget, read once
      if (trace_left < 0) {
        const char* e = std::getenv("DGPP_MMA_TRACE");
        trace_left = (e && *e && *e != '0') ? (std::atoi(e) > 0 ? std::atoi(e) : 64) : 0;
      }
      if (trace_left > 0) {
        --trace_left;
        std::fprintf(stderr, "scale_gemv m=%d n=%d k=%d\n", m, n, k);
        std::fflush(stderr);
      }
    }
    for (int row0 = 0; row0 < m;) {
      int rows = std::min(fp8_gemv::kMaxRows, m - row0);
      while (!fp8_gemv::shape_ok(w_payload, rows, k)) --rows;
      launch_scale_gemv_rows(
          act + static_cast<size_t>(row0) * act_row_stride_elems,
          act_row_stride_elems, w_payload, w_scales,
          out + static_cast<size_t>(row0) * out_stride, rows, n, k, out_stride,
          stream);
      row0 += rows;
    }
    return;
  }
  // Native blockwise-FP8 dense GEMM for prefill (Path B): the weight stays
  // E4M3 and the activation is quantized per call in 128x128 blocks, so the
  // FP8-dequant + cuBLASLt bridge below is skipped. Env-gated
  // (DGPP_FP8BW_DENSE, off unless set) until the parity anchors validate it
  // end to end; any shape outside the 128-block contract falls through.
  if (m > 256 && (k % 128) == 0 && fp8_blockwise_dense_enabled() &&
      fp8_blockwise_dense_supported(act, act_row_stride_elems, w_payload, m, n, k)) {
    if constexpr (std::is_same_v<OutT, float>)
      launch_fp8_blockwise_f32(act, act_row_stride_elems, w_payload, w_scales, out, m, n, k,
                               stream, out_stride);
    else
      launch_fp8_blockwise_bf16(act, act_row_stride_elems, w_payload, w_scales, out, m, n, k,
                                stream, out_stride);
    return;
  }
  // Large m: the 128-row tensor-core kernel (the MoE experts'
  // dense form) — bitwise this file's tile kernel (the same dequantized
  // weights and the same ascending-k16 mma chain), with the weight tile
  // decoded once per 128 rows instead of once per 16 (the dense MLP
  // layers' 2048-row GEMMs: 9.7 ms a call on the tile kernel). k % 16 != 0
  // stays on the tile kernel.
  if (k % 16 == 0) {
    launch_dense_mma(act, act_row_stride_elems, w_payload, w_scales, out, m, n,
                     k, out_stride, stream);
    return;
  }
  const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
  scale_gemm_kernel<OutT><<<grid, kBlockThreads, 0, stream>>>(
      act, act_row_stride_elems, w_payload, w_scales, out, m, n, k, out_stride, 7, 7);
  DGPP_CUDA_OK(cudaGetLastError());
}

// The routed launcher on a stated scale grid (2026-09-13, the DeepSeek-V4.1
// release's 32 x 32 fp8 grid; rs / cs the log2 block sizes, 5..7): small
// m through the GEMV rows (the core reads the grid), larger m through the
// tile kernel (one scale column per 32-deep stage at cs >= 5). The 128-row
// dense form knows the 128 grid only and is not taken here.
template <typename OutT>
void launch_scale_gemm_grid(const uint16_t* act, size_t act_row_stride_elems,
                            const uint8_t* w_payload, const float* w_scales,
                            OutT* out, int m, int n, int k, cudaStream_t stream,
                            size_t out_stride, int rs, int cs) {
  if (m <= 0 || n <= 0) return;
  if (!act || !w_payload || !w_scales || !out)
    throw std::invalid_argument("scale_gemm_grid: null pointer");
  if (rs < 5 || rs > 7 || cs < 5 || cs > 7)
    throw std::invalid_argument("scale_gemm_grid: rs / cs must be 5..7 (32 .. 128 blocks)");
  if (out_stride == 0) out_stride = static_cast<size_t>(n);
  if (out_stride < static_cast<size_t>(n))
    throw std::invalid_argument("scale_gemm_grid: output row stride narrower than n");
  if (k <= 0 || k % 16 != 0)
    throw std::invalid_argument("scale_gemm_grid: k must be a positive multiple of 16");
  if (m <= kGemvMaxM && fp8_gemv::shape_ok(w_payload, /*rows=*/1, k)) {
    for (int row0 = 0; row0 < m;) {
      int rows = std::min(fp8_gemv::kMaxRows, m - row0);
      while (!fp8_gemv::shape_ok(w_payload, rows, k)) --rows;
      launch_scale_gemv_rows(act + static_cast<size_t>(row0) * act_row_stride_elems,
                             act_row_stride_elems, w_payload, w_scales,
                             out + static_cast<size_t>(row0) * out_stride, rows, n, k, out_stride,
                             stream, rs, cs);
      row0 += rows;
    }
    return;
  }
  const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
  scale_gemm_kernel<OutT><<<grid, kBlockThreads, 0, stream>>>(
      act, act_row_stride_elems, w_payload, w_scales, out, m, n, k, out_stride, rs, cs);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace

void launch_scale_gemm_grid_bf16(const uint16_t* act, size_t act_row_stride_elems,
                                 const uint8_t* w_payload, const float* w_scales,
                                 uint16_t* out, int m, int n, int k, cudaStream_t stream,
                                 size_t out_row_stride_elems, int rs, int cs, bool decode_mma) {
  if (decode_mma && m >= 1 && n > 0 && k > 0 && cs >= 4 &&
      mma_gemv_shape_ok(w_payload, act, act_row_stride_elems, m, k)) {
    launch_mma_gemv_fp8_bf16(act, act_row_stride_elems, w_payload, w_scales, out, m, n, k,
                             out_row_stride_elems, rs, cs, stream);
    return;
  }
  launch_scale_gemm_grid<uint16_t>(act, act_row_stride_elems, w_payload, w_scales, out, m, n, k,
                                   stream, out_row_stride_elems, rs, cs);
}

void launch_scale_gemm_grid_f32(const uint16_t* act, size_t act_row_stride_elems,
                                const uint8_t* w_payload, const float* w_scales, float* out,
                                int m, int n, int k, cudaStream_t stream,
                                size_t out_row_stride_elems, int rs, int cs, bool decode_mma) {
  if (decode_mma && m >= 1 && n > 0 && k > 0 && cs >= 4 &&
      mma_gemv_shape_ok(w_payload, act, act_row_stride_elems, m, k)) {
    launch_mma_gemv_fp8_f32(act, act_row_stride_elems, w_payload, w_scales, out, m, n, k,
                            out_row_stride_elems, rs, cs, stream);
    return;
  }
  launch_scale_gemm_grid<float>(act, act_row_stride_elems, w_payload, w_scales, out, m, n, k,
                                stream, out_row_stride_elems, rs, cs);
}

namespace {
// The multi-problem GEMV: the problems in the parameter space
// with their block prefixes; a block finds its problem by the prefix table
// (field-wise selects, as bf16_gemv_multi_kernel) and runs the dense
// launcher's block body over its rows.
struct Fp8GemvMulti {
  Fp8GemvProblem p[kFp8GemvMaxProblems];
  int block_end[kFp8GemvMaxProblems];  // exclusive prefix of blocks per problem
  int n;
  int span;  // rows per block (a multiple of kWarps), the same for every problem
};

template <int kRows>
__global__ void scale_gemv_multi_kernel(Fp8GemvMulti mp, const uint16_t* __restrict__ act,
                                        size_t act_stride, int k) {
  extern __shared__ __align__(16) uint16_t sx[];
  const int bid = static_cast<int>(blockIdx.x);
  int which = 0;
#pragma unroll
  for (int i = 0; i < kFp8GemvMaxProblems - 1; ++i) which += (i + 1 < mp.n && bid >= mp.block_end[i]) ? 1 : 0;
  const uint8_t* w = which == 0 ? mp.p[0].payload : which == 1 ? mp.p[1].payload : which == 2 ? mp.p[2].payload : mp.p[3].payload;
  const float* scales = which == 0 ? mp.p[0].scales : which == 1 ? mp.p[1].scales : which == 2 ? mp.p[2].scales : mp.p[3].scales;
  uint16_t* out = which == 0 ? mp.p[0].out : which == 1 ? mp.p[1].out : which == 2 ? mp.p[2].out : mp.p[3].out;
  const int n = which == 0 ? mp.p[0].n : which == 1 ? mp.p[1].n : which == 2 ? mp.p[2].n : mp.p[3].n;
  const size_t out_stride = which == 0 ? mp.p[0].out_stride : which == 1 ? mp.p[1].out_stride : which == 2 ? mp.p[2].out_stride : mp.p[3].out_stride;
  const int block0 = which == 0 ? 0 : which == 1 ? mp.block_end[0] : which == 2 ? mp.block_end[1] : mp.block_end[2];
  const int rs = which == 0 ? mp.p[0].rs : which == 1 ? mp.p[1].rs : which == 2 ? mp.p[2].rs : mp.p[3].rs;
  const int cs = which == 0 ? mp.p[0].cs : which == 1 ? mp.p[1].cs : which == 2 ? mp.p[2].cs : mp.p[3].cs;
  const uint16_t* w16 = which == 0 ? mp.p[0].bf16_weight : which == 1 ? mp.p[1].bf16_weight
                        : which == 2 ? mp.p[2].bf16_weight : mp.p[3].bf16_weight;
  fp8_gemv::stage_activations<kRows>(act, act_stride, k, sx);
  __syncthreads();
  const int b0 = (bid - block0) * mp.span;
  const int b1 = min(n, b0 + mp.span);
  if (w16 != nullptr) {
    // A bf16 problem: bf16_gemv_kernel's row chain and store, row by row.
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int n0 = b0; n0 < b1; n0 += fp8_gemv::kWarps) {
      const int row = n0 + warp;
      if (row >= n) continue;
      float acc[kRows];
      bf16_gemv::row_dots<kRows>(w16 + static_cast<size_t>(row) * k, sx, k, lane, acc);
      if (lane != 0) continue;
#pragma unroll
      for (int r = 0; r < kRows; ++r) out[static_cast<size_t>(r) * out_stride + row] = float_to_bf16_bits(acc[r]);
    }
    return;
  }
  for (int n0 = b0; n0 < b1; n0 += fp8_gemv::kWarps)
    fp8_gemv::block_rows<kRows>(w, scales, sx, n0, n, k, out, out_stride, rs, cs);
}

template <int kRows>
void launch_multi_rows(const Fp8GemvMulti& mp, const uint16_t* act, size_t act_stride, int k,
                       cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>(mp.block_end[mp.n - 1]));
  scale_gemv_multi_kernel<kRows>
      <<<grid, fp8_gemv::kThreads, fp8_gemv::smem_bytes(kRows, k), stream>>>(mp, act, act_stride, k);
  DGPP_CUDA_OK(cudaGetLastError());
}
}  // namespace

void launch_scale_gemv_multi_bf16(const Fp8GemvProblem* problems, int n_problems,
                                  const uint16_t* act, size_t act_row_stride_elems, int rows,
                                  int k, cudaStream_t stream) {
  if (rows <= 0) return;
  if (n_problems <= 0 || n_problems > kFp8GemvMaxProblems || problems == nullptr || act == nullptr)
    throw std::invalid_argument("scale_gemv_multi: 1..4 problems and an activation");
  if (rows > 2 * fp8_gemv::kMaxRows)
    throw std::invalid_argument("scale_gemv_multi: rows beyond the decode rows");
  if (k <= 0 || k % 16 != 0 || act_row_stride_elems < static_cast<size_t>(k) ||
      !gemv::smem_fits(fp8_gemv::kMaxRows, k))
    throw std::invalid_argument("scale_gemv_multi: k a multiple of 16 that fits the staging");
  Fp8GemvMulti base{};
  base.n = n_problems;
  int groups = 0;
  for (int i = 0; i < n_problems; ++i) groups += (problems[i].n + fp8_gemv::kWarps - 1) / fp8_gemv::kWarps;
  base.span = gemv_span_rows(groups);
  int blocks = 0;
  for (int i = 0; i < n_problems; ++i) {
    Fp8GemvProblem p = problems[i];
    const bool bf16 = p.bf16_weight != nullptr;
    if (bf16 ? (p.payload != nullptr || !gemv::aligned16(p.bf16_weight))
             : (!p.payload || !p.scales || !gemv::aligned16(p.payload)))
      throw std::invalid_argument("scale_gemv_multi: a problem is an fp8 payload with scales or a bf16 matrix, 16-byte aligned");
    if (!p.out || p.n <= 0)
      throw std::invalid_argument("scale_gemv_multi: empty or null problem");
    if (p.out_stride == 0) p.out_stride = static_cast<size_t>(p.n);
    if (p.out_stride < static_cast<size_t>(p.n))
      throw std::invalid_argument("scale_gemv_multi: output row stride narrower than n");
    if (!bf16 && (p.rs < 5 || p.rs > 7 || p.cs < 4 || p.cs > 7))
      throw std::invalid_argument("scale_gemv_multi: a problem's scale grid must be 32..128 rows, 16..128 cols");
    base.p[i] = p;
    blocks += (p.n + base.span - 1) / base.span;
    base.block_end[i] = blocks;
  }
  for (int i = n_problems; i < kFp8GemvMaxProblems; ++i) base.block_end[i] = blocks;
  for (int row0 = 0; row0 < rows; row0 += fp8_gemv::kMaxRows) {
    const int n = std::min(fp8_gemv::kMaxRows, rows - row0);
    Fp8GemvMulti mp = base;
    for (int i = 0; i < n_problems; ++i) mp.p[i].out += static_cast<size_t>(row0) * mp.p[i].out_stride;
    const uint16_t* a = act + static_cast<size_t>(row0) * act_row_stride_elems;
    switch (n) {
      case 1: launch_multi_rows<1>(mp, a, act_row_stride_elems, k, stream); break;
      case 2: launch_multi_rows<2>(mp, a, act_row_stride_elems, k, stream); break;
      case 3: launch_multi_rows<3>(mp, a, act_row_stride_elems, k, stream); break;
      default: launch_multi_rows<4>(mp, a, act_row_stride_elems, k, stream); break;
    }
  }
}

void launch_scale_gemm_bf16(const uint16_t* act, size_t act_row_stride_elems,
                            const uint8_t* w_payload, const float* w_scales,
                            uint16_t* out, int m, int n, int k,
                            cudaStream_t stream, size_t out_row_stride_elems, int mma_from_rows,
                            void* ws, size_t ws_bytes) {
  launch_scale_gemm<uint16_t>(act, act_row_stride_elems, w_payload, w_scales,
                              out, m, n, k, stream, out_row_stride_elems, mma_from_rows,
                              /*last_row_only=*/false, ws, ws_bytes);
}

void launch_scale_gemm_f32(const uint16_t* act, size_t act_row_stride_elems,
                           const uint8_t* w_payload, const float* w_scales,
                           float* out, int m, int n, int k,
                           cudaStream_t stream, size_t out_row_stride_elems, int mma_from_rows,
                           bool last_row_only, void* ws, size_t ws_bytes, int compact_row) {
  launch_scale_gemm<float>(act, act_row_stride_elems, w_payload, w_scales,
                           out, m, n, k, stream, out_row_stride_elems, mma_from_rows, last_row_only, ws,
                           ws_bytes, compact_row);
}

namespace {
template <typename OutT>
void launch_scale_gemm_tile(const uint16_t* act, size_t act_row_stride_elems,
                            const uint8_t* w_payload, const float* w_scales,
                            OutT* out, int m, int n, int k, cudaStream_t stream,
                            int rs, int cs) {
  if (m <= 0 || n <= 0) return;
  if (!act || !w_payload || !w_scales || !out || k <= 0)
    throw std::invalid_argument("scale_gemm_tile: null pointer or k <= 0");
  if (rs < 5 || rs > 7 || cs < 5 || cs > 7)
    throw std::invalid_argument("scale_gemm_tile: the scale grid must be 32, 64 or 128 on each axis");
  const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
  scale_gemm_kernel<OutT><<<grid, kBlockThreads, 0, stream>>>(
      act, act_row_stride_elems, w_payload, w_scales, out, m, n, k, static_cast<size_t>(n), rs, cs);
  DGPP_CUDA_OK(cudaGetLastError());
}
}  // namespace

void launch_scale_gemm_tile_bf16(const uint16_t* act, size_t act_row_stride_elems,
                                 const uint8_t* w_payload, const float* w_scales,
                                 uint16_t* out, int m, int n, int k,
                                 cudaStream_t stream, int rs, int cs) {
  launch_scale_gemm_tile<uint16_t>(act, act_row_stride_elems, w_payload, w_scales,
                                   out, m, n, k, stream, rs, cs);
}

void launch_scale_gemm_tile_f32(const uint16_t* act, size_t act_row_stride_elems,
                                const uint8_t* w_payload, const float* w_scales,
                                float* out, int m, int n, int k,
                                cudaStream_t stream, int rs, int cs) {
  launch_scale_gemm_tile<float>(act, act_row_stride_elems, w_payload, w_scales,
                                out, m, n, k, stream, rs, cs);
}

}  // namespace dgpp
