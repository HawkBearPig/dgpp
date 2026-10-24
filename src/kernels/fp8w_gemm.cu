// The exact fp8-weight GEMM (fp8w_gemm.hpp).
#include "kernels/fp8w_gemm.hpp"
#include "kernels/pdl.cuh"

#include <cuda_bf16.h>

#include <stdexcept>
#include <type_traits>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {

// 128 x 128 x 64 block tiles on sixteen warps (4 x 4: a 32 x 32 warp tile
// of two m16 by four n8 mma tiles), four warps a scheduler so one warp's
// decode latency hides under the others' mma (eight warps ran at 61 TF,
// 2026-10-03). The activation tile rides a three-stage cp.async pipeline
// (bf16 rows at a 144-byte stride, ldmatrix conflict-free). The weight
// tile's e4m3 rows are loaded into registers two steps ahead (16 bytes a
// thread) and placed into one of two bf16 tiles by bit arithmetic — an
// e4m3 code s eeee mmm shifted into s 0000eeee mmm0000 is the bf16 of the
// code's value x 2^-120, subnormals included (both formats' subnormals are
// man / 8 x 2^(1 - bias)) — the decode of step s+1 interleaved into step
// s's mma loop; ldmatrix feeds mma m16n8k16 bf16 with fp32 accumulation
// in ascending k order. The 128 x 128 block scale is applied once per
// 128-wide k group: the group's products accumulate in fp32 unscaled and
// acc += (2^120 scale) x partial (the fp8 GEMM's promotion), so every
// product term is code x activation exactly (bf16 x bf16 in fp32) and the
// scale rounds once per group — tolerance-equal to the dequant bridge's
// bf16(code x scale) terms, finer by the bridge's per-weight rounding. One
// barrier per step.
constexpr int kBM = 128, kBN = 128, kBK = 64, kGroup = 128;
constexpr int kThreads = 512;
constexpr int kStages = 3;                        // activation stages
constexpr int kStride = kBK * 2 + 16;             // smem row bytes (bf16 tile)
constexpr int kTileBytes = kBM * kStride;         // 18,432: one bf16 tile
constexpr int kSmem = (kStages + 2) * kTileBytes; // 92,160: A stages + two W tiles
constexpr float kTwoPow120 = 1.329227995784916e36f;
static_assert(kBN == kBM, "the copy map covers both operands the same way");
static_assert(kGroup == 2 * kBK, "a scale group is two k-steps");

__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::); }
template <int N>
__device__ __forceinline__ void wait_pending() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(address));
}

// Four e4m3 codes (one word, k-ascending bytes) -> two bf16x2 words of the
// codes' values x 2^-120: bytes 0, 1 -> the first pair, 2, 3 -> the second.
__device__ __forceinline__ void place4(uint32_t w, uint32_t& lo, uint32_t& hi) {
  const uint32_t a = __byte_perm(w, 0, 0x4140);  // [0, b1, 0, b0]
  const uint32_t b = __byte_perm(w, 0, 0x4342);  // [0, b3, 0, b2]
  lo = ((a << 4) & 0x07F007F0u) | ((a << 8) & 0x80008000u);
  hi = ((b << 4) & 0x07F007F0u) | ((b << 8) & 0x80008000u);
}
// The per-weight form: the placed pair x (2^120 scale) in fp32, rounded
// once to bf16 — the dequant bridge's bf16(decode(code) x scale) bit for
// bit (cvt.rn.bf16x2.f32 packs the pair).
__device__ __forceinline__ uint32_t scale_pair(uint32_t pair, float sp) {
  const float a = __fmul_rn(__uint_as_float(pair << 16), sp);
  const float b = __fmul_rn(__uint_as_float(pair & 0xFFFF0000u), sp);
  const __nv_bfloat162 r = __floats2bfloat162_rn(a, b);  // .x = a (low half)
  return *reinterpret_cast<const uint32_t*>(&r);
}

// rs / cs: the scale grid's block size per axis as log2 (7, 7: the 128 x 128
// grid; 5, 5: DeepSeek-V4.1's 32 x 32 e8m0 grid held as f32). The per-weight
// form reads its weight's own block: n-row (n >> rs), k-column (k >> cs) —
// a thread's 16 consecutive k never straddle a block at cs >= 4. The
// per-group form takes the 128 x 128 grid only.
template <typename OutT, bool kPerWeight, typename ScaleT>
__global__ __launch_bounds__(kThreads, 1) void fp8w_gemm_kernel(const uint16_t* __restrict__ act,
                                                               size_t act_stride,
                                                               const uint8_t* __restrict__ w,
                                                               const ScaleT* __restrict__ w_scales,
                                                               OutT* __restrict__ out, int m, int n, int k,
                                                               size_t out_stride, int rs, int cs) {
  dgpp_pdl_wait();
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sa = smem;
  uint8_t* sw = smem + kStages * kTileBytes;  // two W tiles
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  // The tile order: groups of kGroupM m-tiles walked n-tile by n-tile (a
  // wave's blocks share kGroupM activation tiles from L2 while each weight
  // tile streams once per group).
  constexpr int kGroupM = 8;
  const int m_tiles = (m + kBM - 1) / kBM, n_tiles = (n + kBN - 1) / kBN;
  const int bid = static_cast<int>(blockIdx.x);
  const int group = bid / (kGroupM * n_tiles);
  const int first_m = group * kGroupM;
  const int group_rows = min(kGroupM, m_tiles - first_m);
  const int in_group = bid - group * kGroupM * n_tiles;
  const int m0 = (first_m + in_group % group_rows) * kBM;
  const int n0 = (in_group / group_rows) * kBN;
  const int row_base = (warp / 4) * 32, col_base = (warp % 4) * 32;
  const int steps = k / kBK;
  const int scale_cols = (k + kGroup - 1) / kGroup;
  const ScaleT* scale_row = w_scales + static_cast<size_t>(n0 / 128) * scale_cols;  // the per-group form's row
  // The per-weight form's own block row (n >> rs) and column stride (ceil(k / 2^cs)).
  const int wrow_scale_cols = (k + (1 << cs) - 1) >> cs;
  // The activation copy map: 128 rows x 8 16-byte chunks = 1024 chunks,
  // two a thread: chunk c = tid + i * 512 -> row c / 8, piece c % 8.
  const uint16_t* a_src[2];
  bool a_ok[2];
  int a_row[2], a_piece[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int c = tid + i * kThreads;
    a_row[i] = c / 8;
    a_piece[i] = (c % 8) * 8;  // elements
    a_ok[i] = m0 + a_row[i] < m;
    a_src[i] = act + static_cast<size_t>(a_ok[i] ? m0 + a_row[i] : 0) * act_stride + a_piece[i];
  }
  auto issue = [&](int step, int slot) {
    uint8_t* as = sa + slot * kTileBytes;
    const int koff = step * kBK;
#pragma unroll
    for (int i = 0; i < 2; ++i) cp16(as + a_row[i] * kStride + a_piece[i] * 2, a_src[i] + koff, a_ok[i]);
  };
  // The weight row this thread places: n-row tid / 4, 16 consecutive k at
  // (tid % 4) * 16 — one 16-byte load a step, two register sets.
  const int wrow = tid / 4, wk = (tid % 4) * 16;
  const bool w_ok = n0 + wrow < n;
  const uint8_t* wsrc = w + static_cast<size_t>(w_ok ? n0 + wrow : 0) * k + wk;
  uint4 raw0, raw1;
  auto fetch = [&](int step, uint4& r) {
    r = w_ok ? *reinterpret_cast<const uint4*>(wsrc + step * kBK) : make_uint4(0, 0, 0, 0);
  };
  // Half h of a step's placement: codes 0-7 (words x, y) or 8-15 (z, w).
  auto place_half = [&](int step, const uint4& r, int h) {
    uint32_t p[4];
    place4(h ? r.z : r.x, p[0], p[1]);
    place4(h ? r.w : r.y, p[2], p[3]);
    if constexpr (kPerWeight) {
      // The row's scale, guarded like its codes: a row past n in the last
      // n-tile has no scale row (the grid ends at ceil(n / 2^rs)); the
      // unguarded read ran up to 2^(7 - rs) - 1 rows past it (2026-10-19).
      const size_t si = static_cast<size_t>((n0 + wrow) >> rs) * wrow_scale_cols +
                        ((step * kBK + wk) >> cs);
      if constexpr (std::is_same_v<ScaleT, uint8_t>) {
        // E8M0 is an exact exponent adjustment of the placed pair. The
        // common range keeps the same fast placement as F32 scales. Above
        // byte 134, scale * 2^120 would overflow: decode before scaling.
        const uint32_t bscale = w_ok ? w_scales[si] : 0;
        if (bscale <= 134) {
          const float sp = __uint_as_float((bscale + 120) << 23);
#pragma unroll
          for (int i = 0; i < 4; ++i) p[i] = scale_pair(p[i], sp);
        } else {
          const float scale = __uint_as_float(bscale << 23);
          const uint32_t words[2] = {h ? r.z : r.x, h ? r.w : r.y};
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const uint32_t pair = words[i / 2] >> ((i % 2) * 16);
            const float a = fp8_e4m3_bits_to_float(static_cast<uint8_t>(pair)) * scale;
            const float b = fp8_e4m3_bits_to_float(static_cast<uint8_t>(pair >> 8)) * scale;
            const __nv_bfloat162 packed = __floats2bfloat162_rn(a, b);
            p[i] = *reinterpret_cast<const uint32_t*>(&packed);
          }
        }
      } else {
        const float sp = w_ok ? w_scales[si] * kTwoPow120 : 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) p[i] = scale_pair(p[i], sp);
      }
    }
    uint8_t* dst = sw + (step & 1) * kTileBytes + wrow * kStride + (wk + h * 8) * 2;
    *reinterpret_cast<uint4*>(dst) = make_uint4(p[0], p[1], p[2], p[3]);
  };

  float acc[2][4][4] = {};
  // The per-group form accumulates the unscaled products here and promotes
  // them by the group's scale; the per-weight form accumulates into acc.
  float partial[2][4][4] = {};
#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < steps) issue(s, s);
    commit();
  }
  fetch(0, raw0);
  place_half(0, raw0, 0);
  place_half(0, raw0, 1);
  if (steps > 1) fetch(1, raw1);
  // One step: `cur` held W(step) (placed already: free for W(step + 2)),
  // `nxt_raw` holds W(step + 1), placed into the other tile under the mma.
  auto body = [&](int step, uint4& cur, uint4& nxt_raw) {
    wait_pending<kStages - 2>();
    __syncthreads();  // A(step) and W(step) are complete; every warp is past step-1's reads
    const int nxt = step + kStages - 1;
    if (nxt < steps) issue(nxt, nxt % kStages);
    commit();
    if (step + 2 < steps) fetch(step + 2, cur);
    const bool place_next = step + 1 < steps;
    const uint8_t* as = sa + (step % kStages) * kTileBytes;
    const uint8_t* ws = sw + (step & 1) * kTileBytes;
#pragma unroll
    for (int kk = 0; kk < kBK; kk += 16) {
      uint32_t af[2][4];
#pragma unroll
      for (int i = 0; i < 2; ++i)
        ldsm_x4(af[i], as + (row_base + i * 16 + lane % 16) * kStride + (kk + (lane / 16) * 8) * 2);
      uint32_t bf[4][2];
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        // Matrices 0/1 = n8 tile 2jj at k kk..kk+7 / kk+8..kk+15, 2/3 = tile 2jj+1.
        const int tile = col_base + jj * 16 + (lane / 16) * 8 + lane % 8;
        const int kb = kk + ((lane / 8) % 2) * 8;
        uint32_t q4[4];
        ldsm_x4(q4, ws + tile * kStride + kb * 2);
        bf[jj * 2][0] = q4[0];
        bf[jj * 2][1] = q4[1];
        bf[jj * 2 + 1][0] = q4[2];
        bf[jj * 2 + 1][1] = q4[3];
      }
      float (&dst)[2][4][4] = kPerWeight ? acc : partial;
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(dst[i][j][0]), "+f"(dst[i][j][1]), "+f"(dst[i][j][2]), "+f"(dst[i][j][3])
              : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(bf[j][0]), "r"(bf[j][1]));
      // The next step's W tile, a half per two k16 steps, into the other tile.
      if (place_next && (kk % 32) == 16) place_half(step + 1, nxt_raw, kk / 32);
    }
    if (!kPerWeight && (step % 2 == 1 || step + 1 == steps)) {
      // A 128-wide group is complete (or the matrix ends mid-group):
      // promote by the group's scale.
      const float sp = scale_row[step / 2] * kTwoPow120;
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int v = 0; v < 4; ++v) {
            acc[i][j][v] = __fmaf_rn(sp, partial[i][j][v], acc[i][j][v]);
            partial[i][j][v] = 0.f;
          }
    }
  };
  for (int step = 0; step < steps; step += 2) {
    body(step, raw0, raw1);
    if (step + 1 < steps) body(step + 1, raw1, raw0);
  }
  const int r = lane / 4, cc = (lane % 4) * 2;
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const int row = m0 + row_base + i * 16 + r + (v / 2) * 8;
        const int col = n0 + col_base + j * 8 + cc + (v % 2);
        if (row < m && col < n) {
          const size_t index = static_cast<size_t>(row) * out_stride + col;
          if constexpr (std::is_same_v<OutT, float>)
            out[index] = acc[i][j][v];
          else
            out[index] = __bfloat16_as_ushort(__float2bfloat16_rn(acc[i][j][v]));
        }
      }
}

template <typename OutT, bool kPerWeight, typename ScaleT>
void launch(const uint16_t* act, size_t act_stride, const uint8_t* w, const ScaleT* w_scales, OutT* out, int m,
            int n, int k, cudaStream_t stream, size_t out_stride, int rs, int cs) {
  if (m <= 0 || n <= 0) return;
  if (!act || !w || !w_scales || !out) throw std::invalid_argument("fp8w gemm: null pointer");
  if (rs < 0 || rs > 7 || cs < 4 || cs > 7) throw std::invalid_argument("fp8w gemm: scale grid log2 blocks rs in [0, 7], cs in [4, 7]");
  if (!kPerWeight && (rs != 7 || cs != 7)) throw std::invalid_argument("fp8w gemm: the per-group form takes the 128 x 128 grid");
  if (!fp8w_gemm_shape_ok(act, act_stride, k))
    throw std::invalid_argument("fp8w gemm: k a positive multiple of 64, 16-byte aligned activation rows");
  if (out_stride == 0) out_stride = static_cast<size_t>(n);
  if (out_stride < static_cast<size_t>(n)) throw std::invalid_argument("fp8w gemm: output row stride narrower than n");
  static bool attr_set = false;  // once per instantiation
  if (!attr_set) {
    DGPP_CUDA_OK(cudaFuncSetAttribute(fp8w_gemm_kernel<OutT, kPerWeight, ScaleT>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    attr_set = true;
  }
  const dim3 grid(static_cast<unsigned>(((n + kBN - 1) / kBN) * ((m + kBM - 1) / kBM)));
  fp8w_gemm_kernel<OutT, kPerWeight, ScaleT>
      <<<grid, kThreads, kSmem, stream>>>(act, act_stride, w, w_scales, out, m, n, k, out_stride, rs, cs);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace

bool fp8w_gemm_shape_ok(const void* act, size_t act_stride, int k) {
  return k > 0 && k % kBK == 0 && act_stride % 8 == 0 && act_stride >= static_cast<size_t>(k) &&
         (reinterpret_cast<uintptr_t>(act) & 15u) == 0;
}

void launch_fp8w_gemm_bf16(const uint16_t* act, size_t act_stride, const uint8_t* w, const float* w_scales,
                           uint16_t* out, int m, int n, int k, Fp8wScale scale, cudaStream_t stream,
                           size_t out_stride, int rs, int cs) {
  if (scale == Fp8wScale::PerWeight)
    launch<uint16_t, true>(act, act_stride, w, w_scales, out, m, n, k, stream, out_stride, rs, cs);
  else
    launch<uint16_t, false>(act, act_stride, w, w_scales, out, m, n, k, stream, out_stride, rs, cs);
}

void launch_fp8w_gemm_f32(const uint16_t* act, size_t act_stride, const uint8_t* w, const float* w_scales,
                          float* out, int m, int n, int k, Fp8wScale scale, cudaStream_t stream,
                          size_t out_stride, int rs, int cs) {
  if (scale == Fp8wScale::PerWeight)
    launch<float, true>(act, act_stride, w, w_scales, out, m, n, k, stream, out_stride, rs, cs);
  else
    launch<float, false>(act, act_stride, w, w_scales, out, m, n, k, stream, out_stride, rs, cs);
}

void launch_fp8w_gemm_e8m0_bf16(const uint16_t* act, size_t act_stride, const uint8_t* w,
                                const uint8_t* scales, uint16_t* out, int m, int n, int k,
                                cudaStream_t stream, size_t out_stride, int rs, int cs) {
  launch<uint16_t, true>(act, act_stride, w, scales, out, m, n, k, stream, out_stride, rs, cs);
}

void launch_fp8w_gemm_e8m0_f32(const uint16_t* act, size_t act_stride, const uint8_t* w,
                               const uint8_t* scales, float* out, int m, int n, int k,
                               cudaStream_t stream, size_t out_stride, int rs, int cs) {
  launch<float, true>(act, act_stride, w, scales, out, m, n, k, stream, out_stride, rs, cs);
}

}  // namespace dgpp
