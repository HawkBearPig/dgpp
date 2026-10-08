// The NVFP4-weight GEMM — see fp4w_gemm.hpp.
#include "kernels/fp4w_gemm.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>

#include <stdexcept>
#include <type_traits>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {

// 128 x 128 x 64 block tiles on sixteen warps (4 x 4: a 32 x 32 warp tile
// of two m16 by four n8 mma tiles), four warps a scheduler so one warp's
// decode latency hides under the others' mma. The activation tile rides a
// three-stage cp.async pipeline (bf16 rows at a 144-byte stride, ldmatrix
// conflict-free). The weight tile's e2m1 pairs are loaded into registers
// two steps ahead (8 payload bytes plus the step's four e4m3 scale bytes
// a thread) and placed into the bf16 tile pair by pair — the decode of
// step s+1 interleaved into step s's mma loop; ldmatrix feeds mma
// m16n8k16 bf16 with fp32 accumulation in ascending k order. The placed
// words carry their e4m3 scale already (converted once per row-step),
// and the F32 global divides the finished dot once in the epilogue, as
// the fp4 GEMV core does. One barrier per step.
constexpr int kBM = 128, kBN = 128, kBK = 64;
constexpr int kThreads = 512;
constexpr int kStages = 3;                        // activation stages
constexpr int kStride = kBK * 2 + 16;             // smem row bytes (bf16 tile)
constexpr int kTileBytes = kBM * kStride;         // 18,432: one bf16 tile
constexpr int kSmem = (kStages + 2) * kTileBytes;  // 92,160: A stages + two W tiles

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

// Two e2m1 codes (one byte) times the group's scale (f16, exact) as a
// bf16x2 word: the products carry <= 5 significant bits, so f16, f32 and
// bf16 all hold them exactly.
__device__ __forceinline__ uint32_t decode_pair_bf16_w(uint32_t byte, __half2 s2) {
  const __half2 h(__nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(byte), __NV_E2M1));
  const __half2 p = __hmul2(h, s2);
  const __nv_bfloat162 b = __floats2bfloat162_rn(__low2float(p), __high2float(p));
  return *reinterpret_cast<const uint32_t*>(&b);
}

template <typename OutT>
__global__ __launch_bounds__(kThreads, 1) void fp4w_gemm_kernel(
    const uint16_t* __restrict__ act, size_t act_stride, const uint8_t* __restrict__ payload,
    const uint8_t* __restrict__ scales, size_t scale_stride, const float* __restrict__ global,
    OutT* __restrict__ out, int m, int n, int k, size_t out_stride) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sa = smem;
  uint8_t* sw = smem + kStages * kTileBytes;  // two W tiles
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const float g = *global;  // one broadcast read: the whole grid's divisor
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
  const size_t payload_stride = static_cast<size_t>(k) / 2;
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
  // The weight row this thread places: n-row tid / 4, 16 consecutive codes
  // at (tid % 4) * 16 — one 8-byte payload load plus the step's four scale
  // bytes (one word) a step, two register sets.
  const int wrow = tid / 4, wk = (tid % 4) * 16;
  const bool w_ok = n0 + wrow < n;
  const uint8_t* wsrc =
      payload + static_cast<size_t>(w_ok ? n0 + wrow : 0) * payload_stride + static_cast<size_t>(wk) / 2;
  const uint8_t* ssrc = scales + static_cast<size_t>(w_ok ? n0 + wrow : 0) * scale_stride;
  struct Raw {
    uint2 codes;
    uint32_t sc;
  };
  Raw raw0, raw1;
  auto fetch = [&](int step, Raw& r) {
    r.codes = w_ok ? *reinterpret_cast<const uint2*>(wsrc + step * (kBK / 2)) : make_uint2(0u, 0u);
    r.sc = w_ok ? *reinterpret_cast<const uint32_t*>(ssrc + step * (kBK / 16)) : 0u;
  };
  // Half h of a step's placement: codes wk+h*8 .. wk+h*8+8. Eight
  // consecutive codes from a multiple of 8 always sit inside one 16-code
  // scale group, so the half takes a single scale byte: (wk + h * 8) / 16
  // is tid % 4 for both halves (wk = (tid % 4) * 16).
  auto place_half = [&](int step, const Raw& r, int h) {
    uint32_t p[4];
    const uint32_t sb = (r.sc >> (8 * (wk / 16))) & 0xFFu;
    const __half2 s2 = __half2half2(__half(
        __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(sb), __NV_E4M3)));
    // Bytes 0..3 in order.
    const uint32_t word = h ? r.codes.y : r.codes.x;
    p[0] = decode_pair_bf16_w(word & 0xFFu, s2);
    p[1] = decode_pair_bf16_w((word >> 8) & 0xFFu, s2);
    p[2] = decode_pair_bf16_w((word >> 16) & 0xFFu, s2);
    p[3] = decode_pair_bf16_w((word >> 24) & 0xFFu, s2);
    uint8_t* dst = sw + (step & 1) * kTileBytes + wrow * kStride + (wk + h * 8) * 2;
    *reinterpret_cast<uint4*>(dst) = make_uint4(p[0], p[1], p[2], p[3]);
  };

  float acc[2][4][4] = {};
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
  auto body = [&](int step, Raw& cur, Raw& nxt_raw) {
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
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
              : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(bf[j][0]), "r"(bf[j][1]));
      // The next step's W tile, a half per two k16 steps, into the other tile.
      if (place_next && (kk % 32) == 16) place_half(step + 1, nxt_raw, kk / 32);
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
          const float val = __fdiv_rn(acc[i][j][v], g);
          if constexpr (std::is_same_v<OutT, float>)
            out[index] = val;
          else
            out[index] = __bfloat16_as_ushort(__float2bfloat16_rn(val));
        }
      }
}

template <typename OutT>
void launch(const uint16_t* act, size_t act_stride, const GlmFp4Matrix& w, OutT* out, int m,
            int n, int k, cudaStream_t stream, size_t out_stride) {
  if (m <= 0 || n <= 0) return;
  if (!act || !w.payload || !w.scales || !w.global_scale || !out)
    throw std::invalid_argument("fp4w gemm: null pointer");
  if (w.rows < n || w.cols != k) throw std::invalid_argument("fp4w gemm: view/shape mismatch");
  if (!fp4w_gemm_shape_ok(act, act_stride, k))
    throw std::invalid_argument("fp4w gemm: k a positive multiple of 64, 16-byte aligned activation rows");
  if (out_stride == 0) out_stride = static_cast<size_t>(n);
  if (out_stride < static_cast<size_t>(n)) throw std::invalid_argument("fp4w gemm: output row stride narrower than n");
  static bool attr_set = false;  // once per instantiation
  if (!attr_set) {
    DGPP_CUDA_OK(cudaFuncSetAttribute(fp4w_gemm_kernel<OutT>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    attr_set = true;
  }
  const dim3 grid(static_cast<unsigned>(((n + kBN - 1) / kBN) * ((m + kBM - 1) / kBM)));
  fp4w_gemm_kernel<OutT>
      <<<grid, kThreads, kSmem, stream>>>(act, act_stride, w.payload, w.scales,
                                          static_cast<size_t>(k) / 16, w.global_scale, out, m, n,
                                          k, out_stride);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace

bool fp4w_gemm_shape_ok(const void* act, size_t act_stride, int k) {
  return k > 0 && k % kBK == 0 && act_stride % 8 == 0 && act_stride >= static_cast<size_t>(k) &&
         (reinterpret_cast<uintptr_t>(act) & 15u) == 0;
}

void launch_fp4w_gemm_bf16(const uint16_t* act, size_t act_stride, const GlmFp4Matrix& w,
                           uint16_t* out, int m, int n, int k, cudaStream_t stream,
                           size_t out_stride) {
  launch<uint16_t>(act, act_stride, w, out, m, n, k, stream, out_stride);
}

}  // namespace dgpp
