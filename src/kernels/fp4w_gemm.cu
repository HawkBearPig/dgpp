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

// 256 x 128 x 64 tiles on eight warps (4 x 2, each covering 64 x 64).
// The wider M tile amortizes each weight decode over twice as many rows.
// Two cp.async activation stages and two weight tiles fit in 96 KiB:
// shared rows use an XOR swizzle instead of padding. Each 8-bf16 vector
// stays aligned, while ldmatrix's eight rows address distinct banks.
// The dot keeps the original ascending-k FP32 accumulation and divides
// by the global scale once in the epilogue.
constexpr int kBM = 256, kBN = 128, kBK = 64;
constexpr int kThreads = 256;
constexpr int kStages = 2;
constexpr int kStride = kBK * 2;
constexpr int kTileBytes = kBM * kStride;
constexpr int kWeightTileBytes = kBN * kStride;
constexpr int kSmem = kStages * kTileBytes + 2 * kWeightTileBytes;

__device__ __forceinline__ int swizzled_col(int row, int col) {
  return col ^ ((row & 7) * 8);
}

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
// bf16x2 word: the products carry <= 6 significant bits, so f16, f32 and
// bf16 all hold them exactly.
__device__ __forceinline__ uint32_t decode_pair_bf16_w(uint32_t byte, __half2 s2) {
  const __half2 h(__nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(byte), __NV_E2M1));
  const __half2 p = __hmul2(h, s2);
  const __nv_bfloat162 b = __floats2bfloat162_rn(__low2float(p), __high2float(p));
  return *reinterpret_cast<const uint32_t*>(&b);
}

template <typename OutT, int kGroupM>
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
  const int m_tiles = (m + kBM - 1) / kBM, n_tiles = (n + kBN - 1) / kBN;
  const int bid = static_cast<int>(blockIdx.x);
  const int group = bid / (kGroupM * n_tiles);
  const int first_m = group * kGroupM;
  const int group_rows = min(kGroupM, m_tiles - first_m);
  const int in_group = bid - group * kGroupM * n_tiles;
  const int m0 = (first_m + in_group % group_rows) * kBM;
  const int n0 = (in_group / group_rows) * kBN;
  const int row_base = (warp / 2) * 64, col_base = (warp % 2) * 64;
  const int steps = k / kBK;
  const size_t payload_stride = static_cast<size_t>(k) / 2;
  // The activation copy map: 256 rows x 8 16-byte chunks = 2048 chunks,
  // eight per thread: chunk c = tid + i * 256 -> row c / 8, piece c % 8.
  const uint16_t* a_src[8];
  bool a_ok[8];
  int a_row[8], a_piece[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
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
    for (int i = 0; i < 8; ++i)
      cp16(as + a_row[i] * kStride + swizzled_col(a_row[i], a_piece[i]) * 2,
           a_src[i] + koff, a_ok[i]);
  };
  // Two threads place one weight row: 32 codes each, fetched in one
  // 16-byte vector. Each half uses its own 16-code scale group.
  const int wrow = tid / 2, wk = (tid % 2) * 32;
  const bool w_ok = n0 + wrow < n;
  const uint8_t* wsrc =
      payload + static_cast<size_t>(w_ok ? n0 + wrow : 0) * payload_stride + static_cast<size_t>(wk) / 2;
  const uint8_t* ssrc = scales + static_cast<size_t>(w_ok ? n0 + wrow : 0) * scale_stride;
  struct Raw {
    uint4 codes;
    uint32_t sc;
  };
  Raw raw0, raw1;
  auto fetch = [&](int step, Raw& r) {
    r.codes = w_ok ? *reinterpret_cast<const uint4*>(wsrc + step * (kBK / 2))
                   : make_uint4(0u, 0u, 0u, 0u);
    r.sc = w_ok ? *reinterpret_cast<const uint32_t*>(ssrc + step * (kBK / 16)) : 0u;
  };
  // Half h places sixteen codes as two swizzled eight-BF16 vectors.
  auto place_half = [&](int step, const Raw& r, int h) {
    const uint32_t sb = (r.sc >> (8 * ((wk + h * 16) / 16))) & 0xFFu;
    const __half2 s2 = __half2half2(
        __half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(sb), __NV_E4M3)));
    uint32_t word[2] = {h ? r.codes.z : r.codes.x, h ? r.codes.w : r.codes.y};
#pragma unroll
    for (int part = 0; part < 2; part++) {
      uint32_t p[4];
      p[0] = decode_pair_bf16_w(word[part] & 0xFFu, s2);
      p[1] = decode_pair_bf16_w((word[part] >> 8) & 0xFFu, s2);
      p[2] = decode_pair_bf16_w((word[part] >> 16) & 0xFFu, s2);
      p[3] = decode_pair_bf16_w((word[part] >> 24) & 0xFFu, s2);
      uint8_t* dst = sw + (step & 1) * kWeightTileBytes + wrow * kStride +
                     swizzled_col(wrow, wk + h * 16 + part * 8) * 2;
      *reinterpret_cast<uint4*>(dst) = make_uint4(p[0], p[1], p[2], p[3]);
    }
  };

  float acc[4][8][4] = {};
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
    const uint8_t* ws = sw + (step & 1) * kWeightTileBytes;
#pragma unroll
    for (int kk = 0; kk < kBK; kk += 16) {
      uint32_t af[4][4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        ldsm_x4(af[i], as + (row_base + i * 16 + lane % 16) * kStride +
                           swizzled_col(row_base + i * 16 + lane % 16,
                                        kk + (lane / 16) * 8) * 2);
      uint32_t bf[8][2];
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        // Matrices 0/1 = n8 tile 2jj at k kk..kk+7 / kk+8..kk+15, 2/3 = tile 2jj+1.
        const int tile = col_base + jj * 16 + (lane / 16) * 8 + lane % 8;
        const int kb = kk + ((lane / 8) % 2) * 8;
        uint32_t q4[4];
        ldsm_x4(q4, ws + tile * kStride + swizzled_col(tile, kb) * 2);
        bf[jj * 2][0] = q4[0];
        bf[jj * 2][1] = q4[1];
        bf[jj * 2 + 1][0] = q4[2];
        bf[jj * 2 + 1][1] = q4[3];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
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
  // Adjacent columns share one BF16 conversion and aligned 32-bit store.
  // Odd strides, unaligned output views and a final odd column stay scalar.
  const bool packed = (reinterpret_cast<uintptr_t>(out) & 3u) == 0 && (out_stride & 1u) == 0;
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        const int row = m0 + row_base + i * 16 + r + pair * 8;
        const int col = n0 + col_base + j * 8 + cc;
        if (row < m && col < n) {
          const size_t index = static_cast<size_t>(row) * out_stride + col;
          const float x = __fdiv_rn(acc[i][j][pair * 2], g);
          const float y = __fdiv_rn(acc[i][j][pair * 2 + 1], g);
          if constexpr (std::is_same_v<OutT, float>) {
            out[index] = x;
            if (col + 1 < n) out[index + 1] = y;
          } else {
            if (packed && col + 1 < n) {
              const __nv_bfloat162 p = __floats2bfloat162_rn(x, y);
              *reinterpret_cast<uint32_t*>(out + index) = *reinterpret_cast<const uint32_t*>(&p);
            } else {
              out[index] = __bfloat16_as_ushort(__float2bfloat16_rn(x));
              if (col + 1 < n) out[index + 1] = __bfloat16_as_ushort(__float2bfloat16_rn(y));
            }
          }
        }
      }
}

template <typename OutT, int kGroupM>
void launch_tiles(const uint16_t* act, size_t act_stride, const GlmFp4Matrix& w, OutT* out, int m,
                  int n, int k, cudaStream_t stream, size_t out_stride) {
  static bool attr_set = false;  // once per instantiation
  if (!attr_set) {
    DGPP_CUDA_OK(cudaFuncSetAttribute(fp4w_gemm_kernel<OutT, kGroupM>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    attr_set = true;
  }
  const dim3 grid(static_cast<unsigned>(((n + kBN - 1) / kBN) * ((m + kBM - 1) / kBM)));
  fp4w_gemm_kernel<OutT, kGroupM><<<grid, kThreads, kSmem, stream>>>(
      act, act_stride, w.payload, w.scales, static_cast<size_t>(k) / 16, w.global_scale, out, m, n,
      k, out_stride);
  DGPP_CUDA_OK(cudaGetLastError());
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
  // Three M tiles per cache group improve the long-prefill shapes. Shorter
  // walks retain four; changing the block order does not reassociate a dot.
  if (m >= 3072)
    launch_tiles<OutT, 3>(act, act_stride, w, out, m, n, k, stream, out_stride);
  else
    launch_tiles<OutT, 4>(act, act_stride, w, out, m, n, k, stream, out_stride);
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
