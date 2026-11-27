// The grouped MoE GEMM with E4M3 activations and MXFP4 weights on GB10's
// block-scaled tensor cores.
//
// Why (2026-11-27): the DeepSeek-V4.1 release multiplies its routed experts
// as fp4_gemm(act_quant(x), w): FP8 activations with a power-of-two scale
// per 32 against E2M1 weights with a power-of-two scale per 32, accumulated
// in FP32. moe_grouped_mma_fp4_ldm_kernel reaches the same sums by decoding
// both operands to BF16 (HMMA m16n8k16) and runs near 27 TFLOP/s on a rank
// whatever the rows per expert. sm_121a multiplies these operand types as
// they are: mma.sync kind::mxf8f6f4.block_scale, m16n8k32, one E8M0 scale
// per A row and per B column of the k32 block. An instruction covers twice
// the k of the BF16 one, the activations move as one byte each and the
// weights need no decode, only their nibbles spread into bytes.
//
// Fragment layouts (m16n8k32, the layout mxfp8_gemm.cu and moe_w4a4.cu use):
// lane = 4q + t.
//   A (row-major 16 x 32 bytes): a0 = row q bytes [4t, 4t + 4), a1 = row
//      q + 8, a2 = row q bytes [16 + 4t, ...), a3 = row q + 8 likewise.
//   B (col-major 32 x 8): b0 = column q bytes [4t, 4t + 4), b1 = + 16 bytes.
//      An E2M1 code sits in bits 2..5 of its byte (0b00ABCD00).
//   sfa: the E8M0 scale of row q + 8 * (lane & 1); sfb: of column q.
//   C: c0, c1 = row q cols 2t, 2t + 1; c2, c3 = row q + 8.
#include "kernels/moe_a8w4.hpp"
#include "kernels/pdl.cuh"

#include <stdexcept>

#include <cuda_bf16.h>

#include "common/cuda_check.hpp"

namespace dgpp {
namespace {

namespace a8w4 {
constexpr int BM = 64, BN = 128, BK = 64;  // a stage: 64 activation bytes and 32 packed weight bytes a row
constexpr int kThreads = 256, kStages = 3;
constexpr int kARow = 80;  // 64 code bytes + 16 pad: conflict-free fragment reads
// A weight row's stage: four groups of 16 codes, each group's 8 packed bytes
// at the start of a 16-byte cell (the unpacking load's source format), + 16 pad.
constexpr int kBRow = 80;
constexpr size_t kACodes = size_t(BM) * kARow;  // 5,120
constexpr size_t kBCodes = size_t(BN) * kBRow;  // 10,240
constexpr size_t kAScales = size_t(BM) * 2;     // one byte per k32 block, two blocks a stage
constexpr size_t kBScales = size_t(BN) * 2;
constexpr size_t kSlot = kACodes + kBCodes + kAScales + kBScales;  // 15,744
constexpr size_t kSmem = kStages * kSlot;                          // 47,232: two blocks an SM (102,400 B)
constexpr int kPrefetchStages = 4;  // a weight row's next 128-byte line: four stages of packed codes
static_assert(kSlot % 16 == 0, "16-byte aligned slots");
static_assert(BM * (BK / 16) == kThreads, "one 16-byte activation copy per thread");
static_assert(BN * 2 == kThreads, "two 8-byte weight copies per thread");
}  // namespace a8w4

__device__ __forceinline__ void cp_async16(void* dst, const void* src, int src_bytes) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async8(void* dst, const void* src, int src_bytes) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(d), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem_ptr) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

// Four matrices of 8 rows x 16 four-bit codes, each row a 16-byte cell whose
// first 8 bytes hold the codes packed (the low nibble the even element):
// lane 4q + t receives, per matrix, codes [4t, 4t + 4) of its row q, one a
// byte, in the byte's low four bits (measured on the GB10, 2026-11-27).
// Lane l supplies row (l & 7) of matrix l >> 3. The instruction then reads
// an E2M1 code from bits 2..5 of its byte: the caller shifts by two.
__device__ __forceinline__ void ldmatrix_fp4_x4(uint32_t (&r)[4], const void* smem_ptr) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

__device__ __forceinline__ void mma_a8w4(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                         uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
      "m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
      "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa),
        "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)), "r"(sfb),
        "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)));
}

__device__ __forceinline__ void store_out(uint16_t* p, float v) {
  const __nv_bfloat16 b = __float2bfloat16_rn(v);
  *p = *reinterpret_cast<const uint16_t*>(&b);
}
__device__ __forceinline__ void store_out(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_pair(uint16_t* p, float a, float b) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  *reinterpret_cast<__nv_bfloat162*>(p) = v;
}
__device__ __forceinline__ void store_pair(float* p, float a, float b) {
  *reinterpret_cast<float2*>(p) = make_float2(a, b);
}

// One 64 x 128 output tile per block, the m-tile the fastest grid index (a
// long segment's m-tiles run together and re-read its weight slice through
// L2). Eight warps as 2 (m32) x 4 (n32): a warp's k32 step is two
// ldmatrix.x4 for A, two unpacking ldmatrix.x4 for B and eight
// instructions. A three-slot cp.async ring carries the stage's activation
// bytes and the weights' packed codes; the scale bytes follow by plain loads.
template <typename OutT>
__global__ __launch_bounds__(a8w4::kThreads, 2) void moe_grouped_a8w4_kernel(
    const uint8_t* __restrict__ a_codes, size_t a_code_stride, const uint8_t* __restrict__ a_scales,
    size_t a_scale_stride, const int32_t* __restrict__ act_rows, const MoeSegment* __restrict__ segs,
    const MoeExpertView* __restrict__ views, int which, OutT* __restrict__ out, size_t out_stride, int n, int k,
    int m_tiles, int pairs) {
  dgpp_pdl_wait();
  using namespace a8w4;
  extern __shared__ __align__(16) uint8_t smem[];
  const MoeSegment seg = segs[blockIdx.y];
  const int m_tile = static_cast<int>(blockIdx.x) % m_tiles;
  const int n_tile = static_cast<int>(blockIdx.x) / m_tiles;
  const int m0 = m_tile * BM;
  if (m0 >= seg.rows) return;  // beyond this segment's rows (no barrier yet)
  const int m_rows = min(BM, seg.rows - m0);
  const MoeExpertView v = views[seg.expert * 3 + which];
  const int n0 = n_tile * BN;
  const size_t w_code_stride = static_cast<size_t>(k) / 2;
  const size_t w_scale_stride = static_cast<size_t>(k) / 32;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5, lane = tid & 31;
  const int q = lane >> 2, t = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;
  const int stages = k / BK;

  // The copies of a stage, fixed per thread. A codes: row tid / 4, 16-byte
  // chunk tid % 4. B codes: weight row tid / 2, the two 16-code groups of
  // half tid % 2 (8 packed bytes each, into their cells). A scales: threads
  // below BM, their row's two bytes. B scales: threads below BN likewise.
  const int a_row = tid >> 2, a_chunk = tid & 3;
  const bool a_ok = a_row < m_rows;
  int a_src_row = 0;
  if (a_ok) {
    const int srow = seg.row0 + m0 + a_row;
    a_src_row = act_rows != nullptr ? act_rows[srow] : srow;
  }
  const uint8_t* a_src = a_codes + static_cast<size_t>(a_src_row) * a_code_stride + a_chunk * 16;
  const bool as_ok = tid < BM && tid < m_rows;
  int as_src_row = 0;
  if (as_ok) {
    const int srow = seg.row0 + m0 + tid;
    as_src_row = act_rows != nullptr ? act_rows[srow] : srow;
  }
  const uint8_t* as_src = a_scales + static_cast<size_t>(as_src_row) * a_scale_stride;
  const int b_row = tid >> 1, b_half = tid & 1;
  const bool b_ok = n0 + b_row < n;
  const uint8_t* b_line = v.payload + static_cast<size_t>(b_ok ? n0 + b_row : 0) * w_code_stride;
  const bool bs_ok = tid < BN && n0 + tid < n;
  const uint8_t* bs_src = v.fp4_scales + static_cast<size_t>(bs_ok ? n0 + tid : 0) * w_scale_stride;

  auto slot_ptr = [&](int slot) { return smem + static_cast<size_t>(slot) * kSlot; };

  // Absent rows arrive as zeros (src-size 0): zero codes and scale bytes.
  auto issue = [&](int s, int slot) {
    uint8_t* base = slot_ptr(slot);
    {
      const uint8_t* src = a_src + static_cast<size_t>(s) * BK;
      // The row's 128-byte line (two stages), tagged evict_last as it
      // enters L2: the segment's other n-tile blocks re-read it while the
      // weight stream passes through.
      if (a_ok && a_chunk == 0 && (s & 1) == 0) asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(src));
      cp_async16(base + a_row * kARow + a_chunk * 16, a_ok ? src : a_codes, a_ok ? 16 : 0);
    }
    {
      const uint8_t* src = b_line + static_cast<size_t>(s) * (BK / 2) + b_half * 16;
      uint8_t* dst = base + kACodes + b_row * kBRow + b_half * 32;
      cp_async8(dst, b_ok ? src : v.payload, b_ok ? 8 : 0);
      cp_async8(dst + 16, b_ok ? src + 8 : v.payload, b_ok ? 8 : 0);
      // A stage reads 32 bytes of each weight row, rows k / 2 bytes apart:
      // pull the row's next 128-byte line into L2 whole, ahead of the ring.
      if (b_half == 0 && b_ok && (s % kPrefetchStages) == 0) {
        const int ns = s + kPrefetchStages;
        if (ns < stages) {
          const uint8_t* line = b_line + static_cast<size_t>(ns) * (BK / 2);
          asm volatile("prefetch.global.L2 [%0];" ::"l"(line));
        }
      }
    }
    if (tid < BM) {
      uint32_t w = 0;
      if (as_ok) w = static_cast<uint32_t>(as_src[2 * s]) | (static_cast<uint32_t>(as_src[2 * s + 1]) << 8);
      *reinterpret_cast<uint16_t*>(base + kACodes + kBCodes + tid * 2) = static_cast<uint16_t>(w);
    }
    if (tid < BN) {
      uint32_t w = 0;
      if (bs_ok) w = static_cast<uint32_t>(bs_src[2 * s]) | (static_cast<uint32_t>(bs_src[2 * s + 1]) << 8);
      *reinterpret_cast<uint16_t*>(base + kACodes + kBCodes + kAScales + tid * 2) = static_cast<uint16_t>(w);
    }
  };

  float acc[2][4][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  // Prologue: stages 0 .. kStages - 2 in flight.
#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < stages) issue(s, s);
    cp_commit();
  }
  // ldmatrix.x4 lane addressing: lane l feeds row (l & 7) of matrix l >> 3.
  const int lm = lane >> 3, lr = lane & 7;
  for (int s = 0; s < stages; ++s) {
    cp_wait<kStages - 2>();  // stage s landed for this thread
    __syncthreads();         // ... for every thread, and everyone is past stage s - 1's multiply
    if (s + kStages - 1 < stages) issue(s + kStages - 1, (s + kStages - 1) % kStages);
    cp_commit();
    const uint8_t* base = slot_ptr(s % kStages);
    const uint8_t* ab = base;
    const uint8_t* bb = base + kACodes;
    const uint16_t* asw = reinterpret_cast<const uint16_t*>(base + kACodes + kBCodes);
    const uint16_t* bsw = reinterpret_cast<const uint16_t*>(base + kACodes + kBCodes + kAScales);
    uint32_t sa2[2], sb2[4];
#pragma unroll
    for (int i = 0; i < 2; ++i) sa2[i] = asw[wm * 32 + i * 16 + q + ((lane & 1) << 3)];
#pragma unroll
    for (int j = 0; j < 4; ++j) sb2[j] = bsw[wn * 32 + j * 8 + q];
#pragma unroll
    for (int kh = 0; kh < 2; ++kh) {  // two k32 blocks a stage
      uint32_t af[2][4], sfa[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        // matrices: rows + 0 / + 8 (lm & 1) x bytes + 0 / + 16 (lm >> 1) -> a0, a1, a2, a3.
        const int row = wm * 32 + i * 16 + (lm & 1) * 8 + lr;
        ldmatrix_x4(af[i], ab + row * kARow + kh * 32 + (lm >> 1) * 16);
        sfa[i] = (sa2[i] >> (8 * kh)) & 0xFFu;
      }
#pragma unroll
      for (int jp = 0; jp < 2; ++jp) {
        // matrices: n8 tiles 2 jp / 2 jp + 1 (lm >> 1) x the block's two
        // 16-code groups (lm & 1) -> b0, b1 of each tile.
        uint32_t bf[4];
        const int col = wn * 32 + (jp * 2 + (lm >> 1)) * 8 + lr;
        ldmatrix_fp4_x4(bf, bb + col * kBRow + (kh * 2 + (lm & 1)) * 16);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int j = jp * 2 + h;
          const uint32_t b0 = bf[2 * h] << 2, b1 = bf[2 * h + 1] << 2;
          const uint32_t sfb = (sb2[j] >> (8 * kh)) & 0xFFu;
#pragma unroll
          for (int i = 0; i < 2; ++i) mma_a8w4(acc[i][j], af[i], b0, b1, sfa[i], sfb);
        }
      }
    }
  }
  cp_wait<0>();

  // Epilogue: the warp's 32 x 32 slice.
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int mm = wm * 32 + i * 16 + q + h * 8;
      if (mm >= m_rows) continue;
      OutT* orow = out + static_cast<size_t>(seg.row0 + m0 + mm) * out_stride;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int col = n0 + wn * 32 + j * 8 + 2 * t;
        if (pairs != 0 && col + 1 < n) {
          store_pair(orow + col, acc[i][j][2 * h], acc[i][j][2 * h + 1]);
        } else {
          if (col < n) store_out(orow + col, acc[i][j][2 * h]);
          if (col + 1 < n) store_out(orow + col + 1, acc[i][j][2 * h + 1]);
        }
      }
    }
  }
}

template <typename OutT>
void launch_a8w4(const uint8_t* codes, size_t code_stride, const uint8_t* scales, size_t scale_stride,
                 const int32_t* act_rows, const MoeSegment* segs, int n_segs, int max_rows,
                 const MoeExpertView* views, int which, OutT* out, size_t out_stride, int n, int k,
                 cudaStream_t stream) {
  using namespace a8w4;
  if (n_segs <= 0 || n <= 0) return;
  if (!codes || !scales || !segs || !views || !out) throw std::invalid_argument("moe a8w4: null pointer");
  if (!moe_grouped_a8w4_supported(k)) throw std::invalid_argument("moe a8w4: k must be a multiple of 64");
  if (reinterpret_cast<uintptr_t>(codes) % 16 != 0 || code_stride % 16 != 0 || code_stride < static_cast<size_t>(k))
    throw std::invalid_argument("moe a8w4: activation code rows must be 16-byte aligned and hold k bytes");
  if (scale_stride < static_cast<size_t>(k) / 32) throw std::invalid_argument("moe a8w4: activation scale rows");
  if (max_rows <= 0) throw std::invalid_argument("moe a8w4: max_rows");
  // The dynamic shared memory opt-in at this instantiation's first launch.
  static const bool opted = [] {
    DGPP_CUDA_OK(cudaFuncSetAttribute(moe_grouped_a8w4_kernel<OutT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(kSmem)));
    return true;
  }();
  (void)opted;
  const int m_tiles = (max_rows + BM - 1) / BM;
  const unsigned n_tiles = static_cast<unsigned>((n + BN - 1) / BN);
  if (static_cast<size_t>(n_tiles) * m_tiles > 0x7fffffffu) throw std::invalid_argument("moe a8w4: grid too large");
  // Column pairs as one store when the rows allow it.
  const int pairs = (out_stride % 2 == 0 && reinterpret_cast<uintptr_t>(out) % (2 * sizeof(OutT)) == 0) ? 1 : 0;
  const dim3 grid(n_tiles * static_cast<unsigned>(m_tiles), static_cast<unsigned>(n_segs), 1u);
  moe_grouped_a8w4_kernel<OutT><<<grid, kThreads, kSmem, stream>>>(codes, code_stride, scales, scale_stride, act_rows,
                                                                  segs, views, which, out, out_stride, n, k, m_tiles,
                                                                  pairs);
  DGPP_CUDA_OK(cudaGetLastError());
}
}  // namespace

void launch_moe_grouped_a8w4_bf16(const uint8_t* codes, size_t code_stride, const uint8_t* scales,
                                  size_t scale_stride, const int32_t* act_rows, const MoeSegment* segs, int n_segs,
                                  int max_rows, const MoeExpertView* views, int which, uint16_t* out,
                                  size_t out_stride, int n, int k, cudaStream_t stream) {
  launch_a8w4<uint16_t>(codes, code_stride, scales, scale_stride, act_rows, segs, n_segs, max_rows, views, which, out,
                        out_stride, n, k, stream);
}

void launch_moe_grouped_a8w4_f32(const uint8_t* codes, size_t code_stride, const uint8_t* scales,
                                 size_t scale_stride, const int32_t* act_rows, const MoeSegment* segs, int n_segs,
                                 int max_rows, const MoeExpertView* views, int which, float* out, size_t out_stride,
                                 int n, int k, cudaStream_t stream) {
  launch_a8w4<float>(codes, code_stride, scales, scale_stride, act_rows, segs, n_segs, max_rows, views, which, out,
                     out_stride, n, k, stream);
}
}  // namespace dgpp
