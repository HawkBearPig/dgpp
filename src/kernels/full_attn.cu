// Dense causal paged attention, one warp per (query, KV group) -- the shape of
// qsa_warp.cu (SGLang's sparse GQA prefill kernel on GB10's tensor cores) with
// the listed top-k gather replaced by a dense causal range: row r attends
// positions [0, pos[r]] over the paged K/V cache.
//
// The whole online softmax stays in one warp's registers: Q as m16
// A-fragments (the group's <= 16 query heads are the M rows), K/V tiles of 16
// tokens gathered into shared memory with cp.async (L1-allocating),
// S = Q K^T and O += P V on mma.sync m16n8k16, P reused from S's accumulator
// layout as the A-fragment (FlashAttention-2), no block barrier. It writes the
// normalized fp32 output straight into the caller's out -- no m/l/c
// partials, no combine pass. Lanes 0-15 resolve a tile's 16 physical rows
// (token -> block table) in one round; a serial chain starved the copies.
// One stage by default: the two-stage form is shared-memory bound at 2
// blocks/SM; one stage gives 5 warps/SM and residency hides the gather.
//
// Numerics: probabilities rounded to bf16 for P V, the denominator summing
// the unrounded values (the QSA rule); the dots' summation order is the
// tensor cores'. Tolerance-equal to a dense reference, not bitwise.
#include "kernels/full_attn.hpp"

#include <cmath>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {

constexpr int kD = 256;
constexpr int kTileTok = 16;
constexpr int kRow = kD + 8;  // padded bf16 row: conflict-free ldmatrix
constexpr int kStageElems = 2 * kTileTok * kRow;  // K then V
#ifndef DGPP_FULL_ATTN_WARP_STAGES
#define DGPP_FULL_ATTN_WARP_STAGES 1
#endif
constexpr int kStages = DGPP_FULL_ATTN_WARP_STAGES;
constexpr size_t kSmem = size_t(kStages) * kStageElems * 2;  // bytes

__device__ __forceinline__ void fldsm_x4(uint32_t (&r)[4], const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void fldsm_x4_t(uint32_t (&r)[4], const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void fmma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void fcp_async16(void* dst, const void* src, bool valid) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  const int bytes = valid ? 16 : 0;  // 0: zero-fill
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(bytes));
}
__device__ __forceinline__ uint32_t fpack_bf16(float lo, float hi) {
  return static_cast<uint32_t>(float_to_bf16_bits(lo)) | (static_cast<uint32_t>(float_to_bf16_bits(hi)) << 16);
}

__global__ __launch_bounds__(32) void full_attn_prefill_warp_kernel(
    const uint16_t* __restrict__ q, int64_t q_row_stride, const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache, const int32_t* __restrict__ req_ids, const int64_t* __restrict__ pos,
    int local_heads, int kv_heads, int block_tokens, const int32_t* __restrict__ block_tables,
    int blocks_per_request, float scale, float* __restrict__ out) {
  extern __shared__ __align__(16) uint16_t sm[];
  const int64_t r = blockIdx.x;
  const int kvh = blockIdx.y;
  const int group = local_heads / kv_heads;
  const int h0 = kvh * group;
  const int lane = threadIdx.x;
  const int g = lane >> 2, t = lane & 3;
  const int width = kv_heads * kD;
  const int64_t cnt = pos[r] + 1;  // dense causal visible set [0, pos[r]]
  const int32_t* bt = block_tables + static_cast<int64_t>(req_ids[r]) * blocks_per_request;
  const float sl2 = scale * 1.4426950408889634f;

  // Q A-fragments, 16 k-steps x 4 registers; rows past the group are zero.
  uint32_t qa[16][4];
  {
    const uint16_t* q0 = q + r * q_row_stride + static_cast<int64_t>(h0) * kD;
    const bool v0 = g < group, v1 = g + 8 < group;
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      const int c = ks * 16 + 2 * t;
      qa[ks][0] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + g * kD + c) : 0u;
      qa[ks][1] = v1 ? *reinterpret_cast<const uint32_t*>(q0 + (g + 8) * kD + c) : 0u;
      qa[ks][2] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + g * kD + c + 8) : 0u;
      qa[ks][3] = v1 ? *reinterpret_cast<const uint32_t*>(q0 + (g + 8) * kD + c + 8) : 0u;
    }
  }

  // Stage a tile: 16 tokens x (K, V) x 32 16-byte chunks = 1024 chunks, 32 per lane.
  auto stage = [&](int64_t tile, int buf) {
    uint16_t* kt = sm + buf * kStageElems;
    uint16_t* vt = kt + kTileTok * kRow;
    const int64_t base = tile * kTileTok;
    // Lanes 0..15 resolve the tile's 16 physical rows in one round of loads
    // (a serial token -> block-table chain per issue step starved the copies).
    int64_t my_off = 0;
    if (lane < kTileTok && base + lane < cnt) {
      const int64_t tok = base + lane;
      const int64_t phys = static_cast<int64_t>(bt[tok / block_tokens]) * block_tokens + tok % block_tokens;
      my_off = phys * width + static_cast<int64_t>(kvh) * kD;
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {            // token i, lane = its 16-byte chunk
      const int64_t off = __shfl_sync(0xffffffffu, my_off, i) + lane * 8;
      const bool valid = base + i < cnt;
      fcp_async16(kt + i * kRow + lane * 8, k_cache + off, valid);
      fcp_async16(vt + i * kRow + lane * 8, v_cache + off, valid);
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };

  float acc[32][4];
#pragma unroll
  for (int d = 0; d < 32; ++d) acc[d][0] = acc[d][1] = acc[d][2] = acc[d][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;  // rows g, g + 8 (lane-partial l)

  const int64_t tiles = (cnt + kTileTok - 1) / kTileTok;
  if (kStages == 2 && tiles > 0) stage(0, 0);
  for (int64_t it = 0; it < tiles; ++it) {
    const int buf = kStages == 2 ? (it & 1) : 0;
    if (kStages == 1) {
      // One stage: residency (more warps per SM) hides the gather instead.
      stage(it, 0);
      asm volatile("cp.async.wait_group 0;\n" ::);
    } else if (it + 1 < tiles) {
      stage(it + 1, buf ^ 1);
      asm volatile("cp.async.wait_group 1;\n" ::);
    } else {
      asm volatile("cp.async.wait_group 0;\n" ::);
    }
    __syncwarp();
    const uint16_t* kt = sm + buf * kStageElems;
    const uint16_t* vt = kt + kTileTok * kRow;
    // S = Q K^T: two n8 tiles (tokens 0-7, 8-15).
    float sc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      uint32_t b[4];
      fldsm_x4(b, kt + ((lane & 7) + ((lane >> 4) << 3)) * kRow + ks * 16 + ((lane >> 3) & 1) * 8);
      fmma_bf16(sc[0], qa[ks], b[0], b[1]);
      fmma_bf16(sc[1], qa[ks], b[2], b[3]);
    }
    // Mask tokens past the visible set, scale into the exp2 domain, row maxima over the quad.
    const int n = min(kTileTok, static_cast<int>(cnt - it * kTileTok));
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int col = j * 8 + 2 * t + e;
        sc[j][e] = col < n ? sc[j][e] * sl2 : -INFINITY;
        sc[j][2 + e] = col < n ? sc[j][2 + e] * sl2 : -INFINITY;
        mx0 = fmaxf(mx0, sc[j][e]);
        mx1 = fmaxf(mx1, sc[j][2 + e]);
      }
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
    }
    const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
    const float a0 = exp2f(m0 - n0), a1 = exp2f(m1 - n1);  // m = -inf on the first tile: 0
    m0 = n0;
    m1 = n1;
    float p[2][4];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      p[j][0] = exp2f(sc[j][0] - n0);
      p[j][1] = exp2f(sc[j][1] - n0);
      p[j][2] = exp2f(sc[j][2] - n1);
      p[j][3] = exp2f(sc[j][3] - n1);
    }
    l0 = l0 * a0 + p[0][0] + p[0][1] + p[1][0] + p[1][1];
    l1 = l1 * a1 + p[0][2] + p[0][3] + p[1][2] + p[1][3];
    // P as the A-fragment over k = the tile's 16 tokens.
    const uint32_t pa[4] = {fpack_bf16(p[0][0], p[0][1]), fpack_bf16(p[0][2], p[0][3]),
                            fpack_bf16(p[1][0], p[1][1]), fpack_bf16(p[1][2], p[1][3])};
#pragma unroll
    for (int d = 0; d < 32; ++d) {
      acc[d][0] *= a0;
      acc[d][1] *= a0;
      acc[d][2] *= a1;
      acc[d][3] *= a1;
    }
    // O += P V: 16 x4.trans loads, two n8 dim tiles each.
#pragma unroll
    for (int dp = 0; dp < 16; ++dp) {
      uint32_t b[4];
      fldsm_x4_t(b, vt + ((lane & 7) + ((lane >> 3) & 1) * 8) * kRow + dp * 16 + (lane >> 4) * 8);
      fmma_bf16(acc[dp * 2], pa, b[0], b[1]);
      fmma_bf16(acc[dp * 2 + 1], pa, b[2], b[3]);
    }
    __syncwarp();  // this buffer's reads done before the next stage() overwrites it
  }
#pragma unroll
  for (int o = 1; o <= 2; o <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, o);
    l1 += __shfl_xor_sync(0xffffffffu, l1, o);
  }
  const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
  float* o0 = out + (r * local_heads + h0 + g) * kD;
  float* o1 = out + (r * local_heads + h0 + g + 8) * kD;
#pragma unroll
  for (int d = 0; d < 32; ++d) {
    const int col = d * 8 + 2 * t;
    if (g < group) *reinterpret_cast<float2*>(o0 + col) = make_float2(acc[d][0] * i0, acc[d][1] * i0);
    if (g + 8 < group) *reinterpret_cast<float2*>(o1 + col) = make_float2(acc[d][2] * i1, acc[d][3] * i1);
  }
}

}  // namespace

bool full_attn_supported(int dim, int local_heads, int kv_heads) {
  return dim == kD && kv_heads > 0 && local_heads % kv_heads == 0 && local_heads / kv_heads <= 16;
}

void full_attn_prefill_warp(const uint16_t* q, int64_t q_row_stride, const uint16_t* k_cache,
                            const uint16_t* v_cache, const int32_t* req_ids, const int64_t* pos,
                            int rows, int local_heads, int kv_heads, int block_tokens,
                            const int32_t* block_tables, int blocks_per_request, float scale,
                            float* out, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!full_attn_supported(kD, local_heads, kv_heads))
    throw std::invalid_argument("full_attn_prefill_warp: dim 256, <= 16 query heads per KV head");
  static const bool opted = [] {  // once per process, thread-safe (TP ranks may share one)
    DGPP_CUDA_OK(cudaFuncSetAttribute(full_attn_prefill_warp_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmem)));
    // All of L1 as shared memory: residency here is bounded by the 33.8 KB stages.
    DGPP_CUDA_OK(cudaFuncSetAttribute(full_attn_prefill_warp_kernel,
                                      cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    return true;
  }();
  (void)opted;
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(kv_heads));
  full_attn_prefill_warp_kernel<<<grid, 32, kSmem, stream>>>(q, q_row_stride, k_cache, v_cache, req_ids, pos,
                                                             local_heads, kv_heads, block_tokens, block_tables,
                                                             blocks_per_request, scale, out);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
