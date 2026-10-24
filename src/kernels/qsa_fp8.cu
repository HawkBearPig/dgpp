// Listed QSA attention over e4m3 K/V, with native e4m3 QK and PV products.
// Q is quantized per query head. K scales multiply individual score columns;
// V scales multiply probabilities before their per-tile e4m3 quantization.
// K/V codes are copied to shared memory verbatim. The online softmax and
// accumulators stay fp32. See docs/qwen_fp8_kv_plan.md for the recipe.
#include "kernels/qsa.hpp"
#include "kernels/pdl.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/dsa.hpp"

namespace dgpp {
namespace {
constexpr int kDim = 256, kTokens = 32, kRow = kDim + 16;
constexpr int kPlaneBytes = kTokens * kRow;
constexpr unsigned kMask = 0xffffffffu;
constexpr float kLog2e = 1.4426950408889634f, kLn2 = 0.6931471805599453f;

__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack(float a, float b, float c, float d) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(b), "f"(a));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(d), "f"(c));
  return uint32_t(lo) | (uint32_t(hi) << 16);
}
__device__ __forceinline__ uint32_t permute(uint32_t a, uint32_t b, uint32_t selector) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(selector));
  return r;
}
__device__ __forceinline__ void load4(uint32_t (&r)[4], const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void load4_trans(uint32_t (&r)[4], const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void copy16(void* dst, const void* src, bool valid) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  const int bytes = valid ? 16 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" :: "r"(d), "l"(src), "r"(bytes));
}
__device__ __forceinline__ float quad_max(float x) {
  x = fmaxf(x, __shfl_xor_sync(kMask, x, 1));
  return fmaxf(x, __shfl_xor_sync(kMask, x, 2));
}

__global__ __launch_bounds__(32) void attention(
    const uint16_t* __restrict__ q, int64_t q_stride, const uint8_t* __restrict__ k,
    const uint8_t* __restrict__ v, const float* __restrict__ ks, const float* __restrict__ vs,
    const int32_t* __restrict__ req, const int32_t* __restrict__ topk, int topk_stride,
    const int32_t* __restrict__ counts, int heads, int kv_heads, int block_tokens,
    const int32_t* __restrict__ tables, int blocks_per_request, float scale, int splits,
    float* __restrict__ mw, float* __restrict__ lw, float* __restrict__ cw, float* __restrict__ out) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint8_t sm[];
  uint8_t* kt = sm;
  uint8_t* vt = sm + kPlaneBytes;
  const int row = blockIdx.x, kvh = blockIdx.y / splits, split = blockIdx.y % splits;
  const int group = heads / kv_heads, head0 = kvh * group + blockIdx.z * 16;
  const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  const bool valid0 = blockIdx.z * 16 + g < group, valid1 = blockIdx.z * 16 + g + 8 < group;
  const int width = kv_heads * kDim;
  const int count = counts[row], chunk = (count + splits - 1) / splits;
  const int begin = split * chunk, end = min(count, begin + chunk);
  const int32_t* bt = tables + int64_t(req[row]) * blocks_per_request;
  const int32_t* tokens = topk + int64_t(row) * topk_stride;

  // A fragments: four lanes per Q head, 64 elements per lane. Do not
  // retain the bf16 row in registers across the absmax reduction.
  uint32_t qa[8][4];
  float qscale[2];
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const bool valid = half == 0 ? valid0 : valid1;
    const uint16_t* qr = q + int64_t(row) * q_stride + int64_t(head0 + g + half * 8) * kDim;
    float amax = 0.f;
#pragma unroll
    for (int c = 0; c < 8; ++c)
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          if (valid) amax = fmaxf(amax, fabsf(bf16_bits_to_float(qr[c * 32 + h * 16 + t * 4 + j])));
    amax = quad_max(amax);
    const float inv = amax > 0.f ? 448.f / amax : 0.f;
    qscale[half] = amax * (1.f / 448.f) * scale * kLog2e;
#pragma unroll
    for (int c = 0; c < 8; ++c)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int d = c * 32 + h * 16 + t * 4;
        qa[c][h * 2 + half] = valid ? pack(bf16_bits_to_float(qr[d]) * inv,
            bf16_bits_to_float(qr[d + 1]) * inv, bf16_bits_to_float(qr[d + 2]) * inv,
            bf16_bits_to_float(qr[d + 3]) * inv) : 0u;
      }
  }
  float acc[32][4] = {};
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {}, pscale[2] = {1.f, 1.f};
  for (int base = begin; base < end; base += kTokens) {
    const int n = min(kTokens, end - base);
    int64_t phys = 0;
    if (lane < n) {
      const int tok = tokens[base + lane];
      phys = int64_t(bt[tok / block_tokens]) * block_tokens + tok % block_tokens;
    }
    const float my_ks = lane < n ? ks[phys * kv_heads + kvh] : 0.f;
    const float my_vs = lane < n ? vs[phys * kv_heads + kvh] : 0.f;
#pragma unroll
    for (int i = 0; i < kTokens * (kDim / 16) / 32; ++i) {
      const int vec = i * 32 + lane, tok = vec / (kDim / 16), col = (vec % (kDim / 16)) * 16;
      const int64_t at = __shfl_sync(kMask, phys, tok) * width + kvh * kDim + col;
      copy16(kt + tok * kRow + col, k + at, tok < n);
      copy16(vt + tok * kRow + col, v + at, tok < n);
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::);
    __syncwarp();

    float scores[4][4] = {};
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      uint32_t b[4];
      const int kr = (lane & 7) + (lane >> 4) * 8;
      const int col = c * 32 + ((lane >> 3) & 1) * 16;
      load4(b, kt + kr * kRow + col);
      mma(scores[0], qa[c], b[0], b[1]);
      mma(scores[1], qa[c], b[2], b[3]);
      load4(b, kt + (kr + 16) * kRow + col);
      mma(scores[2], qa[c], b[0], b[1]);
      mma(scores[3], qa[c], b[2], b[3]);
    }
    float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int tok = j * 8 + 2 * t + e;
        const float kscale = __shfl_sync(kMask, my_ks, tok);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          float& s = scores[j][half * 2 + e];
          s = tok < n ? s * qscale[half] * kscale : -INFINITY;
          mx[half] = fmaxf(mx[half], s);
        }
      }
    float rescale[2], pn[2] = {};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const float nm = fmaxf(m[half], quad_max(mx[half]));
      rescale[half] = exp2f(m[half] - nm);
      m[half] = nm;
      l[half] *= rescale[half];
    }
    // The numerator needs P * scale_V[token]. Quantize that product,
    // with one dynamic scale per query head and tile. Normalizing by its
    // maximum protects small probabilities even when V scales vary widely.
    float p[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float vscale = __shfl_sync(kMask, my_vs, j * 8 + 2 * t + e);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const float prob = exp2f(scores[j][half * 2 + e] - m[half] + 8.f);
          l[half] += prob;
          p[j][half * 2 + e] = prob * vscale;
          pn[half] = fmaxf(pn[half], prob * vscale);
        }
      }
    float pinv[2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const float maximum = quad_max(pn[half]);
      pn[half] = maximum > 0.f ? maximum * (1.f / 256.f) : 1.f;
      pinv[half] = 1.f / pn[half];
      rescale[half] *= pscale[half] * pinv[half];
      pscale[half] = pn[half];
    }
#pragma unroll
    for (int d = 0; d < 32; ++d) {
      acc[d][0] *= rescale[0]; acc[d][1] *= rescale[0];
      acc[d][2] *= rescale[1]; acc[d][3] *= rescale[1];
    }
    const uint32_t pa[4] = {
        pack(p[0][0] * pinv[0], p[0][1] * pinv[0], p[1][0] * pinv[0], p[1][1] * pinv[0]),
        pack(p[0][2] * pinv[1], p[0][3] * pinv[1], p[1][2] * pinv[1], p[1][3] * pinv[1]),
        pack(p[2][0] * pinv[0], p[2][1] * pinv[0], p[3][0] * pinv[0], p[3][1] * pinv[0]),
        pack(p[2][2] * pinv[1], p[2][3] * pinv[1], p[3][2] * pinv[1], p[3][3] * pinv[1])};
#pragma unroll
    for (int dp = 0; dp < 16; ++dp) {
      uint32_t b[4];
      load4_trans(b, vt + ((lane & 7) + (lane >> 3) * 8) * kRow + dp * 16);
      mma(acc[2 * dp], pa, permute(b[0], b[1], 0x6420u), permute(b[2], b[3], 0x6420u));
      mma(acc[2 * dp + 1], pa, permute(b[0], b[1], 0x7531u), permute(b[2], b[3], 0x7531u));
    }
    __syncwarp();
  }
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    l[half] += __shfl_xor_sync(kMask, l[half], 1);
    l[half] += __shfl_xor_sync(kMask, l[half], 2);
    const int head = head0 + g + 8 * half;
    const bool valid = half == 0 ? valid0 : valid1;
    const int64_t at = (int64_t(row) * splits + split) * heads + head;
    float factor = pscale[half];
    float* dest = cw;
    if (splits == 1) {
      dest = out;
      factor *= l[half] > 0.f ? 1.f / l[half] : 0.f;
    } else if (valid && t == 0) {
      mw[at] = m[half] * kLn2;
      lw[at] = l[half];
    }
#pragma unroll
    for (int dp = 0; dp < 16; ++dp)
      if (valid) {
        const int e = half * 2;
        *reinterpret_cast<float4*>(dest + at * kDim + dp * 16 + 4 * t) =
            make_float4(acc[2 * dp][e] * factor, acc[2 * dp + 1][e] * factor,
                        acc[2 * dp][e + 1] * factor, acc[2 * dp + 1][e + 1] * factor);
      }
  }
}
}  // namespace

void qsa_attn_fp8(const uint16_t* q, int64_t q_row_stride, const uint8_t* k, const uint8_t* v,
                   const float* k_scale, const float* v_scale, const int32_t* req_ids,
                   const int32_t* topk, int topk_stride, const int32_t* counts, int rows,
                   int local_heads, int kv_heads, int block_tokens, const int32_t* block_tables,
                   int blocks_per_request, float scale, int n_split, float* m_ws, float* l_ws,
                   float* c_ws, float* out, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!q || !k || !v || !k_scale || !v_scale || !req_ids || !topk || !counts || !block_tables || !out)
    throw std::invalid_argument("qsa_attn_fp8: null pointer");
  if (local_heads <= 0 || kv_heads <= 0 || local_heads % kv_heads != 0 ||
      block_tokens <= 0 || blocks_per_request <= 0 || topk_stride <= 0 || n_split <= 0)
    throw std::invalid_argument("qsa_attn_fp8: invalid geometry");
  if (q_row_stride < int64_t(local_heads) * kDim || q_row_stride % 2 != 0)
    throw std::invalid_argument("qsa_attn_fp8: invalid query stride");
  if (n_split > 1 && (!m_ws || !l_ws || !c_ws))
    throw std::invalid_argument("qsa_attn_fp8: split attention needs workspace");
  const dim3 grid(rows, kv_heads * n_split, (local_heads / kv_heads + 15) / 16);
  attention<<<grid, 32, 2 * kPlaneBytes, stream>>>(q, q_row_stride, k, v, k_scale, v_scale, req_ids,
      topk, topk_stride, counts, local_heads, kv_heads, block_tokens, block_tables,
      blocks_per_request, scale, n_split, m_ws, l_ws, c_ws, out);
  DGPP_CUDA_OK(cudaGetLastError());
  if (n_split > 1)
    dsa_attn_combine(m_ws, l_ws, c_ws, rows, n_split, local_heads, kDim, out, stream);
}
}  // namespace dgpp
