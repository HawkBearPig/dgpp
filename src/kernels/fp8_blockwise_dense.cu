// Native blockwise-FP8 dense GEMM for GB10 (sm_121a). See the header for the
// contract. Tile: one 128x128 output block per 256-thread CTA (8 warps);
// warp w owns rows [w*16, w*16+16) x all 128 columns (16 m16n8k32
// fragments, 64 fp32 accumulators per thread). Per 128-deep k-block: stage
// the BF16 activation tile, block-reduce its absmax (one sfa per block),
// quantize to E4M3 in smem, stage the weight tile transposed (conflict-free
// B-fragment reads), run 4 k32 slices x 16 n-fragments of mma.sync, scale
// the block product by sfa*sfb into the running accumulators.
#include "kernels/fp8_blockwise_dense.hpp"

#include <cstdlib>
#include <type_traits>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
#ifdef FP8BW_DEBUG
__device__ float fp8bw_dbg[16];  // TU scope (not anonymous): readable via cudaMemcpyFromSymbol
#endif
namespace {

constexpr int BW_BM = 128;
constexpr int BW_BN = 128;
constexpr int BW_BK = 128;
constexpr int BW_THREADS = 256;
// Padded smem pitch (bytes): 136 % 4 == 0 keeps the u32 A-fragment loads
// aligned, and distinct rows land in distinct banks (row*34 mod 32 is a
// permutation of the 8 rows a lane group touches).
constexpr int BW_PITCH = 136;

template <typename OutT>
__global__ void fp8bw_kernel(const uint16_t* __restrict__ act, size_t act_stride,
                             const uint8_t* __restrict__ w, const float* __restrict__ scales,
                             OutT* __restrict__ out, size_t out_stride, int m, int n, int k) {
  const int n0 = blockIdx.x * BW_BN;
  const int m0 = blockIdx.y * BW_BM;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int r = lane / 4;      // 0..7
  const int q = (lane % 4) * 4;  // this lane's k-quad base within a 16-half
  const int Kb = k / BW_BK;
#ifdef FP8BW_DEBUG
  if (tid == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    printf("[fp8bw] enter m=%d n=%d k=%d Kb=%d m0=%d n0=%d\n", m, n, k, Kb, m0, n0);
#endif

  __shared__ alignas(16) uint16_t sAb[BW_BM][BW_BK];
  __shared__ alignas(16) uint8_t sA[BW_BM][BW_PITCH];
  __shared__ alignas(16) uint8_t sB[BW_BK][BW_PITCH];  // transposed: [k][n]
  __shared__ float sRed[BW_THREADS];

  float acc[16][4] = {};

  for (int kb = 0, k0 = 0; kb < Kb; ++kb, k0 += BW_BK) {
#ifdef FP8BW_STUB
    for (int i = tid; i < BW_BM * BW_PITCH; i += BW_THREADS) sA[0][i] = 0x38;
    for (int i = tid; i < BW_BK * BW_PITCH; i += BW_THREADS) sB[0][i] = 0x38;
    __syncthreads();
    const float sfa = 1.f / 448.f;
    const float sfb = 1.f;
#else
    // 1. Stage the BF16 activation tile (8 float4 loads per thread).
    for (int i = 0; i < 8; ++i) {
      const int idx = tid * 64 + i * 8;
      const int mm = idx >> 7, kk = idx & 127;
      const int gm = m0 + mm;
      float4 v{0.f, 0.f, 0.f, 0.f};
      if (gm < m) v = reinterpret_cast<const float4*>(act + (size_t)gm * act_stride + k0)[kk >> 3];
      reinterpret_cast<float4*>(&sAb[mm][0])[kk >> 3] = v;
    }
    __syncthreads();
    // 2. Block absmax over the 128x128 tile, reduced by thread 0.
    float lmax = 0.f;
    for (int j = 0; j < 64; ++j) {
      const int idx = tid * 64 + j;
      const float x = bf16_bits_to_float(sAb[idx >> 7][idx & 127]);
      const float ax = fabsf(x);
      lmax = fmaxf(lmax, ax);
    }
    sRed[tid] = lmax;
    __syncthreads();
    if (tid == 0) {
      float gmax = 0.f;
      for (int i = 0; i < BW_THREADS; ++i) gmax = fmaxf(gmax, sRed[i]);
      sRed[0] = gmax;
    }
    __syncthreads();
    const float gmax = sRed[0];
    const float inv = gmax > 0.f ? 448.f / gmax : 0.f;
    const float sfa = gmax > 0.f ? gmax / 448.f : 1.f;
    // 3. Quantize the tile to E4M3 (the clamp keeps inf finite for the
    // encoder; NaN passes through to the canonical NaN code).
    for (int j = 0; j < 64; ++j) {
      const int idx = tid * 64 + j;
      const int mm = idx >> 7, kk = idx & 127;
#ifdef FP8BW_CONSTQ
      (void)inv;
      sA[mm][kk] = 0x38;
#else
      float v = bf16_bits_to_float(sAb[mm][kk]) * inv;
      v = (v > 448.f) ? 448.f : ((v < -448.f) ? -448.f : v);
      sA[mm][kk] = float_to_fp8_e4m3_bits(v);
#endif
    }
    // 4. Stage the weight tile transposed (4 uint4 loads per thread).
    for (int i = 0; i < 4; ++i) {
      const int idx = tid * 64 + i * 16;
      const int nn = idx >> 7, kk = idx & 127;
      uint4 v{0, 0, 0, 0};
      if (n0 + nn < n)
        v = reinterpret_cast<const uint4*>(w + (size_t)(n0 + nn) * k + k0)[kk >> 4];
      const uint8_t* pb = reinterpret_cast<const uint8_t*>(&v);
#pragma unroll
      for (int j = 0; j < 16; ++j) sB[kk + j][nn] = pb[j];
    }
    const float sfb = scales[(size_t)(n0 >> 7) * Kb + kb];
    __syncthreads();
#endif
    // 5. Four k32 slices x 16 n-fragments of mma.sync; the block product
    // is scaled by sfa*sfb into the running accumulators.
    const float sc = sfa * sfb;
    const int arow = warp * 16;
#ifdef FP8BW_DEBUG
    if (tid == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
      fp8bw_dbg[0] = gmax; fp8bw_dbg[1] = sfa; fp8bw_dbg[2] = sfb; fp8bw_dbg[3] = sc;
      fp8bw_dbg[4] = bf16_bits_to_float(sAb[0][0]);
      fp8bw_dbg[5] = (float)sA[0][0];
      fp8bw_dbg[6] = (float)sB[0][0];
    }
#endif
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sA[arow + r][s * 32 + q]);
      const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sA[arow + r + 8][s * 32 + q]);
      const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sA[arow + r][s * 32 + q + 16]);
      const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&sA[arow + r + 8][s * 32 + q + 16]);
#pragma unroll
      for (int nf = 0; nf < 16; ++nf) {
        const int ncol = nf * 8 + r;
        const int kq = s * 32 + q;
        const uint32_t b0 = static_cast<uint32_t>(sB[kq][ncol]) |
                            (static_cast<uint32_t>(sB[kq + 1][ncol]) << 8) |
                            (static_cast<uint32_t>(sB[kq + 2][ncol]) << 16) |
                            (static_cast<uint32_t>(sB[kq + 3][ncol]) << 24);
        const uint32_t b1 = static_cast<uint32_t>(sB[kq + 16][ncol]) |
                            (static_cast<uint32_t>(sB[kq + 17][ncol]) << 8) |
                            (static_cast<uint32_t>(sB[kq + 18][ncol]) << 16) |
                            (static_cast<uint32_t>(sB[kq + 19][ncol]) << 24);
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
#ifdef FP8BW_NOASM
        d0 = (float)(a0 & 0xffu);
        (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
#else
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
            : "memory");
#endif
        acc[nf][0] += d0 * sc;
        acc[nf][1] += d1 * sc;
        acc[nf][2] += d2 * sc;
        acc[nf][3] += d3 * sc;
#ifdef FP8BW_DEBUG
        if (s == 0 && nf == 0 && lane == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
          fp8bw_dbg[7] = d0;
          fp8bw_dbg[8] = (float)(a0 & 0xffu);
          fp8bw_dbg[9] = (float)(b0 & 0xffu);
          fp8bw_dbg[10] = acc[0][0];
        }
#endif
      }
    }
    __syncthreads();  // tile reads done before the next block overwrites
  }

  // Epilogue: warp w holds rows [w*16, w*16+16) x the 128 columns, fragment
  // nf at columns [nf*8, nf*8+8), thread columns {2cq, 2cq+1}.
  const int arow = warp * 16;
  const int cc = (lane % 4) * 2;
#pragma unroll
  for (int nf = 0; nf < 16; ++nf) {
    const int gn = n0 + nf * 8 + cc;
    const int gm0 = m0 + arow + r;
    const int gm1 = gm0 + 8;
    if constexpr (std::is_same_v<OutT, float>) {
      if (gm0 < m && gn < n) out[(size_t)gm0 * out_stride + gn] = acc[nf][0];
      if (gm0 < m && gn + 1 < n) out[(size_t)gm0 * out_stride + gn + 1] = acc[nf][1];
      if (gm1 < m && gn < n) out[(size_t)gm1 * out_stride + gn] = acc[nf][2];
      if (gm1 < m && gn + 1 < n) out[(size_t)gm1 * out_stride + gn + 1] = acc[nf][3];
    } else {
      if (gm0 < m && gn < n)
        out[(size_t)gm0 * out_stride + gn] = float_to_bf16_bits(acc[nf][0]);
      if (gm0 < m && gn + 1 < n)
        out[(size_t)gm0 * out_stride + gn + 1] = float_to_bf16_bits(acc[nf][1]);
      if (gm1 < m && gn < n)
        out[(size_t)gm1 * out_stride + gn] = float_to_bf16_bits(acc[nf][2]);
      if (gm1 < m && gn + 1 < n)
        out[(size_t)gm1 * out_stride + gn + 1] = float_to_bf16_bits(acc[nf][3]);
    }
  }
}

template <typename OutT>
void launch_fp8bw(OutT* out, const uint16_t* act, size_t act_stride, const uint8_t* w,
                  const float* scales, int m, int n, int k, cudaStream_t stream,
                  size_t out_stride) {
  if (!fp8_blockwise_dense_supported(act, act_stride, w, m, n, k))
    throw std::invalid_argument("fp8_blockwise_dense: shape/pointer contract violated");
  if (!out || !scales) throw std::invalid_argument("fp8_blockwise_dense: null pointer");
  if (out_stride == 0) out_stride = static_cast<size_t>(n);
  if (out_stride < static_cast<size_t>(n))
    throw std::invalid_argument("fp8_blockwise_dense: output row stride narrower than n");
  const dim3 grid((static_cast<unsigned>(n) + BW_BN - 1) / BW_BN,
                  (static_cast<unsigned>(m) + BW_BM - 1) / BW_BM);
  fp8bw_kernel<OutT><<<grid, BW_THREADS, 0, stream>>>(act, act_stride, w, scales, out, out_stride,
                                                     m, n, k);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace

bool fp8_blockwise_dense_supported(const uint16_t* act, size_t act_row_stride_elems,
                                   const uint8_t* w_payload, int m, int n, int k) {
  if (m <= 0 || n <= 0 || k <= 0 || (k % BW_BK) != 0) return false;
  if (!act || !w_payload) return false;
  if (act_row_stride_elems < static_cast<size_t>(k)) return false;
  if ((reinterpret_cast<uintptr_t>(act) % 16) != 0) return false;
  if ((act_row_stride_elems % 8) != 0) return false;
  if ((reinterpret_cast<uintptr_t>(w_payload) % 16) != 0) return false;
  return true;
}

bool fp8_blockwise_dense_enabled() {
  const char* e = std::getenv("DGPP_FP8BW_DENSE");
  return e != nullptr && e[0] != '0';
}

void launch_fp8_blockwise_bf16(const uint16_t* act, size_t act_row_stride_elems,
                               const uint8_t* w_payload, const float* w_scales,
                               uint16_t* out, int m, int n, int k, cudaStream_t stream,
                               size_t out_row_stride_elems) {
  launch_fp8bw<uint16_t>(out, act, act_row_stride_elems, w_payload, w_scales, m, n, k, stream,
                         out_row_stride_elems);
}

void launch_fp8_blockwise_f32(const uint16_t* act, size_t act_row_stride_elems,
                               const uint8_t* w_payload, const float* w_scales, float* out,
                               int m, int n, int k, cudaStream_t stream,
                               size_t out_row_stride_elems) {
  launch_fp8bw<float>(out, act, act_row_stride_elems, w_payload, w_scales, m, n, k, stream,
                      out_row_stride_elems);
}

// --- Per-tensor FP8 recipe (DGPP_FP8_PT_DENSE) --------------------------------
// Multi-CTA absmax: each CTA grid-strides over float4 groups (8 BF16 per
// coalesced 16B load; K is a multiple of 8 so groups tile the count, with a
// scalar tail for safety), tree-reduces in smem, then one atomicMax per CTA
// to *out_max. Values are finite non-negative by construction (inf folds to
// 448, NaN compares false and is ignored — same policy as the old single-CTA
// kernel), so integer atomicMax preserves float order. The caller zeroes
// *out_max on-stream first; max is order-independent, hence deterministic.
__global__ void fp8_maxabs_kernel(const uint16_t* __restrict__ data, size_t count,
                                  float* __restrict__ out_max) {
  __shared__ float sRed[BW_THREADS];
  const int tid = threadIdx.x;
  const size_t n8 = count / 8;
  const size_t span = (size_t)gridDim.x * BW_THREADS;
  const size_t base = (size_t)blockIdx.x * BW_THREADS + tid;
  float mx = 0.f;
  const float4* vec = reinterpret_cast<const float4*>(data);
  for (size_t g = base; g < n8; g += span) {
    const float4 v = vec[g];
    const uint32_t* w = reinterpret_cast<const uint32_t*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t pair = w[j];
      float ax0 = fabsf(bf16_bits_to_float((uint16_t)(pair & 0xFFFFu)));
      float ax1 = fabsf(bf16_bits_to_float((uint16_t)(pair >> 16)));
      if (ax0 > 448.f) ax0 = 448.f;  // inf folds; NaN fails compare, ignored
      if (ax1 > 448.f) ax1 = 448.f;
      if (ax0 > mx) mx = ax0;
      if (ax1 > mx) mx = ax1;
    }
  }
  for (size_t j = n8 * 8 + base; j < count; j += span) {
    float ax = fabsf(bf16_bits_to_float(data[j]));
    if (ax > 448.f) ax = 448.f;
    if (ax > mx) mx = ax;
  }
  sRed[tid] = mx;
  __syncthreads();
  for (int s = BW_THREADS / 2; s > 0; s >>= 1) {
    if (tid < s && sRed[tid + s] > sRed[tid]) sRed[tid] = sRed[tid + s];
    __syncthreads();
  }
  if (tid == 0 && sRed[0] > 0.f)
    atomicMax(reinterpret_cast<int*>(out_max), static_cast<int>(__float_as_int(sRed[0])));
}

// Elementwise x/mx quantize (`dev_max` the same-stream maxabs output, read
// once per thread and broadcast-cached): the codes span [-1, 1], so the Lt
// scalar scales are the raw maxabs values. (An x/448 map would need
// mx/448 scales; the relative quant error is identical either way — E4M3's
// precision is relative — but raw-mx scales keep the contract obvious.)
__global__ void fp8_quant_kernel(const uint16_t* __restrict__ in, uint8_t* __restrict__ out,
                                  size_t count, const float* __restrict__ dev_max) {
  float mx = *dev_max;
  if (mx == 0.f) mx = 1.f;  // all-zero guard (matches the old maxabs policy)
  const float inv = 1.f / mx;
  const size_t base = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = base; i < count; i += stride)
    out[i] = float_to_fp8_e4m3_bits(bf16_bits_to_float(in[i]) * inv);
}

bool fp8_per_tensor_enabled() {
  const char* e = std::getenv("DGPP_FP8_PT_DENSE");
  return !(e != nullptr && e[0] == '0');
}

bool fp8_pt_attn_enabled() {
  const char* e = std::getenv("DGPP_FP8_PT_ATTN");
  return !(e != nullptr && e[0] == '0');
}

// Blockwise-FP8 lm head (env gate DGPP_FP8_HEAD, default on): the 248k-row
// BF16 head is the only multi-GB BF16 weight left on the decode path
// (10.7 ms/row at 254 GB/s on LPDDR5X parts — the read, not the kernel).
// Requantized once at boot to E4M3 + 128x128 scales; decode rows read half
// the bytes through the existing F32 scale-GEMM path.
bool fp8_head_enabled() {
  const char* e = std::getenv("DGPP_FP8_HEAD");
  return !(e != nullptr && e[0] == '0');
}

void launch_fp8_row_maxabs(const uint16_t* data, size_t count, float* out_max,
                           cudaStream_t stream) {
  if (!data || !out_max || count == 0) throw std::invalid_argument("fp8 maxabs: bad args");
  if ((reinterpret_cast<uintptr_t>(data) % 16) != 0)
    throw std::invalid_argument("fp8 maxabs: data not 16B aligned for float4 loads");
  DGPP_CUDA_OK(cudaMemsetAsync(out_max, 0, sizeof(float), stream));  // atomicMax baseline
  const size_t n8 = count / 8;
  size_t blocks = (n8 + BW_THREADS - 1) / BW_THREADS;
  if (blocks > 1024) blocks = 1024;
  if (blocks == 0) blocks = 1;
  fp8_maxabs_kernel<<<static_cast<unsigned>(blocks), BW_THREADS, 0, stream>>>(data, count,
                                                                              out_max);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_fp8_quant_bf16(const uint16_t* in, uint8_t* out, size_t count,
                           const float* dev_max, cudaStream_t stream) {
  if (!in || !out || !dev_max || count == 0)
    throw std::invalid_argument("fp8 quant: bad args");
  const size_t blocks = (count + BW_THREADS - 1) / BW_THREADS;
  fp8_quant_kernel<<<(unsigned)(blocks > 1024 ? 1024 : blocks), BW_THREADS, 0, stream>>>(
      in, out, count, dev_max);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
