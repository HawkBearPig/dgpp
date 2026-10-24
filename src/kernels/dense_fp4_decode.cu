#include <stdexcept>
#include "kernels/pdl.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/glm_moe_launch.hpp"

namespace dgpp {
namespace {
// Small-row dense products benefit from loading 64 payload bytes per weight row
// and staging only the 16 activation rows that an MMA can consume. Two
// 46-KiB blocks fit per GB10 SM. The paired launch exposes gate and up to
// one grid, filling SMs left idle by either projection alone at TP4.
constexpr int BM = 16, BN = 128, BK = 128, kThreads = 256, kStages = 3;
constexpr int A_PAD = BK + 8, kRawStride = BK / 2 + 16;
constexpr int kGroupsPerRow = BK / kFp4Group, kGroupsPerStage = kGroupsPerRow;
constexpr size_t kABytes = size_t(BM) * A_PAD * 2;
constexpr size_t kCodeBytes = size_t(BN) * kRawStride;
constexpr size_t kSlotBytes = kABytes + kCodeBytes + size_t(BN) * kGroupsPerRow;
constexpr size_t kSmem = kStages * kSlotBytes;

__device__ __forceinline__ void cp_async(void* dst, const void* src, int bytes, int valid) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  if (bytes == 16)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(src), "r"(valid));
  else
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::"r"(d), "l"(src), "r"(valid));
}
__device__ __forceinline__ void commit() {
  asm volatile("cp.async.commit_group;");
}
template <int N>
__device__ __forceinline__ void wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* ptr) {
  const unsigned p = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(p));
}
__device__ __forceinline__ uint32_t decode_pair_bf16(uint32_t byte, __half2 scale) {
  const __half2 h(__nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(byte), __NV_E2M1));
  const __half2 p = __hmul2(h, scale);
  const __nv_bfloat162 b = __floats2bfloat162_rn(__low2float(p), __high2float(p));
  return *reinterpret_cast<const uint32_t*>(&b);
}
__device__ __forceinline__ void store_dot(uint16_t* out, float value) {
  *out = float_to_bf16_bits(value);
}

// Decode remains exact: FP4 * E4M3 has at most six significant bits,
// every output follows the reference's ascending-k16 FP32 MMA chain,
// and the global scale divides the completed dot exactly once.
template <bool Pair>
__global__ __launch_bounds__(kThreads, 2) void dense_fp4_decode_kernel(
    const uint16_t* __restrict__ act, size_t act_stride, int act_vec,
    const int32_t* __restrict__ act_rows, const MoeSegment* __restrict__ segs,
    const MoeExpertView* __restrict__ views, int which, uint16_t* __restrict__ out,
    uint16_t* __restrict__ up_out, size_t out_stride, int n, int k, int m_tiles) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint8_t smem[];
  const MoeSegment seg = segs[0];
  if constexpr (Pair) {
    if (blockIdx.y) out = up_out;
  }
  const int m_tile = static_cast<int>(blockIdx.x) % m_tiles;
  const int n_tile = static_cast<int>(blockIdx.x) / m_tiles;
  const int z0 = m_tile * BM;
  if (z0 >= seg.rows) return;
  const int z1 = min(seg.rows, z0 + BM);
  const MoeExpertView v = views[seg.expert * 3 + which + (Pair ? blockIdx.y : 0)];
  const float g = *v.fp4_global;
  const int n0 = n_tile * BN;
  const size_t payload_stride = static_cast<size_t>(k) / 2;
  const size_t scale_stride = static_cast<size_t>(k) / kFp4Group;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32, lane = tid % 32;
  const int wn = warp;
  const int r = lane / 4, cc = (lane % 4) * 2;
  const uint32_t byte_sel = static_cast<uint32_t>(lane % 4);
  const int s_row = tid;
  const bool s_ok = tid < BN && n0 + tid < n;
  const uint8_t* s_scales = v.fp4_scales + static_cast<size_t>(s_ok ? n0 + tid : 0) * scale_stride;
  const int stages = (k + BK - 1) / BK;
  const bool codes16 = (reinterpret_cast<uintptr_t>(v.payload) % 16) == 0;
  const bool scales_wide = (reinterpret_cast<uintptr_t>(v.fp4_scales) % 8) == 0;

  auto slotA = [&](int slot) {
    return reinterpret_cast<uint16_t*>(smem + static_cast<size_t>(slot) * kSlotBytes);
  };
  auto slotCodes = [&](int slot) {
    return smem + static_cast<size_t>(slot) * kSlotBytes + kABytes;
  };
  auto slotScales = [&](int slot) {
    return smem + static_cast<size_t>(slot) * kSlotBytes + kABytes + kCodeBytes;
  };

  for (int m0 = z0; m0 < z1; m0 += BM) {
    const int m_rows = min(BM, z1 - m0);
    const int live_slabs = (m_rows + 15) / 16;
    float acc[BM / 16][2][4];
#pragma unroll
    for (int i = 0; i < BM / 16; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    auto issue = [&](int s, int slot) {
      const int k0 = s * BK;
      uint16_t* a = slotA(slot);
#pragma unroll
      for (int i = 0; i < (BM * (BK / 8) + kThreads - 1) / kThreads; ++i) {
        const int ix = tid + i * kThreads;
        const int mm = ix / (BK / 8), kq = (ix % (BK / 8)) * 8;
        if (mm >= BM) continue;
        const int gk = k0 + kq;
        const bool in = mm < m_rows && gk < k;
        uint16_t* dst = a + static_cast<size_t>(mm) * A_PAD + kq;
        const uint16_t* src = act;
        if (in) {
          const int srow = seg.row0 + m0 + mm;
          src = act +
                static_cast<size_t>(act_rows != nullptr ? act_rows[srow] : srow) * act_stride + gk;
        }
        if (act_vec != 0) {
          if (in && kq % 64 == 0) asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(src));
          cp_async(dst, src, 16, in ? 16 : 0);
        } else {
          uint16_t e[8];
#pragma unroll
          for (int h = 0; h < 8; ++h) e[h] = in ? src[h] : static_cast<uint16_t>(0);
          *reinterpret_cast<uint4*>(dst) = make_uint4(e[0] | (e[1] << 16), e[2] | (e[3] << 16),
                                                      e[4] | (e[5] << 16), e[6] | (e[7] << 16));
        }
      }
#pragma unroll
      for (int bi = 0; bi < BK / 64; ++bi) {
        const int ix = tid + bi * kThreads;
        const int b_row = ix / (BK / 32), b_half = ix % (BK / 32);
        const int gn = n0 + b_row;
        const bool b_ok = gn < n;
        const uint8_t* b_codes = v.payload + static_cast<size_t>(b_ok ? gn : 0) * payload_stride;
        const int gk = k0 + b_half * 32;
        const int in_codes = b_ok ? max(0, min(32, k - gk)) : 0;
        uint8_t* cdst = slotCodes(slot) + static_cast<size_t>(b_row) * kRawStride + b_half * 16;
        if (codes16)
          cp_async(cdst, in_codes > 0 ? b_codes + gk / 2 : v.payload, 16, in_codes / 2);
        else {
#pragma unroll
          for (int q = 0; q < 2; ++q) {
            const int in_q = max(0, min(16, in_codes - 16 * q));
            cp_async(cdst + 8 * q, in_q > 0 ? b_codes + gk / 2 + 8 * q : v.payload, 8, in_q / 2);
          }
        }
        if (b_half == 0 && b_ok && (s % (256 / BK)) == 0) {
          const int nk = k0 + 256;
          if (nk < k) asm volatile("prefetch.global.L2 [%0];" ::"l"(b_codes + nk / 2));
        }
      }
      if (tid < BN) {
        const int in_groups = s_ok ? max(0, min(kGroupsPerStage, (k - k0) / kFp4Group)) : 0;
        uint8_t* sdst = slotScales(slot) + static_cast<size_t>(s_row) * kGroupsPerRow;
        if (scales_wide) {
          cp_async(sdst, in_groups > 0 ? s_scales + k0 / kFp4Group : v.fp4_scales, kGroupsPerRow,
                   in_groups);
        } else {
#pragma unroll
          for (int w = 0; w < kGroupsPerRow / 4; ++w) {
            uint32_t value = 0;
#pragma unroll
            for (int gi = 0; gi < 4; ++gi) {
              const int group = w * 4 + gi;
              if (group < in_groups)
                value |= static_cast<uint32_t>(s_scales[k0 / kFp4Group + group]) << (8 * gi);
            }
            reinterpret_cast<uint32_t*>(sdst)[w] = value;
          }
        }
        if (s_ok && (s % (128 / kGroupsPerStage)) == 0) {
          const int nk = k0 + (128 / kGroupsPerStage) * BK;
          if (nk < k) {
            const uint8_t* line = s_scales + nk / kFp4Group;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(line));
          }
        }
      }
    };

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
      if (s < stages) issue(s, s);
      commit();
    }
    for (int s = 0; s < stages; ++s) {
      const int slot = s % kStages;
      wait<kStages - 2>();
      __syncthreads();
      if (s + kStages - 1 < stages) issue(s + kStages - 1, (s + kStages - 1) % kStages);
      commit();
      const uint16_t* a = slotA(slot);
      const uint8_t* codes = slotCodes(slot);
      const uint8_t* scales = slotScales(slot);
      uint32_t sc4[2][BK / 64];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int bn = wn * 16 + j * 8 + r;
#pragma unroll
        for (int w = 0; w < BK / 64; ++w)
          sc4[j][w] = reinterpret_cast<const uint32_t*>(scales +
                                                        static_cast<size_t>(bn) * kGroupsPerRow)[w];
      }
#pragma unroll
      for (int kk = 0; kk < BK; kk += 16) {
        uint32_t bfrag[2][2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int bn = wn * 16 + j * 8 + r;
          const uint2 w = *reinterpret_cast<const uint2*>(
              codes + static_cast<size_t>(bn) * kRawStride + (kk / 16) * 8);
          const uint8_t sc = static_cast<uint8_t>(
              (sc4[j][(kk / kFp4Group) / 4] >> (8 * ((kk / kFp4Group) % 4))) & 0xFFu);
          const uint32_t byte0 = __byte_perm(w.x, 0u, byte_sel);
          const uint32_t byte1 = __byte_perm(w.y, 0u, byte_sel);
          {
            const __half2 s2 = __half2half2(
                __half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(sc), __NV_E4M3)));
            bfrag[j][0] = decode_pair_bf16(byte0, s2);
            bfrag[j][1] = decode_pair_bf16(byte1, s2);
          }
        }
#pragma unroll
        for (int i = 0; i < BM / 16; ++i) {
          if (i >= live_slabs) break;
          uint32_t af[4];
          const int arow = i * 16 + (lane % 16);
          ldmatrix_x4(af, a + static_cast<size_t>(arow) * A_PAD + kk + (lane / 16) * 8);
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
                : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bfrag[j][0]),
                  "r"(bfrag[j][1]));
          }
        }
      }
    }
    wait<0>();
    __syncthreads();

#pragma unroll
    for (int i = 0; i < BM / 16; ++i) {
      const int row_lo = i * 16 + r;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int gcol = n0 + wn * 16 + j * 8 + cc;
        auto st = [&](int row_off, int col_off, float val) {
          const int mm = row_lo + row_off;
          if (mm < m_rows && gcol + col_off < n)
            store_dot(out + static_cast<size_t>(seg.row0 + m0 + mm) * out_stride + gcol + col_off,
                      __fdiv_rn(val, g));
        };
        st(0, 0, acc[i][j][0]);
        st(0, 1, acc[i][j][1]);
        st(8, 0, acc[i][j][2]);
        st(8, 1, acc[i][j][3]);
      }
    }
  }
}

template <bool Pair>
void launch(const uint16_t* act, size_t stride, const MoeSegment* seg, const MoeExpertView* views,
            int which, uint16_t* out, uint16_t* up, int m, int n, int k, cudaStream_t stream) {
  if (!act || !seg || !views || !out || (Pair && !up))
    throw std::invalid_argument("dense fp4 decode: null pointer");
  static const bool initialized = [] {
    DGPP_CUDA_OK(cudaFuncSetAttribute(dense_fp4_decode_kernel<Pair>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    return true;
  }();
  (void)initialized;
  const int act_vec = reinterpret_cast<uintptr_t>(act) % 16 == 0 && stride % 8 == 0;
  const dim3 grid((n + BN - 1) / BN, Pair ? 2 : 1);
  dense_fp4_decode_kernel<Pair><<<grid, kThreads, kSmem, stream>>>(
      act, stride, act_vec, nullptr, seg, views, which, out, up, n, n, k, 1);
  DGPP_CUDA_OK(cudaGetLastError());
}
}  // namespace

bool dense_fp4_decode_shape(int m, int n, int k, int group) {
  return group == kFp4Group && m > 0 && m <= BM && n > 0 && n <= 17408 && k > 0 && k <= 17408 &&
         k % BK == 0;
}

bool try_dense_fp4_decode_bf16(const uint16_t* act, size_t stride, const MoeSegment* seg,
                               const MoeExpertView* views, int which, uint16_t* out, int m, int n,
                               int k, cudaStream_t stream, int group) {
  if (!dense_fp4_decode_shape(m, n, k, group)) return false;
  launch<false>(act, stride, seg, views, which, out, nullptr, m, n, k, stream);
  return true;
}

void launch_dense_mma_fp4_gate_up_bf16(const uint16_t* act, size_t stride, const MoeSegment* seg,
                                       const MoeExpertView* views, uint16_t* gate, uint16_t* up,
                                       int m, int n, int k, cudaStream_t stream, int group) {
  if (dense_fp4_decode_shape(m, n, k, group)) {
    launch<true>(act, stride, seg, views, 0, gate, up, m, n, k, stream);
  } else {
    launch_dense_mma_fp4_prod_bf16(act, stride, seg, views, 0, gate, m, n, k, stream, group);
    launch_dense_mma_fp4_prod_bf16(act, stride, seg, views, 1, up, m, n, k, stream, group);
  }
}
}  // namespace dgpp
