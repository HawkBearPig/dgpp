// The dense tensor-core GEMM over BF16 weights (dense_mma_bf16w.hpp).
#include "kernels/dense_mma_bf16w.hpp"
#include "kernels/pdl.cuh"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {

namespace bf16w {
constexpr int BM = 64, BN = 128, BK = 32;
constexpr int kThreads = 256, kStages = 3;
constexpr int A_PAD = BK + 8;        // elements: 80-byte rows, ldmatrix conflict-free
constexpr int kWStride = 80;         // bytes per weight row slice (64 used): the fragment loads conflict-free
constexpr size_t kABytes = size_t(BM) * A_PAD * 2;     // 5,120
constexpr size_t kWBytes = size_t(BN) * kWStride;      // 10,240
constexpr size_t kSlotBytes = kABytes + kWBytes;       // 15,360
constexpr size_t kSmem = kStages * kSlotBytes;         // 46,080: the default budget, two blocks per SM
static_assert(BM * (BK / 8) == kThreads, "one activation chunk per thread");
static_assert(BN * (BK / 16) == kThreads, "two 16-byte weight copies per thread");
static_assert(kSlotBytes % 16 == 0 && kSmem <= 48 * 1024, "16-byte aligned, inside the default shared memory");
}  // namespace bf16w

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem, int src_bytes) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void store_out(float* o, float v) { *o = v; }
__device__ __forceinline__ void store_out(uint16_t* o, float v) { *o = float_to_bf16_bits(v); }

// One [BM x BN] tile: thread tid stages activation row tid / 4, elements
// (tid % 4) * 8 .. + 8 of the stage, and weight row tid / 2, bytes
// (tid % 2) * 32 .. + 32 of the stage's 64. a_src: this thread's activation
// row (k elements; any readable address when !a_ok); out: the tile's row 0.
// A row's chain is its own lanes': it does not depend on the rows beside it.
template <typename OutT>
__device__ __forceinline__ void run_tile(uint8_t* smem, const uint16_t* a_src, bool a_ok, int act_vec,
                                         const uint16_t* __restrict__ w, OutT* __restrict__ out, size_t out_stride,
                                         int m_rows, int n0, int n, int k) {
  using namespace bf16w;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32, lane = tid % 32;
  const int wn = warp;  // 1 (m64) x 8 (n16)
  const int r = lane / 4, cc = (lane % 4) * 2;
  const int a_row = tid / 4, a_kq = (tid % 4) * 8;
  const int b_row = tid / 2, b_half = tid % 2;
  const int gn = n0 + b_row;
  const bool b_ok = gn < n;
  const uint8_t* b_src = reinterpret_cast<const uint8_t*>(w + static_cast<size_t>(b_ok ? gn : 0) * k);
  const int stages = k / BK;
  const int live_slabs = (m_rows + 15) / 16;

  auto slotA = [&](int slot) { return reinterpret_cast<uint16_t*>(smem + static_cast<size_t>(slot) * kSlotBytes); };
  auto slotW = [&](int slot) { return smem + static_cast<size_t>(slot) * kSlotBytes + kABytes; };
  auto issue = [&](int s, int slot) {
    const int k0 = s * BK;
    {
      uint16_t* dst = slotA(slot) + static_cast<size_t>(a_row) * A_PAD + a_kq;
      const uint16_t* src = a_ok ? a_src + k0 + a_kq : a_src;
      if (act_vec != 0) {
        cp_async_16(dst, src, a_ok ? 16 : 0);
      } else {
        uint16_t e[8];
#pragma unroll
        for (int h = 0; h < 8; ++h) e[h] = a_ok ? src[h] : static_cast<uint16_t>(0);
        *reinterpret_cast<uint4*>(dst) = make_uint4(e[0] | (e[1] << 16), e[2] | (e[3] << 16),
                                                    e[4] | (e[5] << 16), e[6] | (e[7] << 16));
      }
    }
    uint8_t* wd = slotW(slot) + static_cast<size_t>(b_row) * kWStride + b_half * 32;
    const uint8_t* ws = b_ok ? b_src + static_cast<size_t>(k0) * 2 + b_half * 32 : reinterpret_cast<const uint8_t*>(w);
    cp_async_16(wd, ws, b_ok ? 16 : 0);
    cp_async_16(wd + 16, ws + (b_ok ? 16 : 0), b_ok ? 16 : 0);
  };

  float acc[4][2][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < stages) issue(s, s);
    cp_commit();
  }
  for (int s = 0; s < stages; ++s) {
    const int slot = s % kStages;
    cp_wait<kStages - 2>();
    __syncthreads();
    if (s + kStages - 1 < stages) issue(s + kStages - 1, (s + kStages - 1) % kStages);
    cp_commit();
    const uint16_t* a = slotA(slot);
    const uint8_t* wt = slotW(slot);
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t bfrag[2][2];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int brow = wn * 16 + j * 8 + r;
        const uint8_t* row = wt + static_cast<size_t>(brow) * kWStride + kk * 2;
        bfrag[j][0] = *reinterpret_cast<const uint32_t*>(row + cc * 2);        // k = cc, cc + 1
        bfrag[j][1] = *reinterpret_cast<const uint32_t*>(row + (cc + 8) * 2);  // k = cc + 8, cc + 9
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (i >= live_slabs) break;
        uint32_t af[4];
        ldmatrix_x4(af, a + static_cast<size_t>(i * 16 + (lane % 16)) * A_PAD + kk + (lane / 16) * 8);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
              : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
              : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bfrag[j][0]), "r"(bfrag[j][1]));
        }
      }
    }
  }
  cp_wait<0>();

  // Epilogue: the warp's [64 x 16] slice.
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row_lo = i * 16 + r;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int gcol = n0 + wn * 16 + j * 8 + cc;
      auto st = [&](int row_off, int col_off, float val) {
        const int mm = row_lo + row_off;
        if (mm < m_rows && gcol + col_off < n) store_out(out + static_cast<size_t>(mm) * out_stride + gcol + col_off, val);
      };
      st(0, 0, acc[i][j][0]);
      st(0, 1, acc[i][j][1]);
      st(8, 0, acc[i][j][2]);
      st(8, 1, acc[i][j][3]);
    }
  }
}

// Block (x): m-tile x % m_tiles of n-tile x / m_tiles.
template <typename OutT>
__global__ __launch_bounds__(bf16w::kThreads, 2) void dense_mma_bf16w_kernel(
    const uint16_t* __restrict__ act, size_t act_stride, int act_vec, const uint16_t* __restrict__ w,
    OutT* __restrict__ out, size_t out_stride, int m, int n, int k, int m_tiles) {
  dgpp_pdl_wait();
  using namespace bf16w;
  extern __shared__ __align__(16) uint8_t smem[];
  const int m_tile = static_cast<int>(blockIdx.x) % m_tiles;
  const int n_tile = static_cast<int>(blockIdx.x) / m_tiles;
  const int z0 = m_tile * BM;
  if (z0 >= m) return;  // no barrier yet
  const int m_rows = min(BM, m - z0);
  const int a_row = static_cast<int>(threadIdx.x) / 4;
  const bool a_ok = a_row < m_rows;
  const uint16_t* a_src = a_ok ? act + static_cast<size_t>(z0 + a_row) * act_stride : act;
  run_tile<OutT>(smem, a_src, a_ok, act_vec, w, out + static_cast<size_t>(z0) * out_stride, out_stride, m_rows,
                 n_tile * BN, n, k);
}

// The gated ring form (launch_dense_mma_bf16w_groups_f32): block x serves
// batch row x / tiles; a row whose position does not complete a group
// returns at once. The group's rows come off the request's positional ring.
__global__ __launch_bounds__(bf16w::kThreads, 2) void dense_mma_bf16w_groups_kernel(
    const uint16_t* __restrict__ ring, int slots, const int32_t* __restrict__ req_ids,
    const int64_t* __restrict__ pos, int group, const uint16_t* __restrict__ w, float* __restrict__ out, int n,
    int k, int m_tiles, int tiles) {
  dgpp_pdl_wait();
  using namespace bf16w;
  extern __shared__ __align__(16) uint8_t smem[];
  const int row = static_cast<int>(blockIdx.x) / tiles, t = static_cast<int>(blockIdx.x) % tiles;
  const int64_t p = pos[row];
  if (p < 0 || (p + 1) % group != 0) return;  // no barrier yet
  const int m_tile = t % m_tiles, n_tile = t / m_tiles;
  const int z0 = m_tile * BM;
  const int m_rows = min(BM, group - z0);
  const int a_row = static_cast<int>(threadIdx.x) / 4;
  const bool a_ok = a_row < m_rows;
  const int64_t first = p - group + 1 + z0 + (a_ok ? a_row : 0);
  const uint16_t* a_src =
      ring + (static_cast<size_t>(req_ids[row]) * slots + static_cast<size_t>(first % slots)) * static_cast<size_t>(k);
  run_tile<float>(smem, a_src, a_ok, 1, w, out + (static_cast<size_t>(row) * group + z0) * static_cast<size_t>(n),
                  static_cast<size_t>(n), m_rows, n_tile * BN, n, k);
}

template <typename OutT>
void launch(const uint16_t* act, size_t act_stride, const uint16_t* w, OutT* out, int m, int n, int k,
            cudaStream_t stream, size_t out_stride) {
  using namespace bf16w;
  if (m <= 0 || n <= 0) return;
  if (!act || !w || !out) throw std::invalid_argument("dense_mma_bf16w: null pointer");
  if (!dense_mma_bf16w_shape_ok(w, k)) throw std::invalid_argument("dense_mma_bf16w: k a multiple of 32, aligned weights");
  if (out_stride == 0) out_stride = static_cast<size_t>(n);
  if (out_stride < static_cast<size_t>(n)) throw std::invalid_argument("dense_mma_bf16w: output row stride narrower than n");
  const int act_vec = (reinterpret_cast<uintptr_t>(act) % 16 == 0 && (act_stride % 8) == 0) ? 1 : 0;
  const int m_tiles = (m + BM - 1) / BM;
  const unsigned n_tiles = static_cast<unsigned>((n + BN - 1) / BN);
  dense_mma_bf16w_kernel<OutT><<<n_tiles * static_cast<unsigned>(m_tiles), kThreads, kSmem, stream>>>(
      act, act_stride, act_vec, w, out, out_stride, m, n, k, m_tiles);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace

bool dense_mma_bf16w_shape_ok(const void* w, int k) {
  return k > 0 && (k % bf16w::BK) == 0 && (reinterpret_cast<uintptr_t>(w) & 15u) == 0;
}

void launch_dense_mma_bf16w_f32(const uint16_t* act, size_t act_stride, const uint16_t* w, float* out, int m,
                                int n, int k, cudaStream_t stream, size_t out_stride) {
  launch<float>(act, act_stride, w, out, m, n, k, stream, out_stride);
}
void launch_dense_mma_bf16w_bf16(const uint16_t* act, size_t act_stride, const uint16_t* w, uint16_t* out, int m,
                                 int n, int k, cudaStream_t stream, size_t out_stride) {
  launch<uint16_t>(act, act_stride, w, out, m, n, k, stream, out_stride);
}

void launch_dense_mma_bf16w_groups_f32(const uint16_t* ring, int slots, const int32_t* req_ids, const int64_t* pos,
                                       int rows, int group, const uint16_t* w, float* out, int n, int k,
                                       cudaStream_t stream) {
  using namespace bf16w;
  if (rows <= 0) return;
  if (!ring || !req_ids || !pos || !w || !out) throw std::invalid_argument("dense_mma_bf16w groups: null pointer");
  if (group <= 0 || slots < group || n <= 0) throw std::invalid_argument("dense_mma_bf16w groups: group / slots / n");
  if (!dense_mma_bf16w_shape_ok(w, k) || (reinterpret_cast<uintptr_t>(ring) & 15u) != 0 || (k % 8) != 0)
    throw std::invalid_argument("dense_mma_bf16w groups: k a multiple of 32, aligned weights and ring");
  const int m_tiles = (group + BM - 1) / BM;
  const int tiles = m_tiles * ((n + BN - 1) / BN);
  dense_mma_bf16w_groups_kernel<<<static_cast<unsigned>(rows) * static_cast<unsigned>(tiles), kThreads, kSmem, stream>>>(
      ring, slots, req_ids, pos, group, w, out, n, k, m_tiles, tiles);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
