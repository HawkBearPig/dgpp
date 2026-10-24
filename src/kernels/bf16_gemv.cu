#include "kernels/bf16_gemv.hpp"
#include "kernels/pdl.cuh"

#include <algorithm>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/bf16_gemv.cuh"
#include "kernels/gemv_common.cuh"

namespace dgpp {
namespace {

template <int kRows, bool kOutF32>
__global__ void bf16_gemv_kernel(const uint16_t* __restrict__ act,
                                 size_t act_stride,
                                 const uint16_t* __restrict__ w,
                                 void* __restrict__ out, int n, int k) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint16_t sx[];
  gemv::stage_activations<kRows>(act, act_stride, k, sx);
  __syncthreads();
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int row = blockIdx.x * gemv::kWarps + warp;
  if (row >= n) return;
  float acc[kRows];
  bf16_gemv::row_dots<kRows>(w + static_cast<size_t>(row) * k, sx, k, lane, acc);
  if (lane != 0) return;
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const size_t at = static_cast<size_t>(r) * n + row;
    if (kOutF32)
      static_cast<float*>(out)[at] = acc[r];
    else
      static_cast<uint16_t*>(out)[at] = float_to_bf16_bits(acc[r]);
  }
}

// The single kernel's body over one of two problems, chosen by block
// range; the staged activations are the chosen problem's.
template <int kRows, bool kOutF32>
__global__ void bf16_gemv_dual_kernel(Bf16GemvProblem p0, Bf16GemvProblem p1,
                                      int blocks0, int k) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint16_t sx[];
  const bool second = static_cast<int>(blockIdx.x) >= blocks0;
  // Field-wise selects, not a reference into the parameter space: a
  // runtime-chosen reference to a kernel parameter struct forces the
  // parameters into local memory and every field read through it.
  const uint16_t* act = second ? p1.act : p0.act;
  const size_t act_stride = second ? p1.act_row_stride : p0.act_row_stride;
  const uint16_t* w = second ? p1.weight : p0.weight;
  void* out = second ? p1.out : p0.out;
  const int n = second ? p1.n : p0.n;
  const int block = second ? static_cast<int>(blockIdx.x) - blocks0
                           : static_cast<int>(blockIdx.x);
  gemv::stage_activations<kRows>(act, act_stride, k, sx);
  __syncthreads();
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int row = block * gemv::kWarps + warp;
  if (row >= n) return;
  float acc[kRows];
  bf16_gemv::row_dots<kRows>(w + static_cast<size_t>(row) * k, sx, k, lane, acc);
  if (lane != 0) return;
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const size_t at = static_cast<size_t>(r) * n + row;
    if (kOutF32)
      static_cast<float*>(out)[at] = acc[r];
    else
      static_cast<uint16_t*>(out)[at] = float_to_bf16_bits(acc[r]);
  }
}

// The multi-problem kernel: the problems in the parameter
// space with their block prefixes; a block finds its problem by the
// prefix table (field-wise selects, as the dual kernel: no runtime
// reference into the parameter struct).
struct Bf16GemvMulti {
  Bf16GemvProblem p[4];
  int block_end[4];  // exclusive prefix of blocks per problem
  int n;
};

template <int kRows, bool kOutF32>
__global__ void bf16_gemv_multi_kernel(Bf16GemvMulti mp, int k) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint16_t sx[];
  const int bid = static_cast<int>(blockIdx.x);
  int which = 0;
#pragma unroll
  for (int i = 0; i < 3; ++i) which += (i + 1 < mp.n && bid >= mp.block_end[i]) ? 1 : 0;
  const uint16_t* act = which == 0 ? mp.p[0].act : which == 1 ? mp.p[1].act : which == 2 ? mp.p[2].act : mp.p[3].act;
  const size_t act_stride = which == 0 ? mp.p[0].act_row_stride : which == 1 ? mp.p[1].act_row_stride : which == 2 ? mp.p[2].act_row_stride : mp.p[3].act_row_stride;
  const uint16_t* w = which == 0 ? mp.p[0].weight : which == 1 ? mp.p[1].weight : which == 2 ? mp.p[2].weight : mp.p[3].weight;
  void* out = which == 0 ? mp.p[0].out : which == 1 ? mp.p[1].out : which == 2 ? mp.p[2].out : mp.p[3].out;
  const int n = which == 0 ? mp.p[0].n : which == 1 ? mp.p[1].n : which == 2 ? mp.p[2].n : mp.p[3].n;
  const int block0 = which == 0 ? 0 : which == 1 ? mp.block_end[0] : which == 2 ? mp.block_end[1] : mp.block_end[2];
  const int block = bid - block0;
  // blockIdx.y: the kRows-row group (launch_bf16_gemv_multi_rows) — each
  // group's chain is exactly the kRows-row launch's, so a 64-row decode
  // batch is bitwise its rows' own 4-row launches.
  const size_t group_rows = static_cast<size_t>(blockIdx.y) * kRows;
  act += group_rows * act_stride;
  gemv::stage_activations<kRows>(act, act_stride, k, sx);
  __syncthreads();
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int row = block * gemv::kWarps + warp;
  if (row >= n) return;
  float acc[kRows];
  bf16_gemv::row_dots<kRows>(w + static_cast<size_t>(row) * k, sx, k, lane, acc);
  if (lane != 0) return;
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const size_t at = (group_rows + static_cast<size_t>(r)) * n + row;
    if (kOutF32)
      static_cast<float*>(out)[at] = acc[r];
    else
      static_cast<uint16_t*>(out)[at] = float_to_bf16_bits(acc[r]);
  }
}

template <int kRows>
void launch_multi_rows(const Bf16GemvMulti& mp, bool out_f32, int k, cudaStream_t stream, int groups = 1) {
  const dim3 grid(static_cast<unsigned>(mp.block_end[mp.n - 1]), static_cast<unsigned>(groups));
  const size_t smem = gemv::smem_bytes(kRows, k);
  if (out_f32)
    bf16_gemv_multi_kernel<kRows, true><<<grid, gemv::kThreads, smem, stream>>>(mp, k);
  else
    bf16_gemv_multi_kernel<kRows, false><<<grid, gemv::kThreads, smem, stream>>>(mp, k);
  DGPP_CUDA_OK(cudaGetLastError());
}

template <int kRows>
void launch_dual_rows(const Bf16GemvProblem& p0, const Bf16GemvProblem& p1,
                      bool out_f32, int k, cudaStream_t stream) {
  const int blocks0 = (p0.n + gemv::kWarps - 1) / gemv::kWarps;
  const int blocks1 = (p1.n + gemv::kWarps - 1) / gemv::kWarps;
  const dim3 grid(static_cast<unsigned>(blocks0 + blocks1));
  const size_t smem = gemv::smem_bytes(kRows, k);
  if (out_f32)
    bf16_gemv_dual_kernel<kRows, true>
        <<<grid, gemv::kThreads, smem, stream>>>(p0, p1, blocks0, k);
  else
    bf16_gemv_dual_kernel<kRows, false>
        <<<grid, gemv::kThreads, smem, stream>>>(p0, p1, blocks0, k);
  DGPP_CUDA_OK(cudaGetLastError());
}

template <int kRows>
void launch_rows(const uint16_t* act, size_t act_stride, const uint16_t* w,
                 void* out, bool out_f32, int n, int k, cudaStream_t stream) {
  const dim3 grid((n + gemv::kWarps - 1) / gemv::kWarps);
  const size_t smem = gemv::smem_bytes(kRows, k);
  if (out_f32)
    bf16_gemv_kernel<kRows, true>
        <<<grid, gemv::kThreads, smem, stream>>>(act, act_stride, w, out, n, k);
  else
    bf16_gemv_kernel<kRows, false>
        <<<grid, gemv::kThreads, smem, stream>>>(act, act_stride, w, out, n, k);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace

bool bf16_gemv_accepts(const void* weight, int m, int k) {
  return m >= 1 && m <= gemv::kWideRows && k > 0 && (k % 8) == 0 &&
         gemv::aligned16(weight) && gemv::smem_fits_wide(m, k);
}

namespace {
template <int kRows>
bool opt_in_rows() {
  const int bytes = static_cast<int>(gemv::kWideSmemBytes);
  for (const void* f : {reinterpret_cast<const void*>(bf16_gemv_kernel<kRows, true>),
                        reinterpret_cast<const void*>(bf16_gemv_kernel<kRows, false>),
                        reinterpret_cast<const void*>(bf16_gemv_dual_kernel<kRows, true>),
                        reinterpret_cast<const void*>(bf16_gemv_dual_kernel<kRows, false>),
                        reinterpret_cast<const void*>(bf16_gemv_multi_kernel<kRows, true>),
                        reinterpret_cast<const void*>(bf16_gemv_multi_kernel<kRows, false>)})
    if (cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) != cudaSuccess) {
      (void)cudaGetLastError();
      return false;
    }
  return true;
}
}  // namespace

bool bf16_gemv_opt_in_wide() {
  return opt_in_rows<5>() && opt_in_rows<6>() && opt_in_rows<7>() && opt_in_rows<8>();
}

namespace gemv {
namespace {
bool g_wide_rows = false;
}
bool enable_wide_rows() {
  static const bool on = [] { return bf16_gemv_opt_in_wide() && scale_gemv_opt_in_wide(); }();
  g_wide_rows = on;
  return on;
}
bool wide_rows_enabled() { return g_wide_rows; }
bool smem_fits_wide(int rows, int k) {
  // The one-to-four-row instantiations keep the default ceiling (they are
  // not opted in: every existing chunking stays as it was); the five-to-
  // eight-row ones take the wide bound once the opt-in took.
  const size_t bytes = smem_bytes(rows, k);
  return bytes <= kMaxSmemBytes || (g_wide_rows && rows > kMaxRows && bytes <= kWideSmemBytes);
}
}  // namespace gemv

void launch_bf16_gemv(const uint16_t* act, size_t act_row_stride,
                      const uint16_t* weight, void* out, bool out_f32, int m,
                      int n, int k, cudaStream_t stream) {
  if (n <= 0) return;
  if (!act || !weight || !out)
    throw std::invalid_argument("bf16_gemv: null pointer");
  if (!bf16_gemv_accepts(weight, m, k))
    throw std::invalid_argument("bf16_gemv: shape outside the GEMV contract");
  if (act_row_stride < static_cast<size_t>(k))
    throw std::invalid_argument("bf16_gemv: activation stride narrower than k");
  switch (m) {
    case 1: launch_rows<1>(act, act_row_stride, weight, out, out_f32, n, k, stream); break;
    case 2: launch_rows<2>(act, act_row_stride, weight, out, out_f32, n, k, stream); break;
    case 3: launch_rows<3>(act, act_row_stride, weight, out, out_f32, n, k, stream); break;
    case 4: launch_rows<4>(act, act_row_stride, weight, out, out_f32, n, k, stream); break;
    case 5: launch_rows<5>(act, act_row_stride, weight, out, out_f32, n, k, stream); break;
    case 6: launch_rows<6>(act, act_row_stride, weight, out, out_f32, n, k, stream); break;
    case 7: launch_rows<7>(act, act_row_stride, weight, out, out_f32, n, k, stream); break;
    case 8: launch_rows<8>(act, act_row_stride, weight, out, out_f32, n, k, stream); break;
    default: throw std::invalid_argument("bf16_gemv: m outside 1..8");
  }
}

void launch_bf16_gemv_dual(const Bf16GemvProblem& p0, const Bf16GemvProblem& p1,
                           bool out_f32, int m, int k, cudaStream_t stream) {
  for (const Bf16GemvProblem* p : {&p0, &p1}) {
    if (p->n <= 0 || !p->act || !p->weight || !p->out)
      throw std::invalid_argument("bf16_gemv_dual: empty or null problem");
    if (!bf16_gemv_accepts(p->weight, m, k))
      throw std::invalid_argument("bf16_gemv_dual: shape outside the GEMV "
                                  "contract");
    if (p->act_row_stride < static_cast<size_t>(k))
      throw std::invalid_argument("bf16_gemv_dual: activation stride narrower "
                                  "than k");
  }
  switch (m) {
    case 1: launch_dual_rows<1>(p0, p1, out_f32, k, stream); break;
    case 2: launch_dual_rows<2>(p0, p1, out_f32, k, stream); break;
    case 3: launch_dual_rows<3>(p0, p1, out_f32, k, stream); break;
    case 4: launch_dual_rows<4>(p0, p1, out_f32, k, stream); break;
    case 5: launch_dual_rows<5>(p0, p1, out_f32, k, stream); break;
    case 6: launch_dual_rows<6>(p0, p1, out_f32, k, stream); break;
    case 7: launch_dual_rows<7>(p0, p1, out_f32, k, stream); break;
    case 8: launch_dual_rows<8>(p0, p1, out_f32, k, stream); break;
    default: throw std::invalid_argument("bf16_gemv_dual: m outside 1..8");
  }
}

void launch_bf16_gemv_multi(const Bf16GemvProblem* problems, int n_problems,
                            bool out_f32, int m, int k, cudaStream_t stream) {
  if (n_problems <= 0 || n_problems > kBf16GemvMaxProblems || problems == nullptr)
    throw std::invalid_argument("bf16_gemv_multi: 1..4 problems");
  Bf16GemvMulti mp{};
  mp.n = n_problems;
  int blocks = 0;
  for (int i = 0; i < n_problems; ++i) {
    const Bf16GemvProblem& p = problems[i];
    if (p.n <= 0 || !p.act || !p.weight || !p.out)
      throw std::invalid_argument("bf16_gemv_multi: empty or null problem");
    if (!bf16_gemv_accepts(p.weight, m, k))
      throw std::invalid_argument("bf16_gemv_multi: shape outside the GEMV contract");
    if (p.act_row_stride < static_cast<size_t>(k))
      throw std::invalid_argument("bf16_gemv_multi: activation stride narrower than k");
    mp.p[i] = p;
    blocks += (p.n + gemv::kWarps - 1) / gemv::kWarps;
    mp.block_end[i] = blocks;
  }
  for (int i = n_problems; i < 4; ++i) mp.block_end[i] = blocks;
  switch (m) {
    case 1: launch_multi_rows<1>(mp, out_f32, k, stream); break;
    case 2: launch_multi_rows<2>(mp, out_f32, k, stream); break;
    case 3: launch_multi_rows<3>(mp, out_f32, k, stream); break;
    case 4: launch_multi_rows<4>(mp, out_f32, k, stream); break;
    case 5: launch_multi_rows<5>(mp, out_f32, k, stream); break;
    case 6: launch_multi_rows<6>(mp, out_f32, k, stream); break;
    case 7: launch_multi_rows<7>(mp, out_f32, k, stream); break;
    case 8: launch_multi_rows<8>(mp, out_f32, k, stream); break;
    default: throw std::invalid_argument("bf16_gemv_multi: m outside 1..8");
  }
}

void launch_bf16_gemv_multi_rows(const Bf16GemvProblem* problems, int n_problems, bool out_f32, int m, int k,
                                 cudaStream_t stream) {
  if (m < 1 || m > kBf16GemvMultiMaxRows) throw std::invalid_argument("bf16_gemv_multi_rows: m outside the grid");
  if (n_problems <= 0 || n_problems > kBf16GemvMaxProblems || problems == nullptr)
    throw std::invalid_argument("bf16_gemv_multi_rows: 1..4 problems");
  const int full = m / 4, tail = m % 4;
  Bf16GemvMulti mp{};
  mp.n = n_problems;
  int blocks = 0;
  for (int i = 0; i < n_problems; ++i) {
    const Bf16GemvProblem& p = problems[i];
    if (p.n <= 0 || !p.act || !p.weight || !p.out)
      throw std::invalid_argument("bf16_gemv_multi_rows: empty or null problem");
    if (!bf16_gemv_accepts(p.weight, std::min(m, 4), k))
      throw std::invalid_argument("bf16_gemv_multi_rows: shape outside the GEMV contract");
    if (p.act_row_stride < static_cast<size_t>(k))
      throw std::invalid_argument("bf16_gemv_multi_rows: activation stride narrower than k");
    mp.p[i] = p;
    blocks += (p.n + gemv::kWarps - 1) / gemv::kWarps;
    mp.block_end[i] = blocks;
  }
  for (int i = n_problems; i < 4; ++i) mp.block_end[i] = blocks;
  // The full groups of four in one launch (blockIdx.y the group), the
  // tail rows as their own 1..3-row launch: every row bitwise the chunked
  // single launches'.
  if (full > 0) launch_multi_rows<4>(mp, out_f32, k, stream, full);
  if (tail > 0) {
    Bf16GemvMulti mt = mp;
    for (int i = 0; i < n_problems; ++i) {
      mt.p[i].act += static_cast<size_t>(full) * 4 * mt.p[i].act_row_stride;
      const size_t skip = static_cast<size_t>(full) * 4 * static_cast<size_t>(mt.p[i].n);
      mt.p[i].out = out_f32 ? static_cast<void*>(static_cast<float*>(mt.p[i].out) + skip)
                            : static_cast<void*>(static_cast<uint16_t*>(mt.p[i].out) + skip);
    }
    switch (tail) {
      case 1: launch_multi_rows<1>(mt, out_f32, k, stream); break;
      case 2: launch_multi_rows<2>(mt, out_f32, k, stream); break;
      default: launch_multi_rows<3>(mt, out_f32, k, stream); break;
    }
  }
}

}  // namespace dgpp
