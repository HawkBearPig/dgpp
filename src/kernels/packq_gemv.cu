#include "kernels/packq_gemv.hpp"
#include "kernels/pdl.cuh"

#include <algorithm>
#include <stdexcept>
#include <type_traits>

#include "common/cuda_check.hpp"
#include "kernels/packq_gemv.cuh"

namespace dgpp {
namespace {

// One block = kWarps warps x rows_per_warp(Bits, K) rows. The activation
// rows are staged bf16 in dynamic shared memory.
template <int Bits, int K, int kRows, int SF, typename OutT>
__global__ void packq_gemv_kernel(const uint16_t* __restrict__ act,
                                  size_t act_stride,
                                  const uint8_t* __restrict__ w,
                                  const uint16_t* __restrict__ scales,
                                  OutT* __restrict__ out, int n) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint16_t sx[];
  packq_gemv::stage_activations<kRows>(act, act_stride, K, sx);
  __syncthreads();
  packq_gemv::block_rows<Bits, K, kRows, SF>(
      w, scales, sx, blockIdx.x * packq_gemv::Geom<Bits, K, SF>::rows_per_block, n, out,
      static_cast<size_t>(n));
}

template <int kRows, typename OutT>
void launch_rows(const uint16_t* act, size_t act_stride, const GlmPackedMatrix& w,
                 OutT* out, int n, int k, cudaStream_t stream) {
  packq_gemv::dispatch_sf(w.scale_fmt, [&](auto sfc) {
    constexpr int SF = decltype(sfc)::value;
    packq_gemv::dispatch_bits(w.bits, [&](auto bc) {
      constexpr int Bits = decltype(bc)::value;
      packq_gemv::dispatch_k(k, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        if constexpr (packq_gemv::k_supported<Bits, SF>(K) && packq_gemv::sf_compiled(SF, K)) {
          const int rows_per_block = packq_gemv::Geom<Bits, K, SF>::rows_per_block;
          const dim3 grid((n + rows_per_block - 1) / rows_per_block);
          packq_gemv_kernel<Bits, K, kRows, SF, OutT>
              <<<grid, packq_gemv::kThreads, packq_gemv::smem_bytes(kRows, K), stream>>>(
                  act, act_stride, reinterpret_cast<const uint8_t*>(w.packed), w.scales, out, n);
          DGPP_CUDA_OK(cudaGetLastError());
        } else {
          throw std::invalid_argument(
              "packq_gemv: K is outside the scale format's compiled set or the width's chunk budget");
        }
      });
    });
  });
}

template <typename OutT>
void launch(const uint16_t* act, size_t act_stride, const GlmPackedMatrix& w,
            OutT* out, int m, int n, int k, cudaStream_t stream) {
  if (m <= 0 || n <= 0) return;
  if (!act || !w.packed || !w.scales || !out)
    throw std::invalid_argument("packq_gemv: null pointer");
  if (w.rows < n || w.cols != k)
    throw std::invalid_argument("packq_gemv: matrix geometry does not match n, k");
  if (w.layout != kPackedLayoutRows)
    throw std::invalid_argument("packq_gemv: the plane layout takes the head launcher (packq_head.hpp)");
  if (!packq_gemv::scale_fmt_known(w.scale_fmt) || !packq_gemv::sf_compiled(w.scale_fmt, k) ||
      !packq_gemv::shape_ok(w.packed, w.bits, k, w.scale_fmt))
    throw std::invalid_argument(
        "packq_gemv: K must be a multiple of the scale group in the format's compiled set, "
        "the code width 4 or 8, and the payload 16-byte aligned");
  for (int row0 = 0; row0 < m;) {
    int rows = std::min(packq_gemv::kMaxRows, m - row0);
    while (!gemv::smem_fits(rows, k)) --rows;
    const uint16_t* a = act + static_cast<size_t>(row0) * act_stride;
    OutT* o = out + static_cast<size_t>(row0) * n;
    switch (rows) {
      case 4: launch_rows<4, OutT>(a, act_stride, w, o, n, k, stream); break;
      case 3: launch_rows<3, OutT>(a, act_stride, w, o, n, k, stream); break;
      case 2: launch_rows<2, OutT>(a, act_stride, w, o, n, k, stream); break;
      default: launch_rows<1, OutT>(a, act_stride, w, o, n, k, stream); break;
    }
    row0 += rows;
  }
}

}  // namespace

void launch_packq_gemv_bf16(const uint16_t* act, size_t act_row_stride_elems,
                            const GlmPackedMatrix& w, uint16_t* out, int m, int n,
                            int k, cudaStream_t stream) {
  launch<uint16_t>(act, act_row_stride_elems, w, out, m, n, k, stream);
}

void launch_packq_gemv_f32(const uint16_t* act, size_t act_row_stride_elems,
                           const GlmPackedMatrix& w, float* out, int m, int n,
                           int k, cudaStream_t stream) {
  launch<float>(act, act_row_stride_elems, w, out, m, n, k, stream);
}

bool packq_gemv_accepts(const GlmPackedMatrix& w) {
  return w.packed && w.scales && w.cols > 0 && (w.bits == 4 || w.bits == 8) &&
         packq_gemv::scale_fmt_known(w.scale_fmt) &&
         packq_gemv::sf_compiled(w.scale_fmt, static_cast<int>(w.cols)) &&
         packq_gemv::shape_ok(w.packed, w.bits, static_cast<int>(w.cols), w.scale_fmt) &&
         gemv::smem_fits(1, static_cast<int>(w.cols));
}

}  // namespace dgpp
