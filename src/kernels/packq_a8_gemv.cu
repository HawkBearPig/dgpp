#include "kernels/packq_a8_gemv.hpp"
#include "kernels/pdl.cuh"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "kernels/packq_a8_gemv.cuh"

namespace dgpp {
namespace {

template <int Bits, int K, int kRows, typename OutT>
__global__ void packq_a8_gemv_kernel(const int8_t* __restrict__ codes, size_t code_stride,
                                     const float* __restrict__ scales, size_t scale_stride,
                                     const uint8_t* __restrict__ w, const uint16_t* __restrict__ wscales,
                                     OutT* __restrict__ out, size_t out_stride, int n) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint8_t smem[];
  packq_a8::stage_activations<kRows>(reinterpret_cast<const uint8_t*>(codes), code_stride, scales, scale_stride, K,
                                     smem);
  __syncthreads();
  const packq_a8::Staged st = packq_a8::staged_of(smem, kRows, K);
  const int n0 = blockIdx.x * packq_a8::Geom<Bits, K>::rows_per_block;
  if (n0 >= n) return;
  packq_a8::block_rows<Bits, K, kRows>(w, wscales, st, n0, n, out, out_stride);
}

template <typename OutT>
void launch(const int8_t* codes, size_t code_stride, const float* scales, size_t scale_stride,
            const GlmPackedMatrix& w, OutT* out, int m, int n, int k, cudaStream_t stream) {
  if (m <= 0 || n <= 0) return;
  if (!codes || !scales || !out) throw std::invalid_argument("packq_a8_gemv: null pointer");
  if (!packq_a8_gemv_accepts(w) || w.cols != k || w.rows < n)
    throw std::invalid_argument("packq_a8_gemv: the matrix is not a Mixed346 matrix of this geometry");
  if (code_stride < static_cast<size_t>(k) || code_stride % 16 != 0 || reinterpret_cast<uintptr_t>(codes) % 16 != 0)
    throw std::invalid_argument("packq_a8_gemv: code rows must be 16-byte aligned and at least k wide");
  if (scale_stride < static_cast<size_t>(k / packq_a8::kGroup))
    throw std::invalid_argument("packq_a8_gemv: scale rows narrower than k / 128");
  packq_a8::dispatch_bits(w.bits, [&](auto bc) {
    constexpr int Bits = decltype(bc)::value;
    packq_a8::dispatch_k(k, [&](auto kc) {
      constexpr int K = decltype(kc)::value;
      if constexpr (packq_a8::k_supported(K)) {
        constexpr int rpb = packq_a8::Geom<Bits, K>::rows_per_block;
        const dim3 grid((n + rpb - 1) / rpb);
        const uint8_t* wp = reinterpret_cast<const uint8_t*>(w.packed);
        for (int m0 = 0; m0 < m; m0 += packq_a8::kMaxRows) {
          const int rows = m - m0 < packq_a8::kMaxRows ? m - m0 : packq_a8::kMaxRows;
          const int8_t* c = codes + static_cast<size_t>(m0) * code_stride;
          const float* s = scales + static_cast<size_t>(m0) * scale_stride;
          OutT* o = out + static_cast<size_t>(m0) * n;
          switch (rows) {
            case 4:
              packq_a8_gemv_kernel<Bits, K, 4, OutT><<<grid, packq_a8::kThreads, packq_a8::smem_bytes(4, K), stream>>>(
                  c, code_stride, s, scale_stride, wp, w.scales, o, n, n);
              break;
            case 3:
              packq_a8_gemv_kernel<Bits, K, 3, OutT><<<grid, packq_a8::kThreads, packq_a8::smem_bytes(3, K), stream>>>(
                  c, code_stride, s, scale_stride, wp, w.scales, o, n, n);
              break;
            case 2:
              packq_a8_gemv_kernel<Bits, K, 2, OutT><<<grid, packq_a8::kThreads, packq_a8::smem_bytes(2, K), stream>>>(
                  c, code_stride, s, scale_stride, wp, w.scales, o, n, n);
              break;
            default:
              packq_a8_gemv_kernel<Bits, K, 1, OutT><<<grid, packq_a8::kThreads, packq_a8::smem_bytes(1, K), stream>>>(
                  c, code_stride, s, scale_stride, wp, w.scales, o, n, n);
              break;
          }
          DGPP_CUDA_OK(cudaGetLastError());
        }
      } else {
        throw std::invalid_argument("packq_a8_gemv: K outside the core's contract");
      }
    });
  });
}

}  // namespace

void launch_packq_a8_gemv_bf16(const int8_t* codes, size_t code_stride, const float* scales, size_t scale_stride,
                               const GlmPackedMatrix& w, uint16_t* out, int m, int n, int k, cudaStream_t stream) {
  launch<uint16_t>(codes, code_stride, scales, scale_stride, w, out, m, n, k, stream);
}
void launch_packq_a8_gemv_f32(const int8_t* codes, size_t code_stride, const float* scales, size_t scale_stride,
                              const GlmPackedMatrix& w, float* out, int m, int n, int k, cudaStream_t stream) {
  launch<float>(codes, code_stride, scales, scale_stride, w, out, m, n, k, stream);
}

bool packq_a8_gemv_accepts(const GlmPackedMatrix& w) {
  return w.packed != nullptr && w.scales != nullptr && w.scale_fmt == kPackedScaleBf16G128Mixed346 &&
         packq_a8::bits_known(w.bits) && w.cols > 0 && w.cols <= 0x7FFFFFFF &&
         packq_a8::k_supported(static_cast<int>(w.cols)) && packq_a8::k_compiled(static_cast<int>(w.cols)) &&
         reinterpret_cast<uintptr_t>(w.packed) % 16 == 0 && packq_a8::smem_fits(packq_a8::kMaxRows, static_cast<int>(w.cols));
}

}  // namespace dgpp
