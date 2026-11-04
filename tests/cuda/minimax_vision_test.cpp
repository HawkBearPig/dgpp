#include "kernels/minimax_vision.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/fp32_gemm.hpp"
#include "kernels/gemm.hpp"

namespace {
template <class T>
struct Managed {
  T* p = nullptr;
  explicit Managed(size_t n) { DGPP_CUDA_OK(cudaMallocManaged(&p, n * sizeof(T))); }
  ~Managed() { cudaFree(p); }
};
DGPP_TEST(minimax_patch_convolution_preserves_fp32_weights_and_strides) {
  constexpr int K = 1176, S = K + 8, N = 80, M = 17;
  Managed<float> a(M * S), w(N * K), y(M * N);
  Managed<char> workspace(64 << 20);
  for (int i = 0; i < M * S; ++i) a.p[i] = std::sin(float(i) * .371f) * .07321f;
  for (int i = 0; i < N * K; ++i) w.p[i] = std::cos(float(i) * .131f) * .061123f;
  dgpp::CublasLtGemm gemm;
  dgpp::Fp32Gemm sgemm(workspace.p, 64 << 20, nullptr);
  for (bool lt : {false, true})
    for (int rows : {1, 4, M}) {
      if (lt)
        gemm.matmul(a.p, w.p, y.p, rows, N, K, dgpp::DType::F32, dgpp::GemmOut::F32, S, workspace.p,
                    64 << 20, nullptr);
      else
        sgemm.matmul(a.p, w.p, y.p, rows, N, K, S);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      double error = 0, energy = 0;
      for (int r = 0; r < rows; ++r)
        for (int n = 0; n < N; ++n) {
          double expected = 0;
          for (int k = 0; k < K; ++k) expected += double(a.p[r * S + k]) * w.p[n * K + k];
          const double delta = y.p[r * N + n] - expected;
          error += delta * delta;
          energy += expected * expected;
        }
      if (!std::isfinite(error) || std::sqrt(error / energy) > 3e-6)
        throw std::runtime_error("FP32 vision convolution loses operand precision");
    }
}
DGPP_TEST(minimax_vision_partial_rope_preserves_tail_and_value_order) {
  constexpr int T = 24, H = 160, D = 80, R = 78, heads = 2;
  Managed<uint16_t> packed(T * H * 3), q(T * H), k(T * H), v(T * H);
  Managed<float> cos(T * R / 2), sin(T * R / 2);
  for (int i = 0; i < T * H * 3; ++i)
    packed.p[i] = dgpp::float_to_bf16_bits(float((i * 17) % 97 - 48) / 16);
  for (int i = 0; i < T * R / 2; ++i) {
    cos.p[i] = std::cos(float(i) * .01f);
    sin.p[i] = std::sin(float(i) * .01f);
  }
  dgpp::minimax_vision_split_rope(packed.p, q.p, k.p, v.p, cos.p, sin.p, T, H, heads, R, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < heads; ++h)
      for (int d = 0; d < D; ++d) {
        const int src = t * 3 * H + h * D + d, dst = h * T * D + t * D + d;
        if (v.p[dst] != packed.p[src + 2 * H]) throw std::runtime_error("vision V permutation");
        for (int which = 0; which < 2; ++which) {
          const auto* out = which ? k.p : q.p;
          const int at = src + which * H;
          if (d >= R) {
            if (out[dst] != packed.p[at]) throw std::runtime_error("vision RoPE rotated its tail");
          } else {
            const int other = d < R / 2 ? R / 2 : -R / 2;
            const double sign = d < R / 2 ? -1 : 1;
            const int angle = t * (R / 2) + d % (R / 2);
            const double expected =
                double(dgpp::bf16_bits_to_float(packed.p[at])) * cos.p[angle] +
                sign * dgpp::bf16_bits_to_float(packed.p[at + other]) * sin.p[angle];
            if (std::abs(dgpp::bf16_bits_to_float(out[dst]) - expected) >
                .004 * std::abs(expected) + 1e-6)
              throw std::runtime_error("vision partial NeoX rotation");
          }
        }
      }
}
DGPP_TEST(minimax_vision_rotary_frequency_and_trigonometry_rounding) {
  // Independent PyTorch 2.14 CUDA float32 construction at position 3:
  // freq = 1 / pow(10000, arange(13, float32) / 13); cos/sin(3 * freq).
  // A double-precision host pow followed by a float cast changes BF16 Q/K
  // rounding later in the real 32-layer tower, so retain the exact recipe.
  constexpr uint32_t cosine[] = {0xbf7d7026u, 0x3dbf7a38u, 0x3f3f37e5u, 0x3f6fc1efu, 0x3f7c07afu,
                                 0x3f7f091du, 0x3f7fc41du, 0x3f7ff17bu, 0x3f7ffc7bu, 0x3f7fff25u,
                                 0x3f7fffcbu, 0x3f7ffff3u, 0x3f7ffffdu};
  constexpr uint32_t sine[] = {0x3e1081c3u, 0x3f7ee0f0u, 0x3f2a35d7u, 0x3eb3782du, 0x3e33a364u,
                               0x3db199b0u, 0x3d2f102fu, 0x3cac701eu, 0x3c29d287u, 0x3ba73d46u,
                               0x3b24b1a6u, 0x3aa22fd6u, 0x3a1fb7beu};
  Managed<float> table(16 * 78);
  dgpp::minimax_vision_rope_tables(table.p, 56, 56, 78, 10000, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  for (int j = 0; j < 39; ++j) {
    const int at = 15 * 39 + j;
    const uint32_t want_cos = j < 13 ? 0x3f800000u : cosine[j % 13];
    const uint32_t want_sin = j < 13 ? 0 : sine[j % 13];
    if (std::bit_cast<uint32_t>(table.p[at]) != want_cos ||
        std::bit_cast<uint32_t>(table.p[16 * 39 + at]) != want_sin)
      throw std::runtime_error("MiniMax vision rotary table rounding changed");
  }
}
}  // namespace
int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 2;
  return dgpp::test::run_all();
}
