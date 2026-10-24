// The exact fp8-weight GEMM (kernels/fp8w_gemm, #87): the decoded weight
// tile is bitwise the dequant bridge's (bf16(e4m3 x scale), one rounding),
// the product agrees with a double reference over those bf16 weights to
// fp32-accumulation precision on ragged m / n / half-group k shapes, the
// bf16 output is the f32 output rounded once, the launch is deterministic,
// and an output stride wider than n leaves the padding untouched.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/fp8_dequant.hpp"
#include "kernels/fp8w_gemm.hpp"
#include "kernels/scale_gemm.hpp"
#include "kernels/mma_gemv.hpp"
#include "kernels/glm_moe_launch.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    std::exit(1);
  }
}

template <typename T>
struct Dev {
  T* p = nullptr;
  size_t n = 0;
  explicit Dev(size_t count) : n(count) { DGPP_CUDA_OK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T))); }
  ~Dev() { cudaFree(p); }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
  void upload(const std::vector<T>& v) { DGPP_CUDA_OK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
  std::vector<T> download() const {
    std::vector<T> v(n);
    DGPP_CUDA_OK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }
};

struct Shape {
  int m, n, k;
};

void run(const Shape& sh, std::mt19937& rng, dgpp::Fp8wScale form) {
  const int m = sh.m, n = sh.n, k = sh.k;
  const bool per_weight = form == dgpp::Fp8wScale::PerWeight;
  const int scale_rows = (n + 127) / 128, scale_cols = (k + 127) / 128;
  std::normal_distribution<float> normal(0.f, 1.f);
  std::uniform_real_distribution<float> uni(4.0e-4f, 6.0e-4f);
  // Activations: bf16 rows at a padded stride, a few rows scaled up.
  const size_t stride = static_cast<size_t>(k) + 8;
  std::vector<uint16_t> act(static_cast<size_t>(m) * stride, 0);
  for (int r = 0; r < m; ++r) {
    const float rs = (r % 7 == 0) ? 8.f : 1.f;
    for (int c = 0; c < k; ++c) act[r * stride + c] = dgpp::float_to_bf16_bits(normal(rng) * rs);
  }
  // Weights: random e4m3 codes (no NaN) under random block scales.
  std::vector<uint8_t> w(static_cast<size_t>(n) * k);
  std::uniform_int_distribution<int> code(0, 255);
  for (auto& b : w) {
    int c = code(rng);
    if ((c & 0x7F) == 0x7F) c &= 0x7E;
    b = static_cast<uint8_t>(c);
  }
  std::vector<float> ws(static_cast<size_t>(scale_rows) * scale_cols);
  for (auto& s : ws) s = uni(rng);
  // The bridge's weights: bf16(decode(code) x scale), one rounding.
  std::vector<uint16_t> w16(w.size());
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < k; ++c)
      w16[static_cast<size_t>(r) * k + c] = dgpp::float_to_bf16_bits(
          dgpp::fp8_e4m3_bits_to_float(w[static_cast<size_t>(r) * k + c]) * ws[(r / 128) * scale_cols + c / 128]);
  Dev<uint16_t> d_act(act.size());
  Dev<uint8_t> d_w(w.size());
  Dev<float> d_ws(ws.size());
  Dev<uint16_t> d_w16(w16.size());
  d_act.upload(act);
  d_w.upload(w);
  d_ws.upload(ws);
  // The device dequant (the bridge kernel) agrees with the host decode.
  dgpp::launch_fp8_dequant_blocks(d_w.p, d_ws.p, d_w16.p, n, k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  require(d_w16.download() == w16, "the bridge's dequant is not the host decode");
  // The reference in double: the per-weight form over the bridge's bf16
  // weights, the per-group form over the exact code values with the
  // block scale applied per 128-wide k group.
  std::vector<double> ref(static_cast<size_t>(m) * n);
  for (int r = 0; r < m; ++r)
    for (int c = 0; c < n; ++c) {
      double acc = 0.0;
      const uint16_t* xa = &act[r * stride];
      if (per_weight) {
        const uint16_t* wb = &w16[static_cast<size_t>(c) * k];
        for (int i = 0; i < k; ++i)
          acc += static_cast<double>(dgpp::bf16_bits_to_float(xa[i])) * static_cast<double>(dgpp::bf16_bits_to_float(wb[i]));
      } else {
        const uint8_t* wb = &w[static_cast<size_t>(c) * k];
        for (int g = 0; g < scale_cols; ++g) {
          double part = 0.0;
          for (int i = g * 128; i < std::min(k, (g + 1) * 128); ++i)
            part += static_cast<double>(dgpp::bf16_bits_to_float(xa[i])) * static_cast<double>(dgpp::fp8_e4m3_bits_to_float(wb[i]));
          acc += part * static_cast<double>(ws[(c / 128) * scale_cols + g]);
        }
      }
      ref[static_cast<size_t>(r) * n + c] = acc;
    }
  const size_t out_stride = static_cast<size_t>(n) + 16;
  Dev<float> d_f32(static_cast<size_t>(m) * out_stride);
  Dev<uint16_t> d_bf16(static_cast<size_t>(m) * out_stride);
  std::vector<float> sentinel(d_f32.n, -12345.f);
  std::vector<uint16_t> sentinel16(d_bf16.n, 0xBEEF);
  d_f32.upload(sentinel);
  d_bf16.upload(sentinel16);
  dgpp::launch_fp8w_gemm_f32(d_act.p, stride, d_w.p, d_ws.p, d_f32.p, m, n, k, form, nullptr, out_stride);
  dgpp::launch_fp8w_gemm_bf16(d_act.p, stride, d_w.p, d_ws.p, d_bf16.p, m, n, k, form, nullptr, out_stride);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const auto f32 = d_f32.download();
  const auto b16 = d_bf16.download();
  double rms = 0.0;
  for (double v : ref) rms += v * v;
  rms = std::sqrt(rms / static_cast<double>(ref.size()));
  double worst = 0.0;
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < n; ++c) {
      const double got = f32[r * out_stride + c], want = ref[static_cast<size_t>(r) * n + c];
      const double err = std::fabs(got - want) / (std::fabs(want) + rms);
      worst = std::max(worst, err);
      require(b16[r * out_stride + c] == dgpp::float_to_bf16_bits(f32[r * out_stride + c]),
              "the bf16 output is not the f32 output rounded once");
    }
    for (size_t c = n; c < out_stride; ++c) {
      require(f32[r * out_stride + c] == -12345.f && b16[r * out_stride + c] == 0xBEEF,
              "the output padding was written");
    }
  }
  if (per_weight && k % 16 == 0) {
    // The scale GEMM's tile kernel (glm_moe_launch's dense form) computes
    // the same per-weight terms in the same ascending-k16 mma chain: the
    // two must agree bit for bit (the large-row route swaps one for the
    // other, #89).
    Dev<float> t32(static_cast<size_t>(m) * out_stride);
    Dev<uint16_t> t16(static_cast<size_t>(m) * out_stride);
    dgpp::launch_dense_mma_f32(d_act.p, stride, d_w.p, d_ws.p, t32.p, m, n, k, nullptr, out_stride);
    dgpp::launch_dense_mma_bf16(d_act.p, stride, d_w.p, d_ws.p, t16.p, m, n, k, nullptr, out_stride);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const std::vector<float> r32 = t32.download();
    const std::vector<uint16_t> r16 = t16.download();
    size_t diff32 = 0, diff16 = 0;
    for (int r = 0; r < m; ++r)
      for (int c = 0; c < n; ++c) {
        diff32 += std::memcmp(&r32[r * out_stride + c], &f32[r * out_stride + c], 4) != 0;
        diff16 += r16[r * out_stride + c] != b16[r * out_stride + c];
      }
    std::printf("           vs the tile kernel: %zu f32 and %zu bf16 outputs differ\n", diff32, diff16);
    require(diff32 == 0 && diff16 == 0, "the per-weight form is not bitwise the scale GEMM's tile kernel");
  }
  // Deterministic across launches.
  dgpp::launch_fp8w_gemm_f32(d_act.p, stride, d_w.p, d_ws.p, d_f32.p, m, n, k, form, nullptr, out_stride);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  require(d_f32.download() == f32, "the launch is not deterministic");
  std::printf("%-10s m=%d n=%d k=%d: worst relative error %.2e (of |ref| + rms %.3g)\n",
              per_weight ? "per-weight" : "per-group", m, n, k, worst, rms);
  // fp32 accumulation over k terms: the error grows with sqrt(k).
  require(worst < 1.5e-6 * std::sqrt(static_cast<double>(k)),
          "the product disagrees with the double reference beyond fp32 accumulation");
}

// The 32 x 32 scale grid (DeepSeek-V4.1's fp8 projections, 2026-10-18):
// the per-weight form on rs = cs = 5 against the double reference over the
// bridge's weights, and bitwise against the streaming decode form
// (mma_gemv.hpp) on the same grid — the prefill route swaps one for the
// other above 256 rows.
void run32(int m, int n, int k, std::mt19937& rng, int rs = 5) {
  const int cs = 5;
  const int scale_rows = (n + (1 << rs) - 1) >> rs, scale_cols = (k + 31) / 32;
  std::normal_distribution<float> normal(0.f, 1.f);
  const size_t stride = static_cast<size_t>(k) + 8;
  std::vector<uint16_t> act(static_cast<size_t>(m) * stride, 0);
  for (int r = 0; r < m; ++r)
    for (int c = 0; c < k; ++c) act[r * stride + c] = dgpp::float_to_bf16_bits(normal(rng) * ((r % 5 == 0) ? 6.f : 1.f));
  std::vector<uint8_t> w(static_cast<size_t>(n) * k);
  std::uniform_int_distribution<int> code(0, 255);
  for (auto& b : w) {
    int c = code(rng);
    if ((c & 0x7F) == 0x7F) c &= 0x7E;
    b = static_cast<uint8_t>(c);
  }
  // e8m0-like scales: powers of two spread over a few binades.
  std::uniform_int_distribution<int> e(-12, -6);
  std::vector<float> ws(static_cast<size_t>(scale_rows) * scale_cols);
  for (auto& v : ws) v = std::ldexp(1.0f, e(rng));
  std::vector<uint16_t> w16(w.size());
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < k; ++c)
      w16[static_cast<size_t>(r) * k + c] = dgpp::float_to_bf16_bits(
          dgpp::fp8_e4m3_bits_to_float(w[static_cast<size_t>(r) * k + c]) * ws[(r >> rs) * scale_cols + (c >> cs)]);
  std::vector<double> ref(static_cast<size_t>(m) * n);
  for (int r = 0; r < m; ++r)
    for (int c = 0; c < n; ++c) {
      double acc = 0.0;
      for (int i = 0; i < k; ++i)
        acc += static_cast<double>(dgpp::bf16_bits_to_float(act[r * stride + i])) *
               static_cast<double>(dgpp::bf16_bits_to_float(w16[static_cast<size_t>(c) * k + i]));
      ref[static_cast<size_t>(r) * n + c] = acc;
    }
  Dev<uint16_t> d_act(act.size());
  Dev<uint8_t> d_w(w.size());
  Dev<float> d_ws(ws.size());
  d_act.upload(act);
  d_w.upload(w);
  d_ws.upload(ws);
  const size_t out_stride = static_cast<size_t>(n) + 16;
  Dev<float> d_f32(static_cast<size_t>(m) * out_stride), g32(static_cast<size_t>(m) * out_stride);
  Dev<uint16_t> d_b16(static_cast<size_t>(m) * out_stride);
  dgpp::launch_fp8w_gemm_f32(d_act.p, stride, d_w.p, d_ws.p, d_f32.p, m, n, k, dgpp::Fp8wScale::PerWeight, nullptr,
                             out_stride, rs, cs);
  dgpp::launch_fp8w_gemm_bf16(d_act.p, stride, d_w.p, d_ws.p, d_b16.p, m, n, k, dgpp::Fp8wScale::PerWeight, nullptr,
                              out_stride, rs, cs);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const auto f32 = d_f32.download();
  const auto b16 = d_b16.download();
  double rms = 0.0;
  for (double v : ref) rms += v * v;
  rms = std::sqrt(rms / static_cast<double>(ref.size()));
  double worst = 0.0;
  for (int r = 0; r < m; ++r)
    for (int c = 0; c < n; ++c) {
      const double got = f32[r * out_stride + c], want = ref[static_cast<size_t>(r) * n + c];
      worst = std::max(worst, std::fabs(got - want) / (std::fabs(want) + rms));
      require(b16[r * out_stride + c] == dgpp::float_to_bf16_bits(f32[r * out_stride + c]),
              "32-grid: the bf16 output is not the f32 output rounded once");
    }
  require(worst < 1.5e-6 * std::sqrt(static_cast<double>(k)), "32-grid: the product disagrees with the double reference");
  if (rs == 0) {
    // The byte-scale entry point must preserve the tile's arithmetic,
    // including chunk cuts through both 128-row boundaries and graph replay.
    std::vector<uint8_t> bytes(ws.size());
    for (size_t i = 0; i < ws.size(); ++i) bytes[i] = static_cast<uint8_t>(std::ilogb(ws[i]) + 127);
    Dev<uint8_t> d_bytes(bytes.size());
    d_bytes.upload(bytes);
    Dev<float> tiled(d_f32.n);
    Dev<uint16_t> tiled16(d_b16.n);
    tiled.upload(std::vector<float>(tiled.n, -12345.f));
    tiled16.upload(std::vector<uint16_t>(tiled16.n, 0xBEEF));
    cudaStream_t stream;
    DGPP_CUDA_OK(cudaStreamCreate(&stream));
    DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int r0 = 0; r0 < m; r0 += 31) {
      const int rows = std::min(31, m - r0);
      dgpp::launch_scale_gemm_grid_e8m0_f32(d_act.p + r0 * stride, stride, d_w.p, d_bytes.p,
          tiled.p + r0 * out_stride, rows, n, k, stream, out_stride, rs, cs, nullptr, 0, true);
    }
    dgpp::launch_scale_gemm_grid_e8m0_bf16(d_act.p, stride, d_w.p, d_bytes.p, tiled16.p,
        m, n, k, stream, out_stride, rs, cs, nullptr, 0, true);
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
    DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 2; ++replay) {
      DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      const auto got = tiled.download();
      const auto got16 = tiled16.download();
      for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c)
          require(got[r * out_stride + c] == f32[r * out_stride + c] &&
                  got16[r * out_stride + c] == b16[r * out_stride + c],
                  "MXFP8 byte grid/chunk/graph differs from the float grid");
        for (size_t c = n; c < out_stride; ++c)
          require(got[r * out_stride + c] == -12345.f && got16[r * out_stride + c] == 0xBEEF,
                  "MXFP8 graph wrote output padding");
      }
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
  }
  // The streaming form on the same grid, group by group as the scale GEMM's
  // decode route runs it (128 rows a launch).
  for (int r0 = 0; r0 < m; r0 += 128) {
    const int rows = std::min(128, m - r0);
    dgpp::launch_mma_gemv_fp8_f32(d_act.p + static_cast<size_t>(r0) * stride, stride, d_w.p, d_ws.p,
                                  g32.p + static_cast<size_t>(r0) * out_stride, rows, n, k, out_stride, rs, cs, nullptr);
  }
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const auto s32 = g32.download();
  size_t diff = 0;
  for (int r = 0; r < m; ++r)
    for (int c = 0; c < n; ++c) diff += std::memcmp(&s32[r * out_stride + c], &f32[r * out_stride + c], 4) != 0;
  // The streaming form's chain splits a row's dot across a quad of lanes and
  // sums the partials; the tile form accumulates one chain per output. Both
  // place the same bf16(e4m3 x scale) weights: equal within fp32 rounding,
  // not bitwise (the scale GEMM's grid route keeps one form per walk).
  double worst_vs = 0.0;
  for (int r = 0; r < m; ++r)
    for (int c = 0; c < n; ++c)
      worst_vs = std::max(worst_vs, std::fabs(static_cast<double>(s32[r * out_stride + c]) - f32[r * out_stride + c]) /
                                        (std::fabs(ref[static_cast<size_t>(r) * n + c]) + rms));
  std::printf("32-grid   m=%d n=%d k=%d: worst relative error %.2e; vs the streaming form %zu outputs differ, worst %.2e\n",
              m, n, k, worst, diff, worst_vs);
  require(worst_vs < 1.5e-6 * std::sqrt(static_cast<double>(k)), "32-grid: the tile form drifts from the streaming form");
}

void high_byte_scales() {
  // Valid E8M0 exponents above 134 overflow scale * 2^120 in the placed
  // float-grid shortcut. Byte scales must decode before that multiplication.
  Dev<uint16_t> act(64), out(2);
  Dev<uint8_t> weights(2 * 64), scales(2 * 2);
  act.upload(std::vector<uint16_t>(64, dgpp::float_to_bf16_bits(1.f)));
  weights.upload(std::vector<uint8_t>(128, 0x38));  // e4m3 1
  scales.upload({135, 135, 241, 241});
  dgpp::launch_fp8w_gemm_e8m0_bf16(act.p, 64, weights.p, scales.p, out.p, 1, 2, 64, nullptr);
  const auto got = out.download();
  require(got[0] == dgpp::float_to_bf16_bits(std::ldexp(64.f, 8)) &&
          got[1] == dgpp::float_to_bf16_bits(std::ldexp(64.f, 114)), "valid high E8M0 scale overflowed");
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("no CUDA device: skipped\n");
    return 0;
  }
  high_byte_scales();
  std::mt19937 rng(87);
  // Ragged m (a 128-tile plus a tail), ragged n (two scale rows, the second
  // partial), k with a half scale group (192 = 3 x 64, two scale columns),
  // and a 27B-like k on several m-tile groups.
  const Shape shapes[] = {{300, 200, 192}, {137, 384, 128}, {1100, 256, 2048}, {33, 1024, 5120}};
  for (const Shape& sh : shapes) {
    run(sh, rng, dgpp::Fp8wScale::PerWeight);
    run(sh, rng, dgpp::Fp8wScale::PerGroup);
  }
  run32(300, 320, 1280, rng);
  run32(513, 1024, 5120, rng);
  // The dsv41 test fixture's projection shapes at world 1: wq_b [8192 x 128],
  // wo_a [64 x 2048] per group, wo_b [hidden x 256], wq_a [128 x hidden].
  run32(70, 8192, 128, rng);
  run32(70, 64, 2048, rng);
  run32(70, 256, 256, rng);
  run32(70, 128, 256, rng);
  run32(70, 1280, 6144, rng);
  for (int m : {1, 7, 31, 127, 128, 129, 257}) run32(m, 133, 640, rng, 0);
  run32(257, 320, 2560, rng, 0);
  std::printf("fp8w_gemm_test: OK\n");
  return 0;
}
