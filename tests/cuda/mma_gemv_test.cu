// The streaming tensor-core decode GEMM (kernels/mma_gemv.hpp): against a
// double oracle over the dequant bridge's weight values (the tile kernel's
// budget: fp32-order differences only), bitwise m-invariance (a row's
// result identical at m = 1 and inside an m = 30 batch, fp8 and bf16), the
// fp32 and bf16 epilogues agreeing bit for bit, and a cold timing line
// beside the GEMV cores it replaces (informational).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/bf12_gemv.hpp"
#include "kernels/bf16_gemv.hpp"
#include "kernels/dense_mma_bf16w.hpp"
#include "kernels/fp8_dequant.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/mma_gemv.hpp"
#include "kernels/scale_gemm.hpp"

namespace {
void require(bool c, const std::string& w) { if (!c) throw std::runtime_error(w); }

struct Problem {
  int m, n, k, rs, cs;
  std::vector<uint8_t> w8;      // [n, k] e4m3
  std::vector<uint16_t> w16;    // [n, k] bf16
  std::vector<float> scales;    // [n>>rs][k>>cs]
  std::vector<uint16_t> act;    // [m, k] bf16
};

Problem make(int m, int n, int k, int rs, int cs, uint64_t seed) {
  Problem p{m, n, k, rs, cs, {}, {}, {}, {}};
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> code(0, 255);
  std::uniform_real_distribution<float> uf(-1.f, 1.f);
  p.w8.resize(size_t(n) * k);
  for (auto& b : p.w8) { uint8_t c = code(rng); if ((c & 0x7F) == 0x7F) c &= 0x7E; b = c; }  // no NaN codes
  p.w16.resize(size_t(n) * k);
  for (auto& x : p.w16) x = dgpp::float_to_bf16_bits(uf(rng));
  const int sr = (n + (1 << rs) - 1) >> rs, sc = (k + (1 << cs) - 1) >> cs;
  p.scales.resize(size_t(sr) * sc);
  for (auto& s : p.scales) s = std::ldexp(1.f, -(rng() % 8));  // e8m0-like powers of two
  p.act.resize(size_t(m) * k);
  for (auto& x : p.act) x = dgpp::float_to_bf16_bits(uf(rng));
  return p;
}

// The dequant bridge's weight value: bf16(e4m3 x scale).
float wval8(const Problem& p, int r, int c) {
  const float s = p.scales[size_t(r >> p.rs) * ((p.k + (1 << p.cs) - 1) >> p.cs) + (c >> p.cs)];
  return dgpp::bf16_bits_to_float(dgpp::float_to_bf16_bits(dgpp::fp8_e4m3_bits_to_float(p.w8[size_t(r) * p.k + c]) * s));
}

template <typename F>
double max_rel_err(const Problem& p, const std::vector<float>& got, F wv) {
  double worst = 0;
  for (int i = 0; i < p.m; ++i)
    for (int j = 0; j < p.n; ++j) {
      if (!std::isfinite(got[size_t(i) * p.n + j]))
        throw std::runtime_error("non-finite output at row " + std::to_string(i) + " column " +
                                 std::to_string(j));
      double ref = 0, mag = 0;
      for (int c = 0; c < p.k; ++c) {
        const double a = dgpp::bf16_bits_to_float(p.act[size_t(i) * p.k + c]), w = wv(j, c);
        ref += a * w; mag += std::fabs(a * w);
      }
      const double err = std::fabs(got[size_t(i) * p.n + j] - ref) / (mag + 1e-6);
      if (err > worst) worst = err;
    }
  return worst;
}

struct Dev {
  uint8_t* w8 = nullptr; uint16_t* w16 = nullptr; float* scales = nullptr; uint16_t* act = nullptr;
  float* outf = nullptr; uint16_t* outb = nullptr;
  explicit Dev(const Problem& p) {
    DGPP_CUDA_OK(cudaMalloc(&w8, p.w8.size())); DGPP_CUDA_OK(cudaMemcpy(w8, p.w8.data(), p.w8.size(), cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMalloc(&w16, p.w16.size() * 2)); DGPP_CUDA_OK(cudaMemcpy(w16, p.w16.data(), p.w16.size() * 2, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMalloc(&scales, p.scales.size() * 4)); DGPP_CUDA_OK(cudaMemcpy(scales, p.scales.data(), p.scales.size() * 4, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMalloc(&act, p.act.size() * 2)); DGPP_CUDA_OK(cudaMemcpy(act, p.act.data(), p.act.size() * 2, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMalloc(&outf, size_t(std::max(32, p.m)) * p.n * 4)); DGPP_CUDA_OK(cudaMalloc(&outb, size_t(std::max(32, p.m)) * p.n * 2));
  }
  ~Dev() { cudaFree(w8); cudaFree(w16); cudaFree(scales); cudaFree(act); cudaFree(outf); cudaFree(outb); }
};
std::vector<float> fetch_f(const float* d, size_t n) { std::vector<float> h(n); DGPP_CUDA_OK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost)); return h; }
std::vector<uint16_t> fetch_b(const uint16_t* d, size_t n) { std::vector<uint16_t> h(n); DGPP_CUDA_OK(cudaMemcpy(h.data(), d, n * 2, cudaMemcpyDeviceToHost)); return h; }
}  // namespace

DGPP_TEST(mma_gemv_oracle_rejects_nonfinite_outputs) {
  const Problem p{1, 3, 1, 0, 0, {}, {}, {}, {dgpp::float_to_bf16_bits(1.f)}};
  const auto weight = [](int row, int) { return double(row + 1); };
  const std::vector<float> correct{1.f, 2.f, 3.f};
  require(max_rel_err(p, correct, weight) == 0, "exact finite output must match the oracle");
  require(max_rel_err(p, {1.f, 2.f, 4.f}, weight) > 2e-3,
          "incorrect finite output must exceed the oracle budget");
  for (float value :
       {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()}) {
    for (size_t column = 0; column < correct.size(); ++column) {
      auto got = correct;
      got[column] = value;
      bool rejected = false;
      try {
        (void)max_rel_err(p, got, weight);
      } catch (const std::runtime_error& e) {
        rejected = std::string(e.what()).find("non-finite output") != std::string::npos;
      }
      require(rejected, "non-finite output must not pass the oracle comparison");
    }
  }
}

DGPP_TEST(mma_gemv_fp8_matches_oracle_and_is_m_invariant) {
  const Problem p = make(30, 1024, 512, 5, 5, 0x9E3779B97F4A7C15ull);
  Dev d(p);
  // fp32 epilogue at m = 30 vs the oracle.
  dgpp::launch_mma_gemv_fp8_f32(d.act, p.k, d.w8, d.scales, d.outf, p.m, p.n, p.k, 0, p.rs, p.cs, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<float> full = fetch_f(d.outf, size_t(p.m) * p.n);
  const double err = max_rel_err(p, full, [&](int r, int c) { return wval8(p, r, c); });
  std::printf("[ .. ] mma_gemv fp8 m=30 n=%d k=%d: max rel err vs the double oracle %.3e\n", p.n, p.k, err);
  require(err < 2e-3, "fp8: outside the tile kernel's oracle budget");
  // The bf16 epilogue is the fp32 one rounded.
  dgpp::launch_mma_gemv_fp8_bf16(d.act, p.k, d.w8, d.scales, d.outb, p.m, p.n, p.k, 0, p.rs, p.cs, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<uint16_t> fullb = fetch_b(d.outb, size_t(p.m) * p.n);
  for (size_t i = 0; i < fullb.size(); ++i)
    require(fullb[i] == dgpp::float_to_bf16_bits(full[i]), "fp8: bf16 epilogue != bf16(f32 epilogue)");
  // m-invariance: every row alone (m = 1), and rows in m = 7 and m = 16
  // batches, bitwise the m = 30 result.
  for (int m2 : {1, 7, 16}) {
    for (int r0 = 0; r0 + m2 <= p.m; r0 += (m2 == 1 ? 7 : m2)) {
      dgpp::launch_mma_gemv_fp8_f32(d.act + size_t(r0) * p.k, p.k, d.w8, d.scales, d.outf, m2, p.n, p.k, 0, p.rs, p.cs, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const std::vector<float> part = fetch_f(d.outf, size_t(m2) * p.n);
      for (int i = 0; i < m2; ++i)
        require(std::memcmp(part.data() + size_t(i) * p.n, full.data() + size_t(r0 + i) * p.n, size_t(p.n) * 4) == 0,
                "fp8: row " + std::to_string(r0 + i) + " differs at m=" + std::to_string(m2) + " from m=30");
    }
  }
  std::printf("[ OK ] mma_gemv fp8: bitwise m-invariant (m = 1, 7, 16 vs 30), bf16 == bf16(f32)\n");
}

// The wide forms (4 and 8 tiles) and the 128-row grouping above them: the
// oracle at 200 rows, and every row bitwise its m = 1 result (the forms
// share one chain).
DGPP_TEST(mma_gemv_fp8_wide_forms_match_oracle_and_the_scalar_form) {
  const Problem p = make(200, 512, 1024, 5, 5, 0x2545F4914F6CDD1Dull);
  Dev d(p);
  dgpp::launch_mma_gemv_fp8_f32(d.act, p.k, d.w8, d.scales, d.outf, p.m, p.n, p.k, 0, p.rs, p.cs, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<float> full = fetch_f(d.outf, size_t(p.m) * p.n);
  const double err = max_rel_err(p, full, [&](int r, int c) { return wval8(p, r, c); });
  std::printf("[ .. ] mma_gemv fp8 m=200 (128 + 72 rows) n=%d k=%d: max rel err %.3e\n", p.n, p.k, err);
  require(err < 2e-3, "fp8 wide: outside the oracle budget");
  for (int m2 : {1, 40, 64, 100}) {
    for (int r0 = 0; r0 + m2 <= p.m; r0 += (m2 == 1 ? 37 : m2)) {
      dgpp::launch_mma_gemv_fp8_f32(d.act + size_t(r0) * p.k, p.k, d.w8, d.scales, d.outf, m2, p.n, p.k, 0, p.rs, p.cs, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const std::vector<float> part = fetch_f(d.outf, size_t(m2) * p.n);
      for (int i = 0; i < m2; ++i)
        require(std::memcmp(part.data() + size_t(i) * p.n, full.data() + size_t(r0 + i) * p.n, size_t(p.n) * 4) == 0,
                "fp8 wide: row " + std::to_string(r0 + i) + " differs at m=" + std::to_string(m2) + " from m=200");
    }
  }
  std::printf("[ OK ] mma_gemv fp8: the 1/2/4/8-tile forms and the grouped launch are bitwise one chain\n");
  // The fixture-sized shapes (k under one 512-k window, small n) at every form.
  for (const auto [n2, k2] : {std::pair{64, 128}, std::pair{128, 256}, std::pair{256, 64}, std::pair{320, 448}}) {
    for (int m2 : {1, 24, 40, 70, 100, 130}) {
      const Problem q = make(m2, n2, k2, 5, 5, 0xA0761D6478BD642Full + m2 + n2);
      Dev e(q);
      dgpp::launch_mma_gemv_fp8_f32(e.act, q.k, e.w8, e.scales, e.outf, q.m, q.n, q.k, 0, q.rs, q.cs, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const std::vector<float> got = fetch_f(e.outf, size_t(q.m) * q.n);
      const double e2 = max_rel_err(q, got, [&](int r, int c) { return wval8(q, r, c); });
      require(e2 < 2e-3, "fp8 shape n=" + std::to_string(n2) + " k=" + std::to_string(k2) + " m=" + std::to_string(m2) +
                             ": max rel err " + std::to_string(e2));
    }
  }
  std::printf("[ OK ] mma_gemv fp8: fixture-sized shapes at every form match the oracle\n");
}

// The wide forms' bf16 epilogue (the shared expert aside at sixteen DSpark
// slots is an 80-row fp8 -> bf16 launch, 2026-10-23): bf16(f32) at 4 and 8
// tiles, and every row bitwise its m = 64 / m = 16 launch.
DGPP_TEST(mma_gemv_fp8_wide_bf16_epilogue_matches_f32_and_is_m_invariant) {
  for (int m : {80, 100}) {
    const Problem p = make(m, 512, 1024, 5, 5, 0x6A09E667F3BCC909ull + m);
    Dev d(p);
    dgpp::launch_mma_gemv_fp8_f32(d.act, p.k, d.w8, d.scales, d.outf, p.m, p.n, p.k, 0, p.rs, p.cs, nullptr);
    dgpp::launch_mma_gemv_fp8_bf16(d.act, p.k, d.w8, d.scales, d.outb, p.m, p.n, p.k, 0, p.rs, p.cs, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const std::vector<float> full = fetch_f(d.outf, size_t(p.m) * p.n);
    const std::vector<uint16_t> fullb = fetch_b(d.outb, size_t(p.m) * p.n);
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < fullb.size(); ++i)
      if (fullb[i] != dgpp::float_to_bf16_bits(full[i])) { if (!bad) first = i; ++bad; }
    if (bad) std::printf("[ .. ] fp8 wide bf16 m=%d: %zu of %zu elements differ from bf16(f32); first at row %zu col %zu\n", m, bad, fullb.size(), first / p.n, first % p.n);
    require(bad == 0, "fp8 wide bf16 epilogue != bf16(f32 epilogue) at m=" + std::to_string(m));
    for (int m2 : {16, 64}) {
      for (int r0 = 0; r0 + m2 <= p.m; r0 += m2) {
        dgpp::launch_mma_gemv_fp8_bf16(d.act + size_t(r0) * p.k, p.k, d.w8, d.scales, d.outb, m2, p.n, p.k, 0, p.rs, p.cs, nullptr);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        const std::vector<uint16_t> part = fetch_b(d.outb, size_t(m2) * p.n);
        for (int i = 0; i < m2; ++i)
          require(std::memcmp(part.data() + size_t(i) * p.n, fullb.data() + size_t(r0 + i) * p.n, size_t(p.n) * 2) == 0,
                  "fp8 wide bf16: row " + std::to_string(r0 + i) + " differs at m=" + std::to_string(m2) + " from m=" + std::to_string(m));
      }
    }
    std::printf("[ OK ] mma_gemv fp8 wide bf16 epilogue at m=%d: bf16(f32), bitwise the 16/64-row launches\n", m);
  }
}

// With a split workspace, every tile form keeps one chain: rows of an
// 80-row launch (the eight-tile form, 2026-10-23) bitwise their 5-row and
// 40-row launches at a small n that splits (n = 128 -> 2 blocks at the
// decode width).
DGPP_TEST(mma_gemv_fp8_split_chain_is_the_same_at_80_rows) {
  const Problem p = make(80, 128, 2048, 5, 5, 0xBB67AE8584CAA73Bull);
  Dev d(p);
  const size_t ws_bytes = size_t(64) << 20;
  void* ws = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&ws, ws_bytes));
  dgpp::launch_mma_gemv_fp8_bf16(d.act, p.k, d.w8, d.scales, d.outb, p.m, p.n, p.k, 0, p.rs, p.cs, nullptr, ws, ws_bytes);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<uint16_t> full = fetch_b(d.outb, size_t(p.m) * p.n);
  for (int m2 : {5, 40, 64}) {
    for (int r0 = 0; r0 + m2 <= p.m; r0 += m2) {
      dgpp::launch_mma_gemv_fp8_bf16(d.act + size_t(r0) * p.k, p.k, d.w8, d.scales, d.outb, m2, p.n, p.k, 0, p.rs, p.cs, nullptr, ws, ws_bytes);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const std::vector<uint16_t> part = fetch_b(d.outb, size_t(m2) * p.n);
      for (int i = 0; i < m2; ++i)
        require(std::memcmp(part.data() + size_t(i) * p.n, full.data() + size_t(r0 + i) * p.n, size_t(p.n) * 2) == 0,
                "split chain: row " + std::to_string(r0 + i) + " differs at m=" + std::to_string(m2) + " from m=80");
    }
  }
  cudaFree(ws);
  std::printf("[ OK ] mma_gemv fp8 with a split workspace: the 80-row launch's rows are bitwise their 5/40/64-row launches\n");
}

DGPP_TEST(mma_gemv_bf16_matches_oracle_and_is_m_invariant) {
  const Problem p = make(30, 1024, 512, 7, 7, 0xD1B54A32D192ED03ull);
  Dev d(p);
  dgpp::launch_mma_gemv_bf16_f32(d.act, p.k, d.w16, d.outf, p.m, p.n, p.k, 0, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<float> full = fetch_f(d.outf, size_t(p.m) * p.n);
  const double err = max_rel_err(p, full, [&](int r, int c) { return dgpp::bf16_bits_to_float(p.w16[size_t(r) * p.k + c]); });
  std::printf("[ .. ] mma_gemv bf16 m=30: max rel err vs the double oracle %.3e\n", err);
  require(err < 2e-3, "bf16: outside the oracle budget");
  for (int m2 : {1, 16}) {
    for (int r0 = 0; r0 + m2 <= p.m; r0 += (m2 == 1 ? 11 : m2)) {
      dgpp::launch_mma_gemv_bf16_f32(d.act + size_t(r0) * p.k, p.k, d.w16, d.outf, m2, p.n, p.k, 0, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const std::vector<float> part = fetch_f(d.outf, size_t(m2) * p.n);
      for (int i = 0; i < m2; ++i)
        require(std::memcmp(part.data() + size_t(i) * p.n, full.data() + size_t(r0 + i) * p.n, size_t(p.n) * 4) == 0,
                "bf16: row differs across m");
    }
  }
  std::printf("[ OK ] mma_gemv bf16: bitwise m-invariant\n");
}

// The dense tensor-core GEMM over bf16 weights (kernels/dense_mma_bf16w.hpp,
// the DeepSeek-V4-Flash prefill's compressor projections): inside the
// double oracle's budget on a ragged shape (m and n off the tile sizes),
// bitwise the same row whatever rows share the launch — a chunked prefill's
// contract — and the bf16 epilogue the rounded fp32 one.
DGPP_TEST(dense_mma_bf16w_matches_oracle_and_is_row_invariant) {
  const Problem p = make(150, 344, 512, 7, 7, 0x9E3779B97F4A7C15ull);
  Dev d(p);
  require(dgpp::dense_mma_bf16w_shape_ok(d.w16, p.k), "the shape the kernel takes");
  dgpp::launch_dense_mma_bf16w_f32(d.act, p.k, d.w16, d.outf, p.m, p.n, p.k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<float> full = fetch_f(d.outf, size_t(p.m) * p.n);
  const double err = max_rel_err(p, full, [&](int r, int c) { return dgpp::bf16_bits_to_float(p.w16[size_t(r) * p.k + c]); });
  std::printf("[ .. ] dense_mma_bf16w [150 x 344 x 512]: max rel err vs the double oracle %.3e\n", err);
  require(err < 2e-3, "dense bf16w: outside the oracle budget");
  // Any chunking of the rows gives the same rows.
  for (int chunk : {1, 37, 64, 101}) {
    for (int r0 = 0; r0 < p.m; r0 += (chunk == 1 ? 29 : chunk)) {
      const int rows = std::min(chunk, p.m - r0);
      dgpp::launch_dense_mma_bf16w_f32(d.act + size_t(r0) * p.k, p.k, d.w16, d.outf, rows, p.n, p.k, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const std::vector<float> part = fetch_f(d.outf, size_t(rows) * p.n);
      for (int i = 0; i < rows; ++i)
        require(std::memcmp(part.data() + size_t(i) * p.n, full.data() + size_t(r0 + i) * p.n, size_t(p.n) * 4) == 0,
                "dense bf16w: a row differs across chunkings");
    }
  }
  dgpp::launch_dense_mma_bf16w_bf16(d.act, p.k, d.w16, d.outb, p.m, p.n, p.k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<uint16_t> fb = fetch_b(d.outb, size_t(p.m) * p.n);
  for (size_t i = 0; i < fb.size(); ++i)
    require(fb[i] == dgpp::float_to_bf16_bits(full[i]), "dense bf16w: the bf16 epilogue is the rounded fp32 one");
  std::printf("[ OK ] dense_mma_bf16w: row-invariant across chunkings, epilogues agree\n");
}

// The swiglu form: bitwise the clamp launch followed by the plain fp8 GEMV.
DGPP_TEST(mma_gemv_fp8_swiglu_form_is_bitwise_the_clamp_then_the_gemv) {
  const Problem p = make(100, 1024, 512, 7, 7, 0x7F4A7C159E3779B9ull);  // rows for the 8-tile form too
  Dev d(p);
  // gate = the problem's activations, up = a second set (the weights' first rows reused as bf16 values).
  uint16_t* up; uint16_t* act;
  DGPP_CUDA_OK(cudaMalloc(&up, size_t(p.m) * p.k * 2)); DGPP_CUDA_OK(cudaMalloc(&act, size_t(p.m) * p.k * 2));
  DGPP_CUDA_OK(cudaMemcpy(up, p.w16.data(), size_t(p.m) * p.k * 2, cudaMemcpyHostToDevice));
  for (const float limit : {10.0f, 0.25f}) {
    for (int m : {1, 4, 6, 80, 100}) {
      dgpp::launch_moe_swiglu_clamp(d.act, up, act, int64_t(m) * p.k, limit, nullptr);
      dgpp::launch_mma_gemv_fp8_f32(act, p.k, d.w8, d.scales, d.outf, m, p.n, p.k, 0, p.rs, p.cs, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const std::vector<float> want = fetch_f(d.outf, size_t(m) * p.n);
      DGPP_CUDA_OK(cudaMemset(d.outf, 0xA5, size_t(m) * p.n * 4));
      dgpp::launch_mma_gemv_fp8_swiglu_f32(d.act, up, limit, p.k, d.w8, d.scales, d.outf, m, p.n, p.k, 0, p.rs, p.cs, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const std::vector<float> got = fetch_f(d.outf, size_t(m) * p.n);
      require(std::memcmp(want.data(), got.data(), want.size() * 4) == 0, "swiglu form: differs from the clamp launch + the GEMV");
    }
  }
  cudaFree(up); cudaFree(act);
  std::printf("[ OK ] mma_gemv swiglu form: bitwise the clamp launch then the GEMV (m = 1, 4, 6; two limits)\n");
}

// The gated ring form (the lazy ratio-128 compressors): a row that
// completes a group gets the group's rows off its request's ring, bitwise
// the plain launch on the same rows; every other row's output is untouched.
DGPP_TEST(dense_mma_bf16w_groups_are_the_plain_rows_and_gated) {
  const int requests = 3, slots = 104, group = 96, n = 344, k = 512;
  const Problem p = make(requests * slots, n, k, 7, 7, 0xC2B2AE3D27D4EB4Full);  // act: the ring [requests][slots][k]
  Dev d(p);
  struct Row { int req; int64_t pos; };
  // (req 0, 95): the first group, slots 0..95.  (req 1, 191): the second,
  // positions 96..191 = slots 96..103, 0..87 (the wrap).  The rest: no group.
  const std::vector<Row> rows = {{0, 95}, {0, 96}, {1, 191}, {2, -1}, {2, 94}};
  const int R = int(rows.size());
  std::vector<int32_t> req(R); std::vector<int64_t> pos(R);
  for (int i = 0; i < R; ++i) { req[size_t(i)] = rows[size_t(i)].req; pos[size_t(i)] = rows[size_t(i)].pos; }
  int32_t* dreq; int64_t* dpos; float* dout; uint16_t* dgather; float* dref;
  DGPP_CUDA_OK(cudaMalloc(&dreq, size_t(R) * 4)); DGPP_CUDA_OK(cudaMalloc(&dpos, size_t(R) * 8));
  DGPP_CUDA_OK(cudaMalloc(&dout, size_t(R) * group * n * 4)); DGPP_CUDA_OK(cudaMalloc(&dgather, size_t(group) * k * 2));
  DGPP_CUDA_OK(cudaMalloc(&dref, size_t(group) * n * 4));
  DGPP_CUDA_OK(cudaMemcpy(dreq, req.data(), size_t(R) * 4, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(dpos, pos.data(), size_t(R) * 8, cudaMemcpyHostToDevice));
  std::vector<float> sentinel(size_t(R) * group * n, -777.25f);
  DGPP_CUDA_OK(cudaMemcpy(dout, sentinel.data(), sentinel.size() * 4, cudaMemcpyHostToDevice));
  dgpp::launch_dense_mma_bf16w_groups_f32(d.act, slots, dreq, dpos, R, group, d.w16, dout, n, k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<float> got = fetch_f(dout, size_t(R) * group * n);
  int completed = 0;
  for (int i = 0; i < R; ++i) {
    const float* mine = got.data() + size_t(i) * group * n;
    if (pos[size_t(i)] < 0 || (pos[size_t(i)] + 1) % group != 0) {
      for (size_t e = 0; e < size_t(group) * n; ++e) require(mine[e] == -777.25f, "groups: a row without a group wrote");
      continue;
    }
    ++completed;
    std::vector<uint16_t> gathered(size_t(group) * k);
    for (int t = 0; t < group; ++t) {
      const int64_t slot = (pos[size_t(i)] - group + 1 + t) % slots;
      std::memcpy(gathered.data() + size_t(t) * k, p.act.data() + (size_t(req[size_t(i)]) * slots + size_t(slot)) * k, size_t(k) * 2);
    }
    DGPP_CUDA_OK(cudaMemcpy(dgather, gathered.data(), gathered.size() * 2, cudaMemcpyHostToDevice));
    dgpp::launch_dense_mma_bf16w_f32(dgather, k, d.w16, dref, group, n, k, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const std::vector<float> ref = fetch_f(dref, size_t(group) * n);
    require(std::memcmp(mine, ref.data(), ref.size() * 4) == 0, "groups: a group differs from the plain launch on its rows");
  }
  require(completed == 2, "groups: the test's two completing rows");
  cudaFree(dreq); cudaFree(dpos); cudaFree(dout); cudaFree(dgather); cudaFree(dref);
  std::printf("[ OK ] dense_mma_bf16w groups: bitwise the plain rows (a wrapped ring included), gated rows untouched\n");
}

DGPP_TEST(mma_gemv_cold_timing_beside_the_gemv_cores) {
  // Informational: the dense projection shape [5120 x 5120] fp8 (32 x 32
  // scales) and the head shape [32320 x 5120] bf16, cold (weights cycled
  // past the L2), at m = 1, 6, 16, 30.
  const int n = 5120, k = 5120, copies = 24;
  std::vector<uint8_t*> w(copies); std::vector<float*> s(copies);
  const size_t sb = size_t(n / 32) * (k / 32) * 4;
  std::vector<float> hs(sb / 4, 0.5f);
  for (int c = 0; c < copies; ++c) { DGPP_CUDA_OK(cudaMalloc(&w[c], size_t(n) * k)); DGPP_CUDA_OK(cudaMemset(w[c], 0x38, size_t(n) * k)); DGPP_CUDA_OK(cudaMalloc(&s[c], sb)); DGPP_CUDA_OK(cudaMemcpy(s[c], hs.data(), sb, cudaMemcpyHostToDevice)); }
  uint16_t* act; DGPP_CUDA_OK(cudaMalloc(&act, size_t(2048) * k * 2)); DGPP_CUDA_OK(cudaMemset(act, 0x3f, size_t(2048) * k * 2));
  uint16_t* out; DGPP_CUDA_OK(cudaMalloc(&out, size_t(2048) * n * 2));
  cudaEvent_t e0, e1; DGPP_CUDA_OK(cudaEventCreate(&e0)); DGPP_CUDA_OK(cudaEventCreate(&e1));
  std::printf("[ .. ] fp8 [%d x %d] cold, us per product: m  gemv-chunks  mma-stream\n", n, k);
  for (int m : {1, 2, 3, 4, 6, 16, 30}) {
    float t[2];
    for (int path = 0; path < 2; ++path) {
      for (int i = 0; i < 4; ++i) { if (path == 0) dgpp::launch_scale_gemm_grid_bf16(act, k, w[i], s[i], out, m, n, k, nullptr, 0, 5, 5); else dgpp::launch_mma_gemv_fp8_bf16(act, k, w[i], s[i], out, m, n, k, 0, 5, 5, nullptr); }
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      DGPP_CUDA_OK(cudaEventRecord(e0));
      for (int i = 0; i < 48; ++i) { const int c = i % copies; if (path == 0) dgpp::launch_scale_gemm_grid_bf16(act, k, w[c], s[c], out, m, n, k, nullptr, 0, 5, 5); else dgpp::launch_mma_gemv_fp8_bf16(act, k, w[c], s[c], out, m, n, k, 0, 5, 5, nullptr); }
      DGPP_CUDA_OK(cudaEventRecord(e1)); DGPP_CUDA_OK(cudaEventSynchronize(e1));
      float ms = 0; DGPP_CUDA_OK(cudaEventElapsedTime(&ms, e0, e1)); t[path] = ms * 1000.f / 48;
    }
    std::printf("[ .. ]   m=%2d  %8.1f us  %8.1f us  (%.0f GB/s of weights on the mma stream)\n", m, t[0], t[1], double(n) * k / t[1] / 1e3);
  }
  // The prefill's rows: the chunk kernels to 128 rows, the tile kernel above.
  std::printf("[ .. ] fp8 [%d x %d] cold, us per product: m  chunks/tile  mma-forms\n", n, k);
  // The full GLM-5.3 DSA fp8 shapes at one rank of four (kv_a, q_a, q_b, o_proj):
  // the streaming form's block count is n / 64 (a small-n site fills few SMs).
  for (auto [sn, sk] : std::vector<std::pair<int, int>>{{576, 6144}, {2048, 6144}, {4096, 2048}, {5120, 5120}, {6144, 4096}, {12288, 5120}, {32320, 5120}}) {
    uint8_t* sw[4]; float* ss[4];
    const int sc = ((sn + 127) / 128) * ((sk + 127) / 128);
    for (int i = 0; i < 4; ++i) {
      DGPP_CUDA_OK(cudaMalloc(&sw[i], static_cast<size_t>(sn) * sk)); DGPP_CUDA_OK(cudaMemset(sw[i], 0x38, static_cast<size_t>(sn) * sk));
      DGPP_CUDA_OK(cudaMalloc(&ss[i], static_cast<size_t>(sc) * 4)); DGPP_CUDA_OK(cudaMemset(ss[i], 0, static_cast<size_t>(sc) * 4));
    }
    for (int m : {1, 16}) {
      float t[6];
      for (int path = 0; path < 6; ++path) {
        // path 0: the chunks; 1..5: the mma at widths 8, 4, 2, 1 warps and the rule.
        dgpp::mma_gemv_set_decode_width(path == 1 ? 8 : path == 2 ? 4 : path == 3 ? 2 : path == 4 ? 1 : 0);
        auto run = [&](int i) { if (path == 0) dgpp::launch_scale_gemm_bf16(act, sk, sw[i], ss[i], out, m, sn, sk, nullptr, 0); else dgpp::launch_mma_gemv_fp8_bf16(act, sk, sw[i], ss[i], out, m, sn, sk, 0, 7, 7, nullptr); };
        for (int i = 0; i < 4; ++i) run(i);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        DGPP_CUDA_OK(cudaEventRecord(e0));
        for (int i = 0; i < 16; ++i) run(i % 4);
        DGPP_CUDA_OK(cudaEventRecord(e1)); DGPP_CUDA_OK(cudaEventSynchronize(e1));
        float ms = 0; DGPP_CUDA_OK(cudaEventElapsedTime(&ms, e0, e1)); t[path] = ms * 1000.f / 16;
      }
      dgpp::mma_gemv_set_decode_width(0);
      std::printf("[ .. ]   fp8 [%d x %d] m=%3d  chunks %8.1f us  mma w8 %8.1f  w4 %8.1f  w2 %8.1f  w1 %8.1f  rule %8.1f\n", sn, sk, m, t[0], t[1], t[2], t[3], t[4], t[5]);
    }
    for (int i = 0; i < 4; ++i) { cudaFree(sw[i]); cudaFree(ss[i]); }
  }
  // The non-grid scale GEMM (the session-core families' dense sites): the
  // GEMV chunk to 128 rows, the 128-row dense tensor-core kernel above.
  for (int m : {30, 64, 128, 256, 512, 2048}) {
    float t[2];
    for (int path = 0; path < 2; ++path) {
      for (int i = 0; i < 4; ++i) { if (path == 0) dgpp::launch_scale_gemm_bf16(act, k, w[i], s[i], out, m, n, k, nullptr, 0); else dgpp::launch_mma_gemv_fp8_bf16(act, k, w[i], s[i], out, m, n, k, 0, 7, 7, nullptr); }
      DGPP_CUDA_OK(cudaEventRecord(e0)); 
      for (int i = 0; i < 4; ++i) { if (path == 0) dgpp::launch_scale_gemm_bf16(act, k, w[i], s[i], out, m, n, k, nullptr, 0); else dgpp::launch_mma_gemv_fp8_bf16(act, k, w[i], s[i], out, m, n, k, 0, 7, 7, nullptr); }
      DGPP_CUDA_OK(cudaEventRecord(e1)); DGPP_CUDA_OK(cudaEventSynchronize(e1));
      DGPP_CUDA_OK(cudaEventElapsedTime(&t[path], e0, e1)); t[path] = t[path] * 1000.f / 4;
    }
    std::printf("[ .. ]   non-grid scale GEMM vs mma  m=%4d  %9.1f us  %9.1f us\n", m, t[0], t[1]);
  }
  for (int m : {64, 128, 256, 2048}) {
    float t[2];
    const int reps = m > 256 ? 6 : 24;
    for (int path = 0; path < 2; ++path) {
      for (int i = 0; i < 2; ++i) { if (path == 0) dgpp::launch_scale_gemm_grid_bf16(act, k, w[i], s[i], out, m, n, k, nullptr, 0, 5, 5); else dgpp::launch_mma_gemv_fp8_bf16(act, k, w[i], s[i], out, m, n, k, 0, 5, 5, nullptr); }
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      DGPP_CUDA_OK(cudaEventRecord(e0));
      for (int i = 0; i < reps; ++i) { const int c = i % copies; if (path == 0) dgpp::launch_scale_gemm_grid_bf16(act, k, w[c], s[c], out, m, n, k, nullptr, 0, 5, 5); else dgpp::launch_mma_gemv_fp8_bf16(act, k, w[c], s[c], out, m, n, k, 0, 5, 5, nullptr); }
      DGPP_CUDA_OK(cudaEventRecord(e1)); DGPP_CUDA_OK(cudaEventSynchronize(e1));
      float ms = 0; DGPP_CUDA_OK(cudaEventElapsedTime(&ms, e0, e1)); t[path] = ms * 1000.f / reps;
    }
    std::printf("[ .. ]   m=%4d  %8.1f us  %8.1f us  (%.1f TFLOP/s on the mma forms)\n", m, t[0], t[1], 2.0 * m * n * k / t[1] / 1e6);
  }
  for (int c = 0; c < copies; ++c) { cudaFree(w[c]); cudaFree(s[c]); }
  // The head.
  const int hn = 32320; const int hc = 3;
  std::vector<uint16_t*> hw(hc); for (int c = 0; c < hc; ++c) { DGPP_CUDA_OK(cudaMalloc(&hw[c], size_t(hn) * k * 2)); DGPP_CUDA_OK(cudaMemset(hw[c], 0x3c, size_t(hn) * k * 2)); }
  float* outf; DGPP_CUDA_OK(cudaMalloc(&outf, size_t(32) * hn * 4));
  std::printf("[ .. ] bf16 head [%d x %d] cold, us per full product: m  gemv-chunks  mma-stream\n", hn, k);
  for (int m : {1, 6, 16, 30}) {
    float t[2];
    for (int path = 0; path < 2; ++path) {
      DGPP_CUDA_OK(cudaEventRecord(e0));
      for (int i = 0; i < 9; ++i) { const int c = i % hc;
        if (path == 0) { for (int r0 = 0; r0 < m; r0 += 4) { const int rows = std::min(4, m - r0); dgpp::launch_bf16_gemv(act + size_t(r0) * k, k, hw[c], outf + size_t(r0) * hn, true, rows, hn, k, nullptr); } }
        else dgpp::launch_mma_gemv_bf16_f32(act, k, hw[c], outf, m, hn, k, 0, nullptr); }
      DGPP_CUDA_OK(cudaEventRecord(e1)); DGPP_CUDA_OK(cudaEventSynchronize(e1));
      float ms = 0; DGPP_CUDA_OK(cudaEventElapsedTime(&ms, e0, e1)); t[path] = ms * 1000.f / 9;
    }
    std::printf("[ .. ]   m=%2d  %8.1f us  %8.1f us\n", m, t[0], t[1]);
  }
  for (int c = 0; c < hc; ++c) cudaFree(hw[c]);
  cudaFree(outf); cudaFree(act); cudaFree(out);
}

// The decode form at one to four rows on the Qwen3.8-27B shard shapes, cold,
// against the GEMV chunks (the chain the plain T=1 world left for exactness,
// 2026-10-05): the rule's width without and with the split-K workspace (the
// model passes one), per rank of four (narrow) and of one (wide).
DGPP_TEST(mma_gemv_small_rows_on_the_shard_shapes) {
  void* ws = nullptr; const size_t ws_bytes = 64u << 20;
  DGPP_CUDA_OK(cudaMalloc(&ws, ws_bytes));
  uint16_t* act; DGPP_CUDA_OK(cudaMalloc(&act, size_t(8) * 17408 * 2)); DGPP_CUDA_OK(cudaMemset(act, 0x3f, size_t(8) * 17408 * 2));
  uint16_t* out; DGPP_CUDA_OK(cudaMalloc(&out, size_t(8) * 65536 * 2));
  cudaEvent_t e0, e1; DGPP_CUDA_OK(cudaEventCreate(&e0)); DGPP_CUDA_OK(cudaEventCreate(&e1));
  struct Shape { const char* site; int n, k; };
  const Shape shapes[] = {{"w4 gdn qkv", 2560, 5120}, {"w4 gdn z", 1536, 5120}, {"w4 gdn out", 5120, 1536}, {"w4 attn q", 1024, 5120},
                          {"w4 attn kv", 128, 5120}, {"w4 attn o", 5120, 1024}, {"w4 gate/up", 4352, 5120}, {"w4 down", 5120, 4352},
                          {"w1 gdn qkv", 10240, 5120}, {"w1 attn q", 4096, 5120}, {"w1 attn kv", 512, 5120}, {"w1 attn o", 5120, 4096},
                          {"w1 gate/up", 17408, 5120}, {"w1 down", 5120, 17408}};
  std::printf("[ .. ] fp8 decode form at 1..4 rows, cold us per product: site  m  chunks  rule  rule+ws(split)\n");
  for (const Shape& sh : shapes) {
    const size_t bytes = size_t(sh.n) * sh.k;
    const int copies = static_cast<int>(std::max<size_t>(2, std::min<size_t>(48, (64u << 20) / bytes + 1)));
    std::vector<uint8_t*> w(copies); std::vector<float*> sc(copies);
    const size_t sb = size_t((sh.n + 127) / 128) * ((sh.k + 127) / 128) * 4;
    for (int c = 0; c < copies; ++c) { DGPP_CUDA_OK(cudaMalloc(&w[c], bytes)); DGPP_CUDA_OK(cudaMemset(w[c], 0x38, bytes)); DGPP_CUDA_OK(cudaMalloc(&sc[c], sb)); DGPP_CUDA_OK(cudaMemset(sc[c], 0, sb)); }
    for (int m : {1, 2, 4}) {
      float t[3];
      for (int path = 0; path < 3; ++path) {
        auto run = [&](int i) {
          if (path == 0) dgpp::launch_scale_gemm_bf16(act, sh.k, w[i], sc[i], out, m, sh.n, sh.k, nullptr, 0);
          else dgpp::launch_mma_gemv_fp8_bf16(act, sh.k, w[i], sc[i], out, m, sh.n, sh.k, 0, 7, 7, nullptr, path == 2 ? ws : nullptr, path == 2 ? ws_bytes : 0);
        };
        for (int i = 0; i < copies; ++i) run(i);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        const int reps = copies * 2;
        DGPP_CUDA_OK(cudaEventRecord(e0));
        for (int i = 0; i < reps; ++i) run(i % copies);
        DGPP_CUDA_OK(cudaEventRecord(e1)); DGPP_CUDA_OK(cudaEventSynchronize(e1));
        float ms = 0; DGPP_CUDA_OK(cudaEventElapsedTime(&ms, e0, e1)); t[path] = ms * 1000.f / reps;
      }
      std::printf("[ .. ]   %-11s [%5d x %5d] m=%d  %7.1f  %7.1f  %7.1f us  (%.0f GB/s split)\n", sh.site, sh.n, sh.k, m, t[0], t[1], t[2], bytes / t[2] / 1e3);
    }
    for (int c = 0; c < copies; ++c) { cudaFree(w[c]); cudaFree(sc[c]); }
  }
  cudaFree(ws); cudaFree(act); cudaFree(out);
}

// One chain at every decode row count (2026-10-05): with a workspace the
// split ranges are 256-k units (a window boundary of every tile form) and
// the 33..64-row form splits like the 1..32-row forms, so a request's rows
// in an eight-slot batch are bitwise its rows alone — at the four-node
// gate / up site ([4352 x 5120]: 68 eight-warp blocks, split 2 on a 48-SM
// part), a narrow attention site and the hyperconnection down.
DGPP_TEST(mma_gemv_split_k_rows_are_one_chain_through_64) {
  void* ws = nullptr; const size_t ws_bytes = 8u << 20;
  DGPP_CUDA_OK(cudaMalloc(&ws, ws_bytes));
  for (auto [n, k] : std::vector<std::pair<int, int>>{{4352, 5120}, {576, 6144}, {320, 10240}}) {
    const Problem p = make(64, n, k, 7, 7, 0xC0FFEEull + n * 7 + k);
    Dev d(p);
    dgpp::launch_mma_gemv_fp8_f32(d.act, p.k, d.w8, d.scales, d.outf, 64, p.n, p.k, 0, p.rs, p.cs, nullptr, ws, ws_bytes);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const std::vector<float> full = fetch_f(d.outf, size_t(64) * p.n);
    Problem head = p; head.m = 4; head.act.resize(size_t(4) * p.k);  // the oracle over the first rows
    const double err = max_rel_err(head, full, [&](int r, int c) { return wval8(p, r, c); });
    require(err < 2e-3, "split-K at 64 rows [" + std::to_string(n) + " x " + std::to_string(k) + "] vs oracle: " + std::to_string(err));
    dgpp::launch_mma_gemv_fp8_f32(d.act, p.k, d.w8, d.scales, d.outf, 64, p.n, p.k, 0, p.rs, p.cs, nullptr, nullptr, 0);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const std::vector<float> unsplit = fetch_f(d.outf, size_t(64) * p.n);
    const bool split_taken = std::memcmp(full.data(), unsplit.data(), full.size() * 4) != 0;
    for (const int m : {1, 2, 4, 5, 8, 16, 17, 32, 33, 48, 63}) {
      dgpp::launch_mma_gemv_fp8_f32(d.act, p.k, d.w8, d.scales, d.outf, m, p.n, p.k, 0, p.rs, p.cs, nullptr, ws, ws_bytes);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const std::vector<float> got = fetch_f(d.outf, size_t(m) * p.n);
      require(std::memcmp(got.data(), full.data(), got.size() * 4) == 0,
              "split-K [" + std::to_string(n) + " x " + std::to_string(k) + "]: rows differ between m = " +
                  std::to_string(m) + " and m = 64");
    }
    std::printf("[ OK ] mma_gemv [%d x %d] with workspace: one chain at m = 1..64 (oracle %.3e, the split %s at this shape)\n",
                n, k, err, split_taken ? "taken" : "not taken");
  }
  DGPP_CUDA_OK(cudaFree(ws));
}

// Split-K at the small-n sites (2026-09-21): with a workspace the decode
// forms split the k range across blocks. Against the oracle, deterministic
// across launches, the bf16 epilogue bf16(f32), a row's result the same at
// m = 16 and inside m = 30 (the split is a function of the shape), and a
// cold timing line beside the unsplit form.
DGPP_TEST(mma_gemv_split_k_small_n_sites) {
  void* ws = nullptr; const size_t ws_bytes = 4u << 20;
  DGPP_CUDA_OK(cudaMalloc(&ws, ws_bytes));
  for (auto [n, k] : std::vector<std::pair<int, int>>{{320, 10240}, {320, 2560}, {256, 2560}, {640, 2560}}) {
    const Problem p = make(30, n, k, 7, 7, 0xA5A5A5A5ull + n * 131 + k);
    Dev d(p);
    dgpp::launch_mma_gemv_fp8_f32(d.act, p.k, d.w8, d.scales, d.outf, 30, p.n, p.k, 0, p.rs, p.cs, nullptr, ws, ws_bytes);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const std::vector<float> a = fetch_f(d.outf, size_t(30) * p.n);
    const double err = max_rel_err(p, a, [&](int r, int c) { return wval8(p, r, c); });
    require(err < 2e-3, "split-K fp8 [" + std::to_string(n) + " x " + std::to_string(k) + "] vs oracle: " + std::to_string(err));
    dgpp::launch_mma_gemv_fp8_f32(d.act, p.k, d.w8, d.scales, d.outf, 30, p.n, p.k, 0, p.rs, p.cs, nullptr, ws, ws_bytes);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const std::vector<float> b = fetch_f(d.outf, size_t(30) * p.n);
    require(std::memcmp(a.data(), b.data(), a.size() * 4) == 0, "split-K: two launches differ");
    dgpp::launch_mma_gemv_fp8_f32(d.act, p.k, d.w8, d.scales, d.outf, 16, p.n, p.k, 0, p.rs, p.cs, nullptr, ws, ws_bytes);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const std::vector<float> m16 = fetch_f(d.outf, size_t(16) * p.n);
    require(std::memcmp(a.data(), m16.data(), m16.size() * 4) == 0, "split-K: rows differ between m = 16 and m = 30");
    dgpp::launch_mma_gemv_fp8_bf16(d.act, p.k, d.w8, d.scales, d.outb, 30, p.n, p.k, 0, p.rs, p.cs, nullptr, ws, ws_bytes);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const std::vector<uint16_t> ob = fetch_b(d.outb, size_t(30) * p.n);
    for (size_t i = 0; i < ob.size(); ++i)
      require(ob[i] == dgpp::float_to_bf16_bits(a[i]), "split-K: bf16 epilogue != bf16(f32)");
    std::printf("[ OK ] mma_gemv split-K fp8 [%d x %d]: oracle %.3e, deterministic, m-invariant, bf16 == bf16(f32)\n", n, k, err);
  }
  // Cold timing (weights cycled past the 24 MB L2): unsplit vs split, m = 16 and 30.
  cudaEvent_t e0, e1; DGPP_CUDA_OK(cudaEventCreate(&e0)); DGPP_CUDA_OK(cudaEventCreate(&e1));
  uint16_t* act; DGPP_CUDA_OK(cudaMalloc(&act, size_t(32) * 10240 * 2)); DGPP_CUDA_OK(cudaMemset(act, 0x3f, size_t(32) * 10240 * 2));
  uint16_t* out; DGPP_CUDA_OK(cudaMalloc(&out, size_t(32) * 640 * 2));
  for (auto [n, k] : std::vector<std::pair<int, int>>{{320, 10240}, {320, 2560}, {256, 2560}, {640, 2560}}) {
    const size_t bytes = size_t(n) * k;
    const int copies = std::min<size_t>(64, (48u << 20) / bytes + 1);
    std::vector<uint8_t*> w(copies); std::vector<float*> sc(copies);
    const size_t sb = size_t((n + 127) / 128) * ((k + 127) / 128) * 4;
    for (int c = 0; c < copies; ++c) {
      DGPP_CUDA_OK(cudaMalloc(&w[c], bytes)); DGPP_CUDA_OK(cudaMemset(w[c], 0x38, bytes));
      DGPP_CUDA_OK(cudaMalloc(&sc[c], sb)); DGPP_CUDA_OK(cudaMemset(sc[c], 0, sb));
    }
    for (int m : {16, 30}) {
      float t[2];
      for (int path = 0; path < 2; ++path) {
        auto run = [&](int i) {
          dgpp::launch_mma_gemv_fp8_bf16(act, k, w[i], sc[i], out, m, n, k, 0, 7, 7, nullptr,
                                         path ? ws : nullptr, path ? ws_bytes : 0);
        };
        for (int i = 0; i < copies; ++i) run(i);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        const int reps = copies * 3;
        DGPP_CUDA_OK(cudaEventRecord(e0));
        for (int i = 0; i < reps; ++i) run(i % copies);
        DGPP_CUDA_OK(cudaEventRecord(e1)); DGPP_CUDA_OK(cudaEventSynchronize(e1));
        float ms = 0; DGPP_CUDA_OK(cudaEventElapsedTime(&ms, e0, e1)); t[path] = ms * 1000.f / reps;
      }
      std::printf("[ .. ]   fp8 [%d x %d] m=%2d cold: unsplit %7.1f us (%.0f GB/s)  split-K %7.1f us (%.0f GB/s)\n",
                  n, k, m, t[0], bytes / t[0] / 1e3, t[1], bytes / t[1] / 1e3);
    }
    for (int c = 0; c < copies; ++c) { cudaFree(w[c]); cudaFree(sc[c]); }
  }
  cudaFree(act); cudaFree(out); cudaFree(ws);
}

// The bf12 form (the lm head's 12-bit companion under the tensor-core
// chain): bitwise the bf16 launch at every decode width, f32 and bf16
// outputs, on a matrix with escapes and rows the encoder kept raw; a
// partial last block; k of one and five super-blocks. The form declines a
// tail (k not whole super-blocks) and the widths above the decode forms.
DGPP_TEST(mma_gemv_bf12_form_is_bitwise_the_bf16_form) {
  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<float> uf(-1.f, 1.f);
  std::uniform_int_distribution<int> ex(-4, 4);
  std::uniform_int_distribution<int> outlier(0, 199);
  std::uniform_real_distribution<float> mag(0.5f, 1.f);
  for (auto [n, k] : std::vector<std::pair<int, int>>{{1032, 1024}, {4104, 5120}}) {
    std::vector<uint16_t> w(size_t(n) * k);
    // Nine exponents inside the row's fifteen-code window, one element in
    // two hundred far outside it: escapes on most rows, under the row budget.
    for (auto& x : w)
      x = dgpp::float_to_bf16_bits(std::ldexp(uf(rng) < 0 ? -mag(rng) : mag(rng), ex(rng) + (outlier(rng) == 0 ? 24 : 0)));
    // Two rows past the escape budget: kept bf16 by the encoder.
    for (int r : {5, n - 1})
      for (int c = 0; c < k; ++c) w[size_t(r) * k + c] = dgpp::float_to_bf16_bits(std::ldexp(uf(rng), (c & 1) ? 20 : -30));
    const dgpp::Bf12Host h = dgpp::bf12_encode(w.data(), n, k);
    require(h.ok && h.raw_rows >= 2 && h.escapes > 0,
            "bf12 form: the fixture packs with escapes and raw rows (ok " + std::to_string(h.ok) + " raw " +
                std::to_string(h.raw_rows) + " escapes " + std::to_string(h.escapes) + ")");
    uint16_t* dw; uint8_t* dp; uint32_t* dr; uint32_t* de; uint16_t* draw;
    DGPP_CUDA_OK(cudaMalloc(&dw, w.size() * 2)); DGPP_CUDA_OK(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMalloc(&dp, h.packed.size())); DGPP_CUDA_OK(cudaMemcpy(dp, h.packed.data(), h.packed.size(), cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMalloc(&dr, h.rows.size() * 4)); DGPP_CUDA_OK(cudaMemcpy(dr, h.rows.data(), h.rows.size() * 4, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMalloc(&de, std::max<size_t>(h.esc.size(), 1) * 4)); DGPP_CUDA_OK(cudaMemcpy(de, h.esc.data(), h.esc.size() * 4, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMalloc(&draw, std::max<size_t>(h.raw.size(), 8) * 2)); DGPP_CUDA_OK(cudaMemcpy(draw, h.raw.data(), h.raw.size() * 2, cudaMemcpyHostToDevice));
    dgpp::Bf12Matrix pm; pm.packed = dp; pm.rows = dr; pm.esc = de; pm.raw = draw; pm.n = n; pm.k = k;
    std::vector<uint16_t> act(size_t(32) * k);
    for (auto& x : act) x = dgpp::float_to_bf16_bits(uf(rng));
    uint16_t* da; DGPP_CUDA_OK(cudaMalloc(&da, act.size() * 2)); DGPP_CUDA_OK(cudaMemcpy(da, act.data(), act.size() * 2, cudaMemcpyHostToDevice));
    float* o1; float* o2; uint16_t* b1; uint16_t* b2;
    DGPP_CUDA_OK(cudaMalloc(&o1, size_t(32) * n * 4)); DGPP_CUDA_OK(cudaMalloc(&o2, size_t(32) * n * 4));
    DGPP_CUDA_OK(cudaMalloc(&b1, size_t(32) * n * 2)); DGPP_CUDA_OK(cudaMalloc(&b2, size_t(32) * n * 2));
    for (int m : {1, 3, 8, 16, 17, 32}) {
      require(dgpp::mma_gemv_bf12_accepts(pm, m), "bf12 form: accepts the decode width");
      DGPP_CUDA_OK(cudaMemset(o2, 0xFF, size_t(32) * n * 4)); DGPP_CUDA_OK(cudaMemset(b2, 0xFF, size_t(32) * n * 2));
      dgpp::launch_mma_gemv_bf16_f32(da, k, dw, o1, m, n, k, n, nullptr);
      require(dgpp::launch_mma_gemv_bf12_f32(da, k, pm, o2, m, n, k, n, nullptr), "bf12 form: the f32 launch applies");
      dgpp::launch_mma_gemv_bf16_bf16(da, k, dw, b1, m, n, k, n, nullptr);
      require(dgpp::launch_mma_gemv_bf12_bf16(da, k, pm, b2, m, n, k, n, nullptr), "bf12 form: the bf16 launch applies");
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const auto f1 = fetch_f(o1, size_t(m) * n), f2 = fetch_f(o2, size_t(m) * n);
      require(std::memcmp(f1.data(), f2.data(), f1.size() * 4) == 0,
              "bf12 form: f32 output differs from the bf16 form at m=" + std::to_string(m) + " n=" + std::to_string(n) + " k=" + std::to_string(k));
      const auto g1 = fetch_b(b1, size_t(m) * n), g2 = fetch_b(b2, size_t(m) * n);
      require(std::memcmp(g1.data(), g2.data(), g1.size() * 2) == 0,
              "bf12 form: bf16 output differs from the bf16 form at m=" + std::to_string(m) + " n=" + std::to_string(n));
    }
    require(!dgpp::mma_gemv_bf12_accepts(pm, 33), "bf12 form: declines the widths above the decode forms");
    require(!dgpp::launch_mma_gemv_bf12_f32(da, k, pm, o2, 33, n, k, n, nullptr), "bf12 form: 33 rows not launched");
    cudaFree(dw); cudaFree(dp); cudaFree(dr); cudaFree(de); cudaFree(draw); cudaFree(da);
    cudaFree(o1); cudaFree(o2); cudaFree(b1); cudaFree(b2);
  }
  {
    // A companion with a tail (k = 2560: two super-blocks and 512 columns) is declined.
    dgpp::Bf12Matrix pm; pm.n = 64; pm.k = 2560;
    pm.packed = reinterpret_cast<const uint8_t*>(1); pm.rows = reinterpret_cast<const uint32_t*>(1);
    pm.esc = reinterpret_cast<const uint32_t*>(1); pm.raw = reinterpret_cast<const uint16_t*>(1);
    require(!dgpp::mma_gemv_bf12_accepts(pm, 4), "bf12 form: declines a tail");
  }
}

DGPP_TEST(mma_gemv_fp8_pair_is_bitwise_the_two_launches) {
  // The two-segment decode form against two single launches, bitwise, on
  // the DeepSeek-V4.1 projection pairs (wq_a with wkv: 1280 + 512 rows of
  // k = 5120; two wo_a groups: 1024 + 1024 of k = 4096, distinct
  // activations) at 1, 4, 8 and 16 rows, the 32 x 32 scale grid.
  const size_t ws_bytes = 64u << 20;
  void* ws = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&ws, ws_bytes));
  struct Pair { int n0, n1, k; bool same_act; };
  for (const Pair pr : {Pair{1280, 512, 5120, true}, Pair{1024, 1024, 4096, false}}) {
    for (const int m : {1, 4, 8, 16}) {
      const int n = pr.n0 + pr.n1, k = pr.k;
      std::vector<uint8_t> hw(size_t(n) * k);
      for (size_t i = 0; i < hw.size(); ++i) hw[i] = static_cast<uint8_t>(0x30 + ((i * 2654435761u) >> 13) % 0x20);
      const int sr = (n + 31) / 32, sc = (k + 31) / 32;
      std::vector<float> hs(size_t(sr) * sc);
      for (size_t i = 0; i < hs.size(); ++i) hs[i] = 0.5f + float((i * 40503u) & 0xFF) / 512.f;
      std::vector<uint16_t> ha(size_t(2 * m) * k);
      for (size_t i = 0; i < ha.size(); ++i) ha[i] = 0x3C00 | static_cast<uint16_t>((i * 2246822519u) & 0x3FF);
      uint8_t* dw = nullptr; float* ds = nullptr; uint16_t* da = nullptr; uint16_t *o_single = nullptr, *o_pair = nullptr;
      DGPP_CUDA_OK(cudaMalloc(&dw, hw.size())); DGPP_CUDA_OK(cudaMalloc(&ds, hs.size() * 4)); DGPP_CUDA_OK(cudaMalloc(&da, ha.size() * 2));
      DGPP_CUDA_OK(cudaMalloc(&o_single, size_t(m) * n * 2)); DGPP_CUDA_OK(cudaMalloc(&o_pair, size_t(m) * n * 2));
      DGPP_CUDA_OK(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice));
      DGPP_CUDA_OK(cudaMemcpy(ds, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice));
      DGPP_CUDA_OK(cudaMemcpy(da, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice));
      const uint16_t* a0 = da; const uint16_t* a1 = pr.same_act ? da : da + size_t(m) * k;
      const uint8_t* w1 = dw + size_t(pr.n0) * k; const float* s1 = ds + (size_t(pr.n0) / 32) * sc;
      // the two single launches (the split form: the workspace passed), outputs side by side at stride n
      dgpp::launch_mma_gemv_fp8_bf16(a0, k, dw, ds, o_single, m, pr.n0, k, n, 5, 5, nullptr, ws, ws_bytes);
      dgpp::launch_mma_gemv_fp8_bf16(a1, k, w1, s1, o_single + pr.n0, m, pr.n1, k, n, 5, 5, nullptr, ws, ws_bytes);
      require(dgpp::mma_gemv_pair_ok(dw, w1, a0, a1, k, m, pr.n0, pr.n1, k, ws, ws_bytes), "the pair form holds for the shape");
      dgpp::launch_mma_gemv_fp8_bf16_pair(a0, dw, ds, o_pair, pr.n0, n, a1, w1, s1, o_pair + pr.n0, pr.n1, n, k, m, k, 5, 5,
                                          nullptr, ws, ws_bytes);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::vector<uint16_t> hs1(size_t(m) * n), hp(size_t(m) * n);
      DGPP_CUDA_OK(cudaMemcpy(hs1.data(), o_single, hs1.size() * 2, cudaMemcpyDeviceToHost));
      DGPP_CUDA_OK(cudaMemcpy(hp.data(), o_pair, hp.size() * 2, cudaMemcpyDeviceToHost));
      require(hs1 == hp, "the pair launch is bitwise the two launches at m=" + std::to_string(m) + " n0=" + std::to_string(pr.n0));
      cudaFree(dw); cudaFree(ds); cudaFree(da); cudaFree(o_single); cudaFree(o_pair);
    }
  }
  cudaFree(ws);
}

int main() { return dgpp::test::run_all(); }

// The e8m0 byte scale grid (the QAD export's MXFP8 dense stack, rs = 0,
// cs = 5): the streaming MMA and the dequant bridge read 2^(b - 127) from the
// byte exactly as the F32 grid holds it, so both forms are bitwise the F32
// grid's at one, three, seventeen and forty rows (every tile form).
DGPP_TEST(mma_gemv_fp8_e8m0_grid_is_bitwise_the_f32_grid) {
  const int n = 640, k = 1024;
  const int sb = k / 32;
  std::vector<uint8_t> w8(static_cast<size_t>(n) * k), sb8(static_cast<size_t>(n) * sb);
  std::vector<float> sf(sb8.size());
  uint32_t x = 0x9E3779B9u;
  auto rnd = [&] { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; };
  for (auto& b : w8) b = static_cast<uint8_t>(rnd() & 0x7F);  // non-NaN e4m3
  for (size_t i = 0; i < sb8.size(); ++i) {
    sb8[i] = static_cast<uint8_t>(118 + rnd() % 14);  // 2^-9 .. 2^4
    sf[i] = std::ldexp(1.0f, static_cast<int>(sb8[i]) - 127);
  }
  const int mmax = 40;
  std::vector<uint16_t> act(static_cast<size_t>(mmax) * k);
  for (auto& a : act) a = dgpp::float_to_bf16_bits(0.5f * (static_cast<float>(rnd() % 2001) / 1000.0f - 1.0f));
  uint8_t* dw = nullptr; uint8_t* dsb = nullptr; float* dsf = nullptr; uint16_t* dact = nullptr;
  uint16_t* out_f = nullptr; uint16_t* out_b = nullptr; uint16_t* deq_f = nullptr; uint16_t* deq_b = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&dw, w8.size())); DGPP_CUDA_OK(cudaMalloc(&dsb, sb8.size()));
  DGPP_CUDA_OK(cudaMalloc(&dsf, sf.size() * 4)); DGPP_CUDA_OK(cudaMalloc(&dact, act.size() * 2));
  DGPP_CUDA_OK(cudaMalloc(&out_f, static_cast<size_t>(mmax) * n * 2)); DGPP_CUDA_OK(cudaMalloc(&out_b, static_cast<size_t>(mmax) * n * 2));
  DGPP_CUDA_OK(cudaMalloc(&deq_f, w8.size() * 2)); DGPP_CUDA_OK(cudaMalloc(&deq_b, w8.size() * 2));
  DGPP_CUDA_OK(cudaMemcpy(dw, w8.data(), w8.size(), cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(dsb, sb8.data(), sb8.size(), cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(dsf, sf.data(), sf.size() * 4, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(dact, act.data(), act.size() * 2, cudaMemcpyHostToDevice));
  for (const int m : {1, 3, 17, 40}) {
    dgpp::launch_mma_gemv_fp8_bf16(dact, k, dw, dsf, out_f, m, n, k, 0, 0, 5, nullptr);
    dgpp::launch_mma_gemv_fp8e8m0_bf16(dact, k, dw, dsb, out_b, m, n, k, 0, 0, 5, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint16_t> a(static_cast<size_t>(m) * n), b(a.size());
    DGPP_CUDA_OK(cudaMemcpy(a.data(), out_f, a.size() * 2, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(b.data(), out_b, b.size() * 2, cudaMemcpyDeviceToHost));
    if (std::memcmp(a.data(), b.data(), a.size() * 2) != 0)
      throw std::runtime_error("e8m0 grid differs from the f32 grid at m = " + std::to_string(m));
  }
  dgpp::launch_fp8_dequant_blocks(dw, dsf, deq_f, n, k, nullptr, 0, 5);
  dgpp::launch_fp8_dequant_blocks_e8m0(dw, dsb, deq_b, n, k, nullptr, 0, 5);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> a(w8.size()), b(w8.size());
  DGPP_CUDA_OK(cudaMemcpy(a.data(), deq_f, a.size() * 2, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(b.data(), deq_b, b.size() * 2, cudaMemcpyDeviceToHost));
  if (std::memcmp(a.data(), b.data(), a.size() * 2) != 0) throw std::runtime_error("e8m0 dequant differs from the f32 dequant");
  // The values are the exact bf16(e4m3 x 2^k).
  for (size_t i = 0; i < 4096; ++i) {
    const float want = dgpp::fp8_e4m3_bits_to_float(w8[i]) * sf[(i / k) * sb + (i % k) / 32];
    if (b[i] != dgpp::float_to_bf16_bits(want)) throw std::runtime_error("e8m0 dequant value differs from bf16(e4m3 x 2^k)");
  }
  cudaFree(dw); cudaFree(dsb); cudaFree(dsf); cudaFree(dact); cudaFree(out_f); cudaFree(out_b); cudaFree(deq_f); cudaFree(deq_b);
  std::printf("[ OK ] the e8m0 byte grid is bitwise the f32 grid (mma_gemv at 1/3/17/40 rows, the dequant bridge)\n");
}
