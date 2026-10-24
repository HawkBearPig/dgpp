// The fused residual add + two-rounding RMSNorm (kernels/add_rmsnorm.hpp)
// against the two-kernel chain it replaces (glm_residual_add_bf16 then
// glm_rmsnorm_bf16): the updated residual and the normed output bitwise,
// over random rows of several dims and magnitude profiles (uniform rows,
// rows with 1e3 outliers, rows with 1e-3 floors, near-zero rows), with and
// without the add, and a timing line at the MiMo shape.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/add_rmsnorm.hpp"
#include "kernels/glm_norm.hpp"

using namespace dgpp::kda_test;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

template <class T>
DevBuf up(const std::vector<T>& v) {
  DevBuf b(v.size() * sizeof(T));
  b.upload(v.data(), v.size() * sizeof(T));
  return b;
}
template <class T>
std::vector<T> down(const DevBuf& b, size_t n) {
  std::vector<T> v(n);
  b.download(v.data(), n * sizeof(T));
  return v;
}

std::vector<uint16_t> random_rows(std::mt19937& rng, int rows, int dim, int profile) {
  std::vector<uint16_t> v(static_cast<size_t>(rows) * dim);
  std::normal_distribution<float> n01(0.f, 1.f);
  std::uniform_real_distribution<float> u01(0.f, 1.f);
  for (int r = 0; r < rows; ++r)
    for (int d = 0; d < dim; ++d) {
      float x = n01(rng);
      switch (profile) {
        case 1: if (u01(rng) < 0.002f) x *= 1000.f; break;   // outliers
        case 2: x *= (u01(rng) < 0.5f ? 1e-3f : 1.f); break;  // a floor of tiny values
        case 3: x *= 1e-4f; break;                            // near-zero rows
        case 4: x = (u01(rng) < 0.001f) ? x * 3e4f : x * 1e-5f; break;  // wide exponent range
        default: break;
      }
      v[static_cast<size_t>(r) * dim + d] = dgpp::float_to_bf16_bits(x);
    }
  return v;
}

void compare_case(int rows, int dim, int profile, bool with_add, float eps, std::mt19937& rng) {
  const std::vector<uint16_t> x0 = random_rows(rng, rows, dim, profile);
  const std::vector<uint16_t> y = random_rows(rng, rows, dim, profile);
  std::vector<uint16_t> w(static_cast<size_t>(dim));
  std::normal_distribution<float> n01(1.f, 0.3f);
  for (auto& e : w) e = dgpp::float_to_bf16_bits(n01(rng));
  DevBuf d_w = up(w), d_y = up(y);
  DevBuf ref_x = up(x0), fused_x = up(x0);
  DevBuf ref_out(x0.size() * 2), fused_out(x0.size() * 2);
  cudaStream_t s = nullptr;
  // The chain.
  if (with_add) dgpp::glm_residual_add_bf16(ref_x.p, d_y.p, static_cast<int64_t>(rows) * dim, s);
  dgpp::glm_rmsnorm_bf16(ref_x.p, d_w.p, ref_out.p, rows, dim, eps, s);
  // The fused form.
  dgpp::add_rmsnorm_bf16(static_cast<uint16_t*>(fused_x.p),
                         with_add ? static_cast<const uint16_t*>(d_y.p) : nullptr,
                         static_cast<const uint16_t*>(d_w.p), static_cast<uint16_t*>(fused_out.p), rows, dim, eps, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  const auto rx = down<uint16_t>(ref_x, x0.size()), fx = down<uint16_t>(fused_x, x0.size());
  const auto ro = down<uint16_t>(ref_out, x0.size()), fo = down<uint16_t>(fused_out, x0.size());
  size_t bad_x = 0, bad_o = 0;
  for (size_t i = 0; i < x0.size(); ++i) {
    bad_x += rx[i] != fx[i];
    bad_o += ro[i] != fo[i];
  }
  char msg[160];
  std::snprintf(msg, sizeof msg, "dim %d profile %d add %d: residual mismatches %zu, output mismatches %zu", dim,
                profile, with_add ? 1 : 0, bad_x, bad_o);
  require(bad_x == 0 && bad_o == 0, msg);
}

// The one-rounding form against a host oracle: the bf16 add, the fp64 sum
// of squares, rstd = rsqrt(mean + eps) in fp32, y = bf16((x * rstd) * w).
// The device's rsqrtf may sit an ulp from the host's 1/sqrt, which flips a
// bf16 rounding on about one element in 65k: every element within one
// bf16 ulp, at least 99.9 % bitwise; the residual bitwise.
void compare_one_rounding(int rows, int dim, int profile, bool with_add, float eps, std::mt19937& rng) {
  const std::vector<uint16_t> x0 = random_rows(rng, rows, dim, profile);
  const std::vector<uint16_t> y = random_rows(rng, rows, dim, profile);
  std::vector<uint16_t> w(static_cast<size_t>(dim));
  std::normal_distribution<float> n01(1.f, 0.3f);
  for (auto& e : w) e = dgpp::float_to_bf16_bits(n01(rng));
  std::vector<uint16_t> want_x(x0.size()), want_o(x0.size());
  for (int r = 0; r < rows; ++r) {
    const size_t off = static_cast<size_t>(r) * dim;
    double ss = 0.0;
    for (int d = 0; d < dim; ++d) {
      float v = dgpp::bf16_bits_to_float(x0[off + d]);
      if (with_add) v = dgpp::bf16_bits_to_float(dgpp::float_to_bf16_bits(v + dgpp::bf16_bits_to_float(y[off + d])));
      want_x[off + d] = dgpp::float_to_bf16_bits(v);
      ss += static_cast<double>(v) * static_cast<double>(v);
    }
    const float rstd = 1.0f / std::sqrt(static_cast<float>(ss / dim) + eps);
    for (int d = 0; d < dim; ++d) {
      const float v = dgpp::bf16_bits_to_float(want_x[off + d]);
      const float t = v * rstd;
      want_o[off + d] = dgpp::float_to_bf16_bits(t * dgpp::bf16_bits_to_float(w[d]));
    }
  }
  DevBuf d_w = up(w), d_y = up(y), d_x = up(x0), d_out(x0.size() * 2);
  cudaStream_t s = nullptr;
  dgpp::add_rmsnorm_bf16_one(static_cast<uint16_t*>(d_x.p), with_add ? static_cast<const uint16_t*>(d_y.p) : nullptr,
                             static_cast<const uint16_t*>(d_w.p), static_cast<uint16_t*>(d_out.p), rows, dim, eps, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  const auto gx = down<uint16_t>(d_x, x0.size()), go = down<uint16_t>(d_out, x0.size());
  size_t bad_x = 0, off_by_one = 0, far = 0;
  for (size_t i = 0; i < x0.size(); ++i) {
    bad_x += gx[i] != want_x[i];
    if (go[i] == want_o[i]) continue;
    const float a = dgpp::bf16_bits_to_float(go[i]), b = dgpp::bf16_bits_to_float(want_o[i]);
    const float ulp = std::ldexp(1.0f, std::ilogb(std::fabs(b) > 0 ? b : 1.f) - 7);
    if (std::fabs(a - b) <= ulp * 1.0001f) ++off_by_one;
    else ++far;
  }
  char msg[200];
  std::snprintf(msg, sizeof msg, "one-rounding dim %d profile %d add %d: residual mismatches %zu, off-by-one %zu, far %zu",
                dim, profile, with_add ? 1 : 0, bad_x, off_by_one, far);
  require(bad_x == 0 && far == 0 && off_by_one * 1000 <= x0.size(), msg);
}

}  // namespace

DGPP_TEST(add_rmsnorm_one_rounding_matches_the_oracle) {
  std::mt19937 rng(20261022);
  for (int dim : {256, 1024, 2048, 4096})
    for (int profile = 0; profile < 4; ++profile)
      for (bool add : {false, true}) compare_one_rounding(dim == 4096 ? 300 : 64, dim, profile, add, 1e-6f, rng);
  compare_one_rounding(37, 2048, 0, true, 1e-5f, rng);
}

DGPP_TEST(add_rmsnorm_matches_the_chain) {
  std::mt19937 rng(20260922);
  for (int dim : {256, 1024, 2048, 3072, 4096, 8192})
    for (int profile = 0; profile < 5; ++profile)
      for (bool add : {false, true}) compare_case(dim == 4096 ? 300 : 64, dim, profile, add, 1e-6f, rng);
  compare_case(37, 4096, 0, true, 1e-5f, rng);
}

DGPP_TEST(add_rmsnorm_rejects_bad_dims) {
  DevBuf a(64 * 2), b(64 * 2);
  bool threw = false;
  try {
    dgpp::add_rmsnorm_bf16(static_cast<uint16_t*>(a.p), nullptr, static_cast<const uint16_t*>(b.p),
                           static_cast<uint16_t*>(a.p), 1, 12, 1e-6f, nullptr);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "dim 12 accepted");
}

DGPP_TEST(add_rmsnorm_timing) {
  const int rows = 1, dim = 4096;
  std::mt19937 rng(7);
  DevBuf x = up(random_rows(rng, rows, dim, 0)), y = up(random_rows(rng, rows, dim, 0));
  DevBuf w = up(random_rows(rng, 1, dim, 0)), o(static_cast<size_t>(rows) * dim * 2);
  cudaStream_t s = nullptr;
  cudaEvent_t e0, e1;
  DGPP_CUDA_OK(cudaEventCreate(&e0));
  DGPP_CUDA_OK(cudaEventCreate(&e1));
  auto time = [&](auto fn) {
    for (int i = 0; i < 20; ++i) fn();
    float best = 1e30f;
    for (int i = 0; i < 200; ++i) {
      DGPP_CUDA_OK(cudaEventRecord(e0, s));
      fn();
      DGPP_CUDA_OK(cudaEventRecord(e1, s));
      DGPP_CUDA_OK(cudaEventSynchronize(e1));
      float ms = 0.f;
      DGPP_CUDA_OK(cudaEventElapsedTime(&ms, e0, e1));
      best = std::min(best, ms);
    }
    return best * 1e3f;
  };
  const float chain = time([&] {
    dgpp::glm_residual_add_bf16(x.p, y.p, static_cast<int64_t>(rows) * dim, s);
    dgpp::glm_rmsnorm_bf16(x.p, w.p, o.p, rows, dim, 1e-6f, s);
  });
  const float fused = time([&] {
    dgpp::add_rmsnorm_bf16(static_cast<uint16_t*>(x.p), static_cast<const uint16_t*>(y.p),
                           static_cast<const uint16_t*>(w.p), static_cast<uint16_t*>(o.p), rows, dim, 1e-6f, s);
  });
  const float norm_only = time([&] { dgpp::glm_rmsnorm_bf16(x.p, w.p, o.p, rows, dim, 1e-6f, s); });
  const float fused_norm_only = time([&] {
    dgpp::add_rmsnorm_bf16(static_cast<uint16_t*>(x.p), nullptr, static_cast<const uint16_t*>(w.p),
                           static_cast<uint16_t*>(o.p), rows, dim, 1e-6f, s);
  });
  std::printf("  [timing] rows %d dim %d: add+rmsnorm chain %.1f us, fused %.1f us; rmsnorm alone %.1f us, "
              "fused (no add) %.1f us\n",
              rows, dim, chain, fused, norm_only, fused_norm_only);
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;
  return dgpp::test::run_all();
}
