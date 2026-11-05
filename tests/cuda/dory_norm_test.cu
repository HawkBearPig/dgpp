// The Dory nGPT normalizations (kernels/dory_norm.hpp) against the host
// reference transcribed from the vLLM port (models/dory/reference.cpp), over
// random rows of every width the model uses (256 per-head, 4096 the
// attention output, 7168 the MLP intermediate) and a few magnitude profiles
// (uniform, 1e3 outliers, 1e-3 floors, near-zero rows, a wide exponent
// range), with the learned-alpha residual at several scales.
//
// TWO COMPARISONS, because the two oracles answer different questions.
//
// 1. Against the fp64-exact oracle below — BITWISE, every value. The kernel
//    reduces in fp64 because every bf16 square is exact there and the partial
//    sums of a model-sized row stay exact too, so the sum is the exact one and
//    `float(sum) + eps` rounds once. An oracle that sums the same terms in
//    fp64 therefore has the same rstd bit for bit, whatever order either
//    uses, and pins the kernel's arithmetic — the fp32 slerp
//    combine and final bf16 conversion included — exactly.
// 2. Against the transcription — the reference's own sequential fp32 fma
//    chain, whose sum differs from the exact one by ~sqrt(dim)·2^-24 relative
//    (2048 additions of rounding noise), so the kernel's outputs drift from
//    it by that much. The law is absolute and scaled to the case: the largest
//    difference must be under 2% of the case's largest magnitude, and at most
//    1% of the values may differ at all. The observed drift is ~0.03% of the
//    values (0.016% at dim 256, 0.035% at 2048/4096, none at dim 8). A ulp
//    distance is deliberately NOT the metric here: the residual's
//    `na + lr·(nb − na)` crosses zero, and near zero a bf16 ulp is vanishingly
//    small in absolute terms (a 38% relative difference at a value of ~1e-5 is
//    a 1e-8 absolute one). This still fails on any real semantic error, which
//    would move most of the values, not a thousandth of them.
//
#include <algorithm>
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
#include "kernels/dory_norm.hpp"
#include "models/dory/reference.hpp"
#include "../dory_slerp_golden.hpp"

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
        case 1: if (u01(rng) < 0.002f) x *= 1000.f; break;      // outliers
        case 2: x *= (u01(rng) < 0.5f ? 1e-3f : 1.f); break;    // a floor of tiny values
        case 3: x *= 1e-4f; break;                              // near-zero rows
        case 4: x = (u01(rng) < 0.001f) ? x * 3e4f : x * 1e-5f; break;  // wide exponent range
        default: break;
      }
      v[static_cast<size_t>(r) * dim + d] = dgpp::float_to_bf16_bits(x);
    }
  return v;
}

// Monotone key of a bf16 bit pattern, so "one ulp apart" is |key difference|
// <= 1 across the sign boundary too.
int32_t key(uint16_t bits) {
  return (bits & 0x8000u) ? -static_cast<int32_t>(bits & 0x7fffu) : static_cast<int32_t>(bits);
}

struct Diff {
  size_t exact_mismatch = 0;  // any difference
  size_t beyond_1ulp = 0;     // informational: a ulp distance is meaningless near zero
  double max_abs = 0.0;       // largest absolute difference
  double ref_scale = 0.0;     // largest |reference| in the case
};

Diff compare(const std::vector<uint16_t>& ref, const std::vector<uint16_t>& got) {
  Diff d;
  require(ref.size() == got.size(), "size");
  for (size_t i = 0; i < ref.size(); ++i) {
    const float x = dgpp::bf16_bits_to_float(ref[i]), y = dgpp::bf16_bits_to_float(got[i]);
    d.ref_scale = std::max(d.ref_scale, static_cast<double>(std::fabs(x)));
    if (ref[i] == got[i]) continue;
    ++d.exact_mismatch;
    if (std::abs(key(ref[i]) - key(got[i])) > 1) ++d.beyond_1ulp;
    d.max_abs = std::max(d.max_abs, static_cast<double>(std::fabs(x - y)));
  }
  return d;
}

// The transcription's law (file comment): the kernel must track the reference
// to well inside the case's own magnitude, with almost every value identical.
void check_transcription(const char* what, const Diff& d, size_t n) {
  require(d.max_abs <= 2e-2 * d.ref_scale + 1e-30,
          std::string(what) + ": difference beyond the reference's own reduction error");
  require(static_cast<double>(d.exact_mismatch) <= 1e-2 * static_cast<double>(n) + 4,
          std::string(what) + ": too many values differ for a reduction-order difference");
}

// The kernel's contract with the reference's interior difference removed: the
// sums run in fp64, so no order matters and a bitwise match is the right
// criterion. Same expressions, same roundings.
namespace exact_oracle {

float rstd(const uint16_t* x, int dim, float eps) {
  double ss = 0.0;
  for (int i = 0; i < dim; ++i) {
    const double v = dgpp::bf16_bits_to_float(x[i]);
    ss = std::fma(v, v, ss);
  }
  return 1.0f / std::sqrt(static_cast<float>(ss) + eps);
}

void justnorm(const uint16_t* x, uint16_t* y, int rows, int dim, float eps) {
  for (int r = 0; r < rows; ++r) {
    const uint16_t* xr = x + static_cast<size_t>(r) * dim;
    uint16_t* yr = y + static_cast<size_t>(r) * dim;
    const float rd = rstd(xr, dim, eps);
    for (int i = 0; i < dim; ++i)
      yr[i] = dgpp::float_to_bf16_bits(dgpp::bf16_bits_to_float(xr[i]) * rd);
  }
}

void slerp_alpha(const uint16_t* a, const uint16_t* b, const uint16_t* alpha, float scale, uint16_t* y,
                 int rows, int dim, float eps) {
  std::vector<uint16_t> na(static_cast<size_t>(dim)), nb(static_cast<size_t>(dim));
  std::vector<float> mixed(static_cast<size_t>(dim));
  for (int r = 0; r < rows; ++r) {
    const uint16_t* ar = a + static_cast<size_t>(r) * dim;
    const uint16_t* br = b + static_cast<size_t>(r) * dim;
    uint16_t* yr = y + static_cast<size_t>(r) * dim;
    justnorm(ar, na.data(), 1, dim, eps);
    justnorm(br, nb.data(), 1, dim, eps);
    double ss = 0.0;
    for (int i = 0; i < dim; ++i) {
      const float x = dgpp::bf16_bits_to_float(na[static_cast<size_t>(i)]);
      const float lr = std::fabs(dgpp::bf16_bits_to_float(alpha[i]) * scale);
      const float d = dgpp::bf16_bits_to_float(nb[static_cast<size_t>(i)]) - x;
      mixed[static_cast<size_t>(i)] = x + lr * d;
      const double v = mixed[static_cast<size_t>(i)];
      ss = std::fma(v, v, ss);
    }
    const float rd = 1.0f / std::sqrt(static_cast<float>(ss) + eps);
    for (int i = 0; i < dim; ++i)
      yr[i] = dgpp::float_to_bf16_bits(mixed[static_cast<size_t>(i)] * rd);
  }
}

  // The fp32 products summed in fp64 — the reference's one-rounding contract —
  // with the host's `exp` for the sigmoid/silu (the device's `expf` differs by
  // an ulp, which is what the ulp budget below absorbs).
  std::vector<uint16_t> gate_oracle(const std::vector<uint16_t>& attn,
                                    const std::vector<uint16_t>& gate, const std::vector<uint16_t>& s_gate,
                                    float f, int rows, int dim, float eps) {
    std::vector<uint16_t> y(attn.size());
    std::vector<float> g(static_cast<size_t>(dim));
    for (int r = 0; r < rows; ++r) {
      const size_t row = static_cast<size_t>(r) * dim;
      double ss = 0.0;
      for (int i = 0; i < dim; ++i) {
        const float a = dgpp::bf16_bits_to_float(attn[row + i]);
        const float gv = dgpp::bf16_bits_to_float(gate[row + i]);
        const float m = dgpp::bf16_bits_to_float(s_gate[static_cast<size_t>(i)]) * f;
        const float z = gv * (m * m);
        g[static_cast<size_t>(i)] = a * (1.0f / (1.0f + std::exp(-z)));
        ss = std::fma(static_cast<double>(g[static_cast<size_t>(i)]),
                      static_cast<double>(g[static_cast<size_t>(i)]), ss);
      }
      const float rstd = 1.0f / std::sqrt(static_cast<float>(ss) + eps);
      for (int i = 0; i < dim; ++i)
        y[row + i] = dgpp::float_to_bf16_bits(g[static_cast<size_t>(i)] * rstd);
    }
    return y;
  }

  std::vector<uint16_t> silu_oracle(const std::vector<uint16_t>& g_in, const std::vector<uint16_t>& u_in,
                                    const std::vector<uint16_t>& suv_gate,
                                    const std::vector<uint16_t>& suv_up, float f, int rows, int dim,
                                    float eps) {
    std::vector<uint16_t> y(g_in.size());
    std::vector<float> m(static_cast<size_t>(dim));
    for (int r = 0; r < rows; ++r) {
      const size_t row = static_cast<size_t>(r) * dim;
      double ss = 0.0;
      for (int i = 0; i < dim; ++i) {
        const float gv = dgpp::bf16_bits_to_float(g_in[row + i]) *
                         (dgpp::bf16_bits_to_float(suv_gate[static_cast<size_t>(i)]) * f);
        const float uv = dgpp::bf16_bits_to_float(u_in[row + i]) *
                         (dgpp::bf16_bits_to_float(suv_up[static_cast<size_t>(i)]) * f);
        m[static_cast<size_t>(i)] = (gv / (1.0f + std::exp(-gv))) * uv;
        ss = std::fma(static_cast<double>(m[static_cast<size_t>(i)]),
                      static_cast<double>(m[static_cast<size_t>(i)]), ss);
      }
      const float rstd = 1.0f / std::sqrt(static_cast<float>(ss) + eps);
      for (int i = 0; i < dim; ++i)
        y[row + i] = dgpp::float_to_bf16_bits(m[static_cast<size_t>(i)] * rstd);
    }
    return y;
  }

}  // namespace exact_oracle

void justnorm_case(int rows, int dim, int profile, float eps, std::mt19937& rng) {
  const std::vector<uint16_t> x = random_rows(rng, rows, dim, profile);
  std::vector<uint16_t> expect(x.size()), exact(x.size());
  dgpp::dory_ref::justnorm(x.data(), expect.data(), rows, dim, eps);
  exact_oracle::justnorm(x.data(), exact.data(), rows, dim, eps);
  DevBuf d_x = up(x), d_y(x.size() * 2);
  dgpp::dory_justnorm_bf16(static_cast<const uint16_t*>(d_x.p), static_cast<uint16_t*>(d_y.p), rows, dim,
                           eps, nullptr);
  DGPP_CUDA_OK(cudaStreamSynchronize(nullptr));
  const auto got = down<uint16_t>(d_y, x.size());
  const Diff de = compare(exact, got), dr = compare(expect, got);
  std::printf("  [justnorm] dim %d profile %d: exact %zu, transcription %zu/%zu differ (%.4f%%), max abs %.2e "
              "of %.2e\n",
              dim, profile, de.exact_mismatch, dr.exact_mismatch, static_cast<size_t>(rows) * dim,
              100.0 * static_cast<double>(dr.exact_mismatch) / (static_cast<double>(rows) * dim), dr.max_abs,
              dr.ref_scale);
  require(de.exact_mismatch == 0, "justnorm: differs from the fp64-exact oracle");
  check_transcription("justnorm", dr, x.size());
}

void slerp_case(int rows, int dim, float scale, float alpha_spread, std::mt19937& rng) {
  const std::vector<uint16_t> a = random_rows(rng, rows, dim, 0);
  const std::vector<uint16_t> b = random_rows(rng, rows, dim, 1);
  std::vector<uint16_t> alpha(static_cast<size_t>(dim));
  std::normal_distribution<float> n01(0.f, alpha_spread);
  for (auto& e : alpha) e = dgpp::float_to_bf16_bits(n01(rng));
  std::vector<uint16_t> expect(a.size()), exact(a.size());
  dgpp::dory_ref::slerp_alpha(a.data(), b.data(), alpha.data(), scale, expect.data(), rows, dim,
                              dgpp::kDoryJustnormEps);
  exact_oracle::slerp_alpha(a.data(), b.data(), alpha.data(), scale, exact.data(), rows, dim,
                            dgpp::kDoryJustnormEps);
  DevBuf d_a = up(a), d_b = up(b), d_al = up(alpha), d_y(a.size() * 2);
  dgpp::dory_slerp_alpha_bf16(static_cast<const uint16_t*>(d_a.p), static_cast<const uint16_t*>(d_b.p),
                              static_cast<const uint16_t*>(d_al.p), scale, static_cast<uint16_t*>(d_y.p),
                              rows, dim, dgpp::kDoryJustnormEps, nullptr);
  DGPP_CUDA_OK(cudaStreamSynchronize(nullptr));
  const auto got = down<uint16_t>(d_y, a.size());
  const Diff de = compare(exact, got), dr = compare(expect, got);
  std::printf("  [slerp] dim %d scale %g spread %g: exact %zu, transcription %zu/%zu differ (%.4f%%), max abs "
              "%.2e of %.2e\n",
              dim, scale, alpha_spread, de.exact_mismatch, dr.exact_mismatch,
              static_cast<size_t>(rows) * dim,
              100.0 * static_cast<double>(dr.exact_mismatch) / (static_cast<double>(rows) * dim), dr.max_abs,
              dr.ref_scale);
  require(de.exact_mismatch == 0, "slerp_alpha: differs from the fp64-exact oracle");
  check_transcription("slerp_alpha", dr, a.size());
}

// The two fused sites (plan §1.3/§1.4): the bf16 gated output and the bf16
// silu·u, both justnormed in the same launch. The oracle sums the fp32
// products in fp64 with the host's `exp`, so the only differences left are the
// device `expf`'s ulp and the resulting bf16 boundary flips.
void gate_case(int rows, int dim, float sg_factor, float spread, std::mt19937& rng) {
  const std::vector<uint16_t> attn = random_rows(rng, rows, dim, 0);
  const std::vector<uint16_t> gate_b = random_rows(rng, rows, dim, 1);
  std::vector<uint16_t> s_gate(static_cast<size_t>(dim));
  std::normal_distribution<float> n01(0.f, spread);
  for (auto& e : s_gate) e = dgpp::float_to_bf16_bits(n01(rng));
  const std::vector<uint16_t> expect =
      exact_oracle::gate_oracle(attn, gate_b, s_gate, sg_factor, rows, dim, dgpp::kDoryJustnormEps);
  DevBuf d_a = up(attn), d_g = up(gate_b), d_s = up(s_gate), d_y(attn.size() * 2);
  dgpp::dory_gate_justnorm_bf16(static_cast<const uint16_t*>(d_a.p), static_cast<const uint16_t*>(d_g.p),
                                static_cast<const uint16_t*>(d_s.p), sg_factor,
                                static_cast<uint16_t*>(d_y.p), rows, dim, dgpp::kDoryJustnormEps, nullptr);
  DGPP_CUDA_OK(cudaStreamSynchronize(nullptr));
  const auto got = down<uint16_t>(d_y, attn.size());
  const Diff d = compare(expect, got);
  std::printf("  [gate_justnorm] dim %d factor %g spread %g: %zu/%zu differ (%.4f%%), max abs %.2e of %.2e\n",
              dim, sg_factor, spread, d.exact_mismatch, got.size(),
              100.0 * static_cast<double>(d.exact_mismatch) / static_cast<double>(got.size()), d.max_abs,
              d.ref_scale);
  require(d.exact_mismatch <= static_cast<size_t>(0.005 * static_cast<double>(got.size())) + 4,
          "gate_justnorm: too many values differ from the oracle");
  require(d.max_abs <= 2e-2 * d.ref_scale + 1e-30, "gate_justnorm: difference beyond the case scale");
}

void silu_case(int rows, int dim, float suv_factor, float spread, std::mt19937& rng) {
  const std::vector<uint16_t> g_in = random_rows(rng, rows, dim, 0);
  const std::vector<uint16_t> u_in = random_rows(rng, rows, dim, 1);
  std::vector<uint16_t> suv_gate(static_cast<size_t>(dim)), suv_up(static_cast<size_t>(dim));
  std::normal_distribution<float> n01(0.f, spread);
  for (auto& e : suv_gate) e = dgpp::float_to_bf16_bits(n01(rng));
  for (auto& e : suv_up) e = dgpp::float_to_bf16_bits(n01(rng));
  const std::vector<uint16_t> expect = exact_oracle::silu_oracle(g_in, u_in, suv_gate, suv_up,
                                                                 suv_factor, rows, dim, dgpp::kDoryJustnormEps);
  DevBuf d_g = up(g_in), d_u = up(u_in), d_sg = up(suv_gate), d_su = up(suv_up),
         d_y(g_in.size() * 2);
  dgpp::dory_silu_justnorm_bf16(static_cast<const uint16_t*>(d_g.p), static_cast<const uint16_t*>(d_u.p),
                                static_cast<const uint16_t*>(d_sg.p), static_cast<const uint16_t*>(d_su.p),
                                suv_factor, static_cast<uint16_t*>(d_y.p), rows, dim,
                                dgpp::kDoryJustnormEps, nullptr);
  DGPP_CUDA_OK(cudaStreamSynchronize(nullptr));
  const auto got = down<uint16_t>(d_y, g_in.size());
  const Diff d = compare(expect, got);
  std::printf("  [silu_justnorm] dim %d factor %g spread %g: %zu/%zu differ (%.4f%%), max abs %.2e of %.2e\n",
              dim, suv_factor, spread, d.exact_mismatch, got.size(),
              100.0 * static_cast<double>(d.exact_mismatch) / static_cast<double>(got.size()), d.max_abs,
              d.ref_scale);
  require(d.exact_mismatch <= static_cast<size_t>(0.005 * static_cast<double>(got.size())) + 4,
          "silu_justnorm: too many values differ from the oracle");
  require(d.max_abs <= 2e-2 * d.ref_scale + 1e-30, "silu_justnorm: difference beyond the case scale");
}

}  // namespace

DGPP_TEST(dory_norm_justnorm_matches_the_reference) {
  std::mt19937 rng(20261018);
  for (int dim : {8, 256, 512, 2048, 4096, 7168, 8192})
    for (int profile = 0; profile < 5; ++profile) justnorm_case(dim == 4096 ? 256 : 48, dim, profile, 1e-6f, rng);
  justnorm_case(37, 4096, 0, 1e-5f, rng);
}

DGPP_TEST(dory_norm_slerp_alpha_matches_the_reference) {
  std::mt19937 rng(20261019);
  // lr = |alpha · alpha_scale| with alpha_scale = ngpt_alpha_init /
  // initializer_range (0.05/0.0177 ~= 2.8249), the model's own scale; 0 and
  // 1e-3 are the lr -> 0 and lr -> 0+ limits, 1.0 the identity.
  for (int dim : {256, 4096, 7168})
    for (float scale : {0.0f, 1e-3f, 1.0f, 2.8249f})
      for (float spread : {0.3f, 3.0f}) slerp_case(dim == 4096 ? 256 : 48, dim, scale, spread, rng);
  slerp_case(31, 7168, 2.8249f, 0.3f, rng);
  // lr an order of magnitude past anything the model asks for: the combine's
  // amplification is real (it scales a one-ulp flip in na/nb by lr), and the
  // exact-oracle check still holds bitwise there.
  for (int dim : {256, 4096})
    for (float scale : {10.0f, 100.0f}) slerp_case(48, dim, scale, 3.0f, rng);
}

DGPP_TEST(dory_norm_slerp_matches_upstream_golden) {
  namespace g = dory_slerp_golden;
  std::vector<uint16_t> a(g::a.size()), b(a.size()), alpha(a.size()), expected(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = dgpp::float_to_bf16_bits(g::a[i]);
    b[i] = dgpp::float_to_bf16_bits(g::b[i]);
    alpha[i] = dgpp::float_to_bf16_bits(g::alpha);
    expected[i] = dgpp::float_to_bf16_bits(g::output[i]);
  }
  DevBuf d_a = up(a), d_b = up(b), d_alpha = up(alpha), d_y(a.size() * 2);
  dgpp::dory_slerp_alpha_bf16(static_cast<const uint16_t*>(d_a.p),
                            static_cast<const uint16_t*>(d_b.p),
                            static_cast<const uint16_t*>(d_alpha.p), g::scale,
                            static_cast<uint16_t*>(d_y.p), 1, static_cast<int>(a.size()),
                            dgpp::kDoryJustnormEps, nullptr);
  DGPP_CUDA_OK(cudaStreamSynchronize(nullptr));
  require(down<uint16_t>(d_y, a.size()) == expected,
          "the device residual matches the upstream FP32 interpolation golden");
}

DGPP_TEST(dory_norm_gate_justnorm_matches_the_reference) {
  std::mt19937 rng(20261020);
  // The attention output's site: dim 4096 = all 16 query heads (plan §1.3),
  // with the model's own s_gate factor and a wide-spread control.
  for (float factor : {0.5f, 1.0f, 2.8249f})
    for (float spread : {0.3f, 3.0f}) gate_case(64, 4096, factor, spread, rng);
  gate_case(37, 4096, 1.0f, 0.3f, rng);
}

DGPP_TEST(dory_norm_silu_justnorm_matches_the_reference) {
  std::mt19937 rng(20261021);
  // The MLP's site: dim 7168 = the intermediate (plan §1.4).
  for (float factor : {0.5f, 1.0f, 2.8249f})
    for (float spread : {0.3f, 3.0f}) silu_case(64, 7168, factor, spread, rng);
  silu_case(31, 7168, 1.0f, 0.3f, rng);
}

DGPP_TEST(dory_norm_rejects_bad_shapes) {
  DevBuf a(2 * 8192), b(2 * 8192);
  auto* ap = static_cast<uint16_t*>(a.p);
  auto* bp = static_cast<uint16_t*>(b.p);
  for (int dim : {12, 20, 8200}) {
    bool threw = false;
    try {
      dgpp::dory_justnorm_bf16(ap, bp, 1, dim, 1e-6f, nullptr);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    require(threw, "justnorm accepted dim " + std::to_string(dim));
    threw = false;
    try {
      dgpp::dory_slerp_alpha_bf16(ap, bp, bp, 1.0f, ap, 1, dim, 1e-6f, nullptr);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    require(threw, "slerp_alpha accepted dim " + std::to_string(dim));
  }
  bool threw = false;
  try {
    dgpp::dory_justnorm_bf16(nullptr, bp, 1, 256, 1e-6f, nullptr);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "justnorm accepted a null input");
}

DGPP_TEST(dory_norm_timing) {
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
  const float one_row = time([&] {
    dgpp::dory_justnorm_bf16(static_cast<const uint16_t*>(x.p), static_cast<uint16_t*>(o.p), rows, dim,
                             1e-6f, s);
  });
  const float residual = time([&] {
    dgpp::dory_slerp_alpha_bf16(static_cast<const uint16_t*>(x.p), static_cast<const uint16_t*>(y.p),
                                static_cast<const uint16_t*>(w.p), 2.8249f, static_cast<uint16_t*>(o.p),
                                rows, dim, 1e-6f, s);
  });
  std::printf("  [timing] 1 x %d: justnorm %.1f us, slerp_alpha (fused residual) %.1f us\n", dim, one_row,
              residual);

  // The two fused sites: one launch per site instead of a chain of passes.
  DevBuf attn = up(random_rows(rng, 1, 4096, 0)), gate = up(random_rows(rng, 1, 4096, 0));
  DevBuf sg = up(random_rows(rng, 1, 4096, 0)), go(4096 * 2);
  const float gated = time([&] {
    dgpp::dory_gate_justnorm_bf16(static_cast<const uint16_t*>(attn.p),
                                  static_cast<const uint16_t*>(gate.p),
                                  static_cast<const uint16_t*>(sg.p), 1.0f,
                                  static_cast<uint16_t*>(go.p), 1, 4096, 1e-6f, s);
  });
  DevBuf g_in = up(random_rows(rng, 1, 7168, 0)), u_in = up(random_rows(rng, 1, 7168, 0));
  DevBuf sgg = up(random_rows(rng, 1, 7168, 0)), suu = up(random_rows(rng, 1, 7168, 0)), mo(7168 * 2);
  const float silu = time([&] {
    dgpp::dory_silu_justnorm_bf16(
        static_cast<const uint16_t*>(g_in.p), static_cast<const uint16_t*>(u_in.p),
        static_cast<const uint16_t*>(sgg.p), static_cast<const uint16_t*>(suu.p), 1.0f,
        static_cast<uint16_t*>(mo.p), 1, 7168, 1e-6f, s);
  });
  std::printf("  [timing] fused sites: 1 x 4096 gate+justnorm %.1f us, 1 x 7168 silu+justnorm %.1f us\n",
              gated, silu);
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;
  return dgpp::test::run_all();
}
