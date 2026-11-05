// The Dory nGPT primitives (docs/dory_plan.md §1.2, §6): justnorm's value on
// a hand-computed row, its row independence, the WIDTH it spans (the
// attention-output normalization covers all query heads, not one head — the
// silent-error risk of plan §6), and the slerp identities that pin the order
// of operations (normalize each side, interpolate, normalize again).
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "models/dory/reference.hpp"
#include "../dory_slerp_golden.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}
void require_near(double got, double want, double tol, const std::string& what) {
  if (!(std::abs(got - want) <= tol))
    throw std::runtime_error(what + ": got " + std::to_string(got) + ", want " + std::to_string(want));
}

std::vector<uint16_t> bf16_row(const std::vector<float>& v) {
  std::vector<uint16_t> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = dgpp::float_to_bf16_bits(v[i]);
  return out;
}
std::vector<float> to_f32(const std::vector<uint16_t>& v) {
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = dgpp::bf16_bits_to_float(v[i]);
  return out;
}
double sum_sq(const std::vector<uint16_t>& v) {
  double s = 0.0;
  for (uint16_t b : v) {
    const double x = dgpp::bf16_bits_to_float(b);
    s += x * x;
  }
  return s;
}

}  // namespace

DGPP_TEST(dory_justnorm_matches_the_hand_computed_row) {
  const auto x = bf16_row({3.0f, 4.0f});
  std::vector<uint16_t> y(2);
  dgpp::dory_ref::justnorm(x.data(), y.data(), 1, 2);
  const auto f = to_f32(y);
  // 3/5 and 4/5 (the eps moves these by ~1e-8).
  require_near(f[0], 0.6, 2e-3, "x[0]/|x|");
  require_near(f[1], 0.8, 2e-3, "x[1]/|x|");
  // bf16 keeps 8 mantissa bits, so the unit-norm check is the rounding's.
  require_near(sum_sq(y), 1.0, 1.5e-2, "unit norm");
}

DGPP_TEST(dory_justnorm_is_row_independent_and_zero_safe) {
  // A zero row: ss = 0, rstd = 1/sqrt(eps) = 1000, so 0 * 1000 == 0.
  const auto x = bf16_row({3.0f, 4.0f, 0.0f, 0.0f});
  std::vector<uint16_t> y(4);
  dgpp::dory_ref::justnorm(x.data(), y.data(), 2, 2);
  const auto f = to_f32(y);
  require_near(f[0], 0.6, 2e-3, "row 0 untouched by row 1");
  require_near(f[1], 0.8, 2e-3, "row 0 untouched by row 1");
  require(f[2] == 0.0f && f[3] == 0.0f, "a zero row stays zero");
}

DGPP_TEST(dory_justnorm_spans_all_query_heads) {
  // Two 256-wide heads: head 0 all ones, head 1 all 1e-3. Head 0's own norm
  // is 16 either way (it stays 1/16), but head 1's is only 0.016: the wide
  // normalization (width 512) scales it by the whole row's 1/16, while a
  // per-head one would leave it at ~1/16 of a much smaller norm. Plan §6
  // records this as the plausible silent error.
  const int q_size = 512, head_dim = 256;
  std::vector<float> raw(static_cast<size_t>(q_size), 1.0f);
  for (int i = head_dim; i < q_size; ++i) raw[static_cast<size_t>(i)] = 1e-3f;
  const auto x = bf16_row(raw);
  std::vector<uint16_t> wide(static_cast<size_t>(q_size)), per_head(static_cast<size_t>(q_size));
  dgpp::dory_ref::justnorm(x.data(), wide.data(), 1, q_size);
  dgpp::dory_ref::justnorm(x.data(), per_head.data(), q_size / head_dim, head_dim);

  const auto w = to_f32(wide);
  const auto h = to_f32(per_head);
  // Head 0 agrees (its own norm dominates the wide one).
  require_near(w[0], 0.0625, 2e-3, "wide head-0 value");
  require_near(h[0], 0.0625, 2e-3, "per-head head-0 value");
  // Head 1 does not: ~6.25e-5 wide against ~0.0624 per-head.
  require(w[static_cast<size_t>(head_dim)] < 1e-4, "wide head-1 value is the row's");
  require(h[static_cast<size_t>(head_dim)] > 5e-2, "per-head head-1 value is the head's");
  require(h[static_cast<size_t>(head_dim)] / w[static_cast<size_t>(head_dim)] > 100.0,
          "the two widths disagree by ~1000x");
}

DGPP_TEST(dory_slerp_norm_identities) {
  const auto a = bf16_row({1.0f, 2.0f, -3.0f, 0.5f});
  const auto b = bf16_row({-2.0f, 1.0f, 0.25f, 4.0f});
  std::vector<uint16_t> ya(4), yb(4), y1(4), y0(4);
  dgpp::dory_ref::justnorm(a.data(), ya.data(), 1, 4);
  dgpp::dory_ref::justnorm(b.data(), yb.data(), 1, 4);
  const float lr0[1] = {0.0f}, lr1[1] = {1.0f};
  dgpp::dory_ref::slerp_norm_rows(a.data(), b.data(), lr0, y0.data(), 1, 4);
  dgpp::dory_ref::slerp_norm_rows(a.data(), b.data(), lr1, y1.data(), 1, 4);
  // lr = 0 is justnorm(a); lr = 1 is justnorm(b).
  for (int i = 0; i < 4; ++i) {
    require_near(to_f32(y0)[static_cast<size_t>(i)], to_f32(ya)[static_cast<size_t>(i)], 1e-6, "lr 0 -> a");
    require_near(to_f32(y1)[static_cast<size_t>(i)], to_f32(yb)[static_cast<size_t>(i)], 1e-6, "lr 1 -> b");
  }
  // Any lr lands on the unit sphere (the outer justnorm).
  const float lr2[1] = {0.37f};
  std::vector<uint16_t> y2(4);
  dgpp::dory_ref::slerp_norm_rows(a.data(), b.data(), lr2, y2.data(), 1, 4);
  require_near(sum_sq(y2), 1.0, 1.5e-2, "slerp output is unit norm");
  // a == b: the interpolation is a no-op and the result is justnorm(a).
  std::vector<uint16_t> y3(4);
  dgpp::dory_ref::slerp_norm_rows(a.data(), a.data(), lr2, y3.data(), 1, 4);
  for (int i = 0; i < 4; ++i)
    require_near(to_f32(y3)[static_cast<size_t>(i)], to_f32(ya)[static_cast<size_t>(i)], 1e-6, "a == b");
}

DGPP_TEST(dory_slerp_alpha_uses_the_absolute_learned_scale) {
  const auto a = bf16_row({1.0f, 0.0f, 0.0f, 0.0f});
  const auto b = bf16_row({0.0f, 1.0f, 0.0f, 0.0f});
  // alpha = 0: the residual is justnorm(a) whatever the sign of the scale.
  const auto alpha_zero = bf16_row({0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<uint16_t> y(4), ya(4);
  dgpp::dory_ref::justnorm(a.data(), ya.data(), 1, 4);
  dgpp::dory_ref::slerp_alpha(a.data(), b.data(), alpha_zero.data(), 2.8248588f, y.data(), 1, 4);
  for (int i = 0; i < 4; ++i)
    require_near(to_f32(y)[static_cast<size_t>(i)], to_f32(ya)[static_cast<size_t>(i)], 1e-6, "alpha 0 -> a");

  // The sign of the learned alpha does not matter (|alpha * s|), and a
  // negative one must give the same bytes as its positive twin.
  const auto pos = bf16_row({1.0f, 1.0f, 1.0f, 1.0f});
  const auto neg = bf16_row({-1.0f, -1.0f, -1.0f, -1.0f});
  std::vector<uint16_t> yp(4), yn(4);
  dgpp::dory_ref::slerp_alpha(a.data(), b.data(), pos.data(), 0.5f, yp.data(), 1, 4);
  dgpp::dory_ref::slerp_alpha(a.data(), b.data(), neg.data(), 0.5f, yn.data(), 1, 4);
  for (int i = 0; i < 4; ++i) require(yp[static_cast<size_t>(i)] == yn[static_cast<size_t>(i)], "|alpha|");
  require_near(sum_sq(yp), 1.0, 1.5e-2, "unit norm");
}

DGPP_TEST(dory_slerp_preserves_fp32_interpolation_until_normalized) {
  namespace g = dory_slerp_golden;
  const auto a = bf16_row({g::a.begin(), g::a.end()});
  const auto b = bf16_row({g::b.begin(), g::b.end()});
  const auto alpha = bf16_row(std::vector<float>(g::a.size(), g::alpha));
  const auto expected = bf16_row({g::output.begin(), g::output.end()});
  std::vector<uint16_t> got(a.size());
  dgpp::dory_ref::slerp_alpha(a.data(), b.data(), alpha.data(), g::scale,
                            got.data(), 1, static_cast<int64_t>(a.size()));
  require(got == expected, "the residual matches the upstream FP32 interpolation golden");
  const float lr[] = {dgpp::bf16_bits_to_float(alpha[0]) * g::scale};
  dgpp::dory_ref::slerp_norm_rows(a.data(), b.data(), lr, got.data(), 1,
                                static_cast<int64_t>(a.size()));
  require(got == expected, "the row form preserves the same rounding boundaries");
}

DGPP_TEST(dory_slerp_norm_cols_matches_the_row_form) {
  // A per-column lr vector used row-wise must equal the column form.
  const int rows = 3, dim = 8;
  std::vector<float> ra, rb;
  for (int i = 0; i < rows * dim; ++i) {
    ra.push_back(static_cast<float>((i * 7) % 11) - 5.0f);
    rb.push_back(static_cast<float>((i * 5) % 13) - 6.0f);
  }
  const auto a = bf16_row(ra), b = bf16_row(rb);
  std::vector<float> lr(static_cast<size_t>(dim));
  for (int i = 0; i < dim; ++i) lr[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i);
  std::vector<uint16_t> yc(static_cast<size_t>(rows * dim)), yr(static_cast<size_t>(rows * dim));
  dgpp::dory_ref::slerp_norm_cols(a.data(), b.data(), lr.data(), yc.data(), rows, dim);
  for (int r = 0; r < rows; ++r)
    dgpp::dory_ref::slerp_norm_cols(a.data() + r * dim, b.data() + r * dim, lr.data(),
                                    yr.data() + r * dim, 1, dim);
  for (int i = 0; i < rows * dim; ++i) require(yc[static_cast<size_t>(i)] == yr[static_cast<size_t>(i)], "cols");
}
