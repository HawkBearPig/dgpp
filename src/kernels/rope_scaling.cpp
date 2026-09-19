// The YaRN inverse-frequency table (kernels/rope_scaling.hpp). The body is
// the DeepSeek-V4.1 CSA2 builder, moved here verbatim — kernels/csa2.cu's
// `csa2_rope_inv_freq_host` now forwards to it, so the CSA2 tables stay
// bit-identical (tests/unit/qwen_rope_scaling_test.cpp pins the dsv41
// values) while a non-CUDA test can gate the arithmetic and the Qwen path
// can share it.
#include "kernels/rope_scaling.hpp"

#include <algorithm>
#include <vector>

namespace dgpp {

void yarn_rope_inv_freq_host(int rope_dim, double theta, int64_t correction_max_position,
                             double factor, double beta_fast, double beta_slow, float* out) {
  if (rope_dim <= 0 || rope_dim % 2 != 0)
    throw std::invalid_argument("yarn_rope_inv_freq_host: rope_dim must be positive and even");
  if (!(theta > 0)) throw std::invalid_argument("yarn_rope_inv_freq_host: theta must be positive");
  if (out == nullptr) throw std::invalid_argument("yarn_rope_inv_freq_host: null out");
  const int half = rope_dim / 2;
  std::vector<float> freqs(static_cast<size_t>(half));
  for (int i = 0; i < half; ++i) {
    // torch: base ** (arange(0, dim, 2, float32) / dim) in fp32, then 1 / it.
    const float e = float(2 * i) / float(rope_dim);
    const float pw = static_cast<float>(std::pow(theta, static_cast<double>(e)));
    // A subnormal checkpoint theta (1e-308) underflows pow to 0 and 1/0 poisons
    // the table (the 2026-09-19 review, F1): refuse it here, not at the first
    // non-finite lane.
    if (!(pw > 0.0f) || !std::isfinite(pw))
      throw std::invalid_argument("yarn_rope_inv_freq_host: theta underflows the table (too small)");
    freqs[size_t(i)] = 1.0f / pw;
  }
  if (correction_max_position > 0) {
    const auto corrected_dim = [&](double rotations) {
      return rope_dim * std::log(static_cast<double>(correction_max_position) / (rotations * 2.0 * 3.14159265358979323846)) /
             (2.0 * std::log(theta));
    };
    // The band bounds before the int casts (the 2026-09-19 review, F1): theta
    // == 1.0 makes 2 log theta zero (corrected_dim is ±inf or NaN) and a theta
    // barely above 1 pushes it past the int range — floor/ceil of either is
    // undefined at the static_cast<int> below. Refuse the band, not the cast.
    const double low_d = corrected_dim(beta_fast);
    const double high_d = corrected_dim(beta_slow);
    if (!std::isfinite(low_d) || !std::isfinite(high_d) || low_d < -2147483647.0 || low_d > 2147483647.0 ||
        high_d < -2147483647.0 || high_d > 2147483647.0)
      throw std::invalid_argument(
          "yarn_rope_inv_freq_host: the correction band is not representable (theta must clear 1.0 "
          "with room)");
    const int low = std::max(static_cast<int>(std::floor(low_d)), 0);
    const int high = std::min(static_cast<int>(std::ceil(high_d)), rope_dim - 1);
    const float denom = std::max(float(high - low), 1e-3f);
    const float f = static_cast<float>(factor);
    for (int i = 0; i < half; ++i) {
      float ramp = (float(i) - float(low)) / denom;
      ramp = std::min(std::max(ramp, 0.0f), 1.0f);
      const float smooth = 1.0f - ramp;
      const float fr = freqs[size_t(i)];
      freqs[size_t(i)] = fr / f * (1.0f - smooth) + fr * smooth;
    }
  }
  for (int i = 0; i < half; ++i) out[i] = freqs[size_t(i)];
}

}  // namespace dgpp
