#include "models/dory/reference.hpp"

#include <cmath>
#include <vector>

#include "common/dtypes.hpp"

namespace dgpp::dory_ref {
namespace {

// The row's L2 norm as the reference computes it: an fp32 accumulation of the
// squared values, then rsqrt(sum + eps).
float row_rstd(const uint16_t* x, int64_t dim, float eps) {
  float ss = 0.0f;
  for (int64_t i = 0; i < dim; ++i) {
    const float v = bf16_bits_to_float(x[i]);
    ss = std::fma(v, v, ss);
  }
  return 1.0f / std::sqrt(ss + eps);
}

}  // namespace

void justnorm_row(const uint16_t* x, uint16_t* y, int64_t dim, float eps) {
  const float rstd = row_rstd(x, dim, eps);
  for (int64_t i = 0; i < dim; ++i)
    y[i] = float_to_bf16_bits(bf16_bits_to_float(x[i]) * rstd);
}

void justnorm(const uint16_t* x, uint16_t* y, int64_t rows, int64_t dim, float eps) {
  for (int64_t r = 0; r < rows; ++r)
    justnorm_row(x + r * dim, y + r * dim, dim, eps);
}

void add_justnorm(const uint16_t* a, const uint16_t* b, uint16_t* y, int64_t rows, int64_t dim,
                  float eps) {
  std::vector<float> s(static_cast<size_t>(dim));
  for (int64_t r = 0; r < rows; ++r) {
    const uint16_t* ar = a + r * dim;
    const uint16_t* br = b + r * dim;
    uint16_t* yr = y + r * dim;
    float ss = 0.0f;
    for (int64_t i = 0; i < dim; ++i) {
      s[static_cast<size_t>(i)] = bf16_bits_to_float(ar[i]) + bf16_bits_to_float(br[i]);
      ss = std::fma(s[static_cast<size_t>(i)], s[static_cast<size_t>(i)], ss);
    }
    const float rstd = 1.0f / std::sqrt(ss + eps);
    for (int64_t i = 0; i < dim; ++i) yr[i] = float_to_bf16_bits(s[static_cast<size_t>(i)] * rstd);
  }
}

void slerp_norm_rows(const uint16_t* a, const uint16_t* b, const float* lr, uint16_t* y,
                     int64_t rows, int64_t dim, float eps) {
  std::vector<uint16_t> na(static_cast<size_t>(dim)), nb(static_cast<size_t>(dim));
  std::vector<float> mixed(static_cast<size_t>(dim));
  for (int64_t r = 0; r < rows; ++r) {
    const uint16_t* ar = a + r * dim;
    const uint16_t* br = b + r * dim;
    uint16_t* yr = y + r * dim;
    const float s = lr[r];
    justnorm_row(ar, na.data(), dim, eps);
    justnorm_row(br, nb.data(), dim, eps);
    float ss = 0.0f;
    for (int64_t i = 0; i < dim; ++i) {
      const float x = bf16_bits_to_float(na[static_cast<size_t>(i)]);
      const float d = bf16_bits_to_float(nb[static_cast<size_t>(i)]) - x;
      mixed[static_cast<size_t>(i)] = x + s * d;
      ss = std::fma(mixed[static_cast<size_t>(i)], mixed[static_cast<size_t>(i)], ss);
    }
    const float rstd = 1.0f / std::sqrt(ss + eps);
    for (int64_t i = 0; i < dim; ++i)
      yr[i] = float_to_bf16_bits(mixed[static_cast<size_t>(i)] * rstd);
  }
}

void slerp_norm_cols(const uint16_t* a, const uint16_t* b, const float* lr_col, uint16_t* y,
                     int64_t rows, int64_t dim, float eps) {
  std::vector<uint16_t> na(static_cast<size_t>(dim)), nb(static_cast<size_t>(dim));
  std::vector<float> mixed(static_cast<size_t>(dim));
  for (int64_t r = 0; r < rows; ++r) {
    const uint16_t* ar = a + r * dim;
    const uint16_t* br = b + r * dim;
    uint16_t* yr = y + r * dim;
    justnorm_row(ar, na.data(), dim, eps);
    justnorm_row(br, nb.data(), dim, eps);
    float ss = 0.0f;
    for (int64_t i = 0; i < dim; ++i) {
      const float x = bf16_bits_to_float(na[static_cast<size_t>(i)]);
      const float d = bf16_bits_to_float(nb[static_cast<size_t>(i)]) - x;
      mixed[static_cast<size_t>(i)] = x + lr_col[i] * d;
      ss = std::fma(mixed[static_cast<size_t>(i)], mixed[static_cast<size_t>(i)], ss);
    }
    const float rstd = 1.0f / std::sqrt(ss + eps);
    for (int64_t i = 0; i < dim; ++i)
      yr[i] = float_to_bf16_bits(mixed[static_cast<size_t>(i)] * rstd);
  }
}

void slerp_alpha(const uint16_t* a, const uint16_t* b, const uint16_t* alpha, float scale,
                 uint16_t* y, int64_t rows, int64_t dim, float eps) {
  std::vector<float> lr(static_cast<size_t>(dim));
  for (int64_t i = 0; i < dim; ++i)
    lr[static_cast<size_t>(i)] = std::abs(bf16_bits_to_float(alpha[i]) * scale);
  slerp_norm_cols(a, b, lr.data(), y, rows, dim, eps);
}

}  // namespace dgpp::dory_ref
