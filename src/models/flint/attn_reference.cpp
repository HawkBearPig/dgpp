#include "models/flint/attn_reference.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "common/dtypes.hpp"

namespace dgpp::flint_ref {
namespace {
constexpr int D = kDim;
float rb(float v) { return bf16_bits_to_float(float_to_bf16_bits(v)); }
}  // namespace

void qk_finish(const float* dot, const uint16_t* norm_w, float eps, const float* inv_freq, int64_t pos,
               uint16_t* out) {
  std::vector<float> x(static_cast<size_t>(D));
  for (int d = 0; d < D; ++d) x[static_cast<size_t>(d)] = rb(dot[d]);
  if (norm_w) {
    // The one-rounding RMSNorm: fp32 x * rstd * w, one bf16 rounding. The
    // sum of squares in fp64 (every bf16 square is exact in fp64), the
    // mean narrowed to fp32 as the kernel narrows it.
    double ss = 0.0;
    for (int d = 0; d < D; ++d) ss += static_cast<double>(x[static_cast<size_t>(d)]) * x[static_cast<size_t>(d)];
    const float rstd = 1.0f / std::sqrt(static_cast<float>(ss / D) + eps);
    for (int d = 0; d < D; ++d)
      x[static_cast<size_t>(d)] = rb(x[static_cast<size_t>(d)] * rstd * bf16_bits_to_float(norm_w[d]));
  }
  if (inv_freq) {
    // rotate_half over the whole head: pairs (i, i + 64), fp32 cos / sin of
    // fp32 pos x inv_freq, one rounding of each rotated element.
    const int half = D / 2;
    for (int i = 0; i < half; ++i) {
      const float ang = static_cast<float>(pos) * inv_freq[i];
      const float c = std::cos(ang), s = std::sin(ang);
      const float x1 = x[static_cast<size_t>(i)], x2 = x[static_cast<size_t>(i + half)];
      x[static_cast<size_t>(i)] = rb(x1 * c - x2 * s);
      x[static_cast<size_t>(i + half)] = rb(x2 * c + x1 * s);
    }
  }
  for (int d = 0; d < D; ++d) out[d] = float_to_bf16_bits(x[static_cast<size_t>(d)]);
}

void v_finish(const float* dot, uint16_t* out) {
  for (int d = 0; d < D; ++d) out[d] = float_to_bf16_bits(rb(dot[d]));
}

void attention(const uint16_t* q, const uint16_t* k_rows, const uint16_t* v_rows, int n, int local_heads,
               int kv_heads, float scale, const float* sink, std::vector<float>& c_out, int tile, int n_split) {
  const int hpk = local_heads / kv_heads;
  const int kwidth = kv_heads * D, vwidth = kv_heads * D;
  c_out.assign(static_cast<size_t>(local_heads) * D, 0.f);
  std::vector<float> s(static_cast<size_t>(n));
  const int T = tile > 0 ? tile : (n > 0 ? n : 1);
  const int tiles = (n + T - 1) / T;
  const int splits = tile > 0 ? n_split : 1;
  const int chunk = (tiles + splits - 1) / splits;
  for (int h = 0; h < local_heads; ++h) {
    const int kvh = h / hpk;
    for (int t = 0; t < n; ++t) {
      float acc = 0.f;
      for (int d = 0; d < D; ++d)
        acc = std::fma(bf16_bits_to_float(q[h * D + d]),
                       bf16_bits_to_float(k_rows[static_cast<int64_t>(t) * kwidth + kvh * D + d]), acc);
      s[static_cast<size_t>(t)] = acc * scale;
    }
    std::vector<float> m_s(static_cast<size_t>(splits), -INFINITY), l_s(static_cast<size_t>(splits), 0.f);
    std::vector<std::vector<float>> c_s(static_cast<size_t>(splits), std::vector<float>(static_cast<size_t>(D), 0.f));
    if (sink) {
      m_s[0] = sink[h];
      l_s[0] = 1.0f;
    }
    for (int sp = 0; sp < splits; ++sp) {
      const int t0s = sp * chunk * T, t1s = std::min(n, (sp + 1) * chunk * T);
      for (int t0 = t0s; t0 < t1s; t0 += T) {
        const int t1 = std::min(t1s, t0 + T);
        float tile_max = -INFINITY;
        for (int t = t0; t < t1; ++t) tile_max = std::max(tile_max, s[static_cast<size_t>(t)]);
        const float m_new = std::max(m_s[static_cast<size_t>(sp)], tile_max);
        const float rescale = std::exp(m_s[static_cast<size_t>(sp)] - m_new);
        float ladd = 0.f;
        for (int t = t0; t < t1; ++t) ladd += std::exp(s[static_cast<size_t>(t)] - m_new);
        l_s[static_cast<size_t>(sp)] = l_s[static_cast<size_t>(sp)] * rescale + ladd;
        std::vector<float>& c = c_s[static_cast<size_t>(sp)];
        for (int d = 0; d < D; ++d) c[static_cast<size_t>(d)] *= rescale;
        for (int t = t0; t < t1; ++t) {
          const float p = rb(std::exp(s[static_cast<size_t>(t)] - m_new));
          for (int d = 0; d < D; ++d)
            c[static_cast<size_t>(d)] = std::fma(p, bf16_bits_to_float(v_rows[static_cast<int64_t>(t) * vwidth + kvh * D + d]),
                                                 c[static_cast<size_t>(d)]);
        }
        m_s[static_cast<size_t>(sp)] = m_new;
      }
    }
    float M = -INFINITY;
    for (int sp = 0; sp < splits; ++sp) M = std::max(M, m_s[static_cast<size_t>(sp)]);
    float L = 0.f;
    std::vector<float> C(static_cast<size_t>(D), 0.f);
    if (M > -INFINITY) {
      for (int sp = 0; sp < splits; ++sp) {
        const float w = std::exp(m_s[static_cast<size_t>(sp)] - M);
        L = std::fma(l_s[static_cast<size_t>(sp)], w, L);
        for (int d = 0; d < D; ++d)
          C[static_cast<size_t>(d)] = std::fma(c_s[static_cast<size_t>(sp)][static_cast<size_t>(d)], w, C[static_cast<size_t>(d)]);
      }
    }
    for (int d = 0; d < D; ++d) c_out[static_cast<size_t>(h) * D + d] = L > 0.f ? C[static_cast<size_t>(d)] / L : 0.f;
  }
}

}  // namespace dgpp::flint_ref
