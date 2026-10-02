#include "models/qwen/gdn_reference.hpp"

#include <cmath>
#include <vector>

#include "common/dtypes.hpp"

namespace dgpp::gdn_ref {
namespace {

template <typename Acc>
Acc to_acc(uint16_t bits) {
  return static_cast<Acc>(bf16_bits_to_float(bits));
}

template <typename Acc>
Acc softplus(Acc x) {
  // torch.nn.functional.softplus, beta 1, threshold 20.
  return x > Acc(20) ? x : std::log1p(std::exp(x));
}

}  // namespace

template <typename Acc>
void recurrent(const uint16_t* qkv, const uint16_t* a_raw, int64_t a_row_stride,
               const uint16_t* beta_raw, int64_t beta_row_stride,
               const float* a_log, const float* dt_bias, Acc* state,
               uint16_t* out, int tokens, int heads, int kv_ratio, int k_dim,
               int v_dim, float scale) {
  const int heads_k = heads / kv_ratio;
  const int64_t qkv_stride = static_cast<int64_t>(2) * heads_k * k_dim +
                             static_cast<int64_t>(heads) * v_dim;
  std::vector<Acc> kq(static_cast<size_t>(k_dim)), qq(static_cast<size_t>(k_dim));
  for (int h = 0; h < heads; ++h) {
    const int hk = h / kv_ratio;
    const Acc a = std::exp(static_cast<Acc>(a_log[h]));
    Acc* s = state + static_cast<int64_t>(h) * v_dim * k_dim;
    for (int t = 0; t < tokens; ++t) {
      const uint16_t* qrow =
          qkv + static_cast<int64_t>(t) * qkv_stride + static_cast<int64_t>(hk) * k_dim;
      const uint16_t* krow = qrow + static_cast<int64_t>(heads_k) * k_dim;
      const uint16_t* vrow = qkv + static_cast<int64_t>(t) * qkv_stride +
                             static_cast<int64_t>(2) * heads_k * k_dim +
                             static_cast<int64_t>(h) * v_dim;
      // l2norm with eps inside the sqrt, then the K^-1/2 scaling of q.
      Acc qs = 0, ks = 0;
      for (int i = 0; i < k_dim; ++i) {
        qs += to_acc<Acc>(qrow[i]) * to_acc<Acc>(qrow[i]);
        ks += to_acc<Acc>(krow[i]) * to_acc<Acc>(krow[i]);
      }
      const Acc qn = Acc(1) / std::sqrt(qs + Acc(1e-6));
      const Acc kn = Acc(1) / std::sqrt(ks + Acc(1e-6));
      for (int i = 0; i < k_dim; ++i) {
        kq[i] = to_acc<Acc>(krow[i]) * kn;
        qq[i] = to_acc<Acc>(qrow[i]) * qn * static_cast<Acc>(scale);
      }
      // The scalar decay: exp(-exp(A_log) * softplus(a_raw + dt_bias)).
      const Acc x = to_acc<Acc>(a_raw[static_cast<int64_t>(t) * a_row_stride + h]) +
                    static_cast<Acc>(dt_bias[h]);
      const Acc decay = std::exp(-(a * softplus<Acc>(x)));
      for (int64_t i = 0; i < static_cast<int64_t>(v_dim) * k_dim; ++i) s[i] *= decay;
      const Acc beta = Acc(1) / (Acc(1) + std::exp(-to_acc<Acc>(
                                     beta_raw[static_cast<int64_t>(t) * beta_row_stride + h])));
      for (int v = 0; v < v_dim; ++v) {
        Acc* srow = s + static_cast<int64_t>(v) * k_dim;
        Acc dot = 0;
        for (int i = 0; i < k_dim; ++i) dot += srow[i] * kq[i];
        const Acc u = (to_acc<Acc>(vrow[v]) - dot) * beta;
        Acc o = 0;
        for (int i = 0; i < k_dim; ++i) {
          srow[i] += u * kq[i];
          o += srow[i] * qq[i];
        }
        out[(static_cast<int64_t>(t) * heads + h) * v_dim + v] =
            float_to_bf16_bits(static_cast<float>(o));
      }
    }
  }
}

void gated_rmsnorm_sigmoid(const uint16_t* x, const uint16_t* gate,
                           const uint16_t* w, uint16_t* y, int64_t rows,
                           int dim, float eps) {
  for (int64_t r = 0; r < rows; ++r) {
    const uint16_t* xr = x + r * dim;
    const uint16_t* gr = gate + r * dim;
    uint16_t* yr = y + r * dim;
    float ss = 0.0f;
    for (int i = 0; i < dim; ++i) {
      const float v = bf16_bits_to_float(xr[i]);
      ss = std::fma(v, v, ss);
    }
    const float rstd = 1.0f / std::sqrt(ss / static_cast<float>(dim) + eps);
    for (int i = 0; i < dim; ++i) {
      const float u = bf16_bits_to_float(float_to_bf16_bits(bf16_bits_to_float(xr[i]) * rstd));
      const float p = bf16_bits_to_float(float_to_bf16_bits(u * bf16_bits_to_float(w[i])));
      const float g = bf16_bits_to_float(gr[i]);
      const float sig = 1.0f / (1.0f + std::exp(-g));
      yr[i] = float_to_bf16_bits(p * sig);
    }
  }
}

void gated_rmsnorm_swish(const uint16_t* x, const uint16_t* gate,
                         const uint16_t* w, uint16_t* y, int64_t rows,
                         int dim, float eps) {
  for (int64_t r = 0; r < rows; ++r) {
    const uint16_t* xr = x + r * dim;
    const uint16_t* gr = gate + r * dim;
    uint16_t* yr = y + r * dim;
    float ss = 0.0f;
    for (int i = 0; i < dim; ++i) {
      const float v = bf16_bits_to_float(xr[i]);
      ss = std::fma(v, v, ss);
    }
    const float rstd = 1.0f / std::sqrt(ss / static_cast<float>(dim) + eps);
    for (int i = 0; i < dim; ++i) {
      const float u = bf16_bits_to_float(float_to_bf16_bits(bf16_bits_to_float(xr[i]) * rstd));
      const float p = bf16_bits_to_float(float_to_bf16_bits(u * bf16_bits_to_float(w[i])));
      const float g = bf16_bits_to_float(gr[i]);
      const float sig = 1.0f / (1.0f + std::exp(-g));
      yr[i] = float_to_bf16_bits(p * (g * sig));
    }
  }
}

template void recurrent<float>(const uint16_t*, const uint16_t*, int64_t, const uint16_t*, int64_t,
                               const float*, const float*, float*, uint16_t*, int, int, int, int,
                               int, float);
template void recurrent<double>(const uint16_t*, const uint16_t*, int64_t, const uint16_t*, int64_t,
                                const float*, const float*, double*, uint16_t*, int, int, int, int,
                                int, float);

}  // namespace dgpp::gdn_ref
