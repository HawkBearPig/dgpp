#include "models/minimax/reference.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "common/dtypes.hpp"

namespace dgpp::minimax_ref {
float swiglu(float gate, float up, float alpha, float limit) {
  gate = std::min(gate, limit);
  up = std::clamp(up, -limit, limit);
  return (gate / (1.0f + std::exp(-alpha * gate))) * (up + 1.0f);
}
void gemma_norm(const uint16_t* x, const uint16_t* weight, uint16_t* y, int dim, float eps) {
  if (dim <= 0) throw std::invalid_argument("MiniMax reference: invalid norm dimension");
  double sum = 0;
  for (int i = 0; i < dim; ++i) {
    const double v = bf16_bits_to_float(x[i]);
    sum += v * v;
  }
  const float rstd = 1.0f / std::sqrt(static_cast<float>(sum / dim) + eps);
  for (int i = 0; i < dim; ++i)
    y[i] = float_to_bf16_bits(bf16_bits_to_float(x[i]) * rstd *
                              (1.0f + bf16_bits_to_float(weight[i])));
}
void norm_rope(const uint16_t* x, const uint16_t* weight, uint16_t* y, int dim, int rotary_dim,
               float eps, double theta, int64_t pos) {
  if (rotary_dim <= 0 || rotary_dim > dim || rotary_dim % 2)
    throw std::invalid_argument("invalid RoPE width");
  gemma_norm(x, weight, y, dim, eps);
  const int half = rotary_dim / 2;
  for (int i = 0; i < half; ++i) {
    const float inv =
        static_cast<float>(1.0 / std::pow(theta, static_cast<double>(2 * i) / rotary_dim));
    const float a = static_cast<float>(pos) * inv;
    const float cs = std::cos(a), sn = std::sin(a);
    const float lo = bf16_bits_to_float(y[i]), hi = bf16_bits_to_float(y[i + half]);
    y[i] = float_to_bf16_bits(lo * cs - hi * sn);
    y[i + half] = float_to_bf16_bits(hi * cs + lo * sn);
  }
}
std::vector<int> select_blocks(const uint16_t* q, const uint16_t* keys, int64_t pos, int topk) {
  if (topk <= 0 || topk > 16) throw std::invalid_argument("MiniMax reference: invalid top-k");
  if (pos < 0) return {};
  const int blocks = static_cast<int>(pos / 128 + 1);
  std::vector<float> scores(static_cast<size_t>(blocks), -INFINITY);
  for (int64_t t = 0; t <= pos; ++t) {
    float score = 0;
    for (int d = 0; d < 128; ++d)
      score = std::fma(bf16_bits_to_float(q[d]), bf16_bits_to_float(keys[t * 128 + d]), score);
    scores[static_cast<size_t>(t / 128)] = std::max(scores[static_cast<size_t>(t / 128)], score);
  }
  scores.back() = INFINITY;
  std::vector<int> ids(static_cast<size_t>(blocks));
  std::iota(ids.begin(), ids.end(), 0);
  std::sort(ids.begin(), ids.end(),
            [&](int a, int b) { return scores[a] == scores[b] ? a < b : scores[a] > scores[b]; });
  ids.resize(static_cast<size_t>(std::min(blocks, topk)));
  std::sort(ids.begin(), ids.end());
  return ids;
}
std::vector<float> attention(const uint16_t* q, const uint16_t* k, const uint16_t* v, int head,
                             int heads, int kv_heads, int64_t pos, const std::vector<int>& blocks) {
  if (heads <= 0 || kv_heads <= 0 || heads % kv_heads || head < 0 || head >= heads)
    throw std::invalid_argument("MiniMax reference: invalid GQA geometry");
  const int kh = head / (heads / kv_heads);
  std::vector<int64_t> tokens;
  std::vector<float> scores;
  float maximum = -INFINITY;
  for (int b : blocks)
    for (int64_t t = int64_t(b) * 128; t < int64_t(b + 1) * 128 && t <= pos; ++t) {
      float s = 0;
      for (int d = 0; d < 128; ++d)
        s = std::fma(bf16_bits_to_float(q[head * 128 + d]),
                     bf16_bits_to_float(k[(t * kv_heads + kh) * 128 + d]), s);
      s *= static_cast<float>(1.0 / std::sqrt(128.0));
      tokens.push_back(t);
      scores.push_back(s);
      maximum = std::max(maximum, s);
    }
  double denominator = 0;
  std::vector<double> sum(128, 0);
  for (size_t i = 0; i < tokens.size(); ++i) {
    const double p = std::exp(static_cast<double>(scores[i] - maximum));
    denominator += p;
    for (int d = 0; d < 128; ++d)
      sum[d] += p * bf16_bits_to_float(v[(tokens[i] * kv_heads + kh) * 128 + d]);
  }
  std::vector<float> result(128, 0);
  if (denominator > 0)
    for (int d = 0; d < 128; ++d) result[d] = static_cast<float>(sum[d] / denominator);
  return result;
}
void route(const float* logits, const float* bias, int experts, int topk, float factor,
           std::vector<int>& ids, std::vector<float>& weights) {
  if (topk <= 0 || topk > experts) throw std::invalid_argument("invalid routing geometry");
  std::vector<float> scores(static_cast<size_t>(experts));
  ids.resize(static_cast<size_t>(experts));
  std::iota(ids.begin(), ids.end(), 0);
  for (int e = 0; e < experts; ++e) scores[e] = 1.0f / (1.0f + std::exp(-logits[e]));
  std::sort(ids.begin(), ids.end(), [&](int a, int b) {
    const float sa = scores[a] + bias[a], sb = scores[b] + bias[b];
    return sa == sb ? a < b : sa > sb;
  });
  ids.resize(static_cast<size_t>(topk));
  std::sort(ids.begin(), ids.end());
  float sum = 0;
  for (int e : ids) sum += scores[e];
  weights.resize(static_cast<size_t>(topk));
  for (int j = 0; j < topk; ++j) weights[j] = scores[ids[j]] / std::max(sum, 1e-20f) * factor;
}
}  // namespace dgpp::minimax_ref
