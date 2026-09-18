#include "models/qwen/qsa_reference.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "common/dtypes.hpp"
#include "kernels/rope_scaling.hpp"

namespace dgpp::qwen_ref {
namespace {
float rb(float v) { return bf16_bits_to_float(float_to_bf16_bits(v)); }
float sigmoid_f(float v) { return 1.0f / (1.0f + std::exp(-v)); }
}  // namespace

void rope_inv_freq(double theta, int rotary_dim, std::vector<float>& inv_freq) {
  inv_freq.assign(static_cast<size_t>(rotary_dim / 2), 0.f);
  const float base = static_cast<float>(theta);
  for (int i = 0; i < rotary_dim / 2; ++i) {
    const float e = static_cast<float>(2 * i) / static_cast<float>(rotary_dim);
    inv_freq[static_cast<size_t>(i)] = 1.0f / std::pow(base, e);
  }
}

void rope_inv_freq_yarn(double theta, int rotary_dim, int64_t correction_max_position,
                        double factor, double beta_fast, double beta_slow,
                        std::vector<float>& inv_freq) {
  inv_freq.assign(static_cast<size_t>(rotary_dim / 2), 0.f);
  yarn_rope_inv_freq_host(rotary_dim, theta, correction_max_position, factor, beta_fast,
                          beta_slow, inv_freq.data());
}

void qsa_norm_rope(const uint16_t* x, const uint16_t* w, int64_t pos, const float* inv_freq,
                   uint16_t* out, int dim, int rotary_dim, float eps, float mscale) {
  std::vector<float> xn(static_cast<size_t>(dim));
  float ss = 0.f;
  for (int d = 0; d < dim; ++d) {
    const float v = bf16_bits_to_float(x[d]);
    ss += v * v;
  }
  const float rstd = 1.0f / std::sqrt(ss / static_cast<float>(dim) + eps);
  for (int d = 0; d < dim; ++d)
    xn[static_cast<size_t>(d)] = rb(bf16_bits_to_float(x[d]) * rstd * (1.0f + bf16_bits_to_float(w[d])));
  const int half = rotary_dim / 2;
  for (int d = 0; d < dim; ++d) {
    if (d >= rotary_dim) {
      out[d] = float_to_bf16_bits(xn[static_cast<size_t>(d)]);
      continue;
    }
    const int i = d < half ? d : d - half;
    const float ang = static_cast<float>(pos) * inv_freq[i];
    const float c = rb(std::cos(ang) * mscale);
    const float s = rb(std::sin(ang) * mscale);
    const float rot = d < half ? -xn[static_cast<size_t>(d + half)] : xn[static_cast<size_t>(d - half)];
    const float t1 = rb(xn[static_cast<size_t>(d)] * c);
    const float t2 = rb(rot * s);
    out[d] = float_to_bf16_bits(t1 + t2);
  }
}

void qsa_index_compress(const uint16_t* raw, int kpool, const uint16_t* w_k, const float* inv_freq,
                        int64_t pos, uint16_t* out, int dim, int rotary_dim, float eps,
                        float mscale) {
  std::vector<uint16_t> mean(static_cast<size_t>(dim));
  for (int d = 0; d < dim; ++d) {
    float acc = 0.f;
    for (int s = 0; s < kpool; ++s) acc += bf16_bits_to_float(raw[static_cast<int64_t>(s) * dim + d]);
    mean[static_cast<size_t>(d)] = float_to_bf16_bits(acc / static_cast<float>(kpool));
  }
  qsa_norm_rope(mean.data(), w_k, pos, inv_freq, out, dim, rotary_dim, eps, mscale);
}

float qsa_index_score(const uint16_t* q, const uint16_t* c, int heads) {
  // Lane (h, chunk): a 16-dim fma chain; the head's 8 lanes in a 3-level
  // xor tree; relu; the 4 heads in a 2-level tree; one division. Absent
  // heads (heads < 4) are zero lanes.
  float lane[32];
  for (int l = 0; l < 32; ++l) {
    const int h = l >> 3, ch = l & 7;
    float p = 0.f;
    if (h < heads)
      for (int j = 0; j < 16; ++j)
        p = std::fma(bf16_bits_to_float(q[h * 128 + ch * 16 + j]), bf16_bits_to_float(c[ch * 16 + j]), p);
    lane[l] = p;
  }
  auto xor_step = [&](int off) {
    float next[32];
    for (int l = 0; l < 32; ++l) next[l] = lane[l] + lane[l ^ off];
    for (int l = 0; l < 32; ++l) lane[l] = next[l];
  };
  xor_step(1);
  xor_step(2);
  xor_step(4);
  for (int l = 0; l < 32; ++l) lane[l] = std::max(lane[l], 0.f);
  xor_step(8);
  xor_step(16);
  return lane[0] / std::sqrt(128.0f);
}

void qsa_select(const std::vector<float>& scores, int select_k, std::vector<int32_t>& ids) {
  std::vector<int32_t> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
    if (scores[static_cast<size_t>(a)] != scores[static_cast<size_t>(b)])
      return scores[static_cast<size_t>(a)] > scores[static_cast<size_t>(b)];
    return a < b;
  });
  const size_t n = std::min(static_cast<size_t>(select_k), order.size());
  ids.assign(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(n));
  std::sort(ids.begin(), ids.end());
}

int qsa_expand(const std::vector<int32_t>& pool_ids, int64_t pos, int kpool,
               std::vector<int32_t>& tokens) {
  tokens.clear();
  for (int32_t p : pool_ids)
    for (int s = 0; s < kpool; ++s) tokens.push_back(p * kpool + s);
  const int64_t seq_len = pos + 1;
  const int64_t tail_start = (seq_len / kpool) * kpool;
  for (int64_t t = tail_start; t < seq_len; ++t) tokens.push_back(static_cast<int32_t>(t));
  return static_cast<int>(tokens.size());
}

void qsa_attention(const uint16_t* q, const uint16_t* k_rows, const uint16_t* v_rows, int n,
                   int local_heads, int kv_heads, int dim, float scale, std::vector<float>& c_out) {
  const int heads_per_kv = local_heads / kv_heads;
  const int width = kv_heads * dim;
  c_out.assign(static_cast<size_t>(local_heads) * dim, 0.f);
  std::vector<float> s(static_cast<size_t>(n));
  for (int h = 0; h < local_heads; ++h) {
    const int kvh = h / heads_per_kv;
    float m = -INFINITY;
    for (int t = 0; t < n; ++t) {
      float acc = 0.f;
      for (int d = 0; d < dim; ++d)
        acc += bf16_bits_to_float(q[h * dim + d]) *
               bf16_bits_to_float(k_rows[static_cast<int64_t>(t) * width + kvh * dim + d]);
      s[static_cast<size_t>(t)] = acc * scale;
      m = std::max(m, s[static_cast<size_t>(t)]);
    }
    float l = 0.f;
    std::vector<float> c(static_cast<size_t>(dim), 0.f);
    for (int t = 0; t < n; ++t) {
      const float e = std::exp(s[static_cast<size_t>(t)] - m);
      l += e;
      const float p = rb(e);
      for (int d = 0; d < dim; ++d)
        c[static_cast<size_t>(d)] += p * bf16_bits_to_float(v_rows[static_cast<int64_t>(t) * width + kvh * dim + d]);
    }
    for (int d = 0; d < dim; ++d)
      c_out[static_cast<size_t>(h) * dim + d] = (l > 0.f && n > 0) ? c[static_cast<size_t>(d)] / l : 0.f;
  }
}

void qsa_gate_out(const float* c, const uint16_t* gate, int64_t gate_head_stride, uint16_t* out,
                  int heads, int dim) {
  for (int h = 0; h < heads; ++h)
    for (int d = 0; d < dim; ++d) {
      const float o = rb(c[static_cast<int64_t>(h) * dim + d]);
      const float gs = rb(sigmoid_f(bf16_bits_to_float(gate[h * gate_head_stride + d])));
      out[static_cast<int64_t>(h) * dim + d] = float_to_bf16_bits(o * gs);
    }
}

}  // namespace dgpp::qwen_ref
