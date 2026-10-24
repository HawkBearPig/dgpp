// The CSA2 kernels (kernels/csa2.hpp) against host oracles that mirror
// inference/model.py + kernel.py: the one-rounding RMSNorm; the rotary
// frequencies (window and YaRN) and the one-rounding complex rotation
// (forward, inverse, positions across 2^20); the indexer's fp4 e8m0/32
// forms stored as e4m3 + a row scale (bitwise the quantize-dequantize,
// the violation counter on a crafted row, the block-table append); the
// compressor at ratio 2 (pairs across a chunk, the odd tail, the decode
// update in spans with per-row snapshots and the rollback equivalence);
// the window slot lists and the ring scratch round trip; the three
// selections against the DSA oracle's logits and a candidate oracle
// (block max, the newest block pinned, expansion with the partial tail);
// the two-source attention finish with the sink; and a window attention
// end to end (fp8_block ring rows, dsa_attn_partial, the finish) against
// the reference's sparse_attn arithmetic.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/csa2.hpp"
#include "kernels/dsa.hpp"
#include "kernels/latent_format.hpp"
#include "kernels/topk_select.cuh"
#include "models/dsa_reference.hpp"

namespace {
using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;
using dgpp::LatentFormat;
using dgpp::kda_test::compare_bf16;
using dgpp::kda_test::DevBuf;
using dgpp::kda_test::random_bf16_bits;
using dgpp::kda_test::require_bf16;
using dgpp::kda_test::require_bitwise;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}
template <typename T>
DevBuf upload(const std::vector<T>& v) {
  DevBuf b(v.size() * sizeof(T));
  b.upload(v.data(), v.size() * sizeof(T));
  return b;
}
template <typename T>
std::vector<T> download(const DevBuf& b, size_t n) {
  std::vector<T> v(n);
  b.download(v.data(), n * sizeof(T));
  return v;
}
void sync() { DGPP_CUDA_OK(cudaDeviceSynchronize()); }

// ---- oracles ------------------------------------------------------------------------------
// The reference RMSNorm: fp32 interior (double here to measure the kernel's
// fp32 drift), one bf16 rounding.
void ref_rmsnorm(const uint16_t* x, const uint16_t* w, int dim, float eps, uint16_t* y) {
  double ss = 0.0;
  for (int i = 0; i < dim; ++i) {
    const double v = bf16_bits_to_float(x[i]);
    ss += v * v;
  }
  const double rs = 1.0 / std::sqrt(ss / dim + static_cast<double>(eps));
  for (int i = 0; i < dim; ++i)
    y[i] = float_to_bf16_bits(static_cast<float>(bf16_bits_to_float(w[i]) * (bf16_bits_to_float(x[i]) * rs)));
}
// precompute_freqs_cis in double.
std::vector<double> ref_inv_freq(int dim, double base, int64_t orig, double factor, double bf, double bs) {
  const int half = dim / 2;
  std::vector<double> f(static_cast<size_t>(half));
  for (int i = 0; i < half; ++i) f[size_t(i)] = 1.0 / std::pow(base, (2.0 * i) / dim);
  if (orig > 0) {
    auto corrected = [&](double rot) { return dim * std::log(orig / (rot * 2 * 3.14159265358979323846)) / (2 * std::log(base)); };
    const int low = std::max(static_cast<int>(std::floor(corrected(bf))), 0);
    const int high = std::min(static_cast<int>(std::ceil(corrected(bs))), dim - 1);
    for (int i = 0; i < half; ++i) {
      double ramp = (i - low) / std::max(double(high - low), 1e-3);
      ramp = std::min(std::max(ramp, 0.0), 1.0);
      const double smooth = 1 - ramp;
      f[size_t(i)] = f[size_t(i)] / factor * (1 - smooth) + f[size_t(i)] * smooth;
    }
  }
  return f;
}
// apply_rotary_emb over one segment (double, one bf16 rounding) at the
// kernel's fp32 angle.
void ref_rope(uint16_t* x, int rope, int64_t pos, const float* inv_freq, bool inverse) {
  for (int i = 0; i < rope / 2; ++i) {
    const float ang = static_cast<float>(pos) * inv_freq[i];
    const double c = std::cos(static_cast<double>(ang));
    double s = std::sin(static_cast<double>(ang));
    if (inverse) s = -s;
    const double x0 = bf16_bits_to_float(x[2 * i]), x1 = bf16_bits_to_float(x[2 * i + 1]);
    x[2 * i] = float_to_bf16_bits(static_cast<float>(x0 * c - x1 * s));
    x[2 * i + 1] = float_to_bf16_bits(static_cast<float>(x1 * c + x0 * s));
  }
}
// fp4_act_quant(block 32, e8m0) quantize-dequantize of a 128-wide row (fp32).
void ref_fp4_e8m0_dequant(const uint16_t* x, float* out) {
  for (int b = 0; b < 4; ++b) {
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(bf16_bits_to_float(x[b * 32 + j])));
    const float floor_v = 6.0f * 1.1754943508222875e-38f;
    amax = amax > floor_v ? amax : floor_v;
    const float s = dgpp::e8m0_byte_to_float(dgpp::e8m0_ceil_log2_byte(amax * (1.0f / 6.0f)));
    for (int j = 0; j < 32; ++j) {
      const float v = bf16_bits_to_float(x[b * 32 + j]);
      out[b * 32 + j] = dgpp::fp4_e2m1_bits_to_float(dgpp::float_to_fp4_e2m1_bits(v / s)) * s;
    }
  }
}
// The compressor pair: softmax(score) per channel, the kv sum in fp32,
// bf16, then the RMSNorm (the kernel's fp32 arithmetic mirrored in float
// for the pooling, double for the norm).
void ref_pool_pair(const float* kv0, const float* s0, const float* kv1, const float* s1, const uint16_t* w,
                   float eps, uint16_t* out) {
  std::vector<uint16_t> pooled(512);
  for (int c = 0; c < 512; ++c) {
    const float m = std::max(s0[c], s1[c]);
    const float e0 = std::exp(s0[c] - m), e1 = std::exp(s1[c] - m);
    const float w0 = e0 / (e0 + e1), w1 = e1 / (e0 + e1);
    pooled[size_t(c)] = float_to_bf16_bits(kv0[c] * w0 + kv1[c] * w1);
  }
  ref_rmsnorm(pooled.data(), w, 512, eps, out);
}
// The candidate oracle: entries [0, visible) with logits; block max over
// block_size, the newest complete block pinned when the newest entry
// completes it, the topk_blocks best (score desc, id asc) as ascending
// block ids; ref_pool_entries expands them plus the partial newest block.
std::vector<int32_t> ref_candidate_blocks(const float* logits, int64_t visible, int block_size, int topk_blocks) {
  std::vector<int32_t> out;
  if (visible <= 0) return out;
  const int64_t nb = visible / block_size;
  std::vector<std::pair<float, int64_t>> blocks;
  for (int64_t b = 0; b < nb; ++b) {
    float best = -INFINITY;
    for (int k = 0; k < block_size; ++k) best = std::max(best, logits[b * block_size + k]);
    if (visible % block_size == 0 && b == nb - 1) best = INFINITY;
    blocks.emplace_back(best, b);
  }
  std::sort(blocks.begin(), blocks.end(), [](const auto& a, const auto& b) {
    return a.first > b.first || (a.first == b.first && a.second < b.second);
  });
  const size_t keep = std::min(static_cast<size_t>(topk_blocks), blocks.size());
  std::vector<int64_t> ids;
  for (size_t i = 0; i < keep; ++i) ids.push_back(blocks[i].second);
  std::sort(ids.begin(), ids.end());
  for (const int64_t b : ids) out.push_back(static_cast<int32_t>(b));
  return out;
}
std::vector<int32_t> ref_pool_entries(const std::vector<int32_t>& blocks, int64_t visible, int block_size) {
  std::vector<int32_t> out;
  for (const int32_t b : blocks)
    for (int k = 0; k < block_size; ++k) out.push_back(b * block_size + k);
  for (int64_t e = (visible / block_size) * block_size; e < visible; ++e) out.push_back(static_cast<int32_t>(e));
  return out;
}
std::vector<int32_t> ref_topk_among(const float* logits, const std::vector<int32_t>& cand, int64_t visible,
                                    int select_k) {
  std::vector<std::pair<float, int32_t>> v;
  for (const int32_t e : cand)
    if (e >= 0 && e < visible) v.emplace_back(logits[e], e);
  std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
    return a.first > b.first || (a.first == b.first && a.second < b.second);
  });
  std::vector<int32_t> out;
  for (size_t i = 0; i < v.size() && i < static_cast<size_t>(select_k); ++i) out.push_back(v[i].second);
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace

DGPP_TEST(csa2_rmsnorm_one_rounding_matches_the_reference) {
  for (const int dim : {128, 512, 1280}) {
    const int rows = 5;
    const auto x = random_bf16_bits(0xC5A2 + dim, int64_t(rows) * dim, -6, 3);
    const auto w = random_bf16_bits(0xC5A3 + dim, dim, -3, 1);
    DevBuf dx = upload(x), dw = upload(w), dy(x.size() * 2);
    dgpp::csa2_rmsnorm_bf16(dx.p, dim, dw.p, dy.p, dim, rows, dim, 1e-20f, 0);
    sync();
    const auto got = download<uint16_t>(dy, x.size());
    std::vector<uint16_t> want(x.size());
    for (int r = 0; r < rows; ++r) ref_rmsnorm(&x[size_t(r) * dim], w.data(), dim, 1e-20f, &want[size_t(r) * dim]);
    require_bf16("rmsnorm dim " + std::to_string(dim), compare_bf16(got, want, 2), 1e-3, 0.002);
  }
  // In place, with a wider row stride (a head's tail inside a 512 row).
  {
    const int dim = 128, stride = 512, rows = 3;
    auto x = random_bf16_bits(0xC5A9, int64_t(rows) * stride, -4, 2);
    const auto w = random_bf16_bits(0xC5AA, dim, -3, 1);
    DevBuf dx = upload(x), dw = upload(w);
    dgpp::csa2_rmsnorm_bf16(dx.p, stride, dw.p, dx.p, stride, rows, dim, 1e-20f, 0);
    sync();
    const auto got = download<uint16_t>(dx, x.size());
    for (int r = 0; r < rows; ++r) {
      std::vector<uint16_t> want(dim);
      ref_rmsnorm(&x[size_t(r) * stride], w.data(), dim, 1e-20f, want.data());
      const std::vector<uint16_t> g(got.begin() + r * stride, got.begin() + r * stride + dim);
      require_bf16("rmsnorm strided row", compare_bf16(g, want, 2), 1e-3, 0.02);
      for (int i = dim; i < stride; ++i) require(got[size_t(r) * stride + i] == x[size_t(r) * stride + i], "the rest of the row untouched");
    }
  }
}

DGPP_TEST(csa2_rope_frequencies_and_rotation_match_the_reference) {
  // The window table (theta 10000, no YaRN) and the compressed one (160000,
  // YaRN 16x over 65536, betas 32 / 1).
  std::vector<float> win(32), yarn(32);
  dgpp::csa2_rope_inv_freq_host(64, 10000.0, 0, 16.0, 32.0, 1.0, win.data());
  dgpp::csa2_rope_inv_freq_host(64, 160000.0, 65536, 16.0, 32.0, 1.0, yarn.data());
  const auto rwin = ref_inv_freq(64, 10000.0, 0, 16.0, 32.0, 1.0);
  const auto ryarn = ref_inv_freq(64, 160000.0, 65536, 16.0, 32.0, 1.0);
  for (int i = 0; i < 32; ++i) {
    require(std::fabs(win[size_t(i)] - rwin[size_t(i)]) <= 2e-7 * rwin[size_t(i)], "window inv_freq " + std::to_string(i));
    require(std::fabs(yarn[size_t(i)] - ryarn[size_t(i)]) <= 4e-7 * ryarn[size_t(i)], "yarn inv_freq " + std::to_string(i));
  }
  // The ramp: the first dims keep their frequency, the last are divided by 16.
  require(yarn[0] == win[0] * 0 + yarn[0] && std::fabs(yarn[0] - 1.0f) < 1e-6f, "dim 0 unscaled");
  require(std::fabs(yarn[31] * 16.0f - static_cast<float>(1.0 / std::pow(160000.0, 62.0 / 64.0))) < 1e-9f, "dim 31 divided by the factor");

  // Rotation: rows x heads segments of 64 inside 512-wide heads, positions
  // across 2^20, forward then inverse.
  const int rows = 6, heads = 3, head_dim = 512, rope = 64;
  auto x = random_bf16_bits(0xB0BE, int64_t(rows) * heads * head_dim, -3, 2);
  const std::vector<int64_t> pos = {0, 1, 127, 4096, 1048575, -1};
  DevBuf dx = upload(x), dpos = upload(pos), dfreq = upload(yarn);
  uint16_t* tail = static_cast<uint16_t*>(dx.p) + (head_dim - rope);
  dgpp::csa2_rope_apply(tail, int64_t(heads) * head_dim, head_dim, heads, rope, static_cast<const int64_t*>(dpos.p),
                        static_cast<const float*>(dfreq.p), false, rows, 0);
  sync();
  const auto got = download<uint16_t>(dx, x.size());
  std::vector<uint16_t> want = x;
  for (int r = 0; r < rows; ++r) {
    if (pos[size_t(r)] < 0) continue;
    for (int h = 0; h < heads; ++h)
      ref_rope(&want[(size_t(r) * heads + h) * head_dim + head_dim - rope], rope, pos[size_t(r)], yarn.data(), false);
  }
  require_bf16("rope forward", compare_bf16(got, want, 2), 2e-3, 0.01);
  for (size_t i = 0; i < x.size(); ++i)
    if (i % head_dim < size_t(head_dim - rope)) require(got[i] == x[i], "elements outside the tail untouched");
  require(std::equal(got.begin() + 5 * heads * head_dim, got.end(), x.begin() + 5 * heads * head_dim), "a padding row untouched");
  // Inverse: the rotation removed (within the two bf16 roundings).
  dgpp::csa2_rope_apply(tail, int64_t(heads) * head_dim, head_dim, heads, rope, static_cast<const int64_t*>(dpos.p),
                        static_cast<const float*>(dfreq.p), true, rows, 0);
  sync();
  const auto back = download<uint16_t>(dx, x.size());
  require_bf16("rope inverse restores the input", compare_bf16(back, x, 3), 3e-3, 0.02);
}

DGPP_TEST(csa2_index_quant_is_exact_in_the_planar_form_and_counts_violations) {
  const int rows = 4, heads = 32;
  auto q = random_bf16_bits(0x1DE, int64_t(rows) * heads * 128, -5, 3);
  DevBuf dq = upload(q), dq8(size_t(rows) * heads * 128), dqs(size_t(rows) * heads * 4), dviol(4);
  DGPP_CUDA_OK(cudaMemset(dviol.p, 0, 4));
  dgpp::csa2_index_q_quant(dq.p, rows, heads, dq8.p, static_cast<float*>(dqs.p), static_cast<unsigned*>(dviol.p), 0);
  sync();
  const auto codes = download<uint8_t>(dq8, size_t(rows) * heads * 128);
  const auto scales = download<float>(dqs, size_t(rows) * heads);
  require(download<unsigned>(dviol, 1)[0] == 0, "no violations on rows within 14 binades");
  for (int64_t rh = 0; rh < int64_t(rows) * heads; ++rh) {
    float want[128];
    ref_fp4_e8m0_dequant(&q[size_t(rh) * 128], want);
    const float S = scales[size_t(rh)];
    require(S > 0 && std::ldexp(1.0, static_cast<int>(std::log2(S))) == S, "a power-of-two row scale");
    for (int d = 0; d < 128; ++d) {
      const float got = dgpp::fp8_e4m3_bits_to_float(codes[size_t(rh) * 128 + d]) * S;
      require(got == want[d], "index q value " + std::to_string(rh) + ":" + std::to_string(d) + " bitwise the fp4 quant-dequant");
    }
  }
  // A crafted row: block 0 at 2^0 magnitudes, block 3 at 2^-20 — beyond 14
  // binades, the small block's codes are lost and the row is counted.
  std::vector<uint16_t> bad(128, float_to_bf16_bits(0.0f));
  for (int j = 0; j < 32; ++j) bad[size_t(j)] = float_to_bf16_bits(1.0f + 0.03f * j);
  for (int j = 0; j < 32; ++j) bad[size_t(96 + j)] = float_to_bf16_bits(std::ldexp(1.0f + 0.03f * j, -20));
  DevBuf db = upload(bad), db8(128), dbs(4);
  DGPP_CUDA_OK(cudaMemset(dviol.p, 0, 4));
  dgpp::csa2_index_q_quant(db.p, 1, 1, db8.p, static_cast<float*>(dbs.p), static_cast<unsigned*>(dviol.p), 0);
  sync();
  require(download<unsigned>(dviol, 1)[0] == 1, "a row beyond 14 binades is counted once");

  // The k append through a block table: entries land on their slots.
  const int n = 6, epb = 4;
  const std::vector<int32_t> table = {2, 0, 1};  // request 0's blocks
  const std::vector<int32_t> req_ids(n, 0);
  const std::vector<int64_t> entries = {0, 1, 5, 9, -1, 11};
  auto k = random_bf16_bits(0x1DF, int64_t(n) * 128, -4, 2);
  DevBuf dk = upload(k), dt = upload(table), dri = upload(req_ids), de = upload(entries), dik(size_t(12) * 128), dis(12 * 4);
  DGPP_CUDA_OK(cudaMemset(dik.p, 0xEE, size_t(12) * 128));
  DGPP_CUDA_OK(cudaMemset(dviol.p, 0, 4));
  dgpp::csa2_index_k_append(dk.p, static_cast<const int32_t*>(dri.p), static_cast<const int64_t*>(de.p), n,
                            static_cast<const int32_t*>(dt.p), 3, epb, dik.p, static_cast<float*>(dis.p),
                            static_cast<unsigned*>(dviol.p), 0);
  sync();
  const auto ik = download<uint8_t>(dik, size_t(12) * 128);
  const auto is = download<float>(dis, 12);
  for (int i = 0; i < n; ++i) {
    const int64_t e = entries[size_t(i)];
    if (e < 0) continue;
    const int64_t slot = int64_t(table[size_t(e / epb)]) * epb + e % epb;
    float want[128];
    ref_fp4_e8m0_dequant(&k[size_t(i) * 128], want);
    for (int d = 0; d < 128; ++d)
      require(dgpp::fp8_e4m3_bits_to_float(ik[size_t(slot) * 128 + d]) * is[size_t(slot)] == want[d], "index k slot value");
  }
  // The fold: (bf16 w * 2^-6) * q_scale.
  const auto w = random_bf16_bits(0x1E0, int64_t(rows) * heads, -3, 2);
  DevBuf dw = upload(w), dwf(size_t(rows) * heads * 4);
  dgpp::csa2_fold_weights(dw.p, static_cast<const float*>(dqs.p), static_cast<float*>(dwf.p), int64_t(rows) * heads, 0);
  sync();
  const auto wf = download<float>(dwf, size_t(rows) * heads);
  for (size_t i = 0; i < wf.size(); ++i)
    require(wf[i] == (bf16_bits_to_float(w[i]) * 0.015625f) * scales[i], "folded weight");
}

DGPP_TEST(csa2_compressor_pairs_tails_snapshots_and_rollback) {
  const int T = 9;  // four pairs and an odd tail
  std::vector<float> kv(size_t(T) * 512), score(size_t(T) * 512);
  {
    const auto kb = random_bf16_bits(0xC0, int64_t(T) * 512, -4, 2), sb = random_bf16_bits(0xC1, int64_t(T) * 512, -3, 2);
    for (size_t i = 0; i < kv.size(); ++i) { kv[i] = bf16_bits_to_float(kb[i]); score[i] = bf16_bits_to_float(sb[i]); }
  }
  const auto norm_w = random_bf16_bits(0xC2, 512, -2, 1);
  const float eps = 1e-20f;
  DevBuf dkv = upload(kv), dsc = upload(score), dw = upload(norm_w), dlat(size_t(T / 2) * 512 * 2), dtail(2 * 512 * 4);
  DGPP_CUDA_OK(cudaMemset(dtail.p, 0, 2 * 512 * 4));
  dgpp::csa2_compress_pairs_prefill(static_cast<const float*>(dkv.p), static_cast<const float*>(dsc.p), T, dw.p, eps,
                                    dlat.p, static_cast<float*>(dtail.p), 0);
  sync();
  const auto lat = download<uint16_t>(dlat, size_t(T / 2) * 512);
  std::vector<uint16_t> want(size_t(T / 2) * 512);
  for (int j = 0; j < T / 2; ++j)
    ref_pool_pair(&kv[size_t(2 * j) * 512], &score[size_t(2 * j) * 512], &kv[size_t(2 * j + 1) * 512],
                  &score[size_t(2 * j + 1) * 512], norm_w.data(), eps, &want[size_t(j) * 512]);
  require_bf16("prefill pairs", compare_bf16(lat, want, 2), 1e-3, 0.005);
  const auto tail = download<float>(dtail, 2 * 512);
  for (int c = 0; c < 512; ++c) {
    require(tail[size_t(c)] == kv[size_t(T - 1) * 512 + c], "tail kv");
    require(tail[size_t(512 + c)] == score[size_t(T - 1) * 512 + c], "tail score");
  }

  // Decode: two requests. Request A continues the prefill above at
  // positions 9..13 (odd first: pools with the stashed tail); request B is
  // a fresh one at 20..23. Snapshots after every non-last row.
  const int tokens = 9;
  const std::vector<int32_t> req_ids = {0, 0, 0, 0, 0, 1, 1, 1, 1};
  const std::vector<int64_t> pos = {9, 10, 11, 12, 13, 20, 21, 22, 23};
  const std::vector<int32_t> spans = {0, 5, 5, 4};
  std::vector<float> kvd(size_t(tokens) * 512), scd(size_t(tokens) * 512);
  {
    const auto kb = random_bf16_bits(0xC3, int64_t(tokens) * 512, -4, 2), sb = random_bf16_bits(0xC4, int64_t(tokens) * 512, -3, 2);
    for (size_t i = 0; i < kvd.size(); ++i) { kvd[i] = bf16_bits_to_float(kb[i]); scd[i] = bf16_bits_to_float(sb[i]); }
  }
  std::vector<float> tails(2 * 2 * 512, 0.0f);
  std::copy(tail.begin(), tail.end(), tails.begin());  // request 0's tail = the prefill's
  DevBuf dkvd = upload(kvd), dscd = upload(scd), dri = upload(req_ids), dpos = upload(pos), dsp = upload(spans),
      dtails = upload(tails), dlatd(size_t(tokens) * 512 * 2), dent(size_t(tokens) * 8), dsnap(size_t(tokens) * 2 * 512 * 4);
  DGPP_CUDA_OK(cudaMemset(dsnap.p, 0x7F, size_t(tokens) * 2 * 512 * 4));
  dgpp::csa2_compress_decode_update(static_cast<const float*>(dkvd.p), static_cast<const float*>(dscd.p),
                                    static_cast<const int32_t*>(dri.p), static_cast<const int64_t*>(dpos.p),
                                    static_cast<const int32_t*>(dsp.p), 2, dw.p, eps, static_cast<float*>(dtails.p),
                                    dlatd.p, static_cast<int64_t*>(dent.p), tokens, static_cast<float*>(dsnap.p), 0);
  sync();
  const auto latd = download<uint16_t>(dlatd, size_t(tokens) * 512);
  const auto ent = download<int64_t>(dent, tokens);
  const auto snap = download<float>(dsnap, size_t(tokens) * 2 * 512);
  const std::vector<int64_t> want_ent = {4, -1, 5, -1, 6, -1, 10, -1, 11};
  require(ent == want_ent, "entry ids");
  // Row 0 (pos 9) pools the prefill's tail (position 8) with itself.
  {
    std::vector<uint16_t> w0(512);
    ref_pool_pair(tail.data(), tail.data() + 512, &kvd[0], &scd[0], norm_w.data(), eps, w0.data());
    const std::vector<uint16_t> g(latd.begin(), latd.begin() + 512);
    require_bf16("decode pair across the prefill boundary", compare_bf16(g, w0, 2), 1e-3, 0.005);
  }
  // Rows 2 and 4 pool rows 1 and 3; row 6 / 8 of request B pool rows 5 / 7.
  for (const auto [odd, even] : {std::pair{2, 1}, std::pair{4, 3}, std::pair{6, 5}, std::pair{8, 7}}) {
    std::vector<uint16_t> w(512);
    ref_pool_pair(&kvd[size_t(even) * 512], &scd[size_t(even) * 512], &kvd[size_t(odd) * 512], &scd[size_t(odd) * 512],
                  norm_w.data(), eps, w.data());
    const std::vector<uint16_t> g(latd.begin() + odd * 512, latd.begin() + (odd + 1) * 512);
    require_bf16("decode pair " + std::to_string(odd), compare_bf16(g, w, 2), 1e-3, 0.005);
  }
  for (const int t : {1, 3, 5, 7})
    for (int c = 0; c < 512; ++c) require(latd[size_t(t) * 512 + c] == 0, "a non-entry row's latent is zero");
  // Snapshots: after an even row, its own (kv, score); after an odd row,
  // the last even's (the state is not cleared); the last row has none.
  auto check_snap = [&](int t, int even_row) {
    for (int c = 0; c < 512; ++c) {
      require(snap[(size_t(t) * 2) * 512 + c] == kvd[size_t(even_row) * 512 + c], "snapshot kv row " + std::to_string(t));
      require(snap[(size_t(t) * 2 + 1) * 512 + c] == scd[size_t(even_row) * 512 + c], "snapshot score row " + std::to_string(t));
    }
  };
  check_snap(1, 1); check_snap(2, 1); check_snap(3, 3); check_snap(5, 5); check_snap(6, 5); check_snap(7, 7);
  for (int c = 0; c < 2 * 512; ++c) {
    require(snap[size_t(0) * 1024 + c] == tail[size_t(c)], "row 0's snapshot is the prefill tail (unchanged by an odd row)");
  }
  // The final tails: request 0 holds row 3's stash, request 1 row 7's.
  const auto tf = download<float>(dtails, 2 * 2 * 512);
  for (int c = 0; c < 512; ++c) {
    require(tf[size_t(c)] == kvd[size_t(3) * 512 + c], "final tail request 0");
    require(tf[size_t(2 * 512 + c)] == kvd[size_t(7) * 512 + c], "final tail request 1");
  }
  // Rollback: restoring row 1's snapshot over request 0's tail and replaying
  // rows 2.. gives the same latents (the snapshot IS the state).
  {
    std::vector<float> t2 = tf;
    std::copy(snap.begin() + 1 * 1024, snap.begin() + 2 * 1024, t2.begin());
    DevBuf dt2 = upload(t2), dl2(size_t(tokens) * 512 * 2), de2(size_t(tokens) * 8);
    const std::vector<int32_t> sp2 = {2, 3, 5, 4};
    DevBuf dsp2 = upload(sp2);
    dgpp::csa2_compress_decode_update(static_cast<const float*>(dkvd.p), static_cast<const float*>(dscd.p),
                                      static_cast<const int32_t*>(dri.p), static_cast<const int64_t*>(dpos.p),
                                      static_cast<const int32_t*>(dsp2.p), 2, dw.p, eps, static_cast<float*>(dt2.p),
                                      dl2.p, static_cast<int64_t*>(de2.p), tokens, nullptr, 0);
    sync();
    const auto l2 = download<uint16_t>(dl2, size_t(tokens) * 512);
    require(std::equal(l2.begin() + 2 * 512, l2.end(), latd.begin() + 2 * 512), "the replay after a rollback is bitwise");
  }
}

DGPP_TEST(csa2_window_slots_and_ring_scratch_round_trip) {
  const int window = 128, ring = 160;
  const std::vector<int64_t> pos = {0, 5, 127, 128, 500, -1};
  DevBuf dpos = upload(pos), dlist(size_t(pos.size()) * window * 4), dcnt(pos.size() * 4);
  dgpp::csa2_window_slots_decode(static_cast<const int64_t*>(dpos.p), int(pos.size()), window, ring,
                                 static_cast<int32_t*>(dlist.p), static_cast<int32_t*>(dcnt.p), 0);
  sync();
  const auto list = download<int32_t>(dlist, pos.size() * window);
  const auto cnt = download<int32_t>(dcnt, pos.size());
  const std::vector<int32_t> want_cnt = {1, 6, 128, 128, 128, 0};
  require(cnt == want_cnt, "decode window counts");
  for (size_t r = 0; r < pos.size(); ++r)
    for (int k = 0; k < window; ++k) {
      const int32_t v = list[r * window + k];
      if (k < cnt[r]) require(v == int32_t((pos[r] - (cnt[r] - 1) + k) % ring), "decode slot");
      else require(v == -1, "decode padding");
    }
  // Prefill lists over a chunk at pos0 = 100 of T = 40 rows.
  const int64_t pos0 = 100;
  const int T = 40;
  DevBuf dl2(size_t(T) * window * 4), dc2(T * 4);
  dgpp::csa2_window_slots_prefill(pos0, T, window, static_cast<int32_t*>(dl2.p), static_cast<int32_t*>(dc2.p), 0);
  sync();
  const auto l2 = download<int32_t>(dl2, size_t(T) * window);
  const auto c2 = download<int32_t>(dc2, T);
  for (int i = 0; i < T; ++i) {
    const int64_t p = pos0 + i;
    const int64_t first = std::max<int64_t>(0, p - window + 1);
    require(c2[size_t(i)] == int(p - first + 1), "prefill count");
    for (int k = 0; k < c2[size_t(i)]; ++k) require(l2[size_t(i) * window + k] == int32_t(first + k - pos0 + window - 1), "prefill scratch row");
  }
  // The scratch prologue and the writeback: a ring holding distinct rows
  // for positions [0, 100), the scratch's first 127 rows are positions
  // -27..99 (zero rows below 0), the chunk's 40 rows written back land at
  // their slots.
  const size_t rb = 528;
  std::vector<uint8_t> ringv(size_t(ring) * rb, 0);
  for (int64_t p = 0; p < pos0; ++p)
    for (size_t b = 0; b < rb; ++b) ringv[size_t(p % ring) * rb + b] = uint8_t((p * 7 + b) & 0xFF);
  DevBuf dring = upload(ringv), dscr(size_t(window - 1 + T) * rb);
  DGPP_CUDA_OK(cudaMemset(dscr.p, 0xAB, size_t(window - 1 + T) * rb));
  dgpp::csa2_window_scratch_prologue(dring.p, ring, pos0, window, rb, dscr.p, 0);
  sync();
  auto scr = download<uint8_t>(dscr, size_t(window - 1 + T) * rb);
  for (int j = 0; j < window - 1; ++j) {
    const int64_t p = pos0 - (window - 1) + j;
    for (size_t b = 0; b < rb; ++b) {
      const uint8_t want = p < 0 ? 0 : uint8_t((p * 7 + b) & 0xFF);
      require(scr[size_t(j) * rb + b] == want, "scratch prologue row " + std::to_string(j));
    }
  }
  for (int i = 0; i < T; ++i)
    for (size_t b = 0; b < rb; ++b) scr[size_t(window - 1 + i) * rb + b] = uint8_t(((pos0 + i) * 7 + b) & 0xFF);
  dscr.upload(scr.data(), scr.size());
  dgpp::csa2_window_ring_writeback(dscr.p, window, pos0, T, ring, rb, dring.p, 0);
  sync();
  const auto ring2 = download<uint8_t>(dring, ringv.size());
  for (int64_t p = 0; p < pos0 + T; ++p) {
    if (p + ring < pos0 + T) continue;  // overwritten by a later position
    for (size_t b = 0; b < rb; ++b) require(ring2[size_t(p % ring) * rb + b] == uint8_t((p * 7 + b) & 0xFF), "ring after writeback");
  }
  DevBuf dtab(16 * 4);
  dgpp::csa2_ring_table(static_cast<int32_t*>(dtab.p), 16, 0);
  sync();
  const auto tab = download<int32_t>(dtab, 16);
  for (int i = 0; i < 16; ++i) require(tab[size_t(i)] == i, "ring table");
}

namespace {
// An index cache for the select tests: `n` entries through a block table
// of `epb` entries per block (physical blocks permuted), random e4m3 rows
// with power-of-two scales, a query per row.
struct SelectFixture {
  int n, epb, rows;
  std::vector<int32_t> table;  // request 0's blocks
  std::vector<uint8_t> k;      // [slots, 128]
  std::vector<float> ks;       // [slots]
  std::vector<uint8_t> q8;     // [rows, 32, 128]
  std::vector<float> wf;       // [rows, 32]
  std::vector<int64_t> pos_sel;
  std::vector<float> logits;   // [rows, n] the DSA oracle's (relu)
  DevBuf dk, dks, dq8, dwf, dt, dpos, dri;
  SelectFixture(int n_, int epb_, std::vector<int64_t> ps, uint64_t seed, bool host_logits = true)
      : n(n_), epb(epb_), rows(int(ps.size())), pos_sel(std::move(ps)) {
    const int blocks = (n + epb - 1) / epb;
    table.resize(size_t(blocks));
    for (int b = 0; b < blocks; ++b) table[size_t(b)] = (b * 7 + 3) % blocks;
    {
      std::vector<int32_t> seen(size_t(blocks), 0);
      for (int b : table) seen[size_t(b)]++;
      for (int b = 0; b < blocks; ++b) require(seen[size_t(b)] == 1, "fixture table must be a permutation");
    }
    const int slots = blocks * epb;
    k.resize(size_t(slots) * 128);
    ks.resize(size_t(slots));
    for (size_t i = 0; i < k.size(); ++i) {
      uint8_t c = uint8_t((seed * 31 + i * 2654435761u) >> 3);
      if ((c & 0x7F) == 0x7F) c = 0x40;  // no NaN codes
      k[i] = c;
    }
    for (int s = 0; s < slots; ++s) ks[size_t(s)] = std::ldexp(1.0f, -6 - int((seed + s) % 3));
    q8.resize(size_t(rows) * 32 * 128);
    for (size_t i = 0; i < q8.size(); ++i) {
      uint8_t c = uint8_t((seed * 17 + i * 40503u) >> 2);
      if ((c & 0x7F) == 0x7F) c = 0x30;
      q8[i] = c;
    }
    wf.resize(size_t(rows) * 32);
    for (size_t i = 0; i < wf.size(); ++i) wf[i] = 0.01f * float((seed + i * 13) % 100) * std::ldexp(1.0f, -8);
    // The oracle logits over the logical entries.
    logits.assign(size_t(rows) * n, 0.0f);
    std::vector<uint8_t> kl(size_t(n) * 128);
    std::vector<float> ksl(static_cast<size_t>(n));
    for (int e = 0; e < n; ++e) {
      const int slot = table[size_t(e / epb)] * epb + e % epb;
      std::copy(k.begin() + size_t(slot) * 128, k.begin() + size_t(slot + 1) * 128, kl.begin() + size_t(e) * 128);
      ksl[size_t(e)] = ks[size_t(slot)];
    }
    for (int r = 0; host_logits && r < rows; ++r)
      dgpp::dsa_ref::pool_logits<float>(&q8[size_t(r) * 32 * 128], &wf[size_t(r) * 32], kl.data(), ksl.data(), n, 32, 128,
                                        &logits[size_t(r) * n], true);
    dk = upload(k); dks = upload(ks); dq8 = upload(q8); dwf = upload(wf); dt = upload(table); dpos = upload(pos_sel);
    dri = upload(std::vector<int32_t>(size_t(rows), 0));
  }
  const int32_t* ri() const { return static_cast<const int32_t*>(dri.p); }
  const int64_t* ps() const { return static_cast<const int64_t*>(dpos.p); }
  const int32_t* tab() const { return static_cast<const int32_t*>(dt.p); }
  int bpr() const { return int(table.size()); }
};
}  // namespace

namespace {
// Production block size and selection widths (the release's 512 and the 2048
// a deployment may widen engine.index_topk to), a non-tile-aligned long pool,
// all partial-tail lengths, the all-candidates-fit transition, inactive rows,
// and the six-request/depth-four batch width. The unchanged DSA scorer supplies
// score keys; host sorting independently supplies candidate and selected IDs.
// The small fixture below also retains its fully independent host score oracle.
void long_decode_selection_replays(const int select_k) {
  constexpr int rows = 30, entries = 131073, topk_blocks = 2048, block_size = 8;
  std::vector<int64_t> positions(rows);
  const std::vector<int64_t> lengths{0,      1,      511,    512,    513,    4095,   16383,
                                     16384,  16385,  32768,  131065, 131066, 131067, 131068,
                                     131069, 131070, 131071, 131072, 131073};
  for (int r = 0; r < rows; ++r) positions[size_t(r)] = lengths[size_t(r) % lengths.size()] - 1;
  SelectFixture fx(entries, 128, positions, 0x285E1, false);
  DevBuf oracle_ws(dgpp::dsa_select_workspace_bytes(rows, entries)), oracle_counter(16),
      oracle_top(rows * 4), oracle_counts(rows * 4),
      work(dgpp::csa2_select_workspace_bytes(rows, entries)),
      candidates(size_t(rows) * topk_blocks * 4), candidate_counts(rows * 4),
      selected(size_t(rows) * select_k * 4), selected_counts(rows * 4);
  cudaStream_t stream;
  DGPP_CUDA_OK(cudaStreamCreate(&stream));
  auto launch = [&](int count) {
    dgpp::csa2_select_candidates_decode(
        fx.dq8.p, static_cast<const float*>(fx.dwf.p), fx.ri(), fx.ps(), count, fx.tab(), fx.bpr(),
        fx.dk.p, static_cast<const float*>(fx.dks.p), fx.epb, 32, block_size, topk_blocks,
        static_cast<uint64_t*>(work.p), entries, static_cast<int32_t*>(candidates.p),
        static_cast<int32_t*>(candidate_counts.p), stream);
    dgpp::csa2_select_listed_decode(
        fx.dq8.p, static_cast<const float*>(fx.dwf.p), fx.ri(), fx.ps(), count, fx.tab(), fx.bpr(),
        fx.dk.p, static_cast<const float*>(fx.dks.p), fx.epb, 32,
        static_cast<const int32_t*>(candidates.p), topk_blocks,
        static_cast<const int32_t*>(candidate_counts.p), block_size, select_k,
        static_cast<uint64_t*>(work.p), entries, static_cast<int32_t*>(selected.p),
        static_cast<int32_t*>(selected_counts.p), stream);
  };
  DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  launch(rows);
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
  DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  for (int pass = 0; pass < 3; ++pass) {
    if (pass == 1) {
      // Every score ties. Preserve the lowest original IDs, except for the
      // reference's pinned newest complete block in the candidate stage.
      std::fill(fx.wf.begin(), fx.wf.end(), 0.f);
    } else if (pass == 2) {
      // Change signs and visibility under the SAME captured graph.
      for (size_t i = 0; i < fx.wf.size(); ++i)
        fx.wf[i] = (i % 3 == 0 ? -1.f : 1.f) * float(i % 31 + 1) / 1024.f;
      for (int r = 0; r < rows; ++r) fx.pos_sel[size_t(r)] = positions[size_t((r + 7) % rows)];
    }
    fx.dwf.upload(fx.wf.data(), fx.wf.size() * 4);
    fx.dpos.upload(fx.pos_sel.data(), fx.pos_sel.size() * 8);
    dgpp::dsa_select_decode(fx.dq8.p, static_cast<const float*>(fx.dwf.p), fx.ri(), fx.ps(), rows,
                            fx.tab(), fx.bpr(), fx.dk.p, static_cast<const float*>(fx.dks.p),
                            fx.epb, 32, 128, 1, 1, 1, static_cast<int32_t*>(oracle_top.p),
                            static_cast<int32_t*>(oracle_counts.p), oracle_ws.p, entries,
                            static_cast<int32_t*>(oracle_counter.p), 0, stream, true);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    const auto keys = download<uint64_t>(oracle_ws, size_t(rows) * entries);
    std::vector<int32_t> want_candidates(size_t(rows) * topk_blocks, -1), want_nc(rows),
        want_selected(size_t(rows) * select_k, -1), want_ns(rows);
    for (int r = 0; r < rows; ++r) {
      const int64_t visible = fx.pos_sel[size_t(r)] + 1;
      float* scores = fx.logits.data() + size_t(r) * entries;
      for (int64_t i = 0; i < visible; ++i) {
        if (visible == 1) {
          scores[i] = 0.f;
          continue;
        }  // no ranking is needed
        const uint32_t sortable = ~uint32_t(keys[size_t(r) * entries + i] >> dgpp::kIdxBits);
        const uint32_t bits = sortable & 0x80000000u ? sortable ^ 0x80000000u : ~sortable;
        std::memcpy(scores + i, &bits, sizeof(bits));
      }
      const auto blocks = ref_candidate_blocks(scores, visible, block_size, topk_blocks);
      const auto ids =
          ref_topk_among(scores, ref_pool_entries(blocks, visible, block_size), visible, select_k);
      want_nc[size_t(r)] = int(blocks.size());
      want_ns[size_t(r)] = int(ids.size());
      std::copy(blocks.begin(), blocks.end(), want_candidates.begin() + size_t(r) * topk_blocks);
      std::copy(ids.begin(), ids.end(), want_selected.begin() + size_t(r) * select_k);
    }
    auto check = [&](int count) {
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      const auto got_c = download<int32_t>(candidates, size_t(count) * topk_blocks),
                 got_nc = download<int32_t>(candidate_counts, count),
                 got_s = download<int32_t>(selected, size_t(count) * select_k),
                 got_ns = download<int32_t>(selected_counts, count);
      require(std::equal(got_c.begin(), got_c.end(), want_candidates.begin()),
              "long candidate IDs/pass " + std::to_string(pass));
      require(std::equal(got_nc.begin(), got_nc.end(), want_nc.begin()), "long candidate counts");
      require(std::equal(got_s.begin(), got_s.end(), want_selected.begin()),
              "long selected IDs/pass " + std::to_string(pass));
      require(std::equal(got_ns.begin(), got_ns.end(), want_ns.begin()), "long selected counts");
    };
    for (int replay = 0; replay < 2; ++replay) {
      DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
      check(rows);
    }
    // Change the launch row count while reusing the exact same workspace stride.
    for (int count : {1, 5, 2, rows}) {
      launch(count);
      check(count);
    }
  }
  DGPP_CUDA_OK(cudaGraphExecDestroy(exec));
  DGPP_CUDA_OK(cudaGraphDestroy(graph));
  DGPP_CUDA_OK(cudaStreamDestroy(stream));
}
}  // namespace

DGPP_TEST(csa2_long_decode_selection_replays) { long_decode_selection_replays(512); }
DGPP_TEST(csa2_long_decode_selection_replays_at_2048) { long_decode_selection_replays(2048); }

DGPP_TEST(csa2_selections_at_the_widened_budget) {
  // engine.index_topk 2048 (2026-10-18): the restricted and plain decode
  // selects (the radix path, no tile) and the prefill rows select (the
  // 4096-key tile) at the select networks' bound, against the host oracles
  // over a pool wider than the budget.
  const int select_k = 2048, block_size = 8, topk_blocks = 256;
  SelectFixture fx(5000, 8, {-1, 0, 2046, 2047, 2048, 2049, 4095, 4999}, 0x2048);
  const int rows = fx.rows;
  DevBuf dsel_ws(dgpp::csa2_select_workspace_bytes(rows, fx.n));
  DevBuf dtop(size_t(rows) * select_k * 4), dcnt(rows * 4);
  const auto check_plain = [&](const char* who) {
    const auto top = download<int32_t>(dtop, size_t(rows) * select_k);
    const auto cnt = download<int32_t>(dcnt, rows);
    for (int r = 0; r < rows; ++r) {
      const int64_t visible = fx.pos_sel[size_t(r)] + 1;
      std::vector<int32_t> want(size_t(select_k), -1);
      const int wn = dgpp::dsa_ref::select_pools(&fx.logits[size_t(r) * fx.n], visible, select_k, want.data());
      require(cnt[size_t(r)] == wn, std::string(who) + " count row " + std::to_string(r));
      for (int i = 0; i < select_k; ++i)
        require(top[size_t(r) * select_k + i] == (i < wn ? want[size_t(i)] : -1), std::string(who) + " row " + std::to_string(r));
    }
  };
  // 1) the plain decode select at 2048 (cand null: every visible entry).
  dgpp::csa2_select_listed_decode(
      fx.dq8.p, static_cast<const float*>(fx.dwf.p), fx.ri(), fx.ps(), rows, fx.tab(), fx.bpr(),
      fx.dk.p, static_cast<const float*>(fx.dks.p), fx.epb, 32, nullptr, 0, nullptr, 0, select_k,
      static_cast<uint64_t*>(dsel_ws.p), fx.n, static_cast<int32_t*>(dtop.p), static_cast<int32_t*>(dcnt.p), 0);
  sync();
  check_plain("plain decode at 2048");
  // 2) the restricted decode select: 256 candidate blocks of 8 (2048 entries plus the partial
  // newest block: a pool wider than the budget at every long row) from the host oracle.
  std::vector<int32_t> cand(size_t(rows) * topk_blocks, -1), cand_counts(size_t(rows), 0);
  for (int r = 0; r < rows; ++r) {
    const int64_t visible = fx.pos_sel[size_t(r)] + 1;
    if (visible <= 0) continue;
    const auto blocks = ref_candidate_blocks(&fx.logits[size_t(r) * fx.n], visible, block_size, topk_blocks);
    cand_counts[size_t(r)] = int(blocks.size());
    std::copy(blocks.begin(), blocks.end(), cand.begin() + size_t(r) * topk_blocks);
  }
  DevBuf dcand = upload(cand), dcc = upload(cand_counts);
  const auto check_restricted = [&](const char* who) {
    const auto top = download<int32_t>(dtop, size_t(rows) * select_k);
    const auto cnt = download<int32_t>(dcnt, rows);
    for (int r = 0; r < rows; ++r) {
      const int64_t visible = fx.pos_sel[size_t(r)] + 1;
      std::vector<int32_t> blocks(cand.begin() + size_t(r) * topk_blocks,
                                  cand.begin() + size_t(r) * topk_blocks + cand_counts[size_t(r)]);
      const auto want = visible > 0 ? ref_topk_among(&fx.logits[size_t(r) * fx.n],
                                                     ref_pool_entries(blocks, visible, block_size), visible, select_k)
                                    : std::vector<int32_t>{};
      require(cnt[size_t(r)] == int(want.size()), std::string(who) + " count row " + std::to_string(r));
      for (int i = 0; i < select_k; ++i)
        require(top[size_t(r) * select_k + i] == (i < int(want.size()) ? want[size_t(i)] : -1),
                std::string(who) + " row " + std::to_string(r));
    }
  };
  dgpp::csa2_select_listed_decode(
      fx.dq8.p, static_cast<const float*>(fx.dwf.p), fx.ri(), fx.ps(), rows, fx.tab(), fx.bpr(),
      fx.dk.p, static_cast<const float*>(fx.dks.p), fx.epb, 32, static_cast<const int32_t*>(dcand.p), topk_blocks,
      static_cast<const int32_t*>(dcc.p), block_size, select_k, static_cast<uint64_t*>(dsel_ws.p), fx.n,
      static_cast<int32_t*>(dtop.p), static_cast<int32_t*>(dcnt.p), 0);
  sync();
  check_restricted("restricted decode at 2048");
  // 3) the prefill rows select over the oracle's logits rows (stride n): plain, then restricted.
  DevBuf dlog = upload(fx.logits);
  dgpp::csa2_select_rows_prefill(static_cast<const float*>(dlog.p), fx.n, fx.ps(), rows, select_k, nullptr, 0, nullptr, 0,
                                 static_cast<int32_t*>(dtop.p), static_cast<int32_t*>(dcnt.p), 0);
  sync();
  check_plain("plain prefill at 2048");
  dgpp::csa2_select_rows_prefill(static_cast<const float*>(dlog.p), fx.n, fx.ps(), rows, select_k,
                                 static_cast<const int32_t*>(dcand.p), topk_blocks, static_cast<const int32_t*>(dcc.p),
                                 block_size, static_cast<int32_t*>(dtop.p), static_cast<int32_t*>(dcnt.p), 0);
  sync();
  check_restricted("restricted prefill at 2048");
  // 4) the candidate stage at the full 2048-block pool over these rows agrees with the oracle too.
  {
    const int full_blocks = 2048;
    DevBuf dcp(size_t(rows) * full_blocks * 4), dcpc(rows * 4);
    dgpp::csa2_select_candidates_prefill(static_cast<const float*>(dlog.p), fx.n, fx.ps(), rows, block_size, full_blocks,
                                         static_cast<int32_t*>(dcp.p), static_cast<int32_t*>(dcpc.p), 0);
    sync();
    const auto cp = download<int32_t>(dcp, size_t(rows) * full_blocks);
    const auto cpc = download<int32_t>(dcpc, rows);
    for (int r = 0; r < rows; ++r) {
      const int64_t visible = fx.pos_sel[size_t(r)] + 1;
      auto c = visible > 0 ? ref_candidate_blocks(&fx.logits[size_t(r) * fx.n], visible, block_size, full_blocks)
                           : std::vector<int32_t>{};
      require(cpc[size_t(r)] == int(c.size()), "prefill candidate count at 2048 blocks");
      c.resize(size_t(full_blocks), -1);
      for (int i = 0; i < full_blocks; ++i) require(cp[size_t(r) * full_blocks + i] == c[size_t(i)], "prefill candidate list at 2048 blocks");
    }
  }
}

DGPP_TEST(csa2_selections_match_the_oracles) {
  // 300 entries of 8 per block; queries seeing 0, 1, 37, 64 (a whole number
  // of blocks: the newest complete block pinned), 299 and 300 entries.
  SelectFixture fx(300, 8, {-1, 0, 36, 63, 298, 299}, 0x5E1);
  const int rows = fx.rows;
  DevBuf dsel_ws(dgpp::csa2_select_workspace_bytes(rows, fx.n));
  // 1) the listed select with no list = the plain top-k over the visible entries.
  for (const int select_k : {16, 64}) {
    DevBuf dtop(size_t(rows) * select_k * 4), dcnt(rows * 4);
    dgpp::csa2_select_listed_decode(
        fx.dq8.p, static_cast<const float*>(fx.dwf.p), fx.ri(), fx.ps(), rows, fx.tab(), fx.bpr(),
        fx.dk.p, static_cast<const float*>(fx.dks.p), fx.epb, 32, nullptr, 0, nullptr, 0, select_k,
        static_cast<uint64_t*>(dsel_ws.p), fx.n, static_cast<int32_t*>(dtop.p),
        static_cast<int32_t*>(dcnt.p), 0);
    sync();
    const auto top = download<int32_t>(dtop, size_t(rows) * select_k);
    const auto cnt = download<int32_t>(dcnt, rows);
    for (int r = 0; r < rows; ++r) {
      const int64_t visible = fx.pos_sel[size_t(r)] + 1;
      std::vector<int32_t> want(size_t(select_k), -1);
      const int wn = dgpp::dsa_ref::select_pools(&fx.logits[size_t(r) * fx.n], visible, select_k, want.data());
      require(cnt[size_t(r)] == wn, "plain count row " + std::to_string(r));
      for (int i = 0; i < select_k; ++i) require(top[size_t(r) * select_k + i] == (i < wn ? want[size_t(i)] : -1), "plain selection row " + std::to_string(r));
    }
  }
  // 2) candidates: block max, the pinned newest block, expansion + the partial tail; reuse the same
  // workspace twice.
  const int block_size = 8, topk_blocks = 4;
  const int cand_stride = topk_blocks;
  std::vector<int32_t> cand_ref;  // rows x cand_stride block ids from the oracle
  for (int r = 0; r < rows; ++r) {
    auto c = ref_candidate_blocks(&fx.logits[size_t(r) * fx.n], fx.pos_sel[size_t(r)] + 1, block_size, topk_blocks);
    c.resize(size_t(cand_stride), -1);
    cand_ref.insert(cand_ref.end(), c.begin(), c.end());
  }
  DevBuf dcand(size_t(rows) * cand_stride * 4), dcc(rows * 4);
  for (const int replay : {0, 1}) {
    dgpp::csa2_select_candidates_decode(
        fx.dq8.p, static_cast<const float*>(fx.dwf.p), fx.ri(), fx.ps(), rows, fx.tab(), fx.bpr(),
        fx.dk.p, static_cast<const float*>(fx.dks.p), fx.epb, 32, block_size, topk_blocks,
        static_cast<uint64_t*>(dsel_ws.p), fx.n, static_cast<int32_t*>(dcand.p),
        static_cast<int32_t*>(dcc.p), 0);
    sync();
    const auto cand = download<int32_t>(dcand, size_t(rows) * cand_stride);
    const auto cc = download<int32_t>(dcc, rows);
    for (int r = 0; r < rows; ++r) {
      int wn = 0;
      while (wn < cand_stride && cand_ref[size_t(r) * cand_stride + wn] >= 0) ++wn;
      require(cc[size_t(r)] == wn,
              "candidate count row " + std::to_string(r) + " replay " + std::to_string(replay));
      for (int i = 0; i < cand_stride; ++i)
        require(cand[size_t(r) * cand_stride + i] == cand_ref[size_t(r) * cand_stride + i],
                "candidate list row " + std::to_string(r) + " replay " + std::to_string(replay));
    }
  }
  // 3) the restricted select among the candidates.
  {
    const int select_k = 16;
    DevBuf dtop(size_t(rows) * select_k * 4), dcnt(rows * 4);
    dgpp::csa2_select_listed_decode(
        fx.dq8.p, static_cast<const float*>(fx.dwf.p), fx.ri(), fx.ps(), rows, fx.tab(), fx.bpr(),
        fx.dk.p, static_cast<const float*>(fx.dks.p), fx.epb, 32,
        static_cast<const int32_t*>(dcand.p), cand_stride, static_cast<const int32_t*>(dcc.p),
        block_size, select_k, static_cast<uint64_t*>(dsel_ws.p), fx.n,
        static_cast<int32_t*>(dtop.p), static_cast<int32_t*>(dcnt.p), 0);
    sync();
    const auto top = download<int32_t>(dtop, size_t(rows) * select_k);
    const auto cnt = download<int32_t>(dcnt, rows);
    for (int r = 0; r < rows; ++r) {
      std::vector<int32_t> blocks;
      for (int i = 0; i < cand_stride && cand_ref[size_t(r) * cand_stride + i] >= 0; ++i) blocks.push_back(cand_ref[size_t(r) * cand_stride + i]);
      const auto c = ref_pool_entries(blocks, fx.pos_sel[size_t(r)] + 1, block_size);
      const auto want = ref_topk_among(&fx.logits[size_t(r) * fx.n], c, fx.pos_sel[size_t(r)] + 1, select_k);
      require(cnt[size_t(r)] == int(want.size()), "restricted count row " + std::to_string(r));
      for (int i = 0; i < select_k; ++i)
        require(top[size_t(r) * select_k + i] == (i < int(want.size()) ? want[size_t(i)] : -1), "restricted selection row " + std::to_string(r));
    }
  }
  // 4) prefill: the logits kernel from a dot buffer (the host's fp32 dots),
  // then the plain / candidate / restricted selections over the rows.
  {
    const int64_t stride = fx.n + 5;
    std::vector<float> dot(size_t(rows) * 32 * stride, 0.0f);
    for (int r = 0; r < rows; ++r)
      for (int h = 0; h < 32; ++h)
        for (int e = 0; e < fx.n; ++e) {
          const int slot = fx.table[size_t(e / fx.epb)] * fx.epb + e % fx.epb;
          float d = 0.0f;
          for (int i = 0; i < 128; ++i)
            d += dgpp::fp8_e4m3_bits_to_float(fx.q8[(size_t(r) * 32 + h) * 128 + i]) * dgpp::fp8_e4m3_bits_to_float(fx.k[size_t(slot) * 128 + i]);
          dot[(size_t(r) * 32 + h) * stride + e] = d;
        }
    std::vector<float> ksl(size_t(fx.n));
    for (int e = 0; e < fx.n; ++e) ksl[size_t(e)] = fx.ks[size_t(fx.table[size_t(e / fx.epb)] * fx.epb + e % fx.epb)];
    DevBuf ddot = upload(dot), dksl = upload(ksl), dlog(size_t(rows) * stride * 4);
    dgpp::csa2_logits_prefill(static_cast<const float*>(ddot.p), stride, static_cast<const float*>(fx.dwf.p),
                              static_cast<const float*>(dksl.p), fx.ps(), rows, fx.n, 32, static_cast<float*>(dlog.p), stride, 0);
    sync();
    const auto lg = download<float>(dlog, size_t(rows) * stride);
    // The prefill logits use the same (w * ks) * relu(dot) sum as the
    // oracle but a per-head fp32 dot in another order: near-equal.
    std::vector<float> lg_ref(size_t(rows) * fx.n);
    for (int r = 0; r < rows; ++r)
      for (int e = 0; e < fx.n; ++e) {
        const float got = lg[size_t(r) * stride + e];
        if (e >= fx.pos_sel[size_t(r)] + 1) { require(got == -INFINITY, "invisible entries are -inf"); lg_ref[size_t(r) * fx.n + e] = -INFINITY; continue; }
        const float want = fx.logits[size_t(r) * fx.n + e];
        require(std::fabs(got - want) <= 1e-5f * std::max(1.0f, std::fabs(want)), "prefill logit");
        lg_ref[size_t(r) * fx.n + e] = got;
      }
    const int select_k = 32;
    DevBuf dtop(size_t(rows) * select_k * 4), dcnt(rows * 4), dcp(size_t(rows) * cand_stride * 4), dcpc(rows * 4);
    dgpp::csa2_select_rows_prefill(static_cast<const float*>(dlog.p), stride, fx.ps(), rows, select_k, nullptr, 0, nullptr, 0,
                                   static_cast<int32_t*>(dtop.p), static_cast<int32_t*>(dcnt.p), 0);
    dgpp::csa2_select_candidates_prefill(static_cast<const float*>(dlog.p), stride, fx.ps(), rows, block_size, topk_blocks,
                                         static_cast<int32_t*>(dcp.p), static_cast<int32_t*>(dcpc.p), 0);
    sync();
    const auto top = download<int32_t>(dtop, size_t(rows) * select_k);
    const auto cnt = download<int32_t>(dcnt, rows);
    const auto cp = download<int32_t>(dcp, size_t(rows) * cand_stride);
    const auto cpc = download<int32_t>(dcpc, rows);
    for (int r = 0; r < rows; ++r) {
      const int64_t visible = fx.pos_sel[size_t(r)] + 1;
      std::vector<int32_t> want(size_t(select_k), -1);
      const int wn = dgpp::dsa_ref::select_pools(&lg_ref[size_t(r) * fx.n], visible, select_k, want.data());
      require(cnt[size_t(r)] == wn, "prefill plain count");
      for (int i = 0; i < select_k; ++i) require(top[size_t(r) * select_k + i] == (i < wn ? want[size_t(i)] : -1), "prefill plain selection");
      auto c = ref_candidate_blocks(&lg_ref[size_t(r) * fx.n], visible, block_size, topk_blocks);
      require(cpc[size_t(r)] == int(c.size()), "prefill candidate count");
      c.resize(size_t(cand_stride), -1);
      for (int i = 0; i < cand_stride; ++i) require(cp[size_t(r) * cand_stride + i] == c[size_t(i)], "prefill candidate list");
    }
    dgpp::csa2_select_rows_prefill(static_cast<const float*>(dlog.p), stride, fx.ps(), rows, select_k,
                                   static_cast<const int32_t*>(dcp.p), cand_stride, static_cast<const int32_t*>(dcpc.p), block_size,
                                   static_cast<int32_t*>(dtop.p), static_cast<int32_t*>(dcnt.p), 0);
    sync();
    const auto top2 = download<int32_t>(dtop, size_t(rows) * select_k);
    const auto cnt2 = download<int32_t>(dcnt, rows);
    for (int r = 0; r < rows; ++r) {
      std::vector<int32_t> blocks;
      for (int i = 0; i < cpc[size_t(r)]; ++i) blocks.push_back(cp[size_t(r) * cand_stride + i]);
      const auto c = ref_pool_entries(blocks, fx.pos_sel[size_t(r)] + 1, block_size);
      const auto want = ref_topk_among(&lg_ref[size_t(r) * fx.n], c, fx.pos_sel[size_t(r)] + 1, select_k);
      require(cnt2[size_t(r)] == int(want.size()), "prefill restricted count");
      for (int i = 0; i < select_k; ++i) require(top2[size_t(r) * select_k + i] == (i < int(want.size()) ? want[size_t(i)] : -1), "prefill restricted selection");
    }
  }
}

DGPP_TEST(csa2_attn_finish_merges_sources_with_the_sink_and_unrotates) {
  const int rows = 3, lh = 2, n_main = 3, n_win = 1;
  std::vector<float> mm(size_t(rows) * n_main * lh), lm(mm.size()), cm(size_t(rows) * n_main * lh * 512);
  std::vector<float> mw(size_t(rows) * n_win * lh), lw(mw.size()), cw(size_t(rows) * n_win * lh * 512);
  std::vector<float> sink = {0.3f, -1.2f};
  auto fill = [](std::vector<float>& v, uint64_t seed, float lo, float hi) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = lo + (hi - lo) * float((seed + i * 2654435761u) % 1000) / 1000.0f;
  };
  fill(mm, 1, -2.0f, 3.0f); fill(lm, 2, 0.5f, 4.0f); fill(cm, 3, -1.0f, 1.0f);
  fill(mw, 4, -1.0f, 4.0f); fill(lw, 5, 1.0f, 3.0f); fill(cw, 6, -1.0f, 1.0f);
  // Row 1's window split is empty, row 2 is a padding row.
  mw[size_t(1) * lh + 0] = -INFINITY; lw[size_t(1) * lh + 0] = 0.0f;
  const std::vector<int64_t> pos = {40, 4097, -1};
  std::vector<float> freq(32);
  dgpp::csa2_rope_inv_freq_host(64, 160000.0, 65536, 16.0, 32.0, 1.0, freq.data());
  DevBuf dmm = upload(mm), dlm = upload(lm), dcm = upload(cm), dmw = upload(mw), dlw = upload(lw), dcw = upload(cw),
      dsink = upload(sink), dpos = upload(pos), dfreq = upload(freq), dout(size_t(rows) * lh * 512 * 2);
  dgpp::csa2_attn_finish(static_cast<const float*>(dmm.p), static_cast<const float*>(dlm.p), static_cast<const float*>(dcm.p), n_main,
                         static_cast<const float*>(dmw.p), static_cast<const float*>(dlw.p), static_cast<const float*>(dcw.p), n_win,
                         static_cast<const float*>(dsink.p), rows, lh, static_cast<const int64_t*>(dpos.p),
                         static_cast<const float*>(dfreq.p), dout.p, 0);
  sync();
  const auto got = download<uint16_t>(dout, size_t(rows) * lh * 512);
  std::vector<uint16_t> want(got.size(), 0);
  for (int r = 0; r < rows; ++r) {
    if (pos[size_t(r)] < 0) continue;
    for (int h = 0; h < lh; ++h) {
      double mhat = -INFINITY;
      for (int s = 0; s < n_main; ++s) mhat = std::max(mhat, double(mm[(size_t(r) * n_main + s) * lh + h]));
      for (int s = 0; s < n_win; ++s) mhat = std::max(mhat, double(mw[(size_t(r) * n_win + s) * lh + h]));
      double den = std::exp(sink[size_t(h)] - mhat);
      for (int s = 0; s < n_main; ++s) den += std::exp(mm[(size_t(r) * n_main + s) * lh + h] - mhat) * lm[(size_t(r) * n_main + s) * lh + h];
      for (int s = 0; s < n_win; ++s) den += std::exp(mw[(size_t(r) * n_win + s) * lh + h] - mhat) * lw[(size_t(r) * n_win + s) * lh + h];
      uint16_t* o = &want[(size_t(r) * lh + h) * 512];
      for (int d = 0; d < 512; ++d) {
        double num = 0.0;
        for (int s = 0; s < n_main; ++s)
          num += std::exp(mm[(size_t(r) * n_main + s) * lh + h] - mhat) * cm[((size_t(r) * n_main + s) * lh + h) * 512 + d];
        for (int s = 0; s < n_win; ++s)
          num += std::exp(mw[(size_t(r) * n_win + s) * lh + h] - mhat) * cw[((size_t(r) * n_win + s) * lh + h) * 512 + d];
        o[d] = float_to_bf16_bits(static_cast<float>(num / den));
      }
      ref_rope(o + 512 - 64, 64, pos[size_t(r)], freq.data(), true);
    }
  }
  require_bf16("attention finish", compare_bf16(got, want, 3), 3e-3, 0.02);
  for (size_t i = size_t(2) * lh * 512; i < got.size(); ++i) require(got[i] == 0, "a padding row is zero");
}

DGPP_TEST(csa2_window_attention_end_to_end_matches_sparse_attn) {
  // Four heads over a 160-slot ring of fp8_block rows (dsa_attn_partial
  // mis-partitions 1-2 heads; DsaLayer refuses them). Two ring states: the
  // first four positions appended (a query at 3 sees a short window), and
  // positions 0..315 appended in chunks of 16 (the ring holds 156..315;
  // queries at 283, 300 and 315 see full, wrapped windows).
  const int lh = 4, window = 128, ring = 160;
  const int64_t last = 315;
  auto kv = random_bf16_bits(0x77, (last + 1) * 512, -4, 2);  // every position's roped kv
  std::vector<float> sink = {0.5f, -0.7f, 1.1f, -2.0f};
  std::vector<float> freq(32);
  dgpp::csa2_rope_inv_freq_host(64, 10000.0, 0, 16.0, 32.0, 1.0, freq.data());
  const float scale = 1.0f / std::sqrt(512.0f);
  DevBuf dkv = upload(kv), dtab = upload(std::vector<int32_t>{0}), dsink = upload(sink), dfreq = upload(freq);
  const auto run = [&](int64_t appended_upto, const std::vector<int64_t>& pos, const std::string& what) {
    DevBuf dring(size_t(ring) * 528);
    DGPP_CUDA_OK(cudaMemset(dring.p, 0, size_t(ring) * 528));
    for (int64_t p0 = 0; p0 <= appended_upto; p0 += 16) {
      const int n = int(std::min<int64_t>(16, appended_upto + 1 - p0));
      std::vector<int32_t> ri(size_t(n), 0);
      std::vector<int64_t> slots(static_cast<size_t>(n));
      for (int i = 0; i < n; ++i) slots[size_t(i)] = (p0 + i) % ring;
      DevBuf dri = upload(ri), dsl = upload(slots);
      dgpp::dsa_latent_append(static_cast<const uint16_t*>(dkv.p) + p0 * 512, static_cast<const int32_t*>(dri.p),
                              static_cast<const int64_t*>(dsl.p), n, static_cast<const int32_t*>(dtab.p), 1, ring, dring.p, 512, 0,
                              LatentFormat::kFp8Block);
      sync();
    }
    const int rows = int(pos.size());
    auto q = random_bf16_bits(0x78 + appended_upto, int64_t(rows) * lh * 512, -4, 2);
    DevBuf dq = upload(q), dpos = upload(pos), dri = upload(std::vector<int32_t>(size_t(rows), 0)),
        dlist(size_t(rows) * window * 4), dcnt(rows * 4), dm(size_t(rows) * lh * 4), dl(size_t(rows) * lh * 4),
        dc(size_t(rows) * lh * 512 * 4), dout(size_t(rows) * lh * 512 * 2);
    dgpp::csa2_window_slots_decode(static_cast<const int64_t*>(dpos.p), rows, window, ring, static_cast<int32_t*>(dlist.p),
                                   static_cast<int32_t*>(dcnt.p), 0);
    dgpp::dsa_attn_partial(dq.p, dring.p, static_cast<const int32_t*>(dri.p), static_cast<const int32_t*>(dlist.p), window,
                           static_cast<const int32_t*>(dcnt.p), rows, 1, lh, 512, ring, static_cast<const int32_t*>(dtab.p), 1, scale,
                           static_cast<float*>(dm.p), static_cast<float*>(dl.p), static_cast<float*>(dc.p), 0, LatentFormat::kFp8Block);
    dgpp::csa2_attn_finish(nullptr, nullptr, nullptr, 0, static_cast<const float*>(dm.p), static_cast<const float*>(dl.p),
                           static_cast<const float*>(dc.p), 1, static_cast<const float*>(dsink.p), rows, lh,
                           static_cast<const int64_t*>(dpos.p), static_cast<const float*>(dfreq.p), dout.p, 0);
    sync();
    long long an[6];
    require(dgpp::dsa_attn_anomalies(an, true, 0) == 0, what + ": no gather anomalies");
    const auto got = download<uint16_t>(dout, size_t(rows) * lh * 512);
    // The oracle: sparse_attn over the dequantized window rows (double
    // scores; the reference's bf16 rounding of the probabilities before
    // its PV product is a tolerance matter), the sink in the denominator,
    // bf16 out, the inverse rotation.
    std::vector<uint16_t> want(got.size());
    for (int r = 0; r < rows; ++r) {
      const int64_t p = pos[size_t(r)];
      const int64_t first = std::max<int64_t>(0, p - window + 1);
      for (int h = 0; h < lh; ++h) {
        const uint16_t* qh = &q[(size_t(r) * lh + h) * 512];
        std::vector<double> scores;
        std::vector<std::vector<uint16_t>> rowsq;
        for (int64_t t = first; t <= p; ++t) {
          std::vector<uint8_t> enc(528);
          std::vector<uint16_t> deq(512);
          float rs = 0.0f;
          dgpp::latent_quantize_row_host(LatentFormat::kFp8Block, &kv[size_t(t) * 512], 512, enc.data(), &rs);
          dgpp::latent_dequantize_row_host(LatentFormat::kFp8Block, enc.data(), 1.0f, 512, deq.data());
          double sc = 0.0;
          for (int d = 0; d < 512; ++d) sc += double(bf16_bits_to_float(qh[d])) * bf16_bits_to_float(deq[size_t(d)]);
          scores.push_back(sc * scale);
          rowsq.push_back(deq);
        }
        const double m = *std::max_element(scores.begin(), scores.end());
        double den = std::exp(sink[size_t(h)] - m);
        std::vector<double> pr(scores.size());
        for (size_t i = 0; i < scores.size(); ++i) { pr[i] = std::exp(scores[i] - m); den += pr[i]; }
        uint16_t* o = &want[(size_t(r) * lh + h) * 512];
        for (int d = 0; d < 512; ++d) {
          double num = 0.0;
          for (size_t i = 0; i < scores.size(); ++i) num += pr[i] * bf16_bits_to_float(rowsq[i][size_t(d)]);
          o[d] = float_to_bf16_bits(static_cast<float>(num / den));
        }
        ref_rope(o + 512 - 64, 64, p, freq.data(), true);
      }
    }
    require_bf16(what + ": window attention vs sparse_attn", compare_bf16(got, want, 8), 0.01, 0.01);
  };
  run(3, {3, 0, 2}, "short ring");
  run(last, {283, 300, 315}, "wrapped ring");
}

// The fused prefill scorer (csa2_score_prefill) against the host oracle's
// logits over a permuted index cache: every visible entry within the
// prefill logits kernel's tolerance (a tensor-core fp32 dot against the
// oracle's), -inf past the row's visible count, rows at -1 / 0 / a partial
// tile / the pool's end (a row never sees past the gathered keys), a
// tile-aligned and a ragged pool, and the plain,
// candidate and restricted selections over those logits equal to the
// oracle's over the kernel's own logits (the selection kernels unchanged).
DGPP_TEST(csa2_fused_prefill_scores_match_the_oracle) {
  for (const int n : {5000, 4096, 131}) {
    const std::vector<int64_t> positions{-1, 0, 1, 127, 128, 129, int64_t(n) - 2, int64_t(n) - 1, int64_t(n) / 3, 2500 % n};
    SelectFixture fx(n, 128, positions, 0x5C02Eull + uint64_t(n));
    const int rows = fx.rows;
    const int64_t stride = (n + 255) / 256 * 256;
    // The gathered (logical-order) keys and scales, as the layer gathers them.
    std::vector<uint8_t> kl(size_t(n) * 128);
    std::vector<float> ksl(static_cast<size_t>(n), 0.0f);
    for (int e = 0; e < n; ++e) {
      const int slot = fx.table[size_t(e / fx.epb)] * fx.epb + e % fx.epb;
      std::copy(fx.k.begin() + size_t(slot) * 128, fx.k.begin() + size_t(slot + 1) * 128, kl.begin() + size_t(e) * 128);
      ksl[size_t(e)] = fx.ks[size_t(slot)];
    }
    DevBuf dkl = upload(kl), dksl = upload(ksl), dlog(size_t(rows) * stride * 4);
    DGPP_CUDA_OK(cudaMemset(dlog.p, 0x7f, size_t(rows) * stride * 4));  // a NaN fill: every written cell shows
    dgpp::csa2_score_prefill(fx.dq8.p, dkl.p, static_cast<const float*>(fx.dwf.p), static_cast<const float*>(dksl.p),
                             fx.ps(), rows, n, 32, static_cast<float*>(dlog.p), stride, 0);
    sync();
    const auto lg = download<float>(dlog, size_t(rows) * stride);
    std::vector<float> lg_ref(size_t(rows) * n);
    int exact = 0, checked = 0;
    for (int r = 0; r < rows; ++r) {
      const int64_t visible = std::min<int64_t>(std::max<int64_t>(fx.pos_sel[size_t(r)] + 1, 0), n);
      for (int e = 0; e < n; ++e) {
        const float got = lg[size_t(r) * stride + e];
        if (e >= visible) {
          require(got == -INFINITY, "fused prefill: invisible entries are -inf");
          lg_ref[size_t(r) * n + e] = -INFINITY;
          continue;
        }
        const float want = fx.logits[size_t(r) * n + e];
        require(std::isfinite(got), "fused prefill: finite logit");
        require(std::fabs(got - want) <= 1e-5f * std::max(1.0f, std::fabs(want)),
                "fused prefill logit row " + std::to_string(r) + " entry " + std::to_string(e) + ": got " +
                    std::to_string(got) + " want " + std::to_string(want));
        exact += got == want;
        ++checked;
        lg_ref[size_t(r) * n + e] = got;
      }
    }
    // The oracle's fp32 dot is a sequential sum, the tensor cores' another
    // order: near-equal, as the GEMM chain's was (the tolerance above).
    (void)exact;
    (void)checked;
    // The selection kernels over the fused logits, against the oracle over the same logits.
    constexpr int select_k = 32, block_size = 8, topk_blocks = 64, cand_stride = topk_blocks;
    DevBuf dtop(size_t(rows) * select_k * 4), dcnt(rows * 4), dcp(size_t(rows) * cand_stride * 4), dcpc(rows * 4);
    dgpp::csa2_select_candidates_prefill(static_cast<const float*>(dlog.p), stride, fx.ps(), rows, block_size, topk_blocks,
                                         static_cast<int32_t*>(dcp.p), static_cast<int32_t*>(dcpc.p), 0);
    dgpp::csa2_select_rows_prefill(static_cast<const float*>(dlog.p), stride, fx.ps(), rows, select_k,
                                   static_cast<const int32_t*>(dcp.p), cand_stride, static_cast<const int32_t*>(dcpc.p),
                                   block_size, static_cast<int32_t*>(dtop.p), static_cast<int32_t*>(dcnt.p), 0);
    sync();
    const auto top = download<int32_t>(dtop, size_t(rows) * select_k);
    const auto cnt = download<int32_t>(dcnt, rows);
    const auto cp = download<int32_t>(dcp, size_t(rows) * cand_stride);
    const auto cpc = download<int32_t>(dcpc, rows);
    for (int r = 0; r < rows; ++r) {
      const int64_t visible = std::min<int64_t>(std::max<int64_t>(fx.pos_sel[size_t(r)] + 1, 0), n);
      auto c = ref_candidate_blocks(&lg_ref[size_t(r) * n], visible, block_size, topk_blocks);
      require(cpc[size_t(r)] == int(c.size()), "fused prefill: candidate count");
      const auto want = ref_topk_among(&lg_ref[size_t(r) * n], ref_pool_entries(c, visible, block_size), visible, select_k);
      require(cnt[size_t(r)] == int(want.size()), "fused prefill: restricted count row " + std::to_string(r));
      for (int i = 0; i < select_k; ++i)
        require(top[size_t(r) * select_k + i] == (i < int(want.size()) ? want[size_t(i)] : -1),
                "fused prefill: restricted selection row " + std::to_string(r));
    }
  }
}

namespace {
template <typename T>
std::vector<T> fused_dl(const DevBuf& d, size_t n) {
  std::vector<T> v(n);
  d.download(v.data(), n * sizeof(T));
  return v;
}
}  // namespace

DGPP_TEST(csa2_norms_rope_is_bitwise_the_three_kernels) {
  // The fused norms + kv rope launch against rmsnorm x2 and rope_apply on the same rows (2026-10-20).
  const int rows = 5, q_dim = 1280, kv_dim = dgpp::kCsa2Latent, rope_dim = dgpp::kCsa2Rope, half = rope_dim / 2, off = kv_dim - rope_dim;
  auto q = random_bf16_bits(0x51, int64_t(rows) * q_dim, -3, 3), kv = random_bf16_bits(0x52, int64_t(rows) * kv_dim, -3, 3);
  auto qw = random_bf16_bits(0x53, q_dim, -1, 1), kw = random_bf16_bits(0x54, kv_dim, -1, 1);
  std::vector<int64_t> pos = {0, 7, 1000, -1, 65535};
  std::vector<float> inv_h(static_cast<size_t>(half), 0.0f);
  for (int i = 0; i < half; ++i) inv_h[size_t(i)] = 1.0f / std::pow(10000.0f, float(2 * i) / float(2 * half));
  DevBuf dqa = upload(q), dqb = upload(q), dka = upload(kv), dkb = upload(kv), dqw = upload(qw), dkw = upload(kw);
  DevBuf dpos = upload(pos), dinv(inv_h.size() * 4);
  dinv.upload(inv_h.data(), inv_h.size() * 4);
  const float eps = 1e-6f;
  dgpp::csa2_rmsnorm_bf16(dqa.p, q_dim, dqw.p, dqa.p, q_dim, rows, q_dim, eps, nullptr);
  dgpp::csa2_rmsnorm_bf16(dka.p, kv_dim, dkw.p, dka.p, kv_dim, rows, kv_dim, eps, nullptr);
  dgpp::csa2_rope_apply(static_cast<uint16_t*>(dka.p) + off, kv_dim, kv_dim, 1, rope_dim, static_cast<const int64_t*>(dpos.p),
                        static_cast<const float*>(dinv.p), false, rows, nullptr);
  dgpp::csa2_norms_rope(dqb.p, q_dim, dqw.p, q_dim, dkb.p, kv_dim, dkw.p, kv_dim, eps, off, rope_dim,
                        static_cast<const int64_t*>(dpos.p), static_cast<const float*>(dinv.p), rows, nullptr);
  sync();
  const auto qa = fused_dl<uint16_t>(dqa, q.size()), qb = fused_dl<uint16_t>(dqb, q.size());
  const auto ka = fused_dl<uint16_t>(dka, kv.size()), kb = fused_dl<uint16_t>(dkb, kv.size());
  require(qa == qb, "the fused q norm is bitwise rmsnorm");
  require(ka == kb, "the fused kv norm + rope is bitwise rmsnorm + rope_apply");
}

DGPP_TEST(dsa_decode_tail_is_bitwise_the_five_kernels) {
  // The fused decode tail (q rope, ring slot, window slots, entry positions, the ring append) against the five
  // launches it replaces (2026-10-20), fp8-block ring rows of 512 + 16 bytes.
  const int rows = 5, lh = 16, rope_dim = dgpp::kCsa2Rope, half = rope_dim / 2, latent = dgpp::kCsa2Latent, off = latent - rope_dim;
  const int ring_slots = 64, window = 16, ratio = 2, n_req = 2;
  auto q = random_bf16_bits(0x61, int64_t(rows) * lh * latent, -3, 3), kv = random_bf16_bits(0x62, int64_t(rows) * latent, -3, 3);
  std::vector<int64_t> pos = {5, 6, 0, 1, -1};
  std::vector<int32_t> req = {0, 0, 1, 1, 0}, table = {0, 1};
  std::vector<float> inv_h(static_cast<size_t>(half), 0.0f);
  for (int i = 0; i < half; ++i) inv_h[size_t(i)] = 1.0f / std::pow(10000.0f, float(2 * i) / float(2 * half));
  const size_t row_bytes = 528, ring_bytes = size_t(n_req) * ring_slots * row_bytes;
  DevBuf dqa = upload(q), dqb = upload(q), dkv = upload(kv), dpos = upload(pos), dreq = upload(req), dtab = upload(table), dinv(inv_h.size() * 4);
  dinv.upload(inv_h.data(), inv_h.size() * 4);
  DevBuf dsa_(size_t(rows) * 8), dsb(size_t(rows) * 8), dla(size_t(rows) * window * 4), dlb(size_t(rows) * window * 4);
  DevBuf dca(size_t(rows) * 4), dcb(size_t(rows) * 4), dea(size_t(rows) * 8), deb(size_t(rows) * 8), dra(ring_bytes), drb(ring_bytes);
  DGPP_CUDA_OK(cudaMemset(dra.p, 0x5a, ring_bytes));
  DGPP_CUDA_OK(cudaMemset(drb.p, 0x5a, ring_bytes));
  // the five launches
  dgpp::csa2_rope_apply(static_cast<uint16_t*>(dqa.p) + off, int64_t(lh) * latent, latent, lh, rope_dim,
                        static_cast<const int64_t*>(dpos.p), static_cast<const float*>(dinv.p), false, rows, nullptr);
  dgpp::csa2_ring_slot_positions(static_cast<const int64_t*>(dpos.p), static_cast<int64_t*>(dsa_.p), rows, ring_slots, nullptr);
  dgpp::dsa_latent_append(dkv.p, static_cast<const int32_t*>(dreq.p), static_cast<const int64_t*>(dsa_.p), rows,
                          static_cast<const int32_t*>(dtab.p), 1, ring_slots, dra.p, latent, nullptr, LatentFormat::kFp8Block);
  dgpp::csa2_window_slots_decode(static_cast<const int64_t*>(dpos.p), rows, window, ring_slots, static_cast<int32_t*>(dla.p),
                                 static_cast<int32_t*>(dca.p), nullptr);
  dgpp::csa2_entry_positions(static_cast<const int64_t*>(dpos.p), static_cast<int64_t*>(dea.p), rows, ratio, nullptr);
  // the fused tail
  dgpp::dsa_decode_tail_fp8block(static_cast<uint16_t*>(dqb.p) + off, int64_t(lh) * latent, latent, lh, rope_dim,
                                 static_cast<const int64_t*>(dpos.p), static_cast<const float*>(dinv.p),
                                 static_cast<int64_t*>(dsb.p), ring_slots, window, static_cast<int32_t*>(dlb.p),
                                 static_cast<int32_t*>(dcb.p), static_cast<int64_t*>(deb.p), ratio, dkv.p,
                                 static_cast<const int32_t*>(dreq.p), static_cast<const int32_t*>(dtab.p), drb.p, latent, rows,
                                 nullptr);
  sync();
  require(fused_dl<uint16_t>(dqa, q.size()) == fused_dl<uint16_t>(dqb, q.size()), "the tail's q rope is bitwise rope_apply");
  require(fused_dl<int64_t>(dsa_, size_t(rows)) == fused_dl<int64_t>(dsb, size_t(rows)), "the tail's ring slots match");
  require(fused_dl<int32_t>(dla, size_t(rows) * window) == fused_dl<int32_t>(dlb, size_t(rows) * window), "the tail's window lists match");
  require(fused_dl<int32_t>(dca, size_t(rows)) == fused_dl<int32_t>(dcb, size_t(rows)), "the tail's window counts match");
  require(fused_dl<int64_t>(dea, size_t(rows)) == fused_dl<int64_t>(deb, size_t(rows)), "the tail's entry positions match");
  require(fused_dl<uint8_t>(dra, ring_bytes) == fused_dl<uint8_t>(drb, ring_bytes), "the tail's ring append is bitwise dsa_latent_append");
}

int main() { return dgpp::test::run_all(); }
