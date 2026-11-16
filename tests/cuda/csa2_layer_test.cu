// The CSA2 layer (models/dsv41/csa2_layer.hpp) against a host oracle of
// the reference forward (inference/model.py Attention / Compressor /
// Indexer with the engine's numerics at the pinned points), over a
// six-layer schedule on one shared pool: a window-only layer; a ratio-2
// kv + index source and a layer reusing its cache and selection; a
// ratio-1 kv + index + candidate source, a Reindex layer restricted to
// its pool, and a layer reusing that selection. Two requests prefill in
// block-aligned chunks (the compressor's odd tail across a chunk
// boundary, the ring wrapping), then decode in verify batches with
// spans, a rejected draft rolled back (the tail snapshot restored, the
// positional rings and caches overwritten by the accepted rows), at TP 1
// and as rank 0 of TP 4 (4 local heads, one output group). Gates: every
// layer output within the bf16 budget of the oracle's, the selections'
// sets equal or their flips near ties, zero index-key violations.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
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
#include "kernels/gemm.hpp"
#include "kernels/latent_format.hpp"
#include "models/dsv41/csa2_layer.hpp"
#include "models/dsv41/csa2_state.hpp"

namespace {
using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;
using dgpp::LatentFormat;
using dgpp::kda_test::compare_bf16;
using dgpp::kda_test::DevBuf;
using dgpp::kda_test::require_bf16;

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
std::vector<T> download(const void* p, size_t n) {
  std::vector<T> v(n);
  DGPP_CUDA_OK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}
float bf(float v) { return bf16_bits_to_float(float_to_bf16_bits(v)); }

// ---- the fixture's weights --------------------------------------------------------------------
// An fp8 matrix [n, k] on the 32 x 32 grid: values ~ amp * N(0, 1)
// quantized as the release does (e8m0 scale per block = 2^ceil(log2(absmax
// / 448)), e4m3 codes); the host keeps the exact dequantized floats.
struct Fp8Host {
  int64_t n = 0, k = 0;
  std::vector<uint8_t> payload;
  std::vector<float> scales;  // [n/32][k/32]
  std::vector<float> deq;     // [n][k]
  DevBuf dp, ds;
  dgpp::GlmQuantMatrix view() const {
    dgpp::GlmQuantMatrix q;
    q.rows = n;
    q.cols = k;
    q.scale_block_rows = 32;
    q.scale_block_cols = 32;
    q.payload = static_cast<const uint8_t*>(dp.p);
    q.scales = static_cast<const float*>(ds.p);
    return q;
  }
};
uint32_t mix(uint64_t s, uint64_t i) {
  uint64_t x = s * 0x9E3779B97F4A7C15ull + i * 0xBF58476D1CE4E5B9ull;
  x ^= x >> 31;
  x *= 0x94D049BB133111EBull;
  x ^= x >> 29;
  return static_cast<uint32_t>(x);
}
float gauss(uint64_t s, uint64_t i) {
  float acc = 0.0f;
  for (int j = 0; j < 4; ++j) acc += (mix(s, i * 4 + j) >> 8) / float(1u << 24) - 0.5f;
  return acc * 1.7f;
}
Fp8Host make_fp8(int64_t n, int64_t k, uint64_t seed, float amp) {
  require(n % 32 == 0 && k % 32 == 0, "fp8 fixture geometry");
  Fp8Host m;
  m.n = n;
  m.k = k;
  m.payload.resize(size_t(n) * k);
  m.scales.resize(size_t(n / 32) * (k / 32));
  m.deq.resize(size_t(n) * k);
  std::vector<float> raw(size_t(n) * k);
  for (size_t i = 0; i < raw.size(); ++i) raw[i] = amp * gauss(seed, i);
  for (int64_t br = 0; br < n / 32; ++br)
    for (int64_t bc = 0; bc < k / 32; ++bc) {
      float amax = 0.0f;
      for (int r = 0; r < 32; ++r)
        for (int c = 0; c < 32; ++c) amax = std::max(amax, std::fabs(raw[size_t(br * 32 + r) * k + bc * 32 + c]));
      const float s = dgpp::e8m0_byte_to_float(dgpp::e8m0_ceil_log2_byte(std::max(amax, 1e-4f) * (1.0f / 448.0f)));
      m.scales[size_t(br) * (k / 32) + bc] = s;
      for (int r = 0; r < 32; ++r)
        for (int c = 0; c < 32; ++c) {
          const size_t at = size_t(br * 32 + r) * k + bc * 32 + c;
          const uint8_t code = dgpp::float_to_fp8_e4m3_bits(raw[at] / s);
          m.payload[at] = code;
          m.deq[at] = dgpp::fp8_e4m3_bits_to_float(code) * s;
        }
    }
  m.dp = upload(m.payload);
  m.ds = upload(m.scales);
  return m;
}
struct Bf16Host {
  int64_t n = 0, k = 0;
  std::vector<uint16_t> bits;
  std::vector<float> val;
  DevBuf d;
};
Bf16Host make_bf16(int64_t n, int64_t k, uint64_t seed, float amp, float base = 0.0f) {
  Bf16Host m;
  m.n = n;
  m.k = k;
  m.bits.resize(size_t(n) * k);
  m.val.resize(size_t(n) * k);
  for (size_t i = 0; i < m.bits.size(); ++i) {
    m.bits[i] = float_to_bf16_bits(base + amp * gauss(seed, i));
    m.val[i] = bf16_bits_to_float(m.bits[i]);
  }
  m.d = upload(m.bits);
  return m;
}

// ---- the oracle --------------------------------------------------------------------------------------
struct RefLayer {
  int ratio = 0;
  bool kv_source = false, index_source = false, candidate_source = false, uses_candidates = false;
  int cache_ord = -1, tail_ord = -1;
  const Fp8Host *wq_a, *wkv, *wq_b, *wo_a, *wo_b, *idx_wq_b;
  const Bf16Host *q_norm, *kv_norm, *idx_wp, *idx_wk, *idx_k_norm, *comp_wkv, *comp_wgate, *comp_norm;
  std::vector<float> sink;
  std::vector<float> inv_freq;
};
// bf16(x @ W^T) with W the exact float matrix (double accumulation).
std::vector<float> matmul_bf16(const std::vector<float>& x, int64_t rows, const std::vector<float>& w, int64_t n, int64_t k) {
  std::vector<float> out(size_t(rows) * n);
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int64_t i = 0; i < k; ++i) acc += double(x[size_t(r) * k + i]) * w[size_t(j) * k + i];
      out[size_t(r) * n + j] = bf(static_cast<float>(acc));
    }
  return out;
}
std::vector<float> matmul_f32(const std::vector<float>& x, int64_t rows, const std::vector<float>& w, int64_t n, int64_t k) {
  std::vector<float> out(size_t(rows) * n);
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int64_t i = 0; i < k; ++i) acc += double(x[size_t(r) * k + i]) * w[size_t(j) * k + i];
      out[size_t(r) * n + j] = static_cast<float>(acc);
    }
  return out;
}
void rmsnorm_rows(std::vector<float>& x, int64_t rows, int64_t dim, const std::vector<float>& w, float eps) {
  for (int64_t r = 0; r < rows; ++r) {
    double ss = 0.0;
    for (int64_t i = 0; i < dim; ++i) ss += double(x[size_t(r) * dim + i]) * x[size_t(r) * dim + i];
    const double rs = 1.0 / std::sqrt(ss / dim + double(eps));
    for (int64_t i = 0; i < dim; ++i) x[size_t(r) * dim + i] = bf(static_cast<float>(w[size_t(i)] * (x[size_t(r) * dim + i] * rs)));
  }
}
void rope_tail(float* seg, int rope, int64_t pos, const std::vector<float>& inv_freq, bool inverse) {
  for (int i = 0; i < rope / 2; ++i) {
    const float ang = static_cast<float>(pos) * inv_freq[size_t(i)];
    const double c = std::cos(double(ang));
    double s = std::sin(double(ang));
    if (inverse) s = -s;
    const double x0 = seg[2 * i], x1 = seg[2 * i + 1];
    seg[2 * i] = bf(static_cast<float>(x0 * c - x1 * s));
    seg[2 * i + 1] = bf(static_cast<float>(x1 * c + x0 * s));
  }
}
std::vector<float> quant_dequant(LatentFormat f, const std::vector<float>& row) {
  std::vector<uint16_t> bits(row.size());
  for (size_t i = 0; i < row.size(); ++i) bits[i] = float_to_bf16_bits(row[i]);
  std::vector<uint8_t> enc(dgpp::latent_row_bytes(f, int(row.size())));
  std::vector<uint16_t> deq(row.size());
  float rs = 0.0f;
  dgpp::latent_quantize_row_host(f, bits.data(), int(row.size()), enc.data(), &rs);
  dgpp::latent_dequantize_row_host(f, enc.data(), 1.0f, int(row.size()), deq.data());
  std::vector<float> out(row.size());
  for (size_t i = 0; i < row.size(); ++i) out[i] = bf16_bits_to_float(deq[i]);
  return out;
}
std::vector<float> fp4_e8m0_dequant(const std::vector<float>& x /*[128]*/) {
  std::vector<float> out(128);
  for (int b = 0; b < 4; ++b) {
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[size_t(b * 32 + j)]));
    amax = std::max(amax, 6.0f * 1.1754943508222875e-38f);
    const float s = dgpp::e8m0_byte_to_float(dgpp::e8m0_ceil_log2_byte(amax * (1.0f / 6.0f)));
    for (int j = 0; j < 32; ++j)
      out[size_t(b * 32 + j)] = dgpp::fp4_e2m1_bits_to_float(dgpp::float_to_fp4_e2m1_bits(x[size_t(b * 32 + j)] / s)) * s;
  }
  return out;
}

struct RefRequest {
  std::map<int64_t, std::vector<float>> window;              // per layer: position -> 512 (layer * 1e6 + pos keyed below)
  std::vector<std::vector<std::vector<float>>> main;         // [ord][entry] 512
  std::vector<std::vector<std::vector<float>>> keys;         // [ord][entry] 128
  std::vector<std::vector<float>> tail_kv, tail_score;       // [tail_ord] 512 (the pending even token)
  std::vector<bool> tail_valid;
  std::vector<std::vector<int32_t>> cand;                    // per row of the current call: block ids
  std::vector<std::vector<int32_t>> sel;                     // per row: selected entries
  std::vector<std::vector<float>> logits;                    // per row: the logits over visible entries
};

struct RefModel {
  dgpp::Csa2Config cfg;
  std::vector<RefLayer> layers;
  std::vector<int> cache_ratio;
  int tails = 0;
  std::vector<RefRequest> reqs;
  int lh, lg, hpg;
  RefModel(const dgpp::Csa2Config& c, int max_requests, std::vector<int> ratios, int n_tails)
      : cfg(c), cache_ratio(std::move(ratios)), tails(n_tails), reqs(size_t(max_requests)) {
    lh = c.local_heads();
    lg = c.local_groups();
    hpg = c.heads_per_group();
    for (auto& r : reqs) reset(r);
  }
  void reset(RefRequest& r) {
    r.window.clear();
    r.main.assign(cache_ratio.size(), {});
    r.keys.assign(cache_ratio.size(), {});
    r.tail_kv.assign(size_t(tails), std::vector<float>(512, 0.0f));
    r.tail_score.assign(size_t(tails), std::vector<float>(512, 0.0f));
    r.tail_valid.assign(size_t(tails), false);
  }
  static int64_t wkey(int layer, int64_t pos) { return int64_t(layer) * 4000000 + pos; }

  // One layer over `rows` consecutive positions [p0, p0 + rows) of request
  // `req` (a prefill chunk or a decode span): returns the partial output
  // rows [rows, hidden] and records the row's selection / candidates.
  std::vector<float> layer_rows(int layer, const std::vector<float>& hidden, int64_t p0, int rows, int req) {
    const RefLayer& L = layers[size_t(layer)];
    RefRequest& R = reqs[size_t(req)];
    const int H = cfg.hidden, ql = cfg.q_lora;
    // q path
    std::vector<float> qr = matmul_bf16(hidden, rows, L.wq_a->deq, ql, H);
    rmsnorm_rows(qr, rows, ql, L.q_norm->val, cfg.eps);
    std::vector<float> q = matmul_bf16(qr, rows, L.wq_b->deq, int64_t(lh) * 512, ql);
    std::vector<float> kv = matmul_bf16(hidden, rows, L.wkv->deq, 512, H);
    rmsnorm_rows(kv, rows, 512, L.kv_norm->val, cfg.eps);
    for (int i = 0; i < rows; ++i) {
      const int64_t p = p0 + i;
      for (int h = 0; h < lh; ++h) rope_tail(&q[(size_t(i) * lh + h) * 512 + 448], 64, p, L.inv_freq, false);
      rope_tail(&kv[size_t(i) * 512 + 448], 64, p, L.inv_freq, false);
      std::vector<float> row(kv.begin() + i * 512, kv.begin() + (i + 1) * 512);
      R.window[wkey(layer, p)] = quant_dequant(LatentFormat::kFp8Block, row);
    }
    // the compressor (kv source) -> entries
    if (L.kv_source) {
      const int ord = L.cache_ord;
      std::vector<float> lat;  // [n, 512] normed, unrotated
      std::vector<int64_t> ents;
      if (L.ratio == 1) {
        lat = matmul_bf16(hidden, rows, L.comp_wkv->val, 512, H);
        rmsnorm_rows(lat, rows, 512, L.comp_norm->val, cfg.eps);
        for (int i = 0; i < rows; ++i) ents.push_back(p0 + i);
      } else {
        const std::vector<float> ckv = matmul_f32(hidden, rows, L.comp_wkv->val, 512, H);
        const std::vector<float> csc = matmul_f32(hidden, rows, L.comp_wgate->val, 512, H);
        for (int i = 0; i < rows; ++i) {
          const int64_t p = p0 + i;
          if ((p & 1) == 0) {
            std::copy(ckv.begin() + i * 512, ckv.begin() + (i + 1) * 512, R.tail_kv[size_t(L.tail_ord)].begin());
            std::copy(csc.begin() + i * 512, csc.begin() + (i + 1) * 512, R.tail_score[size_t(L.tail_ord)].begin());
            R.tail_valid[size_t(L.tail_ord)] = true;
            continue;
          }
          require(R.tail_valid[size_t(L.tail_ord)], "oracle: an odd token without its even partner");
          std::vector<float> pooled(512);
          for (int c = 0; c < 512; ++c) {
            const float s0 = R.tail_score[size_t(L.tail_ord)][size_t(c)], s1 = csc[size_t(i) * 512 + c];
            const float m = std::max(s0, s1);
            const float e0 = std::exp(s0 - m), e1 = std::exp(s1 - m);
            const float w0 = e0 / (e0 + e1), w1 = e1 / (e0 + e1);
            pooled[size_t(c)] = bf(R.tail_kv[size_t(L.tail_ord)][size_t(c)] * w0 + ckv[size_t(i) * 512 + c] * w1);
          }
          rmsnorm_rows(pooled, 1, 512, L.comp_norm->val, cfg.eps);
          lat.insert(lat.end(), pooled.begin(), pooled.end());
          ents.push_back(p / 2);
        }
      }
      const int n = int(ents.size());
      if (n > 0) {
        std::vector<float> ik = matmul_bf16(lat, n, L.idx_wk->val, 128, 512);
        rmsnorm_rows(ik, n, 128, L.idx_k_norm->val, cfg.eps);
        for (int e = 0; e < n; ++e) {
          const int64_t j = ents[size_t(e)];
          rope_tail(&ik[size_t(e) * 128 + 64], 64, j * L.ratio, L.inv_freq, false);
          std::vector<float> krow(ik.begin() + e * 128, ik.begin() + (e + 1) * 128);
          std::vector<float> lrow(lat.begin() + e * 512, lat.begin() + (e + 1) * 512);
          rope_tail(&lrow[448], 64, j * L.ratio, L.inv_freq, false);
          auto& K = R.keys[size_t(ord)];
          auto& M = R.main[size_t(ord)];
          if (K.size() <= size_t(j)) { K.resize(size_t(j) + 1); M.resize(size_t(j) + 1); }
          K[size_t(j)] = fp4_e8m0_dequant(krow);
          M[size_t(j)] = quant_dequant(LatentFormat::kFp4Block, lrow);
        }
      }
    }
    // the selection
    std::vector<std::vector<int32_t>> sel(static_cast<size_t>(rows));
    if (L.ratio > 0) {
      const int ord = L.cache_ord;
      if (L.index_source) {
        std::vector<float> iq = matmul_bf16(qr, rows, L.idx_wq_b->deq, 32 * 128, ql);
        std::vector<float> w = matmul_bf16(hidden, rows, L.idx_wp->val, 32, H);
        if (L.candidate_source) R.cand.assign(size_t(rows), {});
        R.logits.assign(size_t(rows), {});
        for (int i = 0; i < rows; ++i) {
          const int64_t p = p0 + i;
          const int64_t visible = (p + 1) / L.ratio;
          std::vector<std::vector<float>> qh(32);
          for (int h = 0; h < 32; ++h) {
            std::vector<float> seg(iq.begin() + (i * 32 + h) * 128, iq.begin() + (i * 32 + h + 1) * 128);
            rope_tail(&seg[64], 64, p, L.inv_freq, false);
            qh[size_t(h)] = fp4_e8m0_dequant(seg);
          }
          std::vector<float> logits(size_t(visible), 0.0f);
          for (int64_t j = 0; j < visible; ++j) {
            require(size_t(j) < R.keys[size_t(ord)].size(), "oracle: a visible entry is missing");
            float total = 0.0f;
            for (int h = 0; h < 32; ++h) {
              float d = 0.0f;
              for (int t = 0; t < 128; ++t) d += qh[size_t(h)][size_t(t)] * R.keys[size_t(ord)][size_t(j)][size_t(t)];
              total += (w[size_t(i) * 32 + h] * 0.015625f) * std::max(d, 0.0f);
            }
            logits[size_t(j)] = total;
          }
          R.logits[size_t(i)] = logits;
          std::vector<int32_t> pool;
          if (L.candidate_source || L.uses_candidates) {
            if (L.candidate_source) {
              const int bs = cfg.candidate_block;
              const int64_t nb = visible / bs;
              std::vector<std::pair<float, int64_t>> blocks;
              for (int64_t b = 0; b < nb; ++b) {
                float best = -INFINITY;
                for (int k = 0; k < bs; ++k) best = std::max(best, logits[size_t(b * bs + k)]);
                if (visible % bs == 0 && b == nb - 1) best = INFINITY;
                blocks.emplace_back(best, b);
              }
              std::sort(blocks.begin(), blocks.end(), [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
              std::vector<int32_t> ids;
              for (size_t k = 0; k < blocks.size() && k < size_t(cfg.candidate_blocks); ++k) ids.push_back(int32_t(blocks[k].second));
              std::sort(ids.begin(), ids.end());
              R.cand[size_t(i)] = ids;
            }
            const int bs = cfg.candidate_block;
            for (const int32_t b : R.cand[size_t(i)])
              for (int k = 0; k < bs; ++k) pool.push_back(b * bs + k);
            for (int64_t e = (visible / bs) * bs; e < visible; ++e) pool.push_back(int32_t(e));
          } else {
            for (int64_t e = 0; e < visible; ++e) pool.push_back(int32_t(e));
          }
          std::vector<std::pair<float, int32_t>> v;
          for (const int32_t e : pool) v.emplace_back(logits[size_t(e)], e);
          std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
          for (size_t k = 0; k < v.size() && k < size_t(cfg.index_topk); ++k) sel[size_t(i)].push_back(v[k].second);
          std::sort(sel[size_t(i)].begin(), sel[size_t(i)].end());
        }
        R.sel = sel;
      } else {
        sel = R.sel;  // the last index source's
      }
    }
    // the attention: window rows + selected main rows, the sink, bf16, the inverse rotation
    std::vector<float> o(size_t(rows) * lh * 512);
    for (int i = 0; i < rows; ++i) {
      const int64_t p = p0 + i;
      std::vector<const std::vector<float>*> src;
      for (int64_t t = std::max<int64_t>(0, p - cfg.window + 1); t <= p; ++t) src.push_back(&R.window.at(wkey(layer, t)));
      if (L.ratio > 0)
        for (const int32_t e : sel[size_t(i)]) src.push_back(&R.main[size_t(L.cache_ord)][size_t(e)]);
      for (int h = 0; h < lh; ++h) {
        const float* qh = &q[(size_t(i) * lh + h) * 512];
        std::vector<double> sc(src.size());
        double m = -INFINITY;
        for (size_t k = 0; k < src.size(); ++k) {
          double d = 0.0;
          for (int t = 0; t < 512; ++t) d += double(qh[t]) * (*src[k])[size_t(t)];
          sc[k] = d * (1.0 / std::sqrt(512.0));
          m = std::max(m, sc[k]);
        }
        double den = std::exp(double(L.sink[size_t(h)]) - m);
        std::vector<double> pr(src.size());
        for (size_t k = 0; k < src.size(); ++k) { pr[k] = std::exp(sc[k] - m); den += pr[k]; }
        float* oh = &o[(size_t(i) * lh + h) * 512];
        for (int t = 0; t < 512; ++t) {
          double num = 0.0;
          for (size_t k = 0; k < src.size(); ++k) num += pr[k] * (*src[k])[size_t(t)];
          oh[t] = bf(static_cast<float>(num / den));
        }
        rope_tail(oh + 448, 64, p, L.inv_freq, true);
      }
    }
    // wo_a per group, wo_b
    std::vector<float> oa(size_t(rows) * lg * cfg.o_lora);
    const int K = hpg * 512;
    for (int g = 0; g < lg; ++g) {
      std::vector<float> og(size_t(rows) * K);
      for (int i = 0; i < rows; ++i) std::copy(o.begin() + (size_t(i) * lh * 512 + size_t(g) * K), o.begin() + (size_t(i) * lh * 512 + size_t(g + 1) * K), og.begin() + size_t(i) * K);
      std::vector<float> wg(L.wo_a->deq.begin() + size_t(g) * cfg.o_lora * K, L.wo_a->deq.begin() + size_t(g + 1) * cfg.o_lora * K);
      const std::vector<float> r = matmul_bf16(og, rows, wg, cfg.o_lora, K);
      for (int i = 0; i < rows; ++i) std::copy(r.begin() + size_t(i) * cfg.o_lora, r.begin() + size_t(i + 1) * cfg.o_lora, oa.begin() + size_t(i) * lg * cfg.o_lora + size_t(g) * cfg.o_lora);
    }
    return matmul_bf16(oa, rows, L.wo_b->deq, H, int64_t(lg) * cfg.o_lora);
  }
};

// ---- the device side -----------------------------------------------------------------------------------
struct DevLayer {
  Fp8Host wq_a, wkv, wq_b, wo_a, wo_b, idx_wq_b;
  Bf16Host q_norm, kv_norm, idx_wp, idx_wk, idx_k_norm, comp_wkv, comp_wgate, comp_norm;
  std::vector<float> sink, inv_freq;
  DevBuf dsink, dfreq;
  dgpp::Csa2LayerWeights w;
  RefLayer ref;
};

struct Scenario {
  dgpp::Csa2Config cfg;
  std::vector<int> ratios = {2, 1};
  std::vector<DevLayer> layers;
  std::unique_ptr<RefModel> ref;
  dgpp::Csa2StatePool pool;
  dgpp::CublasLtGemm gemm;
  DevBuf scratch, gemm_ws;
  std::unique_ptr<dgpp::Csa2Layer> layer;
  int max_tokens = 64, max_requests = 2;
  int64_t max_cache = 512;
  std::string tag;
  int flips = 0;  // certified near-tie selection flips
  std::vector<std::vector<uint16_t>> record;  // every prefill's device rows, a layer a call, in order
  std::vector<std::vector<float>> hidden;  // per request, per position: [hidden] rows (all positions ever generated)

  explicit Scenario(int tp, const std::string& tag_, int64_t cache_tokens = 512,
                    size_t dot_budget = 64ull << 20) : max_cache(cache_tokens), tag(tag_) {
    cfg.hidden = 256; cfg.q_lora = 64; cfg.o_lora = 32; cfg.num_heads = 16; cfg.o_groups = 4;
    cfg.index_topk = 16; cfg.candidate_block = 8; cfg.candidate_blocks = 4; cfg.window = 32; cfg.ring_slots = 48;
    cfg.block_tokens = 32; cfg.tp = tp;
    const int lh = cfg.local_heads(), lg = cfg.local_groups(), hpg = cfg.heads_per_group();
    // Six layers: window; ratio-2 kv+index source (ord 0, tail 0); ratio-2
    // reuse; ratio-1 kv+index+candidate source (ord 1); ratio-1 reindex
    // (uses candidates); ratio-1 reuse.
    struct Role { int ratio; bool kv, idx, cand_src, cand_use; int ord, tail; };
    const std::vector<Role> roles = {{0, false, false, false, false, -1, -1}, {2, true, true, false, false, 0, 0},
                                     {2, false, false, false, false, 0, -1}, {1, true, true, true, false, 1, -1},
                                     {1, false, true, false, true, 1, -1}, {1, false, false, false, false, 1, -1}};
    layers.resize(roles.size());
    for (size_t l = 0; l < roles.size(); ++l) {
      DevLayer& d = layers[l];
      const Role& r = roles[l];
      const uint64_t s = 0x1000 * (l + 1) + (tp == 4 ? 7 : 0);
      d.wq_a = make_fp8(cfg.q_lora, cfg.hidden, s + 1, 0.08f);
      d.wkv = make_fp8(512, cfg.hidden, s + 2, 0.08f);
      d.wq_b = make_fp8(int64_t(lh) * 512, cfg.q_lora, s + 3, 0.15f);
      d.wo_a = make_fp8(int64_t(lg) * cfg.o_lora, int64_t(hpg) * 512, s + 4, 0.03f);
      d.wo_b = make_fp8(cfg.hidden, int64_t(lg) * cfg.o_lora, s + 5, 0.15f);
      d.q_norm = make_bf16(1, cfg.q_lora, s + 6, 0.1f, 1.0f);
      d.kv_norm = make_bf16(1, 512, s + 7, 0.1f, 1.0f);
      d.sink.resize(size_t(lh));
      for (int h = 0; h < lh; ++h) d.sink[size_t(h)] = 0.5f * gauss(s + 8, h);
      d.dsink = upload(d.sink);
      d.inv_freq.resize(32);
      if (r.ratio == 0) dgpp::csa2_rope_inv_freq_host(64, 10000.0, 0, 16.0, 32.0, 1.0, d.inv_freq.data());
      else dgpp::csa2_rope_inv_freq_host(64, 160000.0, 64, 16.0, 32.0, 1.0, d.inv_freq.data());
      d.dfreq = upload(d.inv_freq);
      if (r.idx) {
        d.idx_wq_b = make_fp8(32 * 128, cfg.q_lora, s + 9, 0.2f);
        d.idx_wp = make_bf16(32, cfg.hidden, s + 10, 0.3f);
      }
      if (r.kv) {
        d.idx_wk = make_bf16(128, 512, s + 11, 0.1f);
        d.idx_k_norm = make_bf16(1, 128, s + 12, 0.1f, 1.0f);
        d.comp_wkv = make_bf16(512, cfg.hidden, s + 13, 0.08f);
        d.comp_norm = make_bf16(1, 512, s + 14, 0.1f, 1.0f);
        if (r.ratio == 2) d.comp_wgate = make_bf16(512, cfg.hidden, s + 15, 0.3f);
      }
      dgpp::Csa2LayerWeights& w = d.w;
      w.wq_a = d.wq_a.view(); w.wkv = d.wkv.view(); w.wq_b = d.wq_b.view(); w.wo_a = d.wo_a.view(); w.wo_b = d.wo_b.view();
      w.q_norm = static_cast<const uint16_t*>(d.q_norm.d.p); w.kv_norm = static_cast<const uint16_t*>(d.kv_norm.d.p);
      w.attn_sink = static_cast<const float*>(d.dsink.p); w.inv_freq = static_cast<const float*>(d.dfreq.p);
      if (r.idx) { w.idx_wq_b = d.idx_wq_b.view(); w.idx_wp = static_cast<const uint16_t*>(d.idx_wp.d.p); }
      if (r.kv) {
        w.idx_wk = static_cast<const uint16_t*>(d.idx_wk.d.p); w.idx_k_norm = static_cast<const uint16_t*>(d.idx_k_norm.d.p);
        w.comp_wkv = static_cast<const uint16_t*>(d.comp_wkv.d.p); w.comp_norm = static_cast<const uint16_t*>(d.comp_norm.d.p);
        if (r.ratio == 2) w.comp_wgate = static_cast<const uint16_t*>(d.comp_wgate.d.p);
      }
      w.ratio = r.ratio; w.cache_ord = r.ord; w.tail_ord = r.tail; w.kv_source = r.kv; w.index_source = r.idx;
      w.candidate_source = r.cand_src; w.uses_candidates = r.cand_use;
      RefLayer& R = d.ref;
      R.ratio = r.ratio; R.kv_source = r.kv; R.index_source = r.idx; R.candidate_source = r.cand_src; R.uses_candidates = r.cand_use;
      R.cache_ord = r.ord; R.tail_ord = r.tail;
      R.wq_a = &d.wq_a; R.wkv = &d.wkv; R.wq_b = &d.wq_b; R.wo_a = &d.wo_a; R.wo_b = &d.wo_b; R.idx_wq_b = &d.idx_wq_b;
      R.q_norm = &d.q_norm; R.kv_norm = &d.kv_norm; R.idx_wp = &d.idx_wp; R.idx_wk = &d.idx_wk; R.idx_k_norm = &d.idx_k_norm;
      R.comp_wkv = &d.comp_wkv; R.comp_wgate = &d.comp_wgate; R.comp_norm = &d.comp_norm; R.sink = d.sink; R.inv_freq = d.inv_freq;
    }
    ref = std::make_unique<RefModel>(cfg, max_requests, ratios, 1);
    for (auto& d : layers) ref->layers.push_back(d.ref);
    dgpp::Csa2PoolShape shape;
    shape.layers = int(layers.size()); shape.cache_ratio = ratios; shape.tail_ordinals = 1; shape.max_requests = max_requests;
    shape.token_slots = max_cache; shape.block_tokens = cfg.block_tokens; shape.ring_slots = cfg.ring_slots;
    pool.init(shape);
    const size_t sb = dgpp::Csa2Layer::scratch_bytes(cfg, max_tokens, max_cache, 16, 8, dot_budget);
    scratch = DevBuf(sb);
    gemm_ws = DevBuf(64u << 20);
    gemm.set_decode_rows(16);
    layer = std::make_unique<dgpp::Csa2Layer>(gemm, cfg, max_tokens, max_cache, scratch.p, sb, gemm_ws.p,
                                           64u << 20, 16, 8, dot_budget);
    layer->rebind(layers[0].w, 0);
    require(layer->prepare(max_tokens) && layer->prepare(16), "gemm plans");
    hidden.assign(size_t(max_requests), {});
  }
  const std::vector<float>& hidden_row(int req, int64_t pos) {
    auto& h = hidden[size_t(req)];
    while (h.size() < size_t(pos + 1) * cfg.hidden) {
      const int64_t p = int64_t(h.size()) / cfg.hidden;
      for (int i = 0; i < cfg.hidden; ++i) h.push_back(bf(0.9f * gauss(0xABC + req * 977 + p, i)));
    }
    return h;
  }
  std::vector<uint16_t> hidden_bits(int req, int64_t p0, int rows) {
    std::vector<uint16_t> out;
    for (int i = 0; i < rows; ++i) {
      const auto& h = hidden_row(req, p0 + i);
      for (int c = 0; c < cfg.hidden; ++c) out.push_back(float_to_bf16_bits(h[size_t(p0 + i) * cfg.hidden + c]));
    }
    return out;
  }

  // Compare the device rows with the oracle's; certify selection flips.
  void check(int l, const std::vector<uint16_t>& got, const std::vector<float>& want, int rows, const std::string& what,
             const std::vector<std::vector<int32_t>>* dev_sel, RefRequest& R) {
    std::vector<uint16_t> wb(want.size());
    for (size_t i = 0; i < want.size(); ++i) wb[i] = float_to_bf16_bits(want[i]);
    // The l2 budget is the gate; a few elements of a small partial output
    // land past 12 ulps after a certified selection flip.
    require_bf16(tag + " layer " + std::to_string(l) + " " + what + " output", compare_bf16(got, wb, 12), 0.03, 0.06);
    if (dev_sel == nullptr) return;
    for (int i = 0; i < rows; ++i) {
      const auto& d = (*dev_sel)[size_t(i)];
      const auto& r = R.sel[size_t(i)];
      if (d == r) continue;
      ++flips;
      // A flip: every entry in one set and not the other scores within
      // 1e-3 (relative to the logit range) of the boundary.
      const auto& lg = R.logits[size_t(i)];
      float lo = INFINITY, hi = -INFINITY;
      for (const float v : lg) { lo = std::min(lo, v); hi = std::max(hi, v); }
      float boundary = INFINITY;
      for (const int32_t e : r) boundary = std::min(boundary, lg[size_t(e)]);
      for (const int32_t e : d) require(e >= 0 && size_t(e) < lg.size(), "device selected an invisible entry");
      for (const int32_t e : d)
        if (std::find(r.begin(), r.end(), e) == r.end())
          require(std::fabs(lg[size_t(e)] - boundary) <= 2e-3f * std::max(1e-6f, hi - lo) + 1e-6f,
                  tag + " layer " + std::to_string(l) + " " + what + ": a selection flip beyond a near tie");
      for (const int32_t e : r)
        if (std::find(d.begin(), d.end(), e) == d.end())
          require(std::fabs(lg[size_t(e)] - boundary) <= 2e-3f * std::max(1e-6f, hi - lo) + 1e-6f,
                  tag + " layer " + std::to_string(l) + " " + what + ": a dropped entry beyond a near tie");
    }
  }

  void prefill(int req, int64_t p0, int T) {
    require(pool.ensure_request_blocks(req, p0 + T, 0), "blocks");
    const auto hb = hidden_bits(req, p0, T);
    std::vector<float> hf(hb.size());
    for (size_t i = 0; i < hb.size(); ++i) hf[i] = bf16_bits_to_float(hb[i]);
    DevBuf dh = upload(hb), dout(size_t(T) * cfg.hidden * 2);
    for (size_t l = 0; l < layers.size(); ++l) {
      layer->rebind(layers[l].w, int(l));
      layer->enqueue_prefill(dh.p, pool, req, p0, T, dout.p, 0);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const auto got = download<uint16_t>(dout.p, size_t(T) * cfg.hidden);
      record.push_back(got);
      const auto want = ref->layer_rows(int(l), hf, p0, T, req);
      std::vector<std::vector<int32_t>> dsel;
      if (layers[l].w.ratio > 0) {
        const auto topk = download<int32_t>(layer->debug_topk(), size_t(T) * cfg.index_topk);
        const auto counts = download<int32_t>(layer->debug_counts(), size_t(T));
        dsel.resize(size_t(T));
        for (int i = 0; i < T; ++i) dsel[size_t(i)].assign(topk.begin() + i * cfg.index_topk, topk.begin() + i * cfg.index_topk + counts[size_t(i)]);
      }
      check(int(l), got, want, T, "prefill req " + std::to_string(req) + " pos " + std::to_string(p0) + "+" + std::to_string(T),
            layers[l].w.ratio > 0 ? &dsel : nullptr, ref->reqs[size_t(req)]);
    }
  }
  // A decode batch: spans of (req, p0, rows); tail snapshots returned per row.
  struct Span { int req; int64_t p0; int rows; };
  std::vector<float> snapshots;  // [tokens, 2, 512] of the last batch on the ratio-2 source
  void decode(const std::vector<Span>& spans) {
    std::vector<int32_t> req_ids, sp;
    std::vector<int64_t> pos;
    std::vector<uint16_t> hb;
    for (const auto& s : spans) {
      require(pool.ensure_request_blocks(s.req, s.p0 + s.rows, 0), "blocks");
      sp.push_back(int(req_ids.size()));
      sp.push_back(s.rows);
      const auto h = hidden_bits(s.req, s.p0, s.rows);
      hb.insert(hb.end(), h.begin(), h.end());
      for (int i = 0; i < s.rows; ++i) { req_ids.push_back(s.req); pos.push_back(s.p0 + i); }
    }
    const int T = int(pos.size());
    DevBuf dh = upload(hb), dri = upload(req_ids), dpos = upload(pos), dsp = upload(sp), dout(size_t(T) * cfg.hidden * 2),
        dsnap(size_t(T) * 2 * 512 * 4);
    for (size_t l = 0; l < layers.size(); ++l) {
      layer->rebind(layers[l].w, int(l));
      layer->enqueue_decode(dh.p, pool, static_cast<const int32_t*>(dri.p), static_cast<const int64_t*>(dpos.p),
                            static_cast<const int32_t*>(dsp.p), int(spans.size()), T, dout.p, 0,
                            layers[l].w.ratio == 2 && layers[l].w.kv_source ? static_cast<float*>(dsnap.p) : nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      if (layers[l].w.ratio == 2 && layers[l].w.kv_source) snapshots = download<float>(dsnap.p, size_t(T) * 2 * 512);
      const auto got = download<uint16_t>(dout.p, size_t(T) * cfg.hidden);
      std::vector<int32_t> topk, counts;
      if (layers[l].w.ratio > 0) {
        topk = download<int32_t>(layer->debug_topk(), size_t(T) * cfg.index_topk);
        counts = download<int32_t>(layer->debug_counts(), size_t(T));
      }
      int row = 0;
      for (const auto& s : spans) {
        std::vector<float> hf(size_t(s.rows) * cfg.hidden);
        for (size_t i = 0; i < hf.size(); ++i) hf[i] = bf16_bits_to_float(hb[size_t(row) * cfg.hidden + i]);
        const auto want = ref->layer_rows(int(l), hf, s.p0, s.rows, s.req);
        const std::vector<uint16_t> g(got.begin() + size_t(row) * cfg.hidden, got.begin() + size_t(row + s.rows) * cfg.hidden);
        std::vector<std::vector<int32_t>> dsel(size_t(s.rows));
        for (int i = 0; i < s.rows; ++i)
          if (layers[l].w.ratio > 0)
            dsel[size_t(i)].assign(topk.begin() + (row + i) * cfg.index_topk, topk.begin() + (row + i) * cfg.index_topk + counts[size_t(row + i)]);
        check(int(l), g, want, s.rows, "decode req " + std::to_string(s.req) + " pos " + std::to_string(s.p0) + "+" + std::to_string(s.rows),
              layers[l].w.ratio > 0 ? &dsel : nullptr, ref->reqs[size_t(s.req)]);
        row += s.rows;
      }
    }
  }
  // Roll request `req` back to `len` accepted tokens: the device restores
  // the tail snapshot of row `snap_row` of the last batch; the oracle
  // replays its history from scratch.
  void rollback(int req, int64_t len, int snap_row) {
    DGPP_CUDA_OK(cudaMemcpy(pool.tails(0) + size_t(req) * 2 * 512, snapshots.data() + size_t(snap_row) * 2 * 512, 2 * 512 * 4,
                            cudaMemcpyHostToDevice));
    ref->reset(ref->reqs[size_t(req)]);
    // Replay: chunks of block_tokens then the remainder, layer by layer.
    for (int64_t p0 = 0; p0 < len; p0 += cfg.block_tokens) {
      const int T = int(std::min<int64_t>(cfg.block_tokens, len - p0));
      const auto hb = hidden_bits(req, p0, T);
      std::vector<float> hf(hb.size());
      for (size_t i = 0; i < hb.size(); ++i) hf[i] = bf16_bits_to_float(hb[i]);
      for (size_t l = 0; l < layers.size(); ++l) (void)ref->layer_rows(int(l), hf, p0, T, req);
    }
  }
};
}  // namespace

DGPP_TEST(csa2_layer_matches_the_oracle_over_prefill_decode_and_rollback) {
  for (const int tp : {1, 4}) {
    Scenario s(tp, tp == 1 ? "tp1" : "tp4-rank0");
    // Request 0: 96 tokens in three block chunks (the ring wraps at 48);
    // request 1: 32 + 8 (the odd-tail-free path, a short second chunk).
    s.prefill(0, 0, 32);
    s.prefill(0, 32, 32);
    s.prefill(0, 64, 32);
    s.prefill(1, 0, 32);
    s.prefill(1, 32, 8);
    // A verify batch: request 0 rows 96..101 (six spec rows), request 1
    // rows 40..42. Then request 0 accepts three rows: rolled back to 99
    // with the snapshot after row 98 (batch row 2), and decodes 99..104
    // over the overwritten slots; request 1 accepts all three.
    s.decode({{0, 96, 6}, {1, 40, 3}});
    s.rollback(0, 99, 2);
    s.decode({{0, 99, 6}, {1, 43, 2}});
    s.rollback(0, 101, 1);
    s.decode({{1, 45, 1}, {0, 101, 4}});
    require(s.layer->index_violations() == 0, s.tag + ": index-key exactness violations");
    std::printf("[INFO] %s: %d selection flips, every one certified as a near tie\n", s.tag.c_str(), s.flips);
  }
}

DGPP_TEST(csa2_prefill_reuses_score_workspace_at_shorter_contexts) {
  for (const int tp : {1, 4}) {
    for (const int64_t capacity : {512, 4096}) {
      // Both allocations fit only one query at their maximum context.
      // A 32-token prompt uses far fewer entries and must batch queries
      // within that same allocation. The oracle checks ratio-1/ratio-2
      // sources, restricted selection and selection reuse at both TP sizes.
      Scenario s(tp, "workspace-tp" + std::to_string(tp) + "-cap" + std::to_string(capacity),
                 capacity, 64ull << 10);
      s.prefill(0, 0, 32);
      require(s.layer->debug_logits_rows() > 1,
              "prefill still tiles queries for the allocated capacity instead of the visible history");
      s.prefill(0, 32, 32);
      s.prefill(0, 64, 17);  // partial query tile and an odd compressor tail
      s.decode({{0, 81, 3}});
      require(s.layer->index_violations() == 0, s.tag + ": index-key exactness violations");
      std::printf("[INFO] %s: %d selection flips, every one certified as a near tie\n", s.tag.c_str(), s.flips);
    }
  }
}

// A prefill tile's two sources attend from the staged copy of the tile's
// distinct rows (2026-11-16, kernels/dsa.hpp dsa_attn_listed_staged): the
// window scratch's fp8_block rows and the compressed source's fp4_block
// entries, one source after the other through the same workspace. Every
// layer's rows of every chunk are bit for bit the rows without the staged
// form, over the chunks of two requests (the second request's lists name
// its own table's blocks), and the staged launches were taken where the
// listed tensor-core kernel runs (sixteen local heads: TP 1).
DGPP_TEST(csa2_prefill_staged_tiles_are_the_unstaged_output) {
  for (const int tp : {1, 4}) {
    std::vector<std::vector<uint16_t>> want;
    for (const bool staged : {false, true}) {
      Scenario s(tp, std::string("staged-tp") + std::to_string(tp) + (staged ? "" : "-off"));
      s.layer->set_staged_prefill(staged);
      unsigned long long st[4];
      dgpp::dsa_attn_stage_stats(st, /*clear=*/true, 0);
      s.prefill(0, 0, 32);
      s.prefill(0, 32, 32);
      s.prefill(1, 0, 32);
      s.prefill(0, 64, 32);
      s.prefill(1, 32, 8);
      dgpp::dsa_attn_stage_stats(st, /*clear=*/false, 0);
      const bool flash = s.cfg.local_heads() >= 16;
      if (!staged) {
        require(st[0] == 0, s.tag + ": staged launches with the staged form off");
        want = s.record;
        continue;
      }
      require(st[1] == 0, s.tag + ": a tile's distinct rows exceeded the copy");
      require(flash ? st[0] >= 5 * 6 : st[0] == 0,
              s.tag + ": " + std::to_string(st[0]) + " staged launches (the listed kernel " + (flash ? "runs" : "does not run") +
                  " at this head count)");
      require(s.record.size() == want.size(), s.tag + ": the walks differ in their calls");
      for (size_t i = 0; i < want.size(); ++i)
        require(s.record[i] == want[i], s.tag + ": prefill call " + std::to_string(i) +
                                            " differs between the staged and the unstaged form");
    }
  }
}

int main() { return dgpp::test::run_all(); }
