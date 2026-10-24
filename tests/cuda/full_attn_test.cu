// Dense causal paged attention (the Qwen3.5 full-attention kernel) against a
// host reference on small geometry: query rows at positions across tile and
// block boundaries plus a padding row, two requests sharing the miniature
// pool. The reference models the kernel's FA2 chain (16-token tiles, the
// per-tile running max, bf16-rounded probabilities, the unrounded
// denominator, all in the exp2 domain); tolerance follows the glm4 split
// class: two bf16 ulps with a 0.5 % of RMS absolute floor, under 1 % of the
// elements outside it.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/full_attn.hpp"
#include "kernels/qsa.hpp"

using namespace dgpp::kda_test;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

constexpr int D = 256;
constexpr int kTileTok = 16;   // the row form's tile
constexpr int kTileTok32 = 32; // the tile form's

struct Geo {
  int local_heads = 4, kv_heads = 1;
  int block_tokens = 32, blocks_per_request = 8, max_requests = 2;
  int slots() const { return block_tokens * blocks_per_request; }
  int width() const { return kv_heads * D; }
};

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
template <class T>
const T* ptr(const DevBuf& b) { return static_cast<const T*>(b.p); }
template <class T>
T* mptr(DevBuf& b) { return static_cast<T*>(b.p); }

// Every request owns a distinct physical block range: request q's block b
// is physical q * blocks_per_request + b (the pool's tables in miniature).
std::vector<int32_t> tables(const Geo& g) {
  std::vector<int32_t> t(static_cast<size_t>(g.max_requests) * g.blocks_per_request);
  for (int q = 0; q < g.max_requests; ++q)
    for (int b = 0; b < g.blocks_per_request; ++b)
      t[static_cast<size_t>(q) * g.blocks_per_request + b] = q * g.blocks_per_request + b;
  return t;
}
int64_t phys(const Geo& g, int req, int64_t pos) {
  return static_cast<int64_t>(req) * g.blocks_per_request * g.block_tokens + pos;
}

float dot_bf16(const uint16_t* a, const uint16_t* b, int n) {
  float acc = 0.f;
  for (int i = 0; i < n; ++i) acc += dgpp::bf16_bits_to_float(a[i]) * dgpp::bf16_bits_to_float(b[i]);
  return acc;
}

}  // namespace

DGPP_TEST(full_attention_matches_the_reference_dense_causal) {
  Geo g;
  cudaStream_t st = test_stream();
  // Request 0 holds 100 tokens, request 1 holds 10; the cache is filled
  // host-side into each request's paged blocks.
  const int lens[2] = {100, 10};
  const size_t cache_bytes = static_cast<size_t>(g.max_requests) * g.slots() * g.width() * 2;
  std::vector<uint16_t> kh(cache_bytes / 2, 0x7F7F), vh(cache_bytes / 2, 0x7F7F);
  for (int q = 0; q < 2; ++q) {
    const std::vector<uint16_t> k = random_bf16_normal(40 + q, static_cast<int64_t>(lens[q]) * g.width(), 1.0f);
    const std::vector<uint16_t> v = random_bf16_normal(50 + q, static_cast<int64_t>(lens[q]) * g.width(), 1.0f);
    for (int t = 0; t < lens[q]; ++t) {
      std::copy(k.begin() + static_cast<size_t>(t) * g.width(), k.begin() + static_cast<size_t>(t + 1) * g.width(),
                kh.begin() + phys(g, q, t) * g.width());
      std::copy(v.begin() + static_cast<size_t>(t) * g.width(), v.begin() + static_cast<size_t>(t + 1) * g.width(),
                vh.begin() + phys(g, q, t) * g.width());
    }
  }
  DevBuf kc = up(kh), vc = up(vh), dtbl = up(tables(g));
  // Query rows at positions across tile (16) and block (32) boundaries, one
  // padding row, and one row of the short request.
  const std::vector<int32_t> req_ids = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
  const std::vector<int64_t> pos = {0, 1, 15, 16, 17, 31, 32, 33, 63, 99, 9, -1};
  const int rows = static_cast<int>(pos.size());
  const int qw = g.local_heads * D;
  const std::vector<uint16_t> q = random_bf16_normal(60, static_cast<int64_t>(rows) * qw, 1.0f);
  DevBuf dq = up(q), dpos = up(pos), dreq = up(req_ids);
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const float sl2 = scale * 1.4426950408889634f;

  // The reference models the kernel's tile chain: the running max and the
  // denominator in the exp2 domain, probabilities rounded to bf16 per tile.
  std::vector<float> ref(static_cast<size_t>(rows) * qw, 0.f);
  const int group = g.local_heads / g.kv_heads;
  for (int r = 0; r < rows; ++r) {
    const int64_t p = pos[static_cast<size_t>(r)];
    if (p < 0) continue;
    const int req = req_ids[static_cast<size_t>(r)];
    for (int h = 0; h < g.local_heads; ++h) {
      const uint16_t* qh = q.data() + static_cast<size_t>(r) * qw + h * D;
      const int kv_h = h / group;
      float m = -INFINITY, l = 0.f;
      std::vector<float> acc(D, 0.f);
      for (int64_t t0 = 0; t0 <= p; t0 += kTileTok) {
        const int n = static_cast<int>(std::min<int64_t>(kTileTok, p + 1 - t0));
        float s[kTileTok];
        float tile_max = -INFINITY;
        for (int j = 0; j < n; ++j) {
          const uint16_t* kj =
              kh.data() + (phys(g, req, t0 + j) * g.width() + kv_h * D);
          s[j] = dot_bf16(qh, kj, D) * sl2;
          tile_max = std::max(tile_max, s[j]);
        }
        const float nm = std::max(m, tile_max);
        const float resc = exp2f(m - nm);
        float lnew = l * resc;
        std::vector<float> anew(D, 0.f);
        for (int d = 0; d < D; ++d) anew[static_cast<size_t>(d)] = acc[static_cast<size_t>(d)] * resc;
        for (int j = 0; j < n; ++j) {
          const float pj = exp2f(s[j] - nm);
          lnew += pj;
          const float pb = dgpp::bf16_bits_to_float(dgpp::float_to_bf16_bits(pj));
          const uint16_t* vj = vh.data() + (phys(g, req, t0 + j) * g.width() + kv_h * D);
          for (int d = 0; d < D; ++d) anew[static_cast<size_t>(d)] += pb * dgpp::bf16_bits_to_float(vj[d]);
        }
        m = nm;
        l = lnew;
        acc.swap(anew);
      }
      float* out = ref.data() + static_cast<size_t>(r) * qw + h * D;
      const float inv = l > 0.f ? 1.f / l : 0.f;
      for (int d = 0; d < D; ++d) out[d] = acc[static_cast<size_t>(d)] * inv;
    }
  }

  DevBuf out(ref.size() * 4);
  DGPP_CUDA_OK(cudaMemset(out.p, 0x7F, ref.size() * 4));
  dgpp::full_attn_prefill_warp(ptr<uint16_t>(dq), qw, ptr<uint16_t>(kc), ptr<uint16_t>(vc), ptr<int32_t>(dreq),
                               ptr<int64_t>(dpos), rows, g.local_heads, g.kv_heads, g.block_tokens,
                               ptr<int32_t>(dtbl), g.blocks_per_request, scale, mptr<float>(out), st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<float> got = down<float>(out, ref.size());
  double rms = 0;
  for (size_t i = 0; i < got.size(); ++i) rms += static_cast<double>(ref[i]) * ref[i];
  rms = std::sqrt(rms / static_cast<double>(ref.size()));
  const Stats s = compare_abs_rel(got.data(), ref.data(), static_cast<long>(got.size()), 2 * std::pow(2.0, -7.0),
                                  0.005 * rms);
  std::printf("[ .. ] full attention: max_abs %.3g l2_rel %.3g mismatches %ld/%ld (rms %.3g)\n", s.max_abs, s.l2_rel,
              s.mismatches, s.n, rms);
  require_bf16("full attention", s, 2e-3, 0.01);
  // The padding row is zeros.
  for (int i = 0; i < qw; ++i) require(got[static_cast<size_t>(rows - 1) * qw + i] == 0.f, "padding row is zero");
}

DGPP_TEST(full_attention_second_run_is_bitwise) {
  Geo g;
  cudaStream_t st = test_stream();
  const std::vector<int32_t> req_ids = {0, 0, 1};
  const std::vector<int64_t> pos = {17, 40, 5};
  const int rows = 3;
  const int qw = g.local_heads * D;
  const std::vector<uint16_t> q = random_bf16_normal(70, static_cast<int64_t>(rows) * qw, 1.0f);
  const size_t cache_bytes = static_cast<size_t>(g.max_requests) * g.slots() * g.width() * 2;
  const std::vector<uint16_t> kh(cache_bytes / 2, 0x3C00), vh(cache_bytes / 2, 0xBC00);
  DevBuf dq = up(q), dpos = up(pos), dreq = up(req_ids), kc = up(kh), vc = up(vh), dtbl = up(tables(g));
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  DevBuf out(static_cast<size_t>(rows) * qw * 4);
  auto run = [&] {
    dgpp::full_attn_prefill_warp(ptr<uint16_t>(dq), qw, ptr<uint16_t>(kc), ptr<uint16_t>(vc), ptr<int32_t>(dreq),
                                 ptr<int64_t>(dpos), rows, g.local_heads, g.kv_heads, g.block_tokens,
                                 ptr<int32_t>(dtbl), g.blocks_per_request, scale, mptr<float>(out), st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    return down<float>(out, static_cast<size_t>(rows) * qw);
  };
  const std::vector<float> first = run();
  require(run() == first, "second run bitwise the first");
}

// The reference for one query head over its causal range with an FA2 chain
// of `tile` tokens (per-tile max, bf16-rounded P, unrounded denominator).
static void ref_head(const std::vector<uint16_t>& q, int row, int h, int req, int64_t p, const Geo& g,
                     const std::vector<uint16_t>& kh, const std::vector<uint16_t>& vh, float sl2, int tile,
                     float* out) {
  const int qw = g.local_heads * D;
  const int group = g.local_heads / g.kv_heads;
  const uint16_t* qh = q.data() + static_cast<size_t>(row) * qw + h * D;
  const int kv_h = h / group;
  float m = -INFINITY, l = 0.f;
  std::vector<float> acc(D, 0.f);
  std::vector<float> sv(tile);
  for (int64_t t0 = 0; t0 <= p; t0 += tile) {
    const int n = static_cast<int>(std::min<int64_t>(tile, p + 1 - t0));
    float tile_max = -INFINITY;
    for (int j = 0; j < n; ++j) {
      const uint16_t* kj = kh.data() + (phys(g, req, t0 + j) * g.width() + kv_h * D);
      sv[j] = dot_bf16(qh, kj, D) * sl2;
      tile_max = std::max(tile_max, sv[j]);
    }
    const float nm = std::max(m, tile_max);
    const float resc = exp2f(m - nm);
    float lnew = l * resc;
    std::vector<float> anew(D, 0.f);
    for (int d = 0; d < D; ++d) anew[static_cast<size_t>(d)] = acc[static_cast<size_t>(d)] * resc;
    for (int j = 0; j < n; ++j) {
      const float pj = exp2f(sv[j] - nm);
      lnew += pj;
      const float pb = dgpp::bf16_bits_to_float(dgpp::float_to_bf16_bits(pj));
      const uint16_t* vj = vh.data() + (phys(g, req, t0 + j) * g.width() + kv_h * D);
      for (int d = 0; d < D; ++d) anew[static_cast<size_t>(d)] += pb * dgpp::bf16_bits_to_float(vj[d]);
    }
    m = nm;
    l = lnew;
    acc.swap(anew);
  }
  const float inv = l > 0.f ? 1.f / l : 0.f;
  for (int d = 0; d < D; ++d) out[d] = acc[static_cast<size_t>(d)] * inv;
}

// The split walk (kMaxSplits key ranges + the combine) over the same mixed
// rows: every range rescales at its own boundaries, so the result is
// tolerance-equal to the 16-token-tile reference like the unsplit form.
DGPP_TEST(full_attention_split_walk_matches_the_reference) {
  Geo g;
  cudaStream_t st = test_stream();
  const int lens[2] = {700, 10};
  g.blocks_per_request = 32;  // 1024 slots per request
  const size_t cache_bytes = static_cast<size_t>(g.max_requests) * g.slots() * g.width() * 2;
  std::vector<uint16_t> kh(cache_bytes / 2, 0x7F7F), vh(cache_bytes / 2, 0x7F7F);
  for (int q = 0; q < 2; ++q) {
    const std::vector<uint16_t> k = random_bf16_normal(80 + q, static_cast<int64_t>(lens[q]) * g.width(), 1.0f);
    const std::vector<uint16_t> v = random_bf16_normal(90 + q, static_cast<int64_t>(lens[q]) * g.width(), 1.0f);
    for (int t = 0; t < lens[q]; ++t) {
      std::copy(k.begin() + static_cast<size_t>(t) * g.width(), k.begin() + static_cast<size_t>(t + 1) * g.width(),
                kh.begin() + phys(g, q, t) * g.width());
      std::copy(v.begin() + static_cast<size_t>(t) * g.width(), v.begin() + static_cast<size_t>(t + 1) * g.width(),
                vh.begin() + phys(g, q, t) * g.width());
    }
  }
  DevBuf kc = up(kh), vc = up(vh), dtbl = up(tables(g));
  const std::vector<int32_t> req_ids = {0, 0, 0, 1, 0, 0};
  const std::vector<int64_t> pos = {0, 17, 699, 9, 511, -1};
  const int rows = static_cast<int>(pos.size());
  const int qw = g.local_heads * D;
  const std::vector<uint16_t> q = random_bf16_normal(100, static_cast<int64_t>(rows) * qw, 1.0f);
  DevBuf dq = up(q), dpos = up(pos), dreq = up(req_ids);
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const float sl2 = scale * 1.4426950408889634f;
  std::vector<float> ref(static_cast<size_t>(rows) * qw, 0.f);
  for (int r = 0; r < rows; ++r) {
    if (pos[static_cast<size_t>(r)] < 0) continue;
    for (int h = 0; h < g.local_heads; ++h)
      ref_head(q, r, h, req_ids[static_cast<size_t>(r)], pos[static_cast<size_t>(r)], g, kh, vh, sl2, kTileTok,
               ref.data() + static_cast<size_t>(r) * qw + h * D);
  }
  DevBuf out(ref.size() * 4), part(dgpp::full_attn_partials_bytes(rows, g.kv_heads));
  DGPP_CUDA_OK(cudaMemset(out.p, 0x7F, ref.size() * 4));
  dgpp::full_attn_decode(ptr<uint16_t>(dq), qw, ptr<uint16_t>(kc), ptr<uint16_t>(vc), ptr<int32_t>(dreq),
                         ptr<int64_t>(dpos), rows, g.local_heads, g.kv_heads, g.block_tokens, ptr<int32_t>(dtbl),
                         g.blocks_per_request, scale, mptr<float>(out), mptr<float>(part), st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<float> got = down<float>(out, ref.size());
  double rms = 0;
  for (size_t i = 0; i < got.size(); ++i) rms += static_cast<double>(ref[i]) * ref[i];
  rms = std::sqrt(rms / static_cast<double>(ref.size()));
  const Stats s = compare_abs_rel(got.data(), ref.data(), static_cast<long>(got.size()), 2 * std::pow(2.0, -7.0),
                                  0.005 * rms);
  std::printf("[ .. ] split walk: max_abs %.3g l2_rel %.3g mismatches %ld/%ld (rms %.3g)\n", s.max_abs, s.l2_rel,
              s.mismatches, s.n, rms);
  require_bf16("split walk", s, 2e-3, 0.01);
  for (int i = 0; i < qw; ++i) require(got[static_cast<size_t>(rows - 1) * qw + i] == 0.f, "padding row is zero");
  // The block form over the same rows grouped by request (rows 0..2 of
  // request 0, row 3 of request 1, row 4 of request 0, the padding row):
  // bitwise the row form, split and unsplit.
  {
    const std::vector<int32_t> spans = {0, 3, 3, 1, 4, 1, 5, 1};
    DevBuf dspans = up(spans);
    const dgpp::FullAttnStageMaps maps = dgpp::full_attn_stage_maps(
        ptr<uint16_t>(kc), ptr<uint16_t>(vc), static_cast<int64_t>(g.max_requests) * g.slots(), g.kv_heads);
    DevBuf outb(ref.size() * 4), partb(dgpp::full_attn_partials_bytes(rows, g.kv_heads));
    DGPP_CUDA_OK(cudaMemset(outb.p, 0x7F, ref.size() * 4));
    dgpp::full_attn_decode_blocks(maps, 0, ptr<uint16_t>(dq), qw, ptr<int32_t>(dreq), ptr<int64_t>(dpos),
                                  ptr<int32_t>(dspans), 4, rows, ptr<int32_t>(dtbl), g.blocks_per_request,
                                  g.local_heads, g.kv_heads, g.block_tokens, scale, mptr<float>(outb),
                                  mptr<float>(partb), st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    require(down<float>(outb, ref.size()) == got, "block form (split) bitwise the row form");
    DevBuf outu(ref.size() * 4), outr(ref.size() * 4);
    DGPP_CUDA_OK(cudaMemset(outu.p, 0x7F, ref.size() * 4));
    dgpp::full_attn_decode_blocks(maps, 0, ptr<uint16_t>(dq), qw, ptr<int32_t>(dreq), ptr<int64_t>(dpos),
                                  ptr<int32_t>(dspans), 4, rows, ptr<int32_t>(dtbl), g.blocks_per_request,
                                  g.local_heads, g.kv_heads, g.block_tokens, scale, mptr<float>(outu), nullptr, st);
    dgpp::full_attn_decode(ptr<uint16_t>(dq), qw, ptr<uint16_t>(kc), ptr<uint16_t>(vc), ptr<int32_t>(dreq),
                           ptr<int64_t>(dpos), rows, g.local_heads, g.kv_heads, g.block_tokens, ptr<int32_t>(dtbl),
                           g.blocks_per_request, scale, mptr<float>(outr), nullptr, st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    require(down<float>(outu, ref.size()) == down<float>(outr, ref.size()),
            "block form (unsplit) bitwise the row form");
    // One lone group (no spans) of the first three rows: the same bits for them.
    DevBuf outl(ref.size() * 4);
    DGPP_CUDA_OK(cudaMemset(outl.p, 0x7F, ref.size() * 4));
    dgpp::full_attn_decode_blocks(maps, 0, ptr<uint16_t>(dq), qw, ptr<int32_t>(dreq), ptr<int64_t>(dpos), nullptr, 1,
                                  3, ptr<int32_t>(dtbl), g.blocks_per_request, g.local_heads, g.kv_heads,
                                  g.block_tokens, scale, mptr<float>(outl), mptr<float>(partb), st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    const std::vector<float> lone = down<float>(outl, ref.size());
    require(std::equal(lone.begin(), lone.begin() + 3 * qw, got.begin()), "lone group bitwise the row form");
  }
  // The unsplit form over the same rows agrees with the split one to the same tolerance.
  DevBuf out1(ref.size() * 4);
  dgpp::full_attn_prefill_warp(ptr<uint16_t>(dq), qw, ptr<uint16_t>(kc), ptr<uint16_t>(vc), ptr<int32_t>(dreq),
                               ptr<int64_t>(dpos), rows, g.local_heads, g.kv_heads, g.block_tokens,
                               ptr<int32_t>(dtbl), g.blocks_per_request, scale, mptr<float>(out1), st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<float> got1 = down<float>(out1, ref.size());
  const Stats s1 = compare_abs_rel(got.data(), got1.data(), static_cast<long>(got.size()), 2 * std::pow(2.0, -7.0),
                                   0.005 * rms);
  require_bf16("split vs unsplit", s1, 2e-3, 0.01);
}

// The tile form: 100 consecutive rows of one request (seven 16-row tiles, the
// last partial), 32-token K/V tiles, against the 32-token-tile reference.
DGPP_TEST(full_attention_tile_form_matches_the_reference) {
  Geo g;
  g.local_heads = 6;  // the 27B's group (24 heads / 4 kv heads), one kv head here
  g.kv_heads = 1;
  cudaStream_t st = test_stream();
  const int len = 100;
  const size_t cache_bytes = static_cast<size_t>(g.max_requests) * g.slots() * g.width() * 2;
  std::vector<uint16_t> kh(cache_bytes / 2, 0x7F7F), vh(cache_bytes / 2, 0x7F7F);
  {
    const std::vector<uint16_t> k = random_bf16_normal(110, static_cast<int64_t>(len) * g.width(), 1.0f);
    const std::vector<uint16_t> v = random_bf16_normal(111, static_cast<int64_t>(len) * g.width(), 1.0f);
    for (int t = 0; t < len; ++t) {
      std::copy(k.begin() + static_cast<size_t>(t) * g.width(), k.begin() + static_cast<size_t>(t + 1) * g.width(),
                kh.begin() + phys(g, 1, t) * g.width());  // request 1: a non-identity block table row
      std::copy(v.begin() + static_cast<size_t>(t) * g.width(), v.begin() + static_cast<size_t>(t + 1) * g.width(),
                vh.begin() + phys(g, 1, t) * g.width());
    }
  }
  DevBuf kc = up(kh), vc = up(vh), dtbl = up(tables(g));
  const int rows = len;
  std::vector<int64_t> pos(static_cast<size_t>(rows));
  std::vector<int32_t> req_ids(static_cast<size_t>(rows), 1);
  for (int r = 0; r < rows; ++r) pos[static_cast<size_t>(r)] = r;
  const int qw = g.local_heads * D;
  const std::vector<uint16_t> q = random_bf16_normal(120, static_cast<int64_t>(rows) * qw, 1.0f);
  DevBuf dq = up(q), dpos = up(pos), dreq = up(req_ids);
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const float sl2 = scale * 1.4426950408889634f;
  std::vector<float> ref(static_cast<size_t>(rows) * qw, 0.f);
  for (int r = 0; r < rows; ++r)
    for (int h = 0; h < g.local_heads; ++h)
      ref_head(q, r, h, 1, pos[static_cast<size_t>(r)], g, kh, vh, sl2, kTileTok32,
               ref.data() + static_cast<size_t>(r) * qw + h * D);
  DevBuf out(ref.size() * 4);
  DGPP_CUDA_OK(cudaMemset(out.p, 0x7F, ref.size() * 4));
  dgpp::full_attn_prefill(ptr<uint16_t>(dq), qw, ptr<uint16_t>(kc), ptr<uint16_t>(vc),
                          ptr<int32_t>(dtbl) + g.blocks_per_request, ptr<int32_t>(dreq), ptr<int64_t>(dpos), rows,
                          g.local_heads, g.kv_heads, g.block_tokens, ptr<int32_t>(dtbl), g.blocks_per_request, scale,
                          mptr<float>(out), st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<float> got = down<float>(out, ref.size());
  double rms = 0;
  for (size_t i = 0; i < got.size(); ++i) rms += static_cast<double>(ref[i]) * ref[i];
  rms = std::sqrt(rms / static_cast<double>(ref.size()));
  const Stats s = compare_abs_rel(got.data(), ref.data(), static_cast<long>(got.size()), 2 * std::pow(2.0, -7.0),
                                  0.005 * rms);
  std::printf("[ .. ] tile form: max_abs %.3g l2_rel %.3g mismatches %ld/%ld (rms %.3g)\n", s.max_abs, s.l2_rel,
              s.mismatches, s.n, rms);
  require_bf16("tile form", s, 2e-3, 0.01);
  // Twice: bitwise.
  DevBuf out2(ref.size() * 4);
  dgpp::full_attn_prefill(ptr<uint16_t>(dq), qw, ptr<uint16_t>(kc), ptr<uint16_t>(vc),
                          ptr<int32_t>(dtbl) + g.blocks_per_request, ptr<int32_t>(dreq), ptr<int64_t>(dpos), rows,
                          g.local_heads, g.kv_heads, g.block_tokens, ptr<int32_t>(dtbl), g.blocks_per_request, scale,
                          mptr<float>(out2), st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  require(down<float>(out2, ref.size()) == got, "tile form: second run bitwise the first");
  // The staged form (TMA boxes over the miniature pool: one map per plane,
  // request 1's rows = its physical block range): the same bits.
  const dgpp::FullAttnStageMaps maps = dgpp::full_attn_stage_maps(
      ptr<uint16_t>(kc), ptr<uint16_t>(vc), static_cast<int64_t>(g.max_requests) * g.slots(), g.kv_heads);
  DevBuf out3(ref.size() * 4);
  DGPP_CUDA_OK(cudaMemset(out3.p, 0x7F, ref.size() * 4));
  dgpp::full_attn_prefill_tma(maps, /*row_base=*/0, ptr<uint16_t>(dq), qw, ptr<int32_t>(dtbl) + g.blocks_per_request,
                              ptr<int64_t>(dpos), rows, g.local_heads, g.kv_heads, g.block_tokens, scale,
                              mptr<float>(out3), st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  require(down<float>(out3, ref.size()) == got, "staged (TMA) form bitwise the cp.async form");
}


// The fp8 storage form (engine.kv_dtype = fp8): e4m3 planes of K / k_scale
// and V / v_scale with the two scales on the device. Gate: bitwise the bf16
// forms run on the widened planes (exact bf16 of every code) with scale *
// k_scale, their outputs times v_scale — the split and unsplit row walks
// and the staged tile form; and the fp8 append writes the host quantizer's
// codes (fp32 division, round-to-nearest-even, saturating).
DGPP_TEST(full_attention_fp8_storage_is_the_bf16_chain_on_widened_planes) {
  Geo g;
  g.local_heads = 12;  // two KV groups of six
  g.kv_heads = 2;
  g.blocks_per_request = 32;  // 1024 slots per request
  cudaStream_t st = test_stream();
  const float ks = 0.0390625f, vs = 0.1416015625f;  // two of the 27B release's scalars
  const int lens[2] = {700, 100};
  const size_t slots = static_cast<size_t>(g.max_requests) * g.slots();
  // The bf16 rows the layer would append (K after norm + RoPE, V), each
  // request at its block range; the planes: e4m3 codes, and their widening.
  // Slots past a request's rows hold the largest finite code (the staged
  // tile form reads a whole 32-key tile and masks by index; a NaN code
  // there would poison 0 x NaN, which a served pool never holds: reset
  // zeroes the planes and the append saturates).
  std::vector<uint16_t> kh(slots * g.width(), 0), vh(slots * g.width(), 0);
  std::vector<uint8_t> k8(slots * g.width(), 0x7E), v8(slots * g.width(), 0x7E);
  for (int q = 0; q < 2; ++q) {
    const std::vector<uint16_t> k = random_bf16_normal(130 + q, static_cast<int64_t>(lens[q]) * g.width(), 1.0f);
    const std::vector<uint16_t> v = random_bf16_normal(140 + q, static_cast<int64_t>(lens[q]) * g.width(), 1.0f);
    for (int t = 0; t < lens[q]; ++t)
      for (int e = 0; e < g.width(); ++e) {
        const size_t at = static_cast<size_t>(phys(g, q, t)) * g.width() + e;
        const uint16_t kb = k[static_cast<size_t>(t) * g.width() + e], vb = v[static_cast<size_t>(t) * g.width() + e];
        kh[at] = kb;
        vh[at] = vb;
        k8[at] = dgpp::float_to_fp8_e4m3_bits(dgpp::bf16_bits_to_float(kb) / ks);
        v8[at] = dgpp::float_to_fp8_e4m3_bits(dgpp::bf16_bits_to_float(vb) / vs);
      }
  }
  // The widened planes: the exact bf16 of every code, fill included.
  std::vector<uint16_t> kw(slots * g.width()), vw(slots * g.width());
  for (size_t i = 0; i < kw.size(); ++i) {
    kw[i] = dgpp::float_to_bf16_bits(dgpp::fp8_e4m3_bits_to_float(k8[i]));
    vw[i] = dgpp::float_to_bf16_bits(dgpp::fp8_e4m3_bits_to_float(v8[i]));
  }
  // Some codes must be subnormal or zero for the widening to be exercised.
  size_t sub = 0;
  for (size_t i = 0; i < k8.size(); ++i) sub += (k8[i] & 0x78) == 0;
  require(sub > 100, "the planes hold subnormal codes");
  DevBuf dk8 = up(k8), dv8 = up(v8), dkw = up(kw), dvw = up(vw), dtbl = up(tables(g));
  const std::vector<float> scales = {ks, vs};
  DevBuf dscales = up(scales);

  // ---- the append: the layer's bf16 rows into the codes ------------------
  {
    const int rows = lens[0] + lens[1];
    std::vector<int32_t> req(static_cast<size_t>(rows));
    std::vector<int64_t> pos(static_cast<size_t>(rows));
    std::vector<uint16_t> krows(static_cast<size_t>(rows) * g.width()), vrows(krows.size());
    int r = 0;
    for (int q = 0; q < 2; ++q)
      for (int t = 0; t < lens[q]; ++t, ++r) {
        req[static_cast<size_t>(r)] = q;
        pos[static_cast<size_t>(r)] = t;
        const size_t at = static_cast<size_t>(phys(g, q, t)) * g.width();
        std::copy(kh.begin() + at, kh.begin() + at + g.width(), krows.begin() + static_cast<size_t>(r) * g.width());
        std::copy(vh.begin() + at, vh.begin() + at + g.width(), vrows.begin() + static_cast<size_t>(r) * g.width());
      }
    DevBuf dreq = up(req), dpos = up(pos), dkr = up(krows), dvr = up(vrows);
    DevBuf ak(k8.size()), av(v8.size());
    DGPP_CUDA_OK(cudaMemset(ak.p, 0x7E, k8.size()));  // the fill of the expected planes
    DGPP_CUDA_OK(cudaMemset(av.p, 0x7E, v8.size()));
    dgpp::qsa_kv_append_fp8(ptr<uint16_t>(dkr), g.width(), ptr<uint16_t>(dvr), g.width(), ptr<int32_t>(dreq),
                            ptr<int64_t>(dpos), rows, ptr<int32_t>(dtbl), g.blocks_per_request, g.block_tokens,
                            g.kv_heads, D, ptr<float>(dscales), mptr<uint8_t>(ak), mptr<uint8_t>(av), st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    require(down<uint8_t>(ak, k8.size()) == k8, "fp8 append: K codes are the host quantizer's");
    require(down<uint8_t>(av, v8.size()) == v8, "fp8 append: V codes are the host quantizer's");
  }

  // ---- the row walks: decode rows across both requests --------------------
  const std::vector<int32_t> req_ids = {0, 0, 0, 1, 0, 1, 0};
  const std::vector<int64_t> pos = {0, 17, 699, 99, 511, 31, -1};
  const int rows = static_cast<int>(pos.size());
  const int qw = g.local_heads * D;
  const std::vector<uint16_t> q = random_bf16_normal(150, static_cast<int64_t>(rows) * qw, 1.0f);
  DevBuf dq = up(q), dpos = up(pos), dreq = up(req_ids);
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const size_t n = static_cast<size_t>(rows) * qw;
  auto scaled = [&](const DevBuf& b) {
    std::vector<float> o = down<float>(b, n);
    for (float& x : o) x = x * vs;
    return o;
  };
  for (int split = 0; split < 2; ++split) {
    DevBuf ref(n * 4), got(n * 4), part(dgpp::full_attn_partials_bytes(rows, g.kv_heads));
    DGPP_CUDA_OK(cudaMemset(ref.p, 0x7F, n * 4));
    DGPP_CUDA_OK(cudaMemset(got.p, 0x7F, n * 4));
    dgpp::full_attn_decode(ptr<uint16_t>(dq), qw, ptr<uint16_t>(dkw), ptr<uint16_t>(dvw), ptr<int32_t>(dreq),
                           ptr<int64_t>(dpos), rows, g.local_heads, g.kv_heads, g.block_tokens, ptr<int32_t>(dtbl),
                           g.blocks_per_request, scale * ks, mptr<float>(ref), split ? mptr<float>(part) : nullptr, st);
    dgpp::full_attn_decode_fp8(ptr<uint16_t>(dq), qw, ptr<uint8_t>(dk8), ptr<uint8_t>(dv8), ptr<int32_t>(dreq),
                               ptr<int64_t>(dpos), rows, g.local_heads, g.kv_heads, g.block_tokens,
                               ptr<int32_t>(dtbl), g.blocks_per_request, scale, ptr<float>(dscales), mptr<float>(got),
                               split ? mptr<float>(part) : nullptr, st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    const std::vector<float> want = scaled(ref), have = down<float>(got, n);
    require(have == want, split ? "fp8 split row walk bitwise the bf16 chain" : "fp8 row walk bitwise the bf16 chain");
    for (int i = 0; i < qw; ++i) require(have[static_cast<size_t>(rows - 1) * qw + i] == 0.f, "fp8: padding row is zero");
  }

  // ---- the block form: the same decode rows grouped by request ---------
  {
    const std::vector<int32_t> spans = {0, 3, 3, 1, 4, 1, 5, 1, 6, 1};
    DevBuf dspans = up(spans);
    const dgpp::FullAttnStageMaps maps =
        dgpp::full_attn_stage_maps(ptr<uint16_t>(dkw), ptr<uint16_t>(dvw), static_cast<int64_t>(slots), g.kv_heads);
    const dgpp::FullAttnStageMaps maps8 =
        dgpp::full_attn_stage_maps_fp8(ptr<uint8_t>(dk8), ptr<uint8_t>(dv8), static_cast<int64_t>(slots), g.kv_heads);
    for (int split = 0; split < 2; ++split) {
      DevBuf ref(n * 4), got(n * 4), rowf(n * 4), part(dgpp::full_attn_partials_bytes(rows, g.kv_heads));
      DGPP_CUDA_OK(cudaMemset(ref.p, 0x7F, n * 4));
      DGPP_CUDA_OK(cudaMemset(got.p, 0x7F, n * 4));
      dgpp::full_attn_decode_blocks(maps, 0, ptr<uint16_t>(dq), qw, ptr<int32_t>(dreq), ptr<int64_t>(dpos),
                                    ptr<int32_t>(dspans), 5, rows, ptr<int32_t>(dtbl), g.blocks_per_request,
                                    g.local_heads, g.kv_heads, g.block_tokens, scale * ks, mptr<float>(ref),
                                    split ? mptr<float>(part) : nullptr, st);
      dgpp::full_attn_decode_blocks_fp8(maps8, 0, ptr<uint16_t>(dq), qw, ptr<int32_t>(dreq), ptr<int64_t>(dpos),
                                        ptr<int32_t>(dspans), 5, rows, ptr<int32_t>(dtbl), g.blocks_per_request,
                                        g.local_heads, g.kv_heads, g.block_tokens, scale, ptr<float>(dscales),
                                        mptr<float>(got), split ? mptr<float>(part) : nullptr, st);
      dgpp::full_attn_decode_fp8(ptr<uint16_t>(dq), qw, ptr<uint8_t>(dk8), ptr<uint8_t>(dv8), ptr<int32_t>(dreq),
                                 ptr<int64_t>(dpos), rows, g.local_heads, g.kv_heads, g.block_tokens,
                                 ptr<int32_t>(dtbl), g.blocks_per_request, scale, ptr<float>(dscales),
                                 mptr<float>(rowf), split ? mptr<float>(part) : nullptr, st);
      DGPP_CUDA_OK(cudaStreamSynchronize(st));
      const std::vector<float> want = scaled(ref), have = down<float>(got, n);
      require(have == want, split ? "fp8 block form (split) bitwise the bf16 block form on widened planes"
                                  : "fp8 block form bitwise the bf16 block form on widened planes");
      require(have == down<float>(rowf, n), split ? "fp8 block form (split) bitwise the fp8 row form"
                                                  : "fp8 block form bitwise the fp8 row form");
    }
  }

  // ---- the staged tile form: request 0's 700 consecutive rows -----------
  {
    const int prows = lens[0];
    std::vector<int64_t> ppos(static_cast<size_t>(prows));
    for (int r = 0; r < prows; ++r) ppos[static_cast<size_t>(r)] = r;
    const std::vector<uint16_t> pq = random_bf16_normal(160, static_cast<int64_t>(prows) * qw, 1.0f);
    DevBuf dpq = up(pq), dppos = up(ppos);
    const size_t pn = static_cast<size_t>(prows) * qw;
    const dgpp::FullAttnStageMaps maps =
        dgpp::full_attn_stage_maps(ptr<uint16_t>(dkw), ptr<uint16_t>(dvw), static_cast<int64_t>(slots), g.kv_heads);
    const dgpp::FullAttnStageMaps maps8 =
        dgpp::full_attn_stage_maps_fp8(ptr<uint8_t>(dk8), ptr<uint8_t>(dv8), static_cast<int64_t>(slots), g.kv_heads);
    DevBuf ref(pn * 4), got(pn * 4);
    DGPP_CUDA_OK(cudaMemset(ref.p, 0x7F, pn * 4));
    DGPP_CUDA_OK(cudaMemset(got.p, 0x7F, pn * 4));
    dgpp::full_attn_prefill_tma(maps, 0, ptr<uint16_t>(dpq), qw, ptr<int32_t>(dtbl), ptr<int64_t>(dppos), prows,
                                g.local_heads, g.kv_heads, g.block_tokens, scale * ks, mptr<float>(ref), st);
    dgpp::full_attn_prefill_tma_fp8(maps8, 0, ptr<uint16_t>(dpq), qw, ptr<int32_t>(dtbl), ptr<int64_t>(dppos), prows,
                                    g.local_heads, g.kv_heads, g.block_tokens, scale, ptr<float>(dscales),
                                    mptr<float>(got), st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    std::vector<float> want = down<float>(ref, pn);
    for (float& x : want) x = x * vs;
    const std::vector<float> have = down<float>(got, pn);
    size_t diff = 0;
    std::vector<int> rows_hit, heads_hit(static_cast<size_t>(g.local_heads), 0);
    for (size_t i = 0; i < pn; ++i)
      if (have[i] != want[i]) {
        ++diff;
        const int r = static_cast<int>(i / qw), h = static_cast<int>((i % qw) / D);
        if (rows_hit.empty() || rows_hit.back() != r) rows_hit.push_back(r);
        ++heads_hit[static_cast<size_t>(h)];
      }
    std::printf("[ .. ] fp8 staged tile form: %zu/%zu elements differ from the bf16 chain\n", diff, pn);
    if (diff) {
      std::printf("[ .. ]   %zu rows differ:", rows_hit.size());
      for (size_t i = 0; i < rows_hit.size() && i < 40; ++i) std::printf(" %d", rows_hit[i]);
      std::printf("\n[ .. ]   per head:");
      for (int h = 0; h < g.local_heads; ++h) std::printf(" %d", heads_hit[static_cast<size_t>(h)]);
      std::printf("\n[ .. ]   first row: have %.8g %.8g want %.8g %.8g\n", have[static_cast<size_t>(rows_hit[0]) * qw],
                  have[static_cast<size_t>(rows_hit[0]) * qw + 1], want[static_cast<size_t>(rows_hit[0]) * qw],
                  want[static_cast<size_t>(rows_hit[0]) * qw + 1]);
    }
    require(diff == 0, "fp8 staged tile form bitwise the bf16 chain on widened planes");
  }
}


// The block form at a long context with the serving pool's 64-token pages:
// two requests (261 280 and 300 keys), verify-shaped groups of consecutive
// positions and a lone row, against the row form — bitwise, split and
// unsplit, bf16 and fp8. The ring wraps hundreds of times per split here;
// the miniature tests above never wrap it.
DGPP_TEST(full_attention_block_form_long_context_bitwise_the_row_form) {
  Geo g;
  g.local_heads = 12;  // two KV groups of six: the partials are laid out per KV head
  g.kv_heads = 2;
  g.block_tokens = 64;
  g.blocks_per_request = 4096;  // 262 144 slots per request: the serving context
  cudaStream_t st = test_stream();
  const int lens[2] = {261280, 300};
  const size_t slots = static_cast<size_t>(g.max_requests) * g.slots();
  const float ks = 0.0390625f, vs = 0.1416015625f;
  std::vector<uint8_t> k8(slots * g.width(), 0x7E), v8(slots * g.width(), 0x7E);
  for (int q = 0; q < 2; ++q) {
    const std::vector<uint16_t> k = random_bf16_normal(170 + q, static_cast<int64_t>(lens[q]) * g.width(), 1.0f);
    const std::vector<uint16_t> v = random_bf16_normal(180 + q, static_cast<int64_t>(lens[q]) * g.width(), 1.0f);
    for (int t = 0; t < lens[q]; ++t)
      for (int e = 0; e < g.width(); ++e) {
        const size_t at = static_cast<size_t>(phys(g, q, t)) * g.width() + e;
        k8[at] = dgpp::float_to_fp8_e4m3_bits(dgpp::bf16_bits_to_float(k[static_cast<size_t>(t) * g.width() + e]) / ks);
        v8[at] = dgpp::float_to_fp8_e4m3_bits(dgpp::bf16_bits_to_float(v[static_cast<size_t>(t) * g.width() + e]) / vs);
      }
  }
  std::vector<uint16_t> kw(slots * g.width()), vw(slots * g.width());
  for (size_t i = 0; i < kw.size(); ++i) {
    kw[i] = dgpp::float_to_bf16_bits(dgpp::fp8_e4m3_bits_to_float(k8[i]));
    vw[i] = dgpp::float_to_bf16_bits(dgpp::fp8_e4m3_bits_to_float(v8[i]));
  }
  DevBuf dk8 = up(k8), dv8 = up(v8), dkw = up(kw), dvw = up(vw), dtbl = up(tables(g));
  const std::vector<float> scales = {ks, vs};
  DevBuf dscales = up(scales);
  // Groups: request 0's verify block at 19992..19999, request 1's at
  // 292..299 (both with two padding rows), a lone row of request 0 at 5000.
  std::vector<int32_t> req_ids;
  std::vector<int64_t> pos;
  for (int r = 0; r < 8; ++r) { req_ids.push_back(0); pos.push_back(r < 6 ? 261272 + r : -1); }
  for (int r = 0; r < 8; ++r) { req_ids.push_back(1); pos.push_back(r < 6 ? 292 + r : -1); }
  req_ids.push_back(0); pos.push_back(5000);
  const std::vector<int32_t> spans = {0, 8, 8, 8, 16, 1};
  const int rows = static_cast<int>(pos.size());
  const int qw = g.local_heads * D;
  const std::vector<uint16_t> q = random_bf16_normal(190, static_cast<int64_t>(rows) * qw, 1.0f);
  DevBuf dq = up(q), dpos = up(pos), dreq = up(req_ids), dspans = up(spans);
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const size_t n = static_cast<size_t>(rows) * qw;
  const dgpp::FullAttnStageMaps maps =
      dgpp::full_attn_stage_maps(ptr<uint16_t>(dkw), ptr<uint16_t>(dvw), static_cast<int64_t>(slots), g.kv_heads);
  const dgpp::FullAttnStageMaps maps8 =
      dgpp::full_attn_stage_maps_fp8(ptr<uint8_t>(dk8), ptr<uint8_t>(dv8), static_cast<int64_t>(slots), g.kv_heads);
  for (int split = 0; split < 2; ++split) {
    DevBuf part(dgpp::full_attn_partials_bytes(rows, g.kv_heads));
    DevBuf rowb(n * 4), blkb(n * 4), row8(n * 4), blk8(n * 4);
    DGPP_CUDA_OK(cudaMemset(blkb.p, 0x7F, n * 4));
    DGPP_CUDA_OK(cudaMemset(blk8.p, 0x7F, n * 4));
    dgpp::full_attn_decode(ptr<uint16_t>(dq), qw, ptr<uint16_t>(dkw), ptr<uint16_t>(dvw), ptr<int32_t>(dreq),
                           ptr<int64_t>(dpos), rows, g.local_heads, g.kv_heads, g.block_tokens, ptr<int32_t>(dtbl),
                           g.blocks_per_request, scale * ks, mptr<float>(rowb), split ? mptr<float>(part) : nullptr, st);
    dgpp::full_attn_decode_blocks(maps, 0, ptr<uint16_t>(dq), qw, ptr<int32_t>(dreq), ptr<int64_t>(dpos),
                                  ptr<int32_t>(dspans), 3, rows, ptr<int32_t>(dtbl), g.blocks_per_request,
                                  g.local_heads, g.kv_heads, g.block_tokens, scale * ks, mptr<float>(blkb),
                                  split ? mptr<float>(part) : nullptr, st);
    dgpp::full_attn_decode_fp8(ptr<uint16_t>(dq), qw, ptr<uint8_t>(dk8), ptr<uint8_t>(dv8), ptr<int32_t>(dreq),
                               ptr<int64_t>(dpos), rows, g.local_heads, g.kv_heads, g.block_tokens, ptr<int32_t>(dtbl),
                               g.blocks_per_request, scale, ptr<float>(dscales), mptr<float>(row8),
                               split ? mptr<float>(part) : nullptr, st);
    dgpp::full_attn_decode_blocks_fp8(maps8, 0, ptr<uint16_t>(dq), qw, ptr<int32_t>(dreq), ptr<int64_t>(dpos),
                                      ptr<int32_t>(dspans), 3, rows, ptr<int32_t>(dtbl), g.blocks_per_request,
                                      g.local_heads, g.kv_heads, g.block_tokens, scale, ptr<float>(dscales),
                                      mptr<float>(blk8), split ? mptr<float>(part) : nullptr, st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    const std::vector<float> a = down<float>(rowb, n), b = down<float>(blkb, n);
    const std::vector<float> a8 = down<float>(row8, n), b8 = down<float>(blk8, n);
    size_t dbf = 0, d8 = 0, dx = 0;
    for (size_t i = 0; i < n; ++i) { dbf += a[i] != b[i]; d8 += a8[i] != b8[i]; dx += a[i] * vs != a8[i]; }
    std::printf("[ .. ] long context %s: block vs row bf16 %zu/%zu differ, fp8 %zu, fp8 row vs bf16 row x vs %zu\n",
                split ? "split" : "unsplit", dbf, n, d8, dx);
    if (dbf) {
      for (int r = 0; r < rows; ++r) {
        size_t c = 0;
        for (int i = 0; i < qw; ++i) c += a[static_cast<size_t>(r) * qw + i] != b[static_cast<size_t>(r) * qw + i];
        if (c) std::printf("[ .. ]   row %d (pos %lld): %zu differ; row %g block %g\n", r, static_cast<long long>(pos[static_cast<size_t>(r)]), c,
                           a[static_cast<size_t>(r) * qw], b[static_cast<size_t>(r) * qw]);
      }
    }
    require(dbf == 0, "long context: block form bitwise the row form (bf16)");
    require(d8 == 0, "long context: fp8 block form bitwise the fp8 row form");
    require(dx == 0, "long context: fp8 row form bitwise the bf16 row form x v_scale");
  }
}


// The staged prefill forms at a long context (100 000 keys, 64-token pages,
// the ring wrapping a thousand times): fp8 bitwise the bf16 form on the
// widened planes for 32 rows at the end of the request.
DGPP_TEST(full_attention_fp8_prefill_long_context_bitwise_the_bf16_chain) {
  Geo g;
  g.local_heads = 6;
  g.kv_heads = 1;
  g.block_tokens = 64;
  g.blocks_per_request = 1600;  // 102 400 slots per request
  g.max_requests = 1;
  cudaStream_t st = test_stream();
  const int len = 100000;
  const size_t slots = static_cast<size_t>(g.max_requests) * g.slots();
  const float ks = 0.0322265625f, vs = 0.25390625f;
  std::vector<uint8_t> k8(slots * g.width(), 0x7E), v8(slots * g.width(), 0x7E);
  {
    const std::vector<uint16_t> k = random_bf16_normal(200, static_cast<int64_t>(len) * g.width(), 1.0f);
    const std::vector<uint16_t> v = random_bf16_normal(201, static_cast<int64_t>(len) * g.width(), 1.0f);
    for (size_t i = 0; i < static_cast<size_t>(len) * g.width(); ++i) {
      k8[i] = dgpp::float_to_fp8_e4m3_bits(dgpp::bf16_bits_to_float(k[i]) / ks);
      v8[i] = dgpp::float_to_fp8_e4m3_bits(dgpp::bf16_bits_to_float(v[i]) / vs);
    }
  }
  std::vector<uint16_t> kw(slots * g.width()), vw(slots * g.width());
  for (size_t i = 0; i < kw.size(); ++i) {
    kw[i] = dgpp::float_to_bf16_bits(dgpp::fp8_e4m3_bits_to_float(k8[i]));
    vw[i] = dgpp::float_to_bf16_bits(dgpp::fp8_e4m3_bits_to_float(v8[i]));
  }
  DevBuf dk8 = up(k8), dv8 = up(v8), dkw = up(kw), dvw = up(vw), dtbl = up(tables(g));
  const std::vector<float> scales = {ks, vs};
  DevBuf dscales = up(scales);
  const int rows = 32;
  std::vector<int64_t> pos(static_cast<size_t>(rows));
  for (int r = 0; r < rows; ++r) pos[static_cast<size_t>(r)] = len - rows + r;
  const int qw = g.local_heads * D;
  const std::vector<uint16_t> q = random_bf16_normal(210, static_cast<int64_t>(rows) * qw, 1.0f);
  DevBuf dq = up(q), dpos = up(pos);
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const size_t n = static_cast<size_t>(rows) * qw;
  const dgpp::FullAttnStageMaps maps =
      dgpp::full_attn_stage_maps(ptr<uint16_t>(dkw), ptr<uint16_t>(dvw), static_cast<int64_t>(slots), g.kv_heads);
  const dgpp::FullAttnStageMaps maps8 =
      dgpp::full_attn_stage_maps_fp8(ptr<uint8_t>(dk8), ptr<uint8_t>(dv8), static_cast<int64_t>(slots), g.kv_heads);
  DevBuf ref(n * 4), got(n * 4);
  DGPP_CUDA_OK(cudaMemset(ref.p, 0x7F, n * 4));
  DGPP_CUDA_OK(cudaMemset(got.p, 0x7F, n * 4));
  dgpp::full_attn_prefill_tma(maps, 0, ptr<uint16_t>(dq), qw, ptr<int32_t>(dtbl), ptr<int64_t>(dpos), rows,
                              g.local_heads, g.kv_heads, g.block_tokens, scale * ks, mptr<float>(ref), st);
  dgpp::full_attn_prefill_tma_fp8(maps8, 0, ptr<uint16_t>(dq), qw, ptr<int32_t>(dtbl), ptr<int64_t>(dpos), rows,
                                  g.local_heads, g.kv_heads, g.block_tokens, scale, ptr<float>(dscales),
                                  mptr<float>(got), st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  std::vector<float> want = down<float>(ref, n);
  for (float& x : want) x = x * vs;
  const std::vector<float> have = down<float>(got, n);
  size_t diff = 0;
  for (size_t i = 0; i < n; ++i) diff += have[i] != want[i];
  std::printf("[ .. ] fp8 staged prefill at 100K keys: %zu/%zu differ; out[0] %g vs %g\n", diff, n, have[0], want[0]);
  require(diff == 0, "fp8 staged prefill at a long context bitwise the bf16 chain");
}

// The fp8 compute form (engine.prefill_attention = fp8): Q quantized per
// row to e4m3 (amax / 448), S = Q8 K8^T, P = exp2(s - m + 8) rounded to
// e4m3, O += P8 V8 on the fp8 tensor cores, the denominator the unrounded
// P sum, the output times v_scale. Gate: a host model of exactly that
// recipe (double arithmetic over the same codes, 32-key tiles, the running
// max) on every fifth row of a 1990-row prefill over 64-token pages — each
// layout the kernel relies on (the Q/K dim order, P's key order against
// V's transposed fragments, the even/odd dim n-tiles) would scramble it by
// O(1); what remains is the f32 accumulation order and the exp2
// approximation, which move a P across an e4m3 rounding boundary a few
// times per million. Reported beside it: the distance to the bf16 chain
// (the storage form), the mode's deviation by design on this data.
DGPP_TEST(full_attention_fp8_compute_form_is_its_recipe) {
  Geo g;
  g.local_heads = 12;  // two KV groups of six
  g.kv_heads = 2;
  g.block_tokens = 64;  // the serving pool's page
  g.blocks_per_request = 32;  // 2048 slots per request
  cudaStream_t st = test_stream();
  const float ks = 0.0390625f, vs = 0.1416015625f;
  const int lens[2] = {2000, 100};
  const size_t slots = static_cast<size_t>(g.max_requests) * g.slots();
  std::vector<uint8_t> k8(slots * g.width(), 0x7E), v8(slots * g.width(), 0x7E);
  for (int q = 0; q < 2; ++q) {
    const std::vector<uint16_t> k = random_bf16_normal(230 + q, static_cast<int64_t>(lens[q]) * g.width(), 1.0f);
    const std::vector<uint16_t> v = random_bf16_normal(240 + q, static_cast<int64_t>(lens[q]) * g.width(), 1.0f);
    for (int t = 0; t < lens[q]; ++t)
      for (int e = 0; e < g.width(); ++e) {
        const size_t at = static_cast<size_t>(phys(g, q, t)) * g.width() + e;
        k8[at] = dgpp::float_to_fp8_e4m3_bits(dgpp::bf16_bits_to_float(k[static_cast<size_t>(t) * g.width() + e]) / ks);
        v8[at] = dgpp::float_to_fp8_e4m3_bits(dgpp::bf16_bits_to_float(v[static_cast<size_t>(t) * g.width() + e]) / vs);
      }
  }
  std::vector<uint16_t> kw(k8.size()), vw(v8.size());
  for (size_t i = 0; i < kw.size(); ++i) {
    kw[i] = dgpp::float_to_bf16_bits(dgpp::fp8_e4m3_bits_to_float(k8[i]));
    vw[i] = dgpp::float_to_bf16_bits(dgpp::fp8_e4m3_bits_to_float(v8[i]));
  }
  DevBuf dk8 = up(k8), dv8 = up(v8), dkw = up(kw), dvw = up(vw), dtbl = up(tables(g));
  const std::vector<float> scales = {ks, vs};
  DevBuf dscales = up(scales);
  const int rows = 1990;  // the last CTA carries six rows and ten padding rows
  const int qw = g.local_heads * D;
  const int group = g.local_heads / g.kv_heads;
  std::vector<int64_t> pos(static_cast<size_t>(rows));
  for (int r = 0; r < rows; ++r) pos[static_cast<size_t>(r)] = r;
  // Q rows at a spread of magnitudes: the per-row amax is part of the recipe.
  std::vector<uint16_t> q = random_bf16_normal(250, static_cast<int64_t>(rows) * qw, 1.0f);
  for (int r = 0; r < rows; ++r)
    for (int h = 0; h < g.local_heads; ++h) {
      const float mult = 0.25f * static_cast<float>(1 + ((r * 7 + h) % 13));
      for (int d = 0; d < D; ++d) {
        uint16_t& x = q[(static_cast<size_t>(r) * g.local_heads + h) * D + d];
        x = dgpp::float_to_bf16_bits(dgpp::bf16_bits_to_float(x) * mult);
      }
    }
  DevBuf dq = up(q), dpos = up(pos);
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const size_t n = static_cast<size_t>(rows) * qw;
  const dgpp::FullAttnStageMaps maps =
      dgpp::full_attn_stage_maps(ptr<uint16_t>(dkw), ptr<uint16_t>(dvw), static_cast<int64_t>(slots), g.kv_heads);
  const dgpp::FullAttnStageMaps maps8 =
      dgpp::full_attn_stage_maps_fp8(ptr<uint8_t>(dk8), ptr<uint8_t>(dv8), static_cast<int64_t>(slots), g.kv_heads);
  DevBuf ref(n * 4), got(n * 4);
  DGPP_CUDA_OK(cudaMemset(ref.p, 0x7F, n * 4));
  DGPP_CUDA_OK(cudaMemset(got.p, 0x7F, n * 4));
  dgpp::full_attn_prefill_tma(maps, 0, ptr<uint16_t>(dq), qw, ptr<int32_t>(dtbl), ptr<int64_t>(dpos), rows,
                              g.local_heads, g.kv_heads, g.block_tokens, scale * ks, mptr<float>(ref), st);
  dgpp::full_attn_prefill_tma_fp8_compute(maps8, 0, ptr<uint16_t>(dq), qw, ptr<int32_t>(dtbl), ptr<int64_t>(dpos),
                                          rows, g.local_heads, g.kv_heads, g.block_tokens, scale,
                                          ptr<float>(dscales), mptr<float>(got), st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<float> have = down<float>(got, n);
  std::vector<float> chain = down<float>(ref, n);
  for (float& x : chain) x *= vs;
  for (size_t i = 0; i < n; ++i) require(std::isfinite(have[i]), "fp8 compute form: finite output");

  // The recipe on the host, the kernel's f32 steps where they decide a code.
  double num_r = 0, den_r = 0, worst_r = 0, num_c = 0, den_c = 0;
  size_t compared = 0;
  std::vector<double> acc(D), s(kTileTok32);
  std::vector<float> q8(D);
  for (int r = 0; r < rows; r += 5) {
    for (int h = 0; h < g.local_heads; ++h) {
      const int kvh = h / group;
      const uint16_t* qr = q.data() + (static_cast<size_t>(r) * g.local_heads + h) * D;
      float am = 0.f;
      for (int d = 0; d < D; ++d) am = std::max(am, std::fabs(dgpp::bf16_bits_to_float(qr[d])));
      const float iq = am > 0.f ? 448.f / am : 0.f;
      const float qs = am > 0.f ? am * (1.f / 448.f) : 0.f;
      for (int d = 0; d < D; ++d) {
        const float x = std::max(-448.f, std::min(448.f, dgpp::bf16_bits_to_float(qr[d]) * iq));
        q8[static_cast<size_t>(d)] = dgpp::fp8_e4m3_bits_to_float(dgpp::float_to_fp8_e4m3_bits(x));
      }
      const float sl2 = scale * ks * qs * 1.4426950408889634f;
      double m = -INFINITY, l = 0;
      std::fill(acc.begin(), acc.end(), 0.0);
      for (int64_t base = 0; base <= r; base += kTileTok32) {
        double mx = -INFINITY;
        for (int j = 0; j < kTileTok32; ++j) {
          const int64_t key = base + j;
          if (key > r) {
            s[static_cast<size_t>(j)] = -INFINITY;
            continue;
          }
          const uint8_t* kr = k8.data() + static_cast<size_t>(phys(g, 0, key)) * g.width() + kvh * D;
          double dot = 0;
          for (int d = 0; d < D; ++d) dot += static_cast<double>(q8[static_cast<size_t>(d)]) * dgpp::fp8_e4m3_bits_to_float(kr[d]);
          s[static_cast<size_t>(j)] = dot * sl2;
          mx = std::max(mx, s[static_cast<size_t>(j)]);
        }
        const double nm = std::max(m, mx);
        const double a = m == -INFINITY ? 0.0 : std::exp2(m - nm);
        for (int d = 0; d < D; ++d) acc[static_cast<size_t>(d)] *= a;
        l *= a;
        m = nm;
        for (int j = 0; j < kTileTok32; ++j) {
          if (s[static_cast<size_t>(j)] == -INFINITY) continue;
          const double p = std::exp2(s[static_cast<size_t>(j)] - m + 8.0);
          l += p;
          const float p8 = dgpp::fp8_e4m3_bits_to_float(dgpp::float_to_fp8_e4m3_bits(static_cast<float>(p)));
          const uint8_t* vr = v8.data() + static_cast<size_t>(phys(g, 0, base + j)) * g.width() + kvh * D;
          for (int d = 0; d < D; ++d) acc[static_cast<size_t>(d)] += static_cast<double>(p8) * dgpp::fp8_e4m3_bits_to_float(vr[d]);
        }
      }
      for (int d = 0; d < D; ++d) {
        const double want = acc[static_cast<size_t>(d)] / l * vs;
        const size_t i = (static_cast<size_t>(r) * g.local_heads + h) * D + d;
        const double dr = have[i] - want, dc = have[i] - chain[i];
        num_r += dr * dr;
        den_r += want * want;
        worst_r = std::max(worst_r, std::fabs(dr));
        num_c += dc * dc;
        den_c += static_cast<double>(chain[i]) * chain[i];
        ++compared;
      }
    }
  }
  const double rel_r = std::sqrt(num_r / den_r), rel_c = std::sqrt(num_c / den_c);
  std::printf("[ .. ] fp8 compute form vs its recipe on %zu elements: rel rms %.3g (worst |d| %.3g); vs the bf16 chain %.3g\n",
              compared, rel_r, worst_r, rel_c);
  require(rel_r < 1e-3, "fp8 compute form: the recipe within 1e-3 relative rms");
  require(rel_c < 0.1, "fp8 compute form: within 10 % of the bf16 chain on this data");
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;
  return dgpp::test::run_all();
}
