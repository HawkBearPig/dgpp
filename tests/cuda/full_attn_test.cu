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

using namespace dgpp::kda_test;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

constexpr int D = 256;
constexpr int kTileTok = 16;

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
        acc = anew;
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

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;
  return dgpp::test::run_all();
}
