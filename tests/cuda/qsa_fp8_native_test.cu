#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/latent_format.hpp"
#include "kernels/qsa.hpp"

using namespace dgpp;
using namespace dgpp::kda_test;
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class T> DevBuf upload(const std::vector<T>& v) {
  DevBuf b(v.size() * sizeof(T)); b.upload(v.data(), b.bytes); return b;
}
template<class T> std::vector<T> download(const DevBuf& b, size_t count) {
  std::vector<T> out(count); b.download(out.data(), count * sizeof(T)); return out;
}
constexpr int D = 256, B = 64, blocks = 36, requests = 2, stride = 2051;
constexpr int slots = B * blocks * requests;
constexpr float scale = 1.f / 16.f, log2e = 1.4426950408889634f;

struct Case {
  int heads, kv, qstride;
  std::vector<int32_t> counts{0, 1, 7, 15, 16, 17, 31, 32, 33, 127, 257, 2051};
  std::vector<int32_t> req, table, tokens;
  std::vector<uint16_t> q;
  std::vector<uint8_t> k, v;
  std::vector<float> ks, vs;
  Case(int heads_, int kv_) : heads(heads_), kv(kv_), qstride(heads * D + 8) {
    const int rows = counts.size();
    req.resize(rows); table.resize(blocks * requests); tokens.resize(rows * stride, -1);
    for (int i = 0; i < blocks * requests; ++i) table[i] = (i * 17 + 3) % (blocks * requests);
    for (int r = 0; r < rows; ++r) {
      req[r] = r % requests;
      for (int i = 0; i < counts[r]; ++i) tokens[r * stride + i] = (i * 37 + r) % (blocks * B);
    }
    q = random_bf16_normal(250, int64_t(rows) * qstride, 1.f);
    std::fill(q.begin() + qstride, q.begin() + 2 * qstride, uint16_t(0));
    for (int d = 0; d < heads * D; ++d)
      q[2 * qstride + d] = float_to_bf16_bits(bf16_bits_to_float(q[2 * qstride + d]) * 16.f);
    auto kr = random_bf16_normal(260, int64_t(slots) * kv * D, 1.f);
    auto vr = random_bf16_normal(270, kr.size(), 1.f);
    k.resize(kr.size()); v.resize(vr.size()); ks.resize(slots * kv); vs.resize(slots * kv);
    for (int row = 0; row < slots * kv; ++row) {
      const size_t off = size_t(row) * D;
      float km = 0, vm = 0;
      for (int d = 0; d < D; ++d) {
        km = std::max(km, std::fabs(bf16_bits_to_float(kr[off + d])));
        vm = std::max(vm, std::fabs(bf16_bits_to_float(vr[off + d])));
      }
      const auto sk = latent_fp8_row_scale(km), sv = latent_fp8_row_scale(vm);
      ks[row] = row % 19 == 0 ? 0.f : sk.scale;
      // Zero rows and V scales varying by 4096x catch incorrect scale folding.
      vs[row] = row % 23 == 0 ? 0.f : std::ldexp(sv.scale, row % 13 - 6);
      for (int d = 0; d < D; ++d) {
        k[off + d] = latent_fp8_encode(bf16_bits_to_float(kr[off + d]), sk.inv);
        v[off + d] = latent_fp8_encode(bf16_bits_to_float(vr[off + d]), sv.inv);
      }
    }
  }
  int physical(int row, int i) const {
    const int tok = tokens[row * stride + i];
    return table[req[row] * blocks + tok / B] * B + tok % B;
  }
};

// Independent host recipe: accumulate scaled V in double, while the GPU
// keeps its accumulator in the current P scale's units for the next MMA.
std::vector<float> recipe(const Case& c, int splits) {
  std::vector<float> out(c.counts.size() * c.heads * D, 0.f);
  for (int r = 0; r < int(c.counts.size()); ++r)
    for (int h = 0; h < c.heads; ++h) {
      if (c.counts[r] == 0) continue;
      const int kh = h / (c.heads / c.kv), chunk = (c.counts[r] + splits - 1) / splits;
      float amax = 0;
      for (int d = 0; d < D; ++d) amax = std::max(amax, std::fabs(bf16_bits_to_float(c.q[r * c.qstride + h * D + d])));
      const auto qs = latent_fp8_row_scale(amax);
      std::vector<float> q8(D);
      for (int d = 0; d < D; ++d)
        q8[d] = fp8_e4m3_bits_to_float(latent_fp8_encode(bf16_bits_to_float(c.q[r * c.qstride + h * D + d]), qs.inv));
      std::vector<double> acc(splits * D, 0), mass(splits, 0);
      std::vector<float> maximum(splits, -INFINITY);
      for (int s = 0; s < splits; ++s) {
        const int end = std::min(c.counts[r], (s + 1) * chunk);
        for (int base = s * chunk; base < end; base += 32) {
          const int n = std::min(32, end - base);
          float scores[32], weighted[32], tilemax = -INFINITY;
          for (int j = 0; j < n; ++j) {
            const int kr = c.physical(r, base + j) * c.kv + kh;
            double dot = 0;
            for (int d = 0; d < D; ++d) dot += double(q8[d]) * fp8_e4m3_bits_to_float(c.k[size_t(kr) * D + d]);
            scores[j] = float(dot) * (qs.scale * scale * log2e) * c.ks[kr];
            tilemax = std::max(tilemax, scores[j]);
          }
          const float next = std::max(maximum[s], tilemax), rescale = std::exp2(maximum[s] - next);
          mass[s] *= rescale;
          for (int d = 0; d < D; ++d) acc[s * D + d] *= rescale;
          maximum[s] = next;
          float pmax = 0;
          for (int j = 0; j < n; ++j) {
            const float prob = std::exp2(scores[j] - next + 8.f);
            mass[s] += prob;
            weighted[j] = prob * c.vs[c.physical(r, base + j) * c.kv + kh];
            pmax = std::max(pmax, weighted[j]);
          }
          const float ps = pmax > 0 ? pmax * (1.f / 256.f) : 1.f;
          for (int j = 0; j < n; ++j) {
            const float p8 = fp8_e4m3_bits_to_float(float_to_fp8_e4m3_bits(weighted[j] * (1.f / ps)));
            const size_t off = size_t(c.physical(r, base + j) * c.kv + kh) * D;
            for (int d = 0; d < D; ++d)
              acc[s * D + d] += double(p8) * ps * fp8_e4m3_bits_to_float(c.v[off + d]);
          }
        }
      }
      const float mx = *std::max_element(maximum.begin(), maximum.end());
      double den = 0;
      for (int s = 0; s < splits; ++s) den += mass[s] * std::exp2(maximum[s] - mx);
      for (int d = 0; d < D; ++d) {
        double num = 0;
        for (int s = 0; s < splits; ++s) num += acc[s * D + d] * std::exp2(maximum[s] - mx);
        out[(size_t(r) * c.heads + h) * D + d] = float(num / den);
      }
    }
  return out;
}
void check(const Case& c, const std::vector<float>& actual, const std::vector<float>& expected) {
  double num = 0, den = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    require(std::isfinite(actual[i]), "native attention returned nonfinite output");
    if (c.counts[i / (c.heads * D)] == 0) require(actual[i] == 0, "empty list must be zero");
    const double e = double(actual[i]) - expected[i];
    num += e * e; den += double(expected[i]) * expected[i];
  }
  const double rel = std::sqrt(num / std::max(den, 1e-30));
  std::printf("[native fp8] heads=%d kv=%d recipe rel_l2=%.6g\n", c.heads, c.kv, rel);
  require(rel < 1e-3, "native attention differs from the independent FP8 recipe");
}
}  // namespace

DGPP_TEST(qsa_native_fp8_matches_recipe_and_graph_replay) {
  const auto stream = test_stream();
  for (auto geometry : {std::pair{24, 2}, std::pair{18, 1}, std::pair{1, 1}}) {
    Case c(geometry.first, geometry.second);
    auto q = upload(c.q), k = upload(c.k), v = upload(c.v), ks = upload(c.ks), vs = upload(c.vs);
    auto req = upload(c.req), table = upload(c.table), tokens = upload(c.tokens), counts = upload(c.counts);
    const int rows = c.counts.size();
    const size_t size = size_t(rows) * c.heads * D;
    DevBuf out(size * 4);
    for (int splits : {1, 3, 8}) {
      const size_t part = size_t(rows) * splits * c.heads;
      DevBuf m(part * 4), l(part * 4), a(part * D * 4);
      auto launch = [&] {
        qsa_attn_fp8(q.as<uint16_t>(), c.qstride, k.as<uint8_t>(), v.as<uint8_t>(),
            ks.as<float>(), vs.as<float>(), req.as<int32_t>(), tokens.as<int32_t>(), stride,
            counts.as<int32_t>(), rows, c.heads, c.kv, B, table.as<int32_t>(), blocks, scale, splits,
            m.as<float>(), l.as<float>(), a.as<float>(), out.as<float>(), stream);
      };
      launch();
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      check(c, download<float>(out, size), recipe(c, splits));
      cudaGraph_t graph;
      cudaGraphExec_t exec;
      DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      launch();
      DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
      DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
      std::swap(c.counts[0], c.counts[11]); // changed list lengths and request IDs
      for (int r = 0; r < rows; ++r) {
        c.req[r] ^= 1;
        for (int i = 0; i < c.counts[r]; ++i) c.tokens[r * stride + i] = (i * 37 + r) % (blocks * B);
      }
      counts.upload(c.counts.data(), counts.bytes);
      req.upload(c.req.data(), req.bytes);
      tokens.upload(c.tokens.data(), tokens.bytes);
      DGPP_CUDA_OK(cudaMemsetAsync(out.p, 0xff, out.bytes, stream));
      DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      check(c, download<float>(out, size), recipe(c, splits));
      DGPP_CUDA_OK(cudaGraphExecDestroy(exec));
      DGPP_CUDA_OK(cudaGraphDestroy(graph));
    }
  }
}

DGPP_TEST(qsa_fp8_append_covers_every_head_and_skips_padding) {
  const auto stream = test_stream();
  for (int heads : {1, 2, 9, 16}) {
    constexpr int rows = 3, block = 4;
    const int width = heads * D, row_stride = width + 8;
    auto kh = random_bf16_normal(300, rows * row_stride, 1.f), vh = random_bf16_normal(301, rows * row_stride, 1.f);
    auto k = upload(kh), v = upload(vh);
    auto req = upload(std::vector<int32_t>{1, 0, 1});
    auto pos = upload(std::vector<int64_t>{1, 2, -1});
    auto table = upload(std::vector<int32_t>{1, 0});
    auto kc = upload(std::vector<uint8_t>(2 * block * width, 0x55)), vc = upload(std::vector<uint8_t>(2 * block * width, 0x55));
    auto ks = upload(std::vector<float>(2 * block * heads, -13.f)), vs = upload(std::vector<float>(2 * block * heads, -13.f));
    qsa_kv_append(k.as<uint16_t>(), row_stride, v.as<uint16_t>(), row_stride, req.as<int32_t>(),
        pos.as<int64_t>(), rows, table.as<int32_t>(), 1, block, heads, D,
        reinterpret_cast<uint16_t*>(kc.p), reinterpret_cast<uint16_t*>(vc.p), ks.as<float>(), vs.as<float>(), stream);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    for (int which = 0; which < 2; ++which) {
      const auto codes = download<uint8_t>(which ? vc : kc, kc.bytes);
      const auto scales = download<float>(which ? vs : ks, ks.bytes / 4);
      const auto& source = which ? vh : kh;
      for (int slot = 0; slot < 2 * block; ++slot)
        for (int h = 0; h < heads; ++h) {
          const int row = slot == 1 ? 0 : slot == 6 ? 1 : -1;
          if (row < 0) {
            require(scales[slot * heads + h] == -13.f, "append overwrote unused scale");
            for (int d = 0; d < D; ++d)
              require(codes[slot * width + h * D + d] == 0x55, "append overwrote unused codes");
            continue;
          }
          float amax = 0;
          for (int d = 0; d < D; ++d) amax = std::max(amax, std::fabs(bf16_bits_to_float(source[row * row_stride + h * D + d])));
          const auto s = latent_fp8_row_scale(amax);
          require(scales[slot * heads + h] == s.scale, "append scale differs from host codec");
          for (int d = 0; d < D; ++d)
            require(codes[slot * width + h * D + d] ==
                latent_fp8_encode(bf16_bits_to_float(source[row * row_stride + h * D + d]), s.inv),
                "append code differs from host codec");
        }
    }
  }
}

int main() { return dgpp::test::run_all(); }
