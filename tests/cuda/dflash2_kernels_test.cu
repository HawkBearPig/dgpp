// The DFlash2 drafter kernels against host references: the 2-tap dynamic
// grouped conv (block-border gating, side/tap slices), the standard
// norm + rotate-half rope, the block attention (windowed causal context
// + the bidirectional block, paged), the candidate top-K (descending,
// ties to the lower id) and the selector walk (predecessor codes chained
// through slots, the anchor at step 0, greedy argmax).
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/dflash2.hpp"

using namespace dgpp::kda_test;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

auto b16 = [](DevBuf& b) { return static_cast<uint16_t*>(b.p); };
auto f32 = [](DevBuf& b) { return static_cast<float*>(b.p); };
auto i32 = [](DevBuf& b) { return static_cast<int32_t*>(b.p); };
auto cb16 = [](DevBuf& b) { return static_cast<const uint16_t*>(b.p); };
auto cf32 = [](DevBuf& b) { return static_cast<const float*>(b.p); };
auto ci32 = [](DevBuf& b) { return static_cast<const int32_t*>(b.p); };
auto ci64 = [](DevBuf& b) { return static_cast<const int64_t*>(b.p); };

std::vector<float> to_f(const std::vector<uint16_t>& x) {
  std::vector<float> v(x.size());
  for (size_t i = 0; i < x.size(); ++i) v[i] = dgpp::bf16_bits_to_float(x[i]);
  return v;
}

// ---- the grouped conv ------------------------------------------------------

DGPP_TEST(dflash2_grouped_conv_matches_the_reference) {
  cudaStream_t s = test_stream();
  // group_size (channels per group) != groups per tap: hidden 64 in
  // groups of 4 -> 16 groups; tap 1 starts at dr[16], not dr[4]. A test
  // with group_size == groups passes either way and proves nothing.
  const int rows = 16, block_rows = 8, hidden = 64, group_size = 4, G = hidden / group_size;
  auto xr = random_bf16_normal(11, static_cast<int64_t>(rows) * hidden, 1.0f);
  auto base = random_bf16_normal(12, 2ull * 2 * hidden, 0.5f);      // [sides][taps][hidden]
  auto delta = random_bf16_normal(13, static_cast<int64_t>(rows) * 2 * 2 * G, 0.2f);  // [rows][sides][taps][G]
  const int64_t ds = 2 * 2 * G;
  for (int side : {0, 1}) {
    DevBuf dx(xr.size() * 2), db(base.size() * 2), dd(delta.size() * 2), dy(xr.size() * 2);
    dx.upload(xr.data(), xr.size() * 2);
    db.upload(base.data(), base.size() * 2);
    dd.upload(delta.data(), delta.size() * 2);
    dgpp::dflash2_grouped_conv_bf16(cb16(dx), cb16(dd) + side * 2 * G, cb16(db) + side * 2 * hidden,
                                    b16(dy), rows, block_rows, hidden, 2, group_size, ds, s);
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
    std::vector<uint16_t> y(xr.size());
    dy.download(y.data(), y.size() * 2);
    const auto xf = to_f(xr);
    const auto bf = to_f(base), df = to_f(delta);
    std::vector<uint16_t> want(xr.size());
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < hidden; ++c) {
        const int g = c / group_size;
        float acc = (bf[side * 2 * hidden + c] + df[(static_cast<int64_t>(r) * ds + side * 2 * G + g)]) *
                    xf[static_cast<int64_t>(r) * hidden + c];
        if ((r % block_rows) >= 1)
          acc += (bf[side * 2 * hidden + hidden + c] +
                  df[static_cast<int64_t>(r) * ds + side * 2 * G + G + g]) *
                 xf[static_cast<int64_t>(r - 1) * hidden + c];
        want[static_cast<int64_t>(r) * hidden + c] = dgpp::float_to_bf16_bits(acc);
      }
    require_bf16("grouped conv side " + std::to_string(side), compare_bf16(y, want, 0), 0.0, 0.0);
  }
}

// ---- norm + rope -----------------------------------------------------------

DGPP_TEST(dflash2_norm_rope_matches_the_reference) {
  cudaStream_t s = test_stream();
  const int rows = 3, heads = 2, dim = 128;
  const double theta = 1e7;
  std::vector<float> inv(dim / 2);
  for (int i = 0; i < dim / 2; ++i)
    inv[i] = static_cast<float>(std::pow(theta, -2.0 * i / dim));
  auto x = random_bf16_normal(21, static_cast<int64_t>(rows) * heads * dim, 1.0f);
  auto w = random_bf16_uniform(22, dim, 0.4f);
  std::vector<int64_t> pos = {0, 17, 262143};
  DevBuf dx(x.size() * 2), dw(w.size() * 2), dy(x.size() * 2), di(inv.size() * 4), dp(pos.size() * 8);
  dx.upload(x.data(), x.size() * 2);
  dw.upload(w.data(), w.size() * 2);
  di.upload(inv.data(), inv.size() * 4);
  dp.upload(pos.data(), pos.size() * 8);
  dgpp::dflash2_norm_rope_bf16(cb16(dx), heads * dim, cb16(dw), ci64(dp), cf32(di), b16(dy),
                               heads * dim, rows, heads, dim, 1e-6f, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  std::vector<uint16_t> y(x.size());
  dy.download(y.data(), y.size() * 2);
  const auto xf = to_f(x);
  std::vector<uint16_t> want(x.size());
  for (int r = 0; r < rows; ++r)
    for (int h = 0; h < heads; ++h) {
      const int64_t b = (static_cast<int64_t>(r) * heads + h) * dim;
      float ssq = 0.0f;
      for (int d = 0; d < dim; ++d) ssq += xf[b + d] * xf[b + d];
      const float rstd = 1.0f / std::sqrt(ssq / dim + 1e-6f);
      float n[dim];
      for (int d = 0; d < dim; ++d)
        n[d] = dgpp::bf16_bits_to_float(
            dgpp::float_to_bf16_bits(xf[b + d] * rstd * dgpp::bf16_bits_to_float(w[d])));
      for (int d = 0; d < dim / 2; ++d) {
        const float ang = static_cast<float>(pos[r]) * inv[d];
        const float c = dgpp::bf16_bits_to_float(dgpp::float_to_bf16_bits(std::cos(ang)));
        const float si = dgpp::bf16_bits_to_float(dgpp::float_to_bf16_bits(std::sin(ang)));
        const auto rbf = [](float v) { return dgpp::bf16_bits_to_float(dgpp::float_to_bf16_bits(v)); };
        const float a = rbf(rbf(n[d] * c) + rbf(-n[d + dim / 2] * si));
        const float b2 = rbf(rbf(n[d + dim / 2] * c) + rbf(n[d] * si));
        want[b + d] = dgpp::float_to_bf16_bits(a);
        want[b + d + dim / 2] = dgpp::float_to_bf16_bits(b2);
      }
    }
  // The device sums squares with shuffles: one fp32 ulp on rstd moves a
  // hair of the outputs by one bf16 ulp; and the product-rounding scheme
  // (round each product, then the sum) differs from the fused host
  // reference by the same amount.
  require_bf16("norm_rope", compare_bf16(y, want, 2), 1e-2, 0.02);
}

// ---- top-K -------------------------------------------------------------------

DGPP_TEST(dflash2_topk_picks_the_sorted_candidates) {
  cudaStream_t s = test_stream();
  const int rows = 3, k = 16;
  const int64_t V = 10007;
  std::mt19937 rng(7);
  std::vector<float> lg(static_cast<size_t>(rows) * V);
  for (auto& v : lg) v = std::uniform_real_distribution<float>(-8.0f, 8.0f)(rng);
  // Two forced ties at the very top of row 1.
  lg[V + 5] = 100.0f;
  lg[V + 9] = 100.0f;
  DevBuf dl(lg.size() * 4), did(rows * k * 4), dsc(rows * k * 4);
  dl.upload(lg.data(), lg.size() * 4);
  dgpp::dflash2_topk_f32(cf32(dl), i32(did), f32(dsc), V, rows, k, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  std::vector<int32_t> ids(rows * k);
  std::vector<float> sc(rows * k);
  did.download(ids.data(), ids.size() * 4);
  dsc.download(sc.data(), sc.size() * 4);
  for (int r = 0; r < rows; ++r) {
    std::vector<std::pair<float, int32_t>> all;
    for (int64_t v = 0; v < V; ++v) all.push_back({lg[r * V + v], static_cast<int32_t>(v)});
    std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) {
      return a.first > b.first || (a.first == b.first && a.second < b.second);
    });
    for (int j = 0; j < k; ++j) {
      if (sc[r * k + j] != all[j].first)
        throw std::runtime_error("topk score row " + std::to_string(r) + " slot " + std::to_string(j) +
                                 ": got " + std::to_string(sc[r * k + j]) + " want " +
                                 std::to_string(all[j].first));
      if (ids[r * k + j] != all[j].second)
        throw std::runtime_error("topk id row " + std::to_string(r) + " slot " + std::to_string(j) +
                                 ": got " + std::to_string(ids[r * k + j]) + " want " +
                                 std::to_string(all[j].second));
    }
  }
}

// ---- the selector -------------------------------------------------------------

DGPP_TEST(dflash2_selector_walks_the_reference_scores) {
  cudaStream_t s = test_stream();
  // Production shapes (the released checkpoint's selector): k=16, rank=256.
  // A smaller shape runs first as a fast path through the same code.
  for (const auto [steps, k, rank, vocab] :
       {std::tuple{2, 4, 16, 97}, std::tuple{7, 16, 256, 1009}}) {
  const int32_t anchor = 5;
  auto pred = random_bf16_normal(31, static_cast<int64_t>(vocab) * rank, 0.5f);
  auto succ = random_bf16_normal(32, static_cast<int64_t>(vocab) * rank, 0.5f);
  std::mt19937 rng(33);
  std::vector<int32_t> ids(steps * k);
  for (auto& t : ids) t = static_cast<int32_t>(rng() % vocab);
  std::vector<float> unary(steps * k), hidden(steps * rank);
  for (auto& v : unary) v = std::uniform_real_distribution<float>(-4.0f, 4.0f)(rng);
  for (auto& v : hidden) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);

  DevBuf dp(pred.size() * 2), ds(succ.size() * 2), di(ids.size() * 4), du(unary.size() * 4),
      dh(hidden.size() * 4), dt(steps * 4);
  dp.upload(pred.data(), pred.size() * 2);
  ds.upload(succ.data(), succ.size() * 2);
  di.upload(ids.data(), ids.size() * 4);
  du.upload(unary.data(), unary.size() * 4);
  dh.upload(hidden.data(), hidden.size() * 4);
  dgpp::dflash2_selector_walk(ci32(di), cf32(du), cf32(dh), cb16(dp), cb16(ds), anchor, i32(dt),
                              steps, k, rank, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  std::vector<int32_t> got(steps);
  dt.download(got.data(), got.size() * 4);

  // The reference: scores[l][p][c] = unary[l][p] + <pred(id(l-1,p)) *
  // hidden[l], succ(id(l,c))>, walked greedily from slot 0.
  const auto pf = to_f(pred), sf = to_f(succ);
  int prev = 0;
  for (int l = 0; l < steps; ++l) {
    float best = -1e30f;
    int besti = 0;
    for (int c = 0; c < k; ++c) {
      const int32_t pid = l == 0 ? anchor : ids[static_cast<int64_t>(l - 1) * k + prev];
      const int32_t sid = ids[static_cast<int64_t>(l) * k + c];
      float dot = 0.0f;
      for (int r = 0; r < rank; ++r)
        dot += pf[static_cast<int64_t>(pid) * rank + r] * hidden[l * rank + r] *
               sf[static_cast<int64_t>(sid) * rank + r];
      const float score = unary[static_cast<int64_t>(l) * k + prev] + dot;
      if (score > best) {
        best = score;
        besti = c;
      }
    }
    require(got[l] == ids[static_cast<int64_t>(l) * k + besti],
            "selector step " + std::to_string(l) + ": got " + std::to_string(got[l]) + ", want " +
                std::to_string(ids[static_cast<int64_t>(l) * k + besti]) + " (row " + std::to_string(prev) +
                ", best score " + std::to_string(best) + ")");
    prev = besti;
  }
  }  // shape cases
}

// ---- the block attention --------------------------------------------------------

DGPP_TEST(dflash2_block_attends_window_and_block) {
  cudaStream_t s = test_stream();
  const int rows = 8, heads = 4, kv = 2, dim = 128, block_tokens = 16;
  const int64_t ctx_end = 39;  // 40 context positions: blocks 0..2
  const int64_t blk_lo = 40, blk_hi = 47, window = 24;  // a window that actually bites
  const int64_t pos0 = 40;
  const int64_t nslots = 64;  // 4 blocks
  auto q = random_bf16_normal(41, static_cast<int64_t>(rows) * heads * dim, 1.0f);
  auto kc = random_bf16_normal(42, nslots * kv * dim, 1.0f);
  auto vc = random_bf16_normal(43, nslots * kv * dim, 1.0f);
  std::vector<int32_t> table = {3, 1, 2, 0};  // scrambled physical order
  std::vector<int64_t> pos(rows);
  for (int r = 0; r < rows; ++r) pos[r] = pos0 + r;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dim));

  DevBuf dq(q.size() * 2), dkc(kc.size() * 2), dvc(vc.size() * 2), dy(q.size() * 2);
  DevBuf dtb(table.size() * 4), dpos(pos.size() * 8);
  dq.upload(q.data(), q.size() * 2);
  dkc.upload(kc.data(), kc.size() * 2);
  dvc.upload(vc.data(), vc.size() * 2);
  dtb.upload(table.data(), table.size() * 4);
  dpos.upload(pos.data(), pos.size() * 8);
  dgpp::dflash2_block_attn(cb16(dq), heads * dim, cb16(dkc), cb16(dvc), ci32(dtb), block_tokens,
                           static_cast<int>(table.size()), ctx_end, blk_lo, blk_hi, window,
                           ci64(dpos), rows, heads, kv, dim, scale, b16(dy), s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  std::vector<uint16_t> y(q.size());
  dy.download(y.data(), y.size() * 2);

  const auto qf = to_f(q), kf = to_f(kc), vf = to_f(vc);
  const auto slot = [&](int64_t p) {
    return static_cast<int64_t>(table[p / block_tokens]) * block_tokens + p % block_tokens;
  };
  std::vector<uint16_t> want(q.size());
  // Mirror the kernel's online softmax op-for-op (lane partials, the xor
  // butterfly, running rescale) so only the expf implementations float.
  for (int r = 0; r < rows; ++r)
    for (int h = 0; h < heads; ++h) {
      const int kvh = h / (heads / kv);
      float qv[32][4], lanep[32], lanet[32], acc[32][4];
      for (int lane = 0; lane < 32; ++lane)
        for (int j = 0; j < 4; ++j) {
          qv[lane][j] = qf[(static_cast<int64_t>(r) * heads + h) * dim + lane * 4 + j] * scale;
          acc[lane][j] = 0.0f;
        }
      float m = -INFINITY, l = 0.0f;
      const auto visit = [&](int64_t kp) {
        const int64_t kb = (slot(kp) * kv + kvh) * dim;
        for (int lane = 0; lane < 32; ++lane) {
          float d = 0.0f;
          for (int j = 0; j < 4; ++j) d += qv[lane][j] * kf[kb + lane * 4 + j];
          lanep[lane] = d;
        }
        for (int off = 16; off; off >>= 1) {
          for (int lane = 0; lane < 32; ++lane) lanet[lane] = lanep[lane] + lanep[lane ^ off];
          for (int lane = 0; lane < 32; ++lane) lanep[lane] = lanet[lane];
        }
        const float dot = lanep[0];
        const float m_new = dot > m ? dot : m;
        const float correction = m == -INFINITY ? 0.0f : std::exp(m - m_new);
        const float pf = m_new == -INFINITY ? 0.0f : std::exp(dot - m_new);
        const float pb = dgpp::bf16_bits_to_float(dgpp::float_to_bf16_bits(pf));
        l = l * correction + pf;
        for (int lane = 0; lane < 32; ++lane)
          for (int j = 0; j < 4; ++j)
            acc[lane][j] = acc[lane][j] * correction + pb * vf[kb + lane * 4 + j];
        m = m_new;
      };
      const int64_t lo = std::max<int64_t>(0, pos[r] - window + 1);
      for (int64_t p = lo; p <= ctx_end; ++p) visit(p);
      for (int64_t p = blk_lo; p <= blk_hi; ++p) visit(p);
      const float inv = l > 0.0f ? 1.0f / l : 0.0f;
      for (int lane = 0; lane < 32; ++lane)
        for (int j = 0; j < 4; ++j)
          want[(static_cast<int64_t>(r) * heads + h) * dim + lane * 4 + j] =
              dgpp::float_to_bf16_bits(acc[lane][j] * inv);
    }
  require_bf16("block attention", compare_bf16(y, want, 2), 1e-2, 0.02);
}

}  // namespace

int main() {
  return dgpp::test::run_all();
}
