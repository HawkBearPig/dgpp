// The DFlash2 drafter kernels against host references: the 2-tap dynamic
// grouped conv (block-border gating, side/tap slices), the standard
// norm + rotate-half rope, the block attention (windowed causal context
// + the bidirectional block, paged), the candidate top-K (descending,
// ties to the lower id) and the selector walk (predecessor codes chained
// through slots, the anchor at step 0, greedy argmax).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/dflash2.hpp"
#include "kernels/pick.hpp"

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
  // One chunk (the direct write), two uneven chunks, and the drafter's
  // vocabulary (31 chunks of 8011 — every block merges and the warp folds).
  for (const int64_t V : {int64_t{4096}, int64_t{10007}, int64_t{248320}}) {
    std::mt19937 rng(7);
    std::vector<float> lg(static_cast<size_t>(rows) * V);
    for (auto& v : lg) v = std::uniform_real_distribution<float>(-8.0f, 8.0f)(rng);
    // Two forced ties at the very top of row 1, one more straddling a chunk
    // boundary in row 2 (ties go to the lower id, whichever block saw it).
    lg[V + 5] = 100.0f;
    lg[V + 9] = 100.0f;
    lg[2 * V + 8010] = 50.0f;
    lg[2 * V + 8011] = 50.0f;
    DevBuf dl(lg.size() * 4), did(rows * k * 4), dsc(rows * k * 4);
    DevBuf dws(dgpp::dflash2_topk_ws_bytes(V, rows, k));
    dl.upload(lg.data(), lg.size() * 4);
    dgpp::dflash2_topk_f32(cf32(dl), i32(did), f32(dsc), V, rows, k, s, dws.p, dws.bytes);
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
    if (V == 248320) {  // the drafter's row set at a block of eight: the kernel's time
      cudaEvent_t e0, e1;
      DGPP_CUDA_OK(cudaEventCreate(&e0));
      DGPP_CUDA_OK(cudaEventCreate(&e1));
      std::vector<float> lg8(static_cast<size_t>(8) * V);
      for (size_t i = 0; i < lg8.size(); ++i) lg8[i] = lg[i % lg.size()];
      DevBuf dl8(lg8.size() * 4), did8(8 * k * 4), dsc8(8 * k * 4), dws8(dgpp::dflash2_topk_ws_bytes(V, 8, k));
      dl8.upload(lg8.data(), lg8.size() * 4);
      dgpp::dflash2_topk_f32(cf32(dl8), i32(did8), f32(dsc8), V, 8, k, s, dws8.p, dws8.bytes);
      DGPP_CUDA_OK(cudaEventRecord(e0, s));
      for (int it = 0; it < 20; ++it)
        dgpp::dflash2_topk_f32(cf32(dl8), i32(did8), f32(dsc8), V, 8, k, s, dws8.p, dws8.bytes);
      DGPP_CUDA_OK(cudaEventRecord(e1, s));
      DGPP_CUDA_OK(cudaEventSynchronize(e1));
      float ms = 0.f;
      DGPP_CUDA_OK(cudaEventElapsedTime(&ms, e0, e1));
      std::printf("[ .. ] dflash2 top-16 over 8 rows x %lld: %.1f us a call (both passes)\n",
                  static_cast<long long>(V), ms * 1000.f / 20.f);
      DGPP_CUDA_OK(cudaEventDestroy(e0));
      DGPP_CUDA_OK(cudaEventDestroy(e1));
    }
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
          throw std::runtime_error("topk V " + std::to_string(V) + " score row " + std::to_string(r) + " slot " +
                                   std::to_string(j) + ": got " + std::to_string(sc[r * k + j]) + " want " +
                                   std::to_string(all[j].first));
        if (ids[r * k + j] != all[j].second)
          throw std::runtime_error("topk V " + std::to_string(V) + " id row " + std::to_string(r) + " slot " +
                                   std::to_string(j) + ": got " + std::to_string(ids[r * k + j]) + " want " +
                                   std::to_string(all[j].second));
      }
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
  const int64_t anchor64 = anchor;
  auto pred = random_bf16_normal(31, static_cast<int64_t>(vocab) * rank, 0.5f);
  auto succ = random_bf16_normal(32, static_cast<int64_t>(vocab) * rank, 0.5f);
  std::mt19937 rng(33);
  std::vector<int32_t> ids(steps * k);
  for (auto& t : ids) t = static_cast<int32_t>(rng() % vocab);
  std::vector<float> unary(steps * k), hidden(steps * rank);
  for (auto& v : unary) v = std::uniform_real_distribution<float>(-4.0f, 4.0f)(rng);
  for (auto& v : hidden) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);

  DevBuf dp(pred.size() * 2), ds(succ.size() * 2), di(ids.size() * 4), du(unary.size() * 4),
      dh(hidden.size() * 4), dt(steps * 4), da(8), dconf(steps * 4);
  da.upload(&anchor64, 8);
  dp.upload(pred.data(), pred.size() * 2);
  ds.upload(succ.data(), succ.size() * 2);
  di.upload(ids.data(), ids.size() * 4);
  du.upload(unary.data(), unary.size() * 4);
  dh.upload(hidden.data(), hidden.size() * 4);
  dgpp::dflash2_selector_walk(ci32(di), cf32(du), cf32(dh), cb16(dp), cb16(ds), ci64(da), i32(dt),
                              steps, k, rank, s, nullptr, nullptr, nullptr, nullptr, f32(dconf));
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  std::vector<int32_t> got(steps);
  dt.download(got.data(), got.size() * 4);
  std::vector<float> conf(steps);
  dconf.download(conf.data(), conf.size() * 4);

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
      const float score = unary[static_cast<int64_t>(l) * k + c] + dot;
      if (score > best) {
        best = score;
        besti = c;
      }
    }
    require(got[l] == ids[static_cast<int64_t>(l) * k + besti],
            "selector step " + std::to_string(l) + ": got " + std::to_string(got[l]) + ", want " +
                std::to_string(ids[static_cast<int64_t>(l) * k + besti]) + " (row " + std::to_string(prev) +
                ", best score " + std::to_string(best) + ")");
    // The confidence: the argmax's softmax mass over the step's candidates, as a logit.
    {
      double den = 0.0;
      for (int c = 0; c < k; ++c) {
        const int32_t pid = l == 0 ? anchor : ids[static_cast<int64_t>(l - 1) * k + prev];
        const int32_t sid = ids[static_cast<int64_t>(l) * k + c];
        float dot = 0.0f;
        for (int r = 0; r < rank; ++r)
          dot += pf[static_cast<int64_t>(pid) * rank + r] * hidden[l * rank + r] *
                 sf[static_cast<int64_t>(sid) * rank + r];
        den += std::exp(static_cast<double>(unary[static_cast<int64_t>(l) * k + c] + dot) - best);
      }
      const double q = std::min(std::max(1.0 / den, 1e-6), 1.0 - 1e-6);
      const double want = std::log(q / (1.0 - q));
      require(std::fabs(conf[l] - want) < 1e-3 * std::max(1.0, std::fabs(want)),
              "selector step " + std::to_string(l) + ": confidence " + std::to_string(conf[l]) + " vs " +
                  std::to_string(want));
    }
    prev = besti;
  }
  }  // shape cases
}

// The sampled walk (a stochastic spec): every step's token is one of its
// candidates and the proposal lists the k candidates with softmax(scores /
// T) masses and the draw as `token`, on the device and in the pinned mirror;
// the draw is the inverse-CDF walk on the draft stream's uniform keyed by
// the draft's position (reproduced here), so equal inputs draw equal chains
// and a greedy spec walks the argmax with n = 0.
DGPP_TEST(dflash2_selector_sampled_walk_proposes_its_draw) {
  cudaStream_t s = test_stream();
  const int steps = 7, k = 16, rank = 256, vocab = 1009;
  const int32_t anchor = 5;
  const int64_t anchor64 = anchor;
  auto pred = random_bf16_normal(41, static_cast<int64_t>(vocab) * rank, 0.5f);
  auto succ = random_bf16_normal(42, static_cast<int64_t>(vocab) * rank, 0.5f);
  std::mt19937 rng(43);
  std::vector<int32_t> ids(steps * k);
  for (int l = 0; l < steps; ++l)  // distinct candidates per step
    for (int c = 0; c < k; ++c) ids[l * k + c] = static_cast<int32_t>((l * 37 + c * 7 + rng() % 5) % vocab);
  std::vector<float> unary(steps * k), hidden(steps * rank);
  for (auto& v : unary) v = std::uniform_real_distribution<float>(-2.0f, 2.0f)(rng);
  for (auto& v : hidden) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
  std::vector<int64_t> pos(steps + 1);
  for (int j = 0; j <= steps; ++j) pos[j] = 1000 + j;

  DevBuf dp(pred.size() * 2), ds(succ.size() * 2), di(ids.size() * 4), du(unary.size() * 4),
      dh(hidden.size() * 4), dt(steps * 4), da(8), dpos(pos.size() * 8), dspec(sizeof(dgpp::SampleSpec)),
      dprop(sizeof(dgpp::DraftProposal) * dgpp::kSampleProposalSlots);
  da.upload(&anchor64, 8);
  dp.upload(pred.data(), pred.size() * 2);
  ds.upload(succ.data(), succ.size() * 2);
  di.upload(ids.data(), ids.size() * 4);
  du.upload(unary.data(), unary.size() * 4);
  dh.upload(hidden.data(), hidden.size() * 4);
  dpos.upload(pos.data(), pos.size() * 8);
  dgpp::DraftProposal* hprop = nullptr;
  DGPP_CUDA_OK(cudaMallocHost(reinterpret_cast<void**>(&hprop),
                              sizeof(dgpp::DraftProposal) * dgpp::kSampleProposalSlots));
  for (int l = 0; l < dgpp::kSampleProposalSlots; ++l) hprop[l] = dgpp::DraftProposal{};

  auto walk = [&](float temperature, uint64_t seed, std::vector<int32_t>* tokens,
                  std::vector<dgpp::DraftProposal>* props) {
    dgpp::SampleSpec spec{};
    spec.temperature = temperature;
    spec.seed = seed;
    dspec.upload(&spec, sizeof(spec));
    DGPP_CUDA_OK(cudaMemsetAsync(dprop.p, 0x7f, dprop.bytes, s));  // stale rows
    dgpp::dflash2_selector_walk(ci32(di), cf32(du), cf32(dh), cb16(dp), cb16(ds), ci64(da), i32(dt),
                                steps, k, rank, s, ci64(dpos),
                                static_cast<const dgpp::SampleSpec*>(dspec.p),
                                static_cast<dgpp::DraftProposal*>(dprop.p), hprop);
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
    tokens->assign(steps, 0);
    dt.download(tokens->data(), steps * 4);
    props->assign(dgpp::kSampleProposalSlots, dgpp::DraftProposal{});
    dprop.download(props->data(), sizeof(dgpp::DraftProposal) * dgpp::kSampleProposalSlots);
  };

  // The host's view of the chain: scores of step l given the previous pick.
  const auto pf = to_f(pred), sf = to_f(succ);
  auto scores = [&](int l, int prev, std::vector<float>* out) {
    out->assign(k, 0.0f);
    for (int c = 0; c < k; ++c) {
      const int32_t pid = l == 0 ? anchor : ids[static_cast<int64_t>(l - 1) * k + prev];
      const int32_t sid = ids[static_cast<int64_t>(l) * k + c];
      float dot = 0.0f;
      for (int r = 0; r < rank; ++r)
        dot += pf[static_cast<int64_t>(pid) * rank + r] * hidden[l * rank + r] *
               sf[static_cast<int64_t>(sid) * rank + r];
      (*out)[c] = unary[static_cast<int64_t>(l) * k + c] + dot;
    }
  };

  // Greedy spec: the argmax chain, no proposal.
  std::vector<int32_t> greedy;
  std::vector<dgpp::DraftProposal> gprops;
  walk(0.0f, 7, &greedy, &gprops);
  {
    int prev = 0;
    for (int l = 0; l < steps; ++l) {
      std::vector<float> sc;
      scores(l, prev, &sc);
      int best = 0;
      for (int c = 1; c < k; ++c)
        if (sc[c] > sc[best]) best = c;
      require(greedy[l] == ids[l * k + best], "greedy step " + std::to_string(l) + " is the argmax");
      require(gprops[l].n == 0 && hprop[l].n == 0, "a greedy spec writes no proposal (n = 0)");
      prev = best;
    }
  }

  // Sampled spec at temperature 1: candidates, masses, the draw, the mirror.
  std::vector<int32_t> tok;
  std::vector<dgpp::DraftProposal> props;
  walk(1.0f, 12345, &tok, &props);
  {
    int prev = 0;
    for (int l = 0; l < steps; ++l) {
      std::vector<float> sc;
      scores(l, prev, &sc);
      const float mx = *std::max_element(sc.begin(), sc.end());
      std::vector<double> q(k);
      double den = 0.0;
      for (int c = 0; c < k; ++c) {
        q[c] = std::exp(static_cast<double>(sc[c] - mx));
        den += q[c];
      }
      const dgpp::DraftProposal& p = props[l];
      require(p.n == k, "step " + std::to_string(l) + ": the proposal lists the k candidates");
      require(p.token == tok[l], "the proposal's token is the drawn draft");
      require(hprop[l].n == p.n && hprop[l].token == p.token &&
                  std::memcmp(hprop[l].ids, p.ids, sizeof(int32_t) * k) == 0 &&
                  std::memcmp(hprop[l].mass, p.mass, sizeof(float) * k) == 0,
              "the pinned mirror carries the device proposal (n, token, the k ids and masses)");
      int chosen = -1;
      double sum = 0.0;
      for (int c = 0; c < k; ++c) {
        require(p.ids[c] == ids[l * k + c], "the proposal's ids are the step's candidates in order");
        const double want = q[c] / den;
        require(std::fabs(p.mass[c] - want) <= 1e-4 * std::max(1.0, want) + 1e-6,
                "step " + std::to_string(l) + " candidate " + std::to_string(c) + ": mass " +
                    std::to_string(p.mass[c]) + " vs softmax " + std::to_string(want));
        sum += p.mass[c];
        if (p.ids[c] == tok[l]) chosen = c;
      }
      require(std::fabs(sum - 1.0) < 1e-4, "the masses sum to one");
      require(chosen >= 0, "the draw is one of the candidates");
      // The draft stream's uniform at the draft's position decides the draw.
      const uint64_t seed = 12345ull ^ dgpp::kSampleDraftSeedMix ^ (static_cast<uint64_t>(l + 1) * 0xD1B54A32D192ED03ull);
      const double u = dgpp::uniform01(seed, static_cast<uint64_t>(pos[l + 1]));
      double cum = 0.0;
      int want = k - 1;
      for (int c = 0; c < k; ++c) {
        cum += q[c] / den;
        if (cum > u) {
          want = c;
          break;
        }
      }
      // Equal unless the uniform falls within fp32 rounding of a boundary.
      const bool boundary = std::fabs(cum - u) < 1e-5;
      require(chosen == want || boundary, "step " + std::to_string(l) + ": the inverse-CDF draw (" +
                                               std::to_string(want) + ") vs the kernel's (" + std::to_string(chosen) + ")");
      prev = chosen;
    }
  }
  // The request's truncation on the proposal: top-p 0.5 keeps the smallest
  // prefix of the mass-ordered candidates reaching 0.5, renormalized; the
  // draw is one of the survivors.
  {
    dgpp::SampleSpec spec{};
    spec.temperature = 1.0f;
    spec.seed = 777;
    spec.top_p = 0.5f;
    dspec.upload(&spec, sizeof(spec));
    dgpp::dflash2_selector_walk(ci32(di), cf32(du), cf32(dh), cb16(dp), cb16(ds), ci64(da), i32(dt),
                                steps, k, rank, s, ci64(dpos),
                                static_cast<const dgpp::SampleSpec*>(dspec.p),
                                static_cast<dgpp::DraftProposal*>(dprop.p), hprop);
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
    std::vector<int32_t> ttok(steps);
    dt.download(ttok.data(), steps * 4);
    std::vector<dgpp::DraftProposal> tp(dgpp::kSampleProposalSlots);
    dprop.download(tp.data(), sizeof(dgpp::DraftProposal) * dgpp::kSampleProposalSlots);
    int prev = 0;
    for (int l = 0; l < steps; ++l) {
      std::vector<float> sc;
      scores(l, prev, &sc);
      const float mx = *std::max_element(sc.begin(), sc.end());
      std::vector<std::pair<double, int>> q(k);
      double den = 0.0;
      for (int c = 0; c < k; ++c) {
        q[c] = {std::exp(static_cast<double>(sc[c] - mx)), c};
        den += q[c].first;
      }
      std::sort(q.begin(), q.end(), [&](const auto& a, const auto& b) {
        return a.first != b.first ? a.first > b.first : ids[l * k + a.second] < ids[l * k + b.second];
      });
      double cum = 0.0;
      int keep = 0;
      while (keep < k) {
        cum += q[keep].first / den;
        ++keep;
        if (cum >= 0.5) break;
      }
      double kept = 0.0;
      for (int i = 0; i < keep; ++i) kept += q[i].first;
      double sum = 0.0;
      int chosen = -1;
      for (int i = 0; i < k; ++i) {
        const int c = q[i].second;
        const double want = i < keep ? q[i].first / kept : 0.0;
        require(std::fabs(tp[l].mass[c] - want) <= 1e-4 * std::max(1.0, want) + 1e-6,
                "top-p 0.5 step " + std::to_string(l) + " candidate " + std::to_string(c) + ": mass " +
                    std::to_string(tp[l].mass[c]) + " vs " + std::to_string(want));
        sum += tp[l].mass[c];
        if (tp[l].ids[c] == ttok[l]) chosen = c;
      }
      require(std::fabs(sum - 1.0) < 1e-4, "the survivors' masses sum to one");
      require(chosen >= 0 && tp[l].mass[chosen] > 0.0f, "the draw is a survivor");
      prev = chosen;
    }
  }
  // Determinism and seed sensitivity.
  std::vector<int32_t> tok2, tok3;
  std::vector<dgpp::DraftProposal> p2, p3;
  walk(1.0f, 12345, &tok2, &p2);
  require(tok2 == tok, "the same spec draws the same chain");
  walk(1.0f, 99, &tok3, &p3);
  bool moved = false;
  for (int l = 0; l < steps; ++l) moved = moved || tok3[l] != tok[l];
  require(moved || greedy == tok, "another seed moves the chain (unless the scores are near-deterministic)");
  DGPP_CUDA_OK(cudaFreeHost(hprop));
}

// ---- the block attention --------------------------------------------------------

DGPP_TEST(dflash_block_attention_skips_context_ceiling_padding) {
  cudaStream_t s = test_stream();
  constexpr int rows = 16, valid = 4, dim = 128, slots = 64;
  const auto q = random_bf16_normal(741, rows * dim, 1.f);
  const auto k = random_bf16_normal(742, slots * dim, 1.f);
  const auto v = random_bf16_normal(743, slots * dim, 1.f);
  std::vector<int64_t> positions(rows, -1);
  for (int i = 0; i < valid; ++i) positions[i] = 60 + i;
  const int32_t table = 0;  // one owned page: no entry for position 64
  DevBuf dq(q.size() * 2), dk(k.size() * 2), dv(v.size() * 2), dp(rows * 8), dt(4);
  DevBuf out(rows * dim * 2), ref(valid * dim * 2);
  DevBuf parts(dgpp::dflash2_block_attn_partials_bytes(rows, 1));
  dq.upload(q.data(), q.size() * 2); dk.upload(k.data(), k.size() * 2); dv.upload(v.data(), v.size() * 2);
  dp.upload(positions.data(), rows * 8); dt.upload(&table, 4);
  for (bool split : {false, true}) {
    if (split)
      dgpp::dflash2_block_attn_split(cb16(dq), dim, cb16(dk), cb16(dv), ci32(dt), slots, 1,
          rows, 128, ci64(dp), rows, 1, 1, dim, 1.f / std::sqrt(float(dim)), f32(parts), b16(out), s);
    else
      dgpp::dflash2_block_attn(cb16(dq), dim, cb16(dk), cb16(dv), ci32(dt), slots, 1,
          rows, 128, ci64(dp), rows, 1, 1, dim, 1.f / std::sqrt(float(dim)), b16(out), s);
    dgpp::dflash2_block_attn(cb16(dq), dim, cb16(dk), cb16(dv), ci32(dt), slots, 1,
        valid, 128, ci64(dp), valid, 1, 1, dim, 1.f / std::sqrt(float(dim)), b16(ref), s);
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
    std::vector<uint16_t> got(rows * dim), want(valid * dim);
    out.download(got.data(), got.size() * 2); ref.download(want.data(), want.size() * 2);
    for (size_t i = 0; i < got.size(); ++i) {
      const float expected = i < want.size() ? dgpp::bf16_bits_to_float(want[i]) : 0.f;
      require(std::fabs(dgpp::bf16_bits_to_float(got[i]) - expected) < 0.01f,
              "padded draft rows cannot extend the attention key span");
    }
  }
}

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
  // The span comes off the block's first row (pos0 = 40): context [0, 39],
  // block [40, 47] — the reference below spells the same bounds.
  dgpp::dflash2_block_attn(cb16(dq), heads * dim, cb16(dkc), cb16(dvc), ci32(dtb), block_tokens,
                           static_cast<int>(table.size()), rows, window, ci64(dpos), rows, heads, kv,
                           dim, scale, b16(dy), s);
  // The split-key form over the same keys (its partials scratch), judged
  // against the same reference below and against the serial form.
  DevBuf dys(q.size() * 2), dpart(dgpp::dflash2_block_attn_partials_bytes(rows, heads));
  dgpp::dflash2_block_attn_split(cb16(dq), heads * dim, cb16(dkc), cb16(dvc), ci32(dtb), block_tokens,
                                 static_cast<int>(table.size()), rows, window, ci64(dpos), rows, heads, kv,
                                 dim, scale, f32(dpart), b16(dys), s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  std::vector<uint16_t> y(q.size()), ys(q.size());
  dy.download(y.data(), y.size() * 2);
  dys.download(ys.data(), ys.size() * 2);
  {
    float worst = 0.0f;
    for (size_t i = 0; i < y.size(); ++i)
      worst = std::max(worst, std::fabs(dgpp::bf16_bits_to_float(y[i]) - dgpp::bf16_bits_to_float(ys[i])));
    require(worst <= 0.02f, "split-key block attention vs the serial form: worst |diff| " + std::to_string(worst));
  }

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

// ---- the recorded block draft ---------------------------------------------------

// Every rank's slice top-K staged, the bus's fold emulated (the bf16 sum
// of the per-rank tables, exact: one nonzero contributor per slot), the
// merge against the host's whole-vocabulary top-K, at worlds 2 and 4 and
// an uneven slice split.
DGPP_TEST(dflash2_topk_merges_the_ranks_slices) {
  cudaStream_t s = test_stream();
  const int rows = 3, k = 16;
  const int64_t V = 24001;  // odd: uneven slices
  std::mt19937 rng(77);
  std::vector<float> lg(static_cast<size_t>(rows) * V);
  for (auto& v : lg) v = std::uniform_real_distribution<float>(-8.0f, 8.0f)(rng);
  lg[V + 5] = 100.0f;  // a tie across the slice boundary's candidates
  lg[V + 20000] = 100.0f;
  for (const int world : {2, 4}) {
    const size_t elems = dgpp::dflash2_topk_table_elems(rows, k, world);
    std::vector<float> folded(elems, 0.0f);
    for (int rank = 0; rank < world; ++rank) {
      const int64_t begin = V * rank / world, end = V * (rank + 1) / world, count = end - begin;
      std::vector<float> slice(static_cast<size_t>(rows) * count);
      for (int r = 0; r < rows; ++r)
        std::copy(lg.begin() + r * V + begin, lg.begin() + r * V + end, slice.begin() + r * count);
      DevBuf dl(slice.size() * 4), did(rows * k * 4), dsc(rows * k * 4), dt(elems * 2);
      DevBuf dws(dgpp::dflash2_topk_ws_bytes(count, rows, k));
      dl.upload(slice.data(), slice.size() * 4);
      dgpp::dflash2_topk_f32(cf32(dl), i32(did), f32(dsc), count, rows, k, s, dws.p, dws.bytes);
      dgpp::dflash2_topk_stage(ci32(did), cf32(dsc), rows, k, static_cast<int32_t>(begin), rank, world,
                               b16(dt), s);
      DGPP_CUDA_OK(cudaStreamSynchronize(s));
      std::vector<uint16_t> table(elems);
      dt.download(table.data(), elems * 2);
      for (size_t i = 0; i < elems; ++i) folded[i] += dgpp::bf16_bits_to_float(table[i]);
    }
    std::vector<uint16_t> tbl(elems);
    for (size_t i = 0; i < elems; ++i) tbl[i] = dgpp::float_to_bf16_bits(folded[i]);
    DevBuf dt(elems * 2), did(rows * k * 4), dsc(rows * k * 4);
    dt.upload(tbl.data(), elems * 2);
    dgpp::dflash2_topk_merge(cb16(dt), rows, k, world, i32(did), f32(dsc), s);
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
        require(sc[r * k + j] == all[j].first,
                "merged score world " + std::to_string(world) + " row " + std::to_string(r) + " slot " +
                    std::to_string(j) + ": got " + std::to_string(sc[r * k + j]) + " want " +
                    std::to_string(all[j].first));
        require(ids[r * k + j] == all[j].second,
                "merged id world " + std::to_string(world) + " row " + std::to_string(r) + " slot " +
                    std::to_string(j) + ": got " + std::to_string(ids[r * k + j]) + " want " +
                    std::to_string(all[j].second));
      }
    }
  }
}

// The block's rows off a verdict and the committed position, the context
// tail masked; the next feed [next, drafts]; the pinned mirror.
DGPP_TEST(dflash2_block_stage_feed_and_publish) {
  cudaStream_t s = test_stream();
  const int rows = 8, D = 7;
  dgpp::PickVerdict v;
  v.rows = 8;
  v.accepted = 3;
  v.next = 1234;
  const int64_t session_pos = 100, max_context = 104, mask = 248064;
  const std::vector<int32_t> drafts = {11, 12, 13, 14, 15, 16, 17};
  DevBuf dv(sizeof(v)), dsp(8), dpos(rows * 8), dtok(rows * 8), ddr(D * 4), dfeed(rows * 8);
  dv.upload(&v, sizeof(v));
  dsp.upload(&session_pos, 8);
  ddr.upload(drafts.data(), D * 4);
  int32_t* pinned = nullptr;
  DGPP_CUDA_OK(cudaMallocHost(reinterpret_cast<void**>(&pinned), D * 4));
  for (int j = 0; j < D; ++j) pinned[j] = -1;
  dgpp::dflash2_stage_block(static_cast<const dgpp::PickVerdict*>(dv.p), ci64(dsp), mask, rows, max_context,
                            static_cast<int64_t*>(dpos.p), static_cast<int64_t*>(dtok.p), s);
  dgpp::dflash2_block_feed(static_cast<const dgpp::PickVerdict*>(dv.p), ci32(ddr), D,
                           static_cast<int64_t*>(dfeed.p), s);
  dgpp::dflash2_publish_drafts(ci32(ddr), pinned, D, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  std::vector<int64_t> pos(rows), tok(rows), feed(rows);
  dpos.download(pos.data(), rows * 8);
  dtok.download(tok.data(), rows * 8);
  dfeed.download(feed.data(), rows * 8);
  for (int j = 0; j < rows; ++j) {
    const int64_t want = session_pos + j < max_context ? session_pos + j : -1;
    require(pos[j] == want, "staged position " + std::to_string(j) + ": got " + std::to_string(pos[j]));
    require(tok[j] == (j == 0 ? 1234 : mask), "staged token " + std::to_string(j));
    require(feed[j] == (j == 0 ? 1234 : drafts[static_cast<size_t>(j - 1)]), "feed token " + std::to_string(j));
  }
  for (int j = 0; j < D; ++j) require(pinned[j] == drafts[static_cast<size_t>(j)], "published draft " + std::to_string(j));
  DGPP_CUDA_OK(cudaFreeHost(pinned));
}

// The fixed batch's forms: three slots, the middle one inactive (accepted
// 0): its rows at position -1 with token 0, its feed zeroed, the others'
// blocks, feeds and mirrors at their slot offsets.
DGPP_TEST(dflash2_block_batched_stage_feed_and_publish) {
  cudaStream_t s = test_stream();
  const int rows = 8, D = 7, k = 3;
  std::vector<dgpp::PickVerdict> v(k);
  v[0].rows = 8; v[0].accepted = 2; v[0].next = 500;
  v[1].rows = 0; v[1].accepted = 0; v[1].next = -1;
  v[2].rows = 8; v[2].accepted = 8; v[2].next = 900;
  const std::vector<int64_t> session_pos = {40, 7, 1000};
  const int64_t max_context = 1004, mask = 248064;
  std::vector<int32_t> drafts(k * D);
  for (int i = 0; i < k * D; ++i) drafts[i] = 1000 + i;
  DevBuf dv(sizeof(dgpp::PickVerdict) * k), dsp(k * 8), dpos(k * rows * 8), dtok(k * rows * 8), ddr(k * D * 4),
      dfeed(k * rows * 8);
  dv.upload(v.data(), sizeof(dgpp::PickVerdict) * k);
  dsp.upload(session_pos.data(), k * 8);
  ddr.upload(drafts.data(), k * D * 4);
  int32_t* pinned = nullptr;
  DGPP_CUDA_OK(cudaMallocHost(reinterpret_cast<void**>(&pinned), k * D * 4));
  for (int j = 0; j < k * D; ++j) pinned[j] = -1;
  dgpp::dflash2_stage_block_batched(static_cast<const dgpp::PickVerdict*>(dv.p), ci64(dsp), mask, rows, k,
                                    max_context, static_cast<int64_t*>(dpos.p), static_cast<int64_t*>(dtok.p), s);
  dgpp::dflash2_block_feed_batched(static_cast<const dgpp::PickVerdict*>(dv.p), ci32(ddr), D, k, rows,
                                   static_cast<int64_t*>(dfeed.p), s);
  dgpp::dflash2_publish_drafts_batched(ci32(ddr), pinned, D, k, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  std::vector<int64_t> pos(k * rows), tok(k * rows), feed(k * rows);
  dpos.download(pos.data(), k * rows * 8);
  dtok.download(tok.data(), k * rows * 8);
  dfeed.download(feed.data(), k * rows * 8);
  for (int q = 0; q < k; ++q) {
    const bool active = v[static_cast<size_t>(q)].accepted > 0;
    for (int j = 0; j < rows; ++j) {
      const int i = q * rows + j;
      const int64_t p = session_pos[static_cast<size_t>(q)] + j;
      const int64_t want_pos = active && p < max_context ? p : -1;
      require(pos[i] == want_pos, "batched position slot " + std::to_string(q) + " row " + std::to_string(j) + ": got " +
                                      std::to_string(pos[i]));
      const int64_t want_tok = !active ? 0 : (j == 0 ? v[static_cast<size_t>(q)].next : mask);
      require(tok[i] == want_tok, "batched token slot " + std::to_string(q) + " row " + std::to_string(j));
      const int64_t want_feed = !active ? 0 : (j == 0 ? v[static_cast<size_t>(q)].next : drafts[q * D + j - 1]);
      require(feed[i] == want_feed, "batched feed slot " + std::to_string(q) + " row " + std::to_string(j));
    }
    for (int j = 0; j < D; ++j)
      require(pinned[q * D + j] == drafts[q * D + j], "batched published draft slot " + std::to_string(q));
  }
  // Slot 2 runs into the context's tail: rows 1004.. are masked.
  require(pos[2 * rows + 3] == 1003 && pos[2 * rows + 4] == -1, "the context tail masks the rows past it");
  DGPP_CUDA_OK(cudaFreeHost(pinned));
}

int main() {
  return dgpp::test::run_all();
}
