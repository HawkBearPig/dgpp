// The Dory attention kernels (kernels/dory_attn.hpp) — the qkv finish (the
// neoX RoPE, the per-head justnorm, the `sqk` square), the split-KV paged GQA
// attention with the sliding window applied at read time, and the combine —
// against a host oracle transcribed from docs/dory_plan.md §1.3.
//
// WHAT IS PINNED HOW:
//
// * The V cache is bitwise: `v = bf16(dot)` is a single rounding of a value
//   both sides compute the same way, and it is written through the block table
//   — so it pins the physical slot indexing (request table + p/block_tokens)
//   exactly, including a permuted table.
// * The K cache and `q` are compared with a 2-ulp budget: the finish is a
//   chain of bf16 roundings the oracle reproduces op for op, but `cosf`/`sinf`
//   may differ from the host's by an ulp, which occasionally flips a bf16
//   rounding. Everything else (the RoPE form, the fp64-exact row sum, the
//   `sqk` square applied after RoPE and over justnorm's bf16 output, the
//   rounding of every Linear output) is pinned by that budget: a wrong
//   scaling site moves most of the values.
// * The attention output is compared with a 2-ulp budget under three
//   geometries (the model's 16/4 heads, 8/2 and 6/3 for the generic
//   path), n_split 1/2/4, and both layer kinds.
// * The window is pinned by DIFFERENCE: a long history (2048 tokens) with
//   window 1024 must match the windowed oracle and must NOT match the
//   full-history one, so a dropped window mask cannot pass.
// * A padding row (`pos < 0`) writes nothing: `q_out` keeps its sentinel, the
//   caches are untouched, and the combined output row is zero.
//
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/dory_attn.hpp"

using namespace dgpp::kda_test;

namespace {

constexpr int D = dgpp::kDoryHeadDim;  // 256
constexpr float kDoryJustnormEps = 1e-6f;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

uint16_t rbits(float v) { return dgpp::float_to_bf16_bits(v); }
float rb(float v) { return dgpp::bf16_bits_to_float(dgpp::float_to_bf16_bits(v)); }

// A projection's output rows are bf16 (a Linear's rounding).
std::vector<uint16_t> to_bf16(const std::vector<float>& f) {
  std::vector<uint16_t> b(f.size());
  for (size_t i = 0; i < f.size(); ++i) b[i] = rbits(f[i]);
  return b;
}

std::vector<float> rope_inv_freq(double theta, int dim) {
  std::vector<float> f(static_cast<size_t>(dim / 2));
  for (size_t i = 0; i < f.size(); ++i)
    f[i] = static_cast<float>(std::pow(theta, -2.0 * static_cast<double>(i) / dim));
  return f;
}

// ---- the host oracle ---------------------------------------------------------

struct Problem {
  int rows = 0, local_heads = 0, kv_heads = 0, n_split = 1, window = 0;
  int blocks_per_request = 0, block_tokens = 0;
  std::vector<int32_t> req_ids;
  std::vector<int64_t> pos;
  std::vector<int32_t> block_tables;
  std::vector<float> inv_freq;
  std::vector<uint16_t> sqk;
  float sqk_factor = 1.0f, eps = kDoryJustnormEps;
  float scale = 1.0f / std::sqrt(256.0f);
  std::vector<uint16_t> q_dot, k_dot, v_dot;  // bf16 (a Linear's output)

  int requests() const {
    int n = 1;
    for (int32_t r : req_ids) n = std::max(n, static_cast<int>(r) + 1);
    return n;
  }
  // The pool spans every request's blocks: the table holds GLOBAL block ids.
  int slots() const { return requests() * blocks_per_request * block_tokens; }
  int hpk() const { return local_heads / kv_heads; }
  int kv_width() const { return kv_heads * D; }
  int64_t phys(int r, int64_t t) const {
    const int32_t blk = block_tables[static_cast<size_t>(req_ids[static_cast<size_t>(r)]) *
                                         blocks_per_request +
                                     static_cast<size_t>(t / block_tokens)];
    return static_cast<int64_t>(blk) * block_tokens + (t % block_tokens);
  }
  bool visible(int r, int64_t t) const {
    const int64_t p = pos[static_cast<size_t>(r)];
    if (p < 0) return false;
    const int64_t end = p + 1;
    const int64_t first = (window > 0 && end > window) ? end - window : 0;
    return t >= first && t < end;
  }
};

// The finish, transcribed (plan §1.3): bf16(dot) -> neoX RoPE with the
// reference's bf16 ops -> per-head justnorm (fp64 row sum) -> `sqk` square.
struct Oracle {
  std::vector<uint16_t> q_out, k_cache, v_cache;

  // One head's q or k row: `dot` the Linear's bf16 output row, `p` the
  // absolute position.
  static void finish_row(const uint16_t* dot, const float* inv_freq, int64_t p, float eps,
                         uint16_t* out) {
    float x[D];
    for (int i = 0; i < D; ++i) x[i] = dgpp::bf16_bits_to_float(dot[i]);
    constexpr int H = D / 2;
    for (int i = 0; i < H; ++i) {
      const float ang = static_cast<float>(static_cast<double>(p) * static_cast<double>(inv_freq[i]));
      const float c = rb(std::cos(ang)), s = rb(std::sin(ang));
      const float x1 = x[i], x2 = x[i + H];
      x[i] = rb(rb(x1 * c) + rb(-x2 * s));
      x[i + H] = rb(rb(x2 * c) + rb(x1 * s));
    }
    double ss = 0.0;
    for (int i = 0; i < D; ++i) ss = std::fma(static_cast<double>(x[i]), static_cast<double>(x[i]), ss);
    const float rstd = 1.0f / std::sqrt(static_cast<float>(ss) + eps);
    for (int i = 0; i < D; ++i) out[i] = rbits(x[i] * rstd);
  }

  void run(const Problem& pb) {
    const int kvw = pb.kv_width();
    q_out.assign(static_cast<size_t>(pb.rows) * pb.local_heads * D, 0);
    k_cache.assign(static_cast<size_t>(pb.slots()) * kvw, 0);
    v_cache.assign(static_cast<size_t>(pb.slots()) * kvw, 0);
    for (int r = 0; r < pb.rows; ++r) {
      if (pb.pos[static_cast<size_t>(r)] < 0) continue;  // a padding row writes nothing
      const uint16_t* qd = pb.q_dot.data() + static_cast<size_t>(r) * pb.local_heads * D;
      const uint16_t* kd = pb.k_dot.data() + static_cast<size_t>(r) * kvw;
      const uint16_t* vd = pb.v_dot.data() + static_cast<size_t>(r) * kvw;
      for (int h = 0; h < pb.local_heads; ++h) {
        uint16_t row[D];
        finish_row(qd + static_cast<size_t>(h) * D, pb.inv_freq.data(),
                   pb.pos[static_cast<size_t>(r)], pb.eps, row);
        uint16_t* qr = q_out.data() + (static_cast<size_t>(r) * pb.local_heads + h) * D;
        for (int i = 0; i < D; ++i) {
          const float u = dgpp::bf16_bits_to_float(row[i]);
          const float m = dgpp::bf16_bits_to_float(pb.sqk[static_cast<size_t>(h) * D + i]) * pb.sqk_factor;
          qr[i] = rbits(u * (m * m));
        }
      }
      for (int h = 0; h < pb.kv_heads; ++h) {
        const int64_t slot = pb.phys(r, pb.pos[static_cast<size_t>(r)]);
        finish_row(kd + static_cast<size_t>(h) * D, pb.inv_freq.data(),
                   pb.pos[static_cast<size_t>(r)], pb.eps,
                   k_cache.data() + static_cast<size_t>(slot) * kvw + static_cast<size_t>(h) * D);
        for (int i = 0; i < D; ++i)
          v_cache[static_cast<size_t>(slot) * kvw + static_cast<size_t>(h) * D + i] =
              vd[static_cast<size_t>(h) * D + i];
      }
    }
  }

  // The attention: causal GQA over the visible range, the probabilities
  // rounded to bf16 with the UNROUNDED denominator (plan §1.3), fp32
  // accumulation, out = bf16(C/L).
  std::vector<uint16_t> attention(const Problem& pb) const {
    const int kvw = pb.kv_width();
    const int hpk = pb.hpk();
    std::vector<uint16_t> out(static_cast<size_t>(pb.rows) * pb.local_heads * D, 0);
    for (int r = 0; r < pb.rows; ++r) {
      const int64_t p = pb.pos[static_cast<size_t>(r)];
      if (p < 0) continue;
      const uint16_t* qr = q_out.data() + static_cast<size_t>(r) * pb.local_heads * D;
      for (int h = 0; h < pb.local_heads; ++h) {
        const int kvh = h / hpk;
        std::vector<float> sc;
        std::vector<int64_t> tok;
        for (int64_t t = 0; t <= p; ++t) {
          if (!pb.visible(r, t)) continue;
          const uint16_t* krow = k_cache.data() + static_cast<size_t>(pb.phys(r, t)) * kvw +
                                 static_cast<size_t>(kvh) * D;
          float acc = 0.f;
          for (int d = 0; d < D; ++d)
            acc = std::fma(dgpp::bf16_bits_to_float(qr[static_cast<size_t>(h) * D + d]),
                           dgpp::bf16_bits_to_float(krow[d]), acc);
          sc.push_back(acc * pb.scale);
          tok.push_back(t);
        }
        float M = -INFINITY;
        for (float s : sc) M = std::max(M, s);
        float L = 0.f;
        std::vector<float> pv(sc.size());
        for (size_t i = 0; i < sc.size(); ++i) {
          const float e = std::exp(sc[i] - M);
          L += e;
          pv[i] = rb(e);
        }
        float c[D] = {0};
        for (size_t i = 0; i < sc.size(); ++i) {
          const uint16_t* vrow = v_cache.data() + static_cast<size_t>(pb.phys(r, tok[i])) * kvw +
                                 static_cast<size_t>(kvh) * D;
          for (int d = 0; d < D; ++d)
            c[d] = std::fma(pv[i], dgpp::bf16_bits_to_float(vrow[d]), c[d]);
        }
        uint16_t* orow = out.data() + (static_cast<size_t>(r) * pb.local_heads + h) * D;
        for (int d = 0; d < D; ++d) orow[d] = rbits(L > 0.f ? c[d] / L : 0.f);
      }
    }
    return out;
  }
};

// ---- the device side ---------------------------------------------------------

struct Device {
  DevBuf q_dot, k_dot, v_dot, sqk, inv_freq, req_ids, pos, table, q_out, k_cache, v_cache, m_ws,
      l_ws, c_ws, out;

  explicit Device(const Problem& pb) {
    q_dot = DevBuf(pb.q_dot.size() * 2);
    k_dot = DevBuf(pb.k_dot.size() * 2);
    v_dot = DevBuf(pb.v_dot.size() * 2);
    sqk = DevBuf(pb.sqk.size() * 2);
    inv_freq = DevBuf(pb.inv_freq.size() * 4);
    req_ids = DevBuf(pb.req_ids.size() * 4);
    pos = DevBuf(pb.pos.size() * 8);
    table = DevBuf(pb.block_tables.size() * 4);
    q_out = DevBuf(static_cast<size_t>(pb.rows) * pb.local_heads * D * 2);
    k_cache = DevBuf(static_cast<size_t>(pb.slots()) * pb.kv_width() * 2);
    v_cache = DevBuf(static_cast<size_t>(pb.slots()) * pb.kv_width() * 2);
    const size_t nsplit = static_cast<size_t>(pb.rows) * pb.n_split * pb.local_heads;
    m_ws = DevBuf(nsplit * 4);
    l_ws = DevBuf(nsplit * 4);
    c_ws = DevBuf(nsplit * D * 4);
    out = DevBuf(static_cast<size_t>(pb.rows) * pb.local_heads * D * 2);
    q_dot.upload(pb.q_dot.data(), pb.q_dot.size() * 2);
    k_dot.upload(pb.k_dot.data(), pb.k_dot.size() * 2);
    v_dot.upload(pb.v_dot.data(), pb.v_dot.size() * 2);
    sqk.upload(pb.sqk.data(), pb.sqk.size() * 2);
    inv_freq.upload(pb.inv_freq.data(), pb.inv_freq.size() * 4);
    req_ids.upload(pb.req_ids.data(), pb.req_ids.size() * 4);
    pos.upload(pb.pos.data(), pb.pos.size() * 8);
    table.upload(pb.block_tables.data(), pb.block_tables.size() * 4);
  }

  void launch(const Problem& pb, cudaStream_t s) {
    dgpp::dory_qkv_finish(
        q_dot.as<const uint16_t>(), static_cast<int64_t>(pb.local_heads) * D, k_dot.as<const uint16_t>(),
        static_cast<int64_t>(pb.kv_width()), v_dot.as<const uint16_t>(),
        static_cast<int64_t>(pb.kv_width()), sqk.as<const uint16_t>(), pb.sqk_factor,
        inv_freq.as<const float>(), pb.eps, req_ids.as<const int32_t>(), pos.as<const int64_t>(),
        pb.rows, pb.local_heads, pb.kv_heads, table.as<const int32_t>(), pb.blocks_per_request,
        pb.block_tokens, q_out.as<uint16_t>(), static_cast<int64_t>(pb.local_heads) * D,
        k_cache.as<uint16_t>(), v_cache.as<uint16_t>(), s);
    dgpp::dory_attn_partial(
        q_out.as<const uint16_t>(), static_cast<int64_t>(pb.local_heads) * D,
        k_cache.as<const uint16_t>(), v_cache.as<const uint16_t>(), req_ids.as<const int32_t>(),
        pos.as<const int64_t>(), pb.rows, pb.n_split, pb.local_heads, pb.kv_heads, pb.block_tokens,
        table.as<const int32_t>(), pb.blocks_per_request, pb.window, pb.scale, m_ws.as<float>(),
        l_ws.as<float>(), c_ws.as<float>(), s);
    dgpp::dory_attn_combine(m_ws.as<const float>(), l_ws.as<const float>(), c_ws.as<const float>(),
                            pb.rows, pb.n_split, pb.local_heads, out.as<uint16_t>(), s);
  }

  void run(const Problem& pb, cudaStream_t s) {
    launch(pb, s);
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
  }

  std::vector<uint16_t> q_host() const {
    std::vector<uint16_t> v(q_out.bytes / 2);
    q_out.download(v.data(), q_out.bytes);
    return v;
  }
  std::vector<uint16_t> k_host() const {
    std::vector<uint16_t> v(k_cache.bytes / 2);
    k_cache.download(v.data(), k_cache.bytes);
    return v;
  }
  std::vector<uint16_t> v_host() const {
    std::vector<uint16_t> v(v_cache.bytes / 2);
    v_cache.download(v.data(), v_cache.bytes);
    return v;
  }
  std::vector<uint16_t> out_host() const {
    std::vector<uint16_t> v(out.bytes / 2);
    out.download(v.data(), out.bytes);
    return v;
  }
};

Problem make_problem(int seed, int rows, int local_heads, int kv_heads, int window, int n_split,
                     int blocks_per_request, int block_tokens, const std::vector<int64_t>& pos,
                     const std::vector<int32_t>& req_ids, const std::vector<int32_t>& table,
                     double theta) {
  Problem pb;
  pb.rows = rows;
  pb.local_heads = local_heads;
  pb.kv_heads = kv_heads;
  pb.window = window;
  pb.n_split = n_split;
  pb.blocks_per_request = blocks_per_request;
  pb.block_tokens = block_tokens;
  pb.pos = pos;
  pb.req_ids = req_ids;
  pb.block_tables = table;
  pb.inv_freq = rope_inv_freq(theta, D);
  pb.sqk = random_bf16_normal(static_cast<uint64_t>(seed) ^ 0x5C, int64_t(local_heads) * D, 0.7f);
  pb.sqk_factor = 0.7f;
  pb.scale = 1.0f / std::sqrt(static_cast<float>(D));
  pb.q_dot = to_bf16(random_f32_uniform(static_cast<uint64_t>(seed) ^ 0x11,
                                        int64_t(rows) * local_heads * D, 1.5f));
  pb.k_dot = to_bf16(random_f32_uniform(static_cast<uint64_t>(seed) ^ 0x22,
                                        int64_t(rows) * kv_heads * D, 1.5f));
  pb.v_dot = to_bf16(random_f32_uniform(static_cast<uint64_t>(seed) ^ 0x33,
                                        int64_t(rows) * kv_heads * D, 1.5f));
  return pb;
}

// A table of `blocks_per_request` distinct blocks per request, permuted so a
// request's slots interleave (any request-relative indexing bug shows up).
std::vector<int32_t> permuted_table(int requests, int blocks_per_request, int seed) {
  std::vector<int32_t> t(static_cast<size_t>(requests) * blocks_per_request);
  int next = 0;
  for (int r = 0; r < requests; ++r)
    for (int b = 0; b < blocks_per_request; ++b) {
      const int perm = (b * 5 + seed * 3) % blocks_per_request;
      t[static_cast<size_t>(r) * blocks_per_request + perm] = next++;
    }
  return t;
}

// One case: the finish, the caches, the attention, against the oracle.
void run_case(const char* what, const Problem& pb, int ulps, double mismatch_frac) {
  const Oracle oracle_ref = [&] {
    Oracle o;
    o.run(pb);
    return o;
  }();
  Device dev(pb);
  dev.run(pb, test_stream());
  const auto out_ref = oracle_ref.attention(pb);
  const auto q_got = dev.q_host(), k_got = dev.k_host(), v_got = dev.v_host();
  const auto out_got = dev.out_host();

  // The V cache: bf16(dot) at the block table's physical slot — bitwise.
  require_bitwise(std::string(what) + ": v cache", v_got.data(), oracle_ref.v_cache.data(),
                  v_got.size() * 2);
  const Stats qs = compare_bf16(q_got, oracle_ref.q_out, ulps);
  const Stats ks = compare_bf16(k_got, oracle_ref.k_cache, ulps);
  const Stats os = compare_bf16(out_got, out_ref, ulps);
  std::printf("  [%s] q max_rel %.2e (mis %ld/%ld), k max_rel %.2e (mis %ld/%ld), out max_rel %.2e "
              "(mis %ld/%ld)\n",
              what, qs.max_rel, qs.mismatches, qs.n, ks.max_rel, ks.mismatches, ks.n, os.max_rel,
              os.mismatches, os.n);
  require_bf16(std::string(what) + ": q", qs, 0.02, mismatch_frac);
  require_bf16(std::string(what) + ": k", ks, 0.02, mismatch_frac);
  require_bf16(std::string(what) + ": out", os, 0.02, mismatch_frac);
}

}  // namespace

DGPP_TEST(dory_attn_matches_the_reference_at_the_model_geometry) {
  // 16 query heads over 4 kv heads, head_dim 256 (plan §1.3's geometry), a
  // 2048-slot pool of 128-token blocks, and a full-history layer.
  const std::vector<int64_t> pos = {0, 1300, 5, 0, 1023, 2047};
  const std::vector<int32_t> req = {0, 0, 1, 1, 2, 2};
  const auto table = permuted_table(3, 16, 1);
  for (int n_split : {1, 2, 4}) {
    const Problem pb = make_problem(101, 6, 16, 4, 0, n_split, 16, 128, pos, req, table, 1e4);
    run_case(n_split == 1 ? "16/4 split1" : n_split == 2 ? "16/4 split2" : "16/4 split4", pb, 2, 0.01);
  }
}

DGPP_TEST(dory_attn_sliding_window_masks_the_old_tokens) {
  // The sliding kind: window 1024 over a 2048-token history. The windowed
  // oracle must match, and the full-history oracle must NOT (so a dropped
  // mask cannot pass), on rows whose visible range is truncated and on rows
  // that are not.
  const std::vector<int64_t> pos = {2047, 1500, 1023, 0, 1100, 1024};
  const std::vector<int32_t> req = {0, 0, 1, 1, 2, 2};
  const auto table = permuted_table(3, 16, 2);
  const Problem pb = make_problem(202, 6, 16, 4, 1024, 2, 16, 128, pos, req, table, 1e6);
  run_case("sliding window 1024", pb, 2, 0.01);

  Oracle full;  // the same problem with no window: the control
  Problem unwindowed = pb;
  unwindowed.window = 0;
  full.run(unwindowed);
  const auto full_out = full.attention(unwindowed);
  Device dev(pb);
  dev.run(pb, test_stream());
  const auto got = dev.out_host();
  const Stats s = compare_bf16(got, full_out, 2);
  std::printf("  [sliding window 1024 vs full history] max_rel %.2e, mismatches %ld/%ld\n",
              s.max_rel, s.mismatches, s.n);
  require(s.mismatches > static_cast<long>(0.05 * static_cast<double>(s.n)),
          "sliding: the window makes no difference — the mask is not applied");
}

DGPP_TEST(dory_attn_generic_head_ratios) {
  // Ratios other than the model's: 8/2 (hpk 4) and 6/3 (hpk 2), plus a pool
  // whose block size is not the tile's multiple-of-32 minimum.
  const std::vector<int64_t> pos = {0, 300, 7, 63};
  const std::vector<int32_t> req = {0, 0, 1, 1};
  const auto table = permuted_table(2, 8, 3);
  for (auto [lh, kv] : {std::pair<int, int>{8, 2}, std::pair<int, int>{6, 3}}) {
    Problem pb = make_problem(303 + lh, 4, lh, kv, 96, 3, 8, 96, pos, req, table, 1e4);
    run_case(lh == 8 ? "8/2 window 96" : "6/3 window 96", pb, 2, 0.01);
  }
}

DGPP_TEST(dory_attn_padding_rows_write_nothing) {
  const std::vector<int64_t> pos = {12, -1, 40};
  const std::vector<int32_t> req = {0, 0, 1};
  const auto table = permuted_table(2, 4, 4);
  const Problem pb = make_problem(404, 3, 16, 4, 0, 1, 4, 64, pos, req, table, 1e4);
  Device dev(pb);
  // A sentinel in q_out's padding row: the finish must not touch it.
  const size_t pad_row = static_cast<size_t>(1) * 16 * D;
  std::vector<uint16_t> sentinel(static_cast<size_t>(pb.rows) * 16 * D, 0x3C00u);
  dev.q_out.upload(sentinel.data(), sentinel.size() * 2);
  dev.run(pb, test_stream());
  const auto q = dev.q_host();
  require_bitwise("padding: q_out row untouched", q.data() + pad_row, sentinel.data() + pad_row,
                  static_cast<size_t>(16) * D * 2);
  const auto out = dev.out_host();
  for (size_t i = pad_row; i < pad_row + static_cast<size_t>(16) * D; ++i)
    if (out[i] != 0)
      throw std::runtime_error("padding: the output row is not zero at " + std::to_string(i));
}

DGPP_TEST(dory_attn_timing) {
  // One decode row at the model's geometry: 16 query heads over 4 kv heads at
  // head_dim 256, a 2048-token history, the three launches timed apart, and
  // the split count swept — the grid is rows × n_split × kv_heads blocks, so
  // n_split is what fills the device for a single row.
  const std::vector<int64_t> pos = {2047};
  const std::vector<int32_t> req = {0};
  const auto table = permuted_table(1, 16, 7);
  cudaStream_t s = test_stream();
  cudaEvent_t e0, e1;
  DGPP_CUDA_OK(cudaEventCreate(&e0));
  DGPP_CUDA_OK(cudaEventCreate(&e1));
  auto time = [&](const std::function<void()>& fn) {
    for (int i = 0; i < 20; ++i) fn();
    float best = 1e30f;
    for (int i = 0; i < 100; ++i) {
      DGPP_CUDA_OK(cudaEventRecord(e0, s));
      fn();
      DGPP_CUDA_OK(cudaEventRecord(e1, s));
      DGPP_CUDA_OK(cudaEventSynchronize(e1));
      float ms = 0.f;
      DGPP_CUDA_OK(cudaEventElapsedTime(&ms, e0, e1));
      best = std::min(best, ms);
    }
    return best * 1e3f;
  };
  for (int n_split : {4, 8, 16, 32}) {
    const Problem pb = make_problem(606, 1, 16, 4, 0, n_split, 16, 128, pos, req, table, 1e4);
    Device dev(pb);
    const float finish = time([&] {
      dgpp::dory_qkv_finish(
          dev.q_dot.as<const uint16_t>(), static_cast<int64_t>(pb.local_heads) * D,
          dev.k_dot.as<const uint16_t>(), static_cast<int64_t>(pb.kv_width()),
          dev.v_dot.as<const uint16_t>(), static_cast<int64_t>(pb.kv_width()),
          dev.sqk.as<const uint16_t>(), pb.sqk_factor, dev.inv_freq.as<const float>(), pb.eps,
          dev.req_ids.as<const int32_t>(), dev.pos.as<const int64_t>(), pb.rows, pb.local_heads,
          pb.kv_heads, dev.table.as<const int32_t>(), pb.blocks_per_request, pb.block_tokens,
          dev.q_out.as<uint16_t>(), static_cast<int64_t>(pb.local_heads) * D,
          dev.k_cache.as<uint16_t>(), dev.v_cache.as<uint16_t>(), s);
    });
    const float partial = time([&] {
      dgpp::dory_attn_partial(
          dev.q_out.as<const uint16_t>(), static_cast<int64_t>(pb.local_heads) * D,
          dev.k_cache.as<const uint16_t>(), dev.v_cache.as<const uint16_t>(),
          dev.req_ids.as<const int32_t>(), dev.pos.as<const int64_t>(), pb.rows, pb.n_split,
          pb.local_heads, pb.kv_heads, pb.block_tokens, dev.table.as<const int32_t>(),
          pb.blocks_per_request, pb.window, pb.scale, dev.m_ws.as<float>(), dev.l_ws.as<float>(),
          dev.c_ws.as<float>(), s);
    });
    const float combine = time([&] {
      dgpp::dory_attn_combine(dev.m_ws.as<const float>(), dev.l_ws.as<const float>(),
                              dev.c_ws.as<const float>(), pb.rows, pb.n_split, pb.local_heads,
                              dev.out.as<uint16_t>(), s);
    });
    std::printf("  [timing] decode 16/4, 2048 tokens, n_split %d: finish %.1f us, partial %.1f us, "
                "combine %.1f us\n",
                n_split, finish, partial, combine);
  }
}

DGPP_TEST(dory_attn_rejects_bad_shapes) {
  const std::vector<int64_t> pos = {0, 1};
  const std::vector<int32_t> req = {0, 0};
  const auto table = permuted_table(1, 4, 5);
  Problem pb = make_problem(505, 2, 4, 2, 0, 1, 4, 64, pos, req, table, 1e4);
  Device dev(pb);
  cudaStream_t s = test_stream();
  auto throws = [](const std::function<void()>& fn) {
    try {
      fn();
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  // block_tokens must be a multiple of the 32-token tile; the head ratios
  // must divide; n_split must be positive.
  require(throws([&] {
            dgpp::dory_qkv_finish(dev.q_dot.as<const uint16_t>(), 4 * D, dev.k_dot.as<const uint16_t>(),
                                  2 * D, dev.v_dot.as<const uint16_t>(), 2 * D,
                                  dev.sqk.as<const uint16_t>(), pb.sqk_factor,
                                  dev.inv_freq.as<const float>(), pb.eps,
                                  dev.req_ids.as<const int32_t>(), dev.pos.as<const int64_t>(),
                                  pb.rows, 4, 2, dev.table.as<const int32_t>(), 4, 40,
                                  dev.q_out.as<uint16_t>(), 4 * D, dev.k_cache.as<uint16_t>(),
                                  dev.v_cache.as<uint16_t>(), s);
          }),
          "qkv_finish accepted block_tokens 40");
  require(throws([&] {
            dgpp::dory_qkv_finish(dev.q_dot.as<const uint16_t>(), 4 * D, dev.k_dot.as<const uint16_t>(),
                                  2 * D, dev.v_dot.as<const uint16_t>(), 2 * D,
                                  dev.sqk.as<const uint16_t>(), pb.sqk_factor,
                                  dev.inv_freq.as<const float>(), pb.eps,
                                  dev.req_ids.as<const int32_t>(), dev.pos.as<const int64_t>(),
                                  pb.rows, 5, 2, dev.table.as<const int32_t>(), 4, 64,
                                  dev.q_out.as<uint16_t>(), 4 * D, dev.k_cache.as<uint16_t>(),
                                  dev.v_cache.as<uint16_t>(), s);
          }),
          "qkv_finish accepted 5 heads over 2");
  require(throws([&] {
            dgpp::dory_attn_partial(dev.q_out.as<const uint16_t>(), 4 * D,
                                    dev.k_cache.as<const uint16_t>(), dev.v_cache.as<const uint16_t>(),
                                    dev.req_ids.as<const int32_t>(), dev.pos.as<const int64_t>(),
                                    pb.rows, 0, 4, 2, 64, dev.table.as<const int32_t>(), 4, 0,
                                    pb.scale, dev.m_ws.as<float>(), dev.l_ws.as<float>(),
                                    dev.c_ws.as<float>(), s);
          }),
          "attn_partial accepted n_split 0");
  require(throws([&] {
            dgpp::dory_attn_combine(nullptr, dev.l_ws.as<const float>(), dev.c_ws.as<const float>(),
                                    pb.rows, 1, 4, dev.out.as<uint16_t>(), s);
          }),
          "attn_combine accepted a null m_ws");
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;
  return dgpp::test::run_all();
}
