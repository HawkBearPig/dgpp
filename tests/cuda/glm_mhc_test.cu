// Parity tests for the mHC module kernels (M4): compute (norm + 24-logit
// projection + pre/post/comb + Sinkhorn + collapse), the stream update, and
// the final mean — each against the double-precision oracle
// (glm_mhc_reference), which mirrors the transformers reference's dtype
// choreography (bf16 rounding points) while accumulating in double.
//
// Budgets reflect the two error sources: the kernel's fp32 sigmoid/softmax/
// Sinkhorn pipeline vs the oracle's double (relative ~1e-6, so bf16 rounding
// may flip by one ulp near boundaries), and accumulated bf16 rounding in the
// update chain. See DESIGN §7.3 for the semantics.
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
#include "kernels/glm_mhc_launch.hpp"
#include "models/glm/mhc.hpp"
#include "models/glm/mhc_reference.hpp"

namespace {

using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;
using dgpp::GlmMhcConfig;
using dgpp::GlmMhcRefResult;
using dgpp::GlmMhcWeights;
using dgpp::GlmMhcWeightsHost;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed | 1) {}
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  double unit() {  // [-1, 1)
    return static_cast<double>(next() >> 11) /
               static_cast<double>(1ull << 52) - 1.0;
  }
};

GlmMhcConfig real_config() {
  GlmMhcConfig c;  // hc_mult 4, hidden 4096, iters 20, eps 1e-6/1e-5
  return c;
}

struct Case {
  GlmMhcConfig cfg;
  int tokens = 0;
  std::vector<uint16_t> streams;
  GlmMhcWeightsHost host_w;
  GlmMhcWeights dev_w;
  std::vector<uint16_t> sublayer_out;

  // Device buffers.
  uint16_t* d_streams = nullptr;
  uint16_t* d_fn = nullptr;
  float* d_base = nullptr;
  float* d_scale = nullptr;
  uint16_t* d_collapsed = nullptr;
  uint16_t* d_post = nullptr;
  float* d_logits = nullptr;  // launch_mhc_compute's dots scratch
  uint16_t* d_comb = nullptr;
  uint16_t* d_sub = nullptr;
  uint16_t* d_streams_out = nullptr;
  uint16_t* d_mean = nullptr;

  void alloc() {
    const int n = cfg.hc_mult, D = cfg.hidden;
    DGPP_CUDA_OK(cudaMallocManaged(&d_streams, streams.size() * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&d_fn, host_w.fn.size() * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&d_base, host_w.base.size() * 4));
    DGPP_CUDA_OK(cudaMallocManaged(&d_scale, 3 * 4));
    DGPP_CUDA_OK(cudaMallocManaged(&d_collapsed, static_cast<size_t>(tokens) * D * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&d_post, static_cast<size_t>(tokens) * n * 2));
    DGPP_CUDA_OK(cudaMallocManaged(
        &d_logits, static_cast<size_t>(tokens) * cfg.coeff_rows() * 4));
    DGPP_CUDA_OK(cudaMallocManaged(&d_comb, static_cast<size_t>(tokens) * n * n * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&d_sub, sublayer_out.size() * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&d_streams_out, streams.size() * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&d_mean, static_cast<size_t>(tokens) * D * 2));
    std::memcpy(d_streams, streams.data(), streams.size() * 2);
    std::memcpy(d_fn, host_w.fn.data(), host_w.fn.size() * 2);
    std::memcpy(d_base, host_w.base.data(), host_w.base.size() * 4);
    std::memcpy(d_scale, host_w.scale.data(), 3 * 4);
    std::memcpy(d_sub, sublayer_out.data(), sublayer_out.size() * 2);
    dev_w = GlmMhcWeights{d_fn, d_base, d_scale};
  }
  void free_all() {
    cudaFree(d_streams); cudaFree(d_fn); cudaFree(d_base); cudaFree(d_scale);
    cudaFree(d_collapsed); cudaFree(d_post); cudaFree(d_logits);
    cudaFree(d_comb); cudaFree(d_sub);
    cudaFree(d_streams_out); cudaFree(d_mean);
  }
};

Case make_case(const GlmMhcConfig& cfg, int tokens, uint64_t seed,
               bool extreme_logits = false) {
  Case c;
  c.cfg = cfg;
  c.tokens = tokens;
  const int n = cfg.hc_mult, D = cfg.hidden;
  const int coeffs = cfg.coeff_rows();
  Rng rng(seed);

  c.streams.resize(static_cast<size_t>(tokens) * n * D);
  for (auto& v : c.streams)
    v = float_to_bf16_bits(0.25f * static_cast<float>(rng.unit()));
  c.sublayer_out.resize(static_cast<size_t>(tokens) * D);
  for (auto& v : c.sublayer_out)
    v = float_to_bf16_bits(0.5f * static_cast<float>(rng.unit()));

  c.host_w.fn.resize(static_cast<size_t>(coeffs) * n * D);
  for (auto& v : c.host_w.fn)
    v = float_to_bf16_bits(0.02f * static_cast<float>(rng.unit()));
  c.host_w.base.resize(coeffs);
  for (auto& v : c.host_w.base)
    v = extreme_logits ? (rng.unit() > 0 ? 40.f : -40.f)
                       : 0.1f * static_cast<float>(rng.unit());
  c.host_w.scale.resize(3);
  for (auto& v : c.host_w.scale)
    v = static_cast<float>(std::exp(rng.unit()));
  return c;
}

struct UlpReport {
  long over = 0;       // beyond soft budget
  long hard = 0;       // beyond hard budget
  size_t total = 0;
  double max_ulps = 0;
};

// Ulp comparison with a cancellation floor: streams mix positive and
// negative values, so a sum of ~0.25-magnitude terms can cancel to ~1e-8,
// where the bf16 ulp (~4e-11) turns absolutely-tiny fp32-vs-double noise
// (bounded by the fp32 epsilon of the LARGE terms) into dozens of "ulps".
// Elements below floor_frac * max|want| are measured at the floor's
// magnitude instead (the same discipline as scale_gemm's comparator).
UlpReport compare(const std::vector<uint16_t>& got,
                  const std::vector<uint16_t>& want, int soft, int hard,
                  double floor_frac = 1e-3) {
  UlpReport r;
  r.total = want.size();
  double max_abs = 0;
  for (uint16_t v : want)
    max_abs = std::max(max_abs,
                       std::abs(static_cast<double>(bf16_bits_to_float(v))));
  const double floor_abs = floor_frac * max_abs;
  for (size_t i = 0; i < want.size(); ++i) {
    const double w = bf16_bits_to_float(want[i]);
    const double g = bf16_bits_to_float(got[i]);
    // bf16-ulp distance measured at the larger of |want| and the floor.
    const double scale = std::max(std::abs(w), floor_abs);
    const double ulp = scale > 0 ? scale * (1.0 / 256.0) : (1.0 / 256.0);
    const double dist = std::abs(g - w) / ulp;
    const int u = static_cast<int>(dist);
    r.max_ulps = std::max(r.max_ulps, dist);
    if (u > soft) ++r.over;
    if (u > hard) ++r.hard;
  }
  return r;
}

// hard_budget_count: chained-rounding boundary flips. When an intermediate
// (e.g. post*h at magnitude ~1) rounds differently by one bf16 ulp between
// the fp32 kernel and the double oracle — a 1-fp32-ulp product difference
// straddling a rounding boundary, expected at ~1e-7 per element — the
// output inherits that flip as ~1 ulp of the INTERMEDIATE's magnitude,
// which the output-ulp metric over-reports when the output is smaller.
// Budget: soft ulps at over_frac, hard outliers at a small absolute count.
void require_ulp(const UlpReport& r, double over_frac, long hard_budget_count,
                 const std::string& what) {
  if (r.hard > hard_budget_count ||
      static_cast<double>(r.over) / static_cast<double>(r.total) > over_frac)
    throw std::runtime_error(what + ": over=" +
                             std::to_string(r.over) + "/" +
                             std::to_string(r.total) +
                             " hard=" + std::to_string(r.hard) +
                             " max=" + std::to_string(r.max_ulps) + " ulps");
}

std::vector<uint16_t> read_back(const uint16_t* dev, size_t n) {
  std::vector<uint16_t> v(n);
  std::memcpy(v.data(), dev, n * 2);
  return v;
}

// Full pipeline on one case: compute + (isolated) update + mean.
void run_case(const Case& c, const char* label) {
  GlmMhcRefResult ref;
  dgpp::glm_mhc_ref_compute(c.d_streams, c.host_w, c.cfg, c.tokens, ref);

  dgpp::launch_mhc_compute(c.d_streams, c.dev_w, c.cfg, c.d_collapsed,
                           c.d_post, c.d_comb, c.d_logits, c.tokens, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());

  // post/comb: oracle rounded to bf16 (the choreography point).
  std::vector<uint16_t> post_ref(c.tokens * c.cfg.hc_mult);
  std::vector<uint16_t> comb_ref(c.tokens * c.cfg.hc_mult *
                                 c.cfg.hc_mult);
  for (size_t i = 0; i < post_ref.size(); ++i)
    post_ref[i] = float_to_bf16_bits(static_cast<float>(ref.post[i]));
  for (size_t i = 0; i < comb_ref.size(); ++i)
    comb_ref[i] = float_to_bf16_bits(static_cast<float>(ref.comb[i]));
  require_ulp(compare(read_back(c.d_post, post_ref.size()), post_ref, 2, 4),
              0.005, 0, std::string(label) + " post");
  require_ulp(compare(read_back(c.d_comb, comb_ref.size()), comb_ref, 2, 4),
              0.005, 0, std::string(label) + " comb");
  require_ulp(compare(read_back(c.d_collapsed, ref.collapsed.size()),
                      ref.collapsed, 2, 4),
              0.005, 0, std::string(label) + " collapsed");

  // Isolated update: both sides consume the same (oracle) bf16 post/comb.
  std::vector<uint16_t> streams_ref(c.streams.size());
  dgpp::glm_mhc_ref_stream_update(post_ref.data(), comb_ref.data(),
                                  c.sublayer_out.data(), c.streams.data(),
                                  c.cfg, c.tokens, streams_ref.data());
  std::memcpy(c.d_post, post_ref.data(), post_ref.size() * 2);
  std::memcpy(c.d_comb, comb_ref.data(), comb_ref.size() * 2);
  dgpp::launch_mhc_stream_update(c.d_post, c.d_comb, c.d_sub, c.d_streams,
                                 c.d_streams_out, c.cfg, c.tokens, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  // Update chain: allow ~1e-6 rate of chained-rounding boundary flips
  // (post*h and the 4-term mix each round to bf16 before the final add).
  const long flip_budget =
      static_cast<long>(streams_ref.size()) / 1000000 + 1;
  require_ulp(compare(read_back(c.d_streams_out, streams_ref.size()),
                      streams_ref, 4, 8),
              0.01, flip_budget, std::string(label) + " stream update");

  // Final mean.
  std::vector<uint16_t> mean_ref(c.tokens * c.cfg.hidden);
  dgpp::glm_mhc_ref_final_mean(c.streams.data(), c.cfg, c.tokens,
                               mean_ref.data());
  dgpp::launch_mhc_final_mean(c.d_streams, c.d_mean, c.cfg, c.tokens,
                              nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  require_ulp(compare(read_back(c.d_mean, mean_ref.size()), mean_ref, 2, 4),
              0.005, 0, std::string(label) + " final mean");

  std::printf("[ OK ] %s: post/comb/collapsed/update/mean within budgets\n",
              label);
}

}  // namespace

DGPP_TEST(mhc_compute_update_mean_match_oracle_real_geometry) {
  for (int tokens : {1, 3, 17, 257, 2052}) {
    Case c = make_case(real_config(), tokens, 0xABCDEF + tokens);
    c.alloc();
    run_case(c, ("real-geometry tokens=" + std::to_string(tokens)).c_str());
    c.free_all();
  }
}

DGPP_TEST(mhc_small_hidden_and_extreme_logits) {
  {
    GlmMhcConfig cfg;
    cfg.hidden = 512;
    Case c = make_case(cfg, 17, 0x1234);
    c.alloc();
    run_case(c, "hidden=512 tokens=17");
    c.free_all();
  }
  {
    // Saturated sigmoids (|base| = 40): pre -> {eps, 1+eps}, post -> {0, 2},
    // softmax rows peaked — the Sinkhorn loop must stay finite and the
    // budgets must still hold.
    Case c = make_case(real_config(), 33, 0x5678, /*extreme_logits=*/true);
    c.alloc();
    run_case(c, "saturated logits");
    c.free_all();
  }
}

DGPP_TEST(mhc_zero_streams_survive_norm_of_zero) {
  // All-zero streams: sumsq = 0, inv_rms = rsqrt(eps) = large; flat = 0, so
  // logits = 0 and everything downstream is well-defined. A NaN or inf
  // anywhere fails the oracle comparison.
  Case c = make_case(real_config(), 5, 0x9ABC);
  std::fill(c.streams.begin(), c.streams.end(), 0);
  c.alloc();
  run_case(c, "zero streams");
  c.free_all();
}

// The prefill's token-tiled dots form against the per-coefficient form on
// the same inputs (real geometry, 70 tokens: 17 full tiles and a ragged
// one): collapsed, post, comb and the fused-norm output bitwise — the
// tiled form keeps every thread's element order and the same reductions,
// so this is an equality, not a budget. The normed form (ln given) and the
// two-launch form (no ln) both.
DGPP_TEST(mhc_tiled_prefill_form_is_bitwise_the_per_coefficient_form) {
  dgpp::mhc_set_prefill_gemm(false);  // the token-tiled kernel, not the GEMM form
  Case c = make_case(real_config(), 70, 0x71E5);
  c.alloc();
  const int n = c.cfg.hc_mult, D = c.cfg.hidden, T = c.tokens;
  std::vector<uint16_t> ln(D);
  for (int i = 0; i < D; ++i) ln[i] = float_to_bf16_bits(0.5f + 0.001f * float(i % 97));
  uint16_t* d_ln = nullptr;
  uint16_t *d_normed_a = nullptr, *d_normed_b = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&d_ln, D * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_normed_a, size_t(T) * D * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_normed_b, size_t(T) * D * 2));
  std::memcpy(d_ln, ln.data(), D * 2);
  std::vector<uint16_t> col_a(size_t(T) * D), post_a(size_t(T) * n), comb_a(size_t(T) * n * n);
  const auto require = [](bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
  };
  for (int form = 0; form < 2; ++form) {
    // form 0: the normed (fused-finish) call; form 1: the two-launch call.
    for (int tiled = 0; tiled < 2; ++tiled) {
      dgpp::mhc_set_tiled_form(tiled == 1);
      DGPP_CUDA_OK(cudaMemset(c.d_collapsed, 0xA5, size_t(T) * D * 2));
      DGPP_CUDA_OK(cudaMemset(c.d_post, 0xA5, size_t(T) * n * 2));
      DGPP_CUDA_OK(cudaMemset(c.d_comb, 0xA5, size_t(T) * n * n * 2));
      uint16_t* normed = tiled ? d_normed_b : d_normed_a;
      DGPP_CUDA_OK(cudaMemset(normed, 0xA5, size_t(T) * D * 2));
      if (form == 0)
        dgpp::launch_mhc_compute_normed(c.d_streams, c.dev_w, c.cfg, c.d_collapsed, c.d_post,
                                        c.d_comb, c.d_logits, d_ln, normed, 1e-5f, T, nullptr,
                                        nullptr);
      else
        dgpp::launch_mhc_compute(c.d_streams, c.dev_w, c.cfg, c.d_collapsed, c.d_post, c.d_comb,
                                 c.d_logits, T, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      if (!tiled) {
        std::memcpy(col_a.data(), c.d_collapsed, col_a.size() * 2);
        std::memcpy(post_a.data(), c.d_post, post_a.size() * 2);
        std::memcpy(comb_a.data(), c.d_comb, comb_a.size() * 2);
      } else {
        require(std::memcmp(col_a.data(), c.d_collapsed, col_a.size() * 2) == 0,
                "tiled mhc collapsed bitwise the per-coefficient form");
        require(std::memcmp(post_a.data(), c.d_post, post_a.size() * 2) == 0,
                "tiled mhc post bitwise the per-coefficient form");
        require(std::memcmp(comb_a.data(), c.d_comb, comb_a.size() * 2) == 0,
                "tiled mhc comb bitwise the per-coefficient form");
        if (form == 0)
          require(std::memcmp(d_normed_a, d_normed_b, size_t(T) * D * 2) == 0,
                  "tiled mhc normed bitwise the per-coefficient form");
      }
    }
    std::printf("[ OK ] mhc tiled form (%s): %d tokens bitwise the per-coefficient form\n",
                form == 0 ? "normed" : "two-launch", T);
  }
  dgpp::mhc_set_tiled_form(true);
  dgpp::mhc_set_prefill_gemm(true);
  cudaFree(d_ln); cudaFree(d_normed_a); cudaFree(d_normed_b);
  c.free_all();
}

// The tensor-core prefill form against the double oracle on real geometry
// (the default for >= 16 tokens; the oracle test above already runs it at
// 17 / 257 / 2052 tokens — this one names it and also pins its normed
// output against the tiled form's within the same budgets).
DGPP_TEST(mhc_gemm_prefill_form_within_oracle_budgets) {
  for (int tokens : {16, 70, 2052}) {
    Case c = make_case(real_config(), tokens, 0x6E44 + tokens);
    c.alloc();
    dgpp::mhc_set_prefill_gemm(true);
    run_case(c, ("gemm prefill form tokens=" + std::to_string(tokens)).c_str());
    // Against the tiled form on the same inputs: collapsed within 2 ulps
    // (bf16), post/comb likewise — the two differ by fp32 rounding only.
    std::vector<uint16_t> col_g = read_back(c.d_collapsed, static_cast<size_t>(tokens) * c.cfg.hidden);
    std::vector<uint16_t> post_g = read_back(c.d_post, static_cast<size_t>(tokens) * c.cfg.hc_mult);
    dgpp::mhc_set_prefill_gemm(false);
    dgpp::launch_mhc_compute(c.d_streams, c.dev_w, c.cfg, c.d_collapsed, c.d_post, c.d_comb,
                             c.d_logits, tokens, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    dgpp::mhc_set_prefill_gemm(true);
    require_ulp(compare(col_g, read_back(c.d_collapsed, col_g.size()), 2, 4), 0.005, 0,
                "gemm vs tiled collapsed");
    require_ulp(compare(post_g, read_back(c.d_post, post_g.size()), 2, 4), 0.005, 0,
                "gemm vs tiled post");
    c.free_all();
  }
}

// The deferred comb (decode's side-stream form) against the in-block fused
// finish on the same inputs at decode row counts: collapsed, post, normed
// and comb bitwise — the same arithmetic on the same lanes, so an
// equality, not a budget.
DGPP_TEST(mhc_deferred_comb_is_bitwise_the_fused_finish) {
  for (int T : {1, 2, 8}) {
    Case c = make_case(real_config(), T, 0x5EED + T);
    c.alloc();
    const int n = c.cfg.hc_mult, D = c.cfg.hidden;
    std::vector<uint16_t> ln(D);
    for (int i = 0; i < D; ++i) ln[i] = float_to_bf16_bits(0.5f + 0.001f * float(i % 97));
    uint16_t* d_ln = nullptr;
    uint16_t* d_normed = nullptr;
    int* d_counters = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&d_ln, D * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&d_normed, size_t(T) * D * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&d_counters, size_t(T) * sizeof(int)));
    std::memcpy(d_ln, ln.data(), D * 2);
    std::memset(d_counters, 0, size_t(T) * sizeof(int));
    std::vector<uint16_t> col_a(size_t(T) * D), post_a(size_t(T) * n), comb_a(size_t(T) * n * n),
        normed_a(size_t(T) * D);
    const auto require = [](bool ok, const char* what) {
      if (!ok) throw std::runtime_error(what);
    };
    for (int deferred = 0; deferred < 2; ++deferred) {
      DGPP_CUDA_OK(cudaMemset(c.d_collapsed, 0xA5, size_t(T) * D * 2));
      DGPP_CUDA_OK(cudaMemset(c.d_post, 0xA5, size_t(T) * n * 2));
      DGPP_CUDA_OK(cudaMemset(c.d_comb, 0xA5, size_t(T) * n * n * 2));
      DGPP_CUDA_OK(cudaMemset(d_normed, 0xA5, size_t(T) * D * 2));
      dgpp::launch_mhc_compute_normed(c.d_streams, c.dev_w, c.cfg, c.d_collapsed, c.d_post,
                                      c.d_comb, c.d_logits, d_ln, d_normed, 1e-5f, T, nullptr,
                                      d_counters, deferred == 1);
      if (deferred) {
        // comb untouched by the finish; the standalone launch writes it.
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        for (size_t i = 0; i < size_t(T) * n * n; ++i)
          require(c.d_comb[i] == 0xA5A5, "deferred finish must not write comb");
        dgpp::launch_mhc_comb(c.d_logits, c.dev_w, c.cfg, c.d_comb, T, nullptr);
      }
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      if (!deferred) {
        std::memcpy(col_a.data(), c.d_collapsed, col_a.size() * 2);
        std::memcpy(post_a.data(), c.d_post, post_a.size() * 2);
        std::memcpy(comb_a.data(), c.d_comb, comb_a.size() * 2);
        std::memcpy(normed_a.data(), d_normed, normed_a.size() * 2);
      } else {
        require(std::memcmp(col_a.data(), c.d_collapsed, col_a.size() * 2) == 0,
                "deferred-comb collapsed bitwise the fused finish");
        require(std::memcmp(post_a.data(), c.d_post, post_a.size() * 2) == 0,
                "deferred-comb post bitwise the fused finish");
        require(std::memcmp(comb_a.data(), c.d_comb, comb_a.size() * 2) == 0,
                "deferred comb bitwise the in-block Sinkhorn");
        require(std::memcmp(normed_a.data(), d_normed, normed_a.size() * 2) == 0,
                "deferred-comb normed bitwise the fused finish");
      }
    }
    std::printf("[ OK ] mhc deferred comb: %d tokens bitwise the fused finish\n", T);
    cudaFree(d_ln); cudaFree(d_normed); cudaFree(d_counters);
    c.free_all();
  }
}

// A decode batch of sixteen rows and more takes the token-tiled dots
// (2026-10-23): with the fused finish, the deferred comb and the
// single-pass exports as the DeepSeek decode site asks them, the tiled
// form's logits, post, normed, pre and comb are bitwise the per-
// coefficient fused form's (mhc_set_tiled_form flips the form).
DGPP_TEST(mhc_decode_rows_tiled_form_is_bitwise_the_fused_form) {
  for (int T : {16, 20, 40}) {
    Case c = make_case(real_config(), T, 0xD3C0 + T);
    c.alloc();
    const int n = c.cfg.hc_mult, D = c.cfg.hidden;
    std::vector<uint16_t> ln(D);
    for (int i = 0; i < D; ++i) ln[i] = float_to_bf16_bits(0.5f + 0.001f * float(i % 97));
    uint16_t* d_ln = nullptr;
    uint16_t* d_normed = nullptr;
    int* d_counters = nullptr;
    float *d_pre_in = nullptr, *d_pre_out = nullptr, *d_post_f32 = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&d_ln, D * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&d_normed, size_t(T) * D * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&d_counters, size_t(T) * sizeof(int)));
    DGPP_CUDA_OK(cudaMallocManaged(&d_pre_in, size_t(T) * n * sizeof(float)));
    DGPP_CUDA_OK(cudaMallocManaged(&d_pre_out, size_t(T) * n * sizeof(float)));
    DGPP_CUDA_OK(cudaMallocManaged(&d_post_f32, size_t(T) * n * sizeof(float)));
    std::memcpy(d_ln, ln.data(), D * 2);
    std::memset(d_counters, 0, size_t(T) * sizeof(int));
    for (int i = 0; i < T * n; ++i) d_pre_in[i] = 0.25f + 0.01f * float(i % 13);
    std::vector<uint16_t> post_a(size_t(T) * n), comb_a(size_t(T) * n * n), normed_a(size_t(T) * D);
    std::vector<float> logits_a(size_t(T) * c.cfg.coeff_rows()), pre_a(size_t(T) * n), postf_a(size_t(T) * n);
    const auto require = [](bool ok, const char* what) {
      if (!ok) throw std::runtime_error(what);
    };
    for (int tiled = 0; tiled < 2; ++tiled) {
      dgpp::mhc_set_tiled_form(tiled == 1);
      DGPP_CUDA_OK(cudaMemset(c.d_post, 0xA5, size_t(T) * n * 2));
      DGPP_CUDA_OK(cudaMemset(c.d_comb, 0xA5, size_t(T) * n * n * 2));
      DGPP_CUDA_OK(cudaMemset(d_normed, 0xA5, size_t(T) * D * 2));
      DGPP_CUDA_OK(cudaMemset(d_pre_out, 0, size_t(T) * n * sizeof(float)));
      DGPP_CUDA_OK(cudaMemset(d_post_f32, 0, size_t(T) * n * sizeof(float)));
      dgpp::MhcSinglePass sp;
      sp.pre_in = d_pre_in;
      sp.pre_out = d_pre_out;
      sp.post_f32 = d_post_f32;
      sp.comb_f32 = nullptr;
      sp.one_round_norm = true;
      const bool deferred = dgpp::launch_mhc_compute_normed(c.d_streams, c.dev_w, c.cfg, nullptr, c.d_post, c.d_comb,
                                                            c.d_logits, d_ln, d_normed, 1e-5f, T, nullptr, d_counters,
                                                            /*defer_comb=*/true, &sp, /*decode_rows=*/true);
      require(deferred, "a decode batch defers its comb on either form");
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      for (size_t i = 0; i < size_t(T) * n * n; ++i) require(c.d_comb[i] == 0xA5A5, "deferred finish must not write comb");
      dgpp::launch_mhc_comb(c.d_logits, c.dev_w, c.cfg, c.d_comb, T, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      if (!tiled) {
        std::memcpy(post_a.data(), c.d_post, post_a.size() * 2);
        std::memcpy(comb_a.data(), c.d_comb, comb_a.size() * 2);
        std::memcpy(normed_a.data(), d_normed, normed_a.size() * 2);
        std::memcpy(logits_a.data(), c.d_logits, logits_a.size() * 4);
        std::memcpy(pre_a.data(), d_pre_out, pre_a.size() * 4);
        std::memcpy(postf_a.data(), d_post_f32, postf_a.size() * 4);
      } else {
        require(std::memcmp(logits_a.data(), c.d_logits, logits_a.size() * 4) == 0, "decode tiled logits bitwise the fused form");
        require(std::memcmp(post_a.data(), c.d_post, post_a.size() * 2) == 0, "decode tiled post bitwise the fused form");
        require(std::memcmp(normed_a.data(), d_normed, normed_a.size() * 2) == 0, "decode tiled normed bitwise the fused form");
        require(std::memcmp(pre_a.data(), d_pre_out, pre_a.size() * 4) == 0, "decode tiled pre export bitwise the fused form");
        require(std::memcmp(postf_a.data(), d_post_f32, postf_a.size() * 4) == 0, "decode tiled post_f32 bitwise the fused form");
        require(std::memcmp(comb_a.data(), c.d_comb, comb_a.size() * 2) == 0, "decode tiled deferred comb bitwise the fused form");
      }
    }
    dgpp::mhc_set_tiled_form(true);
    std::printf("[ OK ] mhc decode tiled form: %d rows bitwise the fused per-coefficient form\n", T);
    cudaFree(d_ln); cudaFree(d_normed); cudaFree(d_counters); cudaFree(d_pre_in); cudaFree(d_pre_out); cudaFree(d_post_f32);
    c.free_all();
  }
}

DGPP_TEST(mhc_end_to_end_pipeline_is_deterministic) {
  // Real pipeline wiring: kernel's own post/comb feed its own update; two
  // full invocations must be bitwise identical (graph-capture premise).
  Case c = make_case(real_config(), 64, 0xDEF0);
  c.alloc();
  auto pipeline = [&] {
    dgpp::launch_mhc_compute(c.d_streams, c.dev_w, c.cfg, c.d_collapsed,
                             c.d_post, c.d_comb, c.d_logits, c.tokens, nullptr);
    dgpp::launch_mhc_stream_update(c.d_post, c.d_comb, c.d_sub, c.d_streams,
                                   c.d_streams_out, c.cfg, c.tokens,
                                   nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    return read_back(c.d_streams_out, c.streams.size());
  };
  const auto a = pipeline();
  const auto b = pipeline();
  if (std::memcmp(a.data(), b.data(), a.size() * 2) != 0)
    throw std::runtime_error("pipeline is not bitwise deterministic");

  // And the end-to-end pipeline stays within budgets of the oracle pipeline
  // (oracle compute -> oracle-rounded post/comb -> oracle update).
  GlmMhcRefResult ref;
  dgpp::glm_mhc_ref_compute(c.d_streams, c.host_w, c.cfg, c.tokens, ref);
  std::vector<uint16_t> post_ref(c.tokens * c.cfg.hc_mult);
  std::vector<uint16_t> comb_ref(c.tokens * c.cfg.hc_mult * c.cfg.hc_mult);
  for (size_t i = 0; i < post_ref.size(); ++i)
    post_ref[i] = float_to_bf16_bits(static_cast<float>(ref.post[i]));
  for (size_t i = 0; i < comb_ref.size(); ++i)
    comb_ref[i] = float_to_bf16_bits(static_cast<float>(ref.comb[i]));
  std::vector<uint16_t> streams_ref(c.streams.size());
  dgpp::glm_mhc_ref_stream_update(post_ref.data(), comb_ref.data(),
                                  c.sublayer_out.data(), c.streams.data(),
                                  c.cfg, c.tokens, streams_ref.data());
  const long flip_budget =
      static_cast<long>(streams_ref.size()) / 1000000 + 1;
  require_ulp(compare(a, streams_ref, 4, 8), 0.01, flip_budget,
              "end-to-end streams");
  std::printf("[ OK ] end-to-end pipeline deterministic + within budgets\n");
}

// ---- the single-pass form (2026-09-13, DeepSeek-V4.1-Flash, plan D4) --------

DGPP_TEST(mhc_single_pass_exports_pre_and_collapses_with_the_given_pre) {
  // Site A computes its own coefficients and exports pre / post / comb in
  // fp32; site B collapses with A's pre (the shifted coefficients) while
  // predicting its own. Both against the oracle; B's own pre unchanged.
  for (int tokens : {1, 3, 17, 70}) {
    // Distinct seeds after the Rng's `| 1` (0x51A + 70 and 0x51B + 70 collide).
    Case a = make_case(real_config(), tokens, 0xA000 + 2 * tokens);
    Case b = make_case(real_config(), tokens, 0xB000 + 2 * tokens);
    a.alloc();
    b.alloc();
    const int n = a.cfg.hc_mult;
    float *pre_a = nullptr, *pre_b = nullptr, *post_a = nullptr, *comb_a = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&pre_a, static_cast<size_t>(tokens) * n * 4));
    DGPP_CUDA_OK(cudaMallocManaged(&pre_b, static_cast<size_t>(tokens) * n * 4));
    DGPP_CUDA_OK(cudaMallocManaged(&post_a, static_cast<size_t>(tokens) * n * 4));
    DGPP_CUDA_OK(cudaMallocManaged(&comb_a, static_cast<size_t>(tokens) * n * n * 4));
    dgpp::MhcSinglePass sp_a;
    sp_a.pre_out = pre_a;
    sp_a.post_f32 = post_a;
    sp_a.comb_f32 = comb_a;
    dgpp::launch_mhc_compute_normed(a.d_streams, a.dev_w, a.cfg, a.d_collapsed, a.d_post, a.d_comb,
                                    a.d_logits, nullptr, nullptr, 0.f, tokens, nullptr, nullptr,
                                    false, &sp_a);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    GlmMhcRefResult ref_a;
    dgpp::glm_mhc_ref_compute(a.d_streams, a.host_w, a.cfg, tokens, ref_a);
    for (size_t i = 0; i < ref_a.pre.size(); ++i) {
      require(std::fabs(pre_a[i] - ref_a.pre[i]) <= 1e-5 * std::max(1.0, std::fabs(ref_a.pre[i])),
              "exported pre within fp32 of the oracle");
      require(std::fabs(post_a[i] - ref_a.post[i]) <= 1e-5 * std::max(1.0, std::fabs(ref_a.post[i])),
              "exported post within fp32 of the oracle");
    }
    for (size_t i = 0; i < ref_a.comb.size(); ++i)
      require(std::fabs(comb_a[i] - ref_a.comb[i]) <= 1e-5 * std::max(1.0, std::fabs(ref_a.comb[i])),
              "exported comb within fp32 of the oracle");
    // Site B with A's pre.
    dgpp::MhcSinglePass sp_b;
    sp_b.pre_in = pre_a;
    sp_b.pre_out = pre_b;
    dgpp::launch_mhc_compute_normed(b.d_streams, b.dev_w, b.cfg, b.d_collapsed, b.d_post, b.d_comb,
                                    b.d_logits, nullptr, nullptr, 0.f, tokens, nullptr, nullptr,
                                    false, &sp_b);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<double> pre_a_d(ref_a.pre.size());
    for (size_t i = 0; i < pre_a_d.size(); ++i) pre_a_d[i] = pre_a[i];
    GlmMhcRefResult ref_b;
    dgpp::glm_mhc_ref_compute(b.d_streams, b.host_w, b.cfg, tokens, ref_b, pre_a_d.data());
    require_ulp(compare(read_back(b.d_collapsed, ref_b.collapsed.size()), ref_b.collapsed, 2, 4),
                0.005, 0, "single-pass collapse with the previous site's pre");
    for (size_t i = 0; i < ref_b.pre.size(); ++i)
      require(std::fabs(pre_b[i] - ref_b.pre[i]) <= 1e-5 * std::max(1.0, std::fabs(ref_b.pre[i])),
              "site B's own pre exported unchanged by pre_in");
    // The GLM form on the same site is the pre_in-less call: its collapse
    // uses B's own pre and differs from the shifted one.
    std::vector<uint16_t> shifted = read_back(b.d_collapsed, ref_b.collapsed.size());
    dgpp::launch_mhc_compute(b.d_streams, b.dev_w, b.cfg, b.d_collapsed, b.d_post, b.d_comb, b.d_logits,
                             tokens, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    require(read_back(b.d_collapsed, ref_b.collapsed.size()) != shifted,
            "the shifted collapse differs from the site's own");
    std::printf("[ OK ] single-pass mHC tokens=%d: pre/post/comb exported, collapse with pre_in within budget\n",
                tokens);
    cudaFree(pre_a); cudaFree(pre_b); cudaFree(post_a); cudaFree(comb_a);
    a.free_all();
    b.free_all();
  }
}

DGPP_TEST(mhc_single_round_update_matches_oracle) {
  // The fp32 post/comb from the oracle on both sides; one rounding, so the
  // budget is the plain fp32-vs-double one (2 ulps, no chained flips).
  for (int tokens : {1, 5, 33}) {
    Case c = make_case(real_config(), tokens, 0x5F32 + tokens);
    c.alloc();
    const int n = c.cfg.hc_mult;
    GlmMhcRefResult ref;
    dgpp::glm_mhc_ref_compute(c.d_streams, c.host_w, c.cfg, tokens, ref);
    float *post = nullptr, *comb = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&post, static_cast<size_t>(tokens) * n * 4));
    DGPP_CUDA_OK(cudaMallocManaged(&comb, static_cast<size_t>(tokens) * n * n * 4));
    std::vector<double> post_d(ref.post.size()), comb_d(ref.comb.size());
    for (size_t i = 0; i < post_d.size(); ++i) post_d[i] = post[i] = static_cast<float>(ref.post[i]);
    for (size_t i = 0; i < comb_d.size(); ++i) comb_d[i] = comb[i] = static_cast<float>(ref.comb[i]);
    std::vector<uint16_t> want(c.streams.size());
    dgpp::glm_mhc_ref_stream_update_f32(post_d.data(), comb_d.data(), c.sublayer_out.data(),
                                        c.streams.data(), c.cfg, tokens, want.data());
    dgpp::launch_mhc_stream_update_f32(post, comb, c.d_sub, c.d_streams, c.d_streams_out, c.cfg, tokens,
                                       nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    require_ulp(compare(read_back(c.d_streams_out, want.size()), want, 2, 4), 0.005, 0,
                "single-round stream update");
    std::printf("[ OK ] single-round mHC update tokens=%d within budget\n", tokens);
    cudaFree(post); cudaFree(comb);
    c.free_all();
  }
}

DGPP_TEST(mhc_collapse_normed_is_bitwise_the_finish_with_the_same_pre) {
  for (int tokens : {1, 7, 40}) {
    Case c = make_case(real_config(), tokens, 0xC011 + tokens);
    c.alloc();
    const int n = c.cfg.hc_mult, D = c.cfg.hidden;
    float* pre = nullptr;
    uint16_t *ln = nullptr, *normed_a = nullptr, *normed_b = nullptr, *collapsed_b = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&pre, static_cast<size_t>(tokens) * n * 4));
    DGPP_CUDA_OK(cudaMallocManaged(&ln, static_cast<size_t>(D) * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&normed_a, static_cast<size_t>(tokens) * D * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&normed_b, static_cast<size_t>(tokens) * D * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&collapsed_b, static_cast<size_t>(tokens) * D * 2));
    Rng rng(0x1A + tokens);
    for (int i = 0; i < tokens * n; ++i) pre[i] = 0.3f + 0.5f * static_cast<float>(std::fabs(rng.unit()));
    for (int d = 0; d < D; ++d) ln[d] = float_to_bf16_bits(1.0f + 0.1f * static_cast<float>(rng.unit()));
    const float ln_eps = 1e-20f;
    // The finish with pre_in (the site's own logits computed and ignored).
    dgpp::MhcSinglePass sp;
    sp.pre_in = pre;
    dgpp::launch_mhc_compute_normed(c.d_streams, c.dev_w, c.cfg, nullptr, c.d_post, c.d_comb, c.d_logits,
                                    ln, normed_a, ln_eps, tokens, nullptr, nullptr, false, &sp);
    dgpp::launch_mhc_collapse_normed(c.d_streams, pre, ln, ln_eps, collapsed_b, normed_b, c.cfg, tokens,
                                     nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    require(read_back(normed_a, static_cast<size_t>(tokens) * D) == read_back(normed_b, static_cast<size_t>(tokens) * D),
            "collapse_normed bitwise the finish's normed row at the same pre");
    std::vector<double> pre_d(static_cast<size_t>(tokens) * n);
    for (size_t i = 0; i < pre_d.size(); ++i) pre_d[i] = pre[i];
    std::vector<uint16_t> want_c(static_cast<size_t>(tokens) * D), want_n(static_cast<size_t>(tokens) * D);
    dgpp::glm_mhc_ref_collapse_normed(c.streams.data(), pre_d.data(), ln, ln_eps, c.cfg, tokens,
                                      want_c.data(), want_n.data());
    require_ulp(compare(read_back(collapsed_b, want_c.size()), want_c, 2, 4), 0.005, 0,
                "collapse_normed collapsed vs oracle");
    require_ulp(compare(read_back(normed_b, want_n.size()), want_n, 4, 8), 0.01, 1,
                "collapse_normed normed vs oracle");
    std::printf("[ OK ] collapse_normed tokens=%d: bitwise the finish, within budget of the oracle\n", tokens);
    cudaFree(pre); cudaFree(ln); cudaFree(normed_a); cudaFree(normed_b); cudaFree(collapsed_b);
    c.free_all();
  }
}

DGPP_TEST(mhc_config_validation_pins_supported_geometry) {
  bool threw = false;
  try {
    GlmMhcConfig bad;
    bad.hc_mult = 8;
    GlmMhcConfig::validate_config(bad);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  if (!threw) throw std::runtime_error("hc_mult != 4 must be rejected");
  threw = false;
  try {
    GlmMhcConfig bad;
    bad.sinkhorn_iters = 0;
    GlmMhcConfig::validate_config(bad);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  if (!threw) throw std::runtime_error("sinkhorn_iters < 1 must be rejected");
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  return dgpp::test::run_all();
}
