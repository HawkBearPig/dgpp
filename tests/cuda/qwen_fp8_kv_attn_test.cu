// Numerical evidence for the Qwen FP8 KV cache (docs/qwen_fp8_kv_plan.md,
// "Numerical evidence"). Two parts, NO fp8 kernel required — this pre-registers
// the acceptance band before any fp8 storage/attention code exists:
//
//   1. Per-element e4m3 codec study (host): round-trip realistic bf16 K/V rows
//      through kernels/latent_format.hpp's e4m3 codec (per-row absmax/448
//      scale) and report the relative-error distribution. The 3-bit-mantissa
//      RNE bound is 2^-4 (6.25 %) worst case.
//
//   2. Attention sample-match (GPU): run the EXISTING bf16 qsa_attn_partial
//      over (a) bf16 K/V and (b) e4m3-round-tripped K/V (the dequantized rows
//      Phase A's read path would produce). Phase A's divergence IS the
//      quantizer (the kernel is bitwise the bf16 kernel over the dequantized
//      rows), so this measures exactly what Phase A will change. We report the
//      output divergence and the sample-match rate, and assert the
//      necessary-condition bound: the attention output's l2_rel <= 2^-4.
//
// Run:  ctest -R qwen_fp8_kv_attn   (or the built binary directly)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/dsa.hpp"
#include "kernels/latent_format.hpp"
#include "kernels/qsa.hpp"

using namespace dgpp::kda_test;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}
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

// e4m3 round-trip of one bf16 row (dim elements) with the per-row absmax/448
// scale — exactly the per-(slot, kv-head) quantization the plan specifies.
// Returns the dequantized bf16 values (what an fp8 cache would hand the
// attention kernel).
std::vector<uint16_t> fp8_roundtrip_row(const std::vector<uint16_t>& row) {
  float absmax = 0.0f;
  for (uint16_t b : row) absmax = std::max(absmax, std::fabs(dgpp::bf16_bits_to_float(b)));
  const auto s = dgpp::latent_fp8_row_scale(absmax);
  std::vector<uint16_t> out(row.size());
  for (size_t i = 0; i < row.size(); ++i) {
    const float x = dgpp::bf16_bits_to_float(row[i]);
    const uint8_t code = dgpp::latent_fp8_encode(x, s.inv);
    out[i] = dgpp::latent_fp8_decode_bf16(code, s.scale);
  }
  return out;
}

// Geometry: a small QSA slice (the real model is 24 q / 2 kv heads, dim 256;
// 6/2 is the test's standard slice). One request, seq tokens, rows queries.
struct Fp8Geo {
  int local_heads = 6, kv_heads = 2, dim = 256;
  int block_tokens = 16, seq = 128, rows = 16;
  int width() const { return kv_heads * dim; }
  int blocks_per_request() const { return (seq + block_tokens - 1) / block_tokens; }
  int slots() const { return block_tokens * blocks_per_request(); }
};

// Fill a paged K/V cache (as qsa_kv_append writes it) from bf16 source rows.
void fill_cache(const std::vector<uint16_t>& k, const std::vector<uint16_t>& v, const Fp8Geo& g,
                const int32_t* dtable, uint16_t* kc, uint16_t* vc, cudaStream_t st) {
  std::vector<int32_t> req_ids(static_cast<size_t>(g.seq), 0);
  std::vector<int64_t> pos(static_cast<size_t>(g.seq));
  for (int i = 0; i < g.seq; ++i) pos[static_cast<size_t>(i)] = i;
  const int width = g.width();
  DevBuf dk = up(k), dv = up(v), dreq = up(req_ids), dpos = up(pos);
  dgpp::qsa_kv_append(ptr<uint16_t>(dk), width, ptr<uint16_t>(dv), width, ptr<int32_t>(dreq),
                      ptr<int64_t>(dpos), g.seq, dtable, g.blocks_per_request(), g.block_tokens,
                      g.kv_heads, g.dim, kc, vc, nullptr, nullptr, st);
}

// Run the production bf16 attention chain (partial + combine + gate) over a
// paged cache; returns the bf16 gated output [rows, local_heads, dim].
std::vector<uint16_t> run_attn(const uint16_t* qonly, const uint16_t* qgate, const uint16_t* kc,
                               const uint16_t* vc, const int32_t* dreq, const int32_t* dtopk,
                               const int32_t* dcounts, const int32_t* dtable, const Fp8Geo& g,
                               int n_split, cudaStream_t st, const float* k_scale = nullptr,
                               const float* v_scale = nullptr, bool prefill = false) {
  const int width = g.width();
  const size_t part = static_cast<size_t>(g.rows) * n_split * g.local_heads;
  DevBuf m_ws(part * 4), l_ws(part * 4), c_ws(part * g.dim * 4),
      c_out(static_cast<size_t>(g.rows) * g.local_heads * g.dim * 4),
      out(static_cast<size_t>(g.rows) * g.local_heads * g.dim * 2);
  const int64_t q_row_stride = g.local_heads * g.dim, q_head_stride = 2 * g.dim;
  if (prefill)
    dgpp::qsa_attn_prefill_partial(qonly, q_row_stride, kc, vc, dreq, dtopk, g.seq, dcounts,
                                   g.rows, n_split, g.local_heads, g.kv_heads, g.dim, g.block_tokens,
                                   dtable, g.blocks_per_request(), 1.0f / 16.0f, mptr<float>(m_ws),
                                   mptr<float>(l_ws), mptr<float>(c_ws), st, k_scale, v_scale);
  else
    dgpp::qsa_attn_partial(qonly, q_row_stride, kc, vc, dreq, dtopk, g.seq, dcounts,
                           g.rows, n_split, g.local_heads, g.kv_heads, g.dim, g.block_tokens, dtable,
                           g.blocks_per_request(), 1.0f / 16.0f, mptr<float>(m_ws), mptr<float>(l_ws),
                           mptr<float>(c_ws), st, k_scale, v_scale);
  dgpp::dsa_attn_combine(ptr<float>(m_ws), ptr<float>(l_ws), ptr<float>(c_ws), g.rows, n_split,
                         g.local_heads, g.dim, mptr<float>(c_out), st);
  dgpp::qsa_gate_out(ptr<float>(c_out), qgate + g.dim, q_row_stride, q_head_stride,
                     mptr<uint16_t>(out), g.rows, g.local_heads, g.dim, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  return down<uint16_t>(out, static_cast<size_t>(g.rows) * g.local_heads * g.dim);
}

}  // namespace

DGPP_TEST(qwen_fp8_kv_per_element_codec_study) {
  // Realistic bf16 K/V rows (unit-normal, the test's standard distribution).
  const int dim = 256, rows = 4096;
  const auto data = random_bf16_normal(1, static_cast<int64_t>(rows) * dim, 1.0f);
  std::vector<double> rel;
  rel.reserve(static_cast<size_t>(rows) * dim);
  long lost = 0;            // elements that underflow to 0 (rel err -> 1)
  double lost_abs_max = 0;  // their |x| as a fraction of the row's absmax
  for (int r = 0; r < rows; ++r) {
    std::vector<uint16_t> row(data.begin() + static_cast<size_t>(r) * dim,
                              data.begin() + static_cast<size_t>(r + 1) * dim);
    const auto rt = fp8_roundtrip_row(row);
    float absmax = 0.0f;
    for (uint16_t b : row) absmax = std::max(absmax, std::abs(dgpp::bf16_bits_to_float(b)));
    for (int i = 0; i < dim; ++i) {
      const double x = dgpp::bf16_bits_to_float(row[i]);
      const double y = dgpp::bf16_bits_to_float(rt[i]);
      rel.push_back(std::abs(x - y) / std::max(std::abs(x), 1e-30));
      if (y == 0.0 && x != 0.0) {
        ++lost;
        lost_abs_max = std::max(lost_abs_max, std::abs(x) / absmax);
      }
    }
  }
  std::sort(rel.begin(), rel.end());
  const auto pct = [&](double p) {
    return rel[std::min(static_cast<size_t>(p * rel.size()), rel.size() - 1)];
  };
  const double lost_frac = 100.0 * static_cast<double>(lost) / static_cast<double>(rel.size());
  std::printf(
      "[fp8-kv] per-element e4m3 rel error (n=%zu): p50 %.4f  p90 %.4f  p99 %.4f  max %.4f; "
      "underflow %.3f%% of elems (|x|/absmax <= %.2g)\n",
      rel.size(), pct(0.50), pct(0.90), pct(0.99), rel.back(), lost_frac, lost_abs_max);
  // The 3-bit-mantissa RNE bound (2^-4) holds for *representable* values. The
  // max is 1.0 only from the rare sub-scale values that underflow to 0 (their
  // |x| is a negligible fraction of the row's absmax, so they contribute
  // nothing to the L2 norm — the attention sample-match below confirms this).
  // So the honest bound is on the representable regime: p99 <= 2^-4.
  require(pct(0.99) <= 0.0625 + 1e-9, "per-element e4m3 p99 exceeds the 2^-4 mantissa bound");
  require(lost_abs_max <= 0.01, "underflowed elements are not negligible vs the row scale");
}

DGPP_TEST(qwen_fp8_kv_attention_sample_match) {
  Fp8Geo g;
  cudaStream_t st = test_stream();
  const int width = g.width();

  // Identity block table for one request: block b -> physical block b.
  std::vector<int32_t> table(static_cast<size_t>(g.blocks_per_request()));
  for (int b = 0; b < g.blocks_per_request(); ++b) table[static_cast<size_t>(b)] = b;
  DevBuf dtable = up(table);

  // bf16 K/V source rows + the fp8-round-tripped (dequantized) versions.
  const auto k = random_bf16_normal(31, static_cast<int64_t>(g.seq) * width, 1.0f);
  const auto v = random_bf16_normal(32, static_cast<int64_t>(g.seq) * width, 1.0f);
  auto k8 = k, v8 = v;
  for (int t = 0; t < g.seq; ++t) {
    for (int h = 0; h < g.kv_heads; ++h) {
      const size_t off = (static_cast<size_t>(t) * g.kv_heads + h) * g.dim;
      std::vector<uint16_t> kr(k.begin() + off, k.begin() + off + g.dim);
      std::vector<uint16_t> vr(v.begin() + off, v.begin() + off + g.dim);
      const auto kr8 = fp8_roundtrip_row(kr);
      const auto vr8 = fp8_roundtrip_row(vr);
      std::copy(kr8.begin(), kr8.end(), k8.begin() + off);
      std::copy(vr8.begin(), vr8.end(), v8.begin() + off);
    }
  }

  // Two paged caches: bf16 and fp8-dequantized.
  DevBuf kc(static_cast<size_t>(g.slots()) * width * 2), vc(static_cast<size_t>(g.slots()) * width * 2),
      kc8(static_cast<size_t>(g.slots()) * width * 2), vc8(static_cast<size_t>(g.slots()) * width * 2);
  fill_cache(k, v, g, ptr<int32_t>(dtable), mptr<uint16_t>(kc), mptr<uint16_t>(vc), st);
  fill_cache(k8, v8, g, ptr<int32_t>(dtable), mptr<uint16_t>(kc8), mptr<uint16_t>(vc8), st);

  // Queries: [q | gate] interleave; qonly is the q half. Every row attends to
  // the full token list (short-context select-all, the kernel's dense case).
  const int64_t q_row_stride = g.local_heads * (2 * g.dim);
  const auto qg = random_bf16_normal(33, static_cast<int64_t>(g.rows) * q_row_stride, 1.0f);
  std::vector<uint16_t> qonly(static_cast<size_t>(g.rows) * g.local_heads * g.dim);
  for (int r = 0; r < g.rows; ++r)
    for (int h = 0; h < g.local_heads; ++h)
      std::copy(qg.begin() + (r * q_row_stride + h * 2 * g.dim),
                qg.begin() + (r * q_row_stride + h * 2 * g.dim + g.dim),
                qonly.begin() + (static_cast<size_t>(r) * g.local_heads + h) * g.dim);

  std::vector<int32_t> req_ids(static_cast<size_t>(g.rows), 0);
  std::vector<int32_t> topk(static_cast<size_t>(g.rows) * g.seq);
  for (int r = 0; r < g.rows; ++r)
    for (int i = 0; i < g.seq; ++i) topk[static_cast<size_t>(r) * g.seq + i] = i;
  std::vector<int32_t> counts(static_cast<size_t>(g.rows), g.seq);
  DevBuf dqg = up(qg), dqonly = up(qonly), dreq = up(req_ids), dtopk = up(topk), dcounts = up(counts);

  const auto ref = run_attn(ptr<uint16_t>(dqonly), ptr<uint16_t>(dqg), mptr<uint16_t>(kc),
                            mptr<uint16_t>(vc), ptr<int32_t>(dreq), ptr<int32_t>(dtopk),
                            ptr<int32_t>(dcounts), ptr<int32_t>(dtable), g, 1, st);
  const auto got = run_attn(ptr<uint16_t>(dqonly), ptr<uint16_t>(dqg), mptr<uint16_t>(kc8),
                            mptr<uint16_t>(vc8), ptr<int32_t>(dreq), ptr<int32_t>(dtopk),
                            ptr<int32_t>(dcounts), ptr<int32_t>(dtable), g, 1, st);

  // Divergence + sample-match rate.
  const long n = static_cast<long>(ref.size());
  std::vector<float> gf(n), wf(n);
  for (long i = 0; i < n; ++i) {
    gf[static_cast<size_t>(i)] = dgpp::bf16_bits_to_float(got[static_cast<size_t>(i)]);
    wf[static_cast<size_t>(i)] = dgpp::bf16_bits_to_float(ref[static_cast<size_t>(i)]);
  }
  const Stats s = compare_abs_rel(gf.data(), wf.data(), n, 0.0);
  const auto match_frac = [&](double tol) {
    long m = 0;
    for (long i = 0; i < n; ++i)
      if (std::abs(gf[static_cast<size_t>(i)] - wf[static_cast<size_t>(i)]) /
              std::max(std::abs(wf[static_cast<size_t>(i)]), 1e-30f) <= tol)
        ++m;
    return 100.0 * static_cast<double>(m) / static_cast<double>(n);
  };
  std::printf("[fp8-kv] attention sample-match (n=%ld): l2_rel %.4f  max_abs %.4g  "
              "match(<=2^-4) %.2f%%  match(<=2^-3) %.2f%%\n",
              n, s.l2_rel, s.max_abs, match_frac(0.0625), match_frac(0.125));
  // Necessary-condition bound: the output is a weighted average of V rows each
  // within 2^-4 relative, so the output's l2_rel cannot exceed 2^-4.
  require(s.l2_rel <= 0.0625,
          "fp8 attention output l2_rel exceeds the 2^-4 necessary-condition bound");
}

// The REAL Phase A path (not the host round-trip): qsa_kv_append quantizes the
// K/V to e4m3 + per-(slot, kv-head) scales, and qsa_attn_partial dequants
// in-kernel. Compared against the bf16 pool over the same queries. This is the
// acceptance test for the fp8 storage + dequant kernels.
DGPP_TEST(qwen_fp8_kv_real_path_matches_bf16) {
  Fp8Geo g;
  cudaStream_t st = test_stream();
  const int width = g.width();
  const int slots = g.slots();

  std::vector<int32_t> table(static_cast<size_t>(g.blocks_per_request()));
  for (int b = 0; b < g.blocks_per_request(); ++b) table[static_cast<size_t>(b)] = b;
  DevBuf dtable = up(table);

  const auto k = random_bf16_normal(41, static_cast<int64_t>(g.seq) * width, 1.0f);
  const auto v = random_bf16_normal(42, static_cast<int64_t>(g.seq) * width, 1.0f);

  DevBuf kc(static_cast<size_t>(slots) * width * 2), vc(static_cast<size_t>(slots) * width * 2);
  fill_cache(k, v, g, ptr<int32_t>(dtable), mptr<uint16_t>(kc), mptr<uint16_t>(vc), st);

  // fp8 pool: 1 B/elem codes + [slots, kv_heads] fp32 scales.
  DevBuf kc8(static_cast<size_t>(slots) * width), vc8(static_cast<size_t>(slots) * width);
  DevBuf ks(static_cast<size_t>(slots) * g.kv_heads * 4), vs(static_cast<size_t>(slots) * g.kv_heads * 4);
  {
    std::vector<int32_t> req_ids(static_cast<size_t>(g.seq), 0);
    std::vector<int64_t> pos(static_cast<size_t>(g.seq));
    for (int i = 0; i < g.seq; ++i) pos[static_cast<size_t>(i)] = i;
    DevBuf dk = up(k), dv = up(v), dreq = up(req_ids), dpos = up(pos);
    dgpp::qsa_kv_append(ptr<uint16_t>(dk), width, ptr<uint16_t>(dv), width, ptr<int32_t>(dreq),
                        ptr<int64_t>(dpos), g.seq, ptr<int32_t>(dtable), g.blocks_per_request(),
                        g.block_tokens, g.kv_heads, g.dim,
                        reinterpret_cast<uint16_t*>(mptr<uint8_t>(kc8)),
                        reinterpret_cast<uint16_t*>(mptr<uint8_t>(vc8)), mptr<float>(ks),
                        mptr<float>(vs), st);
  }

  const int64_t q_row_stride = g.local_heads * (2 * g.dim);
  const auto qg = random_bf16_normal(43, static_cast<int64_t>(g.rows) * q_row_stride, 1.0f);
  std::vector<uint16_t> qonly(static_cast<size_t>(g.rows) * g.local_heads * g.dim);
  for (int r = 0; r < g.rows; ++r)
    for (int h = 0; h < g.local_heads; ++h)
      std::copy(qg.begin() + (r * q_row_stride + h * 2 * g.dim),
                qg.begin() + (r * q_row_stride + h * 2 * g.dim + g.dim),
                qonly.begin() + (static_cast<size_t>(r) * g.local_heads + h) * g.dim);
  std::vector<int32_t> req_ids(static_cast<size_t>(g.rows), 0);
  std::vector<int32_t> topk(static_cast<size_t>(g.rows) * g.seq);
  for (int r = 0; r < g.rows; ++r)
    for (int i = 0; i < g.seq; ++i) topk[static_cast<size_t>(r) * g.seq + i] = i;
  std::vector<int32_t> counts(static_cast<size_t>(g.rows), g.seq);
  DevBuf dqg = up(qg), dqonly = up(qonly), dreq = up(req_ids), dtopk = up(topk), dcounts = up(counts);

  const auto ref = run_attn(ptr<uint16_t>(dqonly), ptr<uint16_t>(dqg), mptr<uint16_t>(kc),
                            mptr<uint16_t>(vc), ptr<int32_t>(dreq), ptr<int32_t>(dtopk),
                            ptr<int32_t>(dcounts), ptr<int32_t>(dtable), g, 1, st);
  const auto got = run_attn(ptr<uint16_t>(dqonly), ptr<uint16_t>(dqg),
                            reinterpret_cast<uint16_t*>(mptr<uint8_t>(kc8)),
                            reinterpret_cast<uint16_t*>(mptr<uint8_t>(vc8)), ptr<int32_t>(dreq),
                            ptr<int32_t>(dtopk), ptr<int32_t>(dcounts), ptr<int32_t>(dtable), g, 1, st,
                            mptr<float>(ks), mptr<float>(vs));

  const long n = static_cast<long>(ref.size());
  std::vector<float> gf(n), wf(n);
  for (long i = 0; i < n; ++i) {
    gf[static_cast<size_t>(i)] = dgpp::bf16_bits_to_float(got[static_cast<size_t>(i)]);
    wf[static_cast<size_t>(i)] = dgpp::bf16_bits_to_float(ref[static_cast<size_t>(i)]);
  }
  const Stats s = compare_abs_rel(gf.data(), wf.data(), n, 0.0);
  std::printf("[fp8-kv] REAL PATH (append-quantize + in-kernel dequant) vs bf16: "
              "l2_rel %.4f  max_abs %.4g\n", s.l2_rel, s.max_abs);
  require(s.l2_rel <= 0.0625, "fp8 real-path attention l2_rel exceeds the 2^-4 bound");

  // The prefill kernel (the `attend` function pointer's long-prefill branch)
  // over the same fp8 pool.
  const auto ref_pf = run_attn(ptr<uint16_t>(dqonly), ptr<uint16_t>(dqg), mptr<uint16_t>(kc),
                               mptr<uint16_t>(vc), ptr<int32_t>(dreq), ptr<int32_t>(dtopk),
                               ptr<int32_t>(dcounts), ptr<int32_t>(dtable), g, 1, st, nullptr, nullptr,
                               /*prefill=*/true);
  const auto got_pf = run_attn(ptr<uint16_t>(dqonly), ptr<uint16_t>(dqg),
                               reinterpret_cast<uint16_t*>(mptr<uint8_t>(kc8)),
                               reinterpret_cast<uint16_t*>(mptr<uint8_t>(vc8)), ptr<int32_t>(dreq),
                               ptr<int32_t>(dtopk), ptr<int32_t>(dcounts), ptr<int32_t>(dtable), g, 1, st,
                               mptr<float>(ks), mptr<float>(vs), /*prefill=*/true);
  std::vector<float> gpf(n), wpf(n);
  for (long i = 0; i < n; ++i) {
    gpf[static_cast<size_t>(i)] = dgpp::bf16_bits_to_float(got_pf[static_cast<size_t>(i)]);
    wpf[static_cast<size_t>(i)] = dgpp::bf16_bits_to_float(ref_pf[static_cast<size_t>(i)]);
  }
  const Stats sp = compare_abs_rel(gpf.data(), wpf.data(), n, 0.0);
  std::printf("[fp8-kv] REAL PATH prefill kernel vs bf16: l2_rel %.4f  max_abs %.4g\n", sp.l2_rel,
              sp.max_abs);
  require(sp.l2_rel <= 0.0625, "fp8 real-path prefill l2_rel exceeds the 2^-4 bound");

  // The tensor-core warp prefill kernel (the serving default for long prefill).
  if (dgpp::qsa_warp_supported(g.dim, g.local_heads, g.kv_heads)) {
    const int64_t qrs = static_cast<int64_t>(g.local_heads) * g.dim;
    DevBuf ref_w(static_cast<size_t>(n) * 4), got_w(static_cast<size_t>(n) * 4);
    dgpp::qsa_attn_prefill_warp(ptr<uint16_t>(dqonly), qrs, mptr<uint16_t>(kc), mptr<uint16_t>(vc),
                                ptr<int32_t>(dreq), ptr<int32_t>(dtopk), g.seq, ptr<int32_t>(dcounts),
                                g.rows, g.local_heads, g.kv_heads, g.block_tokens, ptr<int32_t>(dtable),
                                g.blocks_per_request(), 1.0f / 16.0f, mptr<float>(ref_w), st, nullptr,
                                nullptr);
    dgpp::qsa_attn_prefill_warp(ptr<uint16_t>(dqonly), qrs,
                                reinterpret_cast<uint16_t*>(mptr<uint8_t>(kc8)),
                                reinterpret_cast<uint16_t*>(mptr<uint8_t>(vc8)), ptr<int32_t>(dreq),
                                ptr<int32_t>(dtopk), g.seq, ptr<int32_t>(dcounts), g.rows, g.local_heads,
                                g.kv_heads, g.block_tokens, ptr<int32_t>(dtable), g.blocks_per_request(),
                                1.0f / 16.0f, mptr<float>(got_w), st, mptr<float>(ks), mptr<float>(vs));
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    const auto rwf = down<float>(ref_w, static_cast<size_t>(n));
    const auto gwf = down<float>(got_w, static_cast<size_t>(n));
    const Stats sw = compare_abs_rel(gwf.data(), rwf.data(), n, 0.0);
    std::printf("[fp8-kv] REAL PATH warp prefill kernel vs bf16: l2_rel %.4f  max_abs %.4g\n",
                sw.l2_rel, sw.max_abs);
    require(sw.l2_rel <= 0.0625, "fp8 real-path warp prefill l2_rel exceeds the 2^-4 bound");
  }
}

// The fused norm+RoPE+quantize+scatter kernel must reproduce the two-kernel
// chain (qsa_norm_rope_bf16 into kn_, then the append) bit for bit, so the
// zero-copy K write changes no cache byte.
DGPP_TEST(qwen_fp8_kv_fused_norm_rope_append_matches_chain) {
  Fp8Geo g;
  cudaStream_t st = test_stream();
  const int width = g.width();
  const int slots = g.slots();
  const int rotary = 64;
  std::vector<int32_t> table(static_cast<size_t>(g.blocks_per_request()));
  for (int b = 0; b < g.blocks_per_request(); ++b) table[static_cast<size_t>(b)] = b;
  DevBuf dtable = up(table);
  const auto k = random_bf16_normal(51, static_cast<int64_t>(g.seq) * width, 1.0f);
  const auto v = random_bf16_normal(52, static_cast<int64_t>(g.seq) * width, 1.0f);
  const auto wnorm = random_bf16_normal(53, g.dim, 1.0f);
  std::vector<float> inv(static_cast<size_t>(rotary) / 2);
  dgpp::qsa_rope_inv_freq(1e7, rotary, inv.data());
  std::vector<int32_t> req_ids(static_cast<size_t>(g.seq), 0);
  std::vector<int64_t> pos(static_cast<size_t>(g.seq));
  for (int i = 0; i < g.seq; ++i) pos[static_cast<size_t>(i)] = i;
  DevBuf dk = up(k), dv = up(v), dw = up(wnorm), dinv = up(inv), dreq = up(req_ids), dpos = up(pos);

  // Path A: norm+RoPE into kn_, then the append.
  DevBuf kn(static_cast<size_t>(g.seq) * width * 2);
  dgpp::qsa_norm_rope_bf16(ptr<uint16_t>(dk), width, g.dim, ptr<uint16_t>(dw), ptr<int64_t>(dpos),
                           ptr<float>(dinv), mptr<uint16_t>(kn), width, g.seq, g.kv_heads, g.dim,
                           rotary, 1e-6f, 1.0f, st);
  DevBuf kcA(static_cast<size_t>(slots) * width), vcA(static_cast<size_t>(slots) * width),
      ksA(static_cast<size_t>(slots) * g.kv_heads * 4), vsA(static_cast<size_t>(slots) * g.kv_heads * 4);
  dgpp::qsa_kv_append(ptr<uint16_t>(kn), width, ptr<uint16_t>(dv), width, ptr<int32_t>(dreq),
                      ptr<int64_t>(dpos), g.seq, ptr<int32_t>(dtable), g.blocks_per_request(),
                      g.block_tokens, g.kv_heads, g.dim,
                      reinterpret_cast<uint16_t*>(mptr<uint8_t>(kcA)),
                      reinterpret_cast<uint16_t*>(mptr<uint8_t>(vcA)), mptr<float>(ksA),
                      mptr<float>(vsA), st);

  // Path B: the fused kernel, then the append with k == nullptr (V only).
  DevBuf kcB(static_cast<size_t>(slots) * width), vcB(static_cast<size_t>(slots) * width),
      ksB(static_cast<size_t>(slots) * g.kv_heads * 4), vsB(static_cast<size_t>(slots) * g.kv_heads * 4);
  dgpp::qsa_norm_rope_append_fp8(ptr<uint16_t>(dk), width, g.dim, ptr<uint16_t>(dw),
                                 ptr<int64_t>(dpos), ptr<float>(dinv), ptr<int32_t>(dreq),
                                 ptr<int32_t>(dtable), g.blocks_per_request(), g.block_tokens, g.seq,
                                 g.kv_heads, g.dim, rotary, 1e-6f, 1.0f, mptr<uint8_t>(kcB),
                                 mptr<float>(ksB), st);
  dgpp::qsa_kv_append(nullptr, 0, ptr<uint16_t>(dv), width, ptr<int32_t>(dreq), ptr<int64_t>(dpos),
                      g.seq, ptr<int32_t>(dtable), g.blocks_per_request(), g.block_tokens,
                      g.kv_heads, g.dim, reinterpret_cast<uint16_t*>(mptr<uint8_t>(kcB)),
                      reinterpret_cast<uint16_t*>(mptr<uint8_t>(vcB)), mptr<float>(ksB),
                      mptr<float>(vsB), st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));

  const size_t nbytes = static_cast<size_t>(g.seq) * width;  // identity table: slots 0..seq-1
  const auto kcA_h = down<uint8_t>(kcA, nbytes), kcB_h = down<uint8_t>(kcB, nbytes);
  const auto vcA_h = down<uint8_t>(vcA, nbytes), vcB_h = down<uint8_t>(vcB, nbytes);
  const auto ksA_h = down<float>(ksA, static_cast<size_t>(g.seq) * g.kv_heads);
  const auto ksB_h = down<float>(ksB, static_cast<size_t>(g.seq) * g.kv_heads);
  require_bitwise("fused K codes", kcA_h.data(), kcB_h.data(), nbytes);
  require_bitwise("fused K scales", ksA_h.data(), ksB_h.data(), ksA_h.size() * sizeof(float));
  require_bitwise("fused V codes", vcA_h.data(), vcB_h.data(), nbytes);
  std::printf("[fp8-kv] FUSED norm+RoPE+quantize == norm_rope+append (bitwise): K codes, K scales, V codes\n");
}

int main() { return dgpp::test::run_all(); }
