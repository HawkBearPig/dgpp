#pragma once
// Selection-divergence audit for layer-level DSA tests (DESIGN 12).
//
// When the device layer's top-k selection differs from the host oracle's,
// "near tie" must be a MEASURED claim, not a hand-wave. This audit makes
// every flipped row prove itself:
//
//   1. Recompute the spec selection FROM THE DEVICE'S OWN CACHE + Q inputs
//      (both downloaded bit-exact) using the pinned oracle arithmetic. The
//      device's selection must match it bitwise — this certifies the whole
//      select pipeline (logit arithmetic, composite key, merge, expansion)
//      is spec-exact for the inputs it actually consumed. A wrong scale,
//      fold order, or misindexed dot buffer fails here, loudly. (A decode
//      row's dots are the tensor cores', which the host cannot reproduce
//      bit for bit and the fused kernel does not leave behind: there the
//      selection may differ from the oracle's only in entries whose oracle
//      logits sit on the cut within the fp32 sums' rounding.)
//   2. Certify the divergence is a boundary near tie: recompute logits from
//      the REFERENCE's inputs too, then require each swapped pool pair to
//      straddle the rank-select_k cut within a small multiple of the
//      measured device-vs-reference logit noise. Pool ranked #600 is not a
//      near tie no matter what the noise says.
//
// Inputs are per-flipped-row; the caller supplies the device cache contents
// (gathered in LOGICAL pool order through the block table) and both sides'
// quantized q rows. The arithmetic mirrors are the kernel-pinned ones from
// dsa_test.cu (contraction-proof separate mul/add in the documented order
// plus the warp butterfly), so parity is bitwise, not statistical, wherever
// the device's dots are captured (the prefill rows).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "models/dsa_geometry.hpp"
#include "models/dsa_reference.hpp"

namespace dgpp::dsa_test {

struct NearTieAudit {
  int64_t rows_flipped = 0;
  int64_t swaps_certified = 0;
  double max_boundary_gap = 0;    // |logit(a) - logit(b)| of a swapped pair
  double max_noise_multiple = 0;  // boundary gap / measured noise
};

// Mirrors the select kernels' warp butterfly sum (parallel-step semantics,
// contraction-proof: each step writes all 32 lanes).
inline float butterfly_sum_32(const float* in) {
  float a[32], b[32];
  std::memcpy(a, in, sizeof(a));
  for (int off = 16; off > 0; off >>= 1) {
    for (int i = 0; i < 32; ++i) b[i] = a[i] + a[i ^ off];
    std::memcpy(a, b, sizeof(a));
  }
  return a[0];
}

// One pool's logit from q_fp8/w_folded against a cache row, exactly as the
// device kernels compute it (separate mul/add, fold order pinned).
// `relu` clamps each head's dot at zero (the full GLM-5.3's indexer).
inline float pool_logit_mirror(const uint8_t* q8_row, const float* w_row,
                               const uint8_t* k8_row, float ks, int heads,
                               int dim, bool relu = false) {
  float contrib[32];
  for (int h = 0; h < heads; ++h) {
    float dot = 0.0f;
    for (int d = 0; d < dim; ++d) {
      dot += fp8_e4m3_bits_to_float(q8_row[h * dim + d]) *
             fp8_e4m3_bits_to_float(k8_row[d]);
    }
    if (relu) dot = std::max(dot, 0.0f);
    contrib[h] = (w_row[h] * ks) * dot;
  }
  return butterfly_sum_32(contrib);
}

// Complete audit of one flipped row. Throws with a precise diagnosis when
// the divergence is not a certified near tie.
//
//   select_k:            pools per query (g.select_k)
//   dev_topk/ref_topk:   the row's expanded token lists (any stride; read
//                        until -1 or end)
//   dev_q8/dev_w:        device q_fp8 row [heads*dim] + w_folded [heads]
//   ref_q8/ref_w:        reference's same
//   dev_k/dev_ks:        device cache in LOGICAL pool order [n_pools, dim]
//                        + scales [n_pools]
//   ref_k/ref_ks:        reference's cache, same layout
//   n_pools:             visible pools for this row (pools to score)
//   dev_dots/dot_stride: the device's actual fp8-dot buffer for this row
//                        ([heads, dot_stride], row r head h at
//                        dev_dots[r*heads + h]... caller indexes) — when
//                        non-null (prefill rows) the mirror consumes the
//                        tensor-core dots bit-exact instead of recomputing
//                        them (the host cannot reproduce a tensor-core
//                        reduction order). Null for decode rows, whose
//                        fused select scores on the tensor cores too
//                        (2026-11-15) and keeps no dots: the mirror's
//                        chain is then the scores within their rounding,
//                        and Part 1 holds the selection to that.
inline void audit_flipped_row(int select_k, const int32_t* dev_topk,
                              const int32_t* ref_topk, const uint8_t* dev_q8,
                              const float* dev_w, const uint8_t* dev_k,
                              const float* dev_ks, const uint8_t* ref_q8,
                              const float* ref_w, const uint8_t* ref_k,
                              const float* ref_ks, int64_t n_pools, int heads,
                              int dim, int kpool, NearTieAudit& out,
                              const float* dev_dots = nullptr,
                              int64_t dot_stride = 0, int64_t row = -1,
                              int64_t topk_width = -1, bool relu = false) {
  // Token sets -> SCORED pool sets. The expanded row is [selected pool
  // tokens (ascending)][incomplete tail tokens][-1 pad]; tail tokens start
  // at visible*kpool and are always appended by both sides (never scored),
  // so they must not enter the pool sets — the tail's pool would otherwise
  // inflate every row's set by one and break the spec-size check.
  const int64_t tail_start = n_pools * kpool;  // n_pools == visible pools
  auto pools_of = [&](const int32_t* toks, std::vector<int32_t>& pools) {
    pools.clear();
    // A full row (select_k pools + kpool-1 tail) has NO -1 padding, so the
    // scan must be bounded by the row width — reading to the first -1
    // would leak into the next row's tokens.
    const int64_t width =
        topk_width > 0 ? topk_width : int64_t(kpool) * (select_k + 1);
    for (int64_t i = 0; i < width && toks[i] >= 0; ++i)
      if (toks[i] < tail_start) pools.push_back(toks[i] / kpool);
    std::sort(pools.begin(), pools.end());
    pools.erase(std::unique(pools.begin(), pools.end()), pools.end());
  };
  std::vector<int32_t> dev_pools, ref_pools;
  pools_of(dev_topk, dev_pools);
  pools_of(ref_topk, ref_pools);

  // The device logit for pool j, using the device's ACTUAL dot values when
  // they were captured (prefill) or the chain over its inputs (decode: the
  // device's score within the fp32 sums' rounding).
  const auto dev_logit = [&](int64_t j) -> float {
    float contrib[32];
    for (int h = 0; h < heads; ++h) {
      float dot = 0.0f;
      if (dev_dots) {
        // Dot buffer layout: [rows*heads, dot_stride], row r head h at
        // (r*heads + h) * dot_stride.
        dot = dev_dots[(row * heads + h) * dot_stride + j];
      } else {
        for (int d = 0; d < dim; ++d)
          dot += fp8_e4m3_bits_to_float(dev_q8[h * dim + d]) *
                 fp8_e4m3_bits_to_float(dev_k[j * dim + d]);
      }
      if (relu) dot = std::max(dot, 0.0f);
      contrib[h] = (dev_w[h] * dev_ks[j]) * dot;
    }
    return butterfly_sum_32(contrib);
  };

  // Part 1: the device selection must be the exact spec output for the
  // device's own inputs (bitwise certification of the select pipeline —
  // dots included when captured).
  {
    std::vector<float> logits(size_t(n_pools), 0.0f);
    for (int64_t j = 0; j < n_pools; ++j)
      logits[size_t(j)] = dev_logit(j);
    std::vector<int32_t> spec(size_t(select_k), 0);
    const int n = dsa_ref::select_pools(logits.data(), n_pools, select_k,
                                        spec.data());
    // A decode row (no captured dots): entries only one side selected must
    // sit on the cut — within 1e-5 of the row's logit range of the
    // select_k-th logit, the bound the kernel's own tests hold a score to.
    bool on_the_cut = false;
    if (dev_dots == nullptr && n == int(dev_pools.size()) && n == select_k && n_pools > select_k) {
      std::vector<float> sorted = logits;
      std::nth_element(sorted.begin(), sorted.begin() + (select_k - 1), sorted.end(), std::greater<float>());
      const double cut = sorted[size_t(select_k - 1)];
      const auto range = std::minmax_element(logits.begin(), logits.end());
      const double tol = 1e-5 * (double(*range.second) - double(*range.first));
      std::vector<int32_t> only;
      std::set_symmetric_difference(dev_pools.begin(), dev_pools.end(), spec.begin(), spec.begin() + n,
                                    std::back_inserter(only));
      on_the_cut = true;
      for (int32_t p : only)
        if (!(std::fabs(double(logits[size_t(p)]) - cut) <= tol)) on_the_cut = false;
    }
    if (!on_the_cut &&
        (n != int(dev_pools.size()) ||
         !std::equal(dev_pools.begin(), dev_pools.end(), spec.begin()))) {
      std::string ds, ss;
      for (int32_t p : dev_pools) ds += " " + std::to_string(p);
      for (int i = 0; i < n; ++i) ss += " " + std::to_string(spec[size_t(i)]);
      throw std::runtime_error(
          "audit: row " + std::to_string(row) +
          " device selection is NOT the spec output for the device's own "
          "inputs (select pipeline bug, not a near tie); dev:" + ds +
          " spec:" + ss);
    }
  }

  // Part 2: measure the cross-implementation noise and certify every swap
  // straddles the boundary within it.
  std::vector<float> logits_dev(size_t(n_pools), 0.0f);
  std::vector<float> logits_ref(size_t(n_pools), 0.0f);
  double noise_sum = 0;
  for (int64_t j = 0; j < n_pools; ++j) {
    logits_dev[size_t(j)] = dev_logit(j);
    logits_ref[size_t(j)] =
        pool_logit_mirror(ref_q8, ref_w, ref_k + j * dim, ref_ks[j], heads,
                          dim, relu);
    noise_sum += std::abs(double(logits_dev[j]) - double(logits_ref[j]));
  }
  // Noise yardstick: mean |dev - ref| logit difference over visible pools.
  // This is the honest scale of cross-implementation disagreement for THIS
  // row — a misrouted cache read would inflate it (caught by the noise
  // multiple), and a zero-noise row with a swap is a hard spec bug.
  const double noise = noise_sum / std::max<double>(n_pools, 1);

  // NaN diagnostics: a NaN logit is never noise — identify the side and
  // the offending cache row (fp8 NaN codes are 0x7F/0xFF; scales are
  // power-of-two and finite by construction).
  {
    bool dev_nan = false, ref_nan = false;
    for (int64_t j = 0; j < n_pools; ++j) {
      if (std::isnan(logits_dev[size_t(j)])) dev_nan = true;
      if (std::isnan(logits_ref[size_t(j)])) ref_nan = true;
    }
    if (dev_nan || ref_nan) {
      for (int64_t j = 0; j < n_pools; ++j) {
        if (std::isnan(logits_dev[size_t(j)]) ||
            std::isnan(logits_ref[size_t(j)])) {
          int dev_nan_codes = 0, ref_nan_codes = 0;
          for (int d = 0; d < dim; ++d) {
            const uint8_t dc = dev_k[size_t(j) * dim + d];
            const uint8_t rc = ref_k[size_t(j) * dim + d];
            if ((dc & 0x7F) == 0x7F) ++dev_nan_codes;
            if ((rc & 0x7F) == 0x7F) ++ref_nan_codes;
          }
          std::printf(
              "[AUDIT-NaN] row %lld pool %lld: dev_logit=%f ref_logit=%f "
              "dev_scale=%f ref_scale=%f dev_nan_codes=%d ref_nan_codes=%d\n",
              (long long)row, (long long)j, logits_dev[size_t(j)],
              logits_ref[size_t(j)], dev_ks[j], ref_ks[j], dev_nan_codes,
              ref_nan_codes);
        }
      }
    }
  }

  // Reference ranking (the spec's ordering of this row's pools).
  std::vector<int32_t> order(size_t(n_pools), 0);
  for (int64_t j = 0; j < n_pools; ++j) order[size_t(j)] = int32_t(j);
  std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
    const float la = logits_ref[size_t(a)], lb = logits_ref[size_t(b)];
    if (la != lb) return la > lb;
    return a < b;  // exact ties -> lower pool index (the pinned spec)
  });

  std::vector<int32_t> only_dev, only_ref;
  std::set_difference(dev_pools.begin(), dev_pools.end(), ref_pools.begin(),
                      ref_pools.end(), std::back_inserter(only_dev));
  std::set_difference(ref_pools.begin(), ref_pools.end(), dev_pools.begin(),
                      dev_pools.end(), std::back_inserter(only_ref));
  if (only_dev.empty() && only_ref.empty()) return;  // not actually flipped

  if (only_dev.size() != only_ref.size())
    throw std::runtime_error(
        "audit: selection differs by " + std::to_string(only_dev.size()) +
        " in / " + std::to_string(only_ref.size()) +
        " out (kpool-swarm or expansion bug, not a near tie)");

  for (size_t i = 0; i < only_dev.size(); ++i) {
    const int32_t pool_in = only_dev[i];   // device selected, reference didn't
    const int32_t pool_out = only_ref[i];  // reference selected, device didn't
    const double gap_in = std::abs(double(logits_ref[pool_in]) -
                                   double(logits_ref[pool_out]));
    // Rank of pool_in in the reference's own ordering (0-based).
    const int64_t rank_in = std::find(order.begin(), order.end(), pool_in) -
                            order.begin();
    const int64_t rank_out = std::find(order.begin(), order.end(), pool_out) -
                             order.begin();
    // A near tie requires: pool_out at/near the select_k boundary, pool_in
    // adjacent in rank, and the reference-gap within the noise multiple.
    const bool ranks_adjacent = rank_out < select_k && rank_in < select_k + 8;
    const double gap_multiple = noise > 0   ? gap_in / noise
                                : gap_in > 0 ? 1e30
                                             : 0;
    if (!ranks_adjacent || gap_multiple > 32.0)
      throw std::runtime_error(
          "audit: pool " + std::to_string(pool_in) + " (ref rank " +
          std::to_string(rank_in) + ", gap " + std::to_string(gap_in) +
          " = " + std::to_string(gap_multiple) + "x noise) displaced pool " +
          std::to_string(pool_out) + " (ref rank " +
          std::to_string(rank_out) + ") — NOT a boundary near tie");
    out.swaps_certified += 1;
    out.max_boundary_gap = std::max(out.max_boundary_gap, gap_in);
    out.max_noise_multiple = std::max(out.max_noise_multiple, gap_multiple);
  }
  out.rows_flipped += 1;
}

}  // namespace dgpp::dsa_test
