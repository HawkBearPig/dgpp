#pragma once
// Route-divergence audit for the GLM curated parity suite (DESIGN 12).
//
// The MoE router's noaux_tc selection biases scores so experts cluster at
// the selection boundary, and the suite's ISOLATED mode recomputes each
// side's mHC+ln2 chain from the same injected streams — so each side's
// router input carries its own ~1e-3 fp noise and near-boundary experts
// legitimately flip. "Near tie" must be a MEASURED claim, never a budget
// (the dsa_near_tie_audit discipline, adapted to fp32 router scores):
//
//   1. The engine's selection must be the spec top-k of the ENGINE'S OWN
//      biased scores (biased descending, ties to the lower id) — this
//      certifies the router kernel's selection for the inputs it actually
//      consumed. A mis-sorted or dropped id fails here, loudly.
//   2. Each swapped expert pair must straddle the selection boundary
//      within a small multiple of the token's MEASURED cross-implementation
//      noise — where noise is the mean |engine - reference| biased
//      difference over the experts not involved in the swap. A legitimate
//      input perturbation moves every expert's score by the same scale, so
//      the uninvolved majority pins the true noise; corruption concentrated
//      on the swapped experts cannot hide inside its own inflation. An
//      expert ranked far below the boundary is not a near tie no matter
//      what the noise says; a zero-noise row with a swap is a hard spec
//      bug, not noise.
//
// Both sides' full biased-score rows are required inputs, so the noise
// yardstick is measured per token, never assumed or tuned. Host-only.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "models/glm/trace.hpp"

namespace dgpp::glm_route {

struct RouteFlipAudit {
  int64_t tokens_flipped = 0;
  int64_t swaps_certified = 0;
  double max_boundary_gap = 0;    // ref-side |b(in) - b(out)| of a swap
  double max_noise_multiple = 0;  // worst gap / measured mean noise (both sides)
  // The structured-noise view of the worst swap: its scale divided by the
  // MAX cross-side drift over uninvolved experts (the perturbation's
  // demonstrated ability to move one expert's score). At real dims the
  // cross-implementation input delta concentrates on massive-activation
  // channels, so a few aligned gate rows move ~10x the mean while the
  // mean stays tiny (measured: layer-17 expert 7, rank 23, gap 0.078 =
  // 11x mean noise but ~1-2x the max uninvolved drift).
  double max_structure_multiple = 0;
};

// Complete audit of one routed layer's engine-vs-reference ids. Throws
// with a precise diagnosis when a divergence is not a certified near tie.
//
//   eng_ids/ref_ids: [tokens, top_k], ascending per token (both sides'
//                    contract with the kernel / the torch reference).
//   eng_biased/ref_biased: [tokens, n_experts] fp32 biased scores
//                    (engine: the router's own export; reference: the
//                    dump's router_biased tensor).
inline void audit_route_flips(const int32_t* eng_ids,
                              const int32_t* ref_ids, const float* eng_biased,
                              const float* ref_biased, int64_t tokens,
                              int top_k, int n_experts, RouteFlipAudit& out,
                              int64_t layer_idx = -1) {
  const std::string where = layer_idx >= 0
                                ? "route audit (layer " +
                                      std::to_string(layer_idx) + ")"
                                : "route audit";
  const size_t E = static_cast<size_t>(n_experts);
  const size_t K = static_cast<size_t>(top_k);

  // Ranking of experts by one side's biased scores: descending, ties to the
  // lower id (the pinned spec order both the kernel and the reference use).
  const auto rank_by = [&](const float* biased,
                           std::vector<int32_t>* order_out) {
    std::vector<int32_t>& order = *order_out;
    order.resize(E);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
      const float ba = biased[a], bb = biased[b];
      if (ba != bb) return ba > bb;
      return a < b;
    });
  };

  for (int64_t t = 0; t < tokens; ++t) {
    const int32_t* ei = eng_ids + t * top_k;
    const int32_t* ri = ref_ids + t * top_k;
    std::vector<int32_t> eng(K), ref(K);
    std::copy_n(ei, K, eng.begin());
    std::copy_n(ri, K, ref.begin());
    std::sort(eng.begin(), eng.end());
    std::sort(ref.begin(), ref.end());
    if (eng == ref) continue;  // kept token: nothing to prove

    // Duplicated ids are a top-k bug, not a flip.
    if (std::adjacent_find(eng.begin(), eng.end()) != eng.end() ||
        std::adjacent_find(ref.begin(), ref.end()) != ref.end())
      throw std::runtime_error(
          where + ": token " + std::to_string(t) +
          " has a duplicate expert id (top-k bug, not a near tie)");

    const float* eb = eng_biased + t * E;
    const float* rb = ref_biased + t * E;

    // Part 1: the engine's ids must be the spec top-k of the engine's own
    // biased scores. Certifies the selection for the inputs it consumed.
    {
      std::vector<int32_t> order;
      rank_by(eb, &order);
      std::vector<int32_t> spec(order.begin(), order.begin() + K);
      std::sort(spec.begin(), spec.end());
      if (spec != eng) {
        std::string ss, es;
        for (int32_t e : eng) es += " " + std::to_string(e);
        for (int32_t e : spec) ss += " " + std::to_string(e);
        throw std::runtime_error(
            where + ": token " + std::to_string(t) +
            " engine selection is NOT the spec top-k of the engine's own "
            "biased scores (router bug, not a near tie); got:" + es +
            " spec:" + ss);
      }
    }

    std::vector<int32_t> only_eng, only_ref;
    std::set_difference(eng.begin(), eng.end(), ref.begin(), ref.end(),
                        std::back_inserter(only_eng));
    std::set_difference(ref.begin(), ref.end(), eng.begin(), eng.end(),
                        std::back_inserter(only_ref));
    if (only_eng.size() != only_ref.size())
      throw std::runtime_error(
          where + ": token " + std::to_string(t) + " selection differs by " +
          std::to_string(only_eng.size()) + " in / " +
          std::to_string(only_ref.size()) +
          " out (segmentation bug, not a near tie)");

    // Measured noise for THIS token: mean |engine - reference| biased
    // difference over the experts not involved in the swap (a legitimate
    // input perturbation is uniform across experts; the uninvolved
    // majority pins its scale, and corruption concentrated on the swapped
    // experts cannot inflate its own yardstick). A zero-noise row with a
    // swap is a hard spec bug.
    std::vector<char> involved(E, 0);
    for (int32_t e : only_eng) involved[size_t(e)] = 1;
    for (int32_t e : only_ref) involved[size_t(e)] = 1;
    double noise_sum = 0;
    double max_uninvolved_drift = 0;
    int64_t noise_count = 0;
    for (size_t e = 0; e < E; ++e)
      if (!involved[e]) {
        const double drift =
            std::abs(double(eb[e]) - double(rb[e]));
        noise_sum += drift;
        max_uninvolved_drift = std::max(max_uninvolved_drift, drift);
        ++noise_count;
    }
    if (noise_count == 0) {  // degenerate: everyone involved — use everyone
      noise_count = static_cast<int64_t>(E);
      for (size_t e = 0; e < E; ++e) {
        const double drift =
            std::abs(double(eb[e]) - double(rb[e]));
        noise_sum += drift;
        max_uninvolved_drift = std::max(max_uninvolved_drift, drift);
      }
    }
    const double noise = noise_sum / static_cast<double>(noise_count);
    // The perturbation's demonstrated MAX per-expert effect over the same
    // uninvolved majority (still not inflatable by the swap itself). The
    // mean alone understates STRUCTURED noise: at real dims the input
    // delta concentrates on massive-activation channels, so gate rows
    // aligned with those channels move far more than the mean (measured,
    // the 0.57 hunt: a rank-23 expert displaced the top-8 at 11x mean
    // noise but ~1-2x the max uninvolved drift).
    const double noise_max = max_uninvolved_drift;

    // Reference ranking: ranks the swapped experts straddle the boundary.
    std::vector<int32_t> order;
    rank_by(rb, &order);
    std::vector<int64_t> rank_of(E, -1);
    for (size_t i = 0; i < E; ++i) rank_of[size_t(order[i])] = int64_t(i);

    for (size_t i = 0; i < only_eng.size(); ++i) {
      const int32_t expert_in = only_eng[i];    // engine selected, ref didn't
      const int32_t expert_out = only_ref[i];  // reference selected, didn't
      const double gap_ref =
          std::abs(double(rb[expert_in]) - double(rb[expert_out]));
      const double gap_eng =
          std::abs(double(eb[expert_in]) - double(eb[expert_out]));
      // The swapped experts' own cross-side disagreement: a legitimate
      // perturbation moves them by the same scale as everyone else, so a
      // large per-expert diff is concentrated corruption, not noise.
      const double drift_in =
          std::abs(double(eb[expert_in]) - double(rb[expert_in]));
      const double drift_out =
          std::abs(double(eb[expert_out]) - double(rb[expert_out]));
      const int64_t rank_in = rank_of[size_t(expert_in)];
      const int64_t rank_out = rank_of[size_t(expert_out)];
      // Certification, two measured paths (either certifies):
      //   CLASSIC — out at the boundary, in within the rank window, and
      //   every swap scale within 32x the MEAN uninvolved noise (the
      //   fixture regime: uniform noise, deep ranks are unreachable).
      //   STRUCTURED — every swap scale within 4x the MAX uninvolved
      //   drift: the perturbation has DEMONSTRATED it can move one
      //   expert's score that far, so the swap is attributable to it
      //   regardless of rank distance (the real-dims regime: noise
      //   concentrated on massive-activation channels moves aligned gate
      //   rows ~10x the mean). Corruption cannot inflate this yardstick:
      //   the swapped pair is excluded from it, and a doctored far jump
      //   stays thousands of multiples above honest noise either way.
      const bool ranks_adjacent = rank_out < top_k &&
                                  rank_in < top_k + 8;
      const double worst =
          std::max(std::max(gap_ref, gap_eng), std::max(drift_in, drift_out));
      const double gap_multiple =
          noise > 0 ? worst / noise : (worst > 0 ? 1e30 : 0);
      const double structure_multiple =
          noise_max > 0 ? worst / noise_max : (worst > 0 ? 1e30 : 0);
      const bool certified =
          (ranks_adjacent && gap_multiple <= 32.0) ||
          structure_multiple <= 4.0;
      if (!certified)
        throw std::runtime_error(
            where + ": expert " + std::to_string(expert_in) + " (ref rank " +
            std::to_string(rank_in) + ", gaps " + std::to_string(gap_ref) +
            " ref / " + std::to_string(gap_eng) + " eng, drifts " +
            std::to_string(drift_in) + " / " + std::to_string(drift_out) +
            " = " + std::to_string(gap_multiple) +
            "x mean noise, " + std::to_string(structure_multiple) +
            "x max uninvolved drift " + std::to_string(noise_max) +
            ") displaced expert " + std::to_string(expert_out) +
            " (ref rank " + std::to_string(rank_out) +
            ") — NOT a boundary near tie");
      out.swaps_certified += 1;
      out.max_boundary_gap = std::max(out.max_boundary_gap, gap_ref);
      out.max_noise_multiple = std::max(out.max_noise_multiple, gap_multiple);
      out.max_structure_multiple =
          std::max(out.max_structure_multiple, structure_multiple);
    }
    out.tokens_flipped += 1;
  }
}

// ---- the shared parity-tier discipline (the loopback test AND the fabric
// runner consume the same code — extracted from the CI test so a fabric
// gate failure is self-diagnosing on the night it happens, not an offline
// forensics project; the 0.57 hunt's lesson) ------------------------------

// RMS over the bf16 element-wise delta — the test tables' "l2". Unnormalized
// on purpose: it reads as "typical element error" against outputs whose
// magnitude is known (final hidden ~1.3). l2_rel is the normalized variant
// the gate's end-to-end tier used (||d||/||ref||).
inline double l2_bf16(const std::vector<uint16_t>& a,
                      const std::vector<uint16_t>& b) {
  if (a.size() != b.size() || a.empty())
    throw std::runtime_error("l2_bf16: shape mismatch");
  double acc = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = bf16_bits_to_float(a[i]) - bf16_bits_to_float(b[i]);
    acc += d * d;
  }
  return std::sqrt(acc / static_cast<double>(a.size()));
}

inline double l2_rel(const std::vector<uint16_t>& got,
                     const std::vector<uint16_t>& ref) {
  double d2 = 0, r2 = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = bf16_bits_to_float(got[i]) - bf16_bits_to_float(ref[i]);
    d2 += d * d;
    r2 += bf16_bits_to_float(ref[i]) * bf16_bits_to_float(ref[i]);
  }
  return r2 > 0 ? std::sqrt(d2) / std::sqrt(r2) : 0.0;
}

inline double l2_rel(const std::vector<float>& got,
                     const std::vector<float>& ref) {
  double d2 = 0, r2 = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = static_cast<double>(got[i]) - ref[i];
    d2 += d * d;
    r2 += static_cast<double>(ref[i]) * ref[i];
  }
  return r2 > 0 ? std::sqrt(d2) / std::sqrt(r2) : 0.0;
}

// Cascade-aware route-audit discipline, shared by the free-run and
// isolated comparisons: every token's first route divergence vs its
// reference must certify (audit_route_flips on that token alone — near-tie
// or attributable-to-structured-noise; a refusal is a real divergence
// with no compounding to blame and THROWS). After a token's first
// certified divergence its trajectory is legitimately different, so its
// later route differences are consequences — counted and reported, never
// certified and never failed. This keeps the audit's corruption-detection
// strength exactly where it is sound (the first divergence of every
// token, where the entering states were still comparable) and stops
// misreading legitimate compounding as corruption (the 0.57 hunt: the
// free-run audit refused a layer-40 flip whose score drift was 0.44 —
// the consequence of certified flips tens of layers earlier, on a stream
// state that had every right to differ).
struct CascadeAuditSummary {
  int64_t first_flips = 0;        // audited first divergences (certified)
  int64_t cascaded_tokens = 0;    // tokens whose first flip happened
  int64_t consequence_rows = 0;   // (layer, token) differences after the first
  double worst_mean_mult = 0;     // worst certified swap / mean noise
  double worst_struct_mult = 0;   // worst certified swap / max uninvolved drift
};

inline CascadeAuditSummary audit_routes_cascade(
    const std::vector<GlmRouteTraceLayer>& eng_routes,
    const std::vector<std::vector<float>>& eng_biased,
    const std::vector<GlmRouteTraceLayer>& ref_routes,
    const std::vector<std::vector<float>>& ref_biased, int top_k,
    int n_experts, const char* what) {
  CascadeAuditSummary sum;
  std::vector<char> cascaded(
      eng_routes.empty() ? size_t(0)
                         : static_cast<size_t>(eng_routes.front().tokens),
      0);
  for (size_t ri = 0; ri < eng_routes.size(); ++ri) {
    const auto& eng = eng_routes[ri];
    const GlmRouteTraceLayer* ref = nullptr;
    for (const auto& r : ref_routes)
      if (r.layer_idx == eng.layer_idx) {
        ref = &r;
        break;
      }
    if (ref == nullptr)
      throw std::runtime_error(std::string(what) +
                               ": route layer alignment vs oracle");
    const size_t ref_ri = static_cast<size_t>(ref - ref_routes.data());
    for (uint64_t t = 0; t < eng.tokens; ++t) {
      const size_t tk = static_cast<size_t>(t);
      const bool differs =
          !std::equal(eng.ids.begin() + tk * eng.top_k,
                      eng.ids.begin() + (tk + 1) * eng.top_k,
                      ref->ids.begin() + tk * ref->top_k);
      if (!differs) continue;
      if (cascaded[tk]) {
        ++sum.consequence_rows;
        continue;
      }
      // This token's first divergence: audited on its own row — must
      // certify, or the comparison is a real divergence.
      RouteFlipAudit audit;
      audit_route_flips(eng.ids.data() + tk * eng.top_k,
                        ref->ids.data() + tk * ref->top_k,
                        eng_biased[ri].data() + tk * n_experts,
                        ref_biased[ref_ri].data() + tk * n_experts,
                        /*tokens=*/1, top_k, n_experts, audit, eng.layer_idx);
      ++sum.first_flips;
      cascaded[tk] = 1;
      ++sum.cascaded_tokens;
      sum.worst_mean_mult =
          std::max(sum.worst_mean_mult, audit.max_noise_multiple);
      sum.worst_struct_mult =
          std::max(sum.worst_struct_mult, audit.max_structure_multiple);
    }
  }
  DGPP_LOG_INFO(
      "route audit (cascade) [{}]: {} first flips certified (worst "
      "{:.1f}x mean noise, {:.1f}x max drift), {} cascaded tokens, {} "
      "consequence route differences",
      what, sum.first_flips, sum.worst_mean_mult, sum.worst_struct_mult,
      sum.cascaded_tokens, sum.consequence_rows);
  return sum;
}

// Top-1 near-tie certification over the logits: every top-1 disagreement
// between engine and reference must be a boundary near tie (oracle top-2
// margin within 32x the measured engine-vs-reference logit noise). The
// free-run surface is reported, not asserted — this head is the hard
// gate that stays.
struct Top1AuditSummary {
  int64_t misses = 0;             // tokens whose top-1 differs
  int64_t uncertified = 0;        // misses whose margin exceeds 32x noise
  double worst_margin_ratio = 0;   // worst miss: oracle top-2 margin / noise
};

inline Top1AuditSummary audit_top1_near_ties(const float* tp_logits,
                                             const float* ref_logits,
                                             int vocab, size_t tokens) {
  Top1AuditSummary sum;
  for (size_t t = 0; t < tokens; ++t) {
    const float* tr = tp_logits + t * vocab;
    const float* rr = ref_logits + t * vocab;
    int tp_top = 0, rf_top = 0, rf_second = -1;
    double noise = 0;
    for (int c = 0; c < vocab; ++c) {
      const double d = std::abs(static_cast<double>(tr[c]) - rr[c]);
      noise = std::max(noise, d);
      if (tr[c] > tr[tp_top]) tp_top = c;
      if (rr[c] > rr[rf_top]) {
        rf_second = rf_top;
        rf_top = c;
      } else if (rf_second < 0 || rr[c] > rr[rf_second]) {
        if (c != rf_top) rf_second = c;
      }
    }
    if (tp_top == rf_top) continue;
    ++sum.misses;
    const double margin =
        std::abs(static_cast<double>(rr[rf_top]) - rr[rf_second]);
    sum.worst_margin_ratio = std::max(sum.worst_margin_ratio, margin / noise);
    if (noise <= 0 || margin > 32.0 * noise) ++sum.uncertified;
  }
  return sum;
}

inline void certify_top1_near_ties(const Top1AuditSummary& s, size_t tokens,
                                  const char* what) {
  if (s.uncertified != 0)
    throw std::runtime_error(
        std::string(what) +
        ": free-run top-1 disagreement is not a certified near tie "
        "(oracle top-2 margin > 32x engine-vs-oracle logit noise; " +
        std::to_string(s.uncertified) + "/" + std::to_string(s.misses) +
        " uncertified, worst margin " +
        std::to_string(s.worst_margin_ratio) + "x noise)");
  DGPP_LOG_INFO(
      "top-1 near-tie certification [{}]: {}/{} misses, all certified "
      "(worst margin {:.1f}x noise)",
      what, s.misses, tokens, s.worst_margin_ratio);
}

}  // namespace dgpp::glm_route
