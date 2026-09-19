#pragma once
// The host-side rope tables and their opt-in YaRN ramp (2026-09-18). One
// builder serves every family that scales its rope: the DeepSeek-V4.1
// CSA2 compressed layers (kernels/csa2.cu — this is that builder, moved
// to a host translation unit so a non-CUDA test can gate it, with the
// arithmetic unchanged) and the Qwen3.8-Flash-Next QSA rope behind the
// engine's `rope_scaling` knob. `RopeScaling` is the knob's spec, shared
// by the cluster-config parser, the Qwen text config and the tests.
//
// The reference for everything here is vLLM as the user's recipe runs it
// (vllm/model_executor/layers/rotary_embedding/): yarn_scaling_rope.py's
// `_compute_inv_freq` + `_compute_cos_sin_cache`, common.py's
// `yarn_find_correction_range` / `yarn_linear_ramp_mask` /
// `yarn_get_mscale`, and mrope.py's cache enlargement. Verified bit for
// bit against vllm/vllm-openai:qwen38-flash-next on 2026-09-18; see
// tests/unit/qwen_rope_scaling_test.cpp for the frozen values.
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace dgpp {

// `engine.rope_scaling`: the YaRN ramp the serving stack applies to a
// family's rope instead of the checkpoint's plain table. Absent = off,
// and with it off every table and every kernel argument is what it was
// before this knob existed.
struct RopeScaling {
  // vLLM's spellings (the `rope_parameters` keys of the yarn branch in
  // get_rope). `mrope_cache_factor` is the one name vLLM does not use as
  // a key: MRotaryEmbedding builds its cache at
  // `max_position_embeddings * 4` (mrope.py: `cache_max_position_num`)
  // and YaRN's correction band is computed from THAT value, so a
  // checkpoint whose rope_parameters carry `mrope_section` — the Qwen3.8
  // recipe does, the launcher's hf-override merges into the existing
  // dict rather than replacing it — gets its band from 4 x original
  // (low/high 16/24 at rotary 64) and not from the original (14/22).
  // 4.0 reproduces the vLLM recipe; 1.0 uses the native context.
  double factor = 1.0;
  int64_t original_max_position_embeddings = 0;
  double beta_fast = 32.0;
  double beta_slow = 1.0;
  double attn_factor = 1.0;  // vLLM's `attn_factor`; the effective scale is yarn_get_mscale(factor) * attn_factor
  double mrope_cache_factor = 4.0;

  // The max position the ramp's correction band is computed from (vLLM:
  // the rope object's own max_position_embeddings, i.e. the enlarged cache).
  int64_t correction_max_position() const {
    return static_cast<int64_t>(std::llround(
        static_cast<double>(original_max_position_embeddings) * mrope_cache_factor));
  }
  // The context the ramp extends to: original x factor (the recipe's
  // MAX_MODEL_LEN = 262144 x 2 = 524288). vLLM additionally holds a
  // factor x mrope_cache_factor x original cache; positions past this
  // limit are not chased, the pool bound is ours to set.
  int64_t context_limit() const {
    return static_cast<int64_t>(std::llround(
        static_cast<double>(original_max_position_embeddings) * factor));
  }
  // vLLM's yarn_get_mscale(factor) * attn_factor, in the f32 the cos/sin
  // tables are built with. It rides the ROTATION, not the softmax scale:
  // vLLM's attention keeps `head_dim**-0.5` (nvidia/qsa.py) and the
  // mscale is baked into the bf16 cos/sin cache, so only the rotated
  // lanes' dot product is scaled (by mscale^2), the pass-through lanes
  // are not.
  float mscale() const {
    const double m = factor <= 1.0 ? 1.0 : 0.1 * std::log(factor) + 1.0;
    return static_cast<float>(m * attn_factor);
  }
  // The knobs an operator may set; every refusal names the field.
  void validate(const std::string& what) const {
    if (!(factor >= 1.0)) throw std::runtime_error(what + ": factor must be >= 1");
    if (original_max_position_embeddings <= 0)
      throw std::runtime_error(what + ": original_max_position_embeddings must be positive");
    if (!(beta_fast > beta_slow) || !(beta_slow > 0))
      throw std::runtime_error(what + ": beta_fast must exceed beta_slow > 0");
    if (!(attn_factor > 0)) throw std::runtime_error(what + ": attn_factor must be > 0");
    if (!(mrope_cache_factor >= 1.0))
      throw std::runtime_error(what + ": mrope_cache_factor must be >= 1");
    // The derived arithmetic's representable ranges (the 2026-09-18
    // review): the cluster-config JSON leaves every field an unbounded
    // number(), so an inf slips through the comparisons above (a NaN is
    // already refused by them, but be explicit), and the table builder
    // and the context products below must not see one.
    if (!std::isfinite(factor)) throw std::runtime_error(what + ": factor must be finite");
    if (!std::isfinite(beta_fast)) throw std::runtime_error(what + ": beta_fast must be finite");
    if (!std::isfinite(beta_slow)) throw std::runtime_error(what + ": beta_slow must be finite");
    if (!std::isfinite(attn_factor)) throw std::runtime_error(what + ": attn_factor must be finite");
    if (!std::isfinite(mrope_cache_factor))
      throw std::runtime_error(what + ": mrope_cache_factor must be finite");
    // The scale the cos/sin table is built with: an inf (attn_factor 1e40
    // overflows the float) or a 0 (1e-46 underflows it) both leave the
    // table degenerate — at position zero the rotated lanes are
    // non-finite or identically unscaled.
    const float m = mscale();
    if (!std::isfinite(m) || !(m > 0.0f))
      throw std::runtime_error(
          what + ": the mscale (yarn_get_mscale(factor) x attn_factor) must be a finite positive float");
    // The two llrounds' products: past 2^63 the llround in
    // context_limit() / correction_max_position() is undefined (the
    // products are compared in the double they land in; 2^63 is exact).
    const double original = static_cast<double>(original_max_position_embeddings);
    // >=, not >: a product of EXACTLY 2^63 (original 2^62 x factor 2, or
    // INT64_MAX x 1.0 rounded up in the double) still overflows the int64
    // llround below (the 2026-09-19 review, F2).
    if (original * factor >= 9.223372036854775808e18)
      throw std::runtime_error(
          what + ": original_max_position_embeddings x factor exceeds 2^63 (the context limit)");
    if (original * mrope_cache_factor >= 9.223372036854775808e18)
      throw std::runtime_error(what +
                               ": original_max_position_embeddings x mrope_cache_factor exceeds 2^63 "
                               "(the correction band)");
    // The correction band's argument, the builder's
    // corrected_dim(beta) = dim * log(correction_max / (beta x 2 pi)) /
    // (2 log theta) (rope_scaling.cpp): beta_fast 1e308 overflows
    // beta x 2 pi to inf, the argument collapses to 0, log(0) = -inf and
    // floor(-inf) is undefined; a subnormal beta_slow pushes the argument
    // to +inf. Both must be a positive finite before the table is built.
    const double kPi = 3.14159265358979323846;  // the builder's literal, not M_PI
    const auto band_argument = [&](double beta) {
      return static_cast<double>(correction_max_position()) / (beta * 2.0 * kPi);
    };
    if (!std::isfinite(band_argument(beta_fast)) || !(band_argument(beta_fast) > 0.0))
      throw std::runtime_error(what + ": beta_fast's correction band argument must be a positive finite");
    if (!std::isfinite(band_argument(beta_slow)) || !(band_argument(beta_slow) > 0.0))
      throw std::runtime_error(what + ": beta_slow's correction band argument must be a positive finite");
  }
  bool operator==(const RopeScaling&) const = default;
};

// The rotary inverse frequencies, fp32, as vLLM's
// `YaRNScalingRotaryEmbedding._compute_inv_freq` builds them:
// 1 / theta^(2i/dim), and with `correction_max_position > 0` the YaRN mix —
// lanes whose wavelength fits the band keep their frequency, those far
// beyond it are divided by `factor`, the beta_fast..beta_slow band in
// between fades linearly. `out`: [rope_dim/2].
void yarn_rope_inv_freq_host(int rope_dim, double theta, int64_t correction_max_position,
                             double factor, double beta_fast, double beta_slow, float* out);

}  // namespace dgpp
