// The Qwen3.8-Flash-Next YaRN rope knob (engine.rope_scaling, 2026-09-18).
//
// The invariant this file guards: with the knob OFF the rope table, the
// cos/sin the kernels build and the model's positional ceiling are what
// they were before the knob existed (the QSA layer calls qsa_rope_inv_freq
// and passes mscale 1.0f, which is bit-exact — tests/cuda/qsa_test.cu
// pins the table itself, this file pins the frozen bits the shared
// builder produces for the same parameters); with it ON the table is the
// one vLLM's YaRN builds for the user's 512K recipe.
//
// The goldens come from vllm/vllm-openai:qwen38-flash-next (the image the
// recipe launches, digest sha256:d464f3b4…), driven through the same call
// chain the QSA layer takes — start.sh's `--hf-overrides` merge into
// text_config.rope_parameters, then qwen3_8_flash_next/nvidia/qsa.py's
// get_rope(head_size=256, max_position=262144, rope_parameters=…):
//
//   effective rope_parameters = {mrope_interleaved: true, mrope_section:
//   [11,11,10], partial_rotary_factor: 0.25, rope_theta: 10000000,
//   rope_type: "yarn", factor: 2.0, original_max_position_embeddings: 262144}
//
// The merge matters: ModelConfig._update_nested recurses into the existing
// dict (verified in the container), so mrope_section survives and the QSA
// rope is an MRotaryEmbedding — whose cache is built at
// max_position_embeddings * 4 (mrope.py), and YaRN's correction band is
// computed from THAT value. Hence the two frozen bands below: 16/24 from
// 4 x 262144 (what the recipe runs) and 14/22 from 262144 (what a
// non-mrope YaRN would).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/rope_scaling.hpp"
#include "models/qwen/config.hpp"
#include "models/qwen/qsa_reference.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// The recipe's knob: factor 2 over the release's 262144 positions.
dgpp::RopeScaling recipe() {
  dgpp::RopeScaling rs;
  rs.factor = 2.0;
  rs.original_max_position_embeddings = 262144;
  return rs;
}

constexpr int kRotary = 64;  // the release's rotary_dim (256 x partial 0.25)
constexpr double kTheta = 1e7;

std::vector<uint32_t> bits_of(const std::vector<float>& v) {
  std::vector<uint32_t> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    uint32_t u = 0;
    static_assert(sizeof(float) == sizeof(uint32_t), "f32");
    std::memcpy(&u, &v[i], 4);
    out[i] = u;
  }
  return out;
}

std::vector<float> table(const dgpp::RopeScaling& rs) {
  std::vector<float> inv(static_cast<size_t>(kRotary / 2));
  dgpp::yarn_rope_inv_freq_host(kRotary, kTheta, rs.correction_max_position(), rs.factor,
                                rs.beta_fast, rs.beta_slow, inv.data());
  return inv;
}

void require_bits(const std::vector<float>& got, const uint32_t* want, size_t n,
                  const std::string& what) {
  require(got.size() == n, what + ": wrong length");
  const std::vector<uint32_t> b = bits_of(got);
  for (size_t i = 0; i < n; ++i) {
    if (b[i] == want[i]) continue;
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%s: lane %zu is 0x%08x, golden 0x%08x", what.c_str(), i,
                  b[i], want[i]);
    throw std::runtime_error(buf);
  }
}

// vLLM, the recipe (mrope cache x4 -> correction band 16/24): the fp32
// inv_freq `YaRNScalingRotaryEmbedding._compute_inv_freq` returned.
const uint32_t kYarnVllm[] = {
    0x3f800000, 0x3f1ab32b, 0x3ebaf81b, 0x3e61f835, 0x3e088d77, 0x3da50956, 0x3d47763f, 0x3cf11177,
    0x3c91ad39, 0x3c301052, 0x3bd4ca15, 0x3b80967d, 0x3b1b690d, 0x3abbd3ed, 0x3a6301e2, 0x3a092e02,
    0x39a5cb60, 0x393bdab5, 0x38d3e746, 0x386dcfcd, 0x3804a77e, 0x3792f6ed, 0x37217916, 0x36afa3bd,
    0x363cb0c1, 0x35e40cc6, 0x3589cf4a, 0x35268e4c, 0x34c94c57, 0x3473499d, 0x3413048e, 0x33b1af44};

// vLLM without mrope_section (YaRNScalingRotaryEmbedding over the native
// 262144 -> band 14/22). Kept as the second golden exactly because the
// difference is invisible in the knob's other fields: only the correction
// max position separates the two tables.
const uint32_t kYarnNativeBand[] = {
    0x3f800000, 0x3f1ab32b, 0x3ebaf81b, 0x3e61f835, 0x3e088d77, 0x3da50956, 0x3d47763f, 0x3cf11177,
    0x3c91ad39, 0x3c301052, 0x3bd4ca15, 0x3b80967d, 0x3b1b690d, 0x3abbd3ed, 0x3a6301e2, 0x3a009b22,
    0x399111f4, 0x3922ce9d, 0x38b5a1aa, 0x384939ae, 0x37dd1727, 0x37707cc9, 0x37012dab, 0x369c1fc4,
    0x363cb0c1, 0x35e40cc6, 0x3589cf4a, 0x35268e4c, 0x34c94c57, 0x3473499d, 0x3413048e, 0x33b1af44};

// The plain table: 1 / 1e7^(2i/64), frozen from the 2026-09-17 build (the
// binary that predates this knob) — the same bits qsa_rope_inv_freq and
// the shared builder's YaRN-off branch produce.
const uint32_t kPlain[] = {
    0x3f800000, 0x3f1ab32b, 0x3ebaf81b, 0x3e61f835, 0x3e088d77, 0x3da50956, 0x3d47763f, 0x3cf11177,
    0x3c91ad39, 0x3c301052, 0x3bd4ca15, 0x3b80967d, 0x3b1b690d, 0x3abbd3ed, 0x3a6301e2, 0x3a092e02,
    0x39a5cb60, 0x394860c1, 0x38f22ce2, 0x3892587e, 0x3830df52, 0x37d5c441, 0x37812dab, 0x371c1fc4,
    0x36bcb0c1, 0x36640cc6, 0x3609cf4a, 0x35a68e4c, 0x35494c57, 0x34f3499d, 0x3493048e, 0x3431af44};

// The DeepSeek-V4.1 CSA2 table (theta 160000, correction max 65536, factor
// 16), frozen from the same build: the shared builder's other caller must
// not move.
const uint32_t kDsv41Csa2[] = {
    0x3f800000, 0x3f300a3a, 0x3ef21c1f, 0x3ea67d01, 0x3e64f92e, 0x3e1d7475, 0x3dd88cb4, 0x3d94e963,
    0x3d4ccccd, 0x3d0cd4fb, 0x3cc1b019, 0x3c8530ce, 0x3c372dbf, 0x3bfbed88, 0x3bad3d5e, 0x3b6e4237,
    0x3b147ae1, 0x3ab714e0, 0x3a5ebdb6, 0x3a0530ce, 0x399bb3af, 0x39305978, 0x38be904d, 0x383e9b5f,
    0x37a3d70c, 0x36b443d0, 0x3677eba6, 0x362a7be8, 0x35ea77ff, 0x35a13bdc, 0x355dbf30, 0x35187c4d};

}  // namespace

DGPP_TEST(qwen_yarn_table_matches_the_vllm_recipe) {
  const dgpp::RopeScaling rs = recipe();
  require(rs.correction_max_position() == 1048576, "the mrope cache enlargement");
  const std::vector<float> inv = table(rs);
  require_bits(inv, kYarnVllm, std::size(kYarnVllm), "the YaRN table");
  // The ramp's band: full frequency through lane 16, half beyond lane 24
  // (plain/2), the mix in between. Lane 17 is the first mixed lane, which
  // is why the two bands above differ from lane 15 on.
  const std::vector<float> plain = [&] {
    std::vector<float> p(static_cast<size_t>(kRotary / 2));
    dgpp::yarn_rope_inv_freq_host(kRotary, kTheta, 0, 1.0, rs.beta_fast, rs.beta_slow, p.data());
    return p;
  }();
  for (int i = 0; i < 16; ++i) require(inv[static_cast<size_t>(i)] == plain[static_cast<size_t>(i)],
                                       "lanes inside the training band keep the plain frequency");
  require(inv[16] == plain[16], "lane 16 (ramp start)");
  require(inv[17] != plain[17], "lane 17 is scaled");
  for (int i = 24; i < 32; ++i)
    require(inv[static_cast<size_t>(i)] == plain[static_cast<size_t>(i)] / 2.0f,
            "beyond the band the frequency is divided by the factor");
}

DGPP_TEST(qwen_yarn_band_comes_from_the_mrope_enlarged_cache) {
  // The knob's mrope_cache_factor is the whole difference between the
  // recipe's table and a naive YaRN over the native context.
  dgpp::RopeScaling rs = recipe();
  require_bits(table(rs), kYarnVllm, std::size(kYarnVllm), "band from 4 x original");
  rs.mrope_cache_factor = 1.0;
  require(rs.correction_max_position() == 262144, "band from the native context");
  require_bits(table(rs), kYarnNativeBand, std::size(kYarnNativeBand), "band from 1 x original");
}

DGPP_TEST(qwen_plain_table_is_frozen) {
  // YaRN off: the shared builder's straight-through path is the plain
  // table, bit for bit what the engine served before the knob existed.
  std::vector<float> inv(static_cast<size_t>(kRotary / 2));
  dgpp::yarn_rope_inv_freq_host(kRotary, kTheta, /*correction_max_position=*/0, 2.0, 32.0, 1.0,
                                inv.data());
  require_bits(inv, kPlain, std::size(kPlain), "the plain table");
  // And the model with no knob reports the checkpoint's own ceiling.
  dgpp::QwenTextConfig cfg;
  cfg.max_position_embeddings = 262144;
  require(cfg.context_limit() == 262144, "the checkpoint's ceiling");
}

DGPP_TEST(dsv41_csa2_table_is_frozen) {
  // kernels/csa2.cu's builder was moved into the shared host function: its
  // other caller (the CSA2 compressed layers, theta 160000, correction max
  // 65536, factor 16) must keep its exact bits.
  std::vector<float> inv(32);
  dgpp::yarn_rope_inv_freq_host(64, 160000.0, 65536, 16.0, 32.0, 1.0, inv.data());
  require_bits(inv, kDsv41Csa2, std::size(kDsv41Csa2), "the CSA2 table");
}

DGPP_TEST(qwen_yarn_cos_sin_ride_the_vllm_cache) {
  // vLLM bakes mscale into the cos/sin cache (`freqs.cos() * self.mscale`
  // in fp32, then the query dtype's bf16) and leaves the attention scale
  // alone (nvidia/qsa.py: self.scaling = head_dim**-0.5). The kernels
  // therefore scale cosf/sinf before their bf16 rounding — these are the
  // bf16 values the recipe's cache holds, at lanes 0..3 of four positions.
  const dgpp::RopeScaling rs = recipe();
  const std::vector<float> inv = table(rs);
  const float mscale = rs.mscale();
  require(bits_of({mscale})[0] == 0x3f88df4e, "yarn_get_mscale(2) * attn_factor in f32");
  struct Golden {
    int64_t pos;
    uint16_t cos[4];
    uint16_t sin[4];
  };
  const Golden goldens[] = {
      {0, {0x3F89, 0x3F89, 0x3F89, 0x3F89}, {0x0000, 0x0000, 0x0000, 0x0000}},
      {12345, {0x3DF4, 0xBEB0, 0xBF88, 0xBF76}, {0xBF88, 0x3F82, 0x3DF2, 0xBEEF}},
      {262143, {0xBF27, 0x3F7D, 0xBF7F, 0x3E93}, {0x3F59, 0x3ED2, 0xBEC8, 0xBF84}},
      {524287, {0x3F38, 0x3E2B, 0x3EED, 0xBF46}, {0xBF4A, 0x3F87, 0x3F77, 0xBF3D}},
  };
  for (const Golden& g : goldens) {
    for (int i = 0; i < 4; ++i) {
      const float ang = static_cast<float>(g.pos) * inv[static_cast<size_t>(i)];
      const uint16_t c = dgpp::float_to_bf16_bits(std::cos(ang) * mscale);
      const uint16_t s = dgpp::float_to_bf16_bits(std::sin(ang) * mscale);
      if (c == g.cos[i] && s == g.sin[i]) continue;
      char buf[160];
      std::snprintf(buf, sizeof(buf),
                    "pos %lld lane %d: cos 0x%04x (golden 0x%04x), sin 0x%04x (golden 0x%04x)",
                    static_cast<long long>(g.pos), i, c, g.cos[i], s, g.sin[i]);
      throw std::runtime_error(buf);
    }
  }
  // The oracle's own reference (models/qwen/qsa_reference.cpp) takes the
  // same mscale, so the CUDA parity gate can run a YaRN model.
  std::vector<uint16_t> x(static_cast<size_t>(kRotary), dgpp::float_to_bf16_bits(1.0f));
  std::vector<uint16_t> w(static_cast<size_t>(kRotary), dgpp::float_to_bf16_bits(0.0f));
  std::vector<uint16_t> out(static_cast<size_t>(kRotary));
  const std::vector<float> yarn = [&] {
    std::vector<float> t(static_cast<size_t>(kRotary / 2));
    dgpp::qwen_ref::rope_inv_freq_yarn(kTheta, kRotary, rs.correction_max_position(), rs.factor,
                                       rs.beta_fast, rs.beta_slow, t);
    return t;
  }();
  require(bits_of(yarn) == bits_of(inv), "the reference's table is the engine's");
  dgpp::qwen_ref::qsa_norm_rope(x.data(), w.data(), 0, yarn.data(), out.data(), kRotary, kRotary,
                                1e-6f, mscale);
  require(out[0] == 0x3F89 && out[32] == 0x3F89, "the reference rotates with the mscale");
}

DGPP_TEST(mrope_interleaved_is_the_identity_for_text_positions) {
  // Qwen's mrope_section [11,11,10] with mrope_interleaved (the release's
  // rope_parameters; tests/unit/qwen_config_test.cpp gates the parse). The
  // layout below is vLLM's apply_interleaved_rope / the triton mrope
  // kernel's mask (mrope.py): height lanes are lane % 3 == 1 below
  // 3 * section[1], width lanes lane % 3 == 2 below 3 * section[2], the
  // rest text. For text-only positions all three position rows carry the
  // same values, so the pick is the identity on every lane and the YaRN
  // table's lane i is exactly the lane the rope pairs: no permutation.
  const int section[3] = {11, 11, 10};
  const int half = kRotary / 2;
  require(section[0] + section[1] + section[2] == half, "the sections sum to rotary_dim / 2");
  const auto interleaved_pick = [&](int lane) {
    const bool is_h = (lane % 3 == 1) && (lane < section[1] * 3);
    const bool is_w = (lane % 3 == 2) && (lane < section[2] * 3);
    return is_w ? 2 : (is_h ? 1 : 0);
  };
  const auto chunked_pick = [&](int lane) {
    return lane < section[0] ? 0 : (lane < section[0] + section[1] ? 1 : 2);
  };
  int counts[3] = {0, 0, 0};
  for (int lane = 0; lane < half; ++lane) ++counts[interleaved_pick(lane)];
  require(counts[0] == section[0] && counts[1] == section[1] && counts[2] == section[2],
          "the interleaved layout covers [11, 11, 10] lanes exactly once");
  // The interleaved layout is not the chunked one: lane 12 is a t lane
  // (12 % 3 == 0) while chunking puts it in the h slice, and lane 1 is an
  // h lane while chunking puts it in the t slice.
  require(interleaved_pick(1) == 1 && chunked_pick(1) == 0, "lane 1");
  require(interleaved_pick(12) == 0 && chunked_pick(12) == 1, "lane 12");
  require(interleaved_pick(2) == 2 && chunked_pick(2) == 0, "lane 2");
  // The picked value: row r's lane i holds (text_only ? i : 1000 * r + i).
  // With the three rows equal (text), every lane resolves to its own lane
  // index; with distinct rows the pick is visible.
  const auto picked = [&](int lane, bool text_only) {
    const int r = interleaved_pick(lane);
    return text_only ? lane : 1000 * r + lane;
  };
  for (int lane = 0; lane < half; ++lane) {
    require(picked(lane, true) == lane, "the table's lane is the crate's lane for text");
    require(picked(lane, false) >= lane, "distinct rows make the pick observable");
  }
  require(picked(1, false) != picked(1, true), "lane 1 comes from the h row when rows differ");
}

DGPP_TEST(rope_scaling_knob_refuses_what_cannot_work) {
  const auto refuses = [](const dgpp::RopeScaling& rs) {
    try {
      rs.validate("engine.rope_scaling");
    } catch (const std::runtime_error&) {
      return true;
    }
    return false;
  };
  dgpp::RopeScaling rs = recipe();
  require(!refuses(rs), "the recipe validates");
  require(rs.mscale() > 1.0f, "factor 2 scales the rotation");
  require(rs.context_limit() == 524288, "262144 x 2");
  rs.factor = 0.5;
  require(refuses(rs), "a sub-unity factor");
  rs = recipe();
  rs.original_max_position_embeddings = 0;
  require(refuses(rs), "no original context");
  rs = recipe();
  rs.beta_slow = 64.0;
  require(refuses(rs), "beta_slow above beta_fast");
  rs = recipe();
  rs.attn_factor = 0.0;
  require(refuses(rs), "a zero attention factor");
  rs = recipe();
  rs.mrope_cache_factor = 0.5;
  require(refuses(rs), "a shrunken mrope cache");
  rs = recipe();
  rs.factor = 1.0;
  require(rs.mscale() == 1.0f && rs.context_limit() == 262144,
          "factor 1 is the plain table and the native ceiling");
  // The derived arithmetic's representable ranges (the 2026-09-18 review):
  // the cluster-config JSON's number() leaves the fields unbounded, so the
  // validation must refuse what the table builder and the context
  // arithmetic cannot represent — overflow and underflow, not just the
  // lower bounds. Each case below passed the lower-bound checks (and the
  // JSON parser) and reached the derived values before it was caught.
  rs = recipe();
  rs.attn_factor = 1e40;
  require(refuses(rs), "an attn_factor that overflows the mscale to infinity");
  rs = recipe();
  rs.attn_factor = 1e-46;
  require(refuses(rs), "an attn_factor that underflows the mscale to zero");
  rs = recipe();
  rs.factor = 1e300;
  require(refuses(rs), "a factor whose context product (original x factor) overflows 2^63");
  rs = recipe();
  rs.mrope_cache_factor = 1e300;
  require(refuses(rs), "an mrope_cache_factor whose correction-band product overflows 2^63");
  rs = recipe();
  rs.beta_fast = 1e308;
  // beta_fast x 2 pi overflows to inf: the band argument collapses to 0,
  // log(0) = -inf, floor(-inf) is undefined in the table builder.
  require(refuses(rs), "a beta_fast whose band argument collapses to zero");
  rs = recipe();
  rs.beta_slow = 5e-324;  // the smallest positive subnormal: it passes beta_slow > 0
  // beta_slow x 2 pi underflows to a subnormal: the band argument diverges
  // to +inf.
  require(refuses(rs), "a subnormal beta_slow whose band argument diverges to infinity");
  rs = recipe();
  rs.original_max_position_embeddings = INT64_MAX;
  rs.factor = 2.0;
  require(refuses(rs), "an original x factor product past 2^63 (the context limit's llround)");
  // The exact-2^63 edges (the 2026-09-19 review, F2): a product of EXACTLY
  // 2^63 still overflows the int64 llround, so the bound is >=, not >.
  rs = recipe();
  rs.original_max_position_embeddings = 1LL << 62;
  rs.factor = 2.0;
  require(refuses(rs), "an original x factor product of exactly 2^63");
  rs = recipe();
  rs.original_max_position_embeddings = INT64_MAX;
  rs.factor = 1.0;
  require(refuses(rs), "an INT64_MAX original x 1.0 (rounds to 2^63 in the double)");
  // The knob is the model's positional ceiling: with it on, the Qwen config
  // reports the scaled limit and the memory plan's context line follows.
  dgpp::QwenTextConfig cfg;
  cfg.max_position_embeddings = 262144;
  cfg.rope_scaling = recipe();
  require(cfg.context_limit() == 524288, "the YaRN ceiling");
  cfg.rope_scaling->mrope_cache_factor = 1.0;
  require(cfg.context_limit() == 524288, "the ceiling is original x factor, not the band");
}

DGPP_TEST(yarn_inv_freq_builder_refuses_degenerate_theta) {
  // The table builder's theta edges (the 2026-09-19 review, F1): validate()
  // cannot see theta (a checkpoint field), so the builder guards its own
  // arithmetic — theta == 1.0 zeroes the correction band's log(theta)
  // division (±inf/NaN into the int casts), and a subnormal theta underflows
  // the plain table's pow to 0 (1/0 = inf).
  const auto throws = [](double theta) {
    std::vector<float> out(32);
    try {
      dgpp::yarn_rope_inv_freq_host(64, theta, 262144 * 4, 2.0, 32.0, 1.0, out.data());
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  require(throws(1.0), "theta == 1.0 zeroes the band's log(theta) division");
  require(throws(1e-308), "a subnormal theta underflows the table's pow");
  // The recipe's theta still builds a finite table (the frozen values pin it).
  std::vector<float> out(32);
  dgpp::yarn_rope_inv_freq_host(64, 1e7, 262144 * 4, 2.0, 32.0, 1.0, out.data());
  require(std::all_of(out.begin(), out.end(), [](float v) { return std::isfinite(v) && v > 0; }),
          "the recipe's theta builds a finite table");
}
