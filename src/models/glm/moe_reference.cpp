#include "models/glm/moe_reference.hpp"

#include "kernels/hadamard32.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/dtypes.hpp"
#include "kernels/latent_format.hpp"
#include "loaders/packq_quant.hpp"

namespace dgpp {

namespace {

double bf16_to_d(uint16_t v) {
  return static_cast<double>(bf16_bits_to_float(v));
}
uint16_t d_to_bf16(double v) {
  return float_to_bf16_bits(static_cast<float>(v));
}
double sigmoid_d(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// Weights in the payload are stored as bf16-rounded dequants on the STRICT
// path — the engine's scale_gemm rounds decoded weights to bf16, and the
// oracle must match that policy to isolate only accumulation gaps.
struct StrictQuantRow {
  std::vector<double> w;  // [cols]
};

// D[rows_out, cols_out] = act[rows_out, cols_in] @ W^T with W [n, k]:
// strict GEMM — bf16-rounded weights, double accumulation, bf16-rounded
// output (the engine's output dtype).
// `divisor` is the NVFP4 matrix's global scale, applied once to the
// finished dot as the engine's epilogue does (1 for FP8).
void strict_gemm(const std::vector<uint16_t>& act, int64_t act_stride,
                 const std::vector<double>& weights_bf16, int m, int n,
                 int k, std::vector<uint16_t>& out, double divisor = 1.0) {
  out.assign(static_cast<size_t>(m) * n, 0);
  for (int nn = 0; nn < n; ++nn) {
    const double* wrow = weights_bf16.data() + static_cast<size_t>(nn) * k;
    for (int mm = 0; mm < m; ++mm) {
      double acc = 0.0;
      const uint16_t* arow =
          act.data() + static_cast<size_t>(mm) * act_stride;
      for (int kk = 0; kk < k; ++kk)
        acc += bf16_to_d(arow[kk]) * wrow[kk];
      out[static_cast<size_t>(mm) * n + nn] = d_to_bf16(acc / divisor);
    }
  }
}

// The down projection: the same strict GEMM with the accumulator handed
// back UNROUNDED (the engine's launch_scale_gemm_f32) — the accumulation
// chain owns the single rounding.
void strict_gemm_raw(const std::vector<uint16_t>& act, int64_t act_stride,
                     const std::vector<double>& weights_bf16, int m, int n,
                     int k, std::vector<double>& out, double divisor = 1.0) {
  out.assign(static_cast<size_t>(m) * n, 0.0);
  for (int nn = 0; nn < n; ++nn) {
    const double* wrow = weights_bf16.data() + static_cast<size_t>(nn) * k;
    for (int mm = 0; mm < m; ++mm) {
      double acc = 0.0;
      const uint16_t* arow =
          act.data() + static_cast<size_t>(mm) * act_stride;
      for (int kk = 0; kk < k; ++kk)
        acc += bf16_to_d(arow[kk]) * wrow[kk];
      out[static_cast<size_t>(mm) * n + nn] = acc / divisor;
    }
  }
}

// NVFP4 weights: e2m1(code) x e4m3(scale), EXACT (<= 6 significant bits;
// the engine keeps it in fp32 unrounded) — the global scale divides the
// finished dot, not the weight (docs/nvfp4_plan.md §3.1).
// MXFP4 (scale_group 32): e2m1(code) x 2^(e8m0 - 127), exact; no global.
std::vector<double> dequant_fp4_weights(const GlmFp4MatrixHost& mat) {
  const int g = mat.scale_group;
  const int64_t pc = mat.cols / 2, sc = mat.cols / g;
  std::vector<double> w(static_cast<size_t>(mat.rows) * mat.cols);
  for (int64_t r = 0; r < mat.rows; ++r)
    for (int64_t c = 0; c < mat.cols; ++c) {
      const uint8_t byte = mat.payload[static_cast<size_t>(r) * pc + c / 2];
      const uint8_t code = (c & 1) ? static_cast<uint8_t>(byte >> 4)
                                   : static_cast<uint8_t>(byte & 0xFu);
      const uint8_t sb = mat.scales[static_cast<size_t>(r) * sc + c / g];
      const double s = g == 32 ? (sb == 255 ? std::nan("") : std::ldexp(1.0, static_cast<int>(sb) - 127))
                               : static_cast<double>(fp8_e4m3_bits_to_float(sb));
      w[static_cast<size_t>(r) * mat.cols + c] =
          static_cast<double>(fp4_e2m1_bits_to_float(code)) * s;
    }
  return w;
}

// Packed-int weights: level x scale, EXACT (the engine's exact policy,
// docs/glm53_plan.md D2: no bf16 rounding of the weight; the
// group-factored fp32 chain differs from this by summation order only).
// The level is code - 2^(bits-1), or the codebook entry the stored index
// selects under the NF4I8 format (packed_code_level).
std::vector<double> dequant_packq_weights(const GlmPackedMatrixHost& mat) {
  const int g = packed_scale_group(mat.scale_fmt);
  const int64_t sc = mat.cols / g;
  std::vector<double> w(static_cast<size_t>(mat.rows) * mat.cols);
  for (int64_t r = 0; r < mat.rows; ++r)
    for (int64_t c = 0; c < mat.cols; ++c) {
      // The stored field (a dense stream under the Mixed346 format: the
      // host codec reads across word boundaries) to its level.
      const int level = packq_level(mat.packed, mat.cols, mat.bits, r, c, mat.scale_fmt);
      const float s = packed_scale_to_float(mat.scales[static_cast<size_t>(r) * sc + c / g], mat.scale_fmt);
      w[static_cast<size_t>(r) * mat.cols + c] = static_cast<double>(level) * static_cast<double>(s);
    }
  return w;
}

// The Mixed346 activation codes of one rotated bf16 row (the quantizer's
// arithmetic, kernels/hadamard32.hpp) as the values they represent, code
// x scale, in double: the oracle multiplies those.
std::vector<double> a8_represented(const std::vector<uint16_t>& rotated) {
  std::vector<double> out(rotated.size());
  int8_t codes[128];
  for (size_t c = 0; c < rotated.size(); c += 128) {
    const float s = hadamard32_quant_int8_group_host(rotated.data() + c, codes);
    for (int j = 0; j < 128; ++j) out[c + j] = static_cast<double>(codes[j]) * static_cast<double>(s);
  }
  return out;
}
// strict_gemm / strict_gemm_raw over double activations (the represented
// int8 codes): bf16-rounded or raw outputs as those.
void strict_gemm_d(const std::vector<double>& act, const std::vector<double>& weights, int n, int k,
                   std::vector<uint16_t>& out) {
  out.assign(static_cast<size_t>(n), 0);
  for (int nn = 0; nn < n; ++nn) {
    const double* wrow = weights.data() + static_cast<size_t>(nn) * k;
    double acc = 0.0;
    for (int kk = 0; kk < k; ++kk) acc += act[static_cast<size_t>(kk)] * wrow[kk];
    out[static_cast<size_t>(nn)] = d_to_bf16(acc);
  }
}
void strict_gemm_raw_d(const std::vector<double>& act, const std::vector<double>& weights, int n, int k,
                       std::vector<double>& out) {
  out.assign(static_cast<size_t>(n), 0.0);
  for (int nn = 0; nn < n; ++nn) {
    const double* wrow = weights.data() + static_cast<size_t>(nn) * k;
    double acc = 0.0;
    for (int kk = 0; kk < k; ++kk) acc += act[static_cast<size_t>(kk)] * wrow[kk];
    out[static_cast<size_t>(nn)] = acc;
  }
}

// The routed experts' H32 input rotation (kernels/hadamard32.hpp's
// arithmetic: fp32 butterflies, one bf16 rounding) of one bf16 row.
std::vector<uint16_t> rotate_h32(const std::vector<uint16_t>& row) {
  std::vector<uint16_t> out(row.size());
  hadamard32_rows_host(row.data(), row.size(), out.data(), out.size(), 1, static_cast<int>(row.size()));
  return out;
}

// Decode payload x scales, rounded to bf16 (the engine's weight policy).
std::vector<double> dequant_bf16_weights(const GlmQuantMatrixHost& mat) {
  const int64_t scale_cols = (mat.cols + 127) / 128;
  std::vector<double> w(static_cast<size_t>(mat.rows) * mat.cols);
  for (int64_t r = 0; r < mat.rows; ++r)
    for (int64_t c = 0; c < mat.cols; ++c) {
      const float dec = fp8_e4m3_bits_to_float(
          mat.payload[static_cast<size_t>(r) * mat.cols + c]);
      const float s =
          mat.scales[static_cast<size_t>(r / 128) * scale_cols + c / 128];
      w[static_cast<size_t>(r) * mat.cols + c] =
          bf16_to_d(float_to_bf16_bits(dec * s));
    }
  return w;
}

// The expert swiglu chain on already-computed gate/up bf16 outputs.
void swiglu_oracle(const std::vector<uint16_t>& gate,
                   const std::vector<uint16_t>& up, int64_t n, float limit,
                   std::vector<uint16_t>& out) {
  out.assign(gate.size(), 0);
  for (int64_t i = 0; i < n; ++i) {
    double g = bf16_to_d(gate[i]);
    double u = bf16_to_d(up[i]);
    if (g > limit) g = limit;
    u = std::min(std::max(u, static_cast<double>(-limit)),
                 static_cast<double>(limit));
    const uint16_t t = d_to_bf16(g * sigmoid_d(g));  // rounding 1
    out[i] = d_to_bf16(bf16_to_d(t) * u);            // rounding 2
  }
}

}  // namespace

GlmQuantMatrixHost glm_moe_host_view(const GlmMoeHostWeights& w,
                                     const GlmMoeConfig& cfg, int index) {
  // Payload layout: [the fp8 expert matrices in index order][3 shared
  // matrices], each expert triple ordered gate, up, down; gate/up [I, H],
  // down [H, I]. `index` is the LOGICAL index (e * 3 + m, the shared
  // triple at E * 3 + m); its position in the fp8 vectors is fp8_ordinal
  // (every matrix has the same element count, so the offsets stride by it).
  const int64_t I = cfg.inter, H = cfg.hidden;
  GlmQuantMatrixHost m;
  const bool down = index % 3 == 2;
  m.rows = down ? H : I;
  m.cols = down ? I : H;
  const int64_t numel = m.rows * m.cols;
  const int64_t scale_numel =
      ((m.rows + 127) / 128) * ((m.cols + 127) / 128);
  const int ord = w.fp8_ordinal(index, cfg.n_experts);
  const size_t p_off = static_cast<size_t>(ord) * numel;
  const size_t s_off = static_cast<size_t>(ord) * scale_numel;
  if (p_off + numel > w.payloads.size() || s_off + scale_numel > w.scales.size())
    throw std::invalid_argument("glm_moe_host_view: index out of range");
  m.payload = w.payloads.data() + p_off;
  m.scales = w.scales.data() + s_off;
  return m;
}

GlmQuantMatrixHost glm_moe_host_shared(const GlmMoeHostWeights& w,
                                       const GlmMoeConfig& cfg, int m) {
  // The shared triple follows the fp8 routed matrices (none under uniform
  // NVFP4 / packq, the fp8 ones of a per-projection mix).
  return glm_moe_host_view(w, cfg, cfg.n_experts * 3 + m);
}

GlmPackedMatrixHost glm_moe_host_view_packq(const GlmMoeHostWeights& w,
                                            const GlmMoeConfig& cfg, int index) {
  const int E = cfg.n_experts;
  if (!w.packq || index < 0 || index >= w.packq_matrices(E))
    throw std::invalid_argument("glm_moe_host_view_packq: not a packed-int index");
  const int64_t I = cfg.inter, H = cfg.hidden;
  GlmPackedMatrixHost m;
  const bool down = index % 3 == 2;
  m.rows = down ? H : I;
  m.cols = down ? I : H;
  // Every matrix has the same element count (gate/up [I,H], down [H,I]);
  // each carries its own width and scale format (uniform routed ones, the
  // shared triple at its width in bf16 per 64, or the Mixed346 per-matrix
  // forms), so a matrix's offset is the sum of the sizes before it.
  const int64_t elems = I * H;
  m.bits = w.packq_bits_of(index, E);
  m.scale_fmt = w.packq_fmt_of(index, E);
  size_t p_off = 0, s_off = 0;
  for (int i = 0; i < index; ++i) {
    p_off += static_cast<size_t>(elems * w.packq_bits_of(i, E) / 32);
    s_off += static_cast<size_t>(elems / packed_scale_group(w.packq_fmt_of(i, E)));
  }
  const size_t words = static_cast<size_t>(elems * m.bits / 32);
  const size_t scales_per = static_cast<size_t>(elems / packed_scale_group(m.scale_fmt));
  if (p_off + words > w.packq_words.size() || s_off + scales_per > w.packq_scales.size())
    throw std::invalid_argument("glm_moe_host_view_packq: index out of range");
  m.packed = w.packq_words.data() + p_off;
  m.scales = w.packq_scales.data() + s_off;
  return m;
}

GlmFp4MatrixHost glm_moe_host_view_fp4(const GlmMoeHostWeights& w,
                                       const GlmMoeConfig& cfg, int index) {
  if (!w.nvfp4 || index < 0 || !w.fp4_of(index, cfg.n_experts))
    throw std::invalid_argument("glm_moe_host_view_fp4: not an NVFP4 index");
  const int ord = w.fp4_ordinal(index, cfg.n_experts);
  const int64_t I = cfg.inter, H = cfg.hidden;
  GlmFp4MatrixHost m;
  const bool down = index % 3 == 2;
  m.rows = down ? H : I;
  m.cols = down ? I : H;
  // Every routed matrix has the same byte counts (gate/up [I,H], down [H,I]).
  if (w.fp4_group != 16 && w.fp4_group != 32)
    throw std::invalid_argument("glm_moe_host_view_fp4: the fp4 scale group must be 16 or 32");
  m.scale_group = w.fp4_group;
  const int64_t pbytes = m.rows * m.cols / 2, sbytes = m.rows * m.cols / w.fp4_group;
  const size_t p_off = static_cast<size_t>(ord) * pbytes;
  const size_t s_off = static_cast<size_t>(ord) * sbytes;
  const bool mx = w.fp4_group == 32;
  if (p_off + pbytes > w.fp4_payloads.size() || s_off + sbytes > w.fp4_scales.size() ||
      (!mx && static_cast<size_t>(ord) >= w.fp4_globals.size()))
    throw std::invalid_argument("glm_moe_host_view_fp4: index out of range");
  m.payload = w.fp4_payloads.data() + p_off;
  m.scales = w.fp4_scales.data() + s_off;
  m.global_scale = mx ? 1.0f : w.fp4_globals[static_cast<size_t>(ord)];
  return m;
}

void glm_moe_ref_router(const uint16_t* hidden, const uint16_t* gate,
                        const float* bias, const GlmMoeConfig& cfg,
                        int tokens, GlmMoeRouterRef& out) {
  GlmMoeConfig::validate_config(cfg);
  const int E = cfg.n_experts, H = cfg.hidden, K = cfg.top_k;
  out.ids.assign(static_cast<size_t>(tokens) * K, 0);
  out.weights.assign(static_cast<size_t>(tokens) * K, 0.f);
  out.biased.assign(static_cast<size_t>(tokens) * E, 0.0);

  const bool softmax = cfg.router_mode == MoeRouterMode::SoftmaxTopk;
  if (!softmax && bias == nullptr)
    throw std::invalid_argument("glm_moe_ref_router: the sigmoid mode needs a bias");
  for (int t = 0; t < tokens; ++t) {
    const uint16_t* x = hidden + static_cast<size_t>(t) * H;
    std::vector<double> scores(E), biased(E);
    for (int e = 0; e < E; ++e) {
      double dot = 0.0;
      const uint16_t* w = gate + static_cast<size_t>(e) * H;
      for (int k = 0; k < H; ++k) dot += bf16_to_d(x[k]) * bf16_to_d(w[k]);
      if (softmax) {
        // The reference's bf16 Linear output: the logit rounded once. It
        // is the selection key and the exported row; the probability is
        // the picked weight (below).
        scores[e] = bf16_to_d(d_to_bf16(dot));
        biased[e] = scores[e];
      } else if (cfg.router_mode == MoeRouterMode::SqrtSoftplusBias) {
        // DeepSeek-V4.1: sqrt(softplus(dot)) with torch's threshold (20).
        scores[e] = std::sqrt(dot > 20.0 ? dot : std::log1p(std::exp(dot)));
        biased[e] = scores[e] + bias[e];
      } else {
        scores[e] = sigmoid_d(dot);
        biased[e] = scores[e] + bias[e];
      }
      out.biased[static_cast<size_t>(t) * E + e] = biased[e];
    }
    if (softmax) {
      // softmax over the logits in double; the weight of expert e is p_e.
      double m = -INFINITY;
      for (int e = 0; e < E; ++e) m = std::max(m, scores[e]);
      double sum = 0.0;
      for (int e = 0; e < E; ++e) sum += std::exp(scores[e] - m);
      for (int e = 0; e < E; ++e) scores[e] = std::exp(scores[e] - m) / sum;
    }
    std::vector<int> sel;
    std::vector<double> wsel;
    std::vector<char> taken(E, 0);
    for (int r = 0; r < K; ++r) {
      int best = -1;
      double bv = -INFINITY;
      for (int e = 0; e < E; ++e)
        if (!taken[e] && biased[e] > bv) {
          bv = biased[e];
          best = e;
        }
      taken[best] = 1;
      sel.push_back(best);
      wsel.push_back(scores[best]);
    }
    // ascending id (insertion sort, mirroring the kernel)
    for (size_t i = 1; i < sel.size(); ++i) {
      const int id = sel[i];
      const double w = wsel[i];
      int j = static_cast<int>(i) - 1;
      while (j >= 0 && sel[j] > id) {
        sel[j + 1] = sel[j];
        wsel[j + 1] = wsel[j];
        --j;
      }
      sel[j + 1] = id;
      wsel[j + 1] = w;
    }
    double denom = 0.0;
    for (int i = 0; i < K; ++i) denom += wsel[i];
    denom += 1e-20;
    for (int i = 0; i < K; ++i) {
      out.ids[static_cast<size_t>(t) * K + i] = sel[i];
      double w =
          cfg.norm_topk_prob ? (wsel[i] / denom) * cfg.routed_scaling_factor
                             : wsel[i] * cfg.routed_scaling_factor;
      // The softmax mode's weights are cast to bf16 by the reference.
      if (softmax) w = bf16_to_d(d_to_bf16(w));
      out.weights[static_cast<size_t>(t) * K + i] = static_cast<float>(w);
    }
  }
}

void glm_moe_ref_forward(const uint16_t* hidden, const GlmMoeHostWeights& w,
                         const GlmMoeConfig& cfg, int tokens,
                         std::vector<uint16_t>& out) {
  GlmMoeConfig::validate_config(cfg);
  const int E = cfg.n_experts, H = cfg.hidden, I = cfg.inter, K = cfg.top_k;

  GlmMoeRouterRef route;
  glm_moe_ref_router(hidden, w.router_gate.data(), w.router_bias.data(), cfg,
                     tokens, route);

  out.assign(static_cast<size_t>(tokens) * H, 0);

  // Pre-dequantize every expert's three matrices (bf16-rounded weights
  // for FP8; exact fp4 x scale for NVFP4, with the global scale returned
  // as the dot's divisor) — small geometries only; the real-checkpoint
  // path is the torch backend.
  struct Mat {
    std::vector<double> w;
    double divisor;
  };
  auto weights_for = [&](int index) -> Mat {
    if (w.exl3_set()) {
      if (index < 0 || static_cast<size_t>(index) >= w.exl3.size())
        throw std::invalid_argument("glm_moe_ref_forward: EXL3 matrix index out of range");
      return Mat{exl3_reconstruct(w.exl3[static_cast<size_t>(index)]), 1.0};
    }
    if (w.fp4_of(index, E)) {
      const GlmFp4MatrixHost v = glm_moe_host_view_fp4(w, cfg, index);
      return Mat{dequant_fp4_weights(v), static_cast<double>(v.global_scale)};
    }
    if (w.packq && (index < E * 3 || w.shared_packq))
      return Mat{dequant_packq_weights(glm_moe_host_view_packq(w, cfg, index)), 1.0};
    return Mat{dequant_bf16_weights(glm_moe_host_view(w, cfg, index)), 1.0};
  };

  // Per token: the fp32-chain semantics in double — the down projection's
  // dots stay unrounded, each expert's contribution is one fma (weight x
  // dot + acc) in ASCENDING expert order, the shared expert last with
  // weight 1, and the sum rounds to bf16 exactly once. (The transformers
  // reference rounds per expert add; the engine deliberately does not —
  // see glm_moe_layer.hpp.)
  // The routed experts' inputs under the H32 rotation (the NF4I8
  // checkpoint): the expert input before gate / up, the SwiGLU output
  // before down — rotated rows in the engine's exact arithmetic (one
  // bf16 rounding); the shared expert reads the plain rows.
  const bool a8 = w.routed_input_transform == kMoeInputHadamard32Int8;
  const bool rotate = w.routed_input_transform == kMoeInputHadamard32 || a8;
  if (rotate && (H % 32 != 0 || I % 32 != 0))
    throw std::invalid_argument("glm_moe_ref_forward: the H32 rotation needs hidden and inter multiples of 32");
  if (a8 && (H % 128 != 0 || I % 128 != 0))
    throw std::invalid_argument("glm_moe_ref_forward: the int8 activation codes need hidden and inter multiples of 128");
  std::vector<uint16_t> gate_out, up_out, act;
  std::vector<double> down_out, acc(static_cast<size_t>(H));
  for (int t = 0; t < tokens; ++t) {
    const uint16_t* x = hidden + static_cast<size_t>(t) * H;
    std::vector<uint16_t> xrow(x, x + H);
    const std::vector<uint16_t> xrot = rotate ? rotate_h32(xrow) : xrow;
    // The Mixed346 codes of the rotated row, as the values they represent.
    const std::vector<double> xq = a8 ? a8_represented(xrot) : std::vector<double>{};
    std::fill(acc.begin(), acc.end(), 0.0);
    for (int i = 0; i < K; ++i) {
      const int e = route.ids[static_cast<size_t>(t) * K + i];
      const double we =
          static_cast<double>(route.weights[static_cast<size_t>(t) * K + i]);
      const Mat wg = weights_for(e * 3 + 0);
      const Mat wu = weights_for(e * 3 + 1);
      const Mat wd = weights_for(e * 3 + 2);
      // A converted (format 3) expert of a Mixed346 layer multiplies the
      // codes; an existing one the plain rows; any other layer its rows.
      const bool conv = a8 && w.packq_fmt_of(e * 3, E) == kPackedScaleBf16G128Mixed346;
      if (conv) {
        strict_gemm_d(xq, wg.w, I, H, gate_out);
        strict_gemm_d(xq, wu.w, I, H, up_out);
        swiglu_oracle(gate_out, up_out, I, cfg.swiglu_limit, act);
        strict_gemm_raw_d(a8_represented(rotate_h32(act)), wd.w, H, I, down_out);
      } else {
        const std::vector<uint16_t>& xin = a8 ? xrow : xrot;
        strict_gemm(xin, H, wg.w, 1, I, H, gate_out, wg.divisor);
        strict_gemm(xin, H, wu.w, 1, I, H, up_out, wu.divisor);
        swiglu_oracle(gate_out, up_out, I, cfg.swiglu_limit, act);
        if (rotate && !a8) act = rotate_h32(act);
        strict_gemm_raw(act, I, wd.w, 1, H, I, down_out, wd.divisor);
      }
      for (int d = 0; d < H; ++d) acc[d] += we * down_out[d];
    }
    // Shared expert, weight 1, added last.
    {
      const Mat wg = weights_for(E * 3 + 0);
      const Mat wu = weights_for(E * 3 + 1);
      const Mat wd = weights_for(E * 3 + 2);
      strict_gemm(xrow, H, wg.w, 1, I, H, gate_out, wg.divisor);
      strict_gemm(xrow, H, wu.w, 1, I, H, up_out, wu.divisor);
      swiglu_oracle(gate_out, up_out, I, cfg.swiglu_limit, act);
      strict_gemm_raw(act, I, wd.w, 1, H, I, down_out, wd.divisor);
      for (int d = 0; d < H; ++d) acc[d] += down_out[d];
    }
    for (int d = 0; d < H; ++d)
      out[static_cast<size_t>(t) * H + d] = d_to_bf16(acc[d]);
  }
}

}  // namespace dgpp
