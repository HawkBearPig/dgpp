#pragma once
// The Dory1.0-4B host-reference forward (docs/adding_a_model.md §6): a CPU
// walk over the fixture directory's RAW safetensors, transcribed from
// docs/dory_plan.md §1.3-§1.6, not from the CUDA kernels. The GEMVs are
// fp64 loops over the checkpoint's NVFP4 triples (the binding.hpp format
// contract: dequant = e2m1 x e4m3 x weight_scale_2), the justnorm / slerp
// sites call the host primitives of models/dory/reference.hpp that
// dory_norm_test pins, and the bf16 roundings follow the plan's dtypes:
// every Linear's output, the roped q/k, the attention output and the
// residual states are bf16, the interior arithmetic fp32/fp64.
//
// What this checks that the kernel tests cannot: the WALK's composition —
// layer order, the loop's re-injection and state, the per-loop caches, the
// sliding window, the scale sites, the head's logit_scale (a host reference
// that encoded the same wiring bug as the kernel would pass forever, so
// this one reads only the checkpoint and the plan).
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "loaders/nvfp4_quant.hpp"
#include "loaders/safetensors.hpp"
#include "models/dory/binding.hpp"
#include "models/dory/config.hpp"
#include "models/dory/reference.hpp"

namespace doryfx {

namespace fs = std::filesystem;
using namespace dgpp;

// One raw tensor: the mapped bytes of a fixture shard, by checkpoint name.
class HostCheckpoint {
 public:
  explicit HostCheckpoint(const std::string& dir) {
    for (const auto& e : fs::directory_iterator(dir)) {
      if (e.path().extension() != ".safetensors") continue;
      auto f = SafetensorsFile::open(e.path().string());
      files_.emplace_back(std::move(f));
    }
    if (files_.empty()) throw std::runtime_error("dory host reference: no shards in " + dir);
    for (const auto& f : files_)
      f->for_each([&](const TensorInfo& t) { by_name_[t.name] = &t; });
  }

  const TensorInfo& at(const std::string& name) const {
    const auto it = by_name_.find(name);
    if (it == by_name_.end()) throw std::runtime_error("dory host reference: missing " + name);
    return *it->second;
  }

  // The dequantized fp32 matrix of an NVFP4 projection (N, K), row-major.
  std::vector<float> dequant(const std::string& base, int64_t n, int64_t k) const {
    const TensorInfo& w = at(base + ".weight");
    const TensorInfo& s = at(base + ".weight_scale");
    const TensorInfo& g = at(base + ".weight_scale_2");
    float ws2 = 0.0f;
    std::memcpy(&ws2, g.data, 4);
    std::vector<float> out(static_cast<size_t>(n) * k);
    for (int64_t r = 0; r < n; ++r)
      for (int64_t c = 0; c < k; ++c) {
        // binding.hpp's contract, nvfp4_quant.hpp's code/scale order:
        // value = e2m1(code) x e4m3(s_b) x weight_scale_2.
        out[static_cast<size_t>(r) * k + c] =
            nvfp4_decode(static_cast<const uint8_t*>(w.data),
                         static_cast<const uint8_t*>(s.data), 1.0f / ws2, k, r, c);
      }
    return out;
  }

  std::vector<uint16_t> bf16(const std::string& name) const {
    const TensorInfo& t = at(name);
    const size_t n = t.numel();
    const auto* p = static_cast<const uint16_t*>(t.data);
    return std::vector<uint16_t>(p, p + n);
  }

 private:
  std::vector<std::unique_ptr<SafetensorsFile>> files_;
  std::map<std::string, const TensorInfo*> by_name_;
};

// The tiny-fixture walk: cold prefill only (the paged engine's decode rows
// are checked against this walk's fresh-prefill equivalence by the test).
// Returns the logits [T, vocab] as fp32.
inline std::vector<float> host_forward(const DoryConfig& cfg, const std::string& dir,
                                       const std::vector<int64_t>& ids, bool last_loop_policy) {
  const HostCheckpoint ckpt(dir);
  const int H = cfg.hidden_size;
  const int D = cfg.head_dim;
  const int NH = cfg.num_attention_heads, NKV = cfg.num_key_value_heads;
  const int Q = NH * D, KV = NKV * D, I = cfg.intermediate_size;
  const int T = static_cast<int>(ids.size());
  const int loops = cfg.n_recurrent_loops;
  const float eps = kDoryJustnormEps;

  // Dequantized projections, every attention layer once (weights shared
  // across the loops; the fixture is tiny). Cache K/V rows bf16, indexed
  // [layer][loop][pos][head*D + d] (the per_loop policy's planes; the
  // last_loop policy shares one plane and lets every loop overwrite it —
  // identical for one cold walk, and a plane bug shows up as a mismatch).
  struct AttnW {
    std::vector<float> q, k, v, o, g;
    std::vector<uint16_t> sqk, s_gate, alpha;
    std::vector<uint16_t> kc[2], vc[2];  // [pos * KV + head*D + d] per loop (loops <= 2)
    std::vector<uint16_t> kc_last, vc_last;
  };
  std::vector<AttnW> attn(cfg.num_hidden_layers);
  struct MlpW {
    std::vector<float> gate, up, down;
    std::vector<uint16_t> suv_gate, suv_up, alpha;
  };
  std::vector<MlpW> mlp(cfg.num_hidden_layers);
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    const std::string p = dory_layer_prefix(cfg, l) + "mixer.";
    if (cfg.is_attention(l)) {
      AttnW& w = attn[static_cast<size_t>(l)];
      w.q = ckpt.dequant(p + "q_proj", Q, H);
      w.k = ckpt.dequant(p + "k_proj", KV, H);
      w.v = ckpt.dequant(p + "v_proj", KV, H);
      w.o = ckpt.dequant(p + "o_proj", H, Q);
      w.g = ckpt.dequant(p + "g_proj", Q, H);
      w.sqk = ckpt.bf16(p + "sqk");
      w.s_gate = ckpt.bf16(p + "s_gate");
      w.alpha = ckpt.bf16(dory_layer_prefix(cfg, l) + "attn_alpha");
      const int planes = cfg.kv_repeats(l, false);
      for (int i = 0; i < planes && i < 2; ++i) {
        w.kc[static_cast<size_t>(i)].assign(static_cast<size_t>(T) * KV, 0);
        w.vc[static_cast<size_t>(i)].assign(static_cast<size_t>(T) * KV, 0);
      }
      w.kc_last.assign(static_cast<size_t>(T) * KV, 0);
      w.vc_last.assign(static_cast<size_t>(T) * KV, 0);
    } else {
      MlpW& w = mlp[static_cast<size_t>(l)];
      w.gate = ckpt.dequant(p + "gate_proj", I, H);
      w.up = ckpt.dequant(p + "up_proj", I, H);
      w.down = ckpt.dequant(p + "down_proj", H, I);
      w.suv_gate = ckpt.bf16(p + "suv_gate");
      w.suv_up = ckpt.bf16(p + "suv_up");
      w.alpha = ckpt.bf16(dory_layer_prefix(cfg, l) + "mlp_alpha");
    }
  }
  std::vector<uint16_t> embed = ckpt.bf16("backbone.embeddings.weight");
  std::vector<uint16_t> head = ckpt.bf16("lm_head.weight");
  std::vector<uint16_t> logit_scale = ckpt.bf16("lm_head.logit_scale");

  // A bf16 row GEMV: bf16( fp64 sum f32(W[n,k]) * f32(x[k]) ).
  auto gemv = [&](const std::vector<float>& w, int64_t n, int64_t k, const uint16_t* x,
                  uint16_t* y) {
    for (int64_t r = 0; r < n; ++r) {
      double acc = 0.0;
      const float* row = w.data() + r * k;
      for (int64_t c = 0; c < k; ++c) acc += static_cast<double>(row[c]) * bf16_bits_to_float(x[c]);
      y[r] = float_to_bf16_bits(static_cast<float>(acc));
    }
  };

  // One sublayer application on every token row (h in, h out, both [T, H]).
  std::vector<uint16_t> hb(static_cast<size_t>(T) * H), xb(static_cast<size_t>(T) * H);
  auto apply = [&](int l, int loop, const uint16_t* h, uint16_t* out) {
    if (cfg.is_attention(l)) {
      AttnW& w = attn[static_cast<size_t>(l)];
      const int plane = last_loop_policy ? -1 : loop;
      std::vector<uint16_t>& kcache = plane < 0 ? w.kc_last : w.kc[static_cast<size_t>(plane)];
      std::vector<uint16_t>& vcache = plane < 0 ? w.vc_last : w.vc[static_cast<size_t>(plane)];
      std::vector<float> invf(static_cast<size_t>(D) / 2);
      const double theta = cfg.rope_theta_of(l);
      for (int i = 0; i < D / 2; ++i)
        invf[static_cast<size_t>(i)] =
            static_cast<float>(std::pow(theta, -2.0 * static_cast<double>(i) / D));
      std::vector<uint16_t> q(static_cast<size_t>(T) * Q), kk(static_cast<size_t>(T) * KV),
          vv(static_cast<size_t>(T) * KV), gate(static_cast<size_t>(T) * Q),
          ao(static_cast<size_t>(T) * Q);
      for (int t = 0; t < T; ++t) {
        const uint16_t* hr = h + static_cast<size_t>(t) * H;
        gemv(w.q, Q, H, hr, q.data() + static_cast<size_t>(t) * Q);
        gemv(w.k, KV, H, hr, kk.data() + static_cast<size_t>(t) * KV);
        gemv(w.v, KV, H, hr, vv.data() + static_cast<size_t>(t) * KV);
        gemv(w.g, Q, H, hr, gate.data() + static_cast<size_t>(t) * Q);
        // RoPE, then the head justnorm, then the sqk square on q (plan §1.3).
        for (int hh = 0; hh < NH + NKV; ++hh) {
          const bool is_q = hh < NH;
          uint16_t* row = is_q ? q.data() + static_cast<size_t>(t) * Q + hh * D
                               : kk.data() + static_cast<size_t>(t) * KV + (hh - NH) * D;
          for (int i = 0; i < D / 2; ++i) {
            const float x1 = bf16_bits_to_float(row[i]), x2 = bf16_bits_to_float(row[i + D / 2]);
            const float ang = static_cast<float>(t) * invf[static_cast<size_t>(i)];
            const float c = std::cos(ang), s = std::sin(ang);
            row[i] = float_to_bf16_bits(x1 * c - x2 * s);
            row[i + D / 2] = float_to_bf16_bits(x2 * c + x1 * s);
          }
          dory_ref::justnorm_row(row, row, D, eps);
          if (is_q)
            for (int i = 0; i < D; ++i) {
              const float m = bf16_bits_to_float(w.sqk[static_cast<size_t>(hh) * D + i]) * cfg.sqk_factor();
              row[i] = float_to_bf16_bits(bf16_bits_to_float(row[i]) * (m * m));
            }
        }
        // Write the K/V plane for this position (the engine appends as it walks).
        for (int hkv = 0; hkv < NKV; ++hkv) {
          for (int i = 0; i < D; ++i) {
            kcache[static_cast<size_t>(t) * KV + hkv * D + i] =
                kk[static_cast<size_t>(t) * KV + hkv * D + i];
            vcache[static_cast<size_t>(t) * KV + hkv * D + i] =
                vv[static_cast<size_t>(t) * KV + hkv * D + i];
          }
        }
      }
      // The causal (windowed) GQA attention, fp64 scores and probabilities.
      const double softmax_scale = 1.0 / std::sqrt(static_cast<double>(D));
      const int window = cfg.is_sliding(l) && cfg.sliding_window > 0 ? cfg.sliding_window : T;
      for (int t = 0; t < T; ++t) {
        for (int hh = 0; hh < NH; ++hh) {
          const int hkv = hh * NKV / NH;
          const uint16_t* qr = q.data() + static_cast<size_t>(t) * Q + hh * D;
          std::vector<double> score(static_cast<size_t>(t) + 1);
          double mx = -1e300;
          for (int p = 0; p <= t; ++p) {
            if (t - p >= window) {
              score[static_cast<size_t>(p)] = -1e300;
              continue;
            }
            const uint16_t* kr = kcache.data() + static_cast<size_t>(p) * KV + hkv * D;
            double acc = 0.0;
            for (int i = 0; i < D; ++i)
              acc += static_cast<double>(bf16_bits_to_float(qr[i])) * bf16_bits_to_float(kr[i]);
            score[static_cast<size_t>(p)] = acc * softmax_scale;
            mx = std::max(mx, score[static_cast<size_t>(p)]);
          }
          double sum = 0.0;
          std::vector<double> prob(score.size());
          for (size_t p = 0; p < prob.size(); ++p) {
            prob[p] = score[p] < -1e200 ? 0.0 : std::exp(score[p] - mx);
            sum += prob[p];
          }
          std::vector<double> oacc(static_cast<size_t>(D), 0.0);
          for (int p = 0; p <= t; ++p) {
            const double w_p = prob[static_cast<size_t>(p)] / sum;
            if (w_p == 0.0) continue;
            const uint16_t* vr = vcache.data() + static_cast<size_t>(p) * KV + hkv * D;
            for (int i = 0; i < D; ++i) oacc[static_cast<size_t>(i)] += w_p * bf16_bits_to_float(vr[i]);
          }
          // The output gate (plan §1.3): fp32(attn) x sigmoid(gate x s_gate^2).
          for (int i = 0; i < D; ++i) {
            const float a = static_cast<float>(oacc[static_cast<size_t>(i)]);
            const float gs = bf16_bits_to_float(w.s_gate[static_cast<size_t>(hh) * D + i]) * cfg.s_gate_factor();
            const float g = bf16_bits_to_float(gate[static_cast<size_t>(t) * Q + hh * D + i]);
            ao[static_cast<size_t>(t) * Q + hh * D + i] =
                float_to_bf16_bits(a / (1.0f + std::exp(-g * gs * gs)));
          }
        }
      }
      // The attention-output justnorm spans ALL query heads (plan §6), then
      // o_proj and the slerp residual.
      dory_ref::justnorm(ao.data(), ao.data(), T, Q, eps);
      for (int t = 0; t < T; ++t) {
        gemv(w.o, H, Q, ao.data() + static_cast<size_t>(t) * Q, xb.data() + static_cast<size_t>(t) * H);
        dory_ref::slerp_alpha(h + static_cast<size_t>(t) * H, xb.data() + static_cast<size_t>(t) * H,
                              w.alpha.data(), cfg.alpha_scale(), out + static_cast<size_t>(t) * H, 1, H,
                              eps);
      }
      return;
    }
    // The MLP block (plan §1.4).
    MlpW& w = mlp[static_cast<size_t>(l)];
    std::vector<uint16_t> g(static_cast<size_t>(T) * I), u(static_cast<size_t>(T) * I),
        m(static_cast<size_t>(T) * I), o(static_cast<size_t>(T) * H);
    for (int t = 0; t < T; ++t) {
      const uint16_t* hr = h + static_cast<size_t>(t) * H;
      gemv(w.gate, I, H, hr, g.data() + static_cast<size_t>(t) * I);
      gemv(w.up, I, H, hr, u.data() + static_cast<size_t>(t) * I);
      for (int i = 0; i < I; ++i) {
        const double gs = bf16_bits_to_float(g[static_cast<size_t>(t) * I + i]) *
                          bf16_bits_to_float(w.suv_gate[static_cast<size_t>(i)]) * cfg.suv_factor();
        const double us = bf16_bits_to_float(u[static_cast<size_t>(t) * I + i]) *
                          bf16_bits_to_float(w.suv_up[static_cast<size_t>(i)]) * cfg.suv_factor();
        m[static_cast<size_t>(t) * I + i] =
            float_to_bf16_bits(static_cast<float>(gs / (1.0 + std::exp(-gs)) * us));
      }
    }
    dory_ref::justnorm(m.data(), m.data(), T, I, eps);
    for (int t = 0; t < T; ++t) {
      gemv(w.down, H, I, m.data() + static_cast<size_t>(t) * I, o.data() + static_cast<size_t>(t) * H);
      dory_ref::slerp_alpha(h + static_cast<size_t>(t) * H, o.data() + static_cast<size_t>(t) * H,
                            w.alpha.data(), cfg.alpha_scale(), out + static_cast<size_t>(t) * H, 1, H, eps);
    }
  };

  // The looped backbone (plan §1.5).
  hb.assign(static_cast<size_t>(T) * H, 0);
  for (int t = 0; t < T; ++t)
    for (int i = 0; i < H; ++i)
      hb[static_cast<size_t>(t) * H + i] = embed[static_cast<size_t>(ids[static_cast<size_t>(t)]) * H + i];
  for (int l = 0; l < cfg.n_input_layers; ++l)
    apply(l, 0, hb.data(), xb.data()), hb.swap(xb);
  std::vector<uint16_t> input_h = hb, state = hb, yb(static_cast<size_t>(T) * H);
  for (int i = 0; i < loops; ++i) {
    dory_ref::add_justnorm(state.data(), input_h.data(), hb.data(), T, H, eps);
    for (int l = cfg.n_input_layers; l < cfg.n_input_layers + cfg.n_recurrent_layers; ++l)
      apply(l, i, hb.data(), xb.data()), hb.swap(xb);
    dory_ref::justnorm(hb.data(), hb.data(), T, H, eps);
    state = hb;
  }
  hb = state;
  for (int l = cfg.n_input_layers + cfg.n_recurrent_layers; l < cfg.num_hidden_layers; ++l)
    apply(l, 0, hb.data(), xb.data()), hb.swap(xb);

  // The head (plan §1.6): logits = f32(h) @ f32(head)^T x (f32(logit_scale) * logit_factor).
  std::vector<float> logits(static_cast<size_t>(T) * cfg.vocab_size);
  for (int t = 0; t < T; ++t)
    for (int64_t v = 0; v < cfg.vocab_size; ++v) {
      double acc = 0.0;
      const uint16_t* hr = head.data() + static_cast<size_t>(v) * H;
      for (int i = 0; i < H; ++i)
        acc += static_cast<double>(bf16_bits_to_float(hr[i])) * bf16_bits_to_float(hb[static_cast<size_t>(t) * H + i]);
      logits[static_cast<size_t>(t) * cfg.vocab_size + static_cast<size_t>(v)] =
          static_cast<float>(acc) * (bf16_bits_to_float(logit_scale[static_cast<size_t>(v)]) * cfg.logit_factor());
    }
  return logits;
}

}  // namespace doryfx
