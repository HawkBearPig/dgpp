#include "models/qwen/layers.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "kernels/bf16_gemv.hpp"
#include "kernels/mma_gemv.hpp"
#include "kernels/dsa.hpp"
#include "kernels/kda.hpp"
#include "kernels/qsa.hpp"
#include "kernels/rope_scaling.hpp"
#include "kernels/qwen_gr.hpp"
#include "kernels/fp8_dequant.hpp"
#include "kernels/scale_gemm.hpp"
#include "kernels/qwen_norm.hpp"
#include "kernels/qwen_ple.hpp"

namespace dgpp {
namespace {

template <class T>
T* dev_alloc(size_t n) {
  T* p = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  return p;
}

void gemm_bf16(const QwenGemmWorkspace& g, const uint16_t* act, int64_t act_stride,
               const uint16_t* w, void* out, GemmOut out_type, int m, int n, int k,
               cudaStream_t stream) {
  g.gemm->matmul(act, w, out, m, n, k, DType::BF16, out_type, static_cast<size_t>(act_stride),
                 g.ws, g.ws_bytes, stream);
}

// A dense projection in the checkpoint's BF16 (the GEMM interface) or, under
// engine.dense_weights = "fp8", the block-FP8 form through the scale GEMM
// (its chunked fp8 GEMV at decode rows, the tile kernel above them —
// 2026-09-10). One of w / w8.payload is set.
void gemm_dense(const QwenGemmWorkspace& g, const uint16_t* act, int64_t act_stride,
                const uint16_t* w, const GlmQuantMatrix& w8, void* out, GemmOut out_type, int m,
                int n, int k, cudaStream_t stream) {
  if (w8.payload) {
    if (w8.rows != n || w8.cols != k)
      throw std::invalid_argument("qwen dense fp8: the matrix's shape disagrees with the product");
    // Prefill-shaped (above the scale GEMM's GEMV lowering): through the
    // BF16 interface on the dequantized matrix when the bridge holds it.
    const size_t bf16_bytes = static_cast<size_t>(n) * static_cast<size_t>(k) * 2;
    if (m > 128 && g.dequant && bf16_bytes <= g.dequant_bytes) {
      launch_fp8_dequant_blocks(w8.payload, w8.scales, g.dequant, n, k, stream);
      gemm_bf16(g, act, act_stride, g.dequant, out, out_type, m, n, k, stream);
      return;
    }
    if (out_type == GemmOut::F32)
      launch_scale_gemm_f32(act, static_cast<size_t>(act_stride), w8.payload, w8.scales,
                            static_cast<float*>(out), m, n, k, stream, static_cast<size_t>(n), g.mma_from_rows);
    else
      launch_scale_gemm_bf16(act, static_cast<size_t>(act_stride), w8.payload, w8.scales,
                             static_cast<uint16_t*>(out), m, n, k, stream, static_cast<size_t>(n), g.mma_from_rows);
    return;
  }
  if (!w) throw std::invalid_argument("qwen dense: null weight");
  gemm_bf16(g, act, act_stride, w, out, out_type, m, n, k, stream);
}

}  // namespace

void qwen_mtp_hidden_projection(const QwenGemmWorkspace& g, const uint16_t* act,
    const uint16_t* weight, uint16_t* out, int tokens, int hc, int hidden,
    bool decode, cudaStream_t stream) {
  const int rows = tokens * hc;
  // Real H=2560, hc=4 at 12/16 verify rows: Lt's 48/64-row algorithm
  // records a memset node, which can deadlock collective graph replay.
  // Only the newly widened decode takes MMA; preserve prefill and all
  // existing <=8-row walks. This draft projection uses the same MMA row
  // chain at every wide shape, including the diagnostic lowering mode.
  if (decode && tokens > 8 &&
      mma_gemv_shape_ok(weight, act, static_cast<size_t>(hidden), rows, hidden)) {
    launch_mma_gemv_bf16_bf16(act, static_cast<size_t>(hidden), weight, out,
                              rows, hidden, hidden, static_cast<size_t>(hidden), stream);
  } else {
    gemm_bf16(g, act, hidden, weight, out, GemmOut::BF16, rows, hidden, hidden, stream);
  }
}

// ---- QwenGrSite -------------------------------------------------------------------

QwenGrSite::QwenGrSite(const QwenGrResident& w, const QwenGemmWorkspace& gemm, int hc, int hidden,
                       int lowrank, int max_tokens, float eps)
    : w_(w), g_(gemm), hc_(hc), hidden_(hidden), lowrank_(lowrank), max_tokens_(max_tokens),
      eps_(eps) {
  if (!g_.gemm || !g_.ws) throw std::invalid_argument("QwenGrSite: GEMM workspace required");
  if (max_tokens_ <= 0) throw std::invalid_argument("QwenGrSite: max_tokens must be positive");
  const size_t M = static_cast<size_t>(max_tokens_), W = static_cast<size_t>(hc_) * hidden_;
  rn_ = dev_alloc<uint16_t>(M * W);
  t_ = dev_alloc<uint16_t>(M * static_cast<size_t>(lowrank_));
  logits_ = dev_alloc<uint16_t>(M * W);
  gates_ = dev_alloc<float>(M * static_cast<size_t>(hc_));
  // The fused decode GEMVs need 16-byte-aligned weights (the resident
  // image's) and the smem budget; DGPP_QWEN_GR_FUSED=off keeps the chain.
  const char* knob = std::getenv("DGPP_QWEN_GR_FUSED");
  const std::string fused_knob = knob ? knob : "";
  const void* down_ptr = w_.down ? static_cast<const void*>(w_.down) : static_cast<const void*>(w_.down_fp8.payload);
  const void* up_ptr = w_.up ? static_cast<const void*>(w_.up) : static_cast<const void*>(w_.up_fp8.payload);
  fused_mix_ = qwen_gr_fused_mix_accepts(hc_, hidden_, lowrank_) &&
               (reinterpret_cast<uintptr_t>(down_ptr) % 16 == 0) &&
               (reinterpret_cast<uintptr_t>(up_ptr) % 16 == 0) && fused_knob != "off";
  // The scalar row only. The fused kernel stages kRows normalized rows in
  // shared memory (20 KB each), so at two rows and beyond it runs one block
  // per SM and loses to the norm + GEMV chain: measured 2026-09-10 with the
  // group norms of a row issued together (bitwise, and still a loss — MTP
  // 26.4 vs 25.6 ms/step, four live requests 101.9 vs 119.7 tok/s), which
  // closes the "make the fused mix multi-row" idea. DGPP_QWEN_GR_FUSED=all
  // re-enables it for another look.
  fused_rows_max_ = fused_knob == "all" ? 8 : 1;
  // The inject dots' side stream: lowest priority, so the chain's kernels
  // keep first claim on the SMs (the dots are one block per row and have a
  // whole branch plus a collective to hide under). Capture-safe: the fork
  // and the join are events recorded on the streams the capture owns.
  // DGPP_QWEN_GR_GATE_SIDE: off = the dots in the chain after the fold
  // (the 2026-09-09 form); side = the batched rows keep the side-stream
  // kernel (the 2026-09-10 morning form); default = the dots ride the down
  // GEMV at every decode row count.
  const char* side = std::getenv("DGPP_QWEN_GR_GATE_SIDE");
  gate_early_ = !(side && std::string(side) == "off");
  gate_side_only_ = side && std::string(side) == "side";
  if (gate_early_) {
    int least = 0, greatest = 0;
    DGPP_CUDA_OK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    DGPP_CUDA_OK(cudaStreamCreateWithPriority(&gate_side_, cudaStreamNonBlocking, least));
    DGPP_CUDA_OK(cudaEventCreateWithFlags(&gate_fork_, cudaEventDisableTiming));
    DGPP_CUDA_OK(cudaEventCreateWithFlags(&gate_join_, cudaEventDisableTiming));
  }
}

QwenGrSite::~QwenGrSite() {
  if (gate_join_) cudaEventDestroy(gate_join_);
  if (gate_fork_) cudaEventDestroy(gate_fork_);
  if (gate_side_) cudaStreamDestroy(gate_side_);
  cudaFree(rn_);
  cudaFree(t_);
  cudaFree(logits_);
  cudaFree(gates_);
}

void QwenGrSite::mix(const uint16_t* r, uint16_t* x, int tokens, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (tokens > max_tokens_) throw std::invalid_argument("QwenGrSite: tokens exceed max_tokens");
  // Either form of the two projections (the checkpoint's BF16, or block FP8).
  if (!w_.hc_norm || !(w_.down || w_.down_fp8.payload) || !(w_.up || w_.up_fp8.payload))
    throw std::invalid_argument("QwenGrSite: null weights");
  const int W = hc_ * hidden_;
  // The fused decode forms in the weights' form: BF16, or block FP8 (the
  // fp8 twins, 2026-09-10 — bitwise the unfused fp8 chain).
  const bool dense_fp8 = w_.down_fp8.payload != nullptr;
  if (tokens <= fused_rows_max_ && fused_mix_) {
    // The scalar decode row: the norm and the activation folded into the
    // two GEMVs' staging (kernels/qwen_gr, bitwise the four-launch chain).
    // One row only: every down block recomputes the row's group norms in
    // sequence, which at two rows cost 1.5 ms per MTP pass against a 0.3
    // ms gain at one (the fabric A/B of 2026-09-09).
    // The inject dots ride the down GEMV's launch when this site combines:
    // its blocks already hold the normalized row (kernels/qwen_gr).
    if (dense_fp8)
      qwen_gr_norm_down_fp8(r, static_cast<size_t>(W), w_.hc_norm, hc_, hidden_, eps_, rn_,
                            w_.down_fp8.payload, w_.down_fp8.scales, t_, lowrank_, tokens, stream,
                            inject_fused() ? w_.inject : nullptr, inject_fused() ? gates_ : nullptr);
    else
      qwen_gr_norm_down_bf16(r, static_cast<size_t>(W), w_.hc_norm, hc_, hidden_, eps_, rn_, w_.down,
                             t_, lowrank_, tokens, stream,
                             inject_fused() ? w_.inject : nullptr,
                             inject_fused() ? gates_ : nullptr);
    if (inject_fused()) gates_ready_ = true;
    if (dense_fp8)
      qwen_gr_act_up_fp8(t_, lowrank_, hc_, w_.up_fp8.payload, w_.up_fp8.scales, logits_, hidden_, tokens, stream);
    else
      qwen_gr_act_up_bf16(t_, lowrank_, hc_, w_.up, logits_, hidden_, tokens, stream);
  } else {
    qwen_group_rmsnorm_bf16(r, w_.hc_norm, rn_, tokens, hc_, hidden_, eps_, stream);
    if (tokens <= 8 && inject_fused() && fused_mix_ && !gate_side_only_) {
      // The batched decode rows (MTP, the row batches): the down GEMV with
      // the inject rows appended — no side stream (kernels/qwen_gr).
      if (dense_fp8)
        qwen_gr_down_inject_fp8(rn_, w_.down_fp8.payload, w_.down_fp8.scales, t_, lowrank_, w_.inject, gates_,
                                hc_, hidden_, tokens, stream);
      else
        qwen_gr_down_inject_bf16(rn_, w_.down, t_, lowrank_, w_.inject, gates_, hc_, hidden_,
                                 tokens, stream);
      gates_ready_ = true;
    } else {
      fork_gate_dots(stream, tokens);
      gemm_dense(g_, rn_, W, w_.down, w_.down_fp8, t_, GemmOut::BF16, tokens, lowrank_, W, stream);
    }
    qwen_gr_gate_act_bf16(t_, tokens, lowrank_, hc_, stream);
    gemm_dense(g_, t_, lowrank_, w_.up, w_.up_fp8, logits_, GemmOut::BF16, tokens, W, lowrank_, stream);
  }
  qwen_gr_mix_finish_bf16(logits_, rn_, x, tokens, hc_, hidden_, stream);
}

// The dots on the side stream, from the Rn the mix has just written. Only
// a site that will combine (one with inject weights) forks: the model-level
// mixer mixes and never combines, so a fork there would never be joined.
void QwenGrSite::fork_gate_dots(cudaStream_t main, int tokens) {
  if (!gate_side_ || !w_.inject) return;
  DGPP_CUDA_OK(cudaEventRecord(gate_fork_, main));
  DGPP_CUDA_OK(cudaStreamWaitEvent(gate_side_, gate_fork_, 0));
  qwen_gr_combine_dots_bf16(rn_, w_.inject, gates_, tokens, hc_, hidden_, gate_side_);
  gate_forked_ = true;
}

void QwenGrSite::combine(uint16_t* r, const uint16_t* y, int tokens, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (!w_.inject) throw std::invalid_argument("QwenGrSite: combine on a site without inject weights");
  if (gates_ready_) {  // the mix's down GEMV wrote them
    gates_ready_ = false;
    qwen_gr_combine_apply_bf16(r, gates_, y, tokens, hc_, hidden_, stream);
    return;
  }
  if (gate_forked_) {
    DGPP_CUDA_OK(cudaEventRecord(gate_join_, gate_side_));
    DGPP_CUDA_OK(cudaStreamWaitEvent(stream, gate_join_, 0));
    gate_forked_ = false;
    qwen_gr_combine_apply_bf16(r, gates_, y, tokens, hc_, hidden_, stream);
    return;
  }
  qwen_gr_combine_bf16(r, rn_, w_.inject, y, gates_, tokens, hc_, hidden_, stream);
}

size_t QwenGrSite::scratch_bytes(int hc, int hidden, int lowrank, int max_tokens) {
  const size_t M = static_cast<size_t>(std::max(max_tokens, 0)), W = static_cast<size_t>(hc) * hidden;
  return M * W * 2 * 2 + M * static_cast<size_t>(lowrank) * 2 +
         M * static_cast<size_t>(hc) * 4;  // rn_, logits_, t_, gates_
}

// ---- QwenGdnLayer ----------------------------------------------------------------

QwenGdnLayer::QwenGdnLayer(const QwenGdnResident& w, const QwenGemmWorkspace& gemm,
                           const QwenTextConfig& cfg, int max_tokens)
    : w_(w), g_(gemm), hidden_(cfg.hidden_size), lk_(w.local_key_heads), lv_(w.local_value_heads),
      k_dim_(cfg.gdn_key_head_dim), v_dim_(cfg.gdn_value_head_dim), conv_width_(cfg.gdn_conv_width),
      max_tokens_(max_tokens), eps_(cfg.rms_norm_eps) {
  if (!g_.gemm || !g_.ws) throw std::invalid_argument("QwenGdnLayer: GEMM workspace required");
  if (lk_ <= 0 || lv_ <= 0 || lv_ % lk_ != 0)
    throw std::invalid_argument("QwenGdnLayer: value heads must be a multiple of key heads");
  if (k_dim_ != 128 || v_dim_ != 128)
    throw std::invalid_argument("QwenGdnLayer: 128-wide heads (the recurrence kernel's shape)");
  // The reference's scale: head_dim ** -0.5 in double, narrowed to fp32.
  scale_ = static_cast<float>(std::pow(static_cast<double>(k_dim_), -0.5));
  conv_channels_ = static_cast<int64_t>(lk_) * k_dim_ * 2 + static_cast<int64_t>(lv_) * v_dim_;
  const size_t M = static_cast<size_t>(max_tokens_);
  qkv_ = dev_alloc<uint16_t>(M * static_cast<size_t>(conv_channels_));
  qkvc_ = dev_alloc<uint16_t>(M * static_cast<size_t>(conv_channels_));
  z_ = dev_alloc<uint16_t>(M * static_cast<size_t>(lv_) * v_dim_);
  a_ = dev_alloc<uint16_t>(M * static_cast<size_t>(lv_));
  b_ = dev_alloc<uint16_t>(M * static_cast<size_t>(lv_));
  core_ = dev_alloc<uint16_t>(M * static_cast<size_t>(lv_) * v_dim_);
  normed_ = dev_alloc<uint16_t>(M * static_cast<size_t>(lv_) * v_dim_);
}

QwenGdnLayer::~QwenGdnLayer() {
  cudaFree(qkv_);
  cudaFree(qkvc_);
  cudaFree(z_);
  cudaFree(a_);
  cudaFree(b_);
  cudaFree(core_);
  cudaFree(normed_);
}

void QwenGdnLayer::rebind(const QwenGdnResident& w) {
  if (w.local_key_heads != lk_ || w.local_value_heads != lv_)
    throw std::invalid_argument("QwenGdnLayer: rebind changes the head geometry");
  w_ = w;
}

// The four input projections off the same rows: qkv [C, H], z [LV, H], a
// and b [lv, H]. The decode rows (<= 4, the GEMV's row cap) go as one
// multi-problem GEMV launch (2026-09-09: four launches and their gaps per
// GDN layer, the two 12-row projections 3.6 us each); every output is
// bitwise its own launch (bf16_gemv_test). More rows take the interface.
void QwenGdnLayer::in_projections(const uint16_t* x, int tokens, cudaStream_t stream) {
  const int H = hidden_;
  const int C = static_cast<int>(conv_channels_);
  const int LV = lv_ * v_dim_;
  if (w_.in_proj_qkv_fp8.payload) {
    // The FP8 form: qkv and z as one multi-problem fp8 GEMV at decode rows
    //, the scale GEMM above them; a and b (BF16, [lv, H]) as
    // one dual GEMV.
    if (tokens <= std::min(8, g_.gemv_rows)) {
      Fp8GemvProblem p[2];
      p[0].payload = w_.in_proj_qkv_fp8.payload; p[0].scales = w_.in_proj_qkv_fp8.scales; p[0].out = qkv_; p[0].n = C;
      p[1].payload = w_.in_proj_z_fp8.payload; p[1].scales = w_.in_proj_z_fp8.scales; p[1].out = z_; p[1].n = LV;
      launch_scale_gemv_multi_bf16(p, 2, x, static_cast<size_t>(H), tokens, H, stream);
    } else {
      gemm_dense(g_, x, H, nullptr, w_.in_proj_qkv_fp8, qkv_, GemmOut::BF16, tokens, C, H, stream);
      gemm_dense(g_, x, H, nullptr, w_.in_proj_z_fp8, z_, GemmOut::BF16, tokens, LV, H, stream);
    }
    if (tokens <= std::min(4, g_.gemv_rows) && bf16_gemv_accepts(w_.in_proj_a, tokens, H) &&
        bf16_gemv_accepts(w_.in_proj_b, tokens, H)) {
      Bf16GemvProblem p[2];
      p[0].act = x; p[0].act_row_stride = static_cast<size_t>(H); p[0].weight = w_.in_proj_a; p[0].out = a_; p[0].n = lv_;
      p[1].act = x; p[1].act_row_stride = static_cast<size_t>(H); p[1].weight = w_.in_proj_b; p[1].out = b_; p[1].n = lv_;
      launch_bf16_gemv_multi(p, 2, /*out_f32=*/false, tokens, H, stream);
    } else {
      gemm_bf16(g_, x, H, w_.in_proj_a, a_, GemmOut::BF16, tokens, lv_, H, stream);
      gemm_bf16(g_, x, H, w_.in_proj_b, b_, GemmOut::BF16, tokens, lv_, H, stream);
    }
    return;
  }
  if (tokens <= std::min(4, g_.gemv_rows) && bf16_gemv_accepts(w_.in_proj_qkv, tokens, H) &&
      bf16_gemv_accepts(w_.in_proj_z, tokens, H) && bf16_gemv_accepts(w_.in_proj_a, tokens, H) &&
      bf16_gemv_accepts(w_.in_proj_b, tokens, H)) {
    Bf16GemvProblem p[4];
    p[0].act = x; p[0].act_row_stride = static_cast<size_t>(H); p[0].weight = w_.in_proj_qkv; p[0].out = qkv_; p[0].n = C;
    p[1].act = x; p[1].act_row_stride = static_cast<size_t>(H); p[1].weight = w_.in_proj_z; p[1].out = z_; p[1].n = LV;
    p[2].act = x; p[2].act_row_stride = static_cast<size_t>(H); p[2].weight = w_.in_proj_a; p[2].out = a_; p[2].n = lv_;
    p[3].act = x; p[3].act_row_stride = static_cast<size_t>(H); p[3].weight = w_.in_proj_b; p[3].out = b_; p[3].n = lv_;
    launch_bf16_gemv_multi(p, 4, /*out_f32=*/false, tokens, H, stream);
    return;
  }
  gemm_bf16(g_, x, H, w_.in_proj_qkv, qkv_, GemmOut::BF16, tokens, C, H, stream);
  gemm_bf16(g_, x, H, w_.in_proj_z, z_, GemmOut::BF16, tokens, LV, H, stream);
  gemm_bf16(g_, x, H, w_.in_proj_a, a_, GemmOut::BF16, tokens, lv_, H, stream);
  gemm_bf16(g_, x, H, w_.in_proj_b, b_, GemmOut::BF16, tokens, lv_, H, stream);
}

void QwenGdnLayer::enqueue(const uint16_t* x, float* recurrent_state, uint16_t* conv_state,
                           uint16_t* out, int tokens, cudaStream_t stream,
                           const KdaStateSnapshots& rec_snap, const KdaConvSnapshots& conv_snap) {
  if (tokens <= 0) return;
  if (tokens > max_tokens_) throw std::invalid_argument("QwenGdnLayer: tokens exceed max_tokens");
  if (!(w_.in_proj_qkv || w_.in_proj_qkv_fp8.payload) || !w_.conv || !(w_.in_proj_z || w_.in_proj_z_fp8.payload) ||
      !w_.in_proj_a || !w_.in_proj_b || !w_.a_log || !w_.dt_bias || !w_.norm ||
      !(w_.out_proj || w_.out_proj_fp8.payload))
    throw std::invalid_argument("QwenGdnLayer: null weights");
  const int H = hidden_;
  const int C = static_cast<int>(conv_channels_);
  const int LV = lv_ * v_dim_;
  in_projections(x, tokens, stream);
  kda_causal_conv_silu_bf16(qkv_, C, w_.conv, conv_state, conv_width_ - 1, qkvc_, tokens, C,
                            conv_width_, stream, conv_snap);
  gdn_recurrent_fwd(qkvc_, a_, lv_, b_, lv_, w_.a_log, w_.dt_bias, recurrent_state, core_, tokens,
                    lv_, lv_ / lk_, k_dim_, v_dim_, scale_, stream, rec_snap);
  gdn_gated_rmsnorm_bf16(core_, z_, w_.norm, normed_, static_cast<int64_t>(tokens) * lv_, v_dim_,
                         eps_, stream);
  gemm_dense(g_, normed_, LV, w_.out_proj, w_.out_proj_fp8, out, GemmOut::BF16, tokens, H, LV, stream);
}

void QwenGdnLayer::enqueue_rows(const uint16_t* x, float* rec_states, int64_t rec_stride,
                                uint16_t* conv_states, int64_t conv_stride, uint16_t* out,
                                int rows, const KdaRequestRows& requests, cudaStream_t stream,
                                const KdaStateSnapshots& rec_snap,
                                const KdaConvSnapshots& conv_snap) {
  if (rows <= 0) return;
  if (rows > max_tokens_) throw std::invalid_argument("QwenGdnLayer: rows exceed max_tokens");
  if (!requests.request_ids || !requests.positions || !requests.spans || requests.num_requests <= 0)
    throw std::invalid_argument("QwenGdnLayer: incomplete row map");
  if (!(w_.in_proj_qkv || w_.in_proj_qkv_fp8.payload) || !w_.conv || !(w_.in_proj_z || w_.in_proj_z_fp8.payload) ||
      !w_.in_proj_a || !w_.in_proj_b || !w_.a_log || !w_.dt_bias || !w_.norm ||
      !(w_.out_proj || w_.out_proj_fp8.payload))
    throw std::invalid_argument("QwenGdnLayer: null weights");
  const int H = hidden_;
  const int C = static_cast<int>(conv_channels_);
  const int LV = lv_ * v_dim_;
  in_projections(x, rows, stream);
  kda_causal_conv_silu_bf16_batched(qkv_, C, w_.conv, conv_states, conv_stride, conv_width_ - 1,
                                    qkvc_, rows, C, conv_width_, requests, stream, conv_snap);
  gdn_recurrent_fwd_batched(qkvc_, a_, lv_, b_, lv_, w_.a_log, w_.dt_bias, rec_states, rec_stride,
                            core_, rows, lv_, lv_ / lk_, k_dim_, v_dim_, scale_, requests, stream,
                            rec_snap);
  gdn_gated_rmsnorm_bf16(core_, z_, w_.norm, normed_, static_cast<int64_t>(rows) * lv_, v_dim_,
                         eps_, stream);
  gemm_dense(g_, normed_, LV, w_.out_proj, w_.out_proj_fp8, out, GemmOut::BF16, rows, H, LV, stream);
}

size_t QwenGdnLayer::scratch_bytes(const QwenTextConfig& cfg, int local_key_heads, int local_value_heads,
                                   int max_tokens) {
  const size_t M = static_cast<size_t>(std::max(max_tokens, 0));
  const size_t lk = static_cast<size_t>(local_key_heads), lv = static_cast<size_t>(local_value_heads);
  const size_t K = static_cast<size_t>(cfg.gdn_key_head_dim), V = static_cast<size_t>(cfg.gdn_value_head_dim);
  const size_t C = 2 * lk * K + lv * V;
  return M * (2 * C + 3 * lv * V + 2 * lv) * 2;  // qkv, qkvc, z, core, normed, a, b
}

// ---- QwenQsaLayer ----------------------------------------------------------------

QwenQsaLayer::QwenQsaLayer(const QwenQsaResident& w, const QwenGemmWorkspace& gemm,
                           const QwenTextConfig& cfg, int max_tokens, int64_t max_pools)
    : w_(w), g_(gemm), hidden_(cfg.hidden_size), lh_(w.local_heads), lkv_(w.local_kv_heads),
      dim_(cfg.head_dim), rotary_(cfg.rotary_dim), idx_heads_(cfg.indexer_n_heads),
      idx_dim_(cfg.indexer_head_dim), kpool_(cfg.indexer_compress_ratio),
      select_k_(cfg.indexer_block_topk()),
      max_selected_(cfg.indexer_budget + cfg.indexer_compress_ratio - 1), max_tokens_(max_tokens),
      max_pools_(max_pools), eps_(cfg.rms_norm_eps) {
  if (!g_.gemm || !g_.ws) throw std::invalid_argument("QwenQsaLayer: GEMM workspace required");
  if (lh_ <= 0 || lkv_ <= 0 || lh_ % lkv_ != 0)
    throw std::invalid_argument("QwenQsaLayer: query heads must be a multiple of kv heads");
  if (dim_ != 256) throw std::invalid_argument("QwenQsaLayer: head_dim 256 (the attention kernel's shape)");
  if (idx_dim_ != 128 || idx_heads_ < 1 || idx_heads_ > 4 || cfg.indexer_kv_heads != 1)
    throw std::invalid_argument("QwenQsaLayer: indexer <= 4 heads x 128, one key head");
  if (max_pools_ <= 0) throw std::invalid_argument("QwenQsaLayer: max_pools must be positive");
  scale_ = static_cast<float>(std::pow(static_cast<double>(dim_), -0.5));
  std::vector<float> inv(static_cast<size_t>(rotary_ / 2));
  // The rope table: the plain one, or the YaRN ramp the engine knob asks
  // for (vLLM's YaRNScalingRotaryEmbedding._compute_inv_freq, verified
  // against the recipe's stack — see kernels/rope_scaling.hpp).
  if (cfg.rope_scaling.has_value()) {
    const RopeScaling& rs = *cfg.rope_scaling;
    rs.validate("rope_scaling");
    yarn_rope_inv_freq_host(rotary_, cfg.rope_theta, rs.correction_max_position(), rs.factor,
                            rs.beta_fast, rs.beta_slow, inv.data());
    mscale_ = rs.mscale();
  } else {
    qsa_rope_inv_freq(cfg.rope_theta, rotary_, inv.data());
  }
  d_inv_freq_ = dev_alloc<float>(inv.size());
  DGPP_CUDA_OK(cudaMemcpy(d_inv_freq_, inv.data(), inv.size() * 4, cudaMemcpyHostToDevice));
  const size_t M = static_cast<size_t>(max_tokens_);
  q_ = dev_alloc<uint16_t>(M * static_cast<size_t>(lh_) * 2 * dim_);
  k_ = dev_alloc<uint16_t>(M * static_cast<size_t>(lkv_) * dim_);
  v_ = dev_alloc<uint16_t>(M * static_cast<size_t>(lkv_) * dim_);
  qn_ = dev_alloc<uint16_t>(M * static_cast<size_t>(lh_) * dim_);
  kn_ = dev_alloc<uint16_t>(M * static_cast<size_t>(lkv_) * dim_);
  idx_ = dev_alloc<uint16_t>(M * static_cast<size_t>(idx_heads_ + 1) * idx_dim_);
  qi_ = dev_alloc<uint16_t>(M * static_cast<size_t>(idx_heads_) * idx_dim_);
  keys_ws_ = dev_alloc<uint64_t>(M * static_cast<size_t>(max_pools_));
  topk_ = dev_alloc<int32_t>(M * static_cast<size_t>(max_selected_));
  counts_ = dev_alloc<int32_t>(M);
  // Splits over the list: 256 tokens each, at most 8.
  n_split_ = std::max(1, std::min(8, (max_selected_ + 255) / 256));
  const size_t part = M * static_cast<size_t>(n_split_) * lh_;
  m_ws_ = dev_alloc<float>(part);
  l_ws_ = dev_alloc<float>(part);
  c_ws_ = dev_alloc<float>(part * dim_);
  c_out_ = dev_alloc<float>(M * static_cast<size_t>(lh_) * dim_);
  o_ = dev_alloc<uint16_t>(M * static_cast<size_t>(lh_) * dim_);
}

QwenQsaLayer::~QwenQsaLayer() {
  cudaFree(d_inv_freq_);
  cudaFree(q_);
  cudaFree(k_);
  cudaFree(v_);
  cudaFree(qn_);
  cudaFree(kn_);
  cudaFree(idx_);
  cudaFree(qi_);
  cudaFree(keys_ws_);
  cudaFree(topk_);
  cudaFree(counts_);
  cudaFree(m_ws_);
  cudaFree(l_ws_);
  cudaFree(c_ws_);
  cudaFree(c_out_);
  cudaFree(o_);
}

void QwenQsaLayer::rebind(const QwenQsaResident& w) {
  if (w.local_heads != lh_ || w.local_kv_heads != lkv_)
    throw std::invalid_argument("QwenQsaLayer: rebind changes the head geometry");
  w_ = w;
}

void QwenQsaLayer::enqueue(const uint16_t* x, int tokens, const QwenQsaRows& rows,
                           QwenQsaCache& cache, uint16_t* out, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (tokens > max_tokens_) throw std::invalid_argument("QwenQsaLayer: tokens exceed max_tokens");
  if (!rows.req_ids || !rows.pos) throw std::invalid_argument("QwenQsaLayer: null row metadata");
  if (rows.decode) {
    if (!rows.spans || rows.num_requests <= 0)
      throw std::invalid_argument("QwenQsaLayer: decode rows need request spans");
  } else {
    if (rows.pos0 < 0 || rows.pos0 % kpool_ != 0)
      throw std::invalid_argument("QwenQsaLayer: pos0 must be a non-negative multiple of kpool");
    if (rows.request < 0 || rows.request >= cache.max_requests)
      throw std::invalid_argument("QwenQsaLayer: request outside the cache");
    if (rows.pos0 + tokens > cache.slots())
      throw std::invalid_argument("QwenQsaLayer: the prefill overruns the cache");
    if ((rows.pos0 + tokens + kpool_ - 1) / kpool_ > max_pools_)
      throw std::invalid_argument("QwenQsaLayer: visible pools exceed the scoring workspace");
  }
  if (!(w_.q_proj || w_.q_proj_fp8.payload) || !(w_.k_proj || w_.k_proj_fp8.payload) ||
      !(w_.v_proj || w_.v_proj_fp8.payload) || !(w_.o_proj || w_.o_proj_fp8.payload) || !w_.q_norm || !w_.k_norm ||
      !(w_.index_qk_proj || w_.index_qk_proj_fp8.payload) || !w_.index_q_norm || !w_.index_k_norm)
    throw std::invalid_argument("QwenQsaLayer: null weights");
  const int H = hidden_, D = dim_, Di = idx_dim_, T = tokens;
  const int QW = lh_ * 2 * D, KW = lkv_ * D, IW = (idx_heads_ + 1) * Di;
  const int32_t* d_req = rows.req_ids;
  const int64_t* d_pos = rows.pos;

  // Projections.
  if (w_.q_proj_fp8.payload && T <= std::min(8, g_.gemv_rows)) {
    // The FP8 form at decode rows: the four projections as one
    // multi-problem fp8 GEMV.
    Fp8GemvProblem p[4];
    p[0].payload = w_.q_proj_fp8.payload; p[0].scales = w_.q_proj_fp8.scales; p[0].out = q_; p[0].n = QW;
    p[1].payload = w_.k_proj_fp8.payload; p[1].scales = w_.k_proj_fp8.scales; p[1].out = k_; p[1].n = KW;
    p[2].payload = w_.v_proj_fp8.payload; p[2].scales = w_.v_proj_fp8.scales; p[2].out = v_; p[2].n = KW;
    p[3].payload = w_.index_qk_proj_fp8.payload; p[3].scales = w_.index_qk_proj_fp8.scales; p[3].out = idx_; p[3].n = IW;
    launch_scale_gemv_multi_bf16(p, 4, x, static_cast<size_t>(H), T, H, stream);
  } else {
    gemm_dense(g_, x, H, w_.q_proj, w_.q_proj_fp8, q_, GemmOut::BF16, T, QW, H, stream);
    gemm_dense(g_, x, H, w_.k_proj, w_.k_proj_fp8, k_, GemmOut::BF16, T, KW, H, stream);
    gemm_dense(g_, x, H, w_.v_proj, w_.v_proj_fp8, v_, GemmOut::BF16, T, KW, H, stream);
    gemm_dense(g_, x, H, w_.index_qk_proj, w_.index_qk_proj_fp8, idx_, GemmOut::BF16, T, IW, H, stream);
  }
  // Norm + RoPE: q (the [q | gate] interleave), k, the indexer q.
  qsa_norm_rope_bf16(q_, QW, 2 * D, w_.q_norm, d_pos, d_inv_freq_, qn_, static_cast<int64_t>(lh_) * D,
                     T, lh_, D, rotary_, eps_, mscale_, stream);
  qsa_norm_rope_bf16(k_, KW, D, w_.k_norm, d_pos, d_inv_freq_, kn_, KW, T, lkv_, D, rotary_, eps_,
                     mscale_, stream);
  qsa_norm_rope_bf16(idx_, IW, Di, w_.index_q_norm, d_pos, d_inv_freq_, qi_,
                     static_cast<int64_t>(idx_heads_) * Di, T, idx_heads_, Di, rotary_, eps_,
                     mscale_, stream);
  // The caches: K/V rows, then the compressed keys and the ring.
  qsa_kv_append(kn_, KW, v_, KW, d_req, d_pos, T, cache.block_tables, cache.blocks_per_request,
                cache.block_tokens, lkv_, D, cache.k_cache, cache.v_cache, stream);
  const int pools_per_block = cache.block_tokens / kpool_;
  const uint16_t* raw_k = idx_ + static_cast<int64_t>(idx_heads_) * Di;
  if (rows.decode) {
    qsa_index_decode_update(raw_k, IW, w_.index_k_norm, d_inv_freq_, d_req, d_pos, rows.spans,
                            rows.num_requests, cache.block_tables, cache.blocks_per_request,
                            cache.ring, cache.index_cache, pools_per_block, kpool_, Di, rotary_,
                            eps_, mscale_, stream, rows.ring_snapshots);
  } else {
    const int32_t* table =
        cache.block_tables + static_cast<int64_t>(rows.request) * cache.blocks_per_request;
    qsa_index_compress_write(raw_k, IW, w_.index_k_norm, d_inv_freq_, table, pools_per_block,
                             rows.pos0 / kpool_, T / kpool_, cache.index_cache, kpool_, Di, rotary_,
                             eps_, mscale_, stream);
    qsa_index_tail_seed(raw_k, IW, d_req, d_pos, T, cache.ring, kpool_, Di, stream);
  }
  // Score every row's visible pools, select, attend.
  qsa_index_score(qi_, static_cast<int64_t>(idx_heads_) * Di, d_req, d_pos, T, cache.block_tables,
                  cache.blocks_per_request, cache.index_cache, pools_per_block, idx_heads_, Di, kpool_,
                  keys_ws_, max_pools_, stream);
  qsa_select_from_keys(keys_ws_, max_pools_, d_pos, T, select_k_, kpool_, max_selected_, topk_,
                       counts_, stream);
  // Small grids do not amortize the wider head group. Keep decode/verify and
  // short prefills on their existing kernel; both paths use identical arithmetic.
  const auto attend = !rows.decode && T >= 128 ? qsa_attn_prefill_partial : qsa_attn_partial;
  attend(qn_, static_cast<int64_t>(lh_) * D, cache.k_cache, cache.v_cache, d_req, topk_,
                   max_selected_, counts_, T, n_split_, lh_, lkv_, D, cache.block_tokens,
                   cache.block_tables, cache.blocks_per_request, scale_, m_ws_, l_ws_, c_ws_, stream);
  dsa_attn_combine(m_ws_, l_ws_, c_ws_, T, n_split_, lh_, D, c_out_, stream);
  qsa_gate_out(c_out_, q_ + D, QW, 2 * D, o_, T, lh_, D, stream);
  gemm_dense(g_, o_, static_cast<int64_t>(lh_) * D, w_.o_proj, w_.o_proj_fp8, out, GemmOut::BF16, T, H,
             lh_ * D, stream);
}

size_t QwenQsaLayer::scratch_bytes(const QwenTextConfig& cfg, int local_heads, int local_kv_heads,
                                   int max_tokens, int64_t max_pools) {
  const size_t M = static_cast<size_t>(std::max(max_tokens, 0));
  const size_t lh = static_cast<size_t>(local_heads), lkv = static_cast<size_t>(local_kv_heads);
  const size_t D = static_cast<size_t>(cfg.head_dim), Di = static_cast<size_t>(cfg.indexer_head_dim);
  const size_t nH = static_cast<size_t>(cfg.indexer_n_heads);
  const size_t max_selected = static_cast<size_t>(cfg.indexer_budget + cfg.indexer_compress_ratio - 1);
  const size_t n_split = static_cast<size_t>(std::max<size_t>(1, std::min<size_t>(8, (max_selected + 255) / 256)));
  size_t b = 0;
  b += M * (lh * 2 * D + 2 * lkv * D + lh * D + lkv * D + (nH + 1) * Di + nH * Di) * 2;  // q, k, v, qn, kn, idx, qi
  b += M * static_cast<size_t>(std::max<int64_t>(max_pools, 0)) * 8;                    // keys_ws
  b += M * max_selected * 4 + M * 4;                                                    // topk, counts
  b += 2 * M * n_split * lh * 4 + M * n_split * lh * D * 4;                             // m, l, c partials
  b += M * lh * D * 4 + M * lh * D * 2;                                                 // c_out, o
  return b;
}

// ---- QwenPleLayer ----------------------------------------------------------------

QwenPleLayer::QwenPleLayer(const QwenPleResident& w, const QwenNgramTableResident& table,
                           const QwenGemmWorkspace& gemm, const QwenTextConfig& cfg, int max_tokens)
    : w_(w), table_(table), g_(gemm), hc_(cfg.hc_count), hidden_(cfg.hidden_size),
      heads_((cfg.ngram_size - 1) * cfg.heads_per_ngram), heads_per_ngram_(cfg.heads_per_ngram),
      head_dim_(cfg.ple_embed_dim / ((cfg.ngram_size - 1) * cfg.heads_per_ngram)),
      width_(cfg.ple_conv_kernel_size), dilation_(cfg.ngram_size),
      state_len_((cfg.ple_conv_kernel_size - 1) * cfg.ngram_size), max_tokens_(max_tokens),
      eos_(cfg.eos_token_ids.empty() ? -1 : static_cast<int32_t>(cfg.eos_token_ids[0])),
      eps_(cfg.rms_norm_eps) {
  if (!g_.gemm || !g_.ws) throw std::invalid_argument("QwenPleLayer: GEMM workspace required");
  if (cfg.ngram_size != 3 || width_ != 4)
    throw std::invalid_argument("QwenPleLayer: ngram_size 3 / conv width 4 (the conv kernel's shape)");
  if (eos_ < 0) throw std::invalid_argument("QwenPleLayer: the config names no EOS token");
  const QwenNgramGeometry ng = cfg.ngram_geometry();
  if (ng.heads != heads_ || ng.head_dim != head_dim_)
    throw std::invalid_argument("QwenPleLayer: n-gram geometry disagrees with the config");
  d_mult_ = dev_alloc<int64_t>(ng.multipliers.size());
  d_vocab_ = dev_alloc<int64_t>(ng.head_vocab.size());
  d_offset_ = dev_alloc<int64_t>(ng.head_offset.size());
  DGPP_CUDA_OK(cudaMemcpy(d_mult_, ng.multipliers.data(), ng.multipliers.size() * 8, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_vocab_, ng.head_vocab.data(), ng.head_vocab.size() * 8, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_offset_, ng.head_offset.data(), ng.head_offset.size() * 8, cudaMemcpyHostToDevice));
  const size_t M = static_cast<size_t>(max_tokens_), W = static_cast<size_t>(hc_) * hidden_;
  if (table_.mmap) {
    // The ids where the host node reads them, the staging where the
    // device converts them; two eager argument blocks (a walk's callback
    // has run by the time the walk returns — the walk syncs) and a pair
    // of events for the fork and the join (captures turn them into
    // edges, so one pair serves every capture and the eager walks).
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_ids_), M * static_cast<size_t>(heads_) * 4,
                               cudaHostAllocMapped));
    DGPP_CUDA_OK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&ids_), h_ids_, 0));
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&staged_), M * static_cast<size_t>(w_.hash_heads) * head_dim_,
                               cudaHostAllocMapped));
    DGPP_CUDA_OK(cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking));
    DGPP_CUDA_OK(cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming));
    DGPP_CUDA_OK(cudaEventCreateWithFlags(&join_, cudaEventDisableTiming));
    stage_args_.push_back(std::make_unique<StageArgs>());
    stage_args_.push_back(std::make_unique<StageArgs>());
  } else {
    ids_ = dev_alloc<int32_t>(M * static_cast<size_t>(heads_));
  }
  e_ = dev_alloc<uint16_t>(M * static_cast<size_t>(w_.hash_heads) * head_dim_);
  key_ = dev_alloc<uint16_t>(M * W);
  kn_ = dev_alloc<uint16_t>(M * W);
  val_ = dev_alloc<uint16_t>(M * static_cast<size_t>(hidden_));
  qn_ = dev_alloc<uint16_t>(M * W);
  gv_ = dev_alloc<uint16_t>(M * W);
  un_ = dev_alloc<uint16_t>(M * W);
}

QwenPleLayer::~QwenPleLayer() {
  cudaFree(d_mult_);
  cudaFree(d_vocab_);
  cudaFree(d_offset_);
  if (h_ids_) {
    if (side_) cudaStreamSynchronize(side_);
    cudaFreeHost(h_ids_);
    cudaFreeHost(staged_);
    if (fork_) cudaEventDestroy(fork_);
    if (join_) cudaEventDestroy(join_);
    if (side_) cudaStreamDestroy(side_);
  } else {
    cudaFree(ids_);
  }
  cudaFree(e_);
  cudaFree(key_);
  cudaFree(kn_);
  cudaFree(val_);
  cudaFree(qn_);
  cudaFree(gv_);
  cudaFree(un_);
}

void QwenPleLayer::rebind(const QwenPleResident& w, const QwenNgramTableResident& table) {
  if (w.hash_heads != w_.hash_heads) throw std::invalid_argument("QwenPleLayer: rebind changes the head slice");
  w_ = w;
  table_ = table;
}

size_t QwenPleLayer::scratch_bytes(const QwenTextConfig& cfg, int hash_heads, int max_tokens) {
  const size_t M = static_cast<size_t>(std::max(max_tokens, 0));
  const size_t H = static_cast<size_t>(cfg.hidden_size), W = static_cast<size_t>(cfg.hc_count) * H;
  const size_t heads = static_cast<size_t>((cfg.ngram_size - 1) * cfg.heads_per_ngram);
  const size_t hd = static_cast<size_t>(cfg.ple_embed_dim) / std::max<size_t>(heads, 1);
  return M * heads * 4 + M * static_cast<size_t>(hash_heads) * hd * 2 + M * W * 2 * 5 + M * H * 2;
}

// The host node's arguments: one block per captured walk (its pointer is
// the node's), two alternating for the eager walks.
struct QwenPleLayer::StageArgs {
  const QwenNgramTableMmap* table = nullptr;
  const int32_t* ids = nullptr;
  uint8_t* dst = nullptr;
  int n = 0, heads = 0, head_begin = 0, heads_local = 0;
  std::atomic<int> error{0};
  std::string what;
};

void QwenPleLayer::stage_callback(void* user) {
  StageArgs* a = static_cast<StageArgs*>(user);
  try {
    a->table->gather(a->ids, a->n, a->heads, a->head_begin, a->heads_local, a->dst);
  } catch (const std::exception& e) {
    a->what = e.what();
    a->error.store(1, std::memory_order_release);
    DGPP_LOG_ERROR("QwenPleLayer: the n-gram staging failed: {}", e.what());
  }
}

void QwenPleLayer::check_staged() const {
  for (const auto& a : stage_args_)
    if (a->error.load(std::memory_order_acquire))
      throw std::runtime_error("QwenPleLayer: an n-gram staging failed on the host: " + a->what);
}

size_t QwenPleLayer::staging_bytes(const QwenTextConfig& cfg, int hash_heads, int max_tokens) {
  const size_t M = static_cast<size_t>(std::max(max_tokens, 0));
  const size_t heads = static_cast<size_t>((cfg.ngram_size - 1) * cfg.heads_per_ngram);
  const size_t hd = static_cast<size_t>(cfg.ple_embed_dim) / std::max<size_t>(heads, 1);
  return M * heads * 4 + M * static_cast<size_t>(hash_heads) * hd;
}

void QwenPleLayer::stage(const int64_t* tokens, int rows, const int32_t* req_ids, const int64_t* pos,
                         const int32_t* req_spans, int num_requests, const int32_t* ctx,
                         cudaStream_t stream) {
  if (!staged()) throw std::logic_error("QwenPleLayer: stage() without the mmap'ed table");
  if (rows <= 0) return;
  if (rows > max_tokens_) throw std::invalid_argument("QwenPleLayer: rows exceed max_tokens");
  if (staged_rows_ != 0) throw std::logic_error("QwenPleLayer: stage() twice without embed()");
  check_staged();
  qwen_ple_hash_ids_rows(tokens, rows, req_ids, pos, req_spans, num_requests, ctx, eos_, d_mult_,
                         d_vocab_, d_offset_, heads_, heads_per_ngram_, ids_, stream);
  cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
  DGPP_CUDA_OK(cudaStreamIsCapturing(stream, &cs));
  StageArgs* a = nullptr;
  if (cs != cudaStreamCaptureStatusNone) {
    stage_args_.push_back(std::make_unique<StageArgs>());
    a = stage_args_.back().get();
  } else {
    a = stage_args_[eager_slot_].get();
    eager_slot_ ^= 1;
  }
  a->table = table_.mmap;
  a->ids = h_ids_;
  a->dst = staged_;
  a->n = rows;
  a->heads = heads_;
  a->head_begin = w_.hash_head_begin;
  a->heads_local = w_.hash_heads;
  DGPP_CUDA_OK(cudaEventRecord(fork_, stream));
  DGPP_CUDA_OK(cudaStreamWaitEvent(side_, fork_, 0));
  DGPP_CUDA_OK(cudaLaunchHostFunc(side_, &QwenPleLayer::stage_callback, a));
  DGPP_CUDA_OK(cudaEventRecord(join_, side_));
  staged_rows_ = rows;
}

void QwenPleLayer::embed(const int64_t* tokens, int rows, const int32_t* req_ids, const int64_t* pos,
                         const int32_t* req_spans, int num_requests, const int32_t* ctx,
                         cudaStream_t stream) {
  if (rows <= 0) return;
  if (rows > max_tokens_) throw std::invalid_argument("QwenPleLayer: rows exceed max_tokens");
  if (staged()) {
    if (staged_rows_ != rows) throw std::logic_error("QwenPleLayer: embed() without a matching stage()");
    DGPP_CUDA_OK(cudaStreamWaitEvent(stream, join_, 0));
    qwen_ple_gather_staged_bf16(staged_, table_.scale, rows, w_.hash_heads, head_dim_, e_, stream);
    staged_rows_ = 0;
    return;
  }
  if (!table_.rows_e4m3) throw std::invalid_argument("QwenPleLayer: null table");
  qwen_ple_hash_ids_rows(tokens, rows, req_ids, pos, req_spans, num_requests, ctx, eos_, d_mult_,
                         d_vocab_, d_offset_, heads_, heads_per_ngram_, ids_, stream);
  qwen_ple_gather_bf16(table_.rows_e4m3, table_.row_begin, table_.rows, table_.scale, ids_, rows, heads_,
                       w_.hash_head_begin, w_.hash_heads, head_dim_, e_, stream);
}

void QwenPleLayer::project_key(uint16_t* key_dst, int rows, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!w_.key_proj && !w_.key_proj_fp8.payload) throw std::invalid_argument("QwenPleLayer: null weights");
  const int E = w_.hash_heads * head_dim_, W = hc_ * hidden_;
  gemm_dense(g_, e_, E, w_.key_proj, w_.key_proj_fp8, key_dst ? key_dst : key_, GemmOut::BF16, rows, W, E, stream);
}

void QwenPleLayer::norm_key(const uint16_t* key, int rows, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!w_.norm_key) throw std::invalid_argument("QwenPleLayer: null weights");
  qwen_group_rmsnorm_bf16(key, w_.norm_key, kn_, rows, hc_, hidden_, eps_, stream);
}

void QwenPleLayer::project_value(uint16_t* val_dst, int rows, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!w_.value_proj && !w_.value_proj_fp8.payload) throw std::invalid_argument("QwenPleLayer: null weights");
  const int E = w_.hash_heads * head_dim_;
  gemm_dense(g_, e_, E, w_.value_proj, w_.value_proj_fp8, val_dst ? val_dst : val_, GemmOut::BF16, rows, hidden_, E, stream);
}

void QwenPleLayer::finish(uint16_t* r, const uint16_t* value, uint16_t* states, int64_t state_stride,
                          const int32_t* req_ids, const int64_t* pos, const int32_t* req_spans,
                          int num_requests, int rows, cudaStream_t stream, uint16_t* snapshots) {
  if (rows <= 0) return;
  if (!w_.norm_query || !w_.norm_conv || !w_.conv)
    throw std::invalid_argument("QwenPleLayer: null weights");
  const int T = rows, H = hidden_, W = hc_ * hidden_;
  qwen_group_rmsnorm_bf16(r, w_.norm_query, qn_, T, hc_, H, eps_, stream);
  qwen_ple_gate_bf16(kn_, qn_, value, gv_, T, hc_, H, stream);
  qwen_group_rmsnorm_bf16(gv_, w_.norm_conv, un_, T, hc_, H, eps_, stream);
  qwen_ple_conv_rows_bf16(un_, gv_, states, state_stride, w_.conv, r, r, req_ids, pos, req_spans,
                          num_requests, W, width_, dilation_, stream, snapshots);
}

}  // namespace dgpp
