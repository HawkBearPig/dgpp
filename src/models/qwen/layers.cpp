#include "models/qwen/layers.hpp"

#include <ctime>

#include "kernels/glm_spec.hpp"

#include <chrono>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "kernels/bf12_gemv.hpp"
#include "kernels/bf16_gemv.hpp"
#include "kernels/mma_gemv.hpp"
#include "kernels/dsa.hpp"
#include "kernels/gdn_chunk.hpp"
#include "kernels/kda.hpp"
#include "kernels/qsa.hpp"
#include "kernels/rope_scaling.hpp"
#include "kernels/qwen_gr.hpp"
#include "kernels/fp8_dequant.hpp"
#include "kernels/fp8_gemm.hpp"
#include "kernels/fp8_blockwise_dense.hpp"
#include "kernels/full_attn.hpp"
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

bool use_chunked_gdn(int tokens, int k_dim, int v_dim) {
  static const bool enabled = [] {
    const char* e = std::getenv("DGPP_GDN_CHUNKED");
    return !(e != nullptr && e[0] == '0');
  }();
  static const int min_tokens = [] {
    const char* e = std::getenv("DGPP_GDN_CHUNKED_MIN");
    return e != nullptr ? std::atoi(e) : 64;
  }();
  return enabled && tokens >= min_tokens && gdn_chunked_supported(k_dim, v_dim);
}

void gemm_bf16(const QwenGemmWorkspace& g, const uint16_t* act, int64_t act_stride,
               const uint16_t* w, void* out, GemmOut out_type, int m, int n, int k,
               cudaStream_t stream) {
  g.gemm->matmul(act, w, out, m, n, k, DType::BF16, out_type, static_cast<size_t>(act_stride),
                 g.ws, g.ws_bytes, stream);
}

// Per-tensor FP8 prefill product (DGPP_FP8_PT_DENSE): D[M,N] = Act[M,K] x
// W[N,K]^T in E4M3 with scalar scales. The qwen35 model owns the weights,
// the scratch and the scale cells; the workspace's GEMM is its CublasLtGemm
// there (the only model that binds an enabled view).
void pt_gemm(const QwenGemmWorkspace& g, const uint8_t* act, const uint8_t* w,
             const float* act_scale, const float* w_scale, uint16_t* out, int m, int n, int k,
             cudaStream_t stream) {
  static_cast<CublasLtGemm*>(g.gemm)->matmul_fp8_scaled(act, w, act_scale, w_scale, out, m, n,
                                                       k, g.ws, g.ws_bytes, stream);
}

// One shared activation quantize over [tokens, width] into the view's
// scratch (the caller's projections all read the input it quantizes).
void pt_quant_input(const QwenPtAttnView& pt, const uint16_t* in, int tokens, int width,
                    cudaStream_t stream) {
  launch_fp8_row_maxabs(in, static_cast<size_t>(tokens) * width, pt.act_scale, stream);
  launch_fp8_quant_bf16(in, pt.act, static_cast<size_t>(tokens) * width, pt.act_scale, stream);
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
    // OPT-IN, NOT BITWISE (engine.prefill_fp8_gemm, 2026-09-30): the
    // product on the fp8 tensor cores from per-token e4m3 activations
    // (kernels/fp8_gemm) when the shape fits its contract; otherwise the
    // bridge below.
    // Not below 1,024 columns: the quantizer's pass over the activation
    // (3 bytes a value at line rate) costs 494 / n of the GEMM's time
    // (the [320 x 10240] GR down: +60 % measured), while the fp8 GEMM
    // itself runs at cuBLAS's rate and saves the dequant alone.
    if (m > 128 && n >= 1024 && g.a8 && QwenLayerStream::prefill_fp8_gemm() && k % 128 == 0 && w8.scale_block_cols == 128 &&
        static_cast<size_t>(m) * static_cast<size_t>(k) <= g.a8_bytes && act_stride % 4 == 0 &&
        (reinterpret_cast<uintptr_t>(act) % 8) == 0) {
      launch_fp8_quantize_rows(act, static_cast<size_t>(act_stride), m, k, g.a8, g.a8_scales, stream);
      if (out_type == GemmOut::F32)
        launch_fp8_gemm_f32(g.a8, g.a8_scales, w8.payload, w8.scales, w8.scale_block_rows, static_cast<float*>(out),
                            m, n, k, stream, static_cast<size_t>(n));
      else
        launch_fp8_gemm_bf16(g.a8, g.a8_scales, w8.payload, w8.scales, w8.scale_block_rows,
                             static_cast<uint16_t*>(out), m, n, k, stream, static_cast<size_t>(n));
      return;
    }
    if (m > 128 && g.dequant && bf16_bytes <= g.dequant_bytes) {
      launch_fp8_dequant_blocks(w8.payload, w8.scales, g.dequant, n, k, stream);
      gemm_bf16(g, act, act_stride, g.dequant, out, out_type, m, n, k, stream);
      return;
    }
    // The GEMM workspace rides along for the streaming form's split-K at a
    // small n (the hyperconnection down projection, [320 x 10240]).
    if (out_type == GemmOut::F32)
      launch_scale_gemm_f32(act, static_cast<size_t>(act_stride), w8.payload, w8.scales,
                            static_cast<float*>(out), m, n, k, stream, static_cast<size_t>(n), g.mma_from_rows,
                            /*last_row_only=*/false, g.ws, g.ws_bytes);
    else
      launch_scale_gemm_bf16(act, static_cast<size_t>(act_stride), w8.payload, w8.scales,
                             static_cast<uint16_t*>(out), m, n, k, stream, static_cast<size_t>(n), g.mma_from_rows,
                             g.ws, g.ws_bytes);
    return;
  }
  if (!w) throw std::invalid_argument("qwen dense: null weight");
  gemm_bf16(g, act, act_stride, w, out, out_type, m, n, k, stream);
}

}  // namespace

void qwen_configure_gemm_rows(CublasLtGemm& gemm, int tokens, bool decode) {
  const bool wide_decode = decode && tokens > 16;
  // Wide decode's GEMMs take the streaming tensor-core form: the weights read
  // ONCE for every stacked row, where the 4-row GEMV chunks (the kernel-only
  // band below) re-read them once per chunk — eight times at 32 rows, the
  // 8K profile's 83 % of a step (docs/mtp.md). Tolerance-equal, not bitwise:
  // the m=1..16 band keeps its dispatch, so the C1 gate's m=1 row stays the
  // GEMV's. The kernel-only band remains the guard (it still throws for a
  // shape neither form takes) and the fallback for the shapes the mma form
  // cannot take; the band's 64-row top extends to the mma's 128 (the c8/c16
  // verify batch was falling to the Lt algorithm).
  gemm.set_decode_mma(wide_decode, 17, kMmaGemvMaxRowsPerLaunch);
  gemm.set_kernel_only_rows(wide_decode ? 17 : 0, wide_decode ? kGemmDecodeLoweringRows : 0);
  gemm.set_bf12_wide(decode);
}

void qwen_mtp_hidden_projection(const QwenGemmWorkspace& g, const uint16_t* act,
    const uint16_t* weight, uint16_t* out, int tokens, int hc, int hidden,
    bool decode, cudaStream_t stream) {
  const int rows = tokens * hc;
  // Real H=2560, hc=4 at 12/16 verify rows: Lt's 48/64-row algorithm
  // records a memset node, which can deadlock collective graph replay.
  // Only the newly widened decode takes MMA; preserve prefill and all
  // existing <=8-row walks. This draft projection uses the same MMA row
  // chain at every wide shape, including the diagnostic lowering mode.
  if (decode && tokens > 8) {
    if (!mma_gemv_shape_ok(weight, act, static_cast<size_t>(hidden), rows, hidden))
      throw std::invalid_argument(
          "qwen_mtp_hidden_projection: wide decode requires an aligned MMA-supported shape; "
          "refusing cuBLASLt fallback");
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
  // DGPP_QWEN_GR_MIX_FUSED=off: the up GEMV and the mix as two launches
  // (the A/B knob of the 2026-09-29 act_up_mix fusion).
  const char* mix_knob = std::getenv("DGPP_QWEN_GR_MIX_FUSED");
  mix_fused_ = !(mix_knob != nullptr && std::string(mix_knob) == "off");
  // DGPP_QWEN_GR_NORM_FOLD=on: the batched rows' group norm staged into the
  // down + inject GEMV (kernels/qwen_gr norm_down_inject, bitwise). Off by
  // default: on the fabric 2026-09-29 the 41 blocks' redundant norms cost
  // more than the launch they save (54.3–54.4 against 53.5–53.6 ms a step).
  const char* nf_knob = std::getenv("DGPP_QWEN_GR_NORM_FOLD");
  norm_fold_ = nf_knob != nullptr && std::string(nf_knob) == "on";
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

void QwenGrSite::apply_pending(uint16_t* r, const PendingCombine& p, int tokens, int hidden,
                               cudaStream_t stream) {
  if (p.y == nullptr || tokens <= 0) return;
  qwen_gr_combine_apply_bf16(r, p.gates, p.y, tokens, p.hc, hidden, stream);
}

QwenGrSite::PendingCombine QwenGrSite::defer_combine(uint16_t* r, const uint16_t* y, int tokens,
                                                     cudaStream_t stream) {
  (void)r;  // the apply is the next mix's (combine_norm) in every form now
  if (tokens <= 0) return {};
  if (!w_.inject) throw std::invalid_argument("QwenGrSite: combine on a site without inject weights");
  if (gates_ready_) {  // the mix's down GEMV wrote them: the next mix applies
    gates_ready_ = false;
    return PendingCombine{y, gates_, hc_};
  }
  if (gate_forked_) {  // the side stream's dots: join, then the next mix applies
    DGPP_CUDA_OK(cudaEventRecord(gate_join_, gate_side_));
    DGPP_CUDA_OK(cudaStreamWaitEvent(stream, gate_join_, 0));
    gate_forked_ = false;
    return PendingCombine{y, gates_, hc_};
  }
  // The prefill rows (2026-09-29): the gates as their own dots kernel here,
  // the apply inside the next mix's norm pass (combine_norm: bitwise apply +
  // group norm, one pass over R instead of two — 12 % of a 32K prefill was
  // these elementwise passes).
  qwen_gr_combine_dots_bf16(rn_, w_.inject, gates_, tokens, hc_, hidden_, stream);
  return PendingCombine{y, gates_, hc_};
}

void QwenGrSite::mix(uint16_t* r, uint16_t* x, int tokens, cudaStream_t stream,
                     const PendingCombine* pending) {
  if (tokens <= 0) return;
  if (tokens > max_tokens_) throw std::invalid_argument("QwenGrSite: tokens exceed max_tokens");
  // Either form of the two projections (the checkpoint's BF16, or block FP8).
  if (!w_.hc_norm || !(w_.down || w_.down_fp8.payload) || !(w_.up || w_.up_fp8.payload))
    throw std::invalid_argument("QwenGrSite: null weights");
  const int W = hc_ * hidden_;
  // The fused decode forms in the weights' form: BF16, or block FP8 (the
  // fp8 twins, 2026-09-10 — bitwise the unfused fp8 chain).
  const bool dense_fp8 = w_.down_fp8.payload != nullptr;
  const bool pend = pending != nullptr && pending->y != nullptr;
  if (pend && pending->hc != hc_) throw std::invalid_argument("QwenGrSite: a pending combine of another width");
  if (tokens <= fused_rows_max_ && fused_mix_) {
    // The one-row fused kernel normalizes R itself: the pending combine
    // lands first, as its own launch.
    if (pend) qwen_gr_combine_apply_bf16(r, pending->gates, pending->y, tokens, hc_, hidden_, stream);
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
    // The up GEMV with the mix in its epilogue (2026-09-29): one launch,
    // no logits round trip, bitwise the act_up + mix_finish chain.
    if (mix_fused_) {
      if (dense_fp8)
        qwen_gr_act_up_mix_fp8(t_, lowrank_, hc_, w_.up_fp8.payload, w_.up_fp8.scales, rn_, x, hidden_, tokens, stream);
      else
        qwen_gr_act_up_mix_bf16(t_, lowrank_, hc_, w_.up, rn_, x, hidden_, tokens, stream);
      return;
    }
    if (dense_fp8)
      qwen_gr_act_up_fp8(t_, lowrank_, hc_, w_.up_fp8.payload, w_.up_fp8.scales, logits_, hidden_, tokens, stream);
    else
      qwen_gr_act_up_bf16(t_, lowrank_, hc_, w_.up, logits_, hidden_, tokens, stream);
  } else {
    const bool batched = tokens <= 8 && inject_fused() && fused_mix_ && !gate_side_only_;
    const Bf12Matrix* pk = (batched && !dense_fp8) ? g_.gemm->bf12_lookup(w_.down) : nullptr;
    if (norm_fold_ && batched && pk == nullptr) {
      // The batched decode rows' group norm staged into the down + inject
      // GEMV (2026-09-29; one block reduction for every (row, group):
      // bitwise the norm + down_inject chain); the previous site's combine
      // as its own launch first — every down block reads R.
      if (pend) qwen_gr_combine_apply_bf16(r, pending->gates, pending->y, tokens, hc_, hidden_, stream);
      if (dense_fp8)
        qwen_gr_norm_down_inject_fp8(r, static_cast<size_t>(W), w_.hc_norm, hc_, hidden_, eps_, rn_,
                                     w_.down_fp8.payload, w_.down_fp8.scales, t_, lowrank_, w_.inject, gates_,
                                     tokens, stream);
      else
        qwen_gr_norm_down_inject_bf16(r, static_cast<size_t>(W), w_.hc_norm, hc_, hidden_, eps_, rn_, w_.down, t_,
                                      lowrank_, w_.inject, gates_, tokens, stream);
      gates_ready_ = true;
    } else {
      if (pend)  // the previous site's combine inside this norm's launch
        qwen_gr_combine_norm_bf16(r, pending->gates, pending->y, w_.hc_norm, rn_, tokens, hc_, hidden_, eps_, stream);
      else
        qwen_group_rmsnorm_bf16(r, w_.hc_norm, rn_, tokens, hc_, hidden_, eps_, stream);
      if (batched) {
        // The batched decode rows (MTP, the row batches): the down GEMV with
        // the inject rows appended — no side stream (kernels/qwen_gr).
        if (dense_fp8)
          qwen_gr_down_inject_fp8(rn_, w_.down_fp8.payload, w_.down_fp8.scales, t_, lowrank_, w_.inject, gates_,
                                  hc_, hidden_, tokens, stream);
        else if (pk != nullptr)  // the down's 12-bit companion (world 1; bitwise the bf16 chain)
          qwen_gr_down_inject_bf12(rn_, *pk, t_, lowrank_, w_.inject, gates_, hc_, hidden_, tokens, stream);
        else
          qwen_gr_down_inject_bf16(rn_, w_.down, t_, lowrank_, w_.inject, gates_, hc_, hidden_,
                                   tokens, stream);
        gates_ready_ = true;
      } else {
        fork_gate_dots(stream, tokens);
        gemm_dense(g_, rn_, W, w_.down, w_.down_fp8, t_, GemmOut::BF16, tokens, lowrank_, W, stream);
      }
    }
    if (tokens <= 8 && fused_mix_) {
      // The batched decode rows' up GEMV with the gate activation folded
      // into its staging and the mix into its epilogue (kernels/qwen_gr
      // act_up_mix: bitwise gate_act + the GEMV + mix_finish) — two launches
      // and the logits round trip fewer per site (2026-09-29).
      if (mix_fused_) {
        if (dense_fp8)
          qwen_gr_act_up_mix_fp8(t_, lowrank_, hc_, w_.up_fp8.payload, w_.up_fp8.scales, rn_, x, hidden_, tokens, stream);
        else
          qwen_gr_act_up_mix_bf16(t_, lowrank_, hc_, w_.up, rn_, x, hidden_, tokens, stream);
        return;
      }
      if (dense_fp8)
        qwen_gr_act_up_fp8(t_, lowrank_, hc_, w_.up_fp8.payload, w_.up_fp8.scales, logits_, hidden_, tokens, stream);
      else
        qwen_gr_act_up_bf16(t_, lowrank_, hc_, w_.up, logits_, hidden_, tokens, stream);
    } else {
      qwen_gr_gate_act_bf16(t_, tokens, lowrank_, hc_, stream);
      gemm_dense(g_, t_, lowrank_, w_.up, w_.up_fp8, logits_, GemmOut::BF16, tokens, W, lowrank_, stream);
    }
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
                           const QwenTextConfig& cfg, int max_tokens, bool swish_gate)
     : w_(w), g_(gemm), hidden_(cfg.hidden_size), lk_(w.local_key_heads), lv_(w.local_value_heads),
       k_dim_(cfg.gdn_key_head_dim), v_dim_(cfg.gdn_value_head_dim), conv_width_(cfg.gdn_conv_width),
       max_tokens_(max_tokens), eps_(cfg.rms_norm_eps), swish_gate_(swish_gate) {
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
  if (use_chunked_gdn(max_tokens_, k_dim_, v_dim_)) {
    chunked_ws_bytes_ = gdn_chunked_workspace_bytes(max_tokens_, lv_);
    chunked_ws_ = dev_alloc<uint8_t>(chunked_ws_bytes_);
  }
}

QwenGdnLayer::~QwenGdnLayer() {
  cudaFree(qkv_);
  cudaFree(qkvc_);
  cudaFree(z_);
  cudaFree(a_);
  cudaFree(b_);
  cudaFree(core_);
  cudaFree(normed_);
  cudaFree(chunked_ws_);
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
void QwenGdnLayer::in_projections(const uint16_t* x, int tokens, cudaStream_t stream, bool resume) {
  const int H = hidden_;
  const int C = static_cast<int>(conv_channels_);
  const int LV = lv_ * v_dim_;
  if (w_.in_proj_qkv_fp8.payload) {
    // The FP8 form: qkv and z as one multi-problem fp8 GEMV at decode rows
    //, the scale GEMM above them; a and b (BF16, [lv, H]) as
    // one dual GEMV.
    const bool ab_gemv = tokens <= std::min(4, g_.gemv_rows) && bf16_gemv_accepts(w_.in_proj_a, tokens, H) &&
                         bf16_gemv_accepts(w_.in_proj_b, tokens, H);
    if (pt_.enabled && (tokens > 128 || resume)) {
      // Per-tensor prefill: one shared activation quantize feeds qkv and z;
      // a and b stay BF16 (2 x [lv, H], negligible next to C + LV).
      pt_quant_input(pt_, x, tokens, H, stream);
      pt_gemm(g_, pt_.act, pt_.xw[0], pt_.act_scale, pt_.xscales, qkv_, tokens, C, H, stream);
      pt_gemm(g_, pt_.act, pt_.xw[1], pt_.act_scale, pt_.xscales + 1, z_, tokens, LV, H, stream);
    } else if (tokens <= std::min(8, g_.gemv_rows)) {
      // a and b (BF16 [lv, H]) as bf16 problems of the same launch when the
      // GEMV takes them (2026-09-29): bitwise the dual launch they replace.
      Fp8GemvProblem p[4];
      p[0].payload = w_.in_proj_qkv_fp8.payload; p[0].scales = w_.in_proj_qkv_fp8.scales; p[0].out = qkv_; p[0].n = C;
      p[1].payload = w_.in_proj_z_fp8.payload; p[1].scales = w_.in_proj_z_fp8.scales; p[1].out = z_; p[1].n = LV;
      p[2].bf16_weight = w_.in_proj_a; p[2].out = a_; p[2].n = lv_;
      p[3].bf16_weight = w_.in_proj_b; p[3].out = b_; p[3].n = lv_;
      launch_scale_gemv_multi_bf16(p, ab_gemv ? 4 : 2, x, static_cast<size_t>(H), tokens, H, stream);
      if (ab_gemv) return;
    } else {
      gemm_dense(g_, x, H, nullptr, w_.in_proj_qkv_fp8, qkv_, GemmOut::BF16, tokens, C, H, stream);
      gemm_dense(g_, x, H, nullptr, w_.in_proj_z_fp8, z_, GemmOut::BF16, tokens, LV, H, stream);
    }
    if (ab_gemv) {
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
    // Their lossless 12-bit companions (engine.bf16_weights), when the GEMM
    // holds any: the multi launch's packed twin — bitwise this launch, 0.75
    // of the bytes; a projection without one runs the bf16 chain in its
    // blocks.
    const Bf12Matrix* pk[4] = {g_.gemm->bf12_lookup(w_.in_proj_qkv), g_.gemm->bf12_lookup(w_.in_proj_z),
                               g_.gemm->bf12_lookup(w_.in_proj_a), g_.gemm->bf12_lookup(w_.in_proj_b)};
    if (pk[0] || pk[1] || pk[2] || pk[3]) {
      const uint16_t* wt[4] = {w_.in_proj_qkv, w_.in_proj_z, w_.in_proj_a, w_.in_proj_b};
      void* outs[4] = {qkv_, z_, a_, b_};
      const int ns[4] = {C, LV, lv_, lv_};
      Bf12GemvProblem q[4];
      for (int i = 0; i < 4; ++i) {
        q[i].act = x;
        q[i].act_row_stride = static_cast<size_t>(H);
        if (pk[i]) q[i].packed = *pk[i];
        q[i].weight = wt[i];
        q[i].out = outs[i];
        q[i].n = ns[i];
      }
      launch_bf12_gemv_multi(q, 4, /*out_f32=*/false, tokens, H, stream);
      return;
    }
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
                           const KdaStateSnapshots& rec_snap, const KdaConvSnapshots& conv_snap,
                           const KdaReplay& replay, bool resume) {
  if (tokens <= 0) return;
  if (tokens > max_tokens_) throw std::invalid_argument("QwenGdnLayer: tokens exceed max_tokens");
  if (!(w_.in_proj_qkv || w_.in_proj_qkv_fp8.payload) || !w_.conv || !(w_.in_proj_z || w_.in_proj_z_fp8.payload) ||
      !w_.in_proj_a || !w_.in_proj_b || !w_.a_log || !w_.dt_bias || !w_.norm ||
      !(w_.out_proj || w_.out_proj_fp8.payload))
    throw std::invalid_argument("QwenGdnLayer: null weights");
  const int H = hidden_;
  const int C = static_cast<int>(conv_channels_);
  const int LV = lv_ * v_dim_;
  in_projections(x, tokens, stream, resume);
  kda_causal_conv_silu_bf16(qkv_, C, w_.conv, conv_state, conv_width_ - 1, qkvc_, tokens, C,
                            conv_width_, stream, conv_snap);
  // Prefill-sized walks take the chunked tensor-core form (kernels/
  // gdn_chunk.cu): flash-linear-attention's chunk_gated_delta_rule, the
  // algorithm SGLang runs for these layers, with bf16 matrix operands.
  // Walks of fewer than DGPP_GDN_CHUNKED_MIN rows (default 64) and
  // speculative snapshot rows keep the recurrence; DGPP_GDN_CHUNKED=0 keeps
  // it everywhere.
  if (use_chunked_gdn(tokens, k_dim_, v_dim_) && rec_snap.states == nullptr && replay.in == nullptr)
    gdn_chunked_fwd(qkvc_, a_, lv_, b_, lv_, w_.a_log, w_.dt_bias, recurrent_state, core_, tokens,
                    lv_, lv_ / lk_, k_dim_, v_dim_, scale_, chunked_ws_, chunked_ws_bytes_, stream);
  else
    gdn_recurrent_fwd(qkvc_, a_, lv_, b_, lv_, w_.a_log, w_.dt_bias, recurrent_state, core_, tokens,
                      lv_, lv_ / lk_, k_dim_, v_dim_, scale_, stream, rec_snap, replay);
  if (swish_gate_)
    gdn_gated_rmsnorm_swish_bf16(core_, z_, w_.norm, normed_, static_cast<int64_t>(tokens) * lv_, v_dim_,
                                 eps_, stream);
  else
    gdn_gated_rmsnorm_bf16(core_, z_, w_.norm, normed_, static_cast<int64_t>(tokens) * lv_, v_dim_,
                           eps_, stream);
  if (pt_.enabled && (tokens > 128 || resume)) {
    pt_quant_input(pt_, normed_, tokens, LV, stream);
    pt_gemm(g_, pt_.act, pt_.ow, pt_.act_scale, pt_.oscale, out, tokens, H, LV, stream);
  } else {
    gemm_dense(g_, normed_, LV, w_.out_proj, w_.out_proj_fp8, out, GemmOut::BF16, tokens, H, LV,
               stream);
  }
}

void QwenGdnLayer::enqueue_rows(const uint16_t* x, float* rec_states, int64_t rec_stride,
                                uint16_t* conv_states, int64_t conv_stride, uint16_t* out,
                                int rows, const KdaRequestRows& requests, cudaStream_t stream,
                                const KdaStateSnapshots& rec_snap,
                                const KdaConvSnapshots& conv_snap, const KdaReplay& replay) {
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
  in_projections(x, rows, stream, /*resume=*/false);
  kda_causal_conv_silu_bf16_batched(qkv_, C, w_.conv, conv_states, conv_stride, conv_width_ - 1,
                                    qkvc_, rows, C, conv_width_, requests, stream, conv_snap);
  gdn_recurrent_fwd_batched(qkvc_, a_, lv_, b_, lv_, w_.a_log, w_.dt_bias, rec_states, rec_stride,
                            core_, rows, lv_, lv_ / lk_, k_dim_, v_dim_, scale_, requests, stream,
                            rec_snap, replay);
  if (swish_gate_)
    gdn_gated_rmsnorm_swish_bf16(core_, z_, w_.norm, normed_, static_cast<int64_t>(rows) * lv_, v_dim_,
                                 eps_, stream);
  else
    gdn_gated_rmsnorm_bf16(core_, z_, w_.norm, normed_, static_cast<int64_t>(rows) * lv_, v_dim_,
                           eps_, stream);
  if (pt_.enabled && rows > 128) {
    pt_quant_input(pt_, normed_, rows, LV, stream);
    pt_gemm(g_, pt_.act, pt_.ow, pt_.act_scale, pt_.oscale, out, rows, H, LV, stream);
  } else {
    gemm_dense(g_, normed_, LV, w_.out_proj, w_.out_proj_fp8, out, GemmOut::BF16, rows, H, LV,
               stream);
  }
}

void QwenGdnLayer::materialize(float* recurrent_state, const KdaReplay& replay, cudaStream_t stream) {
  if (!w_.a_log || !w_.dt_bias) throw std::invalid_argument("QwenGdnLayer: null weights");
  if (replay.in == nullptr || !replay.materialize || (replay.materialize_rows < 0 && replay.count == nullptr))
    throw std::invalid_argument("QwenGdnLayer: materialize needs the replay rows and a count");
  gdn_recurrent_fwd(nullptr, nullptr, lv_, nullptr, lv_, w_.a_log, w_.dt_bias, recurrent_state, nullptr,
                    /*tokens=*/0, lv_, lv_ / lk_, k_dim_, v_dim_, scale_, stream, KdaStateSnapshots{}, replay);
}

size_t QwenGdnLayer::scratch_bytes(const QwenTextConfig& cfg, int local_key_heads, int local_value_heads,
                                   int max_tokens) {
  const size_t M = static_cast<size_t>(std::max(max_tokens, 0));
  const size_t lk = static_cast<size_t>(local_key_heads), lv = static_cast<size_t>(local_value_heads);
  const size_t K = static_cast<size_t>(cfg.gdn_key_head_dim), V = static_cast<size_t>(cfg.gdn_value_head_dim);
  const size_t C = 2 * lk * K + lv * V;
  const size_t chunked = use_chunked_gdn(max_tokens, cfg.gdn_key_head_dim, cfg.gdn_value_head_dim)
                             ? gdn_chunked_workspace_bytes(max_tokens, local_value_heads)
                             : 0;
  return M * (2 * C + 3 * lv * V + 2 * lv) * 2 + chunked;  // qkv, qkvc, z, core, normed, a, b
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
    // The FP8 form at decode rows: the projections as one multi-problem
    // fp8 GEMV — the four of them, or q / k / v alone when the indexer's
    // projection ships in BF16 (the AutoRound hybrid: as shipped, D4), which
    // then takes the dense path. Every problem's rows are bitwise the
    // single launch's, whatever the count.
    Fp8GemvProblem p[4];
    p[0].payload = w_.q_proj_fp8.payload; p[0].scales = w_.q_proj_fp8.scales; p[0].out = q_; p[0].n = QW;
    p[1].payload = w_.k_proj_fp8.payload; p[1].scales = w_.k_proj_fp8.scales; p[1].out = k_; p[1].n = KW;
    p[2].payload = w_.v_proj_fp8.payload; p[2].scales = w_.v_proj_fp8.scales; p[2].out = v_; p[2].n = KW;
    int np = 3;
    if (w_.index_qk_proj_fp8.payload) {
      p[3].payload = w_.index_qk_proj_fp8.payload; p[3].scales = w_.index_qk_proj_fp8.scales; p[3].out = idx_; p[3].n = IW;
      np = 4;
    }
    launch_scale_gemv_multi_bf16(p, np, x, static_cast<size_t>(H), T, H, stream);
    if (np == 3)
      gemm_dense(g_, x, H, w_.index_qk_proj, w_.index_qk_proj_fp8, idx_, GemmOut::BF16, T, IW, H, stream);
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
  // Score only rows that need a selection, then expand pools and attend.
  // Prefill positions are fixed here; decode graph positions can grow on replay.
  qsa_index_score(qi_, static_cast<int64_t>(idx_heads_) * Di, d_req, d_pos, T, cache.block_tables,
                  cache.blocks_per_request, cache.index_cache, pools_per_block, idx_heads_, Di,
                  kpool_, keys_ws_, max_pools_, stream, rows.decode ? -1 : (rows.pos0 + T) / kpool_,
                  select_k_);
  qsa_select_from_keys(keys_ws_, max_pools_, d_pos, T, select_k_, kpool_, max_selected_, topk_,
                       counts_, stream);
  // Small grids do not amortize the wider head group. Keep decode/verify and
  // short prefills on their existing kernel; both paths use identical arithmetic.
  // Long prefill walks take the one-warp tensor-core kernel
  // (kernels/qsa_warp.cu), which writes c_out directly -- the shape and the
  // bf16-probability numerics SGLang's sparse GQA prefill kernel runs.
  // DGPP_QSA_WARP=0 keeps the partial kernels + combine.
  static const bool warp_attn = [] {
    const char* e = std::getenv("DGPP_QSA_WARP");
    return !(e != nullptr && e[0] == '0');
  }();
  if (warp_attn && !rows.decode && T >= 128 && qsa_warp_supported(D, lh_, lkv_)) {
    qsa_attn_prefill_warp(qn_, static_cast<int64_t>(lh_) * D, cache.k_cache, cache.v_cache, d_req, topk_,
                          max_selected_, counts_, T, lh_, lkv_, cache.block_tokens, cache.block_tables,
                          cache.blocks_per_request, scale_, c_out_, stream);
  } else {
    const auto attend = !rows.decode && T >= 128 ? qsa_attn_prefill_partial : qsa_attn_partial;
    attend(qn_, static_cast<int64_t>(lh_) * D, cache.k_cache, cache.v_cache, d_req, topk_,
           max_selected_, counts_, T, n_split_, lh_, lkv_, D, cache.block_tokens,
           cache.block_tables, cache.blocks_per_request, scale_, m_ws_, l_ws_, c_ws_, stream);
    dsa_attn_combine(m_ws_, l_ws_, c_ws_, T, n_split_, lh_, D, c_out_, stream);
  }
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

// ---- QwenFullAttnLayer --------------------------------------------------------------

QwenFullAttnLayer::QwenFullAttnLayer(const QwenFullAttnResident& w, const QwenGemmWorkspace& gemm,
                                     const QwenTextConfig& cfg, int max_tokens)
    : w_(w),
      g_(gemm),
      hidden_(cfg.hidden_size),
      lh_(w.local_heads),
      lkv_(w.local_kv_heads),
      dim_(cfg.head_dim),
      rotary_(cfg.rotary_dim),
      max_tokens_(max_tokens),
      eps_(cfg.rms_norm_eps) {
  if (!g_.gemm || !g_.ws) throw std::invalid_argument("QwenFullAttnLayer: GEMM workspace required");
  if (lh_ <= 0 || lkv_ <= 0 || lh_ % lkv_ != 0)
    throw std::invalid_argument("QwenFullAttnLayer: query heads must be a multiple of kv heads");
  if (dim_ != 256) throw std::invalid_argument("QwenFullAttnLayer: head_dim 256 (the attention kernel's shape)");
  if (rotary_ <= 0 || rotary_ > dim_)
    throw std::invalid_argument("QwenFullAttnLayer: rotary_dim must be within (0, head_dim]");
  if (!full_attn_supported(dim_, lh_, lkv_))
    throw std::invalid_argument("QwenFullAttnLayer: geometry outside the dense kernel's shape");
  scale_ = static_cast<float>(std::pow(static_cast<double>(dim_), -0.5));
  std::vector<float> inv(static_cast<size_t>(rotary_ / 2));
  // Same rope table rule as QSA: the plain table, or the YaRN ramp the
  // engine knob asks for (kernels/rope_scaling.hpp).
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
  c_out_ = dev_alloc<float>(M * static_cast<size_t>(lh_) * dim_);
  o_ = dev_alloc<uint16_t>(M * static_cast<size_t>(lh_) * dim_);
}

QwenFullAttnLayer::~QwenFullAttnLayer() {
  cudaFree(d_inv_freq_);
  cudaFree(q_);
  cudaFree(k_);
  cudaFree(v_);
  cudaFree(qn_);
  cudaFree(kn_);
  cudaFree(c_out_);
  cudaFree(o_);
}

void QwenFullAttnLayer::rebind(const QwenFullAttnResident& w) {
  if (w.local_heads != lh_ || w.local_kv_heads != lkv_)
    throw std::invalid_argument("QwenFullAttnLayer: rebind changes the head geometry");
  w_ = w;
}

void QwenFullAttnLayer::enqueue(const uint16_t* x, int tokens, const QwenQsaRows& rows,
                                QwenFullAttnCache& cache, uint16_t* out, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (tokens > max_tokens_) throw std::invalid_argument("QwenFullAttnLayer: tokens exceed max_tokens");
  if (!rows.req_ids || !rows.pos) throw std::invalid_argument("QwenFullAttnLayer: null row metadata");
  if (rows.decode) {
    if (rows.num_requests <= 0)
      throw std::invalid_argument("QwenFullAttnLayer: decode rows need requests");
  } else {
    if (rows.pos0 < 0) throw std::invalid_argument("QwenFullAttnLayer: pos0 must be non-negative");
    if (rows.request < 0 || rows.request >= cache.max_requests)
      throw std::invalid_argument("QwenFullAttnLayer: request outside the cache");
    if (rows.pos0 + tokens > cache.slots())
      throw std::invalid_argument("QwenFullAttnLayer: the prefill overruns the cache");
  }
  if (!(w_.q_proj || w_.q_proj_fp8.payload) || !(w_.k_proj || w_.k_proj_fp8.payload) ||
      !(w_.v_proj || w_.v_proj_fp8.payload) || !(w_.o_proj || w_.o_proj_fp8.payload) || !w_.q_norm ||
      !w_.k_norm)
    throw std::invalid_argument("QwenFullAttnLayer: null weights");
  const int H = hidden_, D = dim_, T = tokens;
  const int QW = lh_ * 2 * D, KW = lkv_ * D;
  const int32_t* d_req = rows.req_ids;
  const int64_t* d_pos = rows.pos;

  // Projections. A resume chunk (a prefill continuation at pos0 > 0)
  // takes the per-tensor path at any row count — decode walks carry
  // rows.decode and keep their exact GEMV dispatch.
  const bool resume = !rows.decode && rows.pos0 > 0;
  if (pt_.enabled && (T > 128 || resume)) {
    // Per-tensor prefill: one shared activation quantize feeds q, k, v.
    pt_quant_input(pt_, x, T, H, stream);
    pt_gemm(g_, pt_.act, pt_.xw[0], pt_.act_scale, pt_.xscales, q_, T, QW, H, stream);
    pt_gemm(g_, pt_.act, pt_.xw[1], pt_.act_scale, pt_.xscales + 1, k_, T, KW, H, stream);
    pt_gemm(g_, pt_.act, pt_.xw[2], pt_.act_scale, pt_.xscales + 2, v_, T, KW, H, stream);
  } else if (w_.q_proj_fp8.payload && T <= std::min(8, g_.gemv_rows)) {
    // The FP8 form at decode rows: the three projections as one
    // multi-problem fp8 GEMV (no indexer problem here).
    Fp8GemvProblem p[3];
    p[0].payload = w_.q_proj_fp8.payload; p[0].scales = w_.q_proj_fp8.scales; p[0].out = q_; p[0].n = QW;
    p[1].payload = w_.k_proj_fp8.payload; p[1].scales = w_.k_proj_fp8.scales; p[1].out = k_; p[1].n = KW;
    p[2].payload = w_.v_proj_fp8.payload; p[2].scales = w_.v_proj_fp8.scales; p[2].out = v_; p[2].n = KW;
    launch_scale_gemv_multi_bf16(p, 3, x, static_cast<size_t>(H), T, H, stream);
  } else {
    gemm_dense(g_, x, H, w_.q_proj, w_.q_proj_fp8, q_, GemmOut::BF16, T, QW, H, stream);
    gemm_dense(g_, x, H, w_.k_proj, w_.k_proj_fp8, k_, GemmOut::BF16, T, KW, H, stream);
    gemm_dense(g_, x, H, w_.v_proj, w_.v_proj_fp8, v_, GemmOut::BF16, T, KW, H, stream);
  }
  // Norm + RoPE: q (the [q | gate] interleave), k. Same kernels as QSA.
  qsa_norm_rope_bf16(q_, QW, 2 * D, w_.q_norm, d_pos, d_inv_freq_, qn_, static_cast<int64_t>(lh_) * D,
                     T, lh_, D, rotary_, eps_, mscale_, stream);
  qsa_norm_rope_bf16(k_, KW, D, w_.k_norm, d_pos, d_inv_freq_, kn_, KW, T, lkv_, D, rotary_, eps_,
                     mscale_, stream);
  // The caches: K/V rows (no compressed keys, no ring).
  qsa_kv_append(kn_, KW, v_, KW, d_req, d_pos, T, cache.block_tables, cache.blocks_per_request,
                cache.block_tokens, lkv_, D, cache.k_cache, cache.v_cache, stream);
  // Dense causal attention over [0, pos] per row (kernels/full_attn.cu).
  full_attn_prefill_warp(qn_, static_cast<int64_t>(lh_) * D, cache.k_cache, cache.v_cache, d_req,
                         d_pos, T, lh_, lkv_, cache.block_tokens, cache.block_tables,
                         cache.blocks_per_request, scale_, c_out_, stream);
  qsa_gate_out(c_out_, q_ + D, QW, 2 * D, o_, T, lh_, D, stream);
  if (pt_.enabled && (T > 128 || resume)) {
    pt_quant_input(pt_, o_, T, lh_ * D, stream);
    pt_gemm(g_, pt_.act, pt_.ow, pt_.act_scale, pt_.oscale, out, T, H, lh_ * D, stream);
  } else {
    gemm_dense(g_, o_, static_cast<int64_t>(lh_) * D, w_.o_proj, w_.o_proj_fp8, out, GemmOut::BF16,
               T, H, lh_ * D, stream);
  }
}

size_t QwenFullAttnLayer::scratch_bytes(const QwenTextConfig& cfg, int local_heads, int local_kv_heads,
                                        int max_tokens) {
  const size_t M = static_cast<size_t>(std::max(max_tokens, 0));
  const size_t lh = static_cast<size_t>(local_heads), lkv = static_cast<size_t>(local_kv_heads);
  const size_t D = static_cast<size_t>(cfg.head_dim);
  size_t b = 0;
  b += M * (lh * 2 * D + 2 * lkv * D + lh * D + lkv * D) * 2;  // q, k, v, qn, kn
  b += M * lh * D * 4 + M * lh * D * 2;                        // c_out, o
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
    if (const char* v = std::getenv("DGPP_QWEN_PLE_HOST_NODE"); v && *v && std::string(v) != "0") host_node_ = true;
    if (!host_node_) {
      DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_hash_seq_), sizeof(uint64_t)));
      DGPP_CUDA_OK(cudaMemset(d_hash_seq_, 0, sizeof(uint64_t)));
      DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_wait_seq_), sizeof(uint64_t)));
      DGPP_CUDA_OK(cudaMemset(d_wait_seq_, 0, sizeof(uint64_t)));
      DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_hash_seq_), sizeof(uint64_t), cudaHostAllocMapped));
      DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_hash_rows_), sizeof(int32_t), cudaHostAllocMapped));
      DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_done_seq_), sizeof(uint64_t), cudaHostAllocMapped));
      DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_late_), sizeof(uint32_t), cudaHostAllocMapped));
      *h_hash_seq_ = 0;
      *h_hash_rows_ = 0;
      *h_done_seq_ = 0;
      *h_late_ = 0;
      // The prestage channel.
      DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_pre_ids_), M * static_cast<size_t>(heads_) * 4,
                                 cudaHostAllocMapped));
      DGPP_CUDA_OK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&pre_ids_), h_pre_ids_, 0));
      DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&pre_staged_),
                                 M * static_cast<size_t>(w_.hash_heads) * head_dim_, cudaHostAllocMapped));
      DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_pre_seq_), sizeof(uint64_t)));
      DGPP_CUDA_OK(cudaMemset(d_pre_seq_, 0, sizeof(uint64_t)));
      DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_pre_wait_seq_), sizeof(uint64_t)));
      DGPP_CUDA_OK(cudaMemset(d_pre_wait_seq_, 0, sizeof(uint64_t)));
      DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_pre_seq_), sizeof(uint64_t), cudaHostAllocMapped));
      DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_pre_rows_), sizeof(int32_t), cudaHostAllocMapped));
      DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_pre_done_seq_), sizeof(uint64_t), cudaHostAllocMapped));
      DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_pre_need_), sizeof(uint64_t), cudaHostAllocMapped));
      *h_pre_seq_ = 0;
      *h_pre_rows_ = 0;
      *h_pre_done_seq_ = 0;
      *h_pre_need_ = 0;
      DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_pre_tokens_), M * sizeof(int64_t)));
      DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_pre_pos_), M * sizeof(int64_t)));
      DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_pre_req_), M * sizeof(int32_t)));
      DGPP_CUDA_OK(cudaMemset(d_pre_req_, 0, M * sizeof(int32_t)));
      DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_pre_spans_), 2 * sizeof(int32_t)));
      DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_pre_ctx_), 4 * sizeof(int32_t)));
      DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_pre_tokens_), M * sizeof(int64_t), cudaHostAllocDefault));
      DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_pre_pos_), M * sizeof(int64_t), cudaHostAllocDefault));
      DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_pre_ctx_), 8 * sizeof(int32_t), cudaHostAllocDefault));
      gather_thread_ = std::thread([this] { gather_loop(); });
    }
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
    if (gather_thread_.joinable()) {
      gather_stop_.store(true, std::memory_order_release);
      gather_thread_.join();
    }
    if (side_) cudaStreamSynchronize(side_);
    cudaFreeHost(h_ids_);
    cudaFreeHost(staged_);
    if (fork_) cudaEventDestroy(fork_);
    if (join_) cudaEventDestroy(join_);
    if (side_) cudaStreamDestroy(side_);
    if (d_hash_seq_) cudaFree(d_hash_seq_);
    if (d_wait_seq_) cudaFree(d_wait_seq_);
    if (h_hash_seq_) cudaFreeHost(h_hash_seq_);
    if (h_hash_rows_) cudaFreeHost(h_hash_rows_);
    if (h_done_seq_) cudaFreeHost(h_done_seq_);
    if (h_late_) cudaFreeHost(h_late_);
    if (h_pre_ids_) cudaFreeHost(h_pre_ids_);
    if (pre_staged_) cudaFreeHost(pre_staged_);
    if (d_pre_seq_) cudaFree(d_pre_seq_);
    if (d_pre_wait_seq_) cudaFree(d_pre_wait_seq_);
    if (h_pre_seq_) cudaFreeHost(h_pre_seq_);
    if (h_pre_rows_) cudaFreeHost(h_pre_rows_);
    if (h_pre_done_seq_) cudaFreeHost(h_pre_done_seq_);
    if (h_pre_need_) cudaFreeHost(h_pre_need_);
    if (d_pre_tokens_) cudaFree(d_pre_tokens_);
    if (d_pre_pos_) cudaFree(d_pre_pos_);
    if (d_pre_req_) cudaFree(d_pre_req_);
    if (d_pre_spans_) cudaFree(d_pre_spans_);
    if (d_pre_ctx_) cudaFree(d_pre_ctx_);
    if (h_pre_tokens_) cudaFreeHost(h_pre_tokens_);
    if (h_pre_pos_) cudaFreeHost(h_pre_pos_);
    if (h_pre_ctx_) cudaFreeHost(h_pre_ctx_);
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

// DGPP_QWEN_PLE_STAGE_TIMING=1: the host gather's duration, logged every
// 200 callbacks (mean / max us) — the PLE layer's host node left the GPU
// idle 772 us per depth-3 pass in the 2026-09-29 profile.
namespace {
bool ple_stage_timing() {
  static const bool on = [] {
    const char* v = std::getenv("DGPP_QWEN_PLE_STAGE_TIMING");
    return v && *v && std::string(v) != "0";
  }();
  return on;
}
}  // namespace

void QwenPleLayer::stage_callback(void* user) {
  StageArgs* a = static_cast<StageArgs*>(user);
  const bool timing = ple_stage_timing();
  const auto t0 = timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  try {
    a->table->gather(a->ids, a->n, a->heads, a->head_begin, a->heads_local, a->dst);
    if (timing) {
      static std::atomic<long long> calls{0}, total_us{0}, max_us{0};
      const long long us = std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::steady_clock::now() - t0).count();
      const long long n = calls.fetch_add(1) + 1;
      total_us.fetch_add(us);
      long long m = max_us.load();
      while (us > m && !max_us.compare_exchange_weak(m, us)) {}
      if (n % 200 == 0)
        DGPP_LOG_INFO("qwen ple: n-gram host gather x{} — {} rows: mean {} us, max {} us", n, a->n * a->heads_local,
                      total_us.load() / n, max_us.load());
    }
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
  if (gather_error_.load(std::memory_order_acquire))
    throw std::runtime_error("QwenPleLayer: the n-gram gather thread failed: " + gather_what_);
  if (h_late_ && *static_cast<volatile uint32_t*>(h_late_) != 0u)
    throw std::runtime_error("QwenPleLayer: the gather kernel waited past its timeout for the host's rows");
}

// The gather thread: polls the pinned sequence the hash kernel's publish
// raises (5 us sleeps: the answer must land within the two layers the GPU
// runs before the gather kernel), gathers the published rows' n-gram
// vectors into the staging, answers with the same sequence.
void QwenPleLayer::gather_loop() {
  uint64_t seen = 0, seen_pre = 0;
  auto fail = [this](const std::exception& e) {
    gather_what_ = e.what();
    gather_error_.store(1, std::memory_order_release);
    DGPP_LOG_ERROR("QwenPleLayer: the n-gram gather failed: {}", e.what());
  };
  auto serve_main = [&]() -> bool {
    const uint64_t v = __atomic_load_n(h_hash_seq_, __ATOMIC_ACQUIRE);
    if (v == seen) return false;
    seen = v;
    const int rows = __atomic_load_n(h_hash_rows_, __ATOMIC_ACQUIRE);
    try {
      if (table_.mmap) table_.mmap->gather(h_ids_, rows, heads_, w_.hash_head_begin, w_.hash_heads, staged_);
    } catch (const std::exception& e) {
      fail(e);
    }
    __atomic_store_n(h_done_seq_, v, __ATOMIC_RELEASE);
    return true;
  };
  while (!gather_stop_.load(std::memory_order_acquire)) {
    const bool served = serve_main();
    const uint64_t vp = h_pre_seq_ ? __atomic_load_n(h_pre_seq_, __ATOMIC_ACQUIRE) : seen_pre;
    if (vp != seen_pre) {
      // The prestage: gathered in pieces, the main channel served between
      // them (a decode step's rows between two chunks must not wait out a
      // whole chunk's gather).
      seen_pre = vp;
      const int rows = __atomic_load_n(h_pre_rows_, __ATOMIC_ACQUIRE);
      constexpr int kPiece = 512;
      const size_t row_ids = static_cast<size_t>(heads_);
      const size_t row_bytes = static_cast<size_t>(w_.hash_heads) * head_dim_;
      try {
        for (int r0 = 0; r0 < rows; r0 += kPiece) {
          if (gather_stop_.load(std::memory_order_acquire)) break;
          serve_main();
          const int n = std::min(kPiece, rows - r0);
          if (table_.mmap)
            table_.mmap->gather(h_pre_ids_ + r0 * row_ids, n, heads_, w_.hash_head_begin, w_.hash_heads,
                                pre_staged_ + r0 * row_bytes);
        }
      } catch (const std::exception& e) {
        fail(e);
      }
      __atomic_store_n(h_pre_done_seq_, vp, __ATOMIC_RELEASE);
      continue;
    }
    if (!served) {
      timespec ts{0, 5000};
      nanosleep(&ts, nullptr);
    }
  }
}

size_t QwenPleLayer::staging_bytes(const QwenTextConfig& cfg, int hash_heads, int max_tokens) {
  const size_t M = static_cast<size_t>(std::max(max_tokens, 0));
  const size_t heads = static_cast<size_t>((cfg.ngram_size - 1) * cfg.heads_per_ngram);
  const size_t hd = static_cast<size_t>(cfg.ple_embed_dim) / std::max<size_t>(heads, 1);
  return M * heads * 4 + M * static_cast<size_t>(hash_heads) * hd;
}

void QwenPleLayer::stage(const int64_t* tokens, int rows, const int32_t* req_ids, const int64_t* pos,
                         const int32_t* req_spans, int num_requests, const int32_t* ctx,
                         cudaStream_t stream, const int64_t* host_ids, int64_t pos0, int req) {
  if (!staged()) throw std::logic_error("QwenPleLayer: stage() without the mmap'ed table");
  if (rows <= 0) return;
  if (rows > max_tokens_) throw std::invalid_argument("QwenPleLayer: rows exceed max_tokens");
  if (staged_rows_ != 0) throw std::logic_error("QwenPleLayer: stage() twice without embed()");
  check_staged();
  if (pre_pending_) {
    // The chunk a prestage gathered: its rows are on the prestage channel.
    pre_pending_ = false;
    // The key: the same request, rows and first position, the same host ids
    // pointer AND the same first and last token (a prompt vector's address
    // can be reused by a later request).
    if (host_ids != nullptr && host_ids == pre_host_ids_ && rows == pre_rows_ && pos0 == pre_pos0_ &&
        req == pre_req_ && host_ids[0] == h_pre_tokens_[0] && host_ids[rows - 1] == h_pre_tokens_[rows - 1]) {
      use_pre_ = true;
      staged_rows_ = rows;
      // DGPP_QWEN_PLE_PRESTAGE_CHECK=1: run the main staging too and compare
      // the two gathers at embed() (a bitwise self-check of the mechanism).
      static const bool check = [] {
        const char* e = std::getenv("DGPP_QWEN_PLE_PRESTAGE_CHECK");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
      }();
      if (!check) return;
      pre_check_ = true;
      staged_rows_ = 0;  // fall through to the main path's hash and publish
    }
  }
  qwen_ple_hash_ids_rows(tokens, rows, req_ids, pos, req_spans, num_requests, ctx, eos_, d_mult_,
                         d_vocab_, d_offset_, heads_, heads_per_ngram_, ids_, stream);
  if (!host_node_) {
    // The thread's turn: the walk's rows and a sequence to pinned memory
    // (the ids are already there), no host node, no side stream.
    qwen_ple_publish_stage(d_hash_seq_, h_hash_seq_, h_hash_rows_, rows, stream);
    staged_rows_ = rows;
    return;
  }
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

void QwenPleLayer::prestage(const int64_t* host_ids, int rows, int64_t pos0, int req, int32_t ctx_t1,
                            int32_t ctx_t2, cudaStream_t stream) {
  if (!staged() || host_node_ || host_ids == nullptr || rows <= 0 || rows > max_tokens_) return;
  if (!QwenLayerStream::ngram_prestage()) return;  // engine.ngram_prestage (default on)
  check_staged();
  // The previous prestage, if still unclaimed, is dropped: its gather ran
  // or runs to completion (the thread answers every publish) and nothing
  // waits on it. Its pinned inputs may still be read by that copy, so the
  // walk's own sync (the caller runs eagerly) has passed before this call
  // reuses them.
  pre_pending_ = false;
  std::memcpy(h_pre_tokens_, host_ids, static_cast<size_t>(rows) * sizeof(int64_t));
  for (int i = 0; i < rows; ++i) h_pre_pos_[i] = pos0 + i;
  h_pre_ctx_[0] = ctx_t1;
  h_pre_ctx_[1] = ctx_t2;
  h_pre_ctx_[2] = 0;
  h_pre_ctx_[3] = 0;
  h_pre_ctx_[4] = 0;
  h_pre_ctx_[5] = rows;  // the one span: start 0, len rows
  DGPP_CUDA_OK(cudaMemcpyAsync(d_pre_tokens_, h_pre_tokens_, static_cast<size_t>(rows) * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(d_pre_pos_, h_pre_pos_, static_cast<size_t>(rows) * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(d_pre_ctx_, h_pre_ctx_, 4 * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(d_pre_spans_, h_pre_ctx_ + 4, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
  qwen_ple_hash_ids_rows(d_pre_tokens_, rows, d_pre_req_, d_pre_pos_, d_pre_spans_, 1, d_pre_ctx_, eos_, d_mult_,
                         d_vocab_, d_offset_, heads_, heads_per_ngram_, pre_ids_, stream);
  qwen_ple_publish_stage(d_pre_seq_, h_pre_seq_, h_pre_rows_, rows, stream);
  ++pre_publishes_;
  pre_host_ids_ = host_ids;
  pre_rows_ = rows;
  pre_pos0_ = pos0;
  pre_req_ = req;
  pre_pending_ = true;
}

void QwenPleLayer::embed(const int64_t* tokens, int rows, const int32_t* req_ids, const int64_t* pos,
                         const int32_t* req_spans, int num_requests, const int32_t* ctx,
                         cudaStream_t stream) {
  if (rows <= 0) return;
  if (rows > max_tokens_) throw std::invalid_argument("QwenPleLayer: rows exceed max_tokens");
  if (staged()) {
    if (staged_rows_ != rows) throw std::logic_error("QwenPleLayer: embed() without a matching stage()");
    if (host_node_) {
      DGPP_CUDA_OK(cudaStreamWaitEvent(stream, join_, 0));
    } else if (use_pre_) {
      // The prestage channel's answer: the wait word set to the publish
      // count less one (a dropped prestage leaves no wait behind), then
      // the same wait kernel; the rows from the prestage staging.
      use_pre_ = false;
      *h_pre_need_ = pre_publishes_ - 1;
      DGPP_CUDA_OK(cudaMemcpyAsync(d_pre_wait_seq_, h_pre_need_, sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
      glm_stage_wait(h_pre_done_seq_, d_pre_wait_seq_, h_late_, /*timeout_ns=*/int64_t{120} * 1000000000, stream);
      if (pre_check_) {
        // The self-check: the main channel's gather as well, then the two
        // stagings compared on the host (the stream synced here).
        pre_check_ = false;
        glm_stage_wait(h_done_seq_, d_wait_seq_, h_late_, /*timeout_ns=*/int64_t{120} * 1000000000, stream);
        DGPP_CUDA_OK(cudaStreamSynchronize(stream));
        const size_t row_bytes = static_cast<size_t>(w_.hash_heads) * head_dim_;
        size_t bad_rows = 0, first_bad = static_cast<size_t>(-1);
        for (size_t t = 0; t < static_cast<size_t>(rows); ++t)
          if (std::memcmp(staged_ + t * row_bytes, pre_staged_ + t * row_bytes, row_bytes) != 0) {
            ++bad_rows;
            if (first_bad == static_cast<size_t>(-1)) first_bad = t;
          }
        size_t bad_ids = 0;
        for (size_t i = 0; i < static_cast<size_t>(rows) * heads_; ++i)
          if (h_ids_[i] != h_pre_ids_[i]) ++bad_ids;
        DGPP_LOG_INFO("QwenPleLayer: prestage self-check: rows {} — staging rows differing {} (first {}), hash ids differing {}",
                      rows, bad_rows, first_bad == static_cast<size_t>(-1) ? -1 : static_cast<long long>(first_bad), bad_ids);
      }
      qwen_ple_gather_staged_bf16(pre_staged_, table_.scale, rows, w_.hash_heads, head_dim_, e_, stream);
      staged_rows_ = 0;
      return;
    } else {
      // The device waits for the thread's answer (a 2 us poll; a walk's
      // rows arrive within a few hundred microseconds of the publish).
      glm_stage_wait(h_done_seq_, d_wait_seq_, h_late_, /*timeout_ns=*/int64_t{120} * 1000000000, stream);
    }
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
