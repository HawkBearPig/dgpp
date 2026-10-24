#include "models/qwen/moe_layer.hpp"

#include <string>

#include <cstdlib>

#include "kernels/fp8_dequant.hpp"
#include "kernels/scale_gemm.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/qwen_moe.hpp"

namespace dgpp {

GlmMoeConfig QwenMoeLayer::routed_config(int hidden, int inter, int n_experts,
                                         int top_k, bool norm_topk_prob) {
  GlmMoeConfig c;
  c.hidden = hidden;
  c.inter = inter;
  c.n_experts = n_experts;
  c.top_k = top_k;
  c.n_shared_experts = 0;
  c.routed_scaling_factor = 1.0f;
  c.norm_topk_prob = norm_topk_prob;
  c.swiglu_limit = std::numeric_limits<float>::infinity();
  c.router_mode = MoeRouterMode::SoftmaxTopk;
  GlmMoeConfig::validate_config(c);
  return c;
}

GlmMoeWeights QwenMoeLayer::routed_view(const QwenMoeWeights& w) {
  GlmMoeWeights g;
  g.router_gate = w.router;
  g.router_bias = w.router_bias;
  g.experts = w.experts;
  g.experts_fp4 = w.experts_fp4;
  g.routed_fp4_mask = w.routed_fp4_mask;
  g.experts_packed = w.experts_packed;
  g.act_scale_w13 = w.act_scale_w13;
  g.act_scales_dev = w.act_scales_dev;
  g.act_scale_w2 = w.act_scale_w2;
  g.weight_only_fp4 = w.experts_weight_only;
  return g;
}

// The shared expert's fp8 scale grid as the kernels' log2 block sizes:
// 7 / 7 for the 128 x 128 blocks (engine.dense_weights = fp8, the AutoRound
// hybrid), 0 / 5 for the QAD export's MXFP8. The three matrices share it.
void QwenMoeLayer::shared_grid(int& rs, int& cs) const {
  rs = 7;
  cs = 7;
  if (!w_.shared_fp8) return;
  if (!quant_scale_shifts(w_.shared_fp8[0], rs, cs))
    throw std::invalid_argument("QwenMoeLayer: the shared expert's scale grid is not a power-of-two block size");
}
bool QwenMoeLayer::shared_block128() const {
  return w_.shared_fp8 && w_.shared_fp8[0].scale_block_rows == 128 && w_.shared_fp8[0].scale_block_cols == 128;
}

QwenMoeLayer::QwenMoeLayer(const QwenMoeWeights& weights, const GlmMoeConfig& cfg,
                           IGemm& gemm, int max_tokens, int decode_slots,
                           int graph_table_slots)
    : w_(weights), cfg_(cfg), gemm_(gemm), max_tokens_(max_tokens),
      routed_(routed_view(weights), cfg, max_tokens, decode_slots, graph_table_slots) {
  // Measured 2026-09-29 (serve_round4): the gate/up half beside the routed
  // experts takes 35 us there instead of 21 on the chain and the pass does
  // not move — the experts are DRAM-bound and the shared bytes cost the
  // same DRAM time wherever they run. Off by default; DGPP_QWEN_SHARED_SIDE=on.
  if (const char* v = std::getenv("DGPP_QWEN_SHARED_SIDE"); v && std::string(v) == "on") {
    int least = 0, greatest = 0;
    DGPP_CUDA_OK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    DGPP_CUDA_OK(cudaStreamCreateWithPriority(&shared_side_, cudaStreamNonBlocking, greatest));
    DGPP_CUDA_OK(cudaEventCreateWithFlags(&shared_fork_, cudaEventDisableTiming));
    DGPP_CUDA_OK(cudaEventCreateWithFlags(&shared_join_, cudaEventDisableTiming));
  }
  // The routed config: routed_config()'s (the softmax router) or Flint's
  // (the sigmoid router with its zero bias, FlintTextConfig::moe_config);
  // the shared expert never in the chain.
  if (cfg_.n_shared_experts != 0 ||
      (cfg_.router_mode != MoeRouterMode::SoftmaxTopk && cfg_.router_mode != MoeRouterMode::SigmoidBias))
    throw std::invalid_argument(
        "QwenMoeLayer: the routed config must be routed_config()'s (softmax "
        "router, no shared expert in the chain) or a sigmoid router's");
  check_weights();
  const size_t M = static_cast<size_t>(max_tokens_);
  const size_t H = static_cast<size_t>(cfg_.hidden);
  const size_t S = static_cast<size_t>(w_.shared_inter);
  DGPP_CUDA_OK(cudaMalloc(&d_acc_, M * H * sizeof(float)));
  DGPP_CUDA_OK(cudaMalloc(&d_sgate_, M * S * 2));
  DGPP_CUDA_OK(cudaMalloc(&d_sup_, M * S * 2));
  DGPP_CUDA_OK(cudaMalloc(&d_sact_, M * S * 2));
  DGPP_CUDA_OK(cudaMalloc(&d_sdown_, M * H * sizeof(float)));
  DGPP_CUDA_OK(cudaMalloc(&d_sw_, M * sizeof(float)));
  DGPP_CUDA_OK(cudaMalloc(&d_rows_, M * sizeof(int32_t)));
  std::vector<int32_t> identity(M);
  for (size_t i = 0; i < M; ++i) identity[i] = static_cast<int32_t>(i);
  DGPP_CUDA_OK(cudaMemcpy(d_rows_, identity.data(), M * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
  // The GEMM interface's workspace for the three shared-expert shapes.
  const int Hn = cfg_.hidden, Sn = static_cast<int>(w_.shared_inter);
  gemm_ws_bytes_ = std::max({gemm_.query_workspace_bytes(max_tokens_, Sn, Hn, DType::BF16),
                             gemm_.query_workspace_bytes(max_tokens_, Hn, Sn, DType::BF16),
                             static_cast<size_t>(1) << 20});
  DGPP_CUDA_OK(cudaMalloc(&gemm_ws_, gemm_ws_bytes_));
  if (w_.shared_fp8) DGPP_CUDA_OK(cudaMalloc(&d_shared_bridge_, S * H * 2));
}

QwenMoeLayer::~QwenMoeLayer() {
  if (shared_join_) cudaEventDestroy(shared_join_);
  if (shared_fork_) cudaEventDestroy(shared_fork_);
  if (shared_side_) cudaStreamDestroy(shared_side_);
  if (d_shared_bridge_) cudaFree(d_shared_bridge_);
  cudaFree(d_acc_);
  cudaFree(d_sgate_);
  cudaFree(d_sup_);
  cudaFree(d_sact_);
  cudaFree(d_sdown_);
  cudaFree(d_sw_);
  cudaFree(d_rows_);
  cudaFree(gemm_ws_);
}

void QwenMoeLayer::check_weights() const {
  const bool shared_ok = w_.shared_fp8 ? (w_.shared_fp8[0].payload && w_.shared_fp8[1].payload && w_.shared_fp8[2].payload)
                                       : (w_.shared_gate_proj && w_.shared_up_proj && w_.shared_down_proj);
  const int forms = (w_.experts != nullptr) + (w_.experts_fp4 != nullptr) + (w_.experts_packed != nullptr);
  // shared_gate null: the ungated shared expert at weight 1 (Flint).
  if (!w_.router || !shared_ok || forms == 0)
    throw std::invalid_argument("QwenMoeLayer: null weight pointer");
  const bool mixed = forms == 2 && w_.experts && w_.experts_fp4 && w_.routed_fp4_mask != 0;
  if (forms != 1 && !mixed)
    throw std::invalid_argument(
        "QwenMoeLayer: exactly one routed expert form (fp8, nvfp4, packed) may be bound, or the fp8 and NVFP4 "
        "tables with a per-projection mask");
  if (w_.shared_inter <= 0 || w_.shared_inter % 8 != 0)
    throw std::invalid_argument("QwenMoeLayer: shared_inter must be a positive multiple of 8");
  if (cfg_.hidden % 8 != 0)
    throw std::invalid_argument("QwenMoeLayer: hidden must be a multiple of 8");
}

void QwenMoeLayer::rebind(const QwenMoeWeights& w) {
  if (w.shared_inter != w_.shared_inter)
    throw std::invalid_argument("QwenMoeLayer: rebind changes the shared slice width");
  w_ = w;
  check_weights();
  routed_.rebind(routed_view(w_));
}

void QwenMoeLayer::enqueue(const uint16_t* hidden, uint16_t* out, int tokens,
                           cudaStream_t stream, MoeExpertKernel kernel) {
  if (tokens <= 0) return;
  if (tokens > max_tokens_)
    throw std::invalid_argument("QwenMoeLayer: tokens exceed max_tokens");
  if (!hidden || !out) throw std::invalid_argument("QwenMoeLayer: null pointer");

  // 1. The routed experts: the fp32 chain in ascending expert order, left
  //    unrounded in d_acc_.
  routed_.enqueue_f32(hidden, d_acc_, tokens, stream, kernel);
  shared_tail(hidden, out, tokens, stream);
}

void QwenMoeLayer::enqueue_decode(const uint16_t* hidden, uint16_t* out, int tokens,
                                  cudaStream_t stream, int table_slot, MoeTraceStaging* trace) {
  if (tokens <= 0) return;
  if (tokens > max_tokens_)
    throw std::invalid_argument("QwenMoeLayer: tokens exceed max_tokens");
  if (!hidden || !out) throw std::invalid_argument("QwenMoeLayer: null pointer");
  const int Hs = cfg_.hidden;
  const int Ss = static_cast<int>(w_.shared_inter);
  int rs = 7, cs = 7;
  shared_grid(rs, cs);
  if (shared_side_ != nullptr && tokens <= 8) {
    // The shared gate/up beside the routed experts: forked here, joined
    // before the down half below (events; graph edges under capture).
    DGPP_CUDA_OK(cudaEventRecord(shared_fork_, stream));
    DGPP_CUDA_OK(cudaStreamWaitEvent(shared_side_, shared_fork_, 0));
    if (w_.shared_fp8)
      qwen_moe_shared_gate_up_decode_fp8(hidden, static_cast<size_t>(Hs), w_.shared_fp8[0].payload,
                                         w_.shared_fp8[0].scales, w_.shared_fp8[1].payload, w_.shared_fp8[1].scales,
                                         w_.shared_gate, d_sact_, d_sw_, tokens, Hs, Ss, shared_side_, rs, cs);
    else
      qwen_moe_shared_gate_up_decode(hidden, static_cast<size_t>(Hs), w_.shared_gate_proj, w_.shared_up_proj,
                                     w_.shared_gate, d_sact_, d_sw_, tokens, Hs, Ss, shared_side_);
  }
  // The fused tails (tokens <= 8) fold the routed accumulation into the
  // shared down's epilogue (kernels/qwen_moe: bitwise the accumulate + tail
  // chain, one launch fewer per layer, 2026-09-29); the wide path and the
  // side-stream path take the accumulate kernel's output.
  const bool fold = tokens <= 8 && shared_side_ == nullptr;
  routed_.enqueue_decode_f32(hidden, d_acc_, tokens, trace, stream, table_slot, /*accumulate=*/!fold);
  // Preserve the small-row fused tail, including C1 MTP. Wider batches
  // use the dense interface (Lt for BF16, streaming MMA for FP8) so the
  // shared weights are not re-read by another four-row GEMV per chunk.
  if (tokens > 8) {
    shared_tail(hidden, out, tokens, stream);
    return;
  }
  // The fused two-launch tail (bitwise the chain; qwen_moe_test pins it),
  // in the weights' form: BF16, or block FP8. With the side stream the
  // gate/up half ran beside the routed experts (forked above), and only
  // the down half is left on the chain.
  const int H = cfg_.hidden;
  const int S = static_cast<int>(w_.shared_inter);
  if (shared_side_ != nullptr) {
    DGPP_CUDA_OK(cudaEventRecord(shared_join_, shared_side_));
    DGPP_CUDA_OK(cudaStreamWaitEvent(stream, shared_join_, 0));
    if (w_.shared_fp8)
      qwen_moe_shared_down_decode_fp8(d_sact_, w_.shared_fp8[2].payload, w_.shared_fp8[2].scales, d_sw_, d_acc_, out,
                                      tokens, H, S, stream, rs, cs);
    else
      qwen_moe_shared_down_decode(d_sact_, w_.shared_down_proj, d_sw_, d_acc_, out, tokens, H, S, stream);
    return;
  }
  const float* contrib = routed_.decode_slot_down();
  const float* weights = routed_.decode_weights();
  const int top_k = routed_.config().top_k;
  if (w_.shared_fp8) {
    qwen_moe_shared_tail_decode_routed_fp8(hidden, static_cast<size_t>(H), w_.shared_fp8[0].payload,
                                           w_.shared_fp8[0].scales, w_.shared_fp8[1].payload, w_.shared_fp8[1].scales,
                                           w_.shared_fp8[2].payload, w_.shared_fp8[2].scales, w_.shared_gate, d_sact_,
                                           d_sw_, contrib, weights, top_k, out, tokens, H, S, stream, rs, cs);
    return;
  }
  qwen_moe_shared_tail_decode_routed(hidden, static_cast<size_t>(H), w_.shared_gate_proj, w_.shared_up_proj,
                                     w_.shared_down_proj, w_.shared_gate, d_sact_, d_sw_, contrib, weights, top_k, out,
                                     tokens, H, S, stream);
}

void QwenMoeLayer::enqueue_prefill(const uint16_t* hidden, uint16_t* out, int tokens,
                                   cudaStream_t stream, MoeTraceStaging* trace) {
  if (tokens <= 0) return;
  if (tokens > max_tokens_)
    throw std::invalid_argument("QwenMoeLayer: tokens exceed max_tokens");
  if (!hidden || !out) throw std::invalid_argument("QwenMoeLayer: null pointer");
  // The tensor-core chain on the grids it takes (the real slices: 160 on a
  // 32 grid at TP=4, 320 on 64 at TP=2, 640 on 128 at TP=1); the GEMV core
  // on any other (a small fixture's 16-wide slice).
  routed_.enqueue_prefill_f32(hidden, d_acc_, tokens, trace, stream,
                              routed_.mma_takes_grid() ? MoeExpertKernel::kMma
                                                       : MoeExpertKernel::kGemv);
  shared_tail(hidden, out, tokens, stream, true);
}

void QwenMoeLayer::shared_tail(const uint16_t* hidden, uint16_t* out, int tokens,
                               cudaStream_t stream, bool prefill) {
  const int H = cfg_.hidden;
  const int S = static_cast<int>(w_.shared_inter);
  // 2. The BF16 shared expert: gate/up (bf16 out), silu * up with the
  //    chain's two roundings and no clamps, the down projection unrounded.
  int rs = 7, cs = 7;
  shared_grid(rs, cs);
  const bool block128 = shared_block128();
  const bool tile = prefill && prefill_dense_tile_ && !block128 && H % 64 == 0 && S % 64 == 0;
  if (w_.shared_fp8 && tokens > 128 && !tile) {
    // Prefill-shaped: each FP8 matrix dequantized into the bridge (the
    // GEMV core's values) and run through the BF16 interface.
    launch_fp8_dequant_blocks(w_.shared_fp8[0].payload, w_.shared_fp8[0].scales, d_shared_bridge_, S, H, stream, rs, cs);
    gemm_.matmul(hidden, d_shared_bridge_, d_sgate_, tokens, S, H, DType::BF16,
                 GemmOut::BF16, static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream);
    launch_fp8_dequant_blocks(w_.shared_fp8[1].payload, w_.shared_fp8[1].scales, d_shared_bridge_, S, H, stream, rs, cs);
    gemm_.matmul(hidden, d_shared_bridge_, d_sup_, tokens, S, H, DType::BF16,
                 GemmOut::BF16, static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream);
  } else if (w_.shared_fp8 && block128) {
    // The FP8 form (engine.dense_weights): the scale GEMM's chunked GEMV at
    // decode rows; the same roundings.
    launch_scale_gemm_bf16(hidden, static_cast<size_t>(H), w_.shared_fp8[0].payload, w_.shared_fp8[0].scales,
                           d_sgate_, tokens, S, H, stream, static_cast<size_t>(S), mma_from_rows_, gemm_ws_,
                           gemm_ws_bytes_);
    launch_scale_gemm_bf16(hidden, static_cast<size_t>(H), w_.shared_fp8[1].payload, w_.shared_fp8[1].scales,
                           d_sup_, tokens, S, H, stream, static_cast<size_t>(S), mma_from_rows_, gemm_ws_,
                           gemm_ws_bytes_);
  } else if (w_.shared_fp8) {
    // The MXFP8 form (the QAD export's shipped shared experts): the grid
    // launcher on the 1 x 32 grid — the streaming MMA from mma_from_rows_
    // (the dense sites' rule), the GEMV rows below it.
    const bool mma = mma_from_rows_ > 0 && tokens >= mma_from_rows_;
    launch_scale_gemm_grid_bf16(hidden, static_cast<size_t>(H), w_.shared_fp8[0].payload, w_.shared_fp8[0].scales,
                                d_sgate_, tokens, S, H, stream, static_cast<size_t>(S), rs, cs, mma, gemm_ws_,
                                gemm_ws_bytes_, tile);
    launch_scale_gemm_grid_bf16(hidden, static_cast<size_t>(H), w_.shared_fp8[1].payload, w_.shared_fp8[1].scales,
                                d_sup_, tokens, S, H, stream, static_cast<size_t>(S), rs, cs, mma, gemm_ws_,
                                gemm_ws_bytes_, tile);
  } else {
    gemm_.matmul(hidden, w_.shared_gate_proj, d_sgate_, tokens, S, H, DType::BF16,
                 GemmOut::BF16, static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream);
    gemm_.matmul(hidden, w_.shared_up_proj, d_sup_, tokens, S, H, DType::BF16,
                 GemmOut::BF16, static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream);
  }
  launch_moe_swiglu_clamp(d_sgate_, d_sup_, d_sact_,
                          static_cast<int64_t>(tokens) * S,
                          std::numeric_limits<float>::infinity(), stream);
  if (w_.shared_fp8 && tokens > 128 && !tile) {
    launch_fp8_dequant_blocks(w_.shared_fp8[2].payload, w_.shared_fp8[2].scales, d_shared_bridge_, H, S, stream, rs, cs);
    gemm_.matmul(d_sact_, d_shared_bridge_, d_sdown_, tokens, H, S, DType::BF16,
                 GemmOut::F32, static_cast<size_t>(S), gemm_ws_, gemm_ws_bytes_, stream);
  } else if (w_.shared_fp8 && block128)
    launch_scale_gemm_f32(d_sact_, static_cast<size_t>(S), w_.shared_fp8[2].payload, w_.shared_fp8[2].scales,
                          d_sdown_, tokens, H, S, stream, static_cast<size_t>(H), mma_from_rows_,
                          /*last_row_only=*/false, gemm_ws_, gemm_ws_bytes_);
  else if (w_.shared_fp8)
    launch_scale_gemm_grid_f32(d_sact_, static_cast<size_t>(S), w_.shared_fp8[2].payload, w_.shared_fp8[2].scales,
                               d_sdown_, tokens, H, S, stream, static_cast<size_t>(H), rs, cs,
                               mma_from_rows_ > 0 && tokens >= mma_from_rows_, gemm_ws_, gemm_ws_bytes_, tile);
  else
    gemm_.matmul(d_sact_, w_.shared_down_proj, d_sdown_, tokens, H, S, DType::BF16,
                 GemmOut::F32, static_cast<size_t>(S), gemm_ws_, gemm_ws_bytes_, stream);

  // 3. Its weight, sigma(x . g) — bf16 as the reference's — then the
  //    chain's last fma and its one rounding.
  qwen_moe_shared_gate_bf16(hidden, w_.shared_gate, d_sw_, tokens, H, stream);
  launch_moe_accum(d_acc_, d_sdown_, d_rows_, d_sw_, tokens, H, stream);
  launch_moe_round_bf16(out, d_acc_, static_cast<int64_t>(tokens) * H, stream);
}

size_t QwenMoeLayer::scratch_bytes(const GlmMoeConfig& cfg, int64_t shared_inter,
                                   int max_tokens, size_t* pinned_bytes, int decode_slots,
                                   int graph_table_slots) {
  const size_t M = static_cast<size_t>(std::max(max_tokens, 0));
  const size_t H = static_cast<size_t>(cfg.hidden);
  const size_t S = static_cast<size_t>(std::max<int64_t>(shared_inter, 0));
  size_t dev = GlmMoeLayer::scratch_bytes(cfg, max_tokens, decode_slots, graph_table_slots, pinned_bytes);
  dev += M * H * sizeof(float) * 2;  // d_acc_, d_sdown_
  dev += M * S * 2 * 3;              // d_sgate_, d_sup_, d_sact_
  dev += M * sizeof(float);          // d_sw_
  dev += M * sizeof(int32_t);        // d_rows_
  dev += static_cast<size_t>(1) << 20;  // the GEMM workspace floor
  return dev;
}

}  // namespace dgpp
