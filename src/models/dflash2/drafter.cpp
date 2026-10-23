// The DFlash2 block drafter's shared execution: the feature store, the
// block forward and the mask-row select, the eager (re)drafts — scalar and
// stacked-batch — and the recorded block proposal the decode graph
// captures. One implementation for every host family (the Qwen35 and the
// Qwen3.8-Flash-Next models host it, models/qwen): the family supplies
// DFlash2Host (the pool's block table and draft planes, the session and
// graph state, the embeddings, the head dispatch and the block GEMMs) and
// taps its own layer outputs into the feature accumulator (acc /
// tap_gemm_out) at its forward's tap sites.
#include "models/dflash2/drafter.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "engine/decode_outputs.hpp"
#include "kernels/dflash2.hpp"
#include "kernels/gemm.hpp"
#include "kernels/kernels.hpp"
#include "kernels/mma_gemv.hpp"
#include "kernels/qsa.hpp"
#include "kernels/rope_scaling.hpp"

namespace dgpp {

namespace {
// The stacked draft batch's width: one verify batch's slots at the row
// ceiling (max_decode_rows = max(kDecodeRows, decode_rows, max_requests)).
size_t batch_width(const DFlash2Config& cfg, int decode_rows, int max_requests) {
  return std::max<size_t>(1,
                          static_cast<size_t>(std::max({kDecodeRows, decode_rows, max_requests})) /
                              static_cast<size_t>(cfg.query_rows()));
}
}  // namespace

void DFlash2Drafter::init(const DFlash2Host& h, const std::string& dir) {
  if (on_) throw std::logic_error("DFlash2Drafter: init called twice");
  // The drafter's own architecture string decides (never a guess): the
  // block drafter takes the DFlash2 v2 draft only.
  const std::string arch = drafter_architecture(dir);
  if (arch != "DFlash2DraftModel")
    throw std::invalid_argument("DFlash2: the block drafter needs the DFlash2DraftModel architecture (got '" +
                                arch + "')");
  cfg_ = DFlash2Config::from_json_file(dir + "/config.json");
  cfg_.validate_against(DFlash2TargetShape{
      h.hidden, h.target_vocab, h.num_target_layers,
      h.position_limit > 0 ? h.position_limit : int64_t{h.max_context}});
  if (!cfg_.tp_divisible(h.world))
    throw std::invalid_argument(
        "DFlash2: the drafter's heads / kv heads / MLP rows must divide across the world");
  h_ = h;
  tap_.assign(h.num_target_layers + 1, -1);
  for (size_t t = 0; t < cfg_.target_layer_ids.size(); ++t)
    tap_[cfg_.target_layer_ids[t]] = static_cast<int>(t);
  df_nh_ = cfg_.local_heads(h.world);
  df_kvh_ = cfg_.local_kv_heads(h.world);
  df_qw_ = cfg_.local_q_row(h.world);
  df_kvr_ = cfg_.local_kv_row(h.world);
  df_i_ = cfg_.local_intermediate(h.world);
  w_ = load_dflash2_weights(cfg_, dir, h.stream, h.rank, h.world, h.weights_fp8);
  // The draft's rope table: plain at the draft's own theta, or riding the
  // target's YaRN ramp (the same wavelengths at the draft's band — the
  // drafter rotates its full 128-dim head, so its band sits at lanes
  // 33..48 where a rotary-64 target's sits at 16..24). The draft kernel
  // takes no mscale (the ramp's attention factor rides the target's
  // rotation only).
  {
    std::vector<float> host(static_cast<size_t>(cfg_.head_dim) / 2);
    if (h.rope != nullptr) {
      h.rope->validate("rope_scaling");
      yarn_rope_inv_freq_host(cfg_.head_dim, cfg_.rope_theta, h.rope->correction_max_position(),
                              h.rope->factor, h.rope->beta_fast, h.rope->beta_slow, host.data());
    } else {
      const double theta = cfg_.rope_theta;
      for (size_t i = 0; i < host.size(); ++i)
        host[i] = static_cast<float>(
            std::pow(theta, -static_cast<double>(2 * i) / static_cast<double>(cfg_.head_dim)));
    }
    DGPP_CUDA_OK(cudaMalloc(&df_inv_freq_, host.size() * 4));
    DGPP_CUDA_OK(cudaMemcpy(df_inv_freq_, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
  }
  // Feature path at rows_cap rows; the block forward at BR = df_batch_ *
  // query_rows stacked rows (the stacked draft batch widens the row-wise
  // scratch to one verify batch's worth at the row ceiling; the per-slot
  // calls — attention, head, top-K, walk — run at row offsets inside the
  // same buffers).
  const int CR = rows_cap();
  const int QR = cfg_.query_rows();
  const int D = QR - 1;  // mask rows = drafts
  df_batch_ = std::max(1, h.max_decode_rows / QR);
  const int BR = df_batch_ * QR, BD = df_batch_ * D;
  const size_t MT = static_cast<size_t>(h.max_tokens);
  const int H = h.hidden;
  DGPP_CUDA_OK(cudaMalloc(&df_acc_, MT * H * 4));
  DGPP_CUDA_OK(cudaMalloc(&df_t32_, MT * H * 4));
  DGPP_CUDA_OK(cudaMalloc(&df_norm_, static_cast<size_t>(CR) * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_kv_, static_cast<size_t>(CR) * 2 * df_kvr_ * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_resid_, static_cast<size_t>(BR) * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_x_, static_cast<size_t>(BR) * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_xc_, static_cast<size_t>(BR) * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_qkv_, static_cast<size_t>(BR) * (df_qw_ + 2 * df_kvr_) * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_q_, static_cast<size_t>(BR) * df_qw_ * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_attn_, static_cast<size_t>(BR) * df_qw_ * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_attn_part_, dflash2_block_attn_partials_bytes(BR, df_nh_)));
  DGPP_CUDA_OK(cudaMalloc(&df_o_, static_cast<size_t>(BR) * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_mlp_, static_cast<size_t>(BR) * df_i_ * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_gate_, static_cast<size_t>(BR) * df_i_ * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_up_, static_cast<size_t>(BR) * df_i_ * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_zero_, QR * 4));
  DGPP_CUDA_OK(cudaMemsetAsync(df_zero_, 0, QR * 4, h.stream));
  DGPP_CUDA_OK(cudaMalloc(&df_delta_, static_cast<size_t>(BR) * 2 * cfg_.conv_taps * cfg_.conv_groups() * 2));
  DGPP_CUDA_OK(cudaMalloc(&df_h_, static_cast<size_t>(BR) * H * 2));
  // The head rows, top-K and hidden projection run once over every
  // stacked row (the anchor rows' entries unused): one head stream a
  // step instead of one per slot.
  DGPP_CUDA_OK(cudaMalloc(&df_logits_, static_cast<size_t>(BR) * h.vocab_count * 4));
  df_topk_ws_bytes_ = dflash2_topk_ws_bytes(h.vocab_count, BR, cfg_.selector_top_k);
  DGPP_CUDA_OK(cudaMalloc(&df_topk_ws_, df_topk_ws_bytes_));
  DGPP_CUDA_OK(cudaMalloc(&df_hidden32_, static_cast<size_t>(BR) * cfg_.selector_rank * 4));
  DGPP_CUDA_OK(cudaMalloc(&df_ids_, static_cast<size_t>(BR) * cfg_.selector_top_k * 4));
  DGPP_CUDA_OK(cudaMalloc(&df_sc_, static_cast<size_t>(BR) * cfg_.selector_top_k * 4));
  DGPP_CUDA_OK(cudaMalloc(&df_tok_, static_cast<size_t>(BD) * 4));
  DGPP_CUDA_OK(cudaMalloc(&df_pos_, static_cast<size_t>(BR) * 8));
  DGPP_CUDA_OK(cudaMalloc(&df_tokens_, static_cast<size_t>(BR) * 8));
  DGPP_CUDA_OK(cudaMallocHost(&df_tok_h_, static_cast<size_t>(BD) * 4));
  DGPP_CUDA_OK(cudaMallocHost(&df_io64_h_, static_cast<size_t>(2) * BR * 8));
  // World > 1: the ranks' slice top-K lists meet in one boundary fold
  // (the eager fold runs in place on df_table_; a capture folds the
  // recorder's staged buffer). The recorded drafts' pinned mirror, one
  // row per request slot.
  if (h.world > 1)
    DGPP_CUDA_OK(cudaMalloc(&df_table_, dflash2_topk_table_elems(BR, cfg_.selector_top_k, h.world) * 2));
  DGPP_CUDA_OK(cudaMallocHost(&df_mirror_h_, static_cast<size_t>(h.max_requests) * D * 4));
  DGPP_CUDA_OK(cudaMallocHost(&df_cands_h_, static_cast<size_t>(h.max_requests) * D * cfg_.selector_top_k * 4));
  DGPP_CUDA_OK(cudaMalloc(&df_conf_, static_cast<size_t>(h.max_requests) * D * 4));
  DGPP_CUDA_OK(cudaMemset(df_conf_, 0, static_cast<size_t>(h.max_requests) * D * 4));
  DGPP_CUDA_OK(cudaStreamSynchronize(h.stream));
  on_ = true;
}

void DFlash2Drafter::free() {
  cudaFree(df_inv_freq_);
  cudaFree(df_acc_);
  cudaFree(df_t32_);
  cudaFree(df_norm_);
  cudaFree(df_kv_);
  cudaFree(df_resid_);
  cudaFree(df_x_);
  cudaFree(df_xc_);
  cudaFree(df_qkv_);
  cudaFree(df_q_);
  cudaFree(df_attn_);
  cudaFree(df_attn_part_);
  cudaFree(df_o_);
  cudaFree(df_mlp_);
  cudaFree(df_gate_);
  cudaFree(df_up_);
  cudaFree(df_zero_);
  cudaFree(df_delta_);
  cudaFree(df_h_);
  cudaFree(df_logits_);
  cudaFree(df_topk_ws_);
  cudaFree(df_hidden32_);
  cudaFree(df_ids_);
  cudaFree(df_sc_);
  cudaFree(df_tok_);
  cudaFree(df_pos_);
  cudaFree(df_tokens_);
  cudaFree(df_table_);
  cudaFree(df_conf_);
  cudaFreeHost(df_tok_h_);
  cudaFreeHost(df_io64_h_);
  cudaFreeHost(df_mirror_h_);
  cudaFreeHost(df_cands_h_);
  df_inv_freq_ = nullptr;
  df_acc_ = nullptr;
  df_t32_ = nullptr;
  df_norm_ = nullptr;
  df_kv_ = nullptr;
  df_resid_ = nullptr;
  df_x_ = nullptr;
  df_xc_ = nullptr;
  df_qkv_ = nullptr;
  df_q_ = nullptr;
  df_attn_ = nullptr;
  df_attn_part_ = nullptr;
  df_o_ = nullptr;
  df_mlp_ = nullptr;
  df_gate_ = nullptr;
  df_up_ = nullptr;
  df_zero_ = nullptr;
  df_delta_ = nullptr;
  df_h_ = nullptr;
  df_logits_ = nullptr;
  df_topk_ws_ = nullptr;
  df_topk_ws_bytes_ = 0;
  df_hidden32_ = nullptr;
  df_ids_ = nullptr;
  df_sc_ = nullptr;
  df_tok_ = nullptr;
  df_pos_ = nullptr;
  df_tokens_ = nullptr;
  df_table_ = nullptr;
  df_conf_ = nullptr;
  df_tok_h_ = nullptr;
  df_io64_h_ = nullptr;
  df_mirror_h_ = nullptr;
  df_cands_h_ = nullptr;
  w_ = DFlash2Weights();
  on_ = false;
  df_specs_ = nullptr;
  df_props_ = nullptr;
  df_props_h_ = nullptr;
  df_props_stride_ = 0;
}

int DFlash2Drafter::tap(int layer) const {
  if (layer < 0 || layer >= static_cast<int>(tap_.size())) return -1;
  return tap_[layer];
}

const int32_t* DFlash2Drafter::drafts_host(int req) const {
  return df_mirror_h_ + static_cast<size_t>(req) * static_cast<size_t>(cfg_.drafts());
}

const int32_t* DFlash2Drafter::candidates_host(int req) const {
  return df_cands_h_ + static_cast<size_t>(req) * static_cast<size_t>(cfg_.drafts()) *
                           static_cast<size_t>(cfg_.selector_top_k);
}

void DFlash2Drafter::arm_proposals(const SampleSpec* specs, DraftProposal* proposals,
                                   DraftProposal* proposals_host, int stride) {
  if (specs != nullptr && (proposals == nullptr || stride < cfg_.drafts()))
    throw std::invalid_argument("arm_proposals: proposals / stride");
  df_specs_ = specs;
  df_props_ = proposals;
  df_props_h_ = proposals_host;
  df_props_stride_ = stride;
}

void DFlash2Drafter::plan_entries(const DFlash2Config& cfg, int world, int decode_rows, int max_tokens,
                                  int max_requests, int64_t vocab_count, bool weights_fp8,
                                  std::vector<std::pair<std::string, size_t>>* out) {
  if (!cfg.tp_divisible(world))
    throw std::invalid_argument("DFlash2 plan: the drafter's heads / kv heads / MLP rows must divide across the world");
  const size_t H = cfg.hidden_size, I = cfg.local_intermediate(world);
  const size_t QR = static_cast<size_t>(cfg.query_rows()), D = QR - 1;
  const size_t BW = batch_width(cfg, decode_rows, max_requests);
  const size_t BR = BW * QR, BD = BW * D, CR = rows_cap();
  const size_t QW = cfg.local_q_row(world), KV = cfg.local_kv_row(world), NH = cfg.local_heads(world);
  const size_t M = static_cast<size_t>(max_tokens), V = static_cast<size_t>(vocab_count);
  const size_t G = cfg.conv_groups(), TAPS = cfg.conv_taps, K = cfg.selector_top_k, RANK = cfg.selector_rank;
  out->push_back(
      {weights_fp8 ? "dflash2 drafter weights (this rank's slices; the block matrices as fp8, the rest bf16)"
                   : "dflash2 drafter weights (this rank's slices)",
       dflash2_weights_bytes(cfg, world, weights_fp8)});
  // acc + tap F32, norm/kv scratch, rope table.
  out->push_back({"dflash2 feature path", 2 * M * H * 4 + CR * H * 2 + CR * 2 * KV * 2 +
                                              static_cast<size_t>(cfg.head_dim) / 2 * 4});
  // resid, x, xc, o, h; mlp, gate, up; qkv; q, attn; the conv deltas; the
  // block's positions and tokens; the zero requests; the split-key partials.
  out->push_back({"dflash2 block scratch",
                  BR * (5 * H * 2 + 3 * I * 2 + (QW + 2 * KV) * 2 + 2 * QW * 2 + 2 * TAPS * G * 2 + 16) +
                      QR * 4 + dflash2_block_attn_partials_bytes(static_cast<int>(BR), NH)});
  // The draft rows' head logits.
  out->push_back({"dflash2 head logits", BR * V * 4});
  // The top-K partials, the hidden projection, the ids/scores and the
  // proposals.
  out->push_back({"dflash2 candidate buffers",
                  dflash2_topk_ws_bytes(V, BR, K) + BR * (RANK * 4 + 2 * K * 4) + BD * 4});
  // The pinned readbacks and staging (the recorded drafts' mirrors, the
  // walked candidates) and the fold table at world > 1.
  out->push_back({"dflash2 pinned staging",
                  BD * 4 + 2 * BR * 8 +
                      (world > 1 ? dflash2_topk_table_elems(BR, K, world) * 2 : 0) +
                      static_cast<size_t>(max_requests) * D * 4 +
                      static_cast<size_t>(max_requests) * D * K * 4});
  // The selector's confidence per draft, [max_requests][drafts].
  out->push_back({"dflash2 confidence", static_cast<size_t>(max_requests) * D * 4});
}

// ---- the feature path ----------------------------------------------------------

void DFlash2Drafter::store_features(int T, const int32_t* d_req, const int64_t* d_pos) {
  const int H = h_.hidden;
  const int L = cfg_.num_hidden_layers, KV = df_kvr_;
  const int KVH = df_kvh_, HD = cfg_.head_dim;
  const int32_t* tables = h_.device_tables;
  const int bpr = h_.total_blocks;
  const int bt = h_.block_tokens;
  for (int off = 0; off < T; off += rows_cap()) {
    const int rows = std::min(rows_cap(), T - off);
    dflash2_norm_f32_bf16(df_acc_ + static_cast<size_t>(off) * H, w_.hidden_norm, df_norm_, rows, H,
                          cfg_.rms_norm_eps, h_.stream);
    for (int l = 0; l < L; ++l) {
      // k|v rows of the fused layer weight (a view right after q, or their
      // own bf16 slot under the fp8 block matrices).
      const uint16_t* kv_w = w_.layers[l].kv_rows;
      h_.gemm->matmul(df_norm_, kv_w, df_kv_, rows, 2 * KV, H, DType::BF16, GemmOut::BF16,
                      static_cast<size_t>(H), h_.gemm_ws, h_.gemm_ws_bytes, h_.stream);
      dflash2_norm_rope_bf16(df_kv_, 2 * KV, w_.layers[l].k_norm, d_pos + off, df_inv_freq_, df_kv_,
                             2 * KV, rows, KVH, HD, cfg_.rms_norm_eps, h_.stream);
      const auto plane = h_.plane(l);
      qsa_kv_append(df_kv_, 2 * KV, df_kv_ + KV, 2 * KV, d_req + off, d_pos + off, rows, tables, bpr,
                    bt, KVH, HD, plane.k, plane.v, nullptr, nullptr, h_.stream);
    }
  }
}

// ---- the block forward ----------------------------------------------------------

void DFlash2Drafter::block_layers(int slots, const int* reqs, bool capture) {
  const int H = h_.hidden;
  const int QR = cfg_.query_rows();
  const int L = cfg_.num_hidden_layers, QW = df_qw_, KV = df_kvr_;
  const int KVH = df_kvh_, HD = cfg_.head_dim, NH = df_nh_;
  const int G = cfg_.conv_groups(), TAPS = cfg_.conv_taps;
  const int I = df_i_;
  const float eps = cfg_.rms_norm_eps;
  const int R = slots * QR;
  // The sharded drafter's boundary folds (the host run's pattern): the o
  // and down projections' partial [R, H] outputs summed across the ranks
  // before the output-side convs read them.
  const auto stage = [&](uint16_t* fallback, int width) -> uint16_t* {
    if (!h_.boundary) return fallback;
    uint16_t* s = h_.boundary->stage(R, width);
    if (s == nullptr && capture)
      throw std::runtime_error("dflash2 draft: a capture fold does not fit the recorder's staged buffer");
    return s ? s : fallback;
  };
  const auto fold = [&](uint16_t* buf, int width) {
    if (!h_.boundary) return;
    if (!capture) DGPP_CUDA_OK(cudaStreamSynchronize(h_.stream));  // the host-driven fold's contract
    h_.boundary->reduce(buf, R, width);
  };
  const int32_t* tables = h_.device_tables;
  const int bpr = h_.total_blocks;
  const int bt = h_.block_tokens;
  const size_t qkv_stride = static_cast<size_t>(QW) + 2 * KV;
  const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
  for (int l = 0; l < L; ++l) {
    const DFlash2LayerWeights& w = w_.layers[l];
    dflash2_rmsnorm_bf16(df_resid_, w.input_norm, df_x_, R, H, eps, h_.stream);
    // The attention conv: deltas from the normed input, one GEMM; side 0
    // conditions the qkv input, side 1 the attention output.
    h_.gemm->matmul(df_x_, w.attn_conv_kp, df_delta_, R, 2 * TAPS * G, H, DType::BF16, GemmOut::BF16,
                    static_cast<size_t>(H), h_.gemm_ws, h_.gemm_ws_bytes, h_.stream);
    dflash2_grouped_conv_bf16(df_x_, df_delta_, w.attn_conv_base, df_xc_, R, QR, H, TAPS,
                              cfg_.conv_group_size, 2 * TAPS * G, h_.stream);
    if (w.qkv_fp8 != nullptr)
      launch_mma_gemv_fp8_bf16(df_xc_, static_cast<size_t>(H), w.qkv_fp8, w.qkv_scales, df_qkv_, R,
                               static_cast<int>(qkv_stride), H, qkv_stride, 7, 7, h_.stream, h_.gemm_ws,
                               h_.gemm_ws_bytes);
    else
      h_.gemm->matmul(df_xc_, w.qkv, df_qkv_, R, static_cast<int>(qkv_stride), H, DType::BF16,
                      GemmOut::BF16, static_cast<size_t>(H), h_.gemm_ws, h_.gemm_ws_bytes, h_.stream);
    dflash2_norm_rope_bf16(df_qkv_, qkv_stride, w.q_norm, df_pos_, df_inv_freq_, df_q_, QW, R, NH, HD,
                           eps, h_.stream);
    dflash2_norm_rope_bf16(df_qkv_ + QW, qkv_stride, w.k_norm, df_pos_, df_inv_freq_, df_qkv_ + QW,
                           qkv_stride, R, KVH, HD, eps, h_.stream);
    const auto plane = h_.plane(l);
    for (int k = 0; k < slots; ++k) {
      const size_t ro = static_cast<size_t>(k) * QR;
      const int32_t* table = tables + static_cast<size_t>(reqs[k]) * h_.total_blocks;
      const uint16_t* krow = df_qkv_ + ro * qkv_stride + QW;
      qsa_kv_append(krow, qkv_stride, krow + KV, qkv_stride, df_zero_, df_pos_ + ro, QR, table, 1, bt,
                    KVH, HD, plane.k, plane.v, nullptr, nullptr, h_.stream);
      // The split-key form: the row's 2K window in parallel ranges.
      dflash2_block_attn_split(df_q_ + ro * QW, QW, plane.k, plane.v, table, bt, bpr, QR,
                               cfg_.sliding_window, df_pos_ + ro, QR, NH, KVH, HD, scale,
                               df_attn_part_ + ro * static_cast<size_t>(NH) * dflash2_block_attn_splits() * 130,
                               df_attn_ + ro * QW, h_.stream);
    }
    uint16_t* ao = stage(df_o_, H);
    if (w.o_fp8 != nullptr)
      launch_mma_gemv_fp8_bf16(df_attn_, static_cast<size_t>(QW), w.o_fp8, w.o_scales, ao, R, H, QW,
                               static_cast<size_t>(H), 7, 7, h_.stream, h_.gemm_ws, h_.gemm_ws_bytes);
    else
      h_.gemm->matmul(df_attn_, w.o, ao, R, H, QW, DType::BF16, GemmOut::BF16, QW, h_.gemm_ws,
                      h_.gemm_ws_bytes, h_.stream);
    fold(ao, H);
    dflash2_grouped_conv_bf16(ao, df_delta_ + TAPS * G,
                              w.attn_conv_base + static_cast<size_t>(TAPS) * H, df_xc_, R, QR, H, TAPS,
                              cfg_.conv_group_size, 2 * TAPS * G, h_.stream);
    dflash2_add_rmsnorm_bf16(df_resid_, df_xc_, w.post_norm, df_x_, R, H, eps, h_.stream);
    // The MLP conv: same shape around the SwiGLU.
    h_.gemm->matmul(df_x_, w.mlp_conv_kp, df_delta_, R, 2 * TAPS * G, H, DType::BF16, GemmOut::BF16,
                    static_cast<size_t>(H), h_.gemm_ws, h_.gemm_ws_bytes, h_.stream);
    dflash2_grouped_conv_bf16(df_x_, df_delta_, w.mlp_conv_base, df_xc_, R, QR, H, TAPS,
                              cfg_.conv_group_size, 2 * TAPS * G, h_.stream);
    if (w.gate_fp8 != nullptr) {
      launch_mma_gemv_fp8_bf16(df_xc_, static_cast<size_t>(H), w.gate_fp8, w.gate_scales, df_gate_, R, I,
                               H, static_cast<size_t>(I), 7, 7, h_.stream, h_.gemm_ws, h_.gemm_ws_bytes);
      launch_mma_gemv_fp8_bf16(df_xc_, static_cast<size_t>(H), w.up_fp8, w.up_scales, df_up_, R, I, H,
                               static_cast<size_t>(I), 7, 7, h_.stream, h_.gemm_ws, h_.gemm_ws_bytes);
    } else {
      h_.gemm->matmul(df_xc_, w.gate, df_gate_, R, I, H, DType::BF16, GemmOut::BF16,
                      static_cast<size_t>(H), h_.gemm_ws, h_.gemm_ws_bytes, h_.stream);
      h_.gemm->matmul(df_xc_, w.up, df_up_, R, I, H, DType::BF16, GemmOut::BF16, static_cast<size_t>(H),
                      h_.gemm_ws, h_.gemm_ws_bytes, h_.stream);
    }
    swiglu_limit_bf16(df_gate_, df_up_, df_mlp_, static_cast<int64_t>(R) * I, INFINITY, h_.stream);
    uint16_t* mo = stage(df_o_, H);
    if (w.down_fp8 != nullptr)
      launch_mma_gemv_fp8_bf16(df_mlp_, static_cast<size_t>(I), w.down_fp8, w.down_scales, mo, R, H, I,
                               static_cast<size_t>(H), 7, 7, h_.stream, h_.gemm_ws, h_.gemm_ws_bytes);
    else
      h_.gemm->matmul(df_mlp_, w.down, mo, R, H, I, DType::BF16, GemmOut::BF16,
                      static_cast<size_t>(I), h_.gemm_ws, h_.gemm_ws_bytes, h_.stream);
    fold(mo, H);
    dflash2_grouped_conv_bf16(mo, df_delta_ + TAPS * G,
                              w.mlp_conv_base + static_cast<size_t>(TAPS) * H, df_xc_, R, QR, H, TAPS,
                              cfg_.conv_group_size, 2 * TAPS * G, h_.stream);
    add_inplace_bf16(df_resid_, df_xc_, static_cast<int64_t>(R) * H, h_.stream);
  }
  dflash2_rmsnorm_bf16(df_resid_, w_.norm, df_h_, R, H, eps, h_.stream);
}

// The mask rows through the host head and their top-K, per slot at row
// offsets; at world > 1 each rank's slice lists meet in one boundary fold
// and every rank merges the same global top-K (the walk then proposes the
// same tokens on every rank). The proposal: the reference chained selector
// walk (vLLM's _selector_walk_kernel at temperature 0) over each mask
// row's top-K — scores[l][p][c] = unary[l][c] + <pred[id(l-1,p)] *
// hidden[l], succ[id(l,c)]>. The recorded walk draws for a stochastic
// request (the armed proposals); the eager walk is the argmax. Every walk
// with a slot list writes the slots' confidence (the scheduled verify
// depth's input).
void DFlash2Drafter::block_select(int slots, const int* reqs, bool capture) {
  const int H = h_.hidden;
  const int QR = cfg_.query_rows(), D = cfg_.drafts();
  const bool sampled = capture && df_specs_ != nullptr && reqs != nullptr;
  const int64_t V = h_.vocab_count;
  const int topk = cfg_.selector_top_k, rank = cfg_.selector_rank;
  // One head stream for every slot's rows (the mask rows of slot k are
  // rows k*QR + 1 .. k*QR + D; the anchor rows' logits and top-K go
  // unread).
  const int R = slots * QR;
  h_.head(df_h_, df_logits_, R);
  dflash2_topk_f32(df_logits_, df_ids_, df_sc_, V, R, topk, h_.stream, df_topk_ws_, df_topk_ws_bytes_);
  if (h_.world > 1) {
    const int rows = R;
    const size_t elems = dflash2_topk_table_elems(rows, topk, h_.world);
    uint16_t* table = h_.boundary->stage(1, static_cast<int>(elems));
    if (table == nullptr) {
      if (capture)
        throw std::runtime_error("dflash2 draft: the top-K gather does not fit the recorder's staged buffer");
      table = df_table_;
    }
    dflash2_topk_stage(df_ids_, df_sc_, rows, topk, h_.vocab_begin, h_.rank, h_.world, table, h_.stream);
    if (!capture) DGPP_CUDA_OK(cudaStreamSynchronize(h_.stream));  // the host-driven fold's contract
    h_.boundary->reduce(table, 1, static_cast<int>(elems));
    dflash2_topk_merge(table, rows, topk, h_.world, df_ids_, df_sc_, h_.stream);
  }
  h_.gemm->matmul(df_h_, w_.hidden_projection, df_hidden32_, R, rank, H, DType::BF16, GemmOut::F32,
                  static_cast<size_t>(H), h_.gemm_ws, h_.gemm_ws_bytes, h_.stream);
  for (int k = 0; k < slots; ++k) {
    const size_t ro = static_cast<size_t>(k) * QR, co = static_cast<size_t>(k) * D, mo = ro + 1;
    const size_t po = sampled ? static_cast<size_t>(reqs[k]) * static_cast<size_t>(df_props_stride_) : 0;
    dflash2_selector_walk(df_ids_ + mo * topk, df_sc_ + mo * topk, df_hidden32_ + mo * rank,
                          w_.pred_codebook, w_.succ_codebook, df_tokens_ + ro, df_tok_ + co, D, topk,
                          rank, h_.stream, sampled ? df_pos_ + ro : nullptr,
                          sampled ? df_specs_ + reqs[k] : nullptr, sampled ? df_props_ + po : nullptr,
                          sampled && df_props_h_ != nullptr ? df_props_h_ + po : nullptr,
                          reqs != nullptr ? df_conf_ + static_cast<size_t>(reqs[k]) * D : nullptr);
  }
}

// ---- the eager (re)drafts ---------------------------------------------------------

bool DFlash2Drafter::draft(int req, int64_t bonus, std::vector<int32_t>* drafts) {
  if (!on_) throw std::logic_error("dflash2_draft: no drafter loaded");
  const int H = h_.hidden;
  const int QR = cfg_.query_rows(), D = cfg_.drafts();
  const int64_t pos = h_.session_position(req);
  if (pos < 0 || pos + QR > h_.max_context) return false;
  // The block writes K/V at [pos, pos + QR): grow the table now (the next
  // verify's rows live in the same span). Pool exhausted -> plain step.
  if (!h_.ensure_request_blocks(req, pos + QR, h_.stream)) return false;

  // Row inputs: [bonus, mask x D] at positions pos..pos+QR-1.
  for (int j = 0; j < QR; ++j) {
    df_io64_h_[j] = pos + j;
    df_io64_h_[QR + j] = j == 0 ? bonus : cfg_.mask_token_id;
  }
  DGPP_CUDA_OK(cudaMemcpyAsync(df_pos_, df_io64_h_, QR * 8, cudaMemcpyHostToDevice, h_.stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(df_tokens_, df_io64_h_ + QR, QR * 8, cudaMemcpyHostToDevice, h_.stream));
  embed_gather_bf16(h_.embed, df_tokens_, df_resid_, QR, H, h_.stream);
  h_.gemm_rows(QR, true);
  const int reqs[1] = {req};
  block_layers(1, reqs, /*capture=*/false);
  block_select(1, reqs, /*capture=*/false);
  DGPP_CUDA_OK(cudaMemcpyAsync(df_tok_h_, df_tok_, D * 4, cudaMemcpyDeviceToHost, h_.stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(h_.stream));
  drafts->assign(df_tok_h_, df_tok_h_ + D);
  return true;
}

// The batched redraft: the S slots' [bonus, mask x D] blocks stacked into
// one R = S*QR forward. Every row-wise op (embed, norms, GEMMs, the 2-tap
// convs — whose kernel is block-boundary aware via r % block_rows — RoPE
// with the stacked positions) runs over all R rows at once, so the draft
// weights are read once, not S times; the KV appends, the sliding-window
// attention, the head, top-K and selector walk stay per-slot at row
// offsets (the same calls as the scalar path, bitwise identical). A slot
// that fails admission (context bound / pool exhausted) gets empty drafts,
// the scalar failure rule.
void DFlash2Drafter::draft_batch(const std::vector<int>& reqs, const std::vector<int64_t>& bonuses,
                                 std::vector<std::vector<int32_t>>* drafts) {
  if (!on_) throw std::logic_error("dflash2_draft_batch: no drafter loaded");
  if (reqs.empty()) throw std::invalid_argument("dflash2_draft_batch: no requests");
  if (reqs.size() != bonuses.size() || drafts == nullptr)
    throw std::invalid_argument("dflash2_draft_batch: reqs/bonuses/drafts shape");
  const int S = static_cast<int>(reqs.size());
  if (S > df_batch_)
    throw std::invalid_argument("dflash2_draft_batch: batch exceeds the stacked scratch");
  const int H = h_.hidden;
  const int QR = cfg_.query_rows(), D = cfg_.drafts();
  drafts->assign(static_cast<size_t>(S), {});
  // Admission first (no shared state touched): positions and block growth
  // per slot, compacting out the failures.
  std::vector<int> idx, creqs;
  idx.reserve(static_cast<size_t>(S));
  std::vector<int64_t> poss(static_cast<size_t>(S), -1);
  for (int s = 0; s < S; ++s) {
    const int req = reqs[static_cast<size_t>(s)];
    const int64_t pos = h_.session_position(req);
    if (pos < 0 || pos + QR > h_.max_context) continue;
    if (!h_.ensure_request_blocks(req, pos + QR, h_.stream)) continue;
    poss[static_cast<size_t>(s)] = pos;
    idx.push_back(s);
    creqs.push_back(req);
  }
  const int NS = static_cast<int>(idx.size());
  if (NS == 0) return;
  const int R = NS * QR;
  // Row inputs: each stacked block [bonus, mask x D] at its slot's positions.
  for (int k = 0; k < NS; ++k) {
    const int s = idx[static_cast<size_t>(k)];
    const int64_t pos = poss[static_cast<size_t>(s)];
    for (int j = 0; j < QR; ++j) {
      df_io64_h_[static_cast<size_t>(k) * QR + j] = pos + j;
      df_io64_h_[static_cast<size_t>(df_batch_) * QR + static_cast<size_t>(k) * QR + j] =
          j == 0 ? bonuses[static_cast<size_t>(s)] : cfg_.mask_token_id;
    }
  }
  DGPP_CUDA_OK(cudaMemcpyAsync(df_pos_, df_io64_h_, static_cast<size_t>(R) * 8, cudaMemcpyHostToDevice,
                               h_.stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(df_tokens_, df_io64_h_ + static_cast<size_t>(df_batch_) * QR,
                               static_cast<size_t>(R) * 8, cudaMemcpyHostToDevice, h_.stream));
  embed_gather_bf16(h_.embed, df_tokens_, df_resid_, R, H, h_.stream);
  h_.gemm_rows(R, true);
  block_layers(NS, creqs.data(), /*capture=*/false);
  block_select(NS, creqs.data(), /*capture=*/false);
  DGPP_CUDA_OK(cudaMemcpyAsync(df_tok_h_, df_tok_, static_cast<size_t>(NS) * D * 4,
                               cudaMemcpyDeviceToHost, h_.stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(h_.stream));
  for (int k = 0; k < NS; ++k)
    (*drafts)[static_cast<size_t>(idx[static_cast<size_t>(k)])].assign(
        df_tok_h_ + static_cast<size_t>(k) * D, df_tok_h_ + static_cast<size_t>(k + 1) * D);
}

// ---- the recorded block proposal ----------------------------------------------------

void DFlash2Drafter::capture_block_draft(int req, const PickVerdict* verdict, const DFlash2CaptureCtx& cap) {
  if (!on_) throw std::logic_error("session_graph_capture_block_draft: no drafter loaded");
  if (req < 0 || req >= h_.max_requests)
    throw std::invalid_argument("session_graph_capture_block_draft: request");
  if (verdict == nullptr) throw std::invalid_argument("session_graph_capture_block_draft: null verdict");
  if (!cap.graph_device_positions || !cap.graph_device_tokens)
    throw std::logic_error("session_graph_capture_block_draft: the step must be captured with device positions and tokens");
  const int H = h_.hidden;
  const int QR = cfg_.query_rows(), D = cfg_.drafts();
  if ((cap.graph_feed_rows > 0 ? cap.graph_feed_rows : cap.decode_rows) != QR)
    throw std::logic_error("session_graph_capture_block_draft: the token feed is [next, drafts] (T = query rows)");
  // The block's rows off the verdict: the anchor is the verify's next
  // token at the position the recorded commit advanced to; mask rows
  // follow (positions past the context -1: no state write).
  dflash2_stage_block(verdict, h_.session_pos + req, cfg_.mask_token_id, QR, h_.max_context, df_pos_,
                      df_tokens_, h_.stream);
  embed_gather_bf16(h_.embed, df_tokens_, df_resid_, QR, H, h_.stream);
  h_.gemm_rows(QR, true);
  const int reqs[1] = {req};
  block_layers(1, reqs, /*capture=*/true);
  block_select(1, reqs, /*capture=*/true);
  // The next replay's feed and the host's copy of the drafts.
  dflash2_block_feed(verdict, df_tok_, D, cap.step_tokens, h_.stream);
  dflash2_publish_drafts(df_tok_, df_mirror_h_ + static_cast<size_t>(req) * D, D, h_.stream);
  const int K = cfg_.selector_top_k;
  dflash2_publish_words(df_ids_ + K, df_cands_h_ + static_cast<size_t>(req) * D * K, D * K, h_.stream);  // the mask rows' lists
}

void DFlash2Drafter::capture_block_draft_batch(const PickVerdict* verdicts, int requests,
                                               const DFlash2CaptureCtx& cap) {
  if (!on_) throw std::logic_error("session_graph_capture_block_draft_batch: no drafter loaded");
  if (verdicts == nullptr)
    throw std::invalid_argument("session_graph_capture_block_draft_batch: null verdicts");
  if (requests < 1 || requests > h_.max_requests || requests > df_batch_)
    throw std::invalid_argument("session_graph_capture_block_draft_batch: requests exceed the slots or the stacked scratch");
  if (cap.graph_batch_requests != requests || !cap.graph_device_positions || !cap.graph_device_tokens)
    throw std::logic_error("session_graph_capture_block_draft_batch: requires the device-driven fixed batch of these slots");
  if (cap.graph_batch_map_source != nullptr)
    throw std::logic_error("session_graph_capture_block_draft_batch: the block draft takes slot q as request q (no compaction)");
  const int H = h_.hidden;
  const int QR = cfg_.query_rows(), D = cfg_.drafts();
  // The persistent feed is the whole block per slot ([next, drafts] at
  // query rows); a reduced-row verify (engine.dflash_batch_rows) reads it
  // compacted and the block draft still writes the whole of it.
  const int feed_rows = cap.graph_feed_rows > 0 ? cap.graph_feed_rows : cap.graph_rows_per_request;
  if (feed_rows != QR || cap.graph_rows_per_request > QR)
    throw std::logic_error("session_graph_capture_block_draft_batch: the feed is [next, drafts] per slot (T = query rows)");
  const int R = requests * QR;
  dflash2_stage_block_batched(verdicts, h_.session_pos, cfg_.mask_token_id, QR, requests, h_.max_context,
                              df_pos_, df_tokens_, h_.stream);
  embed_gather_bf16(h_.embed, df_tokens_, df_resid_, R, H, h_.stream);
  h_.gemm_rows(R, true);
  std::vector<int> reqs(static_cast<size_t>(requests));
  for (int q = 0; q < requests; ++q) reqs[static_cast<size_t>(q)] = q;
  block_layers(requests, reqs.data(), /*capture=*/true);
  block_select(requests, reqs.data(), /*capture=*/true);
  // The fixed batch's feeds sit behind the decode rows at QR rows per slot
  // (the step capture's step_tokens_).
  int64_t* feeds = h_.decode_tokens + static_cast<size_t>(h_.max_decode_rows);
  dflash2_block_feed_batched(verdicts, df_tok_, D, requests, QR, feeds, h_.stream);
  dflash2_publish_drafts_batched(df_tok_, df_mirror_h_, D, requests, h_.stream);
  const int K = cfg_.selector_top_k;
  for (int q = 0; q < requests; ++q)  // slot q == request q: its mask rows' lists
    dflash2_publish_words(df_ids_ + (static_cast<size_t>(q) * QR + 1) * K,
                          df_cands_h_ + static_cast<size_t>(q) * D * K, D * K, h_.stream);
}

}  // namespace dgpp
