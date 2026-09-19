// The MTP draft block (DESIGN §9): the model's speculative-decode proposer.
//
// The checkpoint's draft layer (cfg.mtp_layer()) is a plain pre-norm
// residual block — DSA attention + MoE, no mHC — fed by
//     x_q = eh_proj([enorm(embed(tok_{q+1})) | hnorm(h_q)])
// where h_q is the main stack's pre-final-norm hidden at position q (the
// mean of its four mHC streams), and headed by shared_head.norm + the
// SHARED lm head. Its row q predicts tok_{q+2}: given what the main stack
// knows at q and the token it just chose for q+1, guess q+2.
//
// The block keeps its own DSA cache (pool ordinal main_dsa_layers_), so it
// must see every position once, in order: session_prefill runs it over the
// prompt's rows 0..P-2 (row P-1 needs tok_P, which the first pick supplies),
// and every draft afterwards runs it over exactly the rows the main stack
// has advanced past since — after a verify accepted `a` rows, the a rows
// whose tokens are those rows' argmaxes. Only accepted tokens ever enter
// the block, so its state never rolls back; positions for row q are q
// (vLLM's convention: the position of the hidden, not of the embedded
// token — a uniform shift would be RoPE-invariant but not kpool-invariant).
#include "models/glm/forward.hpp"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/glm_spec.hpp"
#include "kernels/kernels.hpp"
#include "models/glm/step_timing.hpp"

namespace dgpp {

uint16_t* GlmDiagnosticModel::mtp_hidden_cache(int req) const {
  return mtp_hidden_ + static_cast<size_t>(req) *
                           static_cast<size_t>(max_context_) *
                           static_cast<size_t>(cfg_.hidden_size);
}

// ---------------------------------------------------------------------------
// The block over T rows. Tokens are already in d_tokens_; the hidden rows
// come from the position cache (decode rows: positions from d_step_pos_,
// prefill: first_pos + t).
// ---------------------------------------------------------------------------
void GlmDiagnosticModel::mtp_run_rows(int req, int64_t first_pos, int T,
                                      bool decode_row, bool capture_mode,
                                      int head_rows, int batch_requests) {
  if (!mtp_) throw std::logic_error("mtp_run_rows: MTP is not enabled");
  if (head_rows != 1 && head_rows != T)
    throw std::invalid_argument("mtp_run_rows: head_rows must be 1 or T");
  const int H = cfg_.hidden_size;
  const float eps = cfg_.rms_norm_eps;
  const bool batched = batch_requests > 0;
  const GlmLayerResident& r = stack_layer(cfg_.mtp_layer());
  if (!r.enorm || !r.hnorm || !r.eh_proj || !r.shared_head_norm)
    throw std::runtime_error("mtp: the draft layer's head tensors are unbound");
  const GlmLayerBound b = bind_layer(r, /*dense_mlp=*/false);

  // ---- input: [enorm(embed) | hnorm(hidden)] -> eh_proj ------------------
  if (decode_row && batched) {
    glm_mtp_input_bf16_batched(
        globals_.embed, step_tokens_, mtp_hidden_,
        static_cast<int64_t>(max_context_) * H, d_req_ids_, d_step_pos_,
        r.enorm, r.hnorm, mtp_cat_, T, H, eps, stream_);
  } else {
    glm_mtp_input_bf16(globals_.embed, step_tokens_, mtp_hidden_cache(req),
                       decode_row ? d_step_pos_ : nullptr, first_pos, r.enorm,
                       r.hnorm, mtp_cat_, T, H, eps, stream_);
  }
  // An attached close snapshot can need one eager draft row before the
  // suffix. Its shifted token can be an image token too.
  if (!decode_row || (prefill_images_ && !capture_mode && !batched))
    apply_image_embeddings(mtp_cat_, first_pos + 1, T, r.enorm);
  if (!capture_mode) debug_sync("mtp input", -1, decode_row);
  gemm_.matmul(mtp_cat_, r.eh_proj, mtp_x_, T, H, 2 * H, DType::BF16,
               GemmOut::BF16, static_cast<size_t>(2 * H), gemm_ws_,
               gemm_ws_bytes_, stream_);
  if (!capture_mode) debug_sync("mtp eh_proj", -1, decode_row);

  const auto fold = [&](uint16_t* partial) {
    if (!boundary_) return;
    if (!capture_mode) {
      step_timing::Scope drain(step_timing::kFoldDrain);
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
    }
    boundary_->reduce(partial, T, H);
  };
  const auto staged_or = [&](uint16_t* fallback) {
    if (boundary_)
      if (uint16_t* staged = boundary_->stage(T, H)) return staged;
    return fallback;
  };

  // ---- attention site -----------------------------------------------------
  glm_rmsnorm_bf16(mtp_x_, b.ln1, normed_, T, H, eps, stream_);
  uint16_t* attn_out = staged_or(sub_out_);
  dsa_->rebind(*b.dsa);
  if (!dsa_->prepare(T))
    throw std::runtime_error("mtp: DSA GEMM plans unavailable");
  const int ordinal = main_dsa_layers_;
  if (decode_row) {
    dsa_->enqueue_decode(normed_, pool_, ordinal, d_req_ids_, d_step_pos_,
                         d_req_spans_, batched ? batch_requests : 1, T, attn_out,
                         stream_, &prefetch_);
  } else {
    dsa_->enqueue_prefill(normed_, pool_, ordinal, req, first_pos, T,
                          attn_out, stream_);
  }
  if (!capture_mode) debug_sync("mtp attention", -1, decode_row);
  fold(attn_out);
  glm_residual_add_bf16(mtp_x_, attn_out, static_cast<int64_t>(T) * H,
                        stream_);

  // ---- feed-forward site --------------------------------------------------
  glm_rmsnorm_bf16(mtp_x_, b.ln2, normed_, T, H, eps, stream_);
  uint16_t* ffn_out = staged_or(sub_out_);
  moe_->rebind(*b.moe);
  if (decode_row) {
    moe_->enqueue_decode(normed_, ffn_out, T, /*trace=*/nullptr, stream_,
                         capture_mode ? n_moe_layers_ : -1);
  } else {
    moe_->enqueue(normed_, ffn_out, T, stream_);
  }
  if (!capture_mode) debug_sync("mtp ffn", -1, decode_row);
  fold(ffn_out);
  glm_residual_add_bf16(mtp_x_, ffn_out, static_cast<int64_t>(T) * H,
                        stream_);
  if (decode_row) prefetch_.join(stream_);
  if (!decode_row) return;  // prefill rows fill the cache; no head

  // ---- head: the draft distribution ---------------------------------------
  // The eager draft heads its LAST row (the one after the accepted rows);
  // the in-graph draft heads every row of its fixed batch and the pick
  // selects the last ACCEPTED one — the lm head is bandwidth-bound, m=2
  // costs what m=1 costs.
  const uint16_t* head_in =
      mtp_x_ + static_cast<size_t>(T - head_rows) * H;
  glm_rmsnorm_bf16(head_in, r.shared_head_norm, normed_, head_rows, H, eps,
                   stream_);
  gemm_.matmul(normed_, globals_.lm_head, logits_, head_rows, lm_vocab_count_,
               H, DType::BF16, GemmOut::F32, H, gemm_ws_, gemm_ws_bytes_,
               stream_);
  // A memcpy NODE only while the mirrors are on (the kernels-only decode
  // graph, docs/batched_mtp_graph_stall.md); an eager draft always mirrors.
  if (decode_tail_mirrors_ || !capture_mode)
    DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_logits_, logits_,
                                 static_cast<size_t>(head_rows) *
                                     lm_vocab_count_ * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
}

void GlmDiagnosticModel::push_mtp_position(int req) {
  h_mtp_pos_[req] = mtp_pos_[static_cast<size_t>(req)];
  DGPP_CUDA_OK(cudaMemcpyAsync(d_mtp_pos_ + req, h_mtp_pos_ + req,
                               sizeof(int64_t), cudaMemcpyHostToDevice,
                               stream_));
}

// ---------------------------------------------------------------------------
// Prefill: the draft block over rows [row0, row1) of a chunk — row q embeds
// tok_{q+1}, so `tokens` points at the prompt's token at position row0 + 1.
// Called per main-stack chunk (interleaved, 2026-09-05), which keeps the
// block's state at every chunk cut for the prefix cache's snapshots; the
// row range is one aligned chunk (<= kPrefillChunkTokens rows).
// ---------------------------------------------------------------------------
void GlmDiagnosticModel::mtp_prefill_rows(int req, int64_t row0, int64_t row1,
                                          const int64_t* tokens) {
  const int64_t n = row1 - row0;
  if (n <= 0) return;
  if (n > kPrefillChunkTokens)
    throw std::invalid_argument("mtp_prefill_rows: chunk too long");
  DGPP_CUDA_OK(cudaMemcpyAsync(d_tokens_, tokens, static_cast<size_t>(n) * 8,
                               cudaMemcpyHostToDevice, stream_));
  mtp_run_rows(req, row0, static_cast<int>(n), /*decode_row=*/false,
               /*capture_mode=*/false);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}

void GlmDiagnosticModel::mtp_prefill(int req,
                                     const std::vector<int64_t>& prompt_ids) {
  // The whole-prompt form (kept for callers outside the session chunk loop):
  // the same cuts the main stack would take with no boundaries.
  const int64_t rows_total = static_cast<int64_t>(prompt_ids.size()) - 1;
  mtp_pos_[static_cast<size_t>(req)] = 0;
  if (rows_total > 0) {
    const std::vector<int64_t> cuts = prefill_cuts(0, rows_total, {});
    int64_t c0 = 0;
    size_t ci = 0;
    while (c0 < rows_total) {
      const int64_t c1 = ci < cuts.size() ? cuts[ci++] : rows_total;
      mtp_prefill_rows(req, c0, c1, prompt_ids.data() + c0 + 1);
      c0 = c1;
    }
  }
  mtp_pos_[static_cast<size_t>(req)] = std::max<int64_t>(rows_total, 0);
  push_mtp_position(req);
}

// ---------------------------------------------------------------------------
// Draft: the rows since the last draft, eager and graph-era.
// ---------------------------------------------------------------------------
void GlmDiagnosticModel::mtp_decode_host_prep(
    int req, const std::vector<int64_t>& tokens, bool upload) {
  if (!mtp_) throw std::logic_error("session_draft: MTP is not enabled");
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_draft: request slot " +
                            std::to_string(req));
  const int T = static_cast<int>(tokens.size());
  if (T < 1 || T > kSpecRows)
    throw std::invalid_argument("session_draft: row count must be in [1, " +
                                std::to_string(kSpecRows) + "]");
  const int64_t pos = session_pos_[static_cast<size_t>(req)];
  const int64_t q = mtp_pos_[static_cast<size_t>(req)];
  if (pos <= 0)
    throw std::invalid_argument("session_draft: no open session on slot " +
                                std::to_string(req));
  // The block must see every position once: the draft covers exactly the
  // rows between its counter and the main stack's position.
  if (q + T != pos)
    throw std::invalid_argument(
        "session_draft: " + std::to_string(T) + " rows from draft position " +
        std::to_string(q) + " do not reach the session position " +
        std::to_string(pos) + " (draft exactly the accepted rows)");
  for (int64_t id : tokens)
    if (id < 0 || id >= cfg_.vocab_size)
      throw std::invalid_argument("session_draft: token id out of range");
  // Admission: the main stack's positions >= the draft's, and block tables
  // are per request across layers — already covered by the verify.
  for (int r = 0; r < T; ++r) {
    h_req_ids_[r] = req;
    h_step_pos_[r] = q + r;
    h_token_[r] = tokens[static_cast<size_t>(r)];
  }
  h_req_spans_[0] = 0;
  h_req_spans_[1] = T;
  draft_rows_ = T;
  if (upload) {
    DGPP_CUDA_OK(cudaMemcpyAsync(d_req_ids_, h_req_ids_, sizeof(int32_t) * T,
                                 cudaMemcpyHostToDevice, stream_));
    DGPP_CUDA_OK(cudaMemcpyAsync(d_step_pos_, h_step_pos_, sizeof(int64_t) * T,
                                 cudaMemcpyHostToDevice, stream_));
    DGPP_CUDA_OK(cudaMemcpyAsync(d_req_spans_, h_req_spans_,
                                 2 * sizeof(int32_t), cudaMemcpyHostToDevice,
                                 stream_));
    DGPP_CUDA_OK(cudaMemcpyAsync(d_tokens_, h_token_, sizeof(int64_t) * T,
                                 cudaMemcpyHostToDevice, stream_));
  }
}

GlmDiagnosticModel::Outputs GlmDiagnosticModel::mtp_decode_tail() {
  Outputs out;
  out.logits.assign(h_tail_logits_,
                    h_tail_logits_ + static_cast<size_t>(lm_vocab_count_));
  out.lm_vocab_begin = lm_vocab_begin_;
  out.lm_vocab_count = lm_vocab_count_;
  return out;
}

GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_draft(
    int req, const std::vector<int64_t>& tokens) {
  step_timing::Scope tick(step_timing::kStep);
  step_tokens_ = d_tokens_;
  mtp_decode_host_prep(req, tokens, /*upload=*/true);
  const int64_t q = mtp_pos_[static_cast<size_t>(req)];
  mtp_run_rows(req, q, static_cast<int>(tokens.size()), /*decode_row=*/true,
               /*capture_mode=*/false);
  {
    step_timing::Scope sync_tick(step_timing::kFinalSync);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  }
  mtp_pos_[static_cast<size_t>(req)] = q + static_cast<int64_t>(tokens.size());
  mtp_draft_last_row_ = static_cast<int>(tokens.size()) - 1;
  push_mtp_position(req);
  return mtp_decode_tail();
}

// ---------------------------------------------------------------------------
// The chained draft (depth >= 2, 2026-09-06): one more block row past the
// counter, off the block's own output. Eager and in-graph forms; the ring
// guard around the chain in both.
// ---------------------------------------------------------------------------
void GlmDiagnosticModel::chain_ring_copy(int req, bool restore) {
  const size_t ring = spec_tail_ring_elems();
  uint16_t* live = static_cast<uint16_t*>(pool_.tail(main_dsa_layers_)) +
                   static_cast<size_t>(req) * ring;
  uint16_t* snap = mtp_chain_ring_ + static_cast<size_t>(req) * ring;
  if (restore)
    glm_device_copy(live, snap, ring * 2, stream_);
  else
    glm_device_copy(snap, live, ring * 2, stream_);
}

bool GlmDiagnosticModel::session_draft_chain_fits(int req, int index) const {
  if (!mtp_ || req < 0 || req >= max_requests_ || index < 0) return false;
  return mtp_pos_[static_cast<size_t>(req)] + index < max_context_;
}

GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_draft_chain(
    int req, int64_t token, int index, bool first, bool last) {
  if (!mtp_) throw std::logic_error("session_draft_chain: MTP is not enabled");
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_draft_chain: request slot " +
                            std::to_string(req));
  if (index < 0 || index >= kSpecRows - 1)
    throw std::invalid_argument("session_draft_chain: chain index");
  if (token < 0 || token >= cfg_.vocab_size)
    throw std::invalid_argument("session_draft_chain: token id out of range");
  const int64_t q = mtp_pos_[static_cast<size_t>(req)];
  if (session_pos_[static_cast<size_t>(req)] <= 0 || q < 1)
    throw std::invalid_argument("session_draft_chain: no drafted session on slot " +
                                std::to_string(req));
  const int64_t pos = q + index;
  if (pos >= max_context_)
    throw std::invalid_argument("session_draft_chain: position exceeds the context bound");
  step_timing::Scope tick(step_timing::kStep);
  step_tokens_ = d_tokens_;
  const int H = cfg_.hidden_size;
  if (first) chain_ring_copy(req, /*restore=*/false);
  h_req_ids_[0] = req;
  h_step_pos_[0] = pos;
  h_token_[0] = token;
  h_req_spans_[0] = 0;
  h_req_spans_[1] = 1;
  DGPP_CUDA_OK(cudaMemcpyAsync(d_req_ids_, h_req_ids_, sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream_));
  DGPP_CUDA_OK(cudaMemcpyAsync(d_step_pos_, h_step_pos_, sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_));
  DGPP_CUDA_OK(cudaMemcpyAsync(d_req_spans_, h_req_spans_, 2 * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream_));
  DGPP_CUDA_OK(cudaMemcpyAsync(d_tokens_, h_token_, sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_));
  // The block's previous output row is the row's hidden, at its position.
  glm_device_copy(mtp_hidden_cache(req) + static_cast<size_t>(pos) * H,
                  mtp_x_ + static_cast<size_t>(mtp_draft_last_row_) * H,
                  static_cast<size_t>(H) * 2, stream_);
  mtp_run_rows(req, pos, /*T=*/1, /*decode_row=*/true, /*capture_mode=*/false);
  if (last) chain_ring_copy(req, /*restore=*/true);
  {
    step_timing::Scope sync_tick(step_timing::kFinalSync);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  }
  mtp_draft_last_row_ = 0;
  return mtp_decode_tail();
}

void GlmDiagnosticModel::session_graph_capture_draft_chain(
    int req, const PickVerdict* verify_verdict,
    const PickVerdict* draft_verdict, int index, bool first, bool last) {
  if (!mtp_) throw std::logic_error("session_graph_capture_draft_chain: no MTP");
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_graph_capture_draft_chain: request slot " +
                            std::to_string(req));
  if (!graph_device_positions_ || !graph_device_tokens_ || !graph_has_draft_)
    throw std::logic_error(
        "session_graph_capture_draft_chain: needs the device-driven capture "
        "with the draft in the graph");
  if (draft_verdict == nullptr || (index == 0 && verify_verdict == nullptr))
    throw std::invalid_argument("session_graph_capture_draft_chain: null verdict");
  if (index < 0 || index >= kSpecRows - 1)
    throw std::invalid_argument("session_graph_capture_draft_chain: chain index");
  if (first) chain_ring_copy(req, /*restore=*/false);
  glm_spec_chain_row(index == 0 ? verify_verdict : nullptr, /*src_row=*/0,
                     draft_verdict, mtp_x_, cfg_.hidden_size,
                     mtp_hidden_cache(req), d_mtp_pos_ + req, index,
                     static_cast<int64_t>(max_context_), d_step_pos_,
                     step_tokens_, d_req_spans_, stream_);
  mtp_run_rows(req, /*first_pos=*/0, /*T=*/1, /*decode_row=*/true,
               /*capture_mode=*/true, /*head_rows=*/1);
  if (last) chain_ring_copy(req, /*restore=*/true);
}

void GlmDiagnosticModel::session_graph_capture_draft(
    int req, const std::vector<int64_t>& tokens) {
  mtp_decode_host_prep(req, tokens, /*upload=*/true);
  // Records only — nothing executes, the counter does not move.
  mtp_run_rows(req, mtp_pos_[static_cast<size_t>(req)],
               static_cast<int>(tokens.size()), /*decode_row=*/true,
               /*capture_mode=*/true);
}

void GlmDiagnosticModel::session_graph_stage_draft(
    int req, const std::vector<int64_t>& tokens) {
  // The token upload is a recorded memcpy node off h_token_ (as the verify
  // graph's): the stage only refreshes the pinned members.
  mtp_decode_host_prep(req, tokens, /*upload=*/false);
}

GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_graph_collect_draft(
    int req) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_graph_collect_draft: request slot " +
                            std::to_string(req));
  mtp_pos_[static_cast<size_t>(req)] += draft_rows_;
  push_mtp_position(req);
  return mtp_decode_tail();
}

// ---------------------------------------------------------------------------
// The in-graph draft (phase C): the block's rows come off the verify's
// verdict on the device, the block runs its fixed T rows, the head runs on
// every row. Recorded behind the caller's recorded pick and the commit.
// ---------------------------------------------------------------------------
void GlmDiagnosticModel::session_graph_capture_draft(
    int req, const PickVerdict* verify_verdict) {
  if (!mtp_) throw std::logic_error("session_graph_capture_draft: no MTP");
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_graph_capture_draft: request slot " +
                            std::to_string(req));
  if (!graph_device_positions_)
    throw std::logic_error(
        "session_graph_capture_draft: the step must be captured with device "
        "positions (the draft's rows come off the device verdict)");
  if (verify_verdict == nullptr)
    throw std::invalid_argument("session_graph_capture_draft: null verdict");
  const int T = decode_rows_;
  // The block's ring before its rows: the sampled step's fallback rolls
  // the draft back to here (session_draft_rollback).
  snapshot_draft_ring(req);
  // d_req_ids_/d_req_spans_ still describe T rows of `req` from the verify
  // (same batch shape); the rows' positions and tokens come off the verdict.
  glm_spec_draft_rows(verify_verdict, T, d_mtp_pos_ + req, d_step_pos_,
                      step_tokens_, d_next_ + req, stream_);
  draft_rows_ = T;
  mtp_run_rows(req, /*first_pos=*/0, T, /*decode_row=*/true,
               /*capture_mode=*/true, /*head_rows=*/T);
  graph_has_draft_ = true;
}

void GlmDiagnosticModel::session_graph_capture_draft_batch(
    const PickVerdict* verify_verdicts) {
  if (!mtp_)
    throw std::logic_error("session_graph_capture_draft_batch: no MTP");
  if (graph_batch_requests_ <= 0 || graph_rows_per_request_ != 2 ||
      !graph_device_positions_ || !graph_device_tokens_)
    throw std::logic_error(
        "session_graph_capture_draft_batch: requires a device-driven T=2 "
        "fixed batch");
  if (verify_verdicts == nullptr)
    throw std::invalid_argument(
        "session_graph_capture_draft_batch: null verdicts");
  for (int req = 0; req < graph_batch_requests_; ++req)
    snapshot_draft_ring(req);
  glm_spec_draft_rows_batched(
      verify_verdicts, graph_batch_requests_, graph_rows_per_request_,
      d_mtp_pos_, d_step_pos_, step_tokens_, d_next_, stream_);
  draft_rows_ = decode_rows_;
  mtp_run_rows(/*req=*/0, /*first_pos=*/0, decode_rows_,
               /*decode_row=*/true, /*capture_mode=*/true,
               /*head_rows=*/decode_rows_, graph_batch_requests_);
  graph_has_draft_ = true;
}

void GlmDiagnosticModel::snapshot_draft_ring(int req) {
  const size_t ring = spec_tail_ring_elems();
  glm_device_copy(mtp_ring_snapshot_ + static_cast<size_t>(req) * ring,
                  static_cast<const uint16_t*>(pool_.tail(main_dsa_layers_)) +
                      static_cast<size_t>(req) * ring,
                  ring * 2, stream_);
}

void GlmDiagnosticModel::session_draft_ring_snapshot(int req) {
  if (!mtp_) throw std::logic_error("session_draft_ring_snapshot: no MTP");
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_draft_ring_snapshot: request slot " +
                            std::to_string(req));
  snapshot_draft_ring(req);
}

void GlmDiagnosticModel::session_draft_rollback(int req, int rows) {
  if (!mtp_) throw std::logic_error("session_draft_rollback: no MTP");
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_draft_rollback: request slot " +
                            std::to_string(req));
  if (rows < 1 || rows > mtp_pos_[static_cast<size_t>(req)])
    throw std::invalid_argument("session_draft_rollback: rows outside the "
                                "block's counter");
  const size_t ring = spec_tail_ring_elems();
  DGPP_CUDA_OK(cudaMemcpyAsync(
      static_cast<uint16_t*>(pool_.tail(main_dsa_layers_)) +
          static_cast<size_t>(req) * ring,
      mtp_ring_snapshot_ + static_cast<size_t>(req) * ring, ring * 2,
      cudaMemcpyDeviceToDevice, stream_));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  mtp_pos_[static_cast<size_t>(req)] -= rows;
  push_mtp_position(req);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}

void GlmDiagnosticModel::session_graph_capture_next_tokens(
    int req, const PickVerdict* draft_verdict) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_graph_capture_next_tokens: request slot " +
                            std::to_string(req));
  if (!graph_device_tokens_ || !graph_has_draft_)
    throw std::logic_error(
        "session_graph_capture_next_tokens: needs a device-token capture "
        "with the draft in the graph");
  if ((graph_feed_rows_ > 0 ? graph_feed_rows_ : decode_rows_) != 2)
    throw std::logic_error(
        "session_graph_capture_next_tokens: the token feed is [next, draft] "
        "(T = 2)");
  if (draft_verdict == nullptr)
    throw std::invalid_argument("session_graph_capture_next_tokens: null verdict");
  glm_spec_next_tokens(d_next_ + req, draft_verdict, step_tokens_, stream_);
}

void GlmDiagnosticModel::session_graph_capture_next_tokens(
    int req, const std::vector<const PickVerdict*>& draft_verdicts) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_graph_capture_next_tokens: request slot " +
                            std::to_string(req));
  if (!graph_device_tokens_ || !graph_has_draft_)
    throw std::logic_error(
        "session_graph_capture_next_tokens: needs a device-token capture "
        "with the draft in the graph");
  // The feed carries every draft of the block whatever the verify's rows.
  const int feed_rows = graph_feed_rows_ > 0 ? graph_feed_rows_ : decode_rows_;
  if (draft_verdicts.empty() ||
      draft_verdicts.size() != static_cast<size_t>(feed_rows - 1) ||
      draft_verdicts.size() > static_cast<size_t>(kSpecMaxDrafts))
    throw std::logic_error(
        "session_graph_capture_next_tokens: the token feed is [next, one "
        "draft per row after it] (T = 1 + drafts)");
  GlmSpecDrafts d;
  d.count = static_cast<int>(draft_verdicts.size());
  for (int c = 0; c < d.count; ++c) {
    if (draft_verdicts[static_cast<size_t>(c)] == nullptr)
      throw std::invalid_argument("session_graph_capture_next_tokens: null verdict");
    d.v[c] = draft_verdicts[static_cast<size_t>(c)];
  }
  glm_spec_next_tokens(d_next_ + req, d, step_tokens_, stream_);
}

void GlmDiagnosticModel::session_graph_capture_next_tokens_batch(
    const PickVerdict* draft_verdicts) {
  if (!graph_device_tokens_ || !graph_has_draft_ ||
      graph_batch_requests_ <= 0 || graph_rows_per_request_ != 2)
    throw std::logic_error(
        "session_graph_capture_next_tokens_batch: requires the fixed T=2 "
        "draft graph");
  if (draft_verdicts == nullptr)
    throw std::invalid_argument(
        "session_graph_capture_next_tokens_batch: null verdicts");
  glm_spec_next_tokens_batched(
      d_next_, draft_verdicts, graph_batch_requests_,
      graph_rows_per_request_, step_tokens_, stream_);
}

void GlmDiagnosticModel::session_graph_seed_tokens(
    int req, const std::vector<int64_t>& ids) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_graph_seed_tokens: request slot " +
                            std::to_string(req));
  const int rows_per_request = graph_batch_requests_ > 0
                                   ? graph_rows_per_request_
                                   : decode_rows_;
  if (ids.size() != static_cast<size_t>(rows_per_request))
    throw std::invalid_argument("session_graph_seed_tokens: the seed must "
                                "have the graph's row count");
  for (int64_t id : ids)
    if (id < 0 || id >= cfg_.vocab_size)
      throw std::invalid_argument("session_graph_seed_tokens: token id");
  // On the model's stream, never the legacy stream: a peer rank in the same
  // process (the loopback worlds) may be mid-capture, and a legacy-stream
  // copy would try to synchronize with its capturing stream.
  // The slot's feed rows (device_feed): the batch's and the scalar
  // variant's alike.
  const size_t row0 = static_cast<size_t>(kDecodeRows) +
                      static_cast<size_t>(req) * rows_per_request;
  for (size_t i = 0; i < ids.size(); ++i) h_token_[row0 + i] = ids[i];
  DGPP_CUDA_OK(cudaMemcpyAsync(d_tokens_ + row0, h_token_ + row0,
                               ids.size() * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}

void GlmDiagnosticModel::session_graph_seed_scalar_tokens(
    const std::vector<int64_t>& ids) {
  if (ids.empty() || ids.size() > static_cast<size_t>(kSpecRows))
    throw std::invalid_argument(
        "session_graph_seed_scalar_tokens: invalid row count");
  for (int64_t id : ids)
    if (id < 0 || id >= cfg_.vocab_size)
      throw std::invalid_argument(
          "session_graph_seed_scalar_tokens: token id");
  for (size_t i = 0; i < ids.size(); ++i) h_token_[i] = ids[i];
  DGPP_CUDA_OK(cudaMemcpyAsync(d_tokens_, h_token_,
                               ids.size() * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}

void GlmDiagnosticModel::session_graph_seed_feed(
    int req, const std::vector<int64_t>& ids) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_graph_seed_feed: request slot " +
                            std::to_string(req));
  if (ids.empty() || ids.size() > static_cast<size_t>(kSpecRows))
    throw std::invalid_argument("session_graph_seed_feed: invalid row count");
  for (int64_t id : ids)
    if (id < 0 || id >= cfg_.vocab_size)
      throw std::invalid_argument("session_graph_seed_feed: token id");
  // The slot's own pinned rows (a later seed of another slot must not
  // rewrite a copy's source before it ran) and its device feed rows; on
  // the model's stream, never the legacy stream.
  const size_t row0 = static_cast<size_t>(kDecodeRows) +
                      static_cast<size_t>(req) * ids.size();
  for (size_t i = 0; i < ids.size(); ++i) h_token_[row0 + i] = ids[i];
  DGPP_CUDA_OK(cudaMemcpyAsync(d_tokens_ + row0, h_token_ + row0,
                               ids.size() * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}

}  // namespace dgpp
