#include "kernels/glm_vision.hpp"
#include "models/glm/forward.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "kernels/dsa.hpp"
#include "kernels/glm_mhc_launch.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/kernels.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/glm_spec.hpp"
#include "kernels/scale_gemm.hpp"
#include "models/glm/step_timing.hpp"

namespace dgpp {

// Q0 (2026-09-09): the sliced-GR gate logits' width — the probe collective's
// shape (see GlmBoundaryReducer::probe).
constexpr int kGrProbeCols = 10240;
namespace {


// A quantized matrix's two device ranges (payload and block scales are
// separate allocations), handed to the prefetcher as such.
void prefetch_quant(WeightPrefetcher& pf, const GlmQuantMatrix& m) {
  pf.add(m.payload, static_cast<size_t>(m.rows) * m.cols);
  pf.add(m.scales,
         static_cast<size_t>((m.rows + 127) / 128) * ((m.cols + 127) / 128) *
             sizeof(float));
}

}  // namespace

// ---------------------------------------------------------------------------
// The boundary prefetch windows. Each block boundary is a bus all-reduce
// the chain waits on (~30 us) followed by the mHC site and a norm (~30 us
// more of latency-bound kernels) before the next weight-streaming kernel
// starts: ~60 us during which DRAM would idle. The window opened here
// runs through all of it, so the other side's first ~12 MB of weights are
// in L2 when their kernels arrive. Only the weights that exist regardless
// of routing are prefetchable — the routed experts wait for the router.
// ---------------------------------------------------------------------------
// The comb fork (see forward.hpp): the side stream waits for the finish
// (its logits), runs launch_mhc_comb, and records the join the stream
// update waits on. Under capture the record/wait pairs become the graph's
// fork and join edges, as the prefetcher's do.
void GlmDiagnosticModel::mhc_comb_fork(const GlmMhcWeights& w, int tokens) {
  DGPP_CUDA_OK(cudaEventRecord(mhc_fork_, stream_));
  DGPP_CUDA_OK(cudaStreamWaitEvent(mhc_side_, mhc_fork_, 0));
  launch_mhc_comb(mhc_logits_, w, mhc_cfg_, comb_, tokens, mhc_side_);
  DGPP_CUDA_OK(cudaEventRecord(mhc_join_, mhc_side_));
}

void GlmDiagnosticModel::prefetch_ffn_side(const GlmLayerBound& b,
                                           bool dense_mlp) {
  if (!prefetch_.enabled()) return;
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, 0, prefetch_.boundary_rate());
  prefetch_.add(b.mhc->ffn_fn,
                static_cast<size_t>(mhc_cfg_.coeff_rows()) * mhc_cfg_.hc_mult *
                    H * 2);
  prefetch_.add(b.ln2, H * 2);
  if (dense_mlp) {
    for (int i = 0; i < 3; ++i) prefetch_quant(prefetch_, b.dense[i]);
    return;
  }
  prefetch_.add(b.moe->router_gate,
                static_cast<size_t>(moe_cfg_.n_experts) * H * 2);
  prefetch_.add(b.moe->router_bias,
                static_cast<size_t>(moe_cfg_.n_experts) * sizeof(float));
  for (int i = 0; i < 3; ++i) prefetch_quant(prefetch_, b.moe->shared[i]);
}

void GlmDiagnosticModel::prefetch_attention_side(int layer) {
  if (!prefetch_.enabled()) return;
  if (layer >= cfg_.num_hidden_layers) {
    prefetch_head();
    return;
  }
  // Resident stacks only: load_layer is a lookup there. A streaming stack
  // loads on demand, and touching layer N+1 during layer N would reorder
  // the loader's one-layer-at-a-time contract.
  if (loader_.residency() != GlmResidency::Resident) return;
  const GlmLayerResident& r = loader_.load_layer(layer);
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, 0, prefetch_.boundary_rate());
  prefetch_.add(r.mhc.attn_fn,
                static_cast<size_t>(mhc_cfg_.coeff_rows()) * mhc_cfg_.hc_mult *
                    H * 2);
  prefetch_.add(r.ln1, H * 2);
  // The first projection is far larger than the window; add() clamps to
  // the budget and the GEMV's leading blocks are the ones that hit.
  if (r.kind == GlmLayerKind::Kda) {
    if (kda_) prefetch_.add(r.kda.in_proj, kda_->in_proj_bytes());
  } else {
    if (dsa_) prefetch_.add(r.dsa.qkv_a, dsa_->qkv_a_bytes());
  }
}

void GlmDiagnosticModel::prefetch_head() {
  if (!prefetch_.enabled()) return;
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, 0, prefetch_.boundary_rate());
  prefetch_.add(globals_.final_norm, H * 2);
  prefetch_.add(globals_.lm_head, static_cast<size_t>(lm_vocab_count_) * H * 2);
}

// ---------------------------------------------------------------------------
// session_prefill: opens request slot `req` and processes the prompt.
// ---------------------------------------------------------------------------
GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_prefill(
    int req, const std::vector<int64_t>& prompt_ids) {
  return session_prefill(req, prompt_ids, {}, nullptr);
}

int64_t GlmDiagnosticModel::kv_block_tokens() const {
  return dsa_cfg_.num_dsa_layers > 0 ? dsa_cfg_.block_tokens : 0;
}

int GlmDiagnosticModel::session_snapshot_align() const {
  return dsa_cfg_.num_dsa_layers > 0 ? dsa_cfg_.index_kpool : 1;
}

size_t GlmDiagnosticModel::session_snapshot_bytes(const GlmTextConfig& cfg,
                                                  int tp_world, bool mtp) {
  size_t bytes = 0;
  KdaConfig kda = cfg.kda_config();
  kda.tp_size = tp_world;
  if (kda.num_kda_layers > 0) {
    const KdaGeometry g = KdaGeometry::from_config(kda);
    bytes += static_cast<size_t>(kda.num_kda_layers) *
             (g.recurrent_bytes + g.conv_committed_bytes);
  }
  DsaConfig dsa = cfg.dsa_config();
  dsa.tp_size = tp_world;
  if (dsa.num_dsa_layers > 0) {
    const DsaGeometry g = DsaGeometry::from_config(dsa);
    bytes += static_cast<size_t>(dsa.num_dsa_layers + (mtp ? 1 : 0)) *
             g.tail_bytes_per_request;
  }
  if (mtp) bytes += static_cast<size_t>(cfg.hidden_size) * 2;
  return bytes;
}

size_t GlmDiagnosticModel::session_snapshot_bytes() const {
  size_t bytes = 0;
  if (kda_rec_) {
    bytes += static_cast<size_t>(kda_cfg_.num_kda_layers) * kda_geo_.recurrent_bytes;
    bytes += static_cast<size_t>(kda_cfg_.num_kda_layers) * kda_geo_.conv_committed_bytes;
  }
  if (dsa_cfg_.num_dsa_layers > 0)
    bytes += static_cast<size_t>(dsa_cfg_.num_dsa_layers) *
             pool_.geometry().tail_bytes_per_request;
  if (mtp_) bytes += static_cast<size_t>(cfg_.hidden_size) * 2;
  return bytes;
}

std::vector<int64_t> GlmDiagnosticModel::prefill_cuts(
    int64_t start, int64_t end, const std::vector<int64_t>& boundaries) const {
  const int64_t kpool = session_snapshot_align();
  std::vector<int64_t> cuts;
  for (int64_t m = (start / kPrefillChunkTokens + 1) * kPrefillChunkTokens; m < end;
       m += kPrefillChunkTokens)
    cuts.push_back(m);
  for (int64_t b : boundaries) {
    const int64_t a = (b / kpool) * kpool;  // the pool-aligned image
    if (a > start && a < end) cuts.push_back(a);
  }
  std::sort(cuts.begin(), cuts.end());
  cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
  return cuts;
}

GlmDiagnosticModel::PrefillCursor GlmDiagnosticModel::prefill_cursor(
    int req, const int64_t* ids, int64_t start, int64_t count,
    const std::vector<int64_t>& boundaries, SnapshotRequest* snap) {
  PrefillCursor cursor;
  cursor.req = req;
  cursor.ids = ids;
  cursor.images = prefill_images_;
  cursor.start = cursor.next = start;
  cursor.end = start + count;
  cursor.cuts = prefill_cuts(start, cursor.end, boundaries);
  cursor.snap = snap;
  if (snap != nullptr) {
    if (snap->dst == nullptr || snap->meta == nullptr)
      throw std::invalid_argument("session_prefill: snapshot request without a buffer");
    if (!std::binary_search(cursor.cuts.begin(), cursor.cuts.end(), snap->position) &&
        snap->position != cursor.end)
      throw std::invalid_argument("session_prefill: the snapshot position is not a chunk end");
  }
  return cursor;
}

void GlmDiagnosticModel::prefill_chunk(PrefillCursor& cursor, int64_t budget) {
  const int req = cursor.req;
  const int64_t* ids = cursor.ids;
  const int64_t start = cursor.start, end = cursor.end, c0 = cursor.next;
  auto* snap = cursor.snap;
  auto& out = cursor.output;
  int64_t c1 = cursor.cut_index < cursor.cuts.size() ? cursor.cuts[cursor.cut_index] : end;
  if (budget > 0) c1 = std::min(c1, (c0 / budget + 1) * budget);
  if (cursor.cut_index < cursor.cuts.size() && c1 == cursor.cuts[cursor.cut_index]) ++cursor.cut_index;
  if (c1 <= c0 || c1 - c0 > max_tokens_)
    throw std::invalid_argument("session_prefill: chunk outside the model's row capacity");
  // This borrow is scoped to one synchronous main+MTP chunk. Decode graphs
  // and other requests never inherit its images or device window.
  struct ImageScope {
    GlmDiagnosticModel& model;
    const std::vector<ImageInput>* previous;
    ~ImageScope() { model.prefill_images_ = previous; model.image_embeddings_ = nullptr; }
  } scope{*this, prefill_images_};
  prefill_images_ = cursor.images;
  stage_image_embeddings(c0, std::min(c1 + (mtp_ ? 1 : 0), end));
  if (cursor.catchup) {
    (void)session_draft(req, {ids[0]});
    cursor.catchup = false;
  }
  Outputs chunk = session_run_rows(
      req, std::vector<int64_t>(ids + (c0 - start), ids + (c1 - start)), c0,
      /*decode_row=*/false);
  out.logits = std::move(chunk.logits);
  out.final_hidden_bits = std::move(chunk.final_hidden_bits);
  out.lm_vocab_begin = chunk.lm_vocab_begin;
  out.lm_vocab_count = chunk.lm_vocab_count;
  session_merge_routes(&out, std::move(chunk));
  session_pos_[static_cast<size_t>(req)] = c1;
  push_position(req);
  // The draft block over this chunk's rows — row q embeds tok_{q+1}, so
  // the rows stop one short of the prompt end (the first generated token
  // is that row's input). Interleaving it per chunk keeps its state at
  // every cut, where a snapshot may be taken.
  if (mtp_) {
    const int64_t r1 = std::min<int64_t>(c1, end - 1);
    if (r1 > c0) mtp_prefill_rows(req, c0, r1, ids + (c0 + 1 - start));
    mtp_pos_[static_cast<size_t>(req)] = std::max<int64_t>(mtp_pos_[static_cast<size_t>(req)], r1);
    push_mtp_position(req);
  }
  if (snap != nullptr && !snap->taken && snap->position == c1) {
    *snap->meta = session_snapshot(req, snap->dst);
    snap->taken = true;
  }
  cursor.next = c1;
  report_prefill_progress(req, c1);
}

void GlmDiagnosticModel::reset_prefill_session(int req) {
  ++prefill_epochs_[static_cast<size_t>(req)];
  // Open THIS slot only — other slots' sessions are untouched (the Stage
  // 2b concurrency contract). Slot 0's zeroed state is the same starting
  // state run_stack builds, so a single-chunk prefill there still runs
  // the exact reference op sequence (the bitwise tier of the parity gate).
  if (kda_rec_) {
    float* slot_rec =
        kda_rec_ + static_cast<size_t>(req) * kda_cfg_.num_kda_layers *
                       kda_geo_.recurrent_elems;
    uint16_t* slot_conv =
        kda_conv_ + static_cast<size_t>(req) * kda_cfg_.num_kda_layers *
                        (kda_geo_.conv_committed_bytes / 2);
    DGPP_CUDA_OK(cudaMemsetAsync(
        slot_rec, 0,
        static_cast<size_t>(kda_cfg_.num_kda_layers) * kda_geo_.recurrent_bytes,
        stream_));
    DGPP_CUDA_OK(cudaMemsetAsync(
        slot_conv, 0,
        static_cast<size_t>(kda_cfg_.num_kda_layers) *
            kda_geo_.conv_committed_bytes,
        stream_));
  }
  if (dsa_cfg_.num_dsa_layers > 0) pool_.reset_request(req, stream_);
  session_pos_[static_cast<size_t>(req)] = 0;
  if (mtp_) mtp_pos_[static_cast<size_t>(req)] = 0;
}

GlmDiagnosticModel::PrefillCursor GlmDiagnosticModel::session_prefill_begin(
    int req, const std::vector<int64_t>& prompt, int64_t reserve_tokens, int64_t chunk_tokens,
    const std::vector<int64_t>& boundaries, SnapshotRequest* snap, int64_t attach_position,
    const std::vector<ImageInput>* images) {
  if (req < 0 || req >= max_requests_) throw std::out_of_range("GLM prefill: request slot");
  const int64_t end = static_cast<int64_t>(prompt.size());
  const int64_t align = session_snapshot_align();
  if (end <= 0 || end > max_context_ || reserve_tokens < end || reserve_tokens > max_context_)
    throw std::invalid_argument("GLM prefill: prompt/reservation outside context bounds");
  if (chunk_tokens < align || chunk_tokens > max_tokens_ || chunk_tokens % align != 0)
    throw std::invalid_argument("GLM prefill: invalid aligned chunk budget");
  if (attach_position < 0 || attach_position >= end || attach_position % align != 0)
    throw std::invalid_argument("GLM prefill: invalid attach position");
  for (const auto id : prompt)
    if (id < 0 || id >= cfg_.vocab_size) throw std::invalid_argument("GLM prefill: token id out of range");
  if (images && !images->empty()) {
    if (!vision_) throw std::invalid_argument("GLM: checkpoint has no vision encoder");
    validate_image_inputs(*images, prompt.size());
    for (const auto& im : *images)
      for (int64_t pos = im.offset; pos < im.offset + im.tokens; ++pos)
        if (prompt[static_cast<size_t>(pos)] != 154854)
          throw std::invalid_argument("GLM: image span does not contain image tokens");
  }
  auto cuts = boundaries;
  if (snap && snap->position > attach_position && snap->position < end &&
      snap->position % chunk_tokens == 0) cuts.push_back(snap->position);
  auto cursor = prefill_cursor(req, prompt.data() + attach_position, attach_position,
                               end - attach_position, cuts, snap);
  cursor.images = images;
  cursor.budget_tokens = chunk_tokens;
  if (attach_position == 0) reset_prefill_session(req);
  else if (session_pos_[static_cast<size_t>(req)] != attach_position)
    throw std::logic_error("GLM prefill: slot differs from attached prefix");
  cursor.epoch = ++prefill_epochs_[static_cast<size_t>(req)];
  session_reserve_blocks(req, reserve_tokens);
  if (attach_position > 0 && mtp_) {
    const auto pos = mtp_pos_[static_cast<size_t>(req)];
    cursor.catchup = pos == attach_position - 1;
    if (!cursor.catchup && pos != attach_position)
      throw std::logic_error("GLM prefill: draft differs from attached prefix");
  }
  return cursor;
}

bool GlmDiagnosticModel::session_prefill_advance(PrefillCursor& cursor, int64_t chunk_tokens) {
  if (cursor.req < 0 || cursor.req >= max_requests_ || cursor.next >= cursor.end ||
      prefill_epochs_[static_cast<size_t>(cursor.req)] != cursor.epoch ||
      session_pos_[static_cast<size_t>(cursor.req)] != cursor.next)
    throw std::logic_error("GLM prefill: completed or stale cursor");
  const int64_t budget = chunk_tokens == 0 ? cursor.budget_tokens : chunk_tokens;
  const int64_t align = session_snapshot_align();
  if (budget < align || budget > max_tokens_ || budget % align != 0)
    throw std::invalid_argument("GLM prefill: invalid aligned chunk budget");
  if (cursor.suspended) {
    push_position(cursor.req);
    if (mtp_) push_mtp_position(cursor.req);
    cursor.suspended = false;
  }
  prefill_chunk(cursor, budget);
  const bool done = cursor.next == cursor.end;
  if (!done) {
    // Batched graphs include padding slots. Hide an unfinished prefill's
    // device positions while retaining its host counters and model state.
    DGPP_CUDA_OK(cudaMemsetAsync(d_session_pos_ + cursor.req, 0xff, sizeof(int64_t), stream_));
    if (mtp_) DGPP_CUDA_OK(cudaMemsetAsync(d_mtp_pos_ + cursor.req, 0xff, sizeof(int64_t), stream_));
    cursor.suspended = true;
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  return done;
}

GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_prefill_chunks(
    int req, const int64_t* ids, int64_t start, int64_t count,
    const std::vector<int64_t>& boundaries, SnapshotRequest* snap) {
  auto cursor = prefill_cursor(req, ids, start, count, boundaries, snap);
  while (cursor.next < cursor.end) prefill_chunk(cursor);
  return std::move(cursor.output);
}

GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_prefill(
    int req, const std::vector<int64_t>& prompt_ids,
    const std::vector<int64_t>& boundaries, SnapshotRequest* snap) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_prefill: request slot " +
                           std::to_string(req));
  const int64_t P = static_cast<int64_t>(prompt_ids.size());
  if (P <= 0) throw std::invalid_argument("session_prefill: empty prompt");
  if (P > max_context_)
    throw std::invalid_argument("session_prefill: prompt exceeds the context bound");
  if (P > max_tokens_ && max_tokens_ < kPrefillChunkTokens)
    throw std::invalid_argument(
        "session_prefill: prompt exceeds max_tokens and the model cannot chunk it "
        "(max_tokens is below the prefill chunk)");
  for (int64_t id : prompt_ids)
    if (id < 0 || id >= cfg_.vocab_size)
      throw std::invalid_argument("session_prefill: token id out of range");
  if (dsa_cfg_.num_dsa_layers > 0 &&
      kPrefillChunkTokens % dsa_cfg_.index_kpool != 0)
    throw std::runtime_error(
        "session_prefill: the chunk size broke the kpool-alignment contract");

  reset_prefill_session(req);

  // Chunking (M7's contract, 2026-09-05): cuts at every kPrefillChunkTokens
  // multiple and at the pool-aligned image of every boundary; chunk starts
  // pool-aligned, lengths any (the DSA ring handles a short continuation).
  Outputs out = session_prefill_chunks(req, prompt_ids.data(), 0, P, boundaries, snap);
  session_pos_[static_cast<size_t>(req)] = P;
  push_position(req);
  if (mtp_) {
    mtp_pos_[static_cast<size_t>(req)] = std::max<int64_t>(P - 1, 0);
    push_mtp_position(req);
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  return out;
}

// Several cold prompts as the spans of one walk (2026-09-14, the group
// prefill ported from DeepSeek): every span within max_tokens, the group
// within max_tokens and the mirrors' rows; each slot opened as
// session_prefill opens it (its KDA state and DSA pool reset), the walk's
// KDA scan and DSA attention per span, the mHC, MoE and head sites over
// every row, then per span the position, the draft block over its rows
// and the draft position. Outputs per span: its last row's logits and
// hidden; the walk's route traces ride the first request's.
std::vector<GlmDiagnosticModel::Outputs> GlmDiagnosticModel::session_prefill_group(
    const std::vector<int>& reqs, const std::vector<const std::vector<int64_t>*>& prompts) {
  const int n = static_cast<int>(reqs.size());
  if (n <= 0 || prompts.size() != reqs.size())
    throw std::invalid_argument("session_prefill_group: requests and prompts");
  if (n > kDecodeRows)
    throw std::invalid_argument("session_prefill_group: more spans than the tail mirrors' rows");
  if (dsa_cfg_.num_dsa_layers > 0 && kPrefillChunkTokens % dsa_cfg_.index_kpool != 0)
    throw std::runtime_error("session_prefill_group: the chunk size broke the kpool-alignment contract");
  std::vector<int64_t> ids;
  std::vector<int32_t> span_reqs, span_lens;
  for (int sidx = 0; sidx < n; ++sidx) {
    const int req = reqs[static_cast<size_t>(sidx)];
    if (req < 0 || req >= max_requests_)
      throw std::out_of_range("session_prefill_group: request slot " + std::to_string(req));
    for (int t = 0; t < sidx; ++t)
      if (reqs[static_cast<size_t>(t)] == req) throw std::invalid_argument("session_prefill_group: a request twice in the group");
    const std::vector<int64_t>& P = *prompts[static_cast<size_t>(sidx)];
    if (P.empty()) throw std::invalid_argument("session_prefill_group: empty prompt");
    if (static_cast<int64_t>(P.size()) > max_tokens_)
      throw std::invalid_argument("session_prefill_group: a prompt exceeds the group span limit");
    if (static_cast<int64_t>(P.size()) > max_context_)
      throw std::invalid_argument("session_prefill_group: prompt exceeds the context bound");
    for (int64_t id : P)
      if (id < 0 || id >= cfg_.vocab_size) throw std::invalid_argument("session_prefill_group: token id out of range");
    ids.insert(ids.end(), P.begin(), P.end());
    span_reqs.push_back(req);
    span_lens.push_back(static_cast<int32_t>(P.size()));
  }
  if (static_cast<int64_t>(ids.size()) > max_tokens_)
    throw std::invalid_argument("session_prefill_group: the group's prompts exceed max_tokens");
  // Open every slot as session_prefill does.
  for (int sidx = 0; sidx < n; ++sidx) {
    const int req = span_reqs[static_cast<size_t>(sidx)];
    if (kda_rec_) {
      float* slot_rec = kda_rec_ + static_cast<size_t>(req) * kda_cfg_.num_kda_layers * kda_geo_.recurrent_elems;
      uint16_t* slot_conv =
          kda_conv_ + static_cast<size_t>(req) * kda_cfg_.num_kda_layers * (kda_geo_.conv_committed_bytes / 2);
      DGPP_CUDA_OK(cudaMemsetAsync(slot_rec, 0,
                                   static_cast<size_t>(kda_cfg_.num_kda_layers) * kda_geo_.recurrent_bytes, stream_));
      DGPP_CUDA_OK(cudaMemsetAsync(slot_conv, 0,
                                   static_cast<size_t>(kda_cfg_.num_kda_layers) * kda_geo_.conv_committed_bytes, stream_));
    }
    if (dsa_cfg_.num_dsa_layers > 0) pool_.reset_request(req, stream_);
    session_pos_[static_cast<size_t>(req)] = 0;
    if (mtp_) mtp_pos_[static_cast<size_t>(req)] = 0;
  }
  group_span_reqs_ = span_reqs.data();
  group_span_lens_ = span_lens.data();
  group_num_spans_ = n;
  Outputs all;
  try {
    all = session_run_rows(span_reqs[0], ids, /*token_start=*/0, /*decode_row=*/false);
  } catch (...) {
    group_span_reqs_ = nullptr;
    group_span_lens_ = nullptr;
    group_num_spans_ = 0;
    throw;
  }
  group_span_reqs_ = nullptr;
  group_span_lens_ = nullptr;
  group_num_spans_ = 0;
  std::vector<Outputs> outs(static_cast<size_t>(n));
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  outs[0].routes = std::move(all.routes);
  outs[0].route_biased = std::move(all.route_biased);
  int64_t at = 0;
  for (int sidx = 0; sidx < n; ++sidx) {
    Outputs& o = outs[static_cast<size_t>(sidx)];
    const int req = span_reqs[static_cast<size_t>(sidx)];
    const int64_t P = span_lens[static_cast<size_t>(sidx)];
    o.lm_vocab_begin = all.lm_vocab_begin;
    o.lm_vocab_count = all.lm_vocab_count;
    o.logits.assign(all.logits.begin() + static_cast<std::ptrdiff_t>(sidx) * lm_vocab_count_,
                    all.logits.begin() + static_cast<std::ptrdiff_t>(sidx + 1) * lm_vocab_count_);
    o.final_hidden_bits.assign(all.final_hidden_bits.begin() + static_cast<std::ptrdiff_t>(sidx) * static_cast<std::ptrdiff_t>(H),
                               all.final_hidden_bits.begin() + static_cast<std::ptrdiff_t>(sidx + 1) * static_cast<std::ptrdiff_t>(H));
    session_pos_[static_cast<size_t>(req)] = P;
    push_position(req);
    if (mtp_) {
      // The draft block over the span's rows (row q embeds tok_{q+1}).
      if (P - 1 > 0) mtp_prefill_rows(req, 0, P - 1, ids.data() + at + 1);
      mtp_pos_[static_cast<size_t>(req)] = std::max<int64_t>(P - 1, 0);
      push_mtp_position(req);
    }
    at += P;
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  return outs;
}

GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_prefill_resume(
    int req, const std::vector<int64_t>& suffix_ids,
    const std::vector<int64_t>& boundaries, SnapshotRequest* snap) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_prefill_resume: request slot " +
                            std::to_string(req));
  const int64_t P0 = session_pos_[static_cast<size_t>(req)];
  const int64_t n = static_cast<int64_t>(suffix_ids.size());
  if (P0 <= 0)
    throw std::invalid_argument("session_prefill_resume: the slot is not attached");
  if (P0 % session_snapshot_align() != 0)
    throw std::invalid_argument("session_prefill_resume: the position is not pool-aligned");
  if (n <= 0) throw std::invalid_argument("session_prefill_resume: empty suffix");
  if (P0 + n > max_context_)
    throw std::invalid_argument("session_prefill_resume: prompt exceeds the context bound");
  if (n > max_tokens_ && max_tokens_ < kPrefillChunkTokens)
    throw std::invalid_argument(
        "session_prefill_resume: suffix exceeds max_tokens and the model cannot "
        "chunk it (max_tokens is below the prefill chunk)");
  for (int64_t id : suffix_ids)
    if (id < 0 || id >= cfg_.vocab_size)
      throw std::invalid_argument("session_prefill_resume: token id out of range");
  // A close-time snapshot leaves the draft block one row behind (row P0-1
  // wants tok_{P0}, the suffix's first token): catch it up through the
  // decode-path draft, whose row is exactly that one.
  if (mtp_ && mtp_pos_[static_cast<size_t>(req)] == P0 - 1) {
    stage_image_embeddings(P0, P0 + 1);
    (void)session_draft(req, std::vector<int64_t>{suffix_ids[0]});
  }
  if (mtp_ && mtp_pos_[static_cast<size_t>(req)] != P0)
    throw std::logic_error("session_prefill_resume: the draft block's row counter is "
                           "not at the attach position");
  Outputs out = session_prefill_chunks(req, suffix_ids.data(), P0, n, boundaries, snap);
  session_pos_[static_cast<size_t>(req)] = P0 + n;
  push_position(req);
  if (mtp_) {
    mtp_pos_[static_cast<size_t>(req)] = P0 + n - 1;
    push_mtp_position(req);
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  return out;
}

GlmDiagnosticModel::SessionSnapshotMeta GlmDiagnosticModel::session_snapshot(
    int req, void* dst) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_snapshot: request slot " + std::to_string(req));
  const int64_t pos = session_pos_[static_cast<size_t>(req)];
  if (pos <= 0) throw std::invalid_argument("session_snapshot: the slot is closed");
  if (pos % session_snapshot_align() != 0)
    throw std::invalid_argument("session_snapshot: the position is not pool-aligned");
  if (dst == nullptr) throw std::invalid_argument("session_snapshot: null buffer");
  if (mtp_) {
    const int64_t q = mtp_pos_[static_cast<size_t>(req)];
    if (q != pos && q != pos - 1)
      throw std::logic_error("session_snapshot: the draft block is more than one row behind");
  }
  const int H = cfg_.hidden_size;
  uint8_t* d = static_cast<uint8_t*>(dst);
  if (kda_rec_) {
    const size_t layers = static_cast<size_t>(kda_cfg_.num_kda_layers);
    const size_t rec_bytes = layers * kda_geo_.recurrent_bytes;
    const size_t conv_bytes = layers * kda_geo_.conv_committed_bytes;
    const float* rec = kda_rec_ + static_cast<size_t>(req) * layers * kda_geo_.recurrent_elems;
    const uint16_t* conv =
        kda_conv_ + static_cast<size_t>(req) * layers * (kda_geo_.conv_committed_bytes / 2);
    DGPP_CUDA_OK(cudaMemcpyAsync(d, rec, rec_bytes, cudaMemcpyDeviceToDevice, stream_));
    d += rec_bytes;
    DGPP_CUDA_OK(cudaMemcpyAsync(d, conv, conv_bytes, cudaMemcpyDeviceToDevice, stream_));
    d += conv_bytes;
  }
  SessionSnapshotMeta meta;
  meta.position = pos;
  meta.mtp_position = mtp_ ? mtp_pos_[static_cast<size_t>(req)] : 0;
  if (dsa_cfg_.num_dsa_layers > 0) {
    const size_t tail_bytes = pool_.geometry().tail_bytes_per_request;
    for (int layer = 0; layer < dsa_cfg_.num_dsa_layers; ++layer) {
      const uint8_t* src = static_cast<const uint8_t*>(pool_.tail(layer)) +
                           static_cast<size_t>(req) * tail_bytes;
      DGPP_CUDA_OK(cudaMemcpyAsync(d, src, tail_bytes, cudaMemcpyDeviceToDevice, stream_));
      d += tail_bytes;
    }
    const int64_t block_tokens = dsa_cfg_.block_tokens;
    const int64_t n_full = pos / block_tokens;
    const int32_t* row = pool_.request_table_row(req);
    meta.full_blocks.assign(row, row + n_full);
    pool_.pin_blocks(meta.full_blocks.data(), n_full);
    if (pos % block_tokens != 0) {
      const int32_t b = pool_.acquire_pinned_block();
      if (b < 0) {
        pool_.unpin_blocks(meta.full_blocks.data(), n_full);
        throw std::runtime_error("session_snapshot: cache pool exhausted (the partial block)");
      }
      pool_.copy_block_contents(row[n_full], b, stream_);
      meta.partial_block = b;
    }
  }
  if (mtp_) {
    const uint16_t* hq = mtp_hidden_cache(req) + static_cast<size_t>(pos - 1) * H;
    DGPP_CUDA_OK(cudaMemcpyAsync(d, hq, static_cast<size_t>(H) * 2,
                                 cudaMemcpyDeviceToDevice, stream_));
    d += static_cast<size_t>(H) * 2;
  }
  return meta;
}

GlmDiagnosticModel::SessionSnapshotMeta GlmDiagnosticModel::session_snapshot_post_row0(
    int req, void* dst, int spec_row, int rows_after) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_snapshot_post_row0: request slot " + std::to_string(req));
  if (rows_after < 1 || rows_after >= kSpecRows)
    throw std::invalid_argument("session_snapshot_post_row0: rows after the position");
  const int64_t pos = session_pos_[static_cast<size_t>(req)] - rows_after;  // after row 0
  if (pos <= 0) throw std::invalid_argument("session_snapshot_post_row0: no verified rows");
  if (pos % session_snapshot_align() != 0)
    throw std::invalid_argument(
        "session_snapshot_post_row0: the position after row 0 is not pool-aligned");
  if (spec_row < 0 || spec_row >= kDecodeRows)
    throw std::out_of_range("session_snapshot_post_row0: spec row " + std::to_string(spec_row));
  if (dst == nullptr) throw std::invalid_argument("session_snapshot_post_row0: null buffer");
  // The draft block: its rows ran (counter at P+rows_after, the pre-draft
  // ring is in the rollback snapshot) or have not (counter at P-1, the
  // live ring IS the pre-draft ring).
  bool draft_ring_from_snapshot = false;
  if (mtp_) {
    const int64_t q = mtp_pos_[static_cast<size_t>(req)];
    if (q == pos + rows_after)
      draft_ring_from_snapshot = true;
    else if (q != pos - 1)
      throw std::logic_error(
          "session_snapshot_post_row0: the draft block's counter (" + std::to_string(q) +
          ") is neither before nor after the step's rows (position " + std::to_string(pos) + ")");
  }
  const int H = cfg_.hidden_size;
  uint8_t* d = static_cast<uint8_t*>(dst);
  if (kda_rec_) {
    const size_t layers = static_cast<size_t>(kda_cfg_.num_kda_layers);
    const size_t rec_bytes = layers * kda_geo_.recurrent_bytes;
    const size_t conv_bytes = layers * kda_geo_.conv_committed_bytes;
    const size_t conv_elems = kda_geo_.conv_committed_bytes / 2;
    const float* rec =
        spec_rec_ + static_cast<size_t>(spec_row) * layers * kda_geo_.recurrent_elems;
    const uint16_t* conv = spec_conv_ + static_cast<size_t>(spec_row) * layers * conv_elems;
    DGPP_CUDA_OK(cudaMemcpyAsync(d, rec, rec_bytes, cudaMemcpyDeviceToDevice, stream_));
    d += rec_bytes;
    DGPP_CUDA_OK(cudaMemcpyAsync(d, conv, conv_bytes, cudaMemcpyDeviceToDevice, stream_));
    d += conv_bytes;
  }
  SessionSnapshotMeta meta;
  meta.position = pos;
  meta.mtp_position = mtp_ ? pos - 1 : 0;
  if (dsa_cfg_.num_dsa_layers > 0) {
    const size_t tail_bytes = pool_.geometry().tail_bytes_per_request;
    const size_t ring = spec_tail_ring_elems();
    if (tail_bytes != ring * 2)
      throw std::logic_error("session_snapshot_post_row0: tail ring geometry mismatch");
    for (int layer = 0; layer < dsa_cfg_.num_dsa_layers; ++layer) {
      const uint16_t* src;
      if (layer < main_dsa_layers_) {
        src = spec_tail_ + (static_cast<size_t>(layer) * kDecodeRows +
                            static_cast<size_t>(spec_row)) * ring;
      } else if (draft_ring_from_snapshot) {
        src = mtp_ring_snapshot_ + static_cast<size_t>(req) * ring;
      } else {
        src = static_cast<const uint16_t*>(pool_.tail(layer)) + static_cast<size_t>(req) * ring;
      }
      DGPP_CUDA_OK(cudaMemcpyAsync(d, src, tail_bytes, cudaMemcpyDeviceToDevice, stream_));
      d += tail_bytes;
    }
    const int64_t block_tokens = dsa_cfg_.block_tokens;
    const int64_t n_full = pos / block_tokens;
    const int32_t* row = pool_.request_table_row(req);
    meta.full_blocks.assign(row, row + n_full);
    pool_.pin_blocks(meta.full_blocks.data(), n_full);
    if (pos % block_tokens != 0) {
      // The partial block as it stands: row 1's latent at P is in it, above
      // the position — an attached request overwrites it with its own token
      // at P before anything reads it (positional writes; the visible pool
      // count derives from the query's position), exactly as the stale rows
      // of a rolled-back step are.
      const int32_t b = pool_.acquire_pinned_block();
      if (b < 0) {
        pool_.unpin_blocks(meta.full_blocks.data(), n_full);
        throw std::runtime_error(
            "session_snapshot_post_row0: cache pool exhausted (the partial block)");
      }
      pool_.copy_block_contents(row[n_full], b, stream_);
      meta.partial_block = b;
    }
  }
  if (mtp_) {
    const uint16_t* hq = mtp_hidden_cache(req) + static_cast<size_t>(pos - 1) * H;
    DGPP_CUDA_OK(cudaMemcpyAsync(d, hq, static_cast<size_t>(H) * 2,
                                 cudaMemcpyDeviceToDevice, stream_));
    d += static_cast<size_t>(H) * 2;
  }
  return meta;
}

void GlmDiagnosticModel::session_release_snapshot(const SessionSnapshotMeta& meta) {
  if (dsa_cfg_.num_dsa_layers == 0) return;
  if (!meta.full_blocks.empty())
    pool_.unpin_blocks(meta.full_blocks.data(),
                       static_cast<int64_t>(meta.full_blocks.size()));
  if (meta.partial_block >= 0) pool_.unpin_blocks(&meta.partial_block, 1);
}

void GlmDiagnosticModel::session_attach(int req, const void* src,
                                        const SessionSnapshotMeta& meta) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_attach: request slot " + std::to_string(req));
  if (session_pos_[static_cast<size_t>(req)] != 0)
    throw std::logic_error("session_attach: the slot is open");
  if (meta.position <= 0 || meta.position % session_snapshot_align() != 0)
    throw std::invalid_argument("session_attach: bad snapshot position");
  if (meta.position > max_context_)
    throw std::invalid_argument("session_attach: position exceeds the context bound");
  if (src == nullptr) throw std::invalid_argument("session_attach: null buffer");
  const int H = cfg_.hidden_size;
  const uint8_t* d = static_cast<const uint8_t*>(src);
  if (kda_rec_) {
    const size_t layers = static_cast<size_t>(kda_cfg_.num_kda_layers);
    const size_t rec_bytes = layers * kda_geo_.recurrent_bytes;
    const size_t conv_bytes = layers * kda_geo_.conv_committed_bytes;
    float* rec = kda_rec_ + static_cast<size_t>(req) * layers * kda_geo_.recurrent_elems;
    uint16_t* conv =
        kda_conv_ + static_cast<size_t>(req) * layers * (kda_geo_.conv_committed_bytes / 2);
    DGPP_CUDA_OK(cudaMemcpyAsync(rec, d, rec_bytes, cudaMemcpyDeviceToDevice, stream_));
    d += rec_bytes;
    DGPP_CUDA_OK(cudaMemcpyAsync(conv, d, conv_bytes, cudaMemcpyDeviceToDevice, stream_));
    d += conv_bytes;
  }
  if (dsa_cfg_.num_dsa_layers > 0) {
    pool_.release_request_blocks(req, stream_);
    pool_.reset_request(req, stream_);
    const size_t tail_bytes = pool_.geometry().tail_bytes_per_request;
    for (int layer = 0; layer < dsa_cfg_.num_dsa_layers; ++layer) {
      uint8_t* dstt = static_cast<uint8_t*>(pool_.tail(layer)) +
                      static_cast<size_t>(req) * tail_bytes;
      DGPP_CUDA_OK(cudaMemcpyAsync(dstt, d, tail_bytes, cudaMemcpyDeviceToDevice, stream_));
      d += tail_bytes;
    }
    const int64_t block_tokens = dsa_cfg_.block_tokens;
    const int64_t n_full = meta.position / block_tokens;
    if (static_cast<int64_t>(meta.full_blocks.size()) != n_full)
      throw std::invalid_argument("session_attach: block list does not match the position");
    if (!pool_.share_blocks_into(req, meta.full_blocks.data(), n_full, stream_))
      throw std::runtime_error("session_attach: could not share the prefix blocks");
    if (meta.position % block_tokens != 0) {
      if (meta.partial_block < 0)
        throw std::invalid_argument("session_attach: the snapshot lacks its partial block");
      if (!pool_.ensure_request_blocks(req, meta.position, stream_))
        throw std::runtime_error("session_attach: cache pool exhausted");
      const int32_t* row = pool_.request_table_row(req);
      pool_.copy_block_contents(meta.partial_block, row[n_full], stream_);
    }
  }
  session_pos_[static_cast<size_t>(req)] = meta.position;
  push_position(req);
  if (mtp_) {
    uint16_t* hq = mtp_hidden_cache(req) + static_cast<size_t>(meta.position - 1) * H;
    DGPP_CUDA_OK(cudaMemcpyAsync(hq, d, static_cast<size_t>(H) * 2,
                                 cudaMemcpyDeviceToDevice, stream_));
    d += static_cast<size_t>(H) * 2;
    mtp_pos_[static_cast<size_t>(req)] = meta.mtp_position;
    push_mtp_position(req);
  }
}

// ---------------------------------------------------------------------------
// session_step: one token at slot `req`'s next position.
// ---------------------------------------------------------------------------
GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_step(int req,
                                                             int64_t token_id) {
  return session_verify(req, std::vector<int64_t>{token_id});
}

// ---------------------------------------------------------------------------
// session_verify / session_rollback: T rows in one call, retract the tail.
// ---------------------------------------------------------------------------
GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_verify(
    int req, const std::vector<int64_t>& token_ids) {
  step_timing::Scope tick(step_timing::kStep);
  session_decode_host_prep(req, token_ids, /*upload=*/true);
  const int64_t pos = session_pos_[static_cast<size_t>(req)];
  Outputs out = session_run_rows(req, token_ids, pos, /*decode_row=*/true);
  session_pos_[static_cast<size_t>(req)] += static_cast<int64_t>(token_ids.size());
  push_position(req);
  return out;
}

void GlmDiagnosticModel::push_position(int req) {
  h_session_pos_[req] = session_pos_[static_cast<size_t>(req)];
  DGPP_CUDA_OK(cudaMemcpyAsync(d_session_pos_ + req, h_session_pos_ + req,
                               sizeof(int64_t), cudaMemcpyHostToDevice,
                               stream_));
}

GlmSpecSegments GlmDiagnosticModel::spec_segments(int req,
                                                  int snapshot_row0) {
  GlmSpecSegments segs;
  const auto add = [&](void* dst, const void* snapshots, size_t row_stride,
                       size_t bytes) {
    if (segs.count >= kSpecMaxSegments)
      throw std::logic_error("spec_segments: too many state families");
    segs.seg[segs.count++] = GlmSpecSegment{dst, snapshots, row_stride, bytes};
  };
  if (kda_cfg_.num_kda_layers > 0) {
    const size_t layers = static_cast<size_t>(kda_cfg_.num_kda_layers);
    add(kda_rec_ + static_cast<size_t>(req) * layers * kda_geo_.recurrent_elems,
        spec_rec_ + static_cast<size_t>(snapshot_row0) * layers *
                        kda_geo_.recurrent_elems,
        layers * kda_geo_.recurrent_bytes,
        layers * kda_geo_.recurrent_bytes);
    const size_t conv_elems = kda_geo_.conv_committed_bytes / 2;
    add(kda_conv_ + static_cast<size_t>(req) * layers * conv_elems,
        spec_conv_ + static_cast<size_t>(snapshot_row0) * layers * conv_elems,
        layers * kda_geo_.conv_committed_bytes,
        layers * kda_geo_.conv_committed_bytes);
  }
  const size_t ring = spec_tail_ring_elems();
  for (int layer = 0; layer < main_dsa_layers_; ++layer)
    add(static_cast<uint16_t*>(pool_.tail(layer)) +
            static_cast<size_t>(req) * ring,
        spec_tail_ +
            (static_cast<size_t>(layer) * kDecodeRows + snapshot_row0) * ring,
        ring * 2,
        ring * 2);
  return segs;
}

size_t GlmDiagnosticModel::spec_tail_ring_elems() const {
  return static_cast<size_t>(2) * dsa_cfg_.index_kpool * dsa_cfg_.index_head_dim;
}

void GlmDiagnosticModel::session_rollback(int req, int accepted) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_rollback: request slot " +
                            std::to_string(req));
  const int T = decode_rows_;
  if (accepted < 1 || accepted > T)
    throw std::invalid_argument("session_rollback: accepted rows must be in "
                                "[1, " + std::to_string(T) + "]");
  const int64_t pos = session_pos_[static_cast<size_t>(req)];
  if (pos < T)
    throw std::invalid_argument("session_rollback: no verify to retract");
  if (accepted == T) return;  // every row landed in place already

  // The state after row `accepted-1` lives in snapshot row accepted-1;
  // the request's committed layer slots are contiguous, so each state
  // family rolls back in one copy (the DSA rings per layer).
  const size_t row = static_cast<size_t>(accepted - 1);
  if (kda_cfg_.num_kda_layers > 0) {
    const size_t layers = static_cast<size_t>(kda_cfg_.num_kda_layers);
    DGPP_CUDA_OK(cudaMemcpyAsync(
        kda_rec_ + static_cast<size_t>(req) * layers * kda_geo_.recurrent_elems,
        spec_rec_ + row * layers * kda_geo_.recurrent_elems,
        layers * kda_geo_.recurrent_bytes, cudaMemcpyDeviceToDevice, stream_));
    const size_t conv_elems = kda_geo_.conv_committed_bytes / 2;
    DGPP_CUDA_OK(cudaMemcpyAsync(
        kda_conv_ + static_cast<size_t>(req) * layers * conv_elems,
        spec_conv_ + row * layers * conv_elems,
        layers * kda_geo_.conv_committed_bytes, cudaMemcpyDeviceToDevice,
        stream_));
  }
  const size_t ring = spec_tail_ring_elems();
  for (int layer = 0; layer < main_dsa_layers_; ++layer) {
    uint16_t* tail = static_cast<uint16_t*>(pool_.tail(layer)) +
                     static_cast<size_t>(req) * ring;
    const uint16_t* snap =
        spec_tail_ + (static_cast<size_t>(layer) * kDecodeRows + row) * ring;
    DGPP_CUDA_OK(cudaMemcpyAsync(tail, snap, ring * 2,
                                 cudaMemcpyDeviceToDevice, stream_));
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  session_pos_[static_cast<size_t>(req)] = pos - (T - accepted);
  push_position(req);
}

// The decode step's host-side half (see the header): one implementation
// so the eager, capture, and replay paths validate and stage IDENTICALLY.
void GlmDiagnosticModel::session_decode_host_prep(
    int req, const std::vector<int64_t>& ids, bool upload,
    bool device_positions) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_decode: request slot " +
                            std::to_string(req));
  const int T = static_cast<int>(ids.size());
  if (T < 1 || T > kSpecRows)
    throw std::invalid_argument("session_decode: row count must be in [1, " +
                                std::to_string(kSpecRows) + "]");
  const int64_t pos = session_pos_[static_cast<size_t>(req)];
  if (pos <= 0)
    throw std::invalid_argument("session_decode: no open session on slot " +
                                std::to_string(req));
  for (int64_t id : ids)
    if (id < 0 || id >= cfg_.vocab_size)
      throw std::invalid_argument("session_decode: token id out of range");
  if (pos + T > max_context_)
    throw std::invalid_argument("session_decode: position exceeds the context bound");

  // DSA admission: the block table must cover every row's position
  // BEFORE enqueue_decode (its pos is device state; growth is host
  // control). Blocks a rolled-back row reserved stay reserved — harmless,
  // the scheduler budgets prompt + max_steps up front anyway. The device-
  // driven graph reserved its whole run up front (session_reserve_blocks)
  // and must not grow the table inside a replay.
  if (dsa_cfg_.num_dsa_layers > 0 && !device_positions &&
      !pool_.ensure_request_blocks(req, pos + T, stream_))
    throw std::runtime_error("session_decode: DSA pool exhausted (admission "
                            "budget) — grow the pool or shed requests");

  // Decode-batch metadata: T consecutive rows of one request (time-
  // multiplexed requests). The PINNED members are the upload sources —
  // eager issues the H2Ds, capture records them as memcpy nodes, the
  // replay stage only writes the members (its graph re-uploads). With
  // device positions the rows' positions come off d_session_pos_ by a
  // kernel instead of the h_step_pos_ upload.
  for (int r = 0; r < T; ++r) {
    h_req_ids_[r] = req;
    h_step_pos_[r] = pos + r;
    h_token_[r] = ids[static_cast<size_t>(r)];
  }
  h_req_spans_[0] = 0;
  h_req_spans_[1] = T;
  decode_rows_ = T;
  if (upload) {
    // Kernel uploads, never memcpy nodes: the captured decode graph must be
    // kernels-only (docs/batched_mtp_graph_stall.md; glm_upload_i32).
    glm_upload_i32(h_req_ids_, d_req_ids_, T, stream_);
    if (device_positions) {
      if (d_step_pos_ != nullptr)
        glm_spec_positions(d_session_pos_ + req, T, d_step_pos_, stream_);
    } else {
      glm_upload_i64(h_step_pos_, d_step_pos_, T, stream_);
    }
    glm_upload_i32(h_req_spans_, d_req_spans_, 2, stream_);
  }
}

// ---------------------------------------------------------------------------
// The graph era (DESIGN §6.2). The three replay-side halves the caller
// sequences around ITS bus arm/launch/finish dance — see the header.
// ---------------------------------------------------------------------------
void GlmDiagnosticModel::session_graph_prepare() {
  if (loader_.residency() != GlmResidency::Resident)
    throw std::logic_error(
        "session_graph_prepare: the decode graph needs a resident stack "
        "(a streaming stack rebinds every layer through one slot)");
  int moe_ordinal = 0;
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    if (cfg_.mlps[layer] != GlmMlpKind::Moe) continue;
    const GlmLayerResident& r = stack_layer(layer);
    const GlmLayerBound b = bind_layer(r, /*dense_mlp=*/false);
    if (!moe_)
      moe_ = std::make_unique<GlmMoeLayer>(*b.moe, moe_cfg_, max_tokens_,
                                           kDecodeRows, moe_graph_slots());
    moe_->rebind(*b.moe);
    moe_->prepare_graph_table(moe_ordinal++, stream_);
  }
  if (mtp_) {
    // The draft layer's MoE takes the slot after the main stack's.
    const GlmLayerResident& r = stack_layer(cfg_.mtp_layer());
    const GlmLayerBound b = bind_layer(r, /*dense_mlp=*/false);
    moe_->rebind(*b.moe);
    moe_->prepare_graph_table(n_moe_layers_, stream_);
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}

void GlmDiagnosticModel::session_graph_capture_step(int req,
                                                    int64_t token_id) {
  session_graph_capture_step(req, std::vector<int64_t>{token_id});
}

void GlmDiagnosticModel::session_graph_capture_step(
    int req, const std::vector<int64_t>& ids) {
  session_graph_capture_step(req, ids, /*device_positions=*/false);
}

void GlmDiagnosticModel::session_graph_capture_step(
    int req, const std::vector<int64_t>& ids, bool device_positions,
    bool device_tokens, int feed_rows) {
  if (feed_rows < 0 || feed_rows > kSpecRows ||
      (feed_rows > 0 && static_cast<size_t>(feed_rows) < ids.size()))
    throw std::invalid_argument("session_graph_capture_step: the feed must hold at least the verified rows");
  if (device_tokens && !device_positions)
    throw std::invalid_argument("session_graph_capture_step: device tokens "
                                "need device positions");
  graph_device_positions_ = device_positions;
  graph_device_tokens_ = device_tokens;
  graph_has_draft_ = false;
  graph_batch_requests_ = 0;
  graph_rows_per_request_ = 0;
  graph_feed_rows_ = feed_rows > 0 ? feed_rows : static_cast<int>(ids.size());
  // The slot's scalar variant reads and writes its own persistent feed
  // rows (device_feed), so its feed survives the other slots' replays; a
  // reduced-depth variant (the scheduled verify depth) reads the first
  // ids.size() rows of the same feed.
  step_tokens_ = d_tokens_ + kDecodeRows + static_cast<size_t>(req) * static_cast<size_t>(graph_feed_rows_);
  session_decode_host_prep(req, ids, /*upload=*/true, device_positions);
  const int64_t pos = session_pos_[static_cast<size_t>(req)];
  // The uploads and every launch record; the walk's syncs are skipped
  // inside (capture_mode). NOTHING EXECUTES — no state, no position.
  Outputs out = session_run_rows(req, ids, pos, /*decode_row=*/true,
                                 /*capture_mode=*/true);
  (void)out;  // empty by contract; the caller instantiates the graph
}

void GlmDiagnosticModel::session_graph_capture_batch(int rows_per_request, int requests_arg, int feed_rows) {
  if (feed_rows != 0 && feed_rows != rows_per_request)
    throw std::invalid_argument("session_graph_capture_batch: this family has no reduced-depth feed");
  const int requests = requests_arg > 0 ? requests_arg : max_requests_;
  if (rows_per_request < 1 || rows_per_request > kSpecRows ||
      requests < 1 || requests > max_requests_ || max_requests_ > kDecodeRows ||
      requests * rows_per_request > kDecodeRows)
    throw std::invalid_argument(
        "session_graph_capture_batch: requests * rows_per_request must fit "
        "the fixed decode-row ceiling");
  if (std::none_of(session_pos_.begin(), session_pos_.end(),
                   [](int64_t p) { return p > 0; }))
    throw std::logic_error(
        "session_graph_capture_batch: capture needs one open request");

  const int rows = requests * rows_per_request;
  if (rows > max_tokens_)
    throw std::invalid_argument(
        "session_graph_capture_batch: fixed rows exceed model max_tokens");
  graph_device_positions_ = true;
  graph_device_tokens_ = true;
  graph_has_draft_ = false;
  graph_batch_requests_ = requests;
  graph_rows_per_request_ = rows_per_request;
  graph_feed_rows_ = 0;
  step_tokens_ = d_tokens_ + kDecodeRows;  // the fixed batch: every slot's feed rows
  decode_rows_ = rows;
  for (int q = 0; q < requests; ++q) {
    h_req_spans_[2 * q] = q * rows_per_request;
    h_req_spans_[2 * q + 1] = rows_per_request;
    for (int r = 0; r < rows_per_request; ++r)
      h_req_ids_[q * rows_per_request + r] = q;
  }
  // Kernel uploads, never memcpy nodes (docs/batched_mtp_graph_stall.md).
  glm_upload_i32(h_req_ids_, d_req_ids_, rows, stream_);
  glm_upload_i32(h_req_spans_, d_req_spans_, 2 * requests, stream_);
  glm_spec_positions_batched(d_session_pos_, d_req_ids_, rows,
                             rows_per_request, d_step_pos_, stream_);

  // Device tokens persist from one replay to the next; inactive groups were
  // zero-initialized and admissions seed their group before a replay.
  const std::vector<int64_t> shape(static_cast<size_t>(rows), 0);
  (void)session_run_rows(/*req=*/0, shape, /*token_start=*/0,
                         /*decode_row=*/true, /*capture_mode=*/true,
                         requests);
}

void GlmDiagnosticModel::session_graph_stage_batch() {
  if (graph_batch_requests_ <= 0 || graph_rows_per_request_ <= 0)
    throw std::logic_error(
        "session_graph_stage_batch: no fixed batch was captured");
  for (int q = 0; q < graph_batch_requests_; ++q) {
    h_req_spans_[2 * q] = q * graph_rows_per_request_;
    h_req_spans_[2 * q + 1] = graph_rows_per_request_;
    for (int r = 0; r < graph_rows_per_request_; ++r)
      h_req_ids_[q * graph_rows_per_request_ + r] = q;
  }
}

void GlmDiagnosticModel::session_graph_use_batch_contract(
    int rows_per_request, int requests_arg) {
  const int requests = requests_arg > 0 ? requests_arg : max_requests_;
  if (rows_per_request < 1 || rows_per_request > kSpecRows ||
      requests < 1 || requests > max_requests_ ||
      requests * rows_per_request > kDecodeRows)
    throw std::invalid_argument(
        "session_graph_use_batch_contract: invalid fixed batch shape");
  graph_device_positions_ = true;
  graph_device_tokens_ = true;
  graph_has_draft_ = mtp_;
  graph_batch_requests_ = requests;
  graph_rows_per_request_ = rows_per_request;
  graph_feed_rows_ = 0;
  decode_rows_ = requests * rows_per_request;
  if (mtp_) draft_rows_ = decode_rows_;
}

void GlmDiagnosticModel::session_graph_capture_commit(
    int req, const PickVerdict* device_verdict) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_graph_capture_commit: request slot " +
                            std::to_string(req));
  if (!graph_device_positions_)
    throw std::logic_error(
        "session_graph_capture_commit: the step must be captured with "
        "device positions (the commit advances the device position)");
  glm_spec_commit(device_verdict, decode_rows_, spec_segments(req),
                  d_session_pos_ + req, stream_);
}

void GlmDiagnosticModel::session_graph_capture_commit_batch(
    const PickVerdict* device_verdicts) {
  if (graph_batch_requests_ <= 0 || graph_rows_per_request_ <= 0)
    throw std::logic_error(
        "session_graph_capture_commit_batch: no fixed batch was captured");
  if (device_verdicts == nullptr)
    throw std::invalid_argument(
        "session_graph_capture_commit_batch: null verdicts");
  for (int req = 0; req < graph_batch_requests_; ++req) {
    const int row0 = req * graph_rows_per_request_;
    glm_spec_commit(device_verdicts + req, graph_rows_per_request_,
                    spec_segments(req, row0), d_session_pos_ + req, stream_);
  }
}

void GlmDiagnosticModel::session_graph_capture_verify_next_tokens_batch(
    const PickVerdict* verify_verdicts) {
  if (graph_batch_requests_ <= 0 || graph_rows_per_request_ != 1 || mtp_)
    throw std::logic_error(
        "session_graph_capture_verify_next_tokens_batch: requires the plain "
        "T=1 fixed graph");
  glm_spec_verify_next_tokens_batched(
      verify_verdicts, graph_batch_requests_, graph_rows_per_request_,
      step_tokens_, stream_);
}

void GlmDiagnosticModel::session_reserve_blocks(int req, int64_t tokens) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_reserve_blocks: request slot " +
                            std::to_string(req));
  if (tokens < 1 || tokens > max_context_)
    throw std::invalid_argument("session_reserve_blocks: tokens outside "
                                "[1, max_context]");
  if (dsa_cfg_.num_dsa_layers == 0) return;
  if (!pool_.ensure_request_blocks(req, tokens, stream_))
    throw std::runtime_error(
        "session_reserve_blocks: DSA pool cannot cover " +
        std::to_string(tokens) + " tokens for slot " + std::to_string(req));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}

void GlmDiagnosticModel::session_graph_settle(int req, int accepted, int rows) {
  (void)rows;  // this family verifies one row count
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_graph_settle: request slot " +
                            std::to_string(req));
  if (!graph_device_positions_)
    throw std::logic_error("session_graph_settle: the graph was not captured "
                           "with device positions (use session_graph_collect)");
  const int accepted_bound = graph_batch_requests_ > 0
                                 ? graph_rows_per_request_
                                 : decode_rows_;
  if (accepted < 1 || accepted > accepted_bound)
    throw std::invalid_argument("session_graph_settle: accepted rows outside "
                                "[1, " + std::to_string(accepted_bound) + "]");
  if (session_pos_[static_cast<size_t>(req)] <= 0)
    throw std::invalid_argument("session_graph_settle: no open session");
  // The device advanced its own position in the recorded commit; the
  // mirror follows. No push: the device is the source of truth here. With
  // the draft in the graph the block's counter moved by the same rows.
  session_pos_[static_cast<size_t>(req)] += accepted;
  if (graph_has_draft_) mtp_pos_[static_cast<size_t>(req)] += accepted;
}

void GlmDiagnosticModel::session_graph_stage(int req, int64_t token_id) {
  session_graph_stage(req, std::vector<int64_t>{token_id});
}

void GlmDiagnosticModel::session_graph_stage(int req,
                                             const std::vector<int64_t>& ids) {
  session_decode_host_prep(req, ids, /*upload=*/false,
                           graph_device_positions_);
}

GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_graph_outputs(
    int req) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_graph_outputs: request slot " +
                           std::to_string(req));
  if (session_pos_[static_cast<size_t>(req)] <= 0)
    throw std::invalid_argument("session_graph_outputs: no open session");
  // The caller synced the stream and finished the bus window: the state
  // advanced in place and the logits are stable. With the mirrors on, the
  // graph's D2H nodes joined; with them off (the kernels-only decode
  // graph, docs/batched_mtp_graph_stall.md — a D2H node rides the
  // process-shared copy-engine queue and deadlocked the world-4 loopback
  // at one hardware connection, 10/10), mirror the tail NOW: an eager copy
  // issued after the replay's sync depends on nothing in flight.
  if (!decode_tail_mirrors_) {
    const size_t rows = static_cast<size_t>(decode_rows_);
    const size_t H = static_cast<size_t>(cfg_.hidden_size);
    DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_logits_, logits_,
                                 rows * lm_vocab_count_ * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
    DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_hidden_, normed_, rows * H * 2,
                                 cudaMemcpyDeviceToHost, stream_));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  }
  // Materialize exactly as the eager tail does.
  return session_decode_tail(decode_rows_);
}

GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_graph_collect(
    int req) {
  if (graph_device_positions_)
    throw std::logic_error("session_graph_collect: a device-driven graph "
                           "settles (session_graph_settle)");
  Outputs out = session_graph_outputs(req);
  session_pos_[static_cast<size_t>(req)] += decode_rows_;
  push_position(req);
  return out;
}

// ---------------------------------------------------------------------------
// session_close: retires the slot — blocks return to the free pool (the
// scheduler's admission meters see the capacity again) and the slot may be
// reopened by a later prefill. No collective; safe between any two ops.
// ---------------------------------------------------------------------------
void GlmDiagnosticModel::session_close(int req) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_close: request slot " +
                            std::to_string(req));
  ++prefill_epochs_[static_cast<size_t>(req)];
  if (dsa_cfg_.num_dsa_layers > 0) {
    // The listed attention's guarded gather: an anomaly is a
    // bug survived, logged loudly with its first values. Reported here so
    // the engines need no DSA knowledge; the readback
    // completes before the session resources are released.
    long long a[6] = {0, 0, 0, 0, 0, 0};
    const unsigned long long n = dsa_attn_anomalies(a, /*clear=*/true, stream_);
    if (n != 0)
      DGPP_LOG_ERROR(
          "slot {} closed with {} listed-attention gather(s) out of "
          "range (zero-filled); first: token {} block {} query row {} split {} "
          "list index {} of {}",
          req, n, a[0], a[1], a[2], a[3], a[4], a[5]);
    long long b[6] = {0, 0, 0, 0, 0, 0};
    const unsigned long long m = dsa_select_anomalies(b, /*clear=*/true, stream_);
    if (m != 0)
      DGPP_LOG_ERROR(
          "slot {} closed with {} short select fill(s); first: {} "
          "visible pools, {} below the bin, {} found, {} remaining, bin count "
          "{}, {} candidates found",
          req, m, b[0], b[1], b[2], b[3], b[4], b[5]);
  }
  if (dsa_cfg_.num_dsa_layers > 0)
    pool_.release_request_blocks(req, stream_);
  session_pos_[static_cast<size_t>(req)] = 0;
  push_position(req);
  if (mtp_) {
    mtp_pos_[static_cast<size_t>(req)] = 0;
    push_mtp_position(req);
  }
}

int64_t GlmDiagnosticModel::session_position(int req) const {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_position: request slot " +
                            std::to_string(req));
  return session_pos_[static_cast<size_t>(req)];
}

// ---------------------------------------------------------------------------
// The session's row runner. Modeled on run_stack with three deltas: the
// state is never reset here (prefill owns the one reset, at open), the DSA
// path is chosen by `decode_row` (prefill chunks vs decode positions), and
// only the LAST row's logits/final_hidden are copied out (a prompt-sized
// logits matrix is 100s of MB at real dims; the last row is what greedy
// consumes, and the parity gate compares rows, not matrices).
// ---------------------------------------------------------------------------
// Setting DGPP_SYNC_EAGER synchronizes after each eager stage so errors
// identify the stage that launched the failing kernels. Without it, a
// later CUDA call may report the earlier asynchronous failure.
void GlmDiagnosticModel::debug_sync(const char* what, int layer, bool decode_row) {
  static const bool on = std::getenv("DGPP_SYNC_EAGER") != nullptr;
  if (!on) return;
  const cudaError_t e = cudaStreamSynchronize(stream_);
  if (e != cudaSuccess)
    throw std::runtime_error(std::string("eager row fault after ") + what +
                             " (layer " + std::to_string(layer) + ", " +
                             (decode_row ? "decode" : "prefill") +
                             "): " + cudaGetErrorString(e));
}

GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_run_rows(
    int req, const std::vector<int64_t>& ids, int64_t token_start,
    bool decode_row, bool capture_mode, int batch_requests) {
  const int T = static_cast<int>(ids.size());
  const int H = cfg_.hidden_size;
  const float eps = cfg_.rms_norm_eps;
  const bool batched = batch_requests > 0;

  // The head runs on every row of a prefill chunk although greedy reads
  // only the last: a last-row head would come off the m=1 GEMV while the
  // re-forward reference's comes off the m=T GEMM, and the prefill ==
  // re-forward bitwise gate (glm_tp_test) is worth more than the ~6 ms
  // and 300 MB a 2048-row head costs per chunk.
  if (!gemm_.ensure_plan(T, lm_vocab_count_, H, DType::BF16, GemmOut::F32,
                         H))
    throw std::runtime_error("session: lm head GEMM plan unavailable");

  // The decode rows' token ids ride the PINNED, device-mapped member
  // through a kernel upload — never a memcpy node in a captured graph
  // (docs/batched_mtp_graph_stall.md). Prefill keeps the caller's vector
  // and a plain memcpy (never captured; its syncs amortize over chunks).
  // A device-token capture records no upload: the previous replay's last
  // node (glm_spec_next_tokens) left the fed tokens in d_tokens_.
  if (!capture_mode) step_tokens_ = d_tokens_;  // eager rows: the scratch
  if (!(capture_mode && graph_device_tokens_)) {
    if (decode_row)
      glm_upload_i64(h_token_, step_tokens_, T, stream_);
    else
      DGPP_CUDA_OK(cudaMemcpyAsync(step_tokens_, ids.data(),
                                   static_cast<size_t>(T) * 8,
                                   cudaMemcpyHostToDevice, stream_));
  }
  // The step's first node: when did the GPU actually start this replay?
  // (The bus logs arm -> first collective; this splits it at the graph's
  // own start.) One 1-thread kernel; decode rows only.
  if (decode_row) launch_globaltimer_stamp(h_graph_start_gt_, stream_);
  if (!capture_mode) debug_sync("uploads", -1, decode_row);
  glm_embed_bcast_streams(globals_.embed, step_tokens_, streams_[0], T, H,
                          stream_);
  if (!decode_row) apply_image_embeddings(streams_[0], token_start, T);
  if (!capture_mode) debug_sync("embed", -1, decode_row);

  Outputs out;
  out.routes.reserve(static_cast<size_t>(cfg_.num_hidden_layers));
  uint16_t* cur = streams_[0];
  uint16_t* nxt = streams_[1];
  int dsa_ordinal = 0;
  int kda_ordinal = 0;
  // Decode-path route-trace bookkeeping: enqueue_decode defers its
  // traces (async pinned copies — no round-trip); the entries pushed
  // during the loop are materialized from staging after the final sync.
  int moe_decode_calls = 0;
  int moe_prefill_calls = 0;  // prefill's per-layer trace staging slots

  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const GlmLayerResident& r = stack_layer(layer);
    const GlmLayerBound b = bind_layer(r, cfg_.mlps[layer] == GlmMlpKind::Dense);

    // ---- attention site --------------------------------------------
    GlmMhcWeights hw;
    hw.fn = b.mhc->attn_fn;
    hw.base = b.mhc->attn_base;
    hw.scale = b.mhc->attn_scale;
    // The sites consume the normed row only: no collapsed store (nullptr).
    const bool attn_comb_deferred = launch_mhc_compute_normed(
        cur, hw, mhc_cfg_, nullptr, post_, comb_, mhc_logits_, b.ln1,
        normed_, eps, T, stream_, mhc_counters_, mhc_comb_side_);
    if (attn_comb_deferred) mhc_comb_fork(hw, T);
    uint16_t* attn_out = sub_out_;
    if (boundary_) {
      if (uint16_t* staged = boundary_->stage(T, H)) attn_out = staged;
    }
    if (r.kind == GlmLayerKind::Kda) {
      if (!kda_) {
        kda_ = std::make_unique<KdaLayer>(arena_, gemm_, *b.kda, kda_cfg_,
                                          max_tokens_, gemm_ws_,
                                          gemm_ws_bytes_, eps);
      } else {
        kda_->rebind(*b.kda);
      }
      if (!kda_->prepare(T))
        throw std::runtime_error("session: KDA GEMM plans unavailable");
      // Scalar calls bind one request's layer state. A fixed batch binds
      // this layer in slot 0 and lets the KDA row map select later slots by
      // the full per-request (all-layers) stride.
      const size_t state_req = batched ? 0 : static_cast<size_t>(req);
      float* rec =
          kda_rec_ +
          (state_req * kda_cfg_.num_kda_layers +
           static_cast<size_t>(kda_ordinal)) *
              kda_geo_.recurrent_elems;
      const size_t conv_elems = kda_geo_.conv_committed_bytes / 2;
      uint16_t* conv =
          kda_conv_ +
          (state_req * kda_cfg_.num_kda_layers +
           static_cast<size_t>(kda_ordinal)) * conv_elems;
      // In-place state update: prefill chunks and steps share one
      // recurrence implementation, so the state this enqueue leaves is
      // exactly the state the next row needs (DESIGN §7.1). Speculative
      // rows (decode, T > 1) also leave post-row snapshots for
      // session_rollback; the snapshot row stride spans every KDA layer.
      KdaSpeculativeSinks spec;
      if (decode_row && (T > 1 || batched)) {
        const size_t layers = static_cast<size_t>(kda_cfg_.num_kda_layers);
        spec.recurrent.states =
            spec_rec_ + static_cast<size_t>(kda_ordinal) * kda_geo_.recurrent_elems;
        spec.recurrent.stride_elems =
            static_cast<int64_t>(layers * kda_geo_.recurrent_elems);
        spec.conv.states =
            spec_conv_ + static_cast<size_t>(kda_ordinal) * conv_elems;
        spec.conv.stride_elems = static_cast<int64_t>(layers * conv_elems);
      }
      KdaLayerBatch batch;
      if (batched) {
        batch.rows.request_ids = d_req_ids_;
        batch.rows.positions = d_step_pos_;
        batch.rows.spans = d_req_spans_;
        batch.rows.num_requests = batch_requests;
        batch.recurrent_request_stride_elems =
            static_cast<int64_t>(kda_cfg_.num_kda_layers) *
            kda_geo_.recurrent_elems;
        batch.conv_request_stride_elems =
            static_cast<int64_t>(kda_cfg_.num_kda_layers) * conv_elems;
      }
      if (!decode_row && group_num_spans_ > 0) {
        // A group prefill (2026-09-14): each span's rows scan its own
        // request's recurrent and conv state (the pointers step by rows).
        int64_t row0 = 0;
        for (int sp = 0; sp < group_num_spans_; ++sp) {
          const int len = group_span_lens_[sp];
          const size_t sreq = static_cast<size_t>(group_span_reqs_[sp]);
          float* srec = kda_rec_ + (sreq * kda_cfg_.num_kda_layers + static_cast<size_t>(kda_ordinal)) *
                                       kda_geo_.recurrent_elems;
          uint16_t* sconv = kda_conv_ + (sreq * kda_cfg_.num_kda_layers + static_cast<size_t>(kda_ordinal)) * conv_elems;
          if (!kda_->prepare(len)) throw std::runtime_error("session: KDA GEMM plans unavailable");
          kda_->enqueue(normed_ + static_cast<size_t>(row0) * H, srec, sconv, kda_geo_.conv_hist,
                        attn_out + static_cast<size_t>(row0) * H, len, stream_);
          row0 += len;
        }
      } else {
        kda_->enqueue(normed_, rec, conv, kda_geo_.conv_hist, attn_out, T,
                      stream_, decode_row ? &prefetch_ : nullptr, spec, batch);
      }
      ++kda_ordinal;
    } else {
      if (!dsa_) {
        dsa_ = std::make_unique<DsaLayer>(
            gemm_, *b.dsa, dsa_cfg_, max_tokens_, pool_.max_token_slots(),
            dsa_scratch_,
            DsaLayer::scratch_bytes(dsa_cfg_, max_tokens_,
                                    pool_.max_token_slots()),
            gemm_ws_, gemm_ws_bytes_);
      } else {
        dsa_->rebind(*b.dsa);
      }
      if (!dsa_->prepare(T))
        throw std::runtime_error("session: DSA GEMM plans unavailable");
      if (decode_row) {
        // The decode table's T rows all serve `req` (staged in
        // session_decode_host_prep); num_requests=1 — time-multiplexed
        // steps. Speculative rows leave post-row ring snapshots.
        void* tail_snaps = (T > 1 || batched)
                               ? spec_tail_ +
                                     static_cast<size_t>(dsa_ordinal) *
                                         kDecodeRows * spec_tail_ring_elems()
                               : nullptr;
        dsa_->enqueue_decode(normed_, pool_, dsa_ordinal, d_req_ids_,
                             d_step_pos_, d_req_spans_,
                             batched ? batch_requests : 1, T, attn_out,
                             stream_, &prefetch_, tail_snaps);
      } else if (group_num_spans_ > 0) {
        // A group prefill: each span attends over and publishes to its own
        // request's cache, its selection at its rows' offset in the scratch.
        int64_t row0 = 0;
        for (int sp = 0; sp < group_num_spans_; ++sp) {
          const int len = group_span_lens_[sp];
          if (!dsa_->prepare(len)) throw std::runtime_error("session: DSA GEMM plans unavailable");
          dsa_->enqueue_prefill(normed_ + static_cast<size_t>(row0) * H, pool_, dsa_ordinal, group_span_reqs_[sp],
                                /*token_start=*/0, len, attn_out + static_cast<size_t>(row0) * H, stream_,
                                static_cast<int>(row0));
          row0 += len;
        }
      } else {
        dsa_->enqueue_prefill(normed_, pool_, dsa_ordinal, /*req=*/req,
                              token_start, T, attn_out, stream_);
      }
      ++dsa_ordinal;
    }
    const bool dense_mlp = cfg_.mlps[layer] == GlmMlpKind::Dense;
    if (decode_row) prefetch_ffn_side(b, dense_mlp);
    if (boundary_) {
      // A captured sync is an error — under capture the fold is a
      // recorded node and the stream order IS the drain.
      if (!capture_mode) {
        step_timing::Scope drain(step_timing::kFoldDrain);
        debug_sync("attention", layer, decode_row);
        DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      }
      boundary_->reduce(attn_out, T, H);
      if (decode_row && layer < gr_probe_layers_)
        boundary_->probe(T, kGrProbeCols);
    }
    if (attn_comb_deferred) DGPP_CUDA_OK(cudaStreamWaitEvent(stream_, mhc_join_, 0));
    launch_mhc_stream_update(post_, comb_, attn_out, cur, nxt, mhc_cfg_, T,
                             stream_);
    std::swap(cur, nxt);

    // ---- feed-forward site -----------------------------------------
    GlmMhcWeights fw;
    fw.fn = b.mhc->ffn_fn;
    fw.base = b.mhc->ffn_base;
    fw.scale = b.mhc->ffn_scale;
    const bool ffn_comb_deferred = launch_mhc_compute_normed(
        cur, fw, mhc_cfg_, nullptr, post_, comb_, mhc_logits_, b.ln2,
        normed_, eps, T, stream_, mhc_counters_, mhc_comb_side_);
    if (ffn_comb_deferred) mhc_comb_fork(fw, T);
    uint16_t* ffn_out = sub_out_;
    if (boundary_) {
      if (uint16_t* staged = boundary_->stage(T, H)) ffn_out = staged;
    }
    if (dense_mlp) {
      enqueue_dense_mlp(normed_, ffn_out, b.dense, T, stream_);
    } else {
      if (!moe_) {
        moe_ = std::make_unique<GlmMoeLayer>(*b.moe, moe_cfg_, max_tokens_,
                                             kDecodeRows);
      } else {
        moe_->rebind(*b.moe);
      }
      if (decode_row) {
        // The decode fast path: the MoE runs from the
        // DEVICE-side route — no router round-trip, no host
        // segmentation, no per-segment H2D. Traces ride async copies
        // into this layer's pinned staging slot; the placeholder route
        // entry is filled after the final sync below.
        MoeTraceStaging trace;
        const size_t lay = static_cast<size_t>(moe_decode_calls);
        trace.ids = moe_trace_ids_ + lay * kDecodeRows * moe_cfg_.top_k;
        trace.weights =
            moe_trace_weights_ + lay * kDecodeRows * moe_cfg_.top_k;
        trace.biased =
            moe_trace_biased_ + lay * kDecodeRows * moe_cfg_.n_experts;
        // Capture passes the layer's OWN graph table slot so the
        // recorded upload node replays THIS layer's expert views; the
        // eager path uploads from the layer's guarded ring instead.
        moe_->enqueue_decode(normed_, ffn_out, T,
                             decode_route_traces_ ? &trace : nullptr, stream_,
                             capture_mode ? moe_decode_calls : -1);
        GlmRouteTraceLayer route;
        route.layer_idx = static_cast<uint32_t>(layer);
        route.top_k = static_cast<uint32_t>(moe_cfg_.top_k);
        route.tokens = static_cast<uint64_t>(T);
        out.routes.push_back(std::move(route));  // ids/weights: post-sync
        out.route_biased.emplace_back();
        ++moe_decode_calls;
      } else {
        // Prefill: the device-segmented grouped path — no
        // host sync per layer; the routing traces land in this layer's
        // pinned staging and are materialized after the final sync.
        MoeTraceStaging trace;
        const size_t lay = static_cast<size_t>(moe_prefill_calls);
        const size_t mt = static_cast<size_t>(max_tokens_);
        trace.ids = moe_prefill_trace_ids_ + lay * mt * moe_cfg_.top_k;
        trace.weights = moe_prefill_trace_weights_ + lay * mt * moe_cfg_.top_k;
        trace.biased = moe_prefill_trace_biased_ + lay * mt * moe_cfg_.n_experts;
        moe_->enqueue_prefill(normed_, ffn_out, T, &trace, stream_);
        GlmRouteTraceLayer route;
        route.layer_idx = static_cast<uint32_t>(layer);
        route.top_k = static_cast<uint32_t>(moe_cfg_.top_k);
        route.tokens = static_cast<uint64_t>(T);
        out.routes.push_back(std::move(route));  // ids/weights: post-sync
        out.route_biased.emplace_back();
        ++moe_prefill_calls;
      }
    }
    if (decode_row) prefetch_attention_side(layer + 1);
    if (boundary_) {
      // A captured sync is an error — under capture the fold is a
      // recorded node and the stream order IS the drain.
      if (!capture_mode) {
        step_timing::Scope drain(step_timing::kFoldDrain);
        debug_sync("ffn", layer, decode_row);
        DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      }
      boundary_->reduce(ffn_out, T, H);
    }
    if (ffn_comb_deferred) DGPP_CUDA_OK(cudaStreamWaitEvent(stream_, mhc_join_, 0));
    launch_mhc_stream_update(post_, comb_, ffn_out, cur, nxt, mhc_cfg_, T,
                             stream_);
    std::swap(cur, nxt);
  }

  // ---- head: mean over streams, final norm, lm head -----------------
  launch_mhc_final_mean(cur, collapsed_, mhc_cfg_, T, stream_);
  // The draft block's hnorm input is THIS (pre-final-norm) hidden: keep
  // it per position. Decode rows scatter by device position (the graph
  // replays at moving positions); prefill chunks are contiguous.
  if (mtp_) {
    if (decode_row && batched) {
      glm_rows_scatter_bf16_batched(
          collapsed_, d_req_ids_, d_step_pos_, mtp_hidden_,
          static_cast<int64_t>(max_context_) * H, T, H, stream_);
    } else if (decode_row) {
      glm_rows_scatter_bf16(collapsed_, d_step_pos_, mtp_hidden_cache(req), T,
                            H, stream_);
    } else if (group_num_spans_ > 0) {
      int64_t row0 = 0;
      for (int sp = 0; sp < group_num_spans_; ++sp) {
        const int len = group_span_lens_[sp];
        DGPP_CUDA_OK(cudaMemcpyAsync(mtp_hidden_cache(group_span_reqs_[sp]), collapsed_ + static_cast<size_t>(row0) * H,
                                     static_cast<size_t>(len) * H * 2, cudaMemcpyDeviceToDevice, stream_));
        row0 += len;
      }
    } else {
      uint16_t* cache = mtp_hidden_cache(req);
      DGPP_CUDA_OK(cudaMemcpyAsync(
          cache + static_cast<size_t>(token_start) * H, collapsed_,
          static_cast<size_t>(T) * H * 2, cudaMemcpyDeviceToDevice, stream_));
    }
  }
  glm_rmsnorm_bf16(collapsed_, globals_.final_norm, normed_, T, H, eps,
                   stream_);
  gemm_.matmul(normed_, globals_.lm_head, logits_, T, lm_vocab_count_, H,
               DType::BF16, GemmOut::F32, H, gemm_ws_, gemm_ws_bytes_,
               stream_);
  // The prefetch side stream rejoins here: a capture must end with every
  // forked stream joined, and the eager tail's sync below should cover
  // the prefetches too (they read weights, nothing else).
  if (decode_row) prefetch_.join(stream_);
  // The decode tail's rows ride D2H into pinned mirrors so the host never
  // touches the managed activations — see h_tail_logits_'s comment for the
  // 9 ms stall that bought this. Under CAPTURE the copies would be memcpy
  // nodes, which the decode graph must not carry (the process-shared
  // copy-engine queue, docs/batched_mtp_graph_stall.md): a capture records
  // them only while the mirrors are on, and a device-pick consumer turns
  // them off (session_graph_outputs then copies eagerly after a replay).
  // An eager step always mirrors — it syncs right below.
  if (!decode_row || decode_tail_mirrors_ || !capture_mode) {
    // Decode rows: all T rows (T <= kDecodeRows). Prefill chunks: the
    // LAST row only, into mirror row 0 (a prompt-sized logits matrix is
    // 100s of MB; greedy needs one row).
    const size_t first = decode_row ? 0 : static_cast<size_t>(T - 1);
    const size_t rows = decode_row ? static_cast<size_t>(T) : 1;
    if (!decode_row && group_num_spans_ > 0) {
      // A group prefill: every span's last row, span-major, into the
      // mirrors' rows (the group is bounded by the mirrors' kDecodeRows).
      int64_t row0 = 0;
      for (int sp = 0; sp < group_num_spans_; ++sp) {
        const size_t last = static_cast<size_t>(row0 + group_span_lens_[sp] - 1);
        DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_logits_ + static_cast<size_t>(sp) * lm_vocab_count_,
                                     logits_ + last * lm_vocab_count_, lm_vocab_count_ * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream_));
        DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_hidden_ + static_cast<size_t>(sp) * H, normed_ + last * H, H * 2,
                                     cudaMemcpyDeviceToHost, stream_));
        row0 += group_span_lens_[sp];
      }
    } else {
      DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_logits_,
                                   logits_ + first * lm_vocab_count_,
                                   rows * lm_vocab_count_ * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_));
      DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_hidden_, normed_ + first * H,
                                   rows * H * 2, cudaMemcpyDeviceToHost,
                                   stream_));
    }
  }

  // Capture ends HERE: nothing executed, so there is nothing to sync
  // or materialize — the caller ends the capture, instantiates, and the
  // first replay performs this step for real.
  if (capture_mode) return Outputs{};

  {
    step_timing::Scope sync_tick(step_timing::kFinalSync);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  }

  // The decode tail (route traces from the pinned staging the loop's
  // async copies just joined, last-row logits/hidden) is one shared
  // materializer — the eager step and the graph-era collect produce
  // byte-identical Outputs through it.
  if (decode_row) return session_decode_tail(T);

  // Prefill: the route traces from this chunk's pinned staging — the
  // layers' async D2H copies joined at the sync above — into the
  // placeholder entries the loop pushed, one per MoE layer in layer order.
  {
    const size_t K = static_cast<size_t>(moe_cfg_.top_k);
    const size_t E = static_cast<size_t>(moe_cfg_.n_experts);
    const size_t mt = static_cast<size_t>(max_tokens_);
    const size_t rows = static_cast<size_t>(T);
    for (size_t li = 0; li < out.routes.size() && li < static_cast<size_t>(moe_prefill_calls); ++li) {
      const int32_t* ids = moe_prefill_trace_ids_ + li * mt * K;
      const float* ws = moe_prefill_trace_weights_ + li * mt * K;
      const float* bs = moe_prefill_trace_biased_ + li * mt * E;
      out.routes[li].ids.assign(ids, ids + rows * K);
      out.routes[li].weights.assign(ws, ws + rows * K);
      out.route_biased[li].assign(bs, bs + rows * E);
    }
  }

  // Last row only (see the runner's header note) — from the pinned
  // mirrors' row 0, which the D2H above filled with row T-1.
  const size_t tail_rows = (!decode_row && group_num_spans_ > 0) ? static_cast<size_t>(group_num_spans_) : 1;
  out.final_hidden_bits.assign(h_tail_hidden_, h_tail_hidden_ + tail_rows * H);
  out.logits.assign(h_tail_logits_, h_tail_logits_ + tail_rows * lm_vocab_count_);
  out.lm_vocab_begin = lm_vocab_begin_;
  out.lm_vocab_count = lm_vocab_count_;
  return out;
}

// The decode tail shared by the eager step and the graph-era collect
// (see the header): routes materialized from the per-MoE-layer pinned
// staging — the walk (eager) or the replay's D2H nodes (graph) filled
// them — plus the last row's logits/final_hidden off the stable device
// buffers. The route shape (one entry per MoE layer, actual layer
// indices) is exactly the eager path's.
GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_decode_tail(int T) {
  const int H = cfg_.hidden_size;
  const size_t K = static_cast<size_t>(moe_cfg_.top_k);
  const size_t E = static_cast<size_t>(moe_cfg_.n_experts);
  const size_t rows = static_cast<size_t>(T);
  Outputs out;
  int moe_ordinal = 0;
  // Traces off: the staging was never written this step; report no routes
  // rather than stale ones.
  for (int layer = 0; decode_route_traces_ && layer < cfg_.num_hidden_layers;
       ++layer) {
    if (cfg_.mlps[layer] != GlmMlpKind::Moe) continue;
    const size_t lay = static_cast<size_t>(moe_ordinal);
    const int32_t* ids =
        moe_trace_ids_ + lay * kDecodeRows * moe_cfg_.top_k;
    const float* ws =
        moe_trace_weights_ + lay * kDecodeRows * moe_cfg_.top_k;
    const float* bs =
        moe_trace_biased_ + lay * kDecodeRows * moe_cfg_.n_experts;
    GlmRouteTraceLayer route;
    route.layer_idx = static_cast<uint32_t>(layer);
    route.top_k = static_cast<uint32_t>(moe_cfg_.top_k);
    route.tokens = static_cast<uint64_t>(T);
    route.ids.assign(ids, ids + rows * K);
    route.weights.assign(ws, ws + rows * K);
    out.routes.push_back(std::move(route));
    out.route_biased.emplace_back(bs, bs + rows * E);
    ++moe_ordinal;
  }
  // From the pinned mirrors the step's D2H copies filled (the caller
  // synced), never from the managed activations. Every row: a verify's
  // consumer compares each row's argmax with the next row's token.
  out.final_hidden_bits.assign(h_tail_hidden_, h_tail_hidden_ + rows * H);
  out.logits.assign(h_tail_logits_,
                    h_tail_logits_ + rows * static_cast<size_t>(lm_vocab_count_));
  out.lm_vocab_begin = lm_vocab_begin_;
  out.lm_vocab_count = lm_vocab_count_;
  return out;
}

// Prefill chunks emit one route entry per MoE layer per chunk; the
// reference emits one per layer per forward. Merge same-layer entries
// along the token axis so Outputs.routes has the reference's shape.
void GlmDiagnosticModel::session_merge_routes(Outputs* out,
                                              Outputs&& chunk) const {
  for (size_t i = 0; i < chunk.routes.size(); ++i) {
    if (!out->routes.empty() &&
        out->routes.back().layer_idx == chunk.routes[i].layer_idx) {
      GlmRouteTraceLayer& dst = out->routes.back();
      const GlmRouteTraceLayer& src = chunk.routes[i];
      dst.ids.insert(dst.ids.end(), src.ids.begin(), src.ids.end());
      dst.weights.insert(dst.weights.end(), src.weights.begin(),
                         src.weights.end());
      dst.tokens += src.tokens;
      out->route_biased.back().insert(out->route_biased.back().end(),
                                      chunk.route_biased[i].begin(),
                                      chunk.route_biased[i].end());
    } else {
      out->routes.push_back(std::move(chunk.routes[i]));
      out->route_biased.push_back(std::move(chunk.route_biased[i]));
    }
  }
}

GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_prefill_images(
    int req, const std::vector<int64_t>& prompt, const std::vector<ImageInput>& images) {
  return session_prefill_images(req, prompt, images, {}, nullptr);
}

GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_prefill_images(
    int req, const std::vector<int64_t>& prompt, const std::vector<ImageInput>& images,
    const std::vector<int64_t>& boundaries, SnapshotRequest* snap) {
  return session_prefill_with_images(req, prompt, images, boundaries, snap, false);
}

GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_prefill_resume_images(
    int req, const std::vector<int64_t>& suffix, const std::vector<ImageInput>& images,
    const std::vector<int64_t>& boundaries, SnapshotRequest* snap) {
  return session_prefill_with_images(req, suffix, images, boundaries, snap, true);
}

GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_prefill_with_images(
    int req, const std::vector<int64_t>& ids, const std::vector<ImageInput>& images,
    const std::vector<int64_t>& boundaries, SnapshotRequest* snap, bool resume) {
  if (!vision_) throw std::invalid_argument("GLM: checkpoint has no vision encoder");
  if (req < 0 || req >= max_requests_) throw std::out_of_range("GLM image prefill: request slot");
  const int64_t start = resume ? session_pos_[static_cast<size_t>(req)] : 0;
  if (start < 0) throw std::invalid_argument("GLM image prefill: closed session");
  validate_image_inputs(images, start + ids.size());
  for (const auto& im : images)
    for (int64_t pos = std::max(start, im.offset); pos < im.offset + im.tokens; ++pos)
      if (ids[static_cast<size_t>(pos - start)] != 154854)
        throw std::invalid_argument("GLM: image span does not contain image tokens");
  prefill_images_ = &images;
  try {
    auto out = resume ? session_prefill_resume(req, ids, boundaries, snap)
                      : session_prefill(req, ids, boundaries, snap);
    prefill_images_ = nullptr;
    image_embeddings_ = nullptr;
    return out;
  } catch (...) {
    prefill_images_ = nullptr;
    image_embeddings_ = nullptr;
    throw;
  }
}
void GlmDiagnosticModel::stage_image_embeddings(int64_t first, int64_t end) {
  image_window_first_ = first;
  image_window_end_ = end;
  image_embeddings_ = prefill_images_ && !prefill_images_->empty()
      ? vision_->stage(*prefill_images_, first, end) : nullptr;
}
void GlmDiagnosticModel::apply_image_embeddings(uint16_t* dst, int64_t first, int rows,
                                                const uint16_t* mtp_norm) {
  if (!prefill_images_) return;
  const int h = cfg_.hidden_size;
  for (const auto& im : *prefill_images_) {
    const int64_t begin = std::max(first, im.offset),
                  end = std::min(first + rows, im.offset + im.tokens);
    if (end > begin) {
      if (!image_embeddings_ || begin < image_window_first_ || end > image_window_end_)
        throw std::logic_error("GLM: image consumer escaped its staged window");
      const auto* source = image_embeddings_ + (begin - image_window_first_) * h;
      const int n = static_cast<int>(end - begin);
      if (mtp_norm) {
        glm_rmsnorm_bf16(source, mtp_norm, normed_, n, h, cfg_.rms_norm_eps, stream_);
        DGPP_CUDA_OK(cudaMemcpy2DAsync(dst + (begin - first) * 2 * h, 2 * h * sizeof(uint16_t),
                                       normed_, h * sizeof(uint16_t), h * sizeof(uint16_t), n,
                                       cudaMemcpyDeviceToDevice, stream_));
      } else
        vision_broadcast(source, dst + (begin - first) * 4 * h, n, h, stream_);
    }
  }
}

}  // namespace dgpp
