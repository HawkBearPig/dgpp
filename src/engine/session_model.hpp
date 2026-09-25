#pragma once
// Shared session state and execution protocol for model implementations.
// SessionModel manages request positions, token feeds, prefill cuts,
// verification and rollback, prefix snapshots, graph staging and MTP state.
// The derived family supplies the layer walk and its state operations.
// See engine/decode_outputs.hpp for the engine-facing types.
//
// clang-format off
// A family Derived : SessionModel<Derived> provides:
//   static constexpr int prefill_chunk_tokens();
//   Outputs run_rows(const RowRun&);            // the walk (see RowRun)
//   void reset_slot_state(int req);             // zero its state families, release its blocks
//   GlmSpecSegments spec_segments(int req, int row0) const;  // the rollback table
//   size_t snapshot_state_bytes() const;        // the non-draft state families' bytes
//   size_t draft_state_bytes() const;           // the draft block's own state (0: none)
//   void write_state_snapshot(int req, uint8_t* dst, int spec_row);   // live (spec_row < 0) or from the spec rows
//   void write_draft_snapshot(int req, uint8_t* dst, bool live, int64_t pos);
//   void read_state_snapshot(int req, const uint8_t* src);
//   void read_draft_snapshot(int req, const uint8_t* src);
//   bool has_pool() const;  Pool& pool();       // the paged pool (PagedBlockTable's protocol + copy_block_contents)
//   void graph_prepare();                       // the layers' graph tables, before a capture
//   void mtp_run_rows(int req, const int64_t* tokens, int64_t first_pos, int T, bool decode_row,
//                     bool capture, int head_rows, int batch_requests);
//   void snapshot_draft_state(int req);  void restore_draft_state(int req);  // around an in-graph draft
//   static constexpr bool kDraftChain;          // depth >= 2 drafting wired (the chain rows)
//   static constexpr bool kVerifyConfidence;    // optional (default false here): the draft emits a
//                                               // per-position acceptance logit — confidence_rows()
//                                               // entries per slot at device_confidence() + slot * rows
//                                               // (engine/verify_schedule.hpp, the scheduled verify depth)
//   const uint16_t* draft_hidden_rows() const;  // the block's output rows [T, draft_width] of its last run
//   void snapshot_chain_state(int req);  void restore_chain_state(int req);  // around the chain rows
//   void mtp_select_block(int index);          // optional: catch-up 0, chain index+1
// clang-format on
// and calls init_session(params) from its constructor once its loader is
// up (the base's buffers need the vocab slice). The walk reads the
// base's staged inputs through begin_run() and hands its results to
// finish_run().
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/prefill_progress.hpp"
#include "engine/boundary_reducer.hpp"
#include "engine/cache_planes.hpp"
#include "engine/decode_outputs.hpp"
#include "kernels/gemm.hpp"
#include "kernels/glm_spec.hpp"
#include "kernels/pick.hpp"
#include "kernels/qwen_mtp.hpp"

namespace dgpp {

// The fixed batches the engines carry stay within the bounds the kernels
// mirror: the pick tables' rows, and the GEMM interface's decode lowering (every
// row of a batch keeps the scalar reduction order).
static_assert(kDecodeRowsMax == kPickMaxRows, "the pick kernels' row bound mirrors the decode-row maximum");
static_assert(kDecodeRowsMax <= kGemmDecodeLoweringRows,
              "the GEMM seam must lower every fixed batch to the row-independent GEMV core");

struct SessionParams {
  int max_tokens = 0;
  int64_t max_cache_tokens = 0;  // already rounded to blocks by the family
  int rank = 0, world = 1;
  BoundaryReducer* boundary = nullptr;
  int max_requests = 1;
  // The fixed decode batch's row ceiling — max_concurrency x (1 + mtp
  // depth), the serving app's derivation (engine/decode_outputs.hpp): the
  // per-row scratch, the token feeds and the draft windows are sized to
  // it. 0 or less than kDecodeRows: kDecodeRows (the floor); at most
  // kDecodeRowsMax.
  int decode_rows = 0;
  bool mtp = false;
  int64_t vocab_size = 0;
  int hidden = 0;              // the final hidden's width (DecodeOutputs::final_hidden_bits rows)
  int lm_vocab_begin = 0, lm_vocab_count = 0;
  int64_t max_position_embeddings = 0;
  int block_tokens = 0;        // the pool's block (0: no pool)
  int snapshot_align = 1;      // prefix snapshot positions are multiples of this
  int draft_width = 0;         // the draft block's input hidden width (mtp)
  int32_t eos = -1;
};

template <class Derived>
class SessionModel : public PrefillReporting {
 public:
  // No confidence head unless the family says otherwise (the graph engine
  // schedules the verify depth only for a family that shadows this).
  static constexpr bool kVerifyConfidence = false;
  struct Outputs : DecodeOutputs {
    std::vector<std::vector<uint16_t>> layer_states;  // per layer, when captured
    std::vector<std::vector<int32_t>> route_ids;      // per MoE layer [T, top_k], ascending
    std::vector<std::vector<float>> route_weights;    // per MoE layer [T, top_k]
    // Per indexed DSA layer [T, max_selected] (-1 padded), when captured
    // (the full GLM-5.3's selections; empty for the other families).
    std::vector<std::vector<int32_t>> dsa_selections;
  };
  struct SessionSnapshotMeta {
    int64_t position = 0;
    int64_t mtp_position = 0;
    std::vector<int32_t> full_blocks;  // physical ids, pinned by the entry
    int32_t partial_block = -1;        // the entry's own copy of the partial block
  };
  struct SnapshotRequest {
    int64_t position = 0;
    void* dst = nullptr;
    SessionSnapshotMeta* meta = nullptr;
    bool taken = false;
    SnapshotRequest* next = nullptr;  // additional cuts, owned by the caller
  };
  // One walk over T rows.
  struct RowRun {
    int req = 0;                   // the request (scalar); 0 for the fixed batch
    const int64_t* ids = nullptr;  // host [T] (prefill; eager decode validation)
    int T = 0;
    int64_t pos0 = 0;              // prefill: the chunk's first position
    bool decode = false;           // decode rows: metadata staged by decode_host_prep
    bool all_rows = false;         // every row's logits (else the last row; decode: all rows)
    bool capture_layers = false;
    bool capture = false;          // graph capture: no syncs, no copies, nothing executes
    int batch_requests = 0;        // > 0: the fixed slot-major batch (decode)
    bool snapshots = false;        // decode: leave every row's post-state in the spec rows
    // Prefill: the call's first and last chunk (a family with a replay
    // segment — DeepSeek-V4.1's bounded decoder — carries a tail between
    // the chunks of one call and runs its segment on the last).
    bool first_chunk = true;
    bool last_chunk = true;
    // The group prefill: several requests' cold prompts as the spans of
    // one walk (host arrays of num_spans; 0: the scalar form above). T is
    // the spans' total and `ids` their prompts concatenated.
    const int32_t* span_reqs = nullptr;
    const int64_t* span_pos0 = nullptr;
    const int32_t* span_lens = nullptr;
    int num_spans = 0;
  };
  // The staged inputs of a walk (begin_run).
  struct RowInputs {
    const int64_t* tokens = nullptr;
    const int64_t* pos = nullptr;
    const int32_t* req_ids = nullptr;
    const int32_t* spans = nullptr;
    int num_requests = 1;
    bool batched = false;
  };

  SessionModel() = default;
  virtual ~SessionModel();
  SessionModel(const SessionModel&) = delete;
  SessionModel& operator=(const SessionModel&) = delete;

  // ---- the engine contract --------------------------------------------------
  Outputs session_prefill(int req, const std::vector<int64_t>& prompt_ids,
                          const std::vector<int64_t>& boundaries = {}, SnapshotRequest* snap = nullptr);
  // Several requests' cold prompts in one walk (the spans of one run:
  // the dense sites, the MoE and the head over every row, the family's
  // per-request pieces per span). Each prompt within the family's span
  // limit (prefill_group_span_limit) and the group within max_tokens; no
  // chunking, no snapshots. Returns each request's last-row outputs, in
  // order. A family with no span support refuses groups of more than one.
  std::vector<Outputs> session_prefill_group(const std::vector<int>& reqs,
                                             const std::vector<const std::vector<int64_t>*>& prompts);
  int64_t prefill_group_span_limit() const { return 0; }  // the family widens it
  Outputs session_prefill_resume(int req, const std::vector<int64_t>& suffix_ids,
                                 const std::vector<int64_t>& boundaries, SnapshotRequest* snap = nullptr);
  // A caller-owned continuation. `ids` and `snap` must outlive the cursor.
  // Only families advertising kResumablePrefill may interleave these
  // chunks with other requests; some families carry shared span scratch.
  static constexpr bool kResumablePrefill = false;
  struct PrefillCursor {
    int req = -1;
    const int64_t* ids = nullptr;
    int64_t start = 0, end = 0, next = 0;
    int64_t budget_tokens = 0;
    std::vector<int64_t> cuts;
    size_t cut_index = 0;
    bool span_start = true;
    bool suspended = false;
    SnapshotRequest* snap = nullptr;
    Outputs output;
  };
  PrefillCursor session_prefill_begin(int req, const std::vector<int64_t>& prompt,
      int64_t reserve_tokens, int64_t chunk_tokens, const std::vector<int64_t>& boundaries = {},
      SnapshotRequest* snap = nullptr, int64_t attach_position = 0);
  // Completes one bounded chunk. An unfinished slot has device positions
  // -1 between calls so padded decode graphs cannot advance its state.
  // A positive override changes this chunk's budget; zero uses the begin budget.
  bool session_prefill_advance(PrefillCursor& cursor, int64_t chunk_tokens = 0);
  Outputs session_step(int req, int64_t token_id) { return session_verify(req, std::vector<int64_t>{token_id}); }
  Outputs session_verify(int req, const std::vector<int64_t>& token_ids);
  void session_rollback(int req, int accepted);
  void session_close(int req);
  int64_t session_position(int req) const {
    check_req(req, "session_position");
    return session_pos_[static_cast<size_t>(req)];
  }
  void session_reserve_blocks(int req, int64_t tokens);
  int max_session_requests() const { return max_requests_; }
  // The fixed decode batch's row ceiling (SessionParams::decode_rows).
  int max_decode_rows() const { return max_decode_rows_; }
  int64_t kv_blocks_total() const { return derived().has_pool() ? derived().pool().total_blocks() : 0; }
  int64_t kv_blocks_in_use() const { return derived().has_pool() ? derived().pool().blocks_in_use() : 0; }
  int64_t kv_blocks_for_tokens(int64_t tokens) const {
    return derived().has_pool() ? derived().pool().block_count_for_tokens(tokens) : 0;
  }
  int64_t kv_block_tokens() const { return derived().has_pool() ? block_tokens_ : 0; }
  int session_snapshot_align() const { return snapshot_align_; }
  // The NVMe cold tier's view of the paged cache (issue #26,
  // engine/cache_planes.hpp): every plane a block spans, and a block's
  // identity (id + contents generation). A family without a pool has none.
  std::vector<CachePlane> cache_planes() const {
    if (derived().has_pool()) return derived().pool().planes();
    return {};
  }
  uint64_t cache_block_identity(int32_t block) const {
    if (!derived().has_pool()) throw std::logic_error("cache_block_identity: this family has no pool");
    return derived().pool().block_identity(block);
  }
  // A fresh pool block owned by the caller (refcount 1; -1 when the pool is
  // empty) and the release of such blocks — the tier's restore target.
  int32_t cache_acquire_block() {
    if (!derived().has_pool()) throw std::logic_error("cache_acquire_block: this family has no pool");
    return derived().pool().acquire_pinned_block();
  }
  void cache_release_blocks(const int32_t* blocks, int64_t n) {
    if (n <= 0) return;
    if (!derived().has_pool()) throw std::logic_error("cache_release_blocks: this family has no pool");
    derived().pool().unpin_blocks(blocks, n);
  }
  int64_t max_context() const { return max_context_; }
  cudaStream_t stream() const { return stream_; }
  bool mtp_enabled() const { return mtp_; }
  // Host nodes a captured walk may carry (engine/graph_check.hpp): none by
  // default; a family that stages from the host inside the walk shadows it.
  size_t session_graph_host_nodes() const { return 0; }
  int lm_vocab_begin() const { return lm_vocab_begin_; }
  int lm_vocab_count() const { return lm_vocab_count_; }
  BoundaryReducer* set_boundary(BoundaryReducer* b) {
    if (world_ > 1 && b == nullptr)
      throw std::invalid_argument("session model: a boundary reducer is required at tp_world > 1");
    BoundaryReducer* old = boundary_;
    boundary_ = b;
    return old;
  }
  void set_decode_route_traces(bool on) { route_traces_ = on; }
  bool decode_route_traces() const { return route_traces_; }
  void set_decode_tail_mirrors(bool on) { decode_tail_mirrors_ = on; }
  int max_tokens() const { return max_tokens_; }
  int tp_rank() const { return rank_; }
  int tp_world() const { return world_; }

  // ---- the prefix cache -----------------------------------------------------
  size_t session_snapshot_bytes() const {
    size_t b = derived().snapshot_state_bytes();
    if (mtp_) b += derived().draft_state_bytes() + static_cast<size_t>(draft_width_) * 2;
    return b;
  }
  SessionSnapshotMeta session_snapshot(int req, void* dst);
  SessionSnapshotMeta session_snapshot_post_row0(int req, void* dst, int spec_row, int rows_after = 1);
  void session_release_snapshot(const SessionSnapshotMeta& meta);
  void session_attach(int req, const void* src, const SessionSnapshotMeta& meta);

  // ---- the graph era --------------------------------------------------------
  const float* device_logits() const { return logits_; }
  const int64_t* device_tokens() const { return d_tokens_; }
  const int64_t* device_feed(int req, int rows) const {
    return d_tokens_ + static_cast<size_t>(max_decode_rows_) + static_cast<size_t>(req) * static_cast<size_t>(rows);
  }
  const int64_t* device_positions() const { return d_step_pos_; }
  void session_graph_prepare() {
    derived().graph_prepare();
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  }
  void session_graph_capture_step(int req, int64_t token_id) {
    session_graph_capture_step(req, std::vector<int64_t>{token_id});
  }
  void session_graph_capture_step(int req, const std::vector<int64_t>& ids) {
    session_graph_capture_step(req, ids, /*device_positions=*/false);
  }
  // `feed_rows` (2026-09-14, the scheduled verify depth): the slot's
  // persistent feed holds this many rows and the capture verifies the first
  // ids.size() of them — the reduced-depth variants of one slot share the
  // full-depth variant's feed rows. 0 (the default): the feed is ids.size()
  // rows, as every capture before.
  void session_graph_capture_step(int req, const std::vector<int64_t>& ids, bool device_positions,
                                  bool device_tokens = false, int feed_rows = 0);
  void session_graph_capture_commit(int req, const PickVerdict* device_verdict);
  void session_graph_stage(int req, int64_t token_id) { session_graph_stage(req, std::vector<int64_t>{token_id}); }
  void session_graph_stage(int req, const std::vector<int64_t>& ids) {
    decode_host_prep(req, ids, /*upload=*/false, graph_device_positions_);
  }
  // `rows` (2026-09-14): the replay's own verify rows, the bound on
  // `accepted` — a reduced-depth batch leaves the batch contract at fewer
  // rows than a following full-depth scalar replay verifies (0: the last
  // contract's rows, as before).
  void session_graph_settle(int req, int accepted, int rows = 0);
  Outputs session_graph_collect(int req);
  Outputs session_graph_outputs(int req);
  void session_graph_seed_tokens(int req, const std::vector<int64_t>& ids);
  void session_graph_seed_scalar_tokens(const std::vector<int64_t>& ids);
  void session_graph_seed_feed(int req, const std::vector<int64_t>& ids);
  // `feed_rows` (2026-09-14, the scheduled verify depth): the slots keep
  // their feeds at feed_rows rows; a batch verifying rows_per_request <
  // feed_rows rows per slot reads them compacted (glm_spec_gather_feed) and
  // writes the next step's full-width feeds back. 0: the feed is
  // rows_per_request rows, as every capture before.
  void session_graph_capture_batch(int rows_per_request, int requests = 0, int feed_rows = 0);
  void session_graph_stage_batch();
  // Capture-time source: a pinned map owned by the graph family/parity.
  void session_graph_batch_map_source(const int32_t* source) { graph_batch_map_source_ = source; }
  const int32_t* device_batch_map() const { return d_batch_map_; }

  void session_graph_use_batch_contract(int rows_per_request, int requests = 0);
  void session_graph_capture_commit_batch(const PickVerdict* device_verdicts);
  void session_graph_capture_verify_next_tokens_batch(const PickVerdict* verify_verdicts);
  int graph_batch_requests() const { return graph_batch_requests_; }
  int graph_rows_per_request() const { return graph_rows_per_request_; }

  // ---- the MTP draft block ----------------------------------------------------
  Outputs session_draft(int req, const std::vector<int64_t>& tokens);
  // Depth >= 2 (2026-09-10, the GLM-5.3-Flash chain on the window
  // families): after the block's rows off the verify, one more row per
  // further draft at position counter + index, fed the previous draft's
  // pick and the block's own previous output row as its hidden (the
  // block's approximation of the main stack's hidden there), landed in
  // the slot's window at that position — provisional, positional, the next
  // step's real rows overwrite it; the counter does not move. Eager
  // (session_draft_chain: the row's head logits back) and recorded
  // (session_graph_capture_draft_chain, behind the recorded draft pick);
  // the feed then carries every draft. Families without the chain
  // (kDraftChain false) report no chain row fits.
  bool session_draft_chain_fits(int req, int index) const;
  Outputs session_draft_chain(int req, int64_t token, int index, bool first, bool last);
  void session_draft_rollback(int req, int rows);
  void session_graph_capture_draft(int req, const PickVerdict* verify_verdict);
  void session_graph_capture_draft_chain(int req, const PickVerdict* verify_verdict,
                                         const PickVerdict* draft_verdict, int index, bool first, bool last);
  void session_graph_capture_next_tokens(int req, const PickVerdict* draft_verdict);
  void session_graph_capture_next_tokens(int req, const std::vector<const PickVerdict*>& draft_verdicts);
  void session_graph_capture_draft_batch(const PickVerdict* verify_verdicts);
  // The fixed batch's chain rows (depth >= 2, 2026-09-10): one per request
  // slot, the recursion of session_graph_capture_draft_chain over the
  // batched draft run's rows, landed in each slot's window; the family's
  // chain hooks bracket them per slot.
  void session_graph_capture_draft_chain_batch(const PickVerdict* verify_verdicts, const PickVerdict* draft_verdicts,
                                               int index, bool first, bool last);
  void session_graph_capture_next_tokens_batch(const PickVerdict* draft_verdicts);
  void session_graph_capture_next_tokens_batch(const std::vector<const PickVerdict*>& draft_verdicts);
  int64_t session_draft_position(int req) const {
    check_req(req, "session_draft_position");
    return mtp_ ? mtp_pos_[static_cast<size_t>(req)] : 0;
  }

  // Top-k of fp32 logits per row: highest first, ties to the lower id.
  static std::vector<std::vector<std::pair<int32_t, float>>> topk(const std::vector<float>& logits, int64_t rows,
                                                                  int vocab, int k);

 protected:
  Derived& derived() { return static_cast<Derived&>(*this); }
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

  // Called by the derived constructor: the model stream first (its loader
  // registers it as the reader), then — once the vocab slice is known —
  // the generic buffers.
  void init_stream();
  void init_session(const SessionParams& p);
  void check_req(int req, const char* what) const {
    if (req < 0 || req >= max_requests_)
      throw std::out_of_range(std::string(what) + ": request slot " + std::to_string(req));
  }
  [[noreturn]] void refuse_mtp(const char* what) const {
    throw std::logic_error(std::string(what) + ": the MTP draft block is not enabled");
  }
  // The generic part of a slot's opening (the family's reset first).
  void open_slot(int req);
  void push_position(int req) {
    h_session_pos_[req] = session_pos_[static_cast<size_t>(req)];
    DGPP_CUDA_OK(cudaMemcpyAsync(d_session_pos_ + req, h_session_pos_ + req, sizeof(int64_t),
                                 cudaMemcpyHostToDevice, stream_));
  }
  void push_mtp_position(int req) {
    if (!mtp_) return;
    h_mtp_pos_[req] = mtp_pos_[static_cast<size_t>(req)];
    DGPP_CUDA_OK(cudaMemcpyAsync(d_mtp_pos_ + req, h_mtp_pos_ + req, sizeof(int64_t),
                                 cudaMemcpyHostToDevice, stream_));
  }
  // The prefill rows' metadata (positions pos0 + t, the request, one
  // span) into the device arrays (host uploads, a sync).
  void stage_prefill_meta(int req, int64_t pos0, int T);
  void stage_prefill_group_meta(const RowRun& run);
  void decode_host_prep(int req, const std::vector<int64_t>& ids, bool upload, bool device_positions);
  // A walk's staged inputs (the decode rows' token upload included) and
  // its results (the tail mirrors, the host copies).
  RowInputs begin_run(const RowRun& run);
  Outputs finish_run(const RowRun& run, Outputs&& out);
  std::vector<int64_t> prefill_cuts(int64_t start, int64_t end, const std::vector<int64_t>& boundaries) const;
  Outputs session_prefill_chunks(int req, const int64_t* ids, int64_t start, int64_t count,
                                 const std::vector<int64_t>& boundaries, SnapshotRequest* snap);
  PrefillCursor prefill_cursor(int req, const int64_t* ids, int64_t start, int64_t count,
                               const std::vector<int64_t>& boundaries, SnapshotRequest* snap);
  void prefill_chunk(PrefillCursor& cursor, int64_t budget = 0);
  void write_snapshot(int req, void* dst, int spec_row);
  SessionSnapshotMeta pin_blocks_at(int req, int64_t pos, const char* what);
  // The draft block's rows.
  void mtp_prefill_rows(int req, int64_t row0, int64_t row1, const int64_t* tokens);
  void mtp_decode_host_prep(int req, const std::vector<int64_t>& tokens, bool upload);
  // The draft block's input window per slot ([R][max_decode_rows_][draft_width]).
  uint16_t* mtp_window(int req) const {
    return mtp_window_ +
           static_cast<size_t>(req) * static_cast<size_t>(max_decode_rows_) * static_cast<size_t>(draft_width_);
  }
  // The window store the walk calls for its last rows (n <= max_decode_rows_).
  void store_draft_hidden(const uint16_t* rows, const int32_t* req_ids, const int64_t* pos, int n) {
    qwen_mtp_hidden_store_bf16(rows, req_ids, pos, mtp_window_, static_cast<int64_t>(max_decode_rows_) * draft_width_,
                               max_decode_rows_, n, draft_width_, stream_);
  }
  void gather_draft_hidden(const int32_t* req_ids, const int64_t* pos, uint16_t* dst, int T) {
    qwen_mtp_hidden_gather_bf16(mtp_window_, static_cast<int64_t>(max_decode_rows_) * draft_width_, max_decode_rows_,
                                req_ids, pos, dst, T, draft_width_, stream_);
  }

  int max_tokens_ = 0;
  int64_t max_cache_tokens_ = 0;
  int rank_ = 0, world_ = 1;
  BoundaryReducer* boundary_ = nullptr;
  int max_requests_ = 1;
  int64_t vocab_size_ = 0;
  int hidden_ = 0;
  int lm_vocab_begin_ = 0, lm_vocab_count_ = 0;
  int64_t max_context_ = 0;
  int block_tokens_ = 0;
  int snapshot_align_ = 1;
  int draft_width_ = 0;
  int chunk_tokens_ = 0;  // the prefill cut (0: prompts must fit max_tokens)
  int32_t eos_ = -1;
  cudaStream_t stream_ = nullptr;
  bool route_traces_ = true;
  bool decode_tail_mirrors_ = true;

  std::vector<int64_t> session_pos_;  // [R]; 0 = closed slot (the host mirror)
  int64_t* d_session_pos_ = nullptr;  // device [R]: the graphs' positions
  int64_t* h_session_pos_ = nullptr;  // pinned upload mirror

  // Tokens and the rows' metadata ([rows] = [max_decode_rows_]).
  int64_t* d_tokens_ = nullptr;     // [max(M, rows + R*kSpecRows)]
  int64_t* step_tokens_ = nullptr;  // the decode rows' token source (eager scratch / a feed)
  int64_t* d_prefill_pos_ = nullptr;
  int32_t* d_prefill_req_ = nullptr;
  int32_t* d_prefill_spans_ = nullptr;
  int64_t* h_token_ = nullptr;      // pinned [max(M, rows + R*kSpecRows)]
  int32_t* h_req_ids_ = nullptr;    // pinned [rows]
  int64_t* h_step_pos_ = nullptr;
  int32_t* h_req_spans_ = nullptr;  // pinned [rows, 2]
  int32_t* d_req_ids_ = nullptr;
  int32_t* d_batch_map_ = nullptr;
  const int32_t* graph_batch_map_source_ = nullptr;
  int64_t* d_step_pos_ = nullptr;
  int32_t* d_req_spans_ = nullptr;
  int decode_rows_ = 0;              // the current decode run's rows
  int max_decode_rows_ = kDecodeRows;  // the fixed batch's row ceiling (SessionParams::decode_rows)

  // The head's outputs: fp32 logits [M, vocab slice], the final hidden [M, H].
  float* logits_ = nullptr;
  uint16_t* h_ = nullptr;
  float* h_tail_logits_ = nullptr;     // pinned [rows, vocab slice]
  uint16_t* h_tail_hidden_ = nullptr;  // pinned [rows, H]

  // The graph era's contract flags (the last capture's shape).
  bool graph_device_positions_ = false;
  bool graph_device_tokens_ = false;
  bool graph_has_draft_ = false;
  int graph_batch_requests_ = 0;
  int graph_rows_per_request_ = 0;
  int graph_feed_rows_ = 0;  // a scalar capture's feed rows (>= its verified rows)

  // The draft block.
  bool mtp_ = false;
  std::vector<int64_t> mtp_pos_;   // [R] the block's row counter (host mirror)
  int64_t* d_mtp_pos_ = nullptr;   // device [R]
  int64_t* h_mtp_pos_ = nullptr;   // pinned upload mirror
  int64_t* d_next_ = nullptr;      // device [R]: the verify's next token, parked
  uint16_t* mtp_window_ = nullptr; // [R][rows][draft_width]: the last hidden rows by position
  int draft_rows_ = 0;
  int draft_last_row_ = 0;           // the block's output row the first chain row takes (eager)
  int rows_after_for_snapshot_ = 0;  // write_snapshot's hop context (post_row0)
};

// ---------------------------------------------------------------------------
namespace session_detail {
template <class T>
T* dev_alloc(size_t n) {
  T* p = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  return p;
}
template <class T>
T* pinned_alloc(size_t n) {
  T* p = nullptr;
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&p), std::max<size_t>(n, 1) * sizeof(T), cudaHostAllocMapped));
  return p;
}
inline void d2d(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
  if (bytes == 0) return;
  DGPP_CUDA_OK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream));
}
}  // namespace session_detail

template <class D>
void SessionModel<D>::init_stream() {
  if (stream_ != nullptr) throw std::logic_error("session model: init_stream twice");
  DGPP_CUDA_OK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
}

template <class D>
void SessionModel<D>::init_session(const SessionParams& p) {
  using namespace session_detail;
  if (stream_ == nullptr) init_stream();
  if (d_tokens_ != nullptr) throw std::logic_error("session model: init_session twice");
  if (p.max_tokens <= 0) throw std::invalid_argument("session model: max_tokens must be positive");
  if (p.max_requests <= 0 || p.max_requests > kPickMaxRequests)
    throw std::invalid_argument("session model: max_requests must be in [1, kPickMaxRequests]");
  if (p.decode_rows > kDecodeRowsMax)
    throw std::invalid_argument("session model: decode_rows exceeds kDecodeRowsMax (" +
                                std::to_string(kDecodeRowsMax) + ")");
  if ((p.world > 1) != (p.boundary != nullptr))
    throw std::invalid_argument(
        "session model: a boundary reducer is required exactly when tp_world > 1 — without one the "
        "block-boundary partials would be returned silently as results");
  if (p.mtp && p.draft_width <= 0) throw std::invalid_argument("session model: the draft width");
  max_tokens_ = p.max_tokens;
  max_cache_tokens_ = p.max_cache_tokens;
  rank_ = p.rank;
  world_ = p.world;
  boundary_ = p.boundary;
  max_requests_ = p.max_requests;
  max_decode_rows_ = std::max({kDecodeRows, p.decode_rows, max_requests_});
  mtp_ = p.mtp;
  vocab_size_ = p.vocab_size;
  hidden_ = p.hidden;
  lm_vocab_begin_ = p.lm_vocab_begin;
  lm_vocab_count_ = p.lm_vocab_count;
  block_tokens_ = p.block_tokens;
  snapshot_align_ = std::max(p.snapshot_align, 1);
  draft_width_ = p.draft_width;
  eos_ = p.eos;
  max_context_ = std::min<int64_t>(max_cache_tokens_, p.max_position_embeddings);
  {
    const int chunk = D::prefill_chunk_tokens();
    chunk_tokens_ = max_tokens_ >= chunk ? chunk : (max_tokens_ / snapshot_align_) * snapshot_align_;
  }
  const size_t R = static_cast<size_t>(max_requests_);
  const size_t rows = static_cast<size_t>(max_decode_rows_);
  const size_t M = static_cast<size_t>(max_tokens_);
  session_pos_.assign(R, 0);
  d_session_pos_ = dev_alloc<int64_t>(R);
  h_session_pos_ = pinned_alloc<int64_t>(R);
  DGPP_CUDA_OK(cudaMemset(d_session_pos_, 0, R * sizeof(int64_t)));
  const size_t token_rows = std::max(M, rows + R * static_cast<size_t>(kSpecRows));
  d_tokens_ = dev_alloc<int64_t>(token_rows);
  DGPP_CUDA_OK(cudaMemset(d_tokens_, 0, token_rows * sizeof(int64_t)));
  step_tokens_ = d_tokens_;
  d_prefill_pos_ = dev_alloc<int64_t>(M);
  d_prefill_req_ = dev_alloc<int32_t>(M);
  d_prefill_spans_ = dev_alloc<int32_t>(2 * static_cast<size_t>(std::max(1, max_requests_)));
  h_token_ = pinned_alloc<int64_t>(token_rows);
  h_req_ids_ = pinned_alloc<int32_t>(rows);
  h_step_pos_ = pinned_alloc<int64_t>(rows);
  h_req_spans_ = pinned_alloc<int32_t>(2 * rows);
  d_req_ids_ = dev_alloc<int32_t>(rows);
  d_batch_map_ = dev_alloc<int32_t>(max_requests_);
  d_step_pos_ = dev_alloc<int64_t>(rows);
  d_req_spans_ = dev_alloc<int32_t>(2 * rows);
  logits_ = dev_alloc<float>(M * static_cast<size_t>(lm_vocab_count_));
  h_ = dev_alloc<uint16_t>(M * static_cast<size_t>(hidden_));
  h_tail_logits_ = pinned_alloc<float>(rows * static_cast<size_t>(lm_vocab_count_));
  h_tail_hidden_ = pinned_alloc<uint16_t>(rows * static_cast<size_t>(hidden_));
  if (mtp_) {
    mtp_pos_.assign(R, 0);
    d_mtp_pos_ = dev_alloc<int64_t>(R);
    h_mtp_pos_ = pinned_alloc<int64_t>(R);
    DGPP_CUDA_OK(cudaMemset(d_mtp_pos_, 0, R * sizeof(int64_t)));
    d_next_ = dev_alloc<int64_t>(R);
    DGPP_CUDA_OK(cudaMemset(d_next_, 0, R * sizeof(int64_t)));
    mtp_window_ = dev_alloc<uint16_t>(R * rows * static_cast<size_t>(draft_width_));
    DGPP_CUDA_OK(cudaMemset(mtp_window_, 0, R * rows * static_cast<size_t>(draft_width_) * 2));
  }
}

// The generic bytes of a shape (the memory plan's "session core" line):
// tokens, the rows' metadata, the head's outputs and the draft window.
// `decode_rows`: the fixed batch's row ceiling (SessionParams::decode_rows;
// floored the same way).
inline void session_core_plan_bytes(int max_tokens, int max_requests, int hidden, int lm_vocab_count, bool mtp,
                                    int draft_width, size_t* device, size_t* pinned, int decode_rows = kDecodeRows) {
  const size_t M = static_cast<size_t>(max_tokens), R = static_cast<size_t>(max_requests);
  const size_t rows = static_cast<size_t>(std::max({kDecodeRows, decode_rows, max_requests}));
  const size_t token_rows = std::max(M, rows + R * static_cast<size_t>(kSpecRows));
  const size_t H = static_cast<size_t>(hidden), V = static_cast<size_t>(lm_vocab_count);
  size_t dev = R * 8 + token_rows * 8 + M * 12 + 8 + rows * (4 + 8 + 8) + M * V * 4 + M * H * 2;
  size_t pin = R * 8 + token_rows * 8 + rows * (4 + 8 + 8) + rows * V * 4 + rows * H * 2;
  if (mtp) {
    dev += R * 8 * 2 + R * rows * static_cast<size_t>(draft_width) * 2;
    pin += R * 8;
  }
  *device = dev;
  *pinned = pin;
}

template <class D>
SessionModel<D>::~SessionModel() {
  cudaFree(d_session_pos_);
  cudaFreeHost(h_session_pos_);
  cudaFree(d_tokens_);
  cudaFree(d_prefill_pos_);
  cudaFree(d_prefill_req_);
  cudaFree(d_prefill_spans_);
  cudaFreeHost(h_token_);
  cudaFreeHost(h_req_ids_);
  cudaFreeHost(h_step_pos_);
  cudaFreeHost(h_req_spans_);
  cudaFree(d_req_ids_);
  cudaFree(d_batch_map_);
  cudaFree(d_step_pos_);
  cudaFree(d_req_spans_);
  cudaFree(logits_);
  cudaFree(h_);
  cudaFreeHost(h_tail_logits_);
  cudaFreeHost(h_tail_hidden_);
  cudaFree(d_mtp_pos_);
  cudaFreeHost(h_mtp_pos_);
  cudaFree(d_next_);
  cudaFree(mtp_window_);
  if (stream_) cudaStreamDestroy(stream_);
}

template <class D>
void SessionModel<D>::open_slot(int req) {
  derived().reset_slot_state(req);
  session_pos_[static_cast<size_t>(req)] = 0;
  push_position(req);
  if (mtp_) {
    mtp_pos_[static_cast<size_t>(req)] = 0;
    push_mtp_position(req);
  }
}

template <class D>
void SessionModel<D>::stage_prefill_meta(int req, int64_t pos0, int T) {
  std::vector<int64_t> pos(static_cast<size_t>(T));
  std::vector<int32_t> reqs(static_cast<size_t>(T), req);
  const int32_t span[2] = {0, T};
  for (int t = 0; t < T; ++t) pos[static_cast<size_t>(t)] = pos0 + t;
  DGPP_CUDA_OK(cudaMemcpyAsync(d_prefill_pos_, pos.data(), static_cast<size_t>(T) * 8, cudaMemcpyHostToDevice, stream_));
  DGPP_CUDA_OK(cudaMemcpyAsync(d_prefill_req_, reqs.data(), static_cast<size_t>(T) * 4, cudaMemcpyHostToDevice, stream_));
  DGPP_CUDA_OK(cudaMemcpyAsync(d_prefill_spans_, span, 2 * 4, cudaMemcpyHostToDevice, stream_));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));  // the host vectors' lifetime
}

// The group prefill's rows: span s's positions from span_pos0[s], its
// request id, and the spans as (first row, rows).
template <class D>
void SessionModel<D>::stage_prefill_group_meta(const RowRun& run) {
  if (run.num_spans > max_requests_) throw std::invalid_argument("run_rows: more group spans than request slots");
  std::vector<int64_t> pos(static_cast<size_t>(run.T));
  std::vector<int32_t> reqs(static_cast<size_t>(run.T));
  std::vector<int32_t> spans(static_cast<size_t>(run.num_spans) * 2);
  int at = 0;
  for (int s = 0; s < run.num_spans; ++s) {
    const int len = run.span_lens[s];
    if (len <= 0 || at + len > run.T) throw std::invalid_argument("run_rows: the group's spans do not fit its rows");
    spans[static_cast<size_t>(s) * 2] = at;
    spans[static_cast<size_t>(s) * 2 + 1] = len;
    for (int i = 0; i < len; ++i) {
      pos[static_cast<size_t>(at + i)] = run.span_pos0[s] + i;
      reqs[static_cast<size_t>(at + i)] = run.span_reqs[s];
    }
    at += len;
  }
  if (at != run.T) throw std::invalid_argument("run_rows: the group's spans do not cover its rows");
  DGPP_CUDA_OK(cudaMemcpyAsync(d_prefill_pos_, pos.data(), pos.size() * 8, cudaMemcpyHostToDevice, stream_));
  DGPP_CUDA_OK(cudaMemcpyAsync(d_prefill_req_, reqs.data(), reqs.size() * 4, cudaMemcpyHostToDevice, stream_));
  DGPP_CUDA_OK(cudaMemcpyAsync(d_prefill_spans_, spans.data(), spans.size() * 4, cudaMemcpyHostToDevice, stream_));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));  // the host vectors' lifetime
}

template <class D>
typename SessionModel<D>::RowInputs SessionModel<D>::begin_run(const RowRun& run) {
  const int T = run.T;
  if (T <= 0) throw std::invalid_argument("run_rows: empty row batch");
  if (T > max_tokens_) throw std::invalid_argument("run_rows: rows exceed max_tokens");
  RowInputs in;
  in.batched = run.batch_requests > 0;
  in.num_requests = in.batched ? run.batch_requests : (run.num_spans > 0 ? run.num_spans : 1);
  if (run.decode) {
    // Staged by decode_host_prep / capture_batch: the request ids, the
    // positions and the spans are on the device; the tokens ride the
    // pinned staging through a kernel upload unless the graph reads its
    // device feed (the previous replay's last node left them there).
    if (!run.capture) step_tokens_ = d_tokens_;
    if (!(run.capture && graph_device_tokens_)) glm_upload_i64(h_token_, step_tokens_, T, stream_);
    in.tokens = step_tokens_;
    in.pos = d_step_pos_;
    in.req_ids = d_req_ids_;
    in.spans = d_req_spans_;
  } else {
    DGPP_CUDA_OK(cudaMemcpyAsync(d_tokens_, run.ids, static_cast<size_t>(T) * 8, cudaMemcpyHostToDevice, stream_));
    if (run.num_spans > 0)
      stage_prefill_group_meta(run);
    else
      stage_prefill_meta(run.req, run.pos0, T);
    in.tokens = d_tokens_;
    in.pos = d_prefill_pos_;
    in.req_ids = d_prefill_req_;
    in.spans = d_prefill_spans_;
  }
  return in;
}

template <class D>
typename SessionModel<D>::Outputs SessionModel<D>::finish_run(const RowRun& run, Outputs&& out) {
  const int T = run.T;
  const size_t H = static_cast<size_t>(hidden_);
  // The decode tail's rows into the pinned mirrors: eager rows always
  // (they sync right below); a capture records the copies only while the
  // mirrors are on (the kernels-only decode graph turns them off).
  const bool every_row = run.all_rows || run.decode;
  const bool group = !run.decode && !run.all_rows && run.num_spans > 0;
  const size_t rows_out = every_row ? static_cast<size_t>(T) : group ? static_cast<size_t>(run.num_spans) : 1;
  const size_t first = every_row ? 0 : static_cast<size_t>(T - 1);
  if (run.decode && (!run.capture || decode_tail_mirrors_)) {
    DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_logits_, logits_, rows_out * lm_vocab_count_ * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
    DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_hidden_, h_, rows_out * H * 2, cudaMemcpyDeviceToHost, stream_));
  }
  if (run.capture) return Outputs{};  // nothing executed, nothing to materialize
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  out.lm_vocab_begin = lm_vocab_begin_;
  out.lm_vocab_count = lm_vocab_count_;
  out.final_hidden_bits.resize(rows_out * H);
  out.logits.resize(rows_out * lm_vocab_count_);
  if (run.decode) {
    std::copy(h_tail_logits_, h_tail_logits_ + out.logits.size(), out.logits.begin());
    std::copy(h_tail_hidden_, h_tail_hidden_ + out.final_hidden_bits.size(), out.final_hidden_bits.begin());
  } else if (group) {
    // Each span's last row.
    int at = 0;
    for (int s = 0; s < run.num_spans; ++s) {
      const size_t last = static_cast<size_t>(at + run.span_lens[s] - 1);
      DGPP_CUDA_OK(cudaMemcpy(out.final_hidden_bits.data() + static_cast<size_t>(s) * H, h_ + last * H, H * 2,
                              cudaMemcpyDeviceToHost));
      DGPP_CUDA_OK(cudaMemcpy(out.logits.data() + static_cast<size_t>(s) * lm_vocab_count_, logits_ + last * lm_vocab_count_,
                              static_cast<size_t>(lm_vocab_count_) * 4, cudaMemcpyDeviceToHost));
      at += run.span_lens[s];
    }
  } else {
    DGPP_CUDA_OK(cudaMemcpy(out.final_hidden_bits.data(), h_ + first * H, out.final_hidden_bits.size() * 2,
                            cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(out.logits.data(), logits_ + first * lm_vocab_count_, out.logits.size() * 4,
                            cudaMemcpyDeviceToHost));
  }
  return std::move(out);
}

// ---- the prefill ------------------------------------------------------------

template <class D>
std::vector<int64_t> SessionModel<D>::prefill_cuts(int64_t start, int64_t end,
                                                   const std::vector<int64_t>& boundaries) const {
  const int64_t align = snapshot_align_;
  std::vector<int64_t> cuts;
  if (chunk_tokens_ > 0)
    for (int64_t m = (start / chunk_tokens_ + 1) * chunk_tokens_; m < end; m += chunk_tokens_) cuts.push_back(m);
  for (int64_t b : boundaries) {
    const int64_t a = (b / align) * align;  // the aligned image
    if (a > start && a < end) cuts.push_back(a);
  }
  std::sort(cuts.begin(), cuts.end());
  cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
  return cuts;
}

template <class D>
typename SessionModel<D>::PrefillCursor SessionModel<D>::prefill_cursor(
    int req, const int64_t* ids, int64_t start, int64_t count, const std::vector<int64_t>& boundaries,
    SnapshotRequest* snap) {
  PrefillCursor cursor;
  cursor.req = req;
  cursor.ids = ids;
  cursor.start = cursor.next = start;
  cursor.end = start + count;
  cursor.cuts = prefill_cuts(start, cursor.end, boundaries);
  cursor.snap = snap;
  for (auto* at = snap; at != nullptr; at = at->next) {
    if (at->dst == nullptr || at->meta == nullptr)
      throw std::invalid_argument("session_prefill: snapshot request without a buffer");
    const bool at_cut = std::binary_search(cursor.cuts.begin(), cursor.cuts.end(), at->position) ||
                        at->position == cursor.end;
    if (!at_cut) throw std::invalid_argument("session_prefill: the snapshot position is not a chunk end");
  }
  return cursor;
}

template <class D>
void SessionModel<D>::prefill_chunk(PrefillCursor& cursor, int64_t budget) {
  const int req = cursor.req;
  const int64_t* ids = cursor.ids;
  const int64_t start = cursor.start, end = cursor.end, c0 = cursor.next;
  auto* snap = cursor.snap;
  auto& out = cursor.output;
  int64_t c1 = cursor.cut_index < cursor.cuts.size() ? cursor.cuts[cursor.cut_index] : end;
  // Keep model/snapshot cuts separate from the scheduling grid. A larger
  // budget can coalesce future work without crossing a required snapshot.
  // Global grid boundaries preserve the original fixed-budget chunk shapes.
  if (budget > 0) c1 = std::min(c1, (c0 / budget + 1) * budget);
  if (cursor.cut_index < cursor.cuts.size() && c1 == cursor.cuts[cursor.cut_index]) ++cursor.cut_index;
  if (c1 - c0 > max_tokens_)
    throw std::invalid_argument("session_prefill: a chunk of " + std::to_string(c1 - c0) +
                                " rows exceeds max_tokens " + std::to_string(max_tokens_));
  RowRun run;
  run.req = req;
  run.ids = ids + (c0 - start);
  run.T = static_cast<int>(c1 - c0);
  run.pos0 = c0;
  run.decode = false;
  run.all_rows = false;
  // Preserve logical span boundaries across yields. In particular, the
  // bounded DeepSeek decoder runs only at a span's last chunk, and a
  // snapshot must close the span before its state can be published.
  run.first_chunk = cursor.span_start;
  run.last_chunk = c1 == end;
  for (auto* at = snap; at != nullptr; at = at->next)
    run.last_chunk |= !at->taken && at->position == c1;
  cursor.span_start = run.last_chunk;
  Outputs chunk = derived().run_rows(run);
  out.logits = std::move(chunk.logits);
  out.final_hidden_bits = std::move(chunk.final_hidden_bits);
  out.lm_vocab_begin = chunk.lm_vocab_begin;
  out.lm_vocab_count = chunk.lm_vocab_count;
  if (out.route_ids.empty()) {
    out.route_ids = std::move(chunk.route_ids);
    out.route_weights = std::move(chunk.route_weights);
  } else {
    for (size_t l = 0; l < chunk.route_ids.size() && l < out.route_ids.size(); ++l) {
      out.route_ids[l].insert(out.route_ids[l].end(), chunk.route_ids[l].begin(), chunk.route_ids[l].end());
      out.route_weights[l].insert(out.route_weights[l].end(), chunk.route_weights[l].begin(),
                                  chunk.route_weights[l].end());
    }
  }
  // A capturing walk's per-row selections, chunk after chunk (the
  // families' chunked-prefill gates compare them with the one-shot's).
  // (A chunk may capture fewer sources than the next: the bounded
  // prefill's decoder sources appear on the span's last chunk only.)
  if (out.dsa_selections.size() < chunk.dsa_selections.size()) out.dsa_selections.resize(chunk.dsa_selections.size());
  for (size_t l = 0; l < chunk.dsa_selections.size(); ++l)
    out.dsa_selections[l].insert(out.dsa_selections[l].end(), chunk.dsa_selections[l].begin(),
                                 chunk.dsa_selections[l].end());
  // A capturing walk's per-layer rows likewise, chunk after chunk.
  if (out.layer_states.size() < chunk.layer_states.size()) out.layer_states.resize(chunk.layer_states.size());
  for (size_t l = 0; l < chunk.layer_states.size(); ++l)
    out.layer_states[l].insert(out.layer_states[l].end(), chunk.layer_states[l].begin(), chunk.layer_states[l].end());
  session_pos_[static_cast<size_t>(req)] = c1;
  push_position(req);
  // The draft block over this chunk's rows — row q embeds tok_{q+1}, so
  // the rows stop one short of the prompt end. Per chunk, so its state
  // stands at every cut.
  if (mtp_) {
    const int64_t r1 = std::min<int64_t>(c1, end - 1);
    if (r1 > c0) mtp_prefill_rows(req, c0, r1, ids + (c0 + 1 - start));
    mtp_pos_[static_cast<size_t>(req)] = std::max<int64_t>(mtp_pos_[static_cast<size_t>(req)], r1);
    push_mtp_position(req);
  }
  for (auto* at = snap; at != nullptr; at = at->next) {
    if (!at->taken && at->position == c1) {
      *at->meta = session_snapshot(req, at->dst);
      at->taken = true;
    }
  }
  cursor.next = c1;
  report_prefill_progress(req, c1);
}

template <class D>
typename SessionModel<D>::Outputs SessionModel<D>::session_prefill_chunks(
    int req, const int64_t* ids, int64_t start, int64_t count, const std::vector<int64_t>& boundaries,
    SnapshotRequest* snap) {
  auto cursor = prefill_cursor(req, ids, start, count, boundaries, snap);
  while (cursor.next < cursor.end) prefill_chunk(cursor);
  return std::move(cursor.output);
}

template <class D>
typename SessionModel<D>::PrefillCursor SessionModel<D>::session_prefill_begin(
    int req, const std::vector<int64_t>& prompt, int64_t reserve_tokens, int64_t chunk_tokens,
    const std::vector<int64_t>& boundaries, SnapshotRequest* snap, int64_t attach_position) {
  check_req(req, "session_prefill_begin");
  if constexpr (!D::kResumablePrefill)
    throw std::logic_error("session_prefill_begin: family has not enabled resumable prefill");
  const int64_t end = static_cast<int64_t>(prompt.size());
  if (end <= 0 || end > max_context_ || reserve_tokens < end || reserve_tokens > max_context_)
    throw std::invalid_argument("session_prefill_begin: prompt/reservation outside context bounds");
  if (chunk_tokens < snapshot_align_ || chunk_tokens > max_tokens_ || chunk_tokens % snapshot_align_ != 0)
    throw std::invalid_argument("session_prefill_begin: chunk budget must fit max_tokens and snapshot alignment");
  if (attach_position < 0 || attach_position >= end || attach_position % snapshot_align_ != 0)
    throw std::invalid_argument("session_prefill_begin: invalid attach position");
  for (const int64_t id : prompt)
    if (id < 0 || id >= vocab_size_)
      throw std::invalid_argument("session_prefill_begin: token id out of range");
  std::vector<int64_t> cuts = boundaries;
  // A snapshot accepted on the initial budget grid remains a mandatory cut
  // even if later advances increase the budget.
  for (auto* at = snap; at != nullptr; at = at->next)
    if (at->position > attach_position && at->position < end && at->position % chunk_tokens == 0)
      cuts.push_back(at->position);
  auto cursor = prefill_cursor(req, prompt.data() + attach_position, attach_position,
                               end - attach_position, cuts, snap);
  cursor.budget_tokens = chunk_tokens;
  if (attach_position == 0) open_slot(req);
  else if (session_pos_[static_cast<size_t>(req)] != attach_position)
    throw std::logic_error("session_prefill_begin: slot does not match attached prefix");
  session_reserve_blocks(req, reserve_tokens);
  if (attach_position > 0 && mtp_) {
    if (mtp_pos_[static_cast<size_t>(req)] == attach_position - 1)
      (void)session_draft(req, {prompt[static_cast<size_t>(attach_position)]});
    if (mtp_pos_[static_cast<size_t>(req)] != attach_position)
      throw std::logic_error("session_prefill_begin: draft is not at the attach position");
  }
  // The first advance runs in the same scheduler quantum as begin.
  return cursor;
}

template <class D>
bool SessionModel<D>::session_prefill_advance(PrefillCursor& cursor, int64_t chunk_tokens) {
  check_req(cursor.req, "session_prefill_advance");
  if (cursor.next >= cursor.end || session_pos_[static_cast<size_t>(cursor.req)] != cursor.next)
    throw std::logic_error("session_prefill_advance: completed or stale cursor");
  const int64_t budget = chunk_tokens == 0 ? cursor.budget_tokens : chunk_tokens;
  if (budget < snapshot_align_ || budget > max_tokens_ || budget % snapshot_align_ != 0)
    throw std::invalid_argument("session_prefill_advance: chunk budget must fit max_tokens and snapshot alignment");
  if (cursor.suspended) {
    push_position(cursor.req);
    if (mtp_) push_mtp_position(cursor.req);
    cursor.suspended = false;
  }
  prefill_chunk(cursor, budget);
  const bool done = cursor.next == cursor.end;
  if (!done) {
    // Keep the host positions and all request-owned state. Graph padding
    // derives from these device counters, so an unfinished slot is inert.
    DGPP_CUDA_OK(cudaMemsetAsync(d_session_pos_ + cursor.req, 0xff, sizeof(int64_t), stream_));
    if (mtp_) DGPP_CUDA_OK(cudaMemsetAsync(d_mtp_pos_ + cursor.req, 0xff, sizeof(int64_t), stream_));
    cursor.suspended = true;
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  return done;
}

template <class D>
std::vector<typename SessionModel<D>::Outputs> SessionModel<D>::session_prefill_group(
    const std::vector<int>& reqs, const std::vector<const std::vector<int64_t>*>& prompts) {
  const int n = static_cast<int>(reqs.size());
  if (n <= 0 || prompts.size() != reqs.size()) throw std::invalid_argument("session_prefill_group: requests and prompts");
  if (n == 1) return {session_prefill(reqs[0], *prompts[0])};
  const int64_t limit = derived().prefill_group_span_limit();
  if (limit <= 0) throw std::invalid_argument("session_prefill_group: this family prefills one request per walk");
  std::vector<int64_t> ids;
  std::vector<int32_t> span_reqs, span_lens;
  std::vector<int64_t> span_pos0;
  for (int s = 0; s < n; ++s) {
    check_req(reqs[s], "session_prefill_group");
    for (int t = 0; t < s; ++t)
      if (reqs[t] == reqs[s]) throw std::invalid_argument("session_prefill_group: a request twice in the group");
    const std::vector<int64_t>& p = *prompts[s];
    const int64_t P = static_cast<int64_t>(p.size());
    if (P <= 0) throw std::invalid_argument("session_prefill_group: empty prompt");
    if (P > limit) throw std::invalid_argument("session_prefill_group: a prompt exceeds the group span limit");
    for (int64_t id : p)
      if (id < 0 || id >= vocab_size_) throw std::invalid_argument("session_prefill_group: token id out of range");
    ids.insert(ids.end(), p.begin(), p.end());
    span_reqs.push_back(reqs[s]);
    span_lens.push_back(static_cast<int32_t>(P));
    span_pos0.push_back(0);
  }
  if (static_cast<int64_t>(ids.size()) > max_tokens_)
    throw std::invalid_argument("session_prefill_group: the group's prompts exceed max_tokens");
  for (int s = 0; s < n; ++s) {
    open_slot(reqs[s]);
    if (derived().has_pool() && !derived().pool().ensure_request_blocks(reqs[s], span_lens[s], stream_))
      throw std::runtime_error("session_prefill_group: the cache pool cannot cover a prompt (admission budget)");
  }
  RowRun run;
  run.req = reqs[0];
  run.ids = ids.data();
  run.T = static_cast<int>(ids.size());
  run.pos0 = 0;
  run.decode = false;
  run.all_rows = false;
  run.first_chunk = true;
  run.last_chunk = true;
  run.span_reqs = span_reqs.data();
  run.span_pos0 = span_pos0.data();
  run.span_lens = span_lens.data();
  run.num_spans = n;
  Outputs all = derived().run_rows(run);
  std::vector<Outputs> outs(static_cast<size_t>(n));
  const size_t H = static_cast<size_t>(hidden_);
  // A capturing walk's per-layer rows and selections (the whole group's
  // rows, span-major) ride the first request's outputs.
  outs[0].layer_states = std::move(all.layer_states);
  outs[0].dsa_selections = std::move(all.dsa_selections);
  int64_t at = 0;
  for (int s = 0; s < n; ++s) {
    Outputs& o = outs[static_cast<size_t>(s)];
    o.lm_vocab_begin = all.lm_vocab_begin;
    o.lm_vocab_count = all.lm_vocab_count;
    o.logits.assign(all.logits.begin() + static_cast<std::ptrdiff_t>(s) * lm_vocab_count_,
                    all.logits.begin() + static_cast<std::ptrdiff_t>(s + 1) * lm_vocab_count_);
    o.final_hidden_bits.assign(all.final_hidden_bits.begin() + static_cast<std::ptrdiff_t>(s) * static_cast<std::ptrdiff_t>(H),
                               all.final_hidden_bits.begin() + static_cast<std::ptrdiff_t>(s + 1) * static_cast<std::ptrdiff_t>(H));
    const int req = reqs[s];
    const int64_t P = span_lens[s];
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
  for (int s = 0; s < n; ++s) report_prefill_progress(reqs[s], span_lens[s]);
  return outs;
}

template <class D>
typename SessionModel<D>::Outputs SessionModel<D>::session_prefill(int req, const std::vector<int64_t>& prompt_ids,
                                                                   const std::vector<int64_t>& boundaries,
                                                                   SnapshotRequest* snap) {
  check_req(req, "session_prefill");
  const int64_t P = static_cast<int64_t>(prompt_ids.size());
  if (P <= 0) throw std::invalid_argument("session_prefill: empty prompt");
  if (P > max_context_) throw std::invalid_argument("session_prefill: prompt exceeds the context bound");
  if (P > max_tokens_ && chunk_tokens_ <= 0)
    throw std::invalid_argument(
        "session_prefill: prompt exceeds max_tokens and the model cannot chunk it (max_tokens is below one "
        "snapshot unit)");
  for (int64_t id : prompt_ids)
    if (id < 0 || id >= vocab_size_) throw std::invalid_argument("session_prefill: token id out of range");
  open_slot(req);
  if (derived().has_pool() && !derived().pool().ensure_request_blocks(req, P, stream_))
    throw std::runtime_error("session_prefill: the cache pool cannot cover the prompt (admission budget)");
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

template <class D>
typename SessionModel<D>::Outputs SessionModel<D>::session_prefill_resume(
    int req, const std::vector<int64_t>& suffix_ids, const std::vector<int64_t>& boundaries, SnapshotRequest* snap) {
  check_req(req, "session_prefill_resume");
  const int64_t P0 = session_pos_[static_cast<size_t>(req)];
  const int64_t n = static_cast<int64_t>(suffix_ids.size());
  if (P0 <= 0) throw std::invalid_argument("session_prefill_resume: the slot is not attached");
  if (P0 % snapshot_align_ != 0) throw std::invalid_argument("session_prefill_resume: the position is not aligned");
  if (n <= 0) throw std::invalid_argument("session_prefill_resume: empty suffix");
  if (P0 + n > max_context_) throw std::invalid_argument("session_prefill_resume: prompt exceeds the context bound");
  if (n > max_tokens_ && chunk_tokens_ <= 0)
    throw std::invalid_argument("session_prefill_resume: suffix exceeds max_tokens and the model cannot chunk it");
  for (int64_t id : suffix_ids)
    if (id < 0 || id >= vocab_size_) throw std::invalid_argument("session_prefill_resume: token id out of range");
  if (derived().has_pool() && !derived().pool().ensure_request_blocks(req, P0 + n, stream_))
    throw std::runtime_error("session_prefill_resume: the cache pool cannot cover the prompt");
  // A close-time snapshot leaves the draft block one row behind (row P0-1
  // wants tok_{P0}, the suffix's first token): catch it up through the
  // decode-path draft, whose row is exactly that one.
  if (mtp_ && mtp_pos_[static_cast<size_t>(req)] == P0 - 1) (void)session_draft(req, std::vector<int64_t>{suffix_ids[0]});
  if (mtp_ && mtp_pos_[static_cast<size_t>(req)] != P0)
    throw std::logic_error("session_prefill_resume: the draft block's row counter is not at the attach position");
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

// ---- the decode rows --------------------------------------------------------

template <class D>
void SessionModel<D>::decode_host_prep(int req, const std::vector<int64_t>& ids, bool upload, bool device_positions) {
  check_req(req, "session_decode");
  const int T = static_cast<int>(ids.size());
  if (T < 1 || T > kSpecRows)
    throw std::invalid_argument("session_decode: row count must be in [1, " + std::to_string(kSpecRows) + "]");
  const int64_t pos = session_pos_[static_cast<size_t>(req)];
  if (pos <= 0) throw std::invalid_argument("session_decode: no open session on slot " + std::to_string(req));
  for (int64_t id : ids)
    if (id < 0 || id >= vocab_size_) throw std::invalid_argument("session_decode: token id out of range");
  if (pos + T > max_context_) throw std::invalid_argument("session_decode: position exceeds the context bound");
  // The block table must cover every row BEFORE the rows run; a device-
  // driven graph reserved its whole run up front (session_reserve_blocks)
  // and must not grow the table inside a replay.
  if (derived().has_pool() && !device_positions && !derived().pool().ensure_request_blocks(req, pos + T, stream_))
    throw std::runtime_error("session_decode: cache pool exhausted (admission budget) — grow the pool or shed requests");
  for (int r = 0; r < T; ++r) {
    h_req_ids_[r] = req;
    h_step_pos_[r] = pos + r;
    h_token_[r] = ids[static_cast<size_t>(r)];
  }
  h_req_spans_[0] = 0;
  h_req_spans_[1] = T;
  decode_rows_ = T;
  if (upload) {
    glm_upload_i32(h_req_ids_, d_req_ids_, T, stream_);
    if (device_positions)
      glm_spec_positions(d_session_pos_ + req, T, d_step_pos_, stream_);
    else
      glm_upload_i64(h_step_pos_, d_step_pos_, T, stream_);
    glm_upload_i32(h_req_spans_, d_req_spans_, 2, stream_);
  }
}

template <class D>
typename SessionModel<D>::Outputs SessionModel<D>::session_verify(int req, const std::vector<int64_t>& token_ids) {
  decode_host_prep(req, token_ids, /*upload=*/true, /*device_positions=*/false);
  RowRun run;
  run.req = req;
  run.ids = token_ids.data();
  run.T = static_cast<int>(token_ids.size());
  run.pos0 = session_pos_[static_cast<size_t>(req)];
  run.decode = true;
  run.all_rows = true;
  run.snapshots = run.T > 1;
  Outputs out = derived().run_rows(run);
  session_pos_[static_cast<size_t>(req)] += static_cast<int64_t>(token_ids.size());
  push_position(req);
  return out;
}

template <class D>
void SessionModel<D>::session_rollback(int req, int accepted) {
  check_req(req, "session_rollback");
  const int T = decode_rows_;
  if (accepted < 1 || accepted > T)
    throw std::invalid_argument("session_rollback: accepted rows must be in [1, " + std::to_string(T) + "]");
  const int64_t pos = session_pos_[static_cast<size_t>(req)];
  if (pos < T) throw std::invalid_argument("session_rollback: no verify to retract");
  if (accepted == T) return;  // every row landed in place already
  const GlmSpecSegments segs = derived().spec_segments(req, 0);
  for (int i = 0; i < segs.count; ++i) {
    const GlmSpecSegment& s = segs.seg[i];
    session_detail::d2d(s.dst,
                        static_cast<const uint8_t*>(s.snapshots) + static_cast<size_t>(accepted - 1) * s.row_stride_bytes,
                        s.bytes, stream_);
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  session_pos_[static_cast<size_t>(req)] = pos - (T - accepted);
  push_position(req);
}

template <class D>
void SessionModel<D>::session_close(int req) {
  check_req(req, "session_close");
  derived().reset_slot_state(req);
  session_pos_[static_cast<size_t>(req)] = 0;
  push_position(req);
  if (mtp_) {
    mtp_pos_[static_cast<size_t>(req)] = 0;
    push_mtp_position(req);
  }
}

template <class D>
void SessionModel<D>::session_reserve_blocks(int req, int64_t tokens) {
  check_req(req, "session_reserve_blocks");
  if (tokens < 1 || tokens > max_context_)
    throw std::invalid_argument("session_reserve_blocks: tokens outside [1, max_context]");
  if (!derived().has_pool()) return;
  if (!derived().pool().ensure_request_blocks(req, tokens, stream_))
    throw std::runtime_error("session_reserve_blocks: the cache pool cannot cover " + std::to_string(tokens) +
                             " tokens for slot " + std::to_string(req));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}

// ---- the prefix cache: snapshots and attach --------------------------------

template <class D>
void SessionModel<D>::write_snapshot(int req, void* dst, int spec_row) {
  uint8_t* d = static_cast<uint8_t*>(dst);
  derived().write_state_snapshot(req, d, spec_row);
  d += derived().snapshot_state_bytes();
  if (mtp_) {
    const bool live = spec_row < 0;
    const int64_t pos = live ? session_pos_[static_cast<size_t>(req)]
                             : session_pos_[static_cast<size_t>(req)] - rows_after_for_snapshot_;
    derived().write_draft_snapshot(req, d, live, pos);
    d += derived().draft_state_bytes();
    const size_t W = static_cast<size_t>(draft_width_);
    session_detail::d2d(d, mtp_window(req) + static_cast<size_t>((pos - 1) % max_decode_rows_) * W, W * 2, stream_);
  }
}

template <class D>
typename SessionModel<D>::SessionSnapshotMeta SessionModel<D>::pin_blocks_at(int req, int64_t pos,
                                                                              const char* what) {
  SessionSnapshotMeta meta;
  meta.position = pos;
  if (!derived().has_pool()) return meta;
  auto& pool = derived().pool();
  const int64_t n_full = pos / block_tokens_;
  const int32_t* row = pool.request_table_row(req);
  meta.full_blocks.assign(row, row + n_full);
  pool.pin_blocks(meta.full_blocks.data(), n_full);
  if (pos % block_tokens_ != 0) {
    const int32_t b = pool.acquire_pinned_block();
    if (b < 0) {
      pool.unpin_blocks(meta.full_blocks.data(), n_full);
      throw std::runtime_error(std::string(what) + ": cache pool exhausted (the partial block)");
    }
    pool.copy_block_contents(row[n_full], b, stream_);
    meta.partial_block = b;
  }
  return meta;
}

template <class D>
typename SessionModel<D>::SessionSnapshotMeta SessionModel<D>::session_snapshot(int req, void* dst) {
  check_req(req, "session_snapshot");
  const int64_t pos = session_pos_[static_cast<size_t>(req)];
  if (pos <= 0) throw std::invalid_argument("session_snapshot: the slot is closed");
  if (pos % snapshot_align_ != 0) throw std::invalid_argument("session_snapshot: the position is not aligned");
  if (dst == nullptr) throw std::invalid_argument("session_snapshot: null buffer");
  if (mtp_) {
    const int64_t q = mtp_pos_[static_cast<size_t>(req)];
    if (q != pos && q != pos - 1)
      throw std::logic_error("session_snapshot: the draft block is more than one row behind");
  }
  write_snapshot(req, dst, -1);
  SessionSnapshotMeta meta = pin_blocks_at(req, pos, "session_snapshot");
  meta.mtp_position = mtp_ ? mtp_pos_[static_cast<size_t>(req)] : 0;
  return meta;
}

template <class D>
typename SessionModel<D>::SessionSnapshotMeta SessionModel<D>::session_snapshot_post_row0(int req, void* dst,
                                                                                          int spec_row,
                                                                                          int rows_after) {
  check_req(req, "session_snapshot_post_row0");
  if (rows_after < 1 || rows_after >= kSpecRows)
    throw std::invalid_argument("session_snapshot_post_row0: rows after the position");
  const int64_t pos = session_pos_[static_cast<size_t>(req)] - rows_after;
  if (pos <= 0) throw std::invalid_argument("session_snapshot_post_row0: no verified rows");
  if (pos % snapshot_align_ != 0)
    throw std::invalid_argument("session_snapshot_post_row0: the position after row 0 is not aligned");
  if (spec_row < 0 || spec_row >= max_decode_rows_)
    throw std::out_of_range("session_snapshot_post_row0: spec row " + std::to_string(spec_row));
  if (dst == nullptr) throw std::invalid_argument("session_snapshot_post_row0: null buffer");
  if (mtp_) {
    const int64_t q = mtp_pos_[static_cast<size_t>(req)];
    if (q != pos + rows_after && q != pos - 1)
      throw std::logic_error(
          "session_snapshot_post_row0: the draft block's counter is neither before nor after the step's rows");
  }
  rows_after_for_snapshot_ = rows_after;
  write_snapshot(req, dst, spec_row);
  rows_after_for_snapshot_ = 0;
  SessionSnapshotMeta meta = pin_blocks_at(req, pos, "session_snapshot_post_row0");
  meta.mtp_position = mtp_ ? pos - 1 : 0;
  return meta;
}

template <class D>
void SessionModel<D>::session_release_snapshot(const SessionSnapshotMeta& meta) {
  if (!derived().has_pool()) return;
  auto& pool = derived().pool();
  if (!meta.full_blocks.empty()) pool.unpin_blocks(meta.full_blocks.data(), static_cast<int64_t>(meta.full_blocks.size()));
  if (meta.partial_block >= 0) pool.unpin_blocks(&meta.partial_block, 1);
}

template <class D>
void SessionModel<D>::session_attach(int req, const void* src, const SessionSnapshotMeta& meta) {
  check_req(req, "session_attach");
  if (session_pos_[static_cast<size_t>(req)] != 0) throw std::logic_error("session_attach: the slot is open");
  if (meta.position <= 0 || meta.position % snapshot_align_ != 0)
    throw std::invalid_argument("session_attach: bad snapshot position");
  if (meta.position > max_context_) throw std::invalid_argument("session_attach: position exceeds the context bound");
  if (src == nullptr) throw std::invalid_argument("session_attach: null buffer");
  open_slot(req);
  const uint8_t* d = static_cast<const uint8_t*>(src);
  derived().read_state_snapshot(req, d);
  d += derived().snapshot_state_bytes();
  if (mtp_) {
    derived().read_draft_snapshot(req, d);
    d += derived().draft_state_bytes();
    const size_t W = static_cast<size_t>(draft_width_);
    session_detail::d2d(mtp_window(req) + static_cast<size_t>((meta.position - 1) % max_decode_rows_) * W, d, W * 2,
                        stream_);
    d += W * 2;
    mtp_pos_[static_cast<size_t>(req)] = meta.mtp_position;
    push_mtp_position(req);
  }
  if (derived().has_pool()) {
    auto& pool = derived().pool();
    const int64_t n_full = meta.position / block_tokens_;
    if (static_cast<int64_t>(meta.full_blocks.size()) != n_full)
      throw std::invalid_argument("session_attach: block list does not match the position");
    if (!pool.share_blocks_into(req, meta.full_blocks.data(), n_full, stream_))
      throw std::runtime_error("session_attach: could not share the prefix blocks");
    if (meta.position % block_tokens_ != 0) {
      if (meta.partial_block < 0) throw std::invalid_argument("session_attach: the snapshot lacks its partial block");
      if (!pool.ensure_request_blocks(req, meta.position, stream_))
        throw std::runtime_error("session_attach: cache pool exhausted");
      const int32_t* row = pool.request_table_row(req);
      pool.copy_block_contents(meta.partial_block, row[n_full], stream_);
    }
  }
  session_pos_[static_cast<size_t>(req)] = meta.position;
  push_position(req);
}

// ---- the graph era ----------------------------------------------------------

template <class D>
void SessionModel<D>::session_graph_capture_step(int req, const std::vector<int64_t>& ids, bool device_positions,
                                                 bool device_tokens, int feed_rows) {
  if (device_tokens && !device_positions)
    throw std::invalid_argument("session_graph_capture_step: device tokens need device positions");
  if (feed_rows < 0 || feed_rows > kSpecRows || (feed_rows > 0 && static_cast<size_t>(feed_rows) < ids.size()))
    throw std::invalid_argument("session_graph_capture_step: the feed must hold at least the verified rows");
  graph_device_positions_ = device_positions;
  graph_device_tokens_ = device_tokens;
  graph_has_draft_ = false;
  graph_batch_requests_ = 0;
  graph_rows_per_request_ = 0;
  graph_feed_rows_ = feed_rows > 0 ? feed_rows : static_cast<int>(ids.size());
  // The slot's scalar variant reads and writes its own persistent feed
  // rows (device_feed), so its feed survives the other slots' replays; a
  // reduced-depth variant reads the first ids.size() rows of the same feed.
  step_tokens_ = d_tokens_ + static_cast<size_t>(max_decode_rows_) +
                 static_cast<size_t>(req) * static_cast<size_t>(graph_feed_rows_);
  decode_host_prep(req, ids, /*upload=*/true, device_positions);
  RowRun run;
  run.req = req;
  run.ids = ids.data();
  run.T = static_cast<int>(ids.size());
  run.pos0 = session_pos_[static_cast<size_t>(req)];
  run.decode = true;
  run.all_rows = true;
  run.capture = true;
  run.snapshots = run.T > 1;
  (void)derived().run_rows(run);  // records; nothing executes, no state, no position
}

template <class D>
void SessionModel<D>::session_graph_capture_commit(int req, const PickVerdict* device_verdict) {
  check_req(req, "session_graph_capture_commit");
  if (!graph_device_positions_)
    throw std::logic_error("session_graph_capture_commit: the step must be captured with device positions");
  if (device_verdict == nullptr) throw std::invalid_argument("session_graph_capture_commit: null verdict");
  glm_spec_commit(device_verdict, decode_rows_, derived().spec_segments(req, 0), d_session_pos_ + req, stream_);
}

template <class D>
void SessionModel<D>::session_graph_settle(int req, int accepted, int rows) {
  check_req(req, "session_graph_settle");
  if (!graph_device_positions_)
    throw std::logic_error("session_graph_settle: the graph was not captured with device positions");
  const int bound = rows > 0 ? rows : (graph_batch_requests_ > 0 ? graph_rows_per_request_ : decode_rows_);
  if (accepted < 1 || accepted > bound)
    throw std::invalid_argument("session_graph_settle: accepted rows outside [1, " + std::to_string(bound) + "]");
  if (session_pos_[static_cast<size_t>(req)] <= 0) throw std::invalid_argument("session_graph_settle: no open session");
  // The device advanced its own position in the recorded commit; the
  // mirror follows (no push: the device is the source of truth here). With
  // the draft in the graph the block's counter moved by the same rows.
  session_pos_[static_cast<size_t>(req)] += accepted;
  if (graph_has_draft_) mtp_pos_[static_cast<size_t>(req)] += accepted;
}

template <class D>
typename SessionModel<D>::Outputs SessionModel<D>::session_graph_outputs(int req) {
  check_req(req, "session_graph_outputs");
  if (session_pos_[static_cast<size_t>(req)] <= 0) throw std::invalid_argument("session_graph_outputs: no open session");
  const size_t rows = static_cast<size_t>(decode_rows_);
  const size_t H = static_cast<size_t>(hidden_);
  if (!decode_tail_mirrors_) {
    // The kernels-only graph carried no D2H nodes: mirror the tail now
    // (the caller synced the stream and finished the bus window).
    DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_logits_, logits_, rows * lm_vocab_count_ * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
    DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_hidden_, h_, rows * H * 2, cudaMemcpyDeviceToHost, stream_));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  }
  Outputs out;
  out.lm_vocab_begin = lm_vocab_begin_;
  out.lm_vocab_count = lm_vocab_count_;
  out.logits.assign(h_tail_logits_, h_tail_logits_ + rows * lm_vocab_count_);
  out.final_hidden_bits.assign(h_tail_hidden_, h_tail_hidden_ + rows * H);
  return out;
}

template <class D>
typename SessionModel<D>::Outputs SessionModel<D>::session_graph_collect(int req) {
  if (graph_device_positions_)
    throw std::logic_error("session_graph_collect: a device-driven graph settles (session_graph_settle)");
  Outputs out = session_graph_outputs(req);
  session_pos_[static_cast<size_t>(req)] += decode_rows_;
  push_position(req);
  return out;
}

template <class D>
void SessionModel<D>::session_graph_seed_tokens(int req, const std::vector<int64_t>& ids) {
  check_req(req, "session_graph_seed_tokens");
  const int rows_per_request = graph_batch_requests_ > 0 ? graph_rows_per_request_ : decode_rows_;
  if (ids.size() != static_cast<size_t>(rows_per_request))
    throw std::invalid_argument("session_graph_seed_tokens: the seed must have the graph's row count");
  session_graph_seed_feed(req, ids);
}

template <class D>
void SessionModel<D>::session_graph_seed_scalar_tokens(const std::vector<int64_t>& ids) {
  if (ids.empty() || ids.size() > static_cast<size_t>(kSpecRows))
    throw std::invalid_argument("session_graph_seed_scalar_tokens: invalid row count");
  for (int64_t id : ids)
    if (id < 0 || id >= vocab_size_) throw std::invalid_argument("session_graph_seed_scalar_tokens: token id");
  for (size_t i = 0; i < ids.size(); ++i) h_token_[i] = ids[i];
  DGPP_CUDA_OK(cudaMemcpyAsync(d_tokens_, h_token_, ids.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream_));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}

template <class D>
void SessionModel<D>::session_graph_seed_feed(int req, const std::vector<int64_t>& ids) {
  check_req(req, "session_graph_seed_feed");
  if (ids.empty() || ids.size() > static_cast<size_t>(kSpecRows))
    throw std::invalid_argument("session_graph_seed_feed: invalid row count");
  for (int64_t id : ids)
    if (id < 0 || id >= vocab_size_) throw std::invalid_argument("session_graph_seed_feed: token id");
  // The slot's own pinned rows and its device feed rows; on the model's
  // stream, never the legacy stream (a peer rank in the same process may
  // be mid-capture).
  const size_t row0 = static_cast<size_t>(max_decode_rows_) + static_cast<size_t>(req) * ids.size();
  for (size_t i = 0; i < ids.size(); ++i) h_token_[row0 + i] = ids[i];
  DGPP_CUDA_OK(cudaMemcpyAsync(d_tokens_ + row0, h_token_ + row0, ids.size() * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}

template <class D>
void SessionModel<D>::session_graph_capture_batch(int rows_per_request, int requests_arg, int feed_rows) {
  const int requests = requests_arg > 0 ? requests_arg : max_requests_;
  if (rows_per_request < 1 || rows_per_request > kSpecRows || requests < 1 || requests > max_requests_ ||
      requests * rows_per_request > max_decode_rows_)
    throw std::invalid_argument("session_graph_capture_batch: requests * rows_per_request must fit the decode-row ceiling (" +
                                std::to_string(max_decode_rows_) + ")");
  if (feed_rows < 0 || feed_rows > kSpecRows || (feed_rows > 0 && feed_rows < rows_per_request) ||
      (feed_rows > 0 && requests * feed_rows > max_decode_rows_))
    throw std::invalid_argument("session_graph_capture_batch: the feeds must hold at least the verified rows");
  if (std::none_of(session_pos_.begin(), session_pos_.end(), [](int64_t p) { return p > 0; }))
    throw std::logic_error("session_graph_capture_batch: capture needs one open request");
  const int rows = requests * rows_per_request;
  if (rows > max_tokens_) throw std::invalid_argument("session_graph_capture_batch: fixed rows exceed model max_tokens");
  graph_device_positions_ = true;
  graph_device_tokens_ = true;
  graph_has_draft_ = false;
  graph_batch_requests_ = requests;
  graph_rows_per_request_ = rows_per_request;
  graph_feed_rows_ = feed_rows > rows_per_request ? feed_rows : 0;
  step_tokens_ = d_tokens_ + static_cast<size_t>(max_decode_rows_);  // the fixed batch: every slot's feed rows
  if (graph_batch_map_source_)
    glm_upload_i32(graph_batch_map_source_, d_batch_map_, requests, stream_);
  if (graph_feed_rows_ > 0 || graph_batch_map_source_) {
    // The reduced-depth batch: the first rows_per_request rows of every
    // slot's feed, compacted into the front scratch (free during a replay),
    // so the walk's rows stay contiguous per request; the next-tokens write
    // at the end of the replay lands in the feeds themselves.
    glm_spec_gather_feed(d_tokens_ + static_cast<size_t>(max_decode_rows_), requests,
                         graph_feed_rows_ > 0 ? graph_feed_rows_ : rows_per_request,
                         rows_per_request, d_tokens_, stream_,
                         graph_batch_map_source_ ? d_batch_map_ : nullptr);
    step_tokens_ = d_tokens_;
  }
  decode_rows_ = rows;
  for (int q = 0; q < requests; ++q) {
    h_req_spans_[2 * q] = q * rows_per_request;
    h_req_spans_[2 * q + 1] = rows_per_request;
    for (int r = 0; r < rows_per_request; ++r) h_req_ids_[q * rows_per_request + r] = q;
  }
  if (graph_batch_map_source_)
    glm_batch_rows(d_batch_map_, requests, rows_per_request, d_req_ids_, d_req_spans_, stream_);
  else {
    glm_upload_i32(h_req_ids_, d_req_ids_, rows, stream_);
    glm_upload_i32(h_req_spans_, d_req_spans_, 2 * requests, stream_);
  }
  glm_spec_positions_batched(d_session_pos_, d_req_ids_, rows, rows_per_request, d_step_pos_, stream_);
  const std::vector<int64_t> shape(static_cast<size_t>(rows), 0);
  RowRun run;
  run.req = 0;
  run.ids = shape.data();
  run.T = rows;
  run.pos0 = 0;
  run.decode = true;
  run.all_rows = true;
  run.capture = true;
  run.batch_requests = requests;
  run.snapshots = true;
  (void)derived().run_rows(run);
}

template <class D>
void SessionModel<D>::session_graph_stage_batch() {
  if (graph_batch_requests_ <= 0 || graph_rows_per_request_ <= 0)
    throw std::logic_error("session_graph_stage_batch: no fixed batch was captured");
  for (int q = 0; q < graph_batch_requests_; ++q) {
    h_req_spans_[2 * q] = q * graph_rows_per_request_;
    h_req_spans_[2 * q + 1] = graph_rows_per_request_;
    for (int r = 0; r < graph_rows_per_request_; ++r) h_req_ids_[q * graph_rows_per_request_ + r] = q;
  }
}

template <class D>
void SessionModel<D>::session_graph_use_batch_contract(int rows_per_request, int requests_arg) {
  const int requests = requests_arg > 0 ? requests_arg : max_requests_;
  if (rows_per_request < 1 || rows_per_request > kSpecRows || requests < 1 || requests > max_requests_ ||
      requests * rows_per_request > max_decode_rows_)
    throw std::invalid_argument("session_graph_use_batch_contract: invalid fixed batch shape");
  graph_device_positions_ = true;
  graph_device_tokens_ = true;
  graph_has_draft_ = mtp_;
  graph_batch_requests_ = requests;
  graph_rows_per_request_ = rows_per_request;
  decode_rows_ = requests * rows_per_request;
  if (mtp_) draft_rows_ = decode_rows_;
}

template <class D>
void SessionModel<D>::session_graph_capture_commit_batch(const PickVerdict* device_verdicts) {
  if (graph_batch_requests_ <= 0 || graph_rows_per_request_ <= 0)
    throw std::logic_error("session_graph_capture_commit_batch: no fixed batch was captured");
  if (device_verdicts == nullptr) throw std::invalid_argument("session_graph_capture_commit_batch: null verdicts");
  for (int req = 0; req < graph_batch_requests_; ++req) {
    const int row0 = req * graph_rows_per_request_;
    if (graph_batch_map_source_) {
      if (max_requests_ < 2)
        throw std::logic_error("compact commit: stride requires at least two request slots");
      auto segments = derived().spec_segments(0, row0);
      const auto next = derived().spec_segments(1, row0);
      for (int i = 0; i < segments.count; ++i)
        segments.seg[i].request_stride_bytes = reinterpret_cast<uintptr_t>(next.seg[i].dst) -
                                               reinterpret_cast<uintptr_t>(segments.seg[i].dst);
#ifndef NDEBUG
      // Mapped commits require each destination to be affine in the physical request ID.
      for (int q = 0; q < max_requests_; ++q) {
        const auto actual = derived().spec_segments(q, row0);
        if (actual.count != segments.count)
          throw std::logic_error("compact commit: segment count differs by request");
        for (int i = 0; i < segments.count; ++i) {
          const auto expected = reinterpret_cast<uintptr_t>(segments.seg[i].dst) +
                                q * segments.seg[i].request_stride_bytes;
          if (reinterpret_cast<uintptr_t>(actual.seg[i].dst) != expected)
            throw std::logic_error("compact commit: nonuniform request destination stride");
        }
      }
#endif
      glm_spec_commit(device_verdicts + req, graph_rows_per_request_, segments, d_session_pos_,
                      stream_, d_batch_map_, req);
    } else {
      glm_spec_commit(device_verdicts + req, graph_rows_per_request_,
                      derived().spec_segments(req, row0), d_session_pos_ + req, stream_);
    }
  }
}

template <class D>
void SessionModel<D>::session_graph_capture_verify_next_tokens_batch(const PickVerdict* verify_verdicts) {
  if (graph_batch_requests_ <= 0 || graph_rows_per_request_ != 1 || mtp_)
    throw std::logic_error("session_graph_capture_verify_next_tokens_batch: requires the plain T=1 fixed graph");
  glm_spec_verify_next_tokens_batched(
      verify_verdicts, graph_batch_requests_, graph_rows_per_request_,
      graph_batch_map_source_ ? d_tokens_ + max_decode_rows_ : step_tokens_, stream_,
      graph_batch_map_source_ ? d_batch_map_ : nullptr);
}

// ---- the MTP draft block ------------------------------------------------------

template <class D>
void SessionModel<D>::mtp_prefill_rows(int req, int64_t row0, int64_t row1, const int64_t* tokens) {
  const int64_t n = row1 - row0;
  if (n <= 0) return;
  if (n > max_tokens_) throw std::invalid_argument("mtp_prefill_rows: chunk too long");
  DGPP_CUDA_OK(cudaMemcpyAsync(d_tokens_, tokens, static_cast<size_t>(n) * 8, cudaMemcpyHostToDevice, stream_));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  if constexpr (requires { derived().mtp_select_block(0); }) derived().mtp_select_block(0);
  derived().mtp_run_rows(req, d_tokens_, row0, static_cast<int>(n), /*decode_row=*/false, /*capture=*/false,
                         /*head_rows=*/0, /*batch_requests=*/0);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}

template <class D>
void SessionModel<D>::mtp_decode_host_prep(int req, const std::vector<int64_t>& tokens, bool upload) {
  if (!mtp_) refuse_mtp("session_draft");
  check_req(req, "session_draft");
  const int T = static_cast<int>(tokens.size());
  if (T < 1 || T > kSpecRows)
    throw std::invalid_argument("session_draft: row count must be in [1, " + std::to_string(kSpecRows) + "]");
  const int64_t pos = session_pos_[static_cast<size_t>(req)];
  const int64_t q = mtp_pos_[static_cast<size_t>(req)];
  if (pos <= 0) throw std::invalid_argument("session_draft: no open session on slot " + std::to_string(req));
  if (q + T != pos)
    throw std::invalid_argument("session_draft: " + std::to_string(T) + " rows from draft position " +
                                std::to_string(q) + " do not reach the session position " + std::to_string(pos) +
                                " (draft exactly the accepted rows)");
  for (int64_t id : tokens)
    if (id < 0 || id >= vocab_size_) throw std::invalid_argument("session_draft: token id out of range");
  for (int r = 0; r < T; ++r) {
    h_req_ids_[r] = req;
    h_step_pos_[r] = q + r;
    h_token_[r] = tokens[static_cast<size_t>(r)];
  }
  h_req_spans_[0] = 0;
  h_req_spans_[1] = T;
  draft_rows_ = T;
  if (upload) {
    glm_upload_i32(h_req_ids_, d_req_ids_, T, stream_);
    glm_upload_i64(h_step_pos_, d_step_pos_, T, stream_);
    glm_upload_i32(h_req_spans_, d_req_spans_, 2, stream_);
    glm_upload_i64(h_token_, d_tokens_, T, stream_);
  }
}

template <class D>
typename SessionModel<D>::Outputs SessionModel<D>::session_draft(int req, const std::vector<int64_t>& tokens) {
  step_tokens_ = d_tokens_;
  mtp_decode_host_prep(req, tokens, /*upload=*/true);
  const int64_t q = mtp_pos_[static_cast<size_t>(req)];
  if constexpr (requires { derived().mtp_select_block(0); }) derived().mtp_select_block(0);
  derived().mtp_run_rows(req, d_tokens_, q, static_cast<int>(tokens.size()), /*decode_row=*/true, /*capture=*/false,
                         /*head_rows=*/1, 0);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  mtp_pos_[static_cast<size_t>(req)] = q + static_cast<int64_t>(tokens.size());
  push_mtp_position(req);
  draft_last_row_ = static_cast<int>(tokens.size()) - 1;  // the chain's first hidden row
  Outputs out;
  out.lm_vocab_begin = lm_vocab_begin_;
  out.lm_vocab_count = lm_vocab_count_;
  out.logits.assign(h_tail_logits_, h_tail_logits_ + static_cast<size_t>(lm_vocab_count_));
  return out;
}

template <class D>
void SessionModel<D>::session_draft_rollback(int req, int rows) {
  if (!mtp_) refuse_mtp("session_draft_rollback");
  check_req(req, "session_draft_rollback");
  if (rows < 1 || rows > mtp_pos_[static_cast<size_t>(req)])
    throw std::invalid_argument("session_draft_rollback: rows outside the block's counter");
  // The block's own state before its rows; every other draft state is
  // positional (the re-run overwrites it).
  derived().restore_draft_state(req);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  mtp_pos_[static_cast<size_t>(req)] -= rows;
  push_mtp_position(req);
}

template <class D>
void SessionModel<D>::session_graph_capture_draft(int req, const PickVerdict* verify_verdict) {
  if (!mtp_) refuse_mtp("session_graph_capture_draft");
  check_req(req, "session_graph_capture_draft");
  if (!graph_device_positions_)
    throw std::logic_error("session_graph_capture_draft: the step must be captured with device positions");
  if (verify_verdict == nullptr) throw std::invalid_argument("session_graph_capture_draft: null verdict");
  const int T = decode_rows_;
  derived().snapshot_draft_state(req);
  // d_req_ids_/d_req_spans_ still describe T rows of `req` from the verify
  // (same batch shape); the rows' positions and tokens come off the verdict.
  glm_spec_draft_rows(verify_verdict, T, d_mtp_pos_ + req, d_step_pos_, step_tokens_, d_next_ + req, stream_);
  draft_rows_ = T;
  if constexpr (requires { derived().mtp_select_block(0); }) derived().mtp_select_block(0);
  derived().mtp_run_rows(req, step_tokens_, 0, T, /*decode_row=*/true, /*capture=*/true, /*head_rows=*/T, 0);
  graph_has_draft_ = true;
}

template <class D>
void SessionModel<D>::session_graph_capture_next_tokens(int req,
                                                        const std::vector<const PickVerdict*>& draft_verdicts) {
  check_req(req, "session_graph_capture_next_tokens");
  if (draft_verdicts.size() == 1) {
    session_graph_capture_next_tokens(req, draft_verdicts[0]);
    return;
  }
  if (!graph_device_tokens_ || !graph_has_draft_)
    throw std::logic_error("session_graph_capture_next_tokens: needs a device-token capture with the draft in the graph");
  if (draft_verdicts.empty() || draft_verdicts.size() > static_cast<size_t>(kSpecMaxDrafts))
    throw std::invalid_argument("session_graph_capture_next_tokens: draft count");
  // The feed carries every draft of the block whatever the verify's rows.
  const int feed_rows = graph_feed_rows_ > 0 ? graph_feed_rows_ : decode_rows_;
  if (feed_rows != 1 + static_cast<int>(draft_verdicts.size()))
    throw std::logic_error("session_graph_capture_next_tokens: the token feed is [next, draft_1 .. draft_n] (T = 1 + n)");
  GlmSpecDrafts d;
  for (size_t i = 0; i < draft_verdicts.size(); ++i) {
    if (draft_verdicts[i] == nullptr) throw std::invalid_argument("session_graph_capture_next_tokens: null verdict");
    d.v[i] = draft_verdicts[i];
  }
  d.count = static_cast<int>(draft_verdicts.size());
  glm_spec_next_tokens(d_next_ + req, d, step_tokens_, stream_);
}

template <class D>
bool SessionModel<D>::session_draft_chain_fits(int req, int index) const {
  if (!D::kDraftChain || !mtp_ || req < 0 || req >= max_requests_ || index < 0) return false;
  return mtp_pos_[static_cast<size_t>(req)] + index < max_context_;
}

template <class D>
typename SessionModel<D>::Outputs SessionModel<D>::session_draft_chain(int req, int64_t token, int index,
                                                                       bool first, bool last) {
  if (!D::kDraftChain) throw std::logic_error("session_draft_chain (depth >= 2): not wired for this family");
  if (!mtp_) refuse_mtp("session_draft_chain");
  check_req(req, "session_draft_chain");
  if (index < 0 || index >= kSpecRows - 1) throw std::invalid_argument("session_draft_chain: chain index");
  if (token < 0 || token >= vocab_size_) throw std::invalid_argument("session_draft_chain: token id out of range");
  const int64_t q = mtp_pos_[static_cast<size_t>(req)];
  if (session_pos_[static_cast<size_t>(req)] <= 0 || q < 1)
    throw std::invalid_argument("session_draft_chain: no drafted session on slot " + std::to_string(req));
  const int64_t pos = q + index;
  if (pos >= max_context_) throw std::invalid_argument("session_draft_chain: position exceeds the context bound");
  step_tokens_ = d_tokens_;
  if (first) derived().snapshot_chain_state(req);
  h_req_ids_[0] = req;
  h_step_pos_[0] = pos;
  h_token_[0] = token;
  h_req_spans_[0] = 0;
  h_req_spans_[1] = 1;
  glm_upload_i32(h_req_ids_, d_req_ids_, 1, stream_);
  glm_upload_i64(h_step_pos_, d_step_pos_, 1, stream_);
  glm_upload_i32(h_req_spans_, d_req_spans_, 2, stream_);
  glm_upload_i64(h_token_, d_tokens_, 1, stream_);
  // The block's previous output row is the row's hidden, in the window at
  // the row's position (the store reads the staged position).
  store_draft_hidden(derived().draft_hidden_rows() + static_cast<size_t>(draft_last_row_) * draft_width_, d_req_ids_,
                     d_step_pos_, 1);
  if constexpr (requires { derived().mtp_select_block(0); }) derived().mtp_select_block(index + 1);
  derived().mtp_run_rows(req, d_tokens_, pos, /*T=*/1, /*decode_row=*/true, /*capture=*/false, /*head_rows=*/1, 0);
  if (last) derived().restore_chain_state(req);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  draft_last_row_ = 0;
  Outputs out;
  out.lm_vocab_begin = lm_vocab_begin_;
  out.lm_vocab_count = lm_vocab_count_;
  out.logits.assign(h_tail_logits_, h_tail_logits_ + static_cast<size_t>(lm_vocab_count_));
  return out;
}

template <class D>
void SessionModel<D>::session_graph_capture_draft_chain(int req, const PickVerdict* verify_verdict,
                                                        const PickVerdict* draft_verdict, int index, bool first,
                                                        bool last) {
  if (!D::kDraftChain) throw std::logic_error("session_graph_capture_draft_chain (depth >= 2): not wired for this family");
  if (!mtp_) refuse_mtp("session_graph_capture_draft_chain");
  check_req(req, "session_graph_capture_draft_chain");
  if (!graph_device_positions_ || !graph_device_tokens_ || !graph_has_draft_)
    throw std::logic_error("session_graph_capture_draft_chain: needs the device-driven capture with the draft in the graph");
  if (draft_verdict == nullptr || (index == 0 && verify_verdict == nullptr))
    throw std::invalid_argument("session_graph_capture_draft_chain: null verdict");
  if (index < 0 || index >= kSpecRows - 1) throw std::invalid_argument("session_graph_capture_draft_chain: chain index");
  if (first) derived().snapshot_chain_state(req);
  // The row's position, token and hidden (the block's last accepted row
  // for the first chain row, its previous chain row after) on the device.
  glm_spec_chain_row_window(index == 0 ? verify_verdict : nullptr, /*src_row=*/0, draft_verdict,
                            derived().draft_hidden_rows(), draft_width_, mtp_window(req), max_decode_rows_,
                            d_mtp_pos_ + req, index, max_context_, d_step_pos_, step_tokens_, d_req_spans_, stream_);
  if constexpr (requires { derived().mtp_select_block(0); }) derived().mtp_select_block(index + 1);
  derived().mtp_run_rows(req, step_tokens_, 0, /*T=*/1, /*decode_row=*/true, /*capture=*/true, /*head_rows=*/1, 0);
  if (last) derived().restore_chain_state(req);
}

template <class D>
void SessionModel<D>::session_graph_capture_draft_batch(const PickVerdict* verify_verdicts) {
  if (!mtp_) refuse_mtp("session_graph_capture_draft_batch");
  if (graph_batch_requests_ <= 0 || graph_rows_per_request_ < 2 || !graph_device_positions_ || !graph_device_tokens_)
    throw std::logic_error(
        "session_graph_capture_draft_batch: requires a device-driven fixed batch of 1 + depth rows per request");
  if (verify_verdicts == nullptr) throw std::invalid_argument("session_graph_capture_draft_batch: null verdicts");
  for (int req = 0; req < (graph_batch_map_source_ ? max_requests_ : graph_batch_requests_); ++req)
    derived().snapshot_draft_state(req);
  glm_spec_draft_rows_batched(verify_verdicts, graph_batch_requests_, graph_rows_per_request_,
                              d_mtp_pos_, d_step_pos_, step_tokens_, d_next_, stream_,
                              graph_batch_map_source_ ? d_batch_map_ : nullptr);
  draft_rows_ = decode_rows_;
  if constexpr (requires { derived().mtp_select_block(0); }) derived().mtp_select_block(0);
  derived().mtp_run_rows(/*req=*/0, step_tokens_, 0, decode_rows_, /*decode_row=*/true, /*capture=*/true,
                         /*head_rows=*/decode_rows_, graph_batch_requests_);
  graph_has_draft_ = true;
}

template <class D>
void SessionModel<D>::session_graph_capture_draft_chain_batch(const PickVerdict* verify_verdicts,
                                                              const PickVerdict* draft_verdicts, int index, bool first,
                                                              bool last) {
  if (!D::kDraftChain)
    throw std::logic_error("session_graph_capture_draft_chain_batch (depth >= 2): not wired for this family");
  if (!mtp_) refuse_mtp("session_graph_capture_draft_chain_batch");
  if (graph_batch_requests_ <= 0 || graph_rows_per_request_ < 2 || !graph_device_positions_ || !graph_device_tokens_ ||
      !graph_has_draft_)
    throw std::logic_error("session_graph_capture_draft_chain_batch: needs the fixed batch with the draft in the graph");
  if (verify_verdicts == nullptr || draft_verdicts == nullptr)
    throw std::invalid_argument("session_graph_capture_draft_chain_batch: null verdicts");
  if (index < 0 || index >= kSpecRows - 1)
    throw std::invalid_argument("session_graph_capture_draft_chain_batch: chain index");
  const int k = graph_batch_requests_;
  if (first)
    for (int q = 0; q < (graph_batch_map_source_ ? max_requests_ : k); ++q)
      derived().snapshot_chain_state(q);
  // Every slot's chain row (its position, token and hidden) into the
  // compact one-row-per-request layout, then the block over the k rows
  // with the head on each; the picks read rows 0 .. k-1.
  glm_spec_chain_rows_batched(
      verify_verdicts, draft_verdicts, k, graph_rows_per_request_, derived().draft_hidden_rows(),
      draft_width_, mtp_window_, max_decode_rows_,
      static_cast<size_t>(max_decode_rows_) * static_cast<size_t>(draft_width_), d_mtp_pos_, index,
      max_context_, d_step_pos_, step_tokens_, d_req_ids_, d_req_spans_, stream_,
      graph_batch_map_source_ ? d_batch_map_ : nullptr);
  if constexpr (requires { derived().mtp_select_block(0); }) derived().mtp_select_block(index + 1);
  derived().mtp_run_rows(/*req=*/0, step_tokens_, 0, /*T=*/k, /*decode_row=*/true, /*capture=*/true, /*head_rows=*/k, k);
  if (last)
    for (int q = 0; q < (graph_batch_map_source_ ? max_requests_ : k); ++q)
      derived().restore_chain_state(q);
}

template <class D>
void SessionModel<D>::session_graph_capture_next_tokens(int req, const PickVerdict* draft_verdict) {
  check_req(req, "session_graph_capture_next_tokens");
  if (!graph_device_tokens_ || !graph_has_draft_)
    throw std::logic_error("session_graph_capture_next_tokens: needs a device-token capture with the draft in the graph");
  if ((graph_feed_rows_ > 0 ? graph_feed_rows_ : decode_rows_) != 2)
    throw std::logic_error("session_graph_capture_next_tokens: the token feed is [next, draft] (T = 2)");
  if (draft_verdict == nullptr) throw std::invalid_argument("session_graph_capture_next_tokens: null verdict");
  glm_spec_next_tokens(d_next_ + req, draft_verdict, step_tokens_, stream_);
}

template <class D>
void SessionModel<D>::session_graph_capture_next_tokens_batch(const PickVerdict* draft_verdicts) {
  session_graph_capture_next_tokens_batch(std::vector<const PickVerdict*>{draft_verdicts});
}

template <class D>
void SessionModel<D>::session_graph_capture_next_tokens_batch(const std::vector<const PickVerdict*>& draft_verdicts) {
  if (!graph_device_tokens_ || !graph_has_draft_ || graph_batch_requests_ <= 0 || graph_rows_per_request_ < 2)
    throw std::logic_error("session_graph_capture_next_tokens_batch: requires the fixed draft graph");
  if (draft_verdicts.empty() || draft_verdicts.size() > static_cast<size_t>(kSpecMaxDrafts))
    throw std::invalid_argument("session_graph_capture_next_tokens_batch: draft count");
  // The feeds carry every draft of the block whatever the verify's rows
  // (the reduced-depth batch verified a prefix off the compacted copy).
  const int feed_rows = graph_feed_rows_ > 0 ? graph_feed_rows_ : graph_rows_per_request_;
  if (feed_rows != 1 + static_cast<int>(draft_verdicts.size()))
    throw std::logic_error(
        "session_graph_capture_next_tokens_batch: the token feed is [next, draft_1 .. draft_n] per request (T = 1 + n)");
  GlmSpecDrafts d;
  for (size_t i = 0; i < draft_verdicts.size(); ++i) {
    if (draft_verdicts[i] == nullptr)
      throw std::invalid_argument("session_graph_capture_next_tokens_batch: null verdicts");
    d.v[i] = draft_verdicts[i];
  }
  d.count = static_cast<int>(draft_verdicts.size());
  int64_t* feeds = (graph_feed_rows_ > 0 || graph_batch_map_source_)
                       ? d_tokens_ + static_cast<size_t>(max_decode_rows_)
                       : step_tokens_;
  glm_spec_next_tokens_batched(d_next_, d, graph_batch_requests_, feed_rows, feeds, stream_,
                               graph_batch_map_source_ ? d_batch_map_ : nullptr);
}

template <class D>
std::vector<std::vector<std::pair<int32_t, float>>> SessionModel<D>::topk(const std::vector<float>& logits,
                                                                          int64_t rows, int vocab, int k) {
  if (k <= 0 || k > 64 || k > vocab) throw std::invalid_argument("topk: k out of range");
  std::vector<std::vector<std::pair<int32_t, float>>> out(static_cast<size_t>(rows));
  for (int64_t r = 0; r < rows; ++r) {
    const float* row = logits.data() + static_cast<size_t>(r) * vocab;
    std::vector<std::pair<float, int32_t>> best;
    for (int c = 0; c < vocab; ++c) {
      const float v = row[c];
      bool placed = false;
      for (size_t i = 0; i < best.size(); ++i) {
        if (v > best[i].first) {
          if (best.size() < static_cast<size_t>(k)) best.emplace_back(0.f, 0);
          for (size_t j = best.size() - 1; j > i; --j) best[j] = best[j - 1];
          best[i] = {v, c};
          placed = true;
          break;
        }
      }
      if (!placed && best.size() < static_cast<size_t>(k)) best.emplace_back(v, c);
    }
    for (const auto& [v, id] : best) out[static_cast<size_t>(r)].emplace_back(id, v);
  }
  return out;
}

}  // namespace dgpp
