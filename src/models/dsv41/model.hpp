#pragma once
// DeepSeek-V4.1-Flash (DeepseekV41ForCausalLM) on the shared session core
// (2026-09-13, docs/deepseek_v41_flash_plan.md G5). The layer walk follows
// the reference `Block` with the single-pass mHC (§1.2):
//
//   x = embed(tok) on all four streams;  pre_mix = [1, 0, 0, 0]
//   per layer:  (Engram on layers 1 and 14: x_i += gate_i * value)
//               (pre_a, post_a, comb_a) = hc_mixes(x, attn);  u = norm(hc_pre(x, pre_mix))
//               x = hc_post(CSA2(u), x, post_a, comb_a)
//               (pre_f, post_f, comb_f) = hc_mixes(x, ffn);   u = norm(hc_pre(x, pre_a))
//               x = hc_post(MoE(u), x, post_f, comb_f);  pre_mix = pre_f
//   logits = head(norm(hc_pre(x, pre_mix)))
//
// One row walk (run_rows) serves every entry point; the engine/session_
// model.hpp core owns everything around it. Attention state per request
// slot: its row of the CSA2 pool's block table (the compressed main KV
// and index keys of the four kv sources, 128-token blocks), its window
// rings on every layer (positional, in the prefix snapshot), the
// compressor tails of the three ratio-2 sources and the Engram context
// (the two state families with per-row snapshots: the rollback table).
// The selection and the candidate pool flow between layers through the
// CSA2 layer's scratch (plan D6). The Engram tables stay mmap'ed on the
// NVMe: one host node per walk gathers both layers' rows (plan D5) —
// session_graph_host_nodes() declares it.
//
// TP: `tp_world` > 1 loads this rank's slices (64/W heads and 8/W output
// groups, every expert's intermediate slice, the Engram wkv's K columns
// of its hash heads, the lm-head vocab slice) and folds the attention's
// wo_b partial, the MoE partial and the Engram kv partial per site
// through `boundary`; the indexer, the compressor, the router, the mHC
// coefficients and the norms are replicated — every rank's streams are
// bitwise the others'.
//
// The DSpark draft (plan D8, §1.7; kSpecRows 6) rides the core's chain
// protocol: the verify walk stores the target layers' stream means
// [h_37 | h_38 | h_39] per row in the draft window (draft_width 3H); the
// first draft call of a step gathers the accepted rows', projects main_x,
// appends each draft ring (the rows' window latents from main_x), runs the
// three draft stages ONCE over the block [next, noise x 4] at the next
// five positions (the bidirectional block attention over the ring), the
// shared head over the five rows into base_logits_, and emits block row 0
// biased by the Markov head of `next`; every chain call (depth - 1 of
// them, one per further draft position) emits the next block row biased
// by the previous pick — no state moves, so the chain brackets and the
// draft snapshot are no-ops (the rings are positional: a rejected block
// row's slot lies past every later window). mtp_depth is the verified
// block length (1..5).
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "engine/boundary_reducer.hpp"
#include "engine/decode_outputs.hpp"
#include "engine/memory_plan.hpp"
#include "kernels/bf12_companions.hpp"
#include "engine/session_model.hpp"
#include "kernels/gemm.hpp"
#include "kernels/draft_head.hpp"
#include "kernels/l2_prefetch.hpp"
#include "kernels/glm_mhc_launch.hpp"
#include "kernels/glm_spec.hpp"
#include "models/dsv41/config.hpp"
#include "models/dsv41/csa2_layer.hpp"
#include "models/dsv41/csa2_state.hpp"
#include "models/dsv41/engram_layer.hpp"
#include "models/dsv41/engram_tables.hpp"
#include "models/dsv41/loader.hpp"
#include "models/glm/mhc.hpp"
#include "models/glm/moe_layer.hpp"

namespace dgpp {

class Dsv41Model : public SessionModel<Dsv41Model> {
 public:
  using Base = SessionModel<Dsv41Model>;
  bool prefill_batch_fits(const RowRun& run) const;
  int max_mixed_prefill_readers(int decode_rows) const {
    return prefill_bounded_ ? std::max(0, (max_tokens_ - decode_rows) / cfg_.sliding_window) : max_requests_;
  }

  using Outputs = Base::Outputs;
  using SessionSnapshotMeta = Base::SessionSnapshotMeta;
  using SnapshotRequest = Base::SnapshotRequest;
  using RowRun = Base::RowRun;

  Dsv41Model(const Dsv41TextConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
             int64_t max_cache_tokens, Dsv41Residency residency = Dsv41Residency::Streaming,
             BoundaryReducer* boundary = nullptr, int tp_rank = 0, int tp_world = 1,
             int max_requests = 1, bool mtp = false, int decode_rows = 0,
             bool serving_logits = false);
  ~Dsv41Model();
  Dsv41Model(const Dsv41Model&) = delete;
  Dsv41Model& operator=(const Dsv41Model&) = delete;

  using MemoryPlan = dgpp::MemoryPlan;
  static MemoryPlan plan_memory(const Dsv41TextConfig& cfg, int max_tokens,
                                int64_t max_cache_tokens, int tp_rank = 0, int tp_world = 1,
                                Dsv41Residency residency = Dsv41Residency::Streaming,
                                int max_requests = 1, bool mtp = false, int decode_rows = 0,
                                bool serving_logits = false);

  // The prefill mode (plan §1.8, D8): `exact` runs the forty layers over
  // every prompt row (the parity mode); `bounded` (the production default)
  // runs the twenty encoder layers over the prompt, publishes the
  // decoder's global KV (the last kv source's compressor and index keys)
  // for every row, and runs the decoder over the LAST `sliding_window`
  // rows only — the replay segment — with the window floored at the
  // segment's start. Across the chunks of one prefill call the encoder
  // output of the last window rows carries over (the tail); a resumed
  // prefill replays from its resume point.
  void set_prefill_bounded(bool on) { prefill_bounded_ = on; }
  bool prefill_bounded() const { return prefill_bounded_; }
  // The diagnostic forward walks the first `n` layers only (n <= 0: all;
  // the head is skipped when limited — the cross-check's per-layer dump,
  // dsv41_forward_check --layers).
  void set_debug_layer_limit(int n) { debug_layer_limit_ = n; }
  // The mode new models start in (the server's engine.prefill; the
  // constructor's default is exact — the gates' parity mode).
  static void set_default_prefill_bounded(bool on) { s_default_prefill_bounded = on; }
  static bool default_prefill_bounded() { return s_default_prefill_bounded; }
  // The prefill walk's dense projections (engine.prefill_dense): the
  // fp8-weight tile GEMM (true, the default) or the streaming decode form
  // for every projection a prefill row runs — the CSA2 layer's, the Engram
  // projection's and the draft's main projection's. Decode rows always run
  // the streaming form. Every model built from here on takes the default.
  static void set_default_prefill_dense_tile(bool on) { s_default_prefill_dense_tile = on; }
  void set_prefill_dense_tile(bool on) {
    prefill_dense_tile_ = on;
    if (csa2_) csa2_->set_prefill_dense_tile(on);
  }
  // The prefill walk's window attention (engine.prefill_window): the flash
  // listed kernel (true, the default) or the scalar partial kernel the
  // decode window runs. Every model built from here on takes the default.
  static void set_default_prefill_window_flash(bool on) { s_default_prefill_window_flash = on; }
  void set_prefill_window_flash(bool on) {
    prefill_window_flash_ = on;
    if (csa2_) csa2_->set_prefill_window_flash(on);
  }
  // The decode window attention (engine.decode_window, 2026-10-19): the
  // flash listed kernel unsplit on a side stream beside the main source
  // (true, the default) or the scalar partial kernel split eight ways.
  static void set_default_decode_window_flash(bool on) { s_default_decode_window_flash = on; }
  void set_decode_window_flash(bool on) {
    decode_window_flash_ = on;
    if (csa2_) csa2_->set_decode_window_flash(on);
  }
  // The fused mHC decode site (mhc_site_fused) on or off: the tests'
  // bisection handle; the unfused form is the prefill recipe's.
  void set_decode_mhc_fused(bool on) { decode_mhc_fused_ = on; }
  void set_decode_window_beside(bool on) { decode_window_beside_ = on; }  // the tests' bisection handle
  // The prefill's fold overlap (engine.prefill_fold_overlap; the
  // DeepSeek-V4 walk's form, here since 2026-10-18): a walk of min_rows
  // rows or more on a multi-rank world runs each layer as two row blocks,
  // the boundary a row of the request's block grid nearest the middle, so
  // every boundary fold lands under the other block's compute — the
  // attention per block, the mHC site and the router per block, the expert
  // chain over every row, the accumulation per block, and block B's MoE
  // fold under the next layer's block A. Bitwise the one-block walk
  // (dsv41_tp_test's gates). Tests lower min_rows and may run the row-block
  // form without a reducer. Every model built from here on takes the default.
  static constexpr int kFoldOverlapMinRows = 512;
  static void set_default_prefill_fold_overlap(bool on) { s_default_prefill_fold_overlap = on; }
  // The resident layout the next models build their stream with (the step
  // engine's arena, loaders/resident_stream.hpp ResidentLayout; MOONSHOT-1.md
  // 2.4): the engine selects it by engine.decode_engine before the model is
  // built. An empty layout is the loader's grant order, today's behaviour.
  static void set_default_resident_layout(ResidentLayout layout) { s_default_resident_layout = std::move(layout); }
  static const ResidentLayout& default_resident_layout() { return s_default_resident_layout; }
  void set_prefill_fold_overlap(bool on, int min_rows = kFoldOverlapMinRows, bool without_reducer = false);
  bool prefill_fold_overlap() const { return fold_overlap_; }
  uint64_t debug_overlap_async_folds() const { return overlap_async_folds_; }
  // The lm head's 12-bit companion (engine.bf16_weights): tests read its counters.
  const Bf12Companions& bf12_companions() const { return bf12_; }
  // The Engram chunk-ahead staging's counters (tests): prestages published and claimed.
  uint64_t debug_engram_prestage_publishes() const { return engram_ ? engram_->debug_prestage_publishes() : 0; }
  uint64_t debug_engram_prestage_claims() const { return engram_ ? engram_->debug_prestage_claims() : 0; }

  // The cold diagnostic forward: one request on slot 0, fresh state, every
  // row's logits; capture_layers: every layer's residual streams [T, 4, H]
  // (bounded: the decoder layers' captures hold the segment rows only).
  // `teacher` (needs capture_layers): per layer the streams [T, 4, H] to
  // start the NEXT layer from (layer l > 0 reads teacher[l - 1]; with
  // num_layers entries the head reads teacher[L - 1]), `teacher_pre` the
  // matching collapse coefficients [T, 4] fp32 (debug_layer_pre() of the
  // walk that produced the streams) — the layer-local comparison of two
  // walks (dsv41_tp_test's world-1 twin of a TP world).
  Outputs forward(const std::vector<int64_t>& token_ids, bool capture_layers = false,
                  const std::vector<std::vector<uint16_t>>* teacher = nullptr,
                  const std::vector<std::vector<float>>* teacher_pre = nullptr);

  static constexpr int prefill_chunk_tokens() { return kPrefillChunkTokens; }
  static constexpr int kv_block_tokens_static() { return kBlockTokens; }
  // 64 since 2026-10-23 (32 from 2026-09-14, 16 before): twelve request
  // slots at DSpark depth 4 (60 rows) in one batched replay — the
  // aggregate past six streams, where the step's fixed costs (collectives,
  // the small kernels) amortize over more rows; kDecodeRowsMax bounds it.
  static constexpr int decode_rows_cap() { return 96; }
  // The group prefill (session_prefill_group): a span's longest prompt —
  // each span fits the walk. Bounded prefill packs the spans' final
  // windows into decoder space before projecting their last rows.
  int64_t prefill_group_span_limit() const { return max_tokens_; }
  static size_t session_snapshot_bytes(const Dsv41TextConfig& cfg, int tp_world, bool mtp);
  using Base::session_snapshot_bytes;
  static Csa2Config csa2_config(const Dsv41TextConfig& cfg, int tp_world);
  static Csa2PoolShape pool_shape(const Dsv41TextConfig& cfg, int max_requests, int64_t cache_tokens);
  // One cache block's bytes across every pool plane (the NVMe cold tier's
  // record, issue #26) for a shape that is not built yet.
  static size_t kv_block_bytes_static(const Dsv41TextConfig& cfg);

  const Dsv41TextConfig& config() const { return cfg_; }
  const Csa2StatePool& csa2_pool() const { return pool_; }

  // ---- the session core's hooks --------------------------------------------------
  Outputs run_rows(const RowRun& run);
  void reset_slot_state(int req);
  GlmSpecSegments spec_segments(int req, int snapshot_row0 = 0) const;
  size_t snapshot_state_bytes() const;
  size_t draft_state_bytes() const { return 0; }
  void write_state_snapshot(int req, uint8_t* dst, int spec_row);
  void write_draft_snapshot(int, uint8_t*, bool, int64_t) {}
  void read_state_snapshot(int req, const uint8_t* src);
  void read_draft_snapshot(int, const uint8_t*) {}
  bool has_pool() const { return true; }
  Csa2StatePool& pool() { return pool_; }
  const Csa2StatePool& pool() const { return pool_; }
  void graph_prepare();
  void mtp_run_rows(int req, const int64_t* tokens, int64_t first_pos, int T, bool decode_row, bool capture,
                    int head_rows, int batch_requests);
  void snapshot_draft_state(int) {}
  void restore_draft_state(int) { draft_row_ = 0; }
  static constexpr bool kDraftChain = true;
  static constexpr bool kBatchedDraftChain = true;
  // The DSpark confidence head (engine/verify_schedule.hpp): the block's
  // per-position acceptance logits, [max_requests][block] on the device,
  // written by the step's draft (row 0 by the block, row c by chain row c).
  static constexpr bool kVerifyConfidence = true;
  int confidence_rows() const { return cfg_.dspark_block_size; }
  const float* device_confidence() const { return conf_; }
  const uint16_t* draft_hidden_rows() const { return main_gather_; }
  void snapshot_chain_state(int) {}
  void restore_chain_state(int) {}
  // The DSpark confidence logits of the last block per request slot
  // [max_requests, block] (device fp32; row k for draft position k + 1).
  const float* debug_confidence() const { return conf_; }
  // The block's base logits (the shared head over the five rows, before
  // the Markov bias) of the last first-draft call: [groups * block, vocab slice].
  const float* debug_base_logits() const { return base_logits_; }
  size_t session_graph_host_nodes() const { return engram_ ? 1 : 0; }
  // Test probes (capture_layers walks): per layer, the attention site's
  // normed input, the attention output (folded) and the streams after
  // the attention update.
  struct SiteCapture {
    std::vector<uint16_t> x_attn, attn_out, streams_after_attn, x_ffn, ffn_out;
  };
  const std::vector<SiteCapture>& debug_sites() const { return debug_sites_; }
  // The index sources' logits of a capture_layers prefill walk: per source,
  // [rows, stride] fp32 (the visible entries per row, -inf past them).
  struct IndexLogits {
    int rows = 0;
    int64_t stride = 0, entries = 0;
    std::vector<float> values;
    std::vector<int32_t> cand;        // [rows, candidate_blocks] the pool (a candidate source or user), else empty
    std::vector<int32_t> cand_counts;
    // The coded index query of every walked row (decode walks too): e4m3
    // codes [walk_rows, heads, 128] and the row scales [walk_rows, heads].
    // Two paths' selections can differ only through these (the logits are
    // their dot products with the cached keys), and the coding is a
    // discontinuity: a query element near a code boundary moves by a code
    // step under sub-ulp noise. A flip whose codes differ between the
    // paths is the coding's, whatever the logit gap.
    int q_rows = 0, q_heads = 0;
    std::vector<uint8_t> q_codes;
    std::vector<float> q_scales;
  };
  const std::vector<IndexLogits>& debug_index_logits() const { return debug_index_logits_; }
  // Per layer of a capture_layers walk: the collapse coefficients [T, 4]
  // fp32 the next site reads (the layer's ffn-site pre).
  const std::vector<std::vector<float>>& debug_layer_pre() const { return debug_layer_pre_; }

 private:
  static constexpr int kBlockTokens = 128;
  // Amortize prefill launches while bounding scratch growth on four GB10s.
  static constexpr int kPrefillChunkTokens = 4096;
  static constexpr int kDecodeSplit = 32;
  static constexpr size_t kDotBudget = 64ull << 20;
  static constexpr int kRingSlots = 160;

  struct WalkRows {
    int decode_spans = 0;
    const int64_t* tokens = nullptr;
    const int32_t* req_ids = nullptr;
    const int64_t* pos = nullptr;
    const int32_t* spans = nullptr;
    int num_requests = 1;
    int req = 0;
    int64_t pos0 = 0;
    bool decode = false;
    bool capture = false;
    bool snapshots = false;
    int moe_table_slot = -1;
    MoeTraceStaging* trace = nullptr;
    int64_t window_floor = 0;  // prefill: the earliest position the window attends
    bool publish = true;       // prefill: a kv source publishes its entries (false: published before)
    // The group prefill's spans (host arrays; num_spans 0: one request):
    // span s = request span_reqs[s], positions from span_pos0[s], rows
    // [span_row0[s], span_row0[s] + span_lens[s]) of the walk.
    const int32_t* span_reqs = nullptr;
    const int64_t* span_pos0 = nullptr;
    const int32_t* span_lens = nullptr;
    const int64_t* span_floor = nullptr;  // per span: the earliest position its window attends
    int num_spans = 0;
  };
  void build_layer_objects(const Dsv41LayerResident& r);
  Csa2LayerWeights csa2_view(const Dsv41LayerResident& r, int layer) const;
  static GlmMoeWeights moe_view(const Dsv41MoeResident& m);
  Dsv41EngramLayerWeights engram_view(const Dsv41EngramResident& e) const;
  int tail_ordinal(int layer) const;
  int cache_ordinal(int layer) const;
  // The layer over the streams (cur -> next, swapped), the two folds.
  void enqueue_layer(const Dsv41LayerResident& r, int layer, int T, const WalkRows& rows);
  void mhc_site(const uint16_t* streams, const GlmMhcWeights& w, const uint16_t* ln, int T, bool decode,
                int decode_prefix = 0);
  void stream_update(const uint16_t* sublayer_out, int T);
  // The fused decode site (2026-10-19, DeepSeek-V4's form): the finish in
  // the dots launch (mhc_counters_'s tickets), the one-rounding norm in
  // it (x_ written; collapsed_ not stored), the comb — read by the stream
  // update alone — deferred to mhc_side_ and joined by join_comb.
  void mhc_site_fused(const uint16_t* streams, const GlmMhcWeights& w, const uint16_t* ln, size_t r, int count,
                      MhcSinglePass sp);
  void join_comb();
  uint16_t* stage(uint16_t* fallback, int T, int width, bool capture);
  void fold(uint16_t* buf, int T, int width, bool capture);
  // The fold overlap (set_prefill_fold_overlap): the layer as the row
  // blocks [0, TA) and [TA, T). On entry the previous layer's block-B MoE
  // fold may be outstanding (ffn_b_pending_: A's streams updated in nxt_,
  // B's in cur_ awaiting its update); `flush` lands it before returning.
  void enqueue_layer_overlap(const Dsv41LayerResident& r, int layer, int T, int TA, const WalkRows& rows, bool flush);
  void mhc_site_rows(const uint16_t* streams, const GlmMhcWeights& w, const uint16_t* ln, int r0, int rows,
                     int decode_prefix);
  void update_rows(const uint16_t* sublayer_out, int r0, int rows);
  void swap_pre();
  bool begin_fold_after(uint16_t* buf, int r0, int n, cudaEvent_t ready, int width = 0);  // width 0: hidden
  void begin_pending_fold();
  void flush_overlap();
  // The block boundary of a walk of T rows: the row nearest T/2 at which
  // the attention may start a chunk (a span's start, or a row on its
  // request's block grid inside a publishing span), at least a block from
  // either end, past any decode prefix. 0: none (the one-block walk).
  int overlap_boundary(int T, const WalkRows& rows) const;
  // The lm head's bf12 companion (kernels/bf12_companions.hpp): packed after
  // the load under engine.bf16_weights bf12 | bf12+bf16 — the decode head
  // GEMV (the draft's base logits and the verify's, 331 MB per rank per
  // read on the 1017 checkpoint at world 4) reads 12 bits per weight; the
  // prefill head keeps its tile path, expanding from the companion when
  // the bf16 bytes went back (one slot: the prefill head runs once a walk).
  static constexpr int kBf12ExpandSlots = 1;
  Bf12Companions bf12_;
  static inline bool s_default_prefill_fold_overlap = true;
  static inline ResidentLayout s_default_resident_layout{};
  bool fold_overlap_ = s_default_prefill_fold_overlap;
  int fold_overlap_min_rows_ = kFoldOverlapMinRows;
  bool fold_overlap_without_reducer_ = false;
  // The Engram chunk-ahead staging due this walk (run_rows names it; the
  // last Engram layer issues it after its embed consumed the claimed rows).
  struct PrestageDue {
    const int64_t* ids = nullptr;
    int rows = 0;
    int64_t pos0 = 0;
    int req = -1;
    const int64_t* prefix = nullptr;
    int n_prefix = 0;
  } prestage_due_;
  void prestage_after_embed(int table_index);
  bool ffn_b_pending_ = false;        // block B's MoE fold is outstanding (its rows accumulated)
  bool ffn_b_begun_ = false;          // ... and submitted to the bus (else only its event stands)
  cudaEvent_t ov_ev_[6] = {};         // the blocks' producers: attention A, B; accumulation A, B; Engram projection A, B
  int overlap_rows_ = 0;              // the pending fold's walk rows and block boundary
  int overlap_ta_ = 0;
  uint64_t overlap_async_folds_ = 0;  // folds begun asynchronously (tests)
  void gather_embedding(const int64_t* tokens, int T, bool capture);
  // The DSpark pieces (mtp): the first draft call of a step (the rings, the
  // block, base_logits_, row 0) and a chain row (block row draft_row_).
  void draft_first(int req, const int64_t* tokens, const int64_t* d_pos, const int32_t* d_req, int T, bool decode_row,
                   bool capture, int head_rows, int batch_requests, int source_row = 0);
  void draft_chain_row(int req, const int64_t* tokens, bool capture, int head_rows, int batch_requests);
  const Dsv41DraftResident& draft_stage(int stage);
  // The decode walk's boundary prefetch windows (kernels/l2_prefetch.hpp;
  // bit-identical on or off; resident stacks only): before the attention
  // fold the MoE site's replicated bytes and the shared expert, before the
  // MoE fold the next layer's attention side (or the head's), each under
  // the collective and the mHC site behind it; an Engram layer's own fold
  // carries its attention side (the Engram projection runs first). The
  // CSA2 layer opens its in-layer windows (Csa2Layer::set_prefetcher).
  bool prefetching(bool decode) const {
    return decode && prefetch_.enabled() && loader_.residency() == Dsv41Residency::Resident;
  }
  void prefetch_fp8(const GlmQuantMatrix& m);
  void prefetch_exl3(const Exl3Matrix& m);
  void prefetch_bf16(const uint16_t* w, size_t bytes, int rows);
  // The lm head's logits (fp32 [rows, lm_vocab_count]) for `rows` hidden
  // rows: the bf16 / bf12 head through the GEMM, or the EXL3 head's launcher.
  void head_logits(const uint16_t* h, int rows, float* out, cudaStream_t stream);
  void prefetch_mhc(const uint16_t* fn, const uint16_t* norm, int rows);
  void prefetch_attention_weights(const Dsv41LayerResident& r, int rows);
  void prefetch_ffn_side(const Dsv41LayerResident& r, int rows);
  void prefetch_attention_side(int layer, int rows);
  void prefetch_head(const uint16_t* norm, bool draft, int rows);
  WeightPrefetcher prefetch_;
  int target_ordinal(int layer) const;
  int32_t* ctx(int req) const { return d_ctx_ + static_cast<size_t>(req) * 4; }
  float* spec_tails(int tail_ord, int row) const {
    return spec_tails_ + (static_cast<size_t>(tail_ord) * max_decode_rows_ + row) * 2 * kCsa2Latent;
  }

  Dsv41TextConfig cfg_;
  Dsv41LayerStream loader_;
  CublasLtGemm gemm_;
  // The draft head's screen (kernels/draft_head.hpp): built from the served
  // head at construction under engine.draft_head; answers a decode draft's
  // head call with the masked row or declines (the exact head runs).
  DraftHead draft_head_;
  HeadMatrix draft_head_matrix() const;
  bool dense_mma_ = true;  // the dense projections' and head's decode form (DGPP_DSV41_DENSE_GEMV=1: the GEMV chunks)
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;
  void* exl3_ws_ = nullptr;  // the EXL3 GEMV's own (head, DSpark main projection); zeroed at allocation
  size_t exl3_ws_bytes_ = 0;
  Dsv41GlobalsResident globals_;
  GlmMoeConfig moe_cfg_;
  GlmMhcConfig mhc_cfg_;
  Csa2Config csa2_cfg_;
  bool embed_sharded_ = false;
  int world_ = 1;

  std::unique_ptr<Csa2Layer> csa2_;
  std::unique_ptr<GlmMoeLayer> moe_;
  std::unique_ptr<GlmMoeLayer> draft_moe_;  // the draft stages' routed chain (mtp)
  GlmMoeConfig draft_moe_cfg_;
  std::unique_ptr<Dsv41EngramLayer> engram_;
  Dsv41EngramSidecar sidecar_;
  Csa2StatePool pool_;
  void* csa2_scratch_ = nullptr;
  size_t csa2_scratch_bytes_ = 0;
  float* inv_freq_window_ = nullptr;      // device [32]
  float* inv_freq_compressed_ = nullptr;  // device [32]
  std::vector<int> cache_ord_;            // per layer (-1: window only)
  std::vector<int> tail_ord_;             // per layer (-1: none)
  int tails_ = 0;

  // Activations [M rows].
  uint16_t* streams_a_ = nullptr;   // [M, 4, H]
  uint16_t* streams_b_ = nullptr;
  uint16_t* cur_ = nullptr;         // the live streams (one of the two)
  uint16_t* nxt_ = nullptr;
  uint16_t* x_ = nullptr;           // [M, H] the normed sublayer input
  uint16_t* y_ = nullptr;           // [M, H] the block output (the fold's fallback)
  uint16_t* collapsed_ = nullptr;   // [M, H]
  uint16_t* post_bf16_ = nullptr;   // [M, 4]
  uint16_t* comb_bf16_ = nullptr;   // [M, 16]
  float* post_f32_ = nullptr;       // [M, 4]
  float* comb_f32_ = nullptr;       // [M, 16]
  float* pre_a_ = nullptr;          // [M, 4] the collapse coefficients in use
  float* pre_b_ = nullptr;          // [M, 4] the next site's
  float* pre_cur_ = nullptr;
  float* pre_nxt_ = nullptr;
  float* one_hot_ = nullptr;        // [M, 4] = (1, 0, 0, 0): layer 0's collapse
  float* mhc_logits_ = nullptr;     // [M, 24]
  int* mhc_counters_ = nullptr;     // [M] the fused finish's tickets, zero at rest
  cudaStream_t mhc_side_ = nullptr;
  cudaEvent_t mhc_fork_ = nullptr;
  cudaEvent_t mhc_join_ = nullptr;
  bool comb_pending_ = false;       // a deferred comb the next stream update joins
  cudaStream_t attn_side_ = nullptr;  // the decode window beside the main source (Csa2Layer::set_decode_side)
  cudaEvent_t attn_fork_ = nullptr;
  cudaEvent_t attn_join_ = nullptr;
  uint16_t* engram_kv_ = nullptr;   // [M, 5H] the fold's fallback
  int32_t* d_ctx_ = nullptr;        // [R, 4] the Engram contexts
  int32_t* spec_ctx_ = nullptr;     // [max(M, rows), 4] per-row contexts
  float* spec_tails_ = nullptr;     // [tails][rows][2][512]
  // DSpark (mtp).
  static inline bool s_default_prefill_bounded = false;
  bool prefill_bounded_ = s_default_prefill_bounded;
  static inline bool s_default_prefill_dense_tile = true;
  bool prefill_dense_tile_ = s_default_prefill_dense_tile;
  static inline bool s_default_prefill_window_flash = true;
  bool prefill_window_flash_ = s_default_prefill_window_flash;
  static inline bool s_default_decode_window_flash = true;
  bool decode_window_flash_ = s_default_decode_window_flash;
  bool decode_mhc_fused_ = true;
  bool decode_window_beside_ = true;
  int debug_layer_limit_ = 0;
  // The bounded prefill's segment of the last walk (the draft's prefill
  // rows map into main_hidden_ through it) and the tail carried between a
  // call's chunks: the encoder output of the last window rows.
  int64_t seg_pos0_ = 0;
  int seg_rows_ = 0;
  // Per request: the last walk's draft segment (pos0, rows) — a group
  // prefill sets every span's; mtp_run_rows clamps its prefill rows to it.
  struct WalkSegment { int64_t pos0 = 0; int rows = 0; int row0 = 0; };
  std::vector<WalkSegment> seg_by_req_;
  WalkSegment& segment_of(int req) {
    if (seg_by_req_.size() <= static_cast<size_t>(req)) seg_by_req_.resize(static_cast<size_t>(req) + 1);
    return seg_by_req_[static_cast<size_t>(req)];
  }
  uint16_t* tail_streams_ = nullptr;  // [requests, window, 4H]
  float* tail_pre_ = nullptr;         // [requests, window, 4]
  uint16_t* tail_move_ = nullptr;   // [window, 4H], shared temporary for overlapping tail moves
  float* seg_pre_ = nullptr;          // [window, 4] the segment's collapse coefficients
  struct PrefillTail { int rows = 0; int64_t end = 0, start = 0; };
  std::vector<PrefillTail> prefill_tails_;  // the prefill call's first row (the segment floors unless it starts there)
  int targets_ = 0;                 // dspark target layers (the draft width is targets_ * H)
  int draft_row_ = 0;               // the next block row a chain call emits (0: no block stands)
  uint16_t* main_hidden_ = nullptr; // [M, targets * H] the target layers' stream means of the walk's rows
  uint16_t* main_gather_ = nullptr; // [rows, targets * H] the draft rows' gathered main hidden
  uint16_t* main_x_ = nullptr;      // [max(M, rows), H] main_norm(main_proj(main hidden))
  int64_t* blk_pos_ = nullptr;      // [rows] the block rows' positions
  int64_t* blk_tok_ = nullptr;      // [rows] the block rows' tokens
  int32_t* blk_req_ = nullptr;      // [rows]
  int32_t* blk_spans_ = nullptr;    // [rows + 1]
  float* base_logits_ = nullptr;    // [rows, vocab slice] the block's head rows
  float* conf_ = nullptr;           // [R, block] the confidence logits
  int64_t* d_draft_pos_ = nullptr;  // [M] the prefill draft rows' positions (staged per call)
  int32_t* d_draft_req_ = nullptr;  // [M]
  std::vector<SiteCapture> debug_sites_;
  std::vector<IndexLogits> debug_index_logits_;
  bool debug_capture_ = false;
  std::vector<std::vector<float>> debug_layer_pre_;
  const std::vector<std::vector<uint16_t>>* debug_teacher_ = nullptr;
  const std::vector<std::vector<float>>* debug_teacher_pre_ = nullptr;
  // The prefill walk's route traces.
  int32_t* h_route_ids_ = nullptr;
  float* h_route_weights_ = nullptr;
};

}  // namespace dgpp
