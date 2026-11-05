#pragma once
// DeepSeek-V4-Flash (DeepseekV4ForCausalLM, the 0731 release) on the shared
// session core (2026-10-01). The layer walk follows the reference `Block`
// with the two-pass mHC:
//
//   x = embed(tok) on all four streams
//   per layer:  (pre_a, post_a, comb_a) = hc_mixes(x, attn);  u = norm(hc_pre(x, pre_a))
//               x = hc_post(Attention(u), x, post_a, comb_a)
//               (pre_f, post_f, comb_f) = hc_mixes(x, ffn);   u = norm(hc_pre(x, pre_f))
//               x = hc_post(MoE(u), x, post_f, comb_f)
//   logits = head(norm(hc_head(x)))
//
// One row walk (run_rows) serves every entry point; the engine/session_
// model.hpp core owns everything around it. Attention state per request
// slot: its row of the pool's block table (every compressing layer's own
// compressed KV and, on the ratio-4 layers, index keys; 128-token blocks),
// its window rings on every layer and its compressors' rings. Every ring is
// positional, so the rollback table is empty and a prefix snapshot is the
// rings as they stand.
//
// TP: `tp_world` > 1 loads this rank's slices (64/W heads and 8/W output
// groups, every expert's intermediate slice, the lm-head vocab slice) and
// folds the attention's wo_b partial and the MoE partial per site through
// `boundary`; the indexers, the compressors, the routers, the mHC
// coefficients and the norms are replicated — every rank's streams are
// bitwise the others'.
//
// The DSpark draft (kSpecRows 6) rides the core's chain protocol exactly as
// DeepSeek-V4.1's does (models/dsv41/model.hpp): the verify walk stores the
// target layers' OUTPUT stream means [h_40 | h_41 | h_42] per row in the
// draft window; the first draft call of a step projects main_x, appends
// each draft ring, runs the draft stages once over the block [next, noise x
// 4], the shared head over the five rows, and emits block row 0 biased by
// the Markov head of `next`; every chain call emits the next block row.
// mtp_depth is the verified block length (1..5).
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
#include "engine/session_model.hpp"
#include "kernels/gemm.hpp"
#include "kernels/bf12_companions.hpp"
#include "kernels/glm_mhc_launch.hpp"
#include "kernels/glm_spec.hpp"
#include "kernels/l2_prefetch.hpp"
#include "models/dsv4/attn_layer.hpp"
#include "models/dsv4/config.hpp"
#include "models/dsv4/loader.hpp"
#include "models/dsv4/state.hpp"
#include "models/glm/mhc.hpp"
#include "models/glm/moe_layer.hpp"

namespace dgpp {

class Dsv4Model : public SessionModel<Dsv4Model> {
 public:
  using Base = SessionModel<Dsv4Model>;
  using Outputs = Base::Outputs;
  using SessionSnapshotMeta = Base::SessionSnapshotMeta;
  using SnapshotRequest = Base::SnapshotRequest;
  using RowRun = Base::RowRun;

  Dsv4Model(const Dsv4Config& cfg, const std::string& checkpoint_dir, int max_tokens, int64_t max_cache_tokens,
            Dsv4Residency residency = Dsv4Residency::Streaming, BoundaryReducer* boundary = nullptr, int tp_rank = 0,
            int tp_world = 1, int max_requests = 1, bool mtp = false, int decode_rows = 0,
            bool serving_logits = false);
  ~Dsv4Model();
  Dsv4Model(const Dsv4Model&) = delete;
  Dsv4Model& operator=(const Dsv4Model&) = delete;

  using MemoryPlan = dgpp::MemoryPlan;
  static MemoryPlan plan_memory(const Dsv4Config& cfg, int max_tokens, int64_t max_cache_tokens, int tp_rank = 0,
                                int tp_world = 1, Dsv4Residency residency = Dsv4Residency::Streaming,
                                int max_requests = 1, bool mtp = false, int decode_rows = 0,
                                bool serving_logits = false);

  // The diagnostic forward walks the first `n` layers only (n <= 0: all;
  // the head is skipped when limited — the cross-check's per-layer dump).
  void set_debug_layer_limit(int n) { debug_layer_limit_ = n; }

  // The cold diagnostic forward: one request on slot 0, fresh state, every
  // row's logits; capture_layers: every layer's residual streams [T, 4, H].
  // `teacher` (needs capture_layers): per layer the streams [T, 4, H] to
  // start the NEXT layer from (layer l > 0 reads teacher[l - 1]; with
  // num_layers entries the head reads teacher[L - 1]) — the layer-local
  // comparison of two walks.
  Outputs forward(const std::vector<int64_t>& token_ids, bool capture_layers = false,
                  const std::vector<std::vector<uint16_t>>* teacher = nullptr);

  static constexpr int prefill_chunk_tokens() { return kPrefillChunkTokens; }
  static constexpr int kv_block_tokens_static() { return kBlockTokens; }
  static constexpr int decode_rows_cap() { return 32; }
  // The group prefill (session_prefill_group): a span's longest prompt —
  // each span fits the walk.
  int64_t prefill_group_span_limit() const { return max_tokens_; }
  // The prefill yields between chunks (every request-owned piece of state
  // — the pool, the rings, the open compressor groups, the draft rings —
  // stands at any cut on the block grid, and an unfinished slot is inert
  // to a decode replay), and several unfinished prompts' next chunks ride
  // one walk as its spans.
  static constexpr bool kPrefillChunkGrid = false;
  static size_t session_snapshot_bytes(const Dsv4Config& cfg, int tp_world, bool mtp);
  using Base::session_snapshot_bytes;
  static Dsv4AttnConfig attn_config(const Dsv4Config& cfg, int tp_world);
  static Dsv4PoolShape pool_shape(const Dsv4Config& cfg, int max_requests, int64_t cache_tokens);
  // One cache block's bytes across every pool plane (the NVMe cold tier's
  // record, issue #26) for a shape that is not built yet.
  static size_t kv_block_bytes_static(const Dsv4Config& cfg);

  const Dsv4Config& config() const { return cfg_; }
  const Dsv4StatePool& state_pool() const { return pool_; }

  // ---- the session core's hooks --------------------------------------------------
  Outputs run_rows(const RowRun& run);
  void reset_slot_state(int req);
  GlmSpecSegments spec_segments(int, int = 0) const { return GlmSpecSegments{}; }
  size_t snapshot_state_bytes() const;
  size_t draft_state_bytes() const { return 0; }
  void write_state_snapshot(int req, uint8_t* dst, int spec_row);
  void write_draft_snapshot(int, uint8_t*, bool, int64_t) {}
  void read_state_snapshot(int req, const uint8_t* src);
  void read_draft_snapshot(int, const uint8_t*) {}
  bool has_pool() const { return true; }
  Dsv4StatePool& pool() { return pool_; }
  const Dsv4StatePool& pool() const { return pool_; }
  void graph_prepare();
  void mtp_run_rows(int req, const int64_t* tokens, int64_t first_pos, int T, bool decode_row, bool capture,
                    int head_rows, int batch_requests);
  void snapshot_draft_state(int) {}
  void restore_draft_state(int) { draft_row_ = 0; }
  static constexpr bool kDraftChain = true;
  static constexpr bool kBatchedDraftChain = true;
  // The DSpark confidence head (engine/verify_schedule.hpp): the block's
  // per-position acceptance logits, [max_requests][block] on the device.
  static constexpr bool kVerifyConfidence = true;
  int confidence_rows() const { return cfg_.dspark_block_size; }
  const float* device_confidence() const { return conf_; }
  const uint16_t* draft_hidden_rows() const { return main_gather_; }
  void snapshot_chain_state(int) {}
  void restore_chain_state(int) {}
  const float* debug_confidence() const { return conf_; }
  const float* debug_base_logits() const { return base_logits_; }
  // The draft head back on the bf16 rows alone (tests: the screened head's
  // picks against it). Before a draft.
  void debug_drop_draft_screen();
  bool debug_draft_screened() const { return draft_head_fp8_ != nullptr; }
  size_t session_graph_host_nodes() const { return 0; }
  // Test probes (capture_layers walks): per layer, the attention site's
  // normed input, the attention output (folded), the streams after the
  // attention update, the MoE site's input and output.
  struct SiteCapture {
    std::vector<uint16_t> x_attn, attn_out, streams_after_attn, x_ffn, ffn_out;
    std::vector<float> post_attn;
  };
  const std::vector<SiteCapture>& debug_sites() const { return debug_sites_; }
  // The indexed layers' selections of a capture_layers walk ride
  // Outputs::dsa_selections (one entry per ratio-4 layer, [T, index_topk]);
  // the prefill select's logits of the walk's LAST tile per indexed layer:
  struct IndexLogits {
    int rows = 0;
    int64_t stride = 0, entries = 0;
    std::vector<float> values;
    int q_rows = 0, q_heads = 0;
    std::vector<uint8_t> q_codes;  // [walk rows, heads, 128] e4m3
    std::vector<float> q_scales;   // [walk rows, heads]
  };
  const std::vector<IndexLogits>& debug_index_logits() const { return debug_index_logits_; }

 private:
  static constexpr int kBlockTokens = 128;
  // Amortize prefill launches while bounding scratch growth on four GB10s.
  static constexpr int kPrefillChunkTokens = 4096;
  static constexpr int kDecodeSplit = 32;
  static constexpr size_t kDotBudget = 256ull << 20;
  static constexpr int kRingSlots = 160;

  struct WalkRows {
    int decode_spans = 0;
    const int64_t* tokens = nullptr;
    const int32_t* req_ids = nullptr;
    const int64_t* pos = nullptr;
    int num_requests = 1;
    int req = 0;
    int64_t pos0 = 0;
    bool decode = false;
    bool capture = false;
    int moe_table_slot = -1;
    MoeTraceStaging* trace = nullptr;
    // The group prefill's spans (host arrays; num_spans 0: one request):
    // span s = request span_reqs[s], positions from span_pos0[s], rows
    // [span_row0, span_row0 + span_lens[s]) of the walk.
    const int32_t* span_reqs = nullptr;
    const int64_t* span_pos0 = nullptr;
    const int32_t* span_lens = nullptr;
    int num_spans = 0;
  };
  void build_layer_objects(const Dsv4LayerResident& r);
  Dsv4AttnWeights attn_view(const Dsv4LayerResident& r, int layer) const;
  GlmMoeWeights moe_view(const Dsv4MoeResident& m) const;
  // The layer over the streams (cur -> next, swapped), the two folds.
  void enqueue_layer(const Dsv4LayerResident& r, int layer, int T, const WalkRows& rows);
  void mhc_site(const uint16_t* streams, const GlmMhcWeights& w, const uint16_t* ln, int T, bool decode,
                int decode_prefix = 0);
  // The head's collapse of the live streams into `out` [T, H] (pre-norm).
  void hc_head(const Dsv4HcHeadResident& w, uint16_t* out, int T, bool decode, int decode_prefix = 0);
  void stream_update(const uint16_t* sublayer_out, int T);
  // ---- the prefill's fold overlap (2026-10-02) ---------------------------------
  // A prefill walk (a request's chunk or a group's spans) of at least
  // kFoldOverlapMinRows rows walks
  // each layer in two row blocks, A = [0, TA) and B = [TA, T), so that a
  // block's boundary fold — the GPU otherwise idle through it, a fifth of
  // a 2K prefill — runs under the other block's work: the attention of B
  // under A's fold, the attention site and attention of A (the NEXT
  // layer's) under the FFN fold of B. The expert walk stays whole (its
  // cost is the weights it decodes, not its rows). Every row's arithmetic
  // is the one-block walk's (the attention is chunk-invariant, the sites
  // and the folds per row): bitwise.
  static constexpr int kFoldOverlapMinRows = 512;
  void enqueue_layer_overlap(const Dsv4LayerResident& r, int layer, int T, int TA, const WalkRows& rows, bool flush);
  void mhc_site_rows(const uint16_t* streams, const GlmMhcWeights& w, const uint16_t* ln, int r0, int rows,
                     int decode_prefix = 0);
  void update_rows(const uint16_t* sublayer_out, int r0, int rows);
  // (TRIED 2026-10-02 and reverted: the update and the next site in row
  // tiles so a tile's streams stay in L2 for the site's two passes. 128-row
  // tiles +2 % on a 2K prefill (the launches and the narrow grids), 256-row
  // tiles level at 2K and -0.8 % at 8K, inside the run-to-run spread.)
  void flush_overlap(int T, int TA);
  bool fold_overlap_ = true;
  bool ffn_b_pending_ = false;  // block B's FFN fold is outstanding (A's streams are in nxt_)
  uint64_t overlap_async_folds_ = 0;  // folds begun asynchronously (tests)
 public:
  void set_prefill_fold_overlap(bool on) { fold_overlap_ = on; }  // tests: the one-block walk beside it
  uint64_t debug_overlap_async_folds() const { return overlap_async_folds_; }
  const Bf12Companions& bf12_companions() const { return bf12_; }
 private:
  uint16_t* stage(uint16_t* fallback, int T, int width, bool capture);
  void fold(uint16_t* buf, int T, int width, bool capture);
  void gather_embedding(const int64_t* tokens, int T, bool capture);
  // The DSpark pieces (mtp): the first draft call of a step (the rings, the
  // block, base_logits_, row 0) and a chain row (block row draft_row_).
  void draft_first(int req, const int64_t* tokens, const int64_t* d_pos, const int32_t* d_req, int T, bool decode_row,
                   bool capture, int head_rows, int batch_requests, int src_row0 = 0);
  void draft_chain_row(int req, const int64_t* tokens, bool capture, int head_rows, int batch_requests);
  const Dsv4DraftResident& draft_stage(int stage);
  int target_ordinal(int layer) const;
  // The decode walk's boundary prefetch windows (kernels/l2_prefetch.hpp;
  // bit-identical on or off; resident stacks only): before the attention
  // fold the MoE site's replicated bytes and the shared expert, before the
  // MoE fold the next layer's attention head (or the head's), each under
  // the collective and the mHC site behind it. The attention layer opens
  // its own in-layer windows (Dsv4AttnLayer::set_prefetcher).
  bool prefetching(bool decode) const {
    return decode && prefetch_.enabled() && loader_.residency() == Dsv4Residency::Resident;
  }
  void prefetch_fp8(const GlmQuantMatrix& m);
  void prefetch_mhc(const uint16_t* fn, const uint16_t* norm);
  void prefetch_ffn_side(const Dsv4LayerResident& r);
  void prefetch_attention_side(int layer);
  void prefetch_head(const Dsv4HcHeadResident& hc, const uint16_t* norm, bool draft);
  WeightPrefetcher prefetch_;
  static constexpr size_t kPrefetchWindowBytes = size_t{18} << 20;
  // The decode sites' fused mHC (kernels/glm_mhc_launch.hpp): the dots, the
  // finish (this site's pre / post, the collapse, the one-rounding norm into
  // x_) in one launch; the Sinkhorn mix (comb) deferred to a side stream,
  // forked behind the finish and joined before the stream update — off the
  // sublayer's critical path. Prefill rows keep the two-launch forms.
  cudaStream_t mhc_side_ = nullptr;
  cudaEvent_t mhc_fork_ = nullptr;
  cudaEvent_t mhc_join_ = nullptr;
  int* mhc_counters_ = nullptr;  // [M] the fused finish's tickets, zero at rest
  bool comb_pending_ = false;    // a deferred comb the next stream update joins

  Dsv4Config cfg_;
  Dsv4LayerStream loader_;
  CublasLtGemm gemm_;
  // The lm head's 12-bit companion (engine.bf16_weights): the tensor-core decode
  // form reads it (mma_gemv.hpp's bf12 form); the bf16 bytes stay for prefill.
  Bf12Companions bf12_;
  static constexpr int kBf12ExpandSlots = 1;
  bool dense_mma_ = true;  // the dense projections' and head's decode form
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;
  Dsv4GlobalsResident globals_;
  GlmMoeConfig moe_cfg_;
  GlmMhcConfig mhc_cfg_;
  Dsv4AttnConfig attn_cfg_;
  bool embed_sharded_ = false;
  int world_ = 1;

  std::unique_ptr<Dsv4AttnLayer> attn_;
  std::unique_ptr<GlmMoeLayer> moe_;
  Dsv4StatePool pool_;
  void* attn_scratch_ = nullptr;
  size_t attn_scratch_bytes_ = 0;
  float* inv_freq_window_ = nullptr;      // device [32]
  float* inv_freq_compressed_ = nullptr;  // device [32]

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
  float* mhc_logits_ = nullptr;     // [M, 24]
  float* zero_bias_ = nullptr;      // [E] the hash layers' router bias (none in the reference)
  // DSpark (mtp).
  int debug_layer_limit_ = 0;
  // Per request: the last prefill walk's rows (pos0, rows, and the walk row
  // its span began at) — mtp_run_rows clamps its prefill rows to them and
  // reads their target hidden in place.
  struct WalkSegment {
    int64_t pos0 = 0;
    int rows = 0;
    int row0 = 0;
  };
  std::vector<WalkSegment> seg_by_req_;
  WalkSegment& segment_of(int req) {
    if (seg_by_req_.size() <= static_cast<size_t>(req)) seg_by_req_.resize(static_cast<size_t>(req) + 1);
    return seg_by_req_[static_cast<size_t>(req)];
  }
  int targets_ = 0;                 // dspark target layers (the draft width is targets_ * H)
  int draft_row_ = 0;               // the next block row a chain call emits (0: no block stands)
  uint16_t* main_hidden_ = nullptr; // [M, targets * H] the target layers' stream means of the walk's rows
  uint16_t* main_gather_ = nullptr; // [rows, targets * H] the draft rows' gathered main hidden
  uint16_t* main_x_ = nullptr;      // [max(M, rows), H] main_norm(main_proj(main hidden))
  uint16_t* draft_collapsed_ = nullptr;  // [rows, H] the block's hc_head rows (the confidence head's input)
  int64_t* blk_pos_ = nullptr;      // [rows] the block rows' positions
  int64_t* blk_tok_ = nullptr;      // [rows] the block rows' tokens
  int32_t* blk_req_ = nullptr;      // [rows]
  int32_t* blk_spans_ = nullptr;    // [rows + 1]
  float* base_logits_ = nullptr;    // [rows, vocab slice] the block's head rows
  float* conf_ = nullptr;           // [R, block] the confidence logits
  // The draft head's screen (build_draft_head_screen): a block-FP8 copy of
  // the local lm head — the draft rows' base logits read it instead of the
  // bf16 head (half the bytes), and every logit a pick can land on is then
  // recomputed from the bf16 rows (dsv41_dspark_rescore).
  uint8_t* draft_head_fp8_ = nullptr;     // e4m3 [vocab slice, hidden]
  float* draft_head_scales_ = nullptr;    // [ceil(vocab slice / 128), hidden / 128] powers of two
  float* draft_row_max_ = nullptr;        // [groups][ceil(vocab slice / 256)] a biased row's block maxima
  bool draft_screened_ = false;           // the standing block's base logits came off the copy
  void build_draft_head_screen();
  void draft_rescore(int block_row, int groups, int rows_out);
  // The copy's logits sit within ~0.1 of the bf16 head's; a pick's
  // candidates are the entries this close to its biased row's maximum.
  static constexpr float kDraftRescoreDelta = 2.0f;
  int64_t* d_draft_pos_ = nullptr;  // [M] the prefill draft rows' positions (staged per call)
  int32_t* d_draft_req_ = nullptr;  // [M]
  std::vector<SiteCapture> debug_sites_;
  std::vector<IndexLogits> debug_index_logits_;
  bool debug_capture_ = false;
  const std::vector<std::vector<uint16_t>>* debug_teacher_ = nullptr;
  // The prefill walk's route traces.
  int32_t* h_route_ids_ = nullptr;
  float* h_route_weights_ = nullptr;
};

}  // namespace dgpp
