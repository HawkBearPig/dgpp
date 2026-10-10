#pragma once
// Qwen3.8-Flash-Next model with resident or streaming weights. The shared
// SessionModel core manages request state and execution around run_rows(),
// which follows transformers' Qwen4ExpTextDecoderLayer:
//
//   R = embed(x) on every branch
//   per layer: [R += PLE(R, ids) at the PLE layer]
//              x, Rn = GR_mix(R);  y = GDN(x) | QSA(x);  R += s(Rn) (x) y
//              x, Rn = GR_mix(R);  y = MoE(x);           R += s(Rn) (x) y
//   h = mixer_mix(R)  ->  lm_head
//
// one row walk (run_rows) serves every entry point: the cold diagnostic
// forward (all rows' logits, the per-layer hyper states for the parity
// gate), the prefill chunks of a session (the last row's logits; the state
// left in the slot IS the state the next rows need — the GDN recurrence,
// the conv states, the PLE conv state and n-gram context, the K/V and
// compressed-key caches are updated in place, DESIGN §7.1), the eager
// decode rows (T <= kSpecRows rows of one request at its next positions)
// and the captured decode graphs (the same kernels recorded: the scalar
// device-driven step of one slot and the fixed slot-major row batch —
// engine/graph_engine.hpp's contract). Mapped n-gram storage adds a
// declared host gather node; see session_graph_host_nodes().
//
// Request slots (max_requests): every slot owns its GDN recurrent and conv
// state per GDN layer, its PLE conv state and n-gram context (device int32
// [4]: the two previous ids, a 16-byte family), its rings in every QSA
// layer, and its row of the shared block table; the K/V and index caches
// are one paged pool (models/qwen/kv_pool.hpp). Prefill opens a slot
// (zeroed state, EOS context, released blocks), close releases it.
//
// Speculative rows: a multi-row decode leaves every row's post-state in
// the spec snapshot rows (GDN recurrent/conv, PLE conv, context, rings);
// session_rollback (eager) or the recorded commit (glm_spec_commit behind
// the graph's pick) copies row accepted-1 back over the live state when
// rows are rejected. The MTP draft block uses the same session protocol.
//
// TP (plan D1): `tp_world` > 1 loads this rank's slices (GDN / QSA heads,
// expert and shared-expert intermediates on the re-blocked scale grid, hash
// heads, the lm-head vocab slice), runs the walk on the local geometry, and
// folds the partials through `boundary` — the attention and MoE block
// outputs per layer, and the PLE layer's key/value projections. A reducer
// is required exactly when tp_world > 1. Every rank's hyper state is
// bitwise the others' (the fold is the canonical rank-order sum; the
// router, indexer and GR are replicated).
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/image_input.hpp"
#include "engine/boundary_reducer.hpp"
#include "engine/decode_outputs.hpp"
#include "engine/memory_plan.hpp"
#include "engine/session_model.hpp"
#include "kernels/bf12_companions.hpp"
#include "kernels/l2_prefetch.hpp"
#include "kernels/gemm.hpp"
#include "kernels/mma_gemv.hpp"
#include "kernels/glm_spec.hpp"
#include "kernels/kda.hpp"
#include "kernels/pick.hpp"
#include "models/qwen/config.hpp"
#include "models/qwen/kv_pool.hpp"
#include "models/qwen/layers.hpp"
#include "models/qwen/loader.hpp"
#include "models/qwen/moe_layer.hpp"

namespace dgpp {

class QwenVisionEncoder;  // models/qwen/vision.hpp; built only when the
                          // checkpoint ships a vision tower

class QwenModel : public SessionModel<QwenModel> {
 public:
  using Base = SessionModel<QwenModel>;
  using Outputs = Base::Outputs;
  using SessionSnapshotMeta = Base::SessionSnapshotMeta;
  using SnapshotRequest = Base::SnapshotRequest;
  // DGPP_QWEN_HEAD_DUMP=<file>: append every decode-shaped head input row
  // (the draft's and the verify's hyper-state mixes) after each settled pass
  // — the head bit-plane study's data (2026-09-29). A debug hook: the flush
  // syncs the stream. Slots are written by memcpy nodes recorded in the
  // graph, so a replay refreshes them without host work.
  void head_dump_flush();
  // The argmax-only head (kernels/packq_head.hpp): per-slot device flags
  // the engine writes with the sampling specs — a slot whose verify rows
  // (resp. draft rows) want only the argmax. Null: every head read is full.
  void set_head_greedy_flags(const uint8_t* verify, const uint8_t* draft) {
    head_greedy_verify_ = verify;
    head_greedy_draft_ = draft;
  }
  struct PrefillCursor : Base::PrefillCursor {
    const std::vector<ImageInput>* images = nullptr;
  };
  // Several cold prompts as the spans of one walk (session_prefill_group,
  // 2026-09-14, the group prefill ported from DeepSeek): the GDN scan and
  // the QSA attention run per span (their state and cache are per
  // request), the PLE, GR, MoE and head sites over every row; every span
  // within max_tokens.
  int64_t prefill_group_span_limit() const { return max_tokens_; }
  using RowRun = Base::RowRun;

  // Small graph widths can select different Lt/GEMV reductions. Preserve
  // their physical-prefix width. Wider verification may contract within
  // the split-K (17..32 rows) or unsplit (>32 rows) range, never across it.
  static bool compact_batch_compatible(int physical_rows, int compact_rows) {
    return compact_rows > 0 && compact_rows <= physical_rows &&
           (physical_rows == compact_rows ||
            (physical_rows > 16 && compact_rows > 16 &&
             (physical_rows <= kMmaGemvMaxRows) == (compact_rows <= kMmaGemvMaxRows)));
  }

  // max_tokens bounds a walk's rows (a prefill chunk, the diagnostic
  // forward); max_cache_tokens the paged pool's capacity in tokens (rounded
  // up to a block) — shared by every slot, so one request may take it all
  // (max_context()). max_requests: the session slots.
  // mtp: the MTP draft block (§1.8; stage C) — the draft layer resident
  // beside the stack, its paged caches one more pool layer, the hyper-
  // state window per slot; enables session_draft and the in-graph draft.
  // decode_rows: the fixed decode batch's row ceiling (the slots times the
  // verify rows per request; engine/decode_outputs.hpp); 0 = kDecodeRows.
  // fp8_head_mma: opt in to streaming MMA within that decode envelope,
  // including short prefill. Bitwise parity needs matching settings/capacity.
  // serving_logits bounds head storage to decode rows and packs selected
  // prefill outputs. Leave false for all-row diagnostic forwards.
  QwenModel(const QwenTextConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
            int64_t max_cache_tokens, QwenResidency residency = QwenResidency::Streaming,
            BoundaryReducer* boundary = nullptr, int tp_rank = 0, int tp_world = 1,
            int max_requests = 1, bool mtp = false, int decode_rows = 0, bool fp8_head_mma = false, bool serving_logits = false);
  bool fp8_head_mma() const { return fp8_head_mma_; }
  ~QwenModel();
  QwenModel(const QwenModel&) = delete;
  QwenModel& operator=(const QwenModel&) = delete;

  // Every byte the constructor (and its layer objects) will allocate for
  // a shape, itemized from the same formulas BEFORE anything is allocated
  // (engine/memory_plan.hpp): the serving app's pre-flight check.
  using MemoryPlan = dgpp::MemoryPlan;
  static MemoryPlan plan_memory(const QwenTextConfig& cfg, int max_tokens, int64_t max_cache_tokens,
                                int tp_rank = 0, int tp_world = 1,
                                QwenResidency residency = QwenResidency::Streaming,
                                int max_requests = 1, bool mtp = false, int decode_rows = 0, bool serving_logits = false);

  // The cold diagnostic forward: one request on slot 0 (which must be
  // closed), fresh state, every row's logits; the slot is closed after.
  Outputs forward(const std::vector<int64_t>& token_ids, bool capture_layers = false);
  // Test gate: eager decode walks copy every layer's output residual into
  // Outputs::layer_states (the row-count invariance test's bisection).
  static void set_session_capture_layers(bool on) { session_capture_layers_ = on; }
  inline static bool session_capture_layers_ = false;
  // The draft block over a prompt (the parity gate's surface): the cold
  // forward, then the draft rows q = 0 .. T-2 (token q+1, the hyper state
  // at q) with the head on every row — logits [T-1, count], the mixer's
  // final hidden [T-1, hidden].
  Outputs mtp_forward(const std::vector<int64_t>& token_ids);


  int64_t kv_block_tokens() const { return num_qsa_ > 0 ? kBlockTokens : 0; }
  static constexpr int prefill_chunk_tokens() { return kPrefillChunkTokens; }
  static constexpr int decode_rows_cap() { return kDecodeRowsMax; }
  static constexpr bool kResumablePrefill = true;
  static constexpr bool kCompactBatches = true;
  static constexpr int kv_block_tokens_static() { return kBlockTokens; }
  // The same number for a shape that is not built yet (the memory plan).
  static size_t session_snapshot_bytes(const QwenTextConfig& cfg, int tp_world, bool mtp);
  using Base::session_snapshot_bytes;

  const QwenTextConfig& config() const { return cfg_; }
  // The bf16 decode weights' 12-bit companions (gates read its counters).
  const Bf12Companions& bf12_companions() const { return bf12_; }
  const QwenKvPool& kv_pool() const { return pool_; }

  // ---- the session core's hooks (engine/session_model.hpp) --------------------
  Outputs run_rows(const RowRun& run);
  void reset_slot_state(int req);
  GlmSpecSegments spec_segments(int req, int snapshot_row0 = 0) const;
  size_t snapshot_state_bytes() const;
  size_t draft_state_bytes() const;
  void write_state_snapshot(int req, uint8_t* dst, int spec_row);
  void write_draft_snapshot(int req, uint8_t* dst, bool live, int64_t pos);
  void read_state_snapshot(int req, const uint8_t* src);
  void read_draft_snapshot(int req, const uint8_t* src);
  bool has_pool() const { return num_qsa_ > 0; }
  QwenKvPool& pool() { return pool_; }
  const QwenKvPool& pool() const { return pool_; }

  // ---- image inputs (docs/vision.md) --------------------------------------
  // The checkpoint's own BF16 tower, replicated on every rank and run only
  // on image prefills. Its rows replace the embedding at the prompt's
  // image_token_id positions (deepstack_visual_indexes is empty, so there is
  // nowhere else for them to enter); decode graphs are untouched.
  bool supports_images() const { return vision_ != nullptr; }
  // The pad id repeated once per visual token, -1 without a tower.
  int64_t image_pad_id() const { return cfg_.vision ? cfg_.vision->tokens.pad : -1; }
  // The three delimiters, for the serving frontend that renders them.
  ImageTokens image_tokens() const { return cfg_.image_tokens(); }
  uint64_t vision_digest() const;  // 0 without a tower; see models/qwen/vision.hpp
  Outputs session_prefill_images(int req, const std::vector<int64_t>& prompt_ids,
                                 const std::vector<ImageInput>& images);
  Outputs session_prefill_images(int req, const std::vector<int64_t>& prompt_ids,
                                 const std::vector<ImageInput>& images,
                                 const std::vector<int64_t>& boundaries, SnapshotRequest* snap);
  Outputs session_prefill_resume_images(int req, const std::vector<int64_t>& suffix_ids,
                                         const std::vector<ImageInput>& images,
                                         const std::vector<int64_t>& boundaries,
                                         SnapshotRequest* snap);
  // Chunked image prefill (the engine's yield path): validates the images,
  // borrows them for the cursor's lifetime, and runs the base text machinery
  // with image state scoped to each advance. This overload is what flips
  // supports_image_chunked_prefill() on.
  PrefillCursor session_prefill_begin(int req, const std::vector<int64_t>& prompt,
                                      int64_t reserve_tokens, int64_t chunk_tokens,
                                      const std::vector<int64_t>& boundaries = {},
                                      SnapshotRequest* snap = nullptr, int64_t attach_position = 0,
                                      const std::vector<ImageInput>* images = nullptr);
  bool session_prefill_advance(PrefillCursor& cursor, int64_t chunk_tokens = 0);
  void graph_prepare();
  // Eagerly materializes every resident layer (text stack + MTP draft when
  // resident) on `threads` loader workers; a no-op unless resident and
  // threads > 1. Called once before serving so the first forwards — the
  // graph warmup — find all layers resident instead of faulting them in
  // one by one.
  void preload_resident_layers(int threads);
  void mtp_run_rows(int req, const int64_t* tokens, int64_t first_pos, int T, bool decode_row,
                    bool capture, int head_rows, int batch_requests);
  void snapshot_draft_state(int req);
  void restore_draft_state(int req);
  // Depth >= 2: the chain rows run the draft block forward
  // past the first draft, advancing its QSA ring; the ring is copied aside
  // before the first chain row and restored after the last (GLM-5.3-Flash's
  // chain_ring_copy), into a buffer of its own — the draft snapshot the
  // fallback's rollback restores must keep the pre-draft ring.
  static constexpr bool kDraftChain = true;
  static constexpr bool kBatchedDraftChain = true;
  // The mmap'ed n-gram table's walk carries one host node (the staging
  // gather forked inside the walk; layers.hpp) — a captured step's verify
  // walk; the draft block has no PLE.
  size_t session_graph_host_nodes() const { return has_ple_ && table_.mmap ? 1 : 0; }
  const uint16_t* draft_hidden_rows() const { return mtp_r_; }
  void snapshot_chain_state(int req);
  void restore_chain_state(int req);

 private:
  const bool fp8_head_mma_;
  static constexpr int kBlockTokens = 64;
  static constexpr int kPrefillChunkTokens = 4096;

  // The images of the prefill currently running (null outside one), plus the
  // staged rows covering [image_window_first_, image_window_end_). The engine
  // owns the vector for the call's whole lifetime, across chunks.
  Outputs session_prefill_with_images(int req, const std::vector<int64_t>& ids,
                                     const std::vector<ImageInput>& images,
                                     const std::vector<int64_t>& boundaries, SnapshotRequest* snap,
                                     bool resume);
  void stage_image_embeddings(int64_t first, int64_t end, const std::vector<ImageInput>* images);
  // Rows [first + shift, first + shift + rows) into dst, whose rows are
  // `branches` copies of hidden wide. `shift` is 0 for the main walk's
  // embedding (branches = hc_count) and 1 for the draft's, whose row at
  // position p embeds token p + 1 (branches = 1).
  void apply_image_embeddings(uint16_t* dst, int64_t first, int rows, int shift, int branches,
                              const std::vector<ImageInput>* images);
  std::unique_ptr<QwenVisionEncoder> vision_;
  // The cursor retains the engine-owned image vector across yields. Only
  // an executing prefill borrows it into the model; the scope also invalidates
  // the shared staging window when entering and leaving a request's walk.
  class ImagePrefillScope;
  const std::vector<ImageInput>* images_for_req(int req) const;
  void store_request_images(int req, const std::vector<ImageInput>* images);
  std::vector<const std::vector<ImageInput>*> prefill_images_per_req_;
  const uint16_t* image_embeddings_ = nullptr;
  int64_t image_window_first_ = 0, image_window_end_ = 0;

  void build_layer_objects(const QwenLayerResident& r);
  void lm_head_logits(const uint16_t* hidden, int rows, cudaStream_t stream, bool last_row_only = false,
                      int compact_row = -1, int output_row = 0);
  static size_t dense_bridge_bytes(const QwenTextConfig& cfg, const QwenLocalGeometry& geo);
  // The opt-in fp8 prefill GEMM's activation scratch (engine.prefill_fp8_gemm):
  // the widest dense k of this rank's slice, and the scratch's bytes for
  // max_tokens rows (e4m3 rows plus their 1 x 128 fp32 scales).
  static size_t dense_max_cols(const QwenTextConfig& cfg, const QwenLocalGeometry& geo);
  static size_t fp8_act_scratch_bytes(const QwenTextConfig& cfg, const QwenLocalGeometry& geo, int max_tokens);
  size_t dense_bridge_bytes_ = 0;
  static QwenMoeWeights moe_view(const QwenMoeResident& m);
  float* gdn_rec(int req, int ordinal) const;
  uint16_t* gdn_conv(int req, int ordinal) const;
  // The GDN recurrent state's checkpoint-and-replay form under MTP
  // (kernels/kda.hpp KdaReplay, 2026-09-29): the live buffer holds the
  // state BEFORE the last pass's rows; the pass's rows are kept as their
  // inputs and the next pass replays the accepted ones (gdn_pending_).
  // Nothing but a decode walk may read the live buffer without
  // materialize_gdn() first (a prefill continuation, the prefix cache).
  KdaReplay gdn_replay_view(int req, int ordinal, bool batched) const;
  size_t gdn_replay_block_elems() const;  // one request's saved rows, every GDN layer
  // The state as it stands after the pending rows (rows < 0) or after the
  // first `rows` saved rows: in place (dst null; the pending count resets)
  // or into dst [num_gdn][rec] (the live buffer untouched).
  void materialize_gdn(int req, float* dst = nullptr, int rows = -1);
  void head_dump_record(const uint16_t* hidden, int rows, cudaStream_t stream);
  static constexpr int kHeadDumpSlots = 8;
  uint16_t* head_dump_ring_ = nullptr;   // device [kHeadDumpSlots][max_decode_rows][H]
  uint8_t* head_dump_meta_ = nullptr;    // device [kHeadDumpSlots][2]: rows, site (0 verify, 1 draft)
  int head_dump_next_ = 0;
  int head_dump_site_ = 0;
  const uint8_t* head_greedy_verify_ = nullptr;
  const uint8_t* head_greedy_draft_ = nullptr;
  int head_req_ = 0;                 // the request of a single-request head call
  float* head_ab_ = nullptr;         // [4][V] the argmax pass's upper bounds
  int32_t* head_lo_ = nullptr;       // [4] the best lower bounds (ordered ints)
  int32_t* head_cand_ = nullptr;     // [V] the candidate rows, compacted
  int32_t* head_cand_count_ = nullptr;
  std::string head_dump_path_;
  std::vector<uint16_t> head_dump_host_;
  std::vector<uint8_t> head_dump_meta_host_;
  size_t head_dump_rows_written_ = 0;
  static bool env_is_off_replay();
  uint16_t* ple_conv(int req) const;
  int32_t* ctx(int req) const { return d_ctx_ + static_cast<size_t>(req) * 4; }
  void push_context(int req, int32_t prev1, int32_t prev2);
  size_t ring_elems() const;
  int pool_layers() const { return num_qsa_ + (mtp_ ? 1 : 0); }

  QwenTextConfig cfg_;
  QwenLayerStream loader_;
  CublasLtGemm gemm_;
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;
  QwenGemmWorkspace gw_;
  QwenGlobalsResident globals_;
  QwenNgramTableResident table_;
  bool has_ple_ = false;
  int n_moe_layers_ = 0;

  // Layer objects (built at first use, rebound per layer).
  std::unique_ptr<QwenGrSite> attn_gr_, mlp_gr_, mixer_;
  std::unique_ptr<QwenGdnLayer> gdn_;
  std::unique_ptr<QwenQsaLayer> qsa_;
  std::unique_ptr<QwenMoeLayer> moe_;
  std::unique_ptr<QwenPleLayer> ple_;
  GlmMoeConfig moe_cfg_;

  // Per-slot state.
  int num_gdn_ = 0, num_qsa_ = 0;
  int64_t gdn_rec_elems_ = 0, gdn_conv_elems_ = 0, ple_conv_elems_ = 0;
  float* gdn_rec_ = nullptr;            // [R][num_gdn][lv*V*K]
  uint16_t* gdn_conv_ = nullptr;        // [R][num_gdn][C*(width-1)]
  uint16_t* ple_conv_state_ = nullptr;  // [R][hc*H*state_len]
  int32_t* d_ctx_ = nullptr;            // [R][4]: the n-gram context (prev1, prev2, 0, 0)
  int32_t* h_ctx_ = nullptr;            // pinned upload source [R][4]
  QwenKvPool pool_;                     // the QSA layers' paged caches

  // The spec snapshot rows (every family, row-major by decode row).
  float* spec_rec_ = nullptr;      // [rows][num_gdn][rec]  (rows = max_decode_rows_; null under the replay form)
  bool gdn_replay_ = false;             // the checkpoint-and-replay form (MTP stacks with GDN layers)
  uint16_t* gdn_replay_in_ = nullptr;   // [R][num_gdn][rows_cap][row]: the rows the next pass replays
  uint16_t* gdn_replay_save_ = nullptr; // [R][num_gdn][rows_cap][row]: this pass's rows (the commit copies them over `in`)
  int32_t* gdn_pending_ = nullptr;      // [R]: rows to replay (the commit writes the accepted count)
  int64_t gdn_replay_row_elems_ = 0;    // the post-conv q|k|v row + a_raw + beta_raw
  int gdn_replay_rows_cap_ = 0;
  int gdn_replay_rows_used_ = 1;        // rows per request of the last decode walk (the commit's copy)
  uint16_t* spec_conv_ = nullptr;  // [rows][num_gdn][conv]
  uint16_t* spec_ple_ = nullptr;   // [rows][ple]
  int32_t* spec_ctx_ = nullptr;    // [max(max_tokens, rows)][4]
  uint16_t* spec_ring_ = nullptr;  // [num_qsa][rows][kpool*Di]

  // Activations [M rows] (the token rows, the head's outputs and the tail
  // mirrors are the core's).
  uint16_t* r_ = nullptr;           // [M, hc*H]
  uint16_t* x_ = nullptr;           // [M, H]
  uint16_t* y_ = nullptr;           // [M, H]
  // The prefill walk's route traces: one pinned staging slot per layer
  // ([layers][M][top_k] ids and weights), materialized after the walk's
  // final sync (the MoE's device-segmented chain never syncs per layer).
  int32_t* h_route_ids_ = nullptr;
  float* h_route_weights_ = nullptr;
  bool moe_prefill_host_path_ = false;  // DGPP_QWEN_MOE_PREFILL=host: the host-orchestrated chain
  // The L2 weight prefetcher (2026-09-09, the decode profile: 99
  // collectives per step at 28-57 us each, DRAM idle through every one).
  // Decode rows open a window before each fold with the other side's
  // first weights in consumption order (GLM's boundary windows); the
  // side stream rejoins before the walk's end. engine.l2_prefetch false A/Bs.
  // The bf16 decode weights' lossless 12-bit companions (engine.bf16_weights;
  // kernels/bf12_companions.hpp): the GDN and QSA projections, the draft
  // block's, the head — packed as each layer lands in graph_prepare. The GR
  // sites, the routers and the shared experts keep their bf16 form (their
  // decode kernels are latency-bound at those shapes: the record's round 4).
  void pack_layer_companions(int layer, const QwenLayerResident& r);
  void finish_companions();
  static size_t bf12_plan_bytes(const QwenTextConfig& cfg, const QwenLocalGeometry& geo, bool mtp);
  Bf12Companions bf12_;
  bool bf12_built_ = false;
  double bf12_s_ = 0.0;
  int walk_rows_ = 1;  // the rows of the walk in flight (the prefetch windows' view)
  WeightPrefetcher prefetch_;
  size_t prefetch_window_bytes_ = 0;  // 0 = the prefetcher's window budget (engine.l2_prefetch_boundary_window_mib)
  void prefetch_gr(const QwenGrResident& g, bool inject);
  void prefetch_ffn_side(const QwenLayerResident& r);
  void prefetch_attention_side(int layer);
  void prefetch_head(const QwenGrResident& mixer);
  void prefetch_add(const char* what, const void* p, size_t bytes);
  // A bf16 matmul weight into the open window: the bytes the walk's launch
  // streams — its packed companion's when the GEMM holds one.
  void prefetch_bf16(const char* what, const uint16_t* w, size_t bytes);
  void prefetch_fp8(const char* what, const GlmQuantMatrix& q);
  void prefetch_ple_key_side(const QwenLayerResident& r);
  void prefetch_ple_value_side(const QwenLayerResident& r);

  // The draft block (mtp_): its ring snapshot and fusion scratch (the
  // window, the counters and the feeds are the core's).
  uint16_t* mtp_ring_snapshot_ = nullptr;  // [R][kpool*Di]: the draft ring before its rows
  uint16_t* mtp_chain_ring_ = nullptr;     // [R][kpool*Di]: the ring around the chain rows (depth >= 2)
  uint16_t* mtp_hin_ = nullptr;         // [M, W] the gathered hyper states
  uint16_t* mtp_hn_ = nullptr;          // [M, W] their full-vector norm
  uint16_t* mtp_e_ = nullptr;           // [M, H] the embedding rows
  uint16_t* mtp_en_ = nullptr;          // [M, H] their norm
  uint16_t* mtp_ein_ = nullptr;         // [M, H] fc_embedding
  uint16_t* mtp_enc_ = nullptr;         // [M, W] fc_hidden per branch
  uint16_t* mtp_r_ = nullptr;           // [M, W] the block's hyper state
  std::unique_ptr<QwenGrSite> mtp_mixer_;
};

}  // namespace dgpp
