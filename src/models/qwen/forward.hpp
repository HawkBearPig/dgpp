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
  QwenModel(const QwenTextConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
            int64_t max_cache_tokens, QwenResidency residency = QwenResidency::Streaming,
            BoundaryReducer* boundary = nullptr, int tp_rank = 0, int tp_world = 1,
            int max_requests = 1, bool mtp = false, int decode_rows = 0, bool fp8_head_mma = false);
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
                                int max_requests = 1, bool mtp = false, int decode_rows = 0);

  // The cold diagnostic forward: one request on slot 0 (which must be
  // closed), fresh state, every row's logits; the slot is closed after.
  Outputs forward(const std::vector<int64_t>& token_ids, bool capture_layers = false);
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
  void graph_prepare();
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
  void stage_image_embeddings(int64_t first, int64_t end);
  // Rows [first + shift, first + shift + rows) into dst, whose rows are
  // `branches` copies of hidden wide. `shift` is 0 for the main walk's
  // embedding (branches = hc_count) and 1 for the draft's, whose row at
  // position p embeds token p + 1 (branches = 1).
  void apply_image_embeddings(uint16_t* dst, int64_t first, int rows, int shift, int branches);
  std::unique_ptr<QwenVisionEncoder> vision_;
  const std::vector<ImageInput>* prefill_images_ = nullptr;
  const uint16_t* image_embeddings_ = nullptr;
  int64_t image_window_first_ = 0, image_window_end_ = 0;

  void build_layer_objects(const QwenLayerResident& r);
  void lm_head_logits(const uint16_t* hidden, int rows, cudaStream_t stream, bool last_row_only = false);
  static size_t dense_bridge_bytes(const QwenTextConfig& cfg, const QwenLocalGeometry& geo);
  size_t dense_bridge_bytes_ = 0;
  static QwenMoeWeights moe_view(const QwenMoeResident& m);
  float* gdn_rec(int req, int ordinal) const;
  uint16_t* gdn_conv(int req, int ordinal) const;
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
  float* spec_rec_ = nullptr;      // [rows][num_gdn][rec]  (rows = max_decode_rows_)
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
  // side stream rejoins before the walk's end. DGPP_L2_PREFETCH=off A/Bs.
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
  size_t prefetch_window_bytes_ = 0;  // 0 = the prefetcher's default (the DGPP_L2_PREFETCH_MB knob)
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
