// Qwen3.5-27B dense model (FP8 or mixed NVFP4, standard pre-norm residual).
//
// 48 GDN layers (swish gate) + 16 Full GQA layers, dense SwiGLU MLPs,
// and the checkpoint's optional BF16 vision tower. The attention and GDN
// layers are the shared qwen4_exp components; the K/V pool
// is a slim QwenKvPool clone without the indexer structures; GDN recurrent
// and conv state is model-owned per request slot.
//
// MTP: one Full draft layer (mtp.layers.0) + fused fc [H, 2H] over
// cat(pre_fc_norm_embedding(embed), pre_fc_norm_hidden(hidden)) + mtp.norm,
// sharing the lm head. Draft width is H (no hyperconnections); the session
// core's window holds the post-norm final hidden rows (h_).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/graph.hpp"
#include "engine/paged_blocks.hpp"
#include "engine/session_model.hpp"
#include "models/glm/moe.hpp"
#include "models/qwen/dflash.hpp"
#include "models/qwen/moe_layer.hpp"
#include "kernels/bf12_companions.hpp"
#include "kernels/draft_head.hpp"
#include "kernels/gemm.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/glm_spec.hpp"
#include "loaders/resident_stream.hpp"
#include "models/quant_matrix.hpp"
#include "common/image_input.hpp"
#include "models/qwen/config.hpp"
#include "models/qwen/config35.hpp"

// Every image span must name pad ids in the prompt it rides with (host-side
// rule, pinned by unit test; the engine-owned ids outlive the call).
namespace dgpp {
inline void qwen35_check_image_pads(const std::vector<int64_t>& ids, int64_t start,
                                    const std::vector<ImageInput>& images, int64_t pad) {
  for (const auto& im : images)
    for (int64_t pos = std::max<int64_t>(start, im.offset); pos < im.offset + im.tokens; ++pos) {
      if (pos - start < 0 || pos - start >= static_cast<int64_t>(ids.size()) ||
          ids[static_cast<size_t>(pos - start)] != pad)
        throw std::invalid_argument("Qwen3.5: image span does not contain image tokens");
    }
}
}  // namespace dgpp
#include "models/qwen/vision.hpp"
#include "kernels/l2_prefetch.hpp"
#include "kernels/latent_format.hpp"
#include "kernels/sample_pick.hpp"
#include "models/dflash2/draft_planes.hpp"
#include "models/dflash2/drafter.hpp"
#include "models/qwen/layers.hpp"
#include "models/qwen/loader.hpp"
#include "models/qwen/loader35.hpp"
#include "engine/memory_plan.hpp"

namespace dgpp {

struct Qwen35KvPoolShape {
  int layers = 0;  // Full layers served
  int kv_heads = 0;
  int dim = 0;  // head_dim
  int block_tokens = 0;
  int max_requests = 0;
  int64_t token_slots = 0;  // pool capacity in tokens, a multiple of block_tokens
  // The DFlash2 drafter's planes (0 without a drafter): extra layers in
  // the SAME block table with their own kv geometry — the draft reads its
  // context through the request's own block spans, so prefix sharing and
  // the positional overwrite-rollback apply to them verbatim.
  int draft_layers = 0;
  int draft_kv_heads = 0;
  int draft_dim = 0;
  // The Full layers' planes as e4m3 codes (engine.kv_dtype = fp8: half the
  // bytes, the layers' shipped k_scale / v_scale; kernels/full_attn.hpp).
  // The draft planes stay bf16. Needs the staged attention geometry (dim
  // 256, a power-of-two page of >= 32 tokens).
  bool fp8 = false;
  // With fp8: the prefill attention on the fp8 tensor cores
  // (engine.prefill_attention = fp8 — Q and P quantized to e4m3,
  // kernels/full_attn.hpp); decode rows keep the storage form's exact
  // walk. Same bytes, same maps.
  bool fp8_attention = false;
};

// The Full layers' paged K/V caches: per-layer K and V bf16 rows for the
// rank's kv heads plus one block table int32 [max_requests,
// blocks_per_request] shared by every layer. Same protocol as QwenKvPool
// (engine/paged_blocks.hpp owns the table mechanics); no index cache, no
// ring — dense attention reads only the K/V planes.
class Qwen35KvPool {
 public:
  Qwen35KvPool() = default;
  ~Qwen35KvPool();
  Qwen35KvPool(const Qwen35KvPool&) = delete;
  Qwen35KvPool& operator=(const Qwen35KvPool&) = delete;

  void init(const Qwen35KvPoolShape& shape);
  bool initialized() const { return initialized_; }
  // The device bytes init allocates for a shape (the memory plan).
  static size_t cache_bytes(const Qwen35KvPoolShape& shape);

  const Qwen35KvPoolShape& shape() const { return shape_; }
  int64_t total_blocks() const { return table_.total_blocks(); }
  int64_t token_slots() const { return shape_.token_slots; }
  int64_t blocks_in_use() const { return table_.blocks_in_use(); }
  int64_t free_blocks() const { return table_.free_blocks(); }
  int64_t block_count_for_tokens(int64_t tokens) const { return table_.block_count_for_tokens(tokens); }
  const PagedBlockTable& blocks() const { return table_; }

  // The kernels' view of one layer's caches (the shared table inside).
  // layer < shape().layers: a Full layer; above: a DFlash2 draft plane.
  QwenFullAttnCache view(int layer) const;
  int total_planes() const { return shape_.layers + shape_.draft_layers; }

  // ---- block management (the shared table's protocol) --------------------
  bool ensure_request_blocks(int req, int64_t tokens, cudaStream_t stream) {
    return table_.ensure_request_blocks(req, tokens, stream);
  }
  void release_request_blocks(int req, cudaStream_t stream) {
    table_.release_request_blocks(req, stream);
  }
  int64_t request_blocks(int req) const { return table_.request_blocks(req); }
  const int32_t* request_table_row(int req) const { return table_.request_table_row(req); }
  // Per-request open: release its blocks (K/V rows are always written
  // before they are read, so no zeroing is needed).
  void reset_request(int req, cudaStream_t stream);
  // Cold start: zero every cache plane and table row; all blocks free.
  void reset_all(cudaStream_t stream);

  // ---- sharing (the prefix cache) --------------------------------------
  bool share_blocks_into(int req, const int32_t* blocks, int64_t n, cudaStream_t stream) {
    return table_.share_blocks_into(req, blocks, n, stream);
  }
  void pin_blocks(const int32_t* blocks, int64_t n) { table_.pin_blocks(blocks, n); }
  void unpin_blocks(const int32_t* blocks, int64_t n) { table_.unpin_blocks(blocks, n); }
  int32_t acquire_pinned_block() { return table_.acquire_pinned_block(); }
  // Every layer's rows of physical block `src` into `dst`, stream-ordered.
  void copy_block_contents(int32_t src, int32_t dst, cudaStream_t stream);
  int32_t block_refcount(int32_t block) const { return table_.block_refcount(block); }

 private:
  Qwen35KvPoolShape shape_;
  bool initialized_ = false;
  PagedBlockTable table_;
  uint8_t* k_base_ = nullptr;  // [layers][token_slots][kv_heads * dim] bf16, or e4m3 codes (shape_.fp8)
  // The TMA maps over the Full-layer planes (kernels/full_attn.hpp), built
  // at init when the geometry is the staged form's (dim 256).
  FullAttnStageMaps maps_{};
  bool maps_ready_ = false;
  uint8_t* v_base_ = nullptr;
  // The DFlash2 drafter's plain bf16 planes over the shared block table
  // (models/dflash2/draft_planes.hpp; 0 layers without a v2 drafter).
  DraftKvPlanes draft_;
  size_t kv_row_elems() const { return static_cast<size_t>(shape_.kv_heads) * shape_.dim; }
  size_t layer_kv_elems() const { return static_cast<size_t>(shape_.token_slots) * kv_row_elems(); }
  size_t elem_bytes() const { return shape_.fp8 ? 1 : 2; }
  size_t layer_kv_bytes() const { return layer_kv_elems() * elem_bytes(); }
  void check_req(int req, const char* what) const;
};

// Fused SwiGLU (model35_kernels.cu): out[i] = silu(gate[i]) * up[i], bf16.
void qwen35_swiglu_bf16(const uint16_t* gate, const uint16_t* up, uint16_t* out, int64_t n,
                        cudaStream_t stream);

// The MTP fusion concat: out[t, :] = [e[t, :], h[t, :]] (model35_kernels.cu).
void qwen35_mtp_concat_bf16(const uint16_t* e, const uint16_t* h, uint16_t* out, int64_t rows,
                            int64_t hidden, cudaStream_t stream);

// The v1 draft's column-0 gather (model35_kernels.cu): tok[j] = ids[(j + 1) * stride]
// (mask rows 1.., skipping the anchor row's column 0). One kernel, so the
// captured draft stays kernels-only (no per-draft D2D nodes).
void df1_collect_top1(const int32_t* ids, int stride, int32_t* tok, int n, cudaStream_t stream);

class Qwen35Model : public SessionModel<Qwen35Model> {
 public:
  using Base = SessionModel<Qwen35Model>;
  static constexpr bool kPrefillChunkGrid = false;
  void prepare_mixed_verify(int rows) { gdn_replay_rows_used_ = rows; }

  // `dflash2_dir` (a directory holding the drafter's config.json + shards)
  // replaces the MTP draft with the DFlash2 block drafter: mutually
  // exclusive with `mtp`. The target verify captures (the static
  // padded-8-row-block graph, session_verify_batch_graph); only the
  // block draft itself stays eager (its top-K selection roundtrips to
  // the host), running between replays.
  // `kv_format`: the Full layers' K/V planes — bf16, or fp8 (the e4m3
  // storage form at the release's shipped kv scales; Qwen35KvPoolShape::fp8).
  Qwen35Model(const Qwen35TextConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
              int64_t max_cache_tokens, LoaderResidency residency, BoundaryReducer* boundary, int rank,
              int world, int max_requests, int decode_rows, bool mtp = false,
              const std::string& dflash2_dir = "", LatentFormat kv_format = LatentFormat::kBf16);
  ~Qwen35Model();

  // The decode batch's row ceiling (engine/decode_outputs.hpp's
  // kDecodeRowsMax): eight request slots at the DFlash2 block's eight verify
  // rows (2026-10-05; 32 until then, which left eight live drafter slots
  // past the widest four-slot family on scalar replays — one node's C8 read
  // 17–39 tok/s against C4's 44–96). The verify's per-row GDN snapshots
  // scale with it (3.1 MB a layer a row at world 1: 9.7 GB at 64 rows).
  static constexpr int decode_rows_cap() { return 64; }
  // The two opt-in FP8 levers beyond the checkpoint, set from the cluster
  // config before plan_memory and the constructor read them:
  // engine.prefill_fp8_per_tensor (the per-tensor prefill recipe) and
  // engine.dense_weights = fp8 (the BF16 lm head requantized to block FP8).
  // Both change greedy transcripts; the defaults are the exact paths.
  static void set_prefill_fp8_per_tensor(bool on) { prefill_fp8_per_tensor_ = on; }
  // engine.dflash_weights fp8: the drafter's five block matrices served as
  // block-128 E4M3 (lossy for the proposals only; the target's verify is
  // exact whatever the drafter proposes). Before plan_memory / the ctor.
  static void set_dflash2_weights_fp8(bool on) { dflash2_fp8_ = on; }
  static bool prefill_fp8_per_tensor() { return prefill_fp8_per_tensor_; }
  static void set_dense_weights_fp8(bool on) { dense_weights_fp8_ = on; }
  // engine.prefill_activations = native (2026-10-15): the checkpoint's own
  // activation recipe for every prefill row — the mixed release's NVFP4
  // MLP activations (per row, the shipped input globals) on the block-
  // scaled fp4 tensor cores. Read by plan_memory and the constructor.
  static void set_native_activations(bool on) { native_activations_ = on; }
  static bool native_activations() { return native_activations_; }
  // engine.prefill_attention = fp8 (2026-10-16): the Full layers' prefill
  // attention on the fp8 tensor cores over the fp8 K/V pool (Q and the
  // probabilities quantized to e4m3; kernels/full_attn.hpp). Needs
  // engine.kv_dtype fp8 (refused otherwise). Read by plan_memory and the
  // constructor; decode rows keep the storage form's exact walk.
  static void set_fp8_attention(bool on) { fp8_attention_ = on; }
  static bool fp8_attention() { return fp8_attention_; }
  // Test gate: decode walks (eager, not under a capture) copy every layer's
  // output residual into Outputs::layer_states, as the diagnostic forward
  // does (the row-count invariance test's per-layer bisection).
  static void set_session_capture_layers(bool on) { session_capture_layers_ = on; }
  static bool dense_weights_fp8() { return dense_weights_fp8_; }
  // The DFlash2 drafter's serving options (engine.dflash_verify_graph,
  // engine.dflash_draft_batch, engine.dflash_depth), set from the cluster
  // config before the engine is built; the eager engine reads them.
  static void set_dflash_options(bool verify_graph, bool draft_batch, int depth) {
    dflash_verify_graph_ = verify_graph;
    dflash_draft_batch_ = draft_batch;
    dflash_depth_ = depth;
  }
  static bool dflash_verify_graph() { return dflash_verify_graph_; }
  static bool dflash_draft_batch() { return dflash_draft_batch_; }
  static int dflash_depth() { return dflash_depth_; }
  // Group prefills (one walk, a span per request): spans share the
  // max_tokens activation rows; the scheduler only groups snapshot-free
  // members (admissible_group), so no snapshot plumbing is needed.
  int64_t prefill_group_span_limit() const { return max_tokens(); }
  static constexpr int prefill_chunk_tokens() { return 4096; }
  // The full-attention prefill grid's wave: 16 query rows per CTA, one CTA
  // per SM, and one CTA per KV head of this rank (kernels/full_attn.cu).
  int64_t prefill_wave_rows() const { return wave_rows_; }
  static constexpr int kv_block_tokens_static() { return 64; }
  // Prefix-snapshot arena size: all GDN recurrent + conv state of one slot.
  static size_t session_snapshot_bytes(const Qwen35TextConfig& cfg, int world, bool mtp);
  size_t session_snapshot_bytes() const { return session_snapshot_bytes(cfg_, 1, mtp_); }
  static MemoryPlan plan_memory(const Qwen35TextConfig& cfg, int max_tokens, int64_t max_cache_tokens,
                                int rank, int world, LoaderResidency residency, int max_requests,
                                bool mtp = false, int decode_rows = 0,
                                const std::string& dflash2_dir = "",
                                LatentFormat kv_format = LatentFormat::kBf16);

  const Qwen35TextConfig& config() const { return cfg_; }
  // Actual resident layer allocation, for comparing initialization with the
  // memory plan. An unmaterialized or streamed layer contributes zero bytes.
  size_t resident_layer_bytes(int layer) const { return loader_.resident_layer_span(layer).second; }

  // ---- image inputs ---------------------------------------------------------
  // The checkpoint's own BF16 tower (vision_ != null when parsed). Its rows
  // replace the embedding at the prompt's image_token_id positions; decode
  // graphs are untouched. The DFlash2 draft embeds token ids itself: pad
  // rows read the pad id's embedding there, so drafts at image positions
  // are weaker but the verify stays exact.
  bool supports_images() const { return vision_ != nullptr; }
  // The pad id repeated once per visual token, -1 without a tower.
  int64_t image_pad_id() const { return cfg_.vision ? cfg_.vision->tokens.pad : -1; }
  // The three delimiters, for the serving frontend that renders them.
  ImageTokens image_tokens() const { return cfg_.vision ? cfg_.vision->tokens : ImageTokens{}; }
  struct PrefillCursor : Base::PrefillCursor {
    const std::vector<ImageInput>* images = nullptr;
  };
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

  // ---- the session core's hooks (engine/session_model.hpp) --------------------
  typename Base::Outputs run_rows(const typename Base::RowRun& run);
  void reset_slot_state(int req);
  GlmSpecSegments spec_segments(int req, int snapshot_row0 = 0) const;
  size_t snapshot_state_bytes() const;
  // Paged K/V: the draft block reads its own pool plane, so there is no
  // draft ring to snapshot; the core's window row rides the core plan.
  size_t draft_state_bytes() const { return 0; }
  void write_state_snapshot(int req, uint8_t* dst, int spec_row);
  void read_state_snapshot(int req, const uint8_t* src);
  void write_draft_snapshot(int, uint8_t*, bool, int64_t) {}
  void read_draft_snapshot(int, const uint8_t*) {}
  void snapshot_draft_state(int) {}
  void restore_draft_state(int) {}
  // The MTP draft chain (depth >= 2): the chain rows append draft-plane
  // K/V at uncommitted positions only, and every draft position is written
  // before it is ever read as committed (rejected guesses are overwritten
  // in place by the next draft), so there is no recurrent draft state to
  // snapshot — unlike QwenModel's ring, the paged draft plane needs no
  // chain copy. The hooks stay as documented no-ops.
  static constexpr bool kDraftChain = true;
  static constexpr bool kBatchedDraftChain = true;
  const uint16_t* draft_hidden_rows() const { return mtp_h_; }
  void snapshot_chain_state(int) {}
  void restore_chain_state(int) {}
  bool has_pool() const { return true; }
  Qwen35KvPool& pool() { return pool_; }
  const Qwen35KvPool& pool() const { return pool_; }
  void graph_prepare();
  void mtp_run_rows(int req, const int64_t* tokens, int64_t first_pos, int T, bool decode_row,
                    bool capture, int head_rows, int batch_requests, int hidden_row = 0);
  Outputs mtp_forward(const std::vector<int64_t>& token_ids);
  // The cold diagnostic forward (the fixture gates): slot 0, fresh state,
  // every row's logits and final hidden; capture_layers leaves every
  // layer's residual output [T, H] in layer_states.
  Outputs forward(const std::vector<int64_t>& token_ids, bool capture_layers = false);

  // ---- the DFlash2 drafter --------------------------------------------------
  // With a drafter loaded, every run_rows pass also feeds the draft planes:
  // the tap layers' residual streams accumulate through the split fc, get
  // the hidden_norm, and land as each draft layer's paged K/V at the run's
  // positions (rejected verify rows land too and are masked by position —
  // the plane needs no rollback, the same overwrite protocol as the MTP
  // plane). dflash2_draft then runs the block: [bonus, mask x drafts]
  // through the five bidirectional layers over the plane context; each
  // mask rows' top-K go through the reference chained selector walk, which
  // proposes the `drafts()` following tokens.
  bool dflash2_enabled() const { return dflash2_ || dflash1_; }
  // The verifiable drafts: v2's block; v1's 15 capped by the verify budget
  // (dflash_depth, at most 7 — kSpecRows).
  int dflash2_drafts() const {
    if (dflash1_) return std::min(df1cfg_.drafts(), dflash_depth_ == 0 ? 7 : dflash_depth_);
    return dflash2_ ? df_.drafts() : 0;
  }
  // Physical rows written by a redraft, independent of the verify cap.
  int block_draft_rows() const { return dflash1_ ? df1cfg_.query_rows() : df_.query_rows(); }
  bool dflash1_enabled() const { return dflash1_; }
  // False without a draft (pool exhausted / context bound): the caller
  // runs the step without speculation.
  bool dflash2_draft(int req, int64_t bonus, std::vector<int32_t>* drafts);
  // The batched redraft: one stacked block forward for every slot (row-wise
  // GEMMs/norms/convs over S*query_rows rows; the KV appends, sliding-window
  // attention, head, top-K and selector walk stay per-slot at row offsets).
  // drafts[i] is slot i's proposals, empty when that slot has no draft
  // (the same per-slot failure rule as the scalar call). Empty input is
  // an error; a single slot takes the scalar path in the caller.
  void dflash2_draft_batch(const std::vector<int>& reqs,
                           const std::vector<int64_t>& bonuses,
                           std::vector<std::vector<int32_t>>* drafts);
  // The stacked draft batch width (slots per block forward): the verify
  // batch's slot count at the row ceiling.
  int dflash2_batch_slots() const { return df_.batch_slots(); }
  // The captured verify (batched graph capture with a drafter loaded):
  // one static replay per row size — 8 rows for a lone slot, 16 for two,
  // 32 (4 x 8) otherwise. Every slot is padded to a full 8-row block;
  // kernels are row-independent for compute and skip position -1 for every
  // state write (kv appends, GDN recurrence, snapshots), so real rows read
  // back bitwise the eager batch's. Drafts, judge, rollback and redrafts
  // stay eager between replays. Falls back to the eager batch when the row
  // ceiling is below the needed size or a capture breaks (then the engine
  // retries eager every step).
  bool dflash_graph_verify_available() const {
    return dflash2_ && max_decode_rows() >= query_block_rows();
  }
  std::vector<Outputs> session_verify_batch_graph(
      const std::vector<int>& reqs, const std::vector<std::vector<int64_t>>& feds,
      std::vector<int>* offsets = nullptr);
  static constexpr int query_block_rows() { return 8; }
  // The recorded block draft (2026-10-04, the graph engine's block
  // proposal at every world): off the verify's device verdict — the anchor
  // is its next token, the block's positions follow the committed device
  // position — the drafter's block forward, the mask rows' head rows and
  // their top-K (world > 1: every rank's slice list merged through one
  // boundary fold, kernels/dflash2.hpp), the selector walk, the slot's
  // next feed [next, drafts] and the drafts' pinned mirror. Kernel nodes
  // only; the engine's reserve covers the block's rows.
  void session_graph_capture_block_draft(int req, const PickVerdict* verdict);
  // The fixed batch's form: every slot's block off its verify verdict
  // (verdicts[q], slot q == request q) in one stacked block forward; the
  // feeds and mirrors of all slots. An inactive slot's rows run at
  // position -1 (no state write) and its feed is zeroed.
  void session_graph_capture_block_draft_batch(const PickVerdict* verdicts, int requests);
  // The sampled requests' block proposals (the engine's draft rule): the
  // recorded walk draws each draft from the selector's softmax at the
  // request's draft temperature and writes the set it drew from to
  // proposals[req * stride + l] (and the pinned mirror), the next verify's
  // ratio rule; a greedy request keeps the argmax and n = 0. `specs` is the
  // engine's device SampleSpec table (one per slot). Null specs: the argmax
  // walk, no proposals (the default). Before the first capture.
  void dflash2_arm_proposals(const SampleSpec* specs, DraftProposal* proposals,
                             DraftProposal* proposals_host, int stride) {
    df_.arm_proposals(specs, proposals, proposals_host, stride);
  }
  bool dflash2_proposals_armed() const { return df_.proposals_armed(); }
  // The selector's confidence per draft, [max_requests][drafts] acceptance
  // logits written by every walk (recorded or eager) at the slot's row —
  // the scheduled verify depth's input for the block drafter
  // (engine.mtp_schedule with a drafter).
  const float* dflash2_confidence() const { return df_.confidence(); }
  // The mirror the recorded draft of slot `req` publishes (readable once
  // the replay's end event has passed).
  const int32_t* block_drafts_host(int req) const {
    if (dflash1_)
      return df1_mirror_h_ + static_cast<size_t>(req) * static_cast<size_t>(df1cfg_.drafts());
    return df_.drafts_host(req);
  }
  // The mask rows' top-k candidate ids the recorded draft walked
  // ([drafts][block_candidates()] per slot; the engine's acceptance
  // diagnostics rank the verify's correction among them). v1 publishes no
  // candidate tables (top-1 proposals only) — 0 skips the read.
  int block_candidates() const { return dflash1_ ? 0 : df_.cfg().selector_top_k; }
  const int32_t* block_candidates_host(int req) const { return df_.candidates_host(req); }

 private:
  // The shared static padded staging of the two padded verify variants
  // (validate, grow blocks, stage the 8-row blocks, upload, RowRun).
  RowRun df_stage_padded(const std::vector<int>& reqs,
                         const std::vector<std::vector<int64_t>>& feds,
                         int* slots_out, std::vector<int>* offs_out);
  void build_layer_objects(const Qwen35LayerResident& r);
  // MoE MLP (is_moe): routed NVFP4 experts + BF16 shared tail via the
  // shared QwenMoeLayer (decode fast path vs device prefill path).
  QwenMoeWeights moe_view(const QwenMoeResident& m);
  void moe_mlp(const uint16_t* x, uint16_t* out, int tokens, const QwenMoeResident& m,
               cudaStream_t stream, int table_slot, bool decode);
  // The dense GEMM band for this model's own CublasLtGemm: the shared Qwen
  // rule, plus the streaming mma form for 17..128-row decode batches (the
  // drafter's stacked block forwards and the taps of a wide verify read
  // their bf16 weights once instead of once per 4-row GEMV chunk). Other
  // families' instances keep the shared rule.
  void configure_gemm_rows(int rows, bool decode);
  void dense_mlp(const uint16_t* x, uint16_t* out, int tokens, const Qwen35DenseMlpResident& m,
                 cudaStream_t stream, int layer, bool resume = false, bool prefill = false);
  // The L2 weight prefetcher's boundary windows (2026-10-04, #93; the
  // Qwen3.8-Flash-Next idiom): a decode walk's two folds a layer are bus
  // collectives the chain waits on with the memory system idle (138 x
  // ~49 us a step on four Sparks), so a window opened on the prefetcher's
  // side stream just before each fold streams the other side's first
  // weights into L2 in consumption order — the MLP's gate / up / down before
  // the attention fold, the next layer's input projections (or the head)
  // before the MLP fold. Bit-identical on or off: nothing is written.
  void prefetch_add(const void* p, size_t bytes);
  void prefetch_bf16(const uint16_t* w, size_t bytes);
  void prefetch_fp8(const GlmQuantMatrix& q);
  void prefetch_ffn_side(const Qwen35LayerResident& r);
  void prefetch_attention_side(int layer);
  void prefetch_head();
  WeightPrefetcher prefetch_;
  size_t prefetch_window_bytes_ = 0;  // 0 = the prefetcher's window budget (engine.l2_prefetch_boundary_window_mib)
  int walk_rows_ = 1;                 // the rows of the walk in flight (the companions' view)
  // The lm head over `rows` activation rows into F32 logits: the blockwise
  // FP8 head under engine.dense_weights = fp8 (Resident), else the BF16 matmul.
  // draft: a decode draft's read (the MTP head, the DFlash2 block's mask
  // rows) — through the draft head's screen when it is on.
  void head_gemv(const uint16_t* act, float* out, int rows, cudaStream_t stream, bool draft = false);
  // Per-tensor FP8 boot requant of one layer's MLP into PT slot `layer`
  // (num_hidden_layers addresses the MTP draft layer).
  void requant_mlp_pt(int slot, const Qwen35DenseMlpResident& m, cudaStream_t stream);
  // Per-tensor FP8 boot requant of one layer's attention projections: GDN
  // slots are GDN ordinals, Full slots full ordinals (num_full_ addresses
  // the MTP draft layer).
  void requant_gdn_pt(int slot, const QwenGdnResident& w, cudaStream_t stream);
  void requant_full_pt(int slot, const QwenFullAttnResident& w, cudaStream_t stream);
  // The bound layer's per-tensor attention view (disabled when PT is off).
  QwenPtAttnView pt_gdn_view(int layer) const;
  QwenPtAttnView pt_full_view(int layer) const;
  float* gdn_rec(int slot, int ord) const {
    return gdn_rec_base_ + (static_cast<size_t>(slot) * num_gdn_ + ord) * rec_elems_;
  }
  uint16_t* gdn_conv(int slot, int ord) const {
    return gdn_conv_base_ + (static_cast<size_t>(slot) * num_gdn_ + ord) * conv_elems_;
  }

  Qwen35TextConfig cfg_;
  Qwen35LayerStream loader_;
  CublasLtGemm gemm_;
  // The draft head's screen (kernels/draft_head.hpp): built from the served
  // head at construction under engine.draft_head.
  DraftHead draft_head_;
  HeadMatrix draft_head_matrix() const;
  GlmQuantMatrix head_fp8_view_;  // the boot fp8 head copy as a matrix view (head_fp8_ / head_scales_)
  // The bf16 decode weights' 12-bit companions (engine.bf16_weights, #88):
  // the DFlash2 drafter's layers and fc taps, the MTP fc, the bf16 lm head.
  // This family keeps the bf16 bytes beside them under either packed mode
  // (the drafter's arena is one allocation; the stacked redrafts past 16
  // rows read the bf16 form through the streaming mma).
  Bf12Companions bf12_;
  double bf12_s_ = 0;
  void pack_companions();
  QwenGemmWorkspace gw_;
  Qwen35GlobalsResident globals_;
  // The visual tower (weights resident, executed on image prefills only).
  // The cursor retains the engine-owned image vector across yields. Only
  // an executing prefill borrows it into the model; the scope also invalidates
  // the shared staging window when entering and leaving a request's walk.
  std::unique_ptr<QwenVisionEncoder> vision_;
  Outputs session_prefill_with_images(int req, const std::vector<int64_t>& ids,
                                      const std::vector<ImageInput>& images,
                                      const std::vector<int64_t>& boundaries, SnapshotRequest* snap,
                                      bool resume);
  void stage_image_embeddings(int64_t first, int64_t end, const std::vector<ImageInput>* images);
  // Rows [first + shift, first + shift + rows) into dst. `shift` is 0 for
  // the main walk's embedding and 1 for the MTP draft's, whose row at
  // position p embeds token p + 1.
  void apply_image_embeddings(uint16_t* dst, int64_t first, int rows, int shift,
                              const std::vector<ImageInput>* images);
  class ImagePrefillScope;
  const std::vector<ImageInput>* images_for_req(int req) const;
  void store_request_images(int req, const std::vector<ImageInput>* images);
  std::vector<const std::vector<ImageInput>*> prefill_images_per_req_;
  const uint16_t* image_embeddings_ = nullptr;
  int64_t image_window_first_ = 0, image_window_end_ = 0;
  QwenTextConfig qcfg_;  // adapter: the fields the qwen4_exp layer ctors read
  std::unique_ptr<QwenFullAttnLayer> full_;
  std::unique_ptr<QwenGdnLayer> gdn_;
  Qwen35KvPool pool_;
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;
  size_t dense_bridge_bytes_ = 0;
  inline static bool prefill_fp8_per_tensor_ = false;
  inline static bool dflash2_fp8_ = false;
  inline static bool session_capture_layers_ = false;
  inline static bool dflash_verify_graph_ = true;
  inline static bool dflash_draft_batch_ = true;
  inline static int dflash_depth_ = 0;
  inline static bool dense_weights_fp8_ = false;
  inline static bool fp8_attention_ = false;
  inline static bool native_activations_ = false;
  // The native recipe, this instance: on at construction; the cell the layers'
  // workspace copies point at, true while the walk's rows are prefill rows.
  bool native_on_ = false;
  bool native_prefill_cell_ = false;
  // The channel matrices' TMA maps (stable addresses: reserved once at boot).
  std::vector<CUtensorMap> native_tmaps_;
  void build_native_maps();
  // The native recipe's NVFP4 activation scratch: one quantized row set at
  // max_tokens rows of the widest k (I): codes, e4m3 block scales, globals.
  uint8_t* q_codes_ = nullptr;
  // The dense W4A4 TMA form's maps: over q_codes_ per k (H, I; box rows 128)
  // and over every NVFP4 MLP matrix (box rows 256), built at boot.
  CUtensorMap q_tma_[2];
  int q_tma_k_[2] = {0, 0};
  int q_tma_count_ = 0;
  std::vector<CUtensorMap> native_fp4_tmaps_;
  void build_native_fp4_maps();
  const CUtensorMap* q_map_for(int k) const {
    for (int i = 0; i < q_tma_count_; ++i)
      if (q_tma_k_[i] == k) return &q_tma_[i];
    return nullptr;
  }
  uint8_t* q_scales_ = nullptr;
  float* q_gs_ = nullptr;
  // Per-tensor FP8 prefill recipe (engine.prefill_fp8_per_tensor, Resident only):
  // gate/up/down requantized once at boot to E4M3 with one F32 scale each;
  // prefill calls quantize the activation once and run cuBLASLt FP8.
  uint8_t* pt_gate_ = nullptr;  // [slots][I, H] E4M3
  uint8_t* pt_up_ = nullptr;    // [slots][I, H] E4M3
  uint8_t* pt_down_ = nullptr;  // [slots][H, I] E4M3
  float* pt_scales_ = nullptr;  // [slots][gate, up, down] F32 (device)
  uint8_t* pt_act_ = nullptr;   // [M, I] E4M3 activation scratch
  float* pt_act_scales_ = nullptr;  // [H-use, I-use] F32 (device)
  int pt_slots_ = 0;
  bool pt_enabled_ = false;
  // The attention half of the recipe (always with the MLP half; the views
  // stay disabled under the exact path).
  bool pt_attn_enabled_ = false;
  // Per-tensor attention projections (the same recipe):
  // GDN in_proj_qkv [C, H] + in_proj_z [LV, H] + out_proj [H, LV] per GDN
  // ordinal; Full q [QW, H] + k/v [KW, H] + o [H, FH] per full ordinal
  // (the MTP draft layer takes the last full slot).
  uint8_t* pt_gdn_qkv_ = nullptr;
  uint8_t* pt_gdn_z_ = nullptr;
  uint8_t* pt_gdn_o_ = nullptr;
  float* pt_gdn_scales_ = nullptr;  // [gdn_slots][qkv, z, out] F32 (device)
  uint8_t* pt_full_q_ = nullptr;
  uint8_t* pt_full_k_ = nullptr;
  uint8_t* pt_full_v_ = nullptr;
  uint8_t* pt_full_o_ = nullptr;
  float* pt_full_scales_ = nullptr;  // [full_slots][q, k, v, o] F32 (device)
  int pt_gdn_slots_ = 0, pt_full_slots_ = 0;
  int64_t pt_gdn_C_ = 0, pt_gdn_LV_ = 0;        // GDN projection widths
  int64_t pt_full_QW_ = 0, pt_full_KW_ = 0, pt_full_FH_ = 0;  // Full widths
  // Blockwise-FP8 lm head (engine.dense_weights = fp8, Resident only): boot-quantized
  // E4M3 + 128x128 scales; decode rows read half the bytes.
  uint8_t* head_fp8_ = nullptr;  // [V, H] E4M3
  float* head_scales_ = nullptr;  // [ceil(V/128), H/128] F32 (device)
  bool head_fp8_enabled_ = false;
  std::vector<int> pt_gdn_ord_;   // per main-layer index, else -1
  std::vector<int> pt_full_ord_;  // per main-layer index, else -1
  // Activation scratch at max_tokens rows.
  uint16_t *resid_ = nullptr, *x_ = nullptr, *attn_out_ = nullptr, *mlp_out_ = nullptr;
  uint16_t *gate_tmp_ = nullptr, *up_tmp_ = nullptr;  // dense MLP [M, I]
  // The NVFP4 dense path's pre-staged tables (production ldmatrix kernel):
  // one {0, m, 0} segment per row count 1..max_tokens (the grouped
  // launcher's only per-launch table, read not written), plus the
  // [gate,up,down] view tables. Resident stacks harvest every layer's
  // table once at boot (pointers are stable, and the decode graph forbids
  // copy-engine nodes — docs/batched_mtp_graph_stall.md); streaming
  // stacks rebuild one shared layer allocation per load, so dense_mlp
  // re-stages this 3-entry slot per call (the eager tests never capture).
  MoeSegment* fp4_segs_ = nullptr;      // [max_tokens] device
  MoeExpertView* fp4_views_ = nullptr;  // [layers x 3] resident, [3] slot streaming
  int fp4_view_layers_ = 0;             // 0 = slot mode (streaming)
  // MoE MLP (is_moe): the routed chain over this rank's expert slice.
  GlmMoeConfig moe_cfg_ = {};
  std::unique_ptr<QwenMoeLayer> moe_;
  // MTP draft scratch at max_tokens rows (null when MTP is off): embed rows
  // and their norm, the gathered/gated main hidden and its norm, the
  // [M, 2H] concat, the draft residual, its norm (the chain rows), and the
  // gathered window rows decode drafts read.
  uint16_t *mtp_e_ = nullptr, *mtp_en_ = nullptr, *mtp_hn_ = nullptr, *mtp_hin_ = nullptr;
  uint16_t *mtp_cat_ = nullptr, *mtp_r_ = nullptr, *mtp_h_ = nullptr;
  bool mtp_ = false;
  // The DFlash2 drafter (null unless a drafter directory was given): the
  // The DFlash2 block drafter (models/dflash2/drafter.hpp, shared with the
  // Flash-Next model): the family hosts the component over its own pool
  // planes (view(num_full_ + l)), head dispatch (the draft screen) and gemm
  // row envelope; the target-side taps (run_rows) read the layer outputs
  // into the component's feature accumulator (df_.acc / tap_gemm_out).
  bool dflash2_ = false;
  DFlash2Drafter df_;
  // The DFlash v1 drafter (null unless a v1 directory was given):
  // DFlashConfig + weights, tap map, one block of reusable scratch
  // (query_rows = 16 rows; C1 scope — batching and capture stay open).
  bool dflash1_ = false;
  DFlashConfig df1cfg_;
  DFlashWeights df1w_;
  std::vector<int> df1_tap_;  // per main-layer index: the fc slice, else -1
  uint16_t *df1_resid_ = nullptr, *df1_x_ = nullptr;
  uint16_t *df1_q_ = nullptr, *df1_k_ = nullptr, *df1_v_ = nullptr;  // roped q/k/v rows
  uint16_t *df1_attn_ = nullptr, *df1_o_ = nullptr, *df1_mlp_ = nullptr;
  float* df1_attn_part_ = nullptr;  // the split-key block attention's partials (QR rows)
  uint16_t *df1_gate_ = nullptr, *df1_up_ = nullptr;  // [query_rows, draft I]
  uint16_t* df1_h_ = nullptr;       // [query_rows, H] the draft's final norm rows
  float* df1_logits_ = nullptr;     // [query_rows, V] F32 (the block rows' head rows)
  void* df1_topk_ws_ = nullptr;     // the top-1 chunk partials
  size_t df1_topk_ws_bytes_ = 0;
  int32_t* df1_ids_ = nullptr;   // [query_rows, 16] top-16 per row (the kernel's only K; row's draft is column 0)
  float* df1_sc_ = nullptr;      // [query_rows, 16]
  int32_t* df1_zero_ = nullptr;  // [query_rows] zeros (the single-slot append view)
  int64_t* df1_pos_ = nullptr;   // [query_rows] device
  int64_t* df1_tokens_ = nullptr;  // [query_rows] device
  int64_t* df1_io64_h_ = nullptr;  // pinned: query_rows positions then tokens
  int32_t* df1_tok_ = nullptr;     // [drafts] device (the captured proposal rows)
  int32_t* df1_tok_h_ = nullptr;   // pinned [drafts]: the proposed drafts
  int32_t* df1_mirror_h_ = nullptr;  // pinned [max_requests, drafts]: the recorded drafts
  float* df1_inv_freq_ = nullptr;  // [head_dim/2] F32 rope table
  // The v1 tap accumulator (v1-only: the v2 drafter's are the component's).
  float* df1_acc_ = nullptr;       // [max_tokens, H] F32 (the sum_t tap @ fc_t^T)
  float* df1_t32_ = nullptr;       // [max_tokens, H] F32 (one tap GEMM's out)
  uint16_t* df1_table_ = nullptr;  // world > 1: the v1 top-K fold buffer
  // The v1 block forward + mask proposal (eager, one slot): stages
  // [bonus, mask x D], runs the six layers over the draft planes, and
  // top-1s each mask row off the shared head.
  bool dflash1_draft(int req, int64_t bonus, std::vector<int32_t>* drafts);
  void df1_block_layers(int req, cudaStream_t stream, bool capture);
  void df1_block_select(bool capture);
  void df1_capture_draft(int req, const PickVerdict* verdict, int64_t* feed);
  void df1_store_features(int T, const int32_t* d_req, const int64_t* d_pos);
  // The captured verify's replay state (one static 32-row graph): the
  // pool tables pointer the capture baked in (a mismatch means the pool
  // grew — drop and recapture), and the breakage latch (a failed capture
  // retries eager every step instead of throwing the server over).
  GraphCache df_verify_graph_;
  const int32_t* df_verify_tables_ = nullptr;
  bool df_verify_broken_ = false;
  // The capture-time step fields for the component (the session's graph
  // family, read at every recorded-proposal call).
  DFlash2CaptureCtx df_capture_ctx() const;
  // Model-owned GDN state: [max_requests][num_gdn][elems], plus the
  // verify's per-row CONV snapshots ([max_decode_rows][num_gdn][elems]) the
  // speculative rollback reads (engine/session_model.hpp's commit). The
  // recurrent state takes the checkpoint-and-replay form (kernels/kda.hpp
  // KdaReplay; 2026-10-05, the Flash-Next model's since 2026-09-29): the
  // live buffer holds the checkpoint before the last pass's rows, each
  // pass saves its rows as their inputs (~21 KB a row a layer against the
  // 3.1 MB state a snapshot wrote — 1.2 GB a step at one slot, 9.7 GB at
  // the 64-row batch) and the next pass replays the accepted ones; the
  // replayed state is bitwise the snapshot it stands in for.
  float* gdn_rec_base_ = nullptr;
  uint16_t* gdn_conv_base_ = nullptr;
  float* spec_rec_ = nullptr;             // null under the replay form
  bool gdn_replay_ = false;
  uint16_t* gdn_replay_in_ = nullptr;     // [R][num_gdn][rows_cap][row]: the rows the next pass replays
  uint16_t* gdn_replay_save_ = nullptr;   // [R][num_gdn][rows_cap][row]: this pass's rows (the commit copies them over `in`)
  int32_t* gdn_pending_ = nullptr;        // [R]: rows to replay (the commit writes the accepted count)
  int64_t gdn_replay_row_elems_ = 0;      // the post-conv q|k|v row + a_raw + beta_raw, padded to 16 bytes
  int gdn_replay_rows_cap_ = 0;
  int gdn_replay_rows_used_ = 1;          // rows per request of the last decode walk (the commit's copy)
  KdaReplay gdn_replay_view(int req, int ordinal, bool batched) const;
  size_t gdn_replay_block_elems() const;  // one request's saved rows, every GDN layer
  // The recurrent state as it stands (the pending rows applied): in place
  // (dst null; the pending count cleared) or into dst after `rows` rows.
  void materialize_gdn(int req, float* dst = nullptr, int rows = -1);
  uint16_t* spec_conv_ = nullptr;
  int num_gdn_ = 0, num_full_ = 0;
  int64_t rec_elems_ = 0, conv_elems_ = 0;
  int64_t wave_rows_ = 0;  // prefill attention grid wave (prefill_wave_rows)
};

}  // namespace dgpp
