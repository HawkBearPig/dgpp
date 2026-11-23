#pragma once
// Inkling 2 (looped) on the session core (docs/inkling2_mini_plan.md): the
// family's visit walk, its state and its memory plan.
//
// The model has no residual stream. A walk keeps a stack of completed block
// states and one partial state per row. Every sublayer input is a mix of the
// stack. Layers [core_start, core_end) run num_passes times from the state
// saved at core_start, and a pass after the first blends messages read from
// the previous pass into its sublayer inputs. A visit is one execution of a
// layer, with its own K/V plane and convolution rings.
//
// State per request slot:
//   its row of the paged K/V pool, one plane per visit (models/mimo/
//   kv_pool.hpp at 128 / 128, 64-token blocks, the local visits read through
//   their window);
//   its convolution rings: the inputs of the four short convolutions of
//   every visit at the last kInk2RingSlots positions.
// A rejected speculative row leaves K/V rows and ring slots past the
// position that are rewritten before anything reads them, so rollback moves
// the position alone. A prefix snapshot is the block list plus the three
// ring slots before its position.
//
// TP: tp_world > 1 loads this rank's slices and folds the attention and MLP
// outputs of every visit through `boundary`. Everything after a fold is
// replicated and bitwise equal across ranks.
//
// The draft model (`mtp`, models/inkling2/draft.hpp) rides the shared block
// drafter. A run copies the states the draft reads (the partial state at the
// entry of each tapped visit, and the final-norm output), the drafter turns
// them into context keys and values in one more K/V plane per draft layer,
// and the recurrent head's feature of every position lives in the ring's
// slots beside the convolution inputs, so rollback and prefix snapshots
// cover it without a protocol of their own.
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
#include "kernels/gemm.hpp"
#include "kernels/glm_spec.hpp"
#include "kernels/inkling2.hpp"
#include "kernels/latent_format.hpp"
#include "models/dflash2/drafter.hpp"
#include "models/inkling2/config.hpp"
#include "models/inkling2/layers.hpp"
#include "models/inkling2/loader.hpp"
#include "models/inkling2/vision.hpp"
#include "models/mimo/kv_pool.hpp"

namespace dgpp {

class Inkling2Model : public SessionModel<Inkling2Model> {
 public:
  // Several cold prompts as the spans of one walk: rows carry their own
  // request ids and positions.
  int64_t prefill_group_span_limit() const { return max_tokens_; }
  using Base = SessionModel<Inkling2Model>;
  static constexpr bool kPrefillChunkGrid = false;
  using Outputs = Base::Outputs;
  using SessionSnapshotMeta = Base::SessionSnapshotMeta;
  using SnapshotRequest = Base::SnapshotRequest;
  using RowRun = Base::RowRun;

  // max_tokens bounds a walk's rows. max_cache_tokens is the paged pool's
  // capacity in tokens, shared by every slot. decode_rows is the fixed
  // decode batch's row ceiling (0: kDecodeRows). native_activations selects
  // the checkpoint's quantized expert inputs (engine.activations).
  Inkling2Model(const Inkling2Config& cfg, const std::string& checkpoint_dir, int max_tokens,
                int64_t max_cache_tokens, Inkling2Residency residency = Inkling2Residency::Streaming,
                BoundaryReducer* boundary = nullptr, int tp_rank = 0, int tp_world = 1, int max_requests = 1,
                bool mtp = false, int decode_rows = 0, LatentFormat kv_format = LatentFormat::kBf16,
                bool serving_logits = false, bool native_activations = false);
  ~Inkling2Model();
  Inkling2Model(const Inkling2Model&) = delete;
  Inkling2Model& operator=(const Inkling2Model&) = delete;

  using MemoryPlan = dgpp::MemoryPlan;
  static MemoryPlan plan_memory(const Inkling2Config& cfg, int max_tokens, int64_t max_cache_tokens,
                                int tp_rank = 0, int tp_world = 1,
                                Inkling2Residency residency = Inkling2Residency::Streaming, int max_requests = 1,
                                bool mtp = false, int decode_rows = 0, LatentFormat kv_format = LatentFormat::kBf16,
                                bool serving_logits = false, bool native_activations = false);

  // The cold diagnostic forward: one request on slot 0 (which must be
  // closed), every row's logits. capture_layers: the partial state after
  // every visit in layer_states, the attention contribution of every visit
  // in visit_attention() and the routes in route_ids.
  Outputs forward(const std::vector<int64_t>& token_ids, bool capture_layers = false);
  const std::vector<std::vector<uint16_t>>& visit_attention() const { return visit_attn_; }
  // The experts' input in every visit of the last capturing forward.
  const std::vector<std::vector<uint16_t>>& visit_mlp_input() const { return visit_mlp_in_; }
  // Tests: a decode step also records its rows' per-visit states, attention
  // contributions, expert inputs and routes, as a capturing forward does.
  void set_decode_row_traces(bool on) { decode_row_traces_ = on; }

  // Eight slots of the draft's eight-row block in one pass.
  static constexpr int decode_rows_cap() { return 64; }
  static constexpr int prefill_chunk_tokens() { return 4096; }
  static constexpr int kv_block_tokens_static() { return kBlockTokens; }
  static size_t session_snapshot_bytes(const Inkling2Config& cfg, int tp_world, bool mtp);
  static size_t kv_block_bytes_static(const Inkling2Config& cfg, int tp_rank, int tp_world, LatentFormat kv_format,
                                      bool mtp = false);
  using Base::session_snapshot_bytes;

  const Inkling2Config& config() const { return cfg_; }
  const MimoKvPool& kv_pool() const { return pool_; }
  LatentFormat kv_format() const { return pool_.shape().format; }
  bool native_activations() const { return native_; }

  // ---- image inputs (models/inkling2/image_prefill.cpp) ------------------------
  // A checkpoint with a vision tower serves images: the tower's rows replace
  // the normed embeddings of an image's placeholder tokens.
  bool supports_images() const { return vision_ != nullptr; }
  ImageTokens image_tokens() const { return vision_ ? ImageTokens{-1, cfg_.image_token_id, -1} : ImageTokens{}; }
  uint64_t vision_digest() const;
  struct PrefillCursor : Base::PrefillCursor {
    const std::vector<ImageInput>* images = nullptr;
  };
  Outputs session_prefill_images(int req, const std::vector<int64_t>& ids, const std::vector<ImageInput>& images);
  Outputs session_prefill_images(int req, const std::vector<int64_t>& ids, const std::vector<ImageInput>& images,
                                 const std::vector<int64_t>& boundaries, SnapshotRequest* snap);
  Outputs session_prefill_resume_images(int req, const std::vector<int64_t>& ids,
                                        const std::vector<ImageInput>& images,
                                        const std::vector<int64_t>& boundaries, SnapshotRequest* snap);
  PrefillCursor session_prefill_begin(int req, const std::vector<int64_t>& prompt, int64_t reserve_tokens,
                                      int64_t chunk_tokens, const std::vector<int64_t>& boundaries = {},
                                      SnapshotRequest* snap = nullptr, int64_t attach_position = 0,
                                      const std::vector<ImageInput>* images = nullptr);
  bool session_prefill_advance(PrefillCursor& cursor, int64_t chunk_tokens = 0);
  std::vector<bool> session_prefill_advance_group(const std::vector<PrefillCursor*>& cursors,
                                                  const std::vector<int64_t>& budgets) {
    return Base::session_prefill_advance_group(cursors, budgets);
  }
  void validate_mixed_prefill_cursor(const PrefillCursor& cursor) const {
    if (cursor.images && !cursor.images->empty())
      throw std::invalid_argument("Inkling 2: image spans require their own prefill walk");
  }

  // ---- the session core's hooks (engine/session_model.hpp) --------------------
  Outputs run_rows(const RowRun& run);
  void reset_slot_state(int req);
  GlmSpecSegments spec_segments(int, int = 0) const { return GlmSpecSegments{}; }
  size_t snapshot_state_bytes() const { return snapshot_bytes_; }
  size_t draft_state_bytes() const { return 0; }
  void write_state_snapshot(int req, uint8_t* dst, int spec_row);
  void write_draft_snapshot(int, uint8_t*, bool, int64_t) {}
  void read_state_snapshot(int req, const uint8_t* src);
  void read_draft_snapshot(int, const uint8_t*) {}
  bool has_pool() const { return true; }
  MimoKvPool& pool() { return pool_; }
  const MimoKvPool& pool() const { return pool_; }
  void graph_prepare();
  [[noreturn]] void mtp_run_rows(int, const int64_t*, int64_t, int, bool, bool, int, int, int = 0) {
    throw std::logic_error("Inkling2Model: the draft model is a block drafter, not a chained draft block");
  }
  void snapshot_draft_state(int) {}
  void restore_draft_state(int) {}
  // The widest pass of one request: its pending token and seven proposals.
  static constexpr int kRequestRows = 8;
  static constexpr bool kDraftChain = true;
  static constexpr bool kBatchedDraftChain = true;
  const uint16_t* draft_hidden_rows() const { return nullptr; }
  void snapshot_chain_state(int) {}
  void restore_chain_state(int) {}

  // ---- the block drafter's hooks (the engines find them by name) ----------------
  const DFlash2Drafter& drafter() const { return df_; }
  DFlash2Drafter& drafter() { return df_; }
  bool dflash2_enabled() const { return df_.on(); }
  int dflash_depth() const { return 0; }
  int dflash2_drafts() const { return df_.on() ? df_.drafts() : 0; }
  int block_draft_rows() const { return df_.on() ? df_.query_rows() : 0; }
  int dflash2_batch_slots() const { return df_.batch_slots(); }
  bool dflash_graph_verify_available() const { return df_.on() && max_decode_rows() >= df_.drafts() + 1; }
  bool dflash2_draft(int req, int64_t bonus, std::vector<int32_t>* drafts) { return df_.draft(req, bonus, drafts); }
  void dflash2_draft_batch(const std::vector<int>& reqs, const std::vector<int64_t>& bonuses,
                           std::vector<std::vector<int32_t>>* drafts) {
    df_.draft_batch(reqs, bonuses, drafts);
  }
  void session_graph_capture_block_draft(int req, const PickVerdict* v) {
    df_.capture_block_draft(req, v, df_capture_ctx());
  }
  void session_graph_capture_block_draft_batch(const PickVerdict* v, int requests) {
    df_.capture_block_draft_batch(v, requests, df_capture_ctx());
  }
  const int32_t* block_drafts_host(int req) const { return df_.drafts_host(req); }
  int block_candidates() const { return df_.cfg().selector_top_k; }
  const int32_t* block_candidates_host(int req) const { return df_.candidates_host(req); }
  void dflash2_arm_proposals(const SampleSpec* specs, DraftProposal* proposals, DraftProposal* host, int stride) {
    df_.arm_proposals(specs, proposals, host, stride);
  }
  bool dflash2_proposals_armed() const { return df_.proposals_armed(); }
  const float* dflash2_confidence() const { return df_.confidence(); }
  // The recurrent head's stop floor, one cell per request (the scheduled
  // engine's threshold: models/dflash2/drafter.hpp).
  float* dflash2_stop_floor() { return df_.on() ? df_.stop_floor() : nullptr; }

 private:
  static constexpr int kBlockTokens = 64;

  struct Walk {
    const RowRun* run = nullptr;
    const int32_t* req_ids = nullptr;
    const int64_t* pos = nullptr;
    int T = 0;
    bool capture = false;
    int num_blocks = 0;      // completed blocks
    int prelude_blocks = 0;  // completed blocks at the core's entry
    int parity = 0;          // the stack set the core-range slots read and write
  };
  // The stack slot of block index n: the core's slots alternate between two
  // sets by pass parity, so a pass reads the previous pass's states while it
  // writes its own.
  uint16_t* slot(int n, int parity) const;
  template <class Fn>
  void groups(const Walk& w, Fn&& fn) const;
  void visit(int v, Walk& w, Outputs* capture_out);
  void blend(const float* dest, const float* message, const Inkling2ControllerResident& c, const uint16_t* norm_w,
             uint16_t* out, int rows);
  Inkling2Ring ring_of(int visit) const;
  uint16_t* stage(uint16_t* fallback, int T, int width, bool capture);
  void fold(uint16_t* buf, int T, int width, bool capture);
  bool packed_head_wide(const uint16_t* hidden, float* out, int rows);
  void pack_layer_companions(int layer, const Inkling2LayerResident& r);
  void finish_companions();
  void decode_gemm(bool decode);
  DFlash2CaptureCtx df_capture_ctx() const;
  void init_drafter(const std::string& checkpoint_dir, int tp_rank, int tp_world);
  // The draft model's projections run on a GEMM object of their own: the
  // target's object holds released packed weights, which rule out the
  // tensor-core form the draft's rows take.
  CublasLtGemm draft_gemm_;
  DFlash2Drafter df_;
  // Image inputs: the tower and the images a prefill call borrows.
  class ImagePrefillScope;
  std::unique_ptr<Inkling2VisionEncoder> vision_;
  std::vector<const std::vector<ImageInput>*> request_images_;
  void validate_images(const std::vector<int64_t>& ids, const std::vector<ImageInput>& images,
                       int64_t start = 0) const;
  void apply_image_embeddings(uint16_t* dst, const RowRun& run);

  Inkling2Config cfg_;
  Inkling2LayerStream loader_;
  CublasLtGemm gemm_;
  Bf12Companions bf12_;
  static constexpr int kBf12ExpandSlots = 1;
  bool bf12_built_ = false;
  double bf12_s_ = 0.0;
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;
  Inkling2GemmWorkspace gw_;
  Inkling2GlobalsResident globals_;
  bool native_ = false;
  int n_split_ = Inkling2AttentionLayer::default_decode_splits();
  int visits_ = 0;
  int bank_rows_ = 0;
  int max_blocks_ = 0;

  std::unique_ptr<Inkling2AttentionLayer> attn_;
  std::unique_ptr<Inkling2MoeLayer> moe_;

  MimoKvPool pool_;            // one plane per visit
  uint16_t* ring_ = nullptr;   // [requests][kInk2RingSlots][slot_stride]
  int64_t ring_slot_stride_ = 0;
  int64_t ring_visit_width_ = 0;
  int64_t ring_draft_offset_ = 0;  // the draft's feature row inside a slot (with the draft model)
  size_t snapshot_bytes_ = 0;

  // Activations [M rows].
  uint16_t* stack_ = nullptr;  // [max_blocks + 1 + bank_rows][M, H]
  uint16_t* x_[3] = {nullptr, nullptr, nullptr};  // the normed attention inputs (q, k, v)
  uint16_t* xm_ = nullptr;     // the normed MLP input
  uint16_t* y_ = nullptr;      // a sublayer's output (the fold's fallback)
  uint16_t* a_ = nullptr;      // the attention contribution after its convolution
  float* dest_[3] = {nullptr, nullptr, nullptr};          // FP32 mixes a message blends into
  float* msg_[4] = {nullptr, nullptr, nullptr, nullptr};  // FP32 messages
  uint16_t* pair_ = nullptr;   // [M, 2 H]
  uint16_t* ctrl_h_ = nullptr; // [M, gate_rank]
  float* gain_forget_ = nullptr;  // [M, 2 H]
  std::vector<std::vector<uint16_t>> visit_attn_;
  std::vector<std::vector<uint16_t>> visit_mlp_in_;
  bool decode_row_traces_ = false;
};

}  // namespace dgpp
