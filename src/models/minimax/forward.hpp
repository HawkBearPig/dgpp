#pragma once
#include <functional>
#include <memory>
#include <string_view>

#include "engine/boundary_reducer.hpp"
#include "engine/memory_plan.hpp"
#include "engine/session_model.hpp"
#include "kernels/bf12_companions.hpp"
#include "kernels/gemm.hpp"
#include "kernels/glm_spec.hpp"
#include "kernels/latent_format.hpp"
#include "models/dflash2/drafter.hpp"
#include "models/minimax/layers.hpp"
#include "models/minimax/vision.hpp"

namespace dgpp {
class MinimaxModel : public SessionModel<MinimaxModel> {
 public:
  using Base = SessionModel<MinimaxModel>;
  using Outputs = Base::Outputs;
  using SessionSnapshotMeta = Base::SessionSnapshotMeta;
  using SnapshotRequest = Base::SnapshotRequest;
  using RowRun = Base::RowRun;
  using MemoryPlan = dgpp::MemoryPlan;
  static constexpr bool kPrefillChunkGrid = false;
  MinimaxModel(const MinimaxConfig&, const std::string& checkpoint, int max_tokens,
               int64_t cache_tokens, LoaderResidency = LoaderResidency::Streaming,
               BoundaryReducer* = nullptr, int rank = 0, int world = 1, int slots = 1,
               bool mtp = false, int decode_rows = 0, LatentFormat = LatentFormat::kBf16,
               bool serving_logits = false, bool native_activations = false);
  ~MinimaxModel();
  static MemoryPlan plan_memory(const MinimaxConfig&, int max_tokens, int64_t cache_tokens,
                                int rank = 0, int world = 1,
                                LoaderResidency = LoaderResidency::Streaming, int slots = 1,
                                bool mtp = false, int decode_rows = 0,
                                LatentFormat = LatentFormat::kBf16, bool serving_logits = false,
                                bool native_activations = false);
  static constexpr int decode_rows_cap() { return 64; }
  static constexpr int prefill_chunk_tokens() { return 512; }
  static constexpr int kv_block_tokens_static() { return 128; }
  static size_t session_snapshot_bytes(const MinimaxConfig&, int, bool) { return 16; }
  using Base::session_snapshot_bytes;
  static size_t kv_block_bytes_static(const MinimaxConfig&, int rank, int world, LatentFormat,
                                      bool mtp = false);
  int64_t prefill_group_span_limit() const { return max_tokens_; }
  const MinimaxConfig& config() const { return cfg_; }
  bool supports_images() const { return vision_ != nullptr; }
  bool supports_video() const { return vision_ != nullptr; }
  ImageTokens image_tokens() const { return cfg_.vision ? cfg_.vision->tokens : ImageTokens{}; }
  uint64_t vision_digest() const { return vision_ ? vision_->digest() : 0; }
  struct PrefillCursor : Base::PrefillCursor {
    const std::vector<ImageInput>* images = nullptr;
  };
  Outputs session_prefill_images(int req, const std::vector<int64_t>& ids,
                                 const std::vector<ImageInput>& images);
  Outputs session_prefill_images(int req, const std::vector<int64_t>& ids,
                                 const std::vector<ImageInput>& images,
                                 const std::vector<int64_t>& boundaries, SnapshotRequest* snap);
  Outputs session_prefill_resume_images(int req, const std::vector<int64_t>& ids,
                                        const std::vector<ImageInput>& images,
                                        const std::vector<int64_t>& boundaries,
                                        SnapshotRequest* snap);
  PrefillCursor session_prefill_begin(int req, const std::vector<int64_t>& prompt,
                                      int64_t reserve_tokens, int64_t chunk_tokens,
                                      const std::vector<int64_t>& boundaries = {},
                                      SnapshotRequest* snap = nullptr, int64_t attach_position = 0,
                                      const std::vector<ImageInput>* images = nullptr);
  bool session_prefill_advance(PrefillCursor&, int64_t chunk_tokens = 0);
  std::vector<bool> session_prefill_advance_group(const std::vector<PrefillCursor*>& cursors,
                                                  const std::vector<int64_t>& budgets) {
    return Base::session_prefill_advance_group(cursors, budgets);
  }
  void validate_mixed_prefill_cursor(const PrefillCursor& cursor) const {
    if (cursor.images && !cursor.images->empty())
      throw std::invalid_argument("MiniMax: image spans require their own prefill walk");
  }
  const MinimaxKvPool& kv_pool() const { return pool_; }
  LatentFormat kv_format() const { return pool_.shape().format; }
  void set_decode_row_traces(bool on) { decode_row_traces_ = on; }
  // Optional diagnostics at replicated BF16 sublayer boundaries. The
  // observer receives device memory and orders any reads on the stream.
  using Observer = std::function<void(int, std::string_view, const uint16_t*, int, cudaStream_t)>;
  void set_observer(Observer observer) { observer_ = std::move(observer); }
  Outputs forward(const std::vector<int64_t>& ids, bool capture_layers = false);
  Outputs run_rows(const RowRun&);
  void reset_slot_state(int req);
  GlmSpecSegments spec_segments(int, int = 0) const { return {}; }
  size_t snapshot_state_bytes() const { return 16; }
  size_t draft_state_bytes() const { return 0; }
  void write_state_snapshot(int, uint8_t*, int);
  void write_draft_snapshot(int, uint8_t*, bool, int64_t) {}
  void read_state_snapshot(int, const uint8_t*) {}
  void read_draft_snapshot(int, const uint8_t*) {}
  bool has_pool() const { return true; }
  MinimaxKvPool& pool() { return pool_; }
  const MinimaxKvPool& pool() const { return pool_; }
  void graph_prepare();
  [[noreturn]] void mtp_run_rows(int, const int64_t*, int64_t, int, bool, bool, int, int, int = 0) {
    refuse_mtp("MiniMax DSpark requires the block-draft interface");
  }
  void snapshot_draft_state(int) {}
  void restore_draft_state(int) {}
  static constexpr bool kDraftChain = true;
  static constexpr bool kBatchedDraftChain = true;
  const uint16_t* draft_hidden_rows() const { return nullptr; }
  void snapshot_chain_state(int) {}
  void restore_chain_state(int) {}

  const DFlash2Drafter& dspark_drafter() const { return df_; }
  bool dflash2_enabled() const { return df_.on(); }
  int dflash_depth() const { return 0; }
  int dflash2_drafts() const { return df_.on() ? df_.drafts() : 0; }
  int block_draft_rows() const { return df_.on() ? df_.query_rows() : 0; }
  int dflash2_batch_slots() const { return df_.batch_slots(); }
  bool dflash_graph_verify_available() const {
    return df_.on() && max_decode_rows() >= df_.drafts() + 1;
  }
  bool dflash2_draft(int req, int64_t bonus, std::vector<int32_t>* drafts) {
    return df_.draft(req, bonus, drafts);
  }
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
  void dflash2_arm_proposals(const SampleSpec* specs, DraftProposal* proposals, DraftProposal* host,
                             int stride) {
    df_.arm_proposals(specs, proposals, host, stride);
  }
  bool dflash2_proposals_armed() const { return df_.proposals_armed(); }
  const float* dflash2_confidence() const { return df_.confidence(); }

 private:
  class ImagePrefillScope;
  std::unique_ptr<MinimaxVisionEncoder> vision_;
  std::vector<const std::vector<ImageInput>*> request_images_;
  void validate_images(const std::vector<int64_t>& ids, const std::vector<ImageInput>& images,
                       int64_t start = 0) const;
  void apply_image_embeddings(uint16_t* dst, const RowRun&);
  DFlash2CaptureCtx df_capture_ctx() const;
  DFlash2Drafter df_;
  uint16_t* stage(int rows, bool capture);
  void fold(uint16_t*, int rows, bool capture);
  void norm(uint16_t* resid, const uint16_t* add, const uint16_t* weight, uint16_t* out, int rows);
  void accumulate_draft_tap(int completed_layer, int rows);
  MinimaxConfig cfg_;
  MinimaxLayerStream loader_;
  MinimaxGlobalsResident globals_;
  CublasLtGemm gemm_;
  Bf12Companions bf12_;
  bool bf12_built_ = false, decode_row_traces_ = false;
  void project_head(const uint16_t* hidden, float* out, int rows, bool decode);
  MinimaxKvPool pool_;
  std::unique_ptr<MinimaxAttentionLayer> attn_;
  std::unique_ptr<MinimaxMlp> dense_;
  std::unique_ptr<MinimaxMoeLayer> moe_;
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;
  uint16_t *resid_ = nullptr, *x_ = nullptr, *y_ = nullptr;
  int32_t* trace_ids_ = nullptr;
  float* trace_weights_ = nullptr;
  Observer observer_;
};
}  // namespace dgpp
