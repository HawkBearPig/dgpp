#pragma once
#include <memory>

#include "engine/boundary_reducer.hpp"
#include "engine/memory_plan.hpp"
#include "engine/session_model.hpp"
#include "kernels/bf12_companions.hpp"
#include "kernels/gemm.hpp"
#include "kernels/glm_spec.hpp"
#include "kernels/latent_format.hpp"
#include "models/minimax/layers.hpp"

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
               bool serving_logits = false);
  ~MinimaxModel();
  static MemoryPlan plan_memory(const MinimaxConfig&, int max_tokens, int64_t cache_tokens,
                                int rank = 0, int world = 1,
                                LoaderResidency = LoaderResidency::Streaming, int slots = 1,
                                bool mtp = false, int decode_rows = 0,
                                LatentFormat = LatentFormat::kBf16, bool serving_logits = false);
  static constexpr int decode_rows_cap() { return 32; }
  static constexpr int prefill_chunk_tokens() { return 512; }
  static constexpr int kv_block_tokens_static() { return 128; }
  static size_t session_snapshot_bytes(const MinimaxConfig&, int, bool) { return 16; }
  using Base::session_snapshot_bytes;
  static size_t kv_block_bytes_static(const MinimaxConfig&, int rank, int world, LatentFormat);
  int64_t prefill_group_span_limit() const { return max_tokens_; }
  const MinimaxConfig& config() const { return cfg_; }
  const MinimaxKvPool& kv_pool() const { return pool_; }
  LatentFormat kv_format() const { return LatentFormat::kBf16; }
  void set_decode_row_traces(bool on) { decode_row_traces_ = on; }
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
    refuse_mtp("MiniMax DSpark is not enabled");
  }
  void snapshot_draft_state(int) {}
  void restore_draft_state(int) {}
  static constexpr bool kDraftChain = false;
  static constexpr bool kBatchedDraftChain = false;
  const uint16_t* draft_hidden_rows() const { return nullptr; }
  void snapshot_chain_state(int) {}
  void restore_chain_state(int) {}

 private:
  uint16_t* stage(int rows, bool capture);
  void fold(uint16_t*, int rows, bool capture);
  void norm(uint16_t* resid, const uint16_t* add, const uint16_t* weight, uint16_t* out, int rows);
  MinimaxConfig cfg_;
  MinimaxLayerStream loader_;
  MinimaxGlobalsResident globals_;
  CublasLtGemm gemm_;
  Bf12Companions bf12_;
  bool bf12_built_ = false, decode_row_traces_ = false;
  MinimaxKvPool pool_;
  std::unique_ptr<MinimaxAttentionLayer> attn_;
  std::unique_ptr<MinimaxMlp> dense_;
  std::unique_ptr<MinimaxMoeLayer> moe_;
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;
  uint16_t *resid_ = nullptr, *x_ = nullptr, *y_ = nullptr;
  int32_t* trace_ids_ = nullptr;
  float* trace_weights_ = nullptr;
};
}  // namespace dgpp
