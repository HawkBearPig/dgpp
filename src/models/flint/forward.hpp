#pragma once
// Flint model with resident or streaming weights on the shared session
// core (2026-10-22, docs/flint_plan.md §3). The layer walk follows the
// release's FlintDecoderLayer:
//
//   h = embed(x)
//   per layer: h += attn(input_norm(h))         (GQA 32 q / 4 kv heads of 128,
//                                                the fused BF16 projection, the
//                                                per-head one-rounding q / k
//                                                norms, RoPE over the whole head
//                                                — none on the no-RoPE full
//                                                layers — a 513-token window on
//                                                the sliding layers, a learned
//                                                per-head sink everywhere)
//              h += mlp(post_norm(h))           (the BF16 SwiGLU MLP on the dense
//                                                layers; the sigmoid-routed
//                                                top-6 MoE in the layer's fp8 /
//                                                NVFP4 per-projection formats
//                                                plus the ungated BF16 shared
//                                                expert on the others)
//   logits = lm_head(final_norm(h))
//
// Every norm rounds once (D4). One row walk (run_rows) serves every entry
// point — the cold diagnostic forward, a session's prefill chunks, the
// eager decode rows and the captured decode graphs — the
// engine/session_model.hpp core owns everything around it. No draft head:
// the family has no MTP block (engine.mtp is refused).
//
// State per request slot: its row of the paged K/V pool (models/mimo/
// kv_pool.hpp at 128 / 128, 64-token blocks, every layer's K/V — the
// window layers too, read through their window). Nothing else: the
// prefix snapshot is the block list.
//
// TP (§2): `tp_world` > 1 loads this rank's slices (the fused projection's
// heads, o_proj's columns, every expert's intermediate slice, the dense
// and shared MLP slices, the lm-head vocab slice) and folds the attention
// and MLP block outputs per layer through `boundary` (the canonical
// rank-order sum). The router and the norms are replicated; every rank's
// residual is bitwise the others'.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "engine/boundary_reducer.hpp"
#include "engine/decode_outputs.hpp"
#include "engine/memory_plan.hpp"
#include "engine/session_model.hpp"
#include "kernels/bf12_companions.hpp"
#include "kernels/gemm.hpp"
#include "kernels/glm_spec.hpp"
#include "kernels/l2_prefetch.hpp"
#include "kernels/latent_format.hpp"
#include "models/flint/config.hpp"
#include "models/flint/layers.hpp"
#include "models/flint/loader.hpp"
#include "models/mimo/kv_pool.hpp"
#include "models/qwen/moe_layer.hpp"

namespace dgpp {

class FlintModel : public SessionModel<FlintModel> {
 public:
  // Several cold prompts as the spans of one walk (session_prefill_group):
  // the attention rows carry their own request ids and positions, so a
  // span attends to its own request's cache only; every span within
  // max_tokens.
  int64_t prefill_group_span_limit() const { return max_tokens_; }
  using Base = SessionModel<FlintModel>;
  static constexpr bool kPrefillChunkGrid = false;
  using Outputs = Base::Outputs;
  using SessionSnapshotMeta = Base::SessionSnapshotMeta;
  using SnapshotRequest = Base::SnapshotRequest;
  using RowRun = Base::RowRun;

  // max_tokens bounds a walk's rows (a prefill chunk, the diagnostic
  // forward); max_cache_tokens the paged pool's capacity in tokens (rounded
  // up to a block) — shared by every slot (max_context()). max_requests:
  // the session slots. decode_rows: the fixed decode batch's row ceiling —
  // the slots (no draft rows here); 0 = kDecodeRows, the floor. kv_format:
  // the K/V pool's storage (bf16, or the fp8 row form — engine.kv_dtype).
  // mtp must be false (no draft block).
  FlintModel(const FlintTextConfig& cfg, const std::string& checkpoint_dir, int max_tokens, int64_t max_cache_tokens,
             FlintResidency residency = FlintResidency::Streaming, BoundaryReducer* boundary = nullptr,
             int tp_rank = 0, int tp_world = 1, int max_requests = 1, bool mtp = false, int decode_rows = 0,
             LatentFormat kv_format = LatentFormat::kBf16, bool serving_logits = false);
  ~FlintModel();
  FlintModel(const FlintModel&) = delete;
  FlintModel& operator=(const FlintModel&) = delete;

  // Every byte the constructor (and its layer objects) will allocate for
  // a shape, from the same formulas BEFORE anything is allocated.
  using MemoryPlan = dgpp::MemoryPlan;
  static MemoryPlan plan_memory(const FlintTextConfig& cfg, int max_tokens, int64_t max_cache_tokens,
                                int tp_rank = 0, int tp_world = 1,
                                FlintResidency residency = FlintResidency::Streaming, int max_requests = 1,
                                bool mtp = false, int decode_rows = 0, LatentFormat kv_format = LatentFormat::kBf16,
                                bool serving_logits = false);

  // The cold diagnostic forward: one request on slot 0 (which must be
  // closed), fresh state, every row's logits; the slot is closed after.
  // capture_layers: every layer's residual output [T, H] in layer_states.
  Outputs forward(const std::vector<int64_t>& token_ids, bool capture_layers = false);

  static constexpr int decode_rows_cap() { return 32; }
  static constexpr int prefill_chunk_tokens() { return kPrefillChunkTokens; }
  static constexpr int kv_block_tokens_static() { return kBlockTokens; }
  // The same number for a shape that is not built yet (the memory plan).
  static size_t session_snapshot_bytes(const FlintTextConfig&, int, bool) { return kSnapshotStamp; }
  // One cache block's bytes across every pool plane (the NVMe cold tier's
  // record, issue #26) for a shape that is not built yet.
  static size_t kv_block_bytes_static(const FlintTextConfig& cfg, int tp_rank, int tp_world, LatentFormat kv_format);
  using Base::session_snapshot_bytes;

  const FlintTextConfig& config() const { return cfg_; }
  const MimoKvPool& kv_pool() const { return pool_; }
  LatentFormat kv_format() const { return pool_.shape().format; }
  // Route traces on the eager DECODE rows too (the prefill walk's are
  // always staged while the core's route traces are on). Off by default.
  void set_decode_row_traces(bool on) { decode_row_traces_ = on; }

  // ---- the session core's hooks (engine/session_model.hpp) --------------------
  Outputs run_rows(const RowRun& run);
  void reset_slot_state(int req);
  GlmSpecSegments spec_segments(int, int = 0) const { return GlmSpecSegments{}; }
  size_t snapshot_state_bytes() const { return kSnapshotStamp; }
  size_t draft_state_bytes() const { return 0; }
  void write_state_snapshot(int req, uint8_t* dst, int spec_row);
  void write_draft_snapshot(int, uint8_t*, bool, int64_t) {}
  void read_state_snapshot(int, const uint8_t*) {}
  void read_draft_snapshot(int, const uint8_t*) {}
  bool has_pool() const { return true; }
  MimoKvPool& pool() { return pool_; }
  const MimoKvPool& pool() const { return pool_; }
  void graph_prepare();
  // No draft block: every draft entry refuses (the serving app never
  // enables MTP for this family).
  [[noreturn]] void mtp_run_rows(int, const int64_t*, int64_t, int, bool, bool, int, int, int = 0) {
    refuse_mtp("mtp_run_rows");
  }
  void snapshot_draft_state(int) {}
  void restore_draft_state(int) {}
  static constexpr bool kDraftChain = false;
  static constexpr bool kBatchedDraftChain = false;
  const uint16_t* draft_hidden_rows() const { return nullptr; }
  void snapshot_chain_state(int) {}
  void restore_chain_state(int) {}

 private:
  static constexpr int kBlockTokens = 64;
  static constexpr int kPrefillChunkTokens = 2048;
  // No state families beside the pool's blocks: a 16-byte zero stamp keeps
  // the prefix arena's slots addressable (it refuses a zero-byte model).
  static constexpr size_t kSnapshotStamp = 16;

  void build_layer_objects(const FlintLayerResident& r);
  static QwenMoeWeights moe_view(const FlintMoeResident& m);
  int moe_ordinal(int layer) const { return moe_ordinal_[static_cast<size_t>(layer)]; }
  int table_slots() const;
  // The layer's two blocks over `resid` (the residual, updated in place).
  struct WalkRows {
    const RowRun* run = nullptr;
    const int32_t* req_ids = nullptr;
    const int64_t* pos = nullptr;
    int req = 0;
    bool decode = false;
    bool capture = false;
    int moe_table_slot = -1;   // >= 0: the capture's graph table
    MoeTraceStaging* trace = nullptr;
  };
  // One layer of the stack on `resid`. `pending` is the previous block's
  // output not yet added to the residual (the layer adds it in the fused
  // add + input-norm launch; null: the residual is complete); the return
  // is this layer's MLP block output, left pending the same way.
  const uint16_t* enqueue_layer(const FlintLayerResident& r, int pool_layer, uint16_t* resid, int T,
                                const WalkRows& rows, const uint16_t* pending);
  void add_pending(uint16_t* resid, const uint16_t* pending, int row0, int n);
  uint16_t* stage(uint16_t* fallback, int T, int width, bool capture);
  void fold(uint16_t* buf, int T, int width, bool capture);

  FlintTextConfig cfg_;
  FlintLayerStream loader_;
  CublasLtGemm gemm_;
  // The bf16 matrices' 12-bit companions (kernels/bf12_companions.hpp:
  // the fused projection, o_proj, the dense MLPs, the head) and the rows
  // of the walk in flight (the prefetch windows' view).
  static constexpr int kBf12ExpandSlots = 1;
  void pack_layer_companions(int layer, const FlintLayerResident& r);
  void finish_companions();
  Bf12Companions bf12_;
  bool bf12_built_ = false;
  double bf12_s_ = 0.0;
  int walk_rows_ = 1;
  bool decode_row_traces_ = false;
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;
  FlintGemmWorkspace gw_;
  FlintGlobalsResident globals_;
  GlmMoeConfig moe_cfg_;
  int n_split_ = FlintAttentionLayer::default_decode_splits();
  std::vector<int> moe_ordinal_;  // [layers]: the MoE layer's slot, -1 for dense

  // Layer objects (built at first use, rebound per layer).
  std::unique_ptr<FlintAttentionLayer> attn_;
  std::unique_ptr<FlintBf16Mlp> dense_;
  std::unique_ptr<QwenMoeLayer> moe_;

  MimoKvPool pool_;  // every layer's paged K/V

  // Activations [M rows] (the token rows, the head's outputs and the tail
  // mirrors are the core's).
  uint16_t* resid_ = nullptr;  // [M, H] the residual
  uint16_t* x_ = nullptr;      // [M, H] the normed block input
  uint16_t* y_ = nullptr;      // [M, H] the block output (the fold's fallback)
  // The walk's route traces: one pinned staging slot per MoE layer ([moe
  // layers][M][top_k]), materialized after the walk's final sync — the
  // prefill rows' and, while route traces are on for decode rows, those
  // too, whose staging carries the selection keys ([moe layers][decode
  // rows][E]).
  int32_t* h_route_ids_ = nullptr;
  float* h_route_weights_ = nullptr;
  float* h_route_biased_ = nullptr;

  // The L2 weight prefetcher (the boundary windows, bit-identical on or
  // off; engine.l2_prefetch false A/Bs).
  WeightPrefetcher prefetch_;
  size_t prefetch_window_bytes_ = 0;
  bool bf16_side_grants_ = false;  // the packable bf16 matrices are side grants (bf12-only residency)
  void prefetch_ffn_side(const FlintLayerResident& r);
  void prefetch_attention_side(int layer);
  void prefetch_head();
  void prefetch_attn(const FlintAttnResident& a);
  void prefetch_bf16_view(const uint16_t* w, size_t bytes);
  void prefetch_bf16_raw(const uint16_t* w, size_t bytes);
};

}  // namespace dgpp
