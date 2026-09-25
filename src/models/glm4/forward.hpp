#pragma once
// GLM-4.7 model with resident or streaming weights on the shared session
// core. The layer walk follows transformers' Glm4MoeDecoderLayer:
//
//   h = embed(x)
//   per layer: h += attn(input_norm(h))         (GQA, biased q/k/v, q/k head
//                                                norm, half-split partial RoPE)
//              h += mlp(post_norm(h))           (dense NVFP4 MLP on the first
//                                                layers, the sigmoid-routed MoE
//                                                with its NVFP4 shared expert after)
//   logits = lm_head(final_norm(h))
//
// one row walk (run_rows) serves every entry point — the cold diagnostic
// forward, a session's prefill chunks, the eager decode rows and the
// captured decode graphs (engine/graph_engine.hpp's contract) — the
// engine/session_model.hpp core owns everything around it (positions,
// feeds, chunking, snapshots, the graph era, the draft block's plumbing).
//
// State per request slot: its row of the paged K/V pool (models/glm4/
// kv_pool.hpp, 64-token blocks) and — with the draft block — its hidden
// window. Nothing else: a rejected verify row leaves stale K/V rows past
// the position that no later row reads (plan D5), so the speculative
// commit table is empty and the prefix snapshot is the block list plus
// the draft's hidden row.
//
// TP (plan D7): `tp_world` > 1 loads this rank's slices (96/W query heads,
// 8/W kv heads, every expert's intermediate slice, the dense MLP's slice,
// the lm-head vocab slice) and folds the attention and MLP block outputs
// per layer through `boundary` (the canonical rank-order sum). The router
// and the norms are replicated; every rank's residual is bitwise the
// others'.
//
// The MTP draft block (plan D1, D5): the draft layer resident beside the
// stack (its BF16 experts requantized to NVFP4 at load), one more pool
// layer for its K/V, the hidden window per slot; its input is
// eh_proj([enorm(embed(tok_{q+1})) | hnorm(h_q)]) with h_q the main
// stack's output hidden at q (POST final norm — vLLM's glm4_moe_mtp
// convention; the pre-norm residual halved the acceptance), its head
// shared_head.norm then the shared lm_head.
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
#include "models/glm/moe_layer.hpp"
#include "models/glm4/config.hpp"
#include "models/glm4/kv_pool.hpp"
#include "models/glm4/layers.hpp"
#include "models/glm4/loader.hpp"

namespace dgpp {

class Glm4Model : public SessionModel<Glm4Model> {
 public:
  // Several cold prompts as the spans of one walk (session_prefill_group,
  // 2026-09-14, the group prefill ported from DeepSeek): the attention rows
  // carry their own request ids and positions, so a span attends to its
  // own request's cache only; every span within max_tokens.
  int64_t prefill_group_span_limit() const { return max_tokens_; }
  using Base = SessionModel<Glm4Model>;
  using Outputs = Base::Outputs;
  using SessionSnapshotMeta = Base::SessionSnapshotMeta;
  using SnapshotRequest = Base::SnapshotRequest;
  using RowRun = Base::RowRun;

  // max_tokens bounds a walk's rows (a prefill chunk, the diagnostic
  // forward); max_cache_tokens the paged pool's capacity in tokens (rounded
  // up to a block) — shared by every slot (max_context()). max_requests:
  // the session slots. mtp: the draft block (the draft layer resident
  // beside the stack, one more pool layer, the hidden window per slot).
  // decode_rows: the fixed decode batch's row ceiling — the slots times
  // the verify rows per request (1 + the MTP depth), the serving app's
  // derivation (engine/decode_outputs.hpp); 0 = kDecodeRows, the floor.
  Glm4Model(const Glm4TextConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
            int64_t max_cache_tokens, Glm4Residency residency = Glm4Residency::Streaming,
            BoundaryReducer* boundary = nullptr, int tp_rank = 0, int tp_world = 1,
            int max_requests = 1, bool mtp = false, int decode_rows = 0);
  ~Glm4Model();
  Glm4Model(const Glm4Model&) = delete;
  Glm4Model& operator=(const Glm4Model&) = delete;

  // Every byte the constructor (and its layer objects) will allocate for
  // a shape, from the same formulas BEFORE anything is allocated.
  using MemoryPlan = dgpp::MemoryPlan;
  // One cache block's bytes across every pool plane (the NVMe cold tier's
  // record, issue #26) for a shape that is not built yet.
  static size_t kv_block_bytes_static(const Glm4TextConfig& cfg, int tp_rank, int tp_world, bool mtp);
  static MemoryPlan plan_memory(const Glm4TextConfig& cfg, int max_tokens, int64_t max_cache_tokens,
                                int tp_rank = 0, int tp_world = 1,
                                Glm4Residency residency = Glm4Residency::Streaming,
                                int max_requests = 1, bool mtp = false, int decode_rows = 0);

  // The cold diagnostic forward: one request on slot 0 (which must be
  // closed), fresh state, every row's logits; the slot is closed after.
  // capture_layers: every layer's residual output [T, H] in layer_states.
  Outputs forward(const std::vector<int64_t>& token_ids, bool capture_layers = false);
  // The draft block over a prompt (the parity gate's surface): the cold
  // forward, then the draft rows q = 0 .. T-2 (token q+1, the hidden at q)
  // with the head on every row — logits [T-1, count], the draft's normed
  // final hidden [T-1, hidden].
  Outputs mtp_forward(const std::vector<int64_t>& token_ids);

  static constexpr int decode_rows_cap() { return 32; }
  static constexpr int prefill_chunk_tokens() { return kPrefillChunkTokens; }
  static constexpr int kv_block_tokens_static() { return kBlockTokens; }
  // The same number for a shape that is not built yet (the memory plan).
  static size_t session_snapshot_bytes(const Glm4TextConfig& cfg, int tp_world, bool mtp);
  using Base::session_snapshot_bytes;

  const Glm4TextConfig& config() const { return cfg_; }
  const Glm4KvPool& kv_pool() const { return pool_; }

  // ---- the session core's hooks (engine/session_model.hpp) --------------------
  Outputs run_rows(const RowRun& run);
  void reset_slot_state(int req);
  GlmSpecSegments spec_segments(int req, int snapshot_row0 = 0) const;
  size_t snapshot_state_bytes() const { return kSnapshotStamp; }
  size_t draft_state_bytes() const { return 0; }
  void write_state_snapshot(int req, uint8_t* dst, int spec_row);
  void write_draft_snapshot(int, uint8_t*, bool, int64_t) {}
  void read_state_snapshot(int, const uint8_t*) {}
  void read_draft_snapshot(int, const uint8_t*) {}
  bool has_pool() const { return true; }
  Glm4KvPool& pool() { return pool_; }
  const Glm4KvPool& pool() const { return pool_; }
  void graph_prepare();
  void mtp_run_rows(int req, const int64_t* tokens, int64_t first_pos, int T, bool decode_row,
                    bool capture, int head_rows, int batch_requests);
  void snapshot_draft_state(int) {}
  void restore_draft_state(int) {}
  // Depth >= 2: the chain rows take the block's output residual (pre
  // shared_head.norm — the previous_hidden_states vLLM's glm4_moe_mtp
  // feeds its next step) from mtp_r_; no state to snapshot around them.
  static constexpr bool kDraftChain = true;
  // The fixed batch's chain too (2026-09-10): the graph engine records
  // session_graph_capture_draft_chain_batch past depth 1.
  static constexpr bool kBatchedDraftChain = true;
  const uint16_t* draft_hidden_rows() const { return mtp_r_; }
  void snapshot_chain_state(int) {}
  void restore_chain_state(int) {}

 private:
  static constexpr int kBlockTokens = 64;
  static constexpr int kPrefillChunkTokens = 2048;
  // No state families beside the pool's blocks: a 16-byte zero stamp keeps
  // the prefix arena's slots addressable (it refuses a zero-byte model).
  static constexpr size_t kSnapshotStamp = 16;

  void build_layer_objects(const Glm4LayerResident& r);
  static GlmMoeWeights moe_view(const Glm4MoeResident& m);
  int moe_ordinal(int layer) const { return layer - cfg_.first_k_dense_replace; }
  int table_slots() const;
  int pool_layers() const { return cfg_.num_hidden_layers + (mtp_ ? 1 : 0); }
  // The layer's two blocks over `resid` (the residual, updated in place):
  // the attention block and the MLP/MoE block with their folds.
  struct WalkRows {
    const int32_t* req_ids = nullptr;
    const int64_t* pos = nullptr;
    int req = 0;
    bool decode = false;
    bool capture = false;
    int moe_table_slot = -1;   // >= 0: the capture's graph table
    MoeTraceStaging* trace = nullptr;
  };
  void enqueue_layer(const Glm4LayerResident& r, int pool_layer, uint16_t* resid, int T, const WalkRows& rows);
  uint16_t* stage(uint16_t* fallback, int T, int width, bool capture);
  void fold(uint16_t* buf, int T, int width, bool capture);

  Glm4TextConfig cfg_;
  Glm4LayerStream loader_;
  CublasLtGemm gemm_;
  // The bf16 decode weights' 12-bit companions (kernels/bf12_companions.hpp)
  // and the rows of the walk in flight (the prefetch windows' view).
  static constexpr int kBf12ExpandSlots = 1;
  void pack_layer_companions(int layer, const Glm4LayerResident& r);
  void finish_companions();
  Bf12Companions bf12_;
  bool bf12_built_ = false;
  double bf12_s_ = 0.0;
  int walk_rows_ = 1;
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;
  Glm4GemmWorkspace gw_;
  Glm4GlobalsResident globals_;
  GlmMoeConfig moe_cfg_;
  int n_split_ = 32;

  // Layer objects (built at first use, rebound per layer).
  std::unique_ptr<Glm4AttentionLayer> attn_;
  std::unique_ptr<Glm4DenseMlp> dense_;
  std::unique_ptr<GlmMoeLayer> moe_;

  Glm4KvPool pool_;  // every layer's paged K/V (the draft's too)

  // Activations [M rows] (the token rows, the head's outputs and the tail
  // mirrors are the core's).
  uint16_t* resid_ = nullptr;  // [M, H] the residual
  uint16_t* x_ = nullptr;      // [M, H] the normed block input
  uint16_t* y_ = nullptr;      // [M, H] the block output (the fold's fallback)
  // The prefill walk's route traces: one pinned staging slot per MoE layer
  // ([moe layers + draft][M][top_k]), materialized after the walk's final
  // sync.
  int32_t* h_route_ids_ = nullptr;
  float* h_route_weights_ = nullptr;

  // The L2 weight prefetcher (the boundary windows, bit-identical on or
  // off; DGPP_L2_PREFETCH=off A/Bs).
  WeightPrefetcher prefetch_;
  size_t prefetch_window_bytes_ = 0;
  void prefetch_ffn_side(const Glm4LayerResident& r);
  void prefetch_attention_side(int layer);
  void prefetch_head();
  void prefetch_attn(const Glm4AttnResident& a);
  void prefetch_fp4(const GlmFp4Matrix& m);

  // The draft block (mtp_): its fusion scratch and residual (the window,
  // the counters and the feeds are the core's).
  uint16_t* mtp_h_ = nullptr;   // [M, H] the gathered hidden rows (decode rows)
  uint16_t* mtp_in_ = nullptr;  // [M, 2H] [enorm(embed) | hnorm(h)]
  uint16_t* mtp_r_ = nullptr;   // [M, H] the draft's residual
};

}  // namespace dgpp
