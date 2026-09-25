#pragma once
// MiMo-V2.6-Flash model with resident or streaming weights on the shared
// session core. The layer walk follows the release's MiMoV2DecoderLayer:
//
//   h = embed(x)
//   per layer: h += attn(input_norm(h))         (GQA qk 192 / v 128, the fused
//                                                fp8 projection, half-split
//                                                partial RoPE on 64 dims, the
//                                                value scale; a sliding window
//                                                of 128 with a per-head sink on
//                                                the SWA layers, global on the
//                                                others)
//              h += mlp(post_norm(h))           (the fp8 dense MLP on layer 0,
//                                                the sigmoid-routed MXFP4 MoE
//                                                without a shared expert after)
//   logits = lm_head(final_norm(h))
//
// one row walk (run_rows) serves every entry point — the cold diagnostic
// forward, a session's prefill chunks, the eager decode rows and the
// captured decode graphs (engine/graph_engine.hpp's contract) — the
// engine/session_model.hpp core owns everything around it (positions,
// feeds, chunking, snapshots, the graph era, the draft block's plumbing).
//
// State per request slot: its row of the paged K/V pool (models/mimo/
// kv_pool.hpp, 64-token blocks, every layer's K/V — the window layers
// too, read through their window) and — with the draft block — its
// hidden window. Nothing else: a rejected verify row leaves stale K/V rows
// past the position that no later row reads, so the speculative commit
// table is empty and the prefix snapshot is the block list plus the
// draft's hidden row.
//
// TP (plan D2): `tp_world` > 1 loads this rank's slices (the fused
// projection's chunks: 64/W query heads and 4/W or 8/W kv heads, every
// expert's intermediate slice, the dense MLP's slice, the lm-head vocab
// slice) and folds the attention and MLP block outputs per layer through
// `boundary` (the canonical rank-order sum). The router and the norms are
// replicated; every rank's residual is bitwise the others'.
//
// The default MTP draft block: the first draft layer (model.mtp.layers.0)
// resident beside the stack — a sliding-window layer with the dense MLP —
// one more pool layer for its K/V, the hidden window per slot; its input
// is eh_proj([enorm(embed(tok_{q+1})) | hnorm(h_q)]) with h_q the main
// stack's output hidden at q (POST final norm — vLLM's mimo_v2_mtp
// convention, as GLM-4.7's), its head final_layernorm then the shared
// lm_head. Native opt-in loads all three heads, each conditioned on the
// backbone hidden at p-d and token p+1, with independent paged K/V. Its
// backbone history ring is included in prefix snapshots and draft rollback.
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
#include "models/glm/moe_layer.hpp"
#include "models/mimo/config.hpp"
#include "models/mimo/kv_pool.hpp"
#include "models/mimo/layers.hpp"
#include "models/mimo/loader.hpp"

namespace dgpp {

class MimoModel : public SessionModel<MimoModel> {
 public:
  // Several cold prompts as the spans of one walk (session_prefill_group):
  // the attention rows carry their own request ids and positions, so a
  // span attends to its own request's cache only; every span within
  // max_tokens.
  int64_t prefill_group_span_limit() const { return max_tokens_; }
  using Base = SessionModel<MimoModel>;
  using Outputs = Base::Outputs;
  using SessionSnapshotMeta = Base::SessionSnapshotMeta;
  using SnapshotRequest = Base::SnapshotRequest;
  using RowRun = Base::RowRun;

  // max_tokens bounds a walk's rows (a prefill chunk, the diagnostic
  // forward); max_cache_tokens the paged pool's capacity in tokens (rounded
  // up to a block) — shared by every slot (max_context()). max_requests:
  // the session slots. mtp: the draft block. decode_rows: the fixed decode
  // batch's row ceiling — the slots times the verify rows per request (1 +
  // the MTP depth), the serving app's derivation (engine/decode_outputs.hpp);
  // 0 = kDecodeRows, the floor.
  // kv_format: the K/V pool's storage (models/mimo/kv_pool.hpp: bf16, or the
  // fp8 row form per head — engine.kv_dtype).
  MimoModel(const MimoTextConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
            int64_t max_cache_tokens, MimoResidency residency = MimoResidency::Streaming,
            BoundaryReducer* boundary = nullptr, int tp_rank = 0, int tp_world = 1,
            int max_requests = 1, bool mtp = false, int decode_rows = 0,
            LatentFormat kv_format = LatentFormat::kBf16);
  ~MimoModel();
  MimoModel(const MimoModel&) = delete;
  MimoModel& operator=(const MimoModel&) = delete;

  // Every byte the constructor (and its layer objects) will allocate for
  // a shape, from the same formulas BEFORE anything is allocated.
  using MemoryPlan = dgpp::MemoryPlan;
  // One cache block's bytes across every pool plane (the NVMe cold tier's
  // record, issue #26) for a shape that is not built yet.
  static size_t kv_block_bytes_static(const MimoTextConfig& cfg, int tp_rank, int tp_world, bool mtp,
                                      LatentFormat kv_format);
  static MemoryPlan plan_memory(const MimoTextConfig& cfg, int max_tokens, int64_t max_cache_tokens,
                                int tp_rank = 0, int tp_world = 1,
                                MimoResidency residency = MimoResidency::Streaming,
                                int max_requests = 1, bool mtp = false, int decode_rows = 0,
                                LatentFormat kv_format = LatentFormat::kBf16);

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
  static size_t session_snapshot_bytes(const MimoTextConfig& cfg, int tp_world, bool mtp);
  using Base::session_snapshot_bytes;

  const MimoTextConfig& config() const { return cfg_; }
  const MimoKvPool& kv_pool() const { return pool_; }
  LatentFormat kv_format() const { return pool_.shape().format; }
  // Route traces on the eager DECODE rows too (the prefill walk's are
  // always staged while the core's route traces are on): the decode
  // audit certifies a routing flip from the row's own picks. Off by
  // default — three small async D2H copies per MoE layer per eager step
  // that serving never wants (the graph engine turns every trace off).
  void set_decode_row_traces(bool on) { decode_row_traces_ = on; }

  // ---- the session core's hooks (engine/session_model.hpp) --------------------
  Outputs run_rows(const RowRun& run);
  void reset_slot_state(int req);
  GlmSpecSegments spec_segments(int req, int snapshot_row0 = 0) const;
  size_t snapshot_state_bytes() const { return kSnapshotStamp; }
  size_t draft_state_bytes() const {
    return native_mtp() ? size_t(kNativeHistoryRows) * cfg_.hidden_size * 2 : 0;
  }
  void write_state_snapshot(int req, uint8_t* dst, int spec_row);
  void write_draft_snapshot(int req, uint8_t* dst, bool live, int64_t pos);
  void read_state_snapshot(int, const uint8_t*) {}
  void read_draft_snapshot(int req, const uint8_t* src);
  bool has_pool() const { return true; }
  MimoKvPool& pool() { return pool_; }
  const MimoKvPool& pool() const { return pool_; }
  void graph_prepare();
  void mtp_run_rows(int req, const int64_t* tokens, int64_t first_pos, int T, bool decode_row,
                    bool capture, int head_rows, int batch_requests);
  void snapshot_draft_state(int req);
  void restore_draft_state(int req);
  void mtp_select_block(int index);
  // Depth >= 2: the chain rows take the block's output residual (pre
  // final_layernorm — the previous_hidden_states vLLM's mimo_v2_mtp feeds
  // its next step) from mtp_r_; native heads ignore these recursive inputs.
  // Chain proposals do not mutate the native backbone history ring.
  static constexpr bool kDraftChain = true;
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

  void build_layer_objects(const MimoLayerResident& r);
  static GlmMoeWeights moe_view(const MimoMoeResident& m);
  int moe_ordinal(int layer) const { return moe_ordinal_[static_cast<size_t>(layer)]; }
  int table_slots() const;
  int pool_layers() const { return cfg_.num_hidden_layers + (mtp_ ? cfg_.mtp_layers_loaded : 0); }
  static std::vector<int> pool_kv_heads(const MimoTextConfig& cfg, const MimoLocalGeometry& geo, bool mtp);
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
  // One layer of the stack on `resid`. `pending` is the previous block's
  // output not yet added to the residual (the layer adds it in the fused
  // add + input-norm launch; null: the residual is complete); the return
  // is this layer's MLP block output, left pending the same way — the
  // caller adds it (add_pending) or folds it into the next norm.
  const uint16_t* enqueue_layer(const MimoLayerResident& r, int pool_layer, uint16_t* resid, int T,
                                const WalkRows& rows, const uint16_t* pending);
  // The pending block output added into rows [row0, row0 + n) of `resid`
  // (no norm): the plain add of the two-kernel chain.
  void add_pending(uint16_t* resid, const uint16_t* pending, int row0, int n);
  uint16_t* stage(uint16_t* fallback, int T, int width, bool capture);
  void fold(uint16_t* buf, int T, int width, bool capture);

  MimoTextConfig cfg_;
  MimoLayerStream loader_;
  CublasLtGemm gemm_;
  // The bf16 decode weights' 12-bit companions (kernels/bf12_companions.hpp)
  // and the rows of the walk in flight (the prefetch windows' view).
  static constexpr int kBf12ExpandSlots = 1;
  void pack_layer_companions(int layer, const MimoLayerResident& r);
  void finish_companions();
  Bf12Companions bf12_;
  bool bf12_built_ = false;
  double bf12_s_ = 0.0;
  int walk_rows_ = 1;
  bool dense_mma_ = true;
  bool prefill_last_head_ = false;
  bool mtp_cache_only_ = false;
  bool decode_row_traces_ = false;
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;
  MimoGemmWorkspace gw_;
  MimoGlobalsResident globals_;
  GlmMoeConfig moe_cfg_;
  int n_split_ = 32;
  std::vector<int> moe_ordinal_;  // [layers + 1]: the MoE layer's slot (the draft's last), -1 for dense

  // Layer objects (built at first use, rebound per layer).
  std::unique_ptr<MimoAttentionLayer> attn_;
  std::unique_ptr<MimoDenseMlp> dense_;
  std::unique_ptr<GlmMoeLayer> moe_;

  MimoKvPool pool_;  // every layer's paged K/V (the draft's too)

  // Activations [M rows] (the token rows, the head's outputs and the tail
  // mirrors are the core's).
  uint16_t* resid_ = nullptr;  // [M, H] the residual
  uint16_t* x_ = nullptr;      // [M, H] the normed block input
  uint16_t* y_ = nullptr;      // [M, H] the block output (the fold's fallback)
  // The routed chain's fp32 sum: with no shared expert the MoE layer hands
  // the chain back unrounded (enqueue_*_f32, the Qwen contract) and the
  // block rounds it once onto the wire buffer.
  float* moe_acc_ = nullptr;   // [M, H]
  // The walk's route traces: one pinned staging slot per MoE layer ([moe
  // layers][M][top_k]), materialized after the walk's final sync — the
  // prefill rows' and, while route_traces_ is on (the eager engines; the
  // graph era turns it off), the decode rows' too, whose staging carries
  // the selection keys the decode path insists on ([moe layers][decode
  // rows][E]). The decode gate certifies a routing flip against the
  // re-forward's rows with them.
  int32_t* h_route_ids_ = nullptr;
  float* h_route_weights_ = nullptr;
  float* h_route_biased_ = nullptr;

  // The L2 weight prefetcher (the boundary windows, bit-identical on or
  // off; DGPP_L2_PREFETCH=off A/Bs).
  WeightPrefetcher prefetch_;
  size_t prefetch_window_bytes_ = 0;
  // The persisting L2 set-aside the next layer's projection is prefetched
  // into from the attention fold on (0: off; plan §7.1).
  size_t l2_persist_bytes_ = 0;
  // The early (normal-priority) prefetch of the next layer's projection
  // from fold 1 on, sized to the fold's idle DRAM time (0: off; an
  // experiment behind DGPP_MIMO_EARLY_QKV_MB together with the fp4 slot
  // kernels' streaming loads, -DDGPP_FP4_SLOT_STREAMING=1).
  size_t early_qkv_bytes_ = 0;
  void prefetch_ffn_side(const MimoLayerResident& r);
  void prefetch_attention_side(int layer);
  void prefetch_head();
  void prefetch_attn(const MimoAttnResident& a);
  void prefetch_fp8(const GlmQuantMatrix& m);
  void prefetch_bf16_view(const uint16_t* w, size_t bytes);
  bool bf16_side_grants_ = false;  // the raw bf16 matrices are side grants (bf12-only residency)

  // The draft block (mtp_): its fusion scratch and residual (the window,
  // the counters and the feeds are the core's).
  static constexpr int kNativeHistoryRows = kPrefillChunkTokens + 32 + 3;
  bool native_mtp() const { return mtp_ && cfg_.mtp_layers_loaded == 3; }
  void native_mtp_rows(int req, const int64_t* tokens, int64_t first_pos, int T, bool decode,
                       bool capture, int head_rows);
  int draft_block_ = 0;
  uint16_t* native_history_ = nullptr;
  uint8_t* native_backup_ = nullptr;
  int64_t* native_positions_ = nullptr;
  uint16_t* native_output_ = nullptr;
  uint16_t* mtp_h_ = nullptr;   // [M, H] the gathered hidden rows (decode rows)
  uint16_t* mtp_in_ = nullptr;  // [M, 2H] [enorm(embed) | hnorm(h)]
  uint16_t* mtp_r_ = nullptr;   // [M, H] the draft's residual
};

}  // namespace dgpp
