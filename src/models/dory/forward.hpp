#pragma once
// Dory1.0-4B (DoryForCausalLM) on the shared session core
// (docs/dory_plan.md §1.5, §2). The walk, one row per token:
//
//   h = embed(x)
//   for l in [0, n_input):                            h = layer(l, h)
//   input_h = h;  state = h
//   for i in [0, n_recurrent_loops):
//     h = justnorm(state + input_h)                   the re-injection (fp32 add)
//     for l in [n_input, n_input + n_recurrent):      h = layer(l, h, i)
//     state = justnorm(h);  h = state
//   for l in [n_input + n_recurrent, num_layers):     h = layer(l, h)
//   logits = (h @ lm_headᵀ) ⊙ (fp32(logit_scale) · logit_factor)
//
// One physical layer is ONE sublayer — attention or MLP, never both (plan
// D5) — so the walk holds one object of each kind and rebinds it per layer.
// The recurrent block's weights are shared across the loops; only the K/V
// planes differ (`layer(l, h, i)`), which is what the pool's `repeats` encode
// (plan D2: `engine.dory_kv_cache`, `per_loop` by default, or `last_loop`).
//
// There is no LayerNorm, no final norm before the head, no MTP draft layer
// (plan D4) and no tensor parallelism (one rank: the reference refuses
// tensor_parallel_size != 1 and the release is a single image).
//
// State per request slot: its row of the paged K/V pool alone. A prefix
// snapshot is the block list, and every plane holds the FULL history (the
// sliding window is applied at read time), so any position snapshots.
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
#include "kernels/gemm.hpp"
#include "models/dory/config.hpp"
#include "models/dory/kv_pool.hpp"
#include "models/dory/layers.hpp"
#include "models/dory/loader.hpp"

namespace dgpp {

// The recurrent K/V policy (plan D2, the `engine.dory_kv_cache` key):
// `per_loop` gives every loop of the recurrent block its own planes (the
// exact, lossless path — the reference's `per_loop`, the architecture's
// default); `last_loop` shares one plane per layer and keeps only the last
// loop's rows.
enum class DoryKvPolicy : uint8_t { PerLoop, LastLoop };

class DoryModel : public SessionModel<DoryModel> {
 public:
  using Base = SessionModel<DoryModel>;
  static constexpr bool kPrefillChunkGrid = false;  // split-invariant: any chunk cut
  using Outputs = Base::Outputs;
  using SessionSnapshotMeta = Base::SessionSnapshotMeta;
  using SnapshotRequest = Base::SnapshotRequest;
  using RowRun = Base::RowRun;

  DoryModel(const DoryConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
            int64_t max_cache_tokens, DoryResidency residency = DoryResidency::Streaming,
            int max_requests = 1, int decode_rows = 0, bool serving_logits = false,
            DoryKvPolicy kv_policy = DoryKvPolicy::PerLoop);
  ~DoryModel();
  DoryModel(const DoryModel&) = delete;
  DoryModel& operator=(const DoryModel&) = delete;

  // Every byte the constructor (and its layer objects) will allocate for a
  // shape, from the same formulas BEFORE anything is allocated.
  using MemoryPlan = dgpp::MemoryPlan;
  static MemoryPlan plan_memory(const DoryConfig& cfg, int max_tokens, int64_t max_cache_tokens,
                                DoryResidency residency = DoryResidency::Streaming,
                                int max_requests = 1, int decode_rows = 0,
                                bool serving_logits = false,
                                DoryKvPolicy kv_policy = DoryKvPolicy::PerLoop);

  // The cold diagnostic forward: one request on slot 0 (which must be
  // closed), fresh state, every row's logits; the slot is closed after.
  // `capture_layers` returns one state per sublayer APPLICATION — the input
  // block, the recurrent block once per loop, then the output block (plan
  // §1.5) — each [T, H] bf16, in walk order.
  Outputs forward(const std::vector<int64_t>& token_ids, bool capture_layers = false);

  static constexpr int decode_rows_cap() { return 32; }
  static constexpr int prefill_chunk_tokens() { return kPrefillChunkTokens; }
  static constexpr int kv_block_tokens_static() { return kBlockTokens; }
  static size_t kv_block_bytes_static(const DoryConfig& cfg, DoryKvPolicy policy);
  static size_t session_snapshot_bytes(const DoryConfig& cfg, bool mtp) { (void)cfg; (void)mtp; return kSnapshotStamp; }
  using Base::session_snapshot_bytes;
  // Several cold prompts as the spans of one walk (every row carries its own
  // request id and position, so a span attends to its own cache only).
  int64_t prefill_group_span_limit() const { return max_tokens_; }

  // ---- the engine keys (plan D1, D2; plumbed in stage 7) ----------------------
  // `engine.dory_loops`: the loop count override (the checkpoint's is 2). The
  // exact path is the config's value; this changes what the engine computes.
  static void set_recurrent_loops(int loops);
  static void clear_recurrent_loops();
  static int recurrent_loops_override() { return recurrent_loops_; }
  // `engine.dory_kv_cache`: the recurrent K/V policy, default per_loop.
  static void set_kv_policy(DoryKvPolicy policy) { kv_policy_ = policy; }
  static DoryKvPolicy kv_policy() { return kv_policy_; }
  static DoryKvPolicy parse_kv_policy(const std::string& name);  // throws on an unknown name

  const DoryConfig& config() const { return cfg_; }
  const DoryKvPool& kv_pool() const { return pool_; }
  DoryKvPolicy policy() const { return policy_; }

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
  DoryKvPool& pool() { return pool_; }
  const DoryKvPool& pool() const { return pool_; }
  void graph_prepare();
  static constexpr bool kDraftChain = false;  // no draft layer at all (plan D4)
  static constexpr bool kBatchedDraftChain = false;  // nothing to batch-draft
  const uint16_t* draft_hidden_rows() const { return nullptr; }
  void snapshot_draft_state(int) {}
  void restore_draft_state(int) {}
  void snapshot_chain_state(int) {}
  void restore_chain_state(int) {}
  // The family has no draft layer: the engine's draft surface refuses by name
  // rather than silently ignoring the drafter keys (plan D4).
  void mtp_run_rows(int req, const int64_t* tokens, int64_t first_pos, int T, bool decode_row,
                    bool capture, int head_rows, int batch_requests, int hidden_row = 0);

 private:
  static constexpr int kBlockTokens = 64;
  static constexpr int kPrefillChunkTokens = 2048;
  // No state families beside the pool's blocks: a 16-byte zero stamp keeps
  // the prefix arena's slots addressable (it refuses a zero-byte model).
  static constexpr size_t kSnapshotStamp = 16;
  // A layer-state capture stages its rows on the device and hands them over
  // once, so a capture walk keeps a plain walk's arithmetic (a per-layer
  // synchronize would not). Above this many bytes the walk falls back to the
  // host copy and its per-layer synchronize.
  static constexpr size_t kCaptureStagingCap = size_t{256} << 20;

  struct WalkRows {
    const RowRun* run = nullptr;
    const int32_t* req_ids = nullptr;
    const int64_t* pos = nullptr;
    bool decode = false;
    bool capture = false;
  };
  // One physical layer over `in` into `out` (attention or MLP, plan D5),
  // appending the attention layers' K/V to their (layer, loop) plane.
  void enqueue_layer(int layer, int loop, const uint16_t* in, int T, uint16_t* out,
                     const WalkRows& rows);
  int pool_layers() const { return cfg_.num_hidden_layers; }
  bool last_loop_policy() const { return policy_ == DoryKvPolicy::LastLoop; }

  DoryConfig cfg_;
  DoryLayerStream loader_;
  CublasLtGemm gemm_;
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;
  DoryGlobalsResident globals_;
  int n_split_ = 16;
  DoryKvPolicy policy_ = DoryKvPolicy::PerLoop;

  // The two sublayers, built at first use and rebound per physical layer.
  std::unique_ptr<DoryAttentionLayer> attn_;
  std::unique_ptr<DoryMlpLayer> mlp_;

  DoryKvPool pool_;  // every attention layer's paged K/V (one plane per loop)

  // Activations [M rows]. The walk ping-pongs between them; `input_h_` holds
  // the input block's output (the loop's re-injection) and `state_` the
  // recurrent state across the loops.
  uint16_t* resid_ = nullptr;    // [M, H] the walk's hidden
  uint16_t* x_ = nullptr;        // [M, H] the ping-pong partner
  uint16_t* y_ = nullptr;        // [M, H] the recurrent block's partner
  uint16_t* input_h_ = nullptr;  // [M, H] the input block's output
  uint16_t* state_ = nullptr;    // [M, H] the re-injected state

  // The layer-state capture's device staging (run_rows): reallocated when a
  // capture needs more room, freed with the model.
  uint16_t* snap_dev_ = nullptr;
  size_t snap_dev_bytes_ = 0;
  int snap_apps_ = 0;

  static int recurrent_loops_;      // `engine.dory_loops` (0: the config's)
  static DoryKvPolicy kv_policy_;   // `engine.dory_kv_cache`
};

}  // namespace dgpp
