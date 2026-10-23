#pragma once
// The DFlash2 block drafter's shared execution (models/dflash2/dflash2.hpp
// is the drafter MODEL; this is the engine that runs it): the feature
// store that feeds the draft planes from the target's runs, the block
// forward over [bonus, mask x drafts] rows, the mask rows' head and top-K
// selector walk, the eager (re)drafts — scalar and stacked-batch — and the
// recorded block proposal the decode graph captures (the draft's kernel
// nodes only; the family's run_rows tail and the capture driver order
// them). Hosted by one target model at a time; the target family supplies
// DFlash2Host (the pool's block table and draft planes, the session and
// graph state, the embeddings, the head dispatch and the block GEMMs) and
// reads the taps out of the feature accumulator (acc / tap_gemm_out) at
// its own layer outputs.
//
// The recorded walk may be ARMED to draw for the sampled requests
// (arm_proposals, the Qwen35 family's rule): null specs keep every walk
// the argmax. Every walk writes the slot's confidence (the scheduled
// verify depth's input, confidence()).
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "engine/boundary_reducer.hpp"
#include "kernels/gemm.hpp"
#include "kernels/pick.hpp"
#include "kernels/sample_pick.hpp"
#include "kernels/rope_scaling.hpp"
#include "models/dflash2/dflash2.hpp"

namespace dgpp {

// Everything the drafter reads from the host model. Bound once at
// DFlash2Drafter::init (the model outlives the drafter; the callbacks
// may close over the model).
struct DFlash2Host {
  // geometry and device state.
  int hidden = 0;
  int num_target_layers = 0;
  int target_vocab = 0;     // the target config's vocab (the cross-check's domain)
  int64_t vocab_count = 0;  // the served head's rows (the walk's domain)
  int vocab_begin = 0;      // the first served id (the sharded head's slice)
  int rank = 0, world = 1;
  cudaStream_t stream = nullptr;
  int max_requests = 0;
  int max_tokens = 0;         // the target's row bound (the feature accumulator's depth)
  int max_decode_rows = 0;  // the family's row ceiling (max(kDecodeRows, decode_rows, max_requests))
  int max_context = 0;      // the session's cache bound (the staging gates)
  int64_t position_limit = 0;  // the target config's position ceiling (the config cross-check's domain)
  bool weights_fp8 = false; // engine.dflash_weights fp8
  const RopeScaling* rope = nullptr;  // the target's ramp (null: the draft's own theta)
  int64_t* session_pos = nullptr;     // device [max_requests]: the committed ends
  int64_t* decode_tokens = nullptr;   // device d_tokens_: the fixed batch's feeds sit behind the decode rows
  // the target's pool: the draft planes ride its block table.
  const int32_t* device_tables = nullptr;  // PagedBlockTable::device_tables()
  int total_blocks = 0;
  int block_tokens = 0;
  std::function<bool(int req, int64_t pos, cudaStream_t stream)> ensure_request_blocks = nullptr;
  // One DRAFT layer's plain bf16 K/V plane (l indexes the draft stack).
  struct Plane {
    uint16_t* k = nullptr;
    uint16_t* v = nullptr;
  };
  std::function<Plane(int l)> plane = nullptr;
  std::function<int64_t(int req)> session_position = nullptr;  // the committed end (-1: closed)
  const uint16_t* embed = nullptr;                 // the target's embedding table
  // The drafter's block GEMMs on the target's gemm object (the row
  // envelope is the family's configure call).
  CublasLtGemm* gemm = nullptr;
  void* gemm_ws = nullptr;
  size_t gemm_ws_bytes = 0;
  std::function<void(int rows, bool decode)> gemm_rows = nullptr;
  BoundaryReducer* boundary = nullptr;  // world > 1 (null at world 1)
  // The mask rows' head rows into `out` (rows x vocab_count), the exact
  // product (the selector walk reads the top-K off it): the family's
  // served-head dispatch, with or without its draft screen.
  std::function<void(const uint16_t* hidden, float* out, int rows)> head;
};

// The capture-time step fields (SessionModel members the family keeps
// current): read at every recorded-proposal call, not at init.
struct DFlash2CaptureCtx {
  int64_t* step_tokens = nullptr;      // the step's token source (the feed)
  int decode_rows = 0;                // the current decode run's rows (SessionModel)
  int graph_feed_rows = 0;            // 0: the rows per request
  int graph_batch_requests = 0;
  int graph_rows_per_request = 0;
  const void* graph_batch_map_source = nullptr;  // non-null: a compaction sits in the batch
  bool graph_device_positions = false;
  bool graph_device_tokens = false;
};

class DFlash2Drafter {
 public:
  DFlash2Drafter() = default;
  DFlash2Drafter(const DFlash2Drafter&) = delete;
  DFlash2Drafter& operator=(const DFlash2Drafter&) = delete;
  ~DFlash2Drafter() { free(); }

  // Parses the checkpoint (refusing anything but DFlash2DraftModel by
  // name), cross-checks it against the target (h's geometry), loads this
  // rank's slices and allocates the scratch. The family's pool draft
  // planes (DFlash2Config's num_hidden_layers / local_kv_heads(world) /
  // head_dim over the pool's token slots and block size) must be ready —
  // the drafter addresses them through h.plane.
  void init(const DFlash2Host& h, const std::string& dir);
  void free();
  bool on() const { return on_; }
  const DFlash2Config& cfg() const { return cfg_; }
  const DFlash2Weights& weights() const { return w_; }
  // The target layer's fc slice (the family's tap site reads the layer's
  // output through it into tap_gemm_out and folds into acc), -1 = no tap.
  int tap(int layer) const;
  float* acc() const { return df_acc_; }
  float* tap_gemm_out() const { return df_t32_; }
  int drafts() const { return cfg_.drafts(); }
  int query_rows() const { return cfg_.query_rows(); }
  int batch_slots() const { return df_batch_; }
  static constexpr int rows_cap() { return 2048; }  // the feature path's row chunk

  // The target run's rows into the draft planes (the family's run_rows
  // tail): the tapped features at the run's positions (rejected verify
  // rows included — overwrite, no rollback).
  void store_features(int T, const int32_t* d_req, const int64_t* d_pos);
  // False without a draft (pool exhausted / context bound): the caller
  // runs the step without speculation.
  bool draft(int req, int64_t bonus, std::vector<int32_t>* drafts);
  // The batched redraft: one stacked block forward for every slot; the
  // per-slot failures mirror the scalar call.
  void draft_batch(const std::vector<int>& reqs, const std::vector<int64_t>& bonuses,
                   std::vector<std::vector<int32_t>>* drafts);
  // The recorded block proposal off a verify's verdict (the graph engine
  // records it inside a step replay): the staged rows are device-driven,
  // the feed lands in the slot's persistent feed rows and the drafts +
  // candidate lists in the pinned mirrors.
  void capture_block_draft(int req, const PickVerdict* verdict, const DFlash2CaptureCtx& cap);
  void capture_block_draft_batch(const PickVerdict* verdicts, int requests, const DFlash2CaptureCtx& cap);
  // The recorded drafts' host copy (readable once the replay's end event
  // has passed), the mask rows' candidate lists (the acceptance
  // diagnostics) and the selector's confidence per draft
  // ([max_requests][drafts], written by every walk).
  const int32_t* drafts_host(int req) const;
  const int32_t* candidates_host(int req) const;
  const float* confidence() const { return df_conf_; }
  // The sampled requests' block proposals (the engine's draft rule): the
  // recorded walk draws each draft from the selector's softmax at the
  // request's draft temperature and writes the set it drew from to
  // proposals[req * stride + l] (and the pinned mirror), the next verify's
  // ratio rule; a greedy request keeps the argmax and n = 0. Null specs:
  // the argmax walk, no proposals (the default). Before the first capture.
  void arm_proposals(const SampleSpec* specs, DraftProposal* proposals,
                     DraftProposal* proposals_host, int stride);
  bool proposals_armed() const { return df_specs_ != nullptr; }

  // The memory plan's lines (the family appends them to its own plan,
  // before the family's own tap staging): the weights, the feature path
  // and the block scratch exactly as init allocates them.
  static void plan_entries(const DFlash2Config& cfg, int world, int decode_rows, int max_tokens,
                           int max_requests, int64_t vocab_count, bool weights_fp8,
                           std::vector<std::pair<std::string, size_t>>* out);

 private:
  // The block forward's layers over `slots` stacked [bonus, mask x D]
  // blocks (row-wise ops over every row at once; each slot's K/V appends
  // and attention at its row offset against its own block table).
  void block_layers(int slots, const int* reqs, bool capture);
  // The mask rows' head rows and top-K per slot (world > 1: this rank's
  // slice lists staged, folded and merged into the global top-K), the
  // hidden projection and the selector walk into df_tok_ (D per slot).
  void block_select(int slots, const int* reqs, bool capture);

  DFlash2Host h_{};
  DFlash2CaptureCtx cap_{};  // bound per family at the capture calls
  bool on_ = false;
  DFlash2Config cfg_;
  DFlash2Weights w_;
  std::vector<int> tap_;  // [num_target_layers + 1] fc slice, -1 = no tap
  int df_nh_ = 0, df_kvh_ = 0, df_qw_ = 0, df_kvr_ = 0, df_i_ = 0;
  int df_batch_ = 1;  // the stacked draft batch width (slots per forward)
  // Feature path (store_features): the fp32 tap accumulator and staging,
  // the hidden norm and the per-draft-layer k|v rows at rows_cap rows.
  float* df_acc_ = nullptr;     // [max_tokens, H] the fc sums
  float* df_t32_ = nullptr;     // [max_tokens, H] one tap's GEMM out (F32)
  uint16_t* df_norm_ = nullptr; // [rows_cap, H] hidden_norm
  uint16_t* df_kv_ = nullptr;   // [rows_cap, 2*kvr]
  float* df_inv_freq_ = nullptr;  // [head_dim / 2] the draft's rope
  // Block forward (draft / draft_batch / the captures) at BR = df_batch_ *
  // query_rows stacked rows.
  uint16_t* df_resid_ = nullptr;  // [BR, H]
  uint16_t* df_x_ = nullptr;      // [BR, H]
  uint16_t* df_xc_ = nullptr;     // [BR, H] conv-conditioned rows
  uint16_t* df_qkv_ = nullptr;    // [BR, QW + 2*KV]
  uint16_t* df_q_ = nullptr;      // [BR, QW]
  uint16_t* df_attn_ = nullptr;   // [BR, QW]
  float* df_attn_part_ = nullptr; // the split-K block attention's partials
  uint16_t* df_o_ = nullptr;      // [BR, H] (and the down-proj fold)
  uint16_t* df_mlp_ = nullptr;    // [BR, I]
  uint16_t* df_gate_ = nullptr;   // [BR, I]
  uint16_t* df_up_ = nullptr;     // [BR, I]
  uint16_t* df_delta_ = nullptr;  // [BR, 2 * taps * groups] the conv kernels' deltas
  uint16_t* df_h_ = nullptr;      // [BR, H] the block's final norm
  int32_t* df_zero_ = nullptr;    // [QR] zero requests (constant positions)
  float* df_logits_ = nullptr;    // [BR, V] the mask rows' head rows (F32)
  void* df_topk_ws_ = nullptr;
  size_t df_topk_ws_bytes_ = 0;
  float* df_hidden32_ = nullptr;  // [BR, rank] the selector's hidden projection
  int32_t* df_ids_ = nullptr;     // [BR, top_k]
  float* df_sc_ = nullptr;        // [BR, top_k]
  int32_t* df_tok_ = nullptr;     // [df_batch_ * drafts] the proposals
  uint16_t* df_table_ = nullptr;  // world > 1: the slice top-K fold
  int32_t* df_mirror_h_ = nullptr;  // pinned [max_requests, drafts]: the recorded drafts
  int32_t* df_cands_h_ = nullptr;   // pinned [max_requests, drafts, top_k]: the walked candidates
  int64_t* df_pos_ = nullptr;       // [BR] the block's positions
  int64_t* df_tokens_ = nullptr;    // [BR] [bonus, mask x D]
  int32_t* df_tok_h_ = nullptr;     // pinned readback
  int64_t* df_io64_h_ = nullptr;    // pinned staging: [2][df_batch_ * QR]
  float* df_conf_ = nullptr;        // [max_requests][drafts]
  // The armed proposals (arm_proposals): the engine's spec table and
  // proposal rows, df_props_stride_ rows per request.
  const SampleSpec* df_specs_ = nullptr;
  DraftProposal* df_props_ = nullptr;
  DraftProposal* df_props_h_ = nullptr;
  int df_props_stride_ = 0;
};

}  // namespace dgpp
