#pragma once
// Types and model interface used by the eager and graph engines, prefix
// arena and speculators. The adapters are templates over a model type;
// this header defines shared row limits, output buffers and pick callbacks.
//
// A model type M serves the engines when it provides (the GLM model is the
// reference; every name below is used by engine/*.hpp):
//   struct Outputs : DecodeOutputs;  struct SessionSnapshotMeta;
//   struct SnapshotRequest;  static int prefill_chunk_tokens();
//   cudaStream_t stream();  bool mtp_enabled();  int max_session_requests();
//   int lm_vocab_begin/count();  const float* device_logits();
//   const int64_t* device_tokens/positions();  device_feed(req, rows);
//   int64_t max_context();  BoundaryReducer* set_boundary(BoundaryReducer*);
//   void set_decode_route_traces(bool); void set_decode_tail_mirrors(bool);
//   int64_t kv_blocks_total/in_use();  kv_blocks_for_tokens(tokens);
//   kv_block_tokens();  int session_snapshot_align();
//   the session_* surface (prefill, prefill_resume, step, verify, rollback,
//   draft, draft_chain(_fits), draft_rollback, close, position,
//   reserve_blocks, snapshot(_bytes/_post_row0/_release/attach)) and the
//   session_graph_* capture/stage/seed/settle/collect surface.
#include <cstdint>
#include <functional>
#include <vector>

#include "sample/sampler.hpp"
#include "text/tool_grammar.hpp"

namespace dgpp {

// The decode rows (2026-09-10, the runtime shape). A fixed row batch holds
// every request slot's verify rows — max_concurrency x (1 + mtp_depth) —
// and the session-core families (engine/session_model.hpp) size their
// per-row scratch, their draft windows and the bus's latency slot to that
// number AT RUNTIME (SessionParams::decode_rows; the serving app derives
// it from the cluster config, floored at kDecodeRows so every existing
// recipe keeps its exact shape). kDecodeRowsMax is the build-time bound
// the fixed arrays carry (the pick kernels' per-thread winners,
// PickVerdict::winners — kernels/pick.hpp's kPickMaxRows mirrors it):
// sixteen request slots at MTP3 (four verification rows each).
// kDecodeRows (8) is the floor and the default; GLM-5.3-Flash's own
// decode model keeps it as its fixed batch (models/glm/forward.hpp).
// kSpecRows bounds one request's verify rows (the pending token plus up
// to kSpecRows - 1 drafts; kernels/glm_spec.hpp's kSpecMaxDrafts mirrors
// it).
constexpr int kDecodeRows = 8;
constexpr int kDecodeRowsMax = 64;
constexpr int kSpecRows = 6;  // 2026-09-13: the DSpark block (5 drafts + the pending token)

// One forward's rows as the engines read them.
struct DecodeOutputs {
  std::vector<uint16_t> final_hidden_bits;  // bf16 [tokens, hidden]
  // fp32 [tokens, lm_vocab_count]: this rank's vocabulary columns, or
  // the full vocabulary for an unsharded head. Keep the head's fp32
  // accumulators unrounded so sampling and log-probability comparisons
  // do not incur an extra BF16 rounding.
  std::vector<float> logits;
  int lm_vocab_begin = 0;  // first vocab column of logits
  int lm_vocab_count = 0;
};

// The greedy pick over one row; the stochastic pick (the request's spec,
// its counter RNG advanced by the draw, its context ids for the penalties,
// the grammar mask or null, the logit_bias row or null).
using DecodePick = std::function<int32_t(const DecodeOutputs&)>;
using DecodeSample = std::function<sample::Result(
    const DecodeOutputs&, const sample::Params&, sample::Rng&,
    const std::vector<int32_t>& context, const text::TokenMask* mask,
    const float* bias)>;
// The speculative verify's row-0 decision (accept the draft or the
// residual sample) and the per-row global winners from every rank's local
// maxes (bus_greedy_pick_rows on the fabric, the identity at world 1).
using SpecRow0 = std::function<sample::SpecPrefixDecision(
    const DecodeOutputs& row0, int32_t draft, const sample::Params& p,
    sample::Rng& rng, const std::vector<int32_t>& context)>;
using SpecPickRows = std::function<std::vector<int32_t>(
    const std::vector<sample::Candidate>&)>;

}  // namespace dgpp
