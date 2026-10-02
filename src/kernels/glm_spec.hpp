#pragma once
// The speculative step's DEVICE-SIDE control (DESIGN §9, the on-device
// step, phase B): what the host used to do between a verify and the next
// step — read the verdict, roll the rejected rows' state back, advance the
// position, stage the next rows' positions — as kernels behind the pick's
// verdict, so a replayed graph carries its own control flow and the host
// reads the verdict only to log it.
//
// The rollback is a CONDITIONAL copy: cudaMemcpyAsync nodes cannot be
// predicated, a kernel can read `verdict->accepted` and either copy the
// post-row-(accepted-1) snapshots over the live state or return. Every
// state family the verify snapshots (glm_forward.hpp: spec_rec_, spec_conv_,
// spec_tail_ per DSA layer) is one segment of the table below; the copy is
// bitwise the host path's (session_rollback), the bytes are the same bytes.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "kernels/pick.hpp"

namespace dgpp {

// One rollback segment: `dst` is the live state; the snapshot after row a
// lives at `snapshots + a * row_stride_bytes`; `bytes` are copied. All
// three addresses 16-byte aligned, `bytes` and the stride multiples of 16.
struct GlmSpecSegment {
  void* dst = nullptr;
  const void* snapshots = nullptr;
  size_t row_stride_bytes = 0;
  size_t bytes = 0;
  size_t request_stride_bytes = 0;  // mapped commits: distance between physical destinations
};

constexpr int kSpecMaxSegments = 32;  // KDA rec + KDA conv + DSA layers

struct GlmSpecSegments {
  int count = 0;
  GlmSpecSegment seg[kSpecMaxSegments];
  // A checkpoint-and-replay family (kernels/kda.hpp KdaReplay, 2026-09-29):
  // no snapshot rows; the pass saved its rows' INPUTS and the next pass
  // replays the accepted ones. The commit copies the saved rows
  // (`replay_src`, this pass's) over the buffer the next pass replays from
  // (`replay_dst`) and records the accepted count in replay_pending[req] —
  // on every step with an active verdict, retraction or not. Mapped
  // commits offset both buffers by req * replay_request_stride_bytes and
  // index the pending count by the physical request. Null replay_dst: none.
  void* replay_dst = nullptr;
  const void* replay_src = nullptr;
  size_t replay_bytes = 0;
  size_t replay_request_stride_bytes = 0;
  int32_t* replay_pending = nullptr;
};

// The step's commit, behind the verdict: when verdict->accepted < rows,
// copies every segment's snapshot row (accepted - 1) over its live state;
// always advances *session_pos by verdict->accepted. `rows` is the
// verify's row count (the graph's T); the kernel never trusts
// verdict->rows for the copy bound.
void glm_spec_commit(const PickVerdict* verdict, int rows, const GlmSpecSegments& segments,
                     int64_t* session_pos, cudaStream_t stream,
                     const int32_t* request_map = nullptr, int batch_index = 0);

// The pipelined replay's stage handshake: the graph of a
// slot is launched BEFORE the host has decided what its pick needs (the
// masks of the rows, staged from the previous replay's outcome), so the
// node ahead of the pick waits for the host: it bumps the slot's device
// replay counter and spins until the host's PINNED stage counter reaches
// it (the host stages, then publishes with a release store), sleeping
// between polls. A wait past `timeout_ns` sets *late (pinned) and lets
// the replay proceed — loud on the host, never a hung stream.
void glm_stage_wait(const uint64_t* pinned_stage_seq, uint64_t* device_seq,
                    uint32_t* pinned_late, int64_t timeout_ns,
                    cudaStream_t stream);

// The verdict's publication (2026-09-06, the pipelined replay): a kernel
// node right after the verify's pick bumps the slot's device replay
// counter and stores it to PINNED memory with a system-scope release, so
// the host — polling it — learns the verdict is readable while the
// replay's tail runs on. (An event record node in the graph stalled every
// fifth relaunch of the exec; a kernel node does not.)
void glm_publish_seq(uint64_t* device_seq, uint64_t* pinned_out,
                     cudaStream_t stream);

// The confidence publication (2026-09-14, the scheduled verify depth,
// engine/verify_schedule.hpp): a kernel node at the tail of an MTP replay
// copies the slot's `count` (<= 32) confidence logits from device memory
// to PINNED memory with system-scope stores, fences, then bumps the
// slot's device counter and stores it to `pinned_seq` with a system-scope
// release — the host, polling the counter with an acquire load, reads the
// logits that describe the drafts its next replay will verify.
void glm_publish_f32(const float* src, float* pinned_dst, int count,
                     uint64_t* device_seq, uint64_t* pinned_seq,
                     cudaStream_t stream);

// The reduced-depth fixed batch's token feed (2026-09-14, the scheduled
// verify depth): every slot keeps its persistent feed at `feed_rows` rows
// (the whole block), and a batch variant verifying `rows_per_request` <
// feed_rows rows per slot reads them compacted — out[q * rows_per_request
// + t] = feeds[q * feed_rows + t] for q < requests, t < rows_per_request —
// so the walk's rows stay contiguous per request.
void glm_spec_gather_feed(const int64_t* feeds, int requests, int feed_rows, int rows_per_request,
                          int64_t* out, cudaStream_t stream, const int32_t* request_map = nullptr);

// Uploads `count` uint32 words from PINNED, device-mapped host memory with
// a kernel (system-scope loads: the host wrote them before the handshake
// above released) — the masks of a slot's rows, behind the stage wait.
void glm_upload_words(const uint32_t* pinned_src, uint32_t* dst, size_t count,
                      cudaStream_t stream);

// Uploads `count` (<= 64) int32 words from a PINNED, device-mapped host
// buffer into device memory with a kernel — the decode path's replacement
// for a cudaMemcpyAsync H2D of its row tables. A memcpy node in the
// captured decode graph executes on the copy-engine queue, which is
// in-order and shared by every stream in the process; in a one-process
// multi-rank world a peer rank's queued dependency wait at that queue's
// head blocked the upload (docs/batched_mtp_graph_stall.md). The source is
// read with system-scope loads: the host wrote it before the launch.
void glm_upload_i32(const int32_t* pinned_src, int32_t* dst, int count,
                    cudaStream_t stream);
void glm_upload_i64(const int64_t* pinned_src, int64_t* dst, int count,
                    cudaStream_t stream);

// The decode rows' metadata from the device position: step_pos[r] =
// *session_pos + r for r < rows (the replacement for the host's staged
// h_step_pos_ upload in a device-driven graph). A row at or past
// `max_context` emits -1, the padding sentinel: a request's lifetime
// reservation is capped at the ceiling, so the fixed-width verify's tail
// past it lands in no K/V block, index stripe or state row.
void glm_spec_positions(const int64_t* session_pos, int rows, int64_t max_context,
                        int64_t* step_pos, cudaStream_t stream);

// Fixed slot-major row batch: request_ids[r] selects its device position;
// the offset within that request's `rows_per_request` group is added when
// the slot is open. A closed slot has position <= 0 and emits -1 for every
// row, which is the shared KDA/DSA padding sentinel; so does every row at
// or past `max_context` (see glm_spec_positions).
void glm_spec_positions_batched(const int64_t* session_pos,
                                const int32_t* request_ids, int rows,
                                int rows_per_request, int64_t max_context,
                                int64_t* step_pos, cudaStream_t stream);

// The in-graph draft's rows off the verify's verdict (phase C). The draft
// block runs a FIXED `rows` rows per step; the accepted rows are real
// (step_pos[r] = *block_pos + r, tokens[r] = winners[r]) and the rest are
// padding (step_pos[r] = -1, which the DSA decode path skips — nothing is
// written at any position — and tokens[r] = winners[0], any valid id).
// Then *block_pos += accepted and *next_out = verdict->next (the verify's
// pick is about to be overwritten by the draft's; the token the main stack
// consumes next survives here for glm_spec_next_tokens).
void glm_spec_draft_rows(const PickVerdict* verdict, int rows,
                         int64_t* block_pos, int64_t* step_pos, int64_t* tokens,
                         int64_t* next_out, cudaStream_t stream);

// One independent draft group per fixed request slot. `verdicts[q]` drives
// rows [q * rows_per_request, ...); inactive verdicts (accepted == 0) emit
// only padding and leave block_pos[q] unchanged. The group index is the
// request slot unless request_map selects another physical slot (-1: padding).
void glm_spec_draft_rows_batched(const PickVerdict* verdicts, int requests, int rows_per_request,
                                 int64_t* block_pos, int64_t* step_pos, int64_t* tokens,
                                 int64_t* next_out, cudaStream_t stream,
                                 const int32_t* request_map = nullptr);

// A plain device-to-device copy as a KERNEL (a memcpy node may not enter
// the captured decode graph): `bytes` a multiple of 16, both pointers
// 16-byte aligned. The in-graph draft snapshots its DSA tail ring with it
// before its rows run, so a fallback decided on the host can roll the block
// back (GlmDiagnosticModel::session_draft_rollback).
void glm_device_copy(void* dst, const void* src, size_t bytes,
                     cudaStream_t stream);
// Two bytes stored by a one-thread kernel (a kernel node where a memset
// node is not allowed: the kernels-only decode graph).
void glm_device_store_u8x2(uint8_t* dst, uint8_t a, uint8_t b, cudaStream_t stream);

// The chained draft row (depth >= 2, 2026-09-06): the single draft block's
// recursion. After the block's rows off the verdict (glm_spec_draft_rows)
// the block runs one more row at position *block_pos + chain_index, fed the
// previous draft pick (`draft_verdict->next`) as its token and the block's
// own previous output row as its hidden: row `verify_verdict->accepted - 1`
// of `block_x` for the first chain row (the last accepted row), row
// `src_row` (0) after that. The hidden row is copied into the position
// cache at that position (the block reads its hidden there; the next
// step's real row overwrites it), the row's position and token are staged
// and the request span is set to one row. A position at or past
// `max_context` stages padding (-1) and copies nothing. The counter does
// not move: everything the row writes is provisional and positional.
void glm_spec_chain_row(const PickVerdict* verify_verdict, int src_row,
                        const PickVerdict* draft_verdict,
                        const uint16_t* block_x, int hidden,
                        uint16_t* hidden_cache, const int64_t* block_pos,
                        int chain_index, int64_t max_context,
                        int64_t* step_pos, int64_t* tokens, int32_t* req_spans,
                        cudaStream_t stream);

// The same row for the families whose draft hidden lives in a per-slot
// window of `window_rows` rows by position (the block's output row lands at
// window[pos % window_rows]; engine/session_model.hpp's chain).
void glm_spec_chain_row_window(const PickVerdict* verify_verdict, int src_row,
                               const PickVerdict* draft_verdict,
                               const uint16_t* block_x, int hidden,
                               uint16_t* window, int window_rows,
                               const int64_t* block_pos, int chain_index,
                               int64_t max_context, int64_t* step_pos,
                               int64_t* tokens, int32_t* req_spans,
                               cudaStream_t stream);

// The chain rows of the FIXED BATCH (2026-09-10, the batched depth >= 2
// chain on the window families): one row per request slot q, at rows
// [0, requests) — position block_pos[q] + chain_index, the token
// draft_verdicts[q].next, the hidden the block's own output row: for the
// first chain row the batched draft run's row q * rows_per_request +
// (verify_verdicts[q].accepted - 1) (its last accepted row), after it row
// q of the previous chain run — landed in slot q's window (window + q *
// window_stride_elems, row pos % window_rows). The request ids and the
// spans are rewritten to the compact one-row-per-request layout (the next
// replay's staging restores the verify's). A closed slot
// (verify_verdicts[q].accepted == 0) or a position at or past
// `max_context` stages padding (-1) and copies nothing. The counters do
// not move.
void glm_spec_chain_rows_batched(const PickVerdict* verify_verdicts,
                                 const PickVerdict* draft_verdicts, int requests,
                                 int rows_per_request, const uint16_t* block_x, int hidden,
                                 uint16_t* window, int window_rows, size_t window_stride_elems,
                                 const int64_t* block_pos, int chain_index, int64_t max_context,
                                 int64_t* step_pos, int64_t* tokens, int32_t* req_ids,
                                 int32_t* req_spans, cudaStream_t stream,
                                 const int32_t* request_map = nullptr);

// The next replay's fed tokens, written at the end of this one (phase D):
// tokens[0] = *next (the verify's), tokens[1 + c] = drafts.v[c]->next (the
// block's guesses for the tokens after it, one per draft position).
constexpr int kSpecMaxDrafts = 7;  // kSpecRows - 1 (engine/decode_outputs.hpp)
struct GlmSpecDrafts {
  const PickVerdict* v[kSpecMaxDrafts] = {};
  int count = 0;
};
void glm_spec_next_tokens(const int64_t* next, const GlmSpecDrafts& drafts,
                          int64_t* tokens, cudaStream_t stream);
inline void glm_spec_next_tokens(const int64_t* next,
                                 const PickVerdict* draft_verdict,
                                 int64_t* tokens, cudaStream_t stream) {
  GlmSpecDrafts d;
  d.v[0] = draft_verdict;
  d.count = 1;
  glm_spec_next_tokens(next, d, tokens, stream);
}

// End-of-replay token feeds for the fixed batch. The plain T=1 graph takes
// each verify verdict's next token. The MTP graph takes the parked verify
// next plus one draft verdict per request and draft position (drafts.v[c]
// is a [requests] array; rows_per_request == 1 + drafts.count). Inactive
// groups are zeroed so a later padded replay always embeds a valid token
// id.
void glm_spec_verify_next_tokens_batched(const PickVerdict* verify_verdicts, int requests,
                                         int rows_per_request, int64_t* tokens, cudaStream_t stream,
                                         const int32_t* request_map = nullptr);
void glm_spec_next_tokens_batched(const int64_t* next, const GlmSpecDrafts& drafts, int requests,
                                  int rows_per_request, int64_t* tokens, cudaStream_t stream,
                                  const int32_t* request_map = nullptr);
inline void glm_spec_next_tokens_batched(const int64_t* next,
                                         const PickVerdict* draft_verdicts,
                                         int requests, int rows_per_request,
                                         int64_t* tokens,
                                         cudaStream_t stream) {
  GlmSpecDrafts d;
  d.v[0] = draft_verdicts;
  d.count = 1;
  glm_spec_next_tokens_batched(next, d, requests, rows_per_request, tokens,
                               stream);
}

}  // namespace dgpp

namespace dgpp {
// Expand compact batch groups to physical request IDs; -1 is padding.
void glm_batch_rows(const int32_t* map, int requests, int rows_per_request, int32_t* ids,
                    int32_t* spans, cudaStream_t stream);
}  // namespace dgpp
