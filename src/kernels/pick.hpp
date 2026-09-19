#pragma once
// The distributed greedy pick's DEVICE half (DESIGN §9, the on-device
// step): the per-rank local argmax, the wire encoding the bus's SUM
// all-reduce turns into a gather, and the verdict every rank decodes from
// the folded table. No bus dependency — the collective between the two
// kernels is the caller's (DevicePicker in engine/tp_bus.hpp records
// it as a graph node or runs it eagerly).
//
// Wire table: [rows + 1][world][kPickSlotsPerRank] bf16 slots holding 6-bit
// digits (bf16 carries small integers exactly, and every slot has exactly
// one nonzero contributor, so the fp32-accumulated fold is exact). Group r
// < rows is row r's candidates — rank k's fp32 logit bits (6 digits) and
// vocab id (3 digits) in slots [r*world + k]; group `rows` is each rank's
// DIGEST of its previous verdict (9 digits, 54 bits). The digest is the
// readback invariant of the old host pick, one step late: every rank
// folds an identical table and so computes an identical verdict — a rank
// whose table was corrupt (the 2026-09-01 fabric race) computes a
// different one, and every rank sees its odd digest at the next pick.
//
// Order: canonical (logit desc, id asc), the sampler's candidate_before —
// glm_pick_local's top-2 is bitwise sample::local_max plus the
// runner-up the gen log prints, and glm_pick_verdict's merge is
// merge_greedy. NaN logits are unspecified (the host scan keeps a NaN at
// index 0, a tree reduction never does; a NaN logit is a broken model).
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

constexpr int kPickLogitDigits = 6;  // 36 bits carry the float's 32
constexpr int kPickIdDigits = 3;     // 18 bits carry a vocab id (< 262144)
constexpr int kPickSlotsPerRank = kPickLogitDigits + kPickIdDigits;
constexpr int kPickMaxRows = 64;     // dgpp::kDecodeRowsMax (engine/decode_outputs.hpp)
constexpr int kPickMaxRequests = 16;  // one verdict per fixed request slot (the GLM-5.3 decode batch's cap)
constexpr int kPickMaxWorld = 8;
constexpr uint64_t kPickDigestBits = 6ull * kPickSlotsPerRank;  // 54

// Elements of the wire table for a `rows`-row pick at `world` ranks
// (candidate groups plus the digest group), rounded up to the bus's
// even-element contract (odd worlds; the pad slot is zeroed).
constexpr size_t device_pick_table_elems(int rows, int world) {
  const size_t slots = static_cast<size_t>(rows + 1) *
                       static_cast<size_t>(world) *
                       static_cast<size_t>(kPickSlotsPerRank);
  return slots + (slots & 1);
}

// One row's local result: this rank's slice argmax and the runner-up's
// logit (-inf when the slice has one column).
struct PickLocal {
  int32_t best_id = -1;
  float best_logit = 0.0f;
  float second_logit = 0.0f;
};

// The verdict every rank computes from the folded table. Row r of a
// verify stands iff every row before it stood and row r-1's winner is
// the token fed as row r (row 0 always stands); `next` is the last
// standing row's winner; winners[0..accepted) are the draft block's next
// inputs. digest_mismatch is a rank bitmask: ranks whose carried digest
// differed from this rank's (0 = every rank agreed on the previous
// verdict); peer_digests holds what each rank carried, for the report.
struct PickVerdict {
  int32_t rows = 0;
  int32_t accepted = 0;
  int32_t next = -1;
  int32_t winners[kPickMaxRows] = {
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
  };
  uint64_t digest = 0;
  uint32_t digest_mismatch = 0;
  uint64_t peer_digests[kPickMaxWorld] = {};
};

// Kernel 1 (before the fold): for each of `rows` rows of logits[rows,
// vocab_count] (fp32, this rank's slice starting at vocab id
// `vocab_begin`), the top-2 under the canonical order -> locals[row], and
// the zeroed table with this rank's (best logit bits, best id) digits in
// group `row` and *carry_digest's digits in the digest group.
// `row_select` (optional, rows == 1): the one row read is logits row
// (row_select->accepted - 1) — the in-graph draft's head runs on every
// row of a fixed-size batch and its pick reads the last ACCEPTED one.
void device_pick_local(const float* logits, int rows, int vocab_count,
                    int vocab_begin, int rank, int world,
                    const uint64_t* carry_digest, uint16_t* table,
                    PickLocal* locals, cudaStream_t stream,
                    const PickVerdict* row_select = nullptr);

// Fixed request-slot batch form. Without row_select, `rows` candidate rows
// are contiguous in logits exactly as in glm_pick_local. With row_select,
// there is one candidate per request (`rows == requests`): request q reads
// logits[q * source_row_stride + max(row_select[q].accepted - 1, 0)]. This
// is the MTP draft pick over a fixed [request, spec-row] head output; an
// inactive request has accepted == 0 and reads its harmless padding row.
void device_pick_local_batched(
    const float* logits, int rows, int vocab_count, int vocab_begin, int rank,
    int world, const uint64_t* carry_digest, uint16_t* table,
    PickLocal* locals, cudaStream_t stream,
    const PickVerdict* row_select, int requests, int source_row_stride);

// Kernel 2 (after the fold): decodes every rank's candidates and digests,
// merges per row, judges against fed[rows] (the verify's tokens), writes
// *verdict (the host's pinned mirror), *device_verdict (the device-side
// consumers' copy — the commit kernel, the draft; may be null) and
// *carry_digest = verdict->digest — the next pick's digest group source.
void device_pick_verdict(const uint16_t* table, int rows, int world, int rank,
                      const int64_t* fed, PickVerdict* verdict,
                      PickVerdict* device_verdict, uint64_t* carry_digest,
                      cudaStream_t stream);

// Decodes `requests` independent verdicts from one folded table. Candidate
// rows are packed request-major (`rows == requests * rows_per_request`).
// `positions` is the source row layout and marks an unoccupied request when
// its first position is negative; position_stride permits the draft pick's
// packed one-candidate table to refer back to a wider fixed row group.
// Inactive verdicts have rows/accepted == 0 and next == -1. All verdicts
// carry the same aggregate digest, so the one transport digest chain still
// covers the complete physical pass.
void device_pick_verdict_batched(
    const uint16_t* table, int rows, int world, int rank, const int64_t* fed,
    const int64_t* positions, int requests, int rows_per_request,
    int position_stride, PickVerdict* verdicts,
    PickVerdict* device_verdicts, uint64_t* carry_digest,
    cudaStream_t stream);

// The host mirror of the kernels' digest (the tests' oracle).
uint64_t device_pick_digest(int rows, int accepted, const int32_t* winners);

}  // namespace dgpp
