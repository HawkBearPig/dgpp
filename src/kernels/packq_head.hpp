#pragma once
// The packed int8 head in bit planes (2026-09-29; docs/qwen38_autoround_int4_plan.md
// §6.6, github issue #69): each 128-code group of a row is stored as
//   [64 B: the codes' high nibbles][32 B: bits 3..2 of each code][32 B: bits 1..0]
// so a pass over the high six bits reads three of the group's four
// 32-byte sectors. For an argmax-only row set (a greedy request's draft
// and verify heads) that pass gives every head row an interval that
// provably holds its logit — the centered six-bit dot A plus/minus
// B = 1.5 * sum_g |s_g| sum_{k in g} |x_k| — and only the rows whose
// interval reaches the best lower bound are read in full; those rows'
// logits are the packed core's chain bit for bit, every other row's
// logit is -inf, so the pick's argmax and its logit are the full read's.
// Sampled rows (temperature, logprobs, penalties, bias, grammar) read all
// three planes and are the packed core's chain bit for bit everywhere.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "models/quant_matrix.hpp"

namespace dgpp {

// Host: permute a row-major packed int8 matrix (payload [rows][cols] code
// bytes, cols a multiple of 128) into the plane layout, in place.
void packq_planes_permute_int8(uint8_t* payload, int64_t rows, int64_t cols);

// Which rows of a launch want only the argmax: greedy[slot] != 0 says a
// slot's logits may be the argmax-only form, and row r of the launch
// belongs to slot row_slots[r]. The table is the walk's own per-row
// request ids, the one its attention and state read, so a row's slot never
// depends on the shape of the call: a scalar step, a batch over a slot
// prefix or over a compacted map, a one-row-per-request chain. A negative
// entry is a batch's padding row and takes the full read. A null
// `row_slots` gives every row slot `req0`. A null `greedy` means every row
// takes the full read.
struct PackqHeadMode {
  const uint8_t* greedy = nullptr;
  const int32_t* row_slots = nullptr;  // device [m]
  int req0 = 0;
};

// Device scratch for the argmax pass: ab [max_rows][n] fp32 (the upper
// bounds), lo [max_rows] int32 (the best lower bound, as an ordered int).
constexpr int kPackqHeadChunkRows = 4;  // activation rows per launch chunk (the scratch's rows)

struct PackqHeadScratch {
  float* ab = nullptr;
  int32_t* lo = nullptr;
  int32_t* cand = nullptr;        // [n] the candidate rows (compacted)
  int32_t* cand_count = nullptr;  // [1]
};

// D[m, n] = Act[m, k] x W[n, k]^T, W a plane-layout int8 g128 matrix
// (scale format kPackedScaleF16G128), fp32 out. Rows are chunked four at
// a time; a chunk whose rows are all greedy runs the argmax form. Three
// kernel launches per chunk (kernel nodes only: capture-safe).
// `out_ids` (device [n], optional) scatters row r's logit to column
// out_ids[r] of an out_n-wide output row whose other columns are set to
// -inf first (the draft vocabulary slice); null: column r of an n-wide row.
void launch_packq_head_f32(const uint16_t* act, size_t act_row_stride_elems, const GlmPackedMatrix& w,
                           float* out, int m, int n, int k, const PackqHeadMode& mode,
                           const PackqHeadScratch& scratch, cudaStream_t stream,
                           const int32_t* out_ids = nullptr, int out_n = 0);

}  // namespace dgpp
