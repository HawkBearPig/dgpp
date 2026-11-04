#pragma once
// DFlash2 drafter kernels (docs/performance_improvement_plan.md §7). The
// reference is the released vLLM DFlash2 speculator
// (vllm/model_executor/models/qwen3_dflash2.py and
// vllm/v1/worker/gpu/spec_decode/dflash2/): a block-diffusion drafter whose
// KV context is built from fused target features and whose block of mask
// positions is denoised in ONE bidirectional pass, with two-tap dynamic
// grouped convolutions around attention and the MLP and a codebook path
// selector over the top-K per-position candidates.
//
// All kernels are deterministic; bf16 io (uint16_t bits), fp32 interiors,
// the reference's rounding points named per kernel.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "kernels/sample_pick.hpp"

namespace dgpp {

// The 2-tap dynamic grouped convolution (the DFlash2 attention_conv /
// mlp_conv, one side's slice). x and out are [rows, hidden] bf16 rows;
// base is the side's [taps, hidden] bf16 slice and delta the rows'
// [rows, taps, groups] fp32-of-bf16 slice (row stride in elements — the
// kernel_projection's [rows, 2, taps, groups] output sliced per side).
// out[r, c] = sum_t bf16(base[t, c] + delta[r, t, c/group]) * x[r-t, c],
// the tap t applying only when r % block_rows >= t (the conv lives INSIDE
// a request's query block, position = r % block_rows). fp32 accumulation,
// one bf16 rounding.
void dflash2_grouped_conv_bf16(const uint16_t* x, const uint16_t* delta, const uint16_t* base,
                               uint16_t* out, int rows, int block_rows, int hidden, int taps,
                               int group_size, int64_t delta_row_stride, cudaStream_t stream);

// out += src (fp32, elementwise, fixed order). beta 0: out = src. The
// per-tap fc accumulation: the fused target features
// sum_t tap_t @ fc_t^T computed as one fp32 GEMM per tap layer.
void dflash2_acc_f32(float* acc, const float* src, int64_t n, int beta, cudaStream_t stream);

// The standard (plain-weight) RMSNorm — the drafter is a Qwen3 model, not
// the (1+w) zero-centered family form: y = bf16(f32(x) * rsqrt(mean+eps) * w).
void dflash2_rmsnorm_bf16(const uint16_t* x, const uint16_t* w, uint16_t* y, int64_t rows, int dim,
                          float eps, cudaStream_t stream);
// Fused residual add + the standard norm (bitwise the add + norm pair).
void dflash2_add_rmsnorm_bf16(uint16_t* resid, const uint16_t* add, const uint16_t* w, uint16_t* y,
                              int64_t rows, int dim, float eps, cudaStream_t stream);
// The same norm over the fp32 fused-feature accumulator.
void dflash2_norm_f32_bf16(const float* x, const uint16_t* w, uint16_t* y, int64_t rows, int dim,
                           float eps, cudaStream_t stream);

// Per-head standard RMSNorm + RoPE (rotate_half / neox pairs, the bf16
// three-rounding form: bf16(x*cos), bf16(rot*sin), one add rounding).
// head h of row r at x + r*x_row_stride + h*dim; out likewise. Rows with
// pos < 0 are skipped.
void dflash2_norm_rope_bf16(const uint16_t* x, int64_t x_row_stride, const uint16_t* w,
                            const int64_t* pos, const float* inv_freq, uint16_t* out,
                            int64_t out_row_stride, int rows, int heads, int dim, float eps,
                            cudaStream_t stream);

// The drafter's block attention: rows query rows (one request, whole
// blocks of block_rows), q bf16 [rows, heads, dim] (row stride
// q_row_stride, heads contiguous); GQA kv_heads; the paged planes as
// Qwen35KvPool views (bf16 [slots, kv_heads*dim], one physical block
// table). The block's span comes off its first row's position p0 =
// pos[row - row % block_rows] (device-read, so a recorded draft takes the
// committed position): a key at position p is visible to the query at q
// from either the context span [0, p0 - 1] with q - p < window (causal,
// sliding), or the block span [p0, p0 + block_rows - 1] with no causal and
// no window restriction (the block is bidirectional and far narrower than
// any window). A row whose position (or whose block's p0) is negative
// writes zeros. fp32 online softmax, bf16 probabilities into V (the house
// rule), fp32 output rounded to bf16.
void dflash2_block_attn(const uint16_t* q, int64_t q_row_stride, const uint16_t* k_cache,
                        const uint16_t* v_cache, const int32_t* block_table, int block_tokens,
                        int blocks_per_request, int block_rows, int64_t window, const int64_t* pos,
                        int rows, int heads, int kv_heads, int dim, float scale, uint16_t* out,
                        cudaStream_t stream);
// The split-key form (2026-10-04): the same attention with each (row, kv
// head)'s key walk cut into dflash2_block_attn_splits() ranges that run in
// parallel, each leaving an unnormalized (m, l, acc) partial in `partials`
// (dflash2_block_attn_partials_bytes(rows, heads) bytes), merged by a
// combine pass — the serial walk over a 2K window cost ~2 ms a layer. One
// rescaling point differs from the serial form (tolerance-equal, not
// bitwise); deterministic on every rank.
int dflash2_block_attn_splits();
size_t dflash2_block_attn_partials_bytes(int rows, int heads);
void dflash2_block_attn_split(const uint16_t* q, int64_t q_row_stride, const uint16_t* k_cache,
                              const uint16_t* v_cache, const int32_t* block_table, int block_tokens,
                              int blocks_per_request, int block_rows, int64_t window,
                              const int64_t* pos, int rows, int heads, int kv_heads, int dim,
                              float scale, float* partials, uint16_t* out, cudaStream_t stream,
                              bool fp32_p = false);

// Per-row top-K of fp32 logits (descending by score, ties to the lower id):
// the candidate sets the selector walks. K <= 32.
// The partials workspace the chunked top-K needs for `rows` rows of `vocab`.
size_t dflash2_topk_ws_bytes(int64_t vocab, int rows, int k);
void dflash2_topk_f32(const float* logits, int32_t* ids, float* scores, int64_t vocab, int rows,
                      int k, cudaStream_t stream, void* ws, size_t ws_bytes);

// The DFlash2 candidate path selector: the scores[l][p][c] table
// unary[l][c] + <pred_code[id(l-1, p)] * hidden[l], succ_code[id(l, c)]>
// (step 0's predecessor is the anchor token *anchor — the block's first
// row token, device-read — every slot) walked greedily per step: token =
// ids[l][argmax_c scores[l][prev][c]], prev = that argmax (ties to the
// first). One block, steps sequential, scores within a step computed in
// parallel; the reduction order over the rank is fixed. cb bf16 [vocab, rank].
//
// The sampled walk (`spec` given and the request stochastic — temperature
// > 0, the reference's _selector_walk_kernel with SAMPLE_PROBABILISTIC):
// step l draws its candidate from softmax(scores[l][prev][.] / T), T the
// spec's draft_temperature (its temperature when 0), truncated by the
// spec's top-k / top-p / min-p (the MTP draft's convention; the dropped
// candidates keep mass 0 in the proposal), by the draft stream's
// uniform keyed on the draft's position pos[l + 1] (the same draw on every
// rank), and writes the set it drew from to proposal[l] (n = k, ids, mass,
// token = the draw) — the next verify's ratio rule, min(1, P/Q) with the
// (P - Q)+ residual, whose rate is 1 - TV(P, Q) rather than P(argmax)
// (kernels/sample_pick.hpp DraftProposal). A greedy spec walks the argmax
// and writes n = 0 (the plain P(draft) rule). `proposal_host` (pinned,
// optional) mirrors the proposals for the host's fallback. Without `spec`
// the walk is the argmax and no proposal is written.
//
// `conf` (optional, [steps]): the selector's confidence per step as an
// acceptance logit — logit of the chosen candidate's softmax mass at the
// walk's temperature (1 for the argmax walk, the draft temperature for a
// drawn one), clamped to [-30, 30] — the scheduled verify depth's input
// (engine/verify_schedule.hpp), as the MTP draft pick's own probability is.
void dflash2_selector_walk(const int32_t* ids, const float* unary, const float* hidden,
                           const uint16_t* pred_cb, const uint16_t* succ_cb, const int64_t* anchor,
                           int32_t* tokens, int steps, int k, int rank, cudaStream_t stream,
                           const int64_t* pos = nullptr, const SampleSpec* spec = nullptr,
                           DraftProposal* proposal = nullptr, DraftProposal* proposal_host = nullptr,
                           float* conf = nullptr);

// ---- the recorded block draft (the graph engine's block proposal) -------------
//
// The block's row inputs off a verify verdict: tokens[0] = verdict->next
// (the anchor), tokens[1..rows) = mask_id; pos[j] = *session_pos + j (the
// position the recorded commit advanced to), or -1 for a position at or
// past max_context (every state write skips it).
struct PickVerdict;
void dflash2_stage_block(const PickVerdict* verdict, const int64_t* session_pos, int64_t mask_id,
                         int rows, int64_t max_context, int64_t* pos, int64_t* tokens,
                         cudaStream_t stream);
// The slot's next feed: tokens[0] = verdict->next, tokens[1 + c] = drafts[c].
void dflash2_block_feed(const PickVerdict* verdict, const int32_t* drafts, int count,
                        int64_t* tokens, cudaStream_t stream);
// The drafts' pinned mirror (a kernel node: the host reads it after the
// replay's end event), system-scope release ordered.
void dflash2_publish_drafts(const int32_t* drafts, int32_t* pinned, int count, cudaStream_t stream);
// The same for the mask rows' candidate tables (ids [drafts x k], up to
// 1024 words): the engine's acceptance diagnostics read them.
void dflash2_publish_words(const int32_t* src, int32_t* pinned, int count, cudaStream_t stream);
// The fixed batch's forms over `requests` slots (slot q is request q): the
// stacked blocks' rows [q*rows, (q+1)*rows) off verdicts[q] and
// session_pos[q] (an inactive verdict — accepted 0 — stages positions -1
// and token 0); each slot's feed rows at feeds + q*rows ([next, drafts],
// zeros for an inactive slot); each slot's drafts into pinned + q*count.
void dflash2_stage_block_batched(const PickVerdict* verdicts, const int64_t* session_pos, int64_t mask_id,
                                 int rows, int requests, int64_t max_context, int64_t* pos,
                                 int64_t* tokens, cudaStream_t stream);
void dflash2_block_feed_batched(const PickVerdict* verdicts, const int32_t* drafts, int count, int requests,
                                int rows, int64_t* feeds, cudaStream_t stream);
void dflash2_publish_drafts_batched(const int32_t* drafts, int32_t* pinned, int count, int requests,
                                    cudaStream_t stream);

// The cross-rank top-K at a vocab-sharded head (world > 1): every rank
// holds its slice's per-row top-K; the lists ride the bus's bf16 SUM
// all-reduce as a disjoint-slot gather (kernels/pick.hpp's wire form:
// 6-bit digits in bf16, one nonzero contributor per slot, exact under the
// fp32-accumulated fold). Table [rows][world][k] slots of 9 digits (6 for
// the fp32 score bits, 3 for the global id < 2^18), padded to an even
// count. dflash2_topk_stage zeroes the table and writes this rank's slots
// (ids offset by vocab_begin); after the fold dflash2_topk_merge decodes
// every rank's candidates and writes each row's top-K in the canonical
// order (score desc, id asc) — identical on every rank.
constexpr int kDflash2TopkDigits = 9;
constexpr size_t dflash2_topk_table_elems(int rows, int k, int world) {
  const size_t slots = static_cast<size_t>(rows) * static_cast<size_t>(world) *
                       static_cast<size_t>(k) * static_cast<size_t>(kDflash2TopkDigits);
  return slots + (slots & 1);
}
void dflash2_topk_stage(const int32_t* ids, const float* scores, int rows, int k, int32_t vocab_begin,
                        int rank, int world, uint16_t* table, cudaStream_t stream);
void dflash2_topk_merge(const uint16_t* table, int rows, int k, int world, int32_t* ids,
                        float* scores, cudaStream_t stream);

}  // namespace dgpp
