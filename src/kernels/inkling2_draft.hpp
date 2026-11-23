#pragma once
// Kernels of the Inkling 2 draft model (docs/inkling2_mini_plan.md section
// 1.7). The reference is the released draft, vLLM's
// models/inkling2/nvidia/dspark.py with ops/dspark.py: target features mixed
// per draft layer into context keys and values, a block of an anchor row and
// mask rows through three dense layers, and a recurrent head that corrects
// the block's base logits one proposal after another.
//
// Every kernel is deterministic and capturable. Interior math is FP32, and a
// value is rounded to BF16 at the points the released kernels round it.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

struct SampleSpec;

constexpr int kInk2DraftMaxTaps = 8;
constexpr int kInk2DraftMaxRank = 512;

// The features of `rows` target rows mixed with one row of softmax weights
// and normalized:
//   m = sum_t alpha[t] * tap_t[r]              FP32, the taps in order
//   out[r] = bf16(m * rsqrt(mean(m^2) + eps) * w)
struct Ink2DraftMixArgs {
  const uint16_t* tap[kInk2DraftMaxTaps] = {};  // BF16 [rows, hidden] each
  int taps = 0;
  const float* alpha = nullptr;                 // device F32 [taps]
  const uint16_t* w = nullptr;                  // BF16 [hidden]
  uint16_t* out = nullptr;                      // BF16 [rows, hidden]
  int rows = 0;
  int hidden = 0;
  float eps = 1e-6f;
};
void ink2_draft_mix_norm(const Ink2DraftMixArgs& args, cudaStream_t stream);

// The block's mask rows: every row but the first of each block of
// `block_rows` takes the learned mask row. The first row, the anchor, keeps
// what the caller wrote there (the target's normed embedding of its token).
void ink2_draft_mask_rows(const uint16_t* mask, uint16_t* out, int rows, int block_rows, int hidden,
                          cudaStream_t stream);

// The fused residual add and norm of the draft's layers:
//   s = residual + add in FP32, residual = bf16(s), out = bf16(s * rsqrt(mean(s^2) + eps) * w)
// The norm reads the sum before it is rounded.
void ink2_draft_add_norm(uint16_t* residual, const uint16_t* add, const uint16_t* w, uint16_t* out, int rows,
                         int hidden, float eps, cudaStream_t stream);

// Per-head RMSNorm and rotary embedding of `heads` heads of 128 values per
// row, as the released kernel orders them:
//   n = bf16(x * rsqrt(mean(x^2) + eps) * w)        per head
//   out[i]      = bf16(n[i] * cos_i - n[i + 64] * sin_i)
//   out[i + 64] = bf16(n[i + 64] * cos_i + n[i] * sin_i),  the angle pos * inv_freq[i] in FP32
// Head h of row r is at x + r * x_stride + h * 128, and out likewise. out
// may alias x. A row with a negative position is left untouched.
void ink2_draft_norm_rope(const uint16_t* x, int64_t x_stride, const uint16_t* w, const int64_t* pos,
                          const float* inv_freq, uint16_t* out, int64_t out_stride, int rows, int heads,
                          float eps, cudaStream_t stream);

// One request's feature row out of its positional ring: the `width` values
// at `offset` of the slot of position block_pos[0] - 1, the last committed
// position. Zeros when no position precedes the block.
void ink2_draft_ring_row(const uint16_t* ring_request, int64_t slot_stride, int64_t offset, int ring_slots,
                         const int64_t* block_pos, uint16_t* out, int width, cudaStream_t stream);

// ---- the recurrent head --------------------------------------------------------
//
// rnn_w is BF16 [3 rank, 2 rank + 2 hidden]: the columns of [state |
// previous token's row | block hidden | context feature]. A proposal step
// reads the first 2 rank columns, and the last 2 hidden columns are applied
// to all steps at once:
//   context[s * steps + i][j] = bias[j] + hidden[s, row_offset + i] . W[j, 2r : 2r + H]
//                                       + feature[s] . W[j, 2r + H :]
// hidden is the block's final rows, `block_rows` per slot.
void ink2_draft_markov_context(const uint16_t* hidden, int block_rows, int row_offset, const uint16_t* feature,
                               const uint16_t* rnn_w, const float* rnn_bias, float* context, int slots, int steps,
                               int hidden_size, int rank, cudaStream_t stream);

// One step of every slot's recurrence, the released gate arithmetic with
// each of its operations rounded to BF16:
//   raw = bf16(context[s, step] + [state_s | w1[prev_s]] . W[:, : 2r])   one FP32 chain
//   gate = bf16(sigmoid(raw[0 : r])), cand = bf16(tanh(raw[r : 2r]))
//   state_s = bf16(bf16(gate * state_s) + bf16(bf16(1 - gate) * cand))
//   latent_s = bf16(tanh(raw[2r : 3r]))
// The state before step 0 is zero. prev_s is prev[s * prev_stride]. Two
// launches: the projection, one block per output and slot, then the gates.
// `raw` is FP32 scratch [slots, 3 rank]. Nothing is written while *stop is
// set (see ink2_draft_stop).
void ink2_draft_markov_step(const float* context, int steps, int step, const uint16_t* rnn_w, int hidden_size,
                            const uint16_t* w1, const int64_t* prev, int prev_stride, uint16_t* state,
                            uint16_t* latent, float* raw, const int32_t* stop, int slots, int rank,
                            cudaStream_t stream);

// The step's logits over this rank's vocabulary rows, the released
// add-and-softcap. On entry logits[s, v] holds the latent's product with the
// folded head, latent_s . fold_w[v] in FP32 (a GEMM over the folded matrix).
// On return:
//   bias = bf16(fold_b[v] + logits[s, v])
//   logits[s, v] = bf16(cap * tanh(bf16(bf16(base[s, v]) + bias) / cap))
// base row s is base[s * base_slot_stride + v]. cap 0 leaves the sum uncapped.
// Nothing is written while *stop is set.
void ink2_draft_markov_logits(const float* base, int64_t base_slot_stride, int64_t vocab, const uint16_t* fold_b,
                              float cap, float* logits, const int32_t* stop, int slots, cudaStream_t stream);

// A proposal's confidence as an acceptance logit. The draft has no
// confidence head, so the estimate is read off the step's `k` merged
// candidates: the logit of the chosen candidate's softmax mass among them,
// clamped to [-30, 30], through a linear fit. The draft's own mass
// overstates acceptance (a proposal it gives 94 % was accepted 73 % of the
// time on the release), and the fit below is the logistic regression of
// accepted against that logit over 2,443 verified proposals of five chat
// prompts (code, prose, JSON) under greedy decoding. It predicts within
// three points of the measured rate in every bin from 0.25 to 0.98. -30
// when the token is not among the candidates.
struct Ink2DraftConfidenceFit {
  float bias = -0.64f;
  float slope = 0.568f;
};
// While *stop is set the step did not run and its confidence is -30.
// Otherwise the slot's running survival (the chance every proposal so far is
// accepted) is multiplied by this proposal's estimate, unless `spec` names a
// sampled request (temperature above zero), whose proposals are verified
// whatever their estimate.
void ink2_draft_confidence(const float* scores, const int32_t* ids, const int32_t* token, int k,
                           const Ink2DraftConfidenceFit& fit, float* conf, const int32_t* stop, float* survival,
                           const SampleSpec* spec, cudaStream_t stream);

// Stopping the recurrent head early. A verify schedule checks proposal i only
// while the mean over the pass's slots of their survival through i is above
// its threshold, and survival only falls, so once the mean is below a floor
// under that threshold no later proposal can be verified and the later steps
// are skipped. ink2_draft_stop_reset clears the flag and sets the slots'
// survival to one at the start of a block. ink2_draft_stop, after a step's
// confidences, sets the flag when the mean survival is below floor[0]. A
// floor of zero never stops.
void ink2_draft_stop_reset(int32_t* stop, float* survival, int slots, cudaStream_t stream);
void ink2_draft_stop(const float* survival, int slots, const float* floor, int32_t* stop, cudaStream_t stream);

// The head folded through the recurrent head's output projection, in FP64
// and rounded once, as the reference folds it at load:
//   fold_w[v, j] = bf16(sum_d head[v, d] * out_w[d, j]),  fold_b[v] = bf16(sum_d head[v, d] * out_b[d])
// head BF16 [vocab, hidden], out_w BF16 [hidden, rank], out_b BF16 [hidden].
void ink2_draft_fold_head(const uint16_t* head, const uint16_t* out_w, const uint16_t* out_b, uint16_t* fold_w,
                          uint16_t* fold_b, int64_t vocab, int hidden, int rank, cudaStream_t stream);

}  // namespace dgpp
