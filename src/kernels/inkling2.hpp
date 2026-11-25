#pragma once
// Row-wise kernels of the Inkling 2 family (docs/inkling2_mini_plan.md
// sections 1.2 to 1.6): the attention-residual mix with its sublayer norm,
// the loop controllers' pair norm and blend, the short causal convolution
// over a positional ring, the experts' softcapped activation and the two
// activation quantizers, and the logit softcap.
//
// Every kernel is deterministic, capturable and row-independent: a row's
// bits do not depend on the rows that share its launch. Interior math is
// FP32 and a value is rounded once where it is stored.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

constexpr int kInk2MaxSources = 8;   // completed blocks plus the partial state
constexpr int kInk2MaxQueries = 4;
constexpr int kInk2ConvTaps = 4;
// Positions a request's convolution ring holds. A decode launch may write
// its rows in place while reading the three positions before them, so the
// ring covers the widest speculative step plus the taps with room to spare.
constexpr int kInk2RingSlots = 16;
// The source normalization inside every mix and controller.
constexpr float kInk2MixEps = 1e-6f;

// One mix per (row, query): the softmax over the sources' scores
// (q . x_n) * rstd_n, the sum of the normalized sources, then that
// sublayer's RMSNorm rounded to BF16, or the FP32 mix when no norm weight
// is given. The sources are the completed blocks and one last row: the
// partial state, plus `addend` in FP32 when set.
struct Ink2MixArgs {
  const uint16_t* stack[kInk2MaxSources] = {};  // BF16 [rows, hidden] each
  int n_stack = 0;
  const uint16_t* last = nullptr;               // BF16 [rows, hidden]
  const uint16_t* addend = nullptr;             // BF16 [rows, hidden] or null
  const float* queries = nullptr;               // F32 [n_queries, hidden], scaled by hidden^-0.5
  int n_queries = 1;
  const uint16_t* norm_w[kInk2MaxQueries] = {}; // BF16 [hidden] per query, all set or all null
  uint16_t* out_bf16[kInk2MaxQueries] = {};     // BF16 [rows, hidden] per query (with norm weights)
  float* out_f32[kInk2MaxQueries] = {};         // F32 [rows, hidden] per query (without)
  float norm_eps = 1e-6f;
  int rows = 0;
  int hidden = 0;
};
void ink2_mix(const Ink2MixArgs& args, cudaStream_t stream);

// pair[r] = [bf16(dest * rstd), bf16(message * rstd)], BF16 [rows, 2 x
// hidden], each half RMS-normalized without a weight.
void ink2_pair_norm(const float* dest, const float* message, uint16_t* pair, int rows, int hidden,
                    cudaStream_t stream);

// out = bf16(rmsnorm((1 - tanh(forget)) * dest + tanh(gain) * message) * w).
// gain_forget: F32 [rows, 2 x hidden], the gain half first.
void ink2_blend_norm(const float* dest, const float* message, const float* gain_forget, const uint16_t* norm_w,
                     uint16_t* out, int rows, int hidden, float norm_eps, cudaStream_t stream);

// The short causal convolution of one stream:
//   y[r, c] = (sum_lag (taps[c, lag] + dyn[r, head(c), lag]) * x[position - lag, c] + x[r, c]) * scale
// Lag 0 is the row itself. An earlier position comes from the launch's own
// rows when the row `lag` before is that position of the same request, from
// the request's ring otherwise, and is zero before position 0. A row with a
// negative position writes zeros and no state.
struct Ink2ConvArgs {
  const uint16_t* x = nullptr;        // BF16 [rows, x_stride], the stream's channels from x_offset
  int64_t x_stride = 0;
  int64_t x_offset = 0;
  const float* taps = nullptr;        // F32 [channels, 4] by lag
  const uint16_t* dyn = nullptr;      // BF16 [rows, dyn_stride] per-row tap deltas at dyn_offset + head * 4 + lag, or null
  int64_t dyn_stride = 0;
  int64_t dyn_offset = 0;
  int dyn_heads = 0;
  float scale = 0.0f;                 // (W + 1)^-0.5 static, (2 W + 1)^-0.5 dynamic
  uint16_t* ring = nullptr;           // BF16 [requests][kInk2RingSlots][ring_slot_stride]
  int64_t ring_request_stride = 0;    // elements per request
  int64_t ring_slot_stride = 0;       // elements per slot
  int64_t ring_offset = 0;            // the stream's element offset inside a slot
  const int32_t* req_ids = nullptr;   // device [rows]
  const int64_t* pos = nullptr;       // device [rows]
  int rows = 0;
  int channels = 0;
  // Store every row's input at its ring slot in this launch. Only for
  // launches whose runs of one request are at most kInk2RingSlots - 3 rows
  // (decode and verification rows). Longer runs store with ink2_ring_store
  // after the launch.
  bool store_inline = false;
  uint16_t* out = nullptr;            // BF16 [rows, out_stride], the channels from out_offset
  int64_t out_stride = 0;
  int64_t out_offset = 0;
  // The MLP stream's epilogue: out = bf16((residual + addend) + bf16(y)).
  const uint16_t* addend = nullptr;   // BF16 [rows, channels] or null
  const uint16_t* residual = nullptr; // BF16 [rows, channels] or null (a block's first layer)
};
void ink2_conv(const Ink2ConvArgs& args, cudaStream_t stream);

// Stores the stream's inputs of the launch's rows at their ring slots. A
// row whose slot a later row of the same run takes is skipped, so every
// slot has one writer.
void ink2_ring_store(const uint16_t* x, int64_t x_stride, int64_t x_offset, uint16_t* ring,
                     int64_t ring_request_stride, int64_t ring_slot_stride, int64_t ring_offset,
                     const int32_t* req_ids, const int64_t* pos, int rows, int channels, cudaStream_t stream);

// A prefix snapshot of one request's rings: the position, then the three
// slots before it, newest first ([8-byte position][8 bytes of padding][3 x
// slot_stride BF16]). A slot before position 0 is stored as zeros.
inline size_t ink2_ring_snapshot_bytes(int64_t slot_stride) { return 16 + 3 * static_cast<size_t>(slot_stride) * 2; }
void ink2_ring_snapshot(const uint16_t* ring_request, int64_t slot_stride, int64_t position, uint8_t* snapshot,
                        cudaStream_t stream);
// Restores a snapshot into a request's rings at the slots its stored
// position names.
void ink2_ring_restore(const uint8_t* snapshot, int64_t slot_stride, uint16_t* ring_request, cudaStream_t stream);

// row_slot[slot_row[s]] = s for every routed slot s: the gathered row's
// flat (token, slot) index.
void ink2_slot_rows(const int32_t* slot_row, int slots, int32_t* row_slot, cudaStream_t stream);

// The experts' activation over gathered rows:
//   g = bf16(gate_up[r, 2 i] * s_r), u = bf16(gate_up[r, 2 i + 1] * s_r)
//   act[r, i] = bf16(softcap(g, gate_cap) * sigmoid(g) * softcap(u, up_cap) * w_r)
// softcap(x, c) = c * tanh(x / c) (c = 0: x). s_r = row_scale[row_token[r]]
// (null row_scale: 1) and w_r = slot_w[row_slot[r]].
void ink2_swiglu(const float* gate_up, int64_t gate_up_stride, int rows, int inter, const float* row_scale,
                 const int32_t* row_token, const float* slot_w, const int32_t* row_slot, float gate_cap,
                 float up_cap, uint16_t* act, int64_t act_stride, cudaStream_t stream);

// NVFP4 activation quantization with the four-over-six block scale choice,
// left in its exact dequantized inner form. For a row with maximum
// magnitude amax: outer = amax / 1536. For each 16-value block, the E4M3
// block scale that puts the block maximum at 6 and the one that puts it at
// 4 are both tried and the one with the smaller summed absolute
// reconstruction error is kept (6 on a tie).
//   inner[r, c] = e2m1(x * inverse) * block_scale   exact in BF16
//   x[r, c] ~ inner[r, c] * outer_r
// The float operations follow the released quantizer in order, so codes
// and scales are its bytes. outer[scale_index[r]] (null index: outer[r])
// receives outer_r. k is a multiple of 16.
void ink2_quantize_nvfp4(const uint16_t* x, int64_t x_stride, int rows, int k, uint16_t* inner,
                         int64_t inner_stride, float* outer, const int32_t* scale_index, cudaStream_t stream);

// The same quantizer in the format's own bytes, for the tensor cores'
// block-scaled form: codes [rows, k / 2] (E2M1 pairs, the even element in
// the low nibble), scales [rows, scale_stride] (one E4M3 per 16 values) and
// gs[r] = amax / 1536, with x[r, c] ~ e2m1 * scale * gs[r].
void ink2_quantize_nvfp4_codes(const uint16_t* x, int64_t x_stride, int rows, int k, uint8_t* codes, uint8_t* scales,
                               int64_t scale_stride, float* gs, cudaStream_t stream);

// The experts' activation from BF16 gate / up rows that already carry their
// row scale: act[r, i] as ink2_swiglu writes it.
void ink2_swiglu_bf16(const uint16_t* gate_up, int64_t gate_up_stride, int rows, int inter, const float* slot_w,
                      const int32_t* row_slot, float gate_cap, float up_cap, uint16_t* act, int64_t act_stride,
                      cudaStream_t stream);
// The same activation quantized in the same pass: ink2_swiglu_bf16 then
// ink2_quantize_nvfp4_codes over the routed rows, without the activation
// buffer between them.
void ink2_swiglu_quantize_nvfp4(const uint16_t* gate_up, int64_t gate_up_stride, int rows, int inter,
                                const float* slot_w, const int32_t* row_slot, float gate_cap, float up_cap,
                                uint8_t* codes, uint8_t* scales, int64_t scale_stride, float* gs,
                                cudaStream_t stream);

// out[t] = bf16(the sum over the token's slots, in ascending expert id, of
// bf16(down[slot_row[s]] * slot_scale[s])): an expert's output is rounded at
// its row scale before the sum. `positions` (nullable, one per token): a
// padding token (position < 0) is zero, as launch_moe_segment leaves it in
// no expert's row list.
void ink2_moe_accum_rounded(uint16_t* out, const float* down, int64_t down_stride, const int32_t* slot_row,
                            const int32_t* slot_ids, const float* slot_scale, int tokens, int top_k, int hidden,
                            cudaStream_t stream, const int64_t* positions = nullptr);

// MXFP8 activation quantization in its exact dequantized form: per 32-value
// block a power-of-two scale that covers the block maximum, the values
// rounded to E4M3. out may alias x. k is a multiple of 32.
void ink2_requantize_mxfp8(const uint16_t* x, int64_t x_stride, int rows, int k, uint16_t* out,
                           int64_t out_stride, cudaStream_t stream);

// logits[i] = cap * tanh(logits[i] / cap) in place. A non-finite value is
// left as it is.
void ink2_logits_softcap(float* logits, int64_t count, float cap, cudaStream_t stream);

// Loop guidance (engine.loop_guidance): a logit of the last pass moved away
// from the same logit read out of an earlier pass,
//   guided = (1 + w) * last - w * earlier,
// in one rounding, so w = 0 returns `last` and w = -1 returns `earlier` bit
// for bit.
//
// In place over [rows, vocab]. A row's w is request_strength[row_request[r]]
// (null row_request: request 0; a request outside [0, requests) leaves the
// row as it is). A non-finite value is left as it is. One block per row:
// rows <= 65535. The host statement is the kernel's arithmetic.
void ink2_logits_guide(float* logits, const float* reference, int rows, int64_t vocab, const int32_t* row_request,
                       const float* request_strength, int requests, cudaStream_t stream);
float ink2_guided_value_host(float strength, float last, float earlier);

// Host statements of the two quantizers (the kernels' arithmetic operation
// for operation), for tests and tools.
void ink2_quantize_nvfp4_host(const uint16_t* x, int k, uint16_t* inner, float* outer);
void ink2_requantize_mxfp8_host(const uint16_t* x, int k, uint16_t* out);

}  // namespace dgpp
