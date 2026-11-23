#pragma once
// The Inkling 2 attention kernels (docs/inkling2_mini_plan.md section 1.4):
// paged GQA attention over 128-wide heads with a learned relative bias on
// the nearest keys, one sink logit per query and head, and a sigmoid gate on
// the output. There is no rotary embedding. Position enters through the bias
// and through the short convolutions on the keys and values.
//
//   query finish  per (row, head): q = bf16(l2_clip(q * q_scale)), on a
//                 full-attention layer then bf16(q * tau). r likewise over
//                 its d_rel dimensions, then the bias row
//                 bias[d] = bf16(r . rel_proj[:, d]) for distances d in
//                 [0, extent)
//   kv finish     per (row, kv head): the short convolution of the K and V
//                 projections (static taps, or static plus per-row taps on a
//                 full-attention layer), K = bf16(l2_clip(k)), V = bf16(v *
//                 rstd), both appended to the paged caches at the row's
//                 slot. The convolution inputs go to the request's ring
//   partial       split-KV attention as the sink family runs it: FP32 scores
//                 q . k * scale + bias[position - key], an online softmax,
//                 probabilities rounded to BF16 for the V accumulation, the
//                 denominator unrounded. The row's sink logit enters split
//                 0's running maximum and sum before its first tile
//   combine       the splits merged, out = bf16(C / L * sigmoid(gate))
//
// A local layer attends [max(0, pos - window + 1), pos]. A row with a
// negative position is padding: it writes no cache row and no state.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

constexpr int kInk2HeadDim = 128;
constexpr int kInk2AttnTile = 32;
constexpr int kInk2MaxRelDim = 32;

// The fused query-side projection's BF16 row: [q (lh x 128) | r (lh x
// d_rel) | gate (lh x 128) | sink (lh)].
struct Ink2QueryArgs {
  const uint16_t* q_side = nullptr;
  int64_t q_side_stride = 0;
  const float* q_scale = nullptr;   // F32 [lh, 128]
  const float* r_scale = nullptr;   // F32 [lh, d_rel]
  const float* rel_proj = nullptr;  // F32 [d_rel, extent]
  const int64_t* pos = nullptr;     // device [rows]
  int rows = 0;
  int local_heads = 0;
  int d_rel = 16;
  int extent = 0;
  float qk_clip = 0.0f;
  float rel_clip = 0.0f;
  bool log_scaling = false;         // tau = 1 + alpha * ln(max((pos + 1) / n_floor, 1))
  float log_alpha = 0.0f;
  float log_n_floor = 1.0f;
  uint16_t* q_out = nullptr;        // BF16 [rows, lh x 128]
  uint16_t* bias = nullptr;         // BF16 [rows, lh, extent]
};
void ink2_query_finish(const Ink2QueryArgs& args, cudaStream_t stream);

struct Ink2KvArgs {
  const uint16_t* k_in = nullptr;   // BF16 [rows, kv_stride]: lkv x 128 columns, then lkv x 4 tap deltas when dynamic
  const uint16_t* v_in = nullptr;
  int64_t kv_stride = 0;
  const float* k_taps = nullptr;    // F32 [lkv x 128, 4] by lag
  const float* v_taps = nullptr;
  bool dynamic = false;
  float conv_scale = 0.0f;
  uint16_t* ring = nullptr;         // BF16 [requests][kInk2RingSlots][ring_slot_stride]
  int64_t ring_request_stride = 0;
  int64_t ring_slot_stride = 0;
  int64_t ring_offset_k = 0;
  int64_t ring_offset_v = 0;
  bool store_inline = false;        // see Ink2ConvArgs::store_inline
  const int32_t* req_ids = nullptr;
  const int64_t* pos = nullptr;
  int rows = 0;
  int kv_heads = 0;
  float qk_clip = 0.0f;
  float eps = 1e-6f;
  const int32_t* block_tables = nullptr;
  int blocks_per_request = 0;
  int block_tokens = 0;
  uint16_t* k_cache = nullptr;      // BF16 [slots, lkv x 128]
  uint16_t* v_cache = nullptr;
};
void ink2_kv_finish(const Ink2KvArgs& args, cudaStream_t stream);

struct Ink2AttnArgs {
  const uint16_t* q = nullptr;      // BF16 [rows, lh x 128]
  int64_t q_stride = 0;
  const uint16_t* bias = nullptr;   // BF16 [rows, lh, extent]
  int extent = 0;
  const uint16_t* sink = nullptr;   // BF16, the row's lh sink logits at sink[r * sink_stride + h]
  int64_t sink_stride = 0;
  const uint16_t* k_cache = nullptr;
  const uint16_t* v_cache = nullptr;
  const int32_t* req_ids = nullptr;
  const int64_t* pos = nullptr;
  int rows = 0;
  int n_split = 1;
  int local_heads = 0;
  int kv_heads = 0;
  int block_tokens = 0;
  const int32_t* block_tables = nullptr;
  int blocks_per_request = 0;
  int window = 0;                   // 0: the whole context
  float scale = 1.0f;
  float* m_ws = nullptr;            // F32 [rows, n_split, lh]
  float* l_ws = nullptr;
  float* c_ws = nullptr;            // F32 [rows, n_split, lh, 128]
};
void ink2_attn_partial(const Ink2AttnArgs& args, cudaStream_t stream);

// The reader rows' form (kernels/inkling2_attn_prefill.cu): a block owns a
// tile of query rows x one key/value head's query heads, stages every K/V
// tile once for all of them and runs the scores and the value products on
// the tensor cores. The same scores, bias, sink and probabilities as the
// split form with the tensor core's summation order, and the gate applied in
// the epilogue: out[r, h x 128 + d] as ink2_attn_combine writes it.
struct Ink2AttnPrefillArgs {
  const uint16_t* q = nullptr;      // BF16 [rows, lh x 128]
  int64_t q_stride = 0;
  const uint16_t* bias = nullptr;   // BF16 [rows, lh, extent]
  int extent = 0;
  const uint16_t* sink = nullptr;   // BF16 at sink[r * sink_stride + h]
  int64_t sink_stride = 0;
  const uint16_t* gate = nullptr;   // BF16 gate logits at gate[r * gate_stride + h x 128 + d]
  int64_t gate_stride = 0;
  const uint16_t* k_cache = nullptr;
  const uint16_t* v_cache = nullptr;
  const int32_t* req_ids = nullptr;
  const int64_t* pos = nullptr;
  int rows = 0;
  int local_heads = 0;
  int kv_heads = 0;
  int block_tokens = 0;
  const int32_t* block_tables = nullptr;
  int blocks_per_request = 0;
  int window = 0;                   // 0: the whole context
  float scale = 1.0f;
  uint16_t* out = nullptr;          // BF16 [rows, lh x 128]
};
void ink2_attn_prefill(const Ink2AttnPrefillArgs& args, cudaStream_t stream);
int ink2_attn_prefill_rows_per_block(int local_heads, int kv_heads);

// out[r, h x 128 + d] = bf16(C / L * sigmoid(gate[r * gate_stride + h x 128 + d])),
// 0 when every split is empty. out rows are lh x 128 wide.
void ink2_attn_combine(const float* m_ws, const float* l_ws, const float* c_ws, const uint16_t* gate,
                       int64_t gate_stride, int rows, int n_split, int local_heads, uint16_t* out,
                       cudaStream_t stream);

}  // namespace dgpp
