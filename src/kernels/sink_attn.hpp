#pragma once
// Windowed, sink-biased paged GQA attention over a fused fp32 q/k/v
// projection — the kernels MiMo-V2.6-Flash landed (2026-09-22,
// docs/mimo_v26_flash_plan.md §1.3, §7.1, §7.2), as a family over the head
// geometry (2026-10-22, docs/flint_plan.md D3):
//
//   qk_dim 192 (MiMo)  qk 192 / v 128, a 64-wide rotary slice (pairs (i,
//                      i + 32)) rotated with the reference's bf16 ops, a
//                      value scale before the cache, no head norm;
//   qk_dim 128 (Flint) q / k / v 128 wide, an optional per-head one-rounding
//                      RMSNorm on q and k (q_norm / k_norm), RoPE over the
//                      whole head (pairs (i, i + 64)) with fp32 cos / sin
//                      and one rounding — or none (inv_freq null: the
//                      no-RoPE full-attention layers).
//
// The pieces (one kernel source, kernels/sink_attn_impl.cuh, instantiated
// per geometry):
//   finish      per (row, head): bf16(dot) — the Linear's one rounding —
//               the head norm, the RoPE, the q heads to a bf16 buffer and
//               the k / v heads appended to the paged caches at each row's
//               slot (the v heads scaled when a value scale is set). The
//               fp32 source is the fused projection's output in a chunk
//               layout (SinkQkvLayout): `chunks` chunks of `chunk_stride`
//               columns, each [Q (q_per_chunk x qk) | K (kv_per_chunk x
//               qk) | V (kv_per_chunk x 128)];
//   partial     split-KV paged GQA attention: one block per (row, split,
//               kv head), the kv head's query heads sharing every K/V
//               tile, fp32 scores and online softmax, probabilities
//               rounded to bf16 for the V accumulation, the denominator
//               unrounded; a row attends [max(0, pos - window + 1), pos]
//               of its request (window 0: [0, pos]); the sink of query
//               head h — one more softmax column with logit sink[h] and no
//               value — enters split 0's running (max, sum) before its
//               first tile; rows with pos < 0 are padding;
//   combine     the splits merged, out = bf16(C / L);
//   fused       the three in one launch for a decode batch (the batch's
//               rows of one request overlaid on the staged tiles from the
//               projection, never read from the cache another block is
//               writing; the last block per (row, kv head) combines);
//   prefill     the query-tiled tensor-core form (a block per 64 query
//               vectors stages each 32-token tile once; mma.sync bf16).
//
// Every kernel is deterministic and capturable. Cache layout: bf16 K
// [slots, kv_heads * qk] and V [slots, kv_heads * 128] per layer, or the
// fp8 row form of kernels/latent_format.hpp (e4m3 codes with one fp32
// scale per (slot, kv head), K and V each) when the scale planes are
// given: the fp8 kernels are bitwise the bf16 ones over the dequantized
// rows.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

constexpr int kSinkAttnVDim = 128;
// Tiles of the attention kernels (the split geometry the host derives).
constexpr int kSinkAttnTile = 32;
// Rows the fused decode launch's batch table holds.
constexpr int kSinkAttnFusedMaxRows = 64;
// Query vectors (rows x query heads of one kv head) a prefill block stages
// each K/V tile for: the kernel's arithmetic intensity on the cache bytes.
// The query heads per kv head must divide it.
constexpr int kSinkAttnPrefillQueryVecs = 128;

// Whether the kernels are compiled for this qk width (192 or 128).
bool sink_attn_supported(int qk_dim);

// Where the fused projection's fp32 output keeps each head on this rank.
struct SinkQkvLayout {
  int chunks = 1;              // the checkpoint chunks the rank holds
  int64_t chunk_stride = 0;    // fp32 columns per chunk (the padded chunk rows)
  int q_per_chunk = 16;        // query heads per chunk
  int kv_per_chunk = 1;        // kv heads per chunk
};

// The finish's per-family parameters: the rotary table (fp32 [qk_dim /
// 2 for qk 128; 32 for qk 192]; null = no RoPE — qk 128 only), the head
// norms (bf16 [qk_dim]; null = none — qk 128 only) with their eps, and
// the value scale (1: none).
struct SinkFinishParams {
  const float* inv_freq = nullptr;
  const uint16_t* q_norm = nullptr;
  const uint16_t* k_norm = nullptr;
  float eps = 1e-6f;
  float value_scale = 1.0f;
};

// Rows [0, rows) of the fused projection's fp32 output `qkv` (row stride
// qkv_stride elements, the layout above). pos / req_ids: device [rows]; a
// row with pos < 0 writes nothing. k_scale / v_scale: the fp8 cache's
// scale planes [slots, kv_heads] (both null: the bf16 cache).
void sink_qkv_finish(int qk_dim, const float* qkv, int64_t qkv_stride, const SinkQkvLayout& layout,
                     const SinkFinishParams& fin, const int32_t* req_ids, const int64_t* pos, int rows,
                     int local_heads, int kv_heads, const int32_t* block_tables, int blocks_per_request,
                     int block_tokens, uint16_t* q_out, int64_t q_out_stride, uint16_t* k_cache,
                     uint16_t* v_cache, cudaStream_t stream, float* k_scale = nullptr, float* v_scale = nullptr);

// The partials of row r, split s, head h: m_ws / l_ws fp32 [rows, n_split,
// local_heads], c_ws fp32 [rows, n_split, local_heads, 128]. q: bf16 rows
// of local_heads x qk_dim (row stride q_stride, heads contiguous). Every
// query head h reads kv head h / (local_heads / kv_heads). Split s of a
// row covers tiles [s * chunk, (s + 1) * chunk) of its 32-token tiles
// counted from the first visible token (chunk = ceil(tiles / n_split));
// an empty split writes m = -inf, l = 0. sink: fp32 [local_heads] or null.
void sink_attn_partial(int qk_dim, const uint16_t* q, int64_t q_stride, const uint16_t* k_cache,
                       const uint16_t* v_cache, const int32_t* req_ids, const int64_t* pos, int rows,
                       int n_split, int local_heads, int kv_heads, int block_tokens, const int32_t* block_tables,
                       int blocks_per_request, int window, float scale, const float* sink, float* m_ws,
                       float* l_ws, float* c_ws, cudaStream_t stream, const float* k_scale = nullptr,
                       const float* v_scale = nullptr);

// out[r, h * 128 + d] = bf16(sum_s e^(m_s - M) c_s / sum_s e^(m_s - M) l_s)
// (0 when every split is empty); out rows local_heads * 128 wide.
void sink_attn_combine(const float* m_ws, const float* l_ws, const float* c_ws, int rows, int n_split,
                       int local_heads, uint16_t* out, cudaStream_t stream);

// The fused decode launch's arguments (the counters: an int per (row, kv
// head) that the kernel leaves zero again — allocate zeroed, never touch
// between launches). Rows at most kSinkAttnFusedMaxRows; four to 16 query
// heads per kv head.
struct SinkAttnFusedArgs {
  const float* qkv = nullptr;
  int64_t qkv_stride = 0;
  SinkQkvLayout layout;
  const float* inv_freq = nullptr;
  const uint16_t* q_norm = nullptr;  // qk 128: the per-head norms (null: none)
  const uint16_t* k_norm = nullptr;
  float eps = 1e-6f;
  float value_scale = 1.0f;
  const int32_t* req_ids = nullptr;
  const int64_t* pos = nullptr;
  int rows = 0;
  int n_split = 1;
  int local_heads = 0;
  int kv_heads = 0;
  const int32_t* block_tables = nullptr;
  int blocks_per_request = 0;
  int block_tokens = 0;
  int window = 0;
  float scale = 1.0f;
  const float* sink = nullptr;
  uint16_t* k_cache = nullptr;
  uint16_t* v_cache = nullptr;
  float* k_scale = nullptr;  // fp8 cache: both planes; bf16: both null
  float* v_scale = nullptr;
  float* m_ws = nullptr;     // [rows, n_split, local_heads]
  float* l_ws = nullptr;
  float* c_ws = nullptr;     // [rows, n_split, local_heads, 128]
  int* counters = nullptr;   // [rows, kv_heads], zero between launches
  uint16_t* out = nullptr;   // [rows, local_heads * 128]
};

size_t sink_attn_fused_smem_bytes(int qk_dim, int local_heads, int kv_heads);
void sink_attn_fused(int qk_dim, const SinkAttnFusedArgs& a, cudaStream_t stream);

// The query-tiled prefill attention: q the finish's bf16 rows (local_heads
// x qk_dim, stride q_stride); out rows local_heads x 128. The query heads
// per kv head must divide kSinkAttnPrefillQueryVecs.
struct SinkAttnPrefillArgs {
  const uint16_t* q = nullptr;
  int64_t q_stride = 0;
  const uint16_t* k_cache = nullptr;
  const uint16_t* v_cache = nullptr;
  const float* k_scale = nullptr;  // fp8 cache: both planes; bf16: both null
  const float* v_scale = nullptr;
  const int32_t* req_ids = nullptr;
  const int64_t* pos = nullptr;
  int rows = 0;
  int local_heads = 0;
  int kv_heads = 0;
  int block_tokens = 0;
  const int32_t* block_tables = nullptr;
  int blocks_per_request = 0;
  int window = 0;
  float scale = 1.0f;
  const float* sink = nullptr;
  uint16_t* out = nullptr;
};

size_t sink_attn_prefill_smem_bytes(int qk_dim);
int sink_attn_prefill_rows_per_block(int local_heads, int kv_heads);
void sink_attn_prefill(int qk_dim, const SinkAttnPrefillArgs& a, cudaStream_t stream);

}  // namespace dgpp
