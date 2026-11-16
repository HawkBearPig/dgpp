#pragma once
// One DSA layer's full forward (M3 layer phase, DESIGN §7.2): fused
// [q_a|kv_a] projection, MLA + indexer heads, pool compression, pinned
// top-k selection, absorbed attention, output projection.
//
// Weight layout (device pointers, caller-owned; mirrors dsa_reference.hpp's
// HostWeights field-for-field — the M4 loader materializes them from the
// checkpoint, BF16 except the fp32 APE):
//   Indexer (replicated across TP; absent on a layer that reuses the last
//   indexed layer's selection, plan D4):
//     wq_b [index_n_heads*128, q_lora_rank], wk [128, hidden],
//     wp [index_n_heads, hidden], gate [128, hidden] (kpool > 1 only),
//     k_norm_w/k_norm_b [128] (LayerNorm, eps 1e-6), ape fp32 [kpool, 128]
//     (kpool > 1 only).
//   MLA core (local TP views):
//     qkv_a [q_lora+kv_lora+rope, hidden] fused [q_a|kv_a] (the kv_a rows
//       carry the rope key after the latent, as the checkpoint's),
//     q_aln [q_lora], kv_aln [kv_lora] (RMSNorm, eps rms_norm_eps),
//     q_b [local_heads*(nope+rope), q_lora] (per head [nope | rope]),
//     kv_b [local_heads*(nope+v), kv_lora] — the checkpoint's interleaved
//       per-head layout (head h owns rows [h*(nope+v), h*(nope+v)+nope) of
//       W_uk and the next v rows of W_uv),
//     o_proj [hidden, local_heads*v] (row-parallel).
//   RoPE (rope > 0): rope_table bf16 [rope_table_positions][2][rope/2]
//     from dsa_rope_table_host (cos, sin per position), the model's.
//
// Scratch is caller-provided and SHARED by all DSA layers: they run
// sequentially on one stream, so the transient buffers (q, q_tilde, c, the
// attention workspace, the prefill dot buffer) are sized once for the
// largest layer use and reused — per-layer ownership would multiply the
// footprint by num_dsa_layers for zero benefit. Scratch contents are only
// valid until the next enqueue on ANY layer sharing the buffer.
//
// The forward is allocation-free. The decode path is CUDA-graph capturable:
//   * call prepare(tokens) for every row count you will capture BEFORE
//     capture (GEMM plans + the shared-memory opt-in are context state);
//   * decode batch metadata (req_ids/pos/req_spans) must live in
//     caller-owned DEVICE buffers whose contents are re-uploaded between
//     replays — visible-pool counts are derived on device, so one graph
//     serves any position;
//   * block-table growth is host-side admission control: the pool's
//     ensure_request_blocks() must already cover every position a captured
//     batch will write (enqueue_decode cannot check it — pos is device
//     state). enqueue_prefill grows the table itself and throws on pool
//     exhaustion (prefill is a control-path operation).
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/gemm.hpp"
#include "kernels/l2_prefetch.hpp"
#include "models/dsa_geometry.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

class DsaStatePool;

struct DsaLayerWeights {
  // Indexer (replicated across TP ranks). All null on a layer that attends
  // with the last indexed layer's selection (owns_indexer() false).
  const void* wq_b = nullptr;      // bf16 [index_n_heads*128, q_lora_rank]
  const void* wk = nullptr;        // bf16 [index_head_dim, hidden]
  const void* wp = nullptr;        // bf16 [index_n_heads, hidden]
  const void* gate = nullptr;      // bf16 [index_head_dim, hidden] (kpool > 1)
  const void* k_norm_w = nullptr;  // bf16 [index_head_dim]
  const void* k_norm_b = nullptr;  // bf16 [index_head_dim]
  const float* ape = nullptr;      // fp32 [kpool, index_head_dim] (kpool > 1)

  // MLA core (local TP views). The four quantized projections come EITHER
  // as the bf16 bridge pointers (qkv_a fused, q_b, o_proj — the M3 interface:
  // dequantized at load) OR as the checkpoint's FP8 pairs consumed by the
  // scale-aware GEMM directly (2026-09-08: q_a_q/kv_a_q/q_b_q/o_proj_q set,
  // the bf16 pointers null — half the bytes per token; the loader takes
  // this form whenever this rank's q_b row slice and o_proj column slice
  // start 128-aligned, the bridge otherwise) OR as the full GLM-5.3's
  // packed-int triples (2026-09-12: qkv_a_p/q_b_p/o_proj_p set, consumed
  // by the packed GEMV core with the exact code x scale dequant, plan D2).
  const void* qkv_a = nullptr;   // bf16 [q_lora+kv_lora+rope, hidden] fused
  const void* q_aln = nullptr;   // bf16 [q_lora_rank]
  const void* kv_aln = nullptr;  // bf16 [kv_lora_rank]
  const void* q_b = nullptr;     // bf16 [local_heads*(nope+rope), q_lora_rank]
  const void* kv_b = nullptr;    // bf16 [local_heads*(nope+v), kv_lora_rank]
  const void* o_proj = nullptr;  // bf16 [hidden, local_heads*v]
  GlmQuantMatrix q_a_q{};     // fp8 [q_lora_rank, hidden]
  GlmQuantMatrix kv_a_q{};    // fp8 [kv_lora_rank, hidden]
  GlmQuantMatrix q_b_q{};     // fp8 [local_heads*nope, q_lora_rank]
  GlmQuantMatrix o_proj_q{};  // fp8 [hidden, local_heads*v] (packed columns)
  GlmPackedMatrix qkv_a_p{};   // int8 [q_lora+kv_lora+rope, hidden]
  GlmPackedMatrix q_b_p{};     // int8 [local_heads*(nope+rope), q_lora_rank]
  GlmPackedMatrix o_proj_p{};  // int8 [hidden, local_heads*v]
  // The rotary table (rope > 0): bf16 [positions][2][rope/2].
  const void* rope_table = nullptr;
  int64_t rope_table_positions = 0;
  bool quantized() const { return q_a_q.payload != nullptr; }
  bool packed_int() const { return qkv_a_p.packed != nullptr; }
  bool owns_indexer() const { return wq_b != nullptr; }
};

class DsaLayer {
 public:
  // Carves the fixed scratch layout (see scratch_bytes) out of `scratch`,
  // which must be at least scratch_bytes(cfg, ...) bytes. `max_tokens`
  // bounds both prefill chunks and decode batches; `max_cache_tokens`
  // bounds the context length any request will reach (sizes the prefill
  // gather buffer); `max_decode_rows` (<= 16, the decode batch's cap; the
  // select runs them in groups of eight) and `decode_n_split` shape the
  // decode attention split. `dot_budget`
  // bounds the prefill dot/key buffer; query tiles use the current visible
  // context to fill that allocation, shrinking as the context grows.
  // Full GLM's fused scores use 64-bit keys in place of 32 FP32 head dots.
  DsaLayer(IGemm& gemm, const DsaLayerWeights& w, const DsaConfig& cfg,
           int max_tokens, int64_t max_cache_tokens, void* scratch,
           size_t scratch_capacity, void* gemm_workspace,
           size_t gemm_ws_bytes, int max_decode_rows = 8,
           int decode_n_split = 32, size_t dot_budget = 64ull << 20);

  // Prebuilds projection-GEMM plans for `tokens` rows and performs the
  // shared-memory opt-in. Must run outside graph capture; returns false
  // when a heuristic is unavailable.
  bool prepare(int tokens);

  // Prebuilds the fp8 dot-GEMM plan for a prefill query tile of
  // `tile_rows` rows against `visible_pools` gathered pools (the plan's n
  // is the internally padded pool count). Eager prefill works without
  // this — matmul builds plans lazily — but a cold plan costs a heuristic
  // query mid-chunk.
  bool prepare_prefill(int tile_rows, int64_t visible_pools);

  // Streaming-weight interface (M4 diagnostic forward): swap the device weight
  // view this layer enqueues against (shared scratch is unaffected — the
  // scratch is shape-keyed, not weight-keyed). A view without an indexer
  // makes the next enqueues selection-reusing (plan D4): they skip the
  // indexer projections, the index-cache writes and the select, and attend
  // with the topk_/counts_ the last indexed enqueue left — which must have
  // been the same call shape (rows, chunk, request / position buffer), or
  // the enqueue throws.
  void rebind(const DsaLayerWeights& w) {
    validate_weights(w);
    w_ = w;
  }
  bool reuses_selection() const { return !w_.owns_indexer(); }

  // Prefills one pool-aligned chunk [token_start, token_start+tokens) of
  // request `req` (a single request per call: the prefill gather/select
  // path is per-request; the engine loops requests). token_start must be a
  // multiple of kpool (continuation chunks stay pool-aligned; the final
  // chunk may end mid-pool, leaving the tail for decode).
  //   hidden_in: bf16 [tokens, hidden] (must not alias `out`)
  //   out:       bf16 [tokens, hidden]
  // Grows the request's block table via the pool; throws on exhaustion.
  //   row_base: the call's rows in the selection scratch (a group prefill's
  //             spans, 2026-09-14: span s's rows at its offset in the walk,
  //             so a selection-reusing layer finds every span's selection
  //             where the indexed layer left it; 0 for a one-request chunk)
  //   state_only: write the rows' cache state — the latent rows, the index
  //             pools, the tail ring, all functions of the site's INPUT — and
  //             stop: no selection, no attention, `out` untouched. For a
  //             caller that never reads the rows' output (the draft block's
  //             prefill rows fill its caches and nothing else).
  void enqueue_prefill(const void* hidden_in, DsaStatePool& state, int layer,
                       int req, int64_t token_start, int tokens, void* out,
                       cudaStream_t stream, int row_base = 0, bool state_only = false);

  // Decodes a batch of `tokens` rows over `num_requests` requests.
  //   hidden_in: bf16 [tokens, hidden] (padding rows: any input — kernels
  //              skip latent/ring writes for pos < 0 and the row's output
  //              is written as zeros)
  //   req_ids:   device int32 [tokens] — request id per row
  //   pos:       device int64 [tokens] — absolute position (-1 = padding)
  //   req_spans: device int32 [num_requests, 2] — (start, len) row spans
  //              into the token batch; fixed graphs may include all-padding
  //              spans, and a live request's rows must be contiguous
  //   out:       bf16 [tokens, hidden]; padding rows are all-zero
  // The caller must already have grown the block tables to cover pos+1.
  //   prefetch:  optional L2 weight prefetcher; after the projections it
  //              is pointed at o_proj, so the layer's long latency-bound
  //              middle (select, absorb, attention, v-out: ~130 us at
  //              decode) pulls the output projection into L2.
  //   tail_snapshots: optional bf16 [tokens, 2, kpool, index_head_dim]
  //              post-row ring snapshots for speculative rows (see
  //              dsa_kpool_decode_update); the layer's other state writes
  //              are positional and need no rollback.
  void enqueue_decode(const void* hidden_in, DsaStatePool& state, int layer,
                      const int32_t* req_ids, const int64_t* pos,
                      const int32_t* req_spans, int num_requests, int tokens,
                      void* out, cudaStream_t stream,
                      WeightPrefetcher* prefetch = nullptr,
                      void* tail_snapshots = nullptr);

  // Bytes of the output projection [hidden, local_v_rows] as resident:
  // bf16, the fp8 payload plus its scale grid, or the packed triple.
  size_t o_proj_bytes() const {
    if (w_.packed_int()) return w_.o_proj_p.packed_bytes() + w_.o_proj_p.scale_bytes();
    if (w_.quantized())
      return static_cast<size_t>(w_.o_proj_q.rows) * w_.o_proj_q.cols +
             w_.o_proj_q.scale_bytes();
    return static_cast<size_t>(cfg_.hidden) * geo_.local_v_rows * 2;
  }
  // Bytes of the bf16 kv_b [local_heads * (nope + v), kv_lora] (W_uk | W_uv).
  size_t kv_b_bytes() const {
    return static_cast<size_t>(geo_.local_heads) *
           (cfg_.qk_nope_head_dim + cfg_.v_head_dim) * cfg_.kv_lora_rank * 2;
  }
  // Bytes of the [q_a | kv_a] projection, the layer's first read: the fused
  // bf16 buffer, or the two fp8 pairs.
  size_t qkv_a_bytes() const {
    if (w_.packed_int()) return w_.qkv_a_p.packed_bytes() + w_.qkv_a_p.scale_bytes();
    if (w_.quantized())
      return static_cast<size_t>(w_.q_a_q.rows) * w_.q_a_q.cols +
             w_.q_a_q.scale_bytes() +
             static_cast<size_t>(w_.kv_a_q.rows) * w_.kv_a_q.cols +
             w_.kv_a_q.scale_bytes();
    return static_cast<size_t>(cfg_.q_lora_rank + cfg_.kv_lora_rank +
                               cfg_.qk_rope_head_dim) *
           cfg_.hidden * 2;
  }

  const DsaConfig& config() const { return cfg_; }
  const DsaGeometry& geometry() const { return geo_; }
  int max_tokens() const { return max_tokens_; }

  // Test/diagnostic probes into the shared scratch; valid until the next
  // enqueue on any layer sharing it.
  const void* debug_attn_out() const { return attn_out_; }   // [rows, local_v_rows] bf16
  // Tests: run the per-row split kernel for every prefill row (the dense
  // tensor-core path is the default where the context is below index_topk).
  void set_dense_prefill(bool on) { dense_prefill_ = on; }
  bool dense_prefill() const { return dense_prefill_; }
  // Tests: the prefill tiles' listed attention straight from the cache
  // (the staged copy of a tile's distinct rows is the default over a
  // quantized cache; the two are bit for bit the same output).
  void set_staged_prefill(bool on) { staged_prefill_ = on; }
  bool staged_prefill() const { return staged_prefill_ && stage_ws_ != nullptr; }
  const int32_t* debug_topk() const { return topk_; }        // [rows, max_selected]
  const int32_t* debug_counts() const { return counts_; }    // [rows]
  const void* debug_q_fp8() const { return q_fp8_; }         // [rows*heads, 128] fp8
  const float* debug_w_folded() const { return w_folded_; }  // [rows, heads]
  const void* debug_k_rows() const { return k_rows_; }       // [rows, 128] bf16
  const void* debug_gate_rows() const { return gate_rows_; } // [rows, 128] bf16
  // The last prefill tile's fp8-dot buffer [tile_rows*heads, padded_n] and
  // its pool stride — the exact dots the prefill select consumed for that
  // tile's rows (parity audits consume these bit-exact; the host cannot
  // reproduce a tensor-core reduction order). Zero rows for decode paths.
  const float* debug_dots() const { return dot_; }
  int64_t debug_dot_stride() const { return dot_stride_last_; }

  // Scratch bytes for the given shape (each region 256-byte aligned). One
  // buffer of this size serves every DSA layer.
  static size_t scratch_bytes(const DsaConfig& cfg, int max_tokens,
                              int64_t max_cache_tokens, int max_decode_rows = 8,
                              int decode_n_split = 32,
                              size_t dot_budget = 64ull << 20);

 private:
  // Offsets of every scratch region; computed once and shared by the
  // constructor and scratch_bytes so the formula cannot drift.
  struct ScratchLayout {
    size_t total = 0;
    size_t off_qkv = 0, off_q_c = 0, off_kv_c = 0, off_q = 0, off_q_idx = 0;
    size_t off_k_raw = 0, off_k_rows = 0, off_gate = 0, off_weights = 0;
    size_t off_q_fp8 = 0, off_q_scale = 0, off_w_folded = 0, off_topk = 0;
    size_t off_counts = 0, off_pos = 0, off_req_ids = 0, off_attn_out = 0;
    size_t off_q_tilde = 0, off_c = 0, off_m = 0, off_l = 0, off_cws = 0;
    size_t off_dot = 0, off_gather_k = 0, off_gather_scale = 0;
    size_t off_select_ws = 0, off_counter = 0, off_k_rot = 0, off_stage = 0;
    int64_t max_pools = 0;   // gather buffer capacity, in pools
    int tile_cap = 0;        // query rows that fit at max_pools
    int stage_slots = 0;     // rows the prefill tile's staged copy holds (0: no staged form)
  };
  static ScratchLayout scratch_layout(const DsaConfig& cfg, int max_tokens,
                                      int64_t max_cache_tokens,
                                      int max_decode_rows, int decode_n_split,
                                      size_t dot_budget);
  // Reuse the allocated dot capacity across more query rows at short
  // contexts. The caller supplies a padded pool count within max_pools_.
  int prefill_query_tile_rows(int64_t padded_pools) const;

  // The projection chain shared by both paths: fused qkv GEMM, split
  // RMSNorms, RoPE (rope > 0: the q and k rope slices, the indexer's q and
  // k, at pos[row]), q_b/wq_b projections, indexer k (LayerNorm) + gate +
  // fp32 weights, Hadamard quant, weight fold. Fills q_, k_rot_, q_fp8_,
  // w_folded_, kv_c_, k_rows_, gate_. Latent append and cache writes are
  // path-specific. A selection-reusing layer runs only the MLA half.
  void project_common(const void* hidden_in, int tokens, const int64_t* pos,
                      cudaStream_t stream);
  void project_out(void* out, int tokens, cudaStream_t stream);
  void validate_weights(const DsaLayerWeights& w) const;
  // The selection-reuse contract (plan D4): the last indexed enqueue's
  // call shape at each selection-scratch base, checked by a reusing
  // enqueue at the same base (a group prefill's spans note one each).
  enum class SelKind { kNone, kPrefill, kDecode };
  struct SelNote {
    SelKind kind = SelKind::kNone;
    int rows = 0;
    int64_t start = -1;
    int req = -1;
    const int64_t* pos = nullptr;
  };
  void note_selection(SelKind kind, int rows, int64_t start, int req,
                      const int64_t* pos);
  void require_selection(SelKind kind, int rows, int64_t start, int req,
                         const int64_t* pos) const;

  // absorb + split-KV attention + v-absorb for `rows` query rows starting
  // at chunk-relative row `row0`, in attention tiles of attn_rows_ rows;
  // topk_/counts_ must already hold the tile's selection. `req_ids` is
  // indexed tile-locally (entry r = request of row row0 + r). n_split is
  // the split-KV parallelism for this call (decode: rows are scarce, split
  // wide; prefill: 8-row tiles already parallelize, split narrow).
  // Dense causal attention for query rows [row0, row0 + rows) of the
  // current prefill chunk (positions pos_dev_[row0..]), in tiles of
  // kDensePrefillRows rows; falls back to attend_tile when the geometry
  // is outside the dense kernel's (kv_lora not 512/256).
  // `decode`: the projections keep their warp kernels (a batched decode
  // row is bitwise the row alone; the tensor-core forms are not).
  // `one_req` >= 0: every row is this request's (a prefill chunk), so a
  // listed tile may attend from the staged copy of its distinct rows.
  void attend_dense(DsaStatePool& state, int layer, const int32_t* req_ids,
                    int64_t row0, int rows, bool listed, cudaStream_t stream,
                    int n_split = kDensePrefillSplit, bool decode = false, int one_req = -1);
  void attend_tile(DsaStatePool& state, int layer, const int32_t* req_ids,
                   int64_t row0, int rows, int n_split, cudaStream_t stream);

  void validate_pool(const DsaStatePool& state, int layer) const;

  IGemm& gemm_;
  DsaLayerWeights w_;
  DsaConfig cfg_;
  DsaGeometry geo_;
  int max_tokens_ = 0;
  int64_t max_cache_tokens_ = 0;
  int max_decode_rows_ = 8;
  int decode_n_split_ = 32;
  // The decode attention on the tensor-core listed kernel:
  // two rows x 16 local heads is one M-block of the dense kernel, its two
  // slabs each a row's own selection. DGPP_DSA_DECODE_MMA=off keeps the
  // register split kernel; tolerance-equal, not bitwise (the mma order).
  bool decode_mma_ = true;
  int decode_mma_split_ = 32;  // split-KV parallelism (TRIED 2026-09-06 and kept at 32: 48 and 64 splits measured the same step on the fabric once the partial kernel held its window in registers, and change the combine order)
  int attn_rows_ = 8;         // attention tile rows = max(max_decode_rows_, 8)
  // The dense prefill path: rows whose context is below
  // index_topk tokens attend densely on the tensor-core kernel
  // (dsa_attn_dense) in tiles of kDensePrefillRows query rows split
  // kDensePrefillSplit ways; the rest keep the per-row split kernel.
  // set_dense_prefill(false) forces the split kernel everywhere (tests).
  static constexpr int kDensePrefillRows = 128;
  static constexpr int kDensePrefillSplit = 4;
  bool dense_prefill_ = true;
  // The staged copy of a listed prefill tile's distinct cache rows
  // (dsa_attn_listed_staged). The tile's rows share most of their tokens,
  // so the copy is dsa_attn_stage_slots rows, not the pairs' bound; a tile
  // past it attends from the cache, the same bits.
  bool staged_prefill_ = true;
  bool stage_zeroed_ = false;   // the workspace's bitmap, zeroed before its first launch
  void* stage_ws_ = nullptr;    // dsa_attn_stage_bytes(max_cache_tokens, stage_slots_, score_width); null: no staged form
  int stage_slots_ = 0;
  int tile_cap_ = 0;          // prefill dot-tile rows at maximum context
  int64_t max_pools_ = 0;     // gather/dot capacity, in padded pools
  int64_t gather_zeroed_ = 0; // gather high-water mark already zeroed
  float logit_scale_ = 0.f;   // 128^-0.5 * heads^-0.5, reference-pinned
  float attn_scale_ = 0.f;    // (nope + rope)^-0.5
  std::unordered_map<int, SelNote> sel_notes_;  // by selection-scratch base
  int sel_base_ = 0;  // this call's rows in topk_/counts_ (enqueue_prefill's row_base; decode 0)
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;

  // Fixed-address scratch (carved from the caller's shared buffer).
  uint8_t* scratch_ = nullptr;
  uint16_t* qkv_ = nullptr;      // [T, q_lora+kv_lora+rope]
  uint16_t* q_c_ = nullptr;      // [T, q_lora]
  uint16_t* kv_c_ = nullptr;     // [T, kv_lora]
  uint16_t* k_rot_ = nullptr;    // [T, rope] (the roped rope key; rope > 0)
  uint16_t* q_ = nullptr;        // [T, local_heads*(nope+rope)] (rope slices roped)
  uint16_t* q_idx_ = nullptr;    // [T, index_heads*128]
  uint16_t* k_raw_ = nullptr;    // [T, 128]
  uint16_t* k_rows_ = nullptr;   // [T, 128] (LayerNormed)
  uint16_t* gate_rows_ = nullptr;  // [T, 128]
  float* weights_ = nullptr;     // [T, index_heads]
  uint8_t* q_fp8_ = nullptr;     // [T*index_heads, 128]
  float* q_scale_ = nullptr;     // [T*index_heads]
  float* w_folded_ = nullptr;    // [T*index_heads]
  int32_t* topk_ = nullptr;      // [T, max_selected]
  int32_t* counts_ = nullptr;    // [T]
  int64_t* pos_dev_ = nullptr;   // [T] (prefill staging target)
  int32_t* req_ids_dev_ = nullptr;  // [T] (prefill staging target)
  uint16_t* attn_out_ = nullptr;  // [T, local_heads*v]
  uint16_t* q_tilde_ = nullptr;  // [tile_cap, local_heads*(kv_lora+rope)]
  float* c_ = nullptr;           // [tile_cap, local_heads*kv_lora]
  float* m_ws_ = nullptr;        // [tile_cap, n_split, local_heads]
  float* l_ws_ = nullptr;        // [tile_cap, n_split, local_heads]
  float* c_ws_ = nullptr;        // [tile_cap, n_split, local_heads, kv_lora]
  float* dot_ = nullptr;         // [tile_rows*index_heads, padded_n]
  uint8_t* gather_k_ = nullptr;  // [max_pools, 128] fp8
  float* gather_scale_ = nullptr;  // [max_pools]
  void* select_ws_ = nullptr;       // dsa_select_workspace_bytes(rows, pools)
  int64_t select_ws_pools_ = 0;     // the pool capacity it was sized for
  int32_t* counter_ws_ = nullptr;   // [1]
  int64_t dot_stride_last_ = 0;     // padded pool count of the last dot tile

  // Prefill host staging (pos / req_ids uploads; prefill is never captured).
  std::vector<int64_t> pos_host_;
  std::vector<int32_t> req_ids_host_;
};

}  // namespace dgpp
