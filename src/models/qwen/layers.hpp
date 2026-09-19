#pragma once
// Qwen3.8-Flash-Next layers composed from shared CUDA kernels and GEMM
// interfaces. Each owns shape-specific scratch and can bind resident weight
// views. The same operators support diagnostic forward, prefill and graph
// decode. Streaming mode rebinds them as each layer is loaded.
//
//   QwenGrSite   the gated residual's read (mix) and write (combine)
//   QwenGdnLayer the Gated DeltaNet block: projections, conv, recurrence,
//                gated norm, output projection (the KDA kernel family)
//   QwenQsaLayer the sparse attention block with its paged caches: the
//                q/k/v and indexer projections, norm+RoPE, K/V append,
//                compressed-key maintenance, scoring, selection, listed
//                attention, gate, output projection
//   QwenPleLayer the hashed n-gram embedding injected into the hyper state
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/gemm.hpp"
#include "kernels/kda.hpp"
#include "models/qwen/config.hpp"
#include "models/qwen/loader.hpp"

namespace dgpp {

// The GEMM interface's workspace, shared by every layer object of a model.
struct QwenGemmWorkspace {
  IGemm* gemm = nullptr;
  void* ws = nullptr;
  size_t ws_bytes = 0;
  // The FP8 dense stack's prefill bridge (engine.dense_weights = "fp8",
  // 2026-09-10): a BF16 scratch the size of the largest dense matrix; a
  // prefill-shaped product dequantizes the matrix into it (the GEMV core's
  // own values, bf16(e4m3 x scale)) and runs the BF16 GEMM interface — the fp8
  // tile kernel is a quarter slower than cuBLASLt at these shapes (2.6 s
  // against 2.0 s for a 2K prompt, measured).
  uint16_t* dequant = nullptr;
  size_t dequant_bytes = 0;
  // The dense sites' lowering (kernels/gemm.hpp dense_gemv_rows): the GEMV
  // chunks and the fused multi-problem launches to gemv_rows; fp8 rows from
  // mma_from_rows take the streaming tensor-core GEMM (0: never).
  int gemv_rows = 8;
  int mma_from_rows = 0;
};

// The draft's hidden projection flattens the hyper-state branches into
// tokens * hc rows. Wider decode must remain safe for collective graphs.
void qwen_mtp_hidden_projection(const QwenGemmWorkspace& gemm, const uint16_t* act,
    const uint16_t* weight, uint16_t* out, int tokens, int hc, int hidden,
    bool decode, cudaStream_t stream);

// ---- the gated residual --------------------------------------------------------
class QwenGrSite {
 public:
  QwenGrSite(const QwenGrResident& w, const QwenGemmWorkspace& gemm, int hc, int hidden,
             int lowrank, int max_tokens, float eps);
  ~QwenGrSite();
  QwenGrSite(const QwenGrSite&) = delete;
  QwenGrSite& operator=(const QwenGrSite&) = delete;
  void rebind(const QwenGrResident& w) { w_ = w; }
  // x[T, H] from R[T, hc*H]; Rn stays in this object for combine().
  void mix(const uint16_t* r, uint16_t* x, int tokens, cudaStream_t stream);
  // R += s(Rn) (x) y, y [T, H]; requires the site's inject weights.
  void combine(uint16_t* r, const uint16_t* y, int tokens, cudaStream_t stream);
  const uint16_t* rn() const { return rn_; }
  // The bytes the constructor allocates for a shape (the memory plan).
  static size_t scratch_bytes(int hc, int hidden, int lowrank, int max_tokens);

 private:
  QwenGrResident w_;
  QwenGemmWorkspace g_;
  int hc_, hidden_, lowrank_, max_tokens_;
  float eps_;
  // The inject dots' side stream: the gates read only Rn, so
  // the dots are issued here as soon as mix() has normalized the row and
  // joined in combine() — off the chain that runs the branch, the boundary
  // collective and the apply. Same kernel on either stream, so the values
  // are bitwise the in-chain combine's. DGPP_QWEN_GR_GATE_SIDE=off keeps
  // the dots in the chain.
  cudaStream_t gate_side_ = nullptr;
  cudaEvent_t gate_fork_ = nullptr, gate_join_ = nullptr;
  int fused_rows_max_ = 8;    // DGPP_QWEN_GR_FUSED=one keeps the scalar row alone
  bool gate_early_ = true;
  bool gate_side_only_ = false;  // DGPP_QWEN_GR_GATE_SIDE=side: the batched rows' side stream    // DGPP_QWEN_GR_GATE_SIDE=off puts both back in the chain
  bool gate_forked_ = false;
  bool gates_ready_ = false;  // the mix's down GEMV computed them in its own launch
  void fork_gate_dots(cudaStream_t main, int tokens);
  // The inject rows ride the fused mix's down GEMV: they run the same
  // 16-byte-chunked row chain, so the weight must be aligned like the
  // down matrix.
  bool inject_fused() const {
    return gate_early_ && w_.inject != nullptr &&
           (reinterpret_cast<uintptr_t>(w_.inject) % 16) == 0;
  }
  uint16_t* rn_ = nullptr;      // [M, hc*H]
  uint16_t* t_ = nullptr;       // [M, lowrank]
  uint16_t* logits_ = nullptr;  // [M, hc*H]
  float* gates_ = nullptr;      // [M, hc] the combine's inject gates
  bool fused_mix_ = false;      // the decode rows' norm/act-staged GEMVs
};

// ---- Gated DeltaNet -----------------------------------------------------------------
class QwenGdnLayer {
 public:
  QwenGdnLayer(const QwenGdnResident& w, const QwenGemmWorkspace& gemm, const QwenTextConfig& cfg,
               int max_tokens);
  ~QwenGdnLayer();
  QwenGdnLayer(const QwenGdnLayer&) = delete;
  QwenGdnLayer& operator=(const QwenGdnLayer&) = delete;
  void rebind(const QwenGdnResident& w);
  // out[T, H] = GDN(x[T, H]); recurrent_state fp32 [lv, V, K] and
  // conv_state bf16 [C, conv_width - 1] updated in place. Speculative rows
  // (the engine's verify) hand post-row snapshots of both states through
  // the KDA sinks (rec_snap.states / conv_snap.states, row strides in
  // elements): row r's state lands in snapshot row r.
  void enqueue(const uint16_t* x, float* recurrent_state, uint16_t* conv_state, uint16_t* out,
               int tokens, cudaStream_t stream, const KdaStateSnapshots& rec_snap = {},
               const KdaConvSnapshots& conv_snap = {});
  // The request-indexed row form (the fixed decode batch): rec_states /
  // conv_states are slot 0's states, request strides in elements; the
  // row map selects each span's slot, padding rows (pos < 0) write zero
  // and leave every state alone. Snapshot rows are global batch rows.
  void enqueue_rows(const uint16_t* x, float* rec_states, int64_t rec_stride,
                    uint16_t* conv_states, int64_t conv_stride, uint16_t* out, int rows,
                    const KdaRequestRows& requests, cudaStream_t stream,
                    const KdaStateSnapshots& rec_snap = {},
                    const KdaConvSnapshots& conv_snap = {});
  int64_t conv_channels() const { return conv_channels_; }
  int64_t recurrent_elems() const { return static_cast<int64_t>(lv_) * v_dim_ * k_dim_; }
  int64_t conv_state_elems() const { return conv_channels_ * (conv_width_ - 1); }
  static size_t scratch_bytes(const QwenTextConfig& cfg, int local_key_heads, int local_value_heads,
                              int max_tokens);

 private:
  void in_projections(const uint16_t* x, int tokens, cudaStream_t stream);
  QwenGdnResident w_;
  QwenGemmWorkspace g_;
  int hidden_, lk_, lv_, k_dim_, v_dim_, conv_width_, max_tokens_;
  float eps_, scale_;
  int64_t conv_channels_ = 0;
  uint16_t* qkv_ = nullptr;     // [M, C]
  uint16_t* qkvc_ = nullptr;    // [M, C] post-conv
  uint16_t* z_ = nullptr;       // [M, lv*V]
  uint16_t* a_ = nullptr;       // [M, lv]
  uint16_t* b_ = nullptr;       // [M, lv]
  uint16_t* core_ = nullptr;    // [M, lv*V]
  uint16_t* normed_ = nullptr;  // [M, lv*V]
};

// ---- Qwen Sparse Attention ----------------------------------------------------------
// One attention layer's paged caches as the kernels see them — a view the
// pool (models/qwen/kv_pool.hpp) hands out per layer: the block table is
// shared by every layer, blocks_per_request == the pool's total blocks.
struct QwenQsaCache {
  uint16_t* k_cache = nullptr;      // bf16 [slots, lkv * D]
  uint16_t* v_cache = nullptr;      // bf16 [slots, lkv * D]
  uint16_t* index_cache = nullptr;  // bf16 [pool_slots, Di]
  uint16_t* ring = nullptr;         // bf16 [max_requests, kpool, Di]
  int32_t* block_tables = nullptr;  // int32 [max_requests, blocks_per_request]
  int block_tokens = 0;
  int blocks_per_request = 0;
  int max_requests = 0;
  int64_t slots() const { return static_cast<int64_t>(block_tokens) * blocks_per_request; }
  int64_t pool_slots(int kpool) const { return slots() / kpool; }
};

// The rows one enqueue serves. Prefill: one request's contiguous chunk at
// positions [pos0, pos0 + T), pos0 a multiple of kpool (the ring's contract;
// 0 for the cold start) — the chunk's complete pools are compressed from its
// own rows, the ring re-seeded with its last kpool raw keys. Decode: T rows
// of one or more requests in span order (the KDA row map's shape: req_spans
// int32 [num_requests, 2] = start, len; a request's rows contiguous and in
// position order) — each row stashes its raw key in the request's ring and a
// row completing a pool compresses it. Both: req_ids / pos are device [T].
struct QwenQsaRows {
  const int32_t* req_ids = nullptr;
  const int64_t* pos = nullptr;
  bool decode = false;
  int request = 0;                     // prefill: the request
  int64_t pos0 = 0;                    // prefill: the chunk's first position
  const int32_t* spans = nullptr;      // decode: device [num_requests, 2]
  int num_requests = 0;                // decode
  uint16_t* ring_snapshots = nullptr;  // decode: bf16 [T, kpool, Di] post-row rings (spec rows)
};

class QwenQsaLayer {
 public:
  // max_pools bounds the visible pools of any row (the scoring workspace).
  QwenQsaLayer(const QwenQsaResident& w, const QwenGemmWorkspace& gemm, const QwenTextConfig& cfg,
               int max_tokens, int64_t max_pools);
  ~QwenQsaLayer();
  QwenQsaLayer(const QwenQsaLayer&) = delete;
  QwenQsaLayer& operator=(const QwenQsaLayer&) = delete;
  void rebind(const QwenQsaResident& w);

  // out[T, H] = QSA(x[T, H]) for the rows described: projections, norm +
  // RoPE, the K/V appends, the compressed-key maintenance, the selection
  // and the listed attention per row, the gated output projection.
  void enqueue(const uint16_t* x, int tokens, const QwenQsaRows& rows, QwenQsaCache& cache,
               uint16_t* out, cudaStream_t stream);

  static size_t scratch_bytes(const QwenTextConfig& cfg, int local_heads, int local_kv_heads,
                              int max_tokens, int64_t max_pools);
  int local_heads() const { return lh_; }
  int local_kv_heads() const { return lkv_; }
  int kpool() const { return kpool_; }
  int select_k() const { return select_k_; }
  int max_selected() const { return max_selected_; }

 private:
  QwenQsaResident w_;
  QwenGemmWorkspace g_;
  int hidden_, lh_, lkv_, dim_, rotary_, idx_heads_, idx_dim_, kpool_, select_k_, max_selected_;
  int max_tokens_;
  int64_t max_pools_;
  float eps_, scale_;
  // The YaRN attention factor the cos/sin are built with (1.0f off the
  // knob): vLLM bakes mscale into its cos/sin cache, its softmax scale
  // staying head_dim^-0.5 (nvidia/qsa.py: self.scaling).
  float mscale_ = 1.0f;
  float* d_inv_freq_ = nullptr;    // [rotary/2]
  uint16_t* q_ = nullptr;          // [M, lh * 2D]
  uint16_t* k_ = nullptr;          // [M, lkv * D]
  uint16_t* v_ = nullptr;          // [M, lkv * D]
  uint16_t* qn_ = nullptr;         // [M, lh * D]
  uint16_t* kn_ = nullptr;         // [M, lkv * D]
  uint16_t* idx_ = nullptr;        // [M, (nH + 1) * Di]
  uint16_t* qi_ = nullptr;         // [M, nH * Di]
  uint64_t* keys_ws_ = nullptr;    // [M, max_pools]
  int32_t* topk_ = nullptr;        // [M, max_selected]
  int32_t* counts_ = nullptr;      // [M]
  float* m_ws_ = nullptr;          // [M, n_split, lh]
  float* l_ws_ = nullptr;
  float* c_ws_ = nullptr;          // [M, n_split, lh, D]
  float* c_out_ = nullptr;         // [M, lh * D]
  uint16_t* o_ = nullptr;          // [M, lh * D]
  int n_split_ = 1;
};

// ---- the n-gram embedding layer -----------------------------------------------------
class QwenPleLayer {
 public:
  QwenPleLayer(const QwenPleResident& w, const QwenNgramTableResident& table,
               const QwenGemmWorkspace& gemm, const QwenTextConfig& cfg, int max_tokens);
  ~QwenPleLayer();
  QwenPleLayer(const QwenPleLayer&) = delete;
  QwenPleLayer& operator=(const QwenPleLayer&) = delete;
  void rebind(const QwenPleResident& w, const QwenNgramTableResident& table);
  // The mmap'ed table (2026-09-10, table.mmap set): the table never
  // reaches the device. stage() at the walk's start runs the hash kernel
  // into PINNED ids and forks a host node off `stream` that gathers the
  // rows from the mapping into pinned staging while the walk runs its
  // first layer; embed() at the layer's turn joins and converts them
  // (bf16(e4m3 x scale), bitwise the device gather's). Works eagerly and
  // under capture alike (the fork/join are graph edges, the gather a host
  // node) — every path stages from the device's own tokens and context,
  // so nothing mirrors them on the host.
  bool staged() const { return table_.mmap != nullptr; }
  void stage(const int64_t* tokens, int rows, const int32_t* req_ids, const int64_t* pos,
             const int32_t* req_spans, int num_requests, const int32_t* ctx, cudaStream_t stream);
  // A staging that failed on the host (an id outside the table) surfaces
  // here — the next stage() throws it too.
  void check_staged() const;
  // The pinned bytes of the mmap'ed mode: the ids and the staged rows.
  static size_t staging_bytes(const QwenTextConfig& cfg, int hash_heads, int max_tokens);

  // The row form (the model's one walk): rows of one or more requests in
  // span order (kernels/qwen_ple.hpp), each request's n-gram context at
  // ctx + req * 4 (device int32 [max_requests, 4]) and its conv state at
  // states + req * state_stride. Five steps, so the model can fold each
  // K-sliced projection through its boundary reducer's staged buffer (one
  // stable buffer per handout, plan D4) before the next one is produced:
  //   embed        the hash ids and this rank's hash heads' rows (e_)
  //   project_key  key_dst [T, hc*H] = e_ x key_proj^T (null: key_partial())
  //   norm_key     kn_ = group norm of the (folded) key
  //   project_value  val_dst [T, H] = e_ x value_proj^T (null: value_partial())
  //   finish       the query norm, the gate on the (folded) value, the conv
  //                norm and the conv into r [T, hc*H], every request's conv
  //                state advanced (snapshots: the per-row post-states).
  // At world 1 the partials are the values. The context itself advances
  // in the model's context kernel.
  void embed(const int64_t* tokens, int rows, const int32_t* req_ids, const int64_t* pos,
             const int32_t* req_spans, int num_requests, const int32_t* ctx, cudaStream_t stream);
  void project_key(uint16_t* key_dst, int rows, cudaStream_t stream);
  void norm_key(const uint16_t* key, int rows, cudaStream_t stream);
  void project_value(uint16_t* val_dst, int rows, cudaStream_t stream);
  void finish(uint16_t* r, const uint16_t* value, uint16_t* states, int64_t state_stride,
              const int32_t* req_ids, const int64_t* pos, const int32_t* req_spans, int num_requests,
              int rows, cudaStream_t stream, uint16_t* snapshots = nullptr);
  static size_t scratch_bytes(const QwenTextConfig& cfg, int hash_heads, int max_tokens);
  uint16_t* key_partial() { return key_; }
  uint16_t* value_partial() { return val_; }
  int64_t conv_state_elems() const { return static_cast<int64_t>(hc_) * hidden_ * state_len_; }
  int32_t eos() const { return eos_; }

 private:
  QwenPleResident w_;
  QwenNgramTableResident table_;
  QwenGemmWorkspace g_;
  int hc_, hidden_, heads_, heads_per_ngram_, head_dim_, width_, dilation_, state_len_, max_tokens_;
  int32_t eos_;
  float eps_;
  struct StageArgs;
  static void stage_callback(void* user);
  int32_t* h_ids_ = nullptr;      // mmap: the pinned ids (ids_ their device alias)
  uint8_t* staged_ = nullptr;     // mmap: pinned [M, hash_heads, head_dim]
  cudaStream_t side_ = nullptr;   // mmap: the host node's branch
  cudaEvent_t fork_ = nullptr, join_ = nullptr;
  std::vector<std::unique_ptr<StageArgs>> stage_args_;  // one per capture, two eager
  size_t eager_slot_ = 0;
  int staged_rows_ = 0;           // the rows the pending stage covers
  int64_t* d_mult_ = nullptr;     // [ngram_size]
  int64_t* d_vocab_ = nullptr;    // [heads]
  int64_t* d_offset_ = nullptr;   // [heads]
  int32_t* ids_ = nullptr;        // [M, heads]
  uint16_t* e_ = nullptr;         // [M, hash_heads * head_dim]
  uint16_t* key_ = nullptr;       // [M, hc*H]
  uint16_t* kn_ = nullptr;        // [M, hc*H]
  uint16_t* val_ = nullptr;       // [M, H]
  uint16_t* qn_ = nullptr;        // [M, hc*H]
  uint16_t* gv_ = nullptr;        // [M, hc*H]
  uint16_t* un_ = nullptr;        // [M, hc*H]
};

}  // namespace dgpp
