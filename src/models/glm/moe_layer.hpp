#pragma once
// MoE execution for diagnostic, decode and prefill workloads.
//
// enqueue() runs host segmentation after reading back router IDs and
// weights, then gathers expert inputs and applies each expert in order.
// enqueue_decode() keeps routes and expert views on the device and uses
// slot kernels without a host synchronization. The grouped prefill path
// segments rows on the device and uses GEMV or tensor-core kernels.
//
// Every TP rank holds an intermediate slice of each expert. Partial down
// projections remain in FP32, with one fma per expert in ascending ID order
// and the shared expert last. The chain rounds to BF16 once before the
// FFN all-reduce. This differs from the reference's per-expert BF16
// roundings; glm_moe_reference implements the same accumulation rule in
// double precision, and tests compare execution paths.
//
// Route traces use caller-owned pinned staging and asynchronous copies.
// Read them after the model stream completes; see MoeTraceStaging and
// models/glm/trace.hpp.
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/glm_moe_launch.hpp"
#include "models/glm/moe.hpp"

namespace dgpp {

// Caller-owned PINNED staging for enqueue_decode's async route copies:
// the decode step's D2H copies land here mid-step (stream-ordered, no
// sync) and the caller reads them after its next stream sync — the
// session materializes Outputs.routes this way after the step's final
// sync. A null trace skips all copies; a supplied trace requires ids and
// weights, while biased logits are optional.
struct MoeTraceStaging {
  int32_t* ids = nullptr;     // [tokens * top_k]
  float* weights = nullptr;  // [tokens * top_k]
  float* biased = nullptr;    // [tokens * n_experts]
};

// The grouped expert path's kernel. kGemv: the fp8 GEMV core,
// four rows per pass — the decode path's kernel, bitwise the decode slot
// path at the same routing. kMma: the bf16 tensor-core tile path
// (launch_moe_grouped_mma_*), 128-row m-tiles, bitwise the scale GEMM's
// tile kernel — the PREFILL's kernel, where segments run to hundreds of
// rows and the GEMV core re-read every expert once per four of them.
enum class MoeExpertKernel { kGemv, kMma };

class GlmMoeLayer {
 public:
  // weights: device-visible pointers (e.g. GlmLayerStream's GlmMoeResident
  // wired into GlmMoeWeights); they must outlive this layer.
  //
  // decode_slots: the decode-row bound enqueue_decode accepts (the
  // session engine's kDecodeRows). 0 disables the decode path (its
  // scratch is not allocated) — the M4 forward-only callers and the
  // pre-prefill unit tests use that shape.
  //
  // graph_table_slots: per-call STABLE pinned expert-table sources for
  // CUDA-graph capture (the session passes its MoE-layer ordinal as
  // table_slot). A captured graph replays the table upload NODE with
  // the source address baked — one shared pinned buffer would replay
  // the LAST capture-time refill for every layer (the wrong-weights
  // class). Per-slot buffers freeze each layer's table at capture time
  // (resident bindings are lifetime-stable, so the bytes never go
  // stale); the shared DEVICE destination stays safe because the
  // replay's memcpy nodes and kernels serialize on the stream. 0 keeps
  // the single shared pinned buffer (the eager path).
  GlmMoeLayer(const GlmMoeWeights& weights, const GlmMoeConfig& cfg,
              int max_tokens, int decode_slots = 0,
              int graph_table_slots = 0);
  ~GlmMoeLayer();
  GlmMoeLayer(const GlmMoeLayer&) = delete;
  GlmMoeLayer& operator=(const GlmMoeLayer&) = delete;

  // out[tokens, hidden] = routed_sum + shared(hidden); out is zeroed
  // internally and accumulated in place. Synchronizes the stream once.
  // The prefill path without a host sync: the router's
  // ids stay on the device, a segmentation kernel builds the rows, the
  // slot map and the segment table there, and the grouped chain runs on
  // them — nothing waits on the host. Routing traces ride async copies
  // into `trace` (pinned; materialize after the stream's next sync;
  // last_ids()/last_weights()/last_biased() are not updated). Bitwise the
  // host path's output (glm_moe_test pins it).
  // enqueue_prefill runs the grouped chain on the tensor-core kernel
  // (MoeExpertKernel::kMma); enqueue(), the host-orchestrated reference,
  // takes the kernel as an argument — kGemv by default (the decode pin),
  // kMma to serve as the prefill path's bitwise reference.
  // Mixed rows retain the native slot recipe for the decoder prefix. Only
  // readers enter the grouped chain; each row is computed once. A decoder-only
  // bounded phase bypasses the grouped chain entirely.
  void enqueue_prefill(const uint16_t* hidden, uint16_t* out, int tokens,
                       MoeTraceStaging* trace, cudaStream_t stream, int decode_prefix = 0);
  // The prefill in phases, for a caller that overlaps its boundary folds
  // with the per-row ends of the layer (the DeepSeek-V4 walk, 2026-10-02):
  // the router over a row block ahead of the chain (route_prefill_rows on
  // every row before enqueue_prefill_phased), the chain without its
  // accumulation, the accumulation per row block. The same kernels per
  // row as enqueue_prefill: bitwise its output.
  // A mixed phase writes its native prefix directly to `out`; every later
  // accumulation must pass the same prefix so those rows are retained.
  void route_prefill_rows(const uint16_t* hidden, int r0, int rows, cudaStream_t stream);
  void enqueue_prefill_phased(const uint16_t* hidden, int tokens, cudaStream_t stream,
                             int decode_prefix = 0, uint16_t* out = nullptr);
  void accumulate_prefill_rows(uint16_t* out, int tokens, int r0, int rows, cudaStream_t stream,
                               int decode_prefix = 0);
  void enqueue(const uint16_t* hidden, uint16_t* out, int tokens,
               cudaStream_t stream,
               MoeExpertKernel kernel = MoeExpertKernel::kGemv);
  // The host path with the fp32 chain handed back UNROUNDED (out
  // [tokens, hidden] f32) — the Qwen MoE continues it with its BF16 shared
  // expert and rounds once after (models/qwen/moe_layer.hpp).
  void enqueue_f32(const uint16_t* hidden, float* out, int tokens,
                   cudaStream_t stream,
                   MoeExpertKernel kernel = MoeExpertKernel::kGemv);
  // The prefill path's ROUTED chain alone (the device segmentation, the
  // tensor-core kernel on any 32/64/128 scale grid), handed back unrounded
  // in fp32 — the Qwen prefill (its BF16 shared expert follows).
  void enqueue_prefill_f32(const uint16_t* hidden, float* out, int tokens,
                           MoeTraceStaging* trace, cudaStream_t stream,
                           MoeExpertKernel kernel = MoeExpertKernel::kMma,
                           int decode_prefix = 0);
  // Whether the tensor-core expert kernel takes this layer's fp8 scale grid:
  // the checkpoint's 128 x 128, or a re-blocked TP slice grid of 32/64 with
  // hidden and the slice both multiples of 32 (the ldmatrix tile kernel,
  // 2026-09-09). Otherwise the GEMV core is the expert kernel. NVFP4
  // tables: true.
  bool mma_takes_grid() const;
  // Whether the FP8 shared expert rides this layer's chain
  // (cfg.n_shared_experts == 1). Without it the prefill and decode fast
  // paths are not wired yet (they refuse; the Qwen engine's milestone).
  bool has_shared() const { return cfg_.n_shared_experts == 1; }

  // The decode fast path: same contract, no host round-trip. tokens
  // must fit decode_slots. Bitwise-equal to enqueue() at the same
  // routing (the unit gate's pin); traces land async in `trace` (null:
  // no trace copies at all).
  // table_slot >= 0 is capture mode: the kernels read graph slot's OWN
  // device table, uploaded beforehand by prepare_graph_table() — no
  // upload node is recorded, so a replay moves no table bytes at all.
  void enqueue_decode(const uint16_t* hidden, uint16_t* out, int tokens,
                      MoeTraceStaging* trace, cudaStream_t stream,
                      int table_slot = -1);
  // The decode fast path's ROUTED chain alone, handed back unrounded in
  // fp32 (out [tokens, hidden]) — the Qwen decode (its BF16 shared expert
  // continues the chain, models/qwen/moe_layer.cpp). Bitwise enqueue_f32
  // at the same routing. Works without a shared expert.
  void enqueue_decode_f32(const uint16_t* hidden, float* out, int tokens,
                          MoeTraceStaging* trace, cudaStream_t stream,
                          int table_slot = -1, bool accumulate = true);
  // The routed chain's per-slot down outputs (slot layout tokens *
  // (top_k + 1)) and the router weights of the most recent decode enqueue,
  // for a consumer that folds the accumulation itself (accumulate = false).
  const float* decode_slot_down() const { return d_slot_down_; }
  const float* decode_weights() const { return d_weights_; }

  // The Mixed346 contract's expert-input codes (cfg.routed_int8_activations;
  // null otherwise): a caller that produces the token rows' codes and
  // scales itself — the post-attention norm fused with the quantizer
  // (glm_rmsnorm_bf16_quant_int8) — writes them here ([rows, hidden] int8 at
  // stride hidden, [rows, hidden / 128] fp32) and marks them ready; the
  // next decode or tensor-core prefill enqueue then skips its own quantizer
  // (the gathered-row GEMV chain quantizes its gathered rows regardless).
  // The mark is consumed by that enqueue.
  int8_t* hidden_codes() const { return d_hidden_q_; }
  float* hidden_code_scales() const { return d_hidden_qs_; }
  void mark_hidden_codes_ready(bool ready) { hidden_codes_ready_ = ready; }

  // Fills graph slot `table_slot`'s device expert-view table from the
  // CURRENT binding (an async H2D on `stream`; the caller syncs before
  // capturing). Must run OUTSIDE stream capture, once per slot, before
  // the capture that records the slot; the contents freeze from then on
  // (resident bindings). Rebinding to different weights afterwards
  // without re-preparing is the wrong-weights class — enqueue_decode
  // refuses an unprepared slot.
  void prepare_graph_table(int table_slot, cudaStream_t stream);

  // Host copies of the most recent enqueue's routing decision. last_biased()
  // holds every expert's biased score for the same enqueue
  // ([tokens, n_experts], fp32) — the near-tie certification inputs.
  const std::vector<int32_t>& last_ids() const { return h_ids_; }
  const std::vector<float>& last_weights() const { return h_weights_; }
  const std::vector<float>& last_biased() const { return h_biased_; }
  const GlmMoeConfig& config() const { return cfg_; }
  int max_tokens() const { return max_tokens_; }

  // The bytes the constructor allocates for this shape (2026-09-06, the
  // memory plan): device scratch, and the pinned host staging in
  // `*pinned_bytes` (optional). The formula mirrors the constructor line
  // for line so a plan can be checked before anything is allocated.
  static size_t scratch_bytes(const GlmMoeConfig& cfg, int max_tokens,
                              int decode_slots = 0, int graph_table_slots = 0,
                              size_t* pinned_bytes = nullptr);
  // Additional lazy workspace for NVFP4 experts. Call only for that format.
  // calibrated says the loader supplies a static activation scale; the
  // reservation follows DGPP_MOE_W4A4 alone (opt-in, issue #68), so the
  // default and unsupported shapes need no W4A4 workspace.
  static size_t w4a4_scratch_bytes(const GlmMoeConfig& cfg, int max_tokens,
                                   bool calibrated = false);

  // Hash routing for the NEXT enqueues (DeepSeek-V4-Flash's first layers):
  // a token's experts come from `tid2eid` (int32 [vocab, top_k], rows
  // ascending) at `tokens` (device int64, one per hidden row) instead of
  // the top-k of its scores; the weights stay the normalized scores. Null
  // table: the router's own selection (the default). The caller sets it
  // before a hash layer's enqueue and clears it after; under capture the
  // pointers are baked (stable device buffers).
  void set_hash_routing(const int32_t* tid2eid, const int64_t* tokens, int vocab) {
    hash_tid2eid_ = tid2eid;
    hash_tokens_ = tokens;
    hash_vocab_ = vocab;
  }

  // Streaming-weight interface (M4 diagnostic forward): swap the device weight
  // views (router gate/bias, expert and shared matrices). Device scratch
  // and segmentation buffers are shape-keyed and unaffected.
  void rebind(const GlmMoeWeights& w) { w_ = w; }

 private:
  void check_expert_geometry() const;
  void copy_route_trace(MoeTraceStaging* trace, int tokens, cudaStream_t stream, int output_row = 0);
  // enqueue()/enqueue_f32(): exactly one of out_bf16 / out_f32 is set.
  void enqueue_host(const uint16_t* hidden, uint16_t* out_bf16, float* out_f32,
                    int tokens, cudaStream_t stream, MoeExpertKernel kernel);
  void accumulate_grouped(uint16_t* out_bf16, float* out_f32, int shared_row0, int tokens,
                          cudaStream_t stream);
  // enqueue_decode / enqueue_decode_f32: exactly one of out_bf16 (the
  // full chain with the shared expert) / out_f32 (the routed chain) is set.
  void enqueue_decode_impl(const uint16_t* hidden, uint16_t* out_bf16,
                           float* out_f32, int tokens, MoeTraceStaging* trace,
                           cudaStream_t stream, int table_slot, bool accumulate = true);
  // The eager paths' expert-table upload: fills the ring's next pinned
  // entry with every routed expert's three views (and the shared expert's
  // three after them when with_shared) and copies it to d_dst on stream —
  // waiting first, only if the host is kViewRing uploads ahead of the
  // stream, for that entry's previous upload to have executed.
  void upload_expert_views(MoeExpertView* d_dst, bool with_shared,
                           cudaStream_t stream);
  // The grouped chain shared by enqueue() and enqueue_prefill(): gather,
  // gate/up over the routed segments and the shared segment, swiglu, the
  // down projection — on the chosen kernel. accumulate_grouped consumes the
  // resulting buffer in its actual element format.
  // `tiles` (with its device count and host capacity): the compact tile
  // list of the routed segments for the wide packed kernel
  // (launch_moe_tile_list; null = the segment-major grid over max_rows).
  void grouped_expert_chain(MoeExpertKernel kernel, const uint16_t* hidden,
                            const MoeSegment* segs, int n_segs, int max_rows,
                            const MoeSegment* shared_seg, int tokens,
                            size_t rows_total, cudaStream_t stream,
                            const MoeTile* tiles = nullptr,
                            const int32_t* tile_count = nullptr, int tile_cap = 0);
  // Whether the routed chain's tensor-core launches take the tile list: the
  // wide packed int4 kernel (engine.expert_tile_list = false keeps the max_rows grid).
  bool packq_tile_list() const;
  int tile_list_for(MoeExpertKernel kernel, int n_segs, int routed_rows, cudaStream_t stream);
  // Longest routed segment for grid sizing (the grouped launchers document
  // max_rows as the longest segment's row count): reads the device segment
  // table back into the pinned staging and takes the max over the routed
  // experts. DGPP_MOE_SEG_MAX=0 keeps the old tokens-wide bound.
  int routed_seg_max_rows(const MoeSegment* d_segs, int n_segs, int fallback,
                          cudaStream_t stream);

  // The router, then (hash routing) the routes replaced from the table.
  void route(const uint16_t* hidden, int tokens, cudaStream_t stream, int* counters = nullptr,
             int token_row = 0);
  void prefill_chain(const uint16_t* hidden, int tokens, cudaStream_t stream, int route_row = 0);
  bool shared_aside_takes(bool with_shared, int tokens) const;
  const int32_t* hash_tid2eid_ = nullptr;
  const int64_t* hash_tokens_ = nullptr;
  int hash_vocab_ = 0;

  GlmMoeWeights w_;
  GlmMoeConfig cfg_;
  int max_tokens_;
  int decode_slots_ = 0;

  // device scratch (cudaMalloc; sized to max_tokens)
  int32_t* d_ids_ = nullptr;
  float* d_weights_ = nullptr;
  float* d_biased_ = nullptr;  // [max_tokens, n_experts] biased router scores
  float* d_scores_ = nullptr;  // [max_tokens, n_experts] sigmoid scores (router scratch)
  int32_t* d_rows_ = nullptr;
  float* d_row_w_ = nullptr;
  uint16_t* d_gather_ = nullptr;
  uint16_t* d_gate_ = nullptr;
  uint16_t* d_up_ = nullptr;
  uint16_t* d_act_ = nullptr;
  float* d_down_ = nullptr;  // [max_tokens, hidden] fp32 segment output
  // W4A4 prefill (DGPP_MOE_W4A4=1, 2026-09-23): the NVFP4 activation
  // buffers, grown on first use (eager prefill only, never under capture).
  uint8_t* d_q_codes_ = nullptr;
  uint8_t* d_q_scales_ = nullptr;
  float* d_q_gs_ = nullptr;
  size_t q_rows_cap_ = 0, q_k_cap_ = 0;
  bool down_bf16_ = false;  // the last chain wrote bf16 down rows into d_down_ (W4A4, no shared segment)
  void ensure_w4a4(size_t rows, int k);
  float* d_acc_ = nullptr;   // [max_tokens, hidden] the fp32 chain

  // decode-slot scratch (cudaMalloc; sized to decode_slots*(top_k+1) rows —
  // the slot layout the kernels index: routed K + shared, per token)
  uint16_t* d_slot_act_ = nullptr;  // [slots, inter] (fused gate/up/swiglu)
  uint16_t* d_hidden_rot_ = nullptr;  // [decode_slots, hidden]: the token rows rotated by H32 (NF4I8)
  // The Mixed346 contract's int8 activation codes and fp32 scales
  // (cfg.routed_int8_activations; kernels/hadamard32.cu): the expert-input
  // rows [max(rows_total, decode_slots), hidden] and the SwiGLU rows
  // [max(rows_total, decode rows), inter] — decode and prefill share them.
  int8_t* d_hidden_q_ = nullptr;
  float* d_hidden_qs_ = nullptr;
  bool hidden_codes_ready_ = false;  // mark_hidden_codes_ready: the next enqueue reads them as given
  int8_t* d_act_q_ = nullptr;
  float* d_act_qs_ = nullptr;
  // The EXL3 contract's rotated fp16 copies (cfg.exl3_experts;
  // kernels/exl3_gemv.hpp): the expert-input rows by the gate suh and by
  // the up suh [rows_x, hidden], the SwiGLU rows by the down suh [rows_x,
  // inter] — the decode slots and the grouped chain's gathered rows share them.
  uint16_t* d_xh_g_ = nullptr;
  uint16_t* d_xh_u_ = nullptr;
  uint16_t* d_xh_d_ = nullptr;
  float* d_slot_down_ = nullptr;
  int32_t* d_slot_order_ = nullptr; // [slots] expert-sorted execution order
  int32_t* d_slot_runs_ = nullptr;  // [slots + groups] the equal-expert run length at each position, then the long runs' group list (EXL3 run form)
  int* d_tile_counter_ = nullptr;   // [2] the fp4 down kernel's stealing counters, zero at rest
  // GlmMoeConfig::shared_mma_aside: the shared expert's own chain.
  uint16_t* d_sh_gate_ = nullptr;   // [decode_slots, inter]
  uint16_t* d_sh_up_ = nullptr;
  uint16_t* d_sh_act_ = nullptr;
  cudaStream_t sh_side_[2] = {nullptr, nullptr};  // gate (then the activation and down), up
  cudaEvent_t sh_fork_ = nullptr;
  cudaEvent_t sh_up_done_ = nullptr;
  cudaEvent_t sh_join_ = nullptr;
  int* d_router_counters_ = nullptr;  // [decode_slots] fused-select tickets    // [slots, hidden] fp32 partial dots
  // The device expert-view table, re-uploaded per enqueue_decode call.
  // NO CACHE, DELIBERATELY: the streaming loader refills one
  // GlmLayerResident per layer, so a binding-keyed cache collides across
  // layers (the first MoE layer's table served to every layer after it
  // — glm_tp_test's decode-parity gate caught exactly that, an
  // uncertifiable top-1 flip with infinite margin). One ~14KB async
  // upload per MoE layer per step buys lifetime correctness with zero
  // cleverness; the graph era bakes the tables in properly.
  MoeExpertView* d_expert_views_ = nullptr;  // [n_experts * 3]

  // host staging
  std::vector<int32_t> h_ids_;
  std::vector<float> h_weights_;
  std::vector<float> h_biased_;  // [tokens, n_experts] (certification)
  std::vector<int32_t> h_rows_;
  std::vector<float> h_row_w_;
  std::vector<int> h_counts_;
  // The prefill path's staging: PINNED, so its copies are
  // truly asynchronous — a pageable async copy is a synchronous staged
  // copy, and the per-expert form of it cost 1.4 s of a 5 s 256-token
  // prefill. The router's ids/weights/biased come down into the pinned
  // mirrors (then into the vectors above, which callers keep references
  // to); the segmented rows/weights go up ONCE per layer from the pinned
  // arrays, each segment addressed by offset; the shared expert's identity
  // rows and unit weights live on the device from construction.
  // The grouped expert path: the layer's rows — every routed
  // (token, slot) in ascending-expert segment order, then the tokens once
  // more for the shared expert — gathered at once; one grouped launch per
  // matrix over the routed segments and one over the shared segment; the
  // per-token ordered accumulation in one pass. Buffers are sized to
  // max_tokens * (top_k + 1) rows.
  int32_t* h_ids_pinned_ = nullptr;      // [max_tokens * top_k]
  float* h_weights_pinned_ = nullptr;    // [max_tokens * top_k]
  float* h_biased_pinned_ = nullptr;     // [max_tokens * n_experts]
  int32_t* h_seg_rows_ = nullptr;        // [rows_total], pinned
  int32_t* h_slot_row_ = nullptr;        // [max_tokens * top_k], pinned
  MoeSegment* h_segs_ = nullptr;         // [n_experts + 1], pinned
  int32_t* d_slot_row_ = nullptr;        // [max_tokens * top_k]
  MoeSegment* d_segs_ = nullptr;         // [n_experts + 1]
  MoeExpertView* d_views_prefill_ = nullptr;  // [(n_experts + 1) * 3]
  MoeTile* d_tiles_ = nullptr;           // [tiles_cap_] the compact tile list
  int32_t* d_tile_count_ = nullptr;      // [1]
  int tiles_cap_ = 0;
  // The eager paths' expert-view upload source (the prefill paths' table
  // of n_experts + 1 rows and eager decode's of n_experts): a RING of
  // pinned tables, each guarded by the event its last upload recorded.
  // PINNED, not a plain vector: a pageable-source cudaMemcpyAsync performs
  // a stream sync before the copy initiates (driver contract), which
  // drains the whole step's pipeline once per MoE layer — the first fabric
  // run of the fused path paid 49ms/token for exactly that. Pinned sources
  // are true async DMA — and so a write-after-read hazard (found
  // 2026-09-05): one layer object is rebound for every MoE layer, and a
  // host that has run ahead of the stream (the session prefill has no
  // per-layer sync, and a world of one has no collective to wait on)
  // refilled the single table with the NEXT layer's pointers before the
  // previous layer's copy had executed, so that layer's expert chain ran
  // on the wrong experts' weights — a fresh model's first prefill and its
  // second disagreed by 0.19 rel_l2 (glm_tp_first_run_is_bitwise_the_
  // second_run pins it). Each fill takes the ring's next entry and waits
  // — only when the host is kViewRing uploads ahead — for that entry's
  // previous upload to have executed; the copy is enqueued right after a
  // layer's router and segmentation kernels, so the wait admits a host
  // several layers ahead and never drains the stream. Capture-mode decode
  // reads its per-slot graph tables instead and never touches the ring.
  static constexpr int kViewRing = 4;

 public:
  // The opt-in prefill levers (2026-09-30, engine.prefill_bf16_partials /
  // engine.prefill_fold_scales; both default off, NOT bitwise the default
  // chain — the served transcripts can differ). Set before the first
  // chain; process-wide. bf16_partials: the packed tensor-core chain's
  // down projection written in bf16 and its per-expert partials summed
  // from bf16 (half the bytes a chunk writes and reads back; the reference
  // stack's form). fold_scales: the wide packed GEMM with each group's
  // scale folded into the bf16 weight values and one fp32 accumulator
  // (kernels/packq_gemm variant 3).
  // tile_list (engine.expert_tile_list, default on): the compact tile list
  // for the wide kernels instead of the max_rows grid (bitwise). pair
  // (engine.expert_gemm_pair, default off): gate and up as one launch
  // (bitwise; measured level). No environment switch sets any of these.
  static void set_prefill_options(bool bf16_partials, bool fold_scales, bool tile_list = true, bool pair = false);
  static bool prefill_bf16_partials();
  static bool prefill_fold_scales();
  size_t view_table_entries_ = 0;         // (n_experts + 1) * 3
  MoeExpertView* h_view_ring_ = nullptr;  // [kViewRing][view_table_entries_]
  cudaEvent_t view_ring_event_[kViewRing] = {};
  bool view_ring_armed_[kViewRing] = {};
  int view_ring_next_ = 0;
  // Per-graph-slot tables (capture mode): [graph_table_slots_] rows of
  // [n_experts * 3] each. The pinned rows are the upload sources, the
  // device rows are what the recorded kernels read — each slot its own,
  // uploaded once by prepare_graph_table() (a replay copies nothing; the
  // former per-replay upload node cost 42 H2D nodes per token).
  MoeExpertView* h_expert_views_graph_ = nullptr;
  MoeExpertView* d_expert_views_graph_ = nullptr;
  std::vector<bool> graph_table_ready_;
  int graph_table_slots_ = 0;
};

}  // namespace dgpp
