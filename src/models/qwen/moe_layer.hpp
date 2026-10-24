#pragma once
// Qwen3.8-Flash-Next MoE: softmax top-k routed experts with optional
// renormalization, plus a shared expert weighted by sigmoid(x . g).
// The checkpoint uses 512 routed experts and top-10 selection.
//
// GlmMoeLayer computes the routed FP8 or NVFP4 expert contributions with
// SoftmaxTopk routing and no SwiGLU clamps. This layer adds the shared
// expert, stored as BF16 or optional block FP8, to the unrounded FP32 fmaf
// chain and rounds to BF16 once. Diagnostic, device-slot decode and grouped
// prefill paths preserve that accumulation contract.
//
// At TP world W, every rank holds each expert's I/W intermediate slice
// and the shared expert's S/W slice. The router and shared gate are
// replicated. FP8 expert scales use gcd(128, I/W) on the sliced axis.
// The result is a partial hidden vector for the FFN all-reduce.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "kernels/gemm.hpp"
#include "models/glm/moe.hpp"
#include "models/glm/moe_layer.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

// Device-visible views (the loader's QwenMoeResident wires them).
struct QwenMoeWeights {
  const uint16_t* router = nullptr;            // bf16 [n_experts, hidden]
  const uint16_t* shared_gate = nullptr;       // bf16 [hidden] (shared_expert_gate); null: weight 1 (Flint's ungated shared expert)
  const uint16_t* shared_gate_proj = nullptr;  // bf16 [S, hidden]
  const uint16_t* shared_up_proj = nullptr;    // bf16 [S, hidden]
  const uint16_t* shared_down_proj = nullptr;  // bf16 [hidden, S]
  const GlmQuantMatrix* shared_fp8 = nullptr;  // dense_weights fp8: gate, up, down (the bf16 three null)
  int64_t shared_inter = 0;                    // S: this rank's shared slice
  const GlmQuantMatrix* experts = nullptr;     // [n_experts * 3] gate, up, down (FP8 block form)
  const GlmFp4Matrix* experts_fp4 = nullptr;   // the NVFP4 form instead
  const GlmPackedMatrix* experts_packed = nullptr;  // the packed-int form (the AutoRound hybrid); exactly one of the three is set
  // The per-projection routed form (2026-10-22, Flint): `experts` and
  // `experts_fp4` both bound, the mask naming the NVFP4 projections
  // (models/glm/moe.hpp). The sigmoid router's bias (SigmoidBias mode,
  // Flint: a zero vector); null under SoftmaxTopk.
  uint8_t routed_fp4_mask = 0;
  const float* router_bias = nullptr;
  float act_scale_w13 = 0.0f, act_scale_w2 = 0.0f;  // W4A4 activation scales (0: dynamic)
  const float* act_scales_dev = nullptr;             // [2] on the device, from the layer image
  bool experts_weight_only = false;                  // W4A16_NVFP4 (the QAD export's draft experts): no W4A4
};

class QwenMoeLayer {
 public:
  // The routed chain's configuration: SoftmaxTopk router, no shared
  // expert in the chain, no scaling factor, no swiglu clamps.
  static GlmMoeConfig routed_config(int hidden, int inter, int n_experts,
                                    int top_k, bool norm_topk_prob);

  // gemm: the model's GEMM interface (the BF16 shared expert's three products;
  // decode shapes take the in-house GEMV inside it). max_tokens bounds
  // enqueue()'s rows.
  // decode_slots / graph_table_slots: the routed layer's decode-path
  // provisioning (models/glm/moe_layer.hpp) — 0 keeps the host path only.
  QwenMoeLayer(const QwenMoeWeights& weights, const GlmMoeConfig& cfg,
               IGemm& gemm, int max_tokens, int decode_slots = 0,
               int graph_table_slots = 0);
  ~QwenMoeLayer();
  // The shared expert's fp8 projections: the streaming tensor-core GEMM
  // from this row count (scale_gemm.hpp; 0: the GEMV chunks to 128 rows).
  void set_mma_from_rows(int rows) { mma_from_rows_ = rows; }
  void set_prefill_dense_tile(bool enabled) { prefill_dense_tile_ = enabled; }
  QwenMoeLayer(const QwenMoeLayer&) = delete;
  QwenMoeLayer& operator=(const QwenMoeLayer&) = delete;

  // out[tokens, hidden] = bf16(sum_e w_e down_e(...) + sigma(x.g) shared(x)),
  // the fp32 chain in ascending expert order with the shared expert last.
  // Synchronizes the stream once (the routed layer's host segmentation).
  void enqueue(const uint16_t* hidden, uint16_t* out, int tokens,
               cudaStream_t stream,
               MoeExpertKernel kernel = MoeExpertKernel::kGemv);

  // The routed layer (its last_ids/last_weights/last_biased are this
  // enqueue's routing decision; last_biased holds the bf16 logits).
  // The decode fast path (Q6): the routed chain off the device route (no
  // host round-trip; table_slot >= 0 reads a graph slot's prepared table),
  // then the BF16 shared expert and the single rounding — bitwise
  // enqueue() at the same routing. tokens <= decode_slots.
  void enqueue_decode(const uint16_t* hidden, uint16_t* out, int tokens,
                      cudaStream_t stream, int table_slot = -1, MoeTraceStaging* trace = nullptr);
  // The prefill path (Q7): the routed chain device-segmented on the
  // tensor-core kernel (the slice's 32/64 scale grid included, no host
  // round-trip), then the BF16 shared expert and the one rounding.
  // The routing (ids, weights) lands async in `trace` (pinned; read after
  // the stream's next sync) when given; last_ids()/last_weights() are not
  // updated by this path.
  void enqueue_prefill(const uint16_t* hidden, uint16_t* out, int tokens,
                       cudaStream_t stream, MoeTraceStaging* trace = nullptr);
  void prepare_graph_table(int table_slot, cudaStream_t stream) {
    routed_.prepare_graph_table(table_slot, stream);
  }
  const GlmMoeLayer& routed() const { return routed_; }
  GlmMoeLayer& routed() { return routed_; }
  const GlmMoeConfig& config() const { return cfg_; }

  // Streaming-weight interface: swap the views (shapes unchanged).
  void rebind(const QwenMoeWeights& w);

  // Bytes the constructor allocates (device; pinned in *pinned_bytes),
  // the routed layer's included — the memory plan's line.
  static size_t scratch_bytes(const GlmMoeConfig& cfg, int64_t shared_inter, int max_tokens,
                              size_t* pinned_bytes = nullptr, int decode_slots = 0,
                              int graph_table_slots = 0);

 private:
  static GlmMoeWeights routed_view(const QwenMoeWeights& w);
  void check_weights() const;
  // The shared expert's fp8 scale grid (log2 block rows / cols) and whether
  // it is the 128 x 128 form the block launchers take.
  void shared_grid(int& rs, int& cs) const;
  bool shared_block128() const;
  // The chain's tail after the routed experts left d_acc_: the BF16 shared
  // expert, its sigmoid weight, the last fma and the one rounding.
  void shared_tail(const uint16_t* hidden, uint16_t* out, int tokens, cudaStream_t stream, bool prefill = false);

  bool prefill_dense_tile_ = false;
  QwenMoeWeights w_;
  GlmMoeConfig cfg_;
  IGemm& gemm_;
  // The shared expert's gate/up half on a side stream beside the routed
  // experts (2026-09-29): it reads only the layer's input, and its 21 us
  // launch ran on the chain after the routed stream at 55–61 % of the
  // line rate (a small matrix's ramp). Forked at enqueue_decode's start,
  // joined before the down half. DGPP_QWEN_SHARED_SIDE=off keeps the chain.
  cudaStream_t shared_side_ = nullptr;
  cudaEvent_t shared_fork_ = nullptr, shared_join_ = nullptr;
  int mma_from_rows_ = 0;
  int max_tokens_;
  GlmMoeLayer routed_;

  float* d_acc_ = nullptr;      // [M, H] the fp32 chain
  uint16_t* d_sgate_ = nullptr; // [M, S]
  uint16_t* d_sup_ = nullptr;   // [M, S]
  uint16_t* d_sact_ = nullptr;  // [M, S]
  float* d_sdown_ = nullptr;    // [M, H] the shared down projection, unrounded
  float* d_sw_ = nullptr;       // [M] sigma(x . g), a bf16 value
  int32_t* d_rows_ = nullptr;   // [M] identity (the accumulation's row map)
  void* gemm_ws_ = nullptr;
  uint16_t* d_shared_bridge_ = nullptr;  // dense_weights fp8: a shared matrix dequantized for prefill
  size_t gemm_ws_bytes_ = 0;
};

}  // namespace dgpp
