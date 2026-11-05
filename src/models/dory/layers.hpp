#pragma once
#include "common/paged_attention_rows.hpp"
// Dory layers composed from CUDA kernels, with shape-specific scratch and
// rebindable weight views (docs/dory_plan.md §1.3–§1.4, D5). One physical
// layer is ONE sublayer — attention or MLP, never both — so the walk holds
// one object of each and rebinds it per layer:
//
//   DoryAttentionLayer  the q/k/v projections, the gate projection, the
//                       finish (neoX RoPE at the layer's theta, the per-head
//                       justnorm, the sqk square, the paged K/V append), the
//                       split-KV GQA over the layer's window, the sigmoid
//                       output gate, the 4096-wide justnorm, the o_proj, the
//                       slerp residual
//   DoryMlpLayer        gate/up (two bf16 projections), the suv-scaled
//                       silu-times-up, the 7168-wide justnorm, down_proj,
//                       the slerp residual
//
// Every projection is a packed NVFP4 W4A16 linear through the fp4 launchers
// (a Linear's output is bf16 — the reference's own dtype, and the only
// prefill form the fp4 path has): the GEMV core on decode rows, the dense
// MMA on prefill rows. There is no GEMM workspace to thread through.
//
// The theta and the window follow the layer kind, so rebind() takes the
// physical layer index (plan D6, D7). Decode rows split the visible range
// (fixed grids without host readbacks, so the walk captures); prefill rows
// run one split each.
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "models/dory/config.hpp"
#include "models/dory/kv_pool.hpp"
#include "models/dory/loader.hpp"

namespace dgpp {

// The rows one attention enqueue serves: device [T] request ids and
// positions (a padding row has pos < 0).
using DoryAttnRows = PagedAttentionRows;

// The rotary inverse frequencies of one theta, as the reference computes
// them: f[i] = theta ** (-2i/dim) in DOUBLE, narrowed to fp32 (the host
// reference's rounding — qsa_rope_inv_freq's float pow can differ by an ulp).
void dory_rope_inv_freq(double theta, int dim, float* out);

class DoryAttentionLayer {
 public:
  DoryAttentionLayer(const DoryAttentionResident& w, const DoryConfig& cfg, int layer, int max_tokens,
                     int decode_rows, int n_split_decode, cudaStream_t stream = nullptr);
  ~DoryAttentionLayer();
  DoryAttentionLayer(const DoryAttentionLayer&) = delete;
  DoryAttentionLayer& operator=(const DoryAttentionLayer&) = delete;
  // The weights and the layer kind (its window and its RoPE theta). The
  // inv-frequency upload is ordered on `stream` (null: the legacy stream).
  void rebind(const DoryAttentionResident& w, int layer, cudaStream_t stream = nullptr);

  // out[T, H] = slerp_alpha(x, o_proj(justnorm(attn * sigmoid(gate))), alpha):
  // the projections, the finish (K/V of every row appended to `cache`), the
  // attention over the row's visible range, the gated justnorm, o_proj and
  // the residual into `out`.
  void enqueue(const uint16_t* x, int tokens, const DoryAttnRows& rows, const DoryKvCache& cache,
               uint16_t* out, cudaStream_t stream);

  static size_t scratch_bytes(const DoryConfig& cfg, int max_tokens, int decode_rows, int n_split_decode);
  // The default decode split count (DGPP_DORY_ATTN_SPLITS overrides).
  static int default_decode_splits();
  int layer() const { return layer_; }
  int window() const { return window_; }
  double theta() const { return theta_; }
  int head_dim() const { return dim_; }

 private:
  const DoryAttentionResident* w_ = nullptr;
  DoryConfig cfg_;  // the layer kinds: the window and the theta follow rebind()
  int hidden_, lh_, lkv_, dim_, max_tokens_, decode_rows_, n_split_, layer_;
  int window_ = 0;
  double theta_ = 1e4;
  float scale_, sqk_factor_, s_gate_factor_, alpha_scale_, eps_;
  size_t part_rows_ = 0;
  float* d_inv_freq_ = nullptr;  // [dim/2] the theta table of the rebound layer
  // The two rope tables, staged once at construction (the theta depends only
  // on the layer's kind, config.hpp rope_theta_of): rebind selects one, so
  // the decode walk uploads NOTHING — a graph capture records an H2D copy as
  // a memcpy node, and the decode graph must be kernels-only
  // (engine/graph_check.hpp). Aliased when the two thetas are equal.
  float* d_inv_freq_full_ = nullptr;
  float* d_inv_freq_sliding_ = nullptr;
  uint16_t* qd_ = nullptr;       // [M, Q] bf16 (the q_proj output)
  uint16_t* kd_ = nullptr;       // [M, KV] bf16
  uint16_t* vd_ = nullptr;       // [M, KV] bf16
  uint16_t* q_ = nullptr;        // [M, Q] bf16 (finished, sqk-scaled)
  uint16_t* gate_ = nullptr;     // [M, Q] bf16 (g_proj)
  uint16_t* attn_ = nullptr;     // [M, Q] bf16 (the combined attention)
  uint16_t* normed_ = nullptr;   // [M, Q] bf16 (the gated justnorm)
  uint16_t* o_ = nullptr;        // [M, H] bf16 (o_proj)
  float* m_ws_ = nullptr;        // [part_rows, lh]
  float* l_ws_ = nullptr;
  float* c_ws_ = nullptr;        // [part_rows, lh, dim]
};

class DoryMlpLayer {
 public:
  DoryMlpLayer(const DoryMlpResident& w, const DoryConfig& cfg, int max_tokens, int decode_rows);
  ~DoryMlpLayer();
  DoryMlpLayer(const DoryMlpLayer&) = delete;
  DoryMlpLayer& operator=(const DoryMlpLayer&) = delete;
  void rebind(const DoryMlpResident& w);

  // out[T, H] = slerp_alpha(x, down_proj(justnorm(silu(g) * u)), mlp_alpha):
  // the two projections, the suv-scaled silu-times-up, the justnorm, down_proj
  // and the residual into `out`.
  void enqueue(const uint16_t* x, int tokens, uint16_t* out, cudaStream_t stream);

  static size_t scratch_bytes(const DoryConfig& cfg, int max_tokens);

 private:
  const DoryMlpResident* w_ = nullptr;
  DoryConfig cfg_;
  int hidden_, inter_, max_tokens_, decode_rows_;
  float suv_factor_, alpha_scale_, eps_;
  uint16_t* gate_ = nullptr;  // [M, I] bf16
  uint16_t* up_ = nullptr;    // [M, I] bf16
  uint16_t* m_ = nullptr;     // [M, I] bf16 (the gated justnorm)
  uint16_t* o_ = nullptr;     // [M, H] bf16 (down_proj)
};

}  // namespace dgpp
