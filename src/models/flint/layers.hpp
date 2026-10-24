#pragma once
// Flint layers composed from the shared kernels (2026-10-22,
// docs/flint_plan.md §3, D3, D4): shape-specific scratch, rebindable
// weight views, fixed decode grids (graph-capturable), streaming rebinds.
//
//   FlintAttentionLayer  the fused BF16 [Q | K | V] projection through the
//                        GEMM interface (fp32 out — the Linear's one rounding
//                        is the finish's; the 12-bit companion at decode
//                        rows), the finish (the per-head one-rounding q / k
//                        norms, RoPE over the whole head — none on the
//                        no-RoPE full-attention layers — the paged K/V
//                        append), the split-KV GQA attention with the window
//                        and the per-head sink (kernels/sink_attn.hpp at
//                        qk 128), the combine, the BF16 o_proj
//   FlintBf16Mlp         the BF16 SwiGLU MLP of the dense layers (gate / up
//                        through the GEMM interface, the swiglu, down)
// The MoE layers run QwenMoeLayer (models/qwen/moe_layer.hpp): the sigmoid
// router with the zero bias, the routed chain in the layer's per-projection
// fp8 / NVFP4 formats, the ungated BF16 shared expert on its tail.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "common/paged_attention_rows.hpp"
#include "kernels/gemm.hpp"
#include "models/flint/config.hpp"
#include "models/flint/loader.hpp"
#include "models/mimo/kv_pool.hpp"

namespace dgpp {

// The GEMM interface's workspace, shared by every layer object of a model.
struct FlintGemmWorkspace {
  IGemm* gemm = nullptr;
  void* ws = nullptr;
  size_t ws_bytes = 0;
};

// The rows one attention enqueue serves (common/paged_attention_rows.hpp):
// device [T] request ids and positions (a padding row has pos < 0), decode
// rows (every row's own split geometry, kernels-only) or a prefill chunk.
using FlintAttnRows = PagedAttentionRows;
// The paged K/V pool's per-layer view (models/mimo/kv_pool.hpp: the same
// pool at 128 / 128 heads, bf16 or the fp8 row form).
using FlintKvCache = MimoKvCache;

class FlintAttentionLayer {
 public:
  // decode_rows: the most rows a decode enqueue carries; n_split_decode:
  // the split-KV geometry of a full-attention layer's decode rows (a
  // sliding-window layer's rows split at most per 32-token tile of its
  // window) — the partials' workspace holds max(max_tokens, decode_rows *
  // n_split_decode) row-splits; prefill rows run one split. rebind()
  // switches the window, the RoPE and the sink with the weights.
  FlintAttentionLayer(const FlintAttnResident& w, const FlintGemmWorkspace& gemm, const FlintTextConfig& cfg,
                      int max_tokens, int decode_rows, int n_split_decode);
  ~FlintAttentionLayer();
  FlintAttentionLayer(const FlintAttentionLayer&) = delete;
  FlintAttentionLayer& operator=(const FlintAttentionLayer&) = delete;
  void rebind(const FlintAttnResident& w);

  // out[T, H] = o_proj(attention(x[T, H])): the projection, the finish
  // (K/V of every row appended to `cache` at its slot), the attention over
  // the row's visible range, the output projection into `out` (a partial
  // at world > 1).
  void enqueue(const uint16_t* x, int tokens, const FlintAttnRows& rows, const FlintKvCache& cache, uint16_t* out,
               cudaStream_t stream);

  static size_t scratch_bytes(const FlintTextConfig& cfg, int local_heads, int local_kv_heads, int max_tokens,
                              int decode_rows, int n_split_decode);
  // The decode split count of the full-attention layers (32: at every
  // world eight query heads share a kv head, a T=1 row is 32 blocks).
  static constexpr int default_decode_splits() { return 32; }
  // The split count a layer's decode rows take: the window's tiles for a
  // sliding-window layer, n_split_decode for a full-attention one.
  int decode_splits_of(const FlintAttnResident& w) const;
  int local_heads() const { return lh_; }
  int local_kv_heads() const { return lkv_; }

 private:
  FlintAttnResident w_;
  FlintGemmWorkspace g_;
  int hidden_, lh_, lkv_, max_tokens_, decode_rows_, n_split_;
  int window_ = 0;
  float eps_ = 1e-6f;
  size_t part_rows_ = 0;   // row-splits the partial workspaces hold
  int64_t qkv_cols_ = 0;   // the fp32 projection's row width: (lh + 2 lkv) x 128
  float scale_ = 1.0f;
  float* d_inv_freq_ = nullptr;  // [64] the rotary table over the whole head
  float* qkv_ = nullptr;         // [M, qkv_cols] fp32
  uint16_t* q_ = nullptr;        // [M, lh * 128] bf16 (finished)
  float* m_ws_ = nullptr;        // [part_rows, lh]
  float* l_ws_ = nullptr;
  float* c_ws_ = nullptr;        // [part_rows, lh, 128]
  int* counters_ = nullptr;      // [decode_rows, kv heads] the fused kernel's arrival counters (zero at rest)
  uint16_t* o_ = nullptr;        // [M, lh * 128] bf16
};

class FlintBf16Mlp {
 public:
  FlintBf16Mlp(const FlintBf16MlpResident& w, const FlintGemmWorkspace& gemm, const FlintTextConfig& cfg,
               int max_tokens);
  ~FlintBf16Mlp();
  FlintBf16Mlp(const FlintBf16Mlp&) = delete;
  FlintBf16Mlp& operator=(const FlintBf16Mlp&) = delete;
  void rebind(const FlintBf16MlpResident& w);
  // out[T, H] = down(bf16(bf16(silu(gate(x))) * up(x))) — the reference's
  // bf16 Linears (one rounding each) and the swiglu's two roundings, no
  // clamps (a partial at world > 1).
  void enqueue(const uint16_t* x, int tokens, uint16_t* out, cudaStream_t stream);
  static size_t scratch_bytes(int64_t local_inter, int max_tokens);

 private:
  FlintBf16MlpResident w_;
  FlintGemmWorkspace g_;
  int hidden_, max_tokens_;
  int64_t inter_;
  uint16_t* gate_ = nullptr;  // [M, I]
  uint16_t* up_ = nullptr;    // [M, I]
  uint16_t* act_ = nullptr;   // [M, I]
};

}  // namespace dgpp
