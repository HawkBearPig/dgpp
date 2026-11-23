#pragma once
// Inkling 2 layer objects (docs/inkling2_mini_plan.md sections 1.4 to 1.6):
// the attention and the routed MoE of one layer visit. The objects own
// scratch only. Weights arrive with every call, because a layer of weights
// serves several visits and a visit names its own K/V plane and ring.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "common/paged_attention_rows.hpp"
#include "kernels/gemm.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "models/glm/moe.hpp"
#include "models/inkling2/config.hpp"
#include "models/inkling2/loader.hpp"
#include "models/mimo/kv_pool.hpp"

namespace dgpp {

struct Inkling2GemmWorkspace {
  IGemm* gemm = nullptr;
  void* ws = nullptr;
  size_t ws_bytes = 0;
};

// A visit's convolution rings: every request keeps kInk2RingSlots slots,
// each slot holds every visit's four streams back to back.
//   [K inputs (lkv x 128) | V inputs (lkv x 128) | attention output (hidden) | MLP output (hidden)]
struct Inkling2Ring {
  uint16_t* base = nullptr;     // BF16 [requests][kInk2RingSlots][slot_stride]
  int64_t request_stride = 0;   // elements per request
  int64_t slot_stride = 0;      // elements per slot
  int64_t visit_offset = 0;     // the visit's element offset inside a slot
  int64_t kv_width = 0;         // lkv x 128
  int hidden = 0;
  int64_t k_offset() const { return visit_offset; }
  int64_t v_offset() const { return visit_offset + kv_width; }
  int64_t attn_offset() const { return visit_offset + 2 * kv_width; }
  int64_t mlp_offset() const { return visit_offset + 2 * kv_width + hidden; }
  // Elements one visit takes in a slot.
  static int64_t visit_width(int64_t kv_width, int hidden) { return 2 * kv_width + 2 * static_cast<int64_t>(hidden); }
};

using Inkling2AttnRows = PagedAttentionRows;
using Inkling2KvCache = MimoKvCache;

class Inkling2AttentionLayer {
 public:
  // decode_rows: the most rows a decode enqueue carries. n_split_decode: the
  // split-KV geometry of a full-attention visit's decode rows (a local
  // visit's rows split at most per 32-token tile of the window). Prefill
  // rows run one split.
  Inkling2AttentionLayer(const Inkling2Config& cfg, const Inkling2LocalGeometry& geo,
                         const Inkling2GemmWorkspace& gemm, int max_tokens, int decode_rows, int n_split_decode);
  ~Inkling2AttentionLayer();
  Inkling2AttentionLayer(const Inkling2AttentionLayer&) = delete;
  Inkling2AttentionLayer& operator=(const Inkling2AttentionLayer&) = delete;

  // out[T, hidden] = o_proj(gate * attention): the three projections (the
  // query side from xq, keys from xk, values from xv, which may be one
  // buffer), the query finish, the K/V convolutions and cache append at
  // every row's slot, the attention over the row's visible range, the
  // output projection (a partial at world > 1).
  void enqueue(const Inkling2AttnResident& w, const uint16_t* xq, const uint16_t* xk, const uint16_t* xv, int tokens,
               const Inkling2AttnRows& rows, const Inkling2KvCache& cache, const Inkling2Ring& ring, uint16_t* out,
               cudaStream_t stream);

  static size_t scratch_bytes(const Inkling2Config& cfg, const Inkling2LocalGeometry& geo, int max_tokens,
                              int decode_rows, int n_split_decode);
  static constexpr int default_decode_splits() { return 32; }
  int decode_splits_of(bool local) const;

 private:
  Inkling2Config cfg_;
  Inkling2GemmWorkspace g_;
  int hidden_, lh_, lkv_, max_tokens_, decode_rows_, n_split_;
  int64_t q_side_rows_ = 0;   // lh x (2 x 128 + d_rel + 1)
  int64_t kv_rows_ = 0;       // lkv x (128 + 4): the widest K / V projection row
  int max_extent_ = 0;
  size_t part_rows_ = 0;
  float scale_ = 1.0f;
  uint16_t* q_side_ = nullptr;  // [M, q_side_rows]
  uint16_t* k_ = nullptr;       // [M, kv_rows]
  uint16_t* v_ = nullptr;       // [M, kv_rows]
  uint16_t* q_ = nullptr;       // [M, lh x 128]
  uint16_t* bias_ = nullptr;    // [M, lh, max extent]
  float* m_ws_ = nullptr;       // [part_rows, lh]
  float* l_ws_ = nullptr;
  float* c_ws_ = nullptr;       // [part_rows, lh, 128]
  uint16_t* o_ = nullptr;       // [M, lh x 128]
};

// The routed experts of one visit over the shared grouped kernels: the
// sigmoid router, device segmentation, the interleaved gate/up GEMM, the
// softcapped activation, the down GEMM and the ordered accumulation with
// one rounding. `native` quantizes both GEMM inputs as the checkpoint's
// precision metadata requires. The GEMMs then read the exact dequantized
// rows, and the NVFP4 row scales multiply the finished dots.
class Inkling2MoeLayer {
 public:
  Inkling2MoeLayer(const Inkling2Config& cfg, const Inkling2LocalGeometry& geo, int max_tokens, bool native,
                   bool resident);
  ~Inkling2MoeLayer();
  Inkling2MoeLayer(const Inkling2MoeLayer&) = delete;
  Inkling2MoeLayer& operator=(const Inkling2MoeLayer&) = delete;

  // Uploads the layer's device view table. A resident stack uploads each
  // layer once, outside any capture. A streaming stack uploads at every bind.
  void bind(int layer, const Inkling2MoeResident& w, cudaStream_t stream);
  // out[T, hidden] = the routed sum (a partial at world > 1). `layer` must
  // be bound. `decode`: the rows are decode or verification rows (a few rows
  // against whole expert matrices). Reader rows take the wide forms.
  // `positions` (nullable, [tokens]): a row with a negative position is
  // padding. It reads no experts and its output is zero
  // (launch_moe_segment).
  void enqueue(int layer, const Inkling2MoeResident& w, const uint16_t* x, int tokens, bool decode, uint16_t* out,
               cudaStream_t stream, const int64_t* positions = nullptr);
  const int32_t* route_ids() const { return ids_; }
  const float* route_weights() const { return weights_; }

  static size_t scratch_bytes(const Inkling2Config& cfg, const Inkling2LocalGeometry& geo, int max_tokens,
                              bool native);
  static GlmMoeConfig router_config(const Inkling2Config& cfg, int local_inter);

 private:
  Inkling2Config cfg_;
  GlmMoeConfig router_cfg_;
  int hidden_, inter_, experts_, top_k_, max_tokens_;
  bool native_, resident_;
  std::vector<uint8_t> bound_;      // [layers]
  float* scores_ = nullptr;         // [M, E]
  float* biased_ = nullptr;         // [M, E]
  float* weights_ = nullptr;        // [M, K]
  int32_t* ids_ = nullptr;          // [M, K]
  int32_t* rows_ = nullptr;         // [M x (K + 1)]
  int32_t* slot_row_ = nullptr;     // [M x K]
  int32_t* row_slot_ = nullptr;     // [M x K]
  int* counters_ = nullptr;         // [M], zero at rest
  MoeSegment* segs_ = nullptr;      // [E + 1]
  MoeSegment* compact_ = nullptr;   // [E]
  MoeExpertView* views_ = nullptr;  // [layers][E x 3]
  float* gate_up_ = nullptr;        // [M x K, 2 I]
  uint16_t* act_ = nullptr;         // [M x K, I]
  float* down_ = nullptr;           // [M x K, hidden]
  float* ones_ = nullptr;           // [M x K]
  // native
  uint16_t* x_inner_ = nullptr;     // [M, hidden]
  float* x_outer_ = nullptr;        // [M]
  uint16_t* act_inner_ = nullptr;   // [M x K, I]
  float* act_outer_ = nullptr;      // [M x K] by flat slot
  // native, reader rows: the NVFP4 bytes of the expert inputs
  uint8_t* x_codes_ = nullptr;      // [M, hidden / 2]
  uint8_t* x_scales_ = nullptr;     // [M, x_scale_stride_]
  uint8_t* act_codes_ = nullptr;    // [M x K, I / 2]
  uint8_t* act_scales_ = nullptr;   // [M x K, act_scale_stride_]
  float* act_gs_ = nullptr;         // [M x K] by gathered row
  int64_t x_scale_stride_ = 0, act_scale_stride_ = 0;
};

}  // namespace dgpp
