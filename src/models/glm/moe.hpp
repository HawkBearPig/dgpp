#pragma once
// MoE routing and expert geometry for the GLM text model (M4 deliverables 1
// and 5), CUDA-free so config, oracle, and kernels share the types.
//
// Semantics pinned to the transformers Glm5NextTextTopkRouter /
// Glm5NextTextExperts / Glm5NextTextMLP references (DESIGN §7.4):
//
// Router (noaux_tc, sigmoid):
//   logits = fp32 GEMM of the post-LN hidden (bf16 values cast to f32)
//            against the bf16 gate rows cast to f32.
//   scores = sigmoid(logits)                          [selection uses]
//   biased = scores + e_score_correction_bias         [selection uses]
//   top-k over biased, descending; TIES -> lower expert id (engine rule;
//   torch CUDA topk ties are unspecified).
//   weights = scores gathered at the selected ids (UNCORRECTED values),
//   then norm_topk_prob: w /= (sum + 1e-20), then w *= routed_scaling_factor
//   (2.5). All fp32, division per element.
//
// Experts (swiglu with asymmetric clamps):
//   gate = clamp_max(x @ Wg^T, limit)   — NO lower clamp on the gate
//   up   = clamp(x @ Wu^T, -limit, limit)
//   act  = bf16(bf16(silu_f32(gate)) * up)             — two roundings
//   y    = act @ Wd^T                                  — bf16 out
//
// Reference accumulation (per token, ascending expert id — the reference's index_add
// over experts arrives in ascending order):
//   out = 0; for e ascending: out = bf16(out + bf16(w_e * y_e));
//   out = bf16(out + shared(x))    — shared expert added last, weight 1.
// The production TP path keeps partial down dots and the weighted chain
// in FP32, then rounds once before the FFN all-reduce; see moe_layer.hpp.
#include <cstdint>
#include <stdexcept>
#include <string>

#include "models/exl3_matrix.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

// The router's scoring rule. SigmoidBias: GLM's — sigmoid
// scores, a per-expert bias on the selection key, the picked SCORES
// normalized. SoftmaxTopk: Qwen3.8-Flash-Next's — fp32 softmax over the
// bf16 logits, top_k on the logits, the picked probabilities normalized
// and rounded to bf16; no bias (the router bias pointer may be null).
// SqrtSoftplusBias: DeepSeek-V4.1's (2026-09-13, docs/deepseek_v41_flash_plan.md
// D3) — scores sqrt(softplus(logit)) in fp32, otherwise SigmoidBias's rule
// (the bias on the selection key only, the picked scores normalized with
// + 1e-20, then routed_scaling_factor).
enum class MoeRouterMode { SigmoidBias, SoftmaxTopk, SqrtSoftplusBias };

struct GlmMoeConfig {
  int hidden = 4096;
  int inter = 2048;          // moe_intermediate_size (per expert)
  int n_experts = 288;
  int top_k = 8;
  // 1: the FP8 shared expert rides the routed chain (GLM). 0: the chain
  // is the routed experts alone — the Qwen MoE adds its BF16 shared
  // expert outside (models/qwen/moe_layer.hpp).
  int n_shared_experts = 1;
  float routed_scaling_factor = 2.5f;
  bool norm_topk_prob = true;
  float swiglu_limit = 10.0f;  // +inf: no clamps (the Qwen experts)
  // Decode launches run the shared expert's slots beside the first routed
  // ones instead of last (launch_moe_slot_order's shared_early: the order
  // alone moves, every bit stays). A family opts in after its own fabric
  // A/B (DeepSeek-V4-Flash, 2026-10-01).
  bool shared_slots_early = false;
  // Decode launches run the FP8 shared expert OUTSIDE the slot kernels
  // (2026-10-02): its gate and up as tensor-core GEMVs over all the batch's
  // rows (kernels/mma_gemv.hpp, unsplit: no reduce launches) on two side
  // streams forked at the layer's input, then the activation and the down
  // GEMV, joined before the accumulation. In the slot kernels the shared
  // expert is one slot per ROW on the CUDA cores — on a two-node rank 62 us
  // of a one-row launch and 115 us of a four-row one (moe_dup_bench at the
  // 1024-row slice), a fifth of the layer's expert time; the GEMVs read the
  // L2-resident matrices once for every row.
  // The arrangement matters: a kernel that becomes ready while the slot
  // kernels' blocks are queued is dispatched behind that whole queue (the
  // first form of this, with split-K reduce launches on one side stream,
  // gave back at its join what the slots had shed). Gate and up are
  // launched before the router, so they run at once; the activation and
  // the down GEMV become ready under the gate/up slot kernel and run ahead
  // of the down slot kernel.
  // NOT the slot chain's bits (the tensor-core chain: tolerance-equal; a
  // row's result is still its own whatever rows share the launch). A family
  // opts in after its own fabric A/B (DeepSeek-V4-Flash).
  bool shared_mma_aside = false;
  // The Mixed346 contract (2026-10-09): the layer keeps the int8 code and
  // scale copies of the routed rows the converted experts multiply
  // (kMoeInputHadamard32Int8), sized with the scratch; off for every other
  // family (no bytes).
  bool routed_int8_activations = false;
  // The EXL3 contract (2026-10-22, docs/deepseek_v41_flash_exl3_plan.md
  // §3.3): the layer keeps the rotated fp16 copies of the expert-input rows
  // (gate and up, each by its matrix's suh) and of the SwiGLU rows (by the
  // down suh) the EXL3 kernels read, sized with the scratch; off for every
  // other family (no bytes). Set with experts_exl3 bound.
  bool exl3_experts = false;
  MoeRouterMode router_mode = MoeRouterMode::SigmoidBias;

  // Weight bytes of one routed expert (payload + block scales): the number
  // the traffic model charges per selected expert.
  int64_t expert_bytes() const {
    const int64_t sc = 4 * ((inter + 127) / 128) * ((hidden + 127) / 128);
    return 3LL * inter * hidden + 3 * sc;  // gate, up: [I,H]; down: [H,I]
  }
  // The same expert in NVFP4: half a byte per element, an e4m3 scale per
  // 16, one F32 global per matrix.
  int64_t expert_bytes_fp4() const {
    const int64_t elems = 3LL * inter * hidden;
    return elems / 2 + elems / 16 + 3 * 4;
  }
  // The same expert in MXFP4: half a byte per element, an e8m0 scale per 32.
  int64_t expert_bytes_mxfp4() const {
    const int64_t elems = 3LL * inter * hidden;
    return elems / 2 + elems / 32;
  }

  static void validate_config(const GlmMoeConfig& c) {
    auto fail = [](const char* what) {
      throw std::invalid_argument(std::string("GlmMoeConfig: ") + what);
    };
    if (c.hidden <= 0 || c.inter <= 0) fail("hidden/inter must be positive");
    if (c.n_experts <= 0) fail("n_experts must be positive");
    if (c.top_k <= 0 || c.top_k > 16)
      fail("top_k must be in [1, 16] (kernel register selection)");
    if (c.top_k > c.n_experts) fail("top_k must not exceed n_experts");
    if (c.n_experts > 4096)
      fail("n_experts must be <= 4096 (router smem: 2*n_experts floats)");
    if (c.n_shared_experts != 0 && c.n_shared_experts != 1)
      fail("n_shared_experts must be 0 or 1");
    if (!(c.swiglu_limit > 0)) fail("swiglu_limit must be positive");
  }
};

// Device-pointer weight view (GlmLayerStream's GlmMoeResident wires this).
//
// TP partition (2026-09-02, expert slicing): the router is replicated and
// scores all cfg.n_experts on every rank (selection must be rank-identical);
// EVERY rank holds EVERY expert, sliced on the intermediate dimension —
// gate/up rows and down columns [rank*I/world, (rank+1)*I/world), exactly as
// the shared expert and the dense MLPs are sliced — and produces a partial
// hidden sum for the FFN all-reduce. `experts` therefore always holds
// n_experts triples; the slice width is the matrices' own rows/cols (world=1:
// the full I). Slicing instead of assigning whole experts to ranks keeps the
// per-rank bytes identical for every routing (top_k * 3 slices, always),
// which is what removes the busiest-rank wait at the FFN boundary.
struct GlmMoeWeights {
  const uint16_t* router_gate = nullptr;  // bf16 [n_experts, hidden]
  const float* router_bias = nullptr;     // f32 [n_experts] (null: SoftmaxTopk)
  GlmQuantMatrix shared[3];               // gate, up, down (FP8; unset when n_shared_experts == 0)
  // The shared expert in NVFP4 (2026-09-09, GLM-4.7, docs/glm47_plan.md
  // D3): the routed experts' shapes exactly (one expert wide), so the
  // slot and grouped launches read it as view-table entry n_experts
  // through the fp4 core. payload null = the FP8 shared triple above.
  GlmFp4Matrix shared_fp4[3];
  bool shared_nvfp4() const { return shared_fp4[0].payload != nullptr; }
  // The routed experts in exactly one of the two formats (the layer's
  // GlmExpertFormat): FP8 block-128 triples or NVFP4 triples, [n_experts *
  // 3] gate,up,down either way. The shared expert is FP8 under both for
  // GLM-5.3 (the composed hybrid); NVFP4 for GLM-4.7.
  const GlmQuantMatrix* experts = nullptr;
  const GlmFp4Matrix* experts_fp4 = nullptr;
  // The uniform NVFP4 / MXFP4 table (every routed matrix in experts_fp4).
  bool nvfp4() const { return experts_fp4 != nullptr && experts == nullptr; }
  // The per-projection form (2026-10-22, docs/flint_plan.md D2): BOTH
  // tables bound and `routed_fp4_mask` bit w (0 gate, 1 up, 2 down)
  // naming projection w NVFP4 — read from experts_fp4[e * 3 + w] — else
  // fp8 (experts[e * 3 + w]); the other table's entry stays unset. The
  // slot kernels run each phase on its format (a mixed gate/up pair on
  // the mixed kernel), the grouped chain picks per launch.
  uint8_t routed_fp4_mask = 0;
  bool mixed() const { return experts_fp4 != nullptr && experts != nullptr; }
  bool fp4_of(int which) const {
    return experts_fp4 != nullptr && (experts == nullptr || ((routed_fp4_mask >> which) & 1u));
  }
  // W4A4 activation scales (the checkpoint's max input_scale over the layer's
  // experts, gate/up and down; 0 = quantize with a dynamic per-row scale).
  float act_scale_w13 = 0.0f, act_scale_w2 = 0.0f;
  // [2] F32 on the device (gate/up, down input_scale) from the layer image;
  // the W4A4 quantizers read it there, so a cached restore keeps it.
  const float* act_scales_dev = nullptr;
  // Weight-only NVFP4 (modelopt's W4A16_NVFP4, the Qwen QAD export's draft
  // experts): no activation scale shipped and the activations stay bf16 on
  // this layer whatever the W4A4 policy — the prefill chain keeps W4A16.
  bool weight_only_fp4 = false;
  // The packed-int format (2026-09-12, full GLM-5.3, docs/glm53_plan.md
  // D2): routed experts as int4 or int8 group-64 triples, the shared
  // expert as an int8 triple of the routed experts' shapes read as
  // view-table entry n_experts through the packed core (the slot kernels'
  // one table: a packed routed table needs a packed shared expert, or
  // none). Exactly one of experts / experts_fp4 / experts_packed is set.
  GlmPackedMatrix shared_packed[3];
  bool shared_packq() const { return shared_packed[0].packed != nullptr; }
  const GlmPackedMatrix* experts_packed = nullptr;
  bool packq() const { return experts_packed != nullptr; }
  // The routed experts' input transform (2026-10-08, the NF4I8
  // checkpoint): kMoeInputHadamard32 rotates every routed projection's
  // input by a normalized 32-wide Hadamard (gate/up on the expert input,
  // down on the SwiGLU output) — the basis the experts were quantized in.
  // The residual stream, the router and the shared expert are not rotated.
  // kMoeInputHadamard32Int8 (2026-10-09, the Mixed346 checkpoint): the
  // same rotation, then the row quantized to int8 codes with one fp32
  // scale per 128 values (kernels/hadamard32.cu), the activation form the
  // Mixed346 (format 3) triples multiply; an expert of such a layer whose
  // triple is the baseline's (format 0, an "existing" expert) reads the
  // plain bf16 rows like the shared expert.
  int routed_input_transform = 0;
  bool routed_hadamard32() const { return routed_input_transform == 1 || routed_input_transform == 2; }
  bool routed_activation_int8() const { return routed_input_transform == 2; }
  // The EXL3 trellis format (2026-10-22, docs/deepseek_v41_flash_exl3_plan.md
  // §3.3): routed triples [n_experts * 3] as Exl3Matrix (gate, up, down;
  // one width per layer) and the shared expert as an EXL3 triple read as
  // view-table entry n_experts (kernels/exl3_gemv.hpp). Exactly one of
  // experts / experts_fp4 / experts_packed / experts_exl3 is set; an EXL3
  // routed table needs an EXL3 shared expert, or none.
  const Exl3Matrix* experts_exl3 = nullptr;
  Exl3Matrix shared_exl3[3];
  bool exl3() const { return experts_exl3 != nullptr; }
  bool shared_exl3_set() const { return shared_exl3[0].tiles != nullptr; }
};
constexpr int kMoeInputPlain = 0;
constexpr int kMoeInputHadamard32 = 1;
constexpr int kMoeInputHadamard32Int8 = 2;

// The decode path's device-side expert table entry: the
// slot kernels read the ROUTE from device memory, so the weight views
// they indirect through must live there too. Dims stay kernel args (all
// routed experts share them; only the shared expert's inter differs).
// Layout-compatible with nothing — one type, one producer (the layer's
// per-binding upload), the slot kernels its consumers.
struct MoeExpertView {
  const uint8_t* payload = nullptr;  // fp8: e4m3 [n, k]; nvfp4: e2m1 pairs [n, k/2]
  const float* scales = nullptr;     // fp8: F32 block scales (nvfp4: null)
  const uint8_t* fp4_scales = nullptr;  // nvfp4: e4m3 [n, k/16] (fp8: null)
  const float* fp4_global = nullptr;    // nvfp4: the matrix's F32 global scale
  // The fp8 scale grid as log2 block sizes (7 = 128 on both axes, the
  // checkpoint's grid; a TP slice re-blocked at gcd(128, slice) on its
  // sliced axis — docs/qwen38_flash_next_plan.md D2). The GEMV core reads
  // them; the tensor-core kernels take only 128 (the layer refuses).
  int scale_shift_rows = 7;
  int scale_shift_cols = 7;
  static int shift_of(int block) {
    int s = 0;
    while ((1 << s) < block) ++s;
    return s;
  }
  static MoeExpertView of(const GlmQuantMatrix& m) {
    return MoeExpertView{m.payload, m.scales, nullptr, nullptr,
                         shift_of(m.scale_block_rows), shift_of(m.scale_block_cols)};
  }
  static MoeExpertView of(const GlmFp4Matrix& m) {
    MoeExpertView v{m.payload, nullptr, m.scales, m.global_scale, 7, 7};
    v.fp4_group = m.scale_group;
    return v;
  }
  // The fp4 scale group (2026-09-13): 16 = NVFP4 (e4m3 scales + the global),
  // 32 = MXFP4 (e8m0 scales, no global — fp4_global stays null).
  int fp4_group = 16;
  // The packed-int interpretation: payload = the I32 words [n, k*bits/32]
  // as bytes, packed_scales = the group scales (bf16 per 64 or f16 per 128
  // by packed_scale_fmt, quant_matrix.hpp), bits = 4 or 8 (0: not packed).
  const uint16_t* packed_scales = nullptr;
  int bits = 0;
  int packed_scale_fmt = 0;
  static MoeExpertView of(const GlmPackedMatrix& m) {
    MoeExpertView v;
    v.payload = reinterpret_cast<const uint8_t*>(m.packed);
    v.packed_scales = m.scales;
    v.bits = m.bits;
    v.packed_scale_fmt = m.scale_fmt;
    return v;
  }
  // The EXL3 interpretation (2026-10-22): payload = the n-tile-major tiles,
  // the suh / svh vectors, the width (0: not EXL3) and the matrix's own N
  // and K (the shared expert's differ from the routed slice's).
  const uint16_t* exl3_suh = nullptr;
  const uint16_t* exl3_svh = nullptr;
  int exl3_bits = 0;
  int exl3_rows = 0;
  int exl3_cols = 0;
  static MoeExpertView of(const Exl3Matrix& m) {
    MoeExpertView v;
    v.payload = m.tiles;
    v.exl3_suh = m.suh;
    v.exl3_svh = m.svh;
    v.exl3_bits = m.bits;
    v.exl3_rows = static_cast<int>(m.rows);
    v.exl3_cols = static_cast<int>(m.cols);
    return v;
  }
};

}  // namespace dgpp
