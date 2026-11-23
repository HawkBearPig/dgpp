#pragma once
// Expected-tensor tables for Inkling 2 (Inkling2LoopForConditionalGeneration,
// docs/inkling2_mini_plan.md section 1.9): every tensor the checkpoint must
// contain, with names, dtypes and exact shapes derived from the parsed
// config. The tables drive the offline validator, the resident loader and
// the test fixture.
//
// Naming is checkpoint truth (thinkingmachines/Inkling-2-Mini): the text
// model under `model.llm.`, the vision tower under `model.visual.`, and the
// draft model in mtp.safetensors without a prefix.
//
// Format contract (the release as shipped):
//   BF16    projections, norms, convolution taps, the router weight, the
//           loop controllers, the embedding, the unembedding, the vision
//           tower, the draft model
//   F32     the attention-residual queries, the message queries, the router
//           bias, the NVFP4 experts' amax and denominator
//   NVFP4   X_weight U8 [E, N, K/2] (E2M1 pairs, the even element in the low
//           nibble) + X_weight_scale F8_E4M3 [E, N, K/16] + X_weight_amax
//           F32 [E] + X_weight_outer_scale_denominator F32 [1]
//           a weight is e2m1 * scale * amax[e] / denominator
//   MXFP8   X_weight F8_E4M3 [E, N, K] + X_weight_scale U8 [E, N, K/32]
//           a weight is e4m3 * 2^(scale - 127)
// The gate/up matrix `w13` stores its rows interleaved: row 2i is gate i,
// row 2i + 1 is up i.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "models/inkling2/config.hpp"

namespace dgpp {

enum class Inkling2WeightClass : int {
  Embed,
  Unembed,
  GlobalNorm,     // embed_norm, norm
  MixQuery,       // attn_res.*, the two cross-pass query tensors
  LayerNorm,      // attn_norm or {q,k,v}_input_norm, mlp_norm
  Attention,      // wq_du, wr_du, gate_attn, attn_sink_proj, wk_dv, wv_dv, wo_ud
  AttentionAux,   // q / rel clip scales, the relative table, the K / V convolution taps
  HiddenConv,     // attn_sconv, mlp_sconv
  Router,         // mlp.gate.{weight,bias}
  RoutedExpert,   // w13 / w2 in NVFP4 or MXFP8
  Controller,     // controllers.{attn,k,v,mlp}.{down,gain,forget}_weight
  Vision,         // model.visual.*
  Draft,          // mtp.safetensors
};

enum class Inkling2TensorRole : uint8_t {
  Plain,
  Fp4Payload,      // U8 [E, N, K/2]
  Fp4Scale,        // F8_E4M3 [E, N, K/16]
  Fp4Amax,         // F32 [E]
  Fp4Denominator,  // F32 [1]
  Mx8Payload,      // F8_E4M3 [E, N, K]
  Mx8Scale,        // U8 [E, N, K/32]
};

struct Inkling2ExpectedTensor {
  std::string name;
  DType dtype{};
  std::vector<int64_t> shape;
  Inkling2WeightClass cls = Inkling2WeightClass::Attention;
  int layer = -1;        // layer index, -1 for globals
  int projection = -1;   // routed experts: 0 = w13, 1 = w2
  Inkling2TensorRole role = Inkling2TensorRole::Plain;

  size_t numel() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  size_t nbytes() const { return numel() * dtype_size(dtype); }
};

std::string inkling2_layer_prefix(int layer);

// The text model: globals, then every layer. A layer's table includes its
// mix queries (`attn_res.attn_queries.<layer - 1>`, `attn_res.mlp_queries.<layer>`).
std::vector<Inkling2ExpectedTensor> inkling2_expected_text_tensors(const Inkling2Config& cfg);
std::vector<Inkling2ExpectedTensor> inkling2_expected_layer_tensors(const Inkling2Config& cfg, int layer);
std::vector<Inkling2ExpectedTensor> inkling2_expected_global_tensors(const Inkling2Config& cfg);
// The vision tower (empty without a vision_config).
std::vector<Inkling2ExpectedTensor> inkling2_expected_vision_tensors(const Inkling2Config& cfg);
// mtp.safetensors (empty without an mtp_config).
std::vector<Inkling2ExpectedTensor> inkling2_expected_draft_tensors(const Inkling2Config& cfg);

struct Inkling2TensorDesc {
  DType dtype{};
  std::vector<int64_t> shape;
};

struct Inkling2BindReport {
  size_t expected = 0;
  size_t matched = 0;
  size_t missing = 0;
  size_t dtype_mismatch = 0;
  size_t shape_mismatch = 0;
  size_t unexpected = 0;
  size_t fp4_tensors = 0;
  size_t mxfp8_tensors = 0;
  std::vector<std::string> errors;
  bool ok() const { return missing == 0 && dtype_mismatch == 0 && shape_mismatch == 0 && unexpected == 0; }
};

// Checks `present` against `expected`. Every present tensor must be expected.
Inkling2BindReport inkling2_validate_binding(const std::vector<Inkling2ExpectedTensor>& expected,
                                             const std::unordered_map<std::string, Inkling2TensorDesc>& present,
                                             size_t max_errors = 32);

// Tensor-parallel geometry acceptance (docs/inkling2_mini_plan.md section
// 2): throws std::invalid_argument naming the dimension that does not
// divide. A rank holds whole key/value heads with their query heads and
// the same slice of every expert's intermediate width, on whole 32-value
// scale blocks.
void inkling2_tp_validate_geometry(const Inkling2Config& cfg, int rank, int world);

}  // namespace dgpp
