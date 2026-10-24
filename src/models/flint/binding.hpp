#pragma once
// Expected-tensor table for Flint (FlintForCausalLM, 2026-10-22,
// docs/flint_plan.md §1.5): every tensor the checkpoint must contain —
// names, dtypes, exact shapes — derived from the parsed config, not
// observed from one file. The table drives the offline validator and the
// resident loader, as the MiMo and GLM-4.7 tables do.
//
// Naming is checkpoint truth (HawkBearPig/flint-FP8-NVFP4): layers under
// `model.layers.L.`, the globals `model.embed_tokens.weight`,
// `model.norm.weight`, `lm_head.weight`. Nothing in the release is
// unserved.
//
// Format contract (the export as shipped):
//   BF16        every non-expert tensor: q/k/v/o projections, q_norm /
//               k_norm, softmax_offset, the layer norms, the router, the
//               shared expert, the dense MLPs, the embedding, the head;
//   fp8         X.weight F8_E4M3 [N, K] + X.weight_scale_inv F32
//               [ceil(N/128), K/128] — the routed projections
//               precision_by_layer names fp8;
//   NVFP4       X.weight_packed U8 [N, K/2] + X.weight_scale F8_E4M3
//               [N, K/16] + X.weight_global_scale F32 [1] — the ones it
//               names nvfp4.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "models/flint/config.hpp"

namespace dgpp {

enum class FlintWeightClass : int {
  Embed,
  LmHead,
  FinalNorm,
  LayerNorm,      // input_layernorm, post_attention_layernorm
  Attention,      // q/k/v/o_proj, q_norm, k_norm, softmax_offset
  Router,         // mlp.gate.weight
  SharedExpert,   // mlp.shared_expert.{gate,up,down}_proj
  DenseMlp,       // the dense layers' mlp.{gate,up,down}_proj
  RoutedExpert,   // fp8 pairs / NVFP4 triples
};

enum class FlintTensorRole : uint8_t {
  Plain,
  Fp8Payload,   // F8_E4M3 [N, K]
  Fp8Scale,     // F32 [ceil(N/128), K/128]
  Fp4Payload,   // U8 [N, K/2]
  Fp4Scale,     // F8_E4M3 [N, K/16]
  Fp4Global,    // F32 [1]
};

struct FlintExpectedTensor {
  std::string name;
  DType dtype{};
  std::vector<int64_t> shape;
  FlintWeightClass cls = FlintWeightClass::Attention;
  int layer = -1;   // layer index; -1 for globals
  int expert = -1;  // routed-expert id, -1 otherwise
  FlintTensorRole role = FlintTensorRole::Plain;

  size_t numel() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  size_t nbytes() const { return numel() * dtype_size(dtype); }
  bool unused() const { return false; }
};

std::string flint_layer_prefix(int layer);

std::vector<FlintExpectedTensor> flint_expected_text_tensors(const FlintTextConfig& cfg);
std::vector<FlintExpectedTensor> flint_expected_layer_tensors(const FlintTextConfig& cfg, int layer);
std::vector<FlintExpectedTensor> flint_expected_global_tensors(const FlintTextConfig& cfg);

struct FlintTensorDesc {
  DType dtype{};
  std::vector<int64_t> shape;
};

struct FlintBindReport {
  size_t expected = 0;
  size_t matched = 0;
  size_t missing = 0;
  size_t dtype_mismatch = 0;
  size_t shape_mismatch = 0;
  size_t unexpected = 0;
  size_t fp8_matrices = 0;
  size_t fp4_matrices = 0;
  std::vector<std::string> errors;
  bool ok() const {
    return missing == 0 && dtype_mismatch == 0 && shape_mismatch == 0 && unexpected == 0;
  }
};

FlintBindReport flint_validate_text_binding(
    const FlintTextConfig& cfg,
    const std::unordered_map<std::string, FlintTensorDesc>& present,
    size_t max_errors = 32);

// Tensor-parallel geometry acceptance (docs/flint_plan.md §2): throws
// std::invalid_argument naming the dim that does not divide. world must
// divide the kv head count (a rank's query heads sit on its own kv heads),
// the routed expert slice must be a multiple of 128 (the fp8 scale block;
// every NVFP4 16-block with it), the dense and shared slices multiples of 8.
void flint_tp_validate_geometry(const FlintTextConfig& cfg, int rank, int world);

}  // namespace dgpp
