#pragma once
// Expected-tensor table for Dory1.0-4B (DoryForCausalLM, docs/dory_plan.md
// §1.7): every tensor the release must contain — names, dtypes, exact shapes —
// derived from the parsed config, not observed from one file.
//
// Naming is checkpoint truth (nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4
// @ 2026-10-17): the embedding and the head are globals
// (`backbone.embeddings.weight`, `lm_head.weight`, `lm_head.logit_scale`);
// every per-layer tensor lives under `backbone.layers.N.`, the quantized
// projections under `.mixer.` and the learned BF16 scales beside them
// (`attn_alpha` / `mlp_alpha` at the layer root, `s_gate` / `sqk` and
// `suv_gate` / `suv_up` under `.mixer.`).
//
// Format contract (modelopt W4A16_NVFP4): every [N, K] projection X is the
// triple X.weight U8 [N, K/2] (e2m1 pairs), X.weight_scale F8_E4M3 [N, K/16],
// X.weight_scale_2 F32 [] (dequant = e2m1 x e4m3 x weight_scale_2). There is
// no X.input_scale tensor — the release is weight-only (W4A16), so no
// activation-quantization path exists. The learned scales, the embedding and
// the head are BF16.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "models/dory/config.hpp"

namespace dgpp {

enum class DoryWeightClass : int {
  Embed,
  LmHead,
  LogitScale,
  AttentionProj,   // q/k/v/o/g_proj and their scales
  AttentionScale,  // s_gate, sqk
  AttentionAlpha,  // attn_alpha
  MlpProj,         // gate/up/down_proj and their scales
  MlpScale,        // suv_gate, suv_up
  MlpAlpha,        // mlp_alpha
};

enum class DoryTensorRole : uint8_t {
  Plain,
  Fp4Payload,  // U8 [N, K/2]
  Fp4Scale,    // F8_E4M3 [N, K/16]
  Fp4Global,   // F32 [] — weight_scale_2
};

struct DoryExpectedTensor {
  std::string name;
  DType dtype{};
  std::vector<int64_t> shape;
  DoryWeightClass cls = DoryWeightClass::AttentionProj;
  int layer = -1;  // layer index; -1 for the globals
  DoryTensorRole role = DoryTensorRole::Plain;

  size_t numel() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  size_t nbytes() const { return numel() * dtype_size(dtype); }
};

// The checkpoint name prefix of a physical layer ("backbone.layers.N.").
std::string dory_layer_prefix(const DoryConfig& cfg, int layer);

std::vector<DoryExpectedTensor> dory_expected_attention_tensors(const DoryConfig& cfg, int layer);
std::vector<DoryExpectedTensor> dory_expected_mlp_tensors(const DoryConfig& cfg, int layer);

// Every tensor of one physical layer, dispatched on its `layer_types` kind:
// attention (sliding or full) or MLP, never both (plan D5).
std::vector<DoryExpectedTensor> dory_expected_layer_tensors(const DoryConfig& cfg, int layer);
std::vector<DoryExpectedTensor> dory_expected_global_tensors(const DoryConfig& cfg);
std::vector<DoryExpectedTensor> dory_expected_text_tensors(const DoryConfig& cfg);

struct DoryTensorDesc {
  DType dtype{};
  std::vector<int64_t> shape;
};

struct DoryBindReport {
  size_t expected = 0;
  size_t matched = 0;
  size_t missing = 0;
  size_t dtype_mismatch = 0;
  size_t shape_mismatch = 0;
  size_t unexpected = 0;
  size_t fp4_matrices = 0;
  std::vector<std::string> errors;
  bool ok() const {
    return missing == 0 && dtype_mismatch == 0 && shape_mismatch == 0 && unexpected == 0;
  }
};

DoryBindReport dory_validate_text_binding(
    const DoryConfig& cfg,
    const std::unordered_map<std::string, DoryTensorDesc>& present,
    size_t max_errors = 32);

// The reference refuses tensor_parallel_size != 1 and the release is a single
// 3 GiB node image, so the family is one rank (docs/dory_plan.md §2). This
// throws std::invalid_argument for any world but 1, naming the key.
void dory_validate_world(int world);

}  // namespace dgpp
