#pragma once
// Expected tensors in the published MiniMax M3.1 text checkpoint.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "models/minimax/config.hpp"

namespace dgpp {

enum class MinimaxWeightClass : int {
  Embed,
  LmHead,
  FinalNorm,
  LayerNorm,     // input_layernorm, post_attention_layernorm
  Attention,     // main and index projections and head norms
  Router,        // FP32 router and selection correction bias
  SharedExpert,  // block_sparse_moe.shared_experts.{gate,up,down}_proj
  DenseMlp,      // the dense layers' mlp.{gate,up,down}_proj
  RoutedExpert,  // NVFP4 payload, group scales, weight and activation globals
};

enum class MinimaxTensorRole : uint8_t {
  Plain,
  Fp8Payload,      // F8_E4M3 [N, K]
  Fp8Scale,        // U8 [N, K/32], E8M0 exponent bytes
  Fp4Payload,      // U8 [N, K/2]
  Fp4Scale,        // F8_E4M3 [N, K/16]
  Fp4Global,       // F32 scalar encode scale
  Fp4InputGlobal,  // F32 scalar activation encode scale
};

struct MinimaxExpectedTensor {
  std::string name;
  DType dtype{};
  std::vector<int64_t> shape;
  MinimaxWeightClass cls = MinimaxWeightClass::Attention;
  int layer = -1;   // layer index; -1 for globals
  int expert = -1;  // routed-expert id, -1 otherwise
  MinimaxTensorRole role = MinimaxTensorRole::Plain;

  size_t numel() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  size_t nbytes() const { return numel() * dtype_size(dtype); }
  bool unused() const { return false; }
};

std::string minimax_layer_prefix(int layer);

std::vector<MinimaxExpectedTensor> minimax_expected_text_tensors(const MinimaxConfig& cfg);
std::vector<MinimaxExpectedTensor> minimax_expected_layer_tensors(const MinimaxConfig& cfg,
                                                                  int layer);
std::vector<MinimaxExpectedTensor> minimax_expected_global_tensors(const MinimaxConfig& cfg);

struct MinimaxTensorDesc {
  DType dtype{};
  std::vector<int64_t> shape;
};

struct MinimaxBindReport {
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

MinimaxBindReport minimax_validate_text_binding(
    const MinimaxConfig& cfg, const std::unordered_map<std::string, MinimaxTensorDesc>& present,
    size_t max_errors = 32);

// Whole main and index heads and complete 32-wide MXFP8 / 16-wide NVFP4
// scale groups belong to each tensor-parallel rank.
void minimax_tp_validate_geometry(const MinimaxConfig& cfg, int rank, int world);

}  // namespace dgpp
