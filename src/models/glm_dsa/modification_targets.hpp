#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "models/glm_dsa/config.hpp"

namespace dgpp {

// The main-stack projections Heretic discovers as individual linear modules.
// Fused routed experts, embeddings, head and draft layer are not targets.
struct GlmDsaModificationTarget {
  std::string id;         // layer/component/module_index, also the capture key
  std::string module;     // canonical checkpoint module name, without .weight
  std::string component;  // Heretic's logical component, not the checkpoint path
  int layer = 0;
  int module_index = 0;
  int64_t input_features = 0;
  int64_t output_features = 0;
  int64_t input_begin = 0;
  int64_t input_end = 0;
  int weight_bits = 0;  // 0 = BF16, otherwise the checkpoint's packed integer width
  int group_size = 0;
};

// Host-only metadata. Does not load tensors, create CUDA contexts, or claim
// that capture/adapter execution has been implemented for these targets.
std::vector<GlmDsaModificationTarget> glm_dsa_modification_targets(const GlmDsaTextConfig& config,
                                                                   int rank, int world);

}  // namespace dgpp
