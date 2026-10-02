#pragma once
// Which model family a checkpoint's config.json describes (Q1/Q2,
// 2026-09-09): the apps dispatch on this before parsing a family's config.
// Read from `architectures[0]` (the transformers class name), with
// `model_type` as the cross-check; anything else is refused by name — the
// engine serves the families it implements, it does not guess.
#include <string>

#include "loaders/minijson.hpp"

namespace dgpp {

enum class ModelArchitecture : int {
  Glm5,     // Glm5ForConditionalGeneration / glm5_next (GLM-5.3-Flash)
  Qwen4Exp, // Qwen4ExpForConditionalGeneration / qwen4_exp (Qwen3.8-Flash-Next)
  Glm4Moe,  // Glm4MoeForCausalLM / glm4_moe (GLM-4.7, 2026-09-09)
  GlmMoeDsa, // GlmMoeDsaForCausalLM / glm_moe_dsa (full GLM-5.3, 2026-09-12)
  DeepseekV41, // DeepseekV41ForCausalLM / deepseek_v41 (DeepSeek-V4.1-Flash, 2026-09-13)
  MimoV2,      // MiMoV2ForCausalLM / mimo_v2 (MiMo-V2.6-Flash, 2026-09-22)
  Qwen3_5,     // Qwen3_5ForConditionalGeneration / qwen3_5 (Qwen3.8-27B, text-only support lands first)
};

constexpr const char* model_architecture_name(ModelArchitecture a) {
  switch (a) {
    case ModelArchitecture::Qwen4Exp: return "qwen4_exp";
    case ModelArchitecture::Glm4Moe: return "glm4_moe";
    case ModelArchitecture::GlmMoeDsa: return "glm_moe_dsa";
    case ModelArchitecture::Glm5: return "glm5";
    case ModelArchitecture::DeepseekV41: return "deepseek_v41";
    case ModelArchitecture::MimoV2: return "mimo_v2";
    case ModelArchitecture::Qwen3_5: return "qwen3_5";
  }
  return "glm5";
}

// From a parsed config.json root; throws std::runtime_error naming the
// unsupported value.
ModelArchitecture detect_architecture(const minijson::Value& root);
// Reads DIR/config.json (or the file itself when `path` names a file).
ModelArchitecture detect_architecture_file(const std::string& path);

}  // namespace dgpp
