#include "loaders/architecture.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace dgpp {

ModelArchitecture detect_architecture(const minijson::Value& root) {
  if (!root.is_object())
    throw std::runtime_error("config.json: root is not an object");
  std::string arch;
  if (const minijson::Value* a = root.find("architectures")) {
    if (!a->is_array() || a->items().empty() || !a->items()[0].is_string())
      throw std::runtime_error("config.json: architectures must be a non-empty string array");
    arch = std::string(a->items()[0].as_string());
  }
  std::string type;
  if (const minijson::Value* t = root.find("model_type"))
    if (t->is_string()) type = std::string(t->as_string());
  // Full GLM-5.3 (2026-09-12) is `GlmMoeDsaForCausalLM` / `glm_moe_dsa`;
  // the Flash checkpoint's class starts with `Glm5` (its model_type is
  // `glm5_next`, an older release wrote `glm_moe_dsa` there too, which is
  // why the class name is read first).
  if (arch.rfind("GlmMoeDsa", 0) == 0 || (arch.empty() && type == "glm_moe_dsa"))
    return ModelArchitecture::GlmMoeDsa;
  if (arch.rfind("Glm5", 0) == 0 || (arch.empty() && type == "glm5_next"))
    return ModelArchitecture::Glm5;
  if (arch.rfind("Qwen4Exp", 0) == 0 || (arch.empty() && type == "qwen4_exp"))
    return ModelArchitecture::Qwen4Exp;
  if (arch.rfind("Glm4Moe", 0) == 0 || (arch.empty() && type == "glm4_moe"))
    return ModelArchitecture::Glm4Moe;
  // DeepSeek-V4.1-Flash (2026-09-13, docs/deepseek_v41_flash_plan.md):
  // `DeepseekV41ForCausalLM` / `deepseek_v41` (its text_config's type is
  // `deepseek_v41_text`).
  if (arch.rfind("DeepseekV41", 0) == 0 || (arch.empty() && type == "deepseek_v41"))
    return ModelArchitecture::DeepseekV41;
  // MiMo-V2.6-Flash (2026-09-22, docs/mimo_v26_flash_plan.md):
  // `MiMoV2ForCausalLM` / `mimo_v2`.
  if (arch.rfind("MiMoV2", 0) == 0 || (arch.empty() && type == "mimo_v2"))
    return ModelArchitecture::MimoV2;
  // Qwen3.8-27B (2026-09-27): `Qwen3_5ForConditionalGeneration` / `qwen3_5`
  // (its text_config's type is `qwen3_5_text`).
  if (arch.rfind("Qwen3_5", 0) == 0 || (arch.empty() && type == "qwen3_5"))
    return ModelArchitecture::Qwen3_5;
  throw std::runtime_error(
      "config.json: unsupported architecture '" + arch + "' (model_type '" +
      type + "'); the engine implements Glm5*, Qwen4Exp*, Glm4Moe*, GlmMoeDsa*, DeepseekV41*, MiMoV2* and Qwen3_5*");
}

ModelArchitecture detect_architecture_file(const std::string& path) {
  namespace fs = std::filesystem;
  const fs::path p = fs::is_directory(path) ? fs::path(path) / "config.json"
                                            : fs::path(path);
  FILE* f = std::fopen(p.c_str(), "rb");
  if (!f)
    throw std::runtime_error("cannot open config " + p.string() + ": " +
                             std::strerror(errno));
  std::string text;
  char buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
  std::fclose(f);
  const auto parsed = minijson::parse(text);
  return detect_architecture(parsed.root);
}

}  // namespace dgpp
