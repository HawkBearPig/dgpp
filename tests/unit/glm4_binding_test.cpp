// The GLM-4.7 expected-tensor table: its shape on the
// release's config, the TP geometry acceptance at the deployment worlds,
// and — when the checkpoint is in the hub cache — the full binding
// against every shard's header (the 44 shards and mtp.safetensors: every
// expected tensor present with its dtype and shape, nothing unexpected).
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common/test.hpp"
#include "loaders/safetensors.hpp"
#include "models/glm4/binding.hpp"
#include "models/glm4/config.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

std::filesystem::path landed_snapshot() {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) return {};
  const fs::path root = fs::path(home) / ".cache/huggingface/hub/models--nvidia--GLM-4.7-NVFP4/snapshots";
  if (!fs::is_directory(root)) return {};
  for (const auto& snap : fs::directory_iterator(root))
    if (fs::exists(snap.path() / "config.json") && fs::exists(snap.path() / "model.safetensors.index.json"))
      return snap.path();
  return {};
}

}  // namespace

DGPP_TEST(glm4_binding_table_has_the_release_shape) {
  const auto snap = landed_snapshot();
  if (snap.empty())
    dgpp::test::skip("required checkpoint snapshot unavailable: nvidia/GLM-4.7-NVFP4");
  const dgpp::Glm4TextConfig cfg = dgpp::Glm4TextConfig::from_json_file((snap / "config.json").string());
  // A dense layer: 2 norms + 4 weights + 3 biases + 2 norms + 2 kv scales + 3 x 4 fp4 = 25.
  const auto dense = dgpp::glm4_expected_layer_tensors(cfg, 0);
  require(dense.size() == 25, "dense layer " + std::to_string(dense.size()));
  // A MoE layer: 13 (norms, attention, kv scales) + router 2 + 161 experts x 3 x 4.
  const auto moe = dgpp::glm4_expected_layer_tensors(cfg, 3);
  require(moe.size() == 13 + 2 + 161 * 12, "moe layer " + std::to_string(moe.size()));
  // The draft: 6 head tensors + 11 (norms, attention without kv scales) + router 2 + 161 x 3.
  const auto draft = dgpp::glm4_expected_layer_tensors(cfg, cfg.mtp_layer());
  require(draft.size() == 6 + 11 + 2 + 161 * 3, "draft layer " + std::to_string(draft.size()));
  require(draft[0].name == "model.layers.92.enorm.weight", "draft prefix");
  const auto all = dgpp::glm4_expected_text_tensors(cfg);
  require(all.size() == 3 + 3 * 25 + 89 * (15 + 161 * 12) + draft.size(), "table size " + std::to_string(all.size()));
  size_t unused = 0;
  for (const auto& e : all) unused += e.unused();
  require(unused == 92 * 2 + 3 * 3 * 1 + 89 * 161 * 3 + 2, "unused count " + std::to_string(unused));
}

DGPP_TEST(glm4_tp_geometry_accepts_the_deployment_worlds) {
  const auto snap = landed_snapshot();
  if (snap.empty())
    dgpp::test::skip("required checkpoint snapshot unavailable: nvidia/GLM-4.7-NVFP4");
  const dgpp::Glm4TextConfig cfg = dgpp::Glm4TextConfig::from_json_file((snap / "config.json").string());
  for (const int w : {1, 2, 4, 8})
    for (int r = 0; r < w; ++r) dgpp::glm4_tp_validate_geometry(cfg, r, w);
  bool refused = false;
  try {
    dgpp::glm4_tp_validate_geometry(cfg, 0, 96);  // 1536 / 96 = 16: below the core's K
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "a 16-wide slice is refused");
}

DGPP_TEST(glm4_binding_matches_the_landed_checkpoint) {
  const auto snap = landed_snapshot();
  if (snap.empty())
    dgpp::test::skip("required checkpoint snapshot unavailable: nvidia/GLM-4.7-NVFP4");
  namespace fs = std::filesystem;
  const dgpp::Glm4TextConfig cfg = dgpp::Glm4TextConfig::from_json_file((snap / "config.json").string());
  std::unordered_map<std::string, dgpp::Glm4TensorDesc> present;
  for (const auto& entry : fs::directory_iterator(snap)) {
    if (entry.path().extension() != ".safetensors") continue;
    auto f = dgpp::SafetensorsFile::open(entry.path().string());
    f->for_each([&](const dgpp::TensorInfo& t) {
      present.emplace(t.name, dgpp::Glm4TensorDesc{t.dtype, t.shape});
    });
  }
  const dgpp::Glm4BindReport rep = dgpp::glm4_validate_text_binding(cfg, present);
  std::string errs;
  for (const auto& e : rep.errors) errs += "\n  " + e;
  require(rep.ok(), "binding: missing " + std::to_string(rep.missing) + ", dtype " +
                        std::to_string(rep.dtype_mismatch) + ", shape " + std::to_string(rep.shape_mismatch) +
                        ", unexpected " + std::to_string(rep.unexpected) + errs);
  require(rep.fp4_matrices == 3 * 3 + 89 * 161 * 3, "fp4 matrices " + std::to_string(rep.fp4_matrices));
  require(rep.bf16_expert_matrices == 161 * 3, "draft experts");
}
