// The Flint expected-tensor table: its shape on the release's config, the
// per-projection expert formats, the validator's report, the TP geometry
// acceptance at the deployment worlds, and — when the checkpoint is in the
// hub cache — the full binding against every shard's header.
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common/test.hpp"
#include "flint_config_json.hpp"
#include "loaders/minijson.hpp"
#include "loaders/safetensors.hpp"
#include "models/flint/binding.hpp"
#include "models/flint/config.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

dgpp::FlintTextConfig release_config() {
  const auto t = dgpp::minijson::parse(flint_test::config_json());
  return dgpp::FlintTextConfig::parse(t.root);
}

const dgpp::FlintExpectedTensor& find(const std::vector<dgpp::FlintExpectedTensor>& v, const char* name) {
  for (const auto& e : v)
    if (e.name == name) return e;
  throw std::runtime_error(std::string("expected tensor missing from the table: ") + name);
}
bool has(const std::vector<dgpp::FlintExpectedTensor>& v, const char* name) {
  for (const auto& e : v)
    if (e.name == name) return true;
  return false;
}

std::filesystem::path landed_snapshot() {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) return {};
  const fs::path root = fs::path(home) / ".cache/huggingface/hub/models--HawkBearPig--flint-FP8-NVFP4/snapshots";
  if (!fs::is_directory(root)) return {};
  for (const auto& snap : fs::directory_iterator(root))
    if (fs::exists(snap.path() / "config.json") && fs::exists(snap.path() / "model.safetensors.index.json"))
      return snap.path();
  return {};
}

}  // namespace

DGPP_TEST(flint_binding_table_has_the_release_shape) {
  const dgpp::FlintTextConfig cfg = release_config();
  // A dense layer: 2 norms + 4 projections + 2 head norms + the sink + 3 dense matrices = 12.
  const auto dense = dgpp::flint_expected_layer_tensors(cfg, 0);
  require(dense.size() == 12, "dense layer " + std::to_string(dense.size()));
  // An fp8 MoE layer: 2 norms + 7 attention + router + 3 shared + 256 x 3 x 2.
  const auto fp8 = dgpp::flint_expected_layer_tensors(cfg, 2);
  require(fp8.size() == 2 + 7 + 1 + 3 + 256 * 6, "fp8 moe layer " + std::to_string(fp8.size()));
  // An NVFP4 MoE layer: ... + 256 x 3 x 3.
  const auto fp4 = dgpp::flint_expected_layer_tensors(cfg, 31);
  require(fp4.size() == 2 + 7 + 1 + 3 + 256 * 9, "nvfp4 moe layer " + std::to_string(fp4.size()));
  // The mixed layers: 12 (up alone nvfp4), 37 (down alone fp8).
  const auto l12 = dgpp::flint_expected_layer_tensors(cfg, 12);
  require(l12.size() == 2 + 7 + 1 + 3 + 256 * (2 + 3 + 2), "layer 12 " + std::to_string(l12.size()));
  require(has(l12, "model.layers.12.mlp.experts.5.up_proj.weight_packed") &&
              has(l12, "model.layers.12.mlp.experts.5.gate_proj.weight_scale_inv") &&
              has(l12, "model.layers.12.mlp.experts.5.down_proj.weight_scale_inv") &&
              !has(l12, "model.layers.12.mlp.experts.5.gate_proj.weight_packed"),
          "layer 12's formats");
  const auto l37 = dgpp::flint_expected_layer_tensors(cfg, 37);
  require(l37.size() == 2 + 7 + 1 + 3 + 256 * (3 + 3 + 2), "layer 37 " + std::to_string(l37.size()));
  require(has(l37, "model.layers.37.mlp.experts.0.gate_proj.weight_global_scale") &&
              has(l37, "model.layers.37.mlp.experts.0.down_proj.weight_scale_inv"),
          "layer 37's formats");
  const auto all = dgpp::flint_expected_text_tensors(cfg);
  require(all.size() == 85886, "table size " + std::to_string(all.size()));  // the index's count
  // Shapes.
  require(find(fp8, "model.layers.2.self_attn.q_proj.weight").shape == std::vector<int64_t>{4096, 2048}, "q_proj");
  require(find(fp8, "model.layers.2.self_attn.k_proj.weight").shape == std::vector<int64_t>{512, 2048}, "k_proj");
  require(find(fp8, "model.layers.2.self_attn.o_proj.weight").shape == std::vector<int64_t>{2048, 4096}, "o_proj");
  require(find(fp8, "model.layers.2.self_attn.q_norm.weight").shape == std::vector<int64_t>{128}, "q_norm");
  require(find(fp8, "model.layers.2.self_attn.softmax_offset").shape == std::vector<int64_t>{32}, "sink");
  require(find(fp8, "model.layers.2.mlp.gate.weight").shape == std::vector<int64_t>{256, 2048}, "router");
  require(find(fp8, "model.layers.2.mlp.shared_expert.down_proj.weight").shape == std::vector<int64_t>{2048, 1024}, "shared down");
  const dgpp::FlintExpectedTensor g = find(fp8, "model.layers.2.mlp.experts.7.gate_proj.weight");
  require(g.dtype == dgpp::DType::F8_E4M3 && g.shape == std::vector<int64_t>{512, 2048} && g.expert == 7, "fp8 gate payload");
  require(find(fp8, "model.layers.2.mlp.experts.7.gate_proj.weight_scale_inv").shape == std::vector<int64_t>{4, 16}, "fp8 gate scales");
  require(find(fp8, "model.layers.2.mlp.experts.7.down_proj.weight").shape == std::vector<int64_t>{2048, 512}, "fp8 down payload");
  require(find(fp8, "model.layers.2.mlp.experts.7.down_proj.weight_scale_inv").shape == std::vector<int64_t>{16, 4}, "fp8 down scales");
  const dgpp::FlintExpectedTensor p = find(fp4, "model.layers.31.mlp.experts.9.gate_proj.weight_packed");
  require(p.dtype == dgpp::DType::U8 && p.shape == std::vector<int64_t>{512, 1024}, "nvfp4 gate payload");
  const dgpp::FlintExpectedTensor s = find(fp4, "model.layers.31.mlp.experts.9.gate_proj.weight_scale");
  require(s.dtype == dgpp::DType::F8_E4M3 && s.shape == std::vector<int64_t>{512, 128}, "nvfp4 gate scales");
  require(find(fp4, "model.layers.31.mlp.experts.9.gate_proj.weight_global_scale").shape == std::vector<int64_t>{1}, "nvfp4 global");
  require(find(fp4, "model.layers.31.mlp.experts.9.down_proj.weight_packed").shape == std::vector<int64_t>{2048, 256}, "nvfp4 down payload");
  require(find(fp4, "model.layers.31.mlp.experts.9.down_proj.weight_scale").shape == std::vector<int64_t>{2048, 32}, "nvfp4 down scales");
  require(find(dense, "model.layers.0.mlp.gate_proj.weight").shape == std::vector<int64_t>{6144, 2048}, "dense gate");
  require(find(dense, "model.layers.0.mlp.down_proj.weight").shape == std::vector<int64_t>{2048, 6144}, "dense down");
  size_t unused = 0;
  for (const auto& e : all) unused += e.unused();
  require(unused == 0, "nothing in the table is unused");
}

DGPP_TEST(flint_binding_validator_reports_by_class) {
  const dgpp::FlintTextConfig cfg = release_config();
  const auto all = dgpp::flint_expected_text_tensors(cfg);
  std::unordered_map<std::string, dgpp::FlintTensorDesc> present;
  for (const auto& e : all) present.emplace(e.name, dgpp::FlintTensorDesc{e.dtype, e.shape});
  {
    const dgpp::FlintBindReport rep = dgpp::flint_validate_text_binding(cfg, present);
    require(rep.ok() && rep.matched == all.size(), "a complete checkpoint binds");
    require(rep.fp8_matrices == 23040 && rep.fp4_matrices == 13056, "matrix counts (the model card's)");
  }
  present.erase("model.layers.7.mlp.experts.3.up_proj.weight_scale_inv");
  present["model.layers.2.self_attn.o_proj.weight"].dtype = dgpp::DType::F8_E4M3;
  present["model.layers.31.mlp.experts.0.gate_proj.weight_scale"].shape = {512, 64};
  present.emplace("model.layers.0.self_attn.qkv_proj.weight", dgpp::FlintTensorDesc{dgpp::DType::BF16, {5120, 2048}});
  const dgpp::FlintBindReport rep = dgpp::flint_validate_text_binding(cfg, present);
  require(!rep.ok(), "a broken checkpoint is refused");
  require(rep.missing == 1 && rep.dtype_mismatch == 1 && rep.shape_mismatch == 1 && rep.unexpected == 1,
          "the report counts by class");
  require(rep.errors.size() == 4, "four errors");
}

DGPP_TEST(flint_tp_geometry_accepts_the_deployment_worlds) {
  const dgpp::FlintTextConfig cfg = release_config();
  for (const int w : {1, 2, 4})
    for (int r = 0; r < w; ++r) dgpp::flint_tp_validate_geometry(cfg, r, w);
  for (const int w : {3, 8}) {
    bool refused = false;
    try {
      dgpp::flint_tp_validate_geometry(cfg, 0, w);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    require(refused, "world " + std::to_string(w) + " is refused");
  }
}

DGPP_TEST(flint_binding_matches_the_landed_checkpoint) {
  const auto snap = landed_snapshot();
  if (snap.empty()) dgpp::test::skip("required checkpoint snapshot unavailable: HawkBearPig/flint-FP8-NVFP4");
  namespace fs = std::filesystem;
  {
    std::ifstream idx(snap / "model.safetensors.index.json");
    std::string text((std::istreambuf_iterator<char>(idx)), std::istreambuf_iterator<char>());
    const auto parsed = dgpp::minijson::parse(text);
    for (const auto& m : parsed.root.at("weight_map").members()) {
      const fs::path shard = snap / std::string(m.value.as_string());
      if (!fs::exists(shard)) dgpp::test::skip("checkpoint download incomplete; missing shard: " + shard.string());
    }
  }
  const dgpp::FlintTextConfig cfg = dgpp::FlintTextConfig::from_json_file((snap / "config.json").string());
  std::unordered_map<std::string, dgpp::FlintTensorDesc> present;
  for (const auto& entry : fs::directory_iterator(snap)) {
    if (entry.path().extension() != ".safetensors") continue;
    auto f = dgpp::SafetensorsFile::open(entry.path().string());
    f->for_each([&](const dgpp::TensorInfo& t) { present.emplace(t.name, dgpp::FlintTensorDesc{t.dtype, t.shape}); });
  }
  const dgpp::FlintBindReport rep = dgpp::flint_validate_text_binding(cfg, present);
  std::string errs;
  for (const auto& e : rep.errors) errs += "\n  " + e;
  require(rep.ok(), "binding: missing " + std::to_string(rep.missing) + ", dtype " + std::to_string(rep.dtype_mismatch) +
                        ", shape " + std::to_string(rep.shape_mismatch) + ", unexpected " + std::to_string(rep.unexpected) + errs);
  require(rep.fp8_matrices == 23040 && rep.fp4_matrices == 13056, "matrix counts");
}
