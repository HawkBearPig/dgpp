// The MiMo-V2.6-Flash expected-tensor table: its shape on the release's
// config, the ignore rule for the checkpoint's unserved classes, the fused
// projection's per-chunk scale rows, the validator's report, the TP
// geometry acceptance at the deployment worlds, and — when the checkpoint
// is in the hub cache — the full binding against every shard's header.
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "loaders/safetensors.hpp"
#include "mimo_config_json.hpp"
#include "models/mimo/binding.hpp"
#include "models/mimo/config.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

dgpp::MimoTextConfig release_config() {
  const auto t = dgpp::minijson::parse(mimo_test::config_json());
  return dgpp::MimoTextConfig::parse(t.root);
}

const dgpp::MimoExpectedTensor& find(const std::vector<dgpp::MimoExpectedTensor>& v, const char* name) {
  for (const auto& e : v)
    if (e.name == name) return e;
  throw std::runtime_error(std::string("expected tensor missing from the table: ") + name);
}

std::filesystem::path landed_snapshot() {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) return {};
  const fs::path root = fs::path(home) / ".cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-RL/snapshots";
  if (!fs::is_directory(root)) return {};
  for (const auto& snap : fs::directory_iterator(root))
    if (fs::exists(snap.path() / "config.json") && fs::exists(snap.path() / "model.safetensors.index.json"))
      return snap.path();
  return {};
}

}  // namespace

DGPP_TEST(mimo_binding_table_has_the_release_shape) {
  const dgpp::MimoTextConfig cfg = release_config();
  // Layer 0 (global, dense): 2 norms + qkv pair + o_proj + 3 fp8 pairs = 11.
  const auto dense = dgpp::mimo_expected_layer_tensors(cfg, 0);
  require(dense.size() == 11, "dense layer " + std::to_string(dense.size()));
  // A sliding-window MoE layer: 2 norms + qkv pair + o_proj + sink + router 2 + 256 x 3 x 2.
  const auto swa = dgpp::mimo_expected_layer_tensors(cfg, 1);
  require(swa.size() == 2 + 2 + 1 + 1 + 2 + 256 * 6, "swa moe layer " + std::to_string(swa.size()));
  // A global MoE layer: no sink.
  const auto ga = dgpp::mimo_expected_layer_tensors(cfg, 5);
  require(ga.size() == swa.size() - 1, "global moe layer " + std::to_string(ga.size()));
  // The draft: 4 head tensors + 2 norms + qkv pair + o_proj + sink + 3 fp8 pairs.
  const auto draft = dgpp::mimo_expected_layer_tensors(cfg, cfg.mtp_layer());
  require(draft.size() == 4 + 2 + 2 + 1 + 1 + 6, "draft layer " + std::to_string(draft.size()));
  require(draft[0].name == "model.mtp.layers.0.enorm.weight", "draft prefix");
  require(find(draft, "model.mtp.layers.0.pre_mlp_layernorm.weight").cls == dgpp::MimoWeightClass::LayerNorm,
          "the draft's post norm is pre_mlp_layernorm");
  require(find(draft, "model.mtp.layers.0.final_layernorm.weight").cls == dgpp::MimoWeightClass::MtpHead, "final_layernorm");
  const auto all = dgpp::mimo_expected_text_tensors(cfg);
  require(all.size() == 3 + dense.size() + 39 * swa.size() + 8 * ga.size() + draft.size(),
          "table size " + std::to_string(all.size()));
  // The fused projection's shapes: the chunk-tiled scale grid.
  const dgpp::MimoExpectedTensor qkv0 = find(dense, "model.layers.0.self_attn.qkv_proj.weight");
  require(qkv0.dtype == dgpp::DType::F8_E4M3 && qkv0.shape == std::vector<int64_t>{13568, 4096}, "global qkv payload");
  const dgpp::MimoExpectedTensor qs0 = find(dense, "model.layers.0.self_attn.qkv_proj.weight_scale_inv");
  require(qs0.dtype == dgpp::DType::F32 && qs0.shape == std::vector<int64_t>{108, 32}, "global qkv scales (4 x 27)");
  const dgpp::MimoExpectedTensor qkv1 = find(swa, "model.layers.1.self_attn.qkv_proj.weight");
  require(qkv1.shape == std::vector<int64_t>{14848, 4096}, "swa qkv payload");
  const dgpp::MimoExpectedTensor qs1 = find(swa, "model.layers.1.self_attn.qkv_proj.weight_scale_inv");
  require(qs1.shape == std::vector<int64_t>{116, 32}, "swa qkv scales (4 x 29)");
  require(find(swa, "model.layers.1.self_attn.o_proj.weight").shape == std::vector<int64_t>{4096, 8192}, "o_proj");
  require(find(swa, "model.layers.1.self_attn.attention_sink_bias").shape == std::vector<int64_t>{64}, "sink");
  const dgpp::MimoExpectedTensor gate = find(swa, "model.layers.1.mlp.experts.7.gate_proj.weight");
  require(gate.dtype == dgpp::DType::U8 && gate.shape == std::vector<int64_t>{2048, 2048} && gate.expert == 7,
          "expert gate payload");
  require(find(swa, "model.layers.1.mlp.experts.7.gate_proj.weight_scale").shape == std::vector<int64_t>{2048, 128},
          "expert gate scales");
  require(find(swa, "model.layers.1.mlp.experts.7.down_proj.weight").shape == std::vector<int64_t>{4096, 1024},
          "expert down payload");
  require(find(swa, "model.layers.1.mlp.experts.7.down_proj.weight_scale").shape == std::vector<int64_t>{4096, 64},
          "expert down scales");
  require(find(dense, "model.layers.0.mlp.gate_proj.weight_scale_inv").shape == std::vector<int64_t>{128, 32}, "dense gate scales");
  require(find(dense, "model.layers.0.mlp.down_proj.weight_scale_inv").shape == std::vector<int64_t>{32, 128}, "dense down scales");
  require(find(swa, "model.layers.1.mlp.gate.e_score_correction_bias").dtype == dgpp::DType::F32, "router bias f32");
  size_t unused = 0;
  for (const auto& e : all) unused += e.unused();
  require(unused == 0, "nothing in the table is unused");
}

DGPP_TEST(mimo_binding_ignores_the_unserved_classes) {
  const dgpp::MimoTextConfig cfg = release_config();
  require(dgpp::mimo_ignored_tensor(cfg, "visual.blocks.3.attn.qkv.weight"), "visual");
  require(dgpp::mimo_ignored_tensor(cfg, "visual.merger.ln_q.weight"), "visual merger");
  require(dgpp::mimo_ignored_tensor(cfg, "audio_encoder.input_local_transformer.layers.0.mlp.up_proj.weight"), "audio");
  require(dgpp::mimo_ignored_tensor(cfg, "speech_embeddings.19.weight"), "speech");
  require(dgpp::mimo_ignored_tensor(cfg, "model.mtp.layers.1.enorm.weight"), "the second draft layer");
  require(dgpp::mimo_ignored_tensor(cfg, "model.mtp.layers.2.self_attn.qkv_proj.weight"), "the third draft layer");
  require(!dgpp::mimo_ignored_tensor(cfg, "model.mtp.layers.0.enorm.weight"), "the first draft layer is served");
  require(!dgpp::mimo_ignored_tensor(cfg, "model.layers.3.mlp.experts.0.gate_proj.weight"), "a main layer");
  require(!dgpp::mimo_ignored_tensor(cfg, "lm_head.weight"), "the head");
  require(!dgpp::mimo_ignored_tensor(cfg, "model.mtp.layers.x.enorm.weight"), "a malformed draft index");
  const auto t = dgpp::minijson::parse(
      mimo_test::config_json("\"num_nextn_predict_layers\": 3", "\"num_nextn_predict_layers\": 0"));
  const dgpp::MimoTextConfig no_mtp = dgpp::MimoTextConfig::parse(t.root);
  require(dgpp::mimo_ignored_tensor(no_mtp, "model.mtp.layers.0.enorm.weight"), "every draft layer without mtp");
}

DGPP_TEST(mimo_binding_validator_reports_by_class) {
  // A four-layer config from the fixture's shape: the validator over a
  // present map derived from the table itself, then with one tensor
  // missing, one dtype wrong, one shape wrong, one unexpected, plus the
  // ignored classes.
  const dgpp::MimoTextConfig cfg = release_config();
  const auto all = dgpp::mimo_expected_text_tensors(cfg);
  std::unordered_map<std::string, dgpp::MimoTensorDesc> present;
  for (const auto& e : all) present.emplace(e.name, dgpp::MimoTensorDesc{e.dtype, e.shape});
  present.emplace("visual.blocks.0.norm1.weight", dgpp::MimoTensorDesc{dgpp::DType::BF16, {1280}});
  present.emplace("model.mtp.layers.2.enorm.weight", dgpp::MimoTensorDesc{dgpp::DType::BF16, {4096}});
  {
    const dgpp::MimoBindReport rep = dgpp::mimo_validate_text_binding(cfg, present);
    require(rep.ok() && rep.matched == all.size() && rep.ignored == 2, "a complete checkpoint binds");
    require(rep.fp8_matrices == 49 + 3 + 3 && rep.fp4_matrices == 47 * 256 * 3, "matrix counts");
  }
  present.erase("model.layers.7.mlp.experts.3.up_proj.weight_scale");
  present["model.layers.2.self_attn.o_proj.weight"].dtype = dgpp::DType::F8_E4M3;
  present["model.layers.5.self_attn.qkv_proj.weight_scale_inv"].shape = {106, 32};
  present.emplace("model.layers.0.self_attn.q_proj.weight", dgpp::MimoTensorDesc{dgpp::DType::BF16, {12288, 4096}});
  const dgpp::MimoBindReport rep = dgpp::mimo_validate_text_binding(cfg, present);
  require(!rep.ok(), "a broken checkpoint is refused");
  require(rep.missing == 1 && rep.dtype_mismatch == 1 && rep.shape_mismatch == 1 && rep.unexpected == 1,
          "the report counts by class");
  require(rep.errors.size() == 4, "four errors");
}

DGPP_TEST(mimo_tp_geometry_accepts_the_deployment_worlds) {
  const dgpp::MimoTextConfig cfg = release_config();
  for (const int w : {1, 2, 4})
    for (int r = 0; r < w; ++r) dgpp::mimo_tp_validate_geometry(cfg, r, w);
  for (const int w : {3, 8}) {
    bool refused = false;
    try {
      dgpp::mimo_tp_validate_geometry(cfg, 0, w);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    require(refused, "world " + std::to_string(w) + " is refused (the chunk count)");
  }
}

DGPP_TEST(mimo_binding_matches_the_landed_checkpoint) {
  const auto snap = landed_snapshot();
  if (snap.empty())
    dgpp::test::skip("required checkpoint snapshot unavailable: XiaomiMiMo/MiMo-V2.6-Flash-RL");
  namespace fs = std::filesystem;
  // Only once every shard of the index is present (the download may be in flight).
  {
    std::ifstream idx(snap / "model.safetensors.index.json");
    std::string text((std::istreambuf_iterator<char>(idx)), std::istreambuf_iterator<char>());
    const auto parsed = dgpp::minijson::parse(text);
    for (const auto& m : parsed.root.at("weight_map").members()) {
      const fs::path shard = snap / std::string(m.value.as_string());
      if (!fs::exists(shard))
        dgpp::test::skip("checkpoint download incomplete; missing shard: " + shard.string());
    }
  }
  const dgpp::MimoTextConfig cfg = dgpp::MimoTextConfig::from_json_file((snap / "config.json").string());
  std::unordered_map<std::string, dgpp::MimoTensorDesc> present;
  for (const auto& entry : fs::directory_iterator(snap)) {
    if (entry.path().extension() != ".safetensors") continue;
    auto f = dgpp::SafetensorsFile::open(entry.path().string());
    f->for_each([&](const dgpp::TensorInfo& t) {
      present.emplace(t.name, dgpp::MimoTensorDesc{t.dtype, t.shape});
    });
  }
  const dgpp::MimoBindReport rep = dgpp::mimo_validate_text_binding(cfg, present);
  std::string errs;
  for (const auto& e : rep.errors) errs += "\n  " + e;
  require(rep.ok(), "binding: missing " + std::to_string(rep.missing) + ", dtype " +
                        std::to_string(rep.dtype_mismatch) + ", shape " + std::to_string(rep.shape_mismatch) +
                        ", unexpected " + std::to_string(rep.unexpected) + errs);
  require(rep.fp4_matrices == 47 * 256 * 3, "fp4 matrices " + std::to_string(rep.fp4_matrices));
  require(rep.ignored > 0, "the encoders are ignored");
}
