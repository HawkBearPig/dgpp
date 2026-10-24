// The full GLM-5.3 expected-tensor table: its shape on the release's
// config (the per-layer counts the shard headers carry), the TP geometry
// acceptance at the deployment worlds, and — when every shard of the
// checkpoint is in the hub cache — the full binding against every shard's
// header (78 layer shards and passthrough.safetensors: every expected
// tensor present with its dtype and shape, nothing unexpected).
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common/test.hpp"
#include "glm_dsa_config_json.hpp"
#include "loaders/minijson.hpp"
#include "loaders/safetensors.hpp"
#include "models/glm_dsa/binding.hpp"
#include "models/glm_dsa/config.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

dgpp::GlmDsaTextConfig release_config() {
  // minijson references the text it parses: keep it alive across parse().
  const std::string text = glm_dsa_test::config_json();
  const auto t = dgpp::minijson::parse(text);
  return dgpp::GlmDsaTextConfig::parse(t.root);
}

// The snapshot directory, only when config.json, the index and EVERY file
// the index names are present (a download in progress links the finished
// shards only; a partial snapshot would fail the binding, not skip it).
std::filesystem::path landed_snapshot() {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) return {};
  const fs::path root =
      fs::path(home) / ".cache/huggingface/hub/models--HawkBearPig--GLM-5.3-Int4-Int8Mix-RTN-g64/snapshots";
  if (!fs::is_directory(root)) return {};
  for (const auto& snap : fs::directory_iterator(root)) {
    const fs::path index = snap.path() / "model.safetensors.index.json";
    if (!fs::exists(snap.path() / "config.json") || !fs::exists(index)) continue;
    std::ifstream in(index);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto parsed = dgpp::minijson::parse(text);
    const dgpp::minijson::Value* wm = parsed.root.find("weight_map");
    if (!wm || !wm->is_object()) continue;
    bool complete = true;
    for (const auto& m : wm->members())
      if (!m.value.is_string() || !fs::exists(snap.path() / std::string(m.value.as_string()))) {
        complete = false;
        break;
      }
    if (complete) return snap.path();
  }
  return {};
}

}  // namespace

DGPP_TEST(glm_dsa_binding_table_has_the_release_shape) {
  const dgpp::GlmDsaTextConfig cfg = release_config();
  // A dense layer: 2 norms + 2 latent norms + 5 BF16 projections + 5 indexer + 3 MLP = 17.
  const auto dense = dgpp::glm_dsa_expected_layer_tensors(cfg, 0);
  require(dense.size() == 17, "dense layer " + std::to_string(dense.size()));
  // A MoE layer without an indexer: 4 norms + 5 x 3 attention + router 2 + 257 x 3 x 3 = 2334.
  const auto moe = dgpp::glm_dsa_expected_layer_tensors(cfg, 3);
  require(moe.size() == 2334, "moe layer " + std::to_string(moe.size()));
  // One that owns an indexer: 2334 + 5.
  const auto moe_idx = dgpp::glm_dsa_expected_layer_tensors(cfg, 6);
  require(moe_idx.size() == 2339, "indexer moe layer " + std::to_string(moe_idx.size()));
  // The draft: 4 head + 4 norms + 5 attention + 5 indexer + router 2 + 257 x 3 = 791.
  const auto draft = dgpp::glm_dsa_expected_layer_tensors(cfg, cfg.mtp_layer());
  require(draft.size() == 791, "draft layer " + std::to_string(draft.size()));
  require(draft[0].name == "model.layers.78.enorm.weight", "draft prefix");
  const auto all = dgpp::glm_dsa_expected_text_tensors(cfg);
  require(all.size() == 175985, "table size " + std::to_string(all.size()));
  size_t int4 = 0, int8 = 0, bf16_experts = 0, shapes = 0;
  for (const auto& e : all) {
    if (e.role == dgpp::GlmDsaTensorRole::IntPacked) (e.bits == 4 ? int4 : int8)++;
    if (e.role == dgpp::GlmDsaTensorRole::Bf16Expert) ++bf16_experts;
    if (e.role == dgpp::GlmDsaTensorRole::IntShape) ++shapes;
  }
  require(int4 == 75 * 256 * 3, "int4 matrices " + std::to_string(int4));
  require(int8 == 75 * 8, "int8 matrices " + std::to_string(int8));
  require(bf16_experts == 257 * 3, "draft expert matrices " + std::to_string(bf16_experts));
  require(shapes == int4 + int8, "shape records");
  // The packed shapes: gate [2048, 6144] int4 -> packed [2048, 768], scale [2048, 96];
  // o_proj [6144, 16384] int8 -> packed [6144, 4096], scale [6144, 256].
  for (const auto& e : moe) {
    if (e.name == "model.layers.3.mlp.experts.0.gate_proj.weight_packed")
      require(e.dtype == dgpp::DType::I32 && e.shape == std::vector<int64_t>{2048, 768}, "gate packed shape");
    if (e.name == "model.layers.3.mlp.experts.0.gate_proj.weight_scale")
      require(e.dtype == dgpp::DType::BF16 && e.shape == std::vector<int64_t>{2048, 96}, "gate scale shape");
    if (e.name == "model.layers.3.self_attn.o_proj.weight_packed")
      require(e.shape == std::vector<int64_t>{6144, 4096} && e.bits == 8, "o_proj packed shape");
    if (e.name == "model.layers.3.self_attn.kv_a_proj_with_mqa.weight_packed")
      require(e.shape == std::vector<int64_t>{576, 1536}, "kv_a packed shape (512 + 64 rows)");
    if (e.name == "model.layers.3.self_attn.kv_b_proj.weight_shape")
      require(e.dtype == dgpp::DType::I64 && e.shape == std::vector<int64_t>{2}, "shape record");
  }
}

DGPP_TEST(glm_dsa_tp_geometry_accepts_the_deployment_worlds) {
  const dgpp::GlmDsaTextConfig cfg = release_config();
  for (const int w : {1, 2, 4, 8, 16, 32})
    for (int r = 0; r < w; ++r) dgpp::glm_dsa_tp_validate_geometry(cfg, r, w);
  auto refused = [&](int world) {
    try {
      dgpp::glm_dsa_tp_validate_geometry(cfg, 0, world);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  require(refused(64), "a 32-wide expert slice is refused");  // 2048 / 64 = 32 < the group
  require(refused(3), "a world that does not divide the heads is refused");
}

DGPP_TEST(glm_dsa_binding_matches_the_landed_checkpoint) {
  const auto snap = landed_snapshot();
  if (snap.empty())
    dgpp::test::skip(
        "required checkpoint snapshot unavailable: HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64");
  namespace fs = std::filesystem;
  const dgpp::GlmDsaTextConfig cfg = dgpp::GlmDsaTextConfig::from_json_file((snap / "config.json").string());
  std::unordered_map<std::string, dgpp::GlmDsaTensorDesc> present;
  for (const auto& entry : fs::directory_iterator(snap)) {
    if (entry.path().extension() != ".safetensors") continue;
    auto f = dgpp::SafetensorsFile::open(entry.path().string());
    f->for_each([&](const dgpp::TensorInfo& t) {
      present.emplace(t.name, dgpp::GlmDsaTensorDesc{t.dtype, t.shape});
    });
  }
  const dgpp::GlmDsaBindReport rep = dgpp::glm_dsa_validate_text_binding(cfg, present);
  std::string errs;
  for (const auto& e : rep.errors) errs += "\n  " + e;
  require(rep.ok(), "binding: missing " + std::to_string(rep.missing) + ", dtype " +
                        std::to_string(rep.dtype_mismatch) + ", shape " + std::to_string(rep.shape_mismatch) +
                        ", unexpected " + std::to_string(rep.unexpected) + errs);
  require(rep.expected == 175985 && rep.matched == 175985, "every tensor bound");
  require(rep.packed_int4_matrices == 75 * 256 * 3, "int4 matrices " + std::to_string(rep.packed_int4_matrices));
  require(rep.packed_int8_matrices == 75 * 8, "int8 matrices " + std::to_string(rep.packed_int8_matrices));
  require(rep.bf16_expert_matrices == 257 * 3, "draft experts");
}

// --- the NF4I8 contract (2026-10-08) -------------------------------------------

namespace {
dgpp::GlmDsaTextConfig nf4i8_config() {
  const std::string text = glm_dsa_test::config_json_nf4i8();
  const auto t = dgpp::minijson::parse(text);
  return dgpp::GlmDsaTextConfig::parse(t.root);
}

std::filesystem::path landed_snapshot_of(const char* repo) {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) return {};
  const fs::path root = fs::path(home) / ".cache/huggingface/hub" / repo / "snapshots";
  if (!fs::is_directory(root)) return {};
  for (const auto& snap : fs::directory_iterator(root)) {
    const fs::path index = snap.path() / "model.safetensors.index.json";
    if (!fs::exists(snap.path() / "config.json") || !fs::exists(index)) continue;
    std::ifstream in(index);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto parsed = dgpp::minijson::parse(text);
    const dgpp::minijson::Value* wm = parsed.root.find("weight_map");
    if (!wm || !wm->is_object()) continue;
    bool complete = true;
    for (const auto& m : wm->members())
      if (!m.value.is_string() || !fs::exists(snap.path() / std::string(m.value.as_string()))) {
        complete = false;
        break;
      }
    if (complete) return snap.path();
  }
  return {};
}
}  // namespace

DGPP_TEST(glm_dsa_binding_table_has_the_nf4i8_shape) {
  const dgpp::GlmDsaTextConfig cfg = nf4i8_config();
  // The same census as the baseline: the routed triples change name and
  // scale geometry, nothing else.
  const auto moe = dgpp::glm_dsa_expected_layer_tensors(cfg, 3);
  require(moe.size() == 2334, "moe layer " + std::to_string(moe.size()));
  const auto all = dgpp::glm_dsa_expected_text_tensors(cfg);
  require(all.size() == 175985, "table size " + std::to_string(all.size()));
  size_t int4 = 0, int8 = 0, codebook = 0, packed_named = 0, indices_named = 0;
  for (const auto& e : all) {
    if (e.role != dgpp::GlmDsaTensorRole::IntPacked) continue;
    (e.bits == 4 ? int4 : int8)++;
    if (dgpp::packed_codebook(e.scale_fmt)) ++codebook;
    if (e.name.ends_with(".weight_indices")) ++indices_named;
    if (e.name.ends_with(".weight_packed")) ++packed_named;
    // The int8 triples keep format 0; every 4-bit triple is a codebook one.
    require((e.bits == 8) == (e.scale_fmt == dgpp::kPackedScaleBf16G64), "format by width: " + e.name);
  }
  require(int4 == 75 * 256 * 3 && codebook == int4 && indices_named == int4, "codebook matrices " + std::to_string(codebook));
  require(int8 == 75 * 8 && packed_named == int8, "int8 matrices " + std::to_string(int8));
  // The published shapes (manifests/layer-003.safetensors.json): gate
  // [2048, 6144] -> indices [2048, 768] I32, scale [2048, 48] BF16; down
  // [6144, 2048] -> indices [6144, 256], scale [6144, 16]; the attention
  // and shared triples as the baseline's.
  for (const auto& e : moe) {
    if (e.name == "model.layers.3.mlp.experts.0.gate_proj.weight_indices")
      require(e.dtype == dgpp::DType::I32 && e.shape == std::vector<int64_t>{2048, 768} && e.bits == 4, "gate indices shape");
    if (e.name == "model.layers.3.mlp.experts.0.gate_proj.weight_scale")
      require(e.dtype == dgpp::DType::BF16 && e.shape == std::vector<int64_t>{2048, 48}, "gate scale shape");
    if (e.name == "model.layers.3.mlp.experts.255.down_proj.weight_indices")
      require(e.shape == std::vector<int64_t>{6144, 256}, "down indices shape");
    if (e.name == "model.layers.3.mlp.experts.255.down_proj.weight_scale")
      require(e.shape == std::vector<int64_t>{6144, 16}, "down scale shape");
    if (e.name == "model.layers.3.mlp.shared_experts.gate_proj.weight_packed")
      require(e.shape == std::vector<int64_t>{2048, 1536} && e.bits == 8, "shared packed shape");
    if (e.name == "model.layers.3.mlp.shared_experts.gate_proj.weight_scale")
      require(e.shape == std::vector<int64_t>{2048, 96}, "shared scale shape");
    if (e.name == "model.layers.3.self_attn.o_proj.weight_packed")
      require(e.shape == std::vector<int64_t>{6144, 4096} && e.bits == 8, "o_proj packed shape");
    require(e.name.find("experts.0.gate_proj.weight_packed") == std::string::npos, "no int4 weight_packed under the codebook");
  }
  // The draft's experts stay BF16 (requantized at load, unrotated).
  const auto draft = dgpp::glm_dsa_expected_layer_tensors(cfg, cfg.mtp_layer());
  require(draft.size() == 791, "draft layer " + std::to_string(draft.size()));
  // A baseline-named checkpoint fails the NF4I8 binding by name, and the
  // other way round: the words tensor's name is the format's.
  {
    std::unordered_map<std::string, dgpp::GlmDsaTensorDesc> present;
    for (const auto& e : dgpp::glm_dsa_expected_text_tensors(release_config()))
      present.emplace(e.name, dgpp::GlmDsaTensorDesc{e.dtype, e.shape});
    const dgpp::GlmDsaBindReport rep = dgpp::glm_dsa_validate_text_binding(cfg, present, 4);
    require(!rep.ok() && rep.missing == 75 * 256 * 3 && rep.unexpected == 75 * 256 * 3, "baseline tensors under the nf4i8 config");
    require(!rep.errors.empty() && rep.errors[0].find("weight_indices") != std::string::npos, "named by the missing indices");
  }
  {
    std::unordered_map<std::string, dgpp::GlmDsaTensorDesc> present;
    for (const auto& e : all) present.emplace(e.name, dgpp::GlmDsaTensorDesc{e.dtype, e.shape});
    const dgpp::GlmDsaBindReport rep = dgpp::glm_dsa_validate_text_binding(cfg, present);
    require(rep.ok() && rep.codebook_matrices == 75 * 256 * 3 && rep.packed_int4_matrices == 75 * 256 * 3 &&
                rep.packed_int8_matrices == 75 * 8,
            "the nf4i8 table binds itself");
    const dgpp::GlmDsaBindReport cross = dgpp::glm_dsa_validate_text_binding(release_config(), present, 4);
    require(!cross.ok() && cross.missing == 75 * 256 * 3, "nf4i8 tensors under the baseline config");
  }
}

DGPP_TEST(glm_dsa_tp_geometry_under_the_codebook_group) {
  const dgpp::GlmDsaTextConfig cfg = nf4i8_config();
  // 2048 / W must hold whole 128-groups: worlds 1..16 do, 32 (a 64-wide
  // slice) does not; the baseline's group 64 accepted 32.
  for (const int w : {1, 2, 4, 8, 16})
    for (int r = 0; r < w; ++r) dgpp::glm_dsa_tp_validate_geometry(cfg, r, w);
  bool refused = false;
  try {
    dgpp::glm_dsa_tp_validate_geometry(cfg, 0, 32);
  } catch (const std::invalid_argument& e) {
    refused = std::string(e.what()).find("128") != std::string::npos;
  }
  require(refused, "a 64-wide routed slice is refused under the 128 group");
  dgpp::glm_dsa_tp_validate_geometry(release_config(), 0, 32);
}

namespace {
dgpp::GlmDsaTextConfig mixed346_config() {
  const std::string text = glm_dsa_test::config_json_mixed346();
  const auto t = dgpp::minijson::parse(text);
  return dgpp::GlmDsaTextConfig::parse(t.root);
}
}  // namespace

DGPP_TEST(glm_dsa_binding_table_has_the_mixed346_shape) {
  const dgpp::GlmDsaTextConfig cfg = mixed346_config();
  // The same census as the baseline: every routed triple is named by its
  // own form, nothing else moves.
  const auto moe = dgpp::glm_dsa_expected_layer_tensors(cfg, 3);
  require(moe.size() == 2334, "moe layer " + std::to_string(moe.size()));
  const auto all = dgpp::glm_dsa_expected_text_tensors(cfg);
  require(all.size() == 175985, "table size " + std::to_string(all.size()));
  // Expert e of layer l is kMixed346Forms[(l + e) % 8], 32 experts a form
  // a layer: gate/up at 3 bits (334, 333, 336) 192 matrices, at 4 (444,
  // 446, 443) 192, at 6 (666) 64; down at 3 (333, 443) 64, at 4 (444, 334)
  // 64, at 6 (446, 336, 666) 96; existing 96 int4 g64 matrices.
  size_t by_bits[9] = {}, mixed_fmt = 0, existing_fmt = 0, indices_named = 0, packed_named = 0;
  for (const auto& e : all) {
    if (e.role != dgpp::GlmDsaTensorRole::IntPacked) continue;
    ++by_bits[e.bits];
    if (e.name.ends_with(".weight_indices")) ++indices_named;
    if (e.name.ends_with(".weight_packed")) ++packed_named;
    if (e.cls != dgpp::GlmDsaWeightClass::RoutedExpert) {
      require(e.scale_fmt == dgpp::kPackedScaleBf16G64 && e.bits == 8, "attention and shared triples keep format 0: " + e.name);
      continue;
    }
    if (e.scale_fmt == dgpp::kPackedScaleBf16G128Mixed346) {
      ++mixed_fmt;
      require(e.name.ends_with(".weight_indices") && (e.bits == 3 || e.bits == 4 || e.bits == 6), "a converted matrix: " + e.name);
    } else {
      ++existing_fmt;
      require(e.scale_fmt == dgpp::kPackedScaleBf16G64 && e.bits == 4 && e.name.ends_with(".weight_packed"),
              "an existing expert's matrix: " + e.name);
    }
  }
  require(by_bits[3] == 75u * 256 && by_bits[6] == 75u * 160 && by_bits[4] == 75u * (256 + 96) && by_bits[8] == 75u * 8,
          "matrices by width");
  require(mixed_fmt == 75u * 672 && existing_fmt == 75u * 96 && indices_named == mixed_fmt &&
              packed_named == existing_fmt + 75u * 8,
          "matrices by form");
  // The published shapes: a 3-bit gate [2048, 6144] -> indices [2048, 576]
  // I32 (6144 x 3 / 32), scale [2048, 48]; a 6-bit down [6144, 2048] ->
  // [6144, 384], scale [6144, 16]; an existing gate [2048, 768], scale
  // [2048, 96]. Layer 3's experts 0 (333), 1 (666), 4 (existing), 6 (334).
  for (const auto& e : moe) {
    if (e.name == "model.layers.3.mlp.experts.0.gate_proj.weight_indices")
      require(e.dtype == dgpp::DType::I32 && e.shape == std::vector<int64_t>{2048, 576} && e.bits == 3, "3-bit gate indices");
    if (e.name == "model.layers.3.mlp.experts.0.gate_proj.weight_scale")
      require(e.dtype == dgpp::DType::BF16 && e.shape == std::vector<int64_t>{2048, 48}, "3-bit gate scale");
    if (e.name == "model.layers.3.mlp.experts.0.down_proj.weight_indices")
      require(e.shape == std::vector<int64_t>{6144, 192} && e.bits == 3, "3-bit down indices");
    if (e.name == "model.layers.3.mlp.experts.6.down_proj.weight_indices")
      require(e.shape == std::vector<int64_t>{6144, 256} && e.bits == 4, "334's down is 4-bit");
    if (e.name == "model.layers.3.mlp.experts.1.down_proj.weight_indices")
      require(e.shape == std::vector<int64_t>{6144, 384} && e.bits == 6, "6-bit down indices");
    if (e.name == "model.layers.3.mlp.experts.1.down_proj.weight_scale")
      require(e.shape == std::vector<int64_t>{6144, 16}, "6-bit down scale");
    if (e.name == "model.layers.3.mlp.experts.1.up_proj.weight_indices")
      require(e.shape == std::vector<int64_t>{2048, 1152} && e.bits == 6, "6-bit up indices");
    if (e.name == "model.layers.3.mlp.experts.4.gate_proj.weight_packed")
      require(e.shape == std::vector<int64_t>{2048, 768} && e.bits == 4 && e.scale_fmt == dgpp::kPackedScaleBf16G64,
              "existing gate packed");
    if (e.name == "model.layers.3.mlp.experts.4.gate_proj.weight_scale")
      require(e.shape == std::vector<int64_t>{2048, 96}, "existing gate scale");
    require(e.name.find("experts.4.gate_proj.weight_indices") == std::string::npos, "no indices on an existing expert");
    require(e.name.find("experts.0.gate_proj.weight_packed") == std::string::npos, "no weight_packed on a converted expert");
  }
  const auto draft = dgpp::glm_dsa_expected_layer_tensors(cfg, cfg.mtp_layer());
  require(draft.size() == 791, "draft layer " + std::to_string(draft.size()));
  {
    // The table binds itself; the baseline's tensors fail it by the
    // converted experts' names (the existing experts' names and shapes are
    // the baseline's own).
    std::unordered_map<std::string, dgpp::GlmDsaTensorDesc> present;
    for (const auto& e : all) present.emplace(e.name, dgpp::GlmDsaTensorDesc{e.dtype, e.shape});
    const dgpp::GlmDsaBindReport rep = dgpp::glm_dsa_validate_text_binding(cfg, present);
    require(rep.ok() && rep.codebook_matrices == 75u * 672 && rep.packed_mixed_matrices == 75u * (256 + 160) &&
                rep.packed_int4_matrices == 75u * (256 + 96) && rep.packed_int8_matrices == 75u * 8,
            "the mixed346 table binds itself");
    std::unordered_map<std::string, dgpp::GlmDsaTensorDesc> baseline;
    for (const auto& e : dgpp::glm_dsa_expected_text_tensors(release_config()))
      baseline.emplace(e.name, dgpp::GlmDsaTensorDesc{e.dtype, e.shape});
    const dgpp::GlmDsaBindReport cross = dgpp::glm_dsa_validate_text_binding(cfg, baseline, 4);
    require(!cross.ok() && cross.missing == 75u * 672 && cross.unexpected == 75u * 672, "baseline tensors under the mixed346 config");
  }
}

DGPP_TEST(glm_dsa_tp_geometry_under_the_mixed346_group) {
  const dgpp::GlmDsaTextConfig cfg = mixed346_config();
  for (const int w : {1, 2, 4, 8, 16})
    for (int r = 0; r < w; ++r) dgpp::glm_dsa_tp_validate_geometry(cfg, r, w);
  bool refused = false;
  try {
    dgpp::glm_dsa_tp_validate_geometry(cfg, 0, 32);
  } catch (const std::invalid_argument& e) {
    refused = std::string(e.what()).find("128") != std::string::npos;
  }
  require(refused, "a 64-wide routed slice is refused under the 128 group");
}

DGPP_TEST(glm_dsa_binding_matches_the_landed_nf4i8_checkpoint) {
  const auto snap = landed_snapshot_of("models--HawkBearPig--GLM-5.3-NF4I8-GPTQ-H32-g128");
  if (snap.empty())
    dgpp::test::skip(
        "required checkpoint snapshot unavailable: HawkBearPig/GLM-5.3-NF4I8-GPTQ-H32-g128");
  namespace fs = std::filesystem;
  const dgpp::GlmDsaTextConfig cfg = dgpp::GlmDsaTextConfig::from_json_file((snap / "config.json").string());
  std::unordered_map<std::string, dgpp::GlmDsaTensorDesc> present;
  for (const auto& entry : fs::directory_iterator(snap)) {
    if (entry.path().extension() != ".safetensors") continue;
    auto f = dgpp::SafetensorsFile::open(entry.path().string());
    f->for_each([&](const dgpp::TensorInfo& t) {
      present.emplace(t.name, dgpp::GlmDsaTensorDesc{t.dtype, t.shape});
    });
  }
  const dgpp::GlmDsaBindReport rep = dgpp::glm_dsa_validate_text_binding(cfg, present);
  std::string errs;
  for (const auto& e : rep.errors) errs += "\n  " + e;
  require(rep.ok(), "binding: missing " + std::to_string(rep.missing) + ", dtype " +
                        std::to_string(rep.dtype_mismatch) + ", shape " + std::to_string(rep.shape_mismatch) +
                        ", unexpected " + std::to_string(rep.unexpected) + errs);
  require(rep.expected == 175985 && rep.matched == 175985, "every tensor bound");
  require(rep.codebook_matrices == 75 * 256 * 3, "codebook matrices " + std::to_string(rep.codebook_matrices));
  require(rep.packed_int8_matrices == 75 * 8, "int8 matrices " + std::to_string(rep.packed_int8_matrices));
  require(rep.bf16_expert_matrices == 257 * 3, "draft experts");
}
