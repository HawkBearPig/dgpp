// The Inkling 2 expected-tensor tables: their shape on the release's config,
// the expert formats per layer, the validator's report, the TP geometry
// acceptance at the deployment worlds and, when the checkpoint is in the hub
// cache, the full binding against the two files' headers.
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common/test.hpp"
#include "inkling2_config_json.hpp"
#include "loaders/minijson.hpp"
#include "loaders/safetensors.hpp"
#include "models/inkling2/binding.hpp"
#include "models/inkling2/config.hpp"

namespace {

using Table = std::vector<dgpp::Inkling2ExpectedTensor>;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

dgpp::Inkling2Config release_config() {
  // The parsed tree views the text it was parsed from.
  const std::string config = inkling2_test::config_json();
  const auto c = dgpp::minijson::parse(config);
  const std::string quant = inkling2_test::quant_json();
  const auto q = dgpp::minijson::parse(quant);
  return dgpp::Inkling2Config::parse(c.root, q.root);
}

const dgpp::Inkling2ExpectedTensor& find(const Table& v, const char* name) {
  for (const auto& e : v)
    if (e.name == name) return e;
  throw std::runtime_error(std::string("expected tensor missing from the table: ") + name);
}

bool has(const Table& v, const char* name) {
  for (const auto& e : v)
    if (e.name == name) return true;
  return false;
}

std::unordered_map<std::string, dgpp::Inkling2TensorDesc> present_of(const Table& table) {
  std::unordered_map<std::string, dgpp::Inkling2TensorDesc> present;
  for (const auto& e : table) present.emplace(e.name, dgpp::Inkling2TensorDesc{e.dtype, e.shape});
  return present;
}

std::filesystem::path landed_snapshot() {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) return {};
  const fs::path root = fs::path(home) / ".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots";
  if (!fs::is_directory(root)) return {};
  for (const auto& snap : fs::directory_iterator(root))
    if (fs::exists(snap.path() / "config.json")) return snap.path();
  return {};
}

}  // namespace

DGPP_TEST(inkling2_binding_table_has_the_release_shape) {
  const dgpp::Inkling2Config cfg = release_config();
  const Table text = dgpp::inkling2_expected_text_tensors(cfg);
  const Table vision = dgpp::inkling2_expected_vision_tensors(cfg);
  const Table draft = dgpp::inkling2_expected_draft_tensors(cfg);
  // The safetensors headers' counts.
  require(text.size() == 822, "text table size " + std::to_string(text.size()));
  require(vision.size() == 422, "vision table size " + std::to_string(vision.size()));
  require(draft.size() == 54, "draft table size " + std::to_string(draft.size()));
  using Shape = std::vector<int64_t>;
  const Table local = dgpp::inkling2_expected_layer_tensors(cfg, 7);
  const Table full = dgpp::inkling2_expected_layer_tensors(cfg, 11);
  const Table mx = dgpp::inkling2_expected_layer_tensors(cfg, 21);
  const Table first = dgpp::inkling2_expected_layer_tensors(cfg, 0);
  require(find(local, "model.llm.layers.7.attn.wq_du.weight").shape == Shape{3072, 2304}, "wq");
  require(find(local, "model.llm.layers.7.attn.wr_du.weight").shape == Shape{384, 2304}, "wr");
  require(find(local, "model.llm.layers.7.attn.gate_attn.weight").shape == Shape{3072, 2304}, "gate");
  require(find(local, "model.llm.layers.7.attn.attn_sink_proj.weight").shape == Shape{24, 2304}, "sink");
  require(find(local, "model.llm.layers.7.attn.wk_dv.weight").shape == Shape{1024, 2304}, "wk");
  require(find(local, "model.llm.layers.7.attn.wo_ud.weight").shape == Shape{2304, 3072}, "wo");
  require(find(local, "model.llm.layers.7.attn.q_l2_clip.scale").shape == Shape{24, 128}, "q scale");
  require(find(local, "model.llm.layers.7.attn.rel_l2_clip.scale").shape == Shape{24, 16}, "r scale");
  require(find(local, "model.llm.layers.7.attn.rel_logits_proj.proj").shape == Shape{16, 1024}, "relative table");
  require(find(local, "model.llm.layers.7.attn.k_sconv.weight").shape == Shape{1024, 1, 4}, "static K taps");
  require(find(local, "model.llm.layers.7.attn_norm.weight").shape == Shape{2304}, "attn norm");
  require(find(local, "model.llm.layers.7.attn_sconv.weight").shape == Shape{2304, 1, 4}, "attention taps");
  require(find(local, "model.llm.attn_res.attn_queries.6").shape == Shape{2304}, "a local layer's one query");
  require(find(local, "model.llm.attn_res.mlp_queries.7").shape == Shape{2304}, "mlp query");
  require(!has(first, "model.llm.attn_res.attn_queries.0") && has(first, "model.llm.attn_res.mlp_queries.0"),
          "layer 0 mixes nothing into its attention");
  // A full-attention layer: three norms and queries, dynamic K / V taps.
  require(find(full, "model.llm.attn_res.attn_queries.10").shape == Shape{3, 2304}, "three queries");
  require(has(full, "model.llm.layers.11.q_input_norm.weight") && has(full, "model.llm.layers.11.k_input_norm.weight") &&
              has(full, "model.llm.layers.11.v_input_norm.weight") && !has(full, "model.llm.layers.11.attn_norm.weight"),
          "three input norms");
  require(find(full, "model.llm.layers.11.attn.k_sconv.static_w").shape == Shape{4, 1024}, "static part of dynamic taps");
  require(find(full, "model.llm.layers.11.attn.v_sconv.weight_proj").shape == Shape{32, 2304}, "tap projection");
  // The loop controllers on the core layers only.
  require(find(local, "model.llm.layers.7.controllers.k.down_weight").shape == Shape{64, 4608}, "controller down");
  require(find(local, "model.llm.layers.7.controllers.mlp.gain_weight").shape == Shape{2304, 64}, "controller gain");
  require(!has(first, "model.llm.layers.0.controllers.attn.down_weight") &&
              !has(mx, "model.llm.layers.21.controllers.attn.down_weight"),
          "no controllers outside the core");
  // NVFP4 experts.
  const auto& p13 = find(local, "model.llm.layers.7.mlp.experts.w13_weight");
  require(p13.dtype == dgpp::DType::U8 && p13.shape == Shape{128, 2304, 1152} && p13.projection == 0, "nvfp4 w13 payload");
  const auto& s13 = find(local, "model.llm.layers.7.mlp.experts.w13_weight_scale");
  require(s13.dtype == dgpp::DType::F8_E4M3 && s13.shape == Shape{128, 2304, 144}, "nvfp4 w13 scales");
  require(find(local, "model.llm.layers.7.mlp.experts.w13_weight_amax").shape == Shape{128}, "amax");
  require(find(local, "model.llm.layers.7.mlp.experts.w13_weight_outer_scale_denominator").shape == Shape{1}, "denominator");
  require(find(local, "model.llm.layers.7.mlp.experts.w2_weight").shape == Shape{128, 2304, 576}, "nvfp4 w2 payload");
  require(find(local, "model.llm.layers.7.mlp.experts.w2_weight_scale").shape == Shape{128, 2304, 72}, "nvfp4 w2 scales");
  // MXFP8 experts.
  const auto& m13 = find(mx, "model.llm.layers.21.mlp.experts.w13_weight");
  require(m13.dtype == dgpp::DType::F8_E4M3 && m13.shape == Shape{128, 2304, 2304}, "mxfp8 w13 payload");
  const auto& ms = find(mx, "model.llm.layers.21.mlp.experts.w13_weight_scale");
  require(ms.dtype == dgpp::DType::U8 && ms.shape == Shape{128, 2304, 72}, "mxfp8 w13 scales");
  require(find(mx, "model.llm.layers.21.mlp.experts.w2_weight_scale").shape == Shape{128, 2304, 36}, "mxfp8 w2 scales");
  require(!has(mx, "model.llm.layers.21.mlp.experts.w13_weight_amax"), "mxfp8 has no amax");
  // Globals.
  const Table globals = dgpp::inkling2_expected_global_tensors(cfg);
  require(find(globals, "model.llm.embed.weight").shape == Shape{201024, 2304}, "padded embedding");
  require(find(globals, "model.llm.middle_cycle_cross_pass_kv_queries").shape == Shape{24, 2304}, "message queries");
  // The draft model.
  require(find(draft, "fuse_weight").shape == Shape{4, 4}, "fusion");
  require(find(draft, "markov_head.rnn_proj.weight").shape == Shape{768, 5120}, "rnn_proj");
  require(find(draft, "markov_head.markov_w1.weight").shape == Shape{201024, 256}, "markov_w1");
  require(find(draft, "layers.1.self_attn.k_proj_ctx.weight").shape == Shape{1024, 2304}, "context keys");
  // The vision tower.
  require(find(vision, "model.visual.backbone.embeddings.patch_embedding.weight").shape == Shape{1024, 1200}, "patches");
  require(find(vision, "model.visual.proj_in.weight").shape == Shape{4096, 4096}, "merger");
  require(find(vision, "model.visual.proj_out.weight").shape == Shape{2304, 4096}, "projector");
}

DGPP_TEST(inkling2_binding_validator_reports_by_class) {
  const dgpp::Inkling2Config cfg = release_config();
  const Table all = dgpp::inkling2_expected_text_tensors(cfg);
  auto present = present_of(all);
  {
    const dgpp::Inkling2BindReport rep = dgpp::inkling2_validate_binding(all, present);
    require(rep.ok() && rep.matched == all.size(), "a complete checkpoint binds");
    require(rep.fp4_tensors == 40 && rep.mxfp8_tensors == 8, "expert tensor counts");
  }
  present.erase("model.llm.layers.3.mlp.experts.w2_weight_scale");
  present["model.llm.layers.2.attn.wo_ud.weight"].dtype = dgpp::DType::F8_E4M3;
  present["model.llm.layers.20.mlp.experts.w13_weight_scale"].shape = {128, 2304, 144};
  present.emplace("model.llm.layers.0.attn.qkvr.weight", dgpp::Inkling2TensorDesc{dgpp::DType::BF16, {8600, 2304}});
  const dgpp::Inkling2BindReport rep = dgpp::inkling2_validate_binding(all, present);
  require(!rep.ok(), "a broken checkpoint is refused");
  require(rep.missing == 1 && rep.dtype_mismatch == 1 && rep.shape_mismatch == 1 && rep.unexpected == 1,
          "the report counts by class");
}

DGPP_TEST(inkling2_tp_geometry_accepts_the_deployment_worlds) {
  const dgpp::Inkling2Config cfg = release_config();
  for (const int w : {1, 2, 4})
    for (int r = 0; r < w; ++r) dgpp::inkling2_tp_validate_geometry(cfg, r, w);
  // 3 does not divide the heads. 8 leaves a 144-wide expert slice, not a
  // whole number of 32-value scale blocks.
  for (const int w : {3, 8}) {
    bool refused = false;
    try {
      dgpp::inkling2_tp_validate_geometry(cfg, 0, w);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    require(refused, "world " + std::to_string(w) + " is refused");
  }
}

DGPP_TEST(inkling2_binding_matches_the_landed_checkpoint) {
  namespace fs = std::filesystem;
  const fs::path snap = landed_snapshot();
  if (snap.empty()) dgpp::test::skip("required checkpoint snapshot unavailable: thinkingmachines/Inkling-2-Mini");
  const fs::path main_file = snap / "model-00001-of-00001.safetensors";
  const fs::path draft_file = snap / "mtp.safetensors";
  if (!fs::exists(main_file) || !fs::exists(draft_file))
    dgpp::test::skip("checkpoint download incomplete under " + snap.string());
  const dgpp::Inkling2Config cfg = dgpp::Inkling2Config::from_checkpoint_dir(snap.string());
  const auto bind = [&](const fs::path& file, const Table& expected, const char* what) {
    std::unordered_map<std::string, dgpp::Inkling2TensorDesc> present;
    auto f = dgpp::SafetensorsFile::open(file.string());
    f->for_each([&](const dgpp::TensorInfo& t) { present.emplace(t.name, dgpp::Inkling2TensorDesc{t.dtype, t.shape}); });
    const dgpp::Inkling2BindReport rep = dgpp::inkling2_validate_binding(expected, present);
    std::string errs;
    for (const auto& e : rep.errors) errs += "\n  " + e;
    require(rep.ok(), std::string(what) + " binding: missing " + std::to_string(rep.missing) + ", dtype " +
                          std::to_string(rep.dtype_mismatch) + ", shape " + std::to_string(rep.shape_mismatch) +
                          ", unexpected " + std::to_string(rep.unexpected) + errs);
    return rep;
  };
  Table main_table = dgpp::inkling2_expected_text_tensors(cfg);
  const Table vision = dgpp::inkling2_expected_vision_tensors(cfg);
  main_table.insert(main_table.end(), vision.begin(), vision.end());
  const dgpp::Inkling2BindReport rep = bind(main_file, main_table, "main");
  require(rep.matched == 1244 && rep.fp4_tensors == 40 && rep.mxfp8_tensors == 8, "the main file's counts");
  bind(draft_file, dgpp::inkling2_expected_draft_tensors(cfg), "draft");
}
