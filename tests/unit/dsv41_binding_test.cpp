// The DeepSeek-V4.1-Flash expected-tensor table: its shape on the release's
// config (the per-layer counts the shard headers carry), the TP geometry
// acceptance at the deployment worlds, and — against the checkpoint's
// shard headers — the full binding: every expected tensor present with its
// dtype and shape, nothing unexpected. The checkpoint gate runs on the hub
// snapshot once every indexed file is present, or on the directory named by
// DGPP_DSV41_CHECKPOINT_DIR (a header-only mirror suffices: no payload is read).
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common/test.hpp"
#include "dsv41_config_json.hpp"
#include "loaders/minijson.hpp"
#include "loaders/safetensors.hpp"
#include "models/dsv41/binding.hpp"
#include "models/dsv41/config.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

dgpp::Dsv41TextConfig release_config() {
  const std::string text = dsv41_test::config_json();
  const auto t = dgpp::minijson::parse(text);
  return dgpp::Dsv41TextConfig::parse(t.root);
}

constexpr size_t kTotalTensors = 96085;

bool snapshot_complete(const std::filesystem::path& snap) {
  namespace fs = std::filesystem;
  const fs::path index = snap / "model.safetensors.index.json";
  if (!fs::exists(snap / "config.json") || !fs::exists(index)) return false;
  std::ifstream in(index);
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const auto parsed = dgpp::minijson::parse(text);
  const dgpp::minijson::Value* wm = parsed.root.find("weight_map");
  if (!wm || !wm->is_object()) return false;
  for (const auto& m : wm->members())
    if (!m.value.is_string() || !fs::exists(snap / std::string(m.value.as_string()))) return false;
  return true;
}

std::filesystem::path landed_snapshot() {
  namespace fs = std::filesystem;
  if (const char* dir = std::getenv("DGPP_DSV41_CHECKPOINT_DIR"); dir && *dir) {
    const fs::path p(dir);
    return snapshot_complete(p) ? p : fs::path{};
  }
  const char* home = std::getenv("HOME");
  if (!home) return {};
  // The 1017 refinement (2026-10-18) binds identically to the original
  // release (the same tensors in the same shards); either snapshot serves.
  for (const char* repo : {"models--deepseek-ai--DeepSeek-V4.1-Flash-1017", "models--deepseek-ai--DeepSeek-V4.1-Flash"}) {
    const fs::path root = fs::path(home) / ".cache/huggingface/hub" / repo / "snapshots";
    if (!fs::is_directory(root)) continue;
    for (const auto& snap : fs::directory_iterator(root))
      if (snapshot_complete(snap.path())) return snap.path();
  }
  return {};
}

}  // namespace

DGPP_TEST(dsv41_binding_table_has_the_release_shape) {
  const dgpp::Dsv41TextConfig cfg = release_config();
  // A Reuse layer: 2 norms + 6 mHC + (q_norm, kv_norm, sink, 5 fp8 pairs) 13
  // + router 3 + 384 x 3 pairs 2304 + shared 3 pairs 6 = 2334 — the count of
  // the checkpoint's per-layer shards.
  require(dgpp::dsv41_expected_layer_tensors(cfg, 3).size() == 2334, "reuse layer");
  require(dgpp::dsv41_expected_layer_tensors(cfg, 0).size() == 2334, "window layer");
  // A Reindex layer adds the indexer's wq_b pair and weights_proj.
  require(dgpp::dsv41_expected_layer_tensors(cfg, 24).size() == 2337, "reindex layer");
  // A Full layer at ratio 2 adds wk + k_norm and the compressor's wkv, wgate, norm.
  require(dgpp::dsv41_expected_layer_tensors(cfg, 2).size() == 2342, "full layer (ratio 2)");
  require(dgpp::dsv41_expected_layer_tensors(cfg, 8).size() == 2342, "full layer 8");
  // Layer 20 (ratio 1) has no gate.
  require(dgpp::dsv41_expected_layer_tensors(cfg, 20).size() == 2341, "full layer (ratio 1)");
  // The Engram layers add the table pair, the wkv pair and the two gate weights.
  require(dgpp::dsv41_expected_layer_tensors(cfg, 1).size() == 2340, "engram layer 1");
  require(dgpp::dsv41_expected_layer_tensors(cfg, 14).size() == 2348, "engram + full layer 14");
  // The draft stages: 128 experts, stage 0 with main_proj/main_norm, stage 2 with the heads.
  require(dgpp::dsv41_expected_layer_tensors(cfg, 40).size() == 801, "draft stage 0");
  require(dgpp::dsv41_expected_layer_tensors(cfg, 41).size() == 798, "draft stage 1");
  require(dgpp::dsv41_expected_layer_tensors(cfg, 42).size() == 802, "draft stage 2");
  require(dgpp::dsv41_expected_layer_tensors(cfg, 40)[0].name == "mtp.0.main_proj.weight", "draft prefix");
  require(dgpp::dsv41_expected_global_tensors(cfg).size() == 3, "globals");
  require(dgpp::dsv41_expected_vision_tensors(cfg).size() == 266, "vision skipped set");
  const auto all = dgpp::dsv41_expected_text_tensors(cfg);
  require(all.size() == kTotalTensors, "table size " + std::to_string(all.size()));
  size_t fp4 = 0, fp8 = 0, tables = 0, skipped = 0;
  for (const auto& e : all) {
    switch (e.role) {
      case dgpp::Dsv41TensorRole::Fp4Payload: ++fp4; break;
      case dgpp::Dsv41TensorRole::Fp8Payload: ++fp8; break;
      case dgpp::Dsv41TensorRole::TablePayload: ++tables; break;
      case dgpp::Dsv41TensorRole::Skipped: ++skipped; break;
      default: break;
    }
  }
  require(fp4 == (40 * 384 + 3 * 128) * 3, "fp4 matrices " + std::to_string(fp4));
  // fp8: 5 attention + 3 shared per layer (43), 8 indexer wq_b, 2 engram wkv, 1 main_proj.
  require(fp8 == 43 * 8 + 8 + 2 + 1, "fp8 matrices " + std::to_string(fp8));
  require(tables == 2 && skipped == 266, "tables / skipped");
  // Shapes pinned to the headers.
  for (const auto& e : all) {
    if (e.name == "layers.3.ffn.experts.0.w1.weight")
      require(e.dtype == dgpp::DType::I8 && e.shape == std::vector<int64_t>{2304, 2560}, "w1 payload");
    if (e.name == "layers.3.ffn.experts.0.w1.scale")
      require(e.dtype == dgpp::DType::F8_E8M0 && e.shape == std::vector<int64_t>{2304, 160}, "w1 scale");
    if (e.name == "layers.3.ffn.experts.0.w2.weight") require(e.shape == std::vector<int64_t>{5120, 1152}, "w2 payload");
    if (e.name == "layers.3.ffn.experts.0.w2.scale") require(e.shape == std::vector<int64_t>{5120, 72}, "w2 scale");
    if (e.name == "layers.3.attn.wq_b.weight")
      require(e.dtype == dgpp::DType::F8_E4M3 && e.shape == std::vector<int64_t>{32768, 1280}, "wq_b");
    if (e.name == "layers.3.attn.wq_b.scale") require(e.shape == std::vector<int64_t>{1024, 40}, "wq_b scale");
    if (e.name == "layers.3.attn.wo_a.weight") require(e.shape == std::vector<int64_t>{8192, 4096}, "wo_a");
    if (e.name == "layers.3.attn.wo_a.scale") require(e.shape == std::vector<int64_t>{256, 128}, "wo_a scale");
    if (e.name == "layers.3.attn.wo_b.scale") require(e.shape == std::vector<int64_t>{160, 256}, "wo_b scale");
    if (e.name == "layers.3.ffn.shared_experts.w2.scale") require(e.shape == std::vector<int64_t>{160, 72}, "shared w2 scale");
    if (e.name == "layers.14.engram.embed.weight")
      require(e.dtype == dgpp::DType::F8_E4M3 && e.shape == std::vector<int64_t>{384016682, 256}, "table");
    if (e.name == "layers.14.engram.embed.scale") require(e.shape == std::vector<int64_t>{384016682, 8}, "table scale");
    if (e.name == "layers.1.engram.wkv.weight") require(e.shape == std::vector<int64_t>{25600, 6144}, "engram wkv");
    if (e.name == "layers.1.engram.wkv.scale") require(e.shape == std::vector<int64_t>{800, 192}, "engram wkv scale");
    if (e.name == "layers.3.hc_attn_fn") require(e.dtype == dgpp::DType::F32 && e.shape == std::vector<int64_t>{24, 20480}, "hc fn");
    if (e.name == "layers.2.attn.indexer.wq_b.scale") require(e.shape == std::vector<int64_t>{128, 40}, "indexer scale");
    if (e.name == "mtp.0.main_proj.scale") require(e.shape == std::vector<int64_t>{160, 480}, "main_proj scale");
    if (e.name == "mtp.2.confidence_head.proj.weight") require(e.shape == std::vector<int64_t>{1, 5376}, "confidence");
    if (e.name == "mtp.2.markov_head.head.weight") require(e.shape == std::vector<int64_t>{129280, 256}, "markov");
    if (e.name == "aligner.w1.weight") require(e.shape == std::vector<int64_t>{5120, 9216} && e.skipped(), "aligner");
  }
}

DGPP_TEST(dsv41_tp_geometry_accepts_the_deployment_worlds) {
  const dgpp::Dsv41TextConfig cfg = release_config();
  for (const int w : {1, 2, 4, 8})
    for (int r = 0; r < w; ++r) dgpp::dsv41_tp_validate_geometry(cfg, r, w);
  auto refused = [&](int world) {
    try {
      dgpp::dsv41_tp_validate_geometry(cfg, 0, world);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  require(refused(3), "a world that does not divide the groups is refused");
  require(refused(16), "a world wider than the output groups is refused");
}

DGPP_TEST(dsv41_binding_matches_the_landed_checkpoint) {
  const auto snap = landed_snapshot();
  if (snap.empty())
    dgpp::test::skip("required checkpoint snapshot unavailable: deepseek-ai/DeepSeek-V4.1-Flash");
  namespace fs = std::filesystem;
  const dgpp::Dsv41TextConfig cfg = dgpp::Dsv41TextConfig::from_json_file((snap / "config.json").string());
  std::unordered_map<std::string, dgpp::Dsv41TensorDesc> present;
  for (const auto& entry : fs::directory_iterator(snap)) {
    if (entry.path().extension() != ".safetensors") continue;
    auto f = dgpp::SafetensorsFile::open(entry.path().string());
    f->for_each([&](const dgpp::TensorInfo& t) { present.emplace(t.name, dgpp::Dsv41TensorDesc{t.dtype, t.shape}); });
  }
  const dgpp::Dsv41BindReport rep = dgpp::dsv41_validate_text_binding(cfg, present);
  std::string errs;
  for (const auto& e : rep.errors) errs += "\n  " + e;
  require(rep.ok(), "binding: missing " + std::to_string(rep.missing) + ", dtype " + std::to_string(rep.dtype_mismatch) +
                        ", shape " + std::to_string(rep.shape_mismatch) + ", unexpected " + std::to_string(rep.unexpected) + errs);
  require(rep.expected == kTotalTensors && rep.matched == kTotalTensors, "every tensor bound");
  require(rep.fp4_matrices == (40 * 384 + 3 * 128) * 3, "fp4 matrices");
  require(rep.engram_tables == 2 && rep.skipped == 266, "tables / vision");
}
