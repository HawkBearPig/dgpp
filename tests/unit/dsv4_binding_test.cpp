// The DeepSeek-V4-Flash expected-tensor table: its shape on the 0731
// release's config (the per-layer counts the shard headers carry), the TP
// geometry acceptance at the deployment worlds, and — against the
// checkpoint's shard headers — the full binding: every expected tensor
// present with its dtype and shape, nothing unexpected. The checkpoint gate
// runs on the hub snapshot once every indexed file is present, or on the
// directory named by DGPP_DSV4_CHECKPOINT_DIR (a header-only mirror
// suffices: no payload is read).
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common/test.hpp"
#include "dsv4_config_json.hpp"
#include "loaders/minijson.hpp"
#include "loaders/safetensors.hpp"
#include "models/dsv4/binding.hpp"
#include "models/dsv4/config.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

dgpp::Dsv4Config release_config() {
  const std::string text = dsv4_test::config_json();
  const auto t = dgpp::minijson::parse(text);
  return dgpp::Dsv4Config::parse(t.root);
}

constexpr size_t kTotalTensors = 72317;

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
  if (const char* dir = std::getenv("DGPP_DSV4_CHECKPOINT_DIR"); dir && *dir) {
    const fs::path p(dir);
    return snapshot_complete(p) ? p : fs::path{};
  }
  const char* home = std::getenv("HOME");
  if (!home) return {};
  const fs::path root = fs::path(home) / ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots";
  if (!fs::is_directory(root)) return {};
  for (const auto& snap : fs::directory_iterator(root))
    if (snapshot_complete(snap.path())) return snap.path();
  return {};
}

}  // namespace

DGPP_TEST(dsv4_binding_table_has_the_release_shape) {
  const dgpp::Dsv4Config cfg = release_config();
  // The counts of the checkpoint's shard headers. A window layer: 2 norms +
  // 6 mHC + the attention (q_norm, kv_norm, sink, 5 fp8 pairs) + the gate
  // + 256 x 3 MXFP4 pairs + the shared expert's 3 fp8 pairs.
  require(dgpp::dsv4_expected_layer_tensors(cfg, 0).size() == 1565, "window layer (hashed)");
  require(dgpp::dsv4_expected_layer_tensors(cfg, 1).size() == 1565, "window layer 1");
  // Ratio 4: the compressor (wkv, wgate, ape, norm) and the indexer with
  // its own compressor; layer 2 is also the last hash layer.
  require(dgpp::dsv4_expected_layer_tensors(cfg, 2).size() == 1576, "ratio 4 (hashed)");
  require(dgpp::dsv4_expected_layer_tensors(cfg, 4).size() == 1576, "ratio 4");
  require(dgpp::dsv4_expected_layer_tensors(cfg, 42).size() == 1576, "ratio 4, last layer");
  // Ratio 128: the compressor alone.
  require(dgpp::dsv4_expected_layer_tensors(cfg, 3).size() == 1569, "ratio 128");
  require(dgpp::dsv4_expected_layer_tensors(cfg, 41).size() == 1569, "ratio 128, layer 41");
  // The draft stages: stage 0 with main_proj / main_norm, the last with the heads.
  require(dgpp::dsv4_expected_layer_tensors(cfg, 43).size() == 1568, "draft stage 0");
  require(dgpp::dsv4_expected_layer_tensors(cfg, 44).size() == 1565, "draft stage 1");
  require(dgpp::dsv4_expected_layer_tensors(cfg, 45).size() == 1572, "draft stage 2");
  require(dgpp::dsv4_layer_prefix(cfg, 3) == "layers.3." && dgpp::dsv4_layer_prefix(cfg, 44) == "mtp.1.", "prefixes");
  require(dgpp::dsv4_expected_global_tensors(cfg).size() == 6, "globals");
  const auto all = dgpp::dsv4_expected_tensors(cfg);
  require(all.size() == kTotalTensors, "table size " + std::to_string(all.size()));
  size_t fp4 = 0, fp8 = 0, i64 = 0;
  for (const auto& e : all) {
    if (e.role == dgpp::Dsv4TensorRole::Fp4Payload) ++fp4;
    if (e.role == dgpp::Dsv4TensorRole::Fp8Payload) ++fp8;
    if (e.dtype == dgpp::DType::I64) ++i64;
  }
  require(fp4 == 46 * 256 * 3, "fp4 matrices " + std::to_string(fp4));
  // fp8: 5 attention + 3 shared per layer (46), 21 indexer wq_b, 1 main_proj.
  require(fp8 == 46 * 8 + 21 + 1, "fp8 matrices " + std::to_string(fp8));
  require(i64 == 3, "one tid2eid table per hash layer");
  // Shapes pinned to the headers.
  size_t pinned = 0;
  auto pin = [&](const dgpp::Dsv4ExpectedTensor& e, const char* name, dgpp::DType dtype, std::vector<int64_t> shape) {
    if (e.name != name) return;
    require(e.dtype == dtype && e.shape == shape, std::string("shape of ") + name);
    ++pinned;
  };
  using dgpp::DType;
  for (const auto& e : all) {
    pin(e, "layers.3.ffn.experts.0.w1.weight", DType::I8, {2048, 2048});
    pin(e, "layers.3.ffn.experts.0.w1.scale", DType::F8_E8M0, {2048, 128});
    pin(e, "layers.3.ffn.experts.0.w2.weight", DType::I8, {4096, 1024});
    pin(e, "layers.3.ffn.experts.0.w2.scale", DType::F8_E8M0, {4096, 64});
    pin(e, "layers.3.attn.wq_b.weight", DType::F8_E4M3, {32768, 1024});
    pin(e, "layers.3.attn.wq_b.scale", DType::F8_E8M0, {256, 8});
    pin(e, "layers.3.attn.wo_a.weight", DType::F8_E4M3, {8192, 4096});
    pin(e, "layers.3.attn.wo_a.scale", DType::F8_E8M0, {64, 32});
    pin(e, "layers.3.attn.wo_b.scale", DType::F8_E8M0, {32, 64});
    pin(e, "layers.3.ffn.shared_experts.w2.scale", DType::F8_E8M0, {32, 16});
    pin(e, "layers.3.hc_attn_fn", DType::F32, {24, 16384});
    pin(e, "layers.2.attn.indexer.wq_b.scale", DType::F8_E8M0, {64, 8});
    pin(e, "layers.2.attn.indexer.weights_proj.weight", DType::BF16, {64, 4096});
    pin(e, "layers.2.attn.compressor.ape", DType::F32, {4, 1024});
    pin(e, "layers.3.attn.compressor.ape", DType::F32, {128, 512});
    pin(e, "layers.2.attn.indexer.compressor.ape", DType::F32, {4, 256});
    pin(e, "layers.0.ffn.gate.tid2eid", DType::I64, {129280, 6});
    pin(e, "mtp.0.main_proj.weight", DType::F8_E4M3, {4096, 12288});
    pin(e, "mtp.0.main_proj.scale", DType::F8_E8M0, {32, 96});
    pin(e, "mtp.2.confidence_head.proj.weight", DType::BF16, {1, 4352});
    pin(e, "hc_head_fn", DType::F32, {4, 16384});
  }
  require(pinned == 21, "every pinned name is in the table (" + std::to_string(pinned) + " of 21)");
}

DGPP_TEST(dsv4_tp_geometry_accepts_the_deployment_worlds) {
  const dgpp::Dsv4Config cfg = release_config();
  for (const int w : {1, 2, 4, 8})
    for (int r = 0; r < w; ++r) dgpp::dsv4_tp_validate_geometry(cfg, r, w);
  auto refused = [&](int world) {
    try {
      dgpp::dsv4_tp_validate_geometry(cfg, 0, world);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  require(refused(3), "a world that does not divide the groups is refused");
  require(refused(16), "a world wider than the output groups is refused");
}

DGPP_TEST(dsv4_binding_matches_the_landed_checkpoint) {
  const auto snap = landed_snapshot();
  if (snap.empty())
    dgpp::test::skip(
        "required checkpoint snapshot unavailable: deepseek-ai/DeepSeek-V4-Flash-0731");
  namespace fs = std::filesystem;
  const dgpp::Dsv4Config cfg = dgpp::Dsv4Config::from_json_file((snap / "config.json").string());
  std::unordered_map<std::string, dgpp::Dsv4TensorDesc> present;
  for (const auto& entry : fs::directory_iterator(snap)) {
    if (entry.path().extension() != ".safetensors") continue;
    auto f = dgpp::SafetensorsFile::open(entry.path().string());
    f->for_each([&](const dgpp::TensorInfo& t) { present.emplace(t.name, dgpp::Dsv4TensorDesc{t.dtype, t.shape}); });
  }
  const dgpp::Dsv4BindReport rep = dgpp::dsv4_validate_binding(cfg, present);
  std::string errs;
  for (const auto& e : rep.errors) errs += "\n  " + e;
  require(rep.ok(), "binding: missing " + std::to_string(rep.missing) + ", dtype " + std::to_string(rep.dtype_mismatch) +
                        ", shape " + std::to_string(rep.shape_mismatch) + ", unexpected " + std::to_string(rep.unexpected) + errs);
  require(rep.expected == kTotalTensors && rep.matched == kTotalTensors, "every tensor bound");
  require(rep.fp4_matrices == 46 * 256 * 3 && rep.fp8_matrices == 46 * 8 + 21 + 1, "fp4 / fp8 matrices");
}
