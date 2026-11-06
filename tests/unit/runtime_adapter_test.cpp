#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>

#include "../common/temp_dir.hpp"
#include "adapters/artifact.hpp"
#include "adapters/environment.hpp"
#include "adapters/sha256.hpp"
#include "common/test.hpp"

namespace {
void require(bool ok, const char* why) {
  if (!ok) throw std::runtime_error(why);
}
void write(const std::filesystem::path& path, const std::string& bytes) {
  std::ofstream(path, std::ios::binary).write(bytes.data(), bytes.size());
}
}  // namespace

DGPP_TEST(runtime_adapter_fingerprint_and_bundle_roundtrip) {
  using namespace dgpp::adapters;
  require(sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "SHA-256 standard vector");
  const dgpp::test::TempDir fixture("dgpp_runtime_adapter");
  const auto dir = fixture.path();
  std::filesystem::copy_file(DGPP_SOURCE_DIR "/tests/fixtures/heretic_glm_dsa_config.json",
                             dir / "config.json");
  write(dir / "model.safetensors", "base weights");
  const auto base = fingerprint_checkpoint(dir.string());
  const auto cfg = dgpp::GlmDsaTextConfig::from_json_file((dir / "config.json").string());
  const auto target = dgpp::glm_dsa_modification_targets(cfg, 0, 1).front();
  std::vector<float> a(3 * target.input_features, 0.5f), b(3 * target.output_features, -0.25f);
  const std::string ab(reinterpret_cast<const char*>(a.data()), a.size() * 4);
  const std::string bb(reinterpret_cast<const char*>(b.data()), b.size() * 4);
  const auto an = "base_model.model." + target.module + ".lora_A.weight";
  const auto bn = "base_model.model." + target.module + ".lora_B.weight";
  const auto manifest = std::format(
      R"({{"schema":"dgpp.runtime_adapter","version":1,"operation":"additive_low_rank","runtime":"glm_dsa_eager_fp32_v1","world_size":1,"environment_sha256":"{}","base":{},"projections":[{{"id":"{}","module":"{}","input_features":{},"output_features":{},"rank":3,"a_name":"{}","b_name":"{}","a_sha256":"{}","b_sha256":"{}"}}]}})",
      sha256(numerical_environment()), base.json(), target.id, target.module, target.input_features,
      target.output_features, an, bn, sha256(ab), sha256(bb));
  write(dir / "dgpp_adapter.json", manifest);
  auto header = std::format(
      R"({{"{}":{{"dtype":"F32","shape":[3,{}],"data_offsets":[0,{}]}},"{}":{{"dtype":"F32","shape":[{},3],"data_offsets":[{},{}]}}}})",
      an, target.input_features, ab.size(), bn, target.output_features, ab.size(),
      ab.size() + bb.size());
  header.append((8 - header.size() % 8) % 8, ' ');
  uint64_t size = header.size();
  const std::string tensors =
      std::string(reinterpret_cast<const char*>(&size), 8) + header + ab + bb;
  write(dir / "adapter_model.safetensors", tensors);
  const auto loaded = load_bundle(dir.string(), cfg, base, 1);
  require(loaded.id == sha256(manifest) && loaded.projections.size() == 1, "bundle identity");
  require(loaded.projections[0].a == a && loaded.projections[0].b == b, "bundle factors unchanged");
  auto corrupt = tensors;
  corrupt.back() ^= 1;
  write(dir / "adapter_model.safetensors", corrupt);
  bool refused = false;
  try {
    (void)load_bundle(dir.string(), cfg, base, 1);
  } catch (const std::runtime_error&) {
    refused = true;
  }
  require(refused, "corrupt factor file rejected");
  write(dir / "model.safetensors", "different base weights");
  refused = false;
  try {
    base.check_unchanged(dir.string());
  } catch (const std::runtime_error&) {
    refused = true;
  }
  require(refused, "base change after fingerprint rejected");
  require(fingerprint_checkpoint(dir.string()).sha256 != base.sha256,
          "base bytes bound to identity");
}

DGPP_TEST(runtime_adapter_fingerprint_includes_quantization_sidecars) {
  using namespace dgpp::adapters;
  const dgpp::test::TempDir fixture("dgpp_runtime_sidecars");
  const auto dir = fixture.path();
  write(dir / "config.json", R"({"quantization_config":{"quant_method":"dgpp_mixed346"}})");
  write(dir / "baseline-quantization-config.json", "{}");
  write(dir / "quantization-recipe.json", "{}");
  write(dir / "model.safetensors", "weights");
  const auto base = fingerprint_checkpoint(dir.string());
  require(base.files.size() == 4, "external quantization sidecars are fingerprinted");
  write(dir / "quantization-recipe.json", "{\"changed\":true}");
  require(fingerprint_checkpoint(dir.string()).sha256 != base.sha256,
          "recipe bytes bound to identity");
}
