#include <cerrno>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>
#include <sys/inotify.h>
#include <unistd.h>

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
template <class F>
void rejected(F&& f, const char* why) {
  bool refused = false;
  try {
    f();
  } catch (const std::runtime_error&) {
    refused = true;
  }
  require(refused, why);
}
std::string empty_manifest(const dgpp::adapters::CheckpointIdentity& base) {
  return std::format(
      R"({{"schema":"dgpp.runtime_adapter","version":1,"operation":"additive_low_rank","runtime":"glm_dsa_fp32_v1","world_size":1,"environment_sha256":"{}","base":{},"projections":[]}})",
      dgpp::adapters::sha256(dgpp::adapters::numerical_environment()), base.json());
}
struct FileWatch {
  int fd = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
  FileWatch() { require(fd >= 0, "create checkpoint file watch"); }
  ~FileWatch() { close(fd); }
  void add(const std::filesystem::path& file) {
    require(inotify_add_watch(fd, file.c_str(), IN_OPEN | IN_ACCESS) >= 0, "watch weight reads");
  }
  bool accessed() const {
    char events[4096];
    const auto count = read(fd, events, sizeof(events));
    require(count > 0 || (count < 0 && errno == EAGAIN), "read checkpoint watch events");
    return count > 0;
  }
};
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
  const auto metadata = inspect_checkpoint_metadata(dir.string());
  const auto fast = load_bundle(dir.string(), cfg, metadata, 1);
  require(
      fast.base_sha256 == base.sha256 && fast.projections[0].a == a && fast.projections[0].b == b,
      "metadata verification loads the same bundle and retains its declared fingerprint");
  auto native_manifest = manifest;
  native_manifest.replace(native_manifest.find("glm_dsa_eager_fp32_v1"),
                          std::string("glm_dsa_eager_fp32_v1").size(), "glm_dsa_fp32_v1");
  write(dir / "dgpp_adapter.json", native_manifest);
  const auto native = load_bundle(dir.string(), cfg, base, 1);
  require(native.projections[0].a == a && native.projections[0].b == b,
          "native and legacy runtime names load identical factors");
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
  rejected([&] { (void)load_bundle(dir.string(), cfg, metadata, 1); },
           "metadata verification must still reject corrupt adapter factors");
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

DGPP_TEST(runtime_adapter_metadata_never_opens_weights) {
  using namespace dgpp::adapters;
  for (bool indexed : {false, true}) {
    const dgpp::test::TempDir fixture("dgpp_runtime_metadata_reads");
    const auto dir = fixture.path();
    write(dir / "config.json", "{}");
    const std::vector<std::string> weights =
        indexed ? std::vector<std::string>{"payload.bin", "second.safetensors"}
                : std::vector<std::string>{"model.safetensors"};
    if (indexed)
      write(dir / "model.safetensors.index.json",
            R"({"weight_map":{"a":"payload.bin","b":"second.safetensors"}})");
    FileWatch watch;
    for (const auto& name : weights) {
      write(dir / name, "weights");
      watch.add(dir / name);
    }
    const auto metadata = inspect_checkpoint_metadata(dir.string());
    metadata.check_unchanged(dir.string());
    require(!watch.accessed(), "metadata inspection must not open/read any weight file");
    require(!metadata.weight_contents_verified && metadata.sha256.empty(),
            "metadata inspection does not claim a full fingerprint");
    rejected([&] { (void)metadata.json(); }, "metadata inspection cannot export a fingerprint");
    size_t progress = 0;
    const auto full = fingerprint_checkpoint(dir.string(), [&](const std::string&, uint64_t size) {
      require(size == 7, "progress reports weight file size");
      ++progress;
    });
    require(watch.accessed(), "full verification must read weights (also validates the watch)");
    require(full.weight_contents_verified && !full.sha256.empty() && progress == weights.size(),
            "full verification fingerprints every shard and reports progress");
  }
}

DGPP_TEST(runtime_adapter_metadata_validation_and_explicit_full_verification) {
  using namespace dgpp::adapters;
  const dgpp::test::TempDir fixture("dgpp_runtime_metadata_validation");
  const auto dir = fixture.path();
  const auto cfg = dgpp::GlmDsaTextConfig::from_json_file(
      DGPP_SOURCE_DIR "/tests/fixtures/heretic_glm_dsa_config.json");
  write(dir / "config.json",
        std::string(R"({"quantization_config":{"quant_method":"dgpp_mixed346"}})") + "\n");
  write(dir / "baseline-quantization-config.json", "{}\n");
  write(dir / "quantization-recipe.json", "{}\n");
  write(dir / "tokenizer.json", "{}\n");
  write(dir / "model.safetensors.index.json",
        std::string(R"({"weight_map":{"a":"payload.bin"}})") + "\n");
  write(dir / "payload.bin", "weights");
  const auto base = fingerprint_checkpoint(dir.string());
  const auto manifest = empty_manifest(base);
  const auto metadata = inspect_checkpoint_metadata(dir.string());
  (void)parse_manifest(manifest, cfg, metadata, 1);
  // Deliberately document the default's limit: same-sized weight changes need
  // explicit full verification, whereas metadata contents are always checked.
  write(dir / "payload.bin", "changed");
  (void)parse_manifest(manifest, cfg, inspect_checkpoint_metadata(dir.string()), 1);
  rejected([&] { (void)parse_manifest(manifest, cfg, fingerprint_checkpoint(dir.string()), 1); },
           "full verification detects same-sized weight changes");
  write(dir / "payload.bin", "short");
  rejected(
      [&] { (void)parse_manifest(manifest, cfg, inspect_checkpoint_metadata(dir.string()), 1); },
      "metadata verification detects truncated weights");
  rejected([&] { metadata.check_unchanged(dir.string()); },
           "metadata inspection detects subsequent checkpoint changes");
  write(dir / "payload.bin", "weights");
  for (const auto& file : base.files) {
    if (file.name == "payload.bin") continue;
    std::ifstream input(dir / file.name);
    const std::string original{std::istreambuf_iterator<char>(input),
                               std::istreambuf_iterator<char>()};
    auto changed = original;
    // Equivalent JSON with different bytes and the same size.
    changed.back() = ' ';
    write(dir / file.name, changed);
    rejected(
        [&] { (void)parse_manifest(manifest, cfg, inspect_checkpoint_metadata(dir.string()), 1); },
        "metadata verification rejects changed metadata or sidecars");
    write(dir / file.name, original);
  }
  auto revision = metadata;
  revision.revision = std::string(40, 'a');
  rejected([&] { (void)parse_manifest(manifest, cfg, revision, 1); }, "revision mismatch rejected");
  for (const auto& digest : {std::string(64, 'g'), std::string(64, 'f')}) {
    auto corrupt = manifest;
    corrupt.replace(corrupt.find(base.sha256), 64, digest);
    rejected([&] { (void)parse_manifest(corrupt, cfg, metadata, 1); },
             "invalid or inconsistent declared fingerprint rejected without weight reads");
  }
  auto corrupt_file = manifest;
  corrupt_file.replace(corrupt_file.find(sha256("weights")), 64, std::string(64, 'g'));
  rejected([&] { (void)parse_manifest(corrupt_file, cfg, metadata, 1); },
           "unverified weight digest must still be well formed");
  std::filesystem::remove(dir / "payload.bin");
  rejected([&] { (void)inspect_checkpoint_metadata(dir.string()); },
           "missing weight file rejected");
  write(dir / "model.safetensors.index.json", R"({"weight_map":{"a":"config.json"}})");
  rejected([&] { (void)inspect_checkpoint_metadata(dir.string()); },
           "index cannot bypass verification by marking metadata as weights");
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
