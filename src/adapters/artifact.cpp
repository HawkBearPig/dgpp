#include "adapters/artifact.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <tuple>

#include "adapters/environment.hpp"
#include "adapters/sha256.hpp"
#include "loaders/minijson.hpp"
#include "loaders/safetensors.hpp"
#include "serve/json_out.hpp"

namespace dgpp::adapters {
namespace {
using minijson::Value;
void require(bool ok, const std::string& why) {
  if (!ok) throw std::runtime_error("runtime adapter: " + why);
}
std::string str(const Value& value) {
  require(value.is_string(), "expected a string");
  return std::string(value.as_string());
}
int64_t integer(const Value& value) {
  require(value.kind() == Value::Kind::Int, "expected an integer");
  return value.as_int();
}
void fields(const Value& value, const std::set<std::string>& allowed) {
  require(value.is_object(), "expected an object");
  std::set<std::string> seen;
  for (const auto& member : value.members())
    require(allowed.count(member.key) && seen.insert(member.key).second,
            "unknown or duplicate field: " + member.key);
}
std::string quote(const std::string& value) {
  std::string out;
  serve::append_json_string(&out, value);
  return out;
}
std::string read_text(const std::string& path, size_t limit) {
  require(std::filesystem::file_size(path) <= limit, "file exceeds budget: " + path);
  std::ifstream file(path, std::ios::binary);
  require(static_cast<bool>(file), "cannot read " + path);
  std::string bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  require(!file.bad() && bytes.size() <= limit, "cannot finish reading " + path);
  return bytes;
}
bool hex_digest(const std::string& value) {
  return value.size() == 64 && value.find_first_not_of("0123456789abcdef") == std::string::npos;
}
void filename(const std::string& name) {
  require(!name.empty() && name != "." && name != ".." &&
              std::filesystem::path(name).filename().string() == name,
          "checkpoint references must be simple filenames");
}
void append_file_identity(Sha256& combined, const std::string& name, uint64_t size,
                          const std::string& digest) {
  combined.update(name);
  combined.update(std::string_view("\0", 1));
  combined.update(std::to_string(size));
  combined.update(std::string_view("\0", 1));
  combined.update(digest);
  combined.update("\n");
}
}  // namespace

std::string CheckpointIdentity::json() const {
  require(weight_contents_verified, "metadata inspection cannot be exported as a fingerprint");
  std::string out =
      "{\"sha256\":" + quote(sha256) + ",\"revision\":" + quote(revision) + ",\"files\":[";
  for (size_t i = 0; i < files.size(); ++i) {
    if (i) out += ',';
    out += "{\"name\":" + quote(files[i].name) + ",\"sha256\":" + quote(files[i].sha256) +
           ",\"size\":" + std::to_string(files[i].size) + "}";
  }
  return out + "]}";
}

void CheckpointIdentity::check_unchanged(const std::string& checkpoint) const {
  for (const auto& f : files) {
    const auto path = std::filesystem::path(checkpoint) / f.name;
    require(std::filesystem::file_size(path) == f.size &&
                std::filesystem::last_write_time(path) == f.modified,
            "checkpoint changed after inspection: " + f.name);
  }
}

namespace {
CheckpointIdentity inspect_checkpoint(
    const std::string& checkpoint, bool verify_weights,
    const std::function<void(const std::string&, uint64_t)>& on_weight_file = {}) {
  namespace fs = std::filesystem;
  std::set<std::string> names{"config.json"};
  for (const auto* name : {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                           "chat_template.jinja", "generation_config.json"})
    if (fs::exists(fs::path(checkpoint) / name)) names.insert(name);
  const auto config_bytes = read_text(checkpoint + "/config.json", 8 * 1024 * 1024);
  const auto config = minijson::parse_document(config_bytes);
  if (const auto* quant = config.root.find("quantization_config"); quant && quant->is_object()) {
    const auto* method = quant->find("quant_method");
    if (method && str(*method) == "dgpp_mixed346") {
      for (const auto& [block, key, fallback] :
           {std::tuple{"baseline", "baseline_config_file", "baseline-quantization-config.json"},
            std::tuple{"recipe", "recipe_file", "quantization-recipe.json"}}) {
        const auto* inline_value = quant->find(block);
        if (inline_value && !inline_value->is_null()) continue;
        const auto* named = quant->find(key);
        names.insert(named ? str(*named) : fallback);
      }
    }
  }
  std::set<std::string> weights;
  if (fs::exists(checkpoint + "/model.safetensors.index.json")) {
    names.insert("model.safetensors.index.json");
    const auto bytes = read_text(checkpoint + "/model.safetensors.index.json", 64 * 1024 * 1024);
    const auto parsed = minijson::parse_document(bytes);
    const auto& map = parsed.root.at("weight_map");
    require(map.is_object() && !map.members().empty(), "empty checkpoint weight map");
    for (const auto& entry : map.members()) weights.insert(str(entry.value));
  } else {
    weights.insert("model.safetensors");
  }
  for (const auto& name : weights)
    require(names.insert(name).second, "weight file overlaps checkpoint metadata: " + name);
  require(names.size() <= 512, "too many checkpoint files");
  CheckpointIdentity identity;
  identity.weight_contents_verified = verify_weights;
  const auto revision = fs::path(checkpoint).filename().string();
  if (revision.size() == 40 && revision.find_first_not_of("0123456789abcdef") == std::string::npos)
    identity.revision = revision;
  Sha256 combined;
  for (const auto& name : names) {
    filename(name);
    const auto path = (fs::path(checkpoint) / name).string();
    const auto size = fs::file_size(path);
    const auto modified = fs::last_write_time(path);
    const bool weight = weights.count(name) != 0;
    if (weight && verify_weights && on_weight_file) on_weight_file(name, size);
    const auto digest = weight && !verify_weights ? std::string{} : file_sha256(path);
    require(size == fs::file_size(path) && modified == fs::last_write_time(path),
            "checkpoint changed while inspecting " + name);
    identity.files.push_back({name, digest, size, modified});
    if (verify_weights) append_file_identity(combined, name, size, digest);
  }
  if (verify_weights) identity.sha256 = combined.finish();
  return identity;
}
}  // namespace

CheckpointIdentity inspect_checkpoint_metadata(const std::string& checkpoint) {
  return inspect_checkpoint(checkpoint, false);
}

CheckpointIdentity fingerprint_checkpoint(
    const std::string& checkpoint,
    const std::function<void(const std::string&, uint64_t)>& on_weight_file) {
  return inspect_checkpoint(checkpoint, true, on_weight_file);
}

Manifest parse_manifest(const std::string& bytes, const GlmDsaTextConfig& config,
                        const CheckpointIdentity& base, int world) {
  require(!bytes.empty() && bytes.size() <= kMaxManifestBytes, "manifest exceeds budget");
  const auto parsed = minijson::parse_document(bytes);
  const auto& root = parsed.root;
  fields(root, {"schema", "version", "operation", "runtime", "world_size", "base", "projections",
                "provenance", "environment_sha256"});
  require(str(root.at("schema")) == "dgpp.runtime_adapter" && integer(root.at("version")) == 1,
          "unsupported manifest schema/version");
  require(str(root.at("operation")) == "additive_low_rank" &&
              (str(root.at("runtime")) == "glm_dsa_fp32_v1" ||
               str(root.at("runtime")) == "glm_dsa_eager_fp32_v1"),
          "unsupported execution contract");
  require(str(root.at("environment_sha256")) == sha256(numerical_environment()),
          "numerical environment differs from the fitted adapter");
  require(integer(root.at("world_size")) == world && (world == 1 || world == 2 || world == 4),
          "world size differs from the fitted adapter");
  const auto& declared = root.at("base");
  fields(declared, {"sha256", "revision", "files"});
  const auto declared_sha256 = str(declared.at("sha256"));
  require(hex_digest(declared_sha256), "invalid base SHA-256");
  require(str(declared.at("revision")) == base.revision, "base checkpoint revision mismatch");
  if (base.weight_contents_verified)
    require(declared_sha256 == base.sha256, "base checkpoint fingerprint mismatch");
  const auto& files = declared.at("files");
  require(files.is_array() && files.items().size() == base.files.size(), "base file set mismatch");
  Sha256 declared_combined;
  for (size_t i = 0; i < base.files.size(); ++i) {
    const auto& file = files.items()[i];
    fields(file, {"name", "size", "sha256"});
    const auto digest = str(file.at("sha256"));
    require(hex_digest(digest), "invalid base file SHA-256");
    require(str(file.at("name")) == base.files[i].name && integer(file.at("size")) >= 0 &&
                static_cast<uint64_t>(integer(file.at("size"))) == base.files[i].size &&
                ((!base.weight_contents_verified && base.files[i].sha256.empty()) ||
                 digest == base.files[i].sha256),
            "base file identity mismatch: " + base.files[i].name);
    append_file_identity(declared_combined, base.files[i].name, base.files[i].size, digest);
  }
  require(declared_combined.finish() == declared_sha256, "inconsistent base fingerprint");
  Manifest result{sha256(bytes), bytes, declared_sha256, world, {}};
  const auto targets = glm_dsa_modification_targets(config, 0, world);
  const auto& projections = root.at("projections");
  require(projections.is_array() && projections.items().size() <= targets.size(),
          "invalid target count");
  size_t budget = 0, next_target = 0;
  std::set<std::string> tensor_names;
  for (const auto& item : projections.items()) {
    fields(item, {"id", "module", "input_features", "output_features", "rank", "a_name", "b_name",
                  "a_sha256", "b_sha256"});
    const auto id = str(item.at("id"));
    while (next_target < targets.size() && targets[next_target].id != id) ++next_target;
    require(next_target < targets.size(), "unknown, duplicate or unordered target: " + id);
    const auto& target = targets[next_target++];
    require(str(item.at("module")) == target.module &&
                integer(item.at("input_features")) == target.input_features &&
                integer(item.at("output_features")) == target.output_features,
            "projection shape/module mismatch");
    const auto rank = integer(item.at("rank"));
    require(rank > 0 && rank <= kMaxAdapterRank, "rank must be in [1,256]");
    ProjectionFactors p;
    p.target = target;
    p.rank = static_cast<int>(rank);
    p.a_name = str(item.at("a_name"));
    p.b_name = str(item.at("b_name"));
    require(p.a_name == "base_model.model." + target.module + ".lora_A.weight" &&
                p.b_name == "base_model.model." + target.module + ".lora_B.weight",
            "unexpected factor tensor name");
    require(tensor_names.insert(p.a_name).second && tensor_names.insert(p.b_name).second,
            "duplicate tensor name");
    p.a_sha256 = str(item.at("a_sha256"));
    p.b_sha256 = str(item.at("b_sha256"));
    require(hex_digest(p.a_sha256) && hex_digest(p.b_sha256), "invalid factor SHA-256");
    const size_t bytes = rank * (target.input_features + target.output_features) * sizeof(float);
    require(bytes <= kMaxAdapterBytes - budget, "factors exceed the 1 GiB adapter budget");
    budget += bytes;
    result.projections.push_back(std::move(p));
  }
  return result;
}

void set_factors(ProjectionFactors& p, const void* a, size_t a_bytes, const void* b,
                 size_t b_bytes) {
  require(a_bytes == p.rank * p.target.input_features * sizeof(float) &&
              b_bytes == p.target.output_features * p.rank * sizeof(float),
          "factor byte count mismatch");
  require(sha256(a, a_bytes) == p.a_sha256 && sha256(b, b_bytes) == p.b_sha256,
          "factor SHA-256 mismatch");
  require(p.a.empty() && p.b.empty(), "factor already staged");
  p.a.resize(a_bytes / sizeof(float));
  p.b.resize(b_bytes / sizeof(float));
  std::memcpy(p.a.data(), a, a_bytes);
  std::memcpy(p.b.data(), b, b_bytes);
  for (const auto* factor : {&p.a, &p.b})
    for (float v : *factor) require(std::isfinite(v), "non-finite factor");
}

Manifest load_bundle(const std::string& directory, const GlmDsaTextConfig& config,
                     const CheckpointIdentity& base, int world) {
  auto manifest = parse_manifest(read_text(directory + "/dgpp_adapter.json", kMaxManifestBytes),
                                 config, base, world);
  const auto file = directory + "/adapter_model.safetensors";
  require(std::filesystem::file_size(file) <= kMaxAdapterBytes + kMaxManifestBytes,
          "tensor file exceeds budget");
  const auto tensors = SafetensorsFile::open(file);
  require(tensors->tensor_count() == 2 * manifest.projections.size(), "unexpected tensor set");
  for (auto& p : manifest.projections) {
    const auto& a = tensors->at(p.a_name);
    const auto& b = tensors->at(p.b_name);
    require(a.dtype == DType::F32 && b.dtype == DType::F32 &&
                a.shape == std::vector<int64_t>{p.rank, p.target.input_features} &&
                b.shape == std::vector<int64_t>{p.target.output_features, p.rank},
            "factor tensor dtype/shape mismatch");
    set_factors(p, a.data, a.nbytes(), b.data, b.nbytes());
  }
  return manifest;
}
}  // namespace dgpp::adapters
