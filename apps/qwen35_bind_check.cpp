// qwen35_bind_check: offline validation of a Qwen3.5 checkpoint (dense
// Qwen3.8-27B or MoE Qwen3.5-122B-A10B-NVFP4) against the family's
// expected-tensor table. Reads config.json and every safetensors header;
// no payload bytes are touched. Exit 0 only when the binding is exact.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "loaders/hf_cache.hpp"
#include "loaders/safetensors.hpp"
#include "models/qwen/binding35.hpp"
#include "models/qwen/config35.hpp"

namespace {

int run(int argc, char** argv) {
  std::string config_path, checkpoint_dir, model_id;
  size_t max_errors = 32;
  for (int i = 1; i < argc; ++i) {
    std::string_view a = argv[i];
    auto next = [&]() -> std::string_view {
      if (i + 1 >= argc) throw std::runtime_error(std::format("missing value for {}", a));
      return argv[++i];
    };
    if (a == "--config") config_path = next();
    else if (a == "--checkpoint-dir") checkpoint_dir = next();
    else if (a == "--model") model_id = next();
    else if (a == "--max-errors") max_errors = std::stoul(std::string(next()));
    else throw std::runtime_error(std::format("unknown argument {}", a));
  }
  if (!model_id.empty()) {
    if (!checkpoint_dir.empty()) throw std::runtime_error("--model and --checkpoint-dir are mutually exclusive");
    std::string err;
    const std::string snapshot = dgpp::hf::model_dir(model_id, &err);
    if (snapshot.empty()) throw std::runtime_error(std::format("--model {}: {}", model_id, err));
    checkpoint_dir = snapshot;
    if (config_path.empty()) config_path = snapshot + "/config.json";
    std::printf("model: %s -> %s\n", model_id.c_str(), snapshot.c_str());
  }
  if (checkpoint_dir.empty())
    throw std::runtime_error(
        "usage: qwen35_bind_check --model ORG/NAME | (--config <config.json> --checkpoint-dir <dir>) [--max-errors N]");
  if (config_path.empty()) config_path = checkpoint_dir + "/config.json";

  const dgpp::Qwen35TextConfig cfg = dgpp::Qwen35TextConfig::from_json_file(config_path);
  std::printf("config: hidden %d layers %d (gdn %d full %d) vocab %d moe %s experts %d top-%d inter %d shared %d quant %d mtp %d\n",
              cfg.hidden_size, cfg.num_hidden_layers, cfg.num_gdn_layers(), cfg.num_full_layers(),
              cfg.vocab_size, cfg.is_moe ? "yes" : "no", cfg.num_experts, cfg.num_experts_per_tok,
              cfg.moe_intermediate_size, cfg.shared_expert_intermediate_size,
              static_cast<int>(cfg.quant_kind), cfg.mtp_layer());

  namespace fs = std::filesystem;
  std::vector<fs::path> shards;
  for (const auto& entry : fs::directory_iterator(checkpoint_dir))
    if (entry.path().extension() == ".safetensors") shards.push_back(entry.path());
  if (shards.empty()) throw std::runtime_error("no .safetensors shards in " + checkpoint_dir);
  std::sort(shards.begin(), shards.end());

  std::unordered_map<std::string, dgpp::QwenTensorDesc> present;
  size_t total_tensors = 0;
  for (const auto& shard : shards) {
    auto f = dgpp::SafetensorsFile::open(shard.string());
    f->for_each([&](const dgpp::TensorInfo& t) {
      ++total_tensors;
      auto [it, inserted] = present.emplace(t.name, dgpp::QwenTensorDesc{t.dtype, t.shape});
      if (!inserted) throw std::runtime_error(std::format("duplicate tensor '{}' in {}", t.name, shard.string()));
    });
  }
  std::printf("checkpoint: %zu shards, %zu tensors in headers\n", shards.size(), total_tensors);

  const dgpp::QwenBindReport rep = dgpp::qwen35_validate_text_binding(cfg, present, max_errors);
  std::printf("binding: expected %zu | matched %zu (missing %zu, dtype %zu, shape %zu) | unexpected %zu | vision %zu | out_of_scope %zu\n",
              rep.expected, rep.matched, rep.missing, rep.dtype_mismatch, rep.shape_mismatch,
              rep.unexpected, rep.vision, rep.out_of_scope);
  for (const auto& err : rep.errors) std::printf("  error: %s\n", err.c_str());
  if (rep.errors.size() >= max_errors) std::printf("  ... error list capped at %zu\n", max_errors);
  if (!rep.ok()) {
    std::printf("BINDING FAILED\n");
    return 1;
  }
  std::printf("binding OK: every tensor of the table is present with its dtype and shape\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "qwen35_bind_check: %s\n", e.what());
    return 2;
  }
}
