// dsv41_bind_check: offline validation of a DeepSeek-V4.1-Flash checkpoint
// (deepseek-ai/DeepSeek-V4.1-Flash) against the family's expected-tensor
// table (docs/deepseek_v41_flash_plan.md G1). Reads config.json and every
// safetensors header; no payload bytes are touched. Exit 0 only when the
// binding is exact: every expected tensor present with its dtype and shape,
// nothing unexpected.
//
//   dsv41_bind_check --model ORG/NAME | (--config <dir>/config.json
//                                       --checkpoint-dir <dir>) [--max-errors N]
//                    [--engram-source ORG/NAME | --engram-source-dir <dir>]
// The Engram source is the base release whose shards hold the Engram
// tables an EXL3 re-pack ships without (engine.engram_source): only the
// table tensors are taken from it, as the loader admits them.
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
#include "models/dsv41/binding.hpp"
#include "models/dsv41/config.hpp"

namespace {

int run(int argc, char** argv) {
  std::string config_path, checkpoint_dir, model_id, engram_model, engram_dir;
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
    else if (a == "--engram-source") engram_model = next();
    else if (a == "--engram-source-dir") engram_dir = next();
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
        "usage: dsv41_bind_check --model ORG/NAME | (--config <config.json> --checkpoint-dir <dir>) [--max-errors N] "
        "[--engram-source ORG/NAME | --engram-source-dir <dir>]");
  if (config_path.empty()) config_path = checkpoint_dir + "/config.json";
  if (!engram_model.empty()) {
    if (!engram_dir.empty()) throw std::runtime_error("--engram-source and --engram-source-dir are mutually exclusive");
    std::string err;
    engram_dir = dgpp::hf::model_dir(engram_model, &err);
    if (engram_dir.empty()) throw std::runtime_error(std::format("--engram-source {}: {}", engram_model, err));
    std::printf("engram source: %s -> %s\n", engram_model.c_str(), engram_dir.c_str());
  }

  const dgpp::Dsv41TextConfig cfg = dgpp::Dsv41TextConfig::from_json_file(config_path);
  std::printf("config: %d layers + %d draft stages, %d kv sources, %d index sources, candidate source %d, "
              "engram layers %zu, experts %d top-%d (draft %d top-%d), vocab %d, vision %s\n",
              cfg.num_hidden_layers, cfg.num_nextn_predict_layers, cfg.num_kv_sources(), cfg.num_index_sources(),
              cfg.candidate_source_layer_id, cfg.engram_layer_ids.size(), cfg.n_routed_experts,
              cfg.num_experts_per_tok, cfg.dspark_n_routed_experts, cfg.dspark_num_experts_per_tok, cfg.vocab_size,
              cfg.vision_present ? "present (skipped)" : "absent");

  namespace fs = std::filesystem;
  std::vector<fs::path> shards;
  for (const auto& entry : fs::directory_iterator(checkpoint_dir))
    if (entry.path().extension() == ".safetensors") shards.push_back(entry.path());
  if (shards.empty()) throw std::runtime_error("no .safetensors shards in " + checkpoint_dir);
  std::sort(shards.begin(), shards.end());

  std::unordered_map<std::string, dgpp::Dsv41TensorDesc> present;
  size_t total_tensors = 0;
  for (const auto& shard : shards) {
    auto f = dgpp::SafetensorsFile::open(shard.string());
    f->for_each([&](const dgpp::TensorInfo& t) {
      ++total_tensors;
      auto [it, inserted] = present.emplace(t.name, dgpp::Dsv41TensorDesc{t.dtype, t.shape});
      if (!inserted) throw std::runtime_error(std::format("duplicate tensor '{}' in {}", t.name, shard.string()));
    });
  }
  std::printf("checkpoint: %zu shards, %zu tensors in headers\n", shards.size(), total_tensors);
  if (!engram_dir.empty()) {
    // The Engram tables from the source checkpoint: the table tensors alone
    // (name ".engram.embed."), the loader's admission rule.
    std::vector<fs::path> source_shards;
    for (const auto& entry : fs::directory_iterator(engram_dir))
      if (entry.path().extension() == ".safetensors") source_shards.push_back(entry.path());
    std::sort(source_shards.begin(), source_shards.end());
    size_t admitted = 0;
    for (const auto& shard : source_shards) {
      auto f = dgpp::SafetensorsFile::open(shard.string());
      f->for_each([&](const dgpp::TensorInfo& t) {
        if (t.name.find(".engram.embed.") == std::string::npos) return;
        auto [it, inserted] = present.emplace(t.name, dgpp::Dsv41TensorDesc{t.dtype, t.shape});
        if (!inserted) throw std::runtime_error(std::format("Engram table '{}' is in both checkpoints", t.name));
        ++admitted;
      });
    }
    std::printf("engram source: %zu shards, %zu table tensors admitted\n", source_shards.size(), admitted);
  }

  const dgpp::Dsv41BindReport rep = dgpp::dsv41_validate_text_binding(cfg, present, max_errors);
  std::printf("binding: expected %zu | matched %zu (missing %zu, dtype %zu, shape %zu) | unexpected %zu | beyond stack %d\n",
              rep.expected, rep.matched, rep.missing, rep.dtype_mismatch, rep.shape_mismatch, rep.unexpected,
              rep.beyond_stack);
  std::printf("matrices: MXFP4 %zu | fp8 %zu | EXL3 %zu | Engram tables %zu | vision tensors skipped %zu\n",
              rep.fp4_matrices, rep.fp8_matrices, rep.exl3_matrices, rep.engram_tables, rep.skipped);
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
    std::fprintf(stderr, "dsv41_bind_check: %s\n", e.what());
    return 2;
  }
}
