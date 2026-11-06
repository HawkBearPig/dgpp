#pragma once
// Cold-path metadata shared by the separate Heretic tools.
#include <filesystem>
#include <stdexcept>
#include <string>

#include "loaders/hf_cache.hpp"
#include "models/glm_dsa/modification_targets.hpp"
#include "serve/cluster_config.hpp"
#include "serve/json_out.hpp"

namespace dgpp::heretic {
inline std::string checkpoint_path(const serve::ClusterConfig& cluster, int rank) {
  // Use the same per-node cache override as the cluster launcher, without
  // exporting or reading unrelated environment values.
  std::string checkpoint = dgpp::serve::expand_home(cluster.model);
  if (!std::filesystem::is_directory(checkpoint)) {
    std::string error;
    std::string cache = dgpp::hf::default_cache_root();
    if (static_cast<size_t>(rank) < cluster.node_env.size()) {
      const auto& env = cluster.node_env[rank];
      if (const auto it = env.find("HF_HUB_CACHE"); it != env.end())
        cache = dgpp::serve::expand_home(it->second);
    }
    checkpoint = dgpp::hf::model_dir_in_root(cache, cluster.model, &error);
    if (checkpoint.empty()) throw std::runtime_error(error);
  }
  if (!std::filesystem::is_regular_file(std::filesystem::path(checkpoint) / "config.json"))
    throw std::runtime_error("model must resolve to a checkpoint snapshot containing config.json");
  return checkpoint;
}
inline std::string catalog_json(const serve::ClusterConfig& cluster, const std::string& checkpoint,
                                const GlmDsaTextConfig& cfg, int rank) {
  const int world = static_cast<int>(cluster.nodes.size());
  const auto targets = dgpp::glm_dsa_modification_targets(cfg, rank, world);
  std::string out =
      "{\"schema\":\"dgpp.heretic.targets\",\"version\":1,"
      "\"capabilities\":[\"describe_targets\"],\"model\":";
  dgpp::serve::append_json_string(&out, cluster.model);
  out += ",\"checkpoint\":";
  dgpp::serve::append_json_string(&out, checkpoint);
  out += ",\"rank\":" + std::to_string(rank) + ",\"world\":" + std::to_string(world) +
         ",\"layer_count\":" + std::to_string(cfg.num_hidden_layers) + ",\"targets\":[";
  for (size_t i = 0; i < targets.size(); ++i) {
    const auto& t = targets[i];
    if (i) out += ',';
    out += "{\"id\":";
    dgpp::serve::append_json_string(&out, t.id);
    out += ",\"module\":";
    dgpp::serve::append_json_string(&out, t.module);
    out += ",\"component\":";
    dgpp::serve::append_json_string(&out, t.component);
    out += ",\"layer_index\":" + std::to_string(t.layer) +
           ",\"module_index\":" + std::to_string(t.module_index) +
           ",\"input_features\":" + std::to_string(t.input_features) +
           ",\"output_features\":" + std::to_string(t.output_features) +
           ",\"input_begin\":" + std::to_string(t.input_begin) +
           ",\"input_end\":" + std::to_string(t.input_end) +
           ",\"weight_bits\":" + std::to_string(t.weight_bits) +
           ",\"group_size\":" + std::to_string(t.group_size) + "}";
  }
  out += "]}\n";
  return out;
}
}  // namespace dgpp::heretic
