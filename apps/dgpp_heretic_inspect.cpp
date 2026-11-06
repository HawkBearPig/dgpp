// Read-only Heretic target discovery. No model weights, GPU or fabric startup.
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "heretic_metadata.hpp"

int main(int argc, char** argv) {
  try {
    std::string config_path;
    int rank = 0;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help") {
        std::puts(
            "dgpp-heretic-inspect --config RESOLVED_CLUSTER_JSON [--rank N]\n"
            "Requires engine.heretic_research=true. Reports metadata only; no inference.");
        return 0;
      }
      if ((arg != "--config" && arg != "--rank") || i + 1 >= argc)
        throw std::invalid_argument("expected --config FILE or --rank N");
      const std::string value = argv[++i];
      if (arg == "--config") {
        config_path = value;
      } else {
        const auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), rank);
        if (ec != std::errc{} || end != value.data() + value.size())
          throw std::invalid_argument("--rank must be an integer");
      }
    }
    if (config_path.empty()) throw std::invalid_argument("--config is required");
    const auto cluster = dgpp::serve::load_cluster_config(config_path);
    if (!cluster.engine.heretic_research)
      throw std::runtime_error("Heretic research is disabled; set engine.heretic_research=true");
    const int world = static_cast<int>(cluster.nodes.size());
    if (rank < 0 || rank >= world) throw std::invalid_argument("--rank is outside this cluster");
    const auto checkpoint = dgpp::heretic::checkpoint_path(cluster, rank);
    const auto cfg = dgpp::GlmDsaTextConfig::from_json_file(
        (std::filesystem::path(checkpoint) / "config.json").string());
    const auto out = dgpp::heretic::catalog_json(cluster, checkpoint, cfg, rank);
    std::fputs(out.c_str(), stdout);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "dgpp-heretic-inspect: %s\n", e.what());
    return 1;
  }
}
