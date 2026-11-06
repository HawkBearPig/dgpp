#include <set>
#include <stdexcept>

#include "common/test.hpp"
#include "models/glm_dsa/modification_targets.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}  // namespace

DGPP_TEST(glm_dsa_modification_targets_match_heretic_main_stack) {
  const dgpp::GlmDsaTextConfig config;
  const auto targets = dgpp::glm_dsa_modification_targets(config, 0, 1);
  require(targets.size() == 156, "78 attention and 78 dense/shared down projections");
  std::set<std::string> names, ids;
  int dense = 0, shared = 0;
  for (const auto& t : targets) {
    require(names.insert(t.module).second && ids.insert(t.id).second, "unique targets");
    require(t.output_features == 6144 && t.module_index == 0, "output geometry/capture index");
    require(t.input_begin == 0 && t.input_end == t.input_features, "world-one slice");
    require(t.id == std::to_string(t.layer) + "/" + t.component + "/0", "capture key");
    if (t.component == "attn.o_proj") {
      require(t.input_features == 16384, "attention takes all value-head outputs");
      require(t.weight_bits == (t.layer < 3 ? 0 : 8), "attention checkpoint format");
    } else if (t.layer < 3) {
      ++dense;
      require(t.input_features == 12288 && t.weight_bits == 0, "dense geometry/format");
    } else {
      ++shared;
      require(t.input_features == 2048 && t.weight_bits == 8, "shared geometry/format");
      require(
          t.module == "model.layers." + std::to_string(t.layer) + ".mlp.shared_experts.down_proj",
          "only the shared expert is individually modifiable");
    }
    require(t.group_size == (t.weight_bits ? 64 : 0), "packed group descriptor");
  }
  require(dense == 3 && shared == 75, "dense/shared split excludes draft and routed experts");
}

DGPP_TEST(glm_dsa_modification_targets_partition_columns_without_gaps) {
  const dgpp::GlmDsaTextConfig config;
  const auto global = dgpp::glm_dsa_modification_targets(config, 0, 1);
  for (int world : {1, 2, 4}) {
    std::vector<int64_t> ends(global.size(), 0);
    for (int rank = 0; rank < world; ++rank) {
      const auto local = dgpp::glm_dsa_modification_targets(config, rank, world);
      for (size_t i = 0; i < local.size(); ++i) {
        require(local[i].id == global[i].id && local[i].module == global[i].module,
                "every rank uses the same global ordering");
        require(local[i].input_begin == ends[i], "contiguous nonoverlapping input slices");
        require(local[i].input_end - local[i].input_begin == global[i].input_features / world,
                "A factors follow the weight's input partition");
        ends[i] = local[i].input_end;
      }
    }
    for (size_t i = 0; i < ends.size(); ++i)
      require(ends[i] == global[i].input_features, "every global input column covered once");
  }
}

DGPP_TEST(glm_dsa_modification_targets_reject_invalid_geometry) {
  const auto refused = [](const dgpp::GlmDsaTextConfig& c, int rank, int world) {
    try {
      (void)dgpp::glm_dsa_modification_targets(c, rank, world);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  dgpp::GlmDsaTextConfig c;
  require(refused(c, 0, 0) && refused(c, -1, 4) && refused(c, 4, 4), "rank bounds");
  require(refused(c, 0, 3), "indivisible tensor parallel world");
  c.n_shared_experts = 0;
  require(refused(c, 0, 1), "missing shared expert cannot silently change the target map");
}
