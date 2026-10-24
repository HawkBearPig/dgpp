#include "loaders/shardspec.hpp"

#include <cstdio>
#include <filesystem>

#include "../common/temp_dir.hpp"
#include "common/test.hpp"

namespace fs = std::filesystem;

DGPP_TEST(shardspec_parse_and_lookup) {
  const char* json = R"({
    "model": "synthetic",
    "num_shards": 2,
    "total_nbytes": 48,
    "shards": [
      {"file": "s0.safetensors", "size_bytes": 24},
      {"file": "s1.safetensors", "size_bytes": 24}
    ],
    "tensors": {
      "a": {"shard": 0, "dtype": "BF16", "shape": [2, 3], "role": "attn_o",
            "nbytes": 12},
      "b": {"shard": 1, "dtype": "F8_E4M3", "shape": [8], "role": "lm_head",
            "nbytes": 8}
    }
  })";
  // ShardedCheckpoint expects a file path; eager_bind=false skips mmap of
  // nonexistent shard files while exercising parsing + lookup.
  const dgpp::test::TempDir fixture("dgpp_shspec_test");
  const fs::path& dir = fixture.path();
  auto spec_path = dir / "spec.json";
  {
    FILE* f = fopen(spec_path.c_str(), "wb");
    fwrite(json, strlen(json), 1, f);
    fclose(f);
  }

  dgpp::ShardedCheckpoint::Config cfg;
  cfg.dir = dir.string();
  cfg.spec_path = spec_path.string();
  cfg.eager_bind = false;
  dgpp::ShardedCheckpoint cp(cfg);

  if (cp.model_id() != "synthetic") throw std::runtime_error("model id");
  if (cp.shard_count() != 2) throw std::runtime_error("shard count");
  const dgpp::SpecTensor* a = cp.find("a");
  if (!a || a->dtype != dgpp::DType::BF16 || a->shape.size() != 2 ||
      a->shape[1] != 3 || a->role != "attn_o")
    throw std::runtime_error("tensor a fields");
  if (a->nbytes != 12) throw std::runtime_error("tensor a nbytes");
  const dgpp::SpecTensor* b = cp.find("b");
  if (!b || b->dtype != dgpp::DType::F8_E4M3) throw std::runtime_error("b");
  if (cp.find("missing") != nullptr) throw std::runtime_error("ghost");

  bool found_a = false, found_b = false;
  for (auto& [name, st] : cp.tensors()) {
    if (name == "a") found_a = true;
    if (name == "b") found_b = true;
  }
  if (!found_a || !found_b) throw std::runtime_error("iteration");

  std::error_code ec;
  fs::remove(spec_path, ec);
}
