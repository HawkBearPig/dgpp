#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "models/glm_dsa/modification_targets.hpp"

namespace dgpp::adapters {
constexpr size_t kMaxAdapterBytes = 1024u * 1024u * 1024u;
constexpr int kMaxAdapterRank = 256;
constexpr size_t kMaxManifestBytes = 1024u * 1024u;

struct CheckpointFile {
  std::string name, sha256;
  uint64_t size;
  std::filesystem::file_time_type modified;
};
struct CheckpointIdentity {
  std::string sha256, revision;
  std::vector<CheckpointFile> files;
  std::string json() const;
  void check_unchanged(const std::string& checkpoint) const;
};
// Full content hashes, including quantization sidecars and every weight shard.
// Cold path only: callers must authorize research or runtime adapters first.
CheckpointIdentity fingerprint_checkpoint(const std::string& checkpoint);

struct ProjectionFactors {
  GlmDsaModificationTarget target;
  int rank = 0;
  std::string a_name, b_name, a_sha256, b_sha256;
  std::vector<float> a, b;  // global FP32 [rank,input], [output,rank], scale 1
};
struct Manifest {
  std::string id;  // SHA-256 of the exact manifest bytes
  std::string bytes;
  std::string base_sha256;
  int world = 1;
  std::vector<ProjectionFactors> projections;
};
// Parses the execution contract. Optional provenance is not used by inference.
Manifest parse_manifest(const std::string& bytes, const GlmDsaTextConfig& config,
                        const CheckpointIdentity& base, int world);
void set_factors(ProjectionFactors& projection, const void* a, size_t a_bytes, const void* b,
                 size_t b_bytes);
Manifest load_bundle(const std::string& directory, const GlmDsaTextConfig& config,
                     const CheckpointIdentity& base, int world);
}  // namespace dgpp::adapters
