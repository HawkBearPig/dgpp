#pragma once
// Dory1.0-4B weight loader using ResidentLayerStream (loaders/
// resident_stream.hpp). The family supplies the expected-tensor bindings and
// the modelopt W4A16 NVFP4 builders; the shared stream owns allocations,
// staging, resident images, byte accounting and digests.
//
// Placement (docs/dory_plan.md §2): the reference refuses
// tensor_parallel_size != 1 and the release is a single 3 GiB image, so the
// family is ONE rank. DoryGeometry::from_config rejects any world but 1 and
// every loader reads whole tensors — there are no TP slices, no MTP/draft
// layer, and no load-time requantization (plan D3: the packed e2m1 codes,
// the e4m3 block-scale grid and the F32 per-tensor scale are kept exactly as
// shipped; the activations stay BF16).
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "loaders/resident_stream.hpp"
#include "loaders/safetensors.hpp"
#include "loaders/weight_build.hpp"
#include "models/dory/binding.hpp"
#include "models/dory/config.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

using DoryResidency = LoaderResidency;
using DoryReplicatedDigest = ReplicatedDigest;

// One attention layer's resident weights. The five projections are the
// checkpoint's NVFP4 triples, kept packed; the three learned nGPT vectors
// stay BF16.
struct DoryAttentionResident {
  GlmFp4Matrix q_proj;  // [q_size, hidden]
  GlmFp4Matrix k_proj;  // [kv_size, hidden]
  GlmFp4Matrix v_proj;  // [kv_size, hidden]
  GlmFp4Matrix o_proj;  // [hidden, q_size]
  GlmFp4Matrix g_proj;  // [q_size, hidden] — the attention output gate
  const uint16_t* s_gate = nullptr;      // BF16 [q_size]
  const uint16_t* sqk = nullptr;         // BF16 [q_size]
  const uint16_t* attn_alpha = nullptr;  // BF16 [hidden]
};

// One MLP layer's resident weights.
struct DoryMlpResident {
  GlmFp4Matrix gate;  // [intermediate, hidden]
  GlmFp4Matrix up;    // [intermediate, hidden]
  GlmFp4Matrix down;  // [hidden, intermediate]
  const uint16_t* suv_gate = nullptr;  // BF16 [intermediate]
  const uint16_t* suv_up = nullptr;    // BF16 [intermediate]
  const uint16_t* mlp_alpha = nullptr; // BF16 [hidden]
};

// One physical layer: exactly one of the two sub-residents is populated
// (plan D5 — attention or MLP, never both).
struct DoryLayerResident {
  int layer = -1;
  DoryLayerKind kind = DoryLayerKind::Mlp;
  DoryAttentionResident attn;  // attention (sliding or full) layers
  DoryMlpResident mlp;         // mlp layers
  size_t bytes = 0;
};

struct DoryGlobalsResident {
  const uint16_t* embed = nullptr;        // BF16 [vocab, hidden]
  const uint16_t* lm_head = nullptr;      // BF16 [vocab, hidden]
  const uint16_t* logit_scale = nullptr;  // BF16 [vocab]
  size_t bytes = 0;
};

// The local geometry: one rank, so every field is the full model.
struct DoryGeometry {
  int world = 1, rank = 0;
  int lm_vocab_begin = 0, lm_vocab_count = 0;
  static DoryGeometry from_config(const DoryConfig& cfg, int rank, int world,
                                  LoaderHeadSharding head);
};

// The family behind the shared stream (loaders/resident_stream.hpp).
struct DoryLoaderFamily {
  using Config = DoryConfig;
  using Expected = DoryExpectedTensor;
  using LayerResident = DoryLayerResident;
  using GlobalsResident = DoryGlobalsResident;
  using Geometry = DoryGeometry;
  using PresentMap = std::unordered_map<std::string, DoryTensorDesc>;
  struct Builder;  // models/dory/loader.cpp
  static const char* who() { return "dory loader"; }
  // The resident layout's version: bumped when the byte layout changes.
  static uint64_t loader_format() { return 1; }
  static int max_layer(const Config& c) { return c.num_hidden_layers; }
  static int main_layers(const Config& c) { return c.num_hidden_layers; }
  static std::vector<Expected> layer_table(const Config& c, int layer) {
    return dory_expected_layer_tensors(c, layer);
  }
  static std::vector<Expected> global_table(const Config& c) {
    return dory_expected_global_tensors(c);
  }
  static void validate_binding(const Config& c, const PresentMap& present);
  static void check_sources(const Config& c, const LoaderTensorMap& tensors);
  static bool digest_included(const Expected& e);
  static bool discard_after_pack(const Expected&) { return false; }
  static void build_globals(const Config& c, const Geometry& geo, const LoaderTensorMap& tensors,
                            LayerBump& bump, GlobalsResident& out, uint64_t& source_bytes,
                            uint64_t& verbatim_bytes, LoaderHeadSharding head);
  static size_t globals_bytes(const Config& c, int rank, int world, LoaderHeadSharding head);
  static size_t extra_resident_bytes(const Config&, int, int) { return 0; }
  static size_t min_staging_bytes() { return 0; }
  static void after_restore(const Config&, int, const LoaderTensorMap&, LayerResident&) {}
};

extern template class ResidentLayerStream<DoryLoaderFamily>;

class DoryLayerStream : public ResidentLayerStream<DoryLoaderFamily> {
 public:
  DoryLayerStream(const DoryConfig& cfg, const std::string& checkpoint_dir, int rank = 0,
                  int world = 1, DoryResidency residency = DoryResidency::Streaming,
                  LoaderHeadSharding head = LoaderHeadSharding::Full, bool resident_mtp = false);
  ~DoryLayerStream() override = default;

  static void set_resident_image_dir(const std::string& dir);
  static const std::string& resident_image_dir();

 protected:
  const std::string& image_dir() const override { return resident_image_dir(); }
};

}  // namespace dgpp
