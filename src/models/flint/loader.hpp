#pragma once
// Flint weight loader using ResidentLayerStream (2026-10-22,
// docs/flint_plan.md §2). The family supplies tensor bindings, TP geometry
// and the release's own formats — fp8 block-128 and NVFP4 routed experts
// per projection, every other tensor BF16 — untouched. The shared stream
// owns allocations, staging, resident images, byte accounting and digests.
//
// Placement (every slice a formula in the world size W; W divides the kv
// head count 4: worlds 1, 2, 4):
//   attention: the q / k / v projections' rows of the rank's heads
//              ([32r/W, +32/W) query heads, [4r/W, +4/W) kv heads) stacked
//              as ONE bf16 matrix [(32 + 8) x 128 / W, hidden] in [Q | K |
//              V] order (the finish kernel's one-chunk layout), packable
//              (bf12); o_proj's packed columns (32/W x 128), packable; the
//              sink's heads widened to fp32; q_norm / k_norm replicated.
//   dense MLP: gate/up rows and down columns at 6144/W (bf16, packable).
//   MoE:       the router replicated with a synthesized zero selection
//              bias; the shared expert at 1024/W (bf16, read raw by the
//              MoE tail's decode kernels: not packable); every
//              routed expert sliced on the intermediate dim at 512/W — fp8
//              gate/up rows and down columns on 128-block boundaries, NVFP4
//              rows and 16-block columns with the checkpoint's reciprocal
//              global — in the format precision_by_layer names per
//              projection (the per-layer mask).
//   globals:   embed replicated (a row gather), the final norm, lm_head
//              vocab-sharded under VocabSharded (packable).
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "loaders/resident_stream.hpp"
#include "loaders/safetensors.hpp"
#include "loaders/weight_build.hpp"
#include "models/flint/binding.hpp"
#include "models/flint/config.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

using FlintResidency = LoaderResidency;
using FlintHeadSharding = LoaderHeadSharding;
using FlintReplicatedDigest = ReplicatedDigest;

struct FlintAttnResident {
  const uint16_t* qkv = nullptr;     // BF16 [(lh + 2 lkv) x 128, hidden]: [Q | K | V] rows (packable)
  const uint16_t* o_proj = nullptr;  // BF16 [hidden, lh x 128] (packed columns, packable)
  const uint16_t* q_norm = nullptr;  // BF16 [128]
  const uint16_t* k_norm = nullptr;  // BF16 [128]
  const float* sink = nullptr;       // F32 [lh]
  int local_heads = 0;
  int head_begin = 0;
  int local_kv_heads = 0;
  int kv_head_begin = 0;
  bool swa = false;   // the sliding window applies
  bool rope = false;  // RoPE applies (false on the no-RoPE full-attention layers)
  int64_t qkv_rows() const { return static_cast<int64_t>(local_heads + 2 * local_kv_heads) * 128; }
};

// A BF16 SwiGLU MLP's slices: the dense layers' and the shared expert's.
struct FlintBf16MlpResident {
  const uint16_t* gate = nullptr;  // BF16 [I/W, hidden] (packable)
  const uint16_t* up = nullptr;    // BF16 [I/W, hidden]
  const uint16_t* down = nullptr;  // BF16 [hidden, I/W]
  int64_t local_inter = 0;
};

struct FlintMoeResident {
  const uint16_t* router = nullptr;    // BF16 [E, hidden]
  const float* router_bias = nullptr;  // F32 [E]: zeros (synthesized; the release has no bias)
  FlintBf16MlpResident shared;         // the shared expert's slices
  // Both expert tables, [E * 3] gate, up, down each; an entry is set in
  // exactly one of them as the layer's mask says (bit w: projection w is
  // NVFP4; else fp8).
  std::vector<GlmQuantMatrix> experts_fp8;
  std::vector<GlmFp4Matrix> experts_fp4;
  uint8_t fp4_mask = 0;
  int64_t local_inter = 0;  // I/W
  bool fp4_of(int which) const { return (fp4_mask >> which) & 1u; }
};

struct FlintLayerResident {
  int layer = -1;
  bool moe = false;
  bool swa = false;
  bool rope = false;
  const uint16_t* input_norm = nullptr;  // BF16 [hidden]
  const uint16_t* post_norm = nullptr;   // BF16 [hidden]
  FlintAttnResident attn;
  FlintBf16MlpResident dense;  // the dense layers
  FlintMoeResident moe_w;      // the MoE layers
  size_t bytes = 0;
};

struct FlintGlobalsResident {
  const uint16_t* embed = nullptr;       // BF16 [vocab, hidden]
  const uint16_t* final_norm = nullptr;  // BF16 [hidden]
  const uint16_t* lm_head = nullptr;     // BF16 [lm_vocab_count, hidden]
  int lm_vocab_begin = 0;
  int lm_vocab_count = 0;
  size_t bytes = 0;
};

// The local TP geometry at (rank, world): every slice bound the builders
// and the views use, in one place.
struct FlintLocalGeometry {
  int world = 1, rank = 0;
  int local_heads = 0, head_begin = 0;        // query heads
  int local_kv_heads = 0, kv_head_begin = 0;  // kv heads
  int64_t local_inter = 0;         // moe_intermediate_size / W
  int64_t local_dense_inter = 0;   // intermediate_size / W
  int64_t local_shared_inter = 0;  // shared_expert_intermediate_size / W
  int lm_vocab_begin = 0, lm_vocab_count = 0;
  static FlintLocalGeometry from_config(const FlintTextConfig& cfg, int rank, int world, FlintHeadSharding head);
};

// The family behind the shared stream (loaders/resident_stream.hpp).
struct FlintLoaderFamily {
  using Config = FlintTextConfig;
  using Expected = FlintExpectedTensor;
  using LayerResident = FlintLayerResident;
  using GlobalsResident = FlintGlobalsResident;
  using Geometry = FlintLocalGeometry;
  using PresentMap = std::unordered_map<std::string, FlintTensorDesc>;
  struct Builder;  // models/flint/loader.cpp
  static const char* who() { return "flint loader"; }
  static uint64_t loader_format() { return 1; }
  static int max_layer(const Config& c) { return c.num_hidden_layers; }
  static int main_layers(const Config& c) { return c.num_hidden_layers; }
  static std::vector<Expected> layer_table(const Config& c, int layer) { return flint_expected_layer_tensors(c, layer); }
  static std::vector<Expected> global_table(const Config& c) { return flint_expected_global_tensors(c); }
  static void validate_binding(const Config& c, const PresentMap& present);
  static void check_sources(const Config&, const LoaderTensorMap&) {}
  static bool digest_included(const Expected& e);
  static bool discard_after_pack(const Expected& e);
  static void build_globals(const Config& c, const Geometry& geo, const LoaderTensorMap& tensors, LayerBump& bump,
                            GlobalsResident& out, uint64_t& source_bytes, uint64_t& verbatim_bytes,
                            LoaderHeadSharding head);
  static size_t globals_bytes(const Config& c, int rank, int world, LoaderHeadSharding head);
  static size_t globals_side_bytes(const Config& c, int rank, int world, LoaderHeadSharding head);
  static size_t extra_resident_bytes(const Config&, int, int) { return 0; }
  static size_t min_staging_bytes() { return 0; }
  static void after_restore(const Config&, int, const LoaderTensorMap&, LayerResident&) {}
};

extern template class ResidentLayerStream<FlintLoaderFamily>;

class FlintLayerStream : public ResidentLayerStream<FlintLoaderFamily> {
 public:
  FlintLayerStream(const FlintTextConfig& cfg, const std::string& checkpoint_dir, int rank = 0, int world = 1,
                   FlintResidency residency = FlintResidency::Streaming,
                   FlintHeadSharding head = FlintHeadSharding::Full);
  ~FlintLayerStream() override = default;

  static void set_resident_image_dir(const std::string& dir);
  static const std::string& resident_image_dir();

 protected:
  const std::string& image_dir() const override { return resident_image_dir(); }
};

}  // namespace dgpp
