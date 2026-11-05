#pragma once

#include "loaders/resident_stream.hpp"
#include "loaders/weight_build.hpp"
#include "models/minimax/binding.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

struct MinimaxAttnResident {
  GlmQuantMatrix q, k, v, o, index_q, index_k;
  const uint16_t *q_norm = nullptr, *k_norm = nullptr;
  const uint16_t *index_q_norm = nullptr, *index_k_norm = nullptr;
  int local_heads = 0, local_kv_heads = 0, local_index_heads = 0;
};
struct MinimaxMlpResident {
  GlmQuantMatrix gate, up, down;
  int64_t local_inter = 0;
};
struct MinimaxMoeResident {
  const float *router = nullptr, *router_bias = nullptr;
  MinimaxMlpResident shared;
  std::vector<GlmFp4Matrix> experts;  // [expert][gate, up, down]
  // Preserve the published activation globals in the resident image. The
  // W4A16 path does not apply an activation quantizer; native dynamic NVFP4
  // computes its own scales (the upstream implementation neutralizes these).
  std::vector<const float*> input_globals;
  int64_t local_inter = 0;
};
struct MinimaxLayerResident {
  int layer = -1;
  bool moe = false;
  const uint16_t *input_norm = nullptr, *post_norm = nullptr;
  MinimaxAttnResident attn;
  MinimaxMlpResident dense;
  MinimaxMoeResident moe_w;
  size_t bytes = 0;
};
struct MinimaxGlobalsResident {
  const uint16_t *embed = nullptr, *final_norm = nullptr, *lm_head = nullptr;
  int lm_vocab_begin = 0, lm_vocab_count = 0;
  size_t bytes = 0;
};
struct MinimaxLocalGeometry {
  int world = 1, rank = 0;
  int local_heads = 0, head_begin = 0, local_kv_heads = 0, kv_head_begin = 0;
  int local_index_heads = 0, index_head_begin = 0;
  int64_t local_inter = 0, local_dense_inter = 0, local_shared_inter = 0;
  int lm_vocab_begin = 0, lm_vocab_count = 0;
  static MinimaxLocalGeometry from_config(const MinimaxConfig&, int rank, int world,
                                          LoaderHeadSharding);
};
struct MinimaxLoaderFamily {
  using Config = MinimaxConfig;
  using Expected = MinimaxExpectedTensor;
  using LayerResident = MinimaxLayerResident;
  using GlobalsResident = MinimaxGlobalsResident;
  using Geometry = MinimaxLocalGeometry;
  using PresentMap = std::unordered_map<std::string, MinimaxTensorDesc>;
  struct Builder;
  static const char* who() { return "minimax loader"; }
  static uint64_t loader_format() { return 1; }
  static int max_layer(const Config& c) { return c.num_hidden_layers; }
  static int main_layers(const Config& c) { return c.num_hidden_layers; }
  static std::vector<Expected> layer_table(const Config& c, int l) {
    return minimax_expected_layer_tensors(c, l);
  }
  static std::vector<Expected> global_table(const Config& c) {
    return minimax_expected_global_tensors(c);
  }
  static void validate_binding(const Config&, const PresentMap&);
  static void check_sources(const Config&, const LoaderTensorMap&) {}
  static bool digest_included(const Expected&);
  static bool discard_after_pack(const Expected&) { return false; }
  static void build_globals(const Config&, const Geometry&, const LoaderTensorMap&, LayerBump&,
                            GlobalsResident&, uint64_t& source_bytes, uint64_t& verbatim_bytes,
                            LoaderHeadSharding);
  static size_t globals_bytes(const Config&, int rank, int world, LoaderHeadSharding);
  static size_t globals_side_bytes(const Config&, int rank, int world, LoaderHeadSharding);
  static size_t extra_resident_bytes(const Config&, int, int) { return 0; }
  static size_t min_staging_bytes() { return 0; }
  static void after_restore(const Config&, int, const LoaderTensorMap&, LayerResident&) {}
};
extern template class ResidentLayerStream<MinimaxLoaderFamily>;
class MinimaxLayerStream : public ResidentLayerStream<MinimaxLoaderFamily> {
 public:
  MinimaxLayerStream(const MinimaxConfig&, const std::string& checkpoint_dir, int rank = 0,
                     int world = 1, LoaderResidency = LoaderResidency::Streaming,
                     LoaderHeadSharding = LoaderHeadSharding::Full);
  static void set_resident_image_dir(const std::string&);
  static const std::string& resident_image_dir();

 protected:
  const std::string& image_dir() const override { return resident_image_dir(); }
};
}  // namespace dgpp
