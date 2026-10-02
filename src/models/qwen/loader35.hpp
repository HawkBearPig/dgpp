#pragma once
// Qwen3.5-27B loader family: FP8-native weights (the checkpoint already holds
// e4m3 payload + BF16 scales — no BF16→FP8 re-encode at load), text-only,
// dense SwiGLU MLP, standard pre-norm residual
// (input_layernorm/post_attention_layernorm).
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "loaders/resident_stream.hpp"
#include "loaders/weight_build.hpp"
#include "models/quant_matrix.hpp"
#include "models/qwen/binding.hpp"
#include "models/qwen/binding35.hpp"
#include "models/qwen/config35.hpp"
#include "models/qwen/loader.hpp"

namespace dgpp {

// Dense SwiGLU MLP resident (BF16 or block-FP8 weights).
struct Qwen35DenseMlpResident {
  const uint16_t* gate = nullptr;  // BF16 [I, H]
  const uint16_t* up = nullptr;    // BF16 [I, H]
  const uint16_t* down = nullptr;  // BF16 [H, I]
  GlmQuantMatrix gate_fp8, up_fp8, down_fp8;  // dense_weights fp8
};

struct Qwen35LayerResident {
  Qwen35LayerKind kind = Qwen35LayerKind::Gdn;
  int layer = -1;  // set by Builder::build_layer (the stream checks slot.layer)
  const uint16_t* input_norm = nullptr;  // BF16 [H]
  const uint16_t* post_norm = nullptr;   // BF16 [H]
  QwenGdnResident gdn;                   // kind == Gdn
  QwenFullAttnResident full;             // kind == Full
  Qwen35DenseMlpResident mlp;            // every layer
  size_t bytes = 0;  // set by the stream (bump cursor after build)
};

struct Qwen35GlobalsResident {
  const uint16_t* embed = nullptr;       // BF16 [vocab, H]
  const uint16_t* final_norm = nullptr;  // BF16 [H]
  const uint16_t* lm_head = nullptr;     // BF16 [V/W, H] vocab shard
  int lm_vocab_begin = 0, lm_vocab_count = 0;
  // MTP draft head (BF16, replicated): fused fc [H, 2H] over
  // cat(pre_fc_norm_embedding(embed), pre_fc_norm_hidden(hidden)) plus the
  // draft layer's final norm. Null when the config has no draft layer.
  const uint16_t* mtp_fc = nullptr;                 // BF16 [H, 2H]
  const uint16_t* mtp_norm = nullptr;               // BF16 [H]
  const uint16_t* mtp_pre_fc_norm_embedding = nullptr;  // BF16 [H]
  const uint16_t* mtp_pre_fc_norm_hidden = nullptr;     // BF16 [H]
  size_t bytes = 0;  // set by the stream (globals bump cursor after build)
};

// The local TP geometry at (rank, world): every slice bound the builders use.
struct Qwen35LocalGeometry {
  int world = 1, rank = 0;
  int local_key_heads = 0, local_value_heads = 0;  // GDN
  int local_heads = 0, head_begin = 0;             // Full query heads
  int local_kv_heads = 0, kv_head_begin = 0;       // Full kv heads
  int64_t local_inter = 0;                         // MLP I/W
  int lm_vocab_begin = 0, lm_vocab_count = 0;      // lm head slice
  static Qwen35LocalGeometry from_config(const Qwen35TextConfig& cfg, int rank, int world,
                                          LoaderHeadSharding head);
};

// The family behind the shared stream (loaders/resident_stream.hpp).
struct Qwen35LoaderFamily {
  using Config = Qwen35TextConfig;
  using Expected = QwenExpectedTensor;
  using LayerResident = Qwen35LayerResident;
  using GlobalsResident = Qwen35GlobalsResident;
  using Geometry = Qwen35LocalGeometry;
  using PresentMap = std::unordered_map<std::string, QwenTensorDesc>;
  struct Builder;  // models/qwen/loader35.cpp
  static const char* who();
  // Fresh family: layout version 1 (no old images to be compatible with).
  static uint64_t loader_format();
  static int max_layer(const Config& c);
  static int main_layers(const Config& c);
  static std::vector<Expected> layer_table(const Config& c, int layer);
  static std::vector<Expected> global_table(const Config& c);
  static void validate_binding(const Config& c, const PresentMap& present);
  static void check_sources(const Config& c, const LoaderTensorMap& tensors);
  static bool digest_included(const Expected& e);
  static bool discard_after_pack(const Expected& e);
  static void build_globals(const Config& c, const Geometry& geo, const LoaderTensorMap& tensors,
                            LayerBump& bump, GlobalsResident& out, uint64_t& source_bytes,
                            uint64_t& verbatim_bytes, LoaderHeadSharding head);
  static size_t globals_bytes(const Config& c, int rank, int world, LoaderHeadSharding head);
  static size_t extra_resident_bytes(const Config& c, int rank, int world);
  static size_t min_staging_bytes() { return size_t{256} << 20; }
  static void after_restore(const Config& c, int layer, const LoaderTensorMap& tensors,
                            LayerResident& out);
};

struct Qwen35LayerStream : ResidentLayerStream<Qwen35LoaderFamily> {
  Qwen35LayerStream(const Qwen35TextConfig& cfg, const std::string& checkpoint_dir,
                    int rank = 0, int world = 1,
                    LoaderResidency residency = LoaderResidency::Streaming,
                    LoaderHeadSharding head = LoaderHeadSharding::Full,
                    bool resident_mtp = false);
  static void set_resident_image_dir(const std::string& dir);
  static const std::string& resident_image_dir();
  const std::string& image_dir() const override;
};

extern template class ResidentLayerStream<Qwen35LoaderFamily>;

}  // namespace dgpp
