#pragma once
// The DFlash v1 drafter for Qwen3.5-122B-A10B (docs/dflash_v1_122b_plan.md):
// a block mask-predict draft model from a SEPARATE checkpoint
// (z-lab/Qwen3.5-122B-A10B-DFlash) — six plain sliding-window Qwen3 layers
// over a KV context built from fused target features (eight tap layers'
// residual streams concatenated through fc), no convolutions, no selector:
// the block is [anchor + 15 masks] and each mask row proposes its top-1
// off the target's shared head. The embeddings and lm head are the
// target's (the draft checkpoint ships neither); the draft planes live in
// the model's KV pool so block tables, prefix sharing and rollback ride
// the existing protocol.
//
// v1 scope: config, binding, loader (this header). The block forward and
// the proposal/verify wiring are stage 2.
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "loaders/minijson.hpp"
#include "models/qwen/config35.hpp"

namespace dgpp {

struct DFlashConfig {
  // dflash_config.
  int block_size = 16;  // query rows per draft: 1 anchor + (block_size - 1) mask rows
  int mask_token_id = -1;
  std::vector<int> target_layer_ids;  // into the TARGET stack (features = layer outputs)
  // Qwen3 backbone.
  int num_hidden_layers = 6;
  int hidden_size = 3072;
  int vocab_size = 248320;
  int num_attention_heads = 32;
  int num_key_value_heads = 8;
  int head_dim = 128;
  int intermediate_size = 9216;
  float rms_norm_eps = 1e-6f;
  double rope_theta = 1e7;
  int64_t sliding_window = 4096;  // INT64_MAX when the backbone is not sliding
  int64_t max_position_embeddings = 262144;

  static DFlashConfig from_json_file(const std::string& path);
  static DFlashConfig parse(const minijson::Value& root);

  int drafts() const { return block_size - 1; }
  int query_rows() const { return block_size; }
  int kv_row() const { return num_key_value_heads * head_dim; }
  int q_row() const { return num_attention_heads * head_dim; }
  // The tensor-parallel slices: the attention heads, the kv heads and the
  // MLP rows split across `world` ranks (each must divide); the fc taps
  // and the norms stay replicated, and the o / down projections' partial
  // outputs fold at the boundary.
  bool tp_divisible(int world) const {
    return world >= 1 && num_attention_heads % world == 0 && num_key_value_heads % world == 0 &&
           intermediate_size % world == 0;
  }
  int local_heads(int world) const { return num_attention_heads / world; }
  int local_kv_heads(int world) const { return num_key_value_heads / world; }
  int local_q_row(int world) const { return local_heads(world) * head_dim; }
  int local_kv_row(int world) const { return local_kv_heads(world) * head_dim; }
  int local_intermediate(int world) const { return intermediate_size / world; }
  // fc's input width: the taps' outputs concatenated.
  int fc_in() const { return static_cast<int>(target_layer_ids.size()) * hidden_size; }

  // Cross-checks the draft/target pair (widths, vocab, mask id, tap
  // indices, sliding window vs block). Throws naming the field.
  void validate_against(const Qwen35TextConfig& target) const;
};

// One draft layer's device weights (BF16, arena-resident; at world > 1 the
// q/k/v rows, o columns, gate/up rows and down columns are this rank's
// slices — QW, KW and I below are the local widths).
struct DFlashLayerWeights {
  const uint16_t* input_norm = nullptr;  // [H]
  const uint16_t* post_norm = nullptr;   // [H]
  const uint16_t* q = nullptr;           // [QW, H] (no [q|gate] stacking in v1)
  const uint16_t* k = nullptr;           // [KW, H]
  const uint16_t* v = nullptr;           // [KW, H]
  const uint16_t* o = nullptr;           // [H, QW]
  const uint16_t* gate = nullptr;        // [I, H]
  const uint16_t* up = nullptr;          // [I, H]
  const uint16_t* down = nullptr;        // [H, I]
  const uint16_t* q_norm = nullptr;      // [head_dim]
  const uint16_t* k_norm = nullptr;      // [head_dim]
};

struct DFlashWeights {
  std::vector<DFlashLayerWeights> layers;
  const uint16_t* fc = nullptr;          // nTaps slices [H, H] (the [H, nTaps*H] split)
  const uint16_t* hidden_norm = nullptr;  // [H]
  const uint16_t* norm = nullptr;        // [H]
  void* arena = nullptr;
  size_t bytes = 0;
  ~DFlashWeights();
  DFlashWeights() = default;
  DFlashWeights(const DFlashWeights&) = delete;
  DFlashWeights& operator=(const DFlashWeights&) = delete;
  DFlashWeights(DFlashWeights&& o) noexcept
      : layers(std::move(o.layers)), fc(o.fc), hidden_norm(o.hidden_norm), norm(o.norm),
        arena(o.arena), bytes(o.bytes) {
    o.arena = nullptr;
    o.bytes = 0;
  }
  DFlashWeights& operator=(DFlashWeights&& o) noexcept {
    if (this != &o) {
      cudaFree(arena);
      layers = std::move(o.layers);
      fc = o.fc;
      hidden_norm = o.hidden_norm;
      norm = o.norm;
      arena = o.arena;
      bytes = o.bytes;
      o.arena = nullptr;
      o.bytes = 0;
    }
    return *this;
  }
};

// Reads the checkpoint's safetensors shards (BF16, dense), validates every
// name/shape against cfg and leaves the device-resident set — rank `rank`
// of `world`'s slices of the sharded matrices (cfg.tp_divisible). The
// caller's stream orders the uploads.
DFlashWeights load_dflash_weights(const DFlashConfig& cfg, const std::string& dir,
                                 cudaStream_t stream, int rank = 0, int world = 1);
// The arena size load_dflash_weights allocates (the memory plan's line).
size_t dflash_weights_bytes(const DFlashConfig& cfg, int world = 1);

}  // namespace dgpp
