#pragma once
// The DFlash2 drafter for the Qwen3.5-27B family (docs/
// performance_improvement_plan.md §7): a block-diffusion draft model from a
// SEPARATE checkpoint (z-lab/Qwen3.8-27B-DFlash2) — five bidirectional
// sliding-window Qwen3 layers over a KV context built from fused target
// features (the tap layers' residual streams concatenated through fc),
// two-tap dynamic grouped convolutions, and a codebook path selector over
// the mask rows' top-K candidates. The embeddings and lm head are the
// target's (the draft checkpoint ships neither); the draft planes live in
// the model's KV pool so block tables, prefix sharing and rollback ride
// the existing protocol.
//
// v1 scope: eager greedy C1 (the plan's implementation sequence); graph
// capture, batching and sampled verification stay open.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "loaders/minijson.hpp"
#include "models/qwen/config35.hpp"

namespace dgpp {

struct DFlash2Config {
  // dflash_config.
  int block_size = 8;             // query rows per draft: 1 anchor + (block_size - 1) mask rows
  int mask_token_id = -1;
  int conv_taps = 2;
  int conv_group_size = 16;
  int selector_rank = 256;
  int selector_top_k = 16;
  std::vector<int> target_layer_ids;  // into the TARGET stack (features = layer outputs)
  // Qwen3 backbone.
  int num_hidden_layers = 5;
  int hidden_size = 5120;
  int vocab_size = 248320;
  int num_attention_heads = 32;
  int num_key_value_heads = 8;
  int head_dim = 128;
  int intermediate_size = 17408;
  float rms_norm_eps = 1e-6f;
  double rope_theta = 1e7;
  int64_t sliding_window = 2048;  // INT64_MAX when the backbone is not sliding
  int64_t max_position_embeddings = 262144;

  static DFlash2Config from_json_file(const std::string& path);
  static DFlash2Config parse(const minijson::Value& root);

  int drafts() const { return block_size - 1; }
  int query_rows() const { return block_size; }
  int kv_row() const { return num_key_value_heads * head_dim; }
  int q_row() const { return num_attention_heads * head_dim; }
  int conv_groups() const { return hidden_size / conv_group_size; }
  // fc's input width: the taps' outputs concatenated.
  int fc_in() const { return static_cast<int>(target_layer_ids.size()) * hidden_size; }

  // Cross-checks the draft/target pair (silent-acceptance-failure hazards:
  // §7 step 2 — widths, vocab and the tap indices). Throws with a message
  // naming the field.
  void validate_against(const Qwen35TextConfig& target) const;
};

// One draft layer's device weights (BF16, replicated, arena-resident).
struct DFlash2LayerWeights {
  const uint16_t* input_norm = nullptr;   // [H]
  const uint16_t* post_norm = nullptr;    // [H]
  const uint16_t* qkv = nullptr;          // [QW + 2*KW, H] q|k|v rows stacked
  const uint16_t* o = nullptr;            // [H, QW]
  const uint16_t* gate = nullptr;         // [I, H]
  const uint16_t* up = nullptr;           // [I, H]
  const uint16_t* down = nullptr;         // [H, I]
  const uint16_t* q_norm = nullptr;       // [head_dim]
  const uint16_t* k_norm = nullptr;       // [head_dim]
  const uint16_t* attn_conv_base = nullptr;   // [2 sides][taps][H]
  const uint16_t* attn_conv_kp = nullptr;     // [2*taps*groups, H]
  const uint16_t* mlp_conv_base = nullptr;    // [2 sides][taps][H]
  const uint16_t* mlp_conv_kp = nullptr;      // [2*taps*groups, H]
};

struct DFlash2Weights {
  std::vector<DFlash2LayerWeights> layers;
  const uint16_t* fc = nullptr;        // nTaps slices [H, H] (the [H, nTaps*H] split)
  const uint16_t* hidden_norm = nullptr;  // [H]
  const uint16_t* norm = nullptr;      // [H]
  const uint16_t* pred_codebook = nullptr;  // [vocab, rank]
  const uint16_t* succ_codebook = nullptr;  // [vocab, rank]
  const uint16_t* hidden_projection = nullptr;  // [rank, H]
  void* arena = nullptr;
  size_t bytes = 0;
  ~DFlash2Weights();
  DFlash2Weights() = default;
  DFlash2Weights(const DFlash2Weights&) = delete;
  DFlash2Weights& operator=(const DFlash2Weights&) = delete;
  DFlash2Weights(DFlash2Weights&& o) noexcept
      : layers(std::move(o.layers)), fc(o.fc), hidden_norm(o.hidden_norm), norm(o.norm),
        pred_codebook(o.pred_codebook), succ_codebook(o.succ_codebook),
        hidden_projection(o.hidden_projection), arena(o.arena), bytes(o.bytes) {
    o.arena = nullptr;
    o.bytes = 0;
  }
  DFlash2Weights& operator=(DFlash2Weights&& o) noexcept {
    if (this != &o) {
      cudaFree(arena);
      layers = std::move(o.layers);
      fc = o.fc;
      hidden_norm = o.hidden_norm;
      norm = o.norm;
      pred_codebook = o.pred_codebook;
      succ_codebook = o.succ_codebook;
      hidden_projection = o.hidden_projection;
      arena = o.arena;
      bytes = o.bytes;
      o.arena = nullptr;
      o.bytes = 0;
    }
    return *this;
  }
};

// Reads the checkpoint's safetensors shards (BF16, full-vocab), validates
// every name/shape against cfg and leaves the device-resident set. The
// caller's stream orders the uploads.
DFlash2Weights load_dflash2_weights(const DFlash2Config& cfg, const std::string& dir,
                                    cudaStream_t stream);
// The arena size load_dflash2_weights allocates (the memory plan's line).
size_t dflash2_weights_bytes(const DFlash2Config& cfg);

}  // namespace dgpp
