#pragma once
// The DFlash2 block drafter — the drafter MODEL (config, the standalone
// weight load, the target cross-checks): a block-diffusion draft model from
// a SEPARATE checkpoint (incoai/Qwen3.8-Flash-Next-DFlash2,
// z-lab/Qwen3.8-27B-DFlash2, …) — five bidirectional sliding-window Qwen3
// layers over a KV context built from fused target features (the tap
// layers' outputs concatenated through fc), two-tap dynamic grouped
// convolutions, and a codebook path selector over the mask rows' top-K
// candidates. The embeddings and lm head are the target's (the draft
// checkpoint ships neither); the draft planes live in the target's KV pool
// so block tables, prefix sharing and rollback ride the existing protocol
// (models/dflash2/draft_planes.hpp). The drafter's own backbone is always
// Qwen3 (parse enforces it) whatever the TARGET family is; the target-side
// integration (the taps, the head dispatch, the pool planes) is the host
// model's — models/dflash2/drafter.hpp carries the shared execution.
//
// v1 scope: eager greedy C1 (the plan's implementation sequence); graph
// capture, batching and sampled verification stay open.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "loaders/minijson.hpp"

namespace dgpp {

// The target geometry the drafter cross-checks against (the family adapts
// its own text config into this; the position bound is the YaRN-scaled
// context ceiling where the target ramps it).
struct DFlash2TargetShape {
  int hidden_size = 0;
  int vocab_size = 0;
  int num_hidden_layers = 0;
  int64_t context_limit = 0;
};

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
  // The tensor-parallel slices (2026-10-04): the attention heads, the kv
  // heads and the MLP rows split across `world` ranks (each must divide);
  // the convs, norms, fc taps and the selector stay replicated, and the
  // o / down projections' partial outputs fold at the boundary.
  bool tp_divisible(int world) const {
    return world >= 1 && num_attention_heads % world == 0 && num_key_value_heads % world == 0 &&
           intermediate_size % world == 0;
  }
  int local_heads(int world) const { return num_attention_heads / world; }
  int local_kv_heads(int world) const { return num_key_value_heads / world; }
  int local_q_row(int world) const { return local_heads(world) * head_dim; }
  int local_kv_row(int world) const { return local_kv_heads(world) * head_dim; }
  int local_intermediate(int world) const { return intermediate_size / world; }
  int conv_groups() const { return hidden_size / conv_group_size; }
  // fc's input width: the taps' outputs concatenated.
  int fc_in() const { return static_cast<int>(target_layer_ids.size()) * hidden_size; }

  // Cross-checks the draft/target pair (silent-acceptance-failure hazards:
  // §7 step 2 — widths, vocab and the tap indices). Throws with a message
  // naming the field.
  void validate_against(const DFlash2TargetShape& target) const;
};

// The checkpoint's first `architectures` entry ("" without the field): the
// drafter's own name decides v1 (DFlashDraftModel) vs v2
// (DFlash2DraftModel) — never a guess from the target's family.
std::string drafter_architecture(const std::string& dir);

// One draft layer's device weights (BF16, arena-resident; at world > 1 the
// q|k|v rows, o columns, gate/up rows and down columns are this rank's
// slices — QW, KW and I below are the local widths).
struct DFlash2LayerWeights {
  const uint16_t* input_norm = nullptr;   // [H]
  const uint16_t* post_norm = nullptr;    // [H]
  const uint16_t* qkv = nullptr;          // [QW + 2*KW, H] q|k|v rows stacked (null under fp8)
  const uint16_t* kv_rows = nullptr;      // [2*KW, H] the k|v rows: a view into qkv, or their own bf16 slot under fp8 (the context features' GEMM reads bf16)
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
  // engine.dflash_weights fp8 (2026-10-05): the five block matrices as
  // block-128 E4M3 payloads with fp32 scales (kernels/mma_gemv's decode
  // form); no bf16 copy of the five is kept — only the k|v rows (kv_rows),
  // which the feature path's GEMM reads.
  // Null without the recipe. Lossy for the PROPOSALS only (the target's
  // verify is exact whatever the drafter proposes).
  const uint8_t* qkv_fp8 = nullptr;   const float* qkv_scales = nullptr;
  const uint8_t* o_fp8 = nullptr;     const float* o_scales = nullptr;
  const uint8_t* gate_fp8 = nullptr;  const float* gate_scales = nullptr;
  const uint8_t* up_fp8 = nullptr;    const float* up_scales = nullptr;
  const uint8_t* down_fp8 = nullptr;  const float* down_scales = nullptr;
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
  void* fp8_arena = nullptr;  // the fp8 payloads + scales (engine.dflash_weights fp8)
  size_t fp8_bytes = 0;
  ~DFlash2Weights();
  DFlash2Weights() = default;
  DFlash2Weights(const DFlash2Weights&) = delete;
  DFlash2Weights& operator=(const DFlash2Weights&) = delete;
  DFlash2Weights(DFlash2Weights&& o) noexcept
      : layers(std::move(o.layers)), fc(o.fc), hidden_norm(o.hidden_norm), norm(o.norm),
        pred_codebook(o.pred_codebook), succ_codebook(o.succ_codebook),
        hidden_projection(o.hidden_projection), arena(o.arena), bytes(o.bytes),
        fp8_arena(o.fp8_arena), fp8_bytes(o.fp8_bytes) {
    o.arena = nullptr;
    o.bytes = 0;
    o.fp8_arena = nullptr;
    o.fp8_bytes = 0;
  }
  DFlash2Weights& operator=(DFlash2Weights&& o) noexcept {
    if (this != &o) {
      cudaFree(arena);
      cudaFree(fp8_arena);
      layers = std::move(o.layers);
      fc = o.fc;
      hidden_norm = o.hidden_norm;
      norm = o.norm;
      pred_codebook = o.pred_codebook;
      succ_codebook = o.succ_codebook;
      hidden_projection = o.hidden_projection;
      arena = o.arena;
      bytes = o.bytes;
      fp8_arena = o.fp8_arena;
      fp8_bytes = o.fp8_bytes;
      o.arena = nullptr;
      o.bytes = 0;
      o.fp8_arena = nullptr;
      o.fp8_bytes = 0;
    }
    return *this;
  }
};

// Reads the checkpoint's safetensors shards (BF16, full-vocab), validates
// every name/shape against cfg and leaves the device-resident set — rank
// `rank` of `world`'s slices of the sharded matrices (cfg.tp_divisible).
// The caller's stream orders the uploads.
DFlash2Weights load_dflash2_weights(const DFlash2Config& cfg, const std::string& dir,
                                    cudaStream_t stream, int rank = 0, int world = 1, bool fp8 = false);
// The arena size load_dflash2_weights allocates (the memory plan's line).
size_t dflash2_weights_bytes(const DFlash2Config& cfg, int world = 1, bool fp8 = false);

}  // namespace dgpp
