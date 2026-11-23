#pragma once
// Inkling 2 looped model (Inkling2LoopForConditionalGeneration, model_type
// inkling2_loop) configuration, parsed from the checkpoint's config.json and
// hf_quant_config.json (docs/inkling2_mini_plan.md section 1). The same
// policy as the other families' parsers: every field the assembly consumes
// is parsed into a known-supported value or rejected with a message naming
// the field, at load time. The references are the transformers
// `inkling2_loop` model (the math) and vLLM's `models/inkling2` (the
// released numerics).
//
// The architecture: `num_hidden_layers` layers of weights in blocks of
// `n_layers_per_attn_res_block`. Layers [core_start, core_end) run
// `num_passes` times, so a token passes `num_visits()` layer visits. A
// visit is one execution of a layer with its own K/V cache and convolution
// state. Layers in `local_layer_ids` attend over a sliding window, the
// others over the whole context. Every layer has a routed MoE whose expert
// format (NVFP4 or MXFP8) the precision metadata names per tensor.
#include <cstdint>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"

namespace dgpp {

enum class Inkling2ExpertFormat : uint8_t { Nvfp4 = 0, Mxfp8 = 1 };

// The bundled DSpark draft model (`mtp_config`, weights in mtp.safetensors).
struct Inkling2DraftConfig {
  bool present = false;
  int hidden_size = 0;
  int num_hidden_layers = 0;
  int num_attention_heads = 0;
  int num_key_value_heads = 0;
  int head_dim = 0;
  int intermediate_size = 0;
  int64_t max_position_embeddings = 0;
  double rope_theta = 1e6;
  float rms_norm_eps = 1e-6f;
  std::vector<uint8_t> local_layer;   // [num_hidden_layers]: 1 = sliding window
  int sliding_window_size = 0;
  int block_size = 0;                 // the anchor row plus block_size - 1 mask rows
  int markov_rank = 0;
  // Target visits whose input (the partial state at the visit's entry) the
  // draft reads, then the target's visit count for the final-norm output.
  std::vector<int> target_layer_ids;

  int drafts() const { return block_size - 1; }
};

struct Inkling2VisionConfig {
  bool present = false;
  int text_hidden_size = 0;
  int hidden_size = 0;
  int num_hidden_layers = 0;
  int num_attention_heads = 0;
  int native_patch_size = 0;
  int temporal_patch_size = 0;
  int adapter_merge_size = 0;
  int projection_hidden_dim = 0;
  int posemb_grid_size = 0;
  double mlp_ratio = 4.0;
  float layer_norm_eps = 1e-5f;
  bool projector_bias = false;
};

struct Inkling2Visit {
  int layer = 0;
  int pass = 0;
};

struct Inkling2Config {
  // --- model shape -------------------------------------------------------
  int hidden_size = 2304;
  int num_hidden_layers = 24;
  int vocab_size = 200058;            // the rows the head reads (unpadded_vocab_size)
  int padded_vocab_size = 201024;     // the rows the embedding and unembedding tensors store
  int64_t max_position_embeddings = 1048576;
  float rms_norm_eps = 1e-6f;
  std::vector<int64_t> eos_token_ids;
  int64_t pad_token_id = -1;
  int64_t image_token_id = 200054;
  float final_logit_softcapping = 30.0f;   // 0: none

  // --- blocks and the loop -------------------------------------------------
  int block_layers = 6;               // n_layers_per_attn_res_block
  int core_start = 6;
  int core_end = 18;
  int num_passes = 3;
  int gate_rank = 64;

  // --- attention -----------------------------------------------------------
  int num_attention_heads = 24;
  int num_key_value_heads = 8;
  int head_dim = 128;
  int d_rel = 16;
  int rel_extent = 1024;              // the full-attention layers' bias extent
  int sliding_window_size = 1024;     // a local row attends [pos - W + 1, pos], its bias extent is W
  int sconv_kernel_size = 4;
  float qk_l2_clip_threshold = 22.627416997969522f;
  float rel_l2_clip_threshold = 8.0f;
  bool log_scaling = true;            // log_scaling_n_floor present
  int64_t log_scaling_n_floor = 8192;
  float log_scaling_alpha = 0.1f;
  std::vector<uint8_t> local_layer;   // [num_hidden_layers]: 1 = sliding window

  // --- MoE -------------------------------------------------------------------
  int n_routed_experts = 128;
  int num_experts_per_tok = 8;
  int intermediate_size = 1152;       // per expert
  float route_scale = 8.0f;
  float mlp_gate_softcap = 4.0f;      // 0: none
  float mlp_up_softcap = 25.0f;       // 0: none
  // Per layer and projection (0 = w13, 1 = w2).
  std::vector<Inkling2ExpertFormat> expert_format;

  Inkling2DraftConfig draft;
  Inkling2VisionConfig vision;

  // `quant` is the root of hf_quant_config.json.
  static Inkling2Config parse(const minijson::Value& root, const minijson::Value& quant);
  static Inkling2Config from_checkpoint_dir(const std::string& dir);

  bool is_local_layer(int l) const { return local_layer[static_cast<size_t>(l)] != 0; }
  bool is_core_layer(int l) const { return l >= core_start && l < core_end; }
  bool first_of_block(int l) const { return l % block_layers == 0; }
  Inkling2ExpertFormat format_of(int l, int projection) const {
    return expert_format[static_cast<size_t>(l) * 2 + static_cast<size_t>(projection)];
  }
  bool layer_mxfp8(int l) const { return format_of(l, 0) == Inkling2ExpertFormat::Mxfp8; }

  int core_layers() const { return core_end - core_start; }
  // Layer executions per token: the prelude, every pass of the core, the coda.
  int num_visits() const { return num_hidden_layers + (num_passes - 1) * core_layers(); }
  Inkling2Visit visit(int v) const;
  int visit_of(int layer, int pass) const;
  // The bias extent of layer l's attention.
  int rel_extent_of(int l) const { return is_local_layer(l) ? sliding_window_size : rel_extent; }
  // Completed blocks a forward holds at most (each pass restarts from the
  // prelude's count, the coda continues the last pass).
  int max_blocks() const;
  // The previous pass's states a message reads: the blocks it completed
  // after its first, then its partial.
  int bank_rows() const { return core_layers() / block_layers; }
  int num_full_visits() const;
  int num_local_visits() const { return num_visits() - num_full_visits(); }
  // Rows of the fused query-side projection [q | r | gate | sink].
  int64_t q_side_rows() const {
    return static_cast<int64_t>(num_attention_heads) * (2 * head_dim + d_rel + 1);
  }
  int64_t kv_rows() const { return static_cast<int64_t>(num_key_value_heads) * head_dim; }
  int64_t o_proj_cols() const { return static_cast<int64_t>(num_attention_heads) * head_dim; }
};

}  // namespace dgpp
