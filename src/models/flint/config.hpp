#pragma once
// Flint (FlintForCausalLM, model_type flint) configuration, parsed from the
// checkpoint's config.json root (2026-10-22, docs/flint_plan.md §1). The
// same policy as the MiMo, GLM-4.7, Qwen and DeepSeek parsers: every field
// the assembly consumes is parsed into a known-supported value or rejected
// with a message naming the field, at load time. The reference is the
// release's flint_sglang 0.1.4 model.py and the torch reference the
// quantization study pinned.
//
// The architecture: 49 layers of GQA attention (32 query heads over 4 kv
// heads, 128-wide q / k / v, a per-head RMSNorm on q and k, a learned
// per-head sink logit on every layer) in two kinds on `layer_types` — 35
// sliding-window layers (window 513: the row and the 512 before it, RoPE
// over the whole head) and 14 full-attention layers (every one in
// `no_rope_layer_indices`: no positional encoding) — then a dense SwiGLU
// MLP on `mlp_only_layers` (0 and 1, 6144 wide) or a 256-expert MoE (top 6
// on sigmoid scores, the picked scores renormalized, no bias tensor) with
// an ungated BF16 shared expert (1024 wide) on the rest. No draft head.
//
// The quantization contract (HawkBearPig/flint-FP8-NVFP4 as shipped):
// quant_method dgpp_mixed; every non-expert tensor BF16; each routed
// expert projection of each MoE layer FP8 (e4m3 on 128 x 128 blocks with a
// multiplicative fp32 `weight_scale_inv`) or NVFP4 (`weight_packed` e2m1
// pairs, `weight_scale` e4m3 per 16 along K, `weight_global_scale` an fp32
// reciprocal) as `routed_experts.precision_by_layer` names it per layer
// and projection.
#include <cstdint>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"
#include "models/glm/moe.hpp"

namespace dgpp {

struct FlintTextConfig {
  // --- model shape -------------------------------------------------------
  int hidden_size = 2048;
  int vocab_size = 150144;
  int num_hidden_layers = 49;
  float rms_norm_eps = 1e-6f;
  bool tie_word_embeddings = false;
  std::string hidden_act = "silu";
  int max_position_embeddings = 262144;
  std::vector<int64_t> eos_token_ids;  // config.json's; serving resolves generation stop IDs separately
  int64_t bos_token_id = -1;
  int64_t pad_token_id = -1;
  std::vector<uint8_t> swa_layer;   // [num_hidden_layers]: 1 = sliding_attention, 0 = full_attention
  std::vector<uint8_t> rope_layer;  // [num_hidden_layers]: 1 = RoPE applied (not in no_rope_layer_indices)
  std::vector<uint8_t> moe_layer;   // [num_hidden_layers]: 1 = routed MoE, 0 = dense MLP (mlp_only_layers)

  // --- attention -----------------------------------------------------------
  int num_attention_heads = 32;
  int num_key_value_heads = 4;
  int head_dim = 128;                  // q, k and v
  double rope_theta = 1e4;
  int sliding_window = 513;            // a sliding row attends [pos - W + 1, pos]
  bool attention_bias = false;
  bool sink = true;                    // softmax_type learnable: softmax_offset per head

  // --- MLP / MoE -----------------------------------------------------------
  int intermediate_size = 6144;        // the dense layers
  int moe_intermediate_size = 512;     // per routed expert
  int shared_expert_intermediate_size = 1024;
  int n_routed_experts = 256;
  int num_experts_per_tok = 6;
  bool norm_topk_prob = true;

  // --- weight formats -----------------------------------------------------
  int fp8_block = 128;                 // the fp8 experts' block (both axes)
  int nvfp4_group = 16;                // the NVFP4 experts' scale group along K
  // Per layer: bit 0 gate, bit 1 up, bit 2 down — set when that routed
  // projection is NVFP4 (clear: FP8). Zero on the dense layers.
  std::vector<uint8_t> expert_fp4_mask;

  static FlintTextConfig parse(const minijson::Value& root);
  static FlintTextConfig from_json_file(const std::string& path);

  bool is_swa_layer(int l) const { return swa_layer[static_cast<size_t>(l)] != 0; }
  bool rope_of(int l) const { return rope_layer[static_cast<size_t>(l)] != 0; }
  bool is_moe_layer(int l) const { return moe_layer[static_cast<size_t>(l)] != 0; }
  // Whether projection `which` (0 gate, 1 up, 2 down) of MoE layer `l` is NVFP4.
  bool expert_fp4(int l, int which) const {
    return (expert_fp4_mask[static_cast<size_t>(l)] >> which) & 1u;
  }
  int num_moe_layers() const {
    int n = 0;
    for (uint8_t m : moe_layer) n += m != 0;
    return n;
  }
  int num_swa_layers() const {
    int n = 0;
    for (uint8_t s : swa_layer) n += s != 0;
    return n;
  }
  // The fused [Q | K | V] projection's rows and o_proj's columns.
  int64_t qkv_rows() const {
    return static_cast<int64_t>(num_attention_heads + 2 * num_key_value_heads) * head_dim;
  }
  int64_t o_proj_cols() const { return static_cast<int64_t>(num_attention_heads) * head_dim; }
  // The routed chain's configuration (models/glm/moe.hpp): the sigmoid
  // router with a zero selection bias (docs/flint_plan.md D7), no shared
  // expert in the chain (the BF16 one rides QwenMoeLayer's tail), no
  // swiglu clamps. `local_inter` is this rank's slice of moe_intermediate_size.
  GlmMoeConfig moe_config(int local_inter) const;
};

}  // namespace dgpp
