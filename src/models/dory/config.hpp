#pragma once
// Dory1.0-4B (DoryForCausalLM, model_type dory) configuration, parsed from
// the checkpoint's config.json root (2026-10-16, docs/dory_plan.md §1).
// The same policy as the GLM, Qwen, MiMo and DeepSeek parsers: every field
// the assembly consumes is parsed into a known-supported value or rejected
// with a message naming the field, at load time. The reference is the vLLM
// port (PR #60149: vllm/model_executor/models/dory.py and
// vllm/transformers_utils/configs/dory.py), the only executable description
// of the architecture — transformers has no `dory` model directory, and the
// checkpoint ships no remote code.
//
// The architecture: a normalized GPT (nGPT) of `num_hidden_layers` physical
// layers, each a SINGLE sublayer — attention or MLP, never both
// (`layer_types` alternates; a `full_attention` block every 8). The stack is
// partitioned into an input block, a recurrent block applied
// `n_recurrent_loops` times with the input re-injected and the state
// re-normalized at every loop, and an output block. There is no LayerNorm
// anywhere: normalization is nGPT's `justnorm` (row-wise L2 with eps 1e-6),
// and every residual is a slerp toward the sublayer output. Attention is GQA
// with a per-head nGPT qk-norm applied AFTER RoPE, a learned per-dimension
// `sqk` scale, an output gate (`s_gate`), and a `justnorm` over all query
// heads before o_proj. The MLP is SwiGLU with the learned `suv_gate` /
// `suv_up` scales and a `justnorm` over the intermediate.
//
// The quantization contract (the release as shipped): modelopt
// W4A16_NVFP4, group_size 16, weight-only — the U8 e2m1 codes two per byte
// with an E4M3 scale grid of in/16 columns and one F32 per-tensor
// `weight_scale_2`, and no activation scales (no `input_scale` tensor
// exists). The embedding and the head are BF16 on the ignore list.
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"

namespace dgpp {

// One physical layer's kind, from `layer_types`.
enum class DoryLayerKind : uint8_t {
  SlidingAttention,
  FullAttention,
  Mlp,
};

struct DoryConfig {
  // --- model shape -------------------------------------------------------
  int hidden_size = 2560;
  int vocab_size = 131072;
  int num_hidden_layers = 72;
  int intermediate_size = 7168;
  int max_position_embeddings = 262144;
  // The stop tokens the config names (compatibility fallback: the served
  // defaults ride generation_config.json, see GlmGenerationDefaults). The
  // 4B release names none here — its generation_config.json has [2, 11].
  std::vector<int64_t> eos_token_ids;
  std::string hidden_act = "silu";
  bool tie_word_embeddings = false;
  bool attention_bias = false;
  bool mlp_bias = false;

  // --- the nGPT scales (normalized at inference by initializer_range) -----
  // Every learned scale is divided by `initializer_range`; it is required
  // because it changes the numerics of every block.
  float initializer_range = 0.0f;      // required; must be > 0
  float ngpt_alpha_init = 0.05f;       // the residual (slerp) scale
  // `attention_output_gate_logit_scale_init`; <= 0 means the release's null,
  // which the reference resolves to sqrt(hidden_size).
  float attention_output_gate_logit_scale_init = 0.0f;

  // --- the loop ----------------------------------------------------------
  int n_input_layers = 0;
  int n_recurrent_layers = -1;  // -1: derive from num_hidden_layers - in - out
  int n_output_layers = 0;
  int n_recurrent_loops = 1;

  // --- attention ---------------------------------------------------------
  int num_attention_heads = 16;
  int num_key_value_heads = 4;
  int head_dim = 256;
  int sliding_window = 1024;           // the sliding layers' window; <= 0: none
  double rope_theta_sliding = 1e4;     // `rope_parameters.sliding_attention`
  double rope_theta_full = 1e7;        // `rope_parameters.full_attention`
  std::vector<DoryLayerKind> layer_kind;  // [num_hidden_layers]

  // --- weight formats ----------------------------------------------------
  int nvfp4_group = 16;  // the E4M3 scale grid's column step

  // --- parsed state ------------------------------------------------------
  static DoryConfig parse(const minijson::Value& root);
  static DoryConfig from_json_file(const std::string& path);

  // --- derived geometry ---------------------------------------------------
  int64_t q_size() const { return static_cast<int64_t>(num_attention_heads) * head_dim; }
  int64_t kv_size() const { return static_cast<int64_t>(num_key_value_heads) * head_dim; }
  int64_t gate_up_rows() const { return 2 * static_cast<int64_t>(intermediate_size); }

  // --- layer helpers ------------------------------------------------------
  DoryLayerKind kind_of(int l) const { return layer_kind[static_cast<size_t>(l)]; }
  bool is_attention(int l) const { return kind_of(l) != DoryLayerKind::Mlp; }
  bool is_full_attention(int l) const { return kind_of(l) == DoryLayerKind::FullAttention; }
  bool is_sliding(int l) const { return kind_of(l) == DoryLayerKind::SlidingAttention; }
  bool in_input_block(int l) const { return l < n_input_layers; }
  bool in_recurrent_block(int l) const {
    return l >= n_input_layers && l < n_input_layers + n_recurrent_layers;
  }
  bool in_output_block(int l) const { return l >= n_input_layers + n_recurrent_layers; }
  double rope_theta_of(int l) const { return is_full_attention(l) ? rope_theta_full : rope_theta_sliding; }
  // How many K/V streams the layer's cache carries: the recurrent block's
  // attention layers repeat once per loop unless the last-loop policy shares
  // one cache (the engine key of plan D2).
  int kv_repeats(int l, bool last_loop) const {
    return (in_recurrent_block(l) && !last_loop) ? n_recurrent_loops : 1;
  }
  int num_attention_layers() const {
    int n = 0;
    for (DoryLayerKind k : layer_kind) n += k != DoryLayerKind::Mlp;
    return n;
  }

  // --- the nGPT scale factors (fp32, as the reference computes them) ------
  float alpha_scale() const { return ngpt_alpha_init / initializer_range; }
  float sqk_factor() const {
    return static_cast<float>(std::sqrt(static_cast<double>(head_dim))) / initializer_range;
  }
  float s_gate_factor() const {
    const double g0 = attention_output_gate_logit_scale_init > 0.0f
                          ? static_cast<double>(attention_output_gate_logit_scale_init)
                          : std::sqrt(static_cast<double>(hidden_size));
    return static_cast<float>(std::sqrt(g0) / initializer_range);
  }
  float suv_factor() const {
    return static_cast<float>(std::sqrt(static_cast<double>(hidden_size)));
  }
  float logit_factor() const { return 1.0f / initializer_range; }
};

}  // namespace dgpp
