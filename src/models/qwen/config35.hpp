#pragma once
// Qwen3.8-27B (Qwen3_5ForConditionalGeneration) text-model configuration,
// parsed from the checkpoint's config.json text_config. Same policy as the
// other families: every field the assembly consumes is parsed into a
// known-supported value or rejected with a message naming the field, at
// load time. The reference is transformers' modular_qwen3_5.py.
//
// This family differs from Qwen4Exp (Flash-Next): no hyper-connections, no
// MoE, no indexer, no n-gram table. Its layers are plain Gated DeltaNet
// (linear_attention, swish output gate) and plain full attention (GQA with
// partial rotary + MROPE + a per-head output gate), sandwiching a dense
// SwiGLU MLP, between two RMSNorms. Text-only scope: vision_config is
// ignored (image requests stay refused until the tower lands).
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"

namespace dgpp {

enum class Qwen35LayerKind : int { Gdn, Full };

enum class Qwen35QuantKind : int {
  Fp8Block,   // e4m3 + BF16 128x128 block scales, dynamic activations
  Nvfp4Mixed, // compressed-tensors mixed: MLP nvfp4 group16, attn FP8, kv 8b hint
  Nvfp4Moe,   // ModelOpt NVFP4 (nvidia/Qwen3.5-122B-A10B-NVFP4): routed
              // experts nvfp4 group16 (ModelOpt naming), rest BF16
};

// The format one matrix carries in the Nvfp4Mixed release: which
// config_groups entry (if any) claims the checkpoint module.
enum class Qwen35TensorQuant : int { Bf16, Fp8Channel, Nvfp4 };

// compressed-tensors `config_groups`, parsed: the ignore list and each
// group's target regexes in document order (models/qwen/config35.cpp). The
// binding table derives every matrix's format from these rules — never from
// one file's tensor list — so the loader and a checkpoint cannot disagree
// about a name. Group resolution follows the format's rule: the ignore list
// first, then the first group whose targets match (the release's groups
// overlap on purpose — group_0 claims the late MLP layers by a
// `layers.(56|…)` regex that group_1's blanket mlp regex also matches).
struct Qwen35QuantGroups;

struct Qwen35TextConfig {
  // --- model shape -------------------------------------------------------
  int hidden_size = 5120;
  int vocab_size = 248320;
  int num_hidden_layers = 64;
  float rms_norm_eps = 1e-6f;
  bool tie_word_embeddings = false;
  std::string hidden_act = "silu";
  int max_position_embeddings = 262144;
  std::vector<Qwen35LayerKind> layers;  // size == num_hidden_layers
  std::vector<int64_t> eos_token_ids;
  int64_t bos_token_id = -1;

  // --- Gated DeltaNet (linear_attention) ----------------------------------
  int gdn_key_heads = 16;
  int gdn_value_heads = 48;
  int gdn_key_head_dim = 128;
  int gdn_value_head_dim = 128;
  int gdn_conv_width = 4;
  // This family trains the swish output gate; the gated kernels land with
  // the layer slice (the config records it now so checkpoints bind today).
  std::string output_gate_type = "swish";

  // --- full attention (GQA + partial rotary + MROPE + output gate) --------
  int num_attention_heads = 24;
  int num_key_value_heads = 4;
  int head_dim = 256;
  int rotary_dim = 64;  // head_dim * partial_rotary_factor
  double rope_theta = 1e7;
  std::vector<int> mrope_section;  // [11, 11, 10]; text positions are 1-D
  bool mrope_interleaved = true;
  // The q projection stacks [q | gate] per head (attn_output_gate), so its
  // row count is 2 * heads * head_dim.
  bool attn_output_gate = true;

  // --- dense SwiGLU MLP ----------------------------------------------------
  // Dense variant only (is_moe == false). MoE variant uses moe_* below.
  int intermediate_size = 17408;

  // --- MoE (qwen3_5_moe_text only) -------------------------------------------
  bool is_moe = false;
  int num_experts = 0;
  int num_experts_per_tok = 0;
  int moe_intermediate_size = 0;
  int shared_expert_intermediate_size = 0;
  // Softmax top-k router with renormalization (transformers default true;
  // the 122B checkpoint omits the field, the base class defaults it true).
  bool norm_topk_prob = true;

  // --- MTP ------------------------------------------------------------------
  int mtp_num_layers = 1;  // 0 or 1; the draft layer is a Full layer

  // --- weight formats -------------------------------------------------------
  Qwen35QuantKind quant_kind = Qwen35QuantKind::Fp8Block;
  // Nvfp4Mixed only: the parsed config_groups (null for Fp8Block).
  std::shared_ptr<const Qwen35QuantGroups> quant_groups;

  static Qwen35TextConfig parse(const minijson::Value& text_config,
                                const minijson::Value* quantization_config);
  // Reads config.json from disk (the root object) and dispatches to parse().
  static Qwen35TextConfig from_json_file(const std::string& path);

  int num_gdn_layers() const;
  int num_full_layers() const;
  // The quantization format of one matrix, by its checkpoint module path
  // (e.g. "model.language_model.layers.3.mlp.gate_proj"): the ignore list,
  // then the first config_groups target that matches; an unclaimed matrix is
  // BF16. Only meaningful with quant_kind == Nvfp4Mixed (the Fp8Block
  // binding needs no per-matrix resolution).
  Qwen35TensorQuant tensor_quant(const std::string& module) const;
  // Layer index of the MTP draft layer (num_hidden_layers), -1 when absent.
  int mtp_layer() const { return mtp_num_layers == 1 ? num_hidden_layers : -1; }
  int64_t context_limit() const { return static_cast<int64_t>(max_position_embeddings); }
};

}  // namespace dgpp
