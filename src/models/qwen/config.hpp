#pragma once
// Qwen3.8-Flash-Next (Qwen4ExpForConditionalGeneration) text-model
// configuration, parsed from the checkpoint's config.json text_config
// (docs/qwen38_flash_next_plan.md §1.1). The same policy as the GLM
// config: every field the assembly consumes is parsed into a known-
// supported value or rejected with a message naming the field, at load
// time. The reference is transformers' modular_qwen4_exp.py (5.8).
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "kernels/rope_scaling.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen/vision_config.hpp"

namespace dgpp {

enum class QwenLayerKind : int { Gdn, Qsa };

// The hashed n-gram table's geometry, derived from config alone the way
// the reference derives it (§1.7): head h's vocabulary is the (h+1)-th
// prime after ngram_vocab_size_base - 1, the offsets are their prefix
// sums, the total is padded to the divisor; the three hash multipliers
// come from splitmix64 over the seed. The loader checks the checkpoint's
// stored buffers (`layer_multipliers`, `ngram_heads_vocab_sizes`,
// `ngram_heads_offsets`) against this and refuses a disagreement.
struct QwenNgramGeometry {
  int heads = 0;                     // (ngram_size - 1) * heads_per_ngram
  int head_dim = 0;                  // ple_embed_dim / heads
  std::vector<int64_t> head_vocab;   // per head, primes
  std::vector<int64_t> head_offset;  // per head, prefix sums
  int64_t total_rows = 0;            // sum of head_vocab
  int64_t padded_rows = 0;           // ceil(total / divisor) * divisor
  std::vector<int64_t> multipliers;  // ngram_size of them (int64, odd)
};

struct QwenTextConfig {
  // --- model shape -------------------------------------------------------
  int hidden_size = 2560;
  int vocab_size = 248320;
  int num_hidden_layers = 48;
  float rms_norm_eps = 1e-6f;
  bool tie_word_embeddings = false;
  std::string hidden_act = "silu";
  int max_position_embeddings = 262144;
  std::vector<QwenLayerKind> layers;  // size == num_hidden_layers
  // Trained EOS from config.json, including PLE padding/reset semantics.
  // Generation stop tokens are resolved separately by the serving layer.
  std::vector<int64_t> eos_token_ids;
  int64_t bos_token_id = -1;

  // --- gated residual (hyper-connections) ---------------------------------
  int hc_count = 4;
  int hc_lowrank = 320;

  // --- Gated DeltaNet -------------------------------------------------------
  int gdn_key_heads = 16;
  int gdn_value_heads = 48;
  int gdn_key_head_dim = 128;
  int gdn_value_head_dim = 128;
  int gdn_conv_width = 4;
  std::string output_gate_type = "sigmoid";

  // --- Qwen Sparse Attention ------------------------------------------------
  int num_attention_heads = 24;
  int num_key_value_heads = 2;
  int head_dim = 256;
  int rotary_dim = 64;       // head_dim * partial_rotary_factor
  double rope_theta = 1e7;
  std::vector<int> mrope_section;  // [11, 11, 10]; text positions are 1-D
  bool mrope_interleaved = true;
  int indexer_n_heads = 4;
  int indexer_kv_heads = 1;
  int indexer_head_dim = 128;
  int indexer_budget = 2048;
  int indexer_compress_ratio = 4;
  // The opt-in YaRN ramp (engine.rope_scaling, 2026-09-18), set by the
  // serving layer on the parsed config — never by the checkpoint, whose
  // rope_parameters must stay `rope_type: default` (the NVIDIA NVFP4
  // release declares no scaling at all; vLLM applies YaRN from the
  // launcher's --hf-overrides). Empty = the plain table, bit for bit
  // what this family built before the knob existed.
  std::optional<RopeScaling> rope_scaling;

  // --- MoE ------------------------------------------------------------------
  int num_experts = 512;
  int num_experts_per_tok = 10;
  int moe_intermediate_size = 640;
  int shared_expert_intermediate_size = 640;
  bool norm_topk_prob = true;

  // --- n-gram embedding (PLE) -------------------------------------------
  std::vector<int> ple_layer_ids;  // one-indexed decoder layers (the file's convention)
  int ple_embed_dim = 2560;
  int ple_conv_kernel_size = 4;
  int ngram_size = 3;
  int heads_per_ngram = 8;
  int64_t ngram_vocab_size_base = 20000000;
  int ngram_vocab_divisor = 128;
  int64_t ngram_seed = 1234;
  int split_ngram_parts = 128;

  // --- MTP --------------------------------------------------------------------
  int mtp_num_layers = 1;  // 0 or 1; the layer is a QSA layer

  // --- weight formats -----------------------------------------------------
  // The FP8 release: routed experts e4m3 + BF16 128x128 block scales and
  // the n-gram table e4m3 with one BF16 per-tensor scale; everything else
  // BF16. The engine has no other format for this family yet.
  bool experts_fp8 = true;
  // The NVIDIA NVFP4 release: the backbone's routed experts as
  // e2m1 codes x e4m3 scales per 16 x an F32 per-tensor scale; the MTP
  // layer's experts stay FP8 block-128, the n-gram table FP8 as before.
  bool experts_nvfp4 = false;
  bool ngram_table_fp8 = true;
  // The RadixArk release stores the MTP draft layer's 512 experts as a
  // single fused BF16 tensor pair (gate_up_proj [E, 2*I, H] and
  // down_proj [E, H, I]) instead of per-expert FP8 matrices. When true,
  // the loader encodes the BF16 slices to FP8 at load time (the resident
  // form is identical to the standard FP8 path). Set from
  // engine.mtp_expert_format = "bf16_fused" before model construction.
  bool mtp_experts_bf16_fused = false;
  // The AutoRound hybrid (2026-09-28, docs/qwen38_autoround_int4_plan.md
  // D7): `quant_method: gptq`, bits 4, group 128, symmetric, no act-order.
  // Every layer's routed experts (the draft layer's too) come as GPTQ
  // triples — qweight I32 [K/8, N], scales F16 [K/128, N], qzeros I32
  // [K/128, N/8] holding the symmetric constant — the lm_head as the int8
  // triple the `dynamic` map lifts it to, and the dense classes (GDN
  // qkv/z/out, QSA q/k/v/o, the shared expert) as block FP8 e4m3 with F32
  // `weight_scale_inv`: the hybrid's side layers, a fact of the supported
  // checkpoint (the binding refuses any other form by name). The loader
  // transposes the codes into the packed core's row layout and keeps the
  // scales untouched; the zeros are verified and dropped.
  bool experts_gptq_int4 = false;
  bool lm_head_gptq_int8 = false;
  bool dense_fp8_shipped = false;
  int gptq_group = 128;
  // The quantization-aware-distilled export (local-inference-lab/
  // Qwen3.8-Flash-Next-NVFP4, issue #113): modelopt's MIXED_PRECISION
  // block names every quantized module. The backbone's routed experts are
  // the NVIDIA release's NVFP4 triple; the four forms below are the
  // export's own, each a fact of the checkpoint the binding enforces by
  // name (no engine key selects them):
  //   dense_mxfp8_shipped  the GDN a/b/qkv/z/out, QSA q/k/v/o and indexer
  //                        projections and the shared experts of the 48
  //                        backbone layers as MXFP8 — e4m3 `.weight` [N, K]
  //                        with one e8m0 `.weight_scale` byte per 32
  //                        columns of a row ([N, K/32], U8). The draft
  //                        layer's keep BF16.
  //   mtp_experts_nvfp4    the draft layer's experts as the NVFP4 triple
  //                        WITHOUT an input_scale (W4A16_NVFP4: the
  //                        activations stay bf16; no W4A4 on that layer).
  //   ngram_table_nvfp4    the n-gram table shards as e2m1 codes
  //                        (`shard_s.weight` U8 [rows, 80]) with e4m3
  //                        scales per 16 (`shard_s.weight_scale` [rows,
  //                        10]) and one F32 `ngram_embedding.weight_scale_2`.
  //   norms_f32            the backbone's hyper-connection norms (the
  //                        model mixer's too) and the PLE norms trained in
  //                        F32 and shipped so (the publisher's serving
  //                        requirement); the draft layer's norms stay BF16.
  //   experts_nvfp4_csf    the FP4-Local export (nvidia/Qwen3.8-Flash-Next-
  //                        FP4-Local) declares every backbone experts entry
  //                        with weight_scale_encoding "csf": the per-group
  //                        scales arrive as ModelOpt's fixed-plus-exceptions
  //                        sidecars (docs/fp4local_plan.md), not the plain
  //                        E4M3 weight_scale the binding tables otherwise.
  //                        The loader expands them to plain scales once;
  //                        mixed or unknown encodings refuse.
  bool dense_mxfp8_shipped = false;
  bool mtp_experts_nvfp4 = false;
  bool ngram_table_nvfp4 = false;
  bool norms_f32 = false;
  bool experts_nvfp4_csf = false;
  //   mtp_experts_nvfp4_csf the draft layer's experts in the CSF form (the
  //                        FP4-Local export keeps them plain, so this stays
  //                        false there); the same one-export-one-form rule
  //                        holds within the draft entries.
  bool mtp_experts_nvfp4_csf = false;

  // --- vision (docs/vision.md) --------------------------------------------
  // The multimodal release's tower, parsed from the root config's
  // vision_config plus its three image token ids. Absent for a
  // language_model_only export, and the served model is text-only then:
  // image content is refused at request validation, and /v1/models reports
  // the modality list accordingly. from_json_file fills this; parse() on a
  // bare text_config cannot see it.
  std::optional<QwenVisionConfig> vision;
  // The prompt's image delimiters; an unset triple for a text-only export.
  ImageTokens image_tokens() const { return vision ? vision->tokens : ImageTokens{}; }

  static QwenTextConfig parse(const minijson::Value& text_config,
                              const minijson::Value* quantization_config);
  // Reads config.json from disk (the root object) and dispatches to parse().
  static QwenTextConfig from_json_file(const std::string& path);

  int num_gdn_layers() const;
  int num_qsa_layers() const;
  // Layer index of the MTP draft layer (num_hidden_layers), -1 when absent.
  int mtp_layer() const { return mtp_num_layers == 1 ? num_hidden_layers : -1; }
  // The zero-indexed layer carrying the PLE (exactly one in this family).
  int ple_layer() const { return ple_layer_ids.empty() ? -1 : ple_layer_ids[0] - 1; }
  int indexer_block_topk() const { return indexer_budget / indexer_compress_ratio; }
  // The model's positional ceiling: the checkpoint's
  // max_position_embeddings, or the YaRN-scaled value (original x factor)
  // when the knob is on. Everything that bounds a context — the session
  // core's max_context(), the memory plan's context line, the pool gate —
  // reads this, not the raw field.
  int64_t context_limit() const {
    return rope_scaling.has_value() ? rope_scaling->context_limit()
                                    : static_cast<int64_t>(max_position_embeddings);
  }
  int hyper_width() const { return hc_count * hidden_size; }
  // The n-gram table's derivation (§1.7).
  QwenNgramGeometry ngram_geometry() const;
};

}  // namespace dgpp
