#pragma once
// Expected-tensor table for the Qwen3.8-Flash-Next text model (Q2,
// 2026-09-09, docs/qwen38_flash_next_plan.md §1.11): every tensor the
// checkpoint must contain for the text stack — names, dtypes, exact shapes
// — derived from the parsed config, not observed from one file. The table
// drives the offline validator and the resident loader, as GLM's does.
//
// Naming is checkpoint truth (Qwen/Qwen3.8-Flash-Next-FP8 @ 236dfdf2):
// main layers under `model.language_model.layers.L.`, the draft layer under
// `mtp.layers.0.` with the head's own tensors under `mtp.`, the globals
// `model.language_model.embed_tokens.weight`, `lm_head.weight` and the
// final mixer `model.language_model.hyper_connection_mixer.*`.
//
// Scale contract (the FP8 release): every routed-expert e4m3 matrix
// X.weight carries a BF16 partner X.weight_scale_inv of shape
// [ceil(N/128), ceil(K/128)] — 128x128 dequant blocks, MULTIPLY on dequant
// (the loader widens the scales to F32 at load). The n-gram table is e4m3
// in `split_ngram_parts` row shards with one BF16 per-tensor
// `weight_scale`. Everything else is BF16 (the hash buffers I64).
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "models/qwen/config.hpp"

namespace dgpp {

enum class QwenWeightClass : int {
  Embed,
  LmHead,
  Mixer,         // the model-level hyper_connection_mixer (mix only)
  Gr,            // a layer's attn/mlp gated-residual sites
  Gdn,
  Qsa,
  QsaIndexer,
  Router,        // mlp.gate and mlp.shared_expert_gate
  SharedExpert,
  RoutedExpert,
  Ple,           // the PLE projections, norms, conv and hash buffers
  PleTable,      // the n-gram table shards and their scale
  Mtp,           // the draft head's own tensors (fc_*, pre_fc norms, mixer)
  Norm,          // qwen3_5 RMSNorm vectors (layer norms, final norm)
  FullAttn,      // qwen3_5 full-attention projections and q/k norms
  DenseMlp,      // qwen3_5 dense SwiGLU gate/up/down projections
};

enum class QwenTensorRole : uint8_t {
  Plain,
  Fp8Payload,  // e4m3 [N, K], partner `_scale_inv`
  Fp8Scale,    // BF16 [ceil(N/128), ceil(K/128)]
  NgramShard,  // e4m3 [rows, head_dim], one of split_ngram_parts row shards
  NgramScale,  // BF16 [1], the table's per-tensor scale
  Fp4Payload,  // U8 [N, K/2] e2m1 pairs (the NVFP4 release's routed experts)
  Fp4Scale,    // F8_E4M3 [N, K/16]
  Fp4Global,   // F32 [], the matrix's weight_scale_2
  InputScale,  // F32 [], the recipe's activation scale (unused: W4A16)
  // The AutoRound hybrid's GPTQ triple (docs/qwen38_autoround_int4_plan.md
  // §1.1): `base.qweight` I32 [K*bits/32, N] (codes packed along K, N
  // contiguous — the packed core's rows transposed, at load), `base.scales`
  // F16 [K/128, N], `base.qzeros` I32 [K/128, N*bits/32] (the symmetric
  // constant, verified at load and never resident).
  GptqCodes,
  GptqScales,
  GptqZeros,
};

struct QwenExpectedTensor {
  std::string name;
  DType dtype{};
  std::vector<int64_t> shape;
  QwenWeightClass cls = QwenWeightClass::Gr;
  int layer = -1;   // layer index; -1 for globals; mtp_layer() for the draft
  int expert = -1;  // routed-expert id; the shard index for NgramShard
  QwenTensorRole role = QwenTensorRole::Plain;

  bool quantized() const {
    return role == QwenTensorRole::Fp8Payload || role == QwenTensorRole::Fp4Payload ||
           role == QwenTensorRole::GptqCodes;
  }
  size_t numel() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  size_t nbytes() const { return numel() * dtype_size(dtype); }
};

// [ceil(N/128), ceil(K/128)] — the 128x128 block grid.
std::vector<int64_t> qwen_scale_shape(const std::vector<int64_t>& payload);

// The n-gram table's shard rows: shard s of split_ngram_parts holds rows
// [s*R, min((s+1)*R, padded)) with R = ceil(padded / parts).
int64_t qwen_ngram_shard_rows(const QwenTextConfig& cfg, int shard);
int64_t qwen_ngram_shard_capacity(const QwenTextConfig& cfg);

// The checkpoint name prefix of a layer ("model.language_model.layers.L."
// or "mtp.layers.0." for the draft layer).
std::string qwen_layer_prefix(const QwenTextConfig& cfg, int layer);

// Full text-model table (main layers + the draft layer + globals).
std::vector<QwenExpectedTensor> qwen_expected_text_tensors(const QwenTextConfig& cfg);
// One layer's entries: `layer` in [0, num_hidden_layers) or mtp_layer().
// The PLE layer's entries include the n-gram table shards.
std::vector<QwenExpectedTensor> qwen_expected_layer_tensors(const QwenTextConfig& cfg, int layer);
// The globals: embed, lm_head, the final mixer, and the draft head's own
// tensors when the draft layer exists.
std::vector<QwenExpectedTensor> qwen_expected_global_tensors(const QwenTextConfig& cfg);

struct QwenTensorDesc {
  DType dtype{};
  std::vector<int64_t> shape;
};

struct QwenBindReport {
  size_t expected = 0;
  size_t matched = 0;
  size_t missing = 0;
  size_t dtype_mismatch = 0;
  size_t shape_mismatch = 0;
  size_t unexpected = 0;  // present, not in the table, not vision
  size_t out_of_scope = 0;  // layers past a truncated config (the check apps' --layers)
  size_t vision = 0;      // model.visual.* (counted, not validated)
  size_t quantized_matrices = 0;
  size_t ngram_shards = 0;
  std::vector<std::string> errors;
  bool ok() const {
    return missing == 0 && dtype_mismatch == 0 && shape_mismatch == 0 && unexpected == 0;
  }
};

QwenBindReport qwen_validate_text_binding(
    const QwenTextConfig& cfg,
    const std::unordered_map<std::string, QwenTensorDesc>& present,
    size_t max_errors = 32);

// Tensor-parallel geometry acceptance (docs/qwen38_flash_next_plan.md
// §2.1): throws std::invalid_argument naming the dim that does not divide.
// world must divide the GDN key and value heads, the attention query heads
// and the n-gram hash heads; either world divides the kv heads or the kv
// heads divide world (a kv head then lives on world/kv ranks); the expert
// slice moe_inter/world must be a multiple of 16 (the scale sub-block
// gcd(128, slice) then re-anchors the block grid exactly, D2).
void qwen_tp_validate_geometry(const QwenTextConfig& cfg, int rank, int world);

}  // namespace dgpp
