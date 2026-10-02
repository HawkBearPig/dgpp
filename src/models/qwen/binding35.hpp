#pragma once
// Expected-tensor table for the Qwen3.8-27B text model (Qwen3_5ForConditionalGeneration):
// every tensor the checkpoint must contain for the text stack — names,
// dtypes, exact shapes — derived from the parsed config, not observed from
// one file. The table drives the offline validator and (with the loader
// slice) the resident loader.
//
// Naming is checkpoint truth (Qwen/Qwen3.8-27B-FP8): main layers under
// `model.language_model.layers.L.`, the draft layer under `mtp.layers.0.`
// with the head's own tensors under `mtp.`, the globals
// `model.language_model.embed_tokens.weight`, `lm_head.weight` and
// `model.language_model.norm.weight`. Vision (`model.visual.*`) is counted
// and skipped: text-only scope.
//
// Scale contract (the FP8 release): every e4m3 matrix X.weight carries a
// BF16 partner X.weight_scale_inv of shape [ceil(N/128), ceil(K/128)] —
// 128x128 dequant blocks, MULTIPLY on dequant (the loader widens the
// scales to F32 at load). Everything else is BF16.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "models/qwen/binding.hpp"
#include "models/qwen/config35.hpp"

namespace dgpp {

// The checkpoint name prefix of a layer ("model.language_model.layers.L."
// or "mtp.layers.0." for the draft layer).
std::string qwen35_layer_prefix(const Qwen35TextConfig& cfg, int layer);

// Full text-model table (main layers + the draft layer + globals).
std::vector<QwenExpectedTensor> qwen35_expected_text_tensors(const Qwen35TextConfig& cfg);
// One layer's entries: `layer` in [0, num_hidden_layers) or mtp_layer().
std::vector<QwenExpectedTensor> qwen35_expected_layer_tensors(const Qwen35TextConfig& cfg,
                                                             int layer);
// The globals: embed, lm_head, the final norm, and the draft head's own
// tensors when the draft layer exists.
std::vector<QwenExpectedTensor> qwen35_expected_global_tensors(const Qwen35TextConfig& cfg);

QwenBindReport qwen35_validate_text_binding(
    const Qwen35TextConfig& cfg,
    const std::unordered_map<std::string, QwenTensorDesc>& present,
    size_t max_errors = 32);

// Tensor-parallel geometry acceptance: throws std::invalid_argument naming
// the dim that does not divide. world must divide the GDN key and value
// heads, the attention query heads, and the dense intermediate size; either
// world divides the kv heads or the kv heads divide world. FP8 scale-grid
// slicing (multiples of 128) is the loader's contract, not this check's.
void qwen35_tp_validate_geometry(const Qwen35TextConfig& cfg, int rank, int world);

}  // namespace dgpp
