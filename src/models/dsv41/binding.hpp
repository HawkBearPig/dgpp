#pragma once
// Expected-tensor table for DeepSeek-V4.1-Flash (DeepseekV41ForCausalLM,
// 2026-09-13, docs/deepseek_v41_flash_plan.md §1.10): every tensor the
// checkpoint must contain — names, dtypes, exact shapes — derived from the
// parsed config, not observed from one file. The table drives the offline
// validator and the resident loader, as the other four tables do.
//
// Naming is checkpoint truth (deepseek-ai/DeepSeek-V4.1-Flash, no `model.`
// prefix): backbone layers under `layers.L.`, the DSpark draft stages under
// `mtp.S.`, the globals `embed.weight`, `norm.weight`, `head.weight`, the
// vision tower under `vision.` / `aligner.` / `image_*` (present in the file,
// never loaded — plan D9).
//
// Format contract (plan §0.1, D2): an fp8 matrix X [N, K] is the pair
// X.weight F8_E4M3 [N, K] + X.scale F8_E8M0 [ceil(N/32), ceil(K/32)]; an
// MXFP4 matrix is X.weight I8 [N, K/2] (two e2m1 codes per byte, the low
// nibble the even element) + X.scale F8_E8M0 [N, K/32]; an Engram table is
// embed.weight F8_E4M3 [rows, 256] + embed.scale F8_E8M0 [rows, 8] (one
// e8m0 per 32 along the row). The dequantized value is `code x 2^(scale -
// 127)`, exact in bf16 for both code formats. Everything else is BF16 or
// F32 as listed.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "models/dsv41/config.hpp"

namespace dgpp {

enum class Dsv41WeightClass : int {
  Embed,
  LmHead,
  FinalNorm,
  LayerNorm,      // attn_norm, ffn_norm, q_norm, kv_norm, compressor.norm, indexer.k_norm, draft norms
  Attention,      // wq_a, wq_b, wkv, wo_a, wo_b, attn_sink
  Indexer,        // indexer.wq_b, weights_proj, wk
  Compressor,     // compressor.wkv, wgate
  Router,         // ffn.gate.weight, .bias, .bias_vl
  SharedExpert,
  RoutedExpert,
  Mhc,            // hc_attn_fn/base/scale, hc_ffn_fn/base/scale
  Engram,         // engram.wkv, q_weight, k_weight
  EngramTable,    // engram.embed (the mmap'ed table)
  Draft,          // main_proj, main_norm, markov_head, confidence_head (the draft's own extras)
  Vision,         // vision.*, aligner.*, image_* — present-and-skipped
};

enum class Dsv41TensorRole : uint8_t {
  Plain,        // BF16 / F32 as stored
  Fp8Payload,   // F8_E4M3 [N, K]
  Fp8Scale,     // F8_E8M0 [N/32, K/32]
  Fp4Payload,   // I8 [N, K/2]
  Fp4Scale,     // F8_E8M0 [N, K/32]
  TablePayload, // F8_E4M3 [rows, 256] (the Engram table, never resident)
  TableScale,   // F8_E8M0 [rows, 8]
  Skipped,      // in the file, not loaded (vision)
  // The EXL3 quadruple (docs/deepseek_v41_flash_exl3_plan.md §1.1): one
  // matrix X with input width K and output width N is X.trellis I16
  // [K/16, N/16, 16 * bits], X.suh F16 [K], X.svh F16 [N], X.mul1 I32 [].
  Exl3Trellis,
  Exl3Suh,
  Exl3Svh,
  Exl3Marker,
};

struct Dsv41ExpectedTensor {
  std::string name;
  DType dtype{};
  std::vector<int64_t> shape;
  Dsv41WeightClass cls = Dsv41WeightClass::Attention;
  int layer = -1;   // layer index (draft stages at num_hidden_layers + S); -1 for globals
  int expert = -1;  // routed-expert id, -1 otherwise
  Dsv41TensorRole role = Dsv41TensorRole::Plain;

  size_t numel() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  size_t nbytes() const { return numel() * dtype_size(dtype); }
  bool skipped() const { return role == Dsv41TensorRole::Skipped; }
};

// The checkpoint name prefix of a layer: "layers.L." or "mtp.S.".
std::string dsv41_layer_prefix(const Dsv41TextConfig& cfg, int layer);

std::vector<Dsv41ExpectedTensor> dsv41_expected_text_tensors(const Dsv41TextConfig& cfg);
std::vector<Dsv41ExpectedTensor> dsv41_expected_layer_tensors(const Dsv41TextConfig& cfg, int layer);
std::vector<Dsv41ExpectedTensor> dsv41_expected_global_tensors(const Dsv41TextConfig& cfg);
// The vision tower's tensors (present-and-skipped); empty without a vision_config.
std::vector<Dsv41ExpectedTensor> dsv41_expected_vision_tensors(const Dsv41TextConfig& cfg);

struct Dsv41TensorDesc {
  DType dtype{};
  std::vector<int64_t> shape;
};

struct Dsv41BindReport {
  size_t expected = 0;
  size_t matched = 0;
  size_t missing = 0;
  size_t dtype_mismatch = 0;
  size_t shape_mismatch = 0;
  size_t unexpected = 0;
  size_t fp4_matrices = 0;      // routed + draft expert matrices
  size_t fp8_matrices = 0;      // dense projections
  size_t exl3_matrices = 0;     // EXL3 trellis matrices (the pack: every projection, expert and the head)
  size_t engram_tables = 0;
  size_t skipped = 0;           // vision tensors present
  std::vector<std::string> errors;
  // Tensors of backbone layers beyond the config's stack (a truncated
  // diagnostic forward over the first layers): counted, not errors.
  int beyond_stack = 0;
  bool ok() const {
    return missing == 0 && dtype_mismatch == 0 && shape_mismatch == 0 && unexpected == 0;
  }
};

Dsv41BindReport dsv41_validate_text_binding(
    const Dsv41TextConfig& cfg,
    const std::unordered_map<std::string, Dsv41TensorDesc>& present,
    size_t max_errors = 32);

// Tensor-parallel geometry acceptance (docs/deepseek_v41_flash_plan.md §2):
// throws std::invalid_argument naming the dim that does not divide. world
// must divide the attention heads (whole output groups per rank), the
// index heads, the vocabulary, the Engram hash heads and the expert /
// shared intermediate size, with every intermediate slice a multiple of
// 32 (the MXFP4 block and the fp8 block). Under the EXL3 format the
// intermediate and the vocabulary are split in whole 128-blocks instead
// (exl3_block_split: at least one block per rank), and the wo_b input
// slice must be a 128-multiple (the plan's §2.3).
void dsv41_tp_validate_geometry(const Dsv41TextConfig& cfg, int rank, int world);

}  // namespace dgpp
