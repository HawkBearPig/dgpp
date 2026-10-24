#pragma once
// Double-precision oracle for the MoE module (DESIGN §7.4): fp32 router
// math, the bf16 expert chain up to the activations, then the engine's
// fp32 accumulation semantics in double — unrounded down dots, one fma per
// expert (ascending), shared last, one bf16 rounding of the sum (see
// glm_moe_layer.hpp). The strict path reproduces every bf16 rounding point
// the engine has, isolating the fp32-vs-double and mma-order gaps into the
// parity budgets.
#include <cstdint>
#include <vector>

#include "models/glm/moe.hpp"

namespace dgpp {

struct GlmMoeHostWeights {
  std::vector<uint16_t> router_gate;  // [E, H]
  std::vector<float> router_bias;     // [E]
  // Compressed expert/shared weights, byte-identical to what the engine
  // consumes (payload + block scales). FP8: 3 mats per expert then the
  // shared triple: gate, up, down. NVFP4 (`nvfp4` set): the routed experts
  // live in the fp4 vectors below (expert e matrix m at index e*3+m —
  // packed nibbles [rows, cols/2], e4m3 scales [rows, cols/16], one global
  // per matrix) and `payloads`/`scales` hold only the shared triple.
  std::vector<uint8_t> payloads;   // 3 mats per expert then shared: gate,up,down
  std::vector<float> scales;       // matching scale grids
  bool nvfp4 = false;
  // The shared expert in NVFP4 too (GLM-4.7): the fp4 vectors then hold
  // (E + 1) * 3 matrices, the shared triple last, and the fp8 vectors
  // are empty.
  bool shared_nvfp4 = false;
  std::vector<uint8_t> fp4_payloads;
  std::vector<uint8_t> fp4_scales;
  std::vector<float> fp4_globals;  // [E * 3] (or [(E + 1) * 3] with shared_nvfp4)
  // The fp4 scale group (2026-09-13): 16 = NVFP4 (e4m3 scales [rows,
  // cols/16] + a global per matrix), 32 = MXFP4 (e8m0 scales [rows,
  // cols/32], no globals — fp4_globals may be empty).
  int fp4_group = 16;
  // The per-projection form (2026-10-22, docs/flint_plan.md D2): with
  // `nvfp4`, [E * 3] flags naming the routed matrices that are fp4 (the
  // others fp8); empty = every routed matrix fp4. The fp4 vectors hold
  // the fp4 matrices in index order, the fp8 vectors the fp8 routed ones
  // in index order then the shared triple.
  std::vector<uint8_t> fp4_matrix;
  bool fp4_of(int index, int n_experts) const {
    if (!nvfp4) return false;
    if (index >= n_experts * 3) return shared_nvfp4;
    return fp4_matrix.empty() || fp4_matrix[static_cast<size_t>(index)] != 0;
  }
  // A matrix's position inside its format's vectors.
  int fp4_ordinal(int index, int n_experts) const {
    const int routed = n_experts * 3;
    if (fp4_matrix.empty()) return index;
    int o = 0;
    for (int i = 0; i < index && i < routed; ++i) o += fp4_matrix[static_cast<size_t>(i)] != 0;
    return index < routed ? o : o + (index - routed);
  }
  int fp8_ordinal(int index, int n_experts) const {
    const int routed = n_experts * 3;
    if (packq || (nvfp4 && fp4_matrix.empty())) return index < routed ? index : index - routed;
    if (!nvfp4) return index;
    int o = 0;
    for (int i = 0; i < index && i < routed; ++i) o += fp4_matrix[static_cast<size_t>(i)] == 0;
    return index < routed ? o : o + (index - routed);
  }
  int fp4_matrices(int n_experts) const {
    int routed = n_experts * 3;
    if (!fp4_matrix.empty()) {
      routed = 0;
      for (uint8_t f : fp4_matrix) routed += f != 0;
    }
    return routed + (shared_nvfp4 ? 3 : 0);
  }
  // The packed-int format (`packq` set, docs/glm53_plan.md D2): the routed
  // experts at packq_bits_routed and (with shared_packq) the shared triple
  // at packq_bits_shared, matrices in index order (expert e matrix m at
  // e*3+m, the shared triple last) — I32 words [rows, cols*bits/32] and
  // bf16 scales [rows, cols/64] per matrix; the fp8 vectors then hold the
  // shared triple only when shared_packq is false.
  bool packq = false;
  bool shared_packq = false;
  int packq_bits_routed = 4;
  int packq_bits_shared = 8;
  int packq_scale_fmt = 0;  // every routed matrix's scale format (kPackedScale*; the shared triple: 0)
  std::vector<uint32_t> packq_words;
  std::vector<uint16_t> packq_scales;
  int packq_matrices(int n_experts) const { return (n_experts + (shared_packq ? 1 : 0)) * 3; }
  // The Mixed346 contract (kMoeInputHadamard32Int8, 2026-10-09): per routed
  // matrix its own width and scale format — [E * 3] entries, 3 / 4 / 6
  // under kPackedScaleBf16G128Mixed346 or 4 under format 0 for an existing
  // expert; empty = every routed matrix at packq_bits_routed /
  // packq_scale_fmt. The words and scales are concatenated in matrix order
  // at each matrix's own size either way.
  std::vector<uint8_t> packq_matrix_bits;
  std::vector<uint8_t> packq_matrix_fmt;
  int packq_bits_of(int index, int n_experts) const {
    if (index >= n_experts * 3) return packq_bits_shared;
    return packq_matrix_bits.empty() ? packq_bits_routed : packq_matrix_bits[static_cast<size_t>(index)];
  }
  int packq_fmt_of(int index, int n_experts) const {
    if (index >= n_experts * 3) return 0;
    return packq_matrix_fmt.empty() ? packq_scale_fmt : packq_matrix_fmt[static_cast<size_t>(index)];
  }
  // The routed experts' input transform (kMoeInput*, moe.hpp): under
  // kMoeInputHadamard32 the oracle rotates every routed projection's
  // input by H32 (kernels/hadamard32.hpp) as the engine does; under
  // kMoeInputHadamard32Int8 it rotates and quantizes it to the int8 codes
  // (the represented values code x scale) for every format-3 expert, and
  // feeds a format-0 (existing) expert the plain rows.
  int routed_input_transform = 0;
  // The EXL3 format (2026-10-22, docs/deepseek_v41_flash_exl3_plan.md): every
  // matrix as a host EXL3 matrix in index order, the shared triple last
  // ((E + 1) * 3 entries; gate/up rows I cols H, down rows H cols I). The
  // oracle multiplies the full reconstruct (every rotation in double); the
  // fp8 / fp4 / packed vectors are unused when set.
  std::vector<Exl3HostMatrix> exl3;
  bool exl3_set() const { return !exl3.empty(); }
};

// Expert payload layout inside GlmMoeHostWeights: expert e's matrix m at
// index e*3+m, shared at index E*3+... see glm_moe_reference.cpp — helpers
// below return views so callers never index by hand.
struct GlmQuantMatrixHost {
  const uint8_t* payload = nullptr;
  const float* scales = nullptr;
  int64_t rows = 0;
  int64_t cols = 0;
};
struct GlmFp4MatrixHost {
  const uint8_t* payload = nullptr;  // [rows, cols/2]
  const uint8_t* scales = nullptr;   // e4m3 [rows, cols/16] or e8m0 [rows, cols/32]
  float global_scale = 1.0f;         // 1 under MXFP4 (no global)
  int64_t rows = 0;
  int64_t cols = 0;
  int scale_group = 16;
};

struct GlmPackedMatrixHost {
  const uint32_t* packed = nullptr;  // [rows, cols*bits/32]
  const uint16_t* scales = nullptr;  // bf16 [rows, cols/64] or f16 [rows, cols/128] (scale_fmt)
  int64_t rows = 0;
  int64_t cols = 0;
  int bits = 0;
  int scale_fmt = 0;                 // kPackedScale* (quant_matrix.hpp)
};

struct GlmMoeRouterRef {
  std::vector<int32_t> ids;       // [tokens, K] ascending
  std::vector<float> weights;     // [tokens, K] normalized + scaled
  std::vector<double> biased;     // [tokens, E] (near-tie certification)
};

// Router oracle. Same tie rule as the engine: biased descending, ties to the
// lower expert id.
void glm_moe_ref_router(const uint16_t* hidden, const uint16_t* gate,
                        const float* bias, const GlmMoeConfig& cfg,
                        int tokens, GlmMoeRouterRef& out);

// Full expert-path oracle (strict: bf16 rounding points exactly where the
// engine rounds): router -> per-expert swiglu MLPs in ascending order ->
// shared expert -> the chain's single rounding. hidden [tokens, H] bf16
// in, out [tokens, H] bf16 bits out.
void glm_moe_ref_forward(const uint16_t* hidden,
                         const GlmMoeHostWeights& w, const GlmMoeConfig& cfg,
                         int tokens, std::vector<uint16_t>& out);

// Build the host-weight views by LOGICAL index (expert e matrix m at
// e * 3 + m, the shared triple at E * 3 + m; the fp8 vectors' positions
// are fp8_ordinal's).
GlmQuantMatrixHost glm_moe_host_view(const GlmMoeHostWeights& w,
                                     const GlmMoeConfig& cfg, int index);
GlmQuantMatrixHost glm_moe_host_shared(const GlmMoeHostWeights& w,
                                       const GlmMoeConfig& cfg, int m);
// The NVFP4 expert view (index in [0, E*3), or [0, (E+1)*3) with
// shared_nvfp4; under the per-projection form an index fp4_of names);
// requires w.nvfp4.
GlmFp4MatrixHost glm_moe_host_view_fp4(const GlmMoeHostWeights& w,
                                       const GlmMoeConfig& cfg, int index);
// The packed-int expert view (index in [0, E*3), or [0, (E+1)*3) with
// shared_packq); requires w.packq.
GlmPackedMatrixHost glm_moe_host_view_packq(const GlmMoeHostWeights& w,
                                            const GlmMoeConfig& cfg, int index);

}  // namespace dgpp
