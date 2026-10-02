#pragma once
// Qwen3.8-Flash-Next norm kernels (Q3, 2026-09-09; docs/qwen38_flash_next_plan.md
// §1.2, §1.3, §1.5). The family's norms are the zero-centered form: fp32
// interior, y = bf16(x * rsqrt(mean(x^2) + eps) * (1 + w)) — one rounding
// (transformers Qwen3_5RMSNorm), unlike GLM's two-rounding norm. The GDN
// output norm keeps the reference's three roundings (Qwen3NextRMSNormGated:
// bf16 after the normalization, after the plain-weight multiply, after the
// sigmoid gate). Deterministic reductions (fixed order) everywhere.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// y[rows, dim] = rmsnorm(x) * (1 + w); bf16 io, fp32 inside, one rounding.
void qwen_rmsnorm_bf16(const void* x, const void* weight, void* y, int64_t rows,
                       int dim, float eps, cudaStream_t stream);

// Fused residual-add + rmsnorm, bitwise the add_inplace + qwen_rmsnorm
// pair: resid[] = bf16(resid[] + add[]) elementwise, then the identical
// reduction order (qwen_norm::block_sum_squares' strided fmaf chains),
// rsqrt form and single-rounding (1+w) epilogue. One launch instead of two.
void qwen_add_rmsnorm_bf16(void* resid, const void* add, const void* weight, void* y,
                           int64_t rows, int dim, float eps, cudaStream_t stream);

// The grouped form over a row of `groups` x `group_dim` (the 4-branch hyper
// state, group_dim = hidden): every group normalized on its own, the weight
// [groups * group_dim] indexed by the flat position. bf16 io.
void qwen_group_rmsnorm_bf16(const void* x, const void* weight, void* y,
                             int64_t rows, int groups, int group_dim, float eps,
                             cudaStream_t stream);

// The GDN output norm: y = bf16(bf16(bf16(x * rstd) * w) * sigmoid(gate)),
// one row per (token, head) of width dim; w is the plain weight (init 1).
void gdn_gated_rmsnorm_bf16(const void* x, const void* gate, const void* weight,
                             void* y, int64_t rows, int dim, float eps,
                             cudaStream_t stream);

// The swish-gated form (Qwen3.5): y = bf16(bf16(bf16(x * rstd) * w) *
// (gate * sigmoid(gate))) — same three roundings, one extra fp32 multiply,
// single final rounding.
void gdn_gated_rmsnorm_swish_bf16(const void* x, const void* gate, const void* weight,
                                  void* y, int64_t rows, int dim, float eps,
                                  cudaStream_t stream);

}  // namespace dgpp
