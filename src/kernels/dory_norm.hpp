#pragma once
// The Dory nGPT normalizations (docs/dory_plan.md §1.2, §4): `justnorm` and
// the fused residual that uses it twice, one launch per site instead of a
// chain of elementwise passes.
//
// The arithmetic is the host reference's (models/dory/reference.cpp), op for
// op, because that reference is the only executable description of the model:
//
//   justnorm(x)           = bf16( fp32(x) * rstd ),  rstd = 1/sqrt(sum fp32(x)^2 + eps)
//   slerp_alpha(a, b, α, s) = justnorm( bf16(a·ra) + lr · (bf16(b·rb) − bf16(a·ra)) )
//
// with ra, rb the two input rows' rstds and lr = |fp32(α) · s| per column
// (α is the layer's `attn_alpha` / `mlp_alpha`, broadcast along rows). The
// operands normalize to bf16, but the interpolation and its output
// normalization stay fp32 until the final conversion back to bf16.
//
// The row sum is accumulated in fp64 (the add_rmsnorm trick): every bf16
// square is exact in fp64, so the sum is the exact one and `float(sum) + eps`
// rounds once. The reference's sequential fp32 fma chain differs only in the
// rounding of the additions, which the bf16 output absorbs — pinned on random
// rows of every width the model uses by tests/cuda/dory_norm_test.cu.
//
// WIDTHS (plan §6): the attention-output justnorm spans ALL query heads
// (dim = q_size = 4096) — a per-head implementation is a plausible silent
// error; the one before the `sqk` square spans one head (dim = head_dim =
// 256); the MLP's spans the intermediate (7168).
//
// Contract: dim a multiple of 8, at most 8192; rows 16-byte aligned (row
// stride dim). Deterministic and capturable.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

constexpr int kDoryNormMaxDim = 8192;

// y[r, :] = bf16(fp32(x[r, :]) * rstd), rstd = 1/sqrt(sum_k fp32(x)^2 + eps).
void dory_justnorm_bf16(const uint16_t* x, uint16_t* y, int rows, int dim, float eps,
                        cudaStream_t stream);

// The walk's staging copy as a kernel (a capture-clean alternative to a
// device-to-device cudaMemcpy; elems must be a multiple of 8).
void dory_copy_rows_bf16(const uint16_t* src, uint16_t* dst, size_t elems, cudaStream_t stream);

// y[r, :] = justnorm(bf16(a·ra) + lr·(bf16(b·rb) − bf16(a·ra))).
void dory_slerp_alpha_bf16(const uint16_t* a, const uint16_t* b, const uint16_t* alpha, float scale,
                           uint16_t* y, int rows, int dim, float eps, cudaStream_t stream);

// The loop's re-injection (plan §1.5): y[r, :] = justnorm(a[r, :] + b[r, :]) with
// the add in fp32 and one rounding, the sum running over the fp32 sums —
// `justnorm(state + input_h)` is a plain add then re-normalize.
void dory_add_justnorm_bf16(const uint16_t* a, const uint16_t* b, uint16_t* y, int rows, int dim,
                            float eps, cudaStream_t stream);

// The attention output's gate, fused with the justnorm that follows it over
// ALL query heads (plan §1.3):
//
//   y_i = bf16( g_i · rstd ),  g_i = fp32(attn_i) · sigmoid( fp32(gate_i) · s_i² ),
//   s_i = fp32(s_gate_i) · s_gate_factor,  rstd = 1/sqrt(sum g² + eps)
//
// attn bf16 [rows, dim] (the attention's own bf16 output), gate bf16
// [rows, dim] (the `g_proj` GEMM's, whose one rounding the engine does in the
// GEMM — the reference's `fp32(gate)` lifts a bf16 Linear output, exactly like
// launch_moe_swiglu_clamp's gate/up), s_gate bf16 [dim]. The interior stays
// fp32 (the reference's `fp32(attn) ⊙ sigmoid(...)`), so the row sum runs
// over the fp32 products and only the output rounds.
void dory_gate_justnorm_bf16(const uint16_t* attn, const uint16_t* gate, const uint16_t* s_gate,
                             float s_gate_factor, uint16_t* y, int rows, int dim, float eps,
                             cudaStream_t stream);

// The MLP's activation, fused with the justnorm that follows it over the
// intermediate (plan §1.4):
//
//   y_i = bf16( m_i · rstd ),  m_i = silu( g_i ) · u_i,  rstd = 1/sqrt(sum m² + eps),
//   g_i = fp32(gate_proj_i) · fp32(suv_gate_i) · suv_factor,
//   u_i = fp32(up_proj_i)   · fp32(suv_up_i)   · suv_factor
//
// g_in / u_in bf16 [rows, dim] (the two GEMMs' outputs, as
// launch_moe_swiglu_clamp takes them), suv_gate / suv_up bf16 [dim]. fp32
// interior, one rounding.
void dory_silu_justnorm_bf16(const uint16_t* g_in, const uint16_t* u_in, const uint16_t* suv_gate,
                             const uint16_t* suv_up, float suv_factor, uint16_t* y, int rows,
                             int dim, float eps, cudaStream_t stream);

}  // namespace dgpp
