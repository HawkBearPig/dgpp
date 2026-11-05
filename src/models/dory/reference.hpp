#pragma once
// Host reference of the Dory nGPT primitives (docs/dory_plan.md §1.2),
// transcribed from the vLLM port (PR #60149, dory.py) — the only executable
// description of the architecture. fp32 interior, ONE rounding back to BF16,
// exactly as the reference computes:
//
//   justnorm(x)          = bf16( fp32(x) * rsqrt(sum_k fp32(x)^2 + eps) )
//   slerp_norm(a, b, lr) = justnorm( justnorm(a) + lr * (justnorm(b) - justnorm(a)) )
//   slerp_alpha(a, b, alpha, s) = slerp_norm(a, b, |alpha * s|)
//
// There is no LayerNorm or RMSNorm anywhere in this model: `justnorm` is the
// only normalization. `eps` is 1e-6 (kDoryJustnormEps). The sum runs over the
// LAST dimension as a sequential fp32 fma chain.
//
// The WIDTH is part of the contract (plan §6): the attention-output justnorm
// spans ALL query heads (width q_size = 4096), the per-head justnorm before
// the `sqk` square spans one head (width head_dim = 256), and the MLP's spans
// the intermediate (7168). A per-head implementation of the wide one is a
// plausible silent error, so the reference and the tests pin each width.
#include <cstdint>

namespace dgpp {

inline constexpr float kDoryJustnormEps = 1e-6f;

namespace dory_ref {

// One row: y[i] = bf16(fp32(x[i]) * rsqrt(sum_j fp32(x[j])^2 + eps)).
void justnorm_row(const uint16_t* x, uint16_t* y, int64_t dim, float eps = kDoryJustnormEps);

// `rows` independent rows of `dim`.
void justnorm(const uint16_t* x, uint16_t* y, int64_t rows, int64_t dim,
              float eps = kDoryJustnormEps);

// The loop's re-injection (plan §1.5): y = justnorm(a + b) with the add in
// fp32 — `justnorm(state + input_h)` is a plain add then re-normalize.
void add_justnorm(const uint16_t* a, const uint16_t* b, uint16_t* y, int64_t rows, int64_t dim,
                  float eps = kDoryJustnormEps);

// The operands normalize to bf16, then the interpolation and its final
// normalization stay fp32 until the output conversion to bf16.
// y = justnorm(justnorm(a) + lr * (justnorm(b) - justnorm(a))) with one fp32
// `lr` per row (the residual scales are per-channel, broadcast along rows).
void slerp_norm_rows(const uint16_t* a, const uint16_t* b, const float* lr, uint16_t* y,
                     int64_t rows, int64_t dim, float eps = kDoryJustnormEps);

// The same with one fp32 lr per column (`dim` entries).
void slerp_norm_cols(const uint16_t* a, const uint16_t* b, const float* lr_col, uint16_t* y,
                     int64_t rows, int64_t dim, float eps = kDoryJustnormEps);

// slerp_alpha: lr[col] = |fp32(alpha[col]) * scale| — the residual form the
// attention and MLP blocks use (alpha is the learned attn_alpha / mlp_alpha).
void slerp_alpha(const uint16_t* a, const uint16_t* b, const uint16_t* alpha, float scale,
                 uint16_t* y, int64_t rows, int64_t dim, float eps = kDoryJustnormEps);

}  // namespace dory_ref
}  // namespace dgpp
