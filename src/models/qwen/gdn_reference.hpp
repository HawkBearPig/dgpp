#pragma once
// Host reference of the Gated DeltaNet pieces: the test
// oracle for gdn_recurrent_fwd and the GDN output norm, and the executable
// specification of the reference numerics (transformers
// Qwen3_5GatedDeltaNet: torch_recurrent_gated_delta_rule with
// use_qk_l2norm_in_kernel, Qwen3NextRMSNormGated with the sigmoid gate).
// Acc=float mirrors the device fp32 math; Acc=double measures drift.
#include <cstdint>

namespace dgpp::gdn_ref {

// The recurrence (see gdn_recurrent_fwd for the layouts). State is
// Acc [heads, v_dim, k_dim], k contiguous, updated in place.
template <typename Acc>
void recurrent(const uint16_t* qkv, const uint16_t* a_raw, int64_t a_row_stride,
               const uint16_t* beta_raw, int64_t beta_row_stride,
               const float* a_log, const float* dt_bias, Acc* state,
               uint16_t* out, int tokens, int heads, int kv_ratio, int k_dim,
               int v_dim, float scale);

// The output norm with the reference's rounding sequence:
//   u = bf16(x * rsqrt(mean(x^2) + eps))   (fp32 norm, cast to bf16)
//   p = bf16(u * w)                        (bf16 * bf16, one rounding)
//   y = bf16(p * sigmoid(gate))            (bf16 * fp32 -> fp32, cast)
// x/gate/y: bf16 [rows, dim]; w: bf16 [dim] (plain, init 1).
void gated_rmsnorm_sigmoid(const uint16_t* x, const uint16_t* gate,
                           const uint16_t* w, uint16_t* y, int64_t rows,
                           int dim, float eps);

// The swish-gated form (Qwen3.5 output_gate_type): y = bf16(p * (gate *
// sigmoid(gate))) — same three roundings, the gate applied unrounded in fp32
// with a single final cast.
void gated_rmsnorm_swish(const uint16_t* x, const uint16_t* gate,
                         const uint16_t* w, uint16_t* y, int64_t rows,
                         int dim, float eps);

}  // namespace dgpp::gdn_ref
