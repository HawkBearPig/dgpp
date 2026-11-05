#pragma once
// The Dory head's per-vocabulary scale (docs/dory_plan.md §1.6, D9):
//
//   logits = fp32(h) @ fp32(lm_head.weight)^T ⊙ (logit_scale · logit_factor)
//
// `lm_head.logit_scale` is a BF16 [vocab] vector — one learned scale per
// vocabulary entry, not per row — and `logit_factor = 1/initializer_range`
// normalizes it the way the config normalizes every other learned nGPT scale
// (plan §1.1). The projection itself is the engine's BF16 GEMM with fp32
// accumulators (the head's logits stay unrounded for sampling); this kernel
// applies the scale to that fp32 result in place.
//
// Deterministic and capturable: one thread per (row, column), grid-strided.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// logits[r, v] *= fp32(logit_scale[v]) * factor, in place.
void dory_logits_scale_f32(float* logits, const uint16_t* logit_scale, float factor, int rows,
                           int vocab, cudaStream_t stream);

}  // namespace dgpp
