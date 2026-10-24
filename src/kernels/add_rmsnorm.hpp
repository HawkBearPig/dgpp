#pragma once
// The residual add fused with the two-rounding RMSNorm that follows it
// (2026-09-22, docs/mimo_v26_flash_plan.md §7.1): one launch per block
// boundary instead of glm_residual_add_bf16 + glm_rmsnorm_bf16, with the
// row held in registers between the add and the norm (the two-kernel chain
// writes the residual, reads it back, and stages it through shared memory
// twice). The arithmetic is the chain's, op for op:
//
//   resid[r, :] = bf16(resid + add)             (add == nullptr: no add)
//   rstd        = rsqrtf(float(sum_d resid^2 / dim) + eps)   (fp64 sum)
//   out[r, :]   = bf16(w * bf16(resid * rstd))
//
// The fp64 sum of squares runs in a different order from the two-kernel
// chain's (a warp-shuffle tree over 256 threads instead of a 512-wide
// shared-memory tree). Every term is exact in fp64 (a bf16 square has 16
// significant bits) and partial sums of 4096 such terms stay exact until
// their exponent range exceeds 37 bits, so the fp64 result — and the fp32
// rstd rounded from it — is the chain's for any row a model produces; the
// unit test pins the equality on random rows.
//
// Contract: dim a multiple of 8, at most 8192; rows 16-byte aligned (row
// stride dim). Deterministic and capturable.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

constexpr int kAddRmsnormMaxDim = 8192;

void add_rmsnorm_bf16(uint16_t* resid, const uint16_t* add, const uint16_t* weight, uint16_t* out,
                      int rows, int dim, float eps, cudaStream_t stream);

// The one-rounding form (2026-10-22, docs/flint_plan.md D4 — Flint's
// norms, the released reference's rmsnorm): the same add and fp64 sum,
// then out[r, :] = bf16((resid * rstd) * w) in fp32 with ONE rounding.
void add_rmsnorm_bf16_one(uint16_t* resid, const uint16_t* add, const uint16_t* weight, uint16_t* out,
                          int rows, int dim, float eps, cudaStream_t stream);

}  // namespace dgpp
