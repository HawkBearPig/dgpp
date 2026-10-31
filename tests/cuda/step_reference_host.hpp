#pragma once
// The host side of the step engine's consumer references: the launchers of
// the double-precision reference kernels (tests/cuda/step_reference.cuh,
// built as CUDA) and the acceptance criteria. Plain C++, so a test that
// also parses configurations can include it.
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "kernels/step/fp8_dense.cuh"
#include "kernels/step/mxfp4_expert.cuh"

namespace step_ref {

inline float bf16_ulp(float v);

// One group's gate and up over every row of the expert into act, with the
// exact gate and up (gu, interleaved) and the terms' magnitudes (mag).
void launch_gate_up(const dgpp::step::Mxfp4Expert& e, const dgpp::step::Mxfp4Group& g, const uint16_t* x, int H, int I, uint16_t* act,
                    float* gu, float* mag, cudaStream_t stream);
// One group's down over every row from `act` into the token-major partials and their magnitudes.
void launch_down(const dgpp::step::Mxfp4Expert& e, const dgpp::step::Mxfp4Group& g, const uint16_t* act, int H, int I, int spt, float* out,
                 float* mag, cudaStream_t stream);
// The row-major MXFP4 pair and down.
void launch_mxfp4_row_gate_up(const dgpp::step::Mxfp4Expert& e, const dgpp::step::Mxfp4Group& g, const uint16_t* x, int H, int I, uint16_t* act,
                              float* gu, float* mag, cudaStream_t stream);
void launch_mxfp4_row_down(const dgpp::step::Mxfp4Expert& e, const dgpp::step::Mxfp4Group& g, const uint16_t* act, int H, int I, int spt,
                           float* out, float* mag, cudaStream_t stream);
// One row-major fp8 matrix for a token group, y as bf16 with the exact fp32 and the magnitudes beside it.
void launch_fp8_matrix(const uint8_t* w, const float* scales, const dgpp::step::Mxfp4Group& g, const uint16_t* x, int x_stride, int K, int N,
                       int block, uint16_t* y, float* yf, float* mag, cudaStream_t stream);
// A bf16 result against the exact value: within two ulps, or inside the accumulation envelope widened by two ulps.
inline bool value_ok(uint16_t got_bits, float exact, float magnitude) {
  const float got = dgpp::bf16_bits_to_float(got_bits);
  const float ref = dgpp::bf16_bits_to_float(dgpp::float_to_bf16_bits(exact));
  if (got_bits == dgpp::float_to_bf16_bits(exact)) return true;
  const float slack = 2.0f * bf16_ulp(std::fabs(ref) > std::fabs(got) ? std::fabs(ref) : std::fabs(got)) + magnitude * 4e-6f + 1e-7f;
  return std::fabs(got - exact) <= slack;
}
// The row-major fp8 pair and down with F32 scales on block x block blocks.
void launch_fp8_gate_up(const dgpp::step::Fp8Expert& e, const dgpp::step::Mxfp4Group& g, const uint16_t* x, int H, int I, int block,
                        uint16_t* act, float* gu, float* mag, cudaStream_t stream);
void launch_fp8_down(const dgpp::step::Fp8Expert& e, const dgpp::step::Mxfp4Group& g, const uint16_t* act, int H, int I, int block, int spt,
                     float* out, float* mag, cudaStream_t stream);

// The host's copy of the epilogue, for the error envelope.
inline uint16_t host_swiglu(float gate, float up) {
  float g = dgpp::bf16_bits_to_float(dgpp::float_to_bf16_bits(gate)), u = dgpp::bf16_bits_to_float(dgpp::float_to_bf16_bits(up));
  if (g > dgpp::step::kMxfp4SwigluLimit) g = dgpp::step::kMxfp4SwigluLimit;
  u = u < -dgpp::step::kMxfp4SwigluLimit ? -dgpp::step::kMxfp4SwigluLimit : (u > dgpp::step::kMxfp4SwigluLimit ? dgpp::step::kMxfp4SwigluLimit : u);
  const uint16_t t = dgpp::float_to_bf16_bits(g * (1.0f / (1.0f + std::exp(-g))));
  return dgpp::float_to_bf16_bits(dgpp::bf16_bits_to_float(t) * u);
}
inline float bf16_ulp(float v) {
  int e = 0;
  std::frexp(std::fabs(v), &e);
  return std::ldexp(1.0f, e - 8);
}
// Whether a consumer's bf16 SwiGLU output `got` is acceptable against the
// exact gate and up: within two ulps of the exact result, or inside the
// envelope an fp32 accumulation's error allows widened by two ulps.
inline bool act_ok(uint16_t got_bits, uint16_t ref_bits, float g, float u, float mag_g, float mag_u) {
  if (got_bits == ref_bits) return true;
  if (std::abs(static_cast<int>(got_bits) - static_cast<int>(ref_bits)) <= 2) return true;
  const float dg = mag_g * 4e-6f + 1e-7f, du = mag_u * 4e-6f + 1e-7f;
  float lo = 1e30f, hi = -1e30f;
  for (int c = 0; c < 4; ++c) {
    const float v = dgpp::bf16_bits_to_float(host_swiglu(g + ((c & 1) ? dg : -dg), u + ((c & 2) ? du : -du)));
    lo = lo < v ? lo : v;
    hi = hi > v ? hi : v;
  }
  const float slack = 2.0f * bf16_ulp(std::fabs(lo) > std::fabs(hi) ? std::fabs(lo) : std::fabs(hi));
  const float got = dgpp::bf16_bits_to_float(got_bits);
  return got >= lo - slack && got <= hi + slack;
}
inline bool partial_ok(float got, float ref, float mag) { return std::fabs(got - ref) <= mag * 4e-6f + 1e-7f; }

}  // namespace step_ref
