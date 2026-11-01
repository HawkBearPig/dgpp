#pragma once
// Reference kernels for the step engine's MXFP4 expert consumer: exact
// dequantization of the same tile-major bytes, accumulated in double, with
// the sum of the terms' magnitudes beside each result, the scale an fp32
// accumulation's error is bounded by. Runtime shapes.
#include <cmath>
#include <cstdint>

#include "common/dtypes.hpp"
#include "kernels/step/fp8_dense.cuh"
#include "kernels/step/mxfp4_row.cuh"
#include "kernels/step/mxfp4_expert.cuh"
#include "step_reference_host.hpp"

namespace step_ref {

__device__ inline float e2m1_value(uint32_t code) {
  const int em = (code >> 1) & 3, mm = code & 1;
  float v = em == 0 ? 0.5f * mm : ldexpf(1.0f + 0.5f * mm, em - 1);
  return (code & 8) ? -v : v;
}

// One group's gate and up over every row: act[slot][row] through the SwiGLU,
// gu[(slot * I + row) * 2 + m] the exact gate (m 0) and up (m 1) as fp32,
// mag the terms' magnitudes. Grid: (ceil(I / 128), 1), 128 threads.
__global__ void gate_up(dgpp::step::Mxfp4Expert e, dgpp::step::Mxfp4Group g, const uint16_t* x, int H, int I, uint16_t* act, float* gu,
                        float* mag) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= I) return;
  for (int r = 0; r < g.count; ++r) {
    const int slot = g.slots[r], tok = g.tokens[r];
    float d[2];
    for (int m = 0; m < 2; ++m) {
      const uint8_t* pay = m ? e.up : e.gate;
      const uint8_t* sc = (m ? e.up_scales : e.gate_scales) + static_cast<size_t>(row) * (H / 32);
      double a = 0.0, mg = 0.0;
      for (int k = 0; k < H; ++k) {
        const uint32_t code = (pay[dgpp::step::tile_major_offset(row, k, H)] >> (4 * (k & 1))) & 0xF;
        const double term = static_cast<double>(ldexpf(e2m1_value(code), static_cast<int>(sc[k >> 5]) - 127)) *
                            static_cast<double>(dgpp::bf16_bits_to_float(x[static_cast<size_t>(tok) * H + k]));
        a += term;
        mg += fabs(term);
      }
      d[m] = static_cast<float>(a);
      gu[(static_cast<size_t>(slot) * I + row) * 2 + m] = d[m];
      mag[(static_cast<size_t>(slot) * I + row) * 2 + m] = static_cast<float>(mg);
    }
    act[static_cast<size_t>(slot) * I + row] = dgpp::step::swiglu_bf16(d[0], d[1]);
  }
}

// One group's down over every row: out[idx] the exact partial as fp32 and
// mag[idx] the terms' magnitudes, idx = step_out_index. Grid: (H / 128, 1).
__global__ void down(dgpp::step::Mxfp4Expert e, dgpp::step::Mxfp4Group g, const uint16_t* act, int H, int I, int spt, float* out,
                     float* mag) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= H) return;
  const uint8_t* sc = e.down_scales + static_cast<size_t>(row) * (I / 32);
  for (int r = 0; r < g.count; ++r) {
    const int slot = g.slots[r];
    double a = 0.0, m = 0.0;
    for (int k = 0; k < I; ++k) {
      const uint32_t code = (e.down[dgpp::step::tile_major_offset(row, k, I)] >> (4 * (k & 1))) & 0xF;
      const double term = static_cast<double>(ldexpf(e2m1_value(code), static_cast<int>(sc[k >> 5]) - 127)) *
                          static_cast<double>(dgpp::bf16_bits_to_float(act[static_cast<size_t>(slot) * I + k]));
      a += term;
      m += fabs(term);
    }
    const size_t idx = (static_cast<size_t>(slot / spt) * H + row) * spt + (slot % spt);
    out[idx] = static_cast<float>(a);
    mag[idx] = static_cast<float>(m);
  }
}

// e4m3 as the exact value (no NaN in weights).
__device__ inline float e4m3_value(uint32_t b) {
  const int e = (b >> 3) & 0xF, m = b & 7;
  float v = e == 0 ? ldexpf(static_cast<float>(m) / 8.0f, -6) : ldexpf(1.0f + m / 8.0f, e - 7);
  return (b & 0x80) ? -v : v;
}

// The row-major fp8 pair with F32 block scales (block x block), one group.
__global__ void fp8_gate_up(dgpp::step::Fp8Expert e, dgpp::step::Mxfp4Group g, const uint16_t* x, int H, int I, int block, uint16_t* act,
                            float* gu, float* mag) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= I) return;
  const int sc_cols = H / block;
  for (int r = 0; r < g.count; ++r) {
    const int slot = g.slots[r], tok = g.tokens[r];
    float d[2];
    for (int m = 0; m < 2; ++m) {
      const uint8_t* pay = (m ? e.up : e.gate) + static_cast<size_t>(row) * H;
      const float* sc = (m ? e.up_scales : e.gate_scales) + static_cast<size_t>(row / block) * sc_cols;
      double a = 0.0, mg = 0.0;
      for (int k = 0; k < H; ++k) {
        const double term = static_cast<double>(e4m3_value(pay[k])) * static_cast<double>(sc[k / block]) *
                            static_cast<double>(dgpp::bf16_bits_to_float(x[static_cast<size_t>(tok) * H + k]));
        a += term;
        mg += fabs(term);
      }
      d[m] = static_cast<float>(a);
      gu[(static_cast<size_t>(slot) * I + row) * 2 + m] = d[m];
      mag[(static_cast<size_t>(slot) * I + row) * 2 + m] = static_cast<float>(mg);
    }
    act[static_cast<size_t>(slot) * I + row] = dgpp::step::swiglu_bf16(d[0], d[1]);
  }
}
__global__ void fp8_down(dgpp::step::Fp8Expert e, dgpp::step::Mxfp4Group g, const uint16_t* act, int H, int I, int block, int spt,
                         float* out, float* mag) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= H) return;
  const int sc_cols = (I + block - 1) / block;
  const uint8_t* pay = e.down + static_cast<size_t>(row) * I;
  const float* sc = e.down_scales + static_cast<size_t>(row / block) * sc_cols;
  for (int r = 0; r < g.count; ++r) {
    const int slot = g.slots[r];
    double a = 0.0, m = 0.0;
    for (int k = 0; k < I; ++k) {
      const double term = static_cast<double>(e4m3_value(pay[k])) * static_cast<double>(sc[k / block]) *
                          static_cast<double>(dgpp::bf16_bits_to_float(act[static_cast<size_t>(slot) * I + k]));
      a += term;
      m += fabs(term);
    }
    const size_t idx = (static_cast<size_t>(slot / spt) * H + row) * spt + (slot % spt);
    out[idx] = static_cast<float>(a);
    mag[idx] = static_cast<float>(m);
  }
}

// The row-major MXFP4 pair and down: payload [rows, K / 2], e8m0 scales [rows, K / 32].
__global__ void mxfp4_row_gate_up(dgpp::step::Mxfp4Expert e, dgpp::step::Mxfp4Group g, const uint16_t* x, int H, int I, uint16_t* act,
                                  float* gu, float* mag) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= I) return;
  for (int r = 0; r < g.count; ++r) {
    const int slot = g.slots[r], tok = g.tokens[r];
    float d[2];
    for (int m = 0; m < 2; ++m) {
      const uint8_t* pay = (m ? e.up : e.gate) + static_cast<size_t>(row) * (H / 2);
      const uint8_t* sc = (m ? e.up_scales : e.gate_scales) + static_cast<size_t>(row) * (H / 32);
      double a = 0.0, mg = 0.0;
      for (int k = 0; k < H; ++k) {
        const uint32_t code = (pay[k / 2] >> (4 * (k & 1))) & 0xF;
        const double term = static_cast<double>(ldexpf(e2m1_value(code), static_cast<int>(sc[k >> 5]) - 127)) *
                            static_cast<double>(dgpp::bf16_bits_to_float(x[static_cast<size_t>(tok) * H + k]));
        a += term;
        mg += fabs(term);
      }
      d[m] = static_cast<float>(a);
      gu[(static_cast<size_t>(slot) * I + row) * 2 + m] = d[m];
      mag[(static_cast<size_t>(slot) * I + row) * 2 + m] = static_cast<float>(mg);
    }
    act[static_cast<size_t>(slot) * I + row] = dgpp::step::swiglu_bf16(d[0], d[1]);
  }
}
__global__ void mxfp4_row_down(dgpp::step::Mxfp4Expert e, dgpp::step::Mxfp4Group g, const uint16_t* act, int H, int I, int spt, float* out,
                               float* mag) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= H) return;
  const uint8_t* pay = e.down + static_cast<size_t>(row) * (I / 2);
  const uint8_t* sc = e.down_scales + static_cast<size_t>(row) * (I / 32);
  for (int r = 0; r < g.count; ++r) {
    const int slot = g.slots[r];
    double a = 0.0, m = 0.0;
    for (int k = 0; k < I; ++k) {
      const uint32_t code = (pay[k / 2] >> (4 * (k & 1))) & 0xF;
      const double term = static_cast<double>(ldexpf(e2m1_value(code), static_cast<int>(sc[k >> 5]) - 127)) *
                          static_cast<double>(dgpp::bf16_bits_to_float(act[static_cast<size_t>(slot) * I + k]));
      a += term;
      m += fabs(term);
    }
    const size_t idx = (static_cast<size_t>(slot / spt) * H + row) * spt + (slot % spt);
    out[idx] = static_cast<float>(a);
    mag[idx] = static_cast<float>(m);
  }
}

// One row-major fp8 matrix for a token group: y[slot * N + row] as bf16,
// the exact value as fp32 beside it (yf) and the terms' magnitudes (mag).
__global__ void fp8_matrix(const uint8_t* w, const float* scales, dgpp::step::Mxfp4Group g, const uint16_t* x, int x_stride, int K, int N,
                           int block, uint16_t* y, float* yf, float* mag) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= N) return;
  const int sc_cols = K / block;
  const uint8_t* pay = w + static_cast<size_t>(row) * K;
  const float* sc = scales + static_cast<size_t>(row / block) * sc_cols;
  for (int r = 0; r < g.count; ++r) {
    const int slot = g.slots[r], tok = g.tokens[r];
    double a = 0.0, m = 0.0;
    for (int k = 0; k < K; ++k) {
      const double term = static_cast<double>(e4m3_value(pay[k])) * static_cast<double>(sc[k / block]) *
                          static_cast<double>(dgpp::bf16_bits_to_float(x[static_cast<size_t>(tok) * x_stride + k]));
      a += term;
      m += fabs(term);
    }
    const size_t idx = static_cast<size_t>(slot) * N + row;
    y[idx] = dgpp::float_to_bf16_bits(static_cast<float>(a));
    yf[idx] = static_cast<float>(a);
    mag[idx] = static_cast<float>(m);
  }
}

}  // namespace step_ref
