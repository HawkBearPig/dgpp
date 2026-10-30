#pragma once
// Reference kernels for the step engine's MXFP4 expert consumer: exact
// dequantization of the same tile-major bytes, accumulated in double, with
// the sum of the terms' magnitudes beside each result, the scale an fp32
// accumulation's error is bounded by. Runtime shapes.
#include <cmath>
#include <cstdint>

#include "common/dtypes.hpp"
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

}  // namespace step_ref
