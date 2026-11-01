// The launchers of the reference kernels (step_reference_host.hpp).
#include "step_reference.cuh"

namespace step_ref {

void launch_gate_up(const dgpp::step::Mxfp4Expert& e, const dgpp::step::Mxfp4Group& g, const uint16_t* x, int H, int I, uint16_t* act,
                    float* gu, float* mag, cudaStream_t stream) {
  gate_up<<<(I + 127) / 128, 128, 0, stream>>>(e, g, x, H, I, act, gu, mag);
}
void launch_down(const dgpp::step::Mxfp4Expert& e, const dgpp::step::Mxfp4Group& g, const uint16_t* act, int H, int I, int spt, float* out,
                 float* mag, cudaStream_t stream) {
  down<<<H / 128, 128, 0, stream>>>(e, g, act, H, I, spt, out, mag);
}

void launch_fp8_gate_up(const dgpp::step::Fp8Expert& e, const dgpp::step::Mxfp4Group& g, const uint16_t* x, int H, int I, int block,
                        uint16_t* act, float* gu, float* mag, cudaStream_t stream) {
  fp8_gate_up<<<(I + 127) / 128, 128, 0, stream>>>(e, g, x, H, I, block, act, gu, mag);
}
void launch_fp8_down(const dgpp::step::Fp8Expert& e, const dgpp::step::Mxfp4Group& g, const uint16_t* act, int H, int I, int block, int spt,
                     float* out, float* mag, cudaStream_t stream) {
  fp8_down<<<H / 128, 128, 0, stream>>>(e, g, act, H, I, block, spt, out, mag);
}

void launch_mxfp4_row_gate_up(const dgpp::step::Mxfp4Expert& e, const dgpp::step::Mxfp4Group& g, const uint16_t* x, int H, int I, uint16_t* act,
                              float* gu, float* mag, cudaStream_t stream) {
  mxfp4_row_gate_up<<<(I + 127) / 128, 128, 0, stream>>>(e, g, x, H, I, act, gu, mag);
}
void launch_mxfp4_row_down(const dgpp::step::Mxfp4Expert& e, const dgpp::step::Mxfp4Group& g, const uint16_t* act, int H, int I, int spt,
                           float* out, float* mag, cudaStream_t stream) {
  mxfp4_row_down<<<H / 128, 128, 0, stream>>>(e, g, act, H, I, spt, out, mag);
}

void launch_fp8_matrix(const uint8_t* w, const float* scales, const dgpp::step::Mxfp4Group& g, const uint16_t* x, int x_stride, int K, int N,
                       int block, uint16_t* y, float* yf, float* mag, cudaStream_t stream) {
  fp8_matrix<<<(N + 127) / 128, 128, 0, stream>>>(w, scales, g, x, x_stride, K, N, block, y, yf, mag);
}

}  // namespace step_ref
