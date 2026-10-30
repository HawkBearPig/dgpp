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

}  // namespace step_ref
