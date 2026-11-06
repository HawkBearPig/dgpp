#include <cstdio>
#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"
#include "kernels/minimax_native_attention.cuh"

namespace {
__global__ void decode(const uint8_t* input, uint32_t* output, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) output[i] = dgpp::minimax_cached_fp8x4(input + i * 72, (i % 32) * 4);
}
}  // namespace

int main() {
  try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 2;
    constexpr int count = 127 * 256;
    constexpr float magnitudes[] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    std::vector<uint8_t> input(count * 72);
    std::vector<uint32_t> expected(count), actual(count);
    for (int i = 0; i < count; ++i) {
      const int dim = (i % 32) * 4, pair = i % 256, scale_bits = i / 256;
      const uint16_t codes = static_cast<uint16_t>(pair | ((pair ^ 0xa5) << 8));
      input[i * 72 + dim / 2] = static_cast<uint8_t>(codes);
      input[i * 72 + dim / 2 + 1] = static_cast<uint8_t>(codes >> 8);
      input[i * 72 + 64 + dim / 16] = static_cast<uint8_t>(scale_bits);
      __nv_fp8_e4m3 scale;
      scale.__x = static_cast<uint8_t>(scale_bits);
      for (int j = 0; j < 4; ++j) {
        const int code = (codes >> (j * 4)) & 15;
        const float value = (code & 8 ? -magnitudes[code & 7] : magnitudes[code & 7]) *
                            static_cast<float>(scale);
        expected[i] |= static_cast<uint32_t>(__nv_fp8_e4m3(value).__x) << (j * 8);
      }
    }
    uint8_t* device_input = nullptr;
    uint32_t* device_output = nullptr;
    DGPP_CUDA_OK(cudaMalloc(&device_input, input.size()));
    DGPP_CUDA_OK(cudaMalloc(&device_output, count * sizeof(uint32_t)));
    DGPP_CUDA_OK(cudaMemcpy(device_input, input.data(), input.size(), cudaMemcpyHostToDevice));
    decode<<<(count + 127) / 128, 128>>>(device_input, device_output, count);
    DGPP_CUDA_OK(cudaGetLastError());
    DGPP_CUDA_OK(cudaMemcpy(actual.data(), device_output, count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaFree(device_input));
    DGPP_CUDA_OK(cudaFree(device_output));
    if (actual != expected) throw std::runtime_error("NVFP4 cache reconstruction changed an FP8 byte");
    std::printf("NVFP4 cache: all 256 code pairs x 127 finite scales, both word halves and all groups exact\n");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "minimax_nvfp4_decode_test: %s\n", error.what());
    return 1;
  }
}
