// The DeepSeek release's activation quantize-dequantize (kernels/act_quant.hpp)
// against a host reference written from inference/kernel.py's act_quant:
// bitwise on plain rows (random magnitudes, zero and sub-floor groups, values
// on the scale boundary, in place and with strides), and for the SwiGLU form
// the grid property (its rows are fixed points of the quantizer) plus the
// distance from the fp32 product (the device's expf may differ from the
// host's in the last place, so that form is bounded rather than bitwise).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp8.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/act_quant.hpp"

using namespace dgpp;
namespace {
template <class T>
struct Buffer {
  T* ptr = nullptr;
  size_t count;
  explicit Buffer(size_t size) : count(size) { DGPP_CUDA_OK(cudaMalloc(&ptr, size * sizeof(T))); }
  explicit Buffer(const std::vector<T>& values) : Buffer(values.size()) { write(values); }
  ~Buffer() { cudaFree(ptr); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void write(const std::vector<T>& values) {
    if (values.size() != count) throw std::runtime_error("buffer shape");
    DGPP_CUDA_OK(cudaMemcpy(ptr, values.data(), count * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> read() const {
    std::vector<T> values(count);
    DGPP_CUDA_OK(cudaMemcpy(values.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    return values;
  }
};
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
float fp8(uint8_t bits) {
  __nv_fp8_e4m3 value;
  value.__x = bits;
  return static_cast<float>(value);
}
uint8_t fp8_bits(float value) {
  return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}
// kernel.py fast_round_scale: 2^ceil(log2(amax / 448)) from the fp32
// quotient's bit fields, the absmax floored at 1e-4.
float block_scale(float amax) {
  const float a = std::max(amax, 1e-4f);
  const float q = a * (1.0f / 448.0f);
  uint32_t bits;
  std::memcpy(&bits, &q, 4);
  const int k = static_cast<int>((bits >> 23) & 0xFFu) - 127 + ((bits & 0x7FFFFFu) != 0u ? 1 : 0);
  return std::ldexp(1.0f, k);
}
// One group of 32 BF16 values through the release's quantize-dequantize.
void host_qdq32(const uint16_t* x, uint16_t* y) {
  float amax = 0.0f;
  for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(bf16_bits_to_float(x[i])));
  const float s = block_scale(amax);
  for (int i = 0; i < 32; ++i) {
    const float v = std::min(std::max(bf16_bits_to_float(x[i]) / s, -448.0f), 448.0f);
    y[i] = float_to_bf16_bits(fp8(fp8_bits(v)) * s);
  }
}
std::vector<uint16_t> host_qdq(const std::vector<uint16_t>& x, size_t stride, int rows, int k, size_t out_stride) {
  std::vector<uint16_t> y(static_cast<size_t>(rows) * out_stride, 0x7777);
  for (int r = 0; r < rows; ++r)
    for (int g = 0; g < k / 32; ++g) host_qdq32(&x[r * stride + g * 32], &y[r * out_stride + g * 32]);
  return y;
}

// Rows of mixed magnitude: each group draws its own exponent; some groups
// are all zero, some sit under the 1e-4 floor, some hold a value exactly on
// a scale boundary (448 * 2^e) or just past it.
std::vector<uint16_t> make_rows(int rows, int k, size_t stride, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  std::uniform_int_distribution<int> expo(-14, 9), kind(0, 15);
  std::vector<uint16_t> x(static_cast<size_t>(rows) * stride, 0x7777);
  for (int r = 0; r < rows; ++r)
    for (int g = 0; g < k / 32; ++g) {
      const int kd = kind(rng);
      const float scale = std::ldexp(1.0f, expo(rng));
      for (int i = 0; i < 32; ++i) {
        float v = uni(rng) * scale;
        if (kd == 0) v = 0.0f;
        if (kd == 1) v *= 1e-6f;
        x[r * stride + g * 32 + i] = float_to_bf16_bits(v);
      }
      if (kd == 2) x[r * stride + g * 32 + 5] = float_to_bf16_bits(448.0f * scale);
      if (kd == 3) x[r * stride + g * 32 + 9] = float_to_bf16_bits(-448.0f * scale * 1.0078125f);
      if (kd == 4) x[r * stride + g * 32 + 31] = float_to_bf16_bits(3.0e4f);
    }
  return x;
}

void plain(int rows, int k, size_t stride, size_t out_stride, uint64_t seed) {
  const std::vector<uint16_t> x = make_rows(rows, k, stride, seed);
  const std::vector<uint16_t> want = host_qdq(x, stride, rows, k, out_stride);
  Buffer<uint16_t> dx(x), dy(std::vector<uint16_t>(static_cast<size_t>(rows) * out_stride, 0x7777));
  launch_act_qdq_fp8_block(dx.ptr, stride, dy.ptr, out_stride, rows, k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<uint16_t> got = dy.read();
  size_t bad = 0;
  for (size_t i = 0; i < got.size(); ++i) bad += got[i] != want[i];
  require(bad == 0, "act_qdq rows " + std::to_string(rows) + " k " + std::to_string(k) + ": " + std::to_string(bad) +
                        " values differ from the host reference (padding past k included)");
  // In place, and the quantizer's fixed point: a second pass changes nothing.
  launch_act_qdq_fp8_block(dx.ptr, stride, dx.ptr, stride, rows, k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<uint16_t> inplace = dx.read();
  const std::vector<uint16_t> want_in = host_qdq(x, stride, rows, k, stride);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < k; ++c)
      require(inplace[r * stride + c] == want_in[r * stride + c], "act_qdq in place differs from the host reference");
  launch_act_qdq_fp8_block(dx.ptr, stride, dx.ptr, stride, rows, k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  require(dx.read() == inplace, "act_qdq is not idempotent");
}

void swiglu(int rows, int k, size_t stride, bool weighted, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> normal(0.0f, 4.0f);
  std::uniform_real_distribution<float> wd(0.05f, 1.5f);
  const float limit = 10.0f;
  std::vector<uint16_t> gate(static_cast<size_t>(rows) * stride, 0), up(gate.size(), 0);
  std::vector<float> w(static_cast<size_t>(rows), 1.0f);
  for (int r = 0; r < rows; ++r) {
    if (weighted) w[r] = wd(rng);
    for (int c = 0; c < k; ++c) {
      gate[r * stride + c] = float_to_bf16_bits(normal(rng));
      up[r * stride + c] = float_to_bf16_bits(normal(rng));
    }
  }
  Buffer<uint16_t> dg(gate), du(up), da(std::vector<uint16_t>(gate.size(), 0x7777));
  Buffer<float> dw(w);
  launch_swiglu_weight_act_qdq_fp8_block(dg.ptr, du.ptr, stride, da.ptr, rows, k, limit, weighted ? dw.ptr : nullptr,
                                         nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<uint16_t> got = da.read();
  // Fixed point of the quantizer: the rows sit on their groups' grids.
  const std::vector<uint16_t> again = host_qdq(got, stride, rows, k, stride);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < k; ++c)
      require(again[r * stride + c] == got[r * stride + c], "swiglu qdq: a row is off its group's fp8 grid");
  // Distance from the release's product: half an e4m3 step (1/16 relative on
  // normals, 2^-10 of the scale on subnormals) plus the BF16 rounding.
  for (int r = 0; r < rows; ++r)
    for (int g = 0; g < k / 32; ++g) {
      float p[32], amax = 0.0f;
      for (int i = 0; i < 32; ++i) {
        float gv = bf16_bits_to_float(gate[r * stride + g * 32 + i]);
        float uv = bf16_bits_to_float(up[r * stride + g * 32 + i]);
        if (gv > limit) gv = limit;
        uv = std::min(std::max(uv, -limit), limit);
        const float silu = gv * (1.0f / (1.0f + std::exp(-gv)));
        p[i] = bf16_bits_to_float(float_to_bf16_bits(w[r] * (silu * uv)));
        amax = std::max(amax, std::fabs(p[i]));
      }
      const float s = block_scale(amax);
      for (int i = 0; i < 32; ++i) {
        const float y = bf16_bits_to_float(got[r * stride + g * 32 + i]);
        const float bound = 0.0625f * std::fabs(p[i]) + s * 0x1p-10f + std::fabs(p[i]) * 0x1p-6f + 1e-30f;
        require(std::fabs(y - p[i]) <= bound, "swiglu qdq: row " + std::to_string(r) + " differs from the weighted product by " +
                                                 std::to_string(std::fabs(y - p[i])) + " (bound " + std::to_string(bound) + ")");
      }
    }
}

void weights() {
  const int n = 1000;
  std::vector<int32_t> slot_row(n);
  std::vector<float> w(n);
  for (int i = 0; i < n; ++i) {
    slot_row[i] = (i * 7919) % n;  // a permutation: 7919 is prime to 1000
    w[i] = 0.001f * static_cast<float>(i + 1);
  }
  slot_row[17] = -1;  // an unrouted slot writes nothing
  Buffer<int32_t> ds(slot_row);
  Buffer<float> dw(w), dr(std::vector<float>(n + 8, -5.0f));
  launch_fill_f32(dr.ptr, n, 9.0f, nullptr);
  launch_scatter_row_weights(ds.ptr, dw.ptr, n, dr.ptr, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<float> got = dr.read();
  for (int i = 0; i < n; ++i)
    if (i != 17) require(got[(i * 7919) % n] == w[i], "scatter_row_weights: a route's weight is not at its row");
  require(got[(17 * 7919) % n] == 9.0f, "scatter_row_weights: an unrouted slot wrote a weight");
  for (int i = n; i < n + 8; ++i) require(got[i] == -5.0f, "fill_f32 wrote past its range");
}
}  // namespace

int main() {
  try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 2;
    plain(1, 32, 32, 32, 1);
    plain(37, 5120, 5120, 5120, 2);
    plain(513, 576, 576, 580, 3);
    plain(64, 1280, 1284, 1280, 4);
    plain(3, 8192, 8192, 8192, 5);
    swiglu(301, 576, 576, false, 6);
    swiglu(301, 576, 576, true, 7);
    swiglu(5, 2304, 2308, true, 8);
    weights();
    std::printf("act_quant_test: ok\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "act_quant_test: %s\n", e.what());
    return 1;
  }
}
