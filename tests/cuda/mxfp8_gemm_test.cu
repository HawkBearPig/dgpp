#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/mxfp8_gemm.hpp"

using namespace dgpp;
namespace {
std::filesystem::path dump_dir;
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
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
float fp8(uint8_t bits) {
  __nv_fp8_e4m3 value;
  value.__x = bits;
  return static_cast<float>(value);
}
uint8_t fp8_bits(float value) {
  return __nv_fp8_e4m3(value).__x;
}

template <class T>
void dump(const std::filesystem::path& path, const std::vector<T>& data) {
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
  if (!output) throw std::runtime_error("cannot write quantizer fixture");
}

void check_quantization(const std::vector<uint16_t>& x, int rows, int k, int stride) {
  Buffer<uint16_t> input(x);
  Buffer<uint8_t> codes(static_cast<size_t>(rows) * k), scales(static_cast<size_t>(rows) * k / 32);
  launch_mxfp8_quantize_rows(input.ptr, stride, rows, k, codes.ptr, scales.ptr, nullptr);
  const auto actual = codes.read(), actual_scales = scales.read();
  if (!dump_dir.empty()) {
    const auto base = dump_dir / (std::to_string(rows) + "x" + std::to_string(k));
    dump(base.string() + ".input.bf16", x);
    dump(base.string() + ".codes.u8", actual);
    dump(base.string() + ".scales.u8", actual_scales);
    std::ofstream metadata(base.string() + ".json");
    metadata << "{\"rows\":" << rows << ",\"k\":" << k << ",\"stride\":" << stride << "}\n";
    if (!metadata) throw std::runtime_error("cannot write quantizer fixture metadata");
  }
  for (int row = 0; row < rows; ++row)
    for (int group = 0; group < k / 32; ++group) {
      double maximum = 0;
      for (int i = 0; i < 32; ++i)
        maximum = std::max(maximum,
                           std::abs(double(bf16_bits_to_float(x[row * stride + group * 32 + i]))));
      const int exponent =
          maximum == 0 ? -127 : std::max(-127, int(std::ceil(std::log2(maximum / 448.0))));
      require(actual_scales[row * (k / 32) + group] == exponent + 127, "MXFP8 scale mismatch");
      const float inverse = std::ldexp(1.f, -exponent);
      for (int i = 0; i < 32; ++i) {
        const float value = bf16_bits_to_float(x[row * stride + group * 32 + i]);
        const uint8_t expected = fp8_bits(value * inverse);
        require(actual[row * k + group * 32 + i] == expected, "MXFP8 code mismatch");
      }
    }
}

void quantization() {
  // Every finite BF16 bit pattern, including signed zero and subnormals.
  constexpr int rows = 2040, k = 32, stride = 36;
  std::vector<uint16_t> all(static_cast<size_t>(rows) * stride, 0x7fc0);
  for (int index = 0; index < rows * k; ++index) {
    const uint16_t value = index < 32640 ? index : (index - 32640) | 0x8000;
    all[(index / k) * stride + index % k] = value;
  }
  check_quantization(all, rows, k, stride);
  // 193 groups exercises a partially occupied final quantizer CTA. Exact
  // exponent thresholds and their BF16 neighbours sit in separate groups.
  constexpr int wide = 6176, padded = wide + 4;
  std::vector<uint16_t> edges(3 * padded, 0x7fc0);
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < wide; ++col) {
      const float maximum = std::ldexp(row == 0   ? 446.f
                                       : row == 1 ? 448.f
                                                  : 450.f,
                                       (col / 32) % 49 - 24);
      const float value = col % 32 == 0 ? maximum : maximum * (col % 17 - 8) / 16.f;
      edges[row * padded + col] = float_to_bf16_bits(value);
    }
  check_quantization(edges, 3, wide, padded);
  std::puts(
      "MXFP8 quantizer: all finite BF16 inputs, signed zeros, exponent boundaries and tails "
      "passed");
}

void gemm(int rows, int n, int k, unsigned seed) {
  std::mt19937 random(seed);
  std::vector<uint8_t> a(static_cast<size_t>(rows) * k), w(static_cast<size_t>(n) * k);
  std::vector<uint8_t> as(static_cast<size_t>(rows) * k / 32), ws(static_cast<size_t>(n) * k / 32);
  for (auto* values : {&a, &w})
    for (auto& value : *values) {
      value = static_cast<uint8_t>(random() % 256);
      if ((value & 127) == 127) value &= 0xf0;
    }
  for (auto* values : {&as, &ws})
    for (auto& value : *values) value = static_cast<uint8_t>(117 + random() % 17);
  // Distinct row/group scales catch the lane selector and group-stride bugs
  // that uniform scales hide. One row of signed zeros also checks padding.
  if (rows > 1)
    for (int column = 0; column < k; ++column) a[column] = column & 1 ? 128 : 0;
  Buffer<uint8_t> da(a), dw(w), das(as), dws(ws);
  const int stride = n + 7;
  const float sentinel = 123456.f;
  Buffer<float> out(std::vector<float>(static_cast<size_t>(rows) * stride, sentinel));
  Buffer<uint16_t> rounded(std::vector<uint16_t>(static_cast<size_t>(rows) * stride, 0x3f01));
  launch_mxfp8_gemm_f32(da.ptr, das.ptr, dw.ptr, dws.ptr, out.ptr, rows, n, k, nullptr, stride);
  launch_mxfp8_gemm_bf16(da.ptr, das.ptr, dw.ptr, dws.ptr, rounded.ptr, rows, n, k, nullptr,
                         stride);
  const auto actual = out.read();
  const auto bf16 = rounded.read();
  double error2 = 0, reference2 = 0;
  const double unit = std::numeric_limits<float>::epsilon() / 2;
  const double gamma = k * unit / (1 - k * unit);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < n; ++column) {
      double expected = 0, absolute = 0;
      for (int inner = 0; inner < k; ++inner) {
        const double av =
            std::ldexp(double(fp8(a[row * k + inner])), int(as[row * (k / 32) + inner / 32]) - 127);
        const double wv = std::ldexp(double(fp8(w[column * k + inner])),
                                     int(ws[column * (k / 32) + inner / 32]) - 127);
        expected += av * wv;
        absolute += std::abs(av * wv);
      }
      const double error = double(actual[row * stride + column]) - expected;
      require(std::isfinite(actual[row * stride + column]), "nonfinite MXFP8 result");
      require(std::abs(error) <= gamma * absolute + 1e-12, "MXFP8 FP64 forward-error bound");
      require(bf16[row * stride + column] == float_to_bf16_bits(actual[row * stride + column]),
              "MXFP8 BF16 epilogue");
      error2 += error * error;
      reference2 += expected * expected;
    }
    for (int column = n; column < stride; ++column) {
      require(actual[row * stride + column] == sentinel, "MXFP8 output stride guard");
      require(bf16[row * stride + column] == 0x3f01, "MXFP8 BF16 output stride guard");
    }
  }
  const double relative = std::sqrt(error2 / std::max(reference2, 1e-30));
  require(relative < 4e-6, "MXFP8 FP64 relative RMS");
  // Exercise both tile geometries, repeated launches and arbitrary input
  // offsets. Every scalar uses the same k32 accumulation chain.
  for (int chunk : {1, 8, 17}) {
    for (int start = 0; start < rows; start += chunk) {
      const int count = std::min(chunk, rows - start);
      launch_mxfp8_gemm_f32(da.ptr + start * k, das.ptr + start * (k / 32), dw.ptr, dws.ptr,
                            out.ptr + start * stride, count, n, k, nullptr, stride);
    }
    const auto split = out.read();
    require(std::memcmp(split.data(), actual.data(), actual.size() * sizeof(float)) == 0,
            "MXFP8 matrix row-geometry invariance");
  }
  std::printf("MXFP8 GEMM: m=%d n=%d k=%d FP64 relRMS=%.3g, row geometry exact\n", rows, n, k,
              relative);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--dump-quant") {
      dump_dir = argv[2];
      std::filesystem::create_directories(dump_dir);
    } else if (argc != 1) {
      throw std::runtime_error("usage: mxfp8_gemm_test [--dump-quant directory]");
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 2;
    quantization();
    gemm(1, 1, 64, 1);
    gemm(17, 73, 128, 2);
    gemm(65, 65, 256, 3);
    gemm(33, 128, 6144, 4);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mxfp8_gemm_test: %s\n", e.what());
    return 1;
  }
}
