// Parity tests for the EXL3 streaming decode GEMM (kernels/exl3_gemv.hpp,
// docs/deepseek_v41_flash_exl3_plan.md §5): the launcher against the host
// oracle (models/exl3_matrix.hpp exl3_oracle_row: the kernel's A rounding,
// exact products in double, the rotations in double) at the pack's shapes
// and bit widths, m = 1..64, split and unsplit; the row-count invariance
// bitwise; the f32 epilogue as the unrounded bf16; the shape contract.
//
// Budget: the kernel differs from the oracle by its fp32 accumulation
// order (up to 5,120 terms) and the fp32 rotation — the strict suite's 2
// bf16 ulps with a 1e-3 cancellation floor, zero mismatches.
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/exl3_gemv.hpp"
#include "models/exl3_matrix.hpp"
#include "scale_gemm_test_helpers.hpp"

namespace {

using namespace scale_gemm_test;
using dgpp::Exl3HostMatrix;
using dgpp::Exl3Matrix;

struct Shape {
  int n, k, bits;
  const char* what;
};
const Shape kShapes[] = {
    {1280, 5120, 6, "wq_a"},          {512, 5120, 6, "wkv"},          {8192, 1280, 6, "wq_b slice"},
    {5120, 2048, 6, "wo_b slice"},    {1024, 4096, 6, "wo_a group"},   {4096, 1280, 6, "indexer wq_b"},
    {640, 5120, 4, "expert gate 640"}, {5120, 640, 4, "expert down 640"}, {512, 5120, 3, "expert gate 512 (3b)"},
    {5120, 512, 3, "expert down 512 (3b)"}, {128, 512, 8, "indexer wk"}, {256, 128, 3, "tiny"},
    {1024, 256, 4, "small"},
};

struct Device {
  uint8_t* tiles = nullptr;
  uint16_t* suh = nullptr;
  uint16_t* svh = nullptr;
  uint16_t* act = nullptr;
  uint16_t* out_bf16 = nullptr;
  float* out_f32 = nullptr;
  void* ws = nullptr;  // the parity test's shapes share one (owns_ws false): a layer's sites share theirs
  size_t ws_bytes = 0;
  bool owns_ws = true;
  ~Device() {
    cudaFree(tiles);
    cudaFree(suh);
    cudaFree(svh);
    cudaFree(act);
    cudaFree(out_bf16);
    cudaFree(out_f32);
    if (owns_ws) cudaFree(ws);
  }
};

template <typename T>
T* upload(const std::vector<T>& v) {
  T* d = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&d, v.size() * sizeof(T) + 16));
  DGPP_CUDA_OK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

Exl3Matrix view_of(const Exl3HostMatrix& h, const Device& d) {
  Exl3Matrix m = h.view();
  m.tiles = d.tiles;
  m.suh = d.suh;
  m.svh = d.svh;
  return m;
}

// Activations: rows of N(0, 1)-ish bf16 values scaled by `scale`, a few
// rows with large magnitudes (the fp16 range of the rotated row).
std::vector<uint16_t> make_act(Rng& rng, int m, int k, size_t stride, float scale) {
  std::vector<uint16_t> a(static_cast<size_t>(m) * stride, 0);
  for (int r = 0; r < m; ++r) {
    const float s = (r % 7 == 3) ? scale * 50.0f : scale;
    for (int c = 0; c < k; ++c)
      a[static_cast<size_t>(r) * stride + c] =
          dgpp::float_to_bf16_bits(static_cast<float>(rng.unit() + rng.unit() + rng.unit()) * s);
  }
  return a;
}

std::vector<double> oracle(const Exl3HostMatrix& h, const std::vector<double>& inner, const std::vector<uint16_t>& act,
                           size_t stride, int m) {
  std::vector<double> y;
  std::vector<double> all(static_cast<size_t>(m) * h.rows);
  for (int r = 0; r < m; ++r) {
    dgpp::exl3_oracle_row(h, inner, act.data() + static_cast<size_t>(r) * stride, y);
    std::copy(y.begin(), y.end(), all.begin() + static_cast<size_t>(r) * h.rows);
  }
  return all;
}

void test_parity() {
  Rng rng(0x5eed);
  // One workspace for every shape, zeroed once (the contract): a layer's
  // sites of different widths share theirs the same way, so a shape's
  // launch must leave the counters as it found them for the next shape.
  size_t shared_ws_bytes = 0;
  for (const Shape& s : kShapes) shared_ws_bytes = std::max(shared_ws_bytes, dgpp::exl3_gemm_ws_bytes(s.n, s.k, 16));
  void* shared_ws = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&shared_ws, shared_ws_bytes));
  DGPP_CUDA_OK(cudaMemset(shared_ws, 0, shared_ws_bytes));
  for (const Shape& s : kShapes) {
    Exl3HostMatrix h = dgpp::exl3_random_matrix(rng, s.n, s.k, s.bits, 0.03f, 1.0f);
    const std::vector<double> inner = dgpp::exl3_inner_weights(h);
    Device d;
    d.tiles = upload(h.tiles);
    d.suh = upload(h.suh);
    d.svh = upload(h.svh);
    d.ws = shared_ws;
    d.ws_bytes = shared_ws_bytes;
    d.owns_ws = false;
    const Exl3Matrix w = view_of(h, d);
    const int splits = dgpp::exl3_gemm_splits(s.n, s.k, d.ws, d.ws_bytes);
    const size_t stride = static_cast<size_t>(s.k) + 8;  // a padded stride (16-byte multiple)
    for (int m : {1, 3, 16, 17, 32, 33, 64, 65, 70, 100, 128, 129, 200, 260}) {  // 65-128: the 128-row tile; past 128: the groups
      const std::vector<uint16_t> act = make_act(rng, m, s.k, stride, 1.0f);
      const std::vector<double> ref = oracle(h, inner, act, stride, m);
      uint16_t* dact = upload(act);
      std::vector<uint16_t> got(static_cast<size_t>(m) * s.n);
      std::vector<float> got32(got.size());
      uint16_t* dout = upload(got);
      float* dout32 = upload(got32);
      for (int pass = 0; pass < 2; ++pass) {  // split (the full workspace) and unsplit (room for xh alone)
        const size_t wsb = pass == 0 ? d.ws_bytes : dgpp::exl3_gemm_ws_bytes(s.n, s.k, 1);
        dgpp::launch_exl3_gemm_bf16(dact, stride, w, dout, m, s.n, s.k, 0, nullptr, d.ws, wsb);
        dgpp::launch_exl3_gemm_f32(dact, stride, w, dout32, m, s.n, s.k, 0, nullptr, d.ws, wsb);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        DGPP_CUDA_OK(cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost));
        DGPP_CUDA_OK(cudaMemcpy(got32.data(), dout32, got32.size() * 4, cudaMemcpyDeviceToHost));
        const CompareReport rep = compare_bf16_vs_oracle(got.data(), ref, 2.0, 1e-3);
        if (rep.mismatches != 0 || rep.nan_pattern_errors != 0)
          throw std::runtime_error(std::string("parity ") + s.what + " m=" + std::to_string(m) +
                                   (pass == 0 ? " split" : " unsplit") + ": " + std::to_string(rep.mismatches) +
                                   " mismatches, l2 " + std::to_string(rep.l2_rel));
        for (size_t i = 0; i < got.size(); ++i)
          if (dgpp::float_to_bf16_bits(got32[i]) != got[i])
            throw std::runtime_error(std::string("f32 epilogue ") + s.what + ": bf16(out_f32) != out_bf16 at " +
                                     std::to_string(i));
        if (pass == 0 && m == 1)
          std::printf("  %-24s n=%5d k=%5d bits=%d splits=%2d  l2_rel %.3e\n", s.what, s.n, s.k, s.bits, splits,
                      rep.l2_rel);
        // The same launch again is bitwise the first (the split form's
        // arrival counters left at zero; nothing in the chain is racy).
        std::vector<uint16_t> again(got.size());
        dgpp::launch_exl3_gemm_bf16(dact, stride, w, dout, m, s.n, s.k, 0, nullptr, d.ws, wsb);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        DGPP_CUDA_OK(cudaMemcpy(again.data(), dout, again.size() * 2, cudaMemcpyDeviceToHost));
        if (std::memcmp(again.data(), got.data(), got.size() * 2) != 0)
          throw std::runtime_error(std::string("repeat ") + s.what + " m=" + std::to_string(m) +
                                   (pass == 0 ? " split" : " unsplit") + ": the second launch is not bitwise the first");
      }
      // Row-count invariance (the split form): every row of the m-row
      // launch bitwise the row alone.
      {
        std::vector<uint16_t> one(static_cast<size_t>(s.n));
        uint16_t* done = upload(one);
        dgpp::launch_exl3_gemm_bf16(dact, stride, w, dout, m, s.n, s.k, 0, nullptr, d.ws, d.ws_bytes);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        DGPP_CUDA_OK(cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost));
        for (int r = 0; r < m; r += (m > 8 ? 7 : 1)) {
          dgpp::launch_exl3_gemm_bf16(dact + static_cast<size_t>(r) * stride, stride, w, done, 1, s.n, s.k, 0, nullptr,
                                      d.ws, d.ws_bytes);
          DGPP_CUDA_OK(cudaDeviceSynchronize());
          DGPP_CUDA_OK(cudaMemcpy(one.data(), done, one.size() * 2, cudaMemcpyDeviceToHost));
          if (std::memcmp(one.data(), got.data() + static_cast<size_t>(r) * s.n, one.size() * 2) != 0)
            throw std::runtime_error(std::string("row invariance ") + s.what + " m=" + std::to_string(m) + " row " +
                                     std::to_string(r));
        }
        cudaFree(done);
      }
      cudaFree(dact);
      cudaFree(dout);
      cudaFree(dout32);
    }
  }
  cudaFree(shared_ws);
}

void test_contract() {
  Rng rng(9);
  Exl3HostMatrix h = dgpp::exl3_random_matrix(rng, 256, 256, 4, 0.03f, 1.0f);
  Device d;
  d.tiles = upload(h.tiles);
  d.suh = upload(h.suh);
  d.svh = upload(h.svh);
  std::vector<uint16_t> act(256 + 8, 0);
  uint16_t* dact = upload(act);
  const Exl3Matrix w = view_of(h, d);
  require(dgpp::exl3_gemm_shape_ok(w, dact, 264, 1, 256, 256), "the contract shape is accepted");
  require(!dgpp::exl3_gemm_shape_ok(w, dact, 264, 1, 128, 256), "n != rows is refused");
  require(!dgpp::exl3_gemm_shape_ok(w, dact, 263, 1, 256, 256), "an odd-16-byte stride is refused");
  Exl3Matrix bad = w;
  bad.bits = 5;
  require(!dgpp::exl3_gemm_shape_ok(bad, dact, 264, 1, 256, 256), "an uncompiled width is refused");
  void* ws = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&ws, dgpp::exl3_gemm_ws_bytes(256, 256, 16)));
  DGPP_CUDA_OK(cudaMemset(ws, 0, dgpp::exl3_gemm_ws_bytes(256, 256, 16)));
  bool threw = false;
  try {
    dgpp::launch_exl3_gemm_bf16(dact, 264, bad, nullptr, 1, 256, 256, 0, nullptr, ws, dgpp::exl3_gemm_ws_bytes(256, 256, 16));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "the launcher throws on the contract");
  threw = false;
  try {
    dgpp::launch_exl3_gemm_bf16(dact, 264, w, nullptr, 1, 256, 256, 0, nullptr, nullptr, 0);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "no workspace is refused");
  require(dgpp::exl3_gemm_splits(1280, 5120, d.tiles, 1 << 30) > 1, "a narrow matrix splits with a workspace");
  require(dgpp::exl3_gemm_splits(1280, 5120, d.tiles, dgpp::exl3_gemm_ws_bytes(1280, 5120, 1)) == 1,
          "a workspace without room for partials does not split");
  require(dgpp::exl3_gemm_splits(32768, 1280, d.tiles, 1 << 30) == 1, "a wide matrix does not split");
  cudaFree(ws);
  cudaFree(dact);
}

}  // namespace

int main() {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) {
    std::printf("SKIP: no CUDA device\n");
    return 2;
  }
  struct Case {
    const char* name;
    void (*fn)();
  };
  const Case cases[] = {{"contract", test_contract}, {"parity + row invariance", test_parity}};
  int failed = 0;
  for (const Case& c : cases) {
    try {
      c.fn();
      std::printf("[ OK ] %s\n", c.name);
    } catch (const std::exception& e) {
      std::printf("[FAIL] %s: %s\n", c.name, e.what());
      ++failed;
    }
  }
  return failed ? 1 : 0;
}
