// The MTP hyper-state window's store and gather (qwen_mtp.hpp): a row
// stored at its position is read back bit for bit at an aligned width
// (the 16-byte copy with eight units in flight) and at a width the
// element loop takes; a padding row (pos < 0) stores nothing and gathers
// zeros; rows of two requests land in their own regions.
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/test.hpp"
#include "kernels/qwen_mtp.hpp"

namespace {

void require(bool c, const std::string& w) { if (!c) throw std::runtime_error(w); }

void round_trip(int width, int rows, int window_rows) {
  std::mt19937 rng(static_cast<unsigned>(width));
  std::vector<uint16_t> r(static_cast<size_t>(rows) * width);
  for (auto& x : r) x = static_cast<uint16_t>(rng());
  std::vector<int32_t> req(static_cast<size_t>(rows));
  std::vector<int64_t> pos(static_cast<size_t>(rows));
  for (int i = 0; i < rows; ++i) {
    req[static_cast<size_t>(i)] = i % 2;
    pos[static_cast<size_t>(i)] = i == rows - 1 ? -1 : 1000 + i;  // the last row a padding row
  }
  const int64_t req_stride = static_cast<int64_t>(window_rows) * width;
  uint16_t* dr; uint16_t* dw; uint16_t* dout; int32_t* dreq; int64_t* dpos;
  DGPP_CUDA_OK(cudaMalloc(&dr, r.size() * 2));
  DGPP_CUDA_OK(cudaMemcpy(dr, r.data(), r.size() * 2, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMalloc(&dw, static_cast<size_t>(2) * req_stride * 2));
  DGPP_CUDA_OK(cudaMemset(dw, 0x5A, static_cast<size_t>(2) * req_stride * 2));
  DGPP_CUDA_OK(cudaMalloc(&dout, r.size() * 2));
  DGPP_CUDA_OK(cudaMemset(dout, 0x7F, r.size() * 2));
  DGPP_CUDA_OK(cudaMalloc(&dreq, req.size() * 4));
  DGPP_CUDA_OK(cudaMemcpy(dreq, req.data(), req.size() * 4, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMalloc(&dpos, pos.size() * 8));
  DGPP_CUDA_OK(cudaMemcpy(dpos, pos.data(), pos.size() * 8, cudaMemcpyHostToDevice));
  dgpp::qwen_mtp_hidden_store_bf16(dr, dreq, dpos, dw, req_stride, window_rows, rows, width, nullptr);
  dgpp::qwen_mtp_hidden_gather_bf16(dw, req_stride, window_rows, dreq, dpos, dout, rows, width, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> got(r.size()), win(static_cast<size_t>(2) * req_stride);
  DGPP_CUDA_OK(cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(win.data(), dw, win.size() * 2, cudaMemcpyDeviceToHost));
  for (int i = 0; i < rows; ++i) {
    for (int d = 0; d < width; ++d) {
      const uint16_t want = pos[static_cast<size_t>(i)] < 0 ? 0 : r[static_cast<size_t>(i) * width + d];
      require(got[static_cast<size_t>(i) * width + d] == want,
              "round trip differs at row " + std::to_string(i) + " col " + std::to_string(d) + " width " + std::to_string(width));
    }
    if (pos[static_cast<size_t>(i)] >= 0) {
      const size_t at = static_cast<size_t>(req[static_cast<size_t>(i)]) * req_stride +
                        static_cast<size_t>(pos[static_cast<size_t>(i)] % window_rows) * width;
      for (int d = 0; d < width; ++d)
        require(win[at + d] == r[static_cast<size_t>(i) * width + d], "the window row differs");
    }
  }
  // The whole window is the fill with exactly the stored rows written (the
  // padding row stored nothing): compared as an image, since a random row
  // element can equal the fill.
  std::vector<uint16_t> expect(win.size(), 0x5A5Au);
  for (int i = 0; i < rows; ++i) {
    if (pos[static_cast<size_t>(i)] < 0) continue;
    const size_t at = static_cast<size_t>(req[static_cast<size_t>(i)]) * req_stride +
                      static_cast<size_t>(pos[static_cast<size_t>(i)] % window_rows) * width;
    for (int d = 0; d < width; ++d) expect[at + d] = r[static_cast<size_t>(i) * width + d];
  }
  require(expect == win, "a store touched another row");
  cudaFree(dr); cudaFree(dw); cudaFree(dout); cudaFree(dreq); cudaFree(dpos);
}

DGPP_TEST(hidden_window_round_trip_aligned_width) { round_trip(25600, 6, 8); }
DGPP_TEST(hidden_window_round_trip_element_width) { round_trip(1001, 5, 4); }
DGPP_TEST(hidden_window_round_trip_short_aligned_width) { round_trip(64, 3, 2); }

}  // namespace

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  return dgpp::test::run_all();
}
