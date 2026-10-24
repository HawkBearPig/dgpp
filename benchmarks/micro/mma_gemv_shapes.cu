// mma_gemv_shapes: the fp8 decode GEMV at Qwen3.8-27B's per-rank shapes on
// two and four Sparks (#93), cold in L2 (eight weight copies rotated), at
// 1 / 4 / 8 activation rows: the streaming tensor-core form (mma_gemv) at
// the rule's block width and the forced widths, the row-chunk GEMV cores
// (scale_gemv) and the fused swiglu down form. Split-K follows the library
// rule (DGPP_MMA_SPLITK_FILL overrides the fill target for the sweep).
// Reports us per launch and effective GB/s (weight bytes / time).
#include <cuda_runtime.h>
#include <getopt.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "common/cuda_check.hpp"
#include "kernels/mma_gemv.hpp"
#include "kernels/scale_gemm.hpp"

namespace {
struct Shape { const char* name; int n, k; };
constexpr int kCopies = 8;

template <typename F>
double time_us(F&& launch, int iters) {
  cudaEvent_t b{}, e{};
  cudaEventCreate(&b); cudaEventCreate(&e);
  for (int i = 0; i < 2 * kCopies; ++i) launch(i % kCopies);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  cudaEventRecord(b, nullptr);
  for (int i = 0; i < iters; ++i) launch(i % kCopies);
  cudaEventRecord(e, nullptr);
  cudaEventSynchronize(e);
  float ms = 0.f; cudaEventElapsedTime(&ms, b, e);
  cudaEventDestroy(b); cudaEventDestroy(e);
  return 1000.0 * ms / iters;
}
}  // namespace

int main(int argc, char** argv) {
  int iters = 40, world = 4;
  std::string forms = "mma,w2,w4,w8,chunk,swiglu";
  bool dsv41 = false;  // -d: DeepSeek-V4.1-Flash's attention projections per rank (TP = world), 32 x 32 scale grid
  int opt{};
  while ((opt = getopt(argc, argv, "i:W:f:d")) != -1) {
    if (opt == 'i') iters = atoi(optarg);
    if (opt == 'W') world = atoi(optarg);
    if (opt == 'f') forms = optarg;
    if (opt == 'd') dsv41 = true;
  }
  const int H = 5120, I = 17408 / world, QW = 24 * 256 / world, KVW = 256 * (4 >= world ? 4 / world : 1);
  const int GK = 16 * 128 / world, GV = 48 * 128 / world;  // GDN key / value widths
  const int V = 248320 / world;
  std::vector<Shape> shapes = {
      {"mlp gate|up [I x H]", I, H}, {"mlp down [H x I]", H, I},
      {"gdn in_proj qkz [2K+V x H]", 2 * GK + GV, H}, {"gdn in_proj v [V x H]", GV, H},
      {"gdn out [H x V]", H, GV}, {"attn qkv [2QW x H]", 2 * QW, H}, {"attn kv [KVW x H]", KVW, H},
      {"attn out [H x QW]", H, QW}, {"lm head [V/w x H]", V, H}};
  const int scale_shift = dsv41 ? 5 : 7;  // the scale grid's block size as a shift
  if (dsv41) {
    const int lh = 64 / world, lg = 8 / world;  // heads, o groups per rank
    shapes = {{"wkv [512 x H]", 512, H}, {"wq_a [1280 x H]", 1280, H}, {"wq_b [lh*512 x 1280]", lh * 512, 1280},
              {"idx_wq_b [32*128 x 1280]", 32 * 128, 1280}, {"wo_a group [1024 x 8*512]", 1024, 8 * 512},
              {"wo_b [H x lg*1024]", H, lg * 1024}};
  }
  // Scales grid: 128 x 128 blocks (rs = cs = 7), f32 ones.
  size_t ws_bytes = 64u << 20;
  void* ws = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&ws, ws_bytes));
  uint16_t* act = nullptr;  // 32 rows x max k, bf16 small values
  const int kmax = 17408;
  DGPP_CUDA_OK(cudaMalloc(&act, size_t(32) * kmax * 2));
  {
    std::vector<uint16_t> h(size_t(32) * kmax);
    for (size_t i = 0; i < h.size(); ++i) h[i] = 0x3C00 | ((i * 2654435761u) & 0x7F);  // ~1.0 +- small
    DGPP_CUDA_OK(cudaMemcpy(act, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
  }
  printf("world %d, %d copies rotated (cold L2), %d iters; us per launch / GB/s (weight bytes / time)\n", world, kCopies, iters);
  for (const Shape& s : shapes) {
    const size_t wbytes = size_t(s.n) * s.k;
    const int sb = 1 << scale_shift;
    const int sr = (s.n + sb - 1) / sb, sc = (s.k + sb - 1) / sb;
    std::vector<uint8_t*> w(kCopies); std::vector<float*> sc_(kCopies);
    for (int c = 0; c < kCopies; ++c) {
      DGPP_CUDA_OK(cudaMalloc(&w[c], wbytes));
      DGPP_CUDA_OK(cudaMemset(w[c], 0x38 + c, wbytes));  // e4m3 ~0.5..1 range
      DGPP_CUDA_OK(cudaMalloc(&sc_[c], size_t(sr) * sc * 4));
      std::vector<float> ones(size_t(sr) * sc, 1.0f);
      DGPP_CUDA_OK(cudaMemcpy(sc_[c], ones.data(), ones.size() * 4, cudaMemcpyHostToDevice));
    }
    uint16_t* out = nullptr; DGPP_CUDA_OK(cudaMalloc(&out, size_t(32) * s.n * 2));
    float* outf = nullptr; DGPP_CUDA_OK(cudaMalloc(&outf, size_t(32) * s.n * 4));
    printf("\n== %s  n=%d k=%d  %.1f MB\n", s.name, s.n, s.k, wbytes / 1e6);
    for (int m : {1, 4, 8, 16}) {
      auto report = [&](const char* form, double us) {
        printf("  m=%d %-8s %8.1f us  %6.1f GB/s\n", m, form, us, wbytes / us / 1e3);
      };
      if (forms.find("mma") != std::string::npos) {
        dgpp::mma_gemv_set_decode_width(0);
        report("mma", time_us([&](int c) {
          dgpp::launch_mma_gemv_fp8_bf16(act, s.k, w[c], sc_[c], out, m, s.n, s.k, s.n, scale_shift, scale_shift, nullptr, ws, ws_bytes); }, iters));
      }
      if (forms.find("unsplit") != std::string::npos) {
        dgpp::mma_gemv_set_decode_width(0);
        report("unsplit", time_us([&](int c) {
          dgpp::launch_mma_gemv_fp8_bf16(act, s.k, w[c], sc_[c], out, m, s.n, s.k, s.n, scale_shift, scale_shift, nullptr, nullptr, 0); }, iters));
      }
      for (int width : {2, 4, 8}) {
        const std::string tag = "w" + std::to_string(width);
        if (forms.find(tag) == std::string::npos) continue;
        dgpp::mma_gemv_set_decode_width(width);
        report(tag.c_str(), time_us([&](int c) {
          dgpp::launch_mma_gemv_fp8_bf16(act, s.k, w[c], sc_[c], out, m, s.n, s.k, s.n, scale_shift, scale_shift, nullptr, ws, ws_bytes); }, iters));
        dgpp::mma_gemv_set_decode_width(0);
      }
      if (forms.find("chunk") != std::string::npos && m <= 8) {
        report("chunk", time_us([&](int c) {
          dgpp::launch_scale_gemm_bf16(act, s.k, w[c], sc_[c], out, m, s.n, s.k, nullptr, s.n, /*mma_from_rows=*/0); }, iters));
      }
      if (forms.find("swiglu") != std::string::npos && std::strstr(s.name, "down")) {
        report("swiglu", time_us([&](int c) {
          dgpp::launch_mma_gemv_fp8_swiglu_f32(act, act, 7.0f, s.k, w[c], sc_[c], outf, m, s.n, s.k, s.n, 7, 7, nullptr); }, iters));
      }
    }
    for (int c = 0; c < kCopies; ++c) { cudaFree(w[c]); cudaFree(sc_[c]); }
    cudaFree(out); cudaFree(outf);
  }
  cudaFree(ws); cudaFree(act);
  return 0;
}
