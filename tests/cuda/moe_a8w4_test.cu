// The grouped MoE GEMM with E4M3 activations and MXFP4 weights
// (src/kernels/moe_a8w4.cu) and the quantizers that feed it
// (src/kernels/act_quant.cu):
//  1. the quantizer's codes and scale bytes against the host helpers, and
//     its BF16 rows against launch_act_qdq_fp8_block (bitwise);
//  2. layouts: against an fp64 host dot of the same quantized operands, so
//     only the fp32 summation may differ, on a ragged shape and on the
//     DeepSeek-V4.1 shapes;
//  3. the BF16 output form against the FP32 one rounded (bitwise);
//  4. against the BF16-activation kernel on the dequantized rows (the same
//     real sums in another order);
//  5. the SwiGLU producer's codes against its quantize-dequantize form;
//  6. the tile-list grid against the segment grid (bitwise), on even
//     routing and with one expert in every token's routes (the shape of a
//     DeepSeek-V4.1 layer, where the segment grid is mostly empty blocks);
//  7. speed at prefill shapes against launch_moe_grouped_mma_fp4_*, both
//     grids, both routings.
//   moe_a8w4_test [--tokens T] [--experts E] [--topk K] [--no-speed]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/act_quant.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/latent_format.hpp"
#include "kernels/moe_a8w4.hpp"

using namespace dgpp;

namespace {

double e2m1(uint8_t nib) {
  static const double mag[8] = {0., 0.5, 1., 1.5, 2., 3., 4., 6.};
  return (nib & 8) ? -mag[nib & 7] : mag[nib & 7];
}
double e4m3(uint8_t b) {
  __nv_fp8_e4m3 v;
  v.__x = b;
  return static_cast<double>(static_cast<float>(v));
}
double e8m0(uint8_t b) { return std::ldexp(1.0, static_cast<int>(b) - 127); }
uint16_t bf16(float f) {
  const __nv_bfloat16 b = __float2bfloat16_rn(f);
  uint16_t u;
  std::memcpy(&u, &b, 2);
  return u;
}
template <typename T>
T* dev(const std::vector<T>& h) {
  T* d = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T)));
  DGPP_CUDA_OK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
template <typename T>
T* dev_alloc(size_t n) {
  T* d = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&d, std::max<size_t>(1, n) * sizeof(T)));
  DGPP_CUDA_OK(cudaMemset(d, 0xA5, std::max<size_t>(1, n) * sizeof(T)));
  return d;
}
template <typename T>
std::vector<T> host(const T* d, size_t n) {
  std::vector<T> h(n);
  DGPP_CUDA_OK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

struct Matrix {  // one expert's MXFP4 matrix [n, k]
  std::vector<uint8_t> codes, scales;
};

struct Case {
  int tokens, experts, topk, n, k;
  std::vector<uint16_t> act;      // [tokens, k] bf16
  std::vector<Matrix> w;          // per expert
  std::vector<MoeSegment> segs;
  std::vector<int32_t> act_rows;  // [routed rows] -> token
  int max_rows = 0;
};

// hot: expert 0 is one of every token's routes.
Case make_case(int tokens, int experts, int topk, int n, int k, uint32_t seed, bool hot) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_int_distribution<int> scale(112, 124);  // 2^-15 .. 2^-3
  Case c{tokens, experts, topk, n, k, {}, {}, {}, {}, 0};
  c.act.resize(static_cast<size_t>(tokens) * k);
  for (size_t i = 0; i < c.act.size(); ++i) {
    float x = nd(rng);
    const int roll = byte(rng);
    if (roll == 0) x *= 20.f;  // outliers, as real hidden states have
    if (roll == 1) x = 0.f;
    c.act[i] = bf16(x);
  }
  // A silent group and a silent row: the 1e-4 floor on the absmax.
  for (int j = 0; j < 32 && j < k; ++j) c.act[j] = 0;
  if (tokens > 1)
    for (int j = 0; j < k; ++j) c.act[static_cast<size_t>(k) + j] = bf16(1e-7f * static_cast<float>(j % 3));
  for (int e = 0; e < experts; ++e) {
    Matrix m;
    m.codes.resize(static_cast<size_t>(n) * k / 2);
    for (auto& b : m.codes) b = static_cast<uint8_t>(byte(rng));
    m.scales.resize(static_cast<size_t>(n) * k / 32);
    for (auto& s : m.scales) s = static_cast<uint8_t>(scale(rng));
    c.w.push_back(std::move(m));
  }
  // Routing: each token picks topk distinct experts; segments by expert.
  std::vector<std::vector<int>> by_expert(static_cast<size_t>(experts));
  std::uniform_int_distribution<int> pick(0, experts - 1);
  for (int t = 0; t < tokens; ++t) {
    std::vector<int> chosen;
    if (hot) chosen.push_back(0);
    while (static_cast<int>(chosen.size()) < std::min(topk, experts)) {
      const int e = pick(rng);
      bool dup = false;
      for (int x : chosen) dup |= x == e;
      if (!dup) chosen.push_back(e);
    }
    for (int e : chosen) by_expert[static_cast<size_t>(e)].push_back(t);
  }
  int row = 0;
  for (int e = 0; e < experts; ++e) {
    const auto& toks = by_expert[static_cast<size_t>(e)];
    if (toks.empty()) continue;
    c.segs.push_back(MoeSegment{row, static_cast<int32_t>(toks.size()), e});
    for (int t : toks) c.act_rows.push_back(t);
    row += static_cast<int>(toks.size());
    c.max_rows = std::max(c.max_rows, static_cast<int>(toks.size()));
  }
  return c;
}

struct Device {
  std::vector<void*> bufs;
  MoeExpertView* views = nullptr;
};

Device upload(const Case& c) {
  Device d;
  std::vector<MoeExpertView> views(static_cast<size_t>(c.experts) * 3);
  for (int e = 0; e < c.experts; ++e) {
    uint8_t* codes = dev(c.w[static_cast<size_t>(e)].codes);
    uint8_t* scales = dev(c.w[static_cast<size_t>(e)].scales);
    d.bufs.push_back(codes);
    d.bufs.push_back(scales);
    MoeExpertView v{codes, nullptr, scales, nullptr, 7, 7};
    v.fp4_group = kMxfp4Group;
    views[static_cast<size_t>(e) * 3] = v;
  }
  d.views = dev(views);
  return d;
}

int run(int tokens, int experts, int topk, int n, int k, bool check_all, bool speed, const char* label,
        bool hot = false) {
  const Case c = make_case(tokens, experts, topk, n, k, 4321u + static_cast<uint32_t>(n + k), hot);
  const Device d = upload(c);
  const int rows = static_cast<int>(c.act_rows.size());
  const int ns = static_cast<int>(c.segs.size());
  uint16_t* act = dev(c.act);
  MoeSegment* segs = dev(c.segs);
  int32_t* act_rows = dev(c.act_rows);
  const size_t ss = act_fp8_block_scale_stride(k);
  const size_t out_stride = static_cast<size_t>(n);
  auto* codes = dev_alloc<uint8_t>(static_cast<size_t>(tokens) * k);
  auto* scales = dev_alloc<uint8_t>(static_cast<size_t>(tokens) * ss);
  auto* qdq = dev_alloc<uint16_t>(static_cast<size_t>(tokens) * k);
  auto* qdq_ref = dev_alloc<uint16_t>(static_cast<size_t>(tokens) * k);
  auto* out8 = dev_alloc<float>(static_cast<size_t>(rows) * out_stride);
  auto* out8b = dev_alloc<uint16_t>(static_cast<size_t>(rows) * out_stride);
  auto* out16 = dev_alloc<float>(static_cast<size_t>(rows) * out_stride);
  auto* out8t = dev_alloc<float>(static_cast<size_t>(rows) * out_stride);
  auto* out8bt = dev_alloc<uint16_t>(static_cast<size_t>(rows) * out_stride);
  const int tile_cap = moe_tile_list_capacity(ns, rows, kMoeA8w4TileRows);
  auto* tiles = dev_alloc<MoeTile>(static_cast<size_t>(tile_cap));
  auto* tile_count = dev_alloc<int32_t>(1);
  int fails = 0;

  auto quant = [&] { launch_act_quant_fp8_block(act, k, codes, k, scales, ss, tokens, k, nullptr, qdq, k); };
  auto gemm8 = [&] {
    launch_moe_grouped_a8w4_f32(codes, k, scales, ss, act_rows, segs, ns, c.max_rows, d.views, 0, out8, out_stride, n,
                                k, nullptr);
  };
  auto gemm8b = [&] {
    launch_moe_grouped_a8w4_bf16(codes, k, scales, ss, act_rows, segs, ns, c.max_rows, d.views, 0, out8b, out_stride,
                                 n, k, nullptr);
  };
  auto gemm16 = [&] {
    launch_moe_grouped_mma_fp4_f32(qdq, k, segs, ns, c.max_rows, 0, d.views, 0, out16, out_stride, n, k, nullptr,
                                   act_rows, kMxfp4Group);
  };
  // The tile-list grid (max_rows is not read).
  auto gemm8t = [&] {
    launch_moe_grouped_a8w4_f32(codes, k, scales, ss, act_rows, segs, ns, 0, d.views, 0, out8t, out_stride, n, k,
                                nullptr, tiles, tile_count, tile_cap);
  };
  auto gemm8bt = [&] {
    launch_moe_grouped_a8w4_bf16(codes, k, scales, ss, act_rows, segs, ns, 0, d.views, 0, out8bt, out_stride, n, k,
                                 nullptr, tiles, tile_count, tile_cap);
  };
  quant();
  launch_act_qdq_fp8_block(act, k, qdq_ref, k, tokens, k, nullptr);
  launch_moe_tile_list(segs, ns, kMoeA8w4TileRows, tiles, tile_count, nullptr);
  gemm8();
  gemm8b();
  gemm16();
  gemm8t();
  gemm8bt();
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const auto qc = host(codes, static_cast<size_t>(tokens) * k);
  const auto qs = host(scales, static_cast<size_t>(tokens) * ss);
  const auto qy = host(qdq, static_cast<size_t>(tokens) * k);
  const auto qr = host(qdq_ref, static_cast<size_t>(tokens) * k);
  const auto o8 = host(out8, static_cast<size_t>(rows) * out_stride);
  const auto o8b = host(out8b, static_cast<size_t>(rows) * out_stride);
  const auto o16 = host(out16, static_cast<size_t>(rows) * out_stride);
  const auto o8t = host(out8t, static_cast<size_t>(rows) * out_stride);
  const auto o8bt = host(out8bt, static_cast<size_t>(rows) * out_stride);

  // 1. the quantizer against the host helpers; its BF16 rows against the qdq launch.
  {
    size_t bad_codes = 0, bad_scales = 0, bad_y = 0;
    const int row_step = check_all ? 1 : std::max(1, tokens / 64);
    for (int r = 0; r < tokens; r += row_step) {
      for (int g = 0; g < k / 32; ++g) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j)
          amax = std::max(amax, std::fabs(bf16_bits_to_float(c.act[static_cast<size_t>(r) * k + g * 32 + j])));
        const uint8_t sb = latent_fp8_block_scale_byte(amax);
        const float s = e8m0_byte_to_float(sb);
        bad_scales += qs[static_cast<size_t>(r) * ss + g] != sb;
        for (int j = 0; j < 32; ++j) {
          const size_t at = static_cast<size_t>(r) * k + g * 32 + j;
          const uint8_t code = latent_fp8_block_encode(bf16_bits_to_float(c.act[at]), s);
          bad_codes += qc[at] != code;
          bad_y += qy[at] != latent_fp8_block_decode_bf16(code, s);
        }
      }
    }
    const bool same = std::memcmp(qy.data(), qr.data(), qy.size() * 2) == 0;
    std::printf("[ .. ] %s quantizer: %zu codes, %zu scales, %zu rows' values off the host's; qdq launch %s\n", label,
                bad_codes, bad_scales, bad_y, same ? "bitwise" : "DIFFERS");
    if (bad_codes || bad_scales || bad_y || !same) {
      std::printf("[FAIL] %s: the quantizer's codes are not the reference's\n", label);
      ++fails;
    }
  }

  // 2. layouts: the fp64 dot of the quantized operands.
  {
    double worst = 0;
    size_t checked = 0;
    for (const MoeSegment& s : c.segs) {
      const Matrix& m = c.w[static_cast<size_t>(s.expert)];
      const int row_step = check_all ? 1 : std::max(1, s.rows / 3);
      const int col_step = check_all ? 1 : 37;
      for (int r = 0; r < s.rows; r += row_step) {
        const int tok = c.act_rows[static_cast<size_t>(s.row0 + r)];
        for (int col = 0; col < n; col += col_step) {
          double dot = 0, mag = 0;
          for (int kk = 0; kk < k; ++kk) {
            const double a = e4m3(qc[static_cast<size_t>(tok) * k + kk]) * e8m0(qs[static_cast<size_t>(tok) * ss + kk / 32]);
            const uint8_t wb = m.codes[static_cast<size_t>(col) * k / 2 + kk / 2];
            const double w = e2m1(kk % 2 ? wb >> 4 : wb & 15) * e8m0(m.scales[static_cast<size_t>(col) * (k / 32) + kk / 32]);
            dot += a * w;
            mag += std::fabs(a * w);
          }
          const double got = o8[static_cast<size_t>(s.row0 + r) * out_stride + col];
          worst = std::max(worst, std::fabs(got - dot) / (mag + 1e-30));
          ++checked;
        }
      }
    }
    std::printf("[ .. ] %s layouts: worst |got - fp64| / sum|terms| = %.3g over %zu outputs\n", label, worst, checked);
    if (!(worst < 1e-5)) {
      std::printf("[FAIL] %s: the kernel does not compute the quantized dot\n", label);
      ++fails;
    }
  }

  // 3. the BF16 output form: the FP32 form rounded.
  {
    size_t bad = 0;
    for (size_t i = 0; i < o8.size(); ++i) bad += o8b[i] != bf16(o8[i]);
    std::printf("[ .. ] %s bf16 output: %zu of %zu differ from the rounded fp32 form\n", label, bad, o8.size());
    if (bad) {
      std::printf("[FAIL] %s: the two output forms disagree\n", label);
      ++fails;
    }
  }

  // 4. the same sums on the BF16-activation kernel (the dequantized rows).
  {
    double num = 0, den = 0;
    for (size_t i = 0; i < o8.size(); ++i) {
      num += (static_cast<double>(o8[i]) - o16[i]) * (static_cast<double>(o8[i]) - o16[i]);
      den += static_cast<double>(o16[i]) * o16[i];
    }
    const double rel = std::sqrt(num / (den + 1e-300));
    std::printf("[ .. ] %s against the BF16-activation kernel: relative l2 %.3g\n", label, rel);
    if (!(rel < 1e-5)) {
      std::printf("[FAIL] %s: the two kernels' sums differ beyond summation order\n", label);
      ++fails;
    }
  }

  // 5. the SwiGLU producer's codes against its quantize-dequantize form
  // (the first rows as gate and up, a weight per row).
  if (tokens >= 2) {
    const int pr = std::min(tokens / 2, 64);
    std::vector<float> weights(static_cast<size_t>(pr));
    for (int r = 0; r < pr; ++r) weights[static_cast<size_t>(r)] = 0.05f + 0.013f * static_cast<float>(r % 17);
    float* w_dev = dev(weights);
    auto* pc = dev_alloc<uint8_t>(static_cast<size_t>(pr) * k);
    auto* ps = dev_alloc<uint8_t>(static_cast<size_t>(pr) * ss);
    auto* py = dev_alloc<uint16_t>(static_cast<size_t>(pr) * k);
    const uint16_t* gate = act;
    const uint16_t* up = act + static_cast<size_t>(pr) * k;
    launch_swiglu_weight_act_quant_fp8_block(gate, up, k, pc, k, ps, ss, pr, k, 10.0f, w_dev, nullptr);
    launch_swiglu_weight_act_qdq_fp8_block(gate, up, k, py, pr, k, 10.0f, w_dev, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const auto hc = host(pc, static_cast<size_t>(pr) * k);
    const auto hs = host(ps, static_cast<size_t>(pr) * ss);
    const auto hy = host(py, static_cast<size_t>(pr) * k);
    size_t bad = 0;
    for (int r = 0; r < pr; ++r)
      for (int j = 0; j < k; ++j) {
        const size_t at = static_cast<size_t>(r) * k + j;
        const float s = e8m0_byte_to_float(hs[static_cast<size_t>(r) * ss + j / 32]);
        bad += latent_fp8_block_decode_bf16(hc[at], s) != hy[at];
      }
    std::printf("[ .. ] %s swiglu producer: %zu of %d values differ from the qdq form\n", label, bad, pr * k);
    if (bad) {
      std::printf("[FAIL] %s: the SwiGLU producer's codes are not its qdq form's\n", label);
      ++fails;
    }
    cudaFree(w_dev);
    cudaFree(pc);
    cudaFree(ps);
    cudaFree(py);
  }

  // 6. the tile-list grid: the segment grid's outputs, every one.
  {
    const int count = host(tile_count, 1)[0];
    int want = 0;
    for (const MoeSegment& sg : c.segs) want += (sg.rows + kMoeA8w4TileRows - 1) / kMoeA8w4TileRows;
    const bool f32_same = std::memcmp(o8t.data(), o8.data(), o8.size() * sizeof(float)) == 0;
    const bool bf16_same = std::memcmp(o8bt.data(), o8b.data(), o8b.size() * sizeof(uint16_t)) == 0;
    std::printf("[ .. ] %s tile list: %d tiles (%d expected, capacity %d); fp32 rows %s, bf16 rows %s\n", label, count,
                want, tile_cap, f32_same ? "bitwise" : "DIFFER", bf16_same ? "bitwise" : "DIFFER");
    if (count != want || count > tile_cap || !f32_same || !bf16_same) {
      std::printf("[FAIL] %s: the tile-list grid is not the segment grid's result\n", label);
      ++fails;
    }
  }

  // 7. speed.
  if (speed) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    auto time = [&](auto f) {
      for (int i = 0; i < 3; ++i) f();
      cudaEventRecord(e0);
      for (int i = 0; i < 20; ++i) f();
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      float ms = 0;
      cudaEventElapsedTime(&ms, e0, e1);
      return ms / 20;
    };
    auto qdq_only = [&] { launch_act_qdq_fp8_block(act, k, qdq_ref, k, tokens, k, nullptr); };
    auto quant_only = [&] { launch_act_quant_fp8_block(act, k, codes, k, scales, ss, tokens, k, nullptr); };
    auto gemm16b = [&] {
      launch_moe_grouped_mma_fp4_bf16(qdq, k, segs, ns, c.max_rows, 0, d.views, 0,
                                      reinterpret_cast<uint16_t*>(out16), out_stride, n, k, nullptr, act_rows,
                                      kMxfp4Group);
    };
    const float t16 = time(gemm16), t16b = time(gemm16b), t8 = time(gemm8), t8b = time(gemm8b);
    const float t8t = time(gemm8t), t8bt = time(gemm8bt);
    const float tq = time(quant_only), tqdq = time(qdq_only);
    const double flop = 2.0 * rows * n * k;
    std::printf("[ .. ] %s %d rows (%d segments, longest %d) x n %d x k %d:\n", label, rows, ns, c.max_rows, n, k);
    std::printf("[ .. ]   BF16 activations: f32 out %.3f ms (%.1f TF), bf16 out %.3f ms (%.1f TF); qdq %.3f ms\n", t16,
                flop / t16 / 1e9, t16b, flop / t16b / 1e9, tqdq);
    std::printf("[ .. ]   E4M3 activations: f32 out %.3f ms (%.1f TF), bf16 out %.3f ms (%.1f TF); quantize %.3f ms\n",
                t8, flop / t8 / 1e9, t8b, flop / t8b / 1e9, tq);
    std::printf("[ .. ]   E4M3 activations, tile list: f32 out %.3f ms (%.1f TF), bf16 out %.3f ms (%.1f TF)\n", t8t,
                flop / t8t / 1e9, t8bt, flop / t8bt / 1e9);
    std::printf("[ .. ]   speedup f32 %.2fx, bf16 %.2fx; tile list over the BF16 kernel f32 %.2fx, bf16 %.2fx\n",
                t16 / t8, t16b / t8b, t16 / t8t, t16b / t8bt);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
  }
  for (auto* p : d.bufs) cudaFree(p);
  cudaFree(d.views);
  cudaFree(act);
  cudaFree(segs);
  cudaFree(act_rows);
  cudaFree(codes);
  cudaFree(scales);
  cudaFree(qdq);
  cudaFree(qdq_ref);
  cudaFree(out8);
  cudaFree(out8b);
  cudaFree(out16);
  cudaFree(out8t);
  cudaFree(out8bt);
  cudaFree(tiles);
  cudaFree(tile_count);
  return fails;
}

}  // namespace

int main(int argc, char** argv) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("[SKIP] moe_a8w4_test: no CUDA device\n");
    return 2;
  }
  int tokens = 4096, experts = 384, topk = 6;
  bool speed = true;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--tokens" && i + 1 < argc) tokens = std::atoi(argv[++i]);
    else if (a == "--experts" && i + 1 < argc) experts = std::atoi(argv[++i]);
    else if (a == "--topk" && i + 1 < argc) topk = std::atoi(argv[++i]);
    else if (a == "--no-speed") speed = false;
  }
  int fails = 0;
  // A ragged shape, every output checked: segments of 1 to a few hundred
  // rows, a last n-tile of 72 columns, five stages.
  fails += run(301, 7, 3, 200, 320, true, false, "ragged");
  // One stage, one short segment.
  fails += run(5, 2, 1, 128, 64, true, false, "one stage");
  // One expert in every token's routes: a segment of several m-tiles beside short ones.
  fails += run(301, 7, 3, 200, 320, true, false, "ragged hot", true);
  // The DeepSeek-V4.1 shapes at four ranks: gate and up, then down, on even
  // routing and with the hot expert a served layer has.
  fails += run(tokens, experts, topk, 576, 5120, false, speed, "gate/up");
  fails += run(tokens, experts, topk, 5120, 576, false, speed, "down");
  fails += run(tokens, experts, topk, 576, 5120, false, speed, "gate/up hot", true);
  fails += run(tokens, experts, topk, 5120, 576, false, speed, "down hot", true);
  if (fails) {
    std::printf("[FAIL] moe_a8w4_test: %d check(s) failed\n", fails);
    return 1;
  }
  std::printf("[PASS] moe_a8w4_test\n");
  return 0;
}
