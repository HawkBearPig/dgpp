// Parity tests for the scale-aware GEMM (M4 deliverable 3: "block edges,
// saturation, NaN/Inf policy, and real checkpoint slices" — the real-slice
// half runs via --checkpoint-dir, mirroring the dump-parity runner
// pattern). Every synthetic case is checked against both oracles: strict
// (bf16-rounded weights, fp64 accumulation — isolates the kernel) and
// semantic (true dequant — pins the DESIGN §4 scale contract).
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <cstdio>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/test.hpp"
#include "kernels/fp8_gemv.cuh"
#include "kernels/bf16_gemv.hpp"
#include "kernels/scale_gemm.hpp"
#include "kernels/gemv_wide.hpp"
#include "scale_gemm_test_helpers.hpp"
#include "kda_test_helpers.hpp"

namespace {

using namespace scale_gemm_test;

struct Problem {
  int m, n, k;
  std::vector<uint16_t> act;      // [m, k]
  std::vector<uint8_t> payload;   // [n, k]
  std::vector<float> scales;      // [ceil(n/128), ceil(k/128)]
};

Problem make_problem(int m, int n, int k, uint64_t seed) {
  Problem p;
  p.m = m;
  p.n = n;
  p.k = k;
  Rng rng(seed);
  p.act.resize(static_cast<size_t>(m) * k);
  p.payload.resize(static_cast<size_t>(n) * k);
  p.scales.resize(static_cast<size_t>((n + 127) / 128) * ((k + 127) / 128));
  fill_act(rng, p.act);
  fill_payload(rng, p.payload);
  fill_scales(rng, p.scales);
  return p;
}

// Runs the kernel into managed memory and returns the bf16 output.
std::vector<uint16_t> run_kernel(const Problem& p) {
  uint16_t* act = nullptr;
  uint8_t* w = nullptr;
  float* s = nullptr;
  uint16_t* out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&act, p.act.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&w, p.payload.size()));
  DGPP_CUDA_OK(cudaMallocManaged(&s, p.scales.size() * 4));
  DGPP_CUDA_OK(
      cudaMallocManaged(&out, static_cast<size_t>(p.m) * p.n * 2));
  std::memcpy(act, p.act.data(), p.act.size() * 2);
  std::memcpy(w, p.payload.data(), p.payload.size());
  std::memcpy(s, p.scales.data(), p.scales.size() * 4);
  dgpp::launch_scale_gemm_bf16(act, p.k, w, s, out, p.m, p.n, p.k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> got(static_cast<size_t>(p.m) * p.n);
  std::memcpy(got.data(), out, got.size() * 2);
  DGPP_CUDA_OK(cudaFree(act));
  DGPP_CUDA_OK(cudaFree(w));
  DGPP_CUDA_OK(cudaFree(s));
  DGPP_CUDA_OK(cudaFree(out));
  return got;
}

// The same production dispatch with its unrounded f32 epilogue. Used by the
// row-independence gate because the MoE accumulation consumes these bits.
std::vector<float> run_kernel_f32(const Problem& p, int mma_from_rows = 0,
                                  int expected_mma_warps = 0) {
  uint16_t* act = nullptr;
  uint8_t* w = nullptr;
  float* s = nullptr;
  float* out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&act, p.act.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&w, p.payload.size()));
  DGPP_CUDA_OK(cudaMallocManaged(&s, p.scales.size() * 4));
  DGPP_CUDA_OK(
      cudaMallocManaged(&out, static_cast<size_t>(p.m) * p.n * 4));
  std::memcpy(act, p.act.data(), p.act.size() * 2);
  std::memcpy(w, p.payload.data(), p.payload.size());
  std::memcpy(s, p.scales.data(), p.scales.size() * 4);
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  if (expected_mma_warps) {
    DGPP_CUDA_OK(cudaStreamCreate(&stream));
    DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  }
  dgpp::launch_scale_gemm_f32(act, p.k, w, s, out, p.m, p.n, p.k, stream, 0, mma_from_rows);
  bool width_matches = true;
  if (expected_mma_warps) {
    DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
    size_t count = 0;
    DGPP_CUDA_OK(cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count);
    DGPP_CUDA_OK(cudaGraphGetNodes(graph, nodes.data(), &count));
    width_matches = count == 1;
    if (count == 1) {
      cudaKernelNodeParams params{};
      DGPP_CUDA_OK(cudaGraphKernelNodeGetParams(nodes[0], &params));
      const char* name = nullptr;
      DGPP_CUDA_OK(cudaFuncGetName(&name, params.func));
      width_matches = std::string(name).find("mma_gemv_kernel") != std::string::npos &&
                      params.blockDim.x == static_cast<unsigned>(32 * expected_mma_warps);
    }
    DGPP_CUDA_OK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    DGPP_CUDA_OK(cudaGraphLaunch(executable, stream));
  }
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  if (expected_mma_warps) {
    DGPP_CUDA_OK(cudaGraphExecDestroy(executable));
    DGPP_CUDA_OK(cudaGraphDestroy(graph));
    DGPP_CUDA_OK(cudaStreamDestroy(stream));
  }
  std::vector<float> got(static_cast<size_t>(p.m) * p.n);
  std::memcpy(got.data(), out, got.size() * 4);
  DGPP_CUDA_OK(cudaFree(act));
  DGPP_CUDA_OK(cudaFree(w));
  DGPP_CUDA_OK(cudaFree(s));
  DGPP_CUDA_OK(cudaFree(out));
  require(width_matches, "FP8 head did not select the expected MMA warp width");
  return got;
}

// The tile kernel regardless of m (the routed launcher sends m <= 128 to
// the GEMV core; the grouped tensor-core MoE kernel is bitwise this one).
std::vector<uint16_t> run_tile_kernel(const Problem& p) {
  uint16_t* act = nullptr;
  uint8_t* w = nullptr;
  float* s = nullptr;
  uint16_t* out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&act, p.act.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&w, p.payload.size()));
  DGPP_CUDA_OK(cudaMallocManaged(&s, p.scales.size() * 4));
  DGPP_CUDA_OK(
      cudaMallocManaged(&out, static_cast<size_t>(p.m) * p.n * 2));
  std::memcpy(act, p.act.data(), p.act.size() * 2);
  std::memcpy(w, p.payload.data(), p.payload.size());
  std::memcpy(s, p.scales.data(), p.scales.size() * 4);
  dgpp::launch_scale_gemm_tile_bf16(act, p.k, w, s, out, p.m, p.n, p.k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> got(static_cast<size_t>(p.m) * p.n);
  std::memcpy(got.data(), out, got.size() * 2);
  DGPP_CUDA_OK(cudaFree(act));
  DGPP_CUDA_OK(cudaFree(w));
  DGPP_CUDA_OK(cudaFree(s));
  DGPP_CUDA_OK(cudaFree(out));
  return got;
}

// Runs both oracles and asserts the budgets.
void check_both_oracles(const Problem& p, const std::vector<uint16_t>& got,
                        const char* label) {
  const auto strict = gemm_oracle(p.act, p.k, p.payload, p.scales, p.m, p.n,
                                  p.k, /*strict=*/true);
  auto rep = compare_bf16_vs_oracle(got.data(), strict, /*ulp_budget=*/2.0,
                                    /*floor_frac=*/1e-3);
  std::printf("[ OK ] %s strict: l2_rel=%.3g mismatches=%ld/%zu\n", label,
              rep.l2_rel, rep.mismatches, rep.total);
  require_report(rep, 1e-3, 0, (std::string(label) + " strict").c_str());

  const auto semantic = gemm_oracle(p.act, p.k, p.payload, p.scales, p.m,
                                    p.n, p.k, /*strict=*/false);
  rep = compare_bf16_vs_oracle(got.data(), semantic, /*ulp_budget=*/8.0,
                               /*floor_frac=*/2e-2);
  std::printf("[ OK ] %s semantic: l2_rel=%.3g mismatches=%ld/%zu\n", label,
              rep.l2_rel, rep.mismatches, rep.total);
  require_report(rep, 8e-3, static_cast<long>(rep.total) / 20,
                 (std::string(label) + " semantic").c_str());
}

}  // namespace

// Vocabulary-head dimensions in K, with ragged two- and eight-warp slices. Check
// unrounded FP32 logits against an independent FP64 accumulation of the
// exact BF16-rounded weights, including the GEMV/MMA transition at row 5.
DGPP_TEST(scale_gemm_f32_fp8_head_numerics) {
  // N=4097 selects the same eight-warp specialization as a production
  // TP2 vocabulary shard, while keeping the exhaustive FP64 oracle bounded.
  // Its final block also exercises masked columns in that specialization.
  for (const int n : {257, 4097})
    for (const int m : {4, 5, 8, 16}) {
      const auto p = make_problem(m, n, 2560, 0x12345678 + m + n);
      const auto got = run_kernel_f32(p, 5, n == 4097 && m > 4 ? 8 : 0);
      const auto repeat = run_kernel_f32(p, 5);
      const auto gemv = run_kernel_f32(p);
      require(std::memcmp(got.data(), repeat.data(), got.size() * sizeof(float)) == 0,
              "FP8 head must repeat bitwise at a fixed shape");
      double error2 = 0, reference2 = 0;
      const int scale_cols = (p.k + 127) / 128;
      for (int row = 0; row < p.m; ++row) {
        for (int col = 0; col < p.n; ++col) {
          double reference = 0, magnitude = 0;
          for (int k = 0; k < p.k; ++k) {
            const float decoded = dgpp::fp8_e4m3_bits_to_float(p.payload[col * p.k + k]);
            const float scale = p.scales[(col / 128) * scale_cols + k / 128];
            const double weight = bf16_to_float(dgpp::float_to_bf16_bits(decoded * scale));
            const double product = bf16_to_float(p.act[row * p.k + k]) * weight;
            reference += product;
            magnitude += std::abs(product);
          }
          const size_t i = static_cast<size_t>(row) * p.n + col;
          require(std::isfinite(got[i]) && std::isfinite(gemv[i]), "nonfinite FP8 head logit");
          require(std::abs(got[i] - reference) <= 3e-5 * magnitude,
                  "streaming FP8 head exceeds the FP64 error budget");
          require(std::abs(gemv[i] - reference) <= 3e-5 * magnitude,
                  "reference GEMV exceeds the FP64 error budget");
          error2 += (got[i] - reference) * (got[i] - reference);
          reference2 += reference * reference;
        }
      }
      require(std::sqrt(error2 / reference2) < 1e-5, "FP8 head relative L2 error");
    }
}

DGPP_TEST(scale_gemm_full_blocks_match_both_oracles) {
  // Exact 128-block geometry with n/k tiles straddling block boundaries:
  // N=384 spans block rows 0-2; K=256 crosses the k=128 boundary mid-loop.
  const Problem p = make_problem(29, 384, 256, 0xA1);
  const std::vector<uint16_t> got = run_kernel(p);
  check_both_oracles(p, got, "full-blocks M29xN384xK256");

  // Determinism: a second run must be bitwise identical.
  const std::vector<uint16_t> again = run_kernel(p);
  require(std::memcmp(got.data(), again.data(), got.size() * 2) == 0,
          "second run bitwise identical");
}

DGPP_TEST(scale_gemm_tile_kernel_small_m_matches_both_oracles) {
  // The tile kernel below the routed launcher's GEMV threshold: partial
  // m-tiles (M=1, 5, 17) at the MoE small-case geometry — the prefill's
  // grouped tensor-core kernel is bitwise this kernel per segment, so its
  // accuracy at every segment length is this gate.
  for (const int m : {1, 5, 17}) {
    Problem p = make_problem(m, /*n=*/256, /*k=*/512, 0x7113 + m);
    const auto got = run_tile_kernel(p);
    check_both_oracles(p, got, ("tile M" + std::to_string(m) + "xN256xK512").c_str());
  }
}

// The wide GEMV chunk (gemv_wide.hpp, 2026-10-14): five to eight rows in
// one launch below the streaming tensor-core bound (mma_from_rows 9 — the
// Flash's decode batch) — each row bitwise the row alone, both at the
// Flash's 4096-wide activation (the opted-in 64 KB of staged rows) and at
// a narrow k, and inside both oracles' budgets.
DGPP_TEST(scale_gemm_wide_gemv_rows_bitwise_the_rows_alone) {
  require(dgpp::gemv::enable_wide_rows(), "the wide chunk's shared-memory opt-in took");
  // The Flash's 4096-wide rows and a narrow k, then MiMo's fixture shapes
  // (k = 256: one chunk per lane; the fused qkv's fp32 rows at n = 3008)
  // and a k of a single warp step — every row of every wide launch, f32 and
  // bf16 out, bitwise the row alone.
  for (const auto [n, k] : std::vector<std::pair<int, int>>{{384, 512}, {384, 4096}, {512, 256}, {256, 512},
                                                             {3008, 256}, {200, 64}, {136, 128}}) {
    for (int m = 5; m <= 8; ++m) {
      const Problem p = make_problem(m, n, k, 0x7a00 + m + k + 3 * n);
      const auto got = run_kernel_f32(p, /*mma_from_rows=*/9);
      const auto got16 = run_kernel(p);
      for (int r = 0; r < m; ++r) {
        Problem one = p;
        one.m = 1;
        one.act.assign(p.act.begin() + static_cast<long>(r) * k, p.act.begin() + static_cast<long>(r + 1) * k);
        const auto alone = run_kernel_f32(one, /*mma_from_rows=*/9);
        const auto alone16 = run_kernel(one);
        const std::string shape = "M" + std::to_string(m) + "xN" + std::to_string(n) + "xK" + std::to_string(k) +
                                  " row " + std::to_string(r);
        require(std::memcmp(alone.data(), got.data() + static_cast<size_t>(r) * p.n, static_cast<size_t>(p.n) * 4) == 0,
                ("a wide fp8 launch's f32 row is bitwise the row alone: " + shape).c_str());
        require(std::memcmp(alone16.data(), got16.data() + static_cast<size_t>(r) * p.n,
                            static_cast<size_t>(p.n) * 2) == 0,
                ("a wide fp8 launch's bf16 row is bitwise the row alone: " + shape).c_str());
      }
      check_both_oracles(p, got16, ("wide M" + std::to_string(m) + "xN" + std::to_string(n) + "xK" + std::to_string(k)).c_str());
    }
  }
}

// The wide chunk through the explicit-grid entry (MiMo's fused qkv and dense
// MLP: a strided activation view, an output row stride past n, f32 and bf16
// out): rows 5..8 in one launch against the same rows launched alone.
DGPP_TEST(scale_gemm_grid_wide_rows_strided_bitwise_the_rows_alone) {
  using dgpp::kda_test::DevBuf;
  require(dgpp::gemv::enable_wide_rows(), "the wide chunk's shared-memory opt-in took");
  for (const auto [n, k] : std::vector<std::pair<int, int>>{{3008, 256}, {512, 256}, {256, 512}, {200, 64}}) {
    for (int m = 5; m <= 8; ++m) {
      const size_t as = k + 8, os = n + 16;
      Rng rng(0x5A1D + m * 1000 + k + n);
      std::vector<uint16_t> act(m * as);
      std::vector<uint8_t> weights(static_cast<size_t>(n) * k);
      std::vector<float> scales(((n + 127) >> 7) * ((k + 127) >> 7));
      fill_act(rng, act); fill_payload(rng, weights); fill_scales(rng, scales);
      DevBuf da(act.size() * 2), dw(weights.size()), ds(scales.size() * 4), d32(m * os * 4), r32(m * os * 4),
          d16(m * os * 2), r16(m * os * 2);
      da.upload(act.data(), act.size() * 2); dw.upload(weights.data(), weights.size());
      ds.upload(scales.data(), scales.size() * 4);
      auto* a = static_cast<const uint16_t*>(da.p);
      auto* w = static_cast<const uint8_t*>(dw.p);
      auto* sc = static_cast<const float*>(ds.p);
      DGPP_CUDA_OK(cudaMemset(d32.p, 0, m * os * 4)); DGPP_CUDA_OK(cudaMemset(r32.p, 0, m * os * 4));
      DGPP_CUDA_OK(cudaMemset(d16.p, 0, m * os * 2)); DGPP_CUDA_OK(cudaMemset(r16.p, 0, m * os * 2));
      // All m rows in one call (one wide launch), then each row alone into the same row of the reference.
      dgpp::launch_scale_gemm_grid_f32(a, as, w, sc, static_cast<float*>(d32.p), m, n, k, nullptr, os, 7, 7);
      dgpp::launch_scale_gemm_grid_bf16(a, as, w, sc, static_cast<uint16_t*>(d16.p), m, n, k, nullptr, os, 7, 7);
      for (int r = 0; r < m; ++r) {
        dgpp::launch_scale_gemm_grid_f32(a + static_cast<size_t>(r) * as, as, w, sc,
                                         static_cast<float*>(r32.p) + static_cast<size_t>(r) * os, 1, n, k, nullptr,
                                         os, 7, 7);
        dgpp::launch_scale_gemm_grid_bf16(a + static_cast<size_t>(r) * as, as, w, sc,
                                          static_cast<uint16_t*>(r16.p) + static_cast<size_t>(r) * os, 1, n, k,
                                          nullptr, os, 7, 7);
      }
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::vector<float> got32(m * os), ref32(m * os);
      std::vector<uint16_t> got16(m * os), ref16(m * os);
      d32.download(got32.data(), got32.size() * 4); r32.download(ref32.data(), ref32.size() * 4);
      d16.download(got16.data(), got16.size() * 2); r16.download(ref16.data(), ref16.size() * 2);
      const std::string shape = "M" + std::to_string(m) + "xN" + std::to_string(n) + "xK" + std::to_string(k);
      require(std::memcmp(got32.data(), ref32.data(), got32.size() * 4) == 0,
              ("grid wide f32 rows bitwise the rows alone: " + shape).c_str());
      require(std::memcmp(got16.data(), ref16.data(), got16.size() * 2) == 0,
              ("grid wide bf16 rows bitwise the rows alone: " + shape).c_str());
      std::printf("[ OK ] grid wide %s strided: f32 and bf16 rows bitwise the rows alone\n", shape.c_str());
    }
  }
}

DGPP_TEST(scale_gemm_large_m_route_is_bitwise_the_tile_kernel) {
  // m > 128 routes to the 128-row tensor-core kernel (the MoE experts'
  // dense form): bitwise the tile kernel — the same dequantized weights,
  // the same ascending-k16 mma chain — at a ragged n and m (three m-tiles,
  // the last partial), and within both oracles' budgets.
  Problem p = make_problem(/*m=*/300, /*n=*/200, /*k=*/512, 0x1A26E);
  const auto routed = run_kernel(p);
  const auto tile = run_tile_kernel(p);
  require(std::memcmp(routed.data(), tile.data(), routed.size() * 2) == 0,
          "large-m route bitwise the tile kernel");
  check_both_oracles(p, routed, "large-m M300xN200xK512");
}

DGPP_TEST(scale_gemm_explicit_grid_large_m_dispatch_and_bits) {
  using dgpp::kda_test::DevBuf;
  // MiMo's 128x128 grid should reach the existing fast kernel. Smaller
  // scale grids and unsupported K must retain the tile path. Compare
  // both output types, including padding, against the original kernel.
  for (int rs : {5, 6, 7}) for (int cs : {5, 6, 7}) for (int k : {144, 192}) {
    constexpr int m = 129, n = 200;
    const size_t as = k + 8, os = n + 16;
    Rng rng(0x12800 + rs * 100 + cs * 10 + k);
    std::vector<uint16_t> act(m * as), out16(m * os, 0xBEEF), ref16 = out16;
    std::vector<uint8_t> weights(n * k);
    std::vector<float> scales(((n + (1 << rs) - 1) >> rs) * ((k + (1 << cs) - 1) >> cs));
    std::vector<float> out32(m * os, -12345.f), ref32 = out32;
    fill_act(rng, act); fill_payload(rng, weights); fill_scales(rng, scales);
    DevBuf da(act.size() * 2), dw(weights.size()), ds(scales.size() * 4),
        d16(out16.size() * 2), r16(ref16.size() * 2), d32(out32.size() * 4), r32(ref32.size() * 4);
    da.upload(act.data(), act.size() * 2); dw.upload(weights.data(), weights.size());
    ds.upload(scales.data(), scales.size() * 4);
    d16.upload(out16.data(), out16.size() * 2); r16.upload(ref16.data(), ref16.size() * 2);
    d32.upload(out32.data(), out32.size() * 4); r32.upload(ref32.data(), ref32.size() * 4);
    auto* a = static_cast<const uint16_t*>(da.p);
    auto* w = static_cast<const uint8_t*>(dw.p);
    auto* s = static_cast<const float*>(ds.p);
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    DGPP_CUDA_OK(cudaStreamCreate(&stream));
    DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    dgpp::launch_scale_gemm_grid_f32(a, as, w, s, static_cast<float*>(d32.p),
        m, n, k, stream, os, rs, cs);
    DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
    size_t count = 0;
    DGPP_CUDA_OK(cudaGraphGetNodes(graph, nullptr, &count));
    require(count == 1, "explicit-grid GEMM must launch one kernel");
    cudaGraphNode_t node;
    DGPP_CUDA_OK(cudaGraphGetNodes(graph, &node, &count));
    cudaKernelNodeParams params{};
    DGPP_CUDA_OK(cudaGraphKernelNodeGetParams(node, &params));
    const char* name = nullptr;
    DGPP_CUDA_OK(cudaFuncGetName(&name, params.func));
    const bool fast = std::string(name).find("fp8w_gemm_kernel") != std::string::npos;
    require(fast == (rs == 7 && cs == 7 && k % 64 == 0),
            "explicit-grid GEMM selected the wrong path");
    DGPP_CUDA_OK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    DGPP_CUDA_OK(cudaGraphLaunch(executable, stream));
    dgpp::launch_scale_gemm_grid_bf16(a, as, w, s, static_cast<uint16_t*>(d16.p),
        m, n, k, stream, os, rs, cs);
    dgpp::launch_scale_gemm_tile_f32(a, as, w, s, static_cast<float*>(r32.p),
        m, n, k, stream, rs, cs);
    dgpp::launch_scale_gemm_tile_bf16(a, as, w, s, static_cast<uint16_t*>(r16.p),
        m, n, k, stream, rs, cs);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    d16.download(out16.data(), out16.size() * 2); r16.download(ref16.data(), ref16.size() * 2);
    d32.download(out32.data(), out32.size() * 4); r32.download(ref32.data(), ref32.size() * 4);
    for (int row = 0; row < m; ++row) {
      require(std::memcmp(out16.data() + row * os, ref16.data() + row * n, n * 2) == 0 &&
              std::memcmp(out32.data() + row * os, ref32.data() + row * n, n * 4) == 0,
              "explicit-grid GEMM differs from the original tile");
      for (size_t col = n; col < os; ++col)
        require(out16[row * os + col] == 0xBEEF && out32[row * os + col] == -12345.f,
                "explicit-grid GEMM overwrites output padding");
    }
    DGPP_CUDA_OK(cudaGraphExecDestroy(executable));
    DGPP_CUDA_OK(cudaGraphDestroy(graph));
    DGPP_CUDA_OK(cudaStreamDestroy(stream));
  }
}

DGPP_TEST(scale_gemm_last_row_preserves_full_product_bits_and_output_bounds) {
  // Cross the GEMV/dense boundary with the production head's K, including
  // row positions on both sides of an MMA tile. Also cover streaming MMA
  // and the ragged-K fallback, with padded input/output strides. A workspace
  // enables split-K only for groups of at most 32 rows: cover both sides of
  // that boundary, including the final group of products above 128 rows.
  void* workspace = nullptr;
  constexpr size_t workspace_bytes = 4u << 20;
  DGPP_CUDA_OK(cudaMalloc(&workspace, workspace_bytes));
  for (const int k : {1000, 2560}) {
    for (const int m : {1, 4, 5, 16, 17, 32, 33, 64, 127, 128, 129, 144, 160, 161, 255, 256, 257}) {
      const Problem p = make_problem(m, 137, k, 0x43A0 + m + k);
      const size_t act_stride = static_cast<size_t>(k) + 8;
      const size_t out_stride = static_cast<size_t>(p.n) + 7;
      uint16_t* act = nullptr;
      uint8_t* w = nullptr;
      float *scales = nullptr, *full = nullptr, *last = nullptr;
      DGPP_CUDA_OK(cudaMallocManaged(&act, m * act_stride * sizeof(uint16_t)));
      DGPP_CUDA_OK(cudaMallocManaged(&w, p.payload.size()));
      DGPP_CUDA_OK(cudaMallocManaged(&scales, p.scales.size() * sizeof(float)));
      DGPP_CUDA_OK(cudaMallocManaged(&full, m * out_stride * sizeof(float)));
      DGPP_CUDA_OK(cudaMallocManaged(&last, m * out_stride * sizeof(float)));
      for (int row = 0; row < m; ++row)
        std::memcpy(act + row * act_stride, p.act.data() + row * k, k * sizeof(uint16_t));
      std::memcpy(w, p.payload.data(), p.payload.size());
      std::memcpy(scales, p.scales.data(), p.scales.size() * sizeof(float));
      bool equal = true, untouched = true;
      for (const int mma_from : {0, 5}) {
        for (const size_t capacity : {size_t{0}, size_t{128}, workspace_bytes}) {
          void* ws = capacity ? workspace : nullptr;
          constexpr float sentinel = -12345.0f;
          std::fill(last, last + m * out_stride, sentinel);
          dgpp::launch_scale_gemm_f32(act, act_stride, w, scales, full, m, p.n, k,
                                      nullptr, out_stride, mma_from, false, ws, capacity);
          dgpp::launch_scale_gemm_f32(act, act_stride, w, scales, last, m, p.n, k,
                                      nullptr, out_stride, mma_from, true, ws, capacity);
          DGPP_CUDA_OK(cudaDeviceSynchronize());
          const size_t offset = static_cast<size_t>(m - 1) * out_stride;
          equal &= std::memcmp(full + offset, last + offset, p.n * sizeof(float)) == 0;
          for (size_t i = 0; i < m * out_stride; ++i)
            if (i < offset || i >= offset + p.n) untouched &= last[i] == sentinel;
          // Compact selection uses a genuinely one-row allocation, including
          // padding sentinels. Memcheck catches any original-offset write.
          float* compact = nullptr;
          DGPP_CUDA_OK(cudaMallocManaged(&compact, out_stride * sizeof(float)));
          for (const int selected : {0, m / 2, m - 1}) {
            std::fill(compact, compact + out_stride, sentinel);
            dgpp::launch_scale_gemm_f32(act, act_stride, w, scales, compact, m, p.n, k,
                                        nullptr, out_stride, mma_from, false, ws, capacity, selected);
            DGPP_CUDA_OK(cudaDeviceSynchronize());
            equal &= std::memcmp(full + static_cast<size_t>(selected) * out_stride,
                                 compact, p.n * sizeof(float)) == 0;
            for (size_t i = p.n; i < out_stride; ++i) untouched &= compact[i] == sentinel;
          }
          DGPP_CUDA_OK(cudaFree(compact));
        }
      }
      DGPP_CUDA_OK(cudaFree(act));
      DGPP_CUDA_OK(cudaFree(w));
      DGPP_CUDA_OK(cudaFree(scales));
      DGPP_CUDA_OK(cudaFree(full));
      DGPP_CUDA_OK(cudaFree(last));
      require(equal, ("last-row head differs from the full product at M=" + std::to_string(m) +
                      " K=" + std::to_string(k)).c_str());
      require(untouched, "last-row head overwrote another row or output padding");
    }
  }
  DGPP_CUDA_OK(cudaFree(workspace));
}

DGPP_TEST(scale_gemm_ragged_tails_match_both_oracles) {
  // Nothing aligned: N=K=1000 (7 full blocks + 104-wide tail on both axes),
  // M=17 (ragged m-tile), and the final BK stage only 8 k-values wide.
  const Problem p = make_problem(17, 1000, 1000, 0xB2);
  const std::vector<uint16_t> got = run_kernel(p);
  check_both_oracles(p, got, "ragged M17xN1000xK1000");
}

DGPP_TEST(scale_gemm_decode_shape_single_row_real_qa_geometry) {
  // The decode-critical shape: M=1 at the real q_a geometry [1536, 4096].
  const Problem p = make_problem(1, 1536, 4096, 0xC3);
  const std::vector<uint16_t> got = run_kernel(p);
  check_both_oracles(p, got, "decode M1xN1536xK4096");
}

DGPP_TEST(scale_gemm_saturation_extremes) {
  // All-max finite payloads (0x7E = 448) with scales that push products to
  // ~1e35 (bf16/fp32 representable), and a subnormal-minimum case.
  {
    Problem p;
    p.m = 9;
    p.n = 256;
    p.k = 256;
    p.act.resize(static_cast<size_t>(p.m) * p.k);
    p.payload.assign(static_cast<size_t>(p.n) * p.k, 0x7E);
    p.scales.assign(2 * 2, 1e30f);
    Rng rng(0xD4);
    fill_act(rng, p.act);
    const std::vector<uint16_t> got = run_kernel(p);
    check_both_oracles(p, got, "saturation 448x1e30");
  }
  {
    Problem p;
    p.m = 9;
    p.n = 256;
    p.k = 256;
    p.act.resize(static_cast<size_t>(p.m) * p.k);
    // 0x01 = min subnormal (2^-9); scales 2^-20 keep products representable
    // but exercise the bottom of both exponent ranges.
    p.payload.assign(static_cast<size_t>(p.n) * p.k, 0x01);
    p.scales.assign(2 * 2, 9.53674316e-07f);  // 2^-20
    Rng rng(0xD5);
    fill_act(rng, p.act);
    const std::vector<uint16_t> got = run_kernel(p);
    check_both_oracles(p, got, "subnormal min");
  }
}

DGPP_TEST(scale_gemm_propagates_nan_exactly) {
  // e4m3fn 0x7F is NaN. Two poisoned weight entries must NaN exactly the
  // output columns they touch — every m, through the fp32 accumulator —
  // and nothing else (zero-filled tiles and finite neighbors stay finite).
  Problem p = make_problem(13, 512, 384, 0xE6);
  p.payload[static_cast<size_t>(5) * p.k + 7] = 0x7F;
  p.payload[static_cast<size_t>(130) * p.k + 300] = 0x7F;
  const std::vector<uint16_t> got = run_kernel(p);

  const auto oracle = gemm_oracle(p.act, p.k, p.payload, p.scales, p.m, p.n,
                                  p.k, /*strict=*/false);
  // NaN pattern must match element-for-element.
  for (int mm = 0; mm < p.m; ++mm) {
    for (int nn = 0; nn < p.n; ++nn) {
      const size_t i = static_cast<size_t>(mm) * p.n + nn;
      const bool want_nan = std::isnan(oracle[i]);
      const bool got_nan = std::isnan(bf16_to_float(got[i]));
      if (nn == 5 || nn == 130) {
        require(want_nan && got_nan, "poisoned column is NaN");
      } else {
        require(!want_nan && !got_nan, "unpoisoned column stays finite");
      }
    }
  }
  // Finite columns still within the semantic budget.
  auto rep = compare_bf16_vs_oracle(got.data(), oracle, 8.0, 2e-2);
  require(rep.nan_pattern_errors == 0, "nan pattern");
  require(rep.mismatches <= static_cast<long>(rep.total) / 20, "nan test mismatches");
  std::printf("[ OK ] nan propagation: columns 5 and 130 NaN, others finite\n");
}

DGPP_TEST(scale_gemm_degenerate_k_zeroes_output) {
  Problem p = make_problem(4, 64, 0, 0xF7);
  p.payload.clear();
  p.scales.clear();
  p.act.clear();
  p.act.resize(4);  // act unused for k=0
  uint16_t* act = nullptr;
  uint16_t* out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&act, 8));
  DGPP_CUDA_OK(cudaMallocManaged(&out, 4 * 64 * 2));
  std::memset(out, 0xFF, 4 * 64 * 2);  // garbage
  dgpp::launch_scale_gemm_bf16(act, 0, reinterpret_cast<const uint8_t*>(act),
                               reinterpret_cast<const float*>(act), out, 4,
                               64, 0, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  for (size_t i = 0; i < 4 * 64; ++i)
    if (out[i] != 0) throw std::runtime_error("k=0 must zero outputs");
  DGPP_CUDA_OK(cudaFree(act));
  DGPP_CUDA_OK(cudaFree(out));
}

// ---- the m<=4 GEMV path (fp8_gemv.cuh) ------------------------------------

__global__ void e4m3_convert_all_codes_kernel(float* hw, float* ref) {
  // GIVEN every byte code, converted by the GEMV's hardware path and by the
  // dequant bridge's reference function.
  const int v = threadIdx.x;
  const float2 pair = dgpp::fp8_gemv::e4m3x2_to_float2(
      static_cast<uint16_t>(v | (v << 8)));
  hw[v] = pair.x;
  hw[256 + v] = pair.y;  // the high byte lands in .y
  ref[v] = dgpp::fp8_e4m3_bits_to_float(static_cast<uint8_t>(v));
}

DGPP_TEST(fp8_gemv_hardware_conversion_matches_bridge_on_every_code) {
  float* hw = nullptr;
  float* ref = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&hw, 512 * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&ref, 256 * 4));
  e4m3_convert_all_codes_kernel<<<1, 256>>>(hw, ref);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  // THEN both halves agree with the bridge bitwise on every finite code and
  // are NaN exactly where the bridge is (0x7F / 0xFF).
  for (int v = 0; v < 256; ++v) {
    for (int half = 0; half < 2; ++half) {
      const float got = hw[half * 256 + v];
      if (std::isnan(ref[v])) {
        require(std::isnan(got), "NaN codes convert to NaN");
      } else {
        require(std::memcmp(&got, &ref[v], 4) == 0,
                "finite e4m3 codes convert bit-exactly");
      }
    }
  }
  DGPP_CUDA_OK(cudaFree(hw));
  DGPP_CUDA_OK(cudaFree(ref));
  std::printf("[ OK ] e4m3 hardware conversion == bridge on all 256 codes\n");
}

// The integer form of float_to_bf16_bits (common/dtypes.hpp): what host code
// runs, and the reference the device's conversion instruction is held to.
__host__ __device__ inline uint16_t integer_round_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return static_cast<uint16_t>(((u >> 16) & 0x8000u) | 0x7FC0u);
  const uint32_t lsb = (u >> 16) & 1;
  u += 0x7fffu + lsb;
  return static_cast<uint16_t>(u >> 16);
}

// Every float through the device's two roundings, float_to_bf16_bits (one
// conversion instruction in device code) and the fp8 core's pair form
// (fp8_gemv::dequant_bf16x2, the pack instruction), against the integer
// form: 2^32 bit patterns, each as the pair's low half with a mixed partner
// in the high half. out[0] counts the values that are not NaN and round to
// other bits, out[1] the NaNs that do not stay NaN, out[2] holds one
// offending pattern.
__global__ void bf16_rounding_all_floats_kernel(unsigned long long* out) {
  const uint32_t base = (static_cast<uint32_t>(blockIdx.x) << 16) | (static_cast<uint32_t>(threadIdx.x) << 8);
  unsigned bad = 0, bad_nan = 0;
  uint32_t witness = 0;
  for (uint32_t i = 0; i < 256; ++i) {
    const uint32_t ua = base | i, ub = ua ^ 0xA5A5A5A5u;
    const float2 w = make_float2(__uint_as_float(ua), __uint_as_float(ub));
    const uint32_t pair = dgpp::fp8_gemv::dequant_bf16x2(w, 1.0f);
    const uint32_t got[2][2] = {{pair & 0xFFFFu, dgpp::float_to_bf16_bits(w.x)}, {pair >> 16, dgpp::float_to_bf16_bits(w.y)}};
    const uint32_t in[2] = {ua, ub};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const bool nan = (in[h] & 0x7FFFFFFFu) > 0x7F800000u;
      const uint32_t want = integer_round_bf16(__uint_as_float(in[h]));
#pragma unroll
      for (int form = 0; form < 2; ++form) {
        if (nan) {
          if (!((got[h][form] & 0x7FFFu) > 0x7F80u)) ++bad_nan;
        } else if (got[h][form] != want) {
          ++bad;
          witness = in[h];
        }
      }
    }
  }
  if (bad) {
    atomicAdd(out, static_cast<unsigned long long>(bad));
    out[2] = witness;
  }
  if (bad_nan) atomicAdd(out + 1, static_cast<unsigned long long>(bad_nan));
}

DGPP_TEST(device_bf16_rounding_matches_the_integer_form_on_every_float) {
  // The integer form here is the host's float_to_bf16_bits: a pattern in
  // every 251 across the whole range.
  for (uint64_t u = 0; u < (uint64_t{1} << 32); u += 251) {
    float f;
    const uint32_t bits = static_cast<uint32_t>(u);
    memcpy(&f, &bits, 4);
    require(dgpp::float_to_bf16_bits(f) == integer_round_bf16(f), "the test's integer form is the host's float_to_bf16_bits");
  }
  unsigned long long* out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&out, 3 * sizeof(unsigned long long)));
  out[0] = out[1] = out[2] = 0;
  // 65536 blocks x 256 threads x 256 values = 2^32.
  bf16_rounding_all_floats_kernel<<<65536, 256>>>(out);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  char msg[200];
  std::snprintf(msg, sizeof msg, "the device rounds %llu non-NaN floats to other bits than the integer form (one: %#010llx)", out[0], out[2]);
  require(out[0] == 0, msg);
  require(out[1] == 0, "a NaN must stay a NaN through the device's bf16 rounding");
  DGPP_CUDA_OK(cudaFree(out));
  std::printf("[ OK ] device bf16 rounding (scalar and pair) == the integer form on every non-NaN float, NaN stays NaN\n");
}

// The multi-problem GEMV: several [n_i, k] matrices against
// the same rows in one launch — every problem's output bitwise its own
// launch_scale_gemm_bf16, at one to four rows (one chunk) and six (two).
DGPP_TEST(scale_gemm_gemv_multi_is_bitwise_the_single_launches) {
  const int k = 1008;
  const int ns[3] = {100, 24, 8};
  for (const int rows : {1, 2, 4, 6}) {
    Problem ps[3];
    for (int i = 0; i < 3; ++i) ps[i] = make_problem(rows, ns[i], k, 0xF0 + i + rows);
    // The problems share the activations: problem 0's rows for every one.
    for (int i = 1; i < 3; ++i) ps[i].act = ps[0].act;
    std::vector<uint16_t> single[3];
    for (int i = 0; i < 3; ++i) single[i] = run_kernel(ps[i]);
    uint16_t* act = nullptr;
    uint8_t* w[3] = {};
    float* sc[3] = {};
    uint16_t* out[3] = {};
    DGPP_CUDA_OK(cudaMallocManaged(&act, ps[0].act.size() * 2));
    std::memcpy(act, ps[0].act.data(), ps[0].act.size() * 2);
    dgpp::Fp8GemvProblem probs[3];
    for (int i = 0; i < 3; ++i) {
      DGPP_CUDA_OK(cudaMallocManaged(&w[i], ps[i].payload.size()));
      DGPP_CUDA_OK(cudaMallocManaged(&sc[i], ps[i].scales.size() * 4));
      DGPP_CUDA_OK(cudaMallocManaged(&out[i], static_cast<size_t>(rows) * ns[i] * 2));
      std::memcpy(w[i], ps[i].payload.data(), ps[i].payload.size());
      std::memcpy(sc[i], ps[i].scales.data(), ps[i].scales.size() * 4);
      probs[i].payload = w[i];
      probs[i].scales = sc[i];
      probs[i].out = out[i];
      probs[i].n = ns[i];
    }
    dgpp::launch_scale_gemv_multi_bf16(probs, 3, act, static_cast<size_t>(k), rows, k, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    for (int i = 0; i < 3; ++i) {
      require(std::memcmp(out[i], single[i].data(), single[i].size() * 2) == 0,
              "multi GEMV output differs from the single launch");
      DGPP_CUDA_OK(cudaFree(w[i]));
      DGPP_CUDA_OK(cudaFree(sc[i]));
      DGPP_CUDA_OK(cudaFree(out[i]));
    }
    DGPP_CUDA_OK(cudaFree(act));
    std::printf("[ OK ] the multi-problem fp8 GEMV is bitwise the single launches at %d rows\n", rows);
  }
}

// The multi-problem fp8 GEMV at every decode row count 1..8 (the GDN in-
// projections and the attention q|k|v at a verify block, scheduled 2 / 4 /
// 6 / 8 rows): a row's bits are the 8-row launch's whatever the row count
// — the 4-row chunks and the 1..3-row tail chunk share one chain. Real
// widths: k = 5120, an fp8 problem of 10240 rows, one of 6144, two bf16
// problems of 48.
DGPP_TEST(scale_gemm_gemv_multi_rows_are_one_chain_through_8) {
  const int k = 5120;
  const int ns[4] = {10240, 6144, 48, 48};
  Problem ps[4];
  for (int i = 0; i < 4; ++i) ps[i] = make_problem(8, ns[i], k, 0x8E0 + i);
  for (int i = 1; i < 4; ++i) ps[i].act = ps[0].act;
  std::vector<uint16_t> bf16w[2];
  Rng rng(0x51);
  for (int i = 0; i < 2; ++i) {
    bf16w[i].resize(static_cast<size_t>(48) * k);
    for (auto& v : bf16w[i]) v = dgpp::float_to_bf16_bits(static_cast<float>(rng.unit()));
  }
  uint16_t* act = nullptr;
  uint8_t* w[2] = {};
  float* sc[2] = {};
  uint16_t* wb[2] = {};
  uint16_t* out[4] = {};
  DGPP_CUDA_OK(cudaMallocManaged(&act, ps[0].act.size() * 2));
  std::memcpy(act, ps[0].act.data(), ps[0].act.size() * 2);
  dgpp::Fp8GemvProblem probs[4];
  for (int i = 0; i < 2; ++i) {
    DGPP_CUDA_OK(cudaMallocManaged(&w[i], ps[i].payload.size()));
    DGPP_CUDA_OK(cudaMallocManaged(&sc[i], ps[i].scales.size() * 4));
    std::memcpy(w[i], ps[i].payload.data(), ps[i].payload.size());
    std::memcpy(sc[i], ps[i].scales.data(), ps[i].scales.size() * 4);
    probs[i].payload = w[i];
    probs[i].scales = sc[i];
    probs[i].n = ns[i];
    DGPP_CUDA_OK(cudaMallocManaged(&wb[i], bf16w[i].size() * 2));
    std::memcpy(wb[i], bf16w[i].data(), bf16w[i].size() * 2);
    probs[2 + i].bf16_weight = wb[i];
    probs[2 + i].n = 48;
  }
  for (int i = 0; i < 4; ++i) {
    DGPP_CUDA_OK(cudaMallocManaged(&out[i], static_cast<size_t>(8) * ns[i] * 2));
    probs[i].out = out[i];
  }
  std::vector<uint16_t> full[4];
  dgpp::launch_scale_gemv_multi_bf16(probs, 4, act, static_cast<size_t>(k), 8, k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  for (int i = 0; i < 4; ++i) full[i].assign(out[i], out[i] + static_cast<size_t>(8) * ns[i]);
  for (const int rows : {1, 2, 3, 4, 5, 6, 7}) {
    for (int i = 0; i < 4; ++i) std::memset(out[i], 0, static_cast<size_t>(8) * ns[i] * 2);
    dgpp::launch_scale_gemv_multi_bf16(probs, 4, act, static_cast<size_t>(k), rows, k, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    for (int i = 0; i < 4; ++i)
      require(std::memcmp(out[i], full[i].data(), static_cast<size_t>(rows) * ns[i] * 2) == 0,
              ("multi GEMV: problem " + std::to_string(i) + "'s rows differ between " + std::to_string(rows) +
               " and 8 rows").c_str());
  }
  // The bf16 problems through the bf16 row-group launch (the GDN a / b
  // above four rows): bitwise the fp8 multi launch's bf16 branch at every
  // row count, so a / b read the same at 4 rows (inside the fp8 launch)
  // and at 5..8 (their own).
  for (const int rows : {1, 2, 3, 4, 5, 6, 7, 8}) {
    for (int i = 2; i < 4; ++i) std::memset(out[i], 0, static_cast<size_t>(8) * ns[i] * 2);
    dgpp::Bf16GemvProblem q[2];
    for (int i = 0; i < 2; ++i) {
      q[i].act = act; q[i].act_row_stride = static_cast<size_t>(k); q[i].weight = wb[i]; q[i].out = out[2 + i]; q[i].n = 48;
    }
    dgpp::launch_bf16_gemv_multi_rows(q, 2, /*out_f32=*/false, rows, k, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    for (int i = 2; i < 4; ++i)
      require(std::memcmp(out[i], full[i].data(), static_cast<size_t>(rows) * ns[i] * 2) == 0,
              ("bf16 row-group GEMV: problem " + std::to_string(i) + " differs from the fp8 multi launch's bf16 branch at " +
               std::to_string(rows) + " rows").c_str());
  }
  for (int i = 0; i < 2; ++i) { cudaFree(w[i]); cudaFree(sc[i]); cudaFree(wb[i]); }
  for (int i = 0; i < 4; ++i) cudaFree(out[i]);
  cudaFree(act);
  std::printf("[ OK ] the multi-problem GEMV is one chain at 1..8 rows (k = 5120, fp8 10240 / 6144, bf16 48 / 48)\n");
}

// The GDN in-projection dispatch as the model runs it, at the tiny fixture's
// shapes (k = 256; fp8 qkv 2048 rows, z 1024; bf16 a / b 8 rows): rows <= 4
// take a / b INSIDE the fp8 multi launch, rows 5..8 the row-group bf16
// launch beside a two-problem fp8 launch — the two forms must agree row for
// row (qwen35_forward_test's rows-invariance found rows 2..4 differing from
// 5..8 in the first GDN layer, 2026-10-05).
DGPP_TEST(scale_gemm_gdn_in_projection_dispatch_is_one_chain_1_to_8) {
  for (const int k : {256, 1024, 5120}) {
    const int nf[2] = {2048, 1024};
    Problem ps[2];
    for (int i = 0; i < 2; ++i) ps[i] = make_problem(8, nf[i], k, 0x6D0 + i + k);
    ps[1].act = ps[0].act;
    Rng rng(0x77 + k);
    std::vector<uint16_t> bw[2];
    for (int i = 0; i < 2; ++i) {
      bw[i].resize(static_cast<size_t>(8) * k);
      for (auto& v : bw[i]) v = dgpp::float_to_bf16_bits(static_cast<float>(rng.unit()));
    }
    uint16_t* act = nullptr; uint8_t* w[2] = {}; float* sc[2] = {}; uint16_t* wb[2] = {}; uint16_t* out[4] = {};
    DGPP_CUDA_OK(cudaMallocManaged(&act, ps[0].act.size() * 2));
    std::memcpy(act, ps[0].act.data(), ps[0].act.size() * 2);
    dgpp::Fp8GemvProblem probs[4];
    for (int i = 0; i < 2; ++i) {
      DGPP_CUDA_OK(cudaMallocManaged(&w[i], ps[i].payload.size()));
      DGPP_CUDA_OK(cudaMallocManaged(&sc[i], ps[i].scales.size() * 4));
      std::memcpy(w[i], ps[i].payload.data(), ps[i].payload.size());
      std::memcpy(sc[i], ps[i].scales.data(), ps[i].scales.size() * 4);
      probs[i].payload = w[i]; probs[i].scales = sc[i]; probs[i].n = nf[i];
      DGPP_CUDA_OK(cudaMallocManaged(&wb[i], bw[i].size() * 2));
      std::memcpy(wb[i], bw[i].data(), bw[i].size() * 2);
      probs[2 + i].bf16_weight = wb[i]; probs[2 + i].n = 8;
    }
    const int ns[4] = {nf[0], nf[1], 8, 8};
    for (int i = 0; i < 4; ++i) { DGPP_CUDA_OK(cudaMallocManaged(&out[i], static_cast<size_t>(8) * ns[i] * 2)); probs[i].out = out[i]; }
    // The model's dispatch at `rows`.
    const auto run = [&](int rows) {
      for (int i = 0; i < 4; ++i) std::memset(out[i], 0, static_cast<size_t>(8) * ns[i] * 2);
      if (rows <= 4) {
        dgpp::launch_scale_gemv_multi_bf16(probs, 4, act, static_cast<size_t>(k), rows, k, nullptr);
      } else {
        dgpp::launch_scale_gemv_multi_bf16(probs, 2, act, static_cast<size_t>(k), rows, k, nullptr);
        dgpp::Bf16GemvProblem q[2];
        for (int i = 0; i < 2; ++i) { q[i].act = act; q[i].act_row_stride = static_cast<size_t>(k); q[i].weight = wb[i]; q[i].out = out[2 + i]; q[i].n = 8; }
        dgpp::launch_bf16_gemv_multi_rows(q, 2, /*out_f32=*/false, rows, k, nullptr);
      }
      DGPP_CUDA_OK(cudaDeviceSynchronize());
    };
    run(8);
    std::vector<uint16_t> full[4];
    for (int i = 0; i < 4; ++i) full[i].assign(out[i], out[i] + static_cast<size_t>(8) * ns[i]);
    for (const int rows : {1, 2, 3, 4, 5, 6, 7}) {
      run(rows);
      for (int i = 0; i < 4; ++i) {
        int bad = -1;
        for (int r = 0; r < rows && bad < 0; ++r)
          if (std::memcmp(out[i] + static_cast<size_t>(r) * ns[i], full[i].data() + static_cast<size_t>(r) * ns[i],
                          static_cast<size_t>(ns[i]) * 2) != 0)
            bad = r;
        require(bad < 0, ("GDN in-projection dispatch (k=" + std::to_string(k) + "): problem " + std::to_string(i) +
                          " row " + std::to_string(bad) + " differs between " + std::to_string(rows) + " and 8 rows").c_str());
      }
    }
    for (int i = 0; i < 2; ++i) { cudaFree(w[i]); cudaFree(sc[i]); cudaFree(wb[i]); }
    for (int i = 0; i < 4; ++i) cudaFree(out[i]);
    cudaFree(act);
    std::printf("[ OK ] the GDN in-projection dispatch is one chain at 1..8 rows (k = %d)\n", k);
  }
}

DGPP_TEST(scale_gemm_gemv_multi_takes_bf16_problems_bitwise_the_bf16_launch) {
  const int k = 1008;
  const int ns[4] = {100, 24, 40, 12};  // two fp8 problems, two bf16 ones
  for (const int rows : {1, 2, 4}) {
    Problem ps[2];
    for (int i = 0; i < 2; ++i) ps[i] = make_problem(rows, ns[i], k, 0xA0 + i + rows);
    ps[1].act = ps[0].act;
    std::vector<uint16_t> single[2];
    for (int i = 0; i < 2; ++i) single[i] = run_kernel(ps[i]);
    uint16_t* act = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&act, ps[0].act.size() * 2));
    std::memcpy(act, ps[0].act.data(), ps[0].act.size() * 2);
    uint8_t* w[2] = {};
    float* sc[2] = {};
    uint16_t* w16[2] = {};
    uint16_t* out[4] = {};
    uint16_t* ref16[2] = {};
    dgpp::Fp8GemvProblem probs[4];
    for (int i = 0; i < 2; ++i) {
      DGPP_CUDA_OK(cudaMallocManaged(&w[i], ps[i].payload.size()));
      DGPP_CUDA_OK(cudaMallocManaged(&sc[i], ps[i].scales.size() * 4));
      DGPP_CUDA_OK(cudaMallocManaged(&out[i], static_cast<size_t>(rows) * ns[i] * 2));
      std::memcpy(w[i], ps[i].payload.data(), ps[i].payload.size());
      std::memcpy(sc[i], ps[i].scales.data(), ps[i].scales.size() * 4);
      probs[i].payload = w[i];
      probs[i].scales = sc[i];
      probs[i].out = out[i];
      probs[i].n = ns[i];
    }
    std::mt19937 rng(0xB16 + rows);
    std::normal_distribution<float> nd(0.f, 0.05f);
    for (int i = 0; i < 2; ++i) {
      const size_t elems = static_cast<size_t>(ns[2 + i]) * k;
      DGPP_CUDA_OK(cudaMallocManaged(&w16[i], elems * 2));
      for (size_t j = 0; j < elems; ++j) w16[i][j] = dgpp::float_to_bf16_bits(nd(rng));
      DGPP_CUDA_OK(cudaMallocManaged(&out[2 + i], static_cast<size_t>(rows) * ns[2 + i] * 2));
      DGPP_CUDA_OK(cudaMallocManaged(&ref16[i], static_cast<size_t>(rows) * ns[2 + i] * 2));
      probs[2 + i].bf16_weight = w16[i];
      probs[2 + i].out = out[2 + i];
      probs[2 + i].n = ns[2 + i];
      dgpp::launch_bf16_gemv(act, static_cast<size_t>(k), w16[i], ref16[i], /*out_f32=*/false, rows, ns[2 + i], k, nullptr);
    }
    dgpp::launch_scale_gemv_multi_bf16(probs, 4, act, static_cast<size_t>(k), rows, k, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    for (int i = 0; i < 2; ++i) {
      require(std::memcmp(out[i], single[i].data(), single[i].size() * 2) == 0,
              "mixed multi GEMV: an fp8 problem differs from its single launch");
      require(std::memcmp(out[2 + i], ref16[i], static_cast<size_t>(rows) * ns[2 + i] * 2) == 0,
              "mixed multi GEMV: a bf16 problem differs from launch_bf16_gemv");
      DGPP_CUDA_OK(cudaFree(w[i]));
      DGPP_CUDA_OK(cudaFree(sc[i]));
      DGPP_CUDA_OK(cudaFree(w16[i]));
      DGPP_CUDA_OK(cudaFree(out[i]));
      DGPP_CUDA_OK(cudaFree(out[2 + i]));
      DGPP_CUDA_OK(cudaFree(ref16[i]));
    }
    DGPP_CUDA_OK(cudaFree(act));
    std::printf("[ OK ] the multi-problem GEMV takes bf16 problems bitwise the bf16 launch at %d rows\n", rows);
  }
}

DGPP_TEST(scale_gemm_gemv_path_ragged_n_and_k_match_both_oracles) {
  // GIVEN m=1 with k a multiple of 16 but not of 128 (the last chunk sits
  // in a partial scale block) and n not a multiple of the 8-row block:
  const Problem p = make_problem(1, 1003, 1008, 0xE1);
  const std::vector<uint16_t> got = run_kernel(p);
  check_both_oracles(p, got, "gemv M1xN1003xK1008");
  // AND a k shorter than one warp span (512 bytes): every lane but the
  // first few has no chunk at all.
  const Problem q = make_problem(1, 200, 48, 0xE2);
  check_both_oracles(q, run_kernel(q), "gemv M1xN200xK48");
  // Determinism.
  const std::vector<uint16_t> again = run_kernel(p);
  require(std::memcmp(got.data(), again.data(), got.size() * 2) == 0,
          "gemv second run bitwise identical");
}

DGPP_TEST(scale_gemm_gemv_rows_are_independent_of_row_count) {
  // GIVEN the same activation rows run at m=1 each and at m=3 together,
  // THEN every row's bits agree: each row's chain is the same sequence of
  // FMAs whatever the row count (the property glm_moe_test's slot-vs-host
  // bitwise pin relies on).
  const Problem p3 = make_problem(3, 520, 4096, 0xE3);
  const std::vector<uint16_t> got3 = run_kernel(p3);
  check_both_oracles(p3, got3, "gemv M3xN520xK4096");
  for (int r = 0; r < 3; ++r) {
    Problem p1;
    p1.m = 1;
    p1.n = p3.n;
    p1.k = p3.k;
    p1.act.assign(p3.act.begin() + static_cast<long>(r) * p3.k,
                  p3.act.begin() + static_cast<long>(r + 1) * p3.k);
    p1.payload = p3.payload;
    p1.scales = p3.scales;
    const std::vector<uint16_t> got1 = run_kernel(p1);
    require(std::memcmp(got1.data(), got3.data() + static_cast<size_t>(r) * p3.n,
                        static_cast<size_t>(p3.n) * 2) == 0,
            "row bits independent of m");
  }
  // AND m=4 (the largest GEMV row count) still matches the oracles.
  const Problem p4 = make_problem(4, 264, 2048, 0xE4);
  check_both_oracles(p4, run_kernel(p4), "gemv M4xN264xK2048");
  // AND m=8 at real hidden width must take row-independent GEMV chunks,
  // not the numerically different tile kernel. Check both epilogues against
  // eight scalar invocations so chunk boundaries and output offsets are
  // covered directly.
  const Problem p8 = make_problem(8, 136, 4096, 0xE8);
  const std::vector<uint16_t> got8 = run_kernel(p8);
  const std::vector<float> got8f = run_kernel_f32(p8);
  check_both_oracles(p8, got8, "gemv chunked M8xN136xK4096");
  for (int r = 0; r < p8.m; ++r) {
    Problem p1;
    p1.m = 1;
    p1.n = p8.n;
    p1.k = p8.k;
    p1.act.assign(p8.act.begin() + static_cast<long>(r) * p8.k,
                  p8.act.begin() + static_cast<long>(r + 1) * p8.k);
    p1.payload = p8.payload;
    p1.scales = p8.scales;
    const std::vector<uint16_t> got1 = run_kernel(p1);
    const std::vector<float> got1f = run_kernel_f32(p1);
    require(std::memcmp(got1.data(),
                        got8.data() + static_cast<size_t>(r) * p8.n,
                        static_cast<size_t>(p8.n) * sizeof(uint16_t)) == 0,
            "chunked fp8/bf16 row bits independent of m");
    require(std::memcmp(got1f.data(),
                        got8f.data() + static_cast<size_t>(r) * p8.n,
                        static_cast<size_t>(p8.n) * sizeof(float)) == 0,
            "chunked fp8/f32 row bits independent of m");
  }
  std::printf("[ OK ] gemv rows independent of m through m=8 chunks\n");
}

// The streaming MMA form at every decode row count (mma_from_rows = 1: the
// head and the fp8 projections since 2026-10-05): a row's bits are the
// same whatever the row count rides in the launch, 1..64 — a request's
// logits are the same alone, in a verify block of any depth and in a
// batch. (Asked with mma_from_rows = 5 the rows below five take the GEMV
// chain, a different chain by design: that was this test's first form.)
DGPP_TEST(scale_gemm_streaming_mma_row_bits_vs_row_count) {
  const Problem p64 = make_problem(64, 1024, 5120, 0x5E5E);
  const std::vector<float> got64 = run_kernel_f32(p64, /*mma_from_rows=*/1);
  const std::vector<float> again = run_kernel_f32(p64, /*mma_from_rows=*/1);
  require(std::memcmp(got64.data(), again.data(), got64.size() * sizeof(float)) == 0,
          "streaming mma: run-to-run bitwise at m=64");
  for (const float v : got64) require(std::isfinite(v), "streaming mma: finite logits");
  std::vector<int> failed_m;
  for (const int m : {1, 2, 3, 4, 5, 8, 15, 16, 17, 31, 32, 33, 48, 63}) {
    Problem pm;
    pm.m = m;
    pm.n = p64.n;
    pm.k = p64.k;
    pm.act.assign(p64.act.begin(), p64.act.begin() + static_cast<long>(m) * p64.k);
    pm.payload = p64.payload;
    pm.scales = p64.scales;
    const std::vector<float> gotm = run_kernel_f32(pm, /*mma_from_rows=*/1);
    int differing = -1;
    size_t ndiff = 0;
    double maxrel = 0.0;
    for (int r = 0; r < m; ++r)
      for (int c = 0; c < pm.n; ++c) {
        const float a = gotm[static_cast<size_t>(r) * pm.n + c], b = got64[static_cast<size_t>(r) * p64.n + c];
        if (a != b) {
          if (differing < 0) differing = r;
          ++ndiff;
          maxrel = std::max(maxrel, static_cast<double>(std::abs(a - b)) / std::max(std::abs(b), 1e-6f));
        }
      }
    std::printf("[ .. ] streaming mma m=%d vs m=64: %s (%zu of %zu elements differ, first row %d, max rel %.3g)\n", m,
                differing < 0 ? "bitwise" : "DIFFERENT", ndiff, gotm.size(), differing, maxrel);
    if (differing >= 0) failed_m.push_back(m);
    // Below five rows the form used to hand off to the GEMV path: the
    // streaming form must hold the head's FP64 error budget there too
    // (scale_gemm_f32_fp8_head_numerics' 3e-5 of the products' magnitude).
    if (m <= 4) {
      const int scale_cols = (pm.k + 127) / 128;
      for (int row = 0; row < m; ++row)
        for (int col = 0; col < pm.n; ++col) {
          double reference = 0, magnitude = 0;
          for (int kk = 0; kk < pm.k; ++kk) {
            const float decoded = dgpp::fp8_e4m3_bits_to_float(pm.payload[static_cast<size_t>(col) * pm.k + kk]);
            const float scale = pm.scales[(col / 128) * scale_cols + kk / 128];
            const double weight = bf16_to_float(dgpp::float_to_bf16_bits(decoded * scale));
            const double product = bf16_to_float(pm.act[static_cast<size_t>(row) * pm.k + kk]) * weight;
            reference += product;
            magnitude += std::abs(product);
          }
          const float got = gotm[static_cast<size_t>(row) * pm.n + col];
          require(std::abs(got - reference) <= 3e-5 * magnitude,
                  "streaming mma at m <= 4: outside the head's FP64 error budget");
        }
    }
  }
  require(failed_m.empty(), "streaming mma: a row's bits must not depend on the row count: the decode head relies on it");
}

DGPP_TEST(scale_gemm_gemv_path_propagates_nan_exactly) {
  // GIVEN m=1 with two poisoned weights: exactly their output columns NaN.
  Problem p = make_problem(1, 300, 1024, 0xE5);
  p.payload[static_cast<size_t>(9) * p.k + 700] = 0x7F;
  p.payload[static_cast<size_t>(250) * p.k + 3] = 0xFF;
  const std::vector<uint16_t> got = run_kernel(p);
  for (int nn = 0; nn < p.n; ++nn) {
    const bool got_nan = std::isnan(bf16_to_float(got[static_cast<size_t>(nn)]));
    if (nn == 9 || nn == 250)
      require(got_nan, "poisoned column is NaN (gemv)");
    else
      require(!got_nan, "unpoisoned column stays finite (gemv)");
  }
  std::printf("[ OK ] gemv nan propagation: columns 9 and 250 NaN\n");
}

// Real-checkpoint slice parity lives in scale_gemm_checkpoint.cpp (host-only
// TU: minijson does not mix with nvcc).
int run_scale_gemm_checkpoint_parity(const char* checkpoint_dir);

int main(int argc, char** argv) {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  const int rc = dgpp::test::run_all();
  if (rc != 0) return rc;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--checkpoint-dir") == 0 && i + 1 < argc) {
      const int prc = run_scale_gemm_checkpoint_parity(argv[i + 1]);
      if (prc != 0) return prc;
    }
  }
  return 0;
}
