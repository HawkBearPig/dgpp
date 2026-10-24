// The W4A4 NVFP4 grouped MoE GEMM (src/kernels/moe_w4a4.cu):
//  1. layouts: against an fp64 host dot of the SAME quantized activations and
//     weights (dequantized exactly) -- only fp32 summation order may differ;
//  2. quantization: against the W4A16 kernel on the bf16 activations -- the
//     error W4A4 adds, reported (the model-level gate is BFCL / tool-eval);
//  3. speed at prefill shapes against launch_moe_grouped_mma_fp4_*.
//   moe_w4a4_test [--tokens T] [--experts E] [--topk K]
#include <cmath>
#include <limits>
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
#include "kernels/glm_moe_launch.hpp"
#include "kernels/moe_w4a4.hpp"

using namespace dgpp;

namespace {

// The model's SwiGLU (qwen35_swiglu_kernel's ops), for the fused quantizer's gate.
__global__ void qwen_swiglu_kernel_t(const uint16_t* gate, const uint16_t* up, uint16_t* out, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float g = __uint_as_float(static_cast<uint32_t>(gate[i]) << 16);
  const float s = g / (1.0f + expf(-g));
  const float u = __uint_as_float(static_cast<uint32_t>(up[i]) << 16);
  out[i] = __bfloat16_as_ushort(__float2bfloat16_rn(s * u));
}
void qwen_swiglu_device(const uint16_t* gate, const uint16_t* up, uint16_t* out, int64_t n) {
  qwen_swiglu_kernel_t<<<static_cast<unsigned>((n + 255) / 256), 256>>>(gate, up, out, n);
  DGPP_CUDA_OK(cudaGetLastError());
}
float e2m1(uint8_t nib) {
  static const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  return (nib & 8) ? -mag[nib & 7] : mag[nib & 7];
}
float e4m3(uint8_t b) {
  __nv_fp8_e4m3 v;
  v.__x = b;
  return static_cast<float>(v);
}
uint16_t bf16(float f) {
  const __nv_bfloat16 b = __float2bfloat16_rn(f);
  uint16_t u;
  std::memcpy(&u, &b, 2);
  return u;
}
template <typename T>
T* dev(const std::vector<T>& h) {
  T* d = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&d, h.size() * sizeof(T)));
  DGPP_CUDA_OK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
template <typename T>
std::vector<T> host(const T* d, size_t n) {
  std::vector<T> h(n);
  DGPP_CUDA_OK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

struct Matrix {  // one expert's NVFP4 matrix [n, k]
  std::vector<uint8_t> codes, scales;
  float global = 1.f;
};

struct Case {
  int tokens, experts, topk, n, k;
  std::vector<uint16_t> act;         // [tokens, k] bf16
  std::vector<Matrix> w;             // per expert
  std::vector<MoeSegment> segs;
  std::vector<int32_t> act_rows;     // [tokens * topk] -> token
  int max_rows = 0;
};

Case make_case(int tokens, int experts, int topk, int n, int k, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_int_distribution<int> byte(0, 255);
  Case c{tokens, experts, topk, n, k, {}, {}, {}, {}, 0};
  c.act.resize(static_cast<size_t>(tokens) * k);
  for (size_t i = 0; i < c.act.size(); ++i) {
    float x = nd(rng);
    if (byte(rng) == 0) x *= 20.f;  // outliers, as real hidden states have
    c.act[i] = bf16(x);
  }
  std::uniform_real_distribution<float> sd(0.05f, 1.5f);
  for (int e = 0; e < experts; ++e) {
    Matrix m;
    m.codes.resize(static_cast<size_t>(n) * k / 2);
    for (auto& b : m.codes) b = static_cast<uint8_t>(byte(rng));
    m.scales.resize(static_cast<size_t>(n) * k / 16);
    for (auto& s : m.scales) s = __nv_fp8_e4m3(sd(rng)).__x;
    m.global = 300.f + 200.f * sd(rng);
    c.w.push_back(std::move(m));
  }
  // Routing: each token picks topk distinct experts; segments by expert.
  std::vector<std::vector<int>> by_expert(experts);
  std::uniform_int_distribution<int> pick(0, experts - 1);
  for (int t = 0; t < tokens; ++t) {
    std::vector<int> chosen;
    while (static_cast<int>(chosen.size()) < topk) {
      const int e = pick(rng);
      bool dup = false;
      for (int x : chosen) dup |= x == e;
      if (!dup) chosen.push_back(e);
    }
    for (int e : chosen) by_expert[e].push_back(t);
  }
  int row = 0;
  for (int e = 0; e < experts; ++e) {
    if (by_expert[e].empty()) continue;
    c.segs.push_back(MoeSegment{row, static_cast<int32_t>(by_expert[e].size()), e});
    for (int t : by_expert[e]) c.act_rows.push_back(t);
    row += static_cast<int>(by_expert[e].size());
    c.max_rows = std::max(c.max_rows, static_cast<int>(by_expert[e].size()));
  }
  return c;
}

struct Device {
  std::vector<uint8_t*> bufs;
  std::vector<float*> globals;
  MoeExpertView* views = nullptr;
};

Device upload(const Case& c) {
  Device d;
  std::vector<MoeExpertView> views(static_cast<size_t>(c.experts) * 3);
  for (int e = 0; e < c.experts; ++e) {
    uint8_t* codes = dev(c.w[e].codes);
    uint8_t* scales = dev(c.w[e].scales);
    float* g = dev(std::vector<float>{c.w[e].global});
    d.bufs.push_back(codes);
    d.bufs.push_back(scales);
    d.globals.push_back(g);
    MoeExpertView v{codes, nullptr, scales, g, 7, 7};
    v.fp4_group = kFp4Group;
    views[static_cast<size_t>(e) * 3] = v;
  }
  d.views = dev(views);
  return d;
}

int run(int tokens, int experts, int topk, int n, int k, bool check_exact, const char* label, float static_gs = 0.f) {
  const Case c = make_case(tokens, experts, topk, n, k, 1234u + n + k);
  const Device d = upload(c);
  const int rows = static_cast<int>(c.act_rows.size());
  uint16_t* act = dev(c.act);
  MoeSegment* segs = dev(c.segs);
  int32_t* act_rows = dev(c.act_rows);
  const size_t ss = nvfp4_act_scale_stride(k);
  uint8_t *codes = nullptr, *scales = nullptr;
  float *gs = nullptr, *out4 = nullptr, *out16 = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&codes, static_cast<size_t>(tokens) * k / 2));
  DGPP_CUDA_OK(cudaMalloc(&scales, static_cast<size_t>(tokens) * ss));
  DGPP_CUDA_OK(cudaMalloc(&gs, static_cast<size_t>(tokens) * 4));
  DGPP_CUDA_OK(cudaMalloc(&out4, static_cast<size_t>(rows) * n * 4));
  DGPP_CUDA_OK(cudaMalloc(&out16, static_cast<size_t>(rows) * n * 4));
  const int ns = static_cast<int>(c.segs.size());

  auto w4a4 = [&] {
    launch_quantize_rows_nvfp4(act, k, tokens, k, codes, scales, gs, nullptr, static_gs);
    launch_moe_grouped_w4a4_f32(codes, scales, gs, act_rows, segs, ns, c.max_rows, d.views, 0, out4, n, n, k,
                                nullptr);
  };
  auto w4a16 = [&] {
    launch_moe_grouped_mma_fp4_f32(act, k, segs, ns, c.max_rows, 0, d.views, 0, out16, n, n, k, nullptr, act_rows);
  };
  w4a4();
  w4a16();
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<float> o4 = host(out4, static_cast<size_t>(rows) * n);
  const std::vector<float> o16 = host(out16, static_cast<size_t>(rows) * n);
  int fails = 0;

  // 1. layouts: fp64 dot of the quantized operands, dequantized exactly.
  if (check_exact) {
    const auto qc = host(codes, static_cast<size_t>(tokens) * k / 2);
    const auto qs = host(scales, static_cast<size_t>(tokens) * ss);
    const auto qg = host(gs, static_cast<size_t>(tokens));
    double worst = 0;
    for (const MoeSegment& s : c.segs) {
      const Matrix& m = c.w[s.expert];
      for (int r = 0; r < s.rows; r += std::max(1, s.rows / 3)) {
        const int tok = c.act_rows[s.row0 + r];
        for (int col = 0; col < n; col += 37) {
          double dot = 0, mag = 0;
          for (int kk = 0; kk < k; ++kk) {
            const uint8_t ab = qc[static_cast<size_t>(tok) * k / 2 + kk / 2];
            const double a = e2m1(kk % 2 ? ab >> 4 : ab & 15) * e4m3(qs[tok * ss + kk / 16]) * qg[tok];
            const uint8_t wb = m.codes[static_cast<size_t>(col) * k / 2 + kk / 2];
            const double w = e2m1(kk % 2 ? wb >> 4 : wb & 15) * e4m3(m.scales[static_cast<size_t>(col) * k / 16 + kk / 16]) /
                             m.global;
            dot += a * w;
            mag += std::fabs(a * w);
          }
          const double got = o4[static_cast<size_t>(s.row0 + r) * n + col];
          const double err = std::fabs(got - dot) / (mag + 1e-30);
          worst = std::max(worst, err);
        }
      }
    }
    std::printf("[ .. ] %s layouts: worst |got - fp64| / sum|terms| = %.3g\n", label, worst);
    if (!(worst < 1e-5)) {
      std::printf("[FAIL] %s: the W4A4 kernel does not compute the quantized dot\n", label);
      ++fails;
    }
  }

  // 2. the error W4A4 adds against W4A16 (bf16 activations).
  double num = 0, den = 0;
  for (size_t i = 0; i < o4.size(); ++i) {
    num += (o4[i] - o16[i]) * (o4[i] - o16[i]);
    den += o16[i] * o16[i];
  }
  std::printf("[ .. ] %s W4A4 vs W4A16: relative l2 %.4f\n", label, std::sqrt(num / den));

  // 3. speed.
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
  auto quant_only = [&] { launch_quantize_rows_nvfp4(act, k, tokens, k, codes, scales, gs, nullptr); };
  const float t4 = time(w4a4), tq = time(quant_only), t16 = time(w4a16);
  const double flop = 2.0 * rows * n * k;
  std::printf("[ .. ] %s %d rows x n %d x k %d: W4A16 %.3f ms (%.1f TF)  W4A4 %.3f ms (%.1f TF, quant %.3f)  %.2fx\n",
              label, rows, n, k, t16, flop / t16 / 1e9, t4, flop / t4 / 1e9, tq, t16 / t4);
  // The production output form (bf16 rows, the down projection's down_bf16_).
  auto gemm_bf16 = [&] {
    launch_moe_grouped_w4a4_bf16(codes, scales, gs, act_rows, segs, ns, c.max_rows, d.views, 0,
                                 reinterpret_cast<uint16_t*>(out4), n, n, k, nullptr);
  };
  auto gemm_f32 = [&] {
    launch_moe_grouped_w4a4_f32(codes, scales, gs, act_rows, segs, ns, c.max_rows, d.views, 0, out4, n, n, k, nullptr);
  };
  const float tb = time(gemm_bf16), tf = time(gemm_f32);
  std::printf("[ .. ] %s GEMM only: f32 out %.3f ms (%.1f TF), bf16 out %.3f ms (%.1f TF)\n", label, tf,
              flop / tf / 1e9, tb, flop / tb / 1e9);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  for (auto* p : d.bufs) cudaFree(p);
  for (auto* p : d.globals) cudaFree(p);
  cudaFree(d.views);
  cudaFree(act);
  cudaFree(segs);
  cudaFree(act_rows);
  cudaFree(codes);
  cudaFree(scales);
  cudaFree(gs);
  cudaFree(out4);
  cudaFree(out16);
  return fails;
}

}  // namespace

int main(int argc, char** argv) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("[SKIP] moe_w4a4_test: no CUDA device\n");
    return 2;
  }
  int tokens = 8192, experts = 256, topk = 10;
  for (int i = 1; i + 1 < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--tokens") tokens = std::atoi(argv[++i]);
    else if (a == "--experts") experts = std::atoi(argv[++i]);
    else if (a == "--topk") topk = std::atoi(argv[++i]);
  }
  int fails = 0;
  // The recipe's exact case, bitwise: rows whose values are NVFP4-representable
  // under a block scale e4m3 holds (amax 1.5 -> 1.5 / 6 = 0.25) and a static
  // global of 1 come back from the quantizer as the same bf16 values, so the
  // scale rule (amax / 6, e4m3) and the e2m1 rounding are the checkpoint's.
  for (const int k : {512, 17408}) {
    const int rows = 33;
    const float grid[] = {0.f, 0.125f, -0.125f, 0.25f, -0.25f, 0.375f, -0.375f, 0.5f, -0.5f, 0.75f, -0.75f, 1.0f, -1.0f, 1.5f};
    std::mt19937 rng(17);
    std::vector<uint16_t> x(size_t(rows) * k);
    for (int r = 0; r < rows; ++r)
      for (int g = 0; g < k / 16; ++g) {
        for (int i = 0; i < 16; ++i) x[size_t(r) * k + g * 16 + i] = bf16(grid[rng() % 14]);
        x[size_t(r) * k + g * 16 + (rng() % 16)] = bf16(rng() % 2 ? 1.5f : -1.5f);  // the block's amax
      }
    uint16_t* dx = dev(x);
    const size_t ss = nvfp4_act_scale_stride(k);
    uint8_t *codes = nullptr, *scales = nullptr;
    float* gs = nullptr;
    DGPP_CUDA_OK(cudaMalloc(&codes, size_t(rows) * k / 2));
    DGPP_CUDA_OK(cudaMalloc(&scales, size_t(rows) * ss));
    DGPP_CUDA_OK(cudaMalloc(&gs, size_t(rows) * 4));
    launch_quantize_rows_nvfp4(dx, k, rows, k, codes, scales, gs, nullptr, 1.0f);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const auto qc = host(codes, size_t(rows) * k / 2);
    const auto qs = host(scales, size_t(rows) * ss);
    const auto qg = host(gs, size_t(rows));
    int bad = 0;
    for (int r = 0; r < rows; ++r)
      for (int kk = 0; kk < k; ++kk) {
        const uint8_t ab = qc[size_t(r) * k / 2 + kk / 2];
        const float back = e2m1(kk % 2 ? ab >> 4 : ab & 15) * e4m3(qs[size_t(r) * ss + kk / 16]) * qg[r];
        if (bf16(back) != x[size_t(r) * k + kk]) ++bad;
      }
    std::printf("representable rows through the quantizer (k %d): %d of %d values differ\n", k, bad, rows * k);
    if (bad != 0) ++fails;
    cudaFree(dx); cudaFree(codes); cudaFree(scales); cudaFree(gs);
  }
  // The fused SwiGLU -> quantizer with the Qwen rounding: bitwise the model's
  // swiglu ops (a device kernel here, op for op) followed by the row quantizer.
  {
    const int rows = 61, k = 1024, stride = 1024;
    std::mt19937 rng(31);
    std::normal_distribution<float> nd(0.f, 2.f);
    std::vector<uint16_t> ga(size_t(rows) * stride), up(ga.size());
    for (auto& v : ga) v = bf16(nd(rng));
    for (auto& v : up) v = bf16(nd(rng));
    uint16_t* dg = dev(ga);
    uint16_t* du = dev(up);
    uint16_t* da = nullptr;
    DGPP_CUDA_OK(cudaMalloc(&da, ga.size() * 2));
    qwen_swiglu_device(dg, du, da, static_cast<int64_t>(ga.size()));
    const size_t ss = nvfp4_act_scale_stride(k);
    uint8_t *c1 = nullptr, *c2 = nullptr, *s1 = nullptr, *s2 = nullptr;
    float *g1 = nullptr, *g2 = nullptr;
    DGPP_CUDA_OK(cudaMalloc(&c1, size_t(rows) * k / 2)); DGPP_CUDA_OK(cudaMalloc(&c2, size_t(rows) * k / 2));
    DGPP_CUDA_OK(cudaMalloc(&s1, size_t(rows) * ss)); DGPP_CUDA_OK(cudaMalloc(&s2, size_t(rows) * ss));
    DGPP_CUDA_OK(cudaMalloc(&g1, size_t(rows) * 4)); DGPP_CUDA_OK(cudaMalloc(&g2, size_t(rows) * 4));
    launch_quantize_rows_nvfp4(da, stride, rows, k, c1, s1, g1, nullptr, 0.25f);
    launch_swiglu_quantize_rows_nvfp4(dg, du, stride, rows, k, 0.f, c2, s2, g2, nullptr, 0.25f, nullptr, /*qwen_rounding=*/true);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const bool same = host(c1, size_t(rows) * k / 2) == host(c2, size_t(rows) * k / 2) && host(s1, size_t(rows) * ss) == host(s2, size_t(rows) * ss) &&
                      host(g1, size_t(rows)) == host(g2, size_t(rows));
    std::printf("[%s] fused Qwen swiglu+quantize vs the two-kernel chain: %s\n", same ? " OK " : "FAIL", same ? "bitwise" : "DIFFER");
    if (!same) ++fails;
    cudaFree(dg); cudaFree(du); cudaFree(da); cudaFree(c1); cudaFree(c2); cudaFree(s1); cudaFree(s2); cudaFree(g1); cudaFree(g2);
  }
  // The dense form on TMA boxes: bitwise the grouped kernel over one segment,
  // the output cleared to a sentinel first (a skipped row cannot pass).
  for (const int m : {31, 129, 300, 1152}) {
    const int n = 512, k = 1280;
    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> x(size_t(m) * k);
    for (auto& v : x) v = bf16(nd(rng));
    std::vector<uint8_t> wc(size_t(n) * k / 2), ws(size_t(n) * k / 16);
    for (auto& v : wc) v = static_cast<uint8_t>(rng() & 0xFF);
    for (auto& v : ws) { uint8_t b = static_cast<uint8_t>(rng() & 0x7F); if ((b & 0x7F) == 0x7F) b = 0x7E; v = b; }
    const float one = 1.f;
    uint16_t* dx = dev(x);
    uint8_t* dwc = dev(wc);
    uint8_t* dws = dev(ws);
    float* dg = dev(std::vector<float>{one});
    const size_t ss = nvfp4_act_scale_stride(k);
    uint8_t *codes = nullptr, *scales = nullptr;
    float* gs = nullptr;
    uint16_t *o1 = nullptr, *o2 = nullptr;
    MoeSegment* seg = dev(std::vector<MoeSegment>{MoeSegment{0, m, 0}});
    GlmFp4Matrix fm;
    fm.payload = dwc; fm.scales = dws; fm.global_scale = dg; fm.rows = n; fm.cols = k; fm.scale_group = 16;
    MoeExpertView* views = dev(std::vector<MoeExpertView>{MoeExpertView::of(fm), MoeExpertView::of(fm), MoeExpertView::of(fm)});
    DGPP_CUDA_OK(cudaMalloc(&codes, size_t(m) * k / 2));
    DGPP_CUDA_OK(cudaMalloc(&scales, size_t(m) * ss));
    DGPP_CUDA_OK(cudaMalloc(&gs, size_t(m) * 4));
    DGPP_CUDA_OK(cudaMalloc(&o1, size_t(m) * n * 2));
    DGPP_CUDA_OK(cudaMalloc(&o2, size_t(m) * n * 2));
    launch_quantize_rows_nvfp4(dx, k, m, k, codes, scales, gs, nullptr, 1.f);
    DGPP_CUDA_OK(cudaMemset(o1, 0xFF, size_t(m) * n * 2));
    DGPP_CUDA_OK(cudaMemset(o2, 0x00, size_t(m) * n * 2));
    launch_moe_grouped_w4a4_bf16(codes, scales, gs, nullptr, seg, 1, m, views, 0, o1, n, n, k, nullptr);
    const CUtensorMap am = nvfp4_code_tma_map(codes, m, k, 128), wm = nvfp4_code_tma_map(dwc, n, k, 256);
    launch_dense_w4a4_tma_bf16(am, wm, scales, ss, gs, dws, dg, o2, n, m, n, k, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const auto h1 = host(o1, size_t(m) * n), h2 = host(o2, size_t(m) * n);
    size_t diff = 0;
    for (size_t i = 0; i < h1.size(); ++i) diff += h1[i] != h2[i];
    std::printf("dense tma vs grouped (m %d n %d k %d): %zu of %zu values differ\n", m, n, k, diff, h1.size());
    if (diff != 0) ++fails;
    cudaFree(dx); cudaFree(dwc); cudaFree(dws); cudaFree(dg); cudaFree(codes); cudaFree(scales); cudaFree(gs);
    cudaFree(o1); cudaFree(o2); cudaFree(seg); cudaFree(views);
  }
  // The dense form's pipeline drain (2026-10-17): its last k-step's scale words
  // must have landed before they are read. Forty launches at the Qwen3.8-27B
  // MLP's gate shape, every output bitwise the grouped kernel's; before the
  // drain wait, 4 of 49 launches differed (the NVFP4 27B's temperature-0
  // transcripts moved run to run).
  {
    const int m = 100, n = 17408, k = 5120, iters = 40;
    std::mt19937 rng(29);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> x(size_t(m) * k);
    for (auto& v : x) v = bf16(nd(rng));
    std::vector<uint8_t> wc(size_t(n) * k / 2), ws(size_t(n) * k / 16);
    for (auto& v : wc) v = static_cast<uint8_t>(rng() & 0xFF);
    for (auto& v : ws) { uint8_t b = static_cast<uint8_t>(rng() & 0x7F); if ((b & 0x7F) == 0x7F) b = 0x7E; v = b; }
    const float one = 1.f;
    uint16_t* dx = dev(x);
    uint8_t* dwc = dev(wc);
    uint8_t* dws = dev(ws);
    float* dg = dev(std::vector<float>{one});
    const size_t ss = nvfp4_act_scale_stride(k);
    uint8_t *codes = nullptr, *scales = nullptr;
    float* gs = nullptr;
    uint16_t *o1 = nullptr, *o2 = nullptr;
    MoeSegment* seg = dev(std::vector<MoeSegment>{MoeSegment{0, m, 0}});
    GlmFp4Matrix fm;
    fm.payload = dwc; fm.scales = dws; fm.global_scale = dg; fm.rows = n; fm.cols = k; fm.scale_group = 16;
    MoeExpertView* views = dev(std::vector<MoeExpertView>{MoeExpertView::of(fm), MoeExpertView::of(fm), MoeExpertView::of(fm)});
    DGPP_CUDA_OK(cudaMalloc(&codes, size_t(m) * k / 2));
    DGPP_CUDA_OK(cudaMalloc(&scales, size_t(m) * ss));
    DGPP_CUDA_OK(cudaMalloc(&gs, size_t(m) * 4));
    DGPP_CUDA_OK(cudaMalloc(&o1, size_t(m) * n * 2));
    DGPP_CUDA_OK(cudaMalloc(&o2, size_t(m) * n * 2));
    launch_quantize_rows_nvfp4(dx, k, m, k, codes, scales, gs, nullptr, 1.f);
    launch_moe_grouped_w4a4_bf16(codes, scales, gs, nullptr, seg, 1, m, views, 0, o1, n, n, k, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const auto h1 = host(o1, size_t(m) * n);
    const CUtensorMap am = nvfp4_code_tma_map(codes, m, k, 128), wm = nvfp4_code_tma_map(dwc, n, k, 256);
    int bad = 0;
    for (int it = 0; it < iters; ++it) {
      DGPP_CUDA_OK(cudaMemset(o2, (it & 1) ? 0x00 : 0xFF, size_t(m) * n * 2));
      launch_dense_w4a4_tma_bf16(am, wm, scales, ss, gs, dws, dg, o2, n, m, n, k, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      if (host(o2, size_t(m) * n) != h1) ++bad;
    }
    std::printf("dense tma drain (m %d n %d k %d): %d of %d launches differ from the grouped form\n", m, n, k, bad, iters);
    if (bad != 0) ++fails;
    cudaFree(dx); cudaFree(dwc); cudaFree(dws); cudaFree(dg); cudaFree(codes); cudaFree(scales); cudaFree(gs);
    cudaFree(o1); cudaFree(o2); cudaFree(seg); cudaFree(views);
  }
  // Small shapes with ragged segments and k tails for the exact layout check.
  fails += run(64, 8, 2, 256, 256, true, "small");
  fails += run(200, 16, 4, 320, 320, true, "k320");
  fails += run(100, 8, 3, 130, 2560, true, "n-tail");
  // A static global (the checkpoint's input_scale form) small enough that the
  // 20x outliers clip: the layouts must still be exact on the clipped codes.
  fails += run(200, 16, 4, 320, 320, true, "static-clip", 2.0f / (6.f * 448.f));
  // Prefill shapes (the Qwen experts per rank at TP=2: gate/up n = 320,
  // k = 2560; down n = 2560, k = 320).
  fails += run(tokens, experts, topk, 640, 2560, false, "gate_up");
  fails += run(tokens, experts, topk, 2560, 320, false, "down");
  // The fused activation + quantizer against swiglu -> quantize, bitwise
  // (codes, scales, globals), at the Qwen down projection's k = 320.
  {
    const int rows = 57, k = 320, stride = 320;
    std::mt19937 rng(9);
    std::normal_distribution<float> nd(0.f, 2.f);
    std::vector<uint16_t> ga(size_t(rows) * stride), up(ga.size());
    for (auto& v : ga) v = dgpp::float_to_bf16_bits(nd(rng));
    for (auto& v : up) v = dgpp::float_to_bf16_bits(nd(rng));
    const size_t ss = dgpp::nvfp4_act_scale_stride(k);
    uint16_t *dg, *du, *da;
    uint8_t *c1, *c2, *s1, *s2;
    float *g1, *g2;
    DGPP_CUDA_OK(cudaMalloc(&dg, ga.size() * 2));
    DGPP_CUDA_OK(cudaMalloc(&du, up.size() * 2));
    DGPP_CUDA_OK(cudaMalloc(&da, ga.size() * 2));
    DGPP_CUDA_OK(cudaMemcpy(dg, ga.data(), ga.size() * 2, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(du, up.data(), up.size() * 2, cudaMemcpyHostToDevice));
    for (uint8_t** q : {&c1, &c2}) {
      DGPP_CUDA_OK(cudaMalloc(q, size_t(rows) * k / 2));
      DGPP_CUDA_OK(cudaMemset(*q, 0, size_t(rows) * k / 2));
    }
    for (uint8_t** q : {&s1, &s2}) {
      DGPP_CUDA_OK(cudaMalloc(q, rows * ss));
      DGPP_CUDA_OK(cudaMemset(*q, 0, rows * ss));
    }
    for (float** q : {&g1, &g2}) DGPP_CUDA_OK(cudaMalloc(q, rows * 4));
    const float gs = 5.0f / (6.f * 448.f), limit = std::numeric_limits<float>::infinity();
    dgpp::launch_moe_swiglu_clamp(dg, du, da, int64_t(rows) * stride, limit, 0);
    dgpp::launch_quantize_rows_nvfp4(da, stride, rows, k, c1, s1, g1, 0, gs);
    dgpp::launch_swiglu_quantize_rows_nvfp4(dg, du, stride, rows, k, limit, c2, s2, g2, 0, gs);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint8_t> hc1(size_t(rows) * k / 2), hc2(hc1.size()), hs1(rows * ss), hs2(hs1.size());
    std::vector<float> hg1(rows), hg2(rows);
    DGPP_CUDA_OK(cudaMemcpy(hc1.data(), c1, hc1.size(), cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(hc2.data(), c2, hc2.size(), cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(hs1.data(), s1, hs1.size(), cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(hs2.data(), s2, hs2.size(), cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(hg1.data(), g1, rows * 4, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(hg2.data(), g2, rows * 4, cudaMemcpyDeviceToHost));
    const bool same = hc1 == hc2 && hs1 == hs2 && hg1 == hg2;
    std::printf("[ %s ] fused swiglu+quantize vs the two-kernel chain: %s\n", same ? "OK" : "FAIL",
                same ? "bitwise (codes, scales, globals)" : "DIFFER");
    if (!same) ++fails;
    // The static global read from device memory (the layer image's copy of
    // input_scale) is bitwise the same global passed by value.
    {
      float* dgs;
      DGPP_CUDA_OK(cudaMalloc(&dgs, 4));
      DGPP_CUDA_OK(cudaMemcpy(dgs, &gs, 4, cudaMemcpyHostToDevice));
      DGPP_CUDA_OK(cudaMemset(c2, 0, size_t(rows) * k / 2));
      DGPP_CUDA_OK(cudaMemset(s2, 0, rows * ss));
      dgpp::launch_quantize_rows_nvfp4(da, stride, rows, k, c2, s2, g2, 0, 0.f, dgs);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      DGPP_CUDA_OK(cudaMemcpy(hc2.data(), c2, hc2.size(), cudaMemcpyDeviceToHost));
      DGPP_CUDA_OK(cudaMemcpy(hs2.data(), s2, hs2.size(), cudaMemcpyDeviceToHost));
      DGPP_CUDA_OK(cudaMemcpy(hg2.data(), g2, rows * 4, cudaMemcpyDeviceToHost));
      const bool dev_same = hc1 == hc2 && hs1 == hs2 && hg1 == hg2;
      std::printf("[ %s ] static global from device memory vs by value: %s\n", dev_same ? "OK" : "FAIL",
                  dev_same ? "bitwise (codes, scales, globals)" : "DIFFER");
      if (!dev_same) ++fails;
      cudaFree(dgs);
    }
    for (void* p : {static_cast<void*>(dg), static_cast<void*>(du), static_cast<void*>(da), static_cast<void*>(c1),
                    static_cast<void*>(c2), static_cast<void*>(s1), static_cast<void*>(s2), static_cast<void*>(g1),
                    static_cast<void*>(g2)})
      cudaFree(p);
  }
  std::printf(fails == 0 ? "[ OK ] moe_w4a4_test\n" : "[FAIL] moe_w4a4_test: %d\n", fails);
  return fails == 0 ? 0 : 1;
}
