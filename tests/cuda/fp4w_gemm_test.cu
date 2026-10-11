// fp4w_gemm_test: the NVFP4 prefill GEMM against an exact host oracle,
// against the proven grouped production kernel (tolerance-equal: a
// different fp32 summation order), and sparse codes bitwise (no
// accumulation anywhere, so any slip is exact, not noise).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "kernels/fp4w_gemm.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/latent_format.hpp"
#include "models/quant_matrix.hpp"

namespace {

#define CHECK_CUDA(x) DGPP_CUDA_OK(x)

int failures = 0;
#define EXPECT_TRUE(cond, msg)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      ++failures;                                                              \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);                \
    }                                                                          \
  } while (0)

uint16_t float_to_bf16_bits_host(float f) {
  uint32_t u;
  __builtin_memcpy(&u, &f, 4);
  const uint32_t bias = 0x7FFFu + ((u >> 16) & 1u);
  return static_cast<uint16_t>((u + bias) >> 16);
}
float bf16_bits_to_float_host(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}

struct Case {
  int m, n, k;
  std::vector<uint16_t> act;  // [m, k] bf16 bits
  std::vector<uint8_t> payload, scales;
  float global;
};

Case make_case(int m, int n, int k, float global, unsigned seed) {
  Case c{m, n, k, {}, {}, {}, global};
  std::srand(seed);
  c.act.resize(static_cast<size_t>(m) * k);
  for (auto& a : c.act) {
    const float v = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 0.25f;  // [-0.125, 0.125]
    c.act[&a - c.act.data()] = float_to_bf16_bits_host(v);
  }
  c.payload.resize(static_cast<size_t>(n) * (k / 2));
  for (auto& p : c.payload) p = static_cast<uint8_t>(std::rand() & 0xFF);
  // Well-conditioned scales: powers of two in [0.5, 2]. Wild uniform bytes
  // make the dots ill-conditioned and the failure is fp32 summation order,
  // not the kernel.
  static const float choices[] = {0.5f, 1.0f, 2.0f};
  c.scales.resize(static_cast<size_t>(n) * (k / 16));
  for (auto& s : c.scales) s = dgpp::float_to_fp8_e4m3_bits(choices[std::rand() % 3]);
  return c;
}

// The exact oracle: sum of bf16(act) * (e2m1 * e4m3) in double, divided by
// the global once, rounded to bf16. The device's per-pair product is exact
// (<= 7 significant bits), so only the fp32 accumulation order separates
// the two.
std::vector<uint16_t> oracle(const Case& c) {
  std::vector<uint16_t> out(static_cast<size_t>(c.m) * c.n);
  for (int i = 0; i < c.m; ++i)
    for (int j = 0; j < c.n; ++j) {
      double acc = 0.0;
      for (int t = 0; t < c.k; ++t) {
        const uint8_t byte = c.payload[(static_cast<size_t>(j) * c.k + t) / 2];
        const uint8_t code = (t & 1) ? (byte >> 4) : (byte & 0xF);
        const uint8_t sc = c.scales[static_cast<size_t>(j) * (c.k / 16) + t / 16];
        acc += static_cast<double>(bf16_bits_to_float_host(c.act[static_cast<size_t>(i) * c.k + t])) *
               static_cast<double>(dgpp::fp4_e2m1_bits_to_float(code)) *
               static_cast<double>(dgpp::fp8_e4m3_bits_to_float(sc));
      }
      out[static_cast<size_t>(i) * c.n + j] =
          float_to_bf16_bits_host(static_cast<float>(acc / c.global));
    }
  return out;
}

struct Dev {
  uint16_t *act = nullptr, *out = nullptr;
  uint8_t *payload = nullptr, *scales = nullptr;
  float* global = nullptr;
  cudaStream_t s = nullptr;
};

Dev upload(const Case& c) {
  using namespace dgpp;
  Dev d;
  CHECK_CUDA(cudaMalloc(&d.act, c.act.size() * 2));
  CHECK_CUDA(cudaMalloc(&d.out, static_cast<size_t>(c.m) * c.n * 2));
  CHECK_CUDA(cudaMalloc(&d.payload, c.payload.size()));
  CHECK_CUDA(cudaMalloc(&d.scales, c.scales.size()));
  CHECK_CUDA(cudaMalloc(&d.global, 4));
  CHECK_CUDA(cudaMemcpy(d.act, c.act.data(), c.act.size() * 2, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d.payload, c.payload.data(), c.payload.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d.scales, c.scales.data(), c.scales.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d.global, &c.global, 4, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaStreamCreate(&d.s));
  return d;
}

std::vector<uint16_t> run_fp4w(const Case& c) {
  using namespace dgpp;
  Dev d = upload(c);
  GlmFp4Matrix w{d.payload, d.scales, d.global, c.n, c.k, kFp4Group};
  launch_fp4w_gemm_bf16(d.act, c.k, w, d.out, c.m, c.n, c.k, d.s);
  CHECK_CUDA(cudaStreamSynchronize(d.s));
  std::vector<uint16_t> out(static_cast<size_t>(c.m) * c.n);
  CHECK_CUDA(cudaMemcpy(out.data(), d.out, out.size() * 2, cudaMemcpyDeviceToHost));
  return out;
}

std::vector<uint16_t> run_grouped(const Case& c) {
  using namespace dgpp;
  Dev d = upload(c);
  GlmFp4Matrix w{d.payload, d.scales, d.global, c.n, c.k, kFp4Group};
  MoeSegment hseg{0, c.m, 0};
  MoeExpertView hview = MoeExpertView::of(w);
  MoeSegment* dseg;
  MoeExpertView* dview;
  CHECK_CUDA(cudaMalloc(&dseg, sizeof(hseg)));
  CHECK_CUDA(cudaMalloc(&dview, sizeof(hview)));
  CHECK_CUDA(cudaMemcpy(dseg, &hseg, sizeof(hseg), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dview, &hview, sizeof(hview), cudaMemcpyHostToDevice));
  launch_dense_mma_fp4_prod_bf16(d.act, c.k, dseg, dview, 0, d.out, c.m, c.n, c.k, d.s,
                                 kFp4Group);
  CHECK_CUDA(cudaStreamSynchronize(d.s));
  std::vector<uint16_t> out(static_cast<size_t>(c.m) * c.n);
  CHECK_CUDA(cudaMemcpy(out.data(), d.out, out.size() * 2, cudaMemcpyDeviceToHost));
  return out;
}

double max_rel_err(const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref) {
  double worst = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double g = bf16_bits_to_float_host(got[i]), r = bf16_bits_to_float_host(ref[i]);
    if (fabs(r) < 0.05) continue;  // cancellation floor: near-zero dots carry no signal
    const double e = fabs(g - r) / fabs(r);
    if (e > worst) worst = e;
  }
  return worst;
}

void check_oracle(int m, int n, int k, float global, unsigned seed, const char* tag) {
  Case c = make_case(m, n, k, global, seed);
  const double e = max_rel_err(run_fp4w(c), oracle(c));
  char msg[256];
  std::snprintf(msg, sizeof(msg), "%s: rel err %g", tag, e);
  EXPECT_TRUE(e < 2e-3, msg);
}

void check_vs_grouped(int m, int n, int k, unsigned seed, const char* tag) {
  Case c = make_case(m, n, k, 6400.f, seed);
  const double e = max_rel_err(run_fp4w(c), run_grouped(c));
  char msg[256];
  std::snprintf(msg, sizeof(msg), "%s: rel err %g", tag, e);
  EXPECT_TRUE(e < 1e-2, msg);
}

// Sparse codes (one live pair per row) against the grouped kernel,
// BITWISE: no accumulation anywhere, so any slip is exact, not noise.
void check_sparse_vs_grouped(int m, int n, int k) {
  Case c;
  c.m = m;
  c.n = n;
  c.k = k;
  c.global = 1.0f;
  c.act.assign(static_cast<size_t>(m) * k, 0x3F80);
  c.payload.assign(static_cast<size_t>(n) * (k / 2), 0x00);
  c.scales.assign(static_cast<size_t>(n) * (k / 16), 0x38);
  for (int i = 0; i < m; ++i)
    c.payload[static_cast<size_t>(i % n) * (k / 2) + (i * 7) % (k / 2)] = 0x41;
  const std::vector<uint16_t> mine = run_fp4w(c), ref = run_grouped(c);
  bool same = mine.size() == ref.size();
  for (size_t i = 0; i < mine.size() && same; ++i) same = mine[i] == ref[i];
  char msg[128];
  std::snprintf(msg, sizeof(msg), "sparse vs grouped m=%d n=%d k=%d", m, n, k);
  EXPECT_TRUE(same, msg);
}

// The vector store must preserve row padding and work through unaligned
// output views. Odd N also leaves a final scalar column in an aligned row.
void check_output_layouts() {
  using namespace dgpp;
  const Case c = make_case(3073, 137, 256, 1.7f, 51);
  const auto ref = run_fp4w(c);
  Dev d = upload(c);
  const GlmFp4Matrix w{d.payload, d.scales, d.global, c.n, c.k, kFp4Group};
  constexpr uint16_t sentinel = 0xA55A;
  for (int stride : {c.n, c.n + 1, c.n + 3}) {
    for (int offset : {0, 1}) {
      const size_t count = static_cast<size_t>(c.m) * stride + 16;
      std::vector<uint16_t> host(count, sentinel);
      uint16_t* allocation = nullptr;
      CHECK_CUDA(cudaMalloc(&allocation, count * sizeof(uint16_t)));
      CHECK_CUDA(
          cudaMemcpy(allocation, host.data(), count * sizeof(uint16_t), cudaMemcpyHostToDevice));
      auto* output = allocation + 4 + offset;
      launch_fp4w_gemm_bf16(d.act, c.k, w, output, c.m, c.n, c.k, d.s, stride);
      CHECK_CUDA(cudaStreamSynchronize(d.s));
      CHECK_CUDA(
          cudaMemcpy(host.data(), allocation, count * sizeof(uint16_t), cudaMemcpyDeviceToHost));
      bool same = true;
      for (size_t index = 0; index < count; ++index) {
        const int64_t relative = static_cast<int64_t>(index) - 4 - offset;
        uint16_t expected = sentinel;
        if (relative >= 0 && relative / stride < c.m && relative % stride < c.n)
          expected = ref[static_cast<size_t>(relative / stride) * c.n + relative % stride];
        same &= host[index] == expected;
      }
      EXPECT_TRUE(same, "strided/unaligned output and guard values must match");
      CHECK_CUDA(cudaFree(allocation));
    }
  }
}

}  // namespace

int main() {
  check_oracle(144, 512, 512, 3200.f, 42, "oracle ragged tiles");
  check_oracle(512, 512, 1024, 3200.f, 43, "oracle full tiles");
  check_oracle(256, 1024, 5120, 6400.f, 44, "oracle gate-K slice");
  check_sparse_vs_grouped(144, 512, 512);
  check_sparse_vs_grouped(512, 1024, 5120);
  check_sparse_vs_grouped(200, 5120, 17408);
  check_vs_grouped(512, 4352, 5120, 45, "vs grouped gate");
  check_vs_grouped(256, 5120, 17408, 46, "vs grouped down");
  // Past 1024 rows, the tile order spans several m-groups
  // (m_tiles > kGroupM: a full 4-tile group
  // plus a tail group, the 2080-token prefill's pattern). Below, every
  // case above stays single-group and never walks the group scheduler.
  check_oracle(1152, 512, 256, 3200.f, 47, "oracle multi-group tail");
  // A ragged 256-row tile and N tail, with outputs above the oracle's
  // cancellation floor; this also crosses the model's 1024-row dispatch.
  check_oracle(1025, 136, 256, 1.7f, 49, "oracle wide tile and column tails");
  check_oracle(1024, 136, 256, 1.7f, 50, "oracle prefill dispatch boundary");
  check_oracle(3073, 137, 256, 1.7f, 51, "oracle long prefill and odd column tail");
  check_output_layouts();
  check_sparse_vs_grouped(1152, 512, 256);
  check_vs_grouped(1152, 5120, 17408, 48, "vs grouped multi-group");
  if (failures == 0) std::printf("fp4w_gemm_test: all passed\n");
  return failures == 0 ? 0 : 1;
}
