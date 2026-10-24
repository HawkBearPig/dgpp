// Paired graph timings of BF16 KV, FP8 storage with BF16 compute, and
// native FP8 QSA. No checkpoint is required. Output is CSV on stdout.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/dsa.hpp"
#include "kernels/latent_format.hpp"
#include "kernels/qsa.hpp"

namespace {
template<class T> struct Buffer {
  T* p = nullptr;
  size_t size;
  explicit Buffer(size_t n) : size(n) { DGPP_CUDA_OK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T))); }
  explicit Buffer(const std::vector<T>& v) : Buffer(v.size()) {
    DGPP_CUDA_OK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~Buffer() { cudaFree(p); }
  Buffer(const Buffer&) = delete;
  std::vector<T> read() const {
    std::vector<T> v(size);
    DGPP_CUDA_OK(cudaMemcpy(v.data(), p, size * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }
};
double relative(const std::vector<float>& have, const std::vector<float>& want) {
  double num = 0, den = 0;
  for (size_t i = 0; i < have.size(); ++i) {
    if (!std::isfinite(have[i])) throw std::runtime_error("nonfinite attention output");
    const double e = double(have[i]) - want[i];
    num += e * e; den += double(want[i]) * want[i];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}
}
int main(int argc, char** argv) {
  try {
    int rows = 1, context = 32768, selected = 2051, heads = 12, kv = 1;
    int splits = 8, iters = 50, trials = 7;
    bool prefill = false;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--prefill") { prefill = true; continue; }
      if (i + 1 == argc) throw std::invalid_argument("missing value");
      const int v = std::stoi(argv[++i]);
      if (a == "--rows") rows = v;
      else if (a == "--context") context = v;
      else if (a == "--selected") selected = v;
      else if (a == "--heads") heads = v;
      else if (a == "--kv-heads") kv = v;
      else if (a == "--splits") splits = v;
      else if (a == "--iters") iters = v;
      else if (a == "--trials") trials = v;
      else throw std::invalid_argument("unknown option " + a);
    }
    if (rows < 1 || context < 1 || selected < 1 || selected > context || heads < 1 ||
        kv < 1 || heads % kv || splits < 1 || iters < 1 || trials < 1)
      throw std::invalid_argument("invalid geometry");
    constexpr int dim = 256, block = 64, requests = 2;
    const int blocks = (context + block - 1) / block, slots = blocks * block * requests;
    const int width = kv * dim, qw = heads * dim;
    std::mt19937 rng(20261017);
    std::normal_distribution<float> normal;
    std::vector<uint16_t> qh(size_t(rows) * qw), kh(size_t(slots) * width), vh(kh.size());
    for (auto* a : {&qh, &kh, &vh})
      for (auto& x : *a) x = dgpp::float_to_bf16_bits(normal(rng));
    std::vector<uint8_t> k8h(kh.size()), v8h(vh.size());
    std::vector<float> ksh(size_t(slots) * kv), vsh(ksh.size());
    for (size_t r = 0; r < ksh.size(); ++r) {
      float ka = 0, va = 0;
      for (int d = 0; d < dim; ++d) {
        ka = std::max(ka, std::fabs(dgpp::bf16_bits_to_float(kh[r * dim + d])));
        va = std::max(va, std::fabs(dgpp::bf16_bits_to_float(vh[r * dim + d])));
      }
      const auto ks = dgpp::latent_fp8_row_scale(ka), vs = dgpp::latent_fp8_row_scale(va);
      ksh[r] = ks.scale; vsh[r] = vs.scale;
      for (int d = 0; d < dim; ++d) {
        k8h[r * dim + d] = dgpp::latent_fp8_encode(dgpp::bf16_bits_to_float(kh[r * dim + d]), ks.inv);
        v8h[r * dim + d] = dgpp::latent_fp8_encode(dgpp::bf16_bits_to_float(vh[r * dim + d]), vs.inv);
      }
    }
    std::vector<int32_t> tableh(blocks * requests), reqh(rows), listh(size_t(rows) * selected), counth(rows, selected);
    std::iota(tableh.begin(), tableh.end(), 0);
    std::shuffle(tableh.begin(), tableh.end(), rng);
    for (int r = 0; r < rows; ++r) {
      reqh[r] = r % requests;
      // Different sparse lists, sorted in logical token order.
      std::vector<int32_t> list(context);
      std::iota(list.begin(), list.end(), 0);
      std::shuffle(list.begin(), list.end(), rng);
      list.resize(selected); std::sort(list.begin(), list.end());
      std::copy(list.begin(), list.end(), listh.begin() + size_t(r) * selected);
    }
    Buffer<uint16_t> q(qh), k(kh), v(vh);
    Buffer<uint8_t> k8(k8h), v8(v8h);
    Buffer<float> ks(ksh), vs(vsh);
    Buffer<int32_t> table(tableh), req(reqh), lists(listh), counts(counth);
    const size_t part = size_t(rows) * std::max(8, splits) * heads, output = size_t(rows) * heads * dim;
    Buffer<float> m(part), l(part), c(part * dim), o0(output), o1(output), o2(output);
    float* outs[] = {o0.p, o1.p, o2.p};
    cudaStream_t stream;
    DGPP_CUDA_OK(cudaStreamCreate(&stream));
    auto launch = [&](int mode) {
      if (mode == 2) {
        dgpp::qsa_attn_fp8(q.p, qw, k8.p, v8.p, ks.p, vs.p, req.p, lists.p, selected, counts.p,
            rows, heads, kv, block, table.p, blocks, 1.f / 16.f, splits,
            m.p, l.p, c.p, outs[mode], stream);
        return;
      }
      const auto kc = mode ? reinterpret_cast<uint16_t*>(k8.p) : k.p;
      const auto vc = mode ? reinterpret_cast<uint16_t*>(v8.p) : v.p;
      if (prefill && rows >= 128 && dgpp::qsa_warp_supported(dim, heads, kv)) {
        dgpp::qsa_attn_prefill_warp(q.p, qw, kc, vc, req.p, lists.p, selected, counts.p,
            rows, heads, kv, block, table.p, blocks, 1.f / 16.f, outs[mode], stream,
            mode ? ks.p : nullptr, mode ? vs.p : nullptr);
      } else {
        dgpp::qsa_attn_partial(q.p, qw, kc, vc, req.p, lists.p, selected, counts.p,
            rows, 8, heads, kv, dim, block, table.p, blocks, 1.f / 16.f,
            m.p, l.p, c.p, stream, mode ? ks.p : nullptr, mode ? vs.p : nullptr);
        dgpp::dsa_attn_combine(m.p, l.p, c.p, rows, 8, heads, dim, outs[mode], stream);
      }
    };
    cudaGraph_t graphs[3];
    cudaGraphExec_t execs[3];
    for (int mode = 0; mode < 3; ++mode) {
      launch(mode);
      DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      launch(mode);
      DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graphs[mode]));
      DGPP_CUDA_OK(cudaGraphInstantiate(&execs[mode], graphs[mode], nullptr, nullptr, 0));
    }
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    const auto ref = o0.read(), storage = o1.read(), native = o2.read();
    std::vector<double> times[3];
    cudaEvent_t start, end;
    DGPP_CUDA_OK(cudaEventCreate(&start)); DGPP_CUDA_OK(cudaEventCreate(&end));
    for (int trial = -1; trial < trials; ++trial)
      for (int b = 0; b < 3; ++b) {
        const int mode = (b + trial + 3) % 3;
        for (int i = 0; i < 5; ++i) DGPP_CUDA_OK(cudaGraphLaunch(execs[mode], stream));
        DGPP_CUDA_OK(cudaEventRecord(start, stream));
        for (int i = 0; i < iters; ++i) DGPP_CUDA_OK(cudaGraphLaunch(execs[mode], stream));
        DGPP_CUDA_OK(cudaEventRecord(end, stream)); DGPP_CUDA_OK(cudaEventSynchronize(end));
        float ms; DGPP_CUDA_OK(cudaEventElapsedTime(&ms, start, end));
        if (trial >= 0) times[mode].push_back(ms * 1000 / iters);
      }
    for (auto& t : times) std::sort(t.begin(), t.end());
    std::puts("rows,context,selected,heads,kv,prefill,native_splits,bf16_us,storage_us,native_us,storage_vs_bf16_l2,native_vs_bf16_l2,native_vs_storage_l2");
    std::printf("%d,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.6f,%.6f,%.6f\n",
        rows, context, selected, heads, kv, int(prefill), splits,
        times[0][trials / 2], times[1][trials / 2], times[2][trials / 2],
        relative(storage, ref), relative(native, ref), relative(native, storage));
    for (int b = 0; b < 3; ++b) { cudaGraphExecDestroy(execs[b]); cudaGraphDestroy(graphs[b]); }
    cudaEventDestroy(start); cudaEventDestroy(end); cudaStreamDestroy(stream);
    return 0;
  } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
