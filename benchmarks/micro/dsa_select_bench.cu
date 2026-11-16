// dsa_select_bench: the fused decode select kernel alone (DESIGN §7.2), at
// the MTP shape (two rows) over the real blocked index-cache layout, swept
// across context lengths. The decode step's only context-scaled kernel:
// its cost past 512 visible pools is what a 16K-context agent request pays
// on every step (2026-09-06: ~650 us x 12 calls = 7.9 ms of a 54 ms step).
//
// Usage: dsa_select_bench [--rows N] [--iters N] [--warmup N] [--ctx a,b,c]
//                         [--kpool N] [--heads N] [--layers N] [--split 1]
// Prints microseconds per launch per context (mean / min).
// --kpool: tokens a pool. The default 4 is GLM-5.3-Flash's geometry (512
// pools selected); 1 is the full model's (2,048 tokens selected, a pool a
// token), where the selection's cost by context is four times the entries
// and whose id sort went unmeasured until 2026-11-14 because this bench
// only ran the default. --heads: 32, or 64 for the tensor-core scoring.
// --layers N: N distinct index caches walked in turn, so a launch finds its
// index in DRAM as the engine's does (a layer's index is revisited a pass
// later); 1 keeps one cache resident. --split 1: every row a request of its
// own over the same table, so no rows share a staged tile (the rows of one
// request do: the launch's work before 2026-11-16). The ids' hash is printed
// so two runs can be compared.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "kernels/dsa.hpp"
#include "models/dsa_geometry.hpp"

namespace {

uint32_t hash32(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return uint32_t(x);
}
float unit(uint64_t seed, uint64_t i) {  // [0, 1)
  return float(hash32(seed * 0x9E3779B97F4A7C15ULL + i) & 0xFFFFFFu) / float(1u << 24);
}

struct DevBuf {
  void* p = nullptr;
  size_t bytes = 0;
  explicit DevBuf(size_t n) : bytes(n) { DGPP_CUDA_OK(cudaMalloc(&p, n ? n : 16)); }
  ~DevBuf() { cudaFree(p); }
  void upload(const void* src, size_t n) {
    DGPP_CUDA_OK(cudaMemcpy(p, src, n, cudaMemcpyHostToDevice));
  }
  void zero() { DGPP_CUDA_OK(cudaMemset(p, 0, bytes)); }
};

}  // namespace

int main(int argc, char** argv) {
  int rows = 2, iters = 200, warmup = 20, grid = 0, kpool_arg = 4, heads_arg = 32, layers = 1, split = 0;
  std::vector<int64_t> ctxs = {512, 1024, 2048, 4096, 8192, 16384, 32768, 131072, 393216};
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--rows") == 0 && i + 1 < argc) rows = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--grid") == 0 && i + 1 < argc) grid = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--kpool") == 0 && i + 1 < argc) kpool_arg = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--heads") == 0 && i + 1 < argc) heads_arg = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--layers") == 0 && i + 1 < argc) layers = std::max(1, std::atoi(argv[++i]));
    else if (std::strcmp(argv[i], "--split") == 0 && i + 1 < argc) split = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) {
      ctxs.clear();
      std::string s = argv[++i];
      size_t p = 0;
      while (p < s.size()) {
        size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        ctxs.push_back(std::atoll(s.substr(p, q - p).c_str()));
        p = q + 1;
      }
    }
  }
  dgpp::DsaConfig cfg{};
  cfg.index_kpool = kpool_arg;
  cfg.index_n_heads = heads_arg;
  const dgpp::DsaGeometry g = dgpp::DsaGeometry::from_config(cfg);
  const int heads = cfg.index_n_heads, dim = cfg.index_head_dim;
  const int kpool = cfg.index_kpool;
  const int ppb = g.pools_per_block;
  std::printf("dsa_select_bench: rows=%d kpool=%d heads=%d select_k=%d pools_per_block=%d iters=%d grid=%d\n",
              rows, kpool, heads, g.select_k, ppb, iters, grid);
  std::printf("%10s %10s %12s %12s\n", "ctx tok", "pools", "mean us", "min us");
  for (const int64_t ctx : ctxs) {
    const int64_t visible = (ctx + 1) / kpool;
    const int64_t n_blocks = (visible + ppb - 1) / ppb + 1;
    const int64_t n_pools = n_blocks * ppb;
    // Inputs: fp8 q (no NaN codes), small folded weights, random fp8 index
    // rows with positive scales; an identity block table.
    std::vector<uint8_t> q8(size_t(rows) * heads * dim), k8(size_t(n_pools) * dim);
    for (size_t i = 0; i < q8.size(); ++i) {
      uint8_t v = uint8_t(hash32(11 + i) & 0xFF);
      if ((v & 0x7Fu) == 0x7Fu) v ^= 1u;
      q8[i] = v;
    }
    for (size_t i = 0; i < k8.size(); ++i) {
      uint8_t v = uint8_t(hash32(13 + i) & 0xFF);
      if ((v & 0x7Fu) == 0x7Fu) v ^= 1u;
      k8[i] = v;
    }
    std::vector<float> w(size_t(rows) * heads), ks(static_cast<size_t>(n_pools));
    for (size_t i = 0; i < w.size(); ++i) w[i] = (unit(3, i) - 0.5f) * 0.02f;
    for (size_t i = 0; i < ks.size(); ++i) ks[i] = 0.01f + unit(5, i) * 0.05f;
    std::vector<int64_t> pos(static_cast<size_t>(rows));
    // A table row a request (every one the same blocks).
    std::vector<int32_t> req(static_cast<size_t>(rows), 0), bt(static_cast<size_t>(n_blocks) * rows);
    for (int r = 0; r < rows; ++r) {
      pos[size_t(r)] = ctx - rows + r;
      if (split) req[size_t(r)] = r;
      for (int64_t b = 0; b < n_blocks; ++b) bt[size_t(r) * n_blocks + size_t(b)] = int32_t(b);
    }
    DevBuf dq(q8.size()), dk(k8.size() * layers), dw(w.size() * 4), dks(ks.size() * 4 * layers),
        dpos(pos.size() * 8), dreq(req.size() * 4), dbt(bt.size() * 4),
        dtopk(size_t(rows) * g.max_selected * 4), dcnt(size_t(rows) * 4),
        dws(dgpp::dsa_select_workspace_bytes(rows, n_pools)), dctr(8);
    dq.upload(q8.data(), q8.size());
    for (int l = 0; l < layers; ++l)
      DGPP_CUDA_OK(cudaMemcpy(static_cast<uint8_t*>(dk.p) + size_t(l) * k8.size(), k8.data(), k8.size(),
                              cudaMemcpyHostToDevice));
    dw.upload(w.data(), w.size() * 4);
    for (int l = 0; l < layers; ++l)
      DGPP_CUDA_OK(cudaMemcpy(static_cast<float*>(dks.p) + size_t(l) * ks.size(), ks.data(), ks.size() * 4,
                              cudaMemcpyHostToDevice));
    dpos.upload(pos.data(), pos.size() * 8);
    dreq.upload(req.data(), req.size() * 4);
    dbt.upload(bt.data(), bt.size() * 4);
    dws.zero();
    dctr.zero();
    cudaStream_t s;
    DGPP_CUDA_OK(cudaStreamCreate(&s));
    int layer = 0;
    auto launch = [&] {
      layer = (layer + 1) % layers;
      dgpp::dsa_select_decode(dq.p, static_cast<const float*>(dw.p),
                              static_cast<const int32_t*>(dreq.p),
                              static_cast<const int64_t*>(dpos.p), rows,
                              static_cast<const int32_t*>(dbt.p), int(n_blocks),
                              static_cast<const uint8_t*>(dk.p) + size_t(layer) * k8.size(),
                              static_cast<const float*>(dks.p) + size_t(layer) * ks.size(), ppb, heads,
                              dim, g.select_k, kpool, g.max_selected,
                              static_cast<int32_t*>(dtopk.p),
                              static_cast<int32_t*>(dcnt.p), dws.p, n_pools,
                              static_cast<int32_t*>(dctr.p), grid, s);
    };
    for (int i = 0; i < warmup; ++i) launch();
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
    cudaEvent_t e0, e1;
    DGPP_CUDA_OK(cudaEventCreate(&e0));
    DGPP_CUDA_OK(cudaEventCreate(&e1));
    float best = 1e30f, total = 0.f;
    for (int i = 0; i < iters; ++i) {
      DGPP_CUDA_OK(cudaEventRecord(e0, s));
      launch();
      DGPP_CUDA_OK(cudaEventRecord(e1, s));
      DGPP_CUDA_OK(cudaEventSynchronize(e1));
      float ms = 0.f;
      DGPP_CUDA_OK(cudaEventElapsedTime(&ms, e0, e1));
      total += ms;
      if (ms < best) best = ms;
    }
    // A sanity read: the first row's selected count; the phase stamps.
    int32_t cnt = -1;
    DGPP_CUDA_OK(cudaMemcpy(&cnt, dcnt.p, 4, cudaMemcpyDeviceToHost));
    uint64_t ph[8] = {};
    dgpp::dsa_select_debug_phases(ph);
    uint64_t ids_hash = 1469598103934665603ULL;
    {
      std::vector<int32_t> out(size_t(rows) * g.max_selected);
      DGPP_CUDA_OK(cudaMemcpy(out.data(), dtopk.p, out.size() * 4, cudaMemcpyDeviceToHost));
      for (const int32_t v : out) ids_hash = (ids_hash ^ uint32_t(v)) * 1099511628211ULL;
    }
    std::printf("%10lld %10lld %12.1f %12.1f   (row 0 selected %d tokens; last block: "
                "score %.1f wait %.1f select %.1f expand %.1f us; ids %016llx)\n",
                (long long)ctx, (long long)visible, total / iters * 1e3f,
                best * 1e3f, cnt, (ph[1] - ph[0]) / 1e3, (ph[2] - ph[1]) / 1e3,
                ph[3] / 1e3, ph[4] / 1e3, static_cast<unsigned long long>(ids_hash));
    DGPP_CUDA_OK(cudaEventDestroy(e0));
    DGPP_CUDA_OK(cudaEventDestroy(e1));
    DGPP_CUDA_OK(cudaStreamDestroy(s));
  }
  return 0;
}
