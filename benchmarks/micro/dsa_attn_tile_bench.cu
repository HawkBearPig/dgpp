// dsa_attn_tile_bench: a prefill tile's listed attention alone (the full
// GLM-5.3's geometry: kv_lora 512 + the 64-wide rope tail, fp8 cache rows,
// 16 local heads, 128 rows a tile, 4 splits), straight from the cache
// (dsa_attn_listed) and from the staged copy of the tile's distinct rows
// (dsa_attn_listed_staged, its three staging launches included), and
// whether the two forms' partials are bit for bit the same.
//
// The rows' lists drift as a prefill tile's do: row 0 selects --sel tokens
// of the context and each next row trades --drift of the row before's for
// new ones, so a tile lists sel + (rows - 1) * drift distinct tokens (the
// full model on the fabric, 2026-11-16: 11,500 a tile on average over 8K to
// 65K tokens of context, 38,000 at most). --layers distinct caches are
// walked in turn, so a launch finds its rows in DRAM as the engine's does.
// --slots: the staged copy's rows (default: the layers' own rule,
// dsa_attn_stage_slots); a tile with more distinct rows reads the cache.
//
// Usage: dsa_attn_tile_bench [--rows N] [--ctx N] [--sel N] [--drift N]
//                            [--layers N] [--slots N] [--passes N] [--splits N]
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
#include "kernels/latent_format.hpp"

namespace {
uint32_t hash32(uint64_t x) {
  x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
  return uint32_t(x);
}
float unit(uint64_t seed, uint64_t i) {
  return float(hash32(seed * 0x9E3779B97F4A7C15ULL + i) & 0xFFFFFFu) / float(1u << 24);
}
uint16_t bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
}  // namespace

int main(int argc, char** argv) {
  int rows = 128, ctx = 24576, sel = 2048, drift = 38, layers = 16, slots = 0, passes = 10, splits = 4, combine = 0;
  for (int i = 1; i < argc; ++i) {
    auto is = [&](const char* k) { return std::strcmp(argv[i], k) == 0 && i + 1 < argc; };
    if (is("--rows")) rows = std::atoi(argv[++i]);
    else if (is("--ctx")) ctx = std::atoi(argv[++i]);
    else if (is("--sel")) sel = std::atoi(argv[++i]);
    else if (is("--drift")) drift = std::atoi(argv[++i]);
    else if (is("--layers")) layers = std::atoi(argv[++i]);
    else if (is("--slots")) slots = std::atoi(argv[++i]);
    else if (is("--passes")) passes = std::atoi(argv[++i]);
    else if (is("--splits")) splits = std::atoi(argv[++i]);
    else if (is("--combine")) combine = std::atoi(argv[++i]);
  }
  using dgpp::LatentFormat;
  constexpr int KV = 512, ROPE = 64, SD = KV + ROPE, LH = 16, BT = 128;
  const LatentFormat fmt = LatentFormat::kFp8;
  const size_t row_bytes = dgpp::latent_row_bytes(fmt, KV) + size_t(ROPE) * 2;
  ctx = (ctx + BT - 1) / BT * BT;
  if (slots <= 0) slots = dgpp::dsa_attn_stage_slots(SD, ctx);
  const int blocks = ctx / BT;
  sel = std::min(sel, ctx);

  uint8_t* cache = nullptr; float* scales = nullptr;
  const size_t cache_bytes = size_t(ctx) * row_bytes;
  DGPP_CUDA_OK(cudaMalloc(&cache, cache_bytes * layers));
  DGPP_CUDA_OK(cudaMalloc(&scales, size_t(ctx) * 4 * layers));
  {
    std::vector<uint8_t> h(cache_bytes);
    std::vector<float> s(static_cast<size_t>(ctx));
    for (int64_t t = 0; t < ctx; ++t) {
      uint8_t* r = h.data() + size_t(t) * row_bytes;
      for (int e = 0; e < KV; e += 4) {
        uint32_t v = hash32(uint64_t(t) * 131 + e);
        for (int b = 0; b < 4; ++b) {
          uint8_t c = uint8_t(v >> (8 * b));
          if ((c & 0x7Fu) == 0x7Fu) c ^= 1u;  // no NaN codes
          r[e + b] = c;
        }
      }
      const size_t tail = dgpp::latent_row_bytes(fmt, KV);
      for (int e = 0; e < ROPE; ++e) {
        const uint16_t v = bf16((unit(7, uint64_t(t) * 64 + e) - 0.5f) * 0.5f);
        std::memcpy(r + tail + e * 2, &v, 2);
      }
      s[size_t(t)] = 0.002f + unit(5, uint64_t(t)) * 0.004f;
    }
    for (int l = 0; l < layers; ++l) {
      DGPP_CUDA_OK(cudaMemcpy(cache + size_t(l) * cache_bytes, h.data(), cache_bytes, cudaMemcpyHostToDevice));
      DGPP_CUDA_OK(cudaMemcpy(scales + size_t(l) * ctx, s.data(), size_t(ctx) * 4, cudaMemcpyHostToDevice));
    }
  }
  std::vector<uint16_t> q(size_t(rows) * LH * SD);
  for (size_t i = 0; i < q.size(); ++i) q[i] = bf16((unit(3, i) - 0.5f) * 0.2f);
  // The drifting lists, each ascending.
  std::vector<int32_t> topk(size_t(rows) * sel), counts(size_t(rows), sel), req(size_t(rows), 0),
      bt(static_cast<size_t>(blocks));
  for (int b = 0; b < blocks; ++b) bt[size_t(b)] = b;
  size_t distinct = 0;
  {
    std::vector<char> in(static_cast<size_t>(ctx), 0), ever(static_cast<size_t>(ctx), 0);
    std::vector<int32_t> cur;
    uint64_t n = 0;
    while (int(cur.size()) < sel) {
      const int32_t t = int32_t(hash32(4242 * 7919 + n++) % uint32_t(ctx));
      if (!in[size_t(t)]) { in[size_t(t)] = 1; cur.push_back(t); }
    }
    for (int r = 0; r < rows; ++r) {
      if (r > 0) {
        for (int d = 0; d < std::min(drift, sel); ++d) {
          const size_t out = hash32(uint64_t(r) * 1000003 + d) % cur.size();
          int32_t t;
          do { t = int32_t(hash32(4242 * 7919 + n++) % uint32_t(ctx)); } while (in[size_t(t)]);
          in[size_t(cur[out])] = 0; in[size_t(t)] = 1; cur[out] = t;
        }
      }
      std::vector<int32_t> pick = cur;
      std::sort(pick.begin(), pick.end());
      std::copy(pick.begin(), pick.end(), topk.begin() + size_t(r) * sel);
      for (int32_t t : pick) ever[size_t(t)] = 1;
    }
    for (char c : ever) distinct += c;
  }
  std::printf("dsa_attn_tile_bench: %d rows a tile, ctx %d, %d selected a row, %zu distinct a tile, %d splits, "
              "%d layer cache(s) of %.1f MB, copy of %d rows\n",
              rows, ctx, sel, distinct, splits, layers, double(cache_bytes) / 1e6, slots);
  auto dev = [](const void* src, size_t bytes) {
    void* p = nullptr;
    DGPP_CUDA_OK(cudaMalloc(&p, bytes));
    if (src) DGPP_CUDA_OK(cudaMemcpy(p, src, bytes, cudaMemcpyHostToDevice));
    else DGPP_CUDA_OK(cudaMemset(p, 0, bytes));
    return p;
  };
  auto* dq = static_cast<uint16_t*>(dev(q.data(), q.size() * 2));
  auto* dtopk = static_cast<int32_t*>(dev(topk.data(), topk.size() * 4));
  auto* dcounts = static_cast<int32_t*>(dev(counts.data(), counts.size() * 4));
  auto* dreq = static_cast<int32_t*>(dev(req.data(), req.size() * 4));
  auto* dbt = static_cast<int32_t*>(dev(bt.data(), bt.size() * 4));
  const size_t parts = size_t(rows) * splits * LH;
  float* ws[2][3];
  for (int k = 0; k < 2; ++k) {
    ws[k][0] = static_cast<float*>(dev(nullptr, parts * 4));
    ws[k][1] = static_cast<float*>(dev(nullptr, parts * 4));
    ws[k][2] = static_cast<float*>(dev(nullptr, parts * KV * 4));
  }
  auto* c_out = static_cast<float*>(dev(nullptr, size_t(rows) * LH * KV * 4));
  const size_t stage_bytes = dgpp::dsa_attn_stage_bytes(ctx, slots, SD);
  void* stage = dev(nullptr, stage_bytes);
  cudaStream_t s;
  DGPP_CUDA_OK(cudaStreamCreate(&s));
  cudaEvent_t e0, e1;
  DGPP_CUDA_OK(cudaEventCreate(&e0));
  DGPP_CUDA_OK(cudaEventCreate(&e1));
  const auto pass = [&](int staged) {
    for (int l = 0; l < layers; ++l) {
      const uint8_t* cl = cache + size_t(l) * cache_bytes;
      const float* sl = scales + size_t(l) * ctx;
      bool ok;
      if (staged)
        ok = dgpp::dsa_attn_listed_staged(dq, cl, 0, dreq, dtopk, sel, dcounts, rows, splits, LH, KV, BT, dbt,
                                          blocks, 0.05f, ws[1][0], ws[1][1], ws[1][2], s, fmt, sl, ROPE, -1, stage,
                                          ctx, slots);
      else
        ok = dgpp::dsa_attn_listed(dq, cl, dreq, dtopk, sel, dcounts, rows, splits, LH, KV, BT, dbt, blocks, 0.05f,
                                   ws[0][0], ws[0][1], ws[0][2], s, fmt, sl, ROPE);
      if (!ok) { std::fprintf(stderr, "the kernel refused the geometry\n"); std::exit(1); }
      if (combine)
        dgpp::dsa_attn_combine(ws[staged][0], ws[staged][1], ws[staged][2], rows, splits, LH, KV, c_out, s);
    }
  };
  double us[2] = {0, 0}, best[2] = {1e30, 1e30};
  for (int staged = 0; staged < 2; ++staged) {
    for (int i = 0; i < 2; ++i) pass(staged);
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
    for (int i = 0; i < passes; ++i) {
      DGPP_CUDA_OK(cudaEventRecord(e0, s));
      pass(staged);
      DGPP_CUDA_OK(cudaEventRecord(e1, s));
      DGPP_CUDA_OK(cudaEventSynchronize(e1));
      float ms = 0.f;
      DGPP_CUDA_OK(cudaEventElapsedTime(&ms, e0, e1));
      us[staged] += ms * 1e3 / layers;
      best[staged] = std::min(best[staged], double(ms) * 1e3 / layers);
    }
    us[staged] /= passes;
  }
  // The two forms' partials, bit for bit.
  bool same = true;
  for (int k = 0; k < 3; ++k) {
    const size_t n = parts * (k == 2 ? KV : 1);
    std::vector<float> a(n), b(n);
    DGPP_CUDA_OK(cudaMemcpy(a.data(), ws[0][k], n * 4, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(b.data(), ws[1][k], n * 4, cudaMemcpyDeviceToHost));
    if (std::memcmp(a.data(), b.data(), n * 4) != 0) same = false;
  }
  unsigned long long st[4] = {0, 0, 0, 0};
  dgpp::dsa_attn_stage_stats(st, false, s);
  std::printf("from the cache %8.1f us a tile (min %8.1f) | staged %8.1f us a tile (min %8.1f) | partials %s | "
              "staged launches %llu, not staged %llu, distinct %.0f mean / %llu max\n",
              us[0], best[0], us[1], best[1], same ? "bit for bit" : "DIFFER", st[0], st[1],
              st[0] ? double(st[2]) / double(st[0]) : 0.0, st[3]);
  return same ? 0 : 1;
}
