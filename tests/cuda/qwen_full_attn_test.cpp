// Layer test for QwenFullAttnLayer (Qwen3.5 Full attention, no indexer).
//
// Exercises the real layer class end to end on a tiny geometry (lh=2,
// lkv=1, D=256, H=512) against a host reference of the full chain
// (projections, RMSNorm + partial RoPE via the qwen_ref oracle, dense
// causal GQA with the kernel's FA2 tiling rule, output gate, o_proj):
//   - prefill T=4 (per-row full_attn_prefill_warp path)
//   - decode  T=1 (qsa_attn_partial listed path over the identity topk)
//   - determinism (prefill twice, bitwise equal)
// Tolerance mirrors the glm4 split class (2 ulp abs + 0.5% RMS floor).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/gemm.hpp"
#include "models/qwen/config.hpp"
#include "models/qwen/layers.hpp"
#include "models/qwen/loader.hpp"
#include "models/qwen/qsa_reference.hpp"

namespace {

using namespace dgpp::kda_test;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error("require failed: " + what);
}

#define DGPP_CUDA_OK_TEST(call)                                                  \
  do {                                                                           \
    cudaError_t err__ = (call);                                                  \
    if (err__ != cudaSuccess)                                                    \
      throw std::runtime_error(std::string("cuda: ") + cudaGetErrorString(err__) \
                               + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
  } while (0)

constexpr int kD = 256;       // head dim (matches full_attn_supported)
constexpr int kLh = 2;        // local heads
constexpr int kLkv = 1;       // local kv heads
constexpr int kH = kLh * kD;  // 512
constexpr int kQW = kLh * 2 * kD;   // [q|gate] interleave width
constexpr int kKW = kLkv * kD;
constexpr int kRot = 64;      // partial rotary dim
constexpr double kTheta = 1e7;
constexpr float kEps = 1e-6f;
constexpr float kScale = 1.0f / 16.0f;  // 1/sqrt(D)
constexpr int kBlockTokens = 32;
constexpr int kMaxTokens = 16;
constexpr int kMaxReq = 1;
constexpr int kBlocksPerReq = 2;
constexpr int kPoolTokens = kBlockTokens * kBlocksPerReq;  // 64

uint16_t f2h(float v) {
  uint32_t b;
  std::memcpy(&b, &v, 4);
  // Proper bf16 round-to-nearest-even: keep the top 16 bits.
  uint32_t rounding = 0x7FFFu + ((b >> 16) & 1u);
  return static_cast<uint16_t>((b + rounding) >> 16);
}
float h2f(uint16_t v) {
  uint32_t b = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}
float fsigmoid(float v) { return 1.0f / (1.0f + std::exp(-v)); }

uint16_t* dup(const std::vector<uint16_t>& h) {
  uint16_t* d = nullptr;
  DGPP_CUDA_OK_TEST(cudaMalloc(&d, h.size() * 2));
  DGPP_CUDA_OK_TEST(cudaMemcpy(d, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
  return d;
}
int32_t* dup32(const std::vector<int32_t>& h) {
  int32_t* d = nullptr;
  DGPP_CUDA_OK_TEST(cudaMalloc(&d, h.size() * 4));
  DGPP_CUDA_OK_TEST(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  return d;
}
int64_t* dup64(const std::vector<int64_t>& h) {
  int64_t* d = nullptr;
  DGPP_CUDA_OK_TEST(cudaMalloc(&d, h.size() * 8));
  DGPP_CUDA_OK_TEST(cudaMemcpy(d, h.data(), h.size() * 8, cudaMemcpyHostToDevice));
  return d;
}
std::vector<float> downf(const uint16_t* d, size_t n) {
  std::vector<uint16_t> h(n);
  DGPP_CUDA_OK_TEST(cudaMemcpy(h.data(), d, n * 2, cudaMemcpyDeviceToHost));
  std::vector<float> f(n);
  for (size_t i = 0; i < n; ++i) f[i] = h2f(h[i]);
  return f;
}

std::vector<uint16_t> rand_u16(int seed, size_t n, float scale) {
  uint64_t s = static_cast<uint64_t>(seed) * 0x9E3779B97F4A7C15ull + 0x12345ull;
  std::vector<uint16_t> out(n);
  for (size_t i = 0; i < n; ++i) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    float v = static_cast<float>((s >> 11) & 0x1FFFFF) / static_cast<float>(0x1FFFFF) - 0.5f;
    out[i] = f2h(v * 2.0f * scale);
  }
  return out;
}

// Dense causal attention for one query head, mirroring the kernel's FA2
// tiling rule: 16-token tiles, per-tile max, bf16-rounded P, running
// (unnormalized) denominator accumulation, single pass.
void ref_attn_head(const double* q,                       // [D]
                   const std::vector<std::vector<double>>& K,  // [n][D]
                   const std::vector<std::vector<double>>& V,  // [n][D]
                   int n, double* c) {                    // [D]
  constexpr int kTile = 16;
  double m = -1e300, l = 0.0;
  std::vector<double> acc(kD, 0.0);
  for (int t0 = 0; t0 < n; t0 += kTile) {
    const int t1 = std::min(t0 + kTile, n);
    double tile_max = -1e300;
    for (int i = t0; i < t1; ++i) {
      double s = 0.0;
      for (int d = 0; d < kD; ++d) s += q[d] * K[i][d];
      s *= kScale;
      if (s > tile_max) tile_max = s;
    }
    const double alpha = std::exp(m - tile_max);
    m = tile_max;
    l *= alpha;
    for (int d = 0; d < kD; ++d) acc[d] *= alpha;
    double tile_l = 0.0;
    for (int i = t0; i < t1; ++i) {
      double s = 0.0;
      for (int d = 0; d < kD; ++d) s += q[d] * K[i][d];
      s *= kScale;
      const float p = h2f(f2h(static_cast<float>(std::exp(s - tile_max))));
      tile_l += p;
      for (int d = 0; d < kD; ++d) acc[d] += p * V[i][d];
    }
    l += tile_l;
  }
  for (int d = 0; d < kD; ++d) acc[d] /= l;
  for (int d = 0; d < kD; ++d) c[d] = h2f(f2h(static_cast<float>(acc[d])));
}

struct Fixture {
  dgpp::CublasLtGemm gemm;
  uint16_t* ws = nullptr;
  dgpp::QwenGemmWorkspace gws;
  dgpp::QwenFullAttnResident res;
  dgpp::QwenTextConfig cfg;
  dgpp::QwenFullAttnLayer* layer = nullptr;
  dgpp::QwenFullAttnCache cache;
  cudaStream_t stream = nullptr;
  // Host weight mirrors (floats).
  std::vector<float> hq, hk, hv, ho, hqn, hkn;
  std::vector<float> inv_freq;
  // Host K/V history per head: K_hist[h][pos][d], V likewise.
  std::vector<std::vector<std::vector<double>>> Kh, Vh;

  Fixture() {
    DGPP_CUDA_OK_TEST(cudaStreamCreate(&stream));
    DGPP_CUDA_OK_TEST(cudaMalloc(&ws, 32ull << 20));
    gws.gemm = &gemm;
    gws.ws = ws;
    gws.ws_bytes = 32ull << 20;
    cfg.hidden_size = kH;
    cfg.head_dim = kD;
    cfg.rotary_dim = kRot;
    cfg.rope_theta = kTheta;
    cfg.rms_norm_eps = kEps;
    // BF16 weights, host-filled.
    auto wq = rand_u16(11, static_cast<size_t>(kQW) * kH, 0.05f);
    auto wk = rand_u16(12, static_cast<size_t>(kKW) * kH, 0.05f);
    auto wv = rand_u16(13, static_cast<size_t>(kKW) * kH, 0.05f);
    auto wo = rand_u16(14, static_cast<size_t>(kH) * (kLh * kD), 0.05f);
    auto wnq = rand_u16(15, kD, 0.1f);
    auto wnk = rand_u16(16, kD, 0.1f);
    res.q_proj = dup(wq);
    res.k_proj = dup(wk);
    res.v_proj = dup(wv);
    res.o_proj = dup(wo);
    res.q_norm = dup(wnq);
    res.k_norm = dup(wnk);
    res.local_heads = kLh;
    res.head_begin = 0;
    res.local_kv_heads = kLkv;
    res.kv_head_begin = 0;
    hq.resize(wq.size());
    hk.resize(wk.size());
    hv.resize(wv.size());
    ho.resize(wo.size());
    hqn.resize(wnq.size());
    hkn.resize(wnk.size());
    for (size_t i = 0; i < wq.size(); ++i) hq[i] = h2f(wq[i]);
    for (size_t i = 0; i < wk.size(); ++i) hk[i] = h2f(wk[i]);
    for (size_t i = 0; i < wv.size(); ++i) hv[i] = h2f(wv[i]);
    for (size_t i = 0; i < wo.size(); ++i) ho[i] = h2f(wo[i]);
    for (size_t i = 0; i < wnq.size(); ++i) hqn[i] = h2f(wnq[i]);
    for (size_t i = 0; i < wnk.size(); ++i) hkn[i] = h2f(wnk[i]);
    dgpp::qwen_ref::rope_inv_freq(kTheta, kRot, inv_freq);
    // Cache: zero planes, identity block table.
    std::vector<uint16_t> zero(static_cast<size_t>(kPoolTokens) * kKW, 0);
    cache.k_cache = dup(zero);
    cache.v_cache = dup(zero);
    cache.block_tables = dup32({0, 1});
    cache.block_tokens = kBlockTokens;
    cache.blocks_per_request = kBlocksPerReq;
    cache.max_requests = kMaxReq;
    Kh.assign(kLh, {});
    Vh.assign(kLh, {});
    layer = new dgpp::QwenFullAttnLayer(res, gws, cfg, kMaxTokens);
  }
  ~Fixture() { delete layer; }

  // Run the host reference for rows x[T,H] at positions pos[T], appending
  // to history. Returns bf16-rounded output rows [T,H] as floats.
  std::vector<float> reference(const std::vector<float>& x, const std::vector<int64_t>& pos) {
    const int T = static_cast<int>(pos.size());
    std::vector<float> out(static_cast<size_t>(T) * kH);
    std::vector<uint16_t> qh_b(kD), qb(kD), kh_b(kD);
    std::vector<uint16_t> on_b(kD), cn_b(kD);
    for (int t = 0; t < T; ++t) {
      // Projections (fp64).
      std::vector<double> qrow(kQW, 0.0), krow(kKW, 0.0), vrow(kKW, 0.0);
      for (int i = 0; i < kQW; ++i) {
        double s = 0.0;
        for (int j = 0; j < kH; ++j) s += static_cast<double>(x[static_cast<size_t>(t) * kH + j]) * hq[static_cast<size_t>(i) * kH + j];
        qrow[i] = h2f(f2h(static_cast<float>(s)));
      }
      for (int i = 0; i < kKW; ++i) {
        double sk = 0.0, sv = 0.0;
        for (int j = 0; j < kH; ++j) {
          sk += static_cast<double>(x[static_cast<size_t>(t) * kH + j]) * hk[static_cast<size_t>(i) * kH + j];
          sv += static_cast<double>(x[static_cast<size_t>(t) * kH + j]) * hv[static_cast<size_t>(i) * kH + j];
        }
        krow[i] = h2f(f2h(static_cast<float>(sk)));
        vrow[i] = h2f(f2h(static_cast<float>(sv)));
      }
      // Per-head norm+rope (oracle), attention, gate.
      std::vector<double> orow(kH, 0.0);
      for (int h = 0; h < kLh; ++h) {
        for (int d = 0; d < kD; ++d) {
          qh_b[d] = f2h(static_cast<float>(qrow[static_cast<size_t>(h) * 2 * kD + d]));
          qb[d] = f2h(static_cast<float>(qrow[static_cast<size_t>(h) * 2 * kD + kD + d]));
          kh_b[d] = f2h(static_cast<float>(krow[d]));
        }
        // qsa_norm_rope takes bf16 pointers; pass q half + q norm weights.
        std::vector<uint16_t> qnw(kD);
        for (int d = 0; d < kD; ++d) qnw[d] = f2h(hqn[d]);
        dgpp::qwen_ref::qsa_norm_rope(qh_b.data(), qnw.data(), pos[t], inv_freq.data(), on_b.data(),
                                      kD, kRot, kEps, 1.0f);
        std::vector<uint16_t> knw(kD);
        for (int d = 0; d < kD; ++d) knw[d] = f2h(hkn[d]);
        dgpp::qwen_ref::qsa_norm_rope(kh_b.data(), knw.data(), pos[t], inv_freq.data(), cn_b.data(),
                                      kD, kRot, kEps, 1.0f);
        std::vector<double> qd(kD), kd(kD), vd(kD);
        for (int d = 0; d < kD; ++d) {
          qd[d] = h2f(on_b[d]);
          kd[d] = h2f(cn_b[d]);
          vd[d] = h2f(f2h(static_cast<float>(vrow[d])));
        }
        if (t == 0 && h == 0)
          std::printf("  [trace] x0=%.6g qrow0=%.6g on_b0=%u kd0=%.6g vd0=%.6g ho0=%.6g\n", x[0],
                      qrow[0], on_b[0], kd[0], vd[0], ho[0]);
        Kh[h].push_back(kd);
        Vh[h].push_back(vd);
        std::vector<double> c(kD, 0.0);
        ref_attn_head(qd.data(), Kh[h], Vh[h], static_cast<int>(Kh[h].size()), c.data());
        for (int d = 0; d < kD; ++d) {
          const float g = h2f(qb[d]);
          const float gb = h2f(f2h(fsigmoid(g)));
          const float cb = h2f(f2h(static_cast<float>(c[d])));
          orow[static_cast<size_t>(h) * kD + d] = h2f(f2h(cb * gb));
        }
      }
      // o_proj (fp64) + bf16 round.
      for (int j = 0; j < kH; ++j) {
        double s = 0.0;
        for (int i = 0; i < kH; ++i) s += orow[i] * ho[static_cast<size_t>(j) * kH + i];
        out[static_cast<size_t>(t) * kH + j] = h2f(f2h(static_cast<float>(s)));
      }
    }
    return out;
  }
};

void check_close(const char* name, const std::vector<float>& got, const std::vector<float>& ref) {
  double rms = 0.0;
  for (float v : ref) rms += static_cast<double>(v) * v;
  rms = std::sqrt(rms / static_cast<double>(ref.size()));
  std::printf("  [debug] ref[0..3] = %.6g %.6g %.6g %.6g\n", ref[0], ref[1], ref[2], ref[3]);
  std::printf("  [debug] got[0..3] = %.6g %.6g %.6g %.6g\n", got[0], got[1], got[2], got[3]);
  const Stats s = compare_abs_rel(got.data(), ref.data(), static_cast<long>(got.size()), 2 * std::pow(2.0, -7.0),
                                  0.005 * rms);
  std::printf("[ .. ] %s: max_abs %.3g l2_rel %.3g mismatches %ld/%ld (rms %.3g)\n", name, s.max_abs, s.l2_rel,
              s.mismatches, s.n, rms);
  require_bf16(name, s, 3e-3, 0.01);
}

DGPP_TEST(full_layer_prefill_matches_reference) {
  Fixture fx;
  const int T = 4;
  std::vector<float> xh(static_cast<size_t>(T) * kH);
  {
    auto xb = rand_u16(21, xh.size(), 1.0f);
    for (size_t i = 0; i < xh.size(); ++i) xh[i] = h2f(xb[i]);
  }
  uint16_t* dx = dup([&] {
    std::vector<uint16_t> b(xh.size());
    for (size_t i = 0; i < xh.size(); ++i) b[i] = f2h(xh[i]);
    return b;
  }());
  std::vector<int32_t> req_ids(T, 0);
  std::vector<int64_t> pos = {0, 1, 2, 3};
  int32_t* d_req = dup32(req_ids);
  int64_t* d_pos = dup64(pos);
  dgpp::QwenQsaRows rows;
  rows.req_ids = d_req;
  rows.pos = d_pos;
  rows.decode = false;
  rows.request = 0;
  rows.pos0 = 0;
  std::vector<uint16_t> out_h(static_cast<size_t>(T) * kH, 0);
  uint16_t* d_out = dup(out_h);
  fx.layer->enqueue(dx, T, rows, fx.cache, d_out, fx.stream);
  DGPP_CUDA_OK_TEST(cudaStreamSynchronize(fx.stream));
  const std::vector<float> ref = fx.reference(xh, pos);
  check_close("full layer prefill", downf(d_out, out_h.size()), ref);
}

DGPP_TEST(full_layer_decode_matches_reference) {
  Fixture fx;
  // Prefill 4 rows first (drives the device cache + host history).
  std::vector<float> xh(static_cast<size_t>(4) * kH);
  {
    auto xb = rand_u16(21, xh.size(), 1.0f);
    for (size_t i = 0; i < xh.size(); ++i) xh[i] = h2f(xb[i]);
  }
  uint16_t* dx = dup([&] {
    std::vector<uint16_t> b(xh.size());
    for (size_t i = 0; i < xh.size(); ++i) b[i] = f2h(xh[i]);
    return b;
  }());
  std::vector<int32_t> req4(4, 0);
  std::vector<int64_t> pos4 = {0, 1, 2, 3};
  int32_t* d_req4 = dup32(req4);
  int64_t* d_pos4 = dup64(pos4);
  dgpp::QwenQsaRows rows4;
  rows4.req_ids = d_req4;
  rows4.pos = d_pos4;
  rows4.decode = false;
  rows4.request = 0;
  rows4.pos0 = 0;
  std::vector<uint16_t> out4(static_cast<size_t>(4) * kH, 0);
  uint16_t* d_out4 = dup(out4);
  fx.layer->enqueue(dx, 4, rows4, fx.cache, d_out4, fx.stream);
  DGPP_CUDA_OK_TEST(cudaStreamSynchronize(fx.stream));
  (void)fx.reference(xh, pos4);
  // Decode row at pos 4.
  std::vector<float> xd(kH);
  {
    auto xb = rand_u16(22, xd.size(), 1.0f);
    for (size_t i = 0; i < xd.size(); ++i) xd[i] = h2f(xb[i]);
  }
  uint16_t* dxd = dup([&] {
    std::vector<uint16_t> b(xd.size());
    for (size_t i = 0; i < xd.size(); ++i) b[i] = f2h(xd[i]);
    return b;
  }());
  std::vector<int32_t> req1 = {0};
  std::vector<int64_t> pos1 = {4};
  int32_t* d_req1 = dup32(req1);
  int64_t* d_pos1 = dup64(pos1);
  dgpp::QwenQsaRows rows1;
  rows1.req_ids = d_req1;
  rows1.pos = d_pos1;
  rows1.decode = true;
  rows1.num_requests = 1;
  rows1.request = 0;
  rows1.pos0 = 4;
  std::vector<uint16_t> out1(kH, 0);
  uint16_t* d_out1 = dup(out1);
  fx.layer->enqueue(dxd, 1, rows1, fx.cache, d_out1, fx.stream);
  DGPP_CUDA_OK_TEST(cudaStreamSynchronize(fx.stream));
  const std::vector<float> ref = fx.reference(xd, pos1);
  check_close("full layer decode", downf(d_out1, out1.size()), ref);
}

DGPP_TEST(full_layer_second_prefill_is_bitwise) {
  Fixture fx;
  const int T = 4;
  std::vector<float> xh(static_cast<size_t>(T) * kH);
  {
    auto xb = rand_u16(21, xh.size(), 1.0f);
    for (size_t i = 0; i < xh.size(); ++i) xh[i] = h2f(xb[i]);
  }
  uint16_t* dx = dup([&] {
    std::vector<uint16_t> b(xh.size());
    for (size_t i = 0; i < xh.size(); ++i) b[i] = f2h(xh[i]);
    return b;
  }());
  auto run = [&] {
    // Fresh cache each run (append state must not leak across runs).
    std::vector<uint16_t> zero(static_cast<size_t>(kPoolTokens) * kKW, 0);
    uint16_t* kc = dup(zero);
    uint16_t* vc = dup(zero);
    int32_t* tbl = dup32({0, 1});
    dgpp::QwenFullAttnCache c;
    c.k_cache = kc;
    c.v_cache = vc;
    c.block_tables = tbl;
    c.block_tokens = kBlockTokens;
    c.blocks_per_request = kBlocksPerReq;
    c.max_requests = kMaxReq;
    std::vector<int32_t> req_ids(T, 0);
    std::vector<int64_t> pos = {0, 1, 2, 3};
    int32_t* d_req = dup32(req_ids);
    int64_t* d_pos = dup64(pos);
    dgpp::QwenQsaRows rows;
    rows.req_ids = d_req;
    rows.pos = d_pos;
    rows.decode = false;
    rows.request = 0;
    rows.pos0 = 0;
    std::vector<uint16_t> out_h(static_cast<size_t>(T) * kH, 0);
    uint16_t* d_out = dup(out_h);
    fx.layer->enqueue(dx, T, rows, c, d_out, fx.stream);
    DGPP_CUDA_OK_TEST(cudaStreamSynchronize(fx.stream));
    std::vector<uint16_t> got(out_h.size());
    DGPP_CUDA_OK_TEST(cudaMemcpy(got.data(), d_out, out_h.size() * 2, cudaMemcpyDeviceToHost));
    return got;
  };
  const std::vector<uint16_t> first = run();
  require(run() == first, "second prefill bitwise the first");
}

}  // namespace

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;
  return dgpp::test::run_all();
}
