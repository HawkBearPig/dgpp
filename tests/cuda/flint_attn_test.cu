// The sink-attention kernels' qk-128 instantiation (kernels/sink_attn.hpp,
// Flint) against the host reference on small geometry: the qkv finish
// over the fused projection's fp32 output — the per-head one-rounding
// RMSNorm, the full-head RoPE with fp32 cos / sin (and the no-RoPE form),
// the paged K/V append — within two bf16 ulps of the reference for q/k
// (the device's cosf / sinf / rsqrtf against the host's) and bitwise for
// v; the split-KV attention with the 513-token window (not a tile
// multiple), sinks and splits against the reference modelling the
// kernel's chain; the fused decode launch bitwise the three-kernel chain
// (bf16 and fp8 caches, RoPE and no-RoPE layers, the batch's own rows
// overlaid); the query-tiled prefill against the split-KV chain; the fp8
// cache bitwise the bf16 kernel over the dequantized rows.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/latent_format.hpp"
#include "kernels/qsa.hpp"
#include "kernels/sink_attn.hpp"
#include "models/flint/attn_reference.hpp"

using namespace dgpp::kda_test;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

constexpr int D = dgpp::flint_ref::kDim;  // 128: q, k and v
constexpr int QK = 128;

template <class T>
DevBuf up(const std::vector<T>& v) {
  DevBuf b(v.size() * sizeof(T));
  b.upload(v.data(), v.size() * sizeof(T));
  return b;
}
template <class T>
std::vector<T> down(const DevBuf& b, size_t n) {
  std::vector<T> v(n);
  b.download(v.data(), n * sizeof(T));
  return v;
}
template <class T>
const T* ptr(const DevBuf& b) { return static_cast<const T*>(b.p); }
template <class T>
T* mptr(DevBuf& b) { return static_cast<T*>(b.p); }

int bf16_ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) -> int32_t {
    return (v & 0x8000u) ? -static_cast<int32_t>(v & 0x7FFFu) : static_cast<int32_t>(v & 0x7FFFu);
  };
  return std::abs(static_cast<int>(key(a) - key(b)));
}

std::vector<float> inv_freq_host(double theta) {
  std::vector<float> f(static_cast<size_t>(D / 2));
  dgpp::qsa_rope_inv_freq(theta, D, f.data());
  return f;
}

// Norm weights near 1 (the release's q_norm / k_norm are gains).
std::vector<uint16_t> norm_weights(uint64_t seed) {
  std::vector<uint16_t> w(static_cast<size_t>(D));
  uint64_t s = seed;
  for (int d = 0; d < D; ++d) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    w[static_cast<size_t>(d)] = dgpp::float_to_bf16_bits(0.8f + 0.4f * static_cast<float>(s % 1000) / 1000.f);
  }
  return w;
}

struct Pool {
  int block_tokens = 64, blocks_per_request = 12, max_requests = 2;
  std::vector<int32_t> table;
  Pool() {
    const int n = max_requests * blocks_per_request;
    table.resize(static_cast<size_t>(n));
    std::iota(table.begin(), table.end(), 0);
    uint64_t s = 0x1234567ull;
    for (int i = n - 1; i > 0; --i) {
      s ^= s << 13; s ^= s >> 7; s ^= s << 17;
      std::swap(table[static_cast<size_t>(i)], table[static_cast<size_t>(s % static_cast<uint64_t>(i + 1))]);
    }
  }
  int slots() const { return block_tokens * blocks_per_request * max_requests; }
  int64_t phys(int req, int64_t pos) const {
    const int32_t blk = table[static_cast<size_t>(req) * blocks_per_request + pos / block_tokens];
    return static_cast<int64_t>(blk) * block_tokens + pos % block_tokens;
  }
};

// The host codec of kernels/latent_format.hpp over one head row.
void host_fp8_row(const uint16_t* row, int dim, uint8_t* codes, float* scale) {
  float amax = 0.f;
  for (int d = 0; d < dim; ++d) amax = std::max(amax, std::fabs(dgpp::bf16_bits_to_float(row[d])));
  const dgpp::LatentFp8Scale s = dgpp::latent_fp8_row_scale(amax);
  for (int d = 0; d < dim; ++d) codes[d] = dgpp::latent_fp8_encode(dgpp::bf16_bits_to_float(row[d]), s.inv);
  *scale = s.scale;
}

}  // namespace

DGPP_TEST(flint_qkv_finish_matches_the_reference) {
  cudaStream_t st = test_stream();
  const Pool pool;
  const int rows = 9;
  const std::vector<int32_t> req_ids = {0, 0, 1, 1, 0, 1, 0, 1, 1};
  const std::vector<int64_t> pos = {0, 1, 17, 63, 64, 200, 700, -1, 5};
  DevBuf dreq = up(req_ids), dpos = up(pos), dtbl = up(pool.table);
  const std::vector<float> inv = inv_freq_host(1e4);
  DevBuf dinv = up(inv);
  const std::vector<uint16_t> qn = norm_weights(101), kn = norm_weights(102);
  DevBuf dqn = up(qn), dkn = up(kn);
  const float eps = 1e-6f;
  long checked = 0;
  int max_ulps = 0;
  for (const int kvpc : {1, 2}) {
    for (const bool rope : {true, false}) {
      for (const bool norm : {true, false}) {
        dgpp::SinkQkvLayout layout;
        layout.chunks = 1;
        layout.q_per_chunk = 4 * kvpc;
        layout.kv_per_chunk = kvpc;
        const int64_t chunk_rows = static_cast<int64_t>(layout.q_per_chunk) * QK + kvpc * (QK + D);
        layout.chunk_stride = chunk_rows;
        const int lh = layout.q_per_chunk, lkv = kvpc;
        const int64_t qkv_stride = chunk_rows;
        std::vector<float> qkv = random_f32_uniform(11 + kvpc, static_cast<int64_t>(rows) * qkv_stride, 2.0f);
        DevBuf dqkv = up(qkv);
        const int qw = lh * QK, kw = lkv * QK, vw = lkv * D;
        const size_t kbytes = static_cast<size_t>(pool.slots()) * kw * 2, vbytes = static_cast<size_t>(pool.slots()) * vw * 2;
        DevBuf kc(kbytes), vc(vbytes), qo(static_cast<size_t>(rows) * qw * 2);
        DGPP_CUDA_OK(cudaMemset(kc.p, 0x7F, kbytes));
        DGPP_CUDA_OK(cudaMemset(vc.p, 0x7F, vbytes));
        DGPP_CUDA_OK(cudaMemset(qo.p, 0x7F, static_cast<size_t>(rows) * qw * 2));
        dgpp::SinkFinishParams fin;
        fin.inv_freq = rope ? ptr<float>(dinv) : nullptr;
        fin.q_norm = norm ? ptr<uint16_t>(dqn) : nullptr;
        fin.k_norm = norm ? ptr<uint16_t>(dkn) : nullptr;
        fin.eps = eps;
        auto run = [&] {
          dgpp::sink_qkv_finish(QK, ptr<float>(dqkv), qkv_stride, layout, fin, ptr<int32_t>(dreq), ptr<int64_t>(dpos), rows,
                                lh, lkv, ptr<int32_t>(dtbl), pool.blocks_per_request, pool.block_tokens, mptr<uint16_t>(qo),
                                qw, mptr<uint16_t>(kc), mptr<uint16_t>(vc), st);
          DGPP_CUDA_OK(cudaStreamSynchronize(st));
        };
        run();
        const std::vector<uint16_t> got_q = down<uint16_t>(qo, static_cast<size_t>(rows) * qw);
        const std::vector<uint16_t> got_k = down<uint16_t>(kc, kbytes / 2), got_v = down<uint16_t>(vc, vbytes / 2);
        std::vector<uint16_t> ref(static_cast<size_t>(QK));
        for (int r = 0; r < rows; ++r) {
          const int64_t p = pos[static_cast<size_t>(r)];
          const float* src_row = qkv.data() + static_cast<size_t>(r) * qkv_stride;
          for (int h = 0; h < lh; ++h) {
            const uint16_t* got = got_q.data() + static_cast<size_t>(r) * qw + h * QK;
            if (p < 0) {
              for (int d = 0; d < QK; ++d) require(got[d] == 0x7F7F, "a padding row writes no q");
              continue;
            }
            dgpp::flint_ref::qk_finish(src_row + static_cast<int64_t>(h) * QK, norm ? qn.data() : nullptr, eps,
                                       rope ? inv.data() : nullptr, p, ref.data());
            for (int d = 0; d < QK; ++d) {
              max_ulps = std::max(max_ulps, bf16_ulps(got[d], ref[static_cast<size_t>(d)]));
              ++checked;
            }
          }
          if (p < 0) continue;
          const int64_t slot = pool.phys(req_ids[static_cast<size_t>(r)], p);
          for (int j = 0; j < lkv; ++j) {
            const float* ksrc = src_row + static_cast<int64_t>(layout.q_per_chunk) * QK + static_cast<int64_t>(j) * QK;
            dgpp::flint_ref::qk_finish(ksrc, norm ? kn.data() : nullptr, eps, rope ? inv.data() : nullptr, p, ref.data());
            const uint16_t* gk = got_k.data() + slot * kw + j * QK;
            for (int d = 0; d < QK; ++d) max_ulps = std::max(max_ulps, bf16_ulps(gk[d], ref[static_cast<size_t>(d)]));
            const float* vsrc = src_row + static_cast<int64_t>(layout.q_per_chunk + lkv) * QK + static_cast<int64_t>(j) * D;
            dgpp::flint_ref::v_finish(vsrc, ref.data());
            const uint16_t* gv = got_v.data() + slot * vw + j * D;
            for (int d = 0; d < D; ++d) require(gv[d] == ref[static_cast<size_t>(d)], "v: bf16(dot) bitwise");
            checked += QK + D;
          }
        }
        std::vector<bool> written(static_cast<size_t>(pool.slots()), false);
        for (int r = 0; r < rows; ++r)
          if (pos[static_cast<size_t>(r)] >= 0)
            written[static_cast<size_t>(pool.phys(req_ids[static_cast<size_t>(r)], pos[static_cast<size_t>(r)]))] = true;
        for (int s = 0; s < pool.slots(); ++s) {
          if (written[static_cast<size_t>(s)]) continue;
          for (int i = 0; i < kw; ++i) require(got_k[static_cast<size_t>(s) * kw + i] == 0x7F7F, "untouched K slots keep their poison");
          for (int i = 0; i < vw; ++i) require(got_v[static_cast<size_t>(s) * vw + i] == 0x7F7F, "untouched V slots keep their poison");
        }
        run();
        require(down<uint16_t>(qo, got_q.size()) == got_q && down<uint16_t>(kc, got_k.size()) == got_k &&
                    down<uint16_t>(vc, got_v.size()) == got_v,
                "second run bitwise");
      }
    }
  }
  std::printf("[ .. ] qkv finish (qk 128): %ld elements, max %d bf16 ulps\n", checked, max_ulps);
  require(max_ulps <= 2, "qkv finish within two bf16 ulps of the reference");
}

namespace {

struct Seq {
  int req = 1, len = 700;
  std::vector<uint16_t> k, v;
};

void fill_cache(const Pool& pool, const Seq& seq, int kv, std::vector<uint16_t>& kh, std::vector<uint16_t>& vh) {
  const int kw = kv * QK, vw = kv * D;
  kh.assign(static_cast<size_t>(pool.slots()) * kw, 0x7F7F);
  vh.assign(static_cast<size_t>(pool.slots()) * vw, 0x7F7F);
  for (int t = 0; t < seq.len; ++t) {
    const int64_t s = pool.phys(seq.req, t);
    std::copy(seq.k.begin() + static_cast<size_t>(t) * kw, seq.k.begin() + static_cast<size_t>(t + 1) * kw, kh.begin() + s * kw);
    std::copy(seq.v.begin() + static_cast<size_t>(t) * vw, seq.v.begin() + static_cast<size_t>(t + 1) * vw, vh.begin() + s * vw);
  }
}

}  // namespace

DGPP_TEST(flint_attention_matches_the_reference_windows_sinks_and_splits) {
  cudaStream_t st = test_stream();
  const Pool pool;
  const int kv = 2;
  Seq seq;
  seq.k = random_bf16_normal(31, static_cast<int64_t>(seq.len) * kv * QK, 1.0f);
  seq.v = random_bf16_normal(32, static_cast<int64_t>(seq.len) * kv * D, 1.0f);
  std::vector<uint16_t> kh, vh;
  fill_cache(pool, seq, kv, kh, vh);
  DevBuf kc = up(kh), vc = up(vh), dtbl = up(pool.table);
  // Positions across tile and block boundaries, past the 513 window, a padding row.
  const std::vector<int64_t> pos = {0, 5, 31, 32, 63, 64, 200, 512, 513, 600, 699, -1};
  const int rows = static_cast<int>(pos.size());
  const std::vector<int32_t> req_ids(static_cast<size_t>(rows), seq.req);
  DevBuf dpos = up(pos), dreq = up(req_ids);
  const float scale = ref_scale(QK);
  for (const int hpk : {1, 8}) {
    const int lh = hpk * kv, qw = lh * QK, ow = lh * D;
    const std::vector<uint16_t> q = random_bf16_normal(33 + hpk, static_cast<int64_t>(rows) * qw, 1.0f);
    const std::vector<float> sink = random_f32_uniform(77, lh, 2.0f);
    DevBuf dq = up(q), dsink = up(sink);
    for (const int window : {0, 32, 513}) {
      for (const bool with_sink : {false, true}) {
        for (const int n_split : {1, 4, 17, 32}) {
          const std::string tag = "hpk " + std::to_string(hpk) + " window " + std::to_string(window) + " sink " +
                                  std::to_string(with_sink) + " n_split " + std::to_string(n_split);
          std::vector<uint16_t> ref(static_cast<size_t>(rows) * ow, 0), plain(static_cast<size_t>(rows) * ow, 0);
          for (int r = 0; r < rows; ++r) {
            const int64_t p = pos[static_cast<size_t>(r)];
            if (p < 0) continue;
            const int64_t lo = window > 0 ? std::max<int64_t>(0, p - window + 1) : 0;
            const int n = static_cast<int>(p - lo + 1);
            const uint16_t* krows = seq.k.data() + static_cast<size_t>(lo) * kv * QK;
            const uint16_t* vrows = seq.v.data() + static_cast<size_t>(lo) * kv * D;
            std::vector<float> c;
            dgpp::flint_ref::attention(q.data() + static_cast<size_t>(r) * qw, krows, vrows, n, lh, kv, scale,
                                       with_sink ? sink.data() : nullptr, c, dgpp::kSinkAttnTile, n_split);
            for (int i = 0; i < ow; ++i) ref[static_cast<size_t>(r) * ow + i] = dgpp::float_to_bf16_bits(c[static_cast<size_t>(i)]);
            dgpp::flint_ref::attention(q.data() + static_cast<size_t>(r) * qw, krows, vrows, n, lh, kv, scale,
                                       with_sink ? sink.data() : nullptr, c, 0, 1);
            for (int i = 0; i < ow; ++i) plain[static_cast<size_t>(r) * ow + i] = dgpp::float_to_bf16_bits(c[static_cast<size_t>(i)]);
          }
          const size_t part = static_cast<size_t>(rows) * n_split * lh;
          DevBuf m_ws(part * 4), l_ws(part * 4), c_ws(part * D * 4), out(ref.size() * 2);
          DGPP_CUDA_OK(cudaMemset(out.p, 0x7F, ref.size() * 2));
          dgpp::sink_attn_partial(QK, ptr<uint16_t>(dq), qw, ptr<uint16_t>(kc), ptr<uint16_t>(vc), ptr<int32_t>(dreq),
                                  ptr<int64_t>(dpos), rows, n_split, lh, kv, pool.block_tokens, ptr<int32_t>(dtbl),
                                  pool.blocks_per_request, window, scale, with_sink ? ptr<float>(dsink) : nullptr,
                                  mptr<float>(m_ws), mptr<float>(l_ws), mptr<float>(c_ws), st);
          dgpp::sink_attn_combine(ptr<float>(m_ws), ptr<float>(l_ws), ptr<float>(c_ws), rows, n_split, lh, mptr<uint16_t>(out), st);
          DGPP_CUDA_OK(cudaStreamSynchronize(st));
          const std::vector<uint16_t> got = down<uint16_t>(out, ref.size());
          std::vector<float> gf(got.size()), wf(ref.size()), pf(plain.size());
          double rms = 0;
          for (size_t i = 0; i < got.size(); ++i) {
            gf[i] = dgpp::bf16_bits_to_float(got[i]);
            wf[i] = dgpp::bf16_bits_to_float(ref[i]);
            pf[i] = dgpp::bf16_bits_to_float(plain[i]);
            rms += static_cast<double>(wf[i]) * wf[i];
          }
          rms = std::sqrt(rms / static_cast<double>(ref.size()));
          const Stats s = compare_abs_rel(gf.data(), wf.data(), static_cast<long>(got.size()), 2 * std::pow(2.0, -7.0), 0.005 * rms);
          const Stats sp = compare_abs_rel(gf.data(), pf.data(), static_cast<long>(got.size()), 6 * std::pow(2.0, -7.0), 0.01 * rms);
          std::printf("[ .. ] attention %s: chain max_abs %.3g l2_rel %.3g mismatches %ld/%ld; plain l2_rel %.3g mismatches %ld (rms %.3g)\n",
                      tag.c_str(), s.max_abs, s.l2_rel, s.mismatches, s.n, sp.l2_rel, sp.mismatches, rms);
          require_bf16("attention chain " + tag, s, 2e-3, 0.01);
          require_bf16("attention plain " + tag, sp, 1e-2, 0.02);
          for (int i = 0; i < ow; ++i) require(got[static_cast<size_t>(rows - 1) * ow + i] == 0, "padding row is zero");
        }
      }
    }
  }
}

DGPP_TEST(flint_window_513_sees_exactly_the_row_and_the_512_before_it) {
  // The semantics pin: at position 600 with window 513 the rows before 88
  // are invisible and row 88 is visible — against the kernel's own run.
  cudaStream_t st = test_stream();
  const Pool pool;
  const int kv = 1, hpk = 8, lh = hpk * kv, qw = lh * QK, ow = lh * D;
  Seq seq;
  seq.k = random_bf16_normal(41, static_cast<int64_t>(seq.len) * kv * QK, 1.0f);
  seq.v = random_bf16_normal(42, static_cast<int64_t>(seq.len) * kv * D, 1.0f);
  std::vector<uint16_t> kh, vh;
  fill_cache(pool, seq, kv, kh, vh);
  DevBuf dtbl = up(pool.table);
  const std::vector<int64_t> pos = {600};
  const std::vector<int32_t> req_ids = {seq.req};
  const std::vector<uint16_t> q = random_bf16_normal(43, qw, 1.0f);
  DevBuf dq = up(q), dpos = up(pos), dreq = up(req_ids);
  auto run = [&](const std::vector<uint16_t>& k_host) {
    DevBuf kc = up(k_host), vc = up(vh);
    const int n_split = 17;
    const size_t part = static_cast<size_t>(n_split) * lh;
    DevBuf m_ws(part * 4), l_ws(part * 4), c_ws(part * D * 4), out(static_cast<size_t>(ow) * 2);
    dgpp::sink_attn_partial(QK, ptr<uint16_t>(dq), qw, ptr<uint16_t>(kc), ptr<uint16_t>(vc), ptr<int32_t>(dreq),
                            ptr<int64_t>(dpos), 1, n_split, lh, kv, pool.block_tokens, ptr<int32_t>(dtbl),
                            pool.blocks_per_request, 513, ref_scale(QK), nullptr, mptr<float>(m_ws), mptr<float>(l_ws),
                            mptr<float>(c_ws), st);
    dgpp::sink_attn_combine(ptr<float>(m_ws), ptr<float>(l_ws), ptr<float>(c_ws), 1, n_split, lh, mptr<uint16_t>(out), st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    return down<uint16_t>(out, static_cast<size_t>(ow));
  };
  const std::vector<uint16_t> base = run(kh);
  std::vector<uint16_t> k87 = kh, k88 = kh;
  for (int i = 0; i < kv * QK; ++i) {
    k87[static_cast<size_t>(pool.phys(seq.req, 87)) * kv * QK + i] ^= 0x4000;
    k88[static_cast<size_t>(pool.phys(seq.req, 88)) * kv * QK + i] ^= 0x4000;
  }
  require(run(k87) == base, "position 87 is outside the 513 window of position 600");
  require(run(k88) != base, "position 88 is the first visible position of 600's window");
  std::printf("[ OK ] window 513: [pos - 512, pos]\n");
}

// ---- the fused decode attention: bitwise the three-kernel chain -------------------
DGPP_TEST(flint_fused_decode_attention_is_the_three_kernel_chain) {
  cudaStream_t st = test_stream();
  const Pool pool;
  const std::vector<float> inv = inv_freq_host(1e4);
  DevBuf dinv = up(inv), dtbl = up(pool.table);
  const std::vector<uint16_t> qn = norm_weights(201), kn = norm_weights(202);
  DevBuf dqn = up(qn), dkn = up(kn);
  // The batch: request 1 at 598..600 (three rows), request 0 at 69 and 70,
  // a padding row, request 1 at 4 (its own earlier context only).
  const std::vector<int32_t> req_ids = {1, 1, 1, 0, 0, 1, 0};
  const std::vector<int64_t> pos = {598, 599, 600, 69, 70, -1, 4};
  const int rows = static_cast<int>(pos.size());
  const int64_t first_batch_pos[2] = {69, 598};
  DevBuf dreq = up(req_ids), dpos = up(pos);
  long checked = 0;
  for (const int kvpc : {1, 2}) {
    dgpp::SinkQkvLayout layout;
    layout.chunks = 1;
    layout.q_per_chunk = 8 * kvpc;
    layout.kv_per_chunk = kvpc;
    const int64_t chunk_rows = static_cast<int64_t>(layout.q_per_chunk) * QK + kvpc * (QK + D);
    layout.chunk_stride = chunk_rows;
    const int lh = layout.q_per_chunk, kv = kvpc;
    const int64_t qkv_stride = chunk_rows;
    const int kw = kv * QK, vw = kv * D, ow = lh * D, qw = lh * QK;
    const std::vector<float> qkv = random_f32_uniform(400 + kvpc, static_cast<int64_t>(rows) * qkv_stride, 2.0f);
    const std::vector<float> sink = random_f32_uniform(410 + kvpc, lh, 2.0f);
    DevBuf dqkv = up(qkv), dsink = up(sink);
    for (const bool fp8 : {false, true}) {
      const size_t slots = static_cast<size_t>(pool.slots());
      std::vector<uint16_t> kh(slots * kw, 0), vh(slots * vw, 0);
      std::vector<uint8_t> k8(slots * kw, 0), v8(slots * vw, 0);
      std::vector<float> ks(slots * kv, 0.f), vs(slots * kv, 0.f);
      std::vector<uint8_t> codes(static_cast<size_t>(QK));
      for (int r = 0; r < 2; ++r) {
        const std::vector<uint16_t> k = random_bf16_normal(420 + r, first_batch_pos[r] * kw, 1.0f);
        const std::vector<uint16_t> v = random_bf16_normal(430 + r, first_batch_pos[r] * vw, 1.0f);
        for (int64_t t = 0; t < first_batch_pos[r]; ++t) {
          const size_t s = static_cast<size_t>(pool.phys(r, t));
          std::copy(k.begin() + t * kw, k.begin() + (t + 1) * kw, kh.begin() + s * kw);
          std::copy(v.begin() + t * vw, v.begin() + (t + 1) * vw, vh.begin() + s * vw);
          for (int j = 0; j < kv; ++j) {
            host_fp8_row(kh.data() + s * kw + j * QK, QK, codes.data(), &ks[s * kv + j]);
            std::copy(codes.begin(), codes.begin() + QK, k8.begin() + s * kw + j * QK);
            host_fp8_row(vh.data() + s * vw + j * D, D, codes.data(), &vs[s * kv + j]);
            std::copy(codes.begin(), codes.begin() + D, v8.begin() + s * vw + j * D);
          }
        }
      }
      for (const bool swa : {false, true}) {
        const int window = swa ? 513 : 0;
        const int n_split = swa ? 17 : 32;
        const float* inv_p = swa ? ptr<float>(dinv) : nullptr;  // the full layers: no RoPE
        const size_t part = static_cast<size_t>(rows) * n_split * lh;
        struct Run {
          DevBuf kc, vc, ksc, vsc, m_ws, l_ws, c_ws, out;
        };
        auto make = [&]() {
          Run x;
          x.kc = fp8 ? up(k8) : up(kh);
          x.vc = fp8 ? up(v8) : up(vh);
          x.ksc = up(ks);
          x.vsc = up(vs);
          x.m_ws = DevBuf(part * 4);
          x.l_ws = DevBuf(part * 4);
          x.c_ws = DevBuf(part * D * 4);
          x.out = DevBuf(static_cast<size_t>(rows) * ow * 2);
          DGPP_CUDA_OK(cudaMemset(x.out.p, 0x7F, x.out.bytes));
          return x;
        };
        Run chain = make(), fused = make();
        dgpp::SinkFinishParams fin;
        fin.inv_freq = inv_p;
        fin.q_norm = ptr<uint16_t>(dqn);
        fin.k_norm = ptr<uint16_t>(dkn);
        {
          DevBuf q(static_cast<size_t>(rows) * qw * 2);
          dgpp::sink_qkv_finish(QK, ptr<float>(dqkv), qkv_stride, layout, fin, ptr<int32_t>(dreq), ptr<int64_t>(dpos), rows, lh,
                                kv, ptr<int32_t>(dtbl), pool.blocks_per_request, pool.block_tokens, mptr<uint16_t>(q), qw,
                                mptr<uint16_t>(chain.kc), mptr<uint16_t>(chain.vc), st, fp8 ? mptr<float>(chain.ksc) : nullptr,
                                fp8 ? mptr<float>(chain.vsc) : nullptr);
          dgpp::sink_attn_partial(QK, ptr<uint16_t>(q), qw, ptr<uint16_t>(chain.kc), ptr<uint16_t>(chain.vc), ptr<int32_t>(dreq),
                                  ptr<int64_t>(dpos), rows, n_split, lh, kv, pool.block_tokens, ptr<int32_t>(dtbl),
                                  pool.blocks_per_request, window, ref_scale(QK), ptr<float>(dsink), mptr<float>(chain.m_ws),
                                  mptr<float>(chain.l_ws), mptr<float>(chain.c_ws), st, fp8 ? ptr<float>(chain.ksc) : nullptr,
                                  fp8 ? ptr<float>(chain.vsc) : nullptr);
          dgpp::sink_attn_combine(ptr<float>(chain.m_ws), ptr<float>(chain.l_ws), ptr<float>(chain.c_ws), rows, n_split, lh,
                                  mptr<uint16_t>(chain.out), st);
          DGPP_CUDA_OK(cudaStreamSynchronize(st));
        }
        {
          DevBuf counters(static_cast<size_t>(rows) * kv * 4);
          DGPP_CUDA_OK(cudaMemset(counters.p, 0, counters.bytes));
          dgpp::SinkAttnFusedArgs a;
          a.qkv = ptr<float>(dqkv);
          a.qkv_stride = qkv_stride;
          a.layout = layout;
          a.inv_freq = inv_p;
          a.q_norm = ptr<uint16_t>(dqn);
          a.k_norm = ptr<uint16_t>(dkn);
          a.req_ids = ptr<int32_t>(dreq);
          a.pos = ptr<int64_t>(dpos);
          a.rows = rows;
          a.n_split = n_split;
          a.local_heads = lh;
          a.kv_heads = kv;
          a.block_tables = ptr<int32_t>(dtbl);
          a.blocks_per_request = pool.blocks_per_request;
          a.block_tokens = pool.block_tokens;
          a.window = window;
          a.scale = ref_scale(QK);
          a.sink = ptr<float>(dsink);
          a.k_cache = mptr<uint16_t>(fused.kc);
          a.v_cache = mptr<uint16_t>(fused.vc);
          a.k_scale = fp8 ? mptr<float>(fused.ksc) : nullptr;
          a.v_scale = fp8 ? mptr<float>(fused.vsc) : nullptr;
          a.m_ws = mptr<float>(fused.m_ws);
          a.l_ws = mptr<float>(fused.l_ws);
          a.c_ws = mptr<float>(fused.c_ws);
          a.counters = static_cast<int*>(counters.p);
          a.out = mptr<uint16_t>(fused.out);
          dgpp::sink_attn_fused(QK, a, st);
          dgpp::sink_attn_fused(QK, a, st);
          DGPP_CUDA_OK(cudaStreamSynchronize(st));
          const auto cnt = down<int32_t>(counters, static_cast<size_t>(rows) * kv);
          for (int c : cnt) require(c == 0, "the fused kernel leaves its counters zero");
        }
        const std::string tag = std::string(fp8 ? "fp8" : "bf16") + (swa ? " swa+rope" : " full+nope") + " kvpc " + std::to_string(kvpc);
        const auto co = down<uint16_t>(chain.out, static_cast<size_t>(rows) * ow);
        const auto fo = down<uint16_t>(fused.out, static_cast<size_t>(rows) * ow);
        require(co == fo, "fused output bitwise the chain's (" + tag + ")");
        const size_t kbytes = fp8 ? slots * kw : slots * kw * 2, vbytes = fp8 ? slots * vw : slots * vw * 2;
        require(down<uint8_t>(chain.kc, kbytes) == down<uint8_t>(fused.kc, kbytes), "K cache bitwise (" + tag + ")");
        require(down<uint8_t>(chain.vc, vbytes) == down<uint8_t>(fused.vc, vbytes), "V cache bitwise (" + tag + ")");
        if (fp8) {
          require(down<float>(chain.ksc, slots * kv) == down<float>(fused.ksc, slots * kv), "K scales bitwise (" + tag + ")");
          require(down<float>(chain.vsc, slots * kv) == down<float>(fused.vsc, slots * kv), "V scales bitwise (" + tag + ")");
        }
        checked += static_cast<long>(co.size());
      }
    }
  }
  std::printf("[ .. ] fused decode attention (qk 128): %ld outputs and the appended cache rows bitwise the chain\n", checked);
}

// ---- the query-tiled prefill attention vs the split-KV chain -------------------------
// The fp8 cache's finish at qk 128 (engine.kv_dtype fp8): the K and V
// codes and the per-row scales the finish writes are bitwise the host
// codec of the bf16 finish's rows; the q rows are the bf16 finish's.
DGPP_TEST(flint_fp8_cache_finish_codes_are_the_host_codec_of_the_bf16_finish) {
  cudaStream_t st = test_stream();
  const Pool pool;
  const int rows = 9;
  const std::vector<int32_t> req_ids = {0, 0, 1, 1, 0, 1, 0, 1, 1};
  const std::vector<int64_t> pos = {0, 1, 17, 63, 64, 200, 383, -1, 5};
  DevBuf dreq = up(req_ids), dpos = up(pos), dtbl = up(pool.table);
  const std::vector<float> inv = inv_freq_host(1e4);
  DevBuf dinv = up(inv);
  const std::vector<uint16_t> qn = norm_weights(77), kn = norm_weights(78);
  DevBuf dqn = up(qn), dkn = up(kn);
  for (const bool rope : {true, false}) {
    dgpp::SinkQkvLayout layout;
    layout.chunks = 1;
    layout.q_per_chunk = 8;
    layout.kv_per_chunk = 2;
    const int64_t chunk_rows = static_cast<int64_t>(layout.q_per_chunk) * QK + layout.kv_per_chunk * (QK + D);
    layout.chunk_stride = chunk_rows;
    const int lh = layout.q_per_chunk, lkv = layout.kv_per_chunk;
    const int64_t qkv_stride = chunk_rows;
    const std::vector<float> qkv = random_f32_uniform(19 + (rope ? 1 : 0), static_cast<int64_t>(rows) * qkv_stride, 2.0f);
    DevBuf dqkv = up(qkv);
    const int qw = lh * QK, kw = lkv * QK, vw = lkv * D;
    const size_t slots = static_cast<size_t>(pool.slots());
    dgpp::SinkFinishParams fin;
    fin.inv_freq = rope ? ptr<float>(dinv) : nullptr;
    fin.q_norm = ptr<uint16_t>(dqn);
    fin.k_norm = ptr<uint16_t>(dkn);
    fin.eps = 1e-6f;
    // The bf16 finish: the values the fp8 form quantizes.
    DevBuf kc16(slots * kw * 2), vc16(slots * vw * 2), qo(static_cast<size_t>(rows) * qw * 2);
    dgpp::sink_qkv_finish(QK, ptr<float>(dqkv), qkv_stride, layout, fin, ptr<int32_t>(dreq), ptr<int64_t>(dpos), rows, lh,
                          lkv, ptr<int32_t>(dtbl), pool.blocks_per_request, pool.block_tokens, mptr<uint16_t>(qo), qw,
                          mptr<uint16_t>(kc16), mptr<uint16_t>(vc16), st);
    // The fp8 finish: codes and scales, the untouched slots poisoned.
    DevBuf kc8(slots * kw), vc8(slots * vw), ks(slots * lkv * 4), vs(slots * lkv * 4), qo8(static_cast<size_t>(rows) * qw * 2);
    DGPP_CUDA_OK(cudaMemset(kc8.p, 0x7F, slots * kw));
    DGPP_CUDA_OK(cudaMemset(vc8.p, 0x7F, slots * vw));
    DGPP_CUDA_OK(cudaMemset(ks.p, 0x7F, slots * lkv * 4));
    DGPP_CUDA_OK(cudaMemset(vs.p, 0x7F, slots * lkv * 4));
    dgpp::sink_qkv_finish(QK, ptr<float>(dqkv), qkv_stride, layout, fin, ptr<int32_t>(dreq), ptr<int64_t>(dpos), rows, lh,
                          lkv, ptr<int32_t>(dtbl), pool.blocks_per_request, pool.block_tokens, mptr<uint16_t>(qo8), qw,
                          mptr<uint16_t>(kc8), mptr<uint16_t>(vc8), st, mptr<float>(ks), mptr<float>(vs));
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    const std::vector<uint16_t> k16 = down<uint16_t>(kc16, slots * kw), v16 = down<uint16_t>(vc16, slots * vw);
    const std::vector<uint8_t> k8 = down<uint8_t>(kc8, slots * kw), v8 = down<uint8_t>(vc8, slots * vw);
    const std::vector<float> ks_h = down<float>(ks, slots * lkv), vs_h = down<float>(vs, slots * lkv);
    require(down<uint16_t>(qo8, static_cast<size_t>(rows) * qw) == down<uint16_t>(qo, static_cast<size_t>(rows) * qw),
            "the q rows are the bf16 finish's");
    std::vector<uint8_t> codes(static_cast<size_t>(QK));
    long checked = 0;
    for (int r = 0; r < rows; ++r) {
      const int64_t p = pos[static_cast<size_t>(r)];
      if (p < 0) continue;
      const size_t slot = static_cast<size_t>(pool.phys(req_ids[static_cast<size_t>(r)], p));
      for (int j = 0; j < lkv; ++j) {
        float scale = 0.f;
        host_fp8_row(k16.data() + slot * kw + j * QK, QK, codes.data(), &scale);
        for (int d = 0; d < QK; ++d) require(k8[slot * kw + j * QK + d] == codes[static_cast<size_t>(d)], "K codes bitwise");
        require(ks_h[slot * lkv + j] == scale, "K scale bitwise");
        host_fp8_row(v16.data() + slot * vw + j * D, D, codes.data(), &scale);
        for (int d = 0; d < D; ++d) require(v8[slot * vw + j * D + d] == codes[static_cast<size_t>(d)], "V codes bitwise");
        require(vs_h[slot * lkv + j] == scale, "V scale bitwise");
        checked += QK + D;
      }
    }
    std::printf("[ .. ] fp8 finish (%s): %ld codes bitwise the host codec, the scales bitwise\n", rope ? "rope" : "no-rope", checked);
  }
}

DGPP_TEST(flint_tiled_prefill_attention_matches_the_split_kv_chain) {
  cudaStream_t st = test_stream();
  const Pool pool;
  const std::vector<float> inv = inv_freq_host(1e4);
  DevBuf dinv = up(inv), dtbl = up(pool.table);
  const std::vector<uint16_t> qn = norm_weights(301), kn = norm_weights(302);
  DevBuf dqn = up(qn), dkn = up(kn);
  // Rows: request 0's positions 500..619 (120 rows: past the window from
  // 513 on), then request 1's 0..45, a padding row, request 1's 46..60.
  std::vector<int32_t> req_ids;
  std::vector<int64_t> pos;
  for (int64_t p = 500; p < 620; ++p) { req_ids.push_back(0); pos.push_back(p); }
  for (int64_t p = 0; p < 46; ++p) { req_ids.push_back(1); pos.push_back(p); }
  req_ids.push_back(1); pos.push_back(-1);
  for (int64_t p = 46; p < 61; ++p) { req_ids.push_back(1); pos.push_back(p); }
  const int rows = static_cast<int>(pos.size());
  const int pad_row = 120 + 46;
  DevBuf dreq = up(req_ids), dpos = up(pos);
  long checked = 0;
  int worst = 0;
  for (const int kvpc : {1, 2}) {
    dgpp::SinkQkvLayout layout;
    layout.chunks = 1;
    layout.q_per_chunk = 8 * kvpc;
    layout.kv_per_chunk = kvpc;
    const int64_t chunk_rows = static_cast<int64_t>(layout.q_per_chunk) * QK + kvpc * (QK + D);
    layout.chunk_stride = chunk_rows;
    const int lh = layout.q_per_chunk, kv = kvpc;
    const int64_t qkv_stride = chunk_rows;
    const int kw = kv * QK, vw = kv * D, ow = lh * D, qw = lh * QK;
    const std::vector<float> qkv = random_f32_uniform(500 + kvpc, static_cast<int64_t>(rows) * qkv_stride, 2.0f);
    const std::vector<float> sink = random_f32_uniform(510 + kvpc, lh, 2.0f);
    DevBuf dqkv = up(qkv), dsink = up(sink);
    for (const bool fp8 : {false, true}) {
      const size_t slots = static_cast<size_t>(pool.slots());
      std::vector<uint16_t> kh(slots * kw, 0), vh(slots * vw, 0);
      std::vector<uint8_t> k8(slots * kw, 0), v8(slots * vw, 0);
      std::vector<float> ks(slots * kv, 0.f), vs(slots * kv, 0.f);
      std::vector<uint8_t> codes(static_cast<size_t>(QK));
      const std::vector<uint16_t> k = random_bf16_normal(520, 500 * kw, 1.0f);
      const std::vector<uint16_t> v = random_bf16_normal(530, 500 * vw, 1.0f);
      for (int64_t tk = 0; tk < 500; ++tk) {
        const size_t s = static_cast<size_t>(pool.phys(0, tk));
        std::copy(k.begin() + tk * kw, k.begin() + (tk + 1) * kw, kh.begin() + s * kw);
        std::copy(v.begin() + tk * vw, v.begin() + (tk + 1) * vw, vh.begin() + s * vw);
        for (int j = 0; j < kv; ++j) {
          host_fp8_row(kh.data() + s * kw + j * QK, QK, codes.data(), &ks[s * kv + j]);
          std::copy(codes.begin(), codes.begin() + QK, k8.begin() + s * kw + j * QK);
          host_fp8_row(vh.data() + s * vw + j * D, D, codes.data(), &vs[s * kv + j]);
          std::copy(codes.begin(), codes.begin() + D, v8.begin() + s * vw + j * D);
        }
      }
      for (const bool swa : {false, true}) {
        const int window = swa ? 513 : 0;
        dgpp::SinkFinishParams fin;
        fin.inv_freq = swa ? ptr<float>(dinv) : nullptr;
        fin.q_norm = ptr<uint16_t>(dqn);
        fin.k_norm = ptr<uint16_t>(dkn);
        DevBuf kc = fp8 ? up(k8) : up(kh), vc = fp8 ? up(v8) : up(vh), ksc = up(ks), vsc = up(vs);
        DevBuf q(static_cast<size_t>(rows) * qw * 2);
        dgpp::sink_qkv_finish(QK, ptr<float>(dqkv), qkv_stride, layout, fin, ptr<int32_t>(dreq), ptr<int64_t>(dpos), rows, lh, kv,
                              ptr<int32_t>(dtbl), pool.blocks_per_request, pool.block_tokens, mptr<uint16_t>(q), qw,
                              mptr<uint16_t>(kc), mptr<uint16_t>(vc), st, fp8 ? mptr<float>(ksc) : nullptr,
                              fp8 ? mptr<float>(vsc) : nullptr);
        const size_t part = static_cast<size_t>(rows) * lh;
        DevBuf m_ws(part * 4), l_ws(part * 4), c_ws(part * D * 4), chain_out(static_cast<size_t>(rows) * ow * 2);
        dgpp::sink_attn_partial(QK, ptr<uint16_t>(q), qw, ptr<uint16_t>(kc), ptr<uint16_t>(vc), ptr<int32_t>(dreq), ptr<int64_t>(dpos),
                                rows, 1, lh, kv, pool.block_tokens, ptr<int32_t>(dtbl), pool.blocks_per_request, window,
                                ref_scale(QK), ptr<float>(dsink), mptr<float>(m_ws), mptr<float>(l_ws), mptr<float>(c_ws), st,
                                fp8 ? ptr<float>(ksc) : nullptr, fp8 ? ptr<float>(vsc) : nullptr);
        dgpp::sink_attn_combine(ptr<float>(m_ws), ptr<float>(l_ws), ptr<float>(c_ws), rows, 1, lh, mptr<uint16_t>(chain_out), st);
        DevBuf tiled_out(static_cast<size_t>(rows) * ow * 2);
        DGPP_CUDA_OK(cudaMemset(tiled_out.p, 0x7F, tiled_out.bytes));
        dgpp::SinkAttnPrefillArgs a;
        a.q = ptr<uint16_t>(q);
        a.q_stride = qw;
        a.k_cache = ptr<uint16_t>(kc);
        a.v_cache = ptr<uint16_t>(vc);
        a.k_scale = fp8 ? ptr<float>(ksc) : nullptr;
        a.v_scale = fp8 ? ptr<float>(vsc) : nullptr;
        a.req_ids = ptr<int32_t>(dreq);
        a.pos = ptr<int64_t>(dpos);
        a.rows = rows;
        a.local_heads = lh;
        a.kv_heads = kv;
        a.block_tokens = pool.block_tokens;
        a.block_tables = ptr<int32_t>(dtbl);
        a.blocks_per_request = pool.blocks_per_request;
        a.window = window;
        a.scale = ref_scale(QK);
        a.sink = ptr<float>(dsink);
        a.out = mptr<uint16_t>(tiled_out);
        dgpp::sink_attn_prefill(QK, a, st);
        DGPP_CUDA_OK(cudaStreamSynchronize(st));
        const auto co = down<uint16_t>(chain_out, static_cast<size_t>(rows) * ow);
        const auto to = down<uint16_t>(tiled_out, static_cast<size_t>(rows) * ow);
        const std::string tag = std::string(fp8 ? "fp8" : "bf16") + (swa ? " swa" : " full") + " kvpc " + std::to_string(kvpc);
        std::vector<float> cf(co.size()), tf(co.size());
        double rms = 0;
        for (size_t i = 0; i < co.size(); ++i) {
          cf[i] = dgpp::bf16_bits_to_float(co[i]);
          tf[i] = dgpp::bf16_bits_to_float(to[i]);
          rms += static_cast<double>(cf[i]) * cf[i];
        }
        rms = std::sqrt(rms / static_cast<double>(co.size()));
        const Stats s = compare_abs_rel(tf.data(), cf.data(), static_cast<long>(co.size()), 2 * std::pow(2.0, -7.0), 0.005 * rms);
        std::printf("[ .. ] tiled prefill (qk 128) %s: l2_rel %.3g max_abs %.3g mismatches %ld/%ld (rms %.3g)\n", tag.c_str(),
                    s.l2_rel, s.max_abs, s.mismatches, s.n, rms);
        require_bf16("tiled prefill attention " + tag, s, 2e-3, 0.01);
        worst = std::max(worst, static_cast<int>(s.mismatches));
        for (int i = 0; i < ow; ++i)
          require(to[static_cast<size_t>(pad_row) * ow + i] == 0, "the tiled kernel's padding row is zero (" + tag + ")");
        checked += static_cast<long>(co.size());
      }
    }
  }
  std::printf("[ .. ] tiled prefill attention (qk 128): %ld outputs vs the split-KV chain, at most %d past two ulps per case\n",
              checked, worst);
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;
  return dgpp::test::run_all();
}
