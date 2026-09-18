// The QSA kernels against the host reference on small
// geometry: per-head norm+RoPE within two bf16 ulps; the index compression
// through the prefill kernel, the decode ring and a prefill-then-decode
// split all bitwise one another (and the ring snapshots the ring's own
// history), within two ulps of the reference; the indexer scores and the
// selection bitwise; the listed attention (one and three splits) with its
// gate within two ulps. Plus the frozen plain rope table and the YaRN
// table + mscale the engine's knob (engine.rope_scaling) builds.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/dsa.hpp"
#include "kernels/qsa.hpp"
#include "kernels/rope_scaling.hpp"
#include "models/qwen/qsa_reference.hpp"

using namespace dgpp::kda_test;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

struct Geo {
  int local_heads = 6, kv_heads = 2, dim = 256, rotary = 64;
  int idx_heads = 4, idx_dim = 128;
  int kpool = 4, select_k = 8, block_tokens = 16, blocks_per_request = 16;
  int seq = 200;
  double theta = 1e7;
  float eps = 1e-6f;
  int max_selected() const { return select_k * kpool + kpool - 1; }
  int pools_per_block() const { return block_tokens / kpool; }
  int slots() const { return block_tokens * blocks_per_request; }
  int pool_slots() const { return slots() / kpool; }
};

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

std::vector<float> inv_freq_host(const Geo& g) {
  std::vector<float> f;
  dgpp::qwen_ref::rope_inv_freq(g.theta, g.rotary, f);
  std::vector<float> dev(f.size());
  dgpp::qsa_rope_inv_freq(g.theta, g.rotary, dev.data());
  require(dev == f, "inv_freq host helpers disagree");
  return f;
}

std::vector<int32_t> identity_table(const Geo& g) {
  std::vector<int32_t> t(static_cast<size_t>(g.blocks_per_request));
  for (int i = 0; i < g.blocks_per_request; ++i) t[static_cast<size_t>(i)] = i;
  return t;
}

}  // namespace

DGPP_TEST(qsa_rope_inv_freq_is_frozen_and_yarn_rides_the_cos_sin) {
  // The plain table, frozen from the build that predates the YaRN knob
  // (2026-09-17): the knob must not move it.
  const uint32_t plain[32] = {
      0x3f800000, 0x3f1ab32b, 0x3ebaf81b, 0x3e61f835, 0x3e088d77, 0x3da50956, 0x3d47763f,
      0x3cf11177, 0x3c91ad39, 0x3c301052, 0x3bd4ca15, 0x3b80967d, 0x3b1b690d, 0x3abbd3ed,
      0x3a6301e2, 0x3a092e02, 0x39a5cb60, 0x394860c1, 0x38f22ce2, 0x3892587e, 0x3830df52,
      0x37d5c441, 0x37812dab, 0x371c1fc4, 0x36bcb0c1, 0x36640cc6, 0x3609cf4a, 0x35a68e4c,
      0x35494c57, 0x34f3499d, 0x3493048e, 0x3431af44};
  std::vector<float> f(32);
  dgpp::qsa_rope_inv_freq(1e7, 64, f.data());
  for (int i = 0; i < 32; ++i) {
    uint32_t u = 0;
    std::memcpy(&u, &f[static_cast<size_t>(i)], 4);
    require(u == plain[i], "the plain rope table moved");
  }
  // The YaRN rope the knob builds is the same kernel, cos/sin scaled by
  // the attention factor before their bf16 rounding: the reference (built
  // with the mscale) is the gate's oracle for it, bit for bit.
  Geo g;
  const int rows = 4, heads = 2;
  // 524287 is the recipe's last position at factor 2 over 262144: the
  // largest argument the cosf/sinf reduction sees at 512K.
  const std::vector<int64_t> pos{0, 5, 4096, 524287};
  const int64_t x_head_stride = 2 * g.dim, x_row_stride = heads * x_head_stride;
  const std::vector<uint16_t> x = random_bf16_normal(0x31, rows * x_row_stride, 1.0f);
  const std::vector<uint16_t> w = random_bf16_uniform(0x32, g.dim, 0.5f);
  const float mscale = dgpp::RopeScaling{2.0, 262144, 32.0, 1.0, 1.0, 4.0}.mscale();
  require(mscale > 1.0f, "the recipe's attention factor");
  std::vector<float> yarn(32);
  dgpp::yarn_rope_inv_freq_host(64, 1e7, 262144 * 4, 2.0, 32.0, 1.0, yarn.data());
  std::vector<uint16_t> ref(static_cast<size_t>(rows) * heads * g.dim);
  for (int r = 0; r < rows; ++r)
    for (int h = 0; h < heads; ++h)
      dgpp::qwen_ref::qsa_norm_rope(x.data() + r * x_row_stride + h * x_head_stride, w.data(),
                                    pos[static_cast<size_t>(r)], yarn.data(),
                                    ref.data() + (static_cast<size_t>(r) * heads + h) * g.dim, g.dim,
                                    g.rotary, g.eps, mscale);
  DevBuf dx = up(x), dw = up(w), dpos = up(pos), dinv = up(yarn), dout(ref.size() * 2);
  cudaStream_t st = test_stream();
  dgpp::qsa_norm_rope_bf16(ptr<uint16_t>(dx), x_row_stride, x_head_stride, ptr<uint16_t>(dw),
                           ptr<int64_t>(dpos), ptr<float>(dinv), mptr<uint16_t>(dout),
                           static_cast<int64_t>(heads) * g.dim, rows, heads, g.dim, g.rotary, g.eps,
                           mscale, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> got = down<uint16_t>(dout, ref.size());
  const Stats s = compare_bf16(got, ref, 2);
  std::printf("[ .. ] yarn norm+rope: max_rel %.3g l2_rel %.3g mismatches %ld/%ld\n", s.max_rel,
              s.l2_rel, s.mismatches, s.n);
  require_bf16("yarn norm+rope", s, 2e-3, 0.01);
}

DGPP_TEST(qsa_norm_rope_matches_the_reference) {
  Geo g;
  const int rows = 5, heads = 3;
  const std::vector<int64_t> pos{0, 1, 7, 1000, 123456};
  const int64_t x_head_stride = 2 * g.dim, x_row_stride = heads * x_head_stride;
  const std::vector<uint16_t> x = random_bf16_normal(1, rows * x_row_stride, 1.0f);
  const std::vector<uint16_t> w = random_bf16_uniform(2, g.dim, 0.5f);
  const std::vector<float> inv = inv_freq_host(g);
  std::vector<uint16_t> ref(static_cast<size_t>(rows) * heads * g.dim);
  for (int r = 0; r < rows; ++r)
    for (int h = 0; h < heads; ++h)
      dgpp::qwen_ref::qsa_norm_rope(x.data() + r * x_row_stride + h * x_head_stride, w.data(),
                                    pos[static_cast<size_t>(r)], inv.data(),
                                    ref.data() + (static_cast<size_t>(r) * heads + h) * g.dim, g.dim,
                                    g.rotary, g.eps);
  DevBuf dx = up(x), dw = up(w), dpos = up(pos), dinv = up(inv), dout(ref.size() * 2);
  cudaStream_t st = test_stream();
  dgpp::qsa_norm_rope_bf16(ptr<uint16_t>(dx), x_row_stride, x_head_stride, ptr<uint16_t>(dw),
                           ptr<int64_t>(dpos), ptr<float>(dinv), mptr<uint16_t>(dout),
                           static_cast<int64_t>(heads) * g.dim, rows, heads, g.dim, g.rotary, g.eps,
                           1.0f, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> got = down<uint16_t>(dout, ref.size());
  const Stats s = compare_bf16(got, ref, 2);
  std::printf("[ .. ] norm+rope: max_rel %.3g l2_rel %.3g mismatches %ld/%ld\n", s.max_rel, s.l2_rel,
              s.mismatches, s.n);
  require_bf16("norm+rope", s, 2e-3, 0.01);
}

namespace {

struct IndexFixture {
  Geo g;
  std::vector<uint16_t> raw;   // [seq, idx_dim]
  std::vector<uint16_t> w_k;   // [idx_dim]
  std::vector<float> inv;
  std::vector<int32_t> table;
  DevBuf draw, dwk, dinv, dtable;
  explicit IndexFixture(uint64_t seed) {
    raw = random_bf16_normal(seed, static_cast<int64_t>(g.seq) * g.idx_dim, 1.0f);
    w_k = random_bf16_uniform(seed + 1, g.idx_dim, 0.5f);
    inv = inv_freq_host(g);
    table = identity_table(g);
    draw = up(raw);
    dwk = up(w_k);
    dinv = up(inv);
    dtable = up(table);
  }
  // The oracle's cache: pool p from raw rows [4p, 4p+4) at position 4p.
  std::vector<uint16_t> oracle_cache() const {
    std::vector<uint16_t> c(static_cast<size_t>(g.pool_slots()) * g.idx_dim, 0);
    for (int p = 0; p < g.seq / g.kpool; ++p)
      dgpp::qwen_ref::qsa_index_compress(raw.data() + static_cast<size_t>(p) * g.kpool * g.idx_dim, g.kpool,
                                         w_k.data(), inv.data(), static_cast<int64_t>(p) * g.kpool,
                                         c.data() + static_cast<size_t>(p) * g.idx_dim, g.idx_dim, g.rotary,
                                         g.eps);
    return c;
  }
  // Decode updates over tokens [t0, t1) one span each, into cache/ring.
  void decode_tokens(int t0, int t1, DevBuf& dcache, DevBuf& dring, cudaStream_t st, int span_len = 1,
                     uint16_t* snapshots = nullptr) const {
    for (int t = t0; t < t1; t += span_len) {
      const int n = std::min(span_len, t1 - t);
      std::vector<int32_t> req_ids(static_cast<size_t>(n), 0);
      std::vector<int64_t> pos(static_cast<size_t>(n));
      for (int i = 0; i < n; ++i) pos[static_cast<size_t>(i)] = t + i;
      const std::vector<int32_t> spans{0, n};
      DevBuf dreq = up(req_ids), dpos = up(pos), dspans = up(spans);
      dgpp::qsa_index_decode_update(ptr<uint16_t>(draw) + static_cast<int64_t>(t) * g.idx_dim, g.idx_dim,
                                    ptr<uint16_t>(dwk), ptr<float>(dinv), ptr<int32_t>(dreq),
                                    ptr<int64_t>(dpos), ptr<int32_t>(dspans), 1, ptr<int32_t>(dtable),
                                    g.blocks_per_request, mptr<uint16_t>(dring), mptr<uint16_t>(dcache),
                                    g.pools_per_block(), g.kpool, g.idx_dim, g.rotary, g.eps, 1.0f, st,
                                    snapshots);
      DGPP_CUDA_OK(cudaStreamSynchronize(st));
    }
  }
};

}  // namespace

DGPP_TEST(qsa_index_compression_prefill_decode_and_split_agree) {
  IndexFixture f(10);
  const Geo& g = f.g;
  cudaStream_t st = test_stream();
  const size_t cache_elems = static_cast<size_t>(g.pool_slots()) * g.idx_dim;
  const size_t ring_elems = static_cast<size_t>(g.kpool) * g.idx_dim;
  // 1. Prefill: every complete pool of the sequence.
  DevBuf c_prefill(cache_elems * 2);
  DGPP_CUDA_OK(cudaMemset(c_prefill.p, 0, cache_elems * 2));
  const int n_pools = g.seq / g.kpool;
  dgpp::qsa_index_compress_write(ptr<uint16_t>(f.draw), g.idx_dim, ptr<uint16_t>(f.dwk), ptr<float>(f.dinv),
                                 ptr<int32_t>(f.dtable), g.pools_per_block(), 0, n_pools,
                                 mptr<uint16_t>(c_prefill), g.kpool, g.idx_dim, g.rotary, g.eps, 1.0f, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> prefill = down<uint16_t>(c_prefill, cache_elems);
  // 2. Decode token by token from an empty ring.
  DevBuf c_decode(cache_elems * 2), ring(ring_elems * 2);
  DGPP_CUDA_OK(cudaMemset(c_decode.p, 0, cache_elems * 2));
  DGPP_CUDA_OK(cudaMemset(ring.p, 0, ring_elems * 2));
  f.decode_tokens(0, g.seq, c_decode, ring, st);
  const std::vector<uint16_t> decode = down<uint16_t>(c_decode, cache_elems);
  require_bitwise("decode cache == prefill cache", decode.data(), prefill.data(), cache_elems * 2);
  // 3. Prefill 154 tokens (38 pools + 2 tail), seed the ring, decode the rest.
  const int cut = 154;
  DevBuf c_split(cache_elems * 2), ring2(ring_elems * 2);
  DGPP_CUDA_OK(cudaMemset(c_split.p, 0, cache_elems * 2));
  DGPP_CUDA_OK(cudaMemset(ring2.p, 0, ring_elems * 2));
  dgpp::qsa_index_compress_write(ptr<uint16_t>(f.draw), g.idx_dim, ptr<uint16_t>(f.dwk), ptr<float>(f.dinv),
                                 ptr<int32_t>(f.dtable), g.pools_per_block(), 0, cut / g.kpool,
                                 mptr<uint16_t>(c_split), g.kpool, g.idx_dim, g.rotary, g.eps, 1.0f, st);
  {
    std::vector<int32_t> req_ids(static_cast<size_t>(cut), 0);
    std::vector<int64_t> pos(static_cast<size_t>(cut));
    for (int i = 0; i < cut; ++i) pos[static_cast<size_t>(i)] = i;
    DevBuf dreq = up(req_ids), dpos = up(pos);
    dgpp::qsa_index_tail_seed(ptr<uint16_t>(f.draw), g.idx_dim, ptr<int32_t>(dreq), ptr<int64_t>(dpos), cut,
                              mptr<uint16_t>(ring2), g.kpool, g.idx_dim, st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
  }
  // Spans of three tokens with snapshots: rows that are not their span's
  // last leave the ring as it stood after them.
  DevBuf snaps(static_cast<size_t>(3) * ring_elems * 2);
  for (int t = cut; t < g.seq; t += 3) {
    const int n = std::min(3, g.seq - t);
    std::vector<int32_t> req_ids(static_cast<size_t>(n), 0);
    std::vector<int64_t> pos(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) pos[static_cast<size_t>(i)] = t + i;
    const std::vector<int32_t> spans{0, n};
    DevBuf dreq = up(req_ids), dpos = up(pos), dspans = up(spans);
    // The expected snapshots: the ring after each of the first n-1 tokens,
    // from single-token updates on a copy.
    std::vector<std::vector<uint16_t>> expect;
    {
      DevBuf ring_copy(ring_elems * 2), cache_copy(cache_elems * 2);
      DGPP_CUDA_OK(cudaMemcpy(ring_copy.p, ring2.p, ring_elems * 2, cudaMemcpyDeviceToDevice));
      DGPP_CUDA_OK(cudaMemcpy(cache_copy.p, c_split.p, cache_elems * 2, cudaMemcpyDeviceToDevice));
      for (int i = 0; i + 1 < n; ++i) {
        f.decode_tokens(t + i, t + i + 1, cache_copy, ring_copy, st);
        expect.push_back(down<uint16_t>(ring_copy, ring_elems));
      }
    }
    dgpp::qsa_index_decode_update(ptr<uint16_t>(f.draw) + static_cast<int64_t>(t) * g.idx_dim, g.idx_dim,
                                  ptr<uint16_t>(f.dwk), ptr<float>(f.dinv), ptr<int32_t>(dreq),
                                  ptr<int64_t>(dpos), ptr<int32_t>(dspans), 1, ptr<int32_t>(f.dtable),
                                  g.blocks_per_request, mptr<uint16_t>(ring2), mptr<uint16_t>(c_split),
                                  g.pools_per_block(), g.kpool, g.idx_dim, g.rotary, g.eps, 1.0f, st,
                                  mptr<uint16_t>(snaps));
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    const std::vector<uint16_t> got_snaps = down<uint16_t>(snaps, static_cast<size_t>(n) * ring_elems);
    for (size_t i = 0; i < expect.size(); ++i)
      require_bitwise("ring snapshot", got_snaps.data() + i * ring_elems, expect[i].data(), ring_elems * 2);
  }
  const std::vector<uint16_t> split = down<uint16_t>(c_split, cache_elems);
  require_bitwise("split cache == prefill cache", split.data(), prefill.data(), cache_elems * 2);
  // Both rings hold the last two raw keys (positions 198, 199 at slots 2, 3).
  const std::vector<uint16_t> r1 = down<uint16_t>(ring, ring_elems), r2 = down<uint16_t>(ring2, ring_elems);
  for (int s = 2; s < 4; ++s)
    for (int d = 0; d < g.idx_dim; ++d) {
      const uint16_t want = f.raw[static_cast<size_t>(196 + s) * g.idx_dim + d];
      require(r1[static_cast<size_t>(s) * g.idx_dim + d] == want && r2[static_cast<size_t>(s) * g.idx_dim + d] == want,
              "ring does not hold the last raw keys");
    }
  // 4. The reference.
  const std::vector<uint16_t> oracle = f.oracle_cache();
  const size_t used = static_cast<size_t>(n_pools) * g.idx_dim;
  const Stats s = compare_bf16(std::vector<uint16_t>(prefill.begin(), prefill.begin() + used),
                               std::vector<uint16_t>(oracle.begin(), oracle.begin() + used), 2);
  std::printf("[ .. ] compressed keys: max_rel %.3g l2_rel %.3g mismatches %ld/%ld\n", s.max_rel, s.l2_rel,
              s.mismatches, s.n);
  require_bf16("compressed keys", s, 2e-3, 0.01);
}

namespace {

struct SelectFixture {
  IndexFixture f;
  std::vector<int64_t> pos{3, 4, 7, 31, 100, 150, 199, -1};
  int rows() const { return static_cast<int>(pos.size()); }
  std::vector<uint16_t> q;         // [rows, 4 * 128]
  std::vector<uint16_t> cache;     // the device's compressed cache
  DevBuf dcache, dq, dpos, dreq;
  explicit SelectFixture(uint64_t seed) : f(seed) {
    const Geo& g = f.g;
    q = random_bf16_normal(seed + 5, static_cast<int64_t>(rows()) * g.idx_heads * g.idx_dim, 1.0f);
    cudaStream_t st = test_stream();
    const size_t cache_elems = static_cast<size_t>(g.pool_slots()) * g.idx_dim;
    dcache = DevBuf(cache_elems * 2);
    DGPP_CUDA_OK(cudaMemset(dcache.p, 0, cache_elems * 2));
    dgpp::qsa_index_compress_write(ptr<uint16_t>(f.draw), g.idx_dim, ptr<uint16_t>(f.dwk), ptr<float>(f.dinv),
                                   ptr<int32_t>(f.dtable), g.pools_per_block(), 0, g.seq / g.kpool,
                                   mptr<uint16_t>(dcache), g.kpool, g.idx_dim, g.rotary, g.eps, 1.0f, st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    cache = down<uint16_t>(dcache, cache_elems);
    dq = up(q);
    dpos = up(pos);
    dreq = up(std::vector<int32_t>(static_cast<size_t>(rows()), 0));
  }
  // The reference lists per row.
  void reference_lists(std::vector<std::vector<int32_t>>& lists, std::vector<std::vector<uint64_t>>& keys) const {
    const Geo& g = f.g;
    lists.assign(static_cast<size_t>(rows()), {});
    keys.assign(static_cast<size_t>(rows()), {});
    for (int r = 0; r < rows(); ++r) {
      if (pos[static_cast<size_t>(r)] < 0) continue;
      const int64_t visible = (pos[static_cast<size_t>(r)] + 1) / g.kpool;
      std::vector<float> scores(static_cast<size_t>(visible));
      for (int64_t p = 0; p < visible; ++p) {
        scores[static_cast<size_t>(p)] = dgpp::qwen_ref::qsa_index_score(
            q.data() + static_cast<size_t>(r) * g.idx_heads * g.idx_dim, cache.data() + static_cast<size_t>(p) * g.idx_dim);
        uint32_t u;
        std::memcpy(&u, &scores[static_cast<size_t>(p)], 4);
        const uint32_t sortable = (u >> 31) ? ~u : (u | 0x80000000u);
        keys[static_cast<size_t>(r)].push_back((static_cast<uint64_t>(~sortable) << 21) | static_cast<uint64_t>(p));
      }
      std::vector<int32_t> ids;
      dgpp::qwen_ref::qsa_select(scores, g.select_k, ids);
      dgpp::qwen_ref::qsa_expand(ids, pos[static_cast<size_t>(r)], g.kpool, lists[static_cast<size_t>(r)]);
    }
  }
};

}  // namespace

DGPP_TEST(qsa_index_score_and_select_match_the_reference_bitwise) {
  SelectFixture sf(20);
  const Geo& g = sf.f.g;
  cudaStream_t st = test_stream();
  const int64_t ws_stride = g.pool_slots();
  DevBuf keys(static_cast<size_t>(sf.rows()) * ws_stride * 8);
  DGPP_CUDA_OK(cudaMemset(keys.p, 0xff, static_cast<size_t>(sf.rows()) * ws_stride * 8));
  dgpp::qsa_index_score(ptr<uint16_t>(sf.dq), static_cast<int64_t>(g.idx_heads) * g.idx_dim, ptr<int32_t>(sf.dreq),
                        ptr<int64_t>(sf.dpos), sf.rows(), ptr<int32_t>(sf.f.dtable), g.blocks_per_request,
                        ptr<uint16_t>(sf.dcache), g.pools_per_block(), g.idx_heads, g.idx_dim, g.kpool,
                        mptr<uint64_t>(keys), ws_stride, st);
  DevBuf topk(static_cast<size_t>(sf.rows()) * g.max_selected() * 4), counts(static_cast<size_t>(sf.rows()) * 4);
  dgpp::qsa_select_from_keys(ptr<uint64_t>(keys), ws_stride, ptr<int64_t>(sf.dpos), sf.rows(), g.select_k, g.kpool,
                             g.max_selected(), mptr<int32_t>(topk), mptr<int32_t>(counts), st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint64_t> got_keys = down<uint64_t>(keys, static_cast<size_t>(sf.rows()) * ws_stride);
  const std::vector<int32_t> got_topk = down<int32_t>(topk, static_cast<size_t>(sf.rows()) * g.max_selected());
  const std::vector<int32_t> got_counts = down<int32_t>(counts, static_cast<size_t>(sf.rows()));
  std::vector<std::vector<int32_t>> lists;
  std::vector<std::vector<uint64_t>> keys_ref;
  sf.reference_lists(lists, keys_ref);
  for (int r = 0; r < sf.rows(); ++r) {
    const std::vector<uint64_t>& kr = keys_ref[static_cast<size_t>(r)];
    for (size_t p = 0; p < kr.size(); ++p)
      require(got_keys[static_cast<size_t>(r) * ws_stride + p] == kr[p],
              "score key differs at row " + std::to_string(r) + " pool " + std::to_string(p));
    const std::vector<int32_t>& lr = lists[static_cast<size_t>(r)];
    require(got_counts[static_cast<size_t>(r)] == static_cast<int32_t>(lr.size()),
            "count differs at row " + std::to_string(r));
    for (size_t i = 0; i < lr.size(); ++i)
      require(got_topk[static_cast<size_t>(r) * g.max_selected() + i] == lr[i], "list differs at row " + std::to_string(r));
    for (size_t i = lr.size(); i < static_cast<size_t>(g.max_selected()); ++i)
      require(got_topk[static_cast<size_t>(r) * g.max_selected() + i] == -1, "list padding at row " + std::to_string(r));
  }
  std::printf("[ .. ] select: %d rows, lists of", sf.rows());
  for (int r = 0; r < sf.rows(); ++r) std::printf(" %d", got_counts[static_cast<size_t>(r)]);
  std::printf(" tokens, keys and lists bitwise\n");
}

DGPP_TEST(qsa_listed_attention_and_gate_match_the_reference) {
  SelectFixture sf(30);
  const Geo& g = sf.f.g;
  cudaStream_t st = test_stream();
  const int rows = sf.rows();
  const int width = g.kv_heads * g.dim;
  // K/V rows for the sequence, appended through the paged caches.
  const std::vector<uint16_t> k = random_bf16_normal(31, static_cast<int64_t>(g.seq) * width, 1.0f);
  const std::vector<uint16_t> v = random_bf16_normal(32, static_cast<int64_t>(g.seq) * width, 1.0f);
  DevBuf dk = up(k), dv = up(v);
  DevBuf kc(static_cast<size_t>(g.slots()) * width * 2), vc(static_cast<size_t>(g.slots()) * width * 2);
  {
    std::vector<int32_t> req_ids(static_cast<size_t>(g.seq), 0);
    std::vector<int64_t> pos(static_cast<size_t>(g.seq));
    for (int i = 0; i < g.seq; ++i) pos[static_cast<size_t>(i)] = i;
    DevBuf dreq = up(req_ids), dpos = up(pos);
    dgpp::qsa_kv_append(ptr<uint16_t>(dk), width, ptr<uint16_t>(dv), width, ptr<int32_t>(dreq), ptr<int64_t>(dpos),
                        g.seq, ptr<int32_t>(sf.f.dtable), g.blocks_per_request, g.block_tokens, g.kv_heads, g.dim,
                        mptr<uint16_t>(kc), mptr<uint16_t>(vc), st);
  }
  // Queries with the [q | gate] interleave; the reference lists.
  const int64_t q_head_stride = 2 * g.dim, q_row_stride = g.local_heads * q_head_stride;
  const std::vector<uint16_t> qg = random_bf16_normal(33, rows * q_row_stride, 1.0f);
  std::vector<uint16_t> qonly(static_cast<size_t>(rows) * g.local_heads * g.dim);
  for (int r = 0; r < rows; ++r)
    for (int h = 0; h < g.local_heads; ++h)
      std::copy(qg.begin() + r * q_row_stride + h * q_head_stride, qg.begin() + r * q_row_stride + h * q_head_stride + g.dim,
                qonly.begin() + (static_cast<size_t>(r) * g.local_heads + h) * g.dim);
  std::vector<std::vector<int32_t>> lists;
  std::vector<std::vector<uint64_t>> keys_ref;
  sf.reference_lists(lists, keys_ref);
  std::vector<int32_t> topk(static_cast<size_t>(rows) * g.max_selected(), -1), counts(static_cast<size_t>(rows), 0);
  for (int r = 0; r < rows; ++r) {
    counts[static_cast<size_t>(r)] = static_cast<int32_t>(lists[static_cast<size_t>(r)].size());
    std::copy(lists[static_cast<size_t>(r)].begin(), lists[static_cast<size_t>(r)].end(),
              topk.begin() + static_cast<size_t>(r) * g.max_selected());
  }
  // The reference output per row: attention over the gathered rows, gated.
  std::vector<uint16_t> ref(static_cast<size_t>(rows) * g.local_heads * g.dim, 0);
  for (int r = 0; r < rows; ++r) {
    const int n = counts[static_cast<size_t>(r)];
    std::vector<uint16_t> kr(static_cast<size_t>(n) * width), vr(static_cast<size_t>(n) * width);
    for (int i = 0; i < n; ++i) {
      const int32_t tok = topk[static_cast<size_t>(r) * g.max_selected() + i];
      std::copy(k.begin() + static_cast<size_t>(tok) * width, k.begin() + static_cast<size_t>(tok + 1) * width,
                kr.begin() + static_cast<size_t>(i) * width);
      std::copy(v.begin() + static_cast<size_t>(tok) * width, v.begin() + static_cast<size_t>(tok + 1) * width,
                vr.begin() + static_cast<size_t>(i) * width);
    }
    std::vector<float> c;
    dgpp::qwen_ref::qsa_attention(qonly.data() + static_cast<size_t>(r) * g.local_heads * g.dim, kr.data(), vr.data(), n,
                                  g.local_heads, g.kv_heads, g.dim, 1.0f / 16.0f, c);
    dgpp::qwen_ref::qsa_gate_out(c.data(), qg.data() + r * q_row_stride + g.dim, q_head_stride,
                                 ref.data() + static_cast<size_t>(r) * g.local_heads * g.dim, g.local_heads, g.dim);
  }
  DevBuf dqg = up(qg), dqonly = up(qonly), dtopk = up(topk), dcounts = up(counts);
  for (auto attend : {dgpp::qsa_attn_partial, dgpp::qsa_attn_prefill_partial}) {
    for (int n_split : {1, 3}) {
      const size_t part = static_cast<size_t>(rows) * n_split * g.local_heads;
      DevBuf m_ws(part * 4), l_ws(part * 4), c_ws(part * g.dim * 4), c_out(static_cast<size_t>(rows) * g.local_heads * g.dim * 4),
          out(ref.size() * 2);
      attend(ptr<uint16_t>(dqonly), static_cast<int64_t>(g.local_heads) * g.dim, ptr<uint16_t>(kc),
                             ptr<uint16_t>(vc), ptr<int32_t>(sf.dreq), ptr<int32_t>(dtopk), g.max_selected(),
                             ptr<int32_t>(dcounts), rows, n_split, g.local_heads, g.kv_heads, g.dim, g.block_tokens,
                             ptr<int32_t>(sf.f.dtable), g.blocks_per_request, 1.0f / 16.0f, mptr<float>(m_ws),
                             mptr<float>(l_ws), mptr<float>(c_ws), st);
      dgpp::dsa_attn_combine(ptr<float>(m_ws), ptr<float>(l_ws), ptr<float>(c_ws), rows, n_split, g.local_heads, g.dim,
                             mptr<float>(c_out), st);
      dgpp::qsa_gate_out(ptr<float>(c_out), ptr<uint16_t>(dqg) + g.dim, q_row_stride, q_head_stride, mptr<uint16_t>(out),
                         rows, g.local_heads, g.dim, st);
      DGPP_CUDA_OK(cudaStreamSynchronize(st));
      const std::vector<uint16_t> got = down<uint16_t>(out, ref.size());
      // One split is the reference's chain in the reference's order: two
      // ulps. Several splits round each probability to bf16 against its
      // split's own max and rescale in the combine — a reassociation of
      // the bf16 roundings (the DSA split kernel's tolerance class): eight
      // ulps with a 2 % of RMS absolute floor for the cancelled elements.
      // The padding row (pos -1, count 0) is zeros on both sides.
      std::vector<float> gf(got.size()), wf(ref.size());
      double rms = 0;
      for (size_t i = 0; i < got.size(); ++i) {
        gf[i] = dgpp::bf16_bits_to_float(got[i]);
        wf[i] = dgpp::bf16_bits_to_float(ref[i]);
        rms += static_cast<double>(wf[i]) * wf[i];
      }
      rms = std::sqrt(rms / static_cast<double>(ref.size()));
      const int ulps = n_split == 1 ? 2 : 8;
      const Stats s = compare_abs_rel(gf.data(), wf.data(), static_cast<long>(got.size()),
                                      ulps * std::pow(2.0, -7.0), n_split == 1 ? 1e-7 : 0.02 * rms);
      std::printf("[ .. ] attention n_split=%d: max_abs %.3g l2_rel %.3g mismatches %ld/%ld (rms %.3g)\n", n_split,
                  s.max_abs, s.l2_rel, s.mismatches, s.n, rms);
      require_bf16("attention n_split=" + std::to_string(n_split), s, n_split == 1 ? 2e-3 : 4e-3, 0.01);
    }
  }
}

DGPP_TEST(qsa_prefill_partials_preserve_arithmetic_and_graph_replay) {
  constexpr int dim = 256, block_tokens = 256, blocks = 32, requests = 2, stride = 2051;
  const std::vector<int32_t> counts{0, 1, 7, 31, 32, 33, 63, 64, 65, 255, 256, 257, 2047, 2048, 2051};
  const int rows = static_cast<int>(counts.size());
  std::vector<int32_t> req(rows), topk(rows * stride, -1), table(blocks * requests);
  for (int i = 0; i < blocks * requests; ++i) table[i] = (i * 17 + 3) % (blocks * requests);
  for (int r = 0; r < rows; ++r) {
    req[r] = r % requests;
    for (int i = 0; i < counts[r]; ++i) topk[r * stride + i] = i * 3 + r;
  }
  DevBuf dreq = up(req), dtopk = up(topk), dcounts = up(counts), dtable = up(table);
  cudaStream_t stream = test_stream();
  for (auto shape : {std::pair{24, 2}, std::pair{12, 1}, std::pair{6, 1},
                     std::pair{6, 2}, std::pair{2, 2}, std::pair{1, 1}}) {
    const int heads = shape.first, kv_heads = shape.second, qstride = heads * dim + 8;
    auto q = random_bf16_normal(700 + heads, static_cast<int64_t>(rows) * qstride, 1.0f);
    auto k = random_bf16_normal(710 + kv_heads, static_cast<int64_t>(blocks * requests * block_tokens) * kv_heads * dim, 1.0f);
    auto v = random_bf16_normal(720 + kv_heads, static_cast<int64_t>(k.size()), 1.0f);
    // Include a high-dynamic-range query to exercise repeated maximum rescaling.
    for (int d = 0; d < heads * dim; ++d)
      q[static_cast<size_t>(rows - 1) * qstride + d] = float_to_bf16_bits(bf16_bits_to_float(q[static_cast<size_t>(rows - 1) * qstride + d]) * 32);
    DevBuf dq = up(q), dk = up(k), dv = up(v);
    for (int splits : {1, 3, 8}) {
      const size_t part = static_cast<size_t>(rows) * splits * heads;
      DevBuf m0(part * 4), l0(part * 4), c0(part * dim * 4);
      DevBuf m1(part * 4), l1(part * 4), c1(part * dim * 4);
      auto launch = [&](bool candidate) {
        auto fn = candidate ? dgpp::qsa_attn_prefill_partial : dgpp::qsa_attn_partial;
        fn(ptr<uint16_t>(dq), qstride, ptr<uint16_t>(dk), ptr<uint16_t>(dv), ptr<int32_t>(dreq),
           ptr<int32_t>(dtopk), stride, ptr<int32_t>(dcounts), rows, splits, heads, kv_heads, dim,
           block_tokens, ptr<int32_t>(dtable), blocks, 1.0f / 16,
           mptr<float>(candidate ? m1 : m0), mptr<float>(candidate ? l1 : l0),
           mptr<float>(candidate ? c1 : c0), stream);
      };
      launch(false);
      launch(true);
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      auto check = [&] {
        for (auto pair : {std::pair{&m0, &m1}, std::pair{&l0, &l1}, std::pair{&c0, &c1}}) {
          auto ref = down<uint32_t>(*pair.first, pair.first->bytes / 4);
          auto got = down<uint32_t>(*pair.second, pair.second->bytes / 4);
          require(got == ref, "prefill partial bits differ, heads=" + std::to_string(heads) +
                  " kv=" + std::to_string(kv_heads) + " splits=" + std::to_string(splits));
        }
      };
      check();
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t exec = nullptr;
      DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      launch(true);
      DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
      DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
      DGPP_CUDA_OK(cudaMemsetAsync(m1.p, 0xff, m1.bytes, stream));
      DGPP_CUDA_OK(cudaMemsetAsync(l1.p, 0xff, l1.bytes, stream));
      DGPP_CUDA_OK(cudaMemsetAsync(c1.p, 0xff, c1.bytes, stream));
      DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      check();
      cudaGraphExecDestroy(exec);
      cudaGraphDestroy(graph);
    }
  }
}

int main() { return dgpp::test::run_all(); }
