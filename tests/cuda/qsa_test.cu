// The QSA kernels against the host reference on small
// geometry: per-head norm+RoPE within two bf16 ulps; the index compression
// through the prefill kernel, the decode ring and a prefill-then-decode
// split all bitwise one another (and the ring snapshots the ring's own
// history), within two ulps of the reference — under BOTH rope tables the
// engine can run them with, the plain one and the YaRN one the
// engine.rope_scaling knob builds; the indexer scores and the selection
// bitwise; the listed attention (one and three splits) with its gate
// within two ulps. Plus the frozen plain rope table and the YaRN cos/sin
// scale itself.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
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

// The two rope states a compressed index cache is ever built under. The
// plain one is the checkpoint's own table with mscale 1.0f — bit for bit
// what every QSA path ran before engine.rope_scaling existed. The YaRN one
// is what the knob builds for the 512K recipe: factor 2 over 262144, the
// correction band from the mrope-enlarged 4 x 262144, and the attention
// factor riding the cos/sin, exactly as
// qsa_rope_inv_freq_is_frozen_and_yarn_rides_the_cos_sin freezes them.
// Table and mscale travel together: one without the other is a state the
// engine never runs.
//
// The serving-path half of the reviewer's ask (the Qwen decode, CUDA
// graph, speculative-decoding, prefix-reuse and tensor-parallel checks
// with rope_scaling on) is a noted follow-up: those fixtures live in
// tests/cuda/qwen_*_test.cpp, and this PR's YaRN coverage there is the
// forward smoke test only.
struct RopeMode {
  const char* name;
  bool yarn = false;
};

constexpr RopeMode kPlain{"plain", false};
constexpr RopeMode kYarn{"yarn", true};

// The recipe's ramp as a spec — the same six fields the cluster config
// parses into cfg.rope_scaling.
dgpp::RopeScaling recipe() {
  dgpp::RopeScaling rs;
  rs.factor = 2.0;
  rs.original_max_position_embeddings = 262144;
  rs.beta_fast = 32.0;
  rs.beta_slow = 1.0;
  rs.attn_factor = 1.0;
  rs.mrope_cache_factor = 4.0;
  return rs;
}

std::vector<float> rope_table(const Geo& g, const RopeMode& m) {
  if (!m.yarn) return inv_freq_host(g);  // host bits, gated against the device's
  const dgpp::RopeScaling rs = recipe();
  std::vector<float> f(static_cast<size_t>(g.rotary / 2), 0.f);
  dgpp::yarn_rope_inv_freq_host(g.rotary, g.theta, rs.correction_max_position(), rs.factor,
                                rs.beta_fast, rs.beta_slow, f.data());
  return f;
}

float rope_mscale(const RopeMode& m) { return m.yarn ? recipe().mscale() : 1.0f; }

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
  RopeMode mode;
  float mscale = 1.0f;
  std::vector<uint16_t> raw;   // [seq, idx_dim]
  std::vector<uint16_t> w_k;   // [idx_dim]
  std::vector<float> inv;
  std::vector<int32_t> table;
  DevBuf draw, dwk, dinv, dtable;
  explicit IndexFixture(uint64_t seed, const RopeMode& m = kPlain)
      : mode(m), mscale(rope_mscale(m)) {
    raw = random_bf16_normal(seed, static_cast<int64_t>(g.seq) * g.idx_dim, 1.0f);
    w_k = random_bf16_uniform(seed + 1, g.idx_dim, 0.5f);
    inv = rope_table(g, mode);
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
                                         g.eps, mscale);
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
                                    g.pools_per_block(), g.kpool, g.idx_dim, g.rotary, g.eps, mscale, st,
                                    snapshots);
      DGPP_CUDA_OK(cudaStreamSynchronize(st));
    }
  }
};

// The fixture's whole sequence — prefill, token-by-token decode, the
// prefill-then-decode split with its ring snapshots, and the host oracle —
// under one rope state, returning a checksum of the cache it built so the
// caller can prove the two states really differ. Every invariant is a
// bitwise one; the only tolerance is the two-ulp reference comparison.
uint64_t index_compression_agreement(const RopeMode& mode) {
  IndexFixture f(10, mode);
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
                                 mptr<uint16_t>(c_prefill), g.kpool, g.idx_dim, g.rotary, g.eps,
                                 f.mscale, st);
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
                                 mptr<uint16_t>(c_split), g.kpool, g.idx_dim, g.rotary, g.eps,
                                 f.mscale, st);
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
                                  g.pools_per_block(), g.kpool, g.idx_dim, g.rotary, g.eps, f.mscale, st,
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
  std::printf("[ .. ] compressed keys (%s rope): max_rel %.3g l2_rel %.3g mismatches %ld/%ld\n",
              mode.name, s.max_rel, s.l2_rel, s.mismatches, s.n);
  require_bf16("compressed keys", s, 2e-3, 0.01);
  uint64_t sum = 1469598103934665603ull;  // FNV-1a over the cache this run built
  for (const uint16_t v : prefill) sum = (sum ^ static_cast<uint64_t>(v)) * 1099511628211ull;
  return sum;
}

}  // namespace

DGPP_TEST(qsa_index_compression_prefill_decode_and_split_agree) {
  // Both rope states, every invariant: the plain table with mscale 1.0f
  // (bit-identical to the build before the engine.rope_scaling knob — the
  // frozen table above is what keeps that honest) and the YaRN table with
  // the recipe's attention factor. The compression kernels take the table
  // and the mscale as arguments, so a fixture that only ever passed 1.0f
  // never touched the new ones on the prefill, decode, split or snapshot
  // path.
  const uint64_t plain = index_compression_agreement(kPlain);
  const uint64_t yarn = index_compression_agreement(kYarn);
  require(plain != yarn, "the YaRN run rebuilt the plain cache: the mode is inert");
  std::printf("[ .. ] compressed keys: prefill, decode, split and snapshots agree under both tables\n");
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

DGPP_TEST(qsa_visible_pool_bound_preserves_scores_selection_and_graphs) {
  SelectFixture sf(20);
  const Geo& g = sf.f.g;
  cudaStream_t stream = test_stream();
  // Leave many empty scoring stripes while keeping the same row stride.
  constexpr int64_t stride = 4096;
  constexpr size_t guard = 16;
  struct Result {
    std::vector<uint64_t> keys;
    std::vector<int32_t> selected, counts;
    bool operator==(const Result&) const = default;
  };
  const auto calculate = [&](int64_t bound, bool replay_graph = false) {
    const size_t count = static_cast<size_t>(sf.rows()) * stride;
    const size_t width = static_cast<size_t>(sf.rows()) * g.max_selected();
    DevBuf keys((count + 2 * guard) * sizeof(uint64_t));
    DevBuf selected(width * sizeof(int32_t)), counts(sf.rows() * sizeof(int32_t));
    DGPP_CUDA_OK(cudaMemsetAsync(keys.p, 0xff, keys.bytes, stream));
    const auto launch = [&] {
      dgpp::qsa_index_score(ptr<uint16_t>(sf.dq), int64_t(g.idx_heads) * g.idx_dim,
                            ptr<int32_t>(sf.dreq), ptr<int64_t>(sf.dpos), sf.rows(),
                            ptr<int32_t>(sf.f.dtable), g.blocks_per_request,
                            ptr<uint16_t>(sf.dcache), g.pools_per_block(), g.idx_heads,
                            g.idx_dim, g.kpool, mptr<uint64_t>(keys) + guard, stride, stream, bound);
      dgpp::qsa_select_from_keys(ptr<uint64_t>(keys) + guard, stride, ptr<int64_t>(sf.dpos),
                                 sf.rows(), g.select_k, g.kpool, g.max_selected(),
                                 mptr<int32_t>(selected), mptr<int32_t>(counts), stream);
    };
    if (replay_graph) {
      DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      launch();
      cudaGraph_t graph;
      cudaGraphExec_t exec;
      DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
      DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
      for (int repeat = 0; repeat < 2; ++repeat)
        DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      DGPP_CUDA_OK(cudaGraphExecDestroy(exec));
      DGPP_CUDA_OK(cudaGraphDestroy(graph));
    } else {
      launch();
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    }
    Result result{down<uint64_t>(keys, count + 2 * guard),
                  down<int32_t>(selected, width), down<int32_t>(counts, sf.rows())};
    for (size_t i = 0; i < guard; ++i)
      require(result.keys[i] == UINT64_MAX && result.keys[guard + count + i] == UINT64_MAX,
              "visible-pool scoring changed a workspace guard");
    return result;
  };
  const int64_t visible = (*std::max_element(sf.pos.begin(), sf.pos.end()) + 1) / g.kpool;
  const auto reference = calculate(-1);
  require(reference == calculate(visible), "visible-pool cap changed scores or selection");
  require(reference == calculate(visible, true), "visible-pool graph changed scores or selection");
  require(reference == calculate(stride), "full-stride cap changed scores or selection");

  // With no complete compressed key, selection still includes the token tail.
  sf.pos.assign(static_cast<size_t>(sf.rows()), g.kpool - 2);
  sf.pos.back() = -1;
  sf.dpos.upload(sf.pos.data(), sf.pos.size() * sizeof(int64_t));
  require(calculate(-1) == calculate(0, true), "zero-visible cap changed keys or token tails");
  for (int64_t invalid : {int64_t{-2}, stride + 1}) {
    bool threw = false;
    try {
      (void)calculate(invalid);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    require(threw, "invalid visible-pool bound accepted");
  }
}

DGPP_TEST(qsa_select_all_skips_keys_and_replays_across_the_budget) {
  constexpr int dim = 128, ppb = 16, kpool = 4, stride = 4096, requests = 2, guard = 16;
  constexpr int blocks = stride / ppb, rows = 12;
  cudaStream_t stream = test_stream();
  std::vector<int32_t> table(requests * blocks), req(rows);
  std::iota(table.begin(), table.end(), 0);
  std::mt19937 rng(20260930);
  std::shuffle(table.begin(), table.end(), rng);
  for (int r = 0; r < rows; ++r) req[r] = r % requests;
  const auto cache = random_bf16_normal(301, int64_t(requests) * stride * dim, 1.f);
  DevBuf dc = up(cache), dt = up(table), dr = up(req);
  for (int heads : {1, 2, 4}) {
    const auto q = random_bf16_normal(302, rows * heads * dim, 1.f);
    DevBuf dq = up(q);
    for (int select_k : {8, 512, 1024}) {
      const int width = select_k * kpool + kpool - 1;
      std::vector<int64_t> pos(rows, 0);
      DevBuf dp = up(pos), keys((size_t(rows) * stride + 2 * guard) * sizeof(uint64_t));
      DevBuf selected(rows * width * sizeof(int32_t)), counts(rows * sizeof(int32_t));
      const auto launch = [&](int64_t bound, int budget) {
        dgpp::qsa_index_score(ptr<uint16_t>(dq), heads * dim, ptr<int32_t>(dr), ptr<int64_t>(dp),
                              rows, ptr<int32_t>(dt), blocks, ptr<uint16_t>(dc), ppb, heads, dim,
                              kpool, mptr<uint64_t>(keys) + guard, stride, stream, bound, budget);
        dgpp::qsa_select_from_keys(ptr<uint64_t>(keys) + guard, stride, ptr<int64_t>(dp), rows,
                                   select_k, kpool, width, mptr<int32_t>(selected),
                                   mptr<int32_t>(counts), stream);
      };
      // Capture below the budget, then grow, shrink and deactivate different
      // rows. A host decision based on the capture positions would be wrong.
      cudaGraph_t graph;
      cudaGraphExec_t exec;
      DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      launch(-1, select_k);
      DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
      DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
      const std::vector<int64_t> cases{-1,
                                       0,
                                       2,
                                       3,
                                       (select_k - 1) * kpool - 1,
                                       select_k * kpool - 1,
                                       select_k * kpool + 2,
                                       select_k * kpool + 3,
                                       255 * kpool - 1,
                                       256 * kpool - 1,
                                       257 * kpool - 1,
                                       stride * kpool - 1};
      for (int phase = 0; phase < 5; ++phase) {
        for (int r = 0; r < rows; ++r) {
          if (phase == 0)
            pos[r] = r % kpool - 1;
          else if (phase == 1 || phase == 4)
            pos[r] = r == 0 ? -1 : select_k * kpool - 1 + r % kpool;
          else
            pos[r] = cases[(r + phase * 5) % rows];
          req[r] = pos[r] < 0 ? -1 : r % requests;
        }
        dp.upload(pos.data(), pos.size() * sizeof(int64_t));
        dr.upload(req.data(), req.size() * sizeof(int32_t));
        // Check both the graph's device decision and the prefill host bound.
        for (bool replay : {true, false}) {
          DGPP_CUDA_OK(cudaMemsetAsync(keys.p, 0xff, keys.bytes, stream));
          DGPP_CUDA_OK(cudaMemsetAsync(selected.p, 0x7f, selected.bytes, stream));
          if (replay) {
            DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
          } else {
            const int64_t bound = (*std::max_element(pos.begin(), pos.end()) + 1) / kpool;
            launch(bound, select_k);
          }
          DGPP_CUDA_OK(cudaStreamSynchronize(stream));
          const auto got_keys = down<uint64_t>(keys, size_t(rows) * stride + 2 * guard);
          const auto got = down<int32_t>(selected, rows * width);
          const auto got_counts = down<int32_t>(counts, rows);
          for (int i = 0; i < guard; ++i)
            require(got_keys[i] == UINT64_MAX &&
                        got_keys[guard + size_t(rows) * stride + i] == UINT64_MAX,
                    "select-all scoring overwrote a guard");
          for (int r = 0; r < rows; ++r) {
            const int visible = pos[r] < 0 ? 0 : int((pos[r] + 1) / kpool);
            std::vector<float> scores(visible);
            for (int pool = 0; pool < stride; ++pool) {
              uint64_t want = UINT64_MAX;
              if (pool < visible) {
                const int slot = table[req[r] * blocks + pool / ppb] * ppb + pool % ppb;
                scores[pool] = dgpp::qwen_ref::qsa_index_score(q.data() + r * heads * dim,
                                                               cache.data() + slot * dim, heads);
                if (visible > select_k) {
                  uint32_t bits;
                  std::memcpy(&bits, &scores[pool], sizeof(bits));
                  const uint32_t sortable = bits >> 31 ? ~bits : bits | 0x80000000u;
                  want = (static_cast<uint64_t>(~sortable) << 21) | pool;
                }
              }
              require(got_keys[guard + size_t(r) * stride + pool] == want,
                      "select-all wrote skipped keys or changed a required score");
            }
            std::vector<int32_t> ids, tokens;
            dgpp::qwen_ref::qsa_select(scores, select_k, ids);
            dgpp::qwen_ref::qsa_expand(ids, pos[r], kpool, tokens);
            require(got_counts[r] == static_cast<int>(tokens.size()), "select-all count differs");
            tokens.resize(width, -1);
            require(std::equal(tokens.begin(), tokens.end(), got.begin() + r * width),
                    "select-all tokens or padding differ from host oracle");
          }
        }
      }
      DGPP_CUDA_OK(cudaGraphExecDestroy(exec));
      DGPP_CUDA_OK(cudaGraphDestroy(graph));
      for (int invalid : {-1, 1025}) {
        bool threw = false;
        try {
          launch(-1, invalid);
        } catch (const std::invalid_argument& e) {
          threw = std::string(e.what()).find("select_k") != std::string::npos;
        }
        require(threw, "invalid select-all budget accepted or not named");
      }
    }
  }
}

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

DGPP_TEST(qsa_request_bounded_workspace_preserves_scores_selection_and_graphs) {
  constexpr int heads = 4, dim = 128, ppb = 16, kpool = 4, select_k = 512;
  constexpr int requests = 2, width = select_k * kpool + kpool - 1, guard = 16;
  cudaStream_t stream = test_stream();
  // Full prefill row capacity with short/tail positions, then the native
  // 256K ceiling and an unaligned ceiling. Physical blocks span the shared
  // pool, including addresses beyond the compact logical workspace stride.
  for (int context : {201, 262144, 262145}) {
    const int rows = context == 201 ? 4096 : 8;
    const int wide = context == 201 ? 4096 : 212512;
    const int compact = (context + kpool - 1) / kpool;
    const int blocks = (compact + ppb - 1) / ppb;
    std::vector<int32_t> physical(wide / ppb);
    std::iota(physical.begin(), physical.end(), 0);
    std::mt19937 rng(20260927);
    std::shuffle(physical.begin(), physical.end(), rng);
    std::vector<int32_t> table(physical.begin(), physical.begin() + requests * blocks);
    const std::vector<int64_t> cases{-1, 0, 2, 3, context / 2, context - 3,
                                     context - 2, context - 1};
    std::vector<int64_t> pos(rows);
    std::vector<int32_t> req(rows);
    for (int r = 0; r < rows; ++r) {
      pos[r] = cases[static_cast<size_t>(r) % cases.size()];
      req[r] = pos[r] < 0 ? -1 : r % requests;
    }
    const auto cache = random_bf16_normal(202, int64_t(wide) * dim, 1.0f);
    const auto q = random_bf16_normal(203, int64_t(rows) * heads * dim, 1.0f);
    DevBuf dc = up(cache), dq = up(q), dt = up(table), dp = up(pos), dr = up(req);
    struct Result {
      std::vector<uint64_t> keys;
      std::vector<int32_t> selected, counts;
    };
    const auto calculate = [&](int stride, bool replay_graph) {
      const size_t nkeys = size_t(rows) * stride;
      DevBuf keys((nkeys + 2 * guard) * sizeof(uint64_t));
      DevBuf selected(size_t(rows) * width * sizeof(int32_t)), counts(rows * sizeof(int32_t));
      DGPP_CUDA_OK(cudaMemsetAsync(keys.p, 0xff, keys.bytes, stream));
      const auto launch = [&] {
        dgpp::qsa_index_score(ptr<uint16_t>(dq), heads * dim, ptr<int32_t>(dr),
                             ptr<int64_t>(dp), rows, ptr<int32_t>(dt), blocks,
                             ptr<uint16_t>(dc), ppb, heads, dim, kpool,
                             mptr<uint64_t>(keys) + guard, stride, stream);
        dgpp::qsa_select_from_keys(ptr<uint64_t>(keys) + guard, stride, ptr<int64_t>(dp),
                                  rows, select_k, kpool, width, mptr<int32_t>(selected),
                                  mptr<int32_t>(counts), stream);
      };
      launch();
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      Result result{down<uint64_t>(keys, nkeys + 2 * guard),
                    down<int32_t>(selected, size_t(rows) * width),
                    down<int32_t>(counts, rows)};
      for (int i = 0; i < guard; ++i)
        require(result.keys[i] == UINT64_MAX && result.keys[guard + nkeys + i] == UINT64_MAX,
                "scoring changed a workspace guard");
      for (int r = 0; r < rows; ++r) {
        const int visible = pos[r] < 0 ? 0 : int((pos[r] + 1) / kpool);
        for (int p = visible; p < stride; ++p)
          require(result.keys[guard + size_t(r) * stride + p] == UINT64_MAX,
                  "scoring changed an invisible key or inactive row");
      }
      if (replay_graph) {
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch();
        DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
        DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int repeat = 0; repeat < 2; ++repeat) {
          DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
          DGPP_CUDA_OK(cudaStreamSynchronize(stream));
          require(down<uint64_t>(keys, nkeys + 2 * guard) == result.keys,
                  "compact graph changed scoring keys or guards");
          require(down<int32_t>(selected, size_t(rows) * width) == result.selected &&
                      down<int32_t>(counts, rows) == result.counts,
                  "compact graph changed selected tokens");
        }
        DGPP_CUDA_OK(cudaGraphExecDestroy(exec));
        DGPP_CUDA_OK(cudaGraphDestroy(graph));
      }
      return result;
    };
    const auto reference = calculate(wide, false);
    const auto bounded = calculate(compact, true);
    require(reference.selected == bounded.selected && reference.counts == bounded.counts,
            "request-bounded workspace changed selected tokens");
    for (int r = 0; r < rows; ++r)
      for (int p = 0; p < compact; ++p)
        require(reference.keys[guard + size_t(r) * wide + p] ==
                    bounded.keys[guard + size_t(r) * compact + p],
                "request-bounded workspace changed scoring keys");
    std::printf("[ .. ] %d rows, context %d, stride %d -> %d: keys, selection and graphs exact\n",
                rows, context, wide, compact);
  }
}

DGPP_TEST(qsa_index_large_paged_pools_match_every_host_key_and_token) {
  // Preserve the reviewer's 65322-pool / position-261288 case as a normal
  // gate, then cross the 128K-pool boundary. Neither the score stripe nor
  // the old selection tile divides these visible counts evenly.
  constexpr int heads = 4, dim = 128, ppb = 16, kpool = 4, select_k = 512;
  constexpr int rows = 3, width = select_k * kpool + kpool - 1;
  cudaStream_t stream = test_stream();
  for (int pools : {65322, 131071}) {
    const int blocks = (pools + ppb - 1) / ppb;
    const int stride = blocks * ppb;
    std::vector<int32_t> table(blocks);
    std::iota(table.begin(), table.end(), 0);
    std::mt19937 rng(20260921);
    std::shuffle(table.begin(), table.end(), rng);
    const auto cache = random_bf16_normal(200, int64_t(stride) * dim, 1.0f);
    const auto q = random_bf16_normal(201, rows * heads * dim, 1.0f);
    std::vector<int64_t> pos{int64_t(pools) * kpool, int64_t(pools) * kpool - 2, -1};
    DevBuf dc = up(cache), dq = up(q), dt = up(table), dp = up(pos);
    DevBuf dr = up(std::vector<int32_t>(rows, 0));
    DevBuf keys(size_t(rows) * stride * sizeof(uint64_t));
    DevBuf selected(rows * width * sizeof(int32_t)), counts(rows * sizeof(int32_t));
    DGPP_CUDA_OK(cudaMemsetAsync(keys.p, 0xff, keys.bytes, stream));
    auto launch = [&] {
      dgpp::qsa_index_score(ptr<uint16_t>(dq), heads * dim, ptr<int32_t>(dr), ptr<int64_t>(dp),
                            rows, ptr<int32_t>(dt), blocks, ptr<uint16_t>(dc), ppb, heads, dim,
                            kpool, mptr<uint64_t>(keys), stride, stream);
      dgpp::qsa_select_from_keys(ptr<uint64_t>(keys), stride, ptr<int64_t>(dp), rows, select_k,
                                 kpool, width, mptr<int32_t>(selected), mptr<int32_t>(counts),
                                 stream);
    };
    launch();
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    const auto got_keys = down<uint64_t>(keys, size_t(rows) * stride);
    const auto got_selected = down<int32_t>(selected, rows * width);
    const auto got_counts = down<int32_t>(counts, rows);
    for (int r = 0; r < rows; ++r) {
      const int visible = pos[r] < 0 ? 0 : int((pos[r] + 1) / kpool);
      std::vector<float> scores(visible);
      for (int p = 0; p < visible; ++p) {
        const int slot = table[p / ppb] * ppb + p % ppb;
        const float score =
            dgpp::qwen_ref::qsa_index_score(q.data() + r * heads * dim, cache.data() + slot * dim);
        scores[p] = score;
        uint32_t bits;
        std::memcpy(&bits, &score, sizeof(bits));
        const uint32_t sortable = bits >> 31 ? ~bits : bits | 0x80000000u;
        const uint64_t key = (uint64_t(~sortable) << 21) | uint64_t(p);
        require(got_keys[size_t(r) * stride + p] == key, "large paged score differs from host");
      }
      for (int p = visible; p < stride; ++p)
        require(got_keys[size_t(r) * stride + p] == UINT64_MAX,
                "score overwrote an invisible pool");
      std::vector<int32_t> ids, tokens;
      dgpp::qwen_ref::qsa_select(scores, select_k, ids);
      if (pos[r] >= 0) dgpp::qwen_ref::qsa_expand(ids, pos[r], kpool, tokens);
      require(got_counts[r] == int(tokens.size()), "large paged selected count differs");
      tokens.resize(width, -1);
      require(std::equal(tokens.begin(), tokens.end(), got_selected.begin() + r * width),
              "large paged selected tokens differ from host");
    }
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch();
    DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
    DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int repeat = 0; repeat < 3; ++repeat) {
      DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      require(down<uint64_t>(keys, size_t(rows) * stride) == got_keys, "large graph score changed");
      require(down<int32_t>(selected, rows * width) == got_selected,
              "large graph selection changed");
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    std::printf(
        "[ .. ] %d pools, 64-token permuted paging: every key/token and graph replay exact\n",
        pools);
  }
}

DGPP_TEST(qsa_select_long_ties_boundaries_and_graph_shape_changes) {
  constexpr int rows = 8, stride = 131073, kpool = 4;
  cudaStream_t stream = test_stream();
  std::vector<uint64_t> keys(size_t(rows) * stride);
  const std::vector<int> visible{0, 511, 512, 2048, 2049, 65322, 131071, stride};
  std::vector<int64_t> pos(rows);
  DevBuf dkeys(keys.size() * sizeof(uint64_t)), dpos(rows * sizeof(int64_t));
  for (int select_k : {8, 512, 1024}) {
    const int width = select_k * kpool + kpool - 1;
    DevBuf selected(rows * width * sizeof(int32_t)), counts(rows * sizeof(int32_t));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    dgpp::qsa_select_from_keys(ptr<uint64_t>(dkeys), stride, ptr<int64_t>(dpos), rows, select_k,
                               kpool, width, mptr<int32_t>(selected), mptr<int32_t>(counts),
                               stream);
    DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
    DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int pattern = 0; pattern < 4; ++pattern) {
      // Equal scores force refinement into pool-id bits. Reverse scores
      // put all winners in the final partial tile. Rotation and an inactive
      // row change the work beneath the same captured graph on every replay.
      for (int r = 0; r < rows; ++r) {
        const int n = visible[(r + pattern) % rows];
        pos[r] = pattern == 3 && r == 0 ? -1 : int64_t(n) * kpool + r % kpool - 1;
        for (int p = 0; p < stride; ++p) {
          const float score = pattern == 0   ? 0.f
                              : pattern == 1 ? float(p)
                                             : float((p * 73 + r) % 257);
          uint32_t bits;
          std::memcpy(&bits, &score, sizeof(bits));
          keys[size_t(r) * stride + p] = (uint64_t(~(bits | 0x80000000u)) << 21) | uint64_t(p);
        }
      }
      dkeys.upload(keys.data(), keys.size() * sizeof(uint64_t));
      dpos.upload(pos.data(), pos.size() * sizeof(int64_t));
      DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      const auto got = down<int32_t>(selected, rows * width),
                 got_counts = down<int32_t>(counts, rows);
      for (int r = 0; r < rows; ++r) {
        const int n = pos[r] < 0 ? 0 : int((pos[r] + 1) / kpool);
        std::vector<uint64_t> ordered(keys.begin() + size_t(r) * stride,
                                      keys.begin() + size_t(r) * stride + n);
        std::sort(ordered.begin(), ordered.end());
        ordered.resize(std::min(n, select_k));
        std::vector<int32_t> ids;
        for (uint64_t key : ordered) ids.push_back(int32_t(key & ((1u << 21) - 1)));
        std::sort(ids.begin(), ids.end());
        std::vector<int32_t> want;
        if (pos[r] >= 0) dgpp::qwen_ref::qsa_expand(ids, pos[r], kpool, want);
        require(got_counts[r] == int(want.size()), "tie/edge count differs");
        want.resize(width, -1);
        require(std::equal(want.begin(), want.end(), got.begin() + r * width),
                "tie/edge selection differs: pattern=" + std::to_string(pattern) +
                    " row=" + std::to_string(r));
      }
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
  }
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
                        mptr<uint16_t>(kc), mptr<uint16_t>(vc), nullptr, nullptr, st);
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
                             mptr<float>(l_ws), mptr<float>(c_ws), st, nullptr, nullptr);
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
  // The decode kernel's two gather forms (the cp.async phases and the
  // serial gather) leave bitwise the same partials: the arithmetic is the
  // same sequence either way.
  for (int n_split : {1, 3}) {
    const size_t part = static_cast<size_t>(rows) * n_split * g.local_heads;
    std::vector<uint8_t> got[2];
    for (int form = 0; form < 2; ++form) {
      DevBuf m_ws(part * 4), l_ws(part * 4), c_ws(part * g.dim * 4);
      dgpp::qsa_attn_partial_gather(ptr<uint16_t>(dqonly), static_cast<int64_t>(g.local_heads) * g.dim, ptr<uint16_t>(kc),
                                    ptr<uint16_t>(vc), ptr<int32_t>(sf.dreq), ptr<int32_t>(dtopk), g.max_selected(),
                                    ptr<int32_t>(dcounts), rows, n_split, g.local_heads, g.kv_heads, g.dim,
                                    g.block_tokens, ptr<int32_t>(sf.f.dtable), g.blocks_per_request, 1.0f / 16.0f,
                                    mptr<float>(m_ws), mptr<float>(l_ws), mptr<float>(c_ws), st, form, 0, 0);
      DGPP_CUDA_OK(cudaStreamSynchronize(st));
      const std::vector<uint8_t> m = down<uint8_t>(m_ws, part * 4), l = down<uint8_t>(l_ws, part * 4),
                                 c = down<uint8_t>(c_ws, part * g.dim * 4);
      got[form].insert(got[form].end(), m.begin(), m.end());
      got[form].insert(got[form].end(), l.begin(), l.end());
      got[form].insert(got[form].end(), c.begin(), c.end());
    }
    require_bitwise("async gather partials == serial gather partials, n_split=" + std::to_string(n_split), got[0].data(),
                    got[1].data(), got[0].size());
    // Every head of a kv head in one block, several heads per warp: the
    // same partials, bitwise (the fixture's three heads per kv head as one
    // warp of three, and as a block of three one-head warps).
    for (const int hpw : {1, 3}) {
      DevBuf m_ws(part * 4), l_ws(part * 4), c_ws(part * g.dim * 4);
      dgpp::qsa_attn_partial_gather(ptr<uint16_t>(dqonly), static_cast<int64_t>(g.local_heads) * g.dim, ptr<uint16_t>(kc),
                                    ptr<uint16_t>(vc), ptr<int32_t>(sf.dreq), ptr<int32_t>(dtopk), g.max_selected(),
                                    ptr<int32_t>(dcounts), rows, n_split, g.local_heads, g.kv_heads, g.dim,
                                    g.block_tokens, ptr<int32_t>(sf.f.dtable), g.blocks_per_request, 1.0f / 16.0f,
                                    mptr<float>(m_ws), mptr<float>(l_ws), mptr<float>(c_ws), st, 1,
                                    g.local_heads / g.kv_heads, hpw);
      DGPP_CUDA_OK(cudaStreamSynchronize(st));
      std::vector<uint8_t> got2;
      const std::vector<uint8_t> m = down<uint8_t>(m_ws, part * 4), l = down<uint8_t>(l_ws, part * 4),
                                 c = down<uint8_t>(c_ws, part * g.dim * 4);
      got2.insert(got2.end(), m.begin(), m.end());
      got2.insert(got2.end(), l.begin(), l.end());
      got2.insert(got2.end(), c.begin(), c.end());
      require_bitwise("heads-per-kv block, " + std::to_string(hpw) + " heads per warp == serial partials, n_split=" +
                          std::to_string(n_split),
                      got2.data(), got[0].data(), got2.size());
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
           mptr<float>(candidate ? c1 : c0), stream, nullptr, nullptr);
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

// The one-warp tensor-core prefill kernel against the partial kernel +
// combine it replaces, row by row: representative GQA shapes (1-16 query
// heads per KV head), empty/short/tile-edge/long lists, a high-dynamic-range
// query, the engine's 64-token and a 256-token cache block, scrambled block
// tables, two requests, 1/3/8 reference splits (production uses up to 8),
// and a graph replay. P V runs on bf16 probabilities and
// the dots in the tensor cores' order, so the bound is a tolerance: rel l2
// per (row, head) under 1e-2, empty lists exactly zero in both.
DGPP_TEST(qsa_warp_prefill_matches_the_partial_kernels) {
  constexpr int dim = 256, requests = 2, stride = 2051;
  const std::vector<int32_t> counts{0, 1, 7, 15, 16, 17, 31, 32, 33, 64, 255, 256, 257, 1000, 2048, 2051};
  const int rows = static_cast<int>(counts.size());
  cudaStream_t stream = test_stream();
  double worst_overall = 0.0;
  for (const int block_tokens : {64, 256}) {
    const int blocks = (3 * stride + 2 * rows) / block_tokens + 1;  // covers every listed position
    std::vector<int32_t> req(rows), topk(rows * stride, -1), table(blocks * requests);
    for (int i = 0; i < blocks * requests; ++i) table[i] = (i * 17 + 3) % (blocks * requests);
    for (int r = 0; r < rows; ++r) {
      req[r] = r % requests;
      for (int i = 0; i < counts[r]; ++i) topk[r * stride + i] = i * 3 + r;
    }
    DevBuf dreq = up(req), dtopk = up(topk), dcounts = up(counts), dtable = up(table);
    for (auto shape : {std::pair{12, 1}, std::pair{16, 1}, std::pair{6, 1}, std::pair{24, 2},
                       std::pair{6, 2}, std::pair{2, 2}, std::pair{1, 1}}) {
      const int heads = shape.first, kv_heads = shape.second, qstride = heads * dim + 8;
      require(dgpp::qsa_warp_supported(dim, heads, kv_heads), "shape outside the warp kernel's envelope");
      auto q = random_bf16_normal(800 + heads, static_cast<int64_t>(rows) * qstride, 1.0f);
      const int64_t kv_elems = static_cast<int64_t>(blocks * requests * block_tokens) * kv_heads * dim;
      auto k = random_bf16_normal(810 + kv_heads, kv_elems, 1.0f);
      auto v = random_bf16_normal(820 + kv_heads, kv_elems, 1.0f);
      for (int d = 0; d < heads * dim; ++d)  // repeated maximum rescaling
        q[static_cast<size_t>(rows - 1) * qstride + d] =
            float_to_bf16_bits(bf16_bits_to_float(q[static_cast<size_t>(rows - 1) * qstride + d]) * 32);
      DevBuf dq = up(q), dk = up(k), dv = up(v);
      for (const int splits : {1, 3, 8}) {
        const size_t part = static_cast<size_t>(rows) * heads;
        DevBuf m(part * splits * 4), l(part * splits * 4), c(part * splits * dim * 4),
            ref(part * dim * 4), got(part * dim * 4);
        dgpp::qsa_attn_prefill_partial(ptr<uint16_t>(dq), qstride, ptr<uint16_t>(dk), ptr<uint16_t>(dv),
                                       ptr<int32_t>(dreq), ptr<int32_t>(dtopk), stride,
                                       ptr<int32_t>(dcounts), rows, splits, heads, kv_heads, dim,
                                       block_tokens, ptr<int32_t>(dtable), blocks, 1.0f / 16,
                                       mptr<float>(m), mptr<float>(l), mptr<float>(c), stream);
        dgpp::dsa_attn_combine(ptr<float>(m), ptr<float>(l), ptr<float>(c), rows, splits, heads, dim,
                               mptr<float>(ref), stream);
        auto warp = [&] {
          dgpp::qsa_attn_prefill_warp(ptr<uint16_t>(dq), qstride, ptr<uint16_t>(dk), ptr<uint16_t>(dv),
                                    ptr<int32_t>(dreq), ptr<int32_t>(dtopk), stride,
                                    ptr<int32_t>(dcounts), rows, heads, kv_heads, block_tokens,
                                    ptr<int32_t>(dtable), blocks, 1.0f / 16, mptr<float>(got), stream);
        };
        DGPP_CUDA_OK(cudaMemsetAsync(got.p, 0xff, got.bytes, stream));
        warp();
        DGPP_CUDA_OK(cudaStreamSynchronize(stream));
        const auto want = down<float>(ref, part * dim);
        auto check = [&](const char* what) {
          const auto have = down<float>(got, part * dim);
          const std::string tag = std::string(what) + " splits=" + std::to_string(splits) +
                                  " bt=" + std::to_string(block_tokens) +
                                  " heads=" + std::to_string(heads) + " kv=" + std::to_string(kv_heads);
          for (int r = 0; r < rows; ++r)
            for (int h = 0; h < heads; ++h) {
              const size_t o = (static_cast<size_t>(r) * heads + h) * dim;
              double num = 0.0, den = 0.0;
              for (int d = 0; d < dim; ++d) {
                require(std::isfinite(have[o + d]), tag + ": non-finite output");
                const double e = static_cast<double>(have[o + d]) - want[o + d];
                num += e * e;
                den += static_cast<double>(want[o + d]) * want[o + d];
              }
              if (counts[r] == 0) {
                require(num == 0.0, tag + ": an empty list must write the combine's output exactly");
                continue;
              }
              const double rel = std::sqrt(num / den);
              worst_overall = std::max(worst_overall, rel);
              require(rel < 1e-2, tag + ": row " + std::to_string(r) + " head " + std::to_string(h) +
                                      " rel l2 " + std::to_string(rel));
            }
        };
        check("eager");
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        warp();
        DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
        DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        DGPP_CUDA_OK(cudaMemsetAsync(got.p, 0xff, got.bytes, stream));
        DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
        DGPP_CUDA_OK(cudaStreamSynchronize(stream));
        check("graph");
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
      }
    }
  }
  std::printf("[ .. ] warp vs partial+combine: worst per-(row, head) rel l2 %.3g\n", worst_overall);
}

int main() { return dgpp::test::run_all(); }
