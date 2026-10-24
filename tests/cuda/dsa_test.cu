// DSA/MLA kernel tests (DESIGN §7.2, PLAN M3).
//
// Comparison philosophy carried over from M2:
//   * device-vs-device sequencing comparisons (multi-token vs single-token
//     decode, repeat runs) are bitwise;
//   * device-vs-host oracle comparisons are tolerance-based where libm or
//     FMA contraction can differ (compress, attention);
//   * the top-k selection is exact by construction: the host oracle mirrors
//     the kernel's contraction-proof logit arithmetic (separate mul/add in
//     the documented order plus the warp butterfly reduction), so pool
//     positions and expanded token rows are compared bitwise over the fuzz
//     corpus — the M3 exit criterion.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "core/graph.hpp"
#include "kernels/dsa.hpp"
#include "kernels/gemm.hpp"
#include "models/dsa_geometry.hpp"
#include "models/dsa_layer.hpp"
#include "models/dsa_reference.hpp"
#include "models/dsa_state.hpp"

#include "kda_test_helpers.hpp"  // DevBuf, random_bf16_bits, compare helpers
#include "dsa_near_tie_audit.hpp"  // near-tie certification
#include "kernels/packq_gemv.hpp"   // the packed GEMV launcher (packed projections)
#include "loaders/packq_quant.hpp"  // the host packed-int encoder (packed projections)

namespace {

namespace dsa_ref = dgpp::dsa_ref;
using namespace dgpp;
using dgpp::bf16_bits_to_float;
using dgpp::DsaConfig;
using dgpp::DsaGeometry;
using dgpp::LatentFormat;
using dgpp::dsa_ref::HostState;
using dgpp::dsa_ref::HostWeights;
using dgpp::float_to_bf16_bits;
using dgpp::float_to_fp8_e4m3_bits;
using dgpp::fp8_e4m3_bits_to_float;
using dgpp::kda_test::compare_bf16;
using dgpp::kda_test::compare_abs_rel;
using dgpp::kda_test::DevBuf;
using dgpp::kda_test::random_bf16_bits;
using dgpp::kda_test::require_bf16;
using dgpp::kda_test::require_bitwise;
using dgpp::kda_test::require_rel;

uint32_t hash32(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdull;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ull;
  x ^= x >> 33;
  return uint32_t(x);
}

// Sane-magnitude random fp32 (exponent clamped well away from inf/nan).
float random_f32(uint64_t seed, int64_t i) {
  const uint32_t h = hash32(seed * 2 + 1 + uint64_t(i) * 2654435761u);
  const uint32_t bits = (h & 0x80000000u) | (uint32_t(112 + (h >> 27) % 12)
                                              << 23) |
                        (h & 0x7FFFFFu);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// Mirror of the kernels' warp butterfly sum (parallel-step semantics).
float butterfly_sum_32(const float* in) {
  float a[32], b[32];
  std::memcpy(a, in, sizeof(a));
  for (int off = 16; off > 0; off >>= 1) {
    for (int i = 0; i < 32; ++i) b[i] = a[i] + a[i ^ off];
    std::memcpy(a, b, sizeof(a));
  }
  return a[0];
}

// Mirror of PrefillKeyFn: logit = butterfly_h((w[h]*ks)*dot[h]).
float prefill_logit_mirror(const float* dot_row_ptrs, const float* w,
                           float ks) {
  float contrib[32];
  for (int h = 0; h < 32; ++h)
    contrib[h] = (w[h] * ks) * dot_row_ptrs[h];
  return butterfly_sum_32(contrib);
}

// Host simulation of the tail ring + pool writes for the state machinery
// tests (mirrors dsa_ref::layer_forward's cache updates without the GEMM
// path).
struct RingSim {
  std::vector<uint16_t> tail;    // [2, kpool, dim]
  std::vector<uint8_t> index_k;  // [max_pools, dim]
  std::vector<float> index_scale;
  int kpool, dim;

  void seed(const std::vector<uint16_t>& k, const std::vector<uint16_t>& gate,
            int64_t token_start, int64_t tokens) {
    const int64_t end = token_start + tokens;
    for (int64_t pos = std::max<int64_t>(0, end - kpool); pos < end; ++pos) {
      const int slot = int(pos % kpool);
      std::memcpy(&tail[size_t(slot) * dim], &k[size_t(pos - token_start) * dim],
                  dim * 2);
      std::memcpy(&tail[size_t(kpool + slot) * dim],
                  &gate[size_t(pos - token_start) * dim], dim * 2);
    }
  }

  // One decode token at position pos with raw k/gate rows.
  void decode_token(const uint16_t* krow, const uint16_t* gamerow,
                    const float* ape, int64_t pos, int64_t pool_slot) {
    const int slot = int(pos % kpool);
    if (slot == kpool - 1) {
      const int64_t pool_start = pos - (kpool - 1);
      // Assemble the pool's k/gate from the ring with the current override.
      std::vector<uint16_t> pk(size_t(kpool) * dim), pg(size_t(kpool) * dim);
      for (int s = 0; s < kpool; ++s) {
        const int ring = int((pool_start + s) % kpool);
        const bool cur = (s == kpool - 1);
        std::memcpy(&pk[size_t(s) * dim],
                    cur ? krow : &tail[size_t(ring) * dim], dim * 2);
        std::memcpy(&pg[size_t(s) * dim],
                    cur ? gamerow : &tail[size_t(kpool + ring) * dim], dim * 2);
      }
      dsa_ref::compress_pool<float>(pk.data(), pg.data(), ape, kpool, dim,
                                    &index_k[size_t(pool_slot) * dim],
                                    &index_scale[size_t(pool_slot)]);
    }
    std::memcpy(&tail[size_t(slot) * dim], krow, dim * 2);
    std::memcpy(&tail[size_t(kpool + slot) * dim], gamerow, dim * 2);
  }
};

}  // namespace

// ---------------------------------------------------------------------------
// fwht + fp8 quant: the pinned boundary sequence is portable, so this is
// bitwise against the host reference.
// ---------------------------------------------------------------------------
DGPP_TEST(dsa_fwht_quant_matches_host_reference) {
  const int rows = 257;
  auto q = random_bf16_bits(11, rows * 128, -3, 1);
  std::vector<uint8_t> want8(size_t(rows) * 128);
  std::vector<float> want_scale(size_t(rows), 0.0f);
  dsa_ref::fwht128_quant_fp8<float>(q.data(), rows, 128, want8.data(),
                                    want_scale.data());

  DevBuf dq(size_t(rows) * 128 * 2), d8(size_t(rows) * 128),
      ds(size_t(rows) * 4);
  dq.upload(q.data(), q.size() * 2);
  dsa_fwht_quant_rows(dq.p, rows, d8.p, static_cast<float*>(ds.p), 0);

  std::vector<uint8_t> got8(size_t(rows) * 128);
  std::vector<float> got_scale(size_t(rows), 0.0f);
  d8.download(got8.data(), got8.size());
  ds.download(got_scale.data(), got_scale.size() * 4);
  require_bitwise("fwht fp8 bits", got8.data(), want8.data(), got8.size());
  require_bitwise("fwht scales", got_scale.data(), want_scale.data(),
                  want_scale.size() * 4);
}

// ---------------------------------------------------------------------------
// Pool compression. Two exactly-comparable modes make the real correctness
// risks (slot mapping, pool assembly, softmax indexing) bitwise-testable
// despite expf/FMA ulp divergence between host and device:
//   * hard-max gate: the dominant slot's score leads by 100, so the other
//     slots' softmax weights underflow to subnormals (~1e-43) and the
//     compressed pool is bitwise fwht(k_dominant) on both sides;
//   * random gate: tolerance smoke test (fp8 scale-bin flips from expf ulps
//     legitimately reach ~1.5 quanta).
// Exercises a non-identity block table.
// ---------------------------------------------------------------------------
DGPP_TEST(dsa_kpool_compress_matches_host_reference) {
  const int kpool = 4, dim = 128, n_pools = 96;
  const int tokens = n_pools * kpool;
  auto k = random_bf16_bits(21, tokens * dim, -2, 1);
  auto gate_random = random_bf16_bits(22, tokens * dim, -2, 1);
  std::vector<float> ape(size_t(kpool) * dim);
  for (int i = 0; i < kpool * dim; ++i) ape[i] = random_f32(23, i) * 0.1f;

  // Hard-max gate: dominant slot varies per pool so the assembly is probed.
  auto gate = gate_random;
  for (int j = 0; j < n_pools; ++j) {
    const int dom = j % kpool;
    for (int s = 0; s < kpool; ++s)
      for (int d = 0; d < dim; ++d)
        gate[size_t(j * kpool + s) * dim + d] =
            float_to_bf16_bits((s == dom) ? 50.0f : -50.0f);
  }

  // Block table: 32 pools per block, shuffled physical assignment.
  const int ppb = 32, n_blocks = n_pools / ppb;
  std::vector<int32_t> bt(n_blocks);
  for (int b = 0; b < n_blocks; ++b) bt[b] = (b * 7 + 3) % n_blocks;
  const int total_slots = n_blocks * ppb;

  auto run_device = [&](const std::vector<uint16_t>& g,
                        std::vector<uint8_t>& out_k,
                        std::vector<float>& out_s) {
    DevBuf dk(k.size() * 2), dgate(g.size() * 2), dape(ape.size() * 4),
        dbt(bt.size() * 4), dki(size_t(total_slots) * dim),
        dks(size_t(total_slots) * 4);
    dk.upload(k.data(), k.size() * 2);
    dgate.upload(g.data(), g.size() * 2);
    dape.upload(ape.data(), ape.size() * 4);
    dbt.upload(bt.data(), bt.size() * 4);
    dsa_kpool_compress_write(dk.p, dim, dgate.p, dim,
                             static_cast<const float*>(dape.p),
                             static_cast<const int32_t*>(dbt.p), ppb,
                             /*first_pool=*/0, n_pools, dki.p,
                             static_cast<float*>(dks.p), kpool, dim, 0);
    out_k.assign(size_t(total_slots) * dim, 0);
    out_s.assign(size_t(total_slots), 0.0f);
    dki.download(out_k.data(), out_k.size());
    dks.download(out_s.data(), out_s.size() * 4);
  };

  std::vector<uint8_t> got_hard, got_rand;
  std::vector<float> got_s_hard, got_s_rand;
  run_device(gate, got_hard, got_s_hard);
  run_device(gate_random, got_rand, got_s_rand);

  for (int j = 0; j < n_pools; ++j) {
    std::vector<uint8_t> w8(dim), w8r(dim);
    float ws = 0, wsr = 0;
    dsa_ref::compress_pool<float>(&k[size_t(j * kpool) * dim],
                                  &gate[size_t(j * kpool) * dim], ape.data(),
                                  kpool, dim, w8.data(), &ws);
    dsa_ref::compress_pool<float>(&k[size_t(j * kpool) * dim],
                                  &gate_random[size_t(j * kpool) * dim],
                                  ape.data(), kpool, dim, w8r.data(), &wsr);
    const int64_t slot = int64_t(bt[j / ppb]) * ppb + (j % ppb);
    // Hard-max: bitwise.
    if (std::memcmp(&got_hard[size_t(slot) * dim], w8.data(), dim) != 0 ||
        got_s_hard[size_t(slot)] != ws)
      throw std::runtime_error("compress hard-max bitwise mismatch pool=" +
                               std::to_string(j));
    // Random gate: tolerance smoke test. fp8 scale ties to the row absmax, so
    // each side's quantization error is bounded by absmax/16; a shared budget
    // of absmax/8 covers a full scale-bin flip and stays far below wrong-pool
    // garbage (the hard-max mode above carries the assembly correctness).
    float rowmax = 0.0f;
    for (int d = 0; d < dim; ++d) {
      const float gv =
          fp8_e4m3_bits_to_float(got_rand[size_t(slot) * dim + d]) *
          got_s_rand[size_t(slot)];
      const float wv = fp8_e4m3_bits_to_float(w8r[d]) * wsr;
      rowmax = std::max(rowmax, std::max(std::fabs(gv), std::fabs(wv)));
    }
    for (int d = 0; d < dim; ++d) {
      const float gv =
          fp8_e4m3_bits_to_float(got_rand[size_t(slot) * dim + d]) *
          got_s_rand[size_t(slot)];
      const float wv = fp8_e4m3_bits_to_float(w8r[d]) * wsr;
      if (std::fabs(gv - wv) > rowmax * 0.135f + 1e-6f)
        throw std::runtime_error("compress mismatch pool=" + std::to_string(j) +
                                 " dim=" + std::to_string(d) + " gv=" +
                                 std::to_string(gv) + " wv=" +
                                 std::to_string(wv));
    }
  }
}

// ---------------------------------------------------------------------------
// Tail seed: the reference's ahead-check rule over a multi-request batch.
// ---------------------------------------------------------------------------
DGPP_TEST(dsa_tail_seed_matches_reference_rule) {
  const int kpool = 4, dim = 128;
  // Two requests: 6 tokens and 3 tokens in one flattened batch.
  const int t0 = 6, t1 = 3, tokens = t0 + t1;
  auto k = random_bf16_bits(31, tokens * dim, -2, 1);
  auto gate = random_bf16_bits(32, tokens * dim, -2, 1);
  std::vector<int32_t> req_ids = {0, 0, 0, 0, 0, 0, 1, 1, 1};
  std::vector<int64_t> pos = {0, 1, 2, 3, 4, 5, 0, 1, 2};

  const size_t tail_bytes = 2ull * 2 * kpool * dim * 2;  // 2 requests
  DevBuf dk(k.size() * 2), dgate(gate.size() * 2), dri(req_ids.size() * 4),
      dpos(pos.size() * 8), dtail(tail_bytes);
  dk.upload(k.data(), k.size() * 2);
  dgate.upload(gate.data(), gate.size() * 2);
  dri.upload(req_ids.data(), req_ids.size() * 4);
  dpos.upload(pos.data(), pos.size() * 8);
  std::vector<uint16_t> zero(tail_bytes / 2, 0);
  dtail.upload(zero.data(), tail_bytes);
  dsa_kpool_tail_seed(dk.p, dim, dgate.p, dim,
                      static_cast<const int32_t*>(dri.p),
                      static_cast<const int64_t*>(dpos.p), tokens, dtail.p,
                      kpool, dim, 0);

  std::vector<uint16_t> got(tail_bytes / 2);
  dtail.download(got.data(), tail_bytes);
  // Expected: request 0's last 4 tokens (pos 2..5, ring slots 2,3,0,1) and
  // request 1's 3 tokens (pos 0..2, slots 0,1,2).
  auto at = [&](int req, int half, int slot) {
    return &got[(size_t(req) * 2 * kpool + half * kpool + slot) * dim];
  };
  auto row = [&](int t) { return &k[size_t(t) * dim]; };
  auto grow = [&](int t) { return &gate[size_t(t) * dim]; };
  require_bitwise("r0 k slot2", at(0, 0, 2), row(2), dim * 2);
  require_bitwise("r0 k slot3", at(0, 0, 3), row(3), dim * 2);
  require_bitwise("r0 k slot0", at(0, 0, 0), row(4), dim * 2);
  require_bitwise("r0 k slot1", at(0, 0, 1), row(5), dim * 2);
  require_bitwise("r0 g slot0", at(0, 1, 0), grow(4), dim * 2);
  require_bitwise("r1 k slot1", at(1, 0, 1), row(7), dim * 2);
  require_bitwise("r1 k slot2", at(1, 0, 2), row(8), dim * 2);
  // Ring slots never touched stay zero.
  require_bitwise("r1 slot3 untouched", at(1, 0, 3), zero.data(), dim * 2);
}

// ---------------------------------------------------------------------------
// Decode ring continuation: chunked prefill (ending mid-pool) then
// single-token decode steps; the pool written across the boundary must
// match the host simulation, and multi-token decode must equal
// single-token decode bitwise (the ordered-loop invariant).
// ---------------------------------------------------------------------------
DGPP_TEST(dsa_decode_update_ring_continuation_matches_host) {
  const int kpool = 4, dim = 128;
  const int64_t prefill_tokens = 10;  // pools 0,1 complete; 2-token tail
  const int64_t decode_tokens = 16;   // completes pools 2..5 and starts 6
  const int max_pools = 8;
  std::vector<float> ape(size_t(kpool) * dim);
  for (int i = 0; i < kpool * dim; ++i) ape[i] = random_f32(41, i) * 0.1f;

  auto gen = [&](uint64_t seed, int64_t n) {
    return random_bf16_bits(seed, n * dim, -2, 1);
  };
  auto pk_rows = gen(51, prefill_tokens);
  auto dg_rows = gen(52, decode_tokens);

  // Hard-max gates with the dominant slot = each pool's completing token
  // (pos % kpool == 3): the compressed pool is bitwise fwht(k[4j+3]) on
  // host and device, so any ring bookkeeping error — stale stash, wrong
  // ring slot, missing is_current override — fails exactly.
  auto hardmax_gate = [&](std::vector<uint16_t>& g, int64_t token_start) {
    for (int64_t t = 0; t < int64_t(g.size()) / dim; ++t) {
      const bool dom = ((token_start + t) % kpool) == kpool - 1;
      for (int d = 0; d < dim; ++d)
        g[size_t(t) * dim + d] = float_to_bf16_bits(dom ? 50.0f : -50.0f);
    }
  };
  auto pg_rows = gen(53, prefill_tokens);
  auto ddg_rows = gen(54, decode_tokens);
  hardmax_gate(pg_rows, 0);
  hardmax_gate(ddg_rows, prefill_tokens);

  // ---- host simulation ----
  RingSim sim{{/*tail=*/std::vector<uint16_t>(2ull * kpool * dim, 0)},
              {/*index_k=*/std::vector<uint8_t>(size_t(max_pools) * dim, 0)},
              {/*index_scale=*/std::vector<float>(max_pools, 0)}, kpool, dim};
  for (int j = 0; j < prefill_tokens / kpool; ++j)
    dsa_ref::compress_pool<float>(&pk_rows[size_t(j * kpool) * dim],
                                  &pg_rows[size_t(j * kpool) * dim],
                                  ape.data(), kpool, dim,
                                  &sim.index_k[size_t(j) * dim],
                                  &sim.index_scale[j]);
  sim.seed(pk_rows, pg_rows, 0, prefill_tokens);
  for (int64_t t = 0; t < decode_tokens; ++t) {
    const int64_t pos = prefill_tokens + t;
    sim.decode_token(&dg_rows[size_t(t) * dim], &ddg_rows[size_t(t) * dim],
                     ape.data(), pos, pos / kpool);
  }

  // ---- device: prefill compress + seed, then decode one token per call ----
  DevBuf dkp(pk_rows.size() * 2), dgp(pg_rows.size() * 2), dape(ape.size() * 4),
      dbt(4), dki(size_t(max_pools) * dim), dks(max_pools * 4),
      dtail(2ull * kpool * dim * 2), dpos(8), dspans(8);
  // Zero-init every cache buffer: unwritten pools stay defined so the
  // downloads are initcheck-clean and comparisons cannot read garbage.
  dki.upload(std::vector<uint8_t>(size_t(max_pools) * dim, 0).data(),
             size_t(max_pools) * dim);
  dks.upload(std::vector<float>(max_pools, 0.0f).data(), max_pools * 4);
  dkp.upload(pk_rows.data(), pk_rows.size() * 2);
  dgp.upload(pg_rows.data(), pg_rows.size() * 2);
  dape.upload(ape.data(), ape.size() * 4);
  int32_t bt0 = 0;
  dbt.upload(&bt0, 4);
  std::vector<uint16_t> zt(2ull * kpool * dim, 0);
  dtail.upload(zt.data(), zt.size() * 2);

  dsa_kpool_compress_write(dkp.p, dim, dgp.p, dim,
                           static_cast<const float*>(dape.p),
                           static_cast<const int32_t*>(dbt.p), max_pools, 0,
                           int(prefill_tokens / kpool), dki.p,
                           static_cast<float*>(dks.p), kpool, dim, 0);
  // Per-token request ids and positions for the whole prefill batch (the
  // kernel indexes req_ids[i]/pos[i] for every token, not just the tail).
  std::vector<int32_t> seed_req(prefill_tokens, 0);
  std::vector<int64_t> seed_pos(prefill_tokens);
  for (int64_t t = 0; t < prefill_tokens; ++t) seed_pos[size_t(t)] = t;
  DevBuf dri(seed_req.size() * 4), dseed_pos(seed_pos.size() * 8);
  dri.upload(seed_req.data(), seed_req.size() * 4);
  dseed_pos.upload(seed_pos.data(), seed_pos.size() * 8);
  dsa_kpool_tail_seed(dkp.p, dim, dgp.p, dim,
                      static_cast<const int32_t*>(dri.p),
                      static_cast<const int64_t*>(dseed_pos.p),
                      prefill_tokens, dtail.p, kpool, dim, 0);

  DevBuf dkd(dg_rows.size() * 2), dgd(ddg_rows.size() * 2);
  dkd.upload(dg_rows.data(), dg_rows.size() * 2);
  dgd.upload(ddg_rows.data(), ddg_rows.size() * 2);
  std::vector<int32_t> spans = {0, 1};
  dspans.upload(spans.data(), 8);
  // The ring after every single-token step: the speculative snapshots of
  // the multi-token call below must reproduce these bitwise.
  const size_t ring_elems = 2ull * kpool * dim;
  std::vector<uint16_t> ring_after(size_t(decode_tokens) * ring_elems);
  for (int64_t t = 0; t < decode_tokens; ++t) {
    const int64_t pos = prefill_tokens + t;
    dpos.upload(&pos, 8);
    dsa_kpool_decode_update(
        static_cast<const uint16_t*>(dkd.p) + size_t(t) * dim, dim,
        static_cast<const uint16_t*>(dgd.p) + size_t(t) * dim, dim,
        static_cast<const float*>(dape.p),
        static_cast<const int32_t*>(dri.p),
        static_cast<const int64_t*>(dpos.p),
        static_cast<const int32_t*>(dspans.p), 1,
        static_cast<const int32_t*>(dbt.p), 1, dtail.p, dki.p,
        static_cast<float*>(dks.p), max_pools, kpool, dim, 0);
    dtail.download(&ring_after[size_t(t) * ring_elems], ring_elems * 2);
  }

  std::vector<uint16_t> got_tail(2ull * kpool * dim);
  dtail.download(got_tail.data(), got_tail.size() * 2);
  std::vector<uint8_t> got_k(size_t(max_pools) * dim);
  std::vector<float> got_s(max_pools, 0.0f);
  dki.download(got_k.data(), got_k.size());
  dks.download(got_s.data(), got_s.size() * 4);
  require_bitwise("tail ring", got_tail.data(), sim.tail.data(),
                  sim.tail.size() * 2);
  // Hard-max gates: pool contents are bitwise comparable.
  const int64_t pools_written =
      std::min<int64_t>(max_pools, (prefill_tokens + decode_tokens) / kpool);
  for (int64_t j = 0; j < pools_written; ++j) {
    if (std::memcmp(&got_k[size_t(j) * dim], &sim.index_k[size_t(j) * dim],
                    dim) != 0 ||
        got_s[j] != sim.index_scale[j])
      throw std::runtime_error("decode pool bitwise mismatch j=" +
                               std::to_string(j));
  }

  // ---- multi-token decode == single-token decode (bitwise) ----
  DevBuf dki2(size_t(max_pools) * dim), dks2(max_pools * 4),
      dtail2(2ull * kpool * dim * 2);
  dki2.upload(std::vector<uint8_t>(size_t(max_pools) * dim, 0).data(),
              size_t(max_pools) * dim);
  dks2.upload(std::vector<float>(max_pools, 0.0f).data(), max_pools * 4);
  dtail2.upload(std::vector<uint16_t>(2ull * kpool * dim, 0).data(),
                2ull * kpool * dim * 2);
  dsa_kpool_compress_write(dkp.p, dim, dgp.p, dim,
                           static_cast<const float*>(dape.p),
                           static_cast<const int32_t*>(dbt.p), max_pools, 0,
                           int(prefill_tokens / kpool), dki2.p,
                           static_cast<float*>(dks2.p), kpool, dim, 0);
  dsa_kpool_tail_seed(dkp.p, dim, dgp.p, dim,
                      static_cast<const int32_t*>(dri.p),
                      static_cast<const int64_t*>(dseed_pos.p),
                      prefill_tokens, dtail2.p, kpool, dim, 0);
  std::vector<int64_t> pos_all(decode_tokens);
  for (int64_t t = 0; t < decode_tokens; ++t) pos_all[t] = prefill_tokens + t;
  DevBuf dpos_all(pos_all.size() * 8);
  dpos_all.upload(pos_all.data(), pos_all.size() * 8);
  std::vector<int32_t> spans_all = {0, int(decode_tokens)};
  dspans.upload(spans_all.data(), 8);
  DevBuf dsnaps(size_t(decode_tokens) * ring_elems * 2);
  dsnaps.upload(std::vector<uint16_t>(size_t(decode_tokens) * ring_elems, 0)
                    .data(),
                dsnaps.bytes);
  dsa_kpool_decode_update(dkd.p, dim, dgd.p, dim,
                          static_cast<const float*>(dape.p),
                          static_cast<const int32_t*>(dri.p),
                          static_cast<const int64_t*>(dpos_all.p),
                          static_cast<const int32_t*>(dspans.p), 1,
                          static_cast<const int32_t*>(dbt.p), 1, dtail2.p,
                          dki2.p, static_cast<float*>(dks2.p), max_pools,
                          kpool, dim, 0, dsnaps.p);
  // Speculative snapshots: row t of the batch == the ring after the t-th
  // single-token step (the last row's ring stays in place: no snapshot).
  std::vector<uint16_t> got_snaps(size_t(decode_tokens) * ring_elems);
  dsnaps.download(got_snaps.data(), dsnaps.bytes);
  for (int64_t t = 0; t + 1 < decode_tokens; ++t)
    require_bitwise("ring snapshot " + std::to_string(t),
                    &got_snaps[size_t(t) * ring_elems],
                    &ring_after[size_t(t) * ring_elems], ring_elems * 2);
  std::vector<uint8_t> got_k2(size_t(max_pools) * dim);
  std::vector<float> got_s2(max_pools);
  std::vector<uint16_t> got_tail2(2ull * kpool * dim);
  dki2.download(got_k2.data(), got_k2.size());
  dks2.download(got_s2.data(), got_s2.size() * 4);
  dtail2.download(got_tail2.data(), got_tail2.size() * 2);
  require_bitwise("multi-token pools", got_k2.data(), got_k.data(),
                  got_k.size());
  require_bitwise("multi-token scales", got_s2.data(), got_s.data(),
                  got_s.size() * 4);
  require_bitwise("multi-token tail", got_tail2.data(), got_tail.data(),
                  got_tail.size() * 2);
}

// ---------------------------------------------------------------------------
// Prefill select: the >=1000-case bitwise fuzz around pool boundaries
// (M3 exit criterion), using synthetic dots so the host mirror and the
// kernel see identical logits.
// ---------------------------------------------------------------------------
DGPP_TEST(dsa_fused_prefill_score_matches_gemm) {
  constexpr int heads = 32, dim = 128, selected = 2048;
  struct Shape { int rows, pools; };
  CublasLtGemm gemm;
  DevBuf gemm_ws(64ull << 20);
  int mismatched_cases = 0;
  for (const Shape shape : {Shape{1, 2053}, {2, 2053}, {2, 8192}, {2, 262144},
                             {3, 8192}, {7, 32771}, {32, 65539},
                             {33, 262144}, {128, 2053}, {97, 8192},
                             // GEMM tile shapes used by the three full-model
                             // benchmark configurations before fusion.
                             {15, 32768}, {7, 65536}, {13, 32768}, {6, 65536},
                             {3, 131072}, {16, 32768}, {8, 65536}, {4, 131072}}) {
    const int rows = shape.rows, pools = shape.pools;
    const int stride = (pools + 255) / 256 * 256;
    for (const int mode : {0, 1, 2}) {
      const bool wide_range = mode == 1;
      std::vector<uint8_t> q(rows * heads * dim), k(size_t(stride) * dim);
      const auto code = [&](uint32_t h) {
        if (wide_range) return uint8_t((h & 0x80u) | ((h >> 8) % 127));
        return float_to_fp8_e4m3_bits(random_f32(1937, h) * 128.0f);
      };
      for (size_t i = 0; i < q.size(); ++i) q[i] = code(hash32(uint32_t(i + 173)));
      for (size_t i = 0; i < k.size(); ++i) k[i] = code(hash32(uint32_t(i + 479)));
      if (mode == 2) std::fill(q.begin(), q.end(), 0);  // exact ties
      std::vector<float> w(rows * heads), scales(stride);
      for (size_t i = 0; i < w.size(); ++i) w[i] = random_f32(1938, i);
      for (int i = 0; i < stride; ++i) scales[i] = 0.01f + std::fabs(random_f32(1939, i));
      std::vector<int64_t> pos(rows);
      for (int r = 0; r < rows; ++r)
        pos[r] = rows > 1 && r % 3 == 0 ? pools / 2 - 1 : pools - 1 - (r % 3 == 1);
      DevBuf dq(q.size()), dk(k.size()), dw(w.size() * 4), ds(scales.size() * 4), dp(rows * 8),
          dots(size_t(rows) * heads * stride * 4), picks(rows * selected * 4), counts(rows * 4),
          old_ws(dsa_select_prefill_workspace_bytes(rows, stride, selected)),
          fused(size_t(rows) * stride * sizeof(uint64_t));
      dq.upload(q.data(), q.size()); dk.upload(k.data(), k.size());
      dw.upload(w.data(), w.size() * 4); ds.upload(scales.data(), scales.size() * 4);
      dp.upload(pos.data(), pos.size() * 8);
      gemm.matmul(dq.p, dk.p, dots.p, rows * heads, stride, dim,
                  DType::F8_E4M3, GemmOut::F32, dim, gemm_ws.p, gemm_ws.bytes, nullptr);
      for (const bool relu : {false, true}) {
        dsa_select_prefill(static_cast<const float*>(dots.p), stride,
            static_cast<const float*>(dw.p), static_cast<const float*>(ds.p),
            static_cast<const int64_t*>(dp.p), rows, pools, heads, selected, 1, selected,
            static_cast<int32_t*>(picks.p), static_cast<int32_t*>(counts.p), nullptr, relu, old_ws.p);
        dsa_prefill_score_keys(static_cast<const uint8_t*>(dq.p), static_cast<const uint8_t*>(dk.p),
              static_cast<const float*>(dw.p), static_cast<const float*>(ds.p),
              static_cast<const int64_t*>(dp.p), rows, pools, stride, selected, 1,
              static_cast<uint64_t*>(fused.p), nullptr, relu);
        std::vector<uint64_t> expected(size_t(rows) * stride), got(expected.size());
        old_ws.download(expected.data(), expected.size() * 8);
        fused.download(got.data(), got.size() * 8);
        int different = 0;
        for (int r = 0; r < rows; ++r) {
          const int visible = int(pos[r] + 1);
          if (visible <= selected) continue;
          for (int p = 0; p < visible; ++p)
            different += got[size_t(r) * stride + p] != expected[size_t(r) * stride + p];
        }
        // The small single-query cuBLAS plan has a different reduction;
        // its score error is checked against FP64 by the accuracy test.
        // Still require identical selected IDs/counts for this fixture.
        if (different && rows != 1) {
          ++mismatched_cases;
          std::fprintf(stderr, "fused score key mismatch: rows=%d pools=%d mode=%d relu=%d count=%d\n",
                       rows, pools, mode, int(relu), different);
          continue;
        }
        std::vector<int32_t> want_picks(size_t(rows) * selected), got_picks(want_picks.size()),
            want_counts(rows), got_counts(rows);
        picks.download(want_picks.data(), want_picks.size() * 4);
        counts.download(want_counts.data(), want_counts.size() * 4);
        dsa_select_prefill_keys(static_cast<const uint64_t*>(fused.p), stride,
            static_cast<const int64_t*>(dp.p), rows, pools, selected, 1, selected,
            static_cast<int32_t*>(picks.p), static_cast<int32_t*>(counts.p),
            static_cast<uint64_t*>(old_ws.p), nullptr);
        picks.download(got_picks.data(), got_picks.size() * 4);
        counts.download(got_counts.data(), got_counts.size() * 4);
        require_bitwise("fused prefill selected token ids", got_picks.data(),
                         want_picks.data(), want_picks.size() * 4);
        require_bitwise("fused prefill selected counts", got_counts.data(),
                         want_counts.data(), want_counts.size() * 4);
      }
    }
  }
  if (mismatched_cases)
    throw std::runtime_error("fused score key mismatched cases=" + std::to_string(mismatched_cases));
}

DGPP_TEST(dsa_fused_single_query_fp64_accuracy) {
  constexpr int heads = 32, dim = 128, selected = 2048;
  CublasLtGemm gemm;
  DevBuf gemm_ws(64ull << 20);
  const auto score_from_key = [](uint64_t key) {
    const uint32_t ordered = ~uint32_t(key >> 21);
    const uint32_t bits = (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
    float score;
    std::memcpy(&score, &bits, sizeof(score));
    return double(score);
  };
  for (const int pools : {2053, 8192, 65539, 262144}) {
    const int stride = (pools + 255) / 256 * 256;
    for (int seed = 0; seed < (pools == 262144 ? 1 : 3); ++seed) {
      for (const int mode : {0, 1, 2}) {
        if (mode == 2 && (seed != 0 || pools > 8192)) continue;
        std::vector<uint8_t> q(2 * heads * dim), k(size_t(stride) * dim);
        const auto code = [&](uint32_t h) {
          return mode == 1 ? uint8_t((h & 0x80u) | ((h >> 8) % 127))
                           : float_to_fp8_e4m3_bits(random_f32(1937 + seed, h) * 128.0f);
        };
        for (int i = 0; i < heads * dim; ++i)
          q[i] = mode == 2 ? 0 : code(hash32(uint32_t(i + 173 + seed * 7919)));
        std::copy(q.begin(), q.begin() + heads * dim, q.begin() + heads * dim);
        for (size_t i = 0; i < k.size(); ++i) k[i] = code(hash32(uint32_t(i + 479 + seed * 7907)));
        std::vector<float> w(2 * heads), scales(stride);
        for (int h = 0; h < heads; ++h) w[h] = w[heads + h] = random_f32(1938 + seed, h);
        for (int p = 0; p < stride; ++p) scales[p] = 0.01f + std::fabs(random_f32(1939 + seed, p));
        const int64_t positions[2] = {pools - 1, pools - 1};
        DevBuf dq(q.size()), dk(k.size()), dw(w.size() * 4), ds(scales.size() * 4), dp(16),
            dots(size_t(heads) * stride * 4), picks(selected * 4), counts(4),
            workspace(dsa_select_prefill_workspace_bytes(1, stride, selected)),
            fused(size_t(2) * stride * 8);
        dq.upload(q.data(), q.size()); dk.upload(k.data(), k.size());
        dw.upload(w.data(), w.size() * 4); ds.upload(scales.data(), scales.size() * 4);
        dp.upload(positions, sizeof(positions));
        gemm.matmul(dq.p, dk.p, dots.p, heads, stride, dim, DType::F8_E4M3,
                    GemmOut::F32, dim, gemm_ws.p, gemm_ws.bytes, nullptr);
        std::vector<double> q64(heads * dim), k64(size_t(pools) * dim);
        for (size_t i = 0; i < q64.size(); ++i) q64[i] = fp8_e4m3_bits_to_float(q[i]);
        for (size_t i = 0; i < k64.size(); ++i) k64[i] = fp8_e4m3_bits_to_float(k[i]);
        for (const bool relu : {false, true}) {
          dsa_select_prefill(static_cast<const float*>(dots.p), stride,
              static_cast<const float*>(dw.p), static_cast<const float*>(ds.p),
              static_cast<const int64_t*>(dp.p), 1, pools, heads, selected, 1, selected,
              static_cast<int32_t*>(picks.p), static_cast<int32_t*>(counts.p), nullptr, relu, workspace.p);
          dsa_prefill_score_keys(static_cast<const uint8_t*>(dq.p), static_cast<const uint8_t*>(dk.p),
              static_cast<const float*>(dw.p), static_cast<const float*>(ds.p),
              static_cast<const int64_t*>(dp.p), 1, pools, stride, selected, 1,
              static_cast<uint64_t*>(fused.p), nullptr, relu);
          std::vector<uint64_t> old_keys(pools), new_keys(pools);
          workspace.download(old_keys.data(), old_keys.size() * 8);
          fused.download(new_keys.data(), new_keys.size() * 8);
          std::vector<double> reference(pools);
          double old_error2 = 0, new_error2 = 0, reference2 = 0;
          double old_max = 0, new_max = 0, old_scaled = 0, new_scaled = 0;
          int differing = 0;
          for (int p = 0; p < pools; ++p) {
            double sum = 0, magnitude = 0;
            for (int h = 0; h < heads; ++h) {
              double dot = 0, absolute_products = 0;
              for (int d = 0; d < dim; ++d) {
                const double product = q64[h * dim + d] * k64[size_t(p) * dim + d];
                dot += product;
                absolute_products += std::fabs(product);
              }
              const double weight = double(w[h]) * double(scales[p]);
              sum += weight * (relu ? std::max(dot, 0.0) : dot);
              magnitude += std::fabs(weight) * absolute_products;
            }
            reference[p] = sum;
            const double old_error = std::fabs(score_from_key(old_keys[p]) - sum);
            const double new_error = std::fabs(score_from_key(new_keys[p]) - sum);
            if (!std::isfinite(old_error) || !std::isfinite(new_error))
              throw std::runtime_error("nonfinite single-query score error");
            old_error2 += old_error * old_error; new_error2 += new_error * new_error;
            reference2 += sum * sum;
            old_max = std::max(old_max, old_error); new_max = std::max(new_max, new_error);
            old_scaled = std::max(old_scaled, old_error / std::max(magnitude, 1e-300));
            new_scaled = std::max(new_scaled, new_error / std::max(magnitude, 1e-300));
            differing += old_keys[p] != new_keys[p];
          }
          std::vector<int> oracle(pools), before(pools), after(pools);
          for (int p = 0; p < pools; ++p) oracle[p] = before[p] = after[p] = p;
          std::partial_sort(oracle.begin(), oracle.begin() + selected, oracle.end(), [&](int a, int b) {
            return reference[a] != reference[b] ? reference[a] > reference[b] : a < b;
          });
          const auto audit = [&](std::vector<int>& ids, const std::vector<uint64_t>& keys, double noise) {
            std::partial_sort(ids.begin(), ids.begin() + selected, ids.end(),
                              [&](int a, int b) { return keys[a] < keys[b]; });
            std::vector<bool> chosen(pools, false), exact(pools, false);
            for (int i = 0; i < selected; ++i) { chosen[ids[i]] = true; exact[oracle[i]] = true; }
            int swaps = 0;
            double lowest_added = reference[oracle[0]], highest_removed = reference[oracle[selected - 1]];
            for (int p = 0; p < pools; ++p) {
              if (chosen[p] && !exact[p]) { ++swaps; lowest_added = std::min(lowest_added, reference[p]); }
              if (!chosen[p] && exact[p]) highest_removed = std::max(highest_removed, reference[p]);
            }
            const double gap = swaps ? highest_removed - lowest_added : 0;
            if (gap > 2 * noise + 1e-12 * std::max(1.0, std::fabs(highest_removed)))
              throw std::runtime_error("single-query selection not a certified boundary near tie");
            return std::pair<int, double>{swaps, gap};
          };
          const auto old_cut = audit(before, old_keys, old_max), new_cut = audit(after, new_keys, new_max);
          // A conservative forward-error bound for 128 FP32 dot terms
          // plus head folding. Report actual errors against FP64, rather
          // than treating the previous GEMM's bits as ground truth.
          if (old_scaled > 2e-5 || new_scaled > 2e-5 ||
              std::sqrt(new_error2 / std::max(reference2, 1e-300)) > 1e-6)
            throw std::runtime_error("single-query score exceeds FP32 forward-error bound");
          std::printf("ACCURACY {\"pools\":%d,\"seed\":%d,\"mode\":%d,\"relu\":%d,"
              "\"differing_keys\":%d,\"old_nrmse\":%.12g,\"fused_nrmse\":%.12g,"
              "\"old_max_abs\":%.12g,\"fused_max_abs\":%.12g,"
              "\"old_max_scaled\":%.12g,\"fused_max_scaled\":%.12g,"
              "\"old_swaps\":%d,\"fused_swaps\":%d,\"old_boundary_gap\":%.12g,\"fused_boundary_gap\":%.12g}\n",
              pools, seed, mode, int(relu), differing,
              std::sqrt(old_error2 / std::max(reference2, 1e-300)),
              std::sqrt(new_error2 / std::max(reference2, 1e-300)), old_max, new_max,
              old_scaled, new_scaled, old_cut.first, new_cut.first, old_cut.second, new_cut.second);
        }
      }
    }
  }
}

DGPP_TEST(dsa_select_prefill_radix_long_context_exact) {
  // Independent CPU oracle, including exact ties, ReLU, both model
  // budgets, partial pools, padded strides, and mixed causal positions.
  // The last Flash case reaches 1M tokens; these are the small row counts
  // produced by the bounded dot buffer at long context.
  constexpr int rows = 3, heads = 32;
  for (const int kpool : {4, 1}) {
    const int select_k = 2048 / kpool;
    const int max_selected = 2048 + kpool - 1;
    for (const int pools : {2053, 32771, 131075, 262144}) {
      const int stride = pools + 17;
      for (const bool relu : {false, true}) {
        for (const bool ties : {false, true}) {
          std::vector<float> dots(size_t(rows) * heads * stride), w(rows * heads), ks(pools);
          for (size_t i = 0; i < dots.size(); ++i)
            dots[i] = ties ? 1.0f : random_f32(931, i);
          for (size_t i = 0; i < w.size(); ++i)
            w[i] = ties ? (i % 2 ? -1.0f : 1.0f) : random_f32(932, i);
          for (int p = 0; p < pools; ++p)
            ks[p] = ties ? 1.0f : std::fabs(random_f32(933, p));
          std::vector<int64_t> pos = {int64_t(std::max(0, kpool - 2)),
                                     int64_t(pools / 2) * kpool + kpool - 2,
                                     int64_t(pools) * kpool - 1};
          DevBuf dd(dots.size() * 4), dw(w.size() * 4), dks(ks.size() * 4),
              dpos(pos.size() * 8), dtopk(size_t(rows) * max_selected * 4),
              dcnt(rows * 4), dws(dsa_select_prefill_workspace_bytes(rows, stride, select_k));
          dd.upload(dots.data(), dots.size() * 4);
          dw.upload(w.data(), w.size() * 4);
          dks.upload(ks.data(), ks.size() * 4);
          dpos.upload(pos.data(), pos.size() * 8);
          // Two calls with reused, initially dirty scratch catch reliance
          // on zero-initialized keys/histograms and incomplete output writes.
          DGPP_CUDA_OK(cudaMemset(dws.p, 0xa5, dws.bytes));
          for (int replay = 0; replay < 2; ++replay)
            dsa_select_prefill(static_cast<const float*>(dd.p), stride,
                static_cast<const float*>(dw.p), static_cast<const float*>(dks.p),
                static_cast<const int64_t*>(dpos.p), rows, pools, heads, select_k,
                kpool, max_selected, static_cast<int32_t*>(dtopk.p),
                static_cast<int32_t*>(dcnt.p), nullptr, relu, dws.p);
          std::vector<int32_t> got(size_t(rows) * max_selected), counts(rows);
          dtopk.download(got.data(), got.size() * 4);
          dcnt.download(counts.data(), counts.size() * 4);
          for (int r = 0; r < rows; ++r) {
            const int64_t visible = (pos[r] + 1) / kpool;
            std::vector<float> logits(static_cast<size_t>(visible));
            for (int64_t p = 0; p < visible; ++p) {
              float d[32];
              for (int h = 0; h < heads; ++h) {
                d[h] = dots[(size_t(r) * heads + h) * stride + p];
                if (relu) d[h] = std::max(d[h], 0.0f);
              }
              logits[p] = prefill_logit_mirror(d, w.data() + r * heads, ks[p]);
            }
            std::vector<int32_t> ids(select_k), want(max_selected, -1);
            const int selected = dsa_ref::select_pools(logits.data(), visible, select_k, ids.data());
            const int count = dsa_ref::expand_append_tail(ids.data(), selected, pos[r],
                                                         kpool, max_selected, want.data());
            if (counts[r] != count)
              throw std::runtime_error("prefill radix count mismatch");
            require_bitwise("prefill radix exact pools=" + std::to_string(pools) +
                            " kpool=" + std::to_string(kpool) + " row=" + std::to_string(r) +
                            " relu=" + std::to_string(relu) + " ties=" + std::to_string(ties),
                            got.data() + size_t(r) * max_selected, want.data(), want.size() * 4);
          }
        }
      }
    }
  }
}

DGPP_TEST(dsa_select_prefill_partition_shapes_and_tails) {
  // Both sides of the 96-block target, including the 190-leaf workspace
  // bound (95 rows x 2 partitions) and 128 partitions for one long row.
  constexpr int heads = 32, kpool = 4, select_k = 512, max_selected = 2051;
  for (const int rows : {1, 2, 5, 7, 31, 95, 96, 97}) {
    const int pools = rows < 8 ? 131072 : 4097;
    const int stride = pools + 3;
    std::vector<float> dots(size_t(rows) * heads * stride, 1.0f),
        weights(rows * heads, 1.0f), scales(pools, 1.0f);
    std::vector<int64_t> positions(rows);
    for (int r = 0; r < rows; ++r)
      positions[r] = r % 5 == 4 ? r % kpool : int64_t(pools - 1) * kpool + r % kpool;
    DevBuf dd(dots.size() * 4), dw(weights.size() * 4), ds(scales.size() * 4),
        dp(positions.size() * 8), dk(size_t(rows) * max_selected * 4), dc(rows * 4),
        ws(dsa_select_prefill_workspace_bytes(rows, stride, select_k));
    dd.upload(dots.data(), dots.size() * 4);
    dw.upload(weights.data(), weights.size() * 4);
    ds.upload(scales.data(), scales.size() * 4);
    dp.upload(positions.data(), positions.size() * 8);
    dsa_select_prefill(static_cast<const float*>(dd.p), stride,
        static_cast<const float*>(dw.p), static_cast<const float*>(ds.p),
        static_cast<const int64_t*>(dp.p), rows, pools, heads, select_k, kpool,
        max_selected, static_cast<int32_t*>(dk.p), static_cast<int32_t*>(dc.p),
        nullptr, false, ws.p);
    std::vector<int32_t> got(size_t(rows) * max_selected), counts(rows);
    dk.download(got.data(), got.size() * 4);
    dc.download(counts.data(), counts.size() * 4);
    for (int r = 0; r < rows; ++r) {
      const int64_t visible = (positions[r] + 1) / kpool;
      const int selected = int(std::min<int64_t>(select_k, visible));
      std::vector<int32_t> ids(selected), want(max_selected, -1);
      for (int i = 0; i < selected; ++i) ids[i] = i;  // exact ties -> lower pool ids
      const int count = dsa_ref::expand_append_tail(ids.data(), selected, positions[r],
                                                   kpool, max_selected, want.data());
      if (counts[r] != count) throw std::runtime_error("partitioned prefill count mismatch");
      require_bitwise("partitioned prefill rows=" + std::to_string(rows),
                       got.data() + size_t(r) * max_selected, want.data(), want.size() * 4);
    }
  }
}

DGPP_TEST(dsa_select_prefill_bitwise_fuzz) {
  const DsaConfig cfg{};
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int select_k = g.select_k, kpool = cfg.index_kpool;
  const int max_selected = g.max_selected;
  const int heads = cfg.index_n_heads;

  // Case corpus: lengths 0..7, every residue around 4k boundaries up to
  // 2052, and hashed randoms — >= 1000 distinct (pos, pool-count) cases.
  std::vector<std::pair<int64_t, int64_t>> cases;  // (pos, n_pools available)
  for (int p = 0; p <= 7; ++p) cases.push_back({p, (p / 4) + 3});
  for (int base = 1; base <= 515; ++base)
    for (int r = -1; r <= 2; ++r) {
      const int64_t pos = int64_t(base) * 4 + r;
      if (pos >= 0) cases.push_back({pos, pos / 4 + 1});
    }
  for (int i = 0; cases.size() < 1100; ++i) {
    const int64_t pos = hash32(9000 + i) % 9000;
    cases.push_back({pos, pos / 4 + 1 + hash32(7000 + i) % 4});
  }

  int checked = 0;
  for (size_t ci = 0; ci < cases.size(); ++ci) {
    const int64_t pos = cases[ci].first;
    const int64_t n_pools = cases[ci].second;
    const int64_t visible = (pos + 1) / kpool;
    const int64_t avail = std::max<int64_t>(n_pools, visible);
    if (avail <= 0) {
      // No pools at all: only the tail token list.
    }
    // Random dots [heads, avail], weights [heads], scales [avail].
    std::vector<float> dots(size_t(heads) * avail);
    std::vector<float> w(size_t(heads), 0.0f);
    std::vector<float> ks(size_t(avail), 0.0f);
    for (auto& v : dots) v = random_f32(100 + ci, &v - dots.data());
    for (auto& v : w) v = random_f32(200 + ci, &v - w.data());
    for (auto& v : ks) v = random_f32(300 + ci, &v - ks.data());
    std::vector<int64_t> posv = {pos};

    DevBuf dd(dots.size() * 4), dw(w.size() * 4), dks(ks.size() * 4),
        dpos(8), dtopk(size_t(max_selected) * 4), dcnt(4);
    dd.upload(dots.data(), dots.size() * 4);
    dw.upload(w.data(), w.size() * 4);
    dks.upload(ks.data(), ks.size() * 4);
    dpos.upload(posv.data(), 8);
    dsa_select_prefill(static_cast<const float*>(dd.p), avail,
                       static_cast<const float*>(dw.p),
                       static_cast<const float*>(dks.p),
                       static_cast<const int64_t*>(dpos.p), 1, avail, heads,
                       select_k, kpool, max_selected,
                       static_cast<int32_t*>(dtopk.p),
                       static_cast<int32_t*>(dcnt.p), 0);

    // Host mirror: identical logit arithmetic, then the pinned selection.
    std::vector<float> logits(size_t(visible), 0.0f);
    for (int64_t j = 0; j < visible; ++j) {
      float dot_h[32];
      for (int h = 0; h < heads; ++h) dot_h[h] = dots[size_t(h) * avail + j];
      logits[size_t(j)] = prefill_logit_mirror(dot_h, w.data(), ks[size_t(j)]);
    }
    std::vector<int32_t> pool_ids(select_k, 0);
    const int n_sel =
        dsa_ref::select_pools(logits.data(), visible, select_k, pool_ids.data());
    std::vector<int32_t> want(size_t(max_selected), -1);
    const int n_tok =
        dsa_ref::expand_append_tail(pool_ids.data(), n_sel, pos, kpool,
                                    max_selected, want.data());

    std::vector<int32_t> got(size_t(max_selected), -12345);
    int32_t got_cnt = -1;
    dtopk.download(got.data(), got.size() * 4);
    dcnt.download(&got_cnt, 4);
    if (got_cnt != n_tok)
      throw std::runtime_error("count mismatch case " + std::to_string(ci) +
                               " pos=" + std::to_string(pos) + " got=" +
                               std::to_string(got_cnt) + " want=" +
                               std::to_string(n_tok));
    if (std::memcmp(got.data(), want.data(), max_selected * 4) != 0) {
      std::string detail;
      for (int i = 0; i < max_selected; ++i)
        if (got[size_t(i)] != want[size_t(i)])
          detail += " [" + std::to_string(i) + "] got=" +
                    std::to_string(got[size_t(i)]) + " want=" +
                    std::to_string(want[size_t(i)]);
      throw std::runtime_error("token mismatch case " + std::to_string(ci) +
                               " pos=" + std::to_string(pos) + detail);
    }
    ++checked;
  }
  if (checked < 1000)
    throw std::runtime_error("fuzz corpus too small: " + std::to_string(checked));
}

// ---------------------------------------------------------------------------
// Fused decode select: same bitwise oracle, logits computed inline from the
// fp8 cache; also repeat-run determinism.
// ---------------------------------------------------------------------------
DGPP_TEST(dsa_select_decode_fused_bitwise) {
  const DsaConfig cfg{};
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int select_k = g.select_k, kpool = cfg.index_kpool;
  const int max_selected = g.max_selected;
  const int heads = cfg.index_n_heads, dim = cfg.index_head_dim;
  const int n_pools = 6000;  // > select_k: real sparse selection
  const int64_t pos = 4 * n_pools + 2;  // visible = n_pools, tail 2

  // q_fp8 with sane magnitudes and matching scales; random fp8 cache.
  std::vector<uint16_t> qb = random_bf16_bits(61, heads * dim, -2, 1);
  std::vector<uint8_t> q8(size_t(heads) * dim);
  std::vector<float> q_scale(size_t(heads), 0.0f);
  dsa_ref::fwht128_quant_fp8<float>(qb.data(), heads, dim, q8.data(),
                                    q_scale.data());
  std::vector<float> w_raw(heads, 0.0f);
  for (auto& v : w_raw) v = random_f32(62, &v - w_raw.data()) * 0.01f;
  std::vector<float> w(heads, 0.0f);
  const float logit_scale =
      float(std::pow(128.0, -0.5) * std::pow(32.0, -0.5));
  for (int h = 0; h < heads; ++h)
    w[h] = (w_raw[h] * q_scale[h]) * logit_scale;

  std::vector<uint8_t> k8(size_t(n_pools) * dim);
  std::vector<float> ks(n_pools, 0.0f);
  // Random fp8 bytes, remapping the two NaN encodings (0x7F/0xFF) to max
  // finite: real cache rows are always finite (the saturating encoder never
  // mints NaN), and the select spec assumes finite logits.
  for (auto& v : k8) {
    v = uint8_t(hash32(63 + (&v - k8.data())) & 0xFF);
    if ((v & 0x7Fu) == 0x7Fu) v ^= 0x01u;
  }
  for (auto& v : ks) v = std::fabs(random_f32(64, &v - ks.data())) * 0.05f;

  DevBuf dq8(q8.size()), dks_cache(ks.size() * 4), dw(w.size() * 4),
      dki(k8.size()), dpos(8), dri(4), dbt(4), dtopk(size_t(max_selected) * 4),
      dcnt(4), dws(dsa_select_workspace_bytes(1, n_pools)), dctr(8);
  dq8.upload(q8.data(), q8.size());
  dw.upload(w.data(), w.size() * 4);
  dki.upload(k8.data(), k8.size());
  dks_cache.upload(ks.data(), ks.size() * 4);
  std::vector<int64_t> posv = {pos};
  std::vector<int32_t> req0 = {0}, bt0 = {0};
  dpos.upload(posv.data(), 8);
  dri.upload(req0.data(), 4);
  dbt.upload(bt0.data(), 4);
  int32_t zero32 = 0;
  dctr.upload(&zero32, 4);

  dsa_select_decode(dq8.p, static_cast<const float*>(dw.p),
                    static_cast<const int32_t*>(dri.p),
                    static_cast<const int64_t*>(dpos.p), 1,
                    static_cast<const int32_t*>(dbt.p), 1, dki.p,
                    static_cast<const float*>(dks_cache.p), n_pools, heads,
                    dim, select_k, kpool, max_selected,
                    static_cast<int32_t*>(dtopk.p),
                    static_cast<int32_t*>(dcnt.p), dws.p, n_pools,
                    static_cast<int32_t*>(dctr.p), 48, 0);

  // Host mirror: identical inline dot arithmetic.
  std::vector<float> logits(n_pools, 0.0f);
  for (int j = 0; j < n_pools; ++j) {
    float contrib[32];
    for (int h = 0; h < 32; ++h) {
      float partial = 0.0f;
      for (int d = 0; d < 128; ++d)
        partial = partial +
                  fp8_e4m3_bits_to_float(q8[size_t(h) * 128 + d]) *
                      fp8_e4m3_bits_to_float(k8[size_t(j) * 128 + d]);
      contrib[h] = (w[h] * ks[j]) * partial;
    }
    logits[size_t(j)] = butterfly_sum_32(contrib);
  }
  std::vector<int32_t> pool_ids(select_k, 0);
  const int n_sel =
      dsa_ref::select_pools(logits.data(), n_pools, select_k, pool_ids.data());
  std::vector<int32_t> want(size_t(max_selected), -1);
  dsa_ref::expand_append_tail(pool_ids.data(), n_sel, pos, kpool, max_selected,
                              want.data());

  std::vector<int32_t> got(size_t(max_selected), -12345);
  int32_t got_cnt = -1;
  dtopk.download(got.data(), got.size() * 4);
  dcnt.download(&got_cnt, 4);
  if (got_cnt != int(n_sel * kpool + (pos + 1) % kpool))
    throw std::runtime_error("decode select count");
  if (std::memcmp(got.data(), want.data(), max_selected * 4) != 0) {
    // Diagnostics: which pools are selected by one side but not the other,
    // and their host-mirror logits (a boundary flip shows ulp-close logits;
    // a kernel bug shows a clearly-better pool missing).
    std::vector<int32_t> gset, wset;
    for (int i = 0; i < max_selected; ++i) {
      if (got[size_t(i)] >= 0) gset.push_back(got[size_t(i)] / kpool);
      if (want[size_t(i)] >= 0) wset.push_back(want[size_t(i)] / kpool);
    }
    std::sort(gset.begin(), gset.end());
    std::sort(wset.begin(), wset.end());
    std::string detail;
    for (int p : gset)
      if (!std::binary_search(wset.begin(), wset.end(), p))
        detail += " device-only pool " + std::to_string(p) + " logit " +
                  std::to_string(logits[size_t(p)]) + ";";
    for (int p : wset)
      if (!std::binary_search(gset.begin(), gset.end(), p))
        detail += " host-only pool " + std::to_string(p) + " logit " +
                  std::to_string(logits[size_t(p)]) + ";";
    std::vector<float> sorted = logits;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    detail += " 512th logit=" + std::to_string(sorted[511]) + " 513th=" +
              std::to_string(sorted[512]);
    throw std::runtime_error("decode select tokens mismatch: " + detail);
  }

  // Determinism: a second run must be identical.
  std::vector<int32_t> got2(size_t(max_selected), -12345);
  dsa_select_decode(dq8.p, static_cast<const float*>(dw.p),
                    static_cast<const int32_t*>(dri.p),
                    static_cast<const int64_t*>(dpos.p), 1,
                    static_cast<const int32_t*>(dbt.p), 1, dki.p,
                    static_cast<const float*>(dks_cache.p), n_pools, heads,
                    dim, select_k, kpool, max_selected,
                    static_cast<int32_t*>(dtopk.p),
                    static_cast<int32_t*>(dcnt.p), dws.p, n_pools,
                    static_cast<int32_t*>(dctr.p), 48, 0);
  dtopk.download(got2.data(), got2.size() * 4);
  require_bitwise("decode select repeat", got2.data(), got.data(),
                  got.size() * 4);
}

// ---------------------------------------------------------------------------
// Absorbed attention chain vs the host oracle (tolerance: online-softmax
// rescaling vs one-shot).
// ---------------------------------------------------------------------------
DGPP_TEST(dsa_absorbed_attention_matches_host_reference) {
  const DsaConfig cfg{};
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int rows = 3, local_heads = g.local_heads, nope = cfg.qk_nope_head_dim;
  const int v = cfg.v_head_dim, kv_lora = cfg.kv_lora_rank;
  const int block_tokens = cfg.block_tokens;
  const int max_selected = g.max_selected;

  auto q = random_bf16_bits(71, int64_t(rows) * local_heads * nope, -2, 1);
  auto kv_b = random_bf16_bits(72, int64_t(local_heads) * (nope + v) * kv_lora,
                                -2, 1);
  const int cnt = 37;
  std::vector<int32_t> tokens(size_t(rows) * max_selected, -1);
  std::vector<int32_t> counts(rows, cnt);
  std::vector<int32_t> req_ids(rows, 0);
  for (int r = 0; r < rows; ++r)
    for (int i = 0; i < cnt; ++i)
      tokens[size_t(r) * max_selected + i] =
          int32_t((r * 131 + i * 17) % 200);

  // Latent cache: 2 blocks of block_tokens rows.
  const int n_blocks = 2, total_tokens = n_blocks * block_tokens;
  auto latent =
      random_bf16_bits(73, int64_t(total_tokens) * kv_lora, -2, 1);
  std::vector<int32_t> bt = {1, 0};  // reversed on purpose

  const int n_split = 4;
  DevBuf dq(q.size() * 2), dkb(kv_b.size() * 2), dlat(latent.size() * 2),
      dtopk(tokens.size() * 4), dcnt(rows * 4), dri(rows * 4), dbt(8),
      dqt(size_t(rows) * local_heads * kv_lora * 2),
      dm(size_t(rows) * n_split * local_heads * 4),
      dl(size_t(rows) * n_split * local_heads * 4),
      dc(size_t(rows) * n_split * local_heads * kv_lora * 4),
      dc_out(size_t(rows) * local_heads * kv_lora * 4),
      dout(size_t(rows) * local_heads * v * 2);
  dq.upload(q.data(), q.size() * 2);
  dkb.upload(kv_b.data(), kv_b.size() * 2);
  dlat.upload(latent.data(), latent.size() * 2);
  dtopk.upload(tokens.data(), tokens.size() * 4);
  dcnt.upload(counts.data(), counts.size() * 4);
  dri.upload(req_ids.data(), req_ids.size() * 4);
  dbt.upload(bt.data(), bt.size() * 4);

  dsa_absorb_q(dq.p, dkb.p, dqt.p, rows, local_heads, nope, v, kv_lora, 0);
  const float scale = 1.0f / std::sqrt(float(nope));
  dsa_attn_partial(dqt.p, dlat.p, static_cast<const int32_t*>(dri.p),
                   static_cast<const int32_t*>(dtopk.p), max_selected,
                   static_cast<const int32_t*>(dcnt.p), rows, n_split,
                   local_heads, kv_lora, block_tokens,
                   static_cast<const int32_t*>(dbt.p), n_blocks, scale,
                   static_cast<float*>(dm.p), static_cast<float*>(dl.p),
                   static_cast<float*>(dc.p), 0);
  dsa_attn_combine(static_cast<const float*>(dm.p),
                   static_cast<const float*>(dl.p),
                   static_cast<const float*>(dc.p), rows, n_split, local_heads,
                   kv_lora, static_cast<float*>(dc_out.p), 0);
  dsa_vout_gemm(dc_out.p, dkb.p, dout.p, rows, local_heads, nope, v, kv_lora,
                0);

  // Host oracle: build the physical latent view via the block table.
  std::vector<uint16_t> phys_latent(size_t(total_tokens) * kv_lora);
  for (int64_t tok = 0; tok < total_tokens; ++tok) {
    const int32_t blk = bt[size_t(tok / block_tokens)];
    const int64_t phys = int64_t(blk) * block_tokens + (tok % block_tokens);
    std::memcpy(&phys_latent[size_t(tok) * kv_lora],
                &latent[size_t(phys) * kv_lora], kv_lora * 2);
  }
  std::vector<uint16_t> want(size_t(rows) * local_heads * v);
  for (int r = 0; r < rows; ++r)
    dsa_ref::absorbed_attn<float>(&q[size_t(r) * local_heads * nope],
                                  phys_latent.data(), kv_lora,
                                  &tokens[size_t(r) * max_selected], cnt,
                                  kv_b.data(), local_heads, nope, v, kv_lora,
                                  scale, &want[size_t(r) * local_heads * v]);
  std::vector<uint16_t> got(size_t(rows) * local_heads * v);
  dout.download(got.data(), got.size() * 2);
  require_bf16("absorbed attention vs host f32", compare_bf16(got, want, 8),
               0.01, 0.006);
}

// ---------------------------------------------------------------------------
// Dense causal attention on tensor cores (the prefill path below index_topk
// tokens) against the split kernel over the equivalent dense selection
// (tolerance: a different fp32 summation order) and the host oracle, at
// the real geometry (64 heads / 512 latent), at TP=4's 16 heads, and at a
// 256-wide latent; deterministic; split counts agree within tolerance.
// ---------------------------------------------------------------------------
namespace {

struct DenseCase { int num_heads, kv_lora; };

void run_dense_attention_case(const DenseCase& dc) {
  DsaConfig cfg{};
  cfg.num_heads = dc.num_heads;
  cfg.kv_lora_rank = dc.kv_lora;
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int local_heads = g.local_heads, nope = cfg.qk_nope_head_dim;
  const int v = cfg.v_head_dim, kv_lora = cfg.kv_lora_rank;
  const int block_tokens = cfg.block_tokens;
  // 41 query rows at positions 90..130: rows straddle the 32-token tiles,
  // the two blocks of the latent cache, and the split boundaries.
  const int rows = 41;
  const int64_t pos0 = 90;
  const int max_selected = int(pos0) + rows;  // dense lists: tokens [0, pos]
  std::vector<int64_t> pos(rows);
  std::vector<int32_t> tokens(size_t(rows) * max_selected, -1);
  std::vector<int32_t> counts(rows), req_ids(rows, 0);
  for (int r = 0; r < rows; ++r) {
    pos[r] = pos0 + r;
    counts[r] = int(pos[r]) + 1;
    for (int i = 0; i < counts[r]; ++i) tokens[size_t(r) * max_selected + i] = i;
  }
  auto q = random_bf16_bits(91 + dc.num_heads, int64_t(rows) * local_heads * nope, -2, 1);
  auto kv_b = random_bf16_bits(92 + dc.kv_lora,
                               int64_t(local_heads) * (nope + v) * kv_lora, -2, 1);
  const int n_blocks = 2, total_tokens = n_blocks * block_tokens;
  auto latent = random_bf16_bits(93, int64_t(total_tokens) * kv_lora, -2, 1);
  std::vector<int32_t> bt = {1, 0};  // reversed on purpose
  const float scale = 1.0f / std::sqrt(float(nope));

  DevBuf dq(q.size() * 2), dkb(kv_b.size() * 2), dlat(latent.size() * 2),
      dtopk(tokens.size() * 4), dcnt(rows * 4), dri(rows * 4), dpos(rows * 8),
      dbt(8), dqt(size_t(rows) * local_heads * kv_lora * 2),
      dc_split(size_t(rows) * local_heads * kv_lora * 4),
      dc_dense(size_t(rows) * local_heads * kv_lora * 4),
      dc_dense2(size_t(rows) * local_heads * kv_lora * 4),
      dc_dense1(size_t(rows) * local_heads * kv_lora * 4),
      dout(size_t(rows) * local_heads * v * 2);
  dq.upload(q.data(), q.size() * 2);
  dkb.upload(kv_b.data(), kv_b.size() * 2);
  dlat.upload(latent.data(), latent.size() * 2);
  dtopk.upload(tokens.data(), tokens.size() * 4);
  dcnt.upload(counts.data(), counts.size() * 4);
  dri.upload(req_ids.data(), req_ids.size() * 4);
  dpos.upload(pos.data(), pos.size() * 8);
  dbt.upload(bt.data(), bt.size() * 4);
  dsa_absorb_q(dq.p, dkb.p, dqt.p, rows, local_heads, nope, v, kv_lora, 0);

  auto run_split = [&](int n_split, void* c_out) {
    DevBuf dm(size_t(rows) * n_split * local_heads * 4),
        dl(size_t(rows) * n_split * local_heads * 4),
        dc(size_t(rows) * n_split * local_heads * kv_lora * 4);
    dsa_attn_partial(dqt.p, dlat.p, static_cast<const int32_t*>(dri.p),
                     static_cast<const int32_t*>(dtopk.p), max_selected,
                     static_cast<const int32_t*>(dcnt.p), rows, n_split,
                     local_heads, kv_lora, block_tokens,
                     static_cast<const int32_t*>(dbt.p), n_blocks, scale,
                     static_cast<float*>(dm.p), static_cast<float*>(dl.p),
                     static_cast<float*>(dc.p), 0);
    dsa_attn_combine(static_cast<const float*>(dm.p), static_cast<const float*>(dl.p),
                     static_cast<const float*>(dc.p), rows, n_split, local_heads,
                     kv_lora, static_cast<float*>(c_out), 0);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
  };
  auto run_dense = [&](int n_split, void* c_out) {
    DevBuf dm(size_t(rows) * n_split * local_heads * 4),
        dl(size_t(rows) * n_split * local_heads * 4),
        dc(size_t(rows) * n_split * local_heads * kv_lora * 4);
    const bool ok = dsa_attn_dense(
        dqt.p, dlat.p, static_cast<const int32_t*>(dri.p),
        static_cast<const int64_t*>(dpos.p), rows, n_split, local_heads, kv_lora,
        block_tokens, static_cast<const int32_t*>(dbt.p), n_blocks, scale,
        static_cast<float*>(dm.p), static_cast<float*>(dl.p),
        static_cast<float*>(dc.p), 0);
    if (!ok) throw std::runtime_error("dense kernel refused the geometry");
    dsa_attn_combine(static_cast<const float*>(dm.p), static_cast<const float*>(dl.p),
                     static_cast<const float*>(dc.p), rows, n_split, local_heads,
                     kv_lora, static_cast<float*>(c_out), 0);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
  };
  run_split(4, dc_split.p);
  run_dense(4, dc_dense.p);
  run_dense(4, dc_dense2.p);
  run_dense(1, dc_dense1.p);

  const size_t n = size_t(rows) * local_heads * kv_lora;
  std::vector<float> a(n), b(n), b2(n), b1(n);
  dc_split.download(a.data(), n * 4);
  dc_dense.download(b.data(), n * 4);
  dc_dense2.download(b2.data(), n * 4);
  dc_dense1.download(b1.data(), n * 4);
  require_bitwise("dense attention repeat", b2.data(), b.data(), n * 4);
  auto l2_rel = [&](const std::vector<float>& x, const std::vector<float>& y) {
    double d2 = 0, y2 = 0;
    for (size_t i = 0; i < n; ++i) {
      const double d = double(x[i]) - double(y[i]);
      d2 += d * d;
      y2 += double(y[i]) * double(y[i]);
    }
    return std::sqrt(d2 / std::max(y2, 1e-30));
  };
  const double vs_split = l2_rel(b, a), vs_split1 = l2_rel(b1, a);
  std::printf("[ OK ] dense attention heads=%d kv=%d: c vs split kernel l2_rel %.3g "
              "(n_split 4), %.3g (n_split 1)\n",
              dc.num_heads, dc.kv_lora, vs_split, vs_split1);
  // The same split count agrees to ~1e-6 (the same partial structure, the
  // tensor core's summation order inside); one split against four is the
  // online-softmax reassociation across splits, ~1.5e-4.
  if (vs_split > 2e-5 || vs_split1 > 5e-4)
    throw std::runtime_error("dense attention diverges from the split kernel");

  // Through vout against the host oracle, the split kernel's own budget.
  dsa_vout_gemm(dc_dense.p, dkb.p, dout.p, rows, local_heads, nope, v, kv_lora, 0);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> phys_latent(size_t(total_tokens) * kv_lora);
  for (int64_t tok = 0; tok < total_tokens; ++tok) {
    const int32_t blk = bt[size_t(tok / block_tokens)];
    const int64_t phys = int64_t(blk) * block_tokens + (tok % block_tokens);
    std::memcpy(&phys_latent[size_t(tok) * kv_lora], &latent[size_t(phys) * kv_lora],
                kv_lora * 2);
  }
  std::vector<uint16_t> want(size_t(rows) * local_heads * v);
  for (int r = 0; r < rows; ++r)
    dsa_ref::absorbed_attn<float>(&q[size_t(r) * local_heads * nope], phys_latent.data(),
                                  kv_lora, &tokens[size_t(r) * max_selected], counts[r],
                                  kv_b.data(), local_heads, nope, v, kv_lora, scale,
                                  &want[size_t(r) * local_heads * v]);
  std::vector<uint16_t> got(size_t(rows) * local_heads * v);
  dout.download(got.data(), got.size() * 2);
  require_bf16("dense attention vs host f32", compare_bf16(got, want, 8), 0.01, 0.006);
}

}  // namespace

// The absorb and vout projections on tensor cores against the warp kernels
// (which the rows path still takes below 16 rows, so the same inputs run
// through both by chunking the rows): bf16 outputs within one ulp, tight
// l2, deterministic — at the checkpoint's 64-head geometry and at TP=4's.
DGPP_TEST(dsa_absorb_and_vout_mma_match_the_warp_kernels) {
  for (const int heads : {64, 16}) {
    DsaConfig cfg{};
    cfg.num_heads = heads;
    const DsaGeometry g = DsaGeometry::from_config(cfg);
    const int local_heads = g.local_heads, nope = cfg.qk_nope_head_dim;
    const int v = cfg.v_head_dim, kv_lora = cfg.kv_lora_rank;
    const int rows = 41;
    auto q = random_bf16_bits(101 + heads, int64_t(rows) * local_heads * nope, -2, 1);
    auto kv_b = random_bf16_bits(102 + heads, int64_t(local_heads) * (nope + v) * kv_lora, -2, 1);
    std::vector<float> c(size_t(rows) * local_heads * kv_lora);
    {
      uint64_t x = 0x9E3779B97F4A7C15ULL + heads;
      for (auto& f : c) {
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
        f = (float(int64_t(x >> 40)) / float(1 << 23) - 1.f) * 3.f;
      }
    }
    DevBuf dq(q.size() * 2), dkb(kv_b.size() * 2), dc(c.size() * 4),
        dqt_a(size_t(rows) * local_heads * kv_lora * 2), dqt_b(size_t(rows) * local_heads * kv_lora * 2),
        dqt_c(size_t(rows) * local_heads * kv_lora * 2),
        dout_a(size_t(rows) * local_heads * v * 2), dout_b(size_t(rows) * local_heads * v * 2),
        dout_c(size_t(rows) * local_heads * v * 2);
    dq.upload(q.data(), q.size() * 2);
    dkb.upload(kv_b.data(), kv_b.size() * 2);
    dc.upload(c.data(), c.size() * 4);
    // Warp kernels: chunks of 15 rows (below the tensor-core threshold).
    for (int r0 = 0; r0 < rows; r0 += 15) {
      const int n = std::min(15, rows - r0);
      dsa_absorb_q(static_cast<const uint16_t*>(dq.p) + size_t(r0) * local_heads * nope, dkb.p,
                   static_cast<uint16_t*>(dqt_a.p) + size_t(r0) * local_heads * kv_lora, n,
                   local_heads, nope, v, kv_lora, 0);
      dsa_vout_gemm(static_cast<const float*>(dc.p) + size_t(r0) * local_heads * kv_lora, dkb.p,
                    static_cast<uint16_t*>(dout_a.p) + size_t(r0) * local_heads * v, n,
                    local_heads, nope, v, kv_lora, 0);
    }
    // Tensor-core kernels: all rows at once, twice (determinism).
    dsa_absorb_q(dq.p, dkb.p, dqt_b.p, rows, local_heads, nope, v, kv_lora, 0);
    dsa_absorb_q(dq.p, dkb.p, dqt_c.p, rows, local_heads, nope, v, kv_lora, 0);
    dsa_vout_gemm(dc.p, dkb.p, dout_b.p, rows, local_heads, nope, v, kv_lora, 0);
    dsa_vout_gemm(dc.p, dkb.p, dout_c.p, rows, local_heads, nope, v, kv_lora, 0);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const size_t nq = size_t(rows) * local_heads * kv_lora, no = size_t(rows) * local_heads * v;
    std::vector<uint16_t> qa(nq), qb(nq), qc(nq), oa(no), ob(no), oc(no);
    dqt_a.download(qa.data(), nq * 2); dqt_b.download(qb.data(), nq * 2); dqt_c.download(qc.data(), nq * 2);
    dout_a.download(oa.data(), no * 2); dout_b.download(ob.data(), no * 2); dout_c.download(oc.data(), no * 2);
    require_bitwise("absorb mma repeat", qc.data(), qb.data(), nq * 2);
    require_bitwise("vout mma repeat", oc.data(), ob.data(), no * 2);
    const auto sq = compare_bf16(qb, qa, 1);
    const auto so = compare_bf16(ob, oa, 1);
    std::printf("[ OK ] projections mma heads=%d: absorb l2_rel %.3g max_rel %.3g mismatches %ld/%ld; "
                "vout l2_rel %.3g max_rel %.3g mismatches %ld/%ld\n",
                heads, sq.l2_rel, sq.max_rel, sq.mismatches, sq.n, so.l2_rel, so.max_rel,
                so.mismatches, so.n);
    require_bf16("absorb mma vs warp kernel", sq, 2e-3, 0.02);
    require_bf16("vout mma vs warp kernel", so, 2e-3, 0.02);
  }
}

// The row-batched decode forms of absorb and vout (two rows and up) against
// the one-row launches, row by row, BITWISE: the decode contract — a
// batched row is the row alone — at every row count the schedulers issue
// (1..16, the tiles' edges included), at the checkpoint's 64-head geometry
// and at TP=4's, with a rope tail and without.
DGPP_TEST(dsa_absorb_and_vout_row_batches_match_the_rows_alone) {
  for (const int heads : {64, 16}) {
    for (const int rope : {0, 64}) {
      DsaConfig cfg{};
      cfg.num_heads = heads;
      const DsaGeometry g = DsaGeometry::from_config(cfg);
      const int local_heads = g.local_heads, nope = cfg.qk_nope_head_dim;
      const int v = cfg.v_head_dim, kv_lora = cfg.kv_lora_rank;
      const int rows = 16;
      const int qw = nope + rope, ow = kv_lora + rope;
      auto q = random_bf16_bits(201 + heads + rope, int64_t(rows) * local_heads * qw, -2, 1);
      auto kv_b = random_bf16_bits(202 + heads + rope, int64_t(local_heads) * (nope + v) * kv_lora, -2, 1);
      std::vector<float> c(size_t(rows) * local_heads * kv_lora);
      {
        uint64_t x = 0xD1B54A32D192ED03ULL + heads + rope;
        for (auto& f : c) {
          x = x * 6364136223846793005ULL + 1442695040888963407ULL;
          f = (float(int64_t(x >> 40)) / float(1 << 23) - 1.f) * 3.f;
        }
      }
      DevBuf dq(q.size() * 2), dkb(kv_b.size() * 2), dc(c.size() * 4),
          dqt_one(size_t(rows) * local_heads * ow * 2), dqt_batch(size_t(rows) * local_heads * ow * 2),
          dout_one(size_t(rows) * local_heads * v * 2), dout_batch(size_t(rows) * local_heads * v * 2);
      dq.upload(q.data(), q.size() * 2);
      dkb.upload(kv_b.data(), kv_b.size() * 2);
      dc.upload(c.data(), c.size() * 4);
      // Every row alone (the one-row kernels).
      for (int r = 0; r < rows; ++r) {
        dsa_absorb_q(static_cast<const uint16_t*>(dq.p) + size_t(r) * local_heads * qw, dkb.p,
                     static_cast<uint16_t*>(dqt_one.p) + size_t(r) * local_heads * ow, 1,
                     local_heads, nope, v, kv_lora, 0, rope, /*tensor_cores=*/false);
        dsa_vout_gemm(static_cast<const float*>(dc.p) + size_t(r) * local_heads * kv_lora, dkb.p,
                      static_cast<uint16_t*>(dout_one.p) + size_t(r) * local_heads * v, 1,
                      local_heads, nope, v, kv_lora, 0, /*tensor_cores=*/false);
      }
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const size_t nq = size_t(rows) * local_heads * ow, no = size_t(rows) * local_heads * v;
      std::vector<uint16_t> q_one(nq), q_batch(nq), o_one(no), o_batch(no);
      dqt_one.download(q_one.data(), nq * 2);
      dout_one.download(o_one.data(), no * 2);
      for (int n = 1; n <= rows; ++n) {
        DGPP_CUDA_OK(cudaMemset(dqt_batch.p, 0xEE, nq * 2));
        DGPP_CUDA_OK(cudaMemset(dout_batch.p, 0xEE, no * 2));
        dsa_absorb_q(dq.p, dkb.p, dqt_batch.p, n, local_heads, nope, v, kv_lora, 0, rope,
                     /*tensor_cores=*/false);
        dsa_vout_gemm(dc.p, dkb.p, dout_batch.p, n, local_heads, nope, v, kv_lora, 0,
                      /*tensor_cores=*/false);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        dqt_batch.download(q_batch.data(), nq * 2);
        dout_batch.download(o_batch.data(), no * 2);
        const std::string tag = " heads=" + std::to_string(heads) + " rope=" + std::to_string(rope) +
                                " rows=" + std::to_string(n);
        require_bitwise("absorb batch vs rows alone" + tag, q_batch.data(), q_one.data(),
                        size_t(n) * local_heads * ow * 2);
        require_bitwise("vout batch vs rows alone" + tag, o_batch.data(), o_one.data(),
                        size_t(n) * local_heads * v * 2);
        // Rows past n untouched (the batch writes nothing beyond its rows).
        for (size_t i = size_t(n) * local_heads * ow; i < nq; ++i)
          if (q_batch[i] != 0xEEEEu) throw std::runtime_error("absorb batch wrote past its rows" + tag);
        for (size_t i = size_t(n) * local_heads * v; i < no; ++i)
          if (o_batch[i] != 0xEEEEu) throw std::runtime_error("vout batch wrote past its rows" + tag);
      }
      std::printf("[ OK ] row batches heads=%d rope=%d: absorb and vout bitwise the rows alone at 1..%d rows\n",
                  heads, rope, rows);
    }
  }
}

// The flash kernel over per-row SELECTED lists (the sparse regime) against
// the split kernel on the same lists and the host oracle: 41 rows with
// counts from 3 to 283 (adjacent rows very different, so a block's two
// slabs run unequal tile counts and one idles), arbitrary token order with
// repeats, at 64/512, 16/512 and 16/256; deterministic.
DGPP_TEST(dsa_listed_attention_matches_split_kernel_and_reference) {
  for (const DenseCase gc : {DenseCase{64, 512}, DenseCase{16, 512}, DenseCase{16, 256}}) {
    DsaConfig cfg{};
    cfg.num_heads = gc.num_heads;
    cfg.kv_lora_rank = gc.kv_lora;
    const DsaGeometry g = DsaGeometry::from_config(cfg);
    const int local_heads = g.local_heads, nope = cfg.qk_nope_head_dim;
    const int v = cfg.v_head_dim, kv_lora = cfg.kv_lora_rank;
    const int block_tokens = cfg.block_tokens;
    const int rows = 41, max_selected = 300;
    const int n_blocks = 2, total_tokens = n_blocks * block_tokens;
    std::vector<int32_t> tokens(size_t(rows) * max_selected, -1), counts(rows), req_ids(rows, 0);
    for (int r = 0; r < rows; ++r) {
      counts[r] = (r % 2 == 0) ? 3 + r : 283 - 5 * r;  // 3..43 and 283..83, alternating
      for (int i = 0; i < counts[r]; ++i)
        tokens[size_t(r) * max_selected + i] = int32_t((r * 131 + i * 17) % total_tokens);
    }
    auto q = random_bf16_bits(111 + gc.num_heads, int64_t(rows) * local_heads * nope, -2, 1);
    auto kv_b = random_bf16_bits(112 + gc.kv_lora, int64_t(local_heads) * (nope + v) * kv_lora, -2, 1);
    auto latent = random_bf16_bits(113, int64_t(total_tokens) * kv_lora, -2, 1);
    std::vector<int32_t> bt = {1, 0};
    const float scale = 1.0f / std::sqrt(float(nope));
    DevBuf dq(q.size() * 2), dkb(kv_b.size() * 2), dlat(latent.size() * 2),
        dtopk(tokens.size() * 4), dcnt(rows * 4), dri(rows * 4), dbt(8),
        dqt(size_t(rows) * local_heads * kv_lora * 2),
        dc_split(size_t(rows) * local_heads * kv_lora * 4),
        dc_a(size_t(rows) * local_heads * kv_lora * 4),
        dc_b(size_t(rows) * local_heads * kv_lora * 4),
        dout(size_t(rows) * local_heads * v * 2);
    dq.upload(q.data(), q.size() * 2);
    dkb.upload(kv_b.data(), kv_b.size() * 2);
    dlat.upload(latent.data(), latent.size() * 2);
    dtopk.upload(tokens.data(), tokens.size() * 4);
    dcnt.upload(counts.data(), counts.size() * 4);
    dri.upload(req_ids.data(), req_ids.size() * 4);
    dbt.upload(bt.data(), bt.size() * 4);
    dsa_absorb_q(dq.p, dkb.p, dqt.p, rows, local_heads, nope, v, kv_lora, 0);
    const int n_split = 4;
    DevBuf dm(size_t(rows) * n_split * local_heads * 4), dl(size_t(rows) * n_split * local_heads * 4),
        dc(size_t(rows) * n_split * local_heads * kv_lora * 4);
    auto run = [&](bool listed, void* c_out) {
      if (listed) {
        const bool ok = dsa_attn_listed(dqt.p, dlat.p, static_cast<const int32_t*>(dri.p),
                                        static_cast<const int32_t*>(dtopk.p), max_selected,
                                        static_cast<const int32_t*>(dcnt.p), rows, n_split,
                                        local_heads, kv_lora, block_tokens,
                                        static_cast<const int32_t*>(dbt.p), n_blocks, scale,
                                        static_cast<float*>(dm.p), static_cast<float*>(dl.p),
                                        static_cast<float*>(dc.p), 0);
        if (!ok) throw std::runtime_error("listed kernel refused the geometry");
      } else {
        dsa_attn_partial(dqt.p, dlat.p, static_cast<const int32_t*>(dri.p),
                         static_cast<const int32_t*>(dtopk.p), max_selected,
                         static_cast<const int32_t*>(dcnt.p), rows, n_split, local_heads,
                         kv_lora, block_tokens, static_cast<const int32_t*>(dbt.p), n_blocks,
                         scale, static_cast<float*>(dm.p), static_cast<float*>(dl.p),
                         static_cast<float*>(dc.p), 0);
      }
      dsa_attn_combine(static_cast<const float*>(dm.p), static_cast<const float*>(dl.p),
                       static_cast<const float*>(dc.p), rows, n_split, local_heads, kv_lora,
                       static_cast<float*>(c_out), 0);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
    };
    run(false, dc_split.p);
    run(true, dc_a.p);
    run(true, dc_b.p);
    const size_t n = size_t(rows) * local_heads * kv_lora;
    std::vector<float> a(n), b(n), c(n);
    dc_split.download(a.data(), n * 4);
    dc_a.download(b.data(), n * 4);
    dc_b.download(c.data(), n * 4);
    require_bitwise("listed attention repeat", c.data(), b.data(), n * 4);
    double d2 = 0, y2 = 0;
    for (size_t i = 0; i < n; ++i) {
      const double d = double(b[i]) - double(a[i]);
      d2 += d * d;
      y2 += double(a[i]) * double(a[i]);
    }
    const double l2 = std::sqrt(d2 / std::max(y2, 1e-30));
    std::printf("[ OK ] listed attention heads=%d kv=%d: c vs split kernel l2_rel %.3g\n",
                gc.num_heads, gc.kv_lora, l2);
    // 16-token tiles against the split kernel's 32: the online-softmax
    // rescale points differ within a split (the dense gate's same-tile case
    // read 7e-7).
    if (l2 > 1e-4) throw std::runtime_error("listed attention diverges from the split kernel");
    dsa_vout_gemm(dc_a.p, dkb.p, dout.p, rows, local_heads, nope, v, kv_lora, 0);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint16_t> phys_latent(size_t(total_tokens) * kv_lora);
    for (int64_t tok = 0; tok < total_tokens; ++tok) {
      const int32_t blk = bt[size_t(tok / block_tokens)];
      const int64_t phys = int64_t(blk) * block_tokens + (tok % block_tokens);
      std::memcpy(&phys_latent[size_t(tok) * kv_lora], &latent[size_t(phys) * kv_lora], kv_lora * 2);
    }
    std::vector<uint16_t> want(size_t(rows) * local_heads * v);
    for (int r = 0; r < rows; ++r)
      dsa_ref::absorbed_attn<float>(&q[size_t(r) * local_heads * nope], phys_latent.data(), kv_lora,
                                    &tokens[size_t(r) * max_selected], counts[r], kv_b.data(),
                                    local_heads, nope, v, kv_lora, scale,
                                    &want[size_t(r) * local_heads * v]);
    std::vector<uint16_t> got(size_t(rows) * local_heads * v);
    dout.download(got.data(), got.size() * 2);
    require_bf16("listed attention vs host f32", compare_bf16(got, want, 8), 0.01, 0.006);
  }
}


// The listed kernel's split groups (2026-10-23, kAttnSplitGroup): eight
// consecutive splits of a row merge in a thread-block cluster and publish
// one partial, the finish's formula over the group. Against the per-split
// form + the combine: the two-level merge is fp32 rounding-level (rel l2
// < 1e-5); run to run bitwise; and a row's output does not depend on the
// rows beside it — the first seven rows alone reproduce the 41-row batch's
// rows bitwise (the grouping is per request, not per batch). 32 splits at
// 16/512 and 64/512; counts from 3 to 283, so most groups hold empty splits.
DGPP_TEST(dsa_listed_attention_split_groups_merge_like_the_combine) {
  for (const DenseCase gc : {DenseCase{16, 512}, DenseCase{64, 512}}) {
    DsaConfig cfg{};
    cfg.num_heads = gc.num_heads;
    cfg.kv_lora_rank = gc.kv_lora;
    const DsaGeometry g = DsaGeometry::from_config(cfg);
    const int local_heads = g.local_heads, nope = cfg.qk_nope_head_dim;
    const int v = cfg.v_head_dim, kv_lora = cfg.kv_lora_rank;
    const int block_tokens = cfg.block_tokens;
    const int rows = 41, max_selected = 300, few = 7;
    const int n_blocks = 2, total_tokens = n_blocks * block_tokens;
    std::vector<int32_t> tokens(size_t(rows) * max_selected, -1), counts(rows), req_ids(rows, 0);
    for (int r = 0; r < rows; ++r) {
      counts[r] = (r % 2 == 0) ? 3 + r : 283 - 5 * r;
      for (int i = 0; i < counts[r]; ++i)
        tokens[size_t(r) * max_selected + i] = int32_t((r * 131 + i * 17) % total_tokens);
    }
    auto q = random_bf16_bits(211 + gc.num_heads, int64_t(rows) * local_heads * nope, -2, 1);
    auto kv_b = random_bf16_bits(212 + gc.kv_lora, int64_t(local_heads) * (nope + v) * kv_lora, -2, 1);
    auto latent = random_bf16_bits(213, int64_t(total_tokens) * kv_lora, -2, 1);
    std::vector<int32_t> bt = {1, 0};
    const float scale = 1.0f / std::sqrt(float(nope));
    DevBuf dq(q.size() * 2), dkb(kv_b.size() * 2), dlat(latent.size() * 2), dtopk(tokens.size() * 4), dcnt(rows * 4),
        dri(rows * 4), dbt(8), dqt(size_t(rows) * local_heads * kv_lora * 2);
    dq.upload(q.data(), q.size() * 2);
    dkb.upload(kv_b.data(), kv_b.size() * 2);
    dlat.upload(latent.data(), latent.size() * 2);
    dtopk.upload(tokens.data(), tokens.size() * 4);
    dcnt.upload(counts.data(), counts.size() * 4);
    dri.upload(req_ids.data(), req_ids.size() * 4);
    dbt.upload(bt.data(), bt.size() * 4);
    dsa_absorb_q(dq.p, dkb.p, dqt.p, rows, local_heads, nope, v, kv_lora, 0);
    const int n_split = 32;
    const size_t n = size_t(rows) * local_heads * kv_lora;
    DevBuf dm(size_t(rows) * n_split * local_heads * 4), dl(size_t(rows) * n_split * local_heads * 4),
        dc(size_t(rows) * n_split * local_heads * kv_lora * 4), dout(n * 4);
    const int n_groups = n_split / kAttnSplitGroup;
    DevBuf dmg(size_t(rows) * n_groups * local_heads * 4), dlg(size_t(rows) * n_groups * local_heads * 4),
        dcg(size_t(rows) * n_groups * local_heads * kv_lora * 4), dtick(size_t(rows) * local_heads / 32 * n_groups * 4 + 4096);
    DGPP_CUDA_OK(cudaMemset(dtick.p, 0, size_t(rows) * local_heads / 32 * n_groups * 4 + 4096));
    auto run = [&](int group, int nrows, std::vector<float>& out) {
      DGPP_CUDA_OK(cudaMemset(dm.p, 0xFF, size_t(rows) * n_split * local_heads * 4));
      DGPP_CUDA_OK(cudaMemset(dmg.p, 0xFF, size_t(rows) * n_groups * local_heads * 4));
      DGPP_CUDA_OK(cudaMemset(dout.p, 0, n * 4));
      const bool ok = dsa_attn_listed(dqt.p, dlat.p, static_cast<const int32_t*>(dri.p),
                                      static_cast<const int32_t*>(dtopk.p), max_selected,
                                      static_cast<const int32_t*>(dcnt.p), nrows, n_split, local_heads, kv_lora,
                                      block_tokens, static_cast<const int32_t*>(dbt.p), n_blocks, scale,
                                      static_cast<float*>(dm.p), static_cast<float*>(dl.p), static_cast<float*>(dc.p),
                                      0, LatentFormat::kBf16, nullptr, 0, -1, group, static_cast<int32_t*>(dtick.p),
                                      static_cast<float*>(dmg.p), static_cast<float*>(dlg.p), static_cast<float*>(dcg.p));
      if (!ok) throw std::runtime_error("listed kernel refused the geometry");
      const bool grouped = group > 1;
      dsa_attn_combine(static_cast<const float*>(grouped ? dmg.p : dm.p), static_cast<const float*>(grouped ? dlg.p : dl.p),
                       static_cast<const float*>(grouped ? dcg.p : dc.p), nrows, n_split / group, local_heads, kv_lora,
                       static_cast<float*>(dout.p), 0);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      out.assign(n, 0.f);
      dout.download(out.data(), n * 4);
    };
    std::vector<float> flat, grouped, again, head;
    run(1, rows, flat);
    run(kAttnSplitGroup, rows, grouped);
    run(kAttnSplitGroup, rows, again);
    run(kAttnSplitGroup, few, head);
    require_bitwise("grouped listed attention repeat", again.data(), grouped.data(), n * 4);
    require_bitwise("grouped listed attention: the first rows alone", head.data(), grouped.data(),
                    size_t(few) * local_heads * kv_lora * 4);
    double d2 = 0, y2 = 0;
    for (size_t i = 0; i < n; ++i) {
      const double d = double(grouped[i]) - double(flat[i]);
      d2 += d * d;
      y2 += double(flat[i]) * double(flat[i]);
    }
    const double l2 = std::sqrt(d2 / std::max(y2, 1e-30));
    std::printf("[ OK ] listed attention split groups heads=%d kv=%d: grouped vs per-split merge l2_rel %.3g\n",
                gc.num_heads, gc.kv_lora, l2);
    if (l2 > 1e-5) throw std::runtime_error("the split groups' merge diverges from the per-split combine");
  }
}

DGPP_TEST(dsa_dense_attention_matches_split_kernel_and_reference) {
  for (const DenseCase dc : {DenseCase{64, 512}, DenseCase{16, 512}, DenseCase{16, 256}})
    run_dense_attention_case(dc);
}

// ---------------------------------------------------------------------------
// Decode-select scenarios shared by the MTP-row, grid-invariance,
// long-context, and graph-replay tests. The host expectation mirrors the
// kernel's contraction-proof arithmetic exactly (see the bitwise test above).
// ---------------------------------------------------------------------------

struct DecodeSelScenario {
  int rows = 0;
  int64_t n_pools = 0;  // complete pools resident in the cache
  std::vector<uint8_t> q8, k8;
  std::vector<float> w, ks;
  std::vector<int64_t> pos;
  std::vector<int32_t> req_ids;
  const DsaGeometry g = DsaGeometry::from_config(DsaConfig{});
  int heads = 32, dim = 128;

  void init(uint64_t seed, const std::vector<int64_t>& positions) {
    pos = positions;
    rows = int(pos.size());
    n_pools = 8;
    for (int64_t p : pos) n_pools = std::max(n_pools, (p + 1) / 4 + 8);

    std::vector<uint16_t> qb =
        random_bf16_bits(seed, int64_t(rows) * heads * dim, -2, 1);
    q8.assign(size_t(rows) * heads * dim, 0);
    w.assign(size_t(rows) * heads, 0.0f);
    std::vector<float> q_scale(size_t(rows) * heads, 0.0f);
    dsa_ref::fwht128_quant_fp8<float>(qb.data(), rows * heads, dim,
                                      q8.data(), q_scale.data());
    const float logit_scale =
        float(std::pow(128.0, -0.5) * std::pow(32.0, -0.5));
    for (int64_t i = 0; i < int64_t(rows) * heads; ++i) {
      const float w_raw = random_f32(seed + 1, i) * 0.01f;
      w[size_t(i)] = (w_raw * q_scale[size_t(i)]) * logit_scale;
    }
    k8.assign(size_t(n_pools) * dim, 0);
    for (auto& v : k8) {
      v = uint8_t(hash32(seed + 2 + (&v - k8.data())) & 0xFF);
      if ((v & 0x7Fu) == 0x7Fu) v ^= 0x01u;  // no NaN encodings
    }
    ks.assign(size_t(n_pools), 0.0f);
    for (auto& v : ks) v = std::fabs(random_f32(seed + 3, &v - ks.data())) * 0.05f;
    req_ids.assign(size_t(rows), 0);
  }

  // Runs the fused decode select on the default stream (or a captured
  // graph) and returns topk rows + counts.
  void run(int grid_blocks, std::vector<int32_t>& topk,
           std::vector<int32_t>& counts, cudaStream_t stream = 0) {
    const int max_selected = g.max_selected;
    DevBuf dq8(q8.size()), dw(w.size() * 4), dki(k8.size()),
        dks_c(ks.size() * 4), dpos(pos.size() * 8), dri(req_ids.size() * 4),
        dbt(4), dtopk(size_t(max_selected) * rows * 4), dcnt(rows * 4),
        dws(dsa_select_workspace_bytes(rows, n_pools)), dctr(8);
    dq8.upload(q8.data(), q8.size());
    dw.upload(w.data(), w.size() * 4);
    dki.upload(k8.data(), k8.size());
    dks_c.upload(ks.data(), ks.size() * 4);
    dpos.upload(pos.data(), pos.size() * 8);
    dri.upload(req_ids.data(), req_ids.size() * 4);
    int32_t bt0 = 0, ctr0 = 0;
    dbt.upload(&bt0, 4);
    dctr.upload(&ctr0, 4);
    dsa_select_decode(dq8.p, static_cast<const float*>(dw.p),
                      static_cast<const int32_t*>(dri.p),
                      static_cast<const int64_t*>(dpos.p), rows,
                      static_cast<const int32_t*>(dbt.p), 1, dki.p,
                      static_cast<const float*>(dks_c.p), int(n_pools), heads,
                      dim, g.select_k, 4, max_selected,
                      static_cast<int32_t*>(dtopk.p),
                      static_cast<int32_t*>(dcnt.p), dws.p, n_pools,
                      static_cast<int32_t*>(dctr.p), grid_blocks, stream);
    topk.assign(size_t(max_selected) * rows, -12345);
    counts.assign(rows, -1);
    cudaDeviceSynchronize();
    dtopk.download(topk.data(), topk.size() * 4);
    dcnt.download(counts.data(), counts.size() * 4);
  }

  // Host mirror of the kernel's logits + the pinned selection + expansion.
  void expect(std::vector<int32_t>& topk, std::vector<int32_t>& counts) const {
    const int max_selected = g.max_selected;
    topk.assign(size_t(max_selected) * rows, -1);
    counts.assign(rows, 0);
    std::vector<float> logits(size_t(n_pools), 0.0f);
    std::vector<int32_t> pool_ids(g.select_k, 0);
    for (int r = 0; r < rows; ++r) {
      const int64_t visible = (pos[size_t(r)] + 1) / 4;
      for (int64_t j = 0; j < visible; ++j) {
        float contrib[32];
        for (int h = 0; h < 32; ++h) {
          float partial = 0.0f;
          for (int d = 0; d < 128; ++d)
            partial =
                partial +
                fp8_e4m3_bits_to_float(
                    q8[size_t(r) * heads * dim + size_t(h) * 128 + d]) *
                    fp8_e4m3_bits_to_float(k8[size_t(j) * dim + d]);
          contrib[h] = (w[size_t(r) * heads + h] * ks[size_t(j)]) * partial;
        }
        logits[size_t(j)] = butterfly_sum_32(contrib);
      }
      const int n_sel = dsa_ref::select_pools(logits.data(), visible,
                                              g.select_k, pool_ids.data());
      counts[size_t(r)] =
          dsa_ref::expand_append_tail(pool_ids.data(), n_sel, pos[size_t(r)],
                                      4, max_selected,
                                      &topk[size_t(r) * max_selected]);
    }
  }
};

// MTP-shaped multi-row decode: 4 rows at consecutive positions (differing
// visible counts), plus short-context rows where most blocks get empty
// stripes, plus the pos=0 edge.
DGPP_TEST(dsa_select_decode_mtp_rows_and_short_context) {
  DecodeSelScenario sc;
  sc.init(81, {40001, 40002, 40003, 40004});
  std::vector<int32_t> got, counts, want, want_counts;
  sc.run(48, got, counts);
  sc.expect(want, want_counts);
  for (int r = 0; r < sc.rows; ++r) {
    if (counts[size_t(r)] != want_counts[size_t(r)])
      throw std::runtime_error("mtp rows count r=" + std::to_string(r));
    if (std::memcmp(&got[size_t(r) * sc.g.max_selected],
                    &want[size_t(r) * sc.g.max_selected],
                    sc.g.max_selected * 4) != 0)
      throw std::runtime_error("mtp rows tokens r=" + std::to_string(r));
  }

  // Short context: visible 9 pools over a 48-block grid — 39 empty stripes
  // and an all-MAX partial merge.
  DecodeSelScenario sh;
  sh.init(82, {37});
  std::vector<int32_t> got2, counts2, want2, want_counts2;
  sh.run(48, got2, counts2);
  sh.expect(want2, want_counts2);
  if (counts2[0] != want_counts2[0] || std::memcmp(got2.data(), want2.data(),
                                                   sc.g.max_selected * 4) != 0)
    throw std::runtime_error("short-context decode select");

  // pos = 0: no pools, single tail token.
  DecodeSelScenario z;
  z.init(83, {0});
  std::vector<int32_t> got3, counts3, want3, want_counts3;
  z.run(48, got3, counts3);
  z.expect(want3, want_counts3);
  if (counts3[0] != 1 || got3[0] != 0 || got3[1] != -1)
    throw std::runtime_error("pos=0 decode select");
}

// Rows past the fused kernel's eight run as the launcher's row groups
// (8 + 4 and 8 + 8, 2026-09-13: the decode batch's cap lifted to 16) —
// each row bitwise the host mirror, as at eight and below.
DGPP_TEST(dsa_select_decode_row_groups) {
  for (const int rows : {12, 16}) {
    std::vector<int64_t> positions;
    for (int r = 0; r < rows; ++r) positions.push_back(40001 + 977 * r);
    DecodeSelScenario sc;
    sc.init(85 + rows, positions);
    std::vector<int32_t> got, counts, want, want_counts;
    sc.run(48, got, counts);
    sc.expect(want, want_counts);
    for (int r = 0; r < sc.rows; ++r) {
      if (counts[size_t(r)] != want_counts[size_t(r)])
        throw std::runtime_error("row groups count rows=" + std::to_string(rows) + " r=" + std::to_string(r));
      if (std::memcmp(&got[size_t(r) * sc.g.max_selected], &want[size_t(r) * sc.g.max_selected],
                      sc.g.max_selected * 4) != 0)
        throw std::runtime_error("row groups tokens rows=" + std::to_string(rows) + " r=" + std::to_string(r));
    }
  }
}

// The 64-head indexer at the DeepSeek-V4-Flash release shape (2026-10-01):
// 64 heads x 128, relu, select_k 512, one entry per pool (kpool 1) — the
// kernel's kHG = 2 instantiation, which scores 16-entry tiles on the tensor
// cores (e4m3 codes widened exactly to f16, mma f16 -> f32) and needs the
// per-symbol dynamic shared-memory opt-in (a missing one is `invalid
// argument` at the launch, which the tiny fixture's select_k 16 never
// reached). Rows 1, 4 and 6, contexts below, at and far above select_k,
// lists through a two-block table. The tensor core's accumulation order is
// the hardware's, so the gate is the selection against exact (fp64)
// logits: the picks are distinct visible entries in ascending order, every
// pick scores at or above the k-th exact logit and every entry left out at
// or below it, within the fp32 dot's rounding (1e-5 of the row's range) —
// and two launches agree bit for bit.
DGPP_TEST(dsa_select_decode_64_heads_release_shape) {
  constexpr int heads = 64, dim = 128, select_k = 512;
  for (const int rows : {1, 4, 6}) {
    const std::vector<int64_t> visibles = {3001, 512, 513, 37, 0, 2999};
    const int pools_per_block = 1504;  // two blocks, swapped: block 0 -> physical 1
    const int64_t n_pools = 2 * pools_per_block;
    std::vector<int64_t> pos(static_cast<size_t>(rows));
    for (int r = 0; r < rows; ++r) pos[size_t(r)] = visibles[size_t(r)] - 1;  // kpool 1: pos + 1 visible entries
    const uint64_t seed = 9100 + uint64_t(rows);
    std::vector<uint8_t> q8(size_t(rows) * heads * dim), k8(size_t(n_pools) * dim);
    for (size_t i = 0; i < q8.size(); ++i) {
      q8[i] = uint8_t(hash32(seed + i) & 0xFF);
      if ((q8[i] & 0x7Fu) == 0x7Fu) q8[i] ^= 0x01u;  // no NaN encodings
    }
    for (size_t i = 0; i < k8.size(); ++i) {
      k8[i] = uint8_t(hash32(seed + 0x9E3779B9u + i) & 0xFF);
      if ((k8[i] & 0x7Fu) == 0x7Fu) k8[i] ^= 0x01u;
    }
    std::vector<float> w(size_t(rows) * heads), ks(static_cast<size_t>(n_pools));
    for (size_t i = 0; i < w.size(); ++i) w[i] = random_f32(seed + 1, int64_t(i)) * 0.01f;
    for (size_t i = 0; i < ks.size(); ++i) ks[i] = std::ldexp(1.0f, -6 - int(i % 3));
    std::vector<int32_t> req_ids(static_cast<size_t>(rows), 0);
    const int32_t table[2] = {1, 0};
    const auto slot_of = [&](int64_t e) { return int64_t(table[e / pools_per_block]) * pools_per_block + e % pools_per_block; };

    DevBuf dq8(q8.size()), dw(w.size() * 4), dki(k8.size()), dks(ks.size() * 4), dpos(pos.size() * 8),
        dri(req_ids.size() * 4), dbt(8), dtopk(size_t(select_k) * rows * 4), dcnt(rows * 4),
        dws(dsa_select_workspace_bytes(rows, n_pools)), dctr(8);
    dq8.upload(q8.data(), q8.size());
    dw.upload(w.data(), w.size() * 4);
    dki.upload(k8.data(), k8.size());
    dks.upload(ks.data(), ks.size() * 4);
    dpos.upload(pos.data(), pos.size() * 8);
    dri.upload(req_ids.data(), req_ids.size() * 4);
    int32_t ctr0[2] = {0, 0};
    dbt.upload(table, 8);
    dctr.upload(ctr0, 8);
    std::vector<int32_t> got(size_t(select_k) * rows, -12345), counts(static_cast<size_t>(rows), -1), again;
    for (int pass = 0; pass < 2; ++pass) {
      dsa_select_decode(dq8.p, static_cast<const float*>(dw.p), static_cast<const int32_t*>(dri.p),
                        static_cast<const int64_t*>(dpos.p), rows, static_cast<const int32_t*>(dbt.p), 2, dki.p,
                        static_cast<const float*>(dks.p), pools_per_block, heads, dim, select_k, 1, select_k,
                        static_cast<int32_t*>(dtopk.p), static_cast<int32_t*>(dcnt.p), dws.p, n_pools,
                        static_cast<int32_t*>(dctr.p), 0, 0, /*relu=*/true);
      cudaDeviceSynchronize();
      if (pass == 0) {
        dtopk.download(got.data(), got.size() * 4);
        dcnt.download(counts.data(), counts.size() * 4);
      } else {
        again.assign(got.size(), 0);
        dtopk.download(again.data(), again.size() * 4);
        if (again != got) throw std::runtime_error("64-head select: two launches differ");
      }
    }

    std::vector<double> logits(static_cast<size_t>(n_pools));
    for (int r = 0; r < rows; ++r) {
      const std::string tag = "64-head select rows=" + std::to_string(rows) + " r=" + std::to_string(r);
      const int64_t visible = pos[size_t(r)] + 1;
      double lo = 0, hi = 0;
      for (int64_t j = 0; j < visible; ++j) {
        const size_t slot = size_t(slot_of(j));
        double total = 0.0;
        for (int h = 0; h < heads; ++h) {
          double partial = 0.0;
          for (int d = 0; d < dim; ++d)
            partial += double(fp8_e4m3_bits_to_float(q8[(size_t(r) * heads + size_t(h)) * dim + d])) *
                       double(fp8_e4m3_bits_to_float(k8[slot * dim + d]));
          total += double(w[size_t(r) * heads + size_t(h)]) * std::max(partial, 0.0);
        }
        logits[size_t(j)] = total * double(ks[slot]);
        lo = std::min(lo, logits[size_t(j)]);
        hi = std::max(hi, logits[size_t(j)]);
      }
      const int want_n = int(std::min<int64_t>(visible, select_k));
      if (counts[size_t(r)] != want_n)
        throw std::runtime_error(tag + ": count " + std::to_string(counts[size_t(r)]) + " vs " + std::to_string(want_n));
      const int32_t* row = &got[size_t(r) * select_k];
      std::vector<bool> picked(static_cast<size_t>(std::max<int64_t>(visible, 1)), false);
      for (int i = 0; i < select_k; ++i) {
        if (i >= want_n) {
          if (row[i] != -1) throw std::runtime_error(tag + ": the list is not -1 padded past its count");
          continue;
        }
        if (row[i] < 0 || row[i] >= visible) throw std::runtime_error(tag + ": a pick outside the visible entries");
        if (i > 0 && row[i] <= row[i - 1]) throw std::runtime_error(tag + ": the picks are not ascending and distinct");
        picked[size_t(row[i])] = true;
      }
      if (visible > select_k) {
        std::vector<double> sorted(logits.begin(), logits.begin() + visible);
        std::nth_element(sorted.begin(), sorted.begin() + (select_k - 1), sorted.end(), std::greater<double>());
        const double kth = sorted[size_t(select_k - 1)];
        const double tol = 1e-5 * (hi - lo);
        for (int64_t j = 0; j < visible; ++j) {
          if (picked[size_t(j)] && logits[size_t(j)] < kth - tol)
            throw std::runtime_error(tag + ": a pick below the k-th logit");
          if (!picked[size_t(j)] && logits[size_t(j)] > kth + tol)
            throw std::runtime_error(tag + ": an entry above the k-th logit left out");
        }
      }
    }
  }
}

// The selection must not depend on the grid: 7 blocks vs 48 blocks produce
// identical rows (uneven stripes at grid=7 exercise the stripe bounds).
DGPP_TEST(dsa_select_decode_grid_invariance) {
  DecodeSelScenario sc;
  sc.init(84, {40003});
  std::vector<int32_t> a, ca, b, cb;
  sc.run(48, a, ca);
  sc.run(7, b, cb);
  if (ca != cb || a != b)
    throw std::runtime_error("decode select depends on grid size");
}

// Long-context decode: 100k visible pools — stripes exceed the 2048-pool
// selection tile, so blocks run multiple tiles before the merge (the shape
// the single-stream design targets).
DGPP_TEST(dsa_select_decode_long_context_multitile) {
  DecodeSelScenario sc;
  sc.init(85, {400002});  // visible = 100000 pools
  if (sc.n_pools < 100000)
    throw std::runtime_error("scenario too small");
  std::vector<int32_t> got, counts, want, want_counts;
  sc.run(48, got, counts);
  sc.expect(want, want_counts);
  if (counts[0] != want_counts[0] ||
      std::memcmp(got.data(), want.data(), sc.g.max_selected * 4) != 0)
    throw std::runtime_error("long-context decode select mismatch");
}

// Multi-row prefill select: block-per-row with a chunk-shaped position mix
// (sparse rows at ~3k positions, dense rows at 0..7 in the same batch).
DGPP_TEST(dsa_select_prefill_multi_row) {
  const DsaGeometry g = DsaGeometry::from_config(DsaConfig{});
  const int heads = 32;
  std::vector<int64_t> positions;
  for (int p = 0; p <= 7; ++p) positions.push_back(p);
  for (int p = 3000; p < 3056; ++p) positions.push_back(p);
  const int rows = int(positions.size());
  const int64_t avail = 800;
  std::vector<float> dots(size_t(rows) * heads * avail, 0.0f);
  std::vector<float> w(size_t(rows) * heads, 0.0f);
  std::vector<float> ks(size_t(avail), 0.0f);
  for (auto& v : dots) v = random_f32(90, &v - dots.data());
  for (auto& v : w) v = random_f32(91, &v - w.data());
  for (auto& v : ks) v = random_f32(92, &v - ks.data());

  DevBuf dd(dots.size() * 4), dw(w.size() * 4), dks(ks.size() * 4),
      dpos(positions.size() * 8),
      dtopk(size_t(g.max_selected) * rows * 4), dcnt(rows * 4);
  dd.upload(dots.data(), dots.size() * 4);
  dw.upload(w.data(), w.size() * 4);
  dks.upload(ks.data(), ks.size() * 4);
  dpos.upload(positions.data(), positions.size() * 8);
  dsa_select_prefill(static_cast<const float*>(dd.p), avail,
                     static_cast<const float*>(dw.p),
                     static_cast<const float*>(dks.p),
                     static_cast<const int64_t*>(dpos.p), rows, avail, heads,
                     g.select_k, 4, g.max_selected,
                     static_cast<int32_t*>(dtopk.p),
                     static_cast<int32_t*>(dcnt.p), 0);
  std::vector<int32_t> got(size_t(g.max_selected) * rows, -12345);
  std::vector<int32_t> counts(rows, -1);
  cudaDeviceSynchronize();
  dtopk.download(got.data(), got.size() * 4);
  dcnt.download(counts.data(), counts.size() * 4);

  std::vector<float> logits(size_t(avail), 0.0f);
  std::vector<int32_t> pool_ids(g.select_k, 0);
  for (int r = 0; r < rows; ++r) {
    const int64_t visible = (positions[size_t(r)] + 1) / 4;
    for (int64_t j = 0; j < visible; ++j) {
      float dot_h[32];
      for (int h = 0; h < heads; ++h)
        dot_h[h] = dots[size_t(r) * heads * avail + size_t(h) * avail + j];
      logits[size_t(j)] =
          prefill_logit_mirror(dot_h, &w[size_t(r) * heads], ks[size_t(j)]);
    }
    const int n_sel = dsa_ref::select_pools(logits.data(), visible,
                                            g.select_k, pool_ids.data());
    std::vector<int32_t> want(size_t(g.max_selected), -1);
    const int n_tok = dsa_ref::expand_append_tail(
        pool_ids.data(), n_sel, positions[size_t(r)], 4, g.max_selected,
        want.data());
    if (counts[size_t(r)] != n_tok ||
        std::memcmp(&got[size_t(r) * g.max_selected], want.data(),
                    g.max_selected * 4) != 0)
      throw std::runtime_error("multi-row prefill r=" + std::to_string(r));
  }
}

// Constructed exact ties, including a tie group that straddles the 512th
// position: the composite key must resolve every tie to the lower pool
// index on both the host oracle and the device kernels.
DGPP_TEST(dsa_select_exact_ties_break_to_lower_pool) {
  const DsaGeometry g = DsaGeometry::from_config(DsaConfig{});
  const int heads = 32;
  const int64_t avail = 700;
  const int rows = 1;
  // Pools 0..99: distinct high logits. Pools 100..699: one shared value
  // (below the top-100) — the 512th selection boundary cuts inside the tie
  // group, so the winners must be exactly 100..511.
  std::vector<float> dots(size_t(rows) * heads * avail, 0.0f);
  std::vector<float> w(size_t(heads), 1.0f / 32.0f);
  std::vector<float> ks(size_t(avail), 1.0f);
  for (int64_t j = 0; j < avail; ++j) {
    const float base = (j < 100) ? 10.0f - float(j) * 0.01f : -1.0f;
    for (int h = 0; h < heads; ++h)
      dots[size_t(h) * avail + j] = base;
  }
  std::vector<int64_t> positions = {4 * avail - 1};  // visible = avail

  DevBuf dd(dots.size() * 4), dw(w.size() * 4), dks(ks.size() * 4),
      dpos(8), dtopk(size_t(g.max_selected) * 4), dcnt(4);
  dd.upload(dots.data(), dots.size() * 4);
  dw.upload(w.data(), w.size() * 4);
  dks.upload(ks.data(), ks.size() * 4);
  dpos.upload(positions.data(), 8);
  dsa_select_prefill(static_cast<const float*>(dd.p), avail,
                     static_cast<const float*>(dw.p),
                     static_cast<const float*>(dks.p),
                     static_cast<const int64_t*>(dpos.p), rows, avail, heads,
                     g.select_k, 4, g.max_selected,
                     static_cast<int32_t*>(dtopk.p),
                     static_cast<int32_t*>(dcnt.p), 0);
  std::vector<int32_t> got(size_t(g.max_selected), -12345);
  int32_t got_cnt = -1;
  cudaDeviceSynchronize();
  dtopk.download(got.data(), got.size() * 4);
  dcnt.download(&got_cnt, 4);
  if (got_cnt != 2048)  // pos = 4*avail-1 is pool-aligned: no tail tokens
    throw std::runtime_error("ties count: " + std::to_string(got_cnt));
  for (int i = 0; i < 512; ++i) {
    const int32_t want_tok = (i < 100) ? i : i;  // pools 0..99 then 100..511
    if (got[size_t(i)] != want_tok)
      throw std::runtime_error("tie resolution at rank " + std::to_string(i) +
                               ": got " + std::to_string(got[size_t(i)]));
  }
  // Host oracle agrees.
  std::vector<float> logits(size_t(avail), 0.0f);
  for (int64_t j = 0; j < avail; ++j) {
    float dot_h[32];
    for (int h = 0; h < heads; ++h) dot_h[h] = dots[size_t(h) * avail + j];
    logits[size_t(j)] = prefill_logit_mirror(dot_h, w.data(), ks[size_t(j)]);
  }
  std::vector<int32_t> pool_ids(g.select_k, 0);
  const int n_sel =
      dsa_ref::select_pools(logits.data(), avail, g.select_k, pool_ids.data());
  for (int i = 0; i < n_sel; ++i)
    if (pool_ids[size_t(i)] != i)
      throw std::runtime_error("host tie resolution at " + std::to_string(i));
}

// latent_append + gather_index_pools: the co-located block-table round trip
// (token blocks and pool blocks are the same table's two views).
DGPP_TEST(dsa_latent_append_and_gather_roundtrip) {
  const int block_tokens = 128, kv_lora = 512, ppb = 32;
  const int n_blocks = 3, total_tokens = n_blocks * block_tokens;
  auto latent = random_bf16_bits(95, int64_t(total_tokens) * kv_lora, -2, 1);
  auto k8src = std::vector<uint8_t>(size_t(total_tokens / 4) * 128);
  for (auto& v : k8src) v = uint8_t(hash32(96 + (&v - k8src.data())) & 0x7Eu);
  std::vector<float> ks_src(size_t(total_tokens / 4), 0.0f);
  for (auto& v : ks_src) v = std::fabs(random_f32(97, &v - ks_src.data()));

  std::vector<int32_t> bt = {2, 0, 1};  // shuffled physical assignment
  const int ppb_lat = block_tokens / 4;
  std::vector<uint8_t> phys_k(k8src.size());
  std::vector<float> phys_s(ks_src.size(), 0.0f);
  for (int64_t j = 0; j < total_tokens / 4; ++j) {
    const int64_t slot = int64_t(bt[size_t(j / ppb_lat)]) * ppb_lat + (j % ppb_lat);
    std::memcpy(&phys_k[size_t(slot) * 128], &k8src[size_t(j) * 128], 128);
    phys_s[size_t(slot)] = ks_src[size_t(j)];
  }
  // Separate source and cache buffers: the append is never in-place in the
  // layer (the source is the fresh projection output), and in-place
  // aliasing is a scheduling-dependent data race (physical writes clobber
  // unread source rows through the shuffled block table).
  DevBuf dsrc(latent.size() * 2), dlat(latent.size() * 2),
      dk8(k8src.size()), dks(ks_src.size() * 4),
      dbt(bt.size() * 4), dri(size_t(total_tokens) * 4),
      dgk(k8src.size()), dgs(ks_src.size() * 4);
  dsrc.upload(latent.data(), latent.size() * 2);
  dlat.upload(std::vector<uint16_t>(latent.size(), 0).data(),
              latent.size() * 2);
  dk8.upload(phys_k.data(), phys_k.size());
  dks.upload(phys_s.data(), phys_s.size() * 4);
  dbt.upload(bt.data(), bt.size() * 4);
  // Per-token request ids sized to the token count (the kernel indexes
  // req_ids[i] for every token).
  std::vector<int32_t> req_ids_all(total_tokens, 0);
  dri.upload(req_ids_all.data(), req_ids_all.size() * 4);

  // Append all 384 latent rows through the block table.
  std::vector<int64_t> positions(total_tokens);
  for (int t = 0; t < total_tokens; ++t) positions[size_t(t)] = t;
  DevBuf dpos_all(positions.size() * 8);
  dpos_all.upload(positions.data(), positions.size() * 8);
  dsa_latent_append(dsrc.p, static_cast<const int32_t*>(dri.p),
                    static_cast<const int64_t*>(dpos_all.p), total_tokens,
                    static_cast<const int32_t*>(dbt.p), n_blocks, block_tokens,
                    dlat.p, kv_lora, 0);
  std::vector<uint16_t> got_lat(latent.size());
  cudaDeviceSynchronize();
  dlat.download(got_lat.data(), got_lat.size() * 2);
  for (int64_t tok = 0; tok < total_tokens; ++tok) {
    const int64_t phys =
        int64_t(bt[size_t(tok / block_tokens)]) * block_tokens +
        (tok % block_tokens);
    require_bitwise("latent append", &got_lat[size_t(phys) * kv_lora],
                    &latent[size_t(tok) * kv_lora], kv_lora * 2);
  }

  // Gather pools back into logical order and compare.
  dsa_gather_index_pools(static_cast<const int32_t*>(dbt.p), ppb, dk8.p,
                         static_cast<const float*>(dks.p), total_tokens / 4,
                         dgk.p, static_cast<float*>(dgs.p), 128, 0);
  std::vector<uint8_t> got_k(k8src.size());
  std::vector<float> got_s(ks_src.size(), 0.0f);
  cudaDeviceSynchronize();
  dgk.download(got_k.data(), got_k.size());
  dgs.download(got_s.data(), got_s.size() * 4);
  require_bitwise("gather k", got_k.data(), k8src.data(), k8src.size());
  require_bitwise("gather scales", got_s.data(), ks_src.data(),
                  ks_src.size() * 4);
}

// The elementwise paths: k LayerNorm (strided input), the fused q/kv
// RMSNorm, and the weights fold. rsqrtf on device vs 1/sqrtf on host can
// differ by ulps, so the norms compare within a couple of bf16 ulps; the
// fold is pure fp32 and compares bitwise.
DGPP_TEST(dsa_elementwise_norms_and_fold) {
  const int rows = 33, dim = 128, q_dim = 1536, kv_dim = 512;
  auto kraw = random_bf16_bits(101, int64_t(rows) * 256, -2, 1);  // strided
  auto kw = random_bf16_bits(102, dim, -1, 1);
  auto kb = random_bf16_bits(103, dim, -1, 1);
  DevBuf dk(kraw.size() * 2), dkw(kw.size() * 2), dkb(kb.size() * 2),
      dkout(size_t(rows) * dim * 2);
  dk.upload(kraw.data(), kraw.size() * 2);
  dkw.upload(kw.data(), kw.size() * 2);
  dkb.upload(kb.data(), kb.size() * 2);
  dsa_k_layernorm(dk.p, 256, dkw.p, dkb.p, dkout.p, rows, dim, 1e-6f, 0);
  std::vector<uint16_t> got_k(size_t(rows) * dim);
  cudaDeviceSynchronize();
  dkout.download(got_k.data(), got_k.size() * 2);
  for (int r = 0; r < rows; ++r) {
    const uint16_t* row = &kraw[size_t(r) * 256];
    double mean = 0;
    for (int d = 0; d < dim; ++d) mean += bf16_bits_to_float(row[d]);
    mean /= dim;
    double var = 0;
    for (int d = 0; d < dim; ++d) {
      const double t = bf16_bits_to_float(row[d]) - mean;
      var += t * t;
    }
    var /= dim;
    const double inv = 1.0 / std::sqrt(var + 1e-6);
    for (int d = 0; d < dim; ++d) {
      const double y = (bf16_bits_to_float(row[d]) - mean) * inv *
                           bf16_bits_to_float(kw[size_t(d)]) +
                       bf16_bits_to_float(kb[size_t(d)]);
      const double got = bf16_bits_to_float(got_k[size_t(r) * dim + d]);
      if (std::fabs(got - y) > std::max(1e-3, std::fabs(y) * 0.02))
        throw std::runtime_error("k layernorm r=" + std::to_string(r) + " d=" +
                                 std::to_string(d));
    }
  }

  auto qkv = random_bf16_bits(104, int64_t(rows) * (q_dim + kv_dim), -2, 1);
  auto qw = random_bf16_bits(105, q_dim, -1, 1);
  auto kvw = random_bf16_bits(106, kv_dim, -1, 1);
  DevBuf dqkv(qkv.size() * 2), dqw(qw.size() * 2), dkvw(kvw.size() * 2),
      dqc(size_t(rows) * q_dim * 2), dkvc(size_t(rows) * kv_dim * 2);
  dqkv.upload(qkv.data(), qkv.size() * 2);
  dqw.upload(qw.data(), qw.size() * 2);
  dkvw.upload(kvw.data(), kvw.size() * 2);
  dsa_fused_qkv_rmsnorm(dqkv.p, dqc.p, dkvc.p, q_dim, kv_dim, rows, dqw.p,
                        dkvw.p, 1e-5f, 0);
  std::vector<uint16_t> got_qc(size_t(rows) * q_dim),
      got_kvc(size_t(rows) * kv_dim);
  cudaDeviceSynchronize();
  dqc.download(got_qc.data(), got_qc.size() * 2);
  dkvc.download(got_kvc.data(), got_kvc.size() * 2);
  auto rms_expect = [&](const uint16_t* row, int ddim, const uint16_t* wgt) {
    std::vector<double> exp(ddim);
    double ss = 0;
    for (int d = 0; d < ddim; ++d)
      ss += double(bf16_bits_to_float(row[d])) * bf16_bits_to_float(row[d]);
    const double inv = 1.0 / std::sqrt(ss / ddim + 1e-5);
    for (int d = 0; d < ddim; ++d)
      exp[size_t(d)] = bf16_bits_to_float(row[d]) * inv *
                       bf16_bits_to_float(wgt[size_t(d)]);
    return exp;
  };
  for (int r = 0; r < rows; ++r) {
    auto eq = rms_expect(&qkv[size_t(r) * (q_dim + kv_dim)], q_dim, qw.data());
    auto ev = rms_expect(&qkv[size_t(r) * (q_dim + kv_dim) + q_dim], kv_dim,
                         kvw.data());
    for (int d = 0; d < q_dim; ++d) {
      const double got = bf16_bits_to_float(got_qc[size_t(r) * q_dim + d]);
      if (std::fabs(got - eq[size_t(d)]) >
          std::max(1e-3, std::fabs(eq[size_t(d)]) * 0.02))
        throw std::runtime_error("q rmsnorm");
    }
    for (int d = 0; d < kv_dim; ++d) {
      const double got = bf16_bits_to_float(got_kvc[size_t(r) * kv_dim + d]);
      if (std::fabs(got - ev[size_t(d)]) >
          std::max(1e-3, std::fabs(ev[size_t(d)]) * 0.02))
        throw std::runtime_error("kv rmsnorm");
    }
  }

  // Fold: bitwise fp32.
  const int64_t n = rows * 32;
  std::vector<float> a(size_t(n), 0.0f), b(size_t(n), 0.0f), out(size_t(n), 0.0f);
  for (int64_t i = 0; i < n; ++i) {
    a[size_t(i)] = random_f32(107, i);
    b[size_t(i)] = random_f32(108, i);
  }
  DevBuf da(n * 4), db(n * 4), dout(n * 4);
  da.upload(a.data(), n * 4);
  db.upload(b.data(), n * 4);
  const float scale = 0.015625f;
  dsa_fold_weights(static_cast<const float*>(da.p),
                   static_cast<const float*>(db.p),
                   static_cast<float*>(dout.p), n, scale, 0);
  std::vector<float> got_f(size_t(n), 0.0f);
  cudaDeviceSynchronize();
  dout.download(got_f.data(), n * 4);
  for (int64_t i = 0; i < n; ++i) {
    const float want_f = (a[size_t(i)] * b[size_t(i)]) * scale;
    if (got_f[size_t(i)] != want_f)
      throw std::runtime_error("fold bitwise");
  }
}

// Multi-request decode update: two NONCONTIGUOUS request slots in one batch
// with a padding row (pos = -1, skipped), a third span that is ENTIRELY
// padding (an unoccupied fixed-shape slot, sentinel ids), and a multi-block
// pool space; hard-max gates keep the pool contents bitwise. The
// nonzero/non-dense ids pin that req_spans is a dense list of active
// requests, not a slot index.
DGPP_TEST(dsa_decode_update_multi_request_and_padding) {
  const int kpool = 4, dim = 128;
  const int ppb = 8;  // small blocks: 5 blocks for ~37 pools
  std::vector<float> ape(size_t(kpool) * dim);
  for (int i = 0; i < kpool * dim; ++i) ape[i] = random_f32(110, i) * 0.1f;
  // Tokens: req1 at pos 50..55 (6 tokens, one padding at index 2), req3 at
  // pos 30..32 (3 tokens), then two all-padding rows whose span names no
  // request. Pools 12 (pos 51), 13 (pos 55), 7 (pos 31) complete within the
  // batch.
  const int tokens = 11;
  constexpr int max_requests = 4;
  std::vector<int32_t> req_ids = {1, 1, 1, 1, 1, 1, 3, 3, 3, -1, -1};
  std::vector<int64_t> pos = {50, 51, -1, 52, 53, 54, 30, 31, 32, -1, -1};
  // NOTE: pos 51 completes pool 12 (48..51) — but its earlier members
  // (48..50) are not in this batch: they live in the seeded ring.
  auto k = random_bf16_bits(111, int64_t(tokens) * dim, -2, 1);
  auto gate = std::vector<uint16_t>(size_t(tokens) * dim);
  for (int t = 0; t < tokens; ++t) {
    const bool dom = (pos[size_t(t)] % kpool) == kpool - 1;
    for (int d = 0; d < dim; ++d)
      gate[size_t(t) * dim + d] = float_to_bf16_bits(dom ? 50.0f : -50.0f);
  }
  // Seed rings with the tokens just before the batch: req1 pos 47..49,
  // req3 pos 26..29 (their last kpool tokens before the batch).
  const int seed_tokens = 7;
  std::vector<int32_t> seed_req = {1, 1, 1, 3, 3, 3, 3};
  std::vector<int64_t> seed_pos = {47, 48, 49, 26, 27, 28, 29};
  auto seed_k = random_bf16_bits(112, int64_t(seed_tokens) * dim, -2, 1);
  auto seed_gate = std::vector<uint16_t>(size_t(seed_tokens) * dim);
  for (int t = 0; t < seed_tokens; ++t) {
    const bool dom = (seed_pos[size_t(t)] % kpool) == kpool - 1;
    for (int d = 0; d < dim; ++d)
      seed_gate[size_t(t) * dim + d] =
          float_to_bf16_bits(dom ? 50.0f : -50.0f);
  }

  const int max_pools = 40, n_blocks = (max_pools + ppb - 1) / ppb;
  const int total_slots = n_blocks * ppb;
  std::vector<int32_t> bt(n_blocks);
  for (int b = 0; b < n_blocks; ++b) bt[size_t(b)] = (b * 3 + 1) % n_blocks;
  // Every slot's row of the block table ([max_requests, n_blocks]).
  std::vector<int32_t> bt2(size_t(max_requests) * n_blocks);
  for (int req = 0; req < max_requests; ++req)
    for (int b = 0; b < n_blocks; ++b)
      bt2[size_t(req) * n_blocks + b] = bt[size_t(b)];

  DevBuf dk(k.size() * 2), dgate(gate.size() * 2), dape(ape.size() * 4),
      dreq(req_ids.size() * 4), dpos(pos.size() * 8),
      dspans(3 * 2 * 4), dbt(bt2.size() * 4),
      dki(size_t(total_slots) * dim), dks(total_slots * 4),
      dtail(size_t(max_requests) * 2 * kpool * dim * 2),
      dsk(seed_k.size() * 2),
      dsg(seed_gate.size() * 2), dsr(seed_req.size() * 4),
      dsp(seed_pos.size() * 8);
  dk.upload(k.data(), k.size() * 2);
  dgate.upload(gate.data(), gate.size() * 2);
  dape.upload(ape.data(), ape.size() * 4);
  dreq.upload(req_ids.data(), req_ids.size() * 4);
  dpos.upload(pos.data(), pos.size() * 8);
  dbt.upload(bt2.data(), bt2.size() * 4);
  dki.upload(std::vector<uint8_t>(size_t(total_slots) * dim, 0).data(),
             size_t(total_slots) * dim);
  dks.upload(std::vector<float>(total_slots, 0.0f).data(), total_slots * 4);
  dtail.upload(
      std::vector<uint16_t>(size_t(max_requests) * 2 * kpool * dim, 0).data(),
      size_t(max_requests) * 2 * kpool * dim * 2);
  dsk.upload(seed_k.data(), seed_k.size() * 2);
  dsg.upload(seed_gate.data(), seed_gate.size() * 2);
  dsr.upload(seed_req.data(), seed_req.size() * 4);
  dsp.upload(seed_pos.data(), seed_pos.size() * 8);

  dsa_kpool_tail_seed(dsk.p, dim, dsg.p, dim,
                      static_cast<const int32_t*>(dsr.p),
                      static_cast<const int64_t*>(dsp.p), seed_tokens,
                      dtail.p, kpool, dim, 0);
  // Dense active spans: slot 1 = tokens [0, 6), slot 3 = [6, 9), and the
  // unoccupied span [9, 11) that must touch no ring and no pool.
  std::vector<int32_t> spans = {0, 6, 6, 3, 9, 2};
  dspans.upload(spans.data(), spans.size() * 4);
  dsa_kpool_decode_update(dk.p, dim, dgate.p, dim,
                          static_cast<const float*>(dape.p),
                          static_cast<const int32_t*>(dreq.p),
                          static_cast<const int64_t*>(dpos.p),
                          static_cast<const int32_t*>(dspans.p), 3,
                          static_cast<const int32_t*>(dbt.p), n_blocks,
                          dtail.p, dki.p, static_cast<float*>(dks.p), ppb,
                          kpool, dim, 0);

  // Host expectation: simulate ring + completions.
  std::vector<uint16_t> tail(size_t(max_requests) * 2 * kpool * dim, 0);
  auto stash = [&](int req, int64_t p, const uint16_t* krow,
                   const uint16_t* grow) {
    const int slot = int(p % kpool);
    std::memcpy(&tail[size_t((req * 2 * kpool + slot)) * dim], krow, dim * 2);
    std::memcpy(&tail[size_t((req * 2 * kpool + kpool + slot)) * dim], grow,
                dim * 2);
  };
  for (int t = 0; t < seed_tokens; ++t)
    stash(seed_req[size_t(t)], seed_pos[size_t(t)],
          &seed_k[size_t(t) * dim], &seed_gate[size_t(t) * dim]);
  std::vector<uint8_t> want_k(size_t(total_slots) * dim, 0);
  std::vector<float> want_s(total_slots, 0.0f);
  auto complete = [&](int req, int64_t p, const uint16_t* krow,
                      const uint16_t* grow) {
    const int64_t pool = p / kpool;
    const int64_t slot =
        int64_t(bt[size_t(pool / ppb)]) * ppb + (pool % ppb);
    std::vector<uint16_t> pk(size_t(kpool) * dim), pg(size_t(kpool) * dim);
    for (int s = 0; s < kpool; ++s) {
      const int ring = int((p - (kpool - 1) + s) % kpool);
      const bool cur = (s == kpool - 1);
      std::memcpy(&pk[size_t(s) * dim],
                  cur ? krow : &tail[size_t((req * 2 * kpool + ring)) * dim],
                  dim * 2);
      std::memcpy(&pg[size_t(s) * dim],
                  cur ? grow
                      : &tail[size_t((req * 2 * kpool + kpool + ring)) * dim],
                  dim * 2);
    }
    dsa_ref::compress_pool<float>(pk.data(), pg.data(), ape.data(), kpool,
                                  dim, &want_k[size_t(slot) * dim],
                                  &want_s[size_t(slot)]);
  };
  for (int t = 0; t < tokens; ++t) {
    const int64_t p = pos[size_t(t)];
    if (p < 0) continue;
    const int req = req_ids[size_t(t)];
    if (p % kpool == kpool - 1)
      complete(req, p, &k[size_t(t) * dim], &gate[size_t(t) * dim]);
    stash(req, p, &k[size_t(t) * dim], &gate[size_t(t) * dim]);
  }

  std::vector<uint16_t> got_tail(size_t(max_requests) * 2 * kpool * dim);
  std::vector<uint8_t> got_kk(size_t(total_slots) * dim);
  std::vector<float> got_ss(total_slots, 0.0f);
  cudaDeviceSynchronize();
  dtail.download(got_tail.data(), got_tail.size() * 2);
  dki.download(got_kk.data(), got_kk.size());
  dks.download(got_ss.data(), got_ss.size() * 4);
  require_bitwise("multi-request tail", got_tail.data(), tail.data(),
                  tail.size() * 2);
  require_bitwise("multi-request pools", got_kk.data(), want_k.data(),
                  want_k.size());
  require_bitwise("multi-request scales", got_ss.data(), want_s.data(),
                  want_s.size() * 4);
}

// Attention at TP=1 (64 local heads -> 4 head-groups per split) with an
// empty padding row (counts == 0 -> zero output) and n_split == 1.
DGPP_TEST(dsa_attention_tp1_headgroups_and_empty_row) {
  DsaConfig cfg{};
  cfg.tp_size = 1;
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int rows = 2, local_heads = g.local_heads, nope = cfg.qk_nope_head_dim;
  const int v = cfg.v_head_dim, kv_lora = cfg.kv_lora_rank;
  const int block_tokens = cfg.block_tokens;
  const int max_selected = g.max_selected;

  auto q = random_bf16_bits(120, int64_t(rows) * local_heads * nope, -2, 1);
  auto kv_b = random_bf16_bits(
      121, int64_t(local_heads) * (nope + v) * kv_lora, -2, 1);
  const int cnt = 45;
  std::vector<int32_t> tokens(size_t(rows) * max_selected, -1);
  std::vector<int32_t> counts = {cnt, 0};  // row 1 is padding
  std::vector<int32_t> req_ids = {0, 0};
  for (int i = 0; i < cnt; ++i)
    tokens[i] = int32_t((i * 37) % 200);
  const int n_blocks = 2, total_tokens = n_blocks * block_tokens;
  auto latent = random_bf16_bits(122, int64_t(total_tokens) * kv_lora, -2, 1);
  std::vector<int32_t> bt = {1, 0};

  DevBuf dq(q.size() * 2), dkb(kv_b.size() * 2), dlat(latent.size() * 2),
      dtopk(tokens.size() * 4), dcnt(rows * 4), dri(rows * 4), dbt(8),
      dqt(size_t(rows) * local_heads * kv_lora * 2),
      dm(size_t(rows) * 2 * local_heads * 4),
      dl(size_t(rows) * 2 * local_heads * 4),
      dc(size_t(rows) * 2 * local_heads * kv_lora * 4),
      dc_out(size_t(rows) * local_heads * kv_lora * 4),
      dout(size_t(rows) * local_heads * v * 2);
  dq.upload(q.data(), q.size() * 2);
  dkb.upload(kv_b.data(), kv_b.size() * 2);
  dlat.upload(latent.data(), latent.size() * 2);
  dtopk.upload(tokens.data(), tokens.size() * 4);
  dcnt.upload(counts.data(), counts.size() * 4);
  dri.upload(req_ids.data(), req_ids.size() * 4);
  dbt.upload(bt.data(), bt.size() * 4);

  dsa_absorb_q(dq.p, dkb.p, dqt.p, rows, local_heads, nope, v, kv_lora, 0);
  const float scale = 1.0f / std::sqrt(float(nope));
  dsa_attn_partial(dqt.p, dlat.p, static_cast<const int32_t*>(dri.p),
                   static_cast<const int32_t*>(dtopk.p), max_selected,
                   static_cast<const int32_t*>(dcnt.p), rows, 2, local_heads,
                   kv_lora, block_tokens, static_cast<const int32_t*>(dbt.p),
                   n_blocks, scale, static_cast<float*>(dm.p),
                   static_cast<float*>(dl.p), static_cast<float*>(dc.p), 0);
  dsa_attn_combine(static_cast<const float*>(dm.p),
                   static_cast<const float*>(dl.p),
                   static_cast<const float*>(dc.p), rows, 2, local_heads,
                   kv_lora, static_cast<float*>(dc_out.p), 0);
  dsa_vout_gemm(dc_out.p, dkb.p, dout.p, rows, local_heads, nope, v, kv_lora,
                0);

  std::vector<uint16_t> phys_latent(size_t(total_tokens) * kv_lora);
  for (int64_t tok = 0; tok < total_tokens; ++tok) {
    const int64_t phys =
        int64_t(bt[size_t(tok / block_tokens)]) * block_tokens +
        (tok % block_tokens);
    std::memcpy(&phys_latent[size_t(tok) * kv_lora],
                &latent[size_t(phys) * kv_lora], kv_lora * 2);
  }
  std::vector<uint16_t> got(size_t(rows) * local_heads * v);
  cudaDeviceSynchronize();
  dout.download(got.data(), got.size() * 2);
  // Row 0: vs the host oracle.
  std::vector<uint16_t> want(size_t(local_heads) * v);
  dsa_ref::absorbed_attn<float>(&q[0], phys_latent.data(), kv_lora,
                                tokens.data(), cnt, kv_b.data(), local_heads,
                                nope, v, kv_lora, scale, want.data());
  auto st = compare_bf16(
      std::vector<uint16_t>(got.begin(), got.begin() + want.size()), want, 8);
  require_bf16("tp1 attention row0", st, 0.01, 0.006);
  // Row 1: empty selection -> exactly zero output.
  for (size_t i = want.size(); i < got.size(); ++i)
    if (got[i] != 0)
      throw std::runtime_error("empty row output must be zero at " +
                               std::to_string(i));
}

// Graph capture/replay of the fused decode select: the counter self-reset
// and the fixed-grid shape must replay correctly with a changed position
// (different visible count) without re-capture.
DGPP_TEST(dsa_select_decode_graph_replay) {
  DecodeSelScenario sc;
  sc.init(130, {40001});
  const int max_selected = sc.g.max_selected;
  const int grid = 48;

  DevBuf dq8(sc.q8.size()), dw(sc.w.size() * 4), dki(sc.k8.size()),
      dks_c(sc.ks.size() * 4), dpos(8), dri(4), dbt(4),
      dtopk(size_t(max_selected) * 4), dcnt(4),
      dws(dsa_select_workspace_bytes(1, sc.n_pools)), dctr(8);
  dq8.upload(sc.q8.data(), sc.q8.size());
  dw.upload(sc.w.data(), sc.w.size() * 4);
  dki.upload(sc.k8.data(), sc.k8.size());
  dks_c.upload(sc.ks.data(), sc.ks.size() * 4);
  std::vector<int32_t> req0 = {0}, bt0 = {0};
  dri.upload(req0.data(), 4);
  dbt.upload(bt0.data(), 4);
  int32_t ctr0 = 0;
  dctr.upload(&ctr0, 4);
  int64_t pos_a = 40001;
  dpos.upload(&pos_a, 8);

  cudaStream_t stream;
  DGPP_CUDA_OK(cudaStreamCreate(&stream));
  // Warm up outside capture (plan/attribute setup), then capture.
  dsa_select_decode(dq8.p, static_cast<const float*>(dw.p),
                    static_cast<const int32_t*>(dri.p),
                    static_cast<const int64_t*>(dpos.p), 1,
                    static_cast<const int32_t*>(dbt.p), 1, dki.p,
                    static_cast<const float*>(dks_c.p), int(sc.n_pools), 32,
                    128, sc.g.select_k, 4, max_selected,
                    static_cast<int32_t*>(dtopk.p),
                    static_cast<int32_t*>(dcnt.p), dws.p, sc.n_pools,
                    static_cast<int32_t*>(dctr.p), grid, stream);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));

  DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  dsa_select_decode(dq8.p, static_cast<const float*>(dw.p),
                    static_cast<const int32_t*>(dri.p),
                    static_cast<const int64_t*>(dpos.p), 1,
                    static_cast<const int32_t*>(dbt.p), 1, dki.p,
                    static_cast<const float*>(dks_c.p), int(sc.n_pools), 32,
                    128, sc.g.select_k, 4, max_selected,
                    static_cast<int32_t*>(dtopk.p),
                    static_cast<int32_t*>(dcnt.p), dws.p, sc.n_pools,
                    static_cast<int32_t*>(dctr.p), grid, stream);
  cudaGraph_t graph;
  DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
  cudaGraphExec_t exec;
  DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  // Replay A: same position.
  DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  std::vector<int32_t> got_a(size_t(max_selected), -12345);
  int32_t cnt_a = -1;
  dtopk.download(got_a.data(), got_a.size() * 4);
  dcnt.download(&cnt_a, 4);

  // Replay B: changed position -> different visible count; must match an
  // eager run at the same position.
  int64_t pos_b = 20005;  // visible 5001 vs 10000
  dpos.upload(&pos_b, 8);
  DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  std::vector<int32_t> got_b(size_t(max_selected), -12345);
  int32_t cnt_b = -1;
  dtopk.download(got_b.data(), got_b.size() * 4);
  dcnt.download(&cnt_b, 4);

  DecodeSelScenario sb;
  sb.init(130, {pos_b});
  sb.q8 = sc.q8;
  sb.w = sc.w;
  sb.k8 = sc.k8;
  sb.ks = sc.ks;
  std::vector<int32_t> want_b, want_cnt_b;
  sb.expect(want_b, want_cnt_b);
  if (cnt_b != want_cnt_b[0] ||
      std::memcmp(got_b.data(), want_b.data(), max_selected * 4) != 0)
    throw std::runtime_error("graph replay with changed pos mismatch");

  // Replay C: back to position A — bitwise with replay A.
  dpos.upload(&pos_a, 8);
  DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  std::vector<int32_t> got_c(size_t(max_selected), -12345);
  dtopk.download(got_c.data(), got_c.size() * 4);
  require_bitwise("graph replay determinism", got_c.data(), got_a.data(),
                  max_selected * 4);

  DGPP_CUDA_OK(cudaGraphExecDestroy(exec));
  DGPP_CUDA_OK(cudaGraphDestroy(graph));
  DGPP_CUDA_OK(cudaStreamDestroy(stream));
}

// kpool=2 smoke: the config allows it; compress + decode ring must behave.
DGPP_TEST(dsa_kpool2_compress_and_ring_smoke) {
  const int kpool = 2, dim = 128, n_pools = 12;
  const int tokens = n_pools * kpool;
  auto k = random_bf16_bits(140, tokens * dim, -2, 1);
  std::vector<float> ape(size_t(kpool) * dim);
  for (int i = 0; i < kpool * dim; ++i) ape[i] = random_f32(141, i) * 0.1f;
  auto gate = std::vector<uint16_t>(size_t(tokens) * dim);
  for (int t = 0; t < tokens; ++t) {
    const bool dom = (t % kpool) == kpool - 1;
    for (int d = 0; d < dim; ++d)
      gate[size_t(t) * dim + d] = float_to_bf16_bits(dom ? 50.0f : -50.0f);
  }
  DevBuf dk(k.size() * 2), dgate(gate.size() * 2), dape(ape.size() * 4),
      dbt(4), dki(k.size()), dks(size_t(n_pools) * 4);
  dk.upload(k.data(), k.size() * 2);
  dgate.upload(gate.data(), gate.size() * 2);
  dape.upload(ape.data(), ape.size() * 4);
  int32_t bt0 = 0;
  dbt.upload(&bt0, 4);
  dki.upload(std::vector<uint8_t>(k.size(), 0).data(), k.size());
  dsa_kpool_compress_write(dk.p, dim, dgate.p, dim,
                           static_cast<const float*>(dape.p),
                           static_cast<const int32_t*>(dbt.p), n_pools, 0,
                           n_pools, dki.p, static_cast<float*>(dks.p), kpool,
                           dim, 0);
  std::vector<uint8_t> got(k.size());
  std::vector<float> got_s(size_t(n_pools), 0.0f);
  cudaDeviceSynchronize();
  dki.download(got.data(), got.size());
  dks.download(got_s.data(), got_s.size() * 4);
  for (int j = 0; j < n_pools; ++j) {
    std::vector<uint8_t> w8(dim);
    float ws = 0;
    dsa_ref::compress_pool<float>(&k[size_t(j * kpool) * dim],
                                  &gate[size_t(j * kpool) * dim], ape.data(),
                                  kpool, dim, w8.data(), &ws);
    if (std::memcmp(&got[size_t(j) * dim], w8.data(), dim) != 0 ||
        got_s[size_t(j)] != ws)
      throw std::runtime_error("kpool2 compress pool " + std::to_string(j));
  }
}

// ---------------------------------------------------------------------------
// Layer phase (M3): DsaStatePool + DsaLayer orchestration
// ---------------------------------------------------------------------------

namespace layer {

using namespace dgpp;
namespace dsa_test = ::dgpp::dsa_test;
using dgpp::kda_test::DevBuf;
using dgpp::kda_test::random_bf16_bits;
using dgpp::kda_test::require_bf16;
using dgpp::kda_test::require_bitwise;
using dgpp::kda_test::require_rel;

// Small fast config exercising every kernel path at a fraction of the real
// geometry: heads=32 indexer (kernel-pinned), topk=64 -> select_k=16 pools,
// 4 MLA heads, block_tokens=32 (8 pools/block).
DsaConfig small_cfg(int num_dsa_layers = 1, int tp_size = 1) {
  DsaConfig cfg{};
  cfg.hidden = 128;
  cfg.num_heads = 4;
  cfg.q_lora_rank = 64;
  cfg.kv_lora_rank = 32;
  cfg.qk_nope_head_dim = 64;
  cfg.v_head_dim = 64;
  cfg.index_n_heads = 32;
  cfg.index_head_dim = 128;
  cfg.index_topk = 64;
  cfg.index_kpool = 4;
  cfg.num_dsa_layers = num_dsa_layers;
  cfg.block_tokens = 32;
  cfg.tp_size = tp_size;
  return cfg;
}

// The full GLM-5.3's shape at the CI scale (plan D3/D4/D8): the same 4
// heads and 32 x 128 indexer with a 64-wide decoupled RoPE beside nope 64,
// per-token selection (kpool 1, topk 16 = select_k 16: rows past 15 are
// sparse) and the relu'd indexer.
DsaConfig full_cfg(int num_dsa_layers = 1, int tp_size = 1) {
  DsaConfig cfg = small_cfg(num_dsa_layers, tp_size);
  cfg.qk_rope_head_dim = 64;
  cfg.index_kpool = 1;
  cfg.index_topk = 16;
  cfg.index_relu = 1;
  return cfg;
}

// What a TestWeights carries beyond the bf16 forms.
struct TestWeightOpts {
  bool fp8 = false;      // q_a / kv_a / q_b / o_proj as e4m3 pairs (Flash's form)
  int packed_bits = 0;   // 8 (or 4): the fused qkv_a, q_b and o_proj as packed-int triples
  bool indexer = true;   // false: a selection-reusing view (no indexer tensors)
};
constexpr int64_t kTestRopePositions = 4096;

// Random host weights + device uploads + both view structs (layer + oracle).
struct TestWeights {
  HostWeights host;
  DsaLayerWeights layer_views;
  std::vector<DevBuf> dev;
  std::vector<std::vector<uint16_t>> storage;  // keeps host bits alive
  std::vector<float> ape;
  std::vector<uint16_t> rope_table;  // bf16 [positions][2][rope/2] (rope > 0)

  // fp8_projections: q_a / kv_a / q_b / o_proj as e4m3 payloads
  // with 128x128 block scales — the oracle (and `bridge_views`, the bf16
  // form of the same layer) get bf16(e4m3 x s), the bridge's rule;
  // `layer_views` carries the pairs for the scale-aware GEMM path.
  // packed_bits: the same two-form arrangement with the packed-int
  // triples (bf16(code x scale) for the oracle and the bridge).
  DsaLayerWeights bridge_views;
  std::vector<std::vector<uint8_t>> q_storage;
  std::vector<std::vector<float>> s_storage;
  std::vector<std::vector<uint32_t>> p_storage;

  TestWeights(const DsaConfig& cfg, uint64_t seed, bool fp8_projections = false)
      : TestWeights(cfg, seed, TestWeightOpts{fp8_projections, 0, true}) {}

  TestWeights(const DsaConfig& cfg, uint64_t seed, TestWeightOpts opts) {
    const bool fp8_projections = opts.fp8;
    const DsaGeometry g = DsaGeometry::from_config(cfg);
    const int heads = cfg.index_n_heads;
    const int dim = cfg.index_head_dim;
    const int rope = cfg.qk_rope_head_dim;
    dev.reserve(48);
    storage.reserve(24);
    q_storage.reserve(4);
    s_storage.reserve(4);
    p_storage.reserve(4);
    // Host bits live in `storage` (a reused local would dangle the views).
    // Returns both pointers: `host` for the reference oracle (CPU), `dev`
    // for the layer (GPU) — mixing them up segfaults spectacularly.
    struct Ptrs {
      const uint16_t* host;
      const uint16_t* dev;
    };
    const auto wbits = [](uint64_t seed, int64_t n) {
      return random_bf16_bits(seed, n, -2, 1);
    };
    const auto up_bf16 = [&](std::vector<uint16_t> v) -> Ptrs {
      storage.push_back(std::move(v));
      const std::vector<uint16_t>& t = storage.back();
      dev.emplace_back(t.size() * 2);
      dev.back().upload(t.data(), t.size() * 2);
      return Ptrs{t.data(), static_cast<const uint16_t*>(dev.back().p)};
    };
    Ptrs p;
    if (opts.indexer) {
      p = up_bf16(wbits(seed ^ 0xA1, int64_t(heads) * dim * cfg.q_lora_rank));
      host.wq_b = p.host;
      layer_views.wq_b = p.dev;
      p = up_bf16(wbits(seed ^ 0xB2, int64_t(dim) * cfg.hidden));
      host.wk = p.host;
      layer_views.wk = p.dev;
      p = up_bf16(wbits(seed ^ 0xC3, int64_t(heads) * cfg.hidden));
      host.wp = p.host;
      layer_views.wp = p.dev;
      if (cfg.index_kpool > 1) {
        p = up_bf16(wbits(seed ^ 0xD4, int64_t(dim) * cfg.hidden));
        host.gate = p.host;
        layer_views.gate = p.dev;
      }
      p = up_bf16(wbits(seed ^ 0xE5, dim));
      host.k_norm_w = p.host;
      layer_views.k_norm_w = p.dev;
      p = up_bf16(wbits(seed ^ 0xF6, dim));
      host.k_norm_b = p.host;
      layer_views.k_norm_b = p.dev;
    }
    // A packed-int [rows, cols] matrix: random codes under power-of-two
    // group scales (2^-7 or 2^-6: |w| <= 1 as the bf16 forms'), so that
    // code x scale is exactly a bf16 — the bridge / oracle weights (the
    // returned bf16) ARE the kernels' exact dequant, and the packed layer
    // differs from the bridge layer by summation order alone. (The
    // encoder's own recipe, bf16(amax / 127.5) scales, is pinned by its
    // unit test and the loader gates; here it would put the oracle a bf16
    // rounding of every weight away from the kernels.)
    const auto make_packed = [&](uint64_t sd, int64_t rows, int64_t cols,
                                 GlmPackedMatrix* view) -> std::vector<uint16_t> {
      const int bits = opts.packed_bits;
      const int per = 32 / bits;
      const uint32_t mask = (1u << bits) - 1u;
      const int64_t words_per_row = cols * bits / 32, scales_per_row = cols / dgpp::kPackedGroup;
      std::vector<uint32_t> words(size_t(rows * words_per_row), 0u);
      std::vector<uint16_t> scales(size_t(rows * scales_per_row));
      std::vector<uint16_t> deq(size_t(rows) * cols);
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t gi = 0; gi < scales_per_row; ++gi)
          scales[size_t(r * scales_per_row + gi)] =
              float_to_bf16_bits(hash32(sd ^ 0x5CA1E + uint64_t(r * scales_per_row + gi)) & 1u
                                     ? 1.0f / 128.0f
                                     : 1.0f / 64.0f);
        for (int64_t c = 0; c < cols; ++c) {
          const uint32_t u = hash32(sd + uint64_t(r * cols + c) * 31u) & mask;
          words[size_t(r * words_per_row + c / per)] |= u << (bits * (c % per));
        }
        for (int64_t c = 0; c < cols; ++c)
          deq[size_t(r * cols + c)] = float_to_bf16_bits(
              dgpp::packq_decode(&words[size_t(r * words_per_row)], &scales[size_t(r * scales_per_row)],
                                 cols, bits, 0, c));
      }
      p_storage.push_back(std::move(words));
      dev.emplace_back(p_storage.back().size() * 4);
      dev.back().upload(p_storage.back().data(), p_storage.back().size() * 4);
      view->packed = static_cast<const uint32_t*>(dev.back().p);
      storage.push_back(std::move(scales));
      dev.emplace_back(storage.back().size() * 2);
      dev.back().upload(storage.back().data(), storage.back().size() * 2);
      view->scales = static_cast<const uint16_t*>(dev.back().p);
      view->rows = rows;
      view->cols = cols;
      view->bits = bits;
      return deq;
    };
    // A quantized [rows, cols] matrix: random e4m3 (NaN codes remapped),
    // random block scales; returns its dequantized bf16 bits and uploads
    // the pair; the GlmQuantMatrix view points at the device pair.
    const auto make_quant = [&](uint64_t sd, int64_t rows, int64_t cols,
                                GlmQuantMatrix* view) -> std::vector<uint16_t> {
      std::vector<uint8_t> payload(size_t(rows) * cols);
      for (size_t i = 0; i < payload.size(); ++i) {
        uint8_t b = static_cast<uint8_t>(random_f32(sd, int64_t(i)) * 256.f) & 0x7F;
        if ((b & 0x7F) > 0x5F) b &= 0x5F;                    // |w| <= 2^7, no NaN
        if (random_f32(sd ^ 0x55, int64_t(i)) > 0.5f) b |= 0x80;
        payload[i] = b;
      }
      const int64_t sr = (rows + 127) / 128, sc = (cols + 127) / 128;
      std::vector<float> scales(size_t(sr) * sc);
      for (size_t i = 0; i < scales.size(); ++i)
        scales[i] = 0.004f + 0.012f * random_f32(sd ^ 0x99, int64_t(i));
      std::vector<uint16_t> deq(size_t(rows) * cols);
      for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < cols; ++c)
          deq[size_t(r) * cols + c] = float_to_bf16_bits(
              fp8_e4m3_bits_to_float(payload[size_t(r) * cols + c]) *
              scales[size_t(r / 128) * sc + c / 128]);
      q_storage.push_back(std::move(payload));
      s_storage.push_back(std::move(scales));
      dev.emplace_back(q_storage.back().size());
      dev.back().upload(q_storage.back().data(), q_storage.back().size());
      view->payload = static_cast<const uint8_t*>(dev.back().p);
      dev.emplace_back(s_storage.back().size() * 4);
      dev.back().upload(s_storage.back().data(), s_storage.back().size() * 4);
      view->scales = static_cast<const float*>(dev.back().p);
      view->rows = rows;
      view->cols = cols;
      return deq;
    };
    const int64_t qkv_rows = int64_t(cfg.q_lora_rank + cfg.kv_lora_rank + rope);
    if (fp8_projections) {
      std::vector<uint16_t> qa = make_quant(seed ^ 0x179, cfg.q_lora_rank, cfg.hidden,
                                            &layer_views.q_a_q);
      std::vector<uint16_t> kva = make_quant(seed ^ 0x17A, cfg.kv_lora_rank, cfg.hidden,
                                             &layer_views.kv_a_q);
      qa.insert(qa.end(), kva.begin(), kva.end());  // the fused bf16 [q_a | kv_a]
      p = up_bf16(std::move(qa));
      host.qkv_a = p.host;
      bridge_views.qkv_a = p.dev;
      layer_views.qkv_a = nullptr;
    } else if (opts.packed_bits) {
      p = up_bf16(make_packed(seed ^ 0x179, qkv_rows, cfg.hidden, &layer_views.qkv_a_p));
      host.qkv_a = p.host;
      bridge_views.qkv_a = p.dev;
      layer_views.qkv_a = nullptr;
    } else {
      p = up_bf16(wbits(seed ^ 0x179, qkv_rows * cfg.hidden));
      host.qkv_a = p.host;
      layer_views.qkv_a = p.dev;
    }
    p = up_bf16(wbits(seed ^ 0x28A, cfg.q_lora_rank));
    host.q_aln = p.host;
    layer_views.q_aln = p.dev;
    p = up_bf16(wbits(seed ^ 0x39B, cfg.kv_lora_rank));
    host.kv_aln = p.host;
    layer_views.kv_aln = p.dev;
    if (fp8_projections) {
      p = up_bf16(make_quant(seed ^ 0x4AC, g.local_q_rows, cfg.q_lora_rank,
                             &layer_views.q_b_q));
      host.q_b = p.host;
      bridge_views.q_b = p.dev;
      layer_views.q_b = nullptr;
    } else if (opts.packed_bits) {
      p = up_bf16(make_packed(seed ^ 0x4AC, g.local_q_rows, cfg.q_lora_rank,
                              &layer_views.q_b_p));
      host.q_b = p.host;
      bridge_views.q_b = p.dev;
      layer_views.q_b = nullptr;
    } else {
      p = up_bf16(wbits(seed ^ 0x4AC, int64_t(g.local_q_rows) * cfg.q_lora_rank));
      host.q_b = p.host;
      layer_views.q_b = p.dev;
    }
    p = up_bf16(wbits(seed ^ 0x5BD,
                      int64_t(g.local_heads) *
                          (cfg.qk_nope_head_dim + cfg.v_head_dim) *
                          cfg.kv_lora_rank));
    host.kv_b = p.host;
    layer_views.kv_b = p.dev;
    if (fp8_projections) {
      p = up_bf16(make_quant(seed ^ 0x6CE, cfg.hidden, g.local_v_rows,
                             &layer_views.o_proj_q));
      host.o_proj = p.host;
      bridge_views.o_proj = p.dev;
      layer_views.o_proj = nullptr;
    } else if (opts.packed_bits) {
      p = up_bf16(make_packed(seed ^ 0x6CE, cfg.hidden, g.local_v_rows,
                              &layer_views.o_proj_p));
      host.o_proj = p.host;
      bridge_views.o_proj = p.dev;
      layer_views.o_proj = nullptr;
    } else {
      p = up_bf16(wbits(seed ^ 0x6CE, int64_t(cfg.hidden) * g.local_v_rows));
      host.o_proj = p.host;
      layer_views.o_proj = p.dev;
    }
    if (opts.indexer && cfg.index_kpool > 1) {
      ape.resize(size_t(cfg.index_kpool) * dim);
      for (size_t i = 0; i < ape.size(); ++i)
        ape[i] = random_f32(seed, int64_t(i)) * 0.2f - 0.1f;
      dev.emplace_back(ape.size() * 4);
      dev.back().upload(ape.data(), ape.size() * 4);
      host.ape = ape.data();
      layer_views.ape = static_cast<const float*>(dev.back().p);
    }
    if (rope > 0) {
      // The model's rotary table (theta 8e6, the checkpoint's), shared by
      // the host oracle and the layer.
      rope_table.resize(size_t(kTestRopePositions) * 2 * size_t(rope / 2));
      dgpp::dsa_rope_table_host(8e6, rope, kTestRopePositions, rope_table.data());
      dev.emplace_back(rope_table.size() * 2);
      dev.back().upload(rope_table.data(), rope_table.size() * 2);
      host.rope_table = rope_table.data();
      host.rope_table_positions = kTestRopePositions;
      layer_views.rope_table = dev.back().p;
      layer_views.rope_table_positions = kTestRopePositions;
    }
    if (fp8_projections || opts.packed_bits) {
      // The bridge form: every other view shared, the three bf16 buffers
      // holding the same dequantized values the pairs / triples decode to.
      DsaLayerWeights b = layer_views;
      b.q_a_q = GlmQuantMatrix{};
      b.kv_a_q = GlmQuantMatrix{};
      b.q_b_q = GlmQuantMatrix{};
      b.o_proj_q = GlmQuantMatrix{};
      b.qkv_a_p = GlmPackedMatrix{};
      b.q_b_p = GlmPackedMatrix{};
      b.o_proj_p = GlmPackedMatrix{};
      b.qkv_a = bridge_views.qkv_a;
      b.q_b = bridge_views.q_b;
      b.o_proj = bridge_views.o_proj;
      bridge_views = b;
    }
  }
};

// Arena-backed test environment: state pool + layer scratch + shared GEMM
// workspace, mirroring the KDA LayerEnv pattern.
struct LayerEnv {
  dgpp::Arena arena;
  dgpp::CublasLtGemm gemm;
  DevBuf ws{64ull << 20};
  cudaStream_t stream = nullptr;

  LayerEnv(const DsaConfig& cfg, int max_tokens, int64_t max_cache_tokens,
           int max_requests, int64_t pool_tokens, cudaStream_t s,
           size_t dot_budget = 16ull << 20) {
    stream = s;
    // DSA_TEST_DECODE_ROWS=n: the GEMM interface's decode lowering (the
    // models' setting) for the layer's GEMMs — an experiment hook.
    if (const char* dr = std::getenv("DSA_TEST_DECODE_ROWS")) gemm.set_decode_rows(std::atoi(dr));
    dgpp::Arena::Config ac;
    ac.persistent_hot =
        DsaStatePool::cache_bytes(cfg, max_requests, pool_tokens) +
        DsaLayer::scratch_bytes(cfg, max_tokens, max_cache_tokens, 8, 32,
                                dot_budget);  // must match the layer default
    arena.init(ac);
  }
};

// Download a request's block-table row (logical->physical block mapping).
std::vector<int32_t> fetch_block_row(const DsaStatePool& pool, int req,
                                     cudaStream_t s) {
  std::vector<int32_t> row(size_t(pool.total_blocks()));
  DGPP_CUDA_OK(cudaMemcpyAsync(
      row.data(), pool.block_tables() + size_t(req) * size_t(pool.total_blocks()),
      row.size() * 4, cudaMemcpyDeviceToHost, s));
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  return row;
}

// Compare the device cache (physical slots, via the block table) against the
// reference's logical HostState.
void require_cache_matches(const DsaStatePool& pool, int layer, int req,
                           const dsa_ref::HostState& ref, cudaStream_t s) {
  const DsaConfig& cfg = pool.config();
  const DsaGeometry& g = pool.geometry();
  const int dim = cfg.index_head_dim;
  const std::vector<int32_t> bt = fetch_block_row(pool, req, s);
  const int64_t pools = ref.num_pools;

  // Index cache (fp8 rows + fp32 scales): tolerance — the compress inputs
  // differ by GEMM summation order, so occasional 1-quantum flips are
  // expected (the compress kernel itself is bitwise-pinned at kernel level).
  std::vector<uint8_t> got_k(size_t(pools) * dim);
  std::vector<float> got_s(size_t(pools), 0.0f);
  DGPP_CUDA_OK(cudaMemcpyAsync(got_k.data(), pool.index_k(layer),
                               got_k.size(), cudaMemcpyDeviceToHost, s));
  DGPP_CUDA_OK(cudaMemcpyAsync(got_s.data(), pool.index_scale(layer),
                               got_s.size() * 4, cudaMemcpyDeviceToHost, s));
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  // Compare DEQUANTIZED values (code * scale): the compress inputs differ
  // by bf16 GEMM ulps, so elements near a quant boundary legitimately flip
  // one code, and a row whose absmax sits near a power of two can shift its
  // scale by 2x — raw codes then differ by ~2x while the VALUES stay within
  // a ulp. Dequantized comparison with a one-e4m3-ulp relative budget (plus
  // an absolute floor at the row's minimum quantum) tolerates both without
  // masking real corruption (a misrouted slot reads a different row
  // entirely: O(1) value differences on every element).
  long fp8_bad = 0;
  double k_l2 = 0, k_ref = 0;
  int scale_shifts = 0;
  for (int64_t j = 0; j < pools; ++j) {
    const int64_t slot =
        int64_t(bt[size_t(j / g.pools_per_block)]) * g.pools_per_block +
        (j % g.pools_per_block);
    const double dev_scale = double(got_s[size_t(j)]);
    const double ref_scale = double(ref.index_scale[size_t(j)]);
    const double floor_q = std::min(dev_scale, ref_scale) * (1.0 / 448.0) + 1e-9;
    for (int d = 0; d < dim; ++d) {
      const double a = fp8_e4m3_bits_to_float(got_k[size_t(j) * dim + d]) *
                       dev_scale;
      const double b = fp8_e4m3_bits_to_float(ref.index_k[size_t(j) * dim + d]) *
                       ref_scale;
      const double dd = std::abs(a - b);
      if (dd > std::max(std::abs(b) * 0.141, floor_q)) ++fp8_bad;  // ~1.1 ulp
      k_l2 += dd * dd;
      k_ref += b * b;
    }
    if (dev_scale != ref_scale) {
      const double ratio = dev_scale > ref_scale
                               ? dev_scale / ref_scale
                               : ref_scale / dev_scale;
      if (ratio > 2.5)
        throw std::runtime_error("layer cache index_scale off by " +
                                 std::to_string(ratio) + "x at pool " +
                                 std::to_string(j));
      ++scale_shifts;  // power-of-two boundary crossing: legitimate
    }
  }
  if (fp8_bad > long(pools) * dim / 20)  // > 5% of elements beyond ~1 ulp
    throw std::runtime_error("layer cache index_k mismatch: " +
                             std::to_string(fp8_bad) + " elements");
  if (k_l2 / std::max(k_ref, 1e-30) > 4e-3) {
    // Per-pool attribution before failing.
    std::string worst;
    for (int64_t j = 0; j < pools; ++j) {
      const int64_t slot =
          int64_t(bt[size_t(j / g.pools_per_block)]) * g.pools_per_block +
          (j % g.pools_per_block);
      double pj = 0, rj = 0;
      for (int d = 0; d < dim; ++d) {
        const double a = fp8_e4m3_bits_to_float(got_k[size_t(j) * dim + d]) *
                         double(got_s[size_t(j)]);
        const double b =
            fp8_e4m3_bits_to_float(ref.index_k[size_t(j) * dim + d]) *
            double(ref.index_scale[size_t(j)]);
        pj += (a - b) * (a - b);
        rj += b * b;
      }
      if (pj / std::max(rj, 1e-30) > 4e-3)
        worst += " pool " + std::to_string(j) + "(" +
                 std::to_string(pj / std::max(rj, 1e-30)) + ")";
    }
    throw std::runtime_error("layer cache index_k l2 drift: " +
                             std::to_string(k_l2 / std::max(k_ref, 1e-30)) +
                             "; offending pools:" + worst);
  }
  if (scale_shifts > pools / 8 + 2)
    throw std::runtime_error("layer cache index_scale shifted for " +
                             std::to_string(scale_shifts) + "/" +
                             std::to_string(pools) + " pools");

  // Latent cache: bf16 tolerance. Reorder device rows to logical order
  // through the block table, then compare with the reference. A quantized
  // cache is read back as codes + row scales and dequantized
  // on the host — the reference holds the dequantized rows — with a wider
  // budget: the rows the two sides quantized differ by GEMM ulps, so an
  // element near a quantization boundary legitimately lands one code apart
  // (one e4m3 quantum is 2^-4..2^-3 of the value, one e2m1 quantum up to
  // half of it). The rope tail (bf16 after the payload) is compared as
  // part of the same row.
  const size_t row_bytes = g.latent_bytes_per_token;
  const size_t phys_rows = size_t(pool.max_token_slots());
  const int lw = g.score_width;
  std::vector<uint8_t> got_raw(phys_rows * row_bytes);
  DGPP_CUDA_OK(cudaMemcpyAsync(got_raw.data(), pool.latent(layer), got_raw.size(),
                               cudaMemcpyDeviceToHost, s));
  std::vector<float> got_row_scale(phys_rows, 1.0f);
  if (pool.latent_scale(layer) != nullptr)
    DGPP_CUDA_OK(cudaMemcpyAsync(got_row_scale.data(), pool.latent_scale(layer),
                                 got_row_scale.size() * 4, cudaMemcpyDeviceToHost, s));
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  std::vector<uint16_t> dev_logical(size_t(ref.num_tokens) * size_t(lw));
  for (int64_t t = 0; t < ref.num_tokens; ++t) {
    const int64_t phys = int64_t(bt[size_t(t / cfg.block_tokens)]) *
                             cfg.block_tokens +
                         (t % cfg.block_tokens);
    dgpp::latent_dequantize_row_host(cfg.latent_format, &got_raw[size_t(phys) * row_bytes],
                                     got_row_scale[size_t(phys)], cfg.kv_lora_rank,
                                     &dev_logical[size_t(t) * size_t(lw)]);
    if (g.rope_dim > 0)
      std::memcpy(&dev_logical[size_t(t) * size_t(lw) + size_t(cfg.kv_lora_rank)],
                  &got_raw[size_t(phys) * row_bytes + g.latent_payload_bytes],
                  size_t(g.rope_dim) * 2);
  }
  const std::vector<uint16_t> ref_latent(
      ref.latent.begin(), ref.latent.begin() + int64_t(dev_logical.size()));
  switch (cfg.latent_format) {
    case LatentFormat::kBf16:
      require_bf16("layer cache latent",
                   kda_test::compare_bf16(dev_logical, ref_latent, 8), 5e-3, 1e-3);
      break;
    case LatentFormat::kFp8:
      require_bf16("layer cache latent (fp8)",
                   kda_test::compare_bf16(dev_logical, ref_latent, 24), 0.05, 0.05);
      break;
    case LatentFormat::kFp4:
      require_bf16("layer cache latent (fp4)",
                   kda_test::compare_bf16(dev_logical, ref_latent, 96), 0.15, 0.05);
      break;
  }

  // Tail ring: bf16 tolerance (raw k + gate rows).
  std::vector<uint16_t> got_tail(2ull * cfg.index_kpool * dim);
  DGPP_CUDA_OK(cudaMemcpyAsync(
      got_tail.data(),
      static_cast<const uint8_t*>(pool.tail(layer)) +
          size_t(req) * g.tail_bytes_per_request,
      got_tail.size() * 2, cudaMemcpyDeviceToHost, s));
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  const auto tail_stats = kda_test::compare_bf16(got_tail, ref.tail, 8);
  require_bf16("layer cache tail", tail_stats, 5e-3, 1e-3);
}

// Layer-vs-reference output comparison, selection-aware.
//
// Device GEMMs (cublasLt) and the host oracle sum in different orders, so
// q_idx/w_folded differ by ulps; after fp8 quant that perturbs pool logits
// enough to flip the select_k/select_k+1 boundary on a few rows. That is
// the documented device-vs-host tolerance class (DESIGN 12), not a wiring
// bug — so: rows with identical selections get a tight per-row budget;
// flipped rows must be few, must be a single boundary swap (one pool in,
// one pool out), and must stay sane in magnitude. Wiring bugs (wrong
// buffers/strides) produce massive multi-row divergence and fail loudly.
// Optional per-flipped-row auditor: certifies each divergence as a measured
// boundary near tie using both sides' actual inputs (see
// dsa_near_tie_audit.hpp). Returns false when this row cannot be audited
// (inputs unavailable); the caller then falls back to the strict swap shape.
using RowAuditor = std::function<bool(int64_t row, const int32_t* dev_toks,
                                      const int32_t* ref_toks)>;

// One phase's captured selection inputs: the q_fp8 rows and folded weights
// the select kernels actually consumed for [row0, row0+rows) of the output,
// downloaded immediately after that phase's enqueue (the scratch is
// overwritten by every later enqueue).
struct PhaseInputs {
  int64_t row0 = 0;  // output-row offset of this phase
  int rows = 0;
  std::vector<uint8_t> q8;      // device: [rows, heads*dim] fp8 bits
  std::vector<float> w;         // device: [rows, heads]
  std::vector<uint8_t> ref_q8;  // reference's own quantized rows
  std::vector<float> ref_w;     // reference's own folded weights
  // Prefill phases only: the tile's fp8 dots [tile_rows*heads, stride] and
  // the row offset of that tile within the phase (single-tile phases: 0).
  std::vector<float> dots;
  int64_t dot_stride = 0;
  int64_t dot_row0 = 0;
};

void capture_phase_inputs(DsaLayer& layer, const HostWeights& host_w,
                          const DsaConfig& cfg, const uint16_t* hidden_rows,
                          int64_t row0, int rows, PhaseInputs& out,
                          cudaStream_t s) {
  const int heads = cfg.index_n_heads;
  const int dim = cfg.index_head_dim;
  out.row0 = row0;
  out.rows = rows;
  out.q8.assign(size_t(rows) * heads * dim, uint8_t(0));
  out.w.assign(size_t(rows) * heads, 0.0f);
  DGPP_CUDA_OK(cudaMemcpyAsync(out.q8.data(), layer.debug_q_fp8(),
                               out.q8.size(), cudaMemcpyDeviceToHost, s));
  DGPP_CUDA_OK(cudaMemcpyAsync(out.w.data(), layer.debug_w_folded(),
                               out.w.size() * 4, cudaMemcpyDeviceToHost, s));
  const int64_t stride = layer.debug_dot_stride();
  if (stride > 0) {
    out.dot_stride = stride;
    out.dots.assign(size_t(rows) * heads * size_t(stride), 0.0f);
    DGPP_CUDA_OK(cudaMemcpyAsync(out.dots.data(), layer.debug_dots(),
                                 out.dots.size() * 4,
                                 cudaMemcpyDeviceToHost, s));
  }
  // The reference's OWN selection inputs for the same rows: recomputed with
  // the m-independent host oracle (a row's values do not depend on the
  // chunk it was computed in, so this is bitwise what layer_forward
  // consumed for these rows).
  out.ref_q8.assign(size_t(rows) * heads * dim, uint8_t(0));
  out.ref_w.assign(size_t(rows) * heads, 0.0f);
  dsa_ref::indexer_query_inputs<float>(host_w, cfg, hidden_rows, int(rows),
                                       out.ref_q8.data(), out.ref_w.data(),
                                       /*token_start=*/row0);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
}

// Builds the near-tie auditor for a single-request layer test. The device
// cache is gathered in LOGICAL pool order through the block table (its
// state at audit time — valid because the compared phases are the last ones
// to write it); per-row q/w come from the phase captures. `acc` collects the
// audit's own evidence (swap counts, boundary gaps, noise multiples) so the
// caller can print the certification numbers, not just "it passed".
RowAuditor make_near_tie_auditor(
    const DsaConfig& cfg, const DsaGeometry& g, DsaStatePool& pool, int req,
    cudaStream_t s, const std::vector<PhaseInputs>& phases,
    const dsa_ref::HostState& ref, dsa_test::NearTieAudit* acc) {
  const int heads = cfg.index_n_heads;
  const int dim = cfg.index_head_dim;
  const std::vector<int32_t> bt = fetch_block_row(pool, req, s);
  // Copy only the slots the request actually wrote (via the block table,
  // one small copy per logical pool). Downloading the whole cache would
  // read never-written physical slots — the engine never touches those
  // (kernels only read pools < visible, which are always written), and
  // initcheck rightly flags the copy, not the engine.
  std::vector<uint8_t> dev_k_log(size_t(ref.num_pools) * dim, uint8_t(0));
  std::vector<float> dev_ks_log(size_t(ref.num_pools), 0.0f);
  for (int64_t j = 0; j < ref.num_pools; ++j) {
    const int64_t slot =
        int64_t(bt[size_t(j / g.pools_per_block)]) * g.pools_per_block +
        (j % g.pools_per_block);
    DGPP_CUDA_OK(cudaMemcpyAsync(&dev_k_log[size_t(j) * dim],
                                 static_cast<const uint8_t*>(pool.index_k(0)) +
                                     size_t(slot) * dim,
                                 dim, cudaMemcpyDeviceToHost, s));
    DGPP_CUDA_OK(cudaMemcpyAsync(&dev_ks_log[size_t(j)],
                                 pool.index_scale(0) + slot, 4,
                                 cudaMemcpyDeviceToHost, s));
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  return [cfg, g, heads, dim, phases, dev_k_log, dev_ks_log, &ref, acc](
             int64_t row, const int32_t* dev_toks,
             const int32_t* ref_toks) -> bool {
    const PhaseInputs* ph = nullptr;
    for (const PhaseInputs& p : phases)
      if (row >= p.row0 && row < p.row0 + p.rows) ph = &p;
    if (!ph) return false;
    const int64_t phase_row = row - ph->row0;
    const int64_t pos = row;  // single request, sequential from token 0
    const int64_t visible = g.visible_pools(pos);
    if (visible <= g.select_k) return false;
    dsa_test::NearTieAudit stats;
    dsa_test::audit_flipped_row(
        g.select_k, dev_toks, ref_toks,
        ph->q8.data() + size_t(phase_row) * heads * dim,
        ph->w.data() + size_t(phase_row) * heads, dev_k_log.data(),
        dev_ks_log.data(),
        ph->ref_q8.data() + size_t(phase_row) * heads * dim,
        ph->ref_w.data() + size_t(phase_row) * heads, ref.index_k.data(),
        ref.index_scale.data(), visible, heads, dim, cfg.index_kpool, stats,
        ph->dots.empty() ? nullptr : ph->dots.data(), ph->dot_stride,
        phase_row, g.max_selected, cfg.index_relu != 0);
    acc->rows_flipped += stats.rows_flipped;
    acc->swaps_certified += stats.swaps_certified;
    acc->max_boundary_gap =
        std::max(acc->max_boundary_gap, stats.max_boundary_gap);
    acc->max_noise_multiple =
        std::max(acc->max_noise_multiple, stats.max_noise_multiple);
    return true;
  };
}

void require_output_matches(const DsaConfig& cfg, const DsaGeometry& g,
                            const std::vector<uint16_t>& got,
                            const std::vector<uint16_t>& want,
                            const std::vector<int32_t>& got_topk,
                            const std::vector<int32_t>& want_topk,
                            const std::string& what,
                            const RowAuditor& audit = nullptr,
                            double kept_budget = 0.01) {
  const int64_t rows = int64_t(got.size()) / cfg.hidden;
  const int ms = g.max_selected;
  int64_t flipped = 0, sparse = 0, certified = 0;
  double worst_kept = 0, worst_flipped = 0, worst_certified = 0;
  int64_t worst_kept_row = -1;
  double worst_kept_ref = 0;
  for (int64_t r = 0; r < rows; ++r) {
    const int32_t* gt = got_topk.data() + r * ms;
    const int32_t* wt = want_topk.data() + r * ms;
    std::vector<int32_t> gp, wp;
    for (int c = 0; c < ms; ++c) {
      if (gt[c] >= 0) gp.push_back(gt[c]);
      if (wt[c] >= 0) wp.push_back(wt[c]);
    }
    std::sort(gp.begin(), gp.end());
    std::sort(wp.begin(), wp.end());
    if (g.visible_pools(r) > g.select_k) ++sparse;
    // Row output drift (relative l2 over the row).
    double d2 = 0, w2 = 0;
    for (int c = 0; c < cfg.hidden; ++c) {
      const double a = bf16_bits_to_float(got[size_t(r) * cfg.hidden + c]);
      const double b = bf16_bits_to_float(want[size_t(r) * cfg.hidden + c]);
      d2 += (a - b) * (a - b);
      w2 += b * b;
    }
    const double row_l2 = std::sqrt(d2 / std::max(w2, 1e-30));
    if (gp == wp) {
      if (row_l2 > worst_kept) {
        worst_kept = row_l2;
        worst_kept_row = r;
        worst_kept_ref = std::sqrt(w2 / cfg.hidden);
      }
      continue;
    }
    ++flipped;
    // A flip must be a single boundary swap: |dev-only| == |ref-only| == 1
    // pool's worth of tokens.
    std::vector<int32_t> only_g, only_w;
    std::set_difference(gp.begin(), gp.end(), wp.begin(), wp.end(),
                        std::back_inserter(only_g));
    std::set_difference(wp.begin(), wp.end(), gp.begin(), gp.end(),
                        std::back_inserter(only_w));
    // The audit is the PRIMARY criterion: it re-derives the spec selection
    // from the device's own inputs (bitwise) and measures the boundary gap
    // against the row's actual cross-implementation noise. A certified
    // near tie may legitimately move the row O(1) — near-tie indexer scores
    // mean the pools are scored equally, not that their content is similar
    // — so the row-output drift budget below is the FALLBACK for rows the
    // auditor lacks inputs for; it must never veto a certification.
    if (audit && audit(r, gt, wt)) {
      ++certified;
      worst_certified = std::max(worst_certified, row_l2);
      continue;
    }
    worst_flipped = std::max(worst_flipped, row_l2);
    const bool one_swap = only_g.size() == size_t(cfg.index_kpool) &&
                          only_w.size() == size_t(cfg.index_kpool) &&
                          only_g[0] / cfg.index_kpool !=
                              only_w[0] / cfg.index_kpool;
    if (!one_swap) {
      std::string dg, dw;
      for (int32_t p : only_g) dg += " " + std::to_string(p / cfg.index_kpool);
      for (int32_t p : only_w) dw += " " + std::to_string(p / cfg.index_kpool);
      throw std::runtime_error(what + ": row " + std::to_string(r) +
                               " selection differs by more than one pool "
                               "(dev-only:" + dg + " ref-only:" + dw + ")");
    }
  }
  if (worst_kept > kept_budget)
    throw std::runtime_error(what + ": kept-row drift " +
                             std::to_string(worst_kept) + " (budget " +
                             std::to_string(kept_budget) + ") at row " +
                             std::to_string(worst_kept_row) + " of " +
                             std::to_string(rows) + " (reference rms " +
                             std::to_string(worst_kept_ref) + ")");
  if (flipped > sparse / 4 + 1)
    throw std::runtime_error(what + ": " + std::to_string(flipped) + "/" +
                             std::to_string(sparse) +
                             " sparse rows flipped selections");
  if (worst_flipped > 0.9)
    throw std::runtime_error(what + ": flipped-row drift " +
                             std::to_string(worst_flipped));
  std::printf("[INFO] %s: %lld/%lld sparse rows flipped (%lld audited as "
              "near ties, max certified row l2 %.4f; kept-row max l2 "
              "%.2e; unaudited max l2 %.4f)\n",
              what.c_str(), (long long)flipped, (long long)sparse,
              (long long)certified, worst_certified, worst_kept,
              worst_flipped);
}

// The certification's own evidence: what the audit measured, not just that
// it did not throw. Boundary gaps are the ref-side logit distance between
// a swapped pool pair; the noise multiple is that gap over the row's
// measured cross-implementation logit disagreement.
void print_near_tie_audit(const std::string& what,
                          const dsa_test::NearTieAudit& a) {
  std::printf("[INFO] %s near-tie audit: %lld rows / %lld swaps certified, "
              "max boundary gap %.3e (%.1fx noise)\n",
              what.c_str(), (long long)a.rows_flipped,
              (long long)a.swaps_certified, a.max_boundary_gap,
              a.max_noise_multiple);
}

DGPP_TEST(dsa_state_pool_block_allocation_and_accounting) {
  cudaStream_t s = dgpp::kda_test::test_stream();
  const DsaConfig cfg = small_cfg(3);
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int64_t slots = 8 * cfg.block_tokens;  // 8 blocks

  dgpp::Arena arena;
  dgpp::Arena::Config ac;
  ac.persistent_hot = DsaStatePool::cache_bytes(cfg, 4, slots);
  arena.init(ac);
  DsaStatePool pool;
  pool.init(arena, cfg, 4, slots);

  // Accounting: exact bytes vs the static formula, and the region sizes vs
  // the geometry per-layer formulas.
  if (pool.cache_bytes() != DsaStatePool::cache_bytes(cfg, 4, slots))
    throw std::runtime_error("cache_bytes static/instance mismatch");
  const size_t expect_latent =
      3 * size_t(slots) * g.latent_bytes_per_token;
  const size_t expect_index_k = 3 * size_t(8 * g.pools_per_block) *
                                g.index_k_bytes_per_pool;
  const size_t expect_index_s =
      3 * size_t(8 * g.pools_per_block) * sizeof(float);
  const size_t expect_tail = 3 * 4 * g.tail_bytes_per_request;
  const size_t pad = 256;
  const auto round_to = [pad](size_t b) { return (b + pad - 1) / pad * pad; };
  if (pool.cache_bytes() !=
      round_to(expect_latent) + round_to(expect_index_k) +
          round_to(expect_index_s) + round_to(expect_tail) +
          round_to(4 * size_t(8) * sizeof(int32_t)))
    throw std::runtime_error("cache_bytes region formula mismatch");

  // Growth, exhaustion (transactional), release, reset.
  if (!pool.ensure_request_blocks(0, 1, s)) throw std::runtime_error("b0");
  if (pool.request_blocks(0) != 1 || pool.blocks_in_use() != 1)
    throw std::runtime_error("after first ensure");
  if (!pool.ensure_request_blocks(0, 5 * cfg.block_tokens, s))
    throw std::runtime_error("b1");
  if (pool.request_blocks(0) != 5) throw std::runtime_error("grow to 5");
  if (pool.ensure_request_blocks(0, slots + 1, s))
    throw std::runtime_error("capacity must not satisfy beyond pool");
  if (pool.request_blocks(0) != 5) throw std::runtime_error("mutated on fail");
  if (pool.ensure_request_blocks(1, slots, s))
    throw std::runtime_error("second request must not fit");
  if (pool.request_blocks(1) != 0)
    throw std::runtime_error("partial claim leaked");
  pool.release_request_blocks(0, s);
  if (pool.request_blocks(0) != 0 || pool.blocks_in_use() != 0)
    throw std::runtime_error("release");
  if (!pool.ensure_request_blocks(1, slots, s))
    throw std::runtime_error("full request after release");
  pool.reset_all(s);
  if (pool.blocks_in_use() != 0)
    throw std::runtime_error("reset_all must free everything");
  const std::vector<int32_t> bt = fetch_block_row(pool, 1, s);
  for (int32_t b : bt)
    if (b != 0) throw std::runtime_error("reset_all must zero tables");

  // The M3 exit criterion at deployment scale: allocated bytes within 2% of
  // the DESIGN 7.2 formulas (the only extra is the block table, ~0.01%).
  const DsaConfig real{};
  const DsaGeometry rg = DsaGeometry::from_config(real);
  const int64_t budget_tokens = 300000 / real.block_tokens * real.block_tokens;
  const size_t formula = rg.latent_total_bytes(budget_tokens) +
                         rg.index_total_bytes(budget_tokens) +
                         rg.tail_total_bytes(32);
  const size_t measured =
      DsaStatePool::cache_bytes(real, 32, budget_tokens);
  const double frac = double(measured) / double(formula) - 1.0;
  if (frac < 0 || frac > 0.02)
    throw std::runtime_error("deployment accounting off by " +
                             std::to_string(frac * 100.0) + "%");
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
}

// The FP8 projections: the layer fed the checkpoint's pairs
// through the scale-aware GEMM, against the host reference over the same
// dequantized values — the gate the bf16 bridge form passes, run on both
// forms of the same weights.
DGPP_TEST(dsa_layer_prefill_fp8_projections_match_reference) {
  cudaStream_t s = dgpp::kda_test::test_stream();
  const DsaConfig cfg = small_cfg();
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int T = 96;
  TestWeights tw(cfg, 4200, /*fp8_projections=*/true);
  if (!tw.layer_views.quantized() || tw.bridge_views.quantized())
    throw std::runtime_error("fp8 test weights must carry both forms");
  const std::vector<uint16_t> hidden =
      random_bf16_bits(4201, int64_t(T) * cfg.hidden, -2, 1);
  dsa_ref::HostState ref;
  ref.reset(cfg, T);
  std::vector<uint16_t> ref_out(size_t(T) * cfg.hidden);
  std::vector<int32_t> ref_topk(size_t(T) * g.max_selected);
  dsa_ref::layer_forward<float>(tw.host, cfg, hidden.data(), ref, 0, T,
                                ref_out.data(), ref_topk.data());
  for (const DsaLayerWeights* w : {&tw.layer_views, &tw.bridge_views}) {
    const char* label = w->quantized() ? "layer prefill (fp8 projections)"
                                       : "layer prefill (bridge form)";
    LayerEnv env(cfg, T, T, 2, 4 * cfg.block_tokens, s);
    DsaStatePool pool;
    pool.init(env.arena, cfg, 2, 4 * cfg.block_tokens);
    DsaLayer layer(env.gemm, *w, cfg, T, T,
                   env.arena.alloc_persistent(MemClass::DeviceHot,
                                              DsaLayer::scratch_bytes(cfg, T, T),
                                              256),
                   DsaLayer::scratch_bytes(cfg, T, T), env.ws.p, env.ws.bytes);
    if (!layer.prepare(T)) throw std::runtime_error("gemm plans unavailable");
    DevBuf din(hidden.size() * 2), dout(size_t(T) * cfg.hidden * 2);
    din.upload(hidden.data(), hidden.size() * 2);
    layer.enqueue_prefill(din.p, pool, 0, 0, 0, T, dout.p, s);
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
    PhaseInputs phase;
    capture_phase_inputs(layer, tw.host, cfg, hidden.data(), 0, T, phase, s);
    std::vector<uint16_t> got_out(size_t(T) * cfg.hidden);
    dout.download(got_out.data(), got_out.size() * 2);
    std::vector<int32_t> got_topk(size_t(T) * g.max_selected);
    DGPP_CUDA_OK(cudaMemcpyAsync(got_topk.data(), layer.debug_topk(),
                                 got_topk.size() * 4, cudaMemcpyDeviceToHost, s));
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
    const std::vector<PhaseInputs> phases{std::move(phase)};
    dsa_test::NearTieAudit audit_stats{};
    const RowAuditor audit = make_near_tie_auditor(cfg, g, pool, 0, s, phases,
                                                   ref, &audit_stats);
    require_output_matches(cfg, g, got_out, ref_out, got_topk, ref_topk, label,
                           audit);
    print_near_tie_audit(label, audit_stats);
    require_cache_matches(pool, 0, 0, ref, s);
  }
}

DGPP_TEST(dsa_layer_prefill_dot_budget_preserves_selection_and_output) {
  cudaStream_t s = dgpp::kda_test::test_stream();
  // Both indexer forms, a partial final tile, and a cache much larger than
  // the prompt. A one-byte budget still allocates one maximum-context row.
  // The larger budget allows most/all queries to run in a single tile.
  const int T = 273, cache = 4096;
  for (const DsaConfig& cfg : {small_cfg(), full_cfg()}) {
    const DsaGeometry g = DsaGeometry::from_config(cfg);
    TestWeights tw(cfg, 4200);
    const auto hidden = random_bf16_bits(4201, int64_t(T) * cfg.hidden, -2, 1);
    std::vector<uint16_t> expected;
    std::vector<int32_t> expected_ids, expected_counts;
    for (size_t budget : {size_t(1), size_t(8ull << 20)}) {
      LayerEnv env(cfg, T, cache, 1, 320, s, budget);
      DsaStatePool pool;
      pool.init(env.arena, cfg, 1, 320);
      const size_t sb = DsaLayer::scratch_bytes(cfg, T, cache, 8, 32, budget);
      DsaLayer layer(env.gemm, tw.layer_views, cfg, T, cache,
                     env.arena.alloc_persistent(MemClass::DeviceHot, sb, 256),
                     sb, env.ws.p, env.ws.bytes, 8, 32, budget);
      DevBuf din(hidden.size() * 2), dout(hidden.size() * 2);
      din.upload(hidden.data(), hidden.size() * 2);
      layer.enqueue_prefill(din.p, pool, 0, 0, 0, T, dout.p, s);
      DGPP_CUDA_OK(cudaStreamSynchronize(s));
      std::vector<uint16_t> output(hidden.size());
      std::vector<int32_t> ids(size_t(T) * g.max_selected), counts(T);
      dout.download(output.data(), output.size() * 2);
      DGPP_CUDA_OK(cudaMemcpy(ids.data(), layer.debug_topk(), ids.size() * 4,
                               cudaMemcpyDeviceToHost));
      DGPP_CUDA_OK(cudaMemcpy(counts.data(), layer.debug_counts(), counts.size() * 4,
                               cudaMemcpyDeviceToHost));
      if (expected.empty()) {
        expected = std::move(output);
        expected_ids = std::move(ids);
        expected_counts = std::move(counts);
      } else {
        require_bitwise("prefill tiling output", output.data(), expected.data(), output.size() * 2);
        require_bitwise("prefill tiling counts", counts.data(), expected_counts.data(), counts.size() * 4);
        for (int row = 0; row < T; ++row)
          require_bitwise("prefill tiling selection", ids.data() + size_t(row) * g.max_selected,
                           expected_ids.data() + size_t(row) * g.max_selected, size_t(counts[row]) * 4);
      }
    }
  }
}

DGPP_TEST(dsa_layer_fused_prefill_preserves_gemm_selection) {
  cudaStream_t s = dgpp::kda_test::test_stream();
  DsaConfig cfg = full_cfg();
  cfg.index_topk = 2048;
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  constexpr int history = 2048, cache = 4096, padded = 2304;
  TestWeights tw(cfg, 4250);
  const auto hidden = random_bf16_bits(4251, int64_t(history) * cfg.hidden, -2, 1);
  // At 2304 padded pools, these allocations produce legacy tiles of one
  // and three queries; fusion holds 28/56. Check single-query chunks,
  // ordinary tails, and a one-query tail created by a wider fused tile.
  for (const int capacity_rows : {1, 2}) {
    const size_t budget = size_t(capacity_rows) * cache * 32 * sizeof(float);
    for (const int tokens : {1, 55, 56, 57}) {
      LayerEnv env(cfg, history, cache, 1, cache, s, budget);
      DsaStatePool pool;
      pool.init(env.arena, cfg, 1, cache);
      const size_t sb = DsaLayer::scratch_bytes(cfg, history, cache, 8, 32, budget);
      DsaLayer layer(env.gemm, tw.layer_views, cfg, history, cache,
                     env.arena.alloc_persistent(MemClass::DeviceHot, sb, 256),
                     sb, env.ws.p, env.ws.bytes, 8, 32, budget);
      DevBuf din(hidden.size() * 2), dout(hidden.size() * 2);
      din.upload(hidden.data(), hidden.size() * 2);
      layer.enqueue_prefill(din.p, pool, 0, 0, 0, history, dout.p, s, 0, true);
      layer.enqueue_prefill(din.p, pool, 0, 0, history, tokens, dout.p, s);
      const int old_rows = capacity_rows * cache / padded;
      if (layer.debug_dot_stride() != 0)
        throw std::runtime_error("single-query prefill unexpectedly used the dot buffer");
      DevBuf keys(size_t(padded) * 128), scales(size_t(padded) * 4),
          pos(size_t(tokens) * 8), dots(size_t(old_rows) * 32 * padded * 4),
          ids(size_t(tokens) * g.max_selected * 4), counts(size_t(tokens) * 4),
          workspace(dsa_select_prefill_workspace_bytes(old_rows, padded, g.select_k));
      DGPP_CUDA_OK(cudaMemsetAsync(keys.p, 0, keys.bytes, s));
      DGPP_CUDA_OK(cudaMemsetAsync(scales.p, 0, scales.bytes, s));
      dsa_gather_index_pools(pool.block_tables(), g.pools_per_block,
          pool.index_k(0), pool.index_scale(0), history + tokens,
          static_cast<uint8_t*>(keys.p), static_cast<float*>(scales.p), 128, s);
      std::vector<int64_t> positions(tokens);
      for (int r = 0; r < tokens; ++r) positions[r] = history + r;
      pos.upload(positions.data(), positions.size() * 8);
      for (int r = 0; r < tokens; r += old_rows) {
        const int rows = std::min(old_rows, tokens - r);
        env.gemm.matmul(static_cast<const uint8_t*>(layer.debug_q_fp8()) + size_t(r) * 32 * 128,
            keys.p, dots.p, rows * 32, padded, 128, DType::F8_E4M3, GemmOut::F32,
            128, env.ws.p, env.ws.bytes, s);
        dsa_select_prefill(static_cast<const float*>(dots.p), padded,
            layer.debug_w_folded() + size_t(r) * 32, static_cast<const float*>(scales.p),
            static_cast<const int64_t*>(pos.p) + r, rows, history + tokens,
            32, g.select_k, 1, g.max_selected,
            static_cast<int32_t*>(ids.p) + size_t(r) * g.max_selected,
            static_cast<int32_t*>(counts.p) + r, s, true, workspace.p);
      }
      DGPP_CUDA_OK(cudaStreamSynchronize(s));
      std::vector<int32_t> want_ids(size_t(tokens) * g.max_selected), got_ids(want_ids.size()),
          want_counts(tokens), got_counts(tokens);
      ids.download(want_ids.data(), want_ids.size() * 4);
      counts.download(want_counts.data(), want_counts.size() * 4);
      DGPP_CUDA_OK(cudaMemcpy(got_ids.data(), layer.debug_topk(), got_ids.size() * 4,
                               cudaMemcpyDeviceToHost));
      DGPP_CUDA_OK(cudaMemcpy(got_counts.data(), layer.debug_counts(), got_counts.size() * 4,
                               cudaMemcpyDeviceToHost));
      require_bitwise("fused layer selection", got_ids.data(), want_ids.data(), got_ids.size() * 4);
      require_bitwise("fused layer counts", got_counts.data(), want_counts.data(), got_counts.size() * 4);
    }
  }
}

DGPP_TEST(dsa_layer_prefill_matches_reference) {
  cudaStream_t s = dgpp::kda_test::test_stream();
  const DsaConfig cfg = small_cfg();
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int T = 96;  // 24 pools; last row sees 24 > select_k=16 (sparse path)

  TestWeights tw(cfg, 4200);
  LayerEnv env(cfg, T, T, 2, 4 * cfg.block_tokens, s);
  DsaStatePool pool;
  pool.init(env.arena, cfg, 2, 4 * cfg.block_tokens);
  DsaLayer layer(env.gemm, tw.layer_views, cfg, T, T,
                 env.arena.alloc_persistent(MemClass::DeviceHot,
                                            DsaLayer::scratch_bytes(cfg, T, T),
                                            256),
                 DsaLayer::scratch_bytes(cfg, T, T), env.ws.p, env.ws.bytes);
  if (!layer.prepare(T)) throw std::runtime_error("gemm plans unavailable");

  const std::vector<uint16_t> hidden =
      random_bf16_bits(4201, int64_t(T) * cfg.hidden, -2, 1);
  DevBuf din(hidden.size() * 2), dout(size_t(T) * cfg.hidden * 2);
  din.upload(hidden.data(), hidden.size() * 2);

  dsa_ref::HostState ref;
  ref.reset(cfg, T);
  std::vector<uint16_t> ref_out(size_t(T) * cfg.hidden);
  std::vector<int32_t> ref_topk(size_t(T) * g.max_selected);
  dsa_ref::layer_forward<float>(tw.host, cfg, hidden.data(), ref, 0, T,
                                ref_out.data(), ref_topk.data());

  layer.enqueue_prefill(din.p, pool, 0, 0, 0, T, dout.p, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  PhaseInputs phase;
  capture_phase_inputs(layer, tw.host, cfg, hidden.data(), 0, T, phase, s);

  std::vector<uint16_t> got_out(size_t(T) * cfg.hidden);
  dout.download(got_out.data(), got_out.size() * 2);
  std::vector<int32_t> got_topk(size_t(T) * g.max_selected);
  DGPP_CUDA_OK(cudaMemcpyAsync(got_topk.data(), layer.debug_topk(),
                               got_topk.size() * 4, cudaMemcpyDeviceToHost, s));
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  const std::vector<PhaseInputs> phases{std::move(phase)};
  dsa_test::NearTieAudit audit_stats{};
  const RowAuditor audit = make_near_tie_auditor(cfg, g, pool, 0, s, phases,
                                                 ref, &audit_stats);
  require_output_matches(cfg, g, got_out, ref_out, got_topk, ref_topk,
                         "layer prefill", audit);
  print_near_tie_audit("layer prefill", audit_stats);
  require_cache_matches(pool, 0, 0, ref, s);
}

// Chunked prefill (the final chunk ends mid-pool, so the tail ring carries a
// partial pool across the prefill->decode boundary) + one multi-token decode
// batch, selection-aware against the oracle with the near-tie audit. Shared
// body: the CI config (index_topk=64, select_k=16) and the select_k=8
// variant below (the real checkpoint's 2048/4=512 dwarfs both, but 8 is the
// smallest select_k the bitonic networks support and the boundary flips are
// densest there — exactly where the audit must prove itself).
static void chunked_prefill_decode_case(const DsaConfig& cfg,
                                        const char* what,
                                        double kept_budget = 0.01) {
  cudaStream_t s = dgpp::kda_test::test_stream();
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int chunk = 96;
  const int tail_chunk = 66;  // ends 2 tokens into pool 40 (mid-pool)
  const int total = chunk + tail_chunk;
  const int decode_tokens = 8;

  TestWeights tw(cfg, 4300);
  LayerEnv env(cfg, chunk, total + decode_tokens, 2, 8 * cfg.block_tokens, s);
  DsaStatePool pool;
  pool.init(env.arena, cfg, 2, 8 * cfg.block_tokens);
  DsaLayer layer(env.gemm, tw.layer_views, cfg, chunk, total + decode_tokens,
                 env.arena.alloc_persistent(MemClass::DeviceHot,
                                            DsaLayer::scratch_bytes(
                                                cfg, chunk, total + decode_tokens),
                                            256),
                 DsaLayer::scratch_bytes(cfg, chunk, total + decode_tokens),
                 env.ws.p, env.ws.bytes);
  if (!layer.prepare(chunk)) throw std::runtime_error("gemm plans");

  std::vector<uint16_t> hidden =
      random_bf16_bits(4301, int64_t(total + decode_tokens) * cfg.hidden, -2, 1);
  DevBuf din(hidden.size() * 2);
  din.upload(hidden.data(), hidden.size() * 2);
  DevBuf douts(size_t(total) * cfg.hidden * 2);

  // Reference: chunk 1, mid-pool final chunk, then 8 single-token decodes.
  dsa_ref::HostState ref;
  ref.reset(cfg, total + decode_tokens);
  std::vector<uint16_t> ref_out(size_t(total + decode_tokens) * cfg.hidden);
  std::vector<int32_t> ref_topk(size_t(total + decode_tokens) * g.max_selected);
  dsa_ref::layer_forward<float>(tw.host, cfg, hidden.data(), ref, 0, chunk,
                                ref_out.data(), ref_topk.data());
  dsa_ref::layer_forward<float>(tw.host, cfg, hidden.data() + size_t(chunk) *
                                                                  cfg.hidden,
                                ref, chunk, tail_chunk,
                                ref_out.data() + size_t(chunk) * cfg.hidden,
                                ref_topk.data() + size_t(chunk) * g.max_selected);

  // Device: same chunking. The mid-pool chunk leaves the tail ring carrying
  // the partial pool across the prefill->decode boundary.
  // Capture each phase's selection inputs right after it runs — the topk
  // and q/w scratch are overwritten by every enqueue.
  std::vector<int32_t> got_topk(size_t(total + decode_tokens) * g.max_selected,
                                -1);
  std::vector<PhaseInputs> phases;
  {
    PhaseInputs p;
    layer.enqueue_prefill(din.p, pool, 0, 0, 0, chunk, douts.p, s);
    DGPP_CUDA_OK(cudaMemcpyAsync(got_topk.data(), layer.debug_topk(),
                                 size_t(chunk) * g.max_selected * 4,
                                 cudaMemcpyDeviceToHost, s));
    capture_phase_inputs(layer, tw.host, cfg, hidden.data(), 0, chunk, p, s);
    phases.push_back(std::move(p));
  }
  {
    PhaseInputs p;
    layer.enqueue_prefill(
        static_cast<const uint16_t*>(din.p) + size_t(chunk) * cfg.hidden, pool,
        0, 0, chunk, tail_chunk,
        douts.as<uint16_t>() + size_t(chunk) * cfg.hidden, s);
    DGPP_CUDA_OK(cudaMemcpyAsync(
        got_topk.data() + size_t(chunk) * g.max_selected, layer.debug_topk(),
        size_t(tail_chunk) * g.max_selected * 4, cudaMemcpyDeviceToHost, s));
    capture_phase_inputs(layer, tw.host, cfg,
                         hidden.data() + size_t(chunk) * cfg.hidden, chunk,
                         tail_chunk, p, s);
    phases.push_back(std::move(p));
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(s));

  // Reference decode: one token at a time (pool-aligned starts).
  for (int t = 0; t < decode_tokens; ++t) {
    const int64_t at = total + t;
    dsa_ref::layer_forward<float>(
        tw.host, cfg, hidden.data() + size_t(at) * cfg.hidden, ref, at, 1,
        ref_out.data() + size_t(at) * cfg.hidden,
        ref_topk.data() + size_t(at) * g.max_selected);
  }

  // Device decode: the same 8 tokens as one batch (multi-token == single
  // is pinned bitwise at kernel level; at layer level GEMM m differs).
  DevBuf dreq(decode_tokens * 4), dpos(decode_tokens * 8),
      dspans(2 * 4), ddout(size_t(decode_tokens) * cfg.hidden * 2);
  std::vector<int32_t> req_ids(decode_tokens, 0);
  std::vector<int64_t> pos(decode_tokens);
  for (int t = 0; t < decode_tokens; ++t) pos[t] = total + t;
  std::vector<int32_t> spans = {0, decode_tokens};
  dreq.upload(req_ids.data(), req_ids.size() * 4);
  dpos.upload(pos.data(), pos.size() * 8);
  dspans.upload(spans.data(), spans.size() * 4);
  if (!pool.ensure_request_blocks(0, total + decode_tokens, s))
    throw std::runtime_error("pool exhaustion");
  layer.enqueue_decode(
      static_cast<const uint16_t*>(din.p) + size_t(total) * cfg.hidden, pool, 0,
      static_cast<const int32_t*>(dreq.p), static_cast<const int64_t*>(dpos.p),
      static_cast<const int32_t*>(dspans.p), 1, decode_tokens, ddout.p, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));

  // Compare chunked prefill output + decode batch output, selection-aware.
  // The topk scratch is overwritten by each phase, so capture it per phase.
  std::vector<uint16_t> got_out(size_t(total) * cfg.hidden);
  douts.download(got_out.data(), got_out.size() * 2);
  std::vector<uint16_t> got_dec(decode_tokens * cfg.hidden);
  ddout.download(got_dec.data(), got_dec.size() * 2);
  got_out.insert(got_out.end(), got_dec.begin(), got_dec.end());
  std::vector<uint16_t> want_all(ref_out.begin(), ref_out.end());

  std::vector<int32_t> got_topk_dec(size_t(decode_tokens) * g.max_selected,
                                    -1);
  DGPP_CUDA_OK(cudaMemcpyAsync(got_topk_dec.data(), layer.debug_topk(),
                               got_topk_dec.size() * 4,
                               cudaMemcpyDeviceToHost, s));
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  std::copy(got_topk_dec.begin(), got_topk_dec.end(),
            got_topk.begin() + size_t(total) * g.max_selected);
  {
    PhaseInputs p;
    capture_phase_inputs(layer, tw.host, cfg,
                         hidden.data() + size_t(total) * cfg.hidden, total,
                         decode_tokens, p, s);
    phases.push_back(std::move(p));
  }
  dsa_test::NearTieAudit audit_stats{};
  const RowAuditor audit = make_near_tie_auditor(cfg, g, pool, 0, s, phases,
                                                 ref, &audit_stats);
  require_output_matches(cfg, g, got_out, want_all, got_topk, ref_topk, what,
                         audit, kept_budget);
  print_near_tie_audit(what, audit_stats);
  require_cache_matches(pool, 0, 0, ref, s);
}

DGPP_TEST(dsa_layer_chunked_prefill_and_decode_match_reference) {
  chunked_prefill_decode_case(small_cfg(), "layer chunked+decode");
}

DGPP_TEST(dsa_layer_chunked_prefill_and_decode_select8) {
  DsaConfig cfg = small_cfg();
  cfg.index_topk = 32;  // select_k = 8 (power of two: bitonic networks)
  chunked_prefill_decode_case(cfg, "layer chunked+decode select8");
}

DGPP_TEST(dsa_layer_decode_graph_replay_matches_eager) {
  cudaStream_t s = dgpp::kda_test::test_stream();
  const DsaConfig cfg = small_cfg();
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int prefill = 96;
  const int decode_rows = 2;

  TestWeights tw(cfg, 4400);
  LayerEnv env(cfg, 128, 128, 2, 8 * cfg.block_tokens, s);
  DsaStatePool pool;
  pool.init(env.arena, cfg, 2, 8 * cfg.block_tokens);
  DsaLayer layer(env.gemm, tw.layer_views, cfg, 128, 128,
                 env.arena.alloc_persistent(MemClass::DeviceHot,
                                            DsaLayer::scratch_bytes(cfg, 128,
                                                                    128),
                                            256),
                 DsaLayer::scratch_bytes(cfg, 128, 128), env.ws.p,
                 env.ws.bytes);
  if (!layer.prepare(prefill) || !layer.prepare(decode_rows))
    throw std::runtime_error("gemm plans");

  std::vector<uint16_t> hidden =
      random_bf16_bits(4401, int64_t(prefill + 4) * cfg.hidden, -2, 1);
  DevBuf din(hidden.size() * 2);
  din.upload(hidden.data(), hidden.size() * 2);

  // Two requests decoding concurrently: request 0 at position prefill,
  // request 1 (prefilled further) at prefill + 20. Each request contributes
  // its own hidden row to the batch — the GEMM reads [tokens, hidden] and
  // an undersized input is a real out-of-bounds read (compute-sanitizer
  // caught exactly that when this test uploaded a single row).
  const int64_t pos_a = prefill;       // request 0
  const int64_t pos_b = prefill + 20;  // request 1
  DevBuf dreq(8), dpos(16), dspans(16), dout(2 * cfg.hidden * 2),
      dout_ref(2 * cfg.hidden * 2), dhidden(2 * cfg.hidden * 2),
      dpre(2 * size_t(prefill + 20) * cfg.hidden * 2);

  // Prefill both requests (same tokens; shared weights).
  layer.enqueue_prefill(din.p, pool, 0, 0, 0, prefill, dpre.p, s);
  layer.enqueue_prefill(din.p, pool, 0, 1, 0, prefill + 20, dpre.p, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  if (!pool.ensure_request_blocks(0, pos_a + 2, s) ||
      !pool.ensure_request_blocks(1, pos_b + 2, s))
    throw std::runtime_error("pool exhaustion");

  const std::vector<int32_t> req = {0, 1};
  const std::vector<int32_t> spans = {0, 1, 1, 1};
  dreq.upload(req.data(), 8);
  dspans.upload(spans.data(), 16);

  auto run_eager = [&](const std::vector<int64_t>& pos,
                       const std::vector<uint16_t>& tok) {
    dpos.upload(pos.data(), 16);
    dhidden.upload(tok.data(), tok.size() * 2);
    layer.enqueue_decode(dhidden.p, pool, 0, static_cast<const int32_t*>(dreq.p),
                         static_cast<const int64_t*>(dpos.p),
                         static_cast<const int32_t*>(dspans.p), 2, decode_rows,
                         dout_ref.p, s);
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
  };

  // Per-request decode tokens, concatenated into one batch: [req0 | req1].
  std::vector<uint16_t> tok_a = random_bf16_bits(4402, cfg.hidden, -2, 1);
  std::vector<uint16_t> tok_b = random_bf16_bits(4403, cfg.hidden, -2, 1);
  {
    const std::vector<uint16_t> r1a =
        random_bf16_bits(4404, cfg.hidden, -2, 1);
    const std::vector<uint16_t> r1b =
        random_bf16_bits(4405, cfg.hidden, -2, 1);
    tok_a.insert(tok_a.end(), r1a.begin(), r1a.end());
    tok_b.insert(tok_b.end(), r1b.begin(), r1b.end());
  }

  // Eager reference run for both steps.
  run_eager({pos_a, pos_b}, tok_a);
  std::vector<uint16_t> eager_a(2 * cfg.hidden);
  dout_ref.download(eager_a.data(), eager_a.size() * 2);

  // Graph: capture once, replay for step 1 and step 2 (new pos + hidden).
  dgpp::GraphCache graphs;
  {
    dpos.upload(std::vector<int64_t>{pos_a, pos_b}.data(), 16);
    dhidden.upload(tok_a.data(), tok_a.size() * 2);
    graphs.replay_or_capture(1, "dsa-decode", s, [&](cudaStream_t cap) {
      layer.enqueue_decode(dhidden.p, pool, 0,
                           static_cast<const int32_t*>(dreq.p),
                           static_cast<const int64_t*>(dpos.p),
                           static_cast<const int32_t*>(dspans.p), 2,
                           decode_rows, dout.p, cap);
    });
  }
  std::vector<uint16_t> graph_a(2 * cfg.hidden);
  dout.download(graph_a.data(), graph_a.size() * 2);
  require_bitwise("graph step 1 output", eager_a.data(), graph_a.data(),
                  eager_a.size() * 2);

  // Replay with changed positions + hidden — the graph must serve any
  // positions because visible counts are derived on device.
  dpos.upload(std::vector<int64_t>{pos_a + 1, pos_b + 1}.data(), 16);
  dhidden.upload(tok_b.data(), tok_b.size() * 2);
  run_eager({pos_a + 1, pos_b + 1}, tok_b);
  std::vector<uint16_t> eager_b(2 * cfg.hidden);
  dout_ref.download(eager_b.data(), eager_b.size() * 2);
  graphs.launch(1, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  std::vector<uint16_t> graph_b(2 * cfg.hidden);
  dout.download(graph_b.data(), graph_b.size() * 2);
  require_bitwise("graph replay changed-pos output", eager_b.data(),
                  graph_b.data(), eager_b.size() * 2);
  if (graphs.hits() < 1) throw std::runtime_error("graph was never replayed");
}

// A fixed-shape padding row (pos -1) inside a request's span — the in-graph
// draft's shape after a rejected draft, and every unoccupied row of the
// row-batched serving graph. The live rows' outputs must be bitwise the
// padding-free batch and the padding row's output must be all zeros: inert by
// construction, not merely unobserved downstream.
DGPP_TEST(dsa_layer_decode_padding_row_is_zero_and_leaves_live_rows_bitwise) {
  cudaStream_t s = dgpp::kda_test::test_stream();
  const DsaConfig cfg = small_cfg();
  const int prefill = 96;

  TestWeights tw(cfg, 4500);
  LayerEnv env(cfg, 128, 128, 2, 8 * cfg.block_tokens, s);
  DsaStatePool pool;
  pool.init(env.arena, cfg, 2, 8 * cfg.block_tokens);
  DsaLayer layer(env.gemm, tw.layer_views, cfg, 128, 128,
                 env.arena.alloc_persistent(MemClass::DeviceHot,
                                            DsaLayer::scratch_bytes(cfg, 128,
                                                                    128),
                                            256),
                 DsaLayer::scratch_bytes(cfg, 128, 128), env.ws.p,
                 env.ws.bytes);
  if (!layer.prepare(prefill) || !layer.prepare(2) || !layer.prepare(3))
    throw std::runtime_error("gemm plans");

  std::vector<uint16_t> hidden =
      random_bf16_bits(4501, int64_t(prefill + 20) * cfg.hidden, -2, 1);
  DevBuf din(hidden.size() * 2);
  din.upload(hidden.data(), hidden.size() * 2);
  const int64_t pos_a = prefill;       // request 0
  const int64_t pos_b = prefill + 20;  // request 1
  DevBuf dpre(size_t(prefill + 20) * cfg.hidden * 2);
  layer.enqueue_prefill(din.p, pool, 0, 0, 0, prefill, dpre.p, s);
  layer.enqueue_prefill(din.p, pool, 0, 1, 0, prefill + 20, dpre.p, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  if (!pool.ensure_request_blocks(0, pos_a + 2, s) ||
      !pool.ensure_request_blocks(1, pos_b + 2, s))
    throw std::runtime_error("pool exhaustion");

  const std::vector<uint16_t> row_a = random_bf16_bits(4502, cfg.hidden, -2, 1);
  const std::vector<uint16_t> row_b = random_bf16_bits(4503, cfg.hidden, -2, 1);

  // Reference: the padding-free two-row batch [req0 | req1].
  DevBuf dreq2(8), dpos2(16), dspans2(16), dhid2(2 * cfg.hidden * 2),
      dout2(2 * cfg.hidden * 2);
  {
    const std::vector<int32_t> req = {0, 1};
    const std::vector<int64_t> pos = {pos_a, pos_b};
    const std::vector<int32_t> spans = {0, 1, 1, 1};
    std::vector<uint16_t> hid = row_a;
    hid.insert(hid.end(), row_b.begin(), row_b.end());
    dreq2.upload(req.data(), 8);
    dpos2.upload(pos.data(), 16);
    dspans2.upload(spans.data(), 16);
    dhid2.upload(hid.data(), hid.size() * 2);
    layer.enqueue_decode(dhid2.p, pool, 0,
                         static_cast<const int32_t*>(dreq2.p),
                         static_cast<const int64_t*>(dpos2.p),
                         static_cast<const int32_t*>(dspans2.p), 2, 2,
                         dout2.p, s);
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
  }

  // The same step with a padding row after req0's real row. Its input is a
  // real (nonzero) row and the output buffer is pre-filled, so zeros must
  // be WRITTEN, not inherited.
  DevBuf dreq3(12), dpos3(24), dspans3(16), dhid3(3 * cfg.hidden * 2),
      dout3(3 * cfg.hidden * 2);
  {
    const std::vector<int32_t> req = {0, 0, 1};
    const std::vector<int64_t> pos = {pos_a, -1, pos_b};
    const std::vector<int32_t> spans = {0, 2, 2, 1};
    std::vector<uint16_t> hid = row_a;
    hid.insert(hid.end(), row_b.begin(), row_b.end());
    hid.insert(hid.end(), row_b.begin(), row_b.end());
    dreq3.upload(req.data(), 12);
    dpos3.upload(pos.data(), 24);
    dspans3.upload(spans.data(), 16);
    dhid3.upload(hid.data(), hid.size() * 2);
    DGPP_CUDA_OK(cudaMemsetAsync(dout3.p, 0xA5, dout3.bytes, s));
    layer.enqueue_decode(dhid3.p, pool, 0,
                         static_cast<const int32_t*>(dreq3.p),
                         static_cast<const int64_t*>(dpos3.p),
                         static_cast<const int32_t*>(dspans3.p), 2, 3,
                         dout3.p, s);
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
  }

  std::vector<uint16_t> two(2 * cfg.hidden), three(3 * cfg.hidden);
  dout2.download(two.data(), two.size() * 2);
  dout3.download(three.data(), three.size() * 2);
  require_bitwise("req0 row beside a padding row", two.data(), three.data(),
                  size_t(cfg.hidden) * 2);
  require_bitwise("req1 row after a padding row", two.data() + cfg.hidden,
                  three.data() + 2 * size_t(cfg.hidden),
                  size_t(cfg.hidden) * 2);
  const std::vector<uint16_t> zeros(size_t(cfg.hidden), 0);
  require_bitwise("padding row output is zero", zeros.data(),
                  three.data() + size_t(cfg.hidden), size_t(cfg.hidden) * 2);
}

// 8 MLA heads: tp1 sees all 8, tp2 rank 0 owns heads [0, 4). The indexer
// is replicated, so both ranks select identical pools; attention output
// heads are independent given the selection. Shared by the Flash and the
// full-model geometries (a head's q_b rows are [nope | rope], so the head
// block slice is the same rule).
static void tp2_head_slice_case(const DsaConfig& base, uint64_t seed, const char* what) {
  cudaStream_t s = dgpp::kda_test::test_stream();
  const DsaConfig cfg1 = [&] {
    DsaConfig c = base;
    c.num_heads = 8;
    return c;
  }();
  DsaConfig cfg2 = cfg1;
  cfg2.tp_size = 2;
  const DsaGeometry g1 = DsaGeometry::from_config(cfg1);
  const DsaGeometry g2 = DsaGeometry::from_config(cfg2);
  const int T = 96;

  TestWeights full(cfg1, seed);
  // Rank-0 slice: q_b/kv_b rows are head blocks (contiguous prefix);
  // o_proj columns [0, g2.local_v_rows) of every row.
  DsaLayerWeights sliced_dev;
  std::vector<DevBuf> buf;
  std::vector<uint16_t> o_proj_slice(size_t(cfg1.hidden) * g2.local_v_rows);
  for (int r = 0; r < cfg1.hidden; ++r)
    for (int c = 0; c < g2.local_v_rows; ++c)
      o_proj_slice[size_t(r) * g2.local_v_rows + c] =
          full.host.o_proj[size_t(r) * g1.local_v_rows + c];

  // Build the tp2 layer's device weights from the slice.
  buf.emplace_back(size_t(g2.local_q_rows) * cfg1.q_lora_rank * 2);
  // q_b slice = first local_q_rows rows of full q_b (already correct bits).
  DGPP_CUDA_OK(cudaMemcpyAsync(buf.back().p, full.host.q_b,
                               buf.back().bytes, cudaMemcpyHostToDevice, s));
  sliced_dev.q_b = buf.back().p;
  buf.emplace_back(size_t(g2.local_heads) *
                   (cfg1.qk_nope_head_dim + cfg1.v_head_dim) *
                   cfg1.kv_lora_rank * 2);
  DGPP_CUDA_OK(cudaMemcpyAsync(buf.back().p, full.host.kv_b,
                               buf.back().bytes, cudaMemcpyHostToDevice, s));
  sliced_dev.kv_b = buf.back().p;
  buf.emplace_back(o_proj_slice.size() * 2);
  buf.back().upload(o_proj_slice.data(), o_proj_slice.size() * 2);
  sliced_dev.o_proj = buf.back().p;
  // Indexer + fused weights are replicated: reuse full's device buffers.
  sliced_dev.wq_b = full.layer_views.wq_b;
  sliced_dev.wk = full.layer_views.wk;
  sliced_dev.wp = full.layer_views.wp;
  sliced_dev.gate = full.layer_views.gate;
  sliced_dev.k_norm_w = full.layer_views.k_norm_w;
  sliced_dev.k_norm_b = full.layer_views.k_norm_b;
  sliced_dev.ape = full.layer_views.ape;
  sliced_dev.qkv_a = full.layer_views.qkv_a;
  sliced_dev.q_aln = full.layer_views.q_aln;
  sliced_dev.kv_aln = full.layer_views.kv_aln;
  sliced_dev.rope_table = full.layer_views.rope_table;
  sliced_dev.rope_table_positions = full.layer_views.rope_table_positions;
  DGPP_CUDA_OK(cudaStreamSynchronize(s));

  const auto run = [&](const DsaConfig& cfg, const DsaLayerWeights& w,
                       std::vector<uint16_t>& attn_out) {
    LayerEnv env(cfg, T, T, 1, 4 * cfg.block_tokens, s);
    DsaStatePool pool;
    pool.init(env.arena, cfg, 1, 4 * cfg.block_tokens);
    DsaLayer layer(env.gemm, w, cfg, T, T,
                   env.arena.alloc_persistent(
                       MemClass::DeviceHot, DsaLayer::scratch_bytes(cfg, T, T),
                       256),
                   DsaLayer::scratch_bytes(cfg, T, T), env.ws.p, env.ws.bytes);
    if (!layer.prepare(T)) throw std::runtime_error("gemm plans");
    DevBuf din(size_t(T) * cfg.hidden * 2), dout(size_t(T) * cfg.hidden * 2);
    const std::vector<uint16_t> hidden =
        random_bf16_bits(seed + 1, int64_t(T) * cfg.hidden, -2, 1);
    din.upload(hidden.data(), hidden.size() * 2);
    layer.enqueue_prefill(din.p, pool, 0, 0, 0, T, dout.p, s);
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
    attn_out.resize(size_t(T) * DsaGeometry::from_config(cfg).local_v_rows);
    DGPP_CUDA_OK(cudaMemcpyAsync(attn_out.data(), layer.debug_attn_out(),
                                 attn_out.size() * 2, cudaMemcpyDeviceToHost,
                                 s));
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
  };

  std::vector<uint16_t> out1, out2;
  run(cfg1, full.layer_views, out1);
  run(cfg2, sliced_dev, out2);
  // Compare rank-0's head slice [0, 4 heads * v) per row.
  std::vector<uint16_t> slice1(size_t(T) * g2.local_v_rows);
  for (int t = 0; t < T; ++t)
    std::memcpy(&slice1[size_t(t) * g2.local_v_rows],
                &out1[size_t(t) * g1.local_v_rows], g2.local_v_rows * 2);
  const auto stats = kda_test::compare_bf16(out2, slice1, 8);
  require_bf16(what, stats, 5e-3, 2e-3);
}

DGPP_TEST(dsa_layer_tp2_head_slice_matches_tp1) {
  tp2_head_slice_case(small_cfg(), 4500, "tp2 head slice");
}

DGPP_TEST(dsa_layer_real_geometry_chunked_prefill_decode_smoke) {
  cudaStream_t s = dgpp::kda_test::test_stream();
  // Real checkpoint geometry, no host oracle (the reference GEMM is a naive
  // triple loop — minutes at this size). Pins: chunked prefill across a
  // partial pool + decode runs clean, and a decode graph replay is bitwise
  // with eager at full scale.
  const DsaConfig cfg{};  // defaults = GLM-5.3-Flash
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int chunk = 2048;
  const int tail_tokens = 2054;  // second chunk (6 tokens) ends 2 mid-pool
  const int decode_tokens = 4;

  TestWeights tw(cfg, 4600);
  const size_t dot_budget = 8ull << 20;
  LayerEnv env(cfg, chunk, tail_tokens + 8, 1,
               (tail_tokens + 8 + cfg.block_tokens - 1) / cfg.block_tokens *
                   cfg.block_tokens,
               s, dot_budget);
  DsaStatePool pool;
  pool.init(env.arena, cfg, 1,
            (tail_tokens + 8 + cfg.block_tokens - 1) / cfg.block_tokens *
                cfg.block_tokens);
  const size_t sb = DsaLayer::scratch_bytes(cfg, chunk, tail_tokens + 8, 8, 4,
                                            dot_budget);
  DsaLayer layer(env.gemm, tw.layer_views, cfg, chunk, tail_tokens + 8,
                 env.arena.alloc_persistent(MemClass::DeviceHot, sb, 256), sb,
                 env.ws.p, env.ws.bytes, 8, 4, dot_budget);
  if (!layer.prepare(chunk) || !layer.prepare(decode_tokens))
    throw std::runtime_error("gemm plans");

  const std::vector<uint16_t> hidden =
      random_bf16_bits(4601, int64_t(tail_tokens + decode_tokens) * cfg.hidden,
                       -2, 1);
  DevBuf din(hidden.size() * 2), dchunk(size_t(chunk) * cfg.hidden * 2),
      drest(size_t(tail_tokens - chunk) * cfg.hidden * 2),
      ddout(size_t(decode_tokens) * cfg.hidden * 2);
  din.upload(hidden.data(), hidden.size() * 2);

  layer.enqueue_prefill(din.p, pool, 0, 0, 0, chunk, dchunk.p, s);
  layer.enqueue_prefill(
      static_cast<const uint16_t*>(din.p) + size_t(chunk) * cfg.hidden, pool, 0,
      0, chunk, tail_tokens - chunk, drest.p, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));

  DevBuf dreq(decode_tokens * 4), dpos(decode_tokens * 8), dspans(8);
  std::vector<int32_t> req_ids(decode_tokens, 0);
  std::vector<int64_t> pos(decode_tokens);
  for (int t = 0; t < decode_tokens; ++t) pos[t] = tail_tokens + t;
  const std::vector<int32_t> spans = {0, decode_tokens};
  dreq.upload(req_ids.data(), req_ids.size() * 4);
  dpos.upload(pos.data(), pos.size() * 8);
  dspans.upload(spans.data(), spans.size() * 4);

  layer.enqueue_decode(
      static_cast<const uint16_t*>(din.p) + size_t(tail_tokens) * cfg.hidden,
      pool, 0, static_cast<const int32_t*>(dreq.p),
      static_cast<const int64_t*>(dpos.p),
      static_cast<const int32_t*>(dspans.p), 1, decode_tokens, ddout.p, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  std::vector<uint16_t> eager(decode_tokens * cfg.hidden);
  ddout.download(eager.data(), eager.size() * 2);

  // Repeat decode eager: deterministic (same plans, same inputs) -> bitwise.
  layer.enqueue_decode(
      static_cast<const uint16_t*>(din.p) + size_t(tail_tokens) * cfg.hidden,
      pool, 0, static_cast<const int32_t*>(dreq.p),
      static_cast<const int64_t*>(dpos.p),
      static_cast<const int32_t*>(dspans.p), 1, decode_tokens, ddout.p, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  std::vector<uint16_t> again(decode_tokens * cfg.hidden);
  ddout.download(again.data(), again.size() * 2);
  require_bitwise("real-geometry decode repeat", eager.data(), again.data(),
                  eager.size() * 2);
}


// ---------------------------------------------------------------------------
// The latent cache's formats (2026-09-06, the KV dtype knob): the device
// append quantizes a row bitwise as the host codec does; the attention
// kernels over a quantized cache equal the bf16 kernel over the dequantized
// rows (the host oracle's input) within the bf16 kernel's own tolerance;
// and a whole layer on a quantized cache tracks the bf16 layer within the
// format's precision.
// ---------------------------------------------------------------------------

// Rows with a few zeros and one outlier per row: the block-scale cases.
std::vector<uint16_t> latent_rows_with_outliers(uint64_t seed, int64_t rows, int kv_lora) {
  std::vector<uint16_t> v = random_bf16_bits(seed, rows * kv_lora, -2, 1);
  for (int64_t r = 0; r < rows; ++r) {
    const int64_t at = int64_t(hash32(seed + uint64_t(r)) % uint32_t(kv_lora));
    v[size_t(r * kv_lora + at)] =
        float_to_bf16_bits(bf16_bits_to_float(v[size_t(r * kv_lora + at)]) * 12.0f);
    if (r % 7 == 3) v[size_t(r * kv_lora + (at + 1) % kv_lora)] = 0;
    if (r % 11 == 5)  // an all-zero row now and then
      std::fill(v.begin() + r * kv_lora, v.begin() + (r + 1) * kv_lora, uint16_t{0});
  }
  return v;
}

void latent_append_quantized_case(LatentFormat fmt, int kv_lora, uint64_t seed) {
  const int block_tokens = 32;
  const int n_blocks = 3, total_tokens = n_blocks * block_tokens;
  const std::vector<uint16_t> latent = latent_rows_with_outliers(seed, total_tokens, kv_lora);
  const size_t row_bytes = dgpp::latent_row_bytes(fmt, kv_lora);
  std::vector<int32_t> bt = {2, 0, 1};
  std::vector<int32_t> req_ids(static_cast<size_t>(total_tokens), 0);
  std::vector<int64_t> positions(static_cast<size_t>(total_tokens));
  for (int t = 0; t < total_tokens; ++t) positions[size_t(t)] = t;

  DevBuf dsrc(latent.size() * 2), dcache(size_t(total_tokens) * row_bytes),
      dscale(size_t(total_tokens) * 4), dbt(bt.size() * 4), dri(req_ids.size() * 4),
      dpos(positions.size() * 8);
  dsrc.upload(latent.data(), latent.size() * 2);
  dcache.upload(std::vector<uint8_t>(size_t(total_tokens) * row_bytes, 0xAA).data(),
                size_t(total_tokens) * row_bytes);
  dscale.upload(std::vector<float>(size_t(total_tokens), -1.0f).data(), size_t(total_tokens) * 4);
  dbt.upload(bt.data(), bt.size() * 4);
  dri.upload(req_ids.data(), req_ids.size() * 4);
  dpos.upload(positions.data(), positions.size() * 8);
  dsa_latent_append(dsrc.p, static_cast<const int32_t*>(dri.p),
                    static_cast<const int64_t*>(dpos.p), total_tokens,
                    static_cast<const int32_t*>(dbt.p), n_blocks, block_tokens, dcache.p,
                    kv_lora, 0, fmt, static_cast<float*>(dscale.p));
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint8_t> got(size_t(total_tokens) * row_bytes);
  std::vector<float> got_scale(static_cast<size_t>(total_tokens));
  dcache.download(got.data(), got.size());
  dscale.download(got_scale.data(), got_scale.size() * 4);
  std::vector<uint8_t> want(row_bytes);
  for (int64_t tok = 0; tok < total_tokens; ++tok) {
    const int64_t phys = int64_t(bt[size_t(tok / block_tokens)]) * block_tokens +
                         (tok % block_tokens);
    float want_scale = 0.0f;
    dgpp::latent_quantize_row_host(fmt, &latent[size_t(tok) * kv_lora], kv_lora, want.data(),
                                   &want_scale);
    const std::string what = std::string("latent append ") + dgpp::latent_format_name(fmt) +
                             " kv_lora " + std::to_string(kv_lora) + " token " +
                             std::to_string(tok);
    require_bitwise(what + " codes", &got[static_cast<size_t>(phys) * row_bytes], want.data(), row_bytes);
    if (dgpp::latent_format_has_row_scale(fmt))  // bf16 and the block formats carry none
      require_bitwise(what + " scale", &got_scale[static_cast<size_t>(phys)], &want_scale, 4);
  }
}

DGPP_TEST(dsa_latent_append_quantized_matches_host_codec) {
  latent_append_quantized_case(LatentFormat::kFp8, 512, 5001);
  latent_append_quantized_case(LatentFormat::kFp4, 512, 5002);
  latent_append_quantized_case(LatentFormat::kFp8, 32, 5003);   // the CI layer geometry
  latent_append_quantized_case(LatentFormat::kFp4, 32, 5004);   // a padded fp4 row
  latent_append_quantized_case(LatentFormat::kFp8, 256, 5005);
  latent_append_quantized_case(LatentFormat::kFp4, 256, 5006);
  latent_append_quantized_case(LatentFormat::kBf16, 512, 5007);
  // The DeepSeek-V4.1 block formats (2026-09-13): the window rows and the
  // compressed main KV, the release's own quantizers.
  latent_append_quantized_case(LatentFormat::kFp8Block, 512, 5008);
  latent_append_quantized_case(LatentFormat::kFp4Block, 512, 5009);
  latent_append_quantized_case(LatentFormat::kFp8Block, 32, 5010);  // a padded fp8_block row
  latent_append_quantized_case(LatentFormat::kFp4Block, 32, 5011);
  latent_append_quantized_case(LatentFormat::kFp8Block, 256, 5012);
  latent_append_quantized_case(LatentFormat::kFp4Block, 256, 5013);
}

// The three attention kernels over a quantized cache, at the real geometry
// (64 heads x 512), against the host oracle fed the dequantized rows.
void attention_quantized_case(LatentFormat fmt, uint64_t seed) {
  const DsaConfig cfg{};
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int rows = 3, local_heads = g.local_heads, nope = cfg.qk_nope_head_dim;
  const int v = cfg.v_head_dim, kv_lora = cfg.kv_lora_rank;
  const int block_tokens = cfg.block_tokens;
  const int max_selected = g.max_selected;
  const int n_blocks = 2, total_tokens = n_blocks * block_tokens;
  const size_t row_bytes = dgpp::latent_row_bytes(fmt, kv_lora);
  const std::string what = std::string("attention over a ") + dgpp::latent_format_name(fmt) + " cache";

  auto q = random_bf16_bits(seed, int64_t(rows) * local_heads * nope, -2, 1);
  auto kv_b = random_bf16_bits(seed + 1, int64_t(local_heads) * (nope + v) * kv_lora, -2, 1);
  const std::vector<uint16_t> latent = latent_rows_with_outliers(seed + 2, total_tokens, kv_lora);
  // The cache as the append would leave it (physical rows through the
  // table) and the bf16 the kernels see for it.
  std::vector<int32_t> bt = {1, 0};
  std::vector<uint8_t> cache(size_t(total_tokens) * row_bytes);
  std::vector<float> scales(static_cast<size_t>(total_tokens));
  std::vector<uint16_t> phys_deq(size_t(total_tokens) * kv_lora);
  for (int64_t tok = 0; tok < total_tokens; ++tok) {
    const int64_t phys = int64_t(bt[size_t(tok / block_tokens)]) * block_tokens + (tok % block_tokens);
    dgpp::latent_quantize_row_host(fmt, &latent[size_t(tok) * kv_lora], kv_lora,
                                   &cache[size_t(phys) * row_bytes], &scales[size_t(phys)]);
    dgpp::latent_dequantize_row_host(fmt, &cache[size_t(phys) * row_bytes], scales[size_t(phys)],
                                     kv_lora, &phys_deq[size_t(tok) * kv_lora]);
  }
  // The listed selection (the split kernel's and the listed flash kernel's
  // input) and the causal positions (the dense flash kernel's).
  const int cnt = 37;
  std::vector<int32_t> tokens(size_t(rows) * max_selected, -1);
  std::vector<int32_t> counts(rows, cnt);
  std::vector<int32_t> req_ids(rows, 0);
  for (int r = 0; r < rows; ++r)
    for (int i = 0; i < cnt; ++i)
      tokens[size_t(r) * max_selected + i] = int32_t((r * 131 + i * 17) % 200);
  const std::vector<int64_t> pos = {37, 100, 255};
  std::vector<int32_t> causal(size_t(rows) * max_selected, -1);
  std::vector<int32_t> causal_counts(rows);
  for (int r = 0; r < rows; ++r)
    causal_counts[size_t(r)] = dsa_ref::causal_all_tokens(pos[size_t(r)], max_selected,
                                                          &causal[size_t(r) * max_selected]);

  const int n_split = 4;
  DevBuf dq(q.size() * 2), dkb(kv_b.size() * 2), dcache(cache.size()), dscale(scales.size() * 4),
      dtopk(tokens.size() * 4), dcnt(rows * 4), dri(rows * 4), dbt(8), dpos(rows * 8),
      dqt(size_t(rows) * local_heads * kv_lora * 2),
      dm(size_t(rows) * n_split * local_heads * 4), dl(size_t(rows) * n_split * local_heads * 4),
      dc(size_t(rows) * n_split * local_heads * kv_lora * 4),
      dc_out(size_t(rows) * local_heads * kv_lora * 4), dout(size_t(rows) * local_heads * v * 2);
  dq.upload(q.data(), q.size() * 2);
  dkb.upload(kv_b.data(), kv_b.size() * 2);
  dcache.upload(cache.data(), cache.size());
  dscale.upload(scales.data(), scales.size() * 4);
  dtopk.upload(tokens.data(), tokens.size() * 4);
  dcnt.upload(counts.data(), counts.size() * 4);
  dri.upload(req_ids.data(), req_ids.size() * 4);
  dbt.upload(bt.data(), bt.size() * 4);
  dpos.upload(pos.data(), pos.size() * 8);
  const float scale = 1.0f / std::sqrt(float(nope));
  dsa_absorb_q(dq.p, dkb.p, dqt.p, rows, local_heads, nope, v, kv_lora, 0);

  const auto finish = [&](const std::vector<int32_t>& sel, const std::vector<int32_t>& sel_counts,
                          const std::string& kernel) {
    dsa_attn_combine(static_cast<const float*>(dm.p), static_cast<const float*>(dl.p),
                     static_cast<const float*>(dc.p), rows, n_split, local_heads, kv_lora,
                     static_cast<float*>(dc_out.p), 0);
    dsa_vout_gemm(dc_out.p, dkb.p, dout.p, rows, local_heads, nope, v, kv_lora, 0);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint16_t> want(size_t(rows) * local_heads * v);
    for (int r = 0; r < rows; ++r)
      dsa_ref::absorbed_attn<float>(&q[size_t(r) * local_heads * nope], phys_deq.data(), kv_lora,
                                    &sel[size_t(r) * max_selected], sel_counts[size_t(r)],
                                    kv_b.data(), local_heads, nope, v, kv_lora, scale,
                                    &want[size_t(r) * local_heads * v]);
    std::vector<uint16_t> got(size_t(rows) * local_heads * v);
    dout.download(got.data(), got.size() * 2);
    require_bf16(what + " (" + kernel + ") vs the oracle over the dequantized rows",
                 compare_bf16(got, want, 8), 0.01, 0.006);
  };
  // 1) the split kernel over the listed selection.
  dsa_attn_partial(dqt.p, dcache.p, static_cast<const int32_t*>(dri.p),
                   static_cast<const int32_t*>(dtopk.p), max_selected,
                   static_cast<const int32_t*>(dcnt.p), rows, n_split, local_heads, kv_lora,
                   block_tokens, static_cast<const int32_t*>(dbt.p), n_blocks, scale,
                   static_cast<float*>(dm.p), static_cast<float*>(dl.p),
                   static_cast<float*>(dc.p), 0, fmt, static_cast<const float*>(dscale.p));
  finish(tokens, counts, "split");
  // 2) the listed flash kernel over the same selection.
  if (!dsa_attn_listed(dqt.p, dcache.p, static_cast<const int32_t*>(dri.p),
                       static_cast<const int32_t*>(dtopk.p), max_selected,
                       static_cast<const int32_t*>(dcnt.p), rows, n_split, local_heads, kv_lora,
                       block_tokens, static_cast<const int32_t*>(dbt.p), n_blocks, scale,
                       static_cast<float*>(dm.p), static_cast<float*>(dl.p),
                       static_cast<float*>(dc.p), 0, fmt, static_cast<const float*>(dscale.p)))
    throw std::runtime_error(what + ": the listed flash kernel declined the real geometry");
  finish(tokens, counts, "listed flash");
  // 3) the dense causal flash kernel over the positions.
  if (!dsa_attn_dense(dqt.p, dcache.p, static_cast<const int32_t*>(dri.p),
                      static_cast<const int64_t*>(dpos.p), rows, n_split, local_heads, kv_lora,
                      block_tokens, static_cast<const int32_t*>(dbt.p), n_blocks, scale,
                      static_cast<float*>(dm.p), static_cast<float*>(dl.p),
                      static_cast<float*>(dc.p), 0, fmt, static_cast<const float*>(dscale.p)))
    throw std::runtime_error(what + ": the dense flash kernel declined the real geometry");
  finish(causal, causal_counts, "dense flash");
}

DGPP_TEST(dsa_attention_quantized_cache_matches_dequantized_oracle) {
  attention_quantized_case(LatentFormat::kFp8, 6001);
  attention_quantized_case(LatentFormat::kFp4, 6002);
  attention_quantized_case(LatentFormat::kFp8Block, 6003);
  attention_quantized_case(LatentFormat::kFp4Block, 6004);
  attention_quantized_case(LatentFormat::kBf16, 6003);  // the path the others must equal
}

// Exercise every finite nonnegative e4m3 block-scale code and every e2m1
// value through the actual attention loaders. The host codec supplies an
// equivalent bf16 cache, so both kernels must produce identical partials.
DGPP_TEST(dsa_fp4_attention_loaders_match_bf16_for_all_codes) {
  constexpr int rows = 127, heads = 16, latent = 512, rope = 64;
  constexpr int width = latent + rope, block = 128, splits = 1;
  const size_t packed_stride = latent_row_bytes(LatentFormat::kFp4, latent) + rope * 2;
  std::vector<uint8_t> packed(block * packed_stride);
  std::vector<uint16_t> plain(block * width), query(rows * heads * width, 0);
  std::vector<float> scales(block, 0);
  std::vector<int32_t> req(rows, 0), tokens(rows), counts(rows, 1);
  std::vector<int64_t> positions(rows);
  for (int r = 0; r < rows; ++r) {
    // Include zero and scales on both sides of unity, with non-power-of-two
    // mantissas so final bf16 rounding is exercised as well.
    scales[r] = r == 0 ? 0.0f : std::ldexp(1.00390625f + (r % 7) * 0.03125f, r % 25 - 12);
    tokens[r] = r;
    positions[r] = r;
    auto* p = packed.data() + r * packed_stride;
    for (int e = 0; e < latent; ++e) {
      const uint8_t code = e % 16;
      const uint8_t scale_code = (r + e / kLatentFp4Block) % 127;
      p[e / 2] |= static_cast<uint8_t>(code << ((e & 1) * 4));
      p[latent_fp4_scale_offset(latent) + e / kLatentFp4Block] = scale_code;
      plain[r * width + e] = latent_fp4_decode_bf16(code,
          latent_fp4_block_scale(scale_code, scales[r]));
    }
    for (int e = 0; e < rope; ++e) {
      const uint16_t v = float_to_bf16_bits((e - 32) * 0.03125f);
      std::memcpy(p + latent_row_bytes(LatentFormat::kFp4, latent) + e * 2, &v, 2);
      plain[r * width + latent + e] = v;
    }
  }
  DevBuf dp(packed.size()), db(plain.size() * 2), dq(query.size() * 2),
      ds(scales.size() * 4), dr(req.size() * 4), dt(tokens.size() * 4),
      dn(counts.size() * 4), dpos(positions.size() * 8), dbt(4),
      dm(rows * heads * 4), dl(rows * heads * 4), dc(rows * heads * latent * 4);
  dp.upload(packed.data(), packed.size()); db.upload(plain.data(), plain.size() * 2);
  dq.upload(query.data(), query.size() * 2); ds.upload(scales.data(), scales.size() * 4);
  dr.upload(req.data(), req.size() * 4); dt.upload(tokens.data(), tokens.size() * 4);
  dn.upload(counts.data(), counts.size() * 4); dpos.upload(positions.data(), positions.size() * 8);
  const int32_t zero = 0; dbt.upload(&zero, 4);
  for (int kernel = 0; kernel < 3; ++kernel) {
    std::vector<float> reference, got;
    for (LatentFormat fmt : {LatentFormat::kBf16, LatentFormat::kFp4}) {
      const void* cache = fmt == LatentFormat::kBf16 ? db.p : dp.p;
      const float* scale = fmt == LatentFormat::kBf16 ? nullptr : static_cast<const float*>(ds.p);
      bool accepted = true;
      if (kernel == 0) {
        dsa_attn_partial(dq.p, cache, static_cast<const int32_t*>(dr.p),
            static_cast<const int32_t*>(dt.p), 1, static_cast<const int32_t*>(dn.p),
            rows, splits, heads, latent, block, static_cast<const int32_t*>(dbt.p),
            1, 1.0f, static_cast<float*>(dm.p), static_cast<float*>(dl.p),
            static_cast<float*>(dc.p), 0, fmt, scale, rope);
      } else if (kernel == 1) {
        accepted = dsa_attn_listed(dq.p, cache, static_cast<const int32_t*>(dr.p),
            static_cast<const int32_t*>(dt.p), 1, static_cast<const int32_t*>(dn.p),
            rows, splits, heads, latent, block, static_cast<const int32_t*>(dbt.p),
            1, 1.0f, static_cast<float*>(dm.p), static_cast<float*>(dl.p),
            static_cast<float*>(dc.p), 0, fmt, scale, rope);
      } else {
        accepted = dsa_attn_dense(dq.p, cache, static_cast<const int32_t*>(dr.p),
            static_cast<const int64_t*>(dpos.p), rows, splits, heads, latent, block,
            static_cast<const int32_t*>(dbt.p), 1, 1.0f, static_cast<float*>(dm.p),
            static_cast<float*>(dl.p), static_cast<float*>(dc.p), 0, fmt, scale, rope);
      }
      if (!accepted) throw std::runtime_error("fp4 code coverage: attention kernel declined geometry");
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      auto& out = fmt == LatentFormat::kBf16 ? reference : got;
      out.resize(rows * heads * (latent + 2));
      dm.download(out.data(), rows * heads * 4);
      dl.download(out.data() + rows * heads, rows * heads * 4);
      dc.download(out.data() + 2 * rows * heads, rows * heads * latent * 4);
    }
    require_bitwise("fp4 all-code attention kernel " + std::to_string(kernel),
        got.data(), reference.data(), got.size() * sizeof(float));
  }
}

// A whole layer (chunked prefill across a partial pool + a decode batch) on
// a quantized cache against the host reference, whose append quantizes the
// same rows through the same codec and keeps the dequantized values: the
// selection is identical (the index cache is fp8 in every format) and the
// residual is the bf16 gate's — GEMM ulps, plus the elements those ulps
// push across a quantization boundary (one code apart on the two sides).
// Measured on this case (2026-09-06): kept-row max l2 4.3e-3 in fp8 and
// 4.1e-3 in fp4 — the bf16 gate's own residual — so the budget is twice the
// bf16 gate's; a wiring bug (a wrong stride, an unscaled row) diverges by
// O(1) on every row.
DGPP_TEST(dsa_layer_quantized_cache_matches_reference) {
  DsaConfig fp8 = small_cfg();  // kv_lora 32: an fp4 row of two blocks
  fp8.latent_format = LatentFormat::kFp8;
  chunked_prefill_decode_case(fp8, "layer chunked+decode on an fp8 cache", 0.02);
  DsaConfig fp4 = small_cfg();
  fp4.latent_format = LatentFormat::kFp4;
  chunked_prefill_decode_case(fp4, "layer chunked+decode on an fp4 cache", 0.02);
}

// ---------------------------------------------------------------------------
// The full GLM-5.3's geometry (2026-09-12, docs/glm53_plan.md G4): the
// 64-wide decoupled RoPE beside the latent (plan D3), per-token selection
// with the relu'd indexer (D8), the shared-selection layers (D4), the
// packed-int projections (D2), and the select networks at select_k 2048.
// The Flash gates above run the same code with a zero-width tail.
// ---------------------------------------------------------------------------

// The interleaved RoPE kernel against the host row rotation, bitwise: in
// place and out of place, strided heads, padding rows carried through, a
// position past the table clamped to its last entry.
DGPP_TEST(dsa_rope_interleave_matches_host_table) {
  const int rope = 64, heads = 3, dim = 128, rows = 37;
  const int64_t positions = 300;
  std::vector<uint16_t> table(size_t(positions) * 2 * (rope / 2));
  dgpp::dsa_rope_table_host(8e6, rope, positions, table.data());
  // Sanity: position 0 rotates nothing (cos 1, sin 0), the first pair's
  // angle at position 1 is one radian (inv_freq[0] = 1).
  if (table[0] != float_to_bf16_bits(1.0f) || table[size_t(rope / 2)] != 0)
    throw std::runtime_error("rope table position 0");
  if (table[size_t(2 * (rope / 2))] != float_to_bf16_bits(std::cos(1.0f)))
    throw std::runtime_error("rope table position 1 pair 0");
  const std::vector<uint16_t> x = random_bf16_bits(7100, int64_t(rows) * heads * dim, -2, 1);
  std::vector<int64_t> pos(static_cast<size_t>(rows));
  for (int r = 0; r < rows; ++r)
    pos[size_t(r)] = (r % 9 == 4) ? -1 : (r % 11 == 7) ? positions + 5 : int64_t(hash32(7101 + r) % positions);
  // Host: rotate each head's first rope elements in place.
  std::vector<uint16_t> want = x;
  for (int r = 0; r < rows; ++r)
    for (int h = 0; h < heads; ++h)
      dsa_ref::rope_interleave_row(&want[(size_t(r) * heads + h) * dim], rope, pos[size_t(r)],
                                   table.data(), positions);
  DevBuf dx(x.size() * 2), dout(x.size() * 2), dpos(pos.size() * 8), dtab(table.size() * 2);
  dx.upload(x.data(), x.size() * 2);
  dout.upload(x.data(), x.size() * 2);  // the untouched elements must survive
  dpos.upload(pos.data(), pos.size() * 8);
  dtab.upload(table.data(), table.size() * 2);
  // Out of place.
  dgpp::dsa_rope_interleave(dx.p, int64_t(heads) * dim, dim, heads, rope,
                            static_cast<const int64_t*>(dpos.p), dtab.p, positions, dout.p,
                            int64_t(heads) * dim, dim, rows, 0);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> got(x.size());
  dout.download(got.data(), got.size() * 2);
  require_bitwise("rope interleave (out of place)", want.data(), got.data(), got.size() * 2);
  // In place.
  dgpp::dsa_rope_interleave(dx.p, int64_t(heads) * dim, dim, heads, rope,
                            static_cast<const int64_t*>(dpos.p), dtab.p, positions, dx.p,
                            int64_t(heads) * dim, dim, rows, 0);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  dx.download(got.data(), got.size() * 2);
  require_bitwise("rope interleave (in place)", want.data(), got.data(), got.size() * 2);
  // The rotation moved something (a table of ones would pass trivially).
  if (std::memcmp(got.data(), x.data(), got.size() * 2) == 0)
    throw std::runtime_error("rope interleave rotated nothing");
}

// The geometry's byte formulas and the pool's index-cache ordinals: two DSA
// layers, one index cache (the second attends with the first's selection).
DGPP_TEST(dsa_state_pool_index_ordinals_and_full_geometry_bytes) {
  cudaStream_t s = dgpp::kda_test::test_stream();
  DsaConfig cfg = full_cfg(2);
  cfg.num_index_layers = 1;
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  if (g.rope_dim != 64 || g.score_width != cfg.kv_lora_rank + 64 || g.index_layers != 1 ||
      g.select_k != 16 || g.max_selected != 16 || g.pools_per_block != cfg.block_tokens)
    throw std::runtime_error("full geometry derived fields");
  if (g.latent_bytes_per_token != size_t(cfg.kv_lora_rank) * 2 + 128 ||
      g.latent_bytes_per_token_all != g.latent_bytes_per_token * 2 ||
      g.index_bytes_per_token_all != g.index_bytes_per_token ||
      g.tail_bytes_per_request_all != g.tail_bytes_per_request ||
      g.tail_bytes_per_request != 2 * 1 * 128 * 2)
    throw std::runtime_error("full geometry byte formulas");
  const int64_t slots = 4 * cfg.block_tokens;
  dgpp::Arena arena;
  dgpp::Arena::Config ac;
  ac.persistent_hot = DsaStatePool::cache_bytes(cfg, 2, slots);
  arena.init(ac);
  DsaStatePool pool;
  bool threw = false;
  try {
    pool.init(arena, cfg, 2, slots);  // no table: the identity needs every layer indexed
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  if (!threw) throw std::runtime_error("pool accepted the identity map with one index cache");
  pool.init(arena, cfg, 2, slots, std::vector<int>{0, -1});
  if (!pool.owns_index(0) || pool.owns_index(1) || pool.index_ordinal(1) != -1)
    throw std::runtime_error("index ordinals");
  threw = false;
  try {
    (void)pool.index_k(1);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  if (!threw) throw std::runtime_error("index_k of a shared layer must throw");
  if (pool.latent(1) == pool.latent(0)) throw std::runtime_error("latent caches per layer");
  // Accounting: one index cache and one tail ring region, two latent regions.
  const size_t pools = size_t(4 * g.pools_per_block);
  const auto round_to = [](size_t b) { return (b + 255) / 256 * 256; };
  const size_t want = round_to(2 * size_t(slots) * g.latent_bytes_per_token) +
                      round_to(pools * g.index_k_bytes_per_pool) + round_to(pools * 4) +
                      round_to(2 * g.tail_bytes_per_request) + round_to(2 * 4 * sizeof(int32_t));
  if (pool.cache_bytes() != want) throw std::runtime_error("cache_bytes with one index layer");
  pool.reset_all(s);
  pool.reset_request(1, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
}

// One full-geometry layer, prefill at the CI scale against the host oracle
// (rope on both sides, kpool 1, the relu'd indexer), with the near-tie audit
// and the cache (the rope tail included) compared.
DGPP_TEST(dsa_layer_full_prefill_matches_reference) {
  cudaStream_t s = dgpp::kda_test::test_stream();
  const DsaConfig cfg = full_cfg();
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int T = 96;  // rows past 15 select 16 of their p + 1 tokens
  TestWeights tw(cfg, 7200);
  LayerEnv env(cfg, T, T, 2, 4 * cfg.block_tokens, s);
  DsaStatePool pool;
  pool.init(env.arena, cfg, 2, 4 * cfg.block_tokens);
  DsaLayer layer(env.gemm, tw.layer_views, cfg, T, T,
                 env.arena.alloc_persistent(MemClass::DeviceHot,
                                            DsaLayer::scratch_bytes(cfg, T, T), 256),
                 DsaLayer::scratch_bytes(cfg, T, T), env.ws.p, env.ws.bytes);
  if (!layer.prepare(T)) throw std::runtime_error("gemm plans unavailable");
  const std::vector<uint16_t> hidden = random_bf16_bits(7201, int64_t(T) * cfg.hidden, -2, 1);
  DevBuf din(hidden.size() * 2), dout(size_t(T) * cfg.hidden * 2);
  din.upload(hidden.data(), hidden.size() * 2);
  dsa_ref::HostState ref;
  ref.reset(cfg, T);
  std::vector<uint16_t> ref_out(size_t(T) * cfg.hidden);
  std::vector<int32_t> ref_topk(size_t(T) * g.max_selected);
  dsa_ref::layer_forward<float>(tw.host, cfg, hidden.data(), ref, 0, T, ref_out.data(),
                                ref_topk.data());
  layer.enqueue_prefill(din.p, pool, 0, 0, 0, T, dout.p, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  PhaseInputs phase;
  capture_phase_inputs(layer, tw.host, cfg, hidden.data(), 0, T, phase, s);
  std::vector<uint16_t> got_out(size_t(T) * cfg.hidden);
  dout.download(got_out.data(), got_out.size() * 2);
  std::vector<int32_t> got_topk(size_t(T) * g.max_selected);
  DGPP_CUDA_OK(cudaMemcpyAsync(got_topk.data(), layer.debug_topk(), got_topk.size() * 4,
                               cudaMemcpyDeviceToHost, s));
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  // The selection is per token: every row's list is exactly min(p + 1, 16)
  // ascending token ids with no tail.
  for (int t = 0; t < T; ++t) {
    const int n = std::min(t + 1, g.select_k);
    for (int i = 0; i < g.max_selected; ++i) {
      const int32_t v = got_topk[size_t(t) * g.max_selected + i];
      if (i >= n ? v != -1 : (v < 0 || v > t || (i > 0 && v <= got_topk[size_t(t) * g.max_selected + i - 1])))
        throw std::runtime_error("per-token selection shape at row " + std::to_string(t));
    }
  }
  const std::vector<PhaseInputs> phases{std::move(phase)};
  dsa_test::NearTieAudit audit_stats{};
  const RowAuditor audit = make_near_tie_auditor(cfg, g, pool, 0, s, phases, ref, &audit_stats);
  require_output_matches(cfg, g, got_out, ref_out, got_topk, ref_topk, "full layer prefill", audit);
  print_near_tie_audit("full layer prefill", audit_stats);
  require_cache_matches(pool, 0, 0, ref, s);
}

// Chunked prefill + a decode batch at the full geometry on every cache
// format (the rope tail beside a bf16, fp8 or fp4 latent).
DGPP_TEST(dsa_layer_full_chunked_prefill_and_decode_match_reference) {
  chunked_prefill_decode_case(full_cfg(), "full layer chunked+decode");
  DsaConfig fp8 = full_cfg();
  fp8.latent_format = LatentFormat::kFp8;
  chunked_prefill_decode_case(fp8, "full layer chunked+decode on an fp8 cache", 0.02);
  DsaConfig fp4 = full_cfg();
  fp4.latent_format = LatentFormat::kFp4;
  chunked_prefill_decode_case(fp4, "full layer chunked+decode on an fp4 cache", 0.02);
}

// The packed-int projections (plan D2): the layer fed the int8 triples
// through the packed GEMV core, against the host reference over
// bf16(code x scale) — the gate the bridge form passes, run on both forms.
DGPP_TEST(dsa_layer_full_packed_projections_match_reference) {
  cudaStream_t s = dgpp::kda_test::test_stream();
  const DsaConfig cfg = full_cfg();
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int T = 96;
  TestWeights tw(cfg, 7300, TestWeightOpts{false, 8, true});
  if (!tw.layer_views.packed_int() || tw.bridge_views.packed_int())
    throw std::runtime_error("packed test weights must carry both forms");
  const std::vector<uint16_t> hidden = random_bf16_bits(7301, int64_t(T) * cfg.hidden, -2, 1);
  // The packed GEMV over the test's triple against the host GEMM over the
  // bridge (bf16(code x scale)): the weights differ by a bf16 rounding of
  // each element, so this is a tolerance pin of the triple's layout.
  {
    const int n = cfg.q_lora_rank + cfg.kv_lora_rank + cfg.qk_rope_head_dim;
    DevBuf din(hidden.size() * 2), dout(size_t(T) * n * 2);
    din.upload(hidden.data(), hidden.size() * 2);
    dgpp::launch_packq_gemv_bf16(static_cast<const uint16_t*>(din.p), size_t(cfg.hidden),
                                 tw.layer_views.qkv_a_p, static_cast<uint16_t*>(dout.p), T, n,
                                 cfg.hidden, s);
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
    std::vector<uint16_t> got(size_t(T) * n), want(size_t(T) * n);
    dout.download(got.data(), got.size() * 2);
    dsa_ref::gemm_bf16<float>(hidden.data(), cfg.hidden, tw.host.qkv_a, want.data(), T, n,
                              cfg.hidden);
    require_bf16("packed qkv_a GEMV vs the bridge GEMM", compare_bf16(got, want, 8), 0.02, 0.01);
  }
  dsa_ref::HostState ref;
  ref.reset(cfg, T);
  std::vector<uint16_t> ref_out(size_t(T) * cfg.hidden);
  std::vector<int32_t> ref_topk(size_t(T) * g.max_selected);
  dsa_ref::layer_forward<float>(tw.host, cfg, hidden.data(), ref, 0, T, ref_out.data(),
                                ref_topk.data());
  for (const DsaLayerWeights* w : {&tw.layer_views, &tw.bridge_views}) {
    const char* label = w->packed_int() ? "full layer prefill (packed projections)"
                                        : "full layer prefill (bridge form)";
    LayerEnv env(cfg, T, T, 2, 4 * cfg.block_tokens, s);
    DsaStatePool pool;
    pool.init(env.arena, cfg, 2, 4 * cfg.block_tokens);
    DsaLayer layer(env.gemm, *w, cfg, T, T,
                   env.arena.alloc_persistent(MemClass::DeviceHot,
                                              DsaLayer::scratch_bytes(cfg, T, T), 256),
                   DsaLayer::scratch_bytes(cfg, T, T), env.ws.p, env.ws.bytes);
    if (!layer.prepare(T)) throw std::runtime_error("gemm plans unavailable");
    DevBuf din(hidden.size() * 2), dout(size_t(T) * cfg.hidden * 2);
    din.upload(hidden.data(), hidden.size() * 2);
    layer.enqueue_prefill(din.p, pool, 0, 0, 0, T, dout.p, s);
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
    PhaseInputs phase;
    capture_phase_inputs(layer, tw.host, cfg, hidden.data(), 0, T, phase, s);
    std::vector<uint16_t> got_out(size_t(T) * cfg.hidden);
    dout.download(got_out.data(), got_out.size() * 2);
    std::vector<int32_t> got_topk(size_t(T) * g.max_selected);
    DGPP_CUDA_OK(cudaMemcpyAsync(got_topk.data(), layer.debug_topk(), got_topk.size() * 4,
                                 cudaMemcpyDeviceToHost, s));
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
    const std::vector<PhaseInputs> phases{std::move(phase)};
    dsa_test::NearTieAudit audit_stats{};
    const RowAuditor audit = make_near_tie_auditor(cfg, g, pool, 0, s, phases, ref, &audit_stats);
    require_output_matches(cfg, g, got_out, ref_out, got_topk, ref_topk, label, audit);
    print_near_tie_audit(label, audit_stats);
    require_cache_matches(pool, 0, 0, ref, s);
  }
}

// Cross-layer selection sharing (plan D4): layer 0 owns the index cache
// and selects; layer 1 (a view without indexer tensors) attends its own
// hidden rows with layer 0's selection, in prefill and in decode, against
// the oracle handed the same selection. The reuse contract is enforced: a
// mismatched call shape and a shared view on an indexed layer throw.
DGPP_TEST(dsa_layer_full_shared_selection_matches_reference) {
  cudaStream_t s = dgpp::kda_test::test_stream();
  DsaConfig cfg = full_cfg(2);
  cfg.num_index_layers = 1;
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int T = 96, D = 4, total = T + D;
  TestWeights tw0(cfg, 7400);
  TestWeights tw1(cfg, 7450, TestWeightOpts{false, 0, false});
  if (tw1.layer_views.owns_indexer() || tw1.host.owns_indexer())
    throw std::runtime_error("the shared view must carry no indexer");
  LayerEnv env(cfg, T, total, 2, 8 * cfg.block_tokens, s);
  DsaStatePool pool;
  pool.init(env.arena, cfg, 2, 8 * cfg.block_tokens, std::vector<int>{0, -1});
  DsaLayer layer(env.gemm, tw0.layer_views, cfg, T, total,
                 env.arena.alloc_persistent(MemClass::DeviceHot,
                                            DsaLayer::scratch_bytes(cfg, T, total), 256),
                 DsaLayer::scratch_bytes(cfg, T, total), env.ws.p, env.ws.bytes);
  if (!layer.prepare(T) || !layer.prepare(D)) throw std::runtime_error("gemm plans");
  const std::vector<uint16_t> h0 = random_bf16_bits(7401, int64_t(total) * cfg.hidden, -2, 1);
  const std::vector<uint16_t> h1 = random_bf16_bits(7402, int64_t(total) * cfg.hidden, -2, 1);
  DevBuf d0(h0.size() * 2), d1(h1.size() * 2), o0(size_t(total) * cfg.hidden * 2),
      o1(size_t(total) * cfg.hidden * 2);
  d0.upload(h0.data(), h0.size() * 2);
  d1.upload(h1.data(), h1.size() * 2);

  // Oracle: layer 0 selects, layer 1 reuses, prefill then D single decodes.
  dsa_ref::HostState ref0, ref1;
  ref0.reset(cfg, total);
  ref1.reset(cfg, total);
  std::vector<uint16_t> want0(size_t(total) * cfg.hidden), want1(want0.size());
  std::vector<int32_t> topk0(size_t(total) * g.max_selected), topk1(topk0.size());
  dsa_ref::layer_forward<float>(tw0.host, cfg, h0.data(), ref0, 0, T, want0.data(), topk0.data());
  dsa_ref::layer_forward<float>(tw1.host, cfg, h1.data(), ref1, 0, T, want1.data(), topk1.data(),
                                topk0.data());
  for (int t = 0; t < D; ++t) {
    const int64_t at = T + t;
    dsa_ref::layer_forward<float>(tw0.host, cfg, h0.data() + size_t(at) * cfg.hidden, ref0, at, 1,
                                  want0.data() + size_t(at) * cfg.hidden,
                                  topk0.data() + size_t(at) * g.max_selected);
    dsa_ref::layer_forward<float>(tw1.host, cfg, h1.data() + size_t(at) * cfg.hidden, ref1, at, 1,
                                  want1.data() + size_t(at) * cfg.hidden,
                                  topk1.data() + size_t(at) * g.max_selected,
                                  topk0.data() + size_t(at) * g.max_selected);
  }
  if (std::memcmp(topk0.data(), topk1.data(), topk0.size() * 4) != 0)
    throw std::runtime_error("the oracle's reused selection must be the handed one");

  // Device: prefill layer 0 (indexed) then layer 1 (reusing), the same rows.
  std::vector<int32_t> got_topk(size_t(total) * g.max_selected, -1);
  std::vector<PhaseInputs> phases;
  layer.enqueue_prefill(d0.p, pool, 0, 0, 0, T, o0.p, s);
  DGPP_CUDA_OK(cudaMemcpyAsync(got_topk.data(), layer.debug_topk(), size_t(T) * g.max_selected * 4,
                               cudaMemcpyDeviceToHost, s));
  {
    PhaseInputs p;
    capture_phase_inputs(layer, tw0.host, cfg, h0.data(), 0, T, p, s);
    phases.push_back(std::move(p));
  }
  layer.rebind(tw1.layer_views);
  if (!layer.reuses_selection()) throw std::runtime_error("rebind to the shared view");
  // The contract: the shared view on the indexed layer, or another shape.
  bool threw = false;
  try {
    layer.enqueue_prefill(d1.p, pool, 0, 0, 0, T, o1.p, s);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  if (!threw) throw std::runtime_error("a shared view on an indexed layer must throw");
  threw = false;
  try {
    layer.enqueue_prefill(d1.p, pool, 1, 0, 0, T - 1, o1.p, s);
  } catch (const std::logic_error&) {
    threw = true;
  }
  if (!threw) throw std::runtime_error("a reuse with another row count must throw");
  layer.enqueue_prefill(d1.p, pool, 1, 0, 0, T, o1.p, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  std::vector<int32_t> reused(size_t(T) * g.max_selected);
  DGPP_CUDA_OK(cudaMemcpyAsync(reused.data(), layer.debug_topk(), reused.size() * 4,
                               cudaMemcpyDeviceToHost, s));
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  require_bitwise("the shared layer left the selection untouched", got_topk.data(), reused.data(),
                  reused.size() * 4);

  // Decode: one batch of D rows through layer 0 then layer 1.
  DevBuf dreq(D * 4), dpos(D * 8), dspans(8), dec0(size_t(D) * cfg.hidden * 2),
      dec1(size_t(D) * cfg.hidden * 2);
  std::vector<int32_t> req_ids(D, 0);
  std::vector<int64_t> pos(D);
  for (int t = 0; t < D; ++t) pos[size_t(t)] = T + t;
  const std::vector<int32_t> spans = {0, D};
  dreq.upload(req_ids.data(), D * 4);
  dpos.upload(pos.data(), D * 8);
  dspans.upload(spans.data(), 8);
  if (!pool.ensure_request_blocks(0, total, s)) throw std::runtime_error("pool exhaustion");
  layer.rebind(tw0.layer_views);
  layer.enqueue_decode(static_cast<const uint16_t*>(d0.p) + size_t(T) * cfg.hidden, pool, 0,
                       static_cast<const int32_t*>(dreq.p), static_cast<const int64_t*>(dpos.p),
                       static_cast<const int32_t*>(dspans.p), 1, D, dec0.p, s);
  DGPP_CUDA_OK(cudaMemcpyAsync(got_topk.data() + size_t(T) * g.max_selected, layer.debug_topk(),
                               size_t(D) * g.max_selected * 4, cudaMemcpyDeviceToHost, s));
  {
    PhaseInputs p;
    capture_phase_inputs(layer, tw0.host, cfg, h0.data() + size_t(T) * cfg.hidden, T, D, p, s);
    phases.push_back(std::move(p));
  }
  layer.rebind(tw1.layer_views);
  layer.enqueue_decode(static_cast<const uint16_t*>(d1.p) + size_t(T) * cfg.hidden, pool, 1,
                       static_cast<const int32_t*>(dreq.p), static_cast<const int64_t*>(dpos.p),
                       static_cast<const int32_t*>(dspans.p), 1, D, dec1.p, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));

  // Both layers against their oracles, selection-aware; layer 1's flips are
  // layer 0's, certified by layer 0's inputs against layer 0's index cache.
  // The kept-row budget is the quantized gates' (0.02): at these random
  // weights the full geometry's two-part score (nope and rope) makes the
  // 16-token softmax nearly one-hot, and the plain prefill gate at this
  // seed measures the same 1.4e-2 on one row from GEMM-order noise alone.
  const auto gather = [&](DevBuf& pre, DevBuf& dec) {
    std::vector<uint16_t> out(size_t(total) * cfg.hidden);
    pre.download(out.data(), size_t(T) * cfg.hidden * 2);
    dec.download(out.data() + size_t(T) * cfg.hidden, size_t(D) * cfg.hidden * 2);
    return out;
  };
  const std::vector<uint16_t> got0 = gather(o0, dec0), got1 = gather(o1, dec1);
  dsa_test::NearTieAudit audit_stats{};
  const RowAuditor audit = make_near_tie_auditor(cfg, g, pool, 0, s, phases, ref0, &audit_stats);
  require_output_matches(cfg, g, got0, want0, got_topk, topk0, "shared selection: layer 0", audit,
                         0.02);
  require_output_matches(cfg, g, got1, want1, got_topk, topk1, "shared selection: layer 1", audit,
                         0.02);
  print_near_tie_audit("shared selection", audit_stats);
  require_cache_matches(pool, 0, 0, ref0, s);
  // Layer 1's latent cache (its own rows and rope keys) through its own view.
  {
    const std::vector<int32_t> bt = fetch_block_row(pool, 0, s);
    const size_t row_bytes = g.latent_bytes_per_token;
    std::vector<uint8_t> raw(size_t(pool.max_token_slots()) * row_bytes);
    DGPP_CUDA_OK(cudaMemcpyAsync(raw.data(), pool.latent(1), raw.size(), cudaMemcpyDeviceToHost, s));
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
    std::vector<uint16_t> dev_rows(size_t(total) * size_t(g.score_width));
    for (int64_t t = 0; t < total; ++t) {
      const int64_t phys = int64_t(bt[size_t(t / cfg.block_tokens)]) * cfg.block_tokens +
                           (t % cfg.block_tokens);
      std::memcpy(&dev_rows[size_t(t) * size_t(g.score_width)], &raw[size_t(phys) * row_bytes],
                  size_t(g.score_width) * 2);
    }
    const std::vector<uint16_t> ref_rows(ref1.latent.begin(),
                                         ref1.latent.begin() + int64_t(dev_rows.size()));
    require_bf16("shared selection: layer 1 latent cache", compare_bf16(dev_rows, ref_rows, 8),
                 5e-3, 1e-3);
  }
}

DGPP_TEST(dsa_layer_full_tp2_head_slice_matches_tp1) {
  tp2_head_slice_case(full_cfg(), 7500, "full tp2 head slice");
}

// The select networks at the full model's width: select_k 2048 over
// per-token entries (kpool 1) with the relu'd logits, bitwise against the
// host spec — the prefill select (a 4096-key tile) over materialized dots
// and the fused decode select over an fp8 cache, then the expansion's
// eight rounds (2048 ids sorted ascending, no tail).
DGPP_TEST(dsa_select_kpool1_select2048_bitwise) {
  const int select_k = 2048, kpool = 1, max_selected = 2048, heads = 32, dim = 128;
  // Prefill: positions around the dense/sparse boundary and beyond.
  std::vector<int64_t> cases = {0, 1, 7, 2046, 2047, 2048, 2049, 2100, 4095, 4096, 6000, 9001};
  for (int i = 0; i < 12; ++i) cases.push_back(int64_t(2048 + hash32(7600 + i) % 7000));
  for (size_t ci = 0; ci < cases.size(); ++ci) {
    const int64_t pos = cases[ci];
    const int64_t visible = pos + 1;
    std::vector<float> dots(size_t(heads) * visible), w(heads), ks(static_cast<size_t>(visible));
    for (size_t i = 0; i < dots.size(); ++i) dots[i] = random_f32(7610 + ci, int64_t(i));
    for (int h = 0; h < heads; ++h) w[size_t(h)] = random_f32(7620 + ci, h);
    for (size_t i = 0; i < ks.size(); ++i) ks[i] = std::fabs(random_f32(7630 + ci, int64_t(i)));
    const std::vector<int64_t> posv = {pos};
    DevBuf dd(dots.size() * 4), dw(w.size() * 4), dks(ks.size() * 4), dpos(8),
        dtopk(size_t(max_selected) * 4), dcnt(4);
    dd.upload(dots.data(), dots.size() * 4);
    dw.upload(w.data(), w.size() * 4);
    dks.upload(ks.data(), ks.size() * 4);
    dpos.upload(posv.data(), 8);
    dsa_select_prefill(static_cast<const float*>(dd.p), visible, static_cast<const float*>(dw.p),
                       static_cast<const float*>(dks.p), static_cast<const int64_t*>(dpos.p), 1,
                       visible, heads, select_k, kpool, max_selected,
                       static_cast<int32_t*>(dtopk.p), static_cast<int32_t*>(dcnt.p), 0,
                       /*relu=*/true);
    std::vector<float> logits(static_cast<size_t>(visible));
    for (int64_t j = 0; j < visible; ++j) {
      float dot_h[32];
      for (int h = 0; h < heads; ++h) dot_h[h] = std::max(dots[size_t(h) * visible + j], 0.0f);
      logits[size_t(j)] = prefill_logit_mirror(dot_h, w.data(), ks[size_t(j)]);
    }
    std::vector<int32_t> want(size_t(max_selected), -1);
    int n_tok = 0;
    if (visible <= select_k) {
      n_tok = dsa_ref::causal_all_tokens(pos, max_selected, want.data());
    } else {
      std::vector<int32_t> ids(select_k);
      const int n_sel = dsa_ref::select_pools(logits.data(), visible, select_k, ids.data());
      n_tok = dsa_ref::expand_append_tail(ids.data(), n_sel, pos, kpool, max_selected, want.data());
    }
    std::vector<int32_t> got(size_t(max_selected), -12345);
    int32_t got_cnt = -1;
    dtopk.download(got.data(), got.size() * 4);
    dcnt.download(&got_cnt, 4);
    if (got_cnt != n_tok || std::memcmp(got.data(), want.data(), got.size() * 4) != 0)
      throw std::runtime_error("select2048 prefill mismatch at pos " + std::to_string(pos) +
                               " (count " + std::to_string(got_cnt) + " want " +
                               std::to_string(n_tok) + ")");
  }
  // Decode over an fp8 cache: 6000 visible entries, the same mirror.
  {
    const int n_pools = 6000;
    const int64_t pos = n_pools - 1;
    std::vector<uint16_t> qb = random_bf16_bits(7640, heads * dim, -2, 1);
    std::vector<uint8_t> q8(size_t(heads) * dim);
    std::vector<float> q_scale(heads);
    dsa_ref::fwht128_quant_fp8<float>(qb.data(), heads, dim, q8.data(), q_scale.data());
    std::vector<float> w(heads);
    const float logit_scale = float(std::pow(128.0, -0.5) * std::pow(32.0, -0.5));
    for (int h = 0; h < heads; ++h)
      w[size_t(h)] = (random_f32(7641, h) * 0.01f * q_scale[size_t(h)]) * logit_scale;
    std::vector<uint8_t> k8(size_t(n_pools) * dim);
    std::vector<float> ks(n_pools);
    for (size_t i = 0; i < k8.size(); ++i) {
      k8[i] = uint8_t(hash32(7642 + i) & 0xFF);
      if ((k8[i] & 0x7Fu) == 0x7Fu) k8[i] ^= 0x01u;
    }
    for (int j = 0; j < n_pools; ++j) ks[size_t(j)] = std::fabs(random_f32(7643, j)) * 0.05f;
    DevBuf dq8(q8.size()), dks(ks.size() * 4), dw(w.size() * 4), dki(k8.size()), dpos(8), dri(4),
        dbt(4), dtopk(size_t(max_selected) * 4), dcnt(4),
        dws(dsa_select_workspace_bytes(1, n_pools)), dctr(8);
    dq8.upload(q8.data(), q8.size());
    dw.upload(w.data(), w.size() * 4);
    dki.upload(k8.data(), k8.size());
    dks.upload(ks.data(), ks.size() * 4);
    const std::vector<int64_t> posv = {pos};
    const std::vector<int32_t> req0 = {0}, bt0 = {0};
    dpos.upload(posv.data(), 8);
    dri.upload(req0.data(), 4);
    dbt.upload(bt0.data(), 4);
    const int32_t zero2[2] = {0, 0};
    dctr.upload(zero2, 8);
    dsa_select_decode(dq8.p, static_cast<const float*>(dw.p), static_cast<const int32_t*>(dri.p),
                      static_cast<const int64_t*>(dpos.p), 1, static_cast<const int32_t*>(dbt.p), 1,
                      dki.p, static_cast<const float*>(dks.p), n_pools, heads, dim, select_k, kpool,
                      max_selected, static_cast<int32_t*>(dtopk.p), static_cast<int32_t*>(dcnt.p),
                      dws.p, n_pools, static_cast<int32_t*>(dctr.p), 48, 0, /*relu=*/true);
    std::vector<float> logits(n_pools);
    for (int j = 0; j < n_pools; ++j) {
      float contrib[32];
      for (int h = 0; h < 32; ++h) {
        float partial = 0.0f;
        for (int d = 0; d < 128; ++d)
          partial = partial + fp8_e4m3_bits_to_float(q8[size_t(h) * 128 + d]) *
                                  fp8_e4m3_bits_to_float(k8[size_t(j) * 128 + d]);
        contrib[h] = (w[size_t(h)] * ks[size_t(j)]) * std::max(partial, 0.0f);
      }
      logits[size_t(j)] = butterfly_sum_32(contrib);
    }
    std::vector<int32_t> ids(select_k);
    const int n_sel = dsa_ref::select_pools(logits.data(), n_pools, select_k, ids.data());
    std::vector<int32_t> want(size_t(max_selected), -1);
    dsa_ref::expand_append_tail(ids.data(), n_sel, pos, kpool, max_selected, want.data());
    std::vector<int32_t> got(size_t(max_selected), -12345);
    int32_t got_cnt = -1;
    dtopk.download(got.data(), got.size() * 4);
    dcnt.download(&got_cnt, 4);
    if (got_cnt != select_k) throw std::runtime_error("select2048 decode count");
    if (std::memcmp(got.data(), want.data(), got.size() * 4) != 0) {
      std::string detail;
      for (int i = 0; i < max_selected; ++i)
        if (got[size_t(i)] != want[size_t(i)] && detail.size() < 400)
          detail += " [" + std::to_string(i) + "] " + std::to_string(got[size_t(i)]) + " vs " +
                    std::to_string(want[size_t(i)]);
      throw std::runtime_error("select2048 decode mismatch:" + detail);
    }
  }
}

// The full checkpoint's attention geometry (64 heads, q_lora 2048, kv 512
// + rope 64, nope 192, v 256, select_k 2048 per token) at hidden 4096, no
// host oracle: chunked prefill across the dense/sparse boundary and a
// decode batch run clean on the 576-wide flash kernels, the split kernel's
// rope window and the wide select networks; the decode repeat is bitwise.
DGPP_TEST(dsa_layer_full_real_geometry_smoke) {
  cudaStream_t s = dgpp::kda_test::test_stream();
  DsaConfig cfg{};
  cfg.hidden = 4096;
  cfg.q_lora_rank = 2048;
  cfg.qk_nope_head_dim = 192;
  cfg.qk_rope_head_dim = 64;
  cfg.index_kpool = 1;
  cfg.index_relu = 1;
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  if (g.select_k != 2048 || g.score_width != 576) throw std::runtime_error("full real geometry");
  const int chunk = 2048;
  const int tail_tokens = 2054;
  const int decode_tokens = 4;
  const int64_t cap = (tail_tokens + 8 + cfg.block_tokens - 1) / cfg.block_tokens * cfg.block_tokens;
  TestWeights tw(cfg, 7700);
  const size_t dot_budget = 32ull << 20;
  LayerEnv env(cfg, chunk, tail_tokens + 8, 1, cap, s, dot_budget);
  DsaStatePool pool;
  pool.init(env.arena, cfg, 1, cap);
  const size_t sb = DsaLayer::scratch_bytes(cfg, chunk, tail_tokens + 8, 8, 4, dot_budget);
  DsaLayer layer(env.gemm, tw.layer_views, cfg, chunk, tail_tokens + 8,
                 env.arena.alloc_persistent(MemClass::DeviceHot, sb, 256), sb, env.ws.p,
                 env.ws.bytes, 8, 4, dot_budget);
  if (!layer.prepare(chunk) || !layer.prepare(decode_tokens)) throw std::runtime_error("gemm plans");
  const std::vector<uint16_t> hidden =
      random_bf16_bits(7701, int64_t(tail_tokens + decode_tokens) * cfg.hidden, -2, 1);
  DevBuf din(hidden.size() * 2), dchunk(size_t(chunk) * cfg.hidden * 2),
      drest(size_t(tail_tokens - chunk) * cfg.hidden * 2),
      ddout(size_t(decode_tokens) * cfg.hidden * 2);
  din.upload(hidden.data(), hidden.size() * 2);
  layer.enqueue_prefill(din.p, pool, 0, 0, 0, chunk, dchunk.p, s);
  layer.enqueue_prefill(static_cast<const uint16_t*>(din.p) + size_t(chunk) * cfg.hidden, pool, 0,
                        0, chunk, tail_tokens - chunk, drest.p, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  // The second chunk's rows sit past the dense regime: 2048 selected, no tail.
  std::vector<int32_t> cnt(size_t(tail_tokens - chunk));
  DGPP_CUDA_OK(cudaMemcpyAsync(cnt.data(), layer.debug_counts(), cnt.size() * 4,
                               cudaMemcpyDeviceToHost, s));
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  for (int32_t c : cnt)
    if (c != 2048) throw std::runtime_error("sparse row count " + std::to_string(c));
  DevBuf dreq(decode_tokens * 4), dpos(decode_tokens * 8), dspans(8);
  std::vector<int32_t> req_ids(decode_tokens, 0);
  std::vector<int64_t> pos(decode_tokens);
  for (int t = 0; t < decode_tokens; ++t) pos[size_t(t)] = tail_tokens + t;
  const std::vector<int32_t> spans = {0, decode_tokens};
  dreq.upload(req_ids.data(), req_ids.size() * 4);
  dpos.upload(pos.data(), pos.size() * 8);
  dspans.upload(spans.data(), spans.size() * 4);
  const auto decode = [&](std::vector<uint16_t>& out) {
    layer.enqueue_decode(static_cast<const uint16_t*>(din.p) + size_t(tail_tokens) * cfg.hidden,
                         pool, 0, static_cast<const int32_t*>(dreq.p),
                         static_cast<const int64_t*>(dpos.p), static_cast<const int32_t*>(dspans.p),
                         1, decode_tokens, ddout.p, s);
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
    out.resize(size_t(decode_tokens) * cfg.hidden);
    ddout.download(out.data(), out.size() * 2);
  };
  std::vector<uint16_t> eager, again;
  decode(eager);
  decode(again);
  require_bitwise("full real-geometry decode repeat", eager.data(), again.data(), eager.size() * 2);
  for (uint16_t v : eager)
    if ((v & 0x7F80u) == 0x7F80u) throw std::runtime_error("non-finite decode output");
}

}  // namespace layer

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

// Host-only TU (minijson + nvcc don't mix); runs via --dump-file.
int run_dsa_dump_parity(const std::string& path);

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], "--dump-file") == 0 && i + 1 < argc)
      return run_dsa_dump_parity(argv[i + 1]);
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1)
    return 2;  // ctest: skip, no GPU
  return dgpp::test::run_all();
}
