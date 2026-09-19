// The on-device greedy pick (kernels/glm_pick.hpp) and the step's device
// commit (kernels/glm_spec.hpp) against their host oracles: glm_pick_local's top-2 vs sample::local_max plus the gen
// log's runner-up scan; glm_pick_verdict's merge vs sample::merge_greedy
// and its judge vs judge_verify, over a SIMULATED world (each rank's table
// produced by the kernel on its slice, the fold emulated as an exact host
// sum — which is what the bus's SUM over disjoint slots is); the digest
// group's agreement/mismatch detection; the wire table's layout; and the
// fixed-batch MTP control/cache helpers.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/pick.hpp"
#include "kernels/sample_pick.hpp"
#include "kernels/glm_spec.hpp"
#include "sample/sampler.hpp"
#include "models/glm/speculative.hpp"

namespace {

using dgpp::PickLocal;
using dgpp::PickVerdict;
using dgpp::kPickIdDigits;
using dgpp::kPickLogitDigits;
using dgpp::kPickSlotsPerRank;
using dgpp::sample::Candidate;

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error(what);
}

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed | 1) {}
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
};

// Logits with deliberate structure: values quantized to a handful of
// levels (ties everywhere — the canonical tie-break is what's under test),
// a sprinkling of -inf, and a few distinct peaks.
std::vector<float> make_logits(Rng& rng, size_t n) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) {
    const uint64_t r = rng.next();
    if ((r & 63) == 0) {
      v[i] = -INFINITY;
    } else {
      v[i] = static_cast<float>((r >> 8) % 17) * 0.5f - 4.0f;
    }
  }
  return v;
}

float host_runner_up(const float* slice, int n, int vocab_begin,
                     int32_t best_id) {
  float second = -INFINITY;
  for (int i = 0; i < n; ++i)
    if (vocab_begin + i != best_id && slice[i] > second) second = slice[i];
  return second;
}

uint64_t decode_digits(const uint16_t* slots, int digits) {
  uint64_t value = 0;
  for (int d = 0; d < digits; ++d)
    value |= static_cast<uint64_t>(slots[d] & 63) << (6 * d);
  return value;
}

template <typename T>
T* device_alloc(size_t n) {
  T* p = nullptr;
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)));
  return p;
}

// One rank's local pick on `logits` [rows, count]; returns its table
// (host copy) and locals.
struct LocalRun {
  std::vector<uint16_t> table;
  std::vector<PickLocal> locals;
};

LocalRun run_local(const std::vector<float>& logits, int rows, int count,
                   int vocab_begin, int rank, int world, uint64_t carry) {
  const size_t table_elems = dgpp::device_pick_table_elems(rows, world);
  float* d_logits = device_alloc<float>(logits.size());
  uint16_t* d_table = device_alloc<uint16_t>(table_elems);
  uint64_t* d_carry = device_alloc<uint64_t>(1);
  PickLocal* d_locals = device_alloc<PickLocal>(static_cast<size_t>(rows));
  DGPP_CUDA_OK(cudaMemcpy(d_logits, logits.data(), logits.size() * 4,
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_carry, &carry, 8, cudaMemcpyHostToDevice));
  // Poison the table: the kernel must zero every slot it does not write.
  DGPP_CUDA_OK(cudaMemset(d_table, 0xff, table_elems * 2));
  dgpp::device_pick_local(d_logits, rows, count, vocab_begin, rank, world,
                       d_carry, d_table, d_locals, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  LocalRun out;
  out.table.resize(table_elems);
  out.locals.resize(static_cast<size_t>(rows));
  DGPP_CUDA_OK(cudaMemcpy(out.table.data(), d_table, table_elems * 2,
                          cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(out.locals.data(), d_locals,
                          sizeof(PickLocal) * rows, cudaMemcpyDeviceToHost));
  cudaFree(d_logits);
  cudaFree(d_table);
  cudaFree(d_carry);
  cudaFree(d_locals);
  return out;
}

PickVerdict run_verdict(const std::vector<uint16_t>& table, int rows,
                           int world, int rank,
                           const std::vector<int64_t>& fed,
                           uint64_t* carry_out) {
  uint16_t* d_table = device_alloc<uint16_t>(table.size());
  int64_t* d_fed = device_alloc<int64_t>(fed.size());
  uint64_t* d_carry = device_alloc<uint64_t>(1);
  PickVerdict* d_verdict = device_alloc<PickVerdict>(1);
  DGPP_CUDA_OK(cudaMemcpy(d_table, table.data(), table.size() * 2,
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_fed, fed.data(), fed.size() * 8,
                          cudaMemcpyHostToDevice));
  dgpp::device_pick_verdict(d_table, rows, world, rank, d_fed, d_verdict,
                         /*device_verdict=*/nullptr, d_carry, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  PickVerdict v;
  DGPP_CUDA_OK(cudaMemcpy(&v, d_verdict, sizeof(v), cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(carry_out, d_carry, 8, cudaMemcpyDeviceToHost));
  cudaFree(d_table);
  cudaFree(d_fed);
  cudaFree(d_carry);
  cudaFree(d_verdict);
  return v;
}

std::vector<PickVerdict> run_verdict_batch(
    const std::vector<uint16_t>& table, int rows, int world, int rank,
    const std::vector<int64_t>& fed, const std::vector<int64_t>& positions,
    int requests, int rows_per_request, uint64_t* carry_out) {
  uint16_t* d_table = device_alloc<uint16_t>(table.size());
  int64_t* d_fed = device_alloc<int64_t>(fed.size());
  int64_t* d_positions = device_alloc<int64_t>(positions.size());
  uint64_t* d_carry = device_alloc<uint64_t>(1);
  PickVerdict* d_verdict =
      device_alloc<PickVerdict>(static_cast<size_t>(requests));
  DGPP_CUDA_OK(cudaMemcpy(d_table, table.data(), table.size() * 2,
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_fed, fed.data(), fed.size() * 8,
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_positions, positions.data(), positions.size() * 8,
                          cudaMemcpyHostToDevice));
  dgpp::device_pick_verdict_batched(
      d_table, rows, world, rank, d_fed, d_positions, requests,
      rows_per_request, rows_per_request, d_verdict,
      /*device_verdicts=*/nullptr, d_carry, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<PickVerdict> out(static_cast<size_t>(requests));
  DGPP_CUDA_OK(cudaMemcpy(out.data(), d_verdict,
                          out.size() * sizeof(PickVerdict),
                          cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(carry_out, d_carry, 8, cudaMemcpyDeviceToHost));
  cudaFree(d_table);
  cudaFree(d_fed);
  cudaFree(d_positions);
  cudaFree(d_carry);
  cudaFree(d_verdict);
  return out;
}

}  // namespace

DGPP_TEST(pick_local_top2_matches_host_local_max_and_runner_up) {
  Rng rng(0x5151);
  struct Shape {
    int rows, count, vocab_begin, rank, world;
  };
  const Shape shapes[] = {{1, 1, 0, 0, 1},        {1, 37, 0, 0, 1},
                          {2, 38720, 38720, 1, 4}, {4, 1000, 3000, 3, 4},
                          {3, 2049, 0, 0, 2},      {1, 154880, 0, 0, 1}};
  for (const Shape& s : shapes) {
    const std::vector<float> logits =
        make_logits(rng, static_cast<size_t>(s.rows) * s.count);
    const uint64_t carry = 0x2a5a5a5a5a5a5ull;  // 54 bits
    const LocalRun got =
        run_local(logits, s.rows, s.count, s.vocab_begin, s.rank, s.world, carry);
    for (int r = 0; r < s.rows; ++r) {
      const float* slice = logits.data() + static_cast<size_t>(r) * s.count;
      const Candidate want =
          dgpp::sample::local_max(slice, s.count, s.vocab_begin);
      const PickLocal& l = got.locals[static_cast<size_t>(r)];
      require(l.best_id == want.id,
              "row " + std::to_string(r) + ": best id " +
                  std::to_string(l.best_id) + " != " + std::to_string(want.id));
      require(std::memcmp(&l.best_logit, &want.logit, 4) == 0,
              "row " + std::to_string(r) + ": best logit bits differ");
      const float second = host_runner_up(slice, s.count, s.vocab_begin, want.id);
      require(std::memcmp(&l.second_logit, &second, 4) == 0,
              "row " + std::to_string(r) + ": runner-up " +
                  std::to_string(l.second_logit) + " != " +
                  std::to_string(second));
      // The wire slots: (logit bits, id) at (row, rank); zero elsewhere.
      for (int k = 0; k < s.world; ++k) {
        const uint16_t* slot =
            got.table.data() +
            (static_cast<size_t>(r) * s.world + k) * kPickSlotsPerRank;
        if (k != s.rank) {
          for (int d = 0; d < kPickSlotsPerRank; ++d)
            require(slot[d] == 0, "foreign slot not zeroed");
          continue;
        }
        uint32_t bits = 0;
        std::memcpy(&bits, &want.logit, 4);
        require(decode_digits(slot, kPickLogitDigits) == bits,
                "logit digits do not round-trip");
        require(decode_digits(slot + kPickLogitDigits, kPickIdDigits) ==
                    static_cast<uint64_t>(want.id),
                "id digits do not round-trip");
      }
    }
    // The digest group: this rank's carry, zero for the others.
    const uint16_t* digests =
        got.table.data() +
        static_cast<size_t>(s.rows) * s.world * kPickSlotsPerRank;
    for (int k = 0; k < s.world; ++k) {
      const uint64_t d =
          decode_digits(digests + k * kPickSlotsPerRank, kPickSlotsPerRank);
      require(d == (k == s.rank ? carry : 0), "digest group slot wrong");
    }
  }
}

DGPP_TEST(pick_verdict_over_simulated_world_matches_merge_greedy_and_judge) {
  Rng rng(0x7e57);
  constexpr int kWorld = 4;
  constexpr int kCount = 96;  // the fixture vocab / world
  for (int rows = 1; rows <= dgpp::kPickMaxRows; ++rows) {
    for (int trial = 0; trial < 6; ++trial) {
      // The full [rows, vocab] logits, sliced per rank.
      const std::vector<float> full =
          make_logits(rng, static_cast<size_t>(rows) * kWorld * kCount);
      std::vector<std::vector<Candidate>> per_row_locals(
          static_cast<size_t>(rows));
      std::vector<uint16_t> folded(dgpp::device_pick_table_elems(rows, kWorld), 0);
      const uint64_t carry = 0x123456789abcdull;
      for (int k = 0; k < kWorld; ++k) {
        std::vector<float> slice(static_cast<size_t>(rows) * kCount);
        for (int r = 0; r < rows; ++r)
          std::memcpy(slice.data() + static_cast<size_t>(r) * kCount,
                      full.data() +
                          (static_cast<size_t>(r) * kWorld + k) * kCount,
                      kCount * 4);
        const LocalRun local =
            run_local(slice, rows, kCount, k * kCount, k, kWorld, carry);
        for (int r = 0; r < rows; ++r)
          per_row_locals[static_cast<size_t>(r)].push_back(
              dgpp::sample::local_max(
                  slice.data() + static_cast<size_t>(r) * kCount, kCount,
                  k * kCount));
        // The fold: an exact SUM over disjoint slots (bf16 small ints).
        for (size_t i = 0; i < folded.size(); ++i) {
          require(folded[i] == 0 || local.table[i] == 0,
                  "two ranks wrote the same wire slot");
          folded[i] = static_cast<uint16_t>(folded[i] + local.table[i]);
        }
      }
      std::vector<int32_t> want_winners;
      for (int r = 0; r < rows; ++r)
        want_winners.push_back(dgpp::sample::merge_greedy(
            per_row_locals[static_cast<size_t>(r)]));
      // Fed tokens: alternate trials feed the true next tokens (accept
      // all) and random ones (reject somewhere).
      std::vector<int64_t> fed(static_cast<size_t>(rows));
      fed[0] = static_cast<int64_t>(rng.next() % (kWorld * kCount));
      for (int r = 1; r < rows; ++r)
        fed[static_cast<size_t>(r)] =
            (trial % 2 == 0 || (rng.next() & 1))
                ? want_winners[static_cast<size_t>(r - 1)]
                : static_cast<int64_t>(rng.next() % (kWorld * kCount));
      const dgpp::SpecVerdict want = dgpp::judge_verify(fed, want_winners);

      for (int rank = 0; rank < kWorld; ++rank) {
        uint64_t carry_out = 0;
        const PickVerdict got =
            run_verdict(folded, rows, kWorld, rank, fed, &carry_out);
        require(got.rows == rows, "verdict rows");
        for (int r = 0; r < rows; ++r)
          require(got.winners[r] == want_winners[static_cast<size_t>(r)],
                  "rows " + std::to_string(rows) + " row " +
                      std::to_string(r) + ": winner " +
                      std::to_string(got.winners[r]) + " != merge_greedy's " +
                      std::to_string(want_winners[static_cast<size_t>(r)]));
        require(got.accepted == want.accepted,
                "accepted " + std::to_string(got.accepted) + " != " +
                    std::to_string(want.accepted));
        require(got.next == want.next, "next differs from judge_verify");
        require(got.digest_mismatch == 0, "identical carries flagged");
        require(got.digest ==
                    dgpp::device_pick_digest(rows, got.accepted, got.winners),
                "digest != host digest");
        require(carry_out == got.digest, "carry not updated to the digest");
        for (int k = 0; k < kWorld; ++k)
          require(got.peer_digests[k] == carry, "peer digest decoded wrong");
      }
    }
  }
}

DGPP_TEST(pick_verdict_flags_the_rank_whose_carried_digest_differs) {
  Rng rng(0xd16e);
  constexpr int kWorld = 4;
  constexpr int rows = 2;
  constexpr int kCount = 96;
  std::vector<uint16_t> folded(dgpp::device_pick_table_elems(rows, kWorld), 0);
  for (int k = 0; k < kWorld; ++k) {
    const std::vector<float> slice =
        make_logits(rng, static_cast<size_t>(rows) * kCount);
    // Rank 2 carries a different digest: it computed a different verdict
    // last step (its table was corrupt).
    const uint64_t carry = k == 2 ? 0xbadull : 0x600dull;
    const LocalRun local =
        run_local(slice, rows, kCount, k * kCount, k, kWorld, carry);
    for (size_t i = 0; i < folded.size(); ++i)
      folded[i] = static_cast<uint16_t>(folded[i] + local.table[i]);
  }
  const std::vector<int64_t> fed{1, 2};
  uint64_t carry_out = 0;
  const PickVerdict from_rank0 =
      run_verdict(folded, rows, kWorld, 0, fed, &carry_out);
  require(from_rank0.digest_mismatch == (1u << 2),
          "rank 0 must see exactly rank 2 disagreeing, mask " +
              std::to_string(from_rank0.digest_mismatch));
  const PickVerdict from_rank2 =
      run_verdict(folded, rows, kWorld, 2, fed, &carry_out);
  require(from_rank2.digest_mismatch == 0b1011u,
          "rank 2 must see every peer disagreeing with it");
  require(from_rank0.peer_digests[2] == 0xbadull &&
              from_rank0.peer_digests[1] == 0x600dull,
          "peer digests must decode to what each rank carried");
}

DGPP_TEST(pick_batch_judges_each_request_and_skips_padding) {
  constexpr int requests = 3;
  constexpr int per = 2;
  constexpr int rows = requests * per;
  constexpr int count = 32;
  std::vector<float> logits(static_cast<size_t>(rows) * count, -10.0f);
  const int32_t winners[rows] = {3, 5, 7, 9, 11, 13};
  for (int r = 0; r < rows; ++r)
    logits[static_cast<size_t>(r) * count + winners[r]] = 10.0f + r;
  const LocalRun local =
      run_local(logits, rows, count, /*vocab_begin=*/0, /*rank=*/0,
                /*world=*/1, /*carry=*/0x1234);
  std::vector<int64_t> fed = {17, winners[0], 19, winners[2], 23, 24};
  const std::vector<int64_t> positions = {100, 101, -1, -1, 300, 301};
  uint64_t carry = 0;
  const std::vector<PickVerdict> got = run_verdict_batch(
      local.table, rows, /*world=*/1, /*rank=*/0, fed, positions, requests,
      per, &carry);
  require(got[0].rows == 2 && got[0].accepted == 2 &&
              got[0].next == winners[1],
          "request 0 must accept both rows");
  require(got[1].rows == 0 && got[1].accepted == 0 && got[1].next == -1,
          "request 1 must be an inactive padding verdict");
  require(got[2].rows == 2 && got[2].accepted == 1 &&
              got[2].next == winners[4],
          "request 2 must reject its draft row");
  for (const PickVerdict& v : got)
    require(v.digest == carry && v.digest_mismatch == 0,
            "every request must carry the physical pass digest");
  require((carry >> dgpp::kPickDigestBits) == 0,
          "batched digest must fit the wire's carried digit group");
}

DGPP_TEST(pick_batch_draft_selects_last_accepted_row_per_request) {
  constexpr int requests = 3;
  constexpr int stride = 2;
  constexpr int count = 24;
  std::vector<float> logits(static_cast<size_t>(requests * stride) * count,
                            -20.0f);
  const int32_t winners[requests * stride] = {2, 3, 5, 7, 11, 13};
  for (int r = 0; r < requests * stride; ++r)
    logits[static_cast<size_t>(r) * count + winners[r]] = 20.0f;
  float* d_logits = device_alloc<float>(logits.size());
  uint16_t* d_table = device_alloc<uint16_t>(
      dgpp::device_pick_table_elems(requests, /*world=*/1));
  uint64_t* d_carry = device_alloc<uint64_t>(1);
  PickLocal* d_locals = device_alloc<PickLocal>(requests);
  PickVerdict select[requests];
  select[0].accepted = 1;
  select[1].accepted = 0;  // inactive: harmless first padding row
  select[2].accepted = 2;
  PickVerdict* d_select = device_alloc<PickVerdict>(requests);
  DGPP_CUDA_OK(cudaMemcpy(d_logits, logits.data(), logits.size() * 4,
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemset(d_carry, 0, sizeof(uint64_t)));
  DGPP_CUDA_OK(cudaMemcpy(d_select, select, sizeof(select),
                          cudaMemcpyHostToDevice));
  dgpp::device_pick_local_batched(
      d_logits, requests, count, /*vocab_begin=*/0, /*rank=*/0, /*world=*/1,
      d_carry, d_table, d_locals, nullptr, d_select, requests, stride);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  PickLocal got[requests];
  DGPP_CUDA_OK(cudaMemcpy(got, d_locals, sizeof(got), cudaMemcpyDeviceToHost));
  require(got[0].best_id == winners[0], "request 0 selected wrong draft row");
  require(got[1].best_id == winners[2], "inactive request must select row 0");
  require(got[2].best_id == winners[5], "request 2 selected wrong draft row");
  cudaFree(d_logits);
  cudaFree(d_table);
  cudaFree(d_carry);
  cudaFree(d_locals);
  cudaFree(d_select);
}

// The commit: with rows = 3 and a verdict accepting a rows, every segment's
// live state must become snapshot row a-1 when a < 3 and stay untouched
// when a == 3; the position advances by a either way; the positions kernel
// derives the rows' positions from the device position.
DGPP_TEST(spec_positions_rows64_fills_upper_half_and_rejects_overflow) {
  const int64_t start = 1234;
  auto* pos = device_alloc<int64_t>(1);
  auto* out = device_alloc<int64_t>(64);
  DGPP_CUDA_OK(cudaMemcpy(pos, &start, sizeof(start), cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemset(out, 0xff, 64 * sizeof(int64_t)));
  dgpp::glm_spec_positions(pos, 64, out, nullptr);
  std::vector<int64_t> got(64);
  DGPP_CUDA_OK(cudaMemcpy(got.data(), out, 64 * sizeof(int64_t), cudaMemcpyDeviceToHost));
  for (int r = 0; r < 64; ++r) require(got[r] == start + r, "64-row position coverage");
  bool rejected = false;
  try { dgpp::glm_spec_positions(pos, 65, out, nullptr); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "positions reject a row beyond capacity");
  PickVerdict verdict;
  for (int r = 0; r < 64; ++r) require(verdict.winners[r] == -1, "winner sentinel in every row");
  std::vector<int64_t> positions(16);
  std::vector<int32_t> ids(64);
  for (int q = 0; q < 16; ++q) positions[q] = q == 11 ? 0 : 100 + q * 10;
  for (int r = 0; r < 64; ++r) ids[r] = r / 4;
  auto* dp = device_alloc<int64_t>(16);
  auto* di = device_alloc<int32_t>(64);
  DGPP_CUDA_OK(cudaMemcpy(dp, positions.data(), 16 * sizeof(int64_t), cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(di, ids.data(), 64 * sizeof(int32_t), cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemset(out, 0, 64 * sizeof(int64_t)));
  dgpp::glm_spec_positions_batched(dp, di, 64, 4, out, nullptr);
  DGPP_CUDA_OK(cudaMemcpy(got.data(), out, 64 * sizeof(int64_t), cudaMemcpyDeviceToHost));
  for (int r = 0; r < 64; ++r)
    require(got[r] == (positions[r / 4] ? positions[r / 4] + r % 4 : -1),
            "batched position coverage, including padding above row 32");
  cudaFree(dp);
  cudaFree(di);
  cudaFree(out);
  cudaFree(pos);
}

DGPP_TEST(spec_commit_copies_snapshot_row_when_rejected_and_advances_position) {
  constexpr int rows = 3;
  constexpr size_t kBytesA = 4096, kBytesB = 64;  // two families, uneven
  std::vector<uint8_t> live_a(kBytesA, 0xaa), live_b(kBytesB, 0xbb);
  // Snapshot rows: row r of family A is filled with 0x10 + r, B with 0x20 + r.
  std::vector<uint8_t> snaps_a(kBytesA * (rows - 1)), snaps_b(kBytesB * (rows - 1));
  for (int r = 0; r < rows - 1; ++r) {
    std::fill_n(snaps_a.data() + r * kBytesA, kBytesA, static_cast<uint8_t>(0x10 + r));
    std::fill_n(snaps_b.data() + r * kBytesB, kBytesB, static_cast<uint8_t>(0x20 + r));
  }
  uint8_t* d_live_a = device_alloc<uint8_t>(kBytesA);
  uint8_t* d_live_b = device_alloc<uint8_t>(kBytesB);
  uint8_t* d_snaps_a = device_alloc<uint8_t>(snaps_a.size());
  uint8_t* d_snaps_b = device_alloc<uint8_t>(snaps_b.size());
  PickVerdict* d_verdict = device_alloc<PickVerdict>(1);
  int64_t* d_pos = device_alloc<int64_t>(1);
  int64_t* d_step_pos = device_alloc<int64_t>(rows);
  DGPP_CUDA_OK(cudaMemcpy(d_snaps_a, snaps_a.data(), snaps_a.size(),
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_snaps_b, snaps_b.data(), snaps_b.size(),
                          cudaMemcpyHostToDevice));
  dgpp::GlmSpecSegments segs;
  segs.count = 2;
  segs.seg[0] = dgpp::GlmSpecSegment{d_live_a, d_snaps_a, kBytesA, kBytesA};
  segs.seg[1] = dgpp::GlmSpecSegment{d_live_b, d_snaps_b, kBytesB, kBytesB};

  for (int accepted = 1; accepted <= rows; ++accepted) {
    DGPP_CUDA_OK(cudaMemcpy(d_live_a, live_a.data(), kBytesA, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(d_live_b, live_b.data(), kBytesB, cudaMemcpyHostToDevice));
    PickVerdict v;
    v.rows = rows;
    v.accepted = accepted;
    DGPP_CUDA_OK(cudaMemcpy(d_verdict, &v, sizeof(v), cudaMemcpyHostToDevice));
    const int64_t pos0 = 1000;
    DGPP_CUDA_OK(cudaMemcpy(d_pos, &pos0, 8, cudaMemcpyHostToDevice));
    dgpp::glm_spec_commit(d_verdict, rows, segs, d_pos, nullptr);
    dgpp::glm_spec_positions(d_pos, rows, d_step_pos, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint8_t> got_a(kBytesA), got_b(kBytesB);
    int64_t pos = 0;
    std::vector<int64_t> step_pos(rows);
    DGPP_CUDA_OK(cudaMemcpy(got_a.data(), d_live_a, kBytesA, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(got_b.data(), d_live_b, kBytesB, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(&pos, d_pos, 8, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(step_pos.data(), d_step_pos, 8 * rows, cudaMemcpyDeviceToHost));
    require(pos == pos0 + accepted, "position must advance by accepted");
    for (int r = 0; r < rows; ++r)
      require(step_pos[static_cast<size_t>(r)] == pos + r,
              "step positions must follow the device position");
    const uint8_t want_a = accepted == rows ? 0xaa : static_cast<uint8_t>(0x10 + accepted - 1);
    const uint8_t want_b = accepted == rows ? 0xbb : static_cast<uint8_t>(0x20 + accepted - 1);
    for (size_t i = 0; i < kBytesA; ++i)
      require(got_a[i] == want_a, "family A: accepted " + std::to_string(accepted) +
                                       " byte " + std::to_string(i));
    for (size_t i = 0; i < kBytesB; ++i)
      require(got_b[i] == want_b, "family B: accepted " + std::to_string(accepted));
  }
  // A model with no stateful KDA/DSA family still uses the commit kernel to
  // advance its device position. A rejection must not index the empty
  // segment table.
  PickVerdict position_only;
  position_only.rows = rows;
  position_only.accepted = 1;
  DGPP_CUDA_OK(cudaMemcpy(d_verdict, &position_only, sizeof(position_only),
                          cudaMemcpyHostToDevice));
  const int64_t position_only_start = 77;
  DGPP_CUDA_OK(cudaMemcpy(d_pos, &position_only_start, sizeof(int64_t),
                          cudaMemcpyHostToDevice));
  dgpp::glm_spec_commit(d_verdict, rows, dgpp::GlmSpecSegments{}, d_pos,
                        nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  int64_t position_only_got = 0;
  DGPP_CUDA_OK(cudaMemcpy(&position_only_got, d_pos, sizeof(int64_t),
                          cudaMemcpyDeviceToHost));
  require(position_only_got == position_only_start + 1,
          "position-only commit did not advance exactly once");
  cudaFree(d_live_a);
  cudaFree(d_live_b);
  cudaFree(d_snaps_a);
  cudaFree(d_snaps_b);
  cudaFree(d_verdict);
  cudaFree(d_pos);
  cudaFree(d_step_pos);
}

DGPP_TEST(spec_batch_positions_draft_rows_and_token_feeds_are_slot_local) {
  constexpr int requests = 3;
  constexpr int per = 2;
  constexpr int rows = requests * per;
  const int64_t session_pos[requests] = {100, 0, 300};
  const int32_t req_ids[rows] = {0, 0, 1, 1, 2, 2};
  int64_t* d_session = device_alloc<int64_t>(requests);
  int32_t* d_req = device_alloc<int32_t>(rows);
  int64_t* d_pos = device_alloc<int64_t>(rows);
  int64_t* d_tokens = device_alloc<int64_t>(rows);
  int64_t* d_next = device_alloc<int64_t>(requests);
  int64_t* d_block = device_alloc<int64_t>(requests);
  PickVerdict verify[requests];
  verify[0].rows = 2;
  verify[0].accepted = 2;
  verify[0].next = 31;
  verify[0].winners[0] = 29;
  verify[0].winners[1] = 31;
  verify[1].rows = 0;
  verify[1].accepted = 0;
  verify[2].rows = 2;
  verify[2].accepted = 1;
  verify[2].next = 41;
  verify[2].winners[0] = 41;
  verify[2].winners[1] = 43;
  PickVerdict* d_verify = device_alloc<PickVerdict>(requests);
  DGPP_CUDA_OK(cudaMemcpy(d_session, session_pos, sizeof(session_pos),
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_req, req_ids, sizeof(req_ids),
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_verify, verify, sizeof(verify),
                          cudaMemcpyHostToDevice));
  dgpp::glm_spec_positions_batched(d_session, d_req, rows, per, d_pos,
                                   nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  int64_t pos[rows];
  DGPP_CUDA_OK(cudaMemcpy(pos, d_pos, sizeof(pos), cudaMemcpyDeviceToHost));
  const int64_t want_verify_pos[rows] = {100, 101, -1, -1, 300, 301};
  require(std::equal(std::begin(pos), std::end(pos),
                     std::begin(want_verify_pos)),
          "batched verify positions differ");

  const int64_t block_pos[requests] = {90, 190, 290};
  DGPP_CUDA_OK(cudaMemcpy(d_block, block_pos, sizeof(block_pos),
                          cudaMemcpyHostToDevice));
  dgpp::glm_spec_draft_rows_batched(d_verify, requests, per, d_block, d_pos,
                                    d_tokens, d_next, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  int64_t draft_pos[rows], draft_tokens[rows], next[requests], block[requests];
  DGPP_CUDA_OK(cudaMemcpy(draft_pos, d_pos, sizeof(draft_pos),
                          cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(draft_tokens, d_tokens, sizeof(draft_tokens),
                          cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(next, d_next, sizeof(next), cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(block, d_block, sizeof(block), cudaMemcpyDeviceToHost));
  const int64_t want_draft_pos[rows] = {90, 91, -1, -1, 290, -1};
  const int64_t want_draft_tokens[rows] = {29, 31, 0, 0, 41, 41};
  require(std::equal(std::begin(draft_pos), std::end(draft_pos),
                     std::begin(want_draft_pos)),
          "batched draft positions differ");
  require(std::equal(std::begin(draft_tokens), std::end(draft_tokens),
                     std::begin(want_draft_tokens)),
          "batched draft tokens differ");
  require(block[0] == 92 && block[1] == 190 && block[2] == 291 &&
              next[0] == 31 && next[2] == 41,
          "batched draft counters/next tokens are not slot-local");

  PickVerdict draft[requests];
  draft[0].rows = 1;
  draft[0].accepted = 1;
  draft[0].next = 37;
  draft[1].rows = 0;
  draft[1].accepted = 0;
  draft[2].rows = 1;
  draft[2].accepted = 1;
  draft[2].next = 47;
  PickVerdict* d_draft = device_alloc<PickVerdict>(requests);
  DGPP_CUDA_OK(cudaMemcpy(d_draft, draft, sizeof(draft),
                          cudaMemcpyHostToDevice));
  dgpp::glm_spec_next_tokens_batched(d_next, d_draft, requests, per,
                                     d_tokens, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  DGPP_CUDA_OK(cudaMemcpy(draft_tokens, d_tokens, sizeof(draft_tokens),
                          cudaMemcpyDeviceToHost));
  const int64_t want_next_tokens[rows] = {31, 37, 0, 0, 41, 47};
  require(std::equal(std::begin(draft_tokens), std::end(draft_tokens),
                     std::begin(want_next_tokens)),
          "batched next token feed differs");

  cudaFree(d_session);
  cudaFree(d_req);
  cudaFree(d_pos);
  cudaFree(d_tokens);
  cudaFree(d_next);
  cudaFree(d_block);
  cudaFree(d_verify);
  cudaFree(d_draft);
}

DGPP_TEST(mtp_batch_hidden_cache_input_and_scatter_are_request_indexed) {
  constexpr int requests = 3;
  constexpr int rows = 6;
  constexpr int hidden = 8;
  constexpr int vocab = 4;
  constexpr int cache_positions = 4;
  constexpr int64_t cache_stride = cache_positions * hidden;
  const int64_t tokens[rows] = {0, 1, 2, 3, 1, 0};
  const int32_t req_ids[rows] = {2, 2, 1, 1, 0, 0};
  const int64_t positions[rows] = {0, 1, -1, -1, 2, 3};

  std::vector<uint16_t> embed(vocab * hidden);
  std::vector<uint16_t> cache(requests * cache_stride);
  std::vector<uint16_t> enorm(hidden), hnorm(hidden);
  for (int v = 0; v < vocab; ++v)
    for (int h = 0; h < hidden; ++h)
      embed[static_cast<size_t>(v) * hidden + h] =
          dgpp::float_to_bf16_bits((v + 1) * 0.25f + (h + 1) * 0.03125f);
  for (int q = 0; q < requests; ++q)
    for (int p = 0; p < cache_positions; ++p)
      for (int h = 0; h < hidden; ++h)
        cache[static_cast<size_t>(q) * cache_stride + p * hidden + h] =
            dgpp::float_to_bf16_bits((q + 1) * 2.0f + p * 0.25f +
                                     (h + 1) * 0.015625f);
  for (int h = 0; h < hidden; ++h) {
    enorm[h] = dgpp::float_to_bf16_bits(0.5f + h * 0.0625f);
    hnorm[h] = dgpp::float_to_bf16_bits(1.0f + h * 0.03125f);
  }

  uint16_t* d_embed = device_alloc<uint16_t>(embed.size());
  uint16_t* d_cache = device_alloc<uint16_t>(cache.size());
  uint16_t* d_enorm = device_alloc<uint16_t>(enorm.size());
  uint16_t* d_hnorm = device_alloc<uint16_t>(hnorm.size());
  int64_t* d_tokens = device_alloc<int64_t>(rows);
  int32_t* d_req = device_alloc<int32_t>(rows);
  int64_t* d_pos = device_alloc<int64_t>(rows);
  uint16_t* d_batch = device_alloc<uint16_t>(rows * 2 * hidden);
  uint16_t* d_scalar = device_alloc<uint16_t>(rows * 2 * hidden);
  DGPP_CUDA_OK(cudaMemcpy(d_embed, embed.data(), embed.size() * 2,
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_cache, cache.data(), cache.size() * 2,
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_enorm, enorm.data(), enorm.size() * 2,
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_hnorm, hnorm.data(), hnorm.size() * 2,
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_tokens, tokens, sizeof(tokens),
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_req, req_ids, sizeof(req_ids),
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_pos, positions, sizeof(positions),
                          cudaMemcpyHostToDevice));

  dgpp::glm_mtp_input_bf16_batched(
      d_embed, d_tokens, d_cache, cache_stride, d_req, d_pos, d_enorm,
      d_hnorm, d_batch, rows, hidden, 1e-5f, nullptr);
  for (int r = 0; r < rows; ++r) {
    const uint16_t* request_cache =
        d_cache + static_cast<int64_t>(req_ids[r]) * cache_stride;
    dgpp::glm_mtp_input_bf16(
        d_embed, d_tokens + r, request_cache, d_pos + r, /*first_pos=*/0,
        d_enorm, d_hnorm, d_scalar + r * 2 * hidden, /*rows=*/1, hidden,
        1e-5f, nullptr);
  }
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> batch(rows * 2 * hidden);
  std::vector<uint16_t> scalar(rows * 2 * hidden);
  DGPP_CUDA_OK(cudaMemcpy(batch.data(), d_batch, batch.size() * 2,
                          cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(scalar.data(), d_scalar, scalar.size() * 2,
                          cudaMemcpyDeviceToHost));
  require(batch == scalar,
          "batched MTP input differs from request-based scalar rows");
  for (int r : {2, 3})
    require(std::all_of(batch.begin() + r * 2 * hidden,
                        batch.begin() + (r + 1) * 2 * hidden,
                        [](uint16_t v) { return v == 0; }),
            "negative-position MTP input row was not zero padded");

  std::vector<uint16_t> scatter_rows(rows * hidden);
  for (int r = 0; r < rows; ++r)
    for (int h = 0; h < hidden; ++h)
      scatter_rows[static_cast<size_t>(r) * hidden + h] =
          static_cast<uint16_t>(0x100 + r * hidden + h);
  std::vector<uint16_t> scatter_want(requests * cache_stride, 0x5a5a);
  for (int r = 0; r < rows; ++r) {
    if (positions[r] < 0) continue;
    std::copy_n(scatter_rows.begin() + r * hidden, hidden,
                scatter_want.begin() +
                    static_cast<int64_t>(req_ids[r]) * cache_stride +
                    positions[r] * hidden);
  }
  uint16_t* d_rows = device_alloc<uint16_t>(scatter_rows.size());
  uint16_t* d_scatter = device_alloc<uint16_t>(scatter_want.size());
  DGPP_CUDA_OK(cudaMemcpy(d_rows, scatter_rows.data(), scatter_rows.size() * 2,
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemset(d_scatter, 0x5a, scatter_want.size() * 2));
  dgpp::glm_rows_scatter_bf16_batched(
      d_rows, d_req, d_pos, d_scatter, cache_stride, rows, hidden, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> scatter_got(scatter_want.size());
  DGPP_CUDA_OK(cudaMemcpy(scatter_got.data(), d_scatter,
                          scatter_got.size() * 2, cudaMemcpyDeviceToHost));
  require(scatter_got == scatter_want,
          "batched hidden scatter crossed a request slot or wrote padding");

  cudaFree(d_embed);
  cudaFree(d_cache);
  cudaFree(d_enorm);
  cudaFree(d_hnorm);
  cudaFree(d_tokens);
  cudaFree(d_req);
  cudaFree(d_pos);
  cudaFree(d_batch);
  cudaFree(d_scalar);
  cudaFree(d_rows);
  cudaFree(d_scatter);
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  return dgpp::test::run_all();
}

// ---------------------------------------------------------------------------
// M6 6b: the on-device sampling pick (kernels/glm_sample_pick.hpp) against
// the host oracle, bit for bit — the local top-k and slice normalizer, the
// in-place penalties and the count table, and the verdict's decision
// (sample::sample_from_prefix) over a simulated world with greedy,
// resolving, falling-back and padding requests side by side.
// ---------------------------------------------------------------------------
namespace {

struct SampleWorldRun {
  std::vector<std::vector<uint16_t>> tables;   // per rank, the local table
  std::vector<uint16_t> folded;                 // the exact disjoint fold
  std::vector<std::vector<float>> penalized;    // per rank, the slice after
  std::vector<std::vector<int32_t>> counts;     // per rank, the count table
  std::vector<std::vector<dgpp::PickLocal>> locals;
  std::vector<std::vector<PickVerdict>> verdicts;      // per rank
  std::vector<std::vector<dgpp::SampleOutcome>> outcomes;
  std::vector<std::vector<dgpp::SampleSpec>> specs_after;
  std::vector<uint64_t> carry_out;
  std::vector<std::vector<int32_t>> counts_after;  // per rank, post-verdict
};

// Runs the two kernels on every rank of a simulated world: rank k holds
// columns [k*count, (k+1)*count) of `full` ([requests, world*count]).
// `masks` (optional): the rows' token masks, device_sample_mask_words(vocab)
// words per row (the header word = the allowed count, 0 = unconstrained).
SampleWorldRun run_sample_world(const std::vector<float>& full, int requests,
                                int world, int count,
                                const std::vector<dgpp::SampleSpec>& specs,
                                const std::vector<int32_t>& counts_in,
                                const std::vector<int64_t>& fed,
                                const std::vector<int64_t>& positions,
                                int candidates, uint64_t carry,
                                int rows_per_request = 1,
                                const std::vector<uint32_t>& masks = {},
                                const std::vector<dgpp::DraftProposal>& proposals = {}) {
  const int vocab = world * count;
  const int rows = requests * rows_per_request;
  const int mask_stride = dgpp::device_sample_mask_words(vocab);
  if (!masks.empty())
    require(masks.size() == static_cast<size_t>(rows) * mask_stride,
            "mask table shape");
  const size_t table_elems =
      dgpp::device_sample_table_elems(rows, world, candidates);
  // Per-row positions with the request stride, as the graph's step
  // positions are laid out.
  std::vector<int64_t> row_positions(static_cast<size_t>(rows));
  for (int q = 0; q < requests; ++q)
    for (int t = 0; t < rows_per_request; ++t)
      row_positions[static_cast<size_t>(q * rows_per_request + t)] =
          positions[static_cast<size_t>(q)] < 0 ? -1 : positions[static_cast<size_t>(q)] + t;
  SampleWorldRun out;
  out.folded.assign(table_elems, 0);
  for (int k = 0; k < world; ++k) {
    std::vector<float> slice(static_cast<size_t>(rows) * count);
    for (int q = 0; q < rows; ++q)
      std::memcpy(slice.data() + static_cast<size_t>(q) * count,
                  full.data() + (static_cast<size_t>(q) * world + k) * count,
                  count * sizeof(float));
    float* d_logits = device_alloc<float>(slice.size());
    uint16_t* d_table = device_alloc<uint16_t>(table_elems);
    uint64_t* d_carry = device_alloc<uint64_t>(1);
    PickLocal* d_locals = device_alloc<PickLocal>(rows);
    dgpp::SampleSpec* d_specs = device_alloc<dgpp::SampleSpec>(requests);
    int32_t* d_counts = device_alloc<int32_t>(counts_in.size());
    int64_t* d_fed = device_alloc<int64_t>(fed.size());
    int64_t* d_pos = device_alloc<int64_t>(row_positions.size());
    double* d_scratch =
        device_alloc<double>(dgpp::device_sample_scratch_elems(rows, count));
    DGPP_CUDA_OK(cudaMemcpy(d_logits, slice.data(), slice.size() * 4,
                            cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(d_carry, &carry, 8, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(d_specs, specs.data(),
                            specs.size() * sizeof(dgpp::SampleSpec),
                            cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(d_counts, counts_in.data(), counts_in.size() * 4,
                            cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(d_fed, fed.data(), fed.size() * 8,
                            cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(d_pos, row_positions.data(), row_positions.size() * 8,
                            cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemset(d_table, 0xff, table_elems * 2));  // poison
    uint32_t* d_masks = nullptr;
    if (!masks.empty()) {
      d_masks = device_alloc<uint32_t>(masks.size());
      DGPP_CUDA_OK(cudaMemcpy(d_masks, masks.data(), masks.size() * 4,
                              cudaMemcpyHostToDevice));
    }
    dgpp::device_sample_local(d_logits, rows, count, k * count, vocab, k,
                           world, candidates, d_specs, rows_per_request,
                           d_fed, d_pos, /*position_stride=*/rows_per_request,
                           d_counts, /*bias=*/nullptr, d_masks,
                           d_masks ? mask_stride : 0, d_carry, d_table,
                           d_locals, d_scratch, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    if (d_masks) cudaFree(d_masks);
    std::vector<uint16_t> table(table_elems);
    DGPP_CUDA_OK(cudaMemcpy(table.data(), d_table, table_elems * 2,
                            cudaMemcpyDeviceToHost));
    std::vector<float> penalized(slice.size());
    DGPP_CUDA_OK(cudaMemcpy(penalized.data(), d_logits, slice.size() * 4,
                            cudaMemcpyDeviceToHost));
    std::vector<int32_t> counts(counts_in.size());
    DGPP_CUDA_OK(cudaMemcpy(counts.data(), d_counts, counts.size() * 4,
                            cudaMemcpyDeviceToHost));
    std::vector<PickLocal> locals(rows);
    DGPP_CUDA_OK(cudaMemcpy(locals.data(), d_locals,
                            sizeof(PickLocal) * rows,
                            cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < table_elems; ++i) {
      require(out.folded[i] == 0 || table[i] == 0,
              "two ranks wrote the same sampling wire slot");
      out.folded[i] = static_cast<uint16_t>(out.folded[i] + table[i]);
    }
    out.tables.push_back(std::move(table));
    out.penalized.push_back(std::move(penalized));
    out.counts.push_back(std::move(counts));
    out.locals.push_back(std::move(locals));
    cudaFree(d_logits);
    cudaFree(d_table);
    cudaFree(d_carry);
    cudaFree(d_locals);
    cudaFree(d_specs);
    cudaFree(d_counts);
    cudaFree(d_fed);
    cudaFree(d_pos);
    cudaFree(d_scratch);
  }
  // The verdict on every rank over the folded table (the count table as the
  // local kernel left it: untouched, every rank's identical).
  for (int k = 0; k < world; ++k) {
    uint16_t* d_table = device_alloc<uint16_t>(table_elems);
    uint64_t* d_carry = device_alloc<uint64_t>(1);
    dgpp::SampleSpec* d_specs = device_alloc<dgpp::SampleSpec>(requests);
    int64_t* d_pos = device_alloc<int64_t>(row_positions.size());
    int64_t* d_fed = device_alloc<int64_t>(fed.size());
    int32_t* d_counts = device_alloc<int32_t>(counts_in.size());
    PickVerdict* d_verdicts = device_alloc<PickVerdict>(requests);
    dgpp::SampleOutcome* d_out = device_alloc<dgpp::SampleOutcome>(requests);
    DGPP_CUDA_OK(cudaMemcpy(d_table, out.folded.data(), table_elems * 2,
                            cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(d_specs, specs.data(),
                            specs.size() * sizeof(dgpp::SampleSpec),
                            cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(d_pos, row_positions.data(), row_positions.size() * 8,
                            cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(d_fed, fed.data(), fed.size() * 8,
                            cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(d_counts, out.counts[0].data(),
                            counts_in.size() * 4, cudaMemcpyHostToDevice));
    uint32_t* d_masks = nullptr;
    if (!masks.empty()) {
      d_masks = device_alloc<uint32_t>(masks.size());
      DGPP_CUDA_OK(cudaMemcpy(d_masks, masks.data(), masks.size() * 4,
                              cudaMemcpyHostToDevice));
    }
    dgpp::DraftProposal* d_props = nullptr;
    if (!proposals.empty()) {
      require(proposals.size() ==
                  static_cast<size_t>(requests) * dgpp::kSampleProposalSlots,
              "proposal table shape");
      d_props = device_alloc<dgpp::DraftProposal>(proposals.size());
      DGPP_CUDA_OK(cudaMemcpy(d_props, proposals.data(),
                              proposals.size() * sizeof(dgpp::DraftProposal),
                              cudaMemcpyHostToDevice));
    }
    dgpp::device_sample_verdict(d_table, rows, world, k, candidates, vocab,
                             d_specs, requests, rows_per_request, d_fed, d_pos,
                             /*position_stride=*/rows_per_request, d_counts,
                             d_masks, d_masks ? mask_stride : 0, d_verdicts,
                             /*device_verdicts=*/nullptr, d_out, d_carry,
                             nullptr, d_props);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    if (d_masks) cudaFree(d_masks);
    if (d_props) cudaFree(d_props);
    std::vector<int32_t> counts_after(counts_in.size());
    DGPP_CUDA_OK(cudaMemcpy(counts_after.data(), d_counts,
                            counts_in.size() * 4, cudaMemcpyDeviceToHost));
    out.counts_after.push_back(std::move(counts_after));
    std::vector<PickVerdict> verdicts(requests);
    std::vector<dgpp::SampleOutcome> outcomes(requests);
    std::vector<dgpp::SampleSpec> specs_after(requests);
    uint64_t carry_out = 0;
    DGPP_CUDA_OK(cudaMemcpy(verdicts.data(), d_verdicts,
                            sizeof(PickVerdict) * requests,
                            cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(outcomes.data(), d_out,
                            sizeof(dgpp::SampleOutcome) * requests,
                            cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(specs_after.data(), d_specs,
                            sizeof(dgpp::SampleSpec) * requests,
                            cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(&carry_out, d_carry, 8, cudaMemcpyDeviceToHost));
    out.verdicts.push_back(std::move(verdicts));
    out.outcomes.push_back(std::move(outcomes));
    out.specs_after.push_back(std::move(specs_after));
    out.carry_out.push_back(carry_out);
    cudaFree(d_table);
    cudaFree(d_carry);
    cudaFree(d_specs);
    cudaFree(d_pos);
    cudaFree(d_fed);
    cudaFree(d_counts);
    cudaFree(d_verdicts);
    cudaFree(d_out);
  }
  return out;
}

dgpp::sample::Params params_of(const dgpp::SampleSpec& s) {
  dgpp::sample::Params p;
  p.temperature = s.temperature;
  p.top_p = s.top_p;
  p.min_p = s.min_p;
  p.top_k = s.top_k;
  p.repetition_penalty = s.repetition_penalty;
  p.frequency_penalty = s.frequency_penalty;
  p.presence_penalty = s.presence_penalty;
  return p;
}

bool bits_equal(float a, float b) { return std::memcmp(&a, &b, 4) == 0; }
bool bits_equal(double a, double b) { return std::memcmp(&a, &b, 8) == 0; }

}  // namespace

DGPP_TEST(sample_pick_matches_host_oracle_bitwise_over_simulated_world) {
  Rng rng(0x5a3e);
  constexpr int kWorld = 4;
  struct Shape {
    int count, candidates;
  };
  // One shape below a normalizer chunk, one spanning three chunks per
  // slice, and non-power-of-two widths (the streaming select's bitonic
  // merge wants a power of two; the kernel rounds its selection up).
  const Shape shapes[] = {{96, 32}, {700, 64}, {48, 24}, {700, 112}, {40, 7}};
  int resolved_total = 0, fallback_total = 0;
  for (const Shape& shape : shapes) {
    const int count = shape.count;
    const int vocab = kWorld * count;
    constexpr int requests = 6;
    for (int trial = 0; trial < 5; ++trial) {
      // Requests: 0 greedy; 1 model default (peaked: resolves); 2 model
      // default (flat: falls back); 3 pure temperature; 4 min-p; 5 padding.
      std::vector<float> full(static_cast<size_t>(requests) * vocab);
      for (int q = 0; q < requests; ++q) {
        float* row = full.data() + static_cast<size_t>(q) * vocab;
        const bool flat = q == 2;
        for (int v = 0; v < vocab; ++v) {
          const uint64_t r = rng.next();
          row[v] = flat ? static_cast<float>(r % 3) * 0.01f
                        : static_cast<float>((r >> 8) % 41) * 0.25f - 5.0f;
        }
        if (!flat) {
          row[static_cast<size_t>(rng.next() % vocab)] = 7.0f + trial;
          row[static_cast<size_t>(rng.next() % vocab)] = 6.5f;
        }
      }
      std::vector<dgpp::SampleSpec> specs(requests);
      specs[0].temperature = 0.0f;
      specs[1].temperature = 1.0f;
      specs[1].top_p = 0.95f;
      specs[1].frequency_penalty = 0.3f;
      specs[1].presence_penalty = 0.1f;
      specs[2].temperature = 1.0f;
      specs[2].top_p = 0.95f;
      specs[3].temperature = 0.7f;
      specs[3].top_p = 1.0f;
      specs[3].repetition_penalty = 1.2f;
      specs[4].temperature = 0.9f;
      specs[4].top_p = 0.9f;
      specs[4].min_p = 0.05f;
      specs[5].temperature = 1.0f;
      for (int q = 0; q < requests; ++q) {
        specs[q].seed = 0x1000 + q + 17 * trial;
        specs[q].counter = 3 + q;
      }
      // Count tables: a few context ids per request, some inside the fed
      // token's neighborhood so a penalty actually lands on a candidate.
      std::vector<int32_t> counts(static_cast<size_t>(requests) * vocab, 0);
      std::vector<int64_t> fed(requests), positions(requests);
      for (int q = 0; q < requests; ++q) {
        fed[q] = static_cast<int64_t>(rng.next() % vocab);
        positions[q] = q == 5 ? -1 : 10 + q;
        for (int j = 0; j < 4; ++j)
          counts[static_cast<size_t>(q) * vocab + rng.next() % vocab] += 1 + (j & 1);
      }
      const uint64_t carry = 0x2b2b2b2b2b2bull;
      const SampleWorldRun run = run_sample_world(
          full, requests, kWorld, count, specs, counts, fed, positions,
          shape.candidates, carry);

      // ---- host expectations per request -----------------------------
      std::vector<int32_t> want_winners(requests, -1);
      std::vector<int> want_rows(requests, 1);
      for (int q = 0; q < requests; ++q) {
        const float* row = full.data() + static_cast<size_t>(q) * vocab;
        if (q == 5) {
          want_rows[q] = 0;
          for (int k = 0; k < kWorld; ++k) {
            require(run.verdicts[k][q].rows == 0 && run.verdicts[k][q].accepted == 0 &&
                        run.verdicts[k][q].next == -1,
                    "padding request must be inactive");
            require(run.outcomes[k][q].sampled == 0 && run.outcomes[k][q].fallback == 0,
                    "padding request has no outcome");
          }
          continue;
        }
        if (specs[q].temperature <= 0.0f) {
          const Candidate best = dgpp::sample::local_max(row, vocab, 0);
          want_winners[q] = best.id;
          for (int k = 0; k < kWorld; ++k) {
            require(run.verdicts[k][q].next == best.id &&
                        run.verdicts[k][q].winners[0] == best.id &&
                        run.verdicts[k][q].accepted == 1,
                    "greedy request: device argmax != host");
            require(run.outcomes[k][q].sampled == 0 &&
                        run.outcomes[k][q].counter == specs[q].counter,
                    "greedy request must not draw");
            // The greedy slice is untouched (no penalties).
            for (int i = 0; i < count; ++i)
              require(bits_equal(run.penalized[k][static_cast<size_t>(q) * count + i],
                                 row[k * count + i]),
                      "greedy row's logits must be untouched");
          }
          continue;
        }
        // The sampled request: the host path over the same inputs.
        std::vector<int32_t> host_counts(counts.begin() + static_cast<long>(q) * vocab,
                                         counts.begin() + static_cast<long>(q + 1) * vocab);
        host_counts[static_cast<size_t>(fed[q])] += 1;
        std::unordered_map<int32_t, int32_t> ctx;
        for (int v = 0; v < vocab; ++v)
          if (host_counts[static_cast<size_t>(v)]) ctx[v] = host_counts[static_cast<size_t>(v)];
        const dgpp::sample::Params p = params_of(specs[q]);
        std::vector<float> adjusted(row, row + vocab);
        dgpp::sample::apply_penalties(adjusted.data(), vocab, 0, p, ctx);
        std::vector<std::vector<Candidate>> shards;
        std::vector<double> lses;
        const size_t group = dgpp::device_sample_rank_group_slots(shape.candidates);
        for (int k = 0; k < kWorld; ++k) {
          const float* aslice = adjusted.data() + k * count;
          // The device penalized its slice in place, bitwise.
          for (int i = 0; i < count; ++i)
            require(bits_equal(run.penalized[k][static_cast<size_t>(q) * count + i], aslice[i]),
                    "penalized slice differs from apply_penalties on rank " +
                        std::to_string(k) + " request " + std::to_string(q));
          // ... without touching the count table; the verdict then
          // commits the fed token exactly once.
          for (int v = 0; v < vocab; ++v) {
            require(run.counts[k][static_cast<size_t>(q) * vocab + v] ==
                        counts[static_cast<size_t>(q) * vocab + v],
                    "the local kernels must not write the count table");
            require(run.counts_after[k][static_cast<size_t>(q) * vocab + v] ==
                        host_counts[static_cast<size_t>(v)],
                    "count table after the verdict drifted");
          }
          const std::vector<Candidate> want =
              dgpp::sample::local_topk(aslice, count, k * count, shape.candidates);
          const uint16_t* grp = run.folded.data() +
                                (static_cast<size_t>(q) * kWorld + k) * group;
          for (int j = 0; j < shape.candidates; ++j) {
            const uint16_t* slot = grp + static_cast<size_t>(j) * kPickSlotsPerRank;
            const uint32_t id = static_cast<uint32_t>(
                decode_digits(slot + kPickLogitDigits, kPickIdDigits));
            if (j < static_cast<int>(want.size())) {
              uint32_t bits = 0;
              std::memcpy(&bits, &want[static_cast<size_t>(j)].logit, 4);
              require(id == static_cast<uint32_t>(want[static_cast<size_t>(j)].id) &&
                          decode_digits(slot, kPickLogitDigits) == bits,
                      "local top-k candidate " + std::to_string(j) +
                          " differs from local_topk on rank " + std::to_string(k));
            } else {
              require(id == dgpp::kSampleEmptyId, "unused slot must carry the empty id");
            }
          }
          const double lse = dgpp::sample::slice_logsumexp(aslice, count, p.temperature);
          const uint64_t lse_bits = decode_digits(
              grp + static_cast<size_t>(shape.candidates) * kPickSlotsPerRank,
              dgpp::kSampleLseDigits);
          double got_lse = 0.0;
          std::memcpy(&got_lse, &lse_bits, 8);
          require(bits_equal(got_lse, lse),
                  "slice log-sum-exp differs from the host's chunked order");
          shards.push_back(want);
          lses.push_back(lse);
          require(run.locals[k][q].best_id == want[0].id &&
                      bits_equal(run.locals[k][q].best_logit, want[0].logit),
                  "locals differ");
        }
        const std::vector<Candidate> merged =
            dgpp::sample::merge_topk(shards, shape.candidates);
        const double Z = dgpp::sample::merge_logsumexp(lses);
        dgpp::sample::Rng host_rng{specs[q].seed, specs[q].counter};
        const dgpp::sample::PrefixDecision want =
            dgpp::sample::sample_from_prefix(merged, vocab, Z, p, host_rng);
        if (want.resolved) ++resolved_total; else ++fallback_total;
        for (int k = 0; k < kWorld; ++k) {
          const PickVerdict& v = run.verdicts[k][q];
          const dgpp::SampleOutcome& o = run.outcomes[k][q];
          require(o.sampled == 1, "sampled request must report a decision");
          require(bits_equal(o.normalizer[0], Z), "normalizer differs from merge_logsumexp");
          require(bits_equal(o.covered_mass[0], want.covered_mass), "covered mass differs");
          require(o.counter == host_rng.counter && run.specs_after[k][q].counter == host_rng.counter,
                  "counter differs from the host oracle (request " + std::to_string(q) + ")");
          if (want.resolved) {
            require(o.fallback == 0, "resolved on the host, fallback on the device");
            require(v.next == want.result.token && v.winners[0] == want.result.token,
                    "device token " + std::to_string(v.next) + " != host " +
                        std::to_string(want.result.token) + " (request " +
                        std::to_string(q) + ")");
            require(bits_equal(o.logprob[0], want.result.logprob), "logprob bits differ");
          } else {
            require(o.fallback == 1, "fallback on the host, resolved on the device");
            require(v.next == merged[0].id, "fallback carries the provisional argmax");
          }
          require(v.rows == 1 && v.accepted == 1, "T=1 verdict shape");
        }
        want_winners[q] = run.verdicts[0][q].next;
      }
      // The digest chain is glm_pick_verdict's over the same verdicts.
      {
        uint64_t digest = 0;
        uint64_t seed = 0x9e3779b97f4a7c15ull;  // splitmix64(requests) below
        (void)seed;
        // Recompute with the host mirror: chain over requests as the kernel does.
        auto sm = [](uint64_t x) {
          x += 0x9e3779b97f4a7c15ull;
          x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
          x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
          return x ^ (x >> 31);
        };
        digest = sm(static_cast<uint64_t>(requests));
        for (int q = 0; q < requests; ++q) {
          int32_t w[dgpp::kPickMaxRows];
          std::fill(std::begin(w), std::end(w), -1);
          w[0] = want_winners[q];
          const uint64_t one = dgpp::device_pick_digest(want_rows[q], want_rows[q], w);
          digest = sm(digest ^ one);
        }
        digest &= (1ull << dgpp::kPickDigestBits) - 1;
        for (int k = 0; k < kWorld; ++k) {
          require(run.carry_out[k] == digest, "carry != the digest chain");
          for (int q = 0; q < requests; ++q) {
            require(run.verdicts[k][q].digest == digest, "verdict digest");
            require(run.verdicts[k][q].digest_mismatch == 0, "identical carries flagged");
            for (int j = 0; j < kWorld; ++j)
              require(run.verdicts[k][q].peer_digests[j] == carry, "peer digest decode");
          }
        }
      }
    }
  }
  require(resolved_total > 0 && fallback_total > 0,
          "the sweep must exercise both outcomes (resolved " +
              std::to_string(resolved_total) + ", fallback " +
              std::to_string(fallback_total) + ")");
}

// M6 6g: constrained rows. The host writes a token mask per row; a masked
// id is absent on the device exactly as on the host (-inf in place, never
// a candidate, no mass, the row's vocabulary its allowed count): an
// allow-list of a few ids spanning three ranks with the fourth rank fully
// masked (its lse -inf, its group empty), a free row with a few forbidden
// ids, a greedy row under a mask (the masked argmax through the full
// path), an unconstrained row beside them — every group, lse, decision,
// counter and logprob bitwise the host oracle over the same masks; then
// the T=2 verify with a masked draft (rejected outright, no fallback) and
// row 1 under its own mask.
namespace {

std::vector<uint32_t> free_mask_words(int vocab) {
  const int words = dgpp::device_sample_mask_words(vocab);
  std::vector<uint32_t> m(static_cast<size_t>(words), 0xffffffffu);
  m[0] = static_cast<uint32_t>(vocab);
  if (vocab % 32 != 0) m.back() &= (1u << (vocab % 32)) - 1u;
  return m;
}
void mask_clear(std::vector<uint32_t>& m, int id) {
  uint32_t& w = m[static_cast<size_t>(1 + (id >> 5))];
  const uint32_t bit = 1u << (id & 31);
  if (w & bit) {
    w &= ~bit;
    m[0] -= 1;
  }
}
std::vector<uint32_t> list_mask_words(int vocab, const std::vector<int>& ids) {
  const int words = dgpp::device_sample_mask_words(vocab);
  std::vector<uint32_t> m(static_cast<size_t>(words), 0u);
  for (const int id : ids) {
    uint32_t& w = m[static_cast<size_t>(1 + (id >> 5))];
    const uint32_t bit = 1u << (id & 31);
    if (!(w & bit)) {
      w |= bit;
      m[0] += 1;
    }
  }
  return m;
}

}  // namespace

DGPP_TEST(sample_pick_masked_rows_match_host_oracle_bitwise) {
  Rng rng(0x6a5c);
  constexpr int kWorld = 4;
  constexpr int count = 96;
  constexpr int vocab = kWorld * count;
  constexpr int candidates = 32;
  constexpr int requests = 5;
  const int stride = dgpp::device_sample_mask_words(vocab);
  int list_resolved = 0, free_fallbacks = 0;
  for (int trial = 0; trial < 8; ++trial) {
    std::vector<float> full(static_cast<size_t>(requests) * vocab);
    for (int q = 0; q < requests; ++q) {
      float* row = full.data() + static_cast<size_t>(q) * vocab;
      for (int v = 0; v < vocab; ++v)
        row[v] = static_cast<float>((rng.next() >> 8) % 41) * 0.25f - 5.0f;
      row[static_cast<size_t>(rng.next() % vocab)] = 7.0f;
    }
    // 0: sampled, allow-list of five ids on ranks 0..2 (rank 3 empty);
    // 1: sampled pure temperature, free mask with 40 forbidden ids (a
    //    flat-ish row: fallbacks happen; the gather path is the host's);
    // 2: greedy under the allow-list (the masked argmax);
    // 3: unconstrained sampled (the mask table's header is 0);
    // 4: sampled, allow-list of one id (the degenerate distribution).
    std::vector<dgpp::SampleSpec> specs(requests);
    specs[0].temperature = 1.0f; specs[0].top_p = 0.95f;
    specs[1].temperature = 1.3f; specs[1].top_p = 1.0f;
    specs[1].presence_penalty = 0.2f;
    specs[2].temperature = 0.0f;
    specs[3].temperature = 1.0f; specs[3].top_p = 0.9f;
    specs[4].temperature = 0.8f; specs[4].top_p = 0.95f;
    for (int q = 0; q < requests; ++q) {
      specs[q].seed = 0x3000 + q + 13 * trial;
      specs[q].counter = 2 + q;
    }
    std::vector<int> allow = {5 + trial, 40, count + 3, count + 70 + trial,
                              2 * count + 11};
    const int lone = 2 * count + 50 + trial;
    std::vector<uint32_t> masks(static_cast<size_t>(requests) * stride, 0u);
    const auto put = [&](int row, const std::vector<uint32_t>& m) {
      std::copy(m.begin(), m.end(), masks.begin() + static_cast<long>(row) * stride);
    };
    put(0, list_mask_words(vocab, allow));
    {
      std::vector<uint32_t> m = free_mask_words(vocab);
      for (int j = 0; j < 40; ++j) mask_clear(m, static_cast<int>(rng.next() % vocab));
      put(1, m);
    }
    put(2, list_mask_words(vocab, allow));
    put(4, list_mask_words(vocab, {lone}));
    std::vector<int32_t> counts(static_cast<size_t>(requests) * vocab, 0);
    std::vector<int64_t> fed(requests), positions(requests, 1);
    for (int q = 0; q < requests; ++q) {
      fed[q] = static_cast<int64_t>(rng.next() % vocab);
      counts[static_cast<size_t>(q) * vocab + rng.next() % vocab] += 1;
    }
    const SampleWorldRun run = run_sample_world(full, requests, kWorld, count,
                                                specs, counts, fed, positions,
                                                candidates, 0x4d4dull, 1, masks);
    for (int q = 0; q < requests; ++q) {
      const float* row = full.data() + static_cast<size_t>(q) * vocab;
      const uint32_t* mask = masks.data() + static_cast<size_t>(q) * stride;
      const bool constrained = mask[0] != 0u;
      const int allowed = constrained ? static_cast<int>(mask[0]) : vocab;
      std::vector<int32_t> host_counts(counts.begin() + static_cast<long>(q) * vocab,
                                       counts.begin() + static_cast<long>(q + 1) * vocab);
      host_counts[static_cast<size_t>(fed[q])] += 1;
      std::unordered_map<int32_t, int32_t> ctx;
      for (int v = 0; v < vocab; ++v)
        if (host_counts[static_cast<size_t>(v)]) ctx[v] = host_counts[static_cast<size_t>(v)];
      dgpp::sample::Params p = params_of(specs[q]);
      const bool greedy = p.temperature <= 0.0f;
      if (greedy) p.temperature = 1.0f;  // the full path's raw distribution
      std::vector<float> adjusted(row, row + vocab);
      dgpp::sample::apply_penalties(adjusted.data(), vocab, 0, p, ctx);
      if (constrained)
        dgpp::sample::apply_mask(adjusted.data(), vocab, 0, mask + 1, vocab);
      std::vector<std::vector<Candidate>> shards;
      std::vector<double> lses;
      const size_t group = dgpp::device_sample_rank_group_slots(candidates);
      for (int k = 0; k < kWorld; ++k) {
        const float* aslice = adjusted.data() + k * count;
        for (int i = 0; i < count; ++i)
          require(bits_equal(run.penalized[k][static_cast<size_t>(q) * count + i], aslice[i]),
                  "masked slice differs (-inf in place) on rank " + std::to_string(k) +
                      " request " + std::to_string(q));
        const std::vector<Candidate> want =
            dgpp::sample::local_topk(aslice, count, k * count, candidates);
        const uint16_t* grp = run.folded.data() +
                              (static_cast<size_t>(q) * kWorld + k) * group;
        for (int j = 0; j < candidates; ++j) {
          const uint16_t* slot = grp + static_cast<size_t>(j) * kPickSlotsPerRank;
          const uint32_t id = static_cast<uint32_t>(
              decode_digits(slot + kPickLogitDigits, kPickIdDigits));
          if (j < static_cast<int>(want.size())) {
            uint32_t bits = 0;
            std::memcpy(&bits, &want[static_cast<size_t>(j)].logit, 4);
            require(id == static_cast<uint32_t>(want[static_cast<size_t>(j)].id) &&
                        decode_digits(slot, kPickLogitDigits) == bits,
                    "masked local top-k candidate " + std::to_string(j) +
                        " differs on rank " + std::to_string(k) + " request " +
                        std::to_string(q));
          } else {
            require(id == dgpp::kSampleEmptyId,
                    "an absent candidate slot must carry the empty id (rank " +
                        std::to_string(k) + " request " + std::to_string(q) + ")");
          }
        }
        const double lse = dgpp::sample::slice_logsumexp(aslice, count, p.temperature);
        const uint64_t lse_bits = decode_digits(
            grp + static_cast<size_t>(candidates) * kPickSlotsPerRank,
            dgpp::kSampleLseDigits);
        double got_lse = 0.0;
        std::memcpy(&got_lse, &lse_bits, 8);
        require(bits_equal(got_lse, lse),
                "masked slice lse differs on rank " + std::to_string(k) +
                    " request " + std::to_string(q) + " (want " +
                    std::to_string(lse) + ", got " + std::to_string(got_lse) + ")");
        if (q == 0 && k == 3) require(lse == -INFINITY && want.empty(),
                                      "the fully masked rank has no mass and no candidates");
        shards.push_back(want);
        lses.push_back(lse);
      }
      const std::vector<Candidate> merged =
          dgpp::sample::merge_topk(shards, candidates);
      const double Z = dgpp::sample::merge_logsumexp(lses);
      if (greedy) {
        for (int k = 0; k < kWorld; ++k) {
          const PickVerdict& v = run.verdicts[k][q];
          require(v.next == merged[0].id && v.accepted == 1,
                  "greedy row under a mask: the masked argmax (got " +
                      std::to_string(v.next) + ", want " + std::to_string(merged[0].id) + ")");
          require(run.outcomes[k][q].sampled == 0, "greedy row draws nothing");
          bool in_allow = false;
          for (const int a : allow) in_allow = in_allow || a == v.next;
          require(in_allow, "the greedy pick is inside the allow-list");
        }
        continue;
      }
      dgpp::sample::Rng host_rng{specs[q].seed, specs[q].counter};
      const dgpp::sample::PrefixDecision want =
          dgpp::sample::sample_from_prefix(merged, allowed, Z, p, host_rng);
      if (q == 0 || q == 4) {
        require(want.resolved, "an allow-list inside the prefix resolves");
        ++list_resolved;
      }
      if (q == 1 && !want.resolved) ++free_fallbacks;
      for (int k = 0; k < kWorld; ++k) {
        const PickVerdict& v = run.verdicts[k][q];
        const dgpp::SampleOutcome& o = run.outcomes[k][q];
        require(o.sampled == 1, "sampled row reports a decision");
        require(bits_equal(o.normalizer[0], Z), "masked normalizer differs (request " +
                                                 std::to_string(q) + ")");
        require(o.counter == host_rng.counter, "masked counter differs");
        if (want.resolved) {
          require(o.fallback == 0 && v.next == want.result.token,
                  "masked decision differs: device " + std::to_string(v.next) +
                      " host " + std::to_string(want.result.token) + " (request " +
                      std::to_string(q) + ")");
          require(bits_equal(o.logprob[0], want.result.logprob), "masked logprob differs");
          if (constrained)
            require(((mask[1 + (v.next >> 5)] >> (v.next & 31)) & 1u) != 0u,
                    "the decided token is inside the mask");
        } else {
          require(o.fallback == 1 && v.next == merged[0].id, "masked fallback shape");
        }
      }
      if (q == 4)
        for (int k = 0; k < kWorld; ++k)
          require(run.verdicts[k][q].next == lone, "one allowed id: it is the token");
    }
  }
  require(list_resolved == 16, "every allow-list row resolved on the device");
  (void)free_fallbacks;

  // The T=2 verify under masks: the draft is a masked id on row 0 (rejected
  // outright: no fallback, the residual from the allowed set) or an
  // allowed one (the accept test); row 1 carries its own mask.
  int rejected_masked = 0, accepted = 0;
  for (int trial = 0; trial < 6; ++trial) {
    constexpr int rpr = 2;
    constexpr int reqs = 2;
    std::vector<float> full(static_cast<size_t>(reqs * rpr) * vocab);
    for (int r = 0; r < reqs * rpr; ++r) {
      float* row = full.data() + static_cast<size_t>(r) * vocab;
      for (int v = 0; v < vocab; ++v)
        row[v] = static_cast<float>((rng.next() >> 8) % 41) * 0.25f - 5.0f;
    }
    const std::vector<int> allow0 = {7, count + 9, 2 * count + 4, 3 * count + 30};
    const std::vector<int> allow1 = {12, count + 1, 3 * count + 2};
    // A peak on an allowed id of row 0, so the allowed draft mostly stands.
    for (int q = 0; q < reqs; ++q) {
      full[static_cast<size_t>(2 * q) * vocab + allow0[1]] = 9.0f;
      full[static_cast<size_t>(2 * q + 1) * vocab + allow1[0]] = 8.0f;
    }
    std::vector<dgpp::SampleSpec> specs(reqs);
    for (int q = 0; q < reqs; ++q) {
      specs[q].temperature = 1.0f;
      specs[q].top_p = 1.0f;  // pure: the masked draft would fall back
                              // without the exclusion shortcut
      specs[q].seed = 0x4000 + q + 7 * trial;
      specs[q].counter = 1;
    }
    std::vector<uint32_t> masks(static_cast<size_t>(reqs * rpr) * stride, 0u);
    for (int q = 0; q < reqs; ++q) {
      const std::vector<uint32_t> m0 = list_mask_words(vocab, allow0);
      const std::vector<uint32_t> m1 = list_mask_words(vocab, allow1);
      std::copy(m0.begin(), m0.end(), masks.begin() + static_cast<long>(2 * q) * stride);
      std::copy(m1.begin(), m1.end(), masks.begin() + static_cast<long>(2 * q + 1) * stride);
    }
    std::vector<int32_t> counts(static_cast<size_t>(reqs) * vocab, 0);
    std::vector<int64_t> fed(reqs * rpr), positions(reqs, 3);
    // Request 0's draft is a MASKED id (an unconstrained MTP head's guess);
    // request 1's is the allowed peak.
    fed[0] = 3; fed[1] = 20 + trial;  // 20..25: never in allow0
    fed[2] = 4; fed[3] = allow0[1];
    const SampleWorldRun run = run_sample_world(full, reqs, kWorld, count, specs,
                                                counts, fed, positions,
                                                candidates, 0x5e5eull, rpr, masks);
    for (int q = 0; q < reqs; ++q) {
      const int32_t draft = static_cast<int32_t>(fed[2 * q + 1]);
      const dgpp::sample::Params p = params_of(specs[q]);
      const auto merged_of = [&](int row, const std::vector<int>& allow_ids, double* Z) {
        std::vector<float> adj(full.begin() + static_cast<long>(row) * vocab,
                               full.begin() + static_cast<long>(row + 1) * vocab);
        std::unordered_map<int32_t, int32_t> ctx;
        ctx[static_cast<int32_t>(fed[2 * q])] = 1;
        if (row % 2 == 1) ctx[draft] += 1;
        dgpp::sample::apply_penalties(adj.data(), vocab, 0, p, ctx);
        const std::vector<uint32_t> m = list_mask_words(vocab, allow_ids);
        dgpp::sample::apply_mask(adj.data(), vocab, 0, m.data() + 1, vocab);
        std::vector<std::vector<Candidate>> shards;
        std::vector<double> lses;
        for (int k = 0; k < kWorld; ++k) {
          shards.push_back(dgpp::sample::local_topk(adj.data() + k * count, count,
                                                        k * count, candidates));
          lses.push_back(dgpp::sample::slice_logsumexp(adj.data() + k * count,
                                                            count, p.temperature));
        }
        *Z = dgpp::sample::merge_logsumexp(lses);
        return dgpp::sample::merge_topk(shards, candidates);
      };
      double Z0 = 0.0, Z1 = 0.0;
      const std::vector<Candidate> m0 = merged_of(2 * q, allow0, &Z0);
      const std::vector<Candidate> m1 = merged_of(2 * q + 1, allow1, &Z1);
      bool draft_allowed = false;
      for (const int a : allow0) draft_allowed = draft_allowed || a == draft;
      dgpp::sample::Rng host{specs[q].seed, specs[q].counter};
      const dgpp::sample::SpecPrefixDecision d0 =
          dgpp::sample::spec_accept_from_prefix(
              m0, static_cast<int>(allow0.size()), Z0, draft, p, host,
              /*draft_excluded=*/!draft_allowed);
      require(d0.resolved, "a masked-or-allowed draft over a complete allowed prefix decides");
      int32_t want_w0 = d0.result.token, want_w1 = -1;
      int want_accepted = 1;
      if (d0.accepted) {
        ++accepted;
        want_accepted = 2;
        const dgpp::sample::PrefixDecision d1 =
            dgpp::sample::sample_from_prefix(
                m1, static_cast<int>(allow1.size()), Z1, p, host);
        require(d1.resolved, "row 1 over its allowed set decides");
        want_w1 = d1.result.token;
      } else if (!draft_allowed) {
        ++rejected_masked;
      }
      for (int k = 0; k < kWorld; ++k) {
        const PickVerdict& v = run.verdicts[k][q];
        const dgpp::SampleOutcome& o = run.outcomes[k][q];
        require(o.fallback == 0, "no fallback under complete allowed prefixes (request " +
                                     std::to_string(q) + ", trial " + std::to_string(trial) + ")");
        require(v.accepted == want_accepted && v.winners[0] == want_w0 &&
                    v.winners[1] == want_w1 && o.counter == host.counter,
                "masked T=2 verdict differs from the oracle: got accepted " +
                    std::to_string(v.accepted) + " winners " + std::to_string(v.winners[0]) +
                    "/" + std::to_string(v.winners[1]) + ", want " +
                    std::to_string(want_accepted) + " " + std::to_string(want_w0) + "/" +
                    std::to_string(want_w1));
        bool in0 = false;
        for (const int a : allow0) in0 = in0 || a == v.winners[0];
        require(in0, "row 0's token is inside its mask");
        if (want_accepted == 2) {
          bool in1 = false;
          for (const int a : allow1) in1 = in1 || a == v.winners[1];
          require(in1, "row 1's token is inside its mask");
        }
      }
    }
  }
  require(rejected_masked == 6 && accepted > 0,
          "every masked draft rejected without a fallback (" +
              std::to_string(rejected_masked) + "), some allowed drafts stood (" +
              std::to_string(accepted) + ")");
}

// The local top-k's three select paths, on slices wider than the shared
// list: the ids at or above the per-thread-maxima bound taken whole (a
// slice of distinct values: ~k of them reach the bound), the list radix
// with its index tie pass (hundreds of ids tied at the top), and the
// whole-slice radix (thousands tied: the list overflows). Every rank's
// group is local_topk bitwise, every lse the host's, every decision the
// oracle's.
DGPP_TEST(sample_local_topk_tie_paths_match_local_topk) {
  Rng rng(0x7e5);
  constexpr int kWorld = 2;
  constexpr int count = 8192;
  constexpr int vocab = kWorld * count;
  constexpr int candidates = 128;
  constexpr int requests = 3;
  const int tied_per_slice[] = {0, 450, 3000};
  for (int fixture = 0; fixture < 3; ++fixture) {
    std::vector<float> full(static_cast<size_t>(requests) * vocab);
    for (int q = 0; q < requests; ++q) {
      float* row = full.data() + static_cast<size_t>(q) * vocab;
      for (int v = 0; v < vocab; ++v)
        row[v] = static_cast<float>((rng.next() >> 8) % 100000) * 1e-4f - 5.0f;
      // The ties sit at the top of EVERY slice (each rank must take the
      // path under test), on ids drawn without replacement.
      for (int k = 0; k < kWorld; ++k) {
        std::vector<int> ids(count);
        for (int i = 0; i < count; ++i) ids[static_cast<size_t>(i)] = i;
        for (int j = 0; j < tied_per_slice[fixture]; ++j) {
          const int pick = j + static_cast<int>(rng.next() % static_cast<uint64_t>(count - j));
          std::swap(ids[static_cast<size_t>(j)], ids[static_cast<size_t>(pick)]);
          row[k * count + ids[static_cast<size_t>(j)]] = 9.0f;
        }
      }
    }
    std::vector<dgpp::SampleSpec> specs(requests);
    specs[0].temperature = 1.0f;   // nucleus over the merged prefix
    specs[0].top_p = 0.95f;
    specs[1].temperature = 0.7f;   // the pure fp64 walk
    specs[1].top_p = 1.0f;
    specs[2].temperature = 0.0f;   // greedy through the full path
    specs[2].logprobs = 2;
    for (int q = 0; q < requests; ++q) {
      specs[q].seed = 0x4000 + q + 7 * fixture;
      specs[q].counter = 1;
    }
    std::vector<int32_t> counts(static_cast<size_t>(requests) * vocab, 0);
    std::vector<int64_t> fed(requests), positions(requests, 3);
    for (int q = 0; q < requests; ++q) fed[q] = static_cast<int64_t>(rng.next() % vocab);
    const SampleWorldRun run = run_sample_world(full, requests, kWorld, count,
                                                specs, counts, fed, positions,
                                                candidates, 0x5555ull);
    const size_t group = dgpp::device_sample_rank_group_slots(candidates);
    for (int q = 0; q < requests; ++q) {
      const float* row = full.data() + static_cast<size_t>(q) * vocab;
      const float T = specs[q].temperature > 0.0f ? specs[q].temperature : 1.0f;
      std::vector<std::vector<Candidate>> shards;
      std::vector<double> lses;
      for (int k = 0; k < kWorld; ++k) {
        const float* slice = row + k * count;
        const std::vector<Candidate> want =
            dgpp::sample::local_topk(slice, count, k * count, candidates);
        require(static_cast<int>(want.size()) == candidates, "fixture width");
        const uint16_t* grp = run.folded.data() +
                              (static_cast<size_t>(q) * kWorld + k) * group;
        for (int j = 0; j < candidates; ++j) {
          const uint16_t* slot = grp + static_cast<size_t>(j) * kPickSlotsPerRank;
          uint32_t bits = 0;
          std::memcpy(&bits, &want[static_cast<size_t>(j)].logit, 4);
          require(decode_digits(slot + kPickLogitDigits, kPickIdDigits) ==
                          static_cast<uint64_t>(want[static_cast<size_t>(j)].id) &&
                      decode_digits(slot, kPickLogitDigits) == bits,
                  "fixture " + std::to_string(fixture) + " rank " +
                      std::to_string(k) + " request " + std::to_string(q) +
                      ": candidate " + std::to_string(j) + " differs from local_topk");
        }
        const double lse = dgpp::sample::slice_logsumexp(slice, count, T);
        const uint64_t lse_bits = decode_digits(
            grp + static_cast<size_t>(candidates) * kPickSlotsPerRank,
            dgpp::kSampleLseDigits);
        double got = 0.0;
        std::memcpy(&got, &lse_bits, 8);
        require(bits_equal(got, lse), "slice log-sum-exp differs");
        shards.push_back(want);
        lses.push_back(lse);
      }
      const std::vector<Candidate> merged =
          dgpp::sample::merge_topk(shards, candidates);
      const double Z = dgpp::sample::merge_logsumexp(lses);
      dgpp::sample::Params p = params_of(specs[q]);
      p.logprobs = specs[q].logprobs;
      int32_t want_token = -1;
      bool want_resolved = true;
      if (specs[q].temperature > 0.0f) {
        dgpp::sample::Rng host{specs[q].seed, specs[q].counter};
        const dgpp::sample::PrefixDecision d =
            dgpp::sample::sample_from_prefix(merged, vocab, Z, p, host);
        want_resolved = d.resolved;
        want_token = d.resolved ? d.result.token : merged[0].id;
      } else {
        want_token = merged[0].id;
      }
      for (int k = 0; k < kWorld; ++k) {
        require(run.verdicts[k][q].next == want_token,
                "fixture " + std::to_string(fixture) + " request " +
                    std::to_string(q) + ": device token " +
                    std::to_string(run.verdicts[k][q].next) + " != host " +
                    std::to_string(want_token));
        require((run.outcomes[k][q].fallback == 1) == !want_resolved,
                "fallback flag differs from the oracle");
        if (specs[q].temperature > 0.0f)
          require(bits_equal(run.outcomes[k][q].normalizer[0], Z), "normalizer differs");
      }
    }
  }
}

DGPP_TEST(sample_count_tokens_accumulates_the_prompt) {
  constexpr int vocab = 500;
  std::vector<int64_t> ids{1, 1, 7, 499, 7, 7, 0};
  int32_t* d_counts = device_alloc<int32_t>(vocab);
  int64_t* d_ids = device_alloc<int64_t>(ids.size());
  DGPP_CUDA_OK(cudaMemset(d_counts, 0, vocab * 4));
  DGPP_CUDA_OK(cudaMemcpy(d_ids, ids.data(), ids.size() * 8, cudaMemcpyHostToDevice));
  dgpp::device_sample_count_tokens(d_counts, d_ids, static_cast<int>(ids.size()), vocab, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<int32_t> counts(vocab);
  DGPP_CUDA_OK(cudaMemcpy(counts.data(), d_counts, vocab * 4, cudaMemcpyDeviceToHost));
  require(counts[1] == 2 && counts[7] == 3 && counts[499] == 1 && counts[0] == 1,
          "prompt counts");
  int total = 0;
  for (int c : counts) total += c;
  require(total == static_cast<int>(ids.size()), "no stray counts");
  cudaFree(d_counts);
  cudaFree(d_ids);
}


// The MTP T=2 verify on the device against the speculative host oracle
// (sample::spec_accept_from_prefix for row 0, sample_from_prefix for row 1)
// over a simulated world: a greedy request under the greedy judge, sampled
// requests whose drafts are accepted (then row 1 decides) or rejected (the
// residual, and the draft leaves the count table), and the two fallbacks —
// row 0 undecidable (a provisional reject with the counter untouched) and
// row 1 undecidable after an accept (u1 consumed, u2 reserved).
DGPP_TEST(sample_pick_t2_matches_spec_oracle_over_simulated_world) {
  Rng rng(0x7e2a);
  constexpr int kWorld = 4;
  constexpr int count = 96;
  constexpr int vocab = kWorld * count;
  constexpr int candidates = 32;
  constexpr int requests = 4;
  constexpr int rpr = 2;
  int accepts = 0, rejects = 0, fallback0 = 0, fallback1 = 0;
  for (int trial = 0; trial < 10; ++trial) {
    // Rows: request q's rows are 2q (row 0) and 2q+1 (row 1).
    std::vector<float> full(static_cast<size_t>(requests * rpr) * vocab);
    const auto fill = [&](int row, bool flat) {
      float* v = full.data() + static_cast<size_t>(row) * vocab;
      for (int i = 0; i < vocab; ++i) {
        const uint64_t r = rng.next();
        v[i] = flat ? static_cast<float>(r % 3) * 0.01f
                    : static_cast<float>((r >> 8) % 41) * 0.25f - 5.0f;
      }
      if (!flat) {
        // Two peaks holding ~97% of the mass: the nucleus resolves inside
        // 128 candidates and a draft at the argmax stands ~92% of the time.
        v[static_cast<size_t>(rng.next() % vocab)] = 12.0f;
        v[static_cast<size_t>(rng.next() % vocab)] = 9.0f;
      }
    };
    // 0: greedy. 1: sampled, row 0 peaked, row 1 flat on odd trials (the
    // row-1 fallback). 2: sampled, row 0 flat (the row-0 fallback). 3:
    // sampled, both peaked.
    fill(0, false); fill(1, false);
    fill(2, false); fill(3, trial % 2 == 1);
    fill(4, true);  fill(5, false);
    fill(6, false); fill(7, false);
    std::vector<dgpp::SampleSpec> specs(requests);
    specs[0].temperature = 0.0f;
    for (int q = 1; q < requests; ++q) {
      specs[q].temperature = 1.0f;
      specs[q].top_p = 0.95f;
      specs[q].presence_penalty = 0.1f;
      specs[q].seed = 0x2000 + q + 31 * trial;
      specs[q].counter = 4;
    }
    std::vector<int32_t> counts(static_cast<size_t>(requests) * vocab, 0);
    std::vector<int64_t> fed(requests * rpr), positions(requests);
    for (int q = 0; q < requests; ++q) {
      positions[q] = 5 + q;
      fed[2 * q] = static_cast<int64_t>(rng.next() % vocab);
      const Candidate argmax0 =
          dgpp::sample::local_max(full.data() + static_cast<size_t>(2 * q) * vocab, vocab, 0);
      // The draft: request 1's is always the row-0 argmax (so its odd
      // trials reach the flat row 1 and fall back there); requests 0 and 3
      // take the argmax on even trials and a random token otherwise
      // (accepts and rejects); request 2's row 0 is flat and falls back.
      const bool likely = q == 1 || (q != 2 && trial % 2 == 0);
      fed[2 * q + 1] = likely ? argmax0.id
                              : static_cast<int64_t>(rng.next() % vocab);
      for (int j = 0; j < 3; ++j)
        counts[static_cast<size_t>(q) * vocab + rng.next() % vocab] += 1;
    }
    const uint64_t carry = 0x3c3c3c3c3c3cull;
    const SampleWorldRun run = run_sample_world(
        full, requests, kWorld, count, specs, counts, fed, positions,
        candidates, carry, rpr);

    for (int q = 0; q < requests; ++q) {
      const float* row0 = full.data() + static_cast<size_t>(2 * q) * vocab;
      const float* row1 = full.data() + static_cast<size_t>(2 * q + 1) * vocab;
      const int32_t draft = static_cast<int32_t>(fed[2 * q + 1]);
      if (specs[q].temperature <= 0.0f) {
        const int32_t w0 = dgpp::sample::local_max(row0, vocab, 0).id;
        const int32_t w1 = dgpp::sample::local_max(row1, vocab, 0).id;
        const int accepted = w0 == draft ? 2 : 1;
        for (int k = 0; k < kWorld; ++k) {
          const PickVerdict& v = run.verdicts[k][q];
          require(v.rows == 2 && v.accepted == accepted && v.winners[0] == w0 &&
                      v.winners[1] == w1 && v.next == (accepted == 2 ? w1 : w0),
                  "greedy T=2 request: the greedy judge");
        }
        continue;
      }
      // The host oracle over the same inputs.
      const dgpp::sample::Params p = params_of(specs[q]);
      std::vector<int32_t> ctx0(counts.begin() + static_cast<long>(q) * vocab,
                                counts.begin() + static_cast<long>(q + 1) * vocab);
      ctx0[static_cast<size_t>(fed[2 * q])] += 1;
      std::vector<int32_t> ctx1 = ctx0;
      ctx1[static_cast<size_t>(draft)] += 1;
      const auto as_map = [&](const std::vector<int32_t>& c) {
        std::unordered_map<int32_t, int32_t> m;
        for (int v = 0; v < vocab; ++v)
          if (c[static_cast<size_t>(v)]) m[v] = c[static_cast<size_t>(v)];
        return m;
      };
      std::vector<float> adj0(row0, row0 + vocab), adj1(row1, row1 + vocab);
      dgpp::sample::apply_penalties(adj0.data(), vocab, 0, p, as_map(ctx0));
      dgpp::sample::apply_penalties(adj1.data(), vocab, 0, p, as_map(ctx1));
      const auto merged_of = [&](const std::vector<float>& adj, double* Z) {
        std::vector<std::vector<Candidate>> shards;
        std::vector<double> lses;
        for (int k = 0; k < kWorld; ++k) {
          shards.push_back(dgpp::sample::local_topk(adj.data() + k * count, count,
                                                        k * count, candidates));
          lses.push_back(dgpp::sample::slice_logsumexp(adj.data() + k * count,
                                                            count, p.temperature));
        }
        *Z = dgpp::sample::merge_logsumexp(lses);
        return dgpp::sample::merge_topk(shards, candidates);
      };
      double Z0 = 0.0, Z1 = 0.0;
      const std::vector<Candidate> m0 = merged_of(adj0, &Z0);
      const std::vector<Candidate> m1 = merged_of(adj1, &Z1);
      dgpp::sample::Rng host{specs[q].seed, specs[q].counter};
      const dgpp::sample::SpecPrefixDecision d0 =
          dgpp::sample::spec_accept_from_prefix(m0, vocab, Z0, draft, p, host);
      // Expected verdict/outcome and the count table after the step: the
      // consumed token always, the draft only when it stood (a provisional
      // reject leaves it to the host).
      int want_accepted = 1, want_fallback_row = -1;
      int32_t want_w0 = -1, want_w1 = -1;
      std::vector<int32_t> want_counts = ctx0;
      if (!d0.resolved) {
        ++fallback0;
        want_fallback_row = 0;
        want_w0 = m0[0].id;
      } else if (!d0.accepted) {
        ++rejects;
        want_w0 = d0.result.token;
      } else {
        ++accepts;
        want_accepted = 2;
        want_w0 = draft;
        want_counts = ctx1;
        const dgpp::sample::PrefixDecision d1 =
            dgpp::sample::sample_from_prefix(m1, vocab, Z1, p, host);
        if (d1.resolved) {
          want_w1 = d1.result.token;
        } else {
          ++fallback1;
          want_fallback_row = 1;
          want_w1 = m1[0].id;
        }
      }
      for (int k = 0; k < kWorld; ++k) {
        const PickVerdict& v = run.verdicts[k][q];
        const dgpp::SampleOutcome& o = run.outcomes[k][q];
        require(v.rows == 2 && v.accepted == want_accepted &&
                    v.winners[0] == want_w0 && v.winners[1] == want_w1 &&
                    v.next == (want_accepted == 2 ? want_w1 : want_w0),
                "T=2 verdict differs from the speculative oracle (trial " +
                    std::to_string(trial) + " request " + std::to_string(q) +
                    ": got accepted " + std::to_string(v.accepted) + " winners " +
                    std::to_string(v.winners[0]) + "/" + std::to_string(v.winners[1]) +
                    ", want " + std::to_string(want_accepted) + " " +
                    std::to_string(want_w0) + "/" + std::to_string(want_w1) + ")");
        require(o.sampled == 1 && o.fallback_row == want_fallback_row &&
                    o.fallback == (want_fallback_row >= 0 ? 1 : 0) &&
                    o.accepted_draft == (want_accepted == 2 ? 1 : 0),
                "T=2 outcome flags");
        require(o.counter == host.counter, "T=2 counter differs from the oracle");
        require(bits_equal(o.normalizer[0], Z0), "row-0 normalizer");
        if (want_accepted == 2) require(bits_equal(o.normalizer[1], Z1), "row-1 normalizer");
        if (d0.resolved) require(bits_equal(o.logprob[0], d0.result.logprob), "row-0 logprob");
        require(run.counts_after[k] .size() == counts.size(), "counts shape");
        for (int vtok = 0; vtok < vocab; ++vtok)
          require(run.counts_after[k][static_cast<size_t>(q) * vocab + vtok] ==
                      want_counts[static_cast<size_t>(vtok)],
                  "count table after the T=2 verdict differs");
        require(v.digest_mismatch == 0 && run.carry_out[k] == v.digest, "digest chain");
      }
    }
  }
  require(accepts > 0 && rejects > 0 && fallback0 > 0 && fallback1 > 0,
          "the sweep must exercise every outcome (accepts " +
              std::to_string(accepts) + ", rejects " + std::to_string(rejects) +
              ", row-0 fallbacks " + std::to_string(fallback0) +
              ", row-1 fallbacks " + std::to_string(fallback1) + ")");
}


// The T=3 verify's sampled verdict (2026-09-06, depth 2): row 0 tests the
// first draft, row 1 the second, row 2 samples plainly — each row's
// decision the host oracle's (sample::spec_accept_from_prefix, then
// sample_from_prefix), draw for draw; a fallback at any row leaves the
// rows before it committed; the count table takes every draft that stood.
DGPP_TEST(sample_pick_t3_matches_spec_oracle_over_simulated_world) {
  Rng rng(0x51d3);
  constexpr int kWorld = 4;
  constexpr int count = 96;
  constexpr int vocab = kWorld * count;
  constexpr int candidates = 32;
  constexpr int requests = 2;  // 2 x 3 rows inside the pick's row bound
  constexpr int rpr = 3;
  int accepts3 = 0, rejects0 = 0, rejects1 = 0;
  int fallback0 = 0, fallback1 = 0, fallback2 = 0;
  for (int trial = 0; trial < 24; ++trial) {
    std::vector<float> full(static_cast<size_t>(requests * rpr) * vocab);
    // Request q takes pattern (trial + q) % 4 this trial, so every pattern
    // runs in both slots: 0 greedy; 1 sampled with row 2 flat on odd trials
    // (the row-2 fallback after both drafts stood); 2 sampled with row 0
    // flat (the row-0 fallback); 3 sampled in three sub-cases by trial / 4:
    // row 1 flat (the row-1 fallback), the second draft random (the row-1
    // reject), the first draft random (the row-0 reject).
    const auto pattern_of = [&](int q) { return (trial + q) % 4; };
    const int sub = (trial / 4) % 3;
    const auto fill = [&](int row, bool flat) {
      float* v = full.data() + static_cast<size_t>(row) * vocab;
      for (int i = 0; i < vocab; ++i) {
        const uint64_t r = rng.next();
        v[i] = flat ? static_cast<float>(r % 3) * 0.01f
                    : static_cast<float>((r >> 8) % 41) * 0.25f - 5.0f;
      }
      if (!flat) {
        v[static_cast<size_t>(rng.next() % vocab)] = 12.0f;
        v[static_cast<size_t>(rng.next() % vocab)] = 9.0f;
      }
    };
    for (int q = 0; q < requests; ++q) {
      const int pat = pattern_of(q);
      fill(rpr * q, pat == 2);
      fill(rpr * q + 1, pat == 3 && sub == 0);
      fill(rpr * q + 2, pat == 1 && trial % 2 == 1);
    }
    std::vector<dgpp::SampleSpec> specs(requests);
    for (int q = 0; q < requests; ++q) {
      if (pattern_of(q) == 0) {
        specs[q].temperature = 0.0f;
        continue;
      }
      specs[q].temperature = 1.0f;
      specs[q].top_p = 0.95f;
      specs[q].presence_penalty = 0.1f;
      specs[q].seed = 0x3000 + q + 37 * trial;
      specs[q].counter = 6;
    }
    std::vector<int32_t> counts(static_cast<size_t>(requests) * vocab, 0);
    std::vector<int64_t> fed(requests * rpr), positions(requests);
    for (int q = 0; q < requests; ++q) {
      positions[q] = 5 + q;
      fed[rpr * q] = static_cast<int64_t>(rng.next() % vocab);
      const Candidate argmax0 = dgpp::sample::local_max(
          full.data() + static_cast<size_t>(rpr * q) * vocab, vocab, 0);
      const Candidate argmax1 = dgpp::sample::local_max(
          full.data() + static_cast<size_t>(rpr * q + 1) * vocab, vocab, 0);
      const int pat = pattern_of(q);
      const bool likely1 = !(pat == 0 && trial % 2 == 1) && !(pat == 3 && sub == 2);
      const bool likely2 = !(pat == 3 && sub == 1);
      fed[rpr * q + 1] = likely1 ? argmax0.id : static_cast<int64_t>(rng.next() % vocab);
      fed[rpr * q + 2] = likely2 ? argmax1.id : static_cast<int64_t>(rng.next() % vocab);
      for (int j = 0; j < 3; ++j)
        counts[static_cast<size_t>(q) * vocab + rng.next() % vocab] += 1;
    }
    const uint64_t carry = 0x5a5a5a5a5a5aull;
    const SampleWorldRun run = run_sample_world(
        full, requests, kWorld, count, specs, counts, fed, positions,
        candidates, carry, rpr);

    for (int q = 0; q < requests; ++q) {
      const float* row[rpr];
      for (int t = 0; t < rpr; ++t)
        row[t] = full.data() + static_cast<size_t>(rpr * q + t) * vocab;
      const int32_t draft1 = static_cast<int32_t>(fed[rpr * q + 1]);
      const int32_t draft2 = static_cast<int32_t>(fed[rpr * q + 2]);
      if (specs[q].temperature <= 0.0f) {
        int32_t w[rpr];
        for (int t = 0; t < rpr; ++t) w[t] = dgpp::sample::local_max(row[t], vocab, 0).id;
        int accepted = 1;
        while (accepted < rpr && w[accepted - 1] == fed[rpr * q + accepted]) ++accepted;
        for (int k = 0; k < kWorld; ++k) {
          const PickVerdict& v = run.verdicts[k][q];
          require(v.rows == rpr && v.accepted == accepted && v.winners[0] == w[0] &&
                      v.winners[1] == w[1] && v.winners[2] == w[2] &&
                      v.next == w[accepted - 1],
                  "greedy T=3 request: the greedy judge");
        }
        continue;
      }
      const dgpp::sample::Params p = params_of(specs[q]);
      std::vector<int32_t> ctx[rpr];
      ctx[0].assign(counts.begin() + static_cast<long>(q) * vocab,
                    counts.begin() + static_cast<long>(q + 1) * vocab);
      ctx[0][static_cast<size_t>(fed[rpr * q])] += 1;
      ctx[1] = ctx[0];
      ctx[1][static_cast<size_t>(draft1)] += 1;
      ctx[2] = ctx[1];
      ctx[2][static_cast<size_t>(draft2)] += 1;
      const auto as_map = [&](const std::vector<int32_t>& c) {
        std::unordered_map<int32_t, int32_t> m;
        for (int v = 0; v < vocab; ++v)
          if (c[static_cast<size_t>(v)]) m[v] = c[static_cast<size_t>(v)];
        return m;
      };
      std::vector<Candidate> merged[rpr];
      double Z[rpr] = {0.0, 0.0, 0.0};
      for (int t = 0; t < rpr; ++t) {
        std::vector<float> adj(row[t], row[t] + vocab);
        dgpp::sample::apply_penalties(adj.data(), vocab, 0, p, as_map(ctx[t]));
        std::vector<std::vector<Candidate>> shards;
        std::vector<double> lses;
        for (int k = 0; k < kWorld; ++k) {
          shards.push_back(dgpp::sample::local_topk(adj.data() + k * count, count,
                                                        k * count, candidates));
          lses.push_back(dgpp::sample::slice_logsumexp(adj.data() + k * count,
                                                            count, p.temperature));
        }
        Z[t] = dgpp::sample::merge_logsumexp(lses);
        merged[t] = dgpp::sample::merge_topk(shards, candidates);
      }
      dgpp::sample::Rng host{specs[q].seed, specs[q].counter};
      // The oracle's chain.
      int want_accepted = 1, want_fallback_row = -1, reached = 1;
      int32_t want_w[rpr] = {-1, -1, -1};
      std::vector<int32_t> want_counts = ctx[0];
      const dgpp::sample::SpecPrefixDecision d0 =
          dgpp::sample::spec_accept_from_prefix(merged[0], vocab, Z[0], draft1, p, host);
      if (!d0.resolved) {
        ++fallback0;
        want_fallback_row = 0;
        want_w[0] = merged[0][0].id;
      } else if (!d0.accepted) {
        ++rejects0;
        want_w[0] = d0.result.token;
      } else {
        want_accepted = 2;
        want_w[0] = draft1;
        want_counts = ctx[1];
        reached = 2;
        const dgpp::sample::SpecPrefixDecision d1 =
            dgpp::sample::spec_accept_from_prefix(merged[1], vocab, Z[1], draft2, p, host);
        if (!d1.resolved) {
          ++fallback1;
          want_fallback_row = 1;
          want_w[1] = merged[1][0].id;
        } else if (!d1.accepted) {
          ++rejects1;
          want_w[1] = d1.result.token;
        } else {
          want_accepted = 3;
          want_w[1] = draft2;
          want_counts = ctx[2];
          reached = 3;
          const dgpp::sample::PrefixDecision d2 =
              dgpp::sample::sample_from_prefix(merged[2], vocab, Z[2], p, host);
          if (d2.resolved) {
            ++accepts3;
            want_w[2] = d2.result.token;
          } else {
            ++fallback2;
            want_fallback_row = 2;
            want_w[2] = merged[2][0].id;
          }
        }
      }
      const int32_t want_next = want_w[want_accepted - 1];
      for (int k = 0; k < kWorld; ++k) {
        const PickVerdict& v = run.verdicts[k][q];
        const dgpp::SampleOutcome& o = run.outcomes[k][q];
        std::string got = std::to_string(v.accepted) + " [";
        std::string want = std::to_string(want_accepted) + " [";
        bool same = v.rows == rpr && v.accepted == want_accepted && v.next == want_next;
        for (int t = 0; t < want_accepted; ++t) {
          got += (t ? "," : "") + std::to_string(v.winners[t]);
          want += (t ? "," : "") + std::to_string(want_w[t]);
          same = same && v.winners[t] == want_w[t];
        }
        require(same, "T=3 verdict differs from the speculative oracle (trial " +
                          std::to_string(trial) + " request " + std::to_string(q) +
                          ": got " + got + "] next " + std::to_string(v.next) +
                          ", want " + want + "] next " + std::to_string(want_next) + ")");
        require(o.sampled == 1 && o.fallback_row == want_fallback_row &&
                    o.fallback == (want_fallback_row >= 0 ? 1 : 0) &&
                    o.accepted_draft == (want_accepted >= 2 ? 1 : 0),
                "T=3 outcome flags (trial " + std::to_string(trial) + " request " +
                    std::to_string(q) + ")");
        require(o.counter == host.counter, "T=3 counter differs from the oracle");
        for (int t = 0; t < reached; ++t)
          require(bits_equal(o.normalizer[t], Z[t]),
                  "row-" + std::to_string(t) + " normalizer");
        if (d0.resolved) require(bits_equal(o.logprob[0], d0.result.logprob), "row-0 logprob");
        for (int vtok = 0; vtok < vocab; ++vtok)
          require(run.counts_after[k][static_cast<size_t>(q) * vocab + vtok] ==
                      want_counts[static_cast<size_t>(vtok)],
                  "count table after the T=3 verdict differs");
        require(v.digest_mismatch == 0 && run.carry_out[k] == v.digest, "digest chain");
      }
    }
  }
  require(accepts3 > 0 && rejects0 > 0 && rejects1 > 0 && fallback0 > 0 &&
              fallback1 > 0 && fallback2 > 0,
          "the sweep must exercise every outcome (accept-all " +
              std::to_string(accepts3) + ", row-0 rejects " + std::to_string(rejects0) +
              ", row-1 rejects " + std::to_string(rejects1) + ", fallbacks " +
              std::to_string(fallback0) + "/" + std::to_string(fallback1) + "/" +
              std::to_string(fallback2) + ")");
}


// The full verify block: kSampleVerdictRows rows per request (the DSpark
// block of five drafts, 2026-09-14 — the kernel's per-row tables in dynamic
// shared memory) against the host's speculative chain — every draft row's
// accept test in turn, the last row sampled plainly — with the greedy
// judge, every reject row, the fallbacks and the accept-all outcome
// exercised over the sweep; bitwise on every rank.
template<int requests, int rpr>
void check_sample_full_block() {
  Rng rng(0x51d6);
  constexpr int kWorld = 4;
  constexpr int count = 96;
  constexpr int vocab = kWorld * count;
  constexpr int candidates = 32;
  static_assert(requests * rpr <= dgpp::kPickMaxRows, "the block fits the pick's row bound");
  int accepts_all = 0, fallbacks = 0, greedy = 0;
  int rejects[rpr] = {};
  for (int trial = 0; trial < 40; ++trial) {
    std::vector<float> full(static_cast<size_t>(requests * rpr) * vocab);
    // Request q's pattern this trial: 0 greedy (every draft the argmax);
    // 1 every draft likely and one flat row (the fallback there);
    // 2 a random draft at row j = trial % (rpr - 1) (the reject at j);
    // 3 every draft likely, no flat row (the accept-all path to the
    // sampled last row).
    const auto pattern_of = [&](int q) { return (trial + q) % 4; };
    const int reject_row = trial % (rpr - 1);
    const int flat_row = (trial / (rpr - 1)) % rpr;
    const auto fill = [&](int row, bool flat) {
      float* v = full.data() + static_cast<size_t>(row) * vocab;
      for (int i = 0; i < vocab; ++i) {
        const uint64_t r = rng.next();
        v[i] = flat ? static_cast<float>(r % 3) * 0.01f
                    : static_cast<float>((r >> 8) % 41) * 0.25f - 5.0f;
      }
      if (!flat) {
        v[static_cast<size_t>(rng.next() % vocab)] = 12.0f;
        v[static_cast<size_t>(rng.next() % vocab)] = 9.0f;
      }
    };
    for (int q = 0; q < requests; ++q)
      for (int t = 0; t < rpr; ++t) fill(rpr * q + t, pattern_of(q) == 1 && t == flat_row);
    std::vector<dgpp::SampleSpec> specs(requests);
    for (int q = 0; q < requests; ++q) {
      if (pattern_of(q) == 0) {
        specs[q].temperature = 0.0f;
        continue;
      }
      specs[q].temperature = 1.0f;
      specs[q].top_p = 0.95f;
      specs[q].presence_penalty = 0.1f;
      specs[q].seed = 0x6000 + q + 37 * trial;
      specs[q].counter = 6;
    }
    std::vector<int32_t> counts(static_cast<size_t>(requests) * vocab, 0);
    std::vector<int64_t> fed(requests * rpr), positions(requests);
    for (int q = 0; q < requests; ++q) {
      positions[q] = 5 + q;
      fed[rpr * q] = static_cast<int64_t>(rng.next() % vocab);
      for (int t = 1; t < rpr; ++t) {
        const Candidate argmax = dgpp::sample::local_max(
            full.data() + static_cast<size_t>(rpr * q + t - 1) * vocab, vocab, 0);
        const bool random = pattern_of(q) == 2 && t - 1 == reject_row;
        fed[rpr * q + t] = random ? static_cast<int64_t>(rng.next() % vocab) : argmax.id;
      }
      for (int j = 0; j < 3; ++j)
        counts[static_cast<size_t>(q) * vocab + rng.next() % vocab] += 1;
    }
    const uint64_t carry = 0x6a6a6a6a6a6aull;
    const SampleWorldRun run = run_sample_world(
        full, requests, kWorld, count, specs, counts, fed, positions,
        candidates, carry, rpr);

    for (int q = 0; q < requests; ++q) {
      const float* row[rpr];
      for (int t = 0; t < rpr; ++t)
        row[t] = full.data() + static_cast<size_t>(rpr * q + t) * vocab;
      int32_t draft[rpr];  // draft[t]: the token fed to row t (t >= 1)
      for (int t = 1; t < rpr; ++t) draft[t] = static_cast<int32_t>(fed[rpr * q + t]);
      if (specs[q].temperature <= 0.0f) {
        ++greedy;
        int32_t w[rpr];
        for (int t = 0; t < rpr; ++t) w[t] = dgpp::sample::local_max(row[t], vocab, 0).id;
        int accepted = 1;
        while (accepted < rpr && w[accepted - 1] == fed[rpr * q + accepted]) ++accepted;
        for (int k = 0; k < kWorld; ++k) {
          const PickVerdict& v = run.verdicts[k][q];
          bool same = v.rows == rpr && v.accepted == accepted && v.next == w[accepted - 1];
          for (int t = 0; t < rpr; ++t) same = same && v.winners[t] == w[t];
          require(same, "greedy full-block request: the greedy judge");
        }
        continue;
      }
      const dgpp::sample::Params p = params_of(specs[q]);
      std::vector<int32_t> ctx[rpr];
      ctx[0].assign(counts.begin() + static_cast<long>(q) * vocab,
                    counts.begin() + static_cast<long>(q + 1) * vocab);
      ctx[0][static_cast<size_t>(fed[rpr * q])] += 1;
      for (int t = 1; t < rpr; ++t) {
        ctx[t] = ctx[t - 1];
        ctx[t][static_cast<size_t>(draft[t])] += 1;
      }
      const auto as_map = [&](const std::vector<int32_t>& c) {
        std::unordered_map<int32_t, int32_t> m;
        for (int v = 0; v < vocab; ++v)
          if (c[static_cast<size_t>(v)]) m[v] = c[static_cast<size_t>(v)];
        return m;
      };
      std::vector<Candidate> merged[rpr];
      double Z[rpr] = {};
      for (int t = 0; t < rpr; ++t) {
        std::vector<float> adj(row[t], row[t] + vocab);
        dgpp::sample::apply_penalties(adj.data(), vocab, 0, p, as_map(ctx[t]));
        std::vector<std::vector<Candidate>> shards;
        std::vector<double> lses;
        for (int k = 0; k < kWorld; ++k) {
          shards.push_back(dgpp::sample::local_topk(adj.data() + k * count, count,
                                                        k * count, candidates));
          lses.push_back(dgpp::sample::slice_logsumexp(adj.data() + k * count,
                                                            count, p.temperature));
        }
        Z[t] = dgpp::sample::merge_logsumexp(lses);
        merged[t] = dgpp::sample::merge_topk(shards, candidates);
      }
      dgpp::sample::Rng host{specs[q].seed, specs[q].counter};
      // The oracle's chain: row t tests draft[t + 1]; the last row samples.
      int want_accepted = 1, want_fallback_row = -1, reached = 1;
      int32_t want_w[rpr];
      for (int t = 0; t < rpr; ++t) want_w[t] = -1;
      std::vector<int32_t> want_counts = ctx[0];
      bool row0_logprob_known = false;
      float row0_logprob = 0.0f;
      for (int t = 0; t < rpr; ++t) {
        if (t + 1 < rpr) {
          const dgpp::sample::SpecPrefixDecision d =
              dgpp::sample::spec_accept_from_prefix(merged[t], vocab, Z[t], draft[t + 1], p, host);
          if (t == 0 && d.resolved) {
            row0_logprob_known = true;
            row0_logprob = d.result.logprob;
          }
          if (!d.resolved) {
            ++fallbacks;
            want_fallback_row = t;
            want_w[t] = merged[t][0].id;
            break;
          }
          if (!d.accepted) {
            ++rejects[t];
            want_w[t] = d.result.token;
            break;
          }
          want_accepted = t + 2;
          want_w[t] = draft[t + 1];
          want_counts = ctx[t + 1];
          reached = t + 2;
        } else {
          const dgpp::sample::PrefixDecision d =
              dgpp::sample::sample_from_prefix(merged[t], vocab, Z[t], p, host);
          if (d.resolved) {
            ++accepts_all;
            want_w[t] = d.result.token;
          } else {
            ++fallbacks;
            want_fallback_row = t;
            want_w[t] = merged[t][0].id;
          }
        }
      }
      const int32_t want_next = want_w[want_accepted - 1];
      for (int k = 0; k < kWorld; ++k) {
        const PickVerdict& v = run.verdicts[k][q];
        const dgpp::SampleOutcome& o = run.outcomes[k][q];
        std::string got = std::to_string(v.accepted) + " [";
        std::string want = std::to_string(want_accepted) + " [";
        bool same = v.rows == rpr && v.accepted == want_accepted && v.next == want_next;
        for (int t = 0; t < want_accepted; ++t) {
          got += (t ? "," : "") + std::to_string(v.winners[t]);
          want += (t ? "," : "") + std::to_string(want_w[t]);
          same = same && v.winners[t] == want_w[t];
        }
        require(same, "full-block verdict differs from the speculative oracle (trial " +
                          std::to_string(trial) + " request " + std::to_string(q) +
                          ": got " + got + "] next " + std::to_string(v.next) +
                          ", want " + want + "] next " + std::to_string(want_next) + ")");
        require(o.sampled == 1 && o.fallback_row == want_fallback_row &&
                    o.fallback == (want_fallback_row >= 0 ? 1 : 0) &&
                    o.accepted_draft == (want_accepted >= 2 ? 1 : 0),
                "full-block outcome flags (trial " + std::to_string(trial) + " request " +
                    std::to_string(q) + ")");
        require(o.counter == host.counter, "full-block counter differs from the oracle");
        for (int t = 0; t < reached; ++t)
          require(bits_equal(o.normalizer[t], Z[t]), "row-" + std::to_string(t) + " normalizer");
        if (row0_logprob_known) require(bits_equal(o.logprob[0], row0_logprob), "row-0 logprob");
        for (int vtok = 0; vtok < vocab; ++vtok)
          require(run.counts_after[k][static_cast<size_t>(q) * vocab + vtok] ==
                      want_counts[static_cast<size_t>(vtok)],
                  "count table after the full-block verdict differs");
        require(v.digest_mismatch == 0 && run.carry_out[k] == v.digest, "digest chain");
      }
    }
  }
  std::string reject_text;
  bool every_reject_row = true;
  for (int t = 0; t + 1 < rpr; ++t) {
    reject_text += (t ? "/" : "") + std::to_string(rejects[t]);
    if (rejects[t] == 0) every_reject_row = false;
  }
  require(greedy > 0 && accepts_all > 0 && fallbacks > 0 && every_reject_row,
          "the sweep must exercise every outcome (greedy " + std::to_string(greedy) +
              ", accept-all " + std::to_string(accepts_all) + ", rejects per row " + reject_text +
              ", fallbacks " + std::to_string(fallbacks) + ")");
}

// Logprobs on the device: a greedy request that reports takes the full path
// at temperature 1 (penalties applied) and reports the argmax under the raw
// normalizer with its top-N — sample::greedy_from_prefix — and a
// sampled request's report is its Result's top_logprobs; both bitwise.
DGPP_TEST(sample_pick_full_block_matches_spec_oracle_over_simulated_world) {
  check_sample_full_block<2, dgpp::kSampleVerdictRows>();
}
DGPP_TEST(sample_pick_rows64_mtp3_matches_spec_oracle_over_simulated_world) {
  check_sample_full_block<16, 4>();
}

DGPP_TEST(sample_pick_reports_logprobs_bitwise) {
  Rng rng(0x10b5);
  constexpr int kWorld = 4;
  constexpr int count = 96;
  constexpr int vocab = kWorld * count;
  constexpr int candidates = 32;
  constexpr int requests = 3;
  for (int trial = 0; trial < 4; ++trial) {
    std::vector<float> full(static_cast<size_t>(requests) * vocab);
    for (int q = 0; q < requests; ++q) {
      float* row = full.data() + static_cast<size_t>(q) * vocab;
      for (int v = 0; v < vocab; ++v)
        row[v] = static_cast<float>((rng.next() >> 8) % 41) * 0.25f - 5.0f;
      row[static_cast<size_t>(rng.next() % vocab)] = 11.0f;
      row[static_cast<size_t>(rng.next() % vocab)] = 8.5f;
    }
    std::vector<dgpp::SampleSpec> specs(requests);
    // 0: greedy with logprobs and a penalty; 1: sampled with top-4; 2:
    // pure temperature with top-3.
    specs[0].temperature = 0.0f;
    specs[0].logprobs = 3;
    specs[0].presence_penalty = 0.2f;
    specs[1].temperature = 1.0f;
    specs[1].top_p = 0.95f;
    specs[1].logprobs = 4;
    specs[2].temperature = 0.8f;
    specs[2].top_p = 1.0f;
    specs[2].logprobs = 3;
    for (int q = 0; q < requests; ++q) {
      specs[q].seed = 0x3000 + q + 11 * trial;
      specs[q].counter = 2;
    }
    std::vector<int32_t> counts(static_cast<size_t>(requests) * vocab, 0);
    std::vector<int64_t> fed(requests), positions(requests, 7);
    for (int q = 0; q < requests; ++q) {
      fed[q] = static_cast<int64_t>(rng.next() % vocab);
      counts[static_cast<size_t>(q) * vocab + rng.next() % vocab] += 1;
    }
    const SampleWorldRun run = run_sample_world(full, requests, kWorld, count,
                                                specs, counts, fed, positions,
                                                candidates, 0x1111ull);
    for (int q = 0; q < requests; ++q) {
      const float* row = full.data() + static_cast<size_t>(q) * vocab;
      std::vector<int32_t> ctx(counts.begin() + static_cast<long>(q) * vocab,
                               counts.begin() + static_cast<long>(q + 1) * vocab);
      ctx[static_cast<size_t>(fed[q])] += 1;
      std::unordered_map<int32_t, int32_t> m;
      for (int v = 0; v < vocab; ++v)
        if (ctx[static_cast<size_t>(v)]) m[v] = ctx[static_cast<size_t>(v)];
      dgpp::sample::Params p = params_of(specs[q]);
      p.logprobs = specs[q].logprobs;
      std::vector<float> adj(row, row + vocab);
      dgpp::sample::apply_penalties(adj.data(), vocab, 0, p, m);
      const float T = p.temperature > 0.0f ? p.temperature : 1.0f;
      std::vector<std::vector<Candidate>> shards;
      std::vector<double> lses;
      for (int k = 0; k < kWorld; ++k) {
        shards.push_back(dgpp::sample::local_topk(adj.data() + k * count, count,
                                                      k * count, candidates));
        lses.push_back(dgpp::sample::slice_logsumexp(adj.data() + k * count, count, T));
      }
      const std::vector<Candidate> merged = dgpp::sample::merge_topk(shards, candidates);
      const double Z = dgpp::sample::merge_logsumexp(lses);
      dgpp::sample::Result want;
      bool want_resolved = true;
      if (p.temperature <= 0.0f) {
        want = dgpp::sample::greedy_from_prefix(merged, Z, p.logprobs);
      } else {
        dgpp::sample::Rng host{specs[q].seed, specs[q].counter};
        const auto d = dgpp::sample::sample_from_prefix(merged, vocab, Z, p, host);
        want_resolved = d.resolved;
        want = d.result;
      }
      for (int k = 0; k < kWorld; ++k) {
        const PickVerdict& v = run.verdicts[k][q];
        const dgpp::SampleOutcome& o = run.outcomes[k][q];
        if (!want_resolved) {
          require(o.fallback == 1, "the host fell back, the device must too");
          continue;
        }
        require(v.next == want.token, "reported token differs (request " +
                                          std::to_string(q) + ")");
        require(bits_equal(o.logprob[0], want.logprob),
                "the token's logprob differs (request " + std::to_string(q) + ")");
        require(o.top_count[0] == static_cast<int>(want.top_logprobs.size()),
                "top-N count differs (request " + std::to_string(q) + "): " +
                    std::to_string(o.top_count[0]) + " vs " +
                    std::to_string(want.top_logprobs.size()));
        for (int i = 0; i < o.top_count[0]; ++i)
          require(o.top_ids[0][i] == want.top_logprobs[static_cast<size_t>(i)].first &&
                      bits_equal(o.top_logprobs[0][i],
                                 want.top_logprobs[static_cast<size_t>(i)].second),
                  "top-N entry differs (request " + std::to_string(q) + ")");
        if (q == 0) {
          // The greedy row penalized in place at temperature 1.
          for (int i = 0; i < count; ++i)
            require(bits_equal(run.penalized[k][static_cast<size_t>(q) * count + i],
                               adj[static_cast<size_t>(k * count + i)]),
                    "greedy-with-logprobs row must be penalized in place");
        }
      }
    }
  }
}

// The sampled draft's proposal: when the verify is handed the
// distribution its draft was drawn from, the device's row-0 decision is the
// host oracle's min(1, P/Q) accept and (P - Q)+ residual, bit for bit — and
// a proposal whose token is not the fed draft is ignored (the deterministic
// rule), as a re-drafted row needs.
DGPP_TEST(sample_pick_t2_with_a_proposal_matches_the_ratio_oracle) {
  Rng rng(0x51de);
  constexpr int kWorld = 4;
  constexpr int count = 96;
  constexpr int vocab = kWorld * count;
  constexpr int candidates = 32;
  constexpr int requests = 3;
  constexpr int rpr = 2;
  int accepts = 0, rejects = 0, ignored = 0;
  for (int trial = 0; trial < 12; ++trial) {
    std::vector<float> full(static_cast<size_t>(requests * rpr) * vocab);
    for (int row = 0; row < requests * rpr; ++row) {
      float* v = full.data() + static_cast<size_t>(row) * vocab;
      for (int i = 0; i < vocab; ++i)
        v[i] = static_cast<float>((rng.next() >> 8) % 41) * 0.25f - 5.0f;
      v[static_cast<size_t>(rng.next() % vocab)] = 12.0f;
      v[static_cast<size_t>(rng.next() % vocab)] = 9.0f;
    }
    std::vector<dgpp::SampleSpec> specs(requests);
    for (int q = 0; q < requests; ++q) {
      specs[q].temperature = 1.0f;
      specs[q].top_p = 0.95f;
      specs[q].seed = 0x5100 + q + 41 * trial;
      specs[q].counter = 3;
    }
    std::vector<int32_t> counts(static_cast<size_t>(requests) * vocab, 0);
    std::vector<int64_t> fed(requests * rpr), positions(requests);
    // Every request's draft is a token near the top of row 0 (so accepts
    // and rejects both occur); request 2's proposal names a DIFFERENT token
    // and must therefore be ignored.
    std::vector<dgpp::DraftProposal> props(
        static_cast<size_t>(requests) * dgpp::kSampleProposalSlots);
    std::vector<dgpp::sample::Proposal> host_props(requests);
    for (int q = 0; q < requests; ++q) {
      positions[q] = 9 + q;
      fed[2 * q] = static_cast<int64_t>(rng.next() % vocab);
      const float* row0 = full.data() + static_cast<size_t>(2 * q) * vocab;
      const std::vector<Candidate> top =
          dgpp::sample::local_topk(row0, vocab, 0, 8);
      const int pick = static_cast<int>(rng.next() % 4);
      fed[2 * q + 1] = top[static_cast<size_t>(pick)].id;
      // A proposal over the row's own top-8, softmaxed at temperature 1 —
      // any distribution is legal, and this one resembles P.
      dgpp::DraftProposal& dp = props[static_cast<size_t>(q) * dgpp::kSampleProposalSlots];
      double z = 0.0;
      for (const Candidate& c : top) z += std::exp(static_cast<double>(c.logit) - top[0].logit);
      dp.n = static_cast<int32_t>(top.size());
      for (size_t i = 0; i < top.size(); ++i) {
        dp.ids[i] = top[i].id;
        dp.mass[i] = static_cast<float>(
            std::exp(static_cast<double>(top[i].logit) - top[0].logit) / z);
      }
      dp.token = q == 2 ? static_cast<int32_t>((fed[2 * q + 1] + 1) % vocab)
                        : static_cast<int32_t>(fed[2 * q + 1]);
      if (q != 2) {
        for (int i = 0; i < dp.n; ++i)
          host_props[static_cast<size_t>(q)].mass.emplace_back(dp.ids[i], dp.mass[i]);
      }
    }
    const uint64_t carry = 0x515151515151ull;
    const SampleWorldRun run =
        run_sample_world(full, requests, kWorld, count, specs, counts, fed,
                         positions, candidates, carry, rpr, {}, props);
    for (int q = 0; q < requests; ++q) {
      const float* row0 = full.data() + static_cast<size_t>(2 * q) * vocab;
      const int32_t draft = static_cast<int32_t>(fed[2 * q + 1]);
      const dgpp::sample::Params p = params_of(specs[q]);
      std::vector<std::vector<Candidate>> shards;
      std::vector<double> lses;
      for (int k = 0; k < kWorld; ++k) {
        shards.push_back(dgpp::sample::local_topk(row0 + k * count, count,
                                                  k * count, candidates));
        lses.push_back(dgpp::sample::slice_logsumexp(row0 + k * count, count,
                                                     p.temperature));
      }
      const double Z0 = dgpp::sample::merge_logsumexp(lses);
      const std::vector<Candidate> m0 =
          dgpp::sample::merge_topk(shards, candidates);
      dgpp::sample::Rng host{specs[q].seed, specs[q].counter};
      const bool live = q != 2;
      const dgpp::sample::SpecPrefixDecision d0 =
          dgpp::sample::spec_accept_from_prefix(
              m0, vocab, Z0, draft, p, host, /*draft_excluded=*/false,
              live ? &host_props[static_cast<size_t>(q)] : nullptr);
      require(d0.resolved, "the peaked row resolves inside the prefix");
      if (!live) ++ignored;
      else if (d0.accepted) ++accepts;
      else ++rejects;
      if (d0.accepted) {
        // The stood draft moves the step to row 1, sampled plainly — the
        // draw the device makes next.
        const float* row1 = full.data() + static_cast<size_t>(2 * q + 1) * vocab;
        std::vector<std::vector<Candidate>> sh1;
        std::vector<double> ls1;
        for (int k = 0; k < kWorld; ++k) {
          sh1.push_back(dgpp::sample::local_topk(row1 + k * count, count,
                                                 k * count, candidates));
          ls1.push_back(dgpp::sample::slice_logsumexp(row1 + k * count, count,
                                                      p.temperature));
        }
        const double Z1 = dgpp::sample::merge_logsumexp(ls1);
        const std::vector<Candidate> m1 =
            dgpp::sample::merge_topk(sh1, candidates);
        const dgpp::sample::PrefixDecision d1 =
            dgpp::sample::sample_from_prefix(m1, vocab, Z1, p, host);
        require(d1.resolved, "row 1 resolves");
      }
      for (int k = 0; k < kWorld; ++k) {
        const PickVerdict& v = run.verdicts[k][q];
        const int want_accepted = d0.accepted ? 2 : 1;
        require(v.accepted == want_accepted,
                "rank " + std::to_string(k) + " request " + std::to_string(q) +
                    ": the ratio rule's accept");
        require(v.winners[0] == (d0.accepted ? draft : d0.result.token),
                "row 0's token is the oracle's");
        require(run.specs_after[k][q].counter == host.counter,
                "the draws the device consumed are the oracle's");
      }
    }
  }
  require(accepts > 0 && rejects > 0 && ignored > 0,
          "the sweep must accept, reject and ignore a mismatched proposal "
          "(accepts " + std::to_string(accepts) + ", rejects " +
          std::to_string(rejects) + ", ignored " + std::to_string(ignored) + ")");
}
