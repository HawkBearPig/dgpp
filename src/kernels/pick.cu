#include "kernels/pick.hpp"
#include "kernels/pdl.cuh"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"

namespace dgpp {

namespace {

constexpr int kLocalThreads = 1024;
constexpr int kWarp = 32;
constexpr int32_t kNoId = 0x7fffffff;  // sorts after every real id

struct Pair {
  float v;
  int32_t id;
};

// sample::candidate_before, on the device.
__host__ __device__ inline bool before(Pair a, Pair b) {
  if (a.v != b.v) return a.v > b.v;
  return a.id < b.id;
}

struct Top2 {
  Pair a;  // the row's argmax
  Pair b;  // the runner-up
};

__host__ __device__ inline Top2 empty_top2() {
  const Pair none{-INFINITY, kNoId};
  return Top2{none, none};
}

__host__ __device__ inline void push(Top2& t, Pair p) {
  if (before(p, t.a)) {
    t.b = t.a;
    t.a = p;
  } else if (before(p, t.b)) {
    t.b = p;
  }
}

__device__ inline Top2 merge(Top2 x, const Top2& y) {
  push(x, y.a);
  push(x, y.b);
  return x;
}

// xor butterfly, not shfl_down: every lane pairs with a DISTINCT lane at
// every step, so no Top2 is ever merged with itself (a self-merge would
// duplicate the argmax into the runner-up slot, and ids are unique across
// lanes only as long as no lane's pair is its own copy).
__device__ inline Top2 shfl_xor(const Top2& t, int mask) {
  Top2 r;
  r.a.v = __shfl_xor_sync(0xffffffffu, t.a.v, mask);
  r.a.id = __shfl_xor_sync(0xffffffffu, t.a.id, mask);
  r.b.v = __shfl_xor_sync(0xffffffffu, t.b.v, mask);
  r.b.id = __shfl_xor_sync(0xffffffffu, t.b.id, mask);
  return r;
}

__device__ inline void encode_digits(uint16_t* slots, uint64_t value,
                                     int digits) {
  for (int d = 0; d < digits; ++d)
    slots[d] = static_cast<uint16_t>((value >> (6 * d)) & 63);
}

__device__ inline uint64_t decode_digits(const uint16_t* slots, int digits) {
  uint64_t value = 0;
  for (int d = 0; d < digits; ++d)
    value |= static_cast<uint64_t>(slots[d] & 63) << (6 * d);
  return value;
}

// One block per row: a strided scan keeps each thread's top-2, warp
// shuffles and one shared pass merge them. The canonical order is a total
// order over distinct ids, so the tree's result is the scan's.
__global__ __launch_bounds__(kLocalThreads) void pick_local_kernel(
    const float* __restrict__ logits, int rows, int vocab_count,
    int vocab_begin, int rank, int world,
    const uint64_t* __restrict__ carry_digest, uint16_t* __restrict__ table,
    PickLocal* __restrict__ locals,
    const PickVerdict* __restrict__ row_select, int source_row_stride) {
  dgpp_pdl_wait();
  const int row = blockIdx.x;
  const int selected = row_select ? max(row_select[row].accepted - 1, 0) : 0;
  const int logits_row =
      row_select ? row * source_row_stride + selected : row;
  const int tid = threadIdx.x;
  const int group_slots = world * kPickSlotsPerRank;

  // Zero this row's candidate group; block 0 also owns the digest group
  // and the even-count pad slot. The encodes below run after the
  // reduction's barriers, so the zeroing is ordered before them within
  // the block.
  if (tid < group_slots) table[row * group_slots + tid] = 0;
  if (row == 0) {
    const int tail = static_cast<int>(device_pick_table_elems(rows, world)) -
                     rows * group_slots;
    if (tid < tail) table[rows * group_slots + tid] = 0;
  }

  Top2 mine = empty_top2();
  const float* slice = logits + static_cast<size_t>(logits_row) * vocab_count;
  for (int i = tid; i < vocab_count; i += kLocalThreads)
    push(mine, Pair{slice[i], vocab_begin + i});

  for (int mask = kWarp / 2; mask > 0; mask >>= 1)
    mine = merge(mine, shfl_xor(mine, mask));

  __shared__ Top2 warps[kLocalThreads / kWarp];
  if ((tid & (kWarp - 1)) == 0) warps[tid / kWarp] = mine;
  __syncthreads();
  if (tid != 0) return;
  Top2 best = warps[0];
  for (int w = 1; w < kLocalThreads / kWarp; ++w) best = merge(best, warps[w]);

  locals[row].best_id = best.a.id;
  locals[row].best_logit = best.a.v;
  locals[row].second_logit = best.b.v;

  uint16_t* slot = table + (static_cast<size_t>(row) * world + rank) *
                               kPickSlotsPerRank;
  encode_digits(slot, static_cast<uint64_t>(__float_as_uint(best.a.v)),
                kPickLogitDigits);
  encode_digits(slot + kPickLogitDigits, static_cast<uint64_t>(best.a.id),
                kPickIdDigits);
  if (row == 0)
    encode_digits(table + (static_cast<size_t>(rows) * world + rank) *
                              kPickSlotsPerRank,
                  *carry_digest, kPickSlotsPerRank);
}

__host__ __device__ inline uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

__host__ __device__ inline uint64_t verdict_digest(int rows, int accepted,
                                                   const int32_t* winners) {
  uint64_t h = splitmix64(static_cast<uint64_t>(rows));
  h = splitmix64(h ^ static_cast<uint64_t>(accepted));
  for (int r = 0; r < rows; ++r)
    h = splitmix64(h ^ static_cast<uint64_t>(static_cast<uint32_t>(winners[r])));
  return h & ((1ull << kPickDigestBits) - 1);
}

// One thread: the table is at most (kPickMaxRows + 1) x 8 x 9 slots.
__global__ void pick_verdict_kernel(const uint16_t* __restrict__ table,
                                    int rows, int world, int rank,
                                    const int64_t* __restrict__ fed,
                                    const int64_t* __restrict__ positions,
                                    int requests, int rows_per_request,
                                    int position_stride,
                                    PickVerdict* __restrict__ verdicts,
                                    PickVerdict* __restrict__ device_verdicts,
                                    uint64_t* __restrict__ carry_digest) {
  dgpp_pdl_wait();
  int32_t winners[kPickMaxRows];
  for (int r = 0; r < rows; ++r) {
    Pair best{0.0f, kNoId};
    for (int k = 0; k < world; ++k) {
      const uint16_t* slot =
          table + (static_cast<size_t>(r) * world + k) * kPickSlotsPerRank;
      Pair c;
      c.v = __uint_as_float(
          static_cast<uint32_t>(decode_digits(slot, kPickLogitDigits)));
      c.id = static_cast<int32_t>(
          decode_digits(slot + kPickLogitDigits, kPickIdDigits));
      if (k == 0 || before(c, best)) best = c;
    }
    winners[r] = best.id;
  }
  int accepted[kPickMaxRequests];
  bool active[kPickMaxRequests];
  uint64_t digest = requests == 1 ? 0 : splitmix64(requests);
  for (int q = 0; q < requests; ++q) {
    const int row0 = q * rows_per_request;
    active[q] = positions == nullptr || positions[q * position_stride] >= 0;
    accepted[q] = active[q] ? 1 : 0;
    while (active[q] && accepted[q] < rows_per_request &&
           winners[row0 + accepted[q] - 1] == fed[row0 + accepted[q]])
      ++accepted[q];
    const uint64_t one = verdict_digest(
        active[q] ? rows_per_request : 0, accepted[q], winners + row0);
    if (requests == 1)
      digest = one;
    else
      digest = splitmix64(digest ^ one);
  }
  digest &= (1ull << kPickDigestBits) - 1;

  const uint16_t* digests =
      table + static_cast<size_t>(rows) * world * kPickSlotsPerRank;
  const uint64_t mine =
      decode_digits(digests + rank * kPickSlotsPerRank, kPickSlotsPerRank);
  uint32_t digest_mismatch = 0;
  uint64_t peer_digests[kPickMaxWorld] = {};
  for (int k = 0; k < world; ++k) {
    const uint64_t d =
        decode_digits(digests + k * kPickSlotsPerRank, kPickSlotsPerRank);
    peer_digests[k] = d;
    if (d != mine) digest_mismatch |= 1u << k;
  }
  for (int q = 0; q < requests; ++q) {
    const int row0 = q * rows_per_request;
    PickVerdict v;
    v.rows = active[q] ? rows_per_request : 0;
    v.accepted = accepted[q];
    v.next = active[q] ? winners[row0 + accepted[q] - 1] : -1;
    for (int r = 0; r < rows_per_request; ++r)
      v.winners[r] = active[q] ? winners[row0 + r] : -1;
    v.digest = digest;
    v.digest_mismatch = digest_mismatch;
    for (int k = 0; k < world; ++k) v.peer_digests[k] = peer_digests[k];
    verdicts[q] = v;
    if (device_verdicts != nullptr) device_verdicts[q] = v;
  }
  *carry_digest = digest;
}

void check_shape(int rows, int world, int rank, const char* what) {
  if (rows < 1 || rows > kPickMaxRows)
    throw std::invalid_argument(std::string(what) + ": rows must be in [1, " +
                                std::to_string(kPickMaxRows) + "]");
  if (world < 1 || world > kPickMaxWorld || rank < 0 || rank >= world)
    throw std::invalid_argument(std::string(what) +
                                ": rank/world outside [0, world <= " +
                                std::to_string(kPickMaxWorld) + ")");
}

}  // namespace

void device_pick_local(const float* logits, int rows, int vocab_count,
                    int vocab_begin, int rank, int world,
                    const uint64_t* carry_digest, uint16_t* table,
                    PickLocal* locals, cudaStream_t stream,
                    const PickVerdict* row_select) {
  check_shape(rows, world, rank, "glm_pick_local");
  if (row_select != nullptr && rows != 1)
    throw std::invalid_argument("glm_pick_local: row_select needs rows == 1");
  if (vocab_count < 1 || vocab_begin < 0 ||
      vocab_begin + vocab_count > (1 << (6 * kPickIdDigits)))
    throw std::invalid_argument(
        "glm_pick_local: vocab slice outside the 18-bit id encoding");
  // Zeroing covers the digest group with the candidate groups' threads.
  static_assert(kPickMaxWorld * kPickSlotsPerRank + 1 <= kLocalThreads);
  pick_local_kernel<<<rows, kLocalThreads, 0, stream>>>(
      logits, rows, vocab_count, vocab_begin, rank, world, carry_digest, table,
      locals, row_select, /*source_row_stride=*/rows);
  DGPP_CUDA_OK(cudaGetLastError());
}

void device_pick_local_batched(
    const float* logits, int rows, int vocab_count, int vocab_begin, int rank,
    int world, const uint64_t* carry_digest, uint16_t* table,
    PickLocal* locals, cudaStream_t stream,
    const PickVerdict* row_select, int requests, int source_row_stride) {
  check_shape(rows, world, rank, "glm_pick_local_batched");
  if (requests < 1 || requests > kPickMaxRequests)
    throw std::invalid_argument("glm_pick_local_batched: requests");
  if (row_select != nullptr &&
      (rows != requests || source_row_stride < 1))
    throw std::invalid_argument(
        "glm_pick_local_batched: selected rows need one candidate per "
        "request and a positive source stride");
  if (vocab_count < 1 || vocab_begin < 0 ||
      vocab_begin + vocab_count > (1 << (6 * kPickIdDigits)))
    throw std::invalid_argument(
        "glm_pick_local_batched: vocab slice outside the 18-bit id encoding");
  pick_local_kernel<<<rows, kLocalThreads, 0, stream>>>(
      logits, rows, vocab_count, vocab_begin, rank, world, carry_digest, table,
      locals, row_select, source_row_stride);
  DGPP_CUDA_OK(cudaGetLastError());
}

void device_pick_verdict(const uint16_t* table, int rows, int world, int rank,
                      const int64_t* fed, PickVerdict* verdict,
                      PickVerdict* device_verdict, uint64_t* carry_digest,
                      cudaStream_t stream) {
  check_shape(rows, world, rank, "glm_pick_verdict");
  pick_verdict_kernel<<<1, 1, 0, stream>>>(
      table, rows, world, rank, fed, /*positions=*/nullptr, /*requests=*/1,
      /*rows_per_request=*/rows, /*position_stride=*/rows, verdict,
      device_verdict, carry_digest);
  DGPP_CUDA_OK(cudaGetLastError());
}

void device_pick_verdict_batched(
    const uint16_t* table, int rows, int world, int rank, const int64_t* fed,
    const int64_t* positions, int requests, int rows_per_request,
    int position_stride, PickVerdict* verdicts,
    PickVerdict* device_verdicts, uint64_t* carry_digest,
    cudaStream_t stream) {
  check_shape(rows, world, rank, "glm_pick_verdict_batched");
  if (requests < 1 || requests > kPickMaxRequests || rows_per_request < 1 ||
      requests * rows_per_request != rows)
    throw std::invalid_argument("glm_pick_verdict_batched: request shape");
  if (positions != nullptr && position_stride < rows_per_request)
    throw std::invalid_argument("glm_pick_verdict_batched: position stride");
  if (table == nullptr || fed == nullptr || verdicts == nullptr ||
      carry_digest == nullptr)
    throw std::invalid_argument("glm_pick_verdict_batched: null argument");
  pick_verdict_kernel<<<1, 1, 0, stream>>>(
      table, rows, world, rank, fed, positions, requests, rows_per_request,
      position_stride, verdicts, device_verdicts, carry_digest);
  DGPP_CUDA_OK(cudaGetLastError());
}

uint64_t device_pick_digest(int rows, int accepted, const int32_t* winners) {
  return verdict_digest(rows, accepted, winners);
}

}  // namespace dgpp
