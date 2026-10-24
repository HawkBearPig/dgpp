#include "kernels/sample_pick.hpp"
#include "kernels/pdl.cuh"

#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "common/det_math.hpp"
#include "kernels/topk_select.cuh"

// Compiled with --fmad=false (CMakeLists): no multiply-add here may be
// contracted, so every expression rounds exactly as the host's does under
// -ffp-contract=off. The deterministic exp/log name fma() themselves.
//
// Cost shape (GB10, 38,720-id slices, k = 128, the kernel bench). The
// deterministic fp64 exp is a 13-term polynomial and this part runs fp64
// at 1/64 rate: a slice's 38,720 normalizer terms are ~250 µs of one SM's
// time however a single block spreads them, so they run on a chunk grid
// (one block per 256 ids, every SM busy: ~16 µs for the batch). A
// dependent fp64 add is ~90 ns, so the normalizer's summation order is a
// fixed pairwise tree (sample::chunked_exp_sum) rather than the
// sequential chunk sum that cost ~30 µs per slice. The local top-k reads
// the slice twice: each thread's largest key, whose k-th largest over the
// block is a lower bound only ~k(1 + a few percent) ids reach, then those
// ids into shared memory to sort (or, tie-heavy, to radix-select) — in
// place of the streaming 2,048-key bitonic select, which cost ~680 µs per
// slice. The verdict decodes, merges (by rank counting) and evaluates
// every candidate's fp64 mass in parallel; only the decision's ordered
// walks (one fp64 prefix chain per row) stay on one thread. Measured:
// the three local launches 2 + 16 + 26 µs over two rows (was ~900 µs per
// row, rows serial within a request), the verdict 25/33 µs at T=1/T=2
// (was ~250/290).

namespace dgpp {

namespace {

constexpr int kLocalThreads = 1024;             // the per-row select block
constexpr int kChunkThreads = kSampleLseChunk;  // one id per thread
constexpr int kVerdictThreads = 256;
constexpr int kRadixBins = 256;
constexpr int kKeyIdxBits = 21;       // vocab ids < 2^21
constexpr uint64_t kKeyIdxMask = (1ull << kKeyIdxBits) - 1;
constexpr uint64_t kKeyMax = ~0ull;
constexpr uint32_t kSplitMax = 0xFFFFFFFFu;
constexpr int32_t kNoId = -1;

static_assert(kSampleLseChunk == 256, "the host's sample::kLseChunk");
static_assert(kSampleMaxChunks == kLocalThreads,
              "the row block folds one chunk partial per thread");
static_assert(kSampleMaxCandidates <= kLocalThreads,
              "the row block sorts the survivors with one thread per pair");

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

// The order key of a logit: sortable_f32 of the value with -0 folded onto
// +0, so that the integer order is exactly sample::candidate_before's
// float comparison (which cannot tell the zeros apart and breaks their tie
// by id). Higher key, higher logit.
__device__ inline uint32_t primary_key(float logit) {
  return sortable_f32_dev(logit == 0.0f ? 0.0f : logit);
}
// (~primary << 21) | id: the SMALLEST key is the canonical first candidate.
// A masked id is absent (sample::apply_mask writes -inf): its key is 0,
// below every present key (a finite float never maps to 0), so it is never
// listed, never selected and never counted.
__device__ inline uint32_t key_of(float logit) {
  return logit == -INFINITY ? 0u : primary_key(logit);
}

__device__ inline uint64_t composite_key(float logit, int32_t id) {
  return (static_cast<uint64_t>(~primary_key(logit)) << kKeyIdxBits) |
         static_cast<uint64_t>(id);
}
__device__ inline int32_t key_id(uint64_t key) {
  return static_cast<int32_t>(key & kKeyIdxMask);
}

__host__ __device__ inline uint64_t verdict_digest(int rows, int accepted,
                                                   const int32_t* winners) {
  uint64_t h = splitmix64(static_cast<uint64_t>(rows));
  h = splitmix64(h ^ static_cast<uint64_t>(accepted));
  for (int r = 0; r < rows; ++r)
    h = splitmix64(h ^ static_cast<uint64_t>(static_cast<uint32_t>(winners[r])));
  return h & ((1ull << kPickDigestBits) - 1);
}

// sample::apply_penalties for one id with count `count`.
__device__ inline float penalize(float v, int32_t count,
                                 const SampleSpec& s) {
  if (s.repetition_penalty != 1.0f)
    v = v > 0.0f ? __fdiv_rn(v, s.repetition_penalty)
                 : __fmul_rn(v, s.repetition_penalty);
  v = __fmaf_rn(-s.frequency_penalty, static_cast<float>(count), v);
  v = __fsub_rn(v, s.presence_penalty);
  return v;
}

// Block-wide min over 64-bit keys (one per thread).
__device__ inline uint64_t block_min_key(uint64_t key, uint64_t* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) / 32;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const uint32_t hi = __shfl_xor_sync(~0u, static_cast<uint32_t>(key >> 32), off);
    const uint32_t lo = __shfl_xor_sync(~0u, static_cast<uint32_t>(key), off);
    const uint64_t other = (static_cast<uint64_t>(hi) << 32) | lo;
    key = other < key ? other : key;
  }
  if (lane == 0) scratch[warp] = key;
  __syncthreads();
  key = threadIdx.x < nwarps ? scratch[threadIdx.x] : kKeyMax;
  if (warp == 0) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const uint32_t hi = __shfl_xor_sync(~0u, static_cast<uint32_t>(key >> 32), off);
      const uint32_t lo = __shfl_xor_sync(~0u, static_cast<uint32_t>(key), off);
      const uint64_t other = (static_cast<uint64_t>(hi) << 32) | lo;
      key = other < key ? other : key;
    }
    if (lane == 0) scratch[0] = key;
  }
  __syncthreads();
  const uint64_t out = scratch[0];
  __syncthreads();
  return out;
}

__device__ inline float block_max_f(float v, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) / 32;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_xor_sync(~0u, v, off));
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  v = threadIdx.x < nwarps ? scratch[threadIdx.x] : -INFINITY;
  if (warp == 0) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      v = fmaxf(v, __shfl_xor_sync(~0u, v, off));
    if (lane == 0) scratch[0] = v;
  }
  __syncthreads();
  const float out = scratch[0];
  __syncthreads();
  return out;
}

// The request's spec as a row sees it. The full path (stochastic rows, and
// greedy rows that report logprobs or carry penalties: their normalizer and
// scaled logits are the raw distribution's, temperature 1) is `sampled`.
struct RowSpec {
  bool active;
  bool sampled;
  bool penalized;
  bool constrained;       // the row carries a token mask (M6 6g)
  int32_t allowed;        // the mask's allowed count (the row's vocabulary)
  const uint32_t* mask;   // the mask's bitmask words (after the header)
  SampleSpec spec;
};

// The row's mask, when the host wrote one: masks[row * mask_stride] is the
// allowed count (0: unconstrained), the words after it the bitmask over
// [0, vocab_size). A constrained row takes the full path — its greedy
// argmax must respect the mask too.
__device__ inline RowSpec row_spec(const SampleSpec* specs, int q, const int64_t* positions,
                                   int position_stride, const uint32_t* masks, int mask_stride,
                                   int row, bool draft, const int32_t* request_map) {
  const int req = request_map ? request_map[q] : q;
  RowSpec rs;
  rs.active = req >= 0 && (positions == nullptr || positions[q * position_stride] >= 0);
  rs.spec = req >= 0 ? specs[req] : SampleSpec{};
  // The draft pick samples at the draft temperature (SampleSpec).
  if (draft && rs.spec.draft_temperature > 0.0f && rs.spec.temperature > 0.0f)
    rs.spec.temperature = rs.spec.draft_temperature;
  rs.penalized = rs.spec.repetition_penalty != 1.0f ||
                 rs.spec.frequency_penalty != 0.0f ||
                 rs.spec.presence_penalty != 0.0f;
  rs.mask = nullptr;
  rs.allowed = 0;
  rs.constrained = false;
  if (masks != nullptr) {
    const uint32_t* m = masks + static_cast<size_t>(row) * mask_stride;
    if (m[0] != 0u) {
      rs.constrained = true;
      rs.allowed = static_cast<int32_t>(m[0]);
      rs.mask = m + 1;
    }
  }
  rs.sampled = rs.active && (rs.spec.temperature > 0.0f ||
                             rs.spec.logprobs >= 0 || rs.penalized ||
                             rs.constrained || rs.spec.biased != 0);
  if (rs.spec.temperature <= 0.0f) rs.spec.temperature = 1.0f;
  return rs;
}

__device__ inline bool mask_allows(const uint32_t* mask, int id) {
  return ((mask[id >> 5] >> (id & 31)) & 1u) != 0u;
}

// Row t's context count of id v: the request's table (every token
// committed through the previous step) plus this step's fed tokens up to
// and including row t's — the table itself is untouched until the verdict
// commits (sample::SampledSpeculator's rule: the draft joins the
// context only when it stands).
__device__ inline int32_t context_count(const int32_t* counts_q, int32_t v,
                                        const int64_t* fed, int row0, int t) {
  int32_t c = counts_q[v];
  for (int j = 0; j <= t; ++j)
    if (fed[row0 + j] == static_cast<int64_t>(v)) ++c;
  return c;
}

// ---------------------------------------------------------------------------
// Kernel 1a, one block per 256-id chunk of every row: the penalties in
// place (sample::apply_penalties) and the chunk's max of the
// temperature-scaled slice.
// ---------------------------------------------------------------------------
// The logits row a (request, row) reads. Without `row_select`
// it is the row itself; with it — the draft pick — request q reads the row
// its verify verdict accepted, inside a wider source group.
__device__ __forceinline__ size_t source_row(int row, int q, int rows_per_request,
                                             const PickVerdict* __restrict__ row_select,
                                             int source_row_stride) {
  if (row_select == nullptr) return static_cast<size_t>(row);
  const int acc = row_select[q].accepted;
  const int pick = acc > 0 ? acc - 1 : 0;
  return static_cast<size_t>(q) * static_cast<size_t>(source_row_stride) +
         static_cast<size_t>(pick);
}

__global__ void __launch_bounds__(kChunkThreads)
    sample_prepare_kernel(float* __restrict__ logits, int vocab_count, int vocab_begin,
                          int vocab_size, const SampleSpec* __restrict__ specs,
                          int rows_per_request, const int64_t* __restrict__ fed,
                          const int64_t* __restrict__ positions, int position_stride,
                          const int32_t* __restrict__ counts, const float* __restrict__ bias,
                          const uint32_t* __restrict__ masks, int mask_stride,
                          double* __restrict__ maxes, const PickVerdict* __restrict__ row_select,
                          int source_row_stride, const int32_t* request_map) {
  dgpp_pdl_wait();
  __shared__ float fred[32];
  __shared__ uint64_t kred[32];
  const int c = blockIdx.x;
  const int row = blockIdx.y;
  const int q = row / rows_per_request;
  const int req = request_map ? request_map[q] : q;
  const int t = row % rows_per_request;
  const RowSpec rs = row_spec(specs, q, positions, position_stride, masks, mask_stride, row,
                              row_select != nullptr, request_map);
  const int nchunks = gridDim.x;
  float* slice = logits + source_row(row, q, rows_per_request, row_select,
                                     source_row_stride) * vocab_count;
  const int i = c * kChunkThreads + threadIdx.x;
  if (!rs.sampled) {
    // The greedy row's argmax, chunked (2026-09-29): this chunk's smallest
    // composite key, carried to the row block through its `maxes` slot (a
    // greedy row has no normalizer) as the key's 64-bit pattern. The row
    // block then reduces nchunks keys instead of walking the whole slice
    // with one block (117 us per 248K-id row on the GB10, four times per
    // depth-3 pass); the min over the chunk minima is the min over the
    // slice, so the pick is bitwise the one-block walk's.
    if (!rs.active) return;
    uint64_t key = i < vocab_count ? composite_key(slice[i], vocab_begin + i) : kKeyMax;
    key = block_min_key(key, kred);
    if (threadIdx.x == 0)
      maxes[static_cast<size_t>(row) * nchunks + c] = __longlong_as_double(static_cast<long long>(key));
    return;
  }
  float scaled = -INFINITY;
  if (i < vocab_count) {
    float l = slice[i];
    if (rs.penalized && counts != nullptr) {
      const int32_t cnt = context_count(counts + static_cast<size_t>(req) * vocab_size,
                                        vocab_begin + i, fed, q * rows_per_request, t);
      if (cnt != 0) {
        l = penalize(l, cnt, rs.spec);
        slice[i] = l;
      }
    }
    // The logit bias: the request's dense row, added after the
    // penalties and before the mask, in place — the host's gather fallback
    // and every later stage see the biased logit (sample::apply_bias).
    if (rs.spec.biased != 0 && bias != nullptr) {
      l = __fadd_rn(l, bias[static_cast<size_t>(req) * vocab_size + vocab_begin + i]);
      slice[i] = l;
    }
    // The mask: an excluded id becomes -inf in place (absent to every
    // later stage, the host's gather fallback included).
    if (rs.constrained && !mask_allows(rs.mask, vocab_begin + i)) {
      l = -INFINITY;
      slice[i] = l;
    }
    scaled = __fdiv_rn(l, rs.spec.temperature);
  }
  const float top = block_max_f(scaled, fred);
  if (threadIdx.x == 0)
    maxes[static_cast<size_t>(row) * nchunks + c] = static_cast<double>(top);
}

// ---------------------------------------------------------------------------
// Kernel 1b, the same grid: each chunk's normalizer partial in the host's
// order — the slice max, the chunk's terms exp_d(scaled - max) summed by
// recursive halving (sample::chunked_exp_sum's inner tree).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kChunkThreads)
    sample_partials_kernel(const float* __restrict__ logits, int vocab_count,
                           const SampleSpec* __restrict__ specs, int rows_per_request,
                           const int64_t* __restrict__ positions, int position_stride,
                           const uint32_t* __restrict__ masks, int mask_stride,
                           const double* __restrict__ maxes, double* __restrict__ partials,
                           const PickVerdict* __restrict__ row_select, int source_row_stride,
                           const int32_t* request_map) {
  dgpp_pdl_wait();
  __shared__ double terms[kChunkThreads];
  __shared__ float fred[32];
  const int c = blockIdx.x;
  const int row = blockIdx.y;
  const int q = row / rows_per_request;
  const RowSpec rs = row_spec(specs, q, positions, position_stride, masks, mask_stride, row,
                              row_select != nullptr, request_map);
  if (!rs.sampled) return;
  const int nchunks = gridDim.x;
  float m = -INFINITY;
  for (int j = threadIdx.x; j < nchunks; j += kChunkThreads)
    m = fmaxf(m, static_cast<float>(maxes[static_cast<size_t>(row) * nchunks + j]));
  const float top = block_max_f(m, fred);  // exact: any order
  const float* slice = logits + source_row(row, q, rows_per_request, row_select,
                                           source_row_stride) * vocab_count;
  const int c0 = c * kChunkThreads;
  const int i = c0 + threadIdx.x;
  // A slice with every id masked has no mass (top is -inf): zero terms, so
  // the row's lse folds as -inf rather than NaN.
  terms[threadIdx.x] =
      i < vocab_count && top != -INFINITY
          ? detmath::exp_d(
                static_cast<double>(__fdiv_rn(slice[i], rs.spec.temperature)) -
                static_cast<double>(top))
          : 0.0;
  __syncthreads();
  // sample::halving_sum over the chunk: eight dependent adds.
  for (int h = kChunkThreads >> 1; h >= 1; h >>= 1) {
    if (threadIdx.x < h) terms[threadIdx.x] += terms[threadIdx.x + h];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    partials[static_cast<size_t>(row) * nchunks + c] = terms[0];
}

// ---------------------------------------------------------------------------
// The block-wide radix select: the threshold key of the `need` largest
// 32-bit keys over [0, n) — four 8-bit passes from the top, a shared
// histogram per pass over the keys still matching the resolved prefix.
// Returns the threshold, how many of the keys EQUAL to it are wanted
// (>= 1) and how many there are; keys above it are all wanted. All threads
// of the block call it. Cheap for the shared-memory key sets it normally
// runs on (one or two keys per thread); the whole-slice form behind
// SliceKeyFn is the fallback for tie-heavy slices.
// ---------------------------------------------------------------------------
constexpr int kLoadBatch = 8;  // independent loads in flight per thread

template <typename KeyFn>
__device__ inline void radix_threshold(KeyFn key_of, int n, int need,
                                       uint32_t* hist, int* found,
                                       uint32_t* threshold, int* need_at,
                                       int* count_at) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  uint32_t prefix = 0, mask = 0;
  int count = 0;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int b = tid; b < kRadixBins; b += blockDim.x) hist[b] = 0;
    __syncthreads();
    for (int base = tid; base < n; base += blockDim.x * kLoadBatch) {
      uint32_t key[kLoadBatch];
      bool have[kLoadBatch];
#pragma unroll
      for (int b = 0; b < kLoadBatch; ++b) {
        const int i = base + b * blockDim.x;
        have[b] = i < n;
        key[b] = have[b] ? key_of(i) : 0u;
      }
#pragma unroll
      for (int b = 0; b < kLoadBatch; ++b)
        if (have[b] && (key[b] & mask) == prefix)
          atomicAdd(&hist[(key[b] >> shift) & 255u], 1u);
    }
    __syncthreads();
    if (warp == 0) {
      // Lane l owns bins 255-8l .. 248-8l; the cumulative count from the
      // top bin down locates the bin holding the need-th largest key.
      int local = 0;
      for (int j = 0; j < 8; ++j) local += static_cast<int>(hist[255 - 8 * lane - j]);
      int incl = local;
      for (int off = 1; off < 32; off <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, incl, off);
        if (lane >= off) incl += v;
      }
      const int excl = incl - local;
      if (excl < need && need <= incl) {
        int cum = excl;
        for (int j = 0; j < 8; ++j) {
          const int b = 255 - 8 * lane - j;
          const int h = static_cast<int>(hist[b]);
          if (cum + h >= need) {
            found[0] = b;
            found[1] = need - cum;
            found[2] = h;
            break;
          }
          cum += h;
        }
      }
    }
    __syncthreads();
    const int bin = found[0];
    need = found[1];
    count = found[2];
    prefix |= static_cast<uint32_t>(bin) << shift;
    mask |= 255u << shift;
    __syncthreads();
  }
  *threshold = prefix;
  *need_at = need;
  *count_at = count;
}

// Key sets: a shared-memory list of keys (with the ids' local indices for
// the tie pass), and the whole slice. Among keys equal to the value
// threshold the LOWER index wins: the tie pass selects on ~index, every
// other key 0 (below every tie).
struct ListKeyFn {
  const uint32_t* keys;
  __device__ uint32_t operator()(int i) const { return keys[i]; }
};
struct ListTieKeyFn {
  const uint32_t* keys;
  const int32_t* idx;
  uint32_t threshold;
  __device__ uint32_t operator()(int i) const {
    return keys[i] == threshold ? ~static_cast<uint32_t>(idx[i]) : 0u;
  }
};
struct SliceKeyFn {
  const float* slice;
  __device__ uint32_t operator()(int i) const { return key_of(slice[i]); }
};
struct SliceTieKeyFn {
  const float* slice;
  uint32_t threshold;
  __device__ uint32_t operator()(int i) const {
    return key_of(slice[i]) == threshold ? ~static_cast<uint32_t>(i) : 0u;
  }
};

// bitonic_sort_asc for the survivors (n <= kSampleMaxCandidates) on the
// block's first kSortThreads threads only, synchronized by a named barrier:
// a 1024-thread __syncthreads per stage cost more than the stage.
constexpr int kSortThreads = 128;
static_assert(kSampleMaxCandidates <= 2 * kSortThreads, "two pairs per thread");
__device__ inline void sort_survivors(uint32_t* hi, uint32_t* lo, int n) {
  for (int size = 2; size <= n; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = threadIdx.x; i < n; i += kSortThreads) {
        const int j = i ^ stride;
        if (j > i) cx_split(hi, lo, i, j, (i & size) == 0);
      }
      asm volatile("bar.sync 1, %0;" ::"r"(kSortThreads) : "memory");
    }
  }
}

// The ids at or above the per-thread-maxima bound that the row block
// keeps in shared memory before the exact select; more than this (a
// tie-heavy slice) falls back to the whole-slice radix passes.
constexpr int kListCap = 2048;

// ---------------------------------------------------------------------------
// Kernel 1c, one block per ROW: the row's wire group — the normalizer
// folded from the chunk partials in chunk order, the exact local top-k in
// canonical order (radix select, then one bitonic sort of the survivors) —
// or the greedy row's argmax; the digest carry.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kLocalThreads) sample_local_kernel(
    const float* __restrict__ logits, int rows, int vocab_count, int vocab_begin, int rank,
    int world, int candidates, const SampleSpec* __restrict__ specs, int rows_per_request,
    const int64_t* __restrict__ positions, int position_stride, const uint32_t* __restrict__ masks,
    int mask_stride, const double* __restrict__ maxes, const double* __restrict__ partials,
    int nchunks, const uint64_t* __restrict__ carry_digest, uint16_t* __restrict__ table,
    PickLocal* __restrict__ locals, const PickVerdict* __restrict__ row_select,
    int source_row_stride, const int32_t* request_map) {
  dgpp_pdl_wait();
  __shared__ uint32_t hist[kRadixBins];
  __shared__ int found[3];
  __shared__ int sel_count;
  __shared__ int list_count;
  __shared__ int present_count;
  __shared__ uint32_t tmax[kLocalThreads];
  __shared__ uint32_t list_key[kListCap];
  __shared__ int32_t list_idx[kListCap];
  __shared__ uint32_t sel_hi[kSampleMaxCandidates];
  __shared__ uint32_t sel_lo[kSampleMaxCandidates];
  __shared__ double fold[kSampleMaxChunks];
  __shared__ float fred[32];
  __shared__ uint64_t kred[32];

  const int row = blockIdx.x;
  const int q = row / rows_per_request;
  const int tid = threadIdx.x;
  const size_t group = device_sample_rank_group_slots(candidates);
  const size_t row_slots = static_cast<size_t>(world) * group;
  const RowSpec rs = row_spec(specs, q, positions, position_stride, masks, mask_stride, row,
                              row_select != nullptr, request_map);
  uint16_t* row_base = table + static_cast<size_t>(row) * row_slots;
  uint16_t* mine = row_base + static_cast<size_t>(rank) * group;
  const float* slice = logits + source_row(row, q, rows_per_request, row_select,
                                           source_row_stride) * vocab_count;

  // Zero this row's candidate region (every rank's group: the fold needs
  // zeros in the foreign slots); block 0 also owns the digest group and
  // the even-count pad.
  for (size_t i = tid; i < row_slots; i += kLocalThreads) row_base[i] = 0;
  if (row == 0) {
    const size_t total = device_sample_table_elems(rows, world, candidates);
    const size_t begin = static_cast<size_t>(rows) * row_slots;
    for (size_t i = begin + tid; i < total; i += kLocalThreads) table[i] = 0;
  }
  __syncthreads();

  if (!rs.active) {
    if (tid == 0) {
      locals[row].best_id = kNoId;
      locals[row].best_logit = -INFINITY;
      locals[row].second_logit = -INFINITY;
    }
  } else if (!rs.sampled) {
    // The greedy row: the canonical argmax (the smallest composite key)
    // over the chunk minima sample_prepare_kernel left in `maxes`,
    // written as the one candidate; every other slot carries the empty id.
    uint64_t best = kKeyMax;
    for (int j = tid; j < nchunks; j += kLocalThreads) {
      const uint64_t key = static_cast<uint64_t>(
          __double_as_longlong(maxes[static_cast<size_t>(row) * nchunks + j]));
      best = key < best ? key : best;
    }
    best = block_min_key(best, kred);
    const int32_t best_id = key_id(best);
    const float best_logit = slice[best_id - vocab_begin];
    for (int j = tid; j < candidates; j += kLocalThreads) {
      uint16_t* slot = mine + static_cast<size_t>(j) * kPickSlotsPerRank;
      if (j == 0) {
        encode_digits(slot, static_cast<uint64_t>(__float_as_uint(best_logit)),
                      kPickLogitDigits);
        encode_digits(slot + kPickLogitDigits, static_cast<uint64_t>(best_id),
                      kPickIdDigits);
      } else {
        encode_digits(slot + kPickLogitDigits, kSampleEmptyId, kPickIdDigits);
      }
    }
    if (tid == 0) {
      locals[row].best_id = best_id;
      locals[row].best_logit = best_logit;
      locals[row].second_logit = -INFINITY;
    }
  } else {
    // ---- the sampled row ------------------------------------------------
    // 1. The slice's temperature-scaled log-sum-exp: the max over the chunk
    //    maxes, the chunk partials (zero-padded to a power of two) summed
    //    by recursive halving, then log (sample::slice_logsumexp).
    const float m = tid < nchunks
                        ? static_cast<float>(maxes[static_cast<size_t>(row) * nchunks + tid])
                        : -INFINITY;
    const float top = block_max_f(m, fred);
    fold[tid] = tid < nchunks ? partials[static_cast<size_t>(row) * nchunks + tid]
                              : 0.0;
    __syncthreads();
    int fold_width = 1;
    while (fold_width < nchunks) fold_width <<= 1;
    for (int h = fold_width >> 1; h >= 1; h >>= 1) {
      if (tid < h) fold[tid] += fold[tid + h];
      __syncthreads();
    }
    if (tid == 0) {
      // A slice with every id masked carries no mass: lse -inf (the fold
      // skips it), never log(0).
      const double lse = top == -INFINITY
                             ? -INFINITY
                             : static_cast<double>(top) + detmath::log_d(fold[0]);
      encode_digits(mine + static_cast<size_t>(candidates) * kPickSlotsPerRank,
                    detmath::bits_of(lse), kSampleLseDigits);
    }

    // 2. The exact local top-k (sample::local_topk) over the PRESENT
    //    ids (a masked id's key is 0 and it is never listed).
    //    a. Each thread's largest key over its strided share of the slice,
    //       and the count of present ids. The k-th largest of the maxima
    //       is a lower bound on the k-th largest key of the slice (the k
    //       largest thread maxima are k distinct ids at or above it) and a
    //       tight one: with ~38 ids per thread, the ids reaching it number
    //       ~k(1 + a few percent). The slice does not fit L1 beside the
    //       block's shared memory, so its loads are issued kLoadBatch at a
    //       time from L2.
    uint32_t my_max = 0;
    int my_present = 0;
    if (tid == 0) present_count = 0;
    for (int base = tid; base < vocab_count; base += kLocalThreads * kLoadBatch) {
      uint32_t key[kLoadBatch];
#pragma unroll
      for (int b = 0; b < kLoadBatch; ++b) {
        const int i = base + b * kLocalThreads;
        key[b] = i < vocab_count ? key_of(slice[i]) : 0u;
      }
#pragma unroll
      for (int b = 0; b < kLoadBatch; ++b) {
        my_max = max(my_max, key[b]);
        my_present += key[b] != 0u ? 1 : 0;
      }
    }
    tmax[tid] = my_max;
    if (tid == 0) list_count = 0;
    __syncthreads();
    if (my_present != 0) atomicAdd(&present_count, my_present);
    __syncthreads();
    const int present = present_count;
    const int k = min(candidates, present);
    uint32_t bound = 0;
    int bound_need = 0, bound_count = 0;
    if (k > 0)
      radix_threshold(ListKeyFn{tmax}, kLocalThreads, k, hist, found, &bound,
                      &bound_need, &bound_count);
    // Fewer present ids than threads hold maxima: the k-th maximum may be
    // 0 (an absent id); the bound is then 1 — every present id, and none
    // of the absent ones.
    bound = max(bound, 1u);
    //    b. The ids at or above the bound, with their keys, in shared
    //       memory.
    for (int base = tid; base < vocab_count; base += kLocalThreads * kLoadBatch) {
      uint32_t key[kLoadBatch];
#pragma unroll
      for (int b = 0; b < kLoadBatch; ++b) {
        const int i = base + b * kLocalThreads;
        key[b] = i < vocab_count ? key_of(slice[i]) : 0u;
      }
#pragma unroll
      for (int b = 0; b < kLoadBatch; ++b) {
        const int i = base + b * kLocalThreads;
        if (i < vocab_count && key[b] >= bound) {
          const int pos = atomicAdd(&list_count, 1);
          if (pos < kListCap) {
            list_key[pos] = key[b];
            list_idx[pos] = i;
          }
        }
      }
    }
    if (tid == 0) sel_count = 0;
    for (int i = tid; i < kSampleMaxCandidates; i += kLocalThreads) {
      sel_hi[i] = kSplitMax;
      sel_lo[i] = kSplitMax;
    }
    __syncthreads();
    const int listed = list_count;  // >= k by construction
    //    c. The survivors into sel (hi = ~key, lo = id; ascending is the
    //       canonical order): the whole list when it is no wider than the
    //       sort, else the exact threshold over the list — or, past the
    //       list's capacity, over the whole slice — and its compaction.
    int survivors = k;
    if (k == 0) {
      survivors = 0;  // no present id: every slot empty
    } else if (listed <= kSampleMaxCandidates) {
      survivors = listed;
      for (int i = tid; i < listed; i += kLocalThreads) {
        sel_hi[i] = ~list_key[i];
        sel_lo[i] = static_cast<uint32_t>(vocab_begin + list_idx[i]);
      }
    } else if (listed <= kListCap) {
      uint32_t threshold = 0;
      int need = 0, count = 0;
      radix_threshold(ListKeyFn{list_key}, listed, k, hist, found, &threshold,
                      &need, &count);
      const bool every_tie = need == count;
      uint32_t tie_threshold = 0;
      if (!every_tie) {
        int need2 = 0, count2 = 0;
        radix_threshold(ListTieKeyFn{list_key, list_idx, threshold}, listed,
                        need, hist, found, &tie_threshold, &need2, &count2);
      }
      for (int i = tid; i < listed; i += kLocalThreads) {
        const uint32_t u = list_key[i];
        const int32_t idx = list_idx[i];
        const bool take =
            u > threshold ||
            (u == threshold &&
             (every_tie || ~static_cast<uint32_t>(idx) >= tie_threshold));
        if (take) {
          const int pos = atomicAdd(&sel_count, 1);
          if (pos < kSampleMaxCandidates) {
            sel_hi[pos] = ~u;
            sel_lo[pos] = static_cast<uint32_t>(vocab_begin + idx);
          }
        }
      }
    } else {
      uint32_t threshold = 0;
      int need = 0, count = 0;
      radix_threshold(SliceKeyFn{slice}, vocab_count, k, hist, found,
                      &threshold, &need, &count);
      const bool every_tie = need == count;
      uint32_t tie_threshold = 0;
      if (!every_tie) {
        int need2 = 0, count2 = 0;
        radix_threshold(SliceTieKeyFn{slice, threshold}, vocab_count, need,
                        hist, found, &tie_threshold, &need2, &count2);
      }
      for (int base = tid; base < vocab_count; base += kLocalThreads * kLoadBatch) {
        uint32_t key[kLoadBatch];
#pragma unroll
        for (int b = 0; b < kLoadBatch; ++b) {
          const int i = base + b * kLocalThreads;
          key[b] = i < vocab_count ? key_of(slice[i]) : 0u;
        }
#pragma unroll
        for (int b = 0; b < kLoadBatch; ++b) {
          const int i = base + b * kLocalThreads;
          const uint32_t u = key[b];
          const bool take =
              i < vocab_count && u != 0u &&
              (u > threshold ||
               (u == threshold &&
                (every_tie || ~static_cast<uint32_t>(i) >= tie_threshold)));
          if (take) {
            const int pos = atomicAdd(&sel_count, 1);
            if (pos < kSampleMaxCandidates) {
              sel_hi[pos] = ~u;
              sel_lo[pos] = static_cast<uint32_t>(vocab_begin + i);
            }
          }
        }
      }
    }
    __syncthreads();
    //    d. Sorted; the first k are the canonical local top-k.
    int width = 2;
    while (width < survivors) width <<= 1;
    if (tid < kSortThreads && survivors > 0) sort_survivors(sel_hi, sel_lo, width);
    __syncthreads();
    for (int j = tid; j < candidates; j += kLocalThreads) {
      uint16_t* slot = mine + static_cast<size_t>(j) * kPickSlotsPerRank;
      if (j < k) {
        const int32_t id = static_cast<int32_t>(sel_lo[j]);
        const float logit = slice[id - vocab_begin];
        encode_digits(slot, static_cast<uint64_t>(__float_as_uint(logit)),
                      kPickLogitDigits);
        encode_digits(slot + kPickLogitDigits, static_cast<uint64_t>(id),
                      kPickIdDigits);
      } else {
        encode_digits(slot + kPickLogitDigits, kSampleEmptyId, kPickIdDigits);
      }
    }
    if (tid == 0) {
      if (k > 0) {
        const int32_t id0 = static_cast<int32_t>(sel_lo[0]);
        locals[row].best_id = id0;
        locals[row].best_logit = slice[id0 - vocab_begin];
        locals[row].second_logit =
            k > 1 ? slice[static_cast<int32_t>(sel_lo[1]) - vocab_begin]
                  : -INFINITY;
      } else {
        locals[row].best_id = kNoId;
        locals[row].best_logit = -INFINITY;
        locals[row].second_logit = -INFINITY;
      }
    }
  }
  if (row == 0 && tid == 0)
    encode_digits(table + static_cast<size_t>(rows) * row_slots +
                      static_cast<size_t>(rank) * kPickSlotsPerRank,
                  *carry_digest, kPickSlotsPerRank);
}

// ---------------------------------------------------------------------------
// Kernel 2a: the decision, one block per request. The block decodes,
// merges and evaluates the candidates' masses cooperatively; thread 0 then
// makes the decision, whose ordered walks are the host's arithmetic.
// ---------------------------------------------------------------------------
struct Decision {
  bool resolved;
  bool accepted;  // the speculative accept test (T=2 row 0)
  int32_t token;
  float logprob;
  double covered;
};

// The reported top-N (the host's Result::top_logprobs): the first
// min(N, n) candidates of the decided list under `lse`.
__device__ inline void report_top(const float* logit, const int32_t* id, int n,
                                  float temperature, float lse, int N,
                                  int32_t* out_ids, float* out_lps,
                                  int32_t* out_count) {
  const int count = min(min(N, n), kSampleMaxTopLogprobs);
  for (int i = 0; i < count; ++i) {
    out_ids[i] = id[i];
    out_lps[i] = __fsub_rn(__fdiv_rn(logit[i], temperature), lse);
  }
  *out_count = count;
}

// sample::selector_state's stages 2-5 over prefix [0, n) of the merged
// candidates. `expsrc[i]` is exp_f(scaled_i - scaled_0), precomputed in
// parallel (the same deterministic value the host computes in its loop);
// exps[] receives the survivors' exps. Returns final_count, final_den, lse
// through the out-params.
__device__ inline void selector_state(const float* logit, const float* expsrc,
                                      int n, float temperature, int top_k,
                                      float min_p, float top_p, float* exps,
                                      int* final_count, float* final_den,
                                      float* lse) {
  const float scaled0 = __fdiv_rn(logit[0], temperature);
  int kept = n;
  if (top_k > 0) kept = min(kept, top_k);
  for (int i = 0; i < kept; ++i) exps[i] = expsrc[i];
  float den = 0.0f;
  for (int i = 0; i < kept; ++i) den = __fadd_rn(den, exps[i]);
  int survivors = kept;
  if (min_p > 0.0f) {
    const float threshold = __fmul_rn(min_p, __fdiv_rn(exps[0], den));
    int w = 0;
    for (int i = 0; i < kept; ++i)
      if (__fdiv_rn(exps[i], den) >= threshold) exps[w++] = exps[i];
    survivors = w;
  }
  den = 0.0f;
  for (int i = 0; i < survivors; ++i) den = __fadd_rn(den, exps[i]);
  int fc = survivors;
  if (top_p < 1.0f) {
    float cum = 0.0f;
    int cut = survivors;
    for (int i = 0; i < survivors; ++i) {
      cum = __fadd_rn(cum, __fdiv_rn(exps[i], den));
      if (cum >= top_p) {
        cut = i + 1;
        break;
      }
    }
    fc = cut;
  }
  float fd = 0.0f;
  for (int i = 0; i < fc; ++i) fd = __fadd_rn(fd, exps[i]);
  *final_count = fc;
  *final_den = fd;
  *lse = __fadd_rn(scaled0, detmath::log_f(fd));
}

// sample::select_from_sorted's draw over the state. `top` (optional)
// receives the final set's top-N report.
struct TopReport {
  int N;
  int32_t* ids;
  float* lps;
  int32_t* count;
};

__device__ inline void selector(const float* logit, const int32_t* id,
                                const float* expsrc, int n, float temperature,
                                int top_k, float min_p, float top_p,
                                float* exps, uint64_t seed, uint64_t* counter,
                                int32_t* token, float* logprob,
                                const TopReport* top = nullptr,
                                DraftProposal* proposal = nullptr) {
  int final_count = 0;
  float final_den = 0.0f, lse = 0.0f;
  selector_state(logit, expsrc, n, temperature, top_k, min_p, top_p, exps,
                 &final_count, &final_den, &lse);
  if (top != nullptr)
    report_top(logit, id, final_count, temperature, lse, top->N, top->ids,
               top->lps, top->count);
  // The draw's own distribution, kept for the next step's verify: the
  // masses in the same fp32 quotient the walk below uses. A set wider than
  // the buffer is not carried (n = 0) — the verify then uses the plain
  // rule, which stays exact.
  if (proposal != nullptr) {
    if (final_count <= kSampleProposalMax) {
      proposal->n = final_count;
      for (int i = 0; i < final_count; ++i) {
        proposal->ids[i] = id[i];
        proposal->mass[i] = __fdiv_rn(exps[i], final_den);
      }
    } else {
      proposal->n = 0;
    }
  }
  const double r = uniform01(seed, *counter);
  *counter += 1;
  double cum = 0.0;
  int chosen = final_count;
  int last_positive = 0;  // the rounding guard: the last token with mass
  for (int i = 0; i < final_count; ++i) {
    if (exps[i] > 0.0f) last_positive = i;
    cum += static_cast<double>(__fdiv_rn(exps[i], final_den));
    if (cum > r) {
      chosen = i;
      break;
    }
  }
  if (chosen == final_count) chosen = last_positive;
  *token = id[chosen];
  *logprob = __fsub_rn(__fdiv_rn(logit[chosen], temperature), lse);
}

// sample::spec_select_from_sorted: accept the draft with its exact
// probability under the final set, else the residual walk.
__device__ inline bool spec_select(const float* logit, const int32_t* id,
                                   const float* expsrc, int n,
                                   float temperature, int top_k, float min_p,
                                   float top_p, int32_t draft, float* exps,
                                   uint64_t seed, uint64_t* counter,
                                   int32_t* token, float* logprob,
                                   const TopReport* top = nullptr,
                                   const float* qmass = nullptr,
                                   double q_draft = 0.0) {
  int final_count = 0;
  float final_den = 0.0f, lse = 0.0f;
  selector_state(logit, expsrc, n, temperature, top_k, min_p, top_p, exps,
                 &final_count, &final_den, &lse);
  if (top != nullptr)
    report_top(logit, id, final_count, temperature, lse, top->N, top->ids,
               top->lps, top->count);
  int j = final_count;
  for (int i = 0; i < final_count; ++i)
    if (id[i] == draft) {
      j = i;
      break;
    }
  const float p_draft = j < final_count ? __fdiv_rn(exps[j], final_den) : 0.0f;
  const bool ratio = qmass != nullptr && q_draft > 0.0;
  const double u1 = uniform01(seed, *counter);
  *counter += 1;
  if (ratio ? (static_cast<double>(p_draft) > u1 * q_draft)
            : (static_cast<double>(p_draft) > u1)) {
    *token = draft;
    *logprob = __fsub_rn(__fdiv_rn(logit[j], temperature), lse);
    return true;
  }
  const double u2 = uniform01(seed, *counter);
  *counter += 1;
  if (ratio) {
    // The residual (P - Q)+ over the final set, the host's arithmetic:
    // fp64 masses from the fp32 exps and the fp32 denominator.
    double res_den = 0.0;
    for (int i = 0; i < final_count; ++i) {
      const double pi = static_cast<double>(exps[i]) / static_cast<double>(final_den);
      const double qi = static_cast<double>(qmass[i]);
      if (pi > qi) res_den += pi - qi;
    }
    double cum_r = 0.0;
    int chosen_r = final_count, last_r = final_count;
    if (res_den > 0.0) {
      for (int i = 0; i < final_count; ++i) {
        const double pi = static_cast<double>(exps[i]) / static_cast<double>(final_den);
        const double qi = static_cast<double>(qmass[i]);
        const double ri = pi > qi ? pi - qi : 0.0;
        if (ri > 0.0 || last_r == final_count) last_r = i;
        cum_r += ri / res_den;
        if (cum_r > u2) {
          chosen_r = i;
          break;
        }
      }
    }
    if (chosen_r == final_count) chosen_r = last_r;
    if (chosen_r == final_count) chosen_r = 0;
    *token = id[chosen_r];
    *logprob = __fsub_rn(__fdiv_rn(logit[chosen_r], temperature), lse);
    return false;
  }
  const float res_den =
      j < final_count ? __fsub_rn(final_den, exps[j]) : final_den;
  double cum = 0.0;
  int chosen = final_count;
  int last = final_count;  // the last residual token with mass
  for (int i = 0; i < final_count; ++i) {
    if (i == j) continue;
    if (exps[i] > 0.0f || last == final_count) last = i;
    cum += static_cast<double>(__fdiv_rn(exps[i], res_den));
    if (cum > u2) {
      chosen = i;
      break;
    }
  }
  if (chosen == final_count) chosen = last;
  *token = id[chosen];
  *logprob = __fsub_rn(__fdiv_rn(logit[chosen], temperature), lse);
  return false;
}

// sample::resolve_support over the merged prefix [0, held). `mass[i]`
// is exp_d(scaled_i - Z), precomputed in parallel; `prefix[i]` receives
// the running fp64 sum of the masses through i in the host's order — the
// covered mass, the nucleus crossing and the pure walk all read the same
// chain, computed once (a dependent fp64 add is ~90 ns on GB10).
struct Support {
  int kind;  // 0 fallback, 1 materialized, 2 pure
  int n;
  int top_k;
  float min_p;
  float top_p;
  double covered;
};

__device__ inline Support resolve_support(const float* logit,
                                          const double* mass, double* prefix,
                                          int held, int vocab_size,
                                          const SampleSpec& s) {
  Support out{0, 0, s.top_k, s.min_p, s.top_p, 0.0};
  const bool complete = held == vocab_size;
  const float T = s.temperature;
  {
    double cumulative = 0.0;
    for (int i = 0; i < held; ++i) {
      cumulative += mass[i];
      prefix[i] = cumulative;
    }
    out.covered = cumulative;
  }
  if (s.top_k > 0) {
    const int required = min(s.top_k, vocab_size);
    if (held < required) return out;
    out.kind = 1;
    out.n = required;
    return out;
  }
  if (s.min_p > 0.0f) {
    const float threshold =
        __fadd_rn(__fdiv_rn(logit[0], T), detmath::log_f(s.min_p));
    int survivors = held;
    bool bounded = complete;
    for (int i = 0; i < held; ++i) {
      if (__fdiv_rn(logit[i], T) < threshold) {
        survivors = i;
        bounded = true;
        break;
      }
    }
    if (!bounded) return out;
    out.kind = 1;
    out.n = survivors;
    out.min_p = 0.0f;
    return out;
  }
  if (s.top_p < 1.0f) {
    const double top_p = static_cast<double>(s.top_p);
    int nucleus = 0;
    for (int i = 0; i < held; ++i) {
      if (prefix[i] >= top_p) {
        nucleus = i + 1;
        break;
      }
    }
    if (nucleus == 0) {
      if (!complete) return out;
      nucleus = held;
    }
    out.kind = 1;
    out.n = nucleus;
    out.top_p = 1.0f;
    return out;
  }
  out.kind = 2;
  out.n = held;
  return out;
}

// sample::sample_from_prefix.
__device__ inline Decision decide_prefix(const float* logit, const int32_t* id,
                                         const double* mass, double* prefix,
                                         const float* expsrc, int held,
                                         int vocab_size, double Z,
                                         const SampleSpec& s, float* exps,
                                         uint64_t* counter,
                                         const TopReport* top = nullptr,
                                         DraftProposal* proposal = nullptr) {
  Decision d{false, false, kNoId, 0.0f, 0.0};
  const Support sup =
      resolve_support(logit, mass, prefix, held, vocab_size, s);
  d.covered = sup.covered;
  if (sup.kind == 0) return d;
  const float T = s.temperature;
  if (sup.kind == 1) {
    selector(logit, id, expsrc, sup.n, T, sup.top_k, sup.min_p, sup.top_p,
             exps, s.seed, counter, &d.token, &d.logprob, top, proposal);
    d.resolved = true;
    return d;
  }
  // Pure temperature sampling: the fp64 walk over the fold masses.
  const bool complete = held == vocab_size;
  if (top != nullptr && !complete && top->N > held) return d;  // the host's rule
  const double draw = uniform01(s.seed, *counter);
  int chosen = held;
  int last_positive = 0;  // the rounding guard: the last token with mass
  for (int i = 0; i < held; ++i) {
    if (mass[i] > 0.0) last_positive = i;
    if (prefix[i] > draw) {
      chosen = i;
      break;
    }
  }
  if (chosen == held) {
    if (!complete) return d;
    chosen = last_positive;
  }
  *counter += 1;
  d.token = id[chosen];
  d.logprob = __fsub_rn(__fdiv_rn(logit[chosen], T), static_cast<float>(Z));
  if (top != nullptr)
    report_top(logit, id, held, T, static_cast<float>(Z), top->N, top->ids,
               top->lps, top->count);
  d.resolved = true;
  return d;
}

// sample::spec_accept_from_prefix.
// `draft_excluded`: the draft is a masked id — probability 0 without the
// prefix having to show it (sample::spec_accept_from_prefix).
// `qmass`: the proposal's mass at each merged candidate, or
// null when the draft was deterministic. With it the accept test is
// min(1, P/Q) and the residual is the normalized (P - Q)+ — the marginal is
// P either way (sample::spec_select_from_sorted, the host oracle). In the
// pure regime the residual's total is 1 - sum min(P, Q), exact only when the
// prefix holds every id the proposal gives mass (`q_missing` counts the
// proposed ids outside it: any keeps the row on the deterministic rule).
__device__ inline Decision spec_decide_prefix(
    const float* logit, const int32_t* id, const double* mass, double* prefix,
    const float* expsrc, int held, int vocab_size, double Z, int32_t draft,
    const SampleSpec& s, float* exps, uint64_t* counter,
    const TopReport* top = nullptr, bool draft_excluded = false,
    const float* qmass = nullptr, double q_draft = 0.0, int q_missing = 0) {
  Decision d{false, false, kNoId, 0.0f, 0.0};
  const Support sup =
      resolve_support(logit, mass, prefix, held, vocab_size, s);
  d.covered = sup.covered;
  if (sup.kind == 0) return d;
  const float T = s.temperature;
  if (sup.kind == 1) {
    d.accepted = spec_select(logit, id, expsrc, sup.n, T, sup.top_k,
                             sup.min_p, sup.top_p, draft, exps, s.seed,
                             counter, &d.token, &d.logprob, top, qmass,
                             q_draft);
    d.resolved = true;
    return d;
  }
  const bool complete = held == vocab_size;
  int j = held;
  if (!draft_excluded)
    for (int i = 0; i < held; ++i)
      if (id[i] == draft) {
        j = i;
        break;
      }
  if (j == held && !complete && !draft_excluded) return d;
  const double p_draft = j < held ? mass[j] : 0.0;
  // The ratio rule when the proposal carries the draft (the host's
  // arithmetic: the overlap and the proposal's seen mass summed over the
  // prefix in order, fp64 from the fp32 masses).
  // An unseen proposed id (the pure regime): the row keeps the deterministic rule.
  const bool ratio = qmass != nullptr && q_draft > 0.0 && (complete || q_missing == 0);
  double residual_total = 1.0 - p_draft;
  if (ratio) {
    double overlap = 0.0;
    for (int i = 0; i < held; ++i) {
      const double qi = static_cast<double>(qmass[i]);
      if (qi <= 0.0) continue;
      overlap += mass[i] < qi ? mass[i] : qi;
    }
    residual_total = 1.0 - overlap;
  }
  const uint64_t entry = *counter;
  const double u1 = uniform01(s.seed, *counter);
  *counter += 1;
  const float lse = static_cast<float>(Z);
  if (ratio ? (p_draft > u1 * q_draft) : (p_draft > u1)) {
    d.resolved = true;
    d.accepted = true;
    d.token = draft;
    d.logprob = __fsub_rn(__fdiv_rn(logit[j], T), lse);
    return d;
  }
  const double u2 = uniform01(s.seed, *counter);
  *counter += 1;
  const double threshold = u2 * residual_total;
  double cumulative = 0.0;
  int chosen = held;
  int last = held;  // the last residual token with mass
  for (int i = 0; i < held; ++i) {
    if (!ratio && i == j) continue;
    double m = mass[i];
    if (ratio) {
      const double qi = static_cast<double>(qmass[i]);
      m = m > qi ? m - qi : 0.0;
    }
    if (m > 0.0 || last == held) last = i;
    cumulative += m;
    if (cumulative > threshold) {
      chosen = i;
      break;
    }
  }
  if (chosen == held) {
    if (!complete) {
      *counter = entry;
      return d;
    }
    chosen = last;
  }
  d.resolved = true;
  d.token = id[chosen];
  d.logprob = __fsub_rn(__fdiv_rn(logit[chosen], T), lse);
  return d;
}

// sample::merge_logsumexp's pivot: the largest slice lse.
__device__ inline double lse_top(const double* lses, int world) {
  double top = lses[0];
  for (int r = 1; r < world; ++r) top = lses[r] > top ? lses[r] : top;
  return top;
}

// The number of keys below (hi, lo) in an ascending split-key list.
__device__ inline int lower_bound_split(const uint32_t* hi, const uint32_t* lo,
                                        int n, uint32_t khi, uint32_t klo) {
  int l = 0, r = n;
  while (l < r) {
    const int m = (l + r) >> 1;
    if (key_less(hi[m], lo[m], khi, klo))
      l = m + 1;
    else
      r = m;
  }
  return l;
}

// The verdict kernel's dynamic shared layout (bytes): the rows' fold
// masses (double), the split composite keys of one row (hi, lo), the rows'
// merged logits, ids and selector exps, the proposals' masses.
constexpr size_t kVerdictSmemMass = 0;
constexpr size_t kVerdictSmemCHi =
    kVerdictSmemMass + sizeof(double) * kSampleVerdictRows * kSampleMaxCandidates;
constexpr size_t kVerdictSmemCLo =
    kVerdictSmemCHi + sizeof(uint32_t) * kPickMaxWorld * kSampleMaxCandidates;
constexpr size_t kVerdictSmemLogit =
    kVerdictSmemCLo + sizeof(uint32_t) * kPickMaxWorld * kSampleMaxCandidates;
constexpr size_t kVerdictSmemId =
    kVerdictSmemLogit + sizeof(float) * kSampleVerdictRows * kSampleMaxCandidates;
constexpr size_t kVerdictSmemExpSrc =
    kVerdictSmemId + sizeof(int32_t) * kSampleVerdictRows * kSampleMaxCandidates;
constexpr size_t kVerdictSmemQMass =
    kVerdictSmemExpSrc + sizeof(float) * kSampleVerdictRows * kSampleMaxCandidates;
constexpr size_t kVerdictDynamicSmemBytes =
    kVerdictSmemQMass + sizeof(float) * kSampleProposalSlots * kSampleMaxCandidates;

__global__ void __launch_bounds__(kVerdictThreads) sample_verdict_kernel(
    const uint16_t* __restrict__ table, int rows, int world, int candidates, int vocab_size,
    SampleSpec* __restrict__ specs, int rows_per_request, const int64_t* __restrict__ fed,
    const int64_t* __restrict__ positions, int position_stride, int32_t* __restrict__ counts,
    const uint32_t* __restrict__ masks, int mask_stride, PickVerdict* __restrict__ verdicts,
    PickVerdict* __restrict__ device_verdicts, SampleOutcome* __restrict__ outcomes,
    const DraftProposal* __restrict__ proposals_in, DraftProposal* __restrict__ proposals_out,
    DraftProposal* __restrict__ proposals_out_host, int draft_index, const int32_t* request_map,
    int block_verify) {
  dgpp_pdl_wait();
  // One row's split composite keys at a time (hi = ~primary, lo = id; the
  // empty slot is the maximal key) — the merge is per row, and T rows of
  // keys would not fit the static shared bound — with every row's merged
  // prefix and its masses kept for the decision (2026-09-06: the T-row
  // chain; the arithmetic per row is the two-row kernel's).
  constexpr int R = kSampleVerdictRows;
  constexpr int kQ = kSampleProposalSlots;
  // The per-row tables live in dynamic shared memory (six rows of them
  // pass the 48 KiB static bound: kVerdictDynamicSmemBytes, set on the
  // kernel once by device_sample_verdict_prepare); the doubles first for
  // their alignment. The small per-row scalars stay static.
  extern __shared__ __align__(16) unsigned char verdict_smem[];
  double (*mass)[kSampleMaxCandidates] =
      reinterpret_cast<double (*)[kSampleMaxCandidates]>(verdict_smem + kVerdictSmemMass);
  uint32_t* c_hi = reinterpret_cast<uint32_t*>(verdict_smem + kVerdictSmemCHi);
  uint32_t* c_lo = reinterpret_cast<uint32_t*>(verdict_smem + kVerdictSmemCLo);
  float (*m_logit)[kSampleMaxCandidates] =
      reinterpret_cast<float (*)[kSampleMaxCandidates]>(verdict_smem + kVerdictSmemLogit);
  int32_t (*m_id)[kSampleMaxCandidates] =
      reinterpret_cast<int32_t (*)[kSampleMaxCandidates]>(verdict_smem + kVerdictSmemId);
  float (*expsrc)[kSampleMaxCandidates] =
      reinterpret_cast<float (*)[kSampleMaxCandidates]>(verdict_smem + kVerdictSmemExpSrc);
  // The draft rows' proposals, gathered onto each row's merged candidate
  // list before the scalar decision: row t tests the draft fed
  // to row t + 1 against proposal slot t (the chained drafts of depth >= 2
  // carry theirs since the evening).
  float (*qmass_s)[kSampleMaxCandidates] =
      reinterpret_cast<float (*)[kSampleMaxCandidates]>(verdict_smem + kVerdictSmemQMass);
  __shared__ double lses[R][kPickMaxWorld];
  __shared__ float exps[kSampleMaxCandidates];
  __shared__ double prefix[kSampleMaxCandidates];
  __shared__ double lse_terms[R][kPickMaxWorld];
  __shared__ int total[R];
  __shared__ double Z[R];
  __shared__ double q_draft_s[kQ];
  __shared__ int q_missing_s[kQ];  // proposed ids (with mass) outside the held prefix
  __shared__ int q_live_s[kQ];

  const int q = blockIdx.x;
  const int req = request_map ? request_map[q] : q;
  const int row0 = q * rows_per_request;
  const int tid = threadIdx.x;
  const size_t group = device_sample_rank_group_slots(candidates);
  const bool active = req >= 0 && (positions == nullptr || positions[q * position_stride] >= 0);
  if (!active) {
    if (tid == 0) {
      PickVerdict v;
      v.rows = 0;
      v.accepted = 0;
      v.next = kNoId;
      verdicts[q] = v;
      if (device_verdicts != nullptr) device_verdicts[q] = v;
      outcomes[q] = SampleOutcome{};
    }
    return;
  }
  SampleSpec spec = specs[req];
  if (proposals_out != nullptr && rows_per_request == 1 &&
      spec.draft_temperature > 0.0f && spec.temperature > 0.0f)
    spec.temperature = spec.draft_temperature;  // the draft pick's temperature
  const int entries = world * candidates;
  if (tid < R) total[tid] = 0;
  __syncthreads();
  // A constrained row's vocabulary is its allowed count (the complete list
  // is the allowed set); a masked draft is excluded from its row outright.
  int row_vocab[R];
  bool constrained[R];
  const uint32_t* row_mask[R];
  for (int t = 0; t < R; ++t) {
    row_vocab[t] = vocab_size;
    constrained[t] = false;
    row_mask[t] = nullptr;
  }
  for (int t = 0; t < rows_per_request; ++t) {
    if (masks == nullptr) continue;
    const uint32_t* m = masks + static_cast<size_t>(row0 + t) * mask_stride;
    if (m[0] != 0u) {
      constrained[t] = true;
      row_vocab[t] = static_cast<int>(m[0]);
      row_mask[t] = m + 1;
    }
  }

  // 1 + 2. Per row: decode every rank's group (cooperative), then the k-way
  //    merge in canonical order (sample::merge_topk): every rank's list is
  //    canonical with its empties last, so a candidate's rank in the union
  //    is its index in its own list plus the count of smaller keys in every
  //    other list; the ids are disjoint across ranks, so the ranks are a
  //    permutation and the first `held` land.
  int held[R] = {};
  for (int t = 0; t < rows_per_request; ++t) {
    const uint16_t* row_base =
        table + static_cast<size_t>(row0 + t) * world * group;
    for (int i = tid; i < entries; i += kVerdictThreads) {
      const int r = i / candidates;
      const int j = i % candidates;
      const uint16_t* slot = row_base + static_cast<size_t>(r) * group +
                             static_cast<size_t>(j) * kPickSlotsPerRank;
      const uint32_t id = static_cast<uint32_t>(
          decode_digits(slot + kPickLogitDigits, kPickIdDigits));
      if (id >= static_cast<uint32_t>(vocab_size)) {
        c_hi[i] = kSplitMax;
        c_lo[i] = kSplitMax;
      } else {
        const float l = __uint_as_float(
            static_cast<uint32_t>(decode_digits(slot, kPickLogitDigits)));
        c_hi[i] = ~primary_key(l);
        c_lo[i] = id;
        atomicAdd(&total[t], 1);
      }
    }
    for (int r = tid; r < world; r += kVerdictThreads)
      lses[t][r] = detmath::double_of(decode_digits(
          row_base + static_cast<size_t>(r) * group +
              static_cast<size_t>(candidates) * kPickSlotsPerRank,
          kSampleLseDigits));
    __syncthreads();
    held[t] = min(candidates, total[t]);
    for (int i = tid; i < entries; i += kVerdictThreads) {
      const uint32_t khi = c_hi[i];
      const uint32_t klo = c_lo[i];
      if (khi == kSplitMax && klo == kSplitMax) continue;
      const int r = i / candidates;
      const int j = i % candidates;
      int position = j;
      for (int o = 0; o < world; ++o)
        if (o != r)
          position += lower_bound_split(c_hi + o * candidates,
                                        c_lo + o * candidates, candidates,
                                        khi, klo);
      if (position < held[t]) {
        const uint16_t* slot = row_base + static_cast<size_t>(r) * group +
                               static_cast<size_t>(j) * kPickSlotsPerRank;
        m_logit[t][position] = __uint_as_float(
            static_cast<uint32_t>(decode_digits(slot, kPickLogitDigits)));
        m_id[t][position] = static_cast<int32_t>(klo);
      }
    }
    __syncthreads();  // the next row's keys overwrite this row's
  }

  // 3. The fold normalizers (sample::merge_logsumexp: the slices'
  //    terms in parallel, summed in rank order), and for the stochastic
  //    decision every candidate's fold mass and selector exp in parallel
  //    — the very values the host's loops compute one by one.
  if (tid < rows_per_request * world) {
    const int t = tid / world;
    const int r = tid % world;
    lse_terms[t][r] = detmath::exp_d(lses[t][r] - lse_top(lses[t], world));
  }
  __syncthreads();
  if (tid < rows_per_request) {
    double sum = 0.0;
    for (int r = 0; r < world; ++r) sum += lse_terms[tid][r];
    Z[tid] = held[tid] > 0 ? lse_top(lses[tid], world) + detmath::log_d(sum)
                           : 0.0;
  }
  __syncthreads();
  const bool stochastic = spec.temperature > 0.0f && held[0] > 0;
  const bool full_path = stochastic || spec.logprobs >= 0 || constrained[0] ||
                         spec.repetition_penalty != 1.0f ||
                         spec.frequency_penalty != 0.0f ||
                         spec.presence_penalty != 0.0f;
  (void)full_path;
  if (stochastic) {
    const float T = spec.temperature;
    for (int t = 0; t < rows_per_request; ++t) {
      if (held[t] == 0) continue;
      const float scaled0 = __fdiv_rn(m_logit[t][0], T);
      for (int i = tid; i < held[t]; i += kVerdictThreads) {
        const float scaled = __fdiv_rn(m_logit[t][i], T);
        mass[t][i] = detmath::exp_d(static_cast<double>(scaled) - Z[t]);
        expsrc[t][i] = detmath::exp_f(__fsub_rn(scaled, scaled0));
      }
    }
  }
  if (tid < kQ) {
    q_live_s[tid] = 0;
    q_draft_s[tid] = 0.0;
    q_missing_s[tid] = 0;
  }
  __syncthreads();
  // Each draft row's proposal, when this pick was fed the draft it drew.
  if (proposals_in != nullptr && rows_per_request > 1 && stochastic) {
    for (int t = 0; t + 1 < rows_per_request && t < kQ; ++t) {
      if (held[t] == 0) continue;
      const DraftProposal* prop =
          proposals_in + static_cast<size_t>(req) * kSampleProposalSlots + t;
      const int32_t draft_t = static_cast<int32_t>(fed[row0 + t + 1]);
      if (prop->n <= 0 || prop->token != draft_t) continue;
      for (int i = tid; i < held[t]; i += kVerdictThreads) {
        const int32_t want = m_id[t][i];
        float m = 0.0f;
        for (int e = 0; e < prop->n; ++e)
          if (prop->ids[e] == want) {
            m = prop->mass[e];
            break;
          }
        qmass_s[t][i] = m;
      }
      if (tid == 0) {
        double qd = 0.0;
        int missing = 0;
        bool found = false;
        for (int e = 0; e < prop->n; ++e) {
          if (!found && prop->ids[e] == draft_t) {
            qd = static_cast<double>(prop->mass[e]);  // the first match, as Proposal::at
            found = true;
          }
          if (!(prop->mass[e] > 0.0f)) continue;
          bool held_id = false;
          for (int i = 0; i < held[t]; ++i)
            if (m_id[t][i] == prop->ids[e]) { held_id = true; break; }
          if (!held_id) ++missing;
        }
        if (qd > 0.0) {
          q_draft_s[t] = qd;
          q_missing_s[t] = missing;
          q_live_s[t] = 1;
        }
      }
    }
  }
  __syncthreads();
  if (tid != 0) return;

  // 4. The decision (thread 0).
  PickVerdict v;
  SampleOutcome o;
  const bool penalized = spec.repetition_penalty != 1.0f ||
                         spec.frequency_penalty != 0.0f ||
                         spec.presence_penalty != 0.0f;
  TopReport tops[R];
  const TopReport* reps[R];
  for (int t = 0; t < R; ++t) {
    tops[t] = TopReport{spec.logprobs, o.top_ids[t], o.top_logprobs[t],
                        &o.top_count[t]};
    reps[t] = spec.logprobs >= 0 ? &tops[t] : nullptr;
  }

  v.rows = rows_per_request;
  if (!stochastic) {
    // The greedy request: merge_greedy per row, the greedy judge. With
    // logprobs (or penalties) the rows came through the full path at
    // temperature 1 and report under the raw normalizer
    // (sample::greedy_from_prefix).
    for (int t = 0; t < rows_per_request; ++t)
      v.winners[t] = held[t] > 0 ? m_id[t][0] : kNoId;
    v.accepted = 1;
    while (v.accepted < rows_per_request &&
           v.winners[v.accepted - 1] == fed[row0 + v.accepted])
      ++v.accepted;
    v.next = v.winners[v.accepted - 1];
    o.sampled = 0;
    o.counter = spec.counter;
    if ((spec.logprobs >= 0 || penalized) && held[0] > 0) {
      for (int t = 0; t < rows_per_request; ++t) {
        if (held[t] == 0) continue;
        const double Zt = Z[t];
        o.normalizer[t] = Zt;
        o.logprob[t] = __fsub_rn(m_logit[t][0], static_cast<float>(Zt));
        if (reps[t] != nullptr)
          report_top(m_logit[t], m_id[t], held[t], 1.0f,
                     static_cast<float>(Zt), spec.logprobs, o.top_ids[t],
                     o.top_logprobs[t], &o.top_count[t]);
      }
    }
  } else {
    // The sampled chain: row t < T-1 tests the draft fed to row t+1
    // (sample::spec_accept_from_prefix) — a stand moves to the next row, a
    // reject ends the step on the residual token, an undecidable row falls
    // back to the host from row t with the rows before it committed. The
    // last row reached is sampled plainly. T = 1 is the plain sample.
    // The draft pick draws from the draft stream (kSampleDraftSeedMix), so
    // the request's own counter — the host's draw accounting — is only ever
    // advanced by the verify.
    const bool draft_mode = proposals_out != nullptr && rows_per_request == 1;
    SampleSpec use = spec;
    // The draft stream, keyed by the draft's index too: the chained drafts
    // of one step share the request's counter and must not share a draw.
    if (draft_mode)
      use.seed = spec.seed ^ kSampleDraftSeedMix ^
                 (static_cast<uint64_t>(draft_index + 1) * 0xD1B54A32D192ED03ull);
    uint64_t counter = spec.counter;
    o.sampled = 1;
    v.accepted = 1;
    bool done = false;
    // Block verification (sample::block_verify; Sun et al. 2024, Algorithm
    // 2): the chain decided jointly — p_i = min(1, p_{i-1} P_{i-1}(X_i) /
    // Q_{i-1}(X_i)) along the drafts, h_i = S_i / (S_i + 1 - p_i) with
    // S_i = sum_x max(p_i P_i(x) - Q_i(x), 0) (h_gamma = p_gamma), tau the
    // largest i with eta_i <= h_i, then the residual (p_tau P_tau - Q_tau)+
    // at row tau, or the last row sampled plainly at tau = gamma. Exact for
    // the output distribution and never fewer tokens in expectation than
    // the token rule. Taken only when every row prices its draft and, in
    // the pure regime, holds every id its proposal gives mass (a choice the
    // draws never see); else the token rule below.
    if (block_verify != 0 && !draft_mode && rows_per_request >= 2) {
      const int gamma = rows_per_request - 1;
      const float T = use.temperature;
      // The pre-check and the chain's p_i, with the draft's index per row.
      int kind[R], draft_j[R], fin[R], supn[R], s_topk[R];
      bool row_live[R];
      float s_minp[R], s_topp[R];
      double pcum[R + 1], cover[R];
      float den[R], lse_r[R];
      bool decidable = true;
      pcum[0] = 1.0;
      for (int t = 0; t < gamma && decidable; ++t) {
        kind[t] = 0;
        if (held[t] == 0) { decidable = false; break; }
        const Support sup = resolve_support(m_logit[t], mass[t], prefix, held[t], row_vocab[t], use);
        cover[t] = sup.covered;
        if (sup.kind == 0) { decidable = false; break; }
        kind[t] = sup.kind;
        supn[t] = sup.n; s_topk[t] = sup.top_k; s_minp[t] = sup.min_p; s_topp[t] = sup.top_p;
        const int32_t draft = static_cast<int32_t>(fed[row0 + t + 1]);
        const bool excluded = row_mask[t] != nullptr &&
                              (draft < 0 || draft >= vocab_size || !mask_allows(row_mask[t], draft));
        const bool complete = held[t] == row_vocab[t];
        if (sup.kind == 1) {
          int fc = 0; float fd = 0.0f, ls = 0.0f;
          selector_state(m_logit[t], expsrc[t], sup.n, T, sup.top_k, sup.min_p, sup.top_p, exps, &fc, &fd, &ls);
          fin[t] = fc; den[t] = fd; lse_r[t] = ls;
        } else {
          fin[t] = held[t]; den[t] = 0.0f; lse_r[t] = static_cast<float>(Z[t]);
        }
        int j = fin[t];
        if (!excluded)
          for (int i = 0; i < fin[t]; ++i)
            if (m_id[t][i] == draft) { j = i; break; }
        if (j == fin[t] && !excluded && sup.kind == 2 && !complete) { decidable = false; break; }
        draft_j[t] = j;
        // Q at the draft and, in the pure regime, the proposal's whole mass inside the prefix.
        // A live proposal whose ids are all inside the prefix (the pure
        // regime; a materialized set prices them all): else a point mass.
        const bool live = t < kQ && q_live_s[t] != 0 && !(sup.kind == 2 && !complete && q_missing_s[t] > 0);
        row_live[t] = live;
        const double q_at = live ? q_draft_s[t] : 1.0;
        const double p_x = j < fin[t] ? (sup.kind == 1 ? static_cast<double>(__fdiv_rn(exps[j], den[t])) : mass[t][j]) : 0.0;
        double pi = pcum[t] * (p_x / q_at);
        if (pi > 1.0) pi = 1.0;
        pcum[t + 1] = pi;
      }
      if (decidable) {
        // h_i per position and tau = the largest accepted sub-block.
        int tau = 0;
        for (int i = 1; i <= gamma; ++i) {
          const double pi = pcum[i];
          double h;
          if (i == gamma) {
            h = pi;
          } else {
            const int t = i;  // the distributions after the i-th draft: row i
            const bool live = row_live[t];
            const int32_t next_draft = static_cast<int32_t>(fed[row0 + t + 1]);
            double S = 0.0;
            if (kind[t] == 1) {
              int fc = 0; float fd = 0.0f, ls = 0.0f;  // the row's final set again (the same exps)
              selector_state(m_logit[t], expsrc[t], supn[t], T, s_topk[t], s_minp[t], s_topp[t], exps, &fc, &fd, &ls);
              for (int c = 0; c < fin[t]; ++c) {
                const double px = static_cast<double>(__fdiv_rn(exps[c], den[t]));
                const double qx = live ? static_cast<double>(qmass_s[t][c]) : (m_id[t][c] == next_draft ? 1.0 : 0.0);
                const double r = pi * px - qx;
                if (r > 0.0) S += r;
              }
            } else {
              for (int c = 0; c < held[t]; ++c) {
                const double qx = live ? static_cast<double>(qmass_s[t][c]) : (m_id[t][c] == next_draft ? 1.0 : 0.0);
                const double r = pi * mass[t][c] - qx;
                if (r > 0.0) S += r;
              }
              S += pi * (1.0 - cover[t]);  // the unseen tail carries no proposal mass
            }
            h = S / (S + 1.0 - pi);
          }
          const double eta = uniform01(use.seed, counter + static_cast<uint64_t>(i - 1));
          if (eta <= h) tau = i;
        }
        counter += static_cast<uint64_t>(gamma);
        // Rows [0, tau): the drafts stand.
        for (int t = 0; t < tau; ++t) {
          const int j = draft_j[t];
          v.winners[t] = static_cast<int32_t>(fed[row0 + t + 1]);
          o.normalizer[t] = Z[t];
          o.covered_mass[t] = cover[t];
          o.logprob[t] = j < fin[t] ? __fsub_rn(__fdiv_rn(m_logit[t][j], T), lse_r[t]) : 0.0f;
          if (reps[t] != nullptr)
            report_top(m_logit[t], m_id[t], fin[t], T, lse_r[t], spec.logprobs, o.top_ids[t], o.top_logprobs[t], &o.top_count[t]);
        }
        if (tau >= 1) o.accepted_draft = 1;
        v.accepted = tau + 1;
        o.block = 1;
        if (tau < gamma) {
          // The block residual at row tau: (p_tau P - Q)+ over the row's set
          // (plus the pure regime's unseen tail, a fallback when drawn).
          const int t = tau;
          const double pt = pcum[t];
          const bool live = row_live[t];
          const int32_t draft = static_cast<int32_t>(fed[row0 + t + 1]);
          const bool complete = held[t] == row_vocab[t];
          if (kind[t] == 1) {
            int fc = 0; float fd = 0.0f, ls = 0.0f;
            selector_state(m_logit[t], expsrc[t], supn[t], T, s_topk[t], s_minp[t], s_topp[t], exps, &fc, &fd, &ls);
          }
          double total = 0.0;
          for (int c = 0; c < fin[t]; ++c) {
            const double px = kind[t] == 1 ? static_cast<double>(__fdiv_rn(exps[c], den[t])) : mass[t][c];
            const double qx = live ? static_cast<double>(qmass_s[t][c]) : (m_id[t][c] == draft ? 1.0 : 0.0);
            const double r = pt * px - qx;
            if (r > 0.0) total += r;
          }
          if (kind[t] == 2) total += pt * (1.0 - cover[t]);
          const double u = uniform01(use.seed, counter);
          const double threshold = u * total;
          double cum = 0.0;
          int chosen = fin[t], last = fin[t];
          for (int c = 0; c < fin[t]; ++c) {
            const double px = kind[t] == 1 ? static_cast<double>(__fdiv_rn(exps[c], den[t])) : mass[t][c];
            const double qx = live ? static_cast<double>(qmass_s[t][c]) : (m_id[t][c] == draft ? 1.0 : 0.0);
            const double r = pt * px - qx > 0.0 ? pt * px - qx : 0.0;
            if (r > 0.0 || last == fin[t]) last = c;
            cum += r;
            if (cum > threshold) { chosen = c; break; }
          }
          o.normalizer[t] = Z[t];
          o.covered_mass[t] = cover[t];
          if (chosen == fin[t] && !(kind[t] == 1 || complete)) {
            // The residual lies in the unseen tail: the host draws it with p_tau.
            o.fallback = 1;
            o.fallback_row = t;
            o.block_p = pt;
            v.winners[t] = held[t] > 0 ? m_id[t][0] : kNoId;
          } else {
            if (chosen == fin[t]) chosen = last;
            counter += 1;
            v.winners[t] = m_id[t][chosen];
            o.logprob[t] = __fsub_rn(__fdiv_rn(m_logit[t][chosen], T), lse_r[t]);
            if (reps[t] != nullptr)
              report_top(m_logit[t], m_id[t], fin[t], T, lse_r[t], spec.logprobs, o.top_ids[t], o.top_logprobs[t], &o.top_count[t]);
          }
          v.next = v.winners[t];
        } else {
          // Every draft stood: the last row is sampled plainly.
          const int t = gamma;
          const double Zt = Z[t];
          o.normalizer[t] = Zt;
          const Decision none{false, false, kNoId, 0.0f, 0.0};
          const Decision d = held[t] > 0 ? decide_prefix(m_logit[t], m_id[t], mass[t], prefix, expsrc[t], held[t],
                                                         row_vocab[t], Zt, use, exps, &counter, reps[t])
                                         : none;
          o.covered_mass[t] = d.covered;
          if (d.resolved) {
            v.winners[t] = d.token;
            o.logprob[t] = d.logprob;
          } else {
            o.fallback = 1;
            o.fallback_row = t;
            v.winners[t] = held[t] > 0 ? m_id[t][0] : kNoId;
          }
          v.next = v.winners[t];
        }
        done = true;
      }
    }
    for (int t = 0; t < rows_per_request && !done; ++t) {
      const double Zt = Z[t];
      o.normalizer[t] = Zt;
      const Decision none{false, false, kNoId, 0.0f, 0.0};
      if (t + 1 < rows_per_request) {
        const int32_t draft = static_cast<int32_t>(fed[row0 + t + 1]);
        const bool draft_excluded =
            row_mask[t] != nullptr &&
            (draft < 0 || draft >= vocab_size ||
             !mask_allows(row_mask[t], draft));
        const bool ratio = t < kQ && q_live_s[t] != 0;
        const Decision d =
            held[t] > 0 ? spec_decide_prefix(m_logit[t], m_id[t], mass[t],
                                             prefix, expsrc[t], held[t],
                                             row_vocab[t], Zt, draft, use,
                                             exps, &counter, reps[t],
                                             draft_excluded,
                                             ratio ? qmass_s[t] : nullptr,
                                             ratio ? q_draft_s[t] : 0.0,
                                             ratio ? q_missing_s[t] : 0)
                        : none;
        o.covered_mass[t] = d.covered;
        if (!d.resolved) {
          // Provisional REJECT: the commit keeps the state through row t,
          // the host decides row t (and the rows after it if the draft
          // stands) between windows.
          o.fallback = 1;
          o.fallback_row = t;
          v.accepted = t + 1;
          v.winners[t] = held[t] > 0 ? m_id[t][0] : kNoId;
          v.next = v.winners[t];
          done = true;
        } else if (!d.accepted) {
          v.accepted = t + 1;
          v.winners[t] = d.token;
          v.next = d.token;
          o.logprob[t] = d.logprob;
          done = true;
        } else {
          if (t == 0) o.accepted_draft = 1;
          v.accepted = t + 2;
          v.winners[t] = draft;
          o.logprob[t] = d.logprob;
        }
      } else {
        // The draft pick (one row per request) keeps the set it drew from.
        DraftProposal* out =
            (proposals_out != nullptr && rows_per_request == 1)
                ? proposals_out + static_cast<size_t>(req) * kSampleProposalSlots + draft_index
                : nullptr;
        if (out != nullptr) out->n = 0;
        const Decision d =
            held[t] > 0 ? decide_prefix(m_logit[t], m_id[t], mass[t], prefix,
                                        expsrc[t], held[t], row_vocab[t], Zt,
                                        use, exps, &counter, reps[t], out)
                        : none;
        if (out != nullptr) {
          out->token = d.resolved ? d.token : kNoId;
          if (!d.resolved) out->n = 0;
          if (proposals_out_host != nullptr) {
            DraftProposal* mirror =
                proposals_out_host + static_cast<size_t>(req) * kSampleProposalSlots + draft_index;
            mirror->n = out->n;
            mirror->token = out->token;
            for (int e = 0; e < out->n; ++e) {
              mirror->ids[e] = out->ids[e];
              mirror->mass[e] = out->mass[e];
            }
          }
        }
        o.covered_mass[t] = d.covered;
        if (d.resolved) {
          v.winners[t] = d.token;
          o.logprob[t] = d.logprob;
        } else {
          o.fallback = 1;
          o.fallback_row = t;
          v.winners[t] = held[t] > 0 ? m_id[t][0] : kNoId;
        }
        v.next = v.winners[t];
        done = true;
      }
    }
    o.counter = counter;
    if (!draft_mode) specs[req].counter = counter;
  }

  // 5. The commit of the step's fed tokens into the request's context:
  //    the token this step consumed always, a draft only when it stood
  //    (a provisional reject leaves it out; the host adds it back if its
  //    decision accepts — glm_sample_adjust_count).
  if (counts != nullptr) {
    int32_t* counts_q = counts + static_cast<size_t>(req) * vocab_size;
    const int64_t consumed = fed[row0];
    if (consumed >= 0 && consumed < vocab_size) counts_q[consumed] += 1;
    for (int t = 1; t < v.accepted; ++t) {
      const int64_t draft = fed[row0 + t];
      if (draft >= 0 && draft < vocab_size) counts_q[draft] += 1;
    }
  }

  verdicts[q] = v;
  if (device_verdicts != nullptr) device_verdicts[q] = v;
  outcomes[q] = o;
}

// ---------------------------------------------------------------------------
// Kernel 2b: the digest chain over every request (glm_pick_verdict's), the
// digest group's agreement, and the carry.
// ---------------------------------------------------------------------------
__global__ void sample_digest_kernel(const uint16_t* __restrict__ digests,
                                     int world, int rank, int requests,
                                     int rows_per_request,
                                     PickVerdict* __restrict__ verdicts,
                                     PickVerdict* __restrict__ device_verdicts,
                                     uint64_t* __restrict__ carry_digest) {
  dgpp_pdl_wait();
  PickVerdict* src = device_verdicts != nullptr ? device_verdicts : verdicts;
  uint64_t digest = requests == 1 ? 0 : splitmix64(requests);
  for (int q = 0; q < requests; ++q) {
    const uint64_t one = verdict_digest(src[q].rows, src[q].accepted, src[q].winners);
    if (requests == 1)
      digest = one;
    else
      digest = splitmix64(digest ^ one);
  }
  digest &= (1ull << kPickDigestBits) - 1;
  const uint64_t mine =
      decode_digits(digests + rank * kPickSlotsPerRank, kPickSlotsPerRank);
  uint32_t mismatch = 0;
  uint64_t peers[kPickMaxWorld] = {};
  for (int k = 0; k < world; ++k) {
    const uint64_t d =
        decode_digits(digests + k * kPickSlotsPerRank, kPickSlotsPerRank);
    peers[k] = d;
    if (d != mine) mismatch |= 1u << k;
  }
  for (int q = 0; q < requests; ++q) {
    verdicts[q].digest = digest;
    verdicts[q].digest_mismatch = mismatch;
    for (int k = 0; k < world; ++k) verdicts[q].peer_digests[k] = peers[k];
    if (device_verdicts != nullptr) {
      device_verdicts[q].digest = digest;
      device_verdicts[q].digest_mismatch = mismatch;
      for (int k = 0; k < world; ++k)
        device_verdicts[q].peer_digests[k] = peers[k];
    }
  }
  *carry_digest = digest;
  (void)rows_per_request;
}

__global__ void count_tokens_kernel(int32_t* __restrict__ counts,
                                    const int64_t* __restrict__ ids, int n,
                                    int vocab_size) {
  dgpp_pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t t = ids[i];
  if (t >= 0 && t < vocab_size) atomicAdd(counts + t, 1);
}

void check_common(int rows, int world, int rank, int candidates,
                  int rows_per_request, const char* what) {
  if (rows < 1 || rows > kPickMaxRows)
    throw std::invalid_argument(std::string(what) + ": rows must be in [1, " +
                                std::to_string(kPickMaxRows) + "]");
  if (world < 1 || world > kPickMaxWorld || rank < 0 || rank >= world)
    throw std::invalid_argument(std::string(what) + ": rank/world");
  if (candidates < 1 || candidates > kSampleMaxCandidates)
    throw std::invalid_argument(std::string(what) + ": candidates must be in [1, " +
                                std::to_string(kSampleMaxCandidates) + "]");
  if (rows_per_request < 1 || rows_per_request > kSampleVerdictRows)
    throw std::invalid_argument(std::string(what) +
                                ": the device sampler decides T=1 rows or an "
                                "MTP verify of up to " +
                                std::to_string(kSampleVerdictRows) + " rows");
}

__global__ void adjust_count_kernel(int32_t* __restrict__ counts, int64_t token,
                                    int delta, int vocab_size) {
  dgpp_pdl_wait();
  if (threadIdx.x == 0 && token >= 0 && token < vocab_size)
    counts[token] += delta;
}

}  // namespace

void device_sample_local(float* logits, int rows, int vocab_count, int vocab_begin, int vocab_size,
                         int rank, int world, int candidates, const SampleSpec* specs,
                         int rows_per_request, const int64_t* fed, const int64_t* positions,
                         int position_stride, const int32_t* counts, const float* bias,
                         const uint32_t* masks, int mask_stride, const uint64_t* carry_digest,
                         uint16_t* table, PickLocal* locals, double* scratch, cudaStream_t stream,
                         const PickVerdict* row_select, int source_row_stride,
                         const int32_t* request_map) {
  check_common(rows, world, rank, candidates, rows_per_request,
               "glm_sample_local");
  if (row_select != nullptr &&
      (rows_per_request != 1 || source_row_stride < 1))
    throw std::invalid_argument(
        "glm_sample_local: a selected row needs one row per request and a "
        "source stride");
  if (masks != nullptr && mask_stride < device_sample_mask_words(vocab_size))
    throw std::invalid_argument("glm_sample_local: mask stride");
  if (logits == nullptr || specs == nullptr || fed == nullptr ||
      carry_digest == nullptr || table == nullptr ||
      locals == nullptr || scratch == nullptr)
    throw std::invalid_argument("glm_sample_local: null argument");
  if (vocab_count < 1 || vocab_begin < 0 || vocab_size < 1 ||
      vocab_begin + vocab_count > vocab_size ||
      vocab_size > (1 << (6 * kPickIdDigits)) || vocab_size > (1 << kKeyIdxBits))
    throw std::invalid_argument(
        "glm_sample_local: vocab slice outside the id encodings");
  const int nchunks = (vocab_count + kSampleLseChunk - 1) / kSampleLseChunk;
  if (nchunks > kSampleMaxChunks)
    throw std::invalid_argument(
        "glm_sample_local: the slice has more normalizer chunks than the "
        "row block has threads");
  if (positions != nullptr && position_stride < 1)
    throw std::invalid_argument("glm_sample_local: position stride");
  if (rows % rows_per_request != 0)
    throw std::invalid_argument("glm_sample_local: rows per request");
  double* maxes = scratch;
  double* partials = scratch + static_cast<size_t>(rows) * nchunks;
  const dim3 chunk_grid(static_cast<unsigned>(nchunks), static_cast<unsigned>(rows));
  sample_prepare_kernel<<<chunk_grid, kChunkThreads, 0, stream>>>(
      logits, vocab_count, vocab_begin, vocab_size, specs, rows_per_request, fed, positions,
      position_stride, counts, bias, masks, mask_stride, maxes, row_select, source_row_stride,
      request_map);
  DGPP_CUDA_OK(cudaGetLastError());
  sample_partials_kernel<<<chunk_grid, kChunkThreads, 0, stream>>>(
      logits, vocab_count, specs, rows_per_request, positions, position_stride, masks, mask_stride,
      maxes, partials, row_select, source_row_stride, request_map);
  DGPP_CUDA_OK(cudaGetLastError());
  sample_local_kernel<<<rows, kLocalThreads, 0, stream>>>(
      logits, rows, vocab_count, vocab_begin, rank, world, candidates, specs, rows_per_request,
      positions, position_stride, masks, mask_stride, maxes, partials, nchunks, carry_digest, table,
      locals, row_select, source_row_stride, request_map);
  DGPP_CUDA_OK(cudaGetLastError());
}

void device_sample_verdict_prepare() {
  // Once per process: the kernel's dynamic shared memory above the 48 KiB
  // default (the six-row tables). A stream capture records the launch as
  // any other; the attribute is a function property, set here before the
  // first launch and by the picker's constructor before any capture.
  static std::once_flag once;
  std::call_once(once, [] {
    DGPP_CUDA_OK(cudaFuncSetAttribute(sample_verdict_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(kVerdictDynamicSmemBytes)));
  });
}

void device_sample_verdict(const uint16_t* table, int rows, int world, int rank, int candidates,
                           int vocab_size, SampleSpec* specs, int requests, int rows_per_request,
                           const int64_t* fed, const int64_t* positions, int position_stride,
                           int32_t* counts, const uint32_t* masks, int mask_stride,
                           PickVerdict* verdicts, PickVerdict* device_verdicts,
                           SampleOutcome* outcomes, uint64_t* carry_digest, cudaStream_t stream,
                           const DraftProposal* proposals_in, DraftProposal* proposals_out,
                           DraftProposal* proposals_out_host, int draft_index,
                           const int32_t* request_map, int block_verify) {
  check_common(rows, world, rank, candidates, rows_per_request,
               "glm_sample_verdict");
  if (proposals_out != nullptr &&
      (rows_per_request != 1 || draft_index < 0 ||
       draft_index >= kSampleProposalSlots))
    throw std::invalid_argument(
        "glm_sample_verdict: a proposal is written by a one-row draft pick");
  if (masks != nullptr && mask_stride < device_sample_mask_words(vocab_size))
    throw std::invalid_argument("glm_sample_verdict: mask stride");
  if (requests < 1 || requests > kPickMaxRequests ||
      requests * rows_per_request != rows)
    throw std::invalid_argument("glm_sample_verdict: request shape");
  if (table == nullptr || specs == nullptr || verdicts == nullptr ||
      outcomes == nullptr || carry_digest == nullptr || fed == nullptr)
    throw std::invalid_argument("glm_sample_verdict: null argument");
  if (positions != nullptr && position_stride < rows_per_request)
    throw std::invalid_argument("glm_sample_verdict: position stride");
  device_sample_verdict_prepare();
  sample_verdict_kernel<<<requests, kVerdictThreads, kVerdictDynamicSmemBytes, stream>>>(
      table, rows, world, candidates, vocab_size, specs, rows_per_request, fed, positions,
      position_stride, counts, masks, mask_stride, verdicts, device_verdicts, outcomes,
      proposals_in, proposals_out, proposals_out_host, draft_index, request_map, block_verify);
  DGPP_CUDA_OK(cudaGetLastError());
  const uint16_t* digests =
      table + static_cast<size_t>(rows) * world *
                  device_sample_rank_group_slots(candidates);
  sample_digest_kernel<<<1, 1, 0, stream>>>(digests, world, rank, requests,
                                            rows_per_request, verdicts,
                                            device_verdicts, carry_digest);
  DGPP_CUDA_OK(cudaGetLastError());
}

void device_sample_adjust_count(int32_t* counts_row, int64_t token, int delta,
                             int vocab_size, cudaStream_t stream) {
  if (counts_row == nullptr || vocab_size < 1)
    throw std::invalid_argument("glm_sample_adjust_count: arguments");
  adjust_count_kernel<<<1, 32, 0, stream>>>(counts_row, token, delta,
                                            vocab_size);
  DGPP_CUDA_OK(cudaGetLastError());
}

void device_sample_count_tokens(int32_t* counts_row, const int64_t* ids, int n,
                             int vocab_size, cudaStream_t stream) {
  if (counts_row == nullptr || (n > 0 && ids == nullptr) || vocab_size < 1)
    throw std::invalid_argument("glm_sample_count_tokens: arguments");
  if (n <= 0) return;
  count_tokens_kernel<<<(n + 255) / 256, 256, 0, stream>>>(counts_row, ids, n,
                                                           vocab_size);
  DGPP_CUDA_OK(cudaGetLastError());
}


namespace {
__global__ void draft_confidence_kernel(const SampleOutcome* __restrict__ outcomes,
                                        int slot_stride, int request, int depth,
                                        float* __restrict__ dst) {
  dgpp_pdl_wait();
  const int c = static_cast<int>(threadIdx.x);
  if (c >= depth) return;
  // The outcome lives in pinned memory the verdict kernel wrote earlier on
  // this stream: a system-scope load reads what it stored.
  const float* lp = &outcomes[(1 + c) * slot_stride + request].logprob[0];
  float v;
  asm volatile("ld.relaxed.sys.global.f32 %0, [%1];" : "=f"(v) : "l"(lp) : "memory");
  // logit(p) = lp - log(1 - exp(lp)), bounded to [-16, 16] (p in ~[1e-7, 1 - 1e-7]).
  const float p = __expf(fminf(v, 0.0f));
  const float one_minus = fmaxf(1.0f - p, 1e-7f);
  float logit = fminf(v, 0.0f) - __logf(one_minus);
  logit = fminf(fmaxf(logit, -16.0f), 16.0f);
  dst[c] = logit;
}
}  // namespace

void device_sample_draft_confidence(const SampleOutcome* outcomes,
                                    int slot_stride, int request, int depth,
                                    float* dst, cudaStream_t stream) {
  if (outcomes == nullptr || dst == nullptr || slot_stride < 1 || request < 0 ||
      depth < 1 || depth > 32)
    throw std::invalid_argument("device_sample_draft_confidence: null argument/shape");
  draft_confidence_kernel<<<1, 32, 0, stream>>>(outcomes, slot_stride, request,
                                                depth, dst);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
