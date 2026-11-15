#pragma once
// Block-cooperative exact top-k over 64-bit composite keys (the DSA decode
// select's machinery, lifted so the sampler's local top-k shares it):
//   key = (~sortable_f32(logit) << idx_bits) | idx
// makes the SMALLEST key the highest logit with ties broken toward the
// lower index — sample::candidate_before, as an integer order. The
// streaming selection keeps the running best select_k keys in shared
// memory (split hi/lo 32-bit arrays: one bank per thread at every bitonic
// stride) and returns them sorted ascending, i.e. in canonical order.
// Deterministic on any correct implementation: the result is a set under a
// total order, so no reduction shape can change it.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

constexpr int kSelectTile = 2048;  // keys per selection tile (>= 2*select_k)
// The widest selection the expansion serves (eight rounds of a 256-thread
// block): the full GLM-5.3's per-token select_k 2048 is the bound,
// GLM-5.3-Flash's 512 a quarter of it.
constexpr int kSelectMaxK = 2048;

__device__ inline uint32_t sortable_f32_dev(float f) {
  uint32_t u = __float_as_uint(f);
  return (u >> 31) ? ~u : (u | 0x80000000u);
}

// Bitonic networks over 64-bit composite keys stored as SPLIT 32-bit arrays
// (hi/lo). A u64 key spans two 4-byte banks, so a u64 array with XOR indexing
// bank-conflicts ~16 ways in the sort stages (measured: the merge phase was
// 97% of the select kernel). Split arrays give every thread its own bank for
// every stride: stride >= 32 leaves i%32 untouched; smaller strides XOR-
// permute the banks. Same keys, same comparisons, identical selection.
__device__ inline bool key_less(uint32_t a_hi, uint32_t a_lo, uint32_t b_hi,
                                uint32_t b_lo) {
  return (a_hi < b_hi) || (a_hi == b_hi && a_lo < b_lo);
}

__device__ inline void cx_split(uint32_t* hi, uint32_t* lo, int i, int j,
                                bool asc) {
  const uint32_t a_hi = hi[i], a_lo = lo[i];
  const uint32_t b_hi = hi[j], b_lo = lo[j];
  const bool swap = asc ? key_less(b_hi, b_lo, a_hi, a_lo)
                        : key_less(a_hi, a_lo, b_hi, b_lo);
  if (swap) {
    hi[i] = b_hi; lo[i] = b_lo;
    hi[j] = a_hi; lo[j] = a_lo;
  }
}

// Ascending bitonic sort of n keys (power of two).
__device__ inline void bitonic_sort_asc(uint32_t* hi, uint32_t* lo, int n) {
  for (int size = 2; size <= n; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int j = i ^ stride;
        if (j > i) cx_split(hi, lo, i, j, (i & size) == 0);
      }
      __syncthreads();
    }
  }
}

// Ascending bitonic merge of a bitonic sequence of n (power of two).
__device__ inline void bitonic_merge_asc(uint32_t* hi, uint32_t* lo, int n) {
  for (int stride = n >> 1; stride > 0; stride >>= 1) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      const int j = i ^ stride;
      if (j > i) cx_split(hi, lo, i, j, true);
    }
    __syncthreads();
  }
}

// Streaming top-select_k over key_fn(pool) -> composite key, restricted to
// pools [lo, hi). best_hi/best_lo (smem, [select_k]) must be pre-initialized
// to kKeyMax; on return they hold the smallest select_k keys seen, sorted
// ascending. tile arrays (smem) need TILE entries each; TILE >= 2*select_k
// required (kSelectTile serves select_k <= 1024; a select_k of 2048 takes
// a 4096-key tile — the same network, twice the smem).
template <int TILE = kSelectTile, typename KeyFn>
__device__ inline void select_topk_stream(KeyFn key_fn, int64_t lo, int64_t hi,
                                          uint32_t* best_hi, uint32_t* best_lo,
                                          uint32_t* tile_hi, uint32_t* tile_lo,
                                          int select_k) {
  const int nthreads = blockDim.x;
  const int warp = threadIdx.x >> 5;
  const int nwarp = nthreads >> 5;
  for (int64_t base = lo; base < hi; base += TILE) {
    const int n = int(min((int64_t)TILE, hi - base));
    for (int p = threadIdx.x; p < TILE; p += nthreads) {
      tile_hi[p] = 0xFFFFFFFFu;  // kKeyMax padding; [0, n) overwritten below
      tile_lo[p] = 0xFFFFFFFFu;
    }
    __syncthreads();
    if constexpr (KeyFn::kWarpCooperative) {
      // Warp-cooperative keys (pool logits): one warp evaluates one pool;
      // lane 0 publishes. All lanes load the same cache row — broadcast.
      for (int p = warp; p < n; p += nwarp) {
        const uint64_t key = key_fn(base + p);
        if ((threadIdx.x & 31) == 0) {
          tile_hi[p] = uint32_t(key >> 32);
          tile_lo[p] = uint32_t(key);
        }
      }
    } else {
      // Per-thread keys (merge partials): thread-strided with unrolled
      // independent loads — the warp-strided form serialized 512 dependent
      // global loads per warp and was 57% of the select kernel.
      for (int p = threadIdx.x; p < n; p += nthreads) {
        const uint64_t key = key_fn(base + p);
        tile_hi[p] = uint32_t(key >> 32);
        tile_lo[p] = uint32_t(key);
      }
    }
    __syncthreads();
    // Sort the occupied power of two of the tile, never fewer than
    // select_k entries (the merge below reads the first select_k; the
    // kKeyMax padding beyond n sorts to the end either way) — a short
    // context's 75 keys were sorting all 2 048 padded slots, 66 barrier
    // passes for a 49 us decode select. The selected set is
    // the same: a total order on composite keys, one result.
    int sort_n = 1;
    while (sort_n < select_k || sort_n < n) sort_n <<= 1;
    bitonic_sort_asc(tile_hi, tile_lo, sort_n);
    // Merge best (asc) with the tile's smallest select_k reversed (desc):
    // [asc][desc] is bitonic, so a 2*select_k bitonic merge yields the new
    // best. Elements beyond the tile's first select_k are dominated by
    // select_k better tile keys and can never enter the global top-k.
    for (int i = threadIdx.x; i < select_k; i += nthreads) {
      tile_hi[select_k + i] = tile_hi[select_k - 1 - i];
      tile_lo[select_k + i] = tile_lo[select_k - 1 - i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < select_k; i += nthreads) {
      tile_hi[i] = best_hi[i];
      tile_lo[i] = best_lo[i];
    }
    __syncthreads();
    bitonic_merge_asc(tile_hi, tile_lo, 2 * select_k);
    for (int i = threadIdx.x; i < select_k; i += nthreads) {
      best_hi[i] = tile_hi[i];
      best_lo[i] = tile_lo[i];
    }
    __syncthreads();
  }
}


// ---- shared by the DSA and QSA selections --------------------
constexpr int kIdxBits = 21;  // pool ids < 2^21 (validated at launch)
constexpr uint64_t kIdxMask = (1ull << kIdxBits) - 1;

__device__ inline float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(~0u, v, off);
  return v;
}

// Extract pool ids from sorted best keys, sort ascending, expand to token
// positions, append the query's incomplete tail. Writes out_row[0,
// max_selected) (-1 padded); returns the token count. scratch: smem int32
// [select_k]; smem_count: smem int32 [1].
// Ascending order of the n unique ids in scratch[0, n) by RANK: each id's
// rank — the count of smaller ids, every thread reading the list as
// broadcast smem loads — is a permutation, so the scatter is the sort.
// PER ids per thread (n <= PER * blockDim); n^2 / blockDim compares per
// thread and two barriers, against the bitonic network's 45 barrier
// phases that were most of the select kernel's floor at short contexts.
template <int PER>
__device__ __forceinline__ void rank_sort_ids(int32_t* scratch, int n) {
  const int nthreads = blockDim.x;
  int32_t v[PER];
  int rk[PER];
#pragma unroll
  for (int t = 0; t < PER; ++t) {
    const int i = threadIdx.x + t * nthreads;
    v[t] = i < n ? scratch[i] : 0;
    rk[t] = 0;
  }
#pragma unroll 8
  for (int j = 0; j < n; ++j) {
    const int32_t s = scratch[j];
#pragma unroll
    for (int t = 0; t < PER; ++t) rk[t] += (s < v[t]);
  }
  __syncthreads();
#pragma unroll
  for (int t = 0; t < PER; ++t) {
    const int i = threadIdx.x + t * nthreads;
    if (i < n) scratch[rk[t]] = v[t];
  }
  __syncthreads();
}

// The ascending pool ids scratch[0, n_sel) as token positions: each pool's
// kpool tokens, then the query's incomplete tail. Writes out_row[0,
// max_selected) (-1 padded); returns the token count. A selection that
// already holds its ids in order (the decode select's ordered gather) calls
// this directly; expand_from_best sorts first.
__device__ inline int expand_sorted_ids(const int32_t* scratch, int n_sel, int64_t pos, int kpool,
                                        int max_selected, int32_t* out_row) {
  const int nthreads = blockDim.x;
  const int64_t seq_len = pos + 1;
  const int64_t tail_start = (seq_len / kpool) * kpool;
  const int tail_cnt = int(seq_len - tail_start);
  const int hist = n_sel * kpool;
  for (int col = threadIdx.x; col < max_selected; col += nthreads) {
    int32_t tok;
    if (col < hist)
      tok = scratch[col / kpool] * kpool + (col % kpool);
    else if (col - hist < tail_cnt)
      tok = int32_t(tail_start + (col - hist));
    else
      tok = -1;
    out_row[col] = tok;
  }
  return hist + tail_cnt;
}

__device__ inline int expand_from_best(const uint32_t* best_hi,
                                       const uint32_t* best_lo, int select_k,
                                       int64_t pos, int kpool,
                                       int max_selected, int32_t* out_row,
                                       int32_t* scratch, int* smem_count) {
  const int nthreads = blockDim.x;
  const int nwarp = nthreads >> 5;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // Compaction of the real entries by warp ballot (2026-09-06: one smem
  // atomic per entry serialized 512 deep and was a third of the
  // expansion): per round, each warp's count and each lane's offset within
  // the warp come from one ballot; one thread scans the round x warp
  // counts.
  constexpr int kRounds = 8;  // select_k <= 8 * blockDim (checked at launch)
  __shared__ int warp_cnt[kRounds * 8];
  __shared__ int warp_off[kRounds * 8];
  int32_t id[kRounds];
  int within[kRounds];
  bool real[kRounds];
#pragma unroll
  for (int rd = 0; rd < kRounds; ++rd) {
    const int i = rd * nthreads + threadIdx.x;
    real[rd] = i < select_k &&
               (best_hi[i] != 0xFFFFFFFFu || best_lo[i] != 0xFFFFFFFFu);
    id[rd] = real[rd] ? int32_t((uint64_t(best_hi[i]) << 32 | best_lo[i]) & kIdxMask) : 0;
    // A pool past the row's visible pools is no selection.
    if (real[rd] && int64_t(id[rd]) * kpool > pos) real[rd] = false;
    const uint32_t mask = __ballot_sync(0xffffffffu, real[rd]);
    within[rd] = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) warp_cnt[rd * nwarp + warp] = __popc(mask);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int run = 0;
    for (int k = 0; k < kRounds * nwarp; ++k) {
      warp_off[k] = run;
      run += warp_cnt[k];
    }
    *smem_count = run;
  }
  __syncthreads();
#pragma unroll
  for (int rd = 0; rd < kRounds; ++rd)
    if (real[rd]) scratch[warp_off[rd * nwarp + warp] + within[rd]] = id[rd];
  __syncthreads();
  const int n_sel = *smem_count;
  if (n_sel <= nthreads) rank_sort_ids<1>(scratch, n_sel);
  else if (n_sel <= 2 * nthreads) rank_sort_ids<2>(scratch, n_sel);
  else if (n_sel <= 4 * nthreads) rank_sort_ids<4>(scratch, n_sel);
  else rank_sort_ids<8>(scratch, n_sel);
  return expand_sorted_ids(scratch, n_sel, pos, kpool, max_selected, out_row);
}

}  // namespace dgpp
