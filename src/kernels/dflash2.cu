#include "kernels/dflash2.hpp"
#include "kernels/pdl.cuh"

#include <cmath>
#include <stdexcept>

#include <algorithm>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/pick.hpp"

namespace dgpp {
namespace {

// ---- shared block helpers ----------------------------------------------------

__device__ __forceinline__ float block_sum(float v, float* shared) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  v += __shfl_down_sync(0xffffffffu, v, 16);
  v += __shfl_down_sync(0xffffffffu, v, 8);
  v += __shfl_down_sync(0xffffffffu, v, 4);
  v += __shfl_down_sync(0xffffffffu, v, 2);
  v += __shfl_down_sync(0xffffffffu, v, 1);
  if (lane == 0) shared[wid] = v;
  __syncthreads();
  const int nw = (blockDim.x + 31) >> 5;
  v = (threadIdx.x < nw) ? shared[threadIdx.x] : 0.0f;
  if (wid == 0) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v, 8);
    v += __shfl_down_sync(0xffffffffu, v, 4);
    v += __shfl_down_sync(0xffffffffu, v, 2);
    v += __shfl_down_sync(0xffffffffu, v, 1);
  }
  return v;
}

__device__ __forceinline__ float warp_sum(float v) {
  // The xor butterfly: EVERY lane leaves with the full sum (the shfl_down
  // variant only completes on lane 0, and the rope/attention users here
  // consume the result block-wide in registers).
#pragma unroll
  for (int off = 16; off; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// ---- the dynamic grouped convolution -----------------------------------------

// Taps fixed at 2 (the released checkpoints' conv_kernel_size; the loader
// refuses others). position = r % block_rows gates tap 1: the conv lives
// INSIDE a request's query block and never crosses block borders.
__global__ void grouped_conv_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ delta,
                                    const uint16_t* __restrict__ base, uint16_t* __restrict__ out,
                                    int block_rows, int hidden, int group_size,
                                    int64_t delta_row_stride) {
  dgpp_pdl_wait();
  const int64_t r = blockIdx.x;
  const uint16_t* xr = x + r * hidden;
  const bool has_prev = (r % block_rows) >= 1;
  const uint16_t* xp = x + (r - 1) * hidden;  // in-bounds wherever has_prev (row 0 of a block never reads it)
  const uint16_t* dr = delta + r * delta_row_stride;
  const int groups = hidden / group_size;  // one delta per (tap, group): tap 1 starts at dr[groups]
  for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
    const int g = c / group_size;
    float acc = (bf16_bits_to_float(base[c]) + bf16_bits_to_float(dr[g])) * bf16_bits_to_float(xr[c]);
    if (has_prev)
      acc += (bf16_bits_to_float(base[hidden + c]) + bf16_bits_to_float(dr[groups + g])) *
             bf16_bits_to_float(xp[c]);
    out[r * hidden + c] = float_to_bf16_bits(acc);
  }
}

// ---- the feature accumulator --------------------------------------------------

__global__ void acc_kernel(float* __restrict__ acc, const float* __restrict__ src, int64_t n,
                           int beta) {
  dgpp_pdl_wait();
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  acc[i] = beta ? acc[i] + src[i] : src[i];
}

// ---- the standard norms -------------------------------------------------------

// y = bf16(f32(x) * rsqrt(mean(x^2)/dim + eps) * w). Add: resid is folded
// in first (and rewritten with the plain bf16 sum) — bitwise the pair.
template <bool Add, typename InT>
__global__ void rmsnorm_kernel(const InT* __restrict__ x, const uint16_t* __restrict__ w,
                               uint16_t* __restrict__ y, uint16_t* __restrict__ resid, int dim,
                               float eps) {
  dgpp_pdl_wait();
  __shared__ float red[8];
  const int64_t row = blockIdx.x;
  const InT* xr = x + row * dim;
  float ssq = 0.0f;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    float v = sizeof(InT) == 4 ? xr[d] : bf16_bits_to_float(xr[d]);
    if (Add) v += bf16_bits_to_float(resid[row * dim + d]);
    ssq += v * v;
  }
  const float total = block_sum(ssq, red);
  if (threadIdx.x == 0) red[0] = rsqrtf(total / static_cast<float>(dim) + eps);
  __syncthreads();
  const float rstd = red[0];
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    float v = sizeof(InT) == 4 ? xr[d] : bf16_bits_to_float(xr[d]);
    if (Add) {
      v += bf16_bits_to_float(resid[row * dim + d]);
      resid[row * dim + d] = float_to_bf16_bits(v);
    }
    y[row * dim + d] = float_to_bf16_bits(v * rstd * bf16_bits_to_float(w[d]));
  }
}

// ---- norm + rope (rotate_half) --------------------------------------------------

// One warp per (row, head), four elements per lane. The rope pairs (d,
// d+half): lanes 0..31 cover elements d, d+32, d+64, d+96 (j = 0..3);
// element d < half pairs with d+half, i.e. register j pairs with j+2.
__global__ void norm_rope_kernel(const uint16_t* __restrict__ x, int64_t x_stride,
                                 const uint16_t* __restrict__ w, const int64_t* __restrict__ pos,
                                 const float* __restrict__ inv_freq, uint16_t* __restrict__ out,
                                 int64_t out_stride, int dim, int half, float eps) {
  dgpp_pdl_wait();
  const int64_t row = blockIdx.x;
  const int head = blockIdx.y;
  const int64_t p = pos[row];
  if (p < 0) return;
  const uint16_t* xr = x + row * x_stride + static_cast<int64_t>(head) * dim;
  uint16_t* orow = out + row * out_stride + static_cast<int64_t>(head) * dim;
  const int lane = threadIdx.x;  // blockDim.x == 32
  float e[4];
  float ssq = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    e[j] = bf16_bits_to_float(xr[lane + 32 * j]);
    ssq += e[j] * e[j];
  }
  const float rstd = rsqrtf(warp_sum(ssq) / static_cast<float>(dim) + eps);
#pragma unroll
  for (int j = 0; j < 4; ++j)
    e[j] = bf16_bits_to_float(
        float_to_bf16_bits(e[j] * rstd * bf16_bits_to_float(w[lane + 32 * j])));
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int d = lane + 32 * j;  // d < half
    const float angle = static_cast<float>(p) * inv_freq[d];
    const float c = bf16_bits_to_float(float_to_bf16_bits(std::cos(angle)));
    const float s = bf16_bits_to_float(float_to_bf16_bits(std::sin(angle)));
    const float x1 = -e[j + 2];  // rotate_half: -x[d + half]
    orow[d] = float_to_bf16_bits(bf16_bits_to_float(float_to_bf16_bits(e[j] * c)) +
                                 bf16_bits_to_float(float_to_bf16_bits(x1 * s)));
    orow[d + half] =
        float_to_bf16_bits(bf16_bits_to_float(float_to_bf16_bits(e[j + 2] * c)) +
                           bf16_bits_to_float(float_to_bf16_bits(e[j] * s)));
  }
}

// ---- the block attention --------------------------------------------------------

__device__ __forceinline__ int64_t plane_slot(const int32_t* block_table, int64_t p,
                                              int block_tokens, int blocks_per_request) {
  int64_t b = p / block_tokens;
  if (b >= blocks_per_request) b = blocks_per_request - 1;
  return static_cast<int64_t>(block_table[b]) * block_tokens + (p % block_tokens);
}

// One block per (query row, kv head); each warp iteration owns one query
// head of the group. fp32 online softmax, bf16 probabilities into V (the
// house rule), the denominator unrounded. dim 128 (four elements per lane).
// The key walk covers the causal context window [max(0, pos - window + 1),
// ctx_end], then the bidirectional block [blk_lo, blk_hi] (window-checked
// the same way; blocks sit within a few tokens of the queries).
__global__ void block_attn_kernel(const uint16_t* __restrict__ q, int64_t q_stride,
                                  const uint16_t* __restrict__ k_cache,
                                  const uint16_t* __restrict__ v_cache,
                                  const int32_t* __restrict__ block_table, int block_tokens,
                                  int blocks_per_request, int block_rows, int64_t window,
                                  const int64_t* __restrict__ pos, float scale,
                                  uint16_t* __restrict__ out, int heads, int kv_heads, int dim) {
  dgpp_pdl_wait();
  const int row = blockIdx.x;
  const int kvh = blockIdx.y;
  const int nwarp = blockDim.x >> 5;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int hpq = heads / kv_heads;
  const int64_t p = pos[row];
  // The block's span off its first row's position (device-read: a
  // recorded draft takes the committed position).
  const int64_t p0 = pos[row - row % block_rows];
  const int64_t ctx_end = p0 - 1, blk_lo = p0;
  // A block at the context ceiling has padding positions. Only valid
  // rows own K/V; reading the nominal end could touch an unowned page.
  int last_valid = block_rows - 1;
  while (last_valid > 0 && pos[row - row % block_rows + last_valid] < 0) --last_valid;
  const int64_t blk_hi = p0 + last_valid;
  const int64_t lo = p - window + 1;  // keys below lo are windowed out
  for (int hh = warp; hh < hpq; hh += nwarp) {
    const int head = kvh * hpq + hh;
    const int d0 = lane * 4;  // dim is 128: four elements per lane
    if (p < 0 || p0 < 0) {  // a row past the context: no state, zero output
#pragma unroll
      for (int j = 0; j < 4; ++j) out[(row * heads + head) * dim + d0 + j] = 0;
      continue;
    }
    float qv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      qv[j] = bf16_bits_to_float(
                  q[row * q_stride + static_cast<int64_t>(head) * dim + d0 + j]) *
              scale;
    float m = -INFINITY, l = 0.0f, acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j] = 0.0f;
    const int64_t ctx_lo = lo > 0 ? lo : 0;
    for (int span = 0; span < 2; ++span) {
      const int64_t s0 = span == 0 ? ctx_lo : blk_lo;
      const int64_t s1 = span == 0 ? ctx_end : blk_hi;
      for (int64_t kp = s0; kp <= s1; ++kp) {
        const int64_t slot = plane_slot(block_table, kp, block_tokens, blocks_per_request);
        const int64_t kbase = (slot * kv_heads + kvh) * dim;
        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j)
          dot += qv[j] * bf16_bits_to_float(k_cache[kbase + d0 + j]);
        dot = warp_sum(dot);
        const float m_new = dot > m ? dot : m;
        const float correction = m == -INFINITY ? 0.0f : std::exp(m - m_new);
        const float pf = m_new == -INFINITY ? 0.0f : std::exp(dot - m_new);
        const float pb = bf16_bits_to_float(float_to_bf16_bits(pf));
        l = l * correction + pf;
#pragma unroll
        for (int j = 0; j < 4; ++j)
          acc[j] = acc[j] * correction + pb * bf16_bits_to_float(v_cache[kbase + d0 + j]);
        m = m_new;
      }
    }
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      out[(row * heads + head) * dim + d0 + j] = float_to_bf16_bits(acc[j] * inv);
  }
}

// The split-key form: grid (rows, kv_heads, kSplits), one warp per query
// head of the group. Split s walks the s-th of kSplits contiguous ranges of
// the row's context window [ctx_lo, ctx_end]; the last split also walks the
// block span. Each leaves (m, l, acc[dim]) unnormalized in partials
// [(row * heads + head) * kSplits + s][dim + 2]; the combine pass merges.
constexpr int kBlockAttnSplits = 32;

__global__ void block_attn_split_kernel(const uint16_t* __restrict__ q, int64_t q_stride,
                                        const uint16_t* __restrict__ k_cache,
                                        const uint16_t* __restrict__ v_cache,
                                        const int32_t* __restrict__ block_table, int block_tokens,
                                        int blocks_per_request, int block_rows, int64_t window,
                                        const int64_t* __restrict__ pos, float scale,
                                        float* __restrict__ partials, int heads, int kv_heads, int dim) {
  dgpp_pdl_wait();
  const int row = blockIdx.x;
  const int kvh = blockIdx.y;
  const int split = blockIdx.z;
  const int nwarp = blockDim.x >> 5;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int hpq = heads / kv_heads;
  const int64_t p = pos[row];
  const int64_t p0 = pos[row - row % block_rows];
  const int64_t ctx_end = p0 - 1, blk_lo = p0;
  // A block at the context ceiling has padding positions. Only valid
  // rows own K/V; reading the nominal end could touch an unowned page.
  int last_valid = block_rows - 1;
  while (last_valid > 0 && pos[row - row % block_rows + last_valid] < 0) --last_valid;
  const int64_t blk_hi = p0 + last_valid;
  const int64_t lo = p - window + 1;
  const int64_t ctx_lo = lo > 0 ? lo : 0;
  // This split's share of the context keys (empty for a short context).
  const int64_t ctx_n = ctx_end >= ctx_lo ? ctx_end - ctx_lo + 1 : 0;
  const int64_t per = (ctx_n + kBlockAttnSplits - 1) / kBlockAttnSplits;
  const int64_t s0 = ctx_lo + split * per;
  const int64_t s1 = (s0 + per - 1 < ctx_end) ? s0 + per - 1 : ctx_end;
  const bool last = split == kBlockAttnSplits - 1;
  for (int hh = warp; hh < hpq; hh += nwarp) {
    const int head = kvh * hpq + hh;
    const int d0 = lane * 4;
    float* part = partials + (static_cast<size_t>(row * heads + head) * kBlockAttnSplits + split) * (dim + 2);
    if (p < 0 || p0 < 0) {
#pragma unroll
      for (int j = 0; j < 4; ++j) part[d0 + j] = 0.0f;
      if (lane == 0) {
        part[dim] = -INFINITY;
        part[dim + 1] = 0.0f;
      }
      continue;
    }
    float qv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      qv[j] = bf16_bits_to_float(q[row * q_stride + static_cast<int64_t>(head) * dim + d0 + j]) * scale;
    float m = -INFINITY, l = 0.0f, acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j] = 0.0f;
    for (int span = 0; span < 2; ++span) {
      if (span == 1 && !last) break;
      const int64_t a = span == 0 ? s0 : blk_lo;
      const int64_t b = span == 0 ? s1 : blk_hi;
      for (int64_t kp = a; kp <= b; ++kp) {
        const int64_t slot = plane_slot(block_table, kp, block_tokens, blocks_per_request);
        const int64_t kbase = (slot * kv_heads + kvh) * dim;
        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) dot += qv[j] * bf16_bits_to_float(k_cache[kbase + d0 + j]);
        dot = warp_sum(dot);
        const float m_new = dot > m ? dot : m;
        const float correction = m == -INFINITY ? 0.0f : std::exp(m - m_new);
        const float pf = m_new == -INFINITY ? 0.0f : std::exp(dot - m_new);
        const float pb = bf16_bits_to_float(float_to_bf16_bits(pf));
        l = l * correction + pf;
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] = acc[j] * correction + pb * bf16_bits_to_float(v_cache[kbase + d0 + j]);
        m = m_new;
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) part[d0 + j] = acc[j];
    if (lane == 0) {
      part[dim] = m;
      part[dim + 1] = l;
    }
  }
}

// One warp per (row, head): the splits' partials merged at the global max
// (fixed order over the splits), the output rounded once to bf16. Lane s
// holds split s's m and l (kBlockAttnSplits = the warp's lanes): the maximum
// by shuffle, each split's weight formed once by its lane and broadcast in
// split order (2026-10-20: every lane read every split's m for the maximum
// and its m and l again in the merge, and computed the weight itself — the
// same arithmetic in the same order, so the outputs are bitwise).
static_assert(kBlockAttnSplits == 32, "block_attn_combine_kernel: one lane per split");
__global__ void block_attn_combine_kernel(const float* __restrict__ partials, int heads, int dim,
                                          uint16_t* __restrict__ out) {
  dgpp_pdl_wait();
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int lane = threadIdx.x;
  const float* base = partials + static_cast<size_t>(row * heads + head) * kBlockAttnSplits * (dim + 2);
  const float m_lane = base[lane * (dim + 2) + dim];
  const float l_lane = base[lane * (dim + 2) + dim + 1];
  float M = m_lane;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
  const float w_lane = (M != -INFINITY && m_lane != -INFINITY) ? std::exp(m_lane - M) : 0.0f;
  const int d0 = lane * 4;
  float l = 0.0f, acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (M != -INFINITY) {
    for (int s = 0; s < kBlockAttnSplits; ++s) {
      const float ms = __shfl_sync(0xffffffffu, m_lane, s);
      if (ms == -INFINITY) continue;
      const float w = __shfl_sync(0xffffffffu, w_lane, s);
      const float ls = __shfl_sync(0xffffffffu, l_lane, s);
      const float* part = base + s * (dim + 2);
      l += ls * w;
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[j] += part[d0 + j] * w;
    }
  }
  const float inv = l > 0.0f ? 1.0f / l : 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) out[(row * heads + head) * dim + d0 + j] = float_to_bf16_bits(acc[j] * inv);
}

// ---- the candidate top-K --------------------------------------------------------

struct Cand {
  float score;
  int32_t id;
};
__device__ __forceinline__ bool better_cand(const Cand& a, const Cand& b) {
  return a.score > b.score || (a.score == b.score && a.id < b.id);
}

constexpr int kTopkThreads = 128;
constexpr int kTopkMaxChunks = 32;
constexpr int64_t kTopkChunkSpan = 8192;  // vocabulary entries a block (64 a thread)

// Fold `count` descending K-lists (a power of two, kTopkThreads at most;
// list i's slot j at X[j * kTopkThreads + i]) into list 0, halving the
// count a round: thread t merges lists 2t and 2t+1 into list t of the other
// buffer (two-pointer, K steps, ties to the lower id — ids are distinct, so
// the order is total and the result is the first K of the union's sort
// however the lists split the row). Every index is a shared-memory one:
// no register array is indexed at run time (such arrays spill to local
// memory, which is what made the first chunked form 0.53 ms). Returns the
// buffer that holds the result.
template <int K>
__device__ __forceinline__ Cand* block_fold_lists(Cand* X, Cand* Y, int count) {
  constexpr int S = kTopkThreads;
  for (int n = count >> 1; n >= 1; n >>= 1) {
    const int t = threadIdx.x;
    if (t < n) {
      int i = 0, j = 0;
      for (int o = 0; o < K; ++o) {
        const Cand a = X[i * S + 2 * t], b = X[j * S + 2 * t + 1];
        if (better_cand(a, b)) {
          Y[o * S + t] = a;
          ++i;
        } else {
          Y[o * S + t] = b;
          ++j;
        }
      }
    }
    __syncthreads();
    Cand* tmp = X;
    X = Y;
    Y = tmp;
  }
  return X;
}

// Pass 1: block (chunk, row) scans its span of the row — each thread keeps
// a descending K-list in shared memory (one register tail compare rejects
// almost everything once warm) — then folds the block's lists; the chunk's
// K go to the partials, or straight to the output when the row is one
// chunk. The one-block form read a 248K row from one SM and thread 0
// walked 4096 candidates: 0.63 ms a step for 7 MB.
template <int K>
__global__ void __launch_bounds__(kTopkThreads)
    topk_chunk_kernel(const float* __restrict__ logits, int64_t vocab, int chunks, Cand* __restrict__ partial,
                      int32_t* __restrict__ ids, float* __restrict__ scores) {
  dgpp_pdl_wait();
  constexpr int S = kTopkThreads;
  __shared__ Cand sl[K * S];
  __shared__ Cand ml[K * S];
  const int row = blockIdx.y, chunk = blockIdx.x, t = threadIdx.x;
  const float* lg = logits + static_cast<int64_t>(row) * vocab;
  const int64_t span = (vocab + chunks - 1) / chunks;
  const int64_t v0 = static_cast<int64_t>(chunk) * span;
  const int64_t v1 = min(vocab, v0 + span);
  for (int j = 0; j < K; ++j) sl[j * S + t] = Cand{-INFINITY, 0x7fffffff};
  Cand tail = sl[(K - 1) * S + t];
  for (int64_t v = v0 + t; v < v1; v += S) {
    const Cand c{lg[v], static_cast<int32_t>(v)};
    if (!better_cand(c, tail)) continue;
    int j = K - 1;
    while (j > 0 && better_cand(c, sl[(j - 1) * S + t])) {
      sl[j * S + t] = sl[(j - 1) * S + t];
      --j;
    }
    sl[j * S + t] = c;
    tail = sl[(K - 1) * S + t];
  }
  __syncthreads();
  const Cand* res = block_fold_lists<K>(sl, ml, S);
  if (t < K) {
    const Cand c = res[t * S];
    if (chunks == 1) {
      ids[static_cast<int64_t>(row) * K + t] = c.id;
      scores[static_cast<int64_t>(row) * K + t] = c.score;
    } else {
      partial[(static_cast<int64_t>(row) * chunks + chunk) * K + t] = c;
    }
  }
}

// Pass 2: one block a row folds the chunks' lists (sentinel lists past
// `chunks`, kTopkMaxChunks in all) and writes the row's K.
template <int K>
__global__ void __launch_bounds__(kTopkThreads)
    topk_merge_kernel(const Cand* __restrict__ partial, int chunks, int32_t* __restrict__ ids,
                      float* __restrict__ scores) {
  dgpp_pdl_wait();
  constexpr int S = kTopkThreads;
  __shared__ Cand sl[K * S];
  __shared__ Cand ml[K * S];
  const int row = blockIdx.x, t = threadIdx.x;
  if (t < kTopkMaxChunks) {
    const bool have = t < chunks;
    for (int j = 0; j < K; ++j)
      sl[j * S + t] = have ? partial[(static_cast<int64_t>(row) * chunks + t) * K + j] : Cand{-INFINITY, 0x7fffffff};
  }
  __syncthreads();
  const Cand* res = block_fold_lists<K>(sl, ml, kTopkMaxChunks);
  if (t < K) {
    const Cand c = res[t * S];
    ids[static_cast<int64_t>(row) * K + t] = c.id;
    scores[static_cast<int64_t>(row) * K + t] = c.score;
  }
}

int topk_chunks(int64_t vocab) {
  const int64_t c = (vocab + kTopkChunkSpan - 1) / kTopkChunkSpan;
  return static_cast<int>(c < 1 ? 1 : (c > kTopkMaxChunks ? kTopkMaxChunks : c));
}

// ---- the selector -----------------------------------------------------------------

// The scores[l][p][c] = unary[l][c] + <pred_code[id(l-1,p)] * hidden[l],
// succ_code[id(l,c)]> table and the greedy slot walk (vLLM
// qwen3_dflash2._score_edges + _selector_walk_kernel at temperature 0).
// A proposal row (kernels/sample_pick.hpp DraftProposal): the k candidates
// and their masses under the draft temperature, the draw as `token`.
__device__ __forceinline__ void write_proposal(DraftProposal* pr, const int32_t* ids, const float* exps,
                                               float den, int k, int32_t token) {
  pr->n = k;
  pr->token = token;
  for (int c = 0; c < k; ++c) {
    pr->ids[c] = ids[c];
    pr->mass[c] = __fdiv_rn(exps[c], den);
  }
}

__global__ void selector_kernel(const int32_t* __restrict__ ids, const float* __restrict__ unary,
                                const float* __restrict__ hidden, const uint16_t* __restrict__ pred_cb,
                                const uint16_t* __restrict__ succ_cb,
                                const int64_t* __restrict__ anchor_tok, int32_t* __restrict__ tokens,
                                int steps, int k, int rank, const int64_t* __restrict__ pos,
                                const SampleSpec* __restrict__ spec, DraftProposal* __restrict__ proposal,
                                DraftProposal* __restrict__ proposal_host, float* __restrict__ conf) {
  dgpp_pdl_wait();
  extern __shared__ float sm[];
  const int32_t anchor = static_cast<int32_t>(*anchor_tok);
  float* h = sm;                  // [rank]
  float* pred = h + rank;         // [k][rank]
  float* succ = pred + k * rank;  // [k][rank]
  float* sc = succ + k * rank;    // [k][k]
  float* ex = sc + k * k;         // [k] the sampled step's exps
  const int tid = threadIdx.x;
  // The sampled walk: a stochastic request (temperature > 0) draws each
  // step from the softmax of its scores at the draft temperature and
  // proposes the set it drew from; otherwise the argmax, no proposal.
  const bool sampled = spec != nullptr && spec->temperature > 0.0f;
  const float T = sampled ? (spec->draft_temperature > 0.0f ? spec->draft_temperature : spec->temperature) : 1.0f;
  int prev = 0;
  for (int l = 0; l < steps; ++l) {
    __syncthreads();
    for (int r = tid; r < rank; r += blockDim.x) h[r] = hidden[l * rank + r];  // per-step row
    for (int i = tid; i < k * rank; i += blockDim.x) {
      const int p = i / rank, r = i % rank;
      const int32_t pid = l == 0 ? anchor : ids[static_cast<int64_t>(l - 1) * k + p];
      pred[i] = bf16_bits_to_float(pred_cb[static_cast<int64_t>(pid) * rank + r]);
      const int32_t sid = ids[static_cast<int64_t>(l) * k + p];
      succ[i] = bf16_bits_to_float(succ_cb[static_cast<int64_t>(sid) * rank + r]);
    }
    __syncthreads();
    for (int e = tid; e < k * k; e += blockDim.x) {
      const int p = e / k, c = e % k;
      float dot = 0.0f;
      for (int r = 0; r < rank; ++r) dot += pred[p * rank + r] * h[r] * succ[c * rank + r];
      sc[e] = unary[static_cast<int64_t>(l) * k + c] + dot;  // the candidate's logit (vLLM _score_edges)
    }
    __syncthreads();
    if (tid == 0) {
      const int32_t* lid = ids + static_cast<int64_t>(l) * k;
      int best = 0;
      for (int c = 1; c < k; ++c)
        if (sc[prev * k + c] > sc[prev * k + best]) best = c;
      float den = 0.0f;  // the sampled step's softmax denominator (the confidence reads it too)
      if (sampled) {
        // softmax(scores / T) over the step's k candidates (fp32, the max
        // subtracted), the inverse-CDF walk in fp64 on the draft stream's
        // uniform keyed by the draft's position (kSampleDraftSeedMix ^ the
        // step, as the chained MTP draft keys its draws).
        const float mx = __fdiv_rn(sc[prev * k + best], T);
        for (int c = 0; c < k; ++c) {
          ex[c] = expf(__fsub_rn(__fdiv_rn(sc[prev * k + c], T), mx));
          den = __fadd_rn(den, ex[c]);
        }
        // The request's truncation on the proposal too (the MTP draft's
        // convention: the draft's final set after top-k, top-p and min-p):
        // a draw from the tail the target's final set excludes could only
        // be rejected. Candidates in mass order (ties to the lower id); the
        // survivors renormalized, the rest at mass 0.
        if (spec->top_k > 0 || spec->top_p < 1.0f || spec->min_p > 0.0f) {
          int order[64];
          for (int c = 0; c < k; ++c) order[c] = c;
          for (int i = 1; i < k; ++i) {  // insertion sort by (mass desc, id asc)
            const int oi = order[i];
            int j = i;
            while (j > 0 && (ex[order[j - 1]] < ex[oi] ||
                             (ex[order[j - 1]] == ex[oi] && lid[order[j - 1]] > lid[oi]))) {
              order[j] = order[j - 1];
              --j;
            }
            order[j] = oi;
          }
          int keep = k;
          if (spec->top_k > 0 && spec->top_k < keep) keep = spec->top_k;
          if (spec->min_p > 0.0f) {
            const float floor_mass = spec->min_p * ex[order[0]];
            int n = 0;
            while (n < keep && ex[order[n]] >= floor_mass) ++n;
            keep = n > 0 ? n : 1;
          }
          if (spec->top_p < 1.0f) {
            float cum = 0.0f;
            int n = 0;
            while (n < keep) {
              cum = __fadd_rn(cum, __fdiv_rn(ex[order[n]], den));
              ++n;
              if (cum >= spec->top_p) break;
            }
            keep = n;
          }
          float kept = 0.0f;
          for (int i = 0; i < k; ++i) {
            if (i >= keep) ex[order[i]] = 0.0f;
            else kept = __fadd_rn(kept, ex[order[i]]);
          }
          den = kept;
        }
        const uint64_t seed = spec->seed ^ kSampleDraftSeedMix ^
                              (static_cast<uint64_t>(l + 1) * 0xD1B54A32D192ED03ull);
        const int64_t p = pos != nullptr ? pos[l + 1] : static_cast<int64_t>(l + 1);
        const double u = uniform01(seed, static_cast<uint64_t>(p < 0 ? 0 : p));
        double cum = 0.0;
        int chosen = -1, last_live = 0;
        for (int c = 0; c < k; ++c) {
          if (ex[c] <= 0.0f) continue;
          last_live = c;
          cum += static_cast<double>(ex[c]) / static_cast<double>(den);
          if (cum > u) {
            chosen = c;
            break;
          }
        }
        best = chosen >= 0 ? chosen : last_live;  // rounding past 1: the last survivor
        if (proposal != nullptr) write_proposal(proposal + l, lid, ex, den, k, lid[best]);
        if (proposal_host != nullptr) write_proposal(proposal_host + l, lid, ex, den, k, lid[best]);
      } else if (proposal != nullptr || proposal_host != nullptr) {
        // The argmax is a point mass: no proposal describes it.
        if (proposal != nullptr) proposal[l].n = 0;
        if (proposal_host != nullptr) proposal_host[l].n = 0;
      }
      if (conf != nullptr) {
        // The chosen candidate's softmax mass at the walk's temperature
        // (the drawn walk's ex[] stand; the argmax walk's at 1), as a logit.
        float q;
        if (sampled) {
          q = __fdiv_rn(ex[best], den);
        } else {
          const float mx = sc[prev * k + best];
          float d = 0.0f;
          for (int c = 0; c < k; ++c) d = __fadd_rn(d, expf(__fsub_rn(sc[prev * k + c], mx)));
          q = __fdiv_rn(1.0f, d);
        }
        q = fminf(fmaxf(q, 1e-6f), 1.0f - 1e-6f);
        conf[l] = fminf(fmaxf(logf(q / (1.0f - q)), -30.0f), 30.0f);
      }
      tokens[l] = lid[best];
      prev = best;
    }
  }
}

// ---- the recorded block draft ---------------------------------------------------

__global__ void stage_block_kernel(const PickVerdict* __restrict__ verdict,
                                   const int64_t* __restrict__ session_pos, int64_t mask_id, int rows,
                                   int64_t max_context, int64_t* __restrict__ pos,
                                   int64_t* __restrict__ tokens) {
  dgpp_pdl_wait();
  const int j = threadIdx.x;
  if (j >= rows) return;
  const int64_t p = *session_pos + j;
  pos[j] = p < max_context ? p : -1;
  tokens[j] = j == 0 ? static_cast<int64_t>(verdict->next) : mask_id;
}

__global__ void block_feed_kernel(const PickVerdict* __restrict__ verdict,
                                  const int32_t* __restrict__ drafts, int count,
                                  int64_t* __restrict__ tokens) {
  dgpp_pdl_wait();
  const int j = threadIdx.x;
  if (j > count) return;
  tokens[j] = j == 0 ? static_cast<int64_t>(verdict->next) : static_cast<int64_t>(drafts[j - 1]);
}

__global__ void publish_drafts_kernel(const int32_t* __restrict__ drafts, int32_t* __restrict__ pinned,
                                      int count) {
  dgpp_pdl_wait();
  const int j = threadIdx.x;
  if (j >= count) return;
  asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(pinned + j), "r"(drafts[j]) : "memory");
}

__global__ void publish_words_kernel(const int32_t* __restrict__ src, int32_t* __restrict__ pinned, int count) {
  dgpp_pdl_wait();
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < count; j += gridDim.x * blockDim.x)
    asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(pinned + j), "r"(src[j]) : "memory");
}

// The fixed batch's forms: one block per request slot q (== request q).
__global__ void stage_block_batched_kernel(const PickVerdict* __restrict__ verdicts,
                                           const int64_t* __restrict__ session_pos, int64_t mask_id,
                                           int rows, int64_t max_context, int64_t* __restrict__ pos,
                                           int64_t* __restrict__ tokens) {
  dgpp_pdl_wait();
  const int q = blockIdx.x;
  const int j = threadIdx.x;
  if (j >= rows) return;
  const bool active = verdicts[q].accepted > 0 && verdicts[q].next >= 0;
  const int64_t p = session_pos[q] + j;
  pos[q * rows + j] = active && p < max_context ? p : -1;
  tokens[q * rows + j] = !active ? 0 : (j == 0 ? static_cast<int64_t>(verdicts[q].next) : mask_id);
}

__global__ void block_feed_batched_kernel(const PickVerdict* __restrict__ verdicts,
                                          const int32_t* __restrict__ drafts, int count, int rows,
                                          int64_t* __restrict__ feeds) {
  dgpp_pdl_wait();
  const int q = blockIdx.x;
  const int j = threadIdx.x;
  if (j >= rows) return;
  const bool active = verdicts[q].accepted > 0 && verdicts[q].next >= 0;
  int64_t token = 0;
  if (active && j == 0) token = verdicts[q].next;
  if (active && j >= 1 && j - 1 < count) {
    const int32_t id = drafts[q * count + j - 1];
    token = id >= 0 ? id : 0;  // any valid id; a bad draft never stands
  }
  feeds[q * rows + j] = token;
}

__global__ void publish_drafts_batched_kernel(const int32_t* __restrict__ drafts,
                                              int32_t* __restrict__ pinned, int count) {
  dgpp_pdl_wait();
  const int q = blockIdx.x;
  const int j = threadIdx.x;
  if (j >= count) return;
  asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(pinned + q * count + j),
               "r"(drafts[q * count + j])
               : "memory");
}

// The wire digits (kernels/pick.hpp's form): 6 bits per bf16 slot.
__device__ __forceinline__ uint16_t digit_bits(uint32_t d) {
  return float_to_bf16_bits(static_cast<float>(d));
}
__device__ __forceinline__ uint32_t digit_value(uint16_t b) {
  return static_cast<uint32_t>(bf16_bits_to_float(b));
}

// One thread per (row, rank slot, candidate): this rank's slots carry the
// candidate, every other slot zeros (the fold's disjoint-slot gather).
__global__ void topk_stage_kernel(const int32_t* __restrict__ ids, const float* __restrict__ scores,
                                  int rows, int k, int32_t vocab_begin, int rank, int world,
                                  uint16_t* __restrict__ table, size_t elems) {
  dgpp_pdl_wait();
  const size_t slots = static_cast<size_t>(rows) * world * k;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < slots;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const int c = static_cast<int>(i % k);
    const int w = static_cast<int>((i / k) % world);
    const int r = static_cast<int>(i / (static_cast<size_t>(k) * world));
    uint16_t* slot = table + i * kDflash2TopkDigits;
    if (w != rank) {
#pragma unroll
      for (int d = 0; d < kDflash2TopkDigits; ++d) slot[d] = 0;
      continue;
    }
    const uint32_t bits = __float_as_uint(scores[static_cast<size_t>(r) * k + c]);
    const uint32_t id = static_cast<uint32_t>(ids[static_cast<size_t>(r) * k + c] + vocab_begin);
#pragma unroll
    for (int d = 0; d < 6; ++d) slot[d] = digit_bits((bits >> (6 * d)) & 63u);
#pragma unroll
    for (int d = 0; d < 3; ++d) slot[6 + d] = digit_bits((id >> (6 * d)) & 63u);
  }
  // The even-count pad slot, zeroed by thread 0 of block 0.
  if (blockIdx.x == 0 && threadIdx.x == 0 && elems > slots * kDflash2TopkDigits)
    table[elems - 1] = 0;
}

// One block per row; thread 0 walks the world x k decoded candidates into
// the row's top-K (the canonical order: score desc, id asc).
template <int K>
__global__ void topk_merge_kernel(const uint16_t* __restrict__ table, int world,
                                  int32_t* __restrict__ ids, float* __restrict__ scores) {
  dgpp_pdl_wait();
  const int row = blockIdx.x;
  __shared__ Cand cands[kPickMaxWorld * K];
  const int n = world * K;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const uint16_t* slot = table + (static_cast<size_t>(row) * n + i) * kDflash2TopkDigits;
    uint32_t bits = 0, id = 0;
#pragma unroll
    for (int d = 0; d < 6; ++d) bits |= digit_value(slot[d]) << (6 * d);
#pragma unroll
    for (int d = 0; d < 3; ++d) id |= digit_value(slot[6 + d]) << (6 * d);
    cands[i] = {__uint_as_float(bits), static_cast<int32_t>(id)};
  }
  __syncthreads();
  if (threadIdx.x != 0) return;
  Cand top[K];
  for (int j = 0; j < K; ++j) top[j] = {-INFINITY, 0x7fffffff};
  for (int i = 0; i < n; ++i) {
    const Cand c = cands[i];
    if (!better_cand(c, top[K - 1])) continue;
    int j = K - 1;
    while (j > 0 && better_cand(c, top[j - 1])) {
      top[j] = top[j - 1];
      --j;
    }
    top[j] = c;
  }
  for (int j = 0; j < K; ++j) {
    ids[row * K + j] = top[j].id;
    scores[row * K + j] = top[j].score;
  }
}

}  // namespace

void dflash2_grouped_conv_bf16(const uint16_t* x, const uint16_t* delta, const uint16_t* base,
                               uint16_t* out, int rows, int block_rows, int hidden, int taps,
                               int group_size, int64_t delta_row_stride, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!x || !delta || !base || !out || rows < block_rows || block_rows < 1 || hidden <= 0 || taps != 2 ||
      hidden % group_size)
    throw std::invalid_argument("dflash2_grouped_conv: bad arguments (taps must be 2, group_size must divide hidden)");
  grouped_conv_kernel<<<rows, 256, 0, stream>>>(x, delta, base, out, block_rows, hidden, group_size,
                                                delta_row_stride);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dflash2_acc_f32(float* acc, const float* src, int64_t n, int beta, cudaStream_t stream) {
  if (n <= 0) return;
  const int blocks = static_cast<int>((n + 255) / 256);
  acc_kernel<<<blocks, 256, 0, stream>>>(acc, src, n, beta);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dflash2_rmsnorm_bf16(const uint16_t* x, const uint16_t* w, uint16_t* y, int64_t rows, int dim,
                          float eps, cudaStream_t stream) {
  if (rows <= 0) return;
  rmsnorm_kernel<false, uint16_t><<<static_cast<unsigned>(rows), 256, 0, stream>>>(x, w, y, nullptr,
                                                                                   dim, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dflash2_add_rmsnorm_bf16(uint16_t* resid, const uint16_t* add, const uint16_t* w, uint16_t* y,
                              int64_t rows, int dim, float eps, cudaStream_t stream) {
  if (rows <= 0) return;
  rmsnorm_kernel<true, uint16_t><<<static_cast<unsigned>(rows), 256, 0, stream>>>(add, w, y, resid,
                                                                                  dim, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dflash2_norm_f32_bf16(const float* x, const uint16_t* w, uint16_t* y, int64_t rows, int dim,
                           float eps, cudaStream_t stream) {
  if (rows <= 0) return;
  rmsnorm_kernel<false, float><<<static_cast<unsigned>(rows), 256, 0, stream>>>(x, w, y, nullptr, dim,
                                                                                eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dflash2_norm_rope_bf16(const uint16_t* x, int64_t x_row_stride, const uint16_t* w,
                            const int64_t* pos, const float* inv_freq, uint16_t* out,
                            int64_t out_row_stride, int rows, int heads, int dim, float eps,
                            cudaStream_t stream) {
  if (rows <= 0) return;
  if (dim != 128) throw std::invalid_argument("dflash2_norm_rope: dim must be 128");
  const dim3 grid(static_cast<unsigned>(rows), heads);
  norm_rope_kernel<<<grid, 32, 0, stream>>>(x, x_row_stride, w, pos, inv_freq, out, out_row_stride,
                                            dim, dim / 2, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dflash2_block_attn(const uint16_t* q, int64_t q_row_stride, const uint16_t* k_cache,
                        const uint16_t* v_cache, const int32_t* block_table, int block_tokens,
                        int blocks_per_request, int block_rows, int64_t window, const int64_t* pos,
                        int rows, int heads, int kv_heads, int dim, float scale, uint16_t* out,
                        cudaStream_t stream) {
  if (rows <= 0) return;
  if (dim != 128 || heads % kv_heads || heads / kv_heads > 8)
    throw std::invalid_argument("dflash2_block_attn: dim must be 128 and heads/kv_heads in [1, 8]");
  if (block_rows < 1 || rows % block_rows != 0)
    throw std::invalid_argument("dflash2_block_attn: rows must be whole blocks");
  const dim3 grid(rows, kv_heads);
  block_attn_kernel<<<grid, 256, 0, stream>>>(q, q_row_stride, k_cache, v_cache, block_table,
                                              block_tokens, blocks_per_request, block_rows, window,
                                              pos, scale, out, heads, kv_heads, dim);
  DGPP_CUDA_OK(cudaGetLastError());
}

int dflash2_block_attn_splits() { return kBlockAttnSplits; }

size_t dflash2_block_attn_partials_bytes(int rows, int heads) {
  return static_cast<size_t>(rows) * static_cast<size_t>(heads) * kBlockAttnSplits * (128 + 2) * sizeof(float);
}

void dflash2_block_attn_split(const uint16_t* q, int64_t q_row_stride, const uint16_t* k_cache,
                              const uint16_t* v_cache, const int32_t* block_table, int block_tokens,
                              int blocks_per_request, int block_rows, int64_t window, const int64_t* pos,
                              int rows, int heads, int kv_heads, int dim, float scale, float* partials,
                              uint16_t* out, cudaStream_t stream) {
  if (rows <= 0) return;
  if (dim != 128 || heads % kv_heads || heads / kv_heads > 8)
    throw std::invalid_argument("dflash2_block_attn_split: dim must be 128 and heads/kv_heads in [1, 8]");
  if (block_rows < 1 || rows % block_rows != 0)
    throw std::invalid_argument("dflash2_block_attn_split: rows must be whole blocks");
  if (partials == nullptr) throw std::invalid_argument("dflash2_block_attn_split: null partials");
  const int hpq = heads / kv_heads;
  const dim3 grid(rows, kv_heads, kBlockAttnSplits);
  block_attn_split_kernel<<<grid, 32 * hpq, 0, stream>>>(q, q_row_stride, k_cache, v_cache, block_table,
                                                         block_tokens, blocks_per_request, block_rows, window,
                                                         pos, scale, partials, heads, kv_heads, dim);
  DGPP_CUDA_OK(cudaGetLastError());
  const dim3 cgrid(rows, heads);
  block_attn_combine_kernel<<<cgrid, 32, 0, stream>>>(partials, heads, dim, out);
  DGPP_CUDA_OK(cudaGetLastError());
}

size_t dflash2_topk_ws_bytes(int64_t vocab, int rows, int k) {
  return static_cast<size_t>(rows < 1 ? 1 : rows) * static_cast<size_t>(topk_chunks(vocab)) *
         static_cast<size_t>(k) * sizeof(Cand);
}

void dflash2_topk_f32(const float* logits, int32_t* ids, float* scores, int64_t vocab, int rows,
                      int k, cudaStream_t stream, void* ws, size_t ws_bytes) {
  if (rows <= 0) return;
  if (k != 16) throw std::invalid_argument("dflash2_topk: only k = 16 is implemented");
  // Each row in chunks of kTopkChunkSpan over as many blocks (32 at most),
  // the chunks' lists folded by one block a row.
  const int chunks = topk_chunks(vocab);
  if (chunks > 1 && (ws == nullptr || ws_bytes < dflash2_topk_ws_bytes(vocab, rows, k)))
    throw std::invalid_argument("dflash2_topk: the partials workspace is too small (dflash2_topk_ws_bytes)");
  const dim3 grid(static_cast<unsigned>(chunks), static_cast<unsigned>(rows));
  topk_chunk_kernel<16><<<grid, kTopkThreads, 0, stream>>>(logits, vocab, chunks, static_cast<Cand*>(ws), ids,
                                                           scores);
  DGPP_CUDA_OK(cudaGetLastError());
  if (chunks > 1) {
    topk_merge_kernel<16><<<static_cast<unsigned>(rows), kTopkThreads, 0, stream>>>(
        static_cast<const Cand*>(ws), chunks, ids, scores);
    DGPP_CUDA_OK(cudaGetLastError());
  }
}

void dflash2_selector_walk(const int32_t* ids, const float* unary, const float* hidden,
                           const uint16_t* pred_cb, const uint16_t* succ_cb, const int64_t* anchor,
                           int32_t* tokens, int steps, int k, int rank, cudaStream_t stream,
                           const int64_t* pos, const SampleSpec* spec, DraftProposal* proposal,
                           DraftProposal* proposal_host, float* conf) {
  if (steps <= 0) return;
  if (k < 2 || rank <= 0 || anchor == nullptr) throw std::invalid_argument("dflash2_selector_walk: k/rank/anchor");
  if (spec != nullptr && k > kSampleProposalMax)
    throw std::invalid_argument("dflash2_selector_walk: the proposal holds at most kSampleProposalMax candidates");
  if (spec != nullptr && steps > kSampleProposalSlots && (proposal != nullptr || proposal_host != nullptr))
    throw std::invalid_argument("dflash2_selector_walk: more steps than proposal slots");
  const size_t smem = (static_cast<size_t>(rank) * (2 * k + 1) + k * k + k) * 4;
  if (smem > 47u * 1024)
    throw std::invalid_argument("dflash2_selector_walk: shared memory over the static limit");
  selector_kernel<<<1, 256, smem, stream>>>(ids, unary, hidden, pred_cb, succ_cb, anchor, tokens,
                                            steps, k, rank, pos, spec, proposal, proposal_host, conf);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dflash2_stage_block(const PickVerdict* verdict, const int64_t* session_pos, int64_t mask_id,
                         int rows, int64_t max_context, int64_t* pos, int64_t* tokens,
                         cudaStream_t stream) {
  if (verdict == nullptr || session_pos == nullptr || pos == nullptr || tokens == nullptr || rows < 1 ||
      rows > 32)
    throw std::invalid_argument("dflash2_stage_block: arguments");
  stage_block_kernel<<<1, 32, 0, stream>>>(verdict, session_pos, mask_id, rows, max_context, pos, tokens);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dflash2_block_feed(const PickVerdict* verdict, const int32_t* drafts, int count, int64_t* tokens,
                        cudaStream_t stream) {
  if (verdict == nullptr || drafts == nullptr || tokens == nullptr || count < 1 || count > 31)
    throw std::invalid_argument("dflash2_block_feed: arguments");
  block_feed_kernel<<<1, 32, 0, stream>>>(verdict, drafts, count, tokens);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dflash2_publish_drafts(const int32_t* drafts, int32_t* pinned, int count, cudaStream_t stream) {
  if (drafts == nullptr || pinned == nullptr || count < 1 || count > 32)
    throw std::invalid_argument("dflash2_publish_drafts: arguments");
  publish_drafts_kernel<<<1, 32, 0, stream>>>(drafts, pinned, count);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dflash2_publish_words(const int32_t* src, int32_t* pinned, int count, cudaStream_t stream) {
  if (src == nullptr || pinned == nullptr || count < 1 || count > 1024)
    throw std::invalid_argument("dflash2_publish_words: arguments");
  publish_words_kernel<<<(count + 127) / 128, 128, 0, stream>>>(src, pinned, count);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dflash2_stage_block_batched(const PickVerdict* verdicts, const int64_t* session_pos, int64_t mask_id,
                                 int rows, int requests, int64_t max_context, int64_t* pos,
                                 int64_t* tokens, cudaStream_t stream) {
  if (verdicts == nullptr || session_pos == nullptr || pos == nullptr || tokens == nullptr || rows < 1 ||
      rows > 32 || requests < 1 || requests > kPickMaxRequests)
    throw std::invalid_argument("dflash2_stage_block_batched: arguments");
  stage_block_batched_kernel<<<requests, 32, 0, stream>>>(verdicts, session_pos, mask_id, rows, max_context,
                                                          pos, tokens);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dflash2_block_feed_batched(const PickVerdict* verdicts, const int32_t* drafts, int count, int requests,
                                int rows, int64_t* feeds, cudaStream_t stream) {
  if (verdicts == nullptr || drafts == nullptr || feeds == nullptr || count < 1 || rows != 1 + count ||
      rows > 32 || requests < 1 || requests > kPickMaxRequests)
    throw std::invalid_argument("dflash2_block_feed_batched: arguments");
  block_feed_batched_kernel<<<requests, 32, 0, stream>>>(verdicts, drafts, count, rows, feeds);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dflash2_publish_drafts_batched(const int32_t* drafts, int32_t* pinned, int count, int requests,
                                    cudaStream_t stream) {
  if (drafts == nullptr || pinned == nullptr || count < 1 || count > 32 || requests < 1 ||
      requests > kPickMaxRequests)
    throw std::invalid_argument("dflash2_publish_drafts_batched: arguments");
  publish_drafts_batched_kernel<<<requests, 32, 0, stream>>>(drafts, pinned, count);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dflash2_topk_stage(const int32_t* ids, const float* scores, int rows, int k, int32_t vocab_begin,
                        int rank, int world, uint16_t* table, cudaStream_t stream) {
  if (rows <= 0) return;
  if (ids == nullptr || scores == nullptr || table == nullptr || k < 1 || world < 1 ||
      world > kPickMaxWorld || rank < 0 || rank >= world || vocab_begin < 0)
    throw std::invalid_argument("dflash2_topk_stage: arguments");
  const size_t elems = dflash2_topk_table_elems(rows, k, world);
  const size_t slots = static_cast<size_t>(rows) * world * k;
  const unsigned blocks = static_cast<unsigned>(std::min<size_t>((slots + 255) / 256, 1024));
  topk_stage_kernel<<<blocks, 256, 0, stream>>>(ids, scores, rows, k, vocab_begin, rank, world, table,
                                                elems);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dflash2_topk_merge(const uint16_t* table, int rows, int k, int world, int32_t* ids, float* scores,
                        cudaStream_t stream) {
  if (rows <= 0) return;
  if (k != 16) throw std::invalid_argument("dflash2_topk_merge: only k = 16 is implemented");
  if (table == nullptr || ids == nullptr || scores == nullptr || world < 1 || world > kPickMaxWorld)
    throw std::invalid_argument("dflash2_topk_merge: arguments");
  topk_merge_kernel<16><<<rows, 128, 0, stream>>>(table, world, ids, scores);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
