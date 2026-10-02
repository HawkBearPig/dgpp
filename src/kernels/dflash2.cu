#include "kernels/dflash2.hpp"

#include <cmath>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

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
                                  int blocks_per_request, int64_t ctx_end, int64_t blk_lo,
                                  int64_t blk_hi, int64_t window, const int64_t* __restrict__ pos,
                                  float scale, uint16_t* __restrict__ out, int heads, int kv_heads,
                                  int dim) {
  const int row = blockIdx.x;
  const int kvh = blockIdx.y;
  const int nwarp = blockDim.x >> 5;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int hpq = heads / kv_heads;
  const int64_t p = pos[row];
  const int64_t lo = p - window + 1;  // keys below lo are windowed out
  for (int hh = warp; hh < hpq; hh += nwarp) {
    const int head = kvh * hpq + hh;
    const int d0 = lane * 4;  // dim is 128: four elements per lane
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

// ---- the candidate top-K --------------------------------------------------------

struct Cand {
  float score;
  int32_t id;
};
__device__ __forceinline__ bool better_cand(const Cand& a, const Cand& b) {
  return a.score > b.score || (a.score == b.score && a.id < b.id);
}

// Thread-local top-K lists (register insertion; one tail compare rejects
// almost everything once warm) merged by thread 0's walk over the block's
// blockDim * K candidates. Descending, ties to the lower id.
template <int K>
__global__ void topk_kernel(const float* __restrict__ logits, int32_t* __restrict__ ids,
                            float* __restrict__ scores, int64_t vocab) {
  const int row = blockIdx.y;
  const float* lg = logits + row * vocab;
  Cand best[K];
  for (int j = 0; j < K; ++j) best[j] = {-INFINITY, 0x7fffffff};
  for (int64_t v = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; v < vocab;
       v += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const Cand c{lg[v], static_cast<int32_t>(v)};
    if (!better_cand(c, best[K - 1])) continue;
    int j = K - 1;
    while (j > 0 && better_cand(c, best[j - 1])) {
      best[j] = best[j - 1];
      --j;
    }
    best[j] = c;
  }
  __shared__ Cand merge[256 * K];
  for (int j = 0; j < K; ++j) merge[threadIdx.x * K + j] = best[j];
  __syncthreads();
  if (threadIdx.x != 0) return;
  Cand top[K];
  for (int j = 0; j < K; ++j) top[j] = {-INFINITY, 0x7fffffff};
  const int total = blockDim.x * K;
  for (int i = 0; i < total; ++i) {
    const Cand c = merge[i];
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

// ---- the selector -----------------------------------------------------------------

// The scores[l][p][c] = unary[l][p] + <pred_code[id(l-1,p)] * hidden[l],
// succ_code[id(l,c)]> table and the greedy slot walk (vLLM
// qwen3_dflash2._score_edges + _selector_walk_kernel at temperature 0).
__global__ void selector_kernel(const int32_t* __restrict__ ids, const float* __restrict__ unary,
                                const float* __restrict__ hidden, const uint16_t* __restrict__ pred_cb,
                                const uint16_t* __restrict__ succ_cb, int32_t anchor,
                                int32_t* __restrict__ tokens, int steps, int k, int rank) {
  extern __shared__ float sm[];
  float* h = sm;                  // [rank]
  float* pred = h + rank;         // [k][rank]
  float* succ = pred + k * rank;  // [k][rank]
  float* sc = succ + k * rank;    // [k][k]
  const int tid = threadIdx.x;
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
      sc[e] = unary[static_cast<int64_t>(l) * k + p] + dot;
    }
    __syncthreads();
    if (tid == 0) {
      int best = 0;
      for (int c = 1; c < k; ++c)
        if (sc[prev * k + c] > sc[prev * k + best]) best = c;
      tokens[l] = ids[static_cast<int64_t>(l) * k + best];
      prev = best;
    }
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
                        int blocks_per_request, int64_t ctx_end, int64_t blk_lo, int64_t blk_hi,
                        int64_t window, const int64_t* pos, int rows, int heads, int kv_heads,
                        int dim, float scale, uint16_t* out, cudaStream_t stream) {
  if (rows <= 0) return;
  if (dim != 128 || heads % kv_heads || heads / kv_heads > 8)
    throw std::invalid_argument("dflash2_block_attn: dim must be 128 and heads/kv_heads in [1, 8]");
  const dim3 grid(rows, kv_heads);
  block_attn_kernel<<<grid, 256, 0, stream>>>(q, q_row_stride, k_cache, v_cache, block_table,
                                              block_tokens, blocks_per_request, ctx_end, blk_lo,
                                              blk_hi, window, pos, scale, out, heads, kv_heads, dim);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dflash2_topk_f32(const float* logits, int32_t* ids, float* scores, int64_t vocab, int rows,
                      int k, cudaStream_t stream) {
  if (rows <= 0) return;
  if (k != 16) throw std::invalid_argument("dflash2_topk: only k = 16 is implemented");
  // ONE block per row: every thread strides the vocab into a register
  // top-K and the block's lists merge in shared memory (several blocks
  // would race on the row's output with partial views).
  const dim3 grid(1, static_cast<unsigned>(rows));
  topk_kernel<16><<<grid, 256, 0, stream>>>(logits, ids, scores, vocab);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dflash2_selector_walk(const int32_t* ids, const float* unary, const float* hidden,
                           const uint16_t* pred_cb, const uint16_t* succ_cb, int32_t anchor,
                           int32_t* tokens, int steps, int k, int rank, cudaStream_t stream) {
  if (steps <= 0) return;
  if (k < 2 || rank <= 0) throw std::invalid_argument("dflash2_selector_walk: k/rank");
  const size_t smem = (static_cast<size_t>(rank) * (2 * k + 1) + k * k) * 4;
  if (smem > 47u * 1024)
    throw std::invalid_argument("dflash2_selector_walk: shared memory over the static limit");
  selector_kernel<<<1, 256, smem, stream>>>(ids, unary, hidden, pred_cb, succ_cb, anchor, tokens,
                                            steps, k, rank);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
