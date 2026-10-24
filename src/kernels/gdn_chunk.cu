// Chunked Gated DeltaNet prefill on the tensor cores (the FLA
// chunk_gated_delta_rule algorithm SGLang runs; 2026-09-23).
//
// Why: the recurrence ran token by token (kda_recurrent_kernel, scalar FP32
// FMAs and warp shuffles): 7.6% of a 32k prefill. The chunked form carries
// the state across 64-token chunks and does everything inside a chunk as
// small matrix products (bf16 in, fp32 accumulate: FLA's numerics).
//
// Per value head h (key head h / kv_ratio), chunk tokens i = 0..C-1:
//   g_i = -exp(A_log) * softplus(a_i + dt_bias)   (log decay), G = cumsum(g)
//   q, k l2-normalized (eps 1e-6 inside the sqrt), q *= scale, b = sigmoid(beta)
//   A_ij = b_i (k_i . k_j) exp(G_i - G_j) for j < i;  T = (I + A)^-1
//   W = T (b * exp(G) * k),  U = T (b * v)                   [prep kernel]
//   per chunk, S the state [V, K] entering it:               [state kernel]
//     Vn = U - W S^T
//     O  = (q * exp(G)) S^T + ((q k^T) o exp(G_i - G_j), j <= i) Vn
//     S  = exp(G_C) S + Vn^T (k * exp(G_C - G))
// Every exponent is <= 0, so nothing overflows. This equals the sequential
// recurrence exactly in real arithmetic; in floating point it differs from
// kda_recurrent_kernel by bf16 rounding of the matrix operands (the same
// class as FLA/SGLang). The model-level gate is BFCL / tool-eval.
#include "kernels/gdn_chunk.hpp"
#include "kernels/pdl.cuh"

#include <mma.h>

#include <algorithm>
#include <stdexcept>

#include <cuda_bf16.h>

#include "common/cuda_check.hpp"

namespace dgpp {
namespace {

using namespace nvcuda;
constexpr int C = 64;   // chunk
constexpr int D = 128;  // K = V
constexpr int BV = 32;  // the state kernel's value block (16 measured slower: 6.3 vs 5.06 ms -- each value block re-streams the chunk's full-K operands)
// The prep kernel's block (2026-09-23): its ~88 KB of shared memory allows one
// block per SM, so 128 threads left 4 warps resident; 512 (16) measured best:
// gdn_chunk_test 8192 tokens x 24 heads, chunked 5.57 (128) / 5.20 (256) / 5.06 ms.
constexpr int kPrepThreads = 512;
constexpr int kStateThreads = 256;

__device__ __forceinline__ float bf(uint16_t u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }
__device__ __forceinline__ __nv_bfloat16 tobf(float f) { return __float2bfloat16_rn(f); }

// Chunk-major operand blocks: block (h, c) of a [C, X] operand starts at
// ((h * chunks + c) * C) * X; padded rows past the tokens are zeros.
__device__ __forceinline__ int64_t blk(int h, int c, int chunks, int x) {
  return (static_cast<int64_t>(h) * chunks + c) * C * x;
}

// ---- prep: per (chunk, value head), everything that does not need the state:
//   W = T (b e^G k), U = T (b v), QG = q e^G, KD = k e^(G_C - G),
//   P = (q k^T) o e^(G_i - G_j) (j <= i), and e^(G_C). --------------------------
struct PrepSmem {
  __nv_bfloat16 qb[C][D];       // normalized, scaled q
  __nv_bfloat16 kb[C][D];       // normalized k, then b e^G k
  __nv_bfloat16 vb[C][D];       // b v
  float a[C][C];                // A (strictly lower), then P
  float t[C][C];                // T = (I + A)^-1 (lower, unit diagonal)
  __nv_bfloat16 tb[C][C];       // T (bf16)
  float g[C], G[C], beta[C];
  float eg[C], ed[C], bg[C];    // e^G, e^(G_C - G), b e^G per row
};

__global__ __launch_bounds__(kPrepThreads) void gdn_chunk_prep_kernel(
    const uint16_t* __restrict__ qkv, const uint16_t* __restrict__ a_raw, int64_t a_stride,
    const uint16_t* __restrict__ beta_raw, int64_t beta_stride, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, int tokens, int heads, int kv_ratio, float scale,
    __nv_bfloat16* __restrict__ w_out, __nv_bfloat16* __restrict__ u_out, __nv_bfloat16* __restrict__ qg_out,
    __nv_bfloat16* __restrict__ kd_out, __nv_bfloat16* __restrict__ p_out, float* __restrict__ decay_out) {
  dgpp_pdl_wait();
  extern __shared__ __align__(128) uint8_t smem_raw[];
  PrepSmem& s = *reinterpret_cast<PrepSmem*>(smem_raw);
  const int chunk = blockIdx.x, h = blockIdx.y, chunks = gridDim.x;
  const int hk = h / kv_ratio, heads_k = heads / kv_ratio;
  const int t0 = chunk * C;
  const int n = min(C, tokens - t0);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int64_t qkv_stride = 2LL * heads_k * D + static_cast<int64_t>(heads) * D;
  const float A = expf(a_log[h]);
  const float bias = dt_bias[h];

  if (tid < C) {
    float g = 0.f, b = 0.f;
    if (tid < n) {
      const int t = t0 + tid;
      const float x = bf(a_raw[static_cast<int64_t>(t) * a_stride + h]) + bias;
      const float sp = x > 20.f ? x : log1pf(expf(x));
      g = -(A * sp);
      b = 1.f / (1.f + expf(-bf(beta_raw[static_cast<int64_t>(t) * beta_stride + h])));
    }
    s.g[tid] = g;
    s.beta[tid] = b;
  }
  __syncthreads();
  if (tid == 0) {
    float acc = 0.f;
    for (int i = 0; i < C; ++i) {
      acc += s.g[i];
      s.G[i] = acc;  // padded rows (g = 0) carry G of the last live row
    }
    decay_out[static_cast<int64_t>(h) * chunks + chunk] = expf(acc);
  }
  // Normalized q, k and b v rows: one warp per row, 4 elements per lane.
  for (int i = warp; i < C; i += kPrepThreads / 32) {
    float kv[4], vv[4], qv[4];
    float ks = 0.f, qs = 0.f;
    const bool live = i < n;
    const uint16_t* row = qkv + static_cast<int64_t>(t0 + (live ? i : 0)) * qkv_stride;
    const uint2 q4 = *reinterpret_cast<const uint2*>(row + static_cast<int64_t>(hk) * D + lane * 4);
    const uint2 k4 = *reinterpret_cast<const uint2*>(row + static_cast<int64_t>(heads_k + hk) * D + lane * 4);
    const uint2 v4 = *reinterpret_cast<const uint2*>(row + 2LL * heads_k * D + static_cast<int64_t>(h) * D + lane * 4);
    const uint32_t qw[2] = {q4.x, q4.y}, kw[2] = {k4.x, k4.y}, vw[2] = {v4.x, v4.y};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const uint16_t qu = static_cast<uint16_t>(qw[e / 2] >> (16 * (e % 2)));
      const uint16_t ku = static_cast<uint16_t>(kw[e / 2] >> (16 * (e % 2)));
      const uint16_t vu = static_cast<uint16_t>(vw[e / 2] >> (16 * (e % 2)));
      qv[e] = live ? bf(qu) : 0.f;
      kv[e] = live ? bf(ku) : 0.f;
      vv[e] = live ? bf(vu) : 0.f;
      ks += kv[e] * kv[e];
      qs += qv[e] * qv[e];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      ks += __shfl_xor_sync(0xffffffffu, ks, o);
      qs += __shfl_xor_sync(0xffffffffu, qs, o);
    }
    const float rk = 1.f / sqrtf(ks + 1e-6f), rq = scale / sqrtf(qs + 1e-6f);
    const float b = s.beta[i];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      s.qb[i][lane * 4 + e] = tobf(qv[e] * rq);
      s.kb[i][lane * 4 + e] = tobf(kv[e] * rk);
      s.vb[i][lane * 4 + e] = tobf(vv[e] * b);
    }
  }
  __syncthreads();
  // A = k k^T (lower tiles) on the tensor cores.
  for (int tile = warp; tile < 16; tile += kPrepThreads / 32) {
    const int ti = tile / 4, tj = tile % 4;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    if (tj <= ti) {
      for (int kk = 0; kk < D; kk += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
        wmma::load_matrix_sync(fa, &s.kb[ti * 16][kk], D);
        wmma::load_matrix_sync(fb, &s.kb[tj * 16][kk], D);
        wmma::mma_sync(acc, fa, fb, acc);
      }
    }
    wmma::store_matrix_sync(&s.a[ti * 16][tj * 16], acc, C, wmma::mem_row_major);
  }
  __syncthreads();
  for (int e = tid; e < C * C; e += kPrepThreads) {
    const int i = e / C, j = e % C;
    s.a[i][j] = j < i ? s.beta[i] * s.a[i][j] * expf(s.G[i] - s.G[j]) : 0.f;
  }
  __syncthreads();
  // T = (I + A)^-1 by blocks of 16 (FLA's solve_tril form): each warp
  // inverts one diagonal block by forward substitution (16 warp-synchronous
  // steps, T_ic = -A_ic - sum_{c<j<i} A_ij T_jc), then the off-diagonal
  // blocks by distance d = 1..3: T_IJ = -T_II (sum_{J<=K<I} A_IK T_KJ).
  // The serial critical path is 16 steps and 6 barriers instead of 63.
  for (int e = tid; e < C * C; e += kPrepThreads) s.t[e / C][e % C] = 0.f;
  __syncthreads();
  if (warp < C / 16) {  // one warp per diagonal block (warps past C/16 idle here)
    const int b0 = warp * 16;
    const int c = lane & 15;
    if (lane < 16) s.t[b0 + c][b0 + c] = 1.f;
    for (int i = 1; i < 16; ++i) {
      __syncwarp();
      if (lane < 16 && c < i) {
        float acc = -s.a[b0 + i][b0 + c];
        for (int j = c + 1; j < i; ++j) acc -= s.a[b0 + i][b0 + j] * s.t[b0 + j][b0 + c];
        s.t[b0 + i][b0 + c] = acc;
      }
    }
  }
  __syncthreads();
  for (int d = 1; d < 4; ++d) {
    // M_IJ = sum_{J<=K<I} A_IK T_KJ for each (I = J + d); staged in s.t's
    // block (I, J) as scratch (still zero), then T_IJ = -T_II M_IJ.
    const int pairs = 4 - d;
    float m[8];  // up to 256 * 3 elements over 128 threads: <= 6 each
    int cnt = 0;
    for (int e = tid; e < pairs * 256; e += kPrepThreads, ++cnt) {
      const int J = e / 256, I = J + d, r = (e % 256) / 16, cc = e % 16;
      float acc = 0.f;
      for (int K = J; K < I; ++K)
        for (int k = 0; k < 16; ++k) acc += s.a[I * 16 + r][K * 16 + k] * s.t[K * 16 + k][J * 16 + cc];
      m[cnt] = acc;
    }
    __syncthreads();
    cnt = 0;
    for (int e = tid; e < pairs * 256; e += kPrepThreads, ++cnt) {
      const int J = e / 256, I = J + d, r = (e % 256) / 16, cc = e % 16;
      s.t[I * 16 + r][J * 16 + cc] = m[cnt];
    }
    __syncthreads();
    cnt = 0;
    for (int e = tid; e < pairs * 256; e += kPrepThreads, ++cnt) {
      const int J = e / 256, I = J + d, r = (e % 256) / 16, cc = e % 16;
      float acc = 0.f;
      for (int k = 0; k <= r; ++k) acc -= s.t[I * 16 + r][I * 16 + k] * s.t[I * 16 + k][J * 16 + cc];
      m[cnt] = acc;
    }
    __syncthreads();
    cnt = 0;
    for (int e = tid; e < pairs * 256; e += kPrepThreads, ++cnt) {
      const int J = e / 256, I = J + d, r = (e % 256) / 16, cc = e % 16;
      s.t[I * 16 + r][J * 16 + cc] = m[cnt];
    }
    __syncthreads();
  }
  for (int e = tid; e < C * C; e += kPrepThreads) {
    const int i = e / C, j = e % C;
    s.tb[i][j] = tobf(j <= i ? s.t[i][j] : 0.f);
  }
  __syncthreads();
  // P = q k^T (lower tiles) into s.a, then masked and decayed to global.
  for (int tile = warp; tile < 16; tile += kPrepThreads / 32) {
    const int ti = tile / 4, tj = tile % 4;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    if (tj <= ti) {
      for (int kk = 0; kk < D; kk += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
        wmma::load_matrix_sync(fa, &s.qb[ti * 16][kk], D);
        wmma::load_matrix_sync(fb, &s.kb[tj * 16][kk], D);
        wmma::mma_sync(acc, fa, fb, acc);
      }
    }
    wmma::store_matrix_sync(&s.a[ti * 16][tj * 16], acc, C, wmma::mem_row_major);
  }
  __syncthreads();
  const float G_last = s.G[C - 1];
  __nv_bfloat16* pb = p_out + blk(h, chunk, chunks, C);
  for (int e = tid; e < C * C; e += kPrepThreads) {
    const int i = e / C, j = e % C;
    pb[e] = tobf(j <= i ? s.a[i][j] * expf(s.G[i] - s.G[j]) : 0.f);
  }
  // QG = q e^G, KD = k e^(G_C - G), then kb <- b e^G k. The factors are
  // per row: one expf each per row, not per element.
  if (tid < C) {
    s.eg[tid] = expf(s.G[tid]);
    s.ed[tid] = expf(G_last - s.G[tid]);
    s.bg[tid] = s.beta[tid] * s.eg[tid];
  }
  __syncthreads();
  __nv_bfloat16* qg = qg_out + blk(h, chunk, chunks, D);
  __nv_bfloat16* kd = kd_out + blk(h, chunk, chunks, D);
  for (int e = tid; e < C * D / 2; e += kPrepThreads) {
    const int i = (2 * e) / D, c = (2 * e) % D;
    const __nv_bfloat162 q2 = *reinterpret_cast<const __nv_bfloat162*>(&s.qb[i][c]);
    const __nv_bfloat162 k2 = *reinterpret_cast<const __nv_bfloat162*>(&s.kb[i][c]);
    const float2 qf = __bfloat1622float2(q2), kf = __bfloat1622float2(k2);
    reinterpret_cast<__nv_bfloat162*>(qg)[e] = __floats2bfloat162_rn(qf.x * s.eg[i], qf.y * s.eg[i]);
    reinterpret_cast<__nv_bfloat162*>(kd)[e] = __floats2bfloat162_rn(kf.x * s.ed[i], kf.y * s.ed[i]);
  }
  __syncthreads();
  for (int e = tid; e < C * D; e += kPrepThreads) {
    const int i = e / D, c = e % D;
    s.kb[i][c] = tobf(__bfloat162float(s.kb[i][c]) * s.bg[i]);
  }
  __syncthreads();
  // W = T kb, U = T vb, full blocks (padded rows are zeros).
  __nv_bfloat16* wb = w_out + blk(h, chunk, chunks, D);
  __nv_bfloat16* ub = u_out + blk(h, chunk, chunks, D);
  for (int tile = warp; tile < 64; tile += kPrepThreads / 32) {
    const bool is_u = tile >= 32;
    const int tt = tile % 32, ti = tt / 8, tj = tt % 8;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    for (int kk = 0; kk <= ti * 16; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fb;
      wmma::load_matrix_sync(fa, &s.tb[ti * 16][kk], C);
      wmma::load_matrix_sync(fb, is_u ? &s.vb[kk][tj * 16] : &s.kb[kk][tj * 16], D);
      wmma::mma_sync(acc, fa, fb, acc);
    }
    float* stage = &s.a[0][0] + warp * 256;
    wmma::store_matrix_sync(stage, acc, 16, wmma::mem_row_major);
    __syncwarp();
    __nv_bfloat16* dst = is_u ? ub : wb;
    for (int e = lane; e < 256; e += 32)
      dst[(ti * 16 + e / 16) * D + tj * 16 + e % 16] = tobf(stage[e]);
    __syncwarp();
  }
}

// ---- state: per (value head, value block), sequential over the chunks.
// Per chunk: Vn = U - W S^T; O = QG S^T + P Vn; S = e^(G_C) S + Vn^T KD. ----
__device__ __forceinline__ void cp16(void* dst, const void* src) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
}

constexpr size_t kStateSmem =
    sizeof(__nv_bfloat16) * C * D * 3 +   // W, QG, KD
    sizeof(__nv_bfloat16) * C * BV +      // U block
    sizeof(__nv_bfloat16) * C * C +       // P
    sizeof(float) * BV * D +              // S fp32
    sizeof(__nv_bfloat16) * BV * D +      // S bf16
    sizeof(float) * C * BV +              // Vn / O fp32
    sizeof(__nv_bfloat16) * C * BV;       // Vn bf16

__global__ __launch_bounds__(kStateThreads) void gdn_chunk_state_kernel(
    const __nv_bfloat16* __restrict__ w_in, const __nv_bfloat16* __restrict__ u_in,
    const __nv_bfloat16* __restrict__ qg_in, const __nv_bfloat16* __restrict__ kd_in,
    const __nv_bfloat16* __restrict__ p_in, const float* __restrict__ decay_in, float* __restrict__ state,
    uint16_t* __restrict__ out, int tokens, int heads) {
  dgpp_pdl_wait();
  extern __shared__ __align__(128) uint8_t smem_raw[];
  uint8_t* p = smem_raw;
  auto take = [&](size_t bytes) {
    uint8_t* r = p;
    p += (bytes + 127) & ~size_t(127);
    return r;
  };
  auto* sw = reinterpret_cast<__nv_bfloat16(*)[D]>(take(sizeof(__nv_bfloat16) * C * D));
  auto* sqg = reinterpret_cast<__nv_bfloat16(*)[D]>(take(sizeof(__nv_bfloat16) * C * D));
  auto* skd = reinterpret_cast<__nv_bfloat16(*)[D]>(take(sizeof(__nv_bfloat16) * C * D));
  auto* su = reinterpret_cast<__nv_bfloat16(*)[BV]>(take(sizeof(__nv_bfloat16) * C * BV));
  auto* sp = reinterpret_cast<__nv_bfloat16(*)[C]>(take(sizeof(__nv_bfloat16) * C * C));
  auto* s32 = reinterpret_cast<float(*)[D]>(take(sizeof(float) * BV * D));
  auto* s16 = reinterpret_cast<__nv_bfloat16(*)[D]>(take(sizeof(__nv_bfloat16) * BV * D));
  auto* o32 = reinterpret_cast<float(*)[BV]>(take(sizeof(float) * C * BV));
  auto* vn16 = reinterpret_cast<__nv_bfloat16(*)[BV]>(take(sizeof(__nv_bfloat16) * C * BV));

  const int h = blockIdx.x, vb0 = blockIdx.y * BV;
  const int tid = threadIdx.x, warp = tid >> 5;
  const int chunks = (tokens + C - 1) / C;
  float* st = state + (static_cast<int64_t>(h) * D + vb0) * D;  // S[v][k], v in [vb0, vb0 + BV)

  auto issue = [&](int ch) {
    const __nv_bfloat16* w = w_in + blk(h, ch, chunks, D);
    const __nv_bfloat16* qg = qg_in + blk(h, ch, chunks, D);
    const __nv_bfloat16* kd = kd_in + blk(h, ch, chunks, D);
    const __nv_bfloat16* u = u_in + blk(h, ch, chunks, D);
    const __nv_bfloat16* pp = p_in + blk(h, ch, chunks, C);
    for (int e = tid; e < C * D / 8; e += kStateThreads) {  // 16-byte pieces
      cp16(&sw[0][0] + e * 8, w + e * 8);
      cp16(&sqg[0][0] + e * 8, qg + e * 8);
      cp16(&skd[0][0] + e * 8, kd + e * 8);
    }
    for (int e = tid; e < C * BV / 8; e += kStateThreads) {
      const int i = e / (BV / 8), c = (e % (BV / 8)) * 8;
      cp16(&su[i][c], u + i * D + vb0 + c);
    }
    for (int e = tid; e < C * C / 8; e += kStateThreads) cp16(&sp[0][0] + e * 8, pp + e * 8);
    asm volatile("cp.async.commit_group;\n" ::);
  };

  for (int e = tid; e < BV * D; e += kStateThreads) {
    const float v = st[e];
    s32[e / D][e % D] = v;
    s16[e / D][e % D] = tobf(v);
  }
  for (int ch = 0; ch < chunks; ++ch) {
    const int t0 = ch * C, n = min(C, tokens - t0);
    __syncthreads();  // the previous chunk's readers are done with the buffers
    issue(ch);
    if (ch + 1 < chunks && tid < 3) {
      // The next chunk's operand blocks into L2 while this one computes.
      const __nv_bfloat16* nx = (tid == 0 ? w_in : tid == 1 ? qg_in : kd_in) + blk(h, ch + 1, chunks, D);
      for (int l = 0; l < C * D * 2 / 128; l += 8) asm volatile("prefetch.global.L2 [%0];" ::"l"(nx + l * 64));
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    // 1. Vn = U - W S^T  ((C/16) x (BV/16) tiles) -> fp32 staging.
    for (int tile = warp; tile < (C / 16) * (BV / 16); tile += kStateThreads / 32) {
      const int ti = tile / (BV / 16), tj = tile % (BV / 16);
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.f);
      for (int kk = 0; kk < D; kk += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
        wmma::load_matrix_sync(fa, &sw[ti * 16][kk], D);
        wmma::load_matrix_sync(fb, &s16[tj * 16][kk], D);
        wmma::mma_sync(acc, fa, fb, acc);
      }
      wmma::store_matrix_sync(&o32[ti * 16][tj * 16], acc, BV, wmma::mem_row_major);
    }
    __syncthreads();
    for (int e = tid; e < C * BV; e += kStateThreads) {
      const int i = e / BV, c = e % BV;
      vn16[i][c] = tobf(__bfloat162float(su[i][c]) - o32[i][c]);
    }
    __syncthreads();
    // 2. O = QG S^T + P Vn  ((C/16) x (BV/16) tiles) -> bf16 out.
    for (int tile = warp; tile < (C / 16) * (BV / 16); tile += kStateThreads / 32) {
      const int ti = tile / (BV / 16), tj = tile % (BV / 16);
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.f);
      for (int kk = 0; kk < D; kk += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
        wmma::load_matrix_sync(fa, &sqg[ti * 16][kk], D);
        wmma::load_matrix_sync(fb, &s16[tj * 16][kk], D);
        wmma::mma_sync(acc, fa, fb, acc);
      }
      for (int kk = 0; kk <= ti * 16; kk += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fb;
        wmma::load_matrix_sync(fa, &sp[ti * 16][kk], C);
        wmma::load_matrix_sync(fb, &vn16[kk][tj * 16], BV);
        wmma::mma_sync(acc, fa, fb, acc);
      }
      wmma::store_matrix_sync(&o32[ti * 16][tj * 16], acc, BV, wmma::mem_row_major);
    }
    // 3. S = e^(G_C) S + Vn^T KD  (16 tiles, 4 per warp), fp32 in place.
    const float decay = decay_in[static_cast<int64_t>(h) * chunks + ch];
    for (int tile = warp; tile < (BV / 16) * (D / 16); tile += kStateThreads / 32) {
      const int ti = tile / (D / 16), tj = tile % (D / 16);
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::load_matrix_sync(acc, &s32[ti * 16][tj * 16], D, wmma::mem_row_major);
      for (int e = 0; e < acc.num_elements; ++e) acc.x[e] *= decay;
      for (int kk = 0; kk < C; kk += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> fa;  // Vn^T
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fb;
        wmma::load_matrix_sync(fa, &vn16[kk][ti * 16], BV);
        wmma::load_matrix_sync(fb, &skd[kk][tj * 16], D);
        wmma::mma_sync(acc, fa, fb, acc);
      }
      wmma::store_matrix_sync(&s32[ti * 16][tj * 16], acc, D, wmma::mem_row_major);
    }
    __syncthreads();
    for (int e = tid; e < C * BV / 2; e += kStateThreads) {
      const int i = e / (BV / 2), c = (e % (BV / 2)) * 2;
      if (i < n) {
        const __nv_bfloat162 b = __floats2bfloat162_rn(o32[i][c], o32[i][c + 1]);
        *reinterpret_cast<__nv_bfloat162*>(out + (static_cast<int64_t>(t0 + i) * heads + h) * D + vb0 + c) = b;
      }
    }
    for (int e = tid; e < BV * D; e += kStateThreads) s16[e / D][e % D] = tobf(s32[e / D][e % D]);
  }
  __syncthreads();
  for (int e = tid; e < BV * D; e += kStateThreads) st[e] = s32[e / D][e % D];
}

}  // namespace

bool gdn_chunked_supported(int k_dim, int v_dim) { return k_dim == D && v_dim == D; }

size_t gdn_chunked_workspace_bytes(int tokens, int heads) {
  if (tokens <= 0) return 0;
  if (heads <= 0)
    throw std::invalid_argument("gdn_chunked_workspace_bytes: heads must be positive");
  const size_t chunks = (static_cast<size_t>(tokens) + C - 1) / C;
  const size_t blocks = static_cast<size_t>(heads) * chunks;
  return blocks * (C * (4 * D + C) * sizeof(__nv_bfloat16) + sizeof(float));
}

void gdn_chunked_fwd(const void* qkv, const void* a_raw, int64_t a_row_stride, const void* beta_raw,
                     int64_t beta_row_stride, const float* a_log, const float* dt_bias,
                     float* state, void* out, int tokens, int heads, int kv_ratio, int k_dim,
                     int v_dim, float scale, void* workspace, size_t workspace_bytes,
                     cudaStream_t stream) {
  if (tokens <= 0) return;
  if (!gdn_chunked_supported(k_dim, v_dim)) throw std::invalid_argument("gdn_chunked_fwd: K = V = 128 only");
  if (heads <= 0 || kv_ratio <= 0 || heads % kv_ratio != 0)
    throw std::invalid_argument("gdn_chunked_fwd: heads must be a positive multiple of kv_ratio");
  if (!workspace || workspace_bytes < gdn_chunked_workspace_bytes(tokens, heads) ||
      reinterpret_cast<uintptr_t>(workspace) % 16 != 0)
    throw std::invalid_argument("gdn_chunked_fwd: workspace is too small or not 16-byte aligned");
  const int chunks = 1 + (tokens - 1) / C;
  const size_t blocks = static_cast<size_t>(heads) * chunks;
  const size_t big = blocks * C * D, pn = blocks * C * C;
  auto* w = static_cast<__nv_bfloat16*>(workspace);
  auto* u = w + big;
  auto* qg = u + big;
  auto* kd = qg + big;
  auto* pp = kd + big;
  auto* decay = reinterpret_cast<float*>(pp + pn);
  static const bool opted = [] {  // once per process, thread-safe
    DGPP_CUDA_OK(cudaFuncSetAttribute(gdn_chunk_prep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(sizeof(PrepSmem))));
    DGPP_CUDA_OK(cudaFuncSetAttribute(gdn_chunk_state_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(kStateSmem + 16 * 128)));
    return true;
  }();
  (void)opted;
  gdn_chunk_prep_kernel<<<dim3(chunks, heads), kPrepThreads, sizeof(PrepSmem), stream>>>(
      static_cast<const uint16_t*>(qkv), static_cast<const uint16_t*>(a_raw), a_row_stride,
      static_cast<const uint16_t*>(beta_raw), beta_row_stride, a_log, dt_bias, tokens, heads, kv_ratio, scale, w, u,
      qg, kd, pp, decay);
  DGPP_CUDA_OK(cudaGetLastError());
  gdn_chunk_state_kernel<<<dim3(heads, D / BV), kStateThreads, kStateSmem + 16 * 128, stream>>>(
      w, u, qg, kd, pp, decay, state, static_cast<uint16_t*>(out), tokens, heads);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
