#include "kernels/glm_moe_launch.hpp"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/fp4_gemv.cuh"
#include "kernels/fp8_gemv.cuh"
#include "kernels/packq_gemv.cuh"

namespace dgpp {
namespace {

constexpr int kRouterSelectThreads = 32;
constexpr int kElemThreads = 256;

__device__ inline float sigmoidf_acc(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// The router dots (2026-09-02, reassociated). One WARP per (expert, token):
// each lane accumulates a strided quarter-kilobyte of the 4096-long dot in
// fp32 from 16-byte loads, then a shuffle tree sums the 32 partials. This
// is a plain bandwidth kernel (2.4 MB of gate rows at line rate ~10 us
// cold, ~3 us from L2) where the previous one reproduced the reference's
// SEQUENTIAL fp32 chain on a single thread to stay bit-identical to it —
// 4096 dependent FMAs, ~20 us whatever the memory did. The reassociation
// moves a logit by fp32 rounding (~1e-7 relative); the router test's
// oracle certifies any changed pick as a near-tie, and the transcript
// judge (scripts/fabric_xcript.py) does the same for a fabric run.
constexpr int kRouterDotWarps = 8;
constexpr int kRouterDotThreads = 32 * kRouterDotWarps;

// kMode selects the router's scoring rule (MoeRouterMode): kRouterSigmoid
// (GLM: sigmoid scores + the bias on the selection key), kRouterSoftmax
// (Qwen3.8-Flash-Next, 2026-09-09: the dots leave the bf16-rounded LOGIT
// in both rows — the reference's bf16 Linear output; no bias — the select
// turns the row into fp32 softmax probabilities, picks the top_k on the
// logits (ties to the lower id, the same rule) and normalizes the picked
// probabilities to weights ROUNDED TO BF16, the reference's cast to the
// hidden dtype before the multiply), and kRouterSqrtSoftplus
// (DeepSeek-V4.1-Flash, 2026-09-13, docs/deepseek_v41_flash_plan.md D3:
// scores sqrt(softplus(dot)) in fp32 with torch's threshold — x itself
// above 20 — otherwise the sigmoid mode's rule op for op). Each mode's
// code is its own constexpr branch; the others are untouched.
constexpr int kRouterSigmoid = 0;
constexpr int kRouterSoftmax = 1;
constexpr int kRouterSqrtSoftplus = 2;
__device__ __forceinline__ float sqrt_softplus_acc(float x) {
  const float sp = x > 20.f ? x : log1pf(expf(x));
  return sqrtf(sp);
}
template <int kMode>
__device__ __forceinline__ void router_dot(const uint16_t* __restrict__ hidden,
                                           const uint16_t* __restrict__ gate,
                                           const float* __restrict__ bias,
                                           float* __restrict__ scores,
                                           float* __restrict__ biased,
                                           int token, int e, int hidden_dim,
                                           int n_experts, int vector_loads,
                                           int lane);
template <int kMode>
__device__ __forceinline__ void router_select_warp(
    const float* __restrict__ scores, const float* __restrict__ biased,
    int32_t* __restrict__ ids, float* __restrict__ weights, int token,
    int n_experts, int top_k, float routed_scaling_factor, int norm_topk,
    float* s_scores, float* s_biased, int lane, bool staged = false);

// With `sel_ids` set the select is FUSED: the token's dots
// blocks take a ticket (`counters[token]`, zeroed once, reset by the last —
// replay-safe) and the last one runs router_select_warp on warp 0, so the
// separate select launch (5 us + a graph gap per MoE layer) disappears.
// The fence/ticket order is the mHC finish's: a block fences its scores
// before its ticket; the last block fences again before reading them.
template <int kMode>
__global__ void moe_router_dots_kernel(const uint16_t* __restrict__ hidden,
                                       const uint16_t* __restrict__ gate,
                                       const float* __restrict__ bias,
                                       float* __restrict__ scores,
                                       float* __restrict__ biased, int tokens,
                                       int hidden_dim, int n_experts,
                                       int vector_loads,
                                       int32_t* __restrict__ sel_ids,
                                       float* __restrict__ sel_weights,
                                       int top_k, float routed_scaling_factor,
                                       int norm_topk,
                                       int* __restrict__ counters) {
  extern __shared__ float sel_smem[];  // [2][n_experts] when fused
  __shared__ int s_last;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int e = blockIdx.x * kRouterDotWarps + warp;
  const int token = blockIdx.y;
  if (token >= tokens) return;
  if (e < n_experts)
    router_dot<kMode>(hidden, gate, bias, scores, biased, token, e,
                         hidden_dim, n_experts, vector_loads, lane);
  if (sel_ids == nullptr) return;
  __syncthreads();  // every warp's score is written
  if (threadIdx.x == 0) {
    __threadfence();
    const int ticket = atomicAdd(counters + token, 1);
    s_last = (ticket == static_cast<int>(gridDim.x) - 1) ? 1 : 0;
    if (s_last) counters[token] = 0;  // reset for the next launch
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  // Every warp of the last block stages the token's scores (2026-09-09:
  // warp 0's own strided __ldcg loop was 16 dependent L2 round trips per
  // lane at E = 512 — the router's 21.7 us was half select); the values
  // are the same bytes, so the select is unchanged.
  {
    const size_t row = static_cast<size_t>(token) * n_experts;
    float* s_scores = sel_smem;
    float* s_biased = sel_smem + n_experts;
    for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
      s_scores[e] = __ldcg(scores + row + e);
      s_biased[e] = __ldcg(biased + row + e);
    }
  }
  __syncthreads();
  if (warp != 0) return;
  router_select_warp<kMode>(scores, biased, sel_ids, sel_weights, token,
                               n_experts, top_k, routed_scaling_factor,
                               norm_topk, sel_smem, sel_smem + n_experts, lane,
                               /*staged=*/true);
}

// One warp's dot for (token, expert e): sigmoid(dot) and the biased copy
// (the softmax mode: the bf16-rounded logit in both rows).
template <int kMode>
__device__ __forceinline__ void router_dot(const uint16_t* __restrict__ hidden,
                                           const uint16_t* __restrict__ gate,
                                           const float* __restrict__ bias,
                                           float* __restrict__ scores,
                                           float* __restrict__ biased,
                                           int token, int e, int hidden_dim,
                                           int n_experts, int vector_loads,
                                           int lane) {
  const uint16_t* x = hidden + static_cast<size_t>(token) * hidden_dim;
  const uint16_t* w = gate + static_cast<size_t>(e) * hidden_dim;

  float dot = 0.f;
  if (vector_loads) {
    // 16-byte vectors: the launcher verified 16B alignment of both bases
    // and hidden_dim % 8 == 0 (row strides stay aligned).
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    const uint4* wv = reinterpret_cast<const uint4*>(w);
    const int vecs = hidden_dim / 8;
#pragma unroll 4
    for (int v = lane; v < vecs; v += 32) {
      const uint4 xq = xv[v];
      const uint4 wq = wv[v];
      const uint32_t xw[4] = {xq.x, xq.y, xq.z, xq.w};
      const uint32_t ww[4] = {wq.x, wq.y, wq.z, wq.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        dot = __fmaf_rn(bf16_bits_to_float(static_cast<uint16_t>(xw[i] & 0xFFFFu)),
                        bf16_bits_to_float(static_cast<uint16_t>(ww[i] & 0xFFFFu)),
                        dot);
        dot = __fmaf_rn(bf16_bits_to_float(static_cast<uint16_t>(xw[i] >> 16)),
                        bf16_bits_to_float(static_cast<uint16_t>(ww[i] >> 16)),
                        dot);
      }
    }
  } else {
    for (int k = lane; k < hidden_dim; k += 32)
      dot = __fmaf_rn(bf16_bits_to_float(x[k]), bf16_bits_to_float(w[k]), dot);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
  if (lane != 0) return;
  const size_t at = static_cast<size_t>(token) * n_experts + e;
  if constexpr (kMode == kRouterSoftmax) {
    const float l = bf16_bits_to_float(float_to_bf16_bits(dot));
    scores[at] = l;
    biased[at] = l;
  } else {
    const float s = kMode == kRouterSqrtSoftplus ? sqrt_softplus_acc(dot)
                                                 : 1.0f / (1.0f + expf(-dot));
    scores[at] = s;
    biased[at] = s + bias[e];
  }
}

// One warp per token: top-k over the biased scores (strict > keeps the
// LOWER expert id on ties), ids sorted ascending (the accumulation order),
// weights normalized with per-element division — the reference's
// elementwise ops, op for op. The selection scribbles -INFINITY into a
// shared-memory COPY so the exported biased row (near-tie certification
// reads every expert's true score) survives intact.
//
// The argmax is exact arithmetic (compares, no rounding), so spreading it
// over the warp is bit-identical to the serial scan it replaces: each lane
// scans its strided experts with the same "strict >, from -inf" rule, and
// the shuffle reduction keeps the greater value, the LOWER id on equal
// values — the serial scan's outcome by definition. The serial version
// ran 8 x 288 dependent smem loads on one thread: 16 us per layer.
__device__ __forceinline__ void warp_argmax_lowest_id(float& v, int& id) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_xor_sync(0xFFFFFFFFu, v, off);
    const int oid = __shfl_xor_sync(0xFFFFFFFFu, id, off);
    if (ov > v || (ov == v && oid < id)) {
      v = ov;
      id = oid;
    }
  }
}

// The select, as one warp's work: the token's scores/biased rows land in
// shared memory (the selection scribbles -INFINITY into the copy so the
// exported biased row survives), then top_k rounds of argmax.
template <int kMode>
__device__ __forceinline__ void router_select_warp(
    const float* __restrict__ scores, const float* __restrict__ biased,
    int32_t* __restrict__ ids, float* __restrict__ weights, int token,
    int n_experts, int top_k, float routed_scaling_factor, int norm_topk,
    float* s_scores, float* s_biased, int lane, bool staged) {
  if (!staged) {
    const size_t row = static_cast<size_t>(token) * n_experts;
    for (int e = lane; e < n_experts; e += 32) {
      s_scores[e] = __ldcg(scores + row + e);
      s_biased[e] = __ldcg(biased + row + e);
    }
  }
  __syncwarp();
  if constexpr (kMode == kRouterSoftmax) {
    // The fp32 softmax over the logits, the reference's op order: the row
    // max, the sum of exp(l - max) (each lane its strided share in order,
    // then the xor tree), then p = exp(l - max) / sum per expert. The
    // selection key stays the logit (s_biased; exp is monotone, so the
    // order is the probabilities' with no ties collapsed by rounding); the
    // picked WEIGHT is the probability.
    float m = -INFINITY;
    for (int e = lane; e < n_experts; e += 32) m = fmaxf(m, s_scores[e]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, off));
    float sum = 0.f;
    for (int e = lane; e < n_experts; e += 32)
      sum = __fadd_rn(sum, expf(s_scores[e] - m));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      sum += __shfl_xor_sync(0xFFFFFFFFu, sum, off);
    for (int e = lane; e < n_experts; e += 32)
      s_scores[e] = __fdiv_rn(expf(s_scores[e] - m), sum);
    __syncwarp();
  }

  int sel[16];
  float wsel[16];
  for (int r = 0; r < top_k; ++r) {
    // Lane-local scan, the serial rule verbatim; a lane with no candidate
    // holds (-inf, INT_MAX) and loses every comparison.
    int best = INT_MAX;
    float bv = -INFINITY;
    for (int e = lane; e < n_experts; e += 32) {
      if (s_biased[e] > bv) {
        bv = s_biased[e];
        best = e;
      }
    }
    warp_argmax_lowest_id(bv, best);
    sel[r] = best;
    wsel[r] = s_scores[best];
    __syncwarp();
    if (lane == 0) s_biased[best] = -INFINITY;
    __syncwarp();
  }
  if (lane != 0) return;
  // Ascending expert order (insertion sort; top_k <= 16).
  for (int i = 1; i < top_k; ++i) {
    const int id = sel[i];
    const float w = wsel[i];
    int j = i - 1;
    while (j >= 0 && sel[j] > id) {
      sel[j + 1] = sel[j];
      wsel[j + 1] = wsel[j];
      --j;
    }
    sel[j + 1] = id;
    wsel[j + 1] = w;
  }
  // Normalize with per-element division (the reference's elementwise op,
  // not reciprocal-multiply), then scale.
  float denom = 0.f;
  for (int i = 0; i < top_k; ++i) denom = __fadd_rn(denom, wsel[i]);
  denom = __fadd_rn(denom, 1e-20f);
  for (int i = 0; i < top_k; ++i) {
    float w = norm_topk ? __fdiv_rn(wsel[i], denom) * routed_scaling_factor
                        : wsel[i] * routed_scaling_factor;
    // The softmax mode's weights are bf16 values (the reference's cast),
    // carried exactly in the fp32 slot the chain multiplies by.
    if constexpr (kMode == kRouterSoftmax) w = bf16_bits_to_float(float_to_bf16_bits(w));
    ids[static_cast<size_t>(token) * top_k + i] = sel[i];
    weights[static_cast<size_t>(token) * top_k + i] = w;
  }
}

template <int kMode>
__global__ void moe_router_select_kernel(const float* __restrict__ scores,
                                         const float* __restrict__ biased,
                                         int32_t* __restrict__ ids,
                                         float* __restrict__ weights,
                                         int tokens, int n_experts, int top_k,
                                         float routed_scaling_factor,
                                         int norm_topk) {
  static_assert(kRouterSelectThreads == 32, "one warp per token");
  extern __shared__ float sel_smem[];  // [2][n_experts]: scores | biased
  const int token = blockIdx.x;
  if (token >= tokens) return;
  router_select_warp<kMode>(scores, biased, ids, weights, token, n_experts,
                               top_k, routed_scaling_factor, norm_topk,
                               sel_smem, sel_smem + n_experts, threadIdx.x);
}

__global__ void moe_swiglu_clamp_kernel(const uint16_t* __restrict__ gate,
                                        const uint16_t* __restrict__ up,
                                        uint16_t* __restrict__ out,
                                        int64_t n, float limit) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (i >= n) return;
  float g = bf16_bits_to_float(gate[i]);
  float u = bf16_bits_to_float(up[i]);
  if (g > limit) g = limit;  // gate: NO lower clamp (reference asymmetry)
  u = fminf(fmaxf(u, -limit), limit);
  const uint16_t t =
      float_to_bf16_bits(g * sigmoidf_acc(g));  // rounding 1 (silu)
  out[i] =
      float_to_bf16_bits(bf16_bits_to_float(t) * u);  // rounding 2 (product)
}

__global__ void moe_gather_rows_kernel(const uint16_t* __restrict__ src,
                                       const int32_t* __restrict__ rows,
                                       uint16_t* __restrict__ dst, int64_t n,
                                       int hidden) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (i >= n) return;
  const int64_t r = i / hidden;
  dst[i] = src[static_cast<int64_t>(rows[r]) * hidden + (i - r * hidden)];
}

// The fp32 accumulation chain (2026-09-02, expert slicing). Each expert's
// down projection arrives UNROUNDED (fp32 partial dots over this rank's
// slice of the intermediate dim); the chain is one fma per expert in
// ascending expert order, the shared expert last with weight 1, and the
// sum rounds to bf16 exactly once — when it leaves for the wire. The
// per-expert bf16 roundings the reference's index_add happened to perform
// are gone on purpose: fewer roundings, and a slice of an expert cannot
// reproduce the whole expert's rounding anyway. fmaf(w, y, acc) is the one
// op both paths (host segments, decode slots) issue, in the same order, so
// the decode path's pin against the host path stays bitwise.
__global__ void moe_accum_kernel(float* __restrict__ acc,
                                 const float* __restrict__ y,
                                 const int32_t* __restrict__ rows,
                                 const float* __restrict__ row_w, int64_t n,
                                 int hidden) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (i >= n) return;
  const int64_t r = i / hidden;
  float* dst = acc + static_cast<int64_t>(rows[r]) * hidden + (i - r * hidden);
  *dst = __fmaf_rn(row_w[r], y[i], *dst);
}

__global__ void moe_round_bf16_kernel(uint16_t* __restrict__ out,
                                      const float* __restrict__ acc,
                                      int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (i >= n) return;
  out[i] = float_to_bf16_bits(acc[i]);
}

// ---- decode-slot path (the sync-free MoE, 2026-09-01) --------------------
//
// One block per (8-row group, slot); the block resolves its slot's expert
// from the DEVICE route, stages the slot's activation row in smem, and each
// warp runs the fp8_gemv core on one weight row. The host-orchestrated path
// computes each expert's contribution through launch_scale_gemm_*, which
// dispatches m<=4 to the same core — glm_moe_test's bitwise gate pins the
// two (any change to one side's arithmetic breaks it; that gate is the
// twin-keeping mechanism). Every expert is local (each rank holds a slice
// of all of them), so there is no foreign-expert case.

// The slot's matrix for `which` (0 gate, 1 up, 2 down): routed slots read
// the ROUTE (device data — the whole point); the shared slot (j == top_k)
// uses the launch-arg matrices.
struct SlotMatrix {
  const uint8_t* payload;
  const float* scales;
  int n;
  int k;
  int rs;  // log2 scale block rows (7 = 128)
  int cs;  // log2 scale block cols
};

__device__ __forceinline__ SlotMatrix resolve_slot_matrix(
    int slot, int top_k, const int32_t* __restrict__ ids,
    const MoeExpertView* __restrict__ views, int which, int n_routed,
    int k_routed, int n_shared, int k_shared,
    const uint8_t* __restrict__ sh_payload,
    const float* __restrict__ sh_scales, int sh_rs, int sh_cs) {
  const int t = slot / (top_k + 1);
  const int j = slot - t * (top_k + 1);
  if (j < top_k) {
    const int e = ids[static_cast<size_t>(t) * top_k + j];
    const MoeExpertView& v = views[static_cast<size_t>(e) * 3 + which];
    return SlotMatrix{v.payload, v.scales, n_routed, k_routed, v.scale_shift_rows,
                      v.scale_shift_cols};
  }
  // The fp8 shared expert's own grid (7 = the checkpoint's 128; 5 = the
  // DeepSeek-V4.1 release's 32 x 32 blocks, 2026-09-13).
  return SlotMatrix{sh_payload, sh_scales, n_shared, k_shared, sh_rs, sh_cs};
}

// Slot EXECUTION order for a multi-token batch: slots sorted by expert
// id (the shared expert last, ties by slot — stable), so an expert two
// rows share is read from DRAM once and the second time from L2 (each fp8
// expert is ~6 MB per rank, the L2 24 MB) instead of DRAM — the only
// expert traffic a speculative verify row can share with its neighbour.
// Results are written by LOGICAL slot, so the accumulation (and every bit)
// is unchanged; only the dispatch order moves. One block, one thread per
// slot: 18-72 keys. (TRIED 2026-09-03 and reverted: ranking in the
// consuming kernels' prologue instead — two barriers and a key loop per
// block cost the 9216-block down launch +10 us, six times this kernel.)
//
// SHARED SLOTS EARLY (2026-10-01, `shared_early`): the shared expert's
// slots are compute over L2-resident weights (the walk's prefetch window
// holds them), the routed slots a DRAM stream. Sorted last, the shared
// slots ran as the launch's tail with the DRAM idle — 12 us per row of a
// verify pass behind the stream, and by then the stream had pushed the
// prefetched lines out. Interleaved with the first routed slots (r0 s0 r1
// s1 ... then the rest of the routed order) they run beside the stream
// while their lines are still resident: about half the blocks in flight
// stay DRAM-bound, which holds the stream within a few percent of its
// rate (moe_dup_bench). The order alone moves; every bit stays.
__global__ void moe_slot_order_kernel(const int32_t* __restrict__ ids,
                                      int32_t* __restrict__ order, int slots,
                                      int top_k, int n_experts, int shared_early) {
  extern __shared__ int32_t keys[];
  const int s = threadIdx.x;
  if (s < slots) {
    const int t = s / (top_k + 1);
    const int j = s - t * (top_k + 1);
    keys[s] = j < top_k ? ids[static_cast<size_t>(t) * top_k + j] : n_experts;
  }
  __syncthreads();
  if (s >= slots) return;
  int pos = 0;
  for (int o = 0; o < slots; ++o)
    pos += (keys[o] < keys[s]) || (keys[o] == keys[s] && o < s);
  if (shared_early) {
    // pos: the routed slots' ranks 0 .. routed-1, then the shared ones.
    const int tokens = slots / (top_k + 1);
    const int routed = tokens * top_k;
    pos = pos < routed ? pos + (pos < tokens ? pos : tokens) : 2 * (pos - routed) + 1;
  }
  order[pos] = s;
}

__device__ __forceinline__ int logical_slot(const int32_t* __restrict__ order) {
  return order ? order[blockIdx.y] : static_cast<int>(blockIdx.y);
}

// The down projection per slot: out[slot, :] = fp32 dot(down_row, act[slot]).
// Unrounded — the accumulation chain below owns the single rounding.
// (TRIED 2026-09-03 and reverted: the accumulation fused behind the last
// block per column chunk. The ticket's __syncthreads + fence per block and
// the last block's 18 dependent L2 loads on the kernel's tail cost +22 us
// per layer against the 1.7 us launch they replaced.)
// Rows per warp of the down projection: the sliced down's k
// is 512 bytes per row — one chunk per lane — so a warp per row had one
// load in flight; four rows per warp keep four (moe_slot_bench: the chain
// at two rows measured before/after).
// (TRIED 2026-09-09 and reverted: sixteen rows per warp for the Qwen
// TP=4 down's 160-byte rows — ten lanes' chunks per row, twenty-two idle.
// Four times the bytes in flight per warp changed nothing: T=1 25.3 vs
// 24.7 ms per step, MTP 30 vs 30 ms per pass on the fabric. The kernel at
// that k is instruction-bound on the per-row reduction, not on bytes; the
// lever is a lane remap — three rows per warp load — which changes the
// reduction order and so needs its own oracle gate.)
constexpr int kDownRowsPerWarp = 4;
// The narrow form (k <= 256 bytes, routed-only tables — Qwen at TP=4: k =
// 160, three rows per warp load): kDownNarrowGroups loads in flight per
// lane, 8 warps x 3 x 4 = 96 rows per block. fp8_gemv::block_rows_narrow
// keeps every row bitwise block_rows' (qwen_moe_test's decode gate).
constexpr int kDownNarrowGroups = 4;
constexpr int kDownNarrowMaxK = 256;

__global__ void moe_slot_down_narrow_kernel(
    const uint16_t* __restrict__ act, size_t act_stride,
    const int32_t* __restrict__ ids, const int32_t* __restrict__ order,
    const MoeExpertView* __restrict__ views, int n_routed, int k_routed,
    float* __restrict__ out, int out_stride, int slots, int top_k) {
  extern __shared__ __align__(16) uint16_t sx[];
  const int c = (k_routed + fp8_gemv::kChunkBytes - 1) / fp8_gemv::kChunkBytes;
  const int rows_per_block = fp8_gemv::kWarps * kDownNarrowGroups * (32 / c);
  const int n0 = blockIdx.x * rows_per_block;
  if (static_cast<int>(blockIdx.y) >= slots) return;
  const int slot = logical_slot(order);
  const SlotMatrix m =
      resolve_slot_matrix(slot, top_k, ids, views, /*which=*/2, n_routed, k_routed,
                          /*n_shared=*/0, /*k_shared=*/0, nullptr, nullptr, 7, 7);
  if (n0 >= m.n) return;
  fp8_gemv::stage_activations<1>(act + static_cast<size_t>(slot) * act_stride,
                                 act_stride, m.k, sx);
  __syncthreads();
  fp8_gemv::block_rows_narrow<kDownNarrowGroups>(
      m.payload, m.scales, sx, n0, m.n, m.k,
      out + static_cast<size_t>(slot) * out_stride, m.rs, m.cs);
}

__global__ void moe_slot_down_kernel(
    const uint16_t* __restrict__ act, size_t act_stride,
    const int32_t* __restrict__ ids, const int32_t* __restrict__ order,
    const MoeExpertView* __restrict__ views,
    int n_routed, int k_routed, int n_shared, int k_shared,
    const uint8_t* __restrict__ sh_payload, const float* __restrict__ sh_scales,
    float* __restrict__ out, int out_stride, int slots, int top_k, int sh_rs, int sh_cs) {
  extern __shared__ __align__(16) uint16_t sx[];
  const int n0 = blockIdx.x * (fp8_gemv::kWarps * kDownRowsPerWarp);
  if (static_cast<int>(blockIdx.y) >= slots) return;
  const int slot = logical_slot(order);
  const SlotMatrix m =
      resolve_slot_matrix(slot, top_k, ids, views, /*which=*/2, n_routed,
                          k_routed, n_shared, k_shared, sh_payload, sh_scales, sh_rs, sh_cs);
  if (n0 >= m.n) return;  // entirely outside (shared's shorter n)
  // The down projection consumes THIS SLOT's activation row.
  fp8_gemv::stage_activations<1>(act + static_cast<size_t>(slot) * act_stride,
                                 act_stride, m.k, sx);
  __syncthreads();
  fp8_gemv::block_rows_multi<1, kDownRowsPerWarp>(
      m.payload, m.scales, sx, n0, m.n, m.k,
      out + static_cast<size_t>(slot) * out_stride,
      static_cast<size_t>(out_stride), m.rs, m.cs);
}

// The gate GEMV, the up GEMV and the swiglu in one launch: a warp computes
// its row's gate and up dots from the same staged activation and applies
// moe_swiglu_clamp_kernel's math to the bf16-rounded dots in registers.
// Bit-identical to the three-launch chain (the dots are block_rows' dots,
// rounded to bf16 exactly where the intermediate buffers rounded them);
// what disappears is two launches and the gate/up round trip through
// memory.
// kPair: routed-only tables (Qwen) issue both rows' chunk
// batches before consuming either (fp8_gemv::row_dots_pair); GLM's tables
// keep the two row_dots calls in their own instantiation — one kernel
// carrying both paths sat at 96 registers and GLM's slot kernel lost
// occupancy (T=1 29.9 → 30.5–30.9, MTP 40.4 → 41.4 ms per step).
template <bool kPair>
__global__ void moe_slot_gate_up_swiglu_kernel(
    const uint16_t* __restrict__ x, size_t x_stride,
    const int32_t* __restrict__ ids, const int32_t* __restrict__ order,
    const MoeExpertView* __restrict__ views,
    int n_routed, int k_routed, int n_shared, int k_shared,
    const uint8_t* __restrict__ sh_gate_payload,
    const float* __restrict__ sh_gate_scales,
    const uint8_t* __restrict__ sh_up_payload,
    const float* __restrict__ sh_up_scales, uint16_t* __restrict__ act,
    int act_stride, int slots, int top_k, float limit, int sh_rs, int sh_cs) {
  extern __shared__ __align__(16) uint16_t sx[];
  const int n0 = blockIdx.x * fp8_gemv::kWarps;
  if (static_cast<int>(blockIdx.y) >= slots) return;
  const int slot = logical_slot(order);
  const SlotMatrix gate = resolve_slot_matrix(
      slot, top_k, ids, views, /*which=*/0, n_routed, k_routed, n_shared,
      k_shared, sh_gate_payload, sh_gate_scales, sh_rs, sh_cs);
  const SlotMatrix up = resolve_slot_matrix(
      slot, top_k, ids, views, /*which=*/1, n_routed, k_routed, n_shared,
      k_shared, sh_up_payload, sh_up_scales, sh_rs, sh_cs);
  if (n0 >= gate.n) return;
  const size_t token = static_cast<size_t>(slot / (top_k + 1));
  fp8_gemv::stage_activations<1>(x + token * x_stride, x_stride, gate.k, sx);
  __syncthreads();

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int row = n0 + warp;
  if (row >= gate.n) return;
  // gate and up share one geometry (the layer checks the triple).
  const int scale_cols = (gate.k + (1 << gate.cs) - 1) >> gate.cs;
  const size_t scale_row = static_cast<size_t>(row >> gate.rs) * scale_cols;
  float g_acc[1], u_acc[1];
  if constexpr (kPair) {
    // Both rows' batches in flight together, bitwise the two row_dots.
    fp8_gemv::row_dots_pair<1>(gate.payload + static_cast<size_t>(row) * gate.k,
                               gate.scales + scale_row,
                               up.payload + static_cast<size_t>(row) * up.k,
                               up.scales + scale_row, sx, gate.k, lane, g_acc, u_acc,
                               gate.cs);
  } else {
    fp8_gemv::row_dots<1>(gate.payload + static_cast<size_t>(row) * gate.k,
                          gate.scales + scale_row, sx, gate.k, lane, g_acc, gate.cs);
    fp8_gemv::row_dots<1>(up.payload + static_cast<size_t>(row) * up.k,
                          up.scales + scale_row, sx, up.k, lane, u_acc, up.cs);
  }
  if (lane != 0) return;
  // moe_swiglu_clamp_kernel, op for op, on the bf16-rounded dots.
  float g = bf16_bits_to_float(float_to_bf16_bits(g_acc[0]));
  float u = bf16_bits_to_float(float_to_bf16_bits(u_acc[0]));
  if (g > limit) g = limit;  // gate: NO lower clamp (reference asymmetry)
  u = fminf(fmaxf(u, -limit), limit);
  const uint16_t t = float_to_bf16_bits(g * sigmoidf_acc(g));  // rounding 1
  act[static_cast<size_t>(slot) * act_stride + row] =
      float_to_bf16_bits(bf16_bits_to_float(t) * u);  // rounding 2
}

// The ordered decode accumulation — moe_accum_kernel's chain, op for op:
// start at 0 (the host path's memset), fma each expert ascending (the
// router's id order), the shared expert last with weight 1 (fmaf(1, y, a)
// is exactly a + y), round to bf16 once.
__global__ void moe_slot_accum_kernel(
    uint16_t* __restrict__ out, const float* __restrict__ contrib,
    const float* __restrict__ weights, int64_t n, int hidden, int top_k) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (i >= n) return;
  const int64_t t = i / hidden;
  const int c = static_cast<int>(i - t * hidden);
  const int K = top_k;
  const int64_t base = t * (K + 1);  // routed slots + shared, per token

  float a = 0.f;
  for (int j = 0; j < K; ++j)
    a = __fmaf_rn(weights[static_cast<size_t>(t) * K + j],
                  contrib[static_cast<size_t>(base + j) * hidden + c], a);
  a = __fmaf_rn(1.0f, contrib[static_cast<size_t>(base + K) * hidden + c], a);
  out[t * hidden + c] = float_to_bf16_bits(a);
}

// The routed chain alone, left unrounded in fp32 (the Qwen decode: its
// BF16 shared expert continues the chain — models/qwen/moe_layer.cpp).
__global__ void moe_slot_accum_routed_f32_kernel(
    float* __restrict__ out, const float* __restrict__ contrib,
    const float* __restrict__ weights, int64_t n, int hidden, int top_k) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (i >= n) return;
  const int64_t t = i / hidden;
  const int c = static_cast<int>(i - t * hidden);
  const int K = top_k;
  const int64_t base = t * (K + 1);  // the slot layout keeps its shared slot
  float a = 0.f;
  for (int j = 0; j < K; ++j)
    a = __fmaf_rn(weights[static_cast<size_t>(t) * K + j],
                  contrib[static_cast<size_t>(base + j) * hidden + c], a);
  out[t * hidden + c] = a;
}

void check_router_args(const uint16_t* hidden, const uint16_t* gate,
                       const float* bias, int32_t* ids, float* weights,
                       const float* scores, const float* biased,
                       bool bias_required) {
  if (!hidden || !gate || (bias_required && !bias) || !ids || !weights ||
      !scores || !biased)
    throw std::invalid_argument("moe_router: null pointer");
  if (scores == biased)
    throw std::invalid_argument("moe_router: scores and biased must not alias");
}

bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

}  // namespace

namespace {

// The prefill router's dots: the warp-per-(token, expert) form
// above re-streams every gate row per token and every hidden row per
// expert from L2 (~4.9 GB per 2048-token layer, 1.7 ms). This form tiles
// 16 tokens x 16 experts per block and stages both operands' K-chunks in
// shared memory, so each row is read from L2 16x less; every (token,
// expert) dot is still one warp with the same per-lane element order (lane
// l takes 16-byte vectors v = l, l + 32, ... ascending; chunks are 32-vector
// aligned) and the same shuffle tree, so scores and biased are bitwise the
// warp form's (glm_moe_test pins it). Vector path only (the launcher keeps
// the warp form for unaligned geometry and for decode's fused select).
constexpr int kRouterTileTokens = 16;
constexpr int kRouterTileExperts = 16;
constexpr int kRouterChunkVecs = 64;  // 512 elements per staged chunk
constexpr int kRouterTileThreads = 256;

template <int kMode>
__global__ __launch_bounds__(kRouterTileThreads) void moe_router_dots_tiled_kernel(
    const uint16_t* __restrict__ hidden, const uint16_t* __restrict__ gate,
    const float* __restrict__ bias, float* __restrict__ scores,
    float* __restrict__ biased, int tokens, int hidden_dim, int n_experts) {
  __shared__ __align__(16) uint4 sH[kRouterTileTokens][kRouterChunkVecs];
  __shared__ __align__(16) uint4 sG[kRouterTileExperts][kRouterChunkVecs];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int e0 = blockIdx.x * kRouterTileExperts;
  const int t0 = blockIdx.y * kRouterTileTokens;
  const int vecs = hidden_dim / 8;
  // Warp w owns tokens 2w, 2w+1 of the tile against all 16 experts.
  float dot[2][kRouterTileExperts];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < kRouterTileExperts; ++j) dot[i][j] = 0.f;

  for (int v0 = 0; v0 < vecs; v0 += kRouterChunkVecs) {
    const int nv = min(kRouterChunkVecs, vecs - v0);
    __syncthreads();  // the previous chunk's readers are done
    for (int idx = threadIdx.x; idx < (kRouterTileTokens + kRouterTileExperts) * kRouterChunkVecs;
         idx += kRouterTileThreads) {
      const int row = idx / kRouterChunkVecs, vv = idx % kRouterChunkVecs;
      if (vv >= nv) continue;
      if (row < kRouterTileTokens) {
        const int t = t0 + row;
        if (t < tokens)
          sH[row][vv] = reinterpret_cast<const uint4*>(
              hidden + static_cast<size_t>(t) * hidden_dim)[v0 + vv];
      } else {
        const int e = e0 + row - kRouterTileTokens;
        if (e < n_experts)
          sG[row - kRouterTileTokens][vv] = reinterpret_cast<const uint4*>(
              gate + static_cast<size_t>(e) * hidden_dim)[v0 + vv];
      }
    }
    __syncthreads();
    // Lane l walks vectors v0 + l, v0 + l + 32 — its share of the row, in
    // ascending order, the warp form's exact sequence.
    for (int vv = lane; vv < nv; vv += 32) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint4 xq = sH[warp * 2 + i][vv];
        const uint32_t xw[4] = {xq.x, xq.y, xq.z, xq.w};
#pragma unroll
        for (int j = 0; j < kRouterTileExperts; ++j) {
          const uint4 wq = sG[j][vv];
          const uint32_t ww[4] = {wq.x, wq.y, wq.z, wq.w};
          float d = dot[i][j];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            d = __fmaf_rn(bf16_bits_to_float(static_cast<uint16_t>(xw[q] & 0xFFFFu)),
                          bf16_bits_to_float(static_cast<uint16_t>(ww[q] & 0xFFFFu)), d);
            d = __fmaf_rn(bf16_bits_to_float(static_cast<uint16_t>(xw[q] >> 16)),
                          bf16_bits_to_float(static_cast<uint16_t>(ww[q] >> 16)), d);
          }
          dot[i][j] = d;
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int t = t0 + warp * 2 + i;
#pragma unroll
    for (int j = 0; j < kRouterTileExperts; ++j) {
      float d = dot[i][j];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xFFFFFFFFu, d, off);
      const int e = e0 + j;
      if (lane == 0 && t < tokens && e < n_experts) {
        const size_t at = static_cast<size_t>(t) * n_experts + e;
        if constexpr (kMode == kRouterSoftmax) {
          const float l = bf16_bits_to_float(float_to_bf16_bits(d));
          scores[at] = l;
          biased[at] = l;
        } else {
          const float sc = kMode == kRouterSqrtSoftplus ? sqrt_softplus_acc(d)
                                                        : 1.0f / (1.0f + expf(-d));
          scores[at] = sc;
          biased[at] = sc + bias[e];
        }
      }
    }
  }
}

}  // namespace

namespace {

template <int kMode>
void launch_router_mode(const uint16_t* hidden, const uint16_t* gate,
                        const float* bias, int32_t* ids, float* weights,
                        float* scores, float* biased, const GlmMoeConfig& cfg,
                        int tokens, cudaStream_t stream, int* counters,
                        bool allow_tiled) {
  const int vector_loads =
      (cfg.hidden % 8 == 0 && aligned16(hidden) && aligned16(gate)) ? 1 : 0;
  if (counters == nullptr && allow_tiled && vector_loads &&
      tokens >= kRouterTileTokens) {
    const dim3 grid(
        static_cast<unsigned>((cfg.n_experts + kRouterTileExperts - 1) / kRouterTileExperts),
        static_cast<unsigned>((tokens + kRouterTileTokens - 1) / kRouterTileTokens));
    moe_router_dots_tiled_kernel<kMode><<<grid, kRouterTileThreads, 0, stream>>>(
        hidden, gate, bias, scores, biased, tokens, cfg.hidden, cfg.n_experts);
    DGPP_CUDA_OK(cudaGetLastError());
    const size_t sel_smem = 2 * static_cast<size_t>(cfg.n_experts) * sizeof(float);
    moe_router_select_kernel<kMode><<<tokens, kRouterSelectThreads, sel_smem, stream>>>(
        scores, biased, ids, weights, tokens, cfg.n_experts, cfg.top_k,
        cfg.routed_scaling_factor, cfg.norm_topk_prob ? 1 : 0);
    DGPP_CUDA_OK(cudaGetLastError());
    return;
  }
  const dim3 dots_grid(
      static_cast<unsigned>((cfg.n_experts + kRouterDotWarps - 1) /
                            kRouterDotWarps),
      static_cast<unsigned>(tokens));
  const size_t sel_smem = 2 * static_cast<size_t>(cfg.n_experts) * sizeof(float);
  if (counters != nullptr) {
    // Fused: the last dots block per token selects.
    moe_router_dots_kernel<kMode><<<dots_grid, kRouterDotThreads, sel_smem, stream>>>(
        hidden, gate, bias, scores, biased, tokens, cfg.hidden, cfg.n_experts,
        vector_loads, ids, weights, cfg.top_k, cfg.routed_scaling_factor,
        cfg.norm_topk_prob ? 1 : 0, counters);
    DGPP_CUDA_OK(cudaGetLastError());
    return;
  }
  moe_router_dots_kernel<kMode><<<dots_grid, kRouterDotThreads, 0, stream>>>(
      hidden, gate, bias, scores, biased, tokens, cfg.hidden, cfg.n_experts,
      vector_loads, nullptr, nullptr, 0, 0.f, 0, nullptr);
  DGPP_CUDA_OK(cudaGetLastError());
  moe_router_select_kernel<kMode><<<tokens, kRouterSelectThreads, sel_smem, stream>>>(
      scores, biased, ids, weights, tokens, cfg.n_experts, cfg.top_k,
      cfg.routed_scaling_factor, cfg.norm_topk_prob ? 1 : 0);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace

void launch_moe_router(const uint16_t* hidden, const uint16_t* gate,
                       const float* bias, int32_t* ids, float* weights,
                       float* scores, float* biased, const GlmMoeConfig& cfg,
                       int tokens, cudaStream_t stream, int* counters,
                       bool allow_tiled) {
  GlmMoeConfig::validate_config(cfg);
  if (tokens <= 0) return;
  const bool softmax = cfg.router_mode == MoeRouterMode::SoftmaxTopk;
  check_router_args(hidden, gate, bias, ids, weights, scores, biased,
                    /*bias_required=*/!softmax);
  if (cfg.n_experts > 65535 || tokens > 65535)
    throw std::invalid_argument("moe_router: grid dimension overflow");
  if (softmax)
    launch_router_mode<kRouterSoftmax>(hidden, gate, bias, ids, weights, scores, biased,
                                       cfg, tokens, stream, counters, allow_tiled);
  else if (cfg.router_mode == MoeRouterMode::SqrtSoftplusBias)
    launch_router_mode<kRouterSqrtSoftplus>(hidden, gate, bias, ids, weights, scores, biased,
                                            cfg, tokens, stream, counters, allow_tiled);
  else
    launch_router_mode<kRouterSigmoid>(hidden, gate, bias, ids, weights, scores, biased,
                                       cfg, tokens, stream, counters, allow_tiled);
}

namespace {
// Hash routing (DeepSeek-V4-Flash's first layers, 2026-10-01; the reference
// `Gate` with `hash`): a token's experts are tid2eid[token] (top_k ids,
// ascending) instead of the top-k of its scores; the weights are the
// scores at those ids normalized and scaled exactly as the select does
// (the sum in ascending-id order, + 1e-20, a division per element).
__global__ void moe_hash_routes_kernel(const int64_t* __restrict__ tokens,
                                       const int32_t* __restrict__ tid2eid,
                                       const float* __restrict__ scores,
                                       int32_t* __restrict__ ids,
                                       float* __restrict__ weights, int n_tokens,
                                       int vocab, int n_experts, int top_k,
                                       float routed_scaling_factor, int norm_topk) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n_tokens) return;
  int64_t tok = tokens[t];
  if (tok < 0 || tok >= vocab) tok = 0;  // a padding row: any table row (its output is never read)
  const int32_t* row = tid2eid + tok * top_k;
  float wsel[16];
  float denom = 0.f;
  for (int i = 0; i < top_k; ++i) {
    wsel[i] = scores[static_cast<size_t>(t) * n_experts + row[i]];
    denom = __fadd_rn(denom, wsel[i]);
  }
  denom = __fadd_rn(denom, 1e-20f);
  for (int i = 0; i < top_k; ++i) {
    ids[static_cast<size_t>(t) * top_k + i] = row[i];
    weights[static_cast<size_t>(t) * top_k + i] =
        norm_topk ? __fdiv_rn(wsel[i], denom) * routed_scaling_factor
                  : wsel[i] * routed_scaling_factor;
  }
}
}  // namespace

void launch_moe_hash_routes(const int64_t* tokens, const int32_t* tid2eid,
                            const float* scores, int32_t* ids, float* weights,
                            const GlmMoeConfig& cfg, int vocab, int n_tokens,
                            cudaStream_t stream) {
  if (n_tokens <= 0) return;
  if (!tokens || !tid2eid || !scores || !ids || !weights || vocab <= 0)
    throw std::invalid_argument("moe_hash_routes: null pointer");
  if (cfg.router_mode == MoeRouterMode::SoftmaxTopk)
    throw std::invalid_argument("moe_hash_routes: the softmax router leaves logits, not scores");
  moe_hash_routes_kernel<<<static_cast<unsigned>((n_tokens + 63) / 64), 64, 0, stream>>>(
      tokens, tid2eid, scores, ids, weights, n_tokens, vocab, cfg.n_experts, cfg.top_k,
      cfg.routed_scaling_factor, cfg.norm_topk_prob ? 1 : 0);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_swiglu_clamp(const uint16_t* gate, const uint16_t* up,
                             uint16_t* out, int64_t n, float limit,
                             cudaStream_t stream) {
  if (n <= 0) return;
  if (!gate || !up || !out)
    throw std::invalid_argument("moe_swiglu: null pointer");
  const int64_t blocks = (n + kElemThreads - 1) / kElemThreads;
  moe_swiglu_clamp_kernel<<<static_cast<int>(blocks), kElemThreads, 0,
                            stream>>>(gate, up, out, n, limit);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_gather_rows(const uint16_t* src, const int32_t* rows,
                            uint16_t* dst, int n_rows, int hidden,
                            cudaStream_t stream) {
  if (n_rows <= 0) return;
  if (!src || !rows || !dst)
    throw std::invalid_argument("moe_gather: null pointer");
  const int64_t n = static_cast<int64_t>(n_rows) * hidden;
  const int64_t blocks = (n + kElemThreads - 1) / kElemThreads;
  moe_gather_rows_kernel<<<static_cast<int>(blocks), kElemThreads, 0,
                           stream>>>(src, rows, dst, n, hidden);
  DGPP_CUDA_OK(cudaGetLastError());
}

namespace {


// The prefill's grouped GEMV: see glm_moe_launch.hpp. Every
// thread reaches every __syncthreads — the per-warp row bound is checked
// around block_rows instead of inside it (block_rows' own early return
// would strand a warp before the next group's barrier).
template <typename OutT>
__global__ void moe_grouped_gemv_kernel(const uint16_t* __restrict__ act,
                                        size_t act_stride,
                                        const MoeSegment* __restrict__ segs,
                                        const MoeExpertView* __restrict__ views,
                                        int which, OutT* __restrict__ out,
                                        size_t out_stride, int n, int k,
                                        int rows_per_block) {
  extern __shared__ __align__(16) uint16_t sx[];
  const MoeSegment seg = segs[blockIdx.y];
  // A launch may split its segments across blocks along z, rows_per_block
  // rows each (the weight slab re-read from L2 by every block) — the
  // shared expert's segment is every token, and one block walking 64
  // groups serially was that launch's latency floor. Routed launches keep
  // z = 1: their segments are short and uneven, and a z extent sized to
  // the longest one launches blocks that only exit (measured: +40 % on the
  // down launch at 256 tokens).
  const int z0 = static_cast<int>(blockIdx.z) * rows_per_block;
  if (z0 >= seg.rows) return;  // beyond this segment's rows (no barrier yet)
  const int z1 = min(seg.rows, z0 + rows_per_block);
  const MoeExpertView v = views[seg.expert * 3 + which];
  const int n0 = blockIdx.x * fp8_gemv::kWarps;
  const bool warp_live = n0 + static_cast<int>(threadIdx.x / 32) < n;
  for (int g = z0; g < z1; g += gemv::kMaxRows) {
    const int rows = min(gemv::kMaxRows, z1 - g);
    const uint16_t* x = act + static_cast<size_t>(seg.row0 + g) * act_stride;
    OutT* o = out + static_cast<size_t>(seg.row0 + g) * out_stride;
    if (g > z0) __syncthreads();  // the previous group is done reading sx
    switch (rows) {
      case 4:
        fp8_gemv::stage_activations<4>(x, act_stride, k, sx);
        __syncthreads();
        if (warp_live)
          fp8_gemv::block_rows<4, OutT>(v.payload, v.scales, sx, n0, n, k, o,
                                        out_stride, v.scale_shift_rows, v.scale_shift_cols);
        break;
      case 3:
        fp8_gemv::stage_activations<3>(x, act_stride, k, sx);
        __syncthreads();
        if (warp_live)
          fp8_gemv::block_rows<3, OutT>(v.payload, v.scales, sx, n0, n, k, o,
                                        out_stride, v.scale_shift_rows, v.scale_shift_cols);
        break;
      case 2:
        fp8_gemv::stage_activations<2>(x, act_stride, k, sx);
        __syncthreads();
        if (warp_live)
          fp8_gemv::block_rows<2, OutT>(v.payload, v.scales, sx, n0, n, k, o,
                                        out_stride, v.scale_shift_rows, v.scale_shift_cols);
        break;
      default:
        fp8_gemv::stage_activations<1>(x, act_stride, k, sx);
        __syncthreads();
        if (warp_live)
          fp8_gemv::block_rows<1, OutT>(v.payload, v.scales, sx, n0, n, k, o,
                                        out_stride, v.scale_shift_rows, v.scale_shift_cols);
        break;
    }
  }
}

template <typename OutT>
void launch_moe_grouped_gemv(const uint16_t* act, size_t act_stride,
                             const MoeSegment* segs, int n_segs, int max_rows,
                             int rows_per_block, const MoeExpertView* views,
                             int which, OutT* out, size_t out_stride, int n,
                             int k, cudaStream_t stream) {
  if (n_segs <= 0 || n <= 0) return;
  if (!act || !segs || !views || !out)
    throw std::invalid_argument("moe grouped gemv: null pointer");
  if (k <= 0 || (k % fp8_gemv::kChunkBytes) != 0)
    throw std::invalid_argument("moe grouped gemv: k must be a positive multiple of 16");
  if (!gemv::smem_fits(gemv::kMaxRows, k))
    throw std::invalid_argument("moe grouped gemv: k exceeds the smem budget");
  if (max_rows <= 0)
    throw std::invalid_argument("moe grouped gemv: max_rows must be positive");
  // No split: one z block per segment walking every row (the segments'
  // lengths need not be known on the host — device segmentation).
  const unsigned z_ext = rows_per_block > 0
                             ? static_cast<unsigned>((max_rows + rows_per_block - 1) /
                                                     rows_per_block)
                             : 1u;
  if (rows_per_block <= 0) rows_per_block = INT_MAX;
  const dim3 grid((n + fp8_gemv::kWarps - 1) / fp8_gemv::kWarps,
                  static_cast<unsigned>(n_segs), z_ext);
  moe_grouped_gemv_kernel<OutT>
      <<<grid, fp8_gemv::kThreads, gemv::smem_bytes(gemv::kMaxRows, k), stream>>>(
          act, act_stride, segs, views, which, out, out_stride, n, k,
          rows_per_block);
  DGPP_CUDA_OK(cudaGetLastError());
}

// ---- the prefill's grouped tensor-core GEMM -----------------
// One block per (64-column n-tile, segment): the block walks its segment in
// 128-row m-tiles (eight warps, sixteen rows each), and for every 64-deep
// k-stage stages the activation tile (bf16, 16-byte loads) and the weight
// tile (fp8 decoded, scaled and rounded to bf16 exactly as the dequant
// bridge does — the tile kernel's values, bit for bit) in shared memory,
// then runs mma.sync m16n8k16 bf16 with fp32 accumulation in ascending k16
// order — the same instruction sequence per output element as
// scale_gemm_kernel, so the outputs are bitwise that kernel's whatever the
// segment or tile geometry (glm_moe_test pins it). Against the grouped
// GEMV it replaces on the prefill path: that core re-reads and re-decodes
// the expert's weights once per four rows, so a 57-row segment paid for
// the weights fifteen times; here a segment up to 128 rows pays once.
namespace mma_tile {
constexpr int BM = 128;              // eight warps x m16
constexpr int BN = 64;               // eight n8 fragments per warp
constexpr int BK = 64;               // one scale column per stage (BK | 128)
constexpr int BK_PAD = BK + 8;       // u16 pad: 144-byte rows, conflict-free
constexpr int kThreads = 256;
static_assert(128 % BN == 0 && 128 % BK == 0, "one scale per stage");
static_assert(BK_PAD * 2 % 16 == 0, "16-byte aligned smem rows");
}  // namespace mma_tile

// `act_rows` (nullable): the activation row for segment row i is
// act_rows[i] (the gather folded into the tile load — the same values the
// gathered buffer would hold). `segs == nullptr` is the DENSE form: one
// segment `dense_seg` against `dense_view`, no tables in memory (the scale
// GEMM's large-m route).
template <typename OutT>
__global__ __launch_bounds__(mma_tile::kThreads) void moe_grouped_mma_kernel(
    const uint16_t* __restrict__ act, size_t act_stride, int act_vec,
    const int32_t* __restrict__ act_rows,
    const MoeSegment* __restrict__ segs, const MoeExpertView* __restrict__ views,
    int which, OutT* __restrict__ out, size_t out_stride, int n, int k,
    int rows_per_block, MoeSegment dense_seg, MoeExpertView dense_view) {
  using namespace mma_tile;
  __shared__ __align__(16) uint16_t sA[BM][BK_PAD];
  __shared__ __align__(16) uint16_t sB[BN][BK_PAD];
  const MoeSegment seg = segs != nullptr ? segs[blockIdx.y] : dense_seg;
  const int z0 = static_cast<int>(blockIdx.z) * rows_per_block;
  if (z0 >= seg.rows) return;  // beyond this segment's rows (no barrier yet)
  const int z1 = min(seg.rows, z0 + rows_per_block);
  const MoeExpertView v =
      segs != nullptr ? views[seg.expert * 3 + which] : dense_view;
  const int n0 = static_cast<int>(blockIdx.x) * BN;
  const int scale_cols = (k + 127) / 128;
  const int scale_row = n0 / 128;
  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int r = lane / 4;
  const int cc = (lane % 4) * 2;
  // The weight tile's load geometry: thread t decodes 16 consecutive k of
  // n-row t/4 (64 rows x 4 quads = 256 threads, one 16-byte load each).
  const int b_row = static_cast<int>(threadIdx.x) / 4;
  const int b_kq = (static_cast<int>(threadIdx.x) % 4) * 16;

  for (int m0 = z0; m0 < z1; m0 += BM) {
    const int m_rows = min(BM, z1 - m0);
    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

    for (int k0 = 0; k0 < k; k0 += BK) {
      const float s = v.scales[static_cast<size_t>(scale_row) * scale_cols + (k0 / 128)];
      // Weight tile: decode + scale + one BF16 round, zero outside [n, k).
      {
        const int gn = n0 + b_row, gk = k0 + b_kq;
        uint32_t packed[8];
        if (gn < n && gk + 16 <= k) {
          const uint4 raw = *reinterpret_cast<const uint4*>(
              v.payload + static_cast<size_t>(gn) * k + gk);
          const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const uint32_t w2 = (words[i / 2] >> ((i % 2) * 16)) & 0xffffu;
            const uint16_t lo = float_to_bf16_bits(
                fp8_e4m3_bits_to_float(static_cast<uint8_t>(w2 & 0xffu)) * s);
            const uint16_t hi = float_to_bf16_bits(
                fp8_e4m3_bits_to_float(static_cast<uint8_t>(w2 >> 8)) * s);
            packed[i] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
          }
        } else {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            uint16_t e[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int gkk = gk + 2 * i + h;
              e[h] = (gn < n && gkk < k)
                         ? float_to_bf16_bits(
                               fp8_e4m3_bits_to_float(
                                   v.payload[static_cast<size_t>(gn) * k + gkk]) * s)
                         : static_cast<uint16_t>(0);
            }
            packed[i] = static_cast<uint32_t>(e[0]) | (static_cast<uint32_t>(e[1]) << 16);
          }
        }
        uint4* dst = reinterpret_cast<uint4*>(&sB[b_row][b_kq]);
        dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
        dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
      }
      // Activation tile: rows of this m-tile, zero-filled past the segment
      // and past k (16-byte loads when the rows are 16-byte aligned).
      for (int i = static_cast<int>(threadIdx.x); i < BM * (BK / 8); i += kThreads) {
        const int mm = i / (BK / 8), kq = (i % (BK / 8)) * 8;
        const int gk = k0 + kq;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (mm < m_rows) {
          const int srow = seg.row0 + m0 + mm;
          const uint16_t* row =
              act + static_cast<size_t>(act_rows != nullptr ? act_rows[srow] : srow) *
                        act_stride;
          if (act_vec != 0 && gk + 8 <= k) {
            val = *reinterpret_cast<const uint4*>(row + gk);
          } else {
            uint16_t e[8];
#pragma unroll
            for (int h = 0; h < 8; ++h) e[h] = gk + h < k ? row[gk + h] : 0;
            val = make_uint4(e[0] | (e[1] << 16), e[2] | (e[3] << 16),
                             e[4] | (e[5] << 16), e[6] | (e[7] << 16));
          }
        }
        *reinterpret_cast<uint4*>(&sA[mm][kq]) = val;
      }
      __syncthreads();

#pragma unroll
      for (int kk = 0; kk < BK; kk += 16) {
        const int ar = warp * 16 + r;
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sA[ar][kk + cc]);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sA[ar + 8][kk + cc]);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sA[ar][kk + cc + 8]);
        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&sA[ar + 8][kk + cc + 8]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int bn = j * 8 + r;
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sB[bn][kk + cc]);
          const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sB[bn][kk + cc + 8]);
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
              : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
              : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
      }
      __syncthreads();  // tile reads done before the next stage overwrites
    }

    // Epilogue: per warp a [16 x 64] slice; the thread holds rows r, r+8
    // and columns cc, cc+1 of each n8 fragment.
    const int row_lo = warp * 16 + r;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int gn = n0 + j * 8 + cc;
      auto st = [&](int row_off, int col_off, float val) {
        const int mm = row_lo + row_off;
        if (mm < m_rows && gn + col_off < n)
          fp8_gemv::store_dot(
              out + static_cast<size_t>(seg.row0 + m0 + mm) * out_stride + gn + col_off,
              val);
      };
      st(0, 0, acc[j][0]);
      st(0, 1, acc[j][1]);
      st(8, 0, acc[j][2]);
      st(8, 1, acc[j][3]);
    }
  }
}

bool g_fp8_ldm_enabled = true;  // moe_set_fp8_ldm(): the bench's A/B against the reference
constexpr int kFp8LdmBK = 32;   // fp8_ldm::BK (defined below; static_assert'ed there)
template <typename OutT>
void launch_moe_grouped_mma_fp8_ldm(const uint16_t* act, size_t act_stride,
                                    const int32_t* act_rows, const MoeSegment* segs,
                                    int n_segs, int max_rows, const MoeExpertView* views,
                                    int which, OutT* out, size_t out_stride, int n, int k,
                                    cudaStream_t stream);

template <typename OutT>
void launch_moe_grouped_mma(const uint16_t* act, size_t act_stride,
                            const int32_t* act_rows,
                            const MoeSegment* segs, int n_segs, int max_rows,
                            int rows_per_block, const MoeExpertView* views,
                            int which, OutT* out, size_t out_stride, int n,
                            int k, cudaStream_t stream) {
  using namespace mma_tile;
  if (n_segs <= 0 || n <= 0) return;
  if (!act || !segs || !views || !out)
    throw std::invalid_argument("moe grouped mma: null pointer");
  if (k <= 0 || (k % 16) != 0)
    throw std::invalid_argument("moe grouped mma: k must be a positive multiple of 16");
  if (max_rows <= 0)
    throw std::invalid_argument("moe grouped mma: max_rows must be positive");
  if (rows_per_block > 0 && (rows_per_block % BM) != 0)
    throw std::invalid_argument("moe grouped mma: rows_per_block must be a multiple of 128");
  const unsigned z_ext = rows_per_block > 0
                             ? static_cast<unsigned>((max_rows + rows_per_block - 1) /
                                                     rows_per_block)
                             : 1u;
  if (rows_per_block <= 0) rows_per_block = INT_MAX;
  // The ldmatrix fp8 kernel (2026-09-08 evening) for every k that is a
  // multiple of 32 (the production 4096 and 512; its stages and 16-byte
  // code copies need it), bitwise the reference tile kernel, which keeps
  // the other widths (and stays the dense form and the pin).
  if ((k % kFp8LdmBK) == 0 && g_fp8_ldm_enabled) {
    launch_moe_grouped_mma_fp8_ldm<OutT>(act, act_stride, act_rows, segs, n_segs, max_rows,
                                         views, which, out, out_stride, n, k, stream);
    return;
  }
  // 16-byte activation loads need 16-byte rows: the base and the stride.
  const int act_vec =
      (reinterpret_cast<uintptr_t>(act) % 16 == 0 && (act_stride % 8) == 0) ? 1 : 0;
  const dim3 grid((n + BN - 1) / BN, static_cast<unsigned>(n_segs), z_ext);
  moe_grouped_mma_kernel<OutT><<<grid, kThreads, 0, stream>>>(
      act, act_stride, act_vec, act_rows, segs, views, which, out, out_stride, n,
      k, rows_per_block, MoeSegment{}, MoeExpertView{});
  DGPP_CUDA_OK(cudaGetLastError());
}

inline size_t out_stride_of(int n) { return static_cast<size_t>(n); }

// The dense form: out[m, n] = act[m, k] x W[n, k]^T on the same kernel, one
// m-tile per z block (bitwise the tile kernel, like the grouped form).
template <typename OutT>
void launch_dense_mma(const uint16_t* act, size_t act_stride,
                      const uint8_t* payload, const float* scales, OutT* out,
                      int m, int n, int k, cudaStream_t stream,
                      size_t out_stride = 0) {
  using namespace mma_tile;
  if (m <= 0 || n <= 0) return;
  if (!act || !payload || !scales || !out)
    throw std::invalid_argument("dense mma: null pointer");
  if (k <= 0 || (k % 16) != 0)
    throw std::invalid_argument("dense mma: k must be a positive multiple of 16");
  if (out_stride == 0) out_stride = out_stride_of(n);
  if (out_stride < static_cast<size_t>(n))
    throw std::invalid_argument("dense mma: output row stride narrower than n");
  const int act_vec =
      (reinterpret_cast<uintptr_t>(act) % 16 == 0 && (act_stride % 8) == 0) ? 1 : 0;
  const dim3 grid((n + BN - 1) / BN, 1u, static_cast<unsigned>((m + BM - 1) / BM));
  moe_grouped_mma_kernel<OutT><<<grid, kThreads, 0, stream>>>(
      act, act_stride, act_vec, nullptr, nullptr, nullptr, 0, out, out_stride,
      n, k, BM, MoeSegment{0, m, 0}, MoeExpertView{payload, scales});
  DGPP_CUDA_OK(cudaGetLastError());
}

constexpr int kAccumMaxTopK = 16;
constexpr int kSegmentMaxExperts = 1024;

// Device segmentation in three launches (2026-09-05; the single-block form
// before it had one thread per expert walking every id: 0.84 ms per layer
// at 2048 tokens). (1) counts: one block per expert, a strided count and a
// block reduction, into segs[e].rows; (2) scan: one block, the exclusive
// prefix into segs[e].row0, the shared segment and its identity rows;
// (3) place: one block per expert walks the ids in blockDim-sized chunks
// with a block exclusive scan over the match flags, so its expert's
// (token, slot) pairs land in (token, slot) order — exactly the host
// path's stable placement, exactly the single-block kernel's output.
constexpr int kSegmentThreads = 256;

__device__ __forceinline__ int block_exclusive_scan_int(int v, int* s_warp,
                                                        int* total) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int x = v;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, off);
    if (lane >= off) x += y;
  }
  if (lane == 31) s_warp[warp] = x;
  __syncthreads();
  if (warp == 0) {
    int w = lane < (kSegmentThreads / 32) ? s_warp[lane] : 0;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, w, off);
      if (lane >= off) w += y;
    }
    if (lane < (kSegmentThreads / 32)) s_warp[lane] = w;  // inclusive per warp
  }
  __syncthreads();
  const int warp_base = warp == 0 ? 0 : s_warp[warp - 1];
  *total = s_warp[kSegmentThreads / 32 - 1];
  const int excl = x - v + warp_base;
  __syncthreads();  // s_warp reusable by the next call
  return excl;
}

__global__ void moe_segment_count_kernel(const int32_t* __restrict__ ids, int tk,
                                         MoeSegment* __restrict__ segs) {
  __shared__ int s_warp[kSegmentThreads / 32];
  const int e = blockIdx.x;
  int c = 0;
  for (int i = threadIdx.x; i < tk; i += kSegmentThreads) c += ids[i] == e;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) c += __shfl_xor_sync(0xffffffffu, c, off);
  if (threadIdx.x % 32 == 0) s_warp[threadIdx.x / 32] = c;
  __syncthreads();
  if (threadIdx.x == 0) {
    int total = 0;
    for (int w = 0; w < kSegmentThreads / 32; ++w) total += s_warp[w];
    segs[e] = MoeSegment{0, total, e};
  }
}

__global__ void moe_segment_scan_kernel(int tokens, int K, int E,
                                        int32_t* __restrict__ rows,
                                        MoeSegment* __restrict__ segs) {
  const int tk = tokens * K;
  if (threadIdx.x == 0) {
    int acc = 0;
    for (int e = 0; e < E; ++e) {
      segs[e].row0 = acc;
      acc += segs[e].rows;
    }
    segs[E] = MoeSegment{tk, tokens, E};
  }
  // The shared expert: every token once more, after the routed rows.
  for (int t = threadIdx.x; t < tokens; t += blockDim.x) rows[tk + t] = t;
}

__global__ void moe_segment_place_kernel(const int32_t* __restrict__ ids, int tk,
                                         int K, int32_t* __restrict__ rows,
                                         int32_t* __restrict__ slot_row,
                                         const MoeSegment* __restrict__ segs) {
  __shared__ int s_warp[kSegmentThreads / 32];
  const int e = blockIdx.x;
  const MoeSegment seg = segs[e];
  if (seg.rows == 0) return;
  int running = seg.row0;
  for (int base = 0; base < tk; base += kSegmentThreads) {
    const int i = base + threadIdx.x;
    const int flag = (i < tk && ids[i] == e) ? 1 : 0;
    int total = 0;
    const int excl = block_exclusive_scan_int(flag, s_warp, &total);
    if (flag) {
      const int pos = running + excl;
      rows[pos] = i / K;
      slot_row[i] = pos;
    }
    running += total;
  }
}

// The down rows' element: fp32, or bf16 (the W4A4 prefill chain's
// DGPP_MOE_W4A4 down output, halving its write and this read; 2026-09-23).
__device__ __forceinline__ float down_value(float v) { return v; }
__device__ __forceinline__ float down_value(uint16_t v) { return bf16_bits_to_float(v); }

template <typename OutT, typename DownT = float>
__global__ void moe_accum_ordered_kernel(OutT* __restrict__ out,
                                         const DownT* __restrict__ down,
                                         size_t down_stride,
                                         const int32_t* __restrict__ slot_row,
                                         const int32_t* __restrict__ slot_ids,
                                         const float* __restrict__ slot_w,
                                         int shared_row0, int tokens, int K,
                                         int hidden) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (i >= static_cast<int64_t>(tokens) * hidden) return;
  const int t = static_cast<int>(i / hidden);
  const int h = static_cast<int>(i - static_cast<int64_t>(t) * hidden);
  // Ascending expert id: the chain's order (an insertion sort of K <= 16).
  int order[kAccumMaxTopK];
  for (int j = 0; j < K; ++j) order[j] = j;
  for (int j = 1; j < K; ++j) {
    const int cur = order[j];
    const int key = slot_ids[t * K + cur];
    int p = j - 1;
    while (p >= 0 && slot_ids[t * K + order[p]] > key) {
      order[p + 1] = order[p];
      --p;
    }
    order[p + 1] = cur;
  }
  float acc = 0.f;
  for (int j = 0; j < K; ++j) {
    const int s = t * K + order[j];
    acc = __fmaf_rn(slot_w[s],
                    down_value(down[static_cast<size_t>(slot_row[s]) * down_stride + h]), acc);
  }
  // shared_row0 < 0: no shared expert in this chain (the Qwen MoE adds its
  // BF16 shared branch after the routed sum, with its own weight).
  if (shared_row0 >= 0)
    acc = __fmaf_rn(1.0f,
                    down_value(down[static_cast<size_t>(shared_row0 + t) * down_stride + h]), acc);
  if constexpr (std::is_same<OutT, float>::value)
    out[i] = acc;
  else
    out[i] = float_to_bf16_bits(acc);
}

}  // namespace

void launch_moe_grouped_mma_bf16(const uint16_t* act, size_t act_stride,
                                 const MoeSegment* segs, int n_segs, int max_rows,
                                 int rows_per_block, const MoeExpertView* views,
                                 int which, uint16_t* out, size_t out_stride, int n,
                                 int k, cudaStream_t stream, const int32_t* act_rows) {
  launch_moe_grouped_mma<uint16_t>(act, act_stride, act_rows, segs, n_segs, max_rows,
                                   rows_per_block, views, which, out, out_stride,
                                   n, k, stream);
}

void launch_moe_grouped_mma_f32(const uint16_t* act, size_t act_stride,
                                const MoeSegment* segs, int n_segs, int max_rows,
                                int rows_per_block, const MoeExpertView* views,
                                int which, float* out, size_t out_stride, int n,
                                int k, cudaStream_t stream, const int32_t* act_rows) {
  launch_moe_grouped_mma<float>(act, act_stride, act_rows, segs, n_segs, max_rows,
                                rows_per_block, views, which, out, out_stride, n,
                                k, stream);
}

void launch_dense_mma_bf16(const uint16_t* act, size_t act_stride,
                           const uint8_t* payload, const float* scales,
                           uint16_t* out, int m, int n, int k, cudaStream_t stream,
                           size_t out_stride) {
  launch_dense_mma<uint16_t>(act, act_stride, payload, scales, out, m, n, k, stream,
                             out_stride);
}

void launch_dense_mma_f32(const uint16_t* act, size_t act_stride,
                          const uint8_t* payload, const float* scales, float* out,
                          int m, int n, int k, cudaStream_t stream,
                          size_t out_stride) {
  launch_dense_mma<float>(act, act_stride, payload, scales, out, m, n, k, stream,
                          out_stride);
}

void launch_moe_grouped_gemv_bf16(const uint16_t* act, size_t act_stride,
                                  const MoeSegment* segs, int n_segs,
                                  int max_rows, int rows_per_block,
                                  const MoeExpertView* views, int which,
                                  uint16_t* out, size_t out_stride, int n, int k,
                                  cudaStream_t stream) {
  launch_moe_grouped_gemv<uint16_t>(act, act_stride, segs, n_segs, max_rows,
                                    rows_per_block, views, which, out, out_stride,
                                    n, k, stream);
}

void launch_moe_grouped_gemv_f32(const uint16_t* act, size_t act_stride,
                                 const MoeSegment* segs, int n_segs, int max_rows,
                                 int rows_per_block, const MoeExpertView* views,
                                 int which, float* out, size_t out_stride, int n,
                                 int k, cudaStream_t stream) {
  launch_moe_grouped_gemv<float>(act, act_stride, segs, n_segs, max_rows,
                                 rows_per_block, views, which, out, out_stride, n,
                                 k, stream);
}

namespace {
// The compact tile list (glm_moe_launch.hpp): one block scans the
// segments' tile counts and writes every segment's tiles at its offset.
__global__ void moe_tile_list_kernel(const MoeSegment* __restrict__ segs, int n_segs,
                                     int tile_rows, MoeTile* __restrict__ tiles,
                                     int32_t* __restrict__ count) {
  __shared__ int s_warp[kSegmentThreads / 32];
  int running = 0;
  for (int base = 0; base < n_segs; base += kSegmentThreads) {
    const int e = base + threadIdx.x;
    const int rows = e < n_segs ? segs[e].rows : 0;
    const int mine = (rows + tile_rows - 1) / tile_rows;
    int total = 0;
    const int excl = block_exclusive_scan_int(mine, s_warp, &total);
    for (int j = 0; j < mine; ++j) tiles[running + excl + j] = MoeTile{e, j * tile_rows};
    running += total;
  }
  if (threadIdx.x == 0) *count = running;
}
}  // namespace

void launch_moe_tile_list(const MoeSegment* segs, int n_segs, int tile_rows,
                          MoeTile* tiles, int32_t* count, cudaStream_t stream) {
  if (!segs || !tiles || !count) throw std::invalid_argument("moe tile list: null pointer");
  if (n_segs < 1 || n_segs > 65535 || tile_rows < 1)
    throw std::invalid_argument("moe tile list: n_segs or tile_rows out of range");
  moe_tile_list_kernel<<<1, kSegmentThreads, 0, stream>>>(segs, n_segs, tile_rows, tiles, count);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_segment(const int32_t* ids, int tokens, int top_k,
                        int n_experts, int32_t* rows, int32_t* slot_row,
                        MoeSegment* segs, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (!ids || !rows || !slot_row || !segs)
    throw std::invalid_argument("moe segment: null pointer");
  if (top_k < 1 || top_k > kAccumMaxTopK || n_experts < 1 ||
      n_experts > kSegmentMaxExperts)
    throw std::invalid_argument("moe segment: top_k or n_experts out of range");
  const int tk = tokens * top_k;
  moe_segment_count_kernel<<<n_experts, kSegmentThreads, 0, stream>>>(ids, tk, segs);
  DGPP_CUDA_OK(cudaGetLastError());
  moe_segment_scan_kernel<<<1, 256, 0, stream>>>(tokens, top_k, n_experts, rows, segs);
  DGPP_CUDA_OK(cudaGetLastError());
  moe_segment_place_kernel<<<n_experts, kSegmentThreads, 0, stream>>>(
      ids, tk, top_k, rows, slot_row, segs);
  DGPP_CUDA_OK(cudaGetLastError());
}

namespace {

template <class OutT, class DownT = float>
void launch_accum_ordered_t(OutT* out, const DownT* down, size_t down_stride,
                            const int32_t* slot_row, const int32_t* slot_ids,
                            const float* slot_w, int shared_row0, int tokens,
                            int top_k, int hidden, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (!out || !down || !slot_row || !slot_ids || !slot_w)
    throw std::invalid_argument("moe accum ordered: null pointer");
  if (top_k < 1 || top_k > kAccumMaxTopK)
    throw std::invalid_argument("moe accum ordered: top_k outside [1, 16]");
  const int64_t n = static_cast<int64_t>(tokens) * hidden;
  const int64_t blocks = (n + kElemThreads - 1) / kElemThreads;
  moe_accum_ordered_kernel<OutT, DownT><<<static_cast<int>(blocks), kElemThreads, 0, stream>>>(
      out, down, down_stride, slot_row, slot_ids, slot_w, shared_row0, tokens,
      top_k, hidden);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace

void launch_moe_accum_ordered(uint16_t* out, const float* down,
                              size_t down_stride, const int32_t* slot_row,
                              const int32_t* slot_ids, const float* slot_w,
                              int shared_row0, int tokens, int top_k,
                              int hidden, cudaStream_t stream) {
  launch_accum_ordered_t<uint16_t>(out, down, down_stride, slot_row, slot_ids,
                                   slot_w, shared_row0, tokens, top_k, hidden,
                                   stream);
}

void launch_moe_accum_ordered_f32(float* out, const float* down,
                                  size_t down_stride, const int32_t* slot_row,
                                  const int32_t* slot_ids, const float* slot_w,
                                  int shared_row0, int tokens, int top_k,
                                  int hidden, cudaStream_t stream) {
  launch_accum_ordered_t<float>(out, down, down_stride, slot_row, slot_ids,
                                slot_w, shared_row0, tokens, top_k, hidden,
                                stream);
}

void launch_moe_accum_ordered_bf16down(uint16_t* out, const uint16_t* down, size_t down_stride,
                                      const int32_t* slot_row, const int32_t* slot_ids,
                                      const float* slot_w, int tokens, int top_k, int hidden,
                                      cudaStream_t stream) {
  launch_accum_ordered_t<uint16_t, uint16_t>(out, down, down_stride, slot_row, slot_ids, slot_w, -1,
                                             tokens, top_k, hidden, stream);
}

void launch_moe_accum_ordered_f32_bf16down(float* out, const uint16_t* down, size_t down_stride,
                                           const int32_t* slot_row, const int32_t* slot_ids,
                                           const float* slot_w, int tokens, int top_k, int hidden,
                                           cudaStream_t stream) {
  launch_accum_ordered_t<float, uint16_t>(out, down, down_stride, slot_row, slot_ids, slot_w, -1, tokens,
                                          top_k, hidden, stream);
}

void launch_moe_accum(float* acc, const float* y, const int32_t* rows,
                      const float* row_weights, int n_rows, int hidden,
                      cudaStream_t stream) {
  if (n_rows <= 0) return;
  if (!acc || !y || !rows || !row_weights)
    throw std::invalid_argument("moe_accum: null pointer");
  const int64_t n = static_cast<int64_t>(n_rows) * hidden;
  const int64_t blocks = (n + kElemThreads - 1) / kElemThreads;
  moe_accum_kernel<<<static_cast<int>(blocks), kElemThreads, 0, stream>>>(
      acc, y, rows, row_weights, n, hidden);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_round_bf16(uint16_t* out, const float* acc, int64_t n,
                           cudaStream_t stream) {
  if (n <= 0) return;
  if (!out || !acc) throw std::invalid_argument("moe_round: null pointer");
  const int64_t blocks = (n + kElemThreads - 1) / kElemThreads;
  moe_round_bf16_kernel<<<static_cast<int>(blocks), kElemThreads, 0, stream>>>(
      out, acc, n);
  DGPP_CUDA_OK(cudaGetLastError());
}

namespace {

void check_slot_args(const void* x, const int32_t* ids,
                     const MoeExpertView* views, const void* out,
                     int n_routed, int k_routed, int n_shared, int k_shared,
                     const char* who) {
  if (!x || !ids || !views || !out)
    throw std::invalid_argument(std::string(who) + ": null pointer");
  // n_shared == 0 (2026-09-09, the Qwen decode): no shared slot — the
  // kernels' shared-slot blocks find n == 0 and return; the routed slots
  // are untouched. The shared payload checks below are skipped then.
  if (n_routed <= 0 || k_routed <= 0 || n_shared < 0 || k_shared < 0 ||
      (n_shared > 0) != (k_shared > 0))
    throw std::invalid_argument(std::string(who) + ": degenerate dims");
  // The GEMV core's contract (fp8_gemv.cuh): k a multiple of 16 so every
  // 16-byte chunk lies inside one scale block. The routed payloads live in
  // the device table and ride the loader's 256-byte alignment contract;
  // the shared payloads are checked by the callers below.
  if (k_routed % fp8_gemv::kChunkBytes != 0 || !gemv::smem_fits(1, k_routed))
    throw std::invalid_argument(
        std::string(who) + ": k must be a multiple of 16 (16B-aligned "
                           "payloads)");
}

}  // namespace

void launch_moe_slot_order(const int32_t* ids, int32_t* order, int slots,
                           int top_k, int n_experts, cudaStream_t stream, bool shared_early) {
  if (slots <= 0) return;
  if (!ids || !order) throw std::invalid_argument("moe_slot_order: null");
  if (slots > 1024)
    throw std::invalid_argument("moe_slot_order: more slots than one block");
  moe_slot_order_kernel<<<1, static_cast<unsigned>(slots),
                          static_cast<size_t>(slots) * sizeof(int32_t),
                          stream>>>(ids, order, slots, top_k, n_experts, shared_early ? 1 : 0);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_slot_down(const uint16_t* act, size_t act_stride,
                          const int32_t* ids, const int32_t* order,
                          const MoeExpertView* views,
                          int n_routed, int k_routed, int n_shared,
                          int k_shared, const uint8_t* sh_payload,
                          const float* sh_scales, float* out, int out_stride,
                          int slots, int top_k, cudaStream_t stream, int sh_rs, int sh_cs) {
  if (slots <= 0) return;
  check_slot_args(act, ids, views, out, n_routed, k_routed, n_shared, k_shared,
                  "moe_slot_down");
  if (n_shared > 0 &&
      (!sh_payload || !sh_scales || !fp8_gemv::shape_ok(sh_payload, 1, k_shared)))
    throw std::invalid_argument(
        "moe_slot_down: shared payload must be 16B-aligned with k % 16 == 0");
  if (out_stride < n_routed || out_stride < n_shared)
    throw std::invalid_argument("moe_slot_down: out_stride below n");
  const int max_n = n_routed > n_shared ? n_routed : n_shared;
  const int max_k = k_routed > k_shared ? k_routed : k_shared;
  if (n_shared == 0 && k_routed <= kDownNarrowMaxK) {
    const int c = (k_routed + fp8_gemv::kChunkBytes - 1) / fp8_gemv::kChunkBytes;
    const int rows_per_block = fp8_gemv::kWarps * kDownNarrowGroups * (32 / c);
    const dim3 grid((n_routed + rows_per_block - 1) / rows_per_block,
                    static_cast<unsigned>(slots));
    moe_slot_down_narrow_kernel<<<grid, fp8_gemv::kThreads,
                                  fp8_gemv::smem_bytes(1, k_routed), stream>>>(
        act, act_stride, ids, order, views, n_routed, k_routed, out, out_stride, slots,
        top_k);
    DGPP_CUDA_OK(cudaGetLastError());
    return;
  }
  const int rows_per_block = fp8_gemv::kWarps * kDownRowsPerWarp;
  const dim3 grid((max_n + rows_per_block - 1) / rows_per_block,
                  static_cast<unsigned>(slots));
  moe_slot_down_kernel<<<grid, fp8_gemv::kThreads,
                         fp8_gemv::smem_bytes(1, max_k), stream>>>(
      act, act_stride, ids, order, views, n_routed, k_routed, n_shared,
      k_shared, sh_payload, sh_scales, out, out_stride, slots, top_k, sh_rs, sh_cs);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_slot_gate_up_swiglu(
    const uint16_t* x, size_t x_stride, const int32_t* ids,
    const int32_t* order, const MoeExpertView* views, int n_routed,
    int k_routed, int n_shared, int k_shared, const uint8_t* sh_gate_payload,
    const float* sh_gate_scales, const uint8_t* sh_up_payload,
    const float* sh_up_scales, uint16_t* act, int act_stride, int slots,
    int top_k, float limit, cudaStream_t stream, int sh_rs, int sh_cs) {
  if (slots <= 0) return;
  check_slot_args(x, ids, views, act, n_routed, k_routed, n_shared, k_shared,
                  "moe_slot_gate_up");
  if (n_shared > 0 &&
      (!sh_gate_payload || !sh_gate_scales || !sh_up_payload || !sh_up_scales ||
       !fp8_gemv::shape_ok(sh_gate_payload, 1, k_shared) ||
       !fp8_gemv::shape_ok(sh_up_payload, 1, k_shared)))
    throw std::invalid_argument(
        "moe_slot_gate_up: shared payloads must be 16B-aligned with "
        "k % 16 == 0");
  if (act_stride < n_routed || act_stride < n_shared)
    throw std::invalid_argument("moe_slot_gate_up: act_stride below n");
  const int max_n = n_routed > n_shared ? n_routed : n_shared;
  const int max_k = k_routed > k_shared ? k_routed : k_shared;
  const dim3 grid((max_n + fp8_gemv::kWarps - 1) / fp8_gemv::kWarps,
                  static_cast<unsigned>(slots));
  // The paired form on routed-only tables with more than one row of slots
  // (the fabric A/B of 2026-09-09: at 96 registers it halves the slot
  // kernel's occupancy — a 0.15–0.2 ms loss at one row, ~0.5 ms per token
  // gained at two); DGPP_MOE_PAIR=0|1 forces either form.
  static const int pair_knob = [] {
    const char* v = std::getenv("DGPP_MOE_PAIR");
    return v == nullptr ? -1 : (std::string(v) == "0" ? 0 : 1);
  }();
  const bool pair = n_shared == 0 && (pair_knob == 1 || (pair_knob < 0 && slots > top_k + 1));
  if (pair)
    moe_slot_gate_up_swiglu_kernel<true><<<grid, fp8_gemv::kThreads,
                                   fp8_gemv::smem_bytes(1, max_k), stream>>>(
      x, x_stride, ids, order, views, n_routed, k_routed, n_shared, k_shared,
      sh_gate_payload, sh_gate_scales, sh_up_payload, sh_up_scales, act,
      act_stride, slots, top_k, limit, sh_rs, sh_cs);
  else
    moe_slot_gate_up_swiglu_kernel<false><<<grid, fp8_gemv::kThreads,
                                   fp8_gemv::smem_bytes(1, max_k), stream>>>(
      x, x_stride, ids, order, views, n_routed, k_routed, n_shared, k_shared,
      sh_gate_payload, sh_gate_scales, sh_up_payload, sh_up_scales, act,
      act_stride, slots, top_k, limit, sh_rs, sh_cs);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_slot_accum_routed_f32(float* out, const float* contrib,
                                      const float* weights, int tokens,
                                      int hidden, int top_k,
                                      cudaStream_t stream) {
  if (tokens <= 0) return;
  if (!out || !contrib || !weights)
    throw std::invalid_argument("moe_slot_accum_routed_f32: null pointer");
  const int64_t n = static_cast<int64_t>(tokens) * hidden;
  const int64_t blocks = (n + kElemThreads - 1) / kElemThreads;
  moe_slot_accum_routed_f32_kernel<<<static_cast<int>(blocks), kElemThreads, 0,
                                     stream>>>(out, contrib, weights, n, hidden,
                                               top_k);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_slot_accum(uint16_t* out, const float* contrib,
                           const float* weights, int tokens, int hidden,
                           int top_k, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (!out || !contrib || !weights)
    throw std::invalid_argument("moe_slot_accum: null pointer");
  const int64_t n = static_cast<int64_t>(tokens) * hidden;
  const int64_t blocks = (n + kElemThreads - 1) / kElemThreads;
  moe_slot_accum_kernel<<<static_cast<int>(blocks), kElemThreads, 0, stream>>>(
      out, contrib, weights, n, hidden, top_k);
  DGPP_CUDA_OK(cudaGetLastError());
}

// ---- NVFP4 routed experts (2026-09-08, docs/nvfp4_plan.md §3.2, §3.3) ----
//
// The decode slot kernels and the host grouped kernel over expert-view
// tables whose routed entries are NVFP4. A block's slot decides its format
// uniformly: a routed slot runs the fp4 core (fp4_gemv.cuh, templated on
// the routed K) on the view's packed payload, per-row e4m3 scales and
// global scale; the shared slot (j == top_k) either runs the fp8 core on
// the launch-arg matrices (the composed GLM-5.3 checkpoint's FP8 shared
// expert) — inside the same launch, with the same rows per block (the fp4
// geometry's: Geom<K>::rows_per_block); the fp8 rows go through row_dots /
// block_rows_multi with that rows-per-warp, whose per-row arithmetic is the
// fp8 kernels' whatever the row schedule (fp8_gemv.cuh), so the shared
// expert stays bitwise the FP8 path's — or, with kSharedFp4 (GLM-4.7,
// docs/glm47_plan.md D3), reads the NVFP4 shared triple as view-table
// entry `shared_view_base` (= n_experts * 3) through the same fp4 core,
// at the shared width n_shared and the routed K.
namespace {

// The decode slot kernels' weight loads as streaming (evict-first) lines
// (2026-09-22 experiment, docs/mimo_v26_flash_plan.md §7.1: the expert
// bytes are read once per step; evict-first keeps the L2 for the
// prefetcher's lines). -DDGPP_FP4_SLOT_STREAMING=1 builds it; off until a
// fabric A/B decides.
#ifndef DGPP_FP4_SLOT_STREAMING
#define DGPP_FP4_SLOT_STREAMING 0
#endif
constexpr bool kFp4SlotStreaming = DGPP_FP4_SLOT_STREAMING != 0;

template <int K, bool kSharedFp4, int kGroup = fp4_gemv::kGroup>
__global__ void moe_slot_gate_up_swiglu_fp4_kernel(
    const uint16_t* __restrict__ x, size_t x_stride,
    const int32_t* __restrict__ ids, const int32_t* __restrict__ order,
    const MoeExpertView* __restrict__ views, int n_routed,
    int n_shared, int k_shared, const uint8_t* __restrict__ sh_gate_payload,
    const float* __restrict__ sh_gate_scales,
    const uint8_t* __restrict__ sh_up_payload,
    const float* __restrict__ sh_up_scales, uint16_t* __restrict__ act,
    int act_stride, int slots, int top_k, float limit, int shared_view_base, int sh_rs,
    int sh_cs) {
  using G = fp4_gemv::Geom<K>;
  extern __shared__ __align__(16) uint16_t sx[];
  const int n0 = blockIdx.x * G::rows_per_block;
  if (static_cast<int>(blockIdx.y) >= slots) return;
  const int slot = logical_slot(order);
  const int t = slot / (top_k + 1);
  const int j = slot - t * (top_k + 1);
  const bool shared = j == top_k;
  const int n = shared ? n_shared : n_routed;
  const int k = (shared && !kSharedFp4) ? k_shared : K;
  if (n0 >= n) return;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  uint16_t* act_row = act + static_cast<size_t>(slot) * act_stride;
  if (shared && !kSharedFp4) {
    fp8_gemv::stage_activations<1>(x + static_cast<size_t>(t) * x_stride, x_stride,
                                   k, sx);
    __syncthreads();
    // The fp8 kernel's per-row math (moe_slot_gate_up_swiglu_kernel), one
    // row at a time over this warp's rows.
    const int scale_cols = (k + (1 << sh_cs) - 1) >> sh_cs;
#pragma unroll 1
    for (int i = 0; i < G::rows_per_warp; ++i) {
      const int row = n0 + warp * G::rows_per_warp + i;
      if (row >= n) break;
      const size_t scale_row = static_cast<size_t>(row >> sh_rs) * scale_cols;
      float g_acc[1], u_acc[1];
      fp8_gemv::row_dots<1>(sh_gate_payload + static_cast<size_t>(row) * k,
                            sh_gate_scales + scale_row, sx, k, lane, g_acc, sh_cs);
      fp8_gemv::row_dots<1>(sh_up_payload + static_cast<size_t>(row) * k,
                            sh_up_scales + scale_row, sx, k, lane, u_acc, sh_cs);
      if (lane != 0) continue;
      float g = bf16_bits_to_float(float_to_bf16_bits(g_acc[0]));
      float u = bf16_bits_to_float(float_to_bf16_bits(u_acc[0]));
      if (g > limit) g = limit;
      u = fminf(fmaxf(u, -limit), limit);
      const uint16_t tt = float_to_bf16_bits(g * sigmoidf_acc(g));
      act_row[row] = float_to_bf16_bits(bf16_bits_to_float(tt) * u);
    }
    return;
  }
  const int base = (kSharedFp4 && shared) ? shared_view_base
                                          : ids[static_cast<size_t>(t) * top_k + j] * 3;
  const MoeExpertView vg = views[static_cast<size_t>(base) + 0];
  const MoeExpertView vu = views[static_cast<size_t>(base) + 1];
  // The first pass's weight loads in flight before the activations are
  // staged (fp4_gemv.cuh: PassLoads).
  fp4_gemv::PassLoads<G::pass_chunks(0)> L0, L1;
  fp4_gemv::issue_pass_pair<K, 0, G::pass_chunks(0), kGroup, kFp4SlotStreaming>(
      vg.payload, vg.fp4_scales, vu.payload, vu.fp4_scales, n0, n, L0, L1);
  fp8_gemv::stage_activations<1>(x + static_cast<size_t>(t) * x_stride, x_stride, k, sx);
  __syncthreads();
  float acc_g[fp4_gemv::kSteps][1], acc_u[fp4_gemv::kSteps][1];
  fp4_gemv::warp_row_dots_pair_issued<K, 1, kGroup, kFp4SlotStreaming>(
      L0, L1, vg.payload, vg.fp4_scales, vu.payload, vu.fp4_scales, sx, n0, n, acc_g, acc_u);
  // MXFP4 (kGroup 32) has no global: the dots stand undivided.
  const float gg = kGroup == fp4_gemv::kGroup ? *vg.fp4_global : 1.f;
  const float gu = kGroup == fp4_gemv::kGroup ? *vu.fp4_global : 1.f;
#pragma unroll
  for (int st = 0; st < fp4_gemv::kSteps; ++st) {
    bool mine = false;
    const int row = fp4_gemv::owned_row<K>(n0, st, mine);
    if (!mine || row >= n) continue;
    // The dots divided by their globals, rounded to bf16 where the fp8
    // chain rounds its gate/up outputs, then the swiglu, op for op.
    float g, u;
    if constexpr (kGroup == fp4_gemv::kGroup) {
      g = bf16_bits_to_float(float_to_bf16_bits(__fdiv_rn(acc_g[st][0], gg)));
      u = bf16_bits_to_float(float_to_bf16_bits(__fdiv_rn(acc_u[st][0], gu)));
    } else {
      g = bf16_bits_to_float(float_to_bf16_bits(acc_g[st][0]));
      u = bf16_bits_to_float(float_to_bf16_bits(acc_u[st][0]));
    }
    if (g > limit) g = limit;
    u = fminf(fmaxf(u, -limit), limit);
    const uint16_t tt = float_to_bf16_bits(g * sigmoidf_acc(g));
    act_row[row] = float_to_bf16_bits(bf16_bits_to_float(tt) * u);
  }
}

template <int K, bool kSharedFp4, int kGroup = fp4_gemv::kGroup>
__global__ void moe_slot_down_fp4_kernel(
    const uint16_t* __restrict__ act, size_t act_stride,
    const int32_t* __restrict__ ids, const int32_t* __restrict__ order,
    const MoeExpertView* __restrict__ views, int n_routed,
    int n_shared, int k_shared, const uint8_t* __restrict__ sh_payload,
    const float* __restrict__ sh_scales, float* __restrict__ out,
    int out_stride, int slots, int top_k, int shared_view_base, int sh_rs, int sh_cs) {
  using G = fp4_gemv::Geom<K>;
  extern __shared__ __align__(16) uint16_t sx[];
  const int n0 = blockIdx.x * G::rows_per_block;
  if (static_cast<int>(blockIdx.y) >= slots) return;
  const int slot = logical_slot(order);
  const int t = slot / (top_k + 1);
  const int j = slot - t * (top_k + 1);
  const bool shared = j == top_k;
  const int n = shared ? n_shared : n_routed;
  const int k = (shared && !kSharedFp4) ? k_shared : K;
  if (n0 >= n) return;
  float* out_row = out + static_cast<size_t>(slot) * out_stride;
  if (shared && !kSharedFp4) {
    fp8_gemv::stage_activations<1>(act + static_cast<size_t>(slot) * act_stride,
                                   act_stride, k, sx);
    __syncthreads();
    fp8_gemv::block_rows_multi<1, G::rows_per_warp>(
        sh_payload, sh_scales, sx, n0, n, k, out_row, static_cast<size_t>(out_stride), sh_rs, sh_cs);
    return;
  }
  const int base = (kSharedFp4 && shared) ? shared_view_base
                                          : ids[static_cast<size_t>(t) * top_k + j] * 3;
  const MoeExpertView v = views[static_cast<size_t>(base) + 2];
  const float g = kGroup == fp4_gemv::kGroup ? *v.fp4_global : 1.f;
  // The first pass's weight loads in flight before the activations are
  // staged (fp4_gemv.cuh: PassLoads); the epilogue is block_rows's.
  fp4_gemv::PassLoads<G::pass_chunks(0)> L0;
  fp4_gemv::issue_pass<K, 0, G::pass_chunks(0), kGroup, kFp4SlotStreaming>(v.payload, v.fp4_scales, n0, n, L0);
  fp8_gemv::stage_activations<1>(act + static_cast<size_t>(slot) * act_stride, act_stride, k, sx);
  __syncthreads();
  float acc[fp4_gemv::kSteps][1];
  fp4_gemv::warp_row_dots_issued<K, 1, kGroup, kFp4SlotStreaming>(L0, v.payload, v.fp4_scales, sx, n0, n, acc);
#pragma unroll
  for (int st = 0; st < fp4_gemv::kSteps; ++st) {
    bool mine = false;
    const int row = fp4_gemv::owned_row<K>(n0, st, mine);
    if (mine && row < n) {
      if constexpr (kGroup == fp4_gemv::kGroup)
        fp4_gemv::store_dot(out_row + row, __fdiv_rn(acc[st][0], g));
      else
        fp4_gemv::store_dot(out_row + row, acc[st][0]);
    }
  }
}

// The host path's grouped kernel over NVFP4 segments: rows staged four at a
// time (the multi-group loop), every weight row through the fp4 core. Every
// thread reaches every barrier: the core has no warp-uniform early return.
template <int K, typename OutT, int kGroup = fp4_gemv::kGroup>
__global__ void moe_grouped_gemv_fp4_kernel(const uint16_t* __restrict__ act,
                                            size_t act_stride,
                                            const MoeSegment* __restrict__ segs,
                                            const MoeExpertView* __restrict__ views,
                                            int which, OutT* __restrict__ out,
                                            size_t out_stride, int n,
                                            int rows_per_block) {
  using G = fp4_gemv::Geom<K>;
  extern __shared__ __align__(16) uint16_t sx[];
  const MoeSegment seg = segs[blockIdx.y];
  const int z0 = static_cast<int>(blockIdx.z) * rows_per_block;
  if (z0 >= seg.rows) return;
  const int z1 = min(seg.rows, z0 + rows_per_block);
  const MoeExpertView v = views[seg.expert * 3 + which];
  const float g = kGroup == fp4_gemv::kGroup ? *v.fp4_global : 1.f;
  const int n0 = blockIdx.x * G::rows_per_block;
  for (int gi = z0; gi < z1; gi += gemv::kMaxRows) {
    const int rows = min(gemv::kMaxRows, z1 - gi);
    const uint16_t* xr = act + static_cast<size_t>(seg.row0 + gi) * act_stride;
    OutT* o = out + static_cast<size_t>(seg.row0 + gi) * out_stride;
    if (gi > z0) __syncthreads();
    switch (rows) {
      case 4:
        fp4_gemv::stage_activations<4>(xr, act_stride, K, sx);
        __syncthreads();
        fp4_gemv::block_rows<K, 4, OutT, kGroup>(v.payload, v.fp4_scales, g, sx, n0, n, o, out_stride);
        break;
      case 3:
        fp4_gemv::stage_activations<3>(xr, act_stride, K, sx);
        __syncthreads();
        fp4_gemv::block_rows<K, 3, OutT, kGroup>(v.payload, v.fp4_scales, g, sx, n0, n, o, out_stride);
        break;
      case 2:
        fp4_gemv::stage_activations<2>(xr, act_stride, K, sx);
        __syncthreads();
        fp4_gemv::block_rows<K, 2, OutT, kGroup>(v.payload, v.fp4_scales, g, sx, n0, n, o, out_stride);
        break;
      default:
        fp4_gemv::stage_activations<1>(xr, act_stride, K, sx);
        __syncthreads();
        fp4_gemv::block_rows<K, 1, OutT, kGroup>(v.payload, v.fp4_scales, g, sx, n0, n, o, out_stride);
        break;
    }
  }
}

void check_fp4_slot_args(const void* x, const int32_t* ids,
                         const MoeExpertView* views, const void* out,
                         int n_routed, int k_routed, int n_shared, int k_shared,
                         int shared_view_base, const char* who, int fp4_group = fp4_gemv::kGroup) {
  if (fp4_group != fp4_gemv::kGroup && fp4_group != fp4_gemv::kMxGroup)
    throw std::invalid_argument(std::string(who) + ": the fp4 scale group must be 16 or 32");
  if (!x || !ids || !views || !out)
    throw std::invalid_argument(std::string(who) + ": null pointer");
  // n_shared == 0 (2026-09-10, the Qwen NVFP4 decode): no shared slot —
  // the kernels' shared-slot blocks find n == 0 and return; the shared
  // checks below are skipped then (as the fp8 launchers skip theirs).
  if (n_routed <= 0 || k_routed <= 0 || n_shared < 0 || k_shared < 0 ||
      (n_shared > 0) != (k_shared > 0))
    throw std::invalid_argument(std::string(who) + ": degenerate dims");
  // The fp4 core's contract on the routed k (the table's payloads ride the
  // loader's 256-byte grants, so only the K set is checked here).
  if (!fp4_gemv::k_supported(k_routed) || !fp4_gemv::k_compiled_for(k_routed, fp4_group) ||
      !gemv::smem_fits(1, k_routed))
    throw std::invalid_argument(
        std::string(who) + ": routed k must be a multiple of 32 in the fp4 core's compiled set");
  if (n_shared == 0) return;
  if (shared_view_base >= 0) {
    // The NVFP4 shared expert rides the routed K (the kernel's template).
    if (k_shared != k_routed)
      throw std::invalid_argument(std::string(who) +
                                  ": the NVFP4 shared expert's k must equal the routed k");
    return;
  }
  // The shared expert's fp8 rows share the launch: the fp8 core's contract.
  if (k_shared % fp8_gemv::kChunkBytes != 0 || !gemv::smem_fits(1, k_shared))
    throw std::invalid_argument(std::string(who) + ": shared k must be a multiple of 16");
}

template <typename OutT>
void launch_moe_grouped_gemv_fp4(const uint16_t* act, size_t act_stride,
                                 const MoeSegment* segs, int n_segs, int max_rows,
                                 int rows_per_block, const MoeExpertView* views,
                                 int which, OutT* out, size_t out_stride, int n,
                                 int k, cudaStream_t stream, int fp4_group) {
  if (n_segs <= 0 || n <= 0) return;
  if (!act || !segs || !views || !out)
    throw std::invalid_argument("moe grouped gemv fp4: null pointer");
  if (fp4_group != fp4_gemv::kGroup && fp4_group != fp4_gemv::kMxGroup)
    throw std::invalid_argument("moe grouped gemv fp4: the fp4 scale group must be 16 or 32");
  if (!fp4_gemv::k_supported(k) || !fp4_gemv::k_compiled_for(k, fp4_group))
    throw std::invalid_argument(
        "moe grouped gemv fp4: k must be a multiple of 32 in the fp4 core's compiled set");
  if (!gemv::smem_fits(gemv::kMaxRows, k))
    throw std::invalid_argument("moe grouped gemv fp4: k exceeds the smem budget");
  if (max_rows <= 0)
    throw std::invalid_argument("moe grouped gemv fp4: max_rows must be positive");
  const unsigned z_ext = rows_per_block > 0
                             ? static_cast<unsigned>((max_rows + rows_per_block - 1) / rows_per_block)
                             : 1u;
  if (rows_per_block <= 0) rows_per_block = INT_MAX;
  const auto go = [&](auto kc, auto gc) {
    constexpr int K = decltype(kc)::value;
    constexpr int G = decltype(gc)::value;
    constexpr int rpb = fp4_gemv::Geom<K>::rows_per_block;
    const dim3 grid((n + rpb - 1) / rpb, static_cast<unsigned>(n_segs), z_ext);
    moe_grouped_gemv_fp4_kernel<K, OutT, G>
        <<<grid, fp8_gemv::kThreads, gemv::smem_bytes(gemv::kMaxRows, K), stream>>>(
            act, act_stride, segs, views, which, out, out_stride, n, rows_per_block);
    DGPP_CUDA_OK(cudaGetLastError());
  };
  if (fp4_group == fp4_gemv::kMxGroup)
    fp4_gemv::dispatch_k_mx(k, [&](auto kc) { go(kc, std::integral_constant<int, fp4_gemv::kMxGroup>{}); });
  else
    fp4_gemv::dispatch_k(k, [&](auto kc) { go(kc, std::integral_constant<int, fp4_gemv::kGroup>{}); });
}

// ---- packed-int routed experts (2026-09-12, docs/glm53_plan.md D2) --------
//
// The decode slot kernels and the host grouped kernel over expert-view
// tables whose entries are packed-int (payload = the I32 words, packed_scales
// = bf16 per 64, bits = the code width). A block's slot decides its format
// uniformly: a routed slot runs the packed core (packq_gemv.cuh) at the
// routed width RBits and the routed K; the shared slot (j == top_k) reads
// the shared triple as view-table entry `shared_view_base` (= n_experts * 3,
// the loader's (E+1)-entry table) through the same core at the SHARED width
// (always 8 — the checkpoint's int8 shared expert; the draft's requant
// follows it) and the routed K (the shared expert's inter equals the
// routed experts'). The two widths have their own row geometries, so each
// block derives its row base from its own; the launcher's grid covers the
// finer one. Every row's arithmetic is the packed core's, bitwise across
// the grouped (host) and slot (decode) launchers.
namespace {

constexpr int kPackqSharedBits = 8;

// A warp's rows of gate and up dots from the staged activation row, then
// the swiglu op for op on the bf16-rounded dots (the fp8 kernel's math).
template <int Bits, int K, int SF>
__device__ __forceinline__ void packq_slot_gate_up_rows(const MoeExpertView& vg,
                                                        const MoeExpertView& vu,
                                                        const uint16_t* __restrict__ sx,
                                                        int n0, int n,
                                                        uint16_t* __restrict__ act_row,
                                                        float limit) {
  float acc_g[packq_gemv::kSteps][1], acc_u[packq_gemv::kSteps][1];
  packq_gemv::warp_row_dots_pair<Bits, K, 1, SF>(
      vg.payload, vg.packed_scales, vu.payload, vu.packed_scales, sx, n0, n, acc_g, acc_u);
#pragma unroll
  for (int st = 0; st < packq_gemv::kSteps; ++st) {
    bool mine = false;
    const int row = packq_gemv::owned_row<Bits, K, SF>(n0, st, mine);
    if (!mine || row >= n) continue;
    float g = bf16_bits_to_float(float_to_bf16_bits(acc_g[st][0]));
    float u = bf16_bits_to_float(float_to_bf16_bits(acc_u[st][0]));
    if (g > limit) g = limit;
    u = fminf(fmaxf(u, -limit), limit);
    const uint16_t tt = float_to_bf16_bits(g * sigmoidf_acc(g));
    act_row[row] = float_to_bf16_bits(bf16_bits_to_float(tt) * u);
  }
}

template <int RBits, int K, int SF>
__global__ void moe_slot_gate_up_swiglu_packq_kernel(
    const uint16_t* __restrict__ x, size_t x_stride,
    const int32_t* __restrict__ ids, const int32_t* __restrict__ order,
    const MoeExpertView* __restrict__ views, int n_routed, int n_shared,
    uint16_t* __restrict__ act, int act_stride, int slots, int top_k,
    float limit, int shared_view_base) {
  using GR = packq_gemv::Geom<RBits, K, SF>;
  using GS = packq_gemv::Geom<kPackqSharedBits, K, packq_gemv::kScaleBf16G64>;
  extern __shared__ __align__(16) uint16_t sx[];
  if (static_cast<int>(blockIdx.y) >= slots) return;
  const int slot = logical_slot(order);
  const int t = slot / (top_k + 1);
  const int j = slot - t * (top_k + 1);
  const bool shared = j == top_k;
  const int n = shared ? n_shared : n_routed;
  const int n0 = blockIdx.x * (shared ? GS::rows_per_block : GR::rows_per_block);
  if (n0 >= n) return;
  packq_gemv::stage_activations<1>(x + static_cast<size_t>(t) * x_stride, x_stride, K, sx);
  __syncthreads();
  uint16_t* act_row = act + static_cast<size_t>(slot) * act_stride;
  if (shared) {
    packq_slot_gate_up_rows<kPackqSharedBits, K, packq_gemv::kScaleBf16G64>(
        views[shared_view_base + 0], views[shared_view_base + 1], sx, n0, n, act_row, limit);
    return;
  }
  const int base = ids[static_cast<size_t>(t) * top_k + j] * 3;
  packq_slot_gate_up_rows<RBits, K, SF>(views[base + 0], views[base + 1], sx, n0, n, act_row,
                                        limit);
}

template <int RBits, int K, int SF>
__global__ void moe_slot_down_packq_kernel(
    const uint16_t* __restrict__ act, size_t act_stride,
    const int32_t* __restrict__ ids, const int32_t* __restrict__ order,
    const MoeExpertView* __restrict__ views, int n_routed, int n_shared,
    float* __restrict__ out, int out_stride, int slots, int top_k,
    int shared_view_base) {
  using GR = packq_gemv::Geom<RBits, K, SF>;
  using GS = packq_gemv::Geom<kPackqSharedBits, K, packq_gemv::kScaleBf16G64>;
  extern __shared__ __align__(16) uint16_t sx[];
  if (static_cast<int>(blockIdx.y) >= slots) return;
  const int slot = logical_slot(order);
  const int t = slot / (top_k + 1);
  const int j = slot - t * (top_k + 1);
  const bool shared = j == top_k;
  const int n = shared ? n_shared : n_routed;
  const int n0 = blockIdx.x * (shared ? GS::rows_per_block : GR::rows_per_block);
  if (n0 >= n) return;
  packq_gemv::stage_activations<1>(act + static_cast<size_t>(slot) * act_stride, act_stride,
                                   K, sx);
  __syncthreads();
  float* out_row = out + static_cast<size_t>(slot) * out_stride;
  if (shared) {
    const MoeExpertView v = views[shared_view_base + 2];
    packq_gemv::block_rows<kPackqSharedBits, K, 1, packq_gemv::kScaleBf16G64>(
        v.payload, v.packed_scales, sx, n0, n, out_row, static_cast<size_t>(out_stride));
    return;
  }
  const MoeExpertView v = views[ids[static_cast<size_t>(t) * top_k + j] * 3 + 2];
  packq_gemv::block_rows<RBits, K, 1, SF>(v.payload, v.packed_scales, sx, n0, n, out_row,
                                          static_cast<size_t>(out_stride));
}

// The host path's grouped kernel over packed segments: rows staged four
// at a time, every weight row through the packed core at one width per
// launch (the routed segments' or the shared segment's).
template <int Bits, int K, int SF, typename OutT>
__global__ void moe_grouped_gemv_packq_kernel(const uint16_t* __restrict__ act,
                                              size_t act_stride,
                                              const MoeSegment* __restrict__ segs,
                                              const MoeExpertView* __restrict__ views,
                                              int which, OutT* __restrict__ out,
                                              size_t out_stride, int n,
                                              int rows_per_block) {
  using G = packq_gemv::Geom<Bits, K, SF>;
  extern __shared__ __align__(16) uint16_t sx[];
  const MoeSegment seg = segs[blockIdx.y];
  const int z0 = static_cast<int>(blockIdx.z) * rows_per_block;
  if (z0 >= seg.rows) return;
  const int z1 = min(seg.rows, z0 + rows_per_block);
  const MoeExpertView v = views[seg.expert * 3 + which];
  const int n0 = blockIdx.x * G::rows_per_block;
  for (int gi = z0; gi < z1; gi += gemv::kMaxRows) {
    const int rows = min(gemv::kMaxRows, z1 - gi);
    const uint16_t* xr = act + static_cast<size_t>(seg.row0 + gi) * act_stride;
    OutT* o = out + static_cast<size_t>(seg.row0 + gi) * out_stride;
    if (gi > z0) __syncthreads();
    switch (rows) {
      case 4:
        packq_gemv::stage_activations<4>(xr, act_stride, K, sx);
        __syncthreads();
        packq_gemv::block_rows<Bits, K, 4, SF>(v.payload, v.packed_scales, sx, n0, n, o, out_stride);
        break;
      case 3:
        packq_gemv::stage_activations<3>(xr, act_stride, K, sx);
        __syncthreads();
        packq_gemv::block_rows<Bits, K, 3, SF>(v.payload, v.packed_scales, sx, n0, n, o, out_stride);
        break;
      case 2:
        packq_gemv::stage_activations<2>(xr, act_stride, K, sx);
        __syncthreads();
        packq_gemv::block_rows<Bits, K, 2, SF>(v.payload, v.packed_scales, sx, n0, n, o, out_stride);
        break;
      default:
        packq_gemv::stage_activations<1>(xr, act_stride, K, sx);
        __syncthreads();
        packq_gemv::block_rows<Bits, K, 1, SF>(v.payload, v.packed_scales, sx, n0, n, o, out_stride);
        break;
    }
  }
}

void check_packq_slot_args(const void* x, const int32_t* ids, const MoeExpertView* views,
                           const void* out, int n_routed, int k_routed, int routed_bits,
                           int n_shared, int shared_bits, int shared_view_base,
                           int routed_scale_fmt, const char* who) {
  if (!x || !ids || !views || !out)
    throw std::invalid_argument(std::string(who) + ": null pointer");
  if (n_routed <= 0 || k_routed <= 0 || n_shared < 0)
    throw std::invalid_argument(std::string(who) + ": degenerate dims");
  if (routed_bits != 4 && routed_bits != 8)
    throw std::invalid_argument(std::string(who) + ": the routed code width must be 4 or 8");
  if (!packq_gemv::scale_fmt_known(routed_scale_fmt))
    throw std::invalid_argument(std::string(who) + ": unknown routed packed scale format");
  if (!packq_gemv::k_supported_bits(routed_bits, k_routed, routed_scale_fmt) ||
      !packq_gemv::sf_compiled(routed_scale_fmt, k_routed) || !gemv::smem_fits(1, k_routed))
    throw std::invalid_argument(
        std::string(who) +
        ": routed k must be a multiple of the scale group in the packed core's compiled set");
  if (n_shared == 0) return;
  // The shared expert rides the routed K through the view table at the
  // shared width (the fp8 launch-argument form is not a packed table's).
  if (shared_view_base < 0)
    throw std::invalid_argument(std::string(who) + ": a packed table carries its shared expert in the view table");
  if (shared_bits != kPackqSharedBits)
    throw std::invalid_argument(std::string(who) + ": the packed shared expert is int8");
  if (!packq_gemv::k_supported_bits(kPackqSharedBits, k_routed))
    throw std::invalid_argument(std::string(who) + ": the shared expert's k exceeds the int8 core's budget");
}

template <typename OutT>
void launch_moe_grouped_gemv_packq(const uint16_t* act, size_t act_stride,
                                   const MoeSegment* segs, int n_segs, int max_rows,
                                   int rows_per_block, const MoeExpertView* views,
                                   int which, OutT* out, size_t out_stride, int n,
                                   int k, int bits, int scale_fmt, cudaStream_t stream) {
  if (n_segs <= 0 || n <= 0) return;
  if (!act || !segs || !views || !out)
    throw std::invalid_argument("moe grouped gemv packq: null pointer");
  if (!packq_gemv::scale_fmt_known(scale_fmt))
    throw std::invalid_argument("moe grouped gemv packq: unknown packed scale format");
  if (!packq_gemv::k_supported_bits(bits, k, scale_fmt) || !packq_gemv::sf_compiled(scale_fmt, k))
    throw std::invalid_argument(
        "moe grouped gemv packq: k must be a multiple of the scale group in the packed core's compiled set");
  if (!gemv::smem_fits(gemv::kMaxRows, k))
    throw std::invalid_argument("moe grouped gemv packq: k exceeds the smem budget");
  if (max_rows <= 0)
    throw std::invalid_argument("moe grouped gemv packq: max_rows must be positive");
  const unsigned z_ext = rows_per_block > 0
                             ? static_cast<unsigned>((max_rows + rows_per_block - 1) / rows_per_block)
                             : 1u;
  if (rows_per_block <= 0) rows_per_block = INT_MAX;
  packq_gemv::dispatch_sf(scale_fmt, [&](auto sfc) {
    constexpr int SF = decltype(sfc)::value;
    packq_gemv::dispatch_bits(bits, [&](auto bc) {
      constexpr int Bits = decltype(bc)::value;
      packq_gemv::dispatch_k(k, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        if constexpr (packq_gemv::k_supported<Bits, SF>(K) && packq_gemv::sf_compiled(SF, K)) {
          constexpr int rpb = packq_gemv::Geom<Bits, K, SF>::rows_per_block;
          const dim3 grid((n + rpb - 1) / rpb, static_cast<unsigned>(n_segs), z_ext);
          moe_grouped_gemv_packq_kernel<Bits, K, SF, OutT>
              <<<grid, fp8_gemv::kThreads, gemv::smem_bytes(gemv::kMaxRows, K), stream>>>(
                  act, act_stride, segs, views, which, out, out_stride, n, rows_per_block);
          DGPP_CUDA_OK(cudaGetLastError());
        } else {
          throw std::invalid_argument(
              "moe grouped gemv packq: K is outside the scale format's compiled set or the width's chunk budget");
        }
      });
    });
  });
}

}  // namespace

// ---- the fp4 grouped tensor-core kernel (docs/nvfp4_plan.md §3.4) --------
// The fp8 tile kernel's structure — mma_tile's 128 x 64 x 64 stages, the
// same activation tile and the same ascending-k16 bf16 mma.sync chain —
// with the weight tile decoded from the NVFP4 triple: thread t holds 16
// consecutive k of n-row t/4, one 8-byte load of codes and one scale byte
// (the row's group-16 scale for those k), decoded as e2m1(code) x scale,
// which is EXACT in bf16 (§3.1: <= 6 significant bits), so unlike the fp8
// tile no rounding happens before the MMA; the tensor's global scale
// divides the finished dot once in the epilogue, as the fp4 GEMV core
// does. The dense form (segs == nullptr) is the same kernel over one
// matrix; the grouped form is bitwise the dense form per segment
// (glm_moe_test pins it) — the two differ from the fp4 GEMV core only by
// the fp32 summation order.
__device__ __forceinline__ float e4m3_x16384(uint8_t s) {
  const __half_raw h =
      __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(s), __NV_E4M3);
  return __half2float(__half(h)) * 16384.f;
}

// The first fp4 tile kernel, kept as the TILE REFERENCE the
// dense launcher runs and glm_moe_test pins the pipelined grouped kernel
// against bitwise: mma_tile's shape, one synchronous stage at a time.
// kSkip (the bench's decomposition, never dispatched): 1 skips the weight
// loads, 2 the activation loads, 3 both — the stage's load latency vs its
// decode + barrier + MMA floor.
template <typename OutT, int BM_T = mma_tile::BM, int BN_T = mma_tile::BN,
          int BK_T = mma_tile::BK, int kMinBlocks = 3, int kSkip = 0>
__global__ __launch_bounds__(mma_tile::kThreads, kMinBlocks) void moe_grouped_mma_fp4_ref_kernel(
    const uint16_t* __restrict__ act, size_t act_stride, int act_vec,
    const int32_t* __restrict__ act_rows,
    const MoeSegment* __restrict__ segs, const MoeExpertView* __restrict__ views,
    int which, OutT* __restrict__ out, size_t out_stride, int n, int k,
    int rows_per_block, MoeSegment dense_seg, MoeExpertView dense_view) {
  using namespace mma_tile;
  constexpr int BM = BM_T;
  constexpr int BN = BN_T;               // 64 (the reference) or 128
  constexpr int BK = BK_T;               // 64 (the reference) or 32
  constexpr int BK_PAD = BK + 8;
  static_assert(BK % 16 == 0 && BK <= 64, "a stage is one or more scale groups");
  // Warps: BM = 128 -> 8 (m16) x 1 over BN; BM = 64 -> 4 (m16) x 2 (BN/2 each).
  constexpr int kWm = BM / 16, kWn = 8 / kWm, kFrag = BN / (8 * kWn);
  __shared__ __align__(16) uint16_t sA[BM][BK_PAD];
  __shared__ __align__(16) uint16_t sB[BN][BK_PAD];
  const MoeSegment seg = segs != nullptr ? segs[blockIdx.y] : dense_seg;
  const int z0 = static_cast<int>(blockIdx.z) * rows_per_block;
  if (z0 >= seg.rows) return;  // beyond this segment's rows (no barrier yet)
  const int z1 = min(seg.rows, z0 + rows_per_block);
  const MoeExpertView v =
      segs != nullptr ? views[seg.expert * 3 + which] : dense_view;
  const float g = *v.fp4_global;
  const int n0 = static_cast<int>(blockIdx.x) * BN;
  const size_t payload_stride = static_cast<size_t>(k) / 2;   // bytes per row
  const size_t scale_stride = static_cast<size_t>(k) / kFp4Group;
  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int wm = warp % kWm, wn = warp / kWm;
  const int r = lane / 4;
  const int cc = (lane % 4) * 2;
  // The weight tile's load geometry: thread t decodes 16 consecutive k of
  // n-row t/4 (64 rows x 4 groups = 256 threads; 8 code bytes + 1 scale).
  constexpr int kGroupsPerRow = BK / 16;   // 16-code groups per row per stage
  constexpr int kGroupsPerThread = (BN * kGroupsPerRow + kThreads - 1) / kThreads;  // 1 or 2
  static_assert((BN * kGroupsPerRow) % kThreads == 0 || BN * kGroupsPerRow < kThreads,
                "groups spread evenly over the threads");

  for (int m0 = z0; m0 < z1; m0 += BM) {
    const int m_rows = min(BM, z1 - m0);
    float acc[kFrag][4];
#pragma unroll
    for (int j = 0; j < kFrag; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

    for (int k0 = 0; k0 < k; k0 += BK) {
      // Weight tile: 16 codes x the group's scale, exact bf16; zero outside
      // [n, k) (k is a multiple of 16, so a group is in or out whole).
#pragma unroll
      for (int gi = 0; gi < kGroupsPerThread; ++gi) {
        const int g = static_cast<int>(threadIdx.x) + gi * kThreads;
        const int b_row = g / kGroupsPerRow, b_kq = (g % kGroupsPerRow) * 16;
        if (b_row >= BN) break;
        const int gn = n0 + b_row, gk = k0 + b_kq;
        uint32_t packed[8];
        if (gn < n && gk < k) {
          const uint2 raw = (kSkip & 1) ? make_uint2(0x12345678u ^ gk, 0x9abcdef0u ^ gn)
                                        : *reinterpret_cast<const uint2*>(
                                              v.payload + static_cast<size_t>(gn) * payload_stride + gk / 2);
          const float s = e4m3_x16384(
              (kSkip & 1) ? static_cast<uint8_t>(0x38)
                          : v.fp4_scales[static_cast<size_t>(gn) * scale_stride + gk / kFp4Group]);
          const uint32_t words[2] = {raw.x, raw.y};
#pragma unroll
          for (int q = 0; q < 2; ++q) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const uint32_t byte = (words[q] >> (8 * j)) & 0xFFu;
              const uint16_t lo =
                  float_to_bf16_bits(fp4_gemv::e2m1_scaled(byte & 0xFu) * s);
              const uint16_t hi =
                  float_to_bf16_bits(fp4_gemv::e2m1_scaled(byte >> 4) * s);
              packed[q * 4 + j] =
                  static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
            }
          }
        } else {
#pragma unroll
          for (int i = 0; i < 8; ++i) packed[i] = 0u;
        }
        uint4* dst = reinterpret_cast<uint4*>(&sB[b_row][b_kq]);
        dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
        dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
      }
      // Activation tile: rows of this m-tile, zero-filled past the segment
      // and past k (16-byte loads when the rows are 16-byte aligned).
      for (int i = static_cast<int>(threadIdx.x); i < BM * (BK / 8); i += kThreads) {
        const int mm = i / (BK / 8), kq = (i % (BK / 8)) * 8;
        const int gk = k0 + kq;
        uint4 val = make_uint4(0, 0, 0, 0);
        if ((kSkip & 2) && mm < m_rows) {
          val = make_uint4(0x3f803f80u ^ gk, 0x3f803f80u, 0x3f803f80u ^ mm, 0x3f803f80u);
        } else if (mm < m_rows) {
          const int srow = seg.row0 + m0 + mm;
          const uint16_t* row =
              act + static_cast<size_t>(act_rows != nullptr ? act_rows[srow] : srow) *
                        act_stride;
          if (act_vec != 0 && gk + 8 <= k) {
            val = *reinterpret_cast<const uint4*>(row + gk);
          } else {
            uint16_t e[8];
#pragma unroll
            for (int h = 0; h < 8; ++h) e[h] = gk + h < k ? row[gk + h] : 0;
            val = make_uint4(e[0] | (e[1] << 16), e[2] | (e[3] << 16),
                             e[4] | (e[5] << 16), e[6] | (e[7] << 16));
          }
        }
        *reinterpret_cast<uint4*>(&sA[mm][kq]) = val;
      }
      __syncthreads();

      // A warp whose 16 rows are all past the segment's end has nothing to
      // accumulate (its rows are never stored): skip the MMAs, keep the
      // loads and barriers. Warp-uniform; the ragged tail of a segment no
      // longer costs a full tile of tensor work (2026-09-08: at ~57 rows
      // per expert the 128-row tile was 55 % zero rows).
      const bool warp_live = (kSkip & 4) == 0 && wm * 16 < m_rows;
#pragma unroll
      for (int kk = 0; kk < BK; kk += 16) {
        if (!warp_live) break;
        const int ar = wm * 16 + r;
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sA[ar][kk + cc]);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sA[ar + 8][kk + cc]);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sA[ar][kk + cc + 8]);
        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&sA[ar + 8][kk + cc + 8]);
#pragma unroll
        for (int j = 0; j < kFrag; ++j) {
          const int bn = wn * (kFrag * 8) + j * 8 + r;
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sB[bn][kk + cc]);
          const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sB[bn][kk + cc + 8]);
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
              : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
              : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
      }
      __syncthreads();  // tile reads done before the next stage overwrites
    }

    // Epilogue: per warp a [16 x 64] slice; the thread holds rows r, r+8
    // and columns cc, cc+1 of each n8 fragment. The global scale divides
    // the finished dot once (the only inexact step of the formula).
    const int row_lo = wm * 16 + r;
#pragma unroll
    for (int j = 0; j < kFrag; ++j) {
      const int gn = n0 + wn * (kFrag * 8) + j * 8 + cc;
      auto st = [&](int row_off, int col_off, float val) {
        const int mm = row_lo + row_off;
        if (mm < m_rows && gn + col_off < n)
          fp8_gemv::store_dot(
              out + static_cast<size_t>(seg.row0 + m0 + mm) * out_stride + gn + col_off,
              __fdiv_rn(val, g));
      };
      st(0, 0, acc[j][0]);
      st(0, 1, acc[j][1]);
      st(8, 0, acc[j][2]);
      st(8, 1, acc[j][3]);
    }
  }
}

// ---- the pipelined fp4 tile kernel (phase 5, 2026-09-08) -----------------
// nsys on the 2,048-token prefill put the tile kernels at a third of the
// read floor: with 64-wide n-tiles the activation tile was re-read once
// per n-tile (8x for gate/up, 64x for the down), the stages ran load ->
// sync -> mma -> sync with nothing in flight, and a 128-row m-tile was more
// than half padding at ~57 rows per expert. This kernel: 64 x 128 tiles
// over 32-deep stages (eight warps as 4 m16 x 2 n64), the activation tile
// fetched by 16-byte cp.async into a two-stage shared-memory ring while the
// previous stage multiplies, the next stage's weight codes (8 bytes + one
// scale byte per thread) loaded into registers ahead of the MMA and decoded
// after it. Every output element is still the same ascending-k16 mma.sync
// chain over the same exact bf16 weights, so the outputs are bitwise the
// reference kernel's (the gate). One __syncthreads per stage.
namespace fp4_tile {
constexpr int BM = 64;
constexpr int BN = 128;
constexpr int BK = 32;               // two scale groups per row per stage
constexpr int BK_PAD = BK + 8;       // u16 pad: 80-byte rows, conflict-free
constexpr int kThreads = 256;
constexpr int kStages = 2;
static_assert(BM * (BK / 8) == kThreads, "one 16-byte activation chunk per thread");
static_assert(BN * (BK / 16) == kThreads, "one 16-code group per thread");
static_assert(BK_PAD * 2 % 16 == 0, "16-byte aligned smem rows");

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem, int src_bytes) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :
               : "r"(s), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}

// Decode 16 codes (two words) at scale s (times 2^14) into eight packed
// bf16 pairs, exact.
__device__ __forceinline__ void decode16(uint32_t w0, uint32_t w1, float s,
                                         uint32_t (&packed)[8]) {
  const uint32_t words[2] = {w0, w1};
#pragma unroll
  for (int q = 0; q < 2; ++q) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t byte = (words[q] >> (8 * j)) & 0xFFu;
      const uint16_t lo = float_to_bf16_bits(fp4_gemv::e2m1_scaled(byte & 0xFu) * s);
      const uint16_t hi = float_to_bf16_bits(fp4_gemv::e2m1_scaled(byte >> 4) * s);
      packed[q * 4 + j] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
    }
  }
}
}  // namespace fp4_tile

template <typename OutT>
__global__ __launch_bounds__(fp4_tile::kThreads) void moe_grouped_mma_fp4_kernel(
    const uint16_t* __restrict__ act, size_t act_stride, int act_vec,
    const int32_t* __restrict__ act_rows,
    const MoeSegment* __restrict__ segs, const MoeExpertView* __restrict__ views,
    int which, OutT* __restrict__ out, size_t out_stride, int n, int k,
    int rows_per_block) {
  using namespace fp4_tile;
  __shared__ __align__(16) uint16_t sA[kStages][BM][BK_PAD];
  __shared__ __align__(16) uint16_t sB[kStages][BN][BK_PAD];
  const MoeSegment seg = segs[blockIdx.y];
  const int z0 = static_cast<int>(blockIdx.z) * rows_per_block;
  if (z0 >= seg.rows) return;  // beyond this segment's rows (no barrier yet)
  const int z1 = min(seg.rows, z0 + rows_per_block);
  const MoeExpertView v = views[seg.expert * 3 + which];
  const float g = *v.fp4_global;
  const int n0 = static_cast<int>(blockIdx.x) * BN;
  const size_t payload_stride = static_cast<size_t>(k) / 2;
  const size_t scale_stride = static_cast<size_t>(k) / kFp4Group;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32, lane = tid % 32;
  const int wm = warp % 4, wn = warp / 4;  // m16 block, n64 half
  const int r = lane / 4, cc = (lane % 4) * 2;
  // Weight tile geometry: thread t holds 16 codes of n-row t/2, half t%2.
  const int b_row = tid / 2, b_kq = (tid % 2) * 16;
  const int gn = n0 + b_row;
  const bool b_row_ok = gn < n;
  const uint8_t* b_src = v.payload + static_cast<size_t>(b_row_ok ? gn : 0) * payload_stride;
  const uint8_t* b_scale = v.fp4_scales + static_cast<size_t>(b_row_ok ? gn : 0) * scale_stride;
  // Activation tile geometry: thread t copies 8 elements of m-row t/4.
  const int a_row = tid / 4, a_kq = (tid % 4) * 8;
  const int stages = (k + BK - 1) / BK;

  for (int m0 = z0; m0 < z1; m0 += BM) {
    const int m_rows = min(BM, z1 - m0);
    const uint16_t* a_src = act;  // the row this thread copies (or the base, zero-filled)
    const bool a_row_ok = a_row < m_rows;
    if (a_row_ok) {
      const int srow = seg.row0 + m0 + a_row;
      a_src = act + static_cast<size_t>(act_rows != nullptr ? act_rows[srow] : srow) * act_stride;
    }
    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

    // Stage s's activation chunk: cp.async when the rows are 16-byte
    // aligned, a plain copy otherwise (the fallback keeps every input legal).
    auto issue_a = [&](int s, int buf) {
      const int gk = s * BK + a_kq;
      const bool in = a_row_ok && gk < k;  // k % 8 == 0: a chunk is in or out whole
      uint16_t* dst = &sA[buf][a_row][a_kq];
      if (act_vec != 0) {
        cp_async_16(dst, in ? a_src + gk : act, in ? 16 : 0);
      } else {
        uint16_t e[8];
#pragma unroll
        for (int h = 0; h < 8; ++h) e[h] = in ? a_src[gk + h] : static_cast<uint16_t>(0);
        *reinterpret_cast<uint4*>(dst) =
            make_uint4(e[0] | (e[1] << 16), e[2] | (e[3] << 16), e[4] | (e[5] << 16),
                       e[6] | (e[7] << 16));
      }
    };
    auto load_b = [&](int s, uint32_t& w0, uint32_t& w1, uint8_t& sc) {
      const int gk = s * BK + b_kq;
      if (b_row_ok && gk < k) {
        const uint2 raw = *reinterpret_cast<const uint2*>(b_src + gk / 2);
        w0 = raw.x;
        w1 = raw.y;
        sc = b_scale[gk / kFp4Group];
      } else {
        w0 = w1 = 0u;
        sc = 0;  // e4m3 zero: the codes decode to 0 either way
      }
    };
    auto store_b = [&](int buf, uint32_t w0, uint32_t w1, uint8_t sc) {
      uint32_t packed[8];
      decode16(w0, w1, e4m3_x16384(sc), packed);
      uint4* dst = reinterpret_cast<uint4*>(&sB[buf][b_row][b_kq]);
      dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
      dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
    };

    // Prologue: stage 0 in flight, decoded, visible.
    uint32_t bw0, bw1;
    uint8_t bsc;
    issue_a(0, 0);
    cp_async_commit();
    load_b(0, bw0, bw1, bsc);
    cp_async_wait_all();
    store_b(0, bw0, bw1, bsc);
    __syncthreads();

    for (int s = 0; s < stages; ++s) {
      const int buf = s % kStages;
      const bool more = s + 1 < stages;
      if (more) {
        issue_a(s + 1, (s + 1) % kStages);
        cp_async_commit();
        load_b(s + 1, bw0, bw1, bsc);
      }
      const bool warp_live = wm * 16 < m_rows;  // all-padding rows: no MMA
#pragma unroll
      for (int kk = 0; kk < BK; kk += 16) {
        if (!warp_live) break;
        const int ar = wm * 16 + r;
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sA[buf][ar][kk + cc]);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sA[buf][ar + 8][kk + cc]);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sA[buf][ar][kk + cc + 8]);
        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&sA[buf][ar + 8][kk + cc + 8]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int bn = wn * 64 + j * 8 + r;
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sB[buf][bn][kk + cc]);
          const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sB[buf][bn][kk + cc + 8]);
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
              : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
              : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
      }
      if (more) {
        cp_async_wait_all();
        store_b((s + 1) % kStages, bw0, bw1, bsc);
      }
      __syncthreads();  // stage s+1 visible; everyone past stage s's reads
    }

    // Epilogue: the warp's [16 x 64] slice; ÷g once per finished dot.
    const int row_lo = wm * 16 + r;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int gcol = n0 + wn * 64 + j * 8 + cc;
      auto st = [&](int row_off, int col_off, float val) {
        const int mm = row_lo + row_off;
        if (mm < m_rows && gcol + col_off < n)
          fp8_gemv::store_dot(
              out + static_cast<size_t>(seg.row0 + m0 + mm) * out_stride + gcol + col_off,
              __fdiv_rn(val, g));
      };
      st(0, 0, acc[j][0]);
      st(0, 1, acc[j][1]);
      st(8, 0, acc[j][2]);
      st(8, 1, acc[j][3]);
    }
  }
}

void check_fp4_mma_shape(int k, const char* who, int fp4_group = kFp4Group);

// ---- the three-stage fp4 tile kernel (phase 5, 2026-09-08) ----------------
// The reference's shape (128 x 64 x 64, eight warps x m16, the padding
// warps skipping their MMAs) with the operand loads two stages ahead of
// the multiply: a three-slot ring in shared memory holds, per stage, the
// activation tile (cp.async, 16 bytes per copy) and the raw fp4 codes and
// scales of the weight tile (cp.async, 8 and 4 bytes); the codes are
// decoded out of the ring into a double-buffered bf16 tile just before
// their stage multiplies. The decomposition that motivated it: the
// synchronous kernel's launch was 58 % MMA + decode + barriers and the
// rest un-overlapped loads. Same decode, same ascending-k16 mma chain: the
// outputs are bitwise the reference's. Dynamic shared memory (~73 KB, one
// block per SM), opted in by the launcher.
namespace fp4_pipe3 {
constexpr int BM = 128, BN = 64, BK = 64, BK_PAD = BK + 8;
constexpr int kThreads = 256, kStages = 3;
constexpr int kGroupsPerRow = BK / 16;               // 4 scale groups per row per stage
constexpr int kGroups = BN * kGroupsPerRow;          // 256: one per thread
constexpr size_t kABytes = size_t(BM) * BK_PAD * 2;  // 18,432
constexpr size_t kCodeBytes = size_t(BN) * BK / 2;   // 2,048
constexpr size_t kScaleBytes = size_t(BN) * kGroupsPerRow;  // 256
constexpr size_t kRawBytes = kCodeBytes + kScaleBytes;
constexpr size_t kBBytes = size_t(BN) * BK_PAD * 2;  // 9,216 (decoded)
constexpr size_t kSmem = kStages * (kABytes + kRawBytes) + 2 * kBBytes;
static_assert(kGroups == kThreads, "one 16-code group per thread per stage");
static_assert(BM * (BK / 8) == 4 * kThreads, "four activation chunks per thread");

__device__ __forceinline__ void cp_async(void* smem, const void* gmem, int bytes, int src_bytes) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  if (bytes == 16)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(gmem), "r"(src_bytes));
  else if (bytes == 8)
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(d), "l"(gmem), "r"(src_bytes));
  else
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(d), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
}  // namespace fp4_pipe3

template <typename OutT>
__global__ __launch_bounds__(fp4_pipe3::kThreads, 1) void moe_grouped_mma_fp4_pipe3_kernel(
    const uint16_t* __restrict__ act, size_t act_stride, int act_vec,
    const int32_t* __restrict__ act_rows,
    const MoeSegment* __restrict__ segs, const MoeExpertView* __restrict__ views,
    int which, OutT* __restrict__ out, size_t out_stride, int n, int k,
    int rows_per_block) {
  using namespace fp4_pipe3;
  extern __shared__ __align__(16) uint8_t smem[];
  uint16_t* sA = reinterpret_cast<uint16_t*>(smem);                       // [kStages][BM][BK_PAD]
  uint8_t* sRaw = smem + kStages * kABytes;                                // [kStages][codes | scales]
  uint16_t* sB = reinterpret_cast<uint16_t*>(sRaw + kStages * kRawBytes);  // [2][BN][BK_PAD]
  const MoeSegment seg = segs[blockIdx.y];
  const int z0 = static_cast<int>(blockIdx.z) * rows_per_block;
  if (z0 >= seg.rows) return;
  const int z1 = min(seg.rows, z0 + rows_per_block);
  const MoeExpertView v = views[seg.expert * 3 + which];
  const float g = *v.fp4_global;
  const int n0 = static_cast<int>(blockIdx.x) * BN;
  const size_t payload_stride = static_cast<size_t>(k) / 2;
  const size_t scale_stride = static_cast<size_t>(k) / kFp4Group;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32, lane = tid % 32;
  const int r = lane / 4, cc = (lane % 4) * 2;
  // Weight tile: thread t -> n-row t/4, group t%4 (16 codes = 8 bytes, 1 scale byte;
  // the scales of a row's four groups are one 4-byte copy by the t%4 == 0 thread).
  const int b_row = tid / kGroupsPerRow, b_g = tid % kGroupsPerRow;
  const int gn = n0 + b_row;
  const bool b_ok = gn < n;
  const uint8_t* b_codes = v.payload + static_cast<size_t>(b_ok ? gn : 0) * payload_stride;
  const uint8_t* b_scales = v.fp4_scales + static_cast<size_t>(b_ok ? gn : 0) * scale_stride;
  const int stages = (k + BK - 1) / BK;

  for (int m0 = z0; m0 < z1; m0 += BM) {
    const int m_rows = min(BM, z1 - m0);
    const bool warp_live = warp * 16 < m_rows;
    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

    // Stage s into ring slot `slot`: the activation chunks (16 bytes each,
    // four per thread: rows tid/8 + 32i, 8-element chunk tid%8) and the
    // weight tile's raw codes and scales. Absent rows and k past the end
    // arrive as zeros (src-size 0). A misaligned activation base takes a
    // plain copy (the rows would not be 16-byte aligned for cp.async).
    auto issue = [&](int s, int slot) {
      const int k0 = s * BK;
      uint16_t* a = sA + static_cast<size_t>(slot) * BM * BK_PAD;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int mm = tid / 8 + 32 * i, kq = (tid % 8) * 8;
        const int gk = k0 + kq;
        const bool in = mm < m_rows && gk < k;
        uint16_t* dst = a + static_cast<size_t>(mm) * BK_PAD + kq;
        const uint16_t* src = act;
        if (in) {
          const int srow = seg.row0 + m0 + mm;
          src = act + static_cast<size_t>(act_rows != nullptr ? act_rows[srow] : srow) * act_stride + gk;
        }
        if (act_vec != 0) {
          cp_async(dst, src, 16, in ? 16 : 0);
        } else {
          uint16_t e[8];
#pragma unroll
          for (int h = 0; h < 8; ++h) e[h] = in ? src[h] : static_cast<uint16_t>(0);
          *reinterpret_cast<uint4*>(dst) = make_uint4(e[0] | (e[1] << 16), e[2] | (e[3] << 16),
                                                      e[4] | (e[5] << 16), e[6] | (e[7] << 16));
        }
      }
      uint8_t* raw = sRaw + static_cast<size_t>(slot) * kRawBytes;
      const int gk = k0 + b_g * 16;
      const bool in = b_ok && gk < k;  // k % 16 == 0: a group is in or out whole
      cp_async(raw + static_cast<size_t>(b_row) * (BK / 2) + b_g * 8,
               in ? b_codes + gk / 2 : v.payload, 8, in ? 8 : 0);
      if (b_g == 0) {
        // The row's four scale bytes: whole when all four groups are in, else
        // byte by byte is not possible with cp.async — the tail stage of a k
        // that is not a multiple of 64 copies the in-range prefix (4 bytes
        // when k0 + 64 <= k, else the copy is 4 bytes with src-size limited).
        const int in_groups = b_ok ? max(0, min(kGroupsPerRow, (k - k0) / 16)) : 0;
        cp_async(raw + kCodeBytes + static_cast<size_t>(b_row) * kGroupsPerRow,
                 in_groups > 0 ? b_scales + k0 / 16 : v.fp4_scales, 4, in_groups);
      }
    };
    // Decode ring slot `slot`'s raw tile into the bf16 tile `buf`.
    auto decode = [&](int slot, int buf) {
      const uint8_t* raw = sRaw + static_cast<size_t>(slot) * kRawBytes;
      const uint2 codes = *reinterpret_cast<const uint2*>(
          raw + static_cast<size_t>(b_row) * (BK / 2) + b_g * 8);
      const uint8_t sc = raw[kCodeBytes + static_cast<size_t>(b_row) * kGroupsPerRow + b_g];
      const float sx = e4m3_x16384(sc);
      const uint32_t words[2] = {codes.x, codes.y};
      uint32_t packed[8];
#pragma unroll
      for (int q = 0; q < 2; ++q) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t byte = (words[q] >> (8 * j)) & 0xFFu;
          const uint16_t lo = float_to_bf16_bits(fp4_gemv::e2m1_scaled(byte & 0xFu) * sx);
          const uint16_t hi = float_to_bf16_bits(fp4_gemv::e2m1_scaled(byte >> 4) * sx);
          packed[q * 4 + j] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
        }
      }
      uint4* dst = reinterpret_cast<uint4*>(sB + static_cast<size_t>(buf) * BN * BK_PAD +
                                            static_cast<size_t>(b_row) * BK_PAD + b_g * 16);
      dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
      dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
    };

    // Prologue: stages 0 .. kStages-2 in flight.
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
      if (s < stages) issue(s, s);
      commit();
    }
    for (int s = 0; s < stages; ++s) {
      const int slot = s % kStages, buf = s % 2;
      // Stage s has landed (kStages-2 younger groups may still be in flight).
      wait<kStages - 2>();
      __syncthreads();
      // Everyone is past stage s-1's multiply: its slot takes stage s+kStages-1.
      if (s + kStages - 1 < stages) issue(s + kStages - 1, (s + kStages - 1) % kStages);
      commit();
      decode(slot, buf);
      __syncthreads();
      const uint16_t* a = sA + static_cast<size_t>(slot) * BM * BK_PAD;
      const uint16_t* b = sB + static_cast<size_t>(buf) * BN * BK_PAD;
#pragma unroll
      for (int kk = 0; kk < BK; kk += 16) {
        if (!warp_live) break;
        const int ar = warp * 16 + r;
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(a + static_cast<size_t>(ar) * BK_PAD + kk + cc);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(a + static_cast<size_t>(ar + 8) * BK_PAD + kk + cc);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(a + static_cast<size_t>(ar) * BK_PAD + kk + cc + 8);
        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(a + static_cast<size_t>(ar + 8) * BK_PAD + kk + cc + 8);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int bn = j * 8 + r;
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(b + static_cast<size_t>(bn) * BK_PAD + kk + cc);
          const uint32_t b1 = *reinterpret_cast<const uint32_t*>(b + static_cast<size_t>(bn) * BK_PAD + kk + cc + 8);
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
              : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
              : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
      }
    }
    wait<0>();
    __syncthreads();  // the ring is free for the next m-tile's prologue

    const int row_lo = warp * 16 + r;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int gcol = n0 + j * 8 + cc;
      auto st = [&](int row_off, int col_off, float val) {
        const int mm = row_lo + row_off;
        if (mm < m_rows && gcol + col_off < n)
          fp8_gemv::store_dot(out + static_cast<size_t>(seg.row0 + m0 + mm) * out_stride + gcol + col_off,
                              __fdiv_rn(val, g));
      };
      st(0, 0, acc[j][0]);
      st(0, 1, acc[j][1]);
      st(8, 0, acc[j][2]);
      st(8, 1, acc[j][3]);
    }
  }
}

// ---- the ldmatrix fp4 tile kernel (phase 5, 2026-09-08 evening) ----------
// Hardware counters on the tile kernels above: the reference (3 blocks/SM)
// stalls on loads and barriers at 34 % issue, the 64 x 128 x 32 pipelined
// form on the MIO queue (20 LDS.32 per eight mma), the three-stage ring on
// nothing but its lone block's eight warps. This kernel keeps every
// operand's path short and two blocks resident:
//   * 64 x 128 x 64 tiles, eight warps as 2 (m32) x 4 (n32): a warp's k16
//     step is two ldmatrix.x4 for A and eight mma — no LDS.32 fan-out;
//   * a three-slot cp.async ring of the activation tile (16-byte copies)
//     and the weight tile's RAW fp4 codes and scales (16- and 4-byte
//     copies) — 46.5 KB, two blocks per SM; no decoded bf16 tile exists:
//     the B fragments are decoded at fragment time straight from the raw
//     codes (one LDS.64 of a row's 16-code group, the two bytes this lane
//     needs, cvt e2m1x2 -> f16x2, x scale in f16 — exact, <= 5 significant
//     bits — then to bf16x2, also exact), one barrier per stage.
// Every output element is still the same ascending-k16 mma.sync chain over
// the same exact bf16 weights, so the outputs are bitwise the reference's.
namespace fp4_ldm {
constexpr int BM = 64, BN = 128, BK = 64;
constexpr int kThreads = 256, kStages = 3;
constexpr int kPrefetchLines = 1;  // weight-row lines (128 B = 4 stages) prefetched into L2 ahead of the ring
constexpr int A_PAD = BK + 8;                          // 144-byte rows: ldmatrix conflict-free
constexpr int kRawStride = 48;                         // bytes per raw code row (32 used; 16-aligned, conflict-free LDS.64)
constexpr int kGroupsPerRow = BK / 16;                 // 4 scale bytes per row per stage
constexpr size_t kABytes = size_t(BM) * A_PAD * 2;     // 9,216
constexpr size_t kCodeBytes = size_t(BN) * kRawStride; // 6,144
constexpr size_t kScaleBytes = size_t(BN) * kGroupsPerRow;  // 512
constexpr size_t kSlotBytes = kABytes + kCodeBytes + kScaleBytes;  // 15,872
constexpr size_t kSmem = kStages * kSlotBytes;         // 47,616: two blocks per SM
static_assert(BM * (BK / 8) == 2 * kThreads, "two activation chunks per thread");
static_assert(BN * 2 == kThreads, "one 16-byte code copy per thread (two per row)");
static_assert(A_PAD * 2 % 16 == 0 && kRawStride % 16 == 0, "16-byte aligned rows");

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
// Two e2m1 codes (one byte) times the group's scale (f16, exact) as a
// bf16x2 word: the products carry <= 5 significant bits, so f16, f32 and
// bf16 all hold them exactly.
__device__ __forceinline__ uint32_t decode_pair_bf16(uint32_t byte, __half2 s2) {
  const __half2 h(__nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(byte), __NV_E2M1));
  const __half2 p = __hmul2(h, s2);
  const __nv_bfloat162 b = __floats2bfloat162_rn(__low2float(p), __high2float(p));
  return *reinterpret_cast<const uint32_t*>(&b);
}
// The MXFP4 pair (2026-09-14): the e8m0 scale 2^(byte - 127) can leave
// f16's range, so the product is formed in fp32 — the e2m1 value (exact in
// f16, converted) times the scale as an fp32 power of two (byte 0 = 2^-127,
// a denormal; 255 = NaN, propagated) — exact with <= 2 significant bits,
// and exact again as bf16. No global scale.
__device__ __forceinline__ float e8m0_to_float(uint32_t byte) {
  if (byte == 255u) return __int_as_float(0x7FC00000);
  return byte == 0u ? __uint_as_float(0x00400000u) : __uint_as_float(byte << 23);
}
__device__ __forceinline__ uint32_t decode_pair_bf16_mx(uint32_t byte, float s) {
  const __half2 h(__nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(byte), __NV_E2M1));
  const __nv_bfloat162 b = __floats2bfloat162_rn(__low2float(h) * s, __high2float(h) * s);
  return *reinterpret_cast<const uint32_t*>(&b);
}
// (TRIED 2026-10-02 and reverted: the MXFP4 pair from a 256-entry table in
// shared memory with the scale's exponent added — a load, an and and an
// add for ~30 instructions, bitwise the decode above over every code and
// scale byte. Cold prefill on four nodes 1,327 -> 1,355 ms at 2K and
// 5,394 -> 5,476 at 8K: the kernel is not bound by the decode. Per weight
// byte from DRAM it moves two activation bytes from L2 — a 64-row tile
// re-read by each of an expert's 128-column blocks.)
}  // namespace fp4_ldm

// kGroup: 16 = NVFP4 (e4m3 scales per 16 codes, the global divides the
// finished dot), 32 = MXFP4 (e8m0 per 32, no global; DeepSeek-V4.1-Flash's
// routed experts, 2026-09-14). A stage's 64 k spans 4 or 2 scale bytes per
// row; the ring keeps 4 bytes per row either way.
template <typename OutT, int kGroup>
__global__ __launch_bounds__(fp4_ldm::kThreads, 2) void moe_grouped_mma_fp4_ldm_kernel(
    const uint16_t* __restrict__ act, size_t act_stride, int act_vec,
    const int32_t* __restrict__ act_rows,
    const MoeSegment* __restrict__ segs, const MoeExpertView* __restrict__ views,
    int which, OutT* __restrict__ out, size_t out_stride, int n, int k,
    int m_tiles) {
  using namespace fp4_ldm;
  static_assert(kGroup == 16 || kGroup == 32, "the fp4 scale group");
  constexpr int kGroupsPerStage = BK / kGroup;  // scale bytes per row per stage: 4 or 2
  using fp4_pipe3::cp_async;
  using fp4_pipe3::commit;
  using fp4_pipe3::wait;
  extern __shared__ __align__(16) uint8_t smem[];
  const MoeSegment seg = segs[blockIdx.y];
  // One 64-row m-tile per block, the m-tile the FASTEST grid index: a long
  // segment's m-tiles are launched together and re-read its weight slice
  // through L2 (with z as the m-tile they ran a whole grid apart and the
  // slice came from DRAM each time: 8K tokens, 2026-09-08); its n-tiles
  // are m_tiles blocks apart and still co-resident for the activations.
  const int m_tile = static_cast<int>(blockIdx.x) % m_tiles;
  const int n_tile = static_cast<int>(blockIdx.x) / m_tiles;
  const int z0 = m_tile * BM;
  if (z0 >= seg.rows) return;  // beyond this segment's rows (no barrier yet)
  const int z1 = min(seg.rows, z0 + BM);
  const MoeExpertView v = views[seg.expert * 3 + which];
  const float g = kGroup == kFp4Group ? *v.fp4_global : 1.f;
  const int n0 = n_tile * BN;
  const size_t payload_stride = static_cast<size_t>(k) / 2;
  const size_t scale_stride = static_cast<size_t>(k) / kGroup;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32, lane = tid % 32;
  // Warps 1 (m64) x 8 (n16): a warp owns two n8 tiles over all 64 rows, so
  // every B fragment is decoded once per block and feeds four MMAs.
  const int wn = warp;
  const int r = lane / 4, cc = (lane % 4) * 2;
  const uint32_t byte_sel = static_cast<uint32_t>(lane % 4);  // __byte_perm selector: this lane's byte
  // Weight tile copies: thread t -> n-row t/2, 16-byte half t%2 of the
  // row's 32 code bytes; threads < BN copy their row's four scale bytes.
  const int b_row = tid / 2, b_half = tid % 2;
  const int gn = n0 + b_row;
  const bool b_ok = gn < n;
  const uint8_t* b_codes = v.payload + static_cast<size_t>(b_ok ? gn : 0) * payload_stride;
  const int s_row = tid;  // < BN
  const bool s_ok = tid < BN && n0 + tid < n;
  const uint8_t* s_scales = v.fp4_scales + static_cast<size_t>(s_ok ? n0 + tid : 0) * scale_stride;
  const int stages = (k + BK - 1) / BK;
  // cp.async needs its width's alignment at every row: 16-byte code copies
  // need k % 32 == 0 (else two 8-byte ones — k % 16 == 0 keeps those
  // aligned), 4-byte scale copies need k % 64 == 0 (else the row's bytes
  // are gathered by plain loads). The production shapes (k = 4096, 512)
  // take the wide copies; glm_moe_test's k = 208 / 320 the fallbacks.
  const bool codes16 = (payload_stride % 16) == 0;
  const bool scales4 = kGroup == kFp4Group && (scale_stride % 4) == 0;  // MX: two bytes, plain loads

  auto slotA = [&](int slot) {
    return reinterpret_cast<uint16_t*>(smem + static_cast<size_t>(slot) * kSlotBytes);
  };
  auto slotCodes = [&](int slot) { return smem + static_cast<size_t>(slot) * kSlotBytes + kABytes; };
  auto slotScales = [&](int slot) {
    return smem + static_cast<size_t>(slot) * kSlotBytes + kABytes + kCodeBytes;
  };

  for (int m0 = z0; m0 < z1; m0 += BM) {
    const int m_rows = min(BM, z1 - m0);
    // m16 slabs past the segment's end are never stored: their MMAs are
    // skipped (block-uniform), the loads and barriers kept.
    const int live_slabs = (m_rows + 15) / 16;
    float acc[4][2][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    // Stage s into ring slot `slot`: two 16-byte activation chunks per
    // thread (rows tid/8 and tid/8 + 32, 8-element chunk tid%8), one 16-byte
    // code copy, and the row's scale bytes. Absent rows and k past the end
    // arrive as zeros (src-size 0). A misaligned activation base takes a
    // plain copy.
    auto issue = [&](int s, int slot) {
      const int k0 = s * BK;
      uint16_t* a = slotA(slot);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int mm = tid / 8 + 32 * i, kq = (tid % 8) * 8;
        const int gk = k0 + kq;
        const bool in = mm < m_rows && gk < k;
        uint16_t* dst = a + static_cast<size_t>(mm) * A_PAD + kq;
        const uint16_t* src = act;
        if (in) {
          const int srow = seg.row0 + m0 + mm;
          src = act + static_cast<size_t>(act_rows != nullptr ? act_rows[srow] : srow) * act_stride + gk;
        }
        if (act_vec != 0) {
          // The row's 128-byte line for this stage, tagged evict_last as it
          // enters L2 (the segment's other n-tile blocks re-read it while
          // the weight stream passes through); then the copy.
          if (in && (tid % 8) == 0)
            asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(src));
          cp_async(dst, src, 16, in ? 16 : 0);
        } else {
          uint16_t e[8];
#pragma unroll
          for (int h = 0; h < 8; ++h) e[h] = in ? src[h] : static_cast<uint16_t>(0);
          *reinterpret_cast<uint4*>(dst) = make_uint4(e[0] | (e[1] << 16), e[2] | (e[3] << 16),
                                                      e[4] | (e[5] << 16), e[6] | (e[7] << 16));
        }
      }
      {
        // 16 bytes = 32 codes = two 16-code groups; k % 16 == 0 so a group is
        // in or out whole: src-size 16, 8 or 0.
        const int gk = k0 + b_half * 32;
        const int in_codes = b_ok ? max(0, min(32, k - gk)) : 0;
        uint8_t* cdst = slotCodes(slot) + static_cast<size_t>(b_row) * kRawStride + b_half * 16;
        if (codes16) {
          cp_async(cdst, in_codes > 0 ? b_codes + gk / 2 : v.payload, 16, in_codes / 2);
        } else {
#pragma unroll
          for (int q = 0; q < 2; ++q) {
            const int in_q = max(0, min(16, in_codes - 16 * q));
            cp_async(cdst + 8 * q, in_q > 0 ? b_codes + gk / 2 + 8 * q : v.payload, 8, in_q / 2);
          }
        }
        // A stage reads 32 bytes of each weight row, rows 2 KB apart: the
        // DRAM sees quarter-line requests. Pull the row's NEXT 128-byte line
        // (the codes of stages s+4 .. s+7) into L2 as a whole line, two
        // stages ahead of the ring's own copies.
        if (b_half == 0 && b_ok && (s % 4) == 0) {
#pragma unroll
          for (int l = (s == 0 ? 1 : kPrefetchLines); l <= kPrefetchLines; ++l) {
            const int nk = k0 + 4 * BK * l;
            if (nk < k) {
              const uint8_t* line = b_codes + nk / 2;
              asm volatile("prefetch.global.L2 [%0];" ::"l"(line));
            }
          }
        }
      }
      if (tid < BN) {
        const int in_groups = s_ok ? max(0, min(kGroupsPerStage, (k - k0) / kGroup)) : 0;
        uint8_t* sdst = slotScales(slot) + static_cast<size_t>(s_row) * kGroupsPerRow;
        if (scales4) {
          cp_async(sdst, in_groups > 0 ? s_scales + k0 / kGroup : v.fp4_scales, 4, in_groups);
        } else {
          uint32_t w = 0;
#pragma unroll
          for (int gi = 0; gi < kGroupsPerStage; ++gi)
            if (gi < in_groups) w |= static_cast<uint32_t>(s_scales[k0 / kGroup + gi]) << (8 * gi);
          *reinterpret_cast<uint32_t*>(sdst) = w;
        }
        // The scales' next 128-byte line likewise (32 stages at 4 bytes per stage, 64 at 2).
        if (s_ok && (s % (128 / kGroupsPerStage)) == 0) {
          const int nk = k0 + (128 / kGroupsPerStage) * BK;
          if (nk < k) {
            const uint8_t* line = s_scales + nk / kGroup;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(line));
          }
        }
      }
    };

    // Prologue: stages 0 .. kStages-2 in flight.
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
      if (s < stages) issue(s, s);
      commit();
    }
    for (int s = 0; s < stages; ++s) {
      const int slot = s % kStages;
      wait<kStages - 2>();  // stage s landed for this thread
      __syncthreads();      // ... for every thread; and everyone is past stage s-1's multiply
      if (s + kStages - 1 < stages) issue(s + kStages - 1, (s + kStages - 1) % kStages);
      commit();
      const uint16_t* a = slotA(slot);
      const uint8_t* codes = slotCodes(slot);
      const uint8_t* scales = slotScales(slot);
      // This warp's two n8 rows' scale bytes for the stage's four groups.
      uint32_t sc4[2];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int bn = wn * 16 + j * 8 + r;
        sc4[j] = *reinterpret_cast<const uint32_t*>(scales + static_cast<size_t>(bn) * kGroupsPerRow);
      }
#pragma unroll
      for (int kk = 0; kk < BK; kk += 16) {
        // B fragments of the warp's two n8 tiles at this k16, decoded from
        // the raw codes: row n's 8 group bytes hold its k16 codes; this
        // lane's are bytes lane%4 (k = cc, cc+1) and lane%4 + 4 (k = cc+8,
        // cc+9) of them.
        uint32_t bfrag[2][2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int bn = wn * 16 + j * 8 + r;
          const uint2 w = *reinterpret_cast<const uint2*>(
              codes + static_cast<size_t>(bn) * kRawStride + (kk / 16) * 8);
          const uint8_t sc = static_cast<uint8_t>((sc4[j] >> (8 * (kk / kGroup))) & 0xFFu);
          const uint32_t byte0 = __byte_perm(w.x, 0u, byte_sel);
          const uint32_t byte1 = __byte_perm(w.y, 0u, byte_sel);
          if constexpr (kGroup == kFp4Group) {
            const __half2 s2 = __half2half2(__half(__nv_cvt_fp8_to_halfraw(
                static_cast<__nv_fp8_storage_t>(sc), __NV_E4M3)));
            bfrag[j][0] = decode_pair_bf16(byte0, s2);
            bfrag[j][1] = decode_pair_bf16(byte1, s2);
          } else {
            const float s = e8m0_to_float(sc);
            bfrag[j][0] = decode_pair_bf16_mx(byte0, s);
            bfrag[j][1] = decode_pair_bf16_mx(byte1, s);
          }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          if (i >= live_slabs) break;
          uint32_t af[4];
          const int arow = i * 16 + (lane % 16);
          ldmatrix_x4(af, a + static_cast<size_t>(arow) * A_PAD + kk + (lane / 16) * 8);
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
                : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bfrag[j][0]),
                  "r"(bfrag[j][1]));
          }
        }
      }
    }
    wait<0>();
    __syncthreads();  // the ring is free for the next m-tile's prologue

    // Epilogue: the warp's [64 x 16] slice; ÷g once per finished dot.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row_lo = i * 16 + r;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int gcol = n0 + wn * 16 + j * 8 + cc;
        auto st = [&](int row_off, int col_off, float val) {
          const int mm = row_lo + row_off;
          if (mm < m_rows && gcol + col_off < n)
            fp8_gemv::store_dot(out + static_cast<size_t>(seg.row0 + m0 + mm) * out_stride + gcol + col_off,
                                kGroup == kFp4Group ? __fdiv_rn(val, g) : val);
        };
        st(0, 0, acc[i][j][0]);
        st(0, 1, acc[i][j][1]);
        st(8, 0, acc[i][j][2]);
        st(8, 1, acc[i][j][3]);
      }
    }
  }
}

template <typename OutT>
void launch_moe_grouped_mma_fp4_ldm(const uint16_t* act, size_t act_stride,
                                    const int32_t* act_rows, const MoeSegment* segs,
                                    int n_segs, int max_rows, int rows_per_block,
                                    const MoeExpertView* views, int which, OutT* out,
                                    size_t out_stride, int n, int k, cudaStream_t stream,
                                    int fp4_group = kFp4Group) {
  using namespace fp4_ldm;
  if (n_segs <= 0 || n <= 0) return;
  if (!act || !segs || !views || !out)
    throw std::invalid_argument("moe grouped mma fp4 ldm: null pointer");
  if (fp4_group != kFp4Group && fp4_group != kMxfp4Group)
    throw std::invalid_argument("moe grouped mma fp4 ldm: the fp4 scale group must be 16 or 32");
  check_fp4_mma_shape(k, "moe grouped mma fp4 ldm", fp4_group);
  // rows_per_block is accepted for the family's signature; this kernel
  // always runs one 64-row m-tile per block (see the kernel).
  (void)rows_per_block;
  if (max_rows <= 0) throw std::invalid_argument("moe grouped mma fp4 ldm: max_rows");
  const int m_tiles = (max_rows + BM - 1) / BM;
  const int act_vec =
      (reinterpret_cast<uintptr_t>(act) % 16 == 0 && (act_stride % 8) == 0) ? 1 : 0;
  static bool opted_in[2] = {false, false};  // the dynamic smem opt-in, once per OutT and group
  const int gi = fp4_group == kFp4Group ? 0 : 1;
  if (!opted_in[gi]) {
    DGPP_CUDA_OK(cudaFuncSetAttribute(gi == 0 ? moe_grouped_mma_fp4_ldm_kernel<OutT, kFp4Group>
                                              : moe_grouped_mma_fp4_ldm_kernel<OutT, kMxfp4Group>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(kSmem)));
    opted_in[gi] = true;
  }
  const unsigned n_tiles = static_cast<unsigned>((n + BN - 1) / BN);
  if (static_cast<size_t>(n_tiles) * m_tiles > 0x7fffffffu)
    throw std::invalid_argument("moe grouped mma fp4 ldm: grid too large");
  const dim3 grid(n_tiles * static_cast<unsigned>(m_tiles), static_cast<unsigned>(n_segs), 1u);
  if (gi == 0)
    moe_grouped_mma_fp4_ldm_kernel<OutT, kFp4Group><<<grid, kThreads, kSmem, stream>>>(
        act, act_stride, act_vec, act_rows, segs, views, which, out, out_stride, n, k, m_tiles);
  else
    moe_grouped_mma_fp4_ldm_kernel<OutT, kMxfp4Group><<<grid, kThreads, kSmem, stream>>>(
        act, act_stride, act_vec, act_rows, segs, views, which, out, out_stride, n, k, m_tiles);
  DGPP_CUDA_OK(cudaGetLastError());
}

// ---- the ldmatrix fp8 tile kernel (2026-09-08 evening) ---------------------
// The fp4 kernel above for FP8 block-scaled weights (the FP8 checkpoint's
// routed experts, both checkpoints' shared experts): the same 64 x 128
// tiles, 1 x 8 warps, cp.async ring, L2 line prefetch, evict_last
// activations and m-tile-fastest grid, with 32-deep stages (fp8 codes are
// twice the bytes; four 11 KB slots keep two blocks per SM), the stage's
// one block scale (128 x 128 blocks: n0 % 128 == 0, k0 / 128) copied into
// the slot with the tile, and the B fragments decoded at fragment time:
// e4m3x2 -> f16x2 (exact) -> f32 (exact) x scale (RN) -> bf16 (RN) — the
// reference tile kernel's float_to_bf16_bits(fp8_to_float(code) * s) op
// for op, so the outputs are bitwise the reference's.
namespace fp8_ldm {
constexpr int BM = 64, BN = 128, BK = 32;
constexpr int kThreads = 256, kStages = 4;
constexpr int A_PAD = BK + 8;                             // 80-byte rows: ldmatrix conflict-free
constexpr int kRawStride = 48;                            // bytes per code row (32 used; conflict-free u16 loads)
constexpr size_t kABytes = size_t(BM) * A_PAD * 2;        // 5,120
constexpr size_t kCodeBytes = size_t(BN) * kRawStride;    // 6,144
// One f32 scale PER ROW of the n-tile (2026-09-09, the Qwen slices'
// re-blocked grid, plan D2): a 128-row tile spans one 128-row scale block
// (GLM: every row's scale the same value — bitwise the one-scalar form),
// two 64-row blocks (TP=2) or four 32-row blocks (TP=4); the k axis keeps
// one scale column per 32-deep stage (cs >= 5).
constexpr size_t kScaleBytes = size_t(BN) * 4;            // 512
constexpr size_t kSlotBytes = kABytes + kCodeBytes + kScaleBytes;  // 11,776
constexpr size_t kSmem = kStages * kSlotBytes;            // 47,104: two blocks per SM
static_assert(BM * (BK / 8) == kThreads, "one activation chunk per thread");
static_assert(BN * (BK / 16) == kThreads, "one 16-byte code copy per thread (two per row)");
static_assert(A_PAD * 2 % 16 == 0 && kRawStride % 16 == 0 && kSlotBytes % 16 == 0, "16-byte aligned");
static_assert(BK == kFp8LdmBK, "the launcher's width test matches the kernel's stage");

// Two e4m3 codes (one u16, k and k+1 of a row) times the block scale as a
// bf16x2 word, the reference's roundings exactly.
__device__ __forceinline__ uint32_t decode_pair_bf16(uint32_t two, float s) {
  const __half2 h(__nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two), __NV_E4M3));
  const __nv_bfloat162 b = __floats2bfloat162_rn(__low2float(h) * s, __high2float(h) * s);
  return *reinterpret_cast<const uint32_t*>(&b);
}
}  // namespace fp8_ldm

template <typename OutT>
__global__ __launch_bounds__(fp8_ldm::kThreads, 2) void moe_grouped_mma_fp8_ldm_kernel(
    const uint16_t* __restrict__ act, size_t act_stride, int act_vec,
    const int32_t* __restrict__ act_rows,
    const MoeSegment* __restrict__ segs, const MoeExpertView* __restrict__ views,
    int which, OutT* __restrict__ out, size_t out_stride, int n, int k,
    int m_tiles) {
  using namespace fp8_ldm;
  using fp4_pipe3::cp_async;
  using fp4_pipe3::commit;
  using fp4_pipe3::wait;
  using fp4_ldm::ldmatrix_x4;
  extern __shared__ __align__(16) uint8_t smem[];
  const MoeSegment seg = segs[blockIdx.y];
  const int m_tile = static_cast<int>(blockIdx.x) % m_tiles;
  const int n_tile = static_cast<int>(blockIdx.x) / m_tiles;
  const int z0 = m_tile * BM;
  if (z0 >= seg.rows) return;  // beyond this segment's rows (no barrier yet)
  const int m_rows = min(BM, seg.rows - z0);
  const MoeExpertView v = views[seg.expert * 3 + which];
  const int n0 = n_tile * BN;
  const int rs = v.scale_shift_rows, cs = v.scale_shift_cols;
  const int scale_cols = (k + (1 << cs) - 1) >> cs;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32, lane = tid % 32;
  const int wn = warp;  // 1 (m64) x 8 (n16)
  const int r = lane / 4, cc = (lane % 4) * 2;
  // Copies: activation chunk (row tid/4, 8 elements (tid%4)*8); codes row
  // tid/2, 16-byte half tid%2; the row's scale by the half-0 thread.
  const int a_row = tid / 4, a_kq = (tid % 4) * 8;
  const int b_row = tid / 2, b_half = tid % 2;
  const int gn = n0 + b_row;
  const bool b_ok = gn < n;
  const uint8_t* b_codes = v.payload + static_cast<size_t>(b_ok ? gn : 0) * k;
  const float* b_scales = v.scales + (static_cast<size_t>(b_ok ? gn : 0) >> rs) * scale_cols;
  const bool a_ok = a_row < m_rows;
  const uint16_t* a_src = act;
  if (a_ok) {
    const int srow = seg.row0 + z0 + a_row;
    a_src = act + static_cast<size_t>(act_rows != nullptr ? act_rows[srow] : srow) * act_stride;
  }
  const int stages = k / BK;  // k % 32 == 0 (the launcher's contract)
  const int live_slabs = (m_rows + 15) / 16;

  auto slotA = [&](int slot) { return reinterpret_cast<uint16_t*>(smem + static_cast<size_t>(slot) * kSlotBytes); };
  auto slotCodes = [&](int slot) { return smem + static_cast<size_t>(slot) * kSlotBytes + kABytes; };
  auto slotScale = [&](int slot) {
    return reinterpret_cast<float*>(smem + static_cast<size_t>(slot) * kSlotBytes + kABytes + kCodeBytes);
  };
  auto issue = [&](int s, int slot) {
    const int k0 = s * BK;
    {
      uint16_t* dst = slotA(slot) + static_cast<size_t>(a_row) * A_PAD + a_kq;
      const uint16_t* src = a_ok ? a_src + k0 + a_kq : act;
      if (act_vec != 0) {
        if (a_ok && (tid % 4) == 0)
          asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(src));
        cp_async(dst, src, 16, a_ok ? 16 : 0);
      } else {
        uint16_t e[8];
#pragma unroll
        for (int h = 0; h < 8; ++h) e[h] = a_ok ? src[h] : static_cast<uint16_t>(0);
        *reinterpret_cast<uint4*>(dst) = make_uint4(e[0] | (e[1] << 16), e[2] | (e[3] << 16),
                                                    e[4] | (e[5] << 16), e[6] | (e[7] << 16));
      }
    }
    {
      // 16 codes = 16 bytes per half; k % 32 == 0 so a stage is in whole.
      cp_async(slotCodes(slot) + static_cast<size_t>(b_row) * kRawStride + b_half * 16,
               b_ok ? b_codes + k0 + b_half * 16 : v.payload, 16, b_ok ? 16 : 0);
      // The row's next 128-byte line of codes (stages s+4 .. s+7) into L2.
      if (b_half == 0 && b_ok && (s % 4) == 0) {
        const int nk = k0 + 4 * BK;
        if (nk < k) asm volatile("prefetch.global.L2 [%0];" ::"l"(b_codes + nk));
      }
    }
    if (b_half == 0) cp_async(slotScale(slot) + b_row, b_scales + (k0 >> cs), 4, 4);
  };

  float acc[4][2][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < stages) issue(s, s);
    commit();
  }
  for (int s = 0; s < stages; ++s) {
    const int slot = s % kStages;
    wait<kStages - 2>();
    __syncthreads();
    if (s + kStages - 1 < stages) issue(s + kStages - 1, (s + kStages - 1) % kStages);
    commit();
    const uint16_t* a = slotA(slot);
    const uint8_t* codes = slotCodes(slot);
    const float* sc = slotScale(slot);
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t bfrag[2][2];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int brow = wn * 16 + j * 8 + r;
        const uint8_t* row = codes + static_cast<size_t>(brow) * kRawStride + kk;
        const float s_row = sc[brow];
        const uint32_t lo = *reinterpret_cast<const uint16_t*>(row + cc);      // k = cc, cc+1
        const uint32_t hi = *reinterpret_cast<const uint16_t*>(row + cc + 8);  // k = cc+8, cc+9
        bfrag[j][0] = decode_pair_bf16(lo, s_row);
        bfrag[j][1] = decode_pair_bf16(hi, s_row);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (i >= live_slabs) break;
        uint32_t af[4];
        ldmatrix_x4(af, a + static_cast<size_t>(i * 16 + (lane % 16)) * A_PAD + kk + (lane / 16) * 8);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
              : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
              : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bfrag[j][0]), "r"(bfrag[j][1]));
        }
      }
    }
  }
  wait<0>();

  // Epilogue: the warp's [64 x 16] slice, the accumulator stored as is (the
  // block scale was applied in the decode, as the reference does).
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row_lo = i * 16 + r;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int gcol = n0 + wn * 16 + j * 8 + cc;
      auto st = [&](int row_off, int col_off, float val) {
        const int mm = row_lo + row_off;
        if (mm < m_rows && gcol + col_off < n)
          fp8_gemv::store_dot(out + static_cast<size_t>(seg.row0 + z0 + mm) * out_stride + gcol + col_off, val);
      };
      st(0, 0, acc[i][j][0]);
      st(0, 1, acc[i][j][1]);
      st(8, 0, acc[i][j][2]);
      st(8, 1, acc[i][j][3]);
    }
  }
}

// DGPP_MOE_FP8_SMALLK=on runs the small-k form below for k <= 256 (the
// prefill's down projection at k = 160); off by default until measured.
bool moe_fp8_smallk_enabled() {
  static const bool on = [] {
    const char* v = std::getenv("DGPP_MOE_FP8_SMALLK");
    return v != nullptr && std::string(v) == "on";
  }();
  return on;
}

// The small-k form: the down projection's k is 160 bytes at
// TP=4 — five 32-deep stages, so in the one-tile kernel a block's
// prologue, epilogue and per-stage barriers are most of its life (the
// prefill profile: 669 us per down launch against 370 for a gate/up
// launch of the same FLOPs). Here one block stages its 64 x k activation
// tile ONCE, resident, and walks kNT n-tiles through a codes-only ring as
// one flat sequence of (n-tile, stage) copies, storing each n-tile's
// accumulators when its last stage lands — the same stages, fragments and
// MMA sequence per output element as the one-tile kernel, so bitwise.
// A tile 21.5 KB + a four-slot codes ring 26.6 KB: under the 48 KB default,
// two blocks per SM.
namespace fp8_ldm_smallk {
constexpr int kNT = 4;
constexpr int kMaxK = 256;
constexpr int kStages = 4;
constexpr size_t kSlotBytes = fp8_ldm::kCodeBytes + fp8_ldm::kScaleBytes;  // 6,656
// The resident activation tile's pitch is k + 8 elements (a 336-byte row at
// k = 160: not a multiple of 128, so ldmatrix stays conflict-free) and its
// bytes are the launch's: 21.5 KB at k = 160 plus the 26.6 KB ring = 48.1
// KB, two blocks per SM; at the k = 256 ceiling 33.8 KB and one block.
__host__ __device__ constexpr int a_pitch(int k) { return k + 8; }
__host__ __device__ constexpr size_t a_bytes(int k) {
  return static_cast<size_t>(fp8_ldm::BM) * static_cast<size_t>(a_pitch(k)) * 2;
}
__host__ __device__ constexpr size_t smem_bytes(int k) { return a_bytes(k) + kStages * kSlotBytes; }
static_assert(kSlotBytes % 16 == 0, "16-byte aligned");
}  // namespace fp8_ldm_smallk

template <typename OutT>
__global__ __launch_bounds__(fp8_ldm::kThreads, 2) void moe_grouped_mma_fp8_ldm_smallk_kernel(
    const uint16_t* __restrict__ act, size_t act_stride, int act_vec,
    const int32_t* __restrict__ act_rows,
    const MoeSegment* __restrict__ segs, const MoeExpertView* __restrict__ views,
    int which, OutT* __restrict__ out, size_t out_stride, int n, int k,
    int m_tiles, int n_groups) {
  using namespace fp8_ldm;
  using fp8_ldm_smallk::kNT;
  using fp4_pipe3::cp_async;
  using fp4_pipe3::commit;
  using fp4_pipe3::wait;
  using fp4_ldm::ldmatrix_x4;
  constexpr int kRing = fp8_ldm_smallk::kStages;
  const int kPitch = fp8_ldm_smallk::a_pitch(k);
  const size_t a_bytes = fp8_ldm_smallk::a_bytes(k);
  extern __shared__ __align__(16) uint8_t smem[];
  const MoeSegment seg = segs[blockIdx.y];
  const int m_tile = static_cast<int>(blockIdx.x) % m_tiles;
  const int n_group = static_cast<int>(blockIdx.x) / m_tiles;
  const int z0 = m_tile * BM;
  if (z0 >= seg.rows) return;  // beyond this segment's rows (no barrier yet)
  const int m_rows = min(BM, seg.rows - z0);
  const MoeExpertView v = views[seg.expert * 3 + which];
  const int rs = v.scale_shift_rows, cs = v.scale_shift_cols;
  const int scale_cols = (k + (1 << cs) - 1) >> cs;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32, lane = tid % 32;
  const int wn = warp;  // 1 (m64) x 8 (n16)
  const int r = lane / 4, cc = (lane % 4) * 2;
  const int b_row = tid / 2, b_half = tid % 2;
  const int stages = k / BK;
  const int n_tiles_total = (n + BN - 1) / BN;
  const int nt0 = n_group * kNT;
  const int live_nt = min(kNT, n_tiles_total - nt0);
  const int live_slabs = (m_rows + 15) / 16;

  uint16_t* a_tile = reinterpret_cast<uint16_t*>(smem);
  auto slotCodes = [&](int slot) {
    return smem + a_bytes + static_cast<size_t>(slot) * fp8_ldm_smallk::kSlotBytes;
  };
  auto slotScale = [&](int slot) {
    return reinterpret_cast<float*>(smem + a_bytes +
                                    static_cast<size_t>(slot) * fp8_ldm_smallk::kSlotBytes + kCodeBytes);
  };

  // The activation tile, resident: 64 rows x k, 16-byte chunks, one per
  // thread per pass (256 chunks = 64 rows x 4 chunks of 8 elements).
  {
    const int chunks_per_row = k / 8;
    const int total = BM * chunks_per_row;
    for (int c = tid; c < total; c += kThreads) {
      const int row = c / chunks_per_row, kq = (c % chunks_per_row) * 8;
      const bool ok = row < m_rows;
      uint16_t* dst = a_tile + static_cast<size_t>(row) * kPitch + kq;
      const uint16_t* src = act;
      if (ok) {
        const int srow = seg.row0 + z0 + row;
        src = act + static_cast<size_t>(act_rows != nullptr ? act_rows[srow] : srow) * act_stride + kq;
      }
      if (act_vec != 0) {
        cp_async(dst, src, 16, ok ? 16 : 0);
      } else {
        uint16_t e[8];
#pragma unroll
        for (int h = 0; h < 8; ++h) e[h] = ok ? src[h] : static_cast<uint16_t>(0);
        *reinterpret_cast<uint4*>(dst) = make_uint4(e[0] | (e[1] << 16), e[2] | (e[3] << 16),
                                                    e[4] | (e[5] << 16), e[6] | (e[7] << 16));
      }
    }
    commit();
  }

  // One n-tile's codes for one stage into a ring slot.
  auto issue = [&](int flat, int slot) {
    const int nt = flat / stages, s = flat % stages;
    const int k0 = s * BK;
    const int gn = (nt0 + nt) * BN + b_row;
    const bool b_ok = gn < n;
    const uint8_t* b_codes = v.payload + static_cast<size_t>(b_ok ? gn : 0) * k;
    const float* b_scales = v.scales + (static_cast<size_t>(b_ok ? gn : 0) >> rs) * scale_cols;
    cp_async(slotCodes(slot) + static_cast<size_t>(b_row) * kRawStride + b_half * 16,
             b_ok ? b_codes + k0 + b_half * 16 : v.payload, 16, b_ok ? 16 : 0);
    if (b_half == 0) cp_async(slotScale(slot) + b_row, b_scales + (k0 >> cs), 4, 4);
  };

  const int flat_total = live_nt * stages;
  float acc[4][2][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

#pragma unroll
  for (int f = 0; f < kRing - 1; ++f) {
    if (f < flat_total) issue(f, f);
    commit();
  }
  for (int f = 0; f < flat_total; ++f) {
    const int slot = f % kRing;
    wait<kRing - 2>();  // this flat stage's copies (and the A tile, committed first) have landed
    __syncthreads();
    if (f + kRing - 1 < flat_total) issue(f + kRing - 1, (f + kRing - 1) % kRing);
    commit();
    const int nt = f / stages, s = f % stages;
    const int k0 = s * BK;
    const uint8_t* codes = slotCodes(slot);
    const float* sc = slotScale(slot);
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t bfrag[2][2];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int brow = wn * 16 + j * 8 + r;
        const uint8_t* row = codes + static_cast<size_t>(brow) * kRawStride + kk;
        const float s_row = sc[brow];
        const uint32_t lo = *reinterpret_cast<const uint16_t*>(row + cc);
        const uint32_t hi = *reinterpret_cast<const uint16_t*>(row + cc + 8);
        bfrag[j][0] = decode_pair_bf16(lo, s_row);
        bfrag[j][1] = decode_pair_bf16(hi, s_row);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (i >= live_slabs) break;
        uint32_t af[4];
        ldmatrix_x4(af, a_tile + static_cast<size_t>(i * 16 + (lane % 16)) * kPitch + k0 + kk + (lane / 16) * 8);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
              : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
              : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bfrag[j][0]), "r"(bfrag[j][1]));
        }
      }
    }
    if (s == stages - 1) {
      // This n-tile is complete: its epilogue, then a fresh accumulator.
      const int n0 = (nt0 + nt) * BN;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row_lo = i * 16 + r;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int gcol = n0 + wn * 16 + j * 8 + cc;
          auto st = [&](int row_off, int col_off, float val) {
            const int mm = row_lo + row_off;
            if (mm < m_rows && gcol + col_off < n)
              fp8_gemv::store_dot(out + static_cast<size_t>(seg.row0 + z0 + mm) * out_stride + gcol + col_off, val);
          };
          st(0, 0, acc[i][j][0]);
          st(0, 1, acc[i][j][1]);
          st(8, 0, acc[i][j][2]);
          st(8, 1, acc[i][j][3]);
          acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
        }
      }
    }
  }
  wait<0>();
}

// DGPP_MOE_FP8_SWEEP=on runs the m-sweep form below wherever an expert
// holds more than one m-tile; off by default. Its domain is
// the large prefill chunk — 4x the rows per expert cost 1.9x on the
// one-tile kernel, 1.64x on the sweep — but that chunk measured only 3.6 %
// on an 8 K prefill and is not taken, and at the production 2,048-token
// chunk the sweep is a small loss: ragged routing hands it launches where
// one hot expert has two tiles and every other has one, and one block per
// SM hides less latency there than the one-tile kernel's two (GLM's
// 512-token prefill read 569 vs 544 ms with it on).
bool moe_fp8_sweep_enabled() {
  static const bool on = [] {
    const char* v = std::getenv("DGPP_MOE_FP8_SWEEP");
    return v != nullptr && std::string(v) == "on";
  }();
  return on;
}

// The m-sweep form (2026-09-10): one block per (segment, n-tile, group of
// kMT m-tiles) that copies each k-stage's weight tile ONCE and runs every
// live m-tile of its group under it — where the one-tile kernel above runs
// one block per m-tile and so re-reads the expert's weights once per 64
// rows. At a 2,048-token prefill an expert holds ~40–60 rows (one tile) and
// the two forms are the same traffic; at 4,096 and 8,192 the one-tile form
// re-reads its weights two to four times over (the tile bench: 4x the rows
// cost 1.9x the time at the same weights). Each output element's k chain —
// the stages, the fragments, the MMA sequence — is the one-tile kernel's,
// so the outputs are bitwise. Three 27 KB stages (four A tiles, the codes,
// the scales) and kMT accumulator sets: one block per SM.
namespace fp8_ldm_sweep {
constexpr int kMT = 4;
constexpr int kStages = 3;
constexpr size_t kSlotBytes = size_t(kMT) * fp8_ldm::kABytes + fp8_ldm::kCodeBytes + fp8_ldm::kScaleBytes;
constexpr size_t kSmem = kStages * kSlotBytes;  // 81,408
static_assert(kSlotBytes % 16 == 0, "16-byte aligned slots");
}  // namespace fp8_ldm_sweep

template <typename OutT>
__global__ __launch_bounds__(fp8_ldm::kThreads, 1) void moe_grouped_mma_fp8_ldm_sweep_kernel(
    const uint16_t* __restrict__ act, size_t act_stride, int act_vec,
    const int32_t* __restrict__ act_rows,
    const MoeSegment* __restrict__ segs, const MoeExpertView* __restrict__ views,
    int which, OutT* __restrict__ out, size_t out_stride, int n, int k,
    int m_groups) {
  using namespace fp8_ldm;
  using fp8_ldm_sweep::kMT;
  using fp4_pipe3::cp_async;
  using fp4_pipe3::commit;
  using fp4_pipe3::wait;
  using fp4_ldm::ldmatrix_x4;
  constexpr int kSweepStages = fp8_ldm_sweep::kStages;
  constexpr size_t kSweepSlot = fp8_ldm_sweep::kSlotBytes;
  extern __shared__ __align__(16) uint8_t smem[];
  const MoeSegment seg = segs[blockIdx.y];
  const int m_group = static_cast<int>(blockIdx.x) % m_groups;
  const int n_tile = static_cast<int>(blockIdx.x) / m_groups;
  const int z0 = m_group * kMT * BM;
  if (z0 >= seg.rows) return;  // beyond this segment's rows (no barrier yet)
  const int live_tiles = min(kMT, (seg.rows - z0 + BM - 1) / BM);
  const MoeExpertView v = views[seg.expert * 3 + which];
  const int n0 = n_tile * BN;
  const int rs = v.scale_shift_rows, cs = v.scale_shift_cols;
  const int scale_cols = (k + (1 << cs) - 1) >> cs;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32, lane = tid % 32;
  const int wn = warp;  // 1 (m64) x 8 (n16)
  const int r = lane / 4, cc = (lane % 4) * 2;
  const int a_row = tid / 4, a_kq = (tid % 4) * 8;
  const int b_row = tid / 2, b_half = tid % 2;
  const int gn = n0 + b_row;
  const bool b_ok = gn < n;
  const uint8_t* b_codes = v.payload + static_cast<size_t>(b_ok ? gn : 0) * k;
  const float* b_scales = v.scales + (static_cast<size_t>(b_ok ? gn : 0) >> rs) * scale_cols;
  // Per m-tile: this thread's activation row (or none) and the tile's rows.
  int m_rows[kMT];
  const uint16_t* a_src[kMT];
  bool a_ok[kMT];
#pragma unroll
  for (int mt = 0; mt < kMT; ++mt) {
    const int zt = z0 + mt * BM;
    m_rows[mt] = mt < live_tiles ? min(BM, seg.rows - zt) : 0;
    a_ok[mt] = a_row < m_rows[mt];
    a_src[mt] = act;
    if (a_ok[mt]) {
      const int srow = seg.row0 + zt + a_row;
      a_src[mt] = act + static_cast<size_t>(act_rows != nullptr ? act_rows[srow] : srow) * act_stride;
    }
  }
  const int stages = k / BK;  // k % 32 == 0 (the launcher's contract)

  auto slotA = [&](int slot, int mt) {
    return reinterpret_cast<uint16_t*>(smem + static_cast<size_t>(slot) * kSweepSlot +
                                       static_cast<size_t>(mt) * kABytes);
  };
  auto slotCodes = [&](int slot) {
    return smem + static_cast<size_t>(slot) * kSweepSlot + static_cast<size_t>(kMT) * kABytes;
  };
  auto slotScale = [&](int slot) {
    return reinterpret_cast<float*>(smem + static_cast<size_t>(slot) * kSweepSlot +
                                    static_cast<size_t>(kMT) * kABytes + kCodeBytes);
  };
  auto issue = [&](int s, int slot) {
    const int k0 = s * BK;
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt) {
      if (mt >= live_tiles) break;
      uint16_t* dst = slotA(slot, mt) + static_cast<size_t>(a_row) * A_PAD + a_kq;
      const uint16_t* src = a_ok[mt] ? a_src[mt] + k0 + a_kq : act;
      if (act_vec != 0) {
        if (a_ok[mt] && (tid % 4) == 0)
          asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(src));
        cp_async(dst, src, 16, a_ok[mt] ? 16 : 0);
      } else {
        uint16_t e[8];
#pragma unroll
        for (int h = 0; h < 8; ++h) e[h] = a_ok[mt] ? src[h] : static_cast<uint16_t>(0);
        *reinterpret_cast<uint4*>(dst) = make_uint4(e[0] | (e[1] << 16), e[2] | (e[3] << 16),
                                                    e[4] | (e[5] << 16), e[6] | (e[7] << 16));
      }
    }
    {
      cp_async(slotCodes(slot) + static_cast<size_t>(b_row) * kRawStride + b_half * 16,
               b_ok ? b_codes + k0 + b_half * 16 : v.payload, 16, b_ok ? 16 : 0);
      if (b_half == 0 && b_ok && (s % 4) == 0) {
        const int nk = k0 + 4 * BK;
        if (nk < k) asm volatile("prefetch.global.L2 [%0];" ::"l"(b_codes + nk));
      }
    }
    if (b_half == 0) cp_async(slotScale(slot) + b_row, b_scales + (k0 >> cs), 4, 4);
  };

  float acc[kMT][4][2][4];
#pragma unroll
  for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
        acc[mt][i][j][0] = acc[mt][i][j][1] = acc[mt][i][j][2] = acc[mt][i][j][3] = 0.f;

#pragma unroll
  for (int s = 0; s < kSweepStages - 1; ++s) {
    if (s < stages) issue(s, s);
    commit();
  }
  for (int s = 0; s < stages; ++s) {
    const int slot = s % kSweepStages;
    wait<kSweepStages - 2>();
    __syncthreads();
    if (s + kSweepStages - 1 < stages) issue(s + kSweepStages - 1, (s + kSweepStages - 1) % kSweepStages);
    commit();
    const uint8_t* codes = slotCodes(slot);
    const float* sc = slotScale(slot);
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t bfrag[2][2];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int brow = wn * 16 + j * 8 + r;
        const uint8_t* row = codes + static_cast<size_t>(brow) * kRawStride + kk;
        const float s_row = sc[brow];
        const uint32_t lo = *reinterpret_cast<const uint16_t*>(row + cc);
        const uint32_t hi = *reinterpret_cast<const uint16_t*>(row + cc + 8);
        bfrag[j][0] = decode_pair_bf16(lo, s_row);
        bfrag[j][1] = decode_pair_bf16(hi, s_row);
      }
#pragma unroll
      for (int mt = 0; mt < kMT; ++mt) {
        if (mt >= live_tiles) break;
        const uint16_t* a = slotA(slot, mt);
        const int live_slabs = (m_rows[mt] + 15) / 16;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          if (i >= live_slabs) break;
          uint32_t af[4];
          ldmatrix_x4(af, a + static_cast<size_t>(i * 16 + (lane % 16)) * A_PAD + kk + (lane / 16) * 8);
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[mt][i][j][0]), "+f"(acc[mt][i][j][1]), "+f"(acc[mt][i][j][2]), "+f"(acc[mt][i][j][3])
                : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bfrag[j][0]), "r"(bfrag[j][1]));
          }
        }
      }
    }
  }
  wait<0>();

#pragma unroll
  for (int mt = 0; mt < kMT; ++mt) {
    if (mt >= live_tiles) break;
    const int zt = z0 + mt * BM;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row_lo = i * 16 + r;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int gcol = n0 + wn * 16 + j * 8 + cc;
        auto st = [&](int row_off, int col_off, float val) {
          const int mm = row_lo + row_off;
          if (mm < m_rows[mt] && gcol + col_off < n)
            fp8_gemv::store_dot(out + static_cast<size_t>(seg.row0 + zt + mm) * out_stride + gcol + col_off, val);
        };
        st(0, 0, acc[mt][i][j][0]);
        st(0, 1, acc[mt][i][j][1]);
        st(8, 0, acc[mt][i][j][2]);
        st(8, 1, acc[mt][i][j][3]);
      }
    }
  }
}

template <typename OutT>
void launch_moe_grouped_mma_fp8_ldm(const uint16_t* act, size_t act_stride,
                                    const int32_t* act_rows, const MoeSegment* segs,
                                    int n_segs, int max_rows, const MoeExpertView* views,
                                    int which, OutT* out, size_t out_stride, int n, int k,
                                    cudaStream_t stream) {
  using namespace fp8_ldm;
  const int m_tiles = (max_rows + BM - 1) / BM;
  const int act_vec =
      (reinterpret_cast<uintptr_t>(act) % 16 == 0 && (act_stride % 8) == 0) ? 1 : 0;
  static bool opted_in = false;  // the dynamic smem opt-in, once per OutT
  if (!opted_in) {
    DGPP_CUDA_OK(cudaFuncSetAttribute(moe_grouped_mma_fp8_ldm_kernel<OutT>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(kSmem)));
    opted_in = true;
  }
  const unsigned n_tiles = static_cast<unsigned>((n + BN - 1) / BN);
  if (k <= fp8_ldm_smallk::kMaxK && moe_fp8_smallk_enabled()) {
    // The small-k form: the activation tile resident, kNT n-tiles through
    // a codes-only ring (bitwise the one-tile kernel).
    using fp8_ldm_smallk::kNT;
    const size_t smallk_smem = fp8_ldm_smallk::smem_bytes(k);
    static size_t smallk_opted = 0;
    if (smallk_smem > smallk_opted) {
      DGPP_CUDA_OK(cudaFuncSetAttribute(moe_grouped_mma_fp8_ldm_smallk_kernel<OutT>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(fp8_ldm_smallk::smem_bytes(fp8_ldm_smallk::kMaxK))));
      smallk_opted = fp8_ldm_smallk::smem_bytes(fp8_ldm_smallk::kMaxK);
    }
    const int n_groups = (static_cast<int>(n_tiles) + kNT - 1) / kNT;
    const dim3 grid(static_cast<unsigned>(n_groups) * static_cast<unsigned>(m_tiles),
                    static_cast<unsigned>(n_segs), 1u);
    moe_grouped_mma_fp8_ldm_smallk_kernel<OutT><<<grid, kThreads, smallk_smem, stream>>>(
        act, act_stride, act_vec, act_rows, segs, views, which, out, out_stride, n, k,
        m_tiles, n_groups);
    DGPP_CUDA_OK(cudaGetLastError());
    return;
  }
  if (m_tiles > 1 && moe_fp8_sweep_enabled()) {
    // More than one m-tile somewhere: the sweep form reads each weight tile
    // once per kMT m-tiles instead of once per m-tile (bitwise the one-tile
    // kernel; the tile bench and glm/qwen_moe_test hold both to the
    // reference).
    using fp8_ldm_sweep::kMT;
    static bool sweep_opted = false;
    if (!sweep_opted) {
      DGPP_CUDA_OK(cudaFuncSetAttribute(moe_grouped_mma_fp8_ldm_sweep_kernel<OutT>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(fp8_ldm_sweep::kSmem)));
      sweep_opted = true;
    }
    const int m_groups = (max_rows + kMT * BM - 1) / (kMT * BM);
    const dim3 grid(n_tiles * static_cast<unsigned>(m_groups), static_cast<unsigned>(n_segs), 1u);
    moe_grouped_mma_fp8_ldm_sweep_kernel<OutT><<<grid, kThreads, fp8_ldm_sweep::kSmem, stream>>>(
        act, act_stride, act_vec, act_rows, segs, views, which, out, out_stride, n, k,
        m_groups);
    DGPP_CUDA_OK(cudaGetLastError());
    return;
  }
  const dim3 grid(n_tiles * static_cast<unsigned>(m_tiles), static_cast<unsigned>(n_segs), 1u);
  moe_grouped_mma_fp8_ldm_kernel<OutT><<<grid, kThreads, kSmem, stream>>>(
      act, act_stride, act_vec, act_rows, segs, views, which, out, out_stride, n, k, m_tiles);
  DGPP_CUDA_OK(cudaGetLastError());
}

template <typename OutT>
void launch_moe_grouped_mma_fp4_pipe3(const uint16_t* act, size_t act_stride,
                                      const int32_t* act_rows, const MoeSegment* segs,
                                      int n_segs, int max_rows, int rows_per_block,
                                      const MoeExpertView* views, int which, OutT* out,
                                      size_t out_stride, int n, int k, cudaStream_t stream) {
  using namespace fp4_pipe3;
  if (n_segs <= 0 || n <= 0) return;
  if (!act || !segs || !views || !out)
    throw std::invalid_argument("moe grouped mma fp4 pipe3: null pointer");
  check_fp4_mma_shape(k, "moe grouped mma fp4 pipe3");
  if (rows_per_block > 0 && (rows_per_block % BM) != 0)
    throw std::invalid_argument("moe grouped mma fp4 pipe3: rows_per_block % 128");
  const unsigned z_ext = rows_per_block > 0
                             ? static_cast<unsigned>((max_rows + rows_per_block - 1) / rows_per_block)
                             : 1u;
  if (rows_per_block <= 0) rows_per_block = INT_MAX;
  const int act_vec =
      (reinterpret_cast<uintptr_t>(act) % 16 == 0 && (act_stride % 8) == 0) ? 1 : 0;
  static bool opted_in = false;  // the dynamic smem opt-in, once per OutT
  if (!opted_in) {
    DGPP_CUDA_OK(cudaFuncSetAttribute(moe_grouped_mma_fp4_pipe3_kernel<OutT>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(kSmem)));
    opted_in = true;
  }
  const dim3 grid((n + BN - 1) / BN, static_cast<unsigned>(n_segs), z_ext);
  moe_grouped_mma_fp4_pipe3_kernel<OutT><<<grid, kThreads, kSmem, stream>>>(
      act, act_stride, act_vec, act_rows, segs, views, which, out, out_stride, n, k,
      rows_per_block);
  DGPP_CUDA_OK(cudaGetLastError());
}

void check_fp4_mma_shape(int k, const char* who, int fp4_group) {
  if (k <= 0 || (k % fp4_group) != 0)
    throw std::invalid_argument(std::string(who) + ": k must be a positive multiple of " +
                                std::to_string(fp4_group));
}

// The production fp4 grouped launcher: the ldmatrix kernel on every shape
// (2026-09-08 evening, moe_tile_bench at 512 / 2,048 / 8,192 tokens: gate
// 1.90 / 2.37 / 5.0 ms against the reference tile's 2.76 / 3.70 / 9.9,
// down 2.03 / 3.21 / 7.89 against the two-stage kernel's 2.61 / 3.84 /
// 9.58). Bitwise the reference on both shapes (the bench's checks and
// glm_moe_test's segment pin).
template <typename OutT>
void launch_moe_grouped_mma_fp4(const uint16_t* act, size_t act_stride,
                                const int32_t* act_rows,
                                const MoeSegment* segs, int n_segs, int max_rows,
                                int rows_per_block, const MoeExpertView* views,
                                int which, OutT* out, size_t out_stride, int n,
                                int k, cudaStream_t stream, int fp4_group) {
  launch_moe_grouped_mma_fp4_ldm<OutT>(act, act_stride, act_rows, segs, n_segs, max_rows,
                                       rows_per_block, views, which, out, out_stride, n, k,
                                       stream, fp4_group);
}

// The retired 64 x 128 x 32 two-stage cp.async kernel (bench variant 14).
template <typename OutT>
void launch_moe_grouped_mma_fp4_tile2(const uint16_t* act, size_t act_stride,
                                      const int32_t* act_rows,
                                      const MoeSegment* segs, int n_segs, int max_rows,
                                      int rows_per_block, const MoeExpertView* views,
                                      int which, OutT* out, size_t out_stride, int n,
                                      int k, cudaStream_t stream) {
  if (n_segs <= 0 || n <= 0) return;
  if (!act || !segs || !views || !out)
    throw std::invalid_argument("moe grouped mma fp4: null pointer");
  check_fp4_mma_shape(k, "moe grouped mma fp4");
  if (max_rows <= 0)
    throw std::invalid_argument("moe grouped mma fp4: max_rows must be positive");
  if (rows_per_block > 0 && (rows_per_block % fp4_tile::BM) != 0)
    throw std::invalid_argument(
        "moe grouped mma fp4: rows_per_block must be a multiple of 64");
  const unsigned z_ext = rows_per_block > 0
                             ? static_cast<unsigned>((max_rows + rows_per_block - 1) /
                                                     rows_per_block)
                             : 1u;
  if (rows_per_block <= 0) rows_per_block = INT_MAX;
  const int act_vec =
      (reinterpret_cast<uintptr_t>(act) % 16 == 0 && (act_stride % 8) == 0) ? 1 : 0;
  const dim3 grid((n + fp4_tile::BN - 1) / fp4_tile::BN, static_cast<unsigned>(n_segs),
                  z_ext);
  moe_grouped_mma_fp4_kernel<OutT><<<grid, fp4_tile::kThreads, 0, stream>>>(
      act, act_stride, act_vec, act_rows, segs, views, which, out, out_stride, n,
      k, rows_per_block);
  DGPP_CUDA_OK(cudaGetLastError());
}

template <typename OutT>
void launch_dense_mma_fp4(const uint16_t* act, size_t act_stride,
                          const GlmFp4Matrix& w, OutT* out, int m, int n, int k,
                          cudaStream_t stream) {
  using namespace mma_tile;
  if (m <= 0 || n <= 0) return;
  if (!act || !w.payload || !w.scales || !w.global_scale || !out)
    throw std::invalid_argument("dense mma fp4: null pointer");
  check_fp4_mma_shape(k, "dense mma fp4");
  if (w.rows < n || w.cols != k)
    throw std::invalid_argument("dense mma fp4: matrix geometry does not match n, k");
  const int act_vec =
      (reinterpret_cast<uintptr_t>(act) % 16 == 0 && (act_stride % 8) == 0) ? 1 : 0;
  const dim3 grid((n + BN - 1) / BN, 1u, static_cast<unsigned>((m + BM - 1) / BM));
  moe_grouped_mma_fp4_ref_kernel<OutT, 128, 64, 64, 3><<<grid, kThreads, 0, stream>>>(
      act, act_stride, act_vec, nullptr, nullptr, nullptr, 0, out,
      static_cast<size_t>(n), n, k, BM, MoeSegment{0, m, 0}, MoeExpertView::of(w));
  DGPP_CUDA_OK(cudaGetLastError());
}

// The reference kernel's grouped form (the bench's and the gate's second
// implementation): the synchronous 128 x 64 x 64 tile over the same segments.
template <typename OutT>
void launch_moe_grouped_mma_fp4_ref(const uint16_t* act, size_t act_stride,
                                    const int32_t* act_rows, const MoeSegment* segs,
                                    int n_segs, int max_rows, int rows_per_block,
                                    const MoeExpertView* views, int which, OutT* out,
                                    size_t out_stride, int n, int k,
                                    cudaStream_t stream, int variant = 0) {
  using namespace mma_tile;
  if (n_segs <= 0 || n <= 0) return;
  if (!act || !segs || !views || !out)
    throw std::invalid_argument("moe grouped mma fp4 ref: null pointer");
  check_fp4_mma_shape(k, "moe grouped mma fp4 ref");
  if (variant != 13 && rows_per_block > 0 &&
      (rows_per_block % ((variant == 1 || variant == 14) ? 64 : BM)) != 0)
    throw std::invalid_argument("moe grouped mma fp4 ref: rows_per_block % tile rows");
  const unsigned z_ext = rows_per_block > 0
                             ? static_cast<unsigned>((max_rows + rows_per_block - 1) /
                                                     rows_per_block)
                             : 1u;
  if (rows_per_block <= 0) rows_per_block = INT_MAX;
  const int act_vec =
      (reinterpret_cast<uintptr_t>(act) % 16 == 0 && (act_stride % 8) == 0) ? 1 : 0;
  const int bn = (variant >= 5 && variant <= 7) ? 128 : BN;
  const dim3 grid((n + bn - 1) / bn, static_cast<unsigned>(n_segs), z_ext);
#define DGPP_FP4_REF_LAUNCH(...)                                                        \
  moe_grouped_mma_fp4_ref_kernel<OutT, __VA_ARGS__><<<grid, kThreads, 0, stream>>>(   \
      act, act_stride, act_vec, act_rows, segs, views, which, out, out_stride, n, k, \
      rows_per_block, MoeSegment{}, MoeExpertView{})
  switch (variant) {
    case 1: DGPP_FP4_REF_LAUNCH(64, 64, 64, 3); break;    // 64-row tiles
    case 2: DGPP_FP4_REF_LAUNCH(128, 64, 32, 4); break;   // 32-deep, 4 blocks/SM
    case 3: DGPP_FP4_REF_LAUNCH(128, 64, 32, 2); break;   // 32-deep, 2 blocks/SM
    case 4: DGPP_FP4_REF_LAUNCH(128, 64, 64, 4); break;   // 64-deep, 4 blocks/SM
    case 5: DGPP_FP4_REF_LAUNCH(128, 128, 32, 3); break;  // 128-wide n-tile, 32-deep
    case 6: DGPP_FP4_REF_LAUNCH(128, 128, 64, 2); break;  // 128-wide n-tile, 64-deep
    case 7: DGPP_FP4_REF_LAUNCH(128, 128, 32, 2); break;  // 128-wide, 32-deep, regs free
    case 8: DGPP_FP4_REF_LAUNCH(128, 64, 64, 3, 1); break;  // decomposition: no weight loads
    case 9: DGPP_FP4_REF_LAUNCH(128, 64, 64, 3, 2); break;  // no activation loads
    case 10: DGPP_FP4_REF_LAUNCH(128, 64, 64, 3, 3); break; // neither: decode + barriers + MMA
    case 11: DGPP_FP4_REF_LAUNCH(128, 64, 64, 3, 4); break; // loads + decode + barriers, no MMA
    case 12:  // the three-stage cp.async kernel (bitwise the reference)
      launch_moe_grouped_mma_fp4_pipe3<OutT>(act, act_stride, act_rows, segs, n_segs, max_rows,
                                             rows_per_block == INT_MAX ? 0 : rows_per_block, views,
                                             which, out, out_stride, n, k, stream);
      break;
    case 13:  // the ldmatrix kernel (bitwise the reference; the production launcher)
      launch_moe_grouped_mma_fp4_ldm<OutT>(act, act_stride, act_rows, segs, n_segs, max_rows,
                                           rows_per_block == INT_MAX ? 0 : rows_per_block, views,
                                           which, out, out_stride, n, k, stream, kFp4Group);
      break;
    case 14:  // the retired two-stage 64 x 128 x 32 kernel (bitwise the reference)
      launch_moe_grouped_mma_fp4_tile2<OutT>(act, act_stride, act_rows, segs, n_segs, max_rows,
                                             rows_per_block == INT_MAX ? 0 : rows_per_block, views,
                                             which, out, out_stride, n, k, stream);
      break;
    default: DGPP_FP4_REF_LAUNCH(128, 64, 64, 3);         // the reference
  }
#undef DGPP_FP4_REF_LAUNCH
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace

void launch_moe_grouped_gemv_fp4_bf16(const uint16_t* act, size_t act_stride,
                                      const MoeSegment* segs, int n_segs,
                                      int max_rows, int rows_per_block,
                                      const MoeExpertView* views, int which,
                                      uint16_t* out, size_t out_stride, int n,
                                      int k, cudaStream_t stream, int fp4_group) {
  launch_moe_grouped_gemv_fp4<uint16_t>(act, act_stride, segs, n_segs, max_rows,
                                        rows_per_block, views, which, out,
                                        out_stride, n, k, stream, fp4_group);
}

void moe_set_fp8_ldm(bool on) { g_fp8_ldm_enabled = on; }

void launch_moe_grouped_mma_fp4_bf16(const uint16_t* act, size_t act_stride,
                                     const MoeSegment* segs, int n_segs,
                                     int max_rows, int rows_per_block,
                                     const MoeExpertView* views, int which,
                                     uint16_t* out, size_t out_stride, int n,
                                     int k, cudaStream_t stream,
                                     const int32_t* act_rows, int fp4_group) {
  launch_moe_grouped_mma_fp4<uint16_t>(act, act_stride, act_rows, segs, n_segs,
                                       max_rows, rows_per_block, views, which, out,
                                       out_stride, n, k, stream, fp4_group);
}

void launch_moe_grouped_mma_fp4_f32(const uint16_t* act, size_t act_stride,
                                    const MoeSegment* segs, int n_segs,
                                    int max_rows, int rows_per_block,
                                    const MoeExpertView* views, int which,
                                    float* out, size_t out_stride, int n, int k,
                                    cudaStream_t stream, const int32_t* act_rows, int fp4_group) {
  launch_moe_grouped_mma_fp4<float>(act, act_stride, act_rows, segs, n_segs,
                                    max_rows, rows_per_block, views, which, out,
                                    out_stride, n, k, stream, fp4_group);
}

void launch_moe_grouped_mma_fp4_ref_bf16(const uint16_t* act, size_t act_stride,
                                         const MoeSegment* segs, int n_segs,
                                         int max_rows, int rows_per_block,
                                         const MoeExpertView* views, int which,
                                         uint16_t* out, size_t out_stride, int n,
                                         int k, cudaStream_t stream,
                                         const int32_t* act_rows, int variant) {
  launch_moe_grouped_mma_fp4_ref<uint16_t>(act, act_stride, act_rows, segs, n_segs,
                                           max_rows, rows_per_block, views, which, out,
                                           out_stride, n, k, stream, variant);
}

void launch_moe_grouped_mma_fp4_ref_f32(const uint16_t* act, size_t act_stride,
                                        const MoeSegment* segs, int n_segs,
                                        int max_rows, int rows_per_block,
                                        const MoeExpertView* views, int which,
                                        float* out, size_t out_stride, int n, int k,
                                        cudaStream_t stream, const int32_t* act_rows,
                                        int variant) {
  launch_moe_grouped_mma_fp4_ref<float>(act, act_stride, act_rows, segs, n_segs,
                                        max_rows, rows_per_block, views, which, out,
                                        out_stride, n, k, stream, variant);
}

void launch_dense_mma_fp4_bf16(const uint16_t* act, size_t act_stride,
                               const GlmFp4Matrix& w, uint16_t* out, int m, int n,
                               int k, cudaStream_t stream) {
  launch_dense_mma_fp4<uint16_t>(act, act_stride, w, out, m, n, k, stream);
}

void launch_dense_mma_fp4_f32(const uint16_t* act, size_t act_stride,
                              const GlmFp4Matrix& w, float* out, int m, int n,
                              int k, cudaStream_t stream) {
  launch_dense_mma_fp4<float>(act, act_stride, w, out, m, n, k, stream);
}

void launch_dense_mma_fp4_prod_bf16(const uint16_t* act, size_t act_stride,
                                    const MoeSegment* seg, const MoeExpertView* views,
                                    int which, uint16_t* out, int m, int n, int k,
                                    cudaStream_t stream, int fp4_group) {
  if (m <= 0 || n <= 0) return;
  if (!act || !seg || !views || !out)
    throw std::invalid_argument("dense mma fp4 prod: null pointer");
  launch_moe_grouped_mma_fp4<uint16_t>(act, act_stride, nullptr, seg, /*n_segs=*/1,
                                       /*max_rows=*/m, /*rows_per_block=*/0, views, which,
                                       out, static_cast<size_t>(n), n, k, stream, fp4_group);
}

void launch_moe_grouped_gemv_fp4_f32(const uint16_t* act, size_t act_stride,
                                     const MoeSegment* segs, int n_segs,
                                     int max_rows, int rows_per_block,
                                     const MoeExpertView* views, int which,
                                     float* out, size_t out_stride, int n,
                                     int k, cudaStream_t stream, int fp4_group) {
  launch_moe_grouped_gemv_fp4<float>(act, act_stride, segs, n_segs, max_rows,
                                     rows_per_block, views, which, out,
                                     out_stride, n, k, stream, fp4_group);
}

void launch_moe_slot_gate_up_swiglu_fp4(
    const uint16_t* x, size_t x_stride, const int32_t* ids,
    const int32_t* order, const MoeExpertView* views, int n_routed,
    int k_routed, int n_shared, int k_shared, const uint8_t* sh_gate_payload,
    const float* sh_gate_scales, const uint8_t* sh_up_payload,
    const float* sh_up_scales, uint16_t* act, int act_stride, int slots,
    int top_k, float limit, cudaStream_t stream, int shared_view_base, int fp4_group,
    int sh_rs, int sh_cs) {
  if (slots <= 0) return;
  check_fp4_slot_args(x, ids, views, act, n_routed, k_routed, n_shared, k_shared,
                      shared_view_base, "moe_slot_gate_up_fp4", fp4_group);
  const bool shared_fp4 = shared_view_base >= 0;
  if (n_shared > 0 && !shared_fp4 &&
      (!sh_gate_payload || !sh_gate_scales || !sh_up_payload || !sh_up_scales ||
       !fp8_gemv::shape_ok(sh_gate_payload, 1, k_shared) ||
       !fp8_gemv::shape_ok(sh_up_payload, 1, k_shared)))
    throw std::invalid_argument(
        "moe_slot_gate_up_fp4: shared payloads must be 16B-aligned with k % 16 == 0");
  if (act_stride < n_routed || act_stride < n_shared)
    throw std::invalid_argument("moe_slot_gate_up_fp4: act_stride below n");
  const int max_n = n_routed > n_shared ? n_routed : n_shared;
  const int max_k = k_routed > k_shared ? k_routed : k_shared;
  const auto go = [&](auto kc, auto gc) {
    constexpr int K = decltype(kc)::value;
    constexpr int G = decltype(gc)::value;
    constexpr int rpb = fp4_gemv::Geom<K>::rows_per_block;
    const dim3 grid((max_n + rpb - 1) / rpb, static_cast<unsigned>(slots));
    if (shared_fp4)
      moe_slot_gate_up_swiglu_fp4_kernel<K, true, G>
          <<<grid, fp8_gemv::kThreads, fp8_gemv::smem_bytes(1, max_k), stream>>>(
              x, x_stride, ids, order, views, n_routed, n_shared, k_shared,
              sh_gate_payload, sh_gate_scales, sh_up_payload, sh_up_scales, act,
              act_stride, slots, top_k, limit, shared_view_base, sh_rs, sh_cs);
    else
      moe_slot_gate_up_swiglu_fp4_kernel<K, false, G>
          <<<grid, fp8_gemv::kThreads, fp8_gemv::smem_bytes(1, max_k), stream>>>(
              x, x_stride, ids, order, views, n_routed, n_shared, k_shared,
              sh_gate_payload, sh_gate_scales, sh_up_payload, sh_up_scales, act,
              act_stride, slots, top_k, limit, -1, sh_rs, sh_cs);
    DGPP_CUDA_OK(cudaGetLastError());
  };
  if (fp4_group == fp4_gemv::kMxGroup)
    fp4_gemv::dispatch_k_mx(k_routed, [&](auto kc) { go(kc, std::integral_constant<int, fp4_gemv::kMxGroup>{}); });
  else
    fp4_gemv::dispatch_k(k_routed, [&](auto kc) { go(kc, std::integral_constant<int, fp4_gemv::kGroup>{}); });
}

void launch_moe_slot_down_fp4(const uint16_t* act, size_t act_stride,
                              const int32_t* ids, const int32_t* order,
                              const MoeExpertView* views, int n_routed,
                              int k_routed, int n_shared, int k_shared,
                              const uint8_t* sh_payload, const float* sh_scales,
                              float* out, int out_stride, int slots, int top_k,
                              cudaStream_t stream, int shared_view_base, int fp4_group,
                              int sh_rs, int sh_cs) {
  if (slots <= 0) return;
  check_fp4_slot_args(act, ids, views, out, n_routed, k_routed, n_shared, k_shared,
                      shared_view_base, "moe_slot_down_fp4", fp4_group);
  const bool shared_fp4 = shared_view_base >= 0;
  if (n_shared > 0 && !shared_fp4 && (!sh_payload || !sh_scales || !fp8_gemv::shape_ok(sh_payload, 1, k_shared)))
    throw std::invalid_argument(
        "moe_slot_down_fp4: shared payload must be 16B-aligned with k % 16 == 0");
  if (out_stride < n_routed || out_stride < n_shared)
    throw std::invalid_argument("moe_slot_down_fp4: out_stride below n");
  const int max_n = n_routed > n_shared ? n_routed : n_shared;
  const int max_k = k_routed > k_shared ? k_routed : k_shared;
  const auto go = [&](auto kc, auto gc) {
    constexpr int K = decltype(kc)::value;
    constexpr int G = decltype(gc)::value;
    constexpr int rpb = fp4_gemv::Geom<K>::rows_per_block;
    const dim3 grid((max_n + rpb - 1) / rpb, static_cast<unsigned>(slots));
    if (shared_fp4)
      moe_slot_down_fp4_kernel<K, true, G>
          <<<grid, fp8_gemv::kThreads, fp8_gemv::smem_bytes(1, max_k), stream>>>(
              act, act_stride, ids, order, views, n_routed, n_shared, k_shared,
              sh_payload, sh_scales, out, out_stride, slots, top_k, shared_view_base, sh_rs, sh_cs);
    else
      moe_slot_down_fp4_kernel<K, false, G>
          <<<grid, fp8_gemv::kThreads, fp8_gemv::smem_bytes(1, max_k), stream>>>(
              act, act_stride, ids, order, views, n_routed, n_shared, k_shared,
              sh_payload, sh_scales, out, out_stride, slots, top_k, -1, sh_rs, sh_cs);
    DGPP_CUDA_OK(cudaGetLastError());
  };
  if (fp4_group == fp4_gemv::kMxGroup)
    fp4_gemv::dispatch_k_mx(k_routed, [&](auto kc) { go(kc, std::integral_constant<int, fp4_gemv::kMxGroup>{}); });
  else
    fp4_gemv::dispatch_k(k_routed, [&](auto kc) { go(kc, std::integral_constant<int, fp4_gemv::kGroup>{}); });
}

void launch_moe_grouped_gemv_packq_bf16(const uint16_t* act, size_t act_stride,
                                        const MoeSegment* segs, int n_segs,
                                        int max_rows, int rows_per_block,
                                        const MoeExpertView* views, int which,
                                        uint16_t* out, size_t out_stride, int n,
                                        int k, int bits, cudaStream_t stream,
                                        int scale_fmt) {
  launch_moe_grouped_gemv_packq<uint16_t>(act, act_stride, segs, n_segs, max_rows,
                                          rows_per_block, views, which, out, out_stride,
                                          n, k, bits, scale_fmt, stream);
}

void launch_moe_grouped_gemv_packq_f32(const uint16_t* act, size_t act_stride,
                                       const MoeSegment* segs, int n_segs,
                                       int max_rows, int rows_per_block,
                                       const MoeExpertView* views, int which,
                                       float* out, size_t out_stride, int n,
                                       int k, int bits, cudaStream_t stream,
                                       int scale_fmt) {
  launch_moe_grouped_gemv_packq<float>(act, act_stride, segs, n_segs, max_rows,
                                       rows_per_block, views, which, out, out_stride,
                                       n, k, bits, scale_fmt, stream);
}

void launch_moe_slot_gate_up_swiglu_packq(
    const uint16_t* x, size_t x_stride, const int32_t* ids, const int32_t* order,
    const MoeExpertView* views, int n_routed, int k_routed, int routed_bits,
    int n_shared, int shared_bits, uint16_t* act, int act_stride, int slots,
    int top_k, float limit, cudaStream_t stream, int shared_view_base,
    int routed_scale_fmt) {
  if (slots <= 0) return;
  check_packq_slot_args(x, ids, views, act, n_routed, k_routed, routed_bits, n_shared,
                        shared_bits, shared_view_base, routed_scale_fmt, "moe_slot_gate_up_packq");
  if (act_stride < n_routed || act_stride < n_shared)
    throw std::invalid_argument("moe_slot_gate_up_packq: act_stride below n");
  packq_gemv::dispatch_sf(routed_scale_fmt, [&](auto sfc) {
    constexpr int SF = decltype(sfc)::value;
    packq_gemv::dispatch_bits(routed_bits, [&](auto bc) {
      constexpr int RBits = decltype(bc)::value;
      packq_gemv::dispatch_k(k_routed, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        if constexpr (packq_gemv::k_supported<RBits, SF>(K) && packq_gemv::sf_compiled(SF, K) &&
                      packq_gemv::k_supported<kPackqSharedBits>(K)) {
          constexpr int rpb_r = packq_gemv::Geom<RBits, K, SF>::rows_per_block;
          constexpr int rpb_s = packq_gemv::Geom<kPackqSharedBits, K>::rows_per_block;
          const unsigned bx = static_cast<unsigned>(
              std::max((n_routed + rpb_r - 1) / rpb_r, (n_shared + rpb_s - 1) / rpb_s));
          const dim3 grid(bx, static_cast<unsigned>(slots));
          moe_slot_gate_up_swiglu_packq_kernel<RBits, K, SF>
              <<<grid, fp8_gemv::kThreads, fp8_gemv::smem_bytes(1, K), stream>>>(
                  x, x_stride, ids, order, views, n_routed, n_shared, act, act_stride, slots,
                  top_k, limit, shared_view_base);
          DGPP_CUDA_OK(cudaGetLastError());
        } else {
          throw std::invalid_argument(
              "moe_slot_gate_up_packq: K is outside the scale format's compiled set or a width's chunk budget");
        }
      });
    });
  });
}

void launch_moe_slot_down_packq(const uint16_t* act, size_t act_stride,
                                const int32_t* ids, const int32_t* order,
                                const MoeExpertView* views, int n_routed, int k_routed,
                                int routed_bits, int n_shared, int shared_bits,
                                float* out, int out_stride, int slots, int top_k,
                                cudaStream_t stream, int shared_view_base,
                                int routed_scale_fmt) {
  if (slots <= 0) return;
  check_packq_slot_args(act, ids, views, out, n_routed, k_routed, routed_bits, n_shared,
                        shared_bits, shared_view_base, routed_scale_fmt, "moe_slot_down_packq");
  if (out_stride < n_routed || out_stride < n_shared)
    throw std::invalid_argument("moe_slot_down_packq: out_stride below n");
  packq_gemv::dispatch_sf(routed_scale_fmt, [&](auto sfc) {
    constexpr int SF = decltype(sfc)::value;
    packq_gemv::dispatch_bits(routed_bits, [&](auto bc) {
      constexpr int RBits = decltype(bc)::value;
      packq_gemv::dispatch_k(k_routed, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        if constexpr (packq_gemv::k_supported<RBits, SF>(K) && packq_gemv::sf_compiled(SF, K) &&
                      packq_gemv::k_supported<kPackqSharedBits>(K)) {
          constexpr int rpb_r = packq_gemv::Geom<RBits, K, SF>::rows_per_block;
          constexpr int rpb_s = packq_gemv::Geom<kPackqSharedBits, K>::rows_per_block;
          const unsigned bx = static_cast<unsigned>(
              std::max((n_routed + rpb_r - 1) / rpb_r, (n_shared + rpb_s - 1) / rpb_s));
          const dim3 grid(bx, static_cast<unsigned>(slots));
          moe_slot_down_packq_kernel<RBits, K, SF>
              <<<grid, fp8_gemv::kThreads, fp8_gemv::smem_bytes(1, K), stream>>>(
                  act, act_stride, ids, order, views, n_routed, n_shared, out, out_stride,
                  slots, top_k, shared_view_base);
          DGPP_CUDA_OK(cudaGetLastError());
        } else {
          throw std::invalid_argument(
              "moe_slot_down_packq: K is outside the scale format's compiled set or a width's chunk budget");
        }
      });
    });
  });
}

}  // namespace dgpp
