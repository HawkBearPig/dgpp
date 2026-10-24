#include "kernels/qwen_moe.hpp"
#include "kernels/pdl.cuh"

#include <stdexcept>
#include <type_traits>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/bf16_gemv.cuh"
#include "kernels/fp8_gemv.cuh"
#include "kernels/gemv_common.cuh"

namespace dgpp {
namespace {

__device__ __forceinline__ float round_bf16(float v) {
  return bf16_bits_to_float(float_to_bf16_bits(v));
}

constexpr int kGateWarps = 4;

// One warp's shared gate for token t: the lanes' strided fp32 partials over
// 16-byte vectors, the xor tree, bf16(logit), bf16(sigmoid) — the body of
// shared_gate_kernel below, shared with the fused tail's gate block.
__device__ __forceinline__ float shared_gate_warp(const uint16_t* __restrict__ xrow,
                                                  const uint16_t* __restrict__ g, int hidden,
                                                  int lane) {
  // No gate (Flint's ungated shared expert, 2026-10-22): weight 1 — the
  // accumulate's fmaf(1, y, a) is exactly a + y.
  if (g == nullptr) return 1.0f;
  const uint4* xv = reinterpret_cast<const uint4*>(xrow);
  const uint4* gv = reinterpret_cast<const uint4*>(g);
  const int vecs = hidden / 8;
  float dot = 0.f;
  for (int v = lane; v < vecs; v += 32) {
    const uint4 xq = xv[v];
    const uint4 gq = gv[v];
    const uint32_t xw[4] = {xq.x, xq.y, xq.z, xq.w};
    const uint32_t gw[4] = {gq.x, gq.y, gq.z, gq.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      dot = __fmaf_rn(bf16_bits_to_float(static_cast<uint16_t>(xw[i] & 0xFFFFu)),
                      bf16_bits_to_float(static_cast<uint16_t>(gw[i] & 0xFFFFu)), dot);
      dot = __fmaf_rn(bf16_bits_to_float(static_cast<uint16_t>(xw[i] >> 16)),
                      bf16_bits_to_float(static_cast<uint16_t>(gw[i] >> 16)), dot);
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
  const float logit = round_bf16(dot);
  return round_bf16(1.0f / (1.0f + expf(-logit)));
}

// ---- the fused decode tail --------------------------------------
// The shared expert at decode was seven launches per layer (two GEMVs, the
// swiglu, a GEMV, the gate, the accumulate, the round: 24 us and seven graph
// nodes per layer, 1.15 ms per step). Two kernels now, each bitwise the
// chain it replaces: every dot is bf16_gemv::row_dots on the same staged
// activations (the GEMV interface's own chain at m <= 4), the swiglu is
// moe_swiglu_clamp_kernel's ops on the bf16-rounded dots, the gate is
// shared_gate_warp's, the accumulate is moe_accum_kernel's fma and
// moe_round_bf16_kernel's rounding, applied in the down GEMV's epilogue.

// Blocks [0, row_blocks): warp w's row s = 8b + w of gate_proj and up_proj
// dotted against every staged activation row; act[r][s] = swiglu. The last
// block: warp t's shared gate sw[t].
// The weights' form (2026-09-10, engine.dense_weights = "fp8"): BF16 rows,
// or block-FP8 rows (E4M3 payload + fp32 128 x 128 scales, the scale
// GEMM's own chain through fp8_gemv::row_dots / row_dots_pair) — the FP8
// tail is bitwise the unfused fp8 chain as the BF16 one is the bf16 chain.
// rs / cs (the fp8 form): the scale grid as log2 block sizes — 7 / 7 the
// releases' 128 x 128 blocks and the at-load encode, 0 / 5 the QAD export's
// MXFP8 (a scale per row per 32 columns); the core's indexing reads either.
template <int kRows, bool kFp8>
__global__ void shared_gate_up_swiglu_kernel(const uint16_t* __restrict__ x, size_t x_stride,
                                             const void* __restrict__ gate_w,
                                             const float* __restrict__ gate_s,
                                             const void* __restrict__ up_w,
                                             const float* __restrict__ up_s,
                                             const uint16_t* __restrict__ g,
                                             uint16_t* __restrict__ act, float* __restrict__ sw,
                                             int tokens, int S, int H, int row_blocks, int rs, int cs) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint16_t sx[];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (static_cast<int>(blockIdx.x) == row_blocks) {
    if (warp < tokens) {
      const float w = shared_gate_warp(x + static_cast<size_t>(warp) * x_stride, g, H, lane);
      if (lane == 0) sw[warp] = w;
    }
    return;
  }
  gemv::stage_activations<kRows>(x, x_stride, H, sx);
  __syncthreads();
  const int row = static_cast<int>(blockIdx.x) * gemv::kWarps + warp;
  if (row >= S) return;
  float ga[kRows], ua[kRows];
  if constexpr (kFp8) {
    const int scale_cols = (H + (1 << cs) - 1) >> cs;
    const size_t srow = static_cast<size_t>(row >> rs) * scale_cols;
    fp8_gemv::row_dots_pair<kRows>(static_cast<const uint8_t*>(gate_w) + static_cast<size_t>(row) * H,
                                   gate_s + srow,
                                   static_cast<const uint8_t*>(up_w) + static_cast<size_t>(row) * H,
                                   up_s + srow, sx, H, lane, ga, ua, cs);
  } else {
    bf16_gemv::row_dots<kRows>(static_cast<const uint16_t*>(gate_w) + static_cast<size_t>(row) * H, sx, H, lane, ga);
    bf16_gemv::row_dots<kRows>(static_cast<const uint16_t*>(up_w) + static_cast<size_t>(row) * H, sx, H, lane, ua);
  }
  if (lane != 0) return;
  const float limit = INFINITY;  // the chain's swiglu ran with no clamps
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    float gv = round_bf16(ga[r]);  // the gate GEMV's bf16 output
    float uv = round_bf16(ua[r]);  // the up GEMV's
    if (gv > limit) gv = limit;
    uv = fminf(fmaxf(uv, -limit), limit);
    const uint16_t t = float_to_bf16_bits(gv * (1.0f / (1.0f + expf(-gv))));  // rounding 1
    act[static_cast<size_t>(r) * S + row] =
        float_to_bf16_bits(bf16_bits_to_float(t) * uv);  // rounding 2
  }
}

// The routed accumulation the down epilogue adds to: the accumulate
// kernel's output (`acc` [rows, H]), or its inputs — the per-slot down
// outputs (`contrib`, slot layout tokens * (top_k + 1)) and the router
// weights — summed here in moe_slot_accum_routed_f32_kernel's fma order
// (2026-09-29: the accumulate launch leaves the layer; bitwise).
struct SharedAcc {
  const float* acc = nullptr;
  const float* contrib = nullptr;
  const float* weights = nullptr;
  int top_k = 0;
  int row0 = 0;  // the launch's first token (the slot layout is by token)
};

// Warp w's row h = 8b + w of down_proj [H, S] against the staged act rows;
// out[r][h] = bf16(fma(sw[r], dot, acc[r][h])).
constexpr int kSharedAccMaxK = 16;  // routed slots per token the folded form stages

template <int kRows, bool kFp8>
__global__ void shared_down_accum_round_kernel(const uint16_t* __restrict__ act,
                                               size_t act_stride,
                                               const void* __restrict__ down_w,
                                               const float* __restrict__ down_s,
                                               const float* __restrict__ sw,
                                               SharedAcc src,
                                               uint16_t* __restrict__ out, int H, int S, int rs, int cs) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint16_t sx[];
  // The folded accumulation's inputs for this block's kWarps rows: the
  // slot contributions [kRows][K][kWarps] staged with the loads coalesced
  // across the block's rows (one sector per (row, slot)), the weights
  // [kRows][K]; both land beside the activation staging so the dot's own
  // loads overlap them (2026-09-29: per-warp scattered loads after the dot
  // cost a latency round per row).
  __shared__ float s_contrib[kRows][kSharedAccMaxK][gemv::kWarps];
  __shared__ float s_weights[kRows][kSharedAccMaxK];
  const int h0 = static_cast<int>(blockIdx.x) * gemv::kWarps;
  if (src.contrib != nullptr) {
    const int K = src.top_k;
    for (int i = threadIdx.x; i < kRows * K * gemv::kWarps; i += blockDim.x) {
      const int c = i % gemv::kWarps, j = (i / gemv::kWarps) % K, r = i / (gemv::kWarps * K);
      const int t = src.row0 + r;
      const int h = h0 + c;
      s_contrib[r][j][c] = h < H ? src.contrib[(static_cast<size_t>(t) * (K + 1) + j) * H + h] : 0.f;
    }
    for (int i = threadIdx.x; i < kRows * K; i += blockDim.x) {
      const int j = i % K, r = i / K;
      s_weights[r][j] = src.weights[static_cast<size_t>(src.row0 + r) * K + j];
    }
  }
  gemv::stage_activations<kRows>(act, act_stride, S, sx);
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int row = h0 + warp;
  if (row >= H) return;
  float d[kRows];
  if constexpr (kFp8) {
    const int scale_cols = (S + (1 << cs) - 1) >> cs;
    fp8_gemv::row_dots<kRows>(static_cast<const uint8_t*>(down_w) + static_cast<size_t>(row) * S,
                              down_s + static_cast<size_t>(row >> rs) * scale_cols, sx, S, lane, d, cs);
  } else {
    bf16_gemv::row_dots<kRows>(static_cast<const uint16_t*>(down_w) + static_cast<size_t>(row) * S, sx, S, lane, d);
  }
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const size_t at = static_cast<size_t>(r) * H + row;
    float a;
    if (src.contrib != nullptr) {
      // The accumulate kernel's fma chain in slot order over the staged values.
      a = 0.f;
      for (int j = 0; j < src.top_k; ++j) a = __fmaf_rn(s_weights[r][j], s_contrib[r][j][warp], a);
    } else {
      a = lane == 0 ? src.acc[at] : 0.f;
    }
    if (lane == 0) out[at] = float_to_bf16_bits(__fmaf_rn(sw[r], d[r], a));
  }
}

template <int kRows, bool kFp8>
void launch_tail_rows(const uint16_t* x, size_t x_stride, const void* gate_w, const float* gate_s,
                      const void* up_w, const float* up_s, const void* down_w, const float* down_s,
                      const uint16_t* g, uint16_t* act, float* sw, SharedAcc acc, uint16_t* out,
                      int tokens, int H, int S, cudaStream_t stream, int rs, int cs) {
  const int row_blocks = (S + gemv::kWarps - 1) / gemv::kWarps;
  shared_gate_up_swiglu_kernel<kRows, kFp8>
      <<<static_cast<unsigned>(row_blocks + 1), gemv::kThreads, gemv::smem_bytes(kRows, H),
         stream>>>(x, x_stride, gate_w, gate_s, up_w, up_s, g, act, sw, tokens, S, H, row_blocks, rs, cs);
  DGPP_CUDA_OK(cudaGetLastError());
  const int down_blocks = (H + gemv::kWarps - 1) / gemv::kWarps;
  shared_down_accum_round_kernel<kRows, kFp8>
      <<<static_cast<unsigned>(down_blocks), gemv::kThreads, gemv::smem_bytes(kRows, S),
         stream>>>(act, static_cast<size_t>(S), down_w, down_s, sw, acc, out, H, S, rs, cs);
  DGPP_CUDA_OK(cudaGetLastError());
}

// The launch's slice of the accumulation source: `acc` by row offset, the
// slot form by its first token.
inline SharedAcc acc_rows(const SharedAcc& a, int row0, int H) {
  SharedAcc r = a;
  if (a.acc != nullptr) r.acc = a.acc + static_cast<size_t>(row0) * H;
  r.row0 = a.row0 + row0;
  return r;
}

// The halves (the side-stream form): rows in chunks of kMaxRows as below.
template <bool kFp8>
void gate_up_rows(const uint16_t* x, size_t x_stride, const void* gate_w, const float* gate_s,
                  const void* up_w, const float* up_s, const uint16_t* g, uint16_t* act, float* sw,
                  int tokens, int H, int S, cudaStream_t stream, int rs, int cs) {
  const int row_blocks = (S + gemv::kWarps - 1) / gemv::kWarps;
  for (int row0 = 0; row0 < tokens; row0 += gemv::kMaxRows) {
    const int rows = tokens - row0 < gemv::kMaxRows ? tokens - row0 : gemv::kMaxRows;
    const uint16_t* xr = x + static_cast<size_t>(row0) * x_stride;
    uint16_t* actr = act + static_cast<size_t>(row0) * S;
    float* swr = sw + row0;
    const auto launch = [&](auto rows_tag) {
      constexpr int kRows = decltype(rows_tag)::value;
      shared_gate_up_swiglu_kernel<kRows, kFp8>
          <<<static_cast<unsigned>(row_blocks + 1), gemv::kThreads, gemv::smem_bytes(kRows, H), stream>>>(
              xr, x_stride, gate_w, gate_s, up_w, up_s, g, actr, swr, rows, S, H, row_blocks, rs, cs);
      DGPP_CUDA_OK(cudaGetLastError());
    };
    switch (rows) {
      case 1: launch(std::integral_constant<int, 1>{}); break;
      case 2: launch(std::integral_constant<int, 2>{}); break;
      case 3: launch(std::integral_constant<int, 3>{}); break;
      default: launch(std::integral_constant<int, 4>{}); break;
    }
  }
}
template <bool kFp8>
void down_rows(const uint16_t* act, const void* down_w, const float* down_s, const float* sw,
               SharedAcc acc, uint16_t* out, int tokens, int H, int S, cudaStream_t stream, int rs, int cs) {
  const int down_blocks = (H + gemv::kWarps - 1) / gemv::kWarps;
  for (int row0 = 0; row0 < tokens; row0 += gemv::kMaxRows) {
    const int rows = tokens - row0 < gemv::kMaxRows ? tokens - row0 : gemv::kMaxRows;
    const uint16_t* actr = act + static_cast<size_t>(row0) * S;
    const float* swr = sw + row0;
    const SharedAcc accr = acc_rows(acc, row0, H);
    uint16_t* outr = out + static_cast<size_t>(row0) * H;
    const auto launch = [&](auto rows_tag) {
      constexpr int kRows = decltype(rows_tag)::value;
      shared_down_accum_round_kernel<kRows, kFp8>
          <<<static_cast<unsigned>(down_blocks), gemv::kThreads, gemv::smem_bytes(kRows, S), stream>>>(
              actr, static_cast<size_t>(S), down_w, down_s, swr, accr, outr, H, S, rs, cs);
      DGPP_CUDA_OK(cudaGetLastError());
    };
    switch (rows) {
      case 1: launch(std::integral_constant<int, 1>{}); break;
      case 2: launch(std::integral_constant<int, 2>{}); break;
      case 3: launch(std::integral_constant<int, 3>{}); break;
      default: launch(std::integral_constant<int, 4>{}); break;
    }
  }
}

// Rows in chunks of kMaxRows, as the GEMM interface chunks them: a row's chain
// never depends on how many rows share its launch.
template <bool kFp8>
void tail_rows(const uint16_t* x, size_t x_stride, const void* gate_w, const float* gate_s,
               const void* up_w, const float* up_s, const void* down_w, const float* down_s,
               const uint16_t* g, uint16_t* act, float* sw, SharedAcc acc, uint16_t* out,
               int tokens, int H, int S, cudaStream_t stream, int rs = 7, int cs = 7) {
  for (int row0 = 0; row0 < tokens; row0 += gemv::kMaxRows) {
    const int rows = tokens - row0 < gemv::kMaxRows ? tokens - row0 : gemv::kMaxRows;
    const uint16_t* xr = x + static_cast<size_t>(row0) * x_stride;
    uint16_t* actr = act + static_cast<size_t>(row0) * S;
    float* swr = sw + row0;
    const SharedAcc accr = acc_rows(acc, row0, H);
    uint16_t* outr = out + static_cast<size_t>(row0) * H;
    switch (rows) {
      case 1: launch_tail_rows<1, kFp8>(xr, x_stride, gate_w, gate_s, up_w, up_s, down_w, down_s, g, actr, swr, accr, outr, rows, H, S, stream, rs, cs); break;
      case 2: launch_tail_rows<2, kFp8>(xr, x_stride, gate_w, gate_s, up_w, up_s, down_w, down_s, g, actr, swr, accr, outr, rows, H, S, stream, rs, cs); break;
      case 3: launch_tail_rows<3, kFp8>(xr, x_stride, gate_w, gate_s, up_w, up_s, down_w, down_s, g, actr, swr, accr, outr, rows, H, S, stream, rs, cs); break;
      default: launch_tail_rows<4, kFp8>(xr, x_stride, gate_w, gate_s, up_w, up_s, down_w, down_s, g, actr, swr, accr, outr, rows, H, S, stream, rs, cs); break;
    }
  }
}

__global__ void shared_gate_kernel(const uint16_t* __restrict__ x,
                                   const uint16_t* __restrict__ g,
                                   float* __restrict__ w, int tokens, int hidden) {
  dgpp_pdl_wait();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int t = blockIdx.x * kGateWarps + warp;
  if (t >= tokens) return;
  const float v = shared_gate_warp(x + static_cast<size_t>(t) * hidden, g, hidden, lane);
  if (lane == 0) w[t] = v;
}

}  // namespace

void qwen_moe_shared_gate_bf16(const uint16_t* x, const uint16_t* g, float* w,
                               int tokens, int hidden, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (!x || !w) throw std::invalid_argument("qwen_moe_shared_gate: null pointer");
  if (hidden <= 0 || hidden % 8 != 0 || (reinterpret_cast<uintptr_t>(x) & 15u) ||
      (g && (reinterpret_cast<uintptr_t>(g) & 15u)))
    throw std::invalid_argument(
        "qwen_moe_shared_gate: hidden % 8 == 0 and 16-byte aligned x, g required");
  const int blocks = (tokens + kGateWarps - 1) / kGateWarps;
  shared_gate_kernel<<<blocks, kGateWarps * 32, 0, stream>>>(x, g, w, tokens, hidden);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_moe_shared_tail_decode(const uint16_t* x, size_t x_stride, const uint16_t* gate_w,
                                 const uint16_t* up_w, const uint16_t* down_w,
                                 const uint16_t* g, uint16_t* act, float* sw, const float* acc,
                                 uint16_t* out, int tokens, int H, int S, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (!x || !gate_w || !up_w || !down_w || !act || !sw || !acc || !out)
    throw std::invalid_argument("qwen_moe_shared_tail_decode: null pointer");
  if (tokens > 2 * gemv::kMaxRows)
    throw std::invalid_argument("qwen_moe_shared_tail_decode: tokens beyond the decode rows");
  if (H <= 0 || H % 8 != 0 || S <= 0 || S % 8 != 0 || x_stride < static_cast<size_t>(H) ||
      !gemv::aligned16(x) || !gemv::aligned16(g) || !gemv::aligned16(gate_w) ||
      !gemv::aligned16(up_w) || !gemv::aligned16(down_w) || !gemv::aligned16(act) ||
      (x_stride * 2) % 16 != 0 || !gemv::smem_fits(gemv::kMaxRows, H))
    throw std::invalid_argument(
        "qwen_moe_shared_tail_decode: H and S multiples of 8, 16-byte aligned operands");
  tail_rows<false>(x, x_stride, gate_w, nullptr, up_w, nullptr, down_w, nullptr, g, act, sw, SharedAcc{acc}, out, tokens, H, S, stream);
}

void qwen_moe_shared_gate_up_decode(const uint16_t* x, size_t x_stride, const uint16_t* gate_w,
                                    const uint16_t* up_w, const uint16_t* g, uint16_t* act, float* sw,
                                    int tokens, int H, int S, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (!x || !gate_w || !up_w || !act || !sw) throw std::invalid_argument("qwen moe shared gate_up: null pointer");
  if (H % 8 != 0 || S % 8 != 0) throw std::invalid_argument("qwen moe shared gate_up: H and S multiples of 8");
  gate_up_rows<false>(x, x_stride, gate_w, nullptr, up_w, nullptr, g, act, sw, tokens, H, S, stream, 7, 7);
}
void qwen_moe_shared_down_decode(const uint16_t* act, const uint16_t* down_w, const float* sw,
                                 const float* acc, uint16_t* out, int tokens, int H, int S,
                                 cudaStream_t stream) {
  if (tokens <= 0) return;
  if (!act || !down_w || !sw || !acc || !out) throw std::invalid_argument("qwen moe shared down: null pointer");
  down_rows<false>(act, down_w, nullptr, sw, SharedAcc{acc}, out, tokens, H, S, stream, 7, 7);
}
// The fp8 forms' scale grid (rs, cs: log2 block rows / columns): a 16-byte
// chunk must lie inside one scale block (cs >= 4).
static void check_fp8_grid(int rs, int cs, const char* who) {
  if (rs < 0 || cs < 4 || cs > 30) throw std::invalid_argument(std::string(who) + ": scale grid (rs >= 0, cs >= 4)");
}
void qwen_moe_shared_gate_up_decode_fp8(const uint16_t* x, size_t x_stride, const uint8_t* gate_p,
                                        const float* gate_s, const uint8_t* up_p, const float* up_s,
                                        const uint16_t* g, uint16_t* act, float* sw, int tokens, int H,
                                        int S, cudaStream_t stream, int rs, int cs) {
  if (tokens <= 0) return;
  if (!x || !gate_p || !gate_s || !up_p || !up_s || !act || !sw)
    throw std::invalid_argument("qwen moe shared gate_up fp8: null pointer");
  if (H % 16 != 0 || S % 16 != 0) throw std::invalid_argument("qwen moe shared gate_up fp8: H and S multiples of 16");
  check_fp8_grid(rs, cs, "qwen moe shared gate_up fp8");
  gate_up_rows<true>(x, x_stride, gate_p, gate_s, up_p, up_s, g, act, sw, tokens, H, S, stream, rs, cs);
}
void qwen_moe_shared_down_decode_fp8(const uint16_t* act, const uint8_t* down_p, const float* down_s,
                                     const float* sw, const float* acc, uint16_t* out, int tokens, int H,
                                     int S, cudaStream_t stream, int rs, int cs) {
  if (tokens <= 0) return;
  if (!act || !down_p || !down_s || !sw || !acc || !out) throw std::invalid_argument("qwen moe shared down fp8: null pointer");
  check_fp8_grid(rs, cs, "qwen moe shared down fp8");
  down_rows<true>(act, down_p, down_s, sw, SharedAcc{acc}, out, tokens, H, S, stream, rs, cs);
}

void qwen_moe_shared_tail_decode_fp8(const uint16_t* x, size_t x_stride, const uint8_t* gate_p,
                                     const float* gate_s, const uint8_t* up_p, const float* up_s,
                                     const uint8_t* down_p, const float* down_s, const uint16_t* g,
                                     uint16_t* act, float* sw, const float* acc, uint16_t* out,
                                     int tokens, int H, int S, cudaStream_t stream, int rs, int cs) {
  if (tokens <= 0) return;
  if (!x || !gate_p || !gate_s || !up_p || !up_s || !down_p || !down_s || !act || !sw || !acc || !out)
    throw std::invalid_argument("qwen_moe_shared_tail_decode_fp8: null pointer");
  check_fp8_grid(rs, cs, "qwen_moe_shared_tail_decode_fp8");
  if (tokens > 2 * gemv::kMaxRows)
    throw std::invalid_argument("qwen_moe_shared_tail_decode_fp8: tokens beyond the decode rows");
  if (H <= 0 || H % 16 != 0 || S <= 0 || S % 16 != 0 || x_stride < static_cast<size_t>(H) ||
      !gemv::aligned16(x) || !gemv::aligned16(g) || !gemv::aligned16(gate_p) ||
      !gemv::aligned16(up_p) || !gemv::aligned16(down_p) || !gemv::aligned16(act) ||
      (x_stride * 2) % 16 != 0 || !gemv::smem_fits(gemv::kMaxRows, H))
    throw std::invalid_argument(
        "qwen_moe_shared_tail_decode_fp8: H and S multiples of 16, 16-byte aligned operands");
  tail_rows<true>(x, x_stride, gate_p, gate_s, up_p, up_s, down_p, down_s, g, act, sw, SharedAcc{acc}, out, tokens, H, S, stream, rs, cs);
}


// The routed accumulation folded into the shared down's epilogue
// (2026-09-29): `contrib` the per-slot down outputs (slot layout
// tokens * (top_k + 1), the routed slots first), `weights` the router's
// [tokens, top_k]; the sum runs in moe_slot_accum_routed_f32_kernel's
// fma order, so out is bitwise the accumulate + tail chain's.
void qwen_moe_shared_tail_decode_routed(const uint16_t* x, size_t x_stride, const uint16_t* gate_w,
                                        const uint16_t* up_w, const uint16_t* down_w, const uint16_t* g,
                                        uint16_t* act, float* sw, const float* contrib, const float* weights,
                                        int top_k, uint16_t* out, int tokens, int H, int S, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (!contrib || !weights || top_k <= 0 || top_k > kSharedAccMaxK) throw std::invalid_argument("qwen_moe_shared_tail_decode_routed: slots");
  if (!x || !gate_w || !up_w || !down_w || !act || !sw || !out)
    throw std::invalid_argument("qwen_moe_shared_tail_decode_routed: null pointer");
  if (tokens > 2 * gemv::kMaxRows)
    throw std::invalid_argument("qwen_moe_shared_tail_decode_routed: tokens beyond the decode rows");
  if (H <= 0 || H % 8 != 0 || S <= 0 || S % 8 != 0 || x_stride < static_cast<size_t>(H) ||
      !gemv::aligned16(x) || !gemv::aligned16(g) || !gemv::aligned16(gate_w) ||
      !gemv::aligned16(up_w) || !gemv::aligned16(down_w) || !gemv::aligned16(act) ||
      (x_stride * 2) % 16 != 0 || !gemv::smem_fits(gemv::kMaxRows, H))
    throw std::invalid_argument("qwen_moe_shared_tail_decode_routed: H and S multiples of 8, 16-byte aligned operands");
  SharedAcc a;
  a.contrib = contrib;
  a.weights = weights;
  a.top_k = top_k;
  tail_rows<false>(x, x_stride, gate_w, nullptr, up_w, nullptr, down_w, nullptr, g, act, sw, a, out, tokens, H, S, stream);
}
void qwen_moe_shared_tail_decode_routed_fp8(const uint16_t* x, size_t x_stride, const uint8_t* gate_p,
                                            const float* gate_s, const uint8_t* up_p, const float* up_s,
                                            const uint8_t* down_p, const float* down_s, const uint16_t* g,
                                            uint16_t* act, float* sw, const float* contrib, const float* weights,
                                            int top_k, uint16_t* out, int tokens, int H, int S, cudaStream_t stream,
                                            int rs, int cs) {
  if (tokens <= 0) return;
  if (!contrib || !weights || top_k <= 0 || top_k > kSharedAccMaxK) throw std::invalid_argument("qwen_moe_shared_tail_decode_routed_fp8: slots");
  check_fp8_grid(rs, cs, "qwen_moe_shared_tail_decode_routed_fp8");
  if (!x || !gate_p || !gate_s || !up_p || !up_s || !down_p || !down_s || !act || !sw || !out)
    throw std::invalid_argument("qwen_moe_shared_tail_decode_routed_fp8: null pointer");
  if (tokens > 2 * gemv::kMaxRows)
    throw std::invalid_argument("qwen_moe_shared_tail_decode_routed_fp8: tokens beyond the decode rows");
  if (H <= 0 || H % 16 != 0 || S <= 0 || S % 16 != 0 || x_stride < static_cast<size_t>(H) ||
      !gemv::aligned16(x) || !gemv::aligned16(g) || !gemv::aligned16(gate_p) ||
      !gemv::aligned16(up_p) || !gemv::aligned16(down_p) || !gemv::aligned16(act) ||
      (x_stride * 2) % 16 != 0 || !gemv::smem_fits(gemv::kMaxRows, H))
    throw std::invalid_argument("qwen_moe_shared_tail_decode_routed_fp8: H and S multiples of 16, 16-byte aligned operands");
  SharedAcc a;
  a.contrib = contrib;
  a.weights = weights;
  a.top_k = top_k;
  tail_rows<true>(x, x_stride, gate_p, gate_s, up_p, up_s, down_p, down_s, g, act, sw, a, out, tokens, H, S, stream, rs, cs);
}
void qwen_moe_shared_down_decode_routed(const uint16_t* act, const uint16_t* down_w, const float* sw,
                                        const float* contrib, const float* weights, int top_k, uint16_t* out,
                                        int tokens, int H, int S, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (!act || !down_w || !sw || !contrib || !weights || top_k <= 0 || top_k > kSharedAccMaxK || !out)
    throw std::invalid_argument("qwen moe shared down routed: null pointer");
  SharedAcc a;
  a.contrib = contrib;
  a.weights = weights;
  a.top_k = top_k;
  down_rows<false>(act, down_w, nullptr, sw, a, out, tokens, H, S, stream, 7, 7);
}
void qwen_moe_shared_down_decode_routed_fp8(const uint16_t* act, const uint8_t* down_p, const float* down_s,
                                            const float* sw, const float* contrib, const float* weights, int top_k,
                                            uint16_t* out, int tokens, int H, int S, cudaStream_t stream, int rs,
                                            int cs) {
  if (tokens <= 0) return;
  if (!act || !down_p || !down_s || !sw || !contrib || !weights || top_k <= 0 || top_k > kSharedAccMaxK || !out)
    throw std::invalid_argument("qwen moe shared down routed fp8: null pointer");
  check_fp8_grid(rs, cs, "qwen moe shared down routed fp8");
  SharedAcc a;
  a.contrib = contrib;
  a.weights = weights;
  a.top_k = top_k;
  down_rows<true>(act, down_p, down_s, sw, a, out, tokens, H, S, stream, rs, cs);
}
}  // namespace dgpp
