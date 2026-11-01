// The streaming tensor-core decode GEMM — see mma_gemv.hpp.
#include "kernels/mma_gemv.hpp"
#include "kernels/pdl.cuh"

#include <algorithm>
#include <cstdlib>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/bf12_gemv.cuh"
#include "kernels/fp8_gemv.cuh"
#include "kernels/mma_gemv_body.cuh"

namespace dgpp {
namespace {

// The forms' geometry, the staging helpers and the block body: kernels/mma_gemv_body.cuh (the step kernel runs the same body).
using namespace ::dgpp::mma_gemv_body;

template <int kTiles, bool kFp8, int kW, typename OutT, typename ScaleT, bool kBf12 = false>
__global__ __launch_bounds__(kW * 32, 2) void mma_gemv_kernel(const uint16_t* __restrict__ act_in,
                                                            size_t act_stride,
                                                            const void* __restrict__ wv_in,
                                                            const ScaleT* __restrict__ scales_in,
                                                            OutT* __restrict__ out_in, int m, int n_in,
                                                            int k, size_t out_stride_in, int rs,
                                                            int cs, float* __restrict__ part_in,
                                                            const uint16_t* __restrict__ act2, float limit,
                                                            const uint16_t* __restrict__ act_seg2,
                                                            const void* __restrict__ wv2,
                                                            const ScaleT* __restrict__ scales2,
                                                            OutT* __restrict__ out2, int n2, size_t out_stride2,
                                                            int blocks0,
                                                            const uint32_t* __restrict__ b12_rows,
                                                            const uint32_t* __restrict__ b12_esc,
                                                            const uint16_t* __restrict__ b12_raw) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint8_t mma_smem[];
  mma_gemv_block<kTiles, kFp8, kW, OutT, ScaleT, kBf12>(static_cast<int>(blockIdx.x), static_cast<int>(blockIdx.y), static_cast<int>(gridDim.y), mma_smem,
      act_in, act_stride, wv_in, scales_in, out_in, m, n_in, k, out_stride_in, rs, cs, part_in, act2, limit, act_seg2, wv2, scales2, out2, n2, out_stride2, blocks0, b12_rows, b12_esc, b12_raw);
}

// The split-K reduce: out[m, n] = sum over the splits, in split order, of
// the fp32 partials (deterministic; the split count is a function of the
// shape, never of m).
// (TRIED 2026-10-02 and reverted: the fold inside the GEMV launch — each
// split of a column tile takes a ticket as it finishes, its partials
// fenced, and the last one folds the tile. Bitwise, 640 fewer graph nodes
// per DeepSeek-V4-Flash depth-3 pass, and level on the fabric at 37.0 ms:
// the fence, the ticket and the barrier put the ~2 us this launch costs
// back onto every split GEMV.)
template <typename OutT>
__global__ void mma_gemv_split_reduce_kernel(const float* __restrict__ part, int splits, int m, int n,
                                             OutT* __restrict__ out, size_t out_stride) {
  dgpp_pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m * n) return;
  const int mrow = i / n, col = i - mrow * n;
  float acc = 0.f;
  for (int s = 0; s < splits; ++s) acc += part[(static_cast<size_t>(s) * m + mrow) * n + col];
  fp8_gemv::store_dot(out + static_cast<size_t>(mrow) * out_stride + col, acc);
}

// Split-K for the decode forms (2026-09-21): a small-n site's grid is n / 16
// blocks at two warps (the [320 x 10240] hyperconnection down projection: 20
// blocks on a 48-SM part), and each warp then walks its k windows in
// sequence with nothing else resident — the c=4 profile has that site at
// 68 us for 3.3 MB (48 GB/s) and the shared expert's [320 x 2560] gate and
// up at 19-24 us (under 45 GB/s). Splitting the k range across grid.y
// blocks (fp32 partials in the caller's GEMM workspace, one reduce launch)
// fills the part. The split count is a function of (n, k, width) only, so a
// row's chain is the same whatever m rides in the launch (the decode
// forms' invariant); it is not the unsplit chain (tolerance-equal, the
// contract above the GEMV chunk bound). On by default -- the same trade
// cuBLAS makes by heuristic for small-n shapes; DGPP_MMA_SPLITK=0 keeps the
// unsplit chain. Measured on two GB10s (Qwen3.8-Flash-Next, TP=2, MTP depth
// 3): the decode pass shortens 119.0 -> 113.5 ms (-4.6%); tokens per pass
// also move a little, because the summation order (and so the text) does.
// DGPP_MMA_SPLITK_FILL overrides the block target (default two resident
// blocks per SM).
constexpr int kMaxSplit = 16;

int split_k_target_blocks() {
  static const int target = [] {
    if (const char* v = std::getenv("DGPP_MMA_SPLITK_FILL")) {
      const int t = std::atoi(v);
      if (t >= 1) return t;
    }
    int dev = 0, sms = 48;
    if (cudaGetDevice(&dev) == cudaSuccess) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    return 2 * sms;
  }();
  return target;
}
bool split_k_enabled() {
  static const bool on = [] {
    const char* v = std::getenv("DGPP_MMA_SPLITK");
    return !(v && v[0] == '0' && v[1] == '\0');
  }();
  return on;
}
// The decode forms' width from n (mma_gemv_test's sweep over the DSA and
// attention shapes, 2026-09-14): eight warps from 4096 rows, four from
// 1024, two under (never one: a lone warp per block is issue-bound — the
// [576 x 6144] site at 16 rows 53 us at one warp, 37 at two, 44 at four).
int g_mma_decode_width = 0;  // mma_gemv_set_decode_width: 0 the rule, else a fixed width (the sweep)
int decode_width(int n) {
  const int warps = (n + kRowsPerWarp - 1) / kRowsPerWarp;
  return g_mma_decode_width > 0 ? g_mma_decode_width : warps >= 512 ? 8 : warps >= 128 ? 4 : 2;
}
// The decode forms' grid at that width: the split decision's block count
// for every launch of up to kMmaGemvSplitRows rows (the 33..64-row form's
// own grid is the eight-warp one; it takes the decode width's split so its
// chain is the 1..32-row forms').
int decode_blocks(int n) {
  const int w = decode_width(n);
  return (n + w * kRowsPerWarp - 1) / (w * kRowsPerWarp);
}
int split_k_for(int n, int k, const void* ws, size_t ws_bytes) {
  if (!ws || !split_k_enabled()) return 1;
  const int fill = split_k_target_blocks();
  const int blocks = decode_blocks(n);
  if (blocks >= fill) return 1;
  const int units = (k + kSplitUnitK - 1) / kSplitUnitK;
  const int s = std::min({(fill + blocks - 1) / blocks, units, kMaxSplit});
  // The workspace check at the decode batch bound, so the split (and the
  // chain) never depends on the rows in the launch.
  if (s < 2 || static_cast<size_t>(s) * kMmaGemvSplitRows * static_cast<size_t>(n) * sizeof(float) > ws_bytes)
    return 1;
  return s;
}

// One decode form at one width: the shared-memory opt-in once per
// instantiation (the A windows + the warp weight buffers exceed the default
// budget at the wide widths).
template <int kTiles, bool kFp8, int kW, typename OutT, typename ScaleT>
void launch_decode_form(const uint16_t* a, size_t act_stride, const void* w, const ScaleT* scales, OutT* o,
                        int rows, int n, int k, size_t out_stride, int rs, int cs, cudaStream_t stream,
                        void* ws, size_t ws_bytes, const uint16_t* act2 = nullptr, float limit = 0.f) {
  using L = WarpLoads<kTiles, kFp8, kW>;
  static bool opted_in = false;
  if (!opted_in) {
    DGPP_CUDA_OK(cudaFuncSetAttribute(mma_gemv_kernel<kTiles, kFp8, kW, OutT, ScaleT>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(L::kSmemBytes)));
    opted_in = true;
  }
  const int blocks = (n + kW * kRowsPerWarp - 1) / (kW * kRowsPerWarp);
  const int splits = split_k_for(n, k, ws, ws_bytes);
  if (splits > 1) {
    float* part = static_cast<float*>(ws);
    mma_gemv_kernel<kTiles, kFp8, kW, OutT, ScaleT><<<dim3(blocks, splits), kW * 32, L::kSmemBytes, stream>>>(
        a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs, part, act2, limit, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, nullptr, nullptr, nullptr);
    const int total = rows * n;
    mma_gemv_split_reduce_kernel<OutT><<<(total + 255) / 256, 256, 0, stream>>>(part, splits, rows, n, o,
                                                                                 out_stride);
    return;
  }
  mma_gemv_kernel<kTiles, kFp8, kW, OutT, ScaleT><<<dim3(blocks), kW * 32, L::kSmemBytes, stream>>>(
      a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs, nullptr, act2, limit, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, nullptr, nullptr, nullptr);
}
// The bf12 form's launch (two warps a block; the unsplit chain — the
// caller refused the shapes the bf16 form would split).
int g_bf12_width = 2;  // mma_gemv_set_bf12_width: the sweep's knob (two warps: 226 GB/s on the head; four 218, eight 212)
template <int kTiles, int kW, typename OutT>
void launch_bf12_decode_form(const uint16_t* a, size_t act_stride, const Bf12Matrix& w, OutT* o, int rows, int n,
                             int k, size_t out_stride, cudaStream_t stream) {
  using L = WarpLoads<kTiles, false, kW, true>;
  static bool opted_in = false;
  if (!opted_in) {
    DGPP_CUDA_OK(cudaFuncSetAttribute(mma_gemv_kernel<kTiles, false, kW, OutT, float, true>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(L::kSmemBytes)));
    opted_in = true;
  }
  const int blocks = (n + kW * kRowsPerWarp - 1) / (kW * kRowsPerWarp);
  mma_gemv_kernel<kTiles, false, kW, OutT, float, true><<<dim3(blocks), kW * 32, L::kSmemBytes, stream>>>(
      a, act_stride, w.packed, nullptr, o, rows, n, k, out_stride, 0, 0, nullptr, nullptr, 0.f, nullptr, nullptr, nullptr,
      nullptr, 0, 0, 0, w.rows, w.esc, w.raw);
  DGPP_CUDA_OK(cudaGetLastError());
}
template <typename OutT>
bool launch_bf12(const uint16_t* act, size_t act_stride, const Bf12Matrix& w, OutT* out, int m, int n, int k,
                 size_t out_stride, cudaStream_t stream, void* ws, size_t ws_bytes) {
  if (!mma_gemv_bf12_accepts(w, m) || w.n != n || w.k != k || act == nullptr || out == nullptr) return false;
  if (!mma_gemv_shape_ok(w.packed, act, act_stride, m, k)) return false;
  if (out_stride == 0) out_stride = static_cast<size_t>(n);
  if (out_stride < static_cast<size_t>(n)) return false;
  // The bf16 form's split rule at the shape: a split launch keeps its bytes.
  if (split_k_for(n, k, ws, ws_bytes) > 1) return false;
  const int width = g_bf12_width;
  if (m <= 16) {
    if (width == 8) launch_bf12_decode_form<1, 8, OutT>(act, act_stride, w, out, m, n, k, out_stride, stream);
    else if (width == 2) launch_bf12_decode_form<1, 2, OutT>(act, act_stride, w, out, m, n, k, out_stride, stream);
    else launch_bf12_decode_form<1, 4, OutT>(act, act_stride, w, out, m, n, k, out_stride, stream);
  } else {
    if (width == 8) launch_bf12_decode_form<2, 8, OutT>(act, act_stride, w, out, m, n, k, out_stride, stream);
    else if (width == 2) launch_bf12_decode_form<2, 2, OutT>(act, act_stride, w, out, m, n, k, out_stride, stream);
    else launch_bf12_decode_form<2, 4, OutT>(act, act_stride, w, out, m, n, k, out_stride, stream);
  }
  return true;
}

// The two-segment reduce: segment 0's partials [splits][m][n0] then
// segment 1's [splits][m][n1], each summed in split order to its output.
template <typename OutT>
__global__ void mma_gemv_split_reduce_pair_kernel(const float* __restrict__ part, int splits, int m, int n0,
                                                  OutT* __restrict__ out0, size_t stride0, int n1,
                                                  OutT* __restrict__ out1, size_t stride1) {
  dgpp_pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int nn = n0 + n1;
  if (i >= m * nn) return;
  const int mrow = i / nn, col = i - mrow * nn;
  const bool second = col >= n0;
  const float* base = second ? part + static_cast<size_t>(splits) * m * n0 : part;
  const int n = second ? n1 : n0, c = second ? col - n0 : col;
  float acc = 0.f;
  for (int s = 0; s < splits; ++s) acc += base[(static_cast<size_t>(s) * m + mrow) * n + c];
  OutT* out = second ? out1 : out0;
  fp8_gemv::store_dot(out + static_cast<size_t>(mrow) * (second ? stride1 : stride0) + c, acc);
}

// Two matrices in one decode launch at one width: the first segment's
// split rule for both (the caller checks the second would take the same),
// each segment's n a whole number of blocks.
template <int kTiles, bool kFp8, int kW, typename OutT, typename ScaleT>
void launch_decode_form_pair(const uint16_t* a0, const void* w0, const ScaleT* s0, OutT* o0, int n0, size_t stride0,
                             const uint16_t* a1, const void* w1, const ScaleT* s1, OutT* o1, int n1, size_t stride1,
                             size_t act_stride, int rows, int k, int rs, int cs, cudaStream_t stream, void* ws,
                             size_t ws_bytes) {
  using L = WarpLoads<kTiles, kFp8, kW>;
  static bool opted_in = false;
  if (!opted_in) {
    DGPP_CUDA_OK(cudaFuncSetAttribute(mma_gemv_kernel<kTiles, kFp8, kW, OutT, ScaleT>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(L::kSmemBytes)));
    opted_in = true;
  }
  const int per_block = kW * kRowsPerWarp;
  const int blocks0 = (n0 + per_block - 1) / per_block, blocks1 = (n1 + per_block - 1) / per_block;
  const int splits = split_k_for(n0, k, ws, ws_bytes);
  if (splits > 1) {
    float* part = static_cast<float*>(ws);
    mma_gemv_kernel<kTiles, kFp8, kW, OutT, ScaleT><<<dim3(blocks0 + blocks1, splits), kW * 32, L::kSmemBytes, stream>>>(
        a0, act_stride, w0, s0, o0, rows, n0, k, stride0, rs, cs, part, nullptr, 0.f, a1, w1, s1, o1, n1, stride1, blocks0, nullptr, nullptr, nullptr);
    const int total = rows * (n0 + n1);
    mma_gemv_split_reduce_pair_kernel<OutT><<<(total + 255) / 256, 256, 0, stream>>>(part, splits, rows, n0, o0, stride0,
                                                                                      n1, o1, stride1);
    return;
  }
  mma_gemv_kernel<kTiles, kFp8, kW, OutT, ScaleT><<<dim3(blocks0 + blocks1), kW * 32, L::kSmemBytes, stream>>>(
      a0, act_stride, w0, s0, o0, rows, n0, k, stride0, rs, cs, nullptr, nullptr, 0.f, a1, w1, s1, o1, n1, stride1, blocks0, nullptr, nullptr, nullptr);
}

// The 33..64-row form (four tiles, eight warps, direct loads) with the
// decode width's split: the same k ranges as the 1..32-row forms, so an
// eight-slot batch's rows are bitwise its slots' solo rows (2026-10-05).
template <int kTiles, bool kFp8, typename OutT, typename ScaleT>
void launch_wide_split(const uint16_t* a, size_t act_stride, const void* w, const ScaleT* scales, OutT* o, int rows,
                       int n, int k, size_t out_stride, int rs, int cs, cudaStream_t stream, void* ws,
                       size_t ws_bytes, const uint16_t* act2 = nullptr, float limit = 0.f) {
  constexpr int kWideThreads = kWideWarps * 32;
  const int blocks = (n + kWideWarps * kRowsPerWarp - 1) / (kWideWarps * kRowsPerWarp);
  const int splits = split_k_for(n, k, ws, ws_bytes);
  if (splits > 1) {
    float* part = static_cast<float*>(ws);
    mma_gemv_kernel<kTiles, kFp8, kWideWarps, OutT, ScaleT><<<dim3(blocks, splits), kWideThreads, Win<kTiles, kFp8>::kSmemBytes, stream>>>(
        a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs, part, act2, limit, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, nullptr, nullptr, nullptr);
    const int total = rows * n;
    mma_gemv_split_reduce_kernel<OutT><<<(total + 255) / 256, 256, 0, stream>>>(part, splits, rows, n, o,
                                                                                 out_stride);
    return;
  }
  mma_gemv_kernel<kTiles, kFp8, kWideWarps, OutT, ScaleT><<<dim3(blocks), kWideThreads, Win<kTiles, kFp8>::kSmemBytes, stream>>>(
      a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs, nullptr, act2, limit, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, nullptr, nullptr, nullptr);
}
template <int kTiles, bool kFp8, typename OutT, typename ScaleT>
void launch_decode(const uint16_t* a, size_t act_stride, const void* w, const ScaleT* scales, OutT* o, int rows,
                   int n, int k, size_t out_stride, int rs, int cs, cudaStream_t stream, void* ws,
                   size_t ws_bytes, const uint16_t* act2 = nullptr, float limit = 0.f) {
  const int width = decode_width(n);
  if (width == 8)
    launch_decode_form<kTiles, kFp8, 8, OutT, ScaleT>(a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs, stream, ws, ws_bytes, act2, limit);
  else if (width == 4)
    launch_decode_form<kTiles, kFp8, 4, OutT, ScaleT>(a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs, stream, ws, ws_bytes, act2, limit);
  else if (width == 2)
    launch_decode_form<kTiles, kFp8, 2, OutT, ScaleT>(a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs, stream, ws, ws_bytes, act2, limit);
  else
    launch_decode_form<kTiles, kFp8, 1, OutT, ScaleT>(a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs, stream, ws, ws_bytes, act2, limit);
}

// DGPP_MMA_TRACE=n: print the first n shape decisions (the dispatch
// bisection log: which (m, n, k) lands on which tiles form).
static int g_mma_trace_budget() {
  static const int v = [] {
    const char* e = std::getenv("DGPP_MMA_TRACE");
    if (!e || !*e || *e == '0') return 0;
    return std::atoi(e) > 0 ? std::atoi(e) : 64;
  }();
  return v;
}
static int& g_mma_trace_left() {
  static int v = -1;  // -1: not yet read from the budget
  if (v < 0) v = g_mma_trace_budget();
  return v;
}

template <bool kFp8, typename OutT, typename ScaleT>
void launch(const uint16_t* act, size_t act_stride, const void* w, const ScaleT* scales, OutT* out,
            int m, int n, int k, size_t out_stride, int rs, int cs, cudaStream_t stream, void* ws = nullptr,
            size_t ws_bytes = 0, const uint16_t* act2 = nullptr, float limit = 0.f) {
  if (m < 1) throw std::invalid_argument("mma_gemv: m outside [1, ...)");
  if (n <= 0 || k <= 0 || (k % 64) != 0) throw std::invalid_argument("mma_gemv: k a multiple of 64");
  if (!mma_gemv_shape_ok(w, act, act_stride, m, k)) throw std::invalid_argument("mma_gemv: alignment");
  if (out_stride == 0) out_stride = static_cast<size_t>(n);
  if (out_stride < static_cast<size_t>(n)) throw std::invalid_argument("mma_gemv: out stride");
  if (kFp8 && (scales == nullptr || cs < 4)) throw std::invalid_argument("mma_gemv: fp8 scales");
  // Groups of at most kMmaGemvMaxRowsPerLaunch rows; the form by the group's rows.
  for (int row0 = 0; row0 < m; row0 += kMmaGemvMaxRowsPerLaunch) {
    const int rows = std::min(kMmaGemvMaxRowsPerLaunch, m - row0);
    const uint16_t* a = act + static_cast<size_t>(row0) * act_stride;
    const uint16_t* a2 = act2 != nullptr ? act2 + static_cast<size_t>(row0) * act_stride : nullptr;
    OutT* o = out + static_cast<size_t>(row0) * out_stride;
    if (g_mma_trace_left() > 0) {
      --g_mma_trace_left();
      std::fprintf(stderr, "mma_trace fp8=%d out=%s m=%d n=%d k=%d tiles=%d\n",
                   static_cast<int>(kFp8), std::is_same<OutT, float>::value ? "f32" : "bf16", m, n, k,
                   rows <= 16 ? 1 : rows <= 32 ? 2 : rows <= 64 ? 4 : 8);
      std::fflush(stderr);
    }
    if (rows <= 16)
      launch_decode<1, kFp8, OutT, ScaleT>(a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs, stream, ws, ws_bytes, a2, limit);
    else if (rows <= 32)
      launch_decode<2, kFp8, OutT, ScaleT>(a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs, stream, ws, ws_bytes, a2, limit);
    else if (rows <= 64)
      launch_wide_split<4, kFp8, OutT, ScaleT>(a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs, stream, ws, ws_bytes, a2, limit);
    else
      // The 65..128-row form splits like the narrower ones (2026-10-23: the
      // 96-row decode batch; launched unsplit, its rows' chain differed from
      // their solo rows' — sixteen DSpark slots' transcripts diverged at
      // near-ties while twelve matched).
      launch_wide_split<8, kFp8, OutT, ScaleT>(a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs, stream, ws, ws_bytes, a2, limit);
    DGPP_CUDA_OK(cudaGetLastError());
  }
}
}  // namespace

void mma_gemv_set_bf12_width(int warps) {
  if (warps != 2 && warps != 4 && warps != 8) throw std::invalid_argument("mma_gemv_set_bf12_width: 2, 4 or 8");
  g_bf12_width = warps;
}
void mma_gemv_set_decode_width(int warps) {
  if (warps != 0 && warps != 1 && warps != 2 && warps != 4 && warps != 8)
    throw std::invalid_argument("mma_gemv_set_decode_width: 0, 1, 2, 4 or 8");
  g_mma_decode_width = warps;
}

bool mma_gemv_shape_ok(const void* w, const void* act, size_t act_stride, int m, int k) {
  return m >= 1 && k > 0 && (k % 64) == 0 &&
         (reinterpret_cast<uintptr_t>(w) & 15u) == 0 && (reinterpret_cast<uintptr_t>(act) & 15u) == 0 &&
         ((act_stride * 2) % 16) == 0;
}

void launch_mma_gemv_fp8_bf16(const uint16_t* act, size_t act_stride, const uint8_t* w,
                              const float* scales, uint16_t* out, int m, int n, int k,
                              size_t out_stride, int rs, int cs, cudaStream_t stream, void* ws,
                              size_t ws_bytes) {
  launch<true, uint16_t, float>(act, act_stride, w, scales, out, m, n, k, out_stride, rs, cs, stream, ws, ws_bytes);
}

bool mma_gemv_bf12_accepts(const Bf12Matrix& w, int m) {
  return w.packed != nullptr && w.rows != nullptr && w.esc != nullptr && w.raw != nullptr && m >= 1 &&
         m <= kMmaGemvMaxRows && w.n >= 1 && w.k >= kBf12Super && (w.k % kBf12Super) == 0;
}
bool launch_mma_gemv_bf12_f32(const uint16_t* act, size_t act_stride, const Bf12Matrix& w, float* out, int m, int n,
                              int k, size_t out_stride, cudaStream_t stream, void* ws, size_t ws_bytes) {
  return launch_bf12<float>(act, act_stride, w, out, m, n, k, out_stride, stream, ws, ws_bytes);
}
bool launch_mma_gemv_bf12_bf16(const uint16_t* act, size_t act_stride, const Bf12Matrix& w, uint16_t* out, int m,
                               int n, int k, size_t out_stride, cudaStream_t stream, void* ws, size_t ws_bytes) {
  return launch_bf12<uint16_t>(act, act_stride, w, out, m, n, k, out_stride, stream, ws, ws_bytes);
}

bool mma_gemv_pair_ok(const void* w0, const void* w1, const uint16_t* a0, const uint16_t* a1, size_t act_stride, int m,
                      int n0, int n1, int k, void* ws, size_t ws_bytes) {
  if (!ws || m < 1 || m > 32) return false;  // the one- and two-tile decode forms
  if (!mma_gemv_shape_ok(w0, a0, act_stride, m, k) || !mma_gemv_shape_ok(w1, a1, act_stride, m, k)) return false;
  const int per_block = decode_width(n0) * kRowsPerWarp;
  if (n0 % per_block != 0 || n1 % per_block != 0) return false;
  // Both segments' chains must be their single launches': the same split count
  // (the rule at each segment's own width) at the same k.
  const int s0 = split_k_for(n0, k, ws, ws_bytes), s1 = split_k_for(n1, k, ws, ws_bytes);
  if (s0 != s1) return false;
  const size_t part = static_cast<size_t>(s0) * kMmaGemvSplitRows * static_cast<size_t>(n0 + n1) * sizeof(float);
  return part <= ws_bytes;
}

// Two fp8 projections in one decode launch (mma_gemv_pair_ok first): the
// first segment picks the width and the split, each row's chain its own
// launch's. Rows above kMmaGemvSplitRows are not a decode form (the caller
// launches separately).
void launch_mma_gemv_fp8_bf16_pair(const uint16_t* a0, const uint8_t* w0, const float* s0, uint16_t* o0, int n0,
                                   size_t stride0, const uint16_t* a1, const uint8_t* w1, const float* s1, uint16_t* o1,
                                   int n1, size_t stride1, size_t act_stride, int m, int k, int rs, int cs,
                                   cudaStream_t stream, void* ws, size_t ws_bytes) {
  if (stride0 == 0) stride0 = static_cast<size_t>(n0);
  if (stride1 == 0) stride1 = static_cast<size_t>(n1);
  const int width = decode_width(n0);
  const int tiles = m <= 16 ? 1 : 2;
  auto go = [&](auto tc, auto wc) {
    constexpr int kT = decltype(tc)::value, kWw = decltype(wc)::value;
    launch_decode_form_pair<kT, true, kWw, uint16_t, float>(a0, w0, s0, o0, n0, stride0, a1, w1, s1, o1, n1, stride1,
                                                            act_stride, m, k, rs, cs, stream, ws, ws_bytes);
  };
  auto by_width = [&](auto tc) {
    if (width == 8) go(tc, std::integral_constant<int, 8>{});
    else if (width == 4) go(tc, std::integral_constant<int, 4>{});
    else if (width == 2) go(tc, std::integral_constant<int, 2>{});
    else go(tc, std::integral_constant<int, 1>{});
  };
  if (tiles == 1) by_width(std::integral_constant<int, 1>{});
  else by_width(std::integral_constant<int, 2>{});
  DGPP_CUDA_OK(cudaGetLastError());
}
void launch_mma_gemv_fp8_f32(const uint16_t* act, size_t act_stride, const uint8_t* w,
                             const float* scales, float* out, int m, int n, int k,
                             size_t out_stride, int rs, int cs, cudaStream_t stream, void* ws,
                             size_t ws_bytes) {
  launch<true, float, float>(act, act_stride, w, scales, out, m, n, k, out_stride, rs, cs, stream, ws, ws_bytes);
}
void launch_mma_gemv_fp8e8m0_bf16(const uint16_t* act, size_t act_stride, const uint8_t* w,
                                  const uint8_t* scales, uint16_t* out, int m, int n, int k,
                                  size_t out_stride, int rs, int cs, cudaStream_t stream, void* ws,
                                  size_t ws_bytes) {
  launch<true, uint16_t, uint8_t>(act, act_stride, w, scales, out, m, n, k, out_stride, rs, cs, stream, ws, ws_bytes);
}
void launch_mma_gemv_fp8e8m0_f32(const uint16_t* act, size_t act_stride, const uint8_t* w,
                                 const uint8_t* scales, float* out, int m, int n, int k,
                                 size_t out_stride, int rs, int cs, cudaStream_t stream, void* ws,
                                 size_t ws_bytes) {
  launch<true, float, uint8_t>(act, act_stride, w, scales, out, m, n, k, out_stride, rs, cs, stream, ws, ws_bytes);
}
void launch_mma_gemv_fp8_swiglu_f32(const uint16_t* gate, const uint16_t* up, float limit, size_t act_stride,
                                    const uint8_t* w, const float* scales, float* out, int m, int n, int k,
                                    size_t out_stride, int rs, int cs, cudaStream_t stream) {
  if (up == nullptr || (reinterpret_cast<uintptr_t>(up) & 15u) != 0)
    throw std::invalid_argument("mma_gemv swiglu: the up rows must be 16-byte aligned");
  launch<true, float, float>(gate, act_stride, w, scales, out, m, n, k, out_stride, rs, cs, stream, nullptr, 0, up, limit);
}
void launch_mma_gemv_bf16_bf16(const uint16_t* act, size_t act_stride, const uint16_t* w,
                               uint16_t* out, int m, int n, int k, size_t out_stride,
                               cudaStream_t stream, void* ws, size_t ws_bytes) {
  launch<false, uint16_t, float>(act, act_stride, w, nullptr, out, m, n, k, out_stride, 7, 7, stream, ws, ws_bytes);
}
void launch_mma_gemv_bf16_f32(const uint16_t* act, size_t act_stride, const uint16_t* w,
                              float* out, int m, int n, int k, size_t out_stride,
                              cudaStream_t stream, void* ws, size_t ws_bytes) {
  launch<false, float, float>(act, act_stride, w, nullptr, out, m, n, k, out_stride, 7, 7, stream, ws, ws_bytes);
}

}  // namespace dgpp
