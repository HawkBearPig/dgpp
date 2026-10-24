#include "kernels/bf12_gemv.hpp"
#include "kernels/pdl.cuh"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/bf12_gemv.cuh"
#include "kernels/bf16_gemv.cuh"
#include "kernels/bf16_gemv.hpp"
#include "kernels/gemv_common.cuh"

namespace dgpp {
namespace {

using bf12_gemv::consume_step;
using bf12_gemv::Tail;
using bf12_gemv::tail_of;

// The narrow form: whole activation rows staged, the packed row's chain
// (bf12_gemv.cuh) per warp.
template <int kRows, bool kOutF32, int kSB>
__global__ void bf12_gemv_kernel(const uint16_t* __restrict__ act, size_t act_stride,
                                 const uint8_t* __restrict__ w,
                                 const uint32_t* __restrict__ rows,
                                 const uint32_t* __restrict__ esc,
                                 const uint16_t* __restrict__ raw, void* __restrict__ out, int n,
                                 int k, size_t row_bytes) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint16_t sx[];
  gemv::stage_activations<kRows>(act, act_stride, k, sx);
  __syncthreads();
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int row = blockIdx.x * gemv::kWarps + warp;
  if (row >= n) return;
  float acc[kRows];
  bf12_gemv::row_dots<kRows, kSB>(w, rows, esc, raw, row, k, row_bytes, sx, lane, acc);
  if (lane != 0) return;
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const size_t at = static_cast<size_t>(r) * n + row;
    if (kOutF32)
      static_cast<float*>(out)[at] = acc[r];
    else
      static_cast<uint16_t*>(out)[at] = float_to_bf16_bits(acc[r]);
  }
}

// One window of a row kept bf16 (the wide kernel's raw rows): the
// production chain over columns [col0, col0 + cols) — bf16_gemv::row_dots'
// steps of it, in its order, into the running accumulators. sk: the staged
// window's row stride.
template <int kRows>
__device__ __forceinline__ void raw_window(const uint16_t* __restrict__ w_row, int col0, int cols,
                                           const uint16_t* __restrict__ sx, int sk, int lane,
                                           float (&acc)[kRows]) {
  uint4 wv[4];
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const int c = t * 256 + lane * 8;
    wv[t] = c < cols ? *reinterpret_cast<const uint4*>(w_row + col0 + c) : make_uint4(0u, 0u, 0u, 0u);
  }
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const int c = t * 256 + lane * 8;
    if (c >= cols) break;  // per lane: its later chunks are past the window too
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const uint4 xv = *reinterpret_cast<const uint4*>(sx + static_cast<size_t>(r) * sk + c);
      bf16_gemv::fma_pair(wv[t].x, xv.x, acc[r]);
      bf16_gemv::fma_pair(wv[t].y, xv.y, acc[r]);
      bf16_gemv::fma_pair(wv[t].z, xv.z, acc[r]);
      bf16_gemv::fma_pair(wv[t].w, xv.w, acc[r]);
    }
  }
}

// The WIDE form (2026-09-19): five to eight activation rows — the batch the
// bf16 sites hand to cuBLASLt today, which reads the bf16 bytes once at
// ~210 GB/s — and the narrow rows whose whole staged rows pass the smem
// bound (k = 8192 from four rows). The activations are staged one
// super-block (1024 columns) at a time, 16 KB at eight rows — the row's
// tail, when it has one, as a last and shorter window — with a barrier on
// both sides of every restage; a lane's accumulators live across the
// windows, so a row's chain is still the scalar one: bitwise the GEMV
// chunks at any m. Measured at eight rows against cuBLASLt (real weights,
// cold): 80 -> 57 us on the KDA q slice, 1,400 -> 940 us on the lm head.
template <int kRows, bool kOutF32>
__global__ void bf12_gemv_wide_kernel(const uint16_t* __restrict__ act, size_t act_stride,
                                      const uint8_t* __restrict__ w,
                                      const uint32_t* __restrict__ rows,
                                      const uint32_t* __restrict__ esc,
                                      const uint16_t* __restrict__ raw, void* __restrict__ out,
                                      int n, int k, size_t row_bytes) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint16_t sx[];
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int row = blockIdx.x * gemv::kWarps + warp;
  // A warp past n keeps staging and meeting the barriers (they are the
  // block's), and consumes nothing.
  const bool live = row < n;
  const Tail tl = tail_of(k);
  const int nsb = tl.nsb;
  const int windows = nsb + (tl.cols != 0 ? 1 : 0);
  const uint8_t* wr = w + static_cast<size_t>(live ? row : 0) * row_bytes;
  const uint32_t eb = live ? rows[2 * row] : 0u;
  const uint32_t base23 = live ? rows[2 * row + 1] : 0u;
  const uint32_t ee = live ? rows[2 * row + 2] : 0u;
  const bool raw_row = (base23 & kBf12RawRow) != 0u;
  const uint16_t* raw_w = raw + static_cast<size_t>(base23 & ~kBf12RawRow) * k;
  float acc[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r) acc[r] = 0.f;
  for (int wi = 0; wi < windows; ++wi) {
    const bool tail = wi >= nsb;
    const int cols = tail ? tl.cols : kBf12Super;  // the staged window's row stride too
    gemv::stage_activations<kRows>(act + static_cast<size_t>(wi) * kBf12Super, act_stride, cols, sx);
    __syncthreads();
    if (live) {
      const int cw = lane * 8;                 // the window's column
      const int cb = wi * kBf12Super + cw;     // the row's
      if (raw_row) {
        raw_window<kRows>(raw_w, wi * kBf12Super, cols, sx, cols, lane, acc);
      } else if (tail) {
        bf12_gemv::TailLoads v;
        bf12_gemv::load_tail(wr + static_cast<size_t>(nsb) * kBf12SuperBytes, tl, lane, v);
        bf12_gemv::consume_tail<kRows>(v, tl, lane, base23, sx, cols, cw, cb, esc, eb, ee, acc);
      } else {
        const uint8_t* p = wr + static_cast<size_t>(wi) * kBf12SuperBytes + lane * 16;
        const uint4 a = *reinterpret_cast<const uint4*>(p);
        const uint4 b = *reinterpret_cast<const uint4*>(p + 512);
        const uint4 e = *reinterpret_cast<const uint4*>(p + 1024);
        consume_step<kRows>(a.x, a.y, e.x, base23, sx, kBf12Super, cw, cb, esc, eb, ee, acc);
        consume_step<kRows>(a.z, a.w, e.y, base23, sx, kBf12Super, cw + 256, cb + 256, esc, eb, ee, acc);
        consume_step<kRows>(b.x, b.y, e.z, base23, sx, kBf12Super, cw + 512, cb + 512, esc, eb, ee, acc);
        consume_step<kRows>(b.z, b.w, e.w, base23, sx, kBf12Super, cw + 768, cb + 768, esc, eb, ee, acc);
      }
    }
    // Everyone is done with sx before it is restaged.
    if (wi + 1 < windows) __syncthreads();
  }
  if (!live) return;
  gemv::warp_reduce<kRows>(acc);
  if (lane != 0) return;
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const size_t at = static_cast<size_t>(r) * n + row;
    if (kOutF32)
      static_cast<float*>(out)[at] = acc[r];
    else
      static_cast<uint16_t*>(out)[at] = float_to_bf16_bits(acc[r]);
  }
}

// The multi-problem kernel (bf16_gemv.cu's, over packed rows): a block
// finds its problem by the prefix table — field-wise selects, no runtime
// reference into the parameter struct — and runs the single launch's work.
struct Bf12GemvMulti {
  Bf12GemvProblem p[4];
  int block_end[4];  // exclusive prefix of blocks per problem
  int n;
};

template <int kRows, bool kOutF32, int kSB>
__global__ void bf12_gemv_multi_kernel(Bf12GemvMulti mp, int k, size_t row_bytes) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint16_t sx[];
  const int bid = static_cast<int>(blockIdx.x);
  int which = 0;
#pragma unroll
  for (int i = 0; i < 3; ++i) which += (i + 1 < mp.n && bid >= mp.block_end[i]) ? 1 : 0;
#define DGPP_PICK(field) \
  (which == 0 ? mp.p[0].field : which == 1 ? mp.p[1].field : which == 2 ? mp.p[2].field : mp.p[3].field)
  const uint16_t* act = DGPP_PICK(act);
  const size_t act_stride = DGPP_PICK(act_row_stride);
  const uint8_t* packed = DGPP_PICK(packed.packed);
  const uint32_t* rows = DGPP_PICK(packed.rows);
  const uint32_t* esc = DGPP_PICK(packed.esc);
  const uint16_t* raw = DGPP_PICK(packed.raw);
  const uint16_t* weight = DGPP_PICK(weight);
  void* out = DGPP_PICK(out);
  const int n = DGPP_PICK(n);
#undef DGPP_PICK
  const int block0 = which == 0 ? 0 : which == 1 ? mp.block_end[0] : which == 2 ? mp.block_end[1] : mp.block_end[2];
  const int block = bid - block0;
  gemv::stage_activations<kRows>(act, act_stride, k, sx);
  __syncthreads();
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int row = block * gemv::kWarps + warp;
  if (row >= n) return;
  float acc[kRows];
  // Uniform across the block: a block is one problem's.
  if (packed != nullptr)
    bf12_gemv::row_dots<kRows, kSB>(packed, rows, esc, raw, row, k, row_bytes, sx, lane, acc);
  else
    bf16_gemv::row_dots<kRows>(weight + static_cast<size_t>(row) * k, sx, k, lane, acc);
  if (lane != 0) return;
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const size_t at = static_cast<size_t>(r) * n + row;
    if (kOutF32)
      static_cast<float*>(out)[at] = acc[r];
    else
      static_cast<uint16_t*>(out)[at] = float_to_bf16_bits(acc[r]);
  }
}

// Whole super-blocks in flight four at a time when the row is a multiple
// of four of them (k = 4096: twelve loads), pairs otherwise (k = 2048: six;
// k = 2560: six and the tail's two).
inline bool four_in_flight(int k) { return k >= 4 * kBf12Super && (k / kBf12Super) % 4 == 0; }

template <int kRows>
void launch_multi_rows(const Bf12GemvMulti& mp, bool out_f32, int k, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>(mp.block_end[mp.n - 1]));
  const size_t smem = gemv::smem_bytes(kRows, k);
  const size_t rb = bf12_row_bytes(k);
  const bool four = four_in_flight(k);
  if (out_f32) {
    if (four) bf12_gemv_multi_kernel<kRows, true, 4><<<grid, gemv::kThreads, smem, stream>>>(mp, k, rb);
    else bf12_gemv_multi_kernel<kRows, true, 2><<<grid, gemv::kThreads, smem, stream>>>(mp, k, rb);
  } else {
    if (four) bf12_gemv_multi_kernel<kRows, false, 4><<<grid, gemv::kThreads, smem, stream>>>(mp, k, rb);
    else bf12_gemv_multi_kernel<kRows, false, 2><<<grid, gemv::kThreads, smem, stream>>>(mp, k, rb);
  }
  DGPP_CUDA_OK(cudaGetLastError());
}

template <int kRows>
void launch_wide(const uint16_t* act, size_t act_stride, const Bf12Matrix& w, void* out,
                 bool out_f32, cudaStream_t stream) {
  const dim3 grid((w.n + gemv::kWarps - 1) / gemv::kWarps);
  const size_t smem = gemv::smem_bytes(kRows, kBf12Super);
  const size_t rb = bf12_row_bytes(w.k);
  if (out_f32)
    bf12_gemv_wide_kernel<kRows, true><<<grid, gemv::kThreads, smem, stream>>>(
        act, act_stride, w.packed, w.rows, w.esc, w.raw, out, w.n, w.k, rb);
  else
    bf12_gemv_wide_kernel<kRows, false><<<grid, gemv::kThreads, smem, stream>>>(
        act, act_stride, w.packed, w.rows, w.esc, w.raw, out, w.n, w.k, rb);
  DGPP_CUDA_OK(cudaGetLastError());
}

template <int kRows, int kSB>
void launch_rows(const uint16_t* act, size_t act_stride, const Bf12Matrix& w, void* out,
                 bool out_f32, cudaStream_t stream) {
  const dim3 grid((w.n + gemv::kWarps - 1) / gemv::kWarps);
  const size_t smem = gemv::smem_bytes(kRows, w.k);
  const size_t rb = bf12_row_bytes(w.k);
  if (out_f32)
    bf12_gemv_kernel<kRows, true, kSB><<<grid, gemv::kThreads, smem, stream>>>(
        act, act_stride, w.packed, w.rows, w.esc, w.raw, out, w.n, w.k, rb);
  else
    bf12_gemv_kernel<kRows, false, kSB><<<grid, gemv::kThreads, smem, stream>>>(
        act, act_stride, w.packed, w.rows, w.esc, w.raw, out, w.n, w.k, rb);
  DGPP_CUDA_OK(cudaGetLastError());
}

template <int kRows>
void launch_depth(const uint16_t* act, size_t act_stride, const Bf12Matrix& w, void* out,
                  bool out_f32, cudaStream_t stream) {
  if (four_in_flight(w.k))
    launch_rows<kRows, 4>(act, act_stride, w, out, out_f32, stream);
  else
    launch_rows<kRows, 2>(act, act_stride, w, out, out_f32, stream);
}

// ---- expansion (bf12-only residency's prefill path) ------------------------

using bf12_gemv::expand_step;

// A warp is one weight row, a lane its sixteen bytes of every segment — the
// GEMV's ownership, so every load and every store is a whole line's share
// and a row's 2k output bytes are written by its 32 lanes without overlap.
// kSB super-blocks of loads are in flight before any is consumed.
template <int kSB>
__global__ void bf12_expand_kernel(const uint8_t* __restrict__ w, const uint32_t* __restrict__ rows,
                                   const uint32_t* __restrict__ esc,
                                   const uint16_t* __restrict__ raw, uint16_t* __restrict__ out,
                                   int row0, int nrows, int k, size_t row_bytes) {
  dgpp_pdl_wait();
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int local = blockIdx.x * gemv::kWarps + warp;
  if (local >= nrows) return;
  const int row = row0 + local;
  const Tail tl = tail_of(k);
  const int nsb = tl.nsb;
  const uint8_t* wr = w + static_cast<size_t>(row) * row_bytes;
  const uint32_t eb = rows[2 * row];
  const uint32_t word = rows[2 * row + 1];
  const uint32_t ee = rows[2 * row + 2];
  uint16_t* o = out + static_cast<size_t>(local) * k;
  if ((word & kBf12RawRow) != 0u) {
    // A row kept bf16: its lane's share of the side array, as it is.
    const uint16_t* rw = raw + static_cast<size_t>(word & ~kBf12RawRow) * k;
    for (int c = lane * 8; c < k; c += 256)
      *reinterpret_cast<uint4*>(o + c) = *reinterpret_cast<const uint4*>(rw + c);
    return;
  }
  const uint32_t base7 = (word >> 23) << 7;
  for (int s0 = 0; s0 < nsb; s0 += kSB) {
    uint4 a[kSB], b[kSB], e[kSB];
#pragma unroll
    for (int s = 0; s < kSB; ++s) {
      const int sb = s0 + s < nsb ? s0 + s : s0;
      const uint8_t* p = wr + static_cast<size_t>(sb) * kBf12SuperBytes + lane * 16;
      a[s] = *reinterpret_cast<const uint4*>(p);
      b[s] = *reinterpret_cast<const uint4*>(p + 512);
      e[s] = *reinterpret_cast<const uint4*>(p + 1024);
    }
#pragma unroll
    for (int s = 0; s < kSB; ++s) {
      if (s0 + s >= nsb) break;
      const int cb = (s0 + s) * kBf12Super + lane * 8;
      const uint4 v0 = expand_step(a[s].x, a[s].y, e[s].x, base7, cb, esc, eb, ee);
      const uint4 v1 = expand_step(a[s].z, a[s].w, e[s].y, base7, cb + 256, esc, eb, ee);
      const uint4 v2 = expand_step(b[s].x, b[s].y, e[s].z, base7, cb + 512, esc, eb, ee);
      const uint4 v3 = expand_step(b[s].z, b[s].w, e[s].w, base7, cb + 768, esc, eb, ee);
      *reinterpret_cast<uint4*>(o + cb) = v0;
      *reinterpret_cast<uint4*>(o + cb + 256) = v1;
      *reinterpret_cast<uint4*>(o + cb + 512) = v2;
      *reinterpret_cast<uint4*>(o + cb + 768) = v3;
    }
  }
  if (tl.cols == 0) return;
  // The tail's units, in column order (bf12_gemv.cuh).
  bf12_gemv::TailLoads v;
  bf12_gemv::load_tail(wr + static_cast<size_t>(nsb) * kBf12SuperBytes, tl, lane, v);
  const int odd = lane & 1;
  int c = nsb * kBf12Super + lane * 8;
  if (tl.pair != 0) {
    const uint4 v0 = expand_step(v.pair_sm.x, v.pair_sm.y, odd ? v.pair_exp.z : v.pair_exp.x, base7, c, esc, eb, ee);
    const uint4 v1 = expand_step(v.pair_sm.z, v.pair_sm.w, odd ? v.pair_exp.w : v.pair_exp.y, base7, c + 256, esc, eb, ee);
    *reinterpret_cast<uint4*>(o + c) = v0;
    *reinterpret_cast<uint4*>(o + c + 256) = v1;
    c += 512;
  }
  if (tl.full != 0) {
    *reinterpret_cast<uint4*>(o + c) =
        expand_step(odd ? v.full_sm.z : v.full_sm.x, odd ? v.full_sm.w : v.full_sm.y,
                    bf12_gemv::word_of(v.full_exp, lane & 3), base7, c, esc, eb, ee);
    c += 256;
  }
  if (lane < tl.lanes)
    *reinterpret_cast<uint4*>(o + c) =
        expand_step(odd ? v.part_sm.z : v.part_sm.x, odd ? v.part_sm.w : v.part_sm.y,
                    bf12_gemv::word_of(v.part_exp, lane & 3), base7, c, esc, eb, ee);
}

// The packed position of element c of a row: byte and nibble offsets.
struct Slot {
  size_t sm;   // the sign+mantissa byte
  size_t exp;  // the byte holding the nibble
  int shift;   // 0 (even element) or 4
};
inline Slot slot_of(int c, const Tail& tl) {
  const int sb = c / kBf12Super;
  const int j = c % 8, shift = 4 * (j & 1);
  if (sb < tl.nsb) {
    const int in = c % kBf12Super;
    const int t = in / 256, lane = (in % 256) / 8;
    const size_t blk = static_cast<size_t>(sb) * kBf12SuperBytes;
    return {blk + (t < 2 ? 0 : 512) + static_cast<size_t>(lane) * 16 + (t & 1) * 8 + j,
            blk + 1024 + static_cast<size_t>(lane) * 16 + t * 4 + j / 2, shift};
  }
  // The tail's units: [P]? [F32]? [F lanes]? (bf12_gemv.cuh).
  const int u = c - tl.nsb * kBf12Super;
  int step = u / 256;
  const size_t lane = static_cast<size_t>((u % 256) / 8);
  size_t off = static_cast<size_t>(tl.nsb) * kBf12SuperBytes;
  if (tl.pair != 0) {
    if (step < 2) return {off + lane * 16 + step * 8 + j, off + 512 + lane * 8 + step * 4 + j / 2, shift};
    off += bf12_gemv::kTailPairBytes;
    step -= 2;
  }
  if (tl.full != 0) {
    if (step == 0) return {off + lane * 8 + j, off + 256 + lane * 4 + j / 2, shift};
    off += bf12_gemv::kTailFullBytes;
  }
  return {off + lane * 8 + j, off + bf12_gemv::tail_part_sm_bytes(tl.lanes) + lane * 4 + j / 2, shift};
}

}  // namespace

Bf12Host bf12_encode(const uint16_t* w, int n, int k, int threads) {
  constexpr uint32_t kRawMark = 0xFFFFFFFFu;
  Bf12Host h;
  if (w == nullptr || !bf12_shape_ok(n, k)) return h;
  if (threads <= 0) threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  threads = std::min(threads, n);
  const auto range = [&](int t) {
    return std::pair<int, int>{static_cast<int>(static_cast<int64_t>(n) * t / threads),
                               static_cast<int>(static_cast<int64_t>(n) * (t + 1) / threads)};
  };
  const size_t row_bytes = bf12_packed_bytes(1, k);
  h.packed.assign(static_cast<size_t>(n) * row_bytes, 0);
  h.rows.assign((static_cast<size_t>(n) + 1) * 2, 0);
  std::vector<std::vector<uint32_t>> escs(threads);
  std::vector<uint32_t> counts(static_cast<size_t>(n), 0);
  // The slot of every column, once (the rows share it).
  std::vector<Slot> slots(static_cast<size_t>(k));
  const Tail tl = tail_of(k);
  for (int c = 0; c < k; ++c) slots[c] = slot_of(c, tl);
  {
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
      pool.emplace_back([&, t] {
        const auto [r0, r1] = range(t);
        std::array<uint32_t, 256> hist;
        for (int row = r0; row < r1; ++row) {
          const uint16_t* src = w + static_cast<size_t>(row) * k;
          // The row's window: the fifteen consecutive exponents holding the
          // most of its weights.
          hist.fill(0);
          for (int c = 0; c < k; ++c) ++hist[(src[c] >> 7) & 0xFF];
          uint32_t in = 0;
          for (int i = 0; i < kBf12Window; ++i) in += hist[i];
          uint32_t best = in;
          int base = 0;
          for (int b = 1; b + kBf12Window <= 256; ++b) {
            in += hist[b + kBf12Window - 1] - hist[b - 1];
            if (in > best) {
              best = in;
              base = b;
            }
          }
          h.rows[2 * static_cast<size_t>(row) + 1] = static_cast<uint32_t>(base) << 23;
          if (static_cast<uint32_t>(k) - best > static_cast<uint32_t>(kBf12MaxRowEscapes)) {
            counts[row] = kRawMark;  // kept bf16: indexed after the join
            continue;
          }
          uint8_t* dst = h.packed.data() + static_cast<size_t>(row) * row_bytes;
          for (int c = 0; c < k; ++c) {
            const uint16_t v = src[c];
            int code = static_cast<int>((v >> 7) & 0xFF) - base;
            if (code < 0 || code >= kBf12Window) {
              code = 15;
              escs[t].push_back((static_cast<uint32_t>(c) << 16) | v);
              ++counts[row];
            }
            const Slot& s = slots[c];
            dst[s.sm] = static_cast<uint8_t>(((v >> 8) & 0x80) | (v & 0x7F));
            dst[s.exp] = static_cast<uint8_t>(dst[s.exp] | (code << s.shift));
          }
        }
      });
    for (auto& th : pool) th.join();
  }
  uint32_t at = 0;
  for (int row = 0; row < n; ++row) {
    h.rows[2 * static_cast<size_t>(row)] = at;
    if (counts[row] == kRawMark) {
      h.rows[2 * static_cast<size_t>(row) + 1] = kBf12RawRow | static_cast<uint32_t>(h.raw_rows++);
      const uint16_t* src = w + static_cast<size_t>(row) * k;
      h.raw.insert(h.raw.end(), src, src + k);
      continue;
    }
    at += counts[row];
    h.max_row_escapes = std::max(h.max_row_escapes, static_cast<int>(counts[row]));
  }
  if (h.raw.empty()) h.raw.assign(static_cast<size_t>(k), 0);  // a valid device array
  h.rows[2 * static_cast<size_t>(n)] = at;
  h.escapes = at;
  h.esc.reserve(static_cast<size_t>(at) + 1);
  for (const auto& part : escs) h.esc.insert(h.esc.end(), part.begin(), part.end());
  if (h.esc.empty()) h.esc.push_back(0);  // a valid device array either way
  h.ok = static_cast<int64_t>(h.raw_rows) * kBf12MaxRawFraction <= n;
  return h;
}

void bf12_decode(const Bf12Host& h, int n, int k, uint16_t* out) {
  if (!bf12_shape_ok(n, k) || h.packed.size() != bf12_packed_bytes(n, k))
    throw std::invalid_argument("bf12_decode: shape does not match the packed form");
  const size_t row_bytes = bf12_packed_bytes(1, k);
  const Tail tl = tail_of(k);
  for (int row = 0; row < n; ++row) {
    const uint8_t* src = h.packed.data() + static_cast<size_t>(row) * row_bytes;
    uint32_t e = h.rows[2 * static_cast<size_t>(row)];
    const uint32_t end = h.rows[2 * static_cast<size_t>(row) + 2];
    const uint32_t word = h.rows[2 * static_cast<size_t>(row) + 1];
    if ((word & kBf12RawRow) != 0u) {
      std::memcpy(out + static_cast<size_t>(row) * k,
                  h.raw.data() + static_cast<size_t>(word & ~kBf12RawRow) * k,
                  static_cast<size_t>(k) * 2);
      continue;
    }
    const uint32_t base = word >> 23;
    for (int c = 0; c < k; ++c) {
      const Slot s = slot_of(c, tl);
      const uint32_t b = src[s.sm];
      const uint32_t code = (src[s.exp] >> s.shift) & 0xFu;
      if (code == 15u) {
        // Column-sorted within the row: the next entry is this column's.
        if (e >= end || (h.esc[e] >> 16) != static_cast<uint32_t>(c))
          throw std::runtime_error("bf12_decode: escape table out of step");
        out[static_cast<size_t>(row) * k + c] = static_cast<uint16_t>(h.esc[e++] & 0xFFFFu);
      } else {
        out[static_cast<size_t>(row) * k + c] = static_cast<uint16_t>(
            ((b & 0x80u) << 8) | ((base + code) << 7) | (b & 0x7Fu));
      }
    }
  }
}

bool bf12_gemv_accepts(const Bf12Matrix& w, int m) {
  return w.packed != nullptr && w.rows != nullptr && w.esc != nullptr && w.raw != nullptr &&
         gemv::aligned16(w.raw) && bf12_shape_ok(w.n, w.k) && m >= 1 && m <= kBf12MaxRows &&
         gemv::aligned16(w.packed);
}

void launch_bf12_expand(const Bf12Matrix& w, int row0, int rows, uint16_t* out,
                        cudaStream_t stream) {
  if (!bf12_gemv_accepts(w, 1)) throw std::invalid_argument("bf12_expand: not a packed matrix");
  if (out == nullptr || !gemv::aligned16(out))
    throw std::invalid_argument("bf12_expand: output must be 16-byte aligned");
  if (row0 < 0 || rows <= 0 || row0 > w.n - rows)
    throw std::invalid_argument("bf12_expand: rows outside the matrix");
  const dim3 grid((rows + gemv::kWarps - 1) / gemv::kWarps);
  const size_t rb = bf12_row_bytes(w.k);
  if (four_in_flight(w.k))
    bf12_expand_kernel<4><<<grid, gemv::kThreads, 0, stream>>>(w.packed, w.rows, w.esc, w.raw, out,
                                                                 row0, rows, w.k, rb);
  else
    bf12_expand_kernel<2><<<grid, gemv::kThreads, 0, stream>>>(w.packed, w.rows, w.esc, w.raw, out,
                                                                 row0, rows, w.k, rb);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_bf12_gemv(const uint16_t* act, size_t act_row_stride, const Bf12Matrix& w,
                      void* out, bool out_f32, int m, cudaStream_t stream) {
  if (!act || !out) throw std::invalid_argument("bf12_gemv: null pointer");
  if (!bf12_gemv_accepts(w, m))
    throw std::invalid_argument("bf12_gemv: shape outside the GEMV contract");
  if (act_row_stride < static_cast<size_t>(w.k))
    throw std::invalid_argument("bf12_gemv: activation stride narrower than k");
  // Whole staged rows while they fit the smem bound and the register
  // budget (four rows), the windowed form past either.
  const bool narrow = m <= gemv::kMaxRows && gemv::smem_fits(m, w.k);
  switch (m) {
    case 1: launch_depth<1>(act, act_row_stride, w, out, out_f32, stream); break;
    case 2:
      if (narrow) launch_depth<2>(act, act_row_stride, w, out, out_f32, stream);
      else launch_wide<2>(act, act_row_stride, w, out, out_f32, stream);
      break;
    case 3:
      if (narrow) launch_depth<3>(act, act_row_stride, w, out, out_f32, stream);
      else launch_wide<3>(act, act_row_stride, w, out, out_f32, stream);
      break;
    case 4:
      if (narrow) launch_depth<4>(act, act_row_stride, w, out, out_f32, stream);
      else launch_wide<4>(act, act_row_stride, w, out, out_f32, stream);
      break;
    case 5: launch_wide<5>(act, act_row_stride, w, out, out_f32, stream); break;
    case 6: launch_wide<6>(act, act_row_stride, w, out, out_f32, stream); break;
    case 7: launch_wide<7>(act, act_row_stride, w, out, out_f32, stream); break;
    case 8: launch_wide<8>(act, act_row_stride, w, out, out_f32, stream); break;
    default: throw std::invalid_argument("bf12_gemv: m outside 1..8");
  }
}

void launch_bf12_gemv_multi(const Bf12GemvProblem* problems, int n_problems, bool out_f32, int m,
                            int k, cudaStream_t stream) {
  if (n_problems <= 0 || n_problems > kBf12GemvMaxProblems || problems == nullptr)
    throw std::invalid_argument("bf12_gemv_multi: 1..4 problems");
  Bf12GemvMulti mp{};
  mp.n = n_problems;
  int blocks = 0;
  for (int i = 0; i < n_problems; ++i) {
    const Bf12GemvProblem& p = problems[i];
    if (p.n <= 0 || !p.act || !p.out) throw std::invalid_argument("bf12_gemv_multi: empty or null problem");
    if (p.packed.packed != nullptr) {
      if (!bf12_gemv_accepts(p.packed, m) || p.packed.k != k || p.packed.n != p.n)
        throw std::invalid_argument("bf12_gemv_multi: the companion's shape disagrees with the problem");
      if (m > gemv::kMaxRows || !gemv::smem_fits(m, k))
        throw std::invalid_argument("bf12_gemv_multi: rows outside the narrow GEMV contract");
    } else if (!p.weight || !bf16_gemv_accepts(p.weight, m, k)) {
      throw std::invalid_argument("bf12_gemv_multi: shape outside the GEMV contract");
    }
    if (p.act_row_stride < static_cast<size_t>(k))
      throw std::invalid_argument("bf12_gemv_multi: activation stride narrower than k");
    mp.p[i] = p;
    blocks += (p.n + gemv::kWarps - 1) / gemv::kWarps;
    mp.block_end[i] = blocks;
  }
  for (int i = n_problems; i < 4; ++i) mp.block_end[i] = blocks;
  switch (m) {
    case 1: launch_multi_rows<1>(mp, out_f32, k, stream); break;
    case 2: launch_multi_rows<2>(mp, out_f32, k, stream); break;
    case 3: launch_multi_rows<3>(mp, out_f32, k, stream); break;
    case 4: launch_multi_rows<4>(mp, out_f32, k, stream); break;
    default: throw std::invalid_argument("bf12_gemv_multi: m outside 1..4");
  }
}

}  // namespace dgpp
