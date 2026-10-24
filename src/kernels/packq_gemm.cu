#include <stdexcept>
#include "kernels/pdl.cuh"
#include <algorithm>
#include <cstdlib>
#include <string>
#include <type_traits>

#include <cuda_bf16.h>

#include "common/cuda_check.hpp"
#include "kernels/packq_gemm.hpp"
#include "kernels/packq_a8_gemv.cuh"
#include "kernels/packq_gemv.cuh"

namespace dgpp {

// The kernel choice: 1 = the wide tile (the default for int4 rows), 0 =
// the narrow kernel; engine.expert_gemm = "narrow" turns the default to 0.
// The form and the prefetch distance are deployment settings
// (engine.expert_gemm, engine.expert_gemm_prefetch; 2026-09-30 — no
// environment switch selects a kernel), set once before the first launch.
namespace {
int g_packq_form = 1;      // the wide decoded-tile kernel
int g_packq_prefetch = 3;  // steps ahead
}  // namespace
int packq_gemm_form_index(const std::string& name) {
  if (name == "narrow") return 0;
  if (name == "wide") return 1;
  if (name == "wide3") return 2;   // the register-decode three-stage form
  if (name == "wide4") return 4;   // four warps at 64 x 32 (round 21)
  if (name == "wide4r") return 5;  // four warps, register decode, three stages
  return -1;
}
void packq_gemm_set_form(int variant) { g_packq_form = variant; }
void packq_gemm_set_prefetch(int steps) { g_packq_prefetch = std::max(0, steps); }
int packq_gemm_variant_default() { return g_packq_form; }

// engine.expert_gemm_prefetch = N: the wide kernel prefetches step g+N's
// lines into L2 while issuing step g (0 = off). Default 3 (2026-09-29, the
// device-memory bench at 4,096 tokens x top-10 over 512 experts, Zipf
// routing: gate 4.6-5.0 -> 3.5-3.7 ms at any distance 1-3, down 6.8-7.1
// -> 6.0 at 3; the managed-memory bench had hidden it behind its page
// mapping cost). The two-stage pipeline holds one step of latency; the
// weight stream is the launch's DRAM traffic and the prefetch keeps it
// ahead of the copies.
int packq_gemm_prefetch_ahead() { return g_packq_prefetch; }

namespace {
constexpr int kM = 32, kN = 64, kThreads = 128;
// The k-step: 64 codes, or one 128-code group for the int8 head's plane
// layout (kernels/packq_head.hpp), whose fragments only reassemble
// group-wise.
template <bool Planes>
struct Tile {
  static constexpr int kK = Planes ? 128 : 64;
  static constexpr int kActStride = kK + 8;
};

__device__ __forceinline__ void copy16(void* dst, const void* src, bool valid) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address), "l"(src),
               "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void commit() {
  asm volatile("cp.async.commit_group;" ::);
}
__device__ __forceinline__ void wait() {
  asm volatile("cp.async.wait_group 0;" ::);
}
__device__ __forceinline__ void load_a(uint32_t (&a)[4], const uint16_t* src) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(src));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(address));
}

// NF4I8 indices to the bf16 bits of their codebook levels (every level is
// an integer of magnitude <= 127: exact in bf16), the even k in the low
// half as the offset decode puts it. The sixteen levels' bf16 patterns are
// two byte tables (low and high bytes, entries 0-7 and 8-15 as PRMT's
// eight-byte windows); four indices in the low nibbles of `h`'s four
// bytes (the masks below take them from a 16-bit half-word) select their
// bytes with one PRMT per window and the index's bit 3 picks the window
// (the mask PRMT of expand_nf4_int8) — no integer-to-float conversion.
// Bitwise the __floats2bfloat162_rn of the int8 levels it replaced
// (2026-10-09; the conversions were the decoded tile's cost).
__device__ __forceinline__ uint32_t codebook_bf16_bytes(uint32_t sel, uint32_t win, uint32_t t0,
                                                        uint32_t t1, uint32_t t2, uint32_t t3) {
  const uint32_t a = __byte_perm(t0, t1, sel);  // entries 0-7
  const uint32_t b = __byte_perm(t2, t3, sel);  // entries 8-15
  return (a & ~win) | (b & win);
}
// bf16 of {-127,-88,-67,-50,-36,-23,-12,0,10,20,31,43,56,71,92,127}:
// c2fe c2b0 c286 c248 c210 c1b8 c140 0000 4120 41a0 41f8 422c 4260 428e 42b8 42fe.
constexpr uint32_t kCbLo0 = 0x4886b0feu, kCbLo1 = 0x0040b810u, kCbLo2 = 0x2cf8a020u, kCbLo3 = 0xfeb88e60u;
constexpr uint32_t kCbHi0 = 0xc2c2c2c2u, kCbHi1 = 0x00c1c1c2u, kCbHi2 = 0x42414141u, kCbHi3 = 0x42424242u;
// Four indices (a 16-bit half-word, low nibble first) to two bf16 pairs:
// .x = codes 0, 1 and .y = codes 2, 3.
__device__ __forceinline__ uint2 decode_half_codebook(uint32_t h) {
  const uint32_t sel = h & 0x7777u;
  const uint32_t win = __byte_perm(0u, 0xFFFFFFFFu, (h & 0x8888u) >> 1);
  const uint32_t lo = codebook_bf16_bytes(sel, win, kCbLo0, kCbLo1, kCbLo2, kCbLo3);
  const uint32_t hi = codebook_bf16_bytes(sel, win, kCbHi0, kCbHi1, kCbHi2, kCbHi3);
  return make_uint2(__byte_perm(lo, hi, 0x5140u), __byte_perm(lo, hi, 0x7362u));
}
// Two indices (one byte) to their bf16 pair.
__device__ __forceinline__ uint32_t decode_byte_codebook(uint32_t byte) {
  return decode_half_codebook(byte & 0xFFu).x;
}
// Eight indices (one packed word) to four bf16 pairs in k order.
__device__ __forceinline__ uint4 decode_word_codebook(uint32_t w) {
  const uint2 lo = decode_half_codebook(w & 0xFFFFu);
  const uint2 hi = decode_half_codebook(w >> 16);
  return make_uint4(lo.x, lo.y, hi.x, hi.y);
}

template <int Bits, bool Planes, int SF = packq_gemv::kScaleBf16G64>
__device__ __forceinline__ uint32_t code_pair(const uint8_t* row, int col) {
  // Offset codes, low nibble/byte first. Every decoded integer is exactly
  // representable in BF16, including int8's -128 and +127 endpoints.
  if constexpr (Bits == 4 && !Planes && packq_gemv::ScaleFmt<SF>::codebook)
    return decode_byte_codebook(row[col / 2]);
  unsigned packed;
  if constexpr (Planes) {
    // `row` holds one 128-code group in the plane layout; codes col (even)
    // and col + 1 reassembled from their three fragments.
    // col is even: codes col and col + 1 share a nibble byte and sit two
    // bits apart in the two-bit planes (packq_planes_pos in packq_head.cu:
    // 2 (j & 3) + {0, 16, 8, 24}[j >> 2] for code j of the chunk).
    constexpr int kBase[4] = {0, 16, 8, 24};
    const int jj = col & 15;
    const int pos = 2 * (jj & 3) + kBase[jj >> 2];
    const unsigned nib = row[col >> 1];
    const unsigned mid = *reinterpret_cast<const uint32_t*>(row + 64 + ((col >> 4) << 2)) >> pos;
    const unsigned lo2 = *reinterpret_cast<const uint32_t*>(row + 96 + ((col >> 4) << 2)) >> pos;
    const unsigned c0 = ((nib & 0xFu) << 4) | ((mid & 3u) << 2) | (lo2 & 3u);
    const unsigned c1 = ((nib >> 4) << 4) | (((mid >> 2) & 3u) << 2) | ((lo2 >> 2) & 3u);
    packed = c0 | (c1 << 8);
  } else {
    packed = Bits == 4 ? row[col / 2] : *reinterpret_cast<const uint16_t*>(row + col);
  }
  constexpr unsigned mask = (1u << Bits) - 1;
  constexpr int bias = 1 << (Bits - 1);
  const float lo = static_cast<float>(static_cast<int>(packed & mask) - bias);
  const float hi = static_cast<float>(static_cast<int>((packed >> Bits) & mask) - bias);
  const __nv_bfloat162 pair = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&pair);
}

template <int Bits, typename OutT, bool Grouped, int SF, bool Planes>
__global__ __launch_bounds__(kThreads) void packq_gemm_kernel(
    const uint16_t* __restrict__ act, size_t act_stride, const uint8_t* __restrict__ weights,
    const uint16_t* __restrict__ scales, const MoeSegment* __restrict__ segs,
    const MoeExpertView* __restrict__ views, int which, const int32_t* __restrict__ act_rows,
    OutT* __restrict__ out, size_t out_stride, int m, int n, int k, int n_tiles, bool vector_act) {
  dgpp_pdl_wait();
  constexpr int kK = Tile<Planes>::kK;
  constexpr int kActStride = Tile<Planes>::kActStride;
  constexpr int kCodeStride = kK * Bits / 8 + 16;
  __shared__ __align__(16) uint16_t a[2][kM][kActStride];
  __shared__ __align__(16) uint8_t b[2][kN][kCodeStride];
  __shared__ float scale[2][kN];
  int row0 = 0;
  if constexpr (Grouped) {
    const MoeSegment seg = segs[blockIdx.y];
    row0 = seg.row0;
    m = seg.rows;
    const MoeExpertView view = views[seg.expert * 3 + which];
    weights = view.payload;
    scales = view.packed_scales;
  }
  const int m0 = (blockIdx.x / n_tiles) * kM;
  if (m0 >= m) return;
  const int n0 = (blockIdx.x % n_tiles) * kN;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int r = lane / 4, cc = (lane % 4) * 2;
  const int groups = k / kK;  // k-steps of one tile
  // The scale groups: kK per step, so a step's scale is the group it lies
  // in (one per step at group 64, one per two steps at group 128).
  constexpr int kScaleGroup = packq_gemv::ScaleFmt<SF>::group;
  const int scale_groups = k / kScaleGroup;
  auto issue = [&](int group, int slot) {
    for (int idx = tid; idx < kM * (kK / 8); idx += kThreads) {
      const int row = idx / (kK / 8), col = (idx % (kK / 8)) * 8;
      const bool valid = m0 + row < m;
      const int gathered = row0 + m0 + row;
      const int source_row = valid ? (act_rows ? act_rows[gathered] : gathered) : 0;
      const uint16_t* src = act + static_cast<size_t>(source_row) * act_stride + group * kK + col;
      if (vector_act) {
        copy16(&a[slot][row][col], src, valid);
      } else {
        // Row maps and odd activation strides are valid; only the aligned
        // case uses 16-byte asynchronous copies.
#pragma unroll
        for (int j = 0; j < 8; ++j) a[slot][row][col + j] = valid ? src[j] : 0;
      }
    }
    constexpr int chunks = kK * Bits / 128;
    for (int idx = tid; idx < kN * chunks; idx += kThreads) {
      const int row = idx / chunks, col = (idx % chunks) * 16;
      const bool valid = n0 + row < n;
      const uint8_t* src = weights;
      if (valid) {
        const uint8_t* rb = weights + static_cast<size_t>(n0 + row) * (k * Bits / 8);
        if constexpr (Planes) {
          // The k-step's 128 codes gathered from the row's three plane runs
          // into the smem row as [64 B nibbles][32 B bits 3..2][32 B bits 1..0].
          const int c = col / 16;  // 0..7
          if (c < 4) src = rb + group * 64 + c * 16;
          else if (c < 6) src = rb + k / 2 + group * 32 + (c - 4) * 16;
          else src = rb + k / 2 + k / 4 + group * 32 + (c - 6) * 16;
        } else {
          src = rb + group * (kK * Bits / 8) + col;
        }
      }
      copy16(&b[slot][row][col], src, valid);
    }
    if (tid < kN)
      scale[slot][tid] =
          n0 + tid < n
              ? packq_gemv::ScaleFmt<SF>::to_float(
                    scales[static_cast<size_t>(n0 + tid) * scale_groups + group / (kScaleGroup / kK)])
              : 0.f;
  };

  float acc[2][2][4] = {};
  issue(0, 0);
  commit();
  for (int group = 0; group < groups; ++group) {
    const int slot = group % 2;
    wait();
    __syncthreads();
    if (group + 1 < groups) issue(group + 1, slot ^ 1);
    commit();
    // The scale is applied per 64 codes whatever the k-step (the plane
    // layout's 128-code step runs as two halves), so the fp32 chain is the
    // 64-code kernel's bit for bit.
#pragma unroll
    for (int h = 0; h < kK; h += 64) {
    float partial[2][2][4] = {};
#pragma unroll
    for (int kk = h; kk < h + 64; kk += 16) {
      uint32_t bf[2][2];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const uint8_t* row = b[slot][warp * 16 + j * 8 + r];
        bf[j][0] = code_pair<Bits, Planes, SF>(row, kk + cc);
        bf[j][1] = code_pair<Bits, Planes, SF>(row, kk + cc + 8);
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        uint32_t af[4];
        load_a(af, &a[slot][i * 16 + lane % 16][kk + (lane / 16) * 8]);
#pragma unroll
        for (int j = 0; j < 2; ++j)
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(partial[i][j][0]), "+f"(partial[i][j][1]), "+f"(partial[i][j][2]),
                "+f"(partial[i][j][3])
              : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[j][0]), "r"(bf[j][1]));
      }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int col = warp * 16 + j * 8 + cc;
#pragma unroll
        for (int v = 0; v < 4; ++v)
          acc[i][j][v] = __fmaf_rn(scale[slot][col + (v % 2)], partial[i][j][v], acc[i][j][v]);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const int row = m0 + i * 16 + r + (v / 2) * 8;
        const int col = n0 + warp * 16 + j * 8 + cc + (v % 2);
        if (row < m && col < n) {
          const size_t index = static_cast<size_t>(row0 + row) * out_stride + col;
          if constexpr (std::is_same_v<OutT, float>)
            out[index] = acc[i][j][v];
          else
            out[index] = __bfloat16_as_ushort(__float2bfloat16_rn(acc[i][j][v]));
        }
      }
}


// ---- the wide tile (2026-09-29, the hybrid's prefill) ------------------------
// The kernel above runs 32 x 64 tiles on four warps and decodes every B
// fragment from its codes on the MMA path: 26 TFLOPS on the Qwen expert
// shape against the ~190 the dense stack's cuBLAS GEMMs reach on this
// GPU. This one runs 64 x 128 x 64 tiles on eight warps, a two-stage
// cp.async pipeline for the activation rows and the code rows (46 KB of
// shared memory: two blocks per SM, sixteen warps to hide the ldmatrix ->
// mma latencies ncu showed at one block), and
// decodes each k-step's codes once into a bf16 smem tile (every thread
// 32 codes: nibble | 0x4300 is 128 + code in bf16 exactly, less 136.0 in
// bf16 is code - 8 exactly), so the MMA loop is ldmatrix and mma alone.
// Per output element the chain is the kernel above's exactly: four
// m16n8k16 steps per 64-code group in k order into a fresh partial, then
// one fp32 fma with the group's scale — bitwise the same D. Int4, row
// layout, both scale formats; int8 and the plane layout keep the kernel
// above. engine.expert_gemm = "narrow" keeps it for every shape.
constexpr int kWideM = 64, kWideN = 128, kWideK = 64, kWideStages = 2;
constexpr int kWideMinBlocks = 2;
// The register-decode form (2026-09-30, variant 2): no decoded-weight tile
// in shared memory — each warp decodes its B fragments from the staged
// codes on the MMA path (two bytes a fragment, the same decode_byte, the
// same values in the same slots: bitwise) — so three stages (two steps of
// lookahead) fit two blocks per SM.
constexpr int kWideStagesReg = 3;
static_assert(kWideM == kPackqGemmWideRows, "the tile list's row count is the wide kernel's m-tile");
constexpr int kWideAStride = kWideK + 8;         // bf16 elements per staged activation row
constexpr int kWideCodeStride = kWideK / 2;      // bytes per staged int4 code row (consecutive 16-byte halves: no conflicts)
constexpr int kWideDStride = kWideK + 8;         // bf16 elements per decoded weight row
// The Mixed346 form (SF == kScaleBf16G128Mixed346, 2026-10-09): a staged
// code row holds a 64-code step of a 3-, 4- or 6-bit stream (24, 32 or 48
// bytes), and each stage carries the activation scale of its group per
// row (the int8 codes' fp32 scales).
constexpr int kWideCodeStrideM346 = 48;
__host__ __device__ constexpr int wide_code_stride(int sf) {
  return sf == packq_gemv::kScaleBf16G128Mixed346 ? kWideCodeStrideM346 : kWideCodeStride;
}
// The Mixed346 form's activation tile is swizzled instead of padded: rows
// of 64 bf16 (128 bytes) with the 16-byte chunk index xor'ed by the row
// (mod 8), so ldmatrix's eight rows hit eight bank groups as the padded
// stride does — 2 KB less per two stages. Each step's int8 rows reach the
// tile from registers: a thread loads its rows' eight codes for the step
// after the one it writes (one step of latency hiding, the cp.async
// pipeline's), converts them to exact bf16 integers and stores them on
// the issue path (2026-10-10: the earlier cp.async staging of the codes
// plus the conversion after the wait kept an [S][M][64] byte copy that put
// the tile at 58.9 KB — one block per SM, 18 % slower than the int4 tile;
// the swizzle and the register path bring it to 47.5 KB, two blocks). The
// row's activation scale of the step rides the cp.async pipeline ([S][M]
// floats).
constexpr int kWideAStrideM346 = kWideK;
__host__ __device__ constexpr int wide_a_stride(int sf) {
  return sf == packq_gemv::kScaleBf16G128Mixed346 ? kWideAStrideM346 : kWideAStride;
}
template <bool M346>
__device__ __forceinline__ int wide_a_off(int row, int col) {  // col a multiple of 8
  if constexpr (M346) return row * kWideAStrideM346 + ((((col >> 3) ^ (row & 7))) << 3);
  else return row * kWideAStride + col;
}
constexpr size_t wide_smem_bytes(int stages, bool regb, int sf = packq_gemv::kScaleBf16G64) {
  return static_cast<size_t>(stages) * kWideM * wide_a_stride(sf) * 2 +
         static_cast<size_t>(stages) * kWideN * wide_code_stride(sf) +
         (regb ? size_t{0} : static_cast<size_t>(kWideN) * kWideDStride * 2) +
         static_cast<size_t>(stages) * kWideN * 4 +
         (sf == packq_gemv::kScaleBf16G128Mixed346 ? static_cast<size_t>(stages) * kWideM * 4 : size_t{0});
}


__device__ __forceinline__ void wait_pending(int n) {
  if (n == 0) asm volatile("cp.async.wait_group 0;" ::);
  else if (n == 1) asm volatile("cp.async.wait_group 1;" ::);
  else asm volatile("cp.async.wait_group 2;" ::);
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
// Two int4 codes (one byte, the low nibble first) to the bf16 pair
// (code - 8) exactly: 0x4300 | nibble is 128 + nibble, less 136.
// The bf16 pair times a scale, each rounded to bf16 (the folded form).
__device__ __forceinline__ uint32_t fold_scale(uint32_t pair, float s) {
  const float lo = __uint_as_float(pair << 16) * s;
  const float hi = __uint_as_float(pair & 0xFFFF0000u) * s;
  return static_cast<uint32_t>(float_to_bf16_bits(lo)) | (static_cast<uint32_t>(float_to_bf16_bits(hi)) << 16);
}
template <int SF = packq_gemv::kScaleBf16G64>
__device__ __forceinline__ uint32_t decode_byte(uint32_t byte) {
  if constexpr (packq_gemv::ScaleFmt<SF>::codebook) return decode_byte_codebook(byte);
  const uint32_t pair = ((byte & 0xF0u) << 12) | (byte & 0x0Fu) | 0x43004300u;
  __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&pair);
  const uint32_t bias = 0x43084308u;
  v = __hsub2(v, *reinterpret_cast<const __nv_bfloat162*>(&bias));
  return *reinterpret_cast<const uint32_t*>(&v);
}

// Eight- and four-byte cp.async pieces (the 3-bit code rows' eight-byte
// steps, the rows' activation scales).
__device__ __forceinline__ void copy8(void* dst, const void* src, bool valid) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::"r"(address), "l"(src), "r"(valid ? 8 : 0));
}
__device__ __forceinline__ void copy4(void* dst, const void* src, bool valid) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(address), "l"(src), "r"(valid ? 4 : 0));
}

// The block's per-segment state the tile function works from (block-uniform).
template <typename OutT>
struct WideCtx {
  const uint16_t* act;
  size_t act_stride;
  const uint8_t* weights;
  const uint16_t* scales;
  const int32_t* act_rows;
  OutT* out;
  size_t out_stride;
  int row0, m0, n0, m, n, k, groups, scale_groups, row_bytes;
  bool vector_act;
  uint16_t* a;    // [S][M][AStride]
  uint8_t* bc;    // [S][N][CodeStride]
  uint16_t* bd;   // [N][DStride]
  float* scale;   // [S][N]
  int prefetch_ahead = 0;  // steps ahead to prefetch into L2 (0: off)
  // The Mixed346 form (SF == kScaleBf16G128Mixed346): the tile's view is a
  // converted triple (a8: bits 3 / 4 / 6, the rows' int8 codes `act_q` with
  // one fp32 scale per 128 in `act_s`) or an existing expert's int4 g64
  // triple (a8 false: the bf16 rows of `act`, the group-64 chain).
  const int8_t* act_q = nullptr;
  size_t act_q_stride = 0;
  const float* act_s = nullptr;
  size_t act_s_stride = 0;
  int bits = 4;
  bool a8 = false;
  float* ascale = nullptr;  // [S][M]: the step's group activation scale per row (a8)
};

// One 64 x 128 block tile at a segment-aware warp layout (2026-09-29): M16
// is the m16 tiles the segment's rows fill here — 1 (<= 16 rows: eight
// warps, one m16 tile, two n8 tiles each), 2 (<= 32 rows: two m16 tiles,
// four warps each over four n8 tiles) or 4 (the 2 x 4 layout: two m16 and
// four n8 tiles per warp). The MMA and ldmatrix work per block shrinks with
// the rows while the pipeline, the decode and every element's chain (four
// m16n8k16 steps per 64-code group into a fresh partial, one fma with the
// group's scale) stay the same: bitwise across the modes and the narrow
// kernel. The real routing leaves most experts under 64 rows a chunk.
__device__ __forceinline__ void prefetch_l2(const void* p) {
  asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}

// kFold (2026-09-30, engine.prefill_fold_scales; NOT bitwise): each group's
// scale folded into the decoded bf16 weight values — bf16(code x scale) —
// and one fp32 accumulator across K (the Marlin form: no per-group partial,
// no per-group fma). Decoded-tile path only.
// kWarps (2026-09-30, round 21): eight warps at 32 x 32 (the shipped form)
// or four warps at 64 x 32 / 32 x 32 / 16 x 32 by the rows the segment
// fills — each ldmatrix feeds twice the mma and each barrier covers twice
// the work; the staged tiles, the decode and every element's chain are
// the same, so both are bitwise.
template <typename OutT, int SF, int M16, int kStages, bool kRegB, bool kFold, int kWarps>
__device__ __forceinline__ void wide_tile(const WideCtx<OutT>& c) {
  static_assert(kWarps == 8 || kWarps == 4, "eight or four warps");
  constexpr int kThreads = kWarps * 32;
  constexpr int kMI = kWarps == 8 ? (M16 == 4 ? 2 : 1) : M16;          // m16 tiles per warp
  constexpr int kNJ = kWarps == 8 ? (M16 == 1 ? 2 : 4) : 4;            // n8 tiles per warp
  constexpr int kScaleGroup = packq_gemv::ScaleFmt<SF>::group;
  constexpr bool kM346 = SF == packq_gemv::kScaleBf16G128Mixed346;
  constexpr int kCodeStride = wide_code_stride(SF);
  constexpr int kAStride = wide_a_stride(SF);
  // The step's scale column: one per step at group 64, one per two steps at
  // group 128 — an existing expert's tile in a Mixed346 launch is group 64.
  const int scale_div = (kM346 && !c.a8) ? 1 : kScaleGroup / kWideK;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int r = lane / 4, cc = (lane % 4) * 2;
  const int row_base = kWarps == 8 ? (M16 == 4 ? (warp / 4) * 32 : (M16 == 2 ? (warp / 4) * 16 : 0)) : 0;
  const int col_base = kWarps == 8 ? (M16 == 1 ? warp * 16 : (warp % 4) * 32) : warp * 32;
  // The issue path holds no global round trip on the barrier's critical
  // path: each thread's activation rows (eight threads a row, kThreads / 8
  // rows a pass, kACopies passes over the 64-row tile) resolve their
  // source row once here, its code rows likewise, and the column's scale
  // for a step is read one step ahead into a register.
  constexpr int kARows = kThreads / 8, kACopies = kWideM / kARows;
  constexpr int kBCopies = (kWideN * 2) / kThreads;  // 16-byte code pieces a thread stages
  const int arow0 = tid / 8, acol = (tid % 8) * 8;
  int asrc[kACopies];
  bool avalid[kACopies];
#pragma unroll
  for (int h = 0; h < kACopies; ++h) {
    const int row = arow0 + h * kARows;
    avalid[h] = c.m0 + row < c.m;
    const int gathered = c.row0 + c.m0 + row;
    asrc[h] = avalid[h] ? (c.act_rows ? c.act_rows[gathered] : gathered) : 0;
  }
  int brow[kBCopies], bcol[kBCopies];
  bool bvalid[kBCopies];
  const uint8_t* brow_ptr[kBCopies];
#pragma unroll
  for (int h = 0; h < kBCopies; ++h) {
    const int piece = tid + h * kThreads;
    brow[h] = piece / 2;
    bcol[h] = (piece % 2) * 16;
    bvalid[h] = c.n0 + brow[h] < c.n;
    brow_ptr[h] = c.weights + static_cast<size_t>(bvalid[h] ? c.n0 + brow[h] : 0) * c.row_bytes + bcol[h];
  }
  const bool scale_lane = tid < kWideN && c.n0 + tid < c.n;
  const uint16_t* scale_ptr = c.scales + static_cast<size_t>(scale_lane ? c.n0 + tid : 0) * c.scale_groups;
  auto scale_of = [&](int group) -> float {
    return scale_lane ? packq_gemv::ScaleFmt<SF>::to_float(scale_ptr[group / scale_div]) : 0.f;
  };
  // The L2 prefetch of step pg's lines: a row's activation line (the bf16
  // row, or the int8 code row under the Mixed346 a8 form — half a line a
  // step) and each weight row's line when the step starts a new one (the
  // step's row bytes are 8 x bits under Mixed346: 24, 32 or 48; 32 for the
  // int4 rows).
  auto prefetch_step = [&](int pg) {
    if (acol == 0) {
#pragma unroll
      for (int h = 0; h < kACopies; ++h) {
        if (!avalid[h]) continue;
        if (kM346 && c.a8) {
          if ((pg * kWideK) % 128 == 0)
            prefetch_l2(c.act_q + static_cast<size_t>(asrc[h]) * c.act_q_stride + pg * kWideK);
        } else {
          prefetch_l2(c.act + static_cast<size_t>(asrc[h]) * c.act_stride + pg * kWideK);
        }
      }
    }
    if constexpr (kM346) {
      const int rb = 8 * c.bits;
      if (tid < kWideN && c.n0 + tid < c.n && (pg * rb) / 128 != ((pg - 1) * rb) / 128)
        prefetch_l2(c.weights + static_cast<size_t>(c.n0 + tid) * c.row_bytes + pg * rb);
    } else {
#pragma unroll
      for (int h = 0; h < kBCopies; ++h)
        if (bcol[h] == 0 && bvalid[h] && ((pg * (kWideK / 2)) % 128) == 0) prefetch_l2(brow_ptr[h] + pg * (kWideK / 2));
    }
  };
  auto issue = [&](int group, int slot, float scale_value) {
    uint16_t* as = c.a + static_cast<size_t>(slot) * kWideM * kAStride;
#pragma unroll
    for (int h = 0; h < kACopies; ++h) {
      const int row = arow0 + h * kARows;
      const uint16_t* src = c.act + static_cast<size_t>(asrc[h]) * c.act_stride + group * kWideK + acol;
      if (c.vector_act) {
        copy16(as + wide_a_off<kM346>(row, acol), src, avalid[h]);
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) as[wide_a_off<kM346>(row, acol) + j] = avalid[h] ? src[j] : 0;
      }
    }
    uint8_t* bs = c.bc + static_cast<size_t>(slot) * kWideN * kCodeStride;
    if constexpr (kM346) {
      // A step's code row is 8 x bits bytes: 16-byte pieces at 4 and 6 bits
      // (two and three a row), eight-byte pieces at 3 bits (three a row).
      const int rb = 8 * c.bits;
      const int pb = c.bits == 3 ? 8 : 16;
      const int pieces = rb / pb;
      for (int idx = tid; idx < kWideN * pieces; idx += kThreads) {
        const int row = idx / pieces, piece = idx - row * pieces;
        const bool valid = c.n0 + row < c.n;
        const uint8_t* src = c.weights + static_cast<size_t>(valid ? c.n0 + row : 0) * c.row_bytes + group * rb + piece * pb;
        if (pb == 16) copy16(bs + row * kCodeStride + piece * 16, src, valid);
        else copy8(bs + row * kCodeStride + piece * 8, src, valid);
      }
    } else {
#pragma unroll
      for (int h = 0; h < kBCopies; ++h)
        copy16(bs + brow[h] * kWideCodeStride + bcol[h], brow_ptr[h] + group * (kWideK / 2), bvalid[h]);
    }
    if (tid < kWideN) c.scale[slot * kWideN + tid] = scale_value;
    // The L2 prefetch of a later step's lines (c.prefetch_ahead steps past
    // this one): the two-stage pipeline hides one step of latency, and a
    // step's activation lines (one 128-byte line a row) and code lines
    // (one line per four steps of a row) come from DRAM in the real chunk.
    if (c.prefetch_ahead > 0 && group + c.prefetch_ahead < c.groups) prefetch_step(group + c.prefetch_ahead);
  };

  float acc[kMI][kNJ][4] = {};
  float next_scale = scale_of(0);
  // The first steps' lines (the loop prefetches from step prefetch_ahead
  // on; a 10-step down projection would otherwise expose three of them).
  for (int pg = 0; pg < c.prefetch_ahead && pg < c.groups; ++pg) prefetch_step(pg);
  const int prologue = c.groups < kStages - 1 ? c.groups : kStages - 1;
  for (int g = 0; g < prologue; ++g) {
    issue(g, g, next_scale);
    commit();
    next_scale = g + 1 < c.groups ? scale_of(g + 1) : 0.f;
  }
  for (int group = 0; group < c.groups; ++group) {
    const int slot = group % kStages;
    if (group + kStages - 2 < c.groups) wait_pending(kStages - 2);
    else wait_pending(0);
    __syncthreads();  // stage `group` visible; every warp is done with the previous step's tiles
    if (group + kStages - 1 < c.groups) {
      const int g = group + kStages - 1;
      issue(g, g % kStages, next_scale);
      commit();
      next_scale = g + 1 < c.groups ? scale_of(g + 1) : 0.f;
    }
    // Decode this step's codes once: thread -> (row n, half) pieces of 16 bytes
    // (32 codes of 4 x bits bytes under the Mixed346 form).
    if constexpr (kM346 && !kRegB) {
#pragma unroll
      for (int h = 0; h < kBCopies; ++h) {
        const int piece = tid + h * kThreads;
        const int row = piece / 2, half = piece % 2;
        const uint8_t* bs = c.bc + static_cast<size_t>(slot) * kWideN * kCodeStride + row * kCodeStride;
        uint16_t* dst = c.bd + row * kWideDStride + half * 32;
        {
          // An existing expert's int4 g64 row: the offset decode.
          const uint4 codes = *reinterpret_cast<const uint4*>(bs + half * 16);
          const uint32_t w[4] = {codes.x, codes.y, codes.z, codes.w};
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            uint4 o;
            o.x = decode_byte<packq_gemv::kScaleBf16G64>(w[q] & 0xFFu);
            o.y = decode_byte<packq_gemv::kScaleBf16G64>((w[q] >> 8) & 0xFFu);
            o.z = decode_byte<packq_gemv::kScaleBf16G64>((w[q] >> 16) & 0xFFu);
            o.w = decode_byte<packq_gemv::kScaleBf16G64>(w[q] >> 24);
            *reinterpret_cast<uint4*>(dst + q * 8) = o;
          }
        }
      }
      __syncthreads();
    } else if constexpr (!kRegB) {
      const uint8_t* bs = c.bc + static_cast<size_t>(slot) * kWideN * kWideCodeStride;
#pragma unroll
      for (int h = 0; h < kBCopies; ++h) {
        const int piece = tid + h * kThreads;
        const int row = piece / 2, half = piece % 2;
        const uint4 codes = *reinterpret_cast<const uint4*>(bs + row * kWideCodeStride + half * 16);
        const uint32_t w[4] = {codes.x, codes.y, codes.z, codes.w};
        uint16_t* dst = c.bd + row * kWideDStride + half * 32;
        [[maybe_unused]] const float fs = kFold ? c.scale[slot * kWideN + row] : 1.f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          uint4 o;
          if constexpr (packq_gemv::ScaleFmt<SF>::codebook) {
            o = decode_word_codebook(w[q]);
          } else {
            o.x = decode_byte<SF>(w[q] & 0xFFu);
            o.y = decode_byte<SF>((w[q] >> 8) & 0xFFu);
            o.z = decode_byte<SF>((w[q] >> 16) & 0xFFu);
            o.w = decode_byte<SF>(w[q] >> 24);
          }
          if constexpr (kFold) {
            o.x = fold_scale(o.x, fs);
            o.y = fold_scale(o.y, fs);
            o.z = fold_scale(o.z, fs);
            o.w = fold_scale(o.w, fs);
          }
          *reinterpret_cast<uint4*>(dst + q * 8) = o;
        }
      }
      __syncthreads();
    }
    const uint16_t* as = c.a + static_cast<size_t>(slot) * kWideM * kAStride;
    const uint8_t* bs_codes = c.bc + static_cast<size_t>(slot) * kWideN * kWideCodeStride;
    float partial_storage[kMI][kNJ][4] = {};
    // kFold: the MMAs accumulate straight into acc (the scale is in the values).
    auto& partial = kFold ? acc : partial_storage;
#pragma unroll
    for (int kk = 0; kk < kWideK; kk += 16) {
      uint32_t af[kMI][4];
#pragma unroll
      for (int i = 0; i < kMI; ++i)
        ldsm_x4(af[i], as + wide_a_off<kM346>(row_base + i * 16 + lane % 16, kk + (lane / 16) * 8));
      uint32_t bf[kNJ][2];
      if constexpr (kRegB) {
        // The B fragment from the codes: row n = lane / 4 of the n8 tile, k
        // pairs (2q, 2q+1) and (2q+8, 2q+9) for q = lane % 4 — bytes q and
        // q + 4 of the row's 8-byte k16 chunk; decode_byte puts the even k
        // in the low half, as the decoded tile's ldmatrix did.
        const int q = lane & 3;
#pragma unroll
        for (int j = 0; j < kNJ; ++j) {
          const int n = col_base + j * 8 + (lane >> 2);
          const uint2 cw = *reinterpret_cast<const uint2*>(bs_codes + n * kWideCodeStride + (kk >> 1));
          bf[j][0] = decode_byte<SF>((cw.x >> (8 * q)) & 0xFFu);
          bf[j][1] = decode_byte<SF>((cw.y >> (8 * q)) & 0xFFu);
        }
      } else {
#pragma unroll
        for (int jj = 0; jj < kNJ / 2; ++jj) {
          // Two n8 tiles per x4: matrices 0/1 = tile 2jj at k 0..7 / 8..15, 2/3 = tile 2jj+1.
          const int tile = col_base + jj * 16 + (lane / 16) * 8 + lane % 8;
          const int kpart = ((lane / 8) % 2) * 8;
          uint32_t q4[4];
          ldsm_x4(q4, c.bd + tile * kWideDStride + kk + kpart);
          bf[jj * 2][0] = q4[0];
          bf[jj * 2][1] = q4[1];
          bf[jj * 2 + 1][0] = q4[2];
          bf[jj * 2 + 1][1] = q4[3];
        }
      }
#pragma unroll
      for (int i = 0; i < kMI; ++i)
#pragma unroll
        for (int j = 0; j < kNJ; ++j)
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(partial[i][j][0]), "+f"(partial[i][j][1]), "+f"(partial[i][j][2]),
                "+f"(partial[i][j][3])
              : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(bf[j][0]), "r"(bf[j][1]));
    }
    if constexpr (!kFold) {
#pragma unroll
      for (int i = 0; i < kMI; ++i)
#pragma unroll
        for (int j = 0; j < kNJ; ++j) {
          const int col = col_base + j * 8 + cc;
#pragma unroll
          for (int v = 0; v < 4; ++v)
            acc[i][j][v] = __fmaf_rn(c.scale[slot * kWideN + col + (v % 2)], partial[i][j][v], acc[i][j][v]);
        }
    }
  }
#pragma unroll
  for (int i = 0; i < kMI; ++i)
#pragma unroll
    for (int j = 0; j < kNJ; ++j)
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const int row = c.m0 + row_base + i * 16 + r + (v / 2) * 8;
        const int col = c.n0 + col_base + j * 8 + cc + (v % 2);
        if (row < c.m && col < c.n) {
          const size_t index = static_cast<size_t>(c.row0 + row) * c.out_stride + col;
          if constexpr (std::is_same_v<OutT, float>)
            c.out[index] = acc[i][j][v];
          else
            c.out[index] = __bfloat16_as_ushort(__float2bfloat16_rn(acc[i][j][v]));
        }
      }
}

// ---- the Mixed346 int8 tensor-core tile (2026-10-10) ------------------------
// A converted expert's tile under the Mixed346 form runs on int8 tensor
// cores: the rows' int8 codes (cp.async, no conversion) against the weight
// rows decoded to int8 levels, mma.sync m16n8k32 s8 x s8 -> s32. A 128-code
// group's dot is the exact int32 sum of its two steps' four k32 MMAs,
// scaled once by the weight scale x the row's activation scale and added
// to the fp32 accumulator — the GEMV core's arithmetic (kernels/
// packq_a8_gemv.cuh); the paths differ by the fp32 accumulation order
// across groups only. Per step a thread stages one 16-byte chunk of the
// rows, decodes one 32-code piece of the weight rows to levels (two 16-byte
// stores) and issues two k32 MMA steps per (m16, n8) tile; the bf16 form
// (wide_tile, the existing experts' tiles) converts the rows on the issue
// path, decodes to bf16 and runs four k16 steps — 45 % more instructions a
// step, and the conversion's fixed cost dominated the small remainder tiles
// of a real routing (the chain ran 20 % slower than the int4 one while its
// 64-row tiles ran 15 % faster). Both operands sit at an 80-byte row stride
// (ldmatrix conflict-free, the e4m3 tile's layout).
constexpr int kI8Stride = kWideK + 16;
__host__ __device__ constexpr size_t wide_i8_smem_bytes(int stages) {
  return static_cast<size_t>(stages) * kWideM * kI8Stride + static_cast<size_t>(stages) * kWideN * kWideCodeStrideM346 +
         static_cast<size_t>(kWideN) * kI8Stride + static_cast<size_t>(stages) * kWideN * 4 +
         static_cast<size_t>(stages) * kWideM * 4;
}
static_assert(wide_i8_smem_bytes(kWideStages) <= wide_smem_bytes(kWideStages, false, packq_gemv::kScaleBf16G128Mixed346),
              "the int8 tile fits the Mixed346 launch's shared memory");

template <typename OutT, int M16, int kStages, int kWarps>
__device__ __forceinline__ void wide_tile_i8(const WideCtx<OutT>& c, uint8_t* __restrict__ smem) {
  static_assert(kWarps == 8 || kWarps == 4, "eight or four warps");
  constexpr int kThreads = kWarps * 32;
  constexpr int kMI = kWarps == 8 ? (M16 == 4 ? 2 : 1) : M16;
  constexpr int kNJ = kWarps == 8 ? (M16 == 1 ? 2 : 4) : 4;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int r = lane / 4, cc = (lane % 4) * 2;
  const int row_base = kWarps == 8 ? (M16 == 4 ? (warp / 4) * 32 : (M16 == 2 ? (warp / 4) * 16 : 0)) : 0;
  const int col_base = kWarps == 8 ? (M16 == 1 ? warp * 16 : (warp % 4) * 32) : warp * 32;
  uint8_t* sa = smem;                                                        // [S][64][80] int8 rows
  uint8_t* sb = sa + static_cast<size_t>(kStages) * kWideM * kI8Stride;      // [S][128][48] code rows
  uint8_t* sd = sb + static_cast<size_t>(kStages) * kWideN * kWideCodeStrideM346;  // [128][80] int8 levels
  float* sscale = reinterpret_cast<float*>(sd + static_cast<size_t>(kWideN) * kI8Stride);  // [S][128]
  float* sascale = sscale + static_cast<size_t>(kStages) * kWideN;           // [S][64]
  // The rows' chunks: 64 rows x four 16-byte chunks a step.
  constexpr int kAChunks = kWideM * 4 / kThreads;
  int asrc[kAChunks], arow[kAChunks], achunk[kAChunks];
  bool avalid[kAChunks];
#pragma unroll
  for (int h = 0; h < kAChunks; ++h) {
    const int idx = tid + h * kThreads;
    arow[h] = idx / 4;
    achunk[h] = (idx % 4) * 16;
    avalid[h] = c.m0 + arow[h] < c.m;
    const int gathered = c.row0 + c.m0 + arow[h];
    asrc[h] = avalid[h] ? (c.act_rows ? c.act_rows[gathered] : gathered) : 0;
  }
  const bool scale_lane = tid < kWideN && c.n0 + tid < c.n;
  const uint16_t* scale_ptr = c.scales + static_cast<size_t>(scale_lane ? c.n0 + tid : 0) * c.scale_groups;
  auto scale_of = [&](int group) -> float { return scale_lane ? bf16_bits_to_float(scale_ptr[group / 2]) : 0.f; };
  const bool arow_lane = tid < kWideM && c.m0 + tid < c.m;
  const int arow_src = arow_lane ? (c.act_rows ? c.act_rows[c.row0 + c.m0 + tid] : c.row0 + c.m0 + tid) : 0;
  const int rb = 8 * c.bits;                 // a step's bytes of a code row: 24 / 32 / 48
  const int pb = c.bits == 3 ? 8 : 16;       // the staged piece
  const int pieces = rb / pb;
  // The L2 prefetch of step pg's lines: a row's code line every two steps
  // (64 bytes a step), a weight row's line when the step starts a new one.
  auto prefetch_step = [&](int pg) {
#pragma unroll
    for (int h = 0; h < kAChunks; ++h)
      if (avalid[h] && achunk[h] == 0 && (pg * kWideK) % 128 == 0)
        prefetch_l2(c.act_q + static_cast<size_t>(asrc[h]) * c.act_q_stride + pg * kWideK);
    if (tid < kWideN && c.n0 + tid < c.n && (pg * rb) / 128 != ((pg - 1) * rb) / 128)
      prefetch_l2(c.weights + static_cast<size_t>(c.n0 + tid) * c.row_bytes + pg * rb);
  };
  auto issue = [&](int group, int slot, float scale_value) {
    uint8_t* as = sa + static_cast<size_t>(slot) * kWideM * kI8Stride;
#pragma unroll
    for (int h = 0; h < kAChunks; ++h)
      copy16(as + arow[h] * kI8Stride + achunk[h],
             c.act_q + static_cast<size_t>(asrc[h]) * c.act_q_stride + group * kWideK + achunk[h], avalid[h]);
    uint8_t* bs = sb + static_cast<size_t>(slot) * kWideN * kWideCodeStrideM346;
    for (int idx = tid; idx < kWideN * pieces; idx += kThreads) {
      const int row = idx / pieces, piece = idx - row * pieces;
      const bool valid = c.n0 + row < c.n;
      const uint8_t* src = c.weights + static_cast<size_t>(valid ? c.n0 + row : 0) * c.row_bytes + group * rb + piece * pb;
      if (pb == 16) copy16(bs + row * kWideCodeStrideM346 + piece * 16, src, valid);
      else copy8(bs + row * kWideCodeStrideM346 + piece * 8, src, valid);
    }
    if (tid < kWideM)
      copy4(sascale + slot * kWideM + tid, c.act_s + static_cast<size_t>(arow_src) * c.act_s_stride + group / 2, arow_lane);
    if (tid < kWideN) sscale[slot * kWideN + tid] = scale_value;
    if (c.prefetch_ahead > 0 && group + c.prefetch_ahead < c.groups) prefetch_step(group + c.prefetch_ahead);
  };

  float acc[kMI][kNJ][4] = {};
  int32_t part[kMI][kNJ][4] = {};
  float next_scale = scale_of(0);
  for (int pg = 0; pg < c.prefetch_ahead && pg < c.groups; ++pg) prefetch_step(pg);
  const int prologue = c.groups < kStages - 1 ? c.groups : kStages - 1;
  for (int g = 0; g < prologue; ++g) {
    issue(g, g, next_scale);
    commit();
    next_scale = g + 1 < c.groups ? scale_of(g + 1) : 0.f;
  }
  constexpr int kBCopies = (kWideN * 2) / kThreads;  // 32-code pieces a thread decodes
  for (int group = 0; group < c.groups; ++group) {
    const int slot = group % kStages;
    if (group + kStages - 2 < c.groups) wait_pending(kStages - 2);
    else wait_pending(0);
    __syncthreads();
    if (group + kStages - 1 < c.groups) {
      const int g = group + kStages - 1;
      issue(g, g % kStages, next_scale);
      commit();
      next_scale = g + 1 < c.groups ? scale_of(g + 1) : 0.f;
    }
    // This step's codes to int8 levels: thread -> (row n, half) pieces of 32 codes.
#pragma unroll
    for (int h = 0; h < kBCopies; ++h) {
      const int piece = tid + h * kThreads;
      const int row = piece / 2, half = piece % 2;
      const uint8_t* bs = sb + static_cast<size_t>(slot) * kWideN * kWideCodeStrideM346 + row * kWideCodeStrideM346;
      uint32_t lv[8];
      if (c.bits == 4) {
        const uint4 w = *reinterpret_cast<const uint4*>(bs + half * 16);
        const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          lv[2 * q] = packq_gemv::expand_nf4_int8(ws[q] & 0xFFFFu);
          lv[2 * q + 1] = packq_gemv::expand_nf4_int8(ws[q] >> 16);
        }
      } else if (c.bits == 3) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(bs + half * 12);
        const uint32_t ws[4] = {w[0], w[1], w[2], 0u};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int bit = 12 * j;
          lv[j] = packq_a8::gauss3_levels(__funnelshift_r(ws[bit / 32], ws[bit / 32 + 1], bit % 32));
        }
      } else {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(bs + half * 24);
        const uint32_t ws[7] = {w[0], w[1], w[2], w[3], w[4], w[5], 0u};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int bit = 24 * j;
          lv[j] = __vsub4(packq_a8::int6_bytes(__funnelshift_r(ws[bit / 32], ws[bit / 32 + 1], bit % 32)), 0x20202020u);
        }
      }
      uint8_t* dst = sd + row * kI8Stride + half * 32;
      *reinterpret_cast<uint4*>(dst) = make_uint4(lv[0], lv[1], lv[2], lv[3]);
      *reinterpret_cast<uint4*>(dst + 16) = make_uint4(lv[4], lv[5], lv[6], lv[7]);
    }
    __syncthreads();
    const uint8_t* as = sa + static_cast<size_t>(slot) * kWideM * kI8Stride;
#pragma unroll
    for (int kk = 0; kk < kWideK; kk += 32) {
      uint32_t af[kMI][4];
#pragma unroll
      for (int i = 0; i < kMI; ++i)
        ldsm_x4(af[i], as + (row_base + i * 16 + lane % 16) * kI8Stride + kk + (lane / 16) * 16);
      uint32_t bf[kNJ][2];
#pragma unroll
      for (int jj = 0; jj < kNJ / 2; ++jj) {
        // Matrices 0/1 = n8 tile 2jj at k kk..kk+15 / kk+16..kk+31, 2/3 = tile 2jj+1.
        const int tile = col_base + jj * 16 + (lane / 16) * 8 + lane % 8;
        const int kb = kk + ((lane / 8) % 2) * 16;
        uint32_t q4[4];
        ldsm_x4(q4, sd + tile * kI8Stride + kb);
        bf[jj * 2][0] = q4[0];
        bf[jj * 2][1] = q4[1];
        bf[jj * 2 + 1][0] = q4[2];
        bf[jj * 2 + 1][1] = q4[3];
      }
#pragma unroll
      for (int i = 0; i < kMI; ++i)
#pragma unroll
        for (int j = 0; j < kNJ; ++j)
          asm volatile(
              "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+r"(part[i][j][0]), "+r"(part[i][j][1]), "+r"(part[i][j][2]), "+r"(part[i][j][3])
              : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(bf[j][0]), "r"(bf[j][1]));
    }
    const bool close = (group & 1) || group + 1 == c.groups;
    if (close) {
#pragma unroll
      for (int i = 0; i < kMI; ++i)
#pragma unroll
        for (int j = 0; j < kNJ; ++j) {
          const int col = col_base + j * 8 + cc;
#pragma unroll
          for (int v = 0; v < 4; ++v) {
            const int row = row_base + i * 16 + r + (v / 2) * 8;
            const float sw = sscale[slot * kWideN + col + (v % 2)] * sascale[slot * kWideM + row];
            acc[i][j][v] = __fmaf_rn(sw, static_cast<float>(part[i][j][v]), acc[i][j][v]);
            part[i][j][v] = 0;
          }
        }
    }
  }
#pragma unroll
  for (int i = 0; i < kMI; ++i)
#pragma unroll
    for (int j = 0; j < kNJ; ++j)
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const int row = c.m0 + row_base + i * 16 + r + (v / 2) * 8;
        const int col = c.n0 + col_base + j * 8 + cc + (v % 2);
        if (row < c.m && col < c.n) {
          const size_t index = static_cast<size_t>(c.row0 + row) * c.out_stride + col;
          if constexpr (std::is_same_v<OutT, float>)
            c.out[index] = acc[i][j][v];
          else
            c.out[index] = __bfloat16_as_ushort(__float2bfloat16_rn(acc[i][j][v]));
        }
      }
}

template <typename OutT, bool Grouped, int SF, int kStages, bool kRegB, bool kFold, int kWarps>
__global__ __launch_bounds__(kWarps * 32, kWideMinBlocks) void packq_gemm_wide_kernel(
    const uint16_t* __restrict__ act, size_t act_stride, const uint8_t* __restrict__ weights,
    const uint16_t* __restrict__ scales, const MoeSegment* __restrict__ segs,
    const MoeExpertView* __restrict__ views, int which, const int32_t* __restrict__ act_rows,
    OutT* __restrict__ out, size_t out_stride, int m, int n, int k, int n_tiles, bool vector_act,
    const MoeTile* __restrict__ tiles, const int32_t* __restrict__ tile_count, int prefetch_ahead,
    OutT* __restrict__ out2, int which2, const int8_t* __restrict__ act_q, size_t act_q_stride,
    const float* __restrict__ act_s, size_t act_s_stride, int bits) {
  dgpp_pdl_wait();
  extern __shared__ __align__(16) uint8_t wide_smem[];
  constexpr bool kM346 = SF == packq_gemv::kScaleBf16G128Mixed346;
  constexpr int kCodeStride = wide_code_stride(SF);
  WideCtx<OutT> c;
  c.prefetch_ahead = prefetch_ahead;
  c.a = reinterpret_cast<uint16_t*>(wide_smem);
  c.bc = wide_smem + static_cast<size_t>(kStages) * kWideM * wide_a_stride(SF) * 2;
  c.bd = reinterpret_cast<uint16_t*>(c.bc + static_cast<size_t>(kStages) * kWideN * kCodeStride);
  c.scale = reinterpret_cast<float*>(c.bd + (kRegB ? size_t{0} : static_cast<size_t>(kWideN) * kWideDStride));
  c.ascale = kM346 ? c.scale + static_cast<size_t>(kStages) * kWideN : nullptr;
  c.act_q = act_q;
  c.act_q_stride = act_q_stride;
  c.act_s = act_s;
  c.act_s_stride = act_s_stride;
  c.bits = bits;
  c.a8 = false;
  c.row0 = 0;
  c.m0 = (blockIdx.x / n_tiles) * kWideM;
  if constexpr (Grouped) {
    // The compact tile list (glm_moe_launch.hpp): a 1-D grid over the real
    // tiles, block b at tile b / n_tiles; without it, blockIdx.y is the
    // segment and the grid spans the longest segment's m-tiles.
    int seg_index = blockIdx.y;
    if (tiles) {
      const int t = blockIdx.x / n_tiles;
      if (t >= *tile_count) return;
      const MoeTile tile = tiles[t];
      seg_index = tile.seg;
      c.m0 = tile.m0;
    }
    const MoeSegment seg = segs[seg_index];
    c.row0 = seg.row0;
    m = seg.rows;
    // The paired launch (2026-09-30, gate and up in one): the n-tiles past
    // the first projection's width take the second projection's view and
    // output — every output element's chain is its own launch's.
    int tile_n = blockIdx.x % n_tiles;
    if (out2 != nullptr) {
      const int per = n_tiles / 2;  // the first projection's tiles, then the second's
      if (tile_n >= per) {
        tile_n -= per;
        which = which2;
        out = out2;
      }
    }
    c.n0 = tile_n * kWideN;
    const MoeExpertView view = views[seg.expert * 3 + which];
    weights = view.payload;
    scales = view.packed_scales;
    if constexpr (kM346) {
      // The view names the tile's form: a converted triple at its width,
      // or an existing expert's int4 g64 one.
      c.a8 = view.packed_scale_fmt == kPackedScaleBf16G128Mixed346;
      c.bits = c.a8 ? view.bits : 4;
    }
  } else {
    c.n0 = (blockIdx.x % n_tiles) * kWideN;
    if constexpr (kM346) c.a8 = true;
  }
  if (c.m0 >= m) return;
  c.act = act;
  c.act_stride = act_stride;
  c.weights = weights;
  c.scales = scales;
  c.act_rows = act_rows;
  c.out = out;
  c.out_stride = out_stride;
  c.m = m;
  c.n = n;
  c.k = k;
  c.groups = k / kWideK;
  c.scale_groups = kM346 ? k / (c.a8 ? 128 : 64) : k / packq_gemv::ScaleFmt<SF>::group;
  c.row_bytes = kM346 ? k * c.bits / 8 : k / 2;
  c.vector_act = vector_act;
  const int rows_here = m - c.m0 < kWideM ? m - c.m0 : kWideM;
  if constexpr (kM346) {
    if (c.a8) {
      // A converted expert's tile: the int8 tensor-core form.
      if (rows_here <= 16) wide_tile_i8<OutT, 1, kStages, kWarps>(c, wide_smem);
      else if (rows_here <= 32) wide_tile_i8<OutT, 2, kStages, kWarps>(c, wide_smem);
      else wide_tile_i8<OutT, 4, kStages, kWarps>(c, wide_smem);
      return;
    }
  }
  if (rows_here <= 16) wide_tile<OutT, SF, 1, kStages, kRegB, kFold, kWarps>(c);
  else if (rows_here <= 32) wide_tile<OutT, SF, 2, kStages, kRegB, kFold, kWarps>(c);
  else wide_tile<OutT, SF, 4, kStages, kRegB, kFold, kWarps>(c);
}


template <typename OutT, bool Grouped, int SF, int kStages, bool kRegB, bool kFold, int kWarps>
void launch_wide(const uint16_t* act, size_t act_stride, const uint8_t* weights, const uint16_t* scales,
                 const MoeSegment* segs, int n_segs, const MoeExpertView* views, int which,
                 const int32_t* act_rows, OutT* out, size_t out_stride, int m, int n, int k,
                 cudaStream_t stream, const MoeTile* tiles, const int32_t* tile_count, int tile_cap,
                 OutT* out2, int which2, const int8_t* act_q = nullptr, size_t act_q_stride = 0,
                 const float* act_s = nullptr, size_t act_s_stride = 0, int bits = 4) {
  constexpr size_t kSmem = wide_smem_bytes(kStages, kRegB, SF);
  static bool opted = false;
  if (!opted) {
    DGPP_CUDA_OK(cudaFuncSetAttribute(packq_gemm_wide_kernel<OutT, Grouped, SF, kStages, kRegB, kFold, kWarps>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmem)));
    opted = true;
  }
  const int n_tiles = ((n + kWideN - 1) / kWideN) * (Grouped && out2 != nullptr ? 2 : 1);
  const bool listed = Grouped && tiles != nullptr;
  const dim3 grid(n_tiles * (listed ? tile_cap : (m + kWideM - 1) / kWideM), listed ? 1 : n_segs);
  const bool vector_act = reinterpret_cast<uintptr_t>(act) % 16 == 0 && act_stride % 8 == 0;
  packq_gemm_wide_kernel<OutT, Grouped, SF, kStages, kRegB, kFold, kWarps><<<grid, kWarps * 32, kSmem, stream>>>(
      act, act_stride, weights, scales, segs, views, which, act_rows, out, out_stride, m, n, k, n_tiles,
      vector_act, listed ? tiles : nullptr, listed ? tile_count : nullptr, packq_gemm_prefetch_ahead(),
      Grouped ? out2 : nullptr, which2, act_q, act_q_stride, act_s, act_s_stride, bits);
  DGPP_CUDA_OK(cudaGetLastError());
}


void check_shape(const uint16_t* act, size_t act_stride, const void* out, size_t out_stride, int n,
                 int k, int bits, int scale_fmt) {
  if (!act || !out) throw std::invalid_argument("packq_gemm: null activation/output pointer");
  if (!packq_gemv::scale_fmt_known(scale_fmt) && scale_fmt != packq_gemv::kScaleBf16G128Mixed346)
    throw std::invalid_argument("packq_gemm: unknown packed scale format");
  const bool m346 = scale_fmt == packq_gemv::kScaleBf16G128Mixed346;
  // The Mixed346 launch takes its widths from the views (bits 0), or one
  // width for a single-form launch.
  const bool bits_ok = m346 ? (bits == 0 || bits == 3 || bits == 4 || bits == 6) : (bits == 4 || bits == 8);
  if (k <= 0 || k % packq_gemv::scale_group_of(scale_fmt) || !bits_ok)
    throw std::invalid_argument(
        "packq_gemm: K must be a positive multiple of the scale group and bits 4 or 8 (3 / 4 / 6 under Mixed346)");
  if (scale_fmt == packq_gemv::kScaleBf16G128Nf4i8 && bits != 4)
    throw std::invalid_argument("packq_gemm: the NF4I8 codebook format is 4-bit");
  if (act_stride < static_cast<size_t>(k) || out_stride < static_cast<size_t>(n))
    throw std::invalid_argument("packq_gemm: row stride is smaller than the matrix width");
}

template <typename OutT, bool Grouped>
void launch(const uint16_t* act, size_t act_stride, const uint8_t* weights, const uint16_t* scales,
            const MoeSegment* segs, int n_segs, const MoeExpertView* views, int which,
            const int32_t* act_rows, OutT* out, size_t out_stride, int m, int n, int k, int bits,
            int scale_fmt, cudaStream_t stream, bool planes = false, int variant = -1,
            const MoeTile* tiles = nullptr, const int32_t* tile_count = nullptr, int tile_cap = 0,
            OutT* out2 = nullptr, int which2 = -1, const int8_t* act_q = nullptr, size_t act_q_stride = 0,
            const float* act_s = nullptr, size_t act_s_stride = 0) {
  check_shape(act, act_stride, out, out_stride, n, k, bits, scale_fmt);
  if (variant < 0) variant = packq_gemm_variant_default();
  if (scale_fmt == packq_gemv::kScaleBf16G128Mixed346) {
    // The Mixed346 form (2026-10-09): the wide decoded-tile kernels (the
    // default and the four-warp form) over the views' widths and the rows'
    // int8 codes; the register-decode, folded and narrow forms do not take
    // it. A grouped launch only (the views name each tile's form).
    if (!Grouped || !act_q || !act_s || act_q_stride % 16 != 0 || act_q_stride < static_cast<size_t>(k) ||
        act_s_stride < static_cast<size_t>(k / 128) || reinterpret_cast<uintptr_t>(act_q) % 16 != 0)
      throw std::invalid_argument("packq_gemm: the Mixed346 form needs a grouped launch with 16-byte aligned int8 code rows and their scales");
    if (out2 != nullptr) throw std::invalid_argument("packq_gemm: the paired launch is the int4 kernel's");
    if (variant == 4)
      launch_wide<OutT, Grouped, packq_gemv::kScaleBf16G128Mixed346, kWideStages, false, false, 4>(
          act, act_stride, weights, scales, segs, n_segs, views, which, act_rows, out, out_stride, m, n, k, stream,
          tiles, tile_count, tile_cap, out2, which2, act_q, act_q_stride, act_s, act_s_stride, bits);
    else
      launch_wide<OutT, Grouped, packq_gemv::kScaleBf16G128Mixed346, kWideStages, false, false, 8>(
          act, act_stride, weights, scales, segs, n_segs, views, which, act_rows, out, out_stride, m, n, k, stream,
          tiles, tile_count, tile_cap, out2, which2, act_q, act_q_stride, act_s, act_s_stride, bits);
    return;
  }
  // variant 3 (engine.prefill_fold_scales; NOT bitwise): the decoded-tile
  // kernel with the group scales folded into the bf16 values.
  // variants 4 / 5 (2026-09-30, round 21): the four-warp forms of 1 / 2.
  if (variant >= 1 && variant <= 5 && bits == 4 && !planes) {
#define DGPP_WIDE(SF_, ST_, RB_, FD_, WP_)                                                                     \
  launch_wide<OutT, Grouped, SF_, ST_, RB_, FD_, WP_>(act, act_stride, weights, scales, segs, n_segs, views,   \
                                                       which, act_rows, out, out_stride, m, n, k, stream,      \
                                                       tiles, tile_count, tile_cap, out2, which2)
#define DGPP_WIDE_SF(SF_)                                                    \
  do {                                                                       \
    if (variant == 2) DGPP_WIDE(SF_, kWideStagesReg, true, false, 8);        \
    else if (variant == 3) DGPP_WIDE(SF_, kWideStages, false, true, 8);      \
    else if (variant == 4) DGPP_WIDE(SF_, kWideStages, false, false, 4);     \
    else if (variant == 5) DGPP_WIDE(SF_, kWideStagesReg, true, false, 4);   \
    else DGPP_WIDE(SF_, kWideStages, false, false, 8);                       \
  } while (0)
    if (scale_fmt == packq_gemv::kScaleF16G128) DGPP_WIDE_SF(packq_gemv::kScaleF16G128);
    else if (scale_fmt == packq_gemv::kScaleBf16G128Nf4i8) DGPP_WIDE_SF(packq_gemv::kScaleBf16G128Nf4i8);
    else DGPP_WIDE_SF(packq_gemv::kScaleBf16G64);
#undef DGPP_WIDE_SF
#undef DGPP_WIDE
    return;
  }
  // The narrow kernel (and int8 rows) keep the segment-major grid: m is the
  // longest segment's rows, or the caller's bound when it built a tile list.
  (void)tiles;
  (void)tile_count;
  (void)tile_cap;
  if (out2 != nullptr) throw std::invalid_argument("packq_gemm: the paired launch is the wide int4 kernel's");
  const int n_tiles = (n + kN - 1) / kN;
  const dim3 grid(n_tiles * ((m + kM - 1) / kM), n_segs);
  const bool vector_act = reinterpret_cast<uintptr_t>(act) % 16 == 0 && act_stride % 8 == 0;
#define DGPP_PACKQ_LAUNCH(Bits, SF)                                                         \
  packq_gemm_kernel<Bits, OutT, Grouped, SF, false>                                         \
      <<<grid, kThreads, 0, stream>>>(act, act_stride, weights, scales, segs, views, which, \
                                      act_rows, out, out_stride, m, n, k, n_tiles, vector_act)
  if (planes) {
    // The int8 head's plane layout (dense only): the 128-code k-step.
    if (bits != 8 || scale_fmt != packq_gemv::kScaleF16G128 || Grouped || k % 128 != 0)
      throw std::invalid_argument("packq_gemm: the plane layout is the dense int8 g128 head's");
    packq_gemm_kernel<8, OutT, false, packq_gemv::kScaleF16G128, true>
        <<<grid, kThreads, 0, stream>>>(act, act_stride, weights, scales, segs, views, which, act_rows, out,
                                        out_stride, m, n, k, n_tiles, vector_act);
  } else if (scale_fmt == packq_gemv::kScaleF16G128) {
    if (bits == 4) {
      DGPP_PACKQ_LAUNCH(4, packq_gemv::kScaleF16G128);
    } else {
      DGPP_PACKQ_LAUNCH(8, packq_gemv::kScaleF16G128);
    }
  } else if (scale_fmt == packq_gemv::kScaleBf16G128Nf4i8) {
    // The codebook format is 4-bit (check_shape).
    DGPP_PACKQ_LAUNCH(4, packq_gemv::kScaleBf16G128Nf4i8);
  } else if (bits == 4) {
    DGPP_PACKQ_LAUNCH(4, packq_gemv::kScaleBf16G64);
  } else {
    DGPP_PACKQ_LAUNCH(8, packq_gemv::kScaleBf16G64);
  }
#undef DGPP_PACKQ_LAUNCH
  DGPP_CUDA_OK(cudaGetLastError());
}

template <typename OutT>
void dense(const uint16_t* act, size_t act_stride, const GlmPackedMatrix& w, OutT* out, int m,
           int n, int k, cudaStream_t stream, int variant = -1) {
  if (m <= 0 || n <= 0) return;
  if (!w.packed || !w.scales || reinterpret_cast<uintptr_t>(w.packed) % 16 || w.rows < n ||
      w.cols != k)
    throw std::invalid_argument(
        "packq_gemm: invalid packed matrix pointers, alignment or N/K geometry");
  if (w.scale_fmt == kPackedScaleBf16G128Mixed346)
    throw std::invalid_argument("packq_gemm: the Mixed346 form runs the grouped launchers (one segment for a single matrix)");
  launch<OutT, false>(act, act_stride, reinterpret_cast<const uint8_t*>(w.packed), w.scales,
                      nullptr, 1, nullptr, 0, nullptr, out, n, m, n, k, w.bits, w.scale_fmt,
                      stream, w.layout == kPackedLayoutPlanes8, variant);
}
template <typename OutT>
void grouped(const uint16_t* act, size_t act_stride, const MoeSegment* segs, int n_segs,
             int max_rows, const MoeExpertView* views, int which, OutT* out, size_t out_stride,
             int n, int k, int bits, cudaStream_t stream, const int32_t* act_rows, int scale_fmt,
             int variant = -1, const MoeTile* tiles = nullptr, const int32_t* tile_count = nullptr,
             int tile_cap = 0, OutT* out2 = nullptr, int which2 = -1, const int8_t* act_q = nullptr,
             size_t act_q_stride = 0, const float* act_s = nullptr, size_t act_s_stride = 0) {
  if (n_segs <= 0 || n <= 0) return;
  if (out2 != nullptr && (which2 < 0 || which2 > 2))
    throw std::invalid_argument("packq_gemm: the paired projection index is out of range");
  if (!segs || !views || max_rows <= 0 || which < 0 || which > 2 || n_segs > 65535)
    throw std::invalid_argument(
        "packq_gemm: invalid segments, max_rows, views or projection index");
  if (tiles && (!tile_count || tile_cap <= 0))
    throw std::invalid_argument("packq_gemm: a tile list needs its count and capacity");
  launch<OutT, true>(act, act_stride, nullptr, nullptr, segs, n_segs, views, which, act_rows, out,
                     out_stride, max_rows, n, k, bits, scale_fmt, stream, false, variant, tiles,
                     tile_count, tile_cap, out2, which2, act_q, act_q_stride, act_s, act_s_stride);
}
}  // namespace

void launch_packq_gemm_bf16(const uint16_t* a, size_t stride, const GlmPackedMatrix& w,
                            uint16_t* out, int m, int n, int k, cudaStream_t stream) {
  dense(a, stride, w, out, m, n, k, stream);
}
void launch_packq_gemm_f32(const uint16_t* a, size_t stride, const GlmPackedMatrix& w, float* out,
                           int m, int n, int k, cudaStream_t stream) {
  dense(a, stride, w, out, m, n, k, stream);
}
void launch_packq_gemm_bf16_variant(const uint16_t* a, size_t stride, const GlmPackedMatrix& w,
                                    uint16_t* out, int m, int n, int k, cudaStream_t stream, int variant) {
  dense(a, stride, w, out, m, n, k, stream, variant);
}
void launch_packq_gemm_f32_variant(const uint16_t* a, size_t stride, const GlmPackedMatrix& w, float* out,
                                   int m, int n, int k, cudaStream_t stream, int variant) {
  dense(a, stride, w, out, m, n, k, stream, variant);
}
void launch_moe_grouped_mma_packq_bf16(const uint16_t* a, size_t stride, const MoeSegment* segs,
                                       int ns, int mr, const MoeExpertView* views, int which,
                                       uint16_t* out, size_t os, int n, int k, int bits,
                                       cudaStream_t stream, const int32_t* rows, int scale_fmt,
                                       int variant, const MoeTile* tiles, const int32_t* tile_count,
                                       int tile_cap, uint16_t* out2, int which2, const int8_t* act_q,
                                       size_t act_q_stride, const float* act_s, size_t act_s_stride) {
  grouped(a, stride, segs, ns, mr, views, which, out, os, n, k, bits, stream, rows, scale_fmt, variant,
          tiles, tile_count, tile_cap, out2, which2, act_q, act_q_stride, act_s, act_s_stride);
}
void launch_moe_grouped_mma_packq_f32(const uint16_t* a, size_t stride, const MoeSegment* segs,
                                      int ns, int mr, const MoeExpertView* views, int which,
                                      float* out, size_t os, int n, int k, int bits,
                                      cudaStream_t stream, const int32_t* rows, int scale_fmt,
                                      int variant, const MoeTile* tiles, const int32_t* tile_count,
                                      int tile_cap, const int8_t* act_q, size_t act_q_stride,
                                      const float* act_s, size_t act_s_stride) {
  grouped(a, stride, segs, ns, mr, views, which, out, os, n, k, bits, stream, rows, scale_fmt, variant,
          tiles, tile_count, tile_cap, static_cast<float*>(nullptr), -1, act_q, act_q_stride, act_s, act_s_stride);
}
}  // namespace dgpp
