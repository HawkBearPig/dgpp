#pragma once
// Compressed resident view of one E4M3 matrix (DESIGN §4): payload + block
// scales, nothing else. Device-visible pointers. CUDA-free so the
// host-only model library and the kernel library share the type.
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "common/dtypes.hpp"

namespace dgpp {

struct GlmQuantMatrix {
  const uint8_t* payload = nullptr;  // E4M3 [rows, cols]
  const float* scales = nullptr;     // F32 [ceil(rows/sbr), ceil(cols/sbc)]
  int64_t rows = 0;
  int64_t cols = 0;
  // The resident scale grid's block size per axis (Q2, 2026-09-09,
  // docs/qwen38_flash_next_plan.md D2): the checkpoint's grid is 128x128;
  // a TP slice that starts mid-block is made exact by re-blocking the
  // sliced axis at gcd(128, slice) with the parent block's scale
  // replicated. 128 on both axes is the GLM contract, untouched.
  int scale_block_rows = 128;
  int scale_block_cols = 128;
  int64_t scale_rows() const { return (rows + scale_block_rows - 1) / scale_block_rows; }
  int64_t scale_cols() const { return (cols + scale_block_cols - 1) / scale_block_cols; }
  size_t scale_bytes() const {
    return static_cast<size_t>(scale_rows()) * static_cast<size_t>(scale_cols()) * sizeof(float);
  }
};

// The scale grid as the kernels' log2 block sizes (rs / cs): true with the
// shifts written, false when an axis is not a power of two (the grid is
// then the loaders' own and no kernel consumes it). scale_block_rows = 1
// with a scale_block_cols >= cols is the CHANNEL grid (one scale a row —
// the Qwen3.8-27B-NVFP4 release's attention half): rs = 0, cs large enough
// that every 16-element chunk reads column 0.
inline bool quant_scale_shifts(const GlmQuantMatrix& m, int& rs, int& cs) {
  auto log2_exact = [](int64_t v) {
    if (v <= 0 || (v & (v - 1)) != 0) return -1;
    int s = 0;
    while ((1LL << s) < v) ++s;
    return s;
  };
  rs = log2_exact(m.scale_block_rows);
  cs = log2_exact(m.scale_block_cols);
  return rs >= 0 && cs >= 4;
}

// Row-range view: payload rows are contiguous, so this is a pure pointer
// view — no copy. A 128-ALIGNED row_start re-anchors the scale grid exactly
// (local row r's true block is row_start/128 + r/128, which is what the
// local-frame consumer computes); a MISALIGNED start that crosses a block
// boundary would need two different scale values inside one local block —
// unrepresentable — and is rejected here, loudly. The real checkpoint's
// inter dims (12288 dense / 2048 MoE) are 128-multiples at every TP world;
// the loader-era fixture's 200 was this contract's counterexample and
// produced silently wrong scales (measured: layer folds 0.30 l2-wrong,
// invisible to the stream-state metric that attenuated it 300x — see the
// M5 record).
inline GlmQuantMatrix quant_rows_view(const GlmQuantMatrix& m,
                                      int64_t row_start, int64_t rows) {
  if (row_start < 0 || rows < 0 || rows > m.rows || row_start > m.rows - rows)
    throw std::invalid_argument("quant_rows_view: range out of bounds");
  if (row_start % m.scale_block_rows != 0)
    throw std::invalid_argument(
        "quant_rows_view: row_start must be aligned to the scale block "
        "(quantized scale-grid slice contract; misaligned slices cannot "
        "re-anchor the block scales)");
  const int64_t sb = m.scale_cols();  // scale blocks per scale row
  GlmQuantMatrix v;
  v.payload = m.payload + row_start * m.cols;
  v.scales = m.scales + (row_start / m.scale_block_rows) * sb;
  v.rows = rows;
  v.cols = m.cols;
  v.scale_block_rows = m.scale_block_rows;
  v.scale_block_cols = m.scale_block_cols;
  return v;
}

// Compressed resident view of one NVFP4 matrix (docs/nvfp4_plan.md §3.1):
// the checkpoint's bytes, untouched — e2m1 codes two per byte (low nibble
// = even element), one e4m3 scale per 16 elements along K, one F32 global
// scale per tensor (a DEVICE address; the kernels divide the finished dot
// by it once). The dequantized value of element (n, k) is
//   e2m1(code) * (float(scales[n][k/16]) / *global_scale)
// and `e2m1(code) * float(scale)` is EXACT in bf16 (2 + 4 significant
// bits), which is what the kernels rely on.
constexpr int kFp4Group = 16;    // elements per e4m3 block scale (NVFP4)
constexpr int kMxfp4Group = 32;  // elements per e8m0 block scale (MXFP4)

// The same view carries the MXFP4 variant (2026-09-13, DeepSeek-V4.1-Flash,
// docs/deepseek_v41_flash_plan.md D2): scale_group 32, `scales` one e8m0
// byte per 32 elements along K, NO global scale (global_scale null). The
// dequantized value of element (n, k) is then
//   e2m1(code) * 2^(scales[n][k/32] - 127)
// exact in bf16 and fp32 (a power of two times <= 2 significant bits);
// the kernels apply it as one fp32 multiply and skip the epilogue's
// division. The e8m0 code 255 is NaN and propagates.
struct GlmFp4Matrix {
  const uint8_t* payload = nullptr;       // U8 [rows, cols/2]
  const uint8_t* scales = nullptr;        // e4m3 [rows, cols/16] or e8m0 [rows, cols/32]
  const float* global_scale = nullptr;    // F32 [1], device (null under MXFP4)
  int64_t rows = 0;
  int64_t cols = 0;                       // logical K (elements)
  int scale_group = kFp4Group;            // 16: NVFP4 (e4m3 + global); 32: MXFP4 (e8m0)

  bool mxfp4() const { return scale_group == kMxfp4Group; }
  int64_t payload_cols() const { return cols / 2; }
  int64_t scale_cols() const { return cols / scale_group; }
  size_t payload_bytes() const {
    return static_cast<size_t>(rows) * static_cast<size_t>(cols / 2);
  }
  size_t scale_bytes() const {
    return static_cast<size_t>(rows) * static_cast<size_t>(cols / scale_group);
  }
};

// K must be a multiple of the block (one scale per block); a column slice
// must start on a block boundary (16 or 32, both even — a packed byte) and
// span whole blocks. Row slices are free: every row carries its own
// scales.
inline void fp4_check_cols(int64_t cols, const char* who, int scale_group = kFp4Group) {
  if (scale_group != kFp4Group && scale_group != kMxfp4Group)
    throw std::invalid_argument(std::string(who) + ": the fp4 scale group must be 16 or 32");
  if (cols <= 0 || cols % scale_group != 0)
    throw std::invalid_argument(std::string(who) + ": fp4 K must be a positive multiple of " +
                                std::to_string(scale_group));
}

inline GlmFp4Matrix fp4_rows_view(const GlmFp4Matrix& m, int64_t row_start,
                                  int64_t rows) {
  if (row_start < 0 || rows < 0 || rows > m.rows || row_start > m.rows - rows)
    throw std::invalid_argument("fp4_rows_view: range out of bounds");
  GlmFp4Matrix v;
  v.payload = m.payload + row_start * m.payload_cols();
  v.scales = m.scales + row_start * m.scale_cols();
  v.global_scale = m.global_scale;
  v.rows = rows;
  v.cols = m.cols;
  v.scale_group = m.scale_group;
  return v;
}

// Compressed resident view of one packed-int matrix (docs/glm53_plan.md
// §1.5, D2): the checkpoint's bytes, untouched — compressed-tensors
// `pack-quantized`, symmetric, group 64. Codes are 4- or 8-bit unsigned
// with an offset of 2^(bits-1) (nibble - 8, byte - 128), packed
// `32 / bits` per I32 word along K with the low nibble / byte the first
// element; one bf16 scale per 64 elements along K. The dequantized value
// of element (n, k) is
//   (code - 2^(bits-1)) * float(scales[n][k/64])
// exactly in fp32 (an 8-bit integer times an 8-bit-mantissa scale), and
// the kernels keep it exact: no weight is rounded to bf16 (the exact
// policy, packq_gemv.cuh).
//
// The scale format (2026-09-28, docs/qwen38_autoround_int4_plan.md D1)
// selects the group and the scale dtype; the code packing is the same:
//   kPackedScaleBf16G64 (0): one bf16 scale per 64 codes — the form above.
//   kPackedScaleF16G128 (1): one IEEE f16 scale per 128 codes — GPTQ /
//     AutoRound int4 g128 and int8 g128 (the Qwen3.8 AutoRound hybrid): the
//     checkpoint's own scales, untouched, widened to fp32 exactly in the
//     kernels; the value is (code - 2^(bits-1)) * float(scales[n][k/128]).
constexpr int kPackedGroup = 64;  // elements per bf16 group scale (format 0)
constexpr int kPackedScaleBf16G64 = 0;
constexpr int kPackedScaleF16G128 = 1;
inline int packed_scale_group(int scale_fmt) { return scale_fmt == kPackedScaleF16G128 ? 128 : 64; }
// The host widening of one stored scale (the kernels' exactly).
inline float packed_scale_to_float(uint16_t bits, int scale_fmt) {
  return scale_fmt == kPackedScaleF16G128 ? fp16_bits_to_float(bits) : bf16_bits_to_float(bits);
}

// The payload layout: rows of codes in k order (every packed matrix), or
// the int8 head's bit planes (2026-09-29, kernels/packq_head.hpp): each
// 128-code group as [64 B high nibbles][32 B bits 3..2][32 B bits 1..0].
constexpr int kPackedLayoutRows = 0;
constexpr int kPackedLayoutPlanes8 = 1;

struct GlmPackedMatrix {
  const uint32_t* packed = nullptr;       // I32 [rows, cols*bits/32]
  const uint16_t* scales = nullptr;       // bf16 [rows, cols/64] (format 0) or f16 [rows, cols/128] (format 1)
  int64_t rows = 0;
  int64_t cols = 0;                       // logical K (elements)
  int bits = 0;                           // 4 or 8
  int scale_fmt = kPackedScaleBf16G64;    // kPackedScale*
  int layout = kPackedLayoutRows;         // kPackedLayout*

  int group() const { return packed_scale_group(scale_fmt); }
  int64_t packed_cols() const { return cols * bits / 32; }   // I32 words per row
  int64_t scale_cols() const { return cols / group(); }
  size_t packed_bytes() const {
    return static_cast<size_t>(rows) * static_cast<size_t>(cols) * static_cast<size_t>(bits) / 8;
  }
  size_t scale_bytes() const {
    return static_cast<size_t>(rows) * static_cast<size_t>(scale_cols()) * 2;
  }
};

// K must be a multiple of the group (one scale per group, whole packed
// words); a column slice must start on a group boundary and span whole
// groups. Row slices are free: every row carries its own scales.
inline void packed_check_cols(int64_t cols, int bits, const char* who,
                              int scale_fmt = kPackedScaleBf16G64) {
  if (bits != 4 && bits != 8)
    throw std::invalid_argument(std::string(who) + ": the packed code width must be 4 or 8");
  if (scale_fmt != kPackedScaleBf16G64 && scale_fmt != kPackedScaleF16G128)
    throw std::invalid_argument(std::string(who) + ": unknown packed scale format");
  const int g = packed_scale_group(scale_fmt);
  if (cols <= 0 || cols % g != 0)
    throw std::invalid_argument(std::string(who) + ": packed K must be a positive multiple of " +
                                std::to_string(g));
}

inline GlmPackedMatrix packed_rows_view(const GlmPackedMatrix& m, int64_t row_start,
                                        int64_t rows) {
  if (row_start < 0 || rows < 0 || rows > m.rows || row_start > m.rows - rows)
    throw std::invalid_argument("packed_rows_view: range out of bounds");
  GlmPackedMatrix v;
  v.packed = m.packed + row_start * m.packed_cols();
  v.scales = m.scales + row_start * m.scale_cols();
  v.rows = rows;
  v.cols = m.cols;
  v.bits = m.bits;
  v.scale_fmt = m.scale_fmt;
  return v;
}

}  // namespace dgpp
