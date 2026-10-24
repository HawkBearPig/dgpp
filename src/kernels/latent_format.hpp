#pragma once
// The latent cache's storage formats: the KV cache's dtype.
//
// The DSA latent cache holds one kv_lora-wide row per token per DSA layer
// (DESIGN §7.2). It is written once (the latent append after the kv_a
// RMSNorm) and read by every attention over that token, so its format is
// a storage decision the two attention kernels and the append kernel
// share. Three formats:
//
//   bf16  kv_lora x 2 bytes per row; the rows as the projection left them
//         (every parity gate's format, bitwise the historical cache).
//   fp8   kv_lora e4m3 codes per row plus one fp32 row scale
//         (scale = absmax / 448; code = e4m3(x * 448 / absmax)). Half the
//         bytes; ~2^-4 relative error per element.
//   fp4   e2m1 codes, two per byte, in blocks of 16 elements with one e4m3
//         block scale each (the NVFP4 recipe: block scale in units of a
//         per-row fp32 scale so the largest block reaches 448 and the
//         largest element of a block reaches 6), the row padded to 16
//         bytes. kv_lora x 9/16 bytes per row; ~2^-2 relative error per
//         element — a quality trade an operator makes deliberately.
//
// The codecs below are host/device functions with no libm beyond what
// dtypes.hpp already relies on: the device quantizes a row with the same
// arithmetic the host reference runs (the tests pin the codes bitwise),
// and the attention kernels dequantize a row with the same arithmetic the
// host oracle applies, so a quantized-cache attention equals the bf16
// kernel over the dequantized rows within the bf16 kernel's own tolerance.
// Every operation is a single fp32 multiply, divide or conversion (no
// contractable add), so -ffp-contract and --fmad cannot change a bit.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

#include "common/dtypes.hpp"

namespace dgpp {

//
// Two more formats (2026-09-13, DeepSeek-V4.1-Flash, plan D6/D7) carry
// their scales INSIDE the row and no row scale — the release's own
// activation quantizers (inference/kernel.py act_quant / fp4_act_quant),
// reproduced bitwise so a cache row decodes to exactly the bf16 the
// reference stores:
//   fp8_block  e4m3 codes with one e8m0 (power-of-two) scale per 32
//              elements: s = 2^ceil(log2(max(absmax, 1e-4) / 448)),
//              code = e4m3(x / s). The window KV rows (528 bytes at 512).
//   fp4_block  e2m1 codes with one e4m3 scale per 16 elements:
//              s = e4m3(max(absmax, 6 * 2^-9) / 6), code = e2m1(x / s)
//              (a true fp32 division, as the reference's). The compressed
//              main KV (288 bytes at 512: the card's number).
enum class LatentFormat : int { kBf16 = 0, kFp8 = 1, kFp4 = 2, kFp8Block = 3, kFp4Block = 4 };

constexpr const char* latent_format_name(LatentFormat f) {
  switch (f) {
    case LatentFormat::kBf16: return "bf16";
    case LatentFormat::kFp8: return "fp8";
    case LatentFormat::kFp4: return "fp4";
    case LatentFormat::kFp8Block: return "fp8_block";
    case LatentFormat::kFp4Block: return "fp4_block";
  }
  return "?";
}

inline std::optional<LatentFormat> latent_format_from_string(std::string_view s) {
  if (s == "bf16") return LatentFormat::kBf16;
  if (s == "fp8" || s == "fp8_e4m3" || s == "e4m3") return LatentFormat::kFp8;
  if (s == "fp4" || s == "nvfp4" || s == "e2m1") return LatentFormat::kFp4;
  if (s == "fp8_block" || s == "fp8_e8m0") return LatentFormat::kFp8Block;
  if (s == "fp4_block" || s == "fp4_e4m3") return LatentFormat::kFp4Block;
  return std::nullopt;
}

constexpr int kLatentFp4Block = 16;      // elements per e4m3 block scale (fp4, fp4_block)
constexpr int kLatentFp8BlockGroup = 32; // elements per e8m0 scale (fp8_block)
constexpr float kLatentFp8Max = 448.0f;
constexpr float kLatentFp4Max = 6.0f;

// Bytes of one latent row (a token's kv_lora elements in this format).
constexpr size_t latent_row_bytes(LatentFormat f, int kv_lora) {
  switch (f) {
    case LatentFormat::kBf16: return static_cast<size_t>(kv_lora) * 2;
    case LatentFormat::kFp8: return static_cast<size_t>(kv_lora);
    case LatentFormat::kFp4:
    case LatentFormat::kFp4Block: {
      const size_t raw = static_cast<size_t>(kv_lora) / 2 +
                         static_cast<size_t>(kv_lora) / kLatentFp4Block;
      return (raw + 15) / 16 * 16;
    }
    case LatentFormat::kFp8Block: {
      const size_t raw = static_cast<size_t>(kv_lora) +
                         static_cast<size_t>(kv_lora) / kLatentFp8BlockGroup;
      return (raw + 15) / 16 * 16;
    }
  }
  return 0;
}
// Whether the format keeps a per-row fp32 scale beside the rows.
constexpr bool latent_format_has_row_scale(LatentFormat f) {
  return f == LatentFormat::kFp8 || f == LatentFormat::kFp4;
}
// The row's block-scale bytes start here (fp4 and fp4_block: after the
// nibbles; fp8_block: after the codes).
constexpr size_t latent_fp4_scale_offset(int kv_lora) {
  return static_cast<size_t>(kv_lora) / 2;
}
constexpr size_t latent_fp8_block_scale_offset(int kv_lora) {
  return static_cast<size_t>(kv_lora);
}

// ---- e8m0 -----------------------------------------------------------------
// A power-of-two scale as its e8m0 byte (2^(b - 127); 0 is 2^-127, 255
// NaN) and back. The reference picks the byte as ceil(log2(x)) of the
// fp32 x by its bit fields (kernel.py fast_log2_ceil): the exponent field
// plus one when any mantissa bit is set.
DGPP_HD inline uint8_t e8m0_ceil_log2_byte(float x) {
  const uint32_t u = std::bit_cast<uint32_t>(x);
  const int exp = static_cast<int>((u >> 23) & 0xFFu);
  const int k = exp - 127 + ((u & 0x7FFFFFu) != 0u ? 1 : 0);
  const int b = k + 127;
  return static_cast<uint8_t>(b < 0 ? 0 : (b > 254 ? 254 : b));
}
DGPP_HD inline float e8m0_byte_to_float(uint8_t b) {
  if (b == 255u) return std::bit_cast<float>(0x7FC00000u);
  if (b == 0u) return std::bit_cast<float>(0x00400000u);  // 2^-127, an fp32 subnormal
  return std::bit_cast<float>(static_cast<uint32_t>(b) << 23);
}

// ---- e2m1 ---------------------------------------------------------------
// Values {0, .5, 1, 1.5, 2, 3, 4, 6} with a sign bit (bit 3); the low
// nibble of a byte is the even element. Encode is round-to-nearest-even on
// that grid, saturating; NaN saturates too (no NaN code exists).
DGPP_HD inline float fp4_e2m1_bits_to_float(uint8_t v) {
  // The eight magnitudes as twice their value, one nibble each (code 7
  // first): a shift, a mask, an exact halving and the sign bit — no
  // branch. The switch this replaces (2026-10-18) diverged a warp up to
  // eight ways per element in the attention's row decode, 64 elements per
  // thread per tile, and was the listed kernel's largest instruction
  // stream; the values are bitwise the same.
  const uint32_t mag2 = (0xC8643210u >> (4u * (v & 7u))) & 0xFu;
  const float mag = static_cast<float>(mag2) * 0.5f;
  return std::bit_cast<float>(std::bit_cast<uint32_t>(mag) | ((static_cast<uint32_t>(v) & 8u) << 28));
}

DGPP_HD inline uint8_t float_to_fp4_e2m1_bits(float f) {
  const uint32_t u = std::bit_cast<uint32_t>(f);
  const uint8_t sign = static_cast<uint8_t>((u >> 28) & 8u);
  const uint32_t absbits = u & 0x7FFFFFFFu;
  if (absbits > 0x7F800000u) return static_cast<uint8_t>(sign | 7u);  // NaN
  const float a = std::bit_cast<float>(absbits);
  uint8_t code;
  if (a <= 0.25f) code = 0;         // tie at .25 -> 0 (even)
  else if (a < 0.75f) code = 1;
  else if (a <= 1.25f) code = 2;    // ties at .75 and 1.25 -> 1.0 (even)
  else if (a < 1.75f) code = 3;
  else if (a <= 2.5f) code = 4;     // ties at 1.75 and 2.5 -> 2.0 (even)
  else if (a < 3.5f) code = 5;
  else if (a <= 5.0f) code = 6;     // ties at 3.5 and 5 -> 4.0 (even)
  else code = 7;
  return static_cast<uint8_t>(sign | code);
}

// ---- the fp8 row --------------------------------------------------------
// scale = absmax / 448 (0 for an all-zero row: every code is 0 and the
// decode is exact); inv is what the encoder multiplies by.
struct LatentFp8Scale {
  float scale;
  float inv;
};
DGPP_HD inline LatentFp8Scale latent_fp8_row_scale(float absmax) {
  LatentFp8Scale s;
  s.scale = absmax * (1.0f / kLatentFp8Max);
  s.inv = absmax > 0.0f ? kLatentFp8Max / absmax : 0.0f;
  return s;
}
DGPP_HD inline uint8_t latent_fp8_encode(float x, float inv) {
  return float_to_fp8_e4m3_bits(x * inv);
}
// The bf16 the attention kernels see for a stored code.
DGPP_HD inline uint16_t latent_fp8_decode_bf16(uint8_t code, float scale) {
  return float_to_bf16_bits(fp8_e4m3_bits_to_float(code) * scale);
}

// ---- the fp4 row --------------------------------------------------------
// row scale s_r = absmax / (6 * 448); a block's e4m3 scale code is
// e4m3(block_absmax * 448 / absmax), so its decoded value times s_r is the
// block's absolute scale S (the block's largest element lands near 6);
// codes are e2m1(x / S).
struct LatentFp4RowScale {
  float scale;  // s_r
  float inv;    // 448 / absmax (0 for an all-zero row)
};
DGPP_HD inline LatentFp4RowScale latent_fp4_row_scale(float absmax) {
  LatentFp4RowScale s;
  s.scale = absmax * (1.0f / (kLatentFp4Max * kLatentFp8Max));
  s.inv = absmax > 0.0f ? kLatentFp8Max / absmax : 0.0f;
  return s;
}
DGPP_HD inline uint8_t latent_fp4_block_scale_code(float block_absmax,
                                                   float row_inv) {
  return float_to_fp8_e4m3_bits(block_absmax * row_inv);
}
// The block's absolute scale S from its code and the row scale.
DGPP_HD inline float latent_fp4_block_scale(uint8_t code, float row_scale) {
  return fp8_e4m3_bits_to_float(code) * row_scale;
}
// What the encoder multiplies by: 1/S (0 for an all-zero block).
DGPP_HD inline float latent_fp4_block_inv(float block_scale) {
  return block_scale > 0.0f ? 1.0f / block_scale : 0.0f;
}
DGPP_HD inline uint8_t latent_fp4_encode(float x, float block_inv) {
  return float_to_fp4_e2m1_bits(x * block_inv);
}
DGPP_HD inline uint16_t latent_fp4_decode_bf16(uint8_t code, float block_scale) {
  return float_to_bf16_bits(fp4_e2m1_bits_to_float(code) * block_scale);
}

// ---- the fp8_block row (the release's act_quant, e8m0 scales per 32) ------
// The block's scale byte from its absmax: floor 1e-4, then the fp32
// product absmax * (1 / 448) rounded up to a power of two.
DGPP_HD inline uint8_t latent_fp8_block_scale_byte(float block_absmax) {
  const float a = block_absmax > 1e-4f ? block_absmax : 1e-4f;
  return e8m0_ceil_log2_byte(a * (1.0f / kLatentFp8Max));
}
// code = e4m3(x / s): an exact division by a power of two, a saturating
// RNE encode (the reference clamps at +/-448 first; the encode saturates
// to the same code).
DGPP_HD inline uint8_t latent_fp8_block_encode(float x, float s) {
  return float_to_fp8_e4m3_bits(x / s);
}
DGPP_HD inline uint16_t latent_fp8_block_decode_bf16(uint8_t code, float s) {
  return float_to_bf16_bits(fp8_e4m3_bits_to_float(code) * s);
}

// ---- the fp4_block row (the release's fp4_act_quant with e4m3 scales) ---
// The block's scale code: floor 6 * 2^-9 on the absmax, then e4m3(absmax /
// 6) — a true division, rounded to nearest even.
DGPP_HD inline uint8_t latent_fp4_block_scale_code_abs(float block_absmax) {
  const float floor_v = kLatentFp4Max * 0.001953125f;  // 6 * 2^-9
  const float a = block_absmax > floor_v ? block_absmax : floor_v;
  return float_to_fp8_e4m3_bits(a / kLatentFp4Max);
}
// The block's absolute scale from its code (an e4m3 value, no row scale).
DGPP_HD inline float latent_fp4_block_scale_abs(uint8_t code) {
  return fp8_e4m3_bits_to_float(code);
}
// code = e2m1(x / S): the reference's fp32 division then the saturating
// RNE encode (its clamp at +/-6 lands on the same code).
DGPP_HD inline uint8_t latent_fp4_block_encode_abs(float x, float S) {
  return float_to_fp4_e2m1_bits(x / S);
}

// ---- host reference codecs (the tests' oracle; also any host tooling) ----
// Quantizes one bf16 row into `out` (latent_row_bytes(f, kv_lora) bytes)
// and `*row_scale` (unused for bf16). The device append kernel reproduces
// these bytes bitwise.
inline void latent_quantize_row_host(LatentFormat f, const uint16_t* row,
                                     int kv_lora, uint8_t* out,
                                     float* row_scale) {
  if (f == LatentFormat::kBf16) {
    for (int i = 0; i < kv_lora; ++i) {
      out[2 * i] = static_cast<uint8_t>(row[i] & 0xFFu);
      out[2 * i + 1] = static_cast<uint8_t>(row[i] >> 8);
    }
    if (row_scale) *row_scale = 1.0f;
    return;
  }
  float absmax = 0.0f;
  for (int i = 0; i < kv_lora; ++i) {
    const float a = std::fabs(bf16_bits_to_float(row[i]));
    absmax = a > absmax ? a : absmax;
  }
  if (f == LatentFormat::kFp8) {
    const LatentFp8Scale s = latent_fp8_row_scale(absmax);
    for (int i = 0; i < kv_lora; ++i)
      out[i] = latent_fp8_encode(bf16_bits_to_float(row[i]), s.inv);
    if (row_scale) *row_scale = s.scale;
    return;
  }
  if (f == LatentFormat::kFp8Block) {
    const size_t bytes = latent_row_bytes(f, kv_lora);
    for (size_t i = 0; i < bytes; ++i) out[i] = 0;
    const size_t scale_off = latent_fp8_block_scale_offset(kv_lora);
    for (int b = 0; b < kv_lora / kLatentFp8BlockGroup; ++b) {
      float bmax = 0.0f;
      for (int j = 0; j < kLatentFp8BlockGroup; ++j) {
        const float a = std::fabs(bf16_bits_to_float(row[b * kLatentFp8BlockGroup + j]));
        bmax = a > bmax ? a : bmax;
      }
      const uint8_t sb = latent_fp8_block_scale_byte(bmax);
      out[scale_off + static_cast<size_t>(b)] = sb;
      const float s = e8m0_byte_to_float(sb);
      for (int j = 0; j < kLatentFp8BlockGroup; ++j) {
        const int e = b * kLatentFp8BlockGroup + j;
        out[static_cast<size_t>(e)] = latent_fp8_block_encode(bf16_bits_to_float(row[e]), s);
      }
    }
    if (row_scale) *row_scale = 1.0f;
    return;
  }
  if (f == LatentFormat::kFp4Block) {
    const size_t bytes = latent_row_bytes(f, kv_lora);
    for (size_t i = 0; i < bytes; ++i) out[i] = 0;
    const size_t scale_off = latent_fp4_scale_offset(kv_lora);
    for (int b = 0; b < kv_lora / kLatentFp4Block; ++b) {
      float bmax = 0.0f;
      for (int j = 0; j < kLatentFp4Block; ++j) {
        const float a = std::fabs(bf16_bits_to_float(row[b * kLatentFp4Block + j]));
        bmax = a > bmax ? a : bmax;
      }
      const uint8_t sc = latent_fp4_block_scale_code_abs(bmax);
      out[scale_off + static_cast<size_t>(b)] = sc;
      const float S = latent_fp4_block_scale_abs(sc);
      for (int j = 0; j < kLatentFp4Block; j += 2) {
        const int e = b * kLatentFp4Block + j;
        const uint8_t lo = latent_fp4_block_encode_abs(bf16_bits_to_float(row[e]), S);
        const uint8_t hi = latent_fp4_block_encode_abs(bf16_bits_to_float(row[e + 1]), S);
        out[static_cast<size_t>(e) / 2] = static_cast<uint8_t>(lo | (hi << 4));
      }
    }
    if (row_scale) *row_scale = 1.0f;
    return;
  }
  const LatentFp4RowScale s = latent_fp4_row_scale(absmax);
  const size_t bytes = latent_row_bytes(f, kv_lora);
  for (size_t i = 0; i < bytes; ++i) out[i] = 0;
  const size_t scale_off = latent_fp4_scale_offset(kv_lora);
  for (int b = 0; b < kv_lora / kLatentFp4Block; ++b) {
    float bmax = 0.0f;
    for (int j = 0; j < kLatentFp4Block; ++j) {
      const float a = std::fabs(bf16_bits_to_float(row[b * kLatentFp4Block + j]));
      bmax = a > bmax ? a : bmax;
    }
    const uint8_t sc = latent_fp4_block_scale_code(bmax, s.inv);
    out[scale_off + static_cast<size_t>(b)] = sc;
    const float inv = latent_fp4_block_inv(latent_fp4_block_scale(sc, s.scale));
    for (int j = 0; j < kLatentFp4Block; j += 2) {
      const int e = b * kLatentFp4Block + j;
      const uint8_t lo = latent_fp4_encode(bf16_bits_to_float(row[e]), inv);
      const uint8_t hi = latent_fp4_encode(bf16_bits_to_float(row[e + 1]), inv);
      out[static_cast<size_t>(e) / 2] = static_cast<uint8_t>(lo | (hi << 4));
    }
  }
  if (row_scale) *row_scale = s.scale;
}

// Dequantizes one stored row into bf16 — exactly the values the attention
// kernels load.
inline void latent_dequantize_row_host(LatentFormat f, const uint8_t* in,
                                       float row_scale, int kv_lora,
                                       uint16_t* out) {
  if (f == LatentFormat::kBf16) {
    for (int i = 0; i < kv_lora; ++i)
      out[i] = static_cast<uint16_t>(in[2 * i] | (in[2 * i + 1] << 8));
    return;
  }
  if (f == LatentFormat::kFp8) {
    for (int i = 0; i < kv_lora; ++i)
      out[i] = latent_fp8_decode_bf16(in[i], row_scale);
    return;
  }
  if (f == LatentFormat::kFp8Block) {
    const size_t scale_off = latent_fp8_block_scale_offset(kv_lora);
    for (int i = 0; i < kv_lora; ++i) {
      const float s = e8m0_byte_to_float(in[scale_off + static_cast<size_t>(i / kLatentFp8BlockGroup)]);
      out[i] = latent_fp8_block_decode_bf16(in[i], s);
    }
    return;
  }
  if (f == LatentFormat::kFp4Block) {
    const size_t scale_off = latent_fp4_scale_offset(kv_lora);
    for (int i = 0; i < kv_lora; ++i) {
      const float S = latent_fp4_block_scale_abs(in[scale_off + static_cast<size_t>(i / kLatentFp4Block)]);
      const uint8_t byte = in[static_cast<size_t>(i) / 2];
      const uint8_t code = (i & 1) ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 0xFu);
      out[i] = latent_fp4_decode_bf16(code, S);
    }
    return;
  }
  const size_t scale_off = latent_fp4_scale_offset(kv_lora);
  for (int i = 0; i < kv_lora; ++i) {
    const float S = latent_fp4_block_scale(
        in[scale_off + static_cast<size_t>(i / kLatentFp4Block)], row_scale);
    const uint8_t byte = in[static_cast<size_t>(i) / 2];
    const uint8_t code = (i & 1) ? static_cast<uint8_t>(byte >> 4)
                                 : static_cast<uint8_t>(byte & 0xFu);
    out[i] = latent_fp4_decode_bf16(code, S);
  }
}

}  // namespace dgpp
