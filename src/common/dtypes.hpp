#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string_view>

#if defined(__CUDACC__)
#define DGPP_HD __host__ __device__
#else
#define DGPP_HD
#endif

namespace dgpp {

enum class DType : int {
  F32,
  F16,
  BF16,
  F8_E4M3,
  I64,
  I32,
  U8,
  I8,        // signed bytes (DeepSeek-V4.1's packed e2m1 expert nibbles)
  F8_E8M0,   // unsigned power-of-two scales, 2^(byte - 127) (MXFP4 / block-fp8 scales)
  U32,       // index/value records (ModelOpt NVFP4 CSF exception tables)
  I16,       // the EXL3 trellis words (exllamav3's int16 tiles)
};

constexpr size_t dtype_size(DType t) {
  switch (t) {
    case DType::F32: return 4;
    case DType::F16: return 2;
    case DType::BF16: return 2;
    case DType::F8_E4M3: return 1;
    case DType::I64: return 8;
    case DType::I32: return 4;
    case DType::U8: return 1;
    case DType::I8: return 1;
    case DType::F8_E8M0: return 1;
    case DType::U32: return 4;
    case DType::I16: return 2;
  }
  return 0;
}

constexpr std::string_view dtype_name(DType t) {
  switch (t) {
    case DType::F32: return "F32";
    case DType::F16: return "F16";
    case DType::BF16: return "BF16";
    case DType::F8_E4M3: return "F8_E4M3";
    case DType::I64: return "I64";
    case DType::I32: return "I32";
    case DType::U8: return "U8";
    case DType::I8: return "I8";
    case DType::F8_E8M0: return "F8_E8M0";
    case DType::U32: return "U32";
    case DType::I16: return "I16";
  }
  return "?";
}

inline std::optional<DType> dtype_from_string(std::string_view s) {
  if (s == "F32") return DType::F32;
  if (s == "F16") return DType::F16;
  if (s == "BF16") return DType::BF16;
  if (s == "F8_E4M3" || s == "F8" || s == "FP8") return DType::F8_E4M3;
  if (s == "I64") return DType::I64;
  if (s == "I32") return DType::I32;
  if (s == "U8" || s == "BOOL") return DType::U8;
  if (s == "I8") return DType::I8;
  if (s == "F8_E8M0") return DType::F8_E8M0;
  if (s == "U32") return DType::U32;
  if (s == "I16") return DType::I16;
  return std::nullopt;
}

// ---- bit-exact numeric converters (host/device) ---------------------

DGPP_HD inline float bf16_bits_to_float(uint16_t b) {
  return std::bit_cast<float>(static_cast<uint32_t>(b) << 16);
}

DGPP_HD inline uint16_t float_to_bf16_bits(float f) {
  // Round-to-nearest-even. NaN needs an explicit path: the integer rounding
  // below assumes a finite exponent, but hardware NaN payloads (e.g. the
  // all-ones NaN that FMUL produces on some paths) carry far enough that
  // u += 0x7fff overflows into the sign bit, silently converting NaN to
  // -0.0. Canonical quiet NaN with the sign preserved — the same policy
  // the fp8 encoder documents.
  uint32_t u = std::bit_cast<uint32_t>(f);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u)
    return static_cast<uint16_t>(((u >> 16) & 0x8000u) | 0x7FC0u);
  uint32_t lsb = (u >> 16) & 1;
  u += 0x7fffu + lsb;
  return static_cast<uint16_t>(u >> 16);
}

// IEEE binary16 (F16) — the GPTQ / AutoRound scale dtype (2026-09-28,
// docs/qwen38_autoround_int4_plan.md D1). Widening is exact (subnormals
// are man x 2^-24, exact in fp32); narrowing rounds to nearest even.
DGPP_HD inline float fp16_bits_to_float(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t man = h & 0x3FFu;
  if (exp == 0) {
    const float v = static_cast<float>(man) * 5.9604644775390625e-08f;  // 2^-24
    return std::bit_cast<float>(std::bit_cast<uint32_t>(v) | sign);
  }
  if (exp == 31) return std::bit_cast<float>(sign | 0x7F800000u | (man << 13));
  return std::bit_cast<float>(sign | ((exp + 112u) << 23) | (man << 13));
}

inline uint16_t float_to_fp16_bits(float f) {
  const uint32_t u = std::bit_cast<uint32_t>(f);
  const uint16_t sign = static_cast<uint16_t>((u >> 16) & 0x8000u);
  const uint32_t a = u & 0x7FFFFFFFu;
  if (a > 0x7F800000u) return static_cast<uint16_t>(sign | 0x7E00u);  // NaN
  if (a >= 0x47800000u) return static_cast<uint16_t>(sign | 0x7C00u);  // >= 2^16: inf
  if (a < 0x38800000u) {  // below 2^-14: subnormal or zero, in units of 2^-24
    const float v = std::bit_cast<float>(a) * 16777216.0f;
    return static_cast<uint16_t>(sign | static_cast<uint32_t>(std::nearbyint(v)));
  }
  const uint32_t mant = a & 0x7FFFFFu, exp = a >> 23;
  uint32_t half = ((exp - 112u) << 10) | (mant >> 13);
  const uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) half += 1u;
  return static_cast<uint16_t>(sign | half);
}

// e4m3 (OCP FN, bias 7): no infinities, max finite 448, 0x7F/0xFF => NaN.
DGPP_HD inline float fp8_e4m3_bits_to_float(uint8_t v) {
  uint32_t sign = static_cast<uint32_t>(v >> 7) << 31;
  uint32_t exp = (v >> 3) & 0xFu;
  uint32_t man = v & 0x7u;
  float result;
  if (exp == 0xF && man == 0x7) {
    result = std::bit_cast<float>(0x7FC00000u);
  } else if (exp == 0) {
    result = static_cast<float>(man) * 0.125f * 0.015625f;  // denormal
  } else {
    float mant = static_cast<float>(man) * 0.125f;
    result = (1.0f + mant) * exp2f(static_cast<float>(exp) - 7.0f);
  }
  uint32_t ubits = std::bit_cast<uint32_t>(result) | sign;
  return std::bit_cast<float>(ubits);
}

// Saturating round-to-nearest-even encode; NaN in -> canonical NaN out.
// Everything beyond the max finite magnitude (+/-448) clamps to it (OCP FN
// saturating-cast convention); no other value ever touches the NaN codes.
DGPP_HD inline uint8_t float_to_fp8_e4m3_bits(float f) {
  uint32_t u = std::bit_cast<uint32_t>(f);
  uint32_t sign = (u >> 24) & 0x80u;
  uint32_t absbits = u & 0x7FFFFFFFu;
  if (absbits > 0x7F800000u) return static_cast<uint8_t>(sign | 0x7Fu);  // NaN
  float av = std::bit_cast<float>(absbits);
  uint32_t man3;
  uint32_t fld;
  if (av < 0.015625f) {  // subnormal zone: quantum 2^-9
    float q = av * 512.0f;                     // = x / 2^-9
    int n = static_cast<int>(lrintf(q));       // ties-to-even
    if (n <= 0) return static_cast<uint8_t>(sign);
    if (n >= 8) {                              // rounds up to min normal
      fld = 1;                                 // exp field 1 -> scale 2^-6
      man3 = 0;
    } else {
      fld = 0;
      man3 = static_cast<uint32_t>(n);
    }
  } else {
    // Purely arithmetic binary-exponent search: bit-field shortcuts mis-handle
    // sub-unity magnitudes and can mint illegal NaN patterns on carry.
    int e = 0;
    float q = av;
    while (q < 1.0f) { q *= 2.0f; --e; }
    while (q >= 2.0f) { q *= 0.5f; ++e; }
    const float space = exp2f(static_cast<float>(e - 3));
    int n = static_cast<int>(lrintf(av / space));
    if (n >= 16) { ++e; n >>= 1; }
    if (e > 8 || (e == 8 && n >= 15))           // into NaN code or beyond
      return static_cast<uint8_t>(sign | 0x7Eu);
    fld = static_cast<uint32_t>(e + 7);
    man3 = static_cast<uint32_t>(n - 8);
  }
  return static_cast<uint8_t>(sign | (fld << 3) | man3);
}

}  // namespace dgpp
