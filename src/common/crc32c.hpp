#pragma once
// CRC-32C (Castagnoli) over a byte range: the NVMe cold tier's per-page
// checksum (engine/nvme_tier.hpp). The hardware instruction where the
// build target has it (every GB10 core: ARMv8 CRC32; x86 SSE4.2), else a
// slice-by-8 table — the same polynomial, the same value.
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__aarch64__)
#include <arm_acle.h>
#include <sys/auxv.h>
#ifndef HWCAP_CRC32
#define HWCAP_CRC32 (1 << 7)
#endif
#elif defined(__x86_64__)
#include <cpuid.h>
#include <nmmintrin.h>
#endif

namespace dgpp {

namespace crc32c_detail {

inline const uint32_t* table() {
  static uint32_t t[8][256];
  static bool built = false;
  if (!built) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? (0x82F63B78u ^ (c >> 1)) : (c >> 1);
      t[0][i] = c;
    }
    for (uint32_t i = 0; i < 256; ++i)
      for (int s = 1; s < 8; ++s) t[s][i] = (t[s - 1][i] >> 8) ^ t[0][t[s - 1][i] & 0xff];
    built = true;
  }
  return &t[0][0];
}

inline uint32_t software(uint32_t crc, const uint8_t* p, size_t n) {
  const uint32_t* t = table();
  crc = ~crc;
  while (n >= 8) {
    uint32_t lo, hi;
    std::memcpy(&lo, p, 4);
    std::memcpy(&hi, p + 4, 4);
    lo ^= crc;
    crc = t[7 * 256 + (lo & 0xff)] ^ t[6 * 256 + ((lo >> 8) & 0xff)] ^ t[5 * 256 + ((lo >> 16) & 0xff)] ^
          t[4 * 256 + (lo >> 24)] ^ t[3 * 256 + (hi & 0xff)] ^ t[2 * 256 + ((hi >> 8) & 0xff)] ^
          t[1 * 256 + ((hi >> 16) & 0xff)] ^ t[0 * 256 + (hi >> 24)];
    p += 8;
    n -= 8;
  }
  while (n--) crc = t[(crc ^ *p++) & 0xff] ^ (crc >> 8);
  return ~crc;
}

#if defined(__aarch64__)
__attribute__((target("+crc"))) inline uint32_t hardware(uint32_t crc, const uint8_t* p, size_t n) {
  crc = ~crc;
  while (n >= 8) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    crc = __crc32cd(crc, v);
    p += 8;
    n -= 8;
  }
  while (n--) crc = __crc32cb(crc, *p++);
  return ~crc;
}
inline bool has_hardware() {
  static const bool on = (getauxval(AT_HWCAP) & HWCAP_CRC32) != 0;
  return on;
}
#elif defined(__x86_64__)
__attribute__((target("sse4.2"))) inline uint32_t hardware(uint32_t crc, const uint8_t* p, size_t n) {
  uint64_t c = ~crc;
  while (n >= 8) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    c = _mm_crc32_u64(c, v);
    p += 8;
    n -= 8;
  }
  uint32_t c32 = static_cast<uint32_t>(c);
  while (n--) c32 = _mm_crc32_u8(c32, *p++);
  return ~c32;
}
inline bool has_hardware() {
  static const bool on = [] {
    unsigned a = 0, b = 0, c = 0, d = 0;
    return __get_cpuid(1, &a, &b, &c, &d) && (c & bit_SSE4_2);
  }();
  return on;
}
#else
inline uint32_t hardware(uint32_t crc, const uint8_t* p, size_t n) { return software(crc, p, n); }
inline bool has_hardware() { return false; }
#endif

}  // namespace crc32c_detail

// The CRC-32C of `bytes` bytes at `data`, continuing `crc` (0 to start).
inline uint32_t crc32c(const void* data, size_t bytes, uint32_t crc = 0) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  return crc32c_detail::has_hardware() ? crc32c_detail::hardware(crc, p, bytes)
                                       : crc32c_detail::software(crc, p, bytes);
}

}  // namespace dgpp
