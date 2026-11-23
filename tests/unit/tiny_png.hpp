#pragma once
// A PNG writer for tests that need an image of a chosen size or with
// transparency: 8-bit RGB or RGBA, filter 0 on every row, the image data in
// stored (uncompressed) deflate blocks.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace tinypng {

inline uint32_t crc32(const std::string& bytes) {
  uint32_t crc = 0xFFFFFFFFu;
  for (const char ch : bytes) {
    crc ^= static_cast<uint8_t>(ch);
    for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
  }
  return ~crc;
}

inline void put32(std::string& out, uint32_t v) {
  for (int shift = 24; shift >= 0; shift -= 8) out += static_cast<char>((v >> shift) & 0xFFu);
}

inline void chunk(std::string& out, const char* type, const std::string& body) {
  put32(out, static_cast<uint32_t>(body.size()));
  const std::string tagged = std::string(type, 4) + body;
  out += tagged;
  put32(out, crc32(tagged));
}

// pixels [height, width, channels], channels 3 (RGB) or 4 (RGBA).
inline std::string encode(const std::vector<uint8_t>& pixels, int width, int height, int channels) {
  if ((channels != 3 && channels != 4) || width < 1 || height < 1 ||
      pixels.size() != static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(channels))
    throw std::invalid_argument("tinypng: geometry");
  const size_t row = static_cast<size_t>(width) * static_cast<size_t>(channels);
  std::string raw;
  raw.reserve((row + 1) * static_cast<size_t>(height));
  for (int y = 0; y < height; ++y) {
    raw += '\0';  // filter: none
    raw.append(reinterpret_cast<const char*>(pixels.data()) + static_cast<size_t>(y) * row, row);
  }
  uint32_t a = 1, b = 0;  // Adler-32 of the raw rows
  for (const char ch : raw) {
    a = (a + static_cast<uint8_t>(ch)) % 65521u;
    b = (b + a) % 65521u;
  }
  std::string z;
  z += static_cast<char>(0x78);
  z += static_cast<char>(0x01);
  for (size_t at = 0; at < raw.size();) {
    const size_t n = std::min<size_t>(65535, raw.size() - at);
    z += static_cast<char>(at + n == raw.size() ? 1 : 0);
    z += static_cast<char>(n & 0xFFu);
    z += static_cast<char>((n >> 8) & 0xFFu);
    z += static_cast<char>(~n & 0xFFu);
    z += static_cast<char>((~n >> 8) & 0xFFu);
    z.append(raw, at, n);
    at += n;
  }
  put32(z, (b << 16) | a);
  std::string out("\x89PNG\r\n\x1a\n", 8);
  std::string header;
  put32(header, static_cast<uint32_t>(width));
  put32(header, static_cast<uint32_t>(height));
  header += static_cast<char>(8);                      // bit depth
  header += static_cast<char>(channels == 3 ? 2 : 6);  // color type: RGB or RGBA
  header.append(3, '\0');                              // compression, filter, interlace
  chunk(out, "IHDR", header);
  chunk(out, "IDAT", z);
  chunk(out, "IEND", std::string());
  return out;
}

}  // namespace tinypng
