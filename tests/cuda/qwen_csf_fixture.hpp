#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace qwenfx {
// Reference encoder (the vLLM csf_scales test's rule): base = row min
// capped at 240, codes = raw - base, out-of-window positions carried as
// exceptions with their true byte.
inline void encode_csf(const std::vector<uint8_t>& raw, int64_t rows, int64_t cols,
            std::vector<uint8_t>& fixed, std::vector<uint32_t>& exc) {
  const int64_t slab = 16 * (1 + cols / 2);
  fixed.assign(static_cast<size_t>(rows / 16) * static_cast<size_t>(slab), 0);
  exc.clear();
  for (int64_t g = 0; g < rows / 16; ++g) {
    uint8_t* base = fixed.data() + static_cast<size_t>(g) * static_cast<size_t>(slab);
    uint8_t* packed = base + 16;
    for (int64_t r = 0; r < 16; ++r) {
      const uint8_t* row = raw.data() + static_cast<size_t>(g * 16 + r) * static_cast<size_t>(cols);
      unsigned mn = 255;
      for (int64_t c = 0; c < cols; ++c) mn = std::min(mn, static_cast<unsigned>(row[c]));
      const unsigned b = std::min(mn, 240u);
      base[static_cast<size_t>(r)] = static_cast<uint8_t>(b);
      for (int64_t c = 0; c < cols / 2; ++c) {
        unsigned code[2];
        for (int k = 0; k < 2; ++k) {
          const int64_t cc = 2 * c + k;
          const int v = static_cast<int>(row[cc]) - static_cast<int>(b);
          if (v < 0 || v > 15) {
            const size_t pos = static_cast<size_t>((g * 16 + r) * cols + cc);
            exc.push_back((static_cast<uint32_t>(row[cc]) << 24) | static_cast<uint32_t>(pos));
            code[k] = 0;
          } else {
            code[k] = static_cast<unsigned>(v);
          }
        }
        packed[static_cast<size_t>(r) * static_cast<size_t>(cols / 2) + static_cast<size_t>(c)] =
            static_cast<uint8_t>(code[0] | (code[1] << 4));
      }
    }
  }
}

}  // namespace qwenfx
