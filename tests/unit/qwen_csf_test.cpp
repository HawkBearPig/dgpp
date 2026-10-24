// ModelOpt NVFP4 CSF expansion (qwen_csf_scales_expand): round-trips a
// reference encoder (row base = row min capped at 240, out-of-window
// bytes as exceptions) to the exact source bytes, and refuses malformed
// sidecars by name.
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "../cuda/qwen_csf_fixture.hpp"
#include "models/qwen/loader.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (cond) return;
  throw std::runtime_error(what);
}

// Deterministic filler (an LCG): values across the E4M3 byte range.
uint32_t lcg(uint32_t& s) {
  s = s * 1664525u + 1013904223u;
  return s >> 8;
}

std::string refusal(const std::vector<uint8_t>& fixed, const std::vector<uint32_t>& exc, int64_t rows,
                    int64_t cols) {
  try {
    (void)dgpp::qwen_csf_scales_expand(fixed.data(), fixed.size(), exc.data(), exc.size(), rows, cols);
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}

}  // namespace

DGPP_TEST(qwen_csf_expand_round_trips_the_reference_encoder) {
  for (const auto& [rows, cols] : {std::pair<int64_t, int64_t>{32, 8}, {64, 20}, {640, 160}}) {
    std::vector<uint8_t> raw(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    uint32_t s = 0x12345u;
    for (auto& b : raw) b = static_cast<uint8_t>(0x30 + lcg(s) % 0x50);
    // Out-of-window bytes force exceptions.
    for (int i = 0; i < 64; ++i) raw[(lcg(s) * 2654435761u >> 8) % raw.size()] = 0x7E;
    std::vector<uint8_t> fixed;
    std::vector<uint32_t> exc;
    qwenfx::encode_csf(raw, rows, cols, fixed, exc);
    require(!exc.empty(), "outliers encoded");
    const std::vector<uint8_t> got =
        dgpp::qwen_csf_scales_expand(fixed.data(), fixed.size(), exc.data(), exc.size(), rows, cols);
    require(got == raw, "exact scale bytes back");
  }
}

DGPP_TEST(qwen_csf_expand_refuses_malformed_sidecars) {
  std::vector<uint8_t> fixed(16 * (1 + 4), 0x38);
  std::vector<uint32_t> no_exc;
  require(refusal(fixed, no_exc, 16, 8).empty(), "clean streams decode");
  require(refusal(fixed, no_exc, 15, 8).find("16-row") != std::string::npos, "rows multiple of 16");
  require(refusal(fixed, no_exc, 16, 7).find("even-col") != std::string::npos, "cols even");
  std::vector<uint8_t> short_fixed = fixed;
  short_fixed.pop_back();
  require(refusal(short_fixed, no_exc, 16, 8).find("fixed stream") != std::string::npos,
          "fixed size checked");
  std::vector<uint8_t> hot_base = fixed;
  hot_base[3] = 241;
  require(refusal(hot_base, no_exc, 16, 8).find("240") != std::string::npos, "bases capped at 240");
  std::vector<uint32_t> oob = {(0x60u << 24) | 16u * 8u};
  require(refusal(fixed, oob, 16, 8).find("outside") != std::string::npos, "positions in range");
}
