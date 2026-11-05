#pragma once

#include <array>

// Fixed output of vLLM's slerp_alpha / justnorm at revision
// 55076950b6c155f384d6766d63563c3bc20a0e3f, dory.py:44-50:
// bf16(justnorm(fp32(justnorm(a)) + abs(fp32(alpha) * scale) *
//               (fp32(justnorm(b)) - fp32(justnorm(a)))))
// Rounding the interpolation to bf16 first changes elements 2, 4 and 6.
// These constants gate both implementations without using either as oracle.
namespace dory_slerp_golden {
inline constexpr std::array<float, 8> a{17, 16, 10, 8, 19, 9, 10, 7};
inline constexpr std::array<float, 8> b{6, 18, 6, 4, 16, 3, 12, 14};
inline constexpr float alpha = 0.0177f;
inline constexpr float scale = 0.05f / 0.0177f;
inline constexpr std::array<float, 8> output{
    0.458984375f, 0.451171875f, 0.2734375f, 0.2177734375f,
    0.52734375f, 0.2431640625f, 0.283203125f, 0.2080078125f};
}  // namespace dory_slerp_golden
