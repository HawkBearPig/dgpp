#pragma once

#include <cstdint>

#include <cuda_fp4.h>
#include <cuda_fp8.h>

namespace dgpp {

// Four aligned elements from one NVFP4 cache head, rounded to FP8 exactly
// as in the publisher's Q8KV4 path. E2M1 * E4M3 is exactly representable
// in FP16 (including the smallest nonzero product), so the paired multiply
// introduces no rounding before the required E4M3 conversion.
__device__ __forceinline__ uint32_t minimax_cached_fp8x4(const uint8_t* head, int dim) {
  const uint16_t codes = *reinterpret_cast<const uint16_t*>(head + dim / 2);
  const __half scale(__nv_cvt_fp8_to_halfraw(head[64 + dim / 16], __NV_E4M3));
  const __half2 scales = __halves2half2(scale, scale);
  const __half2 lo = __hmul2(
      __half2(__nv_cvt_fp4x2_to_halfraw2(static_cast<uint8_t>(codes), __NV_E2M1)), scales);
  const __half2 hi = __hmul2(
      __half2(__nv_cvt_fp4x2_to_halfraw2(static_cast<uint8_t>(codes >> 8), __NV_E2M1)), scales);
  return static_cast<uint32_t>(__nv_cvt_halfraw2_to_fp8x2(lo, __NV_SATFINITE, __NV_E4M3)) |
         (static_cast<uint32_t>(__nv_cvt_halfraw2_to_fp8x2(hi, __NV_SATFINITE, __NV_E4M3)) << 16);
}

}  // namespace dgpp
