#pragma once
// The storage-only QSA path keeps BF16 products. Decode E4M3 through the
// hardware conversion (every finite E4M3 value is exact in FP16/FP32),
// then apply the row scale and the same BF16 rounding as the host codec.
#include <cuda_fp8.h>

#include "common/dtypes.hpp"

namespace dgpp {
__device__ __forceinline__ uint16_t qsa_decode_fp8_bf16(uint8_t code, float scale) {
  const float value = __half2float(__half(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
  return float_to_bf16_bits(value * scale);
}
}  // namespace dgpp
