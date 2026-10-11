#pragma once
// Block-scaled E4M3 -> BF16 dequantization (DESIGN §4 contract):
//   out[n, k] = decode_e4m3(payload[n, k]) * scales[n / 128][k / 128]
// This is the transient bridge for interfaces that still consume BF16 (the M3
// DSA layer weights); native block-scaled GEMM consumption is the M4
// scale-aware GEMM's job. Kept as a standalone kernel because parity tests
// and the loader both need the exact same rounding (decode-multiply, then a
// single round-to-nearest-even BF16 conversion).
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// Enqueues the dequant for a [rows, cols] E4M3 payload with an F32 scale
// grid: 128x128 blocks (rs = cs = 7, the classic contract), or a stated
// grid as log2 block sizes — rs = 0 with cs = log2ceil(cols) is a channel
// matrix (one scale a row; the Qwen3.8-27B-NVFP4 release's attention half).
// Ragged tail blocks (rows or cols not a multiple of the block size) are
// handled; out-of-block elements never read their scale. Async on `stream`.
void launch_fp8_dequant_blocks(const uint8_t* payload, const float* scales,
                               uint16_t* out_bf16, int64_t rows, int64_t cols,
                               cudaStream_t stream, int rs = 7, int cs = 7);

}  // namespace dgpp
