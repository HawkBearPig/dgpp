#pragma once
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {
// CLIP normalization -> BF16 rounding -> FP32 patch-convolution input.
// Still images duplicate their RGB frame along the temporal dimension.
void minimax_vision_patchify(const uint8_t* rgb, float* out, int width, int height,
                             cudaStream_t stream, int frames = 1);
void minimax_vision_cast(const float* x, uint16_t* y, int64_t count, cudaStream_t stream);
// FP32 frequency construction and CUDA trigonometry, matching the trained
// processor's device tables. Output is contiguous [cos, sin].
void minimax_vision_rope_tables(float* out, int width, int height, int rotary_dim, float theta,
                                cudaStream_t stream, int frames = 1);
// Packed token-major QKV -> head-major Q/K/V. Only the first rotary_dim
// dimensions rotate, with NeoX pairs (j, j + rotary_dim/2). The table
// contains half-width [t frequencies | h frequencies | w frequencies].
void minimax_vision_split_rope(const uint16_t* qkv, uint16_t* q, uint16_t* k, uint16_t* v,
                               const float* cos, const float* sin, int n, int hidden, int heads,
                               int rotary_dim, cudaStream_t stream);
}  // namespace dgpp
