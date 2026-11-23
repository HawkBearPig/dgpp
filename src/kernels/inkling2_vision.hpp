#pragma once
// Kernels for the Inkling 2 vision tower (models/inkling2/vision.hpp) where
// the Qwen tower's kernels (kernels/qwen_vision.hpp) do not fit. The tower's
// token order is a class row, then the native patches in raster order. A
// visual token is `merge` x `merge` native patches of `patch` pixels.
//
// Rounding follows the released kernels (vLLM's models/inkling2/nvidia/ops/
// vision.py): fp32 arithmetic, rounded to bf16 where the reference's eager
// PyTorch ops store a tensor.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// The canvas [height, width, 3] as native patch rows [n, 3 * patch * patch],
// raster patch order, features ordered (py, px, channel). A pixel is
//   2 * (bf16((v / 255 - mean) / clip_std) * std + mean) - 1
// with the processor's CLIP constants, each operation rounded on its own,
// and v = -1 for a pixel outside [0, valid_width) x [0, valid_height): the
// processor's padding value, whatever the canvas holds there.
void ink2_vision_patches(const uint8_t* rgb, uint16_t* out, int width, int height, int valid_width,
                         int valid_height, int patch, cudaStream_t stream);

// The tower's input rows before ln_pre, [1 + hp * wp, hidden]: row 0 is
// bf16(class + class position). Row 1 + y * wp + x is bf16(patch embedding +
// bf16(position)), the position resampled bilinearly (align_corners false,
// clamped) from the learned [side, side, hidden] table to the image's
// hp x wp native grid.
void ink2_vision_embed_rows(const uint16_t* patch_embed, const uint16_t* position_table,
                            const uint16_t* class_embedding, const uint16_t* class_position, uint16_t* out,
                            int hp, int wp, int side, int hidden, cudaStream_t stream);

// x = bf16(x + bf16(update * gamma)): a sublayer's output through its learned
// scale into the residual. [rows, dim], gamma [dim].
void ink2_vision_scaled_residual(uint16_t* x, const uint16_t* update, const uint16_t* gamma, int rows, int dim,
                                 cudaStream_t stream);

// qkv [n, 3 * hidden] (token-major, the bias applied by the GEMM) into
// head-major q / k [heads, n, head_dim] with the axial rotation, and the
// values widened and transposed, vt [heads, head_dim, n] fp32 (the weight
// operand of the fp32 P @ V). Dimension pairs (2p, 2p + 1) of a head: the
// first head_dim / 4 pairs turn by (column + 1) * inv_freq[p], the rest by
// (row + 1) * inv_freq[p - head_dim / 4]. The class row (row 0) sits at
// position 0 on both axes. inv_freq [head_dim / 4] is 10000^(-4 i / head_dim).
void ink2_vision_split_rope(const uint16_t* qkv, uint16_t* q, uint16_t* k, float* vt, const float* inv_freq,
                            int n, int hidden, int heads, int wp, cudaStream_t stream);

// scores [rows, n] fp32 -> softmax(scores * scale) in place, fp32. The
// released attention keeps its probabilities wide: only the attention's
// output is rounded to bf16.
void ink2_vision_softmax(float* scores, int rows, int n, float scale, cudaStream_t stream);

// The merger's input, [tokens, merge * merge * hidden]: token (ty, tx) takes
// its patches' rows of x [1 + hp * wp, hidden] in (dy, dx) order.
void ink2_vision_pool(const uint16_t* x, uint16_t* out, int hp, int wp, int merge, int hidden,
                      cudaStream_t stream);

}  // namespace dgpp
