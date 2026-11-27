#pragma once
// The grouped MoE GEMM on the DeepSeek release's own operand types: E4M3
// activations with one E8M0 scale per 32 along the row (act_quant) against
// MXFP4 weights (E2M1 codes, one E8M0 scale per 32), multiplied on
// sm_121a's block-scaled tensor cores (mma kind::mxf8f6f4, m16n8k32) with
// FP32 accumulation. See moe_a8w4.cu.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "kernels/glm_moe_launch.hpp"

namespace dgpp {

// The shapes the kernel takes: whole 64-wide stages along k.
inline bool moe_grouped_a8w4_supported(int k) { return k > 0 && k % 64 == 0; }

// out[seg.row0 + m, :n] = act(row) . W(expert, which)^T for every segment,
// act(row) the quantized row act_rows[seg.row0 + m] (or seg.row0 + m when
// act_rows is null): codes [rows, code_stride] E4M3 bytes (16-byte aligned
// rows), scales [rows, scale_stride] E8M0 bytes, the layout of
// launch_act_quant_fp8_block. The views are MXFP4 (payload [n, k/2] with
// the low nibble the even element, fp4_scales [n, k/32] E8M0). Segment,
// view and out conventions as launch_moe_grouped_mma_fp4_{bf16,f32};
// k % 64 == 0. Every output is the sum of its k32 blocks' products in
// ascending k, each block one tensor-core instruction.
void launch_moe_grouped_a8w4_bf16(const uint8_t* codes, size_t code_stride, const uint8_t* scales,
                                  size_t scale_stride, const int32_t* act_rows, const MoeSegment* segs, int n_segs,
                                  int max_rows, const MoeExpertView* views, int which, uint16_t* out,
                                  size_t out_stride, int n, int k, cudaStream_t stream);
void launch_moe_grouped_a8w4_f32(const uint8_t* codes, size_t code_stride, const uint8_t* scales,
                                 size_t scale_stride, const int32_t* act_rows, const MoeSegment* segs, int n_segs,
                                 int max_rows, const MoeExpertView* views, int which, float* out, size_t out_stride,
                                 int n, int k, cudaStream_t stream);

}  // namespace dgpp
