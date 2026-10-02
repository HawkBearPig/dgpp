#pragma once
// Native blockwise-FP8 dense GEMM for GB10 (sm_121a).
//
// Replaces the FP8-dequant + cuBLASLt-BF16 bridge (and the bf16 on-the-fly
// dequant dense path) on prefill-shaped products: the weight stays E4M3,
// the activation is quantized per call in 128x128 blocks, and the math runs
// on mma.sync.m16n8k32.row.col.f32.e4m3.e4m3.f32 (tcgen05/TMEM do not exist
// on this chip, so CUTLASS Sm100 blockwise is not expressible here).
//
//   D[M, N] = Act[M, K] x W[N, K]^T, W E4M3 [N, K] row-major,
//   w_scales F32 [ceil(N/128), K/128] row-major (the checkpoint grid).
//   The activation block scale (absmax/448 over each 128x128 block) is
//   computed on the fly; the k-block product is scaled by sfa*sfb.
//
// Pointer/stride contract mirrors scale_gemm.hpp: act row-major [M, K]
// (act_row_stride_elems >= K), out row-major [M, N] (out stride 0: N).
// k must be a positive multiple of 128; m/n arbitrary (tails guarded).
// act must be 16B-aligned with act_row_stride_elems a multiple of 8;
// w_payload 16B-aligned. See fp8_blockwise_dense_supported().
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// True when the shapes/pointers meet the kernel contract (NOT the env
// gate): k > 0, k % 128 == 0, m/n > 0, alignment for the vectorized tile
// loads.
bool fp8_blockwise_dense_supported(const uint16_t* act, size_t act_row_stride_elems,
                                   const uint8_t* w_payload, int m, int n, int k);

// Env gate (DGPP_FP8BW_DENSE): set to anything but "0" to enable. Unset
// disables: the path stays off until the parity anchors validate it end
// to end.
bool fp8_blockwise_dense_enabled();

void launch_fp8_blockwise_bf16(const uint16_t* act, size_t act_row_stride_elems,
                               const uint8_t* w_payload, const float* w_scales,
                               uint16_t* out, int m, int n, int k, cudaStream_t stream,
                               size_t out_row_stride_elems = 0);
void launch_fp8_blockwise_f32(const uint16_t* act, size_t act_row_stride_elems,
                               const uint8_t* w_payload, const float* w_scales,
                               float* out, int m, int n, int k, cudaStream_t stream,
                               size_t out_row_stride_elems = 0);

// Per-tensor FP8 recipe (on by default; DGPP_FP8_PT_DENSE=0 disables):
// weights requantized once at boot, activations quantized per call, math
// once at boot, activations quantized per call, math via cuBLASLt E4M3xE4M3
// scalar-scale kernels (~175 TFLOP/s on GB10 — the block/outer-vec scale
// modes are sm100-only there, so per-tensor scalar scales are the
// expressible recipe).
bool fp8_per_tensor_enabled();
// The attention half of the recipe (env gate DGPP_FP8_PT_ATTN, default on):
// off keeps the attention projections on the bridge while the MLP runs
// per-tensor (ablation / fallback).
bool fp8_pt_attn_enabled();
// Blockwise-FP8 lm head (env gate DGPP_FP8_HEAD, default on).
bool fp8_head_enabled();

// maxabs over `count` BF16 elements (inf folds to 448, NaN ignored, all-zero
// yields 0 — the quant kernel guards that to scale 1). Multi-CTA vectorized
// reduction, deterministic (max is order-independent).
void launch_fp8_row_maxabs(const uint16_t* data, size_t count, float* out_max,
                           cudaStream_t stream);

// Quantize `count` BF16 elements to E4M3 with the per-tensor map x/mx,
// `dev_max` the same-stream maxabs output (read inside the kernel); the Lt
// scalar scales are then the raw maxabs values.
void launch_fp8_quant_bf16(const uint16_t* in, uint8_t* out, size_t count,
                           const float* dev_max, cudaStream_t stream);

}  // namespace dgpp
