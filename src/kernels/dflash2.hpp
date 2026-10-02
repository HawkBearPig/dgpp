#pragma once
// DFlash2 drafter kernels (docs/performance_improvement_plan.md §7). The
// reference is the released vLLM DFlash2 speculator
// (vllm/model_executor/models/qwen3_dflash2.py and
// vllm/v1/worker/gpu/spec_decode/dflash2/): a block-diffusion drafter whose
// KV context is built from fused target features and whose block of mask
// positions is denoised in ONE bidirectional pass, with two-tap dynamic
// grouped convolutions around attention and the MLP and a codebook path
// selector over the top-K per-position candidates.
//
// All kernels are deterministic; bf16 io (uint16_t bits), fp32 interiors,
// the reference's rounding points named per kernel.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// The 2-tap dynamic grouped convolution (the DFlash2 attention_conv /
// mlp_conv, one side's slice). x and out are [rows, hidden] bf16 rows;
// base is the side's [taps, hidden] bf16 slice and delta the rows'
// [rows, taps, groups] fp32-of-bf16 slice (row stride in elements — the
// kernel_projection's [rows, 2, taps, groups] output sliced per side).
// out[r, c] = sum_t bf16(base[t, c] + delta[r, t, c/group]) * x[r-t, c],
// the tap t applying only when r % block_rows >= t (the conv lives INSIDE
// a request's query block, position = r % block_rows). fp32 accumulation,
// one bf16 rounding.
void dflash2_grouped_conv_bf16(const uint16_t* x, const uint16_t* delta, const uint16_t* base,
                               uint16_t* out, int rows, int block_rows, int hidden, int taps,
                               int group_size, int64_t delta_row_stride, cudaStream_t stream);

// out += src (fp32, elementwise, fixed order). beta 0: out = src. The
// per-tap fc accumulation: the fused target features
// sum_t tap_t @ fc_t^T computed as one fp32 GEMM per tap layer.
void dflash2_acc_f32(float* acc, const float* src, int64_t n, int beta, cudaStream_t stream);

// The standard (plain-weight) RMSNorm — the drafter is a Qwen3 model, not
// the (1+w) zero-centered family form: y = bf16(f32(x) * rsqrt(mean+eps) * w).
void dflash2_rmsnorm_bf16(const uint16_t* x, const uint16_t* w, uint16_t* y, int64_t rows, int dim,
                          float eps, cudaStream_t stream);
// Fused residual add + the standard norm (bitwise the add + norm pair).
void dflash2_add_rmsnorm_bf16(uint16_t* resid, const uint16_t* add, const uint16_t* w, uint16_t* y,
                              int64_t rows, int dim, float eps, cudaStream_t stream);
// The same norm over the fp32 fused-feature accumulator.
void dflash2_norm_f32_bf16(const float* x, const uint16_t* w, uint16_t* y, int64_t rows, int dim,
                           float eps, cudaStream_t stream);

// Per-head standard RMSNorm + RoPE (rotate_half / neox pairs, the bf16
// three-rounding form: bf16(x*cos), bf16(rot*sin), one add rounding).
// head h of row r at x + r*x_row_stride + h*dim; out likewise. Rows with
// pos < 0 are skipped.
void dflash2_norm_rope_bf16(const uint16_t* x, int64_t x_row_stride, const uint16_t* w,
                            const int64_t* pos, const float* inv_freq, uint16_t* out,
                            int64_t out_row_stride, int rows, int heads, int dim, float eps,
                            cudaStream_t stream);

// The drafter's block attention: rows query rows (one request), q bf16
// [rows, heads, dim] (row stride q_row_stride, heads contiguous); GQA
// kv_heads; the paged planes as Qwen35KvPool views (bf16 [slots,
// kv_heads*dim], one physical block table). A key at position p is visible
// to the query at q from either the context span [0, ctx_end] with
// q - p < window (causal, sliding), or the block span [blk_lo, blk_hi]
// with no causal and no window restriction (the block is bidirectional
// and far narrower than any window). fp32 online softmax, bf16
// probabilities into V (the house rule), fp32 output rounded to bf16.
void dflash2_block_attn(const uint16_t* q, int64_t q_row_stride, const uint16_t* k_cache,
                        const uint16_t* v_cache, const int32_t* block_table, int block_tokens,
                        int blocks_per_request, int64_t ctx_end, int64_t blk_lo, int64_t blk_hi,
                        int64_t window, const int64_t* pos, int rows, int heads, int kv_heads,
                        int dim, float scale, uint16_t* out, cudaStream_t stream);

// Per-row top-K of fp32 logits (descending by score, ties to the lower id):
// the candidate sets the selector walks. K <= 32.
void dflash2_topk_f32(const float* logits, int32_t* ids, float* scores, int64_t vocab, int rows,
                      int k, cudaStream_t stream);

// The DFlash2 candidate path selector: the scores[l][p][c] table
// unary[l][p] + <pred_code[id(l-1, p)] * hidden[l], succ_code[id(l, c)]>
// (step 0's predecessor is the anchor token, every slot) walked greedily
// per step: token = ids[l][argmax_c scores[l][prev][c]], prev = that argmax
// (ties to the first). One block, steps sequential, scores within a step
// computed in parallel; the reduction order over the rank is fixed.
// cb bf16 [vocab, rank].
void dflash2_selector_walk(const int32_t* ids, const float* unary, const float* hidden,
                           const uint16_t* pred_cb, const uint16_t* succ_cb, int32_t anchor,
                           int32_t* tokens, int steps, int k, int rank, cudaStream_t stream);

}  // namespace dgpp
