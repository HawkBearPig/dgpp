#pragma once
// Host reference of the Flint attention pieces (2026-10-22;
// kernels/mimo_attn.hpp's 128-wide instantiation): the qkv finish —
// bf16(dot), the per-head one-rounding RMSNorm, the full-head RoPE with
// fp32 cos / sin and one rounding (or none on a no-RoPE layer) — and the
// windowed GQA attention with its sink (fp32 scores, fp32 softmax with the
// sink as one more column, probabilities rounded to bf16 for the value
// sum, the denominator unrounded).
#include <cstdint>
#include <vector>

namespace dgpp::flint_ref {

constexpr int kDim = 128;

// out[128] bf16 bits: a finished q or k head. dot fp32 [128]; norm_w bf16
// [128] (null: no norm), eps; inv_freq fp32 [64] (null: no RoPE); the pair
// (i, i + 64) for i < 64 rotated by pos x inv_freq[i].
void qk_finish(const float* dot, const uint16_t* norm_w, float eps, const float* inv_freq, int64_t pos,
               uint16_t* out);
// out[128] bf16 bits: a finished v head — bf16(dot).
void v_finish(const float* dot, uint16_t* out);

// c_out[h * 128 + d] = attention of q[h] (bf16 [local_heads, 128]) over
// the n K rows (bf16 [n, kv_heads * 128]) / V rows (bf16 [n, kv_heads *
// 128]) of its kv head — the rows the query sees, in order — unrounded
// (the combine's C / L); the caller rounds to bf16. sink: fp32
// [local_heads] or null. `tile` > 0 models the kernel's chain exactly (the
// MiMo reference's contract: tiles, splits, the sink in split 0); tile 0
// is the plain softmax.
void attention(const uint16_t* q, const uint16_t* k_rows, const uint16_t* v_rows, int n, int local_heads,
               int kv_heads, float scale, const float* sink, std::vector<float>& c_out, int tile = 0,
               int n_split = 1);

}  // namespace dgpp::flint_ref
