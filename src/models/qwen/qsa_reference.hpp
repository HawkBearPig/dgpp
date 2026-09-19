#pragma once
// Host reference of the QSA pieces (Q3, 2026-09-09; kernels/qsa.hpp): the
// RoPE tables and the per-head norm+RoPE with the reference's bf16 ops,
// the block compression, the indexer score in the DEVICE's fp32 order
// (bitwise), the pinned selection (highest score first, ties to the lower
// block, ids ascending, tail appended), the listed attention (fp32
// softmax, probabilities rounded to bf16 for the value sum, the
// denominator unrounded) and the gate.
#include <cstdint>
#include <vector>

namespace dgpp::qwen_ref {

void rope_inv_freq(double theta, int rotary_dim, std::vector<float>& inv_freq);
// The YaRN table the engine builds under engine.rope_scaling
// (kernels/rope_scaling.hpp): the oracle has to follow the engine there,
// or a YaRN run's parity gate would compare against the plain rope.
void rope_inv_freq_yarn(double theta, int rotary_dim, int64_t correction_max_position,
                        double factor, double beta_fast, double beta_slow,
                        std::vector<float>& inv_freq);
// out[dim] = RoPE(RMSNorm(x) x (1 + w), pos) — bf16 bits in and out.
// `mscale` scales the cos/sin before their bf16 rounding (1.0f = plain).
void qsa_norm_rope(const uint16_t* x, const uint16_t* w, int64_t pos, const float* inv_freq,
                   uint16_t* out, int dim, int rotary_dim, float eps, float mscale = 1.0f);
// The compressed key of kpool raw keys (rows of `dim`) at position pos.
void qsa_index_compress(const uint16_t* raw, int kpool, const uint16_t* w_k,
                        const float* inv_freq, int64_t pos, uint16_t* out, int dim,
                        int rotary_dim, float eps, float mscale = 1.0f);
// sum_h relu(<q_h, c>) / sqrt(128) in the device's order; q [heads, 128]
// (heads <= 4), c [128].
float qsa_index_score(const uint16_t* q, const uint16_t* c, int heads = 4);
// The select_k highest scores (ties to the lower index), ids ascending.
void qsa_select(const std::vector<float>& scores, int select_k, std::vector<int32_t>& ids);
// Pools -> tokens ascending + the tail of position pos; returns the count.
int qsa_expand(const std::vector<int32_t>& pool_ids, int64_t pos, int kpool,
               std::vector<int32_t>& tokens);
// out[h, :] = bf16(attention of q[h] over the listed K/V rows of its kv
// head): q [local_heads, dim], k_rows / v_rows [n, kv_heads * dim].
void qsa_attention(const uint16_t* q, const uint16_t* k_rows, const uint16_t* v_rows, int n,
                   int local_heads, int kv_heads, int dim, float scale, std::vector<float>& c_out);
void qsa_gate_out(const float* c, const uint16_t* gate, int64_t gate_head_stride, uint16_t* out,
                  int heads, int dim);

}  // namespace dgpp::qwen_ref
