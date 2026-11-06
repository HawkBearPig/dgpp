#pragma once
#include <cstdint>
#include <vector>

namespace dgpp::minimax_ref {
// Independent scalar numerical contract. BF16 values are passed as raw bits.
float swiglu(float gate, float up, float alpha = 1.702f, float limit = 7.0f);
void gemma_norm(const uint16_t* x, const uint16_t* weight, uint16_t* y, int dim, float eps);
void norm_rope(const uint16_t* x, const uint16_t* weight, uint16_t* y, int dim, int rotary_dim,
               float eps, double theta, int64_t pos);
// One index head, contiguous unpooled index keys [tokens,128]. Returns
// logical block ids ascending, current block included inside the budget.
std::vector<int> select_blocks(const uint16_t* q, const uint16_t* index_keys, int64_t pos,
                               int topk);
// GQA attention over that index head's selected blocks, causal through pos.
// Main KV rows have kv_heads * 128 elements; head maps to its KV head.
std::vector<float> attention(const uint16_t* q, const uint16_t* k, const uint16_t* v, int head,
                             int heads, int kv_heads, int64_t pos, const std::vector<int>& blocks);
void route(const float* logits, const float* bias, int experts, int topk, float factor,
           std::vector<int>& ids, std::vector<float>& weights);
}  // namespace dgpp::minimax_ref
