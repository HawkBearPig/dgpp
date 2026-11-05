#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "models/minimax/reference.hpp"

namespace {
void require(bool v, const char* msg) {
  if (!v) throw std::runtime_error(msg);
}
uint16_t b(float v) {
  return dgpp::float_to_bf16_bits(v);
}
}  // namespace
DGPP_TEST(minimax_swiglu_contract) {
  using dgpp::minimax_ref::swiglu;
  require(swiglu(1, -1) == 0, "up has a +1 offset");
  require(swiglu(10, 8) == swiglu(7, 7), "positive gate and up clamps");
  require(swiglu(1, -10) == swiglu(1, -7), "negative up clamp");
  require(swiglu(-10, 1) != swiglu(-7, 1), "negative gate is not clamped");
  require(std::abs(swiglu(1, 1) - 2 / (1 + std::exp(-1.702f))) < 1e-6f, "alpha and beta");
}
DGPP_TEST(minimax_index_max_causal_and_forced_local) {
  std::vector<uint16_t> q(128, b(0)), k(18 * 128 * 128, b(0));
  q[0] = b(1);
  for (int block = 0; block < 17; ++block)
    for (int t = 0; t < 128; ++t) k[(block * 128 + t) * 128] = b(float(block));
  // The first block has one high-scoring token and a low mean. Max selects it.
  k[0] = b(100);
  // Block 17's first token has the worst score, but the local block is forced.
  k[17 * 128 * 128] = b(-100);
  // A future token in block 17 must not affect selection before that position.
  k[(17 * 128 + 1) * 128] = b(1000);
  const auto picked = dgpp::minimax_ref::select_blocks(q.data(), k.data(), 17 * 128, 3);
  require(picked == std::vector<int>({0, 16, 17}), "max pooling and forced local inside top-k");
  require(dgpp::minimax_ref::select_blocks(q.data(), k.data(), 0, 16) == std::vector<int>({0}),
          "short prefix");
  require(dgpp::minimax_ref::select_blocks(q.data(), k.data(), -1, 16).empty(), "padding row");
  k[0] = b(-1);
  const auto before_boundary =
      dgpp::minimax_ref::select_blocks(q.data(), k.data(), 17 * 128 - 1, 2);
  require(before_boundary == std::vector<int>({15, 16}), "future block excluded");
}
DGPP_TEST(minimax_norm_rope_and_router_contract) {
  std::vector<uint16_t> x(128, b(2)), w(128, b(0)), y(128), n(128);
  dgpp::minimax_ref::gemma_norm(x.data(), w.data(), n.data(), 128, 1e-6f);
  require(n[0] == b(1), "Gemma zero weight is unit multiplier");
  dgpp::minimax_ref::norm_rope(x.data(), w.data(), y.data(), 128, 64, 1e-6f, 1e7, 1);
  require(std::equal(n.begin() + 64, n.end(), y.begin() + 64), "nonrotary tail untouched");
  require(y[0] == b(std::cos(1.0f) - std::sin(1.0f)) && y[32] == b(std::cos(1.0f) + std::sin(1.0f)),
          "NeoX pairing");
  const float logits[] = {0, 0, 0, 0}, bias[] = {0, 10, 1, 0};
  std::vector<int> ids;
  std::vector<float> weights;
  dgpp::minimax_ref::route(logits, bias, 4, 2, 2, ids, weights);
  require(ids == std::vector<int>({1, 2}) && weights == std::vector<float>({1, 1}),
          "bias selects, original sigmoid normalizes");
}
DGPP_TEST(minimax_attention_masks_future_and_maps_gqa) {
  std::vector<uint16_t> q(4 * 128, b(0)), k(256 * 2 * 128, b(0)), v(k.size());
  for (int t = 0; t < 256; ++t)
    for (int h = 0; h < 2; ++h)
      for (int d = 0; d < 128; ++d) v[(t * 2 + h) * 128 + d] = b(t > 128 ? 1000.0f : float(h + 1));
  for (int h = 0; h < 4; ++h) {
    const auto a = dgpp::minimax_ref::attention(q.data(), k.data(), v.data(), h, 4, 2, 128, {0, 1});
    for (float x : a) require(x == float(h / 2 + 1), "GQA and causal value mask");
  }
}
