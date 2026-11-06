#include "models/glm_dsa/modification_targets.hpp"

#include <stdexcept>

namespace dgpp {

std::vector<GlmDsaModificationTarget> glm_dsa_modification_targets(const GlmDsaTextConfig& c,
                                                                   int rank, int world) {
  if (world < 1 || rank < 0 || rank >= world)
    throw std::invalid_argument("Heretic targets: require world > 0 and 0 <= rank < world");
  if (c.hidden_size <= 0 || c.num_hidden_layers <= 0 || c.num_attention_heads <= 0 ||
      c.v_head_dim <= 0 || c.intermediate_size <= 0 || c.moe_intermediate_size <= 0 ||
      c.n_shared_experts <= 0 || c.first_k_dense_replace < 0 ||
      c.first_k_dense_replace > c.num_hidden_layers)
    throw std::invalid_argument("Heretic targets: invalid GLM projection geometry");
  const int64_t attn_input = int64_t{c.num_attention_heads} * c.v_head_dim;
  const int64_t shared_input = int64_t{c.n_shared_experts} * c.moe_intermediate_size;
  if (c.num_attention_heads % world || c.intermediate_size % world || shared_input % world)
    throw std::invalid_argument(
        "Heretic targets: world must divide attention heads and MLP widths");

  std::vector<GlmDsaModificationTarget> targets;
  targets.reserve(static_cast<size_t>(c.num_hidden_layers) * 2);
  for (int layer = 0; layer < c.num_hidden_layers; ++layer) {
    const std::string prefix = "model.layers." + std::to_string(layer) + ".";
    const auto add = [&](const std::string& component, const std::string& module, int64_t input,
                         int bits) {
      const int64_t local = input / world;
      targets.push_back({std::to_string(layer) + "/" + component + "/0", prefix + module, component,
                         layer, 0, input, c.hidden_size, rank * local, (rank + 1) * local, bits,
                         bits ? c.packed_group_size : 0});
    };
    add("attn.o_proj", "self_attn.o_proj", attn_input, c.attention_bits_of(layer));
    if (layer < c.first_k_dense_replace)
      add("mlp.down_proj", "mlp.down_proj", c.intermediate_size, 0);
    else
      add("mlp.down_proj", "mlp.shared_experts.down_proj", shared_input, c.shared_bits_of(layer));
  }
  return targets;
}

}  // namespace dgpp
