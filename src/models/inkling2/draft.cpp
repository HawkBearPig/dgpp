#include "models/inkling2/draft.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/inkling2_draft.hpp"
#include "loaders/safetensors.hpp"

namespace dgpp {

DFlash2Config inkling2_draft_config(const Inkling2Config& c) {
  const Inkling2DraftConfig& d = c.draft;
  if (!d.present) throw std::invalid_argument("engine.mtp: the Inkling 2 checkpoint has no mtp_config");
  DFlash2Config o;
  o.backbone = DFlash2Config::Backbone::Inkling2;
  o.rotary_dim = d.head_dim;
  o.block_size = d.block_size;
  // A mask row predicts the token at its own position: the anchor row is not
  // a proposal. The mask rows are a learned row, so the id only stages them.
  o.sample_from_anchor = false;
  o.mask_token_id = 0;
  o.conv_taps = 0;
  o.selector_rank = d.markov_rank;
  o.selector_top_k = 16;
  o.target_layer_ids = d.target_layer_ids;
  o.num_hidden_layers = d.num_hidden_layers;
  o.hidden_size = d.hidden_size;
  o.vocab_size = c.vocab_size;
  o.num_attention_heads = d.num_attention_heads;
  o.num_key_value_heads = d.num_key_value_heads;
  o.head_dim = d.head_dim;
  o.intermediate_size = d.intermediate_size;
  o.rms_norm_eps = d.rms_norm_eps;
  o.rope_theta = d.rope_theta;
  o.sliding_window = d.sliding_window_size;
  o.max_position_embeddings = d.max_position_embeddings;
  o.local_layer = d.local_layer;
  o.logit_softcap = c.final_logit_softcapping;
  return o;
}

namespace {

using Tensors = std::unordered_map<std::string, const TensorInfo*>;

// One description of the arena, walked three times: without tensors it sizes
// the arena, with tensors and a host buffer it validates and stages every
// byte, and with a device base it binds the views.
struct Layout {
  const Tensors* tensors = nullptr;
  uint8_t* host = nullptr;
  uint8_t* device = nullptr;
  size_t bytes = 0;
  std::unordered_set<std::string> used;

  size_t alloc(size_t n) {
    const size_t at = bytes;
    bytes += (n + 255) & ~size_t{255};
    return at;
  }
  template <class T>
  const T* ptr(size_t at) const {
    return device ? reinterpret_cast<const T*>(device + at) : nullptr;
  }
  const TensorInfo* need(const std::string& name, const std::vector<int64_t>& shape) {
    used.insert(name);
    if (!tensors) return nullptr;
    const auto it = tensors->find(name);
    if (it == tensors->end()) throw std::invalid_argument("Inkling 2 draft: missing tensor " + name);
    const TensorInfo* t = it->second;
    if (t->dtype != DType::BF16 || t->shape != shape)
      throw std::invalid_argument("Inkling 2 draft: tensor dtype or shape: " + name);
    return t;
  }
  // Rows [row0, row0 + nrow) and columns [col0, col0 + ncol) of a BF16
  // [rows, cols] tensor to `dst`, row-major.
  void slice(const std::string& name, int64_t rows, int64_t cols, int64_t row0, int64_t nrow, int64_t col0,
             int64_t ncol, size_t dst) {
    const TensorInfo* t = need(name, {rows, cols});
    if (!host) return;
    const auto* src = static_cast<const uint8_t*>(t->data);
    for (int64_t r = 0; r < nrow; ++r)
      std::memcpy(host + dst + static_cast<size_t>(r) * static_cast<size_t>(ncol) * 2,
                  src + (static_cast<size_t>(r + row0) * static_cast<size_t>(cols) + static_cast<size_t>(col0)) * 2,
                  static_cast<size_t>(ncol) * 2);
  }
  const uint16_t* bf16(const std::string& name, const std::vector<int64_t>& shape) {
    size_t elems = 1;
    for (const int64_t d : shape) elems *= static_cast<size_t>(d);
    const size_t at = alloc(elems * 2);
    const TensorInfo* t = need(name, shape);
    if (host) std::memcpy(host + at, t->data, elems * 2);
    return ptr<uint16_t>(at);
  }

  void build(const DFlash2Config& c, int rank, int world, int64_t vocab_count, int64_t padded_vocab,
             DFlash2Weights& w, const uint16_t** hidden_out_w, const uint16_t** hidden_out_b) {
    if (c.backbone != DFlash2Config::Backbone::Inkling2 || !c.tp_divisible(world) || rank < 0 || rank >= world)
      throw std::invalid_argument("Inkling 2 draft: the heads, key/value heads and MLP rows must divide across the world");
    const int64_t H = c.hidden_size, HD = c.head_dim, R = c.selector_rank;
    const int64_t Q = c.local_q_row(world), K = c.local_kv_row(world), I = c.local_intermediate(world);
    const int L = c.num_hidden_layers;
    const int64_t taps = static_cast<int64_t>(c.target_layer_ids.size());
    w.layers.resize(static_cast<size_t>(L));
    for (int l = 0; l < L; ++l) {
      DFlash2LayerWeights& x = w.layers[static_cast<size_t>(l)];
      const std::string p = "layers." + std::to_string(l) + ".";
      x.input_norm = bf16(p + "input_layernorm.weight", {H});
      x.post_norm = bf16(p + "post_attention_layernorm.weight", {H});
      x.dflare_norm = bf16(p + "dflare_norm.weight", {H});
      x.q_norm = bf16(p + "self_attn.q_norm.weight", {HD});
      x.k_norm = bf16(p + "self_attn.k_norm.weight", {HD});
      x.k_norm_ctx = bf16(p + "self_attn.k_norm_ctx.weight", {HD});
      // q | k | v rows stacked: this rank's heads of each.
      const size_t qkv = alloc(static_cast<size_t>(Q + 2 * K) * H * 2);
      slice(p + "self_attn.q_proj.weight", c.q_row(), H, rank * Q, Q, 0, H, qkv);
      slice(p + "self_attn.k_proj.weight", c.kv_row(), H, rank * K, K, 0, H, qkv + static_cast<size_t>(Q) * H * 2);
      slice(p + "self_attn.v_proj.weight", c.kv_row(), H, rank * K, K, 0, H,
            qkv + static_cast<size_t>(Q + K) * H * 2);
      x.qkv = ptr<uint16_t>(qkv);
      x.kv_rows = x.qkv ? x.qkv + static_cast<size_t>(Q) * H : nullptr;
      const size_t ctx = alloc(static_cast<size_t>(2 * K) * H * 2);
      slice(p + "self_attn.k_proj_ctx.weight", c.kv_row(), H, rank * K, K, 0, H, ctx);
      slice(p + "self_attn.v_proj_ctx.weight", c.kv_row(), H, rank * K, K, 0, H, ctx + static_cast<size_t>(K) * H * 2);
      x.kv_ctx = ptr<uint16_t>(ctx);
      const size_t o = alloc(static_cast<size_t>(H) * Q * 2);
      slice(p + "self_attn.o_proj.weight", H, c.q_row(), 0, H, rank * Q, Q, o);
      x.o = ptr<uint16_t>(o);
      const size_t gate = alloc(static_cast<size_t>(I) * H * 2), up = alloc(static_cast<size_t>(I) * H * 2);
      slice(p + "mlp.gate_proj.weight", c.intermediate_size, H, rank * I, I, 0, H, gate);
      slice(p + "mlp.up_proj.weight", c.intermediate_size, H, rank * I, I, 0, H, up);
      x.gate = ptr<uint16_t>(gate);
      x.up = ptr<uint16_t>(up);
      const size_t down = alloc(static_cast<size_t>(H) * I * 2);
      slice(p + "mlp.down_proj.weight", H, c.intermediate_size, 0, H, rank * I, I, down);
      x.down = ptr<uint16_t>(down);
    }
    w.norm = bf16("norm.weight", {H});
    w.ink.mask_embed = bf16("mask_embed", {H});
    w.ink.feature_norm = bf16("markov_head.dflare_norm.weight", {H});
    w.ink.markov_w1 = bf16("markov_head.markov_w1.weight", {padded_vocab, R});
    w.ink.rnn_w = bf16("markov_head.rnn_proj.weight", {3 * R, 2 * R + 2 * H});
    *hidden_out_w = bf16("markov_head.hidden_out.weight", {H, R});
    *hidden_out_b = bf16("markov_head.hidden_out.bias", {H});

    // The recurrent projection's bias in FP32.
    {
      const size_t at = alloc(static_cast<size_t>(3 * R) * 4);
      const TensorInfo* t = need("markov_head.rnn_proj.bias", {3 * R});
      if (host) {
        const auto* src = static_cast<const uint16_t*>(t->data);
        auto* out = reinterpret_cast<float*>(host + at);
        for (int64_t i = 0; i < 3 * R; ++i) out[i] = bf16_bits_to_float(src[i]);
      }
      w.ink.rnn_bias = ptr<float>(at);
    }
    // The mixes' weights: the softmax of each row of fuse_weight in FP32,
    // rounded to BF16 as the reference stores it. A row per draft layer,
    // then the recurrent head's.
    {
      const size_t at = alloc(static_cast<size_t>(L + 1) * static_cast<size_t>(taps) * 4);
      const TensorInfo* t = need("fuse_weight", {L + 1, taps});
      if (host) {
        const auto* src = static_cast<const uint16_t*>(t->data);
        auto* out = reinterpret_cast<float*>(host + at);
        for (int l = 0; l <= L; ++l) {
          float top = -INFINITY, sum = 0.0f;
          for (int64_t i = 0; i < taps; ++i) top = std::max(top, bf16_bits_to_float(src[l * taps + i]));
          for (int64_t i = 0; i < taps; ++i) {
            out[l * taps + i] = std::exp(bf16_bits_to_float(src[l * taps + i]) - top);
            sum += out[l * taps + i];
          }
          for (int64_t i = 0; i < taps; ++i)
            out[l * taps + i] = bf16_bits_to_float(float_to_bf16_bits(out[l * taps + i] / sum));
        }
      }
      w.ink.fuse_alpha = ptr<float>(at);
    }
    // The folded head: filled on the device once the rest is uploaded.
    w.ink.fold_w = ptr<uint16_t>(alloc(static_cast<size_t>(vocab_count) * static_cast<size_t>(R) * 2));
    w.ink.fold_b = ptr<uint16_t>(alloc(static_cast<size_t>(vocab_count) * 2));

    if (tensors)
      for (const auto& [name, info] : *tensors) {
        (void)info;
        if (!used.contains(name)) throw std::invalid_argument("Inkling 2 draft: unexpected tensor " + name);
      }
  }
};

}  // namespace

size_t inkling2_draft_weights_bytes(const DFlash2Config& cfg, int world, int64_t vocab_count, int64_t padded_vocab) {
  Layout plan;
  DFlash2Weights views;
  const uint16_t* out_w = nullptr;
  const uint16_t* out_b = nullptr;
  plan.build(cfg, 0, world, vocab_count, padded_vocab, views, &out_w, &out_b);
  return plan.bytes;
}

DFlash2Weights load_inkling2_draft(const DFlash2Config& cfg, const std::string& checkpoint_dir, cudaStream_t stream,
                                   int rank, int world, const Inkling2DraftTarget& target) {
  if (target.unembed == nullptr || target.embed_norm == nullptr || target.vocab_count <= 0 || target.padded_vocab <= 0)
    throw std::invalid_argument("Inkling 2 draft: the target's head rows and embedding norm are required");
  const std::unique_ptr<SafetensorsFile> file = SafetensorsFile::open(checkpoint_dir + "/mtp.safetensors");
  Tensors tensors;
  file->for_each([&](const TensorInfo& t) {
    if (!tensors.emplace(t.name, &t).second) throw std::invalid_argument("Inkling 2 draft: duplicate tensor " + t.name);
  });
  const uint16_t* out_w = nullptr;
  const uint16_t* out_b = nullptr;
  DFlash2Weights w;
  Layout preflight;
  preflight.tensors = &tensors;
  // Every name, dtype and shape before any allocation.
  preflight.build(cfg, rank, world, target.vocab_count, target.padded_vocab, w, &out_w, &out_b);
  std::vector<uint8_t> staging(preflight.bytes, 0);
  Layout copy;
  copy.tensors = &tensors;
  copy.host = staging.data();
  copy.build(cfg, rank, world, target.vocab_count, target.padded_vocab, w, &out_w, &out_b);
  w.bytes = preflight.bytes;
  DGPP_CUDA_OK(cudaMalloc(&w.arena, w.bytes));
  Layout bind;
  bind.device = static_cast<uint8_t*>(w.arena);
  bind.build(cfg, rank, world, target.vocab_count, target.padded_vocab, w, &out_w, &out_b);
  DGPP_CUDA_OK(cudaMemcpyAsync(w.arena, staging.data(), w.bytes, cudaMemcpyHostToDevice, stream));
  w.ink.embed_norm = target.embed_norm;
  ink2_draft_fold_head(target.unembed, out_w, out_b, const_cast<uint16_t*>(w.ink.fold_w),
                       const_cast<uint16_t*>(w.ink.fold_b), target.vocab_count, cfg.hidden_size, cfg.selector_rank,
                       stream);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  file->close_mapping(true);
  return w;
}

}  // namespace dgpp
