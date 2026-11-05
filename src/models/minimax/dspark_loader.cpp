#include <algorithm>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "loaders/safetensors.hpp"
#include "models/minimax/dspark.hpp"

namespace dgpp {
namespace {
using Tensors = std::unordered_map<std::string, const TensorInfo*>;
constexpr const char* prefix = "language_model.model.dspark.";
struct Layout {
  Layout(const Tensors* t = nullptr, uint8_t* h = nullptr, uint8_t* d = nullptr)
      : tensors(t), host(h), device(d) {}
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
  const TensorInfo* need(const std::string& tail, DType dtype, const std::vector<int64_t>& shape) {
    const std::string name = prefix + tail;
    used.insert(name);
    if (!tensors) return nullptr;
    const auto it = tensors->find(name);
    if (it == tensors->end()) throw std::invalid_argument("MiniMax DSpark: missing tensor " + name);
    const auto* t = it->second;
    if (t->dtype != dtype || t->shape != shape)
      throw std::invalid_argument("MiniMax DSpark tensor dtype/shape: " + name);
    return t;
  }
  void slice(const std::string& name, DType type, int rows, int cols, int row0, int nrow, int col0,
             int ncol, size_t dst) {
    const auto* t = need(name, type, {rows, cols});
    if (!host) return;
    const size_t width = dtype_size(type);
    const auto* src = static_cast<const uint8_t*>(t->data);
    for (int r = 0; r < nrow; ++r) {
      const auto* p = src + (static_cast<size_t>(r + row0) * cols + col0) * width;
      if (type == DType::U8 && std::find(p, p + ncol, uint8_t{255}) != p + ncol)
        throw std::invalid_argument("MiniMax DSpark: NaN E8M0 scale in " + name);
      std::memcpy(host + dst + static_cast<size_t>(r) * ncol * width, p, ncol * width);
    }
  }
  const uint16_t* bf16(const std::string& name, const std::vector<int64_t>& shape) {
    size_t elems = 1;
    for (auto d : shape) elems *= static_cast<size_t>(d);
    const size_t at = alloc(elems * 2);
    const auto* t = need(name, DType::BF16, shape);
    if (host) std::memcpy(host + at, t->data, elems * 2);
    return ptr<uint16_t>(at);
  }
  const float* fp32(const std::string& name, const std::vector<int64_t>& shape) {
    size_t elems = 1;
    for (auto d : shape) elems *= static_cast<size_t>(d);
    const size_t at = alloc(elems * 4);
    const auto* t = need(name, DType::BF16, shape);
    if (host) {
      const auto* src = static_cast<const uint16_t*>(t->data);
      auto* out = reinterpret_cast<float*>(host + at);
      for (size_t i = 0; i < elems; ++i) out[i] = bf16_bits_to_float(src[i]);
    }
    return ptr<float>(at);
  }
  void quant(const std::string& name, int rows, int cols, int row0, int nr, int col0, int nc,
             const uint8_t*& payload, const uint8_t*& scales) {
    if (col0 % 32 || nc % 32)
      throw std::invalid_argument("MiniMax DSpark: unaligned MXFP8 TP slice");
    size_t p = alloc(static_cast<size_t>(nr) * nc), s = alloc(static_cast<size_t>(nr) * nc / 32);
    slice(name + ".weight", DType::F8_E4M3, rows, cols, row0, nr, col0, nc, p);
    slice(name + ".weight_scale_inv", DType::U8, rows, cols / 32, row0, nr, col0 / 32, nc / 32, s);
    payload = ptr<uint8_t>(p);
    scales = ptr<uint8_t>(s);
  }
  void build(const DFlash2Config& c, int rank, int world, DFlash2Weights& w) {
    if (c.backbone != DFlash2Config::Backbone::MiniMaxM3 || !c.tp_divisible(world) || rank < 0 ||
        rank >= world || c.hidden_size % 32 || c.local_intermediate(world) % 32)
      throw std::invalid_argument("MiniMax DSpark: invalid native tensor geometry or TP world");
    const int H = c.hidden_size, Q = c.local_q_row(world), K = c.local_kv_row(world);
    const int I = c.local_intermediate(world), HD = c.head_dim;
    w.layers.resize(c.num_hidden_layers);
    for (int l = 0; l < c.num_hidden_layers; ++l) {
      auto& x = w.layers[l];
      const std::string p = "layers." + std::to_string(l) + ".decoder_layer.";
      x.input_norm = bf16(p + "input_layernorm.weight", {H});
      x.post_norm = bf16(p + "post_attention_layernorm.weight", {H});
      x.q_norm = bf16(p + "self_attn.q_norm.weight", {HD});
      x.k_norm = bf16(p + "self_attn.k_norm.weight", {HD});
      // Fusing the row slices leaves the original MXFP8 bytes unchanged.
      const size_t qp = alloc(static_cast<size_t>(Q + 2 * K) * H);
      const size_t qs = alloc(static_cast<size_t>(Q + 2 * K) * H / 32);
      int offset = 0;
      for (const auto* proj : {"q_proj", "k_proj", "v_proj"}) {
        const int rows = offset == 0 ? c.q_row() : c.kv_row();
        const int n = offset == 0 ? Q : K;
        const std::string a = p + "self_attn." + proj;
        slice(a + ".weight", DType::F8_E4M3, rows, H, rank * n, n, 0, H,
              qp + static_cast<size_t>(offset) * H);
        slice(a + ".weight_scale_inv", DType::U8, rows, H / 32, rank * n, n, 0, H / 32,
              qs + static_cast<size_t>(offset) * H / 32);
        offset += n;
      }
      x.qkv_fp8 = ptr<uint8_t>(qp);
      x.qkv_e8m0 = ptr<uint8_t>(qs);
      quant(p + "self_attn.o_proj", H, c.q_row(), 0, H, rank * Q, Q, x.o_fp8, x.o_e8m0);
      quant(p + "mlp.gate_proj", c.intermediate_size, H, rank * I, I, 0, H, x.gate_fp8,
            x.gate_e8m0);
      quant(p + "mlp.up_proj", c.intermediate_size, H, rank * I, I, 0, H, x.up_fp8, x.up_e8m0);
      quant(p + "mlp.down_proj", H, c.intermediate_size, 0, H, rank * I, I, x.down_fp8,
            x.down_e8m0);
    }
    const int nt = static_cast<int>(c.target_layer_ids.size());
    const size_t fc = alloc(static_cast<size_t>(nt) * H * H * 2);
    for (int i = 0; i < nt; ++i)
      slice("fc.weight", DType::BF16, H, nt * H, 0, H, i * H, H,
            fc + static_cast<size_t>(i) * H * H * 2);
    w.fc = ptr<uint16_t>(fc);
    w.hidden_norm = bf16("hidden_norm.weight", {H});
    w.norm = bf16("final_norm.weight", {H});
    w.pred_codebook = bf16("markov_head.markov_w1.weight", {c.vocab_size, c.selector_rank});
    w.succ_codebook = bf16("markov_head.markov_w2.weight", {c.vocab_size, c.selector_rank});
    w.confidence_weight = fp32("confidence_head.proj.weight", {1, H + c.selector_rank});
    w.confidence_bias = fp32("confidence_head.proj.bias", {1});
    if (tensors)
      for (const auto& [name, _] : *tensors)
        if (!used.contains(name))
          throw std::invalid_argument("MiniMax DSpark: unexpected tensor " + name);
  }
};
}  // namespace
size_t minimax_dspark_weights_bytes(const DFlash2Config& c, int world) {
  Layout plan;
  DFlash2Weights views;
  plan.build(c, 0, world, views);
  return plan.bytes;
}
DFlash2Weights load_minimax_dspark(const DFlash2Config& c, const std::string& dir,
                                   cudaStream_t stream, int rank, int world) {
  std::vector<std::filesystem::path> paths;
  for (const auto& entry : std::filesystem::directory_iterator(dir))
    if (entry.path().extension() == ".safetensors") paths.push_back(entry.path());
  std::sort(paths.begin(), paths.end());
  std::vector<std::unique_ptr<SafetensorsFile>> files;
  Tensors tensors;
  for (const auto& path : paths) {
    files.push_back(SafetensorsFile::open(path.string()));
    files.back()->for_each([&](const TensorInfo& t) {
      if (!tensors.emplace(t.name, &t).second)
        throw std::invalid_argument("MiniMax DSpark: duplicate tensor " + t.name);
    });
  }
  Layout preflight{&tensors};
  DFlash2Weights w;
  preflight.build(c, rank, world, w);  // All names, dtypes and shapes before device allocation.
  std::vector<uint8_t> staging(preflight.bytes);
  Layout copy{&tensors, staging.data()};
  copy.build(c, rank, world, w);  // Validate scale payloads before upload.
  w.bytes = preflight.bytes;
  DGPP_CUDA_OK(cudaMalloc(&w.arena, w.bytes));
  Layout bind{nullptr, nullptr, static_cast<uint8_t*>(w.arena)};
  bind.build(c, rank, world, w);
  DGPP_CUDA_OK(cudaMemcpyAsync(w.arena, staging.data(), w.bytes, cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  for (auto& f : files) f->close_mapping(true);
  return w;
}
}  // namespace dgpp
