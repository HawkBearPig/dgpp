// The Qwen3.8-Flash-Next vision tower (see models/qwen/vision.hpp for the
// shape and docs/vision.md for the serving path). Two things are worth
// knowing while reading:
//
//  * Token order. patches are listed merge-block first, which is what makes
//    the merger's 2x2 shuffle a free re-view of the rows (the concatenation
//    is a stride change, not a gather) and what the patchify kernel, the
//    position table's permutation and the axial RoPE's position ids all have
//    to agree on.
//  * Nothing here is sampled per token: the position-table indices/weights
//    and the RoPE tables are computed on the host from the image geometry,
//    so every rank produces the same bits and no rank-dependent value can
//    enter a collective.
#include "models/qwen/vision.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/gemm.hpp"
#include "kernels/qwen_vision.hpp"
#include "loaders/safetensors.hpp"

namespace dgpp {
namespace {
struct Buffer {
  void* p = nullptr;
  size_t bytes = 0;
  ~Buffer() { cudaFree(p); }
  void resize(size_t size) {
    if (size <= bytes) return;
    void* next = nullptr;
    DGPP_CUDA_OK(cudaMalloc(&next, size));
    cudaFree(p);
    p = next;
    bytes = size;
  }
  void require_capacity(size_t size) const {
    if (size > bytes) throw std::logic_error("Qwen vision: startup workspace is too small");
  }
  uint16_t* b() const { return static_cast<uint16_t*>(p); }
  float* f() const { return static_cast<float*>(p); }
  int32_t* i() const { return static_cast<int32_t*>(p); }
};
std::string read_text(const std::filesystem::path& p) {
  std::ifstream f(p);
  if (!f) throw std::runtime_error("Qwen vision: cannot read " + p.string());
  return std::string(std::istreambuf_iterator<char>(f), {});
}
}  // namespace

// e2m1 code -> float, the OCP microscaling values (fp4_gemv.cuh's
// bit-placement decode, host-side): +0, .5, 1, 1.5, 2, 3, 4, 6, signed.
float qwen_vision_e2m1(unsigned code) {
  static const float kLut[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                 -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  return kLut[code & 0xFu];
}

// A quantized vision linear widened to BF16 at load (the tower runs the
// BF16 path; prefill-shaped work takes the dequant bridge anyway, so
// load-time widening is the same math done once). MXFP8 e4m3 x 2^(s-127)
// is exact; NVFP4 e2m1 x e4m3-group x global rounds once (RNE), the fp8
// bridge's contract. Nibbles low-first (the MoE kernels' order). Both
// refuse NaN payload/scale codes and malformed geometry by tensor name.
std::vector<uint16_t> qwen_vision_widen_mxfp8(const std::string& name, const uint8_t* weight,
                                             size_t wbytes, const uint8_t* scale, size_t sbytes,
                                             int64_t rows, int64_t cols) {
  auto bad = [&](const std::string& what) -> std::vector<uint16_t> {
    throw std::runtime_error("Qwen vision: " + name + " " + what);
  };
  if (rows <= 0 || cols <= 0 || cols % 32 != 0) bad("is not an MXFP8 geometry");
  if (weight == nullptr || scale == nullptr) bad("has null sidecar data");
  if (wbytes != static_cast<size_t>(rows) * static_cast<size_t>(cols) ||
      sbytes != static_cast<size_t>(rows) * static_cast<size_t>(cols / 32))
    bad("byte counts disagree with the matrix");
  std::vector<uint16_t> out(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t k = 0; k < cols; ++k) {
      const unsigned wb = weight[static_cast<size_t>(i) * static_cast<size_t>(cols) +
                                 static_cast<size_t>(k)];
      const unsigned sb = scale[static_cast<size_t>(i) * static_cast<size_t>(cols / 32) +
                                static_cast<size_t>(k / 32)];
      if (wb == 0x7F || wb == 0xFF) bad("payload has a NaN code");
      if (sb == 255) bad("scale has a NaN code");
      const float v = fp8_e4m3_bits_to_float(static_cast<uint8_t>(wb)) *
                      exp2f(static_cast<float>(static_cast<int>(sb) - 127));
      out[static_cast<size_t>(i) * static_cast<size_t>(cols) + static_cast<size_t>(k)] =
          float_to_bf16_bits(v);
    }
  }
  return out;
}

std::vector<uint16_t> qwen_vision_dequant_nvfp4(const std::string& name, const uint8_t* weight,
                                               size_t wbytes, const uint8_t* scale, size_t sbytes,
                                               float global, int64_t rows, int64_t cols) {
  auto bad = [&](const std::string& what) -> std::vector<uint16_t> {
    throw std::runtime_error("Qwen vision: " + name + " " + what);
  };
  if (rows <= 0 || cols <= 0 || cols % 16 != 0) bad("is not an NVFP4 geometry");
  if (weight == nullptr || scale == nullptr) bad("has null sidecar data");
  if (wbytes != static_cast<size_t>(rows) * static_cast<size_t>(cols / 2) ||
      sbytes != static_cast<size_t>(rows) * static_cast<size_t>(cols / 16))
    bad("byte counts disagree with the matrix");
  if (!(global > 0.0f) || !std::isfinite(global)) bad("global scale is not positive finite");
  const int64_t K2 = cols / 2;
  std::vector<uint16_t> out(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t k = 0; k < cols; ++k) {
      const unsigned pair =
          weight[static_cast<size_t>(i) * static_cast<size_t>(K2) + static_cast<size_t>(k / 2)];
      const unsigned code = (k % 2 == 0) ? (pair & 0xF) : (pair >> 4);
      const unsigned sc =
          scale[static_cast<size_t>(i) * static_cast<size_t>(K2 / 8) + static_cast<size_t>(k / 16)];
      if (sc == 0x7F || sc == 0xFF) bad("scale has a NaN code");
      const float v =
          qwen_vision_e2m1(code) * fp8_e4m3_bits_to_float(static_cast<uint8_t>(sc)) * global;
      out[static_cast<size_t>(i) * static_cast<size_t>(cols) + static_cast<size_t>(k)] =
          float_to_bf16_bits(v);
    }
  }
  return out;
}

std::vector<uint16_t> qwen_vision_widen_linear(const TensorInfo& w, const TensorInfo& s,
                                             const TensorInfo* g, int64_t rows, int64_t cols) {
  const auto check = [](const TensorInfo& t, DType dtype, const std::vector<int64_t>& shape) {
    if (t.dtype != dtype || t.shape != shape)
      throw std::runtime_error("Qwen vision: incompatible dtype or shape on " + t.name);
  };
  if (w.dtype == DType::F8_E4M3) {
    check(w, DType::F8_E4M3, {rows, cols});
    check(s, DType::U8, {rows, cols / 32});
    return qwen_vision_widen_mxfp8(w.name, static_cast<const uint8_t*>(w.data), w.nbytes(),
                                  static_cast<const uint8_t*>(s.data), s.nbytes(), rows, cols);
  }
  if (w.dtype != DType::U8)
    throw std::runtime_error("Qwen vision: unsupported quantized weight " + w.name);
  check(w, DType::U8, {rows, cols / 2});
  check(s, DType::F8_E4M3, {rows, cols / 16});
  if (!g || g->dtype != DType::F32 || !g->data || g->nbytes() != sizeof(float) ||
      !(g->shape.empty() || g->shape == std::vector<int64_t>{1}))
    throw std::runtime_error("Qwen vision: incompatible global scale on " + w.name);
  float global;
  std::memcpy(&global, g->data, sizeof(global));
  return qwen_vision_dequant_nvfp4(w.name, static_cast<const uint8_t*>(w.data), w.nbytes(),
                                   static_cast<const uint8_t*>(s.data), s.nbytes(), global, rows, cols);
}

QwenVisionEncoder::Grid QwenVisionEncoder::grid_of(const ImageInput& image) {
  Grid g;
  g.h_patches = image.height / QwenVisionConfig::kPatch;
  g.w_patches = image.width / QwenVisionConfig::kPatch;
  g.h_blocks = image.height / QwenVisionConfig::kGrid;
  g.w_blocks = image.width / QwenVisionConfig::kGrid;
  g.tokens = g.h_blocks * g.w_blocks;
  g.patches = g.h_patches * g.w_patches;
  return g;
}

struct QwenVisionEncoder::Impl {
  QwenVisionConfig c;
  cudaStream_t stream;
  CublasLtGemm gemm;
  std::map<std::string, std::unique_ptr<Buffer>> weights;
  uint64_t hash = 14695981039346656037ull;
  // rgb, patchified input, residual/normalized/projected, head-major q/k/v
  // and attention output, the fp32 scores tile and bf16 probabilities, the
  // MLP's hidden layer, the merged rows, the position table's gather plan,
  // the rope tables, one image's result, the staged window, GEMM scratch.
  Buffer rgb, patches, x, norm, tmp, qkv, q, k, v, attn, scores, probs, hidden, merged, pos_idx,
      pos_weight, rope, result, window, work;
  ImageInput resident_image;  // owned identity of the single-image output

  Impl(const QwenVisionConfig& cfg, const std::string& checkpoint, cudaStream_t s)
      : c(cfg), stream(s) {
    namespace fs = std::filesystem;
    const fs::path dir(checkpoint);
    std::map<std::string, std::string> locations;
    if (fs::exists(dir / "model.safetensors.index.json")) {
      const auto json = read_text(dir / "model.safetensors.index.json");
      const auto index = minijson::parse(json);
      for (const auto& m : index.root.at("weight_map").members())
        if (m.key.starts_with("model.visual."))
          locations.emplace(m.key.substr(13), std::string(m.value.as_string()));
    } else if (fs::exists(dir / "model.safetensors")) {
      auto f = SafetensorsFile::open((dir / "model.safetensors").string());
      for (const auto* t : f->grep("model.visual.")) locations.emplace(t->name.substr(13), "model.safetensors");
    } else {
      throw std::runtime_error("Qwen vision: no safetensors checkpoint at " + dir.string());
    }
    std::map<std::string, std::unique_ptr<SafetensorsFile>> files;
    const auto fetch = [&](const std::string& name) -> const TensorInfo& {
      auto it = locations.find(name);
      if (it == locations.end())
        throw std::runtime_error("Qwen vision: missing model.visual." + name);
      auto& file = files[it->second];
      if (!file) file = SafetensorsFile::open((dir / it->second).string());
      return file->at("model.visual." + name);
    };
    const auto store = [&](const std::string& name, const void* data, size_t nbytes) {
      auto buffer = std::make_unique<Buffer>();
      buffer->resize(nbytes);
      DGPP_CUDA_OK(
          cudaMemcpyAsync(buffer->p, data, nbytes, cudaMemcpyHostToDevice, stream));
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      const auto* bytes = static_cast<const uint8_t*>(data);
      for (size_t i = 0; i < nbytes; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
      }
      weights.emplace(name, std::move(buffer));
    };
    const auto load = [&](const std::string& name, std::vector<int64_t> shape) {
      const auto& tensor = fetch(name);
      if (tensor.dtype != DType::BF16 || tensor.shape != shape)
        throw std::runtime_error("Qwen vision: incompatible BF16 tensor model.visual." + name);
      store(name, tensor.data, tensor.nbytes());
    };
    // A linear in the checkpoint's form: BF16 direct, or widened from the
    // quantized tower (the FP4-Local export's MXFP8/NVFP4 weights; its BF16
    // biases ship beside them). Anything else refuses by name.
    const auto linear = [&](const std::string& name, int out, int in) {
      const auto& w = fetch(name + ".weight");
      if (w.dtype == DType::BF16) {
        load(name + ".weight", {out, in});
      } else {
        const auto& scale = fetch(name + ".weight_scale");
        const TensorInfo* global = w.dtype == DType::U8 ? &fetch(name + ".weight_scale_2") : nullptr;
        const auto wide = qwen_vision_widen_linear(w, scale, global, out, in);
        store(name + ".weight", wide.data(), wide.size() * 2);
      }
      load(name + ".bias", {out});
    };
    const int h = c.hidden, i = c.intermediate, o = c.output, m = c.merged();
    load("patch_embed.proj.weight", {h, 3, QwenVisionConfig::kTemporal, QwenVisionConfig::kPatch,
                                     QwenVisionConfig::kPatch});
    load("patch_embed.proj.bias", {h});
    load("pos_embed.weight", {c.num_position_embeddings, h});
    for (int b = 0; b < c.depth; ++b) {
      const auto p = "blocks." + std::to_string(b) + ".";
      for (const auto* n : {"norm1", "norm2"}) load(p + n + ".weight", {h});
      for (const auto* n : {"norm1", "norm2"}) load(p + n + ".bias", {h});
      linear(p + "attn.qkv", 3 * h, h);
      linear(p + "attn.proj", h, h);
      linear(p + "mlp.linear_fc1", i, h);
      linear(p + "mlp.linear_fc2", h, i);
    }
    load("merger.norm.weight", {h});
    load("merger.norm.bias", {h});
    linear("merger.linear_fc1", m, m);
    linear("merger.linear_fc2", o, m);
    // Scale sidecars of a quantized tower widen at load (above); the
    // remaining ones a W4A16 form may ship (the activation scale, unused by
    // the weight widening) are tolerated too. Anything else present but
    // unloaded is unexpected.
    for (const auto& kv : locations) {
      if (weights.count(kv.first)) continue;
      if (kv.first.ends_with(".weight_scale") || kv.first.ends_with(".weight_scale_2") ||
          kv.first.ends_with(".input_scale"))
        continue;
      throw std::runtime_error("Qwen vision: unexpected vision tensor model.visual." + kv.first);
    }
    for (auto& [name, file] : files) {
      (void)name;
      file->close_mapping(true);
    }
    // Allocate the maximum workspace before the fabric starts: cudaMalloc/
    // cudaFree during a request can wait on a peer's spinning collective.
    const size_t n = static_cast<size_t>(kMaxImageTokens) * c.patches_per_token();
    const size_t rows = n * h;
    rgb.resize(pixels_for(c.kGrid) * 3);
    patches.resize(n * static_cast<size_t>(c.patch_in()) * 2);
    for (auto* buf : {&x, &norm, &tmp, &q, &k, &v, &attn}) buf->resize(rows * 2);
    qkv.resize(rows * 3 * 2);
    // One query tile across all heads and all unmerged patches; the same
    // buffer is reused for the smaller P@V tile.
    scores.resize(c.attention_score_bytes());
    probs.resize(n * n * 2);
    hidden.resize(std::max(rows, n * static_cast<size_t>(c.intermediate) * 2));
    merged.resize(static_cast<size_t>(kMaxImageTokens) * m * 2);
    pos_idx.resize(n * 4 * sizeof(int32_t));
    pos_weight.resize(n * 4 * sizeof(float));
    rope.resize(2 * n * c.head_dim() * sizeof(float));
    result.resize(static_cast<size_t>(kMaxImageTokens) * o * 2);
    window.resize(static_cast<size_t>(QwenVisionConfig::kWindowTokens + kMaxImageTokens) * o * 2);
    work.resize(64ull << 20);
  }

  const uint16_t* w(const std::string& name) const { return weights.at(name)->b(); }
  void linear(const uint16_t* in, uint16_t* out, int rows, int n, int inner,
              const std::string& name) {
    gemm.matmul_linear_bf16(in, w(name + ".weight"), w(name + ".bias"), out, rows, n, inner, work.p,
                            work.bytes, stream);
  }

  Trace observer;
  void trace(const std::string& name, const uint16_t* data, size_t count) {
    if (observer) observer(name, data, count, DType::BF16);
  }

  // The position table's bilinear resample plan and the axial RoPE tables,
  // in the tower's token order: block (bh, bw) at merge offset (mh, mw), so
  // patch row = bh * merge + mh and column = bw * merge + mw.
  void stage_geometry(int width, int height) {
    const int hp = height / QwenVisionConfig::kPatch, wp = width / QwenVisionConfig::kPatch;
    const int hm = QwenVisionConfig::kMerge, wm = QwenVisionConfig::kMerge;
    const int n = (hp / hm) * (wp / wm) * hm * wm;
    const int side = QwenVisionConfig::kPositionGrid;
    std::vector<int32_t> index(static_cast<size_t>(n) * 4);
    std::vector<float> weight(static_cast<size_t>(n) * 4);
    const int dim = c.head_dim();
    // The tower rotates a head of `dim` with dim/4 frequencies per axis: the
    // angle vector is [h grid, w grid] (dim/2 long) and rotate_half reads it
    // as a full head width, so each angle is stored twice. That is
    // transformers' Qwen2VLVisionRotaryEmbedding(head_dim // 2) -- 18
    // frequencies and 36 angles for this checkpoint's 72-wide head -- and
    // exllamav3's qwen2_position_embedding_grid_2d agrees.
    const int angles = dim / 2, n_freq = dim / 4;
    // [cos | sin], each [n, head_dim].
    std::vector<float> table(2 * static_cast<size_t>(n) * dim);
    float* cos_tab = table.data();
    float* sin_tab = table.data() + static_cast<size_t>(n) * dim;
    std::vector<double> inv_freq(n_freq);
    for (int f = 0; f < n_freq; ++f)
      inv_freq[f] =
          std::pow(c.rope_theta, -static_cast<double>(2 * f) / static_cast<double>(angles));
    const double h_span = static_cast<double>(side - 1) / static_cast<double>(hp - 1);
    const double w_span = static_cast<double>(side - 1) / static_cast<double>(wp - 1);
    for (int token = 0; token < n; ++token) {
      const int block = token / (hm * wm), iy = (token % (hm * wm)) / wm, ix = token % wm;
      const int py = (block / (wp / wm)) * hm + iy, px = (block % (wp / wm)) * hm + ix;
      const double hy = h_span * py, wx = w_span * px;
      const int hf = static_cast<int>(hy), wf = static_cast<int>(wx);
      const int hc = std::min(hf + 1, side - 1), wc = std::min(wf + 1, side - 1);
      const double dh = hy - hf, dw = wx - wf;
      const int64_t base_h = static_cast<int64_t>(hf) * side, base_c = static_cast<int64_t>(hc) * side;
      int32_t* idx = index.data() + static_cast<size_t>(token) * 4;
      float* wgt = weight.data() + static_cast<size_t>(token) * 4;
      idx[0] = static_cast<int32_t>(base_h + wf);
      idx[1] = static_cast<int32_t>(base_h + wc);
      idx[2] = static_cast<int32_t>(base_c + wf);
      idx[3] = static_cast<int32_t>(base_c + wc);
      wgt[0] = static_cast<float>((1 - dh) * (1 - dw));
      wgt[1] = static_cast<float>((1 - dh) * dw);
      wgt[2] = static_cast<float>(dh * (1 - dw));
      wgt[3] = static_cast<float>(dh * dw);
      // Angle f of the h grid turns the pair (f, f + dim/2); angle n_freq + f
      // of the w grid turns (n_freq + f, n_freq + f + dim/2).
      for (int f = 0; f < n_freq; ++f) {
        const double ha = inv_freq[f] * static_cast<double>(py);
        const double wa = inv_freq[f] * static_cast<double>(px);
        const int hi = f, wi = n_freq + f;
        for (int copy = 0; copy < 2; ++copy) {
          const size_t at = static_cast<size_t>(token) * dim + copy * angles;
          cos_tab[at + hi] = static_cast<float>(std::cos(ha));
          sin_tab[at + hi] = static_cast<float>(std::sin(ha));
          cos_tab[at + wi] = static_cast<float>(std::cos(wa));
          sin_tab[at + wi] = static_cast<float>(std::sin(wa));
        }
      }
    }
    DGPP_CUDA_OK(cudaMemcpyAsync(pos_idx.p, index.data(), index.size() * sizeof(int32_t),
                                 cudaMemcpyHostToDevice, stream));
    DGPP_CUDA_OK(cudaMemcpyAsync(pos_weight.p, weight.data(), weight.size() * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));
    DGPP_CUDA_OK(cudaMemcpyAsync(rope.p, table.data(), table.size() * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));
  }

  void encode_one(const ImageInput& image, uint16_t* dst) {
    if (image.grid != QwenVisionConfig::kGrid ||
        image.width % QwenVisionConfig::kGrid || image.height % QwenVisionConfig::kGrid ||
        image.tokens != (image.width / QwenVisionConfig::kGrid) *
                            (image.height / QwenVisionConfig::kGrid))
      throw std::invalid_argument("Qwen vision: image grid and token count disagree");
    const auto g = grid_of(image);
    if (g.patches != g.tokens * c.patches_per_token())
      throw std::invalid_argument("Qwen vision: patch count disagrees with the merge size");
    const int n = g.patches, h = c.hidden, dim = c.head_dim(), m = c.merged();
    const int tokens = g.tokens;
    const size_t rows = static_cast<size_t>(n) * h;
    rgb.require_capacity(image.rgb.size());
    patches.require_capacity(static_cast<size_t>(n) * c.patch_in() * 2);
    for (auto* buf : {&x, &norm, &tmp, &q, &k, &v, &attn}) buf->require_capacity(rows * 2);
    qkv.require_capacity(rows * 3 * 2);
    probs.require_capacity(static_cast<size_t>(n) * n * 2);
    hidden.require_capacity(std::max(rows, static_cast<size_t>(n) * c.intermediate * 2));
    merged.require_capacity(static_cast<size_t>(tokens) * m * 2);
    result.require_capacity(static_cast<size_t>(kMaxImageTokens) * c.output * 2);
    DGPP_CUDA_OK(
        cudaMemcpyAsync(rgb.p, image.rgb.data(), image.rgb.size(), cudaMemcpyHostToDevice, stream));
    stage_geometry(image.width, image.height);
    qwen_vision_patchify(static_cast<const uint8_t*>(rgb.p), patches.b(), image.width, image.height,
                         QwenVisionConfig::kPatch, QwenVisionConfig::kGrid, stream);
    trace("patches", patches.b(), static_cast<size_t>(n) * c.patch_in());
    // The conv over a patch is a linear map of the patchified row.
    linear(patches.b(), x.b(), n, h, c.patch_in(), "patch_embed.proj");
    trace("patch_projection", x.b(), rows);
    qwen_pos_embed_gather(w("pos_embed.weight"), pos_idx.i(), pos_weight.f(), tmp.b(), n, h, stream);
    trace("position_embeddings", tmp.b(), rows);
    qwen_vision_residual_add(x.b(), tmp.b(), static_cast<int64_t>(rows), stream);
    trace("patch_embed", x.b(), rows);
    for (int b = 0; b < c.depth; ++b) {
      const auto p = "blocks." + std::to_string(b) + ".";
      const auto t = "layer" + std::to_string(b) + ".";
      qwen_vision_layernorm(x.b(), w(p + "norm1.weight"), w(p + "norm1.bias"), norm.b(), n, h, c.eps,
                            stream);
      trace(t + "norm1", norm.b(), rows);
      linear(norm.b(), qkv.b(), n, 3 * h, h, p + "attn.qkv");
      trace(t + "qkv", qkv.b(), rows * 3);
      qwen_vision_split_rope(qkv.b(), q.b(), k.b(), v.b(), rope.f(),
                             rope.f() + static_cast<size_t>(n) * dim, n, h, c.heads, stream);
      trace(t + "q", q.b(), rows);
      trace(t + "k", k.b(), rows);
      // Full (non-windowed) attention over the whole image, kept in one
      // batched GEMM per query tile; the untiled algorithm keeps the tile
      // size from changing the reduction over the key dimension.
      const int query_tile =
          std::min(QwenVisionConfig::kQueryTile,
                   kMaxImageTokens * c.patches_per_token() / c.heads);
      for (int first = 0; first < n; first += query_tile) {
        const int tile = std::min(query_tile, n - first);
        gemm.matmul_batched_bf16(q.b() + static_cast<int64_t>(first) * dim, k.b(),
                                 static_cast<float*>(scores.p), tile, n, dim, c.heads,
                                 static_cast<int64_t>(n) * dim, static_cast<int64_t>(n) * dim,
                                 static_cast<int64_t>(tile) * n, work.p, work.bytes, stream, n);
        if (observer)
          observer(t + "scores" + std::to_string(first), scores.p,
                   static_cast<size_t>(c.heads) * tile * n, DType::F32);
        qwen_vision_softmax(static_cast<const float*>(scores.p), probs.b(), c.heads * tile, n,
                            static_cast<float>(1.0 / std::sqrt(static_cast<double>(dim))), stream);
        trace(t + "probs" + std::to_string(first), probs.b(), static_cast<size_t>(c.heads) * tile * n);
        gemm.matmul_batched_bf16(probs.b(), v.b(), static_cast<float*>(scores.p), tile, dim, n,
                                 c.heads, static_cast<int64_t>(tile) * n,
                                 static_cast<int64_t>(n) * dim, static_cast<int64_t>(tile) * dim,
                                 work.p, work.bytes, stream, n, true);
        qwen_vision_store_attention(static_cast<const float*>(scores.p), attn.b(), tile, n, h,
                                    c.heads, first, stream);
      }
      trace(t + "attn", attn.b(), rows);
      qwen_vision_unhead(attn.b(), norm.b(), n, h, c.heads, stream);
      linear(norm.b(), tmp.b(), n, h, h, p + "attn.proj");
      qwen_vision_residual_add(x.b(), tmp.b(), static_cast<int64_t>(rows), stream);
      trace(t + "attn_proj", tmp.b(), rows);
      qwen_vision_layernorm(x.b(), w(p + "norm2.weight"), w(p + "norm2.bias"), norm.b(), n, h, c.eps,
                            stream);
      trace(t + "norm2", norm.b(), rows);
      linear(norm.b(), hidden.b(), n, c.intermediate, h, p + "mlp.linear_fc1");
      trace(t + "mlp_hidden", hidden.b(), static_cast<size_t>(n) * c.intermediate);
      qwen_vision_gelu(hidden.b(), static_cast<int64_t>(n) * c.intermediate, stream);
      trace(t + "mlp_act", hidden.b(), static_cast<size_t>(n) * c.intermediate);
      linear(hidden.b(), tmp.b(), n, h, c.intermediate, p + "mlp.linear_fc2");
      trace(t + "mlp_proj", tmp.b(), rows);
      qwen_vision_residual_add(x.b(), tmp.b(), static_cast<int64_t>(rows), stream);
      trace("layer" + std::to_string(b), x.b(), rows);
    }
    // The merger normalizes each patch, then the 2x2 blocks are concatenated
    // -- already consecutive, so the shuffle is this pointer re-view.
    qwen_vision_layernorm(x.b(), w("merger.norm.weight"), w("merger.norm.bias"), norm.b(), n, h,
                          c.eps, stream);
    trace("merger_norm", norm.b(), rows);
    linear(norm.b(), merged.b(), tokens, m, m, "merger.linear_fc1");
    qwen_vision_gelu_exact(merged.b(), static_cast<int64_t>(tokens) * m, stream);
    linear(merged.b(), dst, tokens, c.output, m, "merger.linear_fc2");
    trace("image_rows", dst, static_cast<size_t>(tokens) * c.output);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  }
};

QwenVisionEncoder::QwenVisionEncoder(const QwenVisionConfig& c, const std::string& path,
                                     cudaStream_t s)
    : impl_(std::make_unique<Impl>(c, path, s)) {}
QwenVisionEncoder::~QwenVisionEncoder() = default;

uint64_t QwenVisionEncoder::digest() const { return impl_->hash; }

const uint16_t* QwenVisionEncoder::encode(const ImageInput& image, Trace trace) {
  validate_image_pixels(image);
  // Invalidate before writing: a failed encode cannot leave a reusable result.
  impl_->resident_image = {};
  impl_->observer = std::move(trace);
  try {
    impl_->encode_one(image, impl_->result.b());
    impl_->resident_image = image;
  } catch (...) {
    impl_->observer = {};
    throw;
  }
  impl_->observer = {};
  return impl_->result.b();
}

const uint16_t* QwenVisionEncoder::stage(const std::vector<ImageInput>& images, int64_t first,
                                         int64_t end) {
  if (first < 0 || end < first ||
      end - first > QwenVisionConfig::kWindowTokens)
    throw std::invalid_argument("Qwen vision: image window exceeds the preallocated workspace");
  // Spans were validated at admission; seek past images that end before the
  // window, then copy each image's rows that fall inside it.
  auto it = std::lower_bound(images.begin(), images.end(), first,
                             [](const ImageInput& im, int64_t pos) { return im.offset + im.tokens <= pos; });
  for (; it != images.end() && it->offset < end; ++it) {
    const auto& im = *it;
    const auto& resident = impl_->resident_image;
    if (resident.tokens != im.tokens || resident.width != im.width ||
        resident.height != im.height || resident.grid != im.grid || resident.rgb != im.rgb)
      encode(im);
    const int64_t begin = std::max(first, im.offset), stop = std::min(end, im.offset + im.tokens);
    const size_t o = impl_->c.output;
    DGPP_CUDA_OK(cudaMemcpyAsync(impl_->window.b() + (begin - first) * o,
                                 impl_->result.b() + (begin - im.offset) * o,
                                 static_cast<size_t>(stop - begin) * o * sizeof(uint16_t),
                                 cudaMemcpyDeviceToDevice, impl_->stream));
  }
  return impl_->window.b();
}
}  // namespace dgpp
