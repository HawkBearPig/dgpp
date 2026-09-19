#include "models/glm/vision.hpp"

#include <filesystem>
#include <fstream>
#include <map>

#include "common/cuda_check.hpp"
#include "kernels/gemm.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/glm_vision.hpp"
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
    if (size > bytes) throw std::logic_error("GLM vision: startup workspace is too small");
  }
  uint16_t* b() const { return static_cast<uint16_t*>(p); }
};
std::string read_text(const std::filesystem::path& p) {
  std::ifstream f(p);
  if (!f) throw std::runtime_error("GLM vision: cannot read " + p.string());
  return std::string(std::istreambuf_iterator<char>(f), {});
}
}  // namespace
struct GlmVisionEncoder::Impl {
  GlmVisionConfig c;
  cudaStream_t stream;
  CublasLtGemm gemm;
  std::map<std::string, std::unique_ptr<Buffer>> weights;
  uint64_t hash = 14695981039346656037ull;
  Buffer rgb, patches, x, norm, tmp, qkv, q, k, v, attn, gate, up, scores, probs, work, result, window;
  ImageInput resident_image;  // owned identity of the single-image output
  Impl(const GlmVisionConfig& cfg, const std::string& checkpoint, cudaStream_t s)
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
      for (const auto* t : f->grep("model.visual."))
        if (t->name.starts_with("model.visual."))
          locations.emplace(t->name.substr(13), "model.safetensors");
    }
    std::map<std::string, std::unique_ptr<SafetensorsFile>> files;
    const auto load = [&](const std::string& name, std::vector<int64_t> shape) {
      auto it = locations.find(name);
      if (it == locations.end())
        throw std::runtime_error("GLM vision: missing model.visual." + name);
      auto& file = files[it->second];
      if (!file) file = SafetensorsFile::open((dir / it->second).string());
      const auto& tensor = file->at("model.visual." + name);
      if (tensor.dtype != DType::BF16 || tensor.shape != shape)
        throw std::runtime_error("GLM vision: incompatible BF16 tensor model.visual." + name);
      auto buffer = std::make_unique<Buffer>();
      buffer->resize(tensor.nbytes());
      DGPP_CUDA_OK(
          cudaMemcpyAsync(buffer->p, tensor.data, tensor.nbytes(), cudaMemcpyHostToDevice, stream));
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      const auto* data = static_cast<const uint8_t*>(tensor.data);
      for (size_t i = 0; i < tensor.nbytes(); ++i) {
        hash ^= data[i];
        hash *= 1099511628211ull;
      }
      weights.emplace(name, std::move(buffer));
    };
    const auto linear = [&](const std::string& name, int out, int in, bool bias = true) {
      load(name + ".weight", {out, in});
      if (bias) load(name + ".bias", {out});
    };
    load("patch_embed.proj.weight", {c.hidden, 3, 2, 14, 14});
    load("patch_embed.proj.bias", {c.hidden});
    for (int i = 0; i < c.depth; ++i) {
      const auto p = "blocks." + std::to_string(i) + ".";
      for (const auto* n : {"norm1", "norm2"}) load(p + n + ".weight", {c.hidden});
      for (const auto* n : {"attn.q_norm", "attn.k_norm"})
        load(p + n + ".weight", {c.hidden / c.heads});
      linear(p + "attn.qkv", 3 * c.hidden, c.hidden);
      linear(p + "attn.proj", c.hidden, c.hidden);
      linear(p + "mlp.gate_proj", c.intermediate, c.hidden);
      linear(p + "mlp.up_proj", c.intermediate, c.hidden);
      linear(p + "mlp.down_proj", c.hidden, c.intermediate);
    }
    load("post_layernorm.weight", {c.hidden});
    load("downsample.weight", {c.output, c.hidden, 2, 2});
    load("downsample.bias", {c.output});
    linear("merger.proj", c.output, c.output, false);
    load("merger.post_projection_norm.weight", {c.output});
    load("merger.post_projection_norm.bias", {c.output});
    linear("merger.gate_proj", c.projection, c.output, false);
    linear("merger.up_proj", c.projection, c.output, false);
    linear("merger.down_proj", c.output, c.projection, false);
    if (weights.size() != locations.size())
      throw std::runtime_error("GLM vision: unexpected vision tensors");
    for (auto& [name, file] : files) {
      (void)name;
      file->close_mapping(true);
    }
    // Allocate the maximum workspace before the fabric starts. cudaMalloc/
    // cudaFree during a request can wait on a peer's spinning collective.
    const size_t n = kMaxImageTokens * 4, h = c.hidden;
    const size_t rows = std::max(n * h, static_cast<size_t>(kMaxImageTokens) * c.output);
    rgb.resize(kMaxImagePixels * 3);
    patches.resize(n * 1176 * 2);
    for (auto* buf : {&x, &norm, &tmp}) buf->resize(rows * 2);
    qkv.resize(n * h * 6);
    for (auto* buf : {&q, &k, &v, &attn}) buf->resize(n * h * 2);
    for (auto* buf : {&gate, &up})
      buf->resize(
          std::max(n * c.intermediate, static_cast<size_t>(kMaxImageTokens) * c.projection) * 2);
    scores.resize(std::max({n * n, n * 3 * h, n * c.intermediate,
                            static_cast<size_t>(kMaxImageTokens) * c.output,
                            static_cast<size_t>(kMaxImageTokens) * c.projection}) *
                  4);
    probs.resize(n * n * 2);
    result.resize(static_cast<size_t>(kMaxImageTokens) * c.output * 2);
    window.resize(static_cast<size_t>(GlmVisionConfig::kWindowTokens) * c.output * 2);
    work.resize(64ull << 20);
  }
  const uint16_t* w(const std::string& name) const { return weights.at(name)->b(); }
  void linear(const uint16_t* in, uint16_t* out, int rows, int n, int inner,
              const std::string& name) {
    const auto bias = weights.find(name + ".bias");
    gemm.matmul_linear_bf16(in, w(name + ".weight"),
                            bias == weights.end() ? nullptr : bias->second->b(), out, rows, n,
                            inner, work.p, work.bytes, stream);
  }
  GlmVisionEncoder::Trace observer;
  void trace(const std::string& name, const uint16_t* data, size_t count) {
    if (observer) observer(name, data, count, DType::BF16);
  }
  void encode_one(const ImageInput& image, uint16_t* dst) {
    const int n = image.tokens * 4, h = c.hidden, o = c.output, d = h / c.heads;
    if (image.width % 28 || image.height % 28 ||
        image.tokens != (image.width / 28) * (image.height / 28))
      throw std::invalid_argument("GLM vision: image grid and token count disagree");
    const size_t rows = std::max(static_cast<size_t>(n) * h, static_cast<size_t>(image.tokens) * o);
    rgb.require_capacity(image.rgb.size());
    patches.require_capacity(static_cast<size_t>(n) * 1176 * 2);
    x.require_capacity(rows * 2);
    norm.require_capacity(std::max(rows, static_cast<size_t>(n) * h) * 2);
    tmp.require_capacity(rows * 2);
    qkv.require_capacity(static_cast<size_t>(n) * h * 6);
    for (auto* buf : {&q, &k, &v, &attn}) buf->require_capacity(static_cast<size_t>(n) * h * 2);
    for (auto* buf : {&gate, &up})
      buf->require_capacity(std::max(static_cast<size_t>(n) * c.intermediate,
                           static_cast<size_t>(image.tokens) * c.projection) *
                  2);
    scores.require_capacity(static_cast<size_t>(n) * n * 4);
    probs.require_capacity(static_cast<size_t>(n) * n * 2);
    DGPP_CUDA_OK(
        cudaMemcpyAsync(rgb.p, image.rgb.data(), image.rgb.size(), cudaMemcpyHostToDevice, stream));
    vision_patchify(static_cast<const uint8_t*>(rgb.p), patches.b(), image.width, image.height,
                    stream);
    trace("patches", patches.b(), n * 1176);
    linear(patches.b(), x.b(), n, h, 1176, "patch_embed.proj");
    trace("patch_embed", x.b(), n * h);
    for (int i = 0; i < c.depth; ++i) {
      const auto p = "blocks." + std::to_string(i) + ".";
      const auto t = "layer" + std::to_string(i) + ".";
      vision_rmsnorm(x.b(), w(p + "norm1.weight"), norm.b(), n, h, c.eps, stream);
      trace(t + "norm1", norm.b(), n * h);
      linear(norm.b(), qkv.b(), n, h * 3, h, p + "attn.qkv");
      trace(t + "qkv", qkv.b(), n * h * 3);
      vision_qkv(qkv.b(), q.b(), k.b(), v.b(), w(p + "attn.q_norm.weight"),
                 w(p + "attn.k_norm.weight"), n, h, c.heads, image.width / 14, c.eps, stream);
      trace(t + "q", q.b(), n * h);
      trace(t + "k", k.b(), n * h);
      // Keep all heads in one batched GEMM. Query tiling bounds attention
      // scratch independently of image size. Select the untiled algorithm so
      // query tile dimensions do not change its reduction over K.
      const int query_tile = std::min(128, kMaxImageTokens * 4 / c.heads);
      for (int first = 0; first < n; first += query_tile) {
        const int rows = std::min(query_tile, n - first);
        gemm.matmul_batched_bf16(q.b() + first * d, k.b(), static_cast<float*>(scores.p), rows, n,
                                 d, c.heads, n * d, n * d, rows * n, work.p, work.bytes, stream, n);
        if (observer)
          observer(t + "scores0.tile" + std::to_string(first), scores.p, rows * n, DType::F32);
        vision_softmax(static_cast<const float*>(scores.p), probs.b(), c.heads * rows, n, d,
                       stream);
        if (observer)
          for (int head = 0; head < c.heads; ++head)
            trace(t + "probs" + std::to_string(head) + ".tile" + std::to_string(first),
                  probs.b() + head * rows * n, rows * n);
        gemm.matmul_batched_bf16(probs.b(), v.b(), static_cast<float*>(scores.p), rows, d, n,
                                 c.heads, rows * n, n * d, rows * d, work.p, work.bytes, stream, n,
                                 true);
        vision_store_attention(static_cast<const float*>(scores.p), attn.b(), rows, n, d, c.heads,
                               first, stream);
      }
      vision_unhead(attn.b(), norm.b(), n, h, c.heads, stream);
      trace(t + "attn", norm.b(), n * h);
      linear(norm.b(), tmp.b(), n, h, h, p + "attn.proj");
      trace(t + "attn_proj", tmp.b(), n * h);
      glm_residual_add_bf16(x.p, tmp.p, static_cast<int64_t>(n) * h, stream);
      trace(t + "residual", x.b(), n * h);
      vision_rmsnorm(x.b(), w(p + "norm2.weight"), norm.b(), n, h, c.eps, stream);
      trace(t + "norm2", norm.b(), n * h);
      linear(norm.b(), gate.b(), n, c.intermediate, h, p + "mlp.gate_proj");
      trace(t + "gate", gate.b(), n * c.intermediate);
      linear(norm.b(), up.b(), n, c.intermediate, h, p + "mlp.up_proj");
      trace(t + "up", up.b(), n * c.intermediate);
      vision_swiglu(gate.b(), up.b(), n * c.intermediate, c.swiglu_limit, stream);
      trace(t + "swiglu", gate.b(), n * c.intermediate);
      linear(gate.b(), tmp.b(), n, h, c.intermediate, p + "mlp.down_proj");
      trace(t + "down", tmp.b(), n * h);
      glm_residual_add_bf16(x.p, tmp.p, static_cast<int64_t>(n) * h, stream);
      trace("layer" + std::to_string(i), x.b(), n * h);
    }
    vision_rmsnorm(x.b(), w("post_layernorm.weight"), norm.b(), n, h, c.eps, stream);
    trace("post_norm", norm.b(), n * h);
    vision_merge(norm.b(), x.b(), n, h, stream);
    trace("merge", x.b(), n * h);
    linear(x.b(), tmp.b(), image.tokens, o, 4 * h, "downsample");
    trace("downsample", tmp.b(), image.tokens * o);
    linear(tmp.b(), x.b(), image.tokens, o, o, "merger.proj");
    trace("projection", x.b(), image.tokens * o);
    vision_layernorm_gelu(x.b(), w("merger.post_projection_norm.weight"),
                          w("merger.post_projection_norm.bias"), image.tokens, o, stream,
                          observer ? norm.b() : nullptr);
    trace("projection_norm", norm.b(), image.tokens * o);
    trace("norm_gelu", x.b(), image.tokens * o);
    linear(x.b(), gate.b(), image.tokens, c.projection, o, "merger.gate_proj");
    linear(x.b(), up.b(), image.tokens, c.projection, o, "merger.up_proj");
    vision_swiglu(gate.b(), up.b(), image.tokens * c.projection, c.swiglu_limit, stream);
    trace("merger_swiglu", gate.b(), image.tokens * c.projection);
    linear(gate.b(), dst, image.tokens, o, c.projection, "merger.down_proj");
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  }
};
GlmVisionEncoder::GlmVisionEncoder(const GlmVisionConfig& c, const std::string& path,
                                   cudaStream_t s)
    : impl_(std::make_unique<Impl>(c, path, s)) {}
GlmVisionEncoder::~GlmVisionEncoder() = default;
uint64_t GlmVisionEncoder::digest() const {
  return impl_->hash;
}
const uint16_t* GlmVisionEncoder::encode(const ImageInput& image, Trace trace) {
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
const uint16_t* GlmVisionEncoder::stage(const std::vector<ImageInput>& images,
                                      int64_t first, int64_t end) {
  if (first < 0 || end < first || end - first > GlmVisionConfig::kWindowTokens)
    throw std::invalid_argument("GLM vision: image window exceeds the preallocated workspace");
  // Spans are validated once at admission. Seek past fully cached images.
  auto it = std::lower_bound(images.begin(), images.end(), first,
      [](const ImageInput& im, int64_t pos) { return im.offset + im.tokens <= pos; });
  for (; it != images.end() && it->offset < end; ++it) {
    const auto& im = *it;
    const auto& resident = impl_->resident_image;
    if (resident.tokens != im.tokens || resident.width != im.width ||
        resident.height != im.height || resident.rgb != im.rgb)
      encode(im);
    const int64_t begin = std::max(first, im.offset);
    const int64_t stop = std::min(end, im.offset + im.tokens);
    const size_t h = impl_->c.output;
    DGPP_CUDA_OK(cudaMemcpyAsync(impl_->window.b() + (begin - first) * h,
        impl_->result.b() + (begin - im.offset) * h, (stop - begin) * h * sizeof(uint16_t),
        cudaMemcpyDeviceToDevice, impl_->stream));
  }
  return impl_->window.b();
}
}  // namespace dgpp
