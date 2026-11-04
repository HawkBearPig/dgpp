#include "models/minimax/vision.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>

#include "common/cuda_check.hpp"
#include "kernels/fp32_gemm.hpp"
#include "kernels/gemm.hpp"
#include "kernels/minimax_vision.hpp"
#include "kernels/qwen_vision.hpp"
#include "loaders/safetensors.hpp"

namespace dgpp {
namespace {
struct Buffer {
  void* p = nullptr;
  size_t bytes = 0;
  ~Buffer() { cudaFree(p); }
  void reserve(size_t n) {
    if (p) throw std::logic_error("MiniMax vision: buffer already allocated");
    DGPP_CUDA_OK(cudaMalloc(&p, n));
    bytes = n;
  }
  uint16_t* b() const { return static_cast<uint16_t*>(p); }
  float* f() const { return static_cast<float*>(p); }
};
constexpr const char* tower = "vision_tower.vision_model.";
bool visual_name(const std::string& s) {
  return s.starts_with("vision_tower.") || s.starts_with("multi_modal_projector.") ||
         s.starts_with("patch_merge_mlp.");
}
}  // namespace

struct MinimaxVisionEncoder::Impl {
  MinimaxVisionConfig c;
  cudaStream_t stream;
  CublasLtGemm gemm;
  std::map<std::string, std::unique_ptr<Buffer>> weights;
  uint64_t hash = 14695981039346656037ull;
  Buffer rgb, patches, convolution, x, norm, tmp, qkv, q, k, v, attn, hidden, scores, probs, rope,
      project1, project2, merged, result, window, work, fp32_work, cached_embeddings;
  std::unique_ptr<Fp32Gemm> patch_gemm;
  struct CachedEmbedding {
    ImageInput image;
    uint64_t used = 0;
  };
  std::vector<CachedEmbedding> cache{MinimaxVisionConfig::kEmbeddingCacheSlots};
  uint64_t cache_clock = 0, encodes = 0;
  Trace observer;

  Impl(const MinimaxVisionConfig& cfg, const std::string& checkpoint, cudaStream_t s)
      : c(cfg), stream(s) {
    namespace fs = std::filesystem;
    const fs::path dir(checkpoint);
    std::map<std::string, std::string> locations;
    if (fs::exists(dir / "model.safetensors.index.json")) {
      std::ifstream f(dir / "model.safetensors.index.json");
      const std::string text((std::istreambuf_iterator<char>(f)), {});
      const auto index = minijson::parse(text);
      for (const auto& m : index.root.at("weight_map").members())
        if (visual_name(m.key)) locations.emplace(m.key, m.value.as_string());
    } else {
      auto file = SafetensorsFile::open((dir / "model.safetensors").string());
      for (const auto* t : file->grep(""))
        if (visual_name(t->name)) locations.emplace(t->name, "model.safetensors");
    }
    std::map<std::string, std::unique_ptr<SafetensorsFile>> files;
    std::set<std::string> consumed;
    const auto fetch = [&](const std::string& name, DType type,
                           std::vector<int64_t> shape) -> const TensorInfo& {
      const auto at = locations.find(name);
      if (at == locations.end()) throw std::invalid_argument("MiniMax vision: missing " + name);
      auto& file = files[at->second];
      if (!file) file = SafetensorsFile::open((dir / at->second).string());
      const auto& t = file->at(name);
      if (t.dtype != type || t.shape != shape)
        throw std::invalid_argument("MiniMax vision: incompatible dtype/shape for " + name);
      if (!consumed.insert(name).second)
        throw std::logic_error("MiniMax vision: duplicate load " + name);
      const auto* data = static_cast<const uint8_t*>(t.data);
      for (size_t i = 0; i < t.nbytes(); ++i) hash = (hash ^ data[i]) * 1099511628211ull;
      return t;
    };
    const auto store = [&](const std::string& name, const void* data, size_t n) {
      auto b = std::make_unique<Buffer>();
      b->reserve(n);
      DGPP_CUDA_OK(cudaMemcpyAsync(b->p, data, n, cudaMemcpyHostToDevice, stream));
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      weights.emplace(name, std::move(b));
    };
    const auto load = [&](const std::string& name, std::vector<int64_t> shape,
                          DType type = DType::BF16) {
      const auto& t = fetch(name, type, std::move(shape));
      store(name, t.data, t.nbytes());
    };
    const auto linear = [&](const std::string& name, int n, int inner) {
      load(name + ".weight", {n, inner});
      load(name + ".bias", {n});
    };
    const int h = c.hidden, o = c.output;
    load(std::string(tower) + "embeddings.patch_embedding.weight", {h, 3, 2, 14, 14}, DType::F32);
    load(std::string(tower) + "pre_layrnorm.weight", {h});
    load(std::string(tower) + "pre_layrnorm.bias", {h});
    for (int l = 0; l < c.depth; ++l) {
      const auto p = std::string(tower) + "encoder.layers." + std::to_string(l) + ".";
      for (const auto* n : {"layer_norm1", "layer_norm2"}) {
        load(p + n + ".weight", {h});
        load(p + n + ".bias", {h});
      }
      // Fuse the three checkpoint projections without changing any bytes.
      for (bool bias : {false, true}) {
        const size_t count = bias ? h : size_t(h) * h;
        std::vector<uint16_t> packed(3 * count);
        int row = 0;
        for (const auto* name : {"q_proj", "k_proj", "v_proj"}) {
          const auto& t = fetch(p + "self_attn." + name + (bias ? ".bias" : ".weight"), DType::BF16,
                                bias ? std::vector<int64_t>{h} : std::vector<int64_t>{h, h});
          std::memcpy(packed.data() + row++ * count, t.data, count * 2);
        }
        store(p + "self_attn.qkv" + (bias ? ".bias" : ".weight"), packed.data(), packed.size() * 2);
      }
      linear(p + "self_attn.out_proj", h, h);
      linear(p + "mlp.fc1", c.intermediate, h);
      linear(p + "mlp.fc2", h, c.intermediate);
    }
    linear("multi_modal_projector.linear_1", o, h);
    linear("multi_modal_projector.linear_2", o, o);
    linear("patch_merge_mlp.linear_1", o, 4 * o);
    linear("patch_merge_mlp.linear_2", o, o);
    for (const auto& [name, unused] : locations)
      if (!consumed.contains(name))
        throw std::invalid_argument("MiniMax vision: unexpected tensor " + name);
    size_t loaded = 0;
    for (const auto& [name, b] : weights) loaded += b->bytes;
    if (loaded != c.weight_bytes())
      throw std::logic_error("MiniMax vision: weight memory plan mismatch");
    for (auto& [name, file] : files) file->close_mapping(true);
    // All allocations precede fabric execution; request-time malloc/free
    // can otherwise wait on a peer's spinning collective.
    const size_t n = c.kMaxSegmentTokens * 4, rows = n * h;
    rgb.reserve(pixels_for(c.kGrid) * 3 * c.kMaxFrames);
    patches.reserve(n * c.patch_in() * 4);
    convolution.reserve(rows * 4);
    for (auto* b : {&x, &norm, &tmp, &q, &k, &v, &attn}) b->reserve(rows * 2);
    qkv.reserve(3 * rows * 2);
    hidden.reserve(n * c.intermediate * 2);
    scores.reserve(c.score_elements() * 4);
    probs.reserve(c.score_elements() * 2);
    rope.reserve(n * c.rotary_dim() * 4);
    project1.reserve(n * o * 2);
    project2.reserve(n * o * 2);
    merged.reserve(size_t(c.kMaxSegmentTokens) * o * 2);
    result.reserve(size_t(c.kMaxSegmentTokens) * o * 2);
    window.reserve(size_t(c.kWindowTokens) * o * 2);
    work.reserve(c.kGemmWorkspace);
    fp32_work.reserve(c.kFp32Workspace);
    cached_embeddings.reserve(size_t(c.kEmbeddingCacheSlots) * c.kMaxSegmentTokens * o * 2);
    patch_gemm = std::make_unique<Fp32Gemm>(fp32_work.p, fp32_work.bytes, stream);
  }
  const uint16_t* w(const std::string& name) const { return weights.at(name)->b(); }
  void linear(const uint16_t* a, uint16_t* out, int rows, int n, int inner,
              const std::string& name) {
    gemm.matmul_linear_bf16(a, w(name + ".weight"), w(name + ".bias"), out, rows, n, inner, work.p,
                            work.bytes, stream);
  }
  void trace(const std::string& name, const void* data, size_t count, DType type = DType::BF16) {
    if (observer) observer(name, data, count, type);
  }
  void geometry(const ImageInput& im) {
    minimax_vision_rope_tables(rope.f(), im.width, im.height, c.rotary_dim(), c.rope_theta, stream,
                               im.frames);
    trace("rope", rope.p, size_t(im.tokens) * 4 * c.rotary_dim(), DType::F32);
  }
  void encode_one(const ImageInput& im);
};

void MinimaxVisionEncoder::Impl::encode_one(const ImageInput& im) {
  if (im.grid != c.kGrid)
    throw std::invalid_argument("MiniMax vision: a 28-pixel token grid is required");
  const int n = im.tokens * 4, h = c.hidden, d = c.head_dim(), o = c.output;
  const size_t rows = size_t(n) * h;
  DGPP_CUDA_OK(
      cudaMemcpyAsync(rgb.p, im.rgb.data(), im.rgb.size(), cudaMemcpyHostToDevice, stream));
  geometry(im);
  minimax_vision_patchify(static_cast<const uint8_t*>(rgb.p), patches.f(), im.width, im.height,
                          stream, im.frames);
  trace("patches", patches.p, size_t(n) * c.patch_in(), DType::F32);
  patch_gemm->matmul(patches.f(),
                     weights.at(std::string(tower) + "embeddings.patch_embedding.weight")->f(),
                     convolution.f(), n, h, c.patch_in(), c.patch_in());
  minimax_vision_cast(convolution.f(), x.b(), rows, stream);
  trace("patch_projection", x.p, rows);
  qwen_vision_layernorm(x.b(), w(std::string(tower) + "pre_layrnorm.weight"),
                        w(std::string(tower) + "pre_layrnorm.bias"), norm.b(), n, h, c.eps, stream);
  DGPP_CUDA_OK(cudaMemcpyAsync(x.p, norm.p, rows * 2, cudaMemcpyDeviceToDevice, stream));
  trace("pre_norm", x.p, rows);
  for (int l = 0; l < c.depth; ++l) {
    const auto p = std::string(tower) + "encoder.layers." + std::to_string(l) + ".";
    const auto t = "layer" + std::to_string(l) + ".";
    qwen_vision_layernorm(x.b(), w(p + "layer_norm1.weight"), w(p + "layer_norm1.bias"), norm.b(),
                          n, h, c.eps, stream);
    trace(t + "norm1", norm.p, rows);
    linear(norm.b(), qkv.b(), n, 3 * h, h, p + "self_attn.qkv");
    trace(t + "qkv", qkv.p, rows * 3);
    minimax_vision_split_rope(qkv.b(), q.b(), k.b(), v.b(), rope.f(),
                              rope.f() + size_t(n) * c.rotary_dim() / 2, n, h, c.heads,
                              c.rotary_dim(), stream);
    trace(t + "q", q.p, rows);
    trace(t + "k", k.p, rows);
    trace(t + "v", v.p, rows);
    for (int first = 0; first < n; first += c.kQueryTile) {
      const int tile = std::min(c.kQueryTile, n - first);
      gemm.matmul_batched_bf16(q.b() + size_t(first) * d, k.b(), scores.f(), tile, n, d, c.heads,
                               int64_t(n) * d, int64_t(n) * d, int64_t(tile) * n, work.p,
                               work.bytes, stream, n);
      qwen_vision_softmax(scores.f(), probs.b(), c.heads * tile, n, 1.0f / std::sqrt(float(d)),
                          stream);
      gemm.matmul_batched_bf16(probs.b(), v.b(), scores.f(), tile, d, n, c.heads, int64_t(tile) * n,
                               int64_t(n) * d, int64_t(tile) * d, work.p, work.bytes, stream, n,
                               true);
      qwen_vision_store_attention(scores.f(), attn.b(), tile, n, h, c.heads, first, stream);
    }
    qwen_vision_unhead(attn.b(), norm.b(), n, h, c.heads, stream);
    trace(t + "attn", norm.p, rows);
    linear(norm.b(), tmp.b(), n, h, h, p + "self_attn.out_proj");
    trace(t + "attn_proj", tmp.p, rows);
    qwen_vision_residual_add(x.b(), tmp.b(), rows, stream);
    trace(t + "attn_residual", x.p, rows);
    qwen_vision_layernorm(x.b(), w(p + "layer_norm2.weight"), w(p + "layer_norm2.bias"), norm.b(),
                          n, h, c.eps, stream);
    trace(t + "norm2", norm.p, rows);
    linear(norm.b(), hidden.b(), n, c.intermediate, h, p + "mlp.fc1");
    trace(t + "mlp_hidden", hidden.p, size_t(n) * c.intermediate);
    qwen_vision_gelu_exact(hidden.b(), size_t(n) * c.intermediate, stream);
    trace(t + "mlp_act", hidden.p, size_t(n) * c.intermediate);
    linear(hidden.b(), tmp.b(), n, h, c.intermediate, p + "mlp.fc2");
    trace(t + "mlp_proj", tmp.p, rows);
    qwen_vision_residual_add(x.b(), tmp.b(), rows, stream);
    trace("layer" + std::to_string(l), x.p, rows);
  }
  // The trained tower has no post-LayerNorm. Project every patch first;
  // four consecutive projected patches form one merger input row.
  linear(x.b(), project1.b(), n, o, h, "multi_modal_projector.linear_1");
  qwen_vision_gelu_exact(project1.b(), size_t(n) * o, stream);
  linear(project1.b(), project2.b(), n, o, o, "multi_modal_projector.linear_2");
  trace("projected_patches", project2.p, size_t(n) * o);
  linear(project2.b(), merged.b(), im.tokens, o, 4 * o, "patch_merge_mlp.linear_1");
  qwen_vision_gelu_exact(merged.b(), size_t(im.tokens) * o, stream);
  linear(merged.b(), result.b(), im.tokens, o, o, "patch_merge_mlp.linear_2");
  trace("image_rows", result.p, size_t(im.tokens) * o);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
}

MinimaxVisionEncoder::MinimaxVisionEncoder(const MinimaxVisionConfig& c, const std::string& path,
                                           cudaStream_t s)
    : impl_(std::make_unique<Impl>(c, path, s)) {}
MinimaxVisionEncoder::~MinimaxVisionEncoder() = default;
uint64_t MinimaxVisionEncoder::digest() const {
  return impl_->hash;
}
uint64_t MinimaxVisionEncoder::encode_count() const {
  return impl_->encodes;
}
const uint16_t* MinimaxVisionEncoder::encode(const ImageInput& im, Trace trace) {
  validate_image_pixels(im, true);
  impl_->observer = std::move(trace);
  try {
    impl_->encode_one(im);
    ++impl_->encodes;
  } catch (...) {
    impl_->observer = {};
    throw;
  }
  impl_->observer = {};
  return impl_->result.b();
}
const uint16_t* MinimaxVisionEncoder::stage(const std::vector<ImageInput>& images, int64_t first,
                                            int64_t end) {
  if (first < 0 || end < first || end - first > impl_->c.kWindowTokens)
    throw std::invalid_argument("MiniMax vision: staged window exceeds capacity");
  for (const auto& im : images) {
    const uint16_t* embedding = nullptr;
    for (int span = 0; span < im.spans(); ++span) {
      const int64_t offset = im.span_offset(span);
      const int64_t begin = std::max(first, offset),
                    stop = std::min(end, offset + im.span_tokens());
      if (begin >= stop) continue;
      const size_t h = impl_->c.output;
      if (!embedding) {
        auto& cache = impl_->cache;
        auto found = std::find_if(cache.begin(), cache.end(), [&](const auto& entry) {
          const auto& r = entry.image;
          // Positions affect placement in the prompt, not the encoder. Every
          // frame participates in segment attention, including future spans.
          return entry.used && r.tokens == im.tokens && r.width == im.width &&
                 r.height == im.height && r.grid == im.grid && r.frames == im.frames &&
                 r.rgb == im.rgb;
        });
        const bool hit = found != cache.end();
        if (!hit)
          found = std::min_element(cache.begin(), cache.end(),
                                   [](const auto& a, const auto& b) { return a.used < b.used; });
        auto* data = impl_->cached_embeddings.b() +
                     size_t(found - cache.begin()) * impl_->c.kMaxSegmentTokens * h;
        if (!hit) {
          // Copy host identity before changing the entry, and invalidate the
          // victim before overwriting its device bytes. All GPU storage was
          // allocated at boot; these stream-ordered copies need no malloc.
          auto identity = im;
          encode(im);
          found->used = 0;
          DGPP_CUDA_OK(cudaMemcpyAsync(data, impl_->result.b(), size_t(im.tokens) * h * 2,
                                       cudaMemcpyDeviceToDevice, impl_->stream));
          found->image = std::move(identity);
        }
        found->used = ++impl_->cache_clock;
        embedding = data;
      }
      DGPP_CUDA_OK(cudaMemcpyAsync(impl_->window.b() + (begin - first) * h,
                                   embedding + (span * im.span_tokens() + begin - offset) * h,
                                   size_t(stop - begin) * h * 2, cudaMemcpyDeviceToDevice,
                                   impl_->stream));
    }
  }
  return impl_->window.b();
}
}  // namespace dgpp
