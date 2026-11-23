// The Inkling 2 vision tower (models/inkling2/vision.hpp). Two things to know
// while reading:
//
//  * Token order. Row 0 is the class row and row 1 + y * wp + x the native
//    patch at (y, x). The position rows, the rotation and the merger's
//    gather all count patches that way, on the device, from the image's
//    geometry alone, so every rank produces the same rows.
//  * The canvas is padded. A pixel right of or below the image's content is
//    the processor's padding value (-1 before normalization), whatever the
//    canvas bytes hold, so the content region is part of the input.
#include "models/inkling2/vision.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "kernels/fp32_gemm.hpp"
#include "kernels/gemm.hpp"
#include "kernels/inkling2_vision.hpp"
#include "kernels/qwen_vision.hpp"
#include "loaders/minijson.hpp"
#include "loaders/safetensors.hpp"
#include "models/inkling2/binding.hpp"

namespace dgpp {
namespace {

constexpr const char* kPrefix = "model.visual.";

struct Buffer {
  void* p = nullptr;
  size_t bytes = 0;
  ~Buffer() { cudaFree(p); }
  void reserve(size_t n) {
    if (p) throw std::logic_error("Inkling 2 vision: buffer already allocated");
    DGPP_CUDA_OK(cudaMalloc(&p, std::max<size_t>(n, 16)));
    bytes = n;
  }
  uint16_t* b() const { return static_cast<uint16_t*>(p); }
  float* f() const { return static_cast<float*>(p); }
};

// Every workspace size, from the config alone: the constructor reserves
// these and the memory plan sums them.
struct Sizes {
  size_t rgb, patches, row, qkv, values, hidden, scores, tile_out, pooled, project, result, window, cache, inv_freq,
      work, fp32_work;
  int max_rows, tile;
  explicit Sizes(const Inkling2VisionConfig& v) {
    const size_t m2 = static_cast<size_t>(v.adapter_merge_size) * v.adapter_merge_size;
    const size_t patches_max = static_cast<size_t>(kMaxImageTokens) * m2;
    const size_t H = static_cast<size_t>(v.hidden_size), O = static_cast<size_t>(v.text_hidden_size);
    const size_t heads = static_cast<size_t>(v.num_attention_heads);
    max_rows = static_cast<int>(patches_max + 1);
    tile = std::min(Inkling2VisionEncoder::kQueryTile, max_rows);
    const int grid = v.native_patch_size * v.adapter_merge_size;
    rgb = pixels_for(grid) * 3;
    patches = patches_max * 3 * static_cast<size_t>(v.native_patch_size) * v.native_patch_size * 2;
    row = static_cast<size_t>(max_rows) * H * 2;
    qkv = 3 * row;
    // The values widened to fp32 for the P @ V.
    values = 2 * row;
    hidden = static_cast<size_t>(max_rows) * static_cast<size_t>(Inkling2VisionEncoder::intermediate(v)) * 2;
    scores = heads * static_cast<size_t>(tile) * static_cast<size_t>(max_rows) * 4;
    tile_out = static_cast<size_t>(tile) * H * 4;
    pooled = static_cast<size_t>(kMaxImageTokens) * m2 * H * 2;
    project = static_cast<size_t>(kMaxImageTokens) * static_cast<size_t>(v.projection_hidden_dim) * 2;
    result = static_cast<size_t>(kMaxImageTokens) * O * 2;
    window = static_cast<size_t>(Inkling2VisionEncoder::kWindowTokens) * O * 2;
    cache = static_cast<size_t>(Inkling2VisionEncoder::kEmbeddingCacheSlots) * kMaxImageTokens * O * 2;
    inv_freq = static_cast<size_t>(v.hidden_size / v.num_attention_heads / 4) * 4;
    work = Inkling2VisionEncoder::kGemmWorkspace;
    fp32_work = Inkling2VisionEncoder::kFp32Workspace;
  }
  size_t total() const {
    // x, norm, tmp, q, k, attn are six row buffers.
    return rgb + patches + 6 * row + qkv + values + hidden + scores + tile_out + pooled + project + result + window +
           cache + inv_freq + work + fp32_work;
  }
};

}  // namespace

void Inkling2VisionEncoder::validate(const Inkling2VisionConfig& v) {
  const auto refuse = [](const char* field, const char* why) {
    throw std::invalid_argument(std::string("Inkling 2 vision: vision_config.") + field + " " + why);
  };
  if (!v.present) throw std::invalid_argument("Inkling 2 vision: the checkpoint has no vision_config");
  if (v.native_patch_size * v.adapter_merge_size != kInklingImageGrid)
    refuse("native_patch_size", "times adapter_merge_size must be the 40-pixel visual token");
  if (v.adapter_merge_size < 1 || v.adapter_merge_size > 4) refuse("adapter_merge_size", "must be in [1, 4]");
  if (v.hidden_size % v.num_attention_heads != 0) refuse("hidden_size", "must be a multiple of num_attention_heads");
  const int dim = v.hidden_size / v.num_attention_heads;
  if (dim % 4 != 0) refuse("hidden_size", "needs a head width that is a multiple of four (the axial rotation)");
  if (intermediate(v) <= 0) refuse("mlp_ratio", "gives no MLP width");
  if (v.projector_bias) refuse("projector_bias", "is not implemented");
  if (v.posemb_grid_size < 1) refuse("posemb_grid_size", "must be positive");
}

size_t Inkling2VisionEncoder::weight_bytes(const Inkling2Config& cfg) {
  size_t bytes = 0;
  for (const auto& t : inkling2_expected_vision_tensors(cfg)) bytes += t.nbytes();
  return bytes;
}

size_t Inkling2VisionEncoder::workspace_bytes(const Inkling2VisionConfig& v) { return Sizes(v).total(); }

struct Inkling2VisionEncoder::Impl {
  Inkling2VisionConfig c;
  cudaStream_t stream;
  CublasLtGemm gemm;
  std::map<std::string, std::unique_ptr<Buffer>> weights;
  uint64_t hash = 14695981039346656037ull;
  struct Layer {
    const uint16_t *ln1_w, *ln1_b, *ln2_w, *ln2_b, *ls1, *ls2;
    const uint16_t *qkv_w, *qkv_b, *out_w, *out_b, *fc_w, *fc_b, *proj_w, *proj_b;
  };
  std::vector<Layer> layers;
  const uint16_t *patch_w = nullptr, *position = nullptr, *class_embedding = nullptr, *class_position = nullptr;
  const uint16_t *pre_w = nullptr, *pre_b = nullptr, *proj_in = nullptr, *proj_out = nullptr;
  Sizes sizes;
  Buffer rgb, patches, x, norm, tmp, qkv, q, k, vt, attn, hidden, scores, tile_out, pooled, project, result, window,
      cached_embeddings, inv_freq, work, fp32_work;
  // The fp32 P @ V: the released attention keeps its probabilities wide.
  std::unique_ptr<Fp32Gemm> pv_gemm;
  struct CachedEmbedding {
    ImageInput image;
    uint64_t used = 0;
  };
  std::vector<CachedEmbedding> cache{static_cast<size_t>(kEmbeddingCacheSlots)};
  uint64_t cache_clock = 0, encodes = 0;
  Trace observer;

  Impl(const Inkling2Config& cfg, const std::string& checkpoint, cudaStream_t s)
      : c(cfg.vision), stream(s), sizes(cfg.vision) {
    validate(c);
    namespace fs = std::filesystem;
    const fs::path dir(checkpoint);
    // Where each visual tensor lives: the index of a sharded release, or the
    // single file of a small one.
    std::map<std::string, std::string> locations;
    if (fs::exists(dir / "model.safetensors.index.json")) {
      std::ifstream f(dir / "model.safetensors.index.json");
      if (!f) throw std::runtime_error("Inkling 2 vision: cannot read the checkpoint index");
      const std::string text((std::istreambuf_iterator<char>(f)), {});
      const auto index = minijson::parse(text);
      for (const auto& m : index.root.at("weight_map").members())
        if (m.key.starts_with(kPrefix)) locations.emplace(m.key, std::string(m.value.as_string()));
    } else if (fs::exists(dir / "model.safetensors")) {
      auto file = SafetensorsFile::open((dir / "model.safetensors").string());
      for (const auto* t : file->grep(kPrefix)) locations.emplace(t->name, "model.safetensors");
    } else {
      throw std::runtime_error("Inkling 2 vision: no safetensors checkpoint at " + dir.string());
    }
    std::map<std::string, std::unique_ptr<SafetensorsFile>> files;
    std::set<std::string> consumed;
    std::map<std::string, const TensorInfo*> found;
    // The binding table is the contract: names, dtypes and shapes.
    for (const auto& e : inkling2_expected_vision_tensors(cfg)) {
      const auto at = locations.find(e.name);
      if (at == locations.end()) throw std::invalid_argument("Inkling 2 vision: missing " + e.name);
      auto& file = files[at->second];
      if (!file) file = SafetensorsFile::open((dir / at->second).string());
      const TensorInfo& t = file->at(e.name);
      if (t.dtype != e.dtype || t.shape != e.shape)
        throw std::invalid_argument("Inkling 2 vision: incompatible dtype or shape for " + e.name);
      const auto* data = static_cast<const uint8_t*>(t.data);
      for (size_t i = 0; i < t.nbytes(); ++i) hash = (hash ^ data[i]) * 1099511628211ull;
      consumed.insert(e.name);
      found.emplace(e.name.substr(std::strlen(kPrefix)), &t);
    }
    for (const auto& [name, unused] : locations)
      if (!consumed.contains(name)) throw std::invalid_argument("Inkling 2 vision: unexpected tensor " + name);

    size_t loaded = 0;
    const auto store = [&](const std::string& name, const void* data, size_t n) -> const uint16_t* {
      auto b = std::make_unique<Buffer>();
      b->reserve(n);
      DGPP_CUDA_OK(cudaMemcpyAsync(b->p, data, n, cudaMemcpyHostToDevice, stream));
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      loaded += n;
      const uint16_t* p = b->b();
      weights.emplace(name, std::move(b));
      return p;
    };
    const auto load = [&](const std::string& name) -> const uint16_t* {
      const TensorInfo& t = *found.at(name);
      return store(name, t.data, t.nbytes());
    };
    const std::string b = "backbone.";
    class_embedding = load(b + "embeddings.class_embedding");
    class_position = load(b + "embeddings.cls_position_embedding");
    patch_w = load(b + "embeddings.patch_embedding.weight");
    position = load(b + "embeddings.position_embedding");
    pre_w = load(b + "ln_pre.weight");
    pre_b = load(b + "ln_pre.bias");
    const size_t H = static_cast<size_t>(c.hidden_size);
    layers.resize(static_cast<size_t>(c.num_hidden_layers));
    for (int l = 0; l < c.num_hidden_layers; ++l) {
      const std::string p = b + "layers." + std::to_string(l) + ".";
      Layer& L = layers[static_cast<size_t>(l)];
      L.ln1_w = load(p + "ln_1.weight");
      L.ln1_b = load(p + "ln_1.bias");
      L.ln2_w = load(p + "ln_2.weight");
      L.ln2_b = load(p + "ln_2.bias");
      L.ls1 = load(p + "ls_1.gamma");
      L.ls2 = load(p + "ls_2.gamma");
      // The three checkpoint projections as one, without changing a byte.
      for (const bool bias : {false, true}) {
        const size_t count = bias ? H : H * H;
        std::vector<uint16_t> packed(3 * count);
        size_t row = 0;
        for (const char* name : {"q_proj", "k_proj", "v_proj"}) {
          const TensorInfo& t = *found.at(p + "self_attn." + name + (bias ? ".bias" : ".weight"));
          std::memcpy(packed.data() + row++ * count, t.data, count * 2);
        }
        const uint16_t* at = store(p + "self_attn.qkv" + (bias ? ".bias" : ".weight"), packed.data(), packed.size() * 2);
        (bias ? L.qkv_b : L.qkv_w) = at;
      }
      L.out_w = load(p + "self_attn.out_proj.weight");
      L.out_b = load(p + "self_attn.out_proj.bias");
      L.fc_w = load(p + "mlp.c_fc.weight");
      L.fc_b = load(p + "mlp.c_fc.bias");
      L.proj_w = load(p + "mlp.c_proj.weight");
      L.proj_b = load(p + "mlp.c_proj.bias");
    }
    proj_in = load("proj_in.weight");
    proj_out = load("proj_out.weight");
    if (loaded != weight_bytes(cfg)) throw std::logic_error("Inkling 2 vision: weight memory plan mismatch");
    // Unmapped, the page cache kept: the text loader reads the same shard and owns that policy.
    for (auto& [name, file] : files) file->close_mapping(false);

    // Every allocation precedes fabric execution: a request-time malloc can
    // wait on a peer's spinning collective.
    rgb.reserve(sizes.rgb);
    patches.reserve(sizes.patches);
    for (auto* buf : {&x, &norm, &tmp, &q, &k, &attn}) buf->reserve(sizes.row);
    qkv.reserve(sizes.qkv);
    vt.reserve(sizes.values);
    hidden.reserve(sizes.hidden);
    scores.reserve(sizes.scores);
    tile_out.reserve(sizes.tile_out);
    pooled.reserve(sizes.pooled);
    project.reserve(sizes.project);
    result.reserve(sizes.result);
    window.reserve(sizes.window);
    cached_embeddings.reserve(sizes.cache);
    inv_freq.reserve(sizes.inv_freq);
    work.reserve(sizes.work);
    fp32_work.reserve(sizes.fp32_work);
    size_t reserved = 0;
    for (const Buffer* buf : {&rgb, &patches, &x, &norm, &tmp, &q, &k, &vt, &attn, &qkv, &hidden, &scores, &tile_out,
                              &pooled, &project, &result, &window, &cached_embeddings, &inv_freq, &work, &fp32_work})
      reserved += buf->bytes;
    if (reserved != sizes.total()) throw std::logic_error("Inkling 2 vision: workspace memory plan mismatch");
    pv_gemm = std::make_unique<Fp32Gemm>(fp32_work.p, fp32_work.bytes, stream);
    // The rotation's frequencies, rounded from float64 as the reference's.
    const int dim = c.hidden_size / c.num_attention_heads;
    std::vector<float> freq(static_cast<size_t>(dim / 4));
    for (size_t i = 0; i < freq.size(); ++i)
      freq[i] = static_cast<float>(std::pow(10000.0, -static_cast<double>(4 * i) / static_cast<double>(dim)));
    DGPP_CUDA_OK(cudaMemcpyAsync(inv_freq.p, freq.data(), freq.size() * 4, cudaMemcpyHostToDevice, stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  }

  void linear(const uint16_t* a, const uint16_t* w, const uint16_t* bias, uint16_t* out, int rows, int n, int inner) {
    gemm.matmul_linear_bf16(a, w, bias, out, rows, n, inner, work.p, work.bytes, stream);
  }
  void trace(const std::string& name, const void* data, size_t count, DType type = DType::BF16) {
    if (observer) observer(name, data, count, type);
  }
  void encode_one(const ImageInput& im);
  const uint16_t* cached(const ImageInput& im);
};

void Inkling2VisionEncoder::Impl::encode_one(const ImageInput& im) {
  const int patch = c.native_patch_size, merge = c.adapter_merge_size, grid = patch * merge;
  if (im.grid != grid || im.frames != 1 || im.width % grid || im.height % grid ||
      im.tokens != (im.width / grid) * (im.height / grid))
    throw std::invalid_argument("Inkling 2 vision: the image's grid and token count disagree");
  const int hp = im.height / patch, wp = im.width / patch;
  const int np = hp * wp, n = np + 1;
  if (n > sizes.max_rows) throw std::invalid_argument("Inkling 2 vision: the image exceeds the tower's workspace");
  const int H = c.hidden_size, heads = c.num_attention_heads, dim = H / heads, I = intermediate(c);
  const int features = 3 * patch * patch;
  const size_t rows = static_cast<size_t>(n) * H;
  DGPP_CUDA_OK(cudaMemcpyAsync(rgb.p, im.rgb.data(), im.rgb.size(), cudaMemcpyHostToDevice, stream));
  ink2_vision_patches(static_cast<const uint8_t*>(rgb.p), patches.b(), im.width, im.height, im.content_width(),
                      im.content_height(), patch, stream);
  trace("patches", patches.p, static_cast<size_t>(np) * features);
  // The patch embedding has no bias.
  linear(patches.b(), patch_w, nullptr, tmp.b(), np, H, features);
  trace("patch_embed", tmp.p, static_cast<size_t>(np) * H);
  ink2_vision_embed_rows(tmp.b(), position, class_embedding, class_position, norm.b(), hp, wp, c.posemb_grid_size,
                         H, stream);
  trace("embed", norm.p, rows);
  qwen_vision_layernorm(norm.b(), pre_w, pre_b, x.b(), n, H, c.layer_norm_eps, stream);
  trace("ln_pre", x.p, rows);
  const int tile_rows = std::min(sizes.tile, n);
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(dim)));
  for (int l = 0; l < c.num_hidden_layers; ++l) {
    const Layer& L = layers[static_cast<size_t>(l)];
    const std::string t = "layer" + std::to_string(l) + ".";
    qwen_vision_layernorm(x.b(), L.ln1_w, L.ln1_b, norm.b(), n, H, c.layer_norm_eps, stream);
    linear(norm.b(), L.qkv_w, L.qkv_b, qkv.b(), n, 3 * H, H);
    trace(t + "qkv", qkv.p, rows * 3);
    ink2_vision_split_rope(qkv.b(), q.b(), k.b(), vt.f(), inv_freq.f(), n, H, heads, wp, stream);
    trace(t + "q", q.p, rows);
    trace(t + "k", k.p, rows);
    // Full attention over the image, a query tile at a time: the scores of
    // every head in one batched GEMM (the untiled plan keeps the tile from
    // changing the key reduction), a wide softmax in place, then each head's
    // P @ V in fp32. Only the output is rounded to bf16.
    for (int first = 0; first < n; first += tile_rows) {
      const int tile = std::min(tile_rows, n - first);
      gemm.matmul_batched_bf16(q.b() + static_cast<int64_t>(first) * dim, k.b(), scores.f(), tile, n, dim, heads,
                               static_cast<int64_t>(n) * dim, static_cast<int64_t>(n) * dim,
                               static_cast<int64_t>(tile) * n, work.p, work.bytes, stream, n);
      ink2_vision_softmax(scores.f(), heads * tile, n, scale, stream);
      for (int h = 0; h < heads; ++h)
        pv_gemm->matmul(scores.f() + static_cast<int64_t>(h) * tile * n, vt.f() + static_cast<int64_t>(h) * dim * n,
                        tile_out.f() + static_cast<int64_t>(h) * tile * dim, tile, dim, n, n);
      qwen_vision_store_attention(tile_out.f(), attn.b(), tile, n, H, heads, first, stream);
    }
    qwen_vision_unhead(attn.b(), norm.b(), n, H, heads, stream);
    trace(t + "attn", norm.p, rows);
    linear(norm.b(), L.out_w, L.out_b, tmp.b(), n, H, H);
    ink2_vision_scaled_residual(x.b(), tmp.b(), L.ls1, n, H, stream);
    trace(t + "attn_residual", x.p, rows);
    qwen_vision_layernorm(x.b(), L.ln2_w, L.ln2_b, norm.b(), n, H, c.layer_norm_eps, stream);
    linear(norm.b(), L.fc_w, L.fc_b, hidden.b(), n, I, H);
    qwen_vision_gelu_exact(hidden.b(), static_cast<int64_t>(n) * I, stream);
    linear(hidden.b(), L.proj_w, L.proj_b, tmp.b(), n, H, I);
    ink2_vision_scaled_residual(x.b(), tmp.b(), L.ls2, n, H, stream);
    trace("layer" + std::to_string(l), x.p, rows);
  }
  // No norm follows the last layer. A visual token is its patches' rows
  // side by side through the projector.
  const int merged = merge * merge * H;
  ink2_vision_pool(x.b(), pooled.b(), hp, wp, merge, H, stream);
  trace("pooled", pooled.p, static_cast<size_t>(im.tokens) * merged);
  linear(pooled.b(), proj_in, nullptr, project.b(), im.tokens, c.projection_hidden_dim, merged);
  qwen_vision_gelu_exact(project.b(), static_cast<int64_t>(im.tokens) * c.projection_hidden_dim, stream);
  linear(project.b(), proj_out, nullptr, result.b(), im.tokens, c.text_hidden_size, c.projection_hidden_dim);
  trace("image_rows", result.p, static_cast<size_t>(im.tokens) * c.text_hidden_size);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  ++encodes;
}

// The image's whole embedding in the cache, encoding it on a miss. Positions
// place an image in the prompt and do not enter the tower, so the key is the
// pixels and their geometry.
const uint16_t* Inkling2VisionEncoder::Impl::cached(const ImageInput& im) {
  const size_t O = static_cast<size_t>(c.text_hidden_size);
  auto found = std::find_if(cache.begin(), cache.end(), [&](const CachedEmbedding& entry) {
    const ImageInput& r = entry.image;
    return entry.used && r.tokens == im.tokens && r.width == im.width && r.height == im.height && r.grid == im.grid &&
           r.valid_width == im.valid_width && r.valid_height == im.valid_height && r.rgb == im.rgb;
  });
  const bool hit = found != cache.end();
  if (!hit)
    found = std::min_element(cache.begin(), cache.end(),
                             [](const CachedEmbedding& a, const CachedEmbedding& b) { return a.used < b.used; });
  uint16_t* data = cached_embeddings.b() + static_cast<size_t>(found - cache.begin()) * kMaxImageTokens * O;
  if (!hit) {
    // Copy the host identity first and invalidate the victim before its
    // device bytes change. The storage was allocated at construction.
    ImageInput identity = im;
    validate_image_pixels(im);
    encode_one(im);
    found->used = 0;
    DGPP_CUDA_OK(cudaMemcpyAsync(data, result.b(), static_cast<size_t>(im.tokens) * O * 2, cudaMemcpyDeviceToDevice,
                                 stream));
    found->image = std::move(identity);
  }
  found->used = ++cache_clock;
  return data;
}

Inkling2VisionEncoder::Inkling2VisionEncoder(const Inkling2Config& cfg, const std::string& checkpoint_dir,
                                             cudaStream_t stream)
    : impl_(std::make_unique<Impl>(cfg, checkpoint_dir, stream)) {}
Inkling2VisionEncoder::~Inkling2VisionEncoder() = default;

uint64_t Inkling2VisionEncoder::digest() const { return impl_->hash; }
uint64_t Inkling2VisionEncoder::encode_count() const { return impl_->encodes; }

const uint16_t* Inkling2VisionEncoder::encode(const ImageInput& image, Trace trace) {
  validate_image_pixels(image);
  impl_->observer = std::move(trace);
  try {
    impl_->encode_one(image);
  } catch (...) {
    impl_->observer = {};
    throw;
  }
  impl_->observer = {};
  return impl_->result.b();
}

const uint16_t* Inkling2VisionEncoder::stage(const std::vector<ImageInput>& images, int64_t first, int64_t end) {
  if (first < 0 || end < first || end - first > kWindowTokens)
    throw std::invalid_argument("Inkling 2 vision: staged window exceeds capacity");
  const size_t O = static_cast<size_t>(impl_->c.text_hidden_size);
  for (const ImageInput& im : images) {
    const int64_t begin = std::max(first, im.offset), stop = std::min(end, im.offset + im.tokens);
    if (begin >= stop) continue;
    const uint16_t* embedding = impl_->cached(im);
    DGPP_CUDA_OK(cudaMemcpyAsync(impl_->window.b() + static_cast<size_t>(begin - first) * O,
                                 embedding + static_cast<size_t>(begin - im.offset) * O,
                                 static_cast<size_t>(stop - begin) * O * 2, cudaMemcpyDeviceToDevice,
                                 impl_->stream));
  }
  return impl_->window.b();
}

}  // namespace dgpp
