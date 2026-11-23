// The Inkling 2 vision tower against its numpy reference
// (tools/inkling2_vision_reference.py), stage by stage, on the tiny tower
// that tool writes or on a release.
//
//   inkling2_vision_test [--foreign] CHECKPOINT_DIR DUMP [DUMP ...]
//
// Each dump holds one image (its canvas, its content region) and the
// reference's stages. Judged per dump:
//   * the pixel patches bitwise: the pixel map is fp32 operation by
//     operation on both sides, and the padding rule lives there;
//   * every later stage within the budget (relative l2 3 %, 0.5 % of the
//     elements beyond 128 bf16 ulps outside the cancellation floor): the
//     engine's GEMMs accumulate in fp32, the reference in float64, and the
//     tiny tower's random layers about double a rounding difference each
//     (0.6 % after its first layer, 1.1 % after its second on the smallest
//     image, where a wrong operand moves a stage by tens of percent);
//   * stage() returns encode()'s rows at the image's prompt positions, whole
//     and windowed, and encodes an image once;
//   * the canvas bytes outside the content region do not reach the rows.
// Once per run: the embedding cache keys on pixels and geometry, not on the
// prompt position, and evicts the least recently used image.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/image_input.hpp"
#include "inkling2_reference_compare.hpp"
#include "loaders/minijson.hpp"
#include "models/inkling2/config.hpp"
#include "models/inkling2/vision.hpp"

namespace fs = std::filesystem;
using dgpp::ImageInput;
using dgpp::Inkling2Config;
using dgpp::Inkling2VisionEncoder;
using ink2ref::require;

namespace {

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  require(f.good(), "cannot open " + path);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// A release carries the whole config. The tiny tower carries its
// vision_config.json alone.
Inkling2Config config_of(const std::string& dir) {
  if (fs::exists(fs::path(dir) / "hf_quant_config.json")) return Inkling2Config::from_checkpoint_dir(dir);
  const std::string text = read_file(dir + "/vision_config.json");
  const auto parsed = dgpp::minijson::parse(text);
  const dgpp::minijson::Value& v = parsed.root;
  const auto integer = [&](const char* key) { return static_cast<int>(v.at(key).as_int()); };
  Inkling2Config cfg;
  dgpp::Inkling2VisionConfig& o = cfg.vision;
  o.present = true;
  o.text_hidden_size = integer("text_hidden_size");
  o.hidden_size = integer("hidden_size");
  o.num_hidden_layers = integer("num_hidden_layers");
  o.num_attention_heads = integer("num_attention_heads");
  o.native_patch_size = integer("native_patch_size");
  o.temporal_patch_size = integer("temporal_patch_size");
  o.adapter_merge_size = integer("adapter_merge_size");
  o.projection_hidden_dim = integer("projection_hidden_dim");
  o.posemb_grid_size = integer("posemb_grid_size");
  o.mlp_ratio = v.at("mlp_ratio").as_double(4.0);
  o.layer_norm_eps = static_cast<float>(v.at("layer_norm_eps").as_double(1e-5));
  cfg.hidden_size = o.text_hidden_size;
  return cfg;
}

ImageInput image_of(const ink2ref::Dump& d) {
  const ink2ref::Dump::Tensor& g = d.tensor("geometry");
  require(g.dtype == "I32" && g.nbytes == 6 * sizeof(int32_t), "dump: geometry");
  int32_t v[6];
  std::memcpy(v, g.data, sizeof(v));
  const ink2ref::Dump::Tensor& rgb = d.tensor("rgb");
  ImageInput im;
  im.width = v[0];
  im.height = v[1];
  im.valid_width = v[2];
  im.valid_height = v[3];
  im.grid = v[4];
  im.tokens = v[5];
  im.rgb.assign(rgb.data, rgb.data + rgb.nbytes);
  dgpp::validate_image_pixels(im);
  return im;
}

std::vector<uint16_t> host_rows(const uint16_t* device, size_t count) {
  std::vector<uint16_t> out(count);
  DGPP_CUDA_OK(cudaMemcpy(out.data(), device, count * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  return out;
}

// A deterministic image of `tokens_w` x `tokens_h` cells.
ImageInput synthetic(int tokens_w, int tokens_h, int seed) {
  ImageInput im;
  im.grid = dgpp::kInklingImageGrid;
  im.width = tokens_w * im.grid;
  im.height = tokens_h * im.grid;
  im.tokens = tokens_w * tokens_h;
  im.rgb.resize(static_cast<size_t>(im.width) * im.height * 3);
  for (size_t i = 0; i < im.rgb.size(); ++i) im.rgb[i] = static_cast<uint8_t>((i * 17 + static_cast<size_t>(seed)) % 256);
  return im;
}

// The budget of the stages past the pixel patches. The fixture's is set
// against the numpy reference of the same rounding points. A release
// compared with another implementation's rows (transformers: its own
// rounding points, FP32 pixels) is held to the distance rounding alone
// opens there: transformers' BF16 tower sits 2.0 to 2.9 % from its FP32
// tower on the release, and so does this one.
struct Budget {
  double l2 = 3e-2;
  double hard = 0.005;
};
Budget g_budget;

int check_dump(Inkling2VisionEncoder& encoder, const Inkling2Config& cfg, const std::string& path) {
  const ink2ref::Dump d = ink2ref::Dump::load(path);
  require(d.activations == "vision", path + ": not a vision dump");
  require(d.hidden == cfg.vision.text_hidden_size, path + ": written for another text width");
  ImageInput im = image_of(d);
  const size_t O = static_cast<size_t>(cfg.vision.text_hidden_size);
  const size_t out_count = static_cast<size_t>(im.tokens) * O;
  std::printf("%s: %d x %d canvas, content %d x %d, %d tokens\n", path.c_str(), im.width, im.height,
              im.content_width(), im.content_height(), im.tokens);

  std::map<std::string, std::vector<uint16_t>> got;
  const uint64_t before = encoder.encode_count();
  const uint16_t* rows = encoder.encode(im, [&](const std::string& name, const void* data, size_t count, dgpp::DType type) {
    if (type != dgpp::DType::BF16 || !d.tensors.count(name)) return;
    // The trace runs between launches: the copy below waits for the stream.
    got[name] = host_rows(static_cast<const uint16_t*>(data), count);
  });
  const std::vector<uint16_t> direct = host_rows(rows, out_count);
  require(encoder.encode_count() == before + 1, "encode() must run the tower once");

  int failures = 0;
  for (const auto& [name, ref] : d.tensors) {
    if (ref.dtype != "BF16") continue;
    const auto at = got.find(name);
    if (at == got.end()) {
      std::printf("[FAIL] %-22s the engine did not trace this stage\n", name.c_str());
      ++failures;
      continue;
    }
    const size_t count = ref.nbytes / 2;
    if (at->second.size() != count) {
      std::printf("[FAIL] %-22s %zu values, the reference has %zu\n", name.c_str(), at->second.size(), count);
      ++failures;
      continue;
    }
    std::vector<uint16_t> expected(count);
    std::memcpy(expected.data(), ref.data, ref.nbytes);
    const ink2ref::Stats s = ink2ref::compare_bf16(at->second.data(), expected.data(), count);
    bool ok;
    if (name == "patches") {
      ok = at->second == expected;
    } else {
      ok = s.l2 <= g_budget.l2 && static_cast<double>(s.hard) <= g_budget.hard * static_cast<double>(s.total);
    }
    std::printf("[%s] %-22s relative l2 %.3g, beyond 16 / 128 ulps %ld / %ld of %ld%s\n", ok ? " OK " : "FAIL",
                name.c_str(), s.l2, s.soft, s.hard, s.total, name == "patches" ? " (bitwise)" : "");
    if (!ok) ++failures;
  }
  require(got.count("image_rows") && got["image_rows"] == direct, "the traced rows are not the returned rows");

  // stage(): the image somewhere in a prompt. One tower run serves every
  // window, and a window's rows sit at their positions inside it.
  im.offset = 5;
  const uint64_t staged_from = encoder.encode_count();
  const int64_t end = im.offset + im.tokens;
  require(host_rows(encoder.stage({im}, im.offset, end), out_count) == direct, "stage(): the whole image differs from encode()");
  {
    // From the prompt's start to past the image, as far as a window reaches.
    const int64_t stop = std::min<int64_t>(end + 3, Inkling2VisionEncoder::kWindowTokens);
    const std::vector<uint16_t> window = host_rows(encoder.stage({im}, 0, stop), static_cast<size_t>(stop) * O);
    const size_t inside = static_cast<size_t>(std::min<int64_t>(end, stop) - im.offset) * O;
    require(std::equal(direct.begin(), direct.begin() + static_cast<std::ptrdiff_t>(inside),
                       window.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(im.offset) * O)),
            "stage(): rows are not at the image's offset inside the window");
  }
  if (im.tokens > 1) {
    const std::vector<uint16_t> tail = host_rows(encoder.stage({im}, im.offset + 1, end), out_count - O);
    require(std::equal(tail.begin(), tail.end(), direct.begin() + static_cast<std::ptrdiff_t>(O)),
            "stage(): a window that starts inside the image is misaligned");
  }
  require(encoder.encode_count() == staged_from + 1, "stage(): one image must encode once across its windows");

  // The padding is not content. Other bytes there leave the rows alone, and
  // the same bytes read as content do not.
  if (im.valid_width > 0) {
    ImageInput dirty = im;
    for (int y = 0; y < im.height; ++y)
      for (int x = 0; x < im.width; ++x)
        if (x >= im.valid_width || y >= im.valid_height)
          for (int c = 0; c < 3; ++c)
            dirty.rgb[(static_cast<size_t>(y) * im.width + x) * 3 + c] = static_cast<uint8_t>(200 + (x + y + c) % 50);
    require(host_rows(encoder.encode(dirty), out_count) == direct, "canvas bytes outside the content region reached the rows");
    dirty.valid_width = dirty.valid_height = 0;
    require(host_rows(encoder.encode(dirty), out_count) != direct, "the content region was ignored");
    std::printf("[ OK ] the padding is the processor's value whatever the canvas holds\n");
  }
  std::printf("[ OK ] stage() serves encode()'s rows at the prompt positions, one tower run per image\n");
  return failures;
}

void check_cache(Inkling2VisionEncoder& encoder, const Inkling2Config& cfg) {
  const size_t O = static_cast<size_t>(cfg.vision.text_hidden_size);
  const auto read = [&](const ImageInput& im, bool fresh = false) {
    const uint16_t* data = fresh ? encoder.encode(im) : encoder.stage({im}, im.offset, im.offset + im.tokens);
    return host_rows(data, static_cast<size_t>(im.tokens) * O);
  };
  std::vector<ImageInput> images;
  std::vector<std::vector<uint16_t>> expected;
  for (int i = 0; i < Inkling2VisionEncoder::kEmbeddingCacheSlots; ++i) {
    images.push_back(synthetic(2, 1, 1000 + i * 23));
    expected.push_back(read(images.back()));
  }
  const uint64_t filled = encoder.encode_count();
  for (size_t i = 0; i < images.size(); ++i) require(read(images[i]) == expected[i], "cache: an interleaved image changed its rows");
  ImageInput moved = images.front();
  moved.offset = 73;
  require(read(moved) == expected.front() && encoder.encode_count() == filled,
          "cache: identical pixels encoded again after interleaving or a move in the prompt");
  (void)read(synthetic(2, 1, 241));
  require(encoder.encode_count() == filled + 1, "cache: new pixels did not encode");
  require(read(images[1]) == expected[1] && encoder.encode_count() == filled + 2,
          "cache: the evicted image was stale or did not encode again");
  // The same bytes on another canvas shape, and with a content region.
  ImageInput tall = images.front();
  std::swap(tall.width, tall.height);
  require(read(tall) == read(tall, true) && read(tall) != expected.front(), "cache: the canvas shape is not part of the key");
  ImageInput region = images.front();
  region.valid_width = region.width - 7;
  region.valid_height = region.height - 3;
  require(read(region) == read(region, true) && read(region) != expected.front(),
          "cache: the content region is not part of the key");
  std::printf("[ OK ] the embedding cache keys on pixels, canvas and content region, and evicts the oldest\n");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::vector<std::string> args(argv + 1, argv + argc);
    if (!args.empty() && args.front() == "--foreign") {
      // Dumps another implementation wrote from a release.
      g_budget = Budget{5e-2, 0.03};
      args.erase(args.begin());
    }
    if (args.size() < 2) {
      std::fprintf(stderr, "usage: inkling2_vision_test [--foreign] CHECKPOINT_DIR DUMP [DUMP ...]\n");
      return 2;
    }
    const std::string dir = args[0];
    const Inkling2Config cfg = config_of(dir);
    require(cfg.vision.present, "the checkpoint has no vision_config");
    Inkling2VisionEncoder encoder(cfg, dir, nullptr);
    require(encoder.digest() != 0, "the tower has no weight digest");
    int failures = 0;
    for (size_t i = 1; i < args.size(); ++i) failures += check_dump(encoder, cfg, args[i]);
    check_cache(encoder, cfg);
    if (failures) {
      std::printf("inkling2_vision_test: %d stage(s) outside the budget\n", failures);
      return 1;
    }
    std::printf("inkling2_vision_test: the tower agrees with the reference\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "inkling2_vision_test: %s\n", e.what());
    return 1;
  }
}
