// The Inkling 2 image prefill (models/inkling2/image_prefill.cpp) on the tiny
// release with a tiny vision tower: two images in a 96-token prompt, the
// first on a padded canvas.
//
//   inkling2_image_prefill_test --fixture DIR
//
// The fixture's text layers are a chaotic map (inkling2_decode_test.cpp), so
// two walks of different chunk shapes are not compared. Every judged
// comparison is between walks of the same shapes and is bitwise:
//   * the image rows replace the placeholder embeddings, on either slot;
//   * pixels and the content region reach the logits;
//   * a chunked walk sees its images in every advance, alone and interleaved
//     with another slot's walk;
//   * an image's rows land on exactly its span: a prefix cut at the span's
//     first position does not depend on the image, one position later it
//     does, a cut at the span's last position still takes the resumed
//     request's last row, and a cut at the span's end takes nothing;
//   * a text request after an image request reads ordinary embeddings;
//   * malformed spans are refused before a session opens.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "common/image_input.hpp"
#include "inkling2_fixture.hpp"
#include "loaders/minijson.hpp"
#include "models/inkling2/binding.hpp"
#include "models/inkling2/config.hpp"
#include "models/inkling2/forward.hpp"

namespace fs = std::filesystem;
using dgpp::ImageInput;
using dgpp::Inkling2Config;
using dgpp::Inkling2ExpectedTensor;
using dgpp::Inkling2Model;
using dgpp::Inkling2Residency;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

bool bitwise(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

double relative_l2(const std::vector<float>& got, const std::vector<float>& want) {
  double d2 = 0, w2 = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double d = static_cast<double>(got[i]) - want[i];
    d2 += d * d;
    w2 += static_cast<double>(want[i]) * want[i];
  }
  return std::sqrt(d2) / std::sqrt(w2 + 1e-30);
}

// --- the fixture -------------------------------------------------------------

constexpr int64_t kPlaceholder = 499;

// The tiny release with a tower: 64 wide, two layers, four heads of sixteen
// (four rotation frequencies an axis), a 6 x 6 position grid and a 96-wide
// projector into the text width. The placeholder sits inside the tiny
// vocabulary.
std::string image_config_json() {
  std::string s = ink2fx::tiny_config_json();
  const std::string anchor = "\"quantization_config\"";
  const size_t at = s.find(anchor);
  require(at != std::string::npos, "image_config_json: anchor not found");
  return s.insert(at,
                  "\"image_token_id\": 499,\n  \"vision_config\": {\"model_type\": \"inkling2_vision\", "
                  "\"text_hidden_size\": 256, \"temporal_patch_size\": 2, \"native_patch_size\": 20, "
                  "\"adapter_merge_size\": 2, \"projection_hidden_dim\": 96, \"projector_bias\": false, "
                  "\"hidden_size\": 64, \"num_hidden_layers\": 2, \"num_attention_heads\": 4, \"mlp_ratio\": 4.0, "
                  "\"posemb_grid_size\": 6, \"layer_norm_eps\": 1e-05},\n  ");
}

bool has(const std::string& name, const char* needle) { return name.find(needle) != std::string::npos; }

// A tower whose rows matter: LayerNorm gains near one, sublayer scales of
// about a half, queries and keys large enough for selective attention, and
// a projector that lands the image rows at the magnitude of the text's
// normed embeddings.
std::vector<uint8_t> tower_bytes(const Inkling2ExpectedTensor& e) {
  require(e.dtype == dgpp::DType::BF16, "tower fixture: " + e.name + " is not bf16");
  std::vector<uint8_t> out(e.nbytes());
  ink2fx::Rng rng(ink2fx::seed_for(e.name));
  const std::string& n = e.name;
  const bool norm = has(n, ".ln_");
  const bool weight = has(n, ".weight");
  float sigma = 0.05f;
  if (norm) sigma = weight ? 0.2f : 0.1f;
  else if (has(n, "class_embedding") || has(n, "position_embedding") || has(n, ".gamma")) sigma = 0.5f;
  else if (has(n, "patch_embedding")) sigma = 0.04f;
  else if (!weight) sigma = 0.05f;  // the projections' biases
  else if (has(n, "q_proj") || has(n, "k_proj")) sigma = 0.3f;
  else if (has(n, "v_proj") || has(n, "out_proj") || has(n, "c_fc") || has(n, "proj_out")) sigma = 0.15f;
  else if (has(n, "c_proj") || has(n, "proj_in")) sigma = 0.08f;
  const size_t count = e.numel();
  for (size_t i = 0; i < count; ++i) {
    // normal3() has a deviation of a third.
    const float v = (norm && weight ? 1.0f : 0.0f) + sigma * 3.0f * rng.normal3();
    const uint16_t bits = dgpp::float_to_bf16_bits(v);
    std::memcpy(&out[i * 2], &bits, 2);
  }
  return out;
}

void write_shard(const fs::path& path, const std::vector<Inkling2ExpectedTensor>& table) {
  std::string header = "{";
  std::vector<uint8_t> payload;
  for (size_t i = 0; i < table.size(); ++i) {
    const Inkling2ExpectedTensor& e = table[i];
    const std::vector<uint8_t> bytes = tower_bytes(e);
    header += std::string(i ? "," : "") + "\"" + e.name + "\":{\"dtype\":\"BF16\",\"shape\":[";
    for (size_t d = 0; d < e.shape.size(); ++d) header += (d ? "," : "") + std::to_string(e.shape[d]);
    header += "],\"data_offsets\":[" + std::to_string(payload.size()) + "," +
              std::to_string(payload.size() + bytes.size()) + "]}";
    payload.insert(payload.end(), bytes.begin(), bytes.end());
  }
  header += "}";
  while (header.size() % 8) header += ' ';
  const uint64_t length = header.size();
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(&length), 8);
  f.write(header.data(), static_cast<std::streamsize>(header.size()));
  f.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
  require(f.good(), "cannot write " + path.string());
}

// The shared fixture writes the text model. The tower goes to a shard of its
// own with the magnitudes above, named in the same index.
void write_image_fixture(const std::string& dir) {
  const std::string config = image_config_json();
  const std::string quant = ink2fx::tiny_quant_json();
  const auto c = dgpp::minijson::parse(config);
  const auto q = dgpp::minijson::parse(quant);
  const Inkling2Config cfg = Inkling2Config::parse(c.root, q.root);
  require(cfg.vision.present && cfg.image_token_id == kPlaceholder, "image fixture: the config did not take the tower");
  Inkling2Config text = cfg;
  text.vision = dgpp::Inkling2VisionConfig{};
  ink2fx::write_fixture(text, dir, config.c_str(), quant);

  const char* shard = "model-vision.safetensors";
  const auto table = dgpp::inkling2_expected_vision_tensors(cfg);
  write_shard(fs::path(dir) / shard, table);
  const fs::path index_path = fs::path(dir) / "model.safetensors.index.json";
  std::string text_index;
  {
    std::ifstream f(index_path, std::ios::binary);
    require(f.good(), "image fixture: the index is missing");
    text_index.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  }
  const auto parsed = dgpp::minijson::parse(text_index);
  std::string index = "{\"metadata\": {}, \"weight_map\": {";
  bool first = true;
  const auto add = [&](const std::string& name, const std::string& file) {
    index += std::string(first ? "" : ", ") + "\"" + name + "\": \"" + file + "\"";
    first = false;
  };
  for (const auto& m : parsed.root.at("weight_map").members()) {
    require(!has(m.key, "model.visual."), "image fixture: the text table already holds a tower tensor");
    add(m.key, std::string(m.value.as_string()));
  }
  for (const Inkling2ExpectedTensor& e : table) add(e.name, shard);
  index += "}}";
  ink2fx::write_text(index_path, index);
  std::printf("inkling2 image fixture written: %zu tower tensors\n", table.size());
}

// --- the prompt ----------------------------------------------------------------

ImageInput image(int64_t offset, int cells_w, int cells_h, int seed, int trim_w = 0, int trim_h = 0) {
  ImageInput im;
  im.grid = dgpp::kInklingImageGrid;
  im.offset = offset;
  im.width = cells_w * im.grid;
  im.height = cells_h * im.grid;
  im.tokens = cells_w * cells_h;
  im.rgb.resize(static_cast<size_t>(im.width) * im.height * 3);
  for (size_t i = 0; i < im.rgb.size(); ++i) im.rgb[i] = static_cast<uint8_t>((i * 17 + static_cast<size_t>(seed)) % 256);
  if (trim_w || trim_h) {
    im.valid_width = im.width - trim_w;
    im.valid_height = im.height - trim_h;
  }
  return im;
}

template <class Call>
bool refused(Call&& call) {
  try {
    call();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

int run(const std::string& dir) {
  const Inkling2Config cfg = ink2fx::with_test_loop(Inkling2Config::from_checkpoint_dir(dir));
  Inkling2Model m(cfg, dir, 128, 1024, Inkling2Residency::Resident, nullptr, 0, 1, 2);
  require(m.supports_images() && m.vision_digest() != 0, "the model does not report its tower");
  require(m.image_tokens().pad == kPlaceholder, "the model does not report the placeholder token");

  // 96 tokens. The first image, [20, 26), is 93 x 57 pixels on a 120 x 80
  // canvas. The second, [40, 44), fills its canvas.
  const std::vector<ImageInput> images{image(20, 3, 2, 11, 27, 23), image(40, 2, 2, 31)};
  std::vector<int64_t> ids(96);
  for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int64_t>((i * 13 + 9) % 400);
  for (const ImageInput& im : images) std::fill_n(ids.begin() + im.offset, im.tokens, kPlaceholder);
  const int64_t P = static_cast<int64_t>(ids.size());

  // 1. The rows replace the placeholder embeddings, on either slot.
  const Inkling2Model::Outputs text_before = m.session_prefill(0, ids);
  m.session_close(0);
  const Inkling2Model::Outputs baseline = m.session_prefill_images(0, ids, images);
  const Inkling2Model::Outputs next = m.session_step(0, 7);
  m.session_close(0);
  require(!bitwise(baseline.logits, text_before.logits), "the image rows did not replace the placeholder embeddings");
  {
    const Inkling2Model::Outputs again = m.session_prefill_images(1, ids, images);
    require(bitwise(again.logits, baseline.logits) && bitwise(m.session_step(1, 7).logits, next.logits),
            "an image prefill differs between the slots");
    m.session_close(1);
  }
  std::printf("[ OK ] image rows replace the placeholder embeddings, bitwise alike on both slots (vs text: l2 %.3g)\n",
              relative_l2(baseline.logits, text_before.logits));

  // 2. Pixels and the content region reach the logits.
  std::vector<ImageInput> changed = images;
  std::fill(changed[1].rgb.begin(), changed[1].rgb.end(), uint8_t{0});
  {
    const Inkling2Model::Outputs other = m.session_prefill_images(1, ids, changed);
    m.session_close(1);
    const double l2 = relative_l2(other.logits, baseline.logits);
    require(!bitwise(other.logits, baseline.logits) && l2 > 1e-3, "the second image's pixels were ignored or stale");
    std::vector<ImageInput> whole = images;
    whole[0].valid_width = whole[0].valid_height = 0;
    const Inkling2Model::Outputs unpadded = m.session_prefill_images(1, ids, whole);
    m.session_close(1);
    require(!bitwise(unpadded.logits, baseline.logits), "the content region was ignored");
    std::printf("[ OK ] pixels (l2 %.3g) and the content region (l2 %.3g) reach the logits\n", l2,
                relative_l2(unpadded.logits, baseline.logits));
  }

  // 3. Chunked walks: 31-token chunks put the second image in a later
  //    advance than the first.
  const auto walk = [&](int req, int64_t chunk, const std::vector<ImageInput>* with) {
    auto cursor = m.session_prefill_begin(req, ids, P, chunk, {}, nullptr, 0, with);
    while (!m.session_prefill_advance(cursor, chunk)) {
    }
    return m.session_step(req, 7).logits;
  };
  const std::vector<float> solo = walk(0, 31, &images);
  m.session_close(0);
  {
    require(bitwise(walk(1, 31, &images), solo), "a chunked image walk differs between the slots");
    m.session_close(1);
    require(!bitwise(walk(1, 31, &changed), solo), "a later advance did not see its image");
    m.session_close(1);
    require(!bitwise(walk(1, 31, nullptr), solo), "a chunked walk ignored its images");
    m.session_close(1);
    std::printf("[ .. ] chunked (31) vs one-shot image prefill, first step: relative l2 %.3g (reported)\n",
                relative_l2(solo, next.logits));
    // Two slots' walks interleaved, other pixels and another chunk on the
    // second.
    auto a = m.session_prefill_begin(0, ids, P, 31, {}, nullptr, 0, &images);
    auto b = m.session_prefill_begin(1, ids, P, 17, {}, nullptr, 0, &changed);
    while (a.next < a.end || b.next < b.end) {
      std::vector<Inkling2Model::PrefillCursor*> cursors;
      std::vector<int64_t> budgets;
      if (a.next < a.end) {
        cursors.push_back(&a);
        budgets.push_back(31);
      }
      if (b.next < b.end) {
        cursors.push_back(&b);
        budgets.push_back(17);
      }
      m.session_prefill_advance_group(cursors, budgets);
    }
    require(bitwise(m.session_step(0, 7).logits, solo), "image rows leaked between interleaved slots");
    m.session_close(0);
    m.session_close(1);
    require(m.kv_blocks_in_use() == 0, "image sessions leaked KV blocks");
    std::printf("[ OK ] chunked walks see their images in every advance, alone and interleaved\n");
  }

  // 4. The first image's rows land on exactly [20, 26). A snapshot at `cut`
  //    of a walk with the image X, resumed on the other slot with other
  //    pixels Y in its place, against the cold walks of the same shape.
  {
    std::vector<ImageInput> other = images;
    other[0] = image(20, 3, 2, 97, 27, 23);
    uint8_t* arena = nullptr;
    require(cudaMalloc(reinterpret_cast<void**>(&arena), m.session_snapshot_bytes()) == cudaSuccess, "prefix: arena");
    struct Cut {
      std::vector<float> from_x, cold_y, hot_y;
    };
    const auto cut_at = [&](int64_t cut) {
      const std::vector<int64_t> bounds{cut};
      Cut r;
      r.cold_y = m.session_prefill_images(0, ids, other, bounds, nullptr).logits;
      m.session_close(0);
      Inkling2Model::SessionSnapshotMeta meta;
      Inkling2Model::SnapshotRequest snap;
      snap.position = cut;
      snap.dst = arena;
      snap.meta = &meta;
      r.from_x = m.session_prefill_images(0, ids, images, bounds, &snap).logits;
      require(snap.taken && meta.position == cut, "prefix: no snapshot at the cut");
      m.session_close(0);
      m.session_attach(1, arena, meta);
      require(m.session_position(1) == cut, "prefix: attached at the cut");
      r.hot_y = m.session_prefill_resume_images(1, std::vector<int64_t>(ids.begin() + cut, ids.end()), other, bounds,
                                                nullptr)
                    .logits;
      m.session_close(1);
      m.session_release_snapshot(meta);
      return r;
    };
    const int64_t begin = images[0].offset, end = begin + images[0].tokens;
    Cut r = cut_at(begin);
    require(bitwise(r.hot_y, r.cold_y), "prefix: the positions before an image depend on it, or the resume lost a row");
    require(!bitwise(r.cold_y, r.from_x), "prefix: the two images do not differ");
    r = cut_at(begin + 1);
    require(!bitwise(r.hot_y, r.cold_y) && !bitwise(r.hot_y, r.from_x),
            "prefix: the image's first row is not at its first position");
    r = cut_at(end - 1);
    require(!bitwise(r.hot_y, r.from_x) && !bitwise(r.hot_y, r.cold_y),
            "prefix: the image's last row is not at its last position");
    r = cut_at(end);
    require(bitwise(r.hot_y, r.from_x), "prefix: an image row landed past the image's span");
    require(cudaFree(arena) == cudaSuccess, "prefix: arena release");
    require(m.kv_blocks_in_use() == 0, "prefix: the snapshots leaked KV blocks");
    std::printf("[ OK ] an image's rows land on exactly its span, cold and resumed inside it\n");
  }

  // 5. A text request after the image requests reads ordinary embeddings.
  {
    const Inkling2Model::Outputs t0 = m.session_prefill(0, ids);
    const Inkling2Model::Outputs t1 = m.session_prefill(1, ids);
    require(bitwise(t0.logits, text_before.logits) && bitwise(t1.logits, text_before.logits),
            "a text request read a finished request's image rows");
    m.session_close(0);
    m.session_close(1);
    std::printf("[ OK ] text requests after image requests read ordinary embeddings\n");
  }

  // 6. Malformed requests are refused before a session opens.
  {
    std::vector<int64_t> bad = ids;
    bad[static_cast<size_t>(images[0].offset)] = 7;
    require(refused([&] { (void)m.session_prefill_images(0, bad, images); }), "a span without its placeholder was admitted");
    require(refused([&] { (void)m.session_prefill_begin(0, bad, P, 31, {}, nullptr, 0, &images); }),
            "a cursor over a span without its placeholder was admitted");
    std::vector<ImageInput> overlap = images;
    overlap[1].offset = images[0].offset + 2;
    require(refused([&] { (void)m.session_prefill_images(0, ids, overlap); }), "overlapping spans were admitted");
    std::vector<ImageInput> grid = images;
    grid[1].grid = dgpp::kQwenImageGrid;
    grid[1].width = grid[1].height = 2 * dgpp::kQwenImageGrid;
    grid[1].rgb.assign(static_cast<size_t>(grid[1].width) * grid[1].height * 3, uint8_t{9});
    require(refused([&] { (void)m.session_prefill_images(0, ids, grid); }), "another tower's pixel grid was admitted");
    require(m.session_position(0) == 0 && m.kv_blocks_in_use() == 0, "a refused request left a session open");
    std::printf("[ OK ] malformed spans and foreign grids are refused before a session opens\n");
  }

  // 7. Decode continues from an image prefill, alike on both slots.
  {
    const auto pick = [](const std::vector<float>& logits) {
      return static_cast<int64_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    };
    Inkling2Model::Outputs a = m.session_prefill_images(0, ids, images);
    Inkling2Model::Outputs b = m.session_prefill_images(1, ids, images);
    for (int s = 0; s < 8; ++s) {
      const int64_t token = pick(a.logits);
      a = m.session_step(0, token);
      b = m.session_step(1, token);
      require(bitwise(a.logits, b.logits), "decode after an image prefill differs between the slots");
    }
    m.session_close(0);
    m.session_close(1);
    require(m.kv_blocks_in_use() == 0, "decode after an image prefill leaked KV blocks");
    std::printf("[ OK ] decode continues from an image prefill\n");
  }
  return 0;
}

// A release without a tower refuses images.
void run_text_only(const std::string& dir) {
  const Inkling2Config cfg = ink2fx::with_test_loop(Inkling2Config::from_checkpoint_dir(dir));
  Inkling2Model m(cfg, dir, 64, 256, Inkling2Residency::Resident, nullptr, 0, 1, 1);
  require(!m.supports_images() && m.vision_digest() == 0, "a release without a tower reports one");
  const std::vector<ImageInput> images{image(2, 1, 1, 5)};
  std::vector<int64_t> ids(8, 3);
  ids[2] = cfg.image_token_id;
  require(refused([&] { (void)m.session_prefill_images(0, ids, images); }), "a release without a tower admitted an image");
  require(m.kv_blocks_in_use() == 0, "the refused image left a session open");
  std::printf("[ OK ] a release without a tower refuses images\n");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string fixture;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--fixture" && i + 1 < argc) fixture = argv[++i];
      else throw std::runtime_error("unknown argument " + a);
    }
    require(!fixture.empty(), "usage: inkling2_image_prefill_test --fixture DIR");
    if (!fs::exists(fs::path(fixture) / "model-vision.safetensors")) write_image_fixture(fixture);
    const std::string text_only = fixture + "_text";
    if (!fs::exists(fs::path(text_only) / "config.json")) ink2fx::write_fixture(text_only);
    const int rc = run(fixture);
    run_text_only(text_only);
    if (rc == 0) std::printf("inkling2_image_prefill_test: image prefill, chunking, prefix cuts, isolation and refusals pass\n");
    return rc;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "inkling2_image_prefill_test: %s\n", e.what());
    return 1;
  }
}
