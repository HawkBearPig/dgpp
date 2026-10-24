#include "serve/image_inputs.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <thread>

#include "common/base64.hpp"
#include "common/test.hpp"
#include "models/glm/vision_config.hpp"
#include "models/qwen/vision_config.hpp"

namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
template <class F>
void rejects(F f) {
  bool threw = false;
  try {
    f();
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "invalid image accepted");
}
}  // namespace
DGPP_TEST(image_base64_roundtrip_and_bounds) {
  std::string bytes;
  for (int i = 0; i < 256; ++i) bytes += static_cast<char>(i);
  for (size_t n = 1; n <= bytes.size(); ++n) {
    const auto s = bytes.substr(0, n);
    require(dgpp::decode_base64(dgpp::encode_base64(s), n) == s, "base64 roundtrip");
  }
  for (auto s : {"", "A", "====", "A===", "A?==", "AB==", "AAB=", "AA==AAAA", "AA=A"})
    rejects([&] { dgpp::decode_base64(s, 100); });
  rejects([] { dgpp::decode_base64("AAAA", 2); });
}
DGPP_TEST(image_glm_resize_padding_aspect_and_budget) {
  std::vector<uint8_t> rgb(113 * 117 * 3, 123);
  auto im = dgpp::serve::resize_glm_image(rgb.data(), 113, 117, 1024);
  require(im.width == 140 && im.height == 140 && im.tokens == 25, "aligned canvas");
  require(im.rgb[0] == 123 && im.rgb[(116 * 140 + 112) * 3] == 123, "unchanged content");
  require(im.rgb[(116 * 140 + 113) * 3] == 0 && im.rgb[117 * 140 * 3] == 0,
          "black right/bottom padding");
  std::vector<uint8_t> large(2000 * 1000 * 3, 91);
  im = dgpp::serve::resize_glm_image(large.data(), 2000, 1000, 256);
  require(im.tokens <= 256 && im.width % 28 == 0 && im.height % 28 == 0, "low detail budget");
  require(im.width > im.height && im.rgb[0] == 91, "aspect and constant pixels");
  uint8_t pixel[] = {255, 0, 42};
  im = dgpp::serve::resize_glm_image(pixel, 1, 1, 1024);
  require(im.tokens == 16 && im.width == 112 && im.height == 112, "minimum image budget");
  require(im.rgb[0] == 255 && im.rgb[1] == 0 && im.rgb[2] == 42, "small image rescale");
}
DGPP_TEST(image_inputs_reject_invalid_sources_and_spans) {
  using V = dgpp::minijson::Value;
  for (auto url : {"file:///etc/passwd", "https://example.com/x.png", "data:image/gif;base64,AAAA",
                   "data:image/png;base64,AAAA"}) {
    auto obj = V::make_object({{"url", V::make_string(url)}});
    rejects([&] { dgpp::serve::prepare_glm_image(obj, "messages[0].content[0].image_url"); });
  }
  dgpp::ImageInput im{1, 1, 28, 28, std::vector<uint8_t>(28 * 28 * 3)};
  dgpp::validate_image_inputs({im}, 3);
  rejects([&] { dgpp::validate_image_inputs({im}, 1); });
  rejects([&] { dgpp::validate_image_inputs({im, im}, 3); });
  im.rgb.pop_back();
  rejects([&] { dgpp::validate_image_inputs({im}, 3); });
}

DGPP_TEST(image_inputs_history_uses_context_and_bytes_instead_of_image_count) {
  std::vector<dgpp::ImageInput> images;
  int64_t end = 1;
  for (int i = 0; i < 16; ++i) {
    images.push_back({end, 1024, 896, 896, std::vector<uint8_t>(896 * 896 * 3, 123)});
    end += 1026;
    dgpp::validate_image_inputs(images, end);
  }
  auto extra = images.back();
  extra.offset = end;
  images.push_back(extra);
  dgpp::validate_image_inputs(images, end + 1024);
  rejects([&] { dgpp::validate_image_inputs(images, end + 1023); });
  images.pop_back();
  images.back().tokens = 1025;
  rejects([&] { dgpp::validate_image_inputs(images, end); });
}

DGPP_TEST(image_inputs_reject_geometry_overflow_and_decoded_byte_exhaustion) {
  dgpp::ImageInput im{0, 1, 28, 28, std::vector<uint8_t>(28 * 28 * 3)};
  im.offset = std::numeric_limits<int64_t>::max();
  rejects([&] { dgpp::validate_image_inputs({im}, std::numeric_limits<int64_t>::max()); });
  im.offset = 0;
  im.tokens = 2;
  rejects([&] { dgpp::validate_image_inputs({im}, 2); });
  std::vector<dgpp::ImageInput> images;
  const size_t image_bytes = 896 * 896 * 3;
  const size_t count = dgpp::kMaxRequestImageBytes / image_bytes;
  for (size_t i = 0; i <= count; ++i)
    images.push_back({static_cast<int64_t>(i * 1024), 1024, 896, 896,
                      std::vector<uint8_t>(image_bytes)});
  rejects([&] { dgpp::validate_image_inputs(images, images.size() * 1024); });
  images.pop_back();
  dgpp::validate_image_inputs(images, images.size() * 1024);
}

DGPP_TEST(image_png_decode_and_detail_validation) {
  using V = dgpp::minijson::Value;
  const auto png = V::make_string(
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
      "x8AAwMCAO+a0ioAAAAASUVORK5CYII=");
  const auto image = dgpp::serve::prepare_glm_image(V::make_object({{"url", png}}), "image_url");
  require(image.tokens == 16 && image.width == 112 && image.height == 112, "decode small PNG");
  for (auto detail : {V::make_string("invalid"), V::make_int(1)}) {
    try {
      dgpp::serve::prepare_glm_image(V::make_object({{"url", png}, {"detail", detail}}),
                                     "image_url");
      throw std::runtime_error("invalid detail accepted");
    } catch (const dgpp::serve::ImageInputError& e) {
      require(e.param == "image_url.detail", "precise image detail error");
    }
  }
}

DGPP_TEST(image_webp_decode_matches_its_pixels_and_keeps_the_magic_check) {
  using V = dgpp::minijson::Value;
  // 4x2 lossless: columns 0-1 red, columns 2-3 blue. Small enough to embed, and
  // asymmetric so a decode that returns one flat colour cannot pass.
  const auto webp = V::make_string(
      "data:image/webp;base64,UklGRiIAAABXRUJQVlA4TBYAAAAvA0AAAA8Q87//8x8OFAIIgIImov/x");
  const auto image = dgpp::serve::prepare_glm_image(V::make_object({{"url", webp}}), "image_url");
  // A 4x2 source keeps its 2:1 aspect on the 28-pixel grid: 6x3 cells, 18
  // tokens, rather than the square canvas a 1x1 source lands on.
  require(image.width == 168 && image.height == 84 && image.tokens == 18, "webp canvas");
  // Sample the middle of each half after the resize.
  const auto pixel = [&image](int x, int y) {
    const size_t at = (static_cast<size_t>(y) * image.width + static_cast<size_t>(x)) * 3;
    return std::array<int, 3>{image.rgb[at], image.rgb[at + 1], image.rgb[at + 2]};
  };
  const auto left = pixel(image.width / 4, image.height / 2);
  const auto right = pixel(3 * image.width / 4, image.height / 2);
  require(left[0] > 200 && left[1] < 60 && left[2] < 60, "webp left half is red");
  require(right[2] > 200 && right[0] < 60 && right[1] < 60, "webp right half is blue");

  // The magic check must still hold now that a third format is accepted.
  const auto mismatched = V::make_string(
      "data:image/png;base64,UklGRiIAAABXRUJQVlA4TBYAAAAvA0AAAA8Q87//8x8OFAIIgIImov/x");
  rejects(
      [&] { dgpp::serve::prepare_glm_image(V::make_object({{"url", mismatched}}), "image_url"); });
  const auto gif = V::make_string("data:image/gif;base64,R0lGODlhAQABAAAAACw=");
  rejects([&] { dgpp::serve::prepare_glm_image(V::make_object({{"url", gif}}), "image_url"); });
}

DGPP_TEST(image_webp_lossy_and_alpha_decode) {
  using V = dgpp::minijson::Value;
  // The same red 4x2 image: opaque VP8, VP8 + uncompressed ALPH, and VP8L alpha.
  const std::array<const char*, 3> payloads = {
      "UklGRkQAAABXRUJQVlA4IDgAAABQAgCdASoEAAIAAAAAJaACdLoB+AH6AfwAB5AA/v9vVv/8sPx/Unn/"
      "yxD//xPTm8RU8r/+JkAAAA==",
      "UklGRmgAAABXRUJQVlA4WAoAAAAQAAAAAwAAAQAAQUxQSAkAAAAAgICAgICAgIAAVlA4IDgAAABQAgCdASoE"
      "AAIAAAAAJaACdLoB+AH6AfwAB5AA/v9vVv/8sPx/Unn/yxD//xPTm8RU8r/+JkAAAA==",
      "UklGRhwAAABXRUJQVlA4TA8AAAAvA0AAEAcQ/Y8CBiKi/wEA"};
  for (size_t i = 0; i < payloads.size(); ++i) {
    const auto uri = std::string("data:image/webp;base64,") + payloads[i];
    const auto url = V::make_string(uri);
    const auto image = dgpp::serve::prepare_glm_image(V::make_object({{"url", url}}), "image_url");
    require(image.width == 168 && image.height == 84, "lossy/alpha webp canvas");
    const size_t at = (static_cast<size_t>(image.height / 2) * image.width + image.width / 2) * 3;
    const int other = i == 0 ? 0 : 127;
    require(image.rgb[at] >= 252 && std::abs(image.rgb[at + 1] - other) <= 3 &&
                std::abs(image.rgb[at + 2] - other) <= 3,
            "webp red and alpha composited on white");
  }
}

DGPP_TEST(image_webp_rejects_malformed_chunk_and_partition_lengths) {
  using V = dgpp::minijson::Value;
  const auto reject_bytes = [](const std::string& bytes) {
    const auto uri = "data:image/webp;base64," + dgpp::encode_base64(bytes);
    const auto url = V::make_string(uri);
    try {
      dgpp::serve::prepare_glm_image(V::make_object({{"url", url}}), "image_url");
      throw std::runtime_error("malformed WebP accepted");
    } catch (const dgpp::serve::ImageInputError& e) {
      require(e.param == "image_url.url", "precise WebP decode error");
    }
  };
  const auto set_u32 = [](std::string& bytes, size_t offset, uint32_t value) {
    for (int i = 0; i < 4; ++i) bytes[offset + i] = static_cast<char>(value >> (8 * i));
  };
  // Previously copied eight bytes into the four-byte RIFF type buffer.
  reject_bytes(std::string("RIFF\0\0\0\0WEBPABCD", 16));
  // A ten-byte VP8 frame declares a nine-byte first partition after its header.
  // Previously the partition reader ran past its ten-byte heap allocation.
  reject_bytes(dgpp::decode_base64("UklGRhYAAABXRUJQVlA4IAoAAAAwAQCdASoBAAEA", 100));

  const auto valid =
      dgpp::decode_base64("UklGRiIAAABXRUJQVlA4TBYAAAAvA0AAAA8Q87//8x8OFAIIgIImov/x", 100);
  for (size_t n = 12; n < valid.size(); ++n) reject_bytes(valid.substr(0, n));
  for (size_t offset : {size_t{4}, size_t{16}}) {
    for (uint32_t length : {0u, 1u, 4u, 0xffffffffu}) {
      auto bytes = valid;
      set_u32(bytes, offset, length);
      reject_bytes(bytes);
    }
  }
  // Finding an image chunk must not hide a truncated trailing chunk header.
  for (size_t n = 1; n < 8; ++n) {
    auto bytes = valid + std::string(n, 'X');
    set_u32(bytes, 4, static_cast<uint32_t>(bytes.size() - 8));
    reject_bytes(bytes);
  }
  // Unknown metadata chunks are still allowed, including their odd-byte padding.
  auto metadata = valid + std::string("XMP \1\0\0\0x\0", 10);
  set_u32(metadata, 4, static_cast<uint32_t>(metadata.size() - 8));
  const auto uri = "data:image/webp;base64," + dgpp::encode_base64(metadata);
  const auto url = V::make_string(uri);
  const auto image = dgpp::serve::prepare_glm_image(V::make_object({{"url", url}}), "image_url");
  require(image.width == 168 && image.height == 84, "webp with metadata");
  metadata.pop_back();
  set_u32(metadata, 4, static_cast<uint32_t>(metadata.size() - 8));
  reject_bytes(metadata);
}

DGPP_TEST(image_qwen_smart_resize_rounds_without_padding) {
  // factor 32 (patch 16 x merge 2): the nearest multiple, ties to even, and
  // never a padded canvas. A 100x100 image is under the 64-token pixel floor,
  // so it grows to 256x256 the way the processor's min_pixels does.
  std::vector<uint8_t> rgb(100 * 100 * 3, 77);
  dgpp::serve::ImagePreprocess pp{dgpp::kQwenImageGrid, 64, dgpp::kMaxImageTokens, false, 256};
  auto im = dgpp::serve::resize_image(rgb.data(), 100, 100, pp);
  require(im.width == 256 && im.height == 256 && im.tokens == 64 && im.grid == 32,
          "the min_pixels budget grows a small image");
  require(im.rgb.size() == 256 * 256 * 3 && im.rgb[(255 * 256 + 255) * 3] == 77,
          "no padding canvas");
  // 272/32 = 8.5 rounds down to the even 8, 304/32 = 9.5 rounds up to the even 10.
  std::vector<uint8_t> even(272 * 272 * 3, 61);
  im = dgpp::serve::resize_image(even.data(), 272, 272, pp);
  require(im.width == 256 && im.height == 256 && im.tokens == 64, "a tie rounds to even");
  std::vector<uint8_t> odd(304 * 304 * 3, 61);
  im = dgpp::serve::resize_image(odd.data(), 304, 304, pp);
  require(im.width == 320 && im.height == 320 && im.tokens == 100, "the other tie");
  // The 1024-token ceiling caps a large image, on the 32-pixel grid.
  std::vector<uint8_t> big(4000 * 4000 * 3, 200);
  im = dgpp::serve::resize_image(big.data(), 4000, 4000, pp);
  require(im.tokens <= 1024 && im.width % 32 == 0 && im.height % 32 == 0 && im.width == im.height,
          "the token ceiling");
  require(im.tokens > 900, "the ceiling is not wasteful");
  // A side below the grid grows to the pixel floor; an extreme aspect ratio
  // is refused, as the reference processor does.
  std::vector<uint8_t> thin(16 * 1024 * 3, 3);
  im = dgpp::serve::resize_image(thin.data(), 16, 1024, pp);
  require(im.tokens >= 64 && im.width % 32 == 0 && im.height % 32 == 0, "a short side grows");
  std::vector<uint8_t> sliver(6600 * 32 * 3, 3);
  rejects([&] { dgpp::serve::resize_image(sliver.data(), 6600, 32, pp); });
}

DGPP_TEST(image_resize_taps_are_centered_on_the_source) {
  // A hard edge must land where the geometry says it does: the filter's taps
  // have to be read from the positions they were computed for. Reading them
  // from the window's first index instead shifts the picture by that many
  // pixels and, at the last row, reads past the buffer.
  const int sw = 272, sh = 272, half = 136;
  std::vector<uint8_t> rgb(static_cast<size_t>(sw) * sh * 3);
  for (int y = 0; y < sh; ++y)
    for (int x = 0; x < sw; ++x)
      for (int c = 0; c < 3; ++c) rgb[(static_cast<size_t>(y) * sw + x) * 3 + c] =
          static_cast<uint8_t>(x < half ? 0 : 255);
  const dgpp::serve::ImagePreprocess pp{dgpp::kQwenImageGrid, 64, dgpp::kMaxImageTokens, false, 256};
  const auto im = dgpp::serve::resize_image(rgb.data(), sw, sh, pp);
  require(im.width == 256 && im.height == 256, "272 -> 256");
  const int row = 128 * im.width;
  require(im.rgb[row * 3] < 8 && im.rgb[(row + 255) * 3] > 247, "both halves kept their value");
  int crossing = -1;
  for (int x = 0; x < im.width; ++x)
    if (im.rgb[(row + x) * 3] > 128) { crossing = x; break; }
  require(crossing >= 126 && crossing <= 130, "the edge did not move");
}

DGPP_TEST(image_grid_is_part_of_the_geometry_and_the_identity) {
  dgpp::ImageInput qwen{0, 1, 32, 32, std::vector<uint8_t>(32 * 32 * 3), dgpp::kQwenImageGrid};
  dgpp::validate_image_inputs({qwen}, 2);
  // A 28-grid canvas cannot describe a 32-grid image, in either direction.
  auto wrong_count = qwen;
  wrong_count.tokens = 2;
  rejects([&] { dgpp::validate_image_inputs({wrong_count}, 3); });
  auto wrong_grid = qwen;
  wrong_grid.width = wrong_grid.height = 28;
  rejects([&] { dgpp::validate_image_inputs({wrong_grid}, 2); });
  auto unknown_grid = qwen;
  unknown_grid.grid = 24;
  unknown_grid.width = unknown_grid.height = 24;
  unknown_grid.rgb.resize(24 * 24 * 3);
  rejects([&] { dgpp::validate_image_inputs({unknown_grid}, 2); });
  // The two families' pixels-per-token ceilings differ, so a 1024-token Qwen
  // image is larger than the GLM workspace bound for the same token count.
  require(dgpp::pixels_for(dgpp::kQwenImageGrid) > dgpp::kMaxImagePixels, "per-family pixel cap");
  require(dgpp::pixels_for(dgpp::kGlmImageGrid) == dgpp::kMaxImagePixels, "GLM cap unchanged");
}

DGPP_TEST(image_prepare_uses_the_family_preprocess) {
  using V = dgpp::minijson::Value;
  using namespace std::string_literals;
  // The 1x1 PNG the GLM frontend test uses: 16 tokens on the 28 grid, 64
  // (the min_pixels floor) on the Qwen one.
  const std::string png =
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
      "x8AAwMCAO+a0ioAAAAASUVORK5CYII=";
  const auto url = V::make_owned_string("data:image/png;base64,"s +
                                        png.substr(png.find(',') + 1));
  const dgpp::serve::ImagePreprocess qwen{dgpp::kQwenImageGrid, 64, dgpp::kMaxImageTokens, false, 256};
  const auto glm_image = dgpp::serve::prepare_glm_image(V::make_object({{"url", url}}), "i");
  const auto qwen_image = dgpp::serve::prepare_image(V::make_object({{"url", url}}), "i", qwen);
  require(glm_image.tokens == 16 && glm_image.grid == 28, "GLM geometry");
  require(qwen_image.tokens >= 64 && qwen_image.grid == 32, "Qwen geometry");
  auto low = dgpp::serve::prepare_image(V::make_object({{"url", url}, {"detail", V::make_string("low")}}),
                                       "i", qwen);
  require(low.tokens <= 256, "detail low honors the family budget");
  rejects([&] { dgpp::serve::prepare_image(V::make_object({{"url", V::make_string("http://x/y.png")}}),
                                          "i", qwen); });
}

DGPP_TEST(image_vision_config_shapes_and_numeric_bounds) {
  const std::string config =
      R"({"image_token_id":154854,"image_start_token_id":154830,"image_end_token_id":154831,"vision_config":{"depth":24,"hidden_size":1024,"num_heads":16,"intermediate_size":4096,"out_hidden_size":4096,"projection_intermediate_size":10240,"patch_size":14,"temporal_patch_size":2,"spatial_merge_size":2,"in_channels":3,"rms_norm_eps":0.00001,"swiglu_limit":10,"hidden_act":"silu","attention_bias":true}})";
  const auto parse = [](const std::string& s) {
    return dgpp::GlmVisionConfig::parse(dgpp::minijson::parse(s).root, 4096);
  };
  require(parse(config).weight_bytes() == 1127254016, "actual vision weight budget");
  for (auto change : {std::pair{"\"depth\":24", "\"depth\":1e100"},
                      {"\"depth\":24", "\"depth\":1.5"},
                      {"\"num_heads\":16", "\"num_heads\":3"},
                      {"\"patch_size\":14", "\"patch_size\":16"},
                      {"\"out_hidden_size\":4096", "\"out_hidden_size\":1024"},
                      {"\"rms_norm_eps\":0.00001", "\"rms_norm_eps\":1e100"},
                      {"\"swiglu_limit\":10", "\"swiglu_limit\":1e-100"},
                      {"\"image_token_id\":154854", "\"image_token_id\":154854.5"}}) {
    auto bad = config;
    bad.replace(bad.find(change.first), std::string_view(change.first).size(), change.second);
    rejects([&] { parse(bad); });
  }
}

DGPP_TEST(qwen_vision_config_geometry_ids_and_refusals) {
  // The checkpoint's own numbers (Qwen/Qwen3.8-Flash-Next-FP8 config.json).
  const std::string config =
      R"({"image_token_id":248056,"vision_start_token_id":248053,"vision_end_token_id":248054,"video_token_id":248057,"vision_config":{"deepstack_visual_indexes":[],"depth":27,"hidden_act":"gelu_pytorch_tanh","hidden_size":1152,"in_channels":3,"intermediate_size":4304,"num_heads":16,"num_position_embeddings":2304,"out_hidden_size":2560,"patch_size":16,"spatial_merge_size":2,"temporal_patch_size":2}})";
  const auto parse = [](const std::string& s) {
    return dgpp::QwenVisionConfig::parse(dgpp::minijson::parse(s).root, 2560);
  };
  const auto c = parse(config);
  require(c.depth == 27 && c.hidden == 1152 && c.heads == 16 && c.intermediate == 4304, "the tower");
  require(c.head_dim() == 72 && c.merged() == 4608 && c.patch_in() == 1536, "derived geometry");
  require(c.tokens.start == 248053 && c.tokens.pad == 248056 && c.tokens.end == 248054,
          "the image delimiters");
  require(c.eps == dgpp::QwenVisionConfig::kLayerNormEps &&
              c.rope_theta == dgpp::QwenVisionConfig::kRopeTheta,
          "an absent eps and theta take the library defaults");
  require(c.weight_bytes() == 897862112, "the vision weight budget");
  require(c.workspace_bytes() > c.weight_bytes() / 32 && c.workspace_bytes() < c.weight_bytes(),
          "the workspace is a fraction of the tower");
  require(dgpp::QwenVisionConfig::kGrid == dgpp::kQwenImageGrid &&
              dgpp::QwenVisionConfig::kGrid == c.kPatch * c.kMerge,
          "one grid constant for the processor and the tower");
  for (auto change : {
          std::pair{"\"patch_size\":16", "\"patch_size\":14"},
          {"\"spatial_merge_size\":2", "\"spatial_merge_size\":4"},
          {"\"temporal_patch_size\":2", "\"temporal_patch_size\":1"},
          {"\"num_position_embeddings\":2304", "\"num_position_embeddings\":256"},
          {"\"out_hidden_size\":2560", "\"out_hidden_size\":4096"},
          {"\"hidden_size\":1152", "\"hidden_size\":1155"},
          {"\"num_heads\":16", "\"num_heads\":5"},
          {"\"hidden_act\":\"gelu_pytorch_tanh\"", "\"hidden_act\":\"silu\""},
          {"\"deepstack_visual_indexes\":[]", "\"deepstack_visual_indexes\":[8,16,24]"},
          {"\"depth\":27", "\"depth\":27.5"},
          {"\"image_token_id\":248056", "\"image_token_id\":-1"},
          {"\"image_token_id\":248056", "\"video_token_id\":248056,\"image_token_id\":248056"},
      }) {
    auto bad = config;
    if (bad.find(change.first) == std::string::npos) {
      require(false, "the fixture no longer holds the field under test");
      break;
    }
    bad.replace(bad.find(change.first), std::string_view(change.first).size(), change.second);
    rejects([&] { parse(bad); });
  }
  // A text-only export (language_model_only) carries no vision_config, and
  // parse() refuses to invent one.
  const std::string text_only =
      R"({"image_token_id":248056,"vision_start_token_id":248053,"vision_end_token_id":248054})";
  rejects([&] { parse(text_only); });
  // An explicit eps is honoured, and a nonsense one refused.
  auto eps = config;
  const std::string_view depth{"\"depth\":27"};
  eps.insert(eps.find(depth) + depth.size(), ",\"layer_norm_eps\":1e-5");
  require(parse(eps).eps == 1e-5f, "an explicit layer_norm_eps");
  const std::string_view eps_key{"\"layer_norm_eps\":1e-5"};
  eps.replace(eps.find(eps_key), eps_key.size(), "\"layer_norm_eps\":0");
  rejects([&] { parse(eps); });
}

DGPP_TEST(image_preparation_concurrent_formats_and_errors) {
  using V = dgpp::minijson::Value;
  const std::array<const char*, 3> urls = {
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
      "x8AAwMCAO+a0ioAAAAASUVORK5CYII=",
      "data:image/webp;base64,UklGRiIAAABXRUJQVlA4TBYAAAAvA0AAAA8Q87//8x8OFAIIgIImov/x",
      "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/"
      "2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofH"
      "h0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/"
      "2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyM"
      "jIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAIAAgDASIAAhEBAxEB/"
      "8QAHwAAAQUBAQEBAQEAAAAAAAAAA"
      "AECAwQFBgcICQoL/"
      "8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKF"
      "hcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjp"
      "KWmp"
      "6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/"
      "8QAHwEAAwEBAQEBAQEBAQAAAAAAA"
      "AECAwQFBgcICQoL/"
      "8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkN"
      "OEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmao"
      "qOkp"
      "aanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/"
      "9oADAMBAAIRAxEAPwDkaKKK+7Pqz"
      "//Z"};
  std::array<dgpp::ImageInput, 3> expected;
  for (size_t i = 0; i < urls.size(); ++i)
    expected[i] =
        dgpp::serve::prepare_glm_image(V::make_object({{"url", V::make_string(urls[i])}}), "image");
  std::atomic<bool> mismatch{false};
  std::vector<std::thread> workers;
  for (int worker = 0; worker < 16; ++worker)
    workers.emplace_back([&, worker] {
      try {
        for (int round = 0; round < 8; ++round) {
          const size_t i = (worker + round) % urls.size();
          const auto actual = dgpp::serve::prepare_glm_image(
              V::make_object({{"url", V::make_string(urls[i])}}), "image");
          if (actual.rgb != expected[i].rgb || actual.tokens != expected[i].tokens ||
              actual.width != expected[i].width || actual.height != expected[i].height)
            mismatch = true;
          bool refused = false;
          try {
            dgpp::serve::prepare_glm_image(
                V::make_object({{"url", V::make_string("data:image/png;base64,iVBORw0KGgo=")}}),
                "bad");
          } catch (const dgpp::serve::ImageInputError& e) {
            refused = e.param == "bad.url";
          }
          if (!refused) mismatch = true;
        }
      } catch (...) {
        mismatch = true;
      }
    });
  for (auto& worker : workers) worker.join();
  require(!mismatch, "concurrent PNG/JPEG/WebP preparation preserves pixels and isolates errors");
}
