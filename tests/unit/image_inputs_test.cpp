#include "serve/image_inputs.hpp"

#include "common/base64.hpp"
#include "common/test.hpp"
#include "models/glm/vision_config.hpp"

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
