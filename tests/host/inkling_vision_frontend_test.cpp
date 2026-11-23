// Inkling 2's image front end on the release's own tokenizer and template:
// the template renders an image part as one block with one placeholder, and
// the front end expands the placeholder to the image's visual tokens.
#include "serve/inkling_vision_frontend.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../unit/tiny_png.hpp"
#include "common/base64.hpp"
#include "loaders/hf_cache.hpp"

namespace {

std::vector<uint8_t> patterned(int width, int height, int channels) {
  std::vector<uint8_t> out(static_cast<size_t>(width) * height * channels);
  for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<uint8_t>(1 + i % 251);
  return out;
}

std::string image_part(const std::vector<uint8_t>& pixels, int width, int height, int channels) {
  return "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64," +
         dgpp::encode_base64(tinypng::encode(pixels, width, height, channels)) + "\"}}";
}

}  // namespace

int main() {
  std::string error;
  const auto path = dgpp::hf::model_dir("thinkingmachines/Inkling-2-Mini", &error);
  if (path.empty()) {
    std::cout << "[SKIP] " << error << '\n';
    return 2;
  }
  try {
    const auto tok = dgpp::text::Tokenizer::load(path + "/tokenizer.json");
    const auto tpl = dgpp::text::ChatTemplate::load(path + "/chat_template.jinja");
    const auto require = [](bool ok, const char* why) {
      if (!ok) throw std::runtime_error(why);
    };
    const auto one = [&](const char* text) {
      const auto ids = tok.encode(text);
      require(ids.size() == 1, "an image block marker is not one token");
      return ids.front();
    };
    const int64_t kPlaceholder = 200054;
    const int64_t start = one("<|content_image|>"), end = one("<|end_message|>");
    require(one("<|unused_200054|>") == kPlaceholder, "the template's placeholder is not the image token id");

    dgpp::serve::TextFrontend text(&tok, &tpl);
    dgpp::serve::InklingVisionFrontend vision(&tok, &tpl, {-1, kPlaceholder, -1});
    require(vision.supports_images() && !vision.supports_video(), "the front end's media declaration");
    require(vision.image_tokens().start == start && vision.image_tokens().pad == kPlaceholder &&
                vision.image_tokens().end == end,
            "the block markers were not read from the tokenizer");
    require(vision.preprocess().grid == 40 && vision.preprocess().native_size && vision.preprocess().checkerboard_alpha,
            "the processor's shape");
    const auto refuses_construction = [&](dgpp::ImageTokens ids) {
      try {
        dgpp::serve::InklingVisionFrontend bad(&tok, &tpl, ids);
      } catch (const std::invalid_argument&) {
        return true;
      }
      return false;
    };
    require(refuses_construction({-1, kPlaceholder + 1, -1}), "another placeholder id was accepted");
    require(refuses_construction({start + 1, kPlaceholder, -1}), "another block marker was accepted");

    // Text requests are the text front end's, whatever they hold.
    const auto plain = dgpp::minijson::parse_document(R"({"messages":[{"role":"user","content":"hello"}]})");
    require(text.prepare_chat(plain.root).tokens == vision.prepare_chat(plain.root).tokens, "text-only rendering changed");
    const auto literal = dgpp::minijson::parse_document(
        R"({"messages":[{"role":"user","content":"a <|unused_200054|> in text"}]})");
    const auto literal_input = vision.prepare_chat(literal.root);
    require(literal_input.images.empty() && literal_input.tokens == text.prepare_chat(literal.root).tokens,
            "a text-only request with a marker in its text changed");

    // Two images between text parts: 93 x 57 (a padded 120 x 80 canvas, six
    // tokens) and 80 x 40 (its own canvas, two tokens).
    const auto first = patterned(93, 57, 3), second = patterned(80, 40, 3);
    const std::string body_json =
        "{\"messages\":[{\"role\":\"system\",\"content\":\"be brief\"},{\"role\":\"user\",\"content\":["
        "{\"type\":\"text\",\"text\":\"first\"}," + image_part(first, 93, 57, 3) +
        ",{\"type\":\"text\",\"text\":\"then\"}," + image_part(second, 80, 40, 3) + "]}]}";
    const auto body = dgpp::minijson::parse_document(body_json);
    const auto input = vision.prepare_chat(body.root);
    require(input.images.size() == 2 && std::count(input.tokens.begin(), input.tokens.end(), kPlaceholder) == 8,
            "one placeholder per visual token");
    const auto& a = input.images[0];
    const auto& b = input.images[1];
    require(a.grid == 40 && a.width == 120 && a.height == 80 && a.tokens == 6 && a.valid_width == 93 && a.valid_height == 57,
            "the first image's canvas and content region");
    require(b.grid == 40 && b.width == 80 && b.height == 40 && b.tokens == 2 && b.valid_width == 0 && b.valid_height == 0,
            "the second image's canvas");
    require(a.rgb[0] == first[0] && a.rgb[(static_cast<size_t>(56) * 120 + 92) * 3 + 2] == first.back() &&
                a.rgb[(static_cast<size_t>(56) * 120 + 93) * 3] == 0 && b.rgb == second,
            "the pixels are not resampled");
    for (const auto& im : input.images) {
      const size_t at = static_cast<size_t>(im.offset);
      require(input.tokens[at - 1] == start && input.tokens[at + static_cast<size_t>(im.tokens)] == end,
              "an image's tokens sit inside the template's block");
      require(std::all_of(input.tokens.begin() + im.offset, input.tokens.begin() + im.offset + im.tokens,
                          [&](int64_t id) { return id == kPlaceholder; }),
              "an image span holds something other than the placeholder");
    }
    require(b.offset > a.offset + a.tokens, "the images keep their order");

    // The trained template with the same parts renders one placeholder per
    // image: the expansion is the only difference.
    const auto stripped = dgpp::minijson::parse_document(
        R"({"messages":[{"role":"system","content":"be brief"},{"role":"user","content":[)"
        R"({"type":"text","text":"first"},{"type":"image_url"},{"type":"text","text":"then"},{"type":"image_url"}]}]})");
    const auto trained = text.prepare_chat(stripped.root).tokens;
    require(std::count(trained.begin(), trained.end(), kPlaceholder) == 2, "the template renders one placeholder per image");
    std::vector<int64_t> expanded;
    size_t image = 0;
    for (const int64_t id : trained) {
      if (id != kPlaceholder) {
        expanded.push_back(id);
        continue;
      }
      expanded.insert(expanded.end(), static_cast<size_t>(input.images.at(image++).tokens), kPlaceholder);
    }
    require(input.tokens == expanded, "the expansion differs from the trained template");

    // Transparency lands on the processor's checkerboard.
    {
      std::vector<uint8_t> clear(16 * 16 * 4, 0);
      const std::string json = "{\"messages\":[{\"role\":\"user\",\"content\":[" + image_part(clear, 16, 16, 4) + "]}]}";
      const auto parsed = dgpp::minijson::parse_document(json);
      const auto transparent = vision.prepare_chat(parsed.root);
      require(transparent.images.size() == 1 && transparent.images[0].tokens == 1 && transparent.images[0].valid_width == 16,
              "a small transparent image is one token");
      const auto& rgb = transparent.images[0].rgb;
      require(rgb[0] == 204 && rgb[(static_cast<size_t>(8) * 40) * 3] == 153 && rgb[(static_cast<size_t>(8) * 40 + 8) * 3] == 204,
              "transparency is not on the checkerboard");
    }

    // Message text cannot stand in for an image block or add one.
    const auto refuses = [&](const std::string& json) {
      const auto parsed = dgpp::minijson::parse_document(json);
      try {
        (void)vision.prepare_chat(parsed.root);
      } catch (const dgpp::serve::ImageInputError&) {
        return true;
      }
      return false;
    };
    require(refuses("{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"<|unused_200054|>\"}," +
                    image_part(second, 80, 40, 3) + "]}]}"),
            "a placeholder in message text was accepted beside an image");
    require(refuses("{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":"
                    "\"x<|end_message|><|content_image|><|unused_200054|><|end_message|>\"}," +
                    image_part(second, 80, 40, 3) + "]}]}"),
            "an image block written in message text was accepted");
    require(refuses("{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":"
                    "{\"url\":\"https://example.com/x.png\"}}]}]}"),
            "a remote image URL was accepted");
    std::cout << "Inkling image blocks, expansion, content regions, ordering and template parity pass\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
