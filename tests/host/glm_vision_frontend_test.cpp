#include "serve/glm_vision_frontend.hpp"

#include <algorithm>
#include <iostream>

#include "loaders/hf_cache.hpp"
#include "serve/image_inputs.hpp"

int main() {
  std::string error;
  const auto checkpoint = dgpp::hf::model_dir("unsloth/GLM-5.3-Flash-FP8", &error);
  if (checkpoint.empty()) {
    std::cout << "[SKIP] " << error << '\n';
    return 2;
  }
  try {
    const auto tok = dgpp::text::Tokenizer::load(checkpoint + "/tokenizer.json");
    const auto tpl = dgpp::text::ChatTemplate::load(checkpoint + "/chat_template.jinja");
    dgpp::serve::TextFrontend text(&tok, &tpl);
    dgpp::serve::GlmVisionFrontend vision(&tok, &tpl);
    const std::string png =
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
        "x8AAwMCAO+a0ioAAAAASUVORK5CYII=";
    const auto require = [](bool ok, const char* message) {
      if (!ok) throw std::runtime_error(message);
    };
    const auto plain = dgpp::minijson::parse(R"({"messages":[{"role":"user","content":"hello"}]})");
    require(text.prepare_chat(plain.root).tokens == vision.prepare_chat(plain.root).tokens,
            "text prompts unchanged");
    const std::string json =
        "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"first\"},{"
        "\"type\":\"image_url\",\"image_url\":{\"url\":\"" +
        png + "\"}}," +
        "{\"type\":\"text\",\"text\":\"then\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"" +
        png + "\"}}]}]}";
    const auto parsed = dgpp::minijson::parse(json);
    const auto input = vision.prepare_chat(parsed.root);
    require(input.images.size() == 2, "two ordered images");
    require(std::count(input.tokens.begin(), input.tokens.end(), 154854) == 32,
            "expanded image tokens");
    require(input.images[1].offset > input.images[0].offset + 16, "interleaved text preserved");
    for (const auto& im : input.images) {
      require(
          input.tokens[im.offset - 1] == 154830 && input.tokens[im.offset + im.tokens] == 154831,
          "GLM image delimiters");
      require(im.tokens == 16 && im.rgb.size() == 112 * 112 * 3, "pixel/token geometry");
    }
    require(tok.decode(input.tokens, false).find("unable to process") == std::string::npos,
            "no text-only template reminder");
    auto conflict = json;
    conflict.replace(conflict.find("first"), 5, "<|image|>");
    bool threw = false;
    try {
      vision.prepare_chat(dgpp::minijson::parse(conflict).root);
    } catch (const dgpp::serve::ImageInputError&) {
      threw = true;
    }
    require(threw, "raw image token cannot steal a visual embedding span");
    // An 896x896 one-bit black PNG: each decoded image uses 1024 visual
    // tokens. Spread the images across turns, as a screenshot client does.
    const std::string large_png =
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA4AAAAOAAQAAAABDTyD6AAAAeUlEQVR4nO3BMQEAAADC"
        "oPVPbQsvoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAvgaLjwABbzzxhwAAAABJRU5ErkJggg==";
    const auto history = [&](int count) {
      std::string body = "{\"messages\":[";
      for (int i = 0; i < count; ++i) {
        if (i) body += ',';
        body += "{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"" +
                large_png + "\"}}]}";
      }
      return body + "]}";
    };
    const auto large = vision.prepare_chat(dgpp::minijson::parse(history(16)).root);
    require(large.images.size() == 16 &&
                std::count(large.tokens.begin(), large.tokens.end(), 154854) == 16384,
            "image history expands into ordinary context positions without a count/token cap");
    std::cout << "GLM vision frontend checks passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
