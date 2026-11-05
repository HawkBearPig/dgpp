#include "serve/minimax_vision_frontend.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "loaders/hf_cache.hpp"

int main() {
  std::string error;
  const auto path = dgpp::hf::model_dir("MiniMaxAI/MiniMax-M3.1-Flash", &error);
  if (path.empty()) {
    std::cout << "[SKIP] " << error << '\n';
    return 2;
  }
  try {
    const auto tok = dgpp::text::Tokenizer::load(path + "/tokenizer.json");
    const auto tpl = dgpp::text::ChatTemplate::load(path + "/chat_template.jinja");
    dgpp::serve::TextFrontend text(&tok, &tpl, true);
    dgpp::serve::MinimaxVisionFrontend vision(&tok, &tpl, {200029, 200025, 200030});
    const auto require = [](bool ok, const char* why) {
      if (!ok) throw std::runtime_error(why);
    };
    require(vision.markers().model_opens_thinking, "vision frontend lost model-opened reasoning");
    const auto plain =
        dgpp::minijson::parse_document(R"({"messages":[{"role":"user","content":"hello"}]})");
    require(text.prepare_chat(plain.root).tokens == vision.prepare_chat(plain.root).tokens,
            "text-only rendering changed");
    const std::string png =
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
        "x8AAwMCAO+a0ioAAAAASUVORK5CYII=";
    const auto part = [&] {
      return "{\"type\":\"image_url\",\"image_url\":{\"url\":\"" + png + "\"}}";
    };
    const std::string body_json =
        "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"first\"}," +
        part() + ",{\"type\":\"text\",\"text\":\"then\"}," + part() + "]}]}";
    const auto body = dgpp::minijson::parse_document(body_json);
    const auto input = vision.prepare_chat(body.root);
    require(input.images.size() == 2 &&
                std::count(input.tokens.begin(), input.tokens.end(), 200025) == 8,
            "MiniMax image expansion count");
    for (const auto& im : input.images) {
      require(im.width == 56 && im.height == 56 && im.tokens == 4 && im.grid == 28,
              "MiniMax minimum canvas");
      require(
          input.tokens[im.offset - 1] == 200029 && input.tokens[im.offset + im.tokens] == 200030,
          "MiniMax image delimiters");
    }
    require(input.images[1].offset > input.images[0].offset + input.images[0].tokens,
            "interleaved image order");
    const std::string replacement =
        "]<]start of image[>[]<]image[>[]<]image[>[]<]image[>[]<]image[>[]<]end of image[>[";
    const std::string expanded_json = "{\"messages\":[{\"role\":\"user\",\"content\":\"first" +
                                      replacement + "then" + replacement + "\"}]}";
    const auto expanded = dgpp::minijson::parse_document(expanded_json);
    require(input.tokens == text.prepare_chat(expanded.root).tokens,
            "media substitution differs from the trained template");
    const std::string conflict_json =
        "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"]<]image[>["
        "\"}," +
        part() + "]}]}";
    const auto conflict = dgpp::minijson::parse_document(conflict_json);
    bool refused = false;
    try {
      (void)vision.prepare_chat(conflict.root);
    } catch (const dgpp::serve::ImageInputError&) {
      refused = true;
    }
    require(refused, "conflicting user image marker accepted");
    std::cout << "MiniMax media delimiters, ordering, reasoning and template parity pass\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
