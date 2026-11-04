#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "common/base64.hpp"
#include "loaders/hf_cache.hpp"
#include "serve/minimax_vision_frontend.hpp"
#include "serve/video_inputs.hpp"

namespace {
void require(bool ok, const char* why) {
  if (!ok) throw std::runtime_error(why);
}
template <class F>
void rejects(F&& f) {
  try {
    f();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("invalid video input accepted");
}
using V = dgpp::minijson::Value;
V source(const std::string& url, double fps) {
  return V::make_object({{"url", V::make_owned_string(url)}, {"fps", V::make_double(fps)}});
}
void pixels(const dgpp::ImageInput& im, const std::vector<int>& selected) {
  require(im.width == 56 && im.height == 56 && im.frames == int(selected.size()) &&
              im.tokens == 2 * int(selected.size()),
          "video temporal geometry");
  for (size_t f = 0; f < selected.size(); ++f) {
    const int index = selected[f];
    for (int p = 0; p < 56 * 56; ++p) {
      const auto* rgb = im.rgb.data() + (f * 56 * 56 + p) * 3;
      require(rgb[0] == 17 * index && rgb[1] == 255 - 11 * index && rgb[2] == 10 + 7 * index,
              "video decode/sampling changed pixels");
    }
  }
}
}  // namespace
int main(int argc, char** argv) {
  if (argc != 2) return 1;
  try {
    if (!dgpp::serve::native_video_available()) {
      const auto unavailable = [](auto&& call, std::string_view param, std::string_view message) {
        try {
          call();
        } catch (const dgpp::serve::ImageInputError& error) {
          require(error.param == param &&
                      std::string_view(error.what()).find(message) != std::string_view::npos,
                  "disabled video must identify the unsupported field and build capability");
          return;
        }
        throw std::runtime_error("video accepted without decoder support");
      };
      const auto clip = source("data:video/mp4;base64,AAAA", 1);
      unavailable([&] { dgpp::serve::prepare_minimax_video(clip, "video_url"); }, "video_url",
                  "DGPP_ENABLE_VIDEO");
      std::string error;
      const auto checkpoint = dgpp::hf::model_dir("MiniMaxAI/MiniMax-M3.1-Flash", &error);
      if (!checkpoint.empty()) {
        const auto tok = dgpp::text::Tokenizer::load(checkpoint + "/tokenizer.json");
        const auto tpl = dgpp::text::ChatTemplate::load(checkpoint + "/chat_template.jinja");
        dgpp::serve::MinimaxVisionFrontend frontend(&tok, &tpl, {200029, 200025, 200030});
        require(!frontend.supports_video(), "disabled decoder must not advertise video");
        const auto globals = V::make_object(
            {{"messages",
              V::make_array({V::make_object(
                  {{"role", V::make_string("user")},
                   {"content", V::make_array({V::make_object({{"type", V::make_string("video_url")},
                                                              {"video_url", clip}})})}})})}});
        unavailable([&] { frontend.prepare_chat(globals); }, "messages[0].content[0].video_url",
                    "does not support video");
      }
      std::cout << "Disabled video decoder and capability checks passed\n";
      return 0;
    }
    std::string mp4;
    for (const auto* extension : {"mp4", "webm"}) {
      std::ifstream file(std::string(argv[1]) + "/minimax_video_10frames." + extension,
                         std::ios::binary);
      const std::string bytes((std::istreambuf_iterator<char>(file)), {});
      require(!bytes.empty(), "missing video fixture");
      const std::string uri =
          "data:video/" + std::string(extension) + ";base64," + dgpp::encode_base64(bytes);
      if (std::string_view(extension) == "mp4") mp4 = uri;
      auto video = dgpp::serve::prepare_minimax_video(source(uri, 1), "video_url");
      require(video.segments.size() == 1 && video.timestamps == std::vector<double>{0, 2, 4.5},
              "uniform whole-clip sampling and odd-frame timestamps");
      pixels(video.segments[0], {0, 2, 4, 7, 9, 9});
      video = dgpp::serve::prepare_minimax_video(source(uri, 2), "video_url");
      require(video.segments.size() == 2 && video.timestamps == std::vector<double>{0, 1, 2, 3, 4},
              "eight-frame segment split and timestamps");
      pixels(video.segments[0], {0, 1, 2, 3, 4, 5, 6, 7});
      pixels(video.segments[1], {8, 9});
      rejects([&] { dgpp::serve::prepare_minimax_video(source(uri, 0), "video_url"); });
      rejects([&] { dgpp::serve::prepare_minimax_video(source(uri, 31), "video_url"); });
      rejects([&] { dgpp::serve::prepare_minimax_video(source(uri, 1), "video_url", 1000); });
      rejects([&] {
        dgpp::serve::prepare_minimax_video(source("data:video/mp4;base64,AAAA", 1), "video_url");
      });
      rejects([&] {
        dgpp::serve::prepare_minimax_video(source("https://example.com/clip.mp4", 1), "video_url");
      });
    }
    // Decode checks need no checkpoint. If present, also verify the actual
    // trained template, timestamp strings and discontiguous token spans.
    std::string error;
    const auto checkpoint = dgpp::hf::model_dir("MiniMaxAI/MiniMax-M3.1-Flash", &error);
    if (!checkpoint.empty()) {
      const auto tok = dgpp::text::Tokenizer::load(checkpoint + "/tokenizer.json");
      const auto tpl = dgpp::text::ChatTemplate::load(checkpoint + "/chat_template.jinja");
      dgpp::serve::MinimaxVisionFrontend frontend(&tok, &tpl, {200029, 200025, 200030});
      dgpp::serve::TextFrontend text(&tok, &tpl, true);
      const auto globals = [&](V content) {
        return V::make_object(
            {{"messages", V::make_array({V::make_object({{"role", V::make_string("user")},
                                                         {"content", std::move(content)}})})}});
      };
      const auto video =
          V::make_object({{"type", V::make_string("video_url")}, {"video_url", source(mp4, 2)}});
      const auto input = frontend.prepare_chat(globals(V::make_array({video})));
      require(frontend.supports_video() && input.images.size() == 2, "frontend video capability");
      dgpp::validate_image_inputs(input.images, input.tokens.size(), true);
      std::string expanded;
      for (int i = 0; i < 5; ++i) {
        expanded += "]<]" + std::to_string(i) + ".0 seconds[>[]<]start of image[>[";
        for (int j = 0; j < 4; ++j) expanded += "]<]video[>[";
        expanded += "]<]end of image[>[";
      }
      require(input.tokens == text.prepare_chat(globals(V::make_owned_string(expanded))).tokens,
              "video expansion differs from published timestamp/token recipe");
      for (const auto& im : input.images)
        for (int i = 0; i < im.spans(); ++i) {
          const int64_t at = im.span_offset(i);
          require(input.tokens[at - 1] == 200029 && input.tokens[at + 4] == 200030,
                  "video span delimiters");
          for (int j = 0; j < 4; ++j) require(input.tokens[at + j] == 200026, "video token IDs");
        }
      const auto conflict = V::make_object(
          {{"type", V::make_string("text")}, {"text", V::make_string("]<]video[>[")}});
      rejects([&] { frontend.prepare_chat(globals(V::make_array({conflict, video}))); });
    }
    std::cout << "MiniMax video decode, sampling, segmentation, bounds and template passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
