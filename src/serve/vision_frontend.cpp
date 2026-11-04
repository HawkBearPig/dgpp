#include "serve/vision_frontend.hpp"

#include <iomanip>
#include <locale>
#include <sstream>
#include <utility>

#include "serve/video_inputs.hpp"

namespace dgpp::serve {
VisionFrontend::VisionFrontend(const dgpp::text::Tokenizer* tok,
                               const dgpp::text::ChatTemplate* tpl, VisionSpec spec,
                               bool model_opens_thinking)
    : TextFrontend(tok, tpl, model_opens_thinking), spec_(std::move(spec)) {
  const auto single = [&](const std::string& text, int64_t id, const char* what) {
    const auto ids = encode_text(text);
    if (ids.size() != 1 || ids.front() != id)
      throw std::invalid_argument(std::string("vision frontend: ") + what +
                                  " does not tokenize to the checkpoint's image token id");
  };
  single(spec_.start_text, spec_.ids.start, "the image start marker");
  single(spec_.pad_text, spec_.ids.pad, "the image pad token");
  single(spec_.end_text, spec_.ids.end, "the image end marker");
  if (spec_.video_id >= 0) single(spec_.video_text, spec_.video_id, "the video pad token");
}
bool VisionFrontend::supports_video() const {
  return spec_.video_id >= 0 && native_video_available();
}

ModelFrontend::ChatInput VisionFrontend::prepare_chat(const minijson::Value& globals) const {
  using V = minijson::Value;
  ChatInput out;
  std::vector<V> messages;
  size_t image_bytes = 0;
  const auto& original = globals.at("messages").items();
  for (size_t i = 0; i < original.size(); ++i) {
    const auto& msg = original[i];
    std::vector<minijson::Member> fields;
    for (const auto& field : msg.members()) {
      if (field.key != "content" || !field.value.is_array()) {
        fields.push_back(field);
        continue;
      }
      std::vector<V> parts;
      for (size_t j = 0; j < field.value.items().size(); ++j) {
        const auto& part = field.value.items()[j];
        const auto* type = part.find("type");
        const bool video = type && type->is_string() && type->as_string() == "video_url";
        if (!video && (!type || !type->is_string() || type->as_string() != "image_url")) {
          parts.push_back(part);
          continue;
        }
        const std::string field = video ? "video_url" : "image_url";
        const std::string where =
            "messages[" + std::to_string(i) + "].content[" + std::to_string(j) + "]." + field;
        if (video && !supports_video())
          throw ImageInputError("this model or build does not support video inputs", where);
        VideoInput media;
        if (video)
          media = prepare_minimax_video(part.at(field), where, kMaxRequestImageBytes - image_bytes);
        else
          media.segments.push_back(prepare_image(part.at(field), where, spec_.preprocess));
        std::string placeholder;
        size_t timestamp = 0;
        for (auto& image : media.segments) {
          if (image.rgb.size() > kMaxRequestImageBytes - image_bytes)
            throw ImageInputError("decoded media exceeds the 256 MiB request byte limit", where);
          image_bytes += image.rgb.size();
          for (int span = 0; span < image.spans(); ++span) {
            if (video) {
              std::ostringstream time;
              time.imbue(std::locale::classic());
              time << "]<]" << std::fixed << std::setprecision(1)
                   << media.timestamps.at(timestamp++) << " seconds[>[";
              placeholder += time.str();
            }
            placeholder += spec_.start_text;
            for (int k = 0; k < image.span_tokens(); ++k)
              placeholder += video ? spec_.video_text : spec_.pad_text;
            placeholder += spec_.end_text;
          }
          out.images.push_back(std::move(image));
        }
        parts.push_back(V::make_object({{"type", V::make_string("text")},
                                        {"text", V::make_owned_string(std::move(placeholder))}}));
      }
      fields.push_back({field.key, V::make_array(std::move(parts))});
    }
    messages.push_back(V::make_object(std::move(fields)));
  }
  std::vector<minijson::Member> fields;
  for (const auto& f : globals.members())
    fields.push_back(f.key == "messages" ? minijson::Member{f.key, V::make_array(messages)} : f);
  out.tokens = encode_text(render_chat(V::make_object(std::move(fields))));
  if (out.images.empty()) return out;
  size_t pos = 0;
  const auto is_media = [&](int64_t id) {
    return id == spec_.ids.pad || (spec_.video_id >= 0 && id == spec_.video_id);
  };
  for (auto& im : out.images) {
    const int64_t pad = im.is_video() ? spec_.video_id : spec_.ids.pad;
    for (int span = 0; span < im.spans(); ++span) {
      while (pos < out.tokens.size() && !is_media(out.tokens[pos])) ++pos;
      if (!span) im.offset = static_cast<int64_t>(pos);
      if (im.is_video()) im.frame_offsets.push_back(static_cast<int64_t>(pos));
      if (pos == 0 || pos == out.tokens.size() || out.tokens[pos - 1] != spec_.ids.start)
        throw ImageInputError("media markers in message text conflict with media inputs",
                              "messages");
      for (int j = 0; j < im.span_tokens(); ++j)
        if (pos >= out.tokens.size() || out.tokens[pos++] != pad)
          throw ImageInputError("media token expansion did not match media inputs", "messages");
      if (pos >= out.tokens.size() || out.tokens[pos] != spec_.ids.end)
        throw ImageInputError("media token expansion did not match media inputs", "messages");
    }
  }
  for (; pos < out.tokens.size(); ++pos)
    if (is_media(out.tokens[pos]))
      throw ImageInputError("extra media markers in message text", "messages");
  validate_image_inputs(out.images, out.tokens.size(), spec_.video_id >= 0);
  return out;
}
}  // namespace dgpp::serve
