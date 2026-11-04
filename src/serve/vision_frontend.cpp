#include "serve/vision_frontend.hpp"

#include <utility>

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
        if (!type || !type->is_string() || type->as_string() != "image_url") {
          parts.push_back(part);
          continue;
        }
        const std::string where =
            "messages[" + std::to_string(i) + "].content[" + std::to_string(j) + "].image_url";
        auto image = prepare_image(part.at("image_url"), where, spec_.preprocess);
        if (image.rgb.size() > kMaxRequestImageBytes - image_bytes)
          throw ImageInputError("decoded image data exceeds the 256 MiB request byte limit", where);
        image_bytes += image.rgb.size();
        std::string placeholder = spec_.start_text;
        for (int k = 0; k < image.tokens; ++k) placeholder += spec_.pad_text;
        placeholder += spec_.end_text;
        parts.push_back(V::make_object({{"type", V::make_string("text")},
                                        {"text", V::make_owned_string(std::move(placeholder))}}));
        out.images.push_back(std::move(image));
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
  for (auto& im : out.images) {
    while (pos < out.tokens.size() && out.tokens[pos] != spec_.ids.pad) ++pos;
    im.offset = static_cast<int64_t>(pos);
    if (pos == 0 || out.tokens[pos - 1] != spec_.ids.start)
      throw ImageInputError("image markers in message text conflict with image inputs", "messages");
    for (int j = 0; j < im.tokens; ++j)
      if (pos >= out.tokens.size() || out.tokens[pos++] != spec_.ids.pad)
        throw ImageInputError("image token expansion did not match image inputs", "messages");
    if (pos >= out.tokens.size() || out.tokens[pos] != spec_.ids.end)
      throw ImageInputError("image token expansion did not match image inputs", "messages");
  }
  for (; pos < out.tokens.size(); ++pos)
    if (out.tokens[pos] == spec_.ids.pad)
      throw ImageInputError("extra image markers in message text", "messages");
  validate_image_inputs(out.images, out.tokens.size());
  return out;
}
}  // namespace dgpp::serve
