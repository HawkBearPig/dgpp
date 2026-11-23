#include "serve/inkling_vision_frontend.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dgpp::serve {

InklingVisionFrontend::InklingVisionFrontend(const dgpp::text::Tokenizer* tok, const dgpp::text::ChatTemplate* tpl,
                                             ImageTokens ids, ImagePreprocess preprocess)
    : TextFrontend(tok, tpl), ids_(ids), preprocess_(preprocess) {
  const auto single = [&](const char* text, const char* what) {
    const auto encoded = encode_text(text);
    if (encoded.size() != 1)
      throw std::invalid_argument(std::string("Inkling vision frontend: ") + what + " is not one token of this tokenizer");
    return encoded.front();
  };
  const int64_t start = single("<|content_image|>", "the image block marker");
  const int64_t pad = single("<|unused_200054|>", "the image placeholder");
  const int64_t end = single("<|end_message|>", "the end-of-message marker");
  if (ids_.pad != pad)
    throw std::invalid_argument("Inkling vision frontend: the template's image placeholder is not the checkpoint's image token id");
  if ((ids_.start >= 0 && ids_.start != start) || (ids_.end >= 0 && ids_.end != end))
    throw std::invalid_argument("Inkling vision frontend: the engine's image block markers are not the tokenizer's");
  ids_.start = start;
  ids_.end = end;
  if (!preprocess_.native_size || preprocess_.grid != kInklingImageGrid)
    throw std::invalid_argument("Inkling vision frontend: the processor keeps an image's size on a 40-pixel grid");
}

ModelFrontend::ChatInput InklingVisionFrontend::prepare_chat(const minijson::Value& globals) const {
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
        const std::string where = "messages[" + std::to_string(i) + "].content[" + std::to_string(j) + "].image_url";
        ImageInput image = prepare_image(part.at("image_url"), where, preprocess_);
        if (image.rgb.size() > kMaxRequestImageBytes - image_bytes)
          throw ImageInputError("decoded media exceeds the 256 MiB request byte limit", where);
        image_bytes += image.rgb.size();
        out.images.push_back(std::move(image));
        // The template reads the part's type and nothing else: the data URI
        // stays out of the render.
        parts.push_back(V::make_object({{"type", V::make_string("image_url")}}));
      }
      fields.push_back({field.key, V::make_array(std::move(parts))});
    }
    messages.push_back(V::make_object(std::move(fields)));
  }
  if (out.images.empty()) {
    out.tokens = encode_text(render_chat(globals));
    return out;
  }
  std::vector<minijson::Member> fields;
  for (const auto& f : globals.members())
    fields.push_back(f.key == "messages" ? minijson::Member{f.key, V::make_array(messages)} : f);
  const std::vector<int64_t> rendered = encode_text(render_chat(V::make_object(std::move(fields))));

  // One placeholder per image block, in order. A placeholder that is not
  // inside an image block came from message text, and so did any beyond the
  // images the request carries.
  size_t visual = 0;
  for (const auto& im : out.images) visual += static_cast<size_t>(im.tokens);
  out.tokens.reserve(rendered.size() + visual);
  size_t next = 0;
  for (size_t p = 0; p < rendered.size(); ++p) {
    if (rendered[p] != ids_.pad) {
      out.tokens.push_back(rendered[p]);
      continue;
    }
    if (p == 0 || rendered[p - 1] != ids_.start || p + 1 >= rendered.size() || rendered[p + 1] != ids_.end)
      throw ImageInputError("image markers in message text conflict with image inputs", "messages");
    if (next == out.images.size()) throw ImageInputError("extra image markers in message text", "messages");
    ImageInput& im = out.images[next++];
    im.offset = static_cast<int64_t>(out.tokens.size());
    out.tokens.insert(out.tokens.end(), static_cast<size_t>(im.tokens), ids_.pad);
  }
  if (next != out.images.size())
    throw ImageInputError("image token expansion did not match image inputs", "messages");
  validate_image_inputs(out.images, out.tokens.size());
  return out;
}

}  // namespace dgpp::serve
