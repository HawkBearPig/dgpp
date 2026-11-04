#pragma once
// The shared vision frontend (docs/vision.md): the checkpoint's own Jinja
// template still renders roles, tools and reasoning, and each user image_url
// part is replaced by its trained delimiters plus one pad token per visual
// token before rendering. Every family needs the same three things, so the
// scan and the span bookkeeping live here: the delimiter ids, the text those
// delimiters tokenize to, and the processor geometry. The per-family specs
// live with their frontend (glm_vision_frontend.hpp, qwen_vision_frontend.hpp).
#include <string>

#include "serve/frontend.hpp"
#include "serve/image_inputs.hpp"

namespace dgpp::serve {
struct VisionSpec {
  ImageTokens ids;
  std::string start_text, pad_text, end_text;
  ImagePreprocess preprocess;
  int64_t video_id = -1;
  std::string video_text = {};
};

class VisionFrontend : public TextFrontend {
 public:
  VisionFrontend(const dgpp::text::Tokenizer* tok, const dgpp::text::ChatTemplate* tpl,
                 VisionSpec spec, bool model_opens_thinking = false);
  bool supports_images() const override { return true; }
  bool supports_video() const override;
  ChatInput prepare_chat(const minijson::Value& globals) const override;
  const VisionSpec& spec() const { return spec_; }

 private:
  VisionSpec spec_;
};

}  // namespace dgpp::serve
