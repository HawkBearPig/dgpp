#pragma once
// Inkling 2's image front end (docs/vision.md).
//
// The shared vision front end swaps every image part for text that holds the
// family's delimiters, because those templates know nothing of images. This
// checkpoint's template renders an image part itself, as one block:
//
//   <role> <|content_image|> <|unused_200054|> <|end_message|>
//
// with a single placeholder whatever the image's size. So the image parts
// stay in the messages (without their data URIs, which the template never
// reads), the template renders, and each placeholder is then expanded to one
// token per 40 x 40 pixel cell of its image, in the order the images appear.
// The processor does not resize: see ImagePreprocess::native_size.
#include "serve/frontend.hpp"
#include "serve/image_inputs.hpp"

namespace dgpp::serve {

class InklingVisionFrontend : public TextFrontend {
 public:
  // `ids.pad` is the engine's placeholder id (the checkpoint's
  // image_token_id). The block's opening and closing markers are read from
  // the tokenizer by their text, and `ids.start` / `ids.end` are checked
  // against them when the engine names them.
  InklingVisionFrontend(const dgpp::text::Tokenizer* tok, const dgpp::text::ChatTemplate* tpl, ImageTokens ids,
                        ImagePreprocess preprocess = inkling_image_preprocess());
  bool supports_images() const override { return true; }
  ChatInput prepare_chat(const minijson::Value& globals) const override;
  const ImageTokens& image_tokens() const { return ids_; }
  const ImagePreprocess& preprocess() const { return preprocess_; }

 private:
  ImageTokens ids_;  // start: <|content_image|>, pad: the placeholder, end: <|end_message|>
  ImagePreprocess preprocess_;
};

}  // namespace dgpp::serve
