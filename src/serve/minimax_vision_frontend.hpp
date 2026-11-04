#pragma once
#include "serve/vision_frontend.hpp"

namespace dgpp::serve {
class MinimaxVisionFrontend : public VisionFrontend {
 public:
  MinimaxVisionFrontend(const dgpp::text::Tokenizer* tok, const dgpp::text::ChatTemplate* tpl,
                        ImageTokens ids)
      : VisionFrontend(tok, tpl,
                       VisionSpec{ids, "]<]start of image[>[", "]<]image[>[", "]<]end of image[>[",
                                  ImagePreprocess{28, 4, 576, false, 256}},
                       true) {}
};
}  // namespace dgpp::serve
