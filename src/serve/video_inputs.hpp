#pragma once

#include "serve/image_inputs.hpp"

namespace dgpp::serve {
struct VideoInput {
  std::vector<ImageInput> segments;
  std::vector<double> timestamps;  // one timestamp per temporal patch (two frames)
};
bool native_video_available();
// Inline MP4/WebM, sampled uniformly over the complete clip at 1 fps by
// default (4..768 frames, bounded by the source frame count). MiniMax's
// temporal pairs and eight-frame attention segments are preserved.
VideoInput prepare_minimax_video(const minijson::Value&, const std::string& param,
                                 size_t byte_budget = kMaxRequestImageBytes);
}  // namespace dgpp::serve
