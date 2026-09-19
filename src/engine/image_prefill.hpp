#pragma once

#include <stdexcept>
#include <vector>

#include "common/image_input.hpp"

namespace dgpp {

template <class Model>
inline constexpr bool kImagePrefixCache = requires(
    Model& model, int req, const std::vector<int64_t>& ids,
    const std::vector<ImageInput>& images, const std::vector<int64_t>& boundaries,
    typename Model::SnapshotRequest* snapshot) {
  model.session_prefill_images(req, ids, images, boundaries, snapshot);
  model.session_prefill_resume_images(req, ids, images, boundaries, snapshot);
};

// Both engine adapters use the same image-aware cold/attached model path.
template <class Model>
typename Model::Outputs cached_model_prefill(
    Model* model, int req, const std::vector<int64_t>& ids,
    const std::vector<int64_t>& boundaries, typename Model::SnapshotRequest* snapshot,
    const std::vector<ImageInput>* images, bool resume) {
  if (images && !images->empty()) {
    if constexpr (kImagePrefixCache<Model>) {
      return resume ? model->session_prefill_resume_images(req, ids, *images, boundaries, snapshot)
                    : model->session_prefill_images(req, ids, *images, boundaries, snapshot);
    } else {
      throw std::logic_error("model does not support cached image prefill");
    }
  }
  return resume ? model->session_prefill_resume(req, ids, boundaries, snapshot)
                : model->session_prefill(req, ids, boundaries, snapshot);
}

}  // namespace dgpp
