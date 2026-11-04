#include <algorithm>
#include <limits>

#include "kernels/qwen_vision.hpp"
#include "models/minimax/forward.hpp"

namespace dgpp {
class MinimaxModel::ImagePrefillScope {
 public:
  ImagePrefillScope(MinimaxModel& m, int req, const std::vector<ImageInput>* images)
      : m_(m), req_(req) {
    if (req < 0 || req >= m.max_requests_)
      throw std::out_of_range("MiniMax image prefill: request slot");
    previous_ = m.request_images_[req];
    m.request_images_[req] = images && !images->empty() ? images : nullptr;
  }
  ~ImagePrefillScope() { m_.request_images_[req_] = previous_; }

 private:
  MinimaxModel& m_;
  int req_;
  const std::vector<ImageInput>* previous_;
};

void MinimaxModel::validate_images(const std::vector<int64_t>& ids,
                                   const std::vector<ImageInput>& images, int64_t start) const {
  if (!vision_) throw std::invalid_argument("MiniMax: checkpoint has no vision tower");
  if (start < 0 || ids.size() > size_t(std::numeric_limits<int64_t>::max() - start))
    throw std::invalid_argument("MiniMax image prefill: invalid position range");
  validate_image_inputs(images, size_t(start) + ids.size());
  for (const auto& im : images) {
    if (im.grid != MinimaxVisionConfig::kGrid)
      throw std::invalid_argument("MiniMax image prefill: wrong pixel grid");
    for (int64_t pos = std::max(start, im.offset); pos < im.offset + im.tokens; ++pos)
      if (ids[pos - start] != cfg_.vision->tokens.pad)
        throw std::invalid_argument("MiniMax: image span does not contain image tokens");
  }
}
MinimaxModel::Outputs MinimaxModel::session_prefill_images(int req, const std::vector<int64_t>& ids,
                                                           const std::vector<ImageInput>& images) {
  return session_prefill_images(req, ids, images, {}, nullptr);
}
MinimaxModel::Outputs MinimaxModel::session_prefill_images(int req, const std::vector<int64_t>& ids,
                                                           const std::vector<ImageInput>& images,
                                                           const std::vector<int64_t>& boundaries,
                                                           SnapshotRequest* snap) {
  validate_images(ids, images);
  ImagePrefillScope scope(*this, req, &images);
  return Base::session_prefill(req, ids, boundaries, snap);
}
MinimaxModel::Outputs MinimaxModel::session_prefill_resume_images(
    int req, const std::vector<int64_t>& ids, const std::vector<ImageInput>& images,
    const std::vector<int64_t>& boundaries, SnapshotRequest* snap) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("MiniMax image prefill: request slot");
  validate_images(ids, images, session_pos_[req]);
  ImagePrefillScope scope(*this, req, &images);
  return Base::session_prefill_resume(req, ids, boundaries, snap);
}
MinimaxModel::PrefillCursor MinimaxModel::session_prefill_begin(
    int req, const std::vector<int64_t>& ids, int64_t reserve, int64_t chunk,
    const std::vector<int64_t>& boundaries, SnapshotRequest* snap, int64_t attach,
    const std::vector<ImageInput>* images) {
  const bool present = images && !images->empty();
  if (present) validate_images(ids, *images);
  ImagePrefillScope scope(*this, req, images);
  auto cursor = Base::session_prefill_begin(req, ids, reserve, chunk, boundaries, snap, attach);
  return PrefillCursor{std::move(cursor), present ? images : nullptr};
}
bool MinimaxModel::session_prefill_advance(PrefillCursor& cursor, int64_t chunk) {
  ImagePrefillScope scope(*this, cursor.req, cursor.images);
  const bool done = Base::session_prefill_advance(cursor, chunk);
  if (done) cursor.images = nullptr;
  return done;
}
void MinimaxModel::apply_image_embeddings(uint16_t* dst, const RowRun& run) {
  if (run.req < 0 || run.req >= max_requests_) return;
  const auto* images = request_images_[run.req];
  if (!images) return;
  if (run.decode || run.capture || run.span_reqs)
    throw std::logic_error("MiniMax: image rows reached a non-prefill walk");
  const int h = cfg_.hidden_size;
  for (int64_t first = run.pos0; first < run.pos0 + run.T;) {
    const int64_t end = std::min(run.pos0 + run.T, first + MinimaxVisionConfig::kWindowTokens);
    const uint16_t* rows = nullptr;
    for (const auto& im : *images) {
      const int64_t begin = std::max(first, im.offset), stop = std::min(end, im.offset + im.tokens);
      if (begin >= stop) continue;
      if (!rows) rows = vision_->stage(*images, first, end);
      qwen_image_copy(rows + (begin - first) * h, dst + (begin - run.pos0) * h, stop - begin, h,
                      stream_);
    }
    first = end;
  }
}
}  // namespace dgpp
