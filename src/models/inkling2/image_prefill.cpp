// Image prefills on the Inkling 2 session model (docs/vision.md): the
// request's images are borrowed for the duration of one prefill call, the
// walk's embedding rows at the images' positions are replaced by the vision
// tower's rows, and everything else is the text prefill.
#include <algorithm>
#include <limits>
#include <stdexcept>

#include "kernels/qwen_vision.hpp"
#include "models/inkling2/forward.hpp"
#include "models/inkling2/vision.hpp"

namespace dgpp {

// A request's images, visible to run_rows while one of its prefill calls
// runs. The previous borrow comes back on exit, so a later text request on
// the same slot sees none.
class Inkling2Model::ImagePrefillScope {
 public:
  ImagePrefillScope(Inkling2Model& m, int req, const std::vector<ImageInput>* images) : m_(m), req_(req) {
    if (req < 0 || req >= m.max_requests_) throw std::out_of_range("Inkling 2 image prefill: request slot");
    if (m.request_images_.empty()) m.request_images_.assign(static_cast<size_t>(m.max_requests_), nullptr);
    previous_ = m.request_images_[static_cast<size_t>(req)];
    m.request_images_[static_cast<size_t>(req)] = images && !images->empty() ? images : nullptr;
  }
  ~ImagePrefillScope() { m_.request_images_[static_cast<size_t>(req_)] = previous_; }
  ImagePrefillScope(const ImagePrefillScope&) = delete;
  ImagePrefillScope& operator=(const ImagePrefillScope&) = delete;

 private:
  Inkling2Model& m_;
  int req_;
  const std::vector<ImageInput>* previous_ = nullptr;
};

uint64_t Inkling2Model::vision_digest() const { return vision_ ? vision_->digest() : 0; }

// `ids` are the prompt's tokens from position `start` on. Every image
// position at or after `start` must hold the placeholder: the rows before it
// are an attached prefix's, already in the cache.
void Inkling2Model::validate_images(const std::vector<int64_t>& ids, const std::vector<ImageInput>& images,
                                    int64_t start) const {
  if (!vision_) throw std::invalid_argument("Inkling 2: the checkpoint has no vision tower");
  if (start < 0 || ids.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max() - start))
    throw std::invalid_argument("Inkling 2 image prefill: invalid position range");
  validate_image_inputs(images, static_cast<size_t>(start) + ids.size());
  const int grid = cfg_.vision.native_patch_size * cfg_.vision.adapter_merge_size;
  for (const ImageInput& im : images) {
    if (im.grid != grid) throw std::invalid_argument("Inkling 2 image prefill: wrong pixel grid");
    for (int64_t pos = std::max(start, im.offset); pos < im.offset + im.tokens; ++pos)
      if (ids[static_cast<size_t>(pos - start)] != cfg_.image_token_id)
        throw std::invalid_argument("Inkling 2: an image span does not hold its placeholder tokens");
  }
}

Inkling2Model::Outputs Inkling2Model::session_prefill_images(int req, const std::vector<int64_t>& ids,
                                                             const std::vector<ImageInput>& images) {
  return session_prefill_images(req, ids, images, {}, nullptr);
}

Inkling2Model::Outputs Inkling2Model::session_prefill_images(int req, const std::vector<int64_t>& ids,
                                                             const std::vector<ImageInput>& images,
                                                             const std::vector<int64_t>& boundaries,
                                                             SnapshotRequest* snap) {
  validate_images(ids, images);
  ImagePrefillScope scope(*this, req, &images);
  return Base::session_prefill(req, ids, boundaries, snap);
}

Inkling2Model::Outputs Inkling2Model::session_prefill_resume_images(int req, const std::vector<int64_t>& ids,
                                                                    const std::vector<ImageInput>& images,
                                                                    const std::vector<int64_t>& boundaries,
                                                                    SnapshotRequest* snap) {
  if (req < 0 || req >= max_requests_) throw std::out_of_range("Inkling 2 image prefill: request slot");
  validate_images(ids, images, session_pos_[static_cast<size_t>(req)]);
  ImagePrefillScope scope(*this, req, &images);
  return Base::session_prefill_resume(req, ids, boundaries, snap);
}

Inkling2Model::PrefillCursor Inkling2Model::session_prefill_begin(int req, const std::vector<int64_t>& ids,
                                                                  int64_t reserve, int64_t chunk,
                                                                  const std::vector<int64_t>& boundaries,
                                                                  SnapshotRequest* snap, int64_t attach,
                                                                  const std::vector<ImageInput>* images) {
  const bool present = images && !images->empty();
  if (present) validate_images(ids, *images);
  ImagePrefillScope scope(*this, req, images);
  auto cursor = Base::session_prefill_begin(req, ids, reserve, chunk, boundaries, snap, attach);
  return PrefillCursor{std::move(cursor), present ? images : nullptr};
}

bool Inkling2Model::session_prefill_advance(PrefillCursor& cursor, int64_t chunk) {
  ImagePrefillScope scope(*this, cursor.req, cursor.images);
  const bool done = Base::session_prefill_advance(cursor, chunk);
  if (done) cursor.images = nullptr;
  return done;
}

// The walk's first partial state is the normed embedding. An image row takes
// the tower's row in its place, without the norm. `dst` holds the run's
// rows [run.pos0, run.pos0 + run.T).
void Inkling2Model::apply_image_embeddings(uint16_t* dst, const RowRun& run) {
  if (request_images_.empty() || run.req < 0 || run.req >= max_requests_) return;
  const std::vector<ImageInput>* images = request_images_[static_cast<size_t>(run.req)];
  if (!images) return;
  if (run.decode || run.capture || run.span_reqs)
    throw std::logic_error("Inkling 2: image rows reached a walk that is not a single prefill");
  const int64_t H = cfg_.hidden_size;
  for (int64_t first = run.pos0; first < run.pos0 + run.T;) {
    const int64_t end = std::min<int64_t>(run.pos0 + run.T, first + Inkling2VisionEncoder::kWindowTokens);
    const uint16_t* rows = nullptr;
    for (const ImageInput& im : *images) {
      const int64_t begin = std::max(first, im.offset), stop = std::min(end, im.offset + im.tokens);
      if (begin >= stop) continue;
      if (!rows) rows = vision_->stage(*images, first, end);
      qwen_image_copy(rows + (begin - first) * H, dst + (begin - run.pos0) * H, static_cast<int>(stop - begin),
                      static_cast<int>(H), stream_);
    }
    first = end;
  }
}

}  // namespace dgpp
