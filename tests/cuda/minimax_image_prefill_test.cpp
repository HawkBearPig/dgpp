#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "models/minimax/forward.hpp"
#include "models/minimax/vision.hpp"

namespace {
void require(bool ok, const char* why) {
  if (!ok) throw std::runtime_error(why);
}
dgpp::ImageInput image(int64_t offset, int width, int height, int seed) {
  dgpp::ImageInput im;
  im.offset = offset;
  im.width = width;
  im.height = height;
  im.tokens = width / 28 * (height / 28);
  im.rgb.resize(size_t(width) * height * 3);
  for (size_t i = 0; i < im.rgb.size(); ++i) im.rgb[i] = (i * 17 + seed) % 256;
  return im;
}
void encoder_cache(const std::string& path) {
  const auto cfg = dgpp::MinimaxConfig::from_json_file(path + "/config.json");
  dgpp::MinimaxVisionEncoder encoder(*cfg.vision, path, nullptr);
  const int h = cfg.hidden_size;
  auto read = [&](const dgpp::ImageInput& im, bool fresh = false) {
    const auto* data =
        fresh ? encoder.encode(im) : encoder.stage({im}, im.offset, im.offset + im.tokens);
    std::vector<uint16_t> result(size_t(im.tokens) * h);
    DGPP_CUDA_OK(cudaMemcpy(result.data(), data, result.size() * 2, cudaMemcpyDeviceToHost));
    return result;
  };
  std::vector<dgpp::ImageInput> images;
  std::vector<std::vector<uint16_t>> expected;
  for (int i = 0; i < cfg.vision->kEmbeddingCacheSlots; ++i) {
    images.push_back(image(0, 56, 56, i * 23));
    expected.push_back(read(images.back()));
  }
  const auto filled = encoder.encode_count();
  for (size_t i = 0; i < images.size(); ++i)
    require(read(images[i]) == expected[i], "interleaved encoder cache changed embeddings");
  auto shifted = images.front();
  shifted.offset = 73;
  require(read(shifted) == expected.front() && encoder.encode_count() == filled,
          "pixel-identical images reencoded after interleaving or prompt relocation");
  read(image(0, 56, 56, 241));
  require(encoder.encode_count() == filled + 1, "new pixels did not encode");
  require(read(images[1]) == expected[1] && encoder.encode_count() == filled + 2,
          "evicted encoder entry was stale or did not reencode");
  auto geometry = images.front();
  geometry.width = 28;
  geometry.height = 112;
  require(read(geometry) == read(geometry, true), "encoder cache ignored changed patch geometry");
  auto video = images.front();
  video.frames = 2;
  video.frame_offsets = {0};
  video.rgb.insert(video.rgb.end(), images[1].rgb.begin(), images[1].rgb.end());
  const auto original = read(video);
  std::fill(video.rgb.begin() + video.rgb.size() / 2, video.rgb.end(), 0);
  const auto changed = read(video);
  require(changed != original && changed == read(video, true),
          "encoder cache ignored a changed later video frame");
}
std::vector<int32_t> check(const std::string& path, bool draft, bool video = false) {
  const auto cfg = dgpp::MinimaxConfig::from_json_file(path + "/config.json");
  dgpp::MinimaxModel model(cfg, path, 512, 1024, dgpp::LoaderResidency::Resident, nullptr, 0, 1, 2,
                           draft, 16);
  require(model.supports_images() && model.vision_digest() != 0, "image capability and identity");
  std::vector<dgpp::ImageInput> images{image(126, 84, 56, 11), image(190, 56, 56, 31)};
  if (video) {
    auto& segment = images.front();
    segment.frames = 8;
    segment.frame_offsets = {126, 145, 164, 183};
    segment.tokens *= 4;
    for (int f = 1; f < segment.frames; ++f) {
      const auto frame = image(0, 84, 56, 11 + f * 31);
      segment.rgb.insert(segment.rgb.end(), frame.rgb.begin(), frame.rgb.end());
    }
    images.back().offset = 205;
  }
  std::vector<int64_t> ids(257);
  for (size_t i = 0; i < ids.size(); ++i) ids[i] = (i * 13 + 9) % 400;
  for (const auto& im : images)
    for (int span = 0; span < im.spans(); ++span)
      std::fill_n(ids.begin() + im.span_offset(span), im.span_tokens(),
                  im.is_video() ? cfg.vision->video_token : cfg.vision->tokens.pad);
  auto baseline = model.session_prefill_images(0, ids, images);
  const auto next = model.session_step(0, 7);
  model.session_close(0);
  for (int chunk : {1, 7, 64, 128}) {
    auto cursor = model.session_prefill_begin(0, ids, ids.size(), chunk, {}, nullptr, 0, &images);
    while (!model.session_prefill_advance(cursor, chunk)) {
    }
    require(model.session_step(0, 7).logits == next.logits, "image chunk boundary changed logits");
    model.session_close(0);
  }
  // Snapshot in the middle of the first image, then restore on another slot.
  uint8_t* snapshot = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&snapshot, model.session_snapshot_bytes()));
  dgpp::MinimaxModel::SessionSnapshotMeta meta;
  dgpp::MinimaxModel::SnapshotRequest request;
  request.position = 129;
  request.dst = snapshot;
  request.meta = &meta;
  const auto cold = model.session_prefill_images(0, ids, images, {129}, &request);
  require(request.taken, "partial image snapshot missing");
  const auto cold_step = model.session_step(0, 7);
  model.session_close(0);
  model.session_attach(1, snapshot, meta);
  const auto hot = model.session_prefill_resume_images(1, {ids.begin() + 129, ids.end()}, images,
                                                       {129}, nullptr);
  require(cold.logits == hot.logits && cold_step.logits == model.session_step(1, 7).logits,
          "image prefix restore changed target/draft state");
  model.session_close(1);
  model.session_release_snapshot(meta);
  DGPP_CUDA_OK(cudaFree(snapshot));
  // The encoder result cache must use pixels, including pixels outside an
  // attached prefix. Distinct images must affect the language model.
  auto changed = images;
  for (auto& im : changed) std::fill(im.rgb.begin(), im.rgb.end(), 0);
  const auto other = model.session_prefill_images(1, ids, changed);
  require(other.logits != baseline.logits, "visual pixels were ignored or stale");
  model.session_close(1);
  auto a = model.session_prefill_begin(0, ids, ids.size(), 31, {}, nullptr, 0, &images);
  auto b = model.session_prefill_begin(1, ids, ids.size(), 17, {}, nullptr, 0, &changed);
  while (a.next < a.end || b.next < b.end) {
    std::vector<dgpp::MinimaxModel::PrefillCursor*> cursors;
    std::vector<int64_t> budgets;
    if (a.next < a.end) {
      cursors.push_back(&a);
      budgets.push_back(31);
    }
    if (b.next < b.end) {
      cursors.push_back(&b);
      budgets.push_back(17);
    }
    model.session_prefill_advance_group(cursors, budgets);
  }
  require(model.session_step(0, 7).logits == next.logits,
          "image state leaked across interleaved slots");
  model.session_close(0);
  model.session_close(1);
  require(model.kv_blocks_in_use() == 0, "image sessions leaked KV blocks");
  // Text requests after images must use ordinary embeddings on every slot.
  const auto text = model.session_prefill(0, {7, 11, 13});
  model.session_close(0);
  require(text.logits == model.session_prefill(1, {7, 11, 13}).logits,
          "stale image scope on text request");
  model.session_close(1);
  auto bad = ids;
  bad[images[0].offset] = 7;
  bool refused = false;
  try {
    model.session_prefill_images(0, bad, images);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused && model.kv_blocks_in_use() == 0, "malformed image span admitted");
  auto out = model.session_prefill_images(0, ids, images);
  std::vector<int32_t> tokens;
  auto pick = [](const auto& logits) {
    return int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
  };
  int32_t bonus = pick(out.logits);
  while (tokens.size() < 24) {
    tokens.push_back(bonus);
    if (!draft) {
      bonus = pick(model.session_step(0, bonus).logits);
      continue;
    }
    std::vector<int32_t> proposals;
    require(model.dflash2_draft(0, bonus, &proposals), "image draft failed");
    std::vector<int64_t> block{bonus};
    block.insert(block.end(), proposals.begin(), proposals.end());
    const auto verify = model.session_verify(0, block);
    int accepted = 1;
    for (size_t row = 0; row < block.size(); ++row) {
      const auto first = verify.logits.begin() + row * cfg.vocab_size;
      bonus = std::max_element(first, first + cfg.vocab_size) - first;
      if (row + 1 == block.size() || bonus != block[row + 1] || tokens.size() >= 24) break;
      tokens.push_back(bonus);
      ++accepted;
    }
    model.session_rollback(0, accepted, block.size());
  }
  model.session_close(0);
  tokens.resize(24);
  return tokens;
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: minimax_image_prefill_test CHECKPOINT");
    encoder_cache(argv[1]);
    require(check(argv[1], false) == check(argv[1], true),
            "plain/speculative image transcripts differ");
    require(check(argv[1], false, true) == check(argv[1], true, true),
            "plain/speculative video transcripts differ");
    std::cout
        << "MiniMax image/video prefill, chunking, snapshots, isolation and speculation pass\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
