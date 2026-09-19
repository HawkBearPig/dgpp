#include "sched/prefix_cache.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace dgpp::sched {

namespace {
bool same_image(const ImageInput& a, const ImageInput& b) {
  return a.offset == b.offset && a.tokens == b.tokens && a.width == b.width &&
         a.height == b.height && a.rgb == b.rgb;
}
bool same_image_key(const PrefixCache::ImageKey& a, const PrefixCache::ImageKey& b) {
  return a.input == b.input || (a.hash == b.hash && same_image(*a.input, *b.input));
}
size_t image_count(const PrefixCache::Images& images, int64_t position) {
  size_t n = 0;
  while (n < images.size() && images[n].input->offset <= position) ++n;
  return n;
}
}  // namespace

PrefixCache::Images PrefixCache::image_keys(const std::vector<ImageInput>& images) const {
  Images keys;
  for (const auto& image : images) {
    uint64_t h = extend_hash(kSeed, image.offset);
    h = extend_hash(h, image.tokens);
    h = extend_hash(h, image.width);
    h = extend_hash(h, image.height);
    for (uint8_t byte : image.rgb) h = (h ^ byte) * 0x100000001b3ull;
    std::shared_ptr<const ImageInput> input;
    // Repeated conversation turns share the existing immutable image data.
    for (const auto& entry : entries_) {
      if (!entry.live) continue;
      for (const auto& key : entry.images)
        if (key.hash == h && same_image(*key.input, image)) {
          input = key.input;
          break;
        }
      if (input) break;
    }
    if (!input) input = std::make_shared<const ImageInput>(image);
    keys.push_back({std::move(input), h});
  }
  return keys;
}

uint64_t PrefixCache::with_images(uint64_t token_hash, int64_t position, const Images& images) {
  const size_t count = image_count(images, position);
  if (count == 0) return token_hash;  // preserve text-only keys
  uint64_t h = extend_hash(token_hash, -1);
  h = extend_hash(h, static_cast<int64_t>(count));
  for (size_t i = 0; i < count; ++i) h = extend_hash(h, static_cast<int64_t>(images[i].hash));
  return h;
}

PrefixCache::PrefixCache(const Config& cfg) : cfg_(cfg) {
  if (cfg_.slots < 0) throw std::invalid_argument("PrefixCache: negative slots");
  if (cfg_.align < 1) throw std::invalid_argument("PrefixCache: align must be >= 1");
  if (cfg_.chunk_tokens < 1)
    throw std::invalid_argument("PrefixCache: chunk_tokens must be >= 1");
  free_.reserve(static_cast<size_t>(cfg_.slots));
  for (int s = 0; s < cfg_.slots; ++s) free_.push_back(s);
}

int PrefixCache::live_entries() const {
  int n = 0;
  for (const Entry& e : entries_) n += e.live ? 1 : 0;
  return n;
}

size_t PrefixCache::image_bytes() const {
  std::unordered_set<const ImageInput*> seen;
  size_t bytes = 0;
  for (const auto& entry : entries_)
    if (entry.live)
      for (const auto& image : entry.images)
        if (seen.insert(image.input.get()).second) bytes += image.input->rgb.size();
  return bytes;
}

std::vector<int64_t> PrefixCache::cuts(
    int64_t n, const std::vector<int64_t>& boundaries) const {
  std::vector<int64_t> out;
  for (int64_t m = cfg_.chunk_tokens; m < n; m += cfg_.chunk_tokens) out.push_back(m);
  for (const int64_t b : boundaries) {
    const int64_t a = (b / cfg_.align) * cfg_.align;
    if (a > 0 && a < n) out.push_back(a);
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

uint64_t PrefixCache::extend_hash(uint64_t h, int64_t id) {
  const uint64_t v = static_cast<uint64_t>(id);
  for (int i = 0; i < 8; ++i) {
    h ^= (v >> (8 * i)) & 0xffu;
    h *= 0x100000001b3ull;
  }
  return h;
}

uint64_t PrefixCache::hash_prefix(const int64_t* ids, int64_t n, uint64_t seed) {
  uint64_t h = seed;
  for (int64_t i = 0; i < n; ++i) h = extend_hash(h, ids[i]);
  return h;
}

int PrefixCache::find_exact(const int64_t* ids, int64_t n, uint64_t hash, const Images& images) const {
  const auto range = by_hash_.equal_range(hash);
  int best = -1;
  for (auto it = range.first; it != range.second; ++it) {
    const Entry& e = entries_[static_cast<size_t>(it->second)];
    if (!e.live || e.position != n) continue;
    if (!std::equal(e.ids.begin(), e.ids.end(), ids)) continue;
    const size_t count = image_count(images, n);
    if (e.images.size() != count ||
        !std::equal(e.images.begin(), e.images.end(), images.begin(), same_image_key)) continue;
    // Deterministic among duplicates (there should be none): the oldest.
    if (best < 0 || it->second < best) best = it->second;
  }
  return best;
}

PrefixCache::Ghost PrefixCache::ghost_at(const std::vector<int64_t>& cuts,
                                         const std::vector<uint64_t>& cut_hashes) const {
  Ghost none;
  if (cuts.size() != cut_hashes.size()) return none;
  for (size_t i = cuts.size(); i-- > 0;)
    for (const Ghost& g : ghosts_)
      if (g.position == cuts[i] && g.hash == cut_hashes[i]) return g;
  return none;
}

PrefixCache::Nearest PrefixCache::nearest(const std::vector<int64_t>& prompt, const Images& images) const {
  Nearest best;
  for (size_t i = 0; i < entries_.size(); ++i) {
    const Entry& e = entries_[i];
    if (!e.live) continue;
    const size_t n = std::min(e.ids.size(), prompt.size());
    size_t k = 0;
    while (k < n && e.ids[k] == prompt[k]) ++k;
    size_t j = 0;
    while (j < e.images.size() && j < images.size() && same_image_key(e.images[j], images[j])) ++j;
    if (j < e.images.size()) k = std::min(k, static_cast<size_t>(e.images[j].input->offset));
    if (j < images.size()) k = std::min(k, static_cast<size_t>(images[j].input->offset));
    // The longest shared prefix; among equals the deeper entry, then the
    // older (a stable answer for the log).
    if (best.entry < 0 || static_cast<int64_t>(k) > best.common ||
        (static_cast<int64_t>(k) == best.common &&
         e.position > entries_[static_cast<size_t>(best.entry)].position)) {
      best.entry = static_cast<int>(i);
      best.common = static_cast<int64_t>(k);
    }
  }
  return best;
}

int PrefixCache::lookup(const std::vector<int64_t>& prompt,
                        const std::vector<int64_t>& cuts,
                        const std::vector<uint64_t>& cut_hashes, const Images& images) const {
  if (cuts.size() != cut_hashes.size())
    throw std::invalid_argument("PrefixCache::lookup: cuts and hashes differ");
  for (size_t i = cuts.size(); i-- > 0;) {
    const int64_t c = cuts[i];
    if (c <= 0 || c >= static_cast<int64_t>(prompt.size())) continue;
    const int e = find_exact(prompt.data(), c, cut_hashes[i], images);
    if (e >= 0) return e;
  }
  return -1;
}

int PrefixCache::take_free_slot() {
  if (free_.empty()) return -1;
  const int s = free_.front();
  free_.erase(free_.begin());
  return s;
}

void PrefixCache::give_back_slot(int slot) {
  if (slot < 0 || slot >= cfg_.slots)
    throw std::out_of_range("PrefixCache: slot out of range");
  const auto it = std::lower_bound(free_.begin(), free_.end(), slot);
  if (it != free_.end() && *it == slot)
    throw std::logic_error("PrefixCache: slot given back twice");
  free_.insert(it, slot);
}

int PrefixCache::evict_lru() {
  int victim = -1;
  for (size_t i = 0; i < entries_.size(); ++i) {
    const Entry& e = entries_[i];
    if (!e.live || e.attached > 0) continue;
    if (victim < 0 ||
        e.last_use < entries_[static_cast<size_t>(victim)].last_use)
      victim = static_cast<int>(i);
  }
  if (victim < 0) return -1;
  Entry& e = entries_[static_cast<size_t>(victim)];
  const int slot = e.slot;
  e.live = false;
  e.slot = -1;
  e.ids.clear();
  e.ids.shrink_to_fit();
  Images().swap(e.images);
  ++stats_.evictions;
  Ghost g;
  g.hash = e.hash;
  g.position = e.position;
  g.last_use = e.last_use;
  g.eviction = stats_.evictions;
  if (ghosts_.size() < kGhosts) {
    ghosts_.push_back(g);
  } else {
    ghosts_[ghost_next_] = g;
    ghost_next_ = (ghost_next_ + 1) % kGhosts;
  }
  note(3, static_cast<uint64_t>(e.position), static_cast<uint64_t>(slot));
  return slot;
}

int PrefixCache::acquire_slot() {
  const int s = take_free_slot();
  if (s >= 0) return s;
  return evict_lru();
}

int PrefixCache::insert(const int64_t* ids, int64_t position, int slot,
                        uint64_t now, const Images& images) {
  if (position <= 0) throw std::invalid_argument("PrefixCache: empty entry");
  if (slot < 0 || slot >= cfg_.slots)
    throw std::out_of_range("PrefixCache: slot out of range");
  const uint64_t h = with_images(hash_prefix(ids, position), position, images);
  if (find_exact(ids, position, h, images) >= 0) {
    ++stats_.duplicates;
    return -1;
  }
  // The index retains exact pixels for collision-safe matching. Bound their
  // union independently of the device snapshot arena. Pressure skips this
  // insertion; it neither rejects the request nor evicts an attached prefix.
  if (!images.empty()) {
    std::unordered_set<const ImageInput*> seen;
    size_t bytes = 0;
    const auto account = [&](const ImageKey& key) {
      if (seen.insert(key.input.get()).second) bytes += key.input->rgb.size();
    };
    for (const auto& entry : entries_)
      if (entry.live) for (const auto& key : entry.images) account(key);
    for (size_t i = 0; i < image_count(images, position); ++i) account(images[i]);
    if (bytes > cfg_.image_bytes) {
      ++stats_.skipped_image_bytes;
      return -1;
    }
  }
  Entry e;
  e.ids.assign(ids, ids + position);
  e.images.assign(images.begin(), images.begin() + image_count(images, position));
  e.position = position;
  e.slot = slot;
  e.hash = h;
  e.last_use = now;
  e.live = true;
  // Reuse a dead record's index when one exists (bounded memory), else
  // append; either way the choice is deterministic.
  int index = -1;
  for (size_t i = 0; i < entries_.size(); ++i)
    if (!entries_[i].live) {
      index = static_cast<int>(i);
      break;
    }
  if (index < 0) {
    entries_.push_back(std::move(e));
    index = static_cast<int>(entries_.size()) - 1;
  } else {
    // Drop the dead record's stale hash mapping.
    const uint64_t old = entries_[static_cast<size_t>(index)].hash;
    const auto range = by_hash_.equal_range(old);
    for (auto it = range.first; it != range.second; ++it)
      if (it->second == index) {
        by_hash_.erase(it);
        break;
      }
    entries_[static_cast<size_t>(index)] = std::move(e);
  }
  by_hash_.emplace(h, index);
  note(2, static_cast<uint64_t>(position), static_cast<uint64_t>(slot));
  return index;
}

void PrefixCache::attach(int index, uint64_t now) {
  Entry& e = entries_.at(static_cast<size_t>(index));
  if (!e.live) throw std::logic_error("PrefixCache: attach to a dead entry");
  ++e.attached;
  e.last_use = now;
  ++stats_.hits;
  stats_.tokens_saved += e.position;
  note(1, static_cast<uint64_t>(e.position), static_cast<uint64_t>(e.slot));
}

void PrefixCache::touch(int index, uint64_t now) {
  Entry& e = entries_.at(static_cast<size_t>(index));
  if (!e.live) throw std::logic_error("PrefixCache: touch of a dead entry");
  e.last_use = now;
}

void PrefixCache::detach(int index) {
  Entry& e = entries_.at(static_cast<size_t>(index));
  if (e.attached <= 0) throw std::logic_error("PrefixCache: detach below zero");
  --e.attached;
}

int64_t PrefixCache::blocks_pinned(int64_t block_tokens) const {
  if (block_tokens <= 0) return 0;
  int64_t n = 0;
  for (const Entry& e : entries_)
    if (e.live)
      n += e.position / block_tokens + (e.position % block_tokens != 0 ? 1 : 0);
  return n;
}

void PrefixCache::note(uint64_t a, uint64_t b, uint64_t c) {
  digest_ = extend_hash(digest_, static_cast<int64_t>(a));
  digest_ = extend_hash(digest_, static_cast<int64_t>(b));
  digest_ = extend_hash(digest_, static_cast<int64_t>(c));
}

}  // namespace dgpp::sched
