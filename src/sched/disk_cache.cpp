#include "sched/disk_cache.hpp"

#include <algorithm>
#include <stdexcept>

namespace dgpp::sched {

namespace {
bool same_image(const ImageInput& a, const ImageInput& b) {
  return a.offset == b.offset && a.tokens == b.tokens && a.width == b.width &&
         a.height == b.height && a.grid == b.grid && a.rgb == b.rgb;
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

DiskCache::DiskCache(const Config& cfg) : cfg_(cfg) {
  if (cfg_.pages < 0) throw std::invalid_argument("DiskCache: negative pages");
  if (cfg_.pages > 0 && cfg_.page_bytes == 0) throw std::invalid_argument("DiskCache: zero page bytes");
  if (cfg_.blob_pages < 1) throw std::invalid_argument("DiskCache: a blob spans at least one page");
  if (cfg_.block_tokens < 0 || cfg_.min_tokens < 0)
    throw std::invalid_argument("DiskCache: negative block or minimum tokens");
  used_.assign(static_cast<size_t>(cfg_.pages), 0);
}

int DiskCache::live_entries() const {
  int n = 0;
  for (const Entry& e : entries_) n += e.live ? 1 : 0;
  return n;
}

int64_t DiskCache::live_blocks() const {
  int64_t n = 0;
  for (const Block& b : blocks_) n += b.live ? 1 : 0;
  return n;
}

std::vector<int> DiskCache::live() const {
  std::vector<int> out;
  for (size_t i = 0; i < entries_.size(); ++i)
    if (entries_[i].live) out.push_back(static_cast<int>(i));
  return out;
}

int DiskCache::find_exact(const int64_t* ids, int64_t n, uint64_t hash, const PrefixCache::Images& images,
                          int64_t next_token) const {
  const auto range = by_hash_.equal_range(hash);
  int best = -1;
  for (auto it = range.first; it != range.second; ++it) {
    const Entry& e = entries_[static_cast<size_t>(it->second)];
    if (!e.live || e.position != n) continue;
    if (e.next_token >= 0 && e.next_token != next_token) continue;
    if (!std::equal(e.ids.begin(), e.ids.end(), ids)) continue;
    const size_t count = image_count(images, n);
    if (e.images.size() != count ||
        !std::equal(e.images.begin(), e.images.end(), images.begin(), same_image_key))
      continue;
    if (best < 0 || it->second < best) best = it->second;
  }
  return best;
}

int DiskCache::lookup(const std::vector<int64_t>& prompt, const std::vector<int64_t>& cuts,
                      const std::vector<uint64_t>& cut_hashes, const PrefixCache::Images& images) const {
  if (cuts.size() != cut_hashes.size())
    throw std::invalid_argument("DiskCache::lookup: cuts and hashes differ");
  for (size_t i = cuts.size(); i-- > 0;) {
    const int64_t c = cuts[i];
    if (c <= 0 || c >= static_cast<int64_t>(prompt.size())) continue;
    const int e = find_exact(prompt.data(), c, cut_hashes[i], images, prompt[static_cast<size_t>(c)]);
    if (e < 0) continue;
    if (entries_[static_cast<size_t>(e)].state == State::kSpilling) continue;
    return e;
  }
  return -1;
}

// ---- pages ------------------------------------------------------------------

int64_t DiskCache::take_page() {
  for (int64_t k = 0; k < cfg_.pages; ++k) {
    const int64_t p = (low_cursor_ + k) % cfg_.pages;
    if (used_[static_cast<size_t>(p)]) continue;
    used_[static_cast<size_t>(p)] = 1;
    ++used_pages_;
    low_cursor_ = (p + 1) % cfg_.pages;
    return p;
  }
  return -1;
}

void DiskCache::give_page(int64_t page) {
  if (page < 0 || page >= cfg_.pages || !used_[static_cast<size_t>(page)])
    throw std::logic_error("DiskCache: freeing a page that is not in use");
  used_[static_cast<size_t>(page)] = 0;
  --used_pages_;
}

int64_t DiskCache::take_run(int64_t n) {
  // First fit from the top: blobs and blocks grow toward each other.
  int64_t run = 0;
  for (int64_t p = cfg_.pages - 1; p >= 0; --p) {
    run = used_[static_cast<size_t>(p)] ? 0 : run + 1;
    if (run == n) {
      for (int64_t q = p; q < p + n; ++q) used_[static_cast<size_t>(q)] = 1;
      used_pages_ += n;
      return p;
    }
  }
  return -1;
}

void DiskCache::give_run(int64_t start, int64_t n) {
  for (int64_t q = start; q < start + n; ++q) give_page(q);
}

// ---- blocks -----------------------------------------------------------------

int DiskCache::find_block(uint64_t identity) const {
  const auto it = by_identity_.find(identity);
  if (it == by_identity_.end()) return -1;
  const Block& b = blocks_[static_cast<size_t>(it->second)];
  return b.live ? it->second : -1;
}

int DiskCache::new_block(int64_t page, uint64_t identity) {
  int index = -1;
  for (size_t i = 0; i < blocks_.size(); ++i)
    if (!blocks_[i].live) {
      index = static_cast<int>(i);
      break;
    }
  if (index < 0) {
    blocks_.emplace_back();
    index = static_cast<int>(blocks_.size()) - 1;
  }
  Block& b = blocks_[static_cast<size_t>(index)];
  b = Block{};
  b.page = page;
  b.refs = 1;
  b.live = true;
  b.identities.push_back(identity);
  by_identity_[identity] = index;
  return index;
}

void DiskCache::release_block(int block) {
  Block& b = blocks_.at(static_cast<size_t>(block));
  if (!b.live || b.refs <= 0) throw std::logic_error("DiskCache: releasing a dead or unreferenced block");
  if (--b.refs > 0) return;
  for (const uint64_t id : b.identities) {
    const auto it = by_identity_.find(id);
    if (it != by_identity_.end() && it->second == block) by_identity_.erase(it);
  }
  give_page(b.page);
  b.live = false;
  b.identities.clear();
  b.page = -1;
}

void DiskCache::kill_entry(int entry) {
  Entry& e = entries_.at(static_cast<size_t>(entry));
  if (!e.live) throw std::logic_error("DiskCache: killing a dead entry");
  for (const int b : e.blocks) release_block(b);
  if (e.partial >= 0) release_block(e.partial);
  if (e.blob_page >= 0) give_run(e.blob_page, cfg_.blob_pages);
  const auto range = by_hash_.equal_range(e.hash);
  for (auto it = range.first; it != range.second; ++it)
    if (it->second == entry) {
      by_hash_.erase(it);
      break;
    }
  e.live = false;
  e.blocks.clear();
  e.partial = -1;
  e.blob_page = -1;
  e.memory = -1;
  std::vector<int64_t>().swap(e.ids);
  PrefixCache::Images().swap(e.images);
}

// ---- spills -------------------------------------------------------------------

DiskCache::SpillPlan DiskCache::begin_spill(const int64_t* ids, int64_t position, int64_t next_token,
                                            const PrefixCache::Images& images, int64_t mtp_position,
                                            const std::vector<uint64_t>& identities,
                                            uint64_t partial_identity, uint64_t op, uint64_t now,
                                            int memory_entry) {
  SpillPlan plan;
  if (!enabled()) return plan;
  if (position <= 0) throw std::invalid_argument("DiskCache: empty entry");
  const int64_t n_full = cfg_.block_tokens > 0 ? position / cfg_.block_tokens : 0;
  if (static_cast<int64_t>(identities.size()) != n_full)
    throw std::invalid_argument("DiskCache: the identities do not cover the entry's full blocks");
  const bool partial = cfg_.block_tokens > 0 && position % cfg_.block_tokens != 0;
  if (partial != (partial_identity != 0))
    throw std::invalid_argument("DiskCache: the partial block's identity does not match the position");
  const uint64_t h =
      PrefixCache::with_images(PrefixCache::hash_prefix(ids, position), position, images);
  if (find_exact(ids, position, h, images, next_token) >= 0) {  // already on disk
    plan.duplicate = true;
    return plan;
  }
  // Take references on every shared record first, so the evictions below
  // cannot free them; count the pages the rest need.
  std::vector<int> shared(static_cast<size_t>(n_full), -1);
  int64_t new_pages = partial ? 1 : 0;
  for (int64_t i = 0; i < n_full; ++i) {
    const int b = find_block(identities[static_cast<size_t>(i)]);
    if (b >= 0 && blocks_[static_cast<size_t>(b)].ready) {
      ++blocks_[static_cast<size_t>(b)].refs;
      shared[static_cast<size_t>(i)] = b;
    } else {
      ++new_pages;
    }
  }
  const auto undo_shares = [&] {
    for (const int b : shared)
      if (b >= 0) release_block(b);
  };
  if (new_pages + cfg_.blob_pages > cfg_.pages) {
    undo_shares();
    ++stats_.spill_skipped;
    return plan;
  }
  // Room: evict LRU resident entries until the pages fit and a blob run
  // exists (fragmentation can deny a run with enough pages free).
  int64_t blob_page = -1;
  for (;;) {
    if (free_pages() >= new_pages + cfg_.blob_pages) {
      blob_page = take_run(cfg_.blob_pages);
      if (blob_page >= 0) break;
    }
    Evicted victim;
    int memory = -1;
    // The victim's position is read before the eviction kills the record.
    int candidate = -1;
    for (size_t i = 0; i < entries_.size(); ++i) {
      const Entry& e = entries_[i];
      if (!e.live || e.state != State::kResident) continue;
      if (candidate < 0 || e.last_use < entries_[static_cast<size_t>(candidate)].last_use)
        candidate = static_cast<int>(i);
    }
    if (candidate < 0) {
      undo_shares();
      ++stats_.spill_skipped;
      return plan;
    }
    victim.entry = candidate;
    victim.position = entries_[static_cast<size_t>(candidate)].position;
    if (evict_lru(&memory) != candidate) throw std::logic_error("DiskCache: the eviction chose another victim");
    victim.memory = memory;
    plan.evicted.push_back(victim);
  }
  Entry e;
  e.ids.assign(ids, ids + position);
  e.images.assign(images.begin(), images.begin() + image_count(images, position));
  e.position = position;
  e.next_token = next_token;
  e.mtp_position = mtp_position;
  e.hash = h;
  e.blob_page = blob_page;
  e.state = State::kSpilling;
  e.op = op;
  e.memory = memory_entry;
  e.last_use = now;
  e.live = true;
  plan.pages.assign(static_cast<size_t>(n_full), -1);
  for (int64_t i = 0; i < n_full; ++i) {
    const int b = shared[static_cast<size_t>(i)];
    if (b >= 0) {
      e.blocks.push_back(b);
      ++stats_.blocks_shared;
      continue;
    }
    const int64_t page = take_page();
    if (page < 0) throw std::logic_error("DiskCache: the page count and the free pages disagree");
    e.blocks.push_back(new_block(page, identities[static_cast<size_t>(i)]));
    plan.pages[static_cast<size_t>(i)] = page;
    ++stats_.pages_written;
  }
  if (partial) {
    const int64_t page = take_page();
    if (page < 0) throw std::logic_error("DiskCache: the page count and the free pages disagree");
    e.partial = new_block(page, partial_identity);
    plan.partial_page = page;
    ++stats_.pages_written;
  }
  stats_.pages_written += cfg_.blob_pages;
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
    entries_[static_cast<size_t>(index)] = std::move(e);
  }
  by_hash_.emplace(h, index);
  ++stats_.spills;
  plan.ok = true;
  plan.entry = index;
  plan.blob_page = blob_page;
  return plan;
}

void DiskCache::commit_spill(int entry) {
  Entry& e = entries_.at(static_cast<size_t>(entry));
  if (!e.live || e.state != State::kSpilling) throw std::logic_error("DiskCache: committing an entry not spilling");
  e.state = State::kResident;
  e.op = 0;
  for (const int b : e.blocks) blocks_[static_cast<size_t>(b)].ready = true;
  if (e.partial >= 0) blocks_[static_cast<size_t>(e.partial)].ready = true;
  ++stats_.spilled;
}

void DiskCache::abort_spill(int entry) {
  Entry& e = entries_.at(static_cast<size_t>(entry));
  if (!e.live || e.state != State::kSpilling) throw std::logic_error("DiskCache: aborting an entry not spilling");
  ++stats_.spill_failed;
  kill_entry(entry);
}

// ---- restores -----------------------------------------------------------------

DiskCache::RestorePlan DiskCache::begin_restore(int entry, uint64_t op) {
  Entry& e = entries_.at(static_cast<size_t>(entry));
  if (!e.live || e.state != State::kResident) throw std::logic_error("DiskCache: restoring an entry not resident");
  RestorePlan plan;
  plan.blob_page = e.blob_page;
  for (const int b : e.blocks) plan.pages.push_back(blocks_[static_cast<size_t>(b)].page);
  if (e.partial >= 0) plan.partial_page = blocks_[static_cast<size_t>(e.partial)].page;
  e.state = State::kRestoring;
  e.op = op;
  ++stats_.restores;
  stats_.pages_read += static_cast<int64_t>(plan.pages.size()) + (e.partial >= 0 ? 1 : 0) + cfg_.blob_pages;
  return plan;
}

void DiskCache::commit_restore(int entry, int memory_entry, const std::vector<uint64_t>& identities,
                               uint64_t partial_identity, uint64_t now) {
  Entry& e = entries_.at(static_cast<size_t>(entry));
  if (!e.live || e.state != State::kRestoring) throw std::logic_error("DiskCache: committing an entry not restoring");
  e.state = State::kResident;
  e.op = 0;
  e.last_use = now;
  e.memory = memory_entry;
  ++stats_.restored;
  stats_.tokens_restored += e.position;
  if (memory_entry < 0) return;
  if (identities.size() != e.blocks.size())
    throw std::invalid_argument("DiskCache: the restored identities do not cover the entry's blocks");
  for (size_t i = 0; i < e.blocks.size(); ++i) {
    Block& b = blocks_[static_cast<size_t>(e.blocks[i])];
    b.identities.push_back(identities[i]);
    by_identity_[identities[i]] = e.blocks[i];
  }
  if (e.partial >= 0 && partial_identity != 0) {
    Block& b = blocks_[static_cast<size_t>(e.partial)];
    b.identities.push_back(partial_identity);
    by_identity_[partial_identity] = e.partial;
  }
}

void DiskCache::cancel_restore(int entry) {
  Entry& e = entries_.at(static_cast<size_t>(entry));
  if (!e.live || e.state != State::kRestoring) throw std::logic_error("DiskCache: cancelling an entry not restoring");
  e.state = State::kResident;
  e.op = 0;
  --stats_.restores;
}

void DiskCache::abort_restore(int entry) {
  Entry& e = entries_.at(static_cast<size_t>(entry));
  if (!e.live || e.state != State::kRestoring) throw std::logic_error("DiskCache: aborting an entry not restoring");
  ++stats_.restore_failed;
  kill_entry(entry);
}

void DiskCache::on_memory_evicted(int entry) {
  Entry& e = entries_.at(static_cast<size_t>(entry));
  if (e.live) e.memory = -1;
}

void DiskCache::touch(int entry, uint64_t now) {
  Entry& e = entries_.at(static_cast<size_t>(entry));
  if (e.live) e.last_use = now;
}

int DiskCache::evict_lru(int* memory) {
  int victim = -1;
  for (size_t i = 0; i < entries_.size(); ++i) {
    const Entry& e = entries_[i];
    if (!e.live || e.state != State::kResident) continue;
    if (victim < 0 || e.last_use < entries_[static_cast<size_t>(victim)].last_use)
      victim = static_cast<int>(i);
  }
  if (memory != nullptr) *memory = victim < 0 ? -1 : entries_[static_cast<size_t>(victim)].memory;
  if (victim < 0) return -1;
  ++stats_.evictions;
  kill_entry(victim);
  return victim;
}

}  // namespace dgpp::sched
