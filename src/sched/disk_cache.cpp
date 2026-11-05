#include "sched/disk_cache.hpp"

#include <algorithm>
#include <cstring>
#include <map>
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

bool DiskCache::consistent(std::string* why) const {
  const auto fail = [&](const std::string& what) {
    if (why != nullptr) *why = what;
    return false;
  };
  if (!enabled()) return true;
  std::vector<uint8_t> seen(static_cast<size_t>(cfg_.pages), 0);
  std::vector<int> refs(blocks_.size(), 0);
  int64_t used = 0;
  const auto take = [&](int64_t page, const std::string& what) {
    if (page < 0 || page >= cfg_.pages) return fail(what + ": page " + std::to_string(page) + " outside the slab");
    if (seen[static_cast<size_t>(page)]) return fail(what + ": page " + std::to_string(page) + " used twice");
    seen[static_cast<size_t>(page)] = 1;
    ++used;
    return true;
  };
  for (size_t i = 0; i < entries_.size(); ++i) {
    const Entry& e = entries_[i];
    if (!e.live) continue;
    const std::string what = "entry " + std::to_string(i);
    if (e.position <= 0) return fail(what + ": empty");
    if (static_cast<int64_t>(e.ids.size()) != e.position) return fail(what + ": ids do not cover the position");
    if (e.ids_pages != ids_pages_for(e.position)) return fail(what + ": ids pages differ from the position's");
    if (e.blob_page < 0) return fail(what + ": no blob run");
    for (int64_t q = e.blob_page; q < e.blob_page + cfg_.blob_pages + e.ids_pages; ++q)
      if (!take(q, what + " run")) return false;
    const int64_t n_full = cfg_.block_tokens > 0 ? e.position / cfg_.block_tokens : 0;
    const bool partial = cfg_.block_tokens > 0 && e.position % cfg_.block_tokens != 0;
    if (static_cast<int64_t>(e.blocks.size()) != n_full) return fail(what + ": block records do not cover the full blocks");
    if (partial != (e.partial >= 0)) return fail(what + ": the partial record does not match the position");
    for (const int b : e.blocks) {
      if (b < 0 || static_cast<size_t>(b) >= blocks_.size() || !blocks_[static_cast<size_t>(b)].live)
        return fail(what + ": a dead block record");
      ++refs[static_cast<size_t>(b)];
    }
    if (e.partial >= 0) {
      if (static_cast<size_t>(e.partial) >= blocks_.size() || !blocks_[static_cast<size_t>(e.partial)].live)
        return fail(what + ": a dead partial record");
      ++refs[static_cast<size_t>(e.partial)];
    }
    bool indexed = false;
    const auto range = by_hash_.equal_range(e.hash);
    for (auto it = range.first; it != range.second; ++it) indexed = indexed || it->second == static_cast<int>(i);
    if (!indexed) return fail(what + ": missing from the hash index");
    if (e.state == State::kResident && e.op != 0) return fail(what + ": resident with an op");
    if (e.state != State::kResident && e.op == 0) return fail(what + ": in flight without an op");
  }
  for (size_t b = 0; b < blocks_.size(); ++b) {
    const Block& blk = blocks_[b];
    if (!blk.live) {
      if (refs[b] != 0) return fail("block " + std::to_string(b) + ": dead but referenced");
      continue;
    }
    if (blk.refs != refs[b])
      return fail("block " + std::to_string(b) + ": refcount " + std::to_string(blk.refs) + " against " +
                  std::to_string(refs[b]) + " live references");
    if (blk.refs <= 0) return fail("block " + std::to_string(b) + ": live with no references");
    if (!take(blk.page, "block " + std::to_string(b))) return false;
    for (const uint64_t id : blk.identities) {
      const auto it = by_identity_.find(id);
      if (it != by_identity_.end() && it->second != static_cast<int>(b) &&
          blocks_[static_cast<size_t>(it->second)].live && !blocks_[static_cast<size_t>(it->second)].ready)
        continue;  // a newer, not yet ready record may shadow an identity
    }
  }
  for (int64_t p = 0; p < cfg_.pages; ++p)
    if ((used_[static_cast<size_t>(p)] != 0) != (seen[static_cast<size_t>(p)] != 0))
      return fail("page " + std::to_string(p) + (used_[static_cast<size_t>(p)] ? " marked used but unowned" : " owned but not marked"));
  if (used != used_pages_) return fail("used pages " + std::to_string(used_pages_) + " against " + std::to_string(used) + " owned");
  return true;
}

int DiskCache::ids_pages_for(int64_t position) const {
  if (cfg_.page_bytes == 0 || position <= 0) return 0;
  const size_t bytes = static_cast<size_t>(position) * sizeof(int64_t);
  return static_cast<int>((bytes + cfg_.page_bytes - 1) / cfg_.page_bytes);
}

int DiskCache::live_entries() const {
  int n = 0;
  for (const Entry& e : entries_) n += e.live ? 1 : 0;
  return n;
}

int DiskCache::pinned_entries() const {
  int n = 0;
  for (const Entry& e : entries_) n += e.live && !e.key.empty() ? 1 : 0;
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
  if (e.blob_page >= 0) give_run(e.blob_page, cfg_.blob_pages + e.ids_pages);
  e.ids_pages = 0;
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
  e.key.clear();
  std::vector<int64_t>().swap(e.ids);
  PrefixCache::Images().swap(e.images);
}

// ---- spills -------------------------------------------------------------------

DiskCache::SpillPlan DiskCache::begin_spill(const int64_t* ids, int64_t position, int64_t next_token,
                                            const PrefixCache::Images& images, int64_t mtp_position,
                                            const std::vector<uint64_t>& identities,
                                            uint64_t partial_identity, uint64_t op, uint64_t now,
                                            int memory_entry, const std::string& key) {
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
  // The blob's run carries the token ids behind it (the slab outlives
  // the process; the index beside it names only pages).
  const int ids_pages = ids_pages_for(position);
  const int64_t run = cfg_.blob_pages + ids_pages;
  if (new_pages + run > cfg_.pages) {
    undo_shares();
    ++stats_.spill_skipped;
    return plan;
  }
  // Room: evict LRU resident entries until the pages fit and a blob run
  // exists (fragmentation can deny a run with enough pages free).
  int64_t blob_page = -1;
  for (;;) {
    if (free_pages() >= new_pages + run) {
      blob_page = take_run(run);
      if (blob_page >= 0) break;
    }
    Evicted victim;
    int memory = -1;
    // The victim's position is read before the eviction kills the record.
    const int candidate = lru_victim();
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
  e.ids_pages = ids_pages;
  e.state = State::kSpilling;
  e.op = op;
  e.memory = memory_entry;
  e.last_use = now;
  e.live = true;
  e.key = key;
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
  stats_.pages_written += run;
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
  plan.ids_page = blob_page + cfg_.blob_pages;
  plan.ids_pages = ids_pages;
  return plan;
}

// ---- across restarts ------------------------------------------------------------

namespace {
constexpr char kIndexMagic[8] = {'D', 'G', 'P', 'P', 'N', 'V', 'X', '1'};
constexpr uint64_t kAdoptedIdentity = uint64_t{1} << 63;  // no live block carries the top bit

void put_u64(std::vector<uint8_t>* out, uint64_t v) {
  for (int i = 0; i < 8; ++i) out->push_back(static_cast<uint8_t>(v >> (8 * i)));
}
void put_i64(std::vector<uint8_t>* out, int64_t v) { put_u64(out, static_cast<uint64_t>(v)); }
void put_u32(std::vector<uint8_t>* out, uint32_t v) {
  for (int i = 0; i < 4; ++i) out->push_back(static_cast<uint8_t>(v >> (8 * i)));
}
struct Reader {
  const std::vector<uint8_t>& b;
  size_t at = 0;
  bool bad = false;
  uint64_t u64() {
    if (at + 8 > b.size()) {
      bad = true;
      return 0;
    }
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(b[at + static_cast<size_t>(i)]) << (8 * i);
    at += 8;
    return v;
  }
  int64_t i64() { return static_cast<int64_t>(u64()); }
  uint32_t u32() {
    if (at + 4 > b.size()) {
      bad = true;
      return 0;
    }
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(b[at + static_cast<size_t>(i)]) << (8 * i);
    at += 4;
    return v;
  }
  std::string str(uint32_t n) {
    if (at + n > b.size()) {
      bad = true;
      return {};
    }
    std::string s(reinterpret_cast<const char*>(b.data() + at), n);
    at += n;
    return s;
  }
};
}  // namespace

std::vector<int> DiskCache::serialized_order() const {
  std::vector<int> order;
  for (size_t i = 0; i < entries_.size(); ++i) {
    const Entry& e = entries_[i];
    if (!e.live || e.state == State::kSpilling || !e.images.empty()) continue;
    order.push_back(static_cast<int>(i));
  }
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return entries_[static_cast<size_t>(a)].last_use < entries_[static_cast<size_t>(b)].last_use;
  });
  return order;
}

std::vector<uint8_t> DiskCache::serialize() const {
  Header h;
  h.pages = static_cast<uint64_t>(cfg_.pages);
  h.page_bytes = cfg_.page_bytes;
  h.blob_pages = static_cast<uint64_t>(cfg_.blob_pages);
  h.block_tokens = static_cast<uint64_t>(cfg_.block_tokens);
  std::vector<Record> records;
  for (const int i : serialized_order()) {
    const Entry& e = entries_[static_cast<size_t>(i)];
    Record r;
    r.position = e.position;
    r.next_token = e.next_token;
    r.mtp_position = e.mtp_position;
    r.blob_page = e.blob_page;
    r.ids_pages = static_cast<uint32_t>(e.ids_pages);
    r.key = e.key;
    for (const int b : e.blocks) r.block_pages.push_back(blocks_[static_cast<size_t>(b)].page);
    r.partial_page = e.partial >= 0 ? blocks_[static_cast<size_t>(e.partial)].page : -1;
    records.push_back(std::move(r));
  }
  return write(h, records);
}

std::vector<uint64_t> DiskCache::content_digests() const {
  std::vector<uint64_t> out;
  for (const int i : serialized_order()) out.push_back(entries_[static_cast<size_t>(i)].content);
  return out;
}

std::vector<int64_t> DiskCache::entry_pages(int entry) const {
  const Entry& e = entries_.at(static_cast<size_t>(entry));
  std::vector<int64_t> pages;
  for (int64_t q = 0; q < cfg_.blob_pages + e.ids_pages; ++q) pages.push_back(e.blob_page + q);
  for (const int b : e.blocks) pages.push_back(blocks_[static_cast<size_t>(b)].page);
  if (e.partial >= 0) pages.push_back(blocks_[static_cast<size_t>(e.partial)].page);
  return pages;
}

void DiskCache::set_content(int entry, uint64_t digest) {
  entries_.at(static_cast<size_t>(entry)).content = digest;
}

std::vector<int64_t> DiskCache::record_pages(const Record& r, int64_t blob_pages) {
  std::vector<int64_t> pages;
  for (int64_t q = 0; q < blob_pages + static_cast<int64_t>(r.ids_pages); ++q) pages.push_back(r.blob_page + q);
  for (const int64_t p : r.block_pages) pages.push_back(p);
  if (r.partial_page >= 0) pages.push_back(r.partial_page);
  return pages;
}

std::vector<uint8_t> DiskCache::write(const Header& h, const std::vector<Record>& records) {
  std::vector<uint8_t> out;
  out.insert(out.end(), kIndexMagic, kIndexMagic + 8);
  put_u64(&out, h.pages);
  put_u64(&out, h.page_bytes);
  put_u64(&out, h.blob_pages);
  put_u64(&out, h.block_tokens);
  put_u32(&out, static_cast<uint32_t>(records.size()));
  for (const Record& r : records) {
    put_i64(&out, r.position);
    put_i64(&out, r.next_token);
    put_i64(&out, r.mtp_position);
    put_i64(&out, r.blob_page);
    put_u32(&out, r.ids_pages);
    put_u32(&out, static_cast<uint32_t>(r.key.size()));
    out.insert(out.end(), r.key.begin(), r.key.end());
    put_u32(&out, static_cast<uint32_t>(r.block_pages.size()));
    for (const int64_t p : r.block_pages) put_i64(&out, p);
    put_i64(&out, r.partial_page);
  }
  return out;
}

bool DiskCache::parse(const std::vector<uint8_t>& bytes, Header* header, std::vector<Record>* records) {
  records->clear();
  if (bytes.size() < 8 || std::memcmp(bytes.data(), kIndexMagic, 8) != 0) return false;
  Reader r{bytes};
  r.at = 8;
  header->pages = r.u64();
  header->page_bytes = r.u64();
  header->blob_pages = r.u64();
  header->block_tokens = r.u64();
  const uint32_t count = r.u32();
  for (uint32_t n = 0; n < count && !r.bad; ++n) {
    Record rec;
    rec.position = r.i64();
    rec.next_token = r.i64();
    rec.mtp_position = r.i64();
    rec.blob_page = r.i64();
    rec.ids_pages = r.u32();
    rec.key = r.str(r.u32());
    const uint32_t n_blocks = r.u32();
    // A block count past the bytes left is a corrupted record, not a
    // reservation to make.
    if (r.bad || static_cast<size_t>(n_blocks) > (bytes.size() - r.at) / 8) return false;
    for (uint32_t i = 0; i < n_blocks && !r.bad; ++i) rec.block_pages.push_back(r.i64());
    rec.partial_page = r.i64();
    if (r.bad) return false;
    records->push_back(std::move(rec));
  }
  return !r.bad;
}

DiskCache::Adopted DiskCache::restore(
    const std::vector<uint8_t>& bytes,
    const std::function<bool(int64_t page, int64_t count, void* dst)>& read_pages,
    const std::vector<uint64_t>* digests) {
  Adopted out;
  if (!enabled()) {
    out.error = "the tier is off";
    return out;
  }
  if (live_entries() != 0 || used_pages_ != 0) throw std::logic_error("DiskCache: adopting into a used index");
  Header h;
  std::vector<Record> records;
  const bool whole = parse(bytes, &h, &records);
  if (bytes.size() < 8 || std::memcmp(bytes.data(), kIndexMagic, 8) != 0) {
    out.error = "not an index";
    return out;
  }
  if (bytes.size() < 8 + 4 * 8 + 4) {
    out.error = "the index is truncated";
    return out;
  }
  if (h.pages != static_cast<uint64_t>(cfg_.pages) || h.page_bytes != cfg_.page_bytes ||
      h.blob_pages != static_cast<uint64_t>(cfg_.blob_pages) || h.block_tokens != static_cast<uint64_t>(cfg_.block_tokens)) {
    out.error = "the index was written for another slab geometry (" + std::to_string(h.pages) + " pages of " +
                std::to_string(h.page_bytes) + " bytes, " + std::to_string(h.blob_pages) + " blob pages, " +
                std::to_string(h.block_tokens) + "-token blocks)";
    return out;
  }
  std::map<int64_t, int> by_page;  // block records already rebuilt, by their page
  const auto in_range = [&](int64_t p) { return p >= 0 && p < cfg_.pages; };
  for (size_t n = 0; n < records.size(); ++n) {
    const Record& rec = records[n];
    const int64_t position = rec.position, next_token = rec.next_token, mtp_position = rec.mtp_position,
                  blob_page = rec.blob_page;
    const uint32_t ids_pages = rec.ids_pages;
    const std::vector<int64_t>& block_pages = rec.block_pages;
    const int64_t partial_page = rec.partial_page;
    // Validation before any mutation: geometry, ranges, and runs that
    // do not collide with what is already adopted.
    const int64_t n_full = cfg_.block_tokens > 0 ? position / cfg_.block_tokens : 0;
    const bool partial = cfg_.block_tokens > 0 && position % cfg_.block_tokens != 0;
    const int64_t run = cfg_.blob_pages + static_cast<int64_t>(ids_pages);
    bool ok = position > 0 && static_cast<int64_t>(block_pages.size()) == n_full && partial == (partial_page >= 0) &&
              static_cast<int>(ids_pages) == ids_pages_for(position) && in_range(blob_page) &&
              blob_page + run <= cfg_.pages;
    // The pages this entry claims, against the index so far AND against
    // itself: a corrupted entry may name one page twice (a block page
    // inside its own run, its partial on one of its blocks - the fuzz's
    // find of 2026-10-27).
    std::vector<int64_t> claimed;
    const auto claim = [&](int64_t q) {
      if (std::find(claimed.begin(), claimed.end(), q) != claimed.end()) return false;
      claimed.push_back(q);
      return true;
    };
    for (int64_t q = blob_page; ok && q < blob_page + run; ++q) ok = !used_[static_cast<size_t>(q)] && claim(q);
    for (const int64_t p : block_pages) {
      if (!ok) break;
      // A block page is shared only with another block record; within the
      // entry a page appears once.
      ok = in_range(p) && (!used_[static_cast<size_t>(p)] || by_page.count(p) != 0) && claim(p);
    }
    if (ok && partial_page >= 0)
      ok = in_range(partial_page) && !used_[static_cast<size_t>(partial_page)] && claim(partial_page);
    std::vector<int64_t> ids;
    if (ok) {
      std::vector<uint8_t> raw(static_cast<size_t>(ids_pages) * cfg_.page_bytes);
      ok = read_pages(blob_page + cfg_.blob_pages, static_cast<int64_t>(ids_pages), raw.data());
      if (ok) {
        ids.resize(static_cast<size_t>(position));
        std::memcpy(ids.data(), raw.data(), static_cast<size_t>(position) * sizeof(int64_t));
      }
    }
    if (!ok) {
      ++out.dropped;
      continue;
    }
    Entry e;
    e.ids = std::move(ids);
    e.position = position;
    e.next_token = next_token;
    e.mtp_position = mtp_position;
    e.hash = PrefixCache::with_images(PrefixCache::hash_prefix(e.ids.data(), position), position, {});
    e.blob_page = blob_page;
    e.ids_pages = static_cast<int>(ids_pages);
    e.state = State::kResident;
    e.key = rec.key;
    e.content = digests != nullptr && n < digests->size() ? (*digests)[n] : 0;
    e.last_use = 0;  // older than anything this run makes; the serialized order breaks ties
    e.live = true;
    for (int64_t q = blob_page; q < blob_page + run; ++q) {
      used_[static_cast<size_t>(q)] = 1;
      ++used_pages_;
    }
    for (const int64_t p : block_pages) {
      const auto it = by_page.find(p);
      if (it != by_page.end()) {
        ++blocks_[static_cast<size_t>(it->second)].refs;
        e.blocks.push_back(it->second);
        continue;
      }
      used_[static_cast<size_t>(p)] = 1;
      ++used_pages_;
      const int b = new_block(p, kAdoptedIdentity | static_cast<uint64_t>(blocks_.size()));
      blocks_[static_cast<size_t>(b)].ready = true;
      by_page[p] = b;
      e.blocks.push_back(b);
    }
    if (partial_page >= 0) {
      used_[static_cast<size_t>(partial_page)] = 1;
      ++used_pages_;
      e.partial = new_block(partial_page, kAdoptedIdentity | static_cast<uint64_t>(blocks_.size()));
      blocks_[static_cast<size_t>(e.partial)].ready = true;
    }
    entries_.push_back(std::move(e));
    by_hash_.emplace(entries_.back().hash, static_cast<int>(entries_.size()) - 1);
    ++stats_.adopted;
    ++out.entries;
  }
  if (!whole) out.error = "the index is truncated";
  return out;
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
  stats_.pages_read -= static_cast<int64_t>(e.blocks.size()) + (e.partial >= 0 ? 1 : 0) + cfg_.blob_pages;
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

bool DiskCache::pins_protect() const {
  if (cfg_.pin_max_pct >= 100) return true;
  if (cfg_.pin_max_pct <= 0) return false;
  // The pages the pinned resident entries reference, each once.
  std::vector<uint8_t> seen(static_cast<size_t>(cfg_.pages), 0);
  int64_t pinned = 0;
  for (size_t i = 0; i < entries_.size(); ++i) {
    const Entry& e = entries_[i];
    if (!e.live || e.key.empty()) continue;
    for (const int64_t p : entry_pages(static_cast<int>(i)))
      if (p >= 0 && p < cfg_.pages && !seen[static_cast<size_t>(p)]) {
        seen[static_cast<size_t>(p)] = 1;
        ++pinned;
      }
  }
  return pinned * 100 <= static_cast<int64_t>(cfg_.pin_max_pct) * cfg_.pages;
}

int DiskCache::lru_victim() const {
  // The least recently used resident unpinned entry; a pinned one only
  // when every resident entry is pinned — or when the pinned entries hold
  // more of the slab than the cap allows, where a pin protects nothing.
  const bool protect = pins_protect();
  int victim = -1;
  for (size_t i = 0; i < entries_.size(); ++i) {
    const Entry& e = entries_[i];
    if (!e.live || e.state != State::kResident) continue;
    if (victim < 0) {
      victim = static_cast<int>(i);
      continue;
    }
    const Entry& v = entries_[static_cast<size_t>(victim)];
    const bool pinned = protect && !e.key.empty(), victim_pinned = protect && !v.key.empty();
    if ((victim_pinned && !pinned) || (pinned == victim_pinned && e.last_use < v.last_use))
      victim = static_cast<int>(i);
  }
  return victim;
}

int DiskCache::evict_lru(int* memory) {
  const int victim = lru_victim();
  if (memory != nullptr) *memory = victim < 0 ? -1 : entries_[static_cast<size_t>(victim)].memory;
  if (victim < 0) return -1;
  ++stats_.evictions;
  kill_entry(victim);
  return victim;
}

}  // namespace dgpp::sched
