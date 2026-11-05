// The NVMe cold tier's host index (sched/disk_cache.hpp, issue #26): page
// allocation from both ends, block sharing by identity, LRU retention
// within the capacity, the spill/restore state machine and its aborts.
#include <cstring>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "sched/disk_cache.hpp"

namespace {

using dgpp::sched::DiskCache;
using dgpp::sched::PrefixCache;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

DiskCache::Config config(int64_t pages, int64_t blob_pages = 2, int64_t block_tokens = 4) {
  DiskCache::Config c;
  c.pages = pages;
  c.page_bytes = 4096;
  c.blob_pages = blob_pages;
  c.block_tokens = block_tokens;
  return c;
}

std::vector<int64_t> counted(int n, int64_t base = 100) {
  std::vector<int64_t> p(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) p[static_cast<size_t>(i)] = base + i;
  return p;
}

std::vector<uint64_t> identities(int n, uint64_t base) {
  std::vector<uint64_t> out;
  for (int i = 0; i < n; ++i) out.push_back(base + static_cast<uint64_t>(i));
  return out;
}

// The prompt's cuts and hashes the way the scheduler computes them.
void cuts_for(const std::vector<int64_t>& prompt, const std::vector<int64_t>& cut_list, std::vector<int64_t>* cuts,
              std::vector<uint64_t>* hashes) {
  *cuts = cut_list;
  hashes->clear();
  for (const int64_t c : cut_list) hashes->push_back(PrefixCache::hash_prefix(prompt.data(), c));
}

}  // namespace

DGPP_TEST(disk_cache_spill_restore_state_machine_and_lookup) {
  DiskCache d(config(/*pages=*/16));
  require(d.enabled() && d.pages_used() == 0, "empty at start");
  // An entry at 12 tokens: three full blocks (4 tokens each), no partial.
  const auto prompt = counted(20);
  const auto plan = d.begin_spill(prompt.data(), 12, -1, {}, 0, identities(3, 1000), 0, /*op=*/1, /*now=*/1,
                                  /*memory_entry=*/0);
  require(plan.ok && plan.entry == 0 && plan.pages.size() == 3 && plan.partial_page < 0, "the plan covers 3 blocks");
  require(plan.blob_page == 13 && plan.ids_page == 15 && plan.ids_pages == 1,
          "the blob run sits at the top: pages 13-14, the ids page 15");
  require(plan.pages[0] == 0 && plan.pages[1] == 1 && plan.pages[2] == 2, "block pages from the bottom");
  require(d.pages_used() == 6, "6 pages in use");
  require(d.entry(0).state == DiskCache::State::kSpilling, "spilling until the commit");
  std::vector<int64_t> cuts;
  std::vector<uint64_t> hashes;
  cuts_for(prompt, {4, 8, 12, 16}, &cuts, &hashes);
  require(d.lookup(prompt, cuts, hashes) < 0, "a spilling entry is not offered");
  d.commit_spill(0);
  require(d.entry(0).state == DiskCache::State::kResident, "resident after the commit");
  require(d.lookup(prompt, cuts, hashes) == 0, "the entry is found at its cut");
  // A prompt that diverges inside the entry misses; one with the entry as
  // a prefix at a shallower set of cuts misses too (no cut at 12).
  auto changed = prompt;
  changed[5] = 9999;
  cuts_for(changed, {4, 8, 12, 16}, &cuts, &hashes);
  require(d.lookup(changed, cuts, hashes) < 0, "a changed token inside the entry misses");
  cuts_for(prompt, {4, 8, 16}, &cuts, &hashes);
  require(d.lookup(prompt, cuts, hashes) < 0, "no cut at the entry's position: a miss");
  // The restore: pages come back; the entry is protected while restoring.
  const DiskCache::RestorePlan rp = d.begin_restore(0, /*op=*/2);
  require(rp.blob_page == 13 && rp.pages.size() == 3 && rp.pages[2] == 2, "the restore reads the same pages");
  require(d.entry(0).state == DiskCache::State::kRestoring, "restoring");
  require(d.evict_lru() < 0, "a restoring entry is never evicted");
  cuts_for(prompt, {4, 8, 12, 16}, &cuts, &hashes);
  require(d.lookup(prompt, cuts, hashes) == 0, "a restoring entry is offered (a second request waits on the op)");
  d.commit_restore(0, /*memory_entry=*/3, identities(3, 2000), 0, /*now=*/5);
  require(d.entry(0).state == DiskCache::State::kResident && d.entry(0).memory == 3, "resident, linked to memory");
  require(d.stats().restored == 1 && d.stats().tokens_restored == 12, "restore counted");
  // The fresh identities alias the records: a spill of an entry holding
  // them shares every block.
  const auto plan2 = d.begin_spill(prompt.data(), 16, -1, {}, 0, identities(4, 2000), 0, 3, 6, 4);
  require(plan2.ok && plan2.pages[0] == -1 && plan2.pages[1] == -1 && plan2.pages[2] == -1 && plan2.pages[3] >= 0,
          "three blocks shared through the restored identities, one new");
  require(d.stats().blocks_shared == 3, "shared blocks counted");
  d.commit_spill(plan2.entry);
  require(d.live_entries() == 2 && d.live_blocks() == 4, "two entries over four records");
  // Evicting the first entry keeps the shared records alive for the second.
  d.on_memory_evicted(0);
  const int victim = d.evict_lru();
  require(victim == 0, "the older entry goes first");
  require(d.live_blocks() == 4 && d.pages_used() == 4 + 2 + 1, "shared records survive, the blob run is freed");
  cuts_for(prompt, {4, 8, 12, 16}, &cuts, &hashes);
  require(d.lookup(prompt, cuts, hashes) == plan2.entry, "the deeper entry is offered now");
}

DGPP_TEST(disk_cache_aborts_release_records_and_dedupes_identical_entries) {
  DiskCache d(config(/*pages=*/12));
  const auto prompt = counted(20);
  // A partial block: 10 tokens = two full blocks and one partial.
  const auto plan = d.begin_spill(prompt.data(), 10, 7, {}, 0, identities(2, 10), /*partial=*/77, 1, 1, 0);
  require(plan.ok && plan.partial_page >= 0 && plan.pages.size() == 2, "partial block gets a page");
  require(d.pages_used() == 6, "2 + 1 + 2 blob pages + 1 ids page");
  d.abort_spill(plan.entry);
  require(d.pages_used() == 0 && d.live_entries() == 0 && d.live_blocks() == 0, "an aborted spill frees everything");
  require(d.stats().spill_failed == 1, "the failure counted");
  // An identical entry (same ids, position, lookahead token) is refused.
  const auto a = d.begin_spill(prompt.data(), 10, 7, {}, 0, identities(2, 10), 77, 2, 2, 0);
  require(a.ok, "the first spill plans");
  d.commit_spill(a.entry);
  const auto b = d.begin_spill(prompt.data(), 10, 7, {}, 0, identities(2, 10), 77, 3, 3, 1);
  require(!b.ok && d.live_entries() == 1, "an identical entry is not stored twice");
  // A different lookahead token is another entry (the MTP constraint).
  const auto c = d.begin_spill(prompt.data(), 10, 8, {}, 0, identities(2, 10), 78, 4, 4, 1);
  require(c.ok && c.pages[0] == -1 && c.pages[1] == -1 && c.partial_page >= 0,
          "the full blocks are shared, the partial is its own");
  d.commit_spill(c.entry);
  // A restore that fails drops the entry; the shared records stay with
  // the other entry.
  (void)d.begin_restore(a.entry, 5);
  d.abort_restore(a.entry);
  require(d.live_entries() == 1 && d.live_blocks() == 3, "the failed entry is gone, the shared records remain");
  require(d.stats().restore_failed == 1, "the failure counted");
  // Mismatched identities are refused loudly.
  bool threw = false;
  try {
    (void)d.begin_spill(prompt.data(), 8, -1, {}, 0, identities(1, 50), 0, 6, 6, 2);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "identities must cover every full block");
}

DGPP_TEST(disk_cache_adoption_round_trips_the_committed_index) {
  // Two committed entries sharing two block records (one pinned), a third
  // still spilling: the bytes carry the first two with their pages and
  // key, the ids come back from the slab, the uncommitted one is gone.
  DiskCache d(config(/*pages=*/64));
  std::map<int64_t, std::vector<int64_t>> slab;  // ids runs by page, as the tier would hold them
  int64_t a_ids_page = -1;
  const auto spill = [&](const std::vector<int64_t>& prompt, int64_t position, const std::vector<uint64_t>& ids,
                         uint64_t op, int memory, const std::string& key = "") {
    const auto plan = d.begin_spill(prompt.data(), position, -1, {}, 0, ids, 0, op, op, memory, key);
    require(plan.ok && plan.ids_page == plan.blob_page + 2 && plan.ids_pages == 1, "the ids run follows the blob");
    slab[plan.ids_page] = std::vector<int64_t>(prompt.begin(), prompt.begin() + position);
    if (a_ids_page < 0) a_ids_page = plan.ids_page;
    return plan.entry;
  };
  const auto a = counted(12, 1000), b = counted(12, 1000), c = counted(12, 3000);
  const int ea = spill(a, 12, identities(3, 100), 1, 0, "agent-7");
  d.commit_spill(ea);
  const int eb = spill(b, 8, identities(2, 100), 2, 1);  // shares a's first two blocks
  d.commit_spill(eb);
  (void)spill(c, 12, identities(3, 300), 3, 2);  // in flight at the "crash"
  require(d.live_entries() == 3 && d.stats().blocks_shared == 2, "three entries, two shared records");
  const int64_t committed_pages = d.pages_used() - (3 + 3);  // c's three blocks and its run
  const std::vector<uint8_t> bytes = d.serialize();
  DiskCache fresh(config(/*pages=*/64));
  const auto read = [&](int64_t page, int64_t count, void* dst) {
    const auto it = slab.find(page);
    if (it == slab.end() || count != 1) return false;
    std::memcpy(dst, it->second.data(), it->second.size() * sizeof(int64_t));
    return true;
  };
  const DiskCache::Adopted got = fresh.restore(bytes, read);
  require(got.error.empty() && got.entries == 2 && got.dropped == 0, "two entries adopted: " + got.error);
  require(fresh.live_entries() == 2 && fresh.live_blocks() == 3 && fresh.pages_used() == committed_pages,
          "the shared records and the pages came back: blocks " + std::to_string(fresh.live_blocks()) +
              ", pages " + std::to_string(fresh.pages_used()) + " vs " + std::to_string(committed_pages));
  require(fresh.pinned_entries() == 1 && fresh.stats().adopted == 2, "the key survived");
  const auto longer = counted(16, 1000);  // a's prompt with a tail: the entry at 12 is a cut of it
  const std::vector<int64_t> cuts{8, 12};
  const std::vector<uint64_t> hashes{
      PrefixCache::with_images(PrefixCache::hash_prefix(longer.data(), 8), 8, {}),
      PrefixCache::with_images(PrefixCache::hash_prefix(longer.data(), 12), 12, {})};
  const int hit = fresh.lookup(longer, cuts, hashes);
  require(hit >= 0 && fresh.entry(hit).position == 12 && fresh.entry(hit).key == "agent-7",
          "a's prompt finds the adopted entry at its deepest cut");
  // Another slab geometry refuses the bytes outright.
  DiskCache other(config(/*pages=*/32));
  require(!other.restore(bytes, read).error.empty() && other.live_entries() == 0, "another geometry adopts nothing");
  // An ids run the slab cannot read drops that entry alone.
  DiskCache lossy(config(/*pages=*/64));
  const auto read_a_only = [&](int64_t page, int64_t count, void* dst) {
    return page == a_ids_page && read(page, count, dst);
  };
  const DiskCache::Adopted partial = lossy.restore(bytes, read_a_only);
  require(partial.entries == 1 && partial.dropped == 1 && lossy.live_entries() == 1, "a bad run drops its entry");
}

DGPP_TEST(disk_cache_fuzz_corruptIndexBytesNeverBreakTheIndex) {
  // The bytes a previous run left can be anything: flipped, truncated,
  // spliced. restore() must never throw past its contract, and whatever it
  // adopts must leave a consistent index within the capacity.
  DiskCache d(config(/*pages=*/64));
  std::map<int64_t, std::vector<int64_t>> slab;
  const auto spill = [&](const std::vector<int64_t>& prompt, int64_t position, const std::vector<uint64_t>& ids,
                         uint64_t partial, uint64_t op, int memory, const std::string& key = "") {
    const auto plan = d.begin_spill(prompt.data(), position, -1, {}, 0, ids, partial, op, op, memory, key);
    require(plan.ok, "fits");
    slab[plan.ids_page] = std::vector<int64_t>(prompt.begin(), prompt.begin() + position);
    d.commit_spill(plan.entry);
  };
  spill(counted(20, 1000), 12, identities(3, 100), 0, 1, 0, "k");
  spill(counted(20, 1000), 8, identities(2, 100), 0, 2, 1);
  spill(counted(20, 3000), 10, identities(2, 300), 77, 3, 2);  // a partial block
  const std::vector<uint8_t> good = d.serialize();
  const auto read = [&](int64_t page, int64_t count, void* dst) {
    const auto it = slab.find(page);
    if (it == slab.end() || count != 1) return false;
    std::memset(dst, 0, 4096);
    std::memcpy(dst, it->second.data(), it->second.size() * sizeof(int64_t));
    return true;
  };
  std::mt19937_64 rng(20261027);
  int adopted_total = 0, refused = 0;
  for (int iter = 0; iter < 3000; ++iter) {
    std::vector<uint8_t> bytes = good;
    const int edits = 1 + static_cast<int>(rng() % 6);
    for (int e = 0; e < edits && !bytes.empty(); ++e) {
      const size_t at = rng() % bytes.size();
      switch (rng() % 5) {
        case 0: bytes[at] ^= static_cast<uint8_t>(1u << (rng() % 8)); break;
        case 1: bytes[at] = static_cast<uint8_t>(rng()); break;
        case 2: bytes.resize(at); break;
        case 3: bytes.insert(bytes.begin() + static_cast<std::ptrdiff_t>(at), static_cast<uint8_t>(rng())); break;
        default: {
          const size_t len = std::min<size_t>(rng() % 16, bytes.size() - at);
          const std::vector<uint8_t> chunk(bytes.begin() + static_cast<std::ptrdiff_t>(at),
                                           bytes.begin() + static_cast<std::ptrdiff_t>(at + len));
          bytes.insert(bytes.end(), chunk.begin(), chunk.end());  // a copy: never a self-range insert
        }
      }
    }
    DiskCache fresh(config(/*pages=*/64));
    DiskCache::Adopted got;
    try {
      got = fresh.restore(bytes, read);
    } catch (const std::exception& e) {
      throw std::runtime_error("iteration " + std::to_string(iter) + ": restore threw: " + e.what());
    }
    std::string why;
    if (!fresh.consistent(&why)) throw std::runtime_error("iteration " + std::to_string(iter) + ": " + why);
    require(fresh.pages_used() <= fresh.pages_total(), "within capacity");
    require(fresh.live_entries() == got.entries, "the count reported is the count adopted");
    adopted_total += got.entries;
    refused += got.error.empty() ? 0 : 1;
  }
  require(adopted_total > 0 && refused > 0, "the mutations both passed and failed: " + std::to_string(adopted_total) +
                                                 " adopted, " + std::to_string(refused) + " refused");
}

DGPP_TEST(disk_cache_retention_keeps_pinned_entries_past_unpinned_ones) {
  // Two entries fit; the older one carries a prompt_cache_key. The third
  // evicts the younger unpinned entry, not the pinned one; with only pinned
  // entries left, the least recently used pinned one goes.
  DiskCache d(config(/*pages=*/10));
  const auto a = counted(12, 1000), b = counted(12, 2000), c = counted(12, 3000), e = counted(12, 4000);
  auto plan = d.begin_spill(a.data(), 8, -1, {}, 0, identities(2, 100), 0, 1, 1, 0, "agent-7");
  require(plan.ok, "a fits");
  const int pinned = plan.entry;
  d.commit_spill(pinned);
  plan = d.begin_spill(b.data(), 8, -1, {}, 0, identities(2, 200), 0, 2, 2, 1);
  require(plan.ok, "b fits");
  d.commit_spill(plan.entry);
  require(d.pinned_entries() == 1, "one pinned entry");
  plan = d.begin_spill(c.data(), 8, -1, {}, 0, identities(2, 300), 0, 3, 3, 2, "agent-9");
  require(plan.ok && plan.evicted.size() == 1 && plan.evicted[0].entry != pinned,
          "the unpinned entry made room, the older pinned one stayed");
  d.commit_spill(plan.entry);
  require(d.entry(pinned).live && d.pinned_entries() == 2, "both pinned entries live");
  plan = d.begin_spill(e.data(), 8, -1, {}, 0, identities(2, 400), 0, 4, 4, 3);
  require(plan.ok && plan.evicted.size() == 1 && plan.evicted[0].entry == pinned,
          "with every entry pinned, the least recently used pinned one goes");
  d.commit_spill(plan.entry);
}

DGPP_TEST(disk_cache_retention_evicts_lru_within_the_capacity) {
  // 10 pages, 2 per blob and 1 for the ids: an entry of 2 blocks takes 5
  // pages, so two fit and the third evicts the least recently used.
  DiskCache d(config(/*pages=*/10));
  std::vector<int> entries;
  for (int i = 0; i < 2; ++i) {
    const auto prompt = counted(12, 1000 * (i + 1));
    const auto plan = d.begin_spill(prompt.data(), 8, -1, {}, 0, identities(2, 100 * (i + 1)), 0, i + 1, i + 1, i);
    require(plan.ok, "entry " + std::to_string(i) + " fits");
    d.commit_spill(plan.entry);
    entries.push_back(plan.entry);
  }
  require(d.pages_used() == 10 && d.live_entries() == 2, "two entries, ten pages");
  d.touch(entries[0], 10);  // the first is the most recently used now
  const auto prompt = counted(12, 3000);
  const auto plan = d.begin_spill(prompt.data(), 8, -1, {}, 0, identities(2, 300), 0, 3, 11, 2);
  require(plan.ok && plan.evicted.size() == 1 && plan.evicted[0].entry == entries[1] &&
              plan.evicted[0].position == 8 && plan.evicted[0].memory == 1,
          "the least recently used entry made room and named its memory link");
  require(d.pages_used() == 10 && d.live_entries() == 2 && d.stats().evictions == 1, "capacity held");
  d.commit_spill(plan.entry);
  // An entry the slab can never hold is refused without evicting anything.
  const auto huge_prompt = counted(40, 3000);
  const auto huge = d.begin_spill(huge_prompt.data(), 40, -1, {}, 0, identities(10, 900), 0, 4, 12, 3);
  require(!huge.ok && huge.evicted.empty() && d.live_entries() == 2, "an entry past the capacity is skipped");
  require(d.stats().spill_skipped == 1, "the skip counted");
  // Pages never exceed the capacity across the run.
  require(d.pages_used() <= d.pages_total(), "within capacity");
}

DGPP_TEST(disk_cache_index_codec_round_trips_prunes_and_carries_content) {
  // serialize() is write(parse()) byte for byte, a record's pages are its
  // entry's pages in digest order, a truncated index does not parse, and
  // adoption takes each entry's content digest in the serialized order —
  // from a pruned candidate too (the tier drops entries whose pages no
  // longer carry the content their digest recorded).
  DiskCache d(config(/*pages=*/20));
  std::vector<int> entries;
  std::map<int64_t, std::vector<int64_t>> runs;  // the ids runs by page, as the slab would hold them
  for (int i = 0; i < 3; ++i) {
    const auto prompt = counted(12, 1000 * (i + 1));
    const auto plan = d.begin_spill(prompt.data(), 8, -1, {}, 0, identities(2, 100 * (i + 1)), 0, i + 1, i + 1, i,
                                    i == 1 ? "agent-7" : "");
    require(plan.ok && plan.evicted.empty(), "entry " + std::to_string(i) + " fits");
    d.commit_spill(plan.entry);
    d.set_content(plan.entry, 0x1000 + static_cast<uint64_t>(i));
    runs[plan.ids_page].assign(prompt.begin(), prompt.begin() + 8);
    entries.push_back(plan.entry);
  }
  const std::vector<uint8_t> bytes = d.serialize();
  DiskCache::Header h;
  std::vector<DiskCache::Record> records;
  require(DiskCache::parse(bytes, &h, &records) && records.size() == 3 && h.pages == 20 && h.page_bytes == 4096 &&
              h.blob_pages == 2 && h.block_tokens == 4,
          "the index parses into its records");
  require(DiskCache::write(h, records) == bytes, "write(parse()) is the identity");
  require(records[1].key == "agent-7" && records[0].key.empty(), "keys ride the records");
  const std::vector<uint64_t> digests = d.content_digests();
  require(digests == std::vector<uint64_t>{0x1000, 0x1001, 0x1002}, "the digests follow the serialized order");
  for (size_t n = 0; n < records.size(); ++n)
    require(DiskCache::record_pages(records[n], 2) == d.entry_pages(entries[static_cast<size_t>(n)]),
            "a record's pages are its entry's pages, in digest order");
  require(d.entry_pages(entries[0]).size() == 5, "two blob pages, the ids page, two block pages");
  std::vector<uint8_t> cut(bytes.begin(), bytes.end() - 5);
  require(!DiskCache::parse(cut, &h, &records), "a truncated index does not parse");
  const auto read = [&](int64_t page, int64_t count, void* dst) {
    const auto it = runs.find(page);
    if (it == runs.end()) return false;
    std::memset(dst, 0, static_cast<size_t>(count) * 4096);
    std::memcpy(dst, it->second.data(), it->second.size() * sizeof(int64_t));
    return true;
  };
  DiskCache fresh(config(20));
  const auto a = fresh.restore(bytes, read, &digests);
  require(a.entries == 3 && a.dropped == 0 && a.error.empty(), "every entry adopted");
  for (int i = 0; i < 3; ++i)
    require(fresh.entry(i).content == 0x1000 + static_cast<uint64_t>(i), "the content digest rides adoption");
  require(fresh.consistent(), "the adopted index is consistent");
  // The pruned candidate: the middle entry gone, the digests pruned alike.
  require(DiskCache::parse(bytes, &h, &records), "parses again");
  const std::vector<DiskCache::Record> kept{records[0], records[2]};
  const std::vector<uint64_t> kept_digests{digests[0], digests[2]};
  DiskCache pruned(config(20));
  const auto b = pruned.restore(DiskCache::write(h, kept), read, &kept_digests);
  require(b.entries == 2 && b.dropped == 0 && pruned.live_entries() == 2, "the pruned candidate adopts the kept entries");
  require(pruned.entry(0).content == 0x1000 && pruned.entry(1).content == 0x1002, "with their own digests");
  require(pruned.consistent() && pruned.pages_used() == 10, "consistent, the dropped entry's pages free");
  // Without digests every adopted entry's content is 0 (no tier to bind it).
  DiskCache plain(config(20));
  require(plain.restore(bytes, read).entries == 3 && plain.entry(2).content == 0, "no digests: content 0");
}

DGPP_TEST(disk_cache_pin_cap_stops_protecting_past_the_share) {
  // 20 pages, 5 per entry. Three pinned entries hold 75 % of the slab —
  // past a 50 % cap a pin protects nothing and plain LRU takes the oldest
  // entry, pinned or not; at 100 the unpinned entry goes first.
  for (const int cap : {50, 100}) {
    DiskCache::Config c = config(/*pages=*/20);
    c.pin_max_pct = cap;
    DiskCache d(c);
    std::vector<int> entries;
    for (int i = 0; i < 4; ++i) {
      const auto prompt = counted(12, 1000 * (i + 1));
      const auto plan = d.begin_spill(prompt.data(), 8, -1, {}, 0, identities(2, 100 * (i + 1)), 0, i + 1, i + 1, i,
                                      i < 3 ? "pin" : "");
      require(plan.ok && plan.evicted.empty(), "entry " + std::to_string(i) + " fits");
      d.commit_spill(plan.entry);
      entries.push_back(plan.entry);
    }
    require(d.pinned_entries() == 3 && d.pages_used() == 20, "three pinned entries, the slab full");
    const auto prompt = counted(12, 9000);
    const auto plan = d.begin_spill(prompt.data(), 8, -1, {}, 0, identities(2, 900), 0, 9, 9, 4);
    require(plan.ok && plan.evicted.size() == 1, "one eviction made room");
    if (cap == 50)
      require(plan.evicted[0].entry == entries[0], "past the cap the oldest entry goes, pinned or not");
    else
      require(plan.evicted[0].entry == entries[3], "at 100 the unpinned entry goes first");
    d.commit_spill(plan.entry);
    require(d.consistent(), "consistent");
  }
}
