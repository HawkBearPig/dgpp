// The NVMe cold tier's host index (sched/disk_cache.hpp, issue #26): page
// allocation from both ends, block sharing by identity, LRU retention
// within the capacity, the spill/restore state machine and its aborts.
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
  require(plan.blob_page == 14, "the blob run sits at the top: pages 14-15");
  require(plan.pages[0] == 0 && plan.pages[1] == 1 && plan.pages[2] == 2, "block pages from the bottom");
  require(d.pages_used() == 5, "5 pages in use");
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
  require(rp.blob_page == 14 && rp.pages.size() == 3 && rp.pages[2] == 2, "the restore reads the same pages");
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
  require(d.live_blocks() == 4 && d.pages_used() == 4 + 2, "shared records survive, the blob run is freed");
  cuts_for(prompt, {4, 8, 12, 16}, &cuts, &hashes);
  require(d.lookup(prompt, cuts, hashes) == plan2.entry, "the deeper entry is offered now");
}

DGPP_TEST(disk_cache_aborts_release_records_and_dedupes_identical_entries) {
  DiskCache d(config(/*pages=*/12));
  const auto prompt = counted(20);
  // A partial block: 10 tokens = two full blocks and one partial.
  const auto plan = d.begin_spill(prompt.data(), 10, 7, {}, 0, identities(2, 10), /*partial=*/77, 1, 1, 0);
  require(plan.ok && plan.partial_page >= 0 && plan.pages.size() == 2, "partial block gets a page");
  require(d.pages_used() == 5, "2 + 1 + 2 blob pages");
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

DGPP_TEST(disk_cache_retention_evicts_lru_within_the_capacity) {
  // 10 pages, 2 per blob: an entry of 2 blocks takes 4 pages, so two fit
  // and the third evicts the least recently used.
  DiskCache d(config(/*pages=*/10));
  std::vector<int> entries;
  for (int i = 0; i < 2; ++i) {
    const auto prompt = counted(12, 1000 * (i + 1));
    const auto plan = d.begin_spill(prompt.data(), 8, -1, {}, 0, identities(2, 100 * (i + 1)), 0, i + 1, i + 1, i);
    require(plan.ok, "entry " + std::to_string(i) + " fits");
    d.commit_spill(plan.entry);
    entries.push_back(plan.entry);
  }
  require(d.pages_used() == 8 && d.live_entries() == 2, "two entries, eight pages");
  d.touch(entries[0], 10);  // the first is the most recently used now
  const auto prompt = counted(12, 3000);
  const auto plan = d.begin_spill(prompt.data(), 8, -1, {}, 0, identities(2, 300), 0, 3, 11, 2);
  require(plan.ok && plan.evicted.size() == 1 && plan.evicted[0].entry == entries[1] &&
              plan.evicted[0].position == 8 && plan.evicted[0].memory == 1,
          "the least recently used entry made room and named its memory link");
  require(d.pages_used() == 8 && d.live_entries() == 2 && d.stats().evictions == 1, "capacity held");
  d.commit_spill(plan.entry);
  // An entry the slab can never hold is refused without evicting anything.
  const auto huge = d.begin_spill(prompt.data(), 40, -1, {}, 0, identities(10, 900), 0, 4, 12, 3);
  require(!huge.ok && huge.evicted.empty() && d.live_entries() == 2, "an entry past the capacity is skipped");
  require(d.stats().spill_skipped == 1, "the skip counted");
  // Pages never exceed the capacity across the run.
  require(d.pages_used() <= d.pages_total(), "within capacity");
}
