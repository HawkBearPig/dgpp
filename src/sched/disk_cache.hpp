#pragma once
// The prefix cache's cold tier index (issue #26): which entries and which
// cache blocks sit on this rank's NVMe slab, at which pages, under LRU
// retention within a fixed capacity. Host state only — every mutation is
// a scheduler decision derived from journaled inputs (a spill or restore
// started by the policy, a commit rank 0 journaled once every rank
// reported), so the index is identical on every rank while the slab's
// BYTES are each rank's own (its shard of the caches).
//
// The slab is `pages` records of `page_bytes` (one cache block's planes,
// rounded to 4 KiB). A block record takes one page from the low end; a
// snapshot blob takes `blob_pages` contiguous pages from the high end.
// Blocks are shared between entries by physical identity (the pool's
// block id and the generation of its contents): two entries that hold the
// same physical block in memory — a changed question over a document
// attached to it — reference one record, freed at refcount zero. An entry
// on disk keeps the token ids (the exact-match compare), the cut, the MTP
// lookahead token and draft position, and its records; restoring it
// reads the records into fresh pool blocks and an arena slot.
//
// The slab outlives the process (nvme_cache.persist): the token ids are
// written to the slab too, in a page run right after the blob's, so the
// index the tier keeps beside the slab is small — positions, pages, keys
// — and serialize() / restore() carry the committed index across a
// restart. An adopted entry's block records get synthetic identities (no
// live block has them), so sharing resumes only once a restore commits.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "sched/prefix_cache.hpp"

namespace dgpp::sched {

class DiskCache {
 public:
  struct Config {
    int64_t pages = 0;        // slab records (0: off)
    size_t page_bytes = 0;    // one block record
    int64_t blob_pages = 1;   // pages one snapshot blob spans
    int64_t block_tokens = 0; // the pool's block (0: no pool, blobs alone)
    int64_t min_tokens = 0;   // entries below this position are not spilled
    // prompt_cache_key pins protect their entries from eviction while the
    // pinned entries' pages stay within this share of the slab; past it a
    // pinned entry is as evictable as any other (one tenant cannot pin
    // the whole tier against everyone else). 100: pins always protect.
    int pin_max_pct = 50;
  };
  enum class State : int { kSpilling = 0, kResident = 1, kRestoring = 2 };
  struct Entry {
    std::vector<int64_t> ids;
    PrefixCache::Images images;
    int64_t position = 0;
    int64_t next_token = -1;   // the MTP lookahead constraint (-1: none)
    int64_t mtp_position = 0;  // the draft block's row counter at the cut
    uint64_t hash = 0;
    int64_t blob_page = -1;    // the blob's first page; the ids run follows it
    int ids_pages = 0;         // pages the token ids take, right after the blob
    std::vector<int> blocks;   // block records, one per full block
    int partial = -1;          // the partial block's record, or -1
    State state = State::kResident;
    uint64_t op = 0;           // the pending spill/restore op
    int memory = -1;           // the PrefixCache entry holding it, or -1
    uint64_t last_use = 0;
    bool live = false;
    std::string key;           // prompt_cache_key: evicted only after every unpinned entry
    // This rank's digest of the pages' checksums when the spill committed
    // (never serialized: the index bytes are the world's, the slab's bytes
    // each rank's own). Adoption keeps an entry only while its pages still
    // carry the content the digest recorded.
    uint64_t content = 0;
  };
  struct Block {
    int64_t page = -1;
    int refs = 0;
    bool ready = false;        // its record is on disk (the spill committed)
    std::vector<uint64_t> identities;
    bool live = false;
  };
  struct Stats {
    int64_t spills = 0;         // spills begun
    int64_t spilled = 0;        // committed
    int64_t spill_failed = 0;
    int64_t spill_skipped = 0;  // no room after evicting everything evictable
    int64_t restores = 0;       // restores begun
    int64_t restored = 0;
    int64_t restore_failed = 0;
    int64_t evictions = 0;      // disk entries evicted for room
    int64_t pages_written = 0;  // block and blob pages spills wrote
    int64_t blocks_shared = 0;  // blocks a spill found already on disk
    int64_t pages_read = 0;     // pages restores read
    int64_t tokens_restored = 0;
    int64_t adopted = 0;        // entries taken over from a previous run's index
  };
  // A spill's plan: per full block, the page to write (-1: shared with a
  // record already on disk), the blob's run and the partial block's page.
  struct Evicted {
    int entry = -1;   // the disk entry's index (dead now)
    int64_t position = 0;
    int memory = -1;  // the memory entry that held it too, or -1
  };
  struct SpillPlan {
    bool ok = false;
    bool duplicate = false;  // an identical entry is already on disk (nothing to do)
    int entry = -1;
    int64_t blob_page = -1;
    int64_t ids_page = -1;   // the ids run: blob_page + blob_pages, ids_pages long
    int ids_pages = 0;
    std::vector<int64_t> pages;
    int64_t partial_page = -1;
    std::vector<Evicted> evicted;  // the entries evicted for room
  };
  struct RestorePlan {
    int64_t blob_page = -1;
    std::vector<int64_t> pages;   // per full block
    int64_t partial_page = -1;
  };

  DiskCache() = default;
  explicit DiskCache(const Config& cfg);

  bool enabled() const { return cfg_.pages > 0; }
  const Config& config() const { return cfg_; }
  const Stats& stats() const { return stats_; }
  Stats& stats() { return stats_; }
  int live_entries() const;
  int pinned_entries() const;  // live entries with a key
  int64_t pages_used() const { return used_pages_; }
  int64_t pages_total() const { return cfg_.pages; }
  int64_t live_blocks() const;
  int64_t bytes_used() const { return used_pages_ * static_cast<int64_t>(cfg_.page_bytes); }
  const Entry& entry(int index) const { return entries_.at(static_cast<size_t>(index)); }
  // Every live entry's index, ascending (the metrics, the tests).
  std::vector<int> live() const;
  // The index's invariants, recomputed from scratch (the tests): every
  // live entry's runs and records cover exactly the pages marked used,
  // no page twice, every record's refcount is the live entries naming it,
  // every live entry is in the hash index. `why` names the first breach.
  bool consistent(std::string* why = nullptr) const;
  // Pages an entry's token ids take at a position (8 bytes a token).
  int ids_pages_for(int64_t position) const;

  // ---- across restarts ----------------------------------------------------
  // The committed index as bytes: every resident entry (a spill still in
  // flight is left out, as is one with image inputs) by position, pages,
  // key and lookahead, in least-recently-used order — the token ids stay
  // in the slab. The tier keeps these beside the slab.
  std::vector<uint8_t> serialize() const;
  // The entries serialize() carries, in its order, by their content digest
  // (rank-local; 0 for an entry committed without one).
  std::vector<uint64_t> content_digests() const;
  // An entry's pages in digest order: the blob's run (the blob, then the
  // ids behind it), the full blocks' records, the partial block's record.
  std::vector<int64_t> entry_pages(int entry) const;
  void set_content(int entry, uint64_t digest);
  // The serialized index taken apart and put back together: one record per
  // committed entry in the serialized order. The tier prunes the candidate
  // it offers (entries whose pages no longer hold the content their digest
  // recorded, or whose ids are unreadable) without knowing the format, and
  // record_pages() lists a record's pages in entry_pages() order.
  struct Header {
    uint64_t pages = 0, page_bytes = 0, blob_pages = 0, block_tokens = 0;
  };
  struct Record {
    int64_t position = 0, next_token = -1, mtp_position = 0, blob_page = -1;
    uint32_t ids_pages = 0;
    std::string key;
    std::vector<int64_t> block_pages;
    int64_t partial_page = -1;
  };
  static bool parse(const std::vector<uint8_t>& bytes, Header* header, std::vector<Record>* records);
  static std::vector<uint8_t> write(const Header& header, const std::vector<Record>& records);
  static std::vector<int64_t> record_pages(const Record& record, int64_t blob_pages);
  // Adopts such bytes into an empty index whose geometry matches theirs:
  // each entry's ids are read from its run through `read_pages` (false:
  // that entry is dropped), its block records are rebuilt by page with
  // synthetic identities, and every adopted entry is older than anything
  // new. `digests` (optional) carries the entries' content digests in the
  // serialized order. `error` names a header that does not fit this slab
  // (nothing adopted then).
  struct Adopted {
    int entries = 0;
    int dropped = 0;
    std::string error;
  };
  Adopted restore(const std::vector<uint8_t>& bytes,
                  const std::function<bool(int64_t page, int64_t count, void* dst)>& read_pages,
                  const std::vector<uint64_t>* digests = nullptr);

  // The deepest entry that is on disk (resident, or restoring — a second
  // request then waits on the same op) at one of the prompt's cuts with
  // the prompt's first `position` ids and a matching lookahead token; -1
  // when none. Entries still spilling are the memory index's to find.
  int lookup(const std::vector<int64_t>& prompt, const std::vector<int64_t>& cuts,
             const std::vector<uint64_t>& cut_hashes, const PrefixCache::Images& images = {}) const;
  int find_exact(const int64_t* ids, int64_t n, uint64_t hash, const PrefixCache::Images& images,
                 int64_t next_token) const;

  // ---- spills ----------------------------------------------------------
  // Plans an entry over ids[0..position) whose full blocks have the given
  // identities (`partial_identity` 0 when the position is block-aligned):
  // shares every block a resident record already holds, takes pages for
  // the rest and a run for the blob, evicting LRU resident entries for
  // room. The entry starts kSpilling; commit or abort it by the op's
  // outcome. Not ok (and nothing changed) when the slab cannot hold it
  // even empty, or when an identical entry is already on disk.
  SpillPlan begin_spill(const int64_t* ids, int64_t position, int64_t next_token,
                        const PrefixCache::Images& images, int64_t mtp_position,
                        const std::vector<uint64_t>& identities, uint64_t partial_identity,
                        uint64_t op, uint64_t now, int memory_entry, const std::string& key = "");
  void commit_spill(int entry);
  void abort_spill(int entry);

  // ---- restores --------------------------------------------------------
  // The entry's records, and the entry held kRestoring (never evicted)
  // until the op commits or aborts.
  RestorePlan begin_restore(int entry, uint64_t op);
  // The restore landed in memory entry `memory_entry` (-1: the memory
  // index refused a duplicate and the blocks were released) whose fresh
  // blocks carry `identities`: the records take those identities too, so a
  // later entry sharing them spills nothing twice.
  void commit_restore(int entry, int memory_entry, const std::vector<uint64_t>& identities,
                      uint64_t partial_identity, uint64_t now);
  // The record read back wrong (a checksum, a short read, a device error):
  // the entry is dropped — its blocks' records with it at refcount zero.
  void abort_restore(int entry);
  // The restore never started on the engine (it refused the slot or the
  // blocks): the entry stands resident as it was, nothing counted.
  void cancel_restore(int entry);

  // The memory copy of a dual entry went (the memory index evicted it).
  void on_memory_evicted(int entry);
  void touch(int entry, uint64_t now);
  // Frees the least recently used resident entry; its index, or -1.
  // `memory` (optional) receives the memory entry it was linked to (-1:
  // none), so the caller can drop that entry's disk link.
  int evict_lru(int* memory = nullptr);

 private:
  int64_t take_page();
  void give_page(int64_t page);
  int64_t take_run(int64_t n);
  void give_run(int64_t start, int64_t n);
  int find_block(uint64_t identity) const;
  int lru_victim() const;
  std::vector<int> serialized_order() const;  // resident entries without images, LRU first
  bool pins_protect() const;                  // the pinned pages are within the cap
  int new_block(int64_t page, uint64_t identity);
  void release_block(int block);
  void kill_entry(int entry);
  int64_t free_pages() const { return cfg_.pages - used_pages_; }

  Config cfg_;
  std::vector<Entry> entries_;
  std::vector<Block> blocks_;
  std::unordered_multimap<uint64_t, int> by_hash_;
  std::unordered_map<uint64_t, int> by_identity_;
  std::vector<uint8_t> used_;  // per page
  int64_t used_pages_ = 0;
  int64_t low_cursor_ = 0;      // the single-page search's hint
  Stats stats_;
};

}  // namespace dgpp::sched
