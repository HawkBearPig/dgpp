#pragma once
// Host index and slot ownership for prefix-cache snapshots (DESIGN §8).
// An entry associates the first P token IDs with a model snapshot at a
// valid prefill cut. A request may attach only when its token prefix and
// cold-prefill cut match, so suffix prefill preserves the cold run's
// chunk sequence and arithmetic.
//
// The engine's PrefixArena owns the state. This index tracks references,
// LRU eviction and a decision digest. Entries attached to live requests
// are protected from eviction. Each rank derives the same decisions from
// journaled inputs and compares the digest on the next tick.
//
// Prefill and request retirement can create reusable entries; rolling
// snapshots preserve aligned positions reached during decode, including
// positions crossed by an MTP step. Entries are local to the process.
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/image_input.hpp"

namespace dgpp::sched {

class PrefixCache {
 public:
  struct ImageKey {
    std::shared_ptr<const ImageInput> input;
    uint64_t hash = 0;
  };
  using Images = std::vector<ImageKey>;
  struct Config {
    int slots = 0;              // arena slots the engine holds (0: off)
    int64_t align = 1;          // kpool: every entry position is a multiple
    int64_t chunk_tokens = 2048;  // the cold prefill's chunk length
    size_t image_bytes = kMaxRequestImageBytes;  // shared immutable pixel identities
  };
  struct Entry {
    std::vector<int64_t> ids;   // the first `position` ids of the sequence
    Images images;             // immutable pixels shared across matching entries
    int64_t position = 0;
    int64_t next_token = -1;    // MTP prefill consumed ids[position]; -1: no lookahead constraint
    int slot = -1;              // the arena slot; -1 once evicted
    uint64_t hash = 0;
    uint64_t last_use = 0;      // the tick of the last attach / insert
    int attached = 0;           // live requests attached (never evicted)
    bool live = false;
    // The NVMe cold tier (issue #26): the disk entry holding this
    // entry's state (-1: none), and whether a spill is reading it — a
    // busy entry is never evicted (its slot and blocks are being copied).
    int disk = -1;
    bool busy = false;
  };
  struct Stats {
    int64_t hits = 0;
    int64_t misses = 0;
    int64_t tokens_saved = 0;   // sum of attach positions
    int64_t snapshots = 0;      // entries taken at a prefill cut
    int64_t close_entries = 0;  // entries taken from a rolling snapshot
    int64_t rolling = 0;        // rolling snapshots taken (the hops included)
    int64_t hops = 0;           // of them, taken from a two-row step's first row
    int64_t evictions = 0;
    int64_t duplicates = 0;     // an entry already existed at the position
    int64_t skipped_no_slot = 0;  // a snapshot wanted, no slot free or evictable
    int64_t skipped_no_block = 0;  // a snapshot wanted, no pool block for its partial copy
    int64_t skipped_image_bytes = 0;
  };

  PrefixCache() = default;
  explicit PrefixCache(const Config& cfg);

  bool enabled() const { return cfg_.slots > 0; }
  const Config& config() const { return cfg_; }
  int slots() const { return cfg_.slots; }
  int free_slots() const { return static_cast<int>(free_.size()); }
  int live_entries() const;
  size_t image_bytes() const;
  const Stats& stats() const { return stats_; }
  Stats& stats() { return stats_; }

  // The cold prefill's cut positions in (0, n), ascending and unique: every
  // multiple of chunk_tokens and the aligned image floor(b / align) * align
  // of every boundary b (structural positions in the prompt: role-marker
  // tokens, or whatever the caller derives).
  std::vector<int64_t> cuts(int64_t n,
                            const std::vector<int64_t>& boundaries) const;

  // The prefix hash of ids[0..n): FNV-1a over the id bytes, extendable
  // (hash(ids, n + 1) continues hash(ids, n) with one more id).
  static uint64_t hash_prefix(const int64_t* ids, int64_t n,
                              uint64_t seed = kSeed);
  static uint64_t extend_hash(uint64_t h, int64_t id);
  static constexpr uint64_t kSeed = 0xcbf29ce484222325ull;

  // Hashes narrow the search; hits also compare all pixels and geometry.
  // Include images starting AT the cut: an MTP snapshot can already have
  // consumed the next token's image embedding in its shifted input.
  Images image_keys(const std::vector<ImageInput>& images) const;
  static uint64_t with_images(uint64_t token_hash, int64_t position, const Images& images);

  // The deepest live entry whose position is one of `cuts` (all < the
  // prompt's length) and whose ids equal the prompt's first `position`
  // ids, with a matching lookahead token when the entry has one.
  // `cut_hashes[i]` is hash_prefix(prompt, cuts[i]). -1: none.
  int lookup(const std::vector<int64_t>& prompt,
             const std::vector<int64_t>& cuts,
             const std::vector<uint64_t>& cut_hashes, const Images& images = {}) const;
  // Whether compatible state exists for these ids (the dedupe check).
  // next_token == -1 asks for an unconstrained entry; constrained prefill
  // state must not replace a rolling snapshot that can resume with any token.
  int find_exact(const int64_t* ids, int64_t n, uint64_t hash, const Images& images = {},
                 int64_t next_token = -1) const;
  // The miss diagnostic: the live entry sharing the longest
  // prefix with the prompt, and that length — where a prompt that should
  // have attached first differs from what the cache holds (an agent
  // client's edited system prompt, a compacted history). A linear pass
  // over the entries, taken on a miss only; never a decision.
  struct Nearest {
    int entry = -1;
    int64_t common = 0;  // ids shared with the entry, from the start
  };
  Nearest nearest(const std::vector<int64_t>& prompt, const Images& images = {}) const;
  // The ghosts: the last kGhosts evicted entries by (hash,
  // position, last use, eviction ordinal), so a miss can say "an entry at
  // this prompt's cut was evicted" — the case the nearest entry cannot
  // tell from a changed prompt once the conversation's own entries are
  // gone (the 7-slot pressure run: three streams' entries pushed out by a
  // round of side requests). Never a decision, not in the digest.
  struct Ghost {
    uint64_t hash = 0;
    int64_t position = 0;
    uint64_t last_use = 0;
    int64_t eviction = 0;  // its ordinal among the cache's evictions, from 1
  };
  static constexpr size_t kGhosts = 256;
  // The deepest cut of the prompt an evicted entry sat at, or position 0.
  Ghost ghost_at(const std::vector<int64_t>& cuts,
                 const std::vector<uint64_t>& cut_hashes) const;

  // ---- the slot ledger ----------------------------------------------------
  // The smallest free slot, or -1.
  int take_free_slot();
  void give_back_slot(int slot);
  // The least recently used live entry with no attached request and no
  // spill in flight: its slot is freed (the caller releases it on the
  // engine) and the entry dies. Returns the slot, or -1 when nothing is
  // evictable; `disk` (optional) receives the victim's disk entry (-1:
  // none) so the caller can tell the cold tier its memory copy went.
  int evict_lru(int* disk = nullptr);
  // A free slot, evicting once when none is free. -1 when neither works.
  int acquire_slot();

  // ---- entries -----------------------------------------------------------
  // Inserts an entry over ids[0..position) in `slot`. Returns its index,
  // or -1 when an identical entry is live or pixel storage is full (the caller keeps the slot out
  // of the entry and gives it back).
  int insert(const int64_t* ids, int64_t position, int slot, uint64_t now,
             const Images& images = {}, int64_t next_token = -1);
  const Entry& entry(int index) const { return entries_.at(static_cast<size_t>(index)); }
  int size() const { return static_cast<int>(entries_.size()); }  // records, live or dead
  void attach(int index, uint64_t now);
  void detach(int index);
  // Refreshes an entry's LRU stamp without a decision (no stats, no digest).
  void touch(int index, uint64_t now);
  // The cold tier's links (issue #26): the disk entry an entry is stored
  // in, and the busy flag while a spill copies it.
  void set_disk(int index, int disk);
  void set_busy(int index, bool busy);
  int64_t blocks_pinned(int64_t block_tokens) const;  // every live entry's

  // ---- the decision digest -----------------------------------------------
  // Folds a decision (an attach, a snapshot, an eviction ...) into the
  // running digest; the journal carries it so every rank can compare.
  void note(uint64_t a, uint64_t b, uint64_t c);
  uint64_t digest() const { return digest_; }

 private:
  Config cfg_;
  std::vector<Entry> entries_;
  std::vector<Ghost> ghosts_;  // a ring of kGhosts
  size_t ghost_next_ = 0;
  std::unordered_multimap<uint64_t, int> by_hash_;  // hash -> entry index
  std::vector<int> free_;  // sorted ascending
  Stats stats_;
  uint64_t digest_ = kSeed;
};

}  // namespace dgpp::sched
