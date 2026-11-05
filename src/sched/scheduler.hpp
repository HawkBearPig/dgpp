#pragma once
// Deterministic admission, decoding and cancellation over a session engine.
// Every rank applies the same policy to journaled requests and cancellations.
// Clock feedback is converted into a token limit by the leader and journaled.
// Peers never make local timing decisions: engine operations include
// collectives whose order must match across ranks.
//
// Each tick admits fitting queued requests against the tick's prefill
// budget and performs prefill work, then runs one decode step. Admission chooses the oldest request that fits,
// so a smaller request can pass a larger one; sustained small requests can
// starve a large request. With a zero budget prefill blocks decode for the
// full admission; supported engines yield between chunks with a positive budget.
// Images on engines without image chunking retain one monolithic admission
// as the tick's only prefill work when the prompt exceeds the budget.
//
// Full admission reserves prompt plus maximum completion. Grow admission
// reserves a window and extends it before decode, ending the youngest
// request if capacity runs out. Decode visits active requests in canonical
// order, using the number of slots the engine can advance in one pass.
// Cancellation and retirement release request resources between operations.
//
// Request slots isolate model state. Tests compare concurrent requests with
// independent runs, including cancellation, batching and slot reuse.
// See DESIGN §11 and docs/operations.md for the journal and resource policy.
#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "sched/disk_cache.hpp"
#include "sched/prefix_cache.hpp"
#include "sched/prefill_latency.hpp"
#include "common/prefill_progress.hpp"
#include "common/image_input.hpp"
#include "sample/sampler.hpp"
#include "text/tool_grammar.hpp"

namespace dgpp::sched {

// One logit_bias entry (OpenAI's logit_bias, 2026-09-06): the token's id
// and the bias added to its logit before the pick, on every rank
// identically (it rides the journal with the request).
struct LogitBias {
  int32_t token = 0;
  float bias = 0.0f;
};

// The engine interface the scheduler drives. The real binding (glm_gen_check)
// closes over GlmDiagnosticModel + the pick; the host gate binds a
// recording fake. The pick rides inside each op (at TP>1 it is a
// distributed collective), which is what keeps the scheduler pure host
// code. A step returns every token newly decided by that engine pass: one
// for ordinary decode, or 1..T for a speculative step.
class SchedulerEngine {
 public:
  virtual ~SchedulerEngine() = default;

  const std::shared_ptr<PrefillMonitor>& prefill_monitor() const { return prefill_monitor_; }

 private:
  std::shared_ptr<PrefillMonitor> prefill_monitor_ = std::make_shared<PrefillMonitor>();

 public:
  // Engine session slots (== the decode-row bound at real dims: 8).
  virtual int max_concurrent_requests() const = 0;
  // DSA pool meters (a no-DSA model reports an unbounded pool).
  virtual int64_t pool_blocks_total() const = 0;
  virtual int64_t pool_blocks_in_use() const = 0;
  // Raw block geometry, without engine-specific reservation headroom.
  virtual int64_t blocks_for_tokens(int64_t tokens) const = 0;
  // Physical block footprint of reserve(req, tokens) and begin_prefill's
  // reservation, before prefix sharing and snapshot copies. Engines that
  // reserve extra rows (e.g. speculative verify/draft tails) must include
  // those rows and their context clamp here. Pure, rank-deterministic, and
  // valid before opening a slot; ordinary engines retain the raw quote.
  virtual int64_t reservation_blocks(int64_t tokens) const {
    return blocks_for_tokens(tokens);
  }

  // Opens slot `req` (fresh state), prefills `prompt`, picks the first
  // generated token. Returns a token id in [0, vocab).
  virtual int32_t prefill(int req, const std::vector<int64_t>& prompt) = 0;
  virtual bool supports_images() const { return false; }
  virtual bool supports_video() const { return false; }
  // The checkpoint's image delimiters, for the frontend that renders them.
  virtual ImageTokens image_token_ids() const { return {}; }
  virtual bool supports_image_prefix_cache() const { return false; }
  virtual int32_t prefill_images(int, const std::vector<int64_t>&, const std::vector<ImageInput>&) {
    throw std::invalid_argument("this engine does not support image inputs");
  }

  // Opt-in continuation support. One advance executes at most the aligned
  // token budget passed to begin; it returns a first token only on completion.
  virtual int64_t prefill_chunk_alignment() const { return 0; }
  virtual int64_t prefill_chunk_limit() const { return 0; }
  // The prefill attention grid's wave in query rows (2026-10-15): the rows
  // that fill every SM with one CTA exactly, 0 when the engine states none.
  // The adaptive controller rounds its quantum up to whole waves so a
  // chunk's last wave never leaves a third of the SMs idle (the 27B's tile
  // kernel at 1024 rows ran 5.33 waves on 48 SMs: 47 against 52 TFLOP/s at
  // whole waves).
  virtual int64_t prefill_wave_rows() const { return 0; }
  virtual bool supports_image_chunked_prefill() const { return false; }
  // Several requests' cold prompts in one forward (the scheduler admits a
  // group per tick when the engine allows it): each prompt within
  // prefill_group_span_limit() tokens, the group within
  // prefill_group_total_limit(); no prefix-cache plans. Returns each
  // request's first token, in order. The default runs them one by one.
  virtual int64_t prefill_group_span_limit() const { return 0; }  // 0: no grouping
  virtual int64_t prefill_group_total_limit() const { return 0; }
  virtual std::vector<int32_t> prefill_group(const std::vector<int>& reqs,
                                             const std::vector<const std::vector<int64_t>*>& prompts) {
    std::vector<int32_t> out;
    out.reserve(reqs.size());
    for (size_t i = 0; i < reqs.size(); ++i) out.push_back(prefill(reqs[i], *prompts[i]));
    return out;
  }
  // Pins the slot's lifetime block reservation. The scheduler calls this
  // immediately after prefill and before the first step; `tokens` is
  // prompt.size() + max_steps. Device-driven graphs rely on this because
  // they cannot grow a host-owned block table during replay.
  virtual void reserve(int req, int64_t tokens) = 0;
  // Advances slot `req` and returns the newly decided tokens in transcript
  // order. Must return at least one nonnegative token. The engine owns its
  // pending input token(s), which lets a recorded graph feed itself.
  virtual std::vector<int32_t> step(int req) = 0;
  // Maximum number of independent request slots one physical decode pass
  // can advance. Scalar engines inherit one. A row-batched graph overrides
  // this together with step_batch(); the value must stay fixed for the
  // engine's lifetime and cannot exceed max_concurrent_requests().
  virtual int decode_batch_capacity() const { return 1; }
  // Advances `reqs` in order and returns one token vector per request in the
  // same order. The default deliberately lowers to scalar step() calls, so
  // existing engines keep their exact op stream. Batch-capable engines
  // override this method with one physical pass.
  virtual std::vector<std::vector<int32_t>> step_batch(
      const std::vector<int>& reqs);
  // Retires the slot: blocks return to the pool; the slot may reopen.
  virtual void close(int req) = 0;
  // The most tokens one step() can write into the slot's KV (1 for plain
  // decode; T for a speculative engine whose verify writes T rows). The
  // grow-on-demand policy sizes each reservation's headroom by it.
  virtual int max_tokens_per_step() const { return 1; }
  // Speculative acceptance by draft position: attempts[p] is
  // the steps that verified draft p (0-based; depth of them), accepts[p]
  // the steps that accepted it. Engine-wide since construction, or per
  // slot since it opened (reset at close). A plain engine reports depth 0.
  struct MtpAcceptance {
    int depth = 0;
    uint64_t attempts[8] = {};
    uint64_t accepts[8] = {};
    // A block drafter's first rejected draft, ranked among the candidates
    // its walk scored at that position (2026-10-05, the tree-verify
    // sizing): miss_rank[r] counts corrections that were the drafter's
    // (r+1)-th candidate, miss_rank[kMissRanks] those outside the list.
    static constexpr int kMissRanks = 16;
    uint64_t miss_rank[kMissRanks + 1] = {};
    // Context-lookup drafting (engine.lookup_draft): steps whose drafts a
    // prompt-history match replaced (engine-wide since construction), of
    // which agree_fused took a below-strong match on device-pick agreement.
    uint64_t lookup_fused = 0;
    uint64_t lookup_agree_fused = 0;
    uint64_t lookup_sampled_fused = 0;
  };
  // Last launched decode graph, retained while idle; counters since startup.
  struct DecodeBatchStats {
    static constexpr int kMaxSlots = 16;
    int slots = 0, active = 0, rows_per_request = 0;
    uint64_t replays = 0, rows = 0, padded_rows = 0;
    uint64_t replays_by_slots[kMaxSlots + 1] = {};
  };
  virtual DecodeBatchStats decode_batch_stats() const { return {}; }
  virtual MtpAcceptance mtp_acceptance() const { return {}; }
  virtual MtpAcceptance mtp_acceptance(int req) const {
    (void)req;
    return {};
  }

  // ---- sampling (M6 6b) ---------------------------------------------------
  // An engine that can draw stochastically advertises it; the scheduler
  // then hands every admitted request's spec to its slot immediately BEFORE
  // prefill (the prefill pick is the first draw). The default engine is
  // greedy-only: the scheduler refuses a stochastic request at submit —
  // identically on every rank — so this default only ever sees greedy specs,
  // and treats anything else as the contract violation it is.
  virtual bool supports_sampling() const { return false; }
  virtual void configure_sampling(int req, const sample::Params& sampling,
                                  uint64_t seed) {
    (void)req;
    (void)seed;
    if (sampling.temperature > 0.0f)
      throw std::logic_error(
          "SchedulerEngine: this engine samples greedily only");
  }
  // Logprobs: an engine that reports them returns, after each prefill/step,
  // one Result per token that op returned (in order) for a slot whose spec
  // asked (sampling.logprobs >= 0 through configure_sampling with
  // `logprobs`). The default reports none; the scheduler refuses a request
  // that asks at submit.
  virtual bool supports_logprobs() const { return false; }
  // Arms slot `req` to report logprobs (`logprobs` >= 0: the top-N count,
  // also carried in the spec's sampling.logprobs) or not (-1). Called right
  // after configure_sampling, before the prefill pick.
  virtual void configure_logprobs(int req, int logprobs) {
    (void)req;
    if (logprobs >= 0)
      throw std::logic_error("SchedulerEngine: this engine reports no logprobs");
  }
  virtual std::vector<sample::Result> take_logprobs(int req) {
    (void)req;
    return {};
  }

  // ---- constrained decoding (M6 6g) ----------------------------------------
  // An engine that can mask the pick advertises it; the scheduler then hands
  // every admitted request's grammar spec to its slot right after the
  // sampling spec, before the prefill pick (the first constrained position).
  // The engine keeps the grammar state per slot, advances it with every
  // token it commits, and applies the next position's mask on every rank
  // identically (the mask is a pure function of the spec, the committed
  // ids and the tokenizer). The default engine has no masks: the scheduler
  // refuses an active spec at submit, identically on every rank.
  virtual bool supports_constraints() const { return false; }
  virtual void configure_constraint(int req, const text::GrammarSpec& grammar) {
    (void)req;
    if (grammar.active())
      throw std::logic_error(
          "SchedulerEngine: this engine cannot constrain the pick");
  }
  // The logit bias: an engine that can add a per-request
  // bias to the logits before the pick advertises it; the scheduler hands
  // every admitted request's entries to its slot right after the grammar
  // (an empty list clears the slot's bias). The default engine has none.
  virtual bool supports_logit_bias() const { return false; }
  virtual void configure_logit_bias(int req,
                                    const std::vector<LogitBias>& bias) {
    (void)req;
    if (!bias.empty())
      throw std::logic_error(
          "SchedulerEngine: this engine cannot bias the pick (logit_bias)");
  }

  // ---- prefix cache (M7 stage B, DESIGN §8) ----------------------------------
  // An engine with a snapshot arena advertises it; the scheduler then keeps
  // the index (glm_prefix_cache.hpp) and drives these ops — identically on
  // every rank, because every decision is a function of the journaled
  // request stream. The default engine has no arena: the scheduler never
  // calls any of them and the op stream is exactly the pre-cache one.
  struct PrefixInfo {
    int arena_slots = 0;       // snapshot slots the engine holds (0: none)
    int64_t align = 1;         // every snapshot position is a multiple (kpool)
    int64_t block_tokens = 0;  // the DSA block; 0 = no pool (nothing pinned)
    int64_t chunk_tokens = 2048;  // the prefill's chunk (the cold cuts)
    int step_tokens_max = 1;   // tokens one step may commit (2: the MTP step)
    bool prefill_lookahead = false;  // MTP prefill snapshots also depend on prompt[position]
    bool body_snapshots = true;      // extra snapshots must preserve the prefill's computation
    bool decode_snapshots = true;    // decode state can be reused as a cold-prefill prefix
  };
  virtual PrefixInfo prefix_info() const { return {}; }
  // The hop (M7 under a multi-token step): the request's next step may
  // commit past `position` (= its committed count + 1, pool-aligned) without
  // stopping there; if it commits two tokens, the engine attempts to save the state
  // after the step's first row — the state at `position` — into arena slot
  // `slot` before returning from that step. Capture is decided before sampled
  // fallback: the final token count does NOT say whether it happened.
  // The arm holds until the slot's next step (or its close).
  enum class PrefixHopStatus { kNone, kNotTaken, kStored, kNoBlock };
  struct PrefixHopResult {
    PrefixHopStatus status = PrefixHopStatus::kNone;
    int slot = -1;
    int64_t position = -1;  // the armed position, not an older stored position
  };
  virtual void prefix_arm_hop(int req, int slot, int64_t position) {
    (void)req;
    (void)slot;
    (void)position;
  }
  // Consume the last armed step's outcome. kNotTaken leaves the arena
  // untouched; kNoBlock releases any previous snapshot and leaves it empty.
  // Arming and closing clear previous outcomes; kNone means no result.
  virtual PrefixHopResult take_prefix_hop_result(int req) {
    (void)req;
    return {};
  }
  // The position actually stored in an arena slot, or -1 if it is empty.
  // A multi-token step may skip its armed hop when the cache pool is full;
  // token acceptance alone does not establish that the snapshot was taken.
  virtual int64_t prefix_position(int slot) const {
    (void)slot;
    throw std::logic_error("SchedulerEngine: this engine has no prefix cache");
  }
  // The prefill with the cache: `boundaries` (absolute positions, ascending)
  // are the request's structural cut positions — the cold chunking cuts at
  // every chunk multiple and at the aligned image of each; attach_slot >= 0
  // opens the slot from that arena entry (at attach_position tokens, one
  // of the prompt's cuts) and prefills prompt[attach_position..] only;
  // snap_slot >= 0 asks for a snapshot into that arena slot when a chunk
  // ends at snap_position (a cut, > attach_position), reported in
  // snap_taken. body_snap_slot names an optional earlier document cut in
  // the same walk, reported independently in body_snap_taken. Returns the
  // first token, exactly as prefill() does.
  struct PrefixPrefill {
    const std::vector<int64_t>* boundaries = nullptr;
    const std::vector<ImageInput>* images = nullptr;  // absolute prompt offsets
    int attach_slot = -1;
    int64_t attach_position = 0;
    int snap_slot = -1;
    int64_t snap_position = 0;
    bool snap_taken = false;  // out
    int body_snap_slot = -1;
    int64_t body_snap_position = 0;
    bool body_snap_taken = false;
    // The head cut (2026-09-28): the aligned image of the prompt's first
    // structural boundary past its start, taken on the same walk.
    int head_snap_slot = -1;
    int64_t head_snap_position = 0;
    bool head_snap_taken = false;
  };
  struct PrefillProgress {
    int64_t computed_tokens = 0;
    int32_t first_token = -1;  // -1: more chunks remain
    bool snap_taken = false;
    bool body_snap_taken = false;
    bool head_snap_taken = false;
  };
  virtual void begin_prefill(int, const std::vector<int64_t>&, int64_t, int64_t,
                             const PrefixPrefill&) {
    throw std::logic_error("SchedulerEngine: resumable prefill is unavailable");
  }
  // A positive budget overrides the begin budget for this advance; zero
  // retains it. Implementations must report work within the selected budget.
  virtual PrefillProgress advance_prefill(int, int64_t = 0) {
    throw std::logic_error("SchedulerEngine: resumable prefill is unavailable");
  }
  struct MixedProgress {
    PrefillProgress prefill;
    std::vector<std::vector<int32_t>> decoded;
  };
  // Engine capability, available before any request is admitted. Per-batch
  // eligibility (slots, row capacity, images) is checked by can_mix_prefill.
  virtual bool supports_mixed_prefill() const { return false; }
  virtual bool can_mix_prefill(int, const std::vector<int>&) const { return false; }
  virtual MixedProgress advance_prefill_and_step(int, int64_t, const std::vector<int>&) {
    throw std::logic_error("SchedulerEngine: mixed prefill/decode is unavailable");
  }
  struct MixedGroupProgress {
    std::vector<PrefillProgress> prefills;
    std::vector<std::vector<int32_t>> decoded;
  };
  virtual bool can_mix_prefill_group(const std::vector<int>&, const std::vector<int>&) const { return false; }
  virtual MixedGroupProgress advance_prefill_group_and_step(
      const std::vector<int>&, const std::vector<int64_t>&, const std::vector<int>&) {
    throw std::logic_error("SchedulerEngine: mixed group prefill/decode is unavailable");
  }
  // Several in-flight prefills' next chunks, request i within budgets[i] tokens.
  // An engine with prefill_group_advance() runs them as ONE physical walk
  // (the weights stream once for all of them) — the scheduler then reads
  // every prompt past one aligned chunk in through its cursor and begins
  // as many per tick as the budget has aligned shares, so prompts that
  // arrive together are read in together. The default advances one by one.
  virtual bool prefill_group_advance() const { return false; }
  virtual bool prefill_budget_per_reader() const { return true; }
  virtual std::vector<PrefillProgress> advance_prefill_group(const std::vector<int>& reqs,
                                                             const std::vector<int64_t>& budgets) {
    std::vector<PrefillProgress> out;
    for (size_t i = 0; i < reqs.size(); ++i) out.push_back(advance_prefill(reqs[i], budgets.at(i)));
    return out;
  }
  virtual int32_t prefill_cached(int req, const std::vector<int64_t>& prompt,
                                 PrefixPrefill* plan) {
    (void)req;
    (void)prompt;
    (void)plan;
    throw std::logic_error("SchedulerEngine: this engine has no prefix cache");
  }
  // The live slot's state at its current position (`position` tokens
  // committed, a multiple of align — the scheduler's view; the engine
  // verifies) into arena slot `slot`, replacing whatever the slot held.
  virtual void prefix_snapshot(int req, int slot, int64_t position) {
    (void)req;
    (void)slot;
    (void)position;
    throw std::logic_error("SchedulerEngine: this engine has no prefix cache");
  }
  // Drops arena slot `slot`'s references (the blocks its entry pinned).
  virtual void prefix_release(int slot) {
    (void)slot;
    throw std::logic_error("SchedulerEngine: this engine has no prefix cache");
  }
  // The arena's measured costs, for the metrics.
  struct PrefixEngineStats {
    int64_t snapshots = 0;   // arena snapshots taken (prefill and rolling)
    double snapshot_ms = 0;  // their summed device time, where measured
    int64_t attaches = 0;
    double attach_ms = 0;
    int64_t snapshot_bytes = 0;  // one slot's state bytes
  };
  virtual PrefixEngineStats prefix_engine_stats() const { return {}; }

  // ---- the NVMe cold tier (issue #26) ---------------------------------------
  // An engine with a slab advertises it; the scheduler then keeps the disk
  // index (sched/disk_cache.hpp) and drives these ops — the decisions
  // identically on every rank, the bytes each rank's own. A spill copies
  // an arena entry (its blob and the cache blocks its metadata pins) to
  // slab pages; a restore reads records into fresh pool blocks and an
  // arena slot. Both run asynchronously on the engine's side and report
  // through disk_poll(); rank 0 journals the outcome once every rank has
  // reported, and the scheduler commits or aborts on that record.
  struct DiskInfo {
    bool enabled = false;
    int64_t pages = 0;        // slab records this rank holds
    size_t page_bytes = 0;    // one cache block's planes, rounded to 4 KiB
    int64_t blob_pages = 0;   // pages one snapshot blob spans
    int64_t min_tokens = 0;   // entries below this position are not spilled
    int max_ops = 4;          // spills and restores started but not committed
    bool persist = false;     // the slab outlives the process (its index is kept beside it)
    int op_timeout_s = 0;     // an op without progress this long fails (0: never)
  };
  virtual DiskInfo disk_info() const { return {}; }
  // Opens the tier: the slab file (created, sized to the capacity), the
  // staging buffers and the worker. Before the scheduler is built (it reads
  // disk_info() then) and before any capture. Throws when the file cannot
  // be made or the capacity cannot hold one blob and one block.
  struct DiskEnable {
    std::string slab_path;        // this rank's slab file
    size_t capacity_bytes = 0;    // the slab's size (pages x page bytes below it)
    // One staging buffer (two device + two pinned): two slices of it move
    // per tick, so 64 MiB is 128 MiB of cache state per tick.
    size_t chunk_bytes = size_t{64} << 20;
    int64_t min_tokens = 0;
    int max_ops = 4;              // ops in flight at most (the scheduler's bound)
    int rank = 0;
    // The slab outlives the process: its index and CRC table are kept
    // beside it under `binding` (the checkpoint, the settings, the slab
    // geometry), and a start under the same binding offers them back.
    bool persist = false;
    std::string binding;
    // Explicit reuse must refuse an incompatible/missing index or a world
    // disagreement without clearing the cache the operator selected.
    bool require_existing = false;
    // Explicit allowance for tests/experiments: RAM-backed storage and buffered
    // I/O would otherwise consume memory outside the model's budget.
    bool allow_buffered = false;
    std::string required_mount;  // local placement assertion, checked again before slab mutation
    // An op that makes no progress this long fails — its buffers held by a
    // stuck read or write come back when the worker does — so the requests
    // waiting on a restore admit cold instead of never (0: never).
    int op_timeout_s = 0;
    // A test hook on every page read (write=false) or write: 0 proceeds,
    // a positive value sleeps that many milliseconds first, a negative one
    // fails the I/O.
    std::function<int(bool write, int64_t page)> io_hook;
  };
  virtual void disk_enable(const DiskEnable& enable) {
    (void)enable;
    throw std::logic_error("SchedulerEngine: this engine has no cold tier");
  }
  // An arena entry's cache blocks by physical identity (the pool's block
  // id and the generation of its contents), the partial block's identity
  // (0: none) and the draft block's position.
  struct DiskBlocks {
    std::vector<uint64_t> identities;
    uint64_t partial_identity = 0;
    int64_t mtp_position = 0;
  };
  virtual DiskBlocks disk_entry_blocks(int slot) const {
    (void)slot;
    throw std::logic_error("SchedulerEngine: this engine has no cold tier");
  }
  // Writes arena slot `slot`: the blob to pages [blob_page, +blob_pages),
  // full block i to pages[i] (-1: already on disk, skipped) and the partial
  // block to partial_page (-1: none).
  struct DiskSpill {
    uint64_t op = 0;
    int slot = -1;
    int64_t blob_page = -1;
    std::vector<int64_t> pages;
    int64_t partial_page = -1;
    // The entry's token ids, to the run of `ids_pages` pages at `ids_page`
    // (right behind the blob): the slab describes its entries itself.
    std::vector<int64_t> ids;
    int64_t ids_page = -1;
    int ids_pages = 0;
  };
  virtual void disk_spill_begin(const DiskSpill& spill) {
    (void)spill;
    throw std::logic_error("SchedulerEngine: this engine has no cold tier");
  }
  // Reads an entry back into arena slot `slot`: acquires one fresh pool
  // block per page (the scheduler made them free), adopts the slot's
  // metadata at `position`, and reads the records asynchronously. Returns
  // the fresh blocks' identities. The slot reads as filled from here on; a
  // failed restore is released with prefix_release like any entry.
  struct DiskRestore {
    uint64_t op = 0;
    int slot = -1;
    int64_t position = 0;
    int64_t mtp_position = 0;
    int64_t blob_page = -1;
    std::vector<int64_t> pages;
    int64_t partial_page = -1;
  };
  virtual DiskBlocks disk_restore_begin(const DiskRestore& restore) {
    (void)restore;
    throw std::logic_error("SchedulerEngine: this engine has no cold tier");
  }
  // The tier's per-tick pump: issues the next bounded slice of an op's
  // device work on the model stream (the engine thread's alone — nothing
  // runs on a second stream beside the fabric's collectives) and hands
  // finished slices to the file worker. Called at every tick's top.
  virtual void disk_pump() {}
  // The ops completed on this rank since the last poll, in completion order.
  struct DiskCompletion {
    uint64_t op = 0;
    bool ok = false;
  };
  virtual std::vector<DiskCompletion> disk_poll() { return {}; }
  struct DiskEngineStats {
    int64_t spills = 0, restores = 0, failures = 0;
    int64_t bytes_written = 0, bytes_read = 0;
    double spill_ms = 0, restore_ms = 0;
    int64_t pending = 0;  // ops queued or running on the tier's worker
  };
  virtual DiskEngineStats disk_engine_stats() const { return {}; }
  // ---- across restarts (nvme_cache.persist) ------------------------------
  // The previous run's index the world adopted (empty: none), read at the
  // scheduler's construction; each adopted entry's token ids come back
  // through disk_read_pages (count pages from page into dst, CRC-checked).
  virtual std::vector<uint8_t> disk_adopted_index() const { return {}; }
  virtual bool disk_read_pages(int64_t page, int64_t count, void* dst) {
    (void)page;
    (void)count;
    (void)dst;
    return false;
  }
  // The committed index after every change, for the tier to keep beside
  // the slab.
  // ... with the entries' content digests in the index's order (this
  // rank's: the digest of its pages' checksums at commit).
  virtual void disk_persist_index(const std::vector<uint8_t>& bytes, const std::vector<uint64_t>& digests) {
    (void)bytes;
    (void)digests;
  }
  // This rank's digest of the pages' checksums as they stand (the index
  // binds an entry to its content with it; 0: no tier).
  virtual uint64_t disk_pages_digest(const std::vector<int64_t>& pages, uint64_t position) const {
    (void)pages;
    (void)position;
    return 0;
  }
  // The adopted index's content digests, in its order.
  virtual std::vector<uint64_t> disk_adopted_digests() const { return {}; }
  // The warm handshake: this rank's candidate index digest ("0": none),
  // and the world's verdict — keep the candidate for adoption or start
  // the slab fresh.
  virtual std::string disk_candidate_digest() const { return "0"; }
  virtual void disk_adopt(bool keep) { (void)keep; }
};

// One request, in arrival (manifest) order. `prompt` ids are validated by
// the ENGINE (the scheduler is model-agnostic by design).
struct SchedulerRequest {
  std::string id;
  std::vector<int64_t> prompt;
  int max_steps = 1;      // tokens to generate (prefill pick included)
  int cancel_after = 0;    // 0 = never; N = retire (Cancelled) once N
                           // tokens have been generated. N in [1, max_steps].
  // Generate past the end-of-sequence set: the request runs to max_steps
  // whatever it draws (OpenAI's ignore_eos). A benchmark wants a fixed
  // number of decode steps rather than a reply that stops when the model
  // decides it is done. The EOS set itself is unchanged — an ignored EOS
  // is still sampled, still streamed, and still counts as a token.
  bool ignore_eos = false;
  // The sampling spec (glm_sampler.hpp's warper contract). The default is
  // GREEDY — temperature 0, no draw, no seed consumed — so every manifest
  // and gate that predates sampling keeps its exact op stream.
  sample::Params sampling = sample::greedy_params();
  // The counter RNG's seed. Rank 0 assigns one when the client omits it and
  // the journal carries it, so every rank draws the same sequence.
  uint64_t seed = 0;
  // Logprobs on the wire: -1 = none; N >= 0 = report every generated
  // token's log-probability and its top-N alternatives (sampling.logprobs
  // carries N to the sampler). Greedy requests report under the raw
  // distribution.
  int logprobs = -1;
  // Constrained decoding (M6 6g): the tool-call grammar the pick obeys
  // (tool_choice required / named / none, parallel_tool_calls false). The
  // default is inactive — unconstrained, the exact op stream every gate
  // pins. Rides the journal with the request.
  text::GrammarSpec grammar;
  // The prefix cache (M7): the prompt's structural boundaries — absolute
  // positions in [1, prompt.size()), ascending — where the cold prefill
  // cuts (at their pool-aligned images) so a snapshot of one turn is a cut
  // of the next; the service derives them from the role-marker tokens in
  // the prompt ids. Empty: cuts at chunk multiples only. `no_cache` opts
  // the request out: no attach, no snapshot. Both ride the journal.
  std::vector<int64_t> boundaries;
  bool no_cache = false;
  // prompt_cache_key: pins the prefix-cache entries this request creates
  // or attaches to — memory and the NVMe tier evict unpinned entries
  // first. A priority, never a lookup key (matching stays by exact token
  // prefix). Rides the journal with the request.
  std::string cache_key;
  // The logit bias: the request's entries, applied by the
  // engine on every rank; rides the journal.
  std::vector<LogitBias> logit_bias;
  std::vector<ImageInput> images;
};

// The bounded admission queue at capacity (submit() only). A load-shed
// event, not a manifest error — the service answers 503 + Retry-After.
struct QueueFullError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Defined after Scheduler (it carries Scheduler::Result by value).
class SchedulerObserver;

// The admission policy (M6 6d). Full-reserve pins blocks_for(prompt +
// max_steps) at admission — a request never stalls mid-generation, at the
// cost of blocks held for tokens an early EOS never writes. Grow-on-demand
// reserves blocks_for(prompt + min(max_steps, window)) at admission and
// grows the reservation at tick top, BEFORE any engine op, whenever the
// next step would write past it — by `window_tokens` at a time, or by the
// minimum when the pool is short. Unattached prefix entries are evicted LRU
// to fit that minimum; if it still does not fit, SHEDS the youngest active
// request (retired Done / kPoolExhausted, the
// service's finish_reason "length") until it does; a request with no
// younger peer sheds itself. Every decision is a pure function of the
// scheduler's state, so the journal keeps it identical on every rank. The
// trade is explicit and the policy opt-in: optimistic admission for
// early-EOS workloads against a truncated answer for the youngest when
// everyone runs long (preemption by recompute — vLLM's answer — waits on a
// prefill fast enough to re-read a partial answer).
struct AdmissionPolicy {
  enum class Mode : int { kFullReserve = 0, kGrowOnDemand = 1 };
  Mode mode = Mode::kFullReserve;
  int window_tokens = 256;  // grow: the initial headroom and the growth step
  int prefill_budget_tokens = 0;  // 0: monolithic; otherwise total prefill tokens per tick
  int prefill_idle_budget_tokens = 0;  // 0: use the same budget; otherwise larger chunks without active decode
  // Serving supplies default latency targets; zero retains fixed budgeting.
  int decode_gap_target_ms = 0;
  int first_token_target_ms = 0;
  PrefillLatencyTuning prefill_latency;
  bool prefill_budget_automatic = false;  // busy is a starting point, idle the adaptive ceiling
  // The prefix cache's entry policy (2026-09-28), rank-identical like the
  // rest of the record. No snapshot below prefix_min_tokens: a 45-token
  // probe's entries must not push a 180K conversation out of the arena
  // (the single-Spark replay of 2026-09-28: three probes per turn emptied
  // a 13-slot arena every turn). With prefix_head_snapshots a cold prefill
  // also keeps the cut at its first structural boundary past the start —
  // a long system prompt's end — where the next conversation under that
  // prompt, or the turn after a client compacted its history, attaches.
  int prefix_min_tokens = 0;
  bool prefix_head_snapshots = false;
  // prompt_cache_key pins protect their entries while the pinned entries
  // hold at most this share of the arena's slots and of the NVMe slab's
  // pages; past it a pin protects nothing. 100: always.
  int prefix_pin_max_pct = 50;
  // Resolved admission policy. Serving resolves automatic admission from the
  // engine's execution capabilities; old journal records retain false.
  // Engines without a shared walk can pay for additional decode passes.
  bool prefill_early_admission = false;
  bool operator==(const AdmissionPolicy& o) const {
    return mode == o.mode && window_tokens == o.window_tokens && prefill_budget_tokens == o.prefill_budget_tokens &&
           prefill_idle_budget_tokens == o.prefill_idle_budget_tokens &&
           decode_gap_target_ms == o.decode_gap_target_ms && first_token_target_ms == o.first_token_target_ms &&
           prefill_latency == o.prefill_latency &&
           prefill_budget_automatic == o.prefill_budget_automatic &&
           prefill_early_admission == o.prefill_early_admission &&
           prefix_min_tokens == o.prefix_min_tokens && prefix_head_snapshots == o.prefix_head_snapshots &&
           prefix_pin_max_pct == o.prefix_pin_max_pct;
  }
  bool operator!=(const AdmissionPolicy& o) const { return !(*this == o); }
  static const char* name(Mode m) {
    return m == Mode::kGrowOnDemand ? "grow" : "full";
  }
};

class Scheduler {
 public:
  struct Result {
    enum class Status : int { kQueued, kActive, kDone, kCancelled };
    // kStop: the service's stop string matched — a natural end
    // for the client (finish_reason "stop"), journaled like a cancel.
    enum class Reason : int { kNone, kEos, kSteps, kCancelled, kPoolExhausted, kStop };
    Status status = Status::kQueued;
    Reason reason = Reason::kNone;
    int slot = -1;          // engine slot used; -1 while queued
    int steps_done = 0;     // tokens generated (prefill pick included)
    std::vector<int64_t> generated;
  };

  // The /v1/metrics snapshot.
  struct Meters {
    int active = 0;        // requests with an open engine slot
    int prefilling = 0;    // subset of active, not yet eligible for decode
    int queued = 0;        // admitted-not, waiting on slots/budget
    int terminal = 0;      // retired (any reason), cumulative since construction
    // The request records held — every live request's, plus the retired
    // ones a batch scheduler keeps (set_keep_retired) — and the prompt and
    // generated ids those records hold. A drained service scheduler reads
    // 0 / 0: its memory tracks the live requests, not the traffic.
    int64_t records = 0;
    int64_t record_tokens = 0;
    int64_t pool_blocks_total = 0;
    int64_t pool_blocks_in_use = 0;
    int64_t tokens_generated = 0;  // cumulative across all requests
    int64_t reservations_grown = 0;  // grow-on-demand: growth events
    int64_t requests_shed_pool = 0;  // grow-on-demand: shed at exhaustion
    // The throughput line's counters: prompts completed and tokens
    // processed (including in-progress/cancelled chunks; all, and the
    // ones actually computed — an attach skips the prefix), decode passes
    // and the request-rows they carried, and the
    // wall time spent inside the engine's prefill and step calls.
    int64_t prompts_prefilled = 0;
    int64_t prompt_tokens = 0;
    int64_t prompt_tokens_computed = 0;
    int64_t decode_steps = 0;
    int64_t decode_rows = 0;
    // Budgeted advances only: groups count once, tokens sum their readers.
    int64_t prefill_chunk_steps = 0;
    int64_t prefill_chunk_budget_tokens = 0;
    int64_t prefill_chunk_available_tokens = 0;  // min(budget, remaining prompt), before structural cuts
    int64_t prefill_chunk_computed_tokens = 0;
    int64_t prefill_adaptive_limit_tokens = 0;  // last leader-selected ceiling; 0 = fixed policy
    // The decoders' own pass the controller takes out of the allowance
    // before sizing a chunk, at the last tick's load (ms; 0 with no
    // decoders, none measured yet, or fixed budgeting).
    double prefill_decode_pass_ms = 0.0;
    int64_t mixed_steps = 0;  // subset of decode_steps that also advanced prefill
    double mixed_ms = 0.0;    // included in BOTH prefill_ms and step_ms; subtract once when summing
    double prefill_ms = 0.0;
    double prefill_request_ms = 0.0;  // summed per-request waits (a group credits each member)
    double step_ms = 0.0;
    // Draft acceptance by position, engine-wide cumulative.
    SchedulerEngine::MtpAcceptance mtp;
    SchedulerEngine::DecodeBatchStats decode_batch;
    // The prefix cache (M7): its slots and live entries, the attach and
    // miss counts, the prompt tokens attaches skipped, the entries taken
    // (at prefill cuts / from rolling snapshots at close), rolling
    // snapshots, evictions, duplicates, snapshots skipped for want of a
    // slot, and the DSA blocks entries pin against the pool.
    int prefix_slots = 0;
    int prefix_entries = 0;
    int64_t prefix_hits = 0;
    int64_t prefix_misses = 0;
    int64_t prefix_tokens_saved = 0;
    int64_t prefix_snapshots = 0;
    int64_t prefix_head_snapshots = 0;
    int64_t prefix_close_entries = 0;
    int64_t prefix_rolling = 0;
    int64_t prefix_hops = 0;
    int64_t prefix_evictions = 0;
    int64_t prefix_duplicates = 0;
    int64_t prefix_skipped = 0;
    int64_t prefix_skipped_no_block = 0;
    int64_t prefix_skipped_hop_fallback = 0;
    int64_t prefix_skipped_hop_mismatch = 0;
    int64_t prefix_skipped_image_bytes = 0;
    int64_t prefix_image_bytes = 0;
    int64_t prefix_blocks_pinned = 0;
    int prefix_pinned = 0;       // entries a prompt_cache_key pins
    // The NVMe cold tier (issue #26): the slab's pages, the entries and
    // block records it holds, spills and restores begun / committed /
    // failed, spills skipped for room, disk evictions, the ops in flight
    // and the prompt tokens restores brought back.
    bool disk_enabled = false;
    int64_t disk_pages_total = 0;
    int64_t disk_pages_used = 0;
    int64_t disk_page_bytes = 0;
    int disk_entries = 0;
    int64_t disk_blocks = 0;
    int64_t disk_spills = 0;
    int64_t disk_spilled = 0;
    int64_t disk_spill_failed = 0;
    int64_t disk_spill_skipped = 0;
    int64_t disk_restores = 0;
    int64_t disk_restored = 0;
    int64_t disk_restore_failed = 0;
    int64_t disk_evictions = 0;
    int64_t disk_blocks_shared = 0;
    int64_t disk_pending = 0;
    int64_t disk_tokens_restored = 0;
    int disk_pinned = 0;         // disk entries a prompt_cache_key pins
    int64_t disk_adopted = 0;    // entries taken over from the previous run's index
  };

  // `eos_token_ids` — the config's end-of-sequence set (empty disables
  // EOS retirement, e.g. --no-eos). `queue_limit` — the admission
  // queue's bound (0 = unbounded, the manifest default; a service sets
  // it so a full queue sheds load with a 503 instead of eating memory).
  // `prefix_slots` (M7): the snapshot slots the prefix cache may use — -1
  // = every slot the engine's arena holds, 0 = the cache off, N = at most
  // N (must not exceed the arena: every rank must run the same count, so
  // rank 0's resolved value rides the warm record and the peers pass it).
  Scheduler(SchedulerEngine* engine, std::vector<int64_t> eos_token_ids,
            int queue_limit = 0, AdmissionPolicy policy = AdmissionPolicy{},
            int prefix_slots = -1);
  const AdmissionPolicy& admission_policy() const { return policy_; }
  bool adaptive_prefill() const {
    return policy_.prefill_budget_tokens > 0 &&
           (policy_.decode_gap_target_ms > 0 || policy_.first_token_target_ms > 0);
  }
  // Leader only. Call after applying arrivals/cancellations, then journal the
  // result and pass it to tick on every rank. Standalone tick selects locally.
  std::optional<int64_t> next_prefill_budget() const;
  // The position the next prefill chunk starts at: the deepest reader's
  // (its attached prefix plus the tokens computed so far); 0 with none.
  int64_t prefill_context() const;
  // The prefix cache as configured: its slot count (0 = off) and the
  // running digest of every decision it made (the journal's cross-rank
  // check — identical streams, identical digests).
  int prefix_slots() const { return cache_.slots(); }
  uint64_t prefix_digest() const { return cache_.digest(); }
  const PrefixCache& prefix_cache() const { return cache_; }

  // The NVMe cold tier (issue #26). Spills and restores start as decisions
  // of the tick (every rank the same); their outcomes are journaled
  // inputs: rank 0 collects every rank's completion of an op and journals
  // one commit, and apply_disk_commits() is called with the record's
  // commits BEFORE the tick that follows them — on every rank in the same
  // order. poll_disk_completions() returns this rank's own completions
  // since the last poll (the peers send them to rank 0 on the journal's
  // return path; rank 0 folds its own in). Every decision folds into the
  // prefix digest, so the journal's cross-rank check covers the tier.
  struct DiskCommit {
    uint64_t op = 0;
    bool ok = false;
  };
  void apply_disk_commits(const std::vector<DiskCommit>& commits);
  std::vector<SchedulerEngine::DiskCompletion> poll_disk_completions();
  bool disk_enabled() const { return disk_.enabled(); }
  int64_t disk_ops_pending() const { return static_cast<int64_t>(disk_ops_.size()); }
  const DiskCache& disk_cache() const { return disk_; }

  // Arrival order = FCFS priority. Throws on an empty/duplicate id, a
  // nonpositive max_steps, or a cancel_after outside [1, max_steps] —
  // manifest errors are operator errors, and they must fail identically
  // on every rank (a request that only exists on some ranks would
  // deadlock the fabric). Throws QueueFullError when the bounded queue
  // is full.
  void submit(SchedulerRequest request);

  // The service form: false only on a full bounded queue (a normal,
  // load-shedding event). Validation failures still throw — they are
  // client bugs, not load.
  bool try_submit(SchedulerRequest request);

  // Flags a live request for retirement at the next tick boundary (a
  // client disconnect). Returns false when the id is unknown or already
  // terminal — a late cancel is a no-op, never an error.
  bool cancel(const std::string& id);
  // The service's stop: the client's stop string matched on
  // rank 0 — journaled like a cancel, applied at the next tick's sweep on
  // every rank, retiring the request as Done with Reason::kStop (the
  // service cuts the text at the match; the tokens after it are dropped).
  // A cancel that arrives with it wins.
  bool stop(const std::string& id);

  // One policy quantum: the cancel sweep, at most one admission, then one
  // engine pass over up to decode_batch_capacity() active requests. Returns
  // false when nothing is pending AFTER the tick — the final retirement may
  // ride on the false. Throws on admission deadlock exactly like
  // run_to_completion().
  bool tick(std::optional<int64_t> prefill_budget = std::nullopt);

  // Any queued or active request remains.
  bool has_pending() const;

  // Runs every request to a terminal state: a tick loop. Single-threaded
  // and allocation-free on the hot path (the engine owns all buffers).
  void run_to_completion();

  // Parallel to submit() order; entries reach their terminal Status
  // only via run_to_completion()/tick(). A set_keep_retired(false)
  // scheduler's holds only the live (queued, active) requests' entries
  // after every tick, and nothing once drained.
  const std::vector<Result>& results() const { return results_; }

  // Result by request id (nullptr when unknown) — the service's
  // non-streaming lookup.
  const Result* find(const std::string& id) const;

  // Lifecycle events (the SSE tap). Not owned; may be null.
  void set_observer(SchedulerObserver* observer) { observer_ = observer; }

  // What a retired request leaves behind. Either way its prompt, grammar,
  // bias, boundaries, cache cuts and generated ids are released at retire
  // (the observer has seen every token; the result holds the copy) and
  // only a tombstone — the id, status and counts — stays. The default
  // keeps every tombstone and its result for the scheduler's lifetime:
  // the batch contract, results() read back after run_to_completion().
  // A persistent service passes false: the result's tokens go with the
  // rest once on_retire has seen them, and every retired record is
  // compacted away at the end of the tick — the same quantum on every
  // rank, since the journal carries every tick — so after any tick, under
  // any load, the scheduler holds exactly its live requests. Reported
  // 2026-09-13 by a third-party tester: 8 bytes per prompt token per
  // request for the process's lifetime (21 MiB over one two-hour agent
  // session) and every tick scanning the whole history. Dropped ids
  // leave the duplicate check with them (a service's ids are its own
  // counter); a late cancel of one is the no-op an unknown id gets.
  void set_keep_retired(bool keep) { keep_retired_ = keep; }

  Meters meters() const;

 private:
  enum class State : int { kQueued, kActive, kTerminal, kPrefilling };

  struct Request {
    SchedulerRequest spec;
    State state = State::kQueued;
    uint64_t prefill_epoch = 0;
    uint64_t prefill_tick = 0;  // readers begun together may finish as one group
    uint64_t decode_epoch = 0;
    int slot = -1;
    int steps_done = 0;
    // Tokens of the current step's batch still unapplied when a token is
    // applied: a retire with step_tail > 0 dropped the rest of a multi-
    // token step, so the model's state sits step_tail rows past the
    // committed position (no retire-time snapshot there, 2026-09-10).
    int step_tail = 0;
    std::vector<int64_t> generated;
    bool cancel_requested = false;  // external cancel, applied at the
                                    // next tick's sweep
    bool stop_requested = false;    // the stop-string retire (Reason::kStop)
    int64_t reserved_tokens = 0;    // the slot's current reservation
    // The prefix cache (M7): the prompt's cuts and their prefix hashes
    // (computed once at submit), the entry the request attached to (its
    // index; -1 cold), the position it attached at, and the rolling
    // snapshot slot with the position it holds (-1: none yet).
    std::vector<int64_t> cuts;
    std::vector<uint64_t> cut_hashes;
    int attach_entry = -1;
    int64_t attach_position = 0;
    int rolling_slot = -1;
    int64_t rolling_position = -1;
    int64_t hop_armed = -1;    // the aligned position armed for the next step
    bool cache_off = false;    // the pool cannot hold the cache's blocks for it
    uint64_t restore_op = 0;   // the cold tier restore this queued request waits on (0: none)
    // Its restore could not start this tick because an op in flight holds
    // what it needs (a slot busy spilling, another entry's restore): it
    // waits a tick rather than prefilling cold; re-decided at every tick's top.
    bool restore_blocked = false;
    PrefixCache::Images cache_images;
    // The retire line's numbers: the admission clock, the
    // prefill's wall and the prompt tokens an attach skipped, and the
    // decode passes this request rode (each shared with every other live
    // request in the pass).
    bool admitted = false;
    std::chrono::steady_clock::time_point admitted_at{};
    double prefill_ms = 0.0;
    int prefill_snap_slot = -1;
    int64_t prefill_snap_position = 0;
    int prefill_body_slot = -1;
    int64_t prefill_body_position = 0;
    int prefill_head_slot = -1;
    int64_t prefill_head_position = 0;
    int64_t prefill_computed = 0;
    int64_t attached_tokens = 0;
    int decode_passes = 0;
  };
  // The admission plan the cache proposes for a queued request: the entry
  // to attach (or -1) and the position, and the cut a new entry would be
  // taken at (0: none).
  struct PrefixPlan {
    int attach_entry = -1;
    int64_t attach_position = 0;
    int64_t snap_position = 0;
    int64_t body_snap_position = 0;
    int64_t head_snap_position = 0;
    int disk_entry = -1;  // a cold tier entry deeper than the memory attach (issue #26)
  };
  bool cache_on(const Request& r) const {
    return cache_.enabled() && !r.spec.no_cache && !r.cache_off &&
           (r.spec.images.empty() || engine_->supports_image_prefix_cache());
  }
  PrefixPlan plan_prefix(const Request& r) const;
  // The structural boundaries the engine's cold prefill cuts at (2026-09-28):
  // only where this deployment can take a snapshot — the cache on and the
  // boundary's aligned image at or past the entry floor — so a cache hit
  // and a miss are the same walk, and so is a request that opts out. A cut
  // a snapshot never stands on served nothing and made a prompt's cold walk
  // differ from its rows in a group walk (the DSA sites are not
  // split-invariant).
  std::vector<int64_t> cut_boundaries(const Request& r) const;
  void finish_prefill_snapshot(Request& r, int slot, int64_t position, bool taken,
                               bool head = false);
  // The pool block a snapshot's private partial-block copy takes: one when
  // the position is not block-aligned, none otherwise (or without a pool).
  int64_t snapshot_blocks(int64_t position) const;
  // Whether a snapshot at `position` into a slot whose current snapshot
  // sits at `previous` (-1: none) needs a NEW pool block — the arena
  // releases the old copy before it takes the new one, so only a first
  // partial copy does.
  bool snapshot_needs_block(int64_t position, int64_t previous) const;
  // Frees pool blocks for growth or a snapshot by evicting LRU unattached entries
  // until `need` blocks are free; false when nothing evictable remains.
  bool ensure_free_blocks(int64_t need, const std::string& id);
  // The blocks an admission must find FREE: the reservation less the full
  // blocks an attach shares.
  int64_t new_blocks(const Request& r, const PrefixPlan& plan) const;
  // Rolling snapshots (tick, before the step): every active cached request
  // at an aligned committed position it has not snapshotted yet.
  void rolling_snapshots();
  // A free arena slot, or the LRU unattached entry's (released on the
  // engine, an "evict" event under `id`); -1 when neither exists.
  int acquire_arena_slot(const std::string& id);
  // Releases a slot on the engine and returns it to the ledger.
  void free_arena_slot(int slot);
  void emit_prefix(const std::string& id, const char* op, int64_t position,
                   int slot);
  // A miss explained at INFO: the cuts probed, the entries held, and where
  // the prompt parts from the entry it shares the most with.
  void log_prefix_miss(const Request& r) const;
  // The NVMe cold tier (issue #26). A memory entry worth keeping cold
  // spills as soon as it is inserted (up to the op bound; the tick's top
  // starts what the bound deferred), so it is busy — never evicted — from
  // birth to commit;
  // a restore starts when a queued request's deepest usable state is on
  // disk. Every start is a decision (the digest folds it); the commits
  // arrive through apply_disk_commits.
  void want_spill(int entry, const std::string& id);
  void start_spills();
  bool start_spill(int entry, const std::string& id);
  // At the tick's top, after the spills: every queued request whose deepest
  // usable state is on disk starts its restore (or waits on the one in
  // flight for that entry) before any admission path reads the queue —
  // the chunked begin admits long prompts without next_admissible.
  void start_restores();
  bool start_restore(int disk_entry, Request& r);
  // Evicts one memory entry (LRU, unattached, not busy) and tells the
  // cold tier when it held the entry's state too. The victim's slot, or -1.
  int evict_memory_entry();
  // The committed disk index to the tier's keeping (nvme_cache.persist),
  // after every change to it.
  void persist_disk_index();
  // A memory entry's cold copy went (a disk eviction): drops its link.
  void drop_disk_link(int memory_entry, int disk_entry);

  bool is_eos(int32_t token) const;
  // The reservation an admission pins: the lifetime under full-reserve,
  // prompt + min(max_steps, window) under grow-on-demand.
  int64_t initial_reserve_tokens(const SchedulerRequest& spec) const;
  int64_t reserve_blocks(const Request& r) const;
  // Grow-on-demand's tick-top pass (after the cancel sweep, before any
  // admission or step): every active request whose next step would write
  // past its reservation grows it, evicting LRU unattached cache entries
  // before shedding the youngest if the pool cannot cover even the minimum.
  void grow_reservations();
  int youngest_active_after(int arrival) const;
  int free_slot() const;
  // The LRU unattached cache entries give way to a request that needs their
  // blocks — never a block a live request holds (an in-flight prefill keeps
  // its own refcount on every reserved block, and a victim only drops the
  // cache's extra ref). The planned attach is touched first so it is
  // evicted last; the plan is recomputed after every eviction, because the
  // victim may have been that entry. True when the request's new blocks
  // fit; false when nothing evictable remains and they do not.
  bool evict_to_fit(const Request& r, int64_t& free_blocks);
  // The oldest queued request whose reservation fits a free slot (no
  // head-of-line blocking), or -1. With the prefix cache on it may EVICT
  // unattached entries (LRU) to make the blocks a request needs.
  int next_admissible();
  // The queued requests that admit together with `first` this tick: cold
  // prompts within the engine's group span limit, no prefix-cache plan,
  // fitting the free slots and blocks; empty when the engine has no group
  // prefill or `first` itself takes a cache plan.
  std::vector<int> admissible_group(int first, int64_t budget);
  void admit_group(const std::vector<int>& arrivals);
  // admit()'s slot-side halves: the slot and its configuration before the
  // engine's prefill, the bookkeeping after it.
  int admit_prepare(int arrival);
  void admit_finish(int arrival, int slot, int32_t token, double prefill_ms, int64_t attached,
                     bool resumed = false);
  // The submit() validations, shared by submit()/try_submit().
  void validate_new(const SchedulerRequest& request) const;
  int queued_count() const;
  void admit(int arrival);
  void begin_prefill(int arrival, int64_t budget);
  void advance_prefill(int arrival, int64_t budget);
  void advance_prefill_group(const std::vector<int>& arrivals, const std::vector<int64_t>& budgets);
  void apply_prefill_progress(int arrival, const SchedulerEngine::PrefillProgress& progress, int64_t budget, double ms);
  // The chunked-prefill predicate behind the admit dispatch: a positive
  // budget, chunkable inputs (plain text, or images on an engine that
  // chunks them), and a prompt longer than one tick's budget.
  bool needs_chunked_prefill(int arrival, int64_t budget) const;
  // One iteration of the tick's admit loop: the oldest fitting queued
  // request admits (group, one-shot, or chunked start) against the tick's
  // remaining prefill cap. Only first_prefill may start a chunked prompt
  // or admit an oversized image on an engine without image chunking;
  // either consumes the remaining cap. Other one-shots/groups must fit
  // the cap. Returns false when nothing admitted. A zero policy budget
  // retains one monolithic admission event per tick.
  bool admit_fitting(int64_t& tick_cap, int64_t budget, bool first_prefill);
  void step_batch(const std::vector<int>& arrivals);
  void apply_step_batch(const std::vector<int>& arrivals, const std::vector<std::vector<int32_t>>& batches);
  // Appends one token and applies terminal conditions in their canonical
  // order. Returns true when the request retired. `logprobs` (optional)
  // rides to the observer with it.
  bool append_token(int arrival, int32_t token,
                    const sample::Result* logprobs = nullptr);
  // The engine's logprobs for the tokens it just returned, when the request
  // asked; empty otherwise. Throws when the engine returned a different
  // count than tokens.
  std::vector<sample::Result> collect_logprobs(int arrival, int slot,
                                                   size_t tokens);
  // Retire conditions are checked in this order: natural EOS first, then
  // scripted cancellation, then the steps cap — a cancelled request that
  // had already finished naturally reports Done (client intent cannot
  // rewrite history), and a cancel_at the cap reports Cancelled.
  void retire(int arrival, Result::Status status, Result::Reason reason);
  // The retire's release: everything but the tombstone (set_keep_retired).
  void release_retired(Request& r, Result& res);
  // tick()'s body, the quantum; tick() compacts after it.
  bool quantum(std::optional<int64_t> prefill_budget);
  // Erases the retired records when they are not kept, at the end of every
  // tick: the live requests keep their order, and the slot map, the
  // decode/prefill round-robin cursors and the deferral log follow their
  // new indices (a cursor on a retired record moves to the nearest live one
  // before it, so the next slice starts where it would have).
  void compact_retired();

  SchedulerEngine* engine_ = nullptr;
  std::vector<int64_t> eos_ids_;
  std::vector<Request> requests_;  // arrival order — the FCFS order
  std::vector<int> slots_;         // engine slot -> arrival index, or -1
  std::vector<Result> results_;     // parallel to requests_
  int cursor_ = -1;                // last-stepped arrival (round-robin)
  int prefill_cursor_ = -1;        // last-advanced prefill (round-robin when the budget shrinks)
  std::vector<int> mixed_stepped_; // arrivals already decoded in this quantum's shared walk
  int deferred_logged_ = -1;       // arrival of the current deferral log
  SchedulerObserver* observer_ = nullptr;
  bool keep_retired_ = true;       // the batch contract (set_keep_retired)
  int64_t retired_ = 0;            // cumulative retirements (meters)
  int queue_limit_ = 0;            // 0 = unbounded
  int decode_batch_capacity_ = 1;  // fixed engine pass width
  int64_t tokens_generated_ = 0;   // cumulative on_token counter
  AdmissionPolicy policy_;
  PrefillLatencyController prefill_latency_;
  int64_t prefill_pass_budget_tokens_ = 0;  // before final-tail allocation trims
  int64_t prefill_adaptive_limit_tokens_ = 0;
  double prefill_decode_pass_ms_ = 0;  // meters: the learned decoders' pass at the last tick's load
  int64_t grows_ = 0;              // growth events (meters)
  int64_t pool_sheds_ = 0;         // requests shed at exhaustion (meters)
  int64_t prompts_prefilled_ = 0;  // the throughput line's counters (meters)
  int64_t prompt_tokens_ = 0;
  int64_t prompt_tokens_computed_ = 0;
  int64_t decode_steps_ = 0;
  int64_t decode_rows_ = 0;
  int64_t prefill_chunk_steps_ = 0;
  int64_t prefill_chunk_budget_tokens_ = 0;
  int64_t prefill_chunk_available_tokens_ = 0;
  int64_t prefill_chunk_computed_tokens_ = 0;
  int64_t mixed_steps_ = 0;
  double mixed_ms_ = 0.0;
  double prefill_ms_ = 0.0;
  double prefill_request_ms_ = 0.0;
  double step_ms_ = 0.0;
  PrefixCache cache_;              // the prefix cache's index (M7)
  SchedulerEngine::PrefixInfo prefix_info_;
  uint64_t ticks_ = 0;             // the LRU clock
  uint64_t execution_epoch_ = 0;   // replicated ordering of prefill/decode starts
  // The NVMe cold tier (issue #26): the disk index, the engine's slab
  // geometry, the op counter (identical on every rank: every op is a
  // decision), the ops awaiting a journaled commit and the memory entries
  // queued for a spill (their index and hash, so a reused index is not
  // mistaken for the entry that asked).
  DiskCache disk_;
  SchedulerEngine::DiskInfo disk_info_;
  uint64_t next_disk_op_ = 1;
  struct DiskOp {
    uint64_t op = 0;
    bool restore = false;
    int disk_entry = -1;
    int memory_entry = -1;   // spill: the memory entry copied
    int slot = -1;           // the arena slot involved
    int64_t position = 0;
    std::string id;          // the request the decision was made for (the log)
    SchedulerEngine::DiskBlocks restored;  // restore: the fresh blocks' identities
  };
  std::vector<DiskOp> disk_ops_;
  struct DiskWant {
    int entry = -1;
    uint64_t hash = 0;
    std::string id;
  };
  std::vector<DiskWant> disk_wanted_;
  bool restore_in_flight() const;
};

// Streaming lifecycle events for the service (SSE). Fired inline on the
// ticking thread — see the threading note above.
class SchedulerObserver {
 public:
  virtual ~SchedulerObserver() = default;
  // The observer's running fold of everything it has recorded (the
  // fabric's continuous drift check, M9): has_digest() says it keeps one,
  // digest() is its value now. The default observer keeps none.
  virtual bool has_digest() const { return false; }
  virtual uint64_t digest() const { return 0; }
  // One generated token (the prefill pick is steps_done == 1). Always
  // fires BEFORE the matching retire when the token ends the request.
  virtual void on_token(const std::string& id, int64_t token,
                        int steps_done) = 0;
  // The token's logprobs, right after its on_token, for requests that
  // asked (SchedulerRequest::logprobs >= 0). Default: ignored.
  virtual void on_token_logprobs(const std::string& id, int steps_done,
                                 const sample::Result& logprobs) {
    (void)id;
    (void)steps_done;
    (void)logprobs;
  }
  // The request left the queue for an engine slot and its prefill begins
  // (every admission path, grouped prefills included). Timing only: the
  // op stream does not record it. Default: ignored.
  virtual void on_admit(const std::string& id, int slot) {
    (void)id;
    (void)slot;
  }
  // The request's terminal state, exactly once (EOS, steps cap,
  // scripted or external cancellation, pool exhaustion all land here).
  virtual void on_retire(const std::string& id,
                         const Scheduler::Result& result) = 0;
  // Grow-on-demand grew the request's reservation to `reserved_tokens`
  // (the op stream records it: a growth decision is rank-identical state).
  virtual void on_grow(const std::string& id, int64_t reserved_tokens) {
    (void)id;
    (void)reserved_tokens;
  }
  // A prefix-cache decision (M7): `op` is "attach" (the request opened from
  // an entry at `position`, arena `slot`), "snapshot" (an entry taken at a
  // prefill cut), "rolling" (a live request's rolling snapshot), "close"
  // (a rolling snapshot became an entry at retire), "evict" (an entry
  // freed; `id` is the request whose admission, growth or snapshot needed
  // the slot or blocks),
  // "drop" (a rolling slot released without becoming an entry), "hop"
  // (a snapshot captured inside an MTP step), "hop_fallback" (capture was
  // skipped before sampled fallback passed the hop; `position` is the old
  // snapshot or -1), or "hop_mismatch" (inconsistent snapshot dropped;
  // `position` is what the arena reported). The NVMe cold tier (issue #26)
  // adds "spill" (a copy to disk begun from arena `slot`), "spilled" /
  // "spill_fail" (its journaled outcome), "restore" (a read back into arena
  // `slot` begun), "restored" / "restore_fail" (its outcome) and
  // "disk_evict" (a disk entry freed for room; slot -1). Rank-identical
  // state: the op stream records every one.
  virtual void on_prefix(const std::string& id, const char* op,
                         int64_t position, int slot) {
    (void)id;
    (void)op;
    (void)position;
    (void)slot;
  }
};

}  // namespace dgpp::sched
