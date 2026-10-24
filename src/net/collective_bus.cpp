// CollectiveBus implementation (DESIGN §6/§6.1). The engine thread owns
// every verbs object and all slot/credit state; other threads only submit,
// wait, and stop. One loop iteration is bounded and ordered: latency
// intake, bulk intake, CQ drains, receive recycling, credit harvest,
// watchdogs — never an unbounded pass, never a syscall on the hot path.

#include "net/collective_bus.hpp"

#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <format>
#include <ctime>
#include <deque>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "common/log.hpp"
#include "common/cuda_wait.hpp"
#include "net/bus_types.hpp"
#include "net/tcp.hpp"
#include "net/verbs.hpp"

namespace dgpp::net {

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kMaxIntakeLatency = 64;    // latency requests per iteration
constexpr int kMaxIntakeBulkPosts = 16;  // bulk stripes per iteration
constexpr int kMaxPollPerCq = 32;        // CQEs drained per CQ per iteration
// Hot-spin iterations before the 50us-sleep phase. Large on purpose: the
// engine is a dedicated poller (§6.1), and a round trip (send: ~25us of
// credit latency; collective: the kernel launch tail) must land inside
// the hot phase or the sleep cadence (~50-110us) shows up in every
// measurement. The sleep phase is a power courtesy for genuine idles —
// never a latency mechanism.
constexpr int kEngineSpinIterations = 2000;
constexpr int kEngineIdleSleepUs = 50;
constexpr size_t kMaxLatencySamples = 100000;
// Spin before entering the scheduler on a request wait. LONG on purpose
//: the decode step's eager pick (two ~150us collectives
// between graph windows) fell out of a 500us spin into the futex in ~2% of
// steps, and the wake back came 7-10ms later — the woken main thread
// landed behind a hot spinner and waited out a CFS slice (PREEMPT_NONE,
// HZ=250). Every other rank then waited for that rank at the next step's
// first collective: the whole fabric's p99 was one thread's nap. A
// waiter whose request completes in the spin window costs nothing; one
// that does not burns at most this much of one core before sleeping.
constexpr int kWaitSpinUs = 20000;
constexpr int kConnectRetryMs = 500;
constexpr size_t kMaxErrorText = 4096;

inline uint64_t monotonic_ns() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
         static_cast<uint64_t>(ts.tv_nsec);
}
inline double elapsed_us_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}
double elapsed_us(Clock::time_point from) {
  return std::chrono::duration<double, std::micro>(Clock::now() - from)
      .count();
}

// Pinned-flag acquire load (NIC- and GPU-written cells; DESIGN §2.3/§6).
inline uint32_t acquire_u32(const volatile uint32_t* p) {
  return __atomic_load_n(const_cast<const uint32_t*>(p), __ATOMIC_ACQUIRE);
}

inline uint64_t acquire_u64(const volatile uint64_t* p) {
  return __atomic_load_n(const_cast<const uint64_t*>(p), __ATOMIC_ACQUIRE);
}

}  // namespace

struct BusStripe {
  int lane = 0;
  BusPool pool = BusPool::kLatency;
  uint32_t slot = 0;
  uint32_t seq = 0;
};

struct BusRequest {
  uint64_t id = 0;
  int peer_rank = -1;
  BusMessageClass cls = BusMessageClass::kLatency;
  const void* data = nullptr;  // caller-owned until completion
  size_t len = 0;
  size_t offset = 0;  // bytes submitted as stripes so far

  // Collective (§6.3): device buffers, element count, control-cell stamp.
  // peer_rank stays -1 (the whole world); stripes are per-peer, and
  // owner_stripe indexes stripe_hashes by peer, not by stripe.
  bool is_collective = false;
  const void* dev_src = nullptr;
  void* dev_dst = nullptr;
  size_t elems = 0;
  uint32_t ctl_seq = 0;
  // Staging-ring generation (the §6.3 staging interface): >= 0 marks a
  // pre-staged collective (the producing GEMM wrote the peer-0 buffer
  // through stage_next()); the engine uses this exact generation for every
  // peer. -1 = device-source mode; the engine picks a generation.
  int stage_gen = -1;

  // Bulk (prefill-class) collective: geometry precomputed at submit.
  // Stripes are bulk-slot units of the whole buffer; the engine walks
  // segments of at most bulk_seg_stripes stripes and splits each
  // segment's stripes across the ranks (its plan).
  bool is_bulk = false;
  uint32_t bulk_stripes = 0;       // C = ceil(elems / stripe_elems)
  uint32_t bulk_seg_stripes = 0;    // bulk_slots * lanes (pool-bounded)
  uint32_t bulk_seg_count = 0;     // ceil(C / bulk_seg_stripes)

  int outstanding = 0;  // stripes credited back so far are subtracted
  std::vector<BusStripe> stripes;
  std::vector<uint64_t> stripe_hashes;  // aligned with stripes, filled on credit
  Clock::time_point submitted;

  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  // Set (release) after the mutex-guarded fields: a spinning waiter may
  // read the results through an acquire on this flag without the mutex.
  std::atomic<bool> done_flag{false};
  bool ok = false;
  std::string error;
};

// Cheap spin hint (not the sched_yield syscall — that surrenders the
// timeslice; the engine measured 2.2ms poll latency doing it).
inline void cpu_relax() {
#if defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#elif defined(__x86_64__)
  asm volatile("pause" ::: "memory");
#endif
}

struct CollectiveBus::Impl {
  struct SendSlot {
    uint32_t gen = 0;          // last generation posted into this slot
    uint32_t credit_seen = 0;  // last credit seq harvested from the cell
    bool in_flight = false;
    // Owning reference: a watchdog-failed request may still receive its
    // late credit (or a lane failure) after the waiter reaped it, so the
    // slot keeps the request alive until it is cleared.
    std::shared_ptr<BusRequest> owner;
    size_t owner_stripe = 0;
  };
  struct RecvSlot {
    uint32_t expect_seq = 1;  // next arrival generation
    bool arrived = false;     // doorbell CQE seen, GPU ack pending
    size_t len = 0;
  };
  struct LaneState {
    std::unique_ptr<RcLane> lane;
    BusLaneEndpoint endpoint;      // ours, for the exchange table
    BusLaneEndpoint peer_endpoint;  // the peer's, learned from the table
    uint32_t stage_lkey = 0;       // staging-block MR lkey on this lane's device
    std::vector<SendSlot> send[2];  // [latency, bulk]
    std::vector<RecvSlot> recv[2];
    // Ring cursor per pool: plain SENDs are consumed FIFO by the peer's
    // receive ring, so the i-th message must target ring slot i mod depth —
    // a free-slot-first-fit would break the ring alignment silently.
    uint32_t cursor[2] = {0, 0};
    // The receive side recycles in ring order too: reposting slot r+1
    // before r would reorder the RQ and misalign the sender's cursor.
    uint32_t recycle_cursor[2] = {0, 0};
    size_t in_flight_count = 0;
    uint64_t write_posts = 0;  // the one-sided transport's row WRITEs (signal cadence)
    // Bulk pacing: the earliest time the next bulk stripe may post on
    // this lane (BusOptions::bulk_pace_gbps).
    Clock::time_point bulk_next_post{};
    bool failed = false;
    Clock::time_point last_progress;
    BusLaneStats stats{};
  };

  BusOptions opt;
  BusSlabLayout layout{};
  std::vector<std::unique_ptr<VerbsDevice>> devices;  // deduped by name
  std::vector<int> peer_ranks;                       // ascending
  std::vector<std::vector<LaneState>> peers;         // [peer][lane]

  std::mutex lat_q_mu;
  std::deque<std::shared_ptr<BusRequest>> lat_q;
  std::mutex bulk_q_mu;
  std::deque<std::shared_ptr<BusRequest>> bulk_q;

  std::mutex registry_mu;
  std::unordered_map<uint64_t, std::shared_ptr<BusRequest>> registry;
  uint64_t next_id = 1;

  std::atomic<bool> stopping{false};
  std::mutex stop_mu;  // serializes quiesce()/stop() for one instance
  bool quiesced = false;
  bool stopped = false;
  std::thread engine;
  // The engine loop's iteration times (2026-10-20): the loop body IS the
  // handshake's notice latency, so its histogram (<2 <5 <10 <20 <50 >=50
  // us) and maximum, kept per graph window and printed with the window
  // timeline, bound what the notice can add and show the outliers (an
  // interrupt, a completion burst, a lock) that reach the fold as skew.
  uint64_t loop_hist[6] = {};
  uint64_t loop_n = 0;
  double loop_max_us = 0;
  // DGPP_BUS_TIMELINE_DUMP=<dir> (2026-10-20): every graph generation's
  // stamps (its own clock: start, staged, first and last peer gated, done;
  // per peer landed and claimed), buffered per window and appended to
  // <dir>/bus_gens_rank<r>.tsv at the window's timeline log, so one
  // collective can be aligned across the ranks offline (the symmetric sums
  // of a pair's landing times give the path, their differences the pair's
  // relative lateness up to the clocks' constant offset).
  struct GenRecord {
    uint64_t gen, start, stage, first, last, done;
    uint64_t landed[kBusMaxPeers], claim[kBusMaxPeers];
  };
  std::vector<GenRecord> gen_dump;
  FILE* gen_dump_file = nullptr;
  bool gen_dump_checked = false;

  // One stream per consumer kernel: a persistent kernel starves every
  // launch queued behind it on the same stream, so the lane-1 consumer
  // must not share the lane-0 consumer's stream.
  std::vector<cudaStream_t> consumer_streams;
  bool consumers_launched = false;
  std::mutex stats_mu;
  BusStats stats_store;

  // ---- collective state (§6.3) --------------------------------------------
  // The ctl cell is one pinned cache line the kernel and the engine share;
  // the stream serializes per-collective kernels. coll_q/coll_active make
  // single-outstanding airtight across the submit/engine interface; coll_mode
  // rejects harness sends once the bus is in collective mode (their
  // messages would be claimed — and folded — by a peer's collective
  // kernel: a silent-corruption class we refuse to ship). Atomic: the
  // engine writes it at eager pickup, the graph session writes it at
  // record_begin (a second, non-engine writer — TSan's contract).
  std::mutex coll_mu;
  std::deque<std::shared_ptr<BusRequest>> coll_q;
  std::atomic<bool> coll_active{false};
  std::atomic<bool> coll_mode{false};  // send() and intake() gate on it
  std::atomic<bool> coll_poisoned{false};  // engine or waiter failure closes submissions
  // A world of one (2026-09-10, the single-Spark serve): no lanes, no
  // rendezvous, no engine thread — every collective is the identity (a
  // copy when src and dst differ), every graph hook succeeds, the staging
  // handout is one pinned slot. The engines above run unchanged.
  bool world1 = false;
  std::mutex w1_mu;
  uint64_t w1_next_id = 1;
  std::unordered_set<uint64_t> w1_pending;
  void* w1_stage = nullptr;
  bool w1_stage_held = false;
  bool w1_recording = false;
  int w1_armed = 0;
  cudaStream_t collective_stream = nullptr;
  BusAllReduceCtl* ar_ctl = nullptr;
  BusBulkScratch* bulk_scratch = nullptr;  // the bulk grid's shared records
  // The one-sided transport's per-block records (pinned): one array of
  // kBusWriteMaxBlocks per graph cell, then per stream cell.
  BusWriteBlockRecord* write_recs = nullptr;
  static constexpr size_t kWriteRecCells =
      static_cast<size_t>(kBusMaxGraphVariants) * kBusMaxGraphGens + kBusStreamRing;
  BusWriteBlockRecord* write_recs_of(int variant, int node) const {
    return write_recs + (static_cast<size_t>(variant) * kBusMaxGraphGens + static_cast<size_t>(node)) * kBusWriteMaxBlocks;
  }
  BusWriteBlockRecord* write_recs_of_stream(int cell) const {
    return write_recs + (static_cast<size_t>(kBusMaxGraphVariants) * kBusMaxGraphGens + static_cast<size_t>(cell)) * kBusWriteMaxBlocks;
  }
  // The records of a cell, graph or stream (the cell arrays are contiguous).
  BusWriteBlockRecord* write_recs_of_cell(const BusAllReduceCtl* cell) const {
    if (cell >= graph_cells && cell < graph_cells + static_cast<size_t>(kBusMaxGraphVariants) * kBusMaxGraphGens)
      return write_recs + static_cast<size_t>(cell - graph_cells) * kBusWriteMaxBlocks;
    return write_recs_of_stream(static_cast<int>(cell - strm.cells));
  }
  // The write transport's aggregate of a generation's block records: every
  // block staged (post) / every block done (complete; the cell's stamps and
  // status are filled from the records).
  void write_aggregate(BusAllReduceCtl* cell, uint32_t seq, uint32_t elems, bool* ready_all, bool* done_all) {
    const BusWriteBlockRecord* recs = write_recs_of_cell(cell);
    const uint32_t blocks = bus_write_blocks(bus_write_chunks(elems));
    bool ready = true, done = true;
    for (uint32_t b = 0; b < blocks; ++b) {
      ready = ready && acquire_u32(&recs[b].ready) == seq;
      done = done && acquire_u32(&recs[b].done) == seq;
    }
    *ready_all = ready;
    *done_all = done;
    if (!done) return;
    uint64_t stage = 0, finish = 0, first = 0, last = 0;
    uint32_t status = 0;
    uint64_t claim[kBusMaxPeers] = {};
    uint64_t claim_first[kBusMaxPeers] = {};
    for (uint32_t b = 0; b < blocks; ++b) {
      stage = std::max(stage, recs[b].gt_stage);
      finish = std::max(finish, recs[b].gt_done);
      status |= recs[b].status;
      for (size_t p = 0; p < peer_ranks.size() && p < kBusMaxPeers; ++p) {
        claim[p] = std::max(claim[p], recs[b].gt_peer[p]);
        if (recs[b].gt_peer[p] != 0 && (claim_first[p] == 0 || recs[b].gt_peer[p] < claim_first[p]))
          claim_first[p] = recs[b].gt_peer[p];
      }
    }
    for (size_t p = 0; p < peer_ranks.size() && p < kBusMaxPeers; ++p) {
      cell->gt_claim[p] = claim[p];
      cell->gt_claim_first_neg[p] = claim_first[p] != 0 ? ~claim_first[p] : 0;
      if (claim[p] != 0 && (first == 0 || claim[p] < first)) first = claim[p];
      last = std::max(last, claim[p]);
    }
    cell->gt_stage = stage;
    cell->gt_done = finish;
    cell->gt_first = first;
    cell->gt_last = last;
    __atomic_store_n(&cell->status, status, __ATOMIC_RELEASE);
  }
  bool write_transport() const { return opt.latency_transport == BusLatencyTransport::kWrite; }
  // Collective sequence number: the engine thread is the only writer; the
  // submitter reads it (relaxed) for staging-handout rotation. Atomic so
  // that read is race-free by the letter, not just by the protocol.
  std::atomic<uint32_t> ctl_seq_counter{0};
  std::atomic<uint64_t> completion_epoch{0};
  uint64_t ar_deadline_cycles = 0;  // cached device-clock rate conversion
  uint64_t ar_last_ready_seq = 0;   // one-shot ready-log gating (per bus)
  struct CollectiveFlight {
    std::shared_ptr<BusRequest> req;
    std::vector<BusStripe> claims;  // per peer: claimed lane/slot
    uint64_t posted_bits = 0;       // bit p once peer p's stripe is posted
    Clock::time_point launched_at{};
    Clock::time_point picked_at{};  // engine pickup (queue wait ends)
    double launch_call_us = 0;      // the cudaLaunchKernel call itself
    uint64_t launched_gt_ns = 0;  // launch return, on the globaltimer base
    int stall_dumps = 0;            // bring-up microscope rate control
    int stage_gen = -1;              // staging ring generation in flight
  } coll;                            // engine thread only

  // ---- collective staging block (§6.3 interface, M5 d3) -----------------------
  // Fixed pinned buffers, kStageRing-deep per peer PLUS one dedicated
  // "self" row: collective payloads are sent from the peer rows (a SEND's
  // local address is any MR-registered memory; only the doorbell's remote
  // ring position must track the receiver's ring — the unchanged cursor
  // discipline). The self row is the pre-stage handout: the producing
  // GEMM writes it, and the collective kernel snapshots it into every
  // peer row BEFORE folding in place — the peer rows are the send
  // sources, so the engine's posts can never race the in-place fold (the
  // first cut aliased the handout with peer 0's row and the posts read
  // folded bytes; the rows must be disjoint, by construction).
  // Reuse safety: generation g's peer row is rewritten at g+kStageRing,
  // and the engine's post for g is in-order behind g-1's on the same QP
  // while the rewrite (a later kernel, stream-ordered after the
  // g+kStageRing-1 exit which required arrival of g+kStageRing-1) cannot
  // precede it — the RC per-QP ordering is the reuse fence.
  // Single-outstanding makes it airtight for the eager path.
  static constexpr int kStageRing = 8;
  // The graph kernel derives its staging row from the generation with this
  // rotation — the protocol bound is shared through the header so the
  // recorded launch and the engine's posting agree forever.
  static_assert(kStageRing == kBusMaxGraphStageRing,
                "the engine's staging ring must match the graph protocol's");
  uint8_t* stage_block = nullptr;  // [peers + 1][kStageRing][stage_pitch()]
  std::vector<ibv_mr*> stage_mrs;  // one per distinct device (lkey source)
  // A staging row's pitch: the write ring's chunked row (2026-10-19),
  // which is the latency slot and its trailers — the send machinery's
  // rows use the first lat_slot_bytes of it.
  size_t stage_pitch() const { return layout.write_row_bytes; }
  uint8_t* stage_buf(size_t peer, int gen) const {
    return stage_block +
           ((peer * static_cast<size_t>(kStageRing) +
             static_cast<size_t>(((gen % kStageRing) + kStageRing) % kStageRing)) *
            stage_pitch());
  }
  // The pre-stage handout row (index `peers`, after every peer row).
  uint8_t* self_buf(size_t peer_count, int gen) const {
    return stage_buf(peer_count, gen);
  }
  // Bulk arena: one row per peer after the staging rows; each row holds
  // one segment's outbound stripes (stripe k at row + k*bulk_slot_bytes).
  // Registered with the same per-device MRs (one superblock, one lkey).
  uint8_t* bulk_arena_row(size_t peer) const {
    return stage_block +
           ((peer_ranks.size() + 1) * static_cast<size_t>(kStageRing) *
                stage_pitch()) +
           peer * static_cast<size_t>(bulk_seg_bytes());
  }
  uint64_t* bulk_staged_counters() const {
    return reinterpret_cast<uint64_t*>(
        bulk_arena_row(peer_ranks.size()));
  }
  // The staging kernel's per-(peer, stripe) folds — the hashes the doors
  // carry — after the counters (kBusMaxPeersSized slots), [peer][stripe].
  uint64_t* bulk_staged_hashes() const {
    return bulk_staged_counters() + kBusMaxPeersSized;
  }
  size_t bulk_seg_bytes() const {
    return static_cast<size_t>(opt.bulk_slots) * lane_count() *
           static_cast<size_t>(opt.bulk_slot_bytes);
  }
  uint32_t bulk_seg_stripes() const {
    return static_cast<uint32_t>(bulk_seg_bytes() / opt.bulk_slot_bytes);
  }
  // Submit-thread handout state (guarded by coll_mu with the queue).
  void* stage_held_ptr = nullptr;
  int stage_held_gen = -1;

  // Bulk segment machine, engine thread only. The arena-reuse fence is
  // the pair of monotonic counters below: a segment's staging may only
  // overwrite an arena row after every posted payload WR has its doorbell
  // CQE reaped (bulk_pool SENDs are closed under coll_mode, so all bulk
  // doorbell CEs in the air belong to this machine — the global equality
  // is exact).
  struct BulkFlight {
    int phase = -1;  // -1 idle, 0 = reduce-scatter, 1 = allgather
    uint32_t segment = 0;
    bool launched = false;
    std::vector<uint32_t> posted;  // per peer, this segment's stripes
    BusBulkSegPlan plan{};          // the launched segment's plan
  } bulk;
  uint64_t bulk_pairs_posted = 0;     // every bulk post_pair (harness too)
  uint64_t bulk_doorbell_ce_seen = 0;  // retired doorbell WRs (monotonic)

  // ---- graph era (§6.2: decode graph variants, one replay at a time) -------
  // Each registered graph shape owns a disjoint generation-cell set. The
  // forward thread selects one variant per window; the engine adopts that
  // selection with the same release/acquire publication as its generation
  // range. The era opens at the first record_begin and never closes — eager
  // collectives still run between windows, while harness sends remain closed.
  // Thread ownership:
  //   forward thread — the session fields (coll_mu), arm, finish, poison
  //                    of the remaining window on stop;
  //   engine thread  — the walk (graph_pass) and its flight/carrier;
  //   both           — the window atomics (release publish, acquire adopt)
  //                    and the failure flag (first-writer-wins).
  struct GraphState {
    // Session (forward thread, coll_mu-guarded).
    bool recording = false;
    int recording_variant = -1;
    // The engine reads this in fail_lane (a lane failure poisons the
    // era) with no other synchronization — atomic, relaxed.
    std::atomic<bool> recorded{false};  // at least one graph exists
    struct GenMeta {        // per node; write-once before the era publishes
      uint32_t elems;       // the post length (bytes = elems*2)
    };
    struct Variant {
      bool recorded = false;
      int gens = 0;  // recorded collective nodes (G)
      std::vector<GenMeta> meta;
    };
    std::array<Variant, kBusMaxGraphVariants> variants;
    // The kernel deadline baked into every recorded launch (computed at
    // record_begin, outside capture; the cycle-deadline conversion is
    // per-device and stable).
    uint64_t deadline_cycles = 0;
    // Window publication. window_count signals "a new window exists" (the
    // engine adopts on change); window_variant and window_first describe it.
    // Arm reserves the selected variant's G
    // generations from the SHARED collective counter — eager pickups
    // between windows take theirs from the same one — so firsts no
    // longer derive from window_count (the old arithmetic assumed the
    // era owned the numbering from gen 1; prefill before the first arm
    // already breaks that). Publication order: window_first (relaxed),
    // then window_count (release) — the engine's window_count acquire
    // makes the window_first store visible. window_count 0 = nothing
    // armed (pre-first-arm).
    std::atomic<uint64_t> window_count{0};
    std::atomic<uint64_t> window_first{0};
    std::atomic<int> window_variant{-1};
    // The armed windows (2026-09-06, the pipelined replay): window number
    // w (1-based, window_count after its arm) at ring index (w - 1) %
    // kBusWindowRing, written under coll_mu before window_count's release
    // store and read by the engine after its acquire load. Arm waits for
    // the window kBusMaxLiveWindows back to be walked, so an entry is
    // never rewritten while the engine may still read it.
    struct Window {
      int variant = -1;
      uint64_t first = 0;
      uint64_t gens = 0;
      uint64_t armed_gt = 0;  // arm's host time, on the gt base
    };
    std::array<Window, kBusWindowRing> windows{};
    // coll_mu-guarded (arm/finish and every eager gate run under it):
    // the windows armed and not yet finished — the era's eager gate is
    // live_windows > 0 — the finish count (windows finish in arm order)
    // and the variants whose window is live (their cells are in flight).
    int live_windows = 0;
    uint64_t finished_count = 0;
    std::array<bool, kBusMaxGraphVariants> variant_live{};
    // Engine: the next un-walked generation (monotonic, graph gens only
    // — eager gens between windows are the eager machine's business).
    // Arm c+1 waits for this to pass the armed window's last
    // generation; finish waits for the same bound.
    std::atomic<uint64_t> walk_pub{0};
    std::atomic<bool> failed{false};
    // The era's failure reason (coll_mu-guarded; first failure wins).
    std::string error;
    // Engine walk (engine thread only).
    uint64_t adopted_count = 0;  // window the walk state belongs to
    int adopted_variant = -1;    // selected graph shape for that window
    uint64_t adopted_first = 0;  // the adopted window's first generation
    uint64_t adopted_last = 0;   // first + gens - 1 (inclusive)
    uint64_t walk_seq = 0;        // next generation to complete
    struct Flight {               // at most one generation is un-done at a
      uint64_t gen = 0;          // time: the graph serializes the kernels
      uint64_t posted_bits = 0;   // bit p once peer p's pair is posted
      Clock::time_point progressed_at{};
      Clock::time_point started_at{};
      bool active = false;
    } flight;
    int stall_dumps = 0;  // TEMP bring-up: rate control for the walk dump
    // Per-window timeline accumulators (the kernels' %globaltimer stamps
    // plus the engine's post cost), logged when the window's walk
    // completes — see log_window_timeline.
    struct Timeline {
      uint64_t n = 0;
      double copy_us = 0, handshake_us = 0, skew_us = 0, fold_us = 0,
             total_us = 0, post_us = 0;
      uint64_t passes = 0;  // engine passes while the window was live
      // previous generation's fold done -> this generation's entry: THIS
      // rank's compute between collectives (the ranks' spread here is
      // what the skew measures).
      double compute_us = 0;
      uint64_t prev_gt_done = 0;
      double max_handshake_us = 0, max_skew_us = 0, max_total_us = 0;
      // The window's shortest handshake and, per peer, the shortest staging
      // -> landed (2026-10-20): the path's own latency (engine notice, post,
      // wire, the peer's poll) shows at the minimum, where the averages
      // carry the peers' lateness.
      double min_handshake_us = 0;
      double peer_land_min_us[kBusMaxPeers] = {};
      uint64_t max_total_gen = 0;  // gen_seq of the window's worst total
      uint64_t first_gt_start = 0; // the window's first generation's entry
      uint64_t gate_waits = 0;
      // Wait (handshake + skew) histogram: <20 <50 <100 <200 <500 >=500 us,
      // and the split by generation parity (the decode step alternates
      // attention / FFN boundaries, so parity separates the MoE's
      // imbalance from everything else).
      uint64_t wait_hist[6] = {};
      double wait_even_us = 0, wait_odd_us = 0;
      uint64_t n_even = 0, n_odd = 0;
      double peer_lag_us[kBusMaxPeers] = {};  // staging written -> peer gated
      double peer_land_us[kBusMaxPeers] = {};  // staging written -> the peer's row landed (write transport)
      uint64_t peer_land_n[kBusMaxPeers] = {};
      double peer_first_us[kBusMaxPeers] = {};  // staging written -> the first warp proved the peer

      uint64_t peer_last[kBusMaxPeers] = {};  // times this peer arrived last
      // The gaps between consecutive claims in claim order (2026-09-19): the
      // kernel claims one peer per round, a gate pass over its payload in
      // every round, so co-resident doorbells are gated one after another
      // and the later ones' passes land in `skew`. Gaps pinned at one
      // pass's length are that serialization; gaps that spread are the
      // peers' own arrival spread. Hist: <3 <5 <7 <10 <15 <25 >=25 us.
      double claim_gap_us[kBusMaxPeers] = {};
      uint64_t claim_gap_hist[kBusMaxPeers][7] = {};
    } tl;
    uint64_t armed_gt = 0;  // graph_replay_arm's host time, on the gt base
    // The window's slot-credit carrier: graph posts have no request of
    // their own, but SendSlot ownership (fail_lane completion, credit
    // harvest, the in_flight gate) is request-shaped. A per-window
    // carrier gives the machinery something to hold: unregistered (the
    // watchdog never times it out), is_collective (credits recycle slots
    // but never complete it — completion is the walk's business).
    std::shared_ptr<BusRequest> carrier;
  } graph;
  // ---- the stream collectives (2026-09-14, plan D9) -----------------------
  // The eager fold in the graph kernel form, launched on the caller's
  // stream: the forward thread issues (a generation from the shared
  // counter, a cell of the ring, the FIFO entry, the launch), the engine
  // walks the FIFO in generation order with the replay walk's flight —
  // strictly between windows (the eager gate keeps them apart) and never
  // interleaved with host-driven collectives (both sides reject the
  // other's presence). Thread ownership:
  //   forward thread — issue (coll_mu for the FIFO push), settle;
  //   engine thread  — the walk (stream_pass), cur/active, the timeline;
  //   both           — issued (forward, release) / walked (engine,
  //                    release), the era failure flag.
  struct StreamState {
    BusAllReduceCtl* cells = nullptr;  // pinned [kBusStreamRing]
    struct Gen {
      uint64_t gen = 0;
      int cell = 0;
      uint32_t elems = 0;
    };
    std::deque<Gen> fifo;             // coll_mu: issued, not yet adopted
    std::atomic<uint64_t> issued{0};  // the last generation issued (0: none)
    std::atomic<uint64_t> walked{0};  // the last generation walked (or failed)
    bool active = false;              // engine: cur is adopted
    Gen cur{};
    uint64_t deadline_cycles = 0;     // the kernel deadline (computed at first issue)
    std::shared_ptr<BusRequest> carrier;  // the posts' slot-credit carrier
    GraphState::Timeline tl;          // the pass's generations (settle logs it)
  } strm;
  // Engine-side: stream generations issued and not yet walked.
  bool stream_outstanding() const {
    return strm.active ||
           strm.issued.load(std::memory_order_acquire) > strm.walked.load(std::memory_order_relaxed);
  }
  // Per-generation cells, one kBusMaxGraphGens slab per variant, pinned for
  // the bus's lifetime (baked into each recorded graph's kernel launches).
  BusAllReduceCtl* graph_cells = nullptr;

  BusAllReduceCtl* graph_variant_cells(int variant) const {
    return graph_cells + static_cast<size_t>(variant) * kBusMaxGraphGens;
  }

  BusRankExchange ex_{};  // our frame, built once during start()
  int64_t gt_offset_ns = 0;  // %globaltimer - CLOCK_MONOTONIC (start())

  // ---- helpers -----------------------------------------------------------

  size_t peer_index(int rank) const {
    return static_cast<size_t>(
        std::find(peer_ranks.begin(), peer_ranks.end(), rank) -
        peer_ranks.begin());
  }

  static int pool_index(BusPool pool) {
    return pool == BusPool::kLatency ? 0 : 1;
  }

  size_t pool_slots(BusPool pool) const {
    return pool == BusPool::kLatency ? static_cast<size_t>(opt.lat_slots)
                                     : static_cast<size_t>(opt.bulk_slots);
  }

  size_t lane_count() const { return opt.lane_devices.size(); }

  // Engine-side view builder (the public recv_view() delegates here).
  BusRecvView recv_view_of(const LaneState& state) const {
    BusRecvView view;
    if (!state.lane) return view;
    RcLane& rc = *state.lane;
    const BusSlabLayout& layout = rc.layout();
    uint8_t* slab = rc.slab();
    view.doorbell_lat = layout.recv_doorbell(slab, BusPool::kLatency, 0);
    view.doorbell_bulk = layout.recv_doorbell(slab, BusPool::kBulk, 0);
    view.payload_lat = reinterpret_cast<const uint64_t*>(
        layout.recv_payload(slab, BusPool::kLatency, 0));
    view.payload_bulk = reinterpret_cast<const uint64_t*>(
        layout.recv_payload(slab, BusPool::kBulk, 0));
    view.ack_lat = layout.ack_cell(slab, BusPool::kLatency, 0);
    view.ack_bulk = layout.ack_cell(slab, BusPool::kBulk, 0);
    view.control = layout.control_cell(slab);
    view.lat_slots = static_cast<int>(opt.lat_slots);
    view.bulk_slots = static_cast<int>(opt.bulk_slots);
    view.lat_slot_bytes = static_cast<uint32_t>(opt.lat_slot_bytes);
    view.bulk_slot_bytes = static_cast<uint32_t>(opt.bulk_slot_bytes);
    view.write_ring = layout.write_row(slab, 0);
    view.write_row_bytes = static_cast<uint32_t>(layout.write_row_bytes);
    return view;
  }

  // Terminal path for a collective flight: the flight state resets
  // BEFORE the wake (the single-outstanding gate must see a clean bus
  // before the waiter can submit its next collective), while `finished`
  // keeps the request alive through the completion's trailing notify.
  void finish_flight(std::shared_ptr<BusRequest> finished, bool ok,
                     const std::string& error) {
    if (!ok) coll_poisoned = true;
    if (ok) completion_epoch.fetch_add(1, std::memory_order_release);
    coll = {};
    bulk = {};
    coll_active.store(false, std::memory_order_relaxed);
    complete_request(std::move(finished), ok, error);
  }

  // Makes a (possibly still running) collective kernel exit promptly: the
  // engine stamps the control cell, and the kernel's round check treats any
  // done_seq match as an exit. Idempotent per request.
  void poison_collective(const BusRequest& req) {
    if (req.ctl_seq == 0) return;  // never picked up; no kernel to stop
    __atomic_store_n(&ar_ctl->done_seq, req.ctl_seq, __ATOMIC_RELEASE);
  }

  void record_latency(BusMessageClass cls, double us) {
    std::lock_guard<std::mutex> lock(stats_mu);
    BusClassStats& s = cls == BusMessageClass::kLatency ? stats_store.latency
                                                        : stats_store.bulk;
    if (s.latency_us.size() < kMaxLatencySamples) s.latency_us.push_back(us);
  }

  // Takes the shared_ptr BY VALUE deliberately: the completion's tail
  // (record + notify) must not outlive the request. The waiter can be the
  // last other holder and drops out the moment done_flag publishes —
  // TSan measured the trailing cv notify racing the waiter-side destructor
  // when callers cleared their flight references before calling.
  void complete_request(std::shared_ptr<BusRequest> req, bool ok,
                        const std::string& error) {
    BusRequest* const r = req.get();
    {
      std::lock_guard<std::mutex> lock(r->mu);
      if (r->done) return;
      r->ok = ok;
      r->error = error;
      r->done = true;
    }
    r->done_flag.store(true, std::memory_order_release);
    record_latency(r->cls, elapsed_us(r->submitted));
    r->cv.notify_all();
  }

  // Spin on the done flag BEFORE taking the mutex (spinning under the
  // lock deadlocks the completer out of the request), then fall to the
  // futex. The flag is published release after the mutex-guarded fields,
  // so a waiter that observes it can read the results lock-free. A futex
  // wake measured ~60-100us against a hot engine; the common case lands
  // inside the spin window.
  bool spin_then_wait(BusRequest* req, int timeout_ms) {
    const auto deadline =
        Clock::now() + std::chrono::microseconds(kWaitSpinUs);
    while (!req->done_flag.load(std::memory_order_acquire)) {
      if (Clock::now() > deadline) break;
      cpu_relax();
    }
    if (req->done_flag.load(std::memory_order_acquire)) return true;
    std::unique_lock<std::mutex> lock(req->mu);
    return req->cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                            [req] { return req->done; });
  }

  void fail_lane(LaneState& lane, const std::string& reason) {
    if (lane.failed) return;
    lane.failed = true;
    lane.stats.failed = true;
    const std::string text =
        "bus lane to rank " + std::to_string(lane.stats.peer_rank) +
        " lane " + std::to_string(lane.stats.lane) + " failed: " + reason;
    DGPP_LOG_ERROR("{}", text);
    for (int pool = 0; pool < 2; ++pool) {
      for (SendSlot& s : lane.send[pool]) {
        if (s.in_flight && s.owner) {
          std::shared_ptr<BusRequest> owner = std::move(s.owner);
          s.in_flight = false;
          complete_request(std::move(owner), false, text);
        }
      }
    }
    lane.in_flight_count = 0;
    // A lane failure in the graph era strands every generation that would
    // have posted through it (and the peer's kernels waiting on this
    // side's arrivals). The era is over; the walk's drain-fail path
    // poisons the window's remaining kernels.
    if (graph.recorded.load(std::memory_order_relaxed) ||
        strm.issued.load(std::memory_order_acquire) != 0)
      graph_fail("lane failed: " + reason);
  }

  // ---- engine passes ------------------------------------------------------

  bool submit_stripe(const std::shared_ptr<BusRequest>& req, LaneState& lane,
                     BusPool pool, uint32_t slot, size_t chunk,
                     std::string* error) {
    RcLane& rc = *lane.lane;
    const int p = pool_index(pool);
    SendSlot& ss = lane.send[p][slot];

    uint8_t* dst = rc.layout().send_payload(rc.slab(), pool, slot);
    std::memcpy(dst, static_cast<const uint8_t*>(req->data) + req->offset,
                chunk);

    const uint32_t seq = ss.gen + 1;
    if (!rc.post_send_pair(pool, slot, seq, static_cast<uint32_t>(chunk),
                           error))
      return false;
    // TX retirement accounting (the bulk arena-reuse fence): every bulk
    // post_pair — harness traffic included — must balance the doorbell CEs
    // counted in poll_cqs, or a later collective's arena gate wedges on
    // the mismatch.
    if (pool == BusPool::kBulk) ++bulk_pairs_posted;
    ss.gen = seq;
    ss.in_flight = true;
    ss.owner = req;
    ss.owner_stripe = req->stripes.size();
    ++lane.in_flight_count;
    // A post is progress: arm the lane watchdog from THIS moment, not from
    // whenever traffic last flowed. Otherwise a fresh post after an idle
    // gap (a peer still cold-loading weights on the real mesh) inherits
    // the idle period as if it were a stall and the lane dies on arrival.
    lane.last_progress = Clock::now();
    DGPP_LOG_DEBUG("stripe: peer={} lane={} pool={} slot={} seq={} chunk={}",
                   req->peer_rank, lane.stats.lane, pool_index(pool), slot, seq,
                   chunk);

    req->stripes.push_back(BusStripe{lane.stats.lane, pool, slot, seq});
    req->offset += chunk;
    ++req->outstanding;
    ++lane.stats.posts;
    lane.stats.bytes_sent += chunk;
    return true;
  }

  // Latency intake: one stripe per message, first lane with a free latency
  // slot. Bulk intake: contiguous slot-byte chunks striped round-robin,
  // deterministic lane order so verification can recompute the chunking.
  bool intake() {
    bool worked = false;

    if (coll_mode.load(std::memory_order_relaxed)) {
      std::unique_lock<std::mutex> lock(lat_q_mu);
      while (!lat_q.empty()) {
        complete_request(lat_q.front(), false,
                         "bus is in collective mode; send() is closed");
        lat_q.pop_front();
        worked = true;
      }
      std::unique_lock<std::mutex> bulk_lock(bulk_q_mu);
      while (!bulk_q.empty()) {
        complete_request(bulk_q.front(), false,
                         "bus is in collective mode; send() is closed");
        bulk_q.pop_front();
        worked = true;
      }
      return worked;
    }

    {
      std::unique_lock<std::mutex> lock(lat_q_mu);
      int taken = 0;
      while (!lat_q.empty() && taken < kMaxIntakeLatency) {
        std::shared_ptr<BusRequest> req = lat_q.front();
        if (req->done) {
          lat_q.pop_front();
          ++taken;
          continue;
        }
        const size_t p = peer_index(req->peer_rank);
        bool placed = false;
        std::string hard_error;
        for (LaneState& lane : peers[p]) {
          if (lane.failed) continue;
          const uint32_t slot = lane.cursor[0];
          if (lane.send[0][slot].in_flight) continue;  // ring position busy
          std::string error;
          if (submit_stripe(req, lane, BusPool::kLatency, slot, req->len,
                            &error)) {
            lane.cursor[0] =
                (slot + 1) % static_cast<uint32_t>(opt.lat_slots);
            placed = true;
          } else {
            fail_lane(lane, error);
            hard_error = error;
          }
          break;
        }
        if (placed) {
          req->stripe_hashes.resize(req->stripes.size());
          lat_q.pop_front();
          ++taken;
        } else if (!hard_error.empty()) {
          complete_request(
              req, false,
              "no alive lane to rank " + std::to_string(req->peer_rank));
          lat_q.pop_front();
          ++taken;
        } else {
          break;  // every latency slot busy; retry next iteration
        }
        worked = true;
      }
    }

    {
      std::unique_lock<std::mutex> lock(bulk_q_mu);
      if (!bulk_q.empty()) {
        std::shared_ptr<BusRequest> req = bulk_q.front();
        if (req->done) {
          bulk_q.pop_front();
          worked = true;
        } else {
          const size_t p = peer_index(req->peer_rank);
          const size_t lanes = peers[p].size();
          int posts = 0;
          bool request_done = false;
          while (req->offset < req->len && posts < kMaxIntakeBulkPosts) {
            const size_t stripe_idx = req->stripes.size();
            LaneState& lane = peers[p][stripe_idx % lanes];
            if (lane.failed) {
              complete_request(
                  req, false,
                  "bulk stripe lane " +
                      std::to_string(stripe_idx % lanes) + " to rank " +
                      std::to_string(req->peer_rank) +
                      " is down (no per-lane failover by design)");
              request_done = true;
              break;
            }
            // Ring discipline within the bulk pool: cursor order, gated by
            // the slot still being in flight (the credit for its previous
            // generation).
            const uint32_t slot = lane.cursor[1];
            if (lane.send[1][slot].in_flight) break;  // ring full; defer
            const size_t chunk = std::min(
                static_cast<size_t>(opt.bulk_slot_bytes), req->len - req->offset);
            std::string error;
            if (!submit_stripe(req, lane, BusPool::kBulk, slot, chunk,
                               &error)) {
              fail_lane(lane, error);
              complete_request(req, false, error);
              request_done = true;
              break;
            }
            lane.cursor[1] =
                (slot + 1) % static_cast<uint32_t>(opt.bulk_slots);
            req->stripe_hashes.resize(req->stripes.size());
            ++posts;
            worked = true;
          }
          if (request_done || req->offset == req->len) bulk_q.pop_front();
          // else: partially submitted; stays at the front.
        }
      }
    }
    return worked;
  }

  // ---- collective pass (§6.3) ----------------------------------------------
  // Runs before intake (latency priority). Pickup: claim one latency slot
  // per peer, reset the control cell, launch the per-collective kernel.
  // Posting: peers whose staging bits landed get their payload+doorbell
  // SENDs (the kernel staged the bytes; no host memcpy). Completion: the
  // kernel's ctl stamp completes the request; credits still flow for slot
  // recycling but are off the critical path.
  // The prefill-class machine (§6.3): reduce-scatter over every segment,
  // then allgather. One kernel launch and one posting wave per (phase,
  // segment), everything bounded by the pool depths; the arena-reuse fence
  // is the global TX retirement (bulk_pairs_posted == seen). Engine
  // thread only, sibling of the latency flight path.
  bool bulk_collective_pass(BusRequest& req) {
    bool worked = false;

    if (bulk.phase < 0) {
      // Arena safety: the previous flight's payload WRs must be retired
      // before this flight's first staging overwrites arena rows. All
      // bulk SENDs in the air belong to this machine (harness bulk sends
      // are closed under coll_mode), so the global equality is exact.
      if (bulk_doorbell_ce_seen != bulk_pairs_posted) return worked;
      req.ctl_seq =
          ctl_seq_counter.fetch_add(1, std::memory_order_relaxed) + 1;
      if (req.ctl_seq == 0)  // skip idle 0 (wrap)
        req.ctl_seq =
            ctl_seq_counter.fetch_add(1, std::memory_order_relaxed) + 1;
      req.stripe_hashes.assign(peer_ranks.size(), 0);
      bulk.phase = 0;
      bulk.segment = 0;
      bulk.launched = false;
      worked = true;
    }

    if (!bulk.launched) {
      if (ar_deadline_cycles == 0) {
        ar_deadline_cycles =
            bus_consumer_deadline_cycles(opt.consumer_deadline_s);
        int clock_khz = 0;
        cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
        DGPP_LOG_INFO("allreduce: device clock rate {} kHz (stamps conversion)",
                      clock_khz);
      }
      if (ar_deadline_cycles == 0) {
        finish_flight(coll.req, false,
                      "could not read device clock rate for the "
                      "collective deadline");
        return true;
      }

      // Reset the control cell and the staged counters before launch
      // (program order on this thread covers the kernel's start).
      __atomic_store_n(&ar_ctl->ready_bits, 0, __ATOMIC_RELAXED);
      __atomic_store_n(&ar_ctl->done_seq, 0, __ATOMIC_RELAXED);
      __atomic_store_n(&ar_ctl->status, 0, __ATOMIC_RELAXED);
      ar_ctl->stamp_stage = 0;
      ar_ctl->stamp_first_claim = 0;
      ar_ctl->gt_start = ar_ctl->gt_stage = ar_ctl->gt_first =
          ar_ctl->gt_last = ar_ctl->gt_done = 0;
      for (uint64_t& g : ar_ctl->gt_claim) g = 0;
      for (uint64_t& g : ar_ctl->gt_landed) g = 0;
      for (uint64_t& g : ar_ctl->gt_claim_first_neg) g = 0;
      ar_ctl->stamp_reduce_done = 0;
      __atomic_store_n(&ar_ctl->dbg_gate_waits, 0, __ATOMIC_RELAXED);
      __atomic_store_n(&ar_ctl->dbg_gate_spins, 0, __ATOMIC_RELAXED);
      for (size_t p = 0; p < peer_ranks.size(); ++p)
        __atomic_store_n(&bulk_staged_counters()[p], 0, __ATOMIC_RELAXED);

      // The segment plan: sub-range table = each rank's owned stripes of
      // this segment (RS stages peers' ranges out; AG broadcasts mine;
      // RS folds mine; AG lands peers'). Shards are split PER SEGMENT
      //: a contiguous ceil split of this segment's stripes
      // across the ranks, so every rank folds and broadcasts in every
      // segment. The earlier global-contiguous split put each rank's whole
      // shard inside one segment of a multi-segment buffer, and the world
      // serialized — one rank folding per segment while the others'
      // kernels exited empty (measured: a 16 MiB all-reduce at 42 ms, four
      // 6.8 ms folds end to end per phase). The per-element fold order is
      // the canonical ascending-rank chain whoever owns the stripe, so the
      // result is bitwise the same.
      BusBulkSegPlan plan{};
      plan.total_elems = static_cast<uint32_t>(req.elems);
      plan.stripe_elems = static_cast<uint32_t>(opt.bulk_slot_bytes / 2);
      plan.seg_first = bulk.segment * req.bulk_seg_stripes;
      plan.seg_stripe_count =
          std::min(req.bulk_seg_stripes, req.bulk_stripes - plan.seg_first);
      {
        const uint32_t world = static_cast<uint32_t>(opt.world_size);
        const uint32_t per = plan.seg_stripe_count / world;
        const uint32_t rem = plan.seg_stripe_count % world;
        uint32_t base = 0;
        for (int r = 0; r < opt.world_size; ++r) {
          const uint32_t count = per + (static_cast<uint32_t>(r) < rem ? 1 : 0);
          if (r == opt.my_rank) {
            plan.my_base = base;
            plan.my_count = count;
          } else {
            const size_t p = peer_index(r);
            plan.out_base[p] = base;
            plan.out_count[p] = count;
          }
          base += count;
        }
      }

      BusAllReduceView view{};
      int vi = 0;
      for (size_t p = 0; p < peer_ranks.size(); ++p)
        for (const LaneState& lane : peers[p])
          view.recv[vi++] = recv_view_of(lane);
      view.recv_views = vi;
      view.lanes_per_peer = static_cast<int>(lane_count());
      for (size_t p = 0; p < peer_ranks.size(); ++p)
        view.send_payload[p] = reinterpret_cast<const uint16_t*>(
            bulk_arena_row(p));
      view.send_peers = static_cast<int>(peer_ranks.size());

      const cudaError_t launch = launch_bus_bulk_collective(
          view, opt.my_rank, bulk.phase,
          static_cast<const __nv_bfloat16*>(req.dev_src),
          static_cast<__nv_bfloat16*>(req.dev_dst), plan,
          bulk_staged_counters(), bulk_staged_hashes(), req.ctl_seq, ar_ctl,
          ar_deadline_cycles, bulk_scratch, collective_stream);
      if (launch != cudaSuccess) {
        finish_flight(coll.req, false,
                      std::string("bulk collective kernel launch failed: ") +
                          cudaGetErrorString(launch));
        return true;
      }
      bulk.plan = plan;
      bulk.launched = true;
      bulk.posted.assign(peer_ranks.size(), 0);
      coll.launched_at = Clock::now();
      if (req.ctl_seq <= 4)
        DGPP_LOG_INFO(
            "allreduce: rank {} seq {} bulk {} seg {}/{} launched",
            opt.my_rank, req.ctl_seq, bulk.phase == 0 ? "RS" : "AG",
            bulk.segment + 1, req.bulk_seg_count);
      worked = true;
    }

    // Posting pass: stripes as the kernel's staged counters release them,
    // ring-gated per (peer, lane). The sender's striping is round-robin
    // (stripe k on lane k%lanes) — the receiver's arrival mapping derives
    // from exactly this.
    // Pacing (BusOptions::bulk_inflight_per_lane): peers in rotated rank
    // order — rank + 1 first, so the ranks' aligned bursts land on
    // different receivers — and a bounded window per (peer, lane).
    uint64_t* const staged = bulk_staged_counters();
    const size_t n_peers = peer_ranks.size();
    const size_t rotate =
        n_peers == 0 ? 0 : peer_index((opt.my_rank + 1) % opt.world_size);
    const size_t inflight_cap =
        opt.bulk_inflight_per_lane > 0
            ? std::min(static_cast<size_t>(opt.bulk_inflight_per_lane),
                       static_cast<size_t>(opt.bulk_slots))
            : static_cast<size_t>(opt.bulk_slots);
    for (size_t pi = 0; pi < n_peers; ++pi) {
      const size_t p = (rotate + pi) % n_peers;
      const uint32_t n_out =
          bulk.phase == 0 ? bulk.plan.out_count[p] : bulk.plan.my_count;
      while (bulk.posted[p] < n_out &&
             acquire_u64(&staged[p]) > bulk.posted[p]) {
        const uint32_t k = bulk.posted[p];
        LaneState& lane = peers[p][k % peers[p].size()];
        const uint32_t slot = lane.cursor[1];
        if (lane.send[1][slot].in_flight) break;  // ring full; retry
        size_t live = 0;
        for (const SendSlot& ss_live : lane.send[1]) live += ss_live.in_flight ? 1 : 0;
        if (live >= inflight_cap) break;  // window full; retry
        const Clock::time_point now = Clock::now();
        if (opt.bulk_pace_gbps > 0 && now < lane.bulk_next_post) break;  // paced
        const uint32_t sub_base =
            bulk.phase == 0 ? bulk.plan.out_base[p] : bulk.plan.my_base;
        const uint32_t gs = bulk.plan.seg_first + sub_base + k;
        const uint64_t left =
            req.elems * 2 - static_cast<uint64_t>(gs) * opt.bulk_slot_bytes;
        const uint32_t len = static_cast<uint32_t>(
            left < opt.bulk_slot_bytes ? left : opt.bulk_slot_bytes);
        RcLane& rc = *lane.lane;
        SendSlot& ss = lane.send[1][slot];
        const uint32_t seq = ss.gen + 1;
        std::string error;
        DGPP_LOG_DEBUG("bulk stripe: peer={} stripe={} len={}", peer_ranks[p],
                       k, len);
        // The door carries the stripe's fold — the bulk kernel's PLACEMENT
        // proof (StartSlot::hash). The staging kernel computed it where it
        // read the bytes and published it behind the staged counter
        // (2026-09-05; on the CPU it was ~10 us per 256 KB stripe, the
        // posting thread's ceiling once the fold got fast). The harness
        // send path stays ungated (its consumers are the flag kernels,
        // which never hash-gate).
        if (!rc.post_send_pair(BusPool::kBulk, slot, seq, len, &error,
                               bulk_arena_row(p) +
                                   static_cast<size_t>(k) *
                                       opt.bulk_slot_bytes,
                               lane.stage_lkey, req.ctl_seq,
                               bulk_staged_hashes()[p * kBusMaxBulkSegStripes +
                                                    k])) {
          fail_lane(lane, error);
          poison_collective(req);
          finish_flight(coll.req, false, "bulk post failed: " + error);
          return true;
        }
        ss.gen = seq;
        ss.in_flight = true;
        ss.owner = coll.req;
        ss.owner_stripe = static_cast<size_t>(p);
        ++lane.in_flight_count;
        lane.last_progress = now;  // a post is progress (idle-gap arming)
        if (opt.bulk_pace_gbps > 0) {
          // Space this lane's next post by the stripe's wire time at the
          // pace, measured from the later of now and the previous slot.
          const auto wire_time = std::chrono::nanoseconds(static_cast<int64_t>(
              static_cast<double>(len) * 8.0 / opt.bulk_pace_gbps));
          const Clock::time_point from =
              lane.bulk_next_post > now ? lane.bulk_next_post : now;
          lane.bulk_next_post = from + wire_time;
        }
        lane.cursor[1] = (slot + 1) % static_cast<uint32_t>(opt.bulk_slots);
        ++lane.stats.posts;
        lane.stats.bytes_sent += len;
        ++req.outstanding;
        ++bulk_pairs_posted;
        ++bulk.posted[p];
        worked = true;
      }
    }

    // Completion: the kernel's stamp ends this segment; advance the
    // machine. Clear-before-complete on every terminal path. Two drains
    // gate the advance, both because DONE DOES not IMPLY POSTED:
    //   * this segment's posting must be fully out — a zero-arrival
    //     kernel (empty receive window) stamps done instantly while its
    //     own stripes are still being posted, and the next launch RESETS
    //     staged_counters; advancing first would strand the unposted
    //     stripes with staged < posted forever (measured: tx 31/31 with
    //     5 of 16 AG stripes never posted — the receiver waits for
    //     doorbells that no longer exist);
    //   * every posted stripe's doorbell CQE (the arena-reuse fence) —
    //     the next segment's kernel stages into the same rows, and the
    //     kernel must not overwrite bytes the NIC has not read yet.
    const uint64_t done = acquire_u64(&ar_ctl->done_seq);
    if (done == req.ctl_seq) {
      for (size_t p = 0; p < peer_ranks.size(); ++p) {
        const uint32_t n_out = bulk.phase == 0 ? bulk.plan.out_count[p]
                                               : bulk.plan.my_count;
        if (bulk.posted[p] < n_out) return worked;  // posting not drained
      }
      if (bulk_doorbell_ce_seen != bulk_pairs_posted) return worked;
      const bool ok = acquire_u32(&ar_ctl->status) == 0;
      {
        std::lock_guard<std::mutex> lock(stats_mu);
        ++stats_store.bulk_segments;
        stats_store.bulk_gate_redos += acquire_u32(&ar_ctl->dbg_gate_spins);
      }
      if (!ok) {
        finish_flight(coll.req, false,
                      "bulk collective consumer exited on deadline or "
                      "poison");
        return true;
      }
      ++bulk.segment;
      if (bulk.segment >= req.bulk_seg_count) {
        if (bulk.phase == 0) {
          bulk.phase = 1;
          bulk.segment = 0;
        } else {
          finish_flight(coll.req, true, {});
          return true;
        }
      }
      bulk.launched = false;
      worked = true;
    }
    return worked;
  }

  // ---- graph walk (§6.2, engine thread) ----------------------------------
  // Reacts to the generations a replayed graph produces. Stream order
  // serializes the recorded kernels, so at most one generation is un-done
  // at a time: a single-flight state machine — post each peer's pair when
  // the kernel's staging lands, advance the walk on its done stamp. The
  // window atomics are the only cross-thread state (arm publishes,
  // release; the walk adopts, acquire — the per-gen cells and the node
  // metadata become visible through that chain).

  // The mixed-era eager gate (every caller holds coll_mu): a recording
  // session closes it for the capture's duration, an armed window until
  // its finish, a failed era for good. Returns the rejection text or
  // nullptr when eager collectives are open. The v1 rejection on
  // graph.recorded is gone on purpose: between windows the eager machine
  // runs again (prefill folds, the pick) — the unified generation
  // counter is what makes that safe.
  const char* eager_gate_closed() const {
    if (graph.recording)
      return "a graph session is recording (graph_record_end reopens)";
    if (graph.failed.load(std::memory_order_relaxed))
      return "the graph era failed (this bus must be restarted)";
    if (graph.live_windows > 0)
      return "a graph replay window is armed (graph_replay_finish reopens)";
    return nullptr;
  }

  // First-writer-wins era failure. The error string rides under coll_mu
  // (the engine already takes it at collective pickup; failure paths are
  // rare, never hot).
  void graph_fail(const std::string& why) {
    bool expected = false;
    if (graph.failed.compare_exchange_strong(expected, true)) {
      {
        std::lock_guard<std::mutex> lock(coll_mu);
        graph.error = why;
      }
      DGPP_LOG_ERROR("bus graph era failed: {}", why);
    }
  }

  // Stop-path drain of a live window: the engine is joined (its walk
  // state is safe to read), so stamp every un-walked generation's done
  // cell — the replayed kernels exit promptly when the caller syncs its
  // stream (they run there, not on the collective stream) and stamp
  // their own failure statuses. A completed window poisons nothing.
  void poison_stream_collectives() {
    // Engine joined: its walk state is ours. Every issued, un-walked
    // generation's cell gets its done stamp so the stream's kernels exit.
    bool poisoned = false;
    if (strm.active) {
      __atomic_store_n(&strm.cells[strm.cur.cell].done_seq, strm.cur.gen, __ATOMIC_RELEASE);
      poisoned = true;
    }
    {
      std::lock_guard<std::mutex> lock(coll_mu);
      for (const StreamState::Gen& g : strm.fifo) {
        __atomic_store_n(&strm.cells[g.cell].done_seq, g.gen, __ATOMIC_RELEASE);
        poisoned = true;
      }
      strm.fifo.clear();
    }
    strm.walked.store(strm.issued.load(std::memory_order_acquire), std::memory_order_release);
    strm.active = false;
    if (poisoned) graph_fail("bus stopped with stream collectives in flight");
  }
  void poison_live_graph_window() {
    if (!graph.recorded.load(std::memory_order_relaxed)) return;
    const uint64_t count =
        graph.window_count.load(std::memory_order_relaxed);
    if (count == 0) return;
    // Every armed window from the adopted one on: the adopted window's
    // un-walked remainder, then the windows armed behind it (their ring
    // entries are ours — forward thread, engine joined). Window-relative
    // node index: gen s of a window starting at `first` is node
    // (s - first) — the v1 (gen-1)%gens arithmetic held only when every
    // window started at gen 1, which prefill-before-era collectives
    // broke.
    bool poisoned = false;
    const uint64_t from_window = std::max<uint64_t>(graph.adopted_count, 1);
    for (uint64_t w = from_window; w <= count; ++w) {
      const GraphState::Window& win = graph.windows[(w - 1) % kBusWindowRing];
      const int variant = win.variant;
      if (variant < 0 || variant >= kBusMaxGraphVariants) continue;
      const uint64_t gens = win.gens;
      if (gens == 0) continue;
      BusAllReduceCtl* const cells = graph_variant_cells(variant);
      const uint64_t first = win.first;
      const uint64_t last = first + gens - 1;
      uint64_t s0 = first;
      if (w == graph.adopted_count) {
        if (graph.walk_seq > last) continue;  // window already walked
        s0 = std::max(graph.walk_seq, first);
      }
      for (uint64_t s = s0; s <= last; ++s) {
        BusAllReduceCtl* cell = &cells[(s - first) % gens];
        __atomic_store_n(&cell->done_seq, s, __ATOMIC_RELEASE);
      }
      poisoned = true;
    }
    if (poisoned) graph_fail("bus stopped with a graph window in flight");
  }

  // The generation's kernel timeline, from its cell's %globaltimer stamps:
  //   copy       entry -> staging rows written (phase 1)
  //   handshake  staging written -> first peer payload gated: engine notice
  //              + post + wire + the peer's own readiness (the fastest
  //              peer's arrival)
  //   skew       first -> last peer gated: the slowest peer's lag
  //   fold       last peer gated -> fold done
  void accumulate_timeline(GraphState::Timeline& t, const BusAllReduceCtl& c) {
    if (c.gt_start == 0 || c.gt_done < c.gt_start) return;
    if (!gen_dump_checked) {
      gen_dump_checked = true;
      if (const char* d = std::getenv("DGPP_BUS_TIMELINE_DUMP"); d && *d) {
        const std::string path = std::string(d) + "/bus_gens_rank" + std::to_string(opt.my_rank) + ".tsv";
        gen_dump_file = std::fopen(path.c_str(), "a");
        if (gen_dump_file) {
          std::fprintf(gen_dump_file, "# rank %d peers", opt.my_rank);
          for (int r : peer_ranks) std::fprintf(gen_dump_file, " %d", r);
          std::fprintf(gen_dump_file, "\n# gen start stage first last done landed[p]... claim[p]...\n");
        }
      }
    }
    if (gen_dump_file) {
      GenRecord g{c.gen_seq, c.gt_start, c.gt_stage, c.gt_first, c.gt_last, c.gt_done, {}, {}};
      for (size_t p = 0; p < kBusMaxPeers; ++p) { g.landed[p] = c.gt_landed[p]; g.claim[p] = c.gt_claim[p]; }
      gen_dump.push_back(g);
    }
    auto us = [](uint64_t a, uint64_t b) {
      return b > a ? static_cast<double>(b - a) / 1000.0 : 0.0;
    };
    ++t.n;
    if (t.n == 1) t.first_gt_start = c.gt_start;
    if (t.prev_gt_done != 0) t.compute_us += us(t.prev_gt_done, c.gt_start);
    t.prev_gt_done = c.gt_done;
    const double copy = us(c.gt_start, c.gt_stage);
    const double hs = us(c.gt_stage, c.gt_first);
    const double skew = us(c.gt_first, c.gt_last);
    const double fold = us(c.gt_last, c.gt_done);
    const double total = us(c.gt_start, c.gt_done);
    t.copy_us += copy;
    t.handshake_us += hs;
    t.skew_us += skew;
    t.fold_us += fold;
    t.total_us += total;
    t.max_handshake_us = std::max(t.max_handshake_us, hs);
    if (t.n == 1 || hs < t.min_handshake_us) t.min_handshake_us = hs;
    t.max_skew_us = std::max(t.max_skew_us, skew);
    if (total > t.max_total_us) {
      t.max_total_us = total;
      t.max_total_gen = c.gen_seq;
    }
    t.gate_waits += acquire_u32(&c.dbg_gate_waits);
    const double wait = hs + skew;
    const int b = wait < 20 ? 0 : wait < 50 ? 1 : wait < 100 ? 2
                : wait < 200 ? 3 : wait < 500 ? 4 : 5;
    ++t.wait_hist[b];
    if ((c.gen_seq & 1) == 0) { t.wait_even_us += wait; ++t.n_even; }
    else { t.wait_odd_us += wait; ++t.n_odd; }
    size_t last = 0;
    for (size_t p = 0; p < peer_ranks.size(); ++p) {
      t.peer_lag_us[p] += us(c.gt_stage, c.gt_claim[p]);
      if (c.gt_landed[p] != 0) {
        const double landed = us(c.gt_stage, c.gt_landed[p]);
        if (t.peer_land_n[p] == 0 || landed < t.peer_land_min_us[p]) t.peer_land_min_us[p] = landed;
        t.peer_land_us[p] += landed;
        ++t.peer_land_n[p];
      }
      if (c.gt_claim_first_neg[p] != 0) t.peer_first_us[p] += us(c.gt_stage, ~c.gt_claim_first_neg[p]);
      if (c.gt_claim[p] > c.gt_claim[last]) last = p;
    }
    ++t.peer_last[last];
    // The claims in claim order and the gaps between them.
    uint64_t claims[kBusMaxPeers];
    const size_t np = std::min<size_t>(peer_ranks.size(), kBusMaxPeers);
    for (size_t p = 0; p < np; ++p) claims[p] = c.gt_claim[p];
    std::sort(claims, claims + np);
    for (size_t p = 1; p < np; ++p) {
      const double gap = us(claims[p - 1], claims[p]);
      t.claim_gap_us[p] += gap;
      const int gb = gap < 3 ? 0 : gap < 5 ? 1 : gap < 7 ? 2 : gap < 10 ? 3 : gap < 15 ? 4 : gap < 25 ? 5 : 6;
      ++t.claim_gap_hist[p][gb];
    }
  }

  void log_window_timeline() {
    GraphState::Timeline& t = graph.tl;
    if (t.n == 0) return;
    // The per-window lines are DEBUG since the throughput line
    // (2026-09-06; fabric_xrank.py and serve_pace.py --waves read them
    // under DGPP_LOG_LEVEL=debug). DGPP_BUS_TIMELINE=1 writes them at INFO
    // without the rest of the debug output: the per-tick lines perturb
    // the very skew the timeline measures (a debug-level run showed 0.5
    // ms stalls at two fixed positions per step that an info-level trace
    // does not have).
    static const bool forced = [] {
      const char* e = std::getenv("DGPP_BUS_TIMELINE");
      return e != nullptr && e[0] == '1';
    }();
    if (!forced && dgpp::current_log_level() > dgpp::LogLevel::Debug) {
      t = GraphState::Timeline{};
      return;
    }
    const dgpp::LogLevel lvl = forced ? dgpp::LogLevel::Info : dgpp::LogLevel::Debug;
    const double n = static_cast<double>(t.n);
    ::dgpp::logf(lvl,
        "graph window timeline: rank {} variant {} gens {} avg us: total {:.1f} = copy "
        "{:.1f} + handshake {:.1f} + skew {:.1f} + fold {:.1f}; engine post "
        "{:.1f}; max handshake {:.1f} skew {:.1f} total {:.1f} "
        "(gen {}); gate waits {}; arm->gen0 {:.1f} us; passes/gen {:.0f}; "
        "compute between gens {:.1f} us; min handshake {:.1f} us",
        opt.my_rank, graph.adopted_variant, t.n, t.total_us / n,
        t.copy_us / n, t.handshake_us / n, t.skew_us / n, t.fold_us / n,
        t.post_us / n,
        t.max_handshake_us,
        t.max_skew_us, t.max_total_us, t.max_total_gen, t.gate_waits,
        t.first_gt_start > graph.armed_gt
            ? static_cast<double>(t.first_gt_start - graph.armed_gt) / 1000.0
            : 0.0,
        static_cast<double>(t.passes) / n,
        t.n > 1 ? t.compute_us / static_cast<double>(t.n - 1) : 0.0, t.min_handshake_us);
    ::dgpp::logf(lvl,
        "graph window wait: rank {} hist(<20 <50 <100 <200 <500 >=500 us) {} "
        "{} {} {} {} {}; even gens {} avg {:.1f} us, odd gens {} avg {:.1f} us",
        opt.my_rank, t.wait_hist[0], t.wait_hist[1], t.wait_hist[2],
        t.wait_hist[3], t.wait_hist[4], t.wait_hist[5], t.n_even,
        t.n_even ? t.wait_even_us / t.n_even : 0.0, t.n_odd,
        t.n_odd ? t.wait_odd_us / t.n_odd : 0.0);
    std::string peers_txt;
    for (size_t p = 0; p < peer_ranks.size(); ++p)
      peers_txt += " rank" + std::to_string(peer_ranks[p]) + " lag " +
                   std::to_string(static_cast<int>(t.peer_lag_us[p] / n)) + "us" +
                   (t.peer_land_n[p] != 0
                        ? " (landed " + std::to_string(static_cast<int>(t.peer_land_us[p] / static_cast<double>(t.peer_land_n[p]))) +
                              "us min " + std::to_string(static_cast<int>(t.peer_land_min_us[p])) +
                              "us, first proved " + std::to_string(static_cast<int>(t.peer_first_us[p] / static_cast<double>(t.peer_land_n[p]))) + "us)"
                        : std::string()) +
                   " last " + std::to_string(t.peer_last[p]) + "x;";
    ::dgpp::logf(lvl, "graph window peers: rank {} (staging written -> peer "
                  "payload gated, avg; times arrived last):{}",
                  opt.my_rank, peers_txt);
    std::string gaps_txt;
    for (size_t p = 1; p < std::min<size_t>(peer_ranks.size(), kBusMaxPeers); ++p) {
      gaps_txt += " claim " + std::to_string(p) + "->" + std::to_string(p + 1) + " avg " +
                  std::format("{:.1f}", t.claim_gap_us[p] / n) + " us hist";
      for (int i = 0; i < 7; ++i) gaps_txt += " " + std::to_string(t.claim_gap_hist[p][i]);
      gaps_txt += ";";
    }
    ::dgpp::logf(lvl, "graph window claims: rank {} (gaps between consecutive claims, hist "
                  "<3 <5 <7 <10 <15 <25 >=25 us):{}", opt.my_rank, gaps_txt);
    ::dgpp::logf(lvl, "graph window engine loop: rank {} iterations {} hist(<2 <5 <10 <20 <50 >=50 us) {} {} {} {} {} {}; max {:.1f} us",
                  opt.my_rank, loop_n, loop_hist[0], loop_hist[1], loop_hist[2], loop_hist[3], loop_hist[4], loop_hist[5], loop_max_us);
    loop_n = 0;
    loop_max_us = 0;
    for (auto& h : loop_hist) h = 0;
    if (gen_dump_file && !gen_dump.empty()) {
      for (const GenRecord& g : gen_dump) {
        std::fprintf(gen_dump_file, "%llu %llu %llu %llu %llu %llu", static_cast<unsigned long long>(g.gen),
                     static_cast<unsigned long long>(g.start), static_cast<unsigned long long>(g.stage),
                     static_cast<unsigned long long>(g.first), static_cast<unsigned long long>(g.last),
                     static_cast<unsigned long long>(g.done));
        for (size_t p = 0; p < peer_ranks.size() && p < kBusMaxPeers; ++p)
          std::fprintf(gen_dump_file, " %llu", static_cast<unsigned long long>(g.landed[p]));
        for (size_t p = 0; p < peer_ranks.size() && p < kBusMaxPeers; ++p)
          std::fprintf(gen_dump_file, " %llu", static_cast<unsigned long long>(g.claim[p]));
        std::fputc('\n', gen_dump_file);
      }
      std::fflush(gen_dump_file);
      gen_dump.clear();
    }
    t = GraphState::Timeline{};
  }

  bool graph_pass() {
    const uint64_t count = graph.window_count.load(std::memory_order_acquire);
    if (count == 0) return false;  // recorded but never armed
    bool worked = false;
    // Windows are walked in arm order: the adopted one to its end, then
    // the next armed one (its ring entry was published under coll_mu
    // before the window_count release the acquire above pairs with).
    const bool adopted_done =
        graph.adopted_count == 0 || graph.walk_seq > graph.adopted_last;
    if (adopted_done && graph.adopted_count == count)
      return false;  // every armed window walked; awaiting the next arm
    if (adopted_done) {
      // The stream collectives issued before this arm come first in
      // generation order (stream_pass walks them; the gate admits no
      // stream issue while the window is armed).
      if (stream_outstanding()) return false;
      const GraphState::Window& w =
          graph.windows[graph.adopted_count % kBusWindowRing];
      const int variant = w.variant;
      if (variant < 0 || variant >= kBusMaxGraphVariants ||
          !graph.variants[variant].recorded ||
          graph.variants[variant].gens <= 0 ||
          w.gens != static_cast<uint64_t>(graph.variants[variant].gens)) {
        graph_fail("graph window selected an invalid variant");
        return true;
      }
      const uint64_t gens = w.gens;
      const uint64_t first = w.first;
      // A fresh window. Its first generation was reserved from the shared
      // collective counter at arm. Nothing will walk it if the era already
      // failed — and drained() keeps the engine alive for un-adopted
      // windows, so publish the walk past it instead of wedging the stop
      // path (a failure landing between arm and adoption would otherwise
      // hang quiesce's join forever: the cells were never consumed;
      // poison handles any launched kernels).
      if (graph.failed.load(std::memory_order_relaxed)) {
        graph.adopted_count += 1;
        graph.adopted_variant = variant;
        graph.adopted_first = first;
        graph.adopted_last = first + gens - 1;
        graph.walk_seq = first + gens;
        graph.walk_pub.store(graph.walk_seq, std::memory_order_release);
        return true;
      }
      // Continuity: the walk state points at the next generation to
      // walk, and the unified counter is monotonic with execution, so a
      // window's first can only be AT or BEYOND it (eager collectives
      // between windows consume numbers — the first window starts
      // wherever prefill left the counter). Behind it is a protocol
      // break; the gap ahead is the eager machine's completed business.
      if (first < graph.walk_seq) {
        graph_fail("graph window armed out of walk order (protocol)");
        return true;
      }
      graph.adopted_count += 1;
      graph.adopted_variant = variant;
      graph.adopted_first = first;
      graph.adopted_last = first + gens - 1;
      graph.armed_gt = w.armed_gt;
      graph.walk_seq = first;
      graph.flight = {};
      graph.stall_dumps = 0;  // microscope rate control, per window
      // Quiescence snapshot (the RNR-freeze tripwire, observability form):
      // at adopt every kernel of the previous window exited (walk complete
      // implies every claim acked), so every latency door cell should read
      // consumed (door.seq == ack.seq). An unconsumed doorbell here is the
      // precursor of the ring-recycle freeze — the receive queue drains
      // over the next wrap and the wrap's last sender RNR-retries forever
      // (a once-in-~30k-gen stall, unreproduced but not forgiven). not an
      // era failure: the peer's engine may already be posting this window
      // (its adopt precedes its posts too, but its doorbell can land
      // between this pass's door and ack reads — door=ack+1 mid-flight is
      // a legal observation); the snapshot goes to the log, and the stall
      // microscope plus the lane watchdog own the verdict if the real
      // freeze follows.
      for (size_t p = 0; p < peer_ranks.size(); ++p) {
        const BusRecvView view = recv_view_of(peers[p][0]);
        for (int s = 0; s < opt.lat_slots; ++s) {
          const uint32_t door =
              acquire_u32(&view.doorbell_lat[s].seq);
          const uint32_t ack = acquire_u32(&view.ack_lat[s].seq);
          if (door != ack) {
            std::string lanes;
            for (int s2 = 0; s2 < opt.lat_slots; ++s2)
              lanes += std::to_string(
                           acquire_u32(&view.doorbell_lat[s2].seq)) +
                       "/" +
                       std::to_string(acquire_u32(&view.ack_lat[s2].seq)) +
                       ",";
            DGPP_LOG_DEBUG(
                "graph adopt quiescence note: peer {} lane 0 slot {} "
                "door={} ack={} (mid-flight ok; freeze precursor if it "
                "persists) cells {}",
                peer_ranks[p], s, door, ack, lanes);
            break;  // one note per peer is enough state
          }
        }
      }
      // The window's slot-credit carrier (see GraphState): gives the
      // posts request-shaped ownership without a registry entry.
      auto carrier = std::make_shared<BusRequest>();
      carrier->cls = BusMessageClass::kLatency;
      carrier->peer_rank = -1;
      carrier->is_collective = true;
      carrier->stripe_hashes.assign(peer_ranks.size(), 0);
      graph.carrier = std::move(carrier);
      worked = true;
    }
    const int variant = graph.adopted_variant;
    GraphState::Variant& shape = graph.variants[variant];
    const uint64_t gens = static_cast<uint64_t>(shape.gens);
    BusAllReduceCtl* const cells = graph_variant_cells(variant);

    if (graph.failed.load(std::memory_order_relaxed)) {
      // Drain-fail: poison every un-walked generation so the replay's
      // remaining kernels exit promptly (each treats the done stamp as
      // an exit and stamps its own failure status). finish reports.
      const uint64_t last = graph.adopted_last;
      for (uint64_t s = graph.walk_seq; s <= last; ++s) {
        BusAllReduceCtl* cell =
            &cells[(s - graph.adopted_first) % gens];
        __atomic_store_n(&cell->done_seq, s, __ATOMIC_RELEASE);
      }
      graph.walk_seq = last + 1;
      graph.walk_pub.store(graph.walk_seq, std::memory_order_release);
      graph.flight = {};
      return true;
    }

    // The current generation: node (gen - adopted_first) — window-
    // relative, not (gen-1)%gens (that held only for first==1 windows;
    // prefill before the era moves first). At most one kernel is
    // un-done at a time (the graph serializes them), so the walk's view
    // of the cell is exclusive.
    const uint64_t gen = graph.walk_seq;
    BusAllReduceCtl* const cell =
        &cells[(gen - graph.adopted_first) % gens];
    const uint32_t elems = shape.meta[(gen - graph.adopted_first) % gens].elems;
    ++graph.tl.passes;
    bool completed = false;
    const bool flew =
        gen_flight_pass(cell, gen, elems, cells, shape.gens, graph.carrier,
                        &graph.tl, "graph walk", &completed);
    if (completed) {
      ++graph.walk_seq;
      graph.walk_pub.store(graph.walk_seq, std::memory_order_release);
      if (graph.walk_seq > graph.adopted_last) log_window_timeline();
      return true;
    }
    return worked || flew;
  }
  // The per-generation flight, shared by the replay walk and the stream
  // collectives (D9): activate before the done check, post each peer's
  // pair from the generation's staging row once the kernel's ready bits
  // land, complete on the done stamp with the posting mask full; the
  // stall microscope (over `cells[0..ncells)`) and the flight watchdog.
  // `carrier` owns the posts' slots; `tl` takes the generation's timeline.
  // Returns worked; *completed once the generation is walked (the caller
  // advances walk_seq and publishes).
  bool gen_flight_pass(BusAllReduceCtl* cell, uint64_t gen, uint32_t elems,
                       const BusAllReduceCtl* cells, int ncells,
                       const std::shared_ptr<BusRequest>& carrier,
                       GraphState::Timeline* tl, const char* what,
                       bool* completed) {
    bool worked = false;
    const uint32_t seq = static_cast<uint32_t>(gen);
    const uint64_t ready = acquire_u64(&cell->ready_bits);
    uint64_t done = acquire_u64(&cell->done_seq);
    // The write transport reports per block: posting when every block has
    // staged, completion when every block is done (the poison path still
    // goes through done_seq).
    bool write_ready = false;
    if (write_transport()) {
      bool write_done = false;
      write_aggregate(cell, seq, elems, &write_ready, &write_done);
      if (write_done) done = seq;
    }

    // The flight activates BEFORE the done check: a fast peer can stamp
    // this kernel's done within one engine pass of its ready (the fold
    // waits on the PEER's doorbell, not this side's own post — done does
    // not imply posted), and the advance below requires the posting mask
    // complete. A generation whose pair is still ring-deferred parks
    // here until the drain; advancing first would strand the peer's
    // kernel on a doorbell that never comes (measured: posts 131/132
    // with the peer's last generation spinning 5s on the missing one).
    if (!graph.flight.active) {
      graph.flight.gen = gen;
      graph.flight.posted_bits = 0;
      graph.flight.started_at = Clock::now();
      graph.flight.progressed_at = Clock::now();
      graph.flight.active = true;
      worked = true;
    }

    if (done == seq) {
      // The kernel stamped (or a poison matched). A nonzero status is an
      // era failure regardless of posting; the drain-fail branch poisons
      // the window's remainder on the next iteration.
      if (acquire_u32(&cell->status) != 0) {
        graph_fail("graph generation " + std::to_string(seq) +
                   " exited on deadline or poison");
        return true;
      }
      if (graph.flight.posted_bits == all_peers_mask()) {
        accumulate_timeline(*tl, *cell);
        record_latency(BusMessageClass::kLatency,
                       elapsed_us(graph.flight.started_at));
        graph.flight = {};
        completion_epoch.fetch_add(1, std::memory_order_release);
        *completed = true;
        return true;
      }
      // else: posting not drained — fall through, post, advance next pass.
    }

    // TEMP bring-up microscope: a flight that has not progressed in 500ms
    // dumps the walk state once per 500ms — what THIS thread's acquire
    // loads return, versus the test thread's post-mortem dump. Lane 0 is
    // the graph posting lane; its door/ack/in-flight cells adjudicate a
    // lost doorbell (RNR retry shows as seq stuck ahead of ack).
    if (Clock::now() - graph.flight.progressed_at >
        std::chrono::milliseconds(500 + 500 * graph.stall_dumps)) {
      ++graph.stall_dumps;
      std::string cells_txt;
      for (int g = 0; g < ncells; ++g) {
        const BusAllReduceCtl& c = cells[g];
        cells_txt += " [" + std::to_string(g) + "]g=" +
                 std::to_string(acquire_u64(&c.gen_seq)) + ",r=" +
                 std::to_string(acquire_u64(&c.ready_bits)) + ",d=" +
                 std::to_string(acquire_u64(&c.done_seq)) + ",s=" +
                 std::to_string(acquire_u32(&c.status)) + ",e=" +
                 // kernel ENTRY seen (gt_start stamped): 0 = the node never
                 // started; distinguishes a scheduling stall from a wedge
                 std::to_string(acquire_u64(&c.gt_start) != 0 ? 1 : 0);
      }
      std::string lanes;
      for (size_t p = 0; p < peer_ranks.size(); ++p) {
        const BusRecvView view = recv_view_of(peers[p][0]);
        lanes += " p" + std::to_string(peer_ranks[p]) + "l0:d/a/s!";
        for (int s = 0; s < opt.lat_slots; ++s)
          lanes += std::to_string(acquire_u32(
                       &view.doorbell_lat[s].seq)) +
                   "/" + std::to_string(acquire_u32(&view.ack_lat[s].seq)) +
                   (peers[p][0].send[0][s].in_flight ? "!" : ".") + ",";
      }
      DGPP_LOG_INFO(
          "{} STALLED gen={} posted={:#x} cell[{}]@{} windows={} "
          "adopted={} walk={}{}{}",
          what, gen, graph.flight.posted_bits, static_cast<int>(cell - cells),
          static_cast<const void*>(cell),
          graph.window_count.load(std::memory_order_relaxed), graph.adopted_count,
          graph.walk_seq, cells_txt, lanes);
    }

    // Posting: the kernel staged its peer rows (ready bits landed); post
    // each peer's payload+doorbell pair from the generation's staging
    // row. Deterministic positions — lane 0, the cursor's ring position
    // (the (ctl_seq-1)%lat_slots claim; the cursor IS that ordinal because
    // graph reservations and eager pickups share one execution-ordered
    // generation counter, so shape switches and between-window eager work
    // cannot make the cursor and claim diverge).
    if (graph.flight.posted_bits != all_peers_mask() && write_transport() && write_ready) {
      // The one-sided transport: the kernel's chunked row (the generation's
      // staging row, every chunk its own proof) RDMA-WRITTEN into each
      // peer's write ring at row (gen-1) % kBusWriteRing, lane 0. No slot,
      // no doorbell, no credit — the ring's reuse is the protocol's
      // (bus_types.hpp). Signaled every 16th post per lane so the SQ
      // retires and errors surface.
      const auto post_t0 = Clock::now();
      const uint32_t row = static_cast<uint32_t>((gen - 1) % kBusWriteRing);
      const uint8_t* src_row = stage_buf(0, static_cast<int>((gen - 1) % Impl::kStageRing));
      const uint32_t chunks = bus_write_chunks(elems);
      const int lanes = static_cast<int>(lane_count());
      for (size_t p = 0; p < peer_ranks.size(); ++p) {
        if ((graph.flight.posted_bits >> p) & 1) continue;
        // The row's chunks in contiguous runs, one per lane (bus_types.hpp,
        // bus_write_lane_run): each run through its lane's queue pair into
        // the peer's lane slab ring at the chunks' own offsets.
        for (int l = 0; l < lanes; ++l) {
          uint32_t run_begin = 0, run_end = 0;
          bus_write_lane_run(chunks, lanes, l, &run_begin, &run_end);
          if (run_end <= run_begin) continue;
          LaneState& lane = peers[p][static_cast<size_t>(l)];
          const size_t off = static_cast<size_t>(run_begin) * kBusWriteChunkBytes;
          const uint32_t run_bytes = static_cast<uint32_t>(static_cast<size_t>(run_end - run_begin) * kBusWriteChunkBytes);
          const bool signaled = (lane.write_posts % 16) == 15;
          std::string error;
          if (!lane.lane->post_write(
                  BusPool::kLatency, src_row + off, lane.stage_lkey, run_bytes,
                  lane.lane->layout().write_row_remote_addr(lane.peer_endpoint.slab_base, row) + off,
                  lane.peer_endpoint.rkey, signaled, &error)) {
            fail_lane(lane, error);
            graph_fail("graph row write failed: " + error);
            return true;
          }
          ++lane.write_posts;
          lane.last_progress = Clock::now();
          ++lane.stats.posts;
          lane.stats.bytes_sent += run_bytes;
        }
        graph.flight.posted_bits |= 1ULL << p;
        graph.flight.progressed_at = Clock::now();
        worked = true;
      }
      tl->post_us += elapsed_us(post_t0);
    } else if (graph.flight.posted_bits != all_peers_mask() && ready != 0 && !write_transport()) {
      const auto post_t0 = Clock::now();
      // The row the kernel wrote — peer row 0 of ring slot
      // (g-1)%kStageRing, shared by every peer's post (the graph kernel
      // snapshots once; the eager machine's per-peer rows are untouched).
      // The door carries the row's fold: the graph kernel's PLACEMENT gate
      // (StartSlot::hash), computed during the GPU snapshot and ordered
      // before the ready_bits release acquired above. Reuse it for every
      // peer and any ring-deferred retry; the row stays live until done.
      const uint8_t* row =
          stage_buf(0, static_cast<int>((gen - 1) % Impl::kStageRing));
      const uint64_t row_hash = cell->stage_hash;
      for (size_t p = 0; p < peer_ranks.size(); ++p) {
        if ((graph.flight.posted_bits >> p) & 1) continue;
        if (!((ready >> p) & 1)) continue;  // row not staged (belt+braces)
        LaneState& lane = peers[p][0];  // deterministic lane 0
        const uint32_t slot = lane.cursor[0];
        SendSlot& ss = lane.send[0][slot];
        if (ss.in_flight) break;  // ring position busy; retry next pass
        const uint32_t pair_seq = ss.gen + 1;
        std::string error;
        if (!lane.lane->post_send_pair(
                BusPool::kLatency, slot, pair_seq,
                static_cast<uint32_t>(elems) * 2, &error, row,
                lane.stage_lkey, gen, row_hash)) {
          fail_lane(lane, error);
          graph_fail("graph post failed: " + error);
          return true;
        }
        ss.gen = pair_seq;
        ss.in_flight = true;
        ss.owner = carrier;
        ss.owner_stripe = p;
        ++lane.in_flight_count;
        lane.last_progress = Clock::now();  // a post is progress (idle-gap arming)
        lane.cursor[0] = (slot + 1) % static_cast<uint32_t>(opt.lat_slots);
        ++lane.stats.posts;
        lane.stats.bytes_sent += elems * 2;
        ++carrier->outstanding;
        graph.flight.posted_bits |= 1ULL << p;
        graph.flight.progressed_at = Clock::now();
        worked = true;
        DGPP_LOG_DEBUG("graph stripe: gen={} peer={} lane=0 slot={} seq={}",
                       seq, peer_ranks[p], slot, pair_seq);
      }
      tl->post_us += elapsed_us(post_t0);
    }

    // The flight watchdog: graph generations carry no registry entry, so
    // the request watchdog cannot see them — the flight keeps its own
    // budget (the same completion timeout; long intra-graph compute is a
    // caller configuration concern). The lane watchdog covers a dead
    // peer's unreturned credits independently.
    if (Clock::now() - graph.flight.progressed_at >
        std::chrono::milliseconds(opt.completion_timeout_ms)) {
      graph_fail("graph generation " + std::to_string(seq) +
                 " stalled (no engine progress for " +
                 std::to_string(opt.completion_timeout_ms) + " ms)");
      return true;
    }
    return worked;
  }

  // The stream collectives' walk (D9): between windows, the FIFO's
  // generations in order through the same flight; a failed era poisons
  // the FIFO so the caller's stream drains.
  bool stream_pass() {
    if (graph.adopted_count != 0 && graph.walk_seq <= graph.adopted_last)
      return false;  // an adopted window is walking (the gate kept the stream out)
    if (!strm.active) {
      if (strm.issued.load(std::memory_order_acquire) <=
          strm.walked.load(std::memory_order_relaxed))
        return false;
      std::lock_guard<std::mutex> lock(coll_mu);
      if (strm.fifo.empty()) return false;  // issued, the launch still in progress
      strm.cur = strm.fifo.front();
      strm.fifo.pop_front();
      strm.active = true;
      graph.flight = {};
      if (strm.cur.gen < graph.walk_seq)
        graph_fail("stream collective issued out of walk order (protocol)");
      else
        graph.walk_seq = strm.cur.gen;
      if (!strm.carrier) {
        auto carrier = std::make_shared<BusRequest>();
        carrier->cls = BusMessageClass::kLatency;
        carrier->peer_rank = -1;
        carrier->is_collective = true;
        carrier->stripe_hashes.assign(peer_ranks.size(), 0);
        strm.carrier = std::move(carrier);
      }
    }
    const uint64_t gen = strm.cur.gen;
    BusAllReduceCtl* const cell = &strm.cells[strm.cur.cell];
    if (graph.failed.load(std::memory_order_relaxed)) {
      // Drain-fail: poison this generation and every queued one so the
      // stream's kernels exit; everything issued counts as walked (settle
      // reports the era's failure).
      __atomic_store_n(&cell->done_seq, gen, __ATOMIC_RELEASE);
      {
        std::lock_guard<std::mutex> lock(coll_mu);
        for (const StreamState::Gen& g : strm.fifo)
          __atomic_store_n(&strm.cells[g.cell].done_seq, g.gen, __ATOMIC_RELEASE);
        strm.fifo.clear();
      }
      graph.walk_seq = std::max(graph.walk_seq, gen + 1);
      graph.walk_pub.store(graph.walk_seq, std::memory_order_release);
      strm.walked.store(strm.issued.load(std::memory_order_acquire),
                        std::memory_order_release);
      strm.active = false;
      graph.flight = {};
      return true;
    }
    ++strm.tl.passes;
    bool completed = false;
    const bool worked =
        gen_flight_pass(cell, gen, strm.cur.elems, strm.cells, kBusStreamRing,
                        strm.carrier, &strm.tl, "stream collective", &completed);
    if (completed) {
      log_stream_gen(*cell, gen, strm.cur.elems);
      graph.walk_seq = gen + 1;
      graph.walk_pub.store(graph.walk_seq, std::memory_order_release);
      strm.walked.store(gen, std::memory_order_release);
      strm.active = false;
      return true;
    }
    return worked;
  }
  // DGPP_BUS_TIMELINE=1: every 32nd stream generation's decomposition at
  // INFO (the eager fold's sampled line, in the stream form).
  void log_stream_gen(const BusAllReduceCtl& c, uint64_t gen, uint32_t elems) {
    static const bool forced = [] {
      const char* e = std::getenv("DGPP_BUS_TIMELINE");
      return e != nullptr && e[0] == '1';
    }();
    if (!forced || (gen % 32) != 0) return;
    auto us = [](uint64_t a, uint64_t b) {
      return b > a ? static_cast<double>(b - a) / 1000.0 : 0.0;
    };
    const uint64_t now_gt = static_cast<uint64_t>(
        static_cast<int64_t>(monotonic_ns()) + gt_offset_ns);
    DGPP_LOG_INFO(
        "stream fold: rank {} gen {} elems {} status={} total {:.1f}us = copy "
        "{:.1f} + handshake {:.1f} + skew {:.1f} + fold {:.1f}; done->walked "
        "{:.1f}; gate waits {}",
        opt.my_rank, gen, elems, acquire_u32(&c.status), us(c.gt_start, c.gt_done),
        us(c.gt_start, c.gt_stage), us(c.gt_stage, c.gt_first),
        us(c.gt_first, c.gt_last), us(c.gt_last, c.gt_done), us(c.gt_done, now_gt),
        acquire_u32(&c.dbg_gate_waits));
  }
  // The pass's stream timeline (settle's call, forward thread: the walk's
  // writes are ordered before its `walked` release, settle's acquire).
  void log_stream_timeline() {
    GraphState::Timeline& t = strm.tl;
    if (t.n == 0) return;
    static const bool forced = [] {
      const char* e = std::getenv("DGPP_BUS_TIMELINE");
      return e != nullptr && e[0] == '1';
    }();
    if (!forced && dgpp::current_log_level() > dgpp::LogLevel::Debug) {
      t = GraphState::Timeline{};
      return;
    }
    const dgpp::LogLevel lvl = forced ? dgpp::LogLevel::Info : dgpp::LogLevel::Debug;
    const double n = static_cast<double>(t.n);
    ::dgpp::logf(lvl,
        "stream fold timeline: rank {} gens {} avg us: total {:.1f} = copy "
        "{:.1f} + handshake {:.1f} + skew {:.1f} + fold {:.1f}; engine post "
        "{:.1f}; max handshake {:.1f} skew {:.1f} total {:.1f} (gen {}); "
        "passes/gen {:.0f}; compute between gens {:.1f} us",
        opt.my_rank, t.n, t.total_us / n, t.copy_us / n, t.handshake_us / n,
        t.skew_us / n, t.fold_us / n, t.post_us / n, t.max_handshake_us,
        t.max_skew_us, t.max_total_us, t.max_total_gen,
        static_cast<double>(t.passes) / n,
        t.n > 1 ? t.compute_us / static_cast<double>(t.n - 1) : 0.0);
    t = GraphState::Timeline{};
  }
  uint64_t all_peers_mask() const {
    const size_t peers = peer_ranks.size();
    return peers >= 64 ? ~0ULL : ((1ULL << peers) - 1);
  }

  bool collective_pass() {
    bool worked = false;

    if (!coll.req) {
      std::lock_guard<std::mutex> lock(coll_mu);
      if (!coll_q.empty()) {
        coll.req = coll_q.front();
        coll_q.pop_front();
        coll_active.store(true, std::memory_order_relaxed);
        coll_mode.store(true, std::memory_order_relaxed);  // engine-owned
        coll.launched_at = Clock::now();  // flight clock starts at pickup
        coll.picked_at = coll.launched_at;
      }
    }
    if (!coll.req) return worked;
    // TEMP bring-up microscope: an active flight that has not completed in
    // 500ms dumps its state once per 500ms — the rare lane-watchdog stall
    // in the TP loopback runs otherwise dies with no observables.
    if (coll.req &&
        (coll.claims.size() > 0 || coll.req->is_bulk) &&
        Clock::now() - coll.launched_at >
            std::chrono::milliseconds(500 + 500 * coll.stall_dumps)) {
      ++coll.stall_dumps;
      std::string lanes;
      for (size_t p = 0; p < peer_ranks.size(); ++p)
        for (size_t l = 0; l < peers[p].size(); ++l) {
          const LaneState& lane = peers[p][l];
          lanes += " p" + std::to_string(peer_ranks[p]) + "l" +
                   std::to_string(l) + ":[";
          for (int s = 0; s < opt.lat_slots; ++s) {
            const StartSlot* door =
                recv_view_of(lane).doorbell_lat;  // per-lane view
            // The door's ctl rides the stall dump: a stranded doorbell's
            // ctl discriminates a parked FUTURE collective (benign — the
            // generation gate's design) from THIS collective's claim
            // stuck in the placement-gate spin (seq/ctl vs ack). The
            // payload's first word rides with it — if a claimed door's
            // payload never hashes, the missing bytes are somewhere in
            // the ring (a payload/door split), and this finds them.
            const uint32_t* pay = reinterpret_cast<const uint32_t*>(
                reinterpret_cast<const uint8_t*>(
                    recv_view_of(lane).payload_lat) +
                static_cast<size_t>(s) * recv_view_of(lane).lat_slot_bytes);
            char cellbuf[64];
            std::snprintf(cellbuf, sizeof(cellbuf), "%uc%u/%u=%#010x",
                          door[s].seq, door[s].ctl,
                          recv_view_of(lane).ack_lat[s].seq, pay[0]);
            lanes += cellbuf;
            lanes += lane.send[0][s].in_flight ? "!" : ".";
          }
          lanes += " B[";
          for (int s = 0; s < opt.bulk_slots; ++s) {
            const BusRecvView view = recv_view_of(lane);
            lanes += std::to_string(view.doorbell_bulk[s].seq) + "/" +
                     std::to_string(view.ack_bulk[s].seq) +
                     (lane.send[1][s].in_flight ? "!" : ".");
          }
          lanes += "]";
        }
      // Live QP state for the active pool on every lane — a wedged
      // SQ freezes sq_psn below the posted generations while rq_psn keeps
      // counting arrivals; an ERR QP is a flushed one.
      std::string qps;
      for (size_t p = 0; p < peer_ranks.size(); ++p)
        for (size_t l = 0; l < peers[p].size(); ++l) {
          if (peers[p][l].failed) continue;
          qps += " p" + std::to_string(peer_ranks[p]) + "l" +
                 std::to_string(l) + "[" +
                 peers[p][l].lane->qp_state_dump(coll.req->is_bulk ? BusPool::kBulk : BusPool::kLatency) + "]";
        }
      DGPP_LOG_INFO(
          "hunt qp: rank {} seq {} pool={} {}", opt.my_rank, coll.req->ctl_seq,
          coll.req->is_bulk ? "bulk" : "latency", qps);
      if (!coll.req->is_bulk) {
        DGPP_LOG_INFO(
            "allreduce: rank {} seq {} STALLED {:.0f}ms posted={:#x} "
            "ctl(ready={:#x} done={} status={} gate(waits={} spins={}) "
            "first_claim(cell={} len={} seq={})) {}",
            opt.my_rank, coll.req->ctl_seq,
            std::chrono::duration<double, std::milli>(Clock::now() -
                                                       coll.launched_at)
                .count(),
            coll.posted_bits, acquire_u64(&ar_ctl->ready_bits),
            acquire_u64(&ar_ctl->done_seq), acquire_u32(&ar_ctl->status),
            acquire_u32(&ar_ctl->dbg_gate_waits),
            acquire_u32(&ar_ctl->dbg_gate_spins),
            ar_ctl->dbg_first_cell ? static_cast<int>(ar_ctl->dbg_first_cell) - 1 : -1,
            ar_ctl->dbg_first_len ? static_cast<int>(ar_ctl->dbg_first_len) : -1,
            ar_ctl->dbg_first_seq ? static_cast<int>(ar_ctl->dbg_first_seq) : -1,
            lanes);
      }
      if (coll.req->is_bulk) {
        std::string posted_counts;
        for (size_t p = 0; p < peer_ranks.size(); ++p) {
          // The first launch can be held behind the preceding flight's TX
          // fence. No posting vector or current staged counters exist yet.
          const uint64_t posted = bulk.launched ? bulk.posted.at(p) : 0;
          const uint64_t staged = bulk.launched ? acquire_u64(&bulk_staged_counters()[p]) : 0;
          const uint32_t expected = !bulk.launched ? 0
              : bulk.phase == 0 ? bulk.plan.out_count[p] : bulk.plan.my_count;
          posted_counts += " p" + std::to_string(peer_ranks[p]) +
                           ":posted=" + std::to_string(posted) +
                           "/staged=" + std::to_string(staged) +
                           "/expected=" + std::to_string(expected);
        }
        DGPP_LOG_INFO(
            "allreduce: rank {} seq {} bulk {} seg {}/{} STALLED {:.0f}ms "
            "launched={} tx_retired={}/{} ctl(done={} status={} gate_redos={}) {} {}",
            opt.my_rank, coll.req->ctl_seq,
            bulk.phase < 0 ? "waiting_for_tx" : bulk.phase == 0 ? "RS" : "AG",
            bulk.phase < 0 ? 0 : bulk.segment + 1, coll.req->bulk_seg_count,
            std::chrono::duration<double, std::milli>(Clock::now() - coll.launched_at).count(),
            bulk.launched, bulk_doorbell_ce_seen, bulk_pairs_posted,
            bulk.phase < 0 ? 0 : acquire_u64(&ar_ctl->done_seq),
            bulk.phase < 0 ? 0 : acquire_u32(&ar_ctl->status),
            bulk.phase < 0 ? 0 : acquire_u32(&ar_ctl->dbg_gate_spins), posted_counts, lanes);
      } else if (!coll.claims.empty()) {
        // The KERNEL's actual claimed cells (ctl-cell claim records),
        // not the engine's send-side bookkeeping: for each record, the
        // door's triple + hash, the payload's first words, and the
        // engine's own CPU-side fold of those words — the placement-
        // gate stall discriminator. A folded hash that differs from the
        // door's with recognizable foreign words is a payload/door
        // split; matching words with a wrong door hash is a sender
        // discrepancy; the fold matching the door means the gate has
        // already passed and the stall is elsewhere.
        for (int i = 0; i < 3; ++i) {
          const uint32_t rec = ar_ctl->dbg_cl_cell[i];
          if (rec == 0) continue;
          const int lat_slots = opt.lat_slots;
          const int views_per_peer = lane_count();
          const int cell = static_cast<int>(rec) - 1;
          const int slot = cell % lat_slots;
          const int view_idx = cell / lat_slots;
          const int peer_rank_i = peer_ranks[view_idx / views_per_peer];
          const BusRecvView view = recv_view_of(
              peers[static_cast<size_t>(view_idx / views_per_peer)]
                    [static_cast<size_t>(view_idx % views_per_peer)]);
          const StartSlot* door = view.doorbell_lat + slot;
          const uint32_t* pay = reinterpret_cast<const uint32_t*>(
              reinterpret_cast<const uint8_t*>(view.payload_lat) +
              static_cast<size_t>(slot) * view.lat_slot_bytes);
          const uint32_t words = door->len / 8;
          uint64_t folded = 0;
          for (uint32_t w = 0; w < words && w < 16; ++w)
            folded ^= (pay[w] + w + 1) * 0x9E3779B97F4A7C15ULL;
          DGPP_LOG_INFO(
              "kernel claim: rank {} seq {} rec{} peer {} lane {} slot {} "
              "door(seq={} len={} ctl={} hash={:#x}) pay(w0={:#x} w1={:#x} "
              "w2={:#x} w3={:#x} w4={:#x} w5={:#x} w6={:#x} w7={:#x}) "
              "folded16={:#x}",
              opt.my_rank, coll.req->ctl_seq, i, peer_rank_i,
              view_idx % views_per_peer, slot, door->seq, door->len,
              door->ctl, door->hash, pay[0], pay[1], pay[2], pay[3], pay[4],
              pay[5], pay[6], pay[7], folded);
        }
      }
    }
    // A held flight whose request is already done was reaped out from
    // under us (fail_lane completing the stripe owner, or the watchdog
    // expiry below) — the flight must not survive the request: clear and
    // poison, or every later collective is rejected "one outstanding"
    // forever (the engine is single-threaded; no race with ourselves).
    if (coll.req->done_flag.load(std::memory_order_acquire)) {
      poison_collective(*coll.req);
      coll_poisoned = true;
      coll = {};
      bulk = {};
      coll_active.store(false, std::memory_order_relaxed);
      return true;
    }
    BusRequest& req = *coll.req;

    // The prefill-class machine is self-contained (its own claims,
    // posting, and completion); the latency one-shot continues below.
    if (req.is_bulk) return bulk_collective_pass(req);

    if (coll.claims.empty()) {
      std::vector<BusStripe> claims;
      bool starved_log_shown = false;  // TEMP bring-up: claim-wait tracing
      for (size_t p = 0; p < peer_ranks.size(); ++p) {
        bool have = false;
        for (size_t l = 0; l < peers[p].size(); ++l) {
          LaneState& lane = peers[p][l];
          if (lane.failed) continue;
          const uint32_t slot = lane.cursor[0];
          if (lane.send[0][slot].in_flight) continue;  // ring position busy
          claims.push_back(
              BusStripe{static_cast<int>(l), BusPool::kLatency, slot, 0});
          have = true;
          break;
        }
        if (!have) {
          if (!starved_log_shown) {
            DGPP_LOG_INFO(
                "allreduce: rank {} claim waiting — no free latency slot for "
                "peer {} (credits outstanding)",
                opt.my_rank, peer_ranks[p]);
            starved_log_shown = true;
          }
          return worked;  // a ring is exhausted; retry next iteration
        }
      }

      if (ar_deadline_cycles == 0) {
        ar_deadline_cycles = bus_consumer_deadline_cycles(opt.consumer_deadline_s);
        int clock_khz = 0;
        cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
        DGPP_LOG_INFO("allreduce: device clock rate {} kHz (stamps conversion)",
                      clock_khz);
      }
      if (ar_deadline_cycles == 0) {
        finish_flight(coll.req, false,
                      "could not read device clock rate for the "
                      "collective deadline");
        return true;
      }

      req.ctl_seq = ctl_seq_counter.fetch_add(1, std::memory_order_relaxed) + 1;
      if (req.ctl_seq == 0)  // skip idle 0 (wrap)
        req.ctl_seq = ctl_seq_counter.fetch_add(1, std::memory_order_relaxed) + 1;
      req.stripe_hashes.assign(peer_ranks.size(), 0);
      // Reset before launch: the previous kernel's stamps are stale, and
      // the launch (this thread) is ordered after the reset by program
      // order.
      __atomic_store_n(&ar_ctl->ready_bits, 0, __ATOMIC_RELAXED);
      __atomic_store_n(&ar_ctl->done_seq, 0, __ATOMIC_RELAXED);
      __atomic_store_n(&ar_ctl->status, 0, __ATOMIC_RELAXED);
      ar_ctl->stamp_stage = 0;
      ar_ctl->stamp_first_claim = 0;
      ar_ctl->gt_start = ar_ctl->gt_stage = ar_ctl->gt_first =
          ar_ctl->gt_last = ar_ctl->gt_done = 0;
      for (uint64_t& g : ar_ctl->gt_claim) g = 0;
      for (uint64_t& g : ar_ctl->gt_landed) g = 0;
      for (uint64_t& g : ar_ctl->gt_claim_first_neg) g = 0;
      ar_ctl->stamp_reduce_done = 0;
      // TEMP hunt stamp reset (small-collective corruption).
      __atomic_store_n(&ar_ctl->dbg_first_cell, 0, __ATOMIC_RELAXED);
      __atomic_store_n(&ar_ctl->dbg_first_len, 0, __ATOMIC_RELAXED);
      __atomic_store_n(&ar_ctl->dbg_first_seq, 0, __ATOMIC_RELAXED);
      // PLACEMENT-gate telemetry reset (the 2026-09-01 fix's proof).
      __atomic_store_n(&ar_ctl->dbg_gate_waits, 0, __ATOMIC_RELAXED);
      __atomic_store_n(&ar_ctl->dbg_gate_spins, 0, __ATOMIC_RELAXED);
      for (int i = 0; i < 3; ++i) {
        ar_ctl->dbg_cl_cell[i] = 0;
        ar_ctl->dbg_cl_len[i] = 0;
        ar_ctl->dbg_cl_seq[i] = 0;
        ar_ctl->dbg_cl_hash[i] = 0;
      }

      BusAllReduceView view{};
      int vi = 0;
      for (size_t p = 0; p < peer_ranks.size(); ++p)
        for (const LaneState& lane : peers[p]) view.recv[vi++] = recv_view_of(lane);
      view.recv_views = vi;
      view.lanes_per_peer = static_cast<int>(lane_count());

      // Staging generation: the request's (a pre-staged GEMM handout) or
      // the engine's pick. Single-outstanding means any generation whose
      // previous user completed is safe — and completion implies the peer
      // folded the posted bytes, so the NIC read finished.
      int stage_gen = req.stage_gen;
      if (stage_gen < 0) {
        stage_gen = static_cast<int>(
            (req.ctl_seq - 1) % static_cast<uint32_t>(Impl::kStageRing));
      } else if (req.dev_src !=
                 self_buf(peer_ranks.size(), stage_gen)) {
        // Defense in depth: a pre-staged submit must source the exact
        // handout buffer. Anything else is a caller protocol break.
        finish_flight(coll.req, false,
                      "pre-staged collective source is not the held staging "
                      "buffer");
        return true;
      }
      for (size_t p = 0; p < peer_ranks.size(); ++p)
        view.send_payload[p] = reinterpret_cast<const uint16_t*>(
            stage_buf(p, stage_gen));
      view.send_peers = static_cast<int>(peer_ranks.size());

      const auto t_launch0 = Clock::now();
      const cudaError_t launch = launch_bus_allreduce(
          view, opt.my_rank, static_cast<const __nv_bfloat16*>(req.dev_src),
          static_cast<__nv_bfloat16*>(req.dev_dst),
          static_cast<uint32_t>(req.elems), req.ctl_seq, ar_ctl,
          ar_deadline_cycles, collective_stream);
      DGPP_LOG_DEBUG("allreduce: rank {} seq {} launch_call={:.1f}us",
                     opt.my_rank, req.ctl_seq, elapsed_us(t_launch0));
      if (launch != cudaSuccess) {
        finish_flight(coll.req, false,
                      std::string("allreduce kernel launch failed: ") +
                          cudaGetErrorString(launch));
        return true;
      }
      // TEMP hunt (burst wedge): the claim record — which slots this
      // rank's engine grabbed for this collective, and the busy slots it
      // skipped (the in_flight/credit chain is the wedge's suspect).
      {
        std::string c;
        for (size_t ci = 0; ci < claims.size(); ++ci)
          c += " p" + std::to_string(peer_ranks[ci]) + "l" +
               std::to_string(claims[ci].lane) + "s" +
               std::to_string(claims[ci].slot);
        for (size_t p = 0; p < peer_ranks.size(); ++p)
          for (size_t l = 0; l < peers[p].size(); ++l) {
            const LaneState& lane = peers[p][l];
            const uint32_t s = lane.cursor[0];
            if (lane.send[0][s].in_flight)
              c += " BUSY(p" + std::to_string(peer_ranks[p]) + "l" +
                   std::to_string(l) + "s" + std::to_string(s) +
                   " gen=" + std::to_string(lane.send[0][s].gen) + ")";
          }
        DGPP_LOG_DEBUG("hunt claim: rank {} seq {} slots:{}",
                       opt.my_rank, req.ctl_seq, c);
      }
      coll.claims = std::move(claims);
      coll.launched_at = Clock::now();
      coll.launch_call_us = elapsed_us(t_launch0);
      coll.launched_gt_ns = static_cast<uint64_t>(
          static_cast<int64_t>(monotonic_ns()) + gt_offset_ns);
      coll.stage_gen = stage_gen;
      // First-flight diagnostics at INFO (not DEBUG): the loopback TP
      // bring-up had a stall whose DEBUG logging changed the timing, so
      // the lifecycle must be observable in the failing configuration.
      if (req.ctl_seq <= 16)
        DGPP_LOG_INFO("allreduce: rank {} seq {} launched ({} peers)",
                      opt.my_rank, req.ctl_seq, view.send_peers);
      DGPP_LOG_DEBUG("allreduce: rank {} seq {} launched, {} peers",
                     opt.my_rank, req.ctl_seq, view.send_peers);
      worked = true;
    }

    // DONE DOES not IMPLY POSTED — the one-shot form. The
    // kernel stages, releases the ready bits, claims, folds and stamps
    // done; when every peer had already posted, all of that takes ~20 us
    // and can fit between this thread's two reads of the control cell.
    // Read ready AFTER done: a pass that observes the stamp then
    // necessarily observes the bits (release/acquire, kernel program
    // order) and posts in the same pass, and the completion below refuses
    // to finish a reduced flight before every peer's stripe is out. The
    // failure this closes (glm_tp_test's two-rank loopback, once in
    // fifteen runs): rank 0's seq 4 logged launched then done with no
    // ready between, its stripe never posted, rank 1's kernel waited on
    // it forever while rank 0's next generation parked behind rank 1's
    // generation gate.
    const uint64_t done = acquire_u64(&ar_ctl->done_seq);
    if (opt.debug_pass_delay_us > 0)
      std::this_thread::sleep_for(
          std::chrono::microseconds(opt.debug_pass_delay_us));

    // Posting pass: one stripe per staged peer.
    const uint64_t ready = acquire_u64(&ar_ctl->ready_bits);
    if (ready && req.ctl_seq != ar_last_ready_seq) {
      ar_last_ready_seq = req.ctl_seq;
      if (req.ctl_seq <= 16)
        DGPP_LOG_INFO("allreduce: rank {} seq {} ready after {:.1f}us",
                      opt.my_rank, req.ctl_seq, elapsed_us(coll.launched_at));
      DGPP_LOG_DEBUG("allreduce: rank {} seq {} ready_seen={:.1f}us",
                     opt.my_rank, req.ctl_seq, elapsed_us(coll.launched_at));
    }
    for (size_t p = 0; p < coll.claims.size(); ++p) {
      if (coll.posted_bits & (1ULL << p)) continue;
      if (!((ready >> p) & 1)) continue;
      LaneState& lane = peers[p][static_cast<size_t>(coll.claims[p].lane)];
      RcLane& rc = *lane.lane;
      const uint32_t slot = coll.claims[p].slot;
      SendSlot& ss = lane.send[0][slot];
      const uint32_t seq = ss.gen + 1;
      std::string error;
      // Payload from the staging generation (the slot argument remains the
      // doorbell's remote ring position — the unchanged cursor discipline).
      // The door carries the payload's fold: the kernels' PLACEMENT gate
      // (the doorbell's DMA can be visible before the payload's — see
      // StartSlot::hash). The engine reads the pinned row coherently: the
      // kernel's snapshot preceded its ready_bits release, acquired here.
      // TEMP hunt: the hashed row's own words — the fabric racer caught a
      // door hash that folds NEITHER the old nor the new way over the
      // received bytes; this line pins what the sender actually hashed.
      {
        const uint64_t* hw = reinterpret_cast<const uint64_t*>(
            stage_buf(p, coll.stage_gen));
        DGPP_LOG_DEBUG(
            "hunt hash: rank {} seq {} peer {} gen {} words({:#x},{:#x},"
            "{:#x},{:#x})",
            opt.my_rank, req.ctl_seq, peer_ranks[p], coll.stage_gen, hw[0],
            hw[1], hw[2], hw[3]);
      }
      if (!rc.post_send_pair(BusPool::kLatency, slot, seq,
                              static_cast<uint32_t>(req.elems * 2), &error,
                              stage_buf(p, coll.stage_gen),
                              lane.stage_lkey, req.ctl_seq,
                              bus_fold(stage_buf(p, coll.stage_gen), req.elems * 2))) {
        fail_lane(lane, error);
        poison_collective(req);
        coll_poisoned = true;
        finish_flight(coll.req, false, "allreduce post failed: " + error);
        return true;
      }
      ss.gen = seq;
      ss.in_flight = true;
      ss.owner = coll.req;
      ss.owner_stripe = p;  // peer index: stripe_hashes is peers-sized
      ++lane.in_flight_count;
      lane.last_progress = Clock::now();  // a post is progress (idle-gap arming)
      lane.cursor[0] = (slot + 1) % static_cast<uint32_t>(opt.lat_slots);
      ++lane.stats.posts;
      lane.stats.bytes_sent += req.elems * 2;
      ++req.outstanding;
      coll.posted_bits |= 1ULL << p;
      worked = true;
      DGPP_LOG_DEBUG("hunt post: rank {} seq {} peer={} lane={} slot={} pairseq={}",
                     opt.my_rank, req.ctl_seq,
                     peer_ranks[p], lane.stats.lane, slot, seq);
    }

    // Completion: the kernel's stamp (or a poison) ends the flight — once
    // every peer's stripe is posted (a poisoned or deadline exit finishes
    // regardless; its lanes may be dead).
    if (done == req.ctl_seq) {
      if (acquire_u32(&ar_ctl->status) == 0 &&
          coll.posted_bits != all_peers_mask())
        return worked;  // post first (the bits are visible); finish next pass
      if (req.ctl_seq <= 16)
        DGPP_LOG_INFO(
            "allreduce: rank {} seq {} done status={} after {:.1f}us",
            opt.my_rank, req.ctl_seq, acquire_u32(&ar_ctl->status),
            elapsed_us(coll.launched_at));
      // TEMP hunt dump (small-collective corruption hunt; DEBUG-only):
      // the payload region's first words at each claimed slot plus the
      // doorbell cell — if a doorbell WR ever landed in a payload buffer
      // (half-pair RQ shift), w0/w1 here read as the StartSlot template
      // {seq, len} instead of folded bf16 payload.
      for (size_t p = 0; p < coll.claims.size(); ++p) {
        const uint32_t s = coll.claims[p].slot;
        const BusRecvView view =
            recv_view_of(peers[p][static_cast<size_t>(coll.claims[p].lane)]);
        const uint32_t* pay = reinterpret_cast<const uint32_t*>(
            reinterpret_cast<const uint8_t*>(view.payload_lat) +
            static_cast<size_t>(s) * view.lat_slot_bytes);
        const uint32_t* door = reinterpret_cast<const uint32_t*>(
            reinterpret_cast<const uint8_t*>(view.doorbell_lat) +
            static_cast<size_t>(s) * sizeof(StartSlot));
        DGPP_LOG_DEBUG(
            "hunt: rank {} seq {} peer {} slot {} door(seq={},len={}) "
            "pay(w0={:#x},w1={:#x},w2={:#x},w3={:#x})",
            opt.my_rank, req.ctl_seq, peer_ranks[p], s, door[0], door[1],
            pay[0], pay[1], pay[2], pay[3]);
      }
      DGPP_LOG_DEBUG(
          "allreduce: rank {} seq {} stamped status={} notice={:.1f}us "
          "spans_cycles stage->claim={} claim->reduce={} "
          "first_claim cell={} len={} seq={}",
          opt.my_rank, req.ctl_seq, acquire_u32(&ar_ctl->status),
          elapsed_us(coll.launched_at),
          ar_ctl->stamp_first_claim > ar_ctl->stamp_stage
              ? ar_ctl->stamp_first_claim - ar_ctl->stamp_stage
              : 0,
          ar_ctl->stamp_reduce_done > ar_ctl->stamp_first_claim
              ? ar_ctl->stamp_reduce_done - ar_ctl->stamp_first_claim
              : 0,
          ar_ctl->dbg_first_cell == 0
              ? -1
              : static_cast<int>(ar_ctl->dbg_first_cell) - 1,
          ar_ctl->dbg_first_len == 0xFFFFFFFFu
              ? -1
              : static_cast<int>(ar_ctl->dbg_first_len),
          ar_ctl->dbg_first_seq == 0xFFFFFFFFu
              ? -1
              : static_cast<int>(ar_ctl->dbg_first_seq));
      const bool reduced = acquire_u32(&ar_ctl->status) == 0;
      // Small-collective microscope (the pick class, elems <= 64): the
      // first-claim triple at INFO, always on. The 2026-09-01 fabric race
      // (rank 0 folded garbage while all peers held the correct token)
      // died with no observables — every per-collective log was gated to
      // the first 16 sequences or DEBUG, whose volume changes the very
      // timing under hunt. One line per pick phase is the compromise: a
      // recurrence now carries its discriminating evidence (stale len vs
      // misaligned cell vs fresh-door/stale-payload) at production timing.
      // The eager collective's timeline: host side (queue
      // wait, the launch call) and GPU side (%globaltimer stamps: copy,
      // handshake, skew, fold) plus the two interfaces between them — launch
      // call -> kernel start and kernel done -> engine noticed — read
      // against CLOCK_REALTIME, which %globaltimer tracks on this driver.
      // Always for the pick class (elems <= 64: two lines per decode
      // step), and for ANY eager collective slower than 2 ms — the
      // decode step's p99 lived in exactly one of these interfaces.
      const double total_us = elapsed_us(req.submitted);
      // DGPP_BUS_TIMELINE=1 (the graph windows' switch) also prints the
      // prefill-class folds' decomposition at INFO, every 32nd of them —
      // the eager walk's boundary reductions (2026-09-14: the eager fold
      // kernel is the first cost of a short prompt's prefill).
      static const bool timeline_forced = [] {
        const char* e = std::getenv("DGPP_BUS_TIMELINE");
        return e != nullptr && e[0] == '1';
      }();
      const bool sampled_prefill = timeline_forced && req.elems > 64 && (req.ctl_seq % 32) == 0;
      if (req.elems <= 64 || total_us > 2000.0 || sampled_prefill) {
        auto us = [](uint64_t a, uint64_t b) {
          return b > a ? static_cast<double>(b - a) / 1000.0 : 0.0;
        };
        const uint64_t now_gt = static_cast<uint64_t>(
            static_cast<int64_t>(monotonic_ns()) + gt_offset_ns);
        const std::string line = std::format(
            "allreduce: rank {} seq {} elems {} done status={} total {:.1f}us"
            " = queue {:.1f} + claim {:.1f} + launch_call {:.1f} + "
            "launch->start {:.1f} + "
            "copy {:.1f} + handshake {:.1f} + skew {:.1f} + fold {:.1f} + "
            "done->notice {:.1f}; gate(waits={} spins={}); peer lags us:{}",
            opt.my_rank, req.ctl_seq, req.elems, acquire_u32(&ar_ctl->status),
            total_us, elapsed_us_between(req.submitted, coll.picked_at),
            elapsed_us_between(coll.picked_at, coll.launched_at) -
                coll.launch_call_us,
            coll.launch_call_us,
            us(coll.launched_gt_ns, ar_ctl->gt_start),
            us(ar_ctl->gt_start, ar_ctl->gt_stage),
            us(ar_ctl->gt_stage, ar_ctl->gt_first),
            us(ar_ctl->gt_first, ar_ctl->gt_last),
            us(ar_ctl->gt_last, ar_ctl->gt_done), us(ar_ctl->gt_done, now_gt),
            acquire_u32(&ar_ctl->dbg_gate_waits),
            acquire_u32(&ar_ctl->dbg_gate_spins), [&] {
              std::string lags;
              for (size_t p = 0; p < peer_ranks.size(); ++p)
                lags += " r" + std::to_string(peer_ranks[p]) + "=" +
                        std::to_string(static_cast<int>(
                            us(ar_ctl->gt_stage, ar_ctl->gt_claim[p])));
              return lags;
            }());
        if (sampled_prefill)
          DGPP_LOG_INFO("{}", line);
        else
          DGPP_LOG_DEBUG("{}", line);
      }
      // Clear the flight BEFORE completing (finish_flight): the woken
      // waiter's next submission races the cleanup against the
      // single-outstanding check (measured as a spurious "one
      // outstanding" rejection in the TP loopback bring-up) — the bus
      // must be ready for the next generation before the waiter can
      // observe the result. The helper's keep-alive reference outlives
      // the completion's trailing notify.
      finish_flight(coll.req, reduced,
                   reduced ? "" : "collective consumer exited on deadline "
                                  "or poison");
      worked = true;
    }
    return worked;
  }

  bool poll_cqs() {
    bool worked = false;
    ibv_wc wcs[kMaxPollPerCq];
    for (auto& peer_lanes : peers) {
      for (LaneState& lane : peer_lanes) {
        if (lane.failed) continue;
        for (int pool_i = 0; pool_i < 2 && !lane.failed; ++pool_i) {
          const BusPool pool =
              pool_i == 0 ? BusPool::kLatency : BusPool::kBulk;
          int n = lane.lane->poll_tx(pool, wcs, kMaxPollPerCq);
          if (n < 0) {
            fail_lane(lane, "tx poll failed");
            continue;
          }
          for (int i = 0; i < n; ++i) {
            if (wcs[i].status != IBV_WC_SUCCESS) {
              fail_lane(lane, bus_wc_error(wcs[i]));
              break;
            }
            // TX retirement accounting (the bulk arena-reuse fence):
            // each post_pair's doorbell WR is the signaled half, so its
            // CQE retires exactly one message.
            if (bus_wr_kind(wcs[i].wr_id) == BusWr::kDoorbell &&
                pool == BusPool::kBulk)
              ++bulk_doorbell_ce_seen;
          }
          if (n > 0) {
            lane.last_progress = Clock::now();
            worked = true;
          }
          if (lane.failed) continue;

          n = lane.lane->poll_rx(pool, wcs, kMaxPollPerCq);
          if (n < 0) {
            fail_lane(lane, "rx poll failed");
            continue;
          }
          for (int i = 0; i < n; ++i) {
            const ibv_wc& wc = wcs[i];
            if (wc.status != IBV_WC_SUCCESS) {
              fail_lane(lane, bus_wc_error(wc));
              break;
            }
            if (bus_wr_kind(wc.wr_id) != BusWr::kDoorbell) {
              // The payload WR's own CQE: which slot's recv buffer
              // consumed THIS payload — the RQ-order ground truth for
              // the placement hunt (the doorbell CE below names its own
              // slot; if the two ever diverge for one message, the RQ
              // alternation is broken).
              DGPP_LOG_DEBUG(
                  "payload CE: myrank={} peer={} lane={} pool={} slot={}",
                  opt.my_rank, lane.stats.peer_rank, lane.stats.lane, pool_i,
                  bus_wr_slot(wc.wr_id));
              continue;
            }
            const uint32_t slot = bus_wr_slot(wc.wr_id);
            RecvSlot& rs = lane.recv[pool_i][slot];
            // RC ordering: the payload CQE preceded this doorbell CQE on
            // the same CQ, so this means the whole message landed.
            StartSlot* cell = lane.lane->layout().recv_doorbell(
                lane.lane->slab(), pool, slot);
            const uint32_t seq = acquire_u32(&cell->seq);
            const uint32_t len = acquire_u32(&cell->len);
            DGPP_LOG_DEBUG(
                "doorbell CE: myrank={} peer={} lane={} pool={} slot={} "
                "cell_seq={} expect={} len={} cell={}",
                opt.my_rank, lane.stats.peer_rank, lane.stats.lane, pool_i,
                slot, seq, rs.expect_seq, len, (const void*)cell);
            if (rs.arrived) {
              fail_lane(lane, "duplicate doorbell arrival (protocol)");
              break;
            }
            if (seq != rs.expect_seq) {
              fail_lane(lane, "doorbell seq " + std::to_string(seq) +
                                  " != expected " +
                                  std::to_string(rs.expect_seq));
              break;
            }
            rs.arrived = true;
            rs.len = len;
            ++lane.stats.doorbell_recvs;
            lane.stats.bytes_recv += len;
            lane.last_progress = Clock::now();
            worked = true;
          }
        }
      }
    }
    return worked;
  }

  // Receive recycling, in ring order: a slot's pair is reposted (and its
  // credit granted) only after both the GPU ack and the receive
  // completions are known — §6 step 4 — and only when every earlier ring
  // position of the current wrap has already been recycled (a repost out
  // of ring order would reorder the RQ and misalign the sender's cursor).
  bool recycle_pass() {
    bool worked = false;
    for (auto& peer_lanes : peers) {
      for (LaneState& lane : peer_lanes) {
        if (lane.failed) continue;
        RcLane& rc = *lane.lane;
        for (int pool_i = 0; pool_i < 2 && !lane.failed; ++pool_i) {
          const BusPool pool = pool_i == 0 ? BusPool::kLatency : BusPool::kBulk;
          const uint32_t depth = static_cast<uint32_t>(pool_slots(pool));
          for (;;) {
            const uint32_t slot = lane.recycle_cursor[pool_i];
            RecvSlot& rs = lane.recv[pool_i][slot];
            if (!rs.arrived) break;
            FlagAck* ack = rc.layout().ack_cell(rc.slab(), pool, slot);
            if (acquire_u32(&ack->seq) != rs.expect_seq) break;
            const uint64_t hash = acquire_u64(&ack->hash);
            DGPP_LOG_DEBUG(
                "recycle: peer={} lane={} pool={} slot={} seq={} ack_seq={}",
                lane.stats.peer_rank, lane.stats.lane, pool_index(pool), slot,
                rs.expect_seq, ack->seq);
            std::string error;
            if (!rc.post_recv_pair(pool, slot, &error) ||
                !rc.post_credit_write(pool, slot, rs.expect_seq,
                                      lane.peer_endpoint, hash, &error)) {
              fail_lane(lane, error);
              break;
            }
            ++lane.stats.credits_returned;
            rs.arrived = false;
            rs.expect_seq += 1;
            lane.recycle_cursor[pool_i] = (slot + 1) % depth;
            lane.last_progress = Clock::now();
            worked = true;
          }
        }
      }
    }
    return worked;
  }

  // Credit harvest: the peer's completion cells, read as pinned flags.
  bool credits_pass() {
    bool worked = false;
    for (auto& peer_lanes : peers) {
      for (LaneState& lane : peer_lanes) {
        if (lane.failed) continue;
        RcLane& rc = *lane.lane;
        for (int pool_i = 0; pool_i < 2 && !lane.failed; ++pool_i) {
          const BusPool pool = pool_i == 0 ? BusPool::kLatency : BusPool::kBulk;
          for (uint32_t slot = 0;
               slot < static_cast<uint32_t>(pool_slots(pool)) && !lane.failed;
               ++slot) {
            SendSlot& ss = lane.send[pool_i][slot];
            if (!ss.in_flight) continue;
            BusCredit* cell = reinterpret_cast<BusCredit*>(
                rc.layout().completion_cell(rc.slab(), pool, slot));
            const uint32_t seq = acquire_u32(&cell->seq);
            if (seq == 0 || seq == ss.credit_seen) continue;
            if (seq > ss.gen) {
              fail_lane(lane, "credit for unsent generation " +
                                  std::to_string(seq));
              break;
            }
            const uint32_t cslot = acquire_u32(&cell->slot);
            const uint64_t hash = acquire_u64(&cell->hash);
            if (cslot != slot) {
              fail_lane(lane, "credit cell slot mismatch");
              break;
            }
            ss.credit_seen = seq;
            ++lane.stats.credits_received;
            if (seq == ss.gen && ss.owner) {
              std::shared_ptr<BusRequest> req = ss.owner;
              req->stripe_hashes[ss.owner_stripe] = hash;
              ss.owner = nullptr;
              ss.in_flight = false;
              if (lane.in_flight_count > 0) --lane.in_flight_count;
              --req->outstanding;
              // A collective completes on the kernel's ctl stamp; its
              // credits only recycle slots (off the critical path).
              if (!req->is_collective && req->outstanding == 0)
                complete_request(req, true, {});
            }
            // seq < ss.gen: stale credit for a request already failed by
            // the watchdog and freed — nothing to do, the cell is current.
            lane.last_progress = Clock::now();
            worked = true;
          }
        }
      }
    }
    return worked;
  }

  bool watchdog_pass() {
    const auto now = Clock::now();
    bool worked = false;
    for (auto& peer_lanes : peers) {
      for (LaneState& lane : peer_lanes) {
        if (lane.failed || lane.in_flight_count == 0) continue;
        if (now - lane.last_progress >
            std::chrono::milliseconds(opt.completion_timeout_ms)) {
          fail_lane(lane, "inactivity watchdog (no progress for " +
                              std::to_string(opt.completion_timeout_ms) +
                              " ms)");
          worked = true;
        }
      }
    }
    std::vector<std::shared_ptr<BusRequest>> expired;
    {
      std::lock_guard<std::mutex> lock(registry_mu);
      for (auto& entry : registry) {
        BusRequest* req = entry.second.get();
        if (!req->done &&
            now - req->submitted >
                std::chrono::milliseconds(opt.completion_timeout_ms))
          expired.push_back(entry.second);
      }
    }
    for (const std::shared_ptr<BusRequest>& req : expired) {
      if (req->is_collective) {
        poison_collective(*req);
        coll_poisoned = true;
        // The flight must not outlive the request (same reasoning as the
        // done-but-held check in collective_pass; the engine thread runs
        // both passes, so the clear is race-free).
        if (coll.req && coll.req.get() == req.get()) {
          coll = {};
          coll_active.store(false, std::memory_order_relaxed);
        }
      }
      complete_request(req, false,
                       "completion timeout (" +
                           std::to_string(opt.completion_timeout_ms) + " ms)");
      worked = true;
    }
    return worked;
  }

  size_t total_outstanding() {
    std::lock_guard<std::mutex> lock(registry_mu);
    size_t total = 0;
    for (auto& entry : registry) {
      const BusRequest* req = entry.second.get();
      if (req->done) continue;
      total += req->outstanding;
      if (req->offset < req->len) ++total;  // still queued or partial
    }
    return total;
  }

  // A graph window is in progress: armed and not yet walked past. The
  // engine must never enter its idle sleep here — the replay's kernels
  // stage a generation every few hundred microseconds and each one waits
  // on THIS thread's post; a 50-110us sleep cadence would land inside
  // every collective of the step (the eager one-shot path's round trip is
  // covered by the spin phase; a graph window's inter-node compute gaps
  // are not, and they are where the engine goes idle).
  bool graph_window_live() const {
    const uint64_t count = graph.window_count.load(std::memory_order_acquire);
    return count > 0 && (count != graph.adopted_count ||
                         graph.walk_seq <= graph.adopted_last);
  }

  bool drained() {
    std::lock_guard<std::mutex> lq(lat_q_mu);
    if (!lat_q.empty()) return false;
    std::lock_guard<std::mutex> bq(bulk_q_mu);
    if (!bulk_q.empty()) return false;
    if (total_outstanding() != 0) return false;
    if (stream_outstanding()) return false;  // the stream walk must drain (quiesce poisons it)
    // A live graph window keeps the engine alive: the walk must drain
    // (quiesce's poison path handles the deliberate-stop case). The
    // acquire pairs with arm's window publish, so the adopted window
    // state it orders against is the session's. An un-adopted armed
    // window also keeps the engine alive (the walk must at least reach
    // it — the failed-adopt drain above resolves the dead-era case).
    const uint64_t count = graph.window_count.load(std::memory_order_acquire);
    if (count > 0 &&
        (count != graph.adopted_count ||
         graph.walk_seq <= graph.adopted_last))
      return false;
    return true;
  }

  // Pins the engine to one FAST core. The engine's poll loop IS the
  // handshake latency (staging seen -> post -> peer's payload), and the
  // GB10's two core classes differ by 40% in clock: an engine the
  // scheduler parked on a 2.8 GHz core polled measurably slower than one
  // on a 3.9 GHz core (2026-09-02, ranks 1 and 3). A fixed core also
  // keeps the spinner from wandering under a sleeping main thread's
  // wake-affine placement. DGPP_BUS_ENGINE_CPU=n overrides; an
  // unreadable sysfs leaves the thread unpinned.
  void pin_engine_thread() {
    int cpu = -1;
    if (const char* env = std::getenv("DGPP_BUS_ENGINE_CPU")) {
      cpu = std::atoi(env);
    } else {
      // Every online core with its max clock, fastest first (ties: the
      // HIGHER index first — core 0's neighbourhood carries the interrupt
      // load). Bus instances in one process (the loopback tests run a
      // whole world in-process) take successive cores: four engines on
      // one core is four spinners sharing a timeslice, and the 5 s
      // collective watchdog measured exactly that (ctest -j4, 2026-09-02).
      std::vector<std::pair<long, int>> cores;  // (khz, cpu)
      const long n = sysconf(_SC_NPROCESSORS_ONLN);
      for (long c = 0; c < n; ++c) {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) +
                        "/cpufreq/cpuinfo_max_freq");
        long khz = -1;
        if (f >> khz) cores.emplace_back(khz, static_cast<int>(c));
      }
      if (cores.empty()) return;
      std::sort(cores.begin(), cores.end(),
                [](const auto& a, const auto& b) {
                  return a.first != b.first ? a.first > b.first
                                            : a.second > b.second;
                });
      static std::atomic<unsigned> next_instance{0};
      const unsigned i = next_instance.fetch_add(1, std::memory_order_relaxed);
      cpu = cores[i % cores.size()].second;
    }
    if (cpu < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
      DGPP_LOG_WARN("bus engine: could not pin to cpu {} (errno {})", cpu,
                    errno);
      return;
    }
    DGPP_LOG_INFO("bus engine: rank {} pinned to cpu {}", opt.my_rank, cpu);
  }

  void engine_loop() {
    pin_engine_thread();
    int idle = 0;
    for (;;) {
      if (stopping.load(std::memory_order_relaxed) && drained()) break;
      const auto loop_t0 = Clock::now();
      const bool graph_worked = graph_pass();
      const bool stream_worked = stream_pass();
      const bool coll_worked = collective_pass();
      const bool worked = intake();
      const bool polled = poll_cqs();
      const bool recycled = recycle_pass();
      const bool credited = credits_pass();
      const bool watched = watchdog_pass();
      {
        const double it_us = elapsed_us(loop_t0);
        ++loop_n;
        ++loop_hist[it_us < 2 ? 0 : it_us < 5 ? 1 : it_us < 10 ? 2 : it_us < 20 ? 3 : it_us < 50 ? 4 : 5];
        if (it_us > loop_max_us) loop_max_us = it_us;
      }
      if (graph_worked || stream_worked || coll_worked || worked || polled ||
          recycled || credited || watched) {
        idle = 0;
      } else if (++idle > kEngineSpinIterations && !graph_window_live() &&
                 !stream_outstanding() &&
                 !graph.recorded.load(std::memory_order_relaxed)) {
        // The nap is for the pre-serving idle only. Once a decode graph
        // exists this thread never sleeps: a 50 us nap between
        // windows made its core look idle, the scheduler parked another
        // runnable thread there (the main thread's sync spin, a driver
        // thread), and the wake-up waited out that thread's CFS slice —
        // 7-10 ms, PREEMPT_NONE/HZ=250 — exactly when the next window's
        // first generation needed the post. Every other rank then waited
        // at gen 0; the fabric's p99 was one core-share. One core at 100%
        // while serving is the dedicated poller's contract anyway.
        std::this_thread::sleep_for(
            std::chrono::microseconds(kEngineIdleSleepUs));
      }
      // else: hot spin — the loop body is the poll. sched_yield here was
      // measured at ~2.2 ms of poll latency per idle stretch (the collective
      // path idles between engine events while kernels wait on doorbells,
      // and every yield surrendered the timeslice); the sleep phase polls
      // strictly faster than the "spin" phase did. No syscalls on the hot
      // path — including this one.
    }
  }

  // ---- rendezvous ----------------------------------------------------------

  BusRankExchange build_exchange() const {
    BusRankExchange ex;
    ex.rank = opt.my_rank;
    ex.lane_count = static_cast<uint8_t>(opt.lane_devices.size());
    ex.peer_count = static_cast<uint8_t>(peer_ranks.size());
    ex.lat_slots = static_cast<uint32_t>(opt.lat_slots);
    ex.lat_slot_bytes = static_cast<uint32_t>(opt.lat_slot_bytes);
    ex.bulk_slots = static_cast<uint32_t>(opt.bulk_slots);
    ex.bulk_slot_bytes = static_cast<uint32_t>(opt.bulk_slot_bytes);
    for (size_t p = 0; p < peer_ranks.size(); ++p)
      for (size_t l = 0; l < peers[p].size(); ++l)
        ex.peers[p][l] = peers[p][l].endpoint;
    return ex;
  }

  static bool send_error_frame(TcpConn& conn, const std::string& text) {
    const std::string bounded = text.substr(0, kMaxErrorText);
    uint8_t head[8] = {};
    uint32_t magic = kBusErrorMagic;
    uint32_t len = static_cast<uint32_t>(bounded.size());
    std::memcpy(head, &magic, 4);
    std::memcpy(head + 4, &len, 4);
    if (!conn.write_all(head, 8)) return false;
    return bounded.empty() ||
           conn.write_all(bounded.data(), bounded.size());
  }

  static std::string read_error_frame(TcpConn& conn) {
    uint8_t len_buf[4] = {};
    if (!conn.read_exact(len_buf, 4))
      return "peer closed during error frame";
    uint32_t len = 0;
    std::memcpy(&len, len_buf, 4);
    if (len > kMaxErrorText) return "peer sent oversized error frame";
    std::string text(len, '\0');
    if (len > 0 && !conn.read_exact(text.data(), len))
      return "peer closed during error frame";
    return text;
  }

  size_t exchange_frame_bytes() const {
    return kBusExchangeHeaderBytes +
           peer_ranks.size() * lane_count() * kBusExchangeLaneBytes;
  }

  // Pairwise exchange with one connection. The caller sets io deadlines.
  bool handshake(TcpConn& conn, BusRankExchange* peer_out,
                 std::string* error) {
    const size_t frame_len = exchange_frame_bytes();
    std::vector<uint8_t> mine(kBusMaxExchangeBytes);
    const size_t mine_len =
        bus_rank_exchange_encode(ex_, mine.data(), mine.size());
    if (mine_len != frame_len) {
      *error = "internal: exchange encode size mismatch";
      return false;
    }
    if (!conn.write_all(mine.data(), mine_len)) {
      *error = "rendezvous: writing exchange frame failed";
      return false;
    }

    std::vector<uint8_t> peer_buf(kBusMaxExchangeBytes);
    if (!conn.read_exact(peer_buf.data(), 4)) {
      *error = "rendezvous: peer closed before exchange";
      return false;
    }
    uint32_t magic = 0;
    std::memcpy(&magic, peer_buf.data(), 4);
    if (magic == kBusErrorMagic) {
      *error = "rendezvous rejected by peer: " + read_error_frame(conn);
      return false;
    }
    if (!conn.read_exact(peer_buf.data() + 4, frame_len - 4)) {
      *error = "rendezvous: peer exchange short read";
      return false;
    }
    if (!bus_rank_exchange_decode(peer_buf.data(), frame_len, peer_out)) {
      send_error_frame(conn, "exchange frame decode failed");
      *error = "rendezvous: peer sent a malformed exchange frame";
      return false;
    }
    if (peer_out->lane_count != ex_.lane_count ||
        peer_out->peer_count != ex_.peer_count) {
      send_error_frame(conn, "lane/peer count mismatch: mine lanes=" +
                                 std::to_string(ex_.lane_count) + " peers=" +
                                 std::to_string(ex_.peer_count));
      *error = "rendezvous: lane/peer count mismatch with rank " +
               std::to_string(peer_out->rank);
      return false;
    }
    if (peer_out->lat_slots != ex_.lat_slots ||
        peer_out->lat_slot_bytes != ex_.lat_slot_bytes ||
        peer_out->bulk_slots != ex_.bulk_slots ||
        peer_out->bulk_slot_bytes != ex_.bulk_slot_bytes) {
      send_error_frame(
          conn, "slot geometry mismatch: mine lat=" +
                    std::to_string(ex_.lat_slots) + "x" +
                    std::to_string(ex_.lat_slot_bytes) + "B bulk=" +
                    std::to_string(ex_.bulk_slots) + "x" +
                    std::to_string(ex_.bulk_slot_bytes) + "B");
      *error = "rendezvous: slot geometry mismatch with rank " +
               std::to_string(peer_out->rank);
      return false;
    }
    return true;
  }

  std::vector<uint8_t> encode_table(
      const std::vector<BusRankExchange>& table) const {
    const size_t frame_len = exchange_frame_bytes();
    std::vector<uint8_t> wire(8 + table.size() * frame_len);
    uint32_t magic = kBusTableMagic;
    std::memcpy(wire.data(), &magic, 4);
    wire[4] = kBusExchangeVersion;
    wire[5] = static_cast<uint8_t>(table.size());
    std::memset(wire.data() + 6, 0, 2);
    for (size_t r = 0; r < table.size(); ++r)
      bus_rank_exchange_encode(table[r], wire.data() + 8 + r * frame_len,
                               frame_len);
    return wire;
  }

  bool rendezvous_listen(std::string* error) {
    TcpListener listener;
    try {
      listener = TcpListener::bind(opt.rendezvous_port);
    } catch (const std::exception& e) {
      *error = std::string("rendezvous listen: ") + e.what();
      return false;
    }
    DGPP_LOG_INFO("bus: rank 0 rendezvous listening on port {}",
                  listener.port());

    std::vector<TcpConn> conns;
    std::vector<BusRankExchange> frames;
    for (size_t i = 0; i < peer_ranks.size(); ++i) {
      TcpConn conn = listener.accept(opt.rendezvous_timeout_ms);
      if (!conn.valid()) {
        *error = "rendezvous: accept timed out (" +
                 std::to_string(opt.rendezvous_timeout_ms) + " ms) with " +
                 std::to_string(i) + " of " +
                 std::to_string(peer_ranks.size()) + " peers connected";
        return false;
      }
      conn.set_io_deadline_ms(opt.rendezvous_timeout_ms);
      BusRankExchange peer;
      if (!handshake(conn, &peer, error)) return false;
      conns.push_back(std::move(conn));
      frames.push_back(peer);
    }

    // The connected ranks must be exactly 1..world-1.
    std::vector<int> seen;
    for (const BusRankExchange& f : frames) seen.push_back(f.rank);
    std::sort(seen.begin(), seen.end());
    for (size_t i = 0; i < seen.size(); ++i) {
      if (seen[i] != static_cast<int>(i + 1)) {
        *error = "rendezvous: unexpected rank set (rank " +
                 std::to_string(seen[i]) + " at position " +
                 std::to_string(i) + ")";
        return false;
      }
    }

    std::vector<BusRankExchange> table(static_cast<size_t>(opt.world_size));
    table[0] = ex_;
    for (const BusRankExchange& f : frames)
      table[static_cast<size_t>(f.rank)] = f;

    const std::vector<uint8_t> wire = encode_table(table);
    for (TcpConn& conn : conns) {
      if (!conn.write_all(wire.data(), wire.size())) {
        *error = "rendezvous: table broadcast write failed";
        return false;
      }
    }
    install_table(table);
    // Connect barrier: no rank may post until EVERY rank's QPs are RTS
    // with receives posted (see kBusReadyMagic).
    for (size_t i = 0; i < conns.size(); ++i) {
      if (!read_barrier_frame(conns[i], kBusReadyMagic)) {
        *error = "rendezvous: connect barrier — rank " +
                 std::to_string(frames[i].rank) + " never reported READY";
        return false;
      }
    }
    for (TcpConn& conn : conns) {
      if (!write_barrier_frame(conn, kBusGoMagic)) {
        *error = "rendezvous: connect barrier GO write failed";
        return false;
      }
    }
    return true;
  }

  static bool write_barrier_frame(TcpConn& conn, uint32_t magic) {
    return conn.write_all(&magic, sizeof magic);
  }

  static bool read_barrier_frame(TcpConn& conn, uint32_t want) {
    uint32_t got = 0;
    return conn.read_exact(&got, sizeof got) && got == want;
  }

  bool rendezvous_connect(std::string* error) {
    TcpConn conn;
    const auto deadline =
        Clock::now() + std::chrono::milliseconds(opt.rendezvous_timeout_ms);
    for (;;) {
      try {
        conn = TcpConn::connect(opt.rendezvous_host, opt.rendezvous_port,
                                opt.rendezvous_timeout_ms);
        break;
      } catch (const std::exception& e) {
        if (Clock::now() + std::chrono::milliseconds(kConnectRetryMs) >
            deadline) {
          *error = std::string("rendezvous connect: ") + e.what();
          return false;
        }
        std::this_thread::sleep_for(
            std::chrono::milliseconds(kConnectRetryMs));
      }
    }
    conn.set_io_deadline_ms(opt.rendezvous_timeout_ms);

    BusRankExchange rank0;
    if (!handshake(conn, &rank0, error)) return false;

    std::vector<uint8_t> head(8);
    if (!conn.read_exact(head.data(), 8)) {
      *error = "rendezvous: table read failed";
      return false;
    }
    uint32_t magic = 0;
    std::memcpy(&magic, head.data(), 4);
    if (magic == kBusErrorMagic) {
      *error = "rendezvous rejected by rank 0: " + read_error_frame(conn);
      return false;
    }
    if (magic != kBusTableMagic || head[4] != kBusExchangeVersion) {
      *error = "rendezvous: malformed table header";
      return false;
    }
    const size_t world = head[5];
    if (world != static_cast<size_t>(opt.world_size)) {
      *error = "rendezvous: table world " + std::to_string(world) +
               " != configured " + std::to_string(opt.world_size);
      return false;
    }
    const size_t frame_len = exchange_frame_bytes();
    std::vector<uint8_t> wire(world * frame_len);
    if (!conn.read_exact(wire.data(), wire.size())) {
      *error = "rendezvous: table frames short read";
      return false;
    }
    std::vector<BusRankExchange> table(world);
    for (size_t r = 0; r < world; ++r) {
      if (!bus_rank_exchange_decode(wire.data() + r * frame_len, frame_len,
                                    &table[r])) {
        *error = "rendezvous: table frame decode failed";
        return false;
      }
    }
    install_table(table);
    if (!write_barrier_frame(conn, kBusReadyMagic)) {
      *error = "rendezvous: connect barrier READY write failed";
      return false;
    }
    if (!read_barrier_frame(conn, kBusGoMagic)) {
      *error = "rendezvous: connect barrier — rank 0 never sent GO (a peer "
               "failed to connect its QPs, or the barrier timed out)";
      return false;
    }
    return true;
  }

  // Connect every pool's QP from the distributed table, then post the
  // initial receive pairs — the full initial credit grant.
  void install_table(const std::vector<BusRankExchange>& table) {
    for (size_t p = 0; p < peer_ranks.size(); ++p) {
      const int peer_rank = peer_ranks[p];
      const BusRankExchange& peer_frame =
          table[static_cast<size_t>(peer_rank)];
      const size_t my_slot = bus_peer_index(opt.my_rank, peer_rank);
      for (size_t l = 0; l < peers[p].size(); ++l) {
        peers[p][l].peer_endpoint = peer_frame.peers[my_slot][l];
        DGPP_LOG_DEBUG(
            "bus: rank {} lane {} <- peer {} table qpns lat={} bulk={}",
            opt.my_rank, l, peer_rank, peer_frame.peers[my_slot][l].qpn_lat,
            peer_frame.peers[my_slot][l].qpn_bulk);
        for (int pool_i = 0; pool_i < 2; ++pool_i) {
          const BusPool pool =
              pool_i == 0 ? BusPool::kLatency : BusPool::kBulk;
          std::string error;
          if (!peers[p][l].lane->connect(pool, peer_frame.peers[my_slot][l],
                                          &error))
            fail_lane(peers[p][l], error);
        }
      }
    }
    for (auto& peer_lanes : peers) {
      for (LaneState& lane : peer_lanes) {
        if (lane.failed) continue;
        std::string error;
        for (int pool_i = 0; pool_i < 2 && !lane.failed; ++pool_i) {
          const BusPool pool =
              pool_i == 0 ? BusPool::kLatency : BusPool::kBulk;
          for (uint32_t s = 0;
               s < static_cast<uint32_t>(pool_slots(pool)) && !lane.failed;
               ++s) {
            if (!lane.lane->post_recv_pair(pool, s, &error))
              fail_lane(lane, error);
          }
        }
        lane.last_progress = Clock::now();
      }
    }
  }
};

CollectiveBus::CollectiveBus(BusOptions options)
    : impl_(std::make_unique<Impl>()), options_(std::move(options)) {
  impl_->opt = options_;
}

CollectiveBus::~CollectiveBus() { stop(); }

bool CollectiveBus::start(std::string* error) {
  Impl& impl = *impl_;
  const BusOptions& opt = options_;

  if (opt.world_size == 1) {
    if (opt.my_rank != 0) {
      *error = "a world of one has rank 0 only";
      return false;
    }
    if (opt.lat_slot_bytes == 0 ||
        cudaHostAlloc(&impl.w1_stage, opt.lat_slot_bytes, cudaHostAllocMapped) != cudaSuccess) {
      *error = "a world of one: the staging slot allocation failed";
      return false;
    }
    impl.world1 = true;
    DGPP_LOG_INFO("bus: a world of one — no lanes, every collective the identity");
    return true;
  }
  if (opt.world_size < 2 || opt.world_size > 4) {
    *error = "world_size must be 1..4 (1: a world of one, no fabric)";
    return false;
  }
  if (opt.my_rank < 0 || opt.my_rank >= opt.world_size) {
    *error = "my_rank out of range";
    return false;
  }
  // Before anything can spin: materialize the bus kernels (bus_kernel.hpp,
  // bus_preload_kernels — the lazy-loading deadlock).
  if (const cudaError_t err = bus_preload_kernels(); err != cudaSuccess) {
    *error = std::string("bus kernel preload failed: ") +
             cudaGetErrorString(err);
    return false;
  }
  {
    uint64_t uncertainty = 0;
    if (const cudaError_t err =
            bus_globaltimer_offset(&impl.gt_offset_ns, &uncertainty);
        err != cudaSuccess) {
      *error = std::string("globaltimer calibration failed: ") +
               cudaGetErrorString(err);
      return false;
    }
    DGPP_LOG_INFO("bus: globaltimer offset {} ns (+-{} us)", impl.gt_offset_ns,
                  uncertainty / 1000);
  }
  if (opt.lane_devices.empty() || opt.lane_devices.size() > kBusMaxLanes) {
    *error = "1..2 lane devices required";
    return false;
  }
  if (opt.my_rank != 0 && opt.rendezvous_host.empty()) {
    *error = "rendezvous_host is required for non-zero ranks";
    return false;
  }
  if (static_cast<size_t>(opt.lat_slots) + static_cast<size_t>(opt.bulk_slots) >
          kBusMaxConsumerCells ||
      !bus_slab_layout(static_cast<size_t>(opt.lat_slots), opt.lat_slot_bytes,
                       static_cast<size_t>(opt.bulk_slots),
                       opt.bulk_slot_bytes, &impl.layout)) {
    *error = "invalid slot geometry (slot bytes must be 64-multiples >= 64; "
             "combined slots must fit the consumer)";
    return false;
  }
  // The bulk collective kernel's fixed plan/claim arrays bound the
  // geometry: segments (bulk_slots x lanes stripes) must fit the plan
  // tables, and every bulk receive cell (peers x lanes x bulk_slots)
  // must fit the claim bitmap — beyond it, claimed cells stop acking
  // (credits never return) and the flight hangs. Config errors must be
  // legible, never a silent overflow.
  if (static_cast<int>(opt.bulk_slots) * static_cast<int>(opt.lane_devices.size()) >
      kBusMaxBulkSegStripes) {
    *error = "bulk geometry: bulk_slots x lanes exceeds the segment plan (" +
             std::to_string(kBusMaxBulkSegStripes) + " stripes)";
    return false;
  }
  if (static_cast<int>(opt.world_size - 1) *
          static_cast<int>(opt.lane_devices.size()) *
          static_cast<int>(opt.bulk_slots) >
      kBusMaxBulkCells) {
    *error = "bulk geometry: peers x lanes x bulk_slots exceeds the kernel's "
             "claim cells (" +
             std::to_string(kBusMaxBulkCells) + ")";
    return false;
  }

  // One device per distinct lane name so both lanes of a pair share nothing
  // they don't have to (the two active f0s are distinct PCI functions).
  for (const std::string& name : opt.lane_devices) {
    bool have = false;
    for (auto& d : impl.devices)
      if (d->name() == name) have = true;
    if (have) continue;
    std::string open_error;
    auto device = std::make_unique<VerbsDevice>(name, &open_error);
    if (!device->ok()) {
      *error = "bus device open failed: " + open_error;
      return false;
    }
    impl.devices.push_back(std::move(device));
  }

  impl.peer_ranks.clear();
  for (int r = 0; r < opt.world_size; ++r)
    if (r != opt.my_rank) impl.peer_ranks.push_back(r);

  // Collective staging superblock: (peers + 1) staging rows x kStageRing
  // x one latency slot (the peer rows are collective send sources; the
  // extra row is the pre-stage handout), then the bulk arena (one
  // segment's stripes per peer), then the per-peer bulk staged counters.
  // One pinned allocation, one MR per distinct device's PD — any lane can
  // post a SEND sourced from any row.
  {
    const size_t stage_bytes =
        (impl.peer_ranks.size() + 1) *
        static_cast<size_t>(Impl::kStageRing) * impl.stage_pitch();
    const size_t arena_bytes =
        impl.peer_ranks.size() * impl.bulk_seg_bytes();
    const size_t counters_bytes =  // the staged counters, then the
        (kBusMaxPeersSized +       // staging kernel's stripe hashes
         static_cast<size_t>(kBusMaxPeersSized) * kBusMaxBulkSegStripes) *
        sizeof(uint64_t);
    const size_t block_bytes = stage_bytes + arena_bytes + counters_bytes;
    const cudaError_t alloc = cudaHostAlloc(
        reinterpret_cast<void**>(&impl.stage_block), block_bytes,
        cudaHostAllocDefault);
    if (alloc != cudaSuccess || impl.stage_block == nullptr) {
      *error = std::string("collective staging block alloc failed: ") +
               cudaGetErrorString(alloc);
      return false;
    }
    DGPP_LOG_DEBUG("bus: staging block {} bytes (stage {} + arena {} + counters {}), {} device(s)",
                   block_bytes, stage_bytes, arena_bytes, counters_bytes, impl.devices.size());
    for (auto& device : impl.devices) {
      ibv_mr* mr = device->register_memory(impl.stage_block, block_bytes, 0,
                                          "collective staging block", error);
      if (mr == nullptr) return false;
      impl.stage_mrs.push_back(mr);
    }
    DGPP_LOG_DEBUG("bus: staging block {}B x {} MR(s)", block_bytes,
                   impl.stage_mrs.size());
  }

  // Bulk pacing: derive the per-QP rate from the geometry unless the
  // caller fixed it (BusOptions::bulk_pace_gbps).
  if (impl.opt.bulk_pace_gbps < 0) {
    double port_gbps = 0.0;
    for (auto& d : impl.devices)
      if (d->port_rate_gbps() > 0 &&
          (port_gbps == 0.0 || d->port_rate_gbps() < port_gbps))
        port_gbps = d->port_rate_gbps();
    const int inbound_qps =
        (opt.world_size - 1) * static_cast<int>(opt.lane_devices.size());
    if (port_gbps > 0 && inbound_qps > 0) {
      impl.opt.bulk_pace_gbps = port_gbps / inbound_qps * 0.85;
    } else {
      impl.opt.bulk_pace_gbps = 28.0;  // the measured four-node value
      DGPP_LOG_WARN(
          "bus: rank {} could not read a port rate; bulk pacing fixed at "
          "{} Gb/s per QP",
          opt.my_rank, impl.opt.bulk_pace_gbps);
    }
  }

  impl.peers.resize(impl.peer_ranks.size());
  for (auto& peer_lanes : impl.peers)
    peer_lanes.resize(opt.lane_devices.size());
  for (size_t p = 0; p < impl.peer_ranks.size(); ++p) {
    for (size_t l = 0; l < opt.lane_devices.size(); ++l) {
      Impl::LaneState& state = impl.peers[p][l];
      VerbsDevice* dev = nullptr;
      for (auto& d : impl.devices)
        if (d->name() == opt.lane_devices[l]) dev = d.get();
      std::string lane_error;
      state.lane = std::make_unique<RcLane>(*dev, impl.layout, opt.qp_depth,
                                            &lane_error);
      if (!state.lane->ok()) {
        *error = "bus lane create failed (rank " +
                 std::to_string(impl.peer_ranks[p]) + " lane " +
                 std::to_string(l) + "): " + lane_error;
        return false;
      }
      // The staging MR registered on this lane's device (devices are
      // deduped by name; stage_mrs follows the same order).
      for (size_t di = 0; di < impl.devices.size(); ++di)
        if (impl.devices[di].get() == dev)
          state.stage_lkey = impl.stage_mrs[di]->lkey;
      state.endpoint = state.lane->endpoint();
      DGPP_LOG_DEBUG("bus: rank {} peer {} lane {} my qpns lat={} bulk={}",
                     opt.my_rank, impl.peer_ranks[p], l,
                     state.endpoint.qpn_lat, state.endpoint.qpn_bulk);
      state.stats.peer_rank = impl.peer_ranks[p];
      state.stats.lane = static_cast<int>(l);
      state.send[0].assign(static_cast<size_t>(opt.lat_slots), Impl::SendSlot{});
      state.send[1].assign(static_cast<size_t>(opt.bulk_slots), Impl::SendSlot{});
      state.recv[0].assign(static_cast<size_t>(opt.lat_slots), Impl::RecvSlot{});
      state.recv[1].assign(static_cast<size_t>(opt.bulk_slots), Impl::RecvSlot{});
      state.last_progress = Clock::now();
    }
  }

  impl.ex_ = impl.build_exchange();

  std::string rendezvous_error;
  const bool up = opt.my_rank == 0
                       ? impl.rendezvous_listen(&rendezvous_error)
                       : impl.rendezvous_connect(&rendezvous_error);
  if (!up) {
    *error = rendezvous_error;
    return false;
  }
  for (auto& peer_lanes : impl.peers)
    for (auto& lane : peer_lanes)
      if (lane.failed) {
        *error = "bus bring-up failed on at least one lane (see log)";
        return false;
      }

  // Collective-mode plumbing (§6.3): the shared control cell and the
  // stream that serializes per-collective kernels. Allocated regardless of
  // use — two pointers and one cache line, and teardown stays symmetric.
  // The graph era's per-generation cells ride along: one
  // kBusMaxGraphGens slab per selectable variant, pinned for the bus's
  // lifetime (baked into recorded kernel launches).
  {
    const cudaError_t ctl_err =
        cudaMallocHost(reinterpret_cast<void**>(&impl.ar_ctl),
                       sizeof(BusAllReduceCtl));
    if (ctl_err != cudaSuccess || impl.ar_ctl == nullptr) {
      *error = std::string("collective control cell alloc failed: ") +
               cudaGetErrorString(ctl_err);
      return false;
    }
    *impl.ar_ctl = BusAllReduceCtl{};
    const cudaError_t cells_err = cudaMallocHost(
        reinterpret_cast<void**>(&impl.graph_cells),
        kBusMaxGraphVariants * kBusMaxGraphGens *
            sizeof(BusAllReduceCtl));
    if (cells_err != cudaSuccess || impl.graph_cells == nullptr) {
      cudaFreeHost(impl.ar_ctl);
      impl.ar_ctl = nullptr;
      *error = std::string("graph generation cells alloc failed: ") +
               cudaGetErrorString(cells_err);
      return false;
    }
    for (int i = 0; i < kBusMaxGraphVariants * kBusMaxGraphGens; ++i)
      impl.graph_cells[i] = BusAllReduceCtl{};
    const cudaError_t strm_err = cudaMallocHost(
        reinterpret_cast<void**>(&impl.strm.cells), kBusStreamRing * sizeof(BusAllReduceCtl));
    if (strm_err != cudaSuccess || impl.strm.cells == nullptr) {
      cudaFreeHost(impl.graph_cells);
      impl.graph_cells = nullptr;
      cudaFreeHost(impl.ar_ctl);
      impl.ar_ctl = nullptr;
      *error = std::string("stream collective cells alloc failed: ") + cudaGetErrorString(strm_err);
      return false;
    }
    for (int i = 0; i < kBusStreamRing; ++i) impl.strm.cells[i] = BusAllReduceCtl{};
    // The bulk grid's records live in device memory (atomics and every
    // block's reads). No memset: the kernel resets what it uses at entry,
    // and a synchronous memset here is a device-wide barrier against a
    // peer bus's spinning kernels (the loopback worlds).
    const cudaError_t scratch_err = cudaMalloc(
        reinterpret_cast<void**>(&impl.bulk_scratch), sizeof(BusBulkScratch));
    if (scratch_err != cudaSuccess || impl.bulk_scratch == nullptr) {
      cudaFreeHost(impl.strm.cells);
      impl.strm.cells = nullptr;
      cudaFreeHost(impl.graph_cells);
      impl.graph_cells = nullptr;
      cudaFreeHost(impl.ar_ctl);
      impl.ar_ctl = nullptr;
      *error = std::string("bulk scratch alloc failed: ") +
               cudaGetErrorString(scratch_err);
      return false;
    }
    // The one-sided transport's per-block records: pinned, zeroed once (a
    // record carries its generation, so stale values never match).
    const cudaError_t recs_err = cudaMallocHost(
        reinterpret_cast<void**>(&impl.write_recs),
        Impl::kWriteRecCells * kBusWriteMaxBlocks * sizeof(BusWriteBlockRecord));
    if (recs_err != cudaSuccess || impl.write_recs == nullptr) {
      cudaFree(impl.bulk_scratch);
      impl.bulk_scratch = nullptr;
      cudaFreeHost(impl.strm.cells);
      impl.strm.cells = nullptr;
      cudaFreeHost(impl.graph_cells);
      impl.graph_cells = nullptr;
      cudaFreeHost(impl.ar_ctl);
      impl.ar_ctl = nullptr;
      *error = std::string("write record alloc failed: ") + cudaGetErrorString(recs_err);
      return false;
    }
    for (size_t i = 0; i < Impl::kWriteRecCells * kBusWriteMaxBlocks; ++i) impl.write_recs[i] = BusWriteBlockRecord{};
    const cudaError_t stream_err = cudaStreamCreateWithFlags(
        &impl.collective_stream, cudaStreamNonBlocking);
    if (stream_err != cudaSuccess) {
      cudaFreeHost(impl.write_recs);
      impl.write_recs = nullptr;
      cudaFree(impl.bulk_scratch);
      impl.bulk_scratch = nullptr;
      cudaFreeHost(impl.strm.cells);
      impl.strm.cells = nullptr;
      cudaFreeHost(impl.graph_cells);
      impl.graph_cells = nullptr;
      cudaFreeHost(impl.ar_ctl);
      impl.ar_ctl = nullptr;
      *error = std::string("collective stream create failed: ") +
                cudaGetErrorString(stream_err);
      return false;
    }
  }

  // Receive consumers: the Phase 2 harness folds payloads and acks. The
  // engine's real per-collective consumers (deliverable 3) replace these
  // by launching with launch_consumers=false. Each consumer gets its own
  // stream — persistent kernels serialize a stream, and one lane's
  // consumer must never starve another's launch.
  if (opt.launch_consumers) {
    const uint64_t deadline =
        bus_consumer_deadline_cycles(opt.consumer_deadline_s);
    if (deadline == 0) {
      *error = "could not read device clock rate for the consumer watchdog";
      return false;
    }
    for (size_t p = 0; p < impl.peer_ranks.size(); ++p) {
      for (size_t l = 0; l < impl.peers[p].size(); ++l) {
        cudaStream_t stream = nullptr;
        const cudaError_t stream_err =
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (stream_err != cudaSuccess) {
          *error = std::string("consumer stream create failed: ") +
                   cudaGetErrorString(stream_err);
          return false;
        }
        const BusRecvView view =
            recv_view(impl.peer_ranks[p], static_cast<int>(l));
        const cudaError_t launch =
            launch_bus_consumer(view, deadline, stream);
        if (launch != cudaSuccess) {
          cudaStreamDestroy(stream);
          *error = std::string("consumer launch failed: ") +
                   cudaGetErrorString(launch);
          return false;
        }
        impl.consumer_streams.push_back(stream);
      }
    }
    impl.consumers_launched = true;
  }

  impl.engine = std::thread([&impl] { impl.engine_loop(); });
  DGPP_LOG_INFO(
      "bus: rank {} up — {} peer(s) x {} lane(s), lat {}x{}B, bulk {}x{}B, "
      "bulk pace {:.1f} Gb/s per QP, window {} per lane",
      opt.my_rank, impl.peer_ranks.size(), opt.lane_devices.size(),
      opt.lat_slots, opt.lat_slot_bytes, opt.bulk_slots, opt.bulk_slot_bytes,
      impl.opt.bulk_pace_gbps, opt.bulk_inflight_per_lane);
  return true;
}

uint64_t CollectiveBus::send(int peer_rank, const void* data, size_t bytes,
                              BusMessageClass cls, std::string* error) {
  if (impl_->world1) {
    *error = "a world of one has no peers";
    return 0;
  }
  Impl& impl = *impl_;
  if (impl.stopping.load(std::memory_order_relaxed)) {
    *error = "bus is stopped";
    return 0;
  }
  if (impl.coll_mode.load(std::memory_order_relaxed)) {
    *error = "bus is in collective mode; send() is closed (a peer's "
             "collective kernel would fold harness traffic into a reduce)";
    return 0;
  }
  if (bytes == 0 || bytes % 8 != 0) {
    *error = "send size must be a positive multiple of 8";
    return 0;
  }
  if (std::find(impl.peer_ranks.begin(), impl.peer_ranks.end(), peer_rank) ==
      impl.peer_ranks.end()) {
    *error = "rank " + std::to_string(peer_rank) + " is not a bus peer";
    return 0;
  }
  if (cls == BusMessageClass::kLatency && bytes > options_.lat_slot_bytes) {
    *error = "latency message exceeds lat_slot_bytes (" +
             std::to_string(options_.lat_slot_bytes) +
             "B); route it as bulk";
    return 0;
  }

  auto req = std::make_shared<BusRequest>();
  req->peer_rank = peer_rank;
  req->cls = cls;
  req->data = data;
  req->len = bytes;
  req->submitted = Clock::now();
  {
    std::lock_guard<std::mutex> lock(impl.registry_mu);
    req->id = impl.next_id++;
    impl.registry[req->id] = req;
  }
  const uint64_t id = req->id;
  if (cls == BusMessageClass::kLatency) {
    std::lock_guard<std::mutex> lock(impl.lat_q_mu);
    impl.lat_q.push_back(std::move(req));
  } else {
    std::lock_guard<std::mutex> lock(impl.bulk_q_mu);
    impl.bulk_q.push_back(std::move(req));
  }
  return id;
}

BusSendResult CollectiveBus::wait(uint64_t send_id, int timeout_ms) {
  if (impl_->world1) {
    BusSendResult r;
    r.error = "a world of one has no peers";
    return r;
  }
  BusSendResult result;
  Impl& impl = *impl_;
  std::shared_ptr<BusRequest> req;
  {
    std::lock_guard<std::mutex> lock(impl.registry_mu);
    auto it = impl.registry.find(send_id);
    if (it == impl.registry.end()) {
      result.error = "unknown or already-waited send id";
      return result;
    }
    req = it->second;
  }
  const bool done = impl.spin_then_wait(req.get(), timeout_ms);
  if (!done) {
    // Backstop only. The request is not removed: it may still be in flight
    // (SendSlot::owner points at it) and the engine watchdog owns its
    // lifecycle from here.
    result.error = "wait backstop timeout (engine watchdog should have "
                   "failed the request first)";
    return result;
  }
  result.ok = req->ok;
  result.error = req->error;
  result.stripe_hashes = std::move(req->stripe_hashes);
  result.elapsed_us = elapsed_us(req->submitted);
  {
    std::lock_guard<std::mutex> rlock(impl.registry_mu);
    // Only erase if still this request (stop() reaps by completing).
    auto it = impl.registry.find(send_id);
    if (it != impl.registry.end() && it->second.get() == req.get())
      impl.registry.erase(it);
  }
  return result;
}

void* CollectiveBus::stage_next(std::string* error) {
  if (impl_->world1) {
    Impl& w = *impl_;
    std::lock_guard<std::mutex> lock(w.w1_mu);
    if (w.stopping.load(std::memory_order_relaxed)) { *error = "bus is stopped"; return nullptr; }
    if (w.w1_stage_held) { *error = "one pre-stage handout at a time (consume it with allreduce_staged())"; return nullptr; }
    if (w.w1_recording || w.w1_armed > 0) { *error = "stage_next() is closed while a graph records or a window is armed"; return nullptr; }
    w.w1_stage_held = true;
    return w.w1_stage;
  }
  Impl& impl = *impl_;
  if (impl.stopping.load(std::memory_order_relaxed)) {
    *error = "bus is stopped";
    return nullptr;
  }
  {
    std::lock_guard<std::mutex> lock(impl.coll_mu);
    if (impl.coll_poisoned) {
      *error = "an earlier collective failed; this bus must be restarted";
      return nullptr;
    }
    if (const char* gate = impl.eager_gate_closed()) {
      *error = std::string(gate) + "; stage_next() is closed with it";
      return nullptr;
    }
    if (impl.stage_held_ptr != nullptr) {
      *error = "one pre-stage handout at a time (consume it with "
               "allreduce_staged())";
      return nullptr;
    }
    if (impl.stream_outstanding()) {
      *error = "stream collectives are in flight (allreduce_settle first)";
      return nullptr;
    }
    if (impl.coll_active.load(std::memory_order_relaxed) ||
        !impl.coll_q.empty()) {
      *error = "no pre-stage handout while a collective is in flight (v1)";
      return nullptr;
    }
    // Rotate with the collective sequence: consecutive handouts get fresh
    // buffers, and single-outstanding means every prior user of a
    // generation completed (done implies the peer folded the posted bytes,
    // so the NIC read finished — the reuse fence).
    impl.stage_held_gen =
        static_cast<int>(impl.ctl_seq_counter.load(std::memory_order_relaxed) %
                         Impl::kStageRing);
    // The handout lives in the dedicated self row — disjoint from every
    // peer send row, so the kernel's in-place fold can never race a post.
    impl.stage_held_ptr =
        impl.self_buf(impl.peer_ranks.size(), impl.stage_held_gen);
  }
  return impl.stage_held_ptr;
}

uint64_t CollectiveBus::allreduce_staged(size_t bf16_elems,
                                          std::string* error) {
  if (impl_->world1) {
    Impl& w = *impl_;
    std::lock_guard<std::mutex> lock(w.w1_mu);
    if (w.stopping.load(std::memory_order_relaxed)) { *error = "bus is stopped"; return 0; }
    if (!w.w1_stage_held) { *error = "allreduce_staged() without a handout"; return 0; }
    if (bf16_elems == 0 || bf16_elems * 2 > options_.lat_slot_bytes) {
      *error = "allreduce element count must be positive and fit a latency slot";
      return 0;
    }
    w.w1_stage_held = false;
    const uint64_t id = w.w1_next_id++;
    w.w1_pending.insert(id);
    return id;
  }
  Impl& impl = *impl_;
  if (impl.stopping.load(std::memory_order_relaxed)) {
    *error = "bus is stopped";
    return 0;
  }
  if (impl.consumers_launched) {
    *error = "allreduce requires launch_consumers=false (persistent harness "
             "consumers would race the per-collective kernel for claims)";
    return 0;
  }
  if (bf16_elems == 0 || bf16_elems % 2 != 0 ||
      bf16_elems * 2 > options_.lat_slot_bytes) {
    *error = "allreduce element count must be a positive multiple of 2 and "
             "fit a latency slot (at most " +
             std::to_string(options_.lat_slot_bytes / 2) + " bf16)";
    return 0;
  }

  void* staged = nullptr;
  int staged_gen = -1;
  {
    std::lock_guard<std::mutex> lock(impl.coll_mu);
    if (impl.coll_poisoned) {
      *error = "an earlier collective failed; this bus must be restarted";
      return 0;
    }
    if (const char* gate = impl.eager_gate_closed()) {
      *error = std::string(gate) + "; eager collectives are closed with it";
      return 0;
    }
    if (impl.stage_held_ptr == nullptr) {
      *error = "no held pre-stage handout (call stage_next() first)";
      return 0;
    }
    if (impl.stream_outstanding()) {
      *error = "stream collectives are in flight (allreduce_settle first)";
      return 0;
    }
    if (impl.coll_active.load(std::memory_order_relaxed) ||
        !impl.coll_q.empty()) {
      *error = "one outstanding collective at a time (v1)";
      return 0;
    }
    staged = impl.stage_held_ptr;
    staged_gen = impl.stage_held_gen;
    impl.stage_held_ptr = nullptr;  // consumed
  }
  DGPP_LOG_DEBUG("allreduce_staged: handout consumed (gen {})", staged_gen);

  auto req = std::make_shared<BusRequest>();
  req->cls = BusMessageClass::kLatency;
  req->peer_rank = -1;
  req->is_collective = true;
  req->dev_src = staged;
  req->dev_dst = staged;  // in-place fold; the model reads the result here
  req->elems = bf16_elems;
  req->stage_gen = staged_gen;
  req->submitted = Clock::now();
  {
    std::lock_guard<std::mutex> lock(impl.registry_mu);
    req->id = impl.next_id++;
    impl.registry[req->id] = req;
  }
  const uint64_t id = req->id;
  {
    std::lock_guard<std::mutex> lock(impl.coll_mu);
    impl.coll_q.push_back(std::move(req));
  }
  DGPP_LOG_DEBUG("allreduce_staged: queued id={} gen={}", id, staged_gen);
  return id;
}

uint64_t CollectiveBus::allreduce(const void* device_src, void* device_dst,
                                  size_t bf16_elems, std::string* error) {
  if (impl_->world1) {
    Impl& w = *impl_;
    std::lock_guard<std::mutex> lock(w.w1_mu);
    if (w.stopping.load(std::memory_order_relaxed)) { *error = "bus is stopped"; return 0; }
    if (bf16_elems == 0) { *error = "allreduce element count must be positive"; return 0; }
    if (w.w1_stage_held) { *error = "a pre-stage handout is held"; return 0; }
    if (device_dst != device_src &&
        cudaMemcpy(device_dst, device_src, bf16_elems * 2, cudaMemcpyDeviceToDevice) != cudaSuccess) {
      *error = "a world of one: the identity copy failed";
      return 0;
    }
    const uint64_t id = w.w1_next_id++;
    w.w1_pending.insert(id);
    return id;
  }
  Impl& impl = *impl_;
  if (impl.stopping.load(std::memory_order_relaxed)) {
    *error = "bus is stopped";
    return 0;
  }
  if (impl.consumers_launched) {
    *error = "allreduce requires launch_consumers=false (persistent harness "
             "consumers would race the per-collective kernel for claims)";
    return 0;
  }
  if (bf16_elems == 0 || bf16_elems % 2 != 0 ||
      bf16_elems * 2 > options_.lat_slot_bytes) {
    *error = "allreduce element count must be a positive multiple of 2 and "
             "fit a latency slot (at most " +
             std::to_string(options_.lat_slot_bytes / 2) + " bf16)";
    return 0;
  }

  {
    std::lock_guard<std::mutex> lock(impl.coll_mu);
    if (impl.coll_poisoned) {
      *error = "an earlier collective failed; this bus must be restarted";
      return 0;
    }
    if (const char* gate = impl.eager_gate_closed()) {
      *error = std::string(gate) + "; eager collectives are closed with it";
      return 0;
    }
    if (impl.stage_held_ptr != nullptr) {
      // A device-source collective would stage over the held handout's
      // buffer (the GEMM already wrote it) — the held handout owns its
      // generation until consumed.
      *error = "a pre-stage handout is held; consume it with "
               "allreduce_staged() first";
      return 0;
    }
    if (impl.stream_outstanding()) {
      *error = "stream collectives are in flight (allreduce_settle first)";
      return 0;
    }
    if (impl.coll_active.load(std::memory_order_relaxed) ||
        !impl.coll_q.empty()) {
      *error = "one outstanding collective at a time (v1)";
      return 0;
    }
  }

  auto req = std::make_shared<BusRequest>();
  req->cls = BusMessageClass::kLatency;  // latency-class stats/records
  req->peer_rank = -1;
  req->is_collective = true;
  req->dev_src = device_src;
  req->dev_dst = device_dst;
  req->elems = bf16_elems;
  req->submitted = Clock::now();
  {
    std::lock_guard<std::mutex> lock(impl.registry_mu);
    req->id = impl.next_id++;
    impl.registry[req->id] = req;
  }
  const uint64_t id = req->id;
  {
    std::lock_guard<std::mutex> lock(impl.coll_mu);
    impl.coll_q.push_back(std::move(req));
  }
  return id;
}

uint64_t CollectiveBus::allreduce_bulk(const void* device_src, void* device_dst,
                                      size_t bf16_elems, std::string* error) {
  if (impl_->world1) {
    Impl& w = *impl_;
    std::lock_guard<std::mutex> lock(w.w1_mu);
    if (w.stopping.load(std::memory_order_relaxed)) { *error = "bus is stopped"; return 0; }
    if (bf16_elems == 0) { *error = "allreduce element count must be positive"; return 0; }
    if (w.w1_stage_held) { *error = "a pre-stage handout is held"; return 0; }
    if (device_dst != device_src &&
        cudaMemcpy(device_dst, device_src, bf16_elems * 2, cudaMemcpyDeviceToDevice) != cudaSuccess) {
      *error = "a world of one: the identity copy failed";
      return 0;
    }
    const uint64_t id = w.w1_next_id++;
    w.w1_pending.insert(id);
    return id;
  }
  Impl& impl = *impl_;
  if (impl.stopping.load(std::memory_order_relaxed)) {
    *error = "bus is stopped";
    return 0;
  }
  if (impl.consumers_launched) {
    *error = "allreduce requires launch_consumers=false (persistent harness "
             "consumers would race the per-collective kernel for claims)";
    return 0;
  }
  if (bf16_elems == 0 || bf16_elems % 2 != 0) {
    *error = "allreduce element count must be a positive multiple of 2";
    return 0;
  }
  if (bf16_elems > 0xFFFFFFFFull) {
    *error = "bulk collective element count exceeds the u32 plan";
    return 0;
  }

  {
    std::lock_guard<std::mutex> lock(impl.coll_mu);
    if (impl.coll_poisoned) {
      *error = "an earlier collective failed; this bus must be restarted";
      return 0;
    }
    if (const char* gate = impl.eager_gate_closed()) {
      *error = std::string(gate) + "; eager collectives are closed with it";
      return 0;
    }
    if (impl.stage_held_ptr != nullptr) {
      *error = "a pre-stage handout is held; consume it with "
               "allreduce_staged() first";
      return 0;
    }
    if (impl.stream_outstanding()) {
      *error = "stream collectives are in flight (allreduce_settle first)";
      return 0;
    }
    if (impl.coll_active.load(std::memory_order_relaxed) ||
        !impl.coll_q.empty()) {
      *error = "one outstanding collective at a time (v1)";
      return 0;
    }
  }

  // Geometry: the bulk-slot grid and segments of at most bulk_slots x
  // lanes stripes (every posting wave then fits the pool depths by
  // construction); the shards are split per segment in the engine's plan.
  const size_t stripe_elems = options_.lat_slot_bytes == 0
                                  ? 0
                                  : options_.bulk_slot_bytes / 2;
  const uint32_t C = static_cast<uint32_t>(
      (bf16_elems + stripe_elems - 1) / stripe_elems);
  const uint32_t seg_stripes = impl.bulk_seg_stripes();
  const uint32_t seg_count = (C + seg_stripes - 1) / seg_stripes;

  auto req = std::make_shared<BusRequest>();
  req->cls = BusMessageClass::kBulk;  // stats/records bucket
  req->peer_rank = -1;
  req->is_collective = true;
  req->is_bulk = true;
  req->dev_src = device_src;
  req->dev_dst = device_dst;
  req->elems = bf16_elems;
  req->bulk_stripes = C;
  req->bulk_seg_stripes = seg_stripes;
  req->bulk_seg_count = seg_count;
  req->submitted = Clock::now();
  {
    std::lock_guard<std::mutex> lock(impl.registry_mu);
    req->id = impl.next_id++;
    impl.registry[req->id] = req;
  }
  const uint64_t id = req->id;
  {
    std::lock_guard<std::mutex> lock(impl.coll_mu);
    impl.coll_q.push_back(std::move(req));
  }
  DGPP_LOG_DEBUG("allreduce_bulk: queued id={} elems={} stripes={} segs={}",
                 id, bf16_elems, C, seg_count);
  return id;
}

uint64_t CollectiveBus::allreduce_stream(cudaStream_t stream, const void* device_src,
                                         void* device_dst, size_t bf16_elems,
                                         std::string* error) {
  if (impl_->world1) {
    Impl& w = *impl_;
    std::lock_guard<std::mutex> lock(w.w1_mu);
    if (w.stopping.load(std::memory_order_relaxed)) { *error = "bus is stopped"; return 0; }
    if (bf16_elems == 0) { *error = "allreduce element count must be positive"; return 0; }
    if (w.w1_stage_held) { *error = "a pre-stage handout is held"; return 0; }
    if (device_dst != device_src &&
        cudaMemcpyAsync(device_dst, device_src, bf16_elems * 2, cudaMemcpyDeviceToDevice, stream) !=
            cudaSuccess) {
      *error = "a world of one: the identity copy failed";
      return 0;
    }
    return w.w1_next_id++;
  }
  Impl& impl = *impl_;
  if (impl.stopping.load(std::memory_order_relaxed)) {
    *error = "bus is stopped";
    return 0;
  }
  if (impl.consumers_launched) {
    *error = "allreduce_stream requires launch_consumers=false (persistent harness "
             "consumers would race the per-collective kernel for claims)";
    return 0;
  }
  if (bf16_elems == 0 || bf16_elems % 2 != 0 ||
      bf16_elems * 2 > options_.lat_slot_bytes) {
    *error = "allreduce element count must be a positive multiple of 2 and "
             "fit a latency slot (at most " +
             std::to_string(options_.lat_slot_bytes / 2) + " bf16)";
    return 0;
  }
  // The ring: at most kBusStreamRing generations between issue and walk
  // (the walk frees them in order; the wait is bounded by the flight
  // watchdog's era failure).
  {
    const auto deadline =
        Clock::now() + std::chrono::milliseconds(std::max(options_.completion_timeout_ms + 5000, 30000));
    while (impl.strm.issued.load(std::memory_order_relaxed) -
               impl.strm.walked.load(std::memory_order_acquire) >=
           static_cast<uint64_t>(kBusStreamRing)) {
      if (impl.graph.failed.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> lock(impl.coll_mu);
        *error = "graph era failed: " + impl.graph.error;
        return 0;
      }
      if (impl.stopping.load(std::memory_order_relaxed)) {
        *error = "bus stopped while a stream collective waited for the ring";
        return 0;
      }
      if (Clock::now() > deadline) {
        *error = "stream ring backstop: the engine did not walk the ring's oldest generation";
        return 0;
      }
      cpu_relax();
    }
  }
  uint64_t gen = 0;
  int cell_idx = 0;
  BusAllReduceCtl* cell = nullptr;
  {
    std::lock_guard<std::mutex> lock(impl.coll_mu);
    if (impl.coll_poisoned) {
      *error = "an earlier collective failed; this bus must be restarted";
      return 0;
    }
    if (const char* gate = impl.eager_gate_closed()) {
      *error = std::string(gate) + "; stream collectives are closed with it";
      return 0;
    }
    if (impl.stage_held_ptr != nullptr) {
      *error = "a pre-stage handout is held; consume it with allreduce_staged() first";
      return 0;
    }
    if (impl.coll_active.load(std::memory_order_relaxed) || !impl.coll_q.empty()) {
      *error = "a host-driven collective is queued or in flight (wait for it before "
               "issuing stream collectives)";
      return 0;
    }
    if (impl.strm.deadline_cycles == 0) {
      impl.strm.deadline_cycles = impl.graph.deadline_cycles != 0
                                      ? impl.graph.deadline_cycles
                                      : bus_consumer_deadline_cycles(options_.consumer_deadline_s);
      if (impl.strm.deadline_cycles == 0) {
        *error = "could not read the device clock rate for the collective deadline";
        return 0;
      }
    }
    // The generation from the shared counter (execution order equals
    // generation order: one forward thread issues on one stream); the
    // 32-bit top fails the era as a replay arm does.
    if (impl.ctl_seq_counter.load(std::memory_order_relaxed) == 0xFFFFFFFFu) {
      bool expected = false;
      if (impl.graph.failed.compare_exchange_strong(expected, true)) {
        impl.graph.error = "collective generation space exhausted (restart the process)";
        DGPP_LOG_ERROR("bus graph era failed: {}", impl.graph.error);
      }
      *error = impl.graph.error;
      return 0;
    }
    gen = static_cast<uint64_t>(impl.ctl_seq_counter.fetch_add(1, std::memory_order_relaxed)) + 1;
    cell_idx = static_cast<int>(gen % static_cast<uint64_t>(kBusStreamRing));
    cell = &impl.strm.cells[cell_idx];
    // The cell reset, gen_seq last with release (the kernel's acquire read
    // orders the resets ahead of its execution).
    __atomic_store_n(&cell->ready_bits, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&cell->done_seq, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&cell->status, 0, __ATOMIC_RELAXED);
    cell->stamp_stage = 0;
    cell->stamp_first_claim = 0;
    cell->stamp_reduce_done = 0;
    cell->gt_start = cell->gt_stage = cell->gt_first = cell->gt_last = cell->gt_done = 0;
    for (uint64_t& g : cell->gt_claim) g = 0;
    for (uint64_t& g : cell->gt_landed) g = 0;
    for (uint64_t& g : cell->gt_claim_first_neg) g = 0;
    __atomic_store_n(&cell->dbg_gate_waits, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&cell->dbg_gate_spins, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&cell->gen_seq, gen, __ATOMIC_RELEASE);
    // Issued before the launch: the engine adopts from the FIFO, so a
    // launch that fails is withdrawn below before the walk can see it.
    impl.strm.fifo.push_back(Impl::StreamState::Gen{gen, cell_idx, static_cast<uint32_t>(bf16_elems)});
    impl.strm.issued.store(gen, std::memory_order_release);
  }
  BusAllReduceGraphView v{};
  int vi = 0;
  for (size_t p = 0; p < impl.peer_ranks.size(); ++p)
    for (const Impl::LaneState& lane : impl.peers[p])
      v.recv[vi++] = impl.recv_view_of(lane);
  v.recv_views = vi;
  v.lanes_per_peer = static_cast<int>(impl.lane_count());
  v.stage_row_base = reinterpret_cast<uint16_t*>(impl.stage_buf(0, 0));
  v.send_peers = static_cast<int>(impl.peer_ranks.size());
  v.stage_ring = Impl::kStageRing;
  v.stage_row_bytes = static_cast<uint32_t>(impl.stage_pitch());
  const cudaError_t launch =
      impl.write_transport()
          ? launch_bus_allreduce_write(
                v, options_.my_rank, static_cast<const __nv_bfloat16*>(device_src),
                static_cast<__nv_bfloat16*>(device_dst), static_cast<uint32_t>(bf16_elems), cell,
                impl.write_recs_of_stream(cell_idx), impl.strm.deadline_cycles, stream)
          : launch_bus_allreduce_graph(
                v, options_.my_rank, static_cast<const __nv_bfloat16*>(device_src),
                static_cast<__nv_bfloat16*>(device_dst), static_cast<uint32_t>(bf16_elems), cell,
                impl.strm.deadline_cycles, stream);
  if (launch != cudaSuccess) {
    // No kernel will stamp this generation: fail the era (the walk's
    // drain-fail poisons the queue) — the numbering cannot be given back.
    impl.graph_fail(std::string("stream collective kernel launch failed: ") + cudaGetErrorString(launch));
    *error = impl.graph.error;
    return 0;
  }
  return gen;
}

bool CollectiveBus::allreduce_settle(int timeout_ms, std::string* error) {
  if (impl_->world1) return true;
  Impl& impl = *impl_;
  const uint64_t need = impl.strm.issued.load(std::memory_order_acquire);
  if (need == 0) return true;
  const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
  while (impl.strm.walked.load(std::memory_order_acquire) < need) {
    if (impl.graph.failed.load(std::memory_order_relaxed)) {
      std::lock_guard<std::mutex> lock(impl.coll_mu);
      *error = "graph era failed: " + impl.graph.error;
      return false;
    }
    if (impl.stopping.load(std::memory_order_relaxed)) {
      *error = "bus stopped with stream collectives in flight";
      return false;
    }
    if (Clock::now() > deadline) {
      *error = "settle backstop: the engine did not walk the stream collectives (the "
               "flight watchdog should have failed the era first)";
      return false;
    }
    cpu_relax();
  }
  if (impl.graph.failed.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> lock(impl.coll_mu);
    *error = "graph era failed: " + impl.graph.error;
    return false;
  }
  impl.log_stream_timeline();
  return true;
}

BusAllReduceResult CollectiveBus::wait_allreduce(uint64_t id,
                                                  int timeout_ms) {
  const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
  if (impl_->world1) {
    Impl& w = *impl_;
    BusAllReduceResult r;
    std::lock_guard<std::mutex> lock(w.w1_mu);
    if (w.w1_pending.erase(id) == 0) { r.error = "unknown or already-waited collective id"; return r; }
    r.ok = true;
    return r;
  }
  BusAllReduceResult result;
  Impl& impl = *impl_;
  std::shared_ptr<BusRequest> req;
  {
    std::lock_guard<std::mutex> lock(impl.registry_mu);
    auto it = impl.registry.find(id);
    if (it == impl.registry.end()) {
      result.error = "unknown or already-waited collective id";
      return result;
    }
    req = it->second;
  }
  const bool done = impl.spin_then_wait(req.get(), timeout_ms);
  if (!done) {
    result.error = "wait backstop timeout (engine watchdog should have "
                   "failed the request first)";
    impl.coll_poisoned.store(true, std::memory_order_release);
    return result;
  }
  result.ok = req->ok;
  result.error = req->error;
  result.elapsed_us = elapsed_us(req->submitted);
  DGPP_LOG_DEBUG("allreduce wait: submit->wake={:.1f}us", elapsed_us(req->submitted));
  if (result.ok && impl.collective_stream != nullptr) {
    const cudaError_t sync = dgpp::wait_cuda_stream_until(impl.collective_stream, deadline);
    if (sync != cudaSuccess) {
      result.ok = false;
      impl.coll_poisoned.store(true, std::memory_order_release);
      result.error = sync == cudaErrorNotReady
                         ? "collective stream completion timeout after the request's done stamp"
                         : std::string("collective stream sync failed: ") + cudaGetErrorString(sync);
    }
  }
  {
    std::lock_guard<std::mutex> rlock(impl.registry_mu);
    auto it = impl.registry.find(id);
    if (it != impl.registry.end() && it->second.get() == req.get())
      impl.registry.erase(it);
  }
  return result;
}

// ---- graph capture (DESIGN §6.2, the decode step) ---------------------------

// Bounded spin-then-sleep join on a graph walk predicate: the walk trails
// the kernels by one engine poll cadence (microseconds when healthy), so
// the spin phase is the whole story; the sleep phase and the backstop are
// for a wedged engine — and to fail legibly instead of hanging forever.
int64_t CollectiveBus::globaltimer_offset_ns() const {
  return impl_->gt_offset_ns;
}

bool CollectiveBus::graph_replay_arm(std::string* error, int variant) {
  if (impl_->world1) {
    Impl& w = *impl_;
    std::lock_guard<std::mutex> lock(w.w1_mu);
    (void)variant;
    if (w.stopping.load(std::memory_order_relaxed)) { *error = "bus is stopped"; return false; }
    if (w.w1_stage_held) { *error = "graph_replay_arm: a staging handout is held"; return false; }
    if (w.w1_recording) { *error = "graph_replay_arm: a session is recording"; return false; }
    ++w.w1_armed;
    return true;
  }
  Impl& impl = *impl_;
  if (impl.stopping.load(std::memory_order_relaxed)) {
    *error = "bus is stopped";
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(impl.coll_mu);
    if (impl.graph.failed.load(std::memory_order_relaxed)) {
      *error = "graph era failed: " + impl.graph.error;
      return false;
    }
    if (!impl.graph.recorded.load(std::memory_order_relaxed) ||
        impl.graph.recording) {
      *error = "no recorded graph (open and close a session first)";
      return false;
    }
    if (variant < 0 || variant >= kBusMaxGraphVariants) {
      *error = "graph variant outside [0, " +
               std::to_string(kBusMaxGraphVariants) + ")";
      return false;
    }
    const Impl::GraphState::Variant& shape =
        impl.graph.variants[variant];
    if (!shape.recorded) {
      *error = "graph variant " + std::to_string(variant) +
               " is not recorded";
      return false;
    }
    if (impl.graph.live_windows >= kBusMaxLiveWindows) {
      *error = std::to_string(kBusMaxLiveWindows) +
               " graph replay windows are already armed";
      return false;
    }
    if (impl.graph.variant_live[static_cast<size_t>(variant)]) {
      *error = "graph variant " + std::to_string(variant) +
               " has a live replay window (its cells are in flight)";
      return false;
    }
    const int gens = shape.gens;
    const uint64_t armed_gt = static_cast<uint64_t>(
        static_cast<int64_t>(monotonic_ns()) + impl.gt_offset_ns);
    if (impl.stage_held_ptr != nullptr) {
      // The handout's row was picked for "the next collective" — a window
      // would claim that number range and rewrite the row out from
      // under it (record_begin rejects the same class of collision).
      *error = "a pre-stage handout is held; consume it before arming a "
               "replay window";
      return false;
    }

    // Wait for the engine to walk the window kBusMaxLiveWindows back (its
    // reserved last generation plus one): this arm reuses its ring entry,
    // and the walk order bounds how far the engine can lag. The first
    // arms skip.
    const uint64_t prev =
        impl.graph.window_count.load(std::memory_order_relaxed);
    uint64_t need = 0;
    if (prev >= static_cast<uint64_t>(kBusMaxLiveWindows)) {
      const uint64_t back = prev + 1 - kBusMaxLiveWindows;  // window number
      const Impl::GraphState::Window& w =
          impl.graph.windows[(back - 1) % kBusWindowRing];
      if (w.variant < 0 || w.variant >= kBusMaxGraphVariants ||
          !impl.graph.variants[w.variant].recorded) {
        *error = "an armed graph window has an invalid variant";
        return false;
      }
      need = w.first + w.gens;
    }
    const auto deadline =
        Clock::now() + std::chrono::milliseconds(std::max(
                           options_.completion_timeout_ms + 5000, 30000));
    while (impl.graph.walk_pub.load(std::memory_order_acquire) < need) {
      if (impl.graph.failed.load(std::memory_order_relaxed)) {
        *error = "graph era failed while arming: " + impl.graph.error;
        return false;
      }
      if (impl.stopping.load(std::memory_order_relaxed)) {
        *error = "bus stopped while arming";
        return false;
      }
      if (Clock::now() > deadline) {
        *error = "arm backstop: engine did not walk the previous window";
        return false;
      }
      cpu_relax();
    }

    // Reserve the window's generations from the SHARED collective
    // counter — eager pickups between windows take theirs from the same
    // one, so execution order equals generation order across eras and
    // the staging-ring reuse fences hold verbatim. A reservation that
    // would cross the 32-bit top fails the era loudly: the counter space
    // is ~4.3e9 collectives (~48M decode tokens at 90 gens); the remedy
    // is a process restart. (Inline failure: graph_fail would take
    // coll_mu, which this arm already holds.)
    if (impl.ctl_seq_counter.load(std::memory_order_relaxed) >
        0xFFFFFFFFu - static_cast<uint32_t>(gens)) {
      bool expected = false;
      if (impl.graph.failed.compare_exchange_strong(expected, true)) {
        impl.graph.error =
            "collective generation space exhausted (restart the process)";
        DGPP_LOG_ERROR("bus graph era failed: {}", impl.graph.error);
      }
      *error = impl.graph.error;
      return false;
    }
    const uint64_t first =
        static_cast<uint64_t>(
            impl.ctl_seq_counter.fetch_add(static_cast<uint32_t>(gens),
                                            std::memory_order_relaxed)) +
        1;

    // Reset the cells and assign the window's generations. The gen_seq
    // store is RELEASE and last per cell: the kernel's acquire read of it
    // orders the resets ahead of its execution (monotonicity also means
    // a previous replay's stale done stamp can never match this one).
    for (int g = 0; g < gens; ++g) {
      BusAllReduceCtl* cell = &impl.graph_variant_cells(variant)[g];
      __atomic_store_n(&cell->ready_bits, 0, __ATOMIC_RELAXED);
      __atomic_store_n(&cell->done_seq, 0, __ATOMIC_RELAXED);
      __atomic_store_n(&cell->status, 0, __ATOMIC_RELAXED);
      cell->stamp_stage = 0;
      cell->stamp_first_claim = 0;
      cell->stamp_reduce_done = 0;
      cell->gt_start = cell->gt_stage = cell->gt_first = cell->gt_last =
          cell->gt_done = 0;
      for (uint64_t& g : cell->gt_claim) g = 0;
    for (uint64_t& g : cell->gt_landed) g = 0;
    for (uint64_t& g : cell->gt_claim_first_neg) g = 0;
      __atomic_store_n(&cell->dbg_gate_waits, 0, __ATOMIC_RELAXED);
      __atomic_store_n(&cell->dbg_gate_spins, 0, __ATOMIC_RELAXED);
      __atomic_store_n(&cell->gen_seq, first + g, __ATOMIC_RELEASE);
    }
    // Close the era's eager gate (arm .. finish), then publish: the ring
    // entry, window_first and the variant first (relaxed — the
    // window_count release below makes them visible to the engine's
    // acquire), window_count LAST.
    ++impl.graph.live_windows;
    impl.graph.variant_live[static_cast<size_t>(variant)] = true;
    Impl::GraphState::Window& w =
        impl.graph.windows[prev % kBusWindowRing];  // window number prev + 1
    w.variant = variant;
    w.first = first;
    w.gens = static_cast<uint64_t>(gens);
    w.armed_gt = armed_gt;
    impl.graph.window_variant.store(variant, std::memory_order_relaxed);
    impl.graph.window_first.store(first, std::memory_order_relaxed);
    impl.graph.window_count.fetch_add(1, std::memory_order_release);
  }
  return true;
}

bool CollectiveBus::graph_replay_finish(int timeout_ms, std::string* error) {
  if (impl_->world1) {
    Impl& w = *impl_;
    std::lock_guard<std::mutex> lock(w.w1_mu);
    (void)timeout_ms;
    if (w.w1_armed == 0) { *error = "graph_replay_finish: no armed window"; return false; }
    --w.w1_armed;
    return true;
  }
  Impl& impl = *impl_;
  if (impl.stopping.load(std::memory_order_relaxed)) {
    *error = "bus is stopped";
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(impl.coll_mu);
    if (!impl.graph.recorded.load(std::memory_order_relaxed)) {
      *error = "no recorded graph";
      return false;
    }
    if (impl.graph.failed.load(std::memory_order_relaxed)) {
      *error = "graph era failed: " + impl.graph.error;
      return false;
    }
  }
  // The oldest armed window (windows finish in arm order): its ring
  // entry, published under coll_mu at its arm.
  uint64_t need = 0;
  int variant = -1;
  {
    std::lock_guard<std::mutex> lock(impl.coll_mu);
    const uint64_t count =
        impl.graph.window_count.load(std::memory_order_acquire);
    if (impl.graph.finished_count >= count) {
      *error = "no armed window (arm before finish)";
      return false;
    }
    const Impl::GraphState::Window& w =
        impl.graph.windows[impl.graph.finished_count % kBusWindowRing];
    variant = w.variant;
    if (variant < 0 || variant >= kBusMaxGraphVariants ||
        !impl.graph.variants[variant].recorded) {
      *error = "armed graph window has an invalid variant";
      return false;
    }
    need = w.first + w.gens;
  }
  const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
  while (impl.graph.walk_pub.load(std::memory_order_acquire) < need) {
    if (impl.graph.failed.load(std::memory_order_relaxed)) {
      std::lock_guard<std::mutex> lock(impl.coll_mu);
      *error = "graph era failed: " + impl.graph.error;
      return false;
    }
    if (impl.stopping.load(std::memory_order_relaxed)) {
      *error = "bus stopped mid-walk";
      return false;
    }
    if (Clock::now() > deadline) {
      *error = "finish backstop: engine did not walk the window (the "
               "watchdog should have failed the era first)";
      return false;
    }
    cpu_relax();
  }
  // The window is walked: retire it (the era's eager gate reopens with
  // the last live window; coll_mu-serialized with the eager submissions
  // that check it).
  {
    std::lock_guard<std::mutex> lock(impl.coll_mu);
    ++impl.graph.finished_count;
    --impl.graph.live_windows;
    impl.graph.variant_live[static_cast<size_t>(variant)] = false;
  }
  return true;
}

bool CollectiveBus::graph_record_begin(std::string* error, int variant) {
  if (impl_->world1) {
    Impl& w = *impl_;
    std::lock_guard<std::mutex> lock(w.w1_mu);
    (void)variant;
    if (w.stopping.load(std::memory_order_relaxed)) { *error = "bus is stopped"; return false; }
    if (w.w1_stage_held) { *error = "graph_record_begin: a staging handout is held"; return false; }
    if (w.w1_recording) { *error = "graph_record_begin: a session is already recording"; return false; }
    if (w.w1_armed > 0) { *error = "graph_record_begin: a replay window is armed"; return false; }
    w.w1_recording = true;
    return true;
  }
  Impl& impl = *impl_;
  if (impl.stopping.load(std::memory_order_relaxed)) {
    *error = "bus is stopped";
    return false;
  }
  if (impl.consumers_launched) {
    *error = "graph mode requires launch_consumers=false (persistent "
             "harness consumers would race the collective kernels)";
    return false;
  }
  if (impl.coll_poisoned || impl.graph.failed.load(std::memory_order_relaxed)) {
    *error = "an earlier collective failed; this bus must be restarted";
    return false;
  }
  if (variant < 0 || variant >= kBusMaxGraphVariants) {
    *error = "graph variant outside [0, " +
             std::to_string(kBusMaxGraphVariants) + ")";
    return false;
  }
  // The kernel deadline, computed before anything records (every launch
  // bakes it) and outside any capture — attribute queries during capture
  // are best avoided.
  if (impl.graph.deadline_cycles == 0) {
    int clock_khz = 0;
    if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0) !=
            cudaSuccess ||
        clock_khz <= 0) {
      *error = "could not read device clock rate for the graph deadline";
      return false;
    }
    impl.graph.deadline_cycles = static_cast<uint64_t>(
        static_cast<double>(clock_khz) * 1000.0 *
        options_.consumer_deadline_s);
  }
  {
    std::lock_guard<std::mutex> lock(impl.coll_mu);
    if (impl.graph.recording) {
      *error = "a graph recording session is already open";
      return false;
    }
    if (impl.graph.live_windows > 0) {
      *error = "cannot record a graph variant while a replay is armed";
      return false;
    }
    if (impl.graph.variants[variant].recorded) {
      *error = "graph variant " + std::to_string(variant) +
               " is already recorded";
      return false;
    }
    if (impl.stage_held_ptr != nullptr) {
      *error = "a pre-stage handout is held; consume it before opening a "
               "graph session";
      return false;
    }
    if (impl.stream_outstanding()) {
      *error = "stream collectives are in flight (allreduce_settle first)";
      return false;
    }
    if (impl.coll_active.load(std::memory_order_relaxed) ||
        !impl.coll_q.empty()) {
      *error = "no graph session while an eager collective is in flight (v1)";
      return false;
    }
    impl.graph.recording = true;
    impl.graph.recording_variant = variant;
    Impl::GraphState::Variant& shape = impl.graph.variants[variant];
    shape.gens = 0;
    shape.meta.clear();
    // Close the harness send path for the era: the recorded kernels claim
    // doorbells exactly like eager collectives, and a harness message
    // would be folded into a reduce — the silent-corruption class.
    impl.coll_mode.store(true, std::memory_order_relaxed);
  }
  return true;
}

bool CollectiveBus::allreduce_record(cudaStream_t capture_stream,
                                     const void* device_src, void* device_dst,
                                     size_t bf16_elems, std::string* error) {
  if (impl_->world1) {
    Impl& w = *impl_;
    std::lock_guard<std::mutex> lock(w.w1_mu);
    if (!w.w1_recording) { *error = "allreduce_record outside a recording session"; return false; }
    if (bf16_elems == 0) { *error = "allreduce element count must be positive"; return false; }
    // In place there is nothing to record; a copy would be a memcpy node
    // (the kernels-only census rejects it — no reducer records one).
    if (device_dst != device_src &&
        cudaMemcpyAsync(device_dst, device_src, bf16_elems * 2, cudaMemcpyDeviceToDevice, capture_stream) != cudaSuccess) {
      *error = "a world of one: the identity copy node failed";
      return false;
    }
    return true;
  }
  Impl& impl = *impl_;
  if (bf16_elems == 0 || bf16_elems % 2 != 0 ||
      bf16_elems * 2 > options_.lat_slot_bytes) {
    *error = "graph collective element count must be a positive multiple of 2 "
             "and fit a latency slot (at most " +
             std::to_string(options_.lat_slot_bytes / 2) + " bf16)";
    return false;
  }

  std::lock_guard<std::mutex> lock(impl.coll_mu);
  if (!impl.graph.recording) {
    *error = "no open graph session (call graph_record_begin first)";
    return false;
  }
  const int variant = impl.graph.recording_variant;
  if (variant < 0 || variant >= kBusMaxGraphVariants) {
    *error = "open graph session has an invalid variant";
    return false;
  }
  Impl::GraphState::Variant& shape = impl.graph.variants[variant];
  if (shape.gens >= kBusMaxGraphGens) {
    *error = "graph generation budget exceeded (" +
             std::to_string(kBusMaxGraphGens) + " collective nodes)";
    return false;
  }

  // The replay-stable view: everything baked here must outlive the graph.
  // The recv slabs and the staging block are pinned for the bus's
  // lifetime; src/dst are the caller's contract (documented).
  BusAllReduceGraphView v{};
  int vi = 0;
  for (size_t p = 0; p < impl.peer_ranks.size(); ++p)
    for (const Impl::LaneState& lane : impl.peers[p])
      v.recv[vi++] = impl.recv_view_of(lane);
  v.recv_views = vi;
  v.lanes_per_peer = static_cast<int>(impl.lane_count());
  v.stage_row_base = reinterpret_cast<uint16_t*>(impl.stage_buf(0, 0));
  v.send_peers = static_cast<int>(impl.peer_ranks.size());
  v.stage_ring = Impl::kStageRing;
  v.stage_row_bytes = static_cast<uint32_t>(impl.stage_pitch());

  const cudaError_t launch =
      impl.write_transport()
          ? launch_bus_allreduce_write(
                v, options_.my_rank, static_cast<const __nv_bfloat16*>(device_src),
                static_cast<__nv_bfloat16*>(device_dst), static_cast<uint32_t>(bf16_elems),
                &impl.graph_variant_cells(variant)[shape.gens],
                impl.write_recs_of(variant, shape.gens), impl.graph.deadline_cycles,
                capture_stream)
          : launch_bus_allreduce_graph(
                v, options_.my_rank, static_cast<const __nv_bfloat16*>(device_src),
                static_cast<__nv_bfloat16*>(device_dst), static_cast<uint32_t>(bf16_elems),
                &impl.graph_variant_cells(variant)[shape.gens], impl.graph.deadline_cycles,
                capture_stream);
  if (launch != cudaSuccess) {
    *error = std::string("graph collective kernel launch failed: ") +
             cudaGetErrorString(launch);
    return false;
  }
  shape.meta.push_back(
      Impl::GraphState::GenMeta{static_cast<uint32_t>(bf16_elems)});
  ++shape.gens;
  return true;
}

bool CollectiveBus::graph_record_end(std::string* error) {
  if (impl_->world1) {
    Impl& w = *impl_;
    std::lock_guard<std::mutex> lock(w.w1_mu);
    if (!w.w1_recording) { *error = "graph_record_end without a session"; return false; }
    w.w1_recording = false;
    return true;
  }
  Impl& impl = *impl_;
  std::lock_guard<std::mutex> lock(impl.coll_mu);
  if (!impl.graph.recording) {
    *error = "no open graph session";
    return false;
  }
  const int variant = impl.graph.recording_variant;
  if (variant < 0 || variant >= kBusMaxGraphVariants) {
    impl.graph.recording = false;
    impl.graph.recording_variant = -1;
    *error = "open graph session has an invalid variant";
    return false;
  }
  Impl::GraphState::Variant& shape = impl.graph.variants[variant];
  if (shape.gens == 0) {
    impl.graph.recording = false;
    impl.graph.recording_variant = -1;
    if (!impl.graph.recorded.load(std::memory_order_relaxed))
      impl.coll_mode.store(false, std::memory_order_relaxed);
    *error = "no collectives recorded (call allreduce_record at least once)";
    return false;
  }
  impl.graph.recording = false;
  impl.graph.recording_variant = -1;
  // Publish the era: the engine reads gens/meta only after an arm, which
  // is happens-after this store through the same thread's window_count
  // release. A failed end (empty session) reopens the bus above.
  shape.recorded = true;
  impl.graph.recorded.store(true, std::memory_order_relaxed);
  DGPP_LOG_INFO("bus graph era: variant {} recorded with {} collective node(s)",
                variant, shape.gens);
  return true;
}

void CollectiveBus::dump_graph_cells(const char* why) {
  if (impl_->world1) {
    DGPP_LOG_INFO("bus graph cells ({}): a world of one has none", why);
    return;
  }
  Impl& impl = *impl_;
  // The adopted window's variant (the one being walked), else the latest
  // armed one.
  const int variant = impl.graph.adopted_count > 0
                          ? impl.graph.adopted_variant
                          : impl.graph.window_variant.load(
                                std::memory_order_relaxed);
  if (variant < 0 || variant >= kBusMaxGraphVariants ||
      !impl.graph.variants[variant].recorded ||
      impl.graph_cells == nullptr)
    return;
  const int gens = impl.graph.variants[variant].gens;
  BusAllReduceCtl* const variant_cells =
      impl.graph_variant_cells(variant);
  std::string cells;
  for (int g = 0; g < gens; ++g) {
    const BusAllReduceCtl& c = variant_cells[g];
    cells += " [" + std::to_string(g) + "]g=" +
             std::to_string(acquire_u64(&c.gen_seq)) + ",r=" +
             std::to_string(acquire_u64(&c.ready_bits)) + ",d=" +
             std::to_string(acquire_u64(&c.done_seq)) + ",s=" +
             std::to_string(acquire_u32(&c.status));
  }
  std::string ring;
  const uint64_t count = impl.graph.window_count.load(std::memory_order_relaxed);
  for (uint64_t w = count > kBusWindowRing ? count - kBusWindowRing + 1 : 1; w <= count; ++w) {
    const Impl::GraphState::Window& win = impl.graph.windows[(w - 1) % kBusWindowRing];
    ring += " w" + std::to_string(w) + "(v" + std::to_string(win.variant) + " first " +
            std::to_string(win.first) + " gens " + std::to_string(win.gens) + ")";
  }
  DGPP_LOG_INFO("graph cells ({}, variant {}): walk={} window={} adopted={} finished={} live={} "
                "adopted_first={} adopted_last={} flight_gen={}{};{}",
                why, variant,
                impl.graph.walk_pub.load(std::memory_order_relaxed), count,
                impl.graph.adopted_count, impl.graph.finished_count,
                impl.graph.live_windows, impl.graph.adopted_first,
                impl.graph.adopted_last,
                impl.graph.flight.active ? impl.graph.flight.gen : 0, ring, cells);
}

void CollectiveBus::quiesce() {
  Impl& impl = *impl_;
  std::lock_guard<std::mutex> lock(impl.stop_mu);
  if (impl.quiesced || impl.stopped) return;
  impl.quiesced = true;
  if (impl.world1) {
    impl.stopping.store(true, std::memory_order_relaxed);
    return;
  }

  impl.stopping.store(true, std::memory_order_relaxed);
  if (impl.engine.joinable()) impl.engine.join();

  // A still-running collective kernel must exit before the stream sync:
  // the engine already failed its request (watchdog or lane failure) and
  // poisoned the cell, or it completed normally — either way this stamp
  // is idempotent and merely hastens the exit.
  if (impl.coll.req) impl.poison_collective(*impl.coll.req);
  impl.poison_stream_collectives();
  impl.poison_live_graph_window();
  if (impl.collective_stream) {
    cudaStreamSynchronize(impl.collective_stream);
    cudaStreamDestroy(impl.collective_stream);
    impl.collective_stream = nullptr;
  }

  // Orderly consumer stop: reserved sequence in every control cell.
  for (auto& peer_lanes : impl.peers) {
    for (auto& lane : peer_lanes) {
      if (!lane.lane) continue;
      StartSlot* control =
          lane.lane->layout().control_cell(lane.lane->slab());
      __atomic_store_n(&control->seq, kFlagStopSequence, __ATOMIC_RELEASE);
    }
  }
  for (cudaStream_t stream : impl.consumer_streams) {
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }
  impl.consumer_streams.clear();
  impl.consumers_launched = false;
}

void CollectiveBus::stop() {
  Impl& impl = *impl_;
  std::lock_guard<std::mutex> lock(impl.stop_mu);
  if (impl.stopped) return;
  impl.stopped = true;
  if (impl.world1) {
    impl.quiesced = true;
    impl.stopping.store(true, std::memory_order_relaxed);
    if (impl.w1_stage) {
      cudaFreeHost(impl.w1_stage);
      impl.w1_stage = nullptr;
    }
    return;
  }

  // Inline quiesce (we already hold stop_mu).
  if (!impl.quiesced) {
    impl.quiesced = true;
    impl.stopping.store(true, std::memory_order_relaxed);
    if (impl.engine.joinable()) impl.engine.join();

    if (impl.coll.req) impl.poison_collective(*impl.coll.req);
    impl.poison_live_graph_window();
    if (impl.collective_stream) {
      cudaStreamSynchronize(impl.collective_stream);
      cudaStreamDestroy(impl.collective_stream);
      impl.collective_stream = nullptr;
    }

    for (auto& peer_lanes : impl.peers) {
      for (auto& lane : peer_lanes) {
        if (!lane.lane) continue;
        StartSlot* control =
            lane.lane->layout().control_cell(lane.lane->slab());
        __atomic_store_n(&control->seq, kFlagStopSequence, __ATOMIC_RELEASE);
      }
    }
    for (cudaStream_t stream : impl.consumer_streams) {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
    impl.consumer_streams.clear();
    impl.consumers_launched = false;
  }

  // Waiters must never hang: fail and release anything still registered.
  std::vector<std::shared_ptr<BusRequest>> remaining;
  {
    std::lock_guard<std::mutex> rlock(impl.registry_mu);
    for (auto& entry : impl.registry) remaining.push_back(entry.second);
    impl.registry.clear();
  }
  for (auto& req : remaining)
    impl.complete_request(req, false, "bus stopped");

  // Lane teardown (QP -> CQ -> MR -> slab) and device close. All consumer
  // kernels must be dead by here: cudaFreeHost synchronizes the device
  // implicitly and would otherwise wait out a live peer's consumers. The
  // graph-era contract adds the caller's own stream: the caller must have
  // drained its replay stream before stop() — the per-gen cells are baked
  // into its recorded kernels.
  if (impl.ar_ctl) {
    cudaFreeHost(impl.ar_ctl);
    impl.ar_ctl = nullptr;
  }
  if (impl.graph_cells) {
    cudaFreeHost(impl.graph_cells);
    impl.graph_cells = nullptr;
  }
  if (impl.strm.cells) {
    cudaFreeHost(impl.strm.cells);
    impl.strm.cells = nullptr;
  }
  if (impl.bulk_scratch) {
    cudaFree(impl.bulk_scratch);
    impl.bulk_scratch = nullptr;
  }
  if (impl.write_recs) {
    cudaFreeHost(impl.write_recs);
    impl.write_recs = nullptr;
  }
  // Staging block: deregister before the lanes/devices go away (the MRs
  // hang off the device PDs), then free the pinned block.
  for (ibv_mr* mr : impl.stage_mrs)
    if (mr != nullptr) ibv_dereg_mr(mr);
  impl.stage_mrs.clear();
  if (impl.stage_block != nullptr) {
    cudaFreeHost(impl.stage_block);
    impl.stage_block = nullptr;
  }
  impl.stage_held_ptr = nullptr;
  impl.peers.clear();
  impl.devices.clear();
}

double CollectiveBus::bulk_pace_gbps() const {
  return impl_->opt.bulk_pace_gbps < 0 ? 0.0 : impl_->opt.bulk_pace_gbps;
}

const std::atomic<uint64_t>& CollectiveBus::completion_epoch() const {
  return impl_->completion_epoch;
}

BusStats CollectiveBus::stats() const {
  Impl& impl = *impl_;
  std::lock_guard<std::mutex> lock(impl.stats_mu);
  BusStats out = impl.stats_store;
  for (auto& peer_lanes : impl.peers)
    for (auto& lane : peer_lanes) out.lanes.push_back(lane.stats);
  return out;
}

BusRecvView CollectiveBus::recv_view(int peer_rank, int lane) const {
  Impl& impl = *impl_;
  const auto it =
      std::find(impl.peer_ranks.begin(), impl.peer_ranks.end(), peer_rank);
  if (it == impl.peer_ranks.end() ||
      lane < 0 || lane >= static_cast<int>(impl.lane_count()))
    return {};
  return impl.recv_view_of(
      impl.peers[static_cast<size_t>(it - impl.peer_ranks.begin())]
                [static_cast<size_t>(lane)]);
}

}  // namespace dgpp::net
