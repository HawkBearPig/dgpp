#pragma once

// Receive-side consumer for CollectiveBus validation (DESIGN §6/§6.2).
//
// Phase 2's transport harness: one persistent kernel per bus endpoint that
// watches every doorbell cell, folds the arrived payload with the golden-
// ratio mix, and publishes the per-slot acknowledgement. The engine's real
// consumers (M5 deliverable 3) are per-collective kernels with the same
// doorbell/ack contract; the transport is agnostic to who consumes.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "kernels/flag_protocol_types.hpp"
#include "net/bus_types.hpp"

namespace dgpp::net {

// Raw pinned pointers into one lane's receive slab, shared between host and
// device. Built by CollectiveBus::recv_view(lane).
struct BusRecvView {
  const StartSlot* doorbell_lat = nullptr;  // [lat_slots]
  const StartSlot* doorbell_bulk = nullptr;  // [bulk_slots]
  const uint64_t* payload_lat = nullptr;    // slot-strided lat_slot_bytes
  const uint64_t* payload_bulk = nullptr;    // slot-strided bulk_slot_bytes
  FlagAck* ack_lat = nullptr;                // [lat_slots]
  FlagAck* ack_bulk = nullptr;               // [bulk_slots]
  const StartSlot* control = nullptr;       // orderly stop cell
  int lat_slots = 0;
  int bulk_slots = 0;
  uint32_t lat_slot_bytes = 0;
  uint32_t bulk_slot_bytes = 0;
  // The write ring this peer's collective rows land in (bus_types.hpp):
  // kBusWriteRing rows of write_row_bytes.
  const uint8_t* write_ring = nullptr;
  uint32_t write_row_bytes = 0;
};

// Shared-memory cell budget for the consumer's per-slot last-seen table.
constexpr int kBusMaxConsumerCells = 256;

// Launches the persistent consumer on `stream` (1 block x 256 threads).
// The kernel exits on the control stop sequence or after `deadline_cycles`
// without traffic; the caller still syncs the stream.
cudaError_t launch_bus_consumer(const BusRecvView& view,
                                uint64_t deadline_cycles, cudaStream_t stream);

// Watchdog deadline in device cycles for the given inactivity seconds.
uint64_t bus_consumer_deadline_cycles(double seconds);

// Loads every bus kernel's module up front. CUDA 12+ loads modules LAZILY
// (CUDA_MODULE_LOADING=LAZY): a kernel's first launch in the process loads
// it, and that load waits for the device to go idle. A collective kernel
// spinning on a doorbell is not idle — so in a one-process multi-rank
// world, rank B's first launch of the very kernel rank A is already
// spinning in waits for A, and A waits for B: the 2026-09-02 seq-1 wedge
// (1 start in ~10; a 40-stream probe showed 0/40 kernels starting beside a
// spinner under LAZY, 40/40 under EAGER). The bus never relies on the
// environment for this: start() preloads. Callers with their OWN kernels
// launched beside a live collective (the loopback tests' forward passes)
// still need CUDA_MODULE_LOADING=EAGER or their own preload.
cudaError_t bus_preload_kernels();

// %globaltimer's offset from CLOCK_MONOTONIC (gt = mono + offset), with
// the bracket half-width as its uncertainty — the bridge between the
// engine's host stamps and the kernels' gt_* stamps (see the .cu).
cudaError_t bus_globaltimer_offset(int64_t* offset_ns,
                                   uint64_t* uncertainty_ns);

// Host mirror of the consumer fold (bus_types.hpp's mix); the sender uses
// it to verify the receiver's hash. `bytes` must be a multiple of four.
// A final u32 is zero-extended rather than omitted from the placement proof.
uint64_t bus_fold(const void* data, size_t bytes);

// ---- per-collective all-reduce consumer (DESIGN §6.3) ------------------------
//
// One kernel per collective call: stages the source vector into each peer's
// claimed send slot, signals the engine to post, waits for every peer's
// doorbell, folds bf16 with fp32 accumulation (canonical global-rank order
// — every rank's result is bitwise identical), publishes the standard
// per-slot acks, and stamps the control cell. The engine's recycle/credit
// passes treat these messages exactly like harness traffic; the wire
// protocol is unchanged.

// Pinned 64B handoff cell, one per in-flight eager collective (the bus's
// single ar_ctl) and one per recorded graph generation (the per-gen cell
// array, §6.2 graph mode). Published fields, in order:
//   ready_bits — bit p set (release) once peer p's send slot is staged;
//   done_seq   — the collective's ctl_seq (release) once the kernel is
//                finished (status 0) or gave up (status 1). The engine
//                may also write it as a poison to hasten a failed
//                kernel's exit; the kernel treats any match as an exit.
//   gen_seq    — graph mode only: this execution's generation, written
//                (release) by the arm step AFTER the cell is reset. The
//                replayed kernel acquires it at start and derives its
//                staging row and its exit match from it — the recorded
//                launch parameters stay replay-stable while the
//                generation advances with every replay. Monotonic, so a
//                stale done_seq from the previous replay never matches.
struct alignas(64) BusAllReduceCtl {
  uint64_t ready_bits = 0;
  uint64_t done_seq = 0;
  uint64_t gen_seq = 0;
  // TEMP instrumentation: kernel phase stamps (clock64), cycles 0 if unset
  uint64_t stamp_stage = 0;
  uint64_t stamp_first_claim = 0;
  uint64_t stamp_reduce_done = 0;
  uint32_t status = 0;  // 0 = reduced, 1 = deadline/poison exit
  // TEMP hunt instrumentation (small-collective corruption): the first
  // doorbell this kernel claimed — flat cell index, the doorbell's len
  // field, and its seq — to discriminate a stale-len read from a
  // receive-slot misalignment. 0/0/0 = no claim yet.
  uint32_t dbg_first_cell = 0;
  uint32_t dbg_first_len = 0;
  uint32_t dbg_first_seq = 0;
  // PLACEMENT-gate telemetry (the 2026-09-01 hunt's fix): how many claims
  // had to WAIT for the payload's DMA placement to become visible after
  // the doorbell did (the race the gate exists to close — any nonzero
  // count in a passing run is live proof the race was real), and the
  // total spin iterations those waits took.
  uint32_t dbg_gate_waits = 0;
  uint32_t dbg_gate_spins = 0;
  // Per-claim records (all rounds, not just the first): the kernel's
  // ACTUAL claimed cells with their doors' len/seq/hash-low — the
  // engine's stall dump reads these to print what the kernel really
  // claimed (the engine's own claim slots are send-side bookkeeping
  // and do not track the kernel's scan). One entry per round, in claim
  // order; 0 = no claim in that slot.
  uint32_t dbg_cl_cell[3] = {};
  uint32_t dbg_cl_len[3] = {};
  uint32_t dbg_cl_seq[3] = {};
  uint32_t dbg_cl_hash[3] = {};  // the claimed door's hash, low 32 bits
  // Phase timeline in %globaltimer ns (constant-rate, unlike clock64 under
  // a ramping SM clock): kernel entry, staging rows written, first peer
  // claimed (payload gated), last peer claimed, fold done. The graph walk
  // folds these into the per-window "graph window timeline" summary —
  // the collective's cost decomposed into copy / handshake+skew / fold.
  uint64_t gt_start = 0;
  uint64_t gt_stage = 0;
  uint64_t gt_first = 0;
  uint64_t gt_last = 0;
  uint64_t gt_done = 0;
  uint64_t gt_claim[kBusMaxPeers] = {};  // per peer (peer index), gated
  // Graph/stream snapshot's placement hash, published before ready_bits.
  // Fits the final eight bytes of this cell's first three cache lines.
  uint64_t stage_hash = 0;
  // The write transport's landing stamps (2026-10-19): block 0's first
  // proof of a chunk of peer p (its row has landed) — the handshake splits
  // into staged -> landed (the sender, the fabric) and landed -> proved.
  uint64_t gt_landed[kBusMaxPeers] = {};
  // The first block to finish peer p (bitwise negated, the engine's fill
  // from the block records): the spread to gt_claim[p], the last block,
  // separates a straggler from uniformly slow proving.
  uint64_t gt_claim_first_neg[kBusMaxPeers] = {};
  uint64_t reserved_[2] = {};
};
static_assert(sizeof(BusAllReduceCtl) == 256,
              "BusAllReduceCtl occupies exactly four cache lines");

// Everything the kernel needs, built by the engine at claim time.
struct BusAllReduceView {
  // Every lane of every peer, peer-major (peer ascending, lanes inner).
  BusRecvView recv[kBusMaxPeers * kBusMaxLanes] = {};
  int recv_views = 0;       // peers * lanes
  int lanes_per_peer = 0;
  // Claimed outbound latency slot per peer (the engine posts it once the
  // ready bit lands). bf16 element storage.
  const uint16_t* send_payload[kBusMaxPeers] = {};
  int send_peers = 0;
};

// `elems` is the bf16 element count (multiple of 2). The kernel folds all
// W vectors into dst with fp32 accumulation in canonical global-rank
// order, so every rank's destination is bitwise identical and a host
// oracle of the same chain matches exactly. src may be device memory or a
// pinned pre-stage buffer the producing GEMM wrote (the §6.3 interface); when
// src aliases dst, the fold runs in place per element (one thread per
// element, read before write) after snapshotting every peer's send copy.
cudaError_t launch_bus_allreduce(const BusAllReduceView& v, int my_rank,
                                   const __nv_bfloat16* src, __nv_bfloat16* dst,
                                   uint32_t elems, uint32_t ctl_seq,
                                   BusAllReduceCtl* ctl,
                                   uint64_t deadline_cycles,
                                   cudaStream_t stream);

// ---- graph-captured all-reduce (§6.2 decode path) ----------------------------
//
// The decode step is a fixed launch sequence, so the whole step — GEMMs and
// collectives — records once into a CUDA graph and replays per token. The
// collective kernel is a graph node like any other; the engine no longer
// launches it and instead REACTS: per-generation cells (one per recorded
// collective) carry the kernel->engine handoff, and the engine walks
// generations the replay produces (§6.3 graph mode).
//
// What may be baked at record time and what may not:
//   stable — the recv views, the staging ROW BASES (the block is pinned
//            for the bus's lifetime), src/dst device buffers, elems, the
//            cell pointer, the deadline;
//   per-execution — the generation. The kernel reads cell->gen_seq at
//            start (the arm step wrote it, release) and derives its
//            staging row (gen-1)%kBusMaxGraphStageRing and its exit
//            match from it. Monotonic generations make replays of the
//            same node distinct.
//
// The engine's posting side (walk): the kernel stages peer rows and
// releases ready_bits; the engine posts each peer's pair from the row the
// generation selects, at the lane-0 cursor ring position (the position is
// deterministic because every latency post — graph generation or eager
// collective — goes through the same cursor in generation order; harness
// sends alone are closed for the era). done_seq == gen ends the
// generation; window completion gates the next arm.
// 256 covers every family's decode step with headroom: GLM-5.3-Flash's 45
// layers x 2 boundary folds = 90 collective nodes, Qwen3.8-Flash-Next's 99,
// GLM-4.7's 92 layers x 2 + the draft layer's 2 = 186 (2026-09-10; the
// 128 budget refused its second graph variant), plus margin for nodes a
// future era might record beside them. Sizes the pinned generation cells
// (kBusMaxGraphVariants slabs of this many) only.
constexpr int kBusMaxGraphGens = 256;       // recorded nodes per graph
// A serving era can register several mutually exclusive graph shapes and
// select one per replay (for example one scalar graph per request slot plus
// the full row batch). Each variant owns a disjoint generation-cell set;
// windows still execute strictly one at a time.
constexpr int kBusMaxGraphVariants = 256;  // two per slot and two per batch family, per verify depth (64 from 2026-09-14: six slots with four families at three scheduled depths need 60; 128 since 2026-10-02: the same recipe at all five depths of a DSpark block needs 100 — the two depths the budget dropped were 1.5 % of a sampled stream; 256 since 2026-10-23: sixteen DSpark slots with eight families kept only two of the three scheduled depths at 128 — code and prose paid a fifth row at C1, +2 ms a step — and need 150 for three, 250 for all five)
// Replay windows armed at once (2026-09-06, the pipelined replay): the
// engine arms the next replay's window while the previous replay is still
// running, on a DIFFERENT variant (a variant's cells are in flight until
// its window is finished). The ring holds the last arms; arm waits for the
// window kBusMaxLiveWindows back to be walked before reusing its entry.
constexpr int kBusMaxLiveWindows = 2;
constexpr int kBusWindowRing = 4;
constexpr int kBusMaxGraphStageRing = 8;    // staging rows (kStageRing)
// The write kernel indexes its staging row and the peers' ring rows by the
// same (gen - 1) % kBusWriteRing — the two rings move together.
static_assert(static_cast<int>(kBusWriteRing) == kBusMaxGraphStageRing,
              "the write ring and the staging ring index by one rotation");
// The stream collectives' cell ring (2026-09-14, plan D9: the eager fold
// launched on the model's stream in the graph kernel form): generations in
// flight between the forward thread's issue and the engine's walk — a
// prefill pass issues ~93 folds, the ring lets the host run this far ahead.
constexpr int kBusStreamRing = 64;

// The graph twin of BusAllReduceView: the staging row is a base, not a
// precomputed row — the kernel picks the row at runtime from its
// generation. `stage_row_base` is the SHARED send row 0 (one snapshot
// serves every peer's post: a SEND's local address is any registered
// memory, and the ring-depth reuse argument holds per peer exactly as it
// did with a row per peer); rows are stage_row_bytes apart, stage_ring
// deep.
struct BusAllReduceGraphView {
  BusRecvView recv[kBusMaxPeers * kBusMaxLanes] = {};  // peer-major
  int recv_views = 0;
  int lanes_per_peer = 0;
  uint16_t* stage_row_base = nullptr;
  int send_peers = 0;
  uint32_t stage_ring = 0;      // rows per peer (kBusMaxGraphStageRing)
  uint32_t stage_row_bytes = 0; // == lat_slot_bytes
};

// Records one collective node onto `stream` (the caller's capturing
// stream). The same fold as the eager one-shot (canonical rank order),
// so a graph replay is bitwise the eager result. `cell` is this node's
// per-generation cell; it must stay allocated for the bus's lifetime.
cudaError_t launch_bus_allreduce_graph(const BusAllReduceGraphView& v,
                                       int my_rank, const __nv_bfloat16* src,
                                       __nv_bfloat16* dst, uint32_t elems,
                                       BusAllReduceCtl* cell,
                                       uint64_t deadline_cycles,
                                       cudaStream_t stream);

// ---- the one-sided latency transport (2026-10-19) ---------------------------
//
// The graph node's twin over the write ring (bus_types.hpp): a grid of
// bus_write_blocks(chunks) blocks whose warps each own chunks of the row.
// Phase 1 lays the source out as self-describing chunks in the
// generation's staging row (the engine's RDMA WRITE source); each block
// fences once and sets its own bit in ready_bits — the engine posts when
// every block's bit is up. Phase 2: every warp polls its own chunks, all
// peers' copies in flight at once (the poll is the data read), proves
// each by its trailer, folds in canonical rank order and writes the
// result. No door cells, no claim rounds, no acks, no credits, and no
// global coordination at all: beside the L2 prefetch window a dependent
// memory round trip costs 1-8 us, and one SM cannot keep enough system
// loads in flight at that latency, so the parallelism is the grid, the
// coordination is shared memory, and each block reports to the engine
// through its own pinned record. The chain per element is block_fold_vectors' (bf16
// in, fp32 accumulation in rank order, one rounding), so the result is
// bitwise the eager machine's. The cell's stamps keep the timeline's
// meaning: gt_stage = the last block's row staged, gt_first/gt_last = the
// first/last peer whose every chunk proved, gt_done = the fold written.

// `recs` is this node's pinned per-block record array (kBusWriteMaxBlocks
// entries): block b stages its chunks, fences and stores ready = gen; it
// proves and folds, fences and stores done = gen with its status and
// stamps. The engine posts when every block's ready carries the
// generation and completes the generation when every block's done does
// (it fills the cell's stamps from the records) — no atomics, no last
// block, nothing a block waits on but the peers' rows.
struct alignas(64) BusWriteBlockRecord {
  uint32_t ready = 0;   // the generation whose row this block staged
  uint32_t done = 0;    // the generation this block finished
  uint32_t status = 0;  // 0 = folded, 1 = deadline/poison exit
  uint32_t pad = 0;
  uint64_t gt_stage = 0;
  uint64_t gt_done = 0;
  uint64_t gt_peer[kBusMaxPeers] = {};  // this block's last chunk of peer p proved
  uint64_t pad2 = 0;
};
static_assert(sizeof(BusWriteBlockRecord) == 64, "one line");

cudaError_t launch_bus_allreduce_write(const BusAllReduceGraphView& v,
                                       int my_rank, const __nv_bfloat16* src,
                                       __nv_bfloat16* dst, uint32_t elems,
                                       BusAllReduceCtl* cell,
                                       BusWriteBlockRecord* recs,
                                       uint64_t deadline_cycles,
                                       cudaStream_t stream);

// The host mirror of a chunk's trailer hash: over the 31 data lanes' 496
// bytes (as laid out in the row) and the (generation, chunk) pair — the
// tests pin the row format with it.
uint64_t bus_write_chunk_hash(uint32_t gen, uint32_t chunk, const void* lanes496);

// ---- segment-quantized reduce-scatter/allgather (the prefill class) ------
//
// DESIGN §6.3: large boundaries flow as reduce-scatter + allgather over
// the bulk pool (one-shot's (W-1)x replication is wire-optimal only at
// W=2; RS+AG pays 2(W-1)/W). The buffer is striped on the bulk-slot grid
// (stripe = one bulk slot); shards are contiguous stripe ranges per rank
// (ceil split); the machine processes SEGMENTS of at most
// bulk_slots x lanes stripes so one kernel launch and one flight step
// cover a bounded staging/posting/claiming wave within the pool depths.
//
// Both phases share one sub-range table — each rank's owned stripes of
// the current segment: RS stages every peer's sub-range TO that peer
// (from my src) and folds MY sub-range from all peers' arrivals;
// AG broadcasts MY (now reduced) sub-range to every peer and copies each
// peer's arrivals into dst. The RS fold is the canonical ascending-rank
// fp32 chain per element — bitwise identical to the latency one-shot
// chain, so both paths against one oracle must agree exactly.
//
// Arrival identification (no engine coupling): the sender posts a
// sub-range's stripes in order, striped round-robin over lanes (stripe k
// on lane k%lanes, ring-FIFO per lane); RC delivers in order per lane, so
// the j-th arrival on a lane sits in ring slot j%depth with
// seq = j/depth + 1 — the kernel derives j from the doorbell cell itself
// and maps it to stripe k = i*lanes + lane (i = the in-segment per-lane
// index, from the per-(peer,lane) base the kernel computes by summing
// its OWN ack cells at launch; its acks are the only writer).
//
// Two invariants the mapping forces, both measured into existence:
//   * SEGMENT WINDOW: the rings are flight- and phase-agnostic FIFOs and
//     ranks do not progress in lockstep (shard geometry sees to that), so
//     a receiver's active kernel WILL see future-segment/future-phase
//     doorbells in its cells. It claims only its own window —
//     [base, base + the lane's stripe share) — and leaves the rest
//     unconsumed for the kernel whose window contains them; every post
//     maps to exactly one window of the receiving flight, in ring order,
//     at any skew. Eager claiming acks foreign doorbells away and
//     deadlocks both phases on the loss.
//   * DEFERRED ACKS: the ack certifies the payload CONSUMED — the consume
//     pass after the claims (the RS fold, the AG landing copy) reads the
//     slots, so the ack flushes after it. An ack at claim time returns
//     the sender's credit early and a sender a segment ahead overwrites a
//     fold input. The flush re-reads the door cell (stable until its
//     credit returns) and skips the kernel's own claims in the scan
//     meanwhile — a re-presented claim would steal the shared CAS from a
//     fresh doorbell in the same warp, forever.
//
// THE KERNEL IS A COOPERATIVE GRID: kBulkBlocks blocks,
// co-resident by the cooperative-launch contract, with grid barriers
// between staging | claims | consume rounds | acks. Everything that scales
// with bytes — the outbound staging, the RS fold, the AG landing — runs
// over the whole grid in 16 KB tiles of 16-byte vectors; the claim loop
// and the ack flush stay in block 0. The placement proof (the payload
// folds to its door's hash before any consumption counts) moved from the
// claim into the consume pass: every tile hashes the words it reads, the
// per-(peer, stripe) totals combine by atomic XOR in BusBulkScratch, and
// block 0 proves each stripe between rounds — a stripe read before its
// placement was fully visible is re-folded next round (the fold and the
// copy are pure functions of their inputs); no ack and no done stamp
// until every consumed stripe has proved. The single-block kernel before
// it moved every byte through 256 threads: a 2 MB fold cost 1.4 ms.
// The staging tiles also compute each outbound stripe's fold — the hash
// the door carries — and block 0 publishes the per-(peer, stripe) table
// (`staged_hashes`, pinned, after the staged counters) before releasing
// the counters; the engine posts from it instead of hashing 256 KB per
// stripe on the collective thread.
constexpr int kBusMaxBulkSegStripes = 64;   // >= bulk_slots * kBusMaxLanes
constexpr int kBusMaxPeersSized = 8;        // >= kBusMaxPeers
// Flat bulk receive cells the collective kernel can track (the claimed
// bitmap behind the deferred-ack flush): peers x lanes x bulk_slots must
// fit, or claimed cells stop acking (their credits never return) — the
// bus validates this at start, config-error class.
constexpr int kBusMaxBulkCells = 96;

// The bulk kernel's grid-shared records (device memory, one per bus; the
// kernel resets what it uses at entry). Block 0 writes the claim records
// during its claim loop; every block reads them after the claim barrier
// for the consume pass, and the pass's per-(peer, stripe) hash
// accumulators fold in through atomic XOR. Stripe indices are sub-range
// local: RS records index my sub-range, AG records the peer's.
constexpr int kBusBulkBlocks = 16;  // the bulk kernel's cooperative grid
struct BusBulkScratch {
  // Staging: each block's partial of every outbound stripe's fold (the
  // door's hash, computed where the bytes are read; block 0 reduces the
  // partials into the pinned table the engine posts from).
  uint64_t stage_part[kBusBulkBlocks][kBusMaxPeersSized][kBusMaxBulkSegStripes];
  const uint16_t* ptr[kBusMaxPeersSized][kBusMaxBulkSegStripes];  // payload
  uint64_t want[kBusMaxPeersSized][kBusMaxBulkSegStripes];  // the door's hash
  uint32_t len[kBusMaxPeersSized][kBusMaxBulkSegStripes];   // the door's len
  uint64_t hash_acc[kBusMaxPeersSized][kBusMaxBulkSegStripes];  // consumed fold
  uint64_t ready[kBusMaxBulkSegStripes];  // RS: arrived-peer mask per stripe
  uint64_t agot[kBusMaxPeersSized];       // AG: landed-stripe mask per peer
  uint64_t pending;                       // RS: stripes still to fold/prove
  uint64_t pending_ag[kBusMaxPeersSized]; // AG: per peer
  int failed;
  int total_arrived;
};

// Per-segment launch plan; `send_payload[p]` in the view is peer p's
// arena row (segment stripes, stripe k at row + k*bulk_slot_bytes).
struct BusBulkSegPlan {
  uint32_t total_elems = 0;      // whole collective (tails the last stripe)
  uint32_t stripe_elems = 0;     // bulk_slot_bytes / 2
  uint32_t seg_first = 0;       // first stripe of the segment (global grid)
  uint32_t seg_stripe_count = 0;  // stripes in this segment
  uint32_t my_base = 0;         // my sub-range, segment-local stripe index
  uint32_t my_count = 0;
  uint32_t out_base[kBusMaxPeersSized] = {};   // peer p's sub-range
  uint32_t out_count[kBusMaxPeersSized] = {};
};

// phase 0 = reduce-scatter (fold my sub-range from all peers' arrivals),
// phase 1 = allgather (copy each peer's sub-range arrivals into dst).
cudaError_t launch_bus_bulk_collective(const BusAllReduceView& v, int my_rank,
                                       int phase, const __nv_bfloat16* src,
                                       __nv_bfloat16* dst,
                                       const BusBulkSegPlan& plan,
                                       uint64_t* staged_counters,
                                       uint64_t* staged_hashes,
                                       uint32_t ctl_seq, BusAllReduceCtl* ctl,
                                       uint64_t deadline_cycles,
                                       BusBulkScratch* scratch,
                                       cudaStream_t stream);

// Host-side bf16 helpers matching the device intrinsics' round-to-nearest-
// even. Consumers of the bus (tests, checks, the forward oracle) verify
// against the kernel's exact arithmetic; guessing at the rounding would
// make every comparison tolerance-based for no reason.
inline uint16_t bf16_from_f32_rne(float f) {
  uint32_t x = 0;
  std::memcpy(&x, &f, 4);
  const uint32_t lsb = (x >> 16) & 1;  // ties toward even
  return static_cast<uint16_t>((x + 0x7FFFu + lsb) >> 16);
}

inline float bf16_to_f32(uint16_t h) {
  const uint32_t x = static_cast<uint32_t>(h) << 16;
  float f = 0.0f;
  std::memcpy(&f, &x, 4);
  return f;
}

// Benchmark realism (bus_check): inter-collective compute standing in for
// the decode GEMMs. A bare submit/wait loop is ~1% device duty and
// measures GPU wake latency instead of the collective; real decode keeps
// the clocks boosted between collectives. `spins` iterations of dependent
// FLOPs per thread; the result lands in buf[256].
cudaError_t launch_bus_warm_work(cudaStream_t stream, float* buf, int spins);

}  // namespace dgpp::net
