#pragma once

// CollectiveBus: the M5 data plane (DESIGN §6). One bus per process owns
// every peer pair's RC QPs on every configured lane, the single-threaded
// engine loop that polls them (§6.1), per-class slot pools, and the
// watchdog-bounded receive consumers.
//
// Concurrency contract (§6.1): the engine thread owns all verbs objects,
// slabs, and slot/credit state. Other threads touch the bus only through
// send()/wait()/stop() — submissions go to per-class queues drained in
// priority order with bounded bulk posts, so concurrent latency-under-bulk
// traffic is a supported, tested mode. The class slot pools make the credit
// floors structural: bulk can never occupy a latency slot.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "net/bus_kernel.hpp"
#include "net/site_options.hpp"

namespace dgpp::net {

enum class BusMessageClass { kLatency, kBulk };

// The latency class's collective transport for the graph and stream
// collectives (the decode step's folds and the pick's), 2026-10-19:
//   kSend  — payload + doorbell SEND pairs into the peer's receive ring,
//            claimed through door cells by a single-block kernel, hash-
//            gated, acked and credited (the form until 2026-10-19);
//   kWrite — the row laid out as self-describing chunks and RDMA-WRITTEN
//            into the peer's write ring, folded by a grid of warps as the
//            chunks prove themselves (bus_types.hpp, "The write ring").
// The eager one-shot (allreduce/allreduce_staged) and the harness keep the
// send machinery either way; the bulk class is separate.
enum class BusLatencyTransport { kSend, kWrite };

struct BusOptions {
  int world_size = 2;  // TP group size; ranks are 0..world_size-1
  int my_rank = 0;
  // Lane devices in lane order; configured locally or discovered from sysfs.
  // Lane counts and order must match on every rank.
  std::vector<std::string> lane_devices = configured_lane_devices();

  // Rendezvous: rank 0 listens on `rendezvous_port`; every other rank
  // connects to `rendezvous_host:rendezvous_port` (roster precedent: the
  // coordinator owns admission, then distributes the full endpoint table).
  uint16_t rendezvous_port = 29600;
  std::string rendezvous_host;  // required when my_rank != 0
  int rendezvous_timeout_ms = 20000;

  // Per-lane slot pools. Latency pool: decode-class messages, one slot per
  // message, no striping. Bulk pool: striped round-robin across lanes.
  int lat_slots = 32;
  size_t lat_slot_bytes = 8192;  // 4096 hidden x BF16, the decode unit
  int bulk_slots = 16;
  size_t bulk_slot_bytes = 262144;
  // Bulk collective pacing: at most this many stripes in
  // flight per (peer, lane) — the ring depth otherwise. The multi-block
  // fold made the senders burst whole rings at three peers at once, and
  // the fabric answered with sequence errors, adaptive retransmissions
  // and congestion notifications (a 16 MiB all-reduce bimodal at 2.4 or
  // 30-56 ms). The window bounds every receiver's inbound burst to
  // peers x lanes x this x slot bytes; the posting order also rotates per
  // sender (rank + 1 first) so aligned bursts spread across receivers.
  int bulk_inflight_per_lane = 4;
  // Software pacing per (peer, lane) queue pair, in Gb/s. A receiver has
  // (world - 1) x lanes inbound QPs on one port, and the NICs' packet
  // pacing covers raw-packet QPs only, so the engine spaces a QP's stripe
  // posts by len / rate. Negative (the default) DERIVES the rate at start
  // from the slowest lane's port: port rate / ((world - 1) x lanes) x
  // 0.85, so every inbound QP at the cap stays under the port even when
  // every sender bursts at the same receiver (the ack-driven credit
  // return aligns them: the receiver that finishes a segment releases
  // all three at once) — 28.3 Gb/s on the four-node 200 Gb/s fabric,
  // where 15/28/40 all ran clean and unpaced lost packets. 0 = unpaced;
  // positive = that rate. bulk_pace_gbps() reports the resolved value.
  double bulk_pace_gbps = -1.0;

  // Fault injection for the tests: the one-shot collective
  // pass sleeps this long between its two control-cell reads (the done
  // stamp, then the ready bits), widening the window in which a kernel
  // whose peers already posted can finish before its own engine has
  // posted. Production leaves it 0.
  int debug_pass_delay_us = 0;

  BusLatencyTransport latency_transport = BusLatencyTransport::kSend;

  int qp_depth = 1024;
  int completion_timeout_ms = 5000;  // engine watchdog per request
  double consumer_deadline_s = 30.0;  // receive-kernel inactivity bound
  bool launch_consumers = true;       // Phase 2 harness consumers; the
                                      // engine's real per-collective
                                      // consumers (deliverable 3) replace
                                      // them by disabling this.
};

// Per-lane traffic counters (one entry per peer x lane, peers ascending).
struct BusLaneStats {
  int peer_rank = -1;
  int lane = -1;
  uint64_t posts = 0;             // payload+doorbell pairs posted
  uint64_t bytes_sent = 0;
  uint64_t doorbell_recvs = 0;     // inbound messages arrived
  uint64_t bytes_recv = 0;
  uint64_t credits_returned = 0;   // slots recycled + credit WRITEs posted
  uint64_t credits_received = 0;   // inbound credits observed
  bool failed = false;
};

struct BusClassStats {
  uint64_t messages = 0;
  uint64_t bytes = 0;
  // Completion latency in microseconds, submit-to-credit, capped sample.
  std::vector<double> latency_us;
};

struct BusStats {
  std::vector<BusLaneStats> lanes;
  BusClassStats latency;
  BusClassStats bulk;
  // The bulk collective kernel's placement-proof telemetry:
  // segment kernels completed, and stripes the consume pass had to
  // re-fold because the payload's DMA placement was not yet fully visible
  // when the fold first read it (any nonzero count in a passing run is
  // live proof the race is real; a large one is a wire-side stall).
  uint64_t bulk_segments = 0;
  uint64_t bulk_gate_redos = 0;
};

struct BusSendResult {
  bool ok = false;
  std::string error;
  // Per-stripe consumer hashes, stripe order. The sender can verify each
  // against fold() of the corresponding contiguous chunk (latency: one
  // stripe = whole message; bulk: slot-byte chunks striped round-robin).
  std::vector<uint64_t> stripe_hashes;
  double elapsed_us = 0.0;
};

struct BusAllReduceResult {
  bool ok = false;
  std::string error;
  double elapsed_us = 0.0;
};

class CollectiveBus {
 public:
  explicit CollectiveBus(BusOptions options);
  ~CollectiveBus();

  CollectiveBus(const CollectiveBus&) = delete;
  CollectiveBus& operator=(const CollectiveBus&) = delete;

  // Rendezvous (listener = rank 0), QP bring-up, initial credits posted,
  // receive consumers launched (unless disabled), engine thread started.
  // Returns false with *error on any failure; the bus is then stopped.
  bool start(std::string* error);

  // Submits a message to one peer. `data` must stay valid until wait()
  // returns. Latency class: bytes <= lat_slot_bytes. Bulk class: any size,
  // striped in bulk_slot_bytes chunks. bytes must be a multiple of 8.
  // Returns an id for wait(), or 0 with *error set.
  uint64_t send(int peer_rank, const void* data, size_t bytes,
                BusMessageClass cls, std::string* error);

  // Blocks for the request's completion. The engine enforces the watchdog;
  // timeout_ms is a generous backstop. Removes the request record.
  BusSendResult wait(uint64_t send_id, int timeout_ms);

  // One-shot all-to-all all-reduce over the whole world (DESIGN §6.3),
  // latency pool. `device_src`/`device_dst` are device pointers to
  // `bf16_elems` bf16 elements (multiple of 2, at most lat_slot_bytes/2);
  // they may alias. bf16 loads, fp32 accumulation in rank order —
  // deterministic, so a host oracle of the same chain matches bitwise.
  //
  // Contract (v1): launch_consumers=false (the per-collective kernel owns
  // doorbell claims), at most one outstanding collective (decode is
  // dependency-serialized), and no other latency traffic in flight. After
  // any collective failure the bus rejects further collectives — inbound
  // generations can no longer be trusted to line up.
  uint64_t allreduce(const void* device_src, void* device_dst,
                     size_t bf16_elems, std::string* error);

  // The staging interface (§6.3 evolution, M5 d3): hands out the pinned,
  // device-writable buffer the next latency collective will fold from, so
  // the producing GEMM writes the boundary partial directly into it —
  // the kernel's device→slot staging pass becomes a pinned fan-out (the
  // device round trip disappears), and the fold runs in place. One
  // handout at a time, none while a collective is in flight; the buffer
  // holds lat_slot_bytes/2 bf16. The handout must be consumed by
  // allreduce_staged() before any other collective.
  void* stage_next(std::string* error);

  // Submits the pre-staged one-shot: the held handout is the source and
  // the destination (the fold runs in place; the model's next kernels read
  // the result from the same pinned buffer). Same v1 contract as
  // allreduce(); wait via wait_allreduce().
  uint64_t allreduce_staged(size_t bf16_elems, std::string* error);

  // Segment-quantized reduce-scatter + allgather over the bulk pool
  // (DESIGN §6.3, the prefill class): for boundaries well above one
  // latency slot. The buffer stripes on the bulk-slot grid; the bus drives
  // segments of at most bulk_slots x lanes stripes internally (one kernel
  // launch and one posting wave per segment, within every pool depth) and
  // splits EACH SEGMENT's stripes across the ranks (a contiguous ceil
  // split within the segment), so every rank folds and broadcasts in every
  // segment — a global split had one rank working per segment of a
  // multi-segment buffer. The RS fold is
  // the canonical ascending-rank chain per element — bitwise identical
  // to the latency one-shot, so both paths against one oracle agree
  // exactly. src/dst are device pointers and may alias (staging reads
  // non-owned shards, the fold writes the owned one, AG reads it back —
  // disjoint phases within a launch and across the machine). Wire cost
  // 2(W-1)/W of the buffer (one-shot pays W-1; RS+AG halves it at W=4).
  // Same v1 single-outstanding contract; wait via wait_allreduce().
  uint64_t allreduce_bulk(const void* device_src, void* device_dst,
                          size_t bf16_elems, std::string* error);

  // Blocks for the collective; on success the destination is stream-ordered
  // for the caller (waits for the stream after the engine's completion).
  // timeout_ms also bounds that stream wait. A failure closes subsequent
  // collective submissions; the engine watchdog owns in-flight cleanup.
  // Removes completed request records.
  BusAllReduceResult wait_allreduce(uint64_t id, int timeout_ms);

  // ---- graph capture (DESIGN §6.2, the decode step) ------------------------
  //
  // The decode step is a fixed launch sequence, so it records once into a
  // CUDA graph and replays per token. The collective kernels become graph
  // nodes; the engine stops launching them and instead walks the
  // generations each replay produces (per-gen cells carry the handoff).
  //
  // Session (one or more variants per bus):
  //   graph_record_begin(error, variant) — open one variant. Requires a
  //                              quiet bus: no held staging handout, no eager collective in
  //                              flight or queued, consumers not running.
  //                              Harness send() closes from here (the
  //                              graph kernels claim doorbells exactly
  //                              like eager collectives) and stays closed
  //                              for the bus's lifetime — one graph era.
  //                              A variant id may be recorded once; every
  //                              variant owns a disjoint cell set.
  //   allreduce_record()      — per collective node, between the caller's
  //                              cudaStreamBeginCapture/EndCapture on the
  //                              same stream. src/dst are device pointers
  //                              that must stay valid for the bus's
  //                              lifetime (baked into the graph). Same
  //                              element contract as allreduce(); results
  //                              are bitwise the eager machine's.
  //   graph_record_end()      — close and publish to the engine. At least
  //                              one node required; at most
  //                              kBusMaxGraphGens nodes.
  //
  // Replay (per step):
  //   graph_replay_arm(error, variant) — select a recorded variant, reset
  //                              its per-gen cells, reserve the
  //                              window's generations from the shared
  //                              collective counter (the kernels read
  //                              them at start), publish the window.
  //                              Call BEFORE cudaGraphLaunch. Rejects a
  //                              held staging handout (its row would be
  //                              rewritten by the window's kernels). Up
  //                              to kBusMaxLiveWindows windows may be
  //                              armed at once (2026-09-06, the
  //                              pipelined replay: the next replay's
  //                              window while the previous replay still
  //                              runs), each on a variant whose window is
  //                              not live — its cells are in flight until
  //                              finish; arm waits (bounded) for the
  //                              engine to have walked the window
  //                              kBusMaxLiveWindows back. Windows are
  //                              walked and finished in arm order.
  //   graph_replay_finish()   — finishes the OLDEST armed window, after
  //                              its replay's work completed (the
  //                              caller's stream sync is not enough: the
  //                              engine must observe every generation's
  //                              done). Waits (bounded) for the walk,
  //                              verifies statuses, returns the verdict.
  //
  // Mixed era: eager collectives (allreduce/allreduce_staged/allreduce_bulk)
  // and stage_next() are rejected only while a session RECORDS or while a
  // replay window is ARMED (arm .. finish); between windows they run as
  // before — prefill's bulk folds and the pick's latency collectives ride
  // the same engine. The interface is the generation counter: arm reserves the
  // window's G generations from it, eager pickups take one each, so
  // execution order equals generation order and the staging-ring reuse
  // fences hold across eras verbatim. Session discipline: one forward
  // thread for arm/finish/eager submissions (the gates are coll_mu-
  // serialized, but the numbering contract is single-threaded by design).
  // The counter is 32-bit; an arm whose reservation would cross its top
  // fails the era loudly (~4.3e9 collectives ≈ 48M decode tokens — the
  // remedy is a process restart).
  //
  // Any graph failure (a generation exits on deadline/poison, a lane
  // fails, a post fails) poisons the era: further arm/finish report it,
  // and the eager gate stays closed (the bus must be restarted).
  bool graph_record_begin(std::string* error, int variant = 0);
  bool allreduce_record(cudaStream_t capture_stream, const void* device_src,
                        void* device_dst, size_t bf16_elems,
                        std::string* error);
  bool graph_record_end(std::string* error);
  bool graph_replay_arm(std::string* error, int variant = 0);
  // %globaltimer - CLOCK_MONOTONIC, calibrated at start() (bus_kernel.hpp).
  int64_t globaltimer_offset_ns() const;
  bool graph_replay_finish(int timeout_ms, std::string* error);

  // ---- the GPU-driven eager fold (2026-09-14, plan D9) --------------------
  //
  // The eager one-shot launched on the CALLER's stream in the graph kernel
  // form: no host drain of the model stream before the fold, no engine
  // launch on the collective stream, no host notice of the finish before
  // the model may continue — the stream orders the result exactly as a
  // recorded node's replay does. allreduce_stream() takes the generation
  // from the shared counter at submit (one forward thread, one stream:
  // execution order equals generation order, so the staging-ring reuse
  // fences hold as they do across windows), resets a cell of the stream
  // ring, pushes the generation to the engine's FIFO and launches the
  // kernel on `stream`; the engine posts from the FIFO as it posts from
  // a replay window (the same flight per generation). Returns the
  // generation (0 = rejected): under the eager gate (a recording session,
  // an armed window, a failed era), with a host-driven collective queued or
  // in flight, with a pre-stage handout held, or after a launch failure
  // (the era fails). Blocks (bounded) while kBusStreamRing generations are
  // in flight. Same element contract as allreduce(); the fold is bitwise
  // the eager machine's and the replay's.
  //
  // allreduce_settle(): waits until every stream generation issued so far
  // is walked — once per pass, before the host reads results, submits a
  // host-driven collective or arms a replay window (those reject while
  // stream generations are outstanding). A generation that exits on
  // deadline or poison fails the era (the remaining stream kernels are
  // poisoned so the caller's stream drains); settle returns the verdict.
  // A world of one: the identity copy on the stream; settle succeeds.
  uint64_t allreduce_stream(cudaStream_t stream, const void* device_src,
                            void* device_dst, size_t bf16_elems,
                            std::string* error);
  bool allreduce_settle(int timeout_ms, std::string* error);

  // TEMP bring-up microscope: the per-gen cells' gen/ready/done/status,
  // failure-only observability for the graph walk.
  void dump_graph_cells(const char* why);

  // Phase 1 of an orderly stop: rejects new submissions, joins the engine
  // (draining outstanding requests), stops the receive consumers via the
  // control cell, and destroys their stream. Frees nothing — cudaFreeHost
  // synchronizes the device implicitly, so in-process loopback pairs must
  // quiesce both sides before either stops (one bus per process never
  // notices; stop() alone is correct there).
  void quiesce();

  // quiesce() plus waiter release and full teardown (QP/MR/slab frees).
  // Also invoked by the destructor.
  void stop();

  BusStats stats() const;
  // Successful multi-rank collective completions (eager, bulk, stream and
  // graph). Readable without CUDA or bus locks, including while a model
  // chunk is still running. Submissions, polls and failed flights do not
  // advance it. The reference lives as long as this bus.
  const std::atomic<uint64_t>& completion_epoch() const;
  // The bulk pacing rate in effect after start() (the derived one when
  // the option was negative), Gb/s per (peer, lane) QP; 0 = unpaced.
  double bulk_pace_gbps() const;
  int lane_count() const { return static_cast<int>(options_.lane_devices.size()); }
  size_t slot_bytes(BusMessageClass cls) const {
    return cls == BusMessageClass::kLatency ? options_.lat_slot_bytes
                                            : options_.bulk_slot_bytes;
  }
  size_t bulk_segment_bytes() const {
    return static_cast<size_t>(options_.bulk_slots) * options_.lane_devices.size() * options_.bulk_slot_bytes;
  }

  // Receive-side view of one peer/lane (doorbells, acks, payloads, control).
  BusRecvView recv_view(int peer_rank, int lane) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  BusOptions options_;
};

}  // namespace dgpp::net
