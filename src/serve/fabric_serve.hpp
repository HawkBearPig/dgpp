#pragma once
// Admission journal for synchronizing schedulers across ranks (DESIGN §11).
// Rank 0 accepts HTTP requests and sends newline-framed JSON over a TCP star:
//
//   {"op":"settings", ...}       model, world and engine settings, sent first
//   {"op":"warm","adm":{...},"pc":N,"cfg":"<digest>"}
//   {"op":"tick","s":[{"id":"…","p":[ids],"m":N,"b":[...],"nc":1}],"c":["id"],"pd":D,"od":F}
//   {"op":"stop"}
//
// The settings record precedes model construction. Warm supplies admission
// settings and prefix slot count, checks the configuration digest and
// coordinates graph warm capture before the first tick. A tick carries
// accepted submissions and cancellations; p contains token IDs, b the
// prompt boundaries and nc the cache opt-out flag. Rejected HTTP requests
// stay on rank 0. Tokens, retirements and cache decisions are derived by
// each scheduler from the same inputs.
//
// Rank 0 broadcasts each tick after draining submissions and before calling
// sched.tick(). Peers apply the record and execute that tick. Collectives
// keep engine operations aligned, so stop and subsequent records are
// handled between ticks. A peer rejecting an admitted request is a fatal
// scheduler divergence.
//
// od is rank 0's FNV-1a operation-stream fold after the previous tick; pd
// is its prefix-decision digest. Peers compare their own values before
// applying the record and report the tick on a mismatch. Shutdown hashes
// compare the complete streams.
//
// Journal watches detect socket closure even while an engine is inside a
// collective. Rank 0 fails active streams after their completed tokens;
// peers exit on loss of rank 0. The bus watchdog handles silent node loss.
// There is no failover within a running world.
#include <atomic>
#include <cstdio>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "kernels/rope_scaling.hpp"
#include "sched/scheduler.hpp"
#include "net/tcp.hpp"
#include "serve/generation_service.hpp"
#include "serve/serve_stats.hpp"

namespace dgpp::serve {

// The audit tap on the engine event stream. The fabric run attaches
// one on every rank (rank 0 through GenerationService::set_audit_
// observer, peers through Scheduler::set_observer) and the
// verification hashes the per-rank texts against each other — the
// smoke's md5 procedure, serving edition. The format is deliberately
// deterministic and rank-independent: same events in the same order
// MUST produce the same bytes on every rank. Appends happen on the
// engine thread; readers (the run's exit write, the tests' live polls)
// take copies under the lock.
class OpStreamObserver final : public dgpp::sched::SchedulerObserver {
 public:
  void on_token(const std::string& id, int64_t token,
                int steps_done) override;
  void on_retire(const std::string& id,
                 const dgpp::sched::Scheduler::Result& result) override;
  void on_grow(const std::string& id, int64_t reserved_tokens) override;
  void on_prefix(const std::string& id, const char* op, int64_t position,
                 int slot) override;
  // The stream so far, when held in memory (the tests' live polls; ""
  // once open() streams it to a file).
  std::string text() const;
  // Streams every line to `path` as it is recorded (the file truncated
  // here, flushed at every retire and by flush()) instead of holding the
  // run's whole stream in memory until an exit-time write — the op stream
  // on disk at every moment, a killed rank's evidence included, and a
  // process that does not grow with its traffic (2026-09-13: 37 bytes per
  // generated token for the process's lifetime). False when the file
  // cannot be opened; the stream then stays in memory as before.
  bool open(const std::string& path);
  void flush();
  OpStreamObserver() = default;
  ~OpStreamObserver();
  // The running FNV-1a fold of every line recorded so far (the journal's
  // "od", the continuous drift check).
  bool has_digest() const override { return true; }
  uint64_t digest() const override;

 private:
  void append(const std::string& line, bool flush = false);  // under the lock
  mutable std::mutex mutex_;
  std::string text_;
  std::FILE* file_ = nullptr;   // the sink once open()
  bool write_failed_ = false;   // the short write logged once
  uint64_t digest_ = 0xcbf29ce484222325ull;
};

// ---- wire codec ----------------------------------------------------------

std::string encode_journal_tick(const GenerationService::PassEvents& events);
std::string encode_journal_stop();
// The warm record carries rank 0's admission policy (M6 6d) and its prefix
// cache slot count (M7): every rank's scheduler must run the same ones,
// and the peers take them from here.
std::string encode_journal_warm(
    const dgpp::sched::AdmissionPolicy& policy = dgpp::sched::AdmissionPolicy{},
    int prefix_slots = 0,
    const std::string& config_digest = "");

// The settings record (2026-09-06, the productionizing pass): what rank 0
// pushes to every peer over the journal BEFORE anything builds — the first
// record a peer reads. The head is the one source of the world's shape:
// the model, the world size, the fabric port and every engine knob that
// shapes the op stream; a peer's own flags or file yield to it (logged
// when they differed). The effective-config digest on the warm record
// then holds by construction and stays as an assertion.
struct WorldSettings {
  std::string version;     // rank 0's dgpp version: a mixed-version world refuses to form
  std::string model;       // the HF model id ("" when rank 0 ran from a directory)
  std::string checkpoint;  // rank 0's checkpoint directory (a peer without an id uses it)
  int world = 1;
  int fabric_port = 0;
  int max_concurrency = 0;
  int64_t kv_capacity = 0;
  std::string kv_dtype = "bf16";  // the latent cache's format
  std::string ngram_table = "resident";  // the Qwen n-gram table's residency
  std::string dense_weights = "checkpoint";  // the Qwen dense stack's form
  std::string prefill = "bounded";           // the DeepSeek-V4.1 prefill mode: bounded | exact
  std::optional<dgpp::RopeScaling> rope_scaling;  // the opt-in YaRN ramp: absent = plain
  std::string embed_sharding = "replicated";  // the full GLM-5.3's embedding: replicated | vocab
  int default_max_tokens = 0;
  int queue_limit = 0;
  bool no_eos = false;
  bool decode_graph = false;
  bool mtp = false;
  int mtp_depth = 1;  // draft tokens per step
  // The scheduled verify depth and its constants (identical on every rank:
  // the depth decision must agree across the world).
  bool mtp_schedule = false;
  double mtp_schedule_row_ms = 8.0;
  double mtp_schedule_base_ms = 28.0;
  double mtp_schedule_lambda = 0.0;
  int mtp_schedule_min_depth = 1;
  bool mtp_schedule_adapt = true;
  int graph_batch_min_live = 0;
  int sampling_candidates = 0;
  double prefix_cache_gib = 0.0;
  std::string admission;
  int admission_window = 0;
  int prefill_budget_tokens = 0;
  int prefill_idle_budget_tokens = 0;
  double bulk_pace_gbps = 0.0;
  int bulk_inflight = 0;
  int rendezvous_timeout_ms = 0;
  double stats_interval_s = 0.0;
  bool reasoning_in_content = false;
  bool operator==(const WorldSettings&) const = default;
};
std::string encode_journal_settings(const WorldSettings& s);

struct JournalRecord {
  bool stop = false;
  bool warm = false;
  bool settings = false;  // the settings record (2026-09-06)
  WorldSettings world_settings;
  bool has_admission = false;  // warm: the policy rode along
  dgpp::sched::AdmissionPolicy admission;
  int prefix_slots = 0;          // warm: rank 0's prefix cache slots
  std::string config_digest;     // warm: rank 0's effective-config digest ("cfg", 2026-09-06)
  bool has_prefix_digest = false;  // tick: rank 0's digest rode along
  uint64_t prefix_digest = 0;
  bool has_op_digest = false;      // tick: rank 0's op-stream fold rode along
  uint64_t op_digest = 0;
  std::vector<dgpp::sched::SchedulerRequest> submits;
  std::vector<std::string> cancels;
  std::vector<std::string> stops;  // the stop-string retires,
                                   // applied like cancels ("sp")
};
// The tick record's cross-rank check (M7): rank 0's prefix-cache digest
// after the previous tick against this rank's. Throws on a mismatch — the
// schedulers' cache decisions diverged, a fabric emergency.
void check_prefix_digest(const JournalRecord& rec,
                         const dgpp::sched::Scheduler& sched);
// The continuous drift check (M9): rank 0's op-stream fold after the
// previous tick against this rank's observer's. Throws on a mismatch,
// naming the tick — the ranks' engine event streams diverged, a fabric
// emergency. A record without "od", or a null observer, checks nothing.
void check_op_digest(const JournalRecord& rec,
                     const dgpp::sched::SchedulerObserver* oplog, int64_t tick);
// Throws on any malformed record — a corrupt journal is a fabric
// emergency, not a condition to paper over.
JournalRecord decode_journal_line(std::string_view line);

// ---- rank 0 ----------------------------------------------------------------

class JournalWriter {
 public:
  // Binds the journal port (0 = ephemeral; see port()) but does not
  // wait for peers — the app binds first so peers can be launched
  // against a known port, then accepts.
  explicit JournalWriter(uint16_t listen_port);

  uint16_t port() const { return listener_.port(); }

  // Accepts world-1 peer connections in any order and reads each one's
  // hello ("hello <rank>\n" — the one thing a peer ever writes), so the
  // connections are known by rank. Throws when the full world fails to
  // land within accept_timeout_ms — a short world cannot serve, and
  // pretending otherwise would wedge every request at its first
  // collective.
  void accept_peers(int world, int accept_timeout_ms);

  // One line to every peer. A peer that cannot take it is a dead
  // member of a live-serving world — the fabric is broken, and this
  // says so loudly.
  void broadcast(const std::string& line);

  // The peer liveness watch (the death discipline above): a thread
  // polling every peer connection each `poll_ms`; the first closed or
  // reset one calls `on_death(rank, why)` ONCE, from the watch thread, and
  // the watch ends. The caller fails the service from that callback
  // (GenerationService::fail_engine is thread-safe) without waiting for a
  // bus watchdog. stop_watch() joins the thread (the destructor too).
  void watch_peers(std::function<void(int, const std::string&)> on_death,
                   int poll_ms = 100);
  void stop_watch();
  // The rank of a closed or reset peer connection, or 0.
  int dead_peer() const;
  // Closes every peer connection now — what rank 0's process exit does;
  // the tests' stand-in for it (the peers see EOF).
  void close_peers();
  ~JournalWriter();

 private:
  dgpp::net::TcpListener listener_;
  std::vector<dgpp::net::TcpConn> peers_;  // arrival order; peer_ranks_ names them
  std::vector<int> peer_ranks_;
  std::thread watch_;
  std::atomic<bool> watch_stop_{false};
};

// ---- peers ------------------------------------------------------------------

class JournalReader {
 public:
  // Connects to rank 0's journal port, RETRYING within the window: the
  // peer's bus rendezvous can complete before rank 0 binds the journal
  // (a race measured in tens of milliseconds), and a single-shot
  // connect loses it. Throws once the whole window expires — startup
  // is exceptional and the peer wants the reason in its log. Sends the
  // hello ("hello <rank>\n") that names the connection to rank 0 — the
  // only bytes a peer ever writes.
  JournalReader(const std::string& host, uint16_t port,
                int connect_timeout_ms, int rank);

  // The next complete record (newline stripped). false when rank 0's
  // journal closed (clean stop or crash — EOF either way) or the
  // caller's stop flag fired; the two are indistinguishable to the
  // peer on purpose: both mean "the world is over, exit now".
  bool read_line(const std::function<bool()>& should_stop,
                 std::string* line);
  // Rank 0's connection is closed or reset (nothing consumed; pending
  // records read as alive). The in-tick watch's probe.
  bool rank0_closed() const { return conn_.peer_closed(); }
  // Closes the connection (a peer's death, from rank 0's side — the
  // tests' stand-in for the process going away). shutdown() does it
  // from another thread while the loop reads: the read sees EOF, rank 0
  // sees the FIN.
  void close() { conn_.close(); }
  void shutdown() { conn_.shutdown_rw(); }

 private:
  dgpp::net::TcpConn conn_;
  std::string pending_;
};

// The peer's startup hold: blocks until rank 0's warm record. Returns
// false when the journal ended (EOF or the caller's stop flag) — the
// peer exits cleanly, as it would from the loop. Throws when the first
// record is anything else: rank 0 ticked before warming, a protocol
// order this design says cannot happen.
bool wait_journal_warm(JournalReader* reader,
                       const std::function<bool()>& should_stop,
                       dgpp::sched::AdmissionPolicy* policy = nullptr,
                       int* prefix_slots = nullptr,
                       std::string* config_digest = nullptr);
// The peer's first read (2026-09-06): rank 0's settings record, sent right
// after the journal star forms and before either side builds a model.
// Returns false when the stream ended or a stop arrived first; throws when
// the first record is anything else (a protocol violation).
// `my_version` (optional): this rank's dgpp version; a record from another
// version throws — a mixed-version world refuses to form.
bool wait_journal_settings(JournalReader* reader,
                           const std::function<bool()>& should_stop,
                           WorldSettings* out,
                           const std::string& my_version = "");

// The peer serving loop: apply each record, then tick — the exact
// mirror of rank 0's engine passes (§11). Returns on the stop record,
// journal EOF, or the caller's stop flag. Throws on journal
// corruption or scheduler divergence (both are fabric emergencies).
// Attach the observer BEFORE calling: the loop drives the scheduler,
// so whatever the observer should see, it sees here.
// `on_rank0_death` (optional): the in-tick watch — a thread that, while
// the loop is inside sched->tick() (a collective a dead rank 0 can never
// complete), polls rank 0's connection every `watch_poll_ms` and calls
// the hook ONCE when it closes; the hook is expected not to return to
// the loop (the app writes its op stream and exits nonzero). Between
// ticks the read loop sees the EOF itself and returns as before.
// `oplog` (optional): this rank's op-stream observer, whose fold every
// record's "od" is compared against before the record is applied (the
// continuous drift check) — the same observer attached to the scheduler.
// `stats` (optional): this rank's throughput line, fed the scheduler's
// meters after every tick (serve_stats.hpp).
void run_journal_peer(dgpp::sched::Scheduler* sched, JournalReader* reader,
                      const std::function<bool()>& should_stop,
                      const std::function<void()>& on_rank0_death = nullptr,
                      int watch_poll_ms = 100,
                      const dgpp::sched::SchedulerObserver* oplog = nullptr,
                      ThroughputLog* stats = nullptr);

}  // namespace dgpp::serve
