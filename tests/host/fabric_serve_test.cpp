// M6 Stage 4b gate: the admission journal (the metronome), both loops,
// and the §11 identity — over REAL localhost TCP and REAL HTTP, with
// only the engine faked. What the 4a gate did for the OpenAI contract,
// this does for the fabric interface:
//   * rank-0 side: HttpServer + GenerationService (both real) over a
//     deterministic FakeEngine, the engine loop driving engine_pass
//     with the journal hook — exactly the app's loop;
//   * peer side (two of them — the star broadcast): a real Scheduler
//     over its own FakeEngine (same token functions), driven by
//     run_journal_peer;
//   * the §11 identity, live: every peer's op stream must equal
//     rank 0's audit stream after every scenario, including a
//     mid-stream client disconnect (the cancel must cross the journal
//     and retire on every rank at the same tick);
//   * the stop discipline: the stop record releases the peers, every
//     thread joins, nobody errored.
// The fakes are the 4a gate's (same token laws), so the shapes this
// test skims are already byte-pinned there — here the JOURNAL is the
// thing under test.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/log.hpp"
#include "common/test.hpp"
#include "sched/scheduler.hpp"
#include "serve/fabric_serve.hpp"
#include "serve/generation_service.hpp"
#include "serve/http_server.hpp"

namespace {

using dgpp::sched::SchedulerEngine;
using dgpp::serve::GenerationService;
using dgpp::serve::HttpServer;
using dgpp::serve::ModelFrontend;
using dgpp::serve::ServiceConfig;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

constexpr int32_t kFakeEos = 999;  // the fake's end-of-sequence id

// The 4a gate's deterministic fake model: token i of a request whose
// rendered prompt is P bytes is ((len*31 + i*7) % 250) + 1 — never 0,
// never EOS; prompt len % 4 == 3 answers EOS as its second token,
// len % 4 == 2 answers EOS on the PREFILL pick.
int32_t fake_token(size_t prompt_len, int index) {
  return static_cast<int32_t>((prompt_len * 31 +
                               static_cast<size_t>(index) * 7) % 250) + 1;
}
bool fake_eos_second(size_t prompt_len) { return prompt_len % 4 == 3; }
bool fake_eos_prefill(size_t prompt_len) { return prompt_len % 4 == 2; }

// The prefix cache and the NVMe cold tier the rig's fakes carry (issue
// #26): set before a rig is built (the scheduler reads them at
// construction); 0 = off. Every rank's fake reads the same values, so the
// three engines expose one geometry.
int g_fake_arena_slots = 0;
int64_t g_fake_disk_pages = 0;

class FakeEngine : public SchedulerEngine {
 public:
  FakeEngine(int slots, int64_t total_blocks, int64_t block_tokens)
      : slots_(slots), total_blocks_(total_blocks),
        block_tokens_(block_tokens), arena_slots_(g_fake_arena_slots), disk_pages_(g_fake_disk_pages) {}

  // ---- the prefix cache (M7): pins by arena slot, like scheduler_test's fake ----
  dgpp::sched::SchedulerEngine::PrefixInfo prefix_info() const override {
    dgpp::sched::SchedulerEngine::PrefixInfo info;
    info.arena_slots = arena_slots_;
    info.align = 4;
    info.block_tokens = block_tokens_;
    info.chunk_tokens = 8;  // cuts every 8 tokens: a 21-byte prompt snapshots at 16
    return info;
  }
  int32_t prefill_cached(int req, const std::vector<int64_t>& prompt,
                         dgpp::sched::SchedulerEngine::PrefixPrefill* plan) override {
    if (plan->attach_slot >= 0 && pinned_.count(plan->attach_slot) == 0)
      throw std::runtime_error("fake: attach from an empty arena slot");
    const int32_t token = prefill(req, prompt);
    if (plan->snap_slot >= 0) {
      pin(plan->snap_slot, plan->snap_position);
      plan->snap_taken = true;
    }
    if (plan->body_snap_slot >= 0) {
      pin(plan->body_snap_slot, plan->body_snap_position);
      plan->body_snap_taken = true;
    }
    return token;
  }
  void prefix_snapshot(int, int slot, int64_t position) override { pin(slot, position); }
  void prefix_release(int slot) override { pinned_.erase(slot); }

  // ---- the NVMe cold tier (issue #26): an in-memory slab, ops done a poll later ----
  dgpp::sched::SchedulerEngine::DiskInfo disk_info() const override {
    dgpp::sched::SchedulerEngine::DiskInfo i;
    i.enabled = disk_pages_ > 0;
    i.pages = disk_pages_;
    i.page_bytes = 4096;
    i.blob_pages = 1;
    return i;
  }
  dgpp::sched::SchedulerEngine::DiskBlocks disk_entry_blocks(int slot) const override {
    const auto it = pinned_.find(slot);
    if (it == pinned_.end()) throw std::runtime_error("fake: disk blocks of an empty arena slot");
    dgpp::sched::SchedulerEngine::DiskBlocks b;
    for (int64_t i = 0; i < it->second.position / block_tokens_; ++i)
      b.identities.push_back((it->second.gen << 32) | static_cast<uint64_t>(i));
    b.partial_identity = it->second.position % block_tokens_ != 0 ? ((it->second.gen << 32) | 0xffffu) : 0;
    return b;
  }
  void disk_spill_begin(const dgpp::sched::SchedulerEngine::DiskSpill& sp) override {
    if (pinned_.count(sp.slot) == 0) throw std::runtime_error("fake: spill of an empty arena slot");
    disk_pending_.push_back({sp.op, 1, true});
  }
  dgpp::sched::SchedulerEngine::DiskBlocks disk_restore_begin(
      const dgpp::sched::SchedulerEngine::DiskRestore& rs) override {
    if (pinned_.count(rs.slot) != 0) throw std::runtime_error("fake: restore into an occupied arena slot");
    pin(rs.slot, rs.position);
    disk_pending_.push_back({rs.op, 1, true});
    return disk_entry_blocks(rs.slot);
  }
  std::vector<dgpp::sched::SchedulerEngine::DiskCompletion> disk_poll() override {
    std::vector<dgpp::sched::SchedulerEngine::DiskCompletion> out;
    for (auto it = disk_pending_.begin(); it != disk_pending_.end();) {
      if (--it->left > 0) {
        ++it;
        continue;
      }
      out.push_back({it->op, it->ok});
      it = disk_pending_.erase(it);
    }
    return out;
  }

  // Scenario knob: a per-op sleep that keeps a request in flight long
  // enough for the disconnect test to pull the plug mid-generation
  // (the instant default retires everything before any client could).
  void set_op_delay_ms(int ms) { op_delay_ms_ = ms; }
  // Scenario knob (the failure gates): while `*block` is set, prefill()
  // holds inside the engine op — the peer is "inside a tick", where only
  // the in-tick watch can see rank 0 die; blocked() says it got there.
  void set_block(std::atomic<bool>* block) { block_ = block; }
  bool blocked() const { return blocked_.load(); }
  // Scenario knob (the drift gate): this fake's tokens are offset by `d` —
  // a rank whose engine computes differently from the others.
  void set_token_offset(int d) { token_offset_ = d; }

  int max_concurrent_requests() const override { return slots_; }
  int64_t pool_blocks_total() const override { return total_blocks_; }
  int64_t pool_blocks_in_use() const override {
    int64_t sum = 0;
    for (const auto& [slot, live] : live_) (void)slot, sum += live.held_blocks;
    for (const auto& [slot, pin] : pinned_) (void)slot, sum += pin.blocks;
    return sum;
  }
  int64_t blocks_for_tokens(int64_t tokens) const override {
    return (tokens + block_tokens_ - 1) / block_tokens_;
  }

  int32_t prefill(int req, const std::vector<int64_t>& prompt) override {
    if (int d = op_delay_ms_.load())
      std::this_thread::sleep_for(std::chrono::milliseconds(d));
    if (block_ != nullptr && block_->load()) {
      blocked_.store(true);
      while (block_->load())
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (live_.count(req) != 0)
      throw std::runtime_error("fake: prefill on live slot");
    Live live;
    live.prompt_len = prompt.size();
    live.served = 1;
    live.last_token = fake_eos_prefill(live.prompt_len)
                          ? kFakeEos
                          : fake_token(live.prompt_len, 0) + token_offset_.load();
    live_[req] = live;
    return live.last_token;
  }

  void reserve(int req, int64_t tokens) override {
    Live& live = live_.at(req);
    live.held_blocks = blocks_for_tokens(tokens);
  }

  std::vector<int32_t> step(int req) override {
    if (int d = op_delay_ms_.load())
      std::this_thread::sleep_for(std::chrono::milliseconds(d));
    Live& live = live_.at(req);
    live.last_token =
        fake_eos_second(live.prompt_len) && live.served == 1
            ? kFakeEos
            : fake_token(live.prompt_len, static_cast<int>(live.served)) +
                  token_offset_.load();
    ++live.served;
    return {live.last_token};
  }

  void close(int req) override { live_.erase(req); }

 private:
  struct Live {
    size_t prompt_len = 0;
    int64_t held_blocks = 0;
    int served = 0;
    int32_t last_token = -1;
  };
  struct Pin {
    int64_t position = 0;
    int64_t blocks = 0;
    uint64_t gen = 0;
  };
  struct PendingDisk {
    uint64_t op;
    int left;
    bool ok;
  };
  void pin(int slot, int64_t position) {
    Pin p;
    p.position = position;
    p.blocks = position / block_tokens_ + (position % block_tokens_ != 0 ? 1 : 0);
    p.gen = ++gen_;
    pinned_[slot] = p;
  }
  int slots_;
  int64_t total_blocks_;
  int64_t block_tokens_;
  int arena_slots_ = 0;
  int64_t disk_pages_ = 0;
  std::map<int, Pin> pinned_;
  std::vector<PendingDisk> disk_pending_;
  uint64_t gen_ = 0;
  std::atomic<int> op_delay_ms_{0};
  std::atomic<bool>* block_ = nullptr;
  std::atomic<bool> blocked_{false};
  std::atomic<int> token_offset_{0};
  std::map<int, Live> live_;
};

class FakeFrontend : public ModelFrontend {
 public:
  std::vector<int64_t> encode_text(std::string_view text) const override {
    std::vector<int64_t> ids;
    for (const char c : text) ids.push_back(static_cast<unsigned char>(c));
    return ids;
  }
  std::string decode_ids(const std::vector<int64_t>& ids) const override {
    std::string out;
    for (int64_t id : ids)
      if (id != kFakeEos) out.push_back(static_cast<char>(id));
    return out;
  }
  std::string render_chat(const dgpp::minijson::Value& globals) const override {
    std::string out;
    for (const auto& msg : globals.at("messages").items())
      if (const auto* content = msg.find("content"))
        out.append(content->as_string());
    return out;
  }
};

// --- the raw-socket client (the 4a gate's) -----------------------------
class Client {
 public:
  explicit Client(uint16_t port) {
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    require(fd_ >= 0, "client socket");
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    require(::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1,
            "client inet_pton");
    require(::connect(fd_, reinterpret_cast<sockaddr*>(&addr),
                      sizeof(addr)) == 0,
            "client connect");
    int yes = 1;
    ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
  }
  ~Client() {
    if (fd_ >= 0) ::close(fd_);
  }
  Client(const Client&) = delete;
  Client& operator=(const Client&) = delete;

  void send_all(std::string_view s) {
    size_t off = 0;
    while (off < s.size()) {
      const ssize_t put = ::send(fd_, s.data() + off, s.size() - off, 0);
      require(put > 0, "client send");
      off += static_cast<size_t>(put);
    }
  }
  std::string read_until(const std::string& needle, int budget_ms) {
    std::string out;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(budget_ms);
    while (out.find(needle) == std::string::npos &&
           std::chrono::steady_clock::now() < deadline && !closed_) {
      pollfd p{fd_, POLLIN, 0};
      if (::poll(&p, 1, 25) > 0) {
        char buf[4096];
        const ssize_t got = ::recv(fd_, buf, sizeof(buf), 0);
        if (got > 0) out.append(buf, static_cast<size_t>(got));
        if (got == 0) closed_ = true;
      }
    }
    return out;
  }
  void hard_close() {
    ::shutdown(fd_, SHUT_RDWR);
    ::close(fd_);
    fd_ = -1;
  }

 private:
  int fd_ = -1;
  bool closed_ = false;
};

// --- the settings handshake ------------------------------------------------
// Rank 0 accepts the world and pushes the settings before anything builds;
// the peer's first read is that record, applied whole. A stop record first
// ends the peer cleanly; a tick first is a protocol violation.
void test_settings_handshake() {
  dgpp::serve::WorldSettings sent;
  sent.version = "0.1.0+gtest";
  sent.model = "org/model";
  sent.world = 2;
  sent.fabric_port = 29970;
  sent.max_concurrency = 2;
  sent.kv_capacity = 4096;
  sent.default_max_tokens = 32;
  sent.queue_limit = 8;
  sent.decode_graph = true;
  sent.mtp = true;
  sent.mtp_depth = 2;
  sent.fp8_head = "mma";
  sent.sampling_candidates = 128;
  sent.prefix_cache_gib = 1.5;
  sent.admission = "full";
  sent.admission_window = 256;
  sent.bulk_pace_gbps = -1.0;
  sent.bulk_inflight = -1;
  sent.rendezvous_timeout_ms = 120000;
  sent.stats_interval_s = 10.0;
  {
    dgpp::serve::JournalWriter writer(29938);
    std::atomic<bool> stop{false};
    dgpp::serve::WorldSettings got;
    bool ok = false;
    std::thread peer([&] {
      dgpp::serve::JournalReader reader("127.0.0.1", 29938, 5000, 1);
      ok = dgpp::serve::wait_journal_settings(&reader, [&] { return stop.load(); }, &got,
                                              "0.1.0+gtest");
    });
    writer.accept_peers(2, 5000);
    writer.broadcast(dgpp::serve::encode_journal_settings(sent));
    peer.join();
    require(ok && got == sent, "the peer's first read is rank 0's settings, whole");
    require(got.fp8_head == "mma", "rank 0 overrides the peer default head mode");
  }
  {
    // A peer of another version refuses: a mixed-version world cannot form.
    dgpp::serve::JournalWriter writer(29941);
    std::atomic<bool> stop{false};
    std::string violation;
    std::thread peer([&] {
      dgpp::serve::JournalReader reader("127.0.0.1", 29941, 5000, 1);
      dgpp::serve::WorldSettings got;
      try {
        (void)dgpp::serve::wait_journal_settings(&reader, [&] { return stop.load(); }, &got,
                                                 "0.1.0+gother");
      } catch (const std::runtime_error& e) {
        violation = e.what();
      }
    });
    writer.accept_peers(2, 5000);
    writer.broadcast(dgpp::serve::encode_journal_settings(sent));
    peer.join();
    require(violation.find("mixed-version") != std::string::npos &&
                violation.find("0.1.0+gtest") != std::string::npos &&
                violation.find("0.1.0+gother") != std::string::npos,
            "a different version refuses naming both: " + violation);
  }
  {
    dgpp::serve::JournalWriter writer(29939);
    std::atomic<bool> stop{false};
    bool ended = true;
    std::string violation;
    std::thread peer([&] {
      dgpp::serve::JournalReader reader("127.0.0.1", 29939, 5000, 1);
      dgpp::serve::WorldSettings got;
      try {
        ended = !dgpp::serve::wait_journal_settings(&reader, [&] { return stop.load(); }, &got);
      } catch (const std::runtime_error& e) {
        violation = e.what();
      }
    });
    writer.accept_peers(2, 5000);
    writer.broadcast(dgpp::serve::encode_journal_warm());  // a warm record first: wrong
    peer.join();
    require(violation.find("settings record") != std::string::npos,
            "a first record that is not the settings is a protocol violation: " + violation);
  }
  {
    dgpp::serve::JournalWriter writer(29940);
    std::atomic<bool> stop{false};
    bool ended = false;
    std::thread peer([&] {
      dgpp::serve::JournalReader reader("127.0.0.1", 29940, 5000, 1);
      dgpp::serve::WorldSettings got;
      ended = !dgpp::serve::wait_journal_settings(&reader, [&] { return stop.load(); }, &got);
    });
    writer.accept_peers(2, 5000);
    writer.broadcast(dgpp::serve::encode_journal_stop());
    peer.join();
    require(ended, "a stop record before the settings ends the peer cleanly");
  }
  std::printf("settings handshake: ok\n");
}

// --- the codec micro-gate ------------------------------------------------

void test_journal_codec() {
  GenerationService::PassEvents events;
  dgpp::sched::SchedulerRequest submit;
  submit.id = "chatcmpl-00000000000000ff";
  submit.prompt = {1, 2, 300, 154820 - 1};
  submit.max_steps = 16;
  submit.cancel_after = 3;
  events.submits.push_back(submit);
  events.cancels.push_back("chatcmpl-dead");

  const dgpp::serve::JournalRecord back = dgpp::serve::decode_journal_line(
      dgpp::serve::encode_journal_tick(events));
  require(!back.stop, "codec: tick decoded as stop");
  require(back.submits.size() == 1 && back.cancels.size() == 1,
          "codec: wrong event counts");
  require(back.submits[0].id == submit.id, "codec: id round-trip");
  require(back.submits[0].prompt == submit.prompt, "codec: prompt round-trip");
  require(back.submits[0].max_steps == 16, "codec: max_steps round-trip");
  require(back.submits[0].cancel_after == 3, "codec: cancel_after round-trip");
  require(back.submits[0].boundaries.empty() && !back.submits[0].no_cache &&
              !back.has_prefix_digest && !back.has_op_digest,
          "codec: a request without cache inputs decodes without them");
  {
    // The prefix cache's fields (M7): boundaries, the opt-out, the tick's
    // digest and the warm record's slot count round-trip; a record without
    // them is byte-identical to the pre-cache format.
    GenerationService::PassEvents ev;
    dgpp::sched::SchedulerRequest r;
    r.id = "chatcmpl-0000000000000c01";
    r.prompt = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    r.max_steps = 2;
    r.boundaries = {3, 6};
    r.no_cache = true;
    ev.submits.push_back(r);
    ev.has_prefix_digest = true;
    ev.prefix_digest = 18446744073709551557ull;  // above int64: the string form
    ev.has_op_digest = true;  // the op-stream fold (M9) rides the same way
    ev.op_digest = 18446744073709551533ull;
    const std::string line = dgpp::serve::encode_journal_tick(ev);
    require(line.find("\"b\":[3,6]") != std::string::npos &&
                line.find("\"nc\":1") != std::string::npos &&
                line.find("\"pd\":\"18446744073709551557\"") != std::string::npos &&
                line.find("\"od\":\"18446744073709551533\"") != std::string::npos,
            "codec: cache fields on the wire: " + line);
    const dgpp::serve::JournalRecord b2 = dgpp::serve::decode_journal_line(line);
    require(b2.submits.size() == 1 && b2.submits[0].boundaries == r.boundaries &&
                b2.submits[0].no_cache && b2.has_prefix_digest &&
                b2.prefix_digest == ev.prefix_digest && b2.has_op_digest &&
                b2.op_digest == ev.op_digest,
            "codec: cache fields round-trip");
    const std::string plain = dgpp::serve::encode_journal_tick(events);
    require(plain.find("\"b\"") == std::string::npos &&
                plain.find("\"nc\"") == std::string::npos &&
                plain.find("\"pd\"") == std::string::npos,
            "codec: the pre-cache record is unchanged");
    const dgpp::serve::JournalRecord warm = dgpp::serve::decode_journal_line(
        dgpp::serve::encode_journal_warm(dgpp::sched::AdmissionPolicy{}, 42));
    require(warm.warm && warm.prefix_slots == 42, "codec: warm record slot count");
    const dgpp::serve::JournalRecord warm0 = dgpp::serve::decode_journal_line(
        dgpp::serve::encode_journal_warm(dgpp::sched::AdmissionPolicy{}, 0));
    require(warm0.warm && warm0.prefix_slots == 0, "codec: warm record without a cache");
    dgpp::sched::AdmissionPolicy chunk_policy;
    chunk_policy.prefill_budget_tokens = 256;
    chunk_policy.prefill_idle_budget_tokens = 2048;
    const auto chunk_warm = dgpp::serve::decode_journal_line(dgpp::serve::encode_journal_warm(chunk_policy));
    require(chunk_warm.admission == chunk_policy && warm0.admission.prefill_budget_tokens == 0 &&
                warm0.admission.prefill_idle_budget_tokens == 0,
            "the deterministic prefill budget round-trips; old records keep monolithic admission");
    const auto fixed_warm = dgpp::serve::decode_journal_line(
        R"({"op":"warm","adm":0,"win":256,"pfbudget":256})");
    require(fixed_warm.admission.prefill_budget_tokens == 256 &&
                fixed_warm.admission.prefill_idle_budget_tokens == 0,
            "old budgeted records retain a fixed budget");
    for (const std::string value : {"-1", "1073741825", "\"256\""}) {
      bool rejected = false;
      try {
        (void)dgpp::serve::decode_journal_line(
            R"({"op":"warm","adm":0,"win":256,"pfbudget":256,"pfidle":)" + value + "}");
      } catch (const std::runtime_error& e) {
        rejected = std::string(e.what()).find("idle prefill budget") != std::string::npos;
      }
      require(rejected, "invalid idle budget is rejected by name");
    }
    // The configuration digest (2026-09-06) rides the warm record when rank 0
    // has one; a config-less rank 0's record is unchanged.
    const dgpp::serve::JournalRecord wcfg = dgpp::serve::decode_journal_line(
        dgpp::serve::encode_journal_warm(dgpp::sched::AdmissionPolicy{}, 42, "0123456789abcdef"));
    require(wcfg.warm && wcfg.prefix_slots == 42 && wcfg.config_digest == "0123456789abcdef",
            "codec: the config digest round-trips on the warm record");
    require(warm.config_digest.empty() &&
                dgpp::serve::encode_journal_warm(dgpp::sched::AdmissionPolicy{}, 42).find("cfg") ==
                    std::string::npos,
            "codec: no digest, no field");
    // The settings record (2026-09-06): every field round-trips, the doubles
    // exactly; a record with an impossible world is refused.
    dgpp::serve::WorldSettings ws;
    ws.version = "0.1.0+gabc";
    ws.model = "org/model";
    ws.checkpoint = "/ckpt/dir";
    ws.world = 4;
    ws.fabric_port = 29970;
    ws.max_concurrency = 4;
    ws.kv_capacity = 8192;
    ws.default_max_tokens = 256;
    ws.queue_limit = 8;
    ws.no_eos = true;
    ws.decode_graph = true;
    ws.mtp = true;
    ws.graph_batch_min_live = 3;
    ws.sampling_candidates = 24;
    ws.prefix_cache_gib = 1.25;
    ws.admission = "grow";
    ws.admission_window = 512;
    ws.prefill_budget_tokens = 256;
    ws.prefill_idle_budget_tokens = 2048;
    ws.bulk_pace_gbps = 28.333333333333332;
    ws.bulk_inflight = 4;
    ws.rendezvous_timeout_ms = 120000;
    ws.stats_interval_s = 0.1;
    ws.reasoning_in_content = true;
    ws.kv_dtype = "fp8";
    ws.bf16_weights = "bf12";
    ws.compact_batches = true;
    ws.fp8_head = "mma";
    // The opt-in rope knob rides the record: a peer that ran without it
    // would rope at different frequencies from rank 0 — silently
    // divergent text, the reason the settings record exists at all.
    dgpp::RopeScaling rs;
    rs.factor = 2.0;
    rs.original_max_position_embeddings = 262144;
    ws.rope_scaling = rs;
    const dgpp::serve::JournalRecord sr = dgpp::serve::decode_journal_line(
        dgpp::serve::encode_journal_settings(ws));
    require(sr.settings && !sr.warm && !sr.stop && sr.world_settings == ws,
            "codec: the settings record round-trips");
    for (int budget : {-1, 0, 256}) {
      auto settings = ws;
      settings.prefill_budget_tokens = budget;
      const auto decoded = dgpp::serve::decode_journal_line(
          dgpp::serve::encode_journal_settings(settings));
      require(decoded.world_settings.prefill_budget_tokens == budget,
              "automatic, disabled and explicit prefill budgets survive the settings handshake");
    }
    for (const std::string mode : {"gemv", "mma"}) {
      auto head = ws;
      head.fp8_head = mode;
      const auto decoded =
          dgpp::serve::decode_journal_line(dgpp::serve::encode_journal_settings(head));
      require(decoded.world_settings.fp8_head == mode, "codec: head mode round-trips");
    }
    {
      std::string legacy = dgpp::serve::encode_journal_settings(ws);
      const std::string key = ",\"fp8_head\":\"mma\"";
      const auto pos = legacy.find(key);
      require(pos != std::string::npos, "codec: head key is emitted");
      legacy.erase(pos, key.size());
      require(dgpp::serve::decode_journal_line(legacy).world_settings.fp8_head == "gemv",
              "codec: absent head mode defaults to GEMV");
      for (const std::string value : {"\"auto\"", "true", "null", "4"}) {
        std::string bad = legacy;
        bad.insert(bad.find('{') + 1, "\"fp8_head\":" + value + ",");
        std::string error;
        try {
          (void)dgpp::serve::decode_journal_line(bad);
        } catch (const std::runtime_error& e) {
          error = e.what();
        }
        require(error.find("engine.fp8_head") != std::string::npos,
                "codec: invalid head mode names engine.fp8_head: " + error);
      }
    }
    require(sr.world_settings.rope_scaling.has_value() &&
                sr.world_settings.rope_scaling->context_limit() == 524288,
            "codec: the rope scaling rides the settings record");
    {
      // A plain run carries no key at all: a peer decoding it gets the
      // plain table, which is what it would have run anyway.
      dgpp::serve::WorldSettings plain = ws;
      plain.rope_scaling.reset();
      const std::string line = dgpp::serve::encode_journal_settings(plain);
      require(line.find("\"rs\"") == std::string::npos, "codec: no rope key when off");
      const dgpp::serve::JournalRecord got = dgpp::serve::decode_journal_line(line);
      require(!got.world_settings.rope_scaling.has_value(), "codec: absent decodes to off");
    }
    {
      // An impossible ramp (a sub-unity factor) is refused, not applied.
      dgpp::serve::WorldSettings bad = ws;
      bad.rope_scaling->factor = 0.5;
      bool refused_rope = false;
      try {
        (void)dgpp::serve::decode_journal_line(dgpp::serve::encode_journal_settings(bad));
      } catch (const std::runtime_error&) {
        refused_rope = true;
      }
      require(refused_rope, "codec: a settings record with an impossible rope scaling is refused");
    }
    {
      auto encoded = dgpp::serve::encode_journal_settings(ws);
      const std::string compact_field = ",\"compact\":1";
      const auto at = encoded.find(compact_field);
      require(at != std::string::npos, "settings carry compact_batches");
      auto legacy = encoded;
      legacy.erase(at, compact_field.size());
      require(!dgpp::serve::decode_journal_line(legacy).world_settings.compact_batches,
              "older settings default compaction off");
      encoded.replace(at, compact_field.size(), ",\"compact\":2");
      bool refused = false;
      try {
        (void)dgpp::serve::decode_journal_line(encoded);
      } catch (const std::runtime_error& error) {
        refused = std::string(error.what()).find("compact") != std::string::npos;
      }
      require(refused, "invalid compaction flag is refused by name");
    }

    {
      // The KV dtype rides by name; an unknown one is refused.
      dgpp::serve::WorldSettings bad = ws;
      bad.kv_dtype = "int4";
      bool refused_dtype = false;
      try {
        (void)dgpp::serve::decode_journal_line(dgpp::serve::encode_journal_settings(bad));
      } catch (const std::runtime_error&) {
        refused_dtype = true;
      }
      require(refused_dtype, "codec: a settings record with an unknown kv dtype is refused");
    }
    {
      // The bf16 weights' form rides by name too.
      dgpp::serve::WorldSettings bad = ws;
      bad.bf16_weights = "int8";
      bool refused_form = false;
      try {
        (void)dgpp::serve::decode_journal_line(dgpp::serve::encode_journal_settings(bad));
      } catch (const std::runtime_error&) {
        refused_form = true;
      }
      require(refused_form, "codec: a settings record with an unknown bf16 weight form is refused");
      // Both forms resident: its own name on the wire.
      dgpp::serve::WorldSettings both = ws;
      both.bf16_weights = "bf12+bf16";
      require(dgpp::serve::decode_journal_line(dgpp::serve::encode_journal_settings(both)).world_settings == both,
              "codec: the bf12+bf16 form round-trips");
    }
    bool refused = false;
    try {
      dgpp::serve::WorldSettings one = ws;
      one.world = 1;
      (void)dgpp::serve::decode_journal_line(dgpp::serve::encode_journal_settings(one));
    } catch (const std::runtime_error&) {
      refused = true;
    }
    require(refused, "codec: a settings record with a world of one is refused");
  }
  {
    // The logit bias and the stops: "lb" pairs round-trip bit
    // for bit; "sp" rides beside the cancels; a request without them is
    // unchanged on the wire.
    GenerationService::PassEvents ev;
    dgpp::sched::SchedulerRequest r;
    r.id = "chatcmpl-0000000000000b1a";
    r.prompt = {1, 2, 3};
    r.max_steps = 4;
    r.logit_bias = {{5, -100.0f}, {77, 2.5f}, {154819, -0.125f}};
    ev.submits.push_back(r);
    ev.stops.push_back("chatcmpl-halt");
    const std::string line = dgpp::serve::encode_journal_tick(ev);
    require(line.find("\"lb\":[[5,") != std::string::npos &&
                line.find("\"sp\":[\"chatcmpl-halt\"]") != std::string::npos,
            "codec: bias and stops on the wire: " + line);
    const dgpp::serve::JournalRecord b3 = dgpp::serve::decode_journal_line(line);
    require(b3.submits.size() == 1 && b3.submits[0].logit_bias.size() == 3 &&
                b3.submits[0].logit_bias[0].token == 5 &&
                b3.submits[0].logit_bias[0].bias == -100.0f &&
                b3.submits[0].logit_bias[1].token == 77 &&
                b3.submits[0].logit_bias[1].bias == 2.5f &&
                b3.submits[0].logit_bias[2].token == 154819 &&
                b3.submits[0].logit_bias[2].bias == -0.125f &&
                b3.stops.size() == 1 && b3.stops[0] == "chatcmpl-halt" &&
                b3.cancels.empty(),
            "codec: bias and stops round-trip");
    const std::string plain2 = dgpp::serve::encode_journal_tick(events);
    require(plain2.find("\"lb\"") == std::string::npos &&
                plain2.find("\"sp\"") == std::string::npos,
            "codec: a request without them is unchanged");
  }
  require(back.cancels[0] == "chatcmpl-dead", "codec: cancel round-trip");
  require(!back.submits[0].grammar.active(), "codec: no grammar unless sent");

  // The grammar (M6 6g) rides with the request: mode, parallel flag, the
  // named function, the tools with and without closed key sets.
  {
    GenerationService::PassEvents ev;
    dgpp::sched::SchedulerRequest s = submit;
    s.grammar.mode = dgpp::text::GrammarSpec::Mode::kNamed;
    s.grammar.parallel = false;
    s.grammar.named = "get_weather";
    // The typed arguments (M6 6i): a free string, a JSON-typed integer, an
    // enum's texts; only the constrained ones ride.
    dgpp::text::GrammarTool weather{"get_weather", true, {"city", "days", "unit"}, {}, {}, false};
    weather.strict = true;                    // the 6i follow-on: strict and
    weather.required_keys = {"city", "days"};  // its required keys ride too
    weather.args.push_back(dgpp::text::GrammarArg{"city", dgpp::text::GrammarArg::Kind::kFree, "", {}});
    weather.args.push_back(dgpp::text::GrammarArg{"days", dgpp::text::GrammarArg::Kind::kJson,
                                                 "{\"type\": \"integer\"}", {}});
    weather.args.push_back(dgpp::text::GrammarArg{"unit", dgpp::text::GrammarArg::Kind::kText, "",
                                                 {"celsius", "fahrenheit"}});
    s.grammar.tools.push_back(std::move(weather));
    s.grammar.tools.push_back(dgpp::text::GrammarTool{"get_time", false, {}, {}, {}, false});
    s.grammar.tools.push_back(dgpp::text::GrammarTool{"ping", true, {}, {}, {}, false});
    ev.submits.push_back(s);
    const dgpp::serve::JournalRecord got = dgpp::serve::decode_journal_line(
        dgpp::serve::encode_journal_tick(ev));
    // A free argument does not ride; equality is over the constrained ones.
    s.grammar.tools[0].args.erase(s.grammar.tools[0].args.begin());
    require(got.submits.size() == 1 && got.submits[0].grammar == s.grammar,
            "codec: grammar round-trip (typed arguments included)");
    require(got.submits[0].grammar.tools[0].args.size() == 2 &&
                got.submits[0].grammar.tools[0].args[0].kind == dgpp::text::GrammarArg::Kind::kJson &&
                got.submits[0].grammar.tools[0].args[1].texts.size() == 2,
            "codec: the typed arguments survive");
    require(got.submits[0].grammar.tools[2].constrain_keys &&
                got.submits[0].grammar.tools[2].keys.empty(),
            "codec: a closed empty key set survives");
    require(got.submits[0].grammar.tools[0].strict &&
                got.submits[0].grammar.tools[0].required_keys ==
                    std::vector<std::string>{"city", "days"} &&
                !got.submits[0].grammar.tools[1].strict &&
                got.submits[0].grammar.tools[1].required_keys.empty(),
            "codec: strict and the required keys survive; absent ones stay absent");
  }
  // The JSON grammar (M6 6h) carries its schema text; json_object is the
  // empty text and survives as such.
  for (const char* schema :
       {"", "{\"type\": \"object\", \"properties\": {\"city\": {\"type\": "
            "\"string\"}, \"n\": {\"enum\": [1, \"a\\\"b\", null]}}, "
            "\"required\": [\"city\"], \"additionalProperties\": false}",
        R"({"type":"object","properties":{"id":{"$ref":"#/$defs/id"},"score":{"type":"number","multipleOf":0.1}},"$defs":{"id":{"type":"string","pattern":"^[A-Z]+$"}}})",
        R"({"type":"string","x-dgpp-grammar":{"syntax":"lark","definition":"start: /[A-Z]+/ \"=\" /[0-9]+/"}})"}) {
    for (const auto mode : {dgpp::text::GrammarSpec::Mode::kJson,
                            dgpp::text::GrammarSpec::Mode::kJsonOrTools}) {
      GenerationService::PassEvents ev;
      dgpp::sched::SchedulerRequest s = submit;
      s.grammar.mode = mode;
      s.grammar.json_schema = schema;
      if (mode == dgpp::text::GrammarSpec::Mode::kJsonOrTools)
        s.grammar.tools.push_back(dgpp::text::GrammarTool{"ping", true, {}, {}, {}, false});
      ev.submits.push_back(s);
      const dgpp::serve::JournalRecord got = dgpp::serve::decode_journal_line(
          dgpp::serve::encode_journal_tick(ev));
      require(got.submits.size() == 1 && got.submits[0].grammar == s.grammar &&
                  got.submits[0].grammar.json_schema == schema,
              std::string("codec: JSON grammar round-trip for '") + schema + "'");
    }
  }

  require(dgpp::serve::decode_journal_line(
              dgpp::serve::encode_journal_stop()).stop,
          "codec: stop round-trip");
  {
    const dgpp::serve::JournalRecord warm =
        dgpp::serve::decode_journal_line(
            dgpp::serve::encode_journal_warm());
    require(warm.warm && !warm.stop && warm.submits.empty() &&
                warm.cancels.empty(),
            "codec: warm round-trip");
    require(!back.warm, "codec: tick decoded as warm");
  }

  bool threw = false;
  try {
    (void)dgpp::serve::decode_journal_line("{\"op\":\"nonsense\"}");
  } catch (const std::exception&) {
    threw = true;
  }
  require(threw, "codec: unknown op must throw");

  // The sampling spec (M6 6b): a greedy submit's record carries no spec
  // (byte-identical to the pre-sampling format); a stochastic submit's
  // spec and seed round-trip bitwise; a corrupt spec is refused.
  {
    const std::string greedy_line =
        dgpp::serve::encode_journal_tick(events);
    require(greedy_line.find("\"g\"") == std::string::npos,
            "codec: greedy submit must not carry a sampling spec");
    GenerationService::PassEvents stochastic;
    dgpp::sched::SchedulerRequest s = submit;
    s.sampling.temperature = 0.7f;
    s.sampling.top_p = 0.95f;
    s.sampling.top_k = 40;
    s.sampling.min_p = 0.0125f;
    s.sampling.repetition_penalty = 1.1f;
    s.sampling.frequency_penalty = -0.3f;
    s.sampling.presence_penalty = 1.0e-7f;
    s.sampling.logprobs = 3;
    s.seed = 0xfedcba9876543210ull;
    stochastic.submits.push_back(s);
    const dgpp::serve::JournalRecord got = dgpp::serve::decode_journal_line(
        dgpp::serve::encode_journal_tick(stochastic));
    require(got.submits.size() == 1, "codec: stochastic submit count");
    const dgpp::sched::SchedulerRequest& b = got.submits[0];
    const auto bits_equal = [](float x, float y) {
      return std::memcmp(&x, &y, sizeof(float)) == 0;
    };
    require(bits_equal(b.sampling.temperature, 0.7f) &&
                bits_equal(b.sampling.top_p, 0.95f) &&
                b.sampling.top_k == 40 &&
                bits_equal(b.sampling.min_p, 0.0125f) &&
                bits_equal(b.sampling.repetition_penalty, 1.1f) &&
                bits_equal(b.sampling.frequency_penalty, -0.3f) &&
                bits_equal(b.sampling.presence_penalty, 1.0e-7f) &&
                b.sampling.logprobs == 3 && b.seed == 0xfedcba9876543210ull,
            "codec: sampling spec must round-trip bitwise");
    bool bad = false;
    try {
      // top_p bits of 2.0f: a spec no rank may apply.
      (void)dgpp::serve::decode_journal_line(
          "{\"op\":\"tick\",\"s\":[{\"id\":\"x\",\"p\":[1],\"m\":2,"
          "\"g\":{\"t\":1065353216,\"p\":1073741824,\"k\":0,\"m\":0,"
          "\"r\":1065353216,\"f\":0,\"q\":0,\"l\":0,"
          "\"s\":\"0000000000000001\"}}]}");
    } catch (const std::exception&) {
      bad = true;
    }
    require(bad, "codec: an invalid sampling spec must throw");
    // logprobs ride as "lp" with the spec (a greedy request that asks
    // carries the spec too, at temperature 0).
    GenerationService::PassEvents asking;
    dgpp::sched::SchedulerRequest g = submit;
    g.logprobs = 3;
    g.sampling.logprobs = 3;
    asking.submits.push_back(g);
    const std::string line = dgpp::serve::encode_journal_tick(asking);
    require(line.find("\"lp\":3") != std::string::npos &&
                line.find("\"g\":{") != std::string::npos,
            "codec: logprobs field and spec on a greedy request");
    const dgpp::serve::JournalRecord back2 =
        dgpp::serve::decode_journal_line(line);
    require(back2.submits.size() == 1 && back2.submits[0].logprobs == 3 &&
                back2.submits[0].sampling.logprobs == 3 &&
                back2.submits[0].sampling.temperature == 0.0f,
            "codec: logprobs round-trip");
  }
  std::puts("ok 1 - journal codec round-trip");
}

// --- the fabric rig: rank 0's real stack + two real peer loops ----------

struct PeerRig {
  static constexpr int kSlots = 4;
  static constexpr int kQueue = 8;  // rank 0's ServiceConfig limit —
                                    // the identity is constructive
  FakeEngine engine{kSlots, 100, 4};
  // Constructed after the warm record: rank 0's admission policy rides it
  // (M6 6d), exactly as dgpp-serve's peer does.
  std::unique_ptr<dgpp::sched::Scheduler> sched;
  dgpp::serve::OpStreamObserver oplog;
  std::unique_ptr<dgpp::serve::JournalReader> reader;
  std::thread thread;
  std::string error;  // empty = the peer never complained
  std::atomic<bool> warmed{false};  // held at and released by the warm record
  // The failure gates: `block` holds the peer's engine inside an op (the
  // in-tick case); `death_seen` is its in-tick watch's report; `finished`
  // the thread's end; kill() is a process death from rank 0's side (the
  // connection shut down under the loop, which sees EOF and leaves).
  std::atomic<bool> block{false};
  std::atomic<bool> death_seen{false};
  std::atomic<bool> finished{false};
  void kill() {
    if (reader) reader->shutdown();
  }

  PeerRig(int rank, uint16_t journal_port, const std::atomic<bool>& stop_flag) {
    engine.set_block(&block);
    thread = std::thread([this, rank, journal_port, &stop_flag] {
      struct Done {
        std::atomic<bool>& f;
        ~Done() { f.store(true); }
      } done{finished};
      try {
        // Connect BEFORE the loop: rank 0's accept_peers is waiting
        // for the full world, and it must not wait on a reader that
        // only connects after its first record.
        reader = std::make_unique<dgpp::serve::JournalReader>(
            "127.0.0.1", journal_port, 5000, rank);
        // The production peer holds here for the graph engine's warm
        // capture start signal; the rig holds the same way so the first
        // record's order (warm, then ticks) is pinned end to end.
        dgpp::sched::AdmissionPolicy policy;
        if (!dgpp::serve::wait_journal_warm(
                reader.get(), [&stop_flag] { return stop_flag.load(); },
                &policy))
          return;
        sched = std::make_unique<dgpp::sched::Scheduler>(&engine, std::vector<int64_t>{kFakeEos},
                                                       kQueue, policy);
        sched->set_observer(&oplog);
        warmed.store(true);
        dgpp::serve::run_journal_peer(
            sched.get(), reader.get(), [&stop_flag] { return stop_flag.load(); },
            [this] {
              // dgpp-serve writes its op stream and _Exit(3)s here; the rig
              // records the sighting and lets the blocked op finish so the
              // loop can see the EOF and return.
              death_seen.store(true);
              block.store(false);
            },
            /*watch_poll_ms=*/20, &oplog);
      } catch (const std::exception& e) {
        error = e.what();
        // dgpp-serve's peer dies of this (the exception leaves main); the
        // rig closes the connection the way the process exit would, so
        // rank 0's watch sees it.
        if (reader) reader->close();
      }
    });
  }
  ~PeerRig() {
    if (thread.joinable()) thread.join();
  }
  PeerRig(const PeerRig&) = delete;
  PeerRig& operator=(const PeerRig&) = delete;
};

struct FabricRig {
  static constexpr int kWorld = 3;  // rank 0 + two peers (the star)
  FakeEngine engine{4, 100, 4};
  FakeFrontend frontend;
  ServiceConfig cfg;
  GenerationService service;
  HttpServer http;
  dgpp::serve::OpStreamObserver oplog;  // rank 0's audit leg
  dgpp::serve::JournalWriter journal{0};
  std::optional<dgpp::serve::DiskCommitCollector> disk_commits;  // the cold tier's verdicts (issue #26)
  std::vector<std::unique_ptr<PeerRig>> peers;
  std::thread http_loop;
  std::thread engine_loop;
  std::atomic<bool> stopping{false};
  std::atomic<bool> gate{false};  // holds the engine between passes (tests)
  std::atomic<int> pass_delay_ms{0};  // slows the engine to a human pace (tests)
  std::atomic<bool> failed{false};  // the failure path ran (a peer died or an op threw)
  std::string failure;

  // One engine pass with the journal hook — the loop's body, also callable
  // from a test while the loop is gated (it sleeps then; nothing else
  // touches the service's engine side).
  bool pass() {
    return service.engine_pass(
        [this](const GenerationService::PassEvents& events) {
          journal.broadcast(dgpp::serve::encode_journal_tick(events));
        });
  }

  FabricRig()
      : cfg([] {
          ServiceConfig c;
          c.model_id = "glm-5.3-flash-fp8";
          c.default_max_tokens = 8;
          c.queue_limit = 8;
          // Grow-on-demand (M6 6d) for every scenario: a one-block window (the
          // fake's blocks are 4 tokens) so each request grows several times.
          c.admission.mode = dgpp::sched::AdmissionPolicy::Mode::kGrowOnDemand;
          c.admission.window_tokens = 4;
          return c;
        }()),
        service(cfg, &engine, &frontend, {kFakeEos}),
        http(0, &service, 64) {
    service.set_audit_observer(&oplog);
    // The cold tier's return path (issue #26), as dgpp-serve wires it: the
    // peers' status lines and this rank's completions meet in the
    // collector; the commits ready at a pass ride its record.
    if (engine.disk_info().enabled) {
      disk_commits.emplace(kWorld);
      service.set_disk_hooks(
          [this]() {
            for (const auto& [peer, line] : journal.poll_peer_lines()) {
              uint64_t op = 0;
              bool ok = false;
              if (!dgpp::serve::decode_disk_status(line, &op, &ok))
                throw std::runtime_error("rig: unexpected peer line: " + line);
              disk_commits->add(peer, op, ok);
            }
            return disk_commits->take_ready();
          },
          [this](const std::vector<dgpp::sched::SchedulerEngine::DiskCompletion>& done) {
            for (const auto& c : done) disk_commits->add(0, c.op, c.ok);
          });
    }
    // Peers connect first (they block in the journal read loop),
    // then rank 0 accepts the full world, then HTTP + the engine.
    // Nothing broadcasts before accept_peers returns.
    for (int r = 1; r < kWorld; ++r)
      peers.push_back(std::make_unique<PeerRig>(r, journal.port(), stopping));
    journal.accept_peers(kWorld, 5000);
    // The warm record precedes every tick (dgpp-serve broadcasts it
    // before its warm capture); the peers are holding for it.
    journal.broadcast(dgpp::serve::encode_journal_warm(cfg.admission));
    // The peer death watch, as dgpp-serve installs it: a dead peer fails
    // the service at once (the engine may be inside a collective that
    // rank will never complete).
    journal.watch_peers(
        [this](int peer, const std::string& why) {
          failure = "rank " + std::to_string(peer) + " died (" + why + ")";
          service.fail_engine(failure);
          failed.store(true);
        },
        /*poll_ms=*/20);
    http_loop = std::thread([this] { http.serve(); });
    engine_loop = std::thread([this] {
      while (!stopping.load()) {
        if (gate.load()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          continue;
        }
        if (failed.load()) break;
        bool more = false;
        try {
          more = pass();
        } catch (const std::exception& e) {
          // The app's failure path: the service fails, the loop leaves.
          failure = e.what();
          service.fail_engine(failure);
          failed.store(true);
          break;
        }
        if (!more)
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
        else if (pass_delay_ms.load() > 0)
          std::this_thread::sleep_for(std::chrono::milliseconds(pass_delay_ms.load()));
      }
      if (failed.load()) {
        // dgpp-serve exits here (status 2): its journal connections close
        // and the surviving peers see EOF. The rig closes them the same way.
        journal.stop_watch();
        journal.close_peers();
        return;
      }
      // The peer watch ends before the drain (the stop record releases the
      // peers; their exits are departures, not deaths), as dgpp-serve does.
      journal.stop_watch();
      // Drain-on-stop (M6 6c), as dgpp-serve does it: flag the live
      // requests, one more pass so the cancels ride the journal and every
      // rank retires them at this quantum, then the stop record.
      service.begin_shutdown();
      try {
        pass();
        journal.broadcast(dgpp::serve::encode_journal_stop());
      } catch (const std::exception& e) {
        DGPP_LOG_ERROR("rig: the drain pass or the stop broadcast failed: {}",
                       e.what());
      }
    });
  }

  // The app's stop order: the engine thread drains and broadcasts the
  // stop record, the HTTP pump answers every interrupted client, THEN the
  // server stops and the peers are joined.
  void stop() {
    if (stopping.exchange(true)) return;
    gate.store(false);
    if (engine_loop.joinable()) engine_loop.join();  // drains, broadcasts stop
    for (int i = 0; i < 400 && !service.drained(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    http.stop();
    if (http_loop.joinable()) http_loop.join();
    for (auto& p : peers)
      if (p->thread.joinable()) p->thread.join();
  }
  // Waits for a peer's thread to end (bounded), true if it did.
  bool peer_finished(size_t i, int timeout_ms) {
    for (int t = 0; t < timeout_ms / 5 && !peers[i]->finished.load(); ++t)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return peers[i]->finished.load();
  }
  ~FabricRig() { stop(); }
  FabricRig(const FabricRig&) = delete;
  FabricRig& operator=(const FabricRig&) = delete;

  uint16_t port() const { return http.port(); }

  // The §11 identity: every peer's op stream must equal rank 0's.
  bool oplogs_agree() const {
    const std::string rank0 = oplog.text();
    for (const auto& p : peers)
      if (p->oplog.text() != rank0) return false;
    return true;
  }
  std::string oplog_diff() const {
    std::string out = "rank0:\n" + oplog.text();
    for (size_t i = 0; i < peers.size(); ++i)
      out += "\npeer " + std::to_string(i + 1) + ":\n" +
             peers[i]->oplog.text();
    return out;
  }
  void require_oplogs_agree(const std::string& what) {
    for (int i = 0; i < 2000 && !oplogs_agree(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    require(oplogs_agree(), "op streams diverged after " + what + ":\n" +
                                oplog_diff());
    for (size_t i = 0; i < peers.size(); ++i) {
      require(peers[i]->error.empty(), "peer " + std::to_string(i + 1) +
                                           " errored: " + peers[i]->error);
      require(peers[i]->warmed.load(),
              "peer " + std::to_string(i + 1) + " never saw the warm record");
    }
  }
};

std::string post_request(const std::string& path, const std::string& json) {
  return "POST " + path + " HTTP/1.1\r\nHost: t\r\n"
         "Content-Type: application/json\r\nContent-Length: " +
         std::to_string(json.size()) + "\r\n\r\n" + json;
}

// The concatenated content across every SSE delta in `raw` — deltas
// may coalesce per chunk or not; the client contract is concatenation
// (the fake's tokens are printable bytes, no JSON escaping involved).
std::string concat_content_deltas(const std::string& raw) {
  std::string out;
  size_t at = 0;
  while ((at = raw.find("\"content\":\"", at)) != std::string::npos) {
    at += 11;
    const size_t end = raw.find('"', at);
    if (end == std::string::npos) break;
    out.append(raw, at, end - at);
    at = end;
  }
  return out;
}

// Polls /v1/metrics until `needle` appears (the counters move on the
// engine thread; give them a beat).
std::string wait_metrics(FabricRig& rig, const std::string& needle) {
  for (int i = 0; i < 1000; ++i) {
    Client c(rig.port());
    c.send_all("GET /v1/metrics HTTP/1.1\r\nHost: t\r\n\r\n");
    const std::string raw = c.read_until("}", 2000);
    if (raw.find(needle) != std::string::npos) return raw;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  throw std::runtime_error("metrics never showed " + needle);
}

// --- scenario 1: non-stream completion + the §11 identity ----------------

void test_non_stream_and_identity(FabricRig& rig) {
  // "Hello world, tests!" — 19 characters exactly (13 + 5 + 1; counted
  // twice, trust it): 19 % 4 == 3 → EOS on the second token (finish
  // "stop", completion_tokens 2), and the first token is
  // fake_token(19,0) = 90 = 'Z' — printable, JSON-escape-free, so the
  // raw-body substring assert below is byte-exact.
  Client c(rig.port());
  c.send_all(post_request("/v1/chat/completions",
                          R"({"model":"glm-5.3-flash-fp8","messages":[)"
                          R"({"role":"user","content":"Hello world, tests!"})"
                          R"(],"max_tokens":8})"));
  const std::string raw =
      c.read_until("\"total_tokens\":", 5000);
  require(raw.find("200") != std::string::npos,
          "non-stream: not 200: [" + raw + "]");
  require(raw.find("\"object\":\"chat.completion\"") != std::string::npos,
          "non-stream: wrong object");
  const std::string expect_text(1, static_cast<char>(fake_token(19, 0)));
  require(raw.find("\"content\":\"" + expect_text + "\"") != std::string::npos,
          "non-stream: content mismatch: " + raw);
  require(raw.find("\"finish_reason\":\"stop\"") != std::string::npos,
          "non-stream: EOS should finish stop");
  require(raw.find("\"completion_tokens\":2") != std::string::npos,
          "non-stream: usage mismatch: " + raw);
  rig.require_oplogs_agree("non-stream");
  std::puts("ok 2 - non-stream completion + op-stream identity");
}

// --- scenario 2: streaming lifecycle over the journal --------------------

void test_stream_lifecycle(FabricRig& rig) {
  // A 1-char prompt: the fake's tokens are ((31 + 7i) % 250) + 1 =
  // 32,39,46,53,60,67 — all printable — and len % 4 == 1 → no EOS:
  // the steps cap ends it (finish "length"), 6 tokens with
  // max_tokens 6.
  Client c(rig.port());
  c.send_all(post_request("/v1/chat/completions",
                          R"({"model":"glm-5.3-flash-fp8","messages":[)"
                          R"({"role":"user","content":"x"})"
                          R"(],"max_tokens":6,"stream":true,)"
                          R"("stream_options":{"include_usage":true}})"));
  const std::string raw = c.read_until("data: [DONE]", 5000);
  require(raw.find("200") != std::string::npos, "stream: not 200");
  const size_t role_at = raw.find("\"role\":\"assistant\"");
  const size_t first_content = raw.find("\"content\":\"");
  require(role_at != std::string::npos && first_content != std::string::npos &&
              role_at < first_content,
          "stream: role chunk must lead");
  require(raw.find("\"finish_reason\":\"length\"") != std::string::npos,
          "stream: steps cap should finish length");
  require(raw.find("\"choices\":[],\"usage\":") != std::string::npos,
          "stream: usage chunk missing");
  require(raw.find("data: [DONE]") != std::string::npos, "stream: no DONE");
  std::string expect;
  for (int i = 0; i < 6; ++i)
    expect.push_back(static_cast<char>(fake_token(1, i)));
  require(concat_content_deltas(raw) == expect,
          "stream: concatenated deltas '" + concat_content_deltas(raw) +
              "' != '" + expect + "'");
  rig.require_oplogs_agree("stream");
  std::puts("ok 3 - streaming lifecycle over the journal");
}

// --- scenario 3: disconnect-cancel crosses the journal --------------------

void test_disconnect_cancel(FabricRig& rig) {
  // 10ms per op keeps the request alive ~640ms — the client pulls the
  // plug ~20ms in, comfortably mid-generation. "abcdefgh" (8 bytes,
  // 8 % 4 == 0 → no EOS) with a 64-token budget.
  rig.engine.set_op_delay_ms(10);
  Client c(rig.port());
  c.send_all(post_request("/v1/chat/completions",
                          R"({"model":"glm-5.3-flash-fp8","messages":[)"
                          R"({"role":"user","content":"abcdefgh"})"
                          R"(],"max_tokens":64,"stream":true})"));
  const std::string raw = c.read_until("\"id\":\"", 5000);
  const size_t id_at = raw.find("\"id\":\"chatcmpl-");
  require(id_at != std::string::npos, "disconnect: no id in first chunk");
  const size_t id_begin = id_at + 6;
  const size_t id_end = raw.find('"', id_begin);
  const std::string id = raw.substr(id_begin, id_end - id_begin);
  require(!id.empty(), "disconnect: empty id");
  // Wait for the first CONTENT chunk (the request is live on every
  // rank), then pull the plug.
  const std::string live = c.read_until("\"content\":\"", 5000);
  require(live.find("\"content\":\"") != std::string::npos,
          "disconnect: never went live");
  c.hard_close();

  wait_metrics(rig, "\"requests_cancelled\":1");
  // The retire line for this id must appear on rank 0 AND the peers —
  // with steps well under the 64 cap (cancelled, not capped; EOS is
  // impossible for this prompt).
  for (int i = 0; i < 1000; ++i) {
    const std::string rank0 = rig.oplog.text();
    if (rank0.find("R " + id + " ") != std::string::npos &&
        rig.oplogs_agree())
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  const std::string rank0 = rig.oplog.text();
  const size_t retire_at = rank0.find("R " + id + " ");
  require(retire_at != std::string::npos,
          "disconnect: no retire line for " + id);
  const size_t steps_at = rank0.find(' ', rank0.find(' ', retire_at + 2) + 1);
  const int retired_steps =
      std::atoi(rank0.c_str() + steps_at + 1);
  require(retired_steps < 64, "disconnect: retired at the cap, not the cancel");
  rig.require_oplogs_agree("disconnect-cancel");
  std::puts("ok 4 - disconnect-cancel crossed the journal on every rank");
}

// --- scenario 5: drain-on-stop (M6 6c) -----------------------------------

void test_drain_on_stop(FabricRig& rig) {
  // A stream is mid-generation when the stop lands: the drain pass
  // cancels it THROUGH THE JOURNAL — rank 0 and every peer retire it as
  // cancelled at the same quantum, with no engine op — the client's
  // stream ends with the server_shutdown error event and [DONE] after the
  // tokens it got (no finish chunk), and the stop record releases the
  // peers with nothing in flight.
  rig.pass_delay_ms = 2;  // ~2 ms per token: the pump sees it mid-generation
  Client c(rig.port());
  c.send_all(post_request("/v1/chat/completions",
                          R"({"model":"glm-5.3-flash-fp8","messages":[)"
                          R"({"role":"user","content":"x"})"
                          R"(],"max_tokens":300,"stream":true})"));
  // Wait for a real token, not the role chunk's empty content: only then
  // is the request admitted and generating (a pending admission would be
  // shed, not retired, by the drain).
  const auto has_token = [](const std::string& raw) {
    size_t at = 0;
    while ((at = raw.find("\"content\":\"", at)) != std::string::npos) {
      at += 11;
      if (at < raw.size() && raw[at] != '"') return true;
    }
    return false;
  };
  std::string head;
  for (int i = 0; i < 100 && !has_token(head); ++i)
    head += c.read_until("\"content\":\"", 50);
  require(head.find("200") != std::string::npos && has_token(head),
          "drain: the stream produced a token before the stop: " + head);
  const size_t id_at = head.find("\"id\":\"");
  require(id_at != std::string::npos, "drain: no response id");
  const std::string id =
      head.substr(id_at + 6, head.find('"', id_at + 6) - (id_at + 6));
  rig.gate = true;  // hold the engine between passes: live on every rank
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  require(!rig.service.drained(), "drain: not drained while a stream is live");
  rig.stop();
  const std::string tail = head + c.read_until("data: [DONE]", 5000);
  require(tail.find("\"code\":\"server_shutdown\"") != std::string::npos &&
              tail.find("data: [DONE]") != std::string::npos,
          "drain: the stream ends with the shutdown error event and [DONE]: " +
              tail.substr(tail.size() > 500 ? tail.size() - 500 : 0));
  require(tail.find("\"finish_reason\":\"") == std::string::npos,
          "drain: no finish chunk on an interrupted stream");
  // The retirement crossed the journal: reason 3 (kCancelled) on rank 0
  // and on every peer, identically.
  const std::string rank0 = rig.oplog.text();
  require(rank0.find("R " + id + " 3") != std::string::npos,
          "drain: the interrupted request retired as cancelled on rank 0: " + rank0);
  rig.require_oplogs_agree("drain");
  for (size_t i = 0; i < rig.peers.size(); ++i)
    require(rig.peers[i]->error.empty(),
            "drain: peer " + std::to_string(i + 1) + " errored: " +
                rig.peers[i]->error);
  require(rig.service.drained(), "drain: every answer out");
}

// --- scenario 5: a peer's death fails rank 0 at once ----------------------

void test_peer_death_fails_rank0(FabricRig& rig) {
  // The v1 failure semantics (M8's exit criterion "injected rank failure
  // leaves committed state unchanged"; the death discipline of
  // fabric_serve.hpp): a peer dies while a stream is mid-generation. Rank
  // 0's journal watch sees the closed connection within a poll, fails the
  // service, and the stream ends with the engine_failure error naming the
  // rank AFTER exactly the tokens rank 0 committed (the op stream's T
  // lines for the request — the same lines every rank wrote for them);
  // the door closes (503 engine_failure); rank 0 "exits" (the rig closes
  // its journal connections as the process exit would) and the surviving
  // peer leaves on the EOF without an error.
  rig.pass_delay_ms = 20;  // a human pace: a few tokens before the death
  Client c(rig.port());
  c.send_all(post_request("/v1/chat/completions",
                          R"({"model":"glm-5.3-flash-fp8","messages":[)"
                          R"({"role":"user","content":"x"})"
                          R"(],"max_tokens":300,"stream":true})"));
  const auto has_token = [](const std::string& raw) {
    size_t at = 0;
    while ((at = raw.find("\"content\":\"", at)) != std::string::npos) {
      at += 11;
      if (at < raw.size() && raw[at] != '"') return true;
    }
    return false;
  };
  std::string head;
  for (int i = 0; i < 200 && !has_token(head); ++i)
    head += c.read_until("\"content\":\"", 50);
  require(head.find("200") != std::string::npos && has_token(head),
          "peer death: the stream produced a token first: " + head);
  const size_t id_at = head.find("\"id\":\"");
  require(id_at != std::string::npos, "peer death: no response id");
  const std::string id =
      head.substr(id_at + 6, head.find('"', id_at + 6) - (id_at + 6));
  // Rank 2 dies.
  rig.peers[1]->kill();
  const std::string tail = head + c.read_until("data: [DONE]", 5000);
  require(rig.failed.load(), "peer death: rank 0's failure path ran");
  require(rig.failure.find("rank 2 died") != std::string::npos,
          "peer death: the watch named the rank: " + rig.failure);
  require(tail.find("\"code\":\"engine_failure\"") != std::string::npos &&
              tail.find("rank 2 died") != std::string::npos &&
              tail.find("data: [DONE]") != std::string::npos,
          "peer death: the stream ends with the engine_failure error and [DONE]: " +
              tail.substr(tail.size() > 600 ? tail.size() - 600 : 0));
  require(tail.find("\"finish_reason\":\"") == std::string::npos,
          "peer death: no finish chunk");
  // Exactly the committed tokens: the fake's sequence for the prompt, as
  // many as rank 0's op stream committed for the request.
  const std::string content = concat_content_deltas(tail);
  std::string expect;
  for (size_t i = 0; i < content.size(); ++i)
    expect.push_back(static_cast<char>(fake_token(1, static_cast<int>(i))));
  require(!content.empty() && content == expect,
          "peer death: the tokens are the committed prefix: '" + content + "'");
  size_t committed = 0;
  const std::string ops = rig.oplog.text();
  const std::string needle = "T " + id + " ";
  for (size_t at = 0; (at = ops.find(needle, at)) != std::string::npos; at += needle.size())
    ++committed;
  require(committed == content.size(),
          "peer death: the stream carries every committed token and nothing more (" +
              std::to_string(content.size()) + " vs " + std::to_string(committed) + " committed)");
  // The door.
  Client late(rig.port());
  late.send_all(post_request("/v1/chat/completions",
                             R"({"model":"glm-5.3-flash-fp8","messages":[)"
                             R"({"role":"user","content":"y"})"
                             R"(],"max_tokens":2})"));
  const std::string refused = late.read_until("}}", 3000);
  require(refused.find("503 ") != std::string::npos &&
              refused.find("\"code\":\"engine_failure\"") != std::string::npos,
          "peer death: a later request is refused 503 engine_failure: " + refused.substr(0, 300));
  // The dead peer's thread ended (the loop saw the EOF); the survivor is
  // released by rank 0's exit (the journal closing), without an error.
  require(rig.peer_finished(1, 3000), "peer death: the dead peer's loop ended");
  require(rig.peer_finished(0, 3000), "peer death: the survivor was released");
  require(rig.peers[0]->error.empty(),
          "peer death: the survivor left without an error: " + rig.peers[0]->error);
  for (int i = 0; i < 400 && !rig.service.drained(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  require(rig.service.drained(), "peer death: every answer out");
}

// --- scenario 6: rank 0's death releases a peer stuck inside a tick --------

void test_rank0_death_releases_a_peer_mid_tick(FabricRig& rig) {
  // A peer inside a tick (its engine op holding, as a collective would
  // with rank 0 gone) cannot see the journal EOF itself; the in-tick watch
  // does, within a poll, and the app exits nonzero from it. Here: rank
  // 0's engine is held so exactly one record is out (no pending data on
  // the wire — the lockstep protocol's invariant, which makes a readable
  // journal socket inside a tick a death and nothing else), peer 1's
  // engine blocks inside that tick's prefill, rank 0's connections close
  // (the watch stopped first: a live rank 0 ending its own life is not a
  // peer death), and peer 1's watch fires while peer 2 sees the EOF from
  // its read loop.
  rig.gate = true;
  rig.peers[0]->block = true;
  Client c(rig.port());
  c.send_all(post_request("/v1/chat/completions",
                          R"({"model":"glm-5.3-flash-fp8","messages":[)"
                          R"({"role":"user","content":"x"})"
                          R"(],"max_tokens":8,"stream":true})"));
  for (int i = 0; i < 200 && rig.service.drained(); ++i)  // the admission is pending
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  require(rig.pass(), "rank 0 death: the one pass admitted the request");
  for (int i = 0; i < 400 && !rig.peers[0]->engine.blocked(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  require(rig.peers[0]->engine.blocked(), "rank 0 death: peer 1 is inside the tick");
  require(!rig.peers[0]->death_seen.load(), "rank 0 death: nothing seen while rank 0 lives");
  rig.journal.stop_watch();
  rig.journal.close_peers();  // rank 0's process exit, from the peers' side
  for (int i = 0; i < 600 && !rig.peers[0]->death_seen.load(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  require(rig.peers[0]->death_seen.load(),
          "rank 0 death: the in-tick watch saw the journal close");
  require(rig.peer_finished(0, 3000) && rig.peer_finished(1, 3000),
          "rank 0 death: both peers left");
  require(rig.peers[0]->error.empty() && rig.peers[1]->error.empty(),
          "rank 0 death: no peer errored: " + rig.peers[0]->error + " / " +
              rig.peers[1]->error);
  // Rank 0 is "dead" here: its client sees the connection go, not an
  // event (the app never reaches a pump after its own death); the rig's
  // stop must still come back (its drain pass throws on the closed peers
  // and is caught) — a hang here is the failure.
  rig.stop();
  (void)c.read_until("data: [DONE]", 500);
}

// --- scenario 7: the continuous drift check ------------------------------

void test_op_stream_divergence_kills_the_peer(FabricRig& rig) {
  // M9's counter-drift check: every tick record carries rank 0's op-stream
  // fold after the previous tick, and a peer whose fold differs exits with an error
  // with the tick number — one tick late at most — instead of serving on
  // and being caught by the shutdown procedure's md5. Peer 1's engine is made
  // to produce different tokens; its loop must throw at the record after
  // the first diverging tick, naming the tick; its death then fails rank 0
  // (the watch) and releases the other peer.
  rig.peers[0]->engine.set_token_offset(1);
  Client c(rig.port());
  c.send_all(post_request("/v1/chat/completions",
                          R"({"model":"glm-5.3-flash-fp8","messages":[)"
                          R"({"role":"user","content":"x"})"
                          R"(],"max_tokens":8})"));
  for (int i = 0; i < 600 && rig.peers[0]->error.empty(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  const std::string& err = rig.peers[0]->error;
  require(err.find("op-stream divergence at tick") != std::string::npos,
          "drift: the diverging peer died naming the tick: '" + err + "'");
  // Caught at the record after the first diverging tick: the request's
  // first tick is the first or second record (an idle pass may precede
  // it), so the named tick is small — never the request's eighth.
  const size_t at = err.find("at tick ");
  require(at != std::string::npos && std::atoi(err.c_str() + at + 8) <= 3,
          "drift: caught within a tick of the divergence: '" + err + "'");
  for (int i = 0; i < 600 && !rig.failed.load(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  // Rank 0 fails on the dead peer by whichever path sees it first: the
  // watch ("rank 1 died") or the next broadcast to the closed connection
  // ("rank 1's connection stopped reading").
  require(rig.failed.load() && rig.failure.find("rank 1") != std::string::npos,
          "drift: rank 0 failed the service on the dead peer: " + rig.failure);
  require(rig.peer_finished(1, 3000) && rig.peers[1]->error.empty(),
          "drift: the other peer was released without an error: " + rig.peers[1]->error);
  (void)c.read_until("}}", 3000);  // the one-shot: served, or engine_failure
  for (int i = 0; i < 400 && !rig.service.drained(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  require(rig.service.drained(), "drift: every answer out");
}

}  // namespace

void test_op_stream_file_mode() {
  // GIVEN one observer holding its stream in memory and one streaming to a
  // file,
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("dgpp_opstream_" + std::to_string(::getpid()) + ".ops"))
          .string();
  dgpp::serve::OpStreamObserver memory;
  dgpp::serve::OpStreamObserver file;
  require(file.open(path), "the op stream file opens");

  // WHEN both see the same events (the retire flushes),
  dgpp::sched::Scheduler::Result result;
  result.status = dgpp::sched::Scheduler::Result::Status::kDone;
  result.reason = dgpp::sched::Scheduler::Result::Reason::kEos;
  result.steps_done = 2;
  for (dgpp::serve::OpStreamObserver* o : {&memory, &file}) {
    o->on_token("chatcmpl-1", 5, 1);
    o->on_prefix("chatcmpl-1", "rolling", 8, 0);
    o->on_token("chatcmpl-1", 999, 2);
    o->on_retire("chatcmpl-1", result);
  }

  // THEN the file holds the bytes the memory observer holds, the file
  // observer holds nothing in memory, and the digests agree.
  std::ifstream in(path, std::ios::binary);
  const std::string on_disk((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
  require(on_disk == memory.text() && !on_disk.empty(),
          "the file is the stream:\n" + on_disk);
  require(file.text().empty(), "the file mode holds nothing in memory");
  require(file.digest() == memory.digest(), "the digests agree");
  std::filesystem::remove(path);
}

// --- scenario: the NVMe cold tier across the world (issue #26) -----------
// A one-slot arena over a fake slab on every rank: the first prompt's
// entry (at cut 16) spills, the second prompt's admission evicts it from
// memory, the first prompt again restores it from every rank's slab
// through the journal's return path — the same decisions, the same op
// stream, on all three ranks.
void test_nvme_cache_identity(FabricRig& rig) {
  const auto ask = [&](const std::string& text) {
    Client c(rig.port());
    c.send_all(post_request("/v1/chat/completions",
                            R"({"model":"glm-5.3-flash-fp8","max_tokens":1,"messages":[)"
                            R"({"role":"user","content":")" + text + R"("}]})"));
    const std::string raw = c.read_until("\r\n0\r\n\r\n", 5000);
    if (raw.find("200 OK") == std::string::npos) {
      std::string why = "nvme: the request was refused: " + raw + "\nrank 0: " + rig.failure;
      for (size_t i = 0; i < rig.peers.size(); ++i)
        why += "\npeer " + std::to_string(i + 1) + ": " + rig.peers[i]->error;
      throw std::runtime_error(why);
    }
  };
  ask("Hello world, tests!!!");  // 21 bytes: cuts at 8 and 16, the entry at 16
  wait_metrics(rig, "\"spilled\":1");
  ask("Another prompt, no.2!");  // 21 bytes too: takes the only slot, evicts the first
  wait_metrics(rig, "\"spilled\":2");
  ask("Hello world, tests!!!");  // the first again: restored, then attached at 16
  const std::string metrics = wait_metrics(rig, "\"restored\":1");
  require(metrics.find("\"restore_failed\":0") != std::string::npos, "nvme: no failed restore: " + metrics);
  require(metrics.find("\"tokens_saved\":16") != std::string::npos, "nvme: the third request attached at 16: " + metrics);
  // Every rank's op stream carries the same decisions, in the same order.
  for (int i = 0; i < 200 && !rig.oplogs_agree(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  const std::string ops = rig.oplog.text();
  for (const char* needle : {"X spill ", "X spilled ", "X evict ", "X restore ", "X restored ", "X attach "})
    require(ops.find(needle) != std::string::npos, std::string("nvme: rank 0's op stream lacks '") + needle + "':\n" + ops);
  require(ops.find("X restore ") < ops.find("X restored ") && ops.find("X restored ") < ops.find("X attach "),
          "nvme: restore, its commit, then the attach:\n" + ops);
  rig.require_oplogs_agree("nvme");
}

DGPP_TEST(journal_disk_commits_ride_the_tick_record) {
  GenerationService::PassEvents events;
  events.disk_commits.push_back({7, true});
  events.disk_commits.push_back({9, false});
  const std::string line = dgpp::serve::encode_journal_tick(events);
  require(line.find("\"dk\":[[\"7\",1],[\"9\",0]]") != std::string::npos, "the commits encode as [op, ok]: " + line);
  const dgpp::serve::JournalRecord back = dgpp::serve::decode_journal_line(line);
  require(back.disk_commits.size() == 2 && back.disk_commits[0].op == 7 && back.disk_commits[0].ok &&
              back.disk_commits[1].op == 9 && !back.disk_commits[1].ok,
          "the commits decode");
  // A record without commits is byte for byte the earlier format.
  require(dgpp::serve::encode_journal_tick(GenerationService::PassEvents{}).find("dk") == std::string::npos,
          "no commits, no field");
  for (const char* bad : {"{\"op\":\"tick\",\"dk\":[[\"0\",1]]}", "{\"op\":\"tick\",\"dk\":[[7,1]]}",
                          "{\"op\":\"tick\",\"dk\":[[\"7\",2]]}", "{\"op\":\"tick\",\"dk\":[\"7\"]}"}) {
    bool threw = false;
    try {
      (void)dgpp::serve::decode_journal_line(bad);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    require(threw, std::string("a malformed commit is refused: ") + bad);
  }
  // The status line and its parse.
  uint64_t op = 0;
  bool ok = false;
  require(dgpp::serve::decode_disk_status(dgpp::serve::encode_disk_status(42, true), &op, &ok) && op == 42 && ok,
          "status ok round trip");
  require(dgpp::serve::decode_disk_status(dgpp::serve::encode_disk_status(43, false), &op, &ok) && op == 43 && !ok,
          "status fail round trip");
  require(!dgpp::serve::decode_disk_status("hello 1", &op, &ok) && !dgpp::serve::decode_disk_status("disk 0 1", &op, &ok) &&
              !dgpp::serve::decode_disk_status("disk 5 7", &op, &ok),
          "other lines are not status lines");
  // The settings record carries the tier when on, nothing when off.
  dgpp::serve::WorldSettings off;
  off.world = 2;
  off.max_concurrency = 4;
  off.kv_capacity = 8192;
  off.admission = "full";
  dgpp::serve::WorldSettings on = off;
  on.nvme_cache = true;
  on.nvme_cache_capacity_gib = 64.5;
  on.nvme_cache_min_tokens = 2048;
  require(dgpp::serve::decode_journal_line(dgpp::serve::encode_journal_settings(on)).world_settings == on,
          "the tier's settings round trip");
  require(dgpp::serve::encode_journal_settings(off).find("nvme") == std::string::npos, "off: absent");
  require(dgpp::serve::decode_journal_line(dgpp::serve::encode_journal_settings(off)).world_settings == off,
          "off decodes off");
}

DGPP_TEST(journal_disk_commit_collector_waits_for_every_rank) {
  dgpp::serve::DiskCommitCollector c(3);
  c.add(1, 5, true);
  c.add(0, 5, true);
  require(c.take_ready().empty() && c.pending() == 1, "two of three reports: not ready");
  c.add(2, 5, false);
  c.add(0, 6, true);
  auto ready = c.take_ready();
  require(ready.size() == 1 && ready[0].op == 5 && !ready[0].ok, "ready once every rank reported; any failure fails");
  c.add(1, 6, true);
  c.add(2, 6, true);
  ready = c.take_ready();
  require(ready.size() == 1 && ready[0].op == 6 && ready[0].ok && c.pending() == 0, "all ok: ok");
  bool threw = false;
  try {
    c.add(1, 7, true);
    c.add(1, 7, true);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  require(threw, "a rank reporting twice is a divergence");
}

DGPP_TEST(journal_return_path_carries_peer_status_lines) {
  // A writer and one peer over loopback: the peer's status line reaches
  // the writer's poll without blocking; the death watch is undisturbed.
  dgpp::serve::JournalWriter writer(0);
  std::atomic<bool> stop{false};
  std::unique_ptr<dgpp::serve::JournalReader> reader;
  std::thread peer([&] { reader = std::make_unique<dgpp::serve::JournalReader>("127.0.0.1", writer.port(), 5000, 1); });
  writer.accept_peers(2, 5000);
  peer.join();
  require(writer.poll_peer_lines().empty(), "nothing yet");
  require(reader->send_line(dgpp::serve::encode_disk_status(11, true)), "the peer writes");
  require(reader->send_line(dgpp::serve::encode_disk_status(12, false)), "and again");
  std::vector<std::pair<int, std::string>> lines;
  for (int i = 0; i < 500 && lines.size() < 2; ++i) {
    for (auto& l : writer.poll_peer_lines()) lines.push_back(std::move(l));
    if (lines.size() < 2) std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  require(lines.size() == 2 && lines[0].first == 1 && lines[0].second == "disk 11 1" && lines[1].second == "disk 12 0",
          "both lines, in order, from rank 1");
  require(writer.dead_peer() == 0, "the peer is alive");
  (void)stop;
  reader->close();
  for (int i = 0; i < 500 && writer.dead_peer() == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(2));
  require(writer.dead_peer() == 1, "its close is seen");
}

int main() {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  if (const int result = dgpp::test::run_all(); result != 0) return result;
  try {
    test_journal_codec();
    test_settings_handshake();
    test_op_stream_file_mode();
    {
      FabricRig rig;
      test_non_stream_and_identity(rig);
      test_stream_lifecycle(rig);
      // Grow-on-demand rode the warm record to the peers and every growth
      // crossed the journal identically (the "W" lines are in every op
      // stream the identity checks above compared).
      require(rig.oplog.text().find("W chatcmpl-") != std::string::npos,
              "grow: no growth event in rank 0's op stream");
      test_disconnect_cancel(rig);
      test_drain_on_stop(rig);
      std::puts("ok 5 - drain-on-stop: the in-flight stream retired through "
                "the journal on every rank, answered with server_shutdown");
      // The stop discipline: the stop record releases the peers (a
      // hang here joins forever and the gate times out), and nobody
      // errored on the way down.
      rig.stop();
      for (size_t i = 0; i < rig.peers.size(); ++i)
        require(rig.peers[i]->error.empty(),
                "peer " + std::to_string(i + 1) + " errored: " +
                    rig.peers[i]->error);
      rig.require_oplogs_agree("stop");
    }
    std::puts("ok 6 - stop discipline: peers released, no peer errors");
    {
      FabricRig rig;
      test_peer_death_fails_rank0(rig);
    }
    std::puts("ok 7 - a peer's death fails rank 0 at once: the stream got its "
              "committed tokens then engine_failure, the door closed, the "
              "survivor was released");
    {
      FabricRig rig;
      test_rank0_death_releases_a_peer_mid_tick(rig);
    }
    std::puts("ok 8 - rank 0's death releases a peer stuck inside a tick "
              "through the in-tick watch");
    {
      FabricRig rig;
      test_op_stream_divergence_kills_the_peer(rig);
    }
    std::puts("ok 9 - the continuous drift check: a diverging peer dies naming "
              "the tick, rank 0 fails the service, the other peer is released");
    {
      g_fake_arena_slots = 1;
      g_fake_disk_pages = 64;
      FabricRig rig;
      test_nvme_cache_identity(rig);
      rig.stop();
      for (size_t i = 0; i < rig.peers.size(); ++i)
        require(rig.peers[i]->error.empty(),
                "peer " + std::to_string(i + 1) + " errored: " + rig.peers[i]->error);
      rig.require_oplogs_agree("nvme stop");
      g_fake_arena_slots = 0;
      g_fake_disk_pages = 0;
    }
    std::puts("ok 10 - the NVMe cold tier: spill, evict, restore and attach through the "
              "journal's return path, identical on every rank");
    std::puts("fabric_serve_test: ALL PASS");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}

DGPP_TEST(journal_images_roundtrip_and_validation) {
  dgpp::serve::GenerationService::PassEvents events;
  dgpp::sched::SchedulerRequest r;
  r.id = "vision";
  r.prompt = {42, 154854, 43};
  r.images.push_back({1, 1, 28, 28, std::vector<uint8_t>(28 * 28 * 3)});
  for (size_t i = 0; i < r.images[0].rgb.size(); ++i) r.images[0].rgb[i] = static_cast<uint8_t>(i);
  events.submits.push_back(r);
  const auto line = dgpp::serve::encode_journal_tick(events);
  const auto back = dgpp::serve::decode_journal_line(line);
  require(back.submits[0].images.size() == 1, "image count on peer");
  const auto& im = back.submits[0].images[0];
  require(im.offset == 1 && im.tokens == 1 && im.width == 28 && im.height == 28 &&
              im.rgb == r.images[0].rgb,
          "image bytes and span roundtrip");
  auto bad = line;
  const auto at = bad.find("[1,1,28,28,");
  require(at != std::string::npos, "wire image found");
  bad.replace(at, 11, "[1,1,29,28,");
  bool threw = false;
  try {
    (void)dgpp::serve::decode_journal_line(bad);
  } catch (const std::exception&) {
    threw = true;
  }
  require(threw, "mismatched pixel dimensions rejected");
}

DGPP_TEST(journal_image_history_exceeds_old_count_and_token_caps) {
  dgpp::serve::GenerationService::PassEvents events;
  dgpp::sched::SchedulerRequest request;
  request.id = "image-history";
  request.prompt.assign(12 * 1024, 154854);
  for (int i = 0; i < 12; ++i)
    request.images.push_back({i * 1024, 1024, 896, 896, std::vector<uint8_t>(896 * 896 * 3, i)});
  events.submits.push_back(std::move(request));
  const auto decoded = dgpp::serve::decode_journal_line(dgpp::serve::encode_journal_tick(events));
  require(decoded.submits[0].images.size() == 12, "whole image history reaches peers");
  for (int i = 0; i < 12; ++i) {
    const auto& image = decoded.submits[0].images[i];
    require(image.offset == i * 1024 && image.tokens == 1024 && image.rgb.back() == i,
            "ordered image pixels and context positions roundtrip");
  }
}

DGPP_TEST(journal_large_fragmented_record_scans_linearly_and_preserves_following_lines) {
  auto listener = dgpp::net::TcpListener::bind(0);
  const std::string payload(64ull << 20, 'x');
  std::string sender_error;
  std::jthread sender([&] {
    try {
      auto peer = listener.accept(5000);
      peer.set_io_deadline_ms(15000);
      char hello[8];
      require(peer.read_exact(hello, sizeof(hello)) && std::string(hello, sizeof(hello)) == "hello 1\n",
              "journal reader handshake");
      require(peer.write_all(payload.data(), payload.size()), "large record write");
      const std::string tail = "\n\nshort\nunterminated";
      require(peer.write_all(tail.data(), tail.size()), "following records write");
    } catch (const std::exception& e) { sender_error = e.what(); }
  });
  bool complete = false, empty = false, following = false, eof = false;
  {
    dgpp::serve::JournalReader reader("127.0.0.1", listener.port(), 5000, 1);
    std::string line;
    // A generous deadline catches the old O(bytes^2 / read_size) scan.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    const auto stopped = [&] { return std::chrono::steady_clock::now() > deadline; };
    complete = reader.read_line(stopped, &line) && line == payload;
    if (complete) {
      empty = reader.read_line(stopped, &line) && line.empty();
      following = reader.read_line(stopped, &line) && line == "short";
      eof = !reader.read_line(stopped, &line);
    }
  }
  sender.join();
  require(complete, "large fragmented journal record exceeded deadline or changed bytes");
  require(sender_error.empty(), sender_error);
  require(empty && following && eof, "journal reader lost buffered lines or accepted a partial EOF record");
}
